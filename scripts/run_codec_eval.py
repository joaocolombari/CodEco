"""Paired codec evaluation with resumable, memory-bounded execution."""
from __future__ import annotations

import gc

import pandas as pd
from tqdm import tqdm

import config as C
from codecs_lib import align_to_reference, make_codec
from data import load_clip
from metrics import acoustic_indices, mel_similarity, snr_db


KEY_COLUMNS = ["dataset", "clip_id", "codec", "setting"]


def _existing_keys(path) -> set[tuple[str, str, str, str]]:
    if not path.is_file():
        return set()
    previous = pd.read_csv(path, dtype={"clip_id": str})
    if not set(KEY_COLUMNS).issubset(previous.columns):
        raise ValueError(f"Existing result file has an incompatible schema: {path}")
    return set(map(tuple, previous[KEY_COLUMNS].astype(str).itertuples(index=False, name=None)))


def _append(path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _test_clips(dataset: str) -> pd.DataFrame:
    clips = pd.read_csv(C.OUT_DIR / f"clips_{dataset}.csv", dtype={"clip_id": str})
    clips = clips[(clips.split == "test") & clips.audio_available.astype(bool)]
    if C.EVAL_MAX_CLIPS_PER_DATASET:
        clips = clips.sample(min(len(clips), C.EVAL_MAX_CLIPS_PER_DATASET), random_state=C.SEED)
    return clips.reset_index(drop=True)


def main():
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = C.OUT_DIR / "results.csv"
    done = _existing_keys(out_csv)
    print(f"Retomando avaliação: {len(done)} condições pareadas já registradas.")

    for dataset in C.DATASETS:
        clips = _test_clips(dataset)

        # Baseline is written once per window.
        for row in tqdm(clips.itertuples(index=False), total=len(clips), desc=f"{dataset}: baseline"):
            key = (dataset, str(row.clip_id), "baseline", "orig")
            if key in done:
                continue
            x = load_clip(row.path, row.start)
            result = {
                "dataset": dataset,
                "clip_id": str(row.clip_id),
                "codec": "baseline",
                "setting": "orig",
                "nbytes": x.size * C.BIT_DEPTH // 8,
                "bitrate_kbps": x.size * C.BIT_DEPTH / C.CLIP_SECONDS / 1000,
                "compression_ratio": 1.0,
                "lag": 0,
                "snr_db": float("inf"),
                "mel_mae_db": 0.0,
                "mel_rmse_db": 0.0,
                "mel_ssim": 1.0,
                "mel_cosine": 1.0,
                **acoustic_indices(x),
            }
            _append(out_csv, result)
            done.add(key)

        # Instantiate one codec condition at a time. This is essential for
        # EnCodec and avoids keeping several model copies in memory.
        for codec_name, settings in C.CODECS.items():
            for params in settings:
                codec = make_codec(codec_name, params, dataset=dataset)
                setting = str(codec.setting)
                description = f"{dataset}: {codec_name} {setting}"
                for row in tqdm(clips.itertuples(index=False), total=len(clips), desc=description):
                    key = (dataset, str(row.clip_id), codec_name, setting)
                    if key in done:
                        continue
                    x = load_clip(row.path, row.start)
                    encoded = codec.encode(x)
                    y, lag = align_to_reference(x, codec.decode(encoded))
                    result = {
                        "dataset": dataset,
                        "clip_id": str(row.clip_id),
                        "codec": codec_name,
                        "setting": setting,
                        "nbytes": encoded.nbytes,
                        "bitrate_kbps": encoded.nbytes * 8 / C.CLIP_SECONDS / 1000,
                        "compression_ratio": (x.size * C.BIT_DEPTH // 8) / encoded.nbytes,
                        "lag": lag,
                        "snr_db": snr_db(x, y),
                        **mel_similarity(x, y),
                        **acoustic_indices(y),
                    }
                    _append(out_csv, result)
                    done.add(key)
                del codec
                gc.collect()

    print(f"Resultados pareados: {out_csv}")


if __name__ == "__main__":
    main()
