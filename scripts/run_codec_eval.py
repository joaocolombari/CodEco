"""Passa os clips por todos os codecs e salva metricas (espectrograma, SNR, indices, taxa) em results.csv."""
import pandas as pd
from tqdm import tqdm  # pip install tqdm

import config as C
from codecs_lib import align_to_reference, make_codec
from data import load_clip
from metrics import acoustic_indices, mel_similarity, snr_db


def main():
    C.OUT_DIR.mkdir(exist_ok=True)
    out_csv = C.OUT_DIR / "results.csv"
    done_pairs = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done_pairs = set(zip(prev["dataset"], prev["clip_id"]))
        print(f"Retomando avaliacao: {len(done_pairs)} clipes ja processados.")

    for ds in C.DATASETS:
        clips = pd.read_csv(C.OUT_DIR / f"clips_{ds}.csv")
        clips = clips[clips.split == "test"]
        if C.EVAL_MAX_CLIPS_PER_DATASET:
            clips = clips.sample(min(len(clips), C.EVAL_MAX_CLIPS_PER_DATASET), random_state=C.SEED)
        codecs = [(n, p, make_codec(n, p, dataset=ds))
                  for n, ps in C.CODECS.items() for p in ps]

        for _, r in tqdm(clips.iterrows(), total=len(clips), desc=ds):
            if (ds, r["clip_id"]) in done_pairs:
                continue

            x = load_clip(r["path"], r["start"])
            clip_rows = [dict(dataset=ds, clip_id=r["clip_id"], codec="baseline", setting="orig",
                              nbytes=x.size * C.BIT_DEPTH // 8, **acoustic_indices(x))]
            for name, _, codec in codecs:
                enc = codec.encode(x)
                y, lag = align_to_reference(x, codec.decode(enc))
                clip_rows.append(dict(
                    dataset=ds, clip_id=r["clip_id"], codec=name, setting=codec.setting,
                    nbytes=enc.nbytes, bitrate_kbps=enc.nbytes * 8 / C.CLIP_SECONDS / 1000,
                    compression_ratio=(x.size * C.BIT_DEPTH // 8) / enc.nbytes, lag=lag,
                    snr_db=snr_db(x, y), **mel_similarity(x, y), **acoustic_indices(y)))

            pd.DataFrame(clip_rows).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)


if __name__ == "__main__":
    main()
