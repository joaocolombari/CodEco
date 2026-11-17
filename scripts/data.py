"""Create execution tables from the authoritative frozen manifests.

This module never recomputes corpus membership, labels, groups, or splits.
Class selection is performed transparently in notebook 02 and stored under
``data/derived/selections``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

import config as C

if str(C.SRC) not in sys.path:
    sys.path.insert(0, str(C.SRC))

from codeco.corpora import CORPUS_MANIFESTS, load_frozen_manifest


def _audio_location(corpus: str, row: pd.Series) -> tuple[Path, float]:
    """Return the expected source path and offset within that physical file."""
    if corpus == "ecosoundset":
        # ECOSoundSet distributes one physical WAV per frozen four-second window.
        relative = str(row.get("source_audio_relative_path", ""))
        return C.ROOT / "data" / relative, 0.0

    filename = str(row["source_file_name"])
    candidates = [
        C.ROOT / "data/audio/pteroset_ppa1" / filename,
        C.ROOT / "data/audio/pteroset/PPA1" / filename,
        C.ROOT / "data/audio/pteroset" / filename,
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    return path, float(row["window_start_s"])


def build_clip_table(corpus: str, selection_manifest: Path | None = None) -> pd.DataFrame:
    if selection_manifest is not None and selection_manifest.is_file():
        df = pd.read_csv(selection_manifest, keep_default_na=False)
    else:
        df = load_frozen_manifest(corpus, C.ROOT)

    rows = []
    for _, row in df.iterrows():
        path, start = _audio_location(corpus, row)
        rows.append(
            {
                "dataset": corpus,
                "clip_id": str(row["window_id"]),
                "path": str(path),
                "audio_available": path.is_file(),
                "start": start,
                "end": start + C.CLIP_SECONDS,
                "labels": str(row["labels"]),
                "split": str(row["split"]),
                "source_group_id": str(row["source_group_id"]),
            }
        )
    return pd.DataFrame(rows)


def prepare_all_clip_tables() -> dict[str, Path]:
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = {}
    for corpus in CORPUS_MANIFESTS:
        selected = C.ROOT / "data/derived/selections" / f"{corpus}_selection_manifest.csv"
        table = build_clip_table(corpus, selected if selected.is_file() else None)
        path = C.OUT_DIR / f"clips_{corpus}.csv"
        table.to_csv(path, index=False)
        written[corpus] = path
        print(
            f"{corpus}: {len(table)} windows; "
            f"{int(table['audio_available'].sum())} audio files available; {path}"
        )
    return written


def load_clip(path: str, start: float) -> np.ndarray:
    """Load exactly one canonical 4 s, 48 kHz, mono, PCM16-equivalent window."""
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Audio file not installed: {path_obj}")
    info = sf.info(str(path_obj))
    start_frame = round(start * info.samplerate)
    frames = round(C.CLIP_SECONDS * info.samplerate)
    data, _ = sf.read(
        path_obj,
        start=start_frame,
        frames=frames,
        dtype="float32",
        always_2d=False,
    )
    if data.ndim > 1:
        data = data.mean(axis=1)
    if info.samplerate != C.SR:
        import soxr

        data = soxr.resample(data, info.samplerate, C.SR, quality="HQ")
    if len(data) != C.CLIP_SAMPLES:
        raise ValueError(
            f"Expected a complete {C.CLIP_SECONDS:g} s window ({C.CLIP_SAMPLES} samples); "
            f"received {len(data)} from {path_obj}. Padding is intentionally disabled."
        )
    data = np.round(np.clip(data, -1.0, 1.0) * 32767.0) / 32768.0
    return data.astype(np.float32)


if __name__ == "__main__":
    prepare_all_clip_tables()
