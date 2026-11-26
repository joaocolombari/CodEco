"""Single source of the codec matrix used by the reconstruction and embedding stages."""

from __future__ import annotations

import json
from functools import lru_cache

from codeco.codeco_path import CodecoPath


CODEC_CONDITIONS_FILE_NAME = "codec_conditions.json"
REQUIRED_CODECS = frozenset({"flac", "mp3", "aac", "opus", "lc3", "encodec"})


@lru_cache(maxsize=1)
def codec_config() -> dict:
    path = CodecoPath.discover().config / CODEC_CONDITIONS_FILE_NAME
    codecs = json.loads(path.read_text(encoding="utf-8")).get("codecs", {})
    missing = REQUIRED_CODECS - set(codecs)
    if missing:
        raise ValueError(
            f"Matriz de codecs incompleta em {path}: {', '.join(sorted(missing))}"
        )
    return codecs


CODEC_EVALUATION_FILE_NAME = "codec_evaluation.json"


@lru_cache(maxsize=1)
def codec_evaluation_config() -> dict:
    """Windows evaluated by the reconstruction-metric stages (FFmpeg and EnCodec)."""
    path = CodecoPath.discover().config / CODEC_EVALUATION_FILE_NAME
    values = json.loads(path.read_text(encoding="utf-8")).get("codec_evaluation", {})
    missing = {"split", "max_clips_per_dataset", "seed"} - set(values)
    if missing:
        raise ValueError(
            f"Avaliacao de codecs incompleta em {path}: {', '.join(sorted(missing))}"
        )
    return values
