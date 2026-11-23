"""Single source of truth for the executable experimental matrix."""

from __future__ import annotations

import json
from functools import lru_cache

from codeco.codeco_path import CodecoPath


@lru_cache(maxsize=1)
def experiment_config() -> dict:
    path = CodecoPath.discover().root / "configs" / "experiment-v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"version", "sample_rate_hz", "bit_depth", "window_seconds", "codecs"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Experimental config missing: {', '.join(sorted(missing))}")
    return data
