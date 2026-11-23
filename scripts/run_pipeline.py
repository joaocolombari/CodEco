#!/usr/bin/env python3
"""Compatibility entry point for the optimized CodEco pipeline.

The former script orchestrated a second, sequential implementation and could
silently bypass the checkpointed pipeline in ``src/codeco``. Keep this file so
old commands enter the canonical runner instead of doing different science.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from codeco.main import main


if __name__ == "__main__":
    print(
        "AVISO: scripts/run_pipeline.py agora delega ao pipeline otimizado. "
        "Prefira: python -m codeco.main",
        file=sys.stderr,
    )
    main()
