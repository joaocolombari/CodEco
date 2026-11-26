"""Machine-readable provenance for every integrated experiment run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from codeco.codeco_path import CodecoPath
from codeco.protocol import PROTOCOL


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str | None:
    process = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _packages() -> dict[str, str | None]:
    names = (
        "birdnet", "birdnetlib", "librosa", "numpy", "pandas", "scikit-learn",
        "scikit-maad", "scipy", "soundfile", "tensorflow", "torch", "transformers",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_run_metadata(environment=None, output: Path | None = None) -> Path:
    paths = CodecoPath.discover()
    output = output or paths.results / "run-metadata.json"
    tracked = (
        paths.manifest,
        paths.config / "prepare_baseline_class_selection.json",
        paths.config / "codec_conditions.json",
        paths.config / "signal_protocol.json",
        paths.root / "pyproject.toml",
        paths.root / "uv.lock",
    )
    system = getattr(environment, "system", None)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "git": {
            "commit": _git(paths.root, "rev-parse", "HEAD"),
            "branch": _git(paths.root, "branch", "--show-current"),
            "dirty": bool(_git(paths.root, "status", "--porcelain")),
        },
        "protocol": PROTOCOL.as_dict(),
        "foundation_frontends": {
            "birdnet": {
                "input_sample_rate_hz": PROTOCOL.birdnet_sample_rate_hz,
                "input_seconds": PROTOCOL.birdnet_seconds,
                "nyquist_hz": PROTOCOL.birdnet_sample_rate_hz / 2,
                "effective_passband_status": "requires empirical frontend audit",
            },
            "perch": {
                "input_sample_rate_hz": PROTOCOL.perch_sample_rate_hz,
                "input_seconds": PROTOCOL.perch_seconds,
                "nyquist_hz": PROTOCOL.perch_sample_rate_hz / 2,
                "adaptation": "4 s resampled then symmetric 0.5 s zero padding",
            },
        },
        "artifacts_sha256": {
            paths.project_relative(path): _sha256(path) for path in tracked
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "device": getattr(system, "device", "unknown"),
            "cuda_available": getattr(system, "cuda_available", False),
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "total_memory_bytes": gpu.total_memory_bytes,
                }
                for gpu in getattr(system, "gpu_devices", ())
            ],
        },
        "packages": _packages(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output
