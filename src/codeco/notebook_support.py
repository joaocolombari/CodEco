"""Small helpers shared by the human-readable experiment notebooks."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def dependency_versions(packages: list[str]) -> dict[str, str]:
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def environment_snapshot(root: Path) -> dict:
    git_commit = "unavailable"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit,
        "ffmpeg": shutil.which("ffmpeg") or "not installed",
        "packages": dependency_versions(
            ["numpy", "pandas", "matplotlib", "scikit-learn", "librosa", "soundfile", "soxr"]
        ),
    }


def write_json(data: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
