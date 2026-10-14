#!/usr/bin/env python3
"""Clone the metadata repositories at the exact revisions in the freeze spec."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "data" / "freeze-spec-v1.json").read_text(encoding="utf-8"))


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main() -> None:
    for name, item in SPEC["repositories"].items():
        destination = ROOT / item["path"]
        revision = item["revision"]
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--no-checkout", item["url"], str(destination)],
                check=True,
            )
            subprocess.run(["git", "checkout", "--detach", revision], cwd=destination, check=True)
        elif not (destination / ".git").exists():
            raise SystemExit(f"{destination} exists but is not a Git checkout")

        observed = git("rev-parse", "HEAD", cwd=destination)
        if observed != revision:
            raise SystemExit(
                f"{name} is at {observed}, expected {revision}. "
                "Move or remove the checkout and run the installer again."
            )
        dirty = git("status", "--porcelain", cwd=destination)
        if dirty:
            raise SystemExit(f"{name} has local changes; refusing to alter or certify it")
        print(f"verified repository {name}: {revision}")


if __name__ == "__main__":
    main()
