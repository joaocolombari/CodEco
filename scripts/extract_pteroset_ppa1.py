#!/usr/bin/env python3
"""Extract the verified PteroSet/PPA1 WAV members atomically and reproducibly."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "downloads" / "pteroset" / "PPA1.zip"
AUDIT = ROOT / "data" / "audit" / "pteroset_ppa1_audio_files_v1.csv"
DESTINATION = ROOT / "data" / "audio" / "pteroset_ppa1"
SUMMARY = ROOT / "data" / "audit" / "pteroset_ppa1_extraction_v1.json"
BUFFER_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plan() -> list[dict[str, str]]:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"PPA1 archive not found: {ARCHIVE}")
    if not AUDIT.is_file():
        raise FileNotFoundError(
            f"PPA1 audit not found: {AUDIT}. Run verify_pteroset_ppa1.py first."
        )
    with AUDIT.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 108 or any(row["verified"] != "true" for row in rows):
        raise ValueError("PPA1 audit must contain exactly 108 verified WAV members.")
    names = [row["audio_file"] for row in rows]
    if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        raise ValueError("PPA1 extraction plan contains duplicate or unsafe filenames.")
    return rows


def main() -> None:
    rows = load_plan()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    existing_valid = {
        row["audio_file"]: (
            (target := DESTINATION / row["audio_file"]).is_file()
            and target.stat().st_size == int(row["zip_uncompressed_bytes"])
            and sha256_file(target) == row["sha256"]
        )
        for row in rows
    }
    missing_bytes = sum(
        int(row["zip_uncompressed_bytes"])
        for row in rows
        if not existing_valid[row["audio_file"]]
    )
    free = shutil.disk_usage(DESTINATION).free
    if free < missing_bytes:
        raise RuntimeError(
            f"Insufficient disk space: {missing_bytes / 1024**3:.1f} GiB required; "
            f"{free / 1024**3:.1f} GiB available."
        )

    extracted = skipped = 0
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = {item.filename: item for item in archive.infolist()}
        for index, row in enumerate(rows, 1):
            target = DESTINATION / row["audio_file"]
            expected_size = int(row["zip_uncompressed_bytes"])
            expected_sha256 = row["sha256"]
            if existing_valid[row["audio_file"]]:
                skipped += 1
                print(f"[{index:03d}/108] verified existing: {target.name}")
                continue

            member_name = row["zip_member"]
            if member_name not in members:
                raise KeyError(f"ZIP member missing: {member_name}")
            temporary = target.with_suffix(target.suffix + ".part")
            digest = hashlib.sha256()
            written = 0
            try:
                with archive.open(members[member_name]) as source, temporary.open("wb") as output:
                    while True:
                        block = source.read(BUFFER_SIZE)
                        if not block:
                            break
                        output.write(block)
                        digest.update(block)
                        written += len(block)
                    output.flush()
                    os.fsync(output.fileno())
                if written != expected_size or digest.hexdigest() != expected_sha256:
                    raise ValueError(
                        f"Extracted member mismatch: {member_name}; "
                        f"bytes={written}; sha256={digest.hexdigest()}"
                    )
                os.replace(temporary, target)
                extracted += 1
                print(f"[{index:03d}/108] extracted and verified: {target.name}")
            finally:
                temporary.unlink(missing_ok=True)

    payload = {
        "version": "1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "destination": str(DESTINATION.relative_to(ROOT)),
        "wav_files": len(rows),
        "extracted": extracted,
        "verified_existing": skipped,
        "uncompressed_bytes": sum(int(row["zip_uncompressed_bytes"]) for row in rows),
    }
    SUMMARY.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
