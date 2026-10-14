#!/usr/bin/env python3
"""Verify that metadata and downloaded audio reproduce corpus freeze 1.3.0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "data" / "freeze-spec-v1.json").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def mapping_digest(rows: list[dict[str, str]], key: str) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value[key]):
        digest.update(f"{row[key]}\t{row['sha256']}\n".encode())
    return digest.hexdigest()


def ids_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(f"{value}\n".encode())
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"freeze verification failed: {message}")


def verify_metadata() -> None:
    lock = json.loads((ROOT / "data" / "manifest-lock-v1.json").read_text(encoding="utf-8"))
    require(lock["manifest_version"] == SPEC["freeze_version"], "manifest version mismatch")
    for name, item in SPEC["repositories"].items():
        checkout = ROOT / item["path"]
        require((checkout / ".git").exists(), f"missing checkout for {name}")
        observed = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        require(observed == item["revision"], f"revision mismatch for {name}")
    for relative_path, expected in SPEC["metadata_artifacts"].items():
        path = ROOT / relative_path
        require(path.exists(), f"missing {relative_path}")
        require(sha256_file(path) == expected, f"SHA-256 mismatch for {relative_path}")
    print("metadata freeze verified")


def verify_ecosoundset() -> None:
    expected = SPEC["ecosoundset"]
    audit_path = ROOT / "data" / "audit" / "ecosoundset_audio_files_v1.csv"
    summary_path = ROOT / "data" / "audit" / "ecosoundset_audio_verification_v1.json"
    rows = csv_rows(audit_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(len(rows) == expected["audited_wavs"], "ECOSoundSet audited file count")
    require(mapping_digest(rows, "window_id") == expected["audio_sha256_mapping_digest"], "ECOSoundSet audio mapping digest")
    exclusions = [row["window_id"] for row in rows if row["duration_verified"] != "true"]
    require(len(exclusions) == expected["duration_exclusions"], "ECOSoundSet exclusion count")
    require(ids_digest(exclusions) == expected["duration_exclusion_ids_digest"], "ECOSoundSet exclusion identities")
    require(sum(row["header_verified"] == "true" for row in rows) == len(rows), "ECOSoundSet PCM/header verification")
    require(sum(row["duration_verified"] == "true" for row in rows) == expected["analysis_eligible_wavs"], "ECOSoundSet eligible count")
    for archive, values in expected["archives"].items():
        observed = summary["archive_audit"].get(archive, {})
        for field in ("bytes", "md5", "sha256"):
            require(observed.get(field) == values[field], f"ECOSoundSet {archive} {field}")
    paths = [ROOT / "data" / row["relative_path"] for row in rows]
    require(all(path.is_file() for path in paths), "one or more ECOSoundSet WAV files are missing")
    print("ECOSoundSet audio freeze verified")


def verify_pteroset() -> None:
    expected = SPEC["pteroset_ppa1"]
    audit_path = ROOT / "data" / "audit" / "pteroset_ppa1_audio_files_v1.csv"
    summary_path = ROOT / "data" / "audit" / "pteroset_ppa1_audio_verification_v1.json"
    rows = csv_rows(audit_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(len(rows) == expected["wav_members"], "PPA1 WAV count")
    require(mapping_digest(rows, "audio_file") == expected["audio_sha256_mapping_digest"], "PPA1 audio mapping digest")
    require(all(row["verified"] == "true" for row in rows), "PPA1 per-file verification")
    archive = expected["archive"]
    require(summary["archive_bytes"] == archive["bytes"], "PPA1 archive bytes")
    require(summary["archive_md5"] == archive["md5"], "PPA1 archive MD5")
    require(summary["archive_sha256"] == archive["sha256"], "PPA1 archive SHA-256")
    require((ROOT / summary["archive_relative_path"]).is_file(), "PPA1 archive is missing")
    print("PteroSet/PPA1 audio freeze verified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("metadata", "audio", "all"), default="all")
    parser.add_argument("--corpus", choices=("all", "ecosoundset", "pteroset"), default="all")
    args = parser.parse_args()

    if args.stage in {"metadata", "all"}:
        verify_metadata()
    if args.stage in {"audio", "all"}:
        if args.corpus in {"all", "ecosoundset"}:
            verify_ecosoundset()
        if args.corpus in {"all", "pteroset"}:
            verify_pteroset()
    print(f"freeze {SPEC['freeze_version']} verification passed")


if __name__ == "__main__":
    main()
