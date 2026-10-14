#!/usr/bin/env python3
"""Map the frozen ECOSoundSet windows to the official Zenodo WAV names."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "manifests" / "ecosoundset_manifest_v1.csv"
OFFICIAL = ROOT / "data" / "raw_metadata" / "ecosoundset" / "annotated_audio_segments.csv"
ZENODO = ROOT / "data" / "raw_metadata" / "ecosoundset" / "zenodo-record-15043892.json"
OUTPUT = ROOT / "data" / "manifests" / "ecosoundset_download_manifest_v1.csv"
SUMMARY = ROOT / "data" / "audit" / "ecosoundset_download_plan_v1.json"


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(recording_id: str, start: str, end: str, split: str) -> tuple[str, float, float, str]:
    return recording_id.strip(), float(start), float(end), split.strip()


def main() -> None:
    official: dict[tuple[str, float, float, str], set[str]] = defaultdict(set)
    with OFFICIAL.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            official[
                key(
                    row["recording_id"],
                    row["audio_segment_initial_time"],
                    row["audio_segment_final_time"],
                    row["subset"],
                )
            ].add(row["audio_segment_file_name"].strip())

    rows: list[dict[str, object]] = []
    with FROZEN.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            lookup_key = key(
                row["source_recording_id"], row["start_s"], row["end_s"], row["source_split"]
            )
            names = official.get(lookup_key, set())
            if len(names) != 1:
                raise ValueError(f"Expected one official WAV for {lookup_key}, found {sorted(names)}")
            channels = 2 if row["source_channels"] == "stereo" else 1
            estimated_bytes = (
                round(float(row["duration_s"]) * int(row["source_sampling_rate_hz"]) * channels
                * int(row["source_bit_rate_bps"]) / 8)
                + 44
            )
            file_name = next(iter(names))
            rows.append(
                {
                    "manifest_version": row["manifest_version"],
                    "window_id": row["window_id"],
                    "source_recording_id": row["source_recording_id"],
                    "source_split": row["source_split"],
                    "start_s": row["start_s"],
                    "end_s": row["end_s"],
                    "audio_segment_file_name": file_name,
                    "archive_member_expected": file_name,
                    "source_sampling_rate_hz": row["source_sampling_rate_hz"],
                    "source_channels": row["source_channels"],
                    "source_bits_per_sample": row["source_bit_rate_bps"],
                    "estimated_wav_bytes": estimated_bytes,
                }
            )

    fields = list(rows[0])
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    record = json.loads(ZENODO.read_text(encoding="utf-8"))
    archive_files = {
        item["key"]: {
            "bytes": item["size"],
            "md5": item["checksum"].removeprefix("md5:"),
            "url": item["links"]["self"],
        }
        for item in record["files"]
        if item["key"].startswith("Split recordings")
    }
    summary = {
        "generated_on": "2026-10-13",
        "zenodo_parent_record": 15043892,
        "zenodo_resolved_record": record["id"],
        "zenodo_doi": record["doi"],
        "official_metadata_bytes": OFFICIAL.stat().st_size,
        "official_metadata_md5": file_hash(OFFICIAL, "md5"),
        "official_metadata_sha256": file_hash(OFFICIAL),
        "selected_wav_count": len(rows),
        "estimated_selected_wav_bytes": sum(int(row["estimated_wav_bytes"]) for row in rows),
        "estimated_selected_wav_gib": round(
            sum(int(row["estimated_wav_bytes"]) for row in rows) / 2**30, 6
        ),
        "split_archives": archive_files,
        "split_archives_total_bytes": sum(item["bytes"] for item in archive_files.values()),
        "download_manifest_relative_path": str(OUTPUT.relative_to(ROOT)),
        "download_manifest_sha256": file_hash(OUTPUT),
        "local_execution_decision": "defer_full_download_to_easncloud_due_to_local_disk_capacity",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
