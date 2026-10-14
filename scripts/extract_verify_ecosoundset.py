#!/usr/bin/env python3
"""Selectively extract and verify the frozen ECOSoundSet WAV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tarfile
import wave
from collections import Counter
from datetime import date
from pathlib import Path


ARCHIVES = {
    "Split recordings1.tar.gz": (13_785_342_839, "c61086491aad58583ea2d3fba66b8d42"),
    "Split recordings2.tar.gz": (9_084_639_756, "4069994f8aa6602b4ac9e62b99746eb6"),
}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_selection(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selection = {row["audio_segment_file_name"]: row for row in rows}
    if len(selection) != len(rows):
        raise ValueError("Download manifest contains duplicate WAV names")
    return selection


def inspect_wav(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        sample_rate = stream.getframerate()
        frames = stream.getnframes()
        compression = stream.getcomptype()
    return {
        "channels": channels,
        "bits_per_sample": sample_width * 8,
        "sample_rate_hz": sample_rate,
        "frames": frames,
        "duration_s": frames / sample_rate,
        "wave_compression": compression,
    }


def verify_archive(path: Path, expected_bytes: int, expected_md5: str) -> dict[str, object]:
    observed_bytes = path.stat().st_size
    observed_md5 = digest(path, "md5")
    if observed_bytes != expected_bytes or observed_md5 != expected_md5:
        raise ValueError(
            f"Archive verification failed for {path.name}: "
            f"bytes={observed_bytes}, md5={observed_md5}"
        )
    return {
        "bytes": observed_bytes,
        "md5": observed_md5,
        "sha256": digest(path, "sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    selection = load_selection(args.selection.resolve())
    download_dir = data_root / "downloads" / "ecosoundset"
    output_dir = data_root / "audio" / "ecosoundset" / "source"
    audit_dir = data_root / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    archive_audit: dict[str, dict[str, object]] = {}
    extracted_from: dict[str, str] = {}
    for archive_name, (expected_bytes, expected_md5) in ARCHIVES.items():
        archive_path = download_dir / archive_name
        archive_audit[archive_name] = verify_archive(
            archive_path, expected_bytes, expected_md5
        )
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if name not in selection:
                    continue
                if name in extracted_from:
                    raise ValueError(
                        f"Selected WAV {name} occurs in both {extracted_from[name]} and {archive_name}"
                    )
                split = selection[name]["source_split"]
                destination = output_dir / split / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".partial")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Unable to read {member.name} from {archive_name}")
                with source, temporary.open("wb") as target:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        target.write(block)
                os.replace(temporary, destination)
                extracted_from[name] = archive_name

    missing = sorted(set(selection) - set(extracted_from))
    if missing:
        raise ValueError(f"Missing {len(missing)} selected WAVs; first entries: {missing[:10]}")

    rows: list[dict[str, object]] = []
    for name, expected in sorted(selection.items()):
        path = output_dir / expected["source_split"] / name
        header_reasons: list[str] = []
        duration_reasons: list[str] = []
        try:
            audio = inspect_wav(path)
            if audio["wave_compression"] != "NONE":
                header_reasons.append("not_uncompressed_pcm")
            if audio["sample_rate_hz"] != int(expected["source_sampling_rate_hz"]):
                header_reasons.append("sample_rate_mismatch")
            expected_channels = 2 if expected["source_channels"] == "stereo" else 1
            if audio["channels"] != expected_channels:
                header_reasons.append("channel_mismatch")
            if audio["bits_per_sample"] != int(expected["source_bits_per_sample"]):
                header_reasons.append("bit_depth_mismatch")
            if abs(float(audio["duration_s"]) - 4.0) > 1 / int(audio["sample_rate_hz"]):
                duration_reasons.append("duration_mismatch")
        except Exception as exc:
            audio = {}
            header_reasons.append(f"read_or_parse_error:{type(exc).__name__}:{exc}")
        reasons = header_reasons + duration_reasons
        rows.append(
            {
                "window_id": expected["window_id"],
                "audio_file": name,
                "source_split": expected["source_split"],
                "source_archive": extracted_from[name],
                "relative_path": str(path.relative_to(data_root)),
                "bytes": path.stat().st_size,
                "sha256": digest(path, "sha256"),
                **audio,
                "header_verified": "true" if not header_reasons else "false",
                "duration_verified": "true" if not header_reasons and not duration_reasons else "false",
                "verified": "true" if not reasons else "false",
                "verification_reasons": "|".join(reasons),
            }
        )

    audit_csv = audit_dir / "ecosoundset_audio_files_v1.csv"
    fields = [
        "window_id", "audio_file", "source_split", "source_archive", "relative_path",
        "bytes", "sha256", "channels", "bits_per_sample", "sample_rate_hz", "frames",
        "duration_s", "wave_compression", "header_verified", "duration_verified",
        "verified", "verification_reasons",
    ]
    with audit_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if row["verified"] != "true"]
    blocking_failures = [row for row in rows if row["header_verified"] != "true"]
    summary = {
        "verification_date": date.today().isoformat(),
        "selected_wav_members": len(rows),
        "verified_wav_members": len(rows) - len(failures),
        "failed_wav_members": len(failures),
        "all_audio_verified": not failures,
        "audit_complete": len(rows) == len(selection),
        "analysis_eligible_wav_members": len(rows) - len(failures),
        "header_verified_wav_members": sum(row["header_verified"] == "true" for row in rows),
        "duration_mismatch_wav_members": sum(
            "duration_mismatch" in str(row["verification_reasons"]) for row in rows
        ),
        "blocking_failure_wav_members": len(blocking_failures),
        "archive_audit": archive_audit,
        "total_selected_wav_bytes": sum(int(row["bytes"]) for row in rows),
        "sample_rate_counts": dict(sorted(Counter(str(row.get("sample_rate_hz", "missing")) for row in rows).items())),
        "channel_counts": dict(sorted(Counter(str(row.get("channels", "missing")) for row in rows).items())),
        "bit_depth_counts": dict(sorted(Counter(str(row.get("bits_per_sample", "missing")) for row in rows).items())),
        "split_counts": dict(sorted(Counter(str(row["source_split"]) for row in rows).items())),
        "failure_files": [row["audio_file"] for row in failures],
        "per_file_audit": str(audit_csv.relative_to(data_root)),
    }
    audit_json = audit_dir / "ecosoundset_audio_verification_v1.json"
    audit_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if blocking_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
