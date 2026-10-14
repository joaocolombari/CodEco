#!/usr/bin/env python3
"""Verify the downloaded PteroSet PPA1 archive and every WAV it contains."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import struct
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import BinaryIO


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "downloads" / "pteroset" / "PPA1.zip"
METADATA = ROOT / "data" / "raw_metadata" / "pteroset" / "metadata_v2.csv"
ANNOTATIONS = ROOT / "data" / "raw_metadata" / "pteroset" / "annotations_species.json"
OUTPUT_CSV = ROOT / "data" / "audit" / "pteroset_ppa1_audio_files_v1.csv"
OUTPUT_JSON = ROOT / "data" / "audit" / "pteroset_ppa1_audio_verification_v1.json"

EXPECTED_ARCHIVE_BYTES = 15_257_785_797
EXPECTED_ARCHIVE_MD5 = "d5fce3e9d743bad4988131b57aa1c9bc"
EXPECTED_RECORDINGS = 108
EXPECTED_SAMPLE_RATE = 192_000


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_exact(stream: BinaryIO, size: int, digest: hashlib._Hash) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        block = stream.read(min(8 * 1024 * 1024, remaining))
        if not block:
            raise EOFError(f"Unexpected end of WAV with {remaining} bytes missing")
        digest.update(block)
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def consume(stream: BinaryIO, size: int, digest: hashlib._Hash) -> None:
    remaining = size
    while remaining:
        block = stream.read(min(8 * 1024 * 1024, remaining))
        if not block:
            raise EOFError(f"Unexpected end of WAV with {remaining} bytes missing")
        digest.update(block)
        remaining -= len(block)


def inspect_wav(stream: BinaryIO, expected_size: int) -> dict[str, object]:
    digest = hashlib.sha256()
    riff = read_exact(stream, 12, digest)
    if riff[:4] not in {b"RIFF", b"RF64"} or riff[8:12] != b"WAVE":
        raise ValueError("Not a RIFF/RF64 WAVE file")

    riff_size = struct.unpack("<I", riff[4:8])[0]
    total_read = 12
    fmt: dict[str, int] | None = None
    data_bytes: int | None = None

    while total_read < expected_size:
        header = read_exact(stream, 8, digest)
        total_read += 8
        chunk_id = header[:4]
        chunk_size = struct.unpack("<I", header[4:8])[0]
        if chunk_id == b"fmt ":
            payload = read_exact(stream, chunk_size, digest)
            total_read += chunk_size
            if len(payload) < 16:
                raise ValueError("WAV fmt chunk is shorter than 16 bytes")
            format_code, channels, sample_rate, byte_rate, block_align, bits = struct.unpack(
                "<HHIIHH", payload[:16]
            )
            effective_format = format_code
            if format_code == 0xFFFE and len(payload) >= 40:
                effective_format = struct.unpack("<H", payload[24:26])[0]
            fmt = {
                "format_code": format_code,
                "effective_format_code": effective_format,
                "channels": channels,
                "sample_rate_hz": sample_rate,
                "byte_rate": byte_rate,
                "block_align": block_align,
                "bits_per_sample": bits,
            }
        elif chunk_id == b"data":
            data_bytes = chunk_size
            consume(stream, chunk_size, digest)
            total_read += chunk_size
        else:
            consume(stream, chunk_size, digest)
            total_read += chunk_size

        if chunk_size % 2:
            read_exact(stream, 1, digest)
            total_read += 1

    if total_read != expected_size:
        raise ValueError(f"Parsed {total_read} bytes but ZIP member has {expected_size}")
    if fmt is None or data_bytes is None:
        raise ValueError("WAV is missing fmt or data chunk")

    duration = data_bytes / fmt["byte_rate"]
    return {
        "container": riff[:4].decode("ascii"),
        "riff_size_field": riff_size,
        **fmt,
        "data_bytes": data_bytes,
        "duration_s": duration,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    archive_bytes = ARCHIVE.stat().st_size
    archive_md5 = digest_file(ARCHIVE, "md5")
    archive_sha256 = digest_file(ARCHIVE, "sha256")
    if archive_bytes != EXPECTED_ARCHIVE_BYTES or archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise ValueError(
            f"Archive mismatch: bytes={archive_bytes}, md5={archive_md5}"
        )

    with METADATA.open(encoding="utf-8-sig", newline="") as stream:
        metadata = {
            row["audio_file"]: row
            for row in csv.DictReader(stream)
            if row["project_name"] == "PPA1"
        }
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    expected_audio = {
        Path(sound["file_name_path"]).name: {
            "duration_s": float(sound["duration"]),
            "sample_rate_hz": int(sound["sample_rate"]),
        }
        for sound in annotations["sounds"]
        if sound.get("project") == "PPA1"
    }
    if len(metadata) != EXPECTED_RECORDINGS or len(expected_audio) != EXPECTED_RECORDINGS:
        raise ValueError(
            "PPA1 metadata cardinality mismatch: "
            f"metadata={len(metadata)}, annotations={len(expected_audio)}"
        )

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        wav_members = sorted(
            (item for item in archive.infolist() if item.filename.casefold().endswith(".wav")),
            key=lambda item: Path(item.filename).name,
        )
        if len(wav_members) != EXPECTED_RECORDINGS:
            raise ValueError(f"Expected {EXPECTED_RECORDINGS} WAV files, found {len(wav_members)}")

        for item in wav_members:
            file_name = Path(item.filename).name
            reasons: list[str] = []
            if file_name not in metadata:
                reasons.append("missing_metadata_row")
            if file_name not in expected_audio:
                reasons.append("missing_annotation_sound_row")
            try:
                with archive.open(item) as stream:
                    audio = inspect_wav(stream, item.file_size)
            except Exception as exc:  # Preserve the exact per-file failure in the audit.
                rows.append(
                    {
                        "audio_file": file_name,
                        "zip_member": item.filename,
                        "zip_crc32": f"{item.CRC:08x}",
                        "zip_compressed_bytes": item.compress_size,
                        "zip_uncompressed_bytes": item.file_size,
                        "verified": "false",
                        "verification_reasons": f"read_or_parse_error:{type(exc).__name__}:{exc}",
                    }
                )
                continue

            if audio["effective_format_code"] != 1:
                reasons.append("audio_format_not_pcm_integer")
            expected = expected_audio.get(file_name, {})
            expected_sample_rate = expected.get("sample_rate_hz", EXPECTED_SAMPLE_RATE)
            if audio["sample_rate_hz"] != expected_sample_rate:
                reasons.append("unexpected_sample_rate")
            if audio["channels"] < 1:
                reasons.append("invalid_channel_count")
            if audio["bits_per_sample"] not in {16, 24, 32}:
                reasons.append("unexpected_bit_depth")
            expected_duration = expected.get("duration_s")
            if expected_duration is None or not math.isclose(
                float(audio["duration_s"]), float(expected_duration), abs_tol=0.001
            ):
                reasons.append("unexpected_duration")

            metadata_row = metadata.get(file_name, {})
            rows.append(
                {
                    "audio_file": file_name,
                    "zip_member": item.filename,
                    "project": "PPA1",
                    "location_id": metadata_row.get("location_id", ""),
                    "recorder_id": metadata_row.get("recorder_id", ""),
                    "date_recorded": metadata_row.get("date_recorded", ""),
                    "zip_crc32": f"{item.CRC:08x}",
                    "zip_compressed_bytes": item.compress_size,
                    "zip_uncompressed_bytes": item.file_size,
                    **audio,
                    "verified": "true" if not reasons else "false",
                    "verification_reasons": "|".join(sorted(reasons)),
                }
            )

    fieldnames = [
        "audio_file",
        "zip_member",
        "project",
        "location_id",
        "recorder_id",
        "date_recorded",
        "container",
        "format_code",
        "effective_format_code",
        "channels",
        "sample_rate_hz",
        "bits_per_sample",
        "byte_rate",
        "block_align",
        "data_bytes",
        "duration_s",
        "zip_crc32",
        "zip_compressed_bytes",
        "zip_uncompressed_bytes",
        "sha256",
        "verified",
        "verification_reasons",
    ]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if row["verified"] != "true"]
    summary = {
        "verification_date": date.today().isoformat(),
        "archive_relative_path": str(ARCHIVE.relative_to(ROOT)),
        "archive_bytes": archive_bytes,
        "archive_md5": archive_md5,
        "archive_sha256": archive_sha256,
        "archive_matches_zenodo_record": True,
        "wav_members": len(rows),
        "verified_wav_members": len(rows) - len(failures),
        "failed_wav_members": len(failures),
        "sample_rate_counts": dict(sorted(Counter(str(row.get("sample_rate_hz", "missing")) for row in rows).items())),
        "channel_counts": dict(sorted(Counter(str(row.get("channels", "missing")) for row in rows).items())),
        "bit_depth_counts": dict(sorted(Counter(str(row.get("bits_per_sample", "missing")) for row in rows).items())),
        "format_code_counts": dict(sorted(Counter(str(row.get("effective_format_code", "missing")) for row in rows).items())),
        "duration_min_s": min(float(row["duration_s"]) for row in rows if "duration_s" in row),
        "duration_max_s": max(float(row["duration_s"]) for row in rows if "duration_s" in row),
        "total_uncompressed_bytes": sum(int(row["zip_uncompressed_bytes"]) for row in rows),
        "all_audio_verified": not failures,
        "failure_files": [row["audio_file"] for row in failures],
        "per_file_audit_relative_path": str(OUTPUT_CSV.relative_to(ROOT)),
    }
    OUTPUT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
