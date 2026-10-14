#!/usr/bin/env python3
"""Build deterministic metadata manifests for the Codeco corpora.

The script does not download audio. It transforms the metadata checked out in
``external/`` into versioned, auditable CSV manifests and a JSON lock file.
Only Python's standard library is required.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
MANIFEST_DIR = ROOT / "data" / "manifests"
AUDIT_DIR = ROOT / "data" / "audit"

ECO_REPO = EXTERNAL / "ECOSoundSet_database_analysis"
PTERO_REPO = EXTERNAL / "PteroSet"

ECO_ANNOTATION_DIR = ECO_REPO / "Input" / "Annotations by recording chunk"
ECO_SPECIES_FILE = (
    ECO_REPO
    / "Input"
    / "Species lists"
    / "Orthopteran_and_cicada_species_list_+_stridulation_capacity_in_Northwest_and_Central_Europe.csv"
)
PTERO_METADATA_DIR = ROOT / "data" / "raw_metadata" / "pteroset"
PTERO_ANNOTATIONS_FILE = PTERO_METADATA_DIR / "annotations_species.json"
PTERO_METADATA_FILE = PTERO_METADATA_DIR / "metadata_v2.csv"
PTERO_SEGMENT_FILE = PTERO_METADATA_DIR / "segment_manifest.csv"
PTERO_SPECIES_FILE = PTERO_METADATA_DIR / "species.csv"
PTERO_ZENODO_RECORD_FILE = PTERO_METADATA_DIR / "zenodo-record-21829388.json"
PTERO_AUDIO_AUDIT_CSV = AUDIT_DIR / "pteroset_ppa1_audio_files_v1.csv"
PTERO_AUDIO_AUDIT_JSON = AUDIT_DIR / "pteroset_ppa1_audio_verification_v1.json"
ECO_OFFICIAL_SEGMENTS_FILE = ROOT / "data" / "raw_metadata" / "ecosoundset" / "annotated_audio_segments.csv"
ECO_ZENODO_RECORD_FILE = ROOT / "data" / "raw_metadata" / "ecosoundset" / "zenodo-record-15043892.json"
ECO_DOWNLOAD_MANIFEST = MANIFEST_DIR / "ecosoundset_download_manifest_v1.csv"
ECO_DOWNLOAD_PLAN = AUDIT_DIR / "ecosoundset_download_plan_v1.json"
ECO_AUDIO_AUDIT_CSV = AUDIT_DIR / "ecosoundset_audio_files_v1.csv"
ECO_AUDIO_AUDIT_JSON = AUDIT_DIR / "ecosoundset_audio_verification_v1.json"
ACOUSTIC_INDICES_CONFIG = ROOT / "configs" / "acoustic-indices-v1.yaml"

MANIFEST_VERSION = "1.3.0"
FREEZE_DATE = date(2026, 10, 8).isoformat()
MIN_SAMPLE_RATE = 48_000
WINDOW_SECONDS = 4.0
MAX_ANALYSIS_FREQUENCY_HZ = 24_000.0

# Minimum support is evaluated after all signal-level eligibility rules.
ECO_MIN_TOTAL_WINDOWS = 50
ECO_MIN_TRAIN_WINDOWS = 30
ECO_MIN_VAL_WINDOWS = 5
ECO_MIN_TEST_WINDOWS = 5

PTERO_PROJECT = "PPA1"
PTERO_SOURCE_SAMPLE_RATE_HZ = 192_000
PTERO_TARGET_SAMPLE_RATE_HZ = 48_000
PTERO_MIN_EVENT_OVERLAP_SECONDS = 0.25
PTERO_MIN_TARGET_WINDOWS = 70
PTERO_MIN_TARGET_GROUPS = 15
PTERO_SPLIT_GROUP_COUNTS = {"train": 20, "val": 4, "test": 5}
PTERO_SPLIT_SEED = 20_261_007
PTERO_SPLIT_SEARCH_ITERATIONS = 100_000
PTERO_PPA1_ARCHIVE_BYTES = 15_257_785_797
PTERO_PPA1_ARCHIVE_MD5 = "d5fce3e9d743bad4988131b57aa1c9bc"
EXPECTED_PTERO_RECORDINGS = 108
EXPECTED_ECO_WAVS = 15_097


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def canonical_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean(value))
    return re.sub(r"\s+", " ", text).strip()


def parse_float(value: Any) -> float | None:
    text = clean(value).replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    parsed = parse_float(value)
    return None if parsed is None else int(parsed)


def yes(value: Any) -> bool:
    return canonical_text(value).casefold() in {"yes", "oui", "true", "1"}


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def join_values(values: Iterable[str]) -> str:
    return "|".join(sorted({canonical_text(value) for value in values if clean(value)}))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def dominant_frequency_implies_ultrasound(value: str) -> bool:
    """Return true when the curated description explicitly starts at >=24 kHz."""
    text = canonical_text(value).casefold()
    if not text or "audible" in text:
        return False
    numbers = [float(token.replace(",", ".")) for token in re.findall(r"\d+(?:[.,]\d+)?", text)]
    return bool(numbers) and min(numbers) >= 24.0


def load_ultrasound_taxa() -> tuple[set[str], set[str]]:
    ultrasound: set[str] = set()
    known: set[str] = set()
    with ECO_SPECIES_FILE.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if canonical_text(row.get("Ordre")).casefold() != "orthoptera":
                continue
            species = canonical_text(row.get("Espèce"))
            if not species:
                continue
            known.add(species)
            if yes(row.get("Ultrasons")) or dominant_frequency_implies_ultrasound(
                clean(row.get("Fréquences dominantes (kHz)"))
            ):
                ultrasound.add(species)
    return ultrasound, known


def audit_ecosoundset() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    audio_audit_summary: dict[str, Any] = {}
    audio_by_window: dict[str, dict[str, str]] = {}
    if ECO_AUDIO_AUDIT_CSV.exists() and ECO_AUDIO_AUDIT_JSON.exists():
        audio_audit_summary = json.loads(ECO_AUDIO_AUDIT_JSON.read_text(encoding="utf-8"))
        with ECO_AUDIO_AUDIT_CSV.open(encoding="utf-8-sig", newline="") as stream:
            audio_by_window = {row["window_id"]: row for row in csv.DictReader(stream)}
    audio_verification_complete = bool(
        audio_audit_summary.get("audit_complete")
        and audio_audit_summary.get("selected_wav_members") == EXPECTED_ECO_WAVS
        and len(audio_by_window) == EXPECTED_ECO_WAVS
    )

    ultrasound_taxa, known_orthoptera = load_ultrasound_taxa()
    split_files = {
        "train": ECO_ANNOTATION_DIR / "train_recording_chunks.csv",
        "val": ECO_ANNOTATION_DIR / "val_recording_chunks.csv",
        "test": ECO_ANNOTATION_DIR / "test_recording_chunks.csv",
    }

    chunks: dict[str, dict[str, Any]] = {}
    source_rows = 0
    split_conflicts: set[str] = set()

    for split, path in split_files.items():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for annotation in csv.DictReader(stream):
                source_rows += 1
                chunk_id = canonical_text(annotation.get("recording_chunk_id"))
                current = chunks.get(chunk_id)
                if current is None:
                    audio_audit = audio_by_window.get(chunk_id, {})
                    start = parse_float(annotation.get("chunk_initial_time"))
                    end = parse_float(annotation.get("chunk_final_time"))
                    current = {
                        "manifest_version": MANIFEST_VERSION,
                        "corpus": "ecosoundset",
                        "source_recording_id": canonical_text(annotation.get("recording_id")),
                        "window_id": chunk_id,
                        "source_split": split,
                        "source_group_id": canonical_text(annotation.get("date_location_id")),
                        "start_s": start,
                        "end_s": end,
                        "duration_s": None if start is None or end is None else end - start,
                        "source_sampling_rate_hz": parse_int(annotation.get("sampling_rate")),
                        "source_bit_rate_bps": parse_int(annotation.get("BPS")),
                        "source_channels": "stereo" if yes(annotation.get("stereo")) else "mono",
                        "source_annotation_exhaustive": bool_text(yes(annotation.get("annotation_exhaustive"))),
                        "source_container_expected": "wav",
                        "source_codec_provenance": "dataset_wav_prior_lossy_history_not_declared",
                        "license": "CC-BY-NC-4.0",
                        "source_file_name": canonical_text(audio_audit.get("audio_file")),
                        "source_archive": canonical_text(audio_audit.get("source_archive")),
                        "source_audio_relative_path": canonical_text(audio_audit.get("relative_path")),
                        "source_audio_sha256": canonical_text(audio_audit.get("sha256")),
                        "labels_all_set": set(),
                        "orthoptera_labels_set": set(),
                        "audible_orthoptera_labels_set": set(),
                        "ultrasonic_orthoptera_labels_set": set(),
                        "unmatched_orthoptera_labels_set": set(),
                        "annotation_count": 0,
                        "download_verified": bool_text(
                            audio_verification_complete and bool(audio_audit)
                        ),
                        "source_audio_audited": bool_text(bool(audio_audit)),
                        "audio_header_verified": bool_text(
                            audio_audit.get("header_verified", audio_audit.get("verified")) == "true"
                        ),
                        "source_duration_verified": bool_text(
                            audio_audit.get("duration_verified", audio_audit.get("verified")) == "true"
                        ),
                        "source_audio_analysis_eligible": bool_text(
                            audio_audit.get("verified") == "true"
                        ),
                    }
                    chunks[chunk_id] = current
                elif current["source_split"] != split:
                    split_conflicts.add(chunk_id)

                label = canonical_text(annotation.get("label"))
                category = canonical_text(annotation.get("label_category"))
                current["annotation_count"] += 1
                if label:
                    current["labels_all_set"].add(label)

                if category.casefold() != "orthoptera" or not label:
                    continue

                current["orthoptera_labels_set"].add(label)
                min_frequency = parse_float(annotation.get("min_frequency"))
                frequency_is_ultrasonic = (
                    min_frequency is not None and min_frequency >= MAX_ANALYSIS_FREQUENCY_HZ
                )
                if label not in known_orthoptera:
                    current["unmatched_orthoptera_labels_set"].add(label)
                elif label in ultrasound_taxa or frequency_is_ultrasonic:
                    current["ultrasonic_orthoptera_labels_set"].add(label)
                else:
                    current["audible_orthoptera_labels_set"].add(label)

    catalog: list[dict[str, Any]] = []
    support: dict[str, Counter[str]] = defaultdict(Counter)

    for chunk_id in sorted(chunks, key=lambda value: int(value)):
        row = chunks[chunk_id]
        reasons: list[str] = []
        sample_rate = row["source_sampling_rate_hz"]
        duration = row["duration_s"]

        if sample_rate is None:
            reasons.append("missing_sampling_rate")
        elif sample_rate < MIN_SAMPLE_RATE:
            reasons.append("sampling_rate_below_48000")
        if duration is None or not math.isclose(duration, WINDOW_SECONDS, abs_tol=1e-6):
            reasons.append("window_not_4_seconds")
        if row["ultrasonic_orthoptera_labels_set"]:
            reasons.append("ultrasonic_orthoptera_present")
        if row["unmatched_orthoptera_labels_set"]:
            reasons.append("orthoptera_taxonomy_unmatched")
        if not row["audible_orthoptera_labels_set"]:
            reasons.append("no_eligible_orthoptera_label")
        if chunk_id in split_conflicts:
            reasons.append("split_conflict")
        if (
            audio_verification_complete
            and row["source_audio_audited"] == "true"
            and row["source_audio_analysis_eligible"] != "true"
        ):
            reasons.append("source_audio_not_exactly_4_seconds")

        signal_eligible = not reasons
        if signal_eligible:
            for label in row["audible_orthoptera_labels_set"]:
                support[label][row["source_split"]] += 1

        flat = {
            key: value
            for key, value in row.items()
            if not key.endswith("_set")
        }
        flat.update(
            {
                "labels_all": join_values(row["labels_all_set"]),
                "orthoptera_labels": join_values(row["orthoptera_labels_set"]),
                "audible_orthoptera_labels": join_values(row["audible_orthoptera_labels_set"]),
                "ultrasonic_orthoptera_labels": join_values(row["ultrasonic_orthoptera_labels_set"]),
                "unmatched_orthoptera_labels": join_values(row["unmatched_orthoptera_labels_set"]),
                "signal_eligible": bool_text(signal_eligible),
                "exclusion_reasons": join_values(reasons),
            }
        )
        catalog.append(flat)

    selected_species = {
        species
        for species, counts in support.items()
        if sum(counts.values()) >= ECO_MIN_TOTAL_WINDOWS
        and counts["train"] >= ECO_MIN_TRAIN_WINDOWS
        and counts["val"] >= ECO_MIN_VAL_WINDOWS
        and counts["test"] >= ECO_MIN_TEST_WINDOWS
    }

    manifest: list[dict[str, Any]] = []
    for row in catalog:
        eligible_labels = set(filter(None, row["audible_orthoptera_labels"].split("|")))
        target_labels = eligible_labels & selected_species
        row["target_labels"] = join_values(target_labels)
        row["target_label_count"] = len(target_labels)
        row["included_in_frozen_manifest"] = bool_text(
            row["signal_eligible"] == "true" and bool(target_labels)
        )
        row["freeze_status"] = (
            (
                "frozen_analysis_released"
                if audio_verification_complete
                else "frozen_pending_audio_verification"
            )
            if row["included_in_frozen_manifest"] == "true"
            else "excluded"
        )
        if row["included_in_frozen_manifest"] == "true":
            manifest.append(dict(row))

    split_counts = Counter(row["source_split"] for row in manifest)
    target_support = {
        species: {
            "train": support[species]["train"],
            "val": support[species]["val"],
            "test": support[species]["test"],
            "total": sum(support[species].values()),
        }
        for species in sorted(selected_species)
    }
    summary = {
        "source_annotation_rows": source_rows,
        "unique_windows": len(catalog),
        "signal_eligible_windows": sum(row["signal_eligible"] == "true" for row in catalog),
        "frozen_windows": len(manifest),
        "frozen_split_counts": dict(sorted(split_counts.items())),
        "selected_species_count": len(selected_species),
        "selected_species_support": target_support,
        "curated_ultrasonic_species_count": len(ultrasound_taxa),
        "split_conflicts": len(split_conflicts),
        "audio_verification_complete": audio_verification_complete,
        "audio_verification_summary": audio_audit_summary,
        "analysis_release_status": (
            "released_for_analysis_after_file_level_filtering"
            if audio_verification_complete
            else "conditional_on_audio_download_and_header_verification"
        ),
    }
    return summary, catalog, manifest


def select_pteroset_groups(
    groups: list[str],
    group_windows: Counter[str],
    group_species_windows: dict[str, Counter[str]],
    target_species: list[str],
) -> tuple[dict[str, set[str]], float]:
    """Find a deterministic group-safe split with useful support in every partition."""
    expected_ratios = {
        split: count / len(groups) for split, count in PTERO_SPLIT_GROUP_COUNTS.items()
    }
    total_species_windows: Counter[str] = Counter()
    for counts in group_species_windows.values():
        total_species_windows.update(counts)

    rng = random.Random(PTERO_SPLIT_SEED)
    best: tuple[float, dict[str, set[str]]] | None = None
    for _ in range(PTERO_SPLIT_SEARCH_ITERATIONS):
        shuffled = groups[:]
        rng.shuffle(shuffled)
        val_end = PTERO_SPLIT_GROUP_COUNTS["val"]
        test_end = val_end + PTERO_SPLIT_GROUP_COUNTS["test"]
        assignments = {
            "val": set(shuffled[:val_end]),
            "test": set(shuffled[val_end:test_end]),
            "train": set(shuffled[test_end:]),
        }

        split_species_windows: dict[str, Counter[str]] = {}
        split_species_groups: dict[str, Counter[str]] = {}
        for split, split_groups in assignments.items():
            window_counts: Counter[str] = Counter()
            group_counts: Counter[str] = Counter()
            for group in split_groups:
                window_counts.update(group_species_windows[group])
                for species in target_species:
                    if group_species_windows[group][species] > 0:
                        group_counts[species] += 1
            split_species_windows[split] = window_counts
            split_species_groups[split] = group_counts

        score = 0.0
        for split in ("train", "val", "test"):
            minimum_windows = 20 if split == "train" else 5
            minimum_groups = 5 if split == "train" else 2
            for species in target_species:
                if split_species_windows[split][species] < minimum_windows:
                    score += 10_000
                if split_species_groups[split][species] < minimum_groups:
                    score += 10_000
                expected = total_species_windows[species] * expected_ratios[split]
                score += (
                    (split_species_windows[split][species] - expected)
                    / max(5.0, expected)
                ) ** 2

        split_window_counts = {
            split: sum(group_windows[group] for group in split_groups)
            for split, split_groups in assignments.items()
        }
        total_windows = sum(group_windows.values())
        for split, count in split_window_counts.items():
            expected = total_windows * expected_ratios[split]
            score += ((count - expected) / expected) ** 2

        if best is None or score < best[0]:
            best = (score, assignments)

    if best is None:
        raise RuntimeError("No PteroSet split assignment was generated")
    return best[1], best[0]


def audit_pteroset() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    audio_audit_summary: dict[str, Any] = {}
    audio_by_file: dict[str, dict[str, str]] = {}
    if PTERO_AUDIO_AUDIT_CSV.exists() and PTERO_AUDIO_AUDIT_JSON.exists():
        audio_audit_summary = json.loads(PTERO_AUDIO_AUDIT_JSON.read_text(encoding="utf-8"))
        with PTERO_AUDIO_AUDIT_CSV.open(encoding="utf-8-sig", newline="") as stream:
            audio_by_file = {row["audio_file"]: row for row in csv.DictReader(stream)}
    audio_verification_complete = bool(
        audio_audit_summary.get("all_audio_verified")
        and len(audio_by_file) == EXPECTED_PTERO_RECORDINGS
    )

    annotations = json.loads(PTERO_ANNOTATIONS_FILE.read_text(encoding="utf-8"))
    categories = {
        int(item["id"]): canonical_text(item["species"])
        for item in annotations["categories"]
    }
    sounds = {int(item["id"]): item for item in annotations["sounds"]}
    ppa1_sound_ids = {
        sound_id
        for sound_id, item in sounds.items()
        if canonical_text(item.get("project")) == PTERO_PROJECT
    }
    annotations_by_sound: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    source_annotation_events = 0
    source_annotation_species: set[str] = set()
    for annotation in annotations["annotations"]:
        sound_id = int(annotation["sound_id"])
        if sound_id not in ppa1_sound_ids:
            continue
        source_annotation_events += 1
        source_annotation_species.add(categories[int(annotation["category_id"])])
        annotations_by_sound[sound_id].append(
            (
                float(annotation["t_min"]),
                float(annotation["t_max"]),
                categories[int(annotation["category_id"])],
            )
        )

    with PTERO_METADATA_FILE.open(encoding="utf-8-sig", newline="") as stream:
        metadata_rows = [
            row for row in csv.DictReader(stream) if canonical_text(row.get("project_name")) == PTERO_PROJECT
        ]
    metadata_by_audio = {canonical_text(row["audio_file"]): row for row in metadata_rows}

    with PTERO_SEGMENT_FILE.open(encoding="utf-8-sig", newline="") as stream:
        segments = [
            row for row in csv.DictReader(stream) if canonical_text(row.get("project")) == PTERO_PROJECT
        ]

    catalog: list[dict[str, Any]] = []
    species_windows: Counter[str] = Counter()
    species_groups: dict[str, set[str]] = defaultdict(set)
    group_windows: Counter[str] = Counter()
    group_species_windows: dict[str, Counter[str]] = defaultdict(Counter)

    for segment in segments:
        sound_id = int(segment["sound_id"])
        file_name = canonical_text(segment["audio_file"])
        metadata = metadata_by_audio[file_name]
        audio_audit = audio_by_file.get(file_name, {})
        location_id = canonical_text(segment["location_id"])
        recorder_id = canonical_text(segment["recorder_id"])
        group_id = f"{PTERO_PROJECT}|{location_id}|{recorder_id}"
        segment_start = float(segment["start_sec_in_file"])
        segment_end = float(segment["end_sec_in_file"])
        sample_rate = int(segment["source_sample_rate"])

        if sample_rate != PTERO_SOURCE_SAMPLE_RATE_HZ:
            raise ValueError(f"Unexpected PteroSet sample rate: {sample_rate}")
        if not math.isclose(segment_end - segment_start, 10.0, abs_tol=1e-6):
            raise ValueError(f"Unexpected PteroSet segment duration: {segment['segment_id']}")

        for slot in range(2):
            start = segment_start + slot * WINDOW_SECONDS
            end = start + WINDOW_SECONDS
            labels = {
                species
                for event_start, event_end, species in annotations_by_sound[sound_id]
                if max(0.0, min(end, event_end) - max(start, event_start))
                >= PTERO_MIN_EVENT_OVERLAP_SECONDS
            }
            window_id = f"{canonical_text(segment['segment_id'])}-w{slot}"
            row = {
                "manifest_version": MANIFEST_VERSION,
                "corpus": "pteroset_ppa1",
                "source_project": PTERO_PROJECT,
                "source_recording_id": f"pteroset-sound-{sound_id}",
                "sound_id": sound_id,
                "source_file_name": file_name,
                "window_id": window_id,
                "segment_id": canonical_text(segment["segment_id"]),
                "segment_index": int(segment["segment_index"]),
                "window_slot": slot,
                "source_group_id": group_id,
                "location_id": location_id,
                "recorder_id": recorder_id,
                "recording_date": canonical_text(segment["date_recorded"]),
                "country": canonical_text(metadata.get("country")),
                "department": canonical_text(metadata.get("department")),
                "municipality": canonical_text(metadata.get("municipality")),
                "land_cover": canonical_text(metadata.get("land_cover")),
                "latitude": parse_float(metadata.get("latitude")),
                "longitude": parse_float(metadata.get("longitude")),
                "segment_start_s": segment_start,
                "segment_end_s": segment_end,
                "window_start_s": start,
                "window_end_s": end,
                "duration_s": WINDOW_SECONDS,
                "source_sampling_rate_hz": sample_rate,
                "analysis_sampling_rate_hz": PTERO_TARGET_SAMPLE_RATE_HZ,
                "source_channels": parse_int(audio_audit.get("channels")) or "pending_audio_header_verification",
                "source_bits_per_sample": parse_int(audio_audit.get("bits_per_sample")),
                "source_effective_format_code": parse_int(audio_audit.get("effective_format_code")),
                "source_audio_sha256": canonical_text(audio_audit.get("sha256")),
                "source_container_expected": "wav",
                "source_codec_provenance": "pteroset_original_wav_prior_lossy_history_not_declared",
                "license": "CC-BY-4.0",
                "annotation_scope": "species_identified_events",
                "minimum_event_overlap_s": PTERO_MIN_EVENT_OVERLAP_SECONDS,
                "labels_all_species": join_values(labels),
                "download_verified": bool_text(audio_verification_complete),
                "audio_header_verified": bool_text(audio_audit.get("verified") == "true"),
            }
            catalog.append(row)
            group_windows[group_id] += 1
            for species in labels:
                species_windows[species] += 1
                species_groups[species].add(group_id)
                group_species_windows[group_id][species] += 1

    target_species = sorted(
        species
        for species, count in species_windows.items()
        if count >= PTERO_MIN_TARGET_WINDOWS
        and len(species_groups[species]) >= PTERO_MIN_TARGET_GROUPS
    )
    groups = sorted(group_windows)
    expected_group_count = sum(PTERO_SPLIT_GROUP_COUNTS.values())
    if len(groups) != expected_group_count:
        raise ValueError(
            f"Expected {expected_group_count} PPA1 groups, observed {len(groups)}"
        )
    split_groups, split_score = select_pteroset_groups(
        groups, group_windows, group_species_windows, target_species
    )
    group_split = {
        group: split for split, split_group_set in split_groups.items() for group in split_group_set
    }

    manifest: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    support: dict[str, Counter[str]] = defaultdict(Counter)
    support_groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    target_positive_windows = 0
    for row in catalog:
        all_labels = set(filter(None, row["labels_all_species"].split("|")))
        labels = all_labels & set(target_species)
        split = group_split[row["source_group_id"]]
        row["target_labels"] = join_values(labels)
        row["target_label_count"] = len(labels)
        row["target_background"] = bool_text(not labels)
        row["source_split"] = split
        row["split_group_safe"] = "true"
        row["included_in_frozen_manifest"] = "true"
        row["freeze_status"] = (
            "frozen_analysis_released"
            if audio_verification_complete
            else "frozen_pending_audio_verification"
        )
        manifest.append(dict(row))
        split_counts[split] += 1
        if labels:
            target_positive_windows += 1
        for species in labels:
            support[species][split] += 1
            support_groups[species][split].add(row["source_group_id"])

    target_support = {
        species: {
            "train": support[species]["train"],
            "val": support[species]["val"],
            "test": support[species]["test"],
            "total": sum(support[species].values()),
            "groups_total": len(species_groups[species]),
            "groups_train": len(support_groups[species]["train"]),
            "groups_val": len(support_groups[species]["val"]),
            "groups_test": len(support_groups[species]["test"]),
        }
        for species in target_species
    }
    summary = {
        "source_project": PTERO_PROJECT,
        "source_recordings": len(ppa1_sound_ids),
        "source_metadata_rows": len(metadata_rows),
        "source_annotation_events": source_annotation_events,
        "source_species_annotated": len(source_annotation_species),
        "species_represented_in_derived_windows": len(species_windows),
        "source_segments_10s": len(segments),
        "source_groups": len(groups),
        "source_sample_rate_hz": PTERO_SOURCE_SAMPLE_RATE_HZ,
        "source_container_expected": "wav",
        "source_archive_bytes": PTERO_PPA1_ARCHIVE_BYTES,
        "source_archive_md5": PTERO_PPA1_ARCHIVE_MD5,
        "derived_windows_4s": len(manifest),
        "derived_audio_hours": len(manifest) * WINDOW_SECONDS / 3600,
        "target_positive_windows": target_positive_windows,
        "target_background_windows": len(manifest) - target_positive_windows,
        "selected_species_count": len(target_species),
        "selected_species_support": target_support,
        "split_counts": dict(sorted(split_counts.items())),
        "split_groups": {
            split: sorted(split_group_set) for split, split_group_set in split_groups.items()
        },
        "split_group_counts": {
            split: len(split_group_set) for split, split_group_set in split_groups.items()
        },
        "split_optimization_score": round(split_score, 9),
        "split_group_leakage": 0,
        "window_temporal_overlap_by_design": False,
        "unused_seconds_per_10s_snapshot": 2.0,
        "license": "CC-BY-4.0",
        "audio_verification_complete": audio_verification_complete,
        "audio_verification_summary": audio_audit_summary,
        "analysis_release_status": (
            "released_for_analysis"
            if audio_verification_complete
            else "conditional_on_audio_download_and_header_verification"
        ),
    }
    return summary, catalog, manifest


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    eco_summary, eco_catalog, eco_manifest = audit_ecosoundset()
    bird_summary, bird_catalog, bird_manifest = audit_pteroset()

    eco_fields = [
        "manifest_version",
        "corpus",
        "source_recording_id",
        "window_id",
        "source_split",
        "source_group_id",
        "start_s",
        "end_s",
        "duration_s",
        "source_sampling_rate_hz",
        "source_bit_rate_bps",
        "source_channels",
        "source_annotation_exhaustive",
        "source_container_expected",
        "source_codec_provenance",
        "license",
        "source_file_name",
        "source_archive",
        "source_audio_relative_path",
        "source_audio_sha256",
        "labels_all",
        "orthoptera_labels",
        "audible_orthoptera_labels",
        "ultrasonic_orthoptera_labels",
        "unmatched_orthoptera_labels",
        "target_labels",
        "target_label_count",
        "annotation_count",
        "signal_eligible",
        "included_in_frozen_manifest",
        "exclusion_reasons",
        "download_verified",
        "source_audio_audited",
        "audio_header_verified",
        "source_duration_verified",
        "source_audio_analysis_eligible",
        "freeze_status",
    ]
    bird_fields = [
        "manifest_version",
        "corpus",
        "source_project",
        "source_recording_id",
        "sound_id",
        "source_file_name",
        "window_id",
        "segment_id",
        "segment_index",
        "window_slot",
        "source_group_id",
        "location_id",
        "recorder_id",
        "recording_date",
        "country",
        "department",
        "municipality",
        "land_cover",
        "latitude",
        "longitude",
        "segment_start_s",
        "segment_end_s",
        "window_start_s",
        "window_end_s",
        "duration_s",
        "source_sampling_rate_hz",
        "analysis_sampling_rate_hz",
        "source_channels",
        "source_bits_per_sample",
        "source_effective_format_code",
        "source_audio_sha256",
        "source_container_expected",
        "source_codec_provenance",
        "license",
        "annotation_scope",
        "minimum_event_overlap_s",
        "labels_all_species",
        "target_labels",
        "target_label_count",
        "target_background",
        "source_split",
        "split_group_safe",
        "included_in_frozen_manifest",
        "download_verified",
        "audio_header_verified",
        "freeze_status",
    ]

    outputs = {
        "ecosoundset_catalog": MANIFEST_DIR / "ecosoundset_catalog_v1.csv",
        "ecosoundset_manifest": MANIFEST_DIR / "ecosoundset_manifest_v1.csv",
        "pteroset_ppa1_catalog": MANIFEST_DIR / "pteroset_ppa1_catalog_v1.csv",
        "pteroset_ppa1_manifest": MANIFEST_DIR / "pteroset_ppa1_manifest_v1.csv",
    }
    if PTERO_AUDIO_AUDIT_CSV.exists() and PTERO_AUDIO_AUDIT_JSON.exists():
        outputs["pteroset_ppa1_audio_files_audit"] = PTERO_AUDIO_AUDIT_CSV
        outputs["pteroset_ppa1_audio_verification"] = PTERO_AUDIO_AUDIT_JSON
    if ECO_DOWNLOAD_MANIFEST.exists() and ECO_DOWNLOAD_PLAN.exists():
        outputs["ecosoundset_download_manifest"] = ECO_DOWNLOAD_MANIFEST
        outputs["ecosoundset_download_plan"] = ECO_DOWNLOAD_PLAN
    if ECO_AUDIO_AUDIT_CSV.exists() and ECO_AUDIO_AUDIT_JSON.exists():
        outputs["ecosoundset_audio_files_audit"] = ECO_AUDIO_AUDIT_CSV
        outputs["ecosoundset_audio_verification"] = ECO_AUDIO_AUDIT_JSON
    write_csv(outputs["ecosoundset_catalog"], eco_catalog, eco_fields)
    write_csv(outputs["ecosoundset_manifest"], eco_manifest, eco_fields)
    write_csv(outputs["pteroset_ppa1_catalog"], bird_catalog, bird_fields)
    write_csv(outputs["pteroset_ppa1_manifest"], bird_manifest, bird_fields)

    source_paths = {
        "ecosoundset_species": ECO_SPECIES_FILE,
        "ecosoundset_train": ECO_ANNOTATION_DIR / "train_recording_chunks.csv",
        "ecosoundset_val": ECO_ANNOTATION_DIR / "val_recording_chunks.csv",
        "ecosoundset_test": ECO_ANNOTATION_DIR / "test_recording_chunks.csv",
        "ecosoundset_official_segments": ECO_OFFICIAL_SEGMENTS_FILE,
        "ecosoundset_zenodo_record": ECO_ZENODO_RECORD_FILE,
        "acoustic_indices_config": ACOUSTIC_INDICES_CONFIG,
        "pteroset_annotations_species": PTERO_ANNOTATIONS_FILE,
        "pteroset_metadata": PTERO_METADATA_FILE,
        "pteroset_segments": PTERO_SEGMENT_FILE,
        "pteroset_species": PTERO_SPECIES_FILE,
        "pteroset_zenodo_record": PTERO_ZENODO_RECORD_FILE,
    }

    lock = {
        "manifest_version": MANIFEST_VERSION,
        "freeze_date": FREEZE_DATE,
        "protocol": {
            "window_seconds": WINDOW_SECONDS,
            "minimum_source_sampling_rate_hz": MIN_SAMPLE_RATE,
            "maximum_analysis_frequency_hz": MAX_ANALYSIS_FREQUENCY_HZ,
            "ecosoundset_species_support": {
                "minimum_total_windows": ECO_MIN_TOTAL_WINDOWS,
                "minimum_train_windows": ECO_MIN_TRAIN_WINDOWS,
                "minimum_val_windows": ECO_MIN_VAL_WINDOWS,
                "minimum_test_windows": ECO_MIN_TEST_WINDOWS,
            },
            "pteroset_ppa1": {
                "source_project": PTERO_PROJECT,
                "window_construction": "two_windows_per_10s_snapshot_at_offsets_0s_and_4s",
                "minimum_event_overlap_seconds": PTERO_MIN_EVENT_OVERLAP_SECONDS,
                "minimum_target_windows": PTERO_MIN_TARGET_WINDOWS,
                "minimum_target_groups": PTERO_MIN_TARGET_GROUPS,
                "split_group_counts": PTERO_SPLIT_GROUP_COUNTS,
                "split_seed": PTERO_SPLIT_SEED,
                "split_search_iterations": PTERO_SPLIT_SEARCH_ITERATIONS,
            },
        },
        "repositories": {
            "ecosoundset": {
                "url": "https://github.com/DavidFunosas/ECOSoundSet_database_analysis.git",
                "revision": git_revision(ECO_REPO),
                "dataset_doi": "10.5281/zenodo.18636037",
                "dataset_license": "CC-BY-NC-4.0",
            },
            "pteroset": {
                "url": "https://github.com/microsoft/PteroSet.git",
                "revision": git_revision(PTERO_REPO),
                "dataset_doi": "10.5281/zenodo.21829388",
                "dataset_license": "CC-BY-4.0",
            },
        },
        "source_files": {
            key: {
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for key, path in source_paths.items()
        },
        "outputs": {
            key: {
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for key, path in outputs.items()
        },
        "audit": {
            "ecosoundset": eco_summary,
            "pteroset_ppa1": bird_summary,
        },
    }
    if bird_summary.get("audio_verification_complete"):
        verification = bird_summary["audio_verification_summary"]
        lock["verified_local_audio"] = {
            "pteroset_ppa1_archive": {
                "relative_path": verification["archive_relative_path"],
                "bytes": verification["archive_bytes"],
                "md5": verification["archive_md5"],
                "sha256": verification["archive_sha256"],
            }
        }
    if eco_summary.get("audio_verification_complete"):
        verification = eco_summary["audio_verification_summary"]
        lock["verified_remote_audio"] = {
            "ecosoundset_selected_wavs": {
                "host": "easn-cloud",
                "absolute_path": "/srv/easn/shared/codeco/data/audio/ecosoundset/source",
                "files_audited": verification["selected_wav_members"],
                "files_analysis_eligible": verification["analysis_eligible_wav_members"],
                "files_excluded_after_audio_audit": verification["failed_wav_members"],
                "bytes": verification["total_selected_wav_bytes"],
                "archive_audit": verification["archive_audit"],
            }
        }

    lock_path = ROOT / "data" / "manifest-lock-v1.json"
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (AUDIT_DIR / "corpus-audit-v1.json").write_text(
        json.dumps(lock["audit"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(lock["audit"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
