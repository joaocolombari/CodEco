"""Build the in-memory clip table from the frozen manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as compute
import pyarrow.parquet as parquet
import pandas as pd
from tqdm import tqdm

from codeco.codeco_path import CodecoPath
from codeco.corpora import (
    CORPUS_MANIFESTS,
    SelectionRule,
    deterministic_balanced_training_sample,
    load_frozen_manifest,
    select_classes,
)
from codeco.experiment_config import experiment_config


CLIP_SECONDS = 4.0
WINDOW_SECONDS = CLIP_SECONDS
SAMPLE_RATE = 48_000
BIT_DEPTH = 16
CACHE_SCHEMA_VERSION = "2"
CACHE_FILE_NAME = "clip_table.parquet"
PCM_CACHE_SCHEMA_VERSION = "1"
PCM_CACHE_BATCH_SIZE = 64
PCM_SAMPLES = round(SAMPLE_RATE * CLIP_SECONDS)
PCM_INDEX_FILE_NAME = "index.parquet"
PCM_METADATA_FILE_NAME = "metadata.json"
CACHE_METADATA = {
    b"codeco_cache_kind": b"clip_table",
    b"codeco_cache_schema_version": CACHE_SCHEMA_VERSION.encode("ascii"),
}
LOGGER = logging.getLogger("codeco.prepare_table")
WINDOW_COLUMNS = (
    "dataset",
    "resource_type",
    "window_id",
    "source_file_name",
    "source_audio_relative_path",
    "window_start_s",
    "source_split",
    "target_labels",
    "source_group_id",
)
CLIP_TABLE_SCHEMA = pa.schema(
    [
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("path", pa.string(), nullable=False),
        pa.field("audio_available", pa.bool_(), nullable=False),
        pa.field("start", pa.float64(), nullable=False),
        pa.field("end", pa.float64(), nullable=False),
        pa.field("labels", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("source_group_id", pa.string(), nullable=False),
    ]
)
PCM_INDEX_SCHEMA = pa.schema(
    [
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("pcm_file", pa.string(), nullable=False),
        pa.field("pcm_row", pa.int64(), nullable=False),
    ]
)


@dataclass(frozen=True)
class BaselineState:
    """Canonical clip index and audio protocol shared by pipeline stages."""

    table: pa.Table
    cache_file: Path
    manifest_sha256: str
    sample_rate: int
    clip_seconds: float
    bit_depth: int
    pcm_cache_dir: Path | None = None
    pcm_index: pa.Table | None = None


@lru_cache(maxsize=None)
def _open_pcm_cache(pcm_file: str) -> np.memmap:
    """Open one PCM array once in each persistent worker process."""
    cache_path = Path(pcm_file)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cache PCM nao encontrado: {cache_path}")
    mapped = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    if not isinstance(mapped, np.memmap):
        raise ValueError(f"Cache PCM nao foi mapeado em memoria: {cache_path}")
    if mapped.dtype != np.dtype(np.int16):
        raise ValueError(
            f"Cache PCM com dtype {mapped.dtype}; esperado int16: {cache_path}"
        )
    if mapped.ndim != 2 or mapped.shape[1] != PCM_SAMPLES:
        raise ValueError(
            f"Cache PCM com shape {mapped.shape}; esperado (*, {PCM_SAMPLES}): "
            f"{cache_path}"
        )
    return mapped


def load_pcm_clip(pcm_file: str | Path, pcm_row: int) -> np.ndarray:
    """Load one canonical clip from the shared process-local PCM mapping."""
    cache_path = str(Path(pcm_file).resolve())
    mapped = _open_pcm_cache(cache_path)
    if pcm_row < 0 or pcm_row >= mapped.shape[0]:
        raise IndexError(
            f"Linha PCM {pcm_row} fora do intervalo 0..{mapped.shape[0] - 1}: "
            f"{cache_path}"
        )
    return np.asarray(mapped[pcm_row], dtype=np.float32) / 32768.0


class TqdmLoggingHandler(logging.StreamHandler):
    """Write log records without corrupting active progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


def _configure_logging(paths: CodecoPath) -> tuple[Path, bool]:
    if LOGGER.handlers:
        return paths.logs / "prepare_table.log", False
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    terminal = TqdmLoggingHandler(sys.stdout)
    terminal.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(terminal)

    created = not paths.logs.is_dir()
    paths.logs.mkdir(parents=True, exist_ok=True)
    log_file = paths.logs / "prepare_table.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    LOGGER.addHandler(file_handler)
    return log_file, created


def _pipeline_header(name: str) -> None:
    line = "=" * 72
    script = str(Path(sys.argv[0]).resolve())
    command = subprocess.list2cmdline([sys.executable, script, *sys.argv[1:]])
    LOGGER.info("\n%s\n%s\n[CMD]   %s\n%s", line, name, command, line)


def _section(name: str) -> None:
    LOGGER.info("\n%s\n%s", name, "-" * 72)


def _manifest_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Manifesto nao encontrado: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_definition_sha256(paths: CodecoPath) -> str:
    """Hash the lock, manifests and selection configuration as one definition."""
    sources = (
        paths.manifest,
        *paths.frozen_manifests,
        paths.root / "configs" / "experiment-v1.json",
    )
    digest = hashlib.sha256()
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(f"Definicao experimental ausente: {path}")
        digest.update(paths.project_relative(path).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_path(paths: CodecoPath) -> Path:
    return paths.cache / CACHE_FILE_NAME


def _cache_validation_error(table: pa.Table, manifest_sha256: str) -> str | None:
    if not table.schema.equals(CLIP_TABLE_SCHEMA, check_metadata=False):
        return "schema incompatível"
    metadata = table.schema.metadata or {}
    if metadata.get(b"codeco_cache_kind") != CACHE_METADATA[b"codeco_cache_kind"]:
        return "tipo de cache ausente ou incompatível"
    if metadata.get(b"codeco_cache_schema_version") != CACHE_METADATA[
        b"codeco_cache_schema_version"
    ]:
        return "versão do cache ausente ou incompatível"
    cached_sha256 = metadata.get(b"codeco_source_manifest_sha256", b"").decode(
        "ascii", errors="replace"
    )
    if cached_sha256 != manifest_sha256:
        return "manifesto de origem alterado"
    return None


def _read_cache(
    cache_file: Path,
    manifest_sha256: str,
    paths: CodecoPath,
) -> tuple[pa.Table | None, str | None, float]:
    started = time.perf_counter()
    if not cache_file.is_file():
        return None, "arquivo ausente", time.perf_counter() - started
    try:
        table = parquet.read_table(cache_file)
    except Exception as error:
        return (
            None,
            f"arquivo ilegível ou corrompido ({type(error).__name__}: {error})",
            time.perf_counter() - started,
        )
    error = _cache_validation_error(table, manifest_sha256)
    if error is not None:
        return None, error, time.perf_counter() - started
    table = table.replace_schema_metadata(None)
    path_index = table.schema.get_field_index("path")
    table = table.set_column(
        path_index,
        CLIP_TABLE_SCHEMA.field("path"),
        pa.array(
            [str(paths.project_path(value)) for value in table["path"].to_pylist()],
            type=pa.string(),
        ),
    )
    return table, None, time.perf_counter() - started


def _write_cache(
    table: pa.Table,
    cache_file: Path,
    manifest_sha256: str,
    paths: CodecoPath,
) -> tuple[float, int]:
    started = time.perf_counter()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **CACHE_METADATA,
        b"codeco_source_manifest_sha256": manifest_sha256.encode("ascii"),
    }
    path_index = table.schema.get_field_index("path")
    persisted = table.set_column(
        path_index,
        CLIP_TABLE_SCHEMA.field("path"),
        pa.array(
            [paths.project_relative(value) for value in table["path"].to_pylist()],
            type=pa.string(),
        ),
    )
    cached_table = persisted.replace_schema_metadata(metadata)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_file.parent,
            prefix=f".{cache_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        parquet.write_table(cached_table, temporary_path, compression="zstd")
        os.replace(temporary_path, cache_file)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return time.perf_counter() - started, cache_file.stat().st_size


def _pcm_dataset_file_name(dataset: str) -> str:
    """Return a stable and filesystem-safe PCM cache name for one dataset."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset).strip("._")
    if not stem:
        raise ValueError(f"Nome de dataset invalido para cache PCM: {dataset!r}")
    return f"{stem}.npy"


def _pcm_records(table: pa.Table) -> list[dict[str, object]]:
    """Return available canonical windows in their stable table order."""
    return [
        {
            "dataset": str(row["dataset"]),
            "clip_id": str(row["clip_id"]),
            "path": str(row["path"]),
            "start": float(row["start"]),
        }
        for row in table.to_pylist()
        if bool(row["audio_available"])
    ]


def _pcm_table_sha256(
    records: list[dict[str, object]],
    paths: CodecoPath,
) -> str:
    digest = hashlib.sha256()
    for row in records:
        persisted = {**row, "path": paths.project_relative(str(row["path"]))}
        digest.update(
            json.dumps(
                persisted,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _pcm_expected_index(records: list[dict[str, object]]) -> pa.Table:
    positions: dict[str, int] = {}
    index_records = []
    file_names: dict[str, str] = {}
    for row in records:
        dataset = str(row["dataset"])
        file_name = _pcm_dataset_file_name(dataset)
        previous_dataset = file_names.get(file_name)
        if previous_dataset is not None and previous_dataset != dataset:
            raise ValueError(
                "Datasets geram o mesmo nome de cache PCM: "
                f"{previous_dataset!r} e {dataset!r}."
            )
        file_names[file_name] = dataset
        position = positions.get(dataset, 0)
        index_records.append(
            {
                "dataset": dataset,
                "clip_id": str(row["clip_id"]),
                "pcm_file": file_name,
                "pcm_row": position,
            }
        )
        positions[dataset] = position + 1
    return pa.Table.from_pylist(index_records, schema=PCM_INDEX_SCHEMA)


def _pcm_expected_metadata(
    records: list[dict[str, object]],
    manifest_sha256: str,
    paths: CodecoPath,
) -> dict[str, object]:
    counts: dict[str, int] = {}
    for row in records:
        dataset = str(row["dataset"])
        counts[dataset] = counts.get(dataset, 0) + 1
    return {
        "cache_kind": "baseline_pcm",
        "schema_version": PCM_CACHE_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "table_sha256": _pcm_table_sha256(records, paths),
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "bit_depth": BIT_DEPTH,
        "channels": 1,
        "dtype": "int16",
        "samples_per_clip": PCM_SAMPLES,
        "datasets": counts,
    }


def _read_pcm_cache(
    cache_dir: Path,
    expected_index: pa.Table,
    expected_metadata: dict[str, object],
) -> tuple[pa.Table | None, str | None, float]:
    started = time.perf_counter()
    metadata_file = cache_dir / PCM_METADATA_FILE_NAME
    index_file = cache_dir / PCM_INDEX_FILE_NAME
    if not metadata_file.is_file() or not index_file.is_file():
        return None, "indice ou metadados ausentes", time.perf_counter() - started
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata != expected_metadata:
            return None, "metadados incompatíveis", time.perf_counter() - started
        index = parquet.read_table(index_file)
        if not index.schema.equals(PCM_INDEX_SCHEMA, check_metadata=False):
            return None, "schema do indice incompatível", time.perf_counter() - started
        index = index.replace_schema_metadata(None)
        if not index.equals(expected_index):
            return None, "indice incompatível", time.perf_counter() - started
        for dataset, count in expected_metadata["datasets"].items():
            pcm_file = cache_dir / _pcm_dataset_file_name(str(dataset))
            mapped = np.load(pcm_file, mmap_mode="r", allow_pickle=False)
            if mapped.dtype != np.dtype(np.int16):
                return None, f"dtype incompatível em {pcm_file}", time.perf_counter() - started
            if mapped.shape != (int(count), PCM_SAMPLES):
                return None, f"shape incompatível em {pcm_file}", time.perf_counter() - started
            del mapped
    except Exception as error:
        return (
            None,
            f"cache ilegível ou corrompido ({type(error).__name__}: {error})",
            time.perf_counter() - started,
        )
    return index, None, time.perf_counter() - started


def baseline_pcm_cache_status(
    paths: CodecoPath | None = None,
) -> tuple[bool, str]:
    """Check whether the materialized baseline can be reused without rebuilding."""
    locations = paths or CodecoPath.discover()
    cache_dir = locations.baseline_pcm
    metadata_file = cache_dir / PCM_METADATA_FILE_NAME
    index_file = cache_dir / PCM_INDEX_FILE_NAME
    if not metadata_file.is_file() or not index_file.is_file():
        return False, "indice ou metadados ausentes"
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        expected_protocol = {
            "cache_kind": "baseline_pcm",
            "schema_version": PCM_CACHE_SCHEMA_VERSION,
            "manifest_sha256": _source_definition_sha256(locations),
            "sample_rate": SAMPLE_RATE,
            "clip_seconds": CLIP_SECONDS,
            "bit_depth": BIT_DEPTH,
            "channels": 1,
            "dtype": "int16",
            "samples_per_clip": PCM_SAMPLES,
        }
        for key, expected in expected_protocol.items():
            if metadata.get(key) != expected:
                return False, f"metadado incompatível: {key}"
        datasets = metadata.get("datasets")
        if not isinstance(datasets, dict) or not datasets:
            return False, "contagens por dataset ausentes"
        index = parquet.read_table(index_file)
        if not index.schema.equals(PCM_INDEX_SCHEMA, check_metadata=False):
            return False, "schema do indice incompatível"
        expected_rows = sum(int(count) for count in datasets.values())
        if index.num_rows != expected_rows:
            return False, "quantidade de linhas do indice incompatível"
        for dataset, count in datasets.items():
            pcm_file = cache_dir / _pcm_dataset_file_name(str(dataset))
            mapped = np.load(pcm_file, mmap_mode="r", allow_pickle=False)
            valid = (
                mapped.dtype == np.dtype(np.int16)
                and mapped.shape == (int(count), PCM_SAMPLES)
            )
            del mapped
            if not valid:
                return False, f"arquivo PCM incompatível: {pcm_file.name}"
    except Exception as error:
        return False, f"cache ilegível ou corrompido ({type(error).__name__}: {error})"
    return True, f"{expected_rows} clipes PCM validados"


def _prepare_pcm_record(record: dict[str, object]) -> np.ndarray:
    """Load one window and reproduce the canonical PCM16 conversion."""
    import soundfile as sf

    audio_path = Path(str(record["path"]))
    try:
        info = sf.info(str(audio_path))
        start_frame = round(float(record["start"]) * info.samplerate)
        frames = round(CLIP_SECONDS * info.samplerate)
        audio, _ = sf.read(
            audio_path,
            start=start_frame,
            frames=frames,
            dtype="float32",
            always_2d=False,
        )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if info.samplerate != SAMPLE_RATE:
            import soxr

            audio = soxr.resample(audio, info.samplerate, SAMPLE_RATE, quality="HQ")
        if len(audio) != PCM_SAMPLES:
            raise ValueError(
                f"esperadas {PCM_SAMPLES} amostras; recebidas {len(audio)}"
            )
        return np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    except Exception as error:
        raise RuntimeError(
            f"Falha ao preparar PCM de {record['dataset']}/{record['clip_id']} "
            f"em {audio_path}: {error}"
        ) from error


def _atomic_write_pcm_index(table: pa.Table, destination: Path) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        parquet.write_table(table, temporary_path, compression="zstd")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_pcm_metadata(metadata: dict[str, object], destination: Path) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(metadata, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _build_pcm_cache(
    records: list[dict[str, object]],
    cache_dir: Path,
    index: pa.Table,
    metadata: dict[str, object],
) -> tuple[pa.Table, float, int, int]:
    started = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, psutil.cpu_count(logical=False) or 1)
    records_by_dataset: dict[str, list[dict[str, object]]] = {}
    for record in records:
        records_by_dataset.setdefault(str(record["dataset"]), []).append(record)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for dataset, dataset_records in records_by_dataset.items():
            destination = cache_dir / _pcm_dataset_file_name(dataset)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=cache_dir,
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                mapped = np.lib.format.open_memmap(
                    temporary_path,
                    mode="w+",
                    dtype=np.int16,
                    shape=(len(dataset_records), PCM_SAMPLES),
                )
                progress = tqdm(
                    total=len(dataset_records),
                    desc=f"Baseline PCM: {dataset}",
                    unit="clipe",
                    dynamic_ncols=True,
                    file=sys.stdout,
                )
                try:
                    for start in range(0, len(dataset_records), PCM_CACHE_BATCH_SIZE):
                        batch = dataset_records[start : start + PCM_CACHE_BATCH_SIZE]
                        for offset, pcm in enumerate(executor.map(_prepare_pcm_record, batch)):
                            mapped[start + offset] = pcm
                            progress.update(1)
                        mapped.flush()
                finally:
                    progress.close()
                    del mapped
                os.replace(temporary_path, destination)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    _atomic_write_pcm_index(index, cache_dir / PCM_INDEX_FILE_NAME)
    _atomic_write_pcm_metadata(metadata, cache_dir / PCM_METADATA_FILE_NAME)
    cache_bytes = sum(
        path.stat().st_size
        for path in cache_dir.iterdir()
        if path.is_file()
    )
    return index, time.perf_counter() - started, cache_bytes, workers


def _prepare_pcm_cache(
    table: pa.Table,
    paths: CodecoPath,
    manifest_sha256: str,
) -> tuple[Path, pa.Table]:
    records = _pcm_records(table)
    if not records:
        raise ValueError("Nenhum audio disponivel para construir a baseline PCM.")
    index = _pcm_expected_index(records)
    metadata = _pcm_expected_metadata(records, manifest_sha256, paths)
    cache_dir = paths.baseline_pcm

    _section("Cache PCM da baseline")
    LOGGER.info("Cache: %s", cache_dir)
    cached_index, cache_error, read_seconds = _read_pcm_cache(
        cache_dir,
        index,
        metadata,
    )
    if cached_index is not None:
        LOGGER.info(
            "Cache hit: %d clipes PCM validados em %.3f s.",
            cached_index.num_rows,
            read_seconds,
        )
        return cache_dir, cached_index

    log = LOGGER.info if cache_error == "indice ou metadados ausentes" else LOGGER.warning
    log("Cache miss apos %.3f s: %s.", read_seconds, cache_error)
    LOGGER.info(
        "Construindo %d clipes em PCM16 | batch=%d | nucleos fisicos=%d.",
        len(records),
        PCM_CACHE_BATCH_SIZE,
        max(1, psutil.cpu_count(logical=False) or 1),
    )
    built_index, elapsed, cache_bytes, workers = _build_pcm_cache(
        records,
        cache_dir,
        index,
        metadata,
    )
    LOGGER.info(
        "Cache PCM concluido: %d clipes; %.2f GiB; %.2f s; %d processos.",
        built_index.num_rows,
        cache_bytes / 1024**3,
        elapsed,
        workers,
    )
    return cache_dir, built_index


def _refresh_audio_available(table: pa.Table) -> tuple[pa.Table, int]:
    previous = table["audio_available"].to_pylist()
    current = [Path(path).is_file() for path in table["path"].to_pylist()]
    changed = sum(before != after for before, after in zip(previous, current))
    if not changed:
        return table, 0
    column_index = table.schema.get_field_index("audio_available")
    refreshed = table.set_column(
        column_index,
        CLIP_TABLE_SCHEMA.field("audio_available"),
        pa.array(current, type=pa.bool_()),
    )
    return refreshed, changed


def _read_windows(paths: CodecoPath) -> pa.Table:
    """Rebuild the auditable selected slice from authoritative CSV freezes."""
    records: list[dict[str, object]] = []
    selection = experiment_config()["class_selection"]
    rule = SelectionRule(
        k_sigma=float(selection["k_sigma"]),
        minimum_groups=int(selection["minimum_groups"]),
        seed=int(selection["seed"]),
        fit_split=str(selection["fit_split"]),
    )
    for corpus in CORPUS_MANIFESTS:
        frozen = load_frozen_manifest(corpus, paths.root)
        _, metadata = select_classes(frozen, rule)
        frame, _ = deterministic_balanced_training_sample(
            frozen,
            metadata["selected_classes"],
            quota=int(metadata["count_cutoff"]),
            seed=rule.seed,
        )
        for row in frame.to_dict(orient="records"):
            dataset = corpus
            start = (
                float(row.get("window_start_s", 0.0) or 0.0)
                if dataset == "pteroset_ppa1"
                else 0.0
            )
            records.append(
                {
                    "dataset": dataset,
                    "resource_type": "window",
                    "window_id": str(row["window_id"]),
                    "source_file_name": str(row["source_file_name"]),
                    "source_audio_relative_path": str(
                        row.get("source_audio_relative_path", "") or ""
                    ),
                    "window_start_s": start,
                    "source_split": str(row["split"]),
                    "target_labels": str(row.get("labels", "") or ""),
                    "source_group_id": str(row["source_group_id"]),
                }
            )
    if not records:
        raise ValueError("Os manifests congelados nao contem janelas liberadas.")
    return pa.Table.from_pylist(records)


def _dataset_names(windows: pa.Table) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in windows["dataset"].to_pylist()))


def _normalize_datasets(
    datasets: Iterable[str] | None,
    available: tuple[str, ...],
) -> tuple[str, ...]:
    if datasets is None:
        return available
    selected = tuple(dict.fromkeys(datasets))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(
            f"Datasets desconhecidos: {', '.join(unknown)}. "
            f"Opcoes: {', '.join(available)}"
        )
    if not selected:
        raise ValueError("Selecione pelo menos um dataset.")
    return selected


def _select_datasets(table: pa.Table, datasets: tuple[str, ...]) -> pa.Table:
    available = _dataset_names(table)
    if datasets == available:
        return table
    return pa.concat_tables(
        [table.filter(compute.equal(table["dataset"], dataset)) for dataset in datasets]
    )


def _log_table_counts(table: pa.Table) -> None:
    for dataset in _dataset_names(table):
        selected = table.filter(compute.equal(table["dataset"], dataset))
        available_audio = compute.sum(selected["audio_available"]).as_py()
        LOGGER.info(
            "%s: %d janelas; %d audios disponiveis.",
            dataset,
            selected.num_rows,
            available_audio,
        )


def _audio_location(paths: CodecoPath, row: dict) -> tuple[Path, float]:
    relative = str(row.get("source_audio_relative_path") or "")
    if relative:
        parts = Path(relative).parts
        if parts and parts[0] == "audio":
            parts = parts[1:]
        audio = paths.datasets.joinpath(*parts)
    elif str(row["dataset"]) == "pteroset_ppa1":
        filename = str(row["source_file_name"])
        candidates = (
            paths.datasets / "pteroset_ppa1" / filename,
            paths.datasets / "pteroset" / "PPA1" / filename,
            paths.datasets / "pteroset" / filename,
        )
        audio = next((path for path in candidates if path.is_file()), candidates[0])
    else:
        audio = paths.datasets / str(row["dataset"]) / str(row["source_file_name"])
    start = float(row.get("window_start_s") or 0.0)
    return audio, start


def _dataset_clip_table(
    paths: CodecoPath,
    windows: pa.Table,
    dataset: str,
) -> pa.Table:
    selected = windows.filter(compute.equal(windows["dataset"], dataset))
    records = []
    for row in tqdm(
        selected.to_pylist(),
        desc=f"Dataset: {dataset}",
        unit="janela",
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        audio, start = _audio_location(paths, row)
        records.append(
            {
                "dataset": dataset,
                "clip_id": str(row["window_id"]),
                "path": str(audio),
                "audio_available": audio.is_file(),
                "start": start,
                "end": start + WINDOW_SECONDS,
                "labels": str(row.get("target_labels") or ""),
                "split": str(row["source_split"]),
                "source_group_id": str(row["source_group_id"]),
            }
        )
    return pa.Table.from_pylist(records, schema=CLIP_TABLE_SCHEMA)


def _prepare_table(
    datasets: Iterable[str] | None = None,
) -> pa.Table:
    """Return one in-memory clip table for the selected frozen datasets."""
    paths = CodecoPath.discover()
    log_file, logs_created = _configure_logging(paths)
    _pipeline_header("PREPARE BASELINE")

    _section("Registro da execucao")
    LOGGER.info(
        "%s: %s",
        "Diretorio de logs criado" if logs_created else "Diretorio de logs encontrado",
        paths.logs,
    )
    LOGGER.info("Log da tabela de clipes: %s", log_file)

    _section("Cache da tabela de clipes")
    cache_file = _cache_path(paths)
    LOGGER.info("Cache: %s", cache_file)
    hash_started = time.perf_counter()
    manifest_sha256 = _source_definition_sha256(paths)
    LOGGER.info(
        "SHA-256 do manifesto: %s (%.3f s).",
        manifest_sha256,
        time.perf_counter() - hash_started,
    )
    clip_table, cache_error, read_seconds = _read_cache(
        cache_file,
        manifest_sha256,
        paths,
    )

    if clip_table is not None:
        LOGGER.info(
            "Cache hit: %d registros lidos em %.3f s.",
            clip_table.num_rows,
            read_seconds,
        )
        refresh_started = time.perf_counter()
        clip_table, changed_availability = _refresh_audio_available(clip_table)
        LOGGER.info(
            "Disponibilidade dos audios conferida em %.3f s; %d alteracoes.",
            time.perf_counter() - refresh_started,
            changed_availability,
        )
        if changed_availability:
            write_seconds, cache_bytes = _write_cache(
                clip_table,
                cache_file,
                manifest_sha256,
                paths,
            )
            LOGGER.info(
                "Cache atualizado atomicamente: %.2f MiB em %.3f s.",
                cache_bytes / 1024**2,
                write_seconds,
            )
    else:
        log = LOGGER.info if cache_error == "arquivo ausente" else LOGGER.warning
        log("Cache miss apos %.3f s: %s.", read_seconds, cache_error)

        _section("Manifesto congelado")
        LOGGER.info("Lendo janelas de: %s", paths.manifest)
        build_started = time.perf_counter()
        windows = _read_windows(paths)
        available = _dataset_names(windows)
        LOGGER.info(
            "%d janelas encontradas para: %s",
            windows.num_rows,
            ", ".join(available),
        )

        _section("Tabela de clipes em memoria")
        tables = []
        for dataset_index, dataset in enumerate(available):
            if dataset_index:
                print(file=sys.stdout)
            tables.append(_dataset_clip_table(paths, windows, dataset))
        clip_table = pa.concat_tables(tables)
        LOGGER.info(
            "Tabela completa reconstruida em %.3f s.",
            time.perf_counter() - build_started,
        )
        write_seconds, cache_bytes = _write_cache(
            clip_table,
            cache_file,
            manifest_sha256,
            paths,
        )
        LOGGER.info(
            "Cache gravado atomicamente: %s; %.2f MiB em %.3f s.",
            cache_file,
            cache_bytes / 1024**2,
            write_seconds,
        )

    available = _dataset_names(clip_table)
    selected = _normalize_datasets(datasets, available)
    _log_table_counts(clip_table)
    result = _select_datasets(clip_table, selected)
    _section("Tabela preparada")
    LOGGER.info("Datasets selecionados: %s.", ", ".join(selected))
    LOGGER.info(
        "Tabela em memoria: %d clipes; %.2f MiB.",
        result.num_rows,
        result.nbytes / 1024**2,
    )
    return result


def prepare_baseline() -> BaselineState:
    """Prepare the canonical baseline once for the integrated pipeline."""
    paths = CodecoPath.discover()
    table = _prepare_table()
    manifest_sha256 = _source_definition_sha256(paths)
    pcm_cache_dir, pcm_index = _prepare_pcm_cache(
        table,
        paths,
        manifest_sha256,
    )
    return BaselineState(
        table=table,
        cache_file=_cache_path(paths),
        manifest_sha256=manifest_sha256,
        sample_rate=SAMPLE_RATE,
        clip_seconds=CLIP_SECONDS,
        bit_depth=BIT_DEPTH,
        pcm_cache_dir=pcm_cache_dir,
        pcm_index=pcm_index,
    )


def load_standalone_examples() -> BaselineState:
    """Load the fixed, balanced examples without executing earlier stages."""
    paths = CodecoPath.discover()
    source = paths.standalone_examples
    if not source.is_file():
        raise FileNotFoundError(f"Exemplos autonomos nao encontrados: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Exemplos autonomos invalidos em {source}: {error}") from error

    protocol = payload.get("protocol", {})
    expected_protocol = {
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "bit_depth": BIT_DEPTH,
    }
    if protocol != expected_protocol:
        raise ValueError(
            f"Protocolo invalido em {source}: {protocol}; esperado {expected_protocol}."
        )

    selection = payload.get("selection")
    examples = payload.get("examples")
    if not isinstance(selection, dict) or not isinstance(examples, list):
        raise ValueError(f"Selecao ou exemplos ausentes em {source}.")

    required = {
        "dataset",
        "clip_id",
        "relative_path",
        "start",
        "end",
        "labels",
        "split",
        "source_group_id",
    }
    records = []
    observed_keys: set[tuple[str, str]] = set()
    observed_counts: dict[tuple[str, str, str], int] = {}
    missing_files = []
    for number, example in enumerate(examples, 1):
        if not isinstance(example, dict) or not required.issubset(example):
            missing = sorted(required - set(example if isinstance(example, dict) else ()))
            raise ValueError(
                f"Exemplo {number} invalido em {source}; campos ausentes: {missing}."
            )
        dataset = str(example["dataset"])
        clip_id = str(example["clip_id"])
        split = str(example["split"])
        labels = str(example["labels"])
        key = (dataset, clip_id)
        if key in observed_keys:
            raise ValueError(f"Exemplo duplicado em {source}: {dataset}/{clip_id}.")
        observed_keys.add(key)

        relative = Path(str(example["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Caminho relativo invalido no exemplo {number}: {relative}")
        audio = paths.datasets / relative
        if not audio.is_file():
            missing_files.append(audio)

        start = float(example["start"])
        end = float(example["end"])
        if abs((end - start) - CLIP_SECONDS) > 1e-9:
            raise ValueError(
                f"Duracao invalida no exemplo {dataset}/{clip_id}: {end - start}."
            )
        observed_counts[(dataset, split, labels)] = (
            observed_counts.get((dataset, split, labels), 0) + 1
        )
        records.append(
            {
                "dataset": dataset,
                "clip_id": clip_id,
                "path": str(audio),
                "audio_available": audio.is_file(),
                "start": start,
                "end": end,
                "labels": labels,
                "split": split,
                "source_group_id": str(example["source_group_id"]),
            }
        )

    for dataset, specification in selection.items():
        classes = specification.get("classes", [])
        per_class = specification.get("per_class", {})
        for split, expected in per_class.items():
            for label in classes:
                observed = observed_counts.get((dataset, split, label), 0)
                if observed != int(expected):
                    raise ValueError(
                        f"Amostra desbalanceada em {source}: {dataset}/{split}/{label} "
                        f"possui {observed}; esperado {expected}."
                    )
    expected_total = sum(
        int(count) * len(specification.get("classes", []))
        for specification in selection.values()
        for count in specification.get("per_class", {}).values()
    )
    if len(records) != expected_total:
        raise ValueError(
            f"Quantidade invalida de exemplos em {source}: {len(records)}; "
            f"esperado {expected_total}."
        )
    if missing_files:
        preview = "\n".join(str(path) for path in missing_files[:10])
        raise FileNotFoundError(
            f"{len(missing_files)} audios dos exemplos autonomos estao ausentes.\n{preview}"
        )

    table = pa.Table.from_pylist(records, schema=CLIP_TABLE_SCHEMA)
    return BaselineState(
        table=table,
        cache_file=source,
        manifest_sha256=_manifest_sha256(source),
        sample_rate=SAMPLE_RATE,
        clip_seconds=CLIP_SECONDS,
        bit_depth=BIT_DEPTH,
    )


def prepare_table(
    datasets: Iterable[str] | None = None,
) -> pa.Table:
    """Compatibility entry point for stages not yet integrated into the pipeline."""
    return _prepare_table(datasets)


if __name__ == "__main__":
    try:
        result = prepare_baseline()
        print(result.table)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception:
        LOGGER.exception("Falha ao preparar a tabela de clipes.")
        raise SystemExit(1) from None
