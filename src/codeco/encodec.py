"""Multi-GPU, resumable EnCodec reconstruction and metric evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing as multiprocessing_module
import os
import queue
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import psutil
import soundfile as sf
from scipy import signal
from tqdm import tqdm

from codeco.acoustic_indices import VERSION as ACOUSTIC_INDICES_VERSION
from codeco.acoustic_indices import acoustic_indices_v1
from codeco.codeco_path import CodecoPath
from codeco.environment import EnvironmentState, configure_model_environment
from codeco.prepare_table import (
    BIT_DEPTH,
    CLIP_SECONDS,
    SAMPLE_RATE,
    BaselineState,
    TqdmLoggingHandler,
    load_pcm_clip,
    load_standalone_examples,
)


PHASE = "encodec_metrics"
IMPLEMENTATION_VERSION = "1"
MODEL_NAME = "facebook/encodec_48khz"
CLIP_SAMPLES = round(SAMPLE_RATE * CLIP_SECONDS)
MAX_LAG = 4096
CHECKPOINT_SIZE = 64
METRIC_QUEUE_SIZE = 64
MAX_ATTEMPTS = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_GPU_MEMORY_FRACTION = 0.90
DEFAULT_GPU_PREFETCH_BATCHES = 2
MAX_BATCH_GROWTH_FACTOR = 8
DEFAULT_MAX_CLIPS_PER_DATASET = 300
DEFAULT_SEED = 42
LOGGER = logging.getLogger("codeco.encodec")


@dataclass(frozen=True)
class EncodecCondition:
    codec: str
    setting: str
    bandwidth_kbps: float | None


@dataclass(frozen=True)
class EvaluationTask:
    ordinal: int
    dataset: str
    clip_id: str
    path: str
    start: float
    codec: str
    setting: str
    bandwidth_kbps: float | None
    pcm_file: str | None = None
    pcm_row: int | None = None

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.dataset, self.clip_id, self.codec, self.setting


@dataclass(frozen=True)
class MetricWork:
    task: EvaluationTask
    reference: np.ndarray | None = None
    decoded: np.ndarray | None = None
    nbytes: int | None = None
    gpu_attempts: int = 1
    spool_path: str | None = None
    spool_index: int | None = None


@dataclass(frozen=True)
class MetricOutcome:
    task: EvaluationTask
    attempts: int
    result: dict[str, object] | None
    error: str | None
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class PipelineStats:
    model_startup_seconds: float
    processing_seconds: float
    metric_cpu_seconds: float
    gpu_rows: tuple[dict[str, object], ...]


CONDITIONS = (
    EncodecCondition("encodec", "bandwidth_kbps=6,model=encodec_48khz", 6.0),
    EncodecCondition("encodec", "bandwidth_kbps=12,model=encodec_48khz", 12.0),
    EncodecCondition("encodec", "bandwidth_kbps=24,model=encodec_48khz", 24.0),
)

RESULT_COLUMNS = (
    "dataset", "clip_id", "codec", "setting", "nbytes", "bitrate_kbps",
    "compression_ratio", "lag", "snr_db", "mel_mae_db", "mel_rmse_db",
    "mel_ssim", "mel_cosine", "idx_ACI", "idx_ADI", "idx_AEI", "idx_Ht",
    "idx_Hf", "idx_H", "idx_BIO_2_24", "idx_BIO_2_8", "idx_NDSI_CLASSIC",
    "idx_NDSI_EXTENDED",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _configure_logging(paths: CodecoPath, log_path: Path | None = None) -> Path:
    log_file = log_path or paths.logs / "encodec.log"
    if not LOGGER.handlers:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        terminal = TqdmLoggingHandler(sys.stdout)
        terminal.setFormatter(logging.Formatter("%(message)s"))
        persistent = logging.FileHandler(log_file, encoding="utf-8")
        persistent.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        LOGGER.addHandler(terminal)
        LOGGER.addHandler(persistent)
        LOGGER.setLevel(logging.INFO)
        LOGGER.propagate = False
    return log_file


def _header() -> None:
    line = "=" * 72
    command = subprocess.list2cmdline(
        [sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    )
    LOGGER.info(
        "\n%s\nCOMPRESSAO ENCODEC + RECONSTRUCAO + METRICAS\n[CMD]   %s\n%s",
        line,
        command,
        line,
    )


def _section(title: str) -> None:
    LOGGER.info("\n%s\n%s", title, "-" * 72)


def _log_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    values = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in values))
        for index, header in enumerate(headers)
    ]
    LOGGER.info(
        "  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers))
    )
    LOGGER.info("  ".join("-" * width for width in widths))
    for row in values:
        LOGGER.info("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _physical_cores() -> int:
    return max(1, psutil.cpu_count(logical=False) or 1)


def _load_clip(path: str, start: float) -> np.ndarray:
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio nao encontrado: {audio_path}")
    info = sf.info(str(audio_path))
    audio, _ = sf.read(
        audio_path,
        start=round(start * info.samplerate),
        frames=round(CLIP_SECONDS * info.samplerate),
        dtype="float32",
        always_2d=False,
    )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if info.samplerate != SAMPLE_RATE:
        import soxr

        audio = soxr.resample(audio, info.samplerate, SAMPLE_RATE, quality="HQ")
    if len(audio) != CLIP_SAMPLES:
        raise ValueError(
            f"Esperadas {CLIP_SAMPLES} amostras; recebidas {len(audio)} de {audio_path}."
        )
    audio = np.round(np.clip(audio, -1.0, 1.0) * 32767.0) / 32768.0
    return audio.astype(np.float32)


def _load_task_clip(task: EvaluationTask) -> np.ndarray:
    if task.pcm_file is None and task.pcm_row is None:
        return _load_clip(task.path, task.start)
    if task.pcm_file is None or task.pcm_row is None:
        raise RuntimeError(
            f"Referencia PCM incompleta para {task.dataset}/{task.clip_id}."
        )
    return load_pcm_clip(task.pcm_file, task.pcm_row)


def _align_to_reference(reference: np.ndarray, decoded: np.ndarray) -> tuple[np.ndarray, int]:
    length = len(reference)
    decoded = np.pad(decoded, (0, max(0, length + MAX_LAG - len(decoded))))
    segment_length = min(length, SAMPLE_RATE)
    correlation = signal.correlate(
        decoded[: segment_length + MAX_LAG],
        reference[:segment_length],
        mode="valid",
        method="fft",
    )
    lag = int(np.argmax(correlation))
    return decoded[lag : lag + length].astype(np.float32), lag


def _baseline_result(task: EvaluationTask, audio: np.ndarray) -> dict[str, object]:
    return {
        "dataset": task.dataset,
        "clip_id": task.clip_id,
        "codec": task.codec,
        "setting": task.setting,
        "nbytes": audio.size * BIT_DEPTH // 8,
        "bitrate_kbps": audio.size * BIT_DEPTH / CLIP_SECONDS / 1000,
        "compression_ratio": 1.0,
        "lag": 0,
        "snr_db": float("inf"),
        "mel_mae_db": 0.0,
        "mel_rmse_db": 0.0,
        "mel_ssim": 1.0,
        "mel_cosine": 1.0,
        **acoustic_indices_v1(audio),
    }


def _metric_worker(work: MetricWork) -> MetricOutcome:
    from codeco.metrics import reconstruction_metrics

    started = time.perf_counter()
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if work.task.codec == "baseline":
                reference = _load_task_clip(work.task)
                result = _baseline_result(work.task, reference)
            else:
                reference = work.reference
                decoded = work.decoded
                if work.spool_path is not None and work.spool_index is not None:
                    reference = _load_task_clip(work.task)
                    mapped = np.load(work.spool_path, mmap_mode="r")
                    decoded = np.array(mapped[work.spool_index], dtype=np.float32, copy=True)
                    del mapped
                if reference is None or decoded is None or work.nbytes is None:
                    raise RuntimeError("Reconstrucao ENCODEC incompleta para as metricas.")
                aligned, lag = _align_to_reference(reference, decoded)
                result = {
                    "dataset": work.task.dataset,
                    "clip_id": work.task.clip_id,
                    "codec": work.task.codec,
                    "setting": work.task.setting,
                    "nbytes": work.nbytes,
                    "bitrate_kbps": work.nbytes * 8 / CLIP_SECONDS / 1000,
                    "compression_ratio": (
                        reference.size * BIT_DEPTH // 8
                    ) / work.nbytes,
                    "lag": lag,
                    **reconstruction_metrics(reference, aligned),
                }
            return MetricOutcome(
                task=work.task,
                attempts=max(work.gpu_attempts, attempt),
                result=result,
                error=None,
                elapsed_seconds=time.perf_counter() - started,
            )
        except Exception as exc:
            last_error = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
    return MetricOutcome(
        task=work.task,
        attempts=max(work.gpu_attempts, MAX_ATTEMPTS),
        result=None,
        error=last_error,
        elapsed_seconds=time.perf_counter() - started,
    )


def _is_cuda_oom(torch_module: object, error: BaseException) -> bool:
    oom_type = getattr(getattr(torch_module, "cuda"), "OutOfMemoryError", ())
    return isinstance(error, oom_type) or "out of memory" in str(error).lower()


def _encodec_chunk(
    model: object,
    torch_module: object,
    device: object,
    tasks: list[EvaluationTask],
    codebook_bits: int,
    channels: int,
) -> tuple[list[tuple[EvaluationTask, np.ndarray, np.ndarray, int]], dict[str, float]]:
    references = [_load_task_clip(task) for task in tasks]
    batch = np.stack(references)
    input_values = torch_module.from_numpy(batch)[:, None, :]
    input_values = input_values.repeat(1, channels, 1).pin_memory()
    torch_module.cuda.reset_peak_memory_stats(device)
    base_allocated = torch_module.cuda.memory_allocated(device)
    free_before, total_memory = torch_module.cuda.mem_get_info(device)
    input_values = input_values.to(device, non_blocking=True)
    with torch_module.inference_mode():
        encoded = model.encode(
            input_values,
            bandwidth=float(tasks[0].bandwidth_kbps),
        )
        decoded = model.decode(
            encoded.audio_codes,
            encoded.audio_scales,
            None,
            last_frame_pad_length=encoded.last_frame_pad_length,
        ).audio_values
    decoded = decoded.mean(dim=1).float().cpu().numpy()
    codes = encoded.audio_codes
    scales = encoded.audio_scales
    items = []
    for index, task in enumerate(tasks):
        code_bits = int(codes[:, index].numel()) * codebook_bits
        scale_values = sum(
            int(scale[index].numel())
            for scale in scales
            if scale is not None
        )
        nbytes = int(math.ceil(code_bits / 8)) + 4 * scale_values
        items.append((task, references[index], decoded[index], nbytes))
    statistics = {
        "allocated_gib": torch_module.cuda.memory_allocated(device) / 1024**3,
        "reserved_gib": torch_module.cuda.memory_reserved(device) / 1024**3,
        "peak_gib": torch_module.cuda.max_memory_allocated(device) / 1024**3,
        "total_gib": total_memory / 1024**3,
        "base_bytes": float(base_allocated),
        "free_before_bytes": float(free_before),
        "peak_bytes": float(torch_module.cuda.max_memory_allocated(device)),
    }
    del decoded, encoded, codes, scales, input_values, batch
    return items, statistics


def _gpu_process_job(
    model: object,
    torch_module: object,
    device: object,
    tasks: list[EvaluationTask],
    requested_batch_size: int,
    memory_fraction: float,
    codebook_bits: int,
    channels: int,
) -> tuple[
    list[tuple[EvaluationTask, np.ndarray, int, int]],
    int,
    dict[str, object],
]:
    position = 0
    working_batch_size = min(requested_batch_size, len(tasks))
    output = []
    last_statistics: dict[str, object] = {}
    growth_statistics: dict[str, object] | None = None
    growth_chunk_size = 0
    saw_oom = False
    successful_chunk_sizes = []
    oom_events = 0
    retry_count = 0
    while position < len(tasks):
        chunk_size = min(working_batch_size, len(tasks) - position)
        failures = 0
        while True:
            chunk = tasks[position : position + chunk_size]
            try:
                items, last_statistics = _encodec_chunk(
                    model,
                    torch_module,
                    device,
                    chunk,
                    codebook_bits,
                    channels,
                )
                output.extend(
                    (task, decoded, nbytes, failures + 1)
                    for task, _, decoded, nbytes in items
                )
                successful_chunk_sizes.append(chunk_size)
                retry_count += failures
                if chunk_size >= requested_batch_size:
                    growth_statistics = last_statistics
                    growth_chunk_size = chunk_size
                position += chunk_size
                break
            except Exception as exc:
                failures += 1
                is_oom = _is_cuda_oom(torch_module, exc)
                if is_oom:
                    saw_oom = True
                    oom_events += 1
                    torch_module.cuda.empty_cache()
                    smaller = max(1, chunk_size // 2)
                    working_batch_size = smaller
                    chunk_size = smaller
                if failures >= MAX_ATTEMPTS:
                    raise RuntimeError(
                        f"Falha ENCODEC apos {MAX_ATTEMPTS} tentativas "
                        f"(batch={chunk_size}, OOM={'sim' if is_oom else 'nao'}): {exc}"
                    ) from exc

    next_batch_size = working_batch_size if saw_oom else requested_batch_size
    if not saw_oom and growth_statistics:
        base = float(growth_statistics["base_bytes"])
        peak = float(growth_statistics["peak_bytes"])
        incremental = max(0.0, peak - base)
        bytes_per_item = incremental / max(1, growth_chunk_size)
        usable_free = float(growth_statistics["free_before_bytes"]) * memory_fraction
        if bytes_per_item > 0:
            estimated = max(1, int(usable_free / bytes_per_item))
            estimated = max(8, estimated // 8 * 8)
            estimated = min(
                estimated,
                requested_batch_size * MAX_BATCH_GROWTH_FACTOR,
            )
            next_batch_size = max(requested_batch_size, estimated)
            last_statistics["estimated_safe_batch"] = next_batch_size
    last_statistics["saw_oom"] = float(saw_oom)
    last_statistics["max_attempts"] = float(
        max((item[-1] for item in output), default=1)
    )
    last_statistics["chunk_sizes"] = successful_chunk_sizes
    last_statistics["oom_events"] = oom_events
    last_statistics["retry_count"] = retry_count
    return output, next_batch_size, last_statistics


def _write_spool_batch(
    spool: Path,
    device_index: int,
    work_id: int,
    items: Sequence[tuple[EvaluationTask, np.ndarray, int, int]],
) -> tuple[Path, list[tuple[EvaluationTask, int, int, int]], int]:
    decoded_batch = np.stack([item[1] for item in items])
    spool_path = spool / f"gpu-{device_index}-work-{work_id}.npy"
    temporary_path = spool / f".{spool_path.name}.tmp"
    try:
        with temporary_path.open("wb") as stream:
            np.save(stream, decoded_batch, allow_pickle=False)
        os.replace(temporary_path, spool_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    descriptors = [
        (task, index, nbytes, attempts)
        for index, (task, _, nbytes, attempts) in enumerate(items)
    ]
    return spool_path, descriptors, spool_path.stat().st_size


def _gpu_worker(
    device_index: int,
    model_name: str,
    initial_batch_size: int,
    memory_fraction: float,
    spool_directory: str,
    input_queue: object,
    output_queue: object,
) -> None:
    startup_started = time.perf_counter()
    try:
        import torch
        from transformers import EncodecModel

        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        model = EncodecModel.from_pretrained(model_name).to(device).eval()
        codebook_bits = int(math.ceil(math.log2(model.config.codebook_size)))
        channels = int(model.config.audio_channels)
        properties = torch.cuda.get_device_properties(device)
        output_queue.put(
            {
                "kind": "ready",
                "device": device_index,
                "name": properties.name,
                "total_gib": properties.total_memory / 1024**3,
                "batch_size": initial_batch_size,
                "startup_seconds": time.perf_counter() - startup_started,
            }
        )
    except Exception as exc:
        output_queue.put(
            {
                "kind": "startup_error",
                "device": device_index,
                "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            }
        )
        return

    batch_size = initial_batch_size
    spool = Path(spool_directory)
    while True:
        message = input_queue.get()
        if message is None:
            return
        tasks = message["tasks"]
        work_id = message["work_id"]
        job_started = time.perf_counter()
        try:
            items, next_batch_size, statistics = _gpu_process_job(
                model,
                torch,
                device,
                tasks,
                batch_size,
                memory_fraction,
                codebook_bits,
                channels,
            )
            previous_batch_size = batch_size
            batch_size = next_batch_size
            spool_path, descriptors, written_bytes = _write_spool_batch(
                spool,
                device_index,
                work_id,
                items,
            )
            statistics["spool_bytes"] = written_bytes
            del items
            output_queue.put(
                {
                    "kind": "result",
                    "device": device_index,
                    "work_id": work_id,
                    "items": descriptors,
                    "spool_path": str(spool_path),
                    "batch_size": batch_size,
                    "previous_batch_size": previous_batch_size,
                    "statistics": statistics,
                    "elapsed_seconds": time.perf_counter() - job_started,
                }
            )
        except Exception as exc:
            output_queue.put(
                {
                    "kind": "work_error",
                    "device": device_index,
                    "work_id": work_id,
                    "tasks": tasks,
                    "error": "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                }
            )


def _select_clips(
    baseline: BaselineState,
    datasets: Sequence[str] | None,
    max_per_dataset: int | None,
    seed: int,
) -> pd.DataFrame:
    table = baseline.table.to_pandas()
    if datasets is not None:
        table = table[table["dataset"].isin(datasets)].copy()
    selected = table[table["audio_available"].astype(bool)].copy()
    if selected.empty:
        raise ValueError("Nenhum clipe disponivel.")
    groups = []
    for _, group in selected.groupby("dataset", sort=False):
        if max_per_dataset is not None and len(group) > max_per_dataset:
            group = group.sample(max_per_dataset, random_state=seed)
        groups.append(group.sort_values("clip_id"))
    result = pd.concat(groups, ignore_index=True)
    if baseline.pcm_index is None:
        result["pcm_file"] = None
        result["pcm_row"] = None
        return result
    if baseline.pcm_cache_dir is None:
        raise RuntimeError("Baseline possui indice PCM, mas nao possui diretorio do cache.")

    pcm_index = baseline.pcm_index.to_pandas()[
        ["dataset", "clip_id", "pcm_file", "pcm_row"]
    ].copy()
    if pcm_index.duplicated(["dataset", "clip_id"]).any():
        raise RuntimeError("Indice PCM possui chaves dataset/clip_id duplicadas.")
    pcm_index["pcm_file"] = pcm_index["pcm_file"].map(
        lambda value: str(baseline.pcm_cache_dir / str(value))
    )
    result = result.merge(
        pcm_index,
        on=["dataset", "clip_id"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    missing = result["pcm_file"].isna() | result["pcm_row"].isna()
    if missing.any():
        preview = ", ".join(
            f"{row.dataset}/{row.clip_id}"
            for row in result.loc[missing, ["dataset", "clip_id"]]
            .head(10)
            .itertuples(index=False)
        )
        raise RuntimeError(
            f"{int(missing.sum())} clipes selecionados nao constam no cache PCM: {preview}"
        )
    result["pcm_row"] = result["pcm_row"].astype(int)
    return result


def _build_tasks(clips: pd.DataFrame) -> list[EvaluationTask]:
    tasks = []
    ordinal = 0
    for dataset, dataset_clips in clips.groupby("dataset", sort=False):
        for condition in CONDITIONS:
            for row in dataset_clips.itertuples(index=False):
                tasks.append(
                    EvaluationTask(
                        ordinal=ordinal,
                        dataset=str(dataset),
                        clip_id=str(row.clip_id),
                        path=str(row.path),
                        start=float(row.start),
                        codec=condition.codec,
                        setting=condition.setting,
                        bandwidth_kbps=condition.bandwidth_kbps,
                        pcm_file=(
                            None if pd.isna(row.pcm_file) else str(row.pcm_file)
                        ),
                        pcm_row=(
                            None if pd.isna(row.pcm_row) else int(row.pcm_row)
                        ),
                    )
                )
                ordinal += 1
    return tasks


def _connect_checkpoint(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            phase TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL,
            total_tasks INTEGER NOT NULL,
            completed_tasks INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL,
            PRIMARY KEY (phase, fingerprint)
        );
        CREATE TABLE IF NOT EXISTS tasks (
            phase TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            dataset TEXT NOT NULL,
            clip_id TEXT NOT NULL,
            codec TEXT NOT NULL,
            setting TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            result_json TEXT,
            error TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (phase, fingerprint, dataset, clip_id, codec, setting),
            FOREIGN KEY (phase, fingerprint) REFERENCES runs (phase, fingerprint)
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_run_status
            ON tasks (phase, fingerprint, status, ordinal);
        """
    )
    return connection


def _prepare_checkpoint(
    connection: sqlite3.Connection,
    fingerprint: str,
    metadata: dict[str, object],
    tasks: Sequence[EvaluationTask],
) -> tuple[set[tuple[str, str, str, str]], dict[tuple[str, str, str, str], int]]:
    now = _utc_now()
    metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    with connection:
        connection.execute(
            """INSERT OR IGNORE INTO runs
               (phase, fingerprint, created_at, updated_at, status, total_tasks, completed_tasks, metadata_json)
               VALUES (?, ?, ?, ?, 'pending', ?, 0, ?)""",
            (PHASE, fingerprint, now, now, len(tasks), metadata_json),
        )
        observed = connection.execute(
            "SELECT total_tasks FROM runs WHERE phase = ? AND fingerprint = ?",
            (PHASE, fingerprint),
        ).fetchone()
        if observed is None or observed[0] != len(tasks):
            raise RuntimeError("Checkpoint incompativel com a lista de tarefas atual.")
        connection.executemany(
            """INSERT OR IGNORE INTO tasks
               (phase, fingerprint, ordinal, dataset, clip_id, codec, setting, status, attempts, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?)""",
            [(PHASE, fingerprint, task.ordinal, *task.key, now) for task in tasks],
        )
        connection.execute(
            """UPDATE tasks SET status = 'pending', error = NULL, attempts = 0, updated_at = ?
               WHERE phase = ? AND fingerprint = ? AND status = 'failed'""",
            (now, PHASE, fingerprint),
        )
        connection.execute(
            """UPDATE tasks SET status = 'pending', updated_at = ?
               WHERE phase = ? AND fingerprint = ? AND status = 'running'""",
            (now, PHASE, fingerprint),
        )
        connection.execute(
            """UPDATE runs SET status = 'running', updated_at = ?, metadata_json = ?
               WHERE phase = ? AND fingerprint = ?""",
            (now, metadata_json, PHASE, fingerprint),
        )
    rows = connection.execute(
        """SELECT dataset, clip_id, codec, setting, status, attempts
           FROM tasks WHERE phase = ? AND fingerprint = ?""",
        (PHASE, fingerprint),
    ).fetchall()
    completed = {tuple(row[:4]) for row in rows if row[4] == "completed"}
    attempts = {tuple(row[:4]): int(row[5]) for row in rows}
    return completed, attempts


def _clear_checkpoint_run(connection: sqlite3.Connection, fingerprint: str) -> tuple[int, int]:
    with connection:
        tasks_deleted = connection.execute(
            "DELETE FROM tasks WHERE phase = ? AND fingerprint = ?",
            (PHASE, fingerprint),
        ).rowcount
        runs_deleted = connection.execute(
            "DELETE FROM runs WHERE phase = ? AND fingerprint = ?",
            (PHASE, fingerprint),
        ).rowcount
    return tasks_deleted, runs_deleted


def _checkpoint_results(
    connection: sqlite3.Connection,
    fingerprint: str,
    completed: Sequence[MetricOutcome],
) -> int:
    now = _utc_now()
    with connection:
        connection.executemany(
            """UPDATE tasks SET status = 'completed', attempts = ?, result_json = ?, error = NULL, updated_at = ?
               WHERE phase = ? AND fingerprint = ? AND dataset = ? AND clip_id = ? AND codec = ? AND setting = ?""",
            [
                (
                    outcome.attempts,
                    json.dumps(outcome.result, ensure_ascii=False, sort_keys=True),
                    now,
                    PHASE,
                    fingerprint,
                    *outcome.task.key,
                )
                for outcome in completed
            ],
        )
        count = connection.execute(
            """SELECT COUNT(*) FROM tasks
               WHERE phase = ? AND fingerprint = ? AND status = 'completed'""",
            (PHASE, fingerprint),
        ).fetchone()[0]
        connection.execute(
            """UPDATE runs SET completed_tasks = ?, updated_at = ?
               WHERE phase = ? AND fingerprint = ?""",
            (count, now, PHASE, fingerprint),
        )
    return int(count)


def _checkpoint_failure(
    connection: sqlite3.Connection,
    fingerprint: str,
    tasks: Sequence[EvaluationTask],
    error: str,
) -> None:
    now = _utc_now()
    with connection:
        connection.executemany(
            """UPDATE tasks SET status = 'failed', attempts = ?, error = ?, updated_at = ?
               WHERE phase = ? AND fingerprint = ? AND dataset = ? AND clip_id = ? AND codec = ? AND setting = ?""",
            [(MAX_ATTEMPTS, error, now, PHASE, fingerprint, *task.key) for task in tasks],
        )
        connection.execute(
            """UPDATE runs SET status = 'failed', updated_at = ?
               WHERE phase = ? AND fingerprint = ?""",
            (now, PHASE, fingerprint),
        )


def _export_csv(connection: sqlite3.Connection, fingerprint: str, output: Path) -> int:
    rows = connection.execute(
        """SELECT result_json FROM tasks
           WHERE phase = ? AND fingerprint = ? AND status = 'completed'
           ORDER BY ordinal""",
        (PHASE, fingerprint),
    ).fetchall()
    records = [json.loads(row[0]) for row in rows]
    table = pd.DataFrame.from_records(records, columns=RESULT_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        table.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return len(table)


def _resource_snapshot() -> tuple[float, float]:
    process = psutil.Process()
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return psutil.cpu_percent(interval=None), rss / 1024**3


def _clear_spool_directory(path: Path, remove_directory: bool = False) -> None:
    if not path.is_dir():
        return
    for candidate in path.iterdir():
        if candidate.is_file():
            candidate.unlink(missing_ok=True)
    if remove_directory:
        try:
            path.rmdir()
        except OSError:
            pass


def _cuda_device_count() -> int:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "Nenhuma GPU CUDA esta disponivel. A fase ENCODEC nao usa fallback para CPU."
        )
    return int(torch.cuda.device_count())


def _start_gpu_workers(
    gpu_count: int,
    model_name: str,
    batch_size: int,
    memory_fraction: float,
    prefetch_batches: int,
    spool_directory: Path,
) -> tuple[object, list[object], list[object], dict[int, dict[str, object]]]:
    context = multiprocessing_module.get_context("spawn")
    output_queue = context.Queue()
    input_queues = [
        context.Queue(maxsize=prefetch_batches) for _ in range(gpu_count)
    ]
    processes = [
        context.Process(
            target=_gpu_worker,
            args=(
                index,
                model_name,
                batch_size,
                memory_fraction,
                str(spool_directory),
                input_queues[index],
                output_queue,
            ),
            name=f"encodec-gpu-{index}",
        )
        for index in range(gpu_count)
    ]
    for process in processes:
        process.start()
    devices = {}
    try:
        while len(devices) < gpu_count:
            message = output_queue.get(timeout=600)
            if message["kind"] == "startup_error":
                raise RuntimeError(
                    f"Falha ao iniciar GPU {message['device']}: {message['error']}"
                )
            if message["kind"] == "ready":
                devices[int(message["device"])] = message
    except Exception:
        _stop_gpu_workers(input_queues, processes)
        raise
    return output_queue, input_queues, processes, devices


def _stop_gpu_workers(input_queues: Sequence[object], processes: Sequence[object]) -> None:
    for input_queue in input_queues:
        try:
            input_queue.put_nowait(None)
        except queue.Full:
            pass
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def _run_pipeline(
    connection: sqlite3.Connection,
    fingerprint: str,
    tasks: Sequence[EvaluationTask],
    completed_keys: set[tuple[str, str, str, str]],
    gpu_count: int,
    initial_batch_size: int,
    gpu_memory_fraction: float,
    gpu_prefetch_batches: int,
    spool_directory: Path,
    model_name: str,
    metric_workers: int,
) -> PipelineStats:
    pending = [task for task in tasks if task.key not in completed_keys]
    baseline_pending = deque(task for task in pending if task.bandwidth_kbps is None)
    gpu_pending = {
        bandwidth: deque(
            task for task in pending if task.bandwidth_kbps == bandwidth
        )
        for bandwidth in (6.0, 12.0, 24.0)
    }
    bandwidth_order = deque((6.0, 12.0, 24.0))
    encoded_pending: deque[MetricWork] = deque()
    metric_futures: dict[object, MetricWork] = {}
    checkpoint_buffer: list[MetricOutcome] = []
    gpu_statistics: dict[int, dict[str, object]] = {}
    gpu_batches = {index: initial_batch_size for index in range(gpu_count)}
    gpu_jobs: dict[int, tuple[int, list[EvaluationTask]]] = {}
    gpu_queued = {index: 0 for index in range(gpu_count)}
    work_id = 0
    total = len(tasks)
    metric_cpu_seconds = 0.0
    spool_remaining: dict[str, int] = {}
    spool_bytes: dict[str, int] = {}
    psutil.cpu_percent(interval=None)

    spool_directory.mkdir(parents=True, exist_ok=True)
    _clear_spool_directory(spool_directory)
    pending_gpu_tasks = sum(len(items) for items in gpu_pending.values())
    projected_spool_bytes = pending_gpu_tasks * CLIP_SAMPLES * np.dtype(np.float32).itemsize
    disk_free = shutil.disk_usage(spool_directory).free
    if projected_spool_bytes > disk_free * 0.90:
        raise RuntimeError(
            "Espaco insuficiente para o spool ENCODEC: "
            f"projecao={projected_spool_bytes / 1024**3:.2f} GiB; "
            f"livre={disk_free / 1024**3:.2f} GiB."
        )
    LOGGER.info(
        "Spool externo: %s; projecao maxima=%.2f GiB; disco livre=%.2f GiB.",
        spool_directory,
        projected_spool_bytes / 1024**3,
        disk_free / 1024**3,
    )

    model_startup_started = time.perf_counter()
    output_queue, input_queues, gpu_processes, devices = _start_gpu_workers(
        gpu_count,
        model_name,
        initial_batch_size,
        gpu_memory_fraction,
        gpu_prefetch_batches,
        spool_directory,
    )
    model_startup_seconds = time.perf_counter() - model_startup_started
    gpu_observation = {
        index: {
            "gpu": index,
            "model": devices[index]["name"],
            "total_gib": float(devices[index]["total_gib"]),
            "startup_seconds": float(devices[index]["startup_seconds"]),
            "tasks": 0,
            "jobs": 0,
            "chunk_sizes": [],
            "active_seconds": 0.0,
            "peak_gib": 0.0,
            "spool_produced_bytes": 0,
            "oom_events": 0,
            "retry_count": 0,
        }
        for index in devices
    }
    for index in sorted(devices):
        device = devices[index]
        LOGGER.info(
            "GPU %d: %s; %.2f GiB; batch inicial=%d.",
            index,
            device["name"],
            device["total_gib"],
            initial_batch_size,
        )

    executor = ProcessPoolExecutor(
        max_workers=metric_workers,
        mp_context=multiprocessing_module.get_context("spawn"),
    )
    processing_started = time.perf_counter()

    def next_gpu_tasks(size: int) -> list[EvaluationTask]:
        for _ in range(len(bandwidth_order)):
            bandwidth = bandwidth_order[0]
            bandwidth_order.rotate(-1)
            pending_tasks = gpu_pending[bandwidth]
            if pending_tasks:
                return [pending_tasks.popleft() for _ in range(min(size, len(pending_tasks)))]
        return []

    def fill_gpu_queue(index: int) -> bool:
        nonlocal work_id
        dispatched = False
        while gpu_queued[index] < gpu_prefetch_batches:
            batch = next_gpu_tasks(gpu_batches[index])
            if not batch:
                break
            work_id += 1
            gpu_jobs[work_id] = (index, batch)
            gpu_queued[index] += 1
            input_queues[index].put({"work_id": work_id, "tasks": batch})
            dispatched = True
        return dispatched

    def release_spool(work: MetricWork) -> None:
        if work.spool_path is None:
            return
        remaining = spool_remaining[work.spool_path] - 1
        if remaining > 0:
            spool_remaining[work.spool_path] = remaining
            return
        Path(work.spool_path).unlink(missing_ok=True)
        spool_remaining.pop(work.spool_path, None)
        spool_bytes.pop(work.spool_path, None)

    try:
        for index in range(gpu_count):
            fill_gpu_queue(index)

        with tqdm(
            total=total,
            initial=len(completed_keys),
            desc="ENCODEC + metricas",
            unit="tarefa",
            dynamic_ncols=True,
            file=sys.stdout,
        ) as progress:
            while True:
                activity = False
                for index, process in enumerate(gpu_processes):
                    if not process.is_alive() and gpu_queued[index] > 0:
                        active_tasks = [
                            task
                            for job_device, job_tasks in gpu_jobs.values()
                            if job_device == index
                            for task in job_tasks
                        ]
                        error = (
                            f"Processo da GPU {index} terminou inesperadamente "
                            f"com codigo {process.exitcode}."
                        )
                        if checkpoint_buffer:
                            _checkpoint_results(connection, fingerprint, checkpoint_buffer)
                            checkpoint_buffer.clear()
                        _checkpoint_failure(
                            connection,
                            fingerprint,
                            active_tasks,
                            error,
                        )
                        raise RuntimeError(error)
                while True:
                    try:
                        message = output_queue.get_nowait()
                    except queue.Empty:
                        break
                    activity = True
                    kind = message["kind"]
                    device_index = int(message["device"])
                    returned_work_id = int(message["work_id"])
                    job_device, active_tasks = gpu_jobs.pop(returned_work_id)
                    if job_device != device_index:
                        raise RuntimeError("Resultado ENCODEC retornou pela GPU incorreta.")
                    gpu_queued[device_index] -= 1
                    if kind == "work_error":
                        error = str(message["error"])
                        if checkpoint_buffer:
                            _checkpoint_results(connection, fingerprint, checkpoint_buffer)
                            checkpoint_buffer.clear()
                        _checkpoint_failure(connection, fingerprint, active_tasks, error)
                        raise RuntimeError(
                            f"GPU {device_index} falhou no ENCODEC: {error}"
                        )
                    if kind != "result":
                        continue
                    previous_batch = int(message["previous_batch_size"])
                    gpu_batches[device_index] = int(message["batch_size"])
                    gpu_statistics[device_index] = message["statistics"]
                    observation = gpu_observation[device_index]
                    observation["tasks"] = int(observation["tasks"]) + len(message["items"])
                    observation["jobs"] = int(observation["jobs"]) + 1
                    observation["active_seconds"] = float(observation["active_seconds"]) + float(
                        message["elapsed_seconds"]
                    )
                    observation["peak_gib"] = max(
                        float(observation["peak_gib"]),
                        float(message["statistics"].get("peak_gib", 0)),
                    )
                    observation["spool_produced_bytes"] = int(
                        observation["spool_produced_bytes"]
                    ) + int(message["statistics"].get("spool_bytes", 0))
                    observation["oom_events"] = int(observation["oom_events"]) + int(
                        message["statistics"].get("oom_events", 0)
                    )
                    observation["retry_count"] = int(observation["retry_count"]) + int(
                        message["statistics"].get("retry_count", 0)
                    )
                    observation["chunk_sizes"].extend(
                        int(value)
                        for value in message["statistics"].get("chunk_sizes", [])
                    )
                    if gpu_batches[device_index] != previous_batch:
                        reason = (
                            "apos OOM"
                            if message["statistics"].get("saw_oom", 0)
                            else "por memoria disponivel"
                        )
                        LOGGER.info(
                            "GPU %d: batch adaptado de %d para %d %s.",
                            device_index,
                            previous_batch,
                            gpu_batches[device_index],
                            reason,
                        )
                    if message["statistics"].get("max_attempts", 1) > 1:
                        LOGGER.warning(
                            "GPU %d: lote concluido apos %.0f tentativas.",
                            device_index,
                            message["statistics"]["max_attempts"],
                        )
                    spool_path = str(message["spool_path"])
                    spool_remaining[spool_path] = len(message["items"])
                    spool_bytes[spool_path] = int(
                        message["statistics"].get("spool_bytes", 0)
                    )
                    for task, spool_index, nbytes, attempts in message["items"]:
                        encoded_pending.append(
                            MetricWork(
                                task=task,
                                nbytes=nbytes,
                                gpu_attempts=attempts,
                                spool_path=spool_path,
                                spool_index=spool_index,
                            )
                        )
                    progress.set_postfix_str(
                        "batches="
                        + ",".join(
                            f"gpu{index}:{gpu_batches[index]}"
                            for index in range(gpu_count)
                        )
                        + f" recon={len(encoded_pending)} metricas={len(metric_futures)}"
                    )

                completed_futures = [future for future in metric_futures if future.done()]
                for future in completed_futures:
                    activity = True
                    work = metric_futures.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        outcome = MetricOutcome(
                            task=work.task,
                            attempts=MAX_ATTEMPTS,
                            result=None,
                            error="".join(
                                traceback.format_exception_only(type(exc), exc)
                            ).strip(),
                        )
                    release_spool(work)
                    if outcome.error is not None:
                        if checkpoint_buffer:
                            _checkpoint_results(connection, fingerprint, checkpoint_buffer)
                            checkpoint_buffer.clear()
                        _checkpoint_failure(
                            connection,
                            fingerprint,
                            [outcome.task],
                            outcome.error,
                        )
                        raise RuntimeError(
                            f"Metricas falharam para {outcome.task.key} apos "
                            f"{MAX_ATTEMPTS} tentativas: {outcome.error}"
                        )
                    checkpoint_buffer.append(outcome)
                    metric_cpu_seconds += outcome.elapsed_seconds
                    progress.update(1)

                while encoded_pending and len(metric_futures) < METRIC_QUEUE_SIZE:
                    work = encoded_pending.popleft()
                    metric_futures[executor.submit(_metric_worker, work)] = work
                    activity = True

                baseline_in_flight = sum(
                    work.task.codec == "baseline" for work in metric_futures.values()
                )
                while (
                    baseline_pending
                    and baseline_in_flight < metric_workers
                    and len(metric_futures) < METRIC_QUEUE_SIZE
                ):
                    work = MetricWork(baseline_pending.popleft())
                    metric_futures[executor.submit(_metric_worker, work)] = work
                    baseline_in_flight += 1
                    activity = True

                for index in range(gpu_count):
                    activity = fill_gpu_queue(index) or activity

                while len(checkpoint_buffer) >= CHECKPOINT_SIZE:
                    block = checkpoint_buffer[:CHECKPOINT_SIZE]
                    del checkpoint_buffer[:CHECKPOINT_SIZE]
                    completed_count = _checkpoint_results(
                        connection,
                        fingerprint,
                        block,
                    )
                    cpu_percent, rss_gib = _resource_snapshot()
                    gpu_text = "; ".join(
                        f"GPU {index}: batch={gpu_batches[index]}, "
                        f"pico={gpu_statistics.get(index, {}).get('peak_gib', 0):.2f} GiB"
                        for index in range(gpu_count)
                    )
                    elapsed = max(progress.format_dict.get("elapsed", 0.0), 1e-9)
                    rate = max((completed_count - len(completed_keys)) / elapsed, 1e-9)
                    eta = (total - completed_count) / rate
                    LOGGER.info(
                        "Checkpoint: %d/%d (%.1f%%); recon=%d; metricas=%d; "
                        "fila GPU=%d jobs; spool=%d (%.2f GiB); CPU=%.1f%%; "
                        "RSS agregado=%.2f GiB; "
                        "decorrido=%.1f s; ETA=%.1f min; %s.",
                        completed_count,
                        total,
                        completed_count * 100 / total,
                        len(encoded_pending),
                        len(metric_futures),
                        sum(gpu_queued.values()),
                        len(spool_remaining),
                        sum(spool_bytes.values()) / 1024**3,
                        cpu_percent,
                        rss_gib,
                        elapsed,
                        eta / 60,
                        gpu_text,
                    )

                gpu_work_left = any(gpu_pending.values()) or any(
                    gpu_queued[index] > 0 for index in range(gpu_count)
                )
                if (
                    not gpu_work_left
                    and not baseline_pending
                    and not encoded_pending
                    and not metric_futures
                ):
                    break
                if not activity:
                    time.sleep(0.02)

            if checkpoint_buffer:
                completed_count = _checkpoint_results(
                    connection,
                    fingerprint,
                    checkpoint_buffer,
                )
                LOGGER.info("Checkpoint final: %d/%d tarefas.", completed_count, total)
    finally:
        for future in metric_futures:
            future.cancel()
        _stop_gpu_workers(input_queues, gpu_processes)
        executor.shutdown(wait=True, cancel_futures=True)
        _clear_spool_directory(spool_directory, remove_directory=True)
    processing_seconds = time.perf_counter() - processing_started
    gpu_rows = []
    for index in sorted(gpu_observation):
        observation = gpu_observation[index]
        chunk_sizes = observation.pop("chunk_sizes")
        observation["batch_min"] = min(chunk_sizes) if chunk_sizes else 0
        observation["batch_mean"] = (
            sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0.0
        )
        observation["batch_max"] = max(chunk_sizes) if chunk_sizes else 0
        active_seconds = float(observation["active_seconds"])
        observation["tasks_per_active_second"] = (
            int(observation["tasks"]) / active_seconds if active_seconds else 0.0
        )
        observation["active_percent"] = (
            active_seconds * 100 / processing_seconds if processing_seconds else 0.0
        )
        observation["spool_produced_gib"] = (
            int(observation["spool_produced_bytes"]) / 1024**3
        )
        gpu_rows.append(observation)
    return PipelineStats(
        model_startup_seconds=model_startup_seconds,
        processing_seconds=processing_seconds,
        metric_cpu_seconds=metric_cpu_seconds,
        gpu_rows=tuple(gpu_rows),
    )


def _condition_label(task: EvaluationTask) -> str:
    if task.codec == "baseline":
        return "BASELINE"
    return f"ENCODEC {float(task.bandwidth_kbps):g} kbps"


def _log_final_observability(
    tasks: Sequence[EvaluationTask],
    initially_completed: set[tuple[str, str, str, str]],
    pipeline: PipelineStats,
    preparation_seconds: float,
    export_seconds: float,
    total_seconds: float,
    metric_workers: int,
) -> None:
    _section("Tempos da execucao")
    timing_rows = [
        ("Preparacao dos dados e checkpoint", f"{preparation_seconds:.2f} s"),
        ("Carregamento dos modelos", f"{pipeline.model_startup_seconds:.2f} s"),
        ("Processamento", f"{pipeline.processing_seconds:.2f} s"),
        ("Exportacao do CSV", f"{export_seconds:.2f} s"),
        ("Total", f"{total_seconds:.2f} s"),
    ]
    _log_table(("Etapa", "Tempo"), timing_rows)
    LOGGER.info(
        "CPU de metricas acumulada: %.2f s em %d processos.",
        pipeline.metric_cpu_seconds,
        metric_workers,
    )

    _section("Desempenho por GPU")
    if pipeline.gpu_rows:
        gpu_rows = []
        for row in pipeline.gpu_rows:
            gpu_rows.append(
                (
                    row["gpu"],
                    row["model"],
                    row["tasks"],
                    row["jobs"],
                    f"{row['batch_min']}/{row['batch_mean']:.1f}/{row['batch_max']}",
                    f"{row['startup_seconds']:.2f} s",
                    f"{row['active_seconds']:.2f} s",
                    f"{row['active_percent']:.1f}%",
                    f"{row['tasks_per_active_second']:.2f}",
                    f"{row['peak_gib']:.2f} GiB",
                    f"{row['spool_produced_gib']:.2f} GiB",
                    row["oom_events"],
                    row["retry_count"],
                )
            )
        _log_table(
            (
                "GPU", "Modelo", "Tarefas", "Lotes", "Batch min/med/max",
                "Carga", "Ativa", "Util. ativa", "Tarefas/s", "Pico VRAM",
                "Spool", "OOM", "Retries",
            ),
            gpu_rows,
        )
    else:
        LOGGER.info("Nenhum trabalho de GPU: todas as tarefas vieram do checkpoint.")

    totals: dict[str, int] = {}
    resumed: dict[str, int] = {}
    for task in tasks:
        label = _condition_label(task)
        totals[label] = totals.get(label, 0) + 1
        if task.key in initially_completed:
            resumed[label] = resumed.get(label, 0) + 1
    _section("Resumo por condicao")
    condition_rows = [
        (
            label,
            total,
            resumed.get(label, 0),
            total - resumed.get(label, 0),
            0,
        )
        for label, total in totals.items()
    ]
    _log_table(
        ("Condicao", "Total", "Retomadas", "Processadas", "Falhas"),
        condition_rows,
    )


def run(
    baseline: BaselineState,
    datasets: Sequence[str] | None = None,
    max_clips_per_dataset: int | None = DEFAULT_MAX_CLIPS_PER_DATASET,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    gpu_memory_fraction: float = DEFAULT_GPU_MEMORY_FRACTION,
    gpu_prefetch_batches: int = DEFAULT_GPU_PREFETCH_BATCHES,
    model_name: str = MODEL_NAME,
    checkpoint_path: Path | None = None,
    output_path: Path | None = None,
    log_path: Path | None = None,
    force: bool = False,
    environment: EnvironmentState | None = None,
) -> Path:
    total_started = time.perf_counter()
    paths = CodecoPath.discover()
    configure_model_environment(paths)
    checkpoint = checkpoint_path or paths.cache / "checkpoints.sqlite3"
    output = output_path or paths.tables / "encodec_metrics.csv"
    log_file = _configure_logging(paths, log_path)
    _header()

    _section("Preparacao da execucao")
    gpu_count = (
        _cuda_device_count()
        if environment is None
        else len(environment.system.gpu_devices)
    )
    metric_workers = _physical_cores()
    clips = _select_clips(
        baseline,
        datasets,
        max_clips_per_dataset,
        seed,
    )
    tasks = _build_tasks(clips)
    persisted_selection = clips[
        ["dataset", "clip_id", "path", "start", "pcm_file", "pcm_row"]
    ].copy()
    persisted_selection["path"] = persisted_selection["path"].map(
        paths.project_relative
    )
    persisted_selection["pcm_file"] = persisted_selection["pcm_file"].map(
        lambda value: (
            None
            if value is None or pd.isna(value)
            else paths.project_relative(value)
        )
    )
    selection_records = persisted_selection.to_dict("records")
    metadata = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "manifest_sha256": baseline.manifest_sha256,
        "selection_sha256": _sha256_json(selection_records),
        "datasets": list(dict.fromkeys(clips["dataset"].astype(str))),
        "max_clips_per_dataset": max_clips_per_dataset,
        "seed": seed,
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "protocol": {
            "sample_rate": SAMPLE_RATE,
            "clip_seconds": CLIP_SECONDS,
            "bit_depth": BIT_DEPTH,
            "max_lag": MAX_LAG,
        },
        "pcm_cache": {
            "enabled": baseline.pcm_index is not None,
            "index_sha256": _sha256_json(
                [
                    {
                        "dataset": str(row.dataset),
                        "clip_id": str(row.clip_id),
                        "pcm_file": Path(str(row.pcm_file)).name,
                        "pcm_row": int(row.pcm_row),
                    }
                    for row in clips.itertuples(index=False)
                    if row.pcm_file is not None and not pd.isna(row.pcm_file)
                ]
            ),
        },
        "model_name": model_name,
        "precision": "float32",
        "payload_formula": "ceil(codes_per_item*codebook_bits/8)+4*scales_per_item",
        "acoustic_indices_version": ACOUSTIC_INDICES_VERSION,
    }
    fingerprint = _sha256_json(metadata)
    spool_directory = paths.cache / "encodec_spool" / fingerprint
    counts = clips.groupby("dataset", sort=False).size().to_dict()
    LOGGER.info("Log: %s", log_file)
    LOGGER.info("Checkpoint: %s", checkpoint)
    LOGGER.info("Resultado CSV: %s", output)
    LOGGER.info("Modelo: %s; precisao: float32", model_name)
    LOGGER.info(
        "Entrada PCM: %s",
        baseline.pcm_cache_dir if baseline.pcm_index is not None else "carregamento autonomo",
    )
    LOGGER.info("Fingerprint: %s", fingerprint)
    LOGGER.info(
        "Clipes: %s",
        ", ".join(f"{name}={count}" for name, count in counts.items()),
    )
    LOGGER.info("Condicoes: 3 ENCODEC; tarefas: %d", len(tasks))
    LOGGER.info(
        "GPUs CUDA: %d; um processo/modelo persistente por GPU; batch inicial: %d",
        gpu_count,
        batch_size,
    )
    LOGGER.info(
        "Uso alvo da VRAM: %.0f%%; prefetch: %d batches por GPU.",
        gpu_memory_fraction * 100,
        gpu_prefetch_batches,
    )
    LOGGER.info(
        "Processos CPU para metricas: %d; fila de metricas: %d; checkpoint: %d; tentativas: %d",
        metric_workers,
        METRIC_QUEUE_SIZE,
        CHECKPOINT_SIZE,
        MAX_ATTEMPTS,
    )

    connection = _connect_checkpoint(checkpoint)
    pipeline_stats = PipelineStats(0.0, 0.0, 0.0, ())
    try:
        if force:
            tasks_deleted, runs_deleted = _clear_checkpoint_run(connection, fingerprint)
            LOGGER.warning(
                "--force: checkpoint atual removido (%d execucao; %d tarefas). "
                "Outros fingerprints foram preservados.",
                runs_deleted,
                tasks_deleted,
            )
        completed, _ = _prepare_checkpoint(connection, fingerprint, metadata, tasks)
        preparation_seconds = time.perf_counter() - total_started
        _section("Processamento")
        LOGGER.info(
            "Retomada: %d concluidas; %d pendentes.",
            len(completed),
            len(tasks) - len(completed),
        )
        if len(completed) < len(tasks):
            pipeline_stats = _run_pipeline(
                connection,
                fingerprint,
                tasks,
                completed,
                gpu_count,
                batch_size,
                gpu_memory_fraction,
                gpu_prefetch_batches,
                spool_directory,
                model_name,
                metric_workers,
            )
        else:
            _clear_spool_directory(spool_directory, remove_directory=True)
        export_started = time.perf_counter()
        rows = _export_csv(connection, fingerprint, output)
        export_seconds = time.perf_counter() - export_started
        if rows != len(tasks):
            raise RuntimeError(
                f"Exportacao incompleta: {rows} de {len(tasks)} resultados concluidos."
            )
        now = _utc_now()
        with connection:
            connection.execute(
                """UPDATE runs SET status = 'completed', completed_tasks = ?, updated_at = ?
                   WHERE phase = ? AND fingerprint = ?""",
                (rows, now, PHASE, fingerprint),
            )
    finally:
        connection.close()

    total_seconds = time.perf_counter() - total_started
    _section("Execucao concluida")
    LOGGER.info("Resultados: %d linhas em %s", rows, output)
    _log_final_observability(
        tasks,
        completed,
        pipeline_stats,
        preparation_seconds,
        export_seconds,
        total_seconds,
        metric_workers,
    )
    return output


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa ENCODEC multi-GPU com checkpoint SQLite."
    )
    parser.add_argument("--datasets", nargs="+", help="Datasets selecionados; padrao: todos.")
    parser.add_argument(
        "--max-clips-per-dataset",
        type=int,
        default=0,
        help="Limite por dataset (exemplo autonomo: todos; use 0 para todos).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Microbatch inicial por GPU (padrao: 32).",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=DEFAULT_GPU_MEMORY_FRACTION,
        help="Fracao da VRAM livre usada para estimar o batch (padrao: 0.90).",
    )
    parser.add_argument(
        "--gpu-prefetch-batches",
        type=int,
        default=DEFAULT_GPU_PREFETCH_BATCHES,
        help="Reconstrucoes antecipadas por GPU, em batches (padrao: 2).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalcula a execucao atual, preservando outros fingerprints.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--log-path", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.max_clips_per_dataset < 0:
        parser.error("--max-clips-per-dataset deve ser maior ou igual a zero.")
    if arguments.max_clips_per_dataset == 0:
        arguments.max_clips_per_dataset = None
    if arguments.batch_size < 1:
        parser.error("--batch-size deve ser maior que zero.")
    if not 0 < arguments.gpu_memory_fraction < 1:
        parser.error("--gpu-memory-fraction deve estar entre zero e um.")
    if arguments.gpu_prefetch_batches < 1:
        parser.error("--gpu-prefetch-batches deve ser maior que zero.")
    return arguments


def main() -> None:
    arguments = _parse_arguments()
    paths = CodecoPath.discover()
    test_directory = paths.tests / "encodec"
    baseline = load_standalone_examples()
    run(
        baseline=baseline,
        datasets=arguments.datasets,
        max_clips_per_dataset=arguments.max_clips_per_dataset,
        seed=arguments.seed,
        batch_size=arguments.batch_size,
        gpu_memory_fraction=arguments.gpu_memory_fraction,
        gpu_prefetch_batches=arguments.gpu_prefetch_batches,
        model_name=arguments.model_name,
        checkpoint_path=(
            arguments.checkpoint_path
            or test_directory / "checkpoints.sqlite3"
        ),
        output_path=(
            arguments.output_path or test_directory / "results.csv"
        ),
        log_path=(
            arguments.log_path or test_directory / "execution.log"
        ),
        force=arguments.force,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.error("Execucao interrompida pelo usuario.")
        raise SystemExit(130) from None
    except Exception:
        LOGGER.exception("Falha na fase ENCODEC.")
        raise SystemExit(1) from None
