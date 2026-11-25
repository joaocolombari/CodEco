"""Generic foundation-model embedding pipeline.

This module is the stable orchestration entry point for the BirdNET and Perch
embedding backends.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from codeco.codeco_path import CodecoPath
from codeco.environment import EnvironmentState, configure_model_environment
from codeco.prepare_table import (
    BaselineState,
    TqdmLoggingHandler,
    load_pcm_clip,
    load_standalone_examples,
)


MODEL_ORDER = ("birdnet", "perch")
PERCH_GPU_MODEL_HANDLE = (
    "https://www.kaggle.com/models/google/bird-vocalization-classifier/"
    "TensorFlow2/perch_v2/2"
)
PERCH_CPU_MODEL_HANDLE = (
    "https://www.kaggle.com/models/google/bird-vocalization-classifier/"
    "TensorFlow2/perch_v2_cpu/1"
)
MAX_ATTEMPTS = 3
MIN_THROUGHPUT_GAIN = 0.05
PERCH_CALIBRATION_VERSION = 2
LOGGER = logging.getLogger("codeco.embeddings")


def _adapt_for_perch(audio: np.ndarray) -> np.ndarray:
    """Apply the frozen 4 s/48 kHz -> 5 s/32 kHz Perch protocol."""
    import soxr

    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    expected = 4 * 48_000
    if len(values) != expected:
        raise ValueError(
            f"Perch espera {expected} amostras (4 s a 48 kHz); recebeu {len(values)}."
        )
    resampled = soxr.resample(values, 48_000, 32_000, quality="HQ").astype(
        np.float32, copy=False
    )
    target = 5 * 32_000
    missing = target - len(resampled)
    if missing < 0:
        start = -missing // 2
        return resampled[start : start + target]
    return np.pad(
        resampled,
        (missing // 2, missing - missing // 2),
        mode="constant",
    ).astype(np.float32, copy=False)


def _configure_logging(paths: CodecoPath, log_path: Path | None = None) -> Path:
    log = log_path or paths.logs / "embeddings.log"
    if LOGGER.handlers:
        return log
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    terminal = TqdmLoggingHandler(sys.stdout)
    terminal.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(terminal)
    log.parent.mkdir(parents=True, exist_ok=True)
    file = logging.FileHandler(log, encoding="utf-8")
    file.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(file)
    return log


def _perch_model_handle(gpu_count: int) -> str:
    """Select the official Perch v2 export supported by the runtime device."""

    return PERCH_GPU_MODEL_HANDLE if gpu_count else PERCH_CPU_MODEL_HANDLE


def _resolve_perch_model(cache: Path, model_handle: str) -> str:
    cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TFHUB_CACHE_DIR"] = str(cache)
    command = (
        "import sys, tensorflow_hub as hub; "
        "print(hub.resolve(sys.argv[1]))"
    )
    process = subprocess.run(
        [sys.executable, "-c", command, model_handle],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("TFHub nao retornou o diretorio do modelo Perch.")
    return lines[-1]


def _perch_embedding_worker(
    worker_id: int,
    gpu_index: int | None,
    model_path: str,
    sample: np.ndarray,
    initial_batch_size: int,
    max_batch_size: int,
    input_queue: object,
    output_queue: object,
    calibrate: bool = False,
) -> None:
    if gpu_index is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        import warnings

        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API\..*",
            category=UserWarning,
        )
        output_queue.put(
            {
                "kind": "startup",
                "worker_id": worker_id,
                "gpu_index": gpu_index,
                "stage": "tensorflow",
            }
        )
        import absl.logging
        import tensorflow as tf
        import tensorflow_hub as hub

        absl.logging.set_verbosity(absl.logging.ERROR)
        tf.get_logger().setLevel(logging.ERROR)
        physical = tf.config.list_physical_devices("GPU")
        if gpu_index is not None:
            if len(physical) != 1:
                raise RuntimeError(
                    f"Worker Perch {worker_id}: esperada uma GPU; "
                    f"encontradas {len(physical)}."
                )
            tf.config.experimental.set_memory_growth(physical[0], True)
            device = "/GPU:0"
        else:
            device = "/CPU:0"
        output_queue.put(
            {
                "kind": "startup",
                "worker_id": worker_id,
                "gpu_index": gpu_index,
                "stage": "model",
            }
        )
        model_started = time.perf_counter()
        with tf.device(device):
            model = hub.load(model_path)
        signature = model.signatures["serving_default"]
        output_queue.put(
            {
                "kind": "startup",
                "worker_id": worker_id,
                "gpu_index": gpu_index,
                "stage": "model_ready",
                "seconds": time.perf_counter() - model_started,
            }
        )

        def infer(batch: np.ndarray) -> np.ndarray:
            adapted = np.stack(
                [_adapt_for_perch(audio) for audio in np.asarray(batch)], axis=0
            ).astype(np.float32, copy=False)
            with tf.device(device):
                outputs = signature(
                    inputs=tf.convert_to_tensor(adapted, dtype=tf.float32)
                )
            embedding = outputs.get("embedding", list(outputs.values())[-1])
            result = embedding.numpy().astype(np.float32, copy=False)
            if result.ndim != 2 or result.shape[0] != len(adapted):
                raise RuntimeError(f"Forma inesperada do embedding Perch: {result.shape}")
            return result

        output_queue.put(
            {
                "kind": "startup",
                "worker_id": worker_id,
                "gpu_index": gpu_index,
                "stage": "validation",
            }
        )
        validation_started = time.perf_counter()
        sample_result = infer(sample[None, :])
        embedding_dimension = int(sample_result.shape[1])
        output_queue.put(
            {
                "kind": "startup",
                "worker_id": worker_id,
                "gpu_index": gpu_index,
                "stage": "validation_ready",
                "seconds": time.perf_counter() - validation_started,
                "embedding_dimension": embedding_dimension,
            }
        )
        selected = initial_batch_size
        tuning = []
        stop_reason = "batch fixo"
        if calibrate and gpu_index is not None:
            candidate = initial_batch_size
            best_batch = 0
            best_throughput = 0.0
            while True:
                probe = np.repeat(sample[None, :], candidate, axis=0)
                output_queue.put(
                    {
                        "kind": "tuning",
                        "worker_id": worker_id,
                        "gpu_index": gpu_index,
                        "stage": "started",
                        "batch_size": candidate,
                    }
                )
                try:
                    tf.config.experimental.reset_memory_stats(device)
                    started = time.perf_counter()
                    infer(probe)
                    elapsed = time.perf_counter() - started
                    peak = (
                        tf.config.experimental.get_memory_info(device)["peak"]
                        / 1024**2
                    )
                    throughput = candidate / max(elapsed, 1e-12)
                    gain = (
                        math.inf
                        if best_throughput == 0.0
                        else throughput / best_throughput - 1.0
                    )
                    tuning.append(
                        (candidate, "ok", peak, elapsed, throughput, gain)
                    )
                    output_queue.put(
                        {
                            "kind": "tuning",
                            "worker_id": worker_id,
                            "gpu_index": gpu_index,
                            "stage": "completed",
                            "batch_size": candidate,
                            "seconds": elapsed,
                            "peak_vram_mib": peak,
                            "throughput": throughput,
                            "throughput_gain": gain,
                        }
                    )
                    if best_batch == 0 or gain >= MIN_THROUGHPUT_GAIN:
                        best_batch = candidate
                        best_throughput = throughput
                    else:
                        stop_reason = (
                            f"ganho de throughput inferior a "
                            f"{MIN_THROUGHPUT_GAIN:.0%}"
                        )
                        break
                    if candidate >= max_batch_size:
                        stop_reason = "limite util de batch atingido"
                        break
                    candidate = min(candidate * 2, max_batch_size)
                except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as error:
                    error_type = type(error).__name__
                    tuning.append(
                        (candidate, "rejected", math.nan, math.nan, math.nan, math.nan)
                    )
                    output_queue.put(
                        {
                            "kind": "tuning",
                            "worker_id": worker_id,
                            "gpu_index": gpu_index,
                            "stage": "rejected",
                            "batch_size": candidate,
                            "error_type": error_type,
                        }
                    )
                    stop_reason = f"batch rejeitado por {error_type}"
                    if best_batch:
                        break
                    candidate //= 2
                    if candidate < 1:
                        raise
                finally:
                    del probe
            selected = best_batch
        else:
            tuning.append(
                (selected, "fixed", math.nan, math.nan, math.nan, math.nan)
            )
        if calibrate:
            output_queue.put(
                {
                    "kind": "calibrated",
                    "worker_id": worker_id,
                    "gpu_index": gpu_index,
                    "batch_size": selected,
                    "embedding_dimension": embedding_dimension,
                    "tuning": tuning,
                    "stop_reason": stop_reason,
                }
            )
            return
        output_queue.put(
            {
                "kind": "ready",
                "worker_id": worker_id,
                "gpu_index": gpu_index,
                "batch_size": selected,
                "embedding_dimension": embedding_dimension,
                "tuning": tuning,
            }
        )

        active_seconds = 0.0
        tasks = batches = retries = 0
        current_batch = selected
        while True:
            message = input_queue.get()
            if message is None:
                break
            indices = list(message["indices"])
            audio = np.asarray(message["audio"], dtype=np.float32)
            position = failures = 0
            while position < len(indices):
                size = min(current_batch, len(indices) - position)
                try:
                    started = time.perf_counter()
                    embeddings = infer(audio[position : position + size])
                    elapsed = time.perf_counter() - started
                except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as error:
                    failures += 1
                    retries += 1
                    if failures >= MAX_ATTEMPTS or current_batch == 1:
                        raise RuntimeError(
                            "Falha persistente de GPU no Perch apos "
                            f"{failures} tentativas; batch={current_batch}."
                        ) from error
                    current_batch = max(1, current_batch // 2)
                    continue
                output_queue.put(
                    {
                        "kind": "result",
                        "condition_id": message["condition_id"],
                        "indices": indices[position : position + size],
                        "embeddings": embeddings,
                        "worker_id": worker_id,
                    }
                )
                active_seconds += elapsed
                tasks += size
                batches += 1
                position += size
                failures = 0
        peak = None
        if gpu_index is not None:
            peak = tf.config.experimental.get_memory_info(device)["peak"] / 1024**2
        output_queue.put(
            {
                "kind": "stopped",
                "worker_id": worker_id,
                "gpu_index": gpu_index,
                "selected_batch": selected,
                "final_batch": current_batch,
                "tasks": tasks,
                "batches": batches,
                "active_seconds": active_seconds,
                "peak_vram_mib": peak,
                "retries": retries,
            }
        )
    except BaseException as error:
        output_queue.put(
            {
                "kind": "fatal",
                "worker_id": worker_id,
                "gpu_index": gpu_index,
                "error": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            }
        )


def _log_perch_worker_status(message: dict[str, object]) -> bool:
    """Log one worker lifecycle message; return whether it was consumed."""
    kind = message["kind"]
    if kind not in {"startup", "tuning"}:
        return False
    device = (
        "CPU"
        if message["gpu_index"] is None
        else f"GPU {message['gpu_index']}"
    )
    if kind == "startup":
        stage = message["stage"]
        if stage == "tensorflow":
            LOGGER.info(
                "Worker %d | %s | inicializando TensorFlow e TFHub...",
                message["worker_id"],
                device,
            )
        elif stage == "model":
            LOGGER.info(
                "Worker %d | %s | carregando o modelo Perch...",
                message["worker_id"],
                device,
            )
        elif stage == "model_ready":
            LOGGER.info(
                "Worker %d | %s | modelo carregado em %.2f s.",
                message["worker_id"],
                device,
                message["seconds"],
            )
        elif stage == "validation":
            LOGGER.info(
                "Worker %d | %s | validando a inferencia e a saida...",
                message["worker_id"],
                device,
            )
        elif stage == "validation_ready":
            LOGGER.info(
                "Worker %d | %s | inferencia validada em %.2f s; dimensao=%d.",
                message["worker_id"],
                device,
                message["seconds"],
                message["embedding_dimension"],
            )
        return True

    stage = message["stage"]
    if stage == "started":
        LOGGER.info(
            "Worker %d | %s | calibracao descartavel: testando batch %d...",
            message["worker_id"],
            device,
            message["batch_size"],
        )
    elif stage == "completed":
        gain = message["throughput_gain"]
        gain_text = "referencia" if math.isinf(gain) else f"ganho={gain:.1%}"
        LOGGER.info(
            "Worker %d | %s | batch %d: %.2f tarefas/s; %s; %.2f s; "
            "pico VRAM=%.0f MiB.",
            message["worker_id"],
            device,
            message["batch_size"],
            message["throughput"],
            gain_text,
            message["seconds"],
            message["peak_vram_mib"],
        )
    elif stage == "rejected":
        LOGGER.info(
            "Worker %d | %s | batch %d rejeitado por %s; "
            "mantendo o melhor batch anterior.",
            message["worker_id"],
            device,
            message["batch_size"],
            message["error_type"],
        )
    return True


def _staged_indices(
    connection: object,
    fingerprint: str,
    dataset: str,
    condition_id: str,
) -> set[int]:
    rows = connection.execute(
        """SELECT row_index FROM perch_encodec_staging
           WHERE fingerprint=? AND dataset=? AND condition_id=?
             AND status='completed'""",
        (fingerprint, dataset, condition_id),
    )
    return {int(row[0]) for row in rows}


def _checkpoint_staging(
    connection: object,
    fingerprint: str,
    dataset: str,
    condition_id: str,
    indices: Sequence[int],
) -> None:
    from codeco.birdnet_embeddings import _utc_now

    now = _utc_now()
    with connection:
        connection.executemany(
            """INSERT INTO perch_encodec_staging
               (fingerprint, dataset, condition_id, row_index, status, updated_at)
               VALUES (?, ?, ?, ?, 'completed', ?)
               ON CONFLICT(fingerprint, dataset, condition_id, row_index)
               DO UPDATE SET status='completed', updated_at=excluded.updated_at""",
            [
                (fingerprint, dataset, condition_id, int(index), now)
                for index in indices
            ],
        )


def _stage_encodec_audio(
    connection: object,
    fingerprint: str,
    dataset: str,
    clips: object,
    condition: object,
    stage_file: Path,
    encodec_pool: object,
    producer_batch_size: int,
    checkpoint_size: int,
) -> tuple[int, int]:
    import queue
    import threading

    from tqdm import tqdm
    import codeco.birdnet_embeddings as backend

    completed = _staged_indices(
        connection, fingerprint, dataset, condition.identifier
    )
    if completed and not stage_file.is_file():
        with connection:
            connection.execute(
                """DELETE FROM perch_encodec_staging
                   WHERE fingerprint=? AND dataset=? AND condition_id=?""",
                (fingerprint, dataset, condition.identifier),
            )
        completed = set()
    pending = [index for index in range(len(clips)) if index not in completed]
    staged = backend._prepare_embedding_file(
        stage_file,
        len(clips),
        4 * backend.SAMPLE_RATE,
    )
    if not pending:
        LOGGER.info(
            "[CACHE HIT] Reconstrucao ENCODEC | %s | %s: %d audios.",
            dataset,
            condition.setting,
            len(clips),
        )
        return len(completed), 0

    condition_id = f"{dataset}:{condition.identifier}"
    audio_queue: queue.Queue = queue.Queue(maxsize=2)
    status_queue: queue.Queue = queue.Queue()
    producer = threading.Thread(
        target=backend._produce_encodec,
        args=(
            condition_id,
            condition,
            clips,
            pending,
            encodec_pool,
            producer_batch_size,
            audio_queue,
            status_queue,
        ),
        daemon=True,
    )
    producer.start()
    buffer = []
    received = 0
    progress = tqdm(
        total=len(pending),
        desc=f"{dataset} | encodec reconstruido",
        unit="audio",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    try:
        while received < len(pending):
            try:
                message = audio_queue.get(timeout=1)
            except queue.Empty:
                if not producer.is_alive() and not status_queue.empty():
                    kind, detail = status_queue.get_nowait()
                    if kind == "error":
                        raise RuntimeError(detail)
                continue
            indices = [int(index) for index in message["indices"]]
            audio = np.asarray(message["audio"], dtype=np.float32)
            expected = (len(indices), 4 * backend.SAMPLE_RATE)
            if audio.shape != expected:
                raise RuntimeError(
                    f"Reconstrucao ENCODEC com forma {audio.shape}; esperado {expected}."
                )
            staged[indices] = audio
            buffer.extend(indices)
            received += len(indices)
            progress.update(len(indices))
            while len(buffer) >= checkpoint_size:
                staged.flush()
                _checkpoint_staging(
                    connection,
                    fingerprint,
                    dataset,
                    condition.identifier,
                    buffer[:checkpoint_size],
                )
                buffer = buffer[checkpoint_size:]
        producer.join(timeout=30)
        if not status_queue.empty():
            kind, detail = status_queue.get_nowait()
            if kind == "error":
                raise RuntimeError(detail)
        if buffer:
            staged.flush()
            _checkpoint_staging(
                connection,
                fingerprint,
                dataset,
                condition.identifier,
                buffer,
            )
    finally:
        progress.close()
    return len(completed) + received, received


def _produce_staged_audio(
    condition_id: str,
    pending: Sequence[int],
    staged: np.ndarray,
    producer_batch_size: int,
    embedding_queue: object,
    status_queue: object,
) -> None:
    try:
        for position in range(0, len(pending), producer_batch_size):
            indices = list(pending[position : position + producer_batch_size])
            embedding_queue.put(
                {
                    "condition_id": condition_id,
                    "indices": indices,
                    "audio": np.asarray(staged[indices], dtype=np.float32),
                }
            )
        status_queue.put(("done", None))
    except BaseException as error:
        status_queue.put(
            (
                "error",
                "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            )
        )


def _process_staged_embeddings(
    connection: object,
    fingerprint: str,
    dataset: str,
    clips: object,
    condition: object,
    stage_file: Path,
    embedding_file: Path,
    embedding_queue: object,
    embedding_output: object,
    producer_batch_size: int,
    embedding_dimension: int,
) -> tuple[int, int, int, float]:
    import queue
    import threading

    from tqdm import tqdm
    import codeco.birdnet_embeddings as backend

    started = time.perf_counter()
    completed = backend._completed_indices(
        connection, fingerprint, dataset, condition.identifier
    )
    pending = [index for index in range(len(clips)) if index not in completed]
    embeddings = backend._prepare_embedding_file(
        embedding_file, len(clips), embedding_dimension
    )
    if not pending:
        LOGGER.info(
            "[CACHE HIT] %s | %s: %d embeddings.",
            dataset,
            condition.setting,
            len(clips),
        )
        return len(clips), len(completed), 0, time.perf_counter() - started
    if not stage_file.is_file():
        raise RuntimeError(f"Reconstrucao ENCODEC ausente: {stage_file}")
    staged = np.load(stage_file, mmap_mode="r")
    expected = (len(clips), 4 * backend.SAMPLE_RATE)
    if staged.shape != expected or staged.dtype != np.float32:
        raise RuntimeError(
            f"Cache ENCODEC incompativel: {stage_file} | {staged.shape} | {staged.dtype}"
        )

    condition_id = f"{dataset}:{condition.identifier}"
    status_queue: queue.Queue = queue.Queue()
    producer = threading.Thread(
        target=_produce_staged_audio,
        args=(
            condition_id,
            pending,
            staged,
            producer_batch_size,
            embedding_queue,
            status_queue,
        ),
        daemon=True,
    )
    producer.start()
    buffer = []
    received = 0
    progress = tqdm(
        total=len(pending),
        desc=f"{dataset} | perch encodec",
        unit="embedding",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    try:
        while received < len(pending):
            try:
                message = embedding_output.get(timeout=1)
            except queue.Empty:
                if not producer.is_alive() and not status_queue.empty():
                    kind, detail = status_queue.get_nowait()
                    if kind == "error":
                        raise RuntimeError(detail)
                continue
            if message["kind"] == "fatal":
                raise RuntimeError(message["error"])
            if message["kind"] != "result" or message["condition_id"] != condition_id:
                raise RuntimeError("Resultado Perch pertence a outra configuracao.")
            indices = [int(index) for index in message["indices"]]
            embeddings[indices] = np.asarray(message["embeddings"], dtype=np.float32)
            buffer.extend(indices)
            received += len(indices)
            progress.update(len(indices))
            while len(buffer) >= backend.CHECKPOINT_SIZE:
                embeddings.flush()
                backend._checkpoint_batch(
                    connection,
                    fingerprint,
                    dataset,
                    condition.identifier,
                    clips,
                    buffer[: backend.CHECKPOINT_SIZE],
                )
                buffer = buffer[backend.CHECKPOINT_SIZE :]
        producer.join(timeout=30)
        if not status_queue.empty():
            kind, detail = status_queue.get_nowait()
            if kind == "error":
                raise RuntimeError(detail)
        if buffer:
            embeddings.flush()
            backend._checkpoint_batch(
                connection,
                fingerprint,
                dataset,
                condition.identifier,
                clips,
                buffer,
            )
    finally:
        progress.close()
    return (
        len(completed) + received,
        len(completed),
        received,
        time.perf_counter() - started,
    )


@dataclass(frozen=True)
class EmbeddingRun:
    """Arguments shared by foundation-model embedding backends."""

    datasets: tuple[str, ...] | None
    codecs: tuple[str, ...] | None
    max_clips_per_dataset: int | None
    seed: int
    batch_size: int
    max_batch_size: int
    gpu_memory_fraction: float
    encodec_memory_fraction: float
    checkpoint_path: Path | None
    force: bool
    baseline: BaselineState
    environment: EnvironmentState | None
    artifact_directory: Path | None


@dataclass(frozen=True)
class EmbeddingBackend:
    name: str
    run: Callable[[EmbeddingRun], Path]
    migrated: bool


def _run_birdnet(arguments: EmbeddingRun) -> Path:
    from codeco.birdnet_embeddings import run

    return run(
        baseline=arguments.baseline,
        environment=arguments.environment,
        datasets=arguments.datasets,
        codecs=arguments.codecs,
        max_clips_per_dataset=arguments.max_clips_per_dataset,
        seed=arguments.seed,
        batch_size=arguments.batch_size,
        max_batch_size=arguments.max_batch_size,
        gpu_memory_fraction=arguments.gpu_memory_fraction,
        encodec_memory_fraction=arguments.encodec_memory_fraction,
        checkpoint_path=arguments.checkpoint_path,
        force=arguments.force,
        artifact_directory=arguments.artifact_directory,
    )


def _run_perch(arguments: EmbeddingRun) -> Path:
    import multiprocessing as mp
    import queue
    from dataclasses import asdict

    import pandas as pd
    import psutil
    import codeco.birdnet_embeddings as backend
    from codeco.ffmpeg import _ffmpeg_installation, _load_clip

    paths = CodecoPath.discover()
    backend._configure_cuda_environment()
    os.environ.setdefault("MPLCONFIGDIR", str(paths.cache / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    log = _configure_logging(
        paths,
        (
            arguments.artifact_directory / "perch.log"
            if arguments.artifact_directory is not None
            else None
        ),
    )
    backend.LOGGER.handlers = LOGGER.handlers
    backend.LOGGER.setLevel(logging.INFO)
    backend.LOGGER.propagate = False
    checkpoint = arguments.checkpoint_path or paths.cache / "checkpoints.sqlite3"
    output = (
        arguments.artifact_directory / "perch_embeddings.csv"
        if arguments.artifact_directory is not None
        else paths.tables / "perch_embeddings.csv"
    )
    cache_root = (
        arguments.artifact_directory / "perch_cache"
        if arguments.artifact_directory is not None
        else paths.cache / "perch_embeddings"
    )
    clips = backend._select_clips(
        arguments.baseline,
        arguments.datasets,
        arguments.max_clips_per_dataset,
        arguments.seed,
    )
    if arguments.environment is None:
        ffmpeg_system, ffmpeg_lc3, ffmpeg_version = _ffmpeg_installation()
    else:
        ffmpeg_system = arguments.environment.system.ffmpeg.executable
        ffmpeg_lc3 = arguments.environment.system.ffmpeg.lc3_executable
        ffmpeg_version = (
            f"system={arguments.environment.system.ffmpeg.version}; "
            f"lc3={arguments.environment.system.ffmpeg.lc3_version}"
        )
    ffmpeg = (ffmpeg_system, ffmpeg_lc3)
    cpu_workers = max(1, psutil.cpu_count(logical=False) or 1)
    gpu_count = (
        backend._tensorflow_gpu_count()
        if arguments.environment is None
        else arguments.environment.system.tensorflow_gpu_count
    )
    gpu_indices = list(range(gpu_count))
    selected_conditions = tuple(
        condition
        for condition in backend.CONDITIONS
        if arguments.codecs is None or condition.codec in set(arguments.codecs)
    )
    if not selected_conditions:
        raise ValueError("Nenhuma configuracao de codec selecionada para o Perch.")

    model_handle = _perch_model_handle(gpu_count)
    LOGGER.info(
        "Resolvendo modelo Perch (%s): %s",
        "GPU" if gpu_indices else "CPU",
        model_handle,
    )
    tfhub_cache = paths.models / "tensorflow_hub"
    os.environ["TFHUB_CACHE_DIR"] = str(tfhub_cache)
    model_path = _resolve_perch_model(tfhub_cache, model_handle)
    sample_row = clips.iloc[0]
    sample = (
        _load_clip(str(sample_row["path"]), float(sample_row["start"]))
        if sample_row["pcm_file"] is None
        else load_pcm_clip(str(sample_row["pcm_file"]), int(sample_row["pcm_row"]))
    )
    selection = clips[
        [
            "dataset",
            "clip_id",
            "path",
            "start",
            "split",
            "pcm_file",
            "pcm_row",
        ]
    ].copy()
    selection["path"] = selection["path"].map(paths.project_relative)
    selection["pcm_file"] = selection["pcm_file"].map(
        lambda value: (
            None
            if value is None or pd.isna(value)
            else paths.project_relative(value)
        )
    )
    metadata = {
        "implementation_version": "1",
        "manifest_sha256": arguments.baseline.manifest_sha256,
        "selection": selection.to_dict("records"),
        "conditions": [asdict(condition) for condition in selected_conditions],
        "foundation_model": "perch",
        "model_handle": model_handle,
        "seed": arguments.seed,
        "batch_size": arguments.batch_size,
        "max_batch_size": arguments.max_batch_size,
    }
    fingerprint = backend._sha256_json(metadata)

    LOGGER.info("Log: %s", log)
    LOGGER.info("Checkpoint: %s", checkpoint)
    LOGGER.info("Indice CSV: %s", output)
    LOGGER.info("Modelo: Google Perch v2 | TFHub")
    LOGGER.info("Entrada: 5 s | 32 kHz | padding simetrico de 0,5 s")
    LOGGER.info("FFmpeg: %s", ffmpeg_version)
    LOGGER.info("Nucleos fisicos: %d", cpu_workers)
    if arguments.max_clips_per_dataset is None:
        LOGGER.info("Selecao: corpus completo congelado; nenhuma amostragem aplicada.")
    else:
        LOGGER.info(
            "Selecao: benchmark estratificado; ate %d clipes/dataset; seed=%d.",
            arguments.max_clips_per_dataset,
            arguments.seed,
        )
    if gpu_indices:
        LOGGER.info(
            "GPUs TensorFlow: %d; microbatch=%d; maximo=%d",
            len(gpu_indices),
            arguments.batch_size,
            arguments.max_batch_size,
        )
    else:
        LOGGER.warning(
            "Nenhuma GPU TensorFlow: fallback Perch/CPU em batch %d.",
            arguments.batch_size,
        )

    context = mp.get_context("spawn")
    worker_specs = (
        [(index, index) for index in gpu_indices]
        if gpu_indices
        else [(0, None)]
    )
    largest_dataset = max(
        len(group) for _, group in clips.groupby("dataset", sort=False)
    )
    work_per_worker = math.ceil(largest_dataset / max(1, len(worker_specs)))
    useful_power_of_two = 1 << max(0, work_per_worker.bit_length() - 1)
    useful_batch_limit = min(
        arguments.max_batch_size,
        max(arguments.batch_size, useful_power_of_two),
    )
    LOGGER.info(
        "Calibracao: ate %d clipes/worker; teto util de batch=%d.",
        work_per_worker,
        useful_batch_limit,
    )
    selected_batches: dict[int, int] = {}
    calibration_dimensions: set[int] = set()
    calibration_keys: dict[int, str] = {}
    missing_calibrations = []
    calibration_connection = backend._connect_checkpoint(checkpoint)
    try:
        calibration_connection.execute(
            """CREATE TABLE IF NOT EXISTS perch_calibrations (
                   calibration_key TEXT PRIMARY KEY,
                   metadata_json TEXT NOT NULL,
                   result_json TEXT NOT NULL,
                   updated_at TEXT NOT NULL
               )"""
        )
        calibration_connection.commit()
        for worker_id, gpu_index in worker_specs:
            if gpu_index is None:
                selected_batches[worker_id] = arguments.batch_size
                continue
            calibration_metadata = {
                "version": PERCH_CALIBRATION_VERSION,
                "model_handle": model_handle,
                "logical_worker": worker_id,
                "initial_batch_size": arguments.batch_size,
                "max_batch_size": useful_batch_limit,
                "minimum_throughput_gain": MIN_THROUGHPUT_GAIN,
            }
            calibration_key = backend._sha256_json(calibration_metadata)
            calibration_keys[worker_id] = calibration_key
            row = calibration_connection.execute(
                "SELECT result_json FROM perch_calibrations "
                "WHERE calibration_key=?",
                (calibration_key,),
            ).fetchone()
            if row is None:
                missing_calibrations.append(
                    (worker_id, gpu_index, calibration_metadata)
                )
                continue
            result = json.loads(row[0])
            selected_batches[worker_id] = int(result["batch_size"])
            calibration_dimensions.add(int(result["embedding_dimension"]))
            LOGGER.info(
                "Calibracao reutilizada: GPU %d | batch=%d | %s",
                gpu_index,
                selected_batches[worker_id],
                result["stop_reason"],
            )

        if missing_calibrations:
            LOGGER.info("\nCalibracao descartavel do Perch")
            LOGGER.info("%s", "-" * 72)
            LOGGER.info(
                "As inferencias desta fase nao geram embeddings. Os processos "
                "serao encerrados ao final para liberar toda a VRAM."
            )
            calibration_output = context.Queue()
            calibrators = [
                context.Process(
                    target=_perch_embedding_worker,
                    args=(
                        worker_id,
                        gpu_index,
                        model_path,
                        sample,
                        arguments.batch_size,
                        useful_batch_limit,
                        None,
                        calibration_output,
                        True,
                    ),
                    name=f"Perch-calibration-GPU-{gpu_index}",
                )
                for worker_id, gpu_index, _ in missing_calibrations
            ]
            for calibrator in calibrators:
                calibrator.start()
            calibrated_messages = []
            try:
                while len(calibrated_messages) < len(calibrators):
                    message = calibration_output.get(timeout=1800)
                    if message["kind"] == "fatal":
                        raise RuntimeError(message["error"])
                    if _log_perch_worker_status(message):
                        continue
                    if message["kind"] == "calibrated":
                        calibrated_messages.append(message)
                        selected_batches[int(message["worker_id"])] = int(
                            message["batch_size"]
                        )
                        calibration_dimensions.add(
                            int(message["embedding_dimension"])
                        )
                        LOGGER.info(
                            "Calibracao %d/%d concluida | GPU %d | batch=%d | %s",
                            len(calibrated_messages),
                            len(calibrators),
                            message["gpu_index"],
                            message["batch_size"],
                            message["stop_reason"],
                        )
            finally:
                for calibrator in calibrators:
                    calibrator.join(timeout=10)
                    if calibrator.is_alive():
                        calibrator.terminate()
                        calibrator.join(timeout=10)
                calibration_output.close()
                calibration_output.join_thread()
            metadata_by_worker = {
                worker_id: calibration_metadata
                for worker_id, _, calibration_metadata in missing_calibrations
            }
            with calibration_connection:
                for message in calibrated_messages:
                    worker_id = int(message["worker_id"])
                    result = {
                        "batch_size": int(message["batch_size"]),
                        "embedding_dimension": int(message["embedding_dimension"]),
                        "tuning": message["tuning"],
                        "stop_reason": message["stop_reason"],
                    }
                    calibration_connection.execute(
                        """INSERT INTO perch_calibrations
                           (calibration_key, metadata_json, result_json, updated_at)
                           VALUES (?, ?, ?, ?)
                           ON CONFLICT(calibration_key) DO UPDATE SET
                             metadata_json=excluded.metadata_json,
                             result_json=excluded.result_json,
                             updated_at=excluded.updated_at""",
                        (
                            calibration_keys[worker_id],
                            json.dumps(metadata_by_worker[worker_id], ensure_ascii=False),
                            json.dumps(result, ensure_ascii=False),
                            backend._utc_now(),
                        ),
                    )
            LOGGER.info(
                "Calibradores encerrados; VRAM liberada antes do processamento real."
            )
    finally:
        calibration_connection.close()

    dataset_tables = {
        str(dataset): dataset_clips.reset_index(drop=True)
        for dataset, dataset_clips in clips.groupby("dataset", sort=False)
    }
    stage_files: dict[tuple[str, str], Path] = {}
    staging_connection = backend._connect_checkpoint(checkpoint)
    encodec_pool = None
    try:
        staging_connection.execute(
            """CREATE TABLE IF NOT EXISTS perch_encodec_staging (
                   fingerprint TEXT NOT NULL,
                   dataset TEXT NOT NULL,
                   condition_id TEXT NOT NULL,
                   row_index INTEGER NOT NULL,
                   status TEXT NOT NULL,
                   updated_at TEXT NOT NULL,
                   PRIMARY KEY (fingerprint, dataset, condition_id, row_index)
               )"""
        )
        if arguments.force:
            staging_connection.execute(
                "DELETE FROM perch_encodec_staging WHERE fingerprint=?",
                (fingerprint,),
            )
        staging_connection.commit()

        from codeco.embedding_cache import seed_compatible_embeddings

        imported = 0
        if not arguments.force:
            imported = seed_compatible_embeddings(
                paths=paths,
                model="perch",
                model_version="perch_v2/2",
                fingerprint=fingerprint,
                datasets=dataset_tables,
                conditions=selected_conditions,
                cache_root=cache_root,
                connection=staging_connection,
                logger=LOGGER,
            )
        if imported:
            LOGGER.info(
                "%d caches compatíveis importados de índices anteriores/handoffs.",
                imported,
            )
        encodec_conditions = tuple(
            condition
            for condition in selected_conditions
            if condition.kind == "encodec"
        )
        staging_jobs = []
        for condition in encodec_conditions:
            for dataset, dataset_clips in dataset_tables.items():
                embedding_completed = (
                    set()
                    if arguments.force
                    else backend._completed_indices(
                        staging_connection,
                        fingerprint,
                        dataset,
                        condition.identifier,
                    )
                )
                if len(embedding_completed) == len(dataset_clips):
                    continue
                stage_file = (
                    cache_root
                    / fingerprint
                    / dataset
                    / "encodec_staging"
                    / f"{condition.identifier}.npy"
                )
                if arguments.force:
                    stage_file.unlink(missing_ok=True)
                stage_files[(dataset, condition.identifier)] = stage_file
                staging_jobs.append((condition, dataset, dataset_clips, stage_file))

        if staging_jobs:
            LOGGER.info("\nPreparacao das reconstrucoes ENCODEC")
            LOGGER.info("%s", "-" * 72)
            LOGGER.info(
                "%d configuracoes/datasets pendentes. O modelo ENCODEC sera "
                "carregado uma unica vez; o Perch ainda nao esta residente.",
                len(staging_jobs),
            )
            encodec_workers, encodec_initial_batch, encodec_max_batch = (
                backend._encodec_pool_parameters(
                    gpu_indices,
                    arguments.batch_size,
                    arguments.max_batch_size,
                )
            )
            encodec_job_size = (
                backend.CHECKPOINT_SIZE if gpu_indices else 1
            )
            encodec_checkpoint_size = (
                backend.CHECKPOINT_SIZE if gpu_indices else 1
            )
            LOGGER.info(
                "EnCodec %s | workers=%d | batch=%d | lote=%d | checkpoint=%d.",
                "GPU" if gpu_indices else "CPU",
                len(encodec_workers),
                encodec_initial_batch,
                encodec_job_size,
                encodec_checkpoint_size,
            )
            encodec_pool = backend._start_encodec_pool(
                encodec_workers,
                encodec_initial_batch,
                encodec_max_batch,
                arguments.encodec_memory_fraction,
                center_for_birdnet=False,
            )
            for number, (condition, dataset, dataset_clips, stage_file) in enumerate(
                staging_jobs, 1
            ):
                LOGGER.info(
                    "Reconstrucao %d/%d: %s | %s | %s",
                    number,
                    len(staging_jobs),
                    dataset,
                    condition.codec.upper(),
                    condition.setting,
                )
                _stage_encodec_audio(
                    staging_connection,
                    fingerprint,
                    dataset,
                    dataset_clips,
                    condition,
                    stage_file,
                    encodec_pool,
                    encodec_job_size,
                    encodec_checkpoint_size,
                )
    finally:
        if encodec_pool is not None:
            backend._stop_encodec_pool(encodec_pool)
            LOGGER.info(
                "Reconstrucoes ENCODEC preparadas; modelo encerrado e VRAM liberada."
            )
        staging_connection.close()

    embedding_queue = context.Queue(maxsize=max(2, len(gpu_indices) * 2 or 2))
    embedding_output = context.Queue()
    workers = [
        context.Process(
            target=_perch_embedding_worker,
            args=(
                worker_id,
                gpu_index,
                model_path,
                sample,
                selected_batches[worker_id],
                selected_batches[worker_id],
                embedding_queue,
                embedding_output,
                False,
            ),
            name=f"Perch-{'GPU-' + str(gpu_index) if gpu_index is not None else 'CPU'}",
        )
        for worker_id, gpu_index in worker_specs
    ]
    for worker in workers:
        worker.start()
    LOGGER.info("\nInicializacao dos workers Perch")
    LOGGER.info("%s", "-" * 72)
    LOGGER.info(
        "Progresso: 0/%d workers prontos. O processamento dos clipes comeca "
        "apos o carregamento dos modelos limpos.",
        len(workers),
    )
    ready_messages = []
    try:
        while len(ready_messages) < len(workers):
            message = embedding_output.get(timeout=1800)
            if message["kind"] == "fatal":
                raise RuntimeError(message["error"])
            if _log_perch_worker_status(message):
                continue
            if message["kind"] == "ready":
                ready_messages.append(message)
                LOGGER.info(
                    "Progresso: %d/%d workers prontos | Worker %d: %s; "
                    "batch=%d; dimensao=%d; ajuste=%s",
                    len(ready_messages),
                    len(workers),
                    message["worker_id"],
                    "CPU"
                    if message["gpu_index"] is None
                    else f"GPU {message['gpu_index']}",
                    message["batch_size"],
                    message["embedding_dimension"],
                    "configuracao calibrada",
                )
    except BaseException:
        for _ in workers:
            embedding_queue.put(None)
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
        embedding_queue.close()
        embedding_output.close()
        raise
    dimensions = {int(message["embedding_dimension"]) for message in ready_messages}
    dimensions.update(calibration_dimensions)
    if len(dimensions) != 1:
        raise RuntimeError(f"Workers Perch discordam na dimensao: {dimensions}")
    embedding_dimension = dimensions.pop()

    connection = backend._connect_checkpoint(checkpoint)
    records = []
    started = time.perf_counter()
    producer_batch_size = (
        max(selected_batches.values()) if gpu_indices else arguments.batch_size
    )
    worker_statistics = []
    try:
        now = backend._utc_now()
        with connection:
            if arguments.force:
                connection.execute(
                    "DELETE FROM birdnet_embeddings WHERE fingerprint = ?",
                    (fingerprint,),
                )
                connection.execute(
                    "DELETE FROM birdnet_runs WHERE fingerprint = ?", (fingerprint,)
                )
            connection.execute(
                """INSERT INTO birdnet_runs
                   (fingerprint, metadata_json, status, created_at, updated_at)
                   VALUES (?, ?, 'running', ?, ?)
                   ON CONFLICT(fingerprint) DO UPDATE SET
                     status='running', updated_at=excluded.updated_at""",
                (
                    fingerprint,
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        dataset_tables = {}
        clips_files = {}
        for dataset, dataset_clips in clips.groupby("dataset", sort=False):
            dataset_clips = dataset_clips.reset_index(drop=True)
            clips_file = cache_root / fingerprint / str(dataset) / "clips.parquet"
            columns = [
                "dataset",
                "clip_id",
                "path",
                "start",
                "end",
                "labels",
                "split",
                "source_group_id",
            ]
            clip_index = dataset_clips[columns].copy()
            clip_index["path"] = clip_index["path"].map(paths.project_relative)
            clip_index.insert(
                0, "row_index", np.arange(len(clip_index), dtype=np.int64)
            )
            backend._write_index_parquet(clip_index, clips_file)
            dataset_tables[str(dataset)] = dataset_clips
            clips_files[str(dataset)] = clips_file

        processing_conditions = tuple(
            condition for condition in selected_conditions if condition.kind != "encodec"
        ) + tuple(
            condition for condition in selected_conditions if condition.kind == "encodec"
        )
        for number, condition in enumerate(processing_conditions, 1):
            LOGGER.info(
                "\nConfiguracao %d/%d: %s | %s",
                number,
                len(processing_conditions),
                condition.codec.upper(),
                condition.setting,
            )
            for dataset, dataset_clips in dataset_tables.items():
                embedding_file = (
                    cache_root
                    / fingerprint
                    / dataset
                    / f"{condition.identifier}.npy"
                )
                if condition.kind == "encodec":
                    stage_file = stage_files.get(
                        (dataset, condition.identifier),
                        cache_root
                        / fingerprint
                        / dataset
                        / "encodec_staging"
                        / f"{condition.identifier}.npy",
                    )
                    completed, resumed, processed, seconds = (
                        _process_staged_embeddings(
                            connection=connection,
                            fingerprint=fingerprint,
                            dataset=dataset,
                            clips=dataset_clips,
                            condition=condition,
                            stage_file=stage_file,
                            embedding_file=embedding_file,
                            embedding_queue=embedding_queue,
                            embedding_output=embedding_output,
                            producer_batch_size=producer_batch_size,
                            embedding_dimension=embedding_dimension,
                        )
                    )
                    if completed == len(dataset_clips) and stage_file.is_file():
                        stage_file.unlink()
                        with connection:
                            connection.execute(
                                """DELETE FROM perch_encodec_staging
                                   WHERE fingerprint=? AND dataset=? AND condition_id=?""",
                                (fingerprint, dataset, condition.identifier),
                            )
                        LOGGER.info(
                            "Temporario removido apos checkpoint dos embeddings: %s",
                            stage_file,
                        )
                else:
                    completed, resumed, processed, seconds = (
                        backend._process_condition(
                            connection=connection,
                            fingerprint=fingerprint,
                            dataset=dataset,
                            clips=dataset_clips,
                            condition=condition,
                            embedding_file=embedding_file,
                            ffmpeg=ffmpeg,
                            cpu_workers=cpu_workers,
                            encodec_pool=None,
                            embedding_queue=embedding_queue,
                            embedding_output=embedding_output,
                            producer_batch_size=producer_batch_size,
                            embedding_dimension=embedding_dimension,
                        )
                    )
                records.append(
                    {
                        "dataset": dataset,
                        "foundation_model": "perch",
                        "model_version": "perch_v2/2",
                        "codec": condition.codec,
                        "setting": condition.setting,
                        "clips": len(dataset_clips),
                        "completed": completed,
                        "resumed": resumed,
                        "processed": processed,
                        "seconds": seconds,
                        "embeddings_per_second": processed / max(seconds, 1e-12),
                        "embedding_dimension": embedding_dimension,
                        "clips_file": paths.project_relative(clips_files[dataset]),
                        "embeddings_file": paths.project_relative(embedding_file),
                    }
                )
                backend._write_index_csv(records, output)
        with connection:
            connection.execute(
                "UPDATE birdnet_runs SET status='completed', updated_at=? "
                "WHERE fingerprint=?",
                (backend._utc_now(), fingerprint),
            )
    finally:
        connection.close()
        for _ in workers:
            embedding_queue.put(None)
        stopped = 0
        deadline = time.monotonic() + 120
        while stopped < len(workers) and time.monotonic() < deadline:
            try:
                message = embedding_output.get(timeout=1)
            except queue.Empty:
                continue
            if message["kind"] == "stopped":
                stopped += 1
                worker_statistics.append(message)
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
        embedding_queue.close()
        embedding_queue.join_thread()
        embedding_output.close()
        embedding_output.join_thread()

    LOGGER.info("\nPerch concluido: %d configuracoes | %s", len(records), output)
    for statistics in sorted(worker_statistics, key=lambda row: row["worker_id"]):
        LOGGER.info(
            "Worker %d | %s | batch %s->%s | tarefas=%d | pico VRAM=%s",
            statistics["worker_id"],
            "CPU"
            if statistics["gpu_index"] is None
            else f"GPU {statistics['gpu_index']}",
            statistics["selected_batch"],
            statistics["final_batch"],
            statistics["tasks"],
            "-"
            if statistics["peak_vram_mib"] is None
            else f"{statistics['peak_vram_mib']:.0f} MiB",
        )
    LOGGER.info("Tempo total Perch: %.2f s", time.perf_counter() - started)
    return output


BACKENDS = {
    "birdnet": EmbeddingBackend("birdnet", _run_birdnet, migrated=True),
    "perch": EmbeddingBackend("perch", _run_perch, migrated=True),
}


def _ordered_models(models: Sequence[str] | None) -> tuple[str, ...]:
    requested = set(models or MODEL_ORDER)
    unknown = requested.difference(BACKENDS)
    if unknown:
        raise ValueError(f"Modelos desconhecidos: {', '.join(sorted(unknown))}")
    return tuple(model for model in MODEL_ORDER if model in requested)


def run(
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    codecs: Sequence[str] | None = None,
    max_clips_per_dataset: int | None = None,
    seed: int = 42,
    batch_size: int = 32,
    max_batch_size: int = 512,
    gpu_memory_fraction: float = 0.85,
    encodec_memory_fraction: float = 0.8,
    checkpoint_path: Path | None = None,
    force: bool = False,
    baseline: BaselineState | None = None,
    environment: EnvironmentState | None = None,
) -> list[Path]:
    configure_model_environment(CodecoPath.discover())
    selected = _ordered_models(models)
    standalone = baseline is None
    selected_baseline = baseline or load_standalone_examples()
    paths = CodecoPath.discover()
    artifact_directory = paths.tests / "embeddings" if standalone else None
    if artifact_directory is not None:
        artifact_directory.mkdir(parents=True, exist_ok=True)
    selected_checkpoint = checkpoint_path or (
        artifact_directory / "checkpoints.sqlite3"
        if artifact_directory is not None
        else None
    )
    arguments = EmbeddingRun(
        datasets=tuple(datasets) if datasets is not None else None,
        codecs=tuple(codecs) if codecs is not None else None,
        max_clips_per_dataset=max_clips_per_dataset,
        seed=seed,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        gpu_memory_fraction=gpu_memory_fraction,
        encodec_memory_fraction=encodec_memory_fraction,
        checkpoint_path=selected_checkpoint,
        force=force,
        baseline=selected_baseline,
        environment=environment,
        artifact_directory=artifact_directory,
    )
    outputs = []
    for number, model in enumerate(selected, 1):
        backend = BACKENDS[model]
        print("\n" + "=" * 72)
        print(f"EMBEDDINGS {number}/{len(selected)} | {model.upper()}")
        print("=" * 72)
        outputs.append(backend.run(arguments))
    return outputs


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera embeddings de foundation models em ordem deterministica."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
        help="Modelos a processar; BirdNET sempre precede Perch.",
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--codecs", nargs="+")
    parser.add_argument("--max-clips-per-dataset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.85)
    parser.add_argument("--encodec-memory-fraction", type=float, default=0.8)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.max_clips_per_dataset < 0:
        parser.error("--max-clips-per-dataset nao pode ser negativo.")
    if arguments.batch_size < 1:
        parser.error("--batch-size deve ser maior que zero.")
    if arguments.max_batch_size < arguments.batch_size:
        parser.error("--max-batch-size deve ser maior ou igual a --batch-size.")
    if not 0 < arguments.gpu_memory_fraction < 1:
        parser.error("--gpu-memory-fraction deve estar entre zero e um.")
    if not 0 < arguments.encodec_memory_fraction < 1:
        parser.error("--encodec-memory-fraction deve estar entre zero e um.")
    return arguments


def main() -> None:
    arguments = _parse_arguments()
    run(
        models=arguments.models,
        datasets=arguments.datasets,
        codecs=arguments.codecs,
        max_clips_per_dataset=(
            None
            if arguments.max_clips_per_dataset == 0
            else arguments.max_clips_per_dataset
        ),
        seed=arguments.seed,
        batch_size=arguments.batch_size,
        max_batch_size=arguments.max_batch_size,
        gpu_memory_fraction=arguments.gpu_memory_fraction,
        encodec_memory_fraction=arguments.encodec_memory_fraction,
        checkpoint_path=arguments.checkpoint,
        force=arguments.force,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as error:
        print(f"Falha no pipeline de embeddings: {error}", file=sys.stderr)
        raise SystemExit(1) from None
