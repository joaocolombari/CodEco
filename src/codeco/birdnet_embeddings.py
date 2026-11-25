"""Streaming, resumable BirdNET 2.4 embedding pipeline.

Reconstructed clips are never retained after their embedding has been written.
Baseline and FFmpeg conditions are produced by a physical-core process pool and
fed through a bounded queue to persistent TensorFlow workers.  ENCODEC uses one
persistent producer per CUDA device and feeds the same embedding queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import sqlite3
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import psutil
from numpy.lib.format import open_memmap
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from codeco.codeco_path import CodecoPath
from codeco.environment import EnvironmentState, configure_model_environment
from codeco.prepare_table import (
    BaselineState,
    TqdmLoggingHandler,
    load_pcm_clip,
    load_standalone_examples,
)


PHASE = "birdnet_embeddings"
IMPLEMENTATION_VERSION = "2"
MODEL_VERSION = "2.4"
SAMPLE_RATE = 48_000
CLIP_SECONDS = 4.0
CLIP_SAMPLES = 192_000
MODEL_SAMPLES = 144_000
EMBEDDING_DIMENSION = 1024
CHECKPOINT_SIZE = 64
MAX_ATTEMPTS = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_BATCH_SIZE = 512
DEFAULT_GPU_MEMORY_FRACTION = 0.85
DEFAULT_ENCODEC_MEMORY_FRACTION = 0.80
DEFAULT_SEED = 42
LOGGER = logging.getLogger("codeco.birdnet_embeddings")


@dataclass(frozen=True)
class EmbeddingCondition:
    codec: str
    setting: str
    kind: str
    extension: str = ""
    encoder_arguments: tuple[str, ...] = ()
    bandwidth_kbps: float | None = None

    @property
    def identifier(self) -> str:
        digest = hashlib.sha256(f"{self.codec}\0{self.setting}".encode()).hexdigest()[:12]
        return f"{self.codec}-{digest}"


@dataclass
class EncodecPool:
    input_queue: object
    output_queue: object
    workers: list[mp.Process]


def _conditions() -> tuple[EmbeddingCondition, ...]:
    from codeco.encodec import CONDITIONS as encodec_conditions
    from codeco.ffmpeg import CONDITIONS as ffmpeg_conditions

    classical = tuple(
        EmbeddingCondition(
            condition.codec,
            condition.setting,
            "ffmpeg",
            condition.extension,
            condition.encoder_arguments,
        )
        for condition in ffmpeg_conditions
    )
    neural = tuple(
        EmbeddingCondition(
            condition.codec,
            condition.setting,
            "encodec",
            bandwidth_kbps=condition.bandwidth_kbps,
        )
        for condition in encodec_conditions
    )
    return classical + neural


CONDITIONS = _conditions()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_logging(paths: CodecoPath, log_path: Path | None = None) -> Path:
    log_file = log_path or paths.logs / "birdnet.log"
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
    LOGGER.info("\n%s\nBIRDNET 2.4: EMBEDDINGS EM BATCH\n[CMD]   %s\n%s", line, command, line)


def _section(title: str) -> None:
    LOGGER.info("\n%s\n%s", title, "-" * 72)


def _configure_cuda_environment() -> None:
    purelib = Path(sysconfig.get_paths()["purelib"])
    nvidia = purelib / "nvidia"
    libraries = sorted(path for path in nvidia.glob("*/lib") if path.is_dir())
    existing = [value for value in os.environ.get("LD_LIBRARY_PATH", "").split(":") if value]
    additions = [str(path) for path in libraries if str(path) not in existing]
    if additions:
        os.environ["LD_LIBRARY_PATH"] = ":".join([*additions, *existing])
    nvcc = nvidia / "cuda_nvcc" / "bin"
    if nvcc.is_dir() and str(nvcc) not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = f"{nvcc}:{os.environ.get('PATH', '')}"


def _tensorflow_gpu_count() -> int:
    probe = (
        "import tensorflow as tf; "
        "print('CODECO_GPU_COUNT=' + str(len(tf.config.list_physical_devices('GPU'))))"
    )
    environment = os.environ.copy()
    environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    process = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=environment
    )
    if process.returncode:
        return 0
    for line in reversed(process.stdout.splitlines()):
        if line.startswith("CODECO_GPU_COUNT="):
            return int(line.split("=", 1)[1])
    return 0


def _gpu_memory_totals() -> dict[int, float]:
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}
    totals = {}
    for line in process.stdout.splitlines():
        try:
            index, memory = (value.strip() for value in line.split(",", 1))
            totals[int(index)] = float(memory)
        except ValueError:
            continue
    return totals


def _center_birdnet(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) < MODEL_SAMPLES:
        missing = MODEL_SAMPLES - len(audio)
        audio = np.pad(audio, (missing // 2, missing - missing // 2))
    start = (len(audio) - MODEL_SAMPLES) // 2
    return audio[start : start + MODEL_SAMPLES].astype(np.float32, copy=False)


def _embedding_worker(
    worker_id: int,
    gpu_index: int | None,
    model_path: str,
    sample: np.ndarray,
    initial_batch_size: int,
    max_batch_size: int,
    memory_fraction: float,
    total_memory_mib: float | None,
    input_queue: object,
    output_queue: object,
) -> None:
    if gpu_index is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        import absl.logging
        import tensorflow as tf

        absl.logging.set_verbosity(absl.logging.ERROR)
        tf.get_logger().setLevel(logging.ERROR)
        physical = tf.config.list_physical_devices("GPU")
        if gpu_index is not None:
            if len(physical) != 1:
                raise RuntimeError(f"Worker {worker_id}: esperada uma GPU; encontradas {len(physical)}.")
            tf.config.experimental.set_memory_growth(physical[0], True)
            device = "/GPU:0"
        else:
            device = "/CPU:0"
        with tf.device(device):
            model = tf.saved_model.load(model_path)
        encode = model.signatures["embeddings"]

        def infer(batch: np.ndarray) -> np.ndarray:
            values = np.stack(
                [_center_birdnet(audio) for audio in np.asarray(batch)], axis=0
            ).astype(np.float32, copy=False)
            with tf.device(device):
                result = encode(inputs=tf.convert_to_tensor(values, dtype=tf.float32))["embeddings"]
            result = result.numpy().astype(np.float32, copy=False)
            if result.shape != (len(values), EMBEDDING_DIMENSION):
                raise RuntimeError(f"Forma inesperada: {result.shape}")
            return result

        selected = initial_batch_size
        tuning = []
        if gpu_index is not None:
            candidate = initial_batch_size
            last_success = 0
            while True:
                probe = np.repeat(sample[None, :], candidate, axis=0)
                try:
                    tf.config.experimental.reset_memory_stats(device)
                    started = time.perf_counter()
                    infer(probe)
                    elapsed = time.perf_counter() - started
                    peak = tf.config.experimental.get_memory_info(device)["peak"] / 1024**2
                    last_success = candidate
                    tuning.append((candidate, "ok", peak, elapsed))
                    pressure = peak / total_memory_mib if total_memory_mib else 0.0
                    if candidate >= max_batch_size or pressure >= memory_fraction:
                        break
                    candidate = min(candidate * 2, max_batch_size)
                except tf.errors.ResourceExhaustedError:
                    tuning.append((candidate, "oom", math.nan, math.nan))
                    if last_success:
                        break
                    candidate //= 2
                    if candidate < 1:
                        raise
            selected = last_success
        else:
            infer(sample[None, :])
            tuning.append((selected, "fixed", math.nan, math.nan))
        output_queue.put(
            {
                "kind": "ready",
                "worker_id": worker_id,
                "gpu_index": gpu_index,
                "batch_size": selected,
                "tuning": tuning,
            }
        )

        active_seconds = 0.0
        tasks = 0
        batches = 0
        retries = 0
        current_batch = selected
        while True:
            message = input_queue.get()
            if message is None:
                break
            indices = list(message["indices"])
            audio = np.asarray(message["audio"], dtype=np.float32)
            position = 0
            failures = 0
            while position < len(indices):
                size = min(current_batch, len(indices) - position)
                try:
                    started = time.perf_counter()
                    embeddings = infer(audio[position : position + size])
                    elapsed = time.perf_counter() - started
                except tf.errors.ResourceExhaustedError as error:
                    failures += 1
                    retries += 1
                    if failures >= MAX_ATTEMPTS or current_batch == 1:
                        raise RuntimeError("OOM persistente no BirdNET.") from error
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
                "error": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            }
        )


def _reconstruct_ffmpeg(arguments: tuple[object, ...]) -> tuple[int, np.ndarray, int]:
    from codeco.ffmpeg import (
        SAMPLE_RATE as ffmpeg_sample_rate,
        _align_to_reference,
        _load_clip,
        _run_ffmpeg,
    )

    (
        row_index,
        path,
        start,
        codec,
        extension,
        encoder_arguments,
        executable,
        pcm_file,
        pcm_row,
        attempt,
    ) = arguments
    try:
        reference = (
            _load_clip(str(path), float(start))
            if pcm_file is None
            else load_pcm_clip(str(pcm_file), int(pcm_row))
        )
        if codec == "baseline":
            return int(row_index), np.asarray(reference, dtype=np.float32), int(attempt)
        with tempfile.TemporaryDirectory(prefix="codeco-birdnet-ffmpeg-") as directory:
            import soundfile as sf

            temporary = Path(directory)
            source = temporary / "source.wav"
            encoded = temporary / f"encoded.{extension}"
            decoded_path = temporary / "decoded.wav"
            sf.write(source, reference, ffmpeg_sample_rate, subtype="PCM_16")
            _run_ffmpeg(
                [
                    str(executable), "-nostdin", "-v", "error", "-y", "-i", str(source),
                    "-ac", "1", "-ar", str(ffmpeg_sample_rate), *encoder_arguments, str(encoded),
                ]
            )
            _run_ffmpeg(
                [
                    str(executable), "-nostdin", "-v", "error", "-y", "-i", str(encoded),
                    "-ar", str(ffmpeg_sample_rate), "-ac", "1", "-f", "wav", str(decoded_path),
                ]
            )
            decoded, _ = sf.read(decoded_path, dtype="float32", always_2d=False)
        if decoded.ndim > 1:
            decoded = decoded.mean(axis=1)
        aligned, _ = _align_to_reference(reference, decoded)
        return int(row_index), np.asarray(aligned, dtype=np.float32), int(attempt)
    except BaseException:
        if int(attempt) >= MAX_ATTEMPTS:
            raise
        retry = list(arguments)
        retry[9] = int(attempt) + 1
        return _reconstruct_ffmpeg(tuple(retry))


def _produce_ffmpeg(
    condition_id: str,
    condition: EmbeddingCondition,
    clips: pd.DataFrame,
    pending: Sequence[int],
    executable: Path,
    workers: int,
    producer_batch_size: int,
    embedding_queue: object,
    status_queue: queue.Queue,
) -> None:
    try:
        arguments = [
            (
                index,
                str(clips.iloc[index]["path"]),
                float(clips.iloc[index]["start"]),
                condition.codec,
                condition.extension,
                condition.encoder_arguments,
                str(executable),
                clips.iloc[index]["pcm_file"],
                clips.iloc[index]["pcm_row"],
                1,
            )
            for index in pending
        ]
        batch_indices: list[int] = []
        batch_audio: list[np.ndarray] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = iter(arguments)
            in_flight = set()
            for _ in range(max(1, workers * 2)):
                try:
                    in_flight.add(executor.submit(_reconstruct_ffmpeg, next(iterator)))
                except StopIteration:
                    break
            while in_flight:
                finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in finished:
                    index, audio, _ = future.result()
                    batch_indices.append(index)
                    batch_audio.append(audio)
                    if len(batch_indices) >= producer_batch_size:
                        embedding_queue.put(
                            {
                                "condition_id": condition_id,
                                "indices": batch_indices,
                                "audio": np.stack(batch_audio),
                            }
                        )
                        batch_indices, batch_audio = [], []
                    try:
                        in_flight.add(executor.submit(_reconstruct_ffmpeg, next(iterator)))
                    except StopIteration:
                        pass
        if batch_indices:
            embedding_queue.put(
                {
                    "condition_id": condition_id,
                    "indices": batch_indices,
                    "audio": np.stack(batch_audio),
                }
            )
        status_queue.put(("done", None))
    except BaseException as error:
        status_queue.put(
            ("error", "".join(traceback.format_exception(type(error), error, error.__traceback__)))
        )


def _encodec_worker(
    gpu_index: int | None,
    model_name: str,
    initial_batch_size: int,
    max_batch_size: int,
    memory_fraction: float,
    center_for_birdnet: bool,
    input_queue: object,
    output_queue: object,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu_index is None else str(gpu_index)
    try:
        import torch
        from transformers import EncodecModel
        from codeco.encodec import EvaluationTask, _align_to_reference, _encodec_chunk, _is_cuda_oom

        device = torch.device("cpu" if gpu_index is None else "cuda:0")
        if gpu_index is not None:
            torch.cuda.set_device(device)
        model = EncodecModel.from_pretrained(model_name).to(device).eval()
        codebook_bits = int(math.ceil(math.log2(model.config.codebook_size)))
        channels = int(model.config.audio_channels)
        output_queue.put({"kind": "ready", "gpu_index": gpu_index})
        batch_size = initial_batch_size
        while True:
            message = input_queue.get()
            if message is None:
                break
            records = message["records"]
            condition = message["condition"]
            position = 0
            outputs = []
            while position < len(records):
                size = min(batch_size, len(records) - position)
                failures = 0
                while True:
                    selected = records[position : position + size]
                    tasks = [
                        EvaluationTask(
                            ordinal=int(record["row_index"]),
                            dataset=str(record["dataset"]),
                            clip_id=str(record["clip_id"]),
                            path=str(record["path"]),
                            start=float(record["start"]),
                            codec="encodec",
                            setting=str(condition["setting"]),
                            bandwidth_kbps=float(condition["bandwidth_kbps"]),
                            pcm_file=(
                                None
                                if record.get("pcm_file") is None
                                else str(record["pcm_file"])
                            ),
                            pcm_row=(
                                None
                                if record.get("pcm_row") is None
                                else int(record["pcm_row"])
                            ),
                        )
                        for record in selected
                    ]
                    try:
                        items, statistics = _encodec_chunk(
                            model, torch, device, tasks, codebook_bits, channels
                        )
                        for task, reference, decoded, _ in items:
                            aligned, _ = _align_to_reference(reference, decoded)
                            audio = _center_birdnet(aligned) if center_for_birdnet else aligned
                            outputs.append(
                                (task.ordinal, np.asarray(audio, dtype=np.float32))
                            )
                        peak = float(statistics["peak_bytes"])
                        free = float(statistics["free_before_bytes"])
                        if gpu_index is not None and peak < free * memory_fraction:
                            batch_size = min(max_batch_size, max(batch_size, size * 2))
                        position += size
                        break
                    except BaseException as error:
                        failures += 1
                        if gpu_index is not None and _is_cuda_oom(torch, error):
                            torch.cuda.empty_cache()
                            size = max(1, size // 2)
                            batch_size = size
                        if failures >= MAX_ATTEMPTS:
                            raise
            output_queue.put(
                {
                    "kind": "result",
                    "gpu_index": gpu_index,
                    "condition_id": message["condition_id"],
                    "items": outputs,
                    "batch_size": batch_size,
                }
            )
        output_queue.put({"kind": "stopped", "gpu_index": gpu_index})
    except BaseException as error:
        output_queue.put(
            {
                "kind": "fatal",
                "gpu_index": gpu_index,
                "error": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            }
        )


def _produce_encodec(
    condition_id: str,
    condition: EmbeddingCondition,
    clips: pd.DataFrame,
    pending: Sequence[int],
    pool: EncodecPool,
    producer_batch_size: int,
    embedding_queue: object,
    status_queue: queue.Queue,
) -> None:
    try:
        records = []
        for index in pending:
            row = clips.iloc[index]
            records.append(
                {
                    "row_index": index,
                    "dataset": row["dataset"],
                    "clip_id": row["clip_id"],
                    "path": row["path"],
                    "start": row["start"],
                    "pcm_file": row["pcm_file"],
                    "pcm_row": row["pcm_row"],
                }
            )
        jobs = [
            records[position : position + producer_batch_size]
            for position in range(0, len(records), producer_batch_size)
        ]
        for job in jobs:
            pool.input_queue.put(
                {
                    "condition_id": condition_id,
                    "condition": asdict(condition),
                    "records": job,
                }
            )
        completed_jobs = 0
        while completed_jobs < len(jobs):
            message = pool.output_queue.get(timeout=1800)
            if message["kind"] == "fatal":
                raise RuntimeError(message["error"])
            if message["kind"] != "result":
                raise RuntimeError(f"Mensagem ENCODEC inesperada: {message['kind']}")
            if message["condition_id"] != condition_id:
                raise RuntimeError("Resultado ENCODEC pertence a outra condicao.")
            indices = [item[0] for item in message["items"]]
            audio = np.stack([item[1] for item in message["items"]])
            embedding_queue.put(
                {"condition_id": condition_id, "indices": indices, "audio": audio}
            )
            completed_jobs += 1
        status_queue.put(("done", None))
    except BaseException as error:
        status_queue.put(
            ("error", "".join(traceback.format_exception(type(error), error, error.__traceback__)))
        )


def _start_encodec_pool(
    gpu_indices: Sequence[int | None],
    initial_batch_size: int,
    max_batch_size: int,
    memory_fraction: float,
    center_for_birdnet: bool = True,
) -> EncodecPool:
    context = mp.get_context("spawn")
    input_queue = context.Queue(maxsize=max(2, len(gpu_indices) * 2))
    output_queue = context.Queue()
    workers = [
        context.Process(
            target=_encodec_worker,
            args=(
                index,
                "facebook/encodec_48khz",
                initial_batch_size,
                max_batch_size,
                memory_fraction,
                center_for_birdnet,
                input_queue,
                output_queue,
            ),
            name=f"BirdNET-ENCODEC-{'CPU' if index is None else index}",
        )
        for index in gpu_indices
    ]
    try:
        for worker in workers:
            worker.start()
        ready = 0
        while ready < len(workers):
            message = output_queue.get(timeout=600)
            if message["kind"] == "fatal":
                raise RuntimeError(message["error"])
            if message["kind"] != "ready":
                raise RuntimeError(f"Mensagem ENCODEC inesperada: {message['kind']}")
            ready += 1
            label = "CPU" if message["gpu_index"] is None else f"GPU {message['gpu_index']}"
            LOGGER.info("Worker ENCODEC %s pronto; modelo persistente carregado.", label)
        return EncodecPool(input_queue, output_queue, workers)
    except BaseException:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=10)
        input_queue.close()
        input_queue.join_thread()
        output_queue.close()
        output_queue.join_thread()
        raise


def _encodec_pool_parameters(
    gpu_indices: Sequence[int],
    initial_batch_size: int,
    max_batch_size: int,
) -> tuple[list[int | None], int, int]:
    """Use throughput batches on GPU and bounded one-clip jobs on CPU."""
    if gpu_indices:
        return list(gpu_indices), initial_batch_size, max_batch_size
    return [None], 1, 1


def _stop_encodec_pool(pool: EncodecPool) -> None:
    for _ in pool.workers:
        pool.input_queue.put(None)
    stopped = 0
    deadline = time.monotonic() + 120
    while stopped < len(pool.workers) and time.monotonic() < deadline:
        try:
            message = pool.output_queue.get(timeout=1)
        except queue.Empty:
            continue
        if message["kind"] == "fatal":
            LOGGER.error("Falha ao encerrar worker ENCODEC: %s", message["error"])
        elif message["kind"] == "stopped":
            stopped += 1
    for worker in pool.workers:
        worker.join(timeout=30)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=10)
    pool.input_queue.close()
    pool.input_queue.join_thread()
    pool.output_queue.close()
    pool.output_queue.join_thread()


def _connect_checkpoint(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS birdnet_runs (
            fingerprint TEXT PRIMARY KEY,
            metadata_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS birdnet_embeddings (
            fingerprint TEXT NOT NULL,
            dataset TEXT NOT NULL,
            condition_id TEXT NOT NULL,
            clip_id TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (fingerprint, dataset, condition_id, clip_id)
        );
        """
    )
    return connection


def _select_clips(
    baseline: BaselineState,
    datasets: Sequence[str] | None,
    max_per_dataset: int | None,
    seed: int,
) -> pd.DataFrame:
    table = baseline.table.to_pandas()
    if datasets is not None:
        table = table[table["dataset"].isin(datasets)].copy()
    available = table[table["audio_available"].astype(bool)].copy()
    groups = []
    for dataset, group in available.groupby("dataset", sort=False):
        if max_per_dataset is not None and len(group) > max_per_dataset:
            group = _stratified_benchmark_sample(
                group.reset_index(drop=True), max_per_dataset, seed, str(dataset)
            )
        groups.append(group.sort_values(["split", "clip_id"]))
    if not groups:
        raise ValueError("Nenhum clipe de audio disponivel.")
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


def _split_quotas(group: pd.DataFrame, total: int) -> dict[str, int]:
    counts = group["split"].value_counts(sort=False)
    exact = counts.astype(float) * total / len(group)
    quotas = np.floor(exact).astype(int)
    remaining = total - int(quotas.sum())
    order = sorted(counts.index, key=lambda split: (-(exact[split] - quotas[split]), str(split)))
    for split in order[:remaining]:
        quotas[split] += 1
    return {str(split): int(quotas[split]) for split in counts.index}


def _stratified_split_sample(
    group: pd.DataFrame,
    target: int,
    classes: Sequence[str],
    seed: int,
    dataset: str,
    split: str,
) -> pd.DataFrame:
    label_lists = [
        value.split("|") if isinstance(value, str) and value else []
        for value in group["labels"]
    ]
    binarizer = MultiLabelBinarizer(classes=list(classes))
    labels = binarizer.fit_transform(label_lists).astype(np.int8, copy=False)
    positives = labels.sum(axis=0)
    invalid_source = [classes[index] for index, count in enumerate(positives) if count < 2]
    if invalid_source:
        raise ValueError(
            f"{dataset}/{split}: o split oficial possui menos de 2 positivos para: "
            + ", ".join(invalid_source)
        )

    rng = np.random.RandomState(seed)
    rank = np.empty(len(group), dtype=np.int64)
    rank[rng.permutation(len(group))] = np.arange(len(group))
    selected: set[int] = set()
    selected_counts = np.zeros(labels.shape[1], dtype=np.int64)
    while np.any(selected_counts < 2):
        deficits = selected_counts < 2
        candidates = np.flatnonzero(
            (~np.isin(np.arange(len(group)), list(selected)))
            & np.any(labels[:, deficits] > 0, axis=1)
        )
        if not len(candidates) or len(selected) >= target:
            missing = [classes[index] for index in np.flatnonzero(deficits)]
            raise ValueError(
                f"Amostra {dataset}/{split} com {target} clipes nao comporta "
                f"2 positivos de cada classe: {', '.join(missing)}"
            )
        weights = 1.0 / np.maximum(positives, 1)
        scores = (labels[candidates][:, deficits] * weights[deficits]).sum(axis=1)
        best_score = scores.max()
        tied = candidates[np.isclose(scores, best_score)]
        chosen = int(tied[np.argmin(rank[tied])])
        selected.add(chosen)
        selected_counts += labels[chosen]

    remaining = target - len(selected)
    if remaining:
        candidates = np.array(
            [index for index in range(len(group)) if index not in selected], dtype=int
        )
        if remaining >= len(candidates):
            selected.update(int(index) for index in candidates)
        else:
            splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=remaining,
                random_state=seed,
            )
            _, chosen = next(splitter.split(np.zeros(len(candidates)), labels[candidates]))
            selected.update(int(candidates[index]) for index in chosen)

    if len(selected) < target:
        candidates = np.array(
            [index for index in range(len(group)) if index not in selected], dtype=int
        )
        ordered = candidates[np.argsort(rank[candidates], kind="stable")]
        selected.update(int(index) for index in ordered[: target - len(selected)])
    if len(selected) != target:
        raise RuntimeError(
            f"Amostragem {dataset}/{split} retornou {len(selected)} de {target} clipes."
        )

    result = group.iloc[sorted(selected)].copy()
    sampled_labels = labels[sorted(selected)]
    sampled_positives = sampled_labels.sum(axis=0)
    sampled_negatives = len(result) - sampled_positives
    invalid = [
        classes[index]
        for index in range(len(classes))
        if sampled_positives[index] < 2 or sampled_negatives[index] < 2
    ]
    if invalid:
        raise RuntimeError(
            f"Amostra estratificada invalida em {dataset}/{split}: " + ", ".join(invalid)
        )
    return result


def _stratified_benchmark_sample(
    group: pd.DataFrame,
    total: int,
    seed: int,
    dataset: str,
) -> pd.DataFrame:
    classes = sorted(
        {
            label
            for value in group["labels"]
            for label in (value.split("|") if isinstance(value, str) and value else [])
        }
    )
    quotas = _split_quotas(group, total)
    samples = []
    for number, (split, quota) in enumerate(quotas.items()):
        split_group = group.loc[group["split"] == split].reset_index(drop=True)
        samples.append(
            _stratified_split_sample(
                split_group, quota, classes, seed + number, dataset, split
            )
        )
    return pd.concat(samples, ignore_index=True)


def _prepare_embedding_file(
    path: Path, rows: int, embedding_dimension: int = EMBEDDING_DIMENSION
) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        array = np.load(path, mmap_mode="r+")
        if array.shape != (rows, embedding_dimension) or array.dtype != np.float32:
            raise RuntimeError(f"Cache de embeddings incompativel: {path}")
        return array
    return open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(rows, embedding_dimension),
    )


def _completed_indices(
    connection: sqlite3.Connection,
    fingerprint: str,
    dataset: str,
    condition_id: str,
) -> set[int]:
    rows = connection.execute(
        """SELECT row_index FROM birdnet_embeddings
           WHERE fingerprint = ? AND dataset = ? AND condition_id = ? AND status = 'completed'""",
        (fingerprint, dataset, condition_id),
    )
    return {int(row[0]) for row in rows}


def _checkpoint_batch(
    connection: sqlite3.Connection,
    fingerprint: str,
    dataset: str,
    condition_id: str,
    clips: pd.DataFrame,
    indices: Sequence[int],
) -> None:
    now = _utc_now()
    with connection:
        connection.executemany(
            """INSERT INTO birdnet_embeddings
               (fingerprint, dataset, condition_id, clip_id, row_index, status, updated_at)
               VALUES (?, ?, ?, ?, ?, 'completed', ?)
               ON CONFLICT(fingerprint, dataset, condition_id, clip_id) DO UPDATE SET
                 row_index = excluded.row_index, status = 'completed', updated_at = excluded.updated_at""",
            [
                (
                    fingerprint,
                    dataset,
                    condition_id,
                    str(clips.iloc[index]["clip_id"]),
                    index,
                    now,
                )
                for index in indices
            ],
        )
        connection.execute(
            "UPDATE birdnet_runs SET updated_at = ? WHERE fingerprint = ?",
            (now, fingerprint),
        )


def _process_condition(
    connection: sqlite3.Connection,
    fingerprint: str,
    dataset: str,
    clips: pd.DataFrame,
    condition: EmbeddingCondition,
    embedding_file: Path,
    ffmpeg: tuple[Path, Path],
    cpu_workers: int,
    encodec_pool: EncodecPool | None,
    embedding_queue: object,
    embedding_output: object,
    producer_batch_size: int,
    checkpoint_size: int = CHECKPOINT_SIZE,
    embedding_dimension: int = EMBEDDING_DIMENSION,
) -> tuple[int, int, int, float]:
    condition_started = time.perf_counter()
    condition_id = f"{dataset}:{condition.identifier}"
    completed = _completed_indices(connection, fingerprint, dataset, condition.identifier)
    pending = [index for index in range(len(clips)) if index not in completed]
    embeddings = _prepare_embedding_file(
        embedding_file, len(clips), embedding_dimension
    )
    if not pending:
        LOGGER.info("[CACHE HIT] %s | %s: %d embeddings.", dataset, condition.setting, len(clips))
        return len(clips), len(completed), 0, time.perf_counter() - condition_started

    LOGGER.info(
        "%s | %s: %d retomados; %d pendentes.",
        dataset,
        condition.setting,
        len(completed),
        len(pending),
    )
    status: queue.Queue = queue.Queue()
    if condition.kind == "ffmpeg":
        target = _produce_ffmpeg
        arguments = (
            condition_id,
            condition,
            clips,
            pending,
            ffmpeg[1] if condition.codec == "lc3" else ffmpeg[0],
            cpu_workers,
            producer_batch_size,
            embedding_queue,
            status,
        )
    else:
        if encodec_pool is None:
            raise RuntimeError("Pool ENCODEC persistente nao foi inicializado.")
        target = _produce_encodec
        arguments = (
            condition_id,
            condition,
            clips,
            pending,
            encodec_pool,
            producer_batch_size,
            embedding_queue,
            status,
        )
    producer = threading.Thread(target=target, args=arguments, daemon=True)
    producer.start()
    buffer: list[int] = []
    received = 0
    progress = tqdm(
        total=len(pending),
        desc=f"{dataset} | {condition.codec}",
        unit="embedding",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    try:
        while received < len(pending):
            try:
                message = embedding_output.get(timeout=1)
            except queue.Empty:
                if not producer.is_alive() and not status.empty():
                    kind, detail = status.get_nowait()
                    if kind == "error":
                        raise RuntimeError(detail)
                continue
            if message["kind"] == "fatal":
                raise RuntimeError(message["error"])
            if message["kind"] != "result":
                continue
            if message["condition_id"] != condition_id:
                raise RuntimeError("Resultado de embedding pertence a outra condicao.")
            indices = [int(index) for index in message["indices"]]
            embeddings[indices] = np.asarray(message["embeddings"], dtype=np.float32)
            buffer.extend(indices)
            received += len(indices)
            progress.update(len(indices))
            while len(buffer) >= checkpoint_size:
                embeddings.flush()
                _checkpoint_batch(
                    connection,
                    fingerprint,
                    dataset,
                    condition.identifier,
                    clips,
                    buffer[:checkpoint_size],
                )
                buffer = buffer[checkpoint_size:]
        producer.join(timeout=30)
        if not status.empty():
            kind, detail = status.get_nowait()
            if kind == "error":
                raise RuntimeError(detail)
        if buffer:
            embeddings.flush()
            _checkpoint_batch(
                connection, fingerprint, dataset, condition.identifier, clips, buffer
            )
    finally:
        progress.close()
    return (
        len(completed) + received,
        len(completed),
        received,
        time.perf_counter() - condition_started,
    )


def _write_index_csv(
    records: list[dict[str, object]] | pd.DataFrame,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        table = records if isinstance(records, pd.DataFrame) else pd.DataFrame.from_records(records)
        table.to_csv(temporary, index=False)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_index_parquet(table: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        table.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(
    datasets: Sequence[str] | None = None,
    codecs: Sequence[str] | None = None,
    max_clips_per_dataset: int | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    gpu_memory_fraction: float = DEFAULT_GPU_MEMORY_FRACTION,
    encodec_memory_fraction: float = DEFAULT_ENCODEC_MEMORY_FRACTION,
    checkpoint_path: Path | None = None,
    force: bool = False,
    baseline: BaselineState | None = None,
    environment: EnvironmentState | None = None,
    artifact_directory: Path | None = None,
) -> Path:
    from codeco.ffmpeg import _ffmpeg_installation

    paths = CodecoPath.discover()
    configure_model_environment(paths)
    _configure_cuda_environment()
    os.environ.setdefault("MPLCONFIGDIR", str(paths.cache / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    os.environ["BIRDNET_APP_DATA"] = str(paths.models / "birdnet")
    os.environ["BIRDNET_START_METHOD"] = "spawn"
    selected_artifact_directory = (
        paths.tests / "embeddings"
        if baseline is None and artifact_directory is None
        else artifact_directory
    )
    if selected_artifact_directory is not None:
        selected_artifact_directory.mkdir(parents=True, exist_ok=True)
    log_file = _configure_logging(
        paths,
        (
            selected_artifact_directory / "birdnet.log"
            if selected_artifact_directory is not None
            else None
        ),
    )
    _header()
    checkpoint = checkpoint_path or (
        selected_artifact_directory / "checkpoints.sqlite3"
        if selected_artifact_directory is not None
        else paths.cache / "checkpoints.sqlite3"
    )
    output = (
        selected_artifact_directory / "birdnet_embeddings.csv"
        if selected_artifact_directory is not None
        else paths.tables / "birdnet_embeddings.csv"
    )
    cache_root = (
        selected_artifact_directory / "birdnet_cache"
        if selected_artifact_directory is not None
        else paths.cache / "birdnet_embeddings"
    )
    selected_baseline = baseline or load_standalone_examples()
    clips = _select_clips(
        selected_baseline,
        datasets,
        max_clips_per_dataset,
        seed,
    )
    if environment is None:
        ffmpeg_system, ffmpeg_lc3, ffmpeg_version = _ffmpeg_installation()
    else:
        ffmpeg_system = environment.system.ffmpeg.executable
        ffmpeg_lc3 = environment.system.ffmpeg.lc3_executable
        ffmpeg_version = (
            f"system={environment.system.ffmpeg.version}; "
            f"lc3={environment.system.ffmpeg.lc3_version}"
        )
    ffmpeg = (ffmpeg_system, ffmpeg_lc3)
    cpu_workers = max(1, psutil.cpu_count(logical=False) or 1)
    gpu_count = (
        _tensorflow_gpu_count()
        if environment is None
        else environment.system.tensorflow_gpu_count
    )
    gpu_indices = list(range(gpu_count))
    selected_conditions = tuple(
        condition for condition in CONDITIONS
        if codecs is None or condition.codec in set(codecs)
    )
    if not selected_conditions:
        raise ValueError("Nenhuma condicao selecionada.")

    import birdnet

    model = birdnet.load("acoustic", MODEL_VERSION, "pb")
    sample_row = clips.iloc[0]
    from codeco.ffmpeg import _load_clip

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
        "implementation_version": IMPLEMENTATION_VERSION,
        "manifest_sha256": selected_baseline.manifest_sha256,
        "selection": selection.to_dict("records"),
        "conditions": [asdict(condition) for condition in selected_conditions],
        "model_version": MODEL_VERSION,
        "seed": seed,
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
    }
    fingerprint = _sha256_json(metadata)

    _section("Configuracao")
    LOGGER.info("Log: %s", log_file)
    LOGGER.info("Checkpoint: %s", checkpoint)
    LOGGER.info("Indice CSV: %s", output)
    LOGGER.info("Modelo: BirdNET %s ProtoBuf FP32", MODEL_VERSION)
    LOGGER.info("FFmpeg: %s", ffmpeg_version)
    LOGGER.info("Nucleos fisicos: %d", cpu_workers)
    if max_clips_per_dataset is None:
        LOGGER.info("Selecao: corpus completo congelado; nenhuma amostragem aplicada.")
    else:
        LOGGER.info(
            "Selecao: benchmark estratificado; ate %d clipes/dataset; seed=%d.",
            max_clips_per_dataset,
            seed,
        )
        for dataset, selected in clips.groupby("dataset", sort=False):
            counts = selected["split"].value_counts().to_dict()
            LOGGER.info(
                "%s: %d clipes | train=%d | val=%d | test=%d",
                dataset,
                len(selected),
                counts.get("train", 0),
                counts.get("val", 0),
                counts.get("test", 0),
            )
    if gpu_indices:
        LOGGER.info(
            "GPUs TensorFlow: %d; microbatch=%d; maximo=%d; alvo VRAM=%.0f%%",
            len(gpu_indices),
            batch_size,
            max_batch_size,
            gpu_memory_fraction * 100,
        )
    else:
        LOGGER.warning(
            "Nenhuma GPU TensorFlow: fallback ProtoBuf/CPU em batch %d.", batch_size
        )

    context = mp.get_context("spawn")
    embedding_queue = context.Queue(maxsize=max(2, len(gpu_indices) * 2 or 2))
    embedding_output = context.Queue()
    memory_totals = _gpu_memory_totals()
    worker_specs = (
        [(index, index) for index in gpu_indices]
        if gpu_indices
        else [(0, None)]
    )
    workers = [
        context.Process(
            target=_embedding_worker,
            args=(
                worker_id,
                gpu_index,
                str(model.model_path),
                sample,
                batch_size,
                max_batch_size,
                gpu_memory_fraction,
                memory_totals.get(gpu_index) if gpu_index is not None else None,
                embedding_queue,
                embedding_output,
            ),
            name=f"BirdNET-{'GPU-' + str(gpu_index) if gpu_index is not None else 'CPU'}",
        )
        for worker_id, gpu_index in worker_specs
    ]
    for worker in workers:
        worker.start()
    ready = 0
    while ready < len(workers):
        message = embedding_output.get(timeout=1800)
        if message["kind"] == "fatal":
            raise RuntimeError(message["error"])
        ready += 1
        LOGGER.info(
            "Worker %d pronto: %s; batch=%d; ajuste=%s",
            message["worker_id"],
            "CPU" if message["gpu_index"] is None else f"GPU {message['gpu_index']}",
            message["batch_size"],
            " -> ".join(f"{row[0]} ({row[1]})" for row in message["tuning"]),
        )

    connection = _connect_checkpoint(checkpoint)
    records = []
    started = time.perf_counter()
    producer_batch_size = max_batch_size if gpu_indices else batch_size
    encodec_workers, encodec_initial_batch, encodec_max_batch = (
        _encodec_pool_parameters(gpu_indices, batch_size, max_batch_size)
    )
    encodec_job_size = producer_batch_size if gpu_indices else 1
    encodec_checkpoint_size = CHECKPOINT_SIZE if gpu_indices else 1
    encodec_pool: EncodecPool | None = None
    try:
        now = _utc_now()
        with connection:
            if force:
                connection.execute(
                    "DELETE FROM birdnet_embeddings WHERE fingerprint = ?", (fingerprint,)
                )
                connection.execute("DELETE FROM birdnet_runs WHERE fingerprint = ?", (fingerprint,))
            connection.execute(
                """INSERT INTO birdnet_runs
                   (fingerprint, metadata_json, status, created_at, updated_at)
                   VALUES (?, ?, 'running', ?, ?)
                   ON CONFLICT(fingerprint) DO UPDATE SET status='running', updated_at=excluded.updated_at""",
                (fingerprint, json.dumps(metadata, ensure_ascii=False), now, now),
            )
        dataset_tables: dict[str, pd.DataFrame] = {}
        clips_files: dict[str, Path] = {}
        for dataset, dataset_clips in clips.groupby("dataset", sort=False):
            dataset_clips = dataset_clips.reset_index(drop=True)
            clips_file = cache_root / fingerprint / str(dataset) / "clips.parquet"
            clip_columns = [
                "dataset",
                "clip_id",
                "path",
                "start",
                "end",
                "labels",
                "split",
                "source_group_id",
            ]
            clip_index = dataset_clips[clip_columns].copy()
            clip_index["path"] = clip_index["path"].map(paths.project_relative)
            clip_index.insert(0, "row_index", np.arange(len(clip_index), dtype=np.int64))
            _write_index_parquet(clip_index, clips_file)
            dataset_tables[str(dataset)] = dataset_clips
            clips_files[str(dataset)] = clips_file

        from codeco.embedding_cache import seed_compatible_embeddings

        imported = 0
        if not force:
            imported = seed_compatible_embeddings(
                paths=paths,
                model="birdnet",
                model_version=MODEL_VERSION,
                fingerprint=fingerprint,
                datasets=dataset_tables,
                conditions=selected_conditions,
                cache_root=cache_root,
                connection=connection,
                logger=LOGGER,
            )
        if imported:
            LOGGER.info(
                "%d caches compatíveis importados de índices anteriores/handoffs.",
                imported,
            )

        processing_conditions = tuple(
            condition for condition in selected_conditions if condition.kind != "encodec"
        ) + tuple(
            condition for condition in selected_conditions if condition.kind == "encodec"
        )
        for number, condition in enumerate(processing_conditions, 1):
            _section(
                f"Condicao {number}/{len(processing_conditions)}: "
                f"{condition.codec.upper()} | {condition.setting}"
            )
            if condition.kind == "encodec" and encodec_pool is None:
                needs_encodec = any(
                    len(
                        _completed_indices(
                            connection,
                            fingerprint,
                            dataset,
                            condition.identifier,
                        )
                    )
                    < len(dataset_clips)
                    for dataset, dataset_clips in dataset_tables.items()
                )
                if needs_encodec:
                    LOGGER.info(
                        "Iniciando EnCodec persistente em %s | workers=%d | "
                        "batch=%d | checkpoint=%d.",
                        "GPU" if gpu_indices else "CPU (um clipe por lote)",
                        len(encodec_workers),
                        encodec_initial_batch,
                        encodec_checkpoint_size,
                    )
                    encodec_pool = _start_encodec_pool(
                        encodec_workers,
                        encodec_initial_batch,
                        encodec_max_batch,
                        encodec_memory_fraction,
                    )

            for dataset, dataset_clips in dataset_tables.items():
                LOGGER.info(
                    "Dataset %s | %s | %s",
                    dataset,
                    condition.codec.upper(),
                    condition.setting,
                )
                embedding_file = cache_root / fingerprint / dataset / f"{condition.identifier}.npy"
                completed, resumed, processed, condition_seconds = _process_condition(
                    connection=connection,
                    fingerprint=fingerprint,
                    dataset=dataset,
                    clips=dataset_clips,
                    condition=condition,
                    embedding_file=embedding_file,
                    ffmpeg=ffmpeg,
                    cpu_workers=cpu_workers,
                    encodec_pool=encodec_pool,
                    embedding_queue=embedding_queue,
                    embedding_output=embedding_output,
                    producer_batch_size=(
                        encodec_job_size
                        if condition.kind == "encodec"
                        else producer_batch_size
                    ),
                    checkpoint_size=(
                        encodec_checkpoint_size
                        if condition.kind == "encodec"
                        else CHECKPOINT_SIZE
                    ),
                )
                records.append(
                    {
                        "dataset": dataset,
                        "foundation_model": "birdnet",
                        "model_version": MODEL_VERSION,
                        "codec": condition.codec,
                        "setting": condition.setting,
                        "clips": len(dataset_clips),
                        "completed": completed,
                        "resumed": resumed,
                        "processed": processed,
                        "seconds": condition_seconds,
                        "embeddings_per_second": processed / max(condition_seconds, 1e-12),
                        "embedding_dimension": EMBEDDING_DIMENSION,
                        "clips_file": paths.project_relative(clips_files[dataset]),
                        "embeddings_file": paths.project_relative(embedding_file),
                    }
                )
                _write_index_csv(records, output)
        with connection:
            connection.execute(
                "UPDATE birdnet_runs SET status='completed', updated_at=? WHERE fingerprint=?",
                (_utc_now(), fingerprint),
            )
    finally:
        if encodec_pool is not None:
            LOGGER.info("Encerrando pool ENCODEC persistente.")
            _stop_encodec_pool(encodec_pool)
        connection.close()
        for _ in workers:
            embedding_queue.put(None)
        stopped = 0
        worker_statistics = []
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

    _section("Execucao concluida")
    LOGGER.info("Condicoes: %d; indice: %s", len(records), output)
    for statistics in sorted(worker_statistics, key=lambda row: int(row["worker_id"])):
        LOGGER.info(
            "Worker %d | %s | batch %s->%s | tarefas=%d | lotes=%d | "
            "ativa=%.2f s | retries=%d | pico VRAM=%s",
            statistics["worker_id"],
            "CPU" if statistics["gpu_index"] is None else f"GPU {statistics['gpu_index']}",
            statistics["selected_batch"],
            statistics["final_batch"],
            statistics["tasks"],
            statistics["batches"],
            statistics["active_seconds"],
            statistics["retries"],
            "-" if statistics["peak_vram_mib"] is None else f"{statistics['peak_vram_mib']:.0f} MiB",
        )
    LOGGER.info("Tempo total: %.2f s", time.perf_counter() - started)
    return output


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera embeddings BirdNET 2.4 com streaming, checkpoint e batch adaptativo."
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument(
        "--codecs",
        nargs="+",
        choices=sorted({condition.codec for condition in CONDITIONS}),
        help="Filtro operacional; padrao: todas as condicoes.",
    )
    parser.add_argument(
        "--max-clips-per-dataset",
        type=int,
        default=0,
        help="0 processa todos os clipes.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-batch-size", type=int, default=DEFAULT_MAX_BATCH_SIZE)
    parser.add_argument(
        "--gpu-memory-fraction", type=float, default=DEFAULT_GPU_MEMORY_FRACTION
    )
    parser.add_argument(
        "--encodec-memory-fraction", type=float, default=DEFAULT_ENCODEC_MEMORY_FRACTION
    )
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
        datasets=arguments.datasets,
        codecs=arguments.codecs,
        max_clips_per_dataset=(
            None if arguments.max_clips_per_dataset == 0 else arguments.max_clips_per_dataset
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
        LOGGER.error("Processamento interrompido pelo usuario.")
        raise SystemExit(130) from None
    except Exception:
        LOGGER.exception("Falha no pipeline BirdNET.")
        raise SystemExit(1) from None
