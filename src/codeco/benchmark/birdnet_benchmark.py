"""Benchmark BirdNET 2.4 TFLite/CPU against ProtoBuf on CPU and GPU.

This module is intentionally isolated from the classification pipeline.  It
answers two questions before that pipeline is changed: how much faster the
batched ProtoBuf backend is, and whether it preserves the 1024-dimensional
embeddings produced by the legacy TFLite loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import psutil
import soundfile as sf
from tqdm import tqdm

from codeco.codeco_path import CodecoPath
from codeco.environment import configure_model_environment
from codeco.protocol import PROTOCOL
from codeco.prepare_table import TqdmLoggingHandler, prepare_table


LOGGER = logging.getLogger("codeco.birdnet_benchmark")
SAMPLE_RATE = PROTOCOL.sample_rate_hz
CLIP_SECONDS = PROTOCOL.window_seconds
MODEL_SECONDS = PROTOCOL.birdnet_seconds
CLIP_SAMPLES = round(SAMPLE_RATE * CLIP_SECONDS)
MODEL_SAMPLES = round(SAMPLE_RATE * MODEL_SECONDS)
DEFAULT_CLIPS_PER_DATASET = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_BATCH_SIZE = 512
DEFAULT_GPU_MEMORY_FRACTION = 0.85
DEFAULT_SEED = 42
EMBEDDING_DIMENSION = 1024


@dataclass(frozen=True)
class RunResult:
    strategy: str
    backend: str
    device: str
    workers: int
    batch_size: int | str
    setup_seconds: float
    processing_seconds: float
    peak_ram_mib: float
    peak_vram_mib: float | None
    embeddings: np.ndarray
    gpu_statistics: tuple[dict[str, object], ...] = ()


class ResourceMonitor:
    """Sample aggregate process RAM and NVIDIA VRAM during one strategy."""

    def __init__(self, gpu_indices: Sequence[int] = ()) -> None:
        self._gpu_indices = tuple(gpu_indices)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_ram_mib = 0.0
        self.peak_vram_mib: float | None = None

    def _sample_ram(self) -> None:
        process = psutil.Process()
        processes = [process]
        try:
            processes.extend(process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        total = 0
        for item in processes:
            try:
                total += item.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.peak_ram_mib = max(self.peak_ram_mib, total / 1024**2)

    def _sample_vram(self) -> None:
        if not self._gpu_indices:
            return
        try:
            process = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return
        selected = set(self._gpu_indices)
        total = 0.0
        for line in process.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2:
                continue
            try:
                index = int(fields[0])
                used = float(fields[1])
            except ValueError:
                continue
            if index in selected:
                total += used
        self.peak_vram_mib = max(self.peak_vram_mib or 0.0, total)

    def _run(self) -> None:
        while not self._stop.wait(0.1):
            self._sample_ram()
            self._sample_vram()
        self._sample_ram()
        self._sample_vram()

    def __enter__(self) -> "ResourceMonitor":
        self._sample_ram()
        self._sample_vram()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)


def _configure_logging(paths: CodecoPath) -> Path:
    log_file = paths.logs / "birdnet_benchmark.log"
    if not LOGGER.handlers:
        paths.logs.mkdir(parents=True, exist_ok=True)
        terminal = TqdmLoggingHandler(sys.stdout)
        terminal.setFormatter(logging.Formatter("%(message)s"))
        persistent = logging.FileHandler(log_file, encoding="utf-8")
        persistent.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
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
        "\n%s\nBENCHMARK BIRDNET 2.4: TFLITE X TENSORFLOW PROTOBUF\n[CMD]   %s\n%s",
        line,
        command,
        line,
    )


def _section(title: str) -> None:
    LOGGER.info("\n%s\n%s", title, "-" * 72)


def _log_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    if not rows:
        return
    rendered = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    LOGGER.info(
        "  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers))
    )
    LOGGER.info("  ".join("-" * width for width in widths))
    for row in rendered:
        LOGGER.info("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _atomic_csv(table: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
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


def _atomic_embeddings(output: Path, clip_ids: np.ndarray, embeddings: np.ndarray) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(temporary_path, clip_ids=clip_ids, embeddings=embeddings)
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _select_clips(clips_per_dataset: int, split: str, seed: int) -> pd.DataFrame:
    table = prepare_table().to_pandas()
    available = table[
        (table["split"] == split) & table["audio_available"].astype(bool)
    ].copy()
    if available.empty:
        raise ValueError(f"Nenhum clipe de audio disponivel no split {split!r}.")
    groups = []
    for _, group in available.groupby("dataset", sort=False):
        if len(group) > clips_per_dataset:
            group = group.sample(clips_per_dataset, random_state=seed)
        groups.append(group.sort_values("clip_id"))
    return pd.concat(groups, ignore_index=True)


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
    start_sample = (len(audio) - MODEL_SAMPLES) // 2
    return audio[start_sample : start_sample + MODEL_SAMPLES].astype(np.float32)


def _load_audio(clips: pd.DataFrame) -> list[np.ndarray]:
    return [
        _load_clip(str(row.path), float(row.start))
        for row in tqdm(
            clips.itertuples(index=False),
            total=len(clips),
            desc="Carregando clipes",
            unit="clipe",
            dynamic_ncols=True,
            file=sys.stdout,
        )
    ]


def _configure_cuda_environment() -> list[Path]:
    """Expose CUDA libraries installed by ``tensorflow[and-cuda]`` to workers."""
    site_packages = Path(sysconfig.get_paths()["purelib"])
    nvidia_root = site_packages / "nvidia"
    library_dirs = sorted(path for path in nvidia_root.glob("*/lib") if path.is_dir())
    if library_dirs:
        existing = [value for value in os.environ.get("LD_LIBRARY_PATH", "").split(":") if value]
        additions = [str(path) for path in library_dirs if str(path) not in existing]
        os.environ["LD_LIBRARY_PATH"] = ":".join([*additions, *existing])

    nvcc_bin = nvidia_root / "cuda_nvcc" / "bin"
    if nvcc_bin.is_dir():
        existing_path = [value for value in os.environ.get("PATH", "").split(":") if value]
        if str(nvcc_bin) not in existing_path:
            os.environ["PATH"] = ":".join([str(nvcc_bin), *existing_path])
    return library_dirs


def _tensorflow_gpus() -> list[dict[str, object]]:
    probe = (
        "import json, tensorflow as tf; "
        "print('CODECO_GPU_JSON=' + json.dumps(["
        "{'index': i, 'name': d.name} for i, d in enumerate("
        "tf.config.list_physical_devices('GPU'))]))"
    )
    environment = os.environ.copy()
    environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    process = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=environment,
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        LOGGER.warning("Nao foi possivel consultar GPUs do TensorFlow: %s", detail)
        return []
    for line in reversed(process.stdout.splitlines()):
        if line.startswith("CODECO_GPU_JSON="):
            value = json.loads(line.removeprefix("CODECO_GPU_JSON="))
            return list(value)
    return []


def _gpu_memory_totals() -> dict[int, float]:
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}
    totals = {}
    for line in process.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            totals[int(fields[0])] = float(fields[1])
        except ValueError:
            continue
    return totals


def _legacy_tflite(audio: list[np.ndarray]) -> RunResult:
    setup_started = time.perf_counter()
    paths = CodecoPath.discover()
    os.environ.setdefault("MPLCONFIGDIR", str(paths.cache / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    from birdnetlib.analyzer import Analyzer
    import tensorflow as tf

    analyzer = Analyzer(version="2.4")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"tensorflow\.lite\.python\.interpreter",
        )
        interpreter = tf.lite.Interpreter(
            model_path=analyzer.model_path,
            num_threads=1,
            experimental_preserve_all_tensors=True,
        )
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    embedding_tensors = [
        tensor
        for tensor in interpreter.get_tensor_details()
        if tensor["name"].endswith("GLOBAL_AVG_POOL/Mean")
        and tensor["shape"][-1] == EMBEDDING_DIMENSION
    ]
    if len(embedding_tensors) != 1:
        raise RuntimeError(
            "Nao foi possivel identificar unicamente o tensor de embeddings do BirdNET."
        )
    embedding_index = embedding_tensors[0]["index"]
    interpreter.set_tensor(input_index, audio[0][None, :])
    interpreter.invoke()
    setup_seconds = time.perf_counter() - setup_started

    embeddings = []
    started = time.perf_counter()
    with ResourceMonitor() as resources:
        for clip in tqdm(
            audio,
            desc="TFLite CPU sequencial",
            unit="clipe",
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            interpreter.set_tensor(input_index, clip[None, :])
            interpreter.invoke()
            embeddings.append(
                interpreter.get_tensor(embedding_index)[0].astype(np.float32)
            )
    processing_seconds = time.perf_counter() - started
    return RunResult(
        strategy="Legado: TFLite sequencial",
        backend="TFLite",
        device="CPU",
        workers=1,
        batch_size=1,
        setup_seconds=setup_seconds,
        processing_seconds=processing_seconds,
        peak_ram_mib=resources.peak_ram_mib,
        peak_vram_mib=None,
        embeddings=np.vstack(embeddings),
    )


def _protobuf(
    audio: list[np.ndarray],
    batch_size: int,
    devices: list[str],
    gpu_indices: Sequence[int] = (),
) -> RunResult:
    os.environ.setdefault(
        "BIRDNET_APP_DATA",
        str(CodecoPath.discover().cache / "birdnet"),
    )
    setup_started = time.perf_counter()
    import birdnet

    model = birdnet.load("acoustic", "2.4", "pb")
    inputs = [(clip, SAMPLE_RATE) for clip in audio]
    label = "ProtoBuf " + "+".join(devices)
    progress = tqdm(
        total=len(inputs),
        desc=label,
        unit="clipe",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    def update(stats: object) -> None:
        processed = min(len(inputs), int(getattr(stats, "processed_segments", 0)))
        progress.update(max(0, processed - progress.n))

    with model.encode_session(
        n_producers=1,
        n_workers=len(devices),
        batch_size=batch_size,
        prefetch_ratio=2,
        bandpass_fmin=0,
        bandpass_fmax=15_000,
        show_stats="progress",
        progress_callback=update,
        device=devices,
        max_n_files=len(inputs),
    ) as session:
        session.run_arrays([inputs[0]])
        setup_seconds = time.perf_counter() - setup_started
        started = time.perf_counter()
        with ResourceMonitor(gpu_indices) as resources:
            result = session.run_arrays(inputs)
        processing_seconds = time.perf_counter() - started
    progress.update(len(inputs) - progress.n)
    progress.close()

    if np.asarray(result.unprocessable_inputs()).any():
        raise RuntimeError("O backend ProtoBuf encontrou entradas nao processaveis.")
    embeddings = np.asarray(result.embeddings, dtype=np.float32)
    if embeddings.shape != (len(audio), 1, EMBEDDING_DIMENSION):
        raise RuntimeError(f"Forma inesperada dos embeddings ProtoBuf: {embeddings.shape}")
    device_label = ",".join(devices)
    return RunResult(
        strategy=f"Atual: ProtoBuf batch ({device_label})",
        backend="ProtoBuf",
        device=device_label,
        workers=len(devices),
        batch_size=batch_size,
        setup_seconds=setup_seconds,
        processing_seconds=processing_seconds,
        peak_ram_mib=resources.peak_ram_mib,
        peak_vram_mib=resources.peak_vram_mib,
        embeddings=embeddings[:, 0, :],
    )


def _write_audio_matrix(output: Path, audio: list[np.ndarray]) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.save(temporary, np.stack(audio).astype(np.float32, copy=False))
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output


def _adaptive_gpu_worker(
    gpu_index: int,
    model_path: str,
    audio_path: str,
    initial_batch_size: int,
    max_batch_size: int,
    memory_fraction: float,
    total_memory_mib: float | None,
    work_queue: object,
    output_queue: object,
    start_event: object,
) -> None:
    """Own one physical GPU, tune its batch, then consume a shared work queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import absl.logging
        import tensorflow as tf

        absl.logging.set_verbosity(absl.logging.ERROR)
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        tf.get_logger().setLevel(logging.ERROR)
        physical_devices = tf.config.list_physical_devices("GPU")
        if len(physical_devices) != 1:
            raise RuntimeError(
                f"GPU fisica {gpu_index}: esperada uma GPU isolada; encontradas "
                f"{len(physical_devices)}."
            )
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logical_device = "/GPU:0"
        audio = np.load(audio_path, mmap_mode="r")
        if audio.ndim != 2 or audio.shape[1] != MODEL_SAMPLES:
            raise RuntimeError(f"Matriz de audio invalida: {audio.shape}")

        with tf.device(logical_device):
            saved_model = tf.saved_model.load(model_path)
        if "embeddings" not in saved_model.signatures:
            raise RuntimeError("SavedModel BirdNET sem assinatura 'embeddings'.")
        encode = saved_model.signatures["embeddings"]

        def infer(indices: Sequence[int]) -> np.ndarray:
            batch = np.asarray(audio[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
            with tf.device(logical_device):
                tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
                result = encode(inputs=tensor)["embeddings"]
            embeddings = result.numpy().astype(np.float32, copy=False)
            if embeddings.shape != (len(indices), EMBEDDING_DIMENSION):
                raise RuntimeError(
                    f"GPU {gpu_index}: forma inesperada {embeddings.shape}."
                )
            return embeddings

        maximum_useful = max(1, min(max_batch_size, len(audio)))
        candidate = max(1, min(initial_batch_size, maximum_useful))
        selected_batch = 0
        tuning_rows: list[dict[str, object]] = []
        initial_failures = 0
        while True:
            probe_indices = np.arange(candidate, dtype=np.int64) % len(audio)
            try:
                tf.config.experimental.reset_memory_stats(logical_device)
                started = time.perf_counter()
                infer(probe_indices)
                duration = time.perf_counter() - started
                memory = tf.config.experimental.get_memory_info(logical_device)
                peak_mib = float(memory["peak"]) / 1024**2
                selected_batch = candidate
                tuning_rows.append(
                    {
                        "batch_size": candidate,
                        "seconds": duration,
                        "peak_vram_mib": peak_mib,
                        "status": "ok",
                    }
                )
                pressure = (
                    peak_mib / total_memory_mib
                    if total_memory_mib is not None and total_memory_mib > 0
                    else None
                )
                if (
                    candidate >= maximum_useful
                    or (pressure is not None and pressure >= memory_fraction)
                ):
                    break
                next_candidate = min(candidate * 2, maximum_useful)
                if next_candidate == candidate:
                    break
                candidate = next_candidate
            except tf.errors.ResourceExhaustedError as error:
                tuning_rows.append(
                    {
                        "batch_size": candidate,
                        "seconds": np.nan,
                        "peak_vram_mib": np.nan,
                        "status": f"oom: {error}",
                    }
                )
                if selected_batch > 0:
                    break
                initial_failures += 1
                if initial_failures > 2 or candidate == 1:
                    raise RuntimeError(
                        f"GPU {gpu_index}: batch inicial falhou apos "
                        f"{initial_failures} tentativas."
                    ) from error
                candidate = max(1, candidate // 2)

        if selected_batch < 1:
            raise RuntimeError(f"GPU {gpu_index}: nenhum batch valido encontrado.")
        output_queue.put(
            {
                "kind": "ready",
                "gpu_index": gpu_index,
                "batch_size": selected_batch,
                "tuning": tuning_rows,
            }
        )
        start_event.wait()

        tf.config.experimental.reset_memory_stats(logical_device)
        active_started = time.perf_counter()
        current_batch = selected_batch
        tasks_processed = 0
        batches_processed = 0
        retries = 0
        finish_after_batch = False
        while not finish_after_batch:
            item = work_queue.get()
            if item is None:
                break
            indices = [int(item)]
            while len(indices) < current_batch:
                try:
                    item = work_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    finish_after_batch = True
                    break
                indices.append(int(item))

            offset = 0
            attempts_for_current_work = 0
            while offset < len(indices):
                chunk = indices[offset : offset + current_batch]
                try:
                    started = time.perf_counter()
                    embeddings = infer(chunk)
                    inference_seconds = time.perf_counter() - started
                except tf.errors.ResourceExhaustedError as error:
                    attempts_for_current_work += 1
                    retries += 1
                    if attempts_for_current_work > 2 or current_batch == 1:
                        raise RuntimeError(
                            f"GPU {gpu_index}: OOM persistente durante processamento."
                        ) from error
                    current_batch = max(1, current_batch // 2)
                    continue
                output_queue.put(
                    {
                        "kind": "result",
                        "gpu_index": gpu_index,
                        "indices": chunk,
                        "embeddings": embeddings,
                        "inference_seconds": inference_seconds,
                        "batch_size": len(chunk),
                    }
                )
                offset += len(chunk)
                tasks_processed += len(chunk)
                batches_processed += 1
                attempts_for_current_work = 0

        active_seconds = time.perf_counter() - active_started
        memory = tf.config.experimental.get_memory_info(logical_device)
        output_queue.put(
            {
                "kind": "done",
                "gpu_index": gpu_index,
                "selected_batch_size": selected_batch,
                "final_batch_size": current_batch,
                "tasks": tasks_processed,
                "batches": batches_processed,
                "active_seconds": active_seconds,
                "peak_vram_mib": float(memory["peak"]) / 1024**2,
                "retries": retries,
            }
        )
    except BaseException as error:
        output_queue.put(
            {
                "kind": "fatal",
                "gpu_index": gpu_index,
                "error": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            }
        )


def _protobuf_adaptive_gpu(
    audio: list[np.ndarray],
    initial_batch_size: int,
    max_batch_size: int,
    memory_fraction: float,
    gpu_indices: Sequence[int],
    cache_dir: Path,
) -> RunResult:
    """Run one persistent, independently tuned TensorFlow process per GPU."""
    setup_started = time.perf_counter()
    import birdnet

    model = birdnet.load("acoustic", "2.4", "pb")
    audio_path = _write_audio_matrix(cache_dir / "input_audio.npy", audio)
    memory_totals = _gpu_memory_totals()
    context = mp.get_context("spawn")
    work_queue = context.Queue()
    output_queue = context.Queue()
    start_event = context.Event()
    for index in range(len(audio)):
        work_queue.put(index)
    for _ in gpu_indices:
        work_queue.put(None)

    processes = [
        context.Process(
            target=_adaptive_gpu_worker,
            args=(
                gpu_index,
                str(model.model_path),
                str(audio_path),
                initial_batch_size,
                max_batch_size,
                memory_fraction,
                memory_totals.get(gpu_index),
                work_queue,
                output_queue,
                start_event,
            ),
            name=f"BirdNET-GPU-{gpu_index}",
        )
        for gpu_index in gpu_indices
    ]
    for process in processes:
        process.start()

    ready: dict[int, dict[str, object]] = {}
    try:
        while len(ready) < len(processes):
            message = output_queue.get(timeout=1800)
            if message["kind"] == "fatal":
                raise RuntimeError(
                    f"Falha ao preparar GPU {message['gpu_index']}:\n{message['error']}"
                )
            if message["kind"] != "ready":
                raise RuntimeError(f"Mensagem inesperada durante setup: {message['kind']}")
            ready[int(message["gpu_index"])] = message
            tuning = message["tuning"]
            LOGGER.info(
                "GPU %d pronta: batch adaptativo %d; tentativas: %s",
                message["gpu_index"],
                message["batch_size"],
                " -> ".join(
                    f"{row['batch_size']} ({row['status'].split(':', 1)[0]})"
                    for row in tuning
                ),
            )
        setup_seconds = time.perf_counter() - setup_started
        embeddings = np.empty((len(audio), EMBEDDING_DIMENSION), dtype=np.float32)
        completed = 0
        done: dict[int, dict[str, object]] = {}
        progress = tqdm(
            total=len(audio),
            desc="ProtoBuf multi-GPU adaptativo",
            unit="clipe",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        processing_started = time.perf_counter()
        with ResourceMonitor(gpu_indices) as resources:
            start_event.set()
            while len(done) < len(processes):
                message = output_queue.get(timeout=1800)
                kind = message["kind"]
                if kind == "fatal":
                    raise RuntimeError(
                        f"Falha na GPU {message['gpu_index']}:\n{message['error']}"
                    )
                if kind == "result":
                    indices = np.asarray(message["indices"], dtype=np.int64)
                    values = np.asarray(message["embeddings"], dtype=np.float32)
                    embeddings[indices] = values
                    completed += len(indices)
                    progress.update(len(indices))
                elif kind == "done":
                    done[int(message["gpu_index"])] = message
                else:
                    raise RuntimeError(f"Mensagem inesperada: {kind}")
        processing_seconds = time.perf_counter() - processing_started
        progress.close()
        if completed != len(audio):
            raise RuntimeError(
                f"Resultado multi-GPU incompleto: {completed} de {len(audio)} clipes."
            )
    except BaseException:
        start_event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        work_queue.close()
        output_queue.close()

    statistics = tuple(done[index] for index in sorted(done))
    batches = ";".join(
        f"GPU:{row['gpu_index']}={row['selected_batch_size']}" for row in statistics
    )
    return RunResult(
        strategy="Atual: ProtoBuf batch adaptativo multi-GPU",
        backend="ProtoBuf",
        device=",".join(f"GPU:{index}" for index in gpu_indices),
        workers=len(gpu_indices),
        batch_size=batches,
        setup_seconds=setup_seconds,
        processing_seconds=processing_seconds,
        peak_ram_mib=resources.peak_ram_mib,
        peak_vram_mib=resources.peak_vram_mib,
        embeddings=embeddings,
        gpu_statistics=statistics,
    )


def _comparison_rows(
    clip_ids: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    strategy: str,
) -> pd.DataFrame:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Formas de embeddings diferentes: {reference.shape} e {candidate.shape}."
        )
    left = reference.astype(np.float64)
    right = candidate.astype(np.float64)
    difference = left - right
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.sum(left * right, axis=1) / np.maximum(denominator, 1e-12)
    correlations = []
    for reference_row, candidate_row in zip(left, right, strict=True):
        if reference_row.std() == 0 or candidate_row.std() == 0:
            correlations.append(np.nan)
        else:
            correlations.append(np.corrcoef(reference_row, candidate_row)[0, 1])
    return pd.DataFrame(
        {
            "strategy": strategy,
            "clip_id": clip_ids,
            "mean_absolute_error": np.mean(np.abs(difference), axis=1),
            "root_mean_squared_error": np.sqrt(np.mean(difference**2, axis=1)),
            "maximum_absolute_error": np.max(np.abs(difference), axis=1),
            "cosine_similarity": cosine,
            "pearson_correlation": correlations,
        }
    )


def _summary_row(run: RunResult, clips: int, comparison: pd.DataFrame | None) -> dict[str, object]:
    row: dict[str, object] = {
        "strategy": run.strategy,
        "backend": run.backend,
        "device": run.device,
        "workers": run.workers,
        "batch_size": run.batch_size,
        "clips": clips,
        "setup_seconds": run.setup_seconds,
        "processing_seconds": run.processing_seconds,
        "total_seconds": run.setup_seconds + run.processing_seconds,
        "clips_per_second": clips / run.processing_seconds,
        "speedup_processing": np.nan,
        "peak_ram_mib": run.peak_ram_mib,
        "peak_vram_mib": run.peak_vram_mib,
        "mean_absolute_error": 0.0,
        "maximum_absolute_error": 0.0,
        "mean_cosine_similarity": 1.0,
        "minimum_cosine_similarity": 1.0,
        "mean_pearson_correlation": 1.0,
    }
    if comparison is not None:
        row.update(
            {
                "mean_absolute_error": comparison["mean_absolute_error"].mean(),
                "maximum_absolute_error": comparison["maximum_absolute_error"].max(),
                "mean_cosine_similarity": comparison["cosine_similarity"].mean(),
                "minimum_cosine_similarity": comparison["cosine_similarity"].min(),
                "mean_pearson_correlation": comparison["pearson_correlation"].mean(),
            }
        )
    return row


def run_benchmark(
    clips_per_dataset: int,
    split: str,
    batch_size: int,
    max_batch_size: int,
    gpu_memory_fraction: float,
    seed: int,
    gpu_devices: str,
    skip_cpu_protobuf: bool,
    skip_gpu: bool,
    force: bool,
) -> Path:
    paths = CodecoPath.discover()
    configure_model_environment(paths)
    os.environ["BIRDNET_APP_DATA"] = str(paths.models / "birdnet")
    # TFLite initializes TensorFlow before the ProtoBuf comparison.  Spawn is
    # required so each BirdNET worker can restrict itself to one GPU instead of
    # inheriting an already initialized multi-GPU runtime through ``fork``.
    os.environ["BIRDNET_START_METHOD"] = "spawn"
    cuda_library_dirs = _configure_cuda_environment()
    log_file = _configure_logging(paths)
    _header()
    cache_dir = paths.cache / "birdnet_benchmark"
    summary_output = paths.benchmarks / "birdnet_benchmark.csv"
    details_output = paths.benchmarks / "birdnet_embedding_comparison.csv"
    if force:
        for path in (summary_output, details_output):
            path.unlink(missing_ok=True)
        if cache_dir.is_dir():
            for path in cache_dir.glob("*.npz"):
                path.unlink()

    gpus = _tensorflow_gpus()
    visible_gpu_indices = [int(gpu["index"]) for gpu in gpus]
    if gpu_devices != "auto":
        requested = [int(value.strip()) for value in gpu_devices.split(",") if value.strip()]
        missing = sorted(set(requested) - set(visible_gpu_indices))
        if missing:
            raise ValueError(
                "GPUs solicitadas nao estao visiveis para o TensorFlow: "
                + ", ".join(map(str, missing))
            )
        visible_gpu_indices = requested

    _section("Configuracao")
    LOGGER.info("Log: %s", log_file)
    LOGGER.info("Modelo: BirdNET 2.4 FP32")
    LOGGER.info("Split: %s; clipes por dataset: %d", split, clips_per_dataset)
    LOGGER.info("Microbatch inicial ProtoBuf: %d", batch_size)
    LOGGER.info(
        "Batch GPU maximo: %d; alvo de VRAM: %.0f%%",
        max_batch_size,
        gpu_memory_fraction * 100,
    )
    LOGGER.info("Cache dos modelos: %s", paths.models / "birdnet")
    if cuda_library_dirs:
        LOGGER.info(
            "Bibliotecas CUDA da .venv: %d diretorios adicionados ao ambiente",
            len(cuda_library_dirs),
        )
    if gpus:
        for gpu in gpus:
            LOGGER.info("TensorFlow GPU %s: %s", gpu["index"], gpu["name"])
    else:
        LOGGER.info("TensorFlow: nenhuma GPU visivel")

    _section("Preparacao dos dados")
    clips = _select_clips(clips_per_dataset, split, seed)
    LOGGER.info(
        "Selecionados %d clipes: %s",
        len(clips),
        ", ".join(
            f"{dataset}={count}"
            for dataset, count in clips.groupby("dataset", sort=False).size().items()
        ),
    )
    audio = _load_audio(clips)
    clip_ids = clips["clip_id"].astype(str).to_numpy()

    runs: list[RunResult] = []
    comparisons: list[pd.DataFrame] = []

    _section("1/3 Referencia: TFLite CPU sequencial")
    reference = _legacy_tflite(audio)
    runs.append(reference)
    _atomic_embeddings(cache_dir / "tflite_cpu.npz", clip_ids, reference.embeddings)

    _section("2/3 TensorFlow ProtoBuf: CPU em batch")
    cpu_fallback = not visible_gpu_indices and not skip_gpu
    if not skip_cpu_protobuf or cpu_fallback:
        if cpu_fallback:
            LOGGER.warning(
                "Fallback ativado: nenhuma GPU visivel; executando ProtoBuf/CPU "
                "com batch %d.",
                batch_size,
            )
        cpu = _protobuf(audio, batch_size, ["CPU"])
        runs.append(cpu)
        _atomic_embeddings(cache_dir / "protobuf_cpu.npz", clip_ids, cpu.embeddings)
        comparisons.append(
            _comparison_rows(clip_ids, reference.embeddings, cpu.embeddings, cpu.strategy)
        )
    else:
        LOGGER.info("ProtoBuf CPU ignorado por --skip-cpu-protobuf.")

    if skip_gpu:
        _section("3/3 TensorFlow ProtoBuf: GPU")
        LOGGER.info("ProtoBuf GPU ignorado por --skip-gpu.")
    elif not visible_gpu_indices:
        _section("3/3 TensorFlow ProtoBuf: GPU")
        LOGGER.warning("Etapa GPU ignorada: nenhuma GPU esta visivel para o TensorFlow.")
    else:
        _section("3/3 TensorFlow ProtoBuf: batch multi-GPU")
        gpu = _protobuf_adaptive_gpu(
            audio=audio,
            initial_batch_size=batch_size,
            max_batch_size=max_batch_size,
            memory_fraction=gpu_memory_fraction,
            gpu_indices=visible_gpu_indices,
            cache_dir=cache_dir,
        )
        runs.append(gpu)
        _atomic_embeddings(cache_dir / "protobuf_gpu.npz", clip_ids, gpu.embeddings)
        comparisons.append(
            _comparison_rows(clip_ids, reference.embeddings, gpu.embeddings, gpu.strategy)
        )

    details = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    if not details.empty:
        _atomic_csv(details, details_output)

    comparison_by_strategy = (
        {strategy: group for strategy, group in details.groupby("strategy", sort=False)}
        if not details.empty
        else {}
    )
    summary_rows = [
        _summary_row(run, len(clips), comparison_by_strategy.get(run.strategy)) for run in runs
    ]
    reference_seconds = reference.processing_seconds
    for row in summary_rows:
        row["speedup_processing"] = reference_seconds / float(row["processing_seconds"])
    summary = pd.DataFrame.from_records(summary_rows)
    _atomic_csv(summary, summary_output)

    _section("Tempos e recursos")
    _log_table(
        (
            "Estrategia",
            "Workers",
            "Batch",
            "Setup",
            "Processamento",
            "Clipes/s",
            "Speedup",
            "RAM pico",
            "VRAM pico",
        ),
        [
            (
                run.strategy,
                run.workers,
                run.batch_size,
                f"{run.setup_seconds:.2f} s",
                f"{run.processing_seconds:.2f} s",
                f"{len(clips) / run.processing_seconds:.2f}",
                f"{reference_seconds / run.processing_seconds:.2f}x",
                f"{run.peak_ram_mib:.0f} MiB",
                "-" if run.peak_vram_mib is None else f"{run.peak_vram_mib:.0f} MiB",
            )
            for run in runs
        ],
    )

    gpu_rows = [row for run in runs for row in run.gpu_statistics]
    if gpu_rows:
        _section("Desempenho por GPU")
        _log_table(
            (
                "GPU",
                "Batch escolhido",
                "Batch final",
                "Tarefas",
                "Lotes",
                "Ativa",
                "Tarefas/s",
                "Pico VRAM",
                "Retries",
            ),
            [
                (
                    row["gpu_index"],
                    row["selected_batch_size"],
                    row["final_batch_size"],
                    row["tasks"],
                    row["batches"],
                    f"{float(row['active_seconds']):.2f} s",
                    f"{int(row['tasks']) / max(float(row['active_seconds']), 1e-12):.2f}",
                    f"{float(row['peak_vram_mib']):.0f} MiB",
                    row["retries"],
                )
                for row in gpu_rows
            ],
        )

    if comparisons:
        _section("Equivalencia dos embeddings em relacao ao TFLite")
        _log_table(
            ("Estrategia", "MAE", "Erro maximo", "Cosseno medio", "Cosseno minimo", "Pearson"),
            [
                (
                    strategy,
                    f"{group['mean_absolute_error'].mean():.8g}",
                    f"{group['maximum_absolute_error'].max():.8g}",
                    f"{group['cosine_similarity'].mean():.8f}",
                    f"{group['cosine_similarity'].min():.8f}",
                    f"{group['pearson_correlation'].mean():.8f}",
                )
                for strategy, group in details.groupby("strategy", sort=False)
            ],
        )

    _section("Arquivos gerados")
    LOGGER.info("Resumo: %s", summary_output)
    if comparisons:
        LOGGER.info("Comparacao por clipe: %s", details_output)
    LOGGER.info("Embeddings binarios (cache): %s", cache_dir)
    return summary_output


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara os embeddings BirdNET 2.4 do TFLite sequencial com o "
            "TensorFlow ProtoBuf em batch, CPU e multi-GPU."
        )
    )
    parser.add_argument("--clips-per-dataset", type=int, default=DEFAULT_CLIPS_PER_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-batch-size", type=int, default=DEFAULT_MAX_BATCH_SIZE)
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=DEFAULT_GPU_MEMORY_FRACTION,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--gpu-devices",
        default="auto",
        help="auto ou indices separados por virgula, por exemplo: 0,1",
    )
    parser.add_argument("--skip-cpu-protobuf", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove resultados e embeddings de benchmark anteriores antes de executar.",
    )
    arguments = parser.parse_args()
    if arguments.clips_per_dataset < 1:
        parser.error("--clips-per-dataset deve ser maior que zero.")
    if arguments.batch_size < 1:
        parser.error("--batch-size deve ser maior que zero.")
    if arguments.max_batch_size < arguments.batch_size:
        parser.error("--max-batch-size deve ser maior ou igual a --batch-size.")
    if not 0 < arguments.gpu_memory_fraction < 1:
        parser.error("--gpu-memory-fraction deve estar entre zero e um.")
    if arguments.gpu_devices != "auto":
        try:
            devices = [
                int(value.strip())
                for value in arguments.gpu_devices.split(",")
                if value.strip()
            ]
        except ValueError:
            parser.error("--gpu-devices deve ser auto ou indices separados por virgula.")
        if not devices or any(index < 0 for index in devices) or len(devices) != len(set(devices)):
            parser.error("--gpu-devices contem indices ausentes, negativos ou repetidos.")
    return arguments


def main() -> None:
    arguments = _parse_arguments()
    run_benchmark(
        clips_per_dataset=arguments.clips_per_dataset,
        split=arguments.split,
        batch_size=arguments.batch_size,
        max_batch_size=arguments.max_batch_size,
        gpu_memory_fraction=arguments.gpu_memory_fraction,
        seed=arguments.seed,
        gpu_devices=arguments.gpu_devices,
        skip_cpu_protobuf=arguments.skip_cpu_protobuf,
        skip_gpu=arguments.skip_gpu,
        force=arguments.force,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.error("Benchmark interrompido pelo usuario.")
        raise SystemExit(130) from None
    except Exception:
        LOGGER.exception("Falha no benchmark BirdNET.")
        raise SystemExit(1) from None
