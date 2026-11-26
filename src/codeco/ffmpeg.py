"""Parallel, resumable FFmpeg reconstruction and metric evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import psutil
import soundfile as sf
from tqdm import tqdm

from codeco.acoustic_indices import VERSION as ACOUSTIC_INDICES_VERSION
from codeco.acoustic_indices import acoustic_indices_v1
from codeco.alignment import normalized_cross_correlation_align
from codeco.codeco_path import CodecoPath
from codeco.environment import EnvironmentState, _ffmpeg_installation as _environment_ffmpeg
from codeco.codec_config import codec_config
from codeco.prepare_table import (
    BIT_DEPTH,
    CLIP_SECONDS,
    SAMPLE_RATE,
    BaselineState,
    TqdmLoggingHandler,
    load_pcm_clip,
    load_standalone_examples,
)


PHASE = "ffmpeg_metrics"
IMPLEMENTATION_VERSION = "1"
CLIP_SAMPLES = round(SAMPLE_RATE * CLIP_SECONDS)
MAX_LAG = 4096
CHECKPOINT_SIZE = 64
MAX_ATTEMPTS = 3
DEFAULT_MAX_CLIPS_PER_DATASET = 300
DEFAULT_SEED = 42
LOGGER = logging.getLogger("codeco.ffmpeg")


@dataclass(frozen=True)
class CodecCondition:
    codec: str
    setting: str
    extension: str
    encoder_arguments: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationTask:
    ordinal: int
    dataset: str
    clip_id: str
    path: str
    start: float
    codec: str
    setting: str
    extension: str
    encoder_arguments: tuple[str, ...]
    ffmpeg: str
    pcm_file: str | None = None
    pcm_row: int | None = None

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.dataset, self.clip_id, self.codec, self.setting


def _conditions() -> tuple[CodecCondition, ...]:
    config = codec_config()
    rows = [CodecCondition("baseline", "orig", "", ())]
    for level in config["flac"]["compression_levels"]:
        rows.append(CodecCondition("flac", f"compression_level={level}", "flac", ("-c:a", "flac", "-compression_level", str(level))))
    for codec, encoder, extension in (("mp3", "libmp3lame", "mp3"), ("aac", "aac", "m4a"), ("opus", "libopus", "opus"), ("lc3", "liblc3", "lc3")):
        for bitrate in config[codec]["bitrates_kbps"]:
            arguments = ["-c:a", encoder]
            if codec == "aac":
                arguments += ["-profile:a", config[codec]["profile"]]
            arguments += ["-b:a", f"{bitrate}k"]
            if codec == "opus":
                arguments += ["-vbr", "on" if config[codec]["vbr"] else "off", "-application", config[codec]["application"]]
            if codec == "lc3":
                arguments += ["-frame_duration", str(config[codec]["frame_duration_ms"]), "-high_resolution", "1" if config[codec]["high_resolution"] else "0"]
            rows.append(CodecCondition(codec, f"bitrate_kbps={bitrate}", extension, tuple(arguments)))
    return tuple(rows)


CONDITIONS = _conditions()

RESULT_COLUMNS = (
    "dataset", "clip_id", "codec", "setting", "nbytes", "bitrate_kbps",
    "compression_ratio", "lag", "lag_samples", "lag_seconds", "alignment_score",
    "snr_db", "mel_mae_db", "mel_rmse_db", "raw_snr_db", "raw_mel_rmse_db",
    "mel_ssim", "mel_cosine", "idx_ACI", "idx_ADI", "idx_AEI", "idx_Ht",
    "idx_Hf", "idx_H", "idx_BIO_2_24", "idx_BIO_2_8", "idx_NDSI_CLASSIC",
    "idx_NDSI_EXTENDED",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _configure_logging(paths: CodecoPath, log_path: Path | None = None) -> Path:
    log_file = log_path or paths.logs / "ffmpeg.log"
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
    command = subprocess.list2cmdline([sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]])
    LOGGER.info("\n%s\nCOMPRESSAO FFmpeg + RECONSTRUCAO + METRICAS\n[CMD]   %s\n%s", line, command, line)


def _section(title: str) -> None:
    LOGGER.info("\n%s\n%s", title, "-" * 72)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _physical_cores() -> int:
    return max(1, psutil.cpu_count(logical=False) or 1)


def _ffmpeg_installation() -> tuple[Path, Path, str]:
    installation = _environment_ffmpeg(platform.system())
    versions = (
        f"system={installation.version}; lc3={installation.lc3_version}"
    )
    return installation.executable, installation.lc3_executable, versions


def _load_clip(path: str, start: float) -> np.ndarray:
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio nao encontrado: {audio_path}")
    info = sf.info(str(audio_path))
    start_frame = round(start * info.samplerate)
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
    """Compatibility wrapper around the signed normalized implementation."""
    result = normalized_cross_correlation_align(reference, decoded, MAX_LAG)
    return result.audio, result.lag_samples


def _run_ffmpeg(command: list[str]) -> None:
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip() or f"codigo {process.returncode}"
        raise RuntimeError(f"FFmpeg falhou: {detail}")


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
        "lag_samples": 0,
        "lag_seconds": 0.0,
        "alignment_score": 1.0,
        "snr_db": float("inf"),
        "mel_mae_db": 0.0,
        "mel_rmse_db": 0.0,
        "mel_ssim": 1.0,
        "mel_cosine": 1.0,
        "raw_snr_db": float("inf"),
        "raw_mel_rmse_db": 0.0,
        **acoustic_indices_v1(audio),
    }


def _evaluate_task(task: EvaluationTask) -> dict[str, object]:
    from codeco.metrics import reconstruction_metrics

    reference = _load_task_clip(task)
    if task.codec == "baseline":
        return _baseline_result(task, reference)

    with tempfile.TemporaryDirectory(prefix="codeco-ffmpeg-task-") as directory:
        temporary = Path(directory)
        source = temporary / "source.wav"
        encoded = temporary / f"encoded.{task.extension}"
        decoded_path = temporary / "decoded.wav"
        sf.write(source, reference, SAMPLE_RATE, subtype="PCM_16")
        _run_ffmpeg(
            [
                task.ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(source),
                "-ac", "1", "-ar", str(SAMPLE_RATE), *task.encoder_arguments,
                str(encoded),
            ]
        )
        encoded_bytes = encoded.stat().st_size
        _run_ffmpeg(
            [
                task.ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(encoded),
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", str(decoded_path),
            ]
        )
        decoded, _ = sf.read(decoded_path, dtype="float32", always_2d=False)
    if decoded.ndim > 1:
        decoded = decoded.mean(axis=1)
    from codeco.metrics import reconstruction_metrics

    raw = decoded[: len(reference)]
    if len(raw) < len(reference):
        raw = np.pad(raw, (0, len(reference) - len(raw)))
    raw_metrics = reconstruction_metrics(reference, raw)
    alignment = normalized_cross_correlation_align(reference, decoded, MAX_LAG)
    aligned, lag = alignment.audio, alignment.lag_samples
    return {
        "dataset": task.dataset,
        "clip_id": task.clip_id,
        "codec": task.codec,
        "setting": task.setting,
        "nbytes": encoded_bytes,
        "bitrate_kbps": encoded_bytes * 8 / CLIP_SECONDS / 1000,
        "compression_ratio": (reference.size * BIT_DEPTH // 8) / encoded_bytes,
        "lag": lag,
        "lag_samples": lag,
        "lag_seconds": lag / SAMPLE_RATE,
        "alignment_score": alignment.score,
        "raw_snr_db": raw_metrics["snr_db"],
        "raw_mel_rmse_db": raw_metrics["mel_rmse_db"],
        **reconstruction_metrics(reference, aligned),
    }


def _select_clips(
    baseline: BaselineState,
    datasets: Sequence[str] | None,
    max_per_dataset: int | None,
    seed: int,
    split: str | None = None,
) -> pd.DataFrame:
    table = baseline.table.to_pandas()
    if datasets is not None:
        table = table[table["dataset"].isin(datasets)].copy()
    if split is not None:
        table = table[table["split"] == split].copy()
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


def _build_tasks(
    clips: pd.DataFrame,
    ffmpeg: Path,
    lc3_ffmpeg: Path,
) -> list[EvaluationTask]:
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
                        extension=condition.extension,
                        encoder_arguments=condition.encoder_arguments,
                        ffmpeg=str(lc3_ffmpeg if condition.codec == "lc3" else ffmpeg),
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
            [
                (PHASE, fingerprint, task.ordinal, *task.key, now)
                for task in tasks
            ],
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
            "UPDATE runs SET status = 'running', updated_at = ?, metadata_json = ? WHERE phase = ? AND fingerprint = ?",
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


def _clear_checkpoint_run(
    connection: sqlite3.Connection,
    fingerprint: str,
) -> tuple[int, int]:
    """Delete only the current FFmpeg run, preserving every other fingerprint."""
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


def _mark_running(
    connection: sqlite3.Connection,
    fingerprint: str,
    in_flight: Iterable[tuple[EvaluationTask, int]],
) -> None:
    now = _utc_now()
    with connection:
        connection.executemany(
            """UPDATE tasks SET status = 'running', attempts = ?, updated_at = ?
               WHERE phase = ? AND fingerprint = ? AND dataset = ? AND clip_id = ? AND codec = ? AND setting = ?""",
            [(attempt, now, PHASE, fingerprint, *task.key) for task, attempt in in_flight],
        )


def _checkpoint_results(
    connection: sqlite3.Connection,
    fingerprint: str,
    completed: Sequence[tuple[EvaluationTask, int, dict[str, object]]],
    in_flight: Iterable[tuple[EvaluationTask, int]],
) -> int:
    now = _utc_now()
    with connection:
        connection.executemany(
            """UPDATE tasks SET status = 'completed', attempts = ?, result_json = ?, error = NULL, updated_at = ?
               WHERE phase = ? AND fingerprint = ? AND dataset = ? AND clip_id = ? AND codec = ? AND setting = ?""",
            [
                (
                    attempt,
                    json.dumps(result, ensure_ascii=False, sort_keys=True),
                    now,
                    PHASE,
                    fingerprint,
                    *task.key,
                )
                for task, attempt, result in completed
            ],
        )
        connection.executemany(
            """UPDATE tasks SET status = 'running', attempts = ?, updated_at = ?
               WHERE phase = ? AND fingerprint = ? AND dataset = ? AND clip_id = ? AND codec = ? AND setting = ?
                 AND status != 'completed'""",
            [(attempt, now, PHASE, fingerprint, *task.key) for task, attempt in in_flight],
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
    task: EvaluationTask,
    attempts: int,
    error: str,
) -> None:
    now = _utc_now()
    with connection:
        connection.execute(
            """UPDATE tasks SET status = 'failed', attempts = ?, error = ?, updated_at = ?
               WHERE phase = ? AND fingerprint = ? AND dataset = ? AND clip_id = ? AND codec = ? AND setting = ?""",
            (attempts, error, now, PHASE, fingerprint, *task.key),
        )
        connection.execute(
            """UPDATE runs SET status = 'failed', updated_at = ?
               WHERE phase = ? AND fingerprint = ?""",
            (now, PHASE, fingerprint),
        )


def _export_csv(
    connection: sqlite3.Connection,
    fingerprint: str,
    output: Path,
) -> int:
    rows = connection.execute(
        """SELECT result_json FROM tasks
           WHERE phase = ? AND fingerprint = ? AND status = 'completed'
           ORDER BY ordinal""",
        (PHASE, fingerprint),
    ).fetchall()
    records = [json.loads(row[0]) for row in rows]
    table = pd.DataFrame.from_records(records, columns=RESULT_COLUMNS)
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


def _run_pool(
    connection: sqlite3.Connection,
    fingerprint: str,
    tasks: Sequence[EvaluationTask],
    completed_keys: set[tuple[str, str, str, str]],
    attempts_by_key: dict[tuple[str, str, str, str], int],
    workers: int,
) -> int:
    pending = deque(task for task in tasks if task.key not in completed_keys)
    total = len(tasks)
    checkpoint_buffer: list[tuple[EvaluationTask, int, dict[str, object]]] = []
    psutil.cpu_percent(interval=None)
    executor = ProcessPoolExecutor(max_workers=workers)
    in_flight: dict[object, tuple[EvaluationTask, int]] = {}

    def submit(task: EvaluationTask) -> None:
        attempt = attempts_by_key.get(task.key, 0) + 1
        attempts_by_key[task.key] = attempt
        in_flight[executor.submit(_evaluate_task, task)] = (task, attempt)

    try:
        while pending and len(in_flight) < CHECKPOINT_SIZE:
            submit(pending.popleft())
        _mark_running(connection, fingerprint, in_flight.values())

        with tqdm(
            total=total,
            initial=len(completed_keys),
            desc="FFmpeg + metricas",
            unit="tarefa",
            dynamic_ncols=True,
            file=sys.stdout,
        ) as progress:
            while in_flight:
                done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
                for future in done:
                    task, attempt = in_flight.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                        if attempt < MAX_ATTEMPTS:
                            LOGGER.warning(
                                "Falha em %s/%s/%s/%s; tentativa %d/%d: %s",
                                task.dataset,
                                task.clip_id,
                                task.codec.upper(),
                                task.setting,
                                attempt,
                                MAX_ATTEMPTS,
                                error,
                            )
                            submit(task)
                            continue
                        if checkpoint_buffer:
                            _checkpoint_results(
                                connection,
                                fingerprint,
                                checkpoint_buffer,
                                in_flight.values(),
                            )
                            checkpoint_buffer.clear()
                        _checkpoint_failure(connection, fingerprint, task, attempt, error)
                        raise RuntimeError(
                            f"Tarefa {task.key} falhou apos {MAX_ATTEMPTS} tentativas: {error}"
                        ) from exc

                    checkpoint_buffer.append((task, attempt, result))
                    progress.update(1)
                    if pending:
                        submit(pending.popleft())

                while len(checkpoint_buffer) >= CHECKPOINT_SIZE:
                    block = checkpoint_buffer[:CHECKPOINT_SIZE]
                    del checkpoint_buffer[:CHECKPOINT_SIZE]
                    completed_count = _checkpoint_results(
                        connection,
                        fingerprint,
                        block,
                        in_flight.values(),
                    )
                    cpu_percent, rss_gib = _resource_snapshot()
                    elapsed = max(progress.format_dict.get("elapsed", 0.0), 1e-9)
                    rate = max((completed_count - len(completed_keys)) / elapsed, 1e-9)
                    eta = (total - completed_count) / rate
                    LOGGER.info(
                        "Checkpoint: %d/%d (%.1f%%); fila ativa: %d; CPU: %.1f%%; RSS agregado: %.2f GiB; ETA: %.1f min.",
                        completed_count,
                        total,
                        completed_count * 100 / total,
                        len(in_flight),
                        cpu_percent,
                        rss_gib,
                        eta / 60,
                    )

            if checkpoint_buffer:
                completed_count = _checkpoint_results(
                    connection,
                    fingerprint,
                    checkpoint_buffer,
                    (),
                )
                LOGGER.info("Checkpoint final: %d/%d tarefas.", completed_count, total)
    finally:
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    return total


def run(
    baseline: BaselineState,
    datasets: Sequence[str] | None = None,
    max_clips_per_dataset: int | None = DEFAULT_MAX_CLIPS_PER_DATASET,
    seed: int = DEFAULT_SEED,
    split: str | None = None,
    checkpoint_path: Path | None = None,
    output_path: Path | None = None,
    log_path: Path | None = None,
    force: bool = False,
    environment: EnvironmentState | None = None,
) -> Path:
    paths = CodecoPath.discover()
    checkpoint = checkpoint_path or paths.cache / "checkpoints.sqlite3"
    output = output_path or paths.tables / "ffmpeg_metrics.csv"
    log_file = _configure_logging(paths, log_path)
    _header()

    _section("Preparacao da execucao")
    if environment is None:
        ffmpeg, lc3_ffmpeg, ffmpeg_version = _ffmpeg_installation()
    else:
        ffmpeg = environment.system.ffmpeg.executable
        lc3_ffmpeg = environment.system.ffmpeg.lc3_executable
        ffmpeg_version = (
            f"system={environment.system.ffmpeg.version}; "
            f"lc3={environment.system.ffmpeg.lc3_version}"
        )
    workers = _physical_cores()
    clips = _select_clips(
        baseline,
        datasets,
        max_clips_per_dataset,
        seed,
        split,
    )
    tasks = _build_tasks(clips, ffmpeg, lc3_ffmpeg)
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
    selection_sha256 = _sha256_json(selection_records)
    metadata = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "manifest_sha256": baseline.manifest_sha256,
        "selection_sha256": selection_sha256,
        "datasets": list(dict.fromkeys(clips["dataset"].astype(str))),
        "split": split,
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
        "acoustic_indices_version": ACOUSTIC_INDICES_VERSION,
        "ffmpeg": str(ffmpeg),
        "ffmpeg_version": ffmpeg_version,
    }
    fingerprint = _sha256_json(metadata)
    dataset_counts = clips.groupby("dataset", sort=False).size().to_dict()
    LOGGER.info("Log: %s", log_file)
    LOGGER.info("Checkpoint: %s", checkpoint)
    LOGGER.info("Resultado CSV: %s", output)
    LOGGER.info("FFmpeg: %s", ffmpeg_version)
    LOGGER.info(
        "Entrada PCM: %s",
        baseline.pcm_cache_dir if baseline.pcm_index is not None else "carregamento autonomo",
    )
    LOGGER.info("Fingerprint: %s", fingerprint)
    LOGGER.info(
        "Clipes: %s",
        ", ".join(f"{name}={count}" for name, count in dataset_counts.items()),
    )
    LOGGER.info("Condicoes: 1 baseline + 14 FFmpeg; tarefas: %d", len(tasks))
    LOGGER.info("Nucleos fisicos: %d; processos persistentes: %d", workers, workers)
    LOGGER.info("Janela da fila: %d; checkpoint: %d resultados; tentativas: %d", CHECKPOINT_SIZE, CHECKPOINT_SIZE, MAX_ATTEMPTS)

    connection = _connect_checkpoint(checkpoint)
    started = time.perf_counter()
    try:
        if force:
            tasks_deleted, runs_deleted = _clear_checkpoint_run(
                connection,
                fingerprint,
            )
            LOGGER.warning(
                "--force: checkpoint atual removido (%d execucao; %d tarefas). "
                "Outros fingerprints foram preservados.",
                runs_deleted,
                tasks_deleted,
            )
        completed, attempts = _prepare_checkpoint(connection, fingerprint, metadata, tasks)
        _section("Processamento")
        LOGGER.info("Retomada: %d concluidas; %d pendentes.", len(completed), len(tasks) - len(completed))
        if len(completed) < len(tasks):
            _run_pool(connection, fingerprint, tasks, completed, attempts, workers)
        rows = _export_csv(connection, fingerprint, output)
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

    _section("Execucao concluida")
    LOGGER.info("Resultados: %d linhas em %s", rows, output)
    LOGGER.info("Tempo total: %.1f s", time.perf_counter() - started)
    return output


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa baseline e codecs FFmpeg com pool persistente e checkpoint SQLite."
    )
    parser.add_argument("--datasets", nargs="+", help="Datasets selecionados; padrao: todos.")
    parser.add_argument(
        "--max-clips-per-dataset",
        type=int,
        default=0,
        help="Limite por dataset (exemplo autonomo: todos; use 0 para todos).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalcula a execucao atual, preservando outros fingerprints no checkpoint.",
    )
    parser.add_argument("--checkpoint-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--log-path", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.max_clips_per_dataset < 0:
        parser.error("--max-clips-per-dataset deve ser maior ou igual a zero.")
    if arguments.max_clips_per_dataset == 0:
        arguments.max_clips_per_dataset = None
    return arguments


def main() -> None:
    arguments = _parse_arguments()
    paths = CodecoPath.discover()
    test_directory = paths.tests / "ffmpeg"
    baseline = load_standalone_examples()
    run(
        baseline=baseline,
        datasets=arguments.datasets,
        max_clips_per_dataset=arguments.max_clips_per_dataset,
        seed=arguments.seed,
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
        LOGGER.exception("Falha na fase FFmpeg.")
        raise SystemExit(1) from None
