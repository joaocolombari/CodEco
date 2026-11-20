"""Compare the legacy clip-at-a-time EnCodec flow with the current pipeline."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from codeco.codeco_path import CodecoPath
from codeco.environment import configure_model_environment
from codeco.encodec import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GPU_MEMORY_FRACTION,
    DEFAULT_GPU_PREFETCH_BATCHES,
    DEFAULT_SEED,
    MODEL_NAME,
    RESULT_COLUMNS,
    MetricWork,
    _build_tasks,
    _cuda_device_count,
    _encodec_chunk,
    _metric_worker,
    _select_clips,
    run as run_current,
)
from codeco.prepare_table import TqdmLoggingHandler


LOGGER = logging.getLogger("codeco.encodec_benchmark")


def _configure_logging(paths: CodecoPath) -> Path:
    log_file = paths.logs / "encodec_benchmark.log"
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
        "\n%s\nBENCHMARK ENCODEC: LEGADO X ATUAL\n[CMD]   %s\n%s",
        line,
        command,
        line,
    )


def _section(title: str) -> None:
    LOGGER.info("\n%s\n%s", title, "-" * 72)


def _log_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    values = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in values))
        for index, header in enumerate(headers)
    ]
    LOGGER.info(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    LOGGER.info("  ".join("-" * width for width in widths))
    for row in values:
        LOGGER.info("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _write_csv(table: pd.DataFrame, output: Path) -> None:
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


def _run_legacy(clips: pd.DataFrame, output: Path, model_name: str) -> None:
    import torch
    from transformers import EncodecModel

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = EncodecModel.from_pretrained(model_name).to(device).eval()
    codebook_bits = int(np.ceil(np.log2(model.config.codebook_size)))
    channels = int(model.config.audio_channels)
    tasks = _build_tasks(clips)
    records = []
    try:
        for task in tqdm(
            tasks,
            desc="Legado: 1 clipe por vez",
            unit="tarefa",
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            if task.codec == "baseline":
                outcome = _metric_worker(MetricWork(task))
            else:
                items, _ = _encodec_chunk(
                    model,
                    torch,
                    device,
                    [task],
                    codebook_bits,
                    channels,
                )
                _, reference, decoded, nbytes = items[0]
                outcome = _metric_worker(
                    MetricWork(task, reference, decoded, nbytes, 1)
                )
            if outcome.error is not None or outcome.result is None:
                raise RuntimeError(
                    f"Benchmark legado falhou para {task.key}: {outcome.error}"
                )
            records.append(outcome.result)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    _write_csv(pd.DataFrame.from_records(records, columns=RESULT_COLUMNS), output)


def _comparison(legacy: pd.DataFrame, current: pd.DataFrame) -> dict[str, object]:
    keys = ["dataset", "clip_id", "codec", "setting"]
    merged = legacy.merge(
        current,
        on=keys,
        suffixes=("_legacy", "_current"),
        how="outer",
        indicator=True,
    )
    keys_equal = bool((merged["_merge"] == "both").all())
    common = merged[merged["_merge"] == "both"]
    numeric_columns = [column for column in RESULT_COLUMNS if column not in keys]
    maximum = 0.0
    numerically_equivalent = keys_equal
    for column in numeric_columns:
        legacy_values = pd.to_numeric(common[f"{column}_legacy"], errors="coerce").to_numpy()
        current_values = pd.to_numeric(common[f"{column}_current"], errors="coerce").to_numpy()
        finite = np.isfinite(legacy_values) & np.isfinite(current_values)
        if finite.any():
            maximum = max(
                maximum,
                float(np.max(np.abs(legacy_values[finite] - current_values[finite]))),
            )
        if not np.allclose(
            legacy_values,
            current_values,
            rtol=1e-5,
            atol=1e-7,
            equal_nan=True,
        ):
            numerically_equivalent = False
    return {
        "same_rows": len(legacy) == len(current),
        "same_keys": keys_equal,
        "same_nbytes": bool(
            keys_equal
            and np.array_equal(
                common["nbytes_legacy"].to_numpy(),
                common["nbytes_current"].to_numpy(),
            )
        ),
        "same_lag": bool(
            keys_equal
            and np.array_equal(
                common["lag_legacy"].to_numpy(),
                common["lag_current"].to_numpy(),
            )
        ),
        "numerically_equivalent": numerically_equivalent,
        "max_numeric_abs_diff": maximum,
    }


def run_benchmark(
    clips_per_dataset: int,
    batch_size: int,
    gpu_memory_fraction: float,
    gpu_prefetch_batches: int,
    seed: int,
    model_name: str,
) -> Path:
    paths = CodecoPath.discover()
    configure_model_environment(paths)
    log_file = _configure_logging(paths)
    _header()
    gpu_count = _cuda_device_count()
    legacy_output = paths.benchmarks / "encodec_legacy.csv"
    current_output = paths.benchmarks / "encodec_current.csv"
    summary_output = paths.benchmarks / "encodec_benchmark.csv"
    checkpoint = paths.cache / "encodec_benchmark.sqlite3"

    _section("Configuracao")
    LOGGER.info("Log: %s", log_file)
    LOGGER.info("Modelo: %s", model_name)
    LOGGER.info("GPUs detectadas: %d", gpu_count)
    LOGGER.info("Clipes por dataset: %d", clips_per_dataset)
    LOGGER.info("Batch inicial atual: %d", batch_size)
    LOGGER.info(
        "Uso alvo da VRAM: %.0f%%; prefetch: %d batches por GPU",
        gpu_memory_fraction * 100,
        gpu_prefetch_batches,
    )

    _section("Estrategia legada")
    legacy_started = time.perf_counter()
    clips = _select_clips(None, "test", clips_per_dataset, seed)
    task_count = len(clips) * 4
    _run_legacy(clips, legacy_output, model_name)
    legacy_seconds = time.perf_counter() - legacy_started
    LOGGER.info(
        "Legado concluido: %.2f s; %.2f tarefas/s.",
        legacy_seconds,
        task_count / legacy_seconds,
    )

    _section("Estrategia atual")
    current_started = time.perf_counter()
    run_current(
        split="test",
        max_clips_per_dataset=clips_per_dataset,
        seed=seed,
        batch_size=batch_size,
        gpu_memory_fraction=gpu_memory_fraction,
        gpu_prefetch_batches=gpu_prefetch_batches,
        model_name=model_name,
        checkpoint_path=checkpoint,
        output_path=current_output,
        log_path=paths.logs / "encodec_benchmark_current.log",
        force=True,
    )
    current_seconds = time.perf_counter() - current_started
    LOGGER.info(
        "Atual concluido: %.2f s; %.2f tarefas/s.",
        current_seconds,
        task_count / current_seconds,
    )

    legacy = pd.read_csv(legacy_output, dtype={"clip_id": str})
    current = pd.read_csv(current_output, dtype={"clip_id": str})
    comparison = _comparison(legacy, current)
    summary = pd.DataFrame(
        [
            {
                "strategy": "legacy_single_clip_gpu0",
                "gpus": 1,
                "initial_batch_size": 1,
                "clips": len(clips),
                "tasks": task_count,
                "elapsed_seconds": legacy_seconds,
                "tasks_per_second": task_count / legacy_seconds,
                "speedup_vs_legacy": 1.0,
                **comparison,
            },
            {
                "strategy": "current_adaptive_multigpu",
                "gpus": gpu_count,
                "initial_batch_size": batch_size,
                "clips": len(clips),
                "tasks": task_count,
                "elapsed_seconds": current_seconds,
                "tasks_per_second": task_count / current_seconds,
                "speedup_vs_legacy": legacy_seconds / current_seconds,
                **comparison,
            },
        ]
    )
    _write_csv(summary, summary_output)

    _section("Tabela comparativa")
    _log_table(
        ("Estrategia", "GPUs", "Batch", "Tempo", "Tarefas/s", "Speedup"),
        [
            (
                "Legado: um clipe por vez",
                1,
                1,
                f"{legacy_seconds:.2f} s",
                f"{task_count / legacy_seconds:.2f}",
                "1.00x",
            ),
            (
                "Atual: batch adaptativo multi-GPU",
                gpu_count,
                batch_size,
                f"{current_seconds:.2f} s",
                f"{task_count / current_seconds:.2f}",
                f"{legacy_seconds / current_seconds:.2f}x",
            ),
        ],
    )

    _section("Equivalencia dos resultados")
    _log_table(
        ("Linhas", "Chaves", "Nbytes", "Lag", "Numerica", "Diferenca maxima"),
        [
            (
                comparison["same_rows"],
                comparison["same_keys"],
                comparison["same_nbytes"],
                comparison["same_lag"],
                comparison["numerically_equivalent"],
                f"{comparison['max_numeric_abs_diff']:.6g}",
            )
        ],
    )
    LOGGER.info(
        "Equivalencia: linhas=%s; chaves=%s; nbytes=%s; lag=%s; "
        "numerica=%s; diferenca numerica maxima=%.6g",
        comparison["same_rows"],
        comparison["same_keys"],
        comparison["same_nbytes"],
        comparison["same_lag"],
        comparison["numerically_equivalent"],
        comparison["max_numeric_abs_diff"],
    )
    LOGGER.info("Resumo: %s", summary_output)
    LOGGER.info("Detalhes legado: %s", legacy_output)
    LOGGER.info("Detalhes atual: %s", current_output)
    return summary_output


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara ENCODEC legado sequencial e pipeline atual multi-GPU."
    )
    parser.add_argument("--clips-per-dataset", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=DEFAULT_GPU_MEMORY_FRACTION,
    )
    parser.add_argument(
        "--gpu-prefetch-batches",
        type=int,
        default=DEFAULT_GPU_PREFETCH_BATCHES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-name", default=MODEL_NAME, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.clips_per_dataset < 1:
        parser.error("--clips-per-dataset deve ser maior que zero.")
    if arguments.batch_size < 1:
        parser.error("--batch-size deve ser maior que zero.")
    if not 0 < arguments.gpu_memory_fraction < 1:
        parser.error("--gpu-memory-fraction deve estar entre zero e um.")
    if arguments.gpu_prefetch_batches < 1:
        parser.error("--gpu-prefetch-batches deve ser maior que zero.")
    return arguments


def main() -> None:
    arguments = _parse_arguments()
    run_benchmark(
        clips_per_dataset=arguments.clips_per_dataset,
        batch_size=arguments.batch_size,
        gpu_memory_fraction=arguments.gpu_memory_fraction,
        gpu_prefetch_batches=arguments.gpu_prefetch_batches,
        seed=arguments.seed,
        model_name=arguments.model_name,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.error("Benchmark interrompido pelo usuario.")
        raise SystemExit(130) from None
    except Exception:
        LOGGER.exception("Falha no benchmark ENCODEC.")
        raise SystemExit(1) from None
