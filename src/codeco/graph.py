"""Generate the original scatter and rate-distortion figures."""

from __future__ import annotations

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

from codeco.analyze import PLOT_INDICES, load_metrics
from codeco.codeco_path import CodecoPath
from codeco.prepare_table import TqdmLoggingHandler, load_standalone_examples


LOGGER = logging.getLogger("codeco.graph")


def _configure_logging(paths: CodecoPath, log_path: Path | None = None) -> Path:
    log = log_path or paths.logs / "graph.log"
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


def _header() -> None:
    line = "=" * 72
    command = subprocess.list2cmdline(
        [sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    )
    LOGGER.info("\n%s\nGRAFICOS FINAIS\n[CMD]   %s\n%s", line, command, line)


def _section(title: str) -> None:
    LOGGER.info("\n%s\n%s", title, "-" * 72)


def _save_figure(figure: object, output: Path, **kwargs: object) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        figure.savefig(temporary, format="png", **kwargs)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(
    ffmpeg_path: Path | None = None,
    encodec_path: Path | None = None,
    rate_path: Path | None = None,
    output_directory: Path | None = None,
    log_path: Path | None = None,
    expected_keys: set[tuple[str, str]] | None = None,
) -> list[Path]:
    paths = CodecoPath.discover()
    os.environ.setdefault("MPLCONFIGDIR", str(paths.cache / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    log = _configure_logging(paths, log_path)
    _header()
    started = time.perf_counter()
    rate_path = rate_path or paths.tables / "rate_distortion.csv"
    output_directory = output_directory or paths.graphs
    if not rate_path.is_file():
        raise FileNotFoundError(
            f"Execute `python -m codeco.analyze` antes dos graficos: {rate_path}"
        )
    _, paired, indices = load_metrics(
        ffmpeg_path, encodec_path, expected_keys=expected_keys
    )
    rate_distortion = pd.read_csv(rate_path)
    selected_indices = [index for index in PLOT_INDICES if index in indices]
    datasets = list(paired["dataset"].drop_duplicates())
    outputs = []

    _section("Configuracao")
    LOGGER.info("Log: %s", log)
    LOGGER.info("Graficos: %s", output_directory)
    LOGGER.info("Datasets: %s", ", ".join(datasets))
    LOGGER.info("Indices: %s", ", ".join(selected_indices))
    total = len(datasets) * len(selected_indices) + rate_distortion["dataset"].nunique()
    progress = tqdm(total=total, desc="graficos", unit="grafico", dynamic_ncols=True)
    try:
        for dataset, dataset_group in paired.groupby("dataset"):
            for index in selected_indices:
                groups = list(dataset_group.groupby(["codec", "setting"]))
                count = len(groups)
                columns = min(4, count)
                rows = int(np.ceil(count / columns))
                figure, axes = plt.subplots(
                    rows,
                    columns,
                    figsize=(3.2 * columns, 3.2 * rows),
                    squeeze=False,
                )
                for axis, ((codec, setting), group) in zip(axes.ravel(), groups):
                    axis.scatter(group[f"{index}_base"], group[index], s=6, alpha=0.5)
                    low = np.nanmin(group[[index, f"{index}_base"]].values)
                    high = np.nanmax(group[[index, f"{index}_base"]].values)
                    axis.plot([low, high], [low, high], "k--", lw=0.8)
                    axis.set_title(f"{codec}\n{setting[:28]}", fontsize=7)
                    axis.set_xlabel("baseline")
                    axis.set_ylabel("codec")
                figure.suptitle(f"{dataset}\n{index}")
                figure.tight_layout()
                output = output_directory / f"scatter_{dataset}_{index}.png"
                _save_figure(figure, output, dpi=150)
                plt.close(figure)
                outputs.append(output)
                progress.update(1)

        for dataset, group in rate_distortion.groupby("dataset"):
            figure, axis = plt.subplots(figsize=(6, 4.5))
            for label, curve in group.groupby("curve_label"):
                curve = curve.sort_values("bitrate_kbps")
                axis.plot(curve["bitrate_kbps"], curve["mel_ssim"], "o-", label=label)
            axis.set_xscale("log")
            axis.set_xlabel("Bitrate (kbps)")
            axis.set_ylabel("Mel-Spectrogram SSIM")
            axis.legend(fontsize=8)
            axis.set_title(dataset)
            output = output_directory / f"rate_distortion_{dataset}.png"
            _save_figure(figure, output, dpi=150, bbox_inches="tight")
            plt.close(figure)
            outputs.append(output)
            progress.update(1)
    finally:
        progress.close()

    _section("Resultados")
    LOGGER.info("Graficos gerados: %d", len(outputs))
    LOGGER.info("Diretorio: %s", output_directory)
    LOGGER.info("Tempo total: %.2f s", time.perf_counter() - started)
    return outputs


def main() -> None:
    paths = CodecoPath.discover()
    analyze_directory = paths.tests / "analyze"
    graph_directory = paths.tests / "graph"
    examples = load_standalone_examples().table.select(
        ["dataset", "clip_id"]
    ).to_pandas()
    expected_keys = set(
        map(
            tuple,
            examples.astype(str).itertuples(index=False, name=None),
        )
    )
    run(
        ffmpeg_path=paths.tests / "ffmpeg" / "results.csv",
        encodec_path=paths.tests / "encodec" / "results.csv",
        rate_path=analyze_directory / "rate_distortion.csv",
        output_directory=graph_directory,
        log_path=graph_directory / "execution.log",
        expected_keys=expected_keys,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception:
        LOGGER.exception("Falha na geracao dos graficos.")
        raise SystemExit(1) from None
