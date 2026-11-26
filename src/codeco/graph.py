"""Generate the pipeline figures, including those of the experimental notebooks."""

from __future__ import annotations

import json
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

from codeco.analyze import PLOT_INDICES, index_preservation_summary, load_metrics
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


GRAPH_FIGURES_FILE_NAME = "graph_figures.json"


def _graph_figures_config(paths: CodecoPath) -> dict:
    path = paths.config / GRAPH_FIGURES_FILE_NAME
    values = json.loads(path.read_text(encoding="utf-8")).get("class_selection_figures", {})
    if "sensitivity_k_sigma" not in values:
        raise ValueError(f"Configuracao de figuras incompleta em {path}: sensitivity_k_sigma")
    return values


def _corpus_figures(paths: CodecoPath, plt, output_directory: Path) -> list[Path]:
    """Notebooks 01 and 02: natural, sensitivity and selected class distributions."""
    from codeco.corpora import (
        CORPUS_MANIFESTS,
        SelectionRule,
        class_distribution,
        load_manifest_windows,
        select_classes,
    )
    from codeco.prepare_table import _class_selection_config

    selection = _class_selection_config(paths)
    sensitivity_k = [float(value) for value in _graph_figures_config(paths)["sensitivity_k_sigma"]]
    manifests = {
        name: load_manifest_windows(name, paths.manifest) for name in CORPUS_MANIFESTS
    }
    outputs = []

    fig, axes = plt.subplots(len(manifests), 1, figsize=(12, 5 * len(manifests)))
    if len(manifests) == 1:
        axes = [axes]
    for ax, (corpus, df) in zip(axes, manifests.items()):
        dist = class_distribution(df)
        ax.bar(range(len(dist)), dist["windows"], color="#3b7d5b")
        ax.set_yscale("log")
        ax.set_title(f"{corpus}: {len(dist)} classes")
        ax.set_ylabel("janelas (log)")
        ax.set_xlabel("classes ordenadas por frequência")
    fig.tight_layout()
    output = output_directory / "natural_class_distributions.png"
    _save_figure(fig, output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(output)

    sensitivity = []
    for corpus, df in manifests.items():
        for k in sensitivity_k:
            _, meta = select_classes(
                df,
                SelectionRule(
                    k_sigma=k,
                    minimum_groups=int(selection["minimum_groups"]),
                    seed=int(selection["seed"]),
                ),
            )
            sensitivity.append(
                {"corpus": corpus, "k_sigma": k, "classes": meta["selected_class_count"], "cutoff": meta["count_cutoff"]}
            )
    sensitivity = pd.DataFrame(sensitivity)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for corpus, group in sensitivity.groupby("corpus"):
        axes[0].plot(group.k_sigma, group.classes, "o-", label=corpus)
        axes[1].plot(group.k_sigma, group.cutoff, "o-", label=corpus)
    axes[0].set(xlabel="k σ", ylabel="classes selecionadas", title="Número de classes")
    axes[1].set(xlabel="k σ", ylabel="janelas mínimas no treino", title="Limiar de suporte")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    output = output_directory / "class_selection_sensitivity.png"
    _save_figure(fig, output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(output)

    rule = SelectionRule(
        k_sigma=float(selection["k_sigma"]),
        minimum_groups=int(selection["minimum_groups"]),
        seed=int(selection["seed"]),
        fit_split=str(selection["fit_split"]),
    )
    selection_tables = {corpus: select_classes(df, rule)[0] for corpus, df in manifests.items()}
    fig, axes = plt.subplots(len(selection_tables), 1, figsize=(12, 5 * len(selection_tables)))
    if len(selection_tables) == 1:
        axes = [axes]
    for ax, (corpus, table) in zip(axes, selection_tables.items()):
        colors = table.selected.map({True: "#2b8c5a", False: "#b8b8b8"})
        ax.bar(range(len(table)), table.windows, color=colors)
        ax.axhline(table.count_cutoff.iloc[0], color="#a33", linestyle="--", label="limiar")
        ax.set_yscale("log")
        ax.set_title(f"{corpus}: {int(table.selected.sum())} classes selecionadas")
        ax.set_ylabel("janelas de treino (log)")
        ax.legend()
    fig.tight_layout()
    output = output_directory / "selected_class_distributions.png"
    _save_figure(fig, output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(output)
    return outputs


def _perch_padding_figure(plt, output_directory: Path) -> Path:
    """Notebook 03: Perch adapter, 4 s plus symmetric padding."""
    from codeco.protocol import PROTOCOL, adapt_for_perch

    x = np.ones(PROTOCOL.window_samples, dtype=np.float32)
    perch = adapt_for_perch(x)
    t = np.arange(len(perch)) / PROTOCOL.perch_sample_rate_hz
    padding = (PROTOCOL.perch_seconds - PROTOCOL.window_seconds) / 2
    fig = plt.figure(figsize=(11, 2.8))
    plt.plot(t, perch, color="#356a8a")
    plt.axvspan(0, padding, color="#ccc", alpha=.5, label="padding")
    plt.axvspan(PROTOCOL.perch_seconds - padding, PROTOCOL.perch_seconds, color="#ccc", alpha=.5)
    plt.xlabel("tempo (s)"); plt.ylabel("amplitude"); plt.title("Adaptador Perch: 4 s + padding simétrico")
    plt.legend(); plt.tight_layout()
    output = output_directory / "perch_padding.png"
    _save_figure(fig, output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def _payload_distortion_figure(lossy: pd.DataFrame, plt, output_directory: Path) -> Path:
    """Notebook 05: mean effective bitrate against mean log-Mel SSIM per condition."""
    summary = (lossy.groupby(["dataset", "codec", "setting"], as_index=False, dropna=False)
               .agg(bitrate_kbps_mean=("bitrate_kbps", "mean"),
                    mel_ssim_mean=("mel_ssim", "mean")))
    fig, ax = plt.subplots(figsize=(8, 5))
    for (dataset, codec), group in summary.groupby(["dataset", "codec"]):
        group = group.sort_values("bitrate_kbps_mean")
        ax.plot(group.bitrate_kbps_mean, group.mel_ssim_mean, "o-",
                label=f"{dataset} — {codec}")
    ax.set_xscale("log")
    ax.set(xlabel="bitrate efetivo médio (kbit/s)", ylabel="SSIM log-Mel médio")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    output = output_directory / "payload_distortion.png"
    _save_figure(fig, output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def _index_preservation_figures(
    paired: pd.DataFrame,
    index_columns: list[str],
    plt,
    output_directory: Path,
) -> list[Path]:
    """Notebook 06, figures 1 and 2: agreement heatmaps and full-matrix scatters."""
    index_summary = index_preservation_summary(paired, index_columns)
    outputs = []
    preferred_order = [index for index in PLOT_INDICES if index in index_columns]
    for dataset, dataset_summary in index_summary.groupby("dataset"):
        work = dataset_summary.copy()
        work["condition"] = work["codec"] + " | " + work["setting"]
        ccc_matrix = work.pivot(index="condition", columns="index", values="ccc").reindex(columns=preferred_order)
        nmae_matrix = work.pivot(index="condition", columns="index", values="normalized_mae").reindex(
            index=ccc_matrix.index, columns=preferred_order
        )
        finite_nmae = nmae_matrix.to_numpy(float)[np.isfinite(nmae_matrix.to_numpy(float))]
        nmae_limit = max(float(np.quantile(finite_nmae, 0.95)), 1e-9) if finite_nmae.size else 1.0

        height = max(5.0, 0.34 * len(ccc_matrix.index))
        fig, axes = plt.subplots(1, 2, figsize=(16, height), constrained_layout=True)
        ccc_image = axes[0].imshow(
            np.ma.masked_invalid(ccc_matrix.to_numpy(float)), aspect="auto",
            cmap="coolwarm", vmin=-1, vmax=1,
        )
        nmae_image = axes[1].imshow(
            np.ma.masked_invalid(nmae_matrix.to_numpy(float)), aspect="auto",
            cmap="viridis", vmin=0, vmax=nmae_limit,
        )
        for axis, title in zip(axes, ("CCC (maior é melhor)", "MAE normalizado (menor é melhor)")):
            axis.set_title(title)
            axis.set_xticks(range(len(preferred_order)), [name.removeprefix("idx_") for name in preferred_order], rotation=45, ha="right")
            axis.set_yticks(range(len(ccc_matrix.index)), ccc_matrix.index, fontsize=7)
            axis.set_xlabel("índice ecoacústico")
        axes[0].set_ylabel("codec | configuração")
        fig.colorbar(ccc_image, ax=axes[0], shrink=0.8)
        fig.colorbar(nmae_image, ax=axes[1], shrink=0.8, label=f"escala visual até P95 = {nmae_limit:.3g}")
        fig.suptitle(f"Preservação dos índices — {dataset}")
        output = output_directory / f"index_preservation_heatmap_{dataset}.png"
        _save_figure(fig, output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)

    condition_rates = (paired.groupby(["dataset", "codec", "setting"], as_index=False)
                       .agg(bitrate_kbps_mean=("bitrate_kbps", "mean")))
    for dataset, dataset_conditions in condition_rates.groupby("dataset"):
        panels = list(dataset_conditions.sort_values(["codec", "bitrate_kbps_mean"]).itertuples(index=False))
        columns = min(4, len(panels))
        rows_count = int(np.ceil(len(panels) / columns))
        for scatter_index in index_columns:
            fig, axes = plt.subplots(
                rows_count, columns,
                figsize=(3.6 * columns, 3.45 * rows_count),
                squeeze=False,
            )
            for axis, condition in zip(axes.ravel(), panels):
                group = paired.loc[
                    (paired.dataset == dataset)
                    & (paired.codec == condition.codec)
                    & (paired.setting == condition.setting)
                ]
                reference = group[f"{scatter_index}_base"].to_numpy(float)
                decoded = group[scatter_index].to_numpy(float)
                valid = np.isfinite(reference) & np.isfinite(decoded)
                reference, decoded = reference[valid], decoded[valid]
                axis.scatter(reference, decoded, s=12, alpha=0.5)
                if len(reference):
                    low = float(min(reference.min(), decoded.min()))
                    high = float(max(reference.max(), decoded.max()))
                    axis.plot([low, high], [low, high], "k--", linewidth=0.9)
                metrics_row = index_summary.loc[
                    (index_summary.dataset == dataset)
                    & (index_summary.codec == condition.codec)
                    & (index_summary.setting == condition.setting)
                    & (index_summary["index"] == scatter_index)
                ]
                if metrics_row.empty:
                    metric_label = "CCC/MAE-n indisponíveis"
                else:
                    metric = metrics_row.iloc[0]
                    metric_label = f"CCC={metric.ccc:.3f} | MAE-n={metric.normalized_mae:.3f}"
                axis.set_title(
                    f"{condition.codec} | {condition.setting}\n"
                    f"{condition.bitrate_kbps_mean:.1f} kbit/s | {metric_label}",
                    fontsize=7.5,
                )
                axis.set_xlabel("referência")
                axis.set_ylabel("reconstruído")
                axis.grid(True, alpha=0.2)
            for axis in axes.ravel()[len(panels):]:
                axis.set_visible(False)
            fig.suptitle(f"{dataset} — {scatter_index} — todas as condições", fontsize=12)
            fig.tight_layout()
            output = output_directory / f"index_scatter_{dataset}_{scatter_index}.png"
            _save_figure(fig, output, dpi=160, bbox_inches="tight")
            plt.close(fig)
            outputs.append(output)
    return outputs


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
    lossy, paired, indices = load_metrics(
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
    total += 3 + 1 + 1 + len(datasets) * (1 + len(indices))
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

        for figures in (
            _corpus_figures(paths, plt, output_directory),
            [_perch_padding_figure(plt, output_directory)],
            [_payload_distortion_figure(lossy, plt, output_directory)],
            _index_preservation_figures(paired, list(indices), plt, output_directory),
        ):
            outputs.extend(figures)
            progress.update(len(figures))
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
