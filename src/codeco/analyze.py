"""Faithful statistical analysis of codec metrics from the original pipeline."""

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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score, f1_score

from codeco.codeco_path import CodecoPath
from codeco.prepare_table import TqdmLoggingHandler, load_standalone_examples


PLOT_INDICES = (
    "idx_ACI",
    "idx_ADI",
    "idx_AEI",
    "idx_H",
    "idx_Ht",
    "idx_Hf",
    "idx_BIO_2_24",
    "idx_BIO_2_8",
    "idx_NDSI_CLASSIC",
    "idx_NDSI_EXTENDED",
)
KEY_COLUMNS = ("dataset", "clip_id", "codec", "setting")
RATE_COLUMNS = ("bitrate_kbps", "mel_ssim", "snr_db", "mel_rmse_db")
LOGGER = logging.getLogger("codeco.analyze")
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 42


def _benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not valid.size:
        return adjusted
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _cluster_error_intervals(
    frame: pd.DataFrame,
    group_column: str,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Vectorized cluster bootstrap for bias, MAE and RMSE."""
    grouped = frame.groupby(group_column)["delta"].agg(
        count="size",
        total="sum",
        absolute=lambda values: np.abs(values).sum(),
        squared=lambda values: np.square(values).sum(),
    )
    if len(grouped) < 2:
        nan = (float("nan"), float("nan"))
        return {"bias": nan, "mae": nan, "rmse": nan}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(BOOTSTRAP_REPLICATES, len(grouped)))
    values = grouped[["count", "total", "absolute", "squared"]].to_numpy(dtype=float)
    sampled = values[draws].sum(axis=1)
    counts = sampled[:, 0]
    estimates = {
        "bias": sampled[:, 1] / counts,
        "mae": sampled[:, 2] / counts,
        "rmse": np.sqrt(sampled[:, 3] / counts),
    }
    return {
        name: tuple(np.quantile(metric, [0.025, 0.975]))
        for name, metric in estimates.items()
    }


def _pareto_mask(frame: pd.DataFrame) -> np.ndarray:
    values = frame[["bitrate_kbps", "mel_rmse_db"]].to_numpy(dtype=float)
    quality = -frame["mel_ssim"].to_numpy(dtype=float)
    objectives = np.column_stack([values, quality])
    keep = np.ones(len(frame), dtype=bool)
    for index, point in enumerate(objectives):
        dominated = np.all(objectives <= point, axis=1) & np.any(objectives < point, axis=1)
        dominated[index] = False
        keep[index] = not dominated.any()
    return keep


def _classification_metrics(y_true: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    f1 = float(f1_score(y_true, prediction, average="macro", zero_division=0))
    balanced = float(
        np.mean(
            [
                balanced_accuracy_score(y_true[:, index], prediction[:, index])
                for index in range(y_true.shape[1])
            ]
        )
    )
    return f1, balanced


def _classification_bootstrap(
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict[str, float]:
    unique = np.unique(groups.astype(str))
    grouped = {group: np.flatnonzero(groups.astype(str) == group) for group in unique}
    if len(unique) < 2:
        return {name: float("nan") for name in (
            "f1_delta_ci_low", "f1_delta_ci_high", "balanced_delta_ci_low",
            "balanced_delta_ci_high", "f1_noninferiority_p", "balanced_noninferiority_p",
        )}
    rng = np.random.default_rng(seed)
    f1_deltas, balanced_deltas = [], []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([grouped[group] for group in selected])
        baseline = _classification_metrics(y_true[indices], baseline_prediction[indices])
        candidate = _classification_metrics(y_true[indices], candidate_prediction[indices])
        f1_deltas.append(candidate[0] - baseline[0])
        balanced_deltas.append(candidate[1] - baseline[1])
    f1_values = np.asarray(f1_deltas)
    balanced_values = np.asarray(balanced_deltas)
    return {
        "f1_delta_ci_low": float(np.quantile(f1_values, 0.025)),
        "f1_delta_ci_high": float(np.quantile(f1_values, 0.975)),
        "balanced_delta_ci_low": float(np.quantile(balanced_values, 0.025)),
        "balanced_delta_ci_high": float(np.quantile(balanced_values, 0.975)),
        "f1_noninferiority_p": float(np.mean(f1_values <= -0.02)),
        "balanced_noninferiority_p": float(np.mean(balanced_values <= -0.02)),
    }


def _configure_logging(paths: CodecoPath, log_path: Path | None = None) -> Path:
    log = log_path or paths.logs / "analyze.log"
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
    LOGGER.info("\n%s\nANALISE ESTATISTICA FINAL\n[CMD]   %s\n%s", line, command, line)


def _section(title: str) -> None:
    LOGGER.info("\n%s\n%s", title, "-" * 72)


def _write_csv(table: pd.DataFrame, output: Path) -> None:
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
        table.to_csv(temporary, index=False)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ccc(a: np.ndarray, b: np.ndarray) -> float:
    """Lin's concordance correlation coefficient, identical to the original."""
    a, b = np.asarray(a), np.asarray(b)
    return float(
        2
        * np.cov(a, b, ddof=0)[0, 1]
        / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-20)
    )


def _codec_curve_label(row: pd.Series) -> str:
    if row["codec"] == "ae":
        variant = row["setting"].split(",")[0].replace("variant=", "")
        return f"ae_{variant}"
    return str(row["codec"])


def load_metrics(
    ffmpeg_path: Path | None = None,
    encodec_path: Path | None = None,
    expected_keys: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Rebuild the original unified codec result set from the split stages."""
    paths = CodecoPath.discover()
    ffmpeg_path = ffmpeg_path or paths.tables / "ffmpeg_metrics.csv"
    encodec_path = encodec_path or paths.tables / "encodec_metrics.csv"
    ffmpeg = pd.read_csv(ffmpeg_path, dtype={"clip_id": str})
    encodec = pd.read_csv(encodec_path, dtype={"clip_id": str})
    indices = [column for column in ffmpeg.columns if column.startswith("idx_")]
    required = {*KEY_COLUMNS, *RATE_COLUMNS, *indices}
    for name, table in (("FFmpeg", ffmpeg), ("ENCODEC", encodec)):
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(f"{name}: colunas ausentes: {', '.join(missing)}")
        duplicates = table.duplicated(list(KEY_COLUMNS), keep=False)
        if duplicates.any():
            raise ValueError(
                f"{name}: {int(duplicates.sum())} linhas possuem chaves duplicadas."
            )

    baseline = ffmpeg.loc[ffmpeg["codec"] == "baseline"].copy()
    if baseline.empty:
        raise ValueError("FFmpeg: nenhuma baseline foi encontrada.")
    baseline_keys = baseline.duplicated(["dataset", "clip_id"], keep=False)
    if baseline_keys.any():
        raise ValueError(
            "FFmpeg: "
            f"{int(baseline_keys.sum())} baselines repetem dataset + clip_id."
        )
    if expected_keys is not None:
        baseline_key_set = set(
            map(
                tuple,
                baseline[["dataset", "clip_id"]]
                .astype(str)
                .itertuples(index=False, name=None),
            )
        )
        if baseline_key_set != expected_keys:
            raise ValueError(
                "FFmpeg autonomo incompleto: "
                f"esperados {len(expected_keys)} clipes; encontrados "
                f"{len(baseline_key_set)}. Execute `python -m codeco.ffmpeg` "
                "sem limitar os clipes."
            )
    lossy = pd.concat(
        [
            ffmpeg.loc[ffmpeg["codec"] != "baseline"],
            encodec.loc[encodec["codec"] != "baseline"],
        ],
        ignore_index=True,
    )
    duplicates = lossy.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicates.any():
        raise ValueError(
            f"Codecs: {int(duplicates.sum())} linhas possuem chaves duplicadas."
        )
    if expected_keys is not None:
        for (codec, setting), group in lossy.groupby(["codec", "setting"]):
            condition_keys = set(
                map(
                    tuple,
                    group[["dataset", "clip_id"]]
                    .astype(str)
                    .itertuples(index=False, name=None),
                )
            )
            if condition_keys != expected_keys:
                raise ValueError(
                    f"Condicao autonoma incompleta: {codec} | {setting}; "
                    f"esperados {len(expected_keys)} clipes; encontrados "
                    f"{len(condition_keys)}."
                )
    paired = lossy.merge(
        baseline[["dataset", "clip_id", *indices]],
        on=["dataset", "clip_id"],
        suffixes=("", "_base"),
        validate="many_to_one",
    )
    if len(paired) != len(lossy):
        raise ValueError(
            f"Pareamento incompleto: {len(paired)} de {len(lossy)} linhas."
        )
    return lossy, paired, indices


def index_preservation_summary(paired: pd.DataFrame, index_columns: list[str]) -> pd.DataFrame:
    """Notebook 06: preservation statistics per condition and index."""
    from scipy.stats import pearsonr, spearmanr

    rows = []
    for keys, group in paired.groupby(["dataset", "codec", "setting"]):
        for index in index_columns:
            a = group[f"{index}_base"].to_numpy(float)
            b = group[index].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            if not ok.any():
                continue
            reference, decoded = a[ok], b[ok]
            delta = decoded - reference
            reference_std = float(reference.std(ddof=0))
            decoded_std = float(decoded.std(ddof=0))
            correlations_valid = len(reference) > 3 and reference_std > 0 and decoded_std > 0
            pearson_value = float(pearsonr(reference, decoded).statistic) if correlations_valid else np.nan
            spearman_value = float(spearmanr(reference, decoded).statistic) if correlations_valid else np.nan
            if correlations_valid:
                covariance = float(np.cov(reference, decoded, ddof=0)[0, 1])
                denominator = reference.var() + decoded.var() + (reference.mean() - decoded.mean()) ** 2
                ccc_value = float(2 * covariance / denominator) if denominator > 0 else np.nan
            else:
                ccc_value = np.nan
            bias = float(delta.mean())
            mae = float(np.abs(delta).mean())
            rows.append({
                "dataset": keys[0], "codec": keys[1], "setting": keys[2], "index": index,
                "n_pairs": int(len(reference)),
                "bitrate_kbps_mean": float(group.loc[ok, "bitrate_kbps"].mean()),
                "reference_mean": float(reference.mean()), "reference_std": reference_std,
                "reconstructed_mean": float(decoded.mean()), "reconstructed_std": decoded_std,
                "bias": bias, "mae": mae, "rmse": float(np.sqrt(np.mean(delta ** 2))),
                "standardized_bias": bias / reference_std if reference_std > 0 else np.nan,
                "normalized_mae": mae / reference_std if reference_std > 0 else np.nan,
                "pearson": pearson_value, "spearman": spearman_value, "ccc": ccc_value,
            })
    return pd.DataFrame(rows)


def reconstruction_summary(lossy: pd.DataFrame) -> pd.DataFrame:
    """Notebook 05: one row per corpus x codec x configuration."""
    return (lossy[lossy.codec != "baseline"]
            .groupby(["dataset", "codec", "setting"], as_index=False, dropna=False)
            .agg(n_windows=("clip_id", "nunique"),
                 bytes_mean=("nbytes", "mean"), bytes_std=("nbytes", "std"),
                 bitrate_kbps_mean=("bitrate_kbps", "mean"),
                 compression_ratio_mean=("compression_ratio", "mean"),
                 snr_db_mean=("snr_db", "mean"), snr_db_std=("snr_db", "std"),
                 mel_mae_db_mean=("mel_mae_db", "mean"),
                 mel_rmse_db_mean=("mel_rmse_db", "mean"),
                 mel_rmse_db_std=("mel_rmse_db", "std"),
                 mel_ssim_mean=("mel_ssim", "mean"), mel_ssim_std=("mel_ssim", "std"),
                 mel_cosine_mean=("mel_cosine", "mean")))


def index_diagnostics(index_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Notebook 06: ordered diagnostics of the index preservation table."""
    diagnostic_columns = [
        "dataset", "codec", "setting", "index", "n_pairs",
        "bitrate_kbps_mean", "standardized_bias", "normalized_mae",
        "pearson", "spearman", "ccc",
    ]
    largest_normalized_errors = (index_summary
        .sort_values("normalized_mae", ascending=False)
        .loc[:, diagnostic_columns]
        .head(15))
    agreement_gaps = index_summary.assign(agreement_gap=index_summary.spearman - index_summary.ccc)
    largest_agreement_gaps = (agreement_gaps
        .sort_values("agreement_gap", ascending=False)
        .loc[:, [*diagnostic_columns, "agreement_gap"]]
        .head(15))
    best_ccc_by_index = (index_summary.dropna(subset=["ccc"])
        .sort_values("ccc", ascending=False)
        .groupby(["dataset", "index"], as_index=False)
        .head(1)
        .loc[:, diagnostic_columns])
    return {
        "index_largest_normalized_errors": largest_normalized_errors,
        "index_largest_agreement_gaps": largest_agreement_gaps,
        "index_best_ccc": best_ccc_by_index,
    }


def run(
    ffmpeg_path: Path | None = None,
    encodec_path: Path | None = None,
    correlation_output: Path | None = None,
    rate_output: Path | None = None,
    log_path: Path | None = None,
    expected_keys: set[tuple[str, str]] | None = None,
) -> tuple[Path, ...]:
    paths = CodecoPath.discover()
    log = _configure_logging(paths, log_path)
    _header()
    started = time.perf_counter()
    ffmpeg_path = ffmpeg_path or paths.tables / "ffmpeg_metrics.csv"
    encodec_path = encodec_path or paths.tables / "encodec_metrics.csv"
    correlation_output = correlation_output or paths.tables / "index_correlations.csv"
    rate_output = rate_output or paths.tables / "rate_distortion.csv"
    paired_output = paths.tables / "paired_index_errors.csv"
    alignment_output = paths.tables / "alignment_summary.csv"
    pareto_output = paths.tables / "pareto_front.csv"
    classification_output = paths.tables / "classification_noninferiority.csv"

    _section("Entradas")
    LOGGER.info("FFmpeg: %s", ffmpeg_path)
    LOGGER.info("ENCODEC: %s", encodec_path)
    LOGGER.info("Log: %s", log)
    lossy, paired, indices = load_metrics(
        ffmpeg_path, encodec_path, expected_keys=expected_keys
    )
    LOGGER.info("Codecs reconstruidos: %d linhas.", len(lossy))
    LOGGER.info("Pares baseline -> codec: %d.", len(paired))
    LOGGER.info("Indices: %s.", ", ".join(indices))

    rows = []
    for (dataset, codec, setting), group in paired.groupby(
        ["dataset", "codec", "setting"]
    ):
        for index in indices:
            baseline_values = group[f"{index}_base"].to_numpy()
            codec_values = group[index].to_numpy()
            valid = np.isfinite(baseline_values) & np.isfinite(codec_values)
            if (
                valid.sum() > 3
                and baseline_values[valid].std() > 0
                and codec_values[valid].std() > 0
            ):
                pearson_value, pearson_p = pearsonr(
                    baseline_values[valid], codec_values[valid]
                )
                spearman_value, spearman_p = spearmanr(
                    baseline_values[valid], codec_values[valid]
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "codec": codec,
                        "setting": setting,
                        "index": index,
                        "pearson": pearson_value,
                        "pearson_p": pearson_p,
                        "spearman": spearman_value,
                        "spearman_p": spearman_p,
                        "ccc": ccc(baseline_values[valid], codec_values[valid]),
                    }
                )
    correlations = pd.DataFrame.from_records(
        rows,
        columns=("dataset", "codec", "setting", "index", "pearson", "pearson_p", "spearman", "spearman_p", "ccc"),
    )
    if not correlations.empty:
        correlations["pearson_p_bh"] = _benjamini_hochberg(
            correlations["pearson_p"].to_numpy()
        )
        correlations["spearman_p_bh"] = _benjamini_hochberg(
            correlations["spearman_p"].to_numpy()
        )
    _write_csv(correlations, correlation_output)

    clip_groups = None
    clip_cache = paths.cache / "clip_table.parquet"
    if clip_cache.is_file():
        clip_groups = pd.read_parquet(clip_cache)[
            ["dataset", "clip_id", "source_group_id"]
        ].astype({"clip_id": str})
        paired = paired.merge(
            clip_groups,
            on=["dataset", "clip_id"],
            how="left",
            validate="many_to_one",
        )
    paired["bootstrap_group"] = (
        paired["source_group_id"].fillna(paired["clip_id"]).astype(str)
        if "source_group_id" in paired
        else paired["clip_id"].astype(str)
    )
    error_rows = []
    for ordinal, ((dataset, codec, setting), group) in enumerate(
        paired.groupby(["dataset", "codec", "setting"])
    ):
        for index in indices:
            work = group[["bootstrap_group", index, f"{index}_base"]].dropna().copy()
            if work.empty:
                continue
            work["delta"] = work[index] - work[f"{index}_base"]
            bias = float(work["delta"].mean())
            mae = float(work["delta"].abs().mean())
            rmse = float(np.sqrt(np.mean(work["delta"] ** 2)))
            intervals = _cluster_error_intervals(
                work, "bootstrap_group", BOOTSTRAP_SEED + ordinal
            )
            bias_ci = intervals["bias"]
            mae_ci = intervals["mae"]
            rmse_ci = intervals["rmse"]
            error_rows.append(
                {
                    "dataset": dataset, "codec": codec, "setting": setting,
                    "index": index, "windows": len(work),
                    "groups": work["bootstrap_group"].nunique(),
                    "bias": bias, "bias_ci_low": bias_ci[0], "bias_ci_high": bias_ci[1],
                    "mae": mae, "mae_ci_low": mae_ci[0], "mae_ci_high": mae_ci[1],
                    "rmse": rmse, "rmse_ci_low": rmse_ci[0], "rmse_ci_high": rmse_ci[1],
                }
            )
    paired_errors = pd.DataFrame.from_records(error_rows)
    _write_csv(paired_errors, paired_output)

    rate_distortion = (
        lossy.groupby(["dataset", "codec", "setting"])[list(RATE_COLUMNS)]
        .mean()
        .reset_index()
    )
    rate_distortion["curve_label"] = rate_distortion.apply(
        _codec_curve_label, axis=1
    )
    _write_csv(rate_distortion, rate_output)
    pareto = rate_distortion.copy()
    pareto["pareto_optimal"] = False
    for _, positions in pareto.groupby("dataset").groups.items():
        pareto.loc[positions, "pareto_optimal"] = _pareto_mask(pareto.loc[positions])
    _write_csv(pareto, pareto_output)

    alignment_columns = [
        column for column in ("lag_samples", "lag_seconds", "alignment_score")
        if column in lossy.columns
    ]
    alignment = (
        lossy.groupby(["dataset", "codec", "setting"])[alignment_columns]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    ) if alignment_columns else pd.DataFrame()
    if not alignment.empty:
        alignment.columns = [
            "_".join(str(part) for part in column if part).rstrip("_")
            if isinstance(column, tuple) else str(column)
            for column in alignment.columns
        ]
    _write_csv(alignment, alignment_output)

    classification_rows = []
    probability_roots = sorted(
        paths.cache.rglob("probabilities_baseline_orig.npz")
    )
    for baseline_path in probability_roots:
        with np.load(baseline_path, allow_pickle=False) as baseline:
            y_test = baseline["y_test"]
            groups = baseline["test_groups"].astype(str)
            baseline_prediction = (
                baseline["test"] >= baseline["thresholds"][None, :]
            ).astype(np.int8)
            dataset = str(baseline["dataset"])
            foundation_model = str(baseline["foundation_model"])
            head = str(baseline["head"])
        baseline_metrics = _classification_metrics(y_test, baseline_prediction)
        classification_rows.append(
            {
                "dataset": dataset, "foundation_model": foundation_model,
                "head": head, "codec": "baseline", "setting": "orig",
                "protocol": "clean_to_clean", "f1_macro": baseline_metrics[0],
                "delta_f1_macro": 0.0, "f1_delta_ci_low": 0.0,
                "f1_delta_ci_high": 0.0, "f1_noninferior_margin_0_02": True,
                "f1_noninferiority_p": 0.0,
                "balanced_accuracy": baseline_metrics[1],
                "delta_balanced_accuracy": 0.0, "balanced_delta_ci_low": 0.0,
                "balanced_delta_ci_high": 0.0,
                "balanced_accuracy_noninferior_margin_0_02": True,
                "balanced_noninferiority_p": 0.0,
                "inference_scope": "cluster bootstrap by source_group_id",
            }
        )
        for condition_path in sorted(baseline_path.parent.glob("probabilities_*.npz")):
            if condition_path == baseline_path:
                continue
            with np.load(condition_path, allow_pickle=False) as condition:
                codec = str(condition["codec"])
                setting = str(condition["setting"])
                protocols = (
                    ("train_orig_test_comp", condition["clean_to_compressed"], condition["clean_thresholds"]),
                    ("train_comp_test_comp", condition["matched"], condition["matched_thresholds"]),
                )
                for protocol, probabilities, thresholds in protocols:
                    prediction = (probabilities >= thresholds[None, :]).astype(np.int8)
                    metrics = _classification_metrics(y_test, prediction)
                    intervals = _classification_bootstrap(
                        y_test,
                        baseline_prediction,
                        prediction,
                        groups,
                        BOOTSTRAP_SEED + len(classification_rows),
                    )
                    classification_rows.append(
                        {
                            "dataset": dataset, "foundation_model": foundation_model,
                            "head": head, "codec": codec, "setting": setting,
                            "protocol": protocol, "f1_macro": metrics[0],
                            "delta_f1_macro": metrics[0] - baseline_metrics[0],
                            **intervals,
                            "f1_noninferior_margin_0_02": intervals["f1_delta_ci_low"] >= -0.02,
                            "balanced_accuracy": metrics[1],
                            "delta_balanced_accuracy": metrics[1] - baseline_metrics[1],
                            "balanced_accuracy_noninferior_margin_0_02": intervals["balanced_delta_ci_low"] >= -0.02,
                            "inference_scope": "cluster bootstrap by source_group_id",
                        }
                    )

    # Backward-compatible screening for legacy tables without probabilities.
    if not classification_rows:
        for table_path in sorted(paths.tables.glob("classification_*_*.csv")):
            table = pd.read_csv(table_path)
            required = {"dataset", "foundation_model", "head", "codec", "setting", "protocol", "f1_macro", "balanced_accuracy"}
            if not required.issubset(table.columns):
                continue
            baseline_rows = table.loc[table["codec"] == "baseline"]
            if baseline_rows.empty:
                continue
            baseline_row = baseline_rows.iloc[0]
            for row in table.itertuples(index=False):
                classification_rows.append(
                    {
                        "dataset": row.dataset, "foundation_model": row.foundation_model,
                        "head": row.head, "codec": row.codec, "setting": row.setting,
                        "protocol": row.protocol, "f1_macro": row.f1_macro,
                        "delta_f1_macro": row.f1_macro - baseline_row.f1_macro,
                        "f1_noninferior_margin_0_02": row.f1_macro - baseline_row.f1_macro >= -0.02,
                        "balanced_accuracy": row.balanced_accuracy,
                        "delta_balanced_accuracy": row.balanced_accuracy - baseline_row.balanced_accuracy,
                        "balanced_accuracy_noninferior_margin_0_02": row.balanced_accuracy - baseline_row.balanced_accuracy >= -0.02,
                        "inference_scope": "legacy aggregate only; rerun classification for bootstrap",
                    }
                )
    classification = pd.DataFrame.from_records(classification_rows)
    if not classification.empty and "f1_noninferiority_p" in classification:
        classification["f1_noninferiority_p_bh"] = _benjamini_hochberg(
            classification["f1_noninferiority_p"].to_numpy(dtype=float)
        )
        classification["balanced_noninferiority_p_bh"] = _benjamini_hochberg(
            classification["balanced_noninferiority_p"].to_numpy(dtype=float)
        )
    _write_csv(classification, classification_output)

    reconstruction_output = paths.tables / "reconstruction_summary.csv"
    _write_csv(reconstruction_summary(lossy), reconstruction_output)
    index_summary = index_preservation_summary(paired, list(indices))
    preservation_output = paths.tables / "index_preservation.csv"
    _write_csv(index_summary, preservation_output)
    diagnostic_outputs = []
    if not index_summary.empty:
        for name, table in index_diagnostics(index_summary).items():
            diagnostic_outputs.append(paths.tables / f"{name}.csv")
            _write_csv(table, diagnostic_outputs[-1])

    _section("Resultados")
    LOGGER.info("Correlacoes: %d linhas | %s", len(correlations), correlation_output)
    LOGGER.info(
        "Taxa-distorcao: %d linhas | %s", len(rate_distortion), rate_output
    )
    LOGGER.info("Tempo total: %.2f s", time.perf_counter() - started)
    return (
        correlation_output,
        paired_output,
        rate_output,
        pareto_output,
        alignment_output,
        classification_output,
        reconstruction_output,
        preservation_output,
        *diagnostic_outputs,
    )


def main() -> None:
    paths = CodecoPath.discover()
    test_directory = paths.tests / "analyze"
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
        correlation_output=test_directory / "index_correlations.csv",
        rate_output=test_directory / "rate_distortion.csv",
        log_path=test_directory / "execution.log",
        expected_keys=expected_keys,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception:
        LOGGER.exception("Falha na analise estatistica final.")
        raise SystemExit(1) from None
