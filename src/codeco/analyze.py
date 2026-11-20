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


def run(
    ffmpeg_path: Path | None = None,
    encodec_path: Path | None = None,
    correlation_output: Path | None = None,
    rate_output: Path | None = None,
    log_path: Path | None = None,
    expected_keys: set[tuple[str, str]] | None = None,
) -> tuple[Path, Path]:
    paths = CodecoPath.discover()
    log = _configure_logging(paths, log_path)
    _header()
    started = time.perf_counter()
    ffmpeg_path = ffmpeg_path or paths.tables / "ffmpeg_metrics.csv"
    encodec_path = encodec_path or paths.tables / "encodec_metrics.csv"
    correlation_output = correlation_output or paths.tables / "index_correlations.csv"
    rate_output = rate_output or paths.tables / "rate_distortion.csv"

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
                rows.append(
                    {
                        "dataset": dataset,
                        "codec": codec,
                        "setting": setting,
                        "index": index,
                        "pearson": pearsonr(
                            baseline_values[valid], codec_values[valid]
                        )[0],
                        "spearman": spearmanr(
                            baseline_values[valid], codec_values[valid]
                        )[0],
                        "ccc": ccc(baseline_values[valid], codec_values[valid]),
                    }
                )
    correlations = pd.DataFrame.from_records(
        rows,
        columns=("dataset", "codec", "setting", "index", "pearson", "spearman", "ccc"),
    )
    _write_csv(correlations, correlation_output)

    rate_distortion = (
        lossy.groupby(["dataset", "codec", "setting"])[list(RATE_COLUMNS)]
        .mean()
        .reset_index()
    )
    rate_distortion["curve_label"] = rate_distortion.apply(
        _codec_curve_label, axis=1
    )
    _write_csv(rate_distortion, rate_output)

    _section("Resultados")
    LOGGER.info("Correlacoes: %d linhas | %s", len(correlations), correlation_output)
    LOGGER.info(
        "Taxa-distorcao: %d linhas | %s", len(rate_distortion), rate_output
    )
    LOGGER.info("Tempo total: %.2f s", time.perf_counter() - started)
    return correlation_output, rate_output


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
