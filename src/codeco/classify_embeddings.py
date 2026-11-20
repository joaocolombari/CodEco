"""Classify cached foundation-model embeddings without repeating audio processing."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import queue
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from codeco.codeco_path import CodecoPath
from codeco.prepare_table import TqdmLoggingHandler


IMPLEMENTATION_VERSION = "2"
DEFAULT_SEED = 42
DEFAULT_FOUNDATION_MODEL = "birdnet"
MLP_MAX_ITER = 300
EXPERIMENT_CONDITIONS = {
    "baseline": ("R0", "Referencia", "PCM s16le", "baseline principal"),
    "flac": ("R1", "Sem perdas", "FLAC", "referencia sem perdas"),
    "mp3": ("L1", "Legado", "MP3", "codec legado"),
    "aac": ("A1", "Perceptual opcional", "AAC-LC", "codec perceptual opcional"),
    "opus": ("M1", "Moderno", "Opus", "codec moderno"),
    "lc3": ("M2", "Moderno", "LC3", "codec moderno"),
    "encodec": ("N1", "Codec neural", "EnCodec", "codec neural"),
}
RESULT_COLUMNS = (
    "dataset",
    "foundation_model",
    "head",
    "codec",
    "setting",
    "protocol",
    "embedding_euclidean_mean",
    "embedding_euclidean_median",
    "embedding_cosine_mean",
    "embedding_correlation_mean",
    "f1_macro",
    "balanced_accuracy",
)
SOURCE_COLUMNS = (
    "dataset",
    "foundation_model",
    "codec",
    "setting",
    "clips",
    "completed",
    "clips_file",
    "embeddings_file",
)
LOGGER = logging.getLogger("codeco.classify_embeddings")
_CPU_CONDITION_STATE: dict[str, object] = {}


def _torch_model_for_transfer(model) -> dict[str, object]:
    transferred_models = []
    for binary in model["models"]:
        transferred = {
            "constant": binary["constant"],
            "epochs": binary["epochs"],
            "stop_reason": binary["stop_reason"],
            "loss_curve": binary["loss_curve"],
            "validation_loss_curve": binary.get("validation_loss_curve", []),
        }
        if binary["constant"] is None:
            transferred["parameters"] = [
                parameter.detach().cpu().numpy().copy()
                for parameter in binary["parameters"]
            ]
        transferred_models.append(transferred)
    return {"scaler": model["scaler"], "models": transferred_models}


def _torch_model_to_device(torch, model, device) -> dict[str, object]:
    device_models = []
    for binary in model["models"]:
        transferred = dict(binary)
        if binary["constant"] is None:
            transferred["parameters"] = [
                torch.as_tensor(parameter, device=device)
                for parameter in binary["parameters"]
            ]
        device_models.append(transferred)
    return {"scaler": model["scaler"], "models": device_models}


def _gpu_configuration_worker(
    gpu: int,
    tasks,
    completed,
    baseline_model,
    targets: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    seed: int,
    early_stopping: bool,
    monitor_training: bool,
) -> None:
    try:
        import torch
        from codeco.benchmark.birdnet_classification_benchmark import (
            _torch_predict,
            _torch_train,
        )

        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        clean = _torch_model_to_device(torch, baseline_model, device)
        gpu_name = torch.cuda.get_device_name(device)
        while True:
            source = tasks.get()
            if source is None:
                break
            try:
                compressed = np.load(source["embeddings_file"], mmap_mode="r")
                started = time.perf_counter()
                real_prediction = _torch_predict(
                    torch, clean, compressed[test], device
                )
                real_inference_seconds = time.perf_counter() - started
                matched, matched_training_seconds = _torch_train(
                    torch,
                    compressed,
                    targets,
                    train,
                    device,
                    seed,
                    early_stopping=early_stopping,
                    monitor=validation if monitor_training else None,
                )
                started = time.perf_counter()
                matched_validation_prediction = _torch_predict(
                    torch, matched, compressed[validation], device
                )
                matched_prediction = _torch_predict(
                    torch, matched, compressed[test], device
                )
                matched_inference_seconds = time.perf_counter() - started
                completed.put(
                    {
                        **source,
                        "gpu": gpu,
                        "gpu_name": gpu_name,
                        "real_prediction": real_prediction,
                        "matched_validation_prediction": matched_validation_prediction,
                        "matched_prediction": matched_prediction,
                        "real_inference_seconds": real_inference_seconds,
                        "matched_training_seconds": matched_training_seconds,
                        "matched_inference_seconds": matched_inference_seconds,
                        "training_epochs": _torch_training_epochs(matched),
                        "loss_curves": _torch_loss_curves(matched),
                        "validation_loss_curves": _torch_validation_loss_curves(
                            matched
                        ),
                    }
                )
            except Exception:
                completed.put(
                    {
                        **source,
                        "gpu": gpu,
                        "gpu_name": gpu_name,
                        "error": traceback.format_exc(),
                    }
                )
    except Exception:
        completed.put({"gpu": gpu, "error": traceback.format_exc()})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_logging(paths: CodecoPath, log_path: Path | None = None) -> Path:
    log_file = log_path or paths.logs / "classify_embeddings.log"
    if LOGGER.handlers:
        return log_file
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    terminal = TqdmLoggingHandler(sys.stdout)
    terminal.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(terminal)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    LOGGER.addHandler(file_handler)
    return log_file


def _header(foundation_model: str = DEFAULT_FOUNDATION_MODEL) -> None:
    line = "=" * 72
    command = subprocess.list2cmdline(
        [sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    )
    LOGGER.info(
        "\n%s\nCLASSIFICACAO DOS EMBEDDINGS %s\n[CMD]   %s\n%s",
        line,
        foundation_model.upper(),
        command,
        line,
    )


def _section(name: str) -> None:
    LOGGER.info("\n%s\n%s", name, "-" * 72)


def _experimental_condition(codec: str) -> tuple[str, str, str, str]:
    normalized = codec.lower()
    return EXPERIMENT_CONDITIONS.get(
        normalized,
        (normalized.upper(), "Nao classificada", normalized.upper(), "condicao"),
    )


def _log_experimental_legend(source: pd.DataFrame) -> None:
    _section("Legenda experimental")
    codecs = tuple(dict.fromkeys(source["codec"].astype(str).str.lower()))
    rows = []
    for codec in codecs:
        condition_id, family, condition_name, role = _experimental_condition(codec)
        rows.append(
            {
                "ID": condition_id,
                "Condicao": condition_name,
                "Familia": family,
                "Papel": role,
            }
        )
    LOGGER.info("%s", pd.DataFrame(rows).to_string(index=False))
    LOGGER.info(
        "R0 (PCM s16le) e a baseline principal; R1 (FLAC) e a referencia sem perdas."
    )
    LOGGER.info(
        "Experimento A - robustez sem retreinamento: treina e valida em R0 PCM; "
        "testa no audio reconstruido pelo codec."
    )
    LOGGER.info(
        "Experimento B - adaptacao ao codec: treina, valida e testa no audio "
        "reconstruido pela mesma configuracao de codec."
    )
    LOGGER.info(
        "VALIDACAO e TESTE permanecem separados e usam os mesmos splits e rotulos "
        "em todas as configuracoes."
    )


def _connect_checkpoint(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS birdnet_classification_runs (
            fingerprint TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            head TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS birdnet_classification_results (
            fingerprint TEXT NOT NULL,
            codec TEXT NOT NULL,
            setting TEXT NOT NULL,
            protocol TEXT NOT NULL,
            result_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (fingerprint, codec, setting, protocol)
        );
        """
    )
    return connection


def _atomic_csv(table: pd.DataFrame, output: Path) -> None:
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


def _atomic_json(value: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_npy(value: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp.npy",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        np.save(temporary, value)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _build_classifier(head: str, seed: int, early_stopping: bool = True):
    if head == "linear":
        base = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
    elif head == "mlp":
        base = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=1e-3,
            max_iter=MLP_MAX_ITER,
            early_stopping=early_stopping,
            random_state=seed,
        )
    else:
        raise ValueError(f"Head desconhecida: {head}")
    return make_pipeline(StandardScaler(), OneVsRestClassifier(base))


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    balanced = [
        balanced_accuracy_score(y_true[:, index], y_pred[:, index])
        for index in range(y_true.shape[1])
    ]
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(np.mean(balanced)),
    }


def _report_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    original = _classification_metrics(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_score": original["f1_macro"],
        "balanced_accuracy": original["balanced_accuracy"],
    }


def _sklearn_training_epochs(classifier, head: str) -> list[int]:
    if head != "mlp":
        return []
    one_vs_rest = classifier.steps[-1][1]
    return [int(estimator.n_iter_) for estimator in one_vs_rest.estimators_]


def _torch_training_epochs(classifier) -> list[int]:
    return [int(model["epochs"]) for model in classifier["models"]]


def _sklearn_loss_curves(classifier, head: str) -> list[list[float]]:
    if head != "mlp":
        return []
    one_vs_rest = classifier.steps[-1][1]
    return [
        [float(value) for value in estimator.loss_curve_]
        for estimator in one_vs_rest.estimators_
    ]


def _torch_loss_curves(classifier) -> list[list[float]]:
    return [
        [float(value) for value in model["loss_curve"]]
        for model in classifier["models"]
    ]


def _torch_validation_loss_curves(classifier) -> list[list[float]]:
    return [
        [float(value) for value in model.get("validation_loss_curve", [])]
        for model in classifier["models"]
    ]


def _loss_summary(loss_curves: Sequence[Sequence[float]]) -> dict[str, float] | None:
    available = [curve for curve in loss_curves if curve]
    if not available:
        return None
    return {
        "initial": float(np.mean([curve[0] for curve in available])),
        "final": float(np.mean([curve[-1] for curve in available])),
        "best": float(np.mean([min(curve) for curve in available])),
    }


def _log_performance_report(
    title: str,
    objective: str,
    training_source: str,
    rows: Sequence[tuple[str, str, np.ndarray, np.ndarray]],
) -> None:
    report_rows = []
    for split, input_source, expected, predicted in rows:
        metrics = _report_metrics(expected, predicted)
        report_rows.append(
            {
                "Conjunto": split,
                "Entrada avaliada": input_source,
                "Clipes": len(expected),
                "Acuracia": f'{metrics["accuracy"]:.6f}',
                "Precision": f'{metrics["precision"]:.6f}',
                "Recall": f'{metrics["recall"]:.6f}',
                "F1-score": f'{metrics["f1_score"]:.6f}',
                "Balanced accuracy": f'{metrics["balanced_accuracy"]:.6f}',
            }
        )
    LOGGER.info("\n%s", title)
    LOGGER.info("Objetivo: %s", objective)
    LOGGER.info("Treinamento da MLP: %s", training_source)
    LOGGER.info(
        "Acuracia = correspondencia exata de todos os rotulos do clipe; "
        "Precision, Recall e F1-score = macro entre classes."
    )
    LOGGER.info("%s", pd.DataFrame(report_rows).to_string(index=False))


def _mean_loss_curve(curves: Sequence[Sequence[float]]) -> np.ndarray:
    available = [np.asarray(curve, dtype=np.float64) for curve in curves if curve]
    if not available:
        return np.empty(0, dtype=np.float64)
    width = max(len(curve) for curve in available)
    values = np.full((len(available), width), np.nan, dtype=np.float64)
    for index, curve in enumerate(available):
        values[index, : len(curve)] = curve
    return np.nanmean(values, axis=0)


def _plot_training_curves(
    records: dict[int, tuple[str, list[list[float]], list[list[float]]]],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    if not records:
        LOGGER.warning("Nenhuma curva de treinamento disponivel para o grafico.")
        return
    columns = 3
    rows = int(np.ceil(len(records) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.0 * columns, 4.0 * rows),
        squeeze=False,
    )
    for axis, number in zip(axes.ravel(), sorted(records)):
        label, training_curves, validation_curves = records[number]
        training_mean = _mean_loss_curve(training_curves)
        validation_mean = _mean_loss_curve(validation_curves)
        if training_mean.size:
            axis.plot(
                np.arange(1, len(training_mean) + 1),
                training_mean,
                label="Treino",
                linewidth=1.5,
            )
        if validation_mean.size:
            axis.plot(
                np.arange(1, len(validation_mean) + 1),
                validation_mean,
                label="Validacao externa",
                linewidth=1.5,
            )
        axis.set_title(f"{number}. {label}", fontsize=9)
        axis.set_xlabel("Epoca")
        axis.set_ylabel("Binary cross-entropy media")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes.ravel()[len(records) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Curvas de loss da MLP por configuracao\n"
        "media dos classificadores binarios one-vs-rest",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    figure.savefig(temporary, dpi=160, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, output)
    LOGGER.info("Grafico de overfitting: %s", output)


def _epoch_summary(epochs: Sequence[int]) -> dict[str, float | int]:
    values = np.asarray(epochs, dtype=np.int64)
    return {
        "classifiers": int(values.size),
        "minimum": int(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "maximum": int(values.max()),
        "at_limit": int(np.count_nonzero(values == MLP_MAX_ITER)),
    }


def _cpu_condition_initializer(
    clips_file: str,
    baseline_file: str,
    clean_classifier,
    head: str,
    seed: int,
    early_stopping: bool,
) -> None:
    clips = pd.read_parquet(clips_file).sort_values("row_index").reset_index(drop=True)
    labels = [
        value.split("|") if isinstance(value, str) and value else []
        for value in clips["labels"]
    ]
    _CPU_CONDITION_STATE.clear()
    _CPU_CONDITION_STATE.update(
        targets=MultiLabelBinarizer().fit_transform(labels),
        train=clips["split"].eq("train").to_numpy(),
        validation=clips["split"].eq("val").to_numpy(),
        test=clips["split"].eq("test").to_numpy(),
        original=np.load(baseline_file, mmap_mode="r"),
        clean_classifier=clean_classifier,
        head=head,
        seed=seed,
        early_stopping=early_stopping,
    )


def _cpu_condition_task(source: dict[str, str]) -> dict[str, object]:
    targets = _CPU_CONDITION_STATE["targets"]
    train = _CPU_CONDITION_STATE["train"]
    validation = _CPU_CONDITION_STATE["validation"]
    test = _CPU_CONDITION_STATE["test"]
    original = _CPU_CONDITION_STATE["original"]
    compressed = np.load(source["embeddings_file"], mmap_mode="r")
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        real_prediction = _CPU_CONDITION_STATE["clean_classifier"].predict(
            compressed[test]
        )
        real_inference_seconds = time.perf_counter() - started
        matched = _build_classifier(
            str(_CPU_CONDITION_STATE["head"]),
            int(_CPU_CONDITION_STATE["seed"]),
            bool(_CPU_CONDITION_STATE["early_stopping"]),
        )
        started = time.perf_counter()
        matched.fit(compressed[train], targets[train])
        matched_training_seconds = time.perf_counter() - started
        training_epochs = _sklearn_training_epochs(
            matched, str(_CPU_CONDITION_STATE["head"])
        )
        loss_curves = _sklearn_loss_curves(
            matched, str(_CPU_CONDITION_STATE["head"])
        )
        started = time.perf_counter()
        matched_prediction = matched.predict(compressed[test])
        matched_inference_seconds = time.perf_counter() - started
        matched_validation_prediction = matched.predict(compressed[validation])
    return {
        **source,
        "preservation": _embedding_preservation(original[test], compressed[test]),
        "real_prediction": real_prediction,
        "matched_prediction": matched_prediction,
        "matched_validation_prediction": matched_validation_prediction,
        "real_inference_seconds": real_inference_seconds,
        "matched_training_seconds": matched_training_seconds,
        "matched_inference_seconds": matched_inference_seconds,
        "training_epochs": training_epochs,
        "loss_curves": loss_curves,
    }


def _embedding_preservation(
    reference: np.ndarray,
    reconstructed: np.ndarray,
) -> dict[str, float]:
    reference64 = np.asarray(reference, dtype=np.float64)
    reconstructed64 = np.asarray(reconstructed, dtype=np.float64)
    delta = reference64 - reconstructed64
    euclidean = np.linalg.norm(delta, axis=1)
    denominator = np.linalg.norm(reference64, axis=1) * np.linalg.norm(
        reconstructed64, axis=1
    )
    cosine = np.sum(reference64 * reconstructed64, axis=1) / np.maximum(
        denominator, 1e-12
    )
    correlations = []
    for original, compressed in zip(reference64, reconstructed64):
        if original.std() == 0 or compressed.std() == 0:
            correlations.append(np.nan)
        else:
            correlations.append(np.corrcoef(original, compressed)[0, 1])
    return {
        "embedding_euclidean_mean": float(np.mean(euclidean)),
        "embedding_euclidean_median": float(np.median(euclidean)),
        "embedding_cosine_mean": float(np.mean(cosine)),
        "embedding_correlation_mean": float(np.nanmean(correlations)),
    }


def _load_source_index(
    path: Path,
    datasets: Sequence[str] | None,
    foundation_model: str = DEFAULT_FOUNDATION_MODEL,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Indice de embeddings nao encontrado: {path}. "
            "Execute primeiro `python -m codeco.embeddings`."
        )
    table = pd.read_csv(path)
    missing = set(SOURCE_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(
            f"Indice de embeddings sem colunas obrigatorias: {', '.join(sorted(missing))}"
        )
    table = table.loc[table["foundation_model"] == foundation_model].copy()
    if datasets is not None:
        unknown = sorted(set(datasets) - set(table["dataset"].astype(str)))
        if unknown:
            raise ValueError(f"Datasets ausentes no indice: {', '.join(unknown)}")
        table = table.loc[table["dataset"].isin(datasets)].copy()
    if table.empty:
        raise ValueError(f"Nenhum embedding {foundation_model.upper()} selecionado.")
    incomplete = table.loc[table["completed"].astype(int) != table["clips"].astype(int)]
    if not incomplete.empty:
        conditions = ", ".join(
            f"{row.dataset}/{row.codec}/{row.setting}"
            for row in incomplete.itertuples()
        )
        raise RuntimeError(f"Embeddings incompletos: {conditions}")
    duplicate = table.duplicated(["dataset", "codec", "setting"], keep=False)
    if duplicate.any():
        raise ValueError("O indice contem condicoes duplicadas para o mesmo dataset.")
    paths = CodecoPath.discover()
    for column in ("clips_file", "embeddings_file"):
        table[column] = table[column].map(lambda value: str(paths.project_path(value)))
    return table.reset_index(drop=True)


def _load_clips(source: pd.DataFrame) -> pd.DataFrame:
    clip_files = tuple(dict.fromkeys(source["clips_file"].astype(str)))
    if len(clip_files) != 1:
        raise ValueError("As condicoes do dataset apontam para indices de clipes diferentes.")
    path = Path(clip_files[0])
    if path.suffix != ".parquet":
        raise ValueError(
            f"Indice de clipes legado encontrado: {path}. "
            "Execute novamente codeco.embeddings para gerar clips.parquet."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Indice de clipes nao encontrado: {path}")
    clips = pd.read_parquet(path)
    required = {"row_index", "clip_id", "labels", "split"}
    missing = required - set(clips.columns)
    if missing:
        raise ValueError(f"clips.parquet sem colunas: {', '.join(sorted(missing))}")
    clips = clips.sort_values("row_index").reset_index(drop=True)
    expected = np.arange(len(clips), dtype=np.int64)
    if not np.array_equal(clips["row_index"].to_numpy(), expected):
        raise ValueError("row_index de clips.parquet nao e continuo a partir de zero.")
    if len(clips) != int(source.iloc[0]["clips"]):
        raise ValueError("Quantidade de clipes diverge do indice de embeddings.")
    return clips


def _load_embeddings(path: str, rows: int) -> np.ndarray:
    embedding_path = Path(path)
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Embeddings nao encontrados: {embedding_path}")
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[0] != rows:
        raise ValueError(
            f"Formato de embeddings incompativel em {embedding_path}: {embeddings.shape}"
        )
    return embeddings


def _fingerprint(
    source: pd.DataFrame,
    dataset: str,
    head: str,
    seed: int,
    early_stopping: bool,
    foundation_model: str = DEFAULT_FOUNDATION_MODEL,
) -> tuple[str, dict]:
    paths = CodecoPath.discover()
    conditions = [
        {
            "codec": str(row.codec),
            "setting": str(row.setting),
            "clips_file": paths.project_relative(row.clips_file),
            "embeddings_file": paths.project_relative(row.embeddings_file),
        }
        for row in source.itertuples()
    ]
    metadata = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "dataset": dataset,
        "foundation_model": foundation_model,
        "head": head,
        "seed": seed,
        "conditions": conditions,
    }
    if not early_stopping:
        metadata["early_stopping"] = False
    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), metadata


def _result_key(result: dict[str, object]) -> tuple[str, str, str]:
    return str(result["codec"]), str(result["setting"]), str(result["protocol"])


def _checkpoint_result(
    connection: sqlite3.Connection,
    fingerprint: str,
    result: dict[str, object],
) -> None:
    now = _utc_now()
    with connection:
        connection.execute(
            """INSERT INTO birdnet_classification_results
               (fingerprint, codec, setting, protocol, result_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(fingerprint, codec, setting, protocol) DO UPDATE SET
                 result_json=excluded.result_json, updated_at=excluded.updated_at""",
            (
                fingerprint,
                result["codec"],
                result["setting"],
                result["protocol"],
                json.dumps(result, ensure_ascii=False, allow_nan=False),
                now,
            ),
        )
        connection.execute(
            "UPDATE birdnet_classification_runs SET updated_at=? WHERE fingerprint=?",
            (now, fingerprint),
        )


def _checkpoint_results(
    connection: sqlite3.Connection,
    fingerprint: str,
) -> dict[tuple[str, str, str], dict[str, object]]:
    rows = connection.execute(
        "SELECT result_json FROM birdnet_classification_results WHERE fingerprint=?",
        (fingerprint,),
    )
    results = [json.loads(row[0]) for row in rows]
    return {_result_key(result): result for result in results}


def _expected_keys(source: pd.DataFrame) -> list[tuple[str, str, str]]:
    keys = [("baseline", "orig", "clean_to_clean")]
    for row in source.itertuples():
        if row.codec == "baseline":
            continue
        keys.extend(
            [
                (str(row.codec), str(row.setting), "train_orig_test_comp"),
                (str(row.codec), str(row.setting), "train_comp_test_comp"),
            ]
        )
    return keys


def _write_available_results(
    results: dict[tuple[str, str, str], dict[str, object]],
    expected: Sequence[tuple[str, str, str]],
    output: Path,
) -> None:
    ordered = [results[key] for key in expected if key in results]
    table = pd.DataFrame.from_records(ordered, columns=RESULT_COLUMNS)
    _atomic_csv(table, output)


def _result_record(
    dataset: str,
    foundation_model: str,
    head: str,
    codec: str,
    setting: str,
    protocol: str,
    preservation: dict[str, float],
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "foundation_model": foundation_model,
        "head": head,
        "codec": codec,
        "setting": setting,
        "protocol": protocol,
        **preservation,
        **metrics,
    }


def _classify_dataset(
    connection: sqlite3.Connection,
    source: pd.DataFrame,
    dataset: str,
    head: str,
    seed: int,
    force: bool,
    early_stopping: bool,
    plot_training_curves: bool,
    output_directory: Path,
    cache_root: Path,
    graph_directory: Path,
    foundation_model: str = DEFAULT_FOUNDATION_MODEL,
) -> tuple[Path, int, int, float]:
    dataset_started = time.perf_counter()
    source = source.loc[source["dataset"] == dataset].reset_index(drop=True)
    baseline_rows = source.loc[
        (source["codec"] == "baseline") & (source["setting"] == "orig")
    ]
    if len(baseline_rows) != 1:
        raise ValueError(f"{dataset}: esperada exatamente uma baseline/orig.")
    fingerprint, metadata = _fingerprint(
        source, dataset, head, seed, early_stopping, foundation_model
    )
    output = output_directory / (
        f"classification_{dataset}_{foundation_model}_{head}.csv"
    )
    cache = (
        cache_root
        / f"{foundation_model}_classification"
        / fingerprint
        / dataset
        / head
    )
    expected = _expected_keys(source)

    now = _utc_now()
    with connection:
        if force:
            connection.execute(
                "DELETE FROM birdnet_classification_results WHERE fingerprint=?",
                (fingerprint,),
            )
            connection.execute(
                "DELETE FROM birdnet_classification_runs WHERE fingerprint=?",
                (fingerprint,),
            )
        connection.execute(
            """INSERT INTO birdnet_classification_runs
               (fingerprint, dataset, head, metadata_json, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'running', ?, ?)
               ON CONFLICT(fingerprint) DO UPDATE SET status='running', updated_at=excluded.updated_at""",
            (fingerprint, dataset, head, json.dumps(metadata, ensure_ascii=False), now, now),
        )

    clips = _load_clips(source)
    labels = [
        value.split("|") if isinstance(value, str) and value else []
        for value in clips["labels"]
    ]
    binarizer = MultiLabelBinarizer()
    y_all = binarizer.fit_transform(labels)
    if y_all.shape[1] == 0:
        raise ValueError(f"{dataset}: nenhum rotulo encontrado.")
    _atomic_json([str(value) for value in binarizer.classes_], cache / "classes.json")

    train = clips["split"].eq("train").to_numpy()
    validation = clips["split"].eq("val").to_numpy()
    test = clips["split"].eq("test").to_numpy()
    for split_name, mask in (("train", train), ("val", validation), ("test", test)):
        if not mask.any():
            raise ValueError(f"{dataset}: split {split_name} vazio.")
        positives = y_all[mask].sum(axis=0)
        negatives = int(mask.sum()) - positives
        invalid = [
            str(binarizer.classes_[index])
            for index in range(len(binarizer.classes_))
            if positives[index] < 2 or negatives[index] < 2
        ]
        if invalid:
            raise ValueError(
                f"{dataset}: split {split_name} nao permite metricas validas; "
                "exigidos ao menos 2 positivos e 2 negativos por classe. "
                f"Classes invalidas: {', '.join(invalid)}. "
                "Regenere uma amostra estratificada ou use o corpus completo."
            )

    results = _checkpoint_results(connection, fingerprint)
    resumed = sum(key in results for key in expected)
    if resumed == len(expected):
        if plot_training_curves:
            raise RuntimeError(
                "O grafico exige uma nova execucao para coletar as curvas por epoca. "
                "Use --plot-training-curves junto com --force."
            )
        _write_available_results(results, expected, output)
        with connection:
            connection.execute(
                "UPDATE birdnet_classification_runs SET status='completed', updated_at=? "
                "WHERE fingerprint=?",
                (_utc_now(), fingerprint),
            )
        LOGGER.info("[CACHE HIT] %s | %s: %d resultados.", dataset, head, resumed)
        return output, resumed, 0, time.perf_counter() - dataset_started

    baseline = baseline_rows.iloc[0]
    x_original = _load_embeddings(str(baseline["embeddings_file"]), len(clips))
    torch_module = None
    torch_device = None
    torch_train = torch_predict = None
    if head == "mlp":
        try:
            import torch

            if torch.cuda.is_available():
                from codeco.benchmark.birdnet_classification_benchmark import (
                    _torch_predict as benchmark_torch_predict,
                    _torch_train as benchmark_torch_train,
                )

                gpu = max(
                    range(torch.cuda.device_count()),
                    key=lambda index: torch.cuda.get_device_properties(index).total_memory,
                )
                torch_device = torch.device(f"cuda:{gpu}")
                torch.cuda.set_device(torch_device)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch_module = torch
                torch_train = benchmark_torch_train
                torch_predict = benchmark_torch_predict
                LOGGER.info(
                    "Backend: PyTorch/CUDA | GPU %d: %s.",
                    gpu,
                    torch.cuda.get_device_name(torch_device),
                )
        except (ImportError, RuntimeError):
            LOGGER.warning("PyTorch/CUDA indisponivel; usando fallback CPU.")

    if plot_training_curves and torch_module is None:
        raise RuntimeError(
            "--plot-training-curves exige o backend PyTorch/CUDA para registrar "
            "loss de treino e validacao em todas as epocas."
        )

    LOGGER.info("Treinando %s na baseline: %d clipes.", head.upper(), int(train.sum()))
    baseline_training_started = time.perf_counter()
    if torch_module is not None:
        classifier, baseline_training_seconds = torch_train(
            torch_module,
            x_original,
            y_all,
            train,
            torch_device,
            seed,
            early_stopping=early_stopping,
            monitor=validation if plot_training_curves else None,
        )
        baseline_training_epochs = _torch_training_epochs(classifier)
        baseline_loss_curves = _torch_loss_curves(classifier)
        baseline_validation_loss_curves = _torch_validation_loss_curves(classifier)
        predict = lambda values: torch_predict(
            torch_module, classifier, values, torch_device
        )
    else:
        classifier = _build_classifier(head, seed, early_stopping)
        classifier.fit(x_original[train], y_all[train])
        baseline_training_seconds = time.perf_counter() - baseline_training_started
        baseline_training_epochs = _sklearn_training_epochs(classifier, head)
        baseline_loss_curves = _sklearn_loss_curves(classifier, head)
        baseline_validation_loss_curves = []
        predict = classifier.predict
        physical_cores = max(1, psutil.cpu_count(logical=False) or 1)
        LOGGER.info(
            "Backend: CPU | pool de %d processos (nucleos fisicos).", physical_cores
        )
    validation_started = time.perf_counter()
    validation_prediction = predict(x_original[validation])
    validation_inference_seconds = time.perf_counter() - validation_started

    baseline_key = ("baseline", "orig", "clean_to_clean")
    processed = 0
    baseline_test_started = time.perf_counter()
    baseline_test_prediction = predict(x_original[test])
    baseline_test_inference_seconds = time.perf_counter() - baseline_test_started
    baseline_processed = baseline_key not in results
    if baseline_key not in results:
        result = _result_record(
            dataset,
            foundation_model,
            head,
            "baseline",
            "orig",
            "clean_to_clean",
            {
                "embedding_euclidean_mean": 0.0,
                "embedding_euclidean_median": 0.0,
                "embedding_cosine_mean": 1.0,
                "embedding_correlation_mean": 1.0,
            },
            _classification_metrics(y_all[test], baseline_test_prediction),
        )
        _checkpoint_result(connection, fingerprint, result)
        results[baseline_key] = result
        _atomic_npy(
            multilabel_confusion_matrix(y_all[test], baseline_test_prediction),
            cache / "cm_baseline_orig.npy",
        )
        processed += 1
        _write_available_results(results, expected, output)

    _section(
        "Resultado 1/%d: R0 | PCM s16le | BASELINE PRINCIPAL" % len(source)
    )
    LOGGER.info(
        "Papel experimental: R0 e a referencia PCM/original usada como baseline principal."
    )
    LOGGER.info("Status: %s", "processado" if baseline_processed else "retomado")
    LOGGER.info("Treino: %d clipes | %.3f s", int(train.sum()), baseline_training_seconds)
    training_epoch_summaries: dict[
        int, tuple[str, dict[str, float | int], dict[str, float] | None]
    ] = {}
    training_curve_records: dict[
        int, tuple[str, list[list[float]], list[list[float]]]
    ] = {}
    if baseline_training_epochs:
        baseline_epoch_summary = _epoch_summary(baseline_training_epochs)
        baseline_loss_summary = _loss_summary(baseline_loss_curves)
        training_epoch_summaries[1] = (
            "R0 | PCM s16le | baseline principal",
            baseline_epoch_summary,
            baseline_loss_summary,
        )
        training_curve_records[1] = (
            "R0 | PCM s16le",
            baseline_loss_curves,
            baseline_validation_loss_curves,
        )
        LOGGER.info(
            "Epocas por classificador: min=%d | media=%.1f | mediana=%.1f | max=%d | no limite=%d/%d",
            baseline_epoch_summary["minimum"],
            baseline_epoch_summary["mean"],
            baseline_epoch_summary["median"],
            baseline_epoch_summary["maximum"],
            baseline_epoch_summary["at_limit"],
            baseline_epoch_summary["classifiers"],
        )
        if baseline_loss_summary is not None:
            LOGGER.info(
                "Loss: inicial=%.6f | final=%.6f | melhor=%.6f",
                baseline_loss_summary["initial"],
                baseline_loss_summary["final"],
                baseline_loss_summary["best"],
            )
            LOGGER.info(
                "Loss = binary cross-entropy media dos classificadores one-vs-rest, "
                "com regularizacao L2."
            )
    LOGGER.info(
        "Inferencia: validacao=%.3f s | teste=%.3f s",
        validation_inference_seconds,
        baseline_test_inference_seconds,
    )
    _log_performance_report(
        "Referencia R0 - desempenho sem compressao",
        "Estabelecer o desempenho de referencia da classificacao.",
        "embeddings da baseline R0 (PCM s16le), split de treino",
        (
            (
                "VALIDACAO",
                "R0 PCM (sem compressao)",
                y_all[validation],
                validation_prediction,
            ),
            (
                "TESTE",
                "R0 PCM (sem compressao)",
                y_all[test],
                baseline_test_prediction,
            ),
        )
    )

    compressed = source.loc[source["codec"] != "baseline"]

    condition_numbers = {
        (str(row.codec), str(row.setting)): number
        for number, row in enumerate(compressed.itertuples(index=False), 2)
    }

    def save_condition(
        row,
        preservation,
        real_prediction,
        matched_validation_prediction,
        matched_prediction,
        timings,
    ) -> None:
        nonlocal processed
        real_key = (str(row.codec), str(row.setting), "train_orig_test_comp")
        matched_key = (str(row.codec), str(row.setting), "train_comp_test_comp")
        real_processed = real_key not in results
        matched_processed = matched_key not in results
        if real_key not in results:
            result = _result_record(
                dataset,
                foundation_model,
                head,
                str(row.codec),
                str(row.setting),
                "train_orig_test_comp",
                preservation,
                _classification_metrics(y_all[test], real_prediction),
            )
            _checkpoint_result(connection, fingerprint, result)
            results[real_key] = result
            safe = hashlib.sha256(
                f"{row.codec}\0{row.setting}".encode("utf-8")
            ).hexdigest()[:12]
            _atomic_npy(
                multilabel_confusion_matrix(y_all[test], real_prediction),
                cache / f"cm_{row.codec}_{safe}_real.npy",
            )
            processed += 1

        if matched_key not in results:
            result = _result_record(
                dataset,
                foundation_model,
                head,
                str(row.codec),
                str(row.setting),
                "train_comp_test_comp",
                preservation,
                _classification_metrics(y_all[test], matched_prediction),
            )
            _checkpoint_result(connection, fingerprint, result)
            results[matched_key] = result
            processed += 1
        _write_available_results(results, expected, output)
        number = condition_numbers[(str(row.codec), str(row.setting))]
        condition_id, family, condition_name, role = _experimental_condition(
            str(row.codec)
        )
        reconstructed_source = (
            f"{condition_id} {condition_name} reconstruido ({row.setting})"
        )
        _section(
            f"Resultado {number}/{len(source)}: {condition_id} | "
            f"{condition_name.upper()} | {row.setting}"
        )
        LOGGER.info(
            "Papel experimental: %s | familia=%s | %s.",
            condition_id,
            family,
            role,
        )
        if str(row.codec).lower() == "flac":
            LOGGER.info(
                "Nota: R0 (PCM) e R1 (FLAC) sao referencias distintas. Como FLAC "
                "e sem perdas, os dois protocolos podem produzir metricas identicas."
            )
        if "gpu" in timings:
            LOGGER.info(
                "Processamento: GPU %d | %s",
                timings["gpu"],
                timings["gpu_name"],
            )
        LOGGER.info("Embeddings: %s", row.embeddings_file)
        LOGGER.info(
            "Treino da MLP adaptada a esta configuracao: %d clipes | %.3f s | status=%s",
            int(train.sum()),
            timings["matched_training_seconds"],
            "processado" if matched_processed else "retomado",
        )
        training_epochs = timings["training_epochs"]
        if training_epochs:
            epoch_summary = _epoch_summary(training_epochs)
            loss_summary = _loss_summary(timings["loss_curves"])
            training_epoch_summaries[number] = (
                f"{condition_id} | {condition_name} | {row.setting}",
                epoch_summary,
                loss_summary,
            )
            training_curve_records[number] = (
                f"{condition_id} | {condition_name} | {row.setting}",
                timings["loss_curves"],
                timings.get("validation_loss_curves", []),
            )
            LOGGER.info(
                "Epocas por classificador: min=%d | media=%.1f | mediana=%.1f | max=%d | no limite=%d/%d",
                epoch_summary["minimum"],
                epoch_summary["mean"],
                epoch_summary["median"],
                epoch_summary["maximum"],
                epoch_summary["at_limit"],
                epoch_summary["classifiers"],
            )
            if loss_summary is not None:
                LOGGER.info(
                    "Loss: inicial=%.6f | final=%.6f | melhor=%.6f",
                    loss_summary["initial"],
                    loss_summary["final"],
                    loss_summary["best"],
                )
                LOGGER.info(
                    "Loss = binary cross-entropy media dos classificadores "
                    "one-vs-rest, com regularizacao L2."
                )
        LOGGER.info(
            "Status: experimento A=%s | experimento B=%s",
            "processado" if real_processed else "retomado",
            "processado" if matched_processed else "retomado",
        )
        LOGGER.info(
            "Inferencia no teste: MLP baseline=%.3f s | MLP adaptada=%.3f s",
            timings["real_inference_seconds"],
            timings["matched_inference_seconds"],
        )
        _log_performance_report(
            "Experimento A - robustez sem retreinamento",
            "Medir se a MLP treinada sem compressao continua classificando "
            "corretamente apos a reconstrucao pelo codec. A validacao R0 e "
            "uma referencia fixa; o teste reconstruido muda em cada configuracao.",
            "embeddings da baseline R0 (PCM s16le), split de treino",
            (
                (
                    "VALIDACAO",
                    "R0 PCM (sem compressao)",
                    y_all[validation],
                    validation_prediction,
                ),
                ("TESTE", reconstructed_source, y_all[test], real_prediction),
            ),
        )
        _log_performance_report(
            "Experimento B - adaptacao ao codec",
            "Medir o desempenho quando uma nova MLP e treinada especificamente "
            "com os embeddings reconstruidos por esta configuracao.",
            f"embeddings de {reconstructed_source}, split de treino",
            (
                (
                    "VALIDACAO",
                    reconstructed_source,
                    y_all[validation],
                    matched_validation_prediction,
                ),
                ("TESTE", reconstructed_source, y_all[test], matched_prediction),
            )
        )
        LOGGER.info(
            "Preservacao: euclidiana_media=%.6f | cosseno_medio=%.6f | correlacao_media=%.6f",
            preservation["embedding_euclidean_mean"],
            preservation["embedding_cosine_mean"],
            preservation["embedding_correlation_mean"],
        )

    pending_rows = [
        row
        for row in compressed.itertuples(index=False)
        if (str(row.codec), str(row.setting), "train_orig_test_comp") not in results
        or (str(row.codec), str(row.setting), "train_comp_test_comp") not in results
    ]
    progress = tqdm(
        total=len(pending_rows),
        desc=f"{dataset} | {head}",
        unit="cond",
        dynamic_ncols=True,
    )
    if torch_module is not None and len(pending_rows) > 1 and torch_module.cuda.device_count() > 1:
        gpu_count = min(torch_module.cuda.device_count(), len(pending_rows))
        LOGGER.info(
            "Execucao multi-GPU: %d workers persistentes; uma configuracao de codec por GPU.",
            gpu_count,
        )
        context = __import__("multiprocessing").get_context("spawn")
        task_queues = [context.Queue() for _ in range(gpu_count)]
        completed_queue = context.Queue()
        baseline_for_workers = _torch_model_for_transfer(classifier)
        rows_by_key = {
            (str(row.codec), str(row.setting)): row for row in pending_rows
        }
        workers = [
            context.Process(
                target=_gpu_configuration_worker,
                args=(
                    gpu,
                    task_queues[gpu],
                    completed_queue,
                    baseline_for_workers,
                    y_all,
                    train,
                    validation,
                    test,
                    seed,
                    early_stopping,
                    plot_training_curves,
                ),
                name=f"{foundation_model}-mlp-gpu-{gpu}",
            )
            for gpu in range(gpu_count)
        ]
        for worker in workers:
            worker.start()

        def configuration_task(row) -> dict[str, str]:
            return {
                "codec": str(row.codec),
                "setting": str(row.setting),
                "embeddings_file": str(row.embeddings_file),
            }

        pending_iterator = iter(pending_rows)
        for gpu in range(gpu_count):
            task_queues[gpu].put(configuration_task(next(pending_iterator)))

        remaining = len(pending_rows)
        try:
            while remaining:
                try:
                    data = completed_queue.get(timeout=5)
                except queue.Empty:
                    failed = [
                        worker
                        for worker in workers
                        if worker.exitcode not in (None, 0)
                    ]
                    if failed:
                        statuses = ", ".join(
                            f"{worker.name}={worker.exitcode}" for worker in failed
                        )
                        raise RuntimeError(f"Workers GPU encerrados: {statuses}")
                    continue
                if "error" in data:
                    raise RuntimeError(
                        f"Falha no worker GPU {data.get('gpu')}:\n{data['error']}"
                    )
                key = (str(data["codec"]), str(data["setting"]))
                row = rows_by_key[key]
                x_compressed = _load_embeddings(
                    str(data["embeddings_file"]), len(clips)
                )
                preservation = _embedding_preservation(
                    x_original[test], x_compressed[test]
                )
                save_condition(
                    row,
                    preservation,
                    data["real_prediction"],
                    data["matched_validation_prediction"],
                    data["matched_prediction"],
                    data,
                )
                progress.set_postfix_str(
                    f"GPU {data['gpu']} | {str(row.codec).upper()} | {row.setting}"
                )
                progress.update()
                remaining -= 1
                try:
                    next_row = next(pending_iterator)
                except StopIteration:
                    task_queues[int(data["gpu"])].put(None)
                else:
                    task_queues[int(data["gpu"])].put(
                        configuration_task(next_row)
                    )
        finally:
            for worker in workers:
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            for task_queue in task_queues:
                task_queue.close()
            completed_queue.close()
    elif torch_module is not None:
        for row in pending_rows:
            progress.set_postfix_str(f"{str(row.codec).upper()} | {row.setting}")
            x_compressed = _load_embeddings(str(row.embeddings_file), len(clips))
            preservation = _embedding_preservation(x_original[test], x_compressed[test])
            started = time.perf_counter()
            real_prediction = predict(x_compressed[test])
            real_inference_seconds = time.perf_counter() - started
            matched, matched_training_seconds = torch_train(
                torch_module,
                x_compressed,
                y_all,
                train,
                torch_device,
                seed,
                early_stopping=early_stopping,
                monitor=validation if plot_training_curves else None,
            )
            training_epochs = _torch_training_epochs(matched)
            loss_curves = _torch_loss_curves(matched)
            validation_loss_curves = _torch_validation_loss_curves(matched)
            started = time.perf_counter()
            matched_prediction = torch_predict(
                torch_module, matched, x_compressed[test], torch_device
            )
            matched_inference_seconds = time.perf_counter() - started
            matched_validation_prediction = torch_predict(
                torch_module, matched, x_compressed[validation], torch_device
            )
            save_condition(
                row,
                preservation,
                real_prediction,
                matched_validation_prediction,
                matched_prediction,
                {
                    "real_inference_seconds": real_inference_seconds,
                    "matched_training_seconds": matched_training_seconds,
                    "matched_inference_seconds": matched_inference_seconds,
                    "training_epochs": training_epochs,
                    "loss_curves": loss_curves,
                    "validation_loss_curves": validation_loss_curves,
                },
            )
            progress.update()
    elif pending_rows:
        context = __import__("multiprocessing").get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=physical_cores,
            mp_context=context,
            initializer=_cpu_condition_initializer,
            initargs=(
                str(source.iloc[0]["clips_file"]),
                str(baseline["embeddings_file"]),
                classifier,
                head,
                seed,
                early_stopping,
            ),
        ) as pool:
            futures = {
                pool.submit(
                    _cpu_condition_task,
                    {
                        "codec": str(row.codec),
                        "setting": str(row.setting),
                        "embeddings_file": str(row.embeddings_file),
                    },
                ): row
                for row in pending_rows
            }
            for future in as_completed(futures):
                row = futures[future]
                data = future.result()
                save_condition(
                    row,
                    data["preservation"],
                    data["real_prediction"],
                    data["matched_validation_prediction"],
                    data["matched_prediction"],
                    data,
                )
                progress.set_postfix_str(f"{str(row.codec).upper()} | {row.setting}")
                progress.update()
    progress.close()
    _section("Resumo dos treinamentos e avaliacoes")
    LOGGER.info("Treinamentos da MLP baseline R0: 1")
    LOGGER.info(
        "Treinamentos de MLP adaptada a configuracao: %d", len(pending_rows)
    )
    LOGGER.info("Total de treinamentos nesta execucao: %d", 1 + len(pending_rows))
    LOGGER.info("Teste de referencia R0 sem compressao: 1")
    LOGGER.info(
        "Testes do experimento A (robustez sem retreinamento): %d",
        len(pending_rows),
    )
    LOGGER.info(
        "Testes do experimento B (adaptacao ao codec): %d", len(pending_rows)
    )
    LOGGER.info(
        "Validacao R0 fixa do experimento A: 1 resultado, reutilizado na exibicao."
    )
    LOGGER.info(
        "Validacoes especificas do experimento B: %d", len(pending_rows)
    )
    if training_epoch_summaries:
        _section("Epocas efetivas por condicao")
        epoch_rows = []
        for number in sorted(training_epoch_summaries):
            condition, summary, loss = training_epoch_summaries[number]
            epoch_rows.append(
                {
                    "Resultado": f"{number}/{len(source)}",
                    "Condicao": condition,
                    "MLPs": summary["classifiers"],
                    "Min": summary["minimum"],
                    "Media": f'{summary["mean"]:.1f}',
                    "Mediana": f'{summary["median"]:.1f}',
                    "Max": summary["maximum"],
                    "Limite 300": f'{summary["at_limit"]}/{summary["classifiers"]}',
                    "Loss inicial": f'{loss["initial"]:.6f}' if loss else "-",
                    "Loss final": f'{loss["final"]:.6f}' if loss else "-",
                    "Melhor loss": f'{loss["best"]:.6f}' if loss else "-",
                }
            )
        LOGGER.info("%s", pd.DataFrame(epoch_rows).to_string(index=False))
    if plot_training_curves:
        graph = (
            graph_directory
            / f"classification_{dataset}_{foundation_model}_{head}_overfitting.png"
        )
        _plot_training_curves(training_curve_records, graph)

    with connection:
        connection.execute(
            "UPDATE birdnet_classification_runs SET status='completed', updated_at=? "
            "WHERE fingerprint=?",
            (_utc_now(), fingerprint),
        )
    _write_available_results(results, expected, output)
    return output, resumed, processed, time.perf_counter() - dataset_started


def run(
    datasets: Sequence[str] | None = None,
    head: str = "mlp",
    seed: int = DEFAULT_SEED,
    embeddings_index: Path | None = None,
    checkpoint_path: Path | None = None,
    force: bool = False,
    early_stopping: bool = True,
    plot_training_curves: bool = False,
    foundation_model: str = DEFAULT_FOUNDATION_MODEL,
    artifact_directory: Path | None = None,
) -> list[Path]:
    paths = CodecoPath.discover()
    if artifact_directory is not None:
        artifact_directory.mkdir(parents=True, exist_ok=True)
    log_file = _configure_logging(
        paths,
        artifact_directory / "execution.log"
        if artifact_directory is not None
        else None,
    )
    _header(foundation_model)
    source_path = (
        embeddings_index
        or paths.tables / f"{foundation_model}_embeddings.csv"
    )
    checkpoint = checkpoint_path or (
        artifact_directory / "checkpoints.sqlite3"
        if artifact_directory is not None
        else paths.cache / "checkpoints.sqlite3"
    )
    output_directory = artifact_directory or paths.tables
    cache_root = (
        artifact_directory / "cache"
        if artifact_directory is not None
        else paths.cache
    )
    graph_directory = (
        artifact_directory / "graphs"
        if artifact_directory is not None
        else paths.graphs
    )
    source = _load_source_index(source_path, datasets, foundation_model)
    selected_datasets = tuple(dict.fromkeys(source["dataset"].astype(str)))

    _section("Configuracao")
    LOGGER.info("Log: %s", log_file)
    LOGGER.info("Indice de embeddings: %s", source_path)
    LOGGER.info("Checkpoint: %s", checkpoint)
    LOGGER.info("Datasets: %s", ", ".join(selected_datasets))
    LOGGER.info("Foundation model: %s", foundation_model.upper())
    LOGGER.info("Head: %s", head.upper())
    LOGGER.info("Force: %s", force)
    LOGGER.info("Early stopping: %s", early_stopping)
    LOGGER.info("Grafico de overfitting: %s", plot_training_curves)
    _log_experimental_legend(source)

    connection = _connect_checkpoint(checkpoint)
    outputs: list[Path] = []
    summaries = []
    started = time.perf_counter()
    try:
        for number, dataset in enumerate(selected_datasets, 1):
            _section(f"Dataset {number}/{len(selected_datasets)}: {dataset}")
            output, resumed, processed, seconds = _classify_dataset(
                connection,
                source,
                dataset,
                head,
                seed,
                force,
                early_stopping,
                plot_training_curves,
                output_directory,
                cache_root,
                graph_directory,
                foundation_model,
            )
            outputs.append(output)
            summaries.append((dataset, resumed, processed, seconds, output))
    finally:
        connection.close()

    _section("Execucao concluida")
    for dataset, resumed, processed, seconds, output in summaries:
        LOGGER.info(
            "%s | retomados=%d | processados=%d | tempo=%.2f s | %s",
            dataset,
            resumed,
            processed,
            seconds,
            output,
        )
    LOGGER.info("Tempo total: %.2f s", time.perf_counter() - started)
    return outputs


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classifica embeddings existentes sem reprocessar audio."
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--head", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--embeddings-index", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Treina o MLP pelas 300 epocas completas, sem parada antecipada.",
    )
    parser.add_argument(
        "--plot-training-curves",
        action="store_true",
        help="Salva curvas de loss de treino e validacao em results/graphs.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    run(
        datasets=arguments.datasets,
        head=arguments.head,
        seed=arguments.seed,
        embeddings_index=arguments.embeddings_index,
        checkpoint_path=arguments.checkpoint,
        force=arguments.force,
        early_stopping=not arguments.no_early_stopping,
        plot_training_curves=arguments.plot_training_curves,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.error("Classificacao interrompida pelo usuario.")
        raise SystemExit(130) from None
    except Exception:
        LOGGER.exception("Falha na classificacao dos embeddings.")
        raise SystemExit(1) from None
