"""Benchmark BirdNET classification with parallelism between codec conditions."""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.utils import shuffle as sklearn_shuffle
from threadpoolctl import threadpool_limits

from codeco.codeco_path import CodecoPath
from codeco.prepare_table import TqdmLoggingHandler


SEED = 42
INPUTS = 1024
HIDDEN = 256
ALPHA = 1e-3
MAX_ITER = 300
LOGGER = logging.getLogger("codeco.birdnet_classification_benchmark")
_POOL_STATE: dict[str, object] = {}


def _logging(paths: CodecoPath) -> Path:
    log = paths.logs / "birdnet_classification_benchmark.log"
    if LOGGER.handlers:
        return log
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    terminal = TqdmLoggingHandler(sys.stdout)
    terminal.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(terminal)
    paths.logs.mkdir(parents=True, exist_ok=True)
    file = logging.FileHandler(log, encoding="utf-8")
    file.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(file)
    return log


def _header() -> None:
    line = "=" * 72
    command = subprocess.list2cmdline(
        [sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    )
    LOGGER.info(
        "\n%s\nBENCHMARK BIRDNET: PARALELISMO POR CONDICAO\n[CMD]   %s\n%s",
        line,
        command,
        line,
    )


def _section(title: str) -> None:
    LOGGER.info("\n%s\n%s", title, "-" * 72)


def _write_csv(table: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        table.to_csv(temporary, index=False)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sources(index: Path, dataset: str, limit: int | None):
    table = pd.read_csv(index)
    paths = CodecoPath.discover()
    for column in ("clips_file", "embeddings_file"):
        table[column] = table[column].map(lambda value: str(paths.project_path(value)))
    table = table.loc[
        (table["dataset"] == dataset) & (table["foundation_model"] == "birdnet")
    ]
    baseline = table.loc[(table["codec"] == "baseline") & (table["setting"] == "orig")]
    if len(baseline) != 1:
        raise ValueError(f"Esperada uma baseline para {dataset}; encontradas {len(baseline)}.")
    compressed = table.loc[table["codec"] != "baseline"]
    if limit is not None:
        compressed = compressed.iloc[:limit]
    if compressed.empty:
        raise ValueError("O benchmark requer ao menos uma condicao comprimida.")
    if (table["clips"].astype(int) != table["completed"].astype(int)).any():
        raise RuntimeError("Existem embeddings incompletos no indice.")
    clip_files = tuple(dict.fromkeys(table["clips_file"].astype(str)))
    if len(clip_files) != 1 or Path(clip_files[0]).suffix != ".parquet":
        raise ValueError("As condicoes devem compartilhar um unico clips.parquet.")
    base = {
        "codec": "baseline",
        "setting": "orig",
        "file": str(baseline.iloc[0]["embeddings_file"]),
    }
    conditions = [
        {"codec": str(row.codec), "setting": str(row.setting), "file": str(row.embeddings_file)}
        for row in compressed.itertuples()
    ]
    for source in [base, *conditions]:
        if not Path(source["file"]).is_file():
            raise FileNotFoundError(source["file"])
    return Path(clip_files[0]), base, conditions


def _problem(clips_file: Path):
    clips = pd.read_parquet(clips_file).sort_values("row_index").reset_index(drop=True)
    labels = [value.split("|") if isinstance(value, str) and value else [] for value in clips.labels]
    targets = MultiLabelBinarizer().fit_transform(labels).astype(np.int64, copy=False)
    train = clips.split.eq("train").to_numpy()
    test = clips.split.eq("test").to_numpy()
    if not train.any() or not test.any() or not targets.shape[1]:
        raise ValueError("Treino, teste ou classes ausentes.")
    for split_name, mask in (("train", train), ("test", test)):
        positives = targets[mask].sum(axis=0)
        negatives = int(mask.sum()) - positives
        if np.any(positives < 2) or np.any(negatives < 2):
            raise ValueError(
                f"Split {split_name} invalido para benchmark: exigidos ao menos "
                "2 positivos e 2 negativos por classe. Regenere a amostra estratificada."
            )
    return targets, train, test, len(clips)


def _embeddings(path: str, rows: int) -> np.ndarray:
    values = np.load(path, mmap_mode="r")
    if values.shape != (rows, INPUTS):
        raise ValueError(f"Embedding incompativel: {path}: {values.shape}")
    return values


def _classifier(seed: int):
    return make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            MLPClassifier(
                hidden_layer_sizes=(HIDDEN,),
                activation="relu",
                alpha=ALPHA,
                max_iter=MAX_ITER,
                early_stopping=True,
                random_state=seed,
            )
        ),
    )


def _key(codec: str, setting: str, protocol: str) -> str:
    return "\0".join((codec, setting, protocol))


def _metrics(targets: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    balanced = [
        balanced_accuracy_score(targets[:, index], predicted[:, index])
        for index in range(targets.shape[1])
    ]
    return {
        "f1_macro": float(f1_score(targets, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(np.mean(balanced)),
    }


def _pool_init(clips_file: str, seed: int, clean_classifier) -> None:
    targets, train, test, rows = _problem(Path(clips_file))
    _POOL_STATE.clear()
    _POOL_STATE.update(
        targets=targets,
        train=train,
        test=test,
        rows=rows,
        seed=seed,
        clean_classifier=clean_classifier,
    )


def _condition_task(source: dict[str, str]):
    targets = _POOL_STATE["targets"]
    train = _POOL_STATE["train"]
    test = _POOL_STATE["test"]
    values = _embeddings(source["file"], int(_POOL_STATE["rows"]))
    model = _classifier(int(_POOL_STATE["seed"]))
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        real_prediction = _POOL_STATE["clean_classifier"].predict(values[test]).astype(
            np.int64, copy=False
        )
        model.fit(values[train], targets[train])
        matched_prediction = model.predict(values[test]).astype(np.int64, copy=False)
    return (
        _key(source["codec"], source["setting"], "train_orig_test_comp"),
        real_prediction,
        _key(source["codec"], source["setting"], "train_comp_test_comp"),
        matched_prediction,
        time.perf_counter() - started,
    )


def _sklearn(
    strategy: str,
    clips_file: Path,
    baseline: dict[str, str],
    conditions: list[dict[str, str]],
    workers: int,
    seed: int,
):
    targets, train, test, rows = _problem(clips_file)
    original = _embeddings(baseline["file"], rows)
    clean = _classifier(seed)
    started = time.perf_counter()
    clean.fit(original[train], targets[train])
    training = time.perf_counter() - started
    started = time.perf_counter()
    predictions = {_key("baseline", "orig", "clean_to_clean"): clean.predict(original[test])}
    inference = time.perf_counter() - started

    if strategy == "current":
        for source in conditions:
            compressed = _embeddings(source["file"], rows)
            started = time.perf_counter()
            predictions[_key(source["codec"], source["setting"], "train_orig_test_comp")] = (
                clean.predict(compressed[test]).astype(np.int64, copy=False)
            )
            inference += time.perf_counter() - started
            matched = _classifier(seed)
            started = time.perf_counter()
            matched.fit(compressed[train], targets[train])
            training += time.perf_counter() - started
            started = time.perf_counter()
            predictions[_key(source["codec"], source["setting"], "train_comp_test_comp")] = (
                matched.predict(compressed[test]).astype(np.int64, copy=False)
            )
            inference += time.perf_counter() - started
        return predictions, training, inference, 1

    started = time.perf_counter()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_pool_init,
        initargs=(str(clips_file), seed, clean),
    ) as pool:
        futures = [pool.submit(_condition_task, source) for source in conditions]
        for future in as_completed(futures):
            real_key, real_prediction, matched_key, matched_prediction, _ = future.result()
            predictions[real_key] = real_prediction
            predictions[matched_key] = matched_prediction
    training += time.perf_counter() - started
    return predictions, training, inference, workers


def _gpu_forward(torch, x, parameters):
    w1, b1, w2, b2 = parameters
    hidden = torch.relu(x @ w1 + b1)
    return torch.sigmoid(hidden @ w2 + b2), hidden


def _gpu_binary_mlp(
    torch,
    x_scaled,
    labels,
    device,
    seed,
    early_stopping=True,
    monitor_x_scaled=None,
    monitor_labels=None,
):
    """Port the installed sklearn MLPClassifier binary training loop to CUDA."""
    if np.all(labels == labels[0]):
        return {
            "constant": int(labels[0]),
            "epochs": 0,
            "stop_reason": "constant",
            "loss_curve": [],
            "validation_loss_curve": [],
        }

    dtype = x_scaled.dtype
    input_dimension = int(x_scaled.shape[1])
    random_state = np.random.RandomState(seed)
    bound1 = np.sqrt(6.0 / (input_dimension + HIDDEN))
    bound2 = np.sqrt(6.0 / (HIDDEN + 1))
    initial = [
        random_state.uniform(-bound1, bound1, (input_dimension, HIDDEN)).astype(dtype),
        random_state.uniform(-bound1, bound1, HIDDEN).astype(dtype),
        random_state.uniform(-bound2, bound2, (HIDDEN, 1)).astype(dtype),
        random_state.uniform(-bound2, bound2, 1).astype(dtype),
    ]
    indices = np.arange(len(x_scaled), dtype=int)
    if early_stopping:
        fit_indices, validation_indices = train_test_split(
            indices,
            random_state=random_state,
            test_size=0.1,
            stratify=labels,
        )
    else:
        fit_indices = indices
        validation_indices = None
    x = torch.as_tensor(x_scaled, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels.reshape(-1, 1), dtype=torch.float32, device=device)
    monitor_x = (
        torch.as_tensor(monitor_x_scaled, dtype=torch.float32, device=device)
        if monitor_x_scaled is not None
        else None
    )
    monitor_y = (
        torch.as_tensor(
            np.asarray(monitor_labels).reshape(-1, 1),
            dtype=torch.float32,
            device=device,
        )
        if monitor_labels is not None
        else None
    )
    parameters = [torch.as_tensor(value, device=device).clone() for value in initial]
    moments = [torch.zeros_like(value) for value in parameters]
    velocities = [torch.zeros_like(value) for value in parameters]
    best = [value.clone() for value in parameters]
    best_score = -np.inf
    no_improvement = 0
    adam_step = 0
    sample_indices = np.arange(len(fit_indices), dtype=int)
    batch_size = min(200, len(fit_indices))

    epochs = 0
    stopped_early = False
    loss_curve = []
    validation_loss_curve = []
    with torch.no_grad():
        for epoch in range(MAX_ITER):
            epochs = epoch + 1
            epoch_loss = torch.zeros((), dtype=torch.float32, device=device)
            sample_indices = sklearn_shuffle(
                sample_indices, random_state=random_state
            )
            for offset in range(0, len(fit_indices), batch_size):
                local = sample_indices[offset : offset + batch_size]
                batch = torch.as_tensor(fit_indices[local], device=device)
                xb = x[batch]
                yb = y[batch]
                output, hidden = _gpu_forward(torch, xb, parameters)
                probabilities = output.clamp(1e-7, 1.0 - 1e-7)
                epoch_loss.add_(
                    -(
                        yb * torch.log(probabilities)
                        + (1.0 - yb) * torch.log1p(-probabilities)
                    ).sum()
                )
                delta2 = output - yb
                count = len(local)
                grad_w2 = (hidden.T @ delta2 + ALPHA * parameters[2]) / count
                grad_b2 = delta2.sum(0) / count
                delta1 = (delta2 @ parameters[2].T) * (hidden > 0)
                grad_w1 = (xb.T @ delta1 + ALPHA * parameters[0]) / count
                grad_b1 = delta1.sum(0) / count
                gradients = (grad_w1, grad_b1, grad_w2, grad_b2)

                adam_step += 1
                learning_rate = (
                    0.001
                    * np.sqrt(1.0 - 0.999**adam_step)
                    / (1.0 - 0.9**adam_step)
                )
                for index, (parameter, gradient) in enumerate(
                    zip(parameters, gradients)
                ):
                    moments[index].mul_(0.9).add_(gradient, alpha=0.1)
                    velocities[index].mul_(0.999).addcmul_(
                        gradient, gradient, value=0.001
                    )
                    parameter.addcdiv_(
                        moments[index],
                        torch.sqrt(velocities[index]) + 1e-8,
                        value=-float(learning_rate),
                    )

            regularization = 0.5 * ALPHA * (
                parameters[0].square().sum() + parameters[2].square().sum()
            )
            loss_curve.append(
                float(((epoch_loss + regularization) / len(fit_indices)).cpu())
            )
            if monitor_x is not None and monitor_y is not None:
                monitor_probability, _ = _gpu_forward(
                    torch, monitor_x, parameters
                )
                monitor_probability = monitor_probability.clamp(1e-7, 1.0 - 1e-7)
                monitor_loss = -(
                    monitor_y * torch.log(monitor_probability)
                    + (1.0 - monitor_y) * torch.log1p(-monitor_probability)
                ).mean()
                validation_loss_curve.append(float(monitor_loss.cpu()))

            if early_stopping:
                validation = torch.as_tensor(validation_indices, device=device)
                score = float(
                    ((_gpu_forward(torch, x[validation], parameters)[0] > 0.5) == y[validation])
                    .float()
                    .mean()
                    .cpu()
                )
                if score < best_score + 1e-4:
                    no_improvement += 1
                else:
                    no_improvement = 0
                if score > best_score:
                    best_score = score
                    best = [value.clone() for value in parameters]
                if no_improvement > 10:
                    stopped_early = True
                    break
    selected_parameters = best if early_stopping else parameters
    return {
        "constant": None,
        "parameters": selected_parameters,
        "epochs": epochs,
        "stop_reason": "early_stopping" if stopped_early else "max_iter",
        "loss_curve": loss_curve,
        "validation_loss_curve": validation_loss_curve,
    }


def _torch_train(
    torch,
    values,
    targets,
    train,
    device,
    seed,
    early_stopping=True,
    monitor=None,
):
    started = time.perf_counter()
    scaler = StandardScaler().fit(values[train])
    scaled = scaler.transform(values[train]).astype(np.float32, copy=False)
    monitor_scaled = (
        scaler.transform(values[monitor]).astype(np.float32, copy=False)
        if monitor is not None
        else None
    )
    models = [
        _gpu_binary_mlp(
            torch,
            scaled,
            targets[train, index],
            device,
            seed,
            early_stopping=early_stopping,
            monitor_x_scaled=monitor_scaled,
            monitor_labels=(targets[monitor, index] if monitor is not None else None),
        )
        for index in range(targets.shape[1])
    ]
    torch.cuda.synchronize(device)
    return {"scaler": scaler, "models": models}, time.perf_counter() - started


def _torch_predict(torch, model, values, device):
    scaled = model["scaler"].transform(values).astype(np.float32, copy=False)
    x = torch.as_tensor(scaled, dtype=torch.float32, device=device)
    columns = []
    with torch.no_grad():
        for binary in model["models"]:
            if binary["constant"] is not None:
                column = torch.full(
                    (len(x),), binary["constant"], dtype=torch.int64, device=device
                )
            else:
                probability, _ = _gpu_forward(torch, x, binary["parameters"])
                column = (probability[:, 0] > 0.5).to(torch.int64)
            columns.append(column)
        predicted = torch.stack(columns, dim=1).cpu().numpy()
    return predicted


def _pytorch(clips_file, baseline, conditions, gpu, seed):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch nao encontrou GPU CUDA.")
    if gpu is None:
        gpu = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    targets, train, test, rows = _problem(clips_file)
    original = _embeddings(baseline["file"], rows)
    clean, training = _torch_train(torch, original, targets, train, device, seed)
    started = time.perf_counter()
    predictions = {_key("baseline", "orig", "clean_to_clean"): _torch_predict(torch, clean, original[test], device)}
    inference = time.perf_counter() - started
    for source in conditions:
        compressed = _embeddings(source["file"], rows)
        started = time.perf_counter()
        predictions[_key(source["codec"], source["setting"], "train_orig_test_comp")] = _torch_predict(
            torch, clean, compressed[test], device
        )
        inference += time.perf_counter() - started
        matched, seconds = _torch_train(torch, compressed, targets, train, device, seed)
        training += seconds
        started = time.perf_counter()
        predictions[_key(source["codec"], source["setting"], "train_comp_test_comp")] = _torch_predict(
            torch, matched, compressed[test], device
        )
        inference += time.perf_counter() - started
        del matched
    torch.cuda.synchronize(device)
    return (
        predictions,
        training,
        inference,
        1,
        torch.cuda.max_memory_allocated(device) / 1024**2,
        f"GPU {gpu}: {torch.cuda.get_device_name(device)}",
    )


def _rss(process: psutil.Process) -> int:
    total = 0
    for item in [process, *process.children(recursive=True)]:
        try:
            total += item.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _monitor(stop: threading.Event, peak: list[int]) -> None:
    process = psutil.Process(os.getpid())
    while not stop.wait(0.05):
        peak[0] = max(peak[0], _rss(process))
    peak[0] = max(peak[0], _rss(process))


def _worker(strategy, clips_file, baseline, conditions, workers, gpu, seed, connection):
    stop = threading.Event()
    peak = [0]
    monitor = threading.Thread(target=_monitor, args=(stop, peak), daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        if strategy == "pytorch":
            predictions, training, inference, used, vram, device = _pytorch(
                Path(clips_file), baseline, conditions, gpu, seed
            )
        else:
            predictions, training, inference, used = _sklearn(
                strategy, Path(clips_file), baseline, conditions, workers, seed
            )
            vram, device = 0.0, "CPU"
        targets, _, test, _ = _problem(Path(clips_file))
        details = {key: _metrics(targets[test], value) for key, value in predictions.items()}
        stop.set()
        monitor.join()
        connection.send(
            {
                "kind": "result",
                "strategy_id": strategy,
                "device": device,
                "workers": used,
                "conditions": len(conditions),
                "result_rows": len(predictions),
                "training_seconds": training,
                "inference_seconds": inference,
                "total_seconds": time.perf_counter() - started,
                "peak_ram_mib": peak[0] / 1024**2,
                "peak_vram_mib": vram,
                "predictions": predictions,
                "details": details,
            }
        )
    except Exception:
        stop.set()
        monitor.join()
        connection.send({"kind": "error", "error": traceback.format_exc()})
    finally:
        connection.close()


def _execute(strategy, clips_file, baseline, conditions, workers, gpu, seed):
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(strategy, str(clips_file), baseline, conditions, workers, gpu, seed, child),
    )
    process.start()
    child.close()
    try:
        if not parent.poll(7200):
            process.terminate()
            raise TimeoutError(f"Estrategia {strategy} excedeu duas horas.")
        result = parent.recv()
    finally:
        parent.close()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError(f"Estrategia {strategy} nao encerrou.")
    if result["kind"] == "error":
        raise RuntimeError(result["error"])
    return result


def run(dataset: str, gpu=None, seed=SEED, embeddings_index=None, max_conditions=None):
    paths = CodecoPath.discover()
    log = _logging(paths)
    _header()
    index = embeddings_index or paths.tables / "birdnet_embeddings.csv"
    clips, baseline, conditions = _sources(index, dataset, max_conditions)
    cores = max(1, psutil.cpu_count(logical=False) or 1)
    output = paths.benchmarks / "birdnet_classification_benchmark.csv"
    detail_output = paths.benchmarks / "birdnet_classification_benchmark_details.csv"
    _section("Configuracao")
    LOGGER.info("Log: %s", log)
    LOGGER.info("Dataset: %s | condicoes comprimidas: %d", dataset, len(conditions))
    LOGGER.info("Nucleos fisicos: %d | uma condicao completa por processo", cores)
    LOGGER.info("MLP por classe: %d -> %d ReLU -> 1", INPUTS, HIDDEN)

    labels = {
        "current": "Atual: condicoes sequenciais",
        "parallel": "CPU: condicoes em processos",
        "pytorch": "GPU: condicoes com PyTorch",
    }
    results, detail_rows = [], []
    reference = None
    for number, strategy in enumerate(labels, 1):
        _section(f"Estrategia {number}/3: {labels[strategy]}")
        result = _execute(strategy, clips, baseline, conditions, cores, gpu, seed)
        predictions = result.pop("predictions")
        details = result.pop("details")
        result.pop("kind")
        if reference is None:
            reference = predictions
        agreements = []
        differences = 0
        ordered_keys = sorted(details)
        for key in ordered_keys:
            metrics = details[key]
            codec, setting, protocol = key.split("\0")
            agreement = float(np.mean(predictions[key] == reference[key]))
            difference = int(np.count_nonzero(predictions[key] != reference[key]))
            agreements.append(agreement)
            differences += difference
            detail_rows.append(
                {
                    "strategy_id": strategy,
                    "codec": codec,
                    "setting": setting,
                    "protocol": protocol,
                    **metrics,
                    "prediction_agreement": agreement,
                    "prediction_difference": difference,
                }
            )
        result.update(
            dataset=dataset,
            strategy=labels[strategy],
            f1_macro_mean=float(
                np.mean([details[key]["f1_macro"] for key in ordered_keys])
            ),
            balanced_accuracy_mean=float(
                np.mean(
                    [details[key]["balanced_accuracy"] for key in ordered_keys]
                )
            ),
            prediction_agreement=float(np.mean(agreements)),
            prediction_difference=differences,
            equivalent=differences == 0,
        )
        results.append(result)
        LOGGER.info("Total %.2f s | treino %.2f s | inferencia %.2f s", result["total_seconds"], result["training_seconds"], result["inference_seconds"])

    baseline_time = results[0]["total_seconds"]
    for result in results:
        result["speedup"] = baseline_time / result["total_seconds"]
    summary = pd.DataFrame(results)
    columns = [
        "dataset", "strategy", "strategy_id", "device", "workers", "conditions",
        "result_rows", "training_seconds", "inference_seconds", "total_seconds",
        "speedup", "peak_ram_mib", "peak_vram_mib", "f1_macro_mean",
        "balanced_accuracy_mean", "prediction_agreement", "prediction_difference",
        "equivalent",
    ]
    _write_csv(summary[columns], output)
    _write_csv(pd.DataFrame(detail_rows), detail_output)
    _section("Tabela comparativa")
    LOGGER.info("%s", summary[["strategy", "workers", "conditions", "total_seconds", "speedup", "peak_ram_mib", "peak_vram_mib", "prediction_agreement"]].to_string(index=False))
    LOGGER.info("Resumo: %s\nDetalhes: %s", output, detail_output)
    invalid = [
        result["strategy"]
        for result in results
        if result["strategy_id"] != "current" and not result["equivalent"]
    ]
    if invalid:
        raise RuntimeError(
            "Estrategias reprovadas por divergencia de predicoes: "
            + ", ".join(invalid)
        )
    return output, detail_output


def _arguments():
    parser = argparse.ArgumentParser(description="Benchmark BirdNET por condicao codec/configuracao.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--embeddings-index", type=Path)
    parser.add_argument("--max-conditions", type=int)
    arguments = parser.parse_args()
    if arguments.max_conditions is not None and arguments.max_conditions < 1:
        parser.error("--max-conditions deve ser maior que zero.")
    return arguments


def main() -> None:
    arguments = _arguments()
    run(
        arguments.dataset,
        arguments.gpu,
        arguments.seed,
        arguments.embeddings_index,
        arguments.max_conditions,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception:
        LOGGER.exception("Falha no benchmark BirdNET por condicao.")
        raise SystemExit(1) from None
