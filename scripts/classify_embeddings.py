"""Extracao de embeddings congelados (BirdNET v2.4 e Perch) e treinamento/avaliacao
de cabeças classificadoras (Linear / MLP) nos dois protocolos:
1) Realista (Cross-domain): Treino no Original -> Teste no Comprimido
2) Matched (In-domain): Treino no Comprimido -> Teste no Comprimido
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soxr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, multilabel_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

import config as C
from codecs_lib import align_to_reference, make_codec
from data import load_clip


def adapt_for_birdnet(x_5s: np.ndarray, sr: int = C.SR) -> np.ndarray:
    """Recorte central (center crop) de 3.0s a 48 kHz para BirdNET v2.4."""
    target_sr = 48_000
    target_samples = int(3.0 * target_sr)
    if sr != target_sr:
        x_5s = soxr.resample(x_5s, sr, target_sr, quality="HQ")
    if len(x_5s) > target_samples:
        start = (len(x_5s) - target_samples) // 2
        return x_5s[start : start + target_samples].astype(np.float32)
    elif len(x_5s) < target_samples:
        return np.pad(x_5s, (0, target_samples - len(x_5s))).astype(np.float32)
    return x_5s.astype(np.float32)


def adapt_for_perch(x_5s: np.ndarray, sr: int = C.SR) -> np.ndarray:
    """Reamostragem para 32 kHz e garantia de 5.0s (160.000 amostras) para Google Perch."""
    target_sr = 32_000
    target_samples = int(5.0 * target_sr)
    x_32k = soxr.resample(x_5s, sr, target_sr, quality="HQ") if sr != target_sr else x_5s
    if len(x_32k) > target_samples:
        start = (len(x_32k) - target_samples) // 2
        return x_32k[start : start + target_samples].astype(np.float32)
    elif len(x_32k) < target_samples:
        pad_total = target_samples - len(x_32k)
        return np.pad(x_32k, (pad_total // 2, pad_total - pad_total // 2)).astype(np.float32)
    return x_32k.astype(np.float32)


class BirdNETExtractor:
    """Extrator de embeddings do BirdNET v2.4 diretamente em memoria via TFLite."""
    def __init__(self):
        try:
            from birdnetlib.analyzer import Analyzer
            self.analyzer = Analyzer()
            self.interp = self.analyzer.interpreter
            self.in_idx = self.interp.get_input_details()[0]["index"]
            self.emb_idx = self.analyzer.embeddings_layer_index
        except ImportError:
            raise ImportError("Instale `birdnetlib` e `tflite-runtime` (ou `tensorflow`).")

    def extract_batch(self, clips_5s: list[np.ndarray]) -> np.ndarray:
        embs = []
        for x in clips_5s:
            x_3s = adapt_for_birdnet(x, C.SR)[None, :]  # (1, 144000)
            self.interp.set_tensor(self.in_idx, x_3s)
            self.interp.invoke()
            emb = self.interp.get_tensor(self.emb_idx)[0]
            embs.append(emb.astype(np.float32))
        return np.vstack(embs)


class PerchExtractor:
    """Extrator de embeddings do Google Perch (TFHub)."""
    def __init__(self, model_url: str = "https://tfhub.dev/google/bird-vocalization-classifier/4"):
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
            self.tf = tf
            self.model = hub.load(model_url)
        except ImportError:
            raise ImportError("Instale `tensorflow` e `tensorflow-hub` para usar o Google Perch.")

    def extract_batch(self, clips_5s: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        adapted = np.stack([adapt_for_perch(x, C.SR) for x in clips_5s], axis=0)
        embs = []
        for i in range(0, len(adapted), batch_size):
            batch = self.tf.constant(adapted[i : i + batch_size], dtype=self.tf.float32)
            outputs = self.model.signatures["serving_default"](inputs=batch)
            emb_tensor = outputs.get("embedding", list(outputs.values())[-1])
            embs.append(emb_tensor.numpy())
        return np.vstack(embs).astype(np.float32)


def get_or_extract_embeddings(
    extractor,
    clips_fn,
    cache_file: Path,
) -> np.ndarray:
    """Carrega embeddings do cache se disponivel; caso contrario, gera os clipes via `clips_fn()`,
    extrai os embeddings e salva em disco de forma atomica.
    """
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        print(f"  [CACHE HIT] Carregando embeddings de {cache_file.name}")
        return np.load(cache_file)

    clips = clips_fn()
    embs = extractor.extract_batch(clips)

    # Salva via arquivo temporario + rename para evitar arquivo .npy corrompido se cair no meio
    tmp_file = cache_file.with_suffix(".tmp.npy")
    np.save(tmp_file, embs)
    tmp_file.replace(cache_file)
    return embs


def build_classifier(head_type: str = "mlp"):
    if head_type == "linear":
        base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=C.SEED)
    elif head_type == "mlp":
        base = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=1e-3,
            max_iter=300,
            early_stopping=True,
            random_state=C.SEED,
        )
    else:
        raise ValueError(f"head_type desconhecido: {head_type}")
    return make_pipeline(StandardScaler(), OneVsRestClassifier(base))


def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    f1_mac = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    bal_accs = [
        balanced_accuracy_score(Y_true[:, j], Y_pred[:, j])
        for j in range(Y_true.shape[1])
    ]
    return {
        "f1_macro": float(f1_mac),
        "balanced_accuracy": float(np.mean(bal_accs)),
    }


def run_classification(dataset: str, model_name: str, head_type: str = "mlp"):
    clips_df = pd.read_csv(C.OUT_DIR / f"clips_{dataset}.csv")
    label_lists = [
        s.split("|") if isinstance(s, str) and len(s) > 0 else []
        for s in clips_df["labels"]
    ]
    mlb = MultiLabelBinarizer()
    Y_all = mlb.fit_transform(label_lists)
    classes = list(mlb.classes_)

    with open(C.OUT_DIR / f"classes_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    tr_mask = (clips_df["split"] == "train").values
    va_mask = (clips_df["split"] == "val").values
    te_mask = (clips_df["split"] == "test").values

    # Lazy init do extrator: se TODOS os embeddings ja estiverem em cache, nem carrega o TF/TFLite
    _extractor_instance = None
    def get_extractor():
        nonlocal _extractor_instance
        if _extractor_instance is None:
            _extractor_instance = BirdNETExtractor() if model_name == "birdnet" else PerchExtractor()
        return _extractor_instance

    # Carregamento sob demanda dos clipes originais
    _orig_clips_cache = None
    def get_orig_clips():
        nonlocal _orig_clips_cache
        if _orig_clips_cache is None:
            _orig_clips_cache = [load_clip(r["path"], r["start"]) for _, r in clips_df.iterrows()]
        return _orig_clips_cache

    print(f"[{dataset} | {model_name}] Obtendo embeddings da baseline (original)...")
    base_cache = C.CACHE_DIR / f"emb_{dataset}_{model_name}_baseline_orig.npy"
    X_orig = get_or_extract_embeddings(
        extractor=get_extractor() if not base_cache.exists() else None,
        clips_fn=get_orig_clips,
        cache_file=base_cache,
    )

    clf_clean = build_classifier(head_type)
    clf_clean.fit(X_orig[tr_mask], Y_all[tr_mask])

    val_metrics = compute_metrics(Y_all[va_mask], clf_clean.predict(X_orig[va_mask]))
    print(f"[{dataset} | {model_name} | {head_type}] Val (clean->clean): {val_metrics}")

    Y_pred_base = clf_clean.predict(X_orig[te_mask])
    base_metrics = compute_metrics(Y_all[te_mask], Y_pred_base)

    results = [{
        "dataset": dataset,
        "foundation_model": model_name,
        "head": head_type,
        "codec": "baseline",
        "setting": "orig",
        "protocol": "clean_to_clean",
        **base_metrics,
    }]
    np.save(
        C.OUT_DIR / f"cm_{dataset}_{model_name}_baseline_orig.npy",
        multilabel_confusion_matrix(Y_all[te_mask], Y_pred_base),
    )

    for codec_name, settings in C.CODECS.items():
        for params in settings:
            codec = make_codec(codec_name, params, dataset=dataset)
            safe_setting = codec.setting.replace("=", "_").replace(",", "-")
            comp_cache = C.CACHE_DIR / f"emb_{dataset}_{model_name}_{codec_name}_{safe_setting}.npy"

            print(f"[{dataset} | {model_name}] Processando codec {codec_name} ({codec.setting})...")

            def encode_decode_clips(c=codec):
                comp = []
                for x in get_orig_clips():
                    enc = c.encode(x)
                    y, _ = align_to_reference(x, c.decode(enc))
                    comp.append(y)
                return comp

            X_comp = get_or_extract_embeddings(
                extractor=get_extractor() if not comp_cache.exists() else None,
                clips_fn=encode_decode_clips,
                cache_file=comp_cache,
            )

            # Protocolo 1: Treinado no Original -> Testado no Comprimido (Realista IoT)
            Y_pred_real = clf_clean.predict(X_comp[te_mask])
            m_real = compute_metrics(Y_all[te_mask], Y_pred_real)
            results.append({
                "dataset": dataset,
                "foundation_model": model_name,
                "head": head_type,
                "codec": codec_name,
                "setting": codec.setting,
                "protocol": "train_orig_test_comp",
                **m_real,
            })

            # Protocolo 2: Treinado no Comprimido -> Testado no Comprimido (Matched)
            clf_matched = build_classifier(head_type)
            clf_matched.fit(X_comp[tr_mask], Y_all[tr_mask])
            Y_pred_match = clf_matched.predict(X_comp[te_mask])
            m_match = compute_metrics(Y_all[te_mask], Y_pred_match)
            results.append({
                "dataset": dataset,
                "foundation_model": model_name,
                "head": head_type,
                "codec": codec_name,
                "setting": codec.setting,
                "protocol": "train_comp_test_comp",
                **m_match,
            })

            np.save(
                C.OUT_DIR / f"cm_{dataset}_{model_name}_{codec_name}_{safe_setting}_real.npy",
                multilabel_confusion_matrix(Y_all[te_mask], Y_pred_real),
            )

    out_csv = C.OUT_DIR / f"classification_{dataset}_{model_name}_{head_type}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Resultados de classificacao salvos em: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Classificacao com embeddings congelados (BirdNET / Perch)")
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS.keys()))
    ap.add_argument("--model", required=True, choices=["birdnet", "perch"])
    ap.add_argument("--head", default="mlp", choices=["linear", "mlp"])
    args = ap.parse_args()

    C.OUT_DIR.mkdir(exist_ok=True)
    run_classification(args.dataset, args.model, args.head)