"""Processamento e fatiamento dos datasets (EcoSoundSet via CSV e PteroSet via JSON COCO-style)
em janelas padronizadas de C.CLIP_SECONDS segundos, garantindo:
- Split livre de vazamento (subset oficial no EcoSoundSet; site-disjoint split no PteroSet)
- Filtragem estrita das C.TOP_N_CLASSES classes mais frequentes
- Amostragem alvo (C.SR) e quantizacao PCM de 16-bit
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit

import config as C


def _select_top_classes(raw_labels: list[str], top_n: int = C.TOP_N_CLASSES) -> list[str]:
    """Retorna a lista ordenada das top_n classes mais frequentes."""
    counts = Counter(raw_labels)
    top = [cls for cls, _ in counts.most_common(top_n)]
    return sorted(top)


def _print_split_summary(df: pd.DataFrame, top_classes: list[str], dataset_name: str):
    """Exibe a distribuicao de clipes e classes por split para auditoria metodologica."""
    print(f"\n=== Resumo {dataset_name.upper()} ===")
    print("Clipes por split:", df["split"].value_counts().to_dict())
    for sp in ("train", "val", "test"):
        sub = df[df["split"] == sp]
        cls_counts = {c: 0 for c in top_classes}
        for lbls in sub["labels"].fillna(""):
            if lbls:
                for c in lbls.split("|"):
                    if c in cls_counts:
                        cls_counts[c] += 1
        print(f"  [{sp}] distribuicao de classes: {cls_counts}")


def process_ecosoundset():
    cfg = C.DATASETS["ecosoundset"]
    df = pd.read_csv(cfg["annotations_csv"])

    f_col = cfg["cols"]["file"]
    start_col = cfg["cols"]["start"]
    end_col = cfg["cols"]["end"]
    label_col = cfg["cols"]["label"]

    # 1. Identifica as Top-N classes mais frequentes nas anotacoes
    valid_labels = df[label_col].dropna().astype(str).tolist()
    top_classes = _select_top_classes(valid_labels, C.TOP_N_CLASSES)
    top_set = set(top_classes)

    rows = []
    # Agrupa por arquivo de audio original para fatiar sem vazamento entre janelas
    for filename, group in df.groupby(f_col):
        audio_path = Path(cfg["audio_dir"]) / filename
        if not audio_path.exists():
            matches = list(Path(cfg["audio_dir"]).rglob(filename))
            if matches:
                audio_path = matches[0]
            else:
                continue

        info = sf.info(str(audio_path))
        duration = info.duration

        # Usa a particao oficial do EcoSoundSet (evita vazamento entre gravacoes)
        subset_val = str(group["subset"].iloc[0]).lower() if "subset" in group.columns else "train"
        if subset_val not in ("train", "val", "test"):
            subset_val = "train"

        n_windows = int(duration // C.CLIP_SECONDS)
        for w in range(n_windows):
            w_start = w * C.CLIP_SECONDS
            w_end = (w + 1) * C.CLIP_SECONDS

            clip_labels = set()
            for _, row in group.iterrows():
                lbl = str(row[label_col])
                if lbl not in top_set:
                    continue
                overlap = max(0.0, min(w_end, float(row[end_col])) - max(w_start, float(row[start_col])))
                if overlap >= C.MIN_OVERLAP_SECONDS:
                    clip_labels.add(lbl)

            if not clip_labels and not getattr(C, "KEEP_EMPTY_WINDOWS", False):
                continue

            # Janelas sem vocalizacao ficam com string vazia (vetor zero no MultiLabelBinarizer)
            lbl_str = "|".join(sorted(clip_labels))

            rows.append({
                "clip_id": f"{Path(filename).stem}_win{w:04d}",
                "path": str(audio_path),
                "start": w_start,
                "end": w_end,
                "labels": lbl_str,
                "split": subset_val,
            })

    out_df = pd.DataFrame(rows)
    C.OUT_DIR.mkdir(exist_ok=True)
    out_df.to_csv(C.OUT_DIR / "clips_ecosoundset.csv", index=False)
    _print_split_summary(out_df, top_classes, "ecosoundset")


def _compute_site_disjoint_splits(
    sound_ids: np.ndarray,
    site_groups: np.ndarray,
    sound_classes: dict[int, set[str]],
    top_classes: list[str],
) -> dict[int, str]:
    """Busca uma divisao agrupada por local (site_id) onde todas as Top-N classes
    estejam presentes em train, val e test.
    """
    top_set = set(top_classes)
    best_split_map = None
    best_coverage = -1

    for offset in range(100):
        seed = C.SEED + offset
        gss1 = GroupShuffleSplit(n_splits=1, train_size=C.SPLIT_FRACTIONS[0], random_state=seed)
        train_idx, temp_idx = next(gss1.split(sound_ids, groups=site_groups))

        val_ratio = C.SPLIT_FRACTIONS[1] / (C.SPLIT_FRACTIONS[1] + C.SPLIT_FRACTIONS[2])
        gss2 = GroupShuffleSplit(n_splits=1, train_size=val_ratio, random_state=seed)
        val_sub, test_sub = next(gss2.split(sound_ids[temp_idx], groups=site_groups[temp_idx]))

        val_idx = temp_idx[val_sub]
        test_idx = temp_idx[test_sub]

        # Verifica quantas das top_classes estao presentes em cada split
        tr_cls = set().union(*(sound_classes[sound_ids[i]] for i in train_idx)) & top_set
        va_cls = set().union(*(sound_classes[sound_ids[i]] for i in val_idx)) & top_set
        te_cls = set().union(*(sound_classes[sound_ids[i]] for i in test_idx)) & top_set

        coverage = len(tr_cls) + len(va_cls) + len(te_cls)
        if coverage > best_coverage:
            best_coverage = coverage
            best_split_map = {}
            for i in train_idx:
                best_split_map[sound_ids[i]] = "train"
            for i in val_idx:
                best_split_map[sound_ids[i]] = "val"
            for i in test_idx:
                best_split_map[sound_ids[i]] = "test"

        if coverage == 3 * len(top_classes):
            break

    return best_split_map


def process_pteroset():
    cfg = C.DATASETS["pteroset"]
    json_path = Path(cfg["annotations_json"]) if "annotations_json" in cfg else Path(cfg["annotations_csv"])

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sounds = {s["id"]: s for s in data["sounds"]}
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}

    annos_by_sound = {}
    all_raw_labels = []
    for ann in data["annotations"]:
        s_id = ann["sound_id"]
        cat_name = str(ann.get("category") or categories.get(ann.get("category_id"), "unknown"))
        ann["resolved_label"] = cat_name
        annos_by_sound.setdefault(s_id, []).append(ann)
        all_raw_labels.append(cat_name)

    # 1. Filtra as Top-N classes mais frequentes
    top_classes = _select_top_classes(all_raw_labels, C.TOP_N_CLASSES)
    top_set = set(top_classes)

    # 2. Prepara agrupamento espacial por ponto de gravacao (project + lat/lon)
    valid_sound_ids = []
    site_groups = []
    sound_classes = {}

    for s_id, s_info in sounds.items():
        rel_path = s_info["file_name_path"]
        audio_path = Path(cfg["audio_dir"]).parent / rel_path
        if not audio_path.exists():
            matches = list(Path(cfg["audio_dir"]).rglob(Path(rel_path).name))
            if matches:
                audio_path = matches[0]
            else:
                continue
        s_info["resolved_path"] = str(audio_path)

        proj = str(s_info.get("project", "unk"))
        lat = round(float(s_info.get("latitude", 0.0)), 5)
        lon = round(float(s_info.get("longitude", 0.0)), 5)
        site_id = f"{proj}_{lat}_{lon}"

        valid_sound_ids.append(s_id)
        site_groups.append(site_id)
        sound_classes[s_id] = {a["resolved_label"] for a in annos_by_sound.get(s_id, [])}

    split_map = _compute_site_disjoint_splits(
        np.array(valid_sound_ids),
        np.array(site_groups),
        sound_classes,
        top_classes,
    )

    rows = []
    for s_id in valid_sound_ids:
        s_info = sounds[s_id]
        audio_path = s_info["resolved_path"]
        duration = float(s_info.get("duration") or sf.info(audio_path).duration)
        split_val = split_map[s_id]
        sound_annos = annos_by_sound.get(s_id, [])

        n_windows = int(duration // C.CLIP_SECONDS)
        for w in range(n_windows):
            w_start = w * C.CLIP_SECONDS
            w_end = (w + 1) * C.CLIP_SECONDS

            clip_labels = set()
            for ann in sound_annos:
                lbl = ann["resolved_label"]
                if lbl not in top_set:
                    continue
                overlap = max(0.0, min(w_end, float(ann["t_max"])) - max(w_start, float(ann["t_min"])))
                if overlap >= C.MIN_OVERLAP_SECONDS:
                    clip_labels.add(lbl)

            if not clip_labels and not getattr(C, "KEEP_EMPTY_WINDOWS", False):
                continue

            lbl_str = "|".join(sorted(clip_labels))

            rows.append({
                "clip_id": f"{Path(audio_path).stem}_win{w:04d}",
                "path": audio_path,
                "start": w_start,
                "end": w_end,
                "labels": lbl_str,
                "split": split_val,
            })

    out_df = pd.DataFrame(rows)
    C.OUT_DIR.mkdir(exist_ok=True)
    out_df.to_csv(C.OUT_DIR / "clips_pteroset.csv", index=False)
    _print_split_summary(out_df, top_classes, "pteroset")


def load_clip(path: str, start: float) -> np.ndarray:
    """Carrega um clipe de C.CLIP_SECONDS a C.SR (48 kHz) e aplica quantizacao PCM 16-bit."""
    sr_target = C.SR
    info = sf.info(path)
    sr_orig = info.samplerate

    start_frame = int(start * sr_orig)
    frames_to_read = int(C.CLIP_SECONDS * sr_orig)

    data, _ = sf.read(path, start=start_frame, frames=frames_to_read, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr_orig != sr_target:
        import soxr
        data = soxr.resample(data, sr_orig, sr_target, quality="HQ")

    if len(data) > C.CLIP_SAMPLES:
        data = data[:C.CLIP_SAMPLES]
    elif len(data) < C.CLIP_SAMPLES:
        data = np.pad(data, (0, C.CLIP_SAMPLES - len(data)))

    # Garante profundidade de 16-bit PCM identica para baseline e codecs neurais/classicos
    if C.BIT_DEPTH == 16:
        data = np.round(np.clip(data, -1.0, 1.0) * 32767.0) / 32768.0

    return data.astype(np.float32)


if __name__ == "__main__":
    process_ecosoundset()
    process_pteroset()