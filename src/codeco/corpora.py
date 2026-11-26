"""Read-only access to the frozen manifests and auditable class selection."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CORPUS_MANIFESTS = {
    "ecosoundset": "data/manifests/ecosoundset_manifest_v1.csv",
    "pteroset_ppa1": "data/manifests/pteroset_ppa1_manifest_v1.csv",
}


@dataclass(frozen=True)
class SelectionRule:
    """One-sided cutoff fitted to log1p training-window counts.

    Increasing ``k_sigma`` lowers the cutoff, admitting more classes with less
    support. ``minimum_groups`` prevents a class concentrated in too few
    independent recording groups from entering the experiment.
    """

    k_sigma: float = 1.0
    minimum_groups: int = 3
    seed: int = 42
    fit_split: str = "train"


def repository_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "data" / "freeze-spec-v1.json").is_file():
            return candidate
    raise FileNotFoundError("Could not locate the CodEco repository root.")


def load_frozen_manifest(corpus: str, root: Path | None = None) -> pd.DataFrame:
    if corpus not in CORPUS_MANIFESTS:
        raise KeyError(f"Unknown corpus {corpus!r}; choose from {sorted(CORPUS_MANIFESTS)}")
    root = repository_root(root)
    path = root / CORPUS_MANIFESTS[corpus]
    df = pd.read_csv(path, keep_default_na=False)
    if "freeze_status" in df:
        df = df[df["freeze_status"] == "frozen_analysis_released"].copy()
    df["corpus"] = corpus
    df["split"] = df["source_split"].astype(str)
    df["labels"] = df["target_labels"].fillna("").astype(str)
    df["manifest_path"] = str(path.relative_to(root))
    return df.reset_index(drop=True)


def explode_labels(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    work = df if split is None else df[df["split"] == split]
    rows: list[dict] = []
    for row in work.itertuples(index=False):
        labels = [label.strip() for label in str(row.labels).split("|") if label.strip()]
        for label in labels:
            rows.append(
                {
                    "corpus": row.corpus,
                    "split": row.split,
                    "source_group_id": row.source_group_id,
                    "window_id": row.window_id,
                    "label": label,
                }
            )
    return pd.DataFrame(rows, columns=["corpus", "split", "source_group_id", "window_id", "label"])


def class_distribution(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    long = explode_labels(df, split=split)
    if long.empty:
        return pd.DataFrame(columns=["label", "windows", "groups"])
    return (
        long.groupby("label", as_index=False)
        .agg(windows=("window_id", "nunique"), groups=("source_group_id", "nunique"))
        .sort_values(["windows", "label"], ascending=[False, True])
        .reset_index(drop=True)
    )


def select_classes(
    df: pd.DataFrame,
    rule: SelectionRule = SelectionRule(),
) -> tuple[pd.DataFrame, dict]:
    """Fit the log-count cutoff on the training split and return an audit table."""
    dist = class_distribution(df, split=rule.fit_split)
    if dist.empty:
        raise ValueError(f"No labelled windows in split {rule.fit_split!r}.")
    log_counts = np.log1p(dist["windows"].astype(float).to_numpy())
    mu = float(log_counts.mean())
    sigma = float(log_counts.std(ddof=0))
    log_cutoff = mu - rule.k_sigma * sigma
    cutoff = max(1, int(math.ceil(math.expm1(log_cutoff))))
    out = dist.copy()
    out["log1p_windows"] = log_counts
    out["z_score"] = 0.0 if sigma == 0 else (log_counts - mu) / sigma
    out["count_cutoff"] = cutoff
    out["enough_windows"] = out["windows"] >= cutoff
    out["enough_groups"] = out["groups"] >= rule.minimum_groups
    out["selected"] = out["enough_windows"] & out["enough_groups"]
    out["selection_reason"] = np.select(
        [out["selected"], ~out["enough_windows"], ~out["enough_groups"]],
        ["selected", "below_count_cutoff", "below_group_cutoff"],
        default="excluded",
    )
    selected = out.loc[out["selected"], "label"].tolist()
    metadata = {
        "rule": asdict(rule),
        "fit_distribution": "Gaussian moments on log1p training-window counts",
        "log_mean": mu,
        "log_std": sigma,
        "count_cutoff": cutoff,
        "selected_class_count": len(selected),
        "selected_classes": selected,
    }
    return out, metadata


def filter_selected_classes(df: pd.DataFrame, selected: Iterable[str]) -> pd.DataFrame:
    """Keep windows containing selected classes and remove all other labels."""
    selected_set = set(selected)
    rows = []
    for row in df.to_dict(orient="records"):
        labels = [label for label in str(row["labels"]).split("|") if label in selected_set]
        if not labels:
            continue
        row = dict(row)
        row["labels"] = "|".join(sorted(set(labels)))
        row["selected_label_count"] = len(set(labels))
        rows.append(row)
    return pd.DataFrame(rows)


def deterministic_balanced_training_sample(
    df: pd.DataFrame,
    selected: Iterable[str],
    quota: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily cover each selected class up to a common training quota.

    Multi-label windows can increase another class above the quota. The audit
    table reports achieved counts explicitly; validation and test are retained
    in their natural distributions.
    """
    selected = sorted(set(selected))
    train = filter_selected_classes(df[df["split"] == "train"], selected)
    other = filter_selected_classes(df[df["split"] != "train"], selected)
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    achieved = {label: 0 for label in selected}
    label_sets = {
        idx: set(str(labels).split("|"))
        for idx, labels in train["labels"].items()
    }
    availability = {
        label: sum(label in labels for labels in label_sets.values())
        for label in selected
    }
    for label in sorted(selected, key=lambda item: (availability[item], item)):
        candidates = [idx for idx, labels in label_sets.items() if label in labels and idx not in chosen]
        rng.shuffle(candidates)
        while achieved[label] < quota and candidates:
            idx = candidates.pop()
            chosen.add(idx)
            for present in label_sets[idx] & set(selected):
                achieved[present] += 1
    sampled_train = train.loc[sorted(chosen)].copy()
    sampled_train["selection_partition"] = "balanced_train"
    other = other.copy()
    other["selection_partition"] = "natural_evaluation"
    sampled = pd.concat([sampled_train, other], ignore_index=True)
    audit = pd.DataFrame(
        {
            "label": selected,
            "available_train_windows": [availability[label] for label in selected],
            "requested_quota": quota,
            "achieved_train_windows": [achieved[label] for label in selected],
        }
    )
    return sampled, audit


def freeze_selection(
    corpus: str,
    selection_table: pd.DataFrame,
    metadata: dict,
    sampled_manifest: pd.DataFrame,
    balance_audit: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write a complete, hash-addressable record of the chosen corpus slice."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{corpus}_selection"
    paths = {
        "classes": output_dir / f"{stem}_classes.csv",
        "manifest": output_dir / f"{stem}_manifest.csv",
        "balance": output_dir / f"{stem}_balance.csv",
        "metadata": output_dir / f"{stem}.json",
    }
    selection_table.to_csv(paths["classes"], index=False)
    sampled_manifest.to_csv(paths["manifest"], index=False)
    balance_audit.to_csv(paths["balance"], index=False)
    hashes = {}
    for key in ("classes", "manifest", "balance"):
        hashes[key] = hashlib.sha256(paths[key].read_bytes()).hexdigest()
    payload = dict(metadata)
    payload["corpus"] = corpus
    payload["artifacts_sha256"] = hashes
    paths["metadata"].write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths


def corpus_summary(df: pd.DataFrame) -> dict:
    return {
        "corpus": str(df["corpus"].iloc[0]),
        "windows": int(len(df)),
        "groups": int(df["source_group_id"].nunique()),
        "classes": int(class_distribution(df)["label"].nunique()),
        "splits": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()},
        "multilabel_windows": int((pd.to_numeric(df["target_label_count"], errors="coerce") > 1).sum()),
        "manifest": str(df["manifest_path"].iloc[0]),
    }


def load_manifest_windows(corpus: str, manifest_path: Path) -> pd.DataFrame:
    """Load one corpus from ``config/manifests.parquet`` as ``load_frozen_manifest`` does from CSV."""
    if corpus not in CORPUS_MANIFESTS:
        raise KeyError(f"Unknown corpus {corpus!r}; choose from {sorted(CORPUS_MANIFESTS)}")
    df = pd.read_parquet(
        manifest_path,
        filters=[("resource_type", "=", "window"), ("dataset", "=", corpus)],
    )
    if "freeze_status" in df:
        df = df[df["freeze_status"] == "frozen_analysis_released"].copy()
    df["window_id"] = df["window_id"].astype(str)
    df["corpus"] = corpus
    df["split"] = df["source_split"].astype(str)
    df["labels"] = df["target_labels"].fillna("").astype(str)
    df["manifest_path"] = str(manifest_path)
    return df.reset_index(drop=True)
