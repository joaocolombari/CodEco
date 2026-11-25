"""Reuse compatible foundation-model embeddings across partial runs.

An embedding run fingerprint includes the complete list of requested codec
conditions.  Consequently, a run containing only ``baseline`` and ``encodec``
has a different fingerprint from a later full-matrix run even when the audio,
model and individual condition are identical.  This module validates those
condition-level artifacts and registers them in the active checkpoint.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from codeco.codeco_path import CodecoPath


REQUIRED_COLUMNS = {
    "dataset",
    "foundation_model",
    "model_version",
    "codec",
    "setting",
    "clips",
    "embedding_dimension",
    "clips_file",
    "embeddings_file",
}


def _candidate_indices(paths: CodecoPath, model: str) -> tuple[Path, ...]:
    """Return official and handoff indices in a deterministic order."""

    candidates = [paths.tables / f"{model}_embeddings.csv"]
    handoff_pattern = f"{model}_embeddings*.csv"
    for depth in ("handoff", "*/handoff", "*/*/handoff"):
        candidates.extend(sorted(paths.data.glob(f"{depth}/{handoff_pattern}")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if candidate.is_file() and resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return tuple(unique)


def _load_candidates(paths: CodecoPath, model: str) -> pd.DataFrame:
    tables = []
    for index in _candidate_indices(paths, model):
        try:
            table = pd.read_csv(index)
        except (OSError, pd.errors.ParserError):
            continue
        if not REQUIRED_COLUMNS.issubset(table.columns):
            continue
        table = table.copy()
        table["source_index"] = str(index)
        tables.append(table)
    if not tables:
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS | {"source_index"}))
    return pd.concat(tables, ignore_index=True)


def _same_clips(paths: CodecoPath, clips_file: object, expected: Sequence[str]) -> bool:
    try:
        candidate = pd.read_parquet(
            paths.project_path(str(clips_file)), columns=["clip_id"]
        )
    except (OSError, ValueError, KeyError):
        return False
    return candidate["clip_id"].astype(str).tolist() == list(expected)


def _valid_embedding(
    paths: CodecoPath,
    embeddings_file: object,
    rows: int,
    dimension: int,
    model: str,
) -> Path | None:
    try:
        candidate = paths.project_path(str(embeddings_file))
        values = np.load(candidate, mmap_mode="r")
    except (OSError, ValueError):
        return None
    if values.shape != (rows, dimension) or values.dtype != np.float32:
        return None
    for start in range(0, rows, 4096):
        if not np.isfinite(np.asarray(values[start : start + 4096])).all():
            return None
    if model == "birdnet" and rows:
        from codeco.birdnet_embeddings import _validate_embedding_values

        sample_indices = np.linspace(0, rows - 1, min(rows, 32), dtype=np.int64)
        try:
            _validate_embedding_values(
                np.asarray(values[sample_indices]),
                context=str(candidate),
            )
        except RuntimeError:
            return None
    return candidate


def _materialize(source: Path, target: Path) -> str:
    """Create a zero-copy hard link when possible, with a portable fallback."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and os.path.samefile(source, target):
        return "existing"
    temporary = target.with_name(f".{target.name}.importing")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
        method = "hardlink"
    except OSError:
        shutil.copy2(source, temporary)
        method = "copy"
    os.replace(temporary, target)
    return method


def seed_compatible_embeddings(
    *,
    paths: CodecoPath,
    model: str,
    model_version: str,
    fingerprint: str,
    datasets: dict[str, pd.DataFrame],
    conditions: Sequence[object],
    cache_root: Path,
    connection: object,
    logger: logging.Logger,
) -> int:
    """Register compatible condition-level caches in the active run.

    Every candidate is checked against the foundation model/version, codec
    setting, exact ordered clip IDs, row count, embedding dimension and NumPy
    dtype before it can be reused.
    """

    from codeco.birdnet_embeddings import (
        _checkpoint_batch,
        _completed_indices,
    )

    candidates = _load_candidates(paths, model)
    if candidates.empty:
        return 0
    candidates = candidates.loc[
        candidates["foundation_model"].astype(str).str.lower().eq(model)
        & candidates["model_version"].astype(str).eq(model_version)
    ]
    imported = 0
    for dataset, clips in datasets.items():
        expected_ids = clips["clip_id"].astype(str).tolist()
        for condition in conditions:
            completed = _completed_indices(
                connection, fingerprint, dataset, condition.identifier
            )
            if len(completed) == len(clips):
                continue
            clip_counts = pd.to_numeric(candidates["clips"], errors="coerce")
            matches = candidates.loc[
                candidates["dataset"].astype(str).eq(dataset)
                & candidates["codec"].astype(str).eq(condition.codec)
                & candidates["setting"].astype(str).eq(condition.setting)
                & clip_counts.eq(len(clips))
            ]
            for row in matches.itertuples(index=False):
                dimension = int(row.embedding_dimension)
                if not _same_clips(paths, row.clips_file, expected_ids):
                    continue
                source = _valid_embedding(
                    paths, row.embeddings_file, len(clips), dimension, model
                )
                if source is None:
                    continue
                target = (
                    cache_root
                    / fingerprint
                    / dataset
                    / f"{condition.identifier}.npy"
                )
                method = _materialize(source, target)
                _checkpoint_batch(
                    connection,
                    fingerprint,
                    dataset,
                    condition.identifier,
                    clips,
                    range(len(clips)),
                )
                imported += 1
                logger.info(
                    "[CACHE IMPORT] %s | %s | %s: %d embeddings (%s; %s).",
                    dataset,
                    condition.codec.upper(),
                    condition.setting,
                    len(clips),
                    method,
                    row.source_index,
                )
                break
    return imported
