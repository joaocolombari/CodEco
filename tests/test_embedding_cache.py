from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeco.birdnet_embeddings import _completed_indices, _connect_checkpoint
from codeco.codeco_path import CodecoPath
from codeco.embedding_cache import seed_compatible_embeddings


class EmbeddingCacheTests(unittest.TestCase):
    def test_condition_cache_is_reused_across_run_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = CodecoPath(Path(temporary_directory))
            dataset = "example"
            condition = SimpleNamespace(
                identifier="encodec-test",
                codec="encodec",
                setting="bandwidth_kbps=6,model=encodec_48khz",
            )
            clips = pd.DataFrame({"clip_id": ["clip-a", "clip-b"]})
            source_root = paths.cache / "birdnet_embeddings" / "source" / dataset
            source_root.mkdir(parents=True)
            clips_file = source_root / "clips.parquet"
            pd.DataFrame({"clip_id": clips["clip_id"]}).to_parquet(clips_file)
            embeddings_file = source_root / "encodec-test.npy"
            np.save(
                embeddings_file,
                np.vstack(
                    [
                        np.linspace(-1.0, 1.0, 1024, dtype=np.float32),
                        np.linspace(1.0, -1.0, 1024, dtype=np.float32),
                    ]
                ),
            )

            handoff = paths.data / "filipe" / "package" / "handoff"
            handoff.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "dataset": dataset,
                        "foundation_model": "birdnet",
                        "model_version": "test-model",
                        "codec": condition.codec,
                        "setting": condition.setting,
                        "clips": len(clips),
                        "completed": len(clips),
                        "resumed": 0,
                        "processed": len(clips),
                        "seconds": 1.0,
                        "embeddings_per_second": 2.0,
                        "embedding_dimension": 1024,
                        "clips_file": paths.project_relative(clips_file),
                        "embeddings_file": paths.project_relative(embeddings_file),
                    }
                ]
            ).to_csv(handoff / "birdnet_embeddings_encodec.csv", index=False)

            checkpoint = paths.cache / "checkpoints.sqlite3"
            target = (
                paths.cache
                / "birdnet_embeddings"
                / "full-matrix-fingerprint"
                / dataset
                / "encodec-test.npy"
            )
            target.parent.mkdir(parents=True)
            np.save(target, np.zeros((2, 1024), dtype=np.float32))
            connection = _connect_checkpoint(checkpoint)
            try:
                imported = seed_compatible_embeddings(
                    paths=paths,
                    model="birdnet",
                    model_version="test-model",
                    fingerprint="full-matrix-fingerprint",
                    datasets={dataset: clips},
                    conditions=[condition],
                    cache_root=paths.cache / "birdnet_embeddings",
                    connection=connection,
                    logger=logging.getLogger("test.embedding-cache"),
                )
                self.assertEqual(imported, 1)
                self.assertTrue(target.is_file())
                np.testing.assert_array_equal(np.load(target), np.load(embeddings_file))
                self.assertEqual(
                    _completed_indices(
                        connection,
                        "full-matrix-fingerprint",
                        dataset,
                        condition.identifier,
                    ),
                    {0, 1},
                )

                repeated = seed_compatible_embeddings(
                    paths=paths,
                    model="birdnet",
                    model_version="test-model",
                    fingerprint="full-matrix-fingerprint",
                    datasets={dataset: clips},
                    conditions=[condition],
                    cache_root=paths.cache / "birdnet_embeddings",
                    connection=connection,
                    logger=logging.getLogger("test.embedding-cache"),
                )
                self.assertEqual(repeated, 0)
            finally:
                connection.close()

    def test_degenerate_birdnet_cache_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = CodecoPath(Path(temporary_directory))
            dataset = "example"
            condition = SimpleNamespace(
                identifier="baseline-test",
                codec="baseline",
                setting="orig",
            )
            clips = pd.DataFrame({"clip_id": ["clip-a", "clip-b"]})
            source_root = paths.cache / "birdnet_embeddings" / "source" / dataset
            source_root.mkdir(parents=True)
            clips_file = source_root / "clips.parquet"
            pd.DataFrame({"clip_id": clips["clip_id"]}).to_parquet(clips_file)
            embeddings_file = source_root / "baseline-test.npy"
            np.save(embeddings_file, np.ones((2, 1024), dtype=np.float32))

            handoff = paths.data / "handoff"
            handoff.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "dataset": dataset,
                        "foundation_model": "birdnet",
                        "model_version": "2.4",
                        "codec": condition.codec,
                        "setting": condition.setting,
                        "clips": len(clips),
                        "embedding_dimension": 1024,
                        "clips_file": paths.project_relative(clips_file),
                        "embeddings_file": paths.project_relative(embeddings_file),
                    }
                ]
            ).to_csv(handoff / "birdnet_embeddings.csv", index=False)

            connection = _connect_checkpoint(paths.cache / "checkpoints.sqlite3")
            try:
                imported = seed_compatible_embeddings(
                    paths=paths,
                    model="birdnet",
                    model_version="2.4",
                    fingerprint="new-fingerprint",
                    datasets={dataset: clips},
                    conditions=[condition],
                    cache_root=paths.cache / "birdnet_embeddings",
                    connection=connection,
                    logger=logging.getLogger("test.embedding-cache"),
                )
                self.assertEqual(imported, 0)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
