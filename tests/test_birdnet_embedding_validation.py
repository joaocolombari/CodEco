from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeco.birdnet_embeddings import _validate_embedding_values


class BirdNETEmbeddingValidationTests(unittest.TestCase):
    def test_rejects_constant_feature_vectors(self) -> None:
        values = np.repeat(
            np.asarray([[0.1], [0.2]], dtype=np.float32),
            1024,
            axis=1,
        )
        with self.assertRaisesRegex(RuntimeError, "degenerados"):
            _validate_embedding_values(values, context="test")

    def test_accepts_feature_vectors_with_internal_variation(self) -> None:
        values = np.vstack(
            [
                np.linspace(-1.0, 1.0, 1024, dtype=np.float32),
                np.linspace(1.0, -1.0, 1024, dtype=np.float32),
            ]
        )
        returned = _validate_embedding_values(values, context="test")
        self.assertIs(returned, values)


if __name__ == "__main__":
    unittest.main()
