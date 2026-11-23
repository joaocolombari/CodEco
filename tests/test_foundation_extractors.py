from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from classify_embeddings import BirdNETExtractor


class _FakeTensor:
    def __init__(self, values: np.ndarray):
        self.values = values

    def numpy(self) -> np.ndarray:
        return self.values


class BirdNETExtractorTests(unittest.TestCase):
    def test_official_embedding_signature_is_batched(self) -> None:
        batch_sizes: list[int] = []

        def encode(*, inputs: np.ndarray) -> dict[str, _FakeTensor]:
            batch_sizes.append(len(inputs))
            values = np.repeat(
                inputs.mean(axis=1, keepdims=True),
                BirdNETExtractor.EMBEDDING_DIMENSION,
                axis=1,
            ).astype(np.float32)
            return {"embeddings": _FakeTensor(values)}

        fake_saved_model = types.SimpleNamespace(
            signatures={"embeddings": encode}
        )
        fake_birdnet = types.ModuleType("birdnet")
        fake_birdnet.load = lambda *args: types.SimpleNamespace(
            model_path=ROOT / "fake-birdnet-model"
        )
        fake_tensorflow = types.ModuleType("tensorflow")
        fake_tensorflow.float32 = np.float32
        fake_tensorflow.convert_to_tensor = (
            lambda values, dtype=None: np.asarray(values, dtype=dtype)
        )
        fake_tensorflow.saved_model = types.SimpleNamespace(
            load=lambda path: fake_saved_model
        )
        fake_tensorflow.config = types.SimpleNamespace(
            list_physical_devices=lambda device: [],
            experimental=types.SimpleNamespace(
                set_memory_growth=lambda device, enabled: None
            ),
        )

        clips = [
            np.full(192_000, value, dtype=np.float32)
            for value in (0.1, 0.2, 0.3)
        ]
        with patch.dict(
            sys.modules,
            {"birdnet": fake_birdnet, "tensorflow": fake_tensorflow},
        ), patch.dict(os.environ, {}, clear=False):
            embeddings = BirdNETExtractor(batch_size=2).extract_batch(clips)

        self.assertEqual(batch_sizes, [2, 1])
        self.assertEqual(embeddings.shape, (3, 1024))
        self.assertEqual(embeddings.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
