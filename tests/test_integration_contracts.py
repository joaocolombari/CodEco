from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeco.acoustic_indices import _entropy
from codeco.alignment import normalized_cross_correlation_align
from codeco.experiment_config import experiment_config
from codeco.environment import _check_environment, _uv_executable
from codeco.protocol import PROTOCOL, adapt_for_perch
from codeco.corpora import (
    CORPUS_MANIFESTS,
    SelectionRule,
    deterministic_balanced_training_sample,
    load_frozen_manifest,
    select_classes,
)


class AlignmentTests(unittest.TestCase):
    def test_signed_positive_lag(self) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(size=12_000).astype(np.float32)
        decoded = np.pad(reference, (137, 0))
        result = normalized_cross_correlation_align(reference, decoded, 512)
        self.assertEqual(result.lag_samples, 137)
        self.assertGreater(result.score, 0.999)
        np.testing.assert_allclose(result.audio, reference, atol=1e-6)

    def test_signed_negative_lag(self) -> None:
        rng = np.random.default_rng(7)
        reference = rng.normal(size=12_000).astype(np.float32)
        result = normalized_cross_correlation_align(reference, reference[91:], 512)
        self.assertEqual(result.lag_samples, -91)
        np.testing.assert_allclose(result.audio[91:], reference[91:], atol=1e-6)


class MetricTests(unittest.TestCase):
    def test_entropy_uses_fixed_support(self) -> None:
        self.assertAlmostEqual(_entropy(np.ones(4)), 1.0)
        self.assertAlmostEqual(_entropy(np.array([1.0, 0.0, 0.0, 0.0])), 0.0)
        expected = -(0.5 * np.log(0.5) * 2) / np.log(4)
        self.assertAlmostEqual(_entropy(np.array([1.0, 1.0, 0.0, 0.0])), expected)


class EnvironmentTests(unittest.TestCase):
    def test_uv_is_found_inside_project_virtual_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "pyproject.toml").write_text(
                '[tool.codeco.environment]\nuv-version = "0.12.0"\n',
                encoding="utf-8",
            )
            executable = root / ".venv" / "bin" / "uv"
            executable.parent.mkdir(parents=True)
            executable.write_text(
                "#!/bin/sh\nprintf 'uv 0.12.0\\n'\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)

            with patch("codeco.environment.shutil.which", return_value=None):
                observed, version = _uv_executable(SimpleNamespace(root=root))

            self.assertEqual(observed, executable.resolve())
            self.assertEqual(version, "0.12.0")

    def test_active_environment_check_does_not_mutate_packages(self) -> None:
        root = Path("/tmp/codeco-environment-test")
        uv = Path("/tmp/codeco-environment-test-uv")
        with (
            patch(
                "codeco.environment._uv_executable",
                return_value=(uv, "0.12.0"),
            ),
            patch("codeco.environment.subprocess.run") as run,
        ):
            _check_environment(SimpleNamespace(root=root))

        run.assert_called_once_with(
            [
                str(uv),
                "sync",
                "--project",
                str(root),
                "--locked",
                "--inexact",
                "--check",
            ],
            check=True,
        )


class ProtocolTests(unittest.TestCase):
    def test_perch_padding_does_not_change_codec_window(self) -> None:
        canonical = np.ones(PROTOCOL.window_samples, dtype=np.float32)
        perch = adapt_for_perch(canonical)
        self.assertEqual(len(perch), round(5 * 32_000))
        self.assertEqual(np.count_nonzero(perch[:16_000]), 0)
        self.assertEqual(np.count_nonzero(perch[-16_000:]), 0)

    def test_executable_matrix_is_frozen(self) -> None:
        codecs = experiment_config()["codecs"]
        self.assertEqual(codecs["mp3"]["bitrates_kbps"], [32, 64, 128])
        self.assertEqual(codecs["opus"]["bitrates_kbps"], [12, 24, 48, 96])
        self.assertEqual(codecs["lc3"]["bitrates_kbps"], [32, 64, 96])

        from codeco.birdnet_embeddings import CONDITIONS as embedding_conditions
        from codeco.ffmpeg import CONDITIONS as ffmpeg_conditions

        expected = [(item.codec, item.setting) for item in ffmpeg_conditions]
        observed = [
            (item.codec, item.setting)
            for item in embedding_conditions
            if item.kind == "ffmpeg"
        ]
        self.assertEqual(observed, expected)

    def test_selected_corpus_sizes_are_reproducible(self) -> None:
        expected = {"ecosoundset": 7_127, "pteroset_ppa1": 777}
        rule = SelectionRule(k_sigma=1.0, minimum_groups=3, seed=42)
        for corpus in CORPUS_MANIFESTS:
            frozen = load_frozen_manifest(corpus, ROOT)
            _, metadata = select_classes(frozen, rule)
            selected, _ = deterministic_balanced_training_sample(
                frozen,
                metadata["selected_classes"],
                quota=metadata["count_cutoff"],
                seed=rule.seed,
            )
            self.assertEqual(len(selected), expected[corpus])


class NotebookIntegrationTests(unittest.TestCase):
    def test_foundation_notebook_uses_optimized_pipeline(self) -> None:
        notebook_path = ROOT / "notebooks" / "experimental" / "07_foundation_models.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            str(cell.get("source", "")) for cell in notebook["cells"]
        )

        self.assertIn(
            "from codeco.embeddings import run as run_embeddings",
            source,
        )
        self.assertIn(
            "from codeco.classify import run as run_classification",
            source,
        )
        self.assertIn("paths.tables", source)
        self.assertNotIn(
            "from classify_embeddings import run_classification",
            source,
        )


if __name__ == "__main__":
    unittest.main()
