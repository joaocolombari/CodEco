"""Fluent entry point for the CodEco execution pipeline."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from codeco.codeco_path import TEST_MODE_VARIABLE, CodecoPath, test_mode
from codeco.environment import virtual_env


# --test measures each codec on two repeated clips instead of twenty.
TEST_FLOPS_REPEATS = 2


class Codeco:
    """Execute pipeline stages in order while preserving shared state."""

    def __init__(self) -> None:
        self.context: dict[str, Any] = {}

    def environment(self) -> "Codeco":
        """Validate the environment and make it available to later stages."""
        state = virtual_env()
        self.context["environment"] = state
        return self

    def datasets(self) -> "Codeco":
        """Acquire and validate the frozen datasets for the pipeline."""
        if "environment" not in self.context:
            raise RuntimeError("environment() deve ser concluido antes de datasets().")
        from codeco.download_datasets import download_datasets

        state = download_datasets()
        self.context["datasets"] = state
        from codeco.provenance import write_run_metadata

        self.context["provenance"] = write_run_metadata(
            self.context.get("environment")
        )
        return self

    def prepare_baseline(self) -> "Codeco":
        """Prepare the canonical clip index used by all experiment stages."""
        if "datasets" not in self.context:
            raise RuntimeError(
                "datasets() deve ser concluido antes de prepare_baseline()."
            )
        if test_mode():
            from codeco.prepare_table import load_standalone_examples

            state = load_standalone_examples()
        else:
            from codeco.prepare_table import prepare_baseline

            state = prepare_baseline()
        self.context["baseline"] = state
        return self

    def flops(self) -> "Codeco":
        """Measure computational cost using the validated environment."""
        if "datasets" not in self.context:
            raise RuntimeError("datasets() deve ser concluido antes de flops().")
        from codeco.flops import run

        if test_mode():
            table = run(
                n_repeats=TEST_FLOPS_REPEATS, environment=self.context["environment"]
            )
        else:
            table = run(environment=self.context["environment"])
        self.context["flops"] = table
        return self

    def ffmpeg(self) -> "Codeco":
        """Run FFmpeg reconstruction and metrics from the canonical PCM baseline."""
        if "environment" not in self.context:
            raise RuntimeError("environment() deve ser concluido antes de ffmpeg().")
        if "baseline" not in self.context:
            raise RuntimeError("prepare_baseline() deve ser concluido antes de ffmpeg().")
        from codeco.codec_config import codec_evaluation_config
        from codeco.ffmpeg import run

        evaluation = codec_evaluation_config()
        output = run(
            baseline=self.context["baseline"],
            environment=self.context["environment"],
            max_clips_per_dataset=evaluation["max_clips_per_dataset"],
            seed=int(evaluation["seed"]),
            split=evaluation["split"],
        )
        self.context["ffmpeg"] = output
        return self

    def encodec(self) -> "Codeco":
        """Run multi-GPU EnCodec reconstruction from the canonical PCM baseline."""
        if "environment" not in self.context:
            raise RuntimeError("environment() deve ser concluido antes de encodec().")
        if "baseline" not in self.context:
            raise RuntimeError("prepare_baseline() deve ser concluido antes de encodec().")
        if "ffmpeg" not in self.context:
            raise RuntimeError("ffmpeg() deve ser concluido antes de encodec().")
        from codeco.codec_config import codec_evaluation_config
        from codeco.encodec import run

        evaluation = codec_evaluation_config()
        output = run(
            baseline=self.context["baseline"],
            environment=self.context["environment"],
            max_clips_per_dataset=evaluation["max_clips_per_dataset"],
            seed=int(evaluation["seed"]),
            split=evaluation["split"],
        )
        self.context["encodec"] = output
        return self

    def embeddings(self) -> "Codeco":
        """Generate BirdNET and Perch embeddings from the canonical baseline."""
        if "environment" not in self.context:
            raise RuntimeError("environment() deve ser concluido antes de embeddings().")
        if "baseline" not in self.context:
            raise RuntimeError(
                "prepare_baseline() deve ser concluido antes de embeddings()."
            )
        if "encodec" not in self.context:
            raise RuntimeError("encodec() deve ser concluido antes de embeddings().")
        from codeco.embeddings import run

        outputs = run(
            baseline=self.context["baseline"],
            environment=self.context["environment"],
            max_clips_per_dataset=None,
        )
        self.context["embeddings"] = dict(
            zip(("birdnet", "perch"), outputs, strict=True)
        )
        return self

    def classify(self) -> "Codeco":
        """Classify BirdNET and Perch embeddings with every configured head."""
        if "embeddings" not in self.context:
            raise RuntimeError("embeddings() deve ser concluido antes de classify().")
        from codeco.classify import classification_heads, run

        outputs = []
        for head in classification_heads():
            outputs.extend(run(head=head, embeddings_indices=self.context["embeddings"]))
        self.context["classification"] = outputs
        return self

    def analyze(self) -> "Codeco":
        """Analyze every reconstructed codec against the single PCM baseline."""
        if "ffmpeg" not in self.context:
            raise RuntimeError("ffmpeg() deve ser concluido antes de analyze().")
        if "encodec" not in self.context:
            raise RuntimeError("encodec() deve ser concluido antes de analyze().")
        if "classification" not in self.context:
            raise RuntimeError("classify() deve ser concluido antes de analyze().")
        from codeco.analyze import run

        outputs = run(
            ffmpeg_path=self.context["ffmpeg"],
            encodec_path=self.context["encodec"],
        )
        self.context["analysis"] = outputs
        return self

    def graphs(self) -> "Codeco":
        """Generate the scatter and rate-distortion figures from the analysis."""
        if "analysis" not in self.context:
            raise RuntimeError("analyze() deve ser concluido antes de graphs().")
        from codeco.graph import run

        self.context["graphs"] = run()
        return self


def _clean_generated_state(paths: CodecoPath) -> tuple[Path, ...]:
    """Remove generated pipeline state while preserving inputs and test artifacts."""
    targets = (
        paths.cache,
        paths.checkpoints,
        paths.prepared,
        paths.samples,
        paths.results,
    )
    generated = paths.tests / "pipeline" if test_mode() else paths.data
    allowed = {
        (generated / "cache").absolute(),
        (generated / "checkpoints").absolute(),
        (generated / "prepared").absolute(),
        (generated / "samples").absolute(),
        (generated / "results" if test_mode() else paths.root / "results").absolute(),
    }
    preserved = {
        paths.datasets.absolute(),
        paths.downloads.absolute(),
        paths.models.absolute(),
        paths.tests.absolute(),
    }

    line = "=" * 72
    print(f"\n{line}\nLIMPEZA DO PIPELINE\n{line}")
    removed = []
    for target in targets:
        absolute = target.absolute()
        if absolute not in allowed or absolute in preserved:
            raise RuntimeError(f"Alvo de limpeza nao autorizado: {target}")
        if target.is_symlink():
            raise RuntimeError(f"Link simbolico nao sera removido por --clean: {target}")
        if not target.exists():
            print(f"[AUSENTE]    {target}")
            continue
        if not target.is_dir():
            raise RuntimeError(f"Alvo de limpeza nao e um diretorio: {target}")
        shutil.rmtree(target)
        removed.append(target)
        print(f"[REMOVIDO]   {target}")

    print("\nPreservados:")
    for directory in (paths.datasets, paths.downloads, paths.models, paths.tests):
        print(f"[PRESERVADO] {directory}")
    print("\nLimpeza concluida; iniciando o pipeline.")
    return tuple(removed)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa o pipeline integrado do CodEco."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Remove caches e resultados gerados antes da execucao, preservando "
            "data/datasets, data/downloads, data/models e data/tests. "
            "Com --test, limpa apenas data/tests/pipeline."
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Executa o pipeline completo e rapido com os exemplos fixos de "
            "config/standalone-examples.json, gravando tudo em data/tests/pipeline."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Execute the first integrated pipeline stage."""
    arguments = _parse_arguments()
    if arguments.test:
        # Set before any path is resolved; spawned workers inherit it.
        os.environ[TEST_MODE_VARIABLE] = "1"
    if arguments.clean:
        _clean_generated_state(CodecoPath.discover())
    (
        Codeco()
        .environment()
        .datasets()
        .flops()
        .prepare_baseline()
        .ffmpeg()
        .encodec()
        .embeddings()
        .classify()
        .analyze()
        .graphs()
    )


if __name__ == "__main__":
    main()
