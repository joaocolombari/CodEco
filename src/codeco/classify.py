"""Generic classification orchestration for cached foundation embeddings.

BirdNET and Perch share the original experimental protocols and are processed
in deterministic order while retaining separate result files.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from codeco.codeco_path import CodecoPath


MODEL_ORDER = ("birdnet", "perch")


@dataclass(frozen=True)
class ClassificationRun:
    datasets: tuple[str, ...] | None
    head: str
    seed: int
    checkpoint_path: Path | None
    force: bool
    early_stopping: bool
    plot_training_curves: bool
    embeddings_indices: dict[str, Path]
    artifact_directory: Path | None


@dataclass(frozen=True)
class ClassificationBackend:
    name: str
    run: Callable[[ClassificationRun], list[Path]]
    migrated: bool


def _run_birdnet(arguments: ClassificationRun) -> list[Path]:
    from codeco.classify_embeddings import run

    return run(
        datasets=arguments.datasets,
        head=arguments.head,
        seed=arguments.seed,
        embeddings_index=arguments.embeddings_indices["birdnet"],
        checkpoint_path=arguments.checkpoint_path,
        force=arguments.force,
        early_stopping=arguments.early_stopping,
        plot_training_curves=arguments.plot_training_curves,
        foundation_model="birdnet",
        artifact_directory=arguments.artifact_directory,
    )


def _run_perch(arguments: ClassificationRun) -> list[Path]:
    from codeco.classify_embeddings import run

    return run(
        datasets=arguments.datasets,
        head=arguments.head,
        seed=arguments.seed,
        embeddings_index=arguments.embeddings_indices["perch"],
        checkpoint_path=arguments.checkpoint_path,
        force=arguments.force,
        early_stopping=arguments.early_stopping,
        plot_training_curves=arguments.plot_training_curves,
        foundation_model="perch",
        artifact_directory=arguments.artifact_directory,
    )


BACKENDS = {
    "birdnet": ClassificationBackend("birdnet", _run_birdnet, migrated=True),
    "perch": ClassificationBackend("perch", _run_perch, migrated=True),
}


def _ordered_models(models: Sequence[str] | None) -> tuple[str, ...]:
    requested = set(models or MODEL_ORDER)
    unknown = requested.difference(BACKENDS)
    if unknown:
        raise ValueError(f"Modelos desconhecidos: {', '.join(sorted(unknown))}")
    return tuple(model for model in MODEL_ORDER if model in requested)


def run(
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    head: str = "mlp",
    seed: int = 42,
    embeddings_index: Path | None = None,
    checkpoint_path: Path | None = None,
    force: bool = False,
    early_stopping: bool = True,
    plot_training_curves: bool = False,
    embeddings_indices: dict[str, Path] | None = None,
) -> list[Path]:
    selected = _ordered_models(models)
    paths = CodecoPath.discover()
    standalone = embeddings_indices is None
    artifact_directory = paths.tests / "classify" if standalone else None
    if artifact_directory is not None:
        artifact_directory.mkdir(parents=True, exist_ok=True)
    if embeddings_indices is not None:
        selected_indices = {
            model: Path(embeddings_indices[model]) for model in selected
        }
    elif embeddings_index is not None:
        selected_indices = {model: embeddings_index for model in selected}
    else:
        selected_indices = {
            model: paths.tests / "embeddings" / f"{model}_embeddings.csv"
            for model in selected
        }
    selected_checkpoint = checkpoint_path or (
        artifact_directory / "checkpoints.sqlite3"
        if artifact_directory is not None
        else None
    )
    arguments = ClassificationRun(
        datasets=tuple(datasets) if datasets is not None else None,
        head=head,
        seed=seed,
        checkpoint_path=selected_checkpoint,
        force=force,
        early_stopping=early_stopping,
        plot_training_curves=plot_training_curves,
        embeddings_indices=selected_indices,
        artifact_directory=artifact_directory,
    )
    outputs = []
    for number, model in enumerate(selected, 1):
        backend = BACKENDS[model]
        print("\n" + "=" * 72)
        print(f"CLASSIFICACAO {number}/{len(selected)} | {model.upper()}")
        print("=" * 72)
        outputs.extend(backend.run(arguments))
    return outputs


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classifica embeddings de foundation models em ordem deterministica."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
        help="Modelos a processar; BirdNET sempre precede Perch.",
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--head", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embeddings-index", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--plot-training-curves", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    run(
        models=arguments.models,
        datasets=arguments.datasets,
        head=arguments.head,
        seed=arguments.seed,
        embeddings_index=arguments.embeddings_index,
        checkpoint_path=arguments.checkpoint,
        force=arguments.force,
        early_stopping=not arguments.no_early_stopping,
        plot_training_curves=arguments.plot_training_curves,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as error:
        print(f"Falha no pipeline de classificacao: {error}", file=sys.stderr)
        raise SystemExit(1) from None
