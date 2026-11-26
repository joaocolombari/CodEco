"""Centralized filesystem and executable locations for the experiment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


TEST_MODE_VARIABLE = "CODECO_TEST_MODE"


def test_mode() -> bool:
    """True when ``python -m codeco.main --test`` is running (inherited by child processes)."""
    return os.environ.get(TEST_MODE_VARIABLE) == "1"


@dataclass(frozen=True)
class CodecoPath:
    """Filesystem locations owned by the experiment project."""

    root: Path

    @classmethod
    def discover(cls) -> "CodecoPath":
        return cls(Path(__file__).resolve().parents[2])

    def project_relative(self, path: str | Path) -> str:
        """Serialize a project-owned path independently of the checkout location."""
        candidate = Path(path)
        absolute = candidate if candidate.is_absolute() else self.root / candidate
        try:
            relative = absolute.resolve().relative_to(self.root.resolve())
        except ValueError as error:
            raise ValueError(f"Caminho fora da raiz do projeto: {path}") from error
        return relative.as_posix()

    def project_path(self, path: str | Path) -> Path:
        """Resolve one persisted project-relative path against this checkout."""
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Caminho de projeto deve ser relativo: {path}")
        resolved = (self.root / relative).resolve()
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError as error:
            raise ValueError(f"Caminho fora da raiz do projeto: {path}") from error
        return resolved

    @property
    def src(self) -> Path:
        return self.root / "src"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def datasets(self) -> Path:
        return self.data / "datasets"

    @property
    def models(self) -> Path:
        return self.data / "models"

    @property
    def generated(self) -> Path:
        """Parent of every generated directory; isolated under data/tests in test mode."""
        return self.tests / "pipeline" if test_mode() else self.data

    @property
    def prepared(self) -> Path:
        return self.generated / "prepared"

    @property
    def downloads(self) -> Path:
        return self.data / "downloads"

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def manifest(self) -> Path:
        return self.config / "manifests.parquet"

    @property
    def standalone_examples(self) -> Path:
        return self.config / "standalone-examples.json"

    @property
    def samples(self) -> Path:
        return self.generated / "samples"

    @property
    def tests(self) -> Path:
        return self.data / "tests"

    @property
    def cache(self) -> Path:
        return self.generated / "cache"

    @property
    def baseline_pcm(self) -> Path:
        return self.cache / "baseline_pcm"

    @property
    def checkpoints(self) -> Path:
        return self.generated / "checkpoints"

    @property
    def results(self) -> Path:
        return self.generated / "results" if test_mode() else self.root / "results"

    @property
    def logs(self) -> Path:
        return self.results / "logs"

    @property
    def tables(self) -> Path:
        return self.results / "tables"

    @property
    def benchmarks(self) -> Path:
        return self.results / "benchmarks"

    @property
    def graphs(self) -> Path:
        return self.results / "graphs"

    @property
    def reports(self) -> Path:
        return self.results / "reports"
