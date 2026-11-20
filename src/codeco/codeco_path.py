"""Centralized filesystem and executable locations for the experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodecoPath:
    """Filesystem locations owned by the experiment project."""

    root: Path

    @classmethod
    def discover(cls) -> "CodecoPath":
        return cls(Path(__file__).resolve().parents[2])

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
    def prepared(self) -> Path:
        return self.data / "prepared"

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
        return self.data / "samples"

    @property
    def tests(self) -> Path:
        return self.data / "tests"

    @property
    def cache(self) -> Path:
        return self.data / "cache"

    @property
    def baseline_pcm(self) -> Path:
        return self.cache / "baseline_pcm"

    @property
    def checkpoints(self) -> Path:
        return self.data / "checkpoints"

    @property
    def results(self) -> Path:
        return self.root / "results"

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
