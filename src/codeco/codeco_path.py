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
        # Authoritative installers place both corpora below data/audio.  The
        # integration pipeline reads that frozen layout instead of maintaining
        # a second, incompatible dataset tree.
        return self.data / "audio"

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
        # The lock is the hash-addressable root of the versioned CSV manifests.
        return self.data / "manifest-lock-v1.json"

    @property
    def frozen_manifests(self) -> tuple[Path, ...]:
        return (
            self.data / "manifests" / "ecosoundset_manifest_v1.csv",
            self.data / "manifests" / "pteroset_ppa1_manifest_v1.csv",
        )

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
