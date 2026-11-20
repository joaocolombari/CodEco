"""Public entry points for the independent CodEco execution pipeline."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_PUBLIC_ENTRIES = {
    "Codeco": ("codeco.main", "Codeco"),
    "CodecoPath": ("codeco.codeco_path", "CodecoPath"),
    "BaselineState": ("codeco.prepare_table", "BaselineState"),
    "DownloadState": ("codeco.download_datasets", "DownloadState"),
    "EnvironmentState": ("codeco.environment", "EnvironmentState"),
    "FFmpegInstallation": ("codeco.environment", "FFmpegInstallation"),
    "GPUDevice": ("codeco.environment", "GPUDevice"),
    "SystemDependencyError": ("codeco.environment", "SystemDependencyError"),
    "SystemEnvironment": ("codeco.environment", "SystemEnvironment"),
    "VirtualEnvironmentError": ("codeco.environment", "VirtualEnvironmentError"),
    "download_datasets": ("codeco.download_datasets", "download_datasets"),
    "prepare_table": ("codeco.prepare_table", "prepare_table"),
    "prepare_baseline": ("codeco.prepare_table", "prepare_baseline"),
    "load_standalone_examples": (
        "codeco.prepare_table",
        "load_standalone_examples",
    ),
    "virtual_env": ("codeco.environment", "virtual_env"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _PUBLIC_ENTRIES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = list(_PUBLIC_ENTRIES)

