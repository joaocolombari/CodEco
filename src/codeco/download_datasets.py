"""Resumable, verified and multiprocessing dataset acquisition."""

from __future__ import annotations

import hashlib
import itertools
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable

import psutil
import pyarrow.parquet as parquet
from tqdm import tqdm

from codeco.codeco_path import CodecoPath


DATASET_NAMES = ("ecosoundset", "pteroset_ppa1")
REQUIRED_FREE_GIB = {
    ("ecosoundset",): 38,
    ("pteroset_ppa1",): 17,
    DATASET_NAMES: 55,
}
PARTS_PER_PROCESS = 4
MAX_ATTEMPTS = 3
READ_BUFFER_BYTES = 8 * 1024 * 1024
HTTP_TIMEOUT_SECONDS = 120
BASELINE_CACHE_REQUIRED_FREE_GIB = 15
LOGGER = logging.getLogger("codeco.download_datasets")


class TqdmLoggingHandler(logging.StreamHandler):
    """Write log records without corrupting active tqdm bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


def _configure_logging(paths: CodecoPath) -> tuple[Path, bool]:
    if LOGGER.handlers:
        return paths.logs / "download_datasets.log", False
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    terminal = TqdmLoggingHandler(sys.stdout)
    terminal.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(terminal)

    created = not paths.logs.is_dir()
    paths.logs.mkdir(parents=True, exist_ok=True)
    log_file = paths.logs / "download_datasets.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    LOGGER.addHandler(file_handler)
    return log_file, created


def _pipeline_header(name: str) -> None:
    line = "=" * 72
    script = str(Path(sys.argv[0]).resolve())
    command = subprocess.list2cmdline([sys.executable, script, *sys.argv[1:]])
    LOGGER.info("\n%s\n%s\n[CMD]   %s\n%s", line, name, command, line)


def _section(name: str) -> None:
    line = "-" * 72
    LOGGER.info("\n%s\n%s", name, line)


@dataclass(frozen=True)
class PartTask:
    dataset: str
    archive_name: str
    url: str
    start: int
    end: int
    path: str

    @property
    def expected_bytes(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class PartResult:
    task: PartTask
    reused: bool
    error: str | None = None


@dataclass(frozen=True)
class ProgressEvent:
    task: PartTask
    bytes_written: int


@dataclass(frozen=True)
class WorkerLog:
    level: int
    message: str


@dataclass(frozen=True)
class DownloadState:
    datasets: tuple[str, ...]
    manifest: Path
    downloads_directory: Path
    datasets_directory: Path
    processes: int
    downloaded_archives: int
    reused_archives: int
    downloaded_parts: int
    reused_parts: int
    extracted_files: int
    reused_datasets: tuple[str, ...]
    removed_archives: int
    removed_part_directories: int
    downloads_directory_removed: bool


def _physical_cores() -> int:
    return max(1, int(psutil.cpu_count(logical=False) or 1))


def _digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(READ_BUFFER_BYTES), b""):
            value.update(block)
    return value.hexdigest()


def _read_manifest(paths: CodecoPath) -> list[dict]:
    manifest = paths.manifest
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifesto de datasets nao encontrado: {manifest}")
    rows = parquet.read_table(
        manifest,
        filters=[
            [("resource_type", "=", "archive")],
            [("resource_type", "=", "member")],
        ],
    ).to_pylist()
    required = {
        "dataset",
        "resource_type",
        "archive_name",
        "url",
        "expected_bytes",
        "md5",
        "sha256",
        "member_name",
        "destination",
        "member_bytes",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Schema invalido em {manifest}")
    return rows


def _normalize_datasets(datasets: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(datasets))
    unknown = sorted(set(selected) - set(DATASET_NAMES))
    if unknown:
        raise ValueError(
            f"Datasets desconhecidos: {', '.join(unknown)}. "
            f"Opcoes: {', '.join(DATASET_NAMES)}"
        )
    if not selected:
        raise ValueError("Selecione pelo menos um dataset.")
    return selected


def _prepare_directories(paths: CodecoPath) -> None:
    for directory in (paths.datasets, paths.downloads):
        if directory.is_dir():
            LOGGER.info("Diretorio encontrado: %s", directory)
        else:
            LOGGER.info("Criando diretorio: %s", directory)
            directory.mkdir(parents=True, exist_ok=True)


def _required_free_gib(datasets: tuple[str, ...]) -> int:
    key = tuple(name for name in DATASET_NAMES if name in datasets)
    return REQUIRED_FREE_GIB[key]


def _check_disk(
    paths: CodecoPath,
    datasets: tuple[str, ...],
    rows: list[dict],
) -> tuple[int, int, int]:
    required_gib = _required_free_gib(datasets)
    free = shutil.disk_usage(paths.root).free
    required = required_gib * 1024**3
    stored = _stored_progress_bytes(paths, datasets, rows)
    remaining = max(0, required - stored)
    if free < remaining:
        raise RuntimeError(
            f"Espaco insuficiente: {remaining / 1024**3:.1f} GiB adicionais "
            f"necessarios apos considerar {stored / 1024**3:.1f} GiB ja armazenados; "
            f"{free / 1024**3:.1f} GiB disponiveis."
        )
    return stored, remaining, free


def _check_baseline_cache_disk(paths: CodecoPath) -> tuple[bool, int]:
    """Reserve disk capacity for the materialized PCM baseline after datasets."""
    from codeco.prepare_table import baseline_pcm_cache_status

    complete, detail = baseline_pcm_cache_status(paths)
    _section("Espaco para a baseline PCM")
    if complete:
        LOGGER.info("Cache PCM completo; reserva adicional dispensada: %s.", detail)
        return True, shutil.disk_usage(paths.root).free

    free = shutil.disk_usage(paths.root).free
    required = BASELINE_CACHE_REQUIRED_FREE_GIB * 1024**3
    LOGGER.info("Cache PCM precisa ser construido: %s.", detail)
    LOGGER.info(
        "Espaco exigido: %d GiB; disponivel: %.1f GiB.",
        BASELINE_CACHE_REQUIRED_FREE_GIB,
        free / 1024**3,
    )
    if free < required:
        raise RuntimeError(
            "Espaco insuficiente para a baseline PCM: "
            f"{BASELINE_CACHE_REQUIRED_FREE_GIB} GiB necessarios; "
            f"{free / 1024**3:.1f} GiB disponiveis em {paths.cache}."
        )
    LOGGER.info("Espaco para a baseline PCM confirmado.")
    return False, free


def _archive_rows(rows: list[dict], datasets: tuple[str, ...]) -> list[dict]:
    return [
        row
        for row in rows
        if row["dataset"] in datasets and row["resource_type"] == "archive"
    ]


def _member_rows(rows: list[dict], dataset: str) -> list[dict]:
    return [
        row
        for row in rows
        if row["dataset"] == dataset and row["resource_type"] == "member"
    ]


def _destination(paths: CodecoPath, row: dict) -> Path:
    return paths.datasets / str(row["destination"])


def _dataset_installed(paths: CodecoPath, members: list[dict]) -> bool:
    if not members:
        return False
    for row in members:
        destination = _destination(paths, row)
        if not destination.is_file():
            return False
        if destination.stat().st_size != int(row["member_bytes"]):
            return False
    return True


def _archive_members(
    members: list[dict],
    archive_name: str,
) -> list[dict]:
    return [
        row for row in members if str(row["archive_name"]) == archive_name
    ]


def _archive_path(paths: CodecoPath, row: dict) -> Path:
    return paths.downloads / str(row["dataset"]) / str(row["archive_name"])


def _remove_download_artifacts(
    paths: CodecoPath,
    row: dict,
) -> tuple[int, int]:
    archive = _archive_path(paths, row)
    removed_archive = 0
    removed_parts = 0
    if archive.is_file():
        LOGGER.info("Removendo arquivo compactado ja extraido: %s", archive)
        archive.unlink()
        removed_archive = 1

    assembling = archive.with_name(f"{archive.name}.assembling")
    if assembling.is_file():
        LOGGER.info("Removendo montagem temporaria desnecessaria: %s", assembling)
        assembling.unlink()

    parts = archive.with_name(f"{archive.name}.parts")
    if parts.is_dir():
        LOGGER.info("Removendo partes temporarias desnecessarias: %s", parts)
        shutil.rmtree(parts)
        removed_parts = 1
    return removed_archive, removed_parts


def _remove_empty_download_directories(
    paths: CodecoPath,
    datasets: tuple[str, ...],
) -> bool:
    for dataset in datasets:
        directory = paths.downloads / dataset
        try:
            directory.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            LOGGER.info("Diretorio de download ainda contem arquivos: %s", directory)
        else:
            LOGGER.info("Diretorio de download vazio removido: %s", directory)
    try:
        paths.downloads.rmdir()
    except FileNotFoundError:
        return True
    except OSError:
        LOGGER.info("Diretorio de downloads ainda contem arquivos: %s", paths.downloads)
        return False
    LOGGER.info("Diretorio de downloads vazio removido: %s", paths.downloads)
    return True


def _stored_progress_bytes(
    paths: CodecoPath,
    datasets: tuple[str, ...],
    rows: list[dict],
) -> int:
    stored = 0
    for dataset in datasets:
        dataset_required = _required_free_gib((dataset,)) * 1024**3
        members = _member_rows(rows, dataset)
        if _dataset_installed(paths, members):
            stored += dataset_required
            continue

        dataset_stored = 0
        destinations = set()
        for row in members:
            destination = _destination(paths, row)
            if destination in destinations:
                continue
            destinations.add(destination)
            expected = int(row["member_bytes"])
            if destination.is_file() and destination.stat().st_size == expected:
                dataset_stored += expected

        for row in _archive_rows(rows, (dataset,)):
            expected = int(row["expected_bytes"])
            archive = _archive_path(paths, row)
            archive_bytes = min(archive.stat().st_size, expected) if archive.is_file() else 0
            parts = archive.with_name(f"{archive.name}.parts")
            part_bytes = 0
            if parts.is_dir():
                part_bytes = min(
                    expected,
                    sum(
                        part.stat().st_size
                        for part in parts.iterdir()
                        if part.is_file() and part.suffix == ".part"
                    ),
                )
            dataset_stored += max(archive_bytes, part_bytes)

        stored += min(dataset_stored, dataset_required)
    return stored


def _archive_valid(path: Path, row: dict) -> bool:
    if not path.is_file() or path.stat().st_size != int(row["expected_bytes"]):
        return False
    expected_md5 = str(row["md5"] or "")
    expected_sha256 = str(row["sha256"] or "")
    if expected_md5 and _digest(path, "md5") != expected_md5:
        return False
    if expected_sha256 and _digest(path, "sha256") != expected_sha256:
        return False
    return True


def _part_tasks(paths: CodecoPath, row: dict, processes: int) -> list[PartTask]:
    archive = _archive_path(paths, row)
    part_directory = archive.with_name(f"{archive.name}.parts")
    if part_directory.is_dir():
        LOGGER.info("Partes temporarias encontradas: %s", part_directory)
    else:
        LOGGER.info("Criando diretorio temporario: %s", part_directory)
        part_directory.mkdir(parents=True, exist_ok=True)
    total_bytes = int(row["expected_bytes"])
    part_count = max(1, processes * PARTS_PER_PROCESS)
    part_bytes = (total_bytes + part_count - 1) // part_count
    tasks = []
    for index in range(part_count):
        start = index * part_bytes
        if start >= total_bytes:
            break
        end = min(total_bytes - 1, start + part_bytes - 1)
        part = part_directory / f"{start:020d}-{end:020d}.part"
        tasks.append(
            PartTask(
                dataset=str(row["dataset"]),
                archive_name=str(row["archive_name"]),
                url=str(row["url"]),
                start=start,
                end=end,
                path=str(part),
            )
        )
    return tasks


def _download_part(task: PartTask, events: mp.Queue) -> bool:
    path = Path(task.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    observed = path.stat().st_size if path.is_file() else 0
    if observed == task.expected_bytes:
        return True
    if observed > task.expected_bytes:
        path.unlink()
        observed = 0

    request_start = task.start + observed
    request = urllib.request.Request(
        task.url,
        headers={"Range": f"bytes={request_start}-{task.end}"},
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        if response.status != 206:
            raise RuntimeError(
                f"Servidor ignorou Range para {task.archive_name}: status {response.status}"
            )
        with path.open("ab") as target:
            while True:
                block = response.read(READ_BUFFER_BYTES)
                if not block:
                    break
                target.write(block)
                events.put(ProgressEvent(task=task, bytes_written=len(block)))

    final_size = path.stat().st_size
    if final_size != task.expected_bytes:
        raise RuntimeError(
            f"Parte incompleta de {task.archive_name}: "
            f"{final_size}/{task.expected_bytes} bytes"
        )
    return False


def _worker(tasks: mp.JoinableQueue, events: mp.Queue) -> None:
    while True:
        task = tasks.get()
        try:
            if task is None:
                return
            error = None
            reused = False
            for attempt in range(MAX_ATTEMPTS):
                try:
                    reused = _download_part(task, events)
                    break
                except (OSError, RuntimeError, urllib.error.URLError) as exc:
                    error = exc
                    if attempt + 1 < MAX_ATTEMPTS:
                        events.put(
                            WorkerLog(
                                level=logging.WARNING,
                                message=(
                                    f"Falha em {task.dataset}/{task.archive_name}, "
                                    f"tentativa {attempt + 1}/{MAX_ATTEMPTS}: {exc}. "
                                    "O download sera retomado."
                                ),
                            )
                        )
                        time.sleep(2**attempt)
            else:
                events.put(PartResult(task=task, reused=False, error=str(error)))
                continue
            events.put(PartResult(task=task, reused=reused))
        finally:
            tasks.task_done()


def _existing_part_bytes(task: PartTask) -> int:
    path = Path(task.path)
    if not path.is_file():
        return 0
    observed = path.stat().st_size
    if observed <= task.expected_bytes:
        return observed
    LOGGER.warning("Parte temporaria invalida sera reiniciada: %s", path)
    path.unlink()
    return 0


def _interleave_tasks(
    grouped_tasks: Iterable[list[PartTask]],
) -> list[PartTask]:
    groups = tuple(grouped_tasks)
    if not groups:
        return []
    missing = object()
    return [
        task
        for round_tasks in itertools.zip_longest(*groups, fillvalue=missing)
        for task in round_tasks
        if task is not missing
    ]


def _run_queue(
    tasks: list[PartTask],
    processes: int,
    archive_rows: list[dict],
) -> tuple[int, int]:
    if not tasks:
        LOGGER.info("Nenhum byte precisa ser baixado; arquivos locais serao reutilizados.")
        return 0, 0
    context = mp.get_context("spawn")
    task_queue = context.JoinableQueue()
    event_queue = context.Queue()
    workers = [
        context.Process(target=_worker, args=(task_queue, event_queue))
        for _ in range(processes)
    ]

    archive_tasks: dict[tuple[str, str], list[PartTask]] = {}
    for task in tasks:
        archive_tasks.setdefault((task.dataset, task.archive_name), []).append(task)
    archive_totals = {
        key: sum(task.expected_bytes for task in grouped)
        for key, grouped in archive_tasks.items()
    }
    archive_initial = {
        key: sum(_existing_part_bytes(task) for task in grouped)
        for key, grouped in archive_tasks.items()
    }
    dataset_totals: dict[str, int] = {}
    for row in archive_rows:
        dataset = str(row["dataset"])
        dataset_totals[dataset] = dataset_totals.get(dataset, 0) + int(row["expected_bytes"])
    dataset_queued = {
        dataset: sum(total for (name, _), total in archive_totals.items() if name == dataset)
        for dataset in dataset_totals
    }
    dataset_initial = {
        dataset: dataset_totals[dataset] - dataset_queued[dataset]
        + sum(value for (name, _), value in archive_initial.items() if name == dataset)
        for dataset in dataset_totals
    }

    LOGGER.info(
        "Iniciando fila unica com %d partes e %d processos.",
        len(tasks),
        processes,
    )
    for worker in workers:
        worker.start()
    print(file=sys.stdout)
    with tqdm(
        total=len(tasks),
        desc="Distribuindo tarefas",
        unit="tarefa",
        dynamic_ncols=True,
        file=sys.stdout,
    ) as distribution:
        for task in tasks:
            task_queue.put(task)
            distribution.update(1)
    print(file=sys.stdout)

    bars = []
    dataset_bars = {}
    archive_bars = {}
    position = 0
    for dataset_index, (dataset, total) in enumerate(dataset_totals.items()):
        if dataset_index:
            spacer = tqdm(
                total=0,
                bar_format=" ",
                position=position,
                file=sys.stdout,
            )
            bars.append(spacer)
            position += 1
        bar = tqdm(
            total=total,
            initial=dataset_initial[dataset],
            desc=f"Dataset: {dataset}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            position=position,
            file=sys.stdout,
        )
        dataset_bars[dataset] = bar
        bars.append(bar)
        position += 1
        for key, archive_total in archive_totals.items():
            if key[0] != dataset:
                continue
            archive_bar = tqdm(
                total=archive_total,
                initial=archive_initial[key],
                desc=f"  Arquivo: {key[1]}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                position=position,
                file=sys.stdout,
            )
            archive_bars[key] = archive_bar
            bars.append(archive_bar)
            position += 1

    downloaded = 0
    reused = 0
    failures = []
    completed = 0
    try:
        while completed < len(tasks):
            event = event_queue.get()
            if isinstance(event, ProgressEvent):
                key = (event.task.dataset, event.task.archive_name)
                archive_bars[key].update(event.bytes_written)
                dataset_bars[event.task.dataset].update(event.bytes_written)
            elif isinstance(event, WorkerLog):
                LOGGER.log(event.level, event.message)
            elif isinstance(event, PartResult):
                completed += 1
                if event.error:
                    failures.append(
                        f"{event.task.dataset}/{event.task.archive_name} "
                        f"[{event.task.start}-{event.task.end}]: {event.error}"
                    )
                elif event.reused:
                    reused += 1
                else:
                    downloaded += 1

        for _ in workers:
            task_queue.put(None)
        task_queue.join()
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        LOGGER.warning("Download interrompido; partes concluidas foram preservadas para retomada.")
        for worker in workers:
            worker.terminate()
        for worker in workers:
            worker.join()
        raise
    finally:
        for bar in reversed(bars):
            bar.close()

    if failures:
        raise RuntimeError("Falhas no download:\n" + "\n".join(failures))
    LOGGER.info(
        "Fila concluida: %d partes baixadas; %d partes reutilizadas.",
        downloaded,
        reused,
    )
    return downloaded, reused


def _assemble_archive(paths: CodecoPath, row: dict, tasks: list[PartTask]) -> Path:
    archive = _archive_path(paths, row)
    archive.parent.mkdir(parents=True, exist_ok=True)
    assembling = archive.with_name(f"{archive.name}.assembling")
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    total = 0
    with assembling.open("wb") as target:
        for task in sorted(tasks, key=lambda item: item.start):
            part = Path(task.path)
            if not part.is_file() or part.stat().st_size != task.expected_bytes:
                raise RuntimeError(f"Parte ausente ou incompleta: {part}")
            with part.open("rb") as source:
                while True:
                    block = source.read(READ_BUFFER_BYTES)
                    if not block:
                        break
                    target.write(block)
                    md5.update(block)
                    sha256.update(block)
                    total += len(block)

    if total != int(row["expected_bytes"]):
        raise RuntimeError(f"Tamanho invalido apos montagem de {archive.name}: {total}")
    if row["md5"] and md5.hexdigest() != str(row["md5"]):
        raise RuntimeError(f"MD5 invalido apos montagem de {archive.name}")
    if row["sha256"] and sha256.hexdigest() != str(row["sha256"]):
        raise RuntimeError(f"SHA-256 invalido apos montagem de {archive.name}")
    os.replace(assembling, archive)
    shutil.rmtree(Path(tasks[0].path).parent)
    return archive


def _copy_verified(
    source,
    destination: Path,
    row: dict,
    progress: Callable[[int], None] | None = None,
) -> bool:
    expected_bytes = int(row["member_bytes"])
    if destination.is_file() and destination.stat().st_size == expected_bytes:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    digest = hashlib.sha256()
    written = 0
    with temporary.open("wb") as target:
        while True:
            block = source.read(READ_BUFFER_BYTES)
            if not block:
                break
            target.write(block)
            digest.update(block)
            written += len(block)
            if progress is not None:
                progress(len(block))
    if written != expected_bytes:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Tamanho invalido ao extrair {row['member_name']}: {written}/{expected_bytes}"
        )
    if row["sha256"] and digest.hexdigest() != str(row["sha256"]):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 invalido ao extrair {row['member_name']}")
    os.replace(temporary, destination)
    return True


def _extract_tar(
    paths: CodecoPath,
    archive: Path,
    members: list[dict],
    progress: Callable[[int], None] | None = None,
) -> int:
    by_name = {str(row["member_name"]): row for row in members}
    remaining = {
        name
        for name, row in by_name.items()
        if not (
            _destination(paths, row).is_file()
            and _destination(paths, row).stat().st_size == int(row["member_bytes"])
        )
    }
    extracted = 0
    with tarfile.open(archive, mode="r:gz") as source_archive:
        for member in source_archive:
            name = Path(member.name).name
            if name not in remaining or not member.isfile():
                continue
            source = source_archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Nao foi possivel ler {member.name} em {archive}")
            with source:
                if _copy_verified(
                    source,
                    _destination(paths, by_name[name]),
                    by_name[name],
                    progress,
                ):
                    extracted += 1
            remaining.remove(name)
    if remaining:
        preview = ", ".join(sorted(remaining)[:10])
        raise RuntimeError(
            f"{len(remaining)} arquivos nao encontrados em {archive.name}: {preview}"
        )
    return extracted


def _extract_zip(
    paths: CodecoPath,
    archive: Path,
    members: list[dict],
    progress: Callable[[int], None] | None = None,
) -> int:
    extracted = 0
    with zipfile.ZipFile(archive) as source_archive:
        available = set(source_archive.namelist())
        for row in members:
            member_name = str(row["member_name"])
            if member_name not in available:
                raise RuntimeError(f"Arquivo ausente em {archive.name}: {member_name}")
            destination = _destination(paths, row)
            if destination.is_file() and destination.stat().st_size == int(row["member_bytes"]):
                continue
            with source_archive.open(member_name) as source:
                if _copy_verified(source, destination, row, progress):
                    extracted += 1
    return extracted


def _extract_archive(
    paths: CodecoPath,
    row: dict,
    members: list[dict],
    progress: Callable[[int], None] | None = None,
) -> int:
    archive = _archive_path(paths, row)
    archive_members = [
        member for member in members if member["archive_name"] == row["archive_name"]
    ]
    if archive.suffix == ".zip":
        return _extract_zip(paths, archive, archive_members, progress)
    if archive.name.endswith(".tar.gz"):
        return _extract_tar(paths, archive, archive_members, progress)
    raise ValueError(f"Formato de arquivo nao suportado: {archive}")


def _completed_member_bytes(paths: CodecoPath, members: list[dict]) -> int:
    completed = 0
    for row in members:
        destination = _destination(paths, row)
        expected = int(row["member_bytes"])
        if destination.is_file() and destination.stat().st_size == expected:
            completed += expected
    return completed


def _extract_archives(
    paths: CodecoPath,
    archive_rows: list[dict],
    members_by_dataset: dict[str, list[dict]],
    processes: int,
) -> int:
    if not archive_rows:
        LOGGER.info("Nenhum arquivo precisa ser extraido.")
        return 0

    archive_members = {}
    archive_totals = {}
    archive_initial = {}
    dataset_totals = {}
    dataset_initial = {}
    dataset_order = tuple(dict.fromkeys(str(row["dataset"]) for row in archive_rows))

    for dataset in dataset_order:
        members = members_by_dataset[dataset]
        dataset_totals[dataset] = sum(int(row["member_bytes"]) for row in members)
        dataset_initial[dataset] = _completed_member_bytes(paths, members)

    for row in archive_rows:
        key = (str(row["dataset"]), str(row["archive_name"]))
        members = [
            member
            for member in members_by_dataset[key[0]]
            if str(member["archive_name"]) == key[1]
        ]
        archive_members[key] = members
        archive_totals[key] = sum(int(member["member_bytes"]) for member in members)
        archive_initial[key] = _completed_member_bytes(paths, members)

    print(file=sys.stdout)
    bars = []
    dataset_bars = {}
    archive_bars = {}
    position = 0
    for dataset_index, dataset in enumerate(dataset_order):
        if dataset_index:
            spacer = tqdm(
                total=0,
                bar_format=" ",
                position=position,
                file=sys.stdout,
            )
            bars.append(spacer)
            position += 1
        dataset_bar = tqdm(
            total=dataset_totals[dataset],
            initial=dataset_initial[dataset],
            desc=f"Dataset: {dataset}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            position=position,
            file=sys.stdout,
        )
        dataset_bars[dataset] = dataset_bar
        bars.append(dataset_bar)
        position += 1
        for row in archive_rows:
            key = (str(row["dataset"]), str(row["archive_name"]))
            if key[0] != dataset:
                continue
            archive_bar = tqdm(
                total=archive_totals[key],
                initial=archive_initial[key],
                desc=f"  Arquivo: {key[1]}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                position=position,
                file=sys.stdout,
            )
            archive_bars[key] = archive_bar
            bars.append(archive_bar)
            position += 1

    progress_lock = Lock()
    extracted_files = 0
    workers = min(processes, len(archive_rows))
    LOGGER.info(
        "Extraindo %d arquivos compactados em paralelo com %d trabalhadores.",
        len(archive_rows),
        workers,
    )
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="extract") as executor:
            futures = {}
            for row in archive_rows:
                dataset = str(row["dataset"])
                key = (dataset, str(row["archive_name"]))

                def update(bytes_written: int, *, dataset=dataset, key=key) -> None:
                    with progress_lock:
                        archive_bars[key].update(bytes_written)
                        dataset_bars[dataset].update(bytes_written)

                future = executor.submit(
                    _extract_archive,
                    paths,
                    row,
                    archive_members[key],
                    update,
                )
                futures[future] = row

            for future in as_completed(futures):
                row = futures[future]
                extracted = future.result()
                extracted_files += extracted
                LOGGER.info(
                    "Extracao concluida: %s (%d arquivos novos).",
                    row["archive_name"],
                    extracted,
                )
    finally:
        for bar in reversed(bars):
            bar.close()
    print(file=sys.stdout)
    return extracted_files


def download_datasets(
    datasets: Iterable[str] = DATASET_NAMES,
) -> DownloadState:
    """Download, verify and extract the frozen datasets without loading audio."""
    selected = _normalize_datasets(datasets)
    paths = CodecoPath.discover()
    log_file, logs_created = _configure_logging(paths)
    _pipeline_header("DOWNLOAD DATASETS")
    _section("Registro da execucao")
    LOGGER.info(
        "%s: %s",
        "Diretorio de logs criado" if logs_created else "Diretorio de logs encontrado",
        paths.logs,
    )
    LOGGER.info("Log do download: %s", log_file)
    LOGGER.info("Iniciando etapa de download dos datasets: %s", ", ".join(selected))
    LOGGER.info(
        "Downloads serao preservados durante a retomada e removidos somente "
        "apos a extracao validada."
    )

    _section("Diretorios do download")
    LOGGER.info("Verificando os diretorios usados pelo download...")
    _prepare_directories(paths)

    _section("Manifesto dos datasets")
    LOGGER.info("Lendo manifesto: %s", paths.manifest)
    rows = _read_manifest(paths)
    LOGGER.info("Manifesto carregado: %d registros.", len(rows))

    _section("Espaco em disco")
    LOGGER.info("Verificando espaco livre e progresso ja armazenado...")
    stored, remaining, free = _check_disk(paths, selected, rows)
    LOGGER.info("Progresso armazenado: %.1f GiB.", stored / 1024**3)
    LOGGER.info(
        "Espaco adicional necessario: %.1f GiB; disponivel: %.1f GiB.",
        remaining / 1024**3,
        free / 1024**3,
    )

    processes = _physical_cores()
    LOGGER.info("Nucleos fisicos detectados; processos de download: %d", processes)

    reused_datasets = []
    pending_archives = []
    members_by_dataset = {}
    reused_archives = 0
    removed_archives = 0
    removed_part_directories = 0
    archive_tasks: dict[tuple[str, str], list[PartTask]] = {}

    _section("Datasets, arquivos locais e checkpoints")
    LOGGER.info("Verificando datasets, arquivos locais e checkpoints...")
    for dataset in selected:
        LOGGER.info("Verificando dataset: %s", dataset)
        members = _member_rows(rows, dataset)
        members_by_dataset[dataset] = members
        if _dataset_installed(paths, members):
            LOGGER.info(
                "Dataset ja esta completo: %s (%d arquivos) em %s",
                dataset,
                len(members),
                paths.datasets,
            )
            reused_datasets.append(dataset)
            for archive_row in _archive_rows(rows, (dataset,)):
                removed_archive, removed_parts = _remove_download_artifacts(
                    paths,
                    archive_row,
                )
                removed_archives += removed_archive
                removed_part_directories += removed_parts
            continue
        LOGGER.info(
            "Dataset incompleto; %d arquivos serao conferidos e obtidos: %s",
            len(members),
            dataset,
        )
        for archive_row in _archive_rows(rows, (dataset,)):
            archive_name = str(archive_row["archive_name"])
            members = _archive_members(members_by_dataset[dataset], archive_name)
            if _dataset_installed(paths, members):
                LOGGER.info(
                    "Conteudo do arquivo ja esta completamente extraido: %s",
                    archive_name,
                )
                removed_archive, removed_parts = _remove_download_artifacts(
                    paths,
                    archive_row,
                )
                removed_archives += removed_archive
                removed_part_directories += removed_parts
                continue
            archive = _archive_path(paths, archive_row)
            LOGGER.info(
                "Verificando arquivo do dataset %s: %s",
                dataset,
                archive_row["archive_name"],
            )
            if _archive_valid(archive, archive_row):
                LOGGER.info("Arquivo verificado e reutilizado: %s", archive)
                reused_archives += 1
                pending_archives.append(archive_row)
                continue
            if archive.exists():
                LOGGER.warning("Arquivo local invalido; removendo para baixar novamente: %s", archive)
                archive.unlink()
            tasks = _part_tasks(paths, archive_row, processes)
            resumed_bytes = sum(_existing_part_bytes(task) for task in tasks)
            if resumed_bytes:
                LOGGER.info(
                    "Retomando %s com %.2f GiB ja baixados.",
                    archive_row["archive_name"],
                    resumed_bytes / 1024**3,
                )
            else:
                LOGGER.info(
                    "Preparando download de %s: %.2f GiB em %d partes.",
                    archive_row["archive_name"],
                    int(archive_row["expected_bytes"]) / 1024**3,
                    len(tasks),
                )
            archive_tasks[(dataset, str(archive_row["archive_name"]))] = tasks
            pending_archives.append(archive_row)

    queued_tasks = _interleave_tasks(archive_tasks.values())
    if queued_tasks:
        LOGGER.info(
            "Fila intercalada entre %d arquivos: %d partes.",
            len(archive_tasks),
            len(queued_tasks),
        )

    _section("Downloads pendentes")
    LOGGER.info("Executando os downloads pendentes...")
    downloaded_parts, reused_parts = _run_queue(
        queued_tasks,
        processes,
        pending_archives,
    )

    _section("Montagem e validacao dos arquivos")
    LOGGER.info("Montando e validando os arquivos baixados...")
    downloaded_archives = 0
    for row in pending_archives:
        key = (str(row["dataset"]), str(row["archive_name"]))
        tasks = archive_tasks.get(key)
        if tasks:
            LOGGER.info("Montando e validando arquivo: %s", row["archive_name"])
            archive = _assemble_archive(paths, row, tasks)
            LOGGER.info("Arquivo montado e validado: %s", archive)
            downloaded_archives += 1

    _section("Extracao dos datasets")
    LOGGER.info("Extraindo os datasets...")
    extracted_files = _extract_archives(
        paths,
        pending_archives,
        members_by_dataset,
        processes,
    )

    _section("Verificacao final")
    LOGGER.info("Verificando os datasets finalizados...")
    for dataset in selected:
        if dataset in reused_datasets:
            continue
        LOGGER.info("Verificando o dataset finalizado: %s", dataset)
        if not _dataset_installed(paths, members_by_dataset[dataset]):
            raise RuntimeError(f"Dataset incompleto apos extracao: {dataset}")
        LOGGER.info(
            "Dataset finalizado: %s (%d arquivos) em %s",
            dataset,
            len(members_by_dataset[dataset]),
            paths.datasets,
        )
        for row in _archive_rows(rows, (dataset,)):
            members = _archive_members(
                members_by_dataset[dataset],
                str(row["archive_name"]),
            )
            if not _dataset_installed(paths, members):
                raise RuntimeError(
                    f"Extracao incompleta para o arquivo: {row['archive_name']}"
                )
            removed_archive, removed_parts = _remove_download_artifacts(paths, row)
            removed_archives += removed_archive
            removed_part_directories += removed_parts

    downloads_directory_removed = _remove_empty_download_directories(paths, selected)
    _check_baseline_cache_disk(paths)

    state = DownloadState(
        datasets=selected,
        manifest=paths.manifest,
        downloads_directory=paths.downloads,
        datasets_directory=paths.datasets,
        processes=processes,
        downloaded_archives=downloaded_archives,
        reused_archives=reused_archives,
        downloaded_parts=downloaded_parts,
        reused_parts=reused_parts,
        extracted_files=extracted_files,
        reused_datasets=tuple(reused_datasets),
        removed_archives=removed_archives,
        removed_part_directories=removed_part_directories,
        downloads_directory_removed=downloads_directory_removed,
    )
    LOGGER.info("Etapa de download dos datasets concluida.")
    LOGGER.info("Estado final: %s", state)
    return state


if __name__ == "__main__":
    try:
        state = download_datasets()
        print(state)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception:
        LOGGER.exception("Falha na etapa de download dos datasets.")
        raise SystemExit(1) from None
