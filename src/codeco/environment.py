"""System preflight and isolated Python environment management."""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

from codeco.codeco_path import CodecoPath
from codeco.protocol import PROTOCOL


RECOMMENDED_FFMPEG_VERSION = "9.0.2"
REQUIRED_ENCODERS = frozenset({"aac", "flac", "libmp3lame", "libopus"})
REQUIRED_DECODERS = frozenset()
LOGGER = logging.getLogger("codeco.environment")


def _configure_logging(paths: CodecoPath) -> tuple[Path, bool]:
    if LOGGER.handlers:
        return paths.logs / "environment.log", False
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    terminal = logging.StreamHandler(sys.stdout)
    terminal.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(terminal)

    created = not paths.logs.is_dir()
    paths.logs.mkdir(parents=True, exist_ok=True)
    log_file = paths.logs / "environment.log"
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


class SystemDependencyError(RuntimeError):
    """A required system dependency is missing or incompatible."""


class VirtualEnvironmentError(RuntimeError):
    """The project virtual environment is not active."""


@dataclass(frozen=True)
class FFmpegInstallation:
    executable: Path
    version: str
    lc3_executable: Path
    lc3_version: str
    encoders: frozenset[str]
    decoders: frozenset[str]


@dataclass(frozen=True)
class GPUDevice:
    """One CUDA device available to pipeline workers."""

    index: int
    name: str
    total_memory_bytes: int


@dataclass(frozen=True)
class SystemEnvironment:
    operating_system: str
    python_version: str
    uv: Path
    uv_version: str
    ffmpeg: FFmpegInstallation
    cuda_available: bool
    device: str
    gpu_devices: tuple[GPUDevice, ...]
    tensorflow_gpu_count: int


@dataclass(frozen=True)
class EnvironmentState:
    paths: CodecoPath
    virtual_environment: Path
    python: Path
    system: SystemEnvironment
    variables: dict[str, str | None]


def _installation_request(operating_system: str, detail: str) -> SystemDependencyError:
    if operating_system == "Windows":
        instruction = (
            f"Instale FFmpeg {RECOMMENDED_FFMPEG_VERSION} para Windows com AAC, FLAC, "
            "LAME, Opus e Google liblc3 e adicione o diretorio do executavel ao PATH."
        )
    elif operating_system == "Darwin":
        instruction = (
            f"Instale FFmpeg {RECOMMENDED_FFMPEG_VERSION} para macOS com AAC, FLAC, "
            "LAME, Opus e Google liblc3, ou execute scripts/install_ffmpeg_lc3.sh."
        )
    else:
        instruction = (
            f"Instale FFmpeg {RECOMMENDED_FFMPEG_VERSION} para Linux com AAC, FLAC, "
            "LAME, Opus e Google liblc3 e disponibilize o executavel no PATH."
        )
    return SystemDependencyError(f"{detail} {instruction}")


def _supported_operating_system() -> str:
    observed = platform.system()
    if observed not in {"Darwin", "Linux", "Windows"}:
        raise SystemDependencyError(
            f"Sistema operacional nao suportado: {observed}."
        )
    return observed


def _required_uv_version(paths: CodecoPath) -> str:
    pyproject = paths.root / "pyproject.toml"
    try:
        configuration = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        version = configuration["tool"]["codeco"]["environment"]["uv-version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        raise SystemDependencyError(
            f"Nao foi possivel ler tool.codeco.environment.uv-version em {pyproject}: {exc}"
        ) from exc
    if not isinstance(version, str) or not version.strip():
        raise SystemDependencyError(
            f"tool.codeco.environment.uv-version deve ser uma string em {pyproject}."
        )
    return version.strip()


def _uv_installation_instruction(version: str) -> str:
    return (
        f"Linux: curl -LsSf https://astral.sh/uv/{version}/install.sh | sh\n"
        "Windows PowerShell: "
        f'powershell -ExecutionPolicy ByPass -c "irm '
        f"https://astral.sh/uv/{version}/install.ps1 | iex\""
    )


def _uv_executable(paths: CodecoPath) -> tuple[Path, str]:
    required_version = _required_uv_version(paths)
    executable_name = "uv.exe" if platform.system() == "Windows" else "uv"
    virtual_environment_bin = (
        "Scripts" if platform.system() == "Windows" else "bin"
    )
    candidates = [
        shutil.which("uv"),
        str(
            paths.root
            / ".venv"
            / virtual_environment_bin
            / executable_name
        ),
        str(Path.home() / ".local" / "bin" / executable_name),
    ]
    candidate = next(
        (
            Path(value).expanduser()
            for value in dict.fromkeys(candidates)
            if value is not None and Path(value).expanduser().is_file()
        ),
        None,
    )
    if candidate is None:
        raise SystemDependencyError(
            f"uv {required_version} nao foi encontrado no PATH, em .venv ou em "
            "~/.local/bin. Instale a versao fixada pelo projeto e tente novamente.\n"
            f"{_uv_installation_instruction(required_version)}"
        )
    executable = candidate.resolve()
    try:
        process = subprocess.run(
            [str(executable), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemDependencyError(
            f"Nao foi possivel consultar a versao de {executable}: {exc}"
        ) from exc
    match = re.fullmatch(r"uv\s+([^\s]+)(?:\s+.*)?", process.stdout.strip())
    if match is None:
        raise SystemDependencyError(
            f"Nao foi possivel interpretar a versao retornada por {executable}: "
            f"{process.stdout.strip()!r}"
        )
    observed_version = match.group(1)
    def version_tuple(value: str) -> tuple[int, ...]:
        numbers = re.match(r"(\d+(?:\.\d+)*)", value)
        return tuple(int(part) for part in numbers.group(1).split(".")) if numbers else ()

    if version_tuple(observed_version) < version_tuple(required_version):
        raise SystemDependencyError(
            f"uv {observed_version} encontrado em {executable}; o projeto requer "
            f"uv >= {required_version}.\n{_uv_installation_instruction(required_version)}"
        )
    return executable, observed_version


def _ffmpeg_output(executable: Path, option: str) -> str:
    try:
        process = subprocess.run(
            [str(executable), "-hide_banner", option],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemDependencyError(
            f"Nao foi possivel consultar {option} em {executable}: {exc}"
        ) from exc
    return process.stdout


def _ffmpeg_version(
    executable: Path,
    operating_system: str,
    required_version: str | None = None,
) -> str:
    try:
        process = subprocess.run(
            [str(executable), "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise _installation_request(
            operating_system,
            f"A instalacao FFmpeg em {executable} nao pode ser executada: {exc}.",
        ) from exc
    first_line = process.stdout.splitlines()[0] if process.stdout else ""
    match = re.search(r"ffmpeg version n?([^\s]+)", first_line)
    if match is None:
        raise _installation_request(
            operating_system,
            f"Nao foi possivel identificar a versao do FFmpeg em {executable}.",
        )
    version = match.group(1)
    if required_version is not None and version != required_version:
        raise _installation_request(
            operating_system,
            f"FFmpeg {version} encontrado em {executable}; esta instalacao requer "
            f"a versao fixada {required_version}.",
        )
    return version


def _available_names(output: str, required: frozenset[str]) -> frozenset[str]:
    return frozenset(
        name for name in required if re.search(rf"\b{re.escape(name)}\b", output) is not None
    )


def _round_trip(
    executable: Path,
    extension: str,
    encoder_arguments: tuple[str, ...],
    operating_system: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="codeco-ffmpeg-") as directory:
        temporary = Path(directory)
        encoded = temporary / f"encoded.{extension}"
        decoded = temporary / "decoded.wav"
        commands = (
            [
                str(executable),
                "-v", "error",
                "-f", "lavfi",
                "-i", f"sine=frequency=1000:sample_rate={PROTOCOL.sample_rate_hz}:duration=0.1",
                "-ac", "1",
                "-ar", str(PROTOCOL.sample_rate_hz),
                *encoder_arguments,
                "-y", str(encoded),
            ],
            [
                str(executable),
                "-v", "error",
                "-i", str(encoded),
                "-ac", "1",
                "-ar", str(PROTOCOL.sample_rate_hz),
                "-y", str(decoded),
            ],
        )
        for command in commands:
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                stderr = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
                raise _installation_request(
                    operating_system,
                    f"O FFmpeg falhou no round-trip de {extension}: {stderr}.",
                ) from exc
        if not decoded.is_file() or decoded.stat().st_size == 0:
            raise _installation_request(
                operating_system,
                f"O FFmpeg nao produziu audio no round-trip de {extension}.",
            )


def _validate_round_trips(
    executable: Path,
    operating_system: str,
    include_lc3: bool = False,
) -> None:
    if include_lc3:
        conditions = [(
            "lc3",
            (
                "-c:a", "liblc3",
                "-b:a", "32k",
                "-frame_duration", "10",
                "-high_resolution", "0",
            ),
        )]
    else:
        conditions = [
            ("flac", ("-c:a", "flac", "-compression_level", "5")),
            ("mp3", ("-c:a", "libmp3lame", "-b:a", "32k")),
            ("m4a", ("-c:a", "aac", "-profile:a", "aac_low", "-b:a", "64k")),
            ("opus", ("-c:a", "libopus", "-b:a", "12k", "-vbr", "on", "-application", "audio")),
        ]
    for extension, arguments in conditions:
        LOGGER.info("Testando codificacao e decodificacao FFmpeg: %s", extension)
        _round_trip(executable, extension, arguments, operating_system)
        LOGGER.info("Teste FFmpeg concluido: %s", extension)


def _ffmpeg_installation(operating_system: str) -> FFmpegInstallation:
    paths = CodecoPath.discover()
    discovered = shutil.which("ffmpeg")
    if discovered is None:
        raise _installation_request(operating_system, "FFmpeg nao foi encontrado no PATH.")
    executable = Path(discovered).resolve()
    version = _ffmpeg_version(executable, operating_system)

    local_candidates = (
        paths.root / ".tools" / "ffmpeg-lc3" / RECOMMENDED_FFMPEG_VERSION / "bin" / "ffmpeg",
        Path.home() / ".local" / "ffmpeg-lc3" / RECOMMENDED_FFMPEG_VERSION / "bin" / "ffmpeg",
    )
    lc3_executable = None
    lc3_version = None
    for candidate in local_candidates:
        if not candidate.is_file():
            continue
        try:
            candidate_version = _ffmpeg_version(
                candidate,
                operating_system,
                required_version=RECOMMENDED_FFMPEG_VERSION,
            )
            encoders = _ffmpeg_output(candidate, "-encoders")
            decoders = _ffmpeg_output(candidate, "-decoders")
        except SystemDependencyError:
            continue
        if "liblc3" in encoders and "liblc3" in decoders:
            lc3_executable = candidate.resolve()
            lc3_version = candidate_version
            break
    if lc3_executable is None or lc3_version is None:
        raise _installation_request(
            operating_system,
            "FFmpeg local com encoder/decoder liblc3 nao foi encontrado.",
        )
    encoders = _available_names(
        _ffmpeg_output(executable, "-encoders"),
        REQUIRED_ENCODERS,
    )
    decoders = _available_names(_ffmpeg_output(executable, "-decoders"), REQUIRED_DECODERS)
    missing_encoders = sorted(REQUIRED_ENCODERS - encoders)
    missing_decoders = sorted(REQUIRED_DECODERS - decoders)
    if missing_encoders or missing_decoders:
        details = []
        if missing_encoders:
            details.append(f"encoders ausentes: {', '.join(missing_encoders)}")
        if missing_decoders:
            details.append(f"decoders ausentes: {', '.join(missing_decoders)}")
        raise _installation_request(
            operating_system,
            f"A instalacao FFmpeg em {executable} esta incompleta "
            f"({'; '.join(details)}).",
        )
    _validate_round_trips(executable, operating_system)
    _validate_round_trips(lc3_executable, operating_system, include_lc3=True)
    return FFmpegInstallation(
        executable=executable,
        version=version,
        lc3_executable=lc3_executable,
        lc3_version=lc3_version,
        encoders=encoders | frozenset({"liblc3"}),
        decoders=decoders | frozenset({"liblc3"}),
    )


def _cuda() -> tuple[bool, str, tuple[GPUDevice, ...]]:
    try:
        import torch
    except ImportError as exc:
        raise SystemDependencyError(
            "PyTorch nao esta instalado no ambiente virtual."
        ) from exc
    available = bool(torch.cuda.is_available())
    devices = tuple(
        GPUDevice(
            index=index,
            name=torch.cuda.get_device_name(index),
            total_memory_bytes=int(torch.cuda.get_device_properties(index).total_memory),
        )
        for index in range(torch.cuda.device_count() if available else 0)
    )
    mps_available = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )
    device = "cuda" if available else "mps" if mps_available else "cpu"
    return available, device, devices


def configure_model_environment(paths: CodecoPath) -> dict[str, str]:
    """Keep every downloaded model in the persistent project model directory."""
    directories = {
        "BIRDNET_APP_DATA": paths.models / "birdnet",
        "TFHUB_CACHE_DIR": paths.models / "tensorflow_hub",
        "HF_HOME": paths.models / "huggingface",
        "HF_HUB_CACHE": paths.models / "huggingface" / "hub",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    for name, directory in directories.items():
        os.environ[name] = str(directory)
    return {name: str(directory) for name, directory in directories.items()}


def _configure_process_environment(paths: CodecoPath) -> dict[str, str | None]:
    """Centralize process variables required by the remaining pipeline stages."""
    cache_directories = {
        "MPLCONFIGDIR": paths.cache / "matplotlib",
    }
    for directory in cache_directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    os_environment = os.environ
    os_environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os_environment.setdefault("MPLCONFIGDIR", str(cache_directories["MPLCONFIGDIR"]))
    configure_model_environment(paths)
    os_environment["BIRDNET_START_METHOD"] = "spawn"

    purelib = Path(sysconfig.get_paths()["purelib"])
    nvidia = purelib / "nvidia"
    library_directories = sorted(
        path for path in nvidia.glob("*/lib") if path.is_dir()
    )
    existing_libraries = [
        value
        for value in os_environment.get("LD_LIBRARY_PATH", "").split(":")
        if value
    ]
    additions = [
        str(path) for path in library_directories if str(path) not in existing_libraries
    ]
    if additions:
        os_environment["LD_LIBRARY_PATH"] = ":".join(
            [*additions, *existing_libraries]
        )

    nvcc = nvidia / "cuda_nvcc" / "bin"
    existing_path = [
        value for value in os_environment.get("PATH", "").split(":") if value
    ]
    if nvcc.is_dir() and str(nvcc) not in existing_path:
        os_environment["PATH"] = ":".join([str(nvcc), *existing_path])

    names = (
        "PATH",
        "LD_LIBRARY_PATH",
        "TF_CPP_MIN_LOG_LEVEL",
        "TFHUB_CACHE_DIR",
        "MPLCONFIGDIR",
        "BIRDNET_APP_DATA",
        "HF_HOME",
        "HF_HUB_CACHE",
        "BIRDNET_START_METHOD",
        "CUDA_VISIBLE_DEVICES",
    )
    return {name: os_environment.get(name) for name in names}


def _tensorflow_gpu_count(variables: dict[str, str | None]) -> int:
    environment = {
        key: value
        for key, value in os.environ.items()
        if value is not None
    }
    environment.update(
        {key: value for key, value in variables.items() if value is not None}
    )
    probe = (
        "import tensorflow as tf; "
        "print('CODECO_GPU_COUNT=' + str(len(tf.config.list_physical_devices('GPU'))))"
    )
    process = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=environment,
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise SystemDependencyError(
            f"TensorFlow nao pode validar as GPUs do ambiente: {detail}"
        )
    for line in reversed(process.stdout.splitlines()):
        if line.startswith("CODECO_GPU_COUNT="):
            return int(line.split("=", 1)[1])
    raise SystemDependencyError(
        "TensorFlow nao informou quantas GPUs estao disponiveis."
    )


def _system_environment(
    paths: CodecoPath,
    variables: dict[str, str | None],
) -> SystemEnvironment:
    LOGGER.info("Verificando a versao do Python...")
    expected_python = (paths.root / ".python-version").read_text(encoding="utf-8").strip()
    observed_python = platform.python_version()
    expected_series = ".".join(expected_python.split(".")[:2])
    observed_series = ".".join(observed_python.split(".")[:2])
    if observed_series != expected_series:
        raise SystemDependencyError(
            f"Python {observed_python} esta ativo; o projeto requer Python {expected_series}.x."
        )
    LOGGER.info("Python compativel: %s", observed_python)

    LOGGER.info("Verificando o sistema operacional...")
    operating_system = _supported_operating_system()
    LOGGER.info("Sistema operacional suportado: %s", operating_system)

    required_uv_version = _required_uv_version(paths)
    LOGGER.info("Verificando o uv %s no PATH...", required_uv_version)
    uv, uv_version = _uv_executable(paths)
    LOGGER.info("uv validado: %s (%s)", uv_version, uv)

    LOGGER.info("Procurando FFmpeg no PATH do sistema...")
    ffmpeg = _ffmpeg_installation(operating_system)
    LOGGER.info("FFmpeg validado: %s (%s)", ffmpeg.version, ffmpeg.executable)
    LOGGER.info(
        "FFmpeg LC3 validado: %s (%s)",
        ffmpeg.lc3_version,
        ffmpeg.lc3_executable,
    )

    LOGGER.info("Verificando disponibilidade de GPU CUDA...")
    cuda_available, device, gpu_devices = _cuda()
    LOGGER.info(
        "GPU CUDA disponivel: %s; dispositivo selecionado: %s; GPUs: %d",
        "sim" if cuda_available else "nao",
        device,
        len(gpu_devices),
    )
    for gpu in gpu_devices:
        LOGGER.info(
            "GPU %d: %s; VRAM total: %.0f MiB",
            gpu.index,
            gpu.name,
            gpu.total_memory_bytes / 1024**2,
        )

    LOGGER.info("Verificando GPUs visiveis para TensorFlow...")
    tensorflow_gpu_count = _tensorflow_gpu_count(variables)
    LOGGER.info("GPUs visiveis para TensorFlow: %d", tensorflow_gpu_count)
    return SystemEnvironment(
        operating_system=operating_system,
        python_version=observed_python,
        uv=uv,
        uv_version=uv_version,
        ffmpeg=ffmpeg,
        cuda_available=cuda_available,
        device=device,
        gpu_devices=gpu_devices,
        tensorflow_gpu_count=tensorflow_gpu_count,
    )


def _sync_environment(paths: CodecoPath) -> None:
    uv, uv_version = _uv_executable(paths)
    LOGGER.info("uv %s validado: %s", uv_version, uv)
    LOGGER.info("Sincronizando os pacotes fixados em uv.lock...")
    subprocess.run(
        [
            str(uv),
            "sync",
            "--project",
            str(paths.root),
            "--locked",
            "--inexact",
        ],
        check=True,
    )
    LOGGER.info("Pacotes do ambiente sincronizados.")


def _check_environment(paths: CodecoPath) -> None:
    """Validate the active environment without mutating a running kernel."""
    uv, uv_version = _uv_executable(paths)
    LOGGER.info("uv %s validado: %s", uv_version, uv)
    LOGGER.info("Conferindo os pacotes fixados em uv.lock sem alterar o ambiente...")
    command = [
        str(uv),
        "sync",
        "--project",
        str(paths.root),
        "--locked",
        "--inexact",
        "--check",
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise VirtualEnvironmentError(
            "O ambiente ativo nao corresponde ao uv.lock. Feche ou reinicie o "
            "kernel depois de executar no terminal:\n"
            f"cd {paths.root}\n"
            "uv sync --locked --inexact"
        ) from exc
    LOGGER.info("Ambiente ativo compativel com uv.lock.")


def _virtual_environment(paths: CodecoPath) -> Path:
    return paths.root / ".venv"


def _environment_python(virtual_environment: Path, operating_system: str) -> Path:
    if operating_system == "Windows":
        return virtual_environment / "Scripts" / "python.exe"
    return virtual_environment / "bin" / "python"


def _running_inside(virtual_environment: Path) -> bool:
    try:
        return Path(sys.prefix).resolve() == virtual_environment.resolve()
    except OSError:
        return False


def virtual_env() -> EnvironmentState:
    """Return the active project environment or ask the user to activate it."""
    paths = CodecoPath.discover()
    log_file, logs_created = _configure_logging(paths)
    _pipeline_header("ENVIRONMENT")
    _section("Registro da execucao")
    LOGGER.info(
        "%s: %s",
        "Diretorio de logs criado" if logs_created else "Diretorio de logs encontrado",
        paths.logs,
    )
    LOGGER.info("Log da verificacao de ambiente: %s", log_file)
    LOGGER.info("Iniciando verificacao do ambiente do projeto: %s", paths.root)

    _section("Ambiente virtual")
    virtual_environment = _virtual_environment(paths)
    LOGGER.info("Verificando ambiente virtual: %s", virtual_environment)
    operating_system = _supported_operating_system()
    python = _environment_python(virtual_environment, operating_system)
    if not virtual_environment.is_dir():
        LOGGER.info("Ambiente virtual inexistente; iniciando a instalacao.")
        _sync_environment(paths)
        raise VirtualEnvironmentError(
            f"O ambiente virtual foi instalado em: {virtual_environment}\n"
            f"Linux: source {virtual_environment / 'bin' / 'activate'}\n"
            f"Windows PowerShell: {virtual_environment / 'Scripts' / 'Activate.ps1'}"
        )
    if not _running_inside(virtual_environment):
        LOGGER.error("O ambiente virtual existe, mas nao esta ativo.")
        raise VirtualEnvironmentError(
            f"O ambiente virtual do projeto nao esta ativo: {virtual_environment}\n"
            f"Linux: source {virtual_environment / 'bin' / 'activate'}\n"
            f"Windows PowerShell: {virtual_environment / 'Scripts' / 'Activate.ps1'}"
        )

    LOGGER.info("Ambiente virtual ativo: %s", virtual_environment)

    _section("Variaveis do pipeline")
    variables = _configure_process_environment(paths)
    for name, value in variables.items():
        if name in {"PATH", "LD_LIBRARY_PATH"}:
            status = "configurado" if value else "nao definido"
            LOGGER.info("%s: %s", name, status)
        else:
            LOGGER.info("%s: %s", name, value if value is not None else "dinamico")

    _section("Pacotes do ambiente")
    _check_environment(paths)

    _section("Dependencias do sistema")
    system = _system_environment(paths, variables)
    LOGGER.info("Verificacao do ambiente concluida com sucesso.")
    return EnvironmentState(
        paths=paths,
        virtual_environment=virtual_environment,
        python=python,
        system=system,
        variables=variables,
    )


if __name__ == "__main__":
    try:
        state = virtual_env()
        print(f"Ambiente virtual ativo: {state.virtual_environment}")
        print(f"Sistema operacional: {state.system.operating_system}")
        print(f"Python: {state.system.python_version}")
        print(f"FFmpeg: {state.system.ffmpeg.version}")
        print(f"GPU CUDA disponivel: {'sim' if state.system.cuda_available else 'nao'}")
        print(f"Dispositivo selecionado: {state.system.device}")
        print(state)
    except (VirtualEnvironmentError, SystemDependencyError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from None
