"""Benchmark de custo computacional dos codecs."""
import argparse
import logging
import os
import platform
import sys
import resource
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager, ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from codeco.codeco_path import CodecoPath
from codeco.encodec import CONDITIONS as ENCODEC_CONDITIONS, MODEL_NAME
from codeco.environment import (
    EnvironmentState,
    _ffmpeg_installation,
    configure_model_environment,
)
from codeco.ffmpeg import CONDITIONS as FFMPEG_CONDITIONS
from codeco.observability import TqdmLoggingHandler
from codeco.prepare_table import CLIP_SECONDS, SAMPLE_RATE

LOGGER = logging.getLogger("codeco.flops")
RESULT_COLUMNS = {
    "codec",
    "setting",
    "type",
    "enc_cpu_s_per_clip",
    "dec_cpu_s_per_clip",
    "enc_cpu_ms_per_s_audio",
    "dec_cpu_ms_per_s_audio",
    "enc_instructions_per_s_audio",
    "dec_instructions_per_s_audio",
    "iot_enc_flops",
    "iot_enc_gflops_per_s",
    "iot_enc_params",
    "srv_dec_flops",
    "srv_dec_gflops_per_s",
    "srv_dec_params",
}


def _configure_logging(paths, log_path: Path | None = None):
    log_file = log_path or paths.logs / "flops.log"
    if not LOGGER.handlers:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        persistent = logging.FileHandler(log_file, encoding="utf-8")
        persistent.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        LOGGER.addHandler(persistent)
        LOGGER.addHandler(TqdmLoggingHandler(sys.stdout))
        LOGGER.setLevel(logging.INFO)
        LOGGER.propagate = False
    return log_file


def _section(title):
    LOGGER.info("\n%s\n%s", title, "-" * 72)


def _validate_result(table, source):
    missing = RESULT_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(
            f"Resultado incompatível em {source}; colunas ausentes: "
            f"{', '.join(sorted(missing))}"
        )
    if table.empty:
        raise ValueError(f"Resultado vazio em {source}")
    return table


def _write_csv(table, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        temp_name = tmp.name
    try:
        table.to_csv(temp_name, index=False, na_rep="-")
        os.replace(temp_name, output)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _load_result(paths, output):
    legacy = paths.benchmarks / "flops.parquet"
    if output.is_file():
        return _validate_result(
            pd.read_csv(output, na_values=["-"], keep_default_na=True),
            output,
        ), output, False
    if output == paths.benchmarks / "flops.csv" and legacy.is_file():
        table = _validate_result(pd.read_parquet(legacy), legacy)
        _write_csv(table, output)
        legacy.unlink()
        return table, output, True
    return None, output, False

SR = SAMPLE_RATE
SECONDS = CLIP_SECONDS
SAMPLES = round(SR * SECONDS)


def _measure_ffmpeg_batch(
    condition,
    wav: np.ndarray,
    n_repeats: int = 20,
    use_perf: bool = True,
    executable="ffmpeg",
    fp_event=None,
) -> dict:
    import soundfile as sf

    codec = condition.codec
    long_wav = np.tile(wav, n_repeats)
    total_audio_s = SECONDS * n_repeats
    ext = condition.extension
    enc_args = list(condition.encoder_arguments)

    with tempfile.TemporaryDirectory() as d:
        src_wav = os.path.join(d, "batch_in.wav")
        encoded_file = os.path.join(d, f"batch_out.{ext}")
        decoded_wav = os.path.join(d, "batch_dec.wav")
        sf.write(src_wav, long_wav, SR, subtype="PCM_16")

        cmd_enc = [
            str(executable), "-v", "error", "-y", "-i", src_wav,
            "-ac", "1", "-ar", str(SR), *enc_args, encoded_file,
        ]
        cmd_dec = [
            str(executable), "-v", "error", "-y", "-i", encoded_file,
            "-ar", str(SR), "-ac", "1", "-f", "wav", decoded_wav,
        ]

        subprocess.run(cmd_enc, check=True)
        subprocess.run(cmd_dec, check=True)

        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0_wall = time.perf_counter()
        subprocess.run(cmd_enc, check=True)
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t1_wall = time.perf_counter()
        subprocess.run(cmd_dec, check=True)
        r2 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t2_wall = time.perf_counter()

        enc_cpu_total = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
        dec_cpu_total = (r2.ru_utime - r1.ru_utime) + (r2.ru_stime - r1.ru_stime)

        if enc_cpu_total <= 0:
            enc_cpu_total = t1_wall - t0_wall
        if dec_cpu_total <= 0:
            dec_cpu_total = t2_wall - t1_wall

        res = {
            "enc_cpu_s_per_clip": enc_cpu_total / n_repeats,
            "dec_cpu_s_per_clip": dec_cpu_total / n_repeats,
            "enc_cpu_ms_per_s_audio": (enc_cpu_total / total_audio_s) * 1000.0,
            "dec_cpu_ms_per_s_audio": (dec_cpu_total / total_audio_s) * 1000.0,
            "enc_instructions_per_s_audio": np.nan,
            "dec_instructions_per_s_audio": np.nan,
        }

        if use_perf:
            events = ["instructions"] + ([fp_event] if fp_event else [])
            for direction, prefix, cmd in (("enc", "iot_enc", cmd_enc), ("dec", "srv_dec", cmd_dec)):
                proc = subprocess.run(["perf", "stat", "-x,", "-e", ",".join(events), "--", *cmd],
                                      capture_output=True, text=True, check=True)
                res[f"{direction}_instructions_per_s_audio"] = _counter(proc.stderr, "instructions") / total_audio_s
                if fp_event:
                    ops = _counter(proc.stderr, fp_event)
                    res[f"{prefix}_flops"] = ops / n_repeats
                    res[f"{prefix}_gflops_per_s"] = ops / total_audio_s / 1e9
                    res["fp_event"] = fp_event

        return res


def _counter(report, event):
    for line in report.splitlines():
        parts = line.split(",")
        if len(parts) >= 3 and parts[2].split(":")[0] == event:
            return float(parts[0])
    raise RuntimeError(f"Contador {event} indisponivel: {report.strip()}")


@contextmanager
def _phase_instructions(enabled, result):
    if not enabled:
        yield
        return
    with ExitStack() as stack:
        read_cmd, write_cmd = os.pipe()
        read_ack, write_ack = os.pipe()
        stack.callback(os.close, write_cmd)
        stack.callback(os.close, read_ack)
        try:
            proc = subprocess.Popen(
                ["perf", "stat", "-x,", "-D", "-1", "--control", f"fd:{read_cmd},{write_ack}",
                 "-p", str(os.getpid()), "-e", "instructions", "--", "cat"],
                pass_fds=(read_cmd, write_ack), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        finally:
            os.close(read_cmd)
            os.close(write_ack)
        try:
            def control(command):
                os.write(write_cmd, command)
                if not os.read(read_ack, 32).startswith(b"ack"):
                    raise RuntimeError("perf nao confirmou a delimitacao da fase.")
            control(b"enable\n")
            yield
            control(b"disable\n")
        finally:
            _, report = proc.communicate()
        result["instructions"] = _counter(report.decode(), "instructions")


def _measure_phase(operation, repeats, use_perf, direction):
    counters = {}
    with _phase_instructions(use_perf, counters):
        start = time.process_time()
        output = operation()
        elapsed = time.process_time() - start
    duration = repeats * SECONDS
    return output, {
        f"{direction}_cpu_s_per_clip": elapsed / repeats,
        f"{direction}_cpu_ms_per_s_audio": elapsed * 1000 / duration,
        f"{direction}_instructions_per_s_audio": counters.get("instructions", np.nan) / duration,
    }


def _neural_rows(wav, repeats, use_perf, model_name, conditions):
    import torch
    from thop import profile
    from transformers import EncodecModel

    model = EncodecModel.from_pretrained(model_name).eval()
    x = torch.from_numpy(wav)[None, None].repeat(1, model.config.audio_channels, 1)
    # Count network operations on the exact chunks passed by the model.
    counts = {}
    def capture(module, inputs):
        shape = tuple(inputs[0].shape)
        counts[shape] = counts.get(shape, 0) + 1

    with torch.no_grad():
        hook = model.encoder.register_forward_pre_hook(capture)
        try:
            sample = model.encode(x, bandwidth=6.0)
        finally:
            hook.remove()
        enc = sum(2 * profile(model.encoder, inputs=(x.new_zeros(shape),), verbose=False)[0] * n
                  for shape, n in counts.items())
        counts.clear()
        hook = model.decoder.register_forward_pre_hook(capture)
        try:
            model.decode(sample.audio_codes, sample.audio_scales, None)
        finally:
            hook.remove()
        dec = sum(2 * profile(model.decoder, inputs=(x.new_zeros(shape),), verbose=False)[0] * n
                  for shape, n in counts.items())
        batch = x.repeat(1, 1, repeats)
        for condition in conditions:
            rate = float(condition.bandwidth_kbps)
            codes = model.encode(x, bandwidth=float(rate)).audio_codes
            packed, enc_metrics = _measure_phase(lambda: model.encode(batch, bandwidth=float(rate)),
                                                 repeats, use_perf, "enc")
            _, dec_metrics = _measure_phase(lambda: model.decode(packed.audio_codes, packed.audio_scales, None),
                                            repeats, use_perf, "dec")
            books = codes.shape[-2]
            entries, dim = model.quantizer.layers[0].codebook.embed.shape
            # Dominant RVQ search: one dot product per candidate and code.
            search = 2 * codes.numel() * entries * dim
            additions = codes.numel() * dim
            enc_total = enc + search
            dec_total = dec + additions
            enc_params = sum(p.numel() for p in model.encoder.parameters())
            dec_params = sum(p.numel() for p in model.decoder.parameters())
            yield dict(codec="encodec", setting=condition.setting,
                       type="neural", iot_enc_flops=enc_total,
                       iot_enc_gflops_per_s=enc_total / (1e9 * SECONDS), iot_enc_params=enc_params,
                       srv_dec_flops=dec_total, srv_dec_gflops_per_s=dec_total / (1e9 * SECONDS),
                       srv_dec_params=dec_params, bandwidth_kbps=rate, codebooks=books,
                       enc_params=enc_params, dec_params=dec_params,
                       enc_network_flops=enc, rvq_search_flops=search,
                       dec_network_flops=dec, rvq_sum_flops=additions,
                       enc_gflops_clip=(enc + search) / 1e9,
                       dec_gflops_clip=(dec + additions) / 1e9, **enc_metrics, **dec_metrics)




def _format_results(table):
    def number(value, digits):
        if pd.isna(value):
            return "indisp."
        return f"{value:,.{digits}f}".translate(str.maketrans(",.", ".,"))

    blocks = []
    for codec, group in table.groupby("codec", sort=False):
        lines = [f"- {str(codec).upper()} ({group.iloc[0]['type']})"]
        for row in group.to_dict("records"):
            setting = row["setting"]
            if pd.notna(row.get("codebooks")):
                setting += f" · {int(row['codebooks'])} codebooks"
            lines.extend(["", f"    {setting}"])
            for label, side, prefix in (("Encoder", "enc", "iot_enc"), ("Decoder", "dec", "srv_dec")):
                lines.extend([
                    f"      {label}",
                    f"        CPU: {number(row[f'{side}_cpu_s_per_clip'], 6)} s/clipe"
                    f" · {number(row[f'{side}_cpu_ms_per_s_audio'], 3)} ms/s de áudio",
                    f"        Instruções/s de áudio: {number(row[f'{side}_instructions_per_s_audio'], 2)}",
                    f"        FLOPs/clipe: {number(row[f'{prefix}_flops'], 2)}"
                    f" · GFLOPs/s de áudio: {number(row[f'{prefix}_gflops_per_s'], 9)}",
                ])
                if row["type"] == "neural":
                    lines.append(f"        Parâmetros: {number(row[f'{prefix}_params'], 0)}")
        blocks.append("\n".join(lines))
    return "\n\n--------------------------\n\n".join(blocks)


# ---- Notebook 08 (integration scripts/flops.py) counting, kept side by side ----
# thop/ptflops on the whole encode()/decode() calls. It does not include the RVQ
# codebook search, which the pipeline count above adds explicitly.
def _notebook08_encodec_flops(wav, model_name, bandwidths):
    """EnCodec FLOPs and parameters exactly as notebook 08 measures them."""
    import torch
    from transformers import EncodecModel

    from codeco.benchmark.notebook08_flops import (
        _EncodecDecoderWrapper,
        _EncodecEncoderWrapper,
        _count_neural_flops as count,
    )

    model = EncodecModel.from_pretrained(model_name).eval()
    x_t = torch.from_numpy(wav).float()[None, None]
    x_t = x_t.repeat(1, model.config.audio_channels, 1)
    results = {}
    for bandwidth in bandwidths:
        enc_wrapper = _EncodecEncoderWrapper(model, float(bandwidth))
        enc_flops, enc_params = count(enc_wrapper, x_t)
        with torch.no_grad():
            enc_out = model.encode(x_t, bandwidth=float(bandwidth))
        dec_wrapper = _EncodecDecoderWrapper(model, enc_out.audio_scales)
        dec_flops, dec_params = count(dec_wrapper, enc_out.audio_codes)
        results[float(bandwidth)] = {
            "nb08_iot_enc_flops": enc_flops,
            "nb08_iot_enc_gflops_per_s": (enc_flops / SECONDS) / 1e9,
            "nb08_iot_enc_params": enc_params,
            "nb08_srv_dec_flops": dec_flops,
            "nb08_srv_dec_gflops_per_s": (dec_flops / SECONDS) / 1e9,
            "nb08_srv_dec_params": dec_params,
        }
    return results


def run(
    n_repeats=20,
    use_perf=True,
    force=False,
    environment: EnvironmentState | None = None,
    output_path: Path | None = None,
    log_path: Path | None = None,
):
    paths = CodecoPath.discover()
    configure_model_environment(paths)
    output = output_path or paths.benchmarks / "flops.csv"
    log_file = _configure_logging(paths, log_path)
    LOGGER.info("\n%s\nBENCHMARK DE CUSTO COMPUTACIONAL\n[CMD]   %s\n%s",
                "=" * 72, subprocess.list2cmdline([sys.executable, *sys.argv]), "=" * 72)
    _section("Registro da execucao")
    LOGGER.info("Log: %s", log_file)
    if n_repeats < 1:
        raise ValueError("O lote precisa de pelo menos um clipe.")
    if not force:
        try:
            existing, output, migrated = _load_result(paths, output)
        except (OSError, ValueError) as error:
            LOGGER.warning("Resultado existente invalido; recalculando: %s", error)
        else:
            if existing is not None:
                missing_perf = existing[
                    [
                        "enc_instructions_per_s_audio",
                        "dec_instructions_per_s_audio",
                    ]
                ].isna().all().all()
                missing_notebook08 = "nb08_iot_enc_flops" not in existing.columns
                if missing_notebook08:
                    LOGGER.warning(
                        "Resultado existente nao possui a contagem do notebook 08; recalculando: %s",
                        output,
                    )
                elif use_perf and missing_perf:
                    LOGGER.warning(
                        "Resultado existente nao possui contadores perf; recalculando: %s",
                        output,
                    )
                else:
                    action = "migrado do Parquet legado e carregado" if migrated else "carregado"
                    LOGGER.info(
                        "Resultado existente %s: %d configuracoes em %s\n\n%s",
                        action,
                        len(existing),
                        output,
                        _format_results(existing),
                    )
                    return existing
    else:
        LOGGER.info("Recalculo solicitado por --force; resultado existente sera substituido.")
    if use_perf and not shutil.which("perf"):
        raise RuntimeError("perf indisponivel. Instale perf ou use --no-perf.")
    wav = (
        0.1 * np.random.default_rng(42).standard_normal(SAMPLES)
    ).astype(np.float32)
    ffmpeg = (
        environment.system.ffmpeg
        if environment is not None
        else _ffmpeg_installation(platform.system())
    )
    LOGGER.info("FFmpeg %s: %s", ffmpeg.version, ffmpeg.executable)
    LOGGER.info("Sinal: %d Hz | %d x %d s = %d s | perf: %s",
                SR, n_repeats, SECONDS, n_repeats * SECONDS, use_perf)
    rows = []
    fp_event = None
    if use_perf:
        event = "fp_ret_sse_avx_ops.all"
        probe = subprocess.run(["perf", "stat", "-x,", "-e", event, "--", "true"], capture_output=True, text=True)
        if probe.returncode == 0 and event in probe.stderr:
            _counter(probe.stderr, event)
            fp_event = event
        LOGGER.info("Contador de FLOPs FFmpeg: %s", fp_event or "indisponivel")
    _section("FFmpeg | CPU em ms/clipe")
    configs = [condition for condition in FFMPEG_CONDITIONS if condition.codec != "baseline"]
    with tqdm(total=len(configs) + 3, unit="config", dynamic_ncols=True, file=sys.stdout) as bar:
        for condition in configs:
            bar.set_description_str(f"{condition.codec} {condition.setting}")
            executable = (
                ffmpeg.lc3_executable
                if condition.codec == "lc3"
                else ffmpeg.executable
            )
            result = _measure_ffmpeg_batch(condition, wav, n_repeats, use_perf, executable, fp_event)
            rows.append(dict(codec=condition.codec, setting=condition.setting, type="dsp_c", **result))
            LOGGER.info(f"{condition.codec.upper():5} {condition.setting:24}  enc {1000*result['enc_cpu_s_per_clip']:7.2f}"
                  f"  dec {1000*result['dec_cpu_s_per_clip']:7.2f}")
            bar.update()
        _section("EnCodec | GFLOPs estimados por clipe")
        bar.set_description_str("EnCodec")
        neural_conditions = [
            condition
            for condition in ENCODEC_CONDITIONS
            if condition.codec == "encodec"
        ]
        notebook08 = _notebook08_encodec_flops(
            wav, MODEL_NAME, [condition.bandwidth_kbps for condition in neural_conditions]
        )
        for row in _neural_rows(wav, n_repeats, use_perf, MODEL_NAME, neural_conditions):
            row.update(notebook08[float(row["bandwidth_kbps"])])
            rows.append(row)
            LOGGER.info("%d kb/s  %d codebooks  enc %.3f  dec %.3f",
                        row["bandwidth_kbps"], row["codebooks"], row["enc_gflops_clip"], row["dec_gflops_clip"])
            bar.update()
    _section("Resultados do benchmark")
    table = pd.DataFrame(rows)
    _validate_result(table, "benchmark em memoria")
    _write_csv(table, output)
    legacy = paths.benchmarks / "flops.parquet"
    if output == paths.benchmarks / "flops.csv" and legacy.is_file():
        legacy.unlink()
    LOGGER.info("Resultados salvos em %s\n\n%s", output, _format_results(table))
    LOGGER.info("\nBenchmark concluido: %d configuracoes.\n%s", len(rows), output)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--perf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignora o CSV existente e executa novamente todo o benchmark.",
    )
    args = parser.parse_args()
    try:
        paths = CodecoPath.discover()
        test_directory = paths.tests / "flops"
        run(
            args.repeats,
            args.perf,
            args.force,
            output_path=test_directory / "results.csv",
            log_path=test_directory / "execution.log",
        )
    except Exception:
        LOGGER.exception("Falha no benchmark.")
        raise SystemExit(1) from None
