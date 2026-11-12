"""Benchmark rigoroso de custo computacional para contexto IoT:
- Codecs neurais (EnCodec, Autoencoder): contagem de FLOPs/MACs via thop ou ptflops,
  reportando separadamente Encoder (dispositivo IoT) e Decoder (servidor), alem de parametros.
- Codecs tradicionais (FLAC, MP3, Opus, LC3): benchmark de tempo de CPU isolado em lote
  e suporte opcional a `perf stat` no Linux.
"""
import argparse
import os
import resource
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn

import config as C
from codecs_lib import FFmpegCodec, make_codec


def _count_neural_flops(module: nn.Module, sample_input: torch.Tensor) -> tuple[float, float]:
    module.eval()
    with torch.no_grad():
        try:
            from thop import profile
            macs, params = profile(module, inputs=(sample_input,), verbose=False)
            return float(macs * 2.0), float(params)
        except ImportError:
            pass

        try:
            from ptflops import get_model_complexity_info
            input_shape = tuple(sample_input.shape[1:])
            macs, params = get_model_complexity_info(
                module, input_shape, as_strings=False, print_per_layer_stat=False, verbose=False,
            )
            return float(macs * 2.0), float(params)
        except ImportError:
            pass

        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            _ = module(sample_input)
        params = sum(p.numel() for p in module.parameters())
        return float(fc.get_total_flops()), float(params)


class _EncodecEncoderWrapper(nn.Module):
    def __init__(self, encodec_model, bandwidth: float):
        super().__init__()
        self.model = encodec_model
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor):
        return self.model.encode(x, bandwidth=self.bandwidth)


class _EncodecDecoderWrapper(nn.Module):
    def __init__(self, encodec_model, audio_scales):
        super().__init__()
        self.model = encodec_model
        self.audio_scales = audio_scales

    def forward(self, audio_codes: torch.Tensor):
        return self.model.decode(audio_codes, self.audio_scales, None)


class _AEEncoderWrapper(nn.Module):
    def __init__(self, ae_model, levels: int):
        super().__init__()
        self.model = ae_model
        self.levels = levels

    def forward(self, x: torch.Tensor):
        return self.model.encode_tensor(x, levels=self.levels)


class _AEDecoderWrapper(nn.Module):
    def __init__(self, ae_model):
        super().__init__()
        self.model = ae_model

    def forward(self, zq: torch.Tensor):
        return self.model.decode_tensor(zq, rms=1.0)


def _measure_ffmpeg_batch(
    codec: FFmpegCodec,
    wav: np.ndarray,
    n_repeats: int = 20,
    use_perf: bool = False,
) -> dict:
    long_wav = np.tile(wav, n_repeats)
    total_audio_s = C.CLIP_SECONDS * n_repeats
    ext = codec.ext
    enc_args = codec._args(codec.params)

    with tempfile.TemporaryDirectory() as d:
        src_wav = os.path.join(d, "batch_in.wav")
        encoded_file = os.path.join(d, f"batch_out.{ext}")
        decoded_wav = os.path.join(d, "batch_dec.wav")
        sf.write(src_wav, long_wav, C.SR, subtype="PCM_16")

        cmd_enc = [
            "ffmpeg", "-v", "error", "-y", "-i", src_wav,
            "-ac", "1", "-ar", str(C.SR), *enc_args, encoded_file,
        ]
        cmd_dec = [
            "ffmpeg", "-v", "error", "-y", "-i", encoded_file,
            "-ar", str(C.SR), "-ac", "1", "-f", "wav", decoded_wav,
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

        if use_perf and shutil.which("perf") is not None:
            res["enc_instructions_per_s_audio"] = _run_perf_instructions(cmd_enc, total_audio_s)
            res["dec_instructions_per_s_audio"] = _run_perf_instructions(cmd_dec, total_audio_s)

        return res


def _run_perf_instructions(cmd: list[str], total_audio_s: float) -> float:
    perf_cmd = ["perf", "stat", "-x,", "-e", "instructions", *cmd]
    proc = subprocess.run(perf_cmd, capture_output=True, text=True)
    for line in proc.stderr.splitlines():
        parts = line.split(",")
        if len(parts) >= 3 and "instructions" in parts[2]:
            try:
                return float(parts[0].strip()) / total_audio_s
            except ValueError:
                pass
    return np.nan


def run(n_repeats: int = 20, use_perf: bool = False):
    rng = np.random.default_rng(C.SEED)
    x_np = (0.1 * rng.standard_normal(C.CLIP_SAMPLES)).astype(np.float32)
    rows = []

    for name, settings in C.CODECS.items():
        for params in settings:
            codec = make_codec(name, params, dataset=None, allow_untrained_ae=True)

            if name == "encodec":
                x_t = torch.from_numpy(x_np).float()[None, None]
                if codec.sr != C.SR:
                    import torchaudio.functional as AF
                    x_t = AF.resample(x_t, C.SR, codec.sr)
                x_t = x_t.repeat(1, codec.channels, 1).to(codec.device)

                enc_wrapper = _EncodecEncoderWrapper(codec.model, codec.bw)
                enc_flops, enc_params = _count_neural_flops(enc_wrapper, x_t)

                with torch.no_grad():
                    enc_out = codec.model.encode(x_t, bandwidth=codec.bw)
                dec_wrapper = _EncodecDecoderWrapper(codec.model, enc_out.audio_scales)
                dec_flops, dec_params = _count_neural_flops(dec_wrapper, enc_out.audio_codes)

                rows.append(dict(
                    codec=name,
                    setting=codec.setting,
                    type="neural",
                    iot_enc_flops=enc_flops,
                    iot_enc_gflops_per_s=(enc_flops / C.CLIP_SECONDS) / 1e9,
                    iot_enc_params=enc_params,
                    srv_dec_flops=dec_flops,
                    srv_dec_gflops_per_s=(dec_flops / C.CLIP_SECONDS) / 1e9,
                    srv_dec_params=dec_params,
                ))

            elif name == "ae":
                x_t = torch.from_numpy(x_np).float()[None, None].to(codec.device)

                enc_wrapper = _AEEncoderWrapper(codec.model, codec.levels)
                enc_flops, enc_params = _count_neural_flops(enc_wrapper, x_t)
                enc_flops += float(codec.extra_enc_flops)  # soma custo da STFT na variante stft

                with torch.no_grad():
                    zq_t, _, _ = codec.model.encode_tensor(x_t, levels=codec.levels)
                dec_wrapper = _AEDecoderWrapper(codec.model)
                dec_flops, dec_params = _count_neural_flops(dec_wrapper, zq_t)
                dec_flops += float(codec.extra_dec_flops)  # soma custo da iSTFT na variante stft

                rows.append(dict(
                    codec=name,
                    setting=codec.setting,
                    type="neural",
                    iot_enc_flops=enc_flops,
                    iot_enc_gflops_per_s=(enc_flops / C.CLIP_SECONDS) / 1e9,
                    iot_enc_params=enc_params,
                    srv_dec_flops=dec_flops,
                    srv_dec_gflops_per_s=(dec_flops / C.CLIP_SECONDS) / 1e9,
                    srv_dec_params=dec_params,
                ))

            else:
                bench = _measure_ffmpeg_batch(codec, x_np, n_repeats=n_repeats, use_perf=use_perf)
                rows.append(dict(
                    codec=name,
                    setting=codec.setting,
                    type="dsp_c",
                    **bench,
                ))

    df = pd.DataFrame(rows)
    out_csv = C.OUT_DIR / "flops.csv"
    df.to_csv(out_csv, index=False)
    print(f"Resultados salvos em {out_csv}:\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark rigoroso de custo computacional (IoT vs Servidor)")
    ap.add_argument("--repeats", type=int, default=20,
                    help="Numero de repeticoes do clipe para diluir overhead do processo ffmpeg")
    ap.add_argument("--perf", action="store_true",
                    help="Usa `perf stat` (Linux) para contar instrucoes exatas de CPU nos codecs C")
    args = ap.parse_args()

    C.OUT_DIR.mkdir(exist_ok=True)
    run(n_repeats=args.repeats, use_perf=args.perf)