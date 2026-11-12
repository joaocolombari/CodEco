"""Interface unificada de codecs: encode(wav) -> Encoded(payload, nbytes); decode(Encoded) -> wav."""
import os
import resource
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf

import config as C


@dataclass
class Encoded:
    payload: Any
    nbytes: int


def _children_cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


class Codec:
    name = "base"
    setting = ""
    last_encode_cpu_s = float("nan")
    last_decode_cpu_s = float("nan")

    def encode(self, wav: np.ndarray) -> Encoded: ...
    def decode(self, enc: Encoded) -> np.ndarray: ...


class FFmpegCodec(Codec):
    ARGS = {
        "flac": ("flac", lambda p: ["-c:a", "flac", "-compression_level", str(p["compression_level"])]),
        "mp3":  ("mp3",  lambda p: ["-c:a", "libmp3lame", "-b:a", f"{p['bitrate_kbps']}k"]),
        "opus": ("opus", lambda p: ["-c:a", "libopus", "-b:a", f"{p['bitrate_kbps']}k",
                                    "-vbr", "on", "-application", "audio"]),
        "lc3":  ("lc3",  lambda p: ["-c:a", "liblc3", "-b:a", f"{p['bitrate_kbps']}k"]),
    }

    def __init__(self, name, **params):
        self.name, self.params = name, params
        self.ext, self._args = self.ARGS[name]
        self.setting = ",".join(f"{k}={v}" for k, v in params.items())

    def encode(self, wav):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in.wav"), os.path.join(d, f"out.{self.ext}")
            sf.write(src, wav, C.SR, subtype="PCM_16")
            t0 = _children_cpu()
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", str(C.SR),
                            *self._args(self.params), dst], check=True)
            self.last_encode_cpu_s = _children_cpu() - t0
            data = open(dst, "rb").read()
        return Encoded(data, len(data))

    def decode(self, enc):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, f"in.{self.ext}"), os.path.join(d, "out.wav")
            open(src, "wb").write(enc.payload)
            t0 = _children_cpu()
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ar", str(C.SR), "-ac", "1",
                            "-f", "wav", dst], check=True)
            self.last_decode_cpu_s = _children_cpu() - t0
            y, _ = sf.read(dst, dtype="float32")
        return y


class EncodecCodec(Codec):
    """EnCodec pre-treinado (HuggingFace). Sem treino."""
    name = "encodec"

    def __init__(self, bandwidth_kbps, model_name="facebook/encodec_48khz", device="cpu"):
        import torch
        from transformers import EncodecModel
        self.torch, self.device, self.bw = torch, device, float(bandwidth_kbps)
        self.model = EncodecModel.from_pretrained(model_name).to(device).eval()
        self.sr = self.model.config.sampling_rate
        self.channels = self.model.config.audio_channels
        self.cb_bits = int(np.ceil(np.log2(self.model.config.codebook_size)))
        self.setting = f"bandwidth_kbps={bandwidth_kbps},model={model_name.split('/')[-1]}"

    def encode(self, wav):
        torch = self.torch
        x = torch.from_numpy(wav).float()[None, None]
        if self.sr != C.SR:
            import torchaudio.functional as AF
            x = AF.resample(x, C.SR, self.sr)
        x = x.repeat(1, self.channels, 1).to(self.device)
        with torch.no_grad():
            out = self.model.encode(x, bandwidth=self.bw)
        codes, scales = out.audio_codes, out.audio_scales
        n_scales = sum(s.numel() for s in scales if s is not None)
        nbytes = int(np.ceil(codes.numel() * self.cb_bits / 8)) + 4 * n_scales
        return Encoded((codes, scales), nbytes)

    def decode(self, enc):
        torch = self.torch
        codes, scales = enc.payload
        with torch.no_grad():
            y = self.model.decode(codes, scales, None).audio_values
        y = y.mean(dim=1)[0].cpu()
        if self.sr != C.SR:
            import torchaudio.functional as AF
            y = AF.resample(y, self.sr, C.SR)
        return y.numpy()


class AECodec(Codec):
    name = "ae"

    def __init__(self, variant, levels=C.AE_QUANT_LEVELS, ckpt=None, device="cpu"):
        """ckpt=None -> pesos aleatorios (valido para medir FLOPs, pois FLOPs nao dependem dos pesos)."""
        import torch
        from autoencoder import build_autoencoder, strip_weight_norm
        self.torch, self.device, self.variant = torch, device, variant
        self.levels = int(levels)
        model = build_autoencoder(variant, levels=self.levels).to(device)
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt, map_location=device))
        self.model = strip_weight_norm(model).eval()
        self.extra_enc_flops, self.extra_dec_flops = model.extra_enc_flops, model.extra_dec_flops
        self.bits = int(np.ceil(np.log2(self.levels)))
        self.setting = f"variant={variant},levels={self.levels}"

    def encode(self, wav):
        q, rms_val = self.model.encode(wav, levels=self.levels)
        payload_bits = q.size * self.bits
        payload_bytes = int(np.ceil(payload_bits / 8)) + 4  # +4 bytes do ganho RMS float32
        return Encoded((q, rms_val), payload_bytes)

    def decode(self, enc):
        q, rms_val = enc.payload
        return self.model.decode(q, rms_val, levels=self.levels)


def make_codec(name, params, dataset=None, allow_untrained_ae=False):
    if name in FFmpegCodec.ARGS:
        return FFmpegCodec(name, **params)
    if name == "encodec":
        return EncodecCodec(**params)
    if name == "ae":
        from autoencoder import ckpt_path
        ckpt = ckpt_path(dataset, params["variant"]) if dataset else None
        if ckpt is not None and not ckpt.exists():
            if allow_untrained_ae:
                ckpt = None
            else:
                raise FileNotFoundError(f"Checkpoint do Autoencoder nao encontrado: {ckpt}")
        return AECodec(params["variant"], levels=params.get("levels", C.AE_QUANT_LEVELS), ckpt=ckpt)
    raise ValueError(name)


def align_to_reference(ref: np.ndarray, dec: np.ndarray, max_lag: int = C.ALIGN_MAX_LAG):
    """Compensa atraso (delay do encoder/decoder) e ajusta o comprimento. Retorna (dec_alinhado, lag)."""
    from scipy import signal
    n = len(ref)
    dec = np.pad(dec, (0, max(0, n + max_lag - len(dec))))
    seg = min(n, C.SR)
    corr = signal.correlate(dec[: seg + max_lag], ref[:seg], mode="valid", method="fft")
    lag = int(np.argmax(corr))
    return dec[lag: lag + n].astype(np.float32), lag