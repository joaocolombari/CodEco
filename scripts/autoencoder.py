"""Autoencoders com quantizacao escalar no latente. Tres variantes (ver config.AE_VARIANTS):
   wave, wave_lite (1D na forma de onda) e stft (entrada STFT, saida via iSTFT).
Treino:  python autoencoder.py --dataset pteroset --variant stft
"""
import argparse
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

import config as C


def wn(m):
    return nn.utils.parametrizations.weight_norm(m)


class ResUnit(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.block = nn.Sequential(nn.ELU(), wn(nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)),
                                   nn.ELU(), wn(nn.Conv1d(ch, ch, 1)))

    def forward(self, x):
        return x + self.block(x)


# ------------------------------------------------------------------ forma de onda
class Encoder(nn.Module):
    def __init__(self, strides, base, latent, lite=False):
        super().__init__()
        ch = base
        if lite:
            layers = [wn(nn.Conv1d(1, ch, 2 * strides[0], stride=strides[0], padding=strides[0] // 2))]
            rest = strides[1:]
        else:
            layers = [wn(nn.Conv1d(1, ch, 7, padding=3))]
            rest = strides
        for s in rest:
            layers += [ResUnit(ch, 1), ResUnit(ch, 3), nn.ELU(),
                       wn(nn.Conv1d(ch, ch * 2, 2 * s, stride=s, padding=s // 2))]
            ch *= 2
        layers += [nn.ELU(), wn(nn.Conv1d(ch, latent, 3, padding=1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, strides, base, latent):
        super().__init__()
        ch = base * 2 ** len(strides)
        layers = [wn(nn.Conv1d(latent, ch, 3, padding=1))]
        for s in reversed(strides):
            layers += [nn.ELU(), wn(nn.ConvTranspose1d(ch, ch // 2, 2 * s, stride=s, padding=s // 2)),
                       ResUnit(ch // 2, 1), ResUnit(ch // 2, 3)]
            ch //= 2
        layers += [nn.ELU(), wn(nn.Conv1d(ch, 1, 7, padding=3))]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# ------------------------------------------------------------------ STFT
def _compress(X, p):
    mag = torch.sqrt(X.real ** 2 + X.imag ** 2 + 1e-12)
    return X * mag ** (p - 1)


def _decompress(X, p):
    mag = torch.sqrt(X.real ** 2 + X.imag ** 2 + 1e-12)
    return X * mag ** (1 / p - 1)


class STFTEncoder(nn.Module):
    """x (B,1,T) -> STFT complexa -> compressao de magnitude -> [re;im] como canais -> conv1d no tempo."""
    def __init__(self):
        super().__init__()
        self.n_bins = C.STFT_N_FFT // 2 + 1
        self.register_buffer("window", torch.hann_window(C.STFT_N_FFT))
        H = C.STFT_HIDDEN
        layers = [wn(nn.Conv1d(2 * self.n_bins, H, 1))]
        layers += [ResUnit(H, d) for d in C.STFT_DILATIONS]
        layers += [nn.ELU(), wn(nn.Conv1d(H, C.STFT_LATENT_CH, 3, padding=1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        X = torch.stft(x.squeeze(1), C.STFT_N_FFT, C.STFT_HOP, window=self.window, return_complex=True)
        X = _compress(X, C.STFT_COMPRESS_POWER)
        return self.net(torch.cat([X.real, X.imag], dim=1))


class STFTDecoder(nn.Module):
    """latente -> conv1d -> STFT complexa comprimida -> descompressao -> iSTFT -> onda (B,1,T)."""
    def __init__(self):
        super().__init__()
        self.n_bins = C.STFT_N_FFT // 2 + 1
        self.register_buffer("window", torch.hann_window(C.STFT_N_FFT))
        H = C.STFT_HIDDEN
        layers = [wn(nn.Conv1d(C.STFT_LATENT_CH, H, 3, padding=1))]
        layers += [ResUnit(H, d) for d in C.STFT_DILATIONS]
        layers += [nn.ELU(), wn(nn.Conv1d(H, 2 * self.n_bins, 1))]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        out = self.net(z)
        X = torch.complex(out[:, : self.n_bins], out[:, self.n_bins:])
        X = _decompress(X, C.STFT_COMPRESS_POWER)
        y = torch.istft(X, C.STFT_N_FFT, C.STFT_HOP, window=self.window, length=C.CLIP_SAMPLES)
        return y.unsqueeze(1)


# ------------------------------------------------------------------ quantizacao e wrapper
class ScalarQuant(nn.Module):
    """tanh + arredondamento uniforme em L niveis, com straight-through estimator."""
    def __init__(self, levels=C.AE_QUANT_LEVELS):
        super().__init__()
        self.L = levels

    def forward(self, z, levels: int | None = None):
        L = levels if levels is not None else self.L
        z = torch.tanh(z)
        q = torch.round((z + 1) / 2 * (L - 1))
        zq = q / (L - 1) * 2 - 1
        return z + (zq - z).detach(), q


class AEBase(nn.Module):
    extra_enc_flops = 0
    extra_dec_flops = 0

    def __init__(self, encoder, decoder, levels=C.AE_QUANT_LEVELS):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.quant = ScalarQuant(levels)

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)
        x_norm = x / rms
        zq, _ = self.quant(self.encoder(x_norm))
        y_norm = self.decoder(zq)
        return y_norm * rms

    def encode_tensor(self, x_t: torch.Tensor, levels: int | None = None):
        """Versao puramente em tensor PyTorch (usada no profiler de FLOPs)."""
        rms = torch.sqrt(torch.mean(x_t ** 2, dim=-1, keepdim=True) + 1e-8)
        x_norm = x_t / rms
        zq, q = self.quant(self.encoder(x_norm), levels=levels)
        return zq, q, rms

    def decode_tensor(self, zq: torch.Tensor, rms: torch.Tensor | float = 1.0):
        """Versao puramente em tensor PyTorch (usada no profiler de FLOPs)."""
        return self.decoder(zq) * rms

    def encode(self, x: np.ndarray, levels: int | None = None):
        """Codifica numpy array: retorna indices quantizados uint16 e ganho RMS float32."""
        dev = next(self.parameters()).device
        if x.ndim == 1:
            x_t = torch.from_numpy(x).float()[None, None].to(dev)
        else:
            torch_x = torch.from_numpy(x).float().to(dev)
            x_t = torch_x if torch_x.ndim == 3 else torch_x.unsqueeze(0)

        rms_val = float(torch.sqrt(torch.mean(x_t ** 2)).item())
        x_norm = x_t / (rms_val + 1e-8)

        with torch.no_grad():
            _, q = self.quant(self.encoder(x_norm), levels=levels)

        return q.cpu().numpy().astype(np.uint16), rms_val

    def decode(self, q_payload: np.ndarray, rms_val: float, levels: int | None = None):
        """Decodifica a partir dos indices latentes e do ganho RMS transmitido."""
        dev = next(self.parameters()).device
        L = levels if levels is not None else self.quant.L
        q = torch.from_numpy(q_payload.astype(np.float32)).to(dev)
        if q.ndim == 3:
            q = q.unsqueeze(0)

        zq = q / (L - 1) * 2 - 1
        with torch.no_grad():
            y_norm = self.decoder(zq)

        y = y_norm * rms_val
        return y.cpu().numpy()[0, 0]


def build_autoencoder(variant: str, levels: int = C.AE_QUANT_LEVELS) -> AEBase:
    if variant == "wave":
        s = C.AE_WAVE_STRIDES
        return AEBase(Encoder(s, C.AE_BASE_CH, C.AE_LATENT_CH),
                      Decoder(s, C.AE_BASE_CH, C.AE_LATENT_CH), levels=levels)
    if variant == "wave_lite":
        s = C.AE_LITE_STRIDES
        return AEBase(Encoder(s, C.AE_BASE_CH, C.AE_LATENT_CH, lite=True),
                      Decoder(s, C.AE_BASE_CH, C.AE_LATENT_CH), levels=levels)
    if variant == "stft":
        m = AEBase(STFTEncoder(), STFTDecoder(), levels=levels)
        frames = 1 + C.CLIP_SAMPLES // C.STFT_HOP
        fft = 2.5 * C.STFT_N_FFT * math.log2(C.STFT_N_FFT)
        m.extra_enc_flops = m.extra_dec_flops = int(frames * fft)
        return m
    raise ValueError(variant)


def strip_weight_norm(model):
    for m in list(model.modules()):
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight")
    return model


# ------------------------------------------------------------------ treino
def mrstft_loss(x, y):
    x, y, loss = x.squeeze(1), y.squeeze(1), 0.0
    for n in C.AE_STFT_SIZES:
        w = torch.hann_window(n, device=x.device)
        X = torch.stft(x, n, n // 4, window=w, return_complex=True).abs()
        Y = torch.stft(y, n, n // 4, window=w, return_complex=True).abs()
        loss = loss + (torch.log(X + 1e-5) - torch.log(Y + 1e-5)).abs().mean() \
            + torch.norm(Y - X) / (torch.norm(Y) + 1e-8)
    return loss / len(C.AE_STFT_SIZES)


class ClipDS(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from data import load_clip
        r = self.df.iloc[i]
        return torch.from_numpy(load_clip(r["path"], r["start"]))[None]


def ckpt_path(dataset, variant):
    return C.OUT_DIR / f"ae_{dataset}_{variant}.pt"


def train(dataset, variant, resume=False):
    torch.manual_seed(C.SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clips = pd.read_csv(C.OUT_DIR / f"clips_{dataset}.csv")
    dl_tr = torch.utils.data.DataLoader(ClipDS(clips[clips.split == "train"]), C.AE_BATCH, shuffle=True,
                                        num_workers=4, drop_last=True)
    dl_va = torch.utils.data.DataLoader(ClipDS(clips[clips.split == "val"]), C.AE_BATCH, num_workers=4)
    model = build_autoencoder(variant, levels=C.AE_QUANT_LEVELS).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=C.AE_LR)

    best, start_ep = float("inf"), 0
    ckpt = ckpt_path(dataset, variant)
    last_ckpt = C.OUT_DIR / f"ae_{dataset}_{variant}_last.pt"

    if resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=dev)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_ep = state["epoch"] + 1
        best = state["best_val"]
        print(f"[{variant}] Retomando treino da epoca {start_ep} (best_val={best:.4f})")

    for ep in range(start_ep, C.AE_EPOCHS):
        model.train()
        for x in dl_tr:
            x = x.to(dev)
            y = model(x)
            loss = (x - y).abs().mean() + mrstft_loss(x, y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); v = []
        with torch.no_grad():
            for x in dl_va:
                x = x.to(dev); y = model(x)
                v.append(((x - y).abs().mean() + mrstft_loss(x, y)).item())
        v = float(np.mean(v)); print(f"[{variant}] epoch {ep}  val_loss {v:.4f}")

        if v < best:
            best = v
            torch.save(model.state_dict(), ckpt)

        # Salva progresso da epoca atual para permitir retomada exata
        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "best_val": best,
        }, last_ckpt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", required=True, choices=C.AE_VARIANTS)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    train(a.dataset, a.variant, resume=a.resume)