"""Versioned implementation of ``configs/acoustic-indices-v1.yaml``.

The same functions are used for the reference and every decoded condition.
No gain normalization is applied after decoding.
"""
from __future__ import annotations

import numpy as np
from scipy.signal.windows import hann


VERSION = "1.0.0"
SR = 48_000
N_FFT = 1024
HOP = 512
EPS = 1e-12


def _frames(x: np.ndarray, frame_length: int = N_FFT, hop: int = HOP) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) < frame_length:
        return np.empty((0, frame_length), dtype=np.float64)
    view = np.lib.stride_tricks.sliding_window_view(x, frame_length)
    return view[::hop]


def _spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = _frames(x)
    window = hann(N_FFT, sym=False)
    raw = np.fft.rfft(frames * window, axis=1)
    magnitude = np.abs(raw)
    amplitude = magnitude * (2.0 / window.sum())
    amplitude[:, 0] *= 0.5
    amplitude[:, -1] *= 0.5
    amplitude_dbfs = 20.0 * np.log10(np.maximum(amplitude, EPS))
    power = magnitude**2
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    return freqs, magnitude.T, amplitude_dbfs.T, power.T


def _entropy(values: np.ndarray) -> float:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = values.sum()
    if total <= 0 or len(values) <= 1:
        return 0.0
    p = values / total
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


def _gini(values: np.ndarray) -> float:
    values = np.sort(np.maximum(np.asarray(values, dtype=np.float64), 0.0))
    if values.size == 0 or values.sum() == 0:
        return 0.0
    n = len(values)
    return float((2 * np.sum((np.arange(n) + 1) * values) / (n * values.sum())) - (n + 1) / n)


def _band(freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    return (freqs >= low) & (freqs < high)


def _aci(x: np.ndarray) -> float:
    block_samples = SR
    values = []
    for start in range(0, len(x), block_samples):
        block = x[start : start + block_samples]
        if len(block) != block_samples:
            continue
        freqs, magnitude, _, _ = _spectrum(block)
        selected = magnitude[_band(freqs, 500, 24_000)]
        numerator = np.abs(np.diff(selected, axis=1)).sum(axis=1)
        denominator = selected.sum(axis=1)
        values.append(float(np.sum(numerator / np.maximum(denominator, EPS))))
    return float(np.mean(values)) if values else 0.0


def _occupancies(freqs: np.ndarray, amplitude_dbfs: np.ndarray) -> np.ndarray:
    occupancies = []
    for low in range(0, 24_000, 1000):
        values = amplitude_dbfs[_band(freqs, low, low + 1000)]
        occupancies.append(float(np.mean(values > -50.0)) if values.size else 0.0)
    return np.asarray(occupancies)


def _bio(freqs: np.ndarray, amplitude_dbfs: np.ndarray, low: float, high: float) -> float:
    mask = _band(freqs, low, high)
    mean_amplitude = np.mean(10.0 ** (amplitude_dbfs[mask] / 20.0), axis=1)
    mean_db = 20.0 * np.log10(np.maximum(mean_amplitude, EPS))
    above_baseline = mean_db - mean_db.min()
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(above_baseline, freqs[mask] / 1000.0))
    return float(np.trapz(above_baseline, freqs[mask] / 1000.0))


def _ndsi(freqs: np.ndarray, power: np.ndarray, bio_high: float) -> float:
    mean_power = power.mean(axis=1)
    anthropophony = mean_power[_band(freqs, 1000, 2000)].sum()
    biophony = mean_power[_band(freqs, 2000, bio_high)].sum()
    denominator = biophony + anthropophony
    return float((biophony - anthropophony) / denominator) if denominator > 0 else 0.0


def acoustic_indices_v1(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) != 4 * SR:
        raise ValueError(f"acoustic-indices-v1 expects exactly {4 * SR} samples; received {len(x)}")
    freqs, magnitude, amplitude_dbfs, power = _spectrum(x)
    occupancies = _occupancies(freqs, amplitude_dbfs)

    rms_frames = np.sqrt(np.mean(_frames(x) ** 2, axis=1))
    ht = _entropy(rms_frames)
    hf_mask = _band(freqs, 500, 24_000)
    hf = _entropy(power[hf_mask].mean(axis=1))

    return {
        "idx_ACI": _aci(x),
        "idx_ADI": _entropy(occupancies),
        "idx_AEI": _gini(occupancies),
        "idx_Ht": ht,
        "idx_Hf": hf,
        "idx_H": ht * hf,
        "idx_BIO_2_24": _bio(freqs, amplitude_dbfs, 2000, 24_000),
        "idx_BIO_2_8": _bio(freqs, amplitude_dbfs, 2000, 8000),
        "idx_NDSI_CLASSIC": _ndsi(freqs, power, 11_000),
        "idx_NDSI_EXTENDED": _ndsi(freqs, power, 24_000),
    }
