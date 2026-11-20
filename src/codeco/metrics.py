"""Signal reconstruction metrics shared by the FFmpeg evaluation phase."""

from __future__ import annotations

import librosa
import numpy as np
from skimage.metrics import structural_similarity

from codeco.acoustic_indices import acoustic_indices_v1


SR = 48_000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 0
FMAX = SR // 2
MEL_DB_FLOOR = -100.0


def mel_db(x: np.ndarray) -> np.ndarray:
    spectrum = librosa.feature.melspectrogram(
        y=x,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    return np.maximum(
        librosa.power_to_db(spectrum, ref=1.0, top_db=None),
        MEL_DB_FLOOR,
    )


def mel_similarity(reference: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    reference_mel = mel_db(reference)
    decoded_mel = mel_db(decoded)
    data_range = -MEL_DB_FLOOR
    reference_vector = (reference_mel - MEL_DB_FLOOR).ravel()
    decoded_vector = (decoded_mel - MEL_DB_FLOOR).ravel()
    return {
        "mel_mae_db": float(np.mean(np.abs(reference_mel - decoded_mel))),
        "mel_rmse_db": float(np.sqrt(np.mean((reference_mel - decoded_mel) ** 2))),
        "mel_ssim": float(
            structural_similarity(reference_mel, decoded_mel, data_range=data_range)
        ),
        "mel_cosine": float(
            reference_vector @ decoded_vector
            / (np.linalg.norm(reference_vector) * np.linalg.norm(decoded_vector) + 1e-12)
        ),
    }


def snr_db(reference: np.ndarray, decoded: np.ndarray) -> float:
    error = np.sum((reference - decoded) ** 2)
    return float(
        10
        * np.log10(
            (np.sum(reference**2) + 1e-20)
            / (error + 1e-20)
        )
    )


def reconstruction_metrics(reference: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    return {
        "snr_db": snr_db(reference, decoded),
        **mel_similarity(reference, decoded),
        **acoustic_indices_v1(decoded),
    }
