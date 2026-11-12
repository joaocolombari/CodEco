"""Espectrogramas mel, similaridade espectral, SNR e indices acusticos (scikit-maad)."""
import numpy as np
import librosa
from skimage.metrics import structural_similarity

import config as C


def mel_db(x: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=x, sr=C.SR, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
                                       n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX, power=2.0)
    return np.maximum(librosa.power_to_db(S, ref=1.0, top_db=None), C.MEL_DB_FLOOR)


def mel_similarity(ref: np.ndarray, dec: np.ndarray) -> dict:
    A, B = mel_db(ref), mel_db(dec)
    rng = -C.MEL_DB_FLOOR
    a, b = (A - C.MEL_DB_FLOOR).ravel(), (B - C.MEL_DB_FLOOR).ravel()
    return dict(
        mel_mae_db=float(np.mean(np.abs(A - B))),
        mel_rmse_db=float(np.sqrt(np.mean((A - B) ** 2))),
        mel_ssim=float(structural_similarity(A, B, data_range=rng)),
        mel_cosine=float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
    )


def snr_db(ref, dec) -> float:
    err = np.sum((ref - dec) ** 2)
    return float(10 * np.log10((np.sum(ref ** 2) + 1e-20) / (err + 1e-20)))


def acoustic_indices(x: np.ndarray) -> dict:
    """Indices temporais e espectrais do scikit-maad (prefixo idx_). Conferir nomes na sua versao."""
    from maad import features, sound
    s = x.astype(np.float64)
    out_t = features.all_temporal_alpha_indices(s, C.SR)
    Sxx, tn, fn, _ = sound.spectrogram(s, C.SR, nperseg=C.MAAD_NPERSEG, noverlap=C.MAAD_NOVERLAP)
    out_s = features.all_spectral_alpha_indices(Sxx, tn, fn, flim_low=list(C.MAAD_FLIM_LOW),
                                                flim_mid=list(C.MAAD_FLIM_MID), flim_hi=list(C.MAAD_FLIM_HI))
    df_s = out_s[0] if isinstance(out_s, tuple) else out_s
    d = {**out_t.iloc[0].to_dict(), **df_s.iloc[0].to_dict()}
    return {f"idx_{k}": float(v) for k, v in d.items() if np.isscalar(v)}
