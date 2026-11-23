"""Auditable, signed and normalized temporal alignment for reconstructed audio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class AlignmentResult:
    audio: np.ndarray
    lag_samples: int
    score: float


def normalized_cross_correlation_align(
    reference: np.ndarray,
    decoded: np.ndarray,
    max_lag: int,
) -> AlignmentResult:
    """Align ``decoded`` to ``reference`` using signed normalized correlation.

    Positive lag means the decoded signal starts later and must be cropped on
    the left. Negative lag means it starts earlier and is padded on the left.
    The score is normalized by the overlap energy, avoiding the amplitude bias
    of an unnormalized argmax.
    """
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    decoded = np.asarray(decoded, dtype=np.float64).reshape(-1)
    if reference.size == 0 or decoded.size == 0:
        raise ValueError("Alignment requires two non-empty signals.")
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative.")

    ref = reference - reference.mean()
    dec = decoded - decoded.mean()
    numerator = signal.correlate(dec, ref, mode="full", method="fft")
    lags = signal.correlation_lags(dec.size, ref.size, mode="full")
    dec_energy = signal.correlate(
        dec * dec, np.ones(ref.size, dtype=np.float64), mode="full", method="fft"
    )
    ref_energy = signal.correlate(
        np.ones(dec.size, dtype=np.float64), ref * ref, mode="full", method="fft"
    )
    denominator = np.sqrt(np.maximum(dec_energy * ref_energy, 0.0))
    normalized = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, -np.inf),
        where=denominator > np.finfo(np.float64).eps,
    )
    allowed = np.abs(lags) <= min(max_lag, reference.size - 1)
    if not allowed.any():
        raise ValueError("No valid lag remains inside the requested range.")
    allowed_indices = np.flatnonzero(allowed)
    best_index = int(allowed_indices[np.argmax(normalized[allowed])])
    lag = int(lags[best_index])

    target = reference.size
    if lag >= 0:
        aligned = decoded[lag : lag + target]
        if aligned.size < target:
            aligned = np.pad(aligned, (0, target - aligned.size))
    else:
        aligned = np.pad(decoded, (-lag, 0))[:target]
        if aligned.size < target:
            aligned = np.pad(aligned, (0, target - aligned.size))
    return AlignmentResult(
        audio=aligned.astype(np.float32),
        lag_samples=lag,
        score=float(normalized[best_index]),
    )
