"""Frozen signal protocol and model-specific input adapters.

The codec experiment always operates on the same four-second, mono, 48 kHz,
PCM16-equivalent signal. Foundation-model adaptation happens only after codec
decoding, so padding or cropping never changes the measured payload.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Protocol:
    sample_rate_hz: int = 48_000
    bit_depth: int = 16
    channels: int = 1
    window_seconds: float = 4.0
    segment_hop_seconds: float = 1.0
    birdnet_seconds: float = 3.0
    birdnet_sample_rate_hz: int = 48_000
    perch_seconds: float = 5.0
    perch_sample_rate_hz: int = 32_000
    perch_padding: str = "symmetric_zero"

    @property
    def window_samples(self) -> int:
        return round(self.sample_rate_hz * self.window_seconds)

    @property
    def segment_hop_samples(self) -> int:
        return round(self.sample_rate_hz * self.segment_hop_seconds)

    @property
    def overlap_fraction(self) -> float:
        return 1.0 - self.segment_hop_seconds / self.window_seconds

    @property
    def pcm_payload_bytes(self) -> int:
        return self.window_samples * self.channels * self.bit_depth // 8

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PROTOCOL = Protocol()


def sliding_window_starts(
    duration_seconds: float,
    protocol: Protocol = PROTOCOL,
) -> list[float]:
    """Return complete-window starts for future continuous-recording manifests."""
    if duration_seconds < protocol.window_seconds:
        return []
    count = int(np.floor((duration_seconds - protocol.window_seconds) / protocol.segment_hop_seconds)) + 1
    return [index * protocol.segment_hop_seconds for index in range(count)]


def center_crop_or_pad(x: np.ndarray, target_samples: int) -> np.ndarray:
    """Return an exact centered view, padding both sides with zeros if needed."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) > target_samples:
        start = (len(x) - target_samples) // 2
        return x[start : start + target_samples]
    if len(x) < target_samples:
        missing = target_samples - len(x)
        left = missing // 2
        return np.pad(x, (left, missing - left), mode="constant")
    return x


def _resample(x: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return np.asarray(x, dtype=np.float32)
    try:
        import soxr
    except ImportError as exc:  # pragma: no cover - dependency error is explicit
        raise RuntimeError("Install `soxr` to adapt audio sample rates.") from exc
    return soxr.resample(x, source_sr, target_sr, quality="HQ").astype(np.float32)


def adapt_for_birdnet(
    x: np.ndarray,
    source_sr: int = PROTOCOL.sample_rate_hz,
) -> np.ndarray:
    """Convert a canonical four-second window to BirdNET's centered 3 s input."""
    y = _resample(x, source_sr, PROTOCOL.birdnet_sample_rate_hz)
    target = round(PROTOCOL.birdnet_seconds * PROTOCOL.birdnet_sample_rate_hz)
    return center_crop_or_pad(y, target)


def adapt_for_perch(
    x: np.ndarray,
    source_sr: int = PROTOCOL.sample_rate_hz,
) -> np.ndarray:
    """Convert 4 s to Perch's 5 s input using 0.5 s zero-padding per side.

    Resampling happens before padding. The same adapter must be applied to the
    original and every reconstructed condition.
    """
    expected = round(PROTOCOL.window_seconds * source_sr)
    if len(x) != expected:
        raise ValueError(
            f"Perch adapter expects the canonical {PROTOCOL.window_seconds:g} s "
            f"window ({expected} samples at {source_sr} Hz); received {len(x)}."
        )
    y = _resample(x, source_sr, PROTOCOL.perch_sample_rate_hz)
    target = round(PROTOCOL.perch_seconds * PROTOCOL.perch_sample_rate_hz)
    return center_crop_or_pad(y, target)


def adapter_audit() -> dict[str, Any]:
    """Return lengths and padding used by the two foundation-model adapters."""
    x = np.ones(PROTOCOL.window_samples, dtype=np.float32)
    birdnet = adapt_for_birdnet(x)
    perch = adapt_for_perch(x)
    nonzero = np.flatnonzero(perch)
    left = int(nonzero[0]) if nonzero.size else len(perch)
    right = int(len(perch) - nonzero[-1] - 1) if nonzero.size else 0
    return {
        "canonical_samples": len(x),
        "birdnet_samples": len(birdnet),
        "birdnet_seconds": len(birdnet) / PROTOCOL.birdnet_sample_rate_hz,
        "perch_samples": len(perch),
        "perch_seconds": len(perch) / PROTOCOL.perch_sample_rate_hz,
        "perch_left_padding_samples": left,
        "perch_right_padding_samples": right,
        "perch_padding_seconds_per_side": left / PROTOCOL.perch_sample_rate_hz,
    }
