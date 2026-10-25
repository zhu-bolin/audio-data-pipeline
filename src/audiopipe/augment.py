"""Augmentations applied after room simulation."""

from __future__ import annotations

import numpy as np


def colored_noise(shape: tuple[int, int], color: str, rng: np.random.Generator) -> np.ndarray:
    """Independent noise per channel. 'white', 'pink' (1/f), or 'brown' (1/f^2)."""
    white = rng.standard_normal(shape)
    if color == "white":
        return white
    exponent = {"pink": 0.5, "brown": 1.0}[color]
    spec = np.fft.rfft(white, axis=-1)
    freqs = np.maximum(np.arange(spec.shape[-1]), 1)
    return np.fft.irfft(spec / freqs**exponent, n=shape[-1], axis=-1)


def add_noise(x: np.ndarray, snr_db: float, color: str, rng: np.random.Generator):
    """Scale noise so the mixture hits the target SNR. Returns (mixture, achieved SNR in dB)."""
    noise = colored_noise(x.shape, color, rng)
    p_sig = np.mean(x**2)
    p_noise = np.mean(noise**2)
    noise *= np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    achieved = 10 * np.log10(p_sig / np.mean(noise**2))
    return (x + noise).astype(np.float32), float(achieved)


def apply_gain(x: np.ndarray, gain_db: float) -> np.ndarray:
    return (x * 10 ** (gain_db / 20)).astype(np.float32)


def peak_limit(x: np.ndarray, ceiling: float = 0.99) -> tuple[np.ndarray, float]:
    """Scale down only if the clip would clip. Returns (audio, scale factor applied)."""
    peak = float(np.max(np.abs(x)))
    if peak <= ceiling:
        return x, 1.0
    s = ceiling / peak
    return (x * s).astype(np.float32), s
