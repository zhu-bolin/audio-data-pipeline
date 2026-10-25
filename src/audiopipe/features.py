"""Feature extraction and data-quality checks."""

from __future__ import annotations

import librosa
import numpy as np


def log_mel(x: np.ndarray, sr: int, n_mels: int = 64) -> np.ndarray:
    """Log-mel spectrogram per channel: [M, n_mels, frames]. The usual input for audio ML models."""
    mel = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=512, hop_length=160, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=1.0, top_db=None).astype(np.float32)


def summary_stats(x: np.ndarray, sr: int) -> dict:
    """Scalar descriptors stored in the database for filtering and dataset analysis."""
    ref = x[0]  # first channel
    rms = librosa.feature.rms(y=ref)[0]
    centroid = librosa.feature.spectral_centroid(y=ref, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=13)
    return {
        "rms_dbfs": float(20 * np.log10(np.mean(rms) + 1e-12)),
        "peak_dbfs": float(20 * np.log10(np.max(np.abs(x)) + 1e-12)),
        "spectral_centroid_hz": float(np.mean(centroid)),
        "mfcc_mean": [round(float(v), 4) for v in mfcc.mean(axis=1)],
    }


def quality_checks(x: np.ndarray, sr: int, expected_len: int, n_channels: int) -> list[str]:
    """Return a list of problems; empty means the clip passed."""
    problems = []
    if x.shape != (n_channels, expected_len):
        problems.append(f"shape {x.shape} != {(n_channels, expected_len)}")
    if not np.all(np.isfinite(x)):
        problems.append("non-finite samples")
    if np.max(np.abs(x)) >= 1.0:
        problems.append("clipping")
    if np.max(np.abs(x)) < 1e-3:
        problems.append("near-silent")
    if np.any(np.std(x, axis=1) < 1e-6):
        problems.append("dead channel")
    return problems
