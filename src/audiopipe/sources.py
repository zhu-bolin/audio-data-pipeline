"""Dry (anechoic) source signals: load real recordings or synthesize test signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass
class SourceSignal:
    name: str
    kind: str  # "file" or a synthetic generator name
    audio: np.ndarray  # mono float32, peak-normalized
    sr: int
    origin: str  # file path or generator description


def _normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = np.max(np.abs(x))
    return (x / m * peak).astype(np.float32) if m > 0 else x.astype(np.float32)


def load_directory(path: str | Path, sr: int, duration: float) -> list[SourceSignal]:
    """Load every .wav/.flac/.mp3 in a folder (e.g. ESC-50 or LibriSpeech), mono, resampled, trimmed/padded."""
    files = sorted(p for p in Path(path).rglob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3"})
    n = int(sr * duration)
    out = []
    for f in files:
        y, _ = librosa.load(f, sr=sr, mono=True)
        y = np.pad(y, (0, max(0, n - len(y))))[:n]
        if np.max(np.abs(y)) < 1e-4:  # skip silent files
            continue
        out.append(SourceSignal(f.stem, "file", _normalize(y), sr, str(f)))
    return out


def synthesize(sr: int, duration: float, rng: np.random.Generator) -> list[SourceSignal]:
    """Built-in test signals so the pipeline runs with no downloads."""
    t = np.arange(int(sr * duration)) / sr
    f0 = rng.uniform(110, 220)

    # Harmonic tone with vibrato, a rough stand-in for a sung or spoken voiced sound
    vib = 1 + 0.01 * np.sin(2 * np.pi * 5 * t)
    phase = 2 * np.pi * f0 * np.cumsum(vib) / sr
    harmonic = sum((0.6 ** k) * np.sin(k * phase) for k in range(1, 9))
    harmonic *= np.minimum(1, t / 0.05) * np.minimum(1, (t[-1] - t) / 0.1)

    # Log sweep 100 Hz to 7 kHz, standard for measuring room response
    f1, f2 = 100.0, 7000.0
    k = np.log(f2 / f1)
    sweep = np.sin(2 * np.pi * f1 * duration / k * (np.exp(t / duration * k) - 1))

    # Bursts of filtered noise, a stand-in for transient events (claps, door knocks)
    bursts = np.zeros_like(t)
    for start in rng.uniform(0, duration - 0.2, size=4):
        i = int(start * sr)
        seg = rng.standard_normal(int(0.08 * sr)) * np.exp(-np.linspace(0, 6, int(0.08 * sr)))
        bursts[i : i + len(seg)] += seg

    # Speech-like amplitude modulation (~4 Hz syllable rate) on pink-ish noise
    white = rng.standard_normal(len(t))
    spec = np.fft.rfft(white) / np.sqrt(np.maximum(np.fft.rfftfreq(len(t), 1 / sr), 20))
    pink = np.fft.irfft(spec, n=len(t))
    speechlike = pink * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t + rng.uniform(0, 2 * np.pi))) ** 2

    return [
        SourceSignal("harmonic", "synthetic", _normalize(harmonic), sr, f"harmonic f0={f0:.1f}Hz"),
        SourceSignal("log_sweep", "synthetic", _normalize(sweep), sr, "log sweep 100-7000Hz"),
        SourceSignal("noise_bursts", "synthetic", _normalize(bursts), sr, "4 decaying noise bursts"),
        SourceSignal("speechlike", "synthetic", _normalize(speechlike), sr, "4Hz-modulated pink noise"),
    ]
