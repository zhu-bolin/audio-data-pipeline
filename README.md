# audio-data-pipeline

A reproducible pipeline that turns dry audio into a labeled, multichannel training set for audio ML experiments: it simulates rooms and microphone arrays, augments the results, extracts features, runs quality checks, and catalogs everything in a relational database.

```
dry sources ──► room simulation ──► augmentation ──► QC + features ──► catalog (PostgreSQL / SQLite)
 (files or        (pyroomacoustics,     (colored noise at     (log-mel, MFCC,       runs, sources,
  synthetic)       4-mic arrays)         target SNR, gain,     clipping/silence/     rooms, clips
                                         peak limiting)        dead-channel checks)  + .wav / .npy files
```

## What it does

- **Sources.** Loads any folder of recordings (ESC-50, UrbanSound8K, LibriSpeech), or generates four synthetic test signals (harmonic tone, log sweep, noise bursts, speech-like modulated noise) so it runs with no downloads.
- **Room simulation.** Samples realistic shoebox rooms (3–10 m, RT60 0.2–0.8 s), places a 4-mic circular or linear array at table height and a talker at least 1 m away, and simulates reverberant multichannel audio with the image source method.
- **Labels.** Each clip records direction of arrival (azimuth), source distance, and **measured** RT60, which are the targets for tasks like source localization, dereverberation, and room-acoustics estimation.
- **Augmentation.** Adds white, pink, or brown noise scaled to an exact target SNR (0–30 dB), applies random gain, and peak-limits to prevent clipping. Every parameter is stored.
- **Quality control.** Flags wrong shape, non-finite samples, clipping, near-silence, and dead channels.
- **Catalog.** SQLAlchemy models (`runs`, `sources`, `rooms`, `clips`) work unchanged on PostgreSQL or SQLite, so you can query the dataset ("train clips with SNR < 10 dB in rooms with RT60 > 0.5 s").
- **Leakage-safe splits.** Train/val/test is assigned **per room**, not per clip, so test rooms are acoustic environments the model never saw.
- **Reproducibility.** One seed drives everything; the run's config is stored with its clips.

## Run it

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m audiopipe.pipeline --out data --n-rooms 20 --clips-per-room 3
PYTHONPATH=src python -m pytest -q
```

With your own recordings and PostgreSQL:

```bash
PYTHONPATH=src python -m audiopipe.pipeline --input-dir ~/datasets/ESC-50/audio \
  --db-url postgresql://user:password@localhost/audio --n-rooms 200
```

Example output (60 clips, 20 rooms, about 12 s on a laptop-class CPU):

```
 clips: 60   rooms: 20   passed_qc: 60
splits: {'train': 42, 'val': 6, 'test': 12}
max_snr_error_db: 0.0
mean_rt60_error_s: 0.155
```

## A finding worth knowing

Rooms are configured with Sabine's formula to hit a target RT60, but the RT60 measured from the simulated impulse responses comes out consistently **higher** than the target (mean gap about 0.15 s). Sabine assumes a perfectly diffuse sound field, which shoebox image-source simulation doesn't produce. That's why the catalog stores both values and uses the **measured** RT60 as the label. Training on the target value would teach a model the wrong thing.

## Ideas to extend

- Snowflake or S3 backends for the catalog and audio files
- Real room impulse responses (e.g. from public RIR datasets) alongside simulated ones
- A small baseline model: predict azimuth from the log-mel features of the 4 channels
- Parallel simulation with `multiprocessing` for large runs

## Results on ESC-50

Ran on the ESC-50 environmental sound dataset (2,000 clips, 50 classes):
50 rooms × 4 clips = 200 multichannel clips in 30.5 s on a Windows laptop.

- 200/200 clips passed QC
- SNR error: 0.0 dB (noise scaled exactly to target)
- Mean RT60 gap: 0.165 s, consistent with the 0.155 s gap on synthetic sources
- Room-level splits: 140 train / 24 val / 36 test