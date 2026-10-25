"""End-to-end run: sources -> room simulation -> augmentation -> features -> catalog.

Usage:
    python -m audiopipe.pipeline --out data --n-rooms 20 --clips-per-room 3
    python -m audiopipe.pipeline --input-dir path/to/ESC-50/audio --db-url postgresql://user:pw@localhost/audio
"""

from __future__ import annotations

import argparse
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import soundfile as sf
from sqlalchemy import func, select

from . import augment, features, simulate, sources
from .db import Clip, Room, Run, Source, get_session


def assign_split(room_index: int, seed: int) -> str:
    """Split by ROOM, not by clip, so the test set contains acoustic environments the model never saw."""
    r = np.random.default_rng([seed, room_index]).random()
    return "train" if r < 0.7 else "val" if r < 0.85 else "test"


def run(
    out_dir: str = "data",
    db_url: str | None = None,
    input_dir: str | None = None,
    n_rooms: int = 10,
    clips_per_room: int = 3,
    sr: int = 16000,
    duration: float = 2.0,
    snr_range: tuple[float, float] = (0.0, 30.0),
    rt60_range: tuple[float, float] = (0.2, 0.8),
    seed: int = 0,
    workers: int = 1,
) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "features").mkdir(parents=True, exist_ok=True)
    db_url = db_url or f"sqlite:///{(out / 'catalog.db').resolve()}"
    rng = np.random.default_rng(seed)
    config = dict(
        input_dir=input_dir, n_rooms=n_rooms, clips_per_room=clips_per_room, sr=sr,
        duration=duration, snr_range=list(snr_range), rt60_range=list(rt60_range),
    )

    srcs = sources.load_directory(input_dir, sr, duration) if input_dir else sources.synthesize(sr, duration, rng)
    if not srcs:
        raise SystemExit(f"No usable audio found in {input_dir}")

    session = get_session(db_url)
    run_row = Run(seed=seed, config=config)
    session.add(run_row)
    session.flush()

    src_rows = []
    for s in srcs:
        row = Source(run_id=run_row.id, name=s.name, kind=s.kind, origin=s.origin, sample_rate=sr)
        session.add(row)
        src_rows.append(row)
    session.flush()

    n_len = int(sr * duration)

    # Loop 1 (main process): make every random choice up front, in a fixed order,
    # so the dataset depends only on the seed, never on how many workers run.
    plans = []
    for ri in range(n_rooms):
        spec = simulate.random_room(rng, rt60_range)
        room_row = Room(run_id=run_row.id, dim_x=spec.dims[0], dim_y=spec.dims[1],
                        dim_z=spec.dims[2], rt60_target=spec.rt60_target)
        session.add(room_row)
        session.flush()
        split = assign_split(ri, seed)

        for _ in range(clips_per_room):
            si = int(rng.integers(len(srcs)))
            # New source position per clip so DOA/distance labels vary within a room
            spec = simulate.move_source(spec, rng)
            plans.append(dict(
                si=si, spec=spec, room_row=room_row, split=split,
                snr_t=float(rng.uniform(*snr_range)),
                color=str(rng.choice(["white", "pink", "brown"])),
                gain=float(rng.uniform(-12, 6)),
                noise_seed=int(rng.integers(2**32)),  # each clip gets its own noise generator
            ))

    # The slow part: room simulation, spread across worker processes.
    jobs = [(srcs[p["si"]].audio, sr, p["spec"]) for p in plans]
    if workers > 1:
        # "spawn" starts fresh worker processes, the same way Windows always does
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.starmap(simulate.simulate, jobs)
    else:
        results = [simulate.simulate(*job) for job in jobs]

    # Loop 2 (main process): augment, check, save, and catalog each simulated clip.
    for p, (wet, rt60_meas) in zip(plans, results):
        si, spec, room_row, split = p["si"], p["spec"], p["room_row"], p["split"]
        snr_t, color, gain = p["snr_t"], p["color"], p["gain"]
        x, snr_a = augment.add_noise(wet, snr_t, color, np.random.default_rng(p["noise_seed"]))
        x = augment.apply_gain(x, gain)
        x, scale = augment.peak_limit(x)

        issues = features.quality_checks(x, sr, n_len, x.shape[0])
        clip = Clip(
            run_id=run_row.id, source_id=src_rows[si].id, room_id=room_row.id, split=split,
            audio_path="", features_path="", n_channels=x.shape[0], array_type=spec.array_type,
            azimuth_deg=round(spec.azimuth_deg, 2), distance_m=round(spec.distance, 3),
            rt60_measured=round(rt60_meas, 3), snr_target_db=round(snr_t, 2),
            snr_achieved_db=round(snr_a, 2), noise_color=color, gain_db=round(gain, 2),
            limiter_scale=round(scale, 4), stats=features.summary_stats(x, sr),
            passed_qc=not issues, qc_issues=issues,
        )
        session.add(clip)
        session.flush()
        stem = f"run{run_row.id:03d}_clip{clip.id:05d}"
        clip.audio_path = str(out / "audio" / f"{stem}.wav")
        clip.features_path = str(out / "features" / f"{stem}_logmel.npy")
        sf.write(clip.audio_path, x.T, sr, subtype="FLOAT")
        np.save(clip.features_path, features.log_mel(x, sr))

    session.commit()
    summary = report(session, run_row.id)
    summary["seconds"] = round(time.time() - t0, 1)
    summary["db_url"] = db_url
    session.close()
    return summary


def report(session, run_id: int) -> dict:
    q = select(Clip).where(Clip.run_id == run_id)
    clips = session.scalars(q).all()
    snr_err = [abs(c.snr_achieved_db - c.snr_target_db) for c in clips]
    splits = dict(session.execute(
        select(Clip.split, func.count()).where(Clip.run_id == run_id).group_by(Clip.split)
    ).all())
    rooms = session.scalars(select(Room).where(Room.run_id == run_id)).all()
    rt_pairs = {}
    for c in clips:
        rt_pairs.setdefault(c.room_id, []).append(c.rt60_measured)
    rt_err = [abs(np.mean(rt_pairs[r.id]) - r.rt60_target) for r in rooms if r.id in rt_pairs]
    return {
        "run_id": run_id,
        "clips": len(clips),
        "rooms": len(rooms),
        "passed_qc": sum(c.passed_qc for c in clips),
        "splits": splits,
        "max_snr_error_db": round(max(snr_err), 3),
        "mean_rt60_error_s": round(float(np.mean(rt_err)), 3),
    }


def main():
    p = argparse.ArgumentParser(description="Simulate and catalog multichannel audio for ML experiments.")
    p.add_argument("--out", default="data")
    p.add_argument("--db-url", default=None, help="SQLAlchemy URL; defaults to SQLite in --out")
    p.add_argument("--input-dir", default=None, help="Folder of dry recordings; omit to use synthetic sources")
    p.add_argument("--n-rooms", type=int, default=10)
    p.add_argument("--clips-per-room", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1, help="Processes for room simulation")
    a = p.parse_args()
    s = run(a.out, a.db_url, a.input_dir, a.n_rooms, a.clips_per_room, seed=a.seed, workers=a.workers)
    for k, v in s.items():
        print(f"{k:>18}: {v}")


if __name__ == "__main__":
    main()