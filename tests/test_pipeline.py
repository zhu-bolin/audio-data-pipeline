import numpy as np
import soundfile as sf
from sqlalchemy import select

from audiopipe import augment, features, simulate
from audiopipe.db import Clip, get_session
from audiopipe.pipeline import run


def test_noise_hits_target_snr():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 16000)).astype(np.float32)
    for color in ["white", "pink", "brown"]:
        _, achieved = augment.add_noise(x, 10.0, color, rng)
        assert abs(achieved - 10.0) < 1e-3


def test_peak_limiter_prevents_clipping():
    y, scale = augment.peak_limit(np.array([[0.5, -2.0, 1.5]], dtype=np.float32))
    assert np.max(np.abs(y)) <= 0.99 and scale < 1


def test_room_geometry_is_valid():
    rng = np.random.default_rng(2)
    for _ in range(50):
        spec = simulate.random_room(rng)
        mics = simulate.mic_positions(spec)
        dims = np.array(spec.dims)[:, None]
        assert np.all(mics > 0) and np.all(mics < dims)
        assert spec.distance >= 1.0


def test_qc_flags_bad_audio():
    bad = np.zeros((4, 100), dtype=np.float32)
    assert "near-silent" in features.quality_checks(bad, 16000, 100, 4)


def test_end_to_end(tmp_path):
    s = run(out_dir=str(tmp_path), n_rooms=3, clips_per_room=2, seed=7)
    assert s["clips"] == 6 and s["passed_qc"] == 6
    session = get_session(s["db_url"])
    clip = session.scalars(select(Clip)).first()
    audio, sr = sf.read(clip.audio_path)
    assert audio.shape == (32000, 4) and sr == 16000
    assert np.load(clip.features_path).shape[:2] == (4, 64)


def test_workers_give_identical_dataset(tmp_path):
    a = run(out_dir=str(tmp_path / "a"), n_rooms=3, clips_per_room=2, seed=5, workers=1)
    b = run(out_dir=str(tmp_path / "b"), n_rooms=3, clips_per_room=2, seed=5, workers=3)
    cols = (Clip.azimuth_deg, Clip.rt60_measured, Clip.snr_achieved_db)
    ra = get_session(a["db_url"]).execute(select(*cols).order_by(Clip.id)).all()
    rb = get_session(b["db_url"]).execute(select(*cols).order_by(Clip.id)).all()
    assert ra == rb


def test_reproducible(tmp_path):
    a = run(out_dir=str(tmp_path / "a"), n_rooms=2, clips_per_room=2, seed=3)
    b = run(out_dir=str(tmp_path / "b"), n_rooms=2, clips_per_room=2, seed=3)
    sa = get_session(a["db_url"]).scalars(select(Clip.azimuth_deg)).all()
    sb = get_session(b["db_url"]).scalars(select(Clip.azimuth_deg)).all()
    assert sa == sb