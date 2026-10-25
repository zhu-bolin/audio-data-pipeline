"""Room acoustics simulation with pyroomacoustics (image source method)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyroomacoustics as pra


@dataclass
class RoomSpec:
    dims: tuple[float, float, float]  # meters (x, y, z)
    rt60_target: float  # seconds
    mic_center: tuple[float, float, float]
    source_pos: tuple[float, float, float]
    array_type: str  # "circular4" or "linear4"

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(np.subtract(self.source_pos, self.mic_center)))

    @property
    def azimuth_deg(self) -> float:
        """Direction of arrival in the horizontal plane, relative to the array's x-axis."""
        dx, dy = self.source_pos[0] - self.mic_center[0], self.source_pos[1] - self.mic_center[1]
        return float(np.degrees(np.arctan2(dy, dx)) % 360)


def random_room(rng: np.random.Generator, rt60_range=(0.2, 0.8)) -> RoomSpec:
    """Sample a plausible small/medium room (office, conference room, classroom)."""
    dims = (rng.uniform(3, 10), rng.uniform(3, 8), rng.uniform(2.5, 3.5))
    margin = 0.5

    def point(z):
        return (rng.uniform(margin, dims[0] - margin), rng.uniform(margin, dims[1] - margin), z)

    mic = point(rng.uniform(0.7, 1.2))  # table height
    for _ in range(100):  # keep the source at least 1 m from the array
        src = point(rng.uniform(1.2, 1.8))  # seated/standing talker
        if np.linalg.norm(np.subtract(src, mic)) >= 1.0:
            break
    return RoomSpec(
        dims=tuple(round(d, 3) for d in dims),
        rt60_target=round(float(rng.uniform(*rt60_range)), 3),
        mic_center=tuple(round(c, 3) for c in mic),
        source_pos=tuple(round(c, 3) for c in src),
        array_type=str(rng.choice(["circular4", "linear4"])),
    )


def move_source(spec: RoomSpec, rng: np.random.Generator, min_dist: float = 1.0) -> RoomSpec:
    """Same room and array, new talker position (at least min_dist from the array)."""
    m = 0.5
    for _ in range(100):
        src = (rng.uniform(m, spec.dims[0] - m), rng.uniform(m, spec.dims[1] - m), rng.uniform(1.2, 1.8))
        if np.linalg.norm(np.subtract(src, spec.mic_center)) >= min_dist:
            break
    return RoomSpec(spec.dims, spec.rt60_target, spec.mic_center,
                    tuple(round(c, 3) for c in src), spec.array_type)


def mic_positions(spec: RoomSpec, spacing: float = 0.05) -> np.ndarray:
    """3 x M microphone coordinates for a 4-mic array centered at spec.mic_center."""
    cx, cy, cz = spec.mic_center
    if spec.array_type == "circular4":
        xy = pra.circular_2D_array(center=[cx, cy], M=4, phi0=0, radius=spacing)
    else:
        xy = pra.linear_2D_array(center=[cx, cy], M=4, phi=0, d=spacing)
    return np.vstack([xy, np.full(xy.shape[1], cz)])


def simulate(signal: np.ndarray, sr: int, spec: RoomSpec) -> tuple[np.ndarray, float]:
    """Return (multichannel reverberant audio [M, N], measured RT60 in seconds)."""
    absorption, max_order = pra.inverse_sabine(spec.rt60_target, spec.dims)
    room = pra.ShoeBox(
        spec.dims, fs=sr, materials=pra.Material(absorption), max_order=min(max_order, 40)
    )
    room.add_source(spec.source_pos, signal=signal)
    room.add_microphone_array(mic_positions(spec))
    room.simulate()
    out = room.mic_array.signals[:, : len(signal)]  # keep clip length fixed for batching
    measured = float(np.mean(room.measure_rt60()))
    return out.astype(np.float32), measured
