"""Ground-truth target trajectory generators."""

from __future__ import annotations

import numpy as np

from . import DURATION, SAMPLE_PERIOD, TARGET_SPEED


def make_zigzag(
    duration: float = DURATION,
    dt: float = SAMPLE_PERIOD,
    speed: float = TARGET_SPEED,
    start: tuple[float, float] = (2000.0, 5000.0),
    headings_deg: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Constant-speed zig-zag trajectory with sharp heading changes.

    Returns:
        states: (K, 4) array of [px, py, vx, vy]; px,py in metres, vx,vy in m/s.
        times:  (K,) array of sample times in seconds (also serve as emission times).
    """
    if headings_deg is None:
        headings_deg = [30.0, -30.0, 50.0, -50.0, 30.0]
    n_legs = len(headings_deg)
    leg_seconds = duration / n_legs

    K = int(round(duration / dt)) + 1
    times = np.arange(K) * dt
    states = np.zeros((K, 4))

    px, py = start
    for k in range(K):
        t = times[k]
        leg_idx = min(int(t / leg_seconds + 1e-9), n_legs - 1)
        h = np.deg2rad(headings_deg[leg_idx])
        vx = speed * np.cos(h)
        vy = speed * np.sin(h)
        states[k] = (px, py, vx, vy)
        if k < K - 1:
            px += vx * dt
            py += vy * dt

    return states, times
