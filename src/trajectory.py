"""Ground-truth target trajectory generators."""

from __future__ import annotations

import numpy as np

from . import DURATION, SAMPLE_PERIOD, TARGET_SPEED

# Step indices where the default zig-zag changes heading. Default trajectory has
# 5 legs of 24 steps each (60 s / 5 / 0.5 s), so heading flips occur on the
# boundaries 24, 48, 72, 96. TURN_WINDOW is the number of steps after each flip
# during which the CV-EKF is still washing out the prediction error.
TURN_STEPS: tuple[int, ...] = (24, 48, 72, 96)
TURN_WINDOW: int = 4


def smooth_mask(
    K: int,
    warmup: int = 5,
    turn_steps: tuple[int, ...] = TURN_STEPS,
    turn_window: int = TURN_WINDOW,
) -> np.ndarray:
    """Boolean mask of length K, True on steady-state, between-turn steps.

    Excludes the first `warmup` steps (filter convergence) and a `turn_window`-
    step transient after each heading change.
    """
    mask = np.ones(K, dtype=bool)
    if warmup > 0:
        mask[:warmup] = False
    for ts in turn_steps:
        mask[ts : ts + turn_window] = False
    return mask


def turn_mask(
    K: int,
    turn_steps: tuple[int, ...] = TURN_STEPS,
    turn_window: int = TURN_WINDOW,
) -> np.ndarray:
    """Boolean mask of length K, True only during the post-turn transient."""
    mask = np.zeros(K, dtype=bool)
    for ts in turn_steps:
        mask[ts : ts + turn_window] = True
    return mask


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
