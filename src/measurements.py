"""TOA measurement simulation and time↔range conversion."""

from __future__ import annotations

import numpy as np

from . import C_LIGHT, SAMPLE_PERIOD, SIGMA_T


def simulate_toa(
    states: np.ndarray,
    sensors: np.ndarray,
    sigma_t: float = SIGMA_T,
    emission_times: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate noisy time-of-arrival measurements.

    Physical model: z_i^k = t_k + ‖p^k − s_i‖ / c + n_i^k, n ~ N(0, σ_t²).

    Args:
        states: (K, 4) ground-truth states [px, py, vx, vy].
        sensors: (N, 2) sensor positions.
        sigma_t: time-measurement noise std (seconds).
        emission_times: (K,) emission times t_k. Defaults to k * SAMPLE_PERIOD.
        rng: numpy Generator (omit for fresh randomness).

    Returns:
        z: (K, N) noisy TOA in seconds.
        emission_times: (K,) emission times (echoed for caller convenience).
    """
    if rng is None:
        rng = np.random.default_rng()

    K = states.shape[0]
    if emission_times is None:
        emission_times = SAMPLE_PERIOD * np.arange(K)

    pos = states[:, :2]
    diffs = pos[:, None, :] - sensors[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    tof = distances / C_LIGHT

    noise = rng.normal(0.0, sigma_t, size=distances.shape)
    z = emission_times[:, None] + tof + noise
    return z, emission_times


def to_range(z: np.ndarray, emission_times: np.ndarray) -> np.ndarray:
    """Convert TOA z (seconds) to range r = c · (z − t_k) (metres)."""
    return C_LIGHT * (z - emission_times[:, None])
