"""Extended Kalman Filter for 2D constant-velocity tracking with range measurements."""

from __future__ import annotations

import numpy as np

from . import SAMPLE_PERIOD, SIGMA_R


def cv_F(T: float) -> np.ndarray:
    """4×4 state transition for [px, py, vx, vy] under CV dynamics."""
    F = np.eye(4)
    F[0, 2] = T
    F[1, 3] = T
    return F


def cv_Q(T: float, sigma_a: float) -> np.ndarray:
    """4×4 discrete white-acceleration process-noise covariance."""
    q = sigma_a * sigma_a
    T2 = T * T
    T3 = T * T2
    T4 = T2 * T2
    return q * np.array(
        [
            [T4 / 4, 0.0, T3 / 2, 0.0],
            [0.0, T4 / 4, 0.0, T3 / 2],
            [T3 / 2, 0.0, T2, 0.0],
            [0.0, T3 / 2, 0.0, T2],
        ]
    )


def _measurement_jacobian(x: np.ndarray, sensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = x[:2]
    diffs = p - sensors
    dists = np.linalg.norm(diffs, axis=1)
    H = np.zeros((sensors.shape[0], 4))
    H[:, :2] = diffs / np.maximum(dists, 1e-9)[:, None]
    return dists, H


def ekf_step(
    x: np.ndarray,
    P: np.ndarray,
    ranges: np.ndarray,
    sensors: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    h, H = _measurement_jacobian(x_pred, sensors)
    y = ranges - h
    S = H @ P_pred @ H.T + R

    # K = P_pred Hᵀ S⁻¹ via solve: S Kᵀ = H P_predᵀ ⇒ Kᵀ = solve(S, H P_pred).
    K = np.linalg.solve(S, H @ P_pred).T

    x_new = x_pred + K @ y
    I_KH = np.eye(4) - K @ H
    P_new = I_KH @ P_pred @ I_KH.T + K @ R @ K.T  # Joseph form
    return x_new, P_new, y, S


def run_ekf(
    ranges: np.ndarray,
    sensors: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    T: float = SAMPLE_PERIOD,
    sigma_a: float = 5.0,
    sigma_r: float = SIGMA_R,
    skip_updates: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the EKF across a complete (K, N) range sequence.

    `skip_updates`: number of leading measurements to NOT update on (because
    they were already consumed by the LSE initializer). Predict still runs at
    every step. Default 1 matches the two-step LSE init: r[0] and r[1] feed
    initialize_state, EKF starts updating at k=2.

    Returns
    -------
    states : (K, 4) filtered state at each step (states[0] = x0).
    covs   : (K, 4, 4) filtered covariance at each step.
    innovs : (K, N) innovation sequences (zeros for predict-only steps).
    """
    K_steps, N = ranges.shape
    F = cv_F(T)
    Q = cv_Q(T, sigma_a)
    R = (sigma_r * sigma_r) * np.eye(N)

    states = np.zeros((K_steps, 4))
    covs = np.zeros((K_steps, 4, 4))
    innovs = np.zeros((K_steps, N))

    x = np.asarray(x0, dtype=float).copy()
    P = np.asarray(P0, dtype=float).copy()
    states[0] = x
    covs[0] = P

    for k in range(1, K_steps):
        if k <= skip_updates:
            x = F @ x
            P = F @ P @ F.T + Q
        else:
            x, P, y, _ = ekf_step(x, P, ranges[k], sensors, F, Q, R)
            innovs[k] = y
        states[k] = x
        covs[k] = P

    return states, covs, innovs
