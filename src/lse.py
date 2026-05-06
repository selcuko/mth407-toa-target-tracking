"""Least-squares position estimation and full state vector initialization.

Two regimes:
  - N >= 3 sensors: iterative Gauss-Newton on the non-linear range model.
  - N == 2 sensors: closed-form two-circle intersection with prior-based
    disambiguation (the geometry is fundamentally ambiguous).
"""

from __future__ import annotations

import numpy as np

from . import SIGMA_R


def _range_jacobian(p: np.ndarray, sensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diffs = p - sensors
    dists = np.linalg.norm(diffs, axis=1)
    H = diffs / np.maximum(dists, 1e-9)[:, None]
    return H, dists


def lse_position(
    ranges: np.ndarray,
    sensors: np.ndarray,
    p0: np.ndarray | None = None,
    sigma_r: float = SIGMA_R,
    max_iter: int = 30,
    tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative Gauss-Newton LSE for 2D position from range measurements.

    Returns position estimate (2,) and its 2×2 covariance σ_r² (HᵀH)⁻¹.
    """
    if p0 is None:
        p0 = np.mean(sensors, axis=0)
    p = np.asarray(p0, dtype=float).copy()
    for _ in range(max_iter):
        H, dists = _range_jacobian(p, sensors)
        residual = ranges - dists
        try:
            dp, *_ = np.linalg.lstsq(H, residual, rcond=None)
        except np.linalg.LinAlgError:
            break
        p = p + dp
        if np.linalg.norm(dp) < tol:
            break
    H, _ = _range_jacobian(p, sensors)
    HtH = H.T @ H
    cov_p = sigma_r**2 * np.linalg.pinv(HtH)
    return p, cov_p


def lse_two_sensor(
    ranges: np.ndarray,
    sensors: np.ndarray,
    prior_p: np.ndarray,
    sigma_r: float = SIGMA_R,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form 2-sensor 2D position via circle intersection.

    Two intersection candidates exist in general; the one closer to `prior_p`
    is returned. If the circles fail to intersect (large noise), we fall back
    to the foot of the perpendicular on the baseline.
    """
    s1, s2 = sensors
    r1, r2 = ranges
    d_vec = s2 - s1
    d = float(np.linalg.norm(d_vec))
    if d < 1e-9:
        return np.asarray(prior_p, dtype=float).copy(), np.eye(2) * 1e6

    e_x = d_vec / d
    e_y = np.array([-e_x[1], e_x[0]])

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h_squared = r1**2 - a**2
    h = float(np.sqrt(max(h_squared, 0.0)))

    base = s1 + a * e_x
    cand_plus = base + h * e_y
    cand_minus = base - h * e_y

    if np.linalg.norm(cand_plus - prior_p) <= np.linalg.norm(cand_minus - prior_p):
        p = cand_plus
    else:
        p = cand_minus

    H, _ = _range_jacobian(p, sensors)
    HtH = H.T @ H
    cov_p = sigma_r**2 * np.linalg.pinv(HtH)
    return p, cov_p


def lse_track(
    ranges_seq: np.ndarray,
    sensors: np.ndarray,
    prior_p: np.ndarray,
    sigma_r: float = SIGMA_R,
) -> np.ndarray:
    """Run LSE position estimation independently at each time step.

    Returns (K, 2) position estimates. Uses the previous step's estimate as
    a warm start, keeping the 2-sensor disambiguation continuous.
    """
    K = ranges_seq.shape[0]
    positions = np.zeros((K, 2))
    p_warm = np.asarray(prior_p, dtype=float).copy()
    is_two = sensors.shape[0] == 2
    for k in range(K):
        if is_two:
            p_est, _ = lse_two_sensor(ranges_seq[k], sensors, p_warm, sigma_r)
        else:
            p_est, _ = lse_position(ranges_seq[k], sensors, p_warm, sigma_r)
        positions[k] = p_est
        p_warm = p_est
    return positions


def initialize_state(
    ranges_seq: np.ndarray,
    sensors: np.ndarray,
    T: float,
    prior_p: np.ndarray | None = None,
    sigma_r: float = SIGMA_R,
) -> tuple[np.ndarray, np.ndarray]:
    """Init [px, py, vx, vy] and 4×4 covariance from ranges at the first 2 steps.

    ranges_seq: (2, N) array of ranges at time-step 0 and time-step 1.
    """
    N = sensors.shape[0]
    if prior_p is None:
        prior_p = np.mean(sensors, axis=0)
    prior = np.asarray(prior_p, dtype=float)

    if N == 2:
        p0_est, cov_p0 = lse_two_sensor(ranges_seq[0], sensors, prior, sigma_r)
        p1_est, cov_p1 = lse_two_sensor(ranges_seq[1], sensors, prior, sigma_r)
    else:
        p0_est, cov_p0 = lse_position(ranges_seq[0], sensors, prior, sigma_r)
        p1_est, cov_p1 = lse_position(ranges_seq[1], sensors, prior, sigma_r)

    v_est = (p1_est - p0_est) / T

    cov_v = (cov_p0 + cov_p1) / (T * T)
    cov_pv = -cov_p0 / T

    x0 = np.concatenate([p0_est, v_est])
    P0 = np.zeros((4, 4))
    P0[0:2, 0:2] = cov_p0
    P0[2:4, 2:4] = cov_v
    P0[0:2, 2:4] = cov_pv
    P0[2:4, 0:2] = cov_pv.T
    return x0, P0
