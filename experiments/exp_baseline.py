"""Baseline single-trajectory experiment: N=4 square geometry.

Generates the full set of canonical plots:
  1. trajectory in the full arena (with sensors)
  2. zoomed trajectory comparing ground truth, LSE-only, and EKF
  3. position error vs time (LSE per-step vs EKF)
  4. EKF innovation sequences

Usage: python -m experiments.exp_baseline
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src import SAMPLE_PERIOD, SIGMA_R
from src.ekf import run_ekf
from src.lse import initialize_state, lse_track
from src.measurements import simulate_toa, to_range
from src.metrics import (
    plot_innovations,
    plot_rmse_vs_time,
    plot_trajectory,
    position_error,
    save_figure,
)
from src.sensors import GEOMETRY_LABELS, get_geometry
from src.trajectory import make_zigzag, smooth_mask

SIGMA_A = 10.0
PRIOR_P = np.array([3000.0, 4000.0])
SEED = 42


def run_baseline(
    geometry: str = "N4_square",
    seed: int = SEED,
    sigma_a: float = SIGMA_A,
    prior_p: np.ndarray = PRIOR_P,
) -> dict:
    states_truth, times = make_zigzag()
    sensors = get_geometry(geometry)

    rng = np.random.default_rng(seed)
    z, emit = simulate_toa(states_truth, sensors, rng=rng)
    r = to_range(z, emit)

    lse_positions = lse_track(r, sensors, prior_p)

    x0, P0 = initialize_state(r[:2], sensors, T=SAMPLE_PERIOD, prior_p=prior_p)
    states_ekf, covs, innovs = run_ekf(r, sensors, x0, P0, sigma_a=sigma_a)

    return {
        "truth": states_truth,
        "times": times,
        "sensors": sensors,
        "lse_positions": lse_positions,
        "states_ekf": states_ekf,
        "covs": covs,
        "innovs": innovs,
        "geometry": geometry,
    }


def plot_baseline(results: dict) -> None:
    truth = results["truth"]
    sensors = results["sensors"]
    times = results["times"]
    states_ekf = results["states_ekf"]
    lse_positions = results["lse_positions"]
    innovs = results["innovs"]
    geometry = results["geometry"]

    lse_states = np.column_stack([lse_positions, np.zeros_like(lse_positions)])

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_trajectory(
        truth,
        sensors,
        {"EKF": states_ekf, "LSE per-step": lse_states},
        title=f"Trajectory (full arena) — {GEOMETRY_LABELS[geometry]}",
        ax=ax,
    )
    save_figure(fig, f"baseline_{geometry}_arena.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(truth[:, 0], truth[:, 1], "k-", lw=2.0, label="ground truth")
    ax.plot(
        lse_positions[:, 0],
        lse_positions[:, 1],
        marker=".",
        color="C3",
        ls="",
        ms=5,
        alpha=0.6,
        label="LSE per-step",
    )
    ax.plot(states_ekf[:, 0], states_ekf[:, 1], "C0-", lw=1.5, label="EKF")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Trajectory zoom — {GEOMETRY_LABELS[geometry]}")
    ax.legend(loc="best", fontsize=9)
    save_figure(fig, f"baseline_{geometry}_zoom.png")
    plt.close(fig)

    err_lse = np.linalg.norm(lse_positions - truth[:, :2], axis=1)
    err_ekf = position_error(states_ekf, truth)
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_rmse_vs_time(
        {"LSE per-step": (times, err_lse), "EKF": (times, err_ekf)},
        title=f"Position error vs time — {GEOMETRY_LABELS[geometry]}",
        ax=ax,
    )
    save_figure(fig, f"baseline_{geometry}_rmse.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_innovations(times, innovs, SIGMA_R, ax=ax)
    save_figure(fig, f"baseline_{geometry}_innovations.png")
    plt.close(fig)


def _rmse(err: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err[mask] ** 2)))


def main() -> None:
    results = run_baseline()
    plot_baseline(results)

    truth = results["truth"]
    states_ekf = results["states_ekf"]
    lse_positions = results["lse_positions"]
    err_ekf = position_error(states_ekf, truth)
    err_lse = np.linalg.norm(lse_positions - truth[:, :2], axis=1)

    K = len(err_ekf)
    overall = np.zeros(K, dtype=bool)
    overall[5:] = True
    smooth = smooth_mask(K)

    rmse_lse_overall = _rmse(err_lse, overall)
    rmse_ekf_overall = _rmse(err_ekf, overall)
    rmse_lse_smooth = _rmse(err_lse, smooth)
    rmse_ekf_smooth = _rmse(err_ekf, smooth)
    bound = SIGMA_R / np.sqrt(len(results["sensors"]))

    print(f"\nBaseline — {GEOMETRY_LABELS[results['geometry']]}")
    print(f"  Theoretical bound (σ_r/√N)   : {bound:.2f} m")
    print(f"  LSE per-step RMSE (overall)  : {rmse_lse_overall:.2f} m")
    print(f"  EKF RMSE (overall)           : {rmse_ekf_overall:.2f} m")
    print(f"  LSE per-step RMSE (smooth)   : {rmse_lse_smooth:.2f} m")
    print(f"  EKF RMSE (smooth, no turns)  : {rmse_ekf_smooth:.2f} m")
    print(f"  EKF max error (overall)      : {err_ekf[overall].max():.2f} m")

    smooth_pass = rmse_ekf_smooth < rmse_lse_smooth
    print(
        f"  Sanity: EKF beats LSE in smooth windows [{ 'PASS' if smooth_pass else 'FAIL' }]"
    )


if __name__ == "__main__":
    main()
