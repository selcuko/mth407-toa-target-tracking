"""σ_a (process-noise acceleration) sensitivity sweep.

The CV-EKF assumes a constant-velocity target with white-acceleration process
noise; σ_a controls how readily the filter accepts that the velocity may have
changed. Low σ_a means tight tracking on straight legs but slow recovery from
turns. High σ_a means quick turn recovery but a noisier baseline.

This script holds the sensor geometry fixed at N=4 square and sweeps σ_a over
five decades, running a small Monte Carlo at each value. It plots the
smooth-segment RMSE and turn-window RMSE as functions of σ_a so the trade-off
is visible directly.

Usage: python -m experiments.exp_sigma_a_sweep
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
from src.metrics import OUTPUT_RESULTS, position_error, save_csv, save_figure
from src.sensors import get_geometry
from src.trajectory import make_zigzag, smooth_mask, turn_mask

GEOMETRY = "N4_square"
PRIOR_P = np.array([3000.0, 4000.0])
N_TRIALS = 30
BASE_SEED = 5000
SIGMA_A_VALUES = [1.0, 5.0, 10.0, 50.0, 200.0]


def _trial(sigma_a: float, seed: int, states_truth: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z, emit = simulate_toa(states_truth, sensors, rng=rng)
    r = to_range(z, emit)
    _ = lse_track(r, sensors, PRIOR_P)
    x0, P0 = initialize_state(r[:2], sensors, T=SAMPLE_PERIOD, prior_p=PRIOR_P)
    states_ekf, _, _ = run_ekf(r, sensors, x0, P0, sigma_a=sigma_a)
    return position_error(states_ekf, states_truth)


def main() -> None:
    states_truth, _ = make_zigzag()
    sensors = get_geometry(GEOMETRY)
    K = states_truth.shape[0]
    smooth = smooth_mask(K)
    turn = turn_mask(K)

    rows: list[dict] = []
    for sa in SIGMA_A_VALUES:
        print(f"σ_a = {sa:6.1f} m/s²  ({N_TRIALS} trials) ...")
        err_mat = np.zeros((N_TRIALS, K))
        for i in range(N_TRIALS):
            err_mat[i] = _trial(sa, BASE_SEED + i, states_truth, sensors)
        rmse_smooth = np.sqrt(np.mean(err_mat[:, smooth] ** 2, axis=1))
        rmse_turn = np.sqrt(np.mean(err_mat[:, turn] ** 2, axis=1))
        rows.append(
            {
                "sigma_a": sa,
                "rmse_smooth_mean": float(np.mean(rmse_smooth)),
                "rmse_smooth_std": float(np.std(rmse_smooth)),
                "rmse_turn_mean": float(np.mean(rmse_turn)),
                "rmse_turn_std": float(np.std(rmse_turn)),
            }
        )

    sa_arr = np.array([r["sigma_a"] for r in rows])
    smooth_mean = np.array([r["rmse_smooth_mean"] for r in rows])
    smooth_std = np.array([r["rmse_smooth_std"] for r in rows])
    turn_mean = np.array([r["rmse_turn_mean"] for r in rows])
    turn_std = np.array([r["rmse_turn_std"] for r in rows])

    print("\nN=4 square — σ_a sensitivity")
    print(f"{'σ_a [m/s²]':>10s} {'smooth [m]':>16s} {'turn [m]':>16s}")
    print("-" * 46)
    for r in rows:
        print(
            f"{r['sigma_a']:>10.1f} "
            f"{r['rmse_smooth_mean']:>7.2f} ± {r['rmse_smooth_std']:>4.2f} "
            f"{r['rmse_turn_mean']:>7.2f} ± {r['rmse_turn_std']:>4.2f}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        sa_arr, smooth_mean, yerr=smooth_std,
        marker="o", lw=1.6, capsize=4, color="C2", label="smooth segments",
    )
    ax.errorbar(
        sa_arr, turn_mean, yerr=turn_std,
        marker="s", lw=1.6, capsize=4, color="C1", label="turn windows",
    )
    bound = SIGMA_R / np.sqrt(sensors.shape[0])
    ax.axhline(bound, color="k", ls="--", lw=1.0, alpha=0.7, label=f"σ_r/√N = {bound:.2f} m")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("σ_a (process-noise acceleration std) [m/s²]")
    ax.set_ylabel("EKF position RMSE [m]")
    ax.set_title(
        f"σ_a sensitivity — N=4 square, {N_TRIALS}-trial MC\n"
        "(low σ_a tracks straight legs tightly but lags turns; high σ_a is the reverse)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, "sigma_a_sweep.png")
    plt.close(fig)

    save_csv(
        {
            "sigma_a_m_per_s2": [r["sigma_a"] for r in rows],
            "rmse_smooth_mean_m": [r["rmse_smooth_mean"] for r in rows],
            "rmse_smooth_std_m": [r["rmse_smooth_std"] for r in rows],
            "rmse_turn_mean_m": [r["rmse_turn_mean"] for r in rows],
            "rmse_turn_std_m": [r["rmse_turn_std"] for r in rows],
        },
        "sigma_a_sweep.csv",
    )
    print(f"\nSaved CSV to {OUTPUT_RESULTS / 'sigma_a_sweep.csv'}")


if __name__ == "__main__":
    main()
