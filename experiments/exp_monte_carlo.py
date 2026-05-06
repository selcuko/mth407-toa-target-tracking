"""Monte Carlo experiment: 100 noise realizations per geometry cell.

Aggregates per-trial RMSE and per-step error across trials. Outputs:
  - Per-cell error-vs-time panel grid (median + IQR shading)
  - Cross-cell summary RMSE bar chart with error bars
  - CSV summary table of mean ± std

Usage: python -m experiments.exp_monte_carlo
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
from src.sensors import GEOMETRY_LABELS, get_geometry
from src.trajectory import make_zigzag

SIGMA_A = 10.0
PRIOR_P = np.array([3000.0, 4000.0])
N_TRIALS = 100
BASE_SEED = 1000

CELL_ORDER = [
    "N2_wide",
    "N2_close",
    "N3_triangle",
    "N3_collinear",
    "N4_square",
    "N4_cluster",
]


def _run_trial(geometry: str, seed: int, states_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sensors = get_geometry(geometry)
    rng = np.random.default_rng(seed)
    z, emit = simulate_toa(states_truth, sensors, rng=rng)
    r = to_range(z, emit)

    lse_pos = lse_track(r, sensors, PRIOR_P)
    x0, P0 = initialize_state(r[:2], sensors, T=SAMPLE_PERIOD, prior_p=PRIOR_P)
    states_ekf, _, _ = run_ekf(r, sensors, x0, P0, sigma_a=SIGMA_A)

    err_lse = np.linalg.norm(lse_pos - states_truth[:, :2], axis=1)
    err_ekf = position_error(states_ekf, states_truth)
    return err_lse, err_ekf


def run_mc_for_cell(geometry: str, n_trials: int = N_TRIALS) -> dict:
    states_truth, times = make_zigzag()
    K = len(times)
    err_lse_mat = np.zeros((n_trials, K))
    err_ekf_mat = np.zeros((n_trials, K))
    for i in range(n_trials):
        e_lse, e_ekf = _run_trial(geometry, BASE_SEED + i, states_truth)
        err_lse_mat[i] = e_lse
        err_ekf_mat[i] = e_ekf

    rmse_lse = np.sqrt(np.mean(err_lse_mat[:, 5:] ** 2, axis=1))
    rmse_ekf = np.sqrt(np.mean(err_ekf_mat[:, 5:] ** 2, axis=1))
    return {
        "err_lse_mat": err_lse_mat,
        "err_ekf_mat": err_ekf_mat,
        "times": times,
        "rmse_lse_mean": float(np.mean(rmse_lse)),
        "rmse_lse_std": float(np.std(rmse_lse)),
        "rmse_ekf_mean": float(np.mean(rmse_ekf)),
        "rmse_ekf_std": float(np.std(rmse_ekf)),
    }


def plot_panel_grid(all_data: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, name in zip(axes.flatten(), CELL_ORDER):
        d = all_data[name]
        t = d["times"]
        lse_med = np.median(d["err_lse_mat"], axis=0)
        ekf_med = np.median(d["err_ekf_mat"], axis=0)
        lse_q1 = np.percentile(d["err_lse_mat"], 25, axis=0)
        lse_q3 = np.percentile(d["err_lse_mat"], 75, axis=0)
        ekf_q1 = np.percentile(d["err_ekf_mat"], 25, axis=0)
        ekf_q3 = np.percentile(d["err_ekf_mat"], 75, axis=0)
        ax.fill_between(t, lse_q1, lse_q3, color="C3", alpha=0.2)
        ax.fill_between(t, ekf_q1, ekf_q3, color="C0", alpha=0.2)
        ax.plot(t, lse_med, color="C3", lw=1.4, label="LSE (median)")
        ax.plot(t, ekf_med, color="C0", lw=1.4, label="EKF (median)")
        ax.axhline(SIGMA_R, color="grey", ls=":", lw=0.8, alpha=0.6)
        ax.set_yscale("log")
        ax.set_title(GEOMETRY_LABELS[name], fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("position error [m] (log)")
    fig.suptitle(
        f"Position error vs time — {N_TRIALS}-trial Monte Carlo (median + IQR)",
        y=1.00,
    )
    fig.tight_layout()
    save_figure(fig, "mc_error_vs_time.png")
    plt.close(fig)


def plot_summary_bars(all_data: dict[str, dict]) -> None:
    labels = [GEOMETRY_LABELS[n] for n in CELL_ORDER]
    rmse_lse = [all_data[n]["rmse_lse_mean"] for n in CELL_ORDER]
    rmse_lse_std = [all_data[n]["rmse_lse_std"] for n in CELL_ORDER]
    rmse_ekf = [all_data[n]["rmse_ekf_mean"] for n in CELL_ORDER]
    rmse_ekf_std = [all_data[n]["rmse_ekf_std"] for n in CELL_ORDER]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        x - width / 2,
        rmse_lse,
        width,
        yerr=rmse_lse_std,
        capsize=4,
        color="C3",
        alpha=0.85,
        label="LSE per-step",
    )
    ax.bar(
        x + width / 2,
        rmse_ekf,
        width,
        yerr=rmse_ekf_std,
        capsize=4,
        color="C0",
        alpha=0.85,
        label="EKF",
    )
    bound = SIGMA_R / np.sqrt(np.array([2, 2, 3, 3, 4, 4]))
    ax.plot(x, bound, "k--", lw=1.0, alpha=0.7, label="σ_r/√N (theoretical)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("position RMSE [m] (log)")
    ax.set_title(f"Monte Carlo RMSE summary ({N_TRIALS} trials)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, "mc_rmse_summary.png")
    plt.close(fig)


def main() -> None:
    all_data: dict[str, dict] = {}
    for name in CELL_ORDER:
        print(f"Running {N_TRIALS}-trial MC for {GEOMETRY_LABELS[name]}...")
        all_data[name] = run_mc_for_cell(name)

    plot_panel_grid(all_data)
    plot_summary_bars(all_data)

    print(
        f"\n{'cell':<32s} {'LSE mean ± std [m]':>22s} {'EKF mean ± std [m]':>22s}"
    )
    print("-" * 80)
    for name in CELL_ORDER:
        d = all_data[name]
        lse_str = f"{d['rmse_lse_mean']:.2f} ± {d['rmse_lse_std']:.2f}"
        ekf_str = f"{d['rmse_ekf_mean']:.2f} ± {d['rmse_ekf_std']:.2f}"
        print(f"{GEOMETRY_LABELS[name]:<32s} {lse_str:>22s} {ekf_str:>22s}")

    save_csv(
        {
            "geometry": [GEOMETRY_LABELS[n] for n in CELL_ORDER],
            "rmse_lse_mean_m": [all_data[n]["rmse_lse_mean"] for n in CELL_ORDER],
            "rmse_lse_std_m": [all_data[n]["rmse_lse_std"] for n in CELL_ORDER],
            "rmse_ekf_mean_m": [all_data[n]["rmse_ekf_mean"] for n in CELL_ORDER],
            "rmse_ekf_std_m": [all_data[n]["rmse_ekf_std"] for n in CELL_ORDER],
        },
        "mc_summary.csv",
    )
    print(f"\nSaved CSV summary to {OUTPUT_RESULTS / 'mc_summary.csv'}")


if __name__ == "__main__":
    main()
