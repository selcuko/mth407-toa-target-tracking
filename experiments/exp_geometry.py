"""Sensor-geometry sweep: six (N, geometry) cells, single deterministic run each.

Outputs a 6-panel trajectory grid and a summary RMSE bar chart.

Usage: python -m experiments.exp_geometry
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
from src.sensors import GEOMETRIES, GEOMETRY_LABELS, get_geometry
from src.trajectory import make_zigzag

SIGMA_A = 10.0
PRIOR_P = np.array([3000.0, 4000.0])
SEED = 42

CELL_ORDER = [
    "N2_wide",
    "N2_close",
    "N3_triangle",
    "N3_collinear",
    "N4_square",
    "N4_cluster",
]


def run_one(geometry: str, seed: int = SEED) -> dict:
    states_truth, times = make_zigzag()
    sensors = get_geometry(geometry)
    rng = np.random.default_rng(seed)
    z, emit = simulate_toa(states_truth, sensors, rng=rng)
    r = to_range(z, emit)

    lse_pos = lse_track(r, sensors, PRIOR_P)
    x0, P0 = initialize_state(r[:2], sensors, T=SAMPLE_PERIOD, prior_p=PRIOR_P)
    states_ekf, _, innovs = run_ekf(r, sensors, x0, P0, sigma_a=SIGMA_A)

    err_lse = np.linalg.norm(lse_pos - states_truth[:, :2], axis=1)
    err_ekf = position_error(states_ekf, states_truth)
    rmse_lse = float(np.sqrt(np.mean(err_lse[5:] ** 2)))
    rmse_ekf = float(np.sqrt(np.mean(err_ekf[5:] ** 2)))

    return {
        "truth": states_truth,
        "times": times,
        "sensors": sensors,
        "lse_pos": lse_pos,
        "states_ekf": states_ekf,
        "err_lse": err_lse,
        "err_ekf": err_ekf,
        "rmse_lse": rmse_lse,
        "rmse_ekf": rmse_ekf,
        "geometry": geometry,
    }


def plot_grid(all_results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, name in zip(axes.flatten(), CELL_ORDER):
        res = all_results[name]
        truth = res["truth"]
        sensors = res["sensors"]
        ekf = res["states_ekf"]
        lse = res["lse_pos"]

        ax.plot(truth[:, 0], truth[:, 1], "k-", lw=2.0, label="truth")
        ax.plot(lse[:, 0], lse[:, 1], "C3.", ms=3, alpha=0.5, label="LSE")
        ax.plot(ekf[:, 0], ekf[:, 1], "C0-", lw=1.4, label="EKF")
        ax.scatter(
            sensors[:, 0],
            sensors[:, 1],
            marker="^",
            s=80,
            c="red",
            edgecolors="black",
            zorder=5,
            label="sensors",
        )
        ax.set_aspect("equal")
        ax.set_xlim(-500, 10500)
        ax.set_ylim(-500, 10500)
        ax.set_title(
            f"{GEOMETRY_LABELS[name]}\nLSE RMSE {res['rmse_lse']:.1f}m  |  "
            f"EKF RMSE {res['rmse_ekf']:.1f}m"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.suptitle(
        "Sensor-geometry sweep — single deterministic run, sigma_a=10",
        fontsize=14,
        y=1.00,
    )
    fig.tight_layout()
    save_figure(fig, "geometry_sweep_grid.png")
    plt.close(fig)


def plot_rmse_bars(all_results: dict[str, dict]) -> None:
    labels = [GEOMETRY_LABELS[n] for n in CELL_ORDER]
    rmse_lse = [all_results[n]["rmse_lse"] for n in CELL_ORDER]
    rmse_ekf = [all_results[n]["rmse_ekf"] for n in CELL_ORDER]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, rmse_lse, width, label="LSE per-step", color="C3", alpha=0.85)
    ax.bar(x + width / 2, rmse_ekf, width, label="EKF", color="C0", alpha=0.85)
    bound = SIGMA_R / np.sqrt(np.array([2, 2, 3, 3, 4, 4]))
    ax.plot(x, bound, "k--", lw=1.0, alpha=0.7, label="σ_r/√N (theoretical)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("position RMSE [m] (log)")
    ax.set_title("Position RMSE by sensor geometry — single run")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, "geometry_rmse_bar.png")
    plt.close(fig)


def main() -> None:
    all_results = {name: run_one(name) for name in CELL_ORDER}
    plot_grid(all_results)
    plot_rmse_bars(all_results)

    print(f"\n{'cell':<32s} {'LSE RMSE [m]':>14s} {'EKF RMSE [m]':>14s}")
    print("-" * 62)
    for name in CELL_ORDER:
        r = all_results[name]
        print(f"{GEOMETRY_LABELS[name]:<32s} {r['rmse_lse']:14.2f} {r['rmse_ekf']:14.2f}")

    save_csv(
        {
            "geometry": [GEOMETRY_LABELS[n] for n in CELL_ORDER],
            "rmse_lse_m": [all_results[n]["rmse_lse"] for n in CELL_ORDER],
            "rmse_ekf_m": [all_results[n]["rmse_ekf"] for n in CELL_ORDER],
        },
        "geometry_rmse.csv",
    )
    print(f"\nResults table written to {OUTPUT_RESULTS / 'geometry_rmse.csv'}")


if __name__ == "__main__":
    main()
