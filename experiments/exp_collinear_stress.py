"""Stress test for the N=3 collinear geometry.

The MC summary shows N=3 collinear performs almost identically to N=3
equilateral triangle — surprising, since the classical mirror-image ambiguity
should make a sensor line degenerate. This script exposes when the degeneracy
*does* manifest by varying the LSE warm-start prior.

For three sensors on the line y = 0, every measurement set has two
range-consistent solutions (x, +y) and (x, -y). The Gauss-Newton LSE
converges to whichever side its starting point is on. Once the EKF is seeded,
the velocity term keeps it locked.

Three configurations:
  (A) prior above the line  — converges to the correct side
  (B) prior below the line  — converges to the mirror image
  (C) prior on the line     — pathological; depends on noise

Outputs:
  - printed RMSE table
  - figure: trajectory comparison for the three cases
  - CSV: collinear_stress.csv

Usage: python -m experiments.exp_collinear_stress
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src import SAMPLE_PERIOD
from src.ekf import run_ekf
from src.lse import initialize_state, lse_track
from src.measurements import simulate_toa, to_range
from src.metrics import position_error, save_csv, save_figure
from src.sensors import get_geometry
from src.trajectory import make_zigzag

SIGMA_A = 10.0
SEED = 42

CONFIGS: dict[str, np.ndarray] = {
    "above (correct side)": np.array([3000.0, 4000.0]),
    "below (mirror image)": np.array([3000.0, -4000.0]),
    "on the sensor line ": np.array([5000.0, 0.0]),
}


def _run(prior_p: np.ndarray, states_truth: np.ndarray, sensors: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    z, emit = simulate_toa(states_truth, sensors, rng=rng)
    r = to_range(z, emit)
    lse_pos = lse_track(r, sensors, prior_p)
    x0, P0 = initialize_state(r[:2], sensors, T=SAMPLE_PERIOD, prior_p=prior_p)
    ekf_states, _, _ = run_ekf(r, sensors, x0, P0, sigma_a=SIGMA_A)
    err_ekf = position_error(ekf_states, states_truth)
    err_lse = np.linalg.norm(lse_pos - states_truth[:, :2], axis=1)
    rmse_ekf = float(np.sqrt(np.mean(err_ekf[5:] ** 2)))
    rmse_lse = float(np.sqrt(np.mean(err_lse[5:] ** 2)))
    return {
        "ekf_states": ekf_states,
        "lse_pos": lse_pos,
        "rmse_ekf": rmse_ekf,
        "rmse_lse": rmse_lse,
    }


def main() -> None:
    states_truth, _ = make_zigzag()
    sensors = get_geometry("N3_collinear")

    results = {name: _run(prior, states_truth, sensors) for name, prior in CONFIGS.items()}

    print("\nCollinear N=3 stress — effect of LSE warm-start prior")
    print(f"{'prior position':<22s} {'LSE RMSE [m]':>14s} {'EKF RMSE [m]':>14s}")
    print("-" * 54)
    for name, d in results.items():
        print(f"{name:<22s} {d['rmse_lse']:>14.2f} {d['rmse_ekf']:>14.2f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        states_truth[:, 0],
        states_truth[:, 1],
        "k-",
        lw=2.0,
        alpha=0.85,
        label="ground truth",
        zorder=5,
    )
    ax.scatter(
        sensors[:, 0],
        sensors[:, 1],
        marker="^",
        s=140,
        c="red",
        edgecolors="black",
        zorder=6,
        label="sensors",
    )
    ax.axhline(0.0, color="grey", lw=0.6, ls=":", alpha=0.6)
    colors = ["C0", "C1", "C2"]
    for (name, d), color in zip(results.items(), colors):
        ax.plot(
            d["ekf_states"][:, 0],
            d["ekf_states"][:, 1],
            color=color,
            lw=1.6,
            label=f"EKF — prior {name.strip()} (RMSE {d['rmse_ekf']:.1f} m)",
            zorder=4,
        )
        ax.plot(
            CONFIGS[name][0],
            CONFIGS[name][1],
            "x",
            color=color,
            ms=11,
            mew=2.2,
            zorder=7,
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_xlim(-500, 10500)
    ax.set_ylim(-7000, 9000)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        "N=3 collinear sensors — EKF locks onto whichever side the prior picks\n"
        "(crosses are LSE warm-start positions)"
    )
    ax.legend(loc="upper right", fontsize=9)
    save_figure(fig, "collinear_stress.png")
    plt.close(fig)

    save_csv(
        {
            "prior_label": list(results.keys()),
            "prior_x_m": [CONFIGS[n][0] for n in results],
            "prior_y_m": [CONFIGS[n][1] for n in results],
            "rmse_lse_m": [results[n]["rmse_lse"] for n in results],
            "rmse_ekf_m": [results[n]["rmse_ekf"] for n in results],
        },
        "collinear_stress.csv",
    )


if __name__ == "__main__":
    main()
