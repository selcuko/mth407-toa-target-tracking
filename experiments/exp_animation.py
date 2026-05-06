"""Render simulation animations for representative geometry cells.

Outputs animated GIFs to output/figures/anim_<geometry>.gif showing target
motion, sensor positions, range rings (each sensor's instantaneous reading),
LSE per-step estimate, and EKF estimate with 2σ uncertainty ellipse, plus a
position-error panel with playhead.

Usage: python -m experiments.exp_animation
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from src import ARENA_SIZE, SAMPLE_PERIOD
from src.animate import render_animation
from src.ekf import run_ekf
from src.lse import initialize_state, lse_track
from src.measurements import simulate_toa, to_range
from src.metrics import OUTPUT_FIGURES
from src.sensors import GEOMETRY_LABELS, get_geometry
from src.trajectory import make_zigzag

SIGMA_A = 10.0
PRIOR_P = np.array([3000.0, 4000.0])
SEED = 42

CELLS_TO_ANIMATE = [
    "N2_wide",
    "N2_close",
    "N3_triangle",
    "N3_collinear",
    "N4_square",
    "N4_cluster",
]


def render_cell(geometry: str, fps: int = 12) -> None:
    states_truth, times = make_zigzag()
    sensors = get_geometry(geometry)

    rng = np.random.default_rng(SEED)
    z, emit = simulate_toa(states_truth, sensors, rng=rng)
    r = to_range(z, emit)

    lse_pos = lse_track(r, sensors, PRIOR_P)
    x0, P0 = initialize_state(r[:2], sensors, T=SAMPLE_PERIOD, prior_p=PRIOR_P)
    ekf_states, ekf_covs, _ = run_ekf(r, sensors, x0, P0, sigma_a=SIGMA_A)

    output_path = OUTPUT_FIGURES / f"anim_{geometry}.gif"
    title = f"Simulation — {GEOMETRY_LABELS[geometry]}"
    print(f"  rendering {output_path.name} (this takes ~10-30s) ...")
    render_animation(
        states_truth=states_truth,
        sensors=sensors,
        ranges=r,
        lse_positions=lse_pos,
        ekf_states=ekf_states,
        ekf_covs=ekf_covs,
        times=times,
        output_path=output_path,
        title=title,
        fps=fps,
        arena_size=ARENA_SIZE,
    )
    print(f"  saved {output_path}  ({output_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    print(f"Rendering {len(CELLS_TO_ANIMATE)} simulation animations...")
    for name in CELLS_TO_ANIMATE:
        print(f"\n[{name}] {GEOMETRY_LABELS[name]}")
        render_cell(name)
    print("\nDone. View GIFs in output/figures/.")


if __name__ == "__main__":
    main()
