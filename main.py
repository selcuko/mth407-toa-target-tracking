"""End-to-end runner for the MTH407 term project.

Stages: baseline → geometry sweep → Monte Carlo → collinear stress →
σ_a sensitivity → animations. Pass --no-anim to skip the slow GIF rendering.
"""

import sys

from experiments import (
    exp_animation,
    exp_baseline,
    exp_collinear_stress,
    exp_geometry,
    exp_monte_carlo,
    exp_sigma_a_sweep,
)


def main() -> None:
    skip_anim = "--no-anim" in sys.argv

    print("=" * 60)
    print("MTH407 — 2D target localization & tracking via TOA")
    print("=" * 60)

    print("\n[1/6] Baseline (N=4 square, single deterministic run)")
    exp_baseline.main()

    print("\n[2/6] Sensor-geometry sweep (6 cells, single run each)")
    exp_geometry.main()

    print("\n[3/6] Monte Carlo (100 trials × 6 cells)")
    exp_monte_carlo.main()

    print("\n[4/6] Collinear N=3 stress test (mirror-image ambiguity)")
    exp_collinear_stress.main()

    print("\n[5/6] σ_a sensitivity sweep (N=4 square)")
    exp_sigma_a_sweep.main()

    if skip_anim:
        print("\n[6/6] Animations skipped (--no-anim)")
    else:
        print("\n[6/6] Simulation animations (6 cells, ~3 min)")
        exp_animation.main()

    print("\n" + "=" * 60)
    print("Done. Figures: output/figures/   Results: output/results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
