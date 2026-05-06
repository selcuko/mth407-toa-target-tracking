"""End-to-end runner: baseline → geometry sweep → Monte Carlo → animations."""

import sys

from experiments import exp_animation, exp_baseline, exp_geometry, exp_monte_carlo


def main() -> None:
    skip_anim = "--no-anim" in sys.argv

    print("=" * 60)
    print("MTH407 — 2D target localization & tracking via TOA")
    print("=" * 60)

    print("\n[1/4] Baseline (N=4 square, single deterministic run)")
    exp_baseline.main()

    print("\n[2/4] Sensor-geometry sweep (6 cells, single run each)")
    exp_geometry.main()

    print("\n[3/4] Monte Carlo (100 trials × 6 cells)")
    exp_monte_carlo.main()

    if skip_anim:
        print("\n[4/4] Animations skipped (--no-anim)")
    else:
        print("\n[4/4] Simulation animations (6 cells, ~3 min)")
        exp_animation.main()

    print("\n" + "=" * 60)
    print("Done. Figures: output/figures/   Results: output/results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
