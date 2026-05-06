"""MTH407 dönem projesi — 2D target localization & tracking via TOA + EKF."""

C_LIGHT = 3.0e8
SIGMA_T = 1.0e-8
SIGMA_R = C_LIGHT * SIGMA_T

ARENA_SIZE = 10_000.0
SAMPLE_PERIOD = 0.5
DURATION = 60.0
TARGET_SPEED = 100.0
