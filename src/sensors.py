"""Named sensor geometries for the project's experiment matrix.

Six configurations span N ∈ {2, 3, 4} × {favourable, degraded}.
"""

from __future__ import annotations

import numpy as np

from . import ARENA_SIZE


def widespread_2(side: float = ARENA_SIZE) -> np.ndarray:
    return np.array([[0.0, 0.0], [side, side]])


def closepair_2(side: float = ARENA_SIZE, spread: float = 500.0) -> np.ndarray:
    return np.array([[0.0, side / 2], [spread, side / 2]])


def triangle_3(side: float = ARENA_SIZE) -> np.ndarray:
    cx = cy = side / 2
    r = side / 2
    angles = np.deg2rad([90.0, 210.0, 330.0])
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])


def collinear_3(side: float = ARENA_SIZE) -> np.ndarray:
    return np.array([[0.1 * side, 0.0], [0.5 * side, 0.0], [0.9 * side, 0.0]])


def square_4(side: float = ARENA_SIZE) -> np.ndarray:
    return np.array([[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]])


def cluster_4(side: float = ARENA_SIZE, spread: float = 500.0) -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [spread, 0.0], [0.0, spread], [spread, spread]]
    )


GEOMETRIES: dict[str, callable] = {
    "N2_wide": widespread_2,
    "N2_close": closepair_2,
    "N3_triangle": triangle_3,
    "N3_collinear": collinear_3,
    "N4_square": square_4,
    "N4_cluster": cluster_4,
}

GEOMETRY_LABELS: dict[str, str] = {
    "N2_wide": "N=2, widespread (corners)",
    "N2_close": "N=2, close pair",
    "N3_triangle": "N=3, equilateral triangle",
    "N3_collinear": "N=3, collinear (degenerate)",
    "N4_square": "N=4, square (corners)",
    "N4_cluster": "N=4, clustered",
}


def get_geometry(name: str) -> np.ndarray:
    if name not in GEOMETRIES:
        raise KeyError(f"unknown geometry {name!r}; choose from {list(GEOMETRIES)}")
    return GEOMETRIES[name]()
