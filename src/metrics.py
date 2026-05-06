"""RMSE, position-error helpers and plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FIGURES = Path(__file__).resolve().parents[1] / "output" / "figures"
OUTPUT_RESULTS = Path(__file__).resolve().parents[1] / "output" / "results"
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_RESULTS.mkdir(parents=True, exist_ok=True)


def position_error(estimated_states: np.ndarray, truth_states: np.ndarray) -> np.ndarray:
    return np.linalg.norm(estimated_states[:, :2] - truth_states[:, :2], axis=1)


def velocity_error(estimated_states: np.ndarray, truth_states: np.ndarray) -> np.ndarray:
    return np.linalg.norm(estimated_states[:, 2:] - truth_states[:, 2:], axis=1)


def rmse_position(estimated_states: np.ndarray, truth_states: np.ndarray, skip: int = 0) -> float:
    err = position_error(estimated_states, truth_states)[skip:]
    return float(np.sqrt(np.mean(err * err)))


def plot_trajectory(
    truth: np.ndarray,
    sensors: np.ndarray,
    estimates: dict[str, np.ndarray] | None = None,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.plot(truth[:, 0], truth[:, 1], "k-", lw=2.0, label="ground truth", zorder=4)
    ax.plot(truth[0, 0], truth[0, 1], "ko", ms=8, zorder=5)
    if estimates:
        styles = [("C0", "EKF", "-"), ("C3", "LSE-only", "--"), ("C2", "alt", ":")]
        for (color, default_label, ls), (label, est) in zip(styles, estimates.items()):
            ax.plot(est[:, 0], est[:, 1], color=color, ls=ls, lw=1.5, label=label, zorder=3)
    ax.scatter(
        sensors[:, 0],
        sensors[:, 1],
        marker="^",
        s=120,
        c="red",
        edgecolors="black",
        zorder=6,
        label="sensors",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    return ax


def plot_rmse_vs_time(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "Position error vs time",
    ax: plt.Axes | None = None,
    ylim_top: float | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for label, (t, err) in series.items():
        ax.plot(t, err, lw=1.4, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    if ylim_top is not None:
        ax.set_ylim(0, ylim_top)
    ax.legend(loc="best", fontsize=9)
    return ax


def plot_innovations(times: np.ndarray, innovs: np.ndarray, sigma_r: float, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for j in range(innovs.shape[1]):
        ax.plot(times, innovs[:, j], lw=1.0, alpha=0.7, label=f"sensor {j+1}")
    ax.axhline(3 * sigma_r, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(-3 * sigma_r, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("innovation [m]")
    ax.grid(True, alpha=0.3)
    ax.set_title("EKF innovation sequences (±3σ_r dashed)")
    ax.legend(loc="best", fontsize=9, ncol=4)
    return ax


def save_figure(fig: plt.Figure, name: str) -> Path:
    path = OUTPUT_FIGURES / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


def save_csv(table: dict, name: str) -> Path:
    path = OUTPUT_RESULTS / name
    keys = list(table.keys())
    rows = list(zip(*[table[k] for k in keys]))
    with path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    return path
