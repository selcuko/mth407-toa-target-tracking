"""Simulation visualization: animated 2D arena with sensors, target, and EKF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse


def _cov_ellipse_params(cov_2x2: np.ndarray, n_sigma: float = 2.0) -> tuple[float, float, float]:
    eigvals, eigvecs = np.linalg.eigh(cov_2x2)
    eigvals = np.maximum(eigvals, 0.0)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    width = 2.0 * n_sigma * float(np.sqrt(eigvals[0]))
    height = 2.0 * n_sigma * float(np.sqrt(eigvals[1]))
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    return width, height, angle


def render_animation(
    states_truth: np.ndarray,
    sensors: np.ndarray,
    ranges: np.ndarray,
    lse_positions: np.ndarray,
    ekf_states: np.ndarray,
    ekf_covs: np.ndarray,
    times: np.ndarray,
    output_path: Path,
    title: str = "",
    fps: int = 10,
    trail_length: int = 30,
    arena_size: float = 10_000.0,
    show_rings: bool = True,
    figsize: tuple[float, float] = (10.5, 11.5),
    dpi: int = 90,
) -> Path:
    """Render a multi-panel simulation animation as an animated GIF.

    Top panel: 2D arena with sensors (triangles), range rings around each
    sensor at the currently measured range, ground-truth target with motion
    trail, LSE per-step marker, and EKF estimate with 2σ uncertainty ellipse.

    Bottom panel: position error vs time with a playhead marking the current
    step.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.18)
    ax_main: plt.Axes = fig.add_subplot(gs[0])
    ax_err: plt.Axes = fig.add_subplot(gs[1])

    pad = 0.05 * arena_size
    ax_main.set_xlim(-pad, arena_size + pad)
    ax_main.set_ylim(-pad, arena_size + pad)
    ax_main.set_aspect("equal")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlabel("x [m]")
    ax_main.set_ylabel("y [m]")

    # Static: sensors
    ax_main.scatter(
        sensors[:, 0],
        sensors[:, 1],
        marker="^",
        s=140,
        c="red",
        edgecolors="black",
        zorder=10,
        label="sensors",
    )
    for i, s in enumerate(sensors):
        ax_main.annotate(
            f"S{i+1}",
            xy=(s[0], s[1]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            zorder=11,
        )

    # Faded full ground-truth track (static reference)
    ax_main.plot(
        states_truth[:, 0],
        states_truth[:, 1],
        color="black",
        lw=0.8,
        alpha=0.15,
        zorder=2,
    )

    # Dynamic: trails and current dots. EKF is drawn as a hollow ring at
    # higher zorder so the estimate stays visible even when it sits exactly
    # on top of the (smaller, filled) ground-truth marker.
    (truth_trail,) = ax_main.plot([], [], "k-", lw=2.0, alpha=0.85, zorder=5)
    (truth_dot,) = ax_main.plot(
        [], [], "ko", ms=8, zorder=7, label="ground truth"
    )
    (ekf_trail,) = ax_main.plot([], [], color="C0", lw=1.6, alpha=0.7, zorder=4)
    (ekf_dot,) = ax_main.plot(
        [],
        [],
        "o",
        markerfacecolor="none",
        markeredgecolor="C0",
        markeredgewidth=2.5,
        ms=18,
        zorder=9,
        label="EKF estimate",
    )
    (lse_dot,) = ax_main.plot(
        [], [], "x", color="C3", ms=13, mew=2.5, zorder=8, label="LSE per-step"
    )

    # Range rings
    rings: list[Circle] = []
    if show_rings:
        for _ in sensors:
            ring = Circle(
                (0.0, 0.0),
                0.0,
                fill=False,
                edgecolor="C7",
                lw=0.9,
                alpha=0.5,
                ls="--",
                zorder=3,
            )
            ax_main.add_patch(ring)
            rings.append(ring)

    # Uncertainty ellipse (2σ) for EKF position
    ellipse = Ellipse(
        (0.0, 0.0),
        0.0,
        0.0,
        angle=0.0,
        fill=False,
        edgecolor="C0",
        lw=1.6,
        alpha=0.85,
        zorder=6,
        label="EKF 2σ",
    )
    ax_main.add_patch(ellipse)

    if title:
        ax_main.set_title(title, fontsize=12)
    ax_main.legend(loc="upper right", fontsize=9, framealpha=0.92)

    # Bottom panel: position error vs time
    err_ekf = np.linalg.norm(ekf_states[:, :2] - states_truth[:, :2], axis=1)
    err_lse = np.linalg.norm(lse_positions - states_truth[:, :2], axis=1)

    ax_err.plot(times, err_lse, color="C3", lw=1.0, alpha=0.6, label="LSE error")
    ax_err.plot(times, err_ekf, color="C0", lw=1.4, label="EKF error")
    ymax = float(np.percentile(np.concatenate([err_lse, err_ekf]), 99) * 1.2)
    ymax = max(ymax, 5.0)
    ax_err.set_ylim(0, ymax)
    ax_err.set_xlim(times[0], times[-1])
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("position error [m]")
    ax_err.grid(alpha=0.3)
    ax_err.legend(loc="upper right", fontsize=9)
    playhead = ax_err.axvline(times[0], color="black", lw=1.5, alpha=0.8)
    (cur_ekf_marker,) = ax_err.plot([times[0]], [err_ekf[0]], "C0o", ms=8, zorder=5)

    # Time / step annotation — placed below main axes so it never collides
    # with sensor labels regardless of geometry.
    info = fig.text(
        0.02,
        0.96,
        "",
        ha="left",
        va="top",
        fontsize=11,
        fontfamily="monospace",
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "0.6"},
        zorder=12,
    )

    K = len(times)

    def update(k: int):
        start = max(0, k - trail_length + 1)
        truth_trail.set_data(states_truth[start : k + 1, 0], states_truth[start : k + 1, 1])
        truth_dot.set_data([states_truth[k, 0]], [states_truth[k, 1]])

        ekf_trail.set_data(ekf_states[start : k + 1, 0], ekf_states[start : k + 1, 1])
        ekf_dot.set_data([ekf_states[k, 0]], [ekf_states[k, 1]])

        lse_dot.set_data([lse_positions[k, 0]], [lse_positions[k, 1]])

        if show_rings:
            for i, ring in enumerate(rings):
                ring.center = (sensors[i, 0], sensors[i, 1])
                ring.set_radius(max(float(ranges[k, i]), 0.0))

        w, h, ang = _cov_ellipse_params(ekf_covs[k, :2, :2])
        ellipse.set_center((ekf_states[k, 0], ekf_states[k, 1]))
        ellipse.set_width(w)
        ellipse.set_height(h)
        ellipse.set_angle(ang)

        playhead.set_xdata([times[k], times[k]])
        cur_ekf_marker.set_data([times[k]], [err_ekf[k]])

        info.set_text(
            f"t = {times[k]:5.1f}s   k = {k:3d}/{K-1}\n"
            f"EKF err = {err_ekf[k]:5.2f} m\n"
            f"LSE err = {err_lse[k]:5.2f} m"
        )
        return ()

    anim = FuncAnimation(fig, update, frames=K, interval=1000 // fps, blit=False)
    writer = PillowWriter(fps=fps)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path
