"""Telemetry plotting utilities for integration tests and post-run analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from self_driving.simulation import SimulationLoop


def plot_run(loop: SimulationLoop, output_path: str | Path) -> None:
    """Save a telemetry plot for a completed simulation run.

    The figure has two sub-plots:
      - Top: XY trajectory of the ego vehicle.
      - Bottom (3 traces): steering delta, throttle, and brake over time.

    Parameters
    ----------
    loop:
        A ``SimulationLoop`` that has already been stepped at least once.
    output_path:
        File path for the saved figure (PNG / PDF / SVG etc. — determined by
        the suffix). Parent directories are created automatically.
    """
    import matplotlib.pyplot as plt  # imported lazily — not a hard dep at import time

    records = loop.telemetry
    if not records:
        raise ValueError("No telemetry recorded — run loop.step() first.")

    times = [r.time for r in records]
    xs = [r.x for r in records]
    ys = [r.y for r in records]
    steers = [r.steer for r in records]
    accel_cmds = [r.accel_cmd for r in records]

    timing_components = [
        ("mpc", "mpc_ms", "crimson"),
        ("path_planning", "path_planning_ms", "steelblue"),
        ("actors", "actors_ms", "goldenrod"),
        ("localization", "localization_ms", "mediumpurple"),
        ("behavior", "behavior_ms", "mediumseagreen"),
        ("routing", "routing_ms", "slategray"),
        ("physics", "physics_ms", "peru"),
    ]
    timing_data = {attr: [getattr(r.timings, attr) for r in records] for _, attr, _ in timing_components}

    fig, (ax_traj, ax_ctrl, ax_time) = plt.subplots(
        3, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [1.2, 1, 1]}
    )

    # --- Trajectory ---
    ax_traj.plot(xs, ys, color="steelblue", linewidth=1.5, label="ego path")
    ax_traj.scatter([xs[0]], [ys[0]], color="green", zorder=5, label="start")
    ax_traj.scatter([xs[-1]], [ys[-1]], color="red", zorder=5, label="end")
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.set_title("Ego Trajectory")
    ax_traj.legend(loc="best", fontsize=8)
    ax_traj.grid(True, linestyle="--", alpha=0.4)

    # --- Control traces ---
    ax_ctrl.plot(times, steers, label="steer delta (rad)", color="darkorange")
    ax_ctrl.plot(times, accel_cmds, label="accel cmd", color="seagreen")
    ax_ctrl.set_xlabel("time (s)")
    ax_ctrl.set_ylabel("value")
    ax_ctrl.set_title("Control Inputs Over Time")
    ax_ctrl.legend(loc="best", fontsize=8)
    ax_ctrl.grid(True, linestyle="--", alpha=0.4)

    # --- Timing per component ---
    bottom = [0.0] * len(records)
    for label, attr, color in timing_components:
        values = timing_data[attr]
        ax_time.fill_between(times, bottom, [b + v for b, v in zip(bottom, values)],
                             label=label, color=color, alpha=0.7)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax_time.set_xlabel("time (s)")
    ax_time.set_ylabel("ms per step")
    ax_time.set_title("Per-Component Step Time")
    ax_time.legend(loc="upper right", fontsize=8, ncol=2)
    ax_time.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
