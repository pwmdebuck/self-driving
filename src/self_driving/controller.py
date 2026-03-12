"""Layer 4b — Model Predictive Control (MPC) controller.

Minimises a cost function over a receding horizon using
scipy.optimize.minimize (SLSQP). Returns only the first control
action (receding horizon principle).

Cost terms:
  - Cross-track error (distance from trajectory)
  - Heading error
  - Speed error
  - Control effort (penalises large inputs)
"""

import math

import numpy as np
from scipy.optimize import minimize

from self_driving.models import (
    ControlInput,
    LocalTrajectory,
    MpcParams,
    VehicleParams,
    VehicleState,
)

DEFAULT_MPC = MpcParams()
DEFAULT_VEHICLE = VehicleParams()


def compute_mpc(
    state: VehicleState,
    trajectory: LocalTrajectory,
    mpc_params: MpcParams = DEFAULT_MPC,
    vehicle_params: VehicleParams = DEFAULT_VEHICLE,
) -> ControlInput:
    """Compute the optimal control input for the current vehicle state.

    If the trajectory is empty or an emergency stop is requested, returns
    a full-brake zero-steer command.
    """
    if trajectory.is_emergency_stop or not trajectory.points:
        return ControlInput(steering_delta=0.0, throttle=0.0, brake=1.0)

    n = mpc_params.n_horizon
    traj_pts = trajectory.points[: n + 1]

    # Warm start: gentle throttle so the optimizer sees realistic dynamics
    # from step 0 (a zero-speed vehicle can't steer, biasing toward bad local minima).
    x0 = np.zeros(n * 3)  # [steer_delta, throttle, brake] * n
    x0[1::3] = 0.3  # throttle column

    bounds = []
    max_steer_delta = 0.1  # radians per step
    for _ in range(n):
        bounds += [
            (-max_steer_delta, max_steer_delta),  # steering delta
            (0.0, 1.0),  # throttle
            (0.0, 1.0),  # brake
        ]

    result = minimize(
        _cost_function,
        x0,
        args=(state, traj_pts, mpc_params, vehicle_params),
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": 50, "ftol": 1e-4},
    )

    controls: np.ndarray = result.x  # type: ignore[union-attr]
    return ControlInput(
        steering_delta=float(np.clip(controls[0], -max_steer_delta, max_steer_delta)),
        throttle=float(np.clip(controls[1], 0.0, 1.0)),
        brake=float(np.clip(controls[2], 0.0, 1.0)),
    )


def _cost_function(
    x: np.ndarray,
    state: VehicleState,
    traj_pts: list,
    mpc: MpcParams,
    vp: VehicleParams,
) -> float:
    """Compute total MPC cost over the horizon."""
    cost = 0.0
    cur_x = state.pose.x
    cur_y = state.pose.y
    cur_h = state.pose.heading
    cur_v = state.speed
    cur_steer = state.steering_angle
    lr = vp.wheelbase * vp.lr_ratio

    n = mpc.n_horizon
    for k in range(n):
        sd = x[k * 3]
        th = x[k * 3 + 1]
        br = x[k * 3 + 2]

        # Clamp steering
        new_steer = cur_steer + sd
        new_steer = max(-vp.max_steering_angle, min(vp.max_steering_angle, new_steer))

        # Propagate bicycle model
        beta = math.atan(lr / vp.wheelbase * math.tan(new_steer))
        accel = th * vp.max_acceleration - br * vp.max_deceleration
        cur_x += cur_v * math.cos(cur_h + beta) * mpc.dt
        cur_y += cur_v * math.sin(cur_h + beta) * mpc.dt
        cur_h += (cur_v / max(lr, 0.01)) * math.sin(beta) * mpc.dt
        cur_h = (cur_h + math.pi) % (2 * math.pi) - math.pi
        cur_v = max(0.0, min(vp.max_speed, cur_v + accel * mpc.dt))
        cur_steer = new_steer

        # Reference point for this step
        ref_idx = min(k + 1, len(traj_pts) - 1)
        ref = traj_pts[ref_idx]

        # Cross-track error
        cte = (cur_x - ref.pose.x) ** 2 + (cur_y - ref.pose.y) ** 2
        # Heading error
        dh = cur_h - ref.pose.heading
        dh = (dh + math.pi) % (2 * math.pi) - math.pi
        he = dh**2
        # Speed error
        se = (cur_v - ref.speed) ** 2
        # Control effort
        effort = sd**2 + th**2 + br**2

        cost += (
            mpc.weight_cte * cte
            + mpc.weight_heading * he
            + mpc.weight_speed * se
            + mpc.weight_effort * effort
        )

    return cost
