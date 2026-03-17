"""Layer 4b — Model Predictive Control (MPC) controller.

Minimises a cost function over a receding horizon using
CasADi / IPOPT with analytical (automatic) differentiation.
Returns only the first control action (receding horizon principle).

Cost terms:
  - Cross-track error (distance from trajectory)
  - Heading error
  - Speed error
  - Control effort (penalises large inputs)
  - Rate of change (penalises control input changes between steps)

Performance note: the CasADi Opti problem (NLP graph + compiled gradients)
is built once per unique (MpcParams, VehicleParams) combination and cached.
Subsequent calls only set parameter values and call opti.solve().
"""

import math
from dataclasses import dataclass

import numpy as np
import casadi as ca

from self_driving.models import (
    ControlInput,
    LocalTrajectory,
    MpcParams,
    Pose,
    TrajectoryPoint,
    VehicleParams,
    VehicleState,
)

DEFAULT_MPC = MpcParams()
DEFAULT_VEHICLE = VehicleParams()


@dataclass
class _SolverCache:
    """Compiled CasADi Opti problem with symbolic handles for one (MpcParams, VehicleParams)."""

    opti: ca.Opti
    steer_delta: ca.MX   # decision variable, shape (n,)
    accel_cmd: ca.MX     # decision variable, shape (n,)
    p_state: ca.MX       # parameter: [x, y, psi, v, steer]
    p_refs: ca.MX        # parameter: (n+1) × 4  [x, y, psi, speed]
    p_max_steer: ca.MX   # parameter: scalar speed-dependent steering bound
    p_prev_ac: ca.MX     # parameter: scalar previous accel_cmd (for rate term)


_SOLVER_CACHE: dict[tuple[MpcParams, VehicleParams], _SolverCache] = {}


def _get_solver(mpc_params: MpcParams, vehicle_params: VehicleParams) -> _SolverCache:
    """Return the cached solver, building it on first call for this parameter set."""
    key = (mpc_params, vehicle_params)
    if key in _SOLVER_CACHE:
        return _SOLVER_CACHE[key]

    n = mpc_params.n_horizon
    opti = ca.Opti()

    steer_delta = opti.variable(n)
    accel_cmd = opti.variable(n)
    p_state = opti.parameter(5)        # [x, y, psi, v, steer]
    p_refs = opti.parameter(n + 1, 4)  # rows: horizon steps, cols: [x, y, psi, speed]
    p_max_steer = opti.parameter()
    p_prev_ac = opti.parameter()

    opti.subject_to(opti.bounded(-p_max_steer, steer_delta, p_max_steer))
    opti.subject_to(opti.bounded(-1.0, accel_cmd, 1.0))

    sx: ca.MX = p_state[0]
    sy: ca.MX = p_state[1]
    spsi: ca.MX = p_state[2]
    sv: ca.MX = p_state[3]
    ssteer: ca.MX = p_state[4]
    cost: ca.MX = ca.MX(0.0)
    prev_ac: ca.MX = p_prev_ac

    for k in range(n):
        sx, sy, spsi, sv, ssteer = _symbolic_bicycle_step(
            sx, sy, spsi, sv, ssteer,
            steer_delta[k], accel_cmd[k],
            mpc_params.dt, vehicle_params,
        )
        rx = p_refs[min(k + 1, n), 0]
        ry = p_refs[min(k + 1, n), 1]
        rpsi = p_refs[min(k + 1, n), 2]
        rspeed = p_refs[min(k + 1, n), 3]
        dh = ca.atan2(ca.sin(spsi - rpsi), ca.cos(spsi - rpsi))
        cost += (
            mpc_params.weight_cte * ((sx - rx) ** 2 + (sy - ry) ** 2)
            + mpc_params.weight_heading * dh ** 2
            + mpc_params.weight_speed * (sv - rspeed) ** 2
            + mpc_params.weight_effort * (steer_delta[k] ** 2 + accel_cmd[k] ** 2)
            + mpc_params.weight_rate * (accel_cmd[k] - prev_ac) ** 2
        )
        prev_ac = accel_cmd[k]

    opti.minimize(cost)
    opti.solver("ipopt", {
        "ipopt.print_level": 0,
        "ipopt.max_iter": 50,
        "ipopt.tol": 1e-4,
        "print_time": 0,
    })

    cache = _SolverCache(
        opti=opti,
        steer_delta=steer_delta,
        accel_cmd=accel_cmd,
        p_state=p_state,
        p_refs=p_refs,
        p_max_steer=p_max_steer,
        p_prev_ac=p_prev_ac,
    )
    _SOLVER_CACHE[key] = cache
    return cache


def _build_refs_array(traj_pts: list[TrajectoryPoint], n: int) -> np.ndarray:
    """Build an (n+1, 4) reference array [x, y, psi, speed] from trajectory points."""
    last = traj_pts[-1] if traj_pts else None
    rows = []
    for i in range(n + 1):
        pt = traj_pts[min(i, len(traj_pts) - 1)] if traj_pts else last
        rows.append([pt.pose.x, pt.pose.y, pt.pose.heading, pt.speed])  # type: ignore[union-attr]
    return np.array(rows, dtype=float)


def compute_mpc(
    state: VehicleState,
    trajectory: LocalTrajectory,
    mpc_params: MpcParams = DEFAULT_MPC,
    vehicle_params: VehicleParams = DEFAULT_VEHICLE,
    last_control: ControlInput | None = None,
) -> ControlInput:
    """Compute the optimal control input for the current vehicle state.

    If the trajectory is empty or an emergency stop is requested, returns
    a full-brake zero-steer command.

    Uses a feedforward warm start: the required speed change is spread
    evenly over the horizon, giving the optimiser a physically meaningful
    starting point that avoids pulsing throttle/brake behaviour.
    """
    if trajectory.is_emergency_stop or not trajectory.points:
        return ControlInput(steering_delta=0.0, accel_cmd=-1.0)

    n = mpc_params.n_horizon
    traj_pts = trajectory.points[: n + 1]
    ref_speed = traj_pts[0].speed if traj_pts else 0.0
    speed_err = ref_speed - state.speed
    horizon_time = n * mpc_params.dt

    x0 = np.zeros(n * 2)
    if last_control is not None:
        x0[0::2] = last_control.steering_delta
        x0[1::2] = last_control.accel_cmd

    if speed_err > 0.1:
        ff = min(speed_err / (vehicle_params.max_acceleration * horizon_time), 1.0)
        x0[1::2] = ff
    elif speed_err < -0.1:
        ff = min(-speed_err / (vehicle_params.max_deceleration * horizon_time), 1.0)
        x0[1::2] = -ff

    max_steer_delta = _max_steer_delta(state.speed, vehicle_params.max_speed)

    cache = _get_solver(mpc_params, vehicle_params)
    opti = cache.opti

    opti.set_value(cache.p_state, [
        state.pose.x, state.pose.y, state.pose.heading,
        state.speed, state.steering_angle,
    ])
    opti.set_value(cache.p_refs, _build_refs_array(traj_pts, n))
    opti.set_value(cache.p_max_steer, max_steer_delta)
    opti.set_value(cache.p_prev_ac, last_control.accel_cmd if last_control is not None else 0.0)

    opti.set_initial(cache.steer_delta, x0[0::2])
    opti.set_initial(cache.accel_cmd, x0[1::2])

    try:
        sol = opti.solve()
        sd0 = float(sol.value(cache.steer_delta[0]))
        ac0 = float(sol.value(cache.accel_cmd[0]))
        sd_vals = [float(sol.value(cache.steer_delta[k])) for k in range(n)]
        ac_vals = [float(sol.value(cache.accel_cmd[k])) for k in range(n)]
        predicted = _rollout_poses(state, sd_vals, ac_vals, mpc_params.dt, vehicle_params)
    except Exception:
        print("[MPC] IPOPT failed, using warm-start fallback")
        sd0, ac0 = float(x0[0]), float(x0[1])
        predicted = []

    return ControlInput(
        steering_delta=float(np.clip(sd0, -max_steer_delta, max_steer_delta)),
        accel_cmd=float(np.clip(ac0, -1.0, 1.0)),
        predicted_poses=predicted,
    )


def _max_steer_delta(speed: float, max_speed: float) -> float:
    """Return the max steering delta (rad/step) as a function of speed.

    Decays hyperbolically from 0.05 rad at rest to ~0.01 rad at max speed,
    ensuring stability at high speed while preserving low-speed manoeuvrability.
    """
    base = 0.05
    floor = 0.02
    k = (base - floor) / (floor * max_speed)
    return base / (1.0 + k * speed)


def _rollout_poses(
    state: VehicleState,
    sd_vals: list[float],
    ac_vals: list[float],
    dt: float,
    vp: VehicleParams,
) -> list[Pose]:
    """Numerically integrate the bicycle model over the MPC horizon.

    Returns one Pose per horizon step (not including the initial state).
    """
    lr = vp.wheelbase * vp.lr_ratio
    cx, cy, cpsi = state.pose.x, state.pose.y, state.pose.heading
    cv, csteer = state.speed, state.steering_angle
    poses: list[Pose] = []
    for sd, ac in zip(sd_vals, ac_vals):
        csteer = max(-vp.max_steering_angle, min(vp.max_steering_angle, csteer + sd))
        accel = ac * (vp.max_acceleration if ac >= 0 else vp.max_deceleration)
        beta = math.atan(lr / vp.wheelbase * math.tan(-csteer))
        cx += cv * math.cos(cpsi + beta) * dt
        cy += cv * math.sin(cpsi + beta) * dt
        cpsi += (cv / lr) * math.sin(beta) * dt
        cv = max(0.0, min(vp.max_speed, cv + accel * dt))
        poses.append(Pose(x=cx, y=cy, heading=cpsi))
    return poses


def _symbolic_bicycle_step(
    x: ca.MX,
    y: ca.MX,
    psi: ca.MX,
    v: ca.MX,
    steer: ca.MX,
    steer_delta: ca.MX,
    accel_cmd: ca.MX,
    dt: float,
    vp: VehicleParams,
) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX, ca.MX]:
    """Integrate the kinematic bicycle model one step using CasADi symbolics."""
    lr = vp.wheelbase * vp.lr_ratio
    new_steer = ca.fmin(ca.fmax(steer + steer_delta, -vp.max_steering_angle), vp.max_steering_angle)
    accel = ca.if_else(accel_cmd >= 0, accel_cmd * vp.max_acceleration, accel_cmd * vp.max_deceleration)
    beta = ca.atan(lr / vp.wheelbase * ca.tan(-new_steer))
    x_next = x + v * ca.cos(psi + beta) * dt
    y_next = y + v * ca.sin(psi + beta) * dt
    psi_next = psi + (v / lr) * ca.sin(beta) * dt
    v_next = ca.fmin(ca.fmax(v + accel * dt, 0.0), vp.max_speed)
    return x_next, y_next, psi_next, v_next, new_steer
