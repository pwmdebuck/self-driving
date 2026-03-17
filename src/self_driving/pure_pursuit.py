"""Pure pursuit path-tracking controller.

Geometric controller: finds a lookahead point on the trajectory and computes
the steering angle needed to arc toward it (Coulomb & Borenstein, 1992).
"""

import math

from self_driving.models import (
    ControlInput,
    LocalTrajectory,
    Pose,
    PurePursuitParams,
    VehicleParams,
    VehicleState,
)

_DEFAULT_PP = PurePursuitParams()
_DEFAULT_VP = VehicleParams()


def compute_pure_pursuit(
    state: VehicleState,
    trajectory: LocalTrajectory,
    pp_params: PurePursuitParams = _DEFAULT_PP,
    vehicle_params: VehicleParams = _DEFAULT_VP,
) -> ControlInput:
    """Compute steering + throttle via pure pursuit.

    Returns full-brake ControlInput on emergency stop or empty trajectory.
    """
    if trajectory.is_emergency_stop or not trajectory.points:
        return ControlInput(steering_delta=0.0, accel_cmd=-1.0)

    ego = state.pose
    lookahead_pose, target_speed = _find_lookahead(
        trajectory, ego, pp_params.lookahead_m
    )

    # Lateral error in vehicle frame, right-positive to match our steering convention
    # (positive steering → right turn, so right-offset target → positive delta)
    dx = lookahead_pose.x - ego.x
    dy = lookahead_pose.y - ego.y
    e = dx * math.sin(ego.heading) - dy * math.cos(ego.heading)
    L = math.sqrt(dx * dx + dy * dy) or pp_params.lookahead_m

    # Pure-pursuit steering geometry: δ = atan(2 * wb * e / L²)
    delta_target = math.atan2(
        2.0 * vehicle_params.wheelbase * e, L * L
    )
    delta_target = max(
        -vehicle_params.max_steering_angle,
        min(vehicle_params.max_steering_angle, delta_target),
    )
    raw_delta = delta_target - state.steering_angle
    steering_delta = max(
        -pp_params.max_steering_delta,
        min(pp_params.max_steering_delta, raw_delta),
    )

    # Proportional speed control
    speed_err = target_speed - state.speed
    accel_cmd = max(
        -1.0,
        min(1.0, pp_params.speed_gain * speed_err / vehicle_params.max_acceleration),
    )

    return ControlInput(steering_delta=steering_delta, accel_cmd=accel_cmd)


def _find_lookahead(
    trajectory: LocalTrajectory,
    ego: Pose,
    lookahead_m: float,
) -> tuple[Pose, float]:
    """Return the first trajectory point at or beyond lookahead_m from ego.

    Falls back to the last point if none reach the lookahead distance.
    """
    for pt in trajectory.points:
        d = math.sqrt((pt.pose.x - ego.x) ** 2 + (pt.pose.y - ego.y) ** 2)
        if d >= lookahead_m:
            return pt.pose, pt.speed
    last = trajectory.points[-1]
    return last.pose, last.speed
