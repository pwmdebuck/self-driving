"""Tests for the pure pursuit controller."""

import math

from self_driving.models import (
    ControlInput,
    LocalTrajectory,
    Pose,
    PurePursuitParams,
    TrajectoryPoint,
    VehicleParams,
    VehicleState,
    Vector2,
)
from self_driving.pure_pursuit import compute_pure_pursuit


def _make_state(
    x: float = 0.0,
    y: float = 0.0,
    heading: float = 0.0,
    speed: float = 5.0,
    steering: float = 0.0,
) -> VehicleState:
    return VehicleState(
        pose=Pose(x=x, y=y, heading=heading),
        speed=speed,
        acceleration=0.0,
        steering_angle=steering,
        timestamp=0.0,
    )


def _make_trajectory(
    points: list[tuple[float, float]],
    heading: float = 0.0,
    speed: float = 8.33,
    emergency_stop: bool = False,
) -> LocalTrajectory:
    return LocalTrajectory(
        timestamp=0.0,
        points=[
            TrajectoryPoint(
                pose=Pose(x=x, y=y, heading=heading), speed=speed, curvature=0.0
            )
            for x, y in points
        ],
        is_emergency_stop=emergency_stop,
    )


def test_straight_path_minimal_steering() -> None:
    """Ego aligned with a straight eastward trajectory → nearly zero steering."""
    state = _make_state(x=0.0, y=0.0, heading=0.0, speed=5.0)
    traj = _make_trajectory([(i * 2.0, 0.0) for i in range(20)])
    result = compute_pure_pursuit(state, traj)
    assert abs(result.steering_delta) < 0.01


def test_accelerates_toward_target_speed() -> None:
    """Ego at 0 m/s with 8.33 m/s target → positive accel_cmd."""
    state = _make_state(speed=0.0)
    traj = _make_trajectory([(i * 2.0, 0.0) for i in range(20)], speed=8.33)
    result = compute_pure_pursuit(state, traj)
    assert result.accel_cmd > 0.0


def test_brakes_above_target_speed() -> None:
    """Ego at 10 m/s with 0 m/s target → negative accel_cmd."""
    state = _make_state(speed=10.0)
    traj = _make_trajectory([(i * 2.0, 0.0) for i in range(20)], speed=0.0)
    result = compute_pure_pursuit(state, traj)
    assert result.accel_cmd < 0.0


def test_emergency_stop_returns_full_brake() -> None:
    """Emergency stop flag → accel_cmd == -1, steering_delta == 0."""
    state = _make_state()
    traj = _make_trajectory([(i * 2.0, 0.0) for i in range(20)], emergency_stop=True)
    result = compute_pure_pursuit(state, traj)
    assert result.accel_cmd == -1.0
    assert result.steering_delta == 0.0


def test_lateral_offset_steers_toward_path() -> None:
    """Ego south of an eastward trajectory steers left (negative delta)."""
    # Trajectory runs east at y=0; ego is at y=-3 (south = right of eastward road)
    # Pure pursuit should steer left (north) to return to path → delta < 0
    state = _make_state(x=0.0, y=-3.0, heading=0.0, speed=5.0, steering=0.0)
    traj = _make_trajectory([(i * 2.0, 0.0) for i in range(20)])
    result = compute_pure_pursuit(state, traj)
    assert result.steering_delta < 0.0


def test_output_bounds() -> None:
    """steering_delta and accel_cmd stay within configured limits."""
    pp = PurePursuitParams(max_steering_delta=0.05)
    vp = VehicleParams()
    state = _make_state(x=0.0, y=-10.0, heading=0.0, speed=0.0)
    traj = _make_trajectory([(i * 2.0, 0.0) for i in range(20)], speed=15.0)
    result = compute_pure_pursuit(state, traj, pp_params=pp, vehicle_params=vp)
    assert abs(result.steering_delta) <= pp.max_steering_delta + 1e-9
    assert -1.0 <= result.accel_cmd <= 1.0
