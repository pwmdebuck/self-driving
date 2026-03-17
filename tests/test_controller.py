"""Tests for the MPC controller."""

import pytest

from self_driving.controller import compute_mpc
from self_driving.models import (
    ControlInput,
    LocalTrajectory,
    Pose,
    TrajectoryPoint,
    VehicleState,
)


def _make_straight_trajectory(speed: float = 5.0, n: int = 10) -> LocalTrajectory:
    """Helper: straight-ahead trajectory at constant speed."""
    pts = [
        TrajectoryPoint(
            pose=Pose(x=float(i) * 1.5, y=0.0, heading=0.0),
            speed=speed,
            curvature=0.0,
        )
        for i in range(n)
    ]
    return LocalTrajectory(timestamp=0.0, points=pts, is_emergency_stop=False)


@pytest.fixture
def ego() -> VehicleState:
    return VehicleState(
        pose=Pose(x=0.0, y=0.0, heading=0.0),
        speed=5.0,
        acceleration=0.0,
        steering_angle=0.0,
        timestamp=0.0,
    )


def test_emergency_stop_full_brake(ego):
    traj = LocalTrajectory(timestamp=0.0, points=[], is_emergency_stop=True)
    ctrl = compute_mpc(ego, traj)
    assert ctrl.accel_cmd == -1.0


def test_control_bounds_respected(ego):
    traj = _make_straight_trajectory()
    ctrl = compute_mpc(ego, traj)
    assert -0.15 <= ctrl.steering_delta <= 0.15
    assert -1.0 <= ctrl.accel_cmd <= 1.0


def test_straight_path_low_steering(ego):
    """On a straight trajectory the steering delta should be small."""
    traj = _make_straight_trajectory()
    ctrl = compute_mpc(ego, traj)
    assert abs(ctrl.steering_delta) < 0.1


def test_returns_control_input_type(ego):
    traj = _make_straight_trajectory()
    result = compute_mpc(ego, traj)
    assert isinstance(result, ControlInput)
