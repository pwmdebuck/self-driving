"""Tests for the kinematic bicycle model."""

import math

import pytest

from self_driving.models import ControlInput, Pose, VehicleParams, VehicleState
from self_driving.vehicle import integrate_bicycle_model


@pytest.fixture
def stationary_ego() -> VehicleState:
    return VehicleState(
        pose=Pose(x=0.0, y=0.0, heading=0.0),
        speed=0.0,
        acceleration=0.0,
        steering_angle=0.0,
        timestamp=0.0,
    )


@pytest.fixture
def moving_ego() -> VehicleState:
    return VehicleState(
        pose=Pose(x=0.0, y=0.0, heading=0.0),
        speed=5.0,
        acceleration=0.0,
        steering_angle=0.0,
        timestamp=0.0,
    )


def test_straight_driving_x_increases(moving_ego):
    control = ControlInput(steering_delta=0.0, throttle=0.0, brake=0.0)
    new_state = integrate_bicycle_model(moving_ego, control, dt=0.1)
    assert new_state.pose.x > moving_ego.pose.x
    assert abs(new_state.pose.y) < 0.01  # stays on y=0


def test_throttle_increases_speed(stationary_ego):
    control = ControlInput(steering_delta=0.0, throttle=1.0, brake=0.0)
    new_state = integrate_bicycle_model(stationary_ego, control, dt=0.1)
    assert new_state.speed > stationary_ego.speed


def test_brake_decreases_speed(moving_ego):
    control = ControlInput(steering_delta=0.0, throttle=0.0, brake=1.0)
    new_state = integrate_bicycle_model(moving_ego, control, dt=0.1)
    assert new_state.speed < moving_ego.speed


def test_speed_never_negative(moving_ego):
    control = ControlInput(steering_delta=0.0, throttle=0.0, brake=1.0)
    state = moving_ego
    for _ in range(100):
        state = integrate_bicycle_model(state, control, dt=0.1)
    assert state.speed >= 0.0


def test_steering_clamped(stationary_ego):
    params = VehicleParams(max_steering_angle=0.5)
    control = ControlInput(steering_delta=5.0, throttle=0.0, brake=0.0)
    new_state = integrate_bicycle_model(stationary_ego, control, dt=0.1, params=params)
    assert abs(new_state.steering_angle) <= params.max_steering_angle


def test_left_turn_changes_heading(moving_ego):
    control = ControlInput(steering_delta=0.3, throttle=0.1, brake=0.0)
    new_state = integrate_bicycle_model(moving_ego, control, dt=0.1)
    # Positive steering delta → heading should increase (CCW turn)
    assert new_state.pose.heading != moving_ego.pose.heading


def test_timestamp_advances(moving_ego):
    control = ControlInput(steering_delta=0.0, throttle=0.0, brake=0.0)
    dt = 0.05
    new_state = integrate_bicycle_model(moving_ego, control, dt=dt)
    assert abs(new_state.timestamp - (moving_ego.timestamp + dt)) < 1e-9


def test_heading_normalised(moving_ego):
    control = ControlInput(steering_delta=0.3, throttle=0.5, brake=0.0)
    state = moving_ego
    for _ in range(200):
        state = integrate_bicycle_model(state, control, dt=0.05)
    assert -math.pi <= state.pose.heading <= math.pi
