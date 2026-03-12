"""Kinematic bicycle model for the ego vehicle.

Implements the standard single-track (bicycle) kinematic model:

    beta  = arctan(lr / wheelbase * tan(delta))
    x    += v * cos(psi + beta) * dt
    y    += v * sin(psi + beta) * dt
    psi  += (v / lr) * sin(beta) * dt
    v    += a * dt

where delta is the front steering angle and a is the longitudinal
acceleration derived from throttle/brake inputs.
"""

import math

from self_driving.models import (
    ControlInput,
    Pose,
    VehicleParams,
    VehicleState,
)

DEFAULT_PARAMS = VehicleParams()


def integrate_bicycle_model(
    state: VehicleState,
    control: ControlInput,
    dt: float,
    params: VehicleParams = DEFAULT_PARAMS,
) -> VehicleState:
    """Advance the vehicle state by one time step dt.

    Pure function — no side effects.
    """
    # Apply steering change and clamp
    new_steering = state.steering_angle + control.steering_delta
    new_steering = max(
        -params.max_steering_angle, min(params.max_steering_angle, new_steering)
    )

    # Longitudinal acceleration from throttle/brake
    accel = (
        control.throttle * params.max_acceleration
        - control.brake * params.max_deceleration
    )

    # Bicycle model kinematics
    lr = params.wheelbase * params.lr_ratio
    beta = math.atan(lr / params.wheelbase * math.tan(new_steering))

    x = state.pose.x + state.speed * math.cos(state.pose.heading + beta) * dt
    y = state.pose.y + state.speed * math.sin(state.pose.heading + beta) * dt
    heading = state.pose.heading + (state.speed / max(lr, 0.01)) * math.sin(beta) * dt

    # Normalise heading to [-pi, pi]
    heading = (heading + math.pi) % (2 * math.pi) - math.pi

    # Speed integration with clamping
    new_speed = state.speed + accel * dt
    new_speed = max(0.0, min(params.max_speed, new_speed))

    return VehicleState(
        pose=Pose(x=x, y=y, heading=heading),
        speed=new_speed,
        acceleration=accel,
        steering_angle=new_steering,
        timestamp=state.timestamp + dt,
    )
