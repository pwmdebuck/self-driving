"""Tests for traffic light phase simulation."""

from self_driving.map_gen import (
    add_traffic_lights,
    generate_road_map,
    initial_traffic_light_states,
    update_traffic_lights,
)
from self_driving.models import MapConfig, TrafficLight, TrafficLightPhase, Vector2


def _single_light(phase_offset: float = 0.0) -> TrafficLight:
    return TrafficLight(
        light_id=0,
        node_id=1,
        controlled_edges=[(0, 1)],
        green_duration=10.0,
        yellow_duration=3.0,
        red_duration=10.0,
        phase_offset=phase_offset,
    )


def test_initial_state_green():
    tl = _single_light(phase_offset=0.0)
    states = initial_traffic_light_states([tl])
    assert len(states) == 1
    assert states[0].phase == TrafficLightPhase.GREEN
    assert abs(states[0].time_in_phase) < 1e-9


def test_initial_state_with_offset_yellow():
    # phase_offset=10 → past green(10), in yellow
    tl = _single_light(phase_offset=10.0)
    states = initial_traffic_light_states([tl])
    assert states[0].phase == TrafficLightPhase.YELLOW


def test_initial_state_with_offset_red():
    # phase_offset=13 → past green(10)+yellow(3)=13, in red
    tl = _single_light(phase_offset=13.0)
    states = initial_traffic_light_states([tl])
    assert states[0].phase == TrafficLightPhase.RED


def test_green_transitions_to_yellow():
    tl = _single_light()
    states = initial_traffic_light_states([tl])
    # Advance 10 s in 0.1 s steps
    for _ in range(100):
        states = update_traffic_lights(states, [tl], dt=0.1)
    assert states[0].phase == TrafficLightPhase.YELLOW


def test_yellow_transitions_to_red():
    tl = _single_light()
    states = initial_traffic_light_states([tl])
    for _ in range(130):  # 13 s: past green(10) + yellow(3)
        states = update_traffic_lights(states, [tl], dt=0.1)
    assert states[0].phase == TrafficLightPhase.RED


def test_red_transitions_back_to_green():
    tl = _single_light()
    states = initial_traffic_light_states([tl])
    total_cycle = 100 + 30 + 100  # 10+3+10 s in 0.1 steps
    for _ in range(total_cycle):
        states = update_traffic_lights(states, [tl], dt=0.1)
    assert states[0].phase == TrafficLightPhase.GREEN


def test_time_remaining_decreases():
    tl = _single_light()
    states = initial_traffic_light_states([tl])
    initial_remaining = states[0].time_remaining
    states = update_traffic_lights(states, [tl], dt=1.0)
    assert states[0].time_remaining < initial_remaining


def test_multiple_lights_independent():
    tl1 = TrafficLight(
        light_id=0, node_id=1, controlled_edges=[(0, 1)],
        green_duration=10.0, yellow_duration=3.0, red_duration=10.0,
        phase_offset=0.0,
    )
    tl2 = TrafficLight(
        light_id=1, node_id=2, controlled_edges=[(2, 3)],
        green_duration=10.0, yellow_duration=3.0, red_duration=10.0,
        phase_offset=13.0,  # starts at RED
    )
    states = initial_traffic_light_states([tl1, tl2])
    assert states[0].phase == TrafficLightPhase.GREEN
    assert states[1].phase == TrafficLightPhase.RED


def test_world_initialises_traffic_light_states():
    """SimulationWorld auto-initialises states from road_map.traffic_lights."""
    from self_driving.models import Pose, VehicleState
    from self_driving.world import SimulationWorld

    cfg = MapConfig(grid_rows=3, grid_cols=3)
    rm, _ = generate_road_map(cfg)
    rm = add_traffic_lights(rm)

    ego = VehicleState(
        pose=Pose(x=0.0, y=0.0, heading=0.0),
        speed=0.0, acceleration=0.0, steering_angle=0.0, timestamp=0.0,
    )
    world = SimulationWorld(
        road_map=rm, ego_state=ego, actor_states=[],
        current_route=None, destination=None, clock=0.0,
    )
    assert len(world.traffic_light_states) == len(rm.traffic_lights)
