"""Shared pytest fixtures."""

import pytest

from self_driving.map_gen import generate_road_map
from self_driving.models import (
    MapConfig,
    Pose,
    VehicleState,
)
from self_driving.world import SimulationWorld


@pytest.fixture
def small_map_config() -> MapConfig:
    return MapConfig(grid_rows=3, grid_cols=3, block_size_m=50.0)


@pytest.fixture
def road_map_and_graph(small_map_config: MapConfig):
    return generate_road_map(small_map_config, seed=0)


@pytest.fixture
def road_map(road_map_and_graph):
    return road_map_and_graph[0]


@pytest.fixture
def graph(road_map_and_graph):
    return road_map_and_graph[1]


@pytest.fixture
def default_ego() -> VehicleState:
    return VehicleState(
        pose=Pose(x=25.0, y=25.0, heading=0.0),
        speed=0.0,
        acceleration=0.0,
        steering_angle=0.0,
        timestamp=0.0,
    )


@pytest.fixture
def world(road_map, default_ego) -> SimulationWorld:
    return SimulationWorld(
        road_map=road_map,
        ego_state=default_ego,
        actor_states=[],
        current_route=None,
        destination=road_map.nodes[-1].node_id,
        clock=0.0,
    )
