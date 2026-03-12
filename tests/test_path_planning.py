"""Tests for path planning layer."""

import pytest

from self_driving.map_gen import generate_road_map
from self_driving.models import (
    LocalizationOutput,
    MapConfig,
    Pose,
)
from self_driving.path_planning import _TRAJECTORY_POINTS, plan_path
from self_driving.routing import plan_route


@pytest.fixture
def road_map_graph():
    cfg = MapConfig(grid_rows=4, grid_cols=4, block_size_m=50.0)
    return generate_road_map(cfg, seed=0)


@pytest.fixture
def road_map(road_map_graph):
    return road_map_graph[0]


@pytest.fixture
def graph(road_map_graph):
    return road_map_graph[1]


@pytest.fixture
def route(road_map, graph):
    start = road_map.nodes[0].node_id
    dest = road_map.nodes[-1].node_id
    return plan_route(graph, road_map, start, dest)


@pytest.fixture
def loc(road_map) -> LocalizationOutput:
    node = road_map.nodes[0]
    return LocalizationOutput(
        timestamp=0.0,
        estimated_pose=Pose(x=node.position.x, y=node.position.y, heading=0.0),
        detected_objects=[],
        nearest_road_edge=road_map.edges[0] if road_map.edges else None,
    )


def test_trajectory_has_correct_length(route, loc, road_map):
    traj = plan_path(route, loc, road_map)
    assert len(traj.points) == _TRAJECTORY_POINTS


def test_no_emergency_stop_on_clear_path(route, loc, road_map):
    traj = plan_path(route, loc, road_map)
    assert not traj.is_emergency_stop


def test_trajectory_speeds_positive(route, loc, road_map):
    traj = plan_path(route, loc, road_map)
    for pt in traj.points:
        assert pt.speed >= 0.0


def test_trajectory_timestamp_matches_loc(route, loc, road_map):
    traj = plan_path(route, loc, road_map)
    assert traj.timestamp == loc.timestamp
