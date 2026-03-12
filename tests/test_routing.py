"""Tests for routing layer."""

import pytest

from self_driving.map_gen import generate_road_map
from self_driving.models import MapConfig, Vector2
from self_driving.routing import find_nearest_node, plan_route


@pytest.fixture
def map_and_graph():
    cfg = MapConfig(grid_rows=4, grid_cols=4, block_size_m=50.0)
    return generate_road_map(cfg, seed=0)


def test_plan_route_returns_path(map_and_graph):
    road_map, graph = map_and_graph
    start = road_map.nodes[0].node_id
    dest = road_map.nodes[-1].node_id
    route = plan_route(graph, road_map, start, dest)
    assert len(route.waypoint_ids) >= 2
    assert route.waypoint_ids[0] == start
    assert route.waypoint_ids[-1] == dest


def test_plan_route_positive_distance(map_and_graph):
    road_map, graph = map_and_graph
    start = road_map.nodes[0].node_id
    dest = road_map.nodes[-1].node_id
    route = plan_route(graph, road_map, start, dest)
    assert route.total_distance > 0


def test_plan_route_same_start_dest(map_and_graph):
    road_map, graph = map_and_graph
    nid = road_map.nodes[0].node_id
    route = plan_route(graph, road_map, nid, nid)
    assert route.waypoint_ids == [nid]
    assert route.total_distance == 0.0


def test_find_nearest_node(map_and_graph):
    road_map, _ = map_and_graph
    # Query very close to first node
    pos = road_map.nodes[0].position
    result = find_nearest_node(road_map, Vector2(x=pos.x + 0.1, y=pos.y + 0.1))
    assert result == road_map.nodes[0].node_id


def test_route_estimated_time_positive(map_and_graph):
    road_map, graph = map_and_graph
    start = road_map.nodes[0].node_id
    dest = road_map.nodes[-1].node_id
    route = plan_route(graph, road_map, start, dest)
    assert route.estimated_time > 0
