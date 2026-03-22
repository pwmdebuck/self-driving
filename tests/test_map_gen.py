"""Tests for map generation."""

from self_driving.map_gen import build_graph, generate_road_map
from self_driving.models import MapConfig


def test_node_count():
    cfg = MapConfig(grid_rows=3, grid_cols=4)
    road_map, _ = generate_road_map(cfg)
    assert len(road_map.nodes) == 12


def test_edge_count():
    # 3x4 grid: horizontal = 3*3=9 (each way=18), vertical = 2*4=8 (each way=16) = 34
    cfg = MapConfig(grid_rows=3, grid_cols=4)
    road_map, _ = generate_road_map(cfg)
    # horizontal: rows * (cols-1) * 2, vertical: (rows-1) * cols * 2
    expected = 3 * 3 * 2 + 2 * 4 * 2
    assert len(road_map.edges) == expected


def test_graph_connected():
    cfg = MapConfig(grid_rows=4, grid_cols=4)
    _, graph = generate_road_map(cfg)
    import networkx as nx

    assert nx.is_weakly_connected(graph)  # type: ignore[attr-defined]


def test_node_ids_unique():
    cfg = MapConfig(grid_rows=3, grid_cols=3)
    road_map, _ = generate_road_map(cfg)
    ids = [n.node_id for n in road_map.nodes]
    assert len(ids) == len(set(ids))


def test_reproducible_with_seed():
    cfg = MapConfig(grid_rows=4, grid_cols=4)
    rm1, _ = generate_road_map(cfg, seed=7)
    rm2, _ = generate_road_map(cfg, seed=7)
    assert rm1.nodes[0].position.x == rm2.nodes[0].position.x


def test_build_graph_node_count():
    cfg = MapConfig(grid_rows=3, grid_cols=3)
    road_map, _ = generate_road_map(cfg)
    graph = build_graph(road_map)
    assert graph.number_of_nodes() == len(road_map.nodes)


# ---------------------------------------------------------------------------
# New geometry and rich-road tests
# ---------------------------------------------------------------------------

import math

from self_driving.map_gen import (
    add_stop_signs,
    add_traffic_lights,
    build_ramp_merge,
    edge_arc_length,
    edge_polyline,
    generate_demo_map,
    make_curved_edge,
)
from self_driving.models import RoadNode, Vector2


def _simple_nodes() -> tuple[RoadNode, RoadNode]:
    a = RoadNode(node_id=0, position=Vector2(x=0.0, y=0.0))
    b = RoadNode(node_id=1, position=Vector2(x=100.0, y=0.0))
    return a, b


def test_edge_polyline_straight():
    a, b = _simple_nodes()
    cfg = MapConfig(grid_rows=2, grid_cols=2)
    rm, _ = generate_road_map(cfg, seed=0)
    edge = rm.edges[0]
    node_by_id = {n.node_id: n.position for n in rm.nodes}
    pts = edge_polyline(edge, node_by_id)
    assert pts[0] == node_by_id[edge.from_node]
    assert pts[-1] == node_by_id[edge.to_node]
    assert len(pts) == 2  # straight: just endpoints


def test_edge_arc_length_straight():
    pts = [Vector2(x=0.0, y=0.0), Vector2(x=3.0, y=4.0)]
    assert abs(edge_arc_length(pts) - 5.0) < 1e-9


def test_make_curved_edge_longer_than_chord():
    a, b = _simple_nodes()
    chord = 100.0
    curved = make_curved_edge(a, b, bulge=20.0, speed_limit=8.33, lane_width=3.5)
    assert curved.length > chord
    assert len(curved.control_points) == 2


def test_make_curved_edge_polyline_arc_length_matches():
    a, b = _simple_nodes()
    curved = make_curved_edge(a, b, bulge=20.0, speed_limit=8.33, lane_width=3.5)
    node_by_id = {0: a.position, 1: b.position}
    pts = edge_polyline(curved, node_by_id)
    assert abs(edge_arc_length(pts) - curved.length) < 1e-6


def test_build_ramp_merge_node_count():
    a = RoadNode(node_id=0, position=Vector2(x=0.0, y=0.0))
    b = RoadNode(node_id=1, position=Vector2(x=200.0, y=0.0))
    ramp = RoadNode(node_id=2, position=Vector2(x=120.0, y=-40.0))
    new_nodes, new_edges, conns = build_ramp_merge(a, b, ramp, merge_node_id=3)
    assert len(new_nodes) == 1  # just the merge node
    assert len(new_edges) == 3  # a→merge, merge→b, ramp→merge
    # All through-highway lanes have connections
    highway_num_lanes = 3
    through_conns = [c for c in conns if c.from_edge == (0, 3)]
    assert len(through_conns) == highway_num_lanes


def test_add_traffic_lights():
    cfg = MapConfig(grid_rows=3, grid_cols=3)
    rm, _ = generate_road_map(cfg)
    rm_with_lights = add_traffic_lights(rm)
    assert len(rm_with_lights.traffic_lights) > 0
    # Every light has staggered phase offsets
    offsets = [tl.phase_offset for tl in rm_with_lights.traffic_lights]
    assert len(set(offsets)) > 1  # at least some differ


def test_add_stop_signs():
    cfg = MapConfig(grid_rows=3, grid_cols=3)
    rm, _ = generate_road_map(cfg)
    node_id = rm.nodes[4].node_id  # central node
    rm_with_signs = add_stop_signs(rm, [node_id])
    stop_edges = [s.edge[1] for s in rm_with_signs.road_signs]
    assert node_id in stop_edges


def test_generate_demo_map_structure():
    rm, graph = generate_demo_map()
    # Graph must be weakly connected
    import networkx as nx
    assert nx.is_weakly_connected(graph)
    # Should have traffic lights at central nodes
    assert len(rm.traffic_lights) >= 4
    # Should have stop signs
    assert len(rm.road_signs) > 0
    # Should have at least one highway edge
    hw_edges = [e for e in rm.edges if e.edge_type == "highway"]
    assert len(hw_edges) >= 2
    # Should have at least one curved edge
    curved_edges = [e for e in rm.edges if e.control_points]
    assert len(curved_edges) >= 2
