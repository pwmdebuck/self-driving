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
