"""Procedural road network generation.

Generates a grid-based road map where intersections become nodes and
horizontal/vertical connections become directed edges.
"""

import math
import random

import networkx as nx

from self_driving.models import (
    Building,
    MapConfig,
    RoadEdge,
    RoadMap,
    RoadNode,
    Vector2,
)


def generate_road_map(config: MapConfig, seed: int = 42) -> tuple[RoadMap, nx.DiGraph]:
    """Generate a grid road network from config.

    Returns both the serialisable RoadMap and the nx.DiGraph for routing.
    """
    rng = random.Random(seed)
    nodes: list[RoadNode] = []
    edges: list[RoadEdge] = []

    noise = config.block_size_m * 0.05  # 5 % position noise

    def node_id(row: int, col: int) -> int:
        return row * config.grid_cols + col

    # Create nodes at grid intersections with slight position noise
    for row in range(config.grid_rows):
        for col in range(config.grid_cols):
            x = col * config.block_size_m + rng.uniform(-noise, noise)
            y = row * config.block_size_m + rng.uniform(-noise, noise)
            nodes.append(
                RoadNode(
                    node_id=node_id(row, col),
                    position=Vector2(x=x, y=y),
                )
            )

    node_by_id = {n.node_id: n for n in nodes}

    def add_edge(from_id: int, to_id: int) -> None:
        a = node_by_id[from_id].position
        b = node_by_id[to_id].position
        length = math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
        edges.append(
            RoadEdge(
                from_node=from_id,
                to_node=to_id,
                length=length,
                speed_limit=config.speed_limit,
                lane_width=config.lane_width,
            )
        )

    # Horizontal edges (bidirectional)
    for row in range(config.grid_rows):
        for col in range(config.grid_cols - 1):
            add_edge(node_id(row, col), node_id(row, col + 1))
            add_edge(node_id(row, col + 1), node_id(row, col))

    # Vertical edges (bidirectional)
    for row in range(config.grid_rows - 1):
        for col in range(config.grid_cols):
            add_edge(node_id(row, col), node_id(row + 1, col))
            add_edge(node_id(row + 1, col), node_id(row, col))

    buildings = _generate_buildings(nodes, config, rng)
    road_map = RoadMap(nodes=nodes, edges=edges, buildings=buildings)
    graph = build_graph(road_map)
    return road_map, graph


def _generate_buildings(
    nodes: list[RoadNode], config: MapConfig, rng: random.Random
) -> list[Building]:
    """Place buildings in city blocks between intersections."""
    setback = config.lane_width + 3.0  # road half-width + sidewalk clearance
    buildings: list[Building] = []

    node_pos = {n.node_id: n.position for n in nodes}

    def nid(row: int, col: int) -> int:
        return row * config.grid_cols + col

    for row in range(config.grid_rows - 1):
        for col in range(config.grid_cols - 1):
            sw = node_pos[nid(row, col)]
            se = node_pos[nid(row, col + 1)]
            ne = node_pos[nid(row + 1, col + 1)]
            nw = node_pos[nid(row + 1, col)]

            # Usable zone inside the block after road setback
            min_x = max(sw.x, nw.x) + setback
            max_x = min(se.x, ne.x) - setback
            min_y = max(sw.y, se.y) + setback
            max_y = min(nw.y, ne.y) - setback

            zone_w = max_x - min_x
            zone_h = max_y - min_y
            if zone_w < 8.0 or zone_h < 8.0:
                continue

            # Place buildings in a 2×2 sub-grid within the block
            for br in range(2):
                for bc in range(2):
                    if rng.random() < 0.15:  # occasionally leave a cell empty
                        continue
                    cell_w = zone_w / 2
                    cell_h = zone_h / 2
                    cell_min_x = min_x + bc * cell_w
                    cell_min_y = min_y + br * cell_h

                    bw = rng.uniform(cell_w * 0.45, min(cell_w * 0.85, 20.0))
                    bd = rng.uniform(cell_h * 0.45, min(cell_h * 0.85, 20.0))
                    jitter_x = rng.uniform(-cell_w * 0.08, cell_w * 0.08)
                    jitter_y = rng.uniform(-cell_h * 0.08, cell_h * 0.08)
                    cx = cell_min_x + cell_w / 2 + jitter_x
                    cy = cell_min_y + cell_h / 2 + jitter_y

                    buildings.append(Building(cx=cx, cy=cy, width=bw, depth=bd))

    return buildings


def build_graph(road_map: RoadMap) -> nx.DiGraph:
    """Build a NetworkX DiGraph from a RoadMap."""
    graph: nx.DiGraph = nx.DiGraph()
    for node in road_map.nodes:
        graph.add_node(node.node_id, x=node.position.x, y=node.position.y)
    for edge in road_map.edges:
        graph.add_edge(
            edge.from_node,
            edge.to_node,
            weight=edge.length,
            speed_limit=edge.speed_limit,
        )
    return graph
