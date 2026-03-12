"""Layer 1 — High-level route planning.

Uses NetworkX A* (with Euclidean heuristic) to find shortest paths
on the road network graph. Falls back to Dijkstra if A* fails.
"""

import math

import networkx as nx

from self_driving.models import Pose, RoadMap, Route, Vector2


def find_nearest_node(road_map: RoadMap, pos: Vector2) -> int:
    """Return the node id whose position is closest to pos."""
    best_id = road_map.nodes[0].node_id
    best_dist = math.inf
    for node in road_map.nodes:
        dx = node.position.x - pos.x
        dy = node.position.y - pos.y
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best_id = node.node_id
    return best_id


def find_nearest_node_from_pose(road_map: RoadMap, pose: Pose) -> int:
    """Convenience wrapper that accepts a Pose."""
    return find_nearest_node(road_map, Vector2(x=pose.x, y=pose.y))


def _euclidean_heuristic(graph: nx.DiGraph):  # type: ignore[return]
    """Return a heuristic function for A* based on Euclidean distance."""

    def heuristic(u: int, v: int) -> float:
        ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
        vx, vy = graph.nodes[v]["x"], graph.nodes[v]["y"]
        return math.sqrt((ux - vx) ** 2 + (uy - vy) ** 2)

    return heuristic


def plan_route(
    graph: nx.DiGraph,
    road_map: RoadMap,
    start_node: int,
    dest_node: int,
) -> Route:
    """Plan a route from start_node to dest_node.

    Uses A* with Euclidean heuristic; falls back to Dijkstra on failure.
    Raises nx.NetworkXNoPath if no path exists.
    """
    heuristic = _euclidean_heuristic(graph)
    try:
        path: list[int] = nx.astar_path(  # type: ignore[attr-defined]
            graph, start_node, dest_node, heuristic=heuristic, weight="weight"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        path = nx.dijkstra_path(  # type: ignore[attr-defined]
            graph, start_node, dest_node, weight="weight"
        )

    total_distance = _path_length(graph, path)
    avg_speed_limit = _average_speed_limit(graph, path)
    estimated_time = total_distance / avg_speed_limit if avg_speed_limit > 0 else 0.0

    return Route(
        waypoint_ids=path,
        total_distance=total_distance,
        estimated_time=estimated_time,
    )


def _path_length(graph: nx.DiGraph, path: list[int]) -> float:
    total = 0.0
    for u, v in zip(path, path[1:]):
        total += graph[u][v].get("weight", 0.0)
    return total


def _average_speed_limit(graph: nx.DiGraph, path: list[int]) -> float:
    if len(path) < 2:
        return 1.0
    limits = [graph[u][v].get("speed_limit", 8.33) for u, v in zip(path, path[1:])]
    return sum(limits) / len(limits)
