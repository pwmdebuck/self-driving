"""Non-ego actor simulation: vehicles and pedestrians.

Vehicle actors follow the road graph using shortest-path navigation
and stop when another actor is directly ahead. Pedestrian actors
perform a random walk between nearby road nodes.
"""

import math
import random

import networkx as nx

from self_driving.models import (
    ActorState,
    Pose,
    RoadMap,
    Vector2,
)
from self_driving.routing import find_nearest_node

# Vehicle actor parameters
_VEHICLE_SPEED = 5.0  # m/s
_VEHICLE_STOP_DIST = 5.0  # metres — stop if another vehicle is ahead
_WAYPOINT_REACH_DIST = 3.0  # metres — close enough to consider node reached

# Pedestrian parameters
_PED_SPEED = 1.2  # m/s


def create_vehicle_actors(
    road_map: RoadMap,
    graph: nx.DiGraph,
    count: int,
    rng: random.Random,
) -> list[ActorState]:
    """Spawn vehicle actors at random nodes on the road map."""
    nodes = road_map.nodes
    actors: list[ActorState] = []
    for i in range(count):
        start = rng.choice(nodes)
        target = rng.choice(nodes)
        actors.append(
            ActorState(
                actor_id=i,
                actor_type="vehicle",
                pose=Pose(
                    x=start.position.x,
                    y=start.position.y,
                    heading=0.0,
                ),
                velocity=Vector2(x=0.0, y=0.0),
                target_node=target.node_id,
                next_node=None,  # resolved on first update
            )
        )
    return actors


def create_pedestrian_actors(
    road_map: RoadMap,
    count: int,
    rng: random.Random,
    id_offset: int = 100,
) -> list[ActorState]:
    """Spawn pedestrian actors at random nodes."""
    nodes = road_map.nodes
    actors: list[ActorState] = []
    for i in range(count):
        start = rng.choice(nodes)
        target = rng.choice(nodes)
        actors.append(
            ActorState(
                actor_id=id_offset + i,
                actor_type="pedestrian",
                pose=Pose(
                    x=start.position.x,
                    y=start.position.y,
                    heading=0.0,
                ),
                velocity=Vector2(x=0.0, y=0.0),
                target_node=target.node_id,
                next_node=None,
            )
        )
    return actors


def update_actors(
    actors: list[ActorState],
    road_map: RoadMap,
    graph: nx.DiGraph,
    dt: float,
    rng: random.Random,
) -> list[ActorState]:
    """Advance all actors by one time step.

    Returns a new list of updated ActorState objects.
    """
    updated: list[ActorState] = []
    for actor in actors:
        if actor.actor_type == "vehicle":
            updated.append(_update_vehicle(actor, actors, road_map, graph, dt, rng))
        else:
            updated.append(_update_pedestrian(actor, road_map, graph, dt, rng))
    return updated


# ---------------------------------------------------------------------------
# Private update functions
# ---------------------------------------------------------------------------


def _update_vehicle(
    actor: ActorState,
    all_actors: list[ActorState],
    road_map: RoadMap,
    graph: nx.DiGraph,
    dt: float,
    rng: random.Random,
) -> ActorState:
    """Move vehicle actor along the road graph toward its target node.

    Uses ``next_node`` as the stable immediate waypoint so the vehicle
    always travels along a single graph edge at a time rather than
    cutting diagonally across the map.  The path is only recomputed
    when the actor physically arrives at ``next_node``.
    """
    node_by_id = {n.node_id: n.position for n in road_map.nodes}

    target_id = actor.target_node
    next_node_id = actor.next_node

    # Assign initial destination if missing
    if target_id is None:
        target_id = rng.choice(road_map.nodes).node_id

    # Resolve immediate next waypoint on first step
    if next_node_id is None:
        next_node_id = _next_hop(graph, road_map, actor, target_id)

    next_pos = node_by_id.get(next_node_id)
    if next_pos is None:
        return actor

    dx_n = next_pos.x - actor.pose.x
    dy_n = next_pos.y - actor.pose.y
    dist_next = math.sqrt(dx_n * dx_n + dy_n * dy_n)

    # Arrived at next_node: advance to the following hop
    if dist_next < _WAYPOINT_REACH_DIST:
        if next_node_id == target_id:
            # Reached final destination — pick a new random one
            target_id = rng.choice(road_map.nodes).node_id
        next_node_id = _next_hop(graph, road_map, actor, target_id)
        next_pos = node_by_id.get(next_node_id)
        if next_pos is None:
            return actor.model_copy(
                update={
                    "target_node": target_id,
                    "next_node": next_node_id,
                    "velocity": Vector2(x=0.0, y=0.0),
                }
            )
        dx_n = next_pos.x - actor.pose.x
        dy_n = next_pos.y - actor.pose.y
        dist_next = math.sqrt(dx_n * dx_n + dy_n * dy_n)

    if dist_next < 1e-6:
        return actor

    heading = math.atan2(dy_n, dx_n)

    # Stop if another vehicle is directly ahead on this edge
    if _vehicle_ahead(actor, all_actors, heading):
        return actor.model_copy(
            update={
                "target_node": target_id,
                "next_node": next_node_id,
                "velocity": Vector2(x=0.0, y=0.0),
            }
        )

    vx = _VEHICLE_SPEED * dx_n / dist_next
    vy = _VEHICLE_SPEED * dy_n / dist_next

    return actor.model_copy(
        update={
            "target_node": target_id,
            "next_node": next_node_id,
            "pose": Pose(
                x=actor.pose.x + vx * dt,
                y=actor.pose.y + vy * dt,
                heading=heading,
            ),
            "velocity": Vector2(x=vx, y=vy),
        }
    )


def _next_hop(
    graph: nx.DiGraph,
    road_map: RoadMap,
    actor: ActorState,
    target_id: int,
) -> int:
    """Return the immediate next graph node on the shortest path to target_id."""
    current_node = find_nearest_node(road_map, Vector2(x=actor.pose.x, y=actor.pose.y))
    if current_node == target_id:
        return target_id
    try:
        path: list[int] = nx.shortest_path(  # type: ignore[attr-defined]
            graph, current_node, target_id, weight="weight"
        )
        return path[1] if len(path) > 1 else target_id
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return target_id


def _vehicle_ahead(
    actor: ActorState,
    all_actors: list[ActorState],
    heading: float,
) -> bool:
    """Return True if another vehicle is within stop distance directly ahead."""
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    for other in all_actors:
        if other.actor_id == actor.actor_id or other.actor_type != "vehicle":
            continue
        dx = other.pose.x - actor.pose.x
        dy = other.pose.y - actor.pose.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > _VEHICLE_STOP_DIST:
            continue
        # Project onto heading direction
        forward_proj = dx * cos_h + dy * sin_h
        if forward_proj > 0:
            return True
    return False


def _update_pedestrian(
    actor: ActorState,
    road_map: RoadMap,
    graph: nx.DiGraph,
    dt: float,
    rng: random.Random,
) -> ActorState:
    """Move pedestrian actor toward its target via direct straight-line walk."""
    target_id = actor.target_node
    if target_id is None:
        target_id = rng.choice(road_map.nodes).node_id

    node_by_id = {n.node_id: n.position for n in road_map.nodes}
    target_pos = node_by_id.get(target_id)
    if target_pos is None:
        return actor

    dx = target_pos.x - actor.pose.x
    dy = target_pos.y - actor.pose.y
    dist = math.sqrt(dx * dx + dy * dy)

    if dist < _WAYPOINT_REACH_DIST:
        new_target = rng.choice(road_map.nodes).node_id
        return actor.model_copy(
            update={
                "target_node": new_target,
                "velocity": Vector2(x=0.0, y=0.0),
            }
        )

    heading = math.atan2(dy, dx)
    vx = _PED_SPEED * dx / dist
    vy = _PED_SPEED * dy / dist

    return actor.model_copy(
        update={
            "target_node": target_id,
            "pose": Pose(
                x=actor.pose.x + vx * dt,
                y=actor.pose.y + vy * dt,
                heading=heading,
            ),
            "velocity": Vector2(x=vx, y=vy),
        }
    )
