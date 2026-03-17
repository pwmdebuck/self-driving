"""Entry point for the self-driving car simulation sandbox."""

import argparse
import math
import random

from self_driving.actors import create_pedestrian_actors, create_vehicle_actors
from self_driving.map_gen import generate_road_map
from self_driving.models import (
    MapConfig,
    Pose,
    SimConfig,
    VehicleState,
)
from self_driving.routing import plan_route
from self_driving.simulation import SimulationLoop
from self_driving.visualizer import Visualizer
from self_driving.world import SimulationWorld


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Self-Driving Simulation")
    parser.add_argument("--rows", type=int, default=6, help="Grid rows")
    parser.add_argument("--cols", type=int, default=6, help="Grid columns")
    parser.add_argument(
        "--block", type=float, default=60.0, help="Block size in metres"
    )
    parser.add_argument("--dest", type=int, default=None, help="Destination node ID")
    parser.add_argument("--steps", type=int, default=None, help="Max simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-render", action="store_true", help="Run headless (no window)"
    )
    parser.add_argument(
        "--no-obstacles", action="store_true", help="Disable all obstacle actors"
    )
    parser.add_argument(
        "--controller",
        choices=["mpc", "pure_pursuit"],
        default="mpc",
        help="Controller type (default: mpc)",
    )
    return parser.parse_args()


def main() -> None:
    """Build the world and run the simulation."""
    args = parse_args()
    rng = random.Random(args.seed)

    # --- Map ---
    map_cfg = MapConfig(
        grid_rows=args.rows,
        grid_cols=args.cols,
        block_size_m=args.block,
    )
    road_map, graph = generate_road_map(map_cfg, seed=args.seed)
    node_by_id = {n.node_id: n.position for n in road_map.nodes}

    # --- Destination: last node by default ---
    dest_node_id = args.dest if args.dest is not None else road_map.nodes[-1].node_id

    # --- Compute initial route to determine correct starting heading ---
    start_node = road_map.nodes[0]
    initial_route = plan_route(graph, road_map, start_node.node_id, dest_node_id)
    if len(initial_route.waypoint_ids) >= 2:
        n0 = node_by_id[initial_route.waypoint_ids[0]]
        n1 = node_by_id[initial_route.waypoint_ids[1]]
        initial_heading = math.atan2(n1.y - n0.y, n1.x - n0.x)
    else:
        initial_heading = 0.0

    # --- Ego vehicle: start in curbside lane of the first route edge ---
    # Nodes sit on the road centreline (d=0); lanes lie to the right of it.
    # Offset ego to the curbside lane centre so KEEP_LANE starts with no drift.
    first_edge = next(
        (
            e
            for e in road_map.edges
            if len(initial_route.waypoint_ids) >= 2
            and e.from_node == initial_route.waypoint_ids[0]
            and e.to_node == initial_route.waypoint_ids[1]
        ),
        None,
    )
    if first_edge is not None:
        curbside_d = (first_edge.num_lanes - 1 + 0.5) * first_edge.lane_width
        rx = math.sin(initial_heading)   # right-perpendicular of heading
        ry = -math.cos(initial_heading)
        start_x = start_node.position.x + rx * curbside_d
        start_y = start_node.position.y + ry * curbside_d
    else:
        start_x, start_y = start_node.position.x, start_node.position.y

    ego_state = VehicleState(
        pose=Pose(x=start_x, y=start_y, heading=initial_heading),
        speed=0.0,
        acceleration=0.0,
        steering_angle=0.0,
        timestamp=0.0,
    )

    # --- Actors ---
    if args.no_obstacles:
        actors = []
    else:
        vehicles = create_vehicle_actors(road_map, graph, count=4, rng=rng)
        pedestrians = create_pedestrian_actors(road_map, count=3, rng=rng)
        actors = vehicles + pedestrians

    # --- World ---
    world = SimulationWorld(
        road_map=road_map,
        ego_state=ego_state,
        actor_states=actors,
        current_route=None,
        destination=dest_node_id,
        clock=0.0,
    )

    # --- Simulation config ---
    sim_cfg = SimConfig(
        dt=0.05,
        destination_node=dest_node_id,
        render_every_n=2,
        max_steps=args.steps,
        controller_type=args.controller,
    )

    loop = SimulationLoop(world=world, config=sim_cfg)

    # --- Visualiser ---
    world_size = max(args.rows, args.cols) * args.block * 1.1
    viz = None if args.no_render else Visualizer(world_size_m=world_size)

    loop.run(visualizer=viz)


if __name__ == "__main__":
    main()
