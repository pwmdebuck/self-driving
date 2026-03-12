"""End-to-end integration test: ego drives along a straight single-row road.

No other actors are present so the test focuses purely on the routing →
path planning → MPC → bicycle-model pipeline.

To view this scenario in the Pygame visualiser run:
    uv run python main.py --rows 1 --cols 6
"""

import math

from self_driving.map_gen import generate_road_map
from self_driving.models import MapConfig, Pose, SimConfig, VehicleState
from self_driving.routing import plan_route
from self_driving.simulation import SimulationLoop
from self_driving.world import SimulationWorld

# Number of simulation steps used in most tests (0.05 s each → 5 s total)
_STEPS = 100


def _make_straight_world() -> tuple[SimulationWorld, int]:
    """Create a 1×5 straight road with the ego at the western end."""
    cfg = MapConfig(grid_rows=1, grid_cols=5, block_size_m=60.0)
    road_map, graph = generate_road_map(cfg, seed=0)

    start_node = road_map.nodes[0]
    dest_id = road_map.nodes[-1].node_id
    node_by_id = {n.node_id: n.position for n in road_map.nodes}

    # Set initial heading toward the first route step (not hardcoded 0)
    route = plan_route(graph, road_map, start_node.node_id, dest_id)
    if len(route.waypoint_ids) >= 2:
        n0 = node_by_id[route.waypoint_ids[0]]
        n1 = node_by_id[route.waypoint_ids[1]]
        heading = math.atan2(n1.y - n0.y, n1.x - n0.x)
    else:
        heading = 0.0

    ego = VehicleState(
        pose=Pose(
            x=start_node.position.x,
            y=start_node.position.y,
            heading=heading,
        ),
        speed=0.0,
        acceleration=0.0,
        steering_angle=0.0,
        timestamp=0.0,
    )
    world = SimulationWorld(
        road_map=road_map,
        ego_state=ego,
        actor_states=[],
        current_route=None,
        destination=dest_id,
        clock=0.0,
    )
    return world, dest_id


def _run(steps: int = _STEPS) -> SimulationLoop:
    """Build the world and run for *steps* ticks, returning the loop."""
    world, dest_id = _make_straight_world()
    cfg = SimConfig(dt=0.05, destination_node=dest_id, render_every_n=9999)
    loop = SimulationLoop(world=world, config=cfg)
    for _ in range(steps):
        loop.step()
    return loop


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ego_moves_forward() -> None:
    """Ego must travel a positive distance along the road."""
    world, dest_id = _make_straight_world()
    initial_x = world.ego_state.pose.x
    loop = _run()
    assert loop.world.ego_state.pose.x > initial_x + 5.0, (
        f"Expected ego to move >5 m forward, "
        f"moved {loop.world.ego_state.pose.x - initial_x:.2f} m"
    )


def test_ego_reaches_positive_speed() -> None:
    """Ego must accelerate away from zero within the first 50 steps."""
    world, dest_id = _make_straight_world()
    cfg = SimConfig(dt=0.05, destination_node=dest_id, render_every_n=9999)
    loop = SimulationLoop(world=world, config=cfg)
    for _ in range(50):
        loop.step()
    assert loop.world.ego_state.speed > 0.5, (
        f"Expected speed > 0.5 m/s, got {loop.world.ego_state.speed:.3f}"
    )


def test_ego_stays_near_road_centreline() -> None:
    """On a straight road with no obstacles the ego should not drift sideways."""
    world, dest_id = _make_straight_world()
    initial_y = world.ego_state.pose.y
    loop = _run(steps=200)
    lateral_drift = abs(loop.world.ego_state.pose.y - initial_y)
    assert lateral_drift < 15.0, (
        f"Lateral drift {lateral_drift:.2f} m exceeds 15 m — ego left the road"
    )


def test_no_emergency_stop_on_clear_road() -> None:
    """No emergency stop should fire when there are no obstacles."""
    world, dest_id = _make_straight_world()
    cfg = SimConfig(dt=0.05, destination_node=dest_id, render_every_n=9999)
    loop = SimulationLoop(world=world, config=cfg)
    for _ in range(_STEPS):
        loop.step()
        if loop.last_trajectory is not None:
            assert not loop.last_trajectory.is_emergency_stop, (
                "Emergency stop triggered unexpectedly on a clear road"
            )


def test_route_is_planned() -> None:
    """A valid route must be computed on the first step."""
    world, dest_id = _make_straight_world()
    cfg = SimConfig(dt=0.05, destination_node=dest_id, render_every_n=9999)
    loop = SimulationLoop(world=world, config=cfg)
    loop.step()
    assert loop.world.current_route is not None
    assert len(loop.world.current_route.waypoint_ids) >= 2


def test_trajectory_generated() -> None:
    """Path planning must produce a trajectory on the first step."""
    world, dest_id = _make_straight_world()
    cfg = SimConfig(dt=0.05, destination_node=dest_id, render_every_n=9999)
    loop = SimulationLoop(world=world, config=cfg)
    loop.step()
    assert loop.last_trajectory is not None
    assert len(loop.last_trajectory.points) > 0


def test_ego_reaches_destination() -> None:
    """Ego must come within 20 m of the destination within 1500 steps (~75 s)."""
    world, dest_id = _make_straight_world()
    node_by_id = {n.node_id: n.position for n in world.road_map.nodes}
    dest_pos = node_by_id[dest_id]

    cfg = SimConfig(dt=0.05, destination_node=dest_id, render_every_n=9999)
    loop = SimulationLoop(world=world, config=cfg)

    reached = False
    for _ in range(1500):
        loop.step()
        ego = loop.world.ego_state.pose
        dist = math.sqrt((ego.x - dest_pos.x) ** 2 + (ego.y - dest_pos.y) ** 2)
        if dist < 20.0:
            reached = True
            break

    assert reached, (
        f"Ego did not reach destination within 1500 steps. "
        f"Final position: ({loop.world.ego_state.pose.x:.1f}, "
        f"{loop.world.ego_state.pose.y:.1f}), "
        f"destination: ({dest_pos.x:.1f}, {dest_pos.y:.1f})"
    )
