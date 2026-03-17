"""Tests for path planning layer."""

import math

import pytest

from self_driving.map_gen import generate_road_map
from self_driving.models import (
    DetectedObject,
    LocalizationOutput,
    MapConfig,
    Pose,
    RoadEdge,
    RoadMap,
    RoadNode,
    Route,
    Vector2,
)
from self_driving.path_planning import (
    _TRAJECTORY_POINTS,
    _build_arc_lengths,
    _eval_quintic,
    _route_edge_for_ego,
    _solve_quintic_lateral,
    cartesian_to_frenet,
    frenet_to_cartesian,
    plan_path,
)
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


# ---------------------------------------------------------------------------
# Quintic polynomial tests
# ---------------------------------------------------------------------------


def test_quintic_boundary_conditions():
    """The solved quintic must satisfy all 6 boundary conditions."""
    d0, d0p, d0pp = 2.0, 0.3, 0.0
    d_target = 5.25
    T = 28.5
    coeffs = _solve_quintic_lateral(d0, d0p, d0pp, d_target, T)

    def deriv1(c, s):
        a0, a1, a2, a3, a4, a5 = c
        return a1 + 2 * a2 * s + 3 * a3 * s**2 + 4 * a4 * s**3 + 5 * a5 * s**4

    def deriv2(c, s):
        a0, a1, a2, a3, a4, a5 = c
        return 2 * a2 + 6 * a3 * s + 12 * a4 * s**2 + 20 * a5 * s**3

    assert math.isclose(_eval_quintic(coeffs, 0.0), d0, abs_tol=1e-9)
    assert math.isclose(deriv1(coeffs, 0.0), d0p, abs_tol=1e-9)
    assert math.isclose(deriv2(coeffs, 0.0), d0pp, abs_tol=1e-9)
    assert math.isclose(_eval_quintic(coeffs, T), d_target, abs_tol=1e-6)
    assert math.isclose(deriv1(coeffs, T), 0.0, abs_tol=1e-6)
    assert math.isclose(deriv2(coeffs, T), 0.0, abs_tol=1e-6)


def test_quintic_flat_when_already_at_target():
    """When d0 == d_target and d0_prime == 0, the polynomial is flat."""
    d_target = 5.25
    coeffs = _solve_quintic_lateral(d_target, 0.0, 0.0, d_target, 28.5)
    for s in [0.0, 5.0, 14.0, 28.5]:
        assert math.isclose(_eval_quintic(coeffs, s), d_target, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Frenet round-trip tests
# ---------------------------------------------------------------------------


def _straight_centerline(length: float = 30.0, n: int = 20) -> list[Vector2]:
    """Straight east-heading centreline of given length."""
    step = length / (n - 1)
    return [Vector2(x=i * step, y=0.0) for i in range(n)]


def test_frenet_roundtrip_straight():
    """cartesian_to_frenet → frenet_to_cartesian recovers original point."""
    cl = _straight_centerline()
    arcs = _build_arc_lengths(cl)
    pose = Pose(x=10.0, y=2.5, heading=0.0)
    s, d = cartesian_to_frenet(pose, cl, arcs)
    x, y, h = frenet_to_cartesian(s, d, cl, arcs)
    assert math.isclose(x, pose.x, abs_tol=0.2)
    assert math.isclose(y, pose.y, abs_tol=0.2)


def test_frenet_d_positive_right_of_centreline():
    """A point to the right (south) of an eastward road has positive d."""
    cl = _straight_centerline()
    arcs = _build_arc_lengths(cl)
    # Eastward road, so south = right → positive d
    pose = Pose(x=10.0, y=-3.5, heading=0.0)
    _, d = cartesian_to_frenet(pose, cl, arcs)
    assert d > 0


# ---------------------------------------------------------------------------
# Emergency stop test
# ---------------------------------------------------------------------------


def test_emergency_stop_on_close_obstacle(road_map, route):
    """An obstacle directly ahead within 4 m triggers an emergency stop."""
    node = road_map.nodes[0]
    ego_pose = Pose(x=node.position.x, y=node.position.y, heading=0.0)
    obstacle = DetectedObject(
        object_id=99,
        object_type="vehicle",
        pose=Pose(x=node.position.x + 2.0, y=node.position.y, heading=0.0),
        velocity=Vector2(x=0.0, y=0.0),
        confidence=1.0,
    )
    loc = LocalizationOutput(
        timestamp=0.0,
        estimated_pose=ego_pose,
        detected_objects=[obstacle],
        nearest_road_edge=road_map.edges[0] if road_map.edges else None,
    )
    traj = plan_path(route, loc, road_map)
    assert traj.is_emergency_stop


# ---------------------------------------------------------------------------
# Route-edge selection tests
# ---------------------------------------------------------------------------

_PARAMS = dict(length=50.0, speed_limit=8.33, lane_width=3.5, num_lanes=2)


def _make_4way_intersection_map() -> RoadMap:
    """Noise-free + intersection: nodes W(3)–C(0)–E(1), N(4), S(2)."""
    nodes = [
        RoadNode(node_id=0, position=Vector2(x=50.0, y=50.0)),   # centre
        RoadNode(node_id=1, position=Vector2(x=100.0, y=50.0)),  # east
        RoadNode(node_id=2, position=Vector2(x=50.0, y=0.0)),    # south
        RoadNode(node_id=3, position=Vector2(x=0.0, y=50.0)),    # west
        RoadNode(node_id=4, position=Vector2(x=50.0, y=100.0)),  # north
    ]
    edges = [
        RoadEdge(from_node=3, to_node=0, **_PARAMS),
        RoadEdge(from_node=0, to_node=3, **_PARAMS),
        RoadEdge(from_node=0, to_node=1, **_PARAMS),
        RoadEdge(from_node=1, to_node=0, **_PARAMS),
        RoadEdge(from_node=2, to_node=0, **_PARAMS),
        RoadEdge(from_node=0, to_node=2, **_PARAMS),
        RoadEdge(from_node=0, to_node=4, **_PARAMS),
        RoadEdge(from_node=4, to_node=0, **_PARAMS),
    ]
    return RoadMap(nodes=nodes, edges=edges, buildings=[], lane_connections=[])


def test_route_edge_stable_at_intersection():
    """_route_edge_for_ego picks the next route edge when ego just passed a 4-way intersection."""
    road_map = _make_4way_intersection_map()
    # Route goes west → centre → east
    route = Route(waypoint_ids=[3, 0, 1], total_distance=100.0, estimated_time=12.0)

    # Ego 1 m past the centre node, heading east
    ego_pose = Pose(x=51.0, y=50.0, heading=0.0)

    found = _route_edge_for_ego(route, ego_pose, road_map)
    assert found is not None
    assert found.from_node == 0
    assert found.to_node == 1


def test_plan_path_no_lateral_spike_at_intersection():
    """Trajectory stays near the road when ego is at a 4-way intersection transition."""
    road_map = _make_4way_intersection_map()
    route = Route(waypoint_ids=[3, 0, 1], total_distance=100.0, estimated_time=12.0)

    # Ego 1 m past centre node, heading east; pass the perpendicular south edge as
    # nearest_road_edge to simulate what nearest-distance lookup would wrongly return.
    ego_pose = Pose(x=51.0, y=50.0, heading=0.0)
    wrong_edge = RoadEdge(from_node=2, to_node=0, **_PARAMS)  # south → centre (wrong dir)
    loc = LocalizationOutput(
        timestamp=0.0,
        estimated_pose=ego_pose,
        detected_objects=[],
        nearest_road_edge=wrong_edge,
    )
    traj = plan_path(route, loc, road_map)

    # Without the fix, d0 would be ~50 m (ego is perpendicular to the south edge)
    # and the quintic spike would throw points far off the road.
    planning_horizon_m = 50.0
    for pt in traj.points:
        dist = math.sqrt((pt.pose.x - ego_pose.x) ** 2 + (pt.pose.y - ego_pose.y) ** 2)
        assert dist < planning_horizon_m, (
            f"Lateral spike: ({pt.pose.x:.1f},{pt.pose.y:.1f}) is {dist:.1f}m from ego"
        )
