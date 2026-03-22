"""Tests for the behavioural planner FSM."""

import pytest

from self_driving.behavioral_planner import plan_behavior
from self_driving.map_gen import generate_road_map
from self_driving.models import (
    BehaviorOutput,
    BehaviorState,
    DetectedObject,
    LocalizationOutput,
    MapConfig,
    Pose,
    Route,
    Vector2,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def road_map():
    cfg = MapConfig(grid_rows=3, grid_cols=3, block_size_m=50.0, default_num_lanes=2)
    rm, _ = generate_road_map(cfg, seed=0)
    return rm


@pytest.fixture
def route(road_map):
    start = road_map.nodes[0].node_id
    dest = road_map.nodes[-1].node_id
    return Route(
        waypoint_ids=[n.node_id for n in road_map.nodes],
        total_distance=200.0,
        estimated_time=30.0,
    )


def _make_loc(road_map, ego_pose: Pose, objects: list[DetectedObject]) -> LocalizationOutput:
    return LocalizationOutput(
        timestamp=0.0,
        estimated_pose=ego_pose,
        detected_objects=objects,
        nearest_road_edge=road_map.edges[0] if road_map.edges else None,
    )


def _ego_at_node(road_map, node_idx: int = 0) -> Pose:
    node = road_map.nodes[node_idx]
    return Pose(x=node.position.x, y=node.position.y, heading=0.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_keeps_curbside_lane(road_map, route):
    """Without prior state, FSM defaults to KEEP_LANE on the curbside lane."""
    ego = _ego_at_node(road_map)
    loc = _make_loc(road_map, ego, [])
    out = plan_behavior(loc, route, road_map, prev=None)
    assert out.state == BehaviorState.KEEP_LANE
    assert out.target_lane == road_map.edges[0].num_lanes - 1


def test_keep_lane_on_clear_road(road_map, route):
    """No obstacles → stay KEEP_LANE."""
    ego = _ego_at_node(road_map)
    loc = _make_loc(road_map, ego, [])
    prev = BehaviorOutput(
        state=BehaviorState.KEEP_LANE,
        target_lane=road_map.edges[0].num_lanes - 1,
        target_speed=road_map.edges[0].speed_limit,
    )
    out = plan_behavior(loc, route, road_map, prev=prev)
    assert out.state == BehaviorState.KEEP_LANE
    assert out.target_lane == prev.target_lane


def test_lane_change_left_when_obstacle_ahead(road_map, route):
    """Obstacle in curbside lane ahead + inner lane clear → CHANGE_LANE_LEFT."""
    edge = road_map.edges[0]
    curbside = edge.num_lanes - 1
    ego = _ego_at_node(road_map)

    # Place obstacle in the curbside lane, 8 m ahead along the edge
    node_a = road_map.nodes[edge.from_node] if hasattr(road_map.nodes[0], 'node_id') else None
    # Find from_node position
    node_by_id = {n.node_id: n.position for n in road_map.nodes}
    a = node_by_id[edge.from_node]
    b = node_by_id[edge.to_node]
    import math
    heading = math.atan2(b.y - a.y, b.x - a.x)
    # 8 m ahead along road, shifted to curbside lane centre
    rx, ry = math.sin(heading), -math.cos(heading)
    lane_center_d = (curbside + 0.5) * edge.lane_width
    obs_x = ego.x + math.cos(heading) * 8.0 + rx * lane_center_d
    obs_y = ego.y + math.sin(heading) * 8.0 + ry * lane_center_d

    obstacle = DetectedObject(
        object_id=1,
        object_type="vehicle",
        pose=Pose(x=obs_x, y=obs_y, heading=0.0),
        velocity=Vector2(x=0.0, y=0.0),
        confidence=1.0,
    )
    loc = _make_loc(road_map, ego, [obstacle])
    prev = BehaviorOutput(
        state=BehaviorState.KEEP_LANE,
        target_lane=curbside,
        target_speed=edge.speed_limit,
    )
    out = plan_behavior(loc, route, road_map, prev=prev)
    assert out.state == BehaviorState.CHANGE_LANE_LEFT
    assert out.target_lane == curbside - 1


def test_lane_change_completes_when_d_within_tolerance(road_map, route):
    """CHANGE_LANE_LEFT → KEEP_LANE once ego d is within 0.5 m of target."""
    edge = road_map.edges[0]
    inner = 0
    lane_center_d = (inner + 0.5) * edge.lane_width

    import math
    node_by_id = {n.node_id: n.position for n in road_map.nodes}
    a = node_by_id[edge.from_node]
    b = node_by_id[edge.to_node]
    heading = math.atan2(b.y - a.y, b.x - a.x)
    rx, ry = math.sin(heading), -math.cos(heading)

    # Ego is already at the inner lane centre
    ego = Pose(
        x=a.x + rx * lane_center_d,
        y=a.y + ry * lane_center_d,
        heading=heading,
    )
    loc = _make_loc(road_map, ego, [])
    prev = BehaviorOutput(
        state=BehaviorState.CHANGE_LANE_LEFT,
        target_lane=inner,
        target_speed=edge.speed_limit,
    )
    out = plan_behavior(loc, route, road_map, prev=prev)
    assert out.state == BehaviorState.KEEP_LANE
    assert out.target_lane == inner


def test_returns_to_curbside_when_inner_lane_clear(road_map, route):
    """In inner lane with no obstacles → CHANGE_LANE_RIGHT toward curbside."""
    edge = road_map.edges[0]
    curbside = edge.num_lanes - 1
    ego = _ego_at_node(road_map)
    loc = _make_loc(road_map, ego, [])
    prev = BehaviorOutput(
        state=BehaviorState.KEEP_LANE,
        target_lane=0,  # currently in inner lane
        target_speed=edge.speed_limit,
    )
    out = plan_behavior(loc, route, road_map, prev=prev)
    assert out.state == BehaviorState.CHANGE_LANE_RIGHT
    assert out.target_lane == curbside


def test_no_lane_change_when_inner_lane_also_blocked(road_map, route):
    """Both lanes blocked → KEEP_LANE with reduced speed."""
    edge = road_map.edges[0]
    curbside = edge.num_lanes - 1
    ego = _ego_at_node(road_map)

    import math
    node_by_id = {n.node_id: n.position for n in road_map.nodes}
    a = node_by_id[edge.from_node]
    b = node_by_id[edge.to_node]
    heading = math.atan2(b.y - a.y, b.x - a.x)
    rx, ry = math.sin(heading), -math.cos(heading)

    obstacles = []
    for lane_idx in range(edge.num_lanes):
        d = (lane_idx + 0.5) * edge.lane_width
        obs_x = ego.x + math.cos(heading) * 8.0 + rx * d
        obs_y = ego.y + math.sin(heading) * 8.0 + ry * d
        obstacles.append(
            DetectedObject(
                object_id=lane_idx,
                object_type="vehicle",
                pose=Pose(x=obs_x, y=obs_y, heading=0.0),
                velocity=Vector2(x=0.0, y=0.0),
                confidence=1.0,
            )
        )

    loc = _make_loc(road_map, ego, obstacles)
    prev = BehaviorOutput(
        state=BehaviorState.KEEP_LANE,
        target_lane=curbside,
        target_speed=edge.speed_limit,
    )
    out = plan_behavior(loc, route, road_map, prev=prev)
    assert out.state == BehaviorState.KEEP_LANE
    assert out.target_speed < edge.speed_limit


def test_no_edge_returns_prev(road_map, route):
    """When nearest_road_edge is None, the previous output is returned unchanged."""
    ego = _ego_at_node(road_map)
    loc = LocalizationOutput(
        timestamp=0.0,
        estimated_pose=ego,
        detected_objects=[],
        nearest_road_edge=None,
    )
    prev = BehaviorOutput(
        state=BehaviorState.KEEP_LANE,
        target_lane=1,
        target_speed=8.33,
    )
    out = plan_behavior(loc, route, road_map, prev=prev)
    assert out == prev


# ---------------------------------------------------------------------------
# Traffic light and stop sign tests
# ---------------------------------------------------------------------------

from self_driving.map_gen import (
    add_stop_signs,
    add_traffic_lights,
    edge_polyline,
    initial_traffic_light_states,
)
from self_driving.models import (
    RoadEdge,
    RoadSign,
    SignType,
    TrafficLight,
    TrafficLightPhase,
    TrafficLightState,
    Vector2,
)


def _make_red_state(light_id: int, tl: TrafficLight) -> TrafficLightState:
    return TrafficLightState(
        light_id=light_id,
        phase=TrafficLightPhase.RED,
        time_in_phase=0.0,
        time_remaining=tl.red_duration,
    )


def _make_green_state(light_id: int, tl: TrafficLight) -> TrafficLightState:
    return TrafficLightState(
        light_id=light_id,
        phase=TrafficLightPhase.GREEN,
        time_in_phase=0.0,
        time_remaining=tl.green_duration,
    )


def test_stops_for_red_light(road_map, route):
    """Ego near end of edge with a RED light → target_speed == 0."""
    edge = road_map.edges[0]
    node_by_id = {n.node_id: n.position for n in road_map.nodes}
    # Place ego close to the to_node (within stop distance)
    b = node_by_id[edge.to_node]
    ego = Pose(x=b.x - 5.0, y=b.y, heading=0.0)

    tl = TrafficLight(
        light_id=0, node_id=edge.to_node,
        controlled_edges=[(edge.from_node, edge.to_node)],
        green_duration=20.0, yellow_duration=3.0, red_duration=20.0,
    )
    rm_with_lights = road_map.model_copy(update={"traffic_lights": [tl]})
    red_state = _make_red_state(0, tl)

    loc = _make_loc(rm_with_lights, ego, [])
    loc = LocalizationOutput(
        timestamp=0.0, estimated_pose=ego, detected_objects=[],
        nearest_road_edge=edge,
    )
    result = plan_behavior(loc, route, rm_with_lights, None, [red_state])
    assert result.target_speed == 0.0
    assert result.state == BehaviorState.STOPPING_FOR_RED


def test_proceeds_on_green_light(road_map, route):
    """Same setup but GREEN light → normal speed."""
    edge = road_map.edges[0]
    node_by_id = {n.node_id: n.position for n in road_map.nodes}
    b = node_by_id[edge.to_node]
    ego = Pose(x=b.x - 5.0, y=b.y, heading=0.0)

    tl = TrafficLight(
        light_id=0, node_id=edge.to_node,
        controlled_edges=[(edge.from_node, edge.to_node)],
        green_duration=20.0, yellow_duration=3.0, red_duration=20.0,
    )
    rm_with_lights = road_map.model_copy(update={"traffic_lights": [tl]})
    green_state = _make_green_state(0, tl)

    loc = LocalizationOutput(
        timestamp=0.0, estimated_pose=ego, detected_objects=[],
        nearest_road_edge=edge,
    )
    result = plan_behavior(loc, route, rm_with_lights, None, [green_state])
    assert result.target_speed > 0.0
    assert result.state != BehaviorState.STOPPING_FOR_RED


def test_stops_for_stop_sign(road_map, route):
    """Ego approaching a STOP sign within stopping distance → speed == 0."""
    edge = road_map.edges[0]
    node_by_id = {n.node_id: n.position for n in road_map.nodes}
    a = node_by_id[edge.from_node]
    b = node_by_id[edge.to_node]
    # Place ego near start of edge; sign is 10 m ahead (within _SIGN_STOP_DISTANCE)
    ego = Pose(x=a.x + 1.0, y=a.y, heading=0.0)
    sign = RoadSign(
        sign_id=0, sign_type=SignType.STOP,
        edge=(edge.from_node, edge.to_node),
        distance_along_edge=10.0,
    )
    rm_with_sign = road_map.model_copy(update={"road_signs": [sign]})
    loc = LocalizationOutput(
        timestamp=0.0, estimated_pose=ego, detected_objects=[],
        nearest_road_edge=edge,
    )
    result = plan_behavior(loc, route, rm_with_sign, None)
    assert result.target_speed == 0.0
    assert result.state == BehaviorState.STOPPING_FOR_SIGN


def test_no_stop_past_stop_sign(road_map, route):
    """Ego already past the STOP sign → no stop-for-sign behaviour."""
    edge = road_map.edges[0]
    node_by_id = {n.node_id: n.position for n in road_map.nodes}
    b = node_by_id[edge.to_node]
    # Ego near end of edge; sign is near start (already passed)
    ego = Pose(x=b.x - 2.0, y=b.y, heading=0.0)
    sign = RoadSign(
        sign_id=0, sign_type=SignType.STOP,
        edge=(edge.from_node, edge.to_node),
        distance_along_edge=2.0,  # < ego_s → already passed
    )
    rm_with_sign = road_map.model_copy(update={"road_signs": [sign]})
    loc = LocalizationOutput(
        timestamp=0.0, estimated_pose=ego, detected_objects=[],
        nearest_road_edge=edge,
    )
    result = plan_behavior(loc, route, rm_with_sign, None)
    assert result.state != BehaviorState.STOPPING_FOR_SIGN
