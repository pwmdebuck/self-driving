"""Layer 3a — Behavioural planner.

Sits between the global router and the local path planner.  Uses a finite
state machine to select which lane to drive in and when to change lanes,
based on detected objects and the current road geometry.
"""

import math

from self_driving.models import (
    BehaviorOutput,
    BehaviorState,
    DetectedObject,
    LocalizationOutput,
    Pose,
    RoadEdge,
    RoadMap,
    Route,
    Vector2,
)

# Longitudinal distances (metres)
_SLOW_DOWN_DIST = 12.0
_EMERGENCY_DIST = 4.0

# Lane-change completion threshold (metres lateral error)
_LANE_SETTLE_D = 0.5

# Lateral buffer added to each side of a lane when checking occupancy
_OBSTACLE_BUFFER_D = 0.5

# Fixed arc-length over which a lane change is planned (m)
_LANE_CHANGE_DIST_M = 10.0


def plan_behavior(
    loc: LocalizationOutput,
    route: Route,
    road_map: RoadMap,
    prev: BehaviorOutput | None,
) -> BehaviorOutput:
    """Behavioural FSM: decide which lane to drive in and at what speed.

    Returns a BehaviorOutput describing the FSM state, the target lane
    index (0 = innermost, num_lanes-1 = curbside), and the target speed.
    """
    edge = loc.nearest_road_edge
    if edge is None:
        return prev if prev is not None else _default_output(road_map)

    node_by_id = {n.node_id: n.position for n in road_map.nodes}
    ego = loc.estimated_pose
    objects = loc.detected_objects
    num_lanes = edge.num_lanes
    lane_width = edge.lane_width
    curbside = num_lanes - 1
    speed_limit = edge.speed_limit

    if prev is None:
        return BehaviorOutput(
            state=BehaviorState.KEEP_LANE,
            target_lane=curbside,
            target_speed=speed_limit,
        )

    state = prev.state
    target_lane = prev.target_lane
    ego_d = _ego_d(ego, edge, node_by_id)

    # --- Completing a lane change -------------------------------------------
    if state in (BehaviorState.CHANGE_LANE_LEFT, BehaviorState.CHANGE_LANE_RIGHT):
        target_d = _lane_center_d(target_lane, lane_width)
        if abs(ego_d - target_d) < _LANE_SETTLE_D:
            return BehaviorOutput(
                state=BehaviorState.KEEP_LANE,
                target_lane=target_lane,
                target_speed=speed_limit,
            )
        # Carry the fixed world endpoint forward so path planner sees shrinking T.
        return BehaviorOutput(
            state=state,
            target_lane=target_lane,
            target_speed=speed_limit,
            lane_change_end_pos=prev.lane_change_end_pos,
        )

    # --- KEEP_LANE transitions -----------------------------------------------
    # Check if the current lane is blocked and an inner lane is available.
    if _obstacle_in_lane(objects, target_lane, edge, node_by_id, ego, _SLOW_DOWN_DIST):
        inner = target_lane - 1
        if inner >= 0 and _lane_is_clear(objects, inner, edge, node_by_id, ego, _SLOW_DOWN_DIST):
            return BehaviorOutput(
                state=BehaviorState.CHANGE_LANE_LEFT,
                target_lane=inner,
                target_speed=speed_limit,
                lane_change_end_pos=_road_point_ahead(ego, edge, node_by_id, _LANE_CHANGE_DIST_M),
            )
        # Blocked but no room to change — slow down
        slow_speed = _compute_target_speed(objects, ego, edge)
        return BehaviorOutput(
            state=BehaviorState.KEEP_LANE,
            target_lane=target_lane,
            target_speed=slow_speed,
        )

    # If we ended up in an inner lane (after avoidance), return to curbside
    # when it becomes clear.
    if target_lane < curbside and _lane_is_clear(
        objects, curbside, edge, node_by_id, ego, _SLOW_DOWN_DIST
    ):
        return BehaviorOutput(
            state=BehaviorState.CHANGE_LANE_RIGHT,
            target_lane=curbside,
            target_speed=speed_limit,
            lane_change_end_pos=_road_point_ahead(ego, edge, node_by_id, _LANE_CHANGE_DIST_M),
        )

    return BehaviorOutput(
        state=BehaviorState.KEEP_LANE,
        target_lane=target_lane,
        target_speed=speed_limit,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_output(road_map: RoadMap) -> BehaviorOutput:
    """Produce a sane default when no edge info is available."""
    # Use MapConfig defaults if edges exist, otherwise fall back to 1 lane.
    num_lanes = road_map.edges[0].num_lanes if road_map.edges else 1
    speed_limit = road_map.edges[0].speed_limit if road_map.edges else 8.33
    return BehaviorOutput(
        state=BehaviorState.KEEP_LANE,
        target_lane=num_lanes - 1,
        target_speed=speed_limit,
    )


def _lane_center_d(lane_idx: int, lane_width: float) -> float:
    """Signed lateral offset of lane centre from road centreline (positive = right)."""
    return (lane_idx + 0.5) * lane_width


def _ego_d(ego: Pose, edge: RoadEdge, node_by_id: dict[int, Vector2]) -> float:
    """Signed perpendicular distance of ego from the road centreline (positive = right)."""
    a = node_by_id.get(edge.from_node)
    b = node_by_id.get(edge.to_node)
    if a is None or b is None:
        return 0.0
    dx, dy = b.x - a.x, b.y - a.y
    seg_len = math.sqrt(dx * dx + dy * dy)
    if seg_len < 1e-9:
        return 0.0
    # Road tangent heading
    heading = math.atan2(dy, dx)
    # Right-perpendicular (positive = right of heading in Y-up frame)
    rx, ry = math.sin(heading), -math.cos(heading)
    return (ego.x - a.x) * rx + (ego.y - a.y) * ry


def _object_forward_dist(obj_pose: Pose, ego: Pose, edge: RoadEdge, node_by_id: dict[int, Vector2]) -> float:
    """Signed forward distance from ego to object along the road direction."""
    a = node_by_id.get(edge.from_node)
    b = node_by_id.get(edge.to_node)
    if a is None or b is None:
        return math.inf
    dx, dy = b.x - a.x, b.y - a.y
    seg_len = math.sqrt(dx * dx + dy * dy)
    if seg_len < 1e-9:
        return math.inf
    tx, ty = dx / seg_len, dy / seg_len  # road tangent unit vector
    return (obj_pose.x - ego.x) * tx + (obj_pose.y - ego.y) * ty


def _obstacle_in_lane(
    objects: list[DetectedObject],
    lane_idx: int,
    edge: RoadEdge,
    node_by_id: dict[int, Vector2],
    ego: Pose,
    lookahead_m: float,
) -> bool:
    """Return True if any object is inside the lane band within lookahead_m ahead."""
    lane_width = edge.lane_width
    center_d = _lane_center_d(lane_idx, lane_width)
    half = lane_width / 2.0 + _OBSTACLE_BUFFER_D
    for obj in objects:
        fwd = _object_forward_dist(obj.pose, ego, edge, node_by_id)
        if fwd <= 0 or fwd > lookahead_m:
            continue
        obj_d = _ego_d(obj.pose, edge, node_by_id)
        if abs(obj_d - center_d) < half:
            return True
    return False


def _lane_is_clear(
    objects: list[DetectedObject],
    lane_idx: int,
    edge: RoadEdge,
    node_by_id: dict[int, Vector2],
    ego: Pose,
    check_dist_m: float,
) -> bool:
    return not _obstacle_in_lane(objects, lane_idx, edge, node_by_id, ego, check_dist_m)


def _road_point_ahead(
    ego: Pose,
    edge: RoadEdge,
    node_by_id: dict[int, Vector2],
    dist: float,
) -> Vector2:
    """Return the point `dist` metres ahead of ego along the road centreline."""
    a = node_by_id.get(edge.from_node)
    b = node_by_id.get(edge.to_node)
    if a is None or b is None:
        return Vector2(x=ego.x + math.cos(ego.heading) * dist, y=ego.y + math.sin(ego.heading) * dist)
    dx, dy = b.x - a.x, b.y - a.y
    seg_len = math.sqrt(dx * dx + dy * dy)
    if seg_len < 1e-9:
        return Vector2(x=ego.x + math.cos(ego.heading) * dist, y=ego.y + math.sin(ego.heading) * dist)
    tx, ty = dx / seg_len, dy / seg_len
    proj = (ego.x - a.x) * tx + (ego.y - a.y) * ty
    return Vector2(x=a.x + tx * (proj + dist), y=a.y + ty * (proj + dist))


def _compute_target_speed(
    objects: list[DetectedObject],
    ego: Pose,
    edge: RoadEdge,
) -> float:
    """Scale target speed linearly based on nearest ahead obstacle distance."""
    min_dist = _SLOW_DOWN_DIST
    cos_h = math.cos(ego.heading)
    sin_h = math.sin(ego.heading)
    for obj in objects:
        dx = obj.pose.x - ego.x
        dy = obj.pose.y - ego.y
        fwd = dx * cos_h + dy * sin_h
        dist = math.sqrt(dx * dx + dy * dy)
        if fwd > 0 and dist < min_dist:
            min_dist = dist

    if min_dist <= _EMERGENCY_DIST:
        return 0.5
    t = (min_dist - _EMERGENCY_DIST) / (_SLOW_DOWN_DIST - _EMERGENCY_DIST)
    return max(0.5, min(edge.speed_limit, t * edge.speed_limit))
