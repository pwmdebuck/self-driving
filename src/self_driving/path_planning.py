"""Layer 3 — Local path planning / trajectory generation.

Extracts the next N waypoints from the global route ahead of the
current estimated position, adjusts for detected obstacles, and
generates a LocalTrajectory with per-point speed targets.
"""

import math

from shapely.geometry import Point, Polygon

from self_driving.models import (
    DetectedObject,
    LocalizationOutput,
    LocalTrajectory,
    Pose,
    RoadMap,
    Route,
    TrajectoryPoint,
    Vector2,
)

# Planning parameters
_TRAJECTORY_POINTS = 20
_POINT_SPACING_M = 1.5  # metres between trajectory points
_EMERGENCY_STOP_DIST = 4.0  # metres
_SLOW_DOWN_DIST = 12.0  # metres
_OBSTACLE_BUFFER_M = 2.5  # lateral avoidance buffer
_LATERAL_SHIFT_M = 2.0  # metres to shift around obstacle


def plan_path(
    route: Route,
    loc: LocalizationOutput,
    road_map: RoadMap,
) -> LocalTrajectory:
    """Generate a short-horizon local trajectory.

    Extracts waypoints along the global route ahead of the estimated
    ego position, avoids detected obstacles via lateral shift, and
    assigns per-waypoint target speeds.
    """
    node_by_id = {n.node_id: n.position for n in road_map.nodes}
    ego = loc.estimated_pose

    # Build a list of world positions from the route waypoints
    route_positions: list[Vector2] = [
        node_by_id[nid] for nid in route.waypoint_ids if nid in node_by_id
    ]

    if not route_positions:
        return _emergency_stop_trajectory(ego, loc.timestamp)

    # Find where we are on the route
    ahead = _waypoints_ahead(ego, route_positions)
    if not ahead:
        return _emergency_stop_trajectory(ego, loc.timestamp)

    # Interpolate fine-grained trajectory points
    raw_points = _interpolate_points(ego, ahead, _TRAJECTORY_POINTS, _POINT_SPACING_M)

    # Obstacle analysis
    obstacle_geoms = _build_obstacle_geoms(loc.detected_objects)
    emergency = _any_obstacle_within(
        raw_points[:3], obstacle_geoms, _EMERGENCY_STOP_DIST
    )

    if emergency:
        return _emergency_stop_trajectory(ego, loc.timestamp)

    # Lateral shift around nearby obstacles
    adjusted = _shift_around_obstacles(raw_points, obstacle_geoms, _SLOW_DOWN_DIST)

    # Assign speed targets
    traj_points = _assign_speeds(adjusted, obstacle_geoms, route, ego)

    return LocalTrajectory(
        timestamp=loc.timestamp,
        points=traj_points,
        is_emergency_stop=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _waypoints_ahead(ego: Pose, positions: list[Vector2]) -> list[Vector2]:
    """Return waypoints that are ahead of the ego (ignores behind ones)."""
    min_idx = 0
    min_dist = math.inf
    for i, pos in enumerate(positions):
        dx = pos.x - ego.x
        dy = pos.y - ego.y
        d = dx * dx + dy * dy
        if d < min_dist:
            min_dist = d
            min_idx = i
    return positions[min_idx:]


def _interpolate_points(
    ego: Pose,
    waypoints: list[Vector2],
    n: int,
    spacing: float,
) -> list[Vector2]:
    """Linearly interpolate n equally-spaced points along the waypoints."""
    # Build cumulative distance path
    path: list[tuple[float, float]] = [(ego.x, ego.y)]
    for wp in waypoints:
        path.append((wp.x, wp.y))

    # Compute cumulative distances
    cum = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        cum.append(cum[-1] + math.sqrt(dx * dx + dy * dy))

    total = cum[-1]
    if total < 0.01:
        return [Vector2(x=ego.x, y=ego.y)] * n

    result: list[Vector2] = []
    for k in range(n):
        target_dist = k * spacing
        if target_dist >= total:
            result.append(Vector2(x=path[-1][0], y=path[-1][1]))
            continue
        # Find segment
        for i in range(1, len(cum)):
            if cum[i] >= target_dist:
                seg_len = cum[i] - cum[i - 1]
                t = (target_dist - cum[i - 1]) / max(seg_len, 1e-9)
                ix = path[i - 1][0] + t * (path[i][0] - path[i - 1][0])
                iy = path[i - 1][1] + t * (path[i][1] - path[i - 1][1])
                result.append(Vector2(x=ix, y=iy))
                break

    return result


def _build_obstacle_geoms(
    objects: list[DetectedObject],
) -> list[tuple[DetectedObject, Polygon]]:
    """Buffer each detected object into a Shapely polygon."""
    result = []
    for obj in objects:
        buf = _OBSTACLE_BUFFER_M if obj.object_type != "pedestrian" else 1.5
        geom = Point(obj.pose.x, obj.pose.y).buffer(buf)
        result.append((obj, geom))
    return result


def _any_obstacle_within(
    points: list[Vector2],
    obstacles: list[tuple[DetectedObject, Polygon]],
    radius: float,
) -> bool:
    for pt in points:
        p = Point(pt.x, pt.y)
        for _, geom in obstacles:
            if p.distance(geom) < radius:
                return True
    return False


def _shift_around_obstacles(
    points: list[Vector2],
    obstacles: list[tuple[DetectedObject, Polygon]],
    detect_dist: float,
) -> list[Vector2]:
    """Shift points laterally when they are too close to an obstacle."""
    shifted: list[Vector2] = []
    for i, pt in enumerate(points):
        p = Point(pt.x, pt.y)
        too_close = any(p.distance(geom) < detect_dist for _, geom in obstacles)
        if not too_close or i == 0:
            shifted.append(pt)
            continue

        # Compute perpendicular direction based on local path heading
        prev = shifted[-1] if shifted else pt
        dx = pt.x - prev.x
        dy = pt.y - prev.y
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            shifted.append(pt)
            continue
        # Perpendicular (left turn shift)
        perp_x = -dy / length
        perp_y = dx / length
        shifted.append(
            Vector2(
                x=pt.x + perp_x * _LATERAL_SHIFT_M,
                y=pt.y + perp_y * _LATERAL_SHIFT_M,
            )
        )
    return shifted


def _assign_speeds(
    points: list[Vector2],
    obstacles: list[tuple[DetectedObject, Polygon]],
    route: Route,
    ego: Pose,
) -> list[TrajectoryPoint]:
    """Assign target speed and curvature to each trajectory point."""
    n = len(points)
    result: list[TrajectoryPoint] = []
    base_speed = min(route.total_distance / max(route.estimated_time, 1.0), 8.33)

    for i, pt in enumerate(points):
        # Reduce speed near obstacles
        p = Point(pt.x, pt.y)
        min_obs_dist = min(
            (p.distance(geom) for _, geom in obstacles), default=_SLOW_DOWN_DIST
        )
        speed_factor = min(1.0, min_obs_dist / _SLOW_DOWN_DIST)
        target_speed = max(0.5, base_speed * speed_factor)

        # Curvature from three consecutive points
        curvature = 0.0
        if 0 < i < n - 1:
            p0 = points[i - 1]
            p1 = pt
            p2 = points[i + 1]
            curvature = _three_point_curvature(p0, p1, p2)

        # Heading from previous point
        if i > 0:
            prev = points[i - 1]
            heading = math.atan2(pt.y - prev.y, pt.x - prev.x)
        else:
            heading = ego.heading

        result.append(
            TrajectoryPoint(
                pose=Pose(x=pt.x, y=pt.y, heading=heading),
                speed=float(target_speed),
                curvature=float(curvature),
            )
        )
    return result


def _three_point_curvature(p0: Vector2, p1: Vector2, p2: Vector2) -> float:
    """Menger curvature estimate from three consecutive points."""
    ax, ay = p1.x - p0.x, p1.y - p0.y
    bx, by = p2.x - p1.x, p2.y - p1.y
    cross = ax * by - ay * bx
    la = math.sqrt(ax * ax + ay * ay)
    lb = math.sqrt(bx * bx + by * by)
    lab = math.sqrt((p2.x - p0.x) ** 2 + (p2.y - p0.y) ** 2)
    denom = la * lb * lab
    if denom < 1e-9:
        return 0.0
    return 2.0 * cross / denom


def _emergency_stop_trajectory(pose: Pose, timestamp: float) -> LocalTrajectory:
    """Return a trajectory that commands the vehicle to stop in place."""
    points = [
        TrajectoryPoint(pose=pose, speed=0.0, curvature=0.0)
        for _ in range(_TRAJECTORY_POINTS)
    ]
    return LocalTrajectory(
        timestamp=timestamp,
        points=points,
        is_emergency_stop=True,
    )
