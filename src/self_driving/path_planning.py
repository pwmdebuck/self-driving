"""Layer 3b — Local path planning / trajectory generation.

Builds a short-horizon local trajectory in the Frenet frame:

  1. Construct a dense road-centreline reference from the global route.
  2. Convert the ego state to Frenet (s, d) coordinates.
  3. Fit a quintic polynomial d(s) that smoothly transitions from the
     current lateral position/rate to the behavioural target lane centre.
  4. Sample the polynomial, convert back to Cartesian, and assign
     per-point speed targets.
"""

import math

import numpy as np
from shapely.geometry import Point, Polygon

from self_driving.models import (
    BehaviorOutput,
    BehaviorState,
    DetectedObject,
    LocalizationOutput,
    LocalTrajectory,
    Pose,
    RoadEdge,
    RoadMap,
    Route,
    TrajectoryPoint,
    Vector2,
    VehicleParams,
    VehicleState,
)

# Planning parameters
_TRAJECTORY_POINTS = 20
_POINT_SPACING_M = 1.5  # metres — target_speed × MPC_dt (8.33 m/s × 0.1 s ≈ 0.83, rounded up)
_EMERGENCY_STOP_DIST = 4.0  # metres
_SLOW_DOWN_DIST = 12.0  # metres
_OBSTACLE_BUFFER_M = 2.5  # lateral buffer for obstacle Shapely polygons
_WAYPOINT_PASS_DIST = 3.0  # metres — clearance to consider a waypoint still ahead
_MAX_LAT_ACCEL = 2.5  # m/s² — lateral acceleration limit for curvature-based speed cap
_MIN_QUINTIC_HORIZON = 5.0  # metres — clamp T to avoid ill-conditioning
_LANE_CHANGE_HORIZON_M = 10.0  # metres — quintic T for lane changes (must fit in MPC horizon)


def plan_path(
    route: Route,
    loc: LocalizationOutput,
    road_map: RoadMap,
    behavior: BehaviorOutput | None = None,
    ego_state: VehicleState | None = None,
    vehicle_params: VehicleParams | None = None,
) -> LocalTrajectory:
    """Generate a short-horizon local trajectory in the Frenet frame.

    Parameters
    ----------
    route:          Global route from the router.
    loc:            Localisation output (ego pose + detected objects).
    road_map:       Road network (nodes, edges).
    behavior:       Behavioural planner output; defaults to curbside KEEP_LANE.
    ego_state:      Full vehicle state used to estimate initial d-rate; if
                    None, d-rate is approximated as 0.
    vehicle_params: Bicycle model parameters for d-rate estimation; defaults
                    to VehicleParams().
    """
    vp = vehicle_params or VehicleParams()
    node_by_id = {n.node_id: n.position for n in road_map.nodes}
    ego = loc.estimated_pose

    # Resolve behaviour defaults
    edge = _route_edge_for_ego(route, ego, road_map) or loc.nearest_road_edge
    lane_width = edge.lane_width if edge else 3.5
    num_lanes = edge.num_lanes if edge else 1
    speed_limit = edge.speed_limit if edge else 8.33
    if behavior is None:
        behavior = BehaviorOutput(
            state=BehaviorState.KEEP_LANE,
            target_lane=num_lanes - 1,
            target_speed=speed_limit,
        )

    # Build lane-centred route waypoints
    valid_node_ids = [nid for nid in route.waypoint_ids if nid in node_by_id]
    lane_waypoints = _lane_center_waypoints(
        valid_node_ids, node_by_id, behavior.target_lane, lane_width
    )
    if not lane_waypoints:
        return emergency_stop_trajectory(ego, loc.timestamp)

    ahead = _waypoints_ahead(ego, lane_waypoints)
    if not ahead:
        return emergency_stop_trajectory(ego, loc.timestamp)

    # -----------------------------------------------------------------------
    # Step 1: Build dense road centreline
    # -----------------------------------------------------------------------
    centerline_start = _project_onto_lane(
        ego, edge, node_by_id, behavior.target_lane, lane_width
    )
    centerline = _interpolate_points(
        [centerline_start] + ahead, _TRAJECTORY_POINTS, _POINT_SPACING_M
    )
    arc_lengths = _build_arc_lengths(centerline)

    # -----------------------------------------------------------------------
    # Step 2: Ego Frenet state
    # -----------------------------------------------------------------------
    s0, d0 = cartesian_to_frenet(ego, centerline, arc_lengths)
    psi_road = _road_heading_at(s0, centerline, arc_lengths)
    if ego_state is not None:
        d0_prime, d0_pprime = _estimate_frenet_initial_state(ego_state, psi_road, vp)
    else:
        d0_prime, d0_pprime = 0.0, 0.0

    # -----------------------------------------------------------------------
    # Step 3: Quintic polynomial for lateral profile d(s)
    # -----------------------------------------------------------------------
    d_target = 0.0  # centreline already follows target lane centre
    if behavior.state == BehaviorState.KEEP_LANE:
        # Plan the desired path (centreline); let the controller correct tracking
        # error. Feeding d0 back into the planner causes double-correction wiggle.
        d0, d0_prime, d0_pprime = 0.0, 0.0, 0.0
        T = max(_MIN_QUINTIC_HORIZON, arc_lengths[-1])
    elif behavior.lane_change_end_pos is not None:
        end_pose = Pose(x=behavior.lane_change_end_pos.x, y=behavior.lane_change_end_pos.y, heading=0.0)
        s_end, _ = cartesian_to_frenet(end_pose, centerline, arc_lengths)
        T = max(_MIN_QUINTIC_HORIZON, s_end - s0)
    else:
        T = _LANE_CHANGE_HORIZON_M
    try:
        coeffs = _solve_quintic_lateral(d0, d0_prime, d0_pprime, d_target, T)
    except np.linalg.LinAlgError:
        # Degenerate case — flat profile at target
        coeffs = (d_target, 0.0, 0.0, 0.0, 0.0, 0.0)

    # -----------------------------------------------------------------------
    # Step 4: Sample trajectory — Frenet → Cartesian
    # -----------------------------------------------------------------------
    world_pts: list[Vector2] = []
    headings: list[float] = []
    for i in range(len(centerline)):
        s_i = arc_lengths[i]
        # Clamp s to T so points beyond the quintic horizon hold at d_target
        d_i = _eval_quintic(coeffs, min(s_i, T))
        x, y, h = frenet_to_cartesian(s_i, d_i, centerline, arc_lengths)
        world_pts.append(Vector2(x=x, y=y))
        headings.append(h)

    # -----------------------------------------------------------------------
    # Step 5: Emergency stop check
    # -----------------------------------------------------------------------
    obstacle_geoms = _build_obstacle_geoms(loc.detected_objects)
    if _any_obstacle_within(world_pts[:3], obstacle_geoms, _EMERGENCY_STOP_DIST):
        return emergency_stop_trajectory(ego, loc.timestamp)

    # -----------------------------------------------------------------------
    # Step 6: Assign speed targets
    # -----------------------------------------------------------------------
    traj_points = _assign_speeds_frenet(
        world_pts, headings, obstacle_geoms, behavior.target_speed
    )

    return LocalTrajectory(
        timestamp=loc.timestamp,
        points=traj_points,
        is_emergency_stop=False,
        centerline=centerline,
    )


# ---------------------------------------------------------------------------
# Frenet utilities
# ---------------------------------------------------------------------------


def _build_arc_lengths(pts: list[Vector2]) -> list[float]:
    """Cumulative Euclidean arc-length table for a dense polyline."""
    arcs: list[float] = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i].x - pts[i - 1].x
        dy = pts[i].y - pts[i - 1].y
        arcs.append(arcs[-1] + math.sqrt(dx * dx + dy * dy))
    return arcs


def cartesian_to_frenet(
    pose: Pose,
    centerline: list[Vector2],
    arc_lengths: list[float],
) -> tuple[float, float]:
    """Project *pose* onto the centreline; return (s, d).

    s: arc-length along the centreline from the first point.
    d: signed lateral offset — positive = right of centreline
       (matching the existing right-lane convention).

    Uses nearest-point-on-segment projection for sub-spacing accuracy.
    """
    best_s = arc_lengths[0]
    best_proj_x = centerline[0].x
    best_proj_y = centerline[0].y
    best_dist_sq = math.inf

    for i in range(len(centerline) - 1):
        ax, ay = centerline[i].x, centerline[i].y
        bx, by = centerline[i + 1].x, centerline[i + 1].y
        dx, dy = bx - ax, by - ay
        seg_sq = dx * dx + dy * dy
        if seg_sq < 1e-12:
            continue
        t = ((pose.x - ax) * dx + (pose.y - ay) * dy) / seg_sq
        t = max(0.0, min(1.0, t))
        px, py = ax + t * dx, ay + t * dy
        dist_sq = (pose.x - px) ** 2 + (pose.y - py) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_proj_x = px
            best_proj_y = py
            best_s = arc_lengths[i] + t * (arc_lengths[i + 1] - arc_lengths[i])

    # Tangent heading at the projection point
    heading = _road_heading_at(best_s, centerline, arc_lengths)
    rx, ry = math.sin(heading), -math.cos(heading)
    d = (pose.x - best_proj_x) * rx + (pose.y - best_proj_y) * ry
    return best_s, d


def frenet_to_cartesian(
    s: float,
    d: float,
    centerline: list[Vector2],
    arc_lengths: list[float],
) -> tuple[float, float, float]:
    """Convert (s, d) back to (x, y, heading).

    heading is the road-tangent heading at s; the lateral offset d is
    applied perpendicular to it. d' is neglected (small for smooth paths).
    """
    n = len(arc_lengths)
    if s <= arc_lengths[0]:
        i = 0
        frac = 0.0
    elif s >= arc_lengths[-1]:
        i = n - 2
        frac = 1.0
    else:
        i = 0
        for k in range(1, n):
            if arc_lengths[k] >= s:
                i = k - 1
                break
        seg = arc_lengths[i + 1] - arc_lengths[i]
        frac = (s - arc_lengths[i]) / max(seg, 1e-9)

    base_x = centerline[i].x + frac * (centerline[i + 1].x - centerline[i].x)
    base_y = centerline[i].y + frac * (centerline[i + 1].y - centerline[i].y)
    heading = math.atan2(
        centerline[i + 1].y - centerline[i].y,
        centerline[i + 1].x - centerline[i].x,
    )
    rx, ry = math.sin(heading), -math.cos(heading)
    return base_x + rx * d, base_y + ry * d, heading


def _road_heading_at(
    s: float,
    centerline: list[Vector2],
    arc_lengths: list[float],
) -> float:
    """Road-tangent heading at arc-length s."""
    if len(centerline) < 2:
        return 0.0
    for i in range(len(arc_lengths) - 1):
        if arc_lengths[i + 1] >= s:
            return math.atan2(
                centerline[i + 1].y - centerline[i].y,
                centerline[i + 1].x - centerline[i].x,
            )
    return math.atan2(
        centerline[-1].y - centerline[-2].y,
        centerline[-1].x - centerline[-2].x,
    )


def _lane_center_d(lane_idx: int, lane_width: float) -> float:
    """Lateral offset of lane centre from road centreline (positive = right)."""
    return (lane_idx + 0.5) * lane_width


def _estimate_frenet_initial_state(
    ego_state: VehicleState,
    psi_road: float,
    vp: VehicleParams,
) -> tuple[float, float]:
    """Estimate initial Frenet lateral rate d' and d'' from the bicycle model.

    d'  = sin(delta_psi)                 (lateral slope w.r.t. arc-length s)
    d'' = cos(delta_psi) * yaw_rate / v  (lateral curvature w.r.t. s)
    """
    lr = vp.wheelbase * vp.lr_ratio
    beta = math.atan(lr / vp.wheelbase * math.tan(-ego_state.steering_angle))
    delta_psi = ego_state.pose.heading + beta - psi_road
    d0_prime = math.sin(delta_psi)
    v = max(ego_state.speed, 1e-3)
    d0_pprime = math.cos(delta_psi) * ego_state.yaw_rate / v
    return d0_prime, d0_pprime


# ---------------------------------------------------------------------------
# Quintic polynomial solver
# ---------------------------------------------------------------------------


def _solve_quintic_lateral(
    d0: float,
    d0_prime: float,
    d0_pprime: float,
    d_target: float,
    T: float,
) -> tuple[float, float, float, float, float, float]:
    """Solve for quintic coefficients satisfying the 6 boundary conditions.

    Polynomial:  d(s) = a0 + a1*s + a2*s² + a3*s³ + a4*s⁴ + a5*s⁵

    Initial (s=0):  d=d0,       d'=d0_prime,  d''=d0_pprime
    Final   (s=T):  d=d_target, d'=0,          d''=0
    """
    a0 = d0
    a1 = d0_prime
    a2 = d0_pprime / 2.0

    A = np.array(
        [
            [T**3, T**4, T**5],
            [3 * T**2, 4 * T**3, 5 * T**4],
            [6 * T, 12 * T**2, 20 * T**3],
        ]
    )
    b = np.array(
        [
            d_target - a0 - a1 * T - a2 * T**2,
            -a1 - 2 * a2 * T,
            -2 * a2,
        ]
    )
    a3, a4, a5 = np.linalg.solve(A, b)
    return a0, a1, a2, float(a3), float(a4), float(a5)


def _eval_quintic(coeffs: tuple[float, ...], s: float) -> float:
    """Evaluate the quintic polynomial at arc-length s."""
    a0, a1, a2, a3, a4, a5 = coeffs
    return a0 + a1 * s + a2 * s**2 + a3 * s**3 + a4 * s**4 + a5 * s**5


# ---------------------------------------------------------------------------
# Speed assignment
# ---------------------------------------------------------------------------


def _assign_speeds_frenet(
    points: list[Vector2],
    headings: list[float],
    obstacles: list[tuple[DetectedObject, Polygon]],
    base_speed: float,
) -> list[TrajectoryPoint]:
    """Assign target speed and curvature to each trajectory point.

    Uses curvature-limited speeds and a backward kinematic pass to ensure
    the vehicle decelerates before sharp turns.
    """
    n = len(points)
    curvatures: list[float] = [0.0] * n
    for i in range(1, n - 1):
        curvatures[i] = _three_point_curvature(points[i - 1], points[i], points[i + 1])

    # Forward pass: obstacle proximity + curvature limits
    speeds: list[float] = []
    for i, pt in enumerate(points):
        p = Point(pt.x, pt.y)
        min_obs_dist = min(
            (p.distance(geom) for _, geom in obstacles), default=_SLOW_DOWN_DIST
        )
        obs_factor = min(1.0, min_obs_dist / _SLOW_DOWN_DIST)
        spd = base_speed * obs_factor

        kappa = abs(curvatures[i])
        if kappa > 1e-3:
            spd = min(spd, math.sqrt(_MAX_LAT_ACCEL / kappa))

        speeds.append(max(0.5, spd))

    # Backward pass: kinematic feasibility
    max_decel = 6.0  # m/s²
    for i in range(n - 2, -1, -1):
        v_next = speeds[i + 1]
        v_max = math.sqrt(v_next**2 + 2.0 * max_decel * _POINT_SPACING_M)
        speeds[i] = min(speeds[i], v_max)

    result: list[TrajectoryPoint] = []
    for i, pt in enumerate(points):
        result.append(
            TrajectoryPoint(
                pose=Pose(x=pt.x, y=pt.y, heading=headings[i]),
                speed=float(speeds[i]),
                curvature=float(curvatures[i]),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _route_edge_for_ego(
    route: Route,
    ego: Pose,
    road_map: RoadMap,
) -> RoadEdge | None:
    """Return the route edge ego is currently traversing.

    Scores each consecutive route-edge by perpendicular distance plus an
    out-of-range penalty when ego is before/after the segment. Edges that
    point anti-parallel to ego's heading are skipped (route is directed).
    """
    node_by_id = {n.node_id: n.position for n in road_map.nodes}
    edge_by_pair = {(e.from_node, e.to_node): e for e in road_map.edges}
    cos_h = math.cos(ego.heading)
    sin_h = math.sin(ego.heading)

    best_edge: RoadEdge | None = None
    best_score = math.inf

    for i in range(len(route.waypoint_ids) - 1):
        from_id = route.waypoint_ids[i]
        to_id = route.waypoint_ids[i + 1]
        edge = edge_by_pair.get((from_id, to_id))
        if edge is None:
            continue
        a = node_by_id.get(from_id)
        b = node_by_id.get(to_id)
        if a is None or b is None:
            continue
        dx, dy = b.x - a.x, b.y - a.y
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-9:
            continue
        # Skip edges anti-parallel to ego heading (route is directed forward)
        if dx * cos_h + dy * sin_h < 0:
            continue
        t = ((ego.x - a.x) * dx + (ego.y - a.y) * dy) / seg_len_sq
        t_c = max(0.0, min(1.0, t))
        proj_x = a.x + t_c * dx
        proj_y = a.y + t_c * dy
        perp_sq = (ego.x - proj_x) ** 2 + (ego.y - proj_y) ** 2
        overshoot = max(0.0, t - 1.0, -t)
        penalty = (overshoot * math.sqrt(seg_len_sq)) ** 2
        score = perp_sq + penalty
        if score < best_score:
            best_score = score
            best_edge = edge

    return best_edge


def _project_onto_edge(
    ego: Pose,
    edge: RoadEdge | None,
    node_by_id: dict[int, Vector2],
) -> Vector2:
    """Return the closest point on the nearest road edge's centreline to ego."""
    if edge is not None:
        a = node_by_id.get(edge.from_node)
        b = node_by_id.get(edge.to_node)
        if a is not None and b is not None:
            dx, dy = b.x - a.x, b.y - a.y
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq > 1e-9:
                t = ((ego.x - a.x) * dx + (ego.y - a.y) * dy) / seg_len_sq
                t = max(0.0, min(1.0, t))
                return Vector2(x=a.x + t * dx, y=a.y + t * dy)
    return Vector2(x=ego.x, y=ego.y)


def _project_onto_lane(
    ego: Pose,
    edge: RoadEdge | None,
    node_by_id: dict[int, Vector2],
    target_lane: int,
    lane_width: float,
) -> Vector2:
    """Project ego onto the road edge centreline, then offset to the lane centre."""
    road_proj = _project_onto_edge(ego, edge, node_by_id)
    if edge is None:
        return road_proj
    a = node_by_id.get(edge.from_node)
    b = node_by_id.get(edge.to_node)
    if a is None or b is None:
        return road_proj
    h = math.atan2(b.y - a.y, b.x - a.x)
    d = _lane_center_d(target_lane, lane_width)
    rx, ry = math.sin(h), -math.cos(h)
    return Vector2(x=road_proj.x + rx * d, y=road_proj.y + ry * d)


def _lane_center_waypoints(
    node_ids: list[int],
    node_by_id: dict[int, Vector2],
    target_lane: int,
    lane_width: float,
) -> list[Vector2]:
    """Interleaved edge-midpoints and lane-offset node positions for the route.

    Returns 2 points per edge: a lane-offset midpoint and a lane-offset node
    endpoint. Denser control points reduce per-step spline discontinuities when
    a waypoint is consumed. At intersection nodes the heading bisector places
    the waypoint at the correct lane-corner arc position.
    """
    if len(node_ids) < 2:
        return [node_by_id[nid] for nid in node_ids]

    d = _lane_center_d(target_lane, lane_width)
    n = len(node_ids)
    result: list[Vector2] = []

    for i in range(n - 1):
        a = node_by_id[node_ids[i]]
        b = node_by_id[node_ids[i + 1]]
        h_edge = math.atan2(b.y - a.y, b.x - a.x)
        rx, ry = math.sin(h_edge), -math.cos(h_edge)

        # Midpoint of this edge at lane centre
        result.append(Vector2(x=(a.x + b.x) / 2 + rx * d, y=(a.y + b.y) / 2 + ry * d))

        # Lane-offset node b: heading bisector of incoming + outgoing edges
        if i + 2 < n:
            c = node_by_id[node_ids[i + 2]]
            h_out = math.atan2(c.y - b.y, c.x - b.x)
            h_b = math.atan2(
                math.sin(h_edge) + math.sin(h_out),
                math.cos(h_edge) + math.cos(h_out),
            )
        else:
            h_b = h_edge  # last node: use incoming heading
        brx, bry = math.sin(h_b), -math.cos(h_b)
        result.append(Vector2(x=b.x + brx * d, y=b.y + bry * d))

    return result


def _waypoints_ahead(ego: Pose, positions: list[Vector2]) -> list[Vector2]:
    """Return waypoints from the nearest *forward* waypoint onward.

    Priority order:
      1. Nearest waypoint that is forward AND > _WAYPOINT_PASS_DIST away (normal)
      2. First waypoint that is forward at any distance (inside intersection)
      3. Waypoint after the closest one (overshoot — prevents backward trajectory)
    """
    cos_h = math.cos(ego.heading)
    sin_h = math.sin(ego.heading)

    best_forward_idx: int | None = None
    best_forward_dist = math.inf
    first_forward_idx: int | None = None
    best_any_idx = 0
    best_any_dist = math.inf

    for i, pos in enumerate(positions):
        dx = pos.x - ego.x
        dy = pos.y - ego.y
        dist_sq = dx * dx + dy * dy
        forward = dx * cos_h + dy * sin_h

        if dist_sq < best_any_dist:
            best_any_dist = dist_sq
            best_any_idx = i

        dist = math.sqrt(dist_sq)
        if forward > 0:
            if first_forward_idx is None:
                first_forward_idx = i
            if dist > _WAYPOINT_PASS_DIST and dist_sq < best_forward_dist:
                best_forward_dist = dist_sq
                best_forward_idx = i

    if best_forward_idx is not None:
        idx = best_forward_idx
    elif first_forward_idx is not None:
        idx = first_forward_idx
    else:
        idx = min(best_any_idx + 1, len(positions) - 1)
    return positions[idx:]


def _interpolate_points(
    waypoints: list[Vector2],
    n: int,
    spacing: float,
) -> list[Vector2]:
    """Sample n equally-spaced points along a Catmull-Rom spline.

    The spline passes through the road centreline waypoints, producing smooth
    curves through corners. Ego position is intentionally excluded so the
    resulting reference line is road-relative, not ego-relative.
    """
    ctrl: list[tuple[float, float]] = [(w.x, w.y) for w in waypoints]

    if len(ctrl) < 2:
        p = ctrl[0] if ctrl else (0.0, 0.0)
        return [Vector2(x=p[0], y=p[1])] * n

    ctrl = [ctrl[0]] + ctrl + [ctrl[-1]]

    _SAMPLES_PER_SEG = 20
    dense: list[tuple[float, float]] = [ctrl[1]]
    cum: list[float] = [0.0]

    for i in range(1, len(ctrl) - 2):
        p0, p1, p2, p3 = ctrl[i - 1], ctrl[i], ctrl[i + 1], ctrl[i + 2]
        for s in range(1, _SAMPLES_PER_SEG + 1):
            pt = _catmull_rom(p0, p1, p2, p3, s / _SAMPLES_PER_SEG)
            ddx = pt[0] - dense[-1][0]
            ddy = pt[1] - dense[-1][1]
            cum.append(cum[-1] + math.sqrt(ddx * ddx + ddy * ddy))
            dense.append(pt)

    total = cum[-1]
    if total < 0.01:
        return [Vector2(x=ctrl[1][0], y=ctrl[1][1])] * n

    result: list[Vector2] = []
    j = 1
    for k in range(n):
        target = k * spacing
        if target >= total:
            result.append(Vector2(x=dense[-1][0], y=dense[-1][1]))
            continue
        while j < len(cum) and cum[j] < target:
            j += 1
        seg = cum[j] - cum[j - 1]
        t = (target - cum[j - 1]) / max(seg, 1e-9)
        ix = dense[j - 1][0] + t * (dense[j][0] - dense[j - 1][0])
        iy = dense[j - 1][1] + t * (dense[j][1] - dense[j - 1][1])
        result.append(Vector2(x=ix, y=iy))

    return result


def _catmull_rom(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Evaluate a Catmull-Rom spline at t ∈ [0, 1] between p1 and p2."""
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        2 * p1[0]
        + (-p0[0] + p2[0]) * t
        + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        2 * p1[1]
        + (-p0[1] + p2[1]) * t
        + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
    )
    return x, y


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


def emergency_stop_trajectory(pose: Pose, timestamp: float) -> LocalTrajectory:
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
