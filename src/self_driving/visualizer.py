"""Pygame 2D top-down visualiser.

Renders the simulation world in real time:
  - Road graph (edges in gray, nodes as dots)
  - Active route overlay (magenta)
  - Other actors (blue = vehicle, green = pedestrian)
  - Ego vehicle (red rectangle with heading arrow)
  - LiDAR rays (cyan, semi-transparent)
  - Planned trajectory (yellow dots)
  - HUD: speed, steering angle, simulation clock
"""

import math
from collections import deque
from typing import TYPE_CHECKING

try:
    import pygame

    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

from self_driving.models import (
    ControlInput,
    LidarScan,
    LocalTrajectory,
    RoadEdge,
    Vector2,
)

if TYPE_CHECKING:
    from self_driving.simulation import StepTimings
    from self_driving.world import SimulationWorld

# Colours (RGB)
_COL_BG = (40, 55, 40)  # dark green ground
_COL_ROAD_SURFACE = (70, 70, 70)  # asphalt
_COL_CURB = (160, 160, 160)  # light gray curb edge lines
_COL_LANE_MARK = (200, 200, 200)  # white lane edge markings
_COL_CENTER_MARK = (200, 175, 0)  # yellow dashed centre line
_COL_BUILDING = (95, 80, 70)  # brownish brick walls
_COL_BUILDING_OUTLINE = (60, 50, 45)  # darker outline
_COL_ROUTE = (200, 50, 200)
_COL_EGO = (220, 60, 60)
_COL_VEHICLE = (60, 120, 220)
_COL_PED = (60, 200, 80)
_COL_LIDAR = (0, 200, 220)
_COL_TRAJ = (255, 220, 0)
_COL_CENTERLINE = (0, 200, 255)   # cyan — Catmull-Rom reference path
_COL_MPC_PRED = (255, 100, 0)         # orange — MPC predicted horizon
_COL_LANE_FILL = (0, 180, 120, 35)   # RGBA — semi-transparent green fill
_COL_LANE_EDGE = (0, 220, 140)        # lane boundary lines
_LANE_HALF_W_M = 1.75                 # half of a standard 3.5 m lane
_COL_HUD = (240, 240, 240)
_COL_ARROW_RIGHT = (80, 220, 80)  # green  – right turns
_COL_ARROW_STRAIGHT = (200, 200, 200)  # white  – straight
_COL_ARROW_LEFT = (255, 180, 0)  # amber  – left turns

_DASH_LEN_M = 3.0  # metres per dash stroke
_GAP_LEN_M = 3.0  # metres per dash gap

_WINDOW_W = 1200
_MAP_H = 900
_CHART_H = 220
_WINDOW_H = _MAP_H + _CHART_H
_MARGIN = 40
_CHART_SAMPLES = 300
_TIMING_BUDGET_MS = 50.0  # 20 Hz target step budget


class Visualizer:
    """Pygame-based 2D top-down visualiser for the simulation."""

    def __init__(
        self,
        world_size_m: float = 400.0,
        world_min_x: float = 0.0,
        world_min_y: float = 0.0,
    ) -> None:
        """Initialise Pygame window.

        Parameters
        ----------
        world_size_m: side length of the square world in metres
        world_min_x: minimum x world coordinate (left edge)
        world_min_y: minimum y world coordinate (bottom edge)
        """
        if not _PYGAME_AVAILABLE:
            raise RuntimeError("pygame is not installed. Run: uv add pygame")
        pygame.init()
        self._screen = pygame.display.set_mode((_WINDOW_W, _WINDOW_H))
        pygame.display.set_caption("Self-Driving Simulation")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 14)
        self._scale = (_WINDOW_W - 2 * _MARGIN) / world_size_m
        self._offset_x = _MARGIN - world_min_x * self._scale
        self._offset_y = _MARGIN - world_min_y * self._scale
        # Rolling chart buffer: (speed_kmh, target_speed_kmh|None, steering_deg, steer_delta_deg|None, accel_cmd)
        self._chart_buf: deque[
            tuple[float, float | None, float, float | None, float]
        ] = deque(maxlen=_CHART_SAMPLES)
        # Timing buffer: (total_ms, mpc_ms, path_planning_ms, other_ms)
        self._timing_buf: deque[tuple[float, float, float, float]] = deque(maxlen=_CHART_SAMPLES)

    def _w2s(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to screen pixel coordinates."""
        sx = int(x * self._scale + self._offset_x)
        sy = int(_MAP_H - (y * self._scale + self._offset_y))
        return sx, sy

    def render(
        self,
        world: "SimulationWorld",
        scan: LidarScan | None,
        trajectory: LocalTrajectory | None,
        control: ControlInput | None = None,
        timings: "StepTimings | None" = None,
    ) -> bool:
        """Render one frame. Returns False if the window was closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self._screen.fill(_COL_BG)

        self._draw_buildings(world)
        self._draw_roads(world)
        self._draw_route(world)
        self._draw_destination(world)

        if scan is not None:
            self._draw_lidar(scan)

        if trajectory is not None:
            self._draw_centerline(trajectory)
            self._draw_trajectory(trajectory)

        if control is not None and control.predicted_poses:
            self._draw_mpc_prediction(control)

        self._draw_actors(world)
        self._draw_ego(world)
        self._draw_hud(world, control)
        self._update_chart_buf(world, trajectory, control)
        self._update_timing_buf(timings)
        self._draw_rolling_charts()

        pygame.display.flip()
        self._clock.tick(60)
        return True

    def close(self) -> None:
        """Shut down Pygame."""
        pygame.quit()

    # ------------------------------------------------------------------
    # Private draw helpers
    # ------------------------------------------------------------------

    def _draw_buildings(self, world: "SimulationWorld") -> None:
        for b in world.road_map.buildings:
            hw, hd = b.width / 2, b.depth / 2
            corners = [
                (b.cx - hw, b.cy - hd),
                (b.cx + hw, b.cy - hd),
                (b.cx + hw, b.cy + hd),
                (b.cx - hw, b.cy + hd),
            ]
            pts = [self._w2s(cx, cy) for cx, cy in corners]
            pygame.draw.polygon(self._screen, _COL_BUILDING, pts)
            pygame.draw.polygon(self._screen, _COL_BUILDING_OUTLINE, pts, 1)

    def _draw_roads(self, world: "SimulationWorld") -> None:
        node_by_id = {n.node_id: n.position for n in world.road_map.nodes}
        edge_by_key = {(e.from_node, e.to_node): e for e in world.road_map.edges}

        # Precompute intersection circle radius per node (= max road half-width)
        node_radius: dict[int, float] = {}
        for e in world.road_map.edges:
            rev = edge_by_key.get((e.to_node, e.from_node))
            lanes_ba = rev.num_lanes if rev else e.num_lanes
            r = max(e.num_lanes, lanes_ba) * e.lane_width
            for nid in (e.from_node, e.to_node):
                node_radius[nid] = max(node_radius.get(nid, 0.0), r)

        # 1. Road surfaces
        drawn: set[tuple[int, int]] = set()
        for edge in world.road_map.edges:
            u, v = edge.from_node, edge.to_node
            key = (min(u, v), max(u, v))
            if key in drawn:
                continue
            a = node_by_id.get(u)
            b = node_by_id.get(v)
            if not a or not b:
                continue
            reverse = edge_by_key.get((v, u))
            lanes_ab = edge.num_lanes
            lanes_ba = reverse.num_lanes if reverse else edge.num_lanes
            self._draw_road_surface(
                a.x, a.y, b.x, b.y, edge.lane_width, lanes_ab, lanes_ba
            )
            drawn.add(key)

        # 2. Intersection fill squares on top — cover ragged road-polygon ends
        self._draw_intersection_boxes(world, node_radius)

        # 3. Lane markings, clipped so they stop at each intersection boundary
        drawn_marks: set[tuple[int, int]] = set()
        for edge in world.road_map.edges:
            u, v = edge.from_node, edge.to_node
            key = (min(u, v), max(u, v))
            if key in drawn_marks:
                continue
            a = node_by_id.get(u)
            b = node_by_id.get(v)
            if not a or not b:
                continue
            reverse = edge_by_key.get((v, u))
            lanes_ab = edge.num_lanes
            lanes_ba = reverse.num_lanes if reverse else edge.num_lanes
            self._draw_lane_markings(
                a.x,
                a.y,
                b.x,
                b.y,
                edge.lane_width,
                lanes_ab,
                lanes_ba,
                setback_a=node_radius.get(u, 0.0),
                setback_b=node_radius.get(v, 0.0),
            )
            drawn_marks.add(key)

        # 4. Intersection arrows on top of markings
        self._draw_lane_connection_arrows(world, node_by_id, edge_by_key, node_radius)

    def _draw_road_surface(
        self,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        lane_width: float,
        lanes_ab: int,
        lanes_ba: int,
    ) -> None:
        """Draw filled road polygon between two points.

        The A→B direction occupies lanes_ab lanes to the right of the
        centreline; the B→A direction occupies lanes_ba lanes to the left.
        """
        dx = bx - ax
        dy = by - ay
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.1:
            return
        dx /= length
        dy /= length
        px, py = -dy, dx  # perpendicular pointing LEFT of A→B

        hw_left = lanes_ba * lane_width  # B→A traffic (left of A→B)
        hw_right = lanes_ab * lane_width  # A→B traffic (right of A→B)

        corners = [
            self._w2s(ax + px * hw_left, ay + py * hw_left),
            self._w2s(bx + px * hw_left, by + py * hw_left),
            self._w2s(bx - px * hw_right, by - py * hw_right),
            self._w2s(ax - px * hw_right, ay - py * hw_right),
        ]
        pygame.draw.polygon(self._screen, _COL_ROAD_SURFACE, corners)

    def _draw_lane_markings(
        self,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        lane_width: float,
        lanes_ab: int,
        lanes_ba: int,
        setback_a: float = 0.0,
        setback_b: float = 0.0,
    ) -> None:
        """Draw curb lines, lane dividers, and centre line for a road segment.

        Markings are clipped by setback_a/setback_b so they don't cross
        intersection fill squares.

        Lane layout (perpendicular, looking from A toward B):
          left curb  ← [B→A lanes] ← centre line → [A→B lanes] → right curb
        """
        dx = bx - ax
        dy = by - ay
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.1:
            return
        dx /= length
        dy /= length
        px, py = -dy, dx  # perpendicular LEFT of A→B

        t_start = setback_a
        t_end = length - setback_b
        if t_end <= t_start:
            return

        # Clipped endpoints for solid lines
        sx0 = ax + dx * t_start
        sy0 = ay + dy * t_start
        sx1 = ax + dx * t_end
        sy1 = ay + dy * t_end
        clip_len = t_end - t_start

        hw_left = lanes_ba * lane_width
        hw_right = lanes_ab * lane_width

        # Solid curb lines at road edges
        pygame.draw.line(
            self._screen,
            _COL_CURB,
            self._w2s(sx0 + px * hw_left, sy0 + py * hw_left),
            self._w2s(sx1 + px * hw_left, sy1 + py * hw_left),
            1,
        )
        pygame.draw.line(
            self._screen,
            _COL_CURB,
            self._w2s(sx0 - px * hw_right, sy0 - py * hw_right),
            self._w2s(sx1 - px * hw_right, sy1 - py * hw_right),
            1,
        )

        # Dashed yellow centre line
        self._draw_dashed_line(sx0, sy0, dx, dy, _COL_CENTER_MARK, clip_len)

        # White dashed lane dividers within B→A direction (left side)
        for k in range(1, lanes_ba):
            offset = k * lane_width
            self._draw_dashed_line(
                sx0 + px * offset,
                sy0 + py * offset,
                dx,
                dy,
                _COL_LANE_MARK,
                clip_len,
            )

        # White dashed lane dividers within A→B direction (right side)
        for k in range(1, lanes_ab):
            offset = k * lane_width
            self._draw_dashed_line(
                sx0 - px * offset,
                sy0 - py * offset,
                dx,
                dy,
                _COL_LANE_MARK,
                clip_len,
            )

    def _draw_intersection_boxes(
        self,
        world: "SimulationWorld",
        node_radius: dict[int, float],
    ) -> None:
        """Fill each intersection with a square to cover ragged road-polygon ends."""
        for node in world.road_map.nodes:
            radius = node_radius.get(node.node_id, 0.0)
            if radius < 0.1:
                continue
            sx, sy = self._w2s(node.position.x, node.position.y)
            half_px = int(radius * self._scale)
            pygame.draw.rect(
                self._screen,
                _COL_ROAD_SURFACE,
                (sx - half_px, sy - half_px, half_px * 2, half_px * 2),
            )

    def _draw_lane_connection_arrows(
        self,
        world: "SimulationWorld",
        node_by_id: dict[int, Vector2],
        edge_by_key: dict[tuple[int, int], RoadEdge],
        node_radius: dict[int, float],
    ) -> None:
        """Draw lane-direction arrows on the road surface just before intersections.

        Each incoming lane gets one composite arrow showing its permitted
        movements (straight / left / right).  Arrows are painted at the lane
        centre, positioned just before the intersection boundary.
        """
        # Collect permitted turn directions per (node_id, from_edge, from_lane)
        lane_turns: dict[tuple, set[str]] = {}
        for conn in world.road_map.lane_connections:
            nid = conn.node_id
            node_pos = node_by_id.get(nid)
            if node_pos is None:
                continue
            u, _ = conn.from_edge
            _, w = conn.to_edge
            a = node_by_id.get(u)
            c = node_by_id.get(w)
            if a is None or c is None:
                continue

            inc_h = math.atan2(node_pos.y - a.y, node_pos.x - a.x)
            out_h = math.atan2(c.y - node_pos.y, c.x - node_pos.x)
            turn = out_h - inc_h
            while turn > math.pi:
                turn -= 2 * math.pi
            while turn <= -math.pi:
                turn += 2 * math.pi

            if turn < -math.pi / 4:
                direction = "right"
            elif turn > math.pi / 4:
                direction = "left"
            else:
                direction = "straight"

            key = (nid, conn.from_edge, conn.from_lane)
            lane_turns.setdefault(key, set()).add(direction)

        # Draw one composite arrow per lane approach
        for (nid, from_edge, from_lane), turns in lane_turns.items():
            node_pos = node_by_id.get(nid)
            if node_pos is None:
                continue
            u, v = from_edge
            a = node_by_id.get(u)
            if a is None:
                continue
            inc_edge = edge_by_key.get((u, v))
            if inc_edge is None:
                continue

            lw = inc_edge.lane_width
            node_r = node_radius.get(nid, lw)

            # Unit vector along incoming direction
            ddx = node_pos.x - a.x
            ddy = node_pos.y - a.y
            seg = math.sqrt(ddx * ddx + ddy * ddy)
            if seg < 0.1:
                continue
            fdx, fdy = ddx / seg, ddy / seg  # forward (travel direction)
            rdx, rdy = fdy, -fdx  # right perpendicular

            # Lane centre (lane 0 = rightmost)
            lane_cx = node_pos.x - fdx * (node_r + lw) + rdx * (from_lane + 0.5) * lw
            lane_cy = node_pos.y - fdy * (node_r + lw) + rdy * (from_lane + 0.5) * lw

            self._draw_composite_arrow(lane_cx, lane_cy, fdx, fdy, rdx, rdy, lw, turns)

    def _draw_composite_arrow(
        self,
        cx: float,
        cy: float,
        fdx: float,
        fdy: float,
        rdx: float,
        rdy: float,
        lane_width: float,
        turns: set[str],
    ) -> None:
        """Draw a composite lane arrow at world position (cx, cy).

        Renders a short stem in the travel direction with a small arrowhead
        for each permitted turn: straight (white), right (green), left (amber).

        The stem base is behind the centre; each branch tip lies in the
        corresponding turn direction so the whole symbol reads like real
        road markings.
        """
        stem = lane_width * 0.55  # half-stem length (world metres)

        base_w = self._w2s(cx - fdx * stem, cy - fdy * stem)
        fork_w = self._w2s(cx, cy)

        # Stem
        pygame.draw.line(self._screen, _COL_ARROW_STRAIGHT, base_w, fork_w, 1)

        # Branch tips and arrowheads
        branches: list[tuple[float, float, float, float, tuple[int, int, int]]] = []
        if "straight" in turns:
            tx = cx + fdx * stem
            ty = cy + fdy * stem
            branches.append((fdx, fdy, tx, ty, _COL_ARROW_STRAIGHT))
        if "right" in turns:
            tx = cx + fdx * stem * 0.5 + rdx * stem * 0.8
            ty = cy + fdy * stem * 0.5 + rdy * stem * 0.8
            branches.append((rdx, rdy, tx, ty, _COL_ARROW_RIGHT))
        if "left" in turns:
            tx = cx + fdx * stem * 0.5 - rdx * stem * 0.8
            ty = cy + fdy * stem * 0.5 - rdy * stem * 0.8
            branches.append((-rdx, -rdy, tx, ty, _COL_ARROW_LEFT))

        for brdx, brdy, tx, ty, col in branches:
            tip_s = self._w2s(tx, ty)
            pygame.draw.line(self._screen, col, fork_w, tip_s, 1)
            # Arrowhead
            head = lane_width * 0.2
            wing = lane_width * 0.15
            base_cx_w = tx - brdx * head
            base_cy_w = ty - brdy * head
            left_w = self._w2s(base_cx_w - brdy * wing, base_cy_w + brdx * wing)
            right_w = self._w2s(base_cx_w + brdy * wing, base_cy_w - brdx * wing)
            pygame.draw.polygon(self._screen, col, [tip_s, left_w, right_w])

    def _draw_dashed_line(
        self,
        ax: float,
        ay: float,
        dx: float,
        dy: float,
        colour: tuple[int, int, int],
        length: float,
    ) -> None:
        """Draw a dashed line starting at (ax, ay) in direction (dx, dy)."""
        t = 0.0
        drawing = True
        while t < length:
            seg = _DASH_LEN_M if drawing else _GAP_LEN_M
            t2 = min(t + seg, length)
            if drawing:
                p1 = self._w2s(ax + dx * t, ay + dy * t)
                p2 = self._w2s(ax + dx * t2, ay + dy * t2)
                pygame.draw.line(self._screen, colour, p1, p2, 1)
            t = t2
            drawing = not drawing

    def _draw_destination(self, world: "SimulationWorld") -> None:
        """Draw a marker at the destination node."""
        dest = world.destination
        if dest is None:
            return
        node_pos = next(
            (n.position for n in world.road_map.nodes if n.node_id == dest), None
        )
        if node_pos is None:
            return
        sx, sy = self._w2s(node_pos.x, node_pos.y)
        r = 12
        pygame.draw.circle(self._screen, (255, 220, 0), (sx, sy), r, 3)
        arm = 7
        pygame.draw.line(self._screen, (255, 220, 0), (sx - arm, sy - arm), (sx + arm, sy + arm), 2)
        pygame.draw.line(self._screen, (255, 220, 0), (sx + arm, sy - arm), (sx - arm, sy + arm), 2)

    def _draw_route(self, world: "SimulationWorld") -> None:
        route = world.current_route
        if route is None or len(route.waypoint_ids) < 2:
            return
        node_by_id = {n.node_id: n.position for n in world.road_map.nodes}
        pts = [
            self._w2s(node_by_id[nid].x, node_by_id[nid].y)
            for nid in route.waypoint_ids
            if nid in node_by_id
        ]
        if len(pts) >= 2:
            pygame.draw.lines(self._screen, _COL_ROUTE, False, pts, 2)

    def _draw_lidar(self, scan: LidarScan) -> None:
        surf = pygame.Surface((_WINDOW_W, _WINDOW_H), pygame.SRCALPHA)
        for pt in scan.points:
            if not pt.hit:
                continue
            end_x = scan.ego_pose.x + pt.distance * math.cos(pt.angle)
            end_y = scan.ego_pose.y + pt.distance * math.sin(pt.angle)
            pygame.draw.line(
                surf,
                (*_COL_LIDAR, 60),
                self._w2s(scan.ego_pose.x, scan.ego_pose.y),
                self._w2s(end_x, end_y),
                1,
            )
        self._screen.blit(surf, (0, 0))

    def _draw_mpc_prediction(self, control: ControlInput) -> None:
        """Draw the MPC horizon rollout as an orange polyline with dots."""
        pts = [self._w2s(p.x, p.y) for p in control.predicted_poses]
        if len(pts) >= 2:
            pygame.draw.lines(self._screen, _COL_MPC_PRED, False, pts, 2)
        for pt in pts:
            pygame.draw.circle(self._screen, _COL_MPC_PRED, pt, 3)

    def _draw_centerline(self, trajectory: LocalTrajectory) -> None:
        """Draw the Catmull-Rom reference path as a dashed cyan line."""
        pts = [self._w2s(p.x, p.y) for p in trajectory.centerline]
        if len(pts) >= 2:
            self._draw_dashed_screen_line(pts, _COL_CENTERLINE)

    def _draw_trajectory(self, trajectory: LocalTrajectory) -> None:
        pts = trajectory.points
        if not pts:
            return

        # Build left/right edge points for the lane corridor
        left_s: list[tuple[int, int]] = []
        right_s: list[tuple[int, int]] = []
        for pt in pts:
            h = pt.pose.heading
            # Perpendicular: left = (-sin, cos), right = (sin, -cos)
            lx = pt.pose.x - math.sin(h) * _LANE_HALF_W_M
            ly = pt.pose.y + math.cos(h) * _LANE_HALF_W_M
            rx = pt.pose.x + math.sin(h) * _LANE_HALF_W_M
            ry = pt.pose.y - math.cos(h) * _LANE_HALF_W_M
            left_s.append(self._w2s(lx, ly))
            right_s.append(self._w2s(rx, ry))

        # Semi-transparent fill
        if len(left_s) >= 2:
            polygon = left_s + list(reversed(right_s))
            surf = pygame.Surface((_WINDOW_W, _MAP_H), pygame.SRCALPHA)
            pygame.draw.polygon(surf, _COL_LANE_FILL, polygon)
            self._screen.blit(surf, (0, 0))
            # Lane edge lines
            pygame.draw.lines(self._screen, _COL_LANE_EDGE, False, left_s, 1)
            pygame.draw.lines(self._screen, _COL_LANE_EDGE, False, right_s, 1)

        # Centerline dots
        for pt in pts:
            sx, sy = self._w2s(pt.pose.x, pt.pose.y)
            pygame.draw.circle(self._screen, _COL_TRAJ, (sx, sy), 3)

    def _draw_actors(self, world: "SimulationWorld") -> None:
        for actor in world.actor_states:
            col = _COL_VEHICLE if actor.actor_type == "vehicle" else _COL_PED
            sx, sy = self._w2s(actor.pose.x, actor.pose.y)
            if actor.actor_type == "vehicle":
                self._draw_rotated_rect(sx, sy, actor.pose.heading, 12, 6, col)
            else:
                pygame.draw.circle(self._screen, col, (sx, sy), 5)

    def _draw_ego(self, world: "SimulationWorld") -> None:
        ego = world.ego_state
        sx, sy = self._w2s(ego.pose.x, ego.pose.y)
        self._draw_rotated_rect(sx, sy, ego.pose.heading, 14, 7, _COL_EGO)
        # Heading arrow
        arrow_len = 18
        ax = sx + int(arrow_len * math.cos(ego.pose.heading))
        ay = sy - int(arrow_len * math.sin(ego.pose.heading))
        pygame.draw.line(self._screen, (255, 255, 255), (sx, sy), (ax, ay), 2)

    def _draw_rotated_rect(
        self,
        cx: int,
        cy: int,
        heading: float,
        half_l: int,
        half_w: int,
        color: tuple[int, int, int],
    ) -> None:
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        corners_local = [
            (-half_l, -half_w),
            (half_l, -half_w),
            (half_l, half_w),
            (-half_l, half_w),
        ]
        pts = []
        for lx, ly in corners_local:
            rx = lx * cos_h - ly * sin_h
            ry = lx * sin_h + ly * cos_h
            pts.append((cx + int(rx), cy - int(ry)))
        pygame.draw.polygon(self._screen, color, pts)

    def _draw_hud(
        self, world: "SimulationWorld", control: ControlInput | None = None
    ) -> None:
        ego = world.ego_state
        lines = [
            f"t = {world.clock:6.1f} s",
            f"speed  = {ego.speed * 3.6:5.1f} km/h",
            f"steer  = {math.degrees(ego.steering_angle):+5.1f} deg",
            f"dest   = {world.destination}",
        ]
        for i, line in enumerate(lines):
            surf = self._font.render(line, True, _COL_HUD)
            self._screen.blit(surf, (10, 10 + i * 18))

        self._draw_gauges(world, control)

    def _draw_gauges(
        self, world: "SimulationWorld", control: ControlInput | None
    ) -> None:
        """Draw steering, throttle and brake gauges in the bottom-left corner."""
        ego = world.ego_state
        max_steer = 0.5  # radians, matches VehicleParams.max_steering_angle

        bx = _WINDOW_W - 170  # right-aligned gauge panel
        by = 10  # top of gauge panel

        # --- Steering bar ---
        bar_w = 160
        bar_h = 14
        steer_norm = ego.steering_angle / max_steer  # -1..1
        label = self._font.render("steer", True, _COL_HUD)
        self._screen.blit(label, (bx, by))
        # Track background
        pygame.draw.rect(self._screen, (80, 80, 80), (bx, by + 16, bar_w, bar_h))
        # Centre tick
        cx = bx + bar_w // 2
        pygame.draw.line(
            self._screen, (160, 160, 160), (cx, by + 14), (cx, by + 16 + bar_h + 2), 1
        )
        # Filled region from centre toward steer direction
        fill_w = int(abs(steer_norm) * bar_w // 2)
        fill_col = (255, 200, 0)
        if steer_norm < 0:
            pygame.draw.rect(
                self._screen, fill_col, (cx - fill_w, by + 16, fill_w, bar_h)
            )
        else:
            pygame.draw.rect(self._screen, fill_col, (cx, by + 16, fill_w, bar_h))
        # Outline
        pygame.draw.rect(self._screen, _COL_HUD, (bx, by + 16, bar_w, bar_h), 1)

        # --- Throttle & brake bars ---
        vbar_h = 60
        vbar_w = 18
        gap = 8
        tx = bx
        bry = by + 16 + bar_h + 10  # top of vertical bars

        raw = control.accel_cmd if control else 0.0
        for label_text, value, col, xoff in [
            ("thr", max(0.0, raw), (60, 200, 80), 0),
            ("brk", max(0.0, -raw), (220, 60, 60), vbar_w + gap),
        ]:
            x = tx + xoff
            # Background
            pygame.draw.rect(self._screen, (80, 80, 80), (x, bry, vbar_w, vbar_h))
            # Fill from bottom
            fill_h = int(value * vbar_h)
            if fill_h > 0:
                pygame.draw.rect(
                    self._screen, col, (x, bry + vbar_h - fill_h, vbar_w, fill_h)
                )
            # Outline
            pygame.draw.rect(self._screen, _COL_HUD, (x, bry, vbar_w, vbar_h), 1)
            # Label below
            lbl = self._font.render(label_text, True, _COL_HUD)
            self._screen.blit(lbl, (x, bry + vbar_h + 2))

    def _update_chart_buf(
        self,
        world: "SimulationWorld",
        trajectory: LocalTrajectory | None,
        control: ControlInput | None,
    ) -> None:
        """Append one sample to the rolling chart buffer."""
        ego = world.ego_state
        speed_kmh = ego.speed * 3.6
        steering_deg = math.degrees(ego.steering_angle)
        target_speed_kmh = (
            trajectory.points[0].speed * 3.6
            if trajectory and trajectory.points
            else None
        )
        steer_delta_deg = (
            math.degrees(control.steering_delta) if control is not None else None
        )
        accel_cmd = control.accel_cmd if control is not None else 0.0
        self._chart_buf.append(
            (speed_kmh, target_speed_kmh, steering_deg, steer_delta_deg, accel_cmd)
        )

    def _update_timing_buf(self, timings: "StepTimings | None") -> None:
        """Append one timing sample (or zeros) to the rolling timing buffer."""
        if timings is None:
            self._timing_buf.append((0.0, 0.0, 0.0, 0.0))
            return
        other = (
            timings.actors_ms
            + timings.localization_ms
            + timings.routing_ms
            + timings.behavior_ms
            + timings.physics_ms
        )
        self._timing_buf.append((
            timings.total_ms,
            timings.mpc_ms,
            timings.path_planning_ms,
            other,
        ))

    def _draw_rolling_charts(self) -> None:
        """Draw four rolling line charts in the bottom strip of the window."""
        if not self._chart_buf:
            return

        pad = 10
        gap = 10
        chart_w = (_WINDOW_W - 2 * pad - 3 * gap) // 4
        title_h = 20
        plot_h = _CHART_H - title_h - 2 * pad
        chart_y = _MAP_H + pad  # top of chart strip
        plot_y = chart_y + title_h  # top of plot area

        # Unzip buffer into series
        buf = list(self._chart_buf)
        speeds = [s[0] for s in buf]
        targets = [s[1] for s in buf]
        steers = [s[2] for s in buf]
        deltas = [s[3] for s in buf]
        accels = [s[4] for s in buf]

        n = len(buf)
        xs = [int(i * chart_w / max(n - 1, 1)) for i in range(n)]

        def draw_series(
            series: list[float | None],
            lo: float,
            hi: float,
            x_off: int,
            color: tuple[int, int, int],
            dashed: bool = False,
        ) -> None:
            pts = []
            for i, val in enumerate(series):
                if val is None:
                    if len(pts) >= 2:
                        if not dashed:
                            pygame.draw.lines(self._screen, color, False, pts, 1)
                        else:
                            self._draw_dashed_screen_line(pts, color)
                    pts = []
                    continue
                clamped = max(lo, min(hi, val))
                norm = (clamped - lo) / (hi - lo) if hi != lo else 0.5
                px = x_off + xs[i]
                py = plot_y + plot_h - int(norm * plot_h)
                pts.append((px, py))
            if len(pts) >= 2:
                if not dashed:
                    pygame.draw.lines(self._screen, color, False, pts, 1)
                else:
                    self._draw_dashed_screen_line(pts, color)

        def draw_zero_line(lo: float, hi: float, x_off: int) -> None:
            if lo < 0 < hi:
                norm = (0.0 - lo) / (hi - lo)
                zy = plot_y + plot_h - int(norm * plot_h)
                pygame.draw.line(
                    self._screen, (100, 100, 100), (x_off, zy), (x_off + chart_w, zy), 1
                )

        def draw_legend(
            x_off: int, entries: list[tuple[str, tuple[int, int, int], bool]]
        ) -> None:
            """Draw a legend in the top-right corner of the chart."""
            swatch_w = 14
            text_gap = 3
            row_h = 14
            # Measure total width to right-align
            col_widths = [
                swatch_w + text_gap + self._font.size(label)[0]
                for label, _, _ in entries
            ]
            col_gap = 8
            total_w = sum(col_widths) + col_gap * (len(entries) - 1)
            lx = x_off + chart_w - total_w - 4
            ly = plot_y + 3
            for i, (label, color, dashed) in enumerate(entries):
                ex = lx + sum(col_widths[:i]) + col_gap * i
                mid_y = ly + row_h // 2
                if dashed:
                    self._draw_dashed_screen_line(
                        [(ex, mid_y), (ex + swatch_w, mid_y)], color
                    )
                else:
                    pygame.draw.line(
                        self._screen, color, (ex, mid_y), (ex + swatch_w, mid_y), 2
                    )
                lbl_surf = self._font.render(label, True, (200, 200, 200))
                self._screen.blit(lbl_surf, (ex + swatch_w + text_gap, ly))

        # Chart backgrounds and titles
        charts = [
            (0, "vel (km/h)"),
            (1, "steer (deg)"),
            (2, "thr / brk"),
            (3, "step time (ms)"),
        ]
        for idx, title in charts:
            cx = pad + idx * (chart_w + gap)
            pygame.draw.rect(
                self._screen, (30, 30, 30), (cx, chart_y, chart_w, _CHART_H - 2 * pad)
            )
            lbl = self._font.render(title, True, _COL_HUD)
            self._screen.blit(lbl, (cx + 2, chart_y + 2))

        # Chart 1 – Velocity
        x0 = pad
        draw_series(speeds, 0.0, 60.0, x0, (255, 220, 0))
        draw_series(targets, 0.0, 60.0, x0, (0, 200, 220), dashed=True)
        draw_legend(x0, [("act", (255, 220, 0), False), ("tgt", (0, 200, 220), True)])
        cur = f"{speeds[-1]:.1f}" if speeds else "—"
        tgt = f"{targets[-1]:.1f}" if targets and targets[-1] is not None else "—"
        self._screen.blit(
            self._font.render(f"act {cur}  tgt {tgt}", True, (180, 180, 180)),
            (x0 + 2, plot_y + plot_h - 14),
        )

        # Chart 2 – Steering
        x1 = pad + chart_w + gap
        draw_zero_line(-30.0, 30.0, x1)
        draw_series(steers, -30.0, 30.0, x1, (255, 220, 0))
        draw_series(deltas, -30.0, 30.0, x1, (255, 140, 0))
        draw_legend(x1, [("act", (255, 220, 0), False), ("MPC δ", (255, 140, 0), False)])
        cur_s = f"{steers[-1]:+.1f}" if steers else "—"
        cur_d = f"{deltas[-1]:+.1f}" if deltas and deltas[-1] is not None else "—"
        self._screen.blit(
            self._font.render(f"act {cur_s}  Δ {cur_d}", True, (180, 180, 180)),
            (x1 + 2, plot_y + plot_h - 14),
        )

        # Chart 3 – Throttle / Brake
        x2 = pad + 2 * (chart_w + gap)
        throttles = [max(0.0, a) for a in accels]
        brakes = [max(0.0, -a) for a in accels]
        draw_series(throttles, 0.0, 1.0, x2, (60, 200, 80))
        draw_series(brakes, 0.0, 1.0, x2, (220, 60, 60))
        draw_legend(x2, [("thr", (60, 200, 80), False), ("brk", (220, 60, 60), False)])
        cur_a = accels[-1] if accels else 0.0
        label_a = f"thr {cur_a:.2f}" if cur_a >= 0 else f"brk {-cur_a:.2f}"
        self._screen.blit(
            self._font.render(label_a, True, (180, 180, 180)),
            (x2 + 2, plot_y + plot_h - 14),
        )

        # Chart 4 – Step timing breakdown
        x3 = pad + 3 * (chart_w + gap)
        if self._timing_buf:
            tbuf = list(self._timing_buf)
            t_total = [s[0] for s in tbuf]
            t_mpc   = [s[1] for s in tbuf]
            t_path  = [s[2] for s in tbuf]
            t_other = [s[3] for s in tbuf]
            t_max = max(max(t_total), _TIMING_BUDGET_MS * 1.1)

            # Stacked filled areas: other (bottom), path_planning, mpc (top)
            def draw_stacked(
                lower: list[float],
                upper: list[float],
                color: tuple[int, int, int],
            ) -> None:
                pts_top: list[tuple[int, int]] = []
                pts_bot: list[tuple[int, int]] = []
                m = len(tbuf)
                for i in range(m):
                    xi = x3 + int(i * chart_w / max(m - 1, 1))
                    norm_u = max(0.0, min(1.0, upper[i] / t_max))
                    norm_l = max(0.0, min(1.0, lower[i] / t_max))
                    pts_top.append((xi, plot_y + plot_h - int(norm_u * plot_h)))
                    pts_bot.append((xi, plot_y + plot_h - int(norm_l * plot_h)))
                polygon = pts_top + list(reversed(pts_bot))
                if len(polygon) >= 3:
                    surf = pygame.Surface((_WINDOW_W, _WINDOW_H), pygame.SRCALPHA)
                    pygame.draw.polygon(surf, (*color, 180), polygon)
                    self._screen.blit(surf, (0, 0))

            base = [0.0] * len(tbuf)
            other_top = [base[i] + t_other[i] for i in range(len(tbuf))]
            path_top  = [other_top[i] + t_path[i] for i in range(len(tbuf))]
            mpc_top   = [path_top[i] + t_mpc[i]   for i in range(len(tbuf))]

            draw_stacked(base,      other_top, (120, 120, 120))   # other: gray
            draw_stacked(other_top, path_top,  (70, 130, 180))    # path: steel blue
            draw_stacked(path_top,  mpc_top,   (200, 50, 50))     # mpc: crimson

            # Total step time line
            draw_series(t_total, 0.0, t_max, x3, (220, 220, 220))

            # Budget line at 50 ms
            budget_norm = min(1.0, _TIMING_BUDGET_MS / t_max)
            by_px = plot_y + plot_h - int(budget_norm * plot_h)
            pygame.draw.line(
                self._screen, (255, 200, 0),
                (x3, by_px), (x3 + chart_w, by_px), 1,
            )

            draw_legend(x3, [
                ("mpc", (200, 50, 50), False),
                ("path", (70, 130, 180), False),
                ("other", (120, 120, 120), False),
            ])
            cur_total = t_total[-1] if t_total else 0.0
            self._screen.blit(
                self._font.render(f"total {cur_total:.1f}ms", True, (180, 180, 180)),
                (x3 + 2, plot_y + plot_h - 14),
            )

    def _draw_dashed_screen_line(
        self, pts: list[tuple[int, int]], color: tuple[int, int, int]
    ) -> None:
        """Draw a dashed polyline through screen-space points."""
        dash_px = 8
        gap_px = 6
        drawing = True
        acc = 0
        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len < 0.5:
                continue
            dx, dy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
            t = 0.0
            while t < seg_len:
                remain = (dash_px if drawing else gap_px) - acc
                step = min(remain, seg_len - t)
                if drawing:
                    sx0 = int(x0 + dx * t)
                    sy0 = int(y0 + dy * t)
                    sx1 = int(x0 + dx * (t + step))
                    sy1 = int(y0 + dy * (t + step))
                    pygame.draw.line(self._screen, color, (sx0, sy0), (sx1, sy1), 1)
                t += step
                acc += step
                if acc >= (dash_px if drawing else gap_px):
                    acc = 0
                    drawing = not drawing
