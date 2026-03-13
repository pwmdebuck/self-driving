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
from typing import TYPE_CHECKING

try:
    import pygame

    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

from self_driving.models import (
    LidarScan,
    LocalTrajectory,
)

if TYPE_CHECKING:
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
_COL_HUD = (240, 240, 240)

_DASH_LEN_M = 3.0  # metres per dash stroke
_GAP_LEN_M = 3.0  # metres per dash gap

_WINDOW_W = 900
_WINDOW_H = 900
_MARGIN = 40


class Visualizer:
    """Pygame-based 2D top-down visualiser for the simulation."""

    def __init__(self, world_size_m: float = 400.0) -> None:
        """Initialise Pygame window.

        Parameters
        ----------
        world_size_m: side length of the square world in metres
        """
        if not _PYGAME_AVAILABLE:
            raise RuntimeError("pygame is not installed. Run: uv add pygame")
        pygame.init()
        self._screen = pygame.display.set_mode((_WINDOW_W, _WINDOW_H))
        pygame.display.set_caption("Self-Driving Simulation")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 14)
        self._scale = (_WINDOW_W - 2 * _MARGIN) / world_size_m
        self._offset_x = _MARGIN
        self._offset_y = _MARGIN

    def _w2s(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to screen pixel coordinates."""
        sx = int(x * self._scale + self._offset_x)
        sy = int(_WINDOW_H - (y * self._scale + self._offset_y))
        return sx, sy

    def render(
        self,
        world: "SimulationWorld",
        scan: LidarScan | None,
        trajectory: LocalTrajectory | None,
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

        if scan is not None:
            self._draw_lidar(scan)

        if trajectory is not None:
            self._draw_trajectory(trajectory)

        self._draw_actors(world)
        self._draw_ego(world)
        self._draw_hud(world)

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

        # Draw road surfaces first (deduplicated per undirected pair)
        drawn: set[tuple[int, int]] = set()
        for edge in world.road_map.edges:
            key = (min(edge.from_node, edge.to_node), max(edge.from_node, edge.to_node))
            a = node_by_id.get(edge.from_node)
            b = node_by_id.get(edge.to_node)
            if not a or not b:
                continue
            if key not in drawn:
                self._draw_road_surface(a.x, a.y, b.x, b.y, edge.lane_width)
                drawn.add(key)

        # Draw lane markings on top
        drawn_marks: set[tuple[int, int]] = set()
        for edge in world.road_map.edges:
            key = (min(edge.from_node, edge.to_node), max(edge.from_node, edge.to_node))
            a = node_by_id.get(edge.from_node)
            b = node_by_id.get(edge.to_node)
            if not a or not b or key in drawn_marks:
                continue
            self._draw_lane_markings(a.x, a.y, b.x, b.y, edge.lane_width)
            drawn_marks.add(key)

    def _draw_road_surface(
        self, ax: float, ay: float, bx: float, by: float, lane_width: float
    ) -> None:
        """Draw filled road polygon between two points."""
        dx = bx - ax
        dy = by - ay
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.1:
            return
        dx /= length
        dy /= length
        px, py = -dy, dx  # perpendicular (left)
        hw = lane_width  # half road width = one lane per direction

        corners = [
            self._w2s(ax + px * hw, ay + py * hw),
            self._w2s(bx + px * hw, by + py * hw),
            self._w2s(bx - px * hw, by - py * hw),
            self._w2s(ax - px * hw, ay - py * hw),
        ]
        pygame.draw.polygon(self._screen, _COL_ROAD_SURFACE, corners)

    def _draw_lane_markings(
        self, ax: float, ay: float, bx: float, by: float, lane_width: float
    ) -> None:
        """Draw curb lines and dashed centre line for a road segment."""
        dx = bx - ax
        dy = by - ay
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.1:
            return
        dx /= length
        dy /= length
        px, py = -dy, dx
        hw = lane_width

        # Solid curb lines at road edges
        pygame.draw.line(
            self._screen,
            _COL_CURB,
            self._w2s(ax + px * hw, ay + py * hw),
            self._w2s(bx + px * hw, by + py * hw),
            1,
        )
        pygame.draw.line(
            self._screen,
            _COL_CURB,
            self._w2s(ax - px * hw, ay - py * hw),
            self._w2s(bx - px * hw, by - py * hw),
            1,
        )

        # Dashed yellow centre line
        t = 0.0
        drawing = True
        while t < length:
            seg = _DASH_LEN_M if drawing else _GAP_LEN_M
            t2 = min(t + seg, length)
            if drawing:
                p1 = self._w2s(ax + dx * t, ay + dy * t)
                p2 = self._w2s(ax + dx * t2, ay + dy * t2)
                pygame.draw.line(self._screen, _COL_CENTER_MARK, p1, p2, 1)
            t = t2
            drawing = not drawing

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

    def _draw_trajectory(self, trajectory: LocalTrajectory) -> None:
        for pt in trajectory.points:
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

    def _draw_hud(self, world: "SimulationWorld") -> None:
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
