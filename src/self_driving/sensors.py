"""Layer 2a — Simulated LiDAR sensor.

Casts rays from the ego vehicle position and finds the nearest
intersection with road boundaries or actor bounding boxes using Shapely.
"""

import math

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from self_driving.models import (
    ActorState,
    LidarConfig,
    LidarPoint,
    LidarScan,
    RoadMap,
    VehicleState,
)

# Half-dimensions of actor bounding boxes (metres)
_VEHICLE_HALF_W = 1.0
_VEHICLE_HALF_L = 2.2
_PEDESTRIAN_RADIUS = 0.4

DEFAULT_CONFIG = LidarConfig()


def _build_road_boundaries(road_map: RoadMap) -> MultiLineString:
    """Build road edge centre lines as Shapely geometry for ray intersection."""
    lines = []
    node_by_id = {n.node_id: n.position for n in road_map.nodes}
    for edge in road_map.edges:
        a = node_by_id[edge.from_node]
        b = node_by_id[edge.to_node]
        lines.append(LineString([(a.x, a.y), (b.x, b.y)]))
    return MultiLineString(lines)


def _actor_polygon(actor: ActorState) -> Polygon:
    """Return a rotated rectangle polygon for an actor."""
    if actor.actor_type == "pedestrian":
        return Point(actor.pose.x, actor.pose.y).buffer(_PEDESTRIAN_RADIUS)

    hw, hl = _VEHICLE_HALF_W, _VEHICLE_HALF_L
    corners = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
    cos_h = math.cos(actor.pose.heading)
    sin_h = math.sin(actor.pose.heading)
    rotated = [
        (
            actor.pose.x + x * cos_h - y * sin_h,
            actor.pose.y + x * sin_h + y * cos_h,
        )
        for x, y in corners
    ]
    return Polygon(rotated)


def simulate_lidar(
    ego: VehicleState,
    road_map: RoadMap,
    actors: list[ActorState],
    config: LidarConfig = DEFAULT_CONFIG,
) -> LidarScan:
    """Cast num_rays rays from the ego pose and return a LidarScan.

    Each ray originates at (ego.x, ego.y) and extends to max_range.
    The nearest intersection with road boundaries or actor geometry
    determines the reported distance.
    """
    angles = np.linspace(0.0, 2 * math.pi, config.num_rays, endpoint=False)
    road_geom = _build_road_boundaries(road_map)
    actor_polys = [_actor_polygon(a) for a in actors]

    origin = Point(ego.pose.x, ego.pose.y)
    points: list[LidarPoint] = []

    for angle in angles:
        end_x = ego.pose.x + config.max_range * math.cos(angle)
        end_y = ego.pose.y + config.max_range * math.sin(angle)
        ray = LineString([(ego.pose.x, ego.pose.y), (end_x, end_y)])

        min_dist = config.max_range
        hit = False

        # Check road boundary intersections
        road_hit = ray.intersection(road_geom)
        if not road_hit.is_empty:
            d = origin.distance(road_hit)
            if d < min_dist:
                min_dist = d
                hit = True

        # Check actor intersections
        for poly in actor_polys:
            actor_hit = ray.intersection(poly)
            if not actor_hit.is_empty:
                d = origin.distance(actor_hit)
                if d < min_dist:
                    min_dist = d
                    hit = True

        points.append(LidarPoint(angle=float(angle), distance=float(min_dist), hit=hit))

    return LidarScan(
        timestamp=ego.timestamp,
        ego_pose=ego.pose,
        points=points,
    )
