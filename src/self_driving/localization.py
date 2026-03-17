"""Layer 2b — SLAM-lite localisation and object detection.

Adds Gaussian noise to ground-truth pose to simulate sensor drift,
clusters LiDAR hit points into detected objects, and snaps the
estimated pose to the nearest road edge.
"""

import math
import random

from shapely.geometry import LineString, Point

from self_driving.models import (
    ActorState,
    DetectedObject,
    LidarScan,
    LocalizationOutput,
    Pose,
    RoadEdge,
    RoadMap,
    Vector2,
    VehicleState,
)

# Sensor noise standard deviations
_POS_NOISE_STD = 0.3  # metres
_HEADING_NOISE_STD = 0.02  # radians
_CLUSTER_DIST_M = 3.0  # metres — max gap within a cluster


def localize(
    scan: LidarScan,
    ground_truth: VehicleState,
    road_map: RoadMap,
    rng: random.Random | None = None,
) -> LocalizationOutput:
    """Estimate ego pose and detect objects from a LiDAR scan.

    Parameters
    ----------
    scan:          LiDAR scan produced by sensors.simulate_lidar
    ground_truth:  True vehicle state (used as noisy base)
    road_map:      Road network for snapping / object context
    rng:           Optional seeded random instance for reproducibility
    """
    if rng is None:
        rng = random.Random()

    # ------------------------------------------------------------------
    # 1. Noisy pose estimate
    # ------------------------------------------------------------------
    noisy_x = ground_truth.pose.x + rng.gauss(0.0, _POS_NOISE_STD)
    noisy_y = ground_truth.pose.y + rng.gauss(0.0, _POS_NOISE_STD)
    noisy_h = ground_truth.pose.heading + rng.gauss(0.0, _HEADING_NOISE_STD)
    noisy_h = (noisy_h + math.pi) % (2 * math.pi) - math.pi

    estimated_pose = Pose(x=noisy_x, y=noisy_y, heading=noisy_h)

    # ------------------------------------------------------------------
    # 2. Detect objects from LiDAR hit clusters
    # ------------------------------------------------------------------
    hit_points: list[tuple[float, float]] = []
    for p in scan.points:
        if p.hit and p.distance < 40.0:
            hx = scan.ego_pose.x + p.distance * math.cos(p.angle)
            hy = scan.ego_pose.y + p.distance * math.sin(p.angle)
            hit_points.append((hx, hy))

    clusters = _cluster_points(hit_points, _CLUSTER_DIST_M)
    detected: list[DetectedObject] = []
    for idx, cluster in enumerate(clusters):
        cx = sum(p[0] for p in cluster) / len(cluster)
        cy = sum(p[1] for p in cluster) / len(cluster)
        detected.append(
            DetectedObject(
                object_id=idx,
                object_type="static",
                pose=Pose(x=cx, y=cy, heading=0.0),
                velocity=Vector2(x=0.0, y=0.0),
                confidence=min(1.0, len(cluster) / 5.0),
            )
        )

    # ------------------------------------------------------------------
    # 3. Snap to nearest road edge
    # ------------------------------------------------------------------
    nearest_edge = nearest_road_edge(road_map, noisy_x, noisy_y)

    return LocalizationOutput(
        timestamp=scan.timestamp,
        estimated_pose=estimated_pose,
        detected_objects=detected,
        nearest_road_edge=nearest_edge,
    )


def update_actor_detections(
    loc: LocalizationOutput,
    actors: list[ActorState],
) -> LocalizationOutput:
    """Replace generic static detections with typed actor information.

    In a real system this would be a proper object tracker; here we
    use ground-truth actor states to enrich the detections.
    """
    typed: list[DetectedObject] = []
    for actor in actors:
        typed.append(
            DetectedObject(
                object_id=actor.actor_id,
                object_type=actor.actor_type,
                pose=actor.pose,
                velocity=actor.velocity,
                confidence=1.0,
            )
        )
    return LocalizationOutput(
        timestamp=loc.timestamp,
        estimated_pose=loc.estimated_pose,
        detected_objects=typed,
        nearest_road_edge=loc.nearest_road_edge,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cluster_points(
    points: list[tuple[float, float]], max_gap: float
) -> list[list[tuple[float, float]]]:
    """Simple greedy distance-based clustering."""
    if not points:
        return []
    clusters: list[list[tuple[float, float]]] = []
    used = [False] * len(points)
    for i, pt in enumerate(points):
        if used[i]:
            continue
        cluster = [pt]
        used[i] = True
        for j in range(i + 1, len(points)):
            if used[j]:
                continue
            dx = points[j][0] - pt[0]
            dy = points[j][1] - pt[1]
            if math.sqrt(dx * dx + dy * dy) <= max_gap:
                cluster.append(points[j])
                used[j] = True
        if len(cluster) >= 2:
            clusters.append(cluster)
    return clusters


def nearest_road_edge(road_map: RoadMap, x: float, y: float) -> RoadEdge | None:
    """Return the road edge whose line is closest to (x, y)."""
    if not road_map.edges:
        return None
    node_by_id = {n.node_id: n.position for n in road_map.nodes}
    best_edge: RoadEdge | None = None
    best_dist = math.inf
    pt = Point(x, y)
    for edge in road_map.edges:
        a = node_by_id.get(edge.from_node)
        b = node_by_id.get(edge.to_node)
        if a is None or b is None:
            continue
        seg = LineString([(a.x, a.y), (b.x, b.y)])
        d = pt.distance(seg)
        if d < best_dist:
            best_dist = d
            best_edge = edge
    return best_edge
