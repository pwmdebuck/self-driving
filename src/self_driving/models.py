"""Shared Pydantic data models — the single contract between all layers."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Vector2(BaseModel):
    """2D vector or position."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class Pose(BaseModel):
    """2D pose: position + heading in radians (0 = east, CCW positive)."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    heading: float


# ---------------------------------------------------------------------------
# Map / World
# ---------------------------------------------------------------------------


class RoadNode(BaseModel):
    """A node (intersection) in the road network."""

    model_config = ConfigDict(frozen=True)

    node_id: int
    position: Vector2


class RoadEdge(BaseModel):
    """A directed edge (road segment) between two nodes."""

    model_config = ConfigDict(frozen=True)

    from_node: int
    to_node: int
    length: float
    speed_limit: float  # m/s
    lane_width: float  # metres


class Building(BaseModel):
    """A static building footprint in the world."""

    model_config = ConfigDict(frozen=True)

    cx: float  # centre x in world metres
    cy: float  # centre y in world metres
    width: float  # metres along x-axis
    depth: float  # metres along y-axis


class RoadMap(BaseModel):
    """Serialisable road network (nodes + edges).

    The nx.DiGraph is built separately by world.py from these lists.
    """

    model_config = ConfigDict(frozen=True)

    nodes: list[RoadNode]
    edges: list[RoadEdge]
    buildings: list[Building] = []


# ---------------------------------------------------------------------------
# Layer 1 — Routing
# ---------------------------------------------------------------------------


class Route(BaseModel):
    """High-level route: ordered node IDs from start to destination."""

    model_config = ConfigDict(frozen=True)

    waypoint_ids: list[int]
    total_distance: float  # metres
    estimated_time: float  # seconds


# ---------------------------------------------------------------------------
# Layer 2 — Sensors / Localisation
# ---------------------------------------------------------------------------


class LidarPoint(BaseModel):
    """A single LiDAR ray result."""

    model_config = ConfigDict(frozen=True)

    angle: float  # radians relative to world frame
    distance: float  # metres to hit, or max_range if no hit
    hit: bool


class LidarScan(BaseModel):
    """Full 360-degree (or partial) LiDAR scan."""

    model_config = ConfigDict(frozen=True)

    timestamp: float
    ego_pose: Pose
    points: list[LidarPoint]


class DetectedObject(BaseModel):
    """An object perceived from sensor data."""

    model_config = ConfigDict(frozen=True)

    object_id: int
    object_type: Literal["vehicle", "pedestrian", "static"]
    pose: Pose
    velocity: Vector2
    confidence: float  # 0.0 – 1.0


class LocalizationOutput(BaseModel):
    """Layer 2 output: estimated ego pose + perceived scene objects."""

    model_config = ConfigDict(frozen=True)

    timestamp: float
    estimated_pose: Pose
    detected_objects: list[DetectedObject]
    nearest_road_edge: RoadEdge | None


# ---------------------------------------------------------------------------
# Layer 3 — Path Planning
# ---------------------------------------------------------------------------


class TrajectoryPoint(BaseModel):
    """A single point on the planned local trajectory."""

    model_config = ConfigDict(frozen=True)

    pose: Pose
    speed: float  # target speed in m/s
    curvature: float  # 1/radius (0 = straight)


class LocalTrajectory(BaseModel):
    """Short-horizon driveable path (~2 s, ~0.1 s spacing)."""

    model_config = ConfigDict(frozen=True)

    timestamp: float
    points: list[TrajectoryPoint]
    is_emergency_stop: bool


# ---------------------------------------------------------------------------
# Layer 4 — Vehicle / Control
# ---------------------------------------------------------------------------


class VehicleState(BaseModel):
    """Full kinematic state of the ego vehicle."""

    model_config = ConfigDict(frozen=True)

    pose: Pose
    speed: float  # longitudinal speed m/s
    acceleration: float  # m/s²
    steering_angle: float  # front wheel angle in radians
    timestamp: float


class ControlInput(BaseModel):
    """MPC output → vehicle actuators."""

    model_config = ConfigDict(frozen=True)

    steering_delta: float  # change to steering angle (radians)
    throttle: float  # 0.0 – 1.0
    brake: float  # 0.0 – 1.0


# ---------------------------------------------------------------------------
# Actors
# ---------------------------------------------------------------------------


class ActorState(BaseModel):
    """State of a non-ego actor (vehicle or pedestrian)."""

    model_config = ConfigDict(frozen=True)

    actor_id: int
    actor_type: Literal["vehicle", "pedestrian"]
    pose: Pose
    velocity: Vector2
    target_node: int | None  # final destination node
    next_node: int | None = None  # immediate next waypoint on the path


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class MapConfig(BaseModel):
    """Parameters for procedural map generation."""

    model_config = ConfigDict(frozen=True)

    grid_rows: int = 6
    grid_cols: int = 6
    block_size_m: float = 60.0  # metres between intersections
    speed_limit: float = 8.33  # m/s (~30 km/h)
    lane_width: float = 3.5  # metres


class LidarConfig(BaseModel):
    """LiDAR sensor configuration."""

    model_config = ConfigDict(frozen=True)

    num_rays: int = 180
    max_range: float = 50.0  # metres


class VehicleParams(BaseModel):
    """Kinematic bicycle model parameters."""

    model_config = ConfigDict(frozen=True)

    wheelbase: float = 2.7  # metres (l_f + l_r)
    lr_ratio: float = 0.5  # l_r / wheelbase
    max_steering_angle: float = 0.5  # radians (~28°)
    max_speed: float = 15.0  # m/s
    max_acceleration: float = 3.0  # m/s²
    max_deceleration: float = 6.0  # m/s²


class MpcParams(BaseModel):
    """MPC controller parameters."""

    model_config = ConfigDict(frozen=True)

    n_horizon: int = 8
    dt: float = 0.1  # seconds per horizon step
    weight_cte: float = 5.0  # cross-track error weight
    weight_heading: float = 2.0  # heading error weight
    weight_speed: float = 1.0  # speed error weight
    weight_effort: float = 0.5  # control effort weight


class SimConfig(BaseModel):
    """Top-level simulation configuration."""

    model_config = ConfigDict(frozen=True)

    dt: float = 0.05  # simulation time step (20 Hz)
    destination_node: int = 0
    render_every_n: int = 2  # render every N sim steps
    num_vehicle_actors: int = 4
    num_pedestrian_actors: int = 3
    max_steps: int | None = None
