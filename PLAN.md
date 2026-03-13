# Self-Driving Car Simulation Sandbox

## Context

Build an end-to-end self-driving car simulation sandbox in Python from scratch. The goal is a runnable 2D simulation that implements all 4 classical autonomous driving layers: routing, localization/sensing, path planning, and vehicle control. The layers are separate modules communicating through typed Pydantic interfaces, with a central simulation loop orchestrating them. Intended as an educational/research sandbox — not production code.

---

## Architecture

### File Structure

```
self-driving/
├── pyproject.toml
├── main.py                          # Entry point
├── src/
│   └── self_driving/
│       ├── __init__.py
│       ├── models.py                # ALL Pydantic data models (shared contract)
│       ├── world.py                 # SimulationWorld: global state container
│       ├── map_gen.py               # Procedural road network generation
│       ├── routing.py               # Layer 1: A*/Dijkstra on road graph
│       ├── sensors.py               # Layer 2a: LiDAR raycasting simulation
│       ├── localization.py          # Layer 2b: SLAM-lite, object detection
│       ├── path_planning.py         # Layer 3: Local trajectory generation
│       ├── vehicle.py               # Layer 4a: Kinematic bicycle model
│       ├── controller.py            # Layer 4b: MPC controller (scipy.optimize)
│       ├── actors.py                # Other vehicle + pedestrian behavioral models
│       ├── simulation.py            # Central simulation loop orchestrator
│       └── visualizer.py            # Pygame 2D top-down visualization
└── tests/
    ├── conftest.py
    ├── test_map_gen.py
    ├── test_routing.py
    ├── test_sensors.py
    ├── test_localization.py
    ├── test_path_planning.py
    ├── test_vehicle.py
    ├── test_controller.py
    └── test_actors.py
```

**Key rule**: No layer module imports from another layer module. All cross-layer communication is via `models.py` types only. `simulation.py` is the sole orchestrator.

---

## Dependencies (`pyproject.toml`)

```toml
[project]
name = "self-driving"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7",
    "networkx>=3.3",
    "numpy>=1.26",
    "scipy>=1.13",
    "shapely>=2.0",
    "pygame>=2.6",
]

[tool.uv.dev-dependencies]
dev = [
    "pytest>=8.2",
    "anyio[trio]>=4.4",
    "pytest-anyio",
    "ruff>=0.4",
    "pyrefly>=0.15",
]
```

- **networkx**: road graph + built-in A* (`nx.astar_path`) and Dijkstra
- **numpy**: raycasting math, bicycle model, MPC cost function
- **scipy**: `scipy.optimize.minimize` with SLSQP for MPC
- **shapely**: ray-geometry intersections, obstacle buffering, road geometry
- **pygame**: real-time 2D top-down visualization

---

## Core Data Models (`models.py`)

All frozen Pydantic models. This is the single shared contract between layers.

```python
# Primitives
class Vector2(BaseModel): x: float; y: float
class Pose(BaseModel): x: float; y: float; heading: float  # radians

# Map
class RoadNode(BaseModel): node_id: int; position: Vector2
class RoadEdge(BaseModel): from_node: int; to_node: int; length: float; speed_limit: float; lane_width: float
class RoadMap(BaseModel): nodes: list[RoadNode]; edges: list[RoadEdge]

# Layer 1 output
class Route(BaseModel): waypoint_ids: list[int]; total_distance: float; estimated_time: float

# Layer 2 output
class LidarPoint(BaseModel): angle: float; distance: float; hit: bool
class LidarScan(BaseModel): timestamp: float; ego_pose: Pose; points: list[LidarPoint]
class DetectedObject(BaseModel): object_id: int; object_type: Literal["vehicle","pedestrian","static"]; pose: Pose; velocity: Vector2; confidence: float
class LocalizationOutput(BaseModel): timestamp: float; estimated_pose: Pose; detected_objects: list[DetectedObject]; nearest_road_edge: RoadEdge | None

# Layer 3 output
class TrajectoryPoint(BaseModel): pose: Pose; speed: float; curvature: float
class LocalTrajectory(BaseModel): timestamp: float; points: list[TrajectoryPoint]; is_emergency_stop: bool

# Layer 4
class VehicleState(BaseModel): pose: Pose; speed: float; acceleration: float; steering_angle: float; timestamp: float
class ControlInput(BaseModel): steering_delta: float; throttle: float; brake: float

# Actors
class ActorState(BaseModel): actor_id: int; actor_type: Literal["vehicle","pedestrian"]; pose: Pose; velocity: Vector2; target_node: int | None

# Config models
class MapConfig(BaseModel): grid_rows: int; grid_cols: int; block_size_m: float; speed_limit: float
class LidarConfig(BaseModel): num_rays: int = 360; max_range: float = 50.0
class VehicleParams(BaseModel): wheelbase: float; max_steering_angle: float; max_acceleration: float
class MpcParams(BaseModel): n_horizon: int = 10; dt: float = 0.05; weight_cte: float; weight_speed: float; weight_effort: float
class SimConfig(BaseModel): dt: float = 0.05; destination_node: int; render_every_n: int = 2
```

---

## Layer Specifications

### `world.py` — `SimulationWorld`
Functional state container. Each update returns a new world (immutable pattern).
- Fields: `road_map`, `ego_state`, `actor_states`, `current_route`, `destination`, `clock`
- Methods: `with_ego_state()`, `with_route()`, `with_actors()`, `advance_clock(dt)`
- Caches and exposes `nx.DiGraph` built from `RoadMap` on first access (not serialized)

### `map_gen.py` — `generate_road_map(config: MapConfig) -> tuple[RoadMap, nx.DiGraph]`
- Grid intersections → `RoadNode`s with small random position noise
- Horizontal/vertical connections → `RoadEdge`s with computed lengths
- Returns both the serializable `RoadMap` and the `nx.DiGraph` for routing

### `routing.py` — `plan_route(graph, start_node, dest_node) -> Route`
- `nx.astar_path` with Euclidean heuristic; fallback to `nx.dijkstra_path`
- `find_nearest_node(road_map, pos) -> int` — snaps position to closest graph node
- Returns `Route` with ordered `waypoint_ids`, `total_distance`, `estimated_time`

### `sensors.py` — `simulate_lidar(ego, road_map, actors, config) -> LidarScan`
- Cast N rays from ego position using Shapely `LineString.intersection()`
- Check against: road boundary segments + actor bounding box `Polygon`s
- Return nearest hit distance per ray; `max_range` if no intersection

### `localization.py` — `localize(scan, ground_truth, road_map) -> LocalizationOutput`
- Add calibrated Gaussian noise to ground truth pose (simulate sensor drift)
- Cluster nearby LiDAR hits by distance grouping → `DetectedObject` list
- Snap estimated pose to nearest road edge via Shapely nearest-points

### `path_planning.py` — `plan_path(route, loc, road_map) -> LocalTrajectory`
- Extract next 20 waypoints ahead of estimated position from `Route`
- Buffer each detected obstacle with Shapely and shift waypoints laterally
- Compute target speed per waypoint based on curvature + obstacle proximity
- Set `is_emergency_stop=True` if obstacle within braking distance

### `vehicle.py` — `integrate_bicycle_model(state, control, dt, params) -> VehicleState`
Kinematic bicycle model:
```
beta = arctan(l_r / (l_f + l_r) * tan(delta))
x += v * cos(psi + beta) * dt
y += v * sin(psi + beta) * dt
psi += (v / l_r) * sin(beta) * dt
v += a * dt
```
Pure function, no side effects. Clamps steering and speed to physical limits.

### `controller.py` — `compute_mpc(state, trajectory, params) -> ControlInput`
- Cost function over `n_horizon` steps: cross-track error + heading error + speed error + control effort
- `scipy.optimize.minimize(cost_fn, x0, method='SLSQP', bounds=...)`
- `x0` = flattened `(steering_delta, throttle, brake)` sequence over horizon
- Return only first control action (receding horizon principle)

### `actors.py` — `update_actors(world, dt) -> list[ActorState]`
- Vehicle actors: follow road graph via shortest path, stop if 5m ahead is occupied
- Pedestrian actors: random walk between nearby road nodes
- Calls `find_nearest_node()` from `routing.py` to replan when waypoint reached

### `simulation.py` — `SimulationLoop`
Orchestrates all layers in order per tick:
1. `world.advance_clock(dt)` + `actors.update_actors()`
2. `sensors.simulate_lidar()` → `LidarScan`
3. `localization.localize()` → `LocalizationOutput`
4. `routing.plan_route()` (cached, only re-runs on destination change) → `Route`
5. `path_planning.plan_path()` → `LocalTrajectory`
6. `controller.compute_mpc()` → `ControlInput`
7. `vehicle.integrate_bicycle_model()` → new `VehicleState`
8. `world.with_ego_state(new_state)` → updated world
9. `visualizer.render()` every `render_every_n` ticks

### `visualizer.py` — `Visualizer`
Pygame top-down view:
- Road edges (gray), road nodes (dots), actors (blue=vehicle, green=pedestrian)
- Ego vehicle (red rectangle + heading arrow)
- LiDAR rays (cyan, semi-transparent), planned trajectory (yellow dots)
- Active route overlay (magenta), HUD: speed/steering/clock

---

## Simulation Loop Flow

```
map_gen ──────> RoadMap + nx.DiGraph ──────> SimulationWorld
                                                    │
                              ┌─────────────── TICK LOOP ──────────────────┐
                              │                                             │
                              │  sensors   ← ego_state, actors, road_map  │
                              │     ↓ LidarScan                            │
                              │  localize  ← road_map                      │
                              │     ↓ LocalizationOutput                   │
                              │  routing   ← destination (cached)          │
                              │     ↓ Route                                │
                              │  path_plan ← road_map                      │
                              │     ↓ LocalTrajectory                      │
                              │  controller← ego_state                     │
                              │     ↓ ControlInput                         │
                              │  vehicle   ← dt                            │
                              │     ↓ VehicleState (new)                   │
                              │  world.update()                             │
                              └─────────────────────────────────────────────┘
                                          │
                                    visualizer (sub-frequency)
```

---

## MVP Implementation Order

1. **`models.py`** — define all data models first (everything depends on this)
2. **`map_gen.py`** — generate a simple 4x4 grid `RoadMap`
3. **`world.py`** — `SimulationWorld` with functional update methods
4. **`visualizer.py`** — basic Pygame road map rendering
5. **`vehicle.py`** — bicycle model (pure math, easy to test)
6. **`routing.py`** — A* route planning
7. **`sensors.py`** — LiDAR raycasting (start with 36 rays)
8. **`localization.py`** — noisy pose + object clustering
9. **`path_planning.py`** — waypoint extraction (no obstacle avoidance yet)
10. **`controller.py`** — MPC (start with cross-track + heading error only)
11. **`simulation.py`** — full orchestration loop
12. **`actors.py`** — other vehicles following road graph
13. **`main.py`** — entry point wiring everything together
14. **Tests** — one test file per module

---

## Verification

- `uv run python main.py` should open a Pygame window showing: a grid road map, ego vehicle moving toward destination, LiDAR rays, planned trajectory overlay
- `uv run pytest` should pass all unit tests
- Manually verify: ego vehicle follows A* route, slows for obstacles, MPC keeps it on path
- `uv run ruff check .` + `uv run ruff format .` pass clean
- `uv run pyrefly check` passes with no errors
