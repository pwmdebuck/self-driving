"""Central simulation loop.

Orchestrates all four layers in order each tick:
  1. Advance clock + update actor positions
  2. Sense (LiDAR)
  3. Localise
  4. Route plan (cached)
  5. Path plan
  6. MPC control
  7. Vehicle physics integration
  8. Update world state
  9. Render (sub-frequency)
"""

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import mean

from self_driving.actors import update_actors
from self_driving.behavioral_planner import plan_behavior
from self_driving.controller import MpcParams, VehicleParams, compute_mpc
from self_driving.map_gen import update_traffic_lights
from self_driving.pure_pursuit import compute_pure_pursuit
from self_driving.localization import localize, nearest_road_edge, update_actor_detections
from self_driving.models import (
    BehaviorOutput,
    ControlInput,
    LidarConfig,
    LidarScan,
    LocalizationOutput,
    LocalTrajectory,
    PurePursuitParams,
    Route,
    SimConfig,
    VehicleState,
)
from self_driving.path_planning import emergency_stop_trajectory, plan_path
from self_driving.routing import (
    find_nearest_node_from_pose,
    plan_route,
)
from self_driving.sensors import simulate_lidar
from self_driving.vehicle import integrate_bicycle_model
from self_driving.world import SimulationWorld


@dataclass
class StepTimings:
    """Wall-clock time (ms) spent in each component of a single simulation step."""

    actors_ms: float
    localization_ms: float
    routing_ms: float
    behavior_ms: float
    path_planning_ms: float
    mpc_ms: float
    physics_ms: float
    total_ms: float


@dataclass
class TelemetryRecord:
    """One row of recorded telemetry per simulation step."""

    time: float
    x: float
    y: float
    steer: float
    accel_cmd: float
    timings: StepTimings


@dataclass
class SimulationLoop:
    """Runs the simulation and holds transient (non-serialisable) state."""

    world: SimulationWorld
    config: SimConfig
    lidar_config: LidarConfig = field(default_factory=LidarConfig)
    use_sensors: bool = False
    mpc_params: MpcParams = field(default_factory=MpcParams)
    vehicle_params: VehicleParams = field(default_factory=VehicleParams)
    pp_params: PurePursuitParams = field(default_factory=PurePursuitParams)
    _rng: random.Random = field(default_factory=random.Random, init=False)

    # Last outputs for visualiser access
    last_scan: LidarScan | None = field(default=None, init=False)
    last_loc: LocalizationOutput | None = field(default=None, init=False)
    last_trajectory: LocalTrajectory | None = field(default=None, init=False)
    last_control: "ControlInput | None" = field(default=None, init=False)
    _step_count: int = field(default=0, init=False)
    last_behavior: BehaviorOutput | None = field(default=None, init=False)
    telemetry: deque[TelemetryRecord] = field(
        default_factory=lambda: deque(maxlen=5000), init=False
    )

    def step(self) -> None:
        """Advance the simulation by one time step."""
        dt = self.config.dt
        world = self.world
        t_step = time.perf_counter()

        # 1. Clock + actors + traffic lights
        t0 = time.perf_counter()
        world = world.advance_clock(dt)
        if world.road_map.traffic_lights:
            new_tl_states = update_traffic_lights(
                world.traffic_light_states, world.road_map.traffic_lights, dt
            )
            world = world.with_traffic_light_states(new_tl_states)
        new_actors = update_actors(
            world.actor_states,
            world.road_map,
            world.graph,
            dt,
            self._rng,
        )
        world = world.with_actors(new_actors)
        actors_ms = (time.perf_counter() - t0) * 1000

        # 2. Sense + 3. Localise
        t0 = time.perf_counter()
        if self.use_sensors:
            scan = simulate_lidar(
                world.ego_state,
                world.road_map,
                world.actor_states,
                self.lidar_config,
            )
            self.last_scan = scan
            loc = localize(scan, world.ego_state, world.road_map, self._rng)
            loc = update_actor_detections(loc, world.actor_states)
        else:
            self.last_scan = None
            loc = LocalizationOutput(
                timestamp=world.clock,
                estimated_pose=world.ego_state.pose,
                detected_objects=[],
                nearest_road_edge=nearest_road_edge(
                    world.road_map,
                    world.ego_state.pose.x,
                    world.ego_state.pose.y,
                ),
            )
        self.last_loc = loc
        localization_ms = (time.perf_counter() - t0) * 1000

        # 4. Route (cached; re-plan only when needed)
        t0 = time.perf_counter()
        route = self._maybe_replan_route(world, loc)
        if route is not world.current_route:
            world = world.with_route(route)
        routing_ms = (time.perf_counter() - t0) * 1000

        # 4b. Behavioural planning
        t0 = time.perf_counter()
        if route is not None:
            behavior = plan_behavior(
                loc, route, world.road_map, self.last_behavior,
                world.traffic_light_states,
            )
            self.last_behavior = behavior
        else:
            behavior = None
        behavior_ms = (time.perf_counter() - t0) * 1000

        # 5. Path plan
        t0 = time.perf_counter()
        if route is not None:
            trajectory = plan_path(
                route,
                loc,
                world.road_map,
                behavior,
                ego_state=world.ego_state,
                vehicle_params=self.vehicle_params,
            )
        else:
            trajectory = emergency_stop_trajectory(loc.estimated_pose, world.clock)
        self.last_trajectory = trajectory
        path_planning_ms = (time.perf_counter() - t0) * 1000

        # 6. Control
        t0 = time.perf_counter()
        if self.config.controller_type == "pure_pursuit":
            control = compute_pure_pursuit(
                world.ego_state, trajectory, self.pp_params, self.vehicle_params
            )
        else:
            control = compute_mpc(
                world.ego_state,
                trajectory,
                self.mpc_params,
                self.vehicle_params,
                self.last_control,
            )
        self.last_control = control
        mpc_ms = (time.perf_counter() - t0) * 1000

        # 7. Vehicle physics
        t0 = time.perf_counter()
        new_ego: VehicleState = integrate_bicycle_model(
            world.ego_state, control, dt, self.vehicle_params
        )
        world = world.with_ego_state(new_ego)
        physics_ms = (time.perf_counter() - t0) * 1000

        self.world = world
        self._step_count += 1
        total_ms = (time.perf_counter() - t_step) * 1000

        # Record telemetry
        self.telemetry.append(
            TelemetryRecord(
                time=world.clock,
                x=new_ego.pose.x,
                y=new_ego.pose.y,
                steer=control.steering_delta,
                accel_cmd=control.accel_cmd,
                timings=StepTimings(
                    actors_ms=actors_ms,
                    localization_ms=localization_ms,
                    routing_ms=routing_ms,
                    behavior_ms=behavior_ms,
                    path_planning_ms=path_planning_ms,
                    mpc_ms=mpc_ms,
                    physics_ms=physics_ms,
                    total_ms=total_ms,
                ),
            )
        )

    def run(
        self,
        visualizer: object | None = None,
        max_steps: int | None = None,
    ) -> None:
        """Run the simulation loop until max_steps or window is closed.

        Parameters
        ----------
        visualizer: optional Visualizer instance for rendering
        max_steps:  stop after this many steps (None = run forever)
        """
        steps = max_steps or self.config.max_steps
        step = 0
        while True:
            if steps is not None and step >= steps:
                break

            self.step()

            if step % 20 == 0:
                _print_status(self.world.ego_state, self.last_behavior, step)

            if visualizer is not None and step % self.config.render_every_n == 0:
                last_timings = self.telemetry[-1].timings if self.telemetry else None
                running = visualizer.render(  # type: ignore[union-attr]
                    self.world,
                    self.last_scan,
                    self.last_trajectory,
                    self.last_control,
                    last_timings,
                )
                if not running:
                    break

            step += 1

        if visualizer is not None:
            visualizer.close()  # type: ignore[union-attr]

    def timing_summary(self) -> None:
        """Print a per-component timing table (mean / p95 / max) over all recorded steps."""
        if not self.telemetry:
            print("No telemetry recorded — run step() first.")
            return

        fields = [
            ("actors", "actors_ms"),
            ("localization", "localization_ms"),
            ("routing", "routing_ms"),
            ("behavior", "behavior_ms"),
            ("path_planning", "path_planning_ms"),
            ("mpc", "mpc_ms"),
            ("physics", "physics_ms"),
            ("total", "total_ms"),
        ]

        def p95(values: list[float]) -> float:
            sorted_vals = sorted(values)
            idx = int(len(sorted_vals) * 0.95)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

        header = f"{'Component':<16} {'mean(ms)':>9} {'p95(ms)':>9} {'max(ms)':>9}"
        separator = "─" * len(header)
        print(separator)
        print(header)
        print(separator)
        for label, attr in fields:
            values = [getattr(r.timings, attr) for r in self.telemetry]
            row_mean = mean(values)
            row_p95 = p95(values)
            row_max = max(values)
            if label == "total":
                print(separator)
            print(f"{label:<16} {row_mean:>9.1f} {row_p95:>9.1f} {row_max:>9.1f}")
        print(separator)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_replan_route(
        self, world: SimulationWorld, loc: LocalizationOutput
    ) -> Route | None:
        """Return current route or replan if destination changed / no route."""
        dest = world.destination
        if dest is None:
            return None

        if world.current_route is not None:
            # Check if we have reached the destination
            route = world.current_route
            if route.waypoint_ids and route.waypoint_ids[-1] == dest:
                # Route destination matches — keep existing route
                return route

        # (Re-)plan
        start_node = find_nearest_node_from_pose(world.road_map, loc.estimated_pose)
        try:
            return plan_route(world.graph, world.road_map, start_node, dest)
        except Exception:
            return world.current_route


def _print_status(ego: VehicleState, behavior: BehaviorOutput | None, step: int) -> None:
    """Print a one-line debug summary to stdout."""
    t = ego.timestamp
    hdg = math.degrees(ego.pose.heading)
    yr = ego.yaw_rate
    if behavior is not None:
        bstate = behavior.state.value
        lane = behavior.target_lane
        tspd = behavior.target_speed
        lc = f"  lc_end=({behavior.lane_change_end_pos.x:.0f},{behavior.lane_change_end_pos.y:.0f})" if behavior.lane_change_end_pos else ""
    else:
        bstate, lane, tspd, lc = "NO_BEHAVIOR", "-", 0.0, ""
    print(
        f"t={t:6.1f}s  step={step:4d}  "
        f"pos=({ego.pose.x:7.1f},{ego.pose.y:7.1f})  "
        f"v={ego.speed:5.2f}m/s  hdg={hdg:6.1f}°  yr={yr:+.3f}r/s  "
        f"{bstate}  lane={lane}  tspd={tspd:.2f}{lc}"
    )
