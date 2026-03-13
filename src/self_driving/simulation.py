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

import random
from dataclasses import dataclass, field

from self_driving.actors import update_actors
from self_driving.controller import MpcParams, VehicleParams, compute_mpc
from self_driving.localization import localize, update_actor_detections
from self_driving.models import (
    LidarConfig,
    LidarScan,
    LocalizationOutput,
    LocalTrajectory,
    Route,
    SimConfig,
    VehicleState,
)
from self_driving.path_planning import plan_path
from self_driving.routing import (
    find_nearest_node_from_pose,
    plan_route,
)
from self_driving.sensors import simulate_lidar
from self_driving.vehicle import integrate_bicycle_model
from self_driving.world import SimulationWorld


@dataclass
class SimulationLoop:
    """Runs the simulation and holds transient (non-serialisable) state."""

    world: SimulationWorld
    config: SimConfig
    lidar_config: LidarConfig = field(default_factory=LidarConfig)
    mpc_params: MpcParams = field(default_factory=MpcParams)
    vehicle_params: VehicleParams = field(default_factory=VehicleParams)
    _rng: random.Random = field(default_factory=random.Random, init=False)

    # Last outputs for visualiser access
    last_scan: LidarScan | None = field(default=None, init=False)
    last_loc: LocalizationOutput | None = field(default=None, init=False)
    last_trajectory: LocalTrajectory | None = field(default=None, init=False)
    _step_count: int = field(default=0, init=False)

    def step(self) -> None:
        """Advance the simulation by one time step."""
        dt = self.config.dt
        world = self.world

        # 1. Clock + actors
        world = world.advance_clock(dt)
        new_actors = update_actors(
            world.actor_states,
            world.road_map,
            world.graph,
            dt,
            self._rng,
        )
        world = world.with_actors(new_actors)

        # 2. Sense
        scan = simulate_lidar(
            world.ego_state,
            world.road_map,
            world.actor_states,
            self.lidar_config,
        )
        self.last_scan = scan

        # 3. Localise
        loc = localize(scan, world.ego_state, world.road_map, self._rng)
        loc = update_actor_detections(loc, world.actor_states)
        self.last_loc = loc

        # 4. Route (cached; re-plan only when needed)
        route = self._maybe_replan_route(world, loc)
        if route is not world.current_route:
            world = world.with_route(route)

        # 5. Path plan
        if route is not None:
            trajectory = plan_path(route, loc, world.road_map)
        else:
            from self_driving.path_planning import _emergency_stop_trajectory

            trajectory = _emergency_stop_trajectory(loc.estimated_pose, world.clock)
        self.last_trajectory = trajectory

        # 6. MPC control
        control = compute_mpc(
            world.ego_state, trajectory, self.mpc_params, self.vehicle_params
        )

        # 7. Vehicle physics
        new_ego: VehicleState = integrate_bicycle_model(
            world.ego_state, control, dt, self.vehicle_params
        )
        world = world.with_ego_state(new_ego)

        self.world = world
        self._step_count += 1

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

            if visualizer is not None and step % self.config.render_every_n == 0:
                running = visualizer.render(  # type: ignore[union-attr]
                    self.world,
                    self.last_scan,
                    self.last_trajectory,
                )
                if not running:
                    break

            step += 1

        if visualizer is not None:
            visualizer.close()  # type: ignore[union-attr]

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
