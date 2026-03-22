"""SimulationWorld: the central immutable state container.

Each update method returns a new world rather than mutating in place,
making the simulation loop easy to reason about and test.
"""

import networkx as nx
from pydantic import BaseModel, ConfigDict, model_validator

from self_driving.map_gen import build_graph, initial_traffic_light_states
from self_driving.models import (
    ActorState,
    RoadMap,
    Route,
    TrafficLightState,
    VehicleState,
)


class SimulationWorld(BaseModel):
    """Holds the complete state of the simulation at one point in time."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    road_map: RoadMap
    ego_state: VehicleState
    actor_states: list[ActorState]
    current_route: Route | None
    destination: int | None
    clock: float  # simulation seconds elapsed
    traffic_light_states: list[TrafficLightState] = []

    # nx.DiGraph is not Pydantic-serialisable; excluded from schema.
    # It is built once on first access via model_validator.
    _graph: nx.DiGraph | None = None

    @model_validator(mode="after")
    def _build_graph(self) -> "SimulationWorld":
        object.__setattr__(self, "_graph", build_graph(self.road_map))
        # Initialise traffic light states from static config if not provided
        if not self.traffic_light_states and self.road_map.traffic_lights:
            object.__setattr__(
                self,
                "traffic_light_states",
                initial_traffic_light_states(self.road_map.traffic_lights),
            )
        return self

    @property
    def graph(self) -> nx.DiGraph:
        """NetworkX DiGraph built from road_map."""
        if self._graph is None:  # should not happen after validator
            object.__setattr__(self, "_graph", build_graph(self.road_map))
        return self._graph  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Functional update helpers
    # ------------------------------------------------------------------

    def with_ego_state(self, state: VehicleState) -> "SimulationWorld":
        """Return world with updated ego vehicle state."""
        return self.model_copy(update={"ego_state": state})

    def with_route(self, route: Route | None) -> "SimulationWorld":
        """Return world with updated route."""
        return self.model_copy(update={"current_route": route})

    def with_actors(self, actors: list[ActorState]) -> "SimulationWorld":
        """Return world with updated actor list."""
        return self.model_copy(update={"actor_states": actors})

    def with_destination(self, dest: int | None) -> "SimulationWorld":
        """Return world with a new destination node id."""
        return self.model_copy(update={"destination": dest, "current_route": None})

    def with_traffic_light_states(
        self, states: list[TrafficLightState]
    ) -> "SimulationWorld":
        """Return world with updated traffic light runtime states."""
        return self.model_copy(update={"traffic_light_states": states})

    def advance_clock(self, dt: float) -> "SimulationWorld":
        """Return world with simulation clock advanced by dt seconds."""
        return self.model_copy(update={"clock": self.clock + dt})
