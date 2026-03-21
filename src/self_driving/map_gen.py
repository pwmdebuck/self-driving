"""Procedural road network generation.

Generates a grid-based road map where intersections become nodes and
horizontal/vertical connections become directed edges. Supports multiple
lanes per direction with automatic lane connection generation at intersections.
"""

import math
import random

import networkx as nx

from self_driving.models import (
    Building,
    LaneConnection,
    MapConfig,
    RoadEdge,
    RoadMap,
    RoadNode,
    RoadSign,
    SignType,
    TrafficLight,
    TrafficLightPhase,
    TrafficLightState,
    Vector2,
)


def generate_road_map(config: MapConfig, seed: int = 42) -> tuple[RoadMap, nx.DiGraph]:
    """Generate a grid road network from config.

    Returns both the serialisable RoadMap and the nx.DiGraph for routing.
    """
    rng = random.Random(seed)
    nodes: list[RoadNode] = []
    edges: list[RoadEdge] = []

    noise = config.block_size_m * 0.05  # 5 % position noise

    def node_id(row: int, col: int) -> int:
        return row * config.grid_cols + col

    # Create nodes at grid intersections with slight position noise
    for row in range(config.grid_rows):
        for col in range(config.grid_cols):
            x = col * config.block_size_m + rng.uniform(-noise, noise)
            y = row * config.block_size_m + rng.uniform(-noise, noise)
            nodes.append(
                RoadNode(
                    node_id=node_id(row, col),
                    position=Vector2(x=x, y=y),
                )
            )

    node_by_id = {n.node_id: n for n in nodes}

    def add_edge(from_id: int, to_id: int) -> None:
        a = node_by_id[from_id].position
        b = node_by_id[to_id].position
        length = math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
        edges.append(
            RoadEdge(
                from_node=from_id,
                to_node=to_id,
                length=length,
                speed_limit=config.speed_limit,
                lane_width=config.lane_width,
                num_lanes=config.default_num_lanes,
            )
        )

    # Horizontal edges (bidirectional)
    for row in range(config.grid_rows):
        for col in range(config.grid_cols - 1):
            add_edge(node_id(row, col), node_id(row, col + 1))
            add_edge(node_id(row, col + 1), node_id(row, col))

    # Vertical edges (bidirectional)
    for row in range(config.grid_rows - 1):
        for col in range(config.grid_cols):
            add_edge(node_id(row, col), node_id(row + 1, col))
            add_edge(node_id(row + 1, col), node_id(row, col))

    buildings = _generate_buildings(nodes, config, rng)
    lane_connections = _generate_lane_connections(nodes, edges)
    road_map = RoadMap(
        nodes=nodes,
        edges=edges,
        buildings=buildings,
        lane_connections=lane_connections,
    )
    graph = build_graph(road_map)
    return road_map, graph


def _generate_lane_connections(
    nodes: list[RoadNode],
    edges: list[RoadEdge],
) -> list[LaneConnection]:
    """Generate permitted lane-to-lane movements at every intersection.

    For each (incoming, outgoing) edge pair at a node the turn angle
    determines which lanes connect:
      - Right turn  (≈ -90°): only curbside lane (N-1) → curbside lane (N-1)
      - Straight    (≈   0°): lane i → lane i for all shared lanes
      - Left turn   (≈ +90°): only innermost lane (0) → innermost lane (0)
      - U-turns are skipped.
    """
    node_by_id = {n.node_id: n.position for n in nodes}

    incoming: dict[int, list[RoadEdge]] = {}
    outgoing: dict[int, list[RoadEdge]] = {}
    for edge in edges:
        incoming.setdefault(edge.to_node, []).append(edge)
        outgoing.setdefault(edge.from_node, []).append(edge)

    connections: list[LaneConnection] = []

    for node in nodes:
        nid = node.node_id
        for inc in incoming.get(nid, []):
            a = node_by_id[inc.from_node]
            b = node_by_id[inc.to_node]
            inc_heading = math.atan2(b.y - a.y, b.x - a.x)

            for out in outgoing.get(nid, []):
                if out.to_node == inc.from_node:
                    continue  # skip U-turns

                c = node_by_id[out.to_node]
                out_heading = math.atan2(c.y - b.y, c.x - b.x)
                turn = _normalise_angle(out_heading - inc_heading)

                from_edge = (inc.from_node, inc.to_node)
                to_edge = (out.from_node, out.to_node)

                if turn < -math.pi / 4:  # right turn — curbside lane only
                    connections.append(
                        LaneConnection(
                            node_id=nid,
                            from_edge=from_edge,
                            from_lane=inc.num_lanes - 1,
                            to_edge=to_edge,
                            to_lane=out.num_lanes - 1,
                        )
                    )
                elif turn > math.pi / 4:  # left turn — innermost lane only
                    connections.append(
                        LaneConnection(
                            node_id=nid,
                            from_edge=from_edge,
                            from_lane=0,
                            to_edge=to_edge,
                            to_lane=0,
                        )
                    )
                else:  # straight — connect lanes by index
                    for lane in range(min(inc.num_lanes, out.num_lanes)):
                        connections.append(
                            LaneConnection(
                                node_id=nid,
                                from_edge=from_edge,
                                from_lane=lane,
                                to_edge=to_edge,
                                to_lane=lane,
                            )
                        )

    return connections


def _normalise_angle(angle: float) -> float:
    """Normalise angle to the range (-π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle


def _generate_buildings(
    nodes: list[RoadNode], config: MapConfig, rng: random.Random
) -> list[Building]:
    """Place buildings in city blocks between intersections."""
    road_half_width = config.lane_width * config.default_num_lanes
    setback = road_half_width + 3.0  # road half-width + sidewalk clearance
    buildings: list[Building] = []

    node_pos = {n.node_id: n.position for n in nodes}

    def nid(row: int, col: int) -> int:
        return row * config.grid_cols + col

    for row in range(config.grid_rows - 1):
        for col in range(config.grid_cols - 1):
            sw = node_pos[nid(row, col)]
            se = node_pos[nid(row, col + 1)]
            ne = node_pos[nid(row + 1, col + 1)]
            nw = node_pos[nid(row + 1, col)]

            # Usable zone inside the block after road setback
            min_x = max(sw.x, nw.x) + setback
            max_x = min(se.x, ne.x) - setback
            min_y = max(sw.y, se.y) + setback
            max_y = min(nw.y, ne.y) - setback

            zone_w = max_x - min_x
            zone_h = max_y - min_y
            if zone_w < 8.0 or zone_h < 8.0:
                continue

            # Place buildings in a 2×2 sub-grid within the block
            for br in range(2):
                for bc in range(2):
                    if rng.random() < 0.15:  # occasionally leave a cell empty
                        continue
                    cell_w = zone_w / 2
                    cell_h = zone_h / 2
                    cell_min_x = min_x + bc * cell_w
                    cell_min_y = min_y + br * cell_h

                    bw = rng.uniform(cell_w * 0.45, min(cell_w * 0.85, 20.0))
                    bd = rng.uniform(cell_h * 0.45, min(cell_h * 0.85, 20.0))
                    jitter_x = rng.uniform(-cell_w * 0.08, cell_w * 0.08)
                    jitter_y = rng.uniform(-cell_h * 0.08, cell_h * 0.08)
                    cx = cell_min_x + cell_w / 2 + jitter_x
                    cy = cell_min_y + cell_h / 2 + jitter_y

                    buildings.append(Building(cx=cx, cy=cy, width=bw, depth=bd))

    return buildings


def build_graph(road_map: RoadMap) -> nx.DiGraph:
    """Build a NetworkX DiGraph from a RoadMap.

    Edge weight is travel time (length / speed_limit) so A* naturally
    prefers highways over slower urban roads of equal distance.
    """
    graph: nx.DiGraph = nx.DiGraph()
    for node in road_map.nodes:
        graph.add_node(node.node_id, x=node.position.x, y=node.position.y)
    for edge in road_map.edges:
        graph.add_edge(
            edge.from_node,
            edge.to_node,
            weight=edge.length / max(edge.speed_limit, 0.1),
            speed_limit=edge.speed_limit,
            edge_type=edge.edge_type,
        )
    return graph


# ---------------------------------------------------------------------------
# Road geometry utilities
# ---------------------------------------------------------------------------


def edge_polyline(edge: RoadEdge, node_by_id: dict[int, Vector2]) -> list[Vector2]:
    """Return the full ordered point list for an edge.

    ``[from_node_pos] + control_points + [to_node_pos]``
    An edge with no control points is a straight line between its two nodes.
    """
    return [node_by_id[edge.from_node]] + list(edge.control_points) + [node_by_id[edge.to_node]]


def edge_arc_length(pts: list[Vector2]) -> float:
    """Cumulative Euclidean length along a polyline."""
    total = 0.0
    for i in range(len(pts) - 1):
        dx = pts[i + 1].x - pts[i].x
        dy = pts[i + 1].y - pts[i].y
        total += math.sqrt(dx * dx + dy * dy)
    return total


# ---------------------------------------------------------------------------
# Curved road factory
# ---------------------------------------------------------------------------


def make_curved_edge(
    from_node: RoadNode,
    to_node: RoadNode,
    bulge: float,
    speed_limit: float,
    lane_width: float,
    num_lanes: int = 2,
    edge_type: str = "urban",
) -> RoadEdge:
    """Create a RoadEdge that curves between two nodes.

    ``bulge`` is the perpendicular offset (metres) at the chord midpoint.
    Two control points at 1/3 and 2/3 arc positions approximate the arc.
    """
    ax, ay = from_node.position.x, from_node.position.y
    bx, by = to_node.position.x, to_node.position.y
    # Chord midpoint
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    # Chord direction (unit vector)
    cdx, cdy = bx - ax, by - ay
    chord_len = math.sqrt(cdx * cdx + cdy * cdy)
    if chord_len < 1e-9:
        perp_x, perp_y = 0.0, 1.0
    else:
        # Perpendicular (left of chord direction in Y-up frame)
        perp_x, perp_y = -cdy / chord_len, cdx / chord_len

    # Peak of arc
    peak_x = mx + perp_x * bulge
    peak_y = my + perp_y * bulge

    # Control points at 1/3 and 2/3 along the arc (linear interp from endpoints to peak)
    cp1 = Vector2(
        x=ax + (peak_x - ax) * 2.0 / 3.0,
        y=ay + (peak_y - ay) * 2.0 / 3.0,
    )
    cp2 = Vector2(
        x=bx + (peak_x - bx) * 2.0 / 3.0,
        y=by + (peak_y - by) * 2.0 / 3.0,
    )

    pts = [from_node.position, cp1, cp2, to_node.position]
    length = edge_arc_length(pts)

    return RoadEdge(
        from_node=from_node.node_id,
        to_node=to_node.node_id,
        length=length,
        speed_limit=speed_limit,
        lane_width=lane_width,
        num_lanes=num_lanes,
        control_points=[cp1, cp2],
        edge_type=edge_type,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Highway / ramp factories
# ---------------------------------------------------------------------------


def build_highway_segment(
    from_node: RoadNode,
    to_node: RoadNode,
    num_lanes: int = 3,
    speed_limit: float = 27.8,  # ~100 km/h
    lane_width: float = 3.5,
) -> RoadEdge:
    """Create a straight highway edge between two existing nodes."""
    a, b = from_node.position, to_node.position
    length = math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
    return RoadEdge(
        from_node=from_node.node_id,
        to_node=to_node.node_id,
        length=length,
        speed_limit=speed_limit,
        lane_width=lane_width,
        num_lanes=num_lanes,
        edge_type="highway",
    )


def build_ramp_merge(
    highway_a: RoadNode,
    highway_b: RoadNode,
    ramp_origin: RoadNode,
    merge_node_id: int,
    highway_num_lanes: int = 3,
    ramp_speed_limit: float = 13.9,  # ~50 km/h
    highway_speed_limit: float = 27.8,
    lane_width: float = 3.5,
) -> tuple[list[RoadNode], list[RoadEdge], list[LaneConnection]]:
    """Build a merge ramp topology: ramp_origin → merge_node → highway_b.

    Inserts a merge node between highway_a and highway_b.  The ramp_origin
    connects into the merge node.  Lane connections at the merge node route:
      - highway through-lanes: lane i → lane i
      - ramp lane 0 → curbside highway lane (highway_num_lanes - 1)

    Returns the new nodes, edges (including updated highway segments), and
    the lane connections at the merge node.
    """
    # Place merge node 2/3 of the way along highway_a → highway_b
    ax, ay = highway_a.position.x, highway_a.position.y
    bx, by = highway_b.position.x, highway_b.position.y
    mx = ax + (bx - ax) * 2.0 / 3.0
    my = ay + (by - ay) * 2.0 / 3.0
    merge_node = RoadNode(node_id=merge_node_id, position=Vector2(x=mx, y=my))

    # Highway segments: A → merge and merge → B
    def straight_highway(fn: RoadNode, tn: RoadNode) -> RoadEdge:
        a2, b2 = fn.position, tn.position
        length = math.sqrt((b2.x - a2.x) ** 2 + (b2.y - a2.y) ** 2)
        return RoadEdge(
            from_node=fn.node_id,
            to_node=tn.node_id,
            length=length,
            speed_limit=highway_speed_limit,
            lane_width=lane_width,
            num_lanes=highway_num_lanes,
            edge_type="highway",
        )

    seg_a_merge = straight_highway(highway_a, merge_node)
    seg_merge_b = straight_highway(merge_node, highway_b)

    # Ramp segment: ramp_origin → merge_node
    rx, ry = ramp_origin.position.x, ramp_origin.position.y
    ramp_length = math.sqrt((mx - rx) ** 2 + (my - ry) ** 2)
    ramp_edge = RoadEdge(
        from_node=ramp_origin.node_id,
        to_node=merge_node.node_id,
        length=ramp_length,
        speed_limit=ramp_speed_limit,
        lane_width=lane_width,
        num_lanes=1,
        edge_type="ramp",
    )

    # Lane connections at merge_node
    from_highway = (highway_a.node_id, merge_node.node_id)
    to_highway = (merge_node.node_id, highway_b.node_id)
    from_ramp = (ramp_origin.node_id, merge_node.node_id)

    connections: list[LaneConnection] = []
    # Through-highway lanes
    for lane in range(highway_num_lanes):
        connections.append(
            LaneConnection(
                node_id=merge_node.node_id,
                from_edge=from_highway,
                from_lane=lane,
                to_edge=to_highway,
                to_lane=lane,
            )
        )
    # Ramp merges into curbside (outermost) highway lane
    connections.append(
        LaneConnection(
            node_id=merge_node.node_id,
            from_edge=from_ramp,
            from_lane=0,
            to_edge=to_highway,
            to_lane=highway_num_lanes - 1,
        )
    )

    return (
        [merge_node],
        [seg_a_merge, seg_merge_b, ramp_edge],
        connections,
    )


# ---------------------------------------------------------------------------
# Traffic light helpers
# ---------------------------------------------------------------------------


def initial_traffic_light_states(
    traffic_lights: list[TrafficLight],
) -> list[TrafficLightState]:
    """Build the initial runtime states from static traffic light config."""
    states: list[TrafficLightState] = []
    for tl in traffic_lights:
        cycle = tl.green_duration + tl.yellow_duration + tl.red_duration
        t = tl.phase_offset % cycle
        if t < tl.green_duration:
            phase = TrafficLightPhase.GREEN
            time_in_phase = t
            time_remaining = tl.green_duration - t
        elif t < tl.green_duration + tl.yellow_duration:
            phase = TrafficLightPhase.YELLOW
            time_in_phase = t - tl.green_duration
            time_remaining = tl.yellow_duration - time_in_phase
        else:
            phase = TrafficLightPhase.RED
            time_in_phase = t - tl.green_duration - tl.yellow_duration
            time_remaining = tl.red_duration - time_in_phase
        states.append(
            TrafficLightState(
                light_id=tl.light_id,
                phase=phase,
                time_in_phase=time_in_phase,
                time_remaining=max(time_remaining, 0.0),
            )
        )
    return states


def update_traffic_lights(
    states: list[TrafficLightState],
    lights: list[TrafficLight],
    dt: float,
) -> list[TrafficLightState]:
    """Advance all traffic light phases by dt seconds.

    Returns a new list of states — no mutation.
    """
    light_by_id = {tl.light_id: tl for tl in lights}
    new_states: list[TrafficLightState] = []
    for state in states:
        tl = light_by_id[state.light_id]
        time_in_phase = state.time_in_phase + dt
        phase = state.phase

        # Advance through phase(s) if elapsed
        phase_durations = {
            TrafficLightPhase.GREEN: tl.green_duration,
            TrafficLightPhase.YELLOW: tl.yellow_duration,
            TrafficLightPhase.RED: tl.red_duration,
        }
        _next_phase = {
            TrafficLightPhase.GREEN: TrafficLightPhase.YELLOW,
            TrafficLightPhase.YELLOW: TrafficLightPhase.RED,
            TrafficLightPhase.RED: TrafficLightPhase.GREEN,
        }
        while time_in_phase >= phase_durations[phase] - 1e-9:
            time_in_phase -= phase_durations[phase]
            phase = _next_phase[phase]

        time_remaining = phase_durations[phase] - time_in_phase
        new_states.append(
            TrafficLightState(
                light_id=state.light_id,
                phase=phase,
                time_in_phase=time_in_phase,
                time_remaining=max(time_remaining, 0.0),
            )
        )
    return new_states


def add_traffic_lights(
    road_map: RoadMap,
    node_ids: list[int] | None = None,
    green_duration: float = 20.0,
    yellow_duration: float = 3.0,
    red_duration: float = 20.0,
) -> RoadMap:
    """Return a new RoadMap with traffic lights added at chosen intersections.

    If ``node_ids`` is None, lights are added at every node with ≥ 3 edges.
    Phase offsets are staggered by index so not all lights are green at once.
    """
    if node_ids is None:
        incoming_count: dict[int, int] = {}
        for edge in road_map.edges:
            incoming_count[edge.to_node] = incoming_count.get(edge.to_node, 0) + 1
        node_ids = [nid for nid, cnt in incoming_count.items() if cnt >= 3]

    # Build lookup: node_id → incoming edges
    incoming_edges: dict[int, list[tuple[int, int]]] = {}
    for edge in road_map.edges:
        incoming_edges.setdefault(edge.to_node, []).append(
            (edge.from_node, edge.to_node)
        )

    stagger = green_duration + yellow_duration  # offset between adjacent lights
    new_lights: list[TrafficLight] = []
    for idx, nid in enumerate(sorted(node_ids)):
        new_lights.append(
            TrafficLight(
                light_id=len(road_map.traffic_lights) + idx,
                node_id=nid,
                controlled_edges=incoming_edges.get(nid, []),
                green_duration=green_duration,
                yellow_duration=yellow_duration,
                red_duration=red_duration,
                phase_offset=(idx * stagger) % (green_duration + yellow_duration + red_duration),
            )
        )

    return road_map.model_copy(
        update={"traffic_lights": list(road_map.traffic_lights) + new_lights}
    )


def add_stop_signs(
    road_map: RoadMap,
    node_ids: list[int],
    distance_before_m: float = 3.0,
) -> RoadMap:
    """Return a new RoadMap with STOP signs on all incoming edges to ``node_ids``."""
    new_signs: list[RoadSign] = []
    sign_id = len(road_map.road_signs)
    for edge in road_map.edges:
        if edge.to_node in node_ids:
            stop_dist = max(0.0, edge.length - distance_before_m)
            new_signs.append(
                RoadSign(
                    sign_id=sign_id,
                    sign_type=SignType.STOP,
                    edge=(edge.from_node, edge.to_node),
                    distance_along_edge=stop_dist,
                )
            )
            sign_id += 1

    return road_map.model_copy(
        update={"road_signs": list(road_map.road_signs) + new_signs}
    )


# ---------------------------------------------------------------------------
# Demo map
# ---------------------------------------------------------------------------


def generate_demo_map() -> tuple[RoadMap, nx.DiGraph]:
    """Generate a mixed demo map for testing rich road features.

    Layout:
    - 4×4 urban grid (node IDs 0–15)
    - One highway segment running east-west along the top row, with an on-ramp
    - Two curved urban roads replacing straight edges in the grid
    - Traffic lights at the four central intersections
    - Stop signs at two minor intersections
    """
    config = MapConfig(grid_rows=4, grid_cols=4, block_size_m=80.0, default_num_lanes=2)
    road_map, _ = generate_road_map(config, seed=7)

    node_by_id = {n.node_id: n for n in road_map.nodes}
    # Collect mutable lists
    nodes = list(road_map.nodes)
    edges = list(road_map.edges)
    lane_connections = list(road_map.lane_connections)

    # ---- Replace two top-row horizontal edges with curved alternatives -----
    # Top row nodes in a 4×4 grid are IDs 12, 13, 14, 15
    # Replace edge 12→13 and 14→15 with curved versions
    curved_pairs = [(12, 13), (14, 15)]
    new_edges: list[RoadEdge] = []
    for edge in edges:
        pair = (edge.from_node, edge.to_node)
        if pair in curved_pairs:
            fn = node_by_id[edge.from_node]
            tn = node_by_id[edge.to_node]
            new_edges.append(
                make_curved_edge(
                    fn, tn,
                    bulge=18.0,
                    speed_limit=edge.speed_limit,
                    lane_width=edge.lane_width,
                    num_lanes=edge.num_lanes,
                )
            )
        else:
            new_edges.append(edge)
    edges = new_edges

    # ---- Highway segment: runs east-west parallel to bottom row ------------
    # Add two highway nodes south of node 0 and node 3
    next_id = max(n.node_id for n in nodes) + 1
    n0 = node_by_id[0]
    n3 = node_by_id[3]
    hw_west = RoadNode(node_id=next_id, position=Vector2(x=n0.position.x, y=n0.position.y - 60.0))
    hw_east = RoadNode(node_id=next_id + 1, position=Vector2(x=n3.position.x, y=n3.position.y - 60.0))
    nodes += [hw_west, hw_east]

    # Highway edges (one-directional west→east and east→west)
    edges.append(build_highway_segment(hw_west, hw_east))
    edges.append(build_highway_segment(hw_east, hw_west))

    # On-ramp from node 0 (bottom-left corner of grid) onto highway east
    ramp_new_nodes, ramp_edges, ramp_connections = build_ramp_merge(
        highway_a=hw_west,
        highway_b=hw_east,
        ramp_origin=node_by_id[0],
        merge_node_id=next_id + 2,
    )
    nodes += ramp_new_nodes
    edges += ramp_edges
    lane_connections += ramp_connections

    road_map = RoadMap(
        nodes=nodes,
        edges=edges,
        buildings=road_map.buildings,
        lane_connections=lane_connections,
    )

    # ---- Traffic lights at four central intersections (nodes 5, 6, 9, 10) --
    road_map = add_traffic_lights(road_map, node_ids=[5, 6, 9, 10])

    # ---- Stop signs at two corner intersections (nodes 0 and 15) -----------
    road_map = add_stop_signs(road_map, node_ids=[0, 15])

    graph = build_graph(road_map)
    return road_map, graph
