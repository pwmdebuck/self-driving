"""Tests for localisation layer."""

import math
import random

import pytest

from self_driving.localization import localize
from self_driving.map_gen import generate_road_map
from self_driving.models import (
    LidarConfig,
    MapConfig,
    Pose,
    VehicleState,
)
from self_driving.sensors import simulate_lidar


@pytest.fixture
def road_map():
    cfg = MapConfig(grid_rows=3, grid_cols=3, block_size_m=50.0)
    rm, _ = generate_road_map(cfg, seed=0)
    return rm


@pytest.fixture
def ego(road_map) -> VehicleState:
    node = road_map.nodes[0]
    return VehicleState(
        pose=Pose(x=node.position.x, y=node.position.y, heading=0.0),
        speed=0.0,
        acceleration=0.0,
        steering_angle=0.0,
        timestamp=1.0,
    )


def test_estimated_pose_near_ground_truth(ego, road_map):
    cfg = LidarConfig(num_rays=36, max_range=50.0)
    scan = simulate_lidar(ego, road_map, [], cfg)
    rng = random.Random(0)
    loc = localize(scan, ego, road_map, rng=rng)
    dist = math.sqrt(
        (loc.estimated_pose.x - ego.pose.x) ** 2
        + (loc.estimated_pose.y - ego.pose.y) ** 2
    )
    assert dist < 2.0  # within 2 m of ground truth


def test_timestamp_propagated(ego, road_map):
    cfg = LidarConfig(num_rays=36, max_range=50.0)
    scan = simulate_lidar(ego, road_map, [], cfg)
    loc = localize(scan, ego, road_map)
    assert loc.timestamp == scan.timestamp


def test_nearest_road_edge_found(ego, road_map):
    cfg = LidarConfig(num_rays=36, max_range=50.0)
    scan = simulate_lidar(ego, road_map, [], cfg)
    loc = localize(scan, ego, road_map)
    assert loc.nearest_road_edge is not None
