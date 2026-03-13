"""Tests for the LiDAR sensor simulation."""

import pytest

from self_driving.map_gen import generate_road_map
from self_driving.models import (
    LidarConfig,
    MapConfig,
    Pose,
    VehicleState,
)
from self_driving.sensors import simulate_lidar


@pytest.fixture
def ego() -> VehicleState:
    return VehicleState(
        pose=Pose(x=25.0, y=25.0, heading=0.0),
        speed=0.0,
        acceleration=0.0,
        steering_angle=0.0,
        timestamp=0.0,
    )


@pytest.fixture
def road_map():
    cfg = MapConfig(grid_rows=3, grid_cols=3, block_size_m=50.0)
    rm, _ = generate_road_map(cfg, seed=0)
    return rm


def test_scan_returns_correct_ray_count(ego, road_map):
    cfg = LidarConfig(num_rays=36, max_range=50.0)
    scan = simulate_lidar(ego, road_map, [], cfg)
    assert len(scan.points) == 36


def test_scan_distances_within_range(ego, road_map):
    cfg = LidarConfig(num_rays=36, max_range=50.0)
    scan = simulate_lidar(ego, road_map, [], cfg)
    for pt in scan.points:
        assert 0.0 <= pt.distance <= cfg.max_range


def test_scan_timestamp_matches_ego(ego, road_map):
    cfg = LidarConfig(num_rays=36, max_range=50.0)
    scan = simulate_lidar(ego, road_map, [], cfg)
    assert scan.timestamp == ego.timestamp


def test_no_hit_returns_max_range(road_map):
    # Place ego far from any road geometry
    far_ego = VehicleState(
        pose=Pose(x=1000.0, y=1000.0, heading=0.0),
        speed=0.0,
        acceleration=0.0,
        steering_angle=0.0,
        timestamp=0.0,
    )
    cfg = LidarConfig(num_rays=36, max_range=5.0)
    scan = simulate_lidar(far_ego, road_map, [], cfg)
    for pt in scan.points:
        assert not pt.hit
        assert pt.distance == cfg.max_range


def test_ego_pose_stored_in_scan(ego, road_map):
    cfg = LidarConfig(num_rays=12, max_range=50.0)
    scan = simulate_lidar(ego, road_map, [], cfg)
    assert scan.ego_pose.x == ego.pose.x
    assert scan.ego_pose.y == ego.pose.y
