"""Tests for actor behavioural models."""

import random

import pytest

from self_driving.actors import (
    create_pedestrian_actors,
    create_vehicle_actors,
    update_actors,
)
from self_driving.map_gen import generate_road_map
from self_driving.models import MapConfig


@pytest.fixture
def map_and_graph():
    cfg = MapConfig(grid_rows=4, grid_cols=4, block_size_m=50.0)
    return generate_road_map(cfg, seed=0)


@pytest.fixture
def road_map(map_and_graph):
    return map_and_graph[0]


@pytest.fixture
def graph(map_and_graph):
    return map_and_graph[1]


def test_create_vehicle_actors_count(road_map, graph):
    rng = random.Random(0)
    actors = create_vehicle_actors(road_map, graph, count=5, rng=rng)
    assert len(actors) == 5


def test_create_pedestrian_actors_count(road_map):
    rng = random.Random(0)
    actors = create_pedestrian_actors(road_map, count=3, rng=rng)
    assert len(actors) == 3


def test_actor_types_correct(road_map, graph):
    rng = random.Random(0)
    vehicles = create_vehicle_actors(road_map, graph, count=2, rng=rng)
    peds = create_pedestrian_actors(road_map, count=2, rng=rng)
    for v in vehicles:
        assert v.actor_type == "vehicle"
    for p in peds:
        assert p.actor_type == "pedestrian"


def test_actors_move_after_update(road_map, graph):
    rng = random.Random(0)
    actors = create_vehicle_actors(road_map, graph, count=2, rng=rng)
    # Give them a target far away
    updated = update_actors(actors, road_map, graph, dt=0.5, rng=rng)
    assert len(updated) == len(actors)
    # At least one actor should have moved
    moved = any(
        updated[i].pose.x != actors[i].pose.x or updated[i].pose.y != actors[i].pose.y
        for i in range(len(actors))
    )
    assert moved


def test_pedestrian_actors_move(road_map, graph):
    rng = random.Random(0)
    peds = create_pedestrian_actors(road_map, count=2, rng=rng)
    updated = update_actors(peds, road_map, graph, dt=0.5, rng=rng)
    moved = any(
        updated[i].pose.x != peds[i].pose.x or updated[i].pose.y != peds[i].pose.y
        for i in range(len(peds))
    )
    assert moved
