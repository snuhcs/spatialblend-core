# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Tests for spatialblend Simulator Python bindings."""
import math
import numpy as np
import unittest

from spatialblend.api.python import (
    Simulator,
    SimulatorRunConfig,
    SimulationResult,
    Trajectory,
    SpaceBuilder,
    ElementHandle,
    ActorHandle,
    KernelType,
    Polygon2D,
)


class TestTrajectory(unittest.TestCase):
    def test_straight_line(self):
        traj = Trajectory.straight_line(
            np.array([0.0, 0.0]), np.array([10.0, 0.0]), 5.0
        )
        self.assertAlmostEqual(traj.duration(), 5.0)
        pos = traj.position_at(2.5)
        self.assertAlmostEqual(pos[0], 5.0, places=3)
        self.assertAlmostEqual(pos[1], 0.0, places=3)

    def test_from_waypoints(self):
        waypoints = [
            (0.0, np.array([0.0, 0.0]), 0.0),
            (1.0, np.array([5.0, 0.0]), 0.5),
            (2.0, np.array([5.0, 5.0]), 1.0),
        ]
        traj = Trajectory.from_waypoints(waypoints)
        self.assertAlmostEqual(traj.duration(), 2.0)

    def test_add_waypoint(self):
        traj = Trajectory()
        traj.add_waypoint(0.0, np.array([0.0, 0.0]), 0.0)
        traj.add_waypoint(1.0, np.array([3.0, 4.0]), 1.57)
        self.assertAlmostEqual(traj.duration(), 1.0)


class TestSimulatorRunConfig(unittest.TestCase):
    def test_defaults(self):
        config = SimulatorRunConfig()
        self.assertAlmostEqual(config.max_linear_speed, 1.4)
        self.assertAlmostEqual(config.max_angular_speed, 2.0)
        self.assertTrue(config.use_velocity_bounded)

    def test_set_fields(self):
        config = SimulatorRunConfig()
        config.max_linear_speed = 2.0
        config.max_angular_speed = 3.0
        config.use_velocity_bounded = False
        config.max_catchup_time = 5.0
        config.catchup_convergence = 0.01
        self.assertAlmostEqual(config.max_linear_speed, 2.0)
        self.assertFalse(config.use_velocity_bounded)


class TestSimulatorCreate(unittest.TestCase):
    def test_create(self):
        sim = Simulator.create()
        self.assertIsNotNone(sim)

    def test_add_space_returns_builder(self):
        sim = Simulator.create()
        builder = sim.add_space()
        self.assertIsInstance(builder, SpaceBuilder)


class TestFluentAPI(unittest.TestCase):
    def test_element_handle_chaining(self):
        sim = Simulator.create()
        builder = sim.add_space()

        room_pts = [
            np.array([0.0, 0.0]),
            np.array([20.0, 0.0]),
            np.array([20.0, 20.0]),
            np.array([0.0, 20.0]),
        ]

        room = (
            builder.add_element("Room")
            .with_polygon(room_pts)
            .with_shared_id(100)
        )
        self.assertIsInstance(room, ElementHandle)
        self.assertEqual(room.id(), 1)

        board = room.add_child("Board").at(5.0, 15.0).with_shared_id(200)
        self.assertIsInstance(board, ElementHandle)

    def test_actor_handle_chaining(self):
        sim = Simulator.create()
        builder = sim.add_space()

        room_pts = [
            np.array([0.0, 0.0]),
            np.array([20.0, 0.0]),
            np.array([20.0, 20.0]),
            np.array([0.0, 20.0]),
        ]
        room = builder.add_element("Room").with_polygon(room_pts).with_shared_id(100)

        actor = (
            builder.add_actor()
            .with_anchor("root", 0.0, 0.0, 0.0)
            .in_scope(room)
        )
        self.assertIsInstance(actor, ActorHandle)


class TestSimulatorRun(unittest.TestCase):
    def test_run_returns_result(self):
        sim = Simulator.create()
        builder = sim.add_space()

        room_pts = [
            np.array([0.0, 0.0]),
            np.array([20.0, 0.0]),
            np.array([20.0, 20.0]),
            np.array([0.0, 20.0]),
        ]
        room = (
            builder.add_element("Room")
            .with_polygon(room_pts)
            .with_shared_id(100)
        )
        board = room.add_child("Board").at(5.0, 15.0).with_shared_id(200)

        actor = (
            builder.add_actor()
            .with_anchor("root", 0.0, 0.0, 0.0)
            .in_scope(room)
        )
        actor.bind("root", KernelType.SPACING)
        actor.move_to(np.array([3.0, 3.0]), 0.0)

        avatar = (
            builder.add_actor()
            .with_anchor("root", 0.0, 0.0, 0.0)
            .in_scope(room)
        )
        avatar.bind("root", KernelType.SPACING)
        avatar.move_to(np.array([10.0, 10.0]), 0.0)

        traj = Trajectory.straight_line(
            np.array([3.0, 3.0]), np.array([3.0, 13.0]), 5.0
        )
        result = sim.run(actor, avatar, traj, 0.1)

        self.assertIsInstance(result, SimulationResult)
        self.assertGreater(result.num_frames, 0)
        self.assertAlmostEqual(result.dt, 0.1)
        self.assertEqual(len(result.avatar_positions), result.num_frames)
        self.assertEqual(len(result.avatar_yaws), result.num_frames)

    def test_quality_metrics_finite(self):
        sim = Simulator.create()
        builder = sim.add_space()

        room_pts = [
            np.array([0.0, 0.0]),
            np.array([20.0, 0.0]),
            np.array([20.0, 20.0]),
            np.array([0.0, 20.0]),
        ]
        room = (
            builder.add_element("Room")
            .with_polygon(room_pts)
            .with_shared_id(100)
        )
        room.add_child("Board").at(5.0, 15.0).with_shared_id(200)

        actor = (
            builder.add_actor()
            .with_anchor("root", 0.0, 0.0, 0.0)
            .in_scope(room)
        )
        actor.bind("root", KernelType.SPACING)
        actor.move_to(np.array([3.0, 3.0]), 0.0)

        avatar = (
            builder.add_actor()
            .with_anchor("root", 0.0, 0.0, 0.0)
            .in_scope(room)
        )
        avatar.bind("root", KernelType.SPACING)
        avatar.move_to(np.array([10.0, 10.0]), 0.0)

        traj = Trajectory.straight_line(
            np.array([3.0, 3.0]), np.array([3.0, 13.0]), 5.0
        )
        result = sim.run(actor, avatar, traj, 0.1)

        self.assertTrue(math.isfinite(result.total_avatar_travel()))
        self.assertTrue(math.isfinite(result.max_speed()))
        self.assertTrue(
            math.isfinite(result.final_distance_to(np.array([5.0, 15.0])))
        )
        self.assertTrue(math.isfinite(result.max_acceleration()))
        self.assertTrue(math.isfinite(result.mean_jerk()))
        self.assertIsInstance(result.direction_changes(), int)
        self.assertTrue(
            math.isfinite(result.arrival_lag(np.array([5.0, 15.0])))
        )
        self.assertTrue(math.isfinite(result.mean_relation_error()))

    def test_avatar_always_in_polygon(self):
        sim = Simulator.create()
        builder = sim.add_space()

        room_pts = [
            np.array([0.0, 0.0]),
            np.array([20.0, 0.0]),
            np.array([20.0, 20.0]),
            np.array([0.0, 20.0]),
        ]
        room = (
            builder.add_element("Room")
            .with_polygon(room_pts)
            .with_shared_id(100)
        )
        room.add_child("Board").at(5.0, 15.0).with_shared_id(200)

        actor = (
            builder.add_actor()
            .with_anchor("root", 0.0, 0.0, 0.0)
            .in_scope(room)
        )
        actor.bind("root", KernelType.SPACING)
        actor.move_to(np.array([3.0, 3.0]), 0.0)

        avatar = (
            builder.add_actor()
            .with_anchor("root", 0.0, 0.0, 0.0)
            .in_scope(room)
        )
        avatar.bind("root", KernelType.SPACING)
        avatar.move_to(np.array([10.0, 10.0]), 0.0)

        traj = Trajectory.straight_line(
            np.array([3.0, 3.0]), np.array([3.0, 13.0]), 5.0
        )
        result = sim.run(actor, avatar, traj, 0.1)

        polygon = Polygon2D(room_pts)
        self.assertTrue(result.avatar_always_in_polygon(polygon))

    def test_tick(self):
        sim = Simulator.create()
        builder = sim.add_space()
        room_pts = [
            np.array([0.0, 0.0]),
            np.array([20.0, 0.0]),
            np.array([20.0, 20.0]),
            np.array([0.0, 20.0]),
        ]
        builder.add_element("Room").with_polygon(room_pts).with_shared_id(100)
        actor = builder.add_actor().with_anchor("root", 0.0, 0.0, 0.0)
        actor.move_to(np.array([5.0, 5.0]), 0.0)
        # Should not raise
        sim.tick(0.1)


if __name__ == "__main__":
    unittest.main()
