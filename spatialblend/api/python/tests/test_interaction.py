# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Tests for InteractionScript."""
import math
import unittest

import numpy as np

from spatialblend.api.python import (
    KernelType,
    Polygon2D,
    Simulator,
    SpaceBuilder,
    Trajectory,
)
from spatialblend.api.python.interaction import InteractionScript


def _make_space():
    """Create a simple space with a room, Board, and Table elements."""
    sim = Simulator.create()
    builder = sim.add_space()

    room_pts = [
        np.array([0.0, 0.0]),
        np.array([20.0, 0.0]),
        np.array([20.0, 20.0]),
        np.array([0.0, 20.0]),
    ]
    room = builder.add_element("Room").with_polygon(room_pts).with_shared_id(100)
    room.add_child("Board").at(5.0, 15.0).with_shared_id(200)
    room.add_child("Table").at(10.0, 3.0).with_shared_id(300)

    actor = (
        builder.add_actor()
        .with_anchor("root", 0.0, 0.0, 0.0)
        .with_anchor("hand", 0.3, 0.0, 0.0)
        .in_scope(room)
    )
    actor.bind("root", KernelType.SPACING)
    actor.move_to(np.array([3.0, 3.0]), 0.0)

    return sim, builder, actor, room


class TestLookAt(unittest.TestCase):
    def test_look_at_produces_trajectory(self):
        sim, builder, actor, room = _make_space()
        traj = InteractionScript().look_at("Board", 2.0).build(builder, actor)
        self.assertIsInstance(traj, Trajectory)
        self.assertGreaterEqual(traj.duration(), 2.0)

    def test_look_at_position_near_element(self):
        """Final position should be ~2.5 m from Board at (5, 15)."""
        sim, builder, actor, room = _make_space()
        traj = InteractionScript().look_at("Board", 2.0).build(builder, actor)
        end_pos = traj.position_at(traj.duration())
        board_pos = np.array([5.0, 15.0])
        dist = np.linalg.norm(end_pos - board_pos)
        self.assertAlmostEqual(dist, 2.5, delta=0.1)


class TestApproachTo(unittest.TestCase):
    def test_approach_to_produces_trajectory(self):
        sim, builder, actor, room = _make_space()
        traj = InteractionScript().approach_to("Table", 3.0).build(builder, actor)
        self.assertIsInstance(traj, Trajectory)
        self.assertGreaterEqual(traj.duration(), 3.0)


class TestLookAtThenApproach(unittest.TestCase):
    def test_chained_duration(self):
        """look_at(Board, 2) + approach_to(Table, 3) should have total >= 5 + transit."""
        sim, builder, actor, room = _make_space()
        traj = (
            InteractionScript()
            .look_at("Board", 2.0)
            .approach_to("Table", 3.0)
            .build(builder, actor)
        )
        self.assertGreaterEqual(traj.duration(), 5.0)


class TestReachTo(unittest.TestCase):
    def test_reach_to_produces_trajectory(self):
        sim, builder, actor, room = _make_space()
        traj = (
            InteractionScript()
            .reach_to("Table", "hand", 2.0)
            .build(builder, actor)
        )
        self.assertIsInstance(traj, Trajectory)
        self.assertGreaterEqual(traj.duration(), 2.0)

    def test_reach_to_binds_anchor(self):
        """After build, the actor should have an AFFORDANCE_POSE binding."""
        sim, builder, actor, room = _make_space()
        InteractionScript().reach_to("Table", "hand", 2.0).build(builder, actor)
        # Check that a binding exists on the space
        space = builder.get_space()
        bindings = space.get_bindings(actor.id())
        has_affordance = any(
            b.kernel_type == KernelType.AFFORDANCE_POSE
            and b.anchor_name == "hand"
            for b in bindings
        )
        self.assertTrue(has_affordance)


class TestWander(unittest.TestCase):
    def test_wander_produces_trajectory(self):
        sim, builder, actor, room = _make_space()
        traj = InteractionScript(seed=123).wander(4.0).build(builder, actor)
        self.assertIsInstance(traj, Trajectory)
        self.assertGreaterEqual(traj.duration(), 4.0)


class TestMoveTo(unittest.TestCase):
    def test_move_to_exact_position(self):
        sim, builder, actor, room = _make_space()
        traj = InteractionScript().move_to(10.0, 10.0, 1.0).build(builder, actor)
        end_pos = traj.position_at(traj.duration())
        self.assertAlmostEqual(end_pos[0], 10.0, delta=0.1)
        self.assertAlmostEqual(end_pos[1], 10.0, delta=0.1)


class TestFullChain(unittest.TestCase):
    def test_chain_all_commands(self):
        sim, builder, actor, room = _make_space()
        traj = (
            InteractionScript()
            .look_at("Board", 2.0)
            .approach_to("Table", 3.0)
            .reach_to("Table", "hand", 2.0)
            .move_to(15.0, 15.0, 1.0)
            .wander(4.0)
            .build(builder, actor)
        )
        self.assertIsInstance(traj, Trajectory)
        # Minimum dwell time alone is 2 + 3 + 2 + 1 + 4 = 12 s
        self.assertGreaterEqual(traj.duration(), 12.0)


class TestElementNotFound(unittest.TestCase):
    def test_raises_on_missing_tag(self):
        sim, builder, actor, room = _make_space()
        with self.assertRaises(ValueError):
            InteractionScript().look_at("NonExistent", 1.0).build(builder, actor)


if __name__ == "__main__":
    unittest.main()
