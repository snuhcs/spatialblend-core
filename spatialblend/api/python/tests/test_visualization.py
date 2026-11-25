# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Tests that visualization renders without errors."""
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import os
import tempfile
import numpy as np
import unittest

from spatialblend.api.python import (
    LocalContext, KernelType, SubjectAnchor,
    Polygon2D, NavigableSpatialDefinition,
    Simulator, SimulationResult, Trajectory, SpaceBuilder,
)
from spatialblend.api.python.visualization import (
    plot, render_simulation, export_mp4, KERNEL_COLORS,
)


def _make_space(ctx, board_x=10.0, board_y=10.0):
    """Create a space with a room (id=1) and one object (id=2)."""
    space = ctx.add_space()
    tree = space.element_tree
    tree.add_element(1)
    tree.get_element(1).set_spatial_definition(
        NavigableSpatialDefinition(
            Polygon2D([np.array([0, 0]), np.array([20, 0]),
                       np.array([20, 20]), np.array([0, 20])])))
    tree.add_element(2, parent_id=1)
    t = np.eye(4)
    t[0, 3] = board_x
    t[1, 3] = board_y
    tree.get_element(2).set_local_transform(t)
    return space


class TestPlot(unittest.TestCase):
    def setUp(self):
        self.ctx = LocalContext.create()
        self.space = _make_space(self.ctx, board_x=10.0, board_y=10.0)

        self.actor_id = self.space.create_actor([SubjectAnchor("root")])
        self.space.get_actor(self.actor_id).set_element_id(1)
        self.space.bind(self.actor_id, "root", KernelType.SPACING)
        self.space.update_actor_state(self.actor_id, np.array([5.0, 5.0]), 0.0)

    def test_space_plot(self):
        fig = plot.space(self.space, self.actor_id, title="Test")
        self.assertIsNotNone(fig)
        matplotlib.pyplot.close(fig)

    def test_space_plot_no_actor(self):
        fig = plot.space(self.space, title="No Actor")
        self.assertIsNotNone(fig)
        matplotlib.pyplot.close(fig)

    def test_trajectory_plot(self):
        positions = [[5, 5], [7, 7], [10, 10]]
        fig = plot.trajectory(self.space, positions, self.actor_id)
        self.assertIsNotNone(fig)
        matplotlib.pyplot.close(fig)

    def test_trajectory_plot_no_actor(self):
        positions = [[2, 2], [8, 8], [15, 15]]
        fig = plot.trajectory(self.space, positions)
        self.assertIsNotNone(fig)
        matplotlib.pyplot.close(fig)


class TestCrossSpacePlot(unittest.TestCase):
    def setUp(self):
        self.ctx = LocalContext.create()
        self.space_a = _make_space(self.ctx, board_x=5.0, board_y=15.0)
        self.space_b = _make_space(self.ctx, board_x=10.0, board_y=3.0)

        # Actor in space A
        self.actor_id = self.space_a.create_actor([SubjectAnchor("root")])
        self.space_a.get_actor(self.actor_id).set_element_id(1)
        self.space_a.set_hub(self.ctx.hub)

        # Avatar in space B
        self.avatar_id = self.space_b.create_actor([SubjectAnchor("root")])
        self.space_b.get_actor(self.avatar_id).set_element_id(1)
        self.space_b.publish_embodied_pose(self.avatar_id, 0.0, 0.0, 0.0, 0.0, 1)

        self.space_a.bind(self.actor_id, "root", KernelType.SPACING)
        self.space_a.update_actor_state(
            self.actor_id, np.array([3.0, 13.0]), 0.0)

    def test_cross_space_plot(self):
        fig = plot.cross_space(
            self.space_a, self.space_b,
            self.actor_id, self.avatar_id,
            title="Cross-Space Test")
        self.assertIsNotNone(fig)
        matplotlib.pyplot.close(fig)


class TestKernelColors(unittest.TestCase):
    def test_expected_keys(self):
        self.assertIn("SPACING", KERNEL_COLORS)
        self.assertIn("FACING", KERNEL_COLORS)
        self.assertIn("AFFORDANCE_POSE", KERNEL_COLORS)
        self.assertEqual(len(KERNEL_COLORS), 3)


class TestRenderSimulation(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator.create()
        builder_a = self.sim.add_space()
        builder_b = self.sim.add_space()

        room_pts = [
            np.array([0.0, 0.0]),
            np.array([20.0, 0.0]),
            np.array([20.0, 20.0]),
            np.array([0.0, 20.0]),
        ]
        room_a = (
            builder_a.add_element("Room")
            .with_polygon(room_pts)
            .with_shared_id(100)
        )
        room_a.add_child("Board").at(5.0, 15.0).with_shared_id(200)
        self.actor = (
            builder_a.add_actor()
            .with_anchor("root", 0.0, 0.0, 0.0)
            .in_scope(room_a)
        )
        self.actor.bind("root", KernelType.SPACING)
        self.actor.move_to(np.array([3.0, 3.0]), 0.0)

        room_b = (
            builder_b.add_element("Room")
            .with_polygon(room_pts)
            .with_shared_id(100)
        )
        room_b.add_child("Board").at(5.0, 15.0).with_shared_id(200)
        self.avatar = (
            builder_b.add_actor()
            .with_anchor("root", 0.0, 0.0, 0.0)
            .in_scope(room_b)
        )
        self.avatar.bind("root", KernelType.SPACING)
        self.avatar.move_to(np.array([10.0, 10.0]), 0.0)

        traj = Trajectory.straight_line(
            np.array([3.0, 3.0]), np.array([3.0, 13.0]), 5.0
        )
        self.result = self.sim.run(self.actor, self.avatar, traj, 0.1)
        self.space_a = builder_a.get_space()
        self.space_b = builder_b.get_space()

    def test_render_simulation_returns_animation(self):
        import matplotlib.animation as animation
        anim = render_simulation(
            self.result, self.space_a, self.space_b,
            self.actor.id(), self.avatar.id(),
            fps=10, title="Test Anim")
        self.assertIsInstance(anim, animation.FuncAnimation)
        # Verify frame count matches by checking the internal frame sequence
        # FuncAnimation._init_draw is set; frame_seq is the frames arg
        self.assertEqual(anim._save_count, self.result.num_frames)
        matplotlib.pyplot.close("all")

    def test_render_simulation_with_timeline(self):
        import matplotlib.animation as animation
        labels = [
            (0.0, 2.0, "ApproachTo board"),
            (2.0, 5.0, "LookAt board"),
        ]
        anim = render_simulation(
            self.result, self.space_a, self.space_b,
            self.actor.id(), self.avatar.id(),
            interaction_labels=labels,
            fps=10)
        self.assertIsInstance(anim, animation.FuncAnimation)
        matplotlib.pyplot.close("all")


class TestExportMp4(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator.create()
        builder_a = self.sim.add_space()
        builder_b = self.sim.add_space()

        room_pts = [
            np.array([0.0, 0.0]),
            np.array([20.0, 0.0]),
            np.array([20.0, 20.0]),
            np.array([0.0, 20.0]),
        ]
        room_a = (
            builder_a.add_element("Room")
            .with_polygon(room_pts)
            .with_shared_id(100)
        )
        self.actor = (
            builder_a.add_actor()
            .with_anchor("root", 0.0, 0.0, 0.0)
            .in_scope(room_a)
        )
        self.actor.bind("root", KernelType.SPACING)
        self.actor.move_to(np.array([3.0, 3.0]), 0.0)

        room_b = (
            builder_b.add_element("Room")
            .with_polygon(room_pts)
            .with_shared_id(100)
        )
        self.avatar = (
            builder_b.add_actor()
            .with_anchor("root", 0.0, 0.0, 0.0)
            .in_scope(room_b)
        )
        self.avatar.bind("root", KernelType.SPACING)
        self.avatar.move_to(np.array([10.0, 10.0]), 0.0)

        traj = Trajectory.straight_line(
            np.array([3.0, 3.0]), np.array([3.0, 13.0]), 1.0
        )
        self.result = self.sim.run(self.actor, self.avatar, traj, 0.5)
        self.space_a = builder_a.get_space()
        self.space_b = builder_b.get_space()

    def test_export_mp4_creates_file(self):
        anim = render_simulation(
            self.result, self.space_a, self.space_b,
            self.actor.id(), self.avatar.id(),
            fps=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.mp4")
            try:
                export_mp4(anim, path, fps=2, dpi=72)
                self.assertTrue(os.path.exists(path))
                self.assertGreater(os.path.getsize(path), 0)
            except FileNotFoundError:
                # ffmpeg not installed in CI - skip gracefully
                self.skipTest("ffmpeg not available")
        matplotlib.pyplot.close("all")


if __name__ == "__main__":
    unittest.main()
