# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Tests for spatialblend Python bindings."""
import numpy as np
import unittest
import sys
import os

# Import the bindings
from spatialblend.api.python import (
    LocalContext, Space, GeometryRole, KernelType, StateType,
    SubjectAnchor, Polygon2D, NavigableSpatialDefinition,
    Polygon3D, SurfaceSpatialDefinition, PolytopeSpatialDefinition,
    RealizationTarget, SolverStatus,
)


class TestElementTree(unittest.TestCase):
    def test_create_elements(self):
        space = Space()
        tree = space.element_tree
        tree.add_element(1)
        tree.add_element(2, parent_id=1)
        self.assertEqual(tree.size(), 2)

    def test_element_properties(self):
        space = Space()
        tree = space.element_tree
        tree.add_element(1)
        elem = tree.get_element(1)
        self.assertEqual(elem.id, 1)
        self.assertEqual(elem.geometry_role, GeometryRole.UNSPECIFIED)

        elem.shared_id = 100
        self.assertEqual(elem.shared_id, 100)

        elem.tag = "Writable"
        self.assertEqual(elem.tag, "Writable")


class TestActor(unittest.TestCase):
    def test_create_actor(self):
        space = Space()
        tree = space.element_tree
        tree.add_element(1)

        anchors = [
            SubjectAnchor("root", np.array([0, 0, 0], dtype=float)),
            SubjectAnchor("chest", np.array([0, 0, 1.2], dtype=float)),
        ]
        actor_id = space.create_actor(anchors)
        actor = space.get_actor(actor_id)
        self.assertIsNotNone(actor)
        self.assertEqual(len(actor.get_anchors()), 2)


class TestBind(unittest.TestCase):
    def test_bind_and_update(self):
        ctx = LocalContext.create()
        space = ctx.add_space()

        tree = space.element_tree
        tree.add_element(1)
        walkable = Polygon2D(
            [np.array([0, 0]), np.array([20, 0]),
             np.array([20, 20]), np.array([0, 20])])
        tree.get_element(1).set_spatial_definition(
            NavigableSpatialDefinition(walkable))

        tree.add_element(2, parent_id=1)
        tree.get_element(2).shared_id = 200
        t = np.eye(4)
        t[0, 3] = 5.0
        t[1, 3] = 8.0
        tree.get_element(2).set_local_transform(t)

        anchors = [SubjectAnchor("root")]
        actor_id = space.create_actor(anchors)
        space.get_actor(actor_id).set_element_id(1)

        space.bind(actor_id, "root", KernelType.SPACING)

        space.update_actor_state(actor_id, np.array([3, 6]), 0.0)

        bindings = space.get_bindings(actor_id)
        self.assertEqual(len(bindings), 1)
        self.assertGreater(len(bindings[0].latest_matches), 0)

        match = bindings[0].latest_matches[0]
        self.assertEqual(match.element_id, 2)
        self.assertGreater(match.eval.grounded.proximity, 0)


class TestPolytopeSpatialDefinition(unittest.TestCase):
    def test_box_volume_geometry(self):
        box = PolytopeSpatialDefinition(
            np.array([0, 0, 0], dtype=float),
            np.array([2, 2, 1], dtype=float),
            GeometryRole.CONTAINMENT,
        )

        self.assertEqual(box.geometry_role, GeometryRole.CONTAINMENT)
        self.assertTrue(box.contains(np.array([1, 1], dtype=float)))
        self.assertTrue(box.contains_3d(np.array([1, 1, 0.5], dtype=float)))
        self.assertFalse(box.contains_3d(np.array([1, 1, 2], dtype=float)))
        np.testing.assert_allclose(
            box.get_closest_point(np.array([3, 1, 0.5], dtype=float)),
            np.array([2, 1, 0.5], dtype=float),
        )
        self.assertAlmostEqual(
            box.get_distance(np.array([3, 1, 0.5], dtype=float)), 1.0)

    def test_element_role_comes_from_polytope_geometry(self):
        space = Space()
        tree = space.element_tree
        tree.add_element(1)
        elem = tree.get_element(1)
        elem.set_spatial_definition(
            PolytopeSpatialDefinition(
                np.array([0, 0, 0], dtype=float),
                np.array([1, 1, 1], dtype=float),
                GeometryRole.OBSTACLE,
            ))

        self.assertEqual(elem.geometry_role, GeometryRole.OBSTACLE)


class TestCrossSpace(unittest.TestCase):
    def test_cross_space_transfer(self):
        ctx = LocalContext.create()
        space_a = ctx.add_space()
        space_b = ctx.add_space()

        # Setup both spaces
        for space, board_x, board_y in [(space_a, 5, 15), (space_b, 10, 3)]:
            tree = space.element_tree
            tree.add_element(1)
            tree.get_element(1).shared_id = 100
            walkable = Polygon2D(
                [np.array([0, 0]), np.array([20, 0]),
                 np.array([20, 20]), np.array([0, 20])])
            tree.get_element(1).set_spatial_definition(
                NavigableSpatialDefinition(walkable))

            tree.add_element(2, parent_id=1)
            tree.get_element(2).shared_id = 200
            t = np.eye(4)
            t[0, 3] = board_x
            t[1, 3] = board_y
            tree.get_element(2).set_local_transform(t)

        # Actor in space A
        actor_id = space_a.create_actor([SubjectAnchor("root")])
        space_a.get_actor(actor_id).set_element_id(1)
        space_a.set_hub(ctx.hub)

        # Avatar in space B
        avatar_id = space_b.create_actor([SubjectAnchor("root")])
        space_b.get_actor(avatar_id).set_element_id(1)
        # The application (this test) owns the avatar's embodied pose and
        # publishes it through the sampling port; core only proposes.
        space_b.publish_embodied_pose(avatar_id, 0.0, 0.0, 0.0, 0.0, 1)
        seq = 1

        # Bind and update; disable movement gate so repeated same-pose updates
        # propagate (progressive output needs ~30-50 iters to converge).
        space_a.bind(actor_id, "root", KernelType.SPACING)
        import os
        os.environ["SB_V2_MIN_MOVE"] = "0"
        src_pos = np.array([3, 13])
        avatar_pos = None
        for _ in range(50):
            space_a.update_actor_state(actor_id, src_pos, 0.0)
            status, px, py, pyaw, _cause, _sseq = space_b.read_proposal(avatar_id)
            if status in (2, 3):  # exact / inexact: the driver adopts it
                seq += 1
                space_b.publish_embodied_pose(avatar_id, px, py, pyaw, 0.0, seq)
                avatar_pos = np.array([px, py])

        # The avatar's driver moved to the proposal
        self.assertIsNotNone(avatar_pos)
        # Should be near Board_B at (10, 3)
        dist = np.linalg.norm(avatar_pos - np.array([10, 3]))
        self.assertLess(dist, 10.0)


class TestElementAnchor(unittest.TestCase):
    def test_element_anchor(self):
        """Test element anchor add/get."""
        context = LocalContext.create()
        space = context.add_space()
        tree = space.element_tree
        tree.add_element(1)
        elem = tree.get_element(1)

        anchor_transform = np.eye(4)
        anchor_transform[2, 3] = 1.5  # z offset
        elem.add_anchor("surface_center", anchor_transform)

        anchor = elem.get_anchor("surface_center")
        self.assertIsNotNone(anchor)

    def test_element_get_anchors(self):
        """Test get_anchors returns all anchors."""
        space = Space()
        tree = space.element_tree
        tree.add_element(1)
        elem = tree.get_element(1)

        elem.add_anchor("top", np.eye(4))
        elem.add_anchor("bottom", np.eye(4))
        anchors = elem.get_anchors()
        self.assertEqual(len(anchors), 2)


class TestRealizationTarget(unittest.TestCase):
    def test_realization_target(self):
        """Test RealizationTarget struct is accessible."""
        target = RealizationTarget()
        self.assertEqual(target.arrival_time, 0.0)
        self.assertEqual(target.weight, 0.0)

    def test_solver_status_enum(self):
        """Test SolverStatus enum values."""
        self.assertEqual(SolverStatus.CONVERGED, SolverStatus.CONVERGED)
        self.assertNotEqual(SolverStatus.CONVERGED, SolverStatus.FAILED)


class TestSpaceNewMethods(unittest.TestCase):
    def test_set_update_interval(self):
        """Test set_update_interval is callable."""
        space = Space()
        space.set_update_interval(0.016)  # 60fps

    def test_share_unshare_element(self):
        """Test share/unshare element."""
        context = LocalContext.create()
        space = context.add_space()
        tree = space.element_tree
        tree.add_element(1)
        elem = tree.get_element(1)
        elem.shared_id = 42
        # Should not raise
        space.share_element(1)
        space.unshare_element(1)

    def test_get_realization_targets(self):
        """Test get_realization_targets returns None for unknown actor."""
        space = Space()
        result = space.get_realization_targets(9999)
        self.assertIsNone(result)


class TestAnchorRealizationTargets(unittest.TestCase):
    """get_realization_targets() surfaces per-anchor AffordancePose
    targets (not just the root), across a real cross-space propagation —
    mirrors the C++ e2e coverage in
    simulator/anchor_realization_test.cc / scenario_replay_test.cc."""

    def test_hand_anchor_target_surfaces_with_height(self):
        ctx = LocalContext.create()
        space_a = ctx.add_space()
        space_b = ctx.add_space()

        # Mirrored 1:1 layout in both spaces: Room(100) -> Board(200) at the
        # same (10, 10), with the board raised 0.75m off the floor, so the
        # decoded "hand" anchor target lands within the default 0.9m reach of
        # a root that has also converged onto the board (both driven off the
        # same referent), without needing to widen the reach radius.
        for space in (space_a, space_b):
            tree = space.element_tree
            tree.add_element(1)
            tree.get_element(1).shared_id = 100
            walkable = Polygon2D(
                [np.array([0, 0]), np.array([20, 0]),
                 np.array([20, 20]), np.array([0, 20])])
            tree.get_element(1).set_spatial_definition(
                NavigableSpatialDefinition(walkable))

            tree.add_element(2, parent_id=1)
            tree.get_element(2).shared_id = 200
            t = np.eye(4)
            t[0, 3] = 10.0
            t[1, 3] = 10.0
            t[2, 3] = 0.75
            tree.get_element(2).set_local_transform(t)
            # A hand support needs geometry (a geometryless
            # referent is now a grounding failure, never a target at its
            # origin). The board top is a 2 x 2 m surface at z = 0.75.
            tree.get_element(2).set_spatial_definition(
                SurfaceSpatialDefinition(Polygon3D([
                    np.array([9.0, 9.0, 0.75]), np.array([11.0, 9.0, 0.75]),
                    np.array([11.0, 11.0, 0.75]), np.array([9.0, 11.0, 0.75])])))

        # Actor in space A: root bound to Spacing, hand bound to
        # AffordancePose, both auto-resolving to the sole referent (Board).
        anchors = [
            SubjectAnchor("root"),
            # A hand target exists only while the source hand TOUCHES its
            # support, so the hand rests ON the board top (z = 0.75).
            SubjectAnchor("hand", np.array([0.4, 0, 0.75], dtype=float)),
        ]
        actor_id = space_a.create_actor(anchors)
        space_a.get_actor(actor_id).set_element_id(1)
        space_a.set_hub(ctx.hub)
        space_a.bind(actor_id, "root", KernelType.SPACING)
        space_a.bind(actor_id, "hand", KernelType.AFFORDANCE_POSE)

        # Avatar in space B.
        avatar_anchors = [
            SubjectAnchor("root"),
            SubjectAnchor("hand", np.array([0.4, 0, 0.9], dtype=float)),
        ]
        avatar_id = space_b.create_actor(avatar_anchors)
        space_b.get_actor(avatar_id).set_element_id(1)
        space_b.publish_embodied_pose(avatar_id, 0.0, 0.0, 0.0, 0.0, 1)
        seq = 1

        # Walk the actor toward the board, close enough to engage the
        # AffordancePose gate (>= enter threshold) without many iterations.
        # Disable the movement gate so repeated same-pose updates propagate
        # (progressive output needs several iters to converge) -- same
        # precedent as TestCrossSpace.test_cross_space_transfer above.
        os.environ["SB_V2_MIN_MOVE"] = "0"
        src_pos = np.array([9.7, 9.4])
        for _ in range(30):
            space_a.update_actor_state(actor_id, src_pos, 0.0)
            status, px, py, pyaw, _cause, _sseq = space_b.read_proposal(avatar_id)
            if status in (2, 3):
                seq += 1
                space_b.publish_embodied_pose(avatar_id, px, py, pyaw, 0.0, seq)

        result = space_b.get_realization_targets(avatar_id)
        self.assertIsNotNone(result)
        by_anchor = {t.anchor_name: t for t in result.targets}
        self.assertIn("root", by_anchor)
        self.assertIn("hand", by_anchor)

        hand = by_anchor["hand"]
        self.assertTrue(hand.has_z)
        self.assertAlmostEqual(hand.z, 0.75, places=6)
        # Within the anchor's reach (default hand 0.9 m) of root's converged
        # position. Reach is no longer a clamp; this layout keeps the
        # engaged contact reachable, so the target is not flagged.
        root = by_anchor["root"]
        dist = np.linalg.norm(hand.position - root.position)
        self.assertLessEqual(dist, 0.9 + 1e-6)
        self.assertAlmostEqual(hand.position[0], 10.1, places=6)
        self.assertAlmostEqual(hand.position[1], 9.4, places=6)


class TestAnchorLiveness(unittest.TestCase):
    """Anchor world-pose override + relation-state query."""

    def _build(self):
        ctx = LocalContext.create()
        space = ctx.add_space()

        tree = space.element_tree
        tree.add_element(1)
        walkable = Polygon2D(
            [np.array([0, 0]), np.array([20, 0]),
             np.array([20, 20]), np.array([0, 20])])
        tree.get_element(1).set_spatial_definition(
            NavigableSpatialDefinition(walkable))

        # "Tabletop": a leaf element with no boundary of its own, at world
        # (5, 8). Affordance-pose proximity to a boundary-less referent
        # genuinely varies with anchor distance (see space_test.cc for why a
        # referent WITH a boundary makes proximity trivially 1.0 always).
        tree.add_element(2, parent_id=1)
        t = np.eye(4)
        t[0, 3] = 5.0
        t[1, 3] = 8.0
        tree.get_element(2).set_local_transform(t)
        tree.get_element(2).tag = "Tabletop"
        # shared: an element with no shared id is never a referent (SP-D7-03)
        tree.get_element(2).shared_id = 2

        anchors = [
            SubjectAnchor("root"),
            SubjectAnchor("hand", np.array([0.4, 0, 0.9], dtype=float)),
        ]
        actor_id = space.create_actor(anchors)
        space.get_actor(actor_id).set_element_id(1)
        space.bind(actor_id, "hand", KernelType.AFFORDANCE_POSE, "Tabletop")
        # Keep ctx alive: LocalContext uniquely owns the Space, so letting
        # ctx be garbage-collected would free the Space out from under the
        # caller's reference (a use-after-free on space_mutex_).
        self._ctx = ctx
        return space, actor_id

    def test_set_anchor_world_pose_grounds_affordance_root_unchanged(self):
        space, actor_id = self._build()
        actor = space.get_actor(actor_id)

        space.update_actor_state(actor_id, np.array([5.0, 8.0]), 0.0)
        before = space.get_relation_states(actor_id)
        self.assertEqual(len(before), 1)
        self.assertLess(before[0].proximity, 0.95)

        root_pos_before = actor.get_position().copy()
        root_yaw_before = actor.get_rotation()

        # Move the "hand" anchor exactly onto the table's world origin.
        space.set_anchor_world_pose(
            actor_id, "hand", np.array([5.0, 8.0, 0.0]), 0.3)
        space.update_actor_state(actor_id, np.array([5.0, 8.0]), 0.0)

        after = space.get_relation_states(actor_id)
        self.assertEqual(len(after), 1)
        self.assertGreaterEqual(after[0].proximity, 1.0 - 1e-6)
        self.assertEqual(after[0].anchor_name, "hand")

        # Root position bit-unchanged.
        root_pos_after = actor.get_position()
        np.testing.assert_array_equal(root_pos_after, root_pos_before)
        self.assertEqual(actor.get_rotation(), root_yaw_before)

        # Clearing restores the offset-driven baseline exactly.
        space.clear_anchor_world_pose(actor_id, "hand")
        space.update_actor_state(actor_id, np.array([5.0, 8.0]), 0.0)
        cleared = space.get_relation_states(actor_id)
        self.assertEqual(len(cleared), 1)
        self.assertEqual(cleared[0].proximity, before[0].proximity)

    def test_get_relation_states_matches_binding_match(self):
        space, actor_id = self._build()
        space.update_actor_state(actor_id, np.array([5.0, 8.0]), 0.0)

        states = space.get_relation_states(actor_id)
        bindings = space.get_bindings(actor_id)
        self.assertEqual(len(bindings), 1)
        self.assertEqual(len(bindings[0].latest_matches), 1)

        match = bindings[0].latest_matches[0]
        self.assertEqual(states[0].proximity, match.eval.grounded.proximity)
        self.assertEqual(states[0].kernel_type, KernelType.AFFORDANCE_POSE)

    def test_get_relation_states_empty_without_bindings(self):
        space = Space()
        actor_id = space.create_actor([SubjectAnchor("root")])
        self.assertEqual(space.get_relation_states(actor_id), [])


if __name__ == "__main__":
    unittest.main()
