# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Round-trip and parity tests for the spatialblend_capi ctypes package.

Run under Bazel's cp310 toolchain via the `test_capi` py_test target. Also
verified manually against a genuine Python 3.8 interpreter (outside Bazel,
with SPATIALBLEND_V2_LIB pointed at a bazel-built libspatialblend);
this file itself has no Python-version-specific code so it runs unmodified
under either interpreter.

Key fact this file's scenarios are built around (confirmed by reading
spatialblend/space/space.cc): `realization_results_` -- the map queried by
sb_poll_realization_targets()/sb_register_on_targets()/GetRealizationTargets()
-- is populated only inside `Space::OnRemoteUpdate`, i.e. only when a space
*receives* a relation update from a peer space over the (possibly in-process)
hub. A single space with a self-bound actor legitimately produces bindings
(inspectable via the pybind get_bindings()/latest_matches API) but NEVER a
"realization target" through this callback/poll surface -- that mechanism is
inherently cross-space. The round-trip test below documents and asserts this
(poll returns [] in-space), and the dedicated two-space test is what exercises
the non-empty, cross-space target-delivery path that poll_realization_targets
is actually for.
"""
import os
import unittest

import numpy as np

# Disable the propagation movement deadband so repeated/identical
# update_actor_state calls in these tests reliably re-propagate (mirrors the
# pattern used by spatialblend/api/python/tests/test_space.py's cross-space test).
os.environ.setdefault("SB_V2_MIN_MOVE", "0")

from spatialblend.api.python.capi import api as capi

ROOM_POLYGON = [(0, 0), (20, 0), (20, 20), (0, 20)]
DESK_MIN = (4.0, 4.0, 0.0)
DESK_MAX = (6.0, 6.0, 1.0)


def _build_room_and_desk(space, room_shared_id=100, desk_shared_id=200):
    """Room (navigable polygon) with a furniture child (box obstacle),
    sharing IDs so peer spaces / pybind can address the same elements."""
    room = space.add_element()
    room.set_shared_id(room_shared_id)
    room.set_tag("Room")
    room.set_transform()  # identity, exercise the row-major setter path
    room.set_polygon(capi.GeometryRole.NAVIGABLE, ROOM_POLYGON)

    desk = space.add_element(parent=room)
    desk.set_shared_id(desk_shared_id)
    desk.set_tag("Desk")
    desk.set_collider(capi.ColliderType.BOX, [5, 5, 0.5, 2, 2, 1])
    desk.set_box(capi.GeometryRole.OBSTACLE, DESK_MIN, DESK_MAX)
    return room, desk


class RoundTripTest(unittest.TestCase):
    """session -> space -> room element (polygon) -> furniture element
    (shared id/tag/transform/collider) -> actor with anchors -> bind ->
    loop update_actor_state -> poll targets."""

    def test_single_space_round_trip(self):
        with capi.Session() as session:
            space = session.add_space()
            self.assertNotEqual(space.handle, 0)

            room, desk = _build_room_and_desk(space)
            self.assertNotEqual(room.id, 0)
            self.assertNotEqual(desk.id, 0)

            found_desk = space.find_element_by_shared_id(200)
            self.assertEqual(found_desk, desk)

            actor = space.create_actor()
            actor.add_anchor("root", 0.0, 0.0, 0.0)
            actor.add_standard_anchors()
            actor.set_element(room)
            actor.bind("root", capi.KernelType.SPACING)

            found_actor = None
            # The actor's element (room) carries shared_id 100; actors are
            # located by their *element's* shared id (see sb_set_actor_element
            # / GroundingEntityId wiring in spatialblend_c.cc).
            found_actor = space.find_actor_by_shared_id(100)
            self.assertEqual(found_actor, actor)

            last_state = None
            for _ in range(3):
                handle = actor.update_state(3.0, 5.0, 0.0)
                self.assertTrue(handle.try_wait(1000))
                self.assertIsNone(handle.get_error())
                # Same-space poll: by design (see module docstring) this is
                # always empty -- realization targets are a cross-space
                # delivery mechanism, not a same-space introspection one.
                targets = space.poll_realization_targets(actor)
                self.assertEqual(targets, [])
                last_state = actor.get_state()

            self.assertAlmostEqual(last_state.position_x, 3.0)
            self.assertAlmostEqual(last_state.position_y, 5.0)
            self.assertAlmostEqual(last_state.yaw, 0.0)

    def test_invalid_handles_do_not_crash(self):
        with capi.Session() as session:
            space = session.add_space()
            bogus_actor = 999999
            self.assertEqual(
                space.poll_realization_targets(bogus_actor), [])
            self.assertIsNone(space.find_element_by_shared_id(424242))
            self.assertIsNone(space.find_actor_by_shared_id(424242))


class TwoSpaceCrossSpaceTest(unittest.TestCase):
    """Two sb_add_space() calls in ONE session share the same in-process
    LocalContext/hub (no networking needed) -- updating the bound actor in
    the source space auto-dispatches a registered sb_on_targets_t callback
    on the peer ("target") space's matching actor."""

    def test_cross_space_target_arrives(self):
        with capi.Session() as session:
            source = session.add_space()
            room_a, _desk_a = _build_room_and_desk(source)

            actor = source.create_actor()
            actor.add_anchor("root", 0.0, 0.0, 0.0)
            actor.set_element(room_a)
            actor.bind("root", capi.KernelType.SPACING)

            target = session.add_space()
            room_b, _desk_b = _build_room_and_desk(target)

            avatar = target.create_actor()

            avatar.publish_embodied_pose(0.0, 0.0, 0.0, 0.0, 1)
            avatar.add_anchor("root", 0.0, 0.0, 0.0)
            avatar.set_element(room_b)

            actor.update_state(3.0, 5.0, 0.0)

            targets = target.poll_realization_targets(avatar)
            self.assertEqual(len(targets), 1)
            target0 = targets[0]
            self.assertIsInstance(target0, capi.RealizationTarget)
            self.assertTrue(all(
                abs(v) < 1e6 for v in
                (target0.position_x, target0.position_y)))


class ProposalTest(unittest.TestCase):
    """ABI 4 wrappers: sb_publish_anchor_pose / sb_read_proposals /
    sb_raise_source_exception / sb_poll_actor_event / sb_query_course,
    sb_set_body_radius (core proposes, the driver applies)."""

    def test_proposal_wrappers(self):
        with capi.Session() as session:
            source = session.add_space()
            room_a, _ = _build_room_and_desk(source)
            actor = source.create_actor()
            actor.add_anchor("root", 0.0, 0.0, 0.0)
            actor.set_element(room_a)
            actor.bind("root", capi.KernelType.SPACING)

            target = session.add_space()
            room_b, _ = _build_room_and_desk(target)
            target.set_body_radius(0.25)
            with self.assertRaises(ValueError):
                target.set_body_radius(-1.0)
            avatar = target.create_actor()
            avatar.add_anchor("root", 0.0, 0.0, 0.0)
            avatar.set_element(room_b)

            # No embodied pose published yet: core proposes nothing and says so.
            actor.update_state(3.0, 5.0, 0.0)
            p = avatar.read_proposal()
            self.assertEqual(p.status, 1)  # SB_PROPOSAL_NO_SAMPLE
            self.assertFalse(p.sample_valid)

            # The driver publishes its pose; the next source update solves.
            self.assertEqual(avatar.publish_embodied_pose(1.0, 1.0, 0.0, 0.0, 1), 0)
            self.assertEqual(avatar.publish_embodied_pose(1.0, 1.0, 0.0, 0.0, 1), -2)
            actor.update_state(3.2, 5.0, 0.0)
            p = avatar.read_proposal()
            self.assertIn(p.status, (2, 3))
            self.assertTrue(p.sample_valid)
            self.assertEqual(p.sample_sequence, 1)
            self.assertGreaterEqual(p.sample_age_s, 0.0)
            self.assertTrue(p.cause & 0x001)  # FIRST_PROPOSAL
            course = avatar.query_course(1.0, 1.0, p.x, p.y)
            self.assertGreaterEqual(len(course), 2)
            # One proposal per relation-bearing anchor: here only the root.
            self.assertEqual(len(p.anchors), 1)
            self.assertEqual(p.anchors[0].anchor_id, 0)
            self.assertTrue(p.root.sampled)
            self.assertEqual(avatar.anchor_id("root"), 0)
            pelvis = avatar.anchor_id("pelvis")
            self.assertGreater(pelvis, 0)
            self.assertEqual(avatar.anchor_name(pelvis), "pelvis")
            # Each anchor has its own port: sequence 1 is fresh on the pelvis.
            self.assertEqual(avatar.publish_anchor_pose(
                pelvis, 1.0, 1.0, 0.0, 0.0, 1, z=0.45), 0)
            self.assertEqual(avatar.publish_anchor_pose(
                pelvis, 1.0, 1.0, 0.0, 0.0, 1, z=0.45), -2)

    def test_source_exception_is_propagated_with_its_payload(self):
        with capi.Session() as session:
            source = session.add_space()
            room_a, _ = _build_room_and_desk(source)
            actor = source.create_actor()
            actor.add_anchor("root", 0.0, 0.0, 0.0)
            actor.set_element(room_a)
            actor.bind("root", capi.KernelType.SPACING)

            target = session.add_space()
            room_b, _ = _build_room_and_desk(target)
            avatar = target.create_actor()
            avatar.add_anchor("root", 0.0, 0.0, 0.0)
            avatar.set_element(room_b)
            avatar.publish_embodied_pose(1.0, 1.0, 0.0, 0.0, 1)

            actor.update_state(3.0, 5.0, 0.0)
            join = avatar.poll_event()  # the first update is the join exception
            self.assertEqual(join.kind, 1)
            self.assertEqual(join.exception_id, 1)
            self.assertIsNone(avatar.poll_event())
            actor.raise_source_exception(7, b"\x01\x02\x00\xfa")
            actor.update_state(3.2, 5.0, 0.0)
            e = avatar.poll_event()
            self.assertEqual((e.kind, e.exception_id), (1, 7))
            self.assertEqual(e.payload, b"\x01\x02\x00\xfa")
            p = avatar.read_proposal()
            self.assertTrue(p.cause & 0x010)  # SOURCE_EXCEPTION
            self.assertEqual(e.revision, p.revision)
            self.assertIsNone(avatar.poll_event())


class ParityTest(unittest.TestCase):
    """Run the identical two-space scenario through BOTH the pybind
    (spatialblend.api.python) and ctypes (spatialblend_capi) bindings and assert
    the solved target position matches to within 1e-9 -- both paths call the
    exact same C++ Space::OnRemoteUpdate/solver code, so this is expected to
    be exact (not merely close) modulo Python-layer marshalling, which for
    plain doubles introduces no precision loss either way."""

    def test_pybind_and_capi_produce_identical_targets(self):
        from spatialblend.api.python import (
            LocalContext,
            GeometryRole as PbGeometryRole,
            KernelType as PbKernelType,
            SubjectAnchor,
            Polygon2D,
            NavigableSpatialDefinition,
            PolytopeSpatialDefinition,
        )

        # --- pybind path -----------------------------------------------
        ctx = LocalContext.create()
        pb_source = ctx.add_space()
        pb_target = ctx.add_space()

        for space in (pb_source, pb_target):
            tree = space.element_tree
            tree.add_element(1)
            tree.get_element(1).shared_id = 100
            walkable = Polygon2D([
                np.array(list(p), dtype=float) for p in ROOM_POLYGON
            ])
            tree.get_element(1).set_spatial_definition(
                NavigableSpatialDefinition(walkable))

            tree.add_element(2, parent_id=1)
            tree.get_element(2).shared_id = 200
            tree.get_element(2).set_spatial_definition(
                PolytopeSpatialDefinition(
                    np.array(DESK_MIN, dtype=float),
                    np.array(DESK_MAX, dtype=float),
                    PbGeometryRole.OBSTACLE,
                ))

        pb_actor_id = pb_source.create_actor([SubjectAnchor("root")])
        pb_source.get_actor(pb_actor_id).set_element_id(1)
        pb_source.bind(pb_actor_id, "root", PbKernelType.SPACING)
        pb_source.set_hub(ctx.hub)

        pb_avatar_id = pb_target.create_actor([SubjectAnchor("root")])
        pb_target.get_actor(pb_avatar_id).set_element_id(1)
        pb_target.publish_embodied_pose(pb_avatar_id, 0.0, 0.0, 0.0, 0.0, 1)

        pb_source.update_actor_state(pb_actor_id, np.array([3.0, 5.0]), 0.0)

        pb_result = pb_target.get_realization_targets(pb_avatar_id)
        self.assertIsNotNone(pb_result)
        self.assertGreater(len(pb_result.targets), 0)
        pb_target0 = pb_result.targets[0]

        # --- ctypes/capi path (separate Session -- independent process
        # state, exercises the exact same underlying C++ Space code) -------
        with capi.Session() as session:
            source = session.add_space()
            room_a, _ = _build_room_and_desk(source)
            actor = source.create_actor()
            actor.add_anchor("root", 0.0, 0.0, 0.0)
            actor.set_element(room_a)
            actor.bind("root", capi.KernelType.SPACING)

            target = session.add_space()
            room_b, _ = _build_room_and_desk(target)
            avatar = target.create_actor()
            avatar.publish_embodied_pose(0.0, 0.0, 0.0, 0.0, 1)
            avatar.add_anchor("root", 0.0, 0.0, 0.0)
            avatar.set_element(room_b)

            actor.update_state(3.0, 5.0, 0.0)
            targets = target.poll_realization_targets(avatar)
            self.assertEqual(len(targets), 1)
            capi_target0 = targets[0]

        dx = abs(pb_target0.position[0] - capi_target0.position_x)
        dy = abs(pb_target0.position[1] - capi_target0.position_y)
        dyaw = abs(pb_target0.yaw - capi_target0.yaw)
        print(
            f"[parity] pybind=({pb_target0.position[0]!r}, "
            f"{pb_target0.position[1]!r}, yaw={pb_target0.yaw!r}) "
            f"capi=({capi_target0.position_x!r}, "
            f"{capi_target0.position_y!r}, yaw={capi_target0.yaw!r}) "
            f"|dx|={dx!r} |dy|={dy!r} |dyaw|={dyaw!r}"
        )
        self.assertLess(dx, 1e-9)
        self.assertLess(dy, 1e-9)
        self.assertLess(dyaw, 1e-9)


class StructLayoutTest(unittest.TestCase):
    """Every ctypes Structure against the library's sb_sizeof / offset tables."""

    def test_abi_and_every_struct(self):
        import ctypes

        from spatialblend.api.python.capi import _api, _obs_types, _types

        native = _api.get_native()
        self.assertEqual(int(native.sb_abi_version()), 4)
        self.assertEqual(int(native.sb_abi_minor_version()), 14)
        self.assertEqual(_api.EXPECTED_ABI_VERSION, 4)
        self.assertEqual(_api.EXPECTED_ABI_MINOR, 14)

        checked = _api.check_struct_layout(native.lib)
        declared = {cls for _, cls in _api.iter_native_layouts()}
        self.assertEqual(checked, declared)

        discovered = set()
        for mod in (_types, _obs_types):
            for obj in vars(mod).values():
                if (isinstance(obj, type) and issubclass(obj, ctypes.Structure)
                        and obj is not ctypes.Structure):
                    discovered.add(obj)
        # No sb_struct_id, so the library publishes no size or offset table.
        no_table = {
            _obs_types.SbActorObservation,
            _obs_types.SbSupportCorrespondence,
        }
        self.assertEqual(discovered - checked, no_table)
        self.assertLessEqual(checked, discovered)

        # ABI 4: one proposal per relation-bearing anchor. Every new struct
        # is checked field by field by check_struct_layout above; here the sizes
        # and one trailing offset of each are pinned against the library too.
        for struct_id, cls, last in (
                (_obs_types.SB_STRUCT_ANCHOR_POSE, _types.SbAnchorPose, "epoch"),
                (_obs_types.SB_STRUCT_ANCHOR_PROPOSAL, _types.SbAnchorProposal,
                 "referent_shared_id"),
                (_obs_types.SB_STRUCT_PROPOSAL_SET, _types.SbProposalSet,
                 "sample_age_s"),
                (_obs_types.SB_STRUCT_ACTOR_EVENT, _types.SbActorEvent,
                 "payload")):
            self.assertEqual(ctypes.sizeof(cls), native.sb_sizeof(struct_id))
            index = next(i for i, (name, _) in enumerate(cls._fields_)
                         if name == last)
            self.assertEqual(getattr(cls, last).offset,
                             native.sb_get_struct_field_offset(struct_id, index))
        # The retired ABI 3 ids answer 0: they are not reused.
        self.assertEqual(native.sb_sizeof(19), 0)
        self.assertEqual(native.sb_sizeof(20), 0)
        self.assertEqual(
            ctypes.sizeof(_types.SbOrientation),
            native.sb_sizeof(_obs_types.SB_STRUCT_ORIENTATION))


if __name__ == "__main__":
    unittest.main()
