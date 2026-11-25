# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""capi coverage of the observability / injection ABI.

Each test names the contract it checks. Everything goes through
spatialblend.api.python.capi -- no hand-written ctypes signatures -- which is
itself an acceptance check ("the harness reads each through the capi").
Analytic values are derived in the comments, not read back.
"""
import dataclasses
import json
import math
import os
import tempfile
import unittest

from spatialblend.api.python.capi import _api
from spatialblend.api.python.capi import _obs_types
from spatialblend.api.python.capi import api as capi
from spatialblend.api.python.capi import observability as obs
from spatialblend.api.python.capi import rich

ROOM = [(0, 0), (10, 0), (10, 10), (0, 10)]


def translate(x, y, z=0.0):
    return [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]


def progressive_second(n_min, n2, gamma):
    """v1 SoftAnnealing weight of the later member.

    eta = (n_min - 1 + eps)^gamma. Below eps the minimum takes the whole weight.
    """
    eta = (n_min - 1.0 + 1e-3) ** gamma
    safe = max(eta, 1e-6)
    if safe < 1e-3:
        return 0.0
    e = math.exp(-(n2 - n_min) / safe)
    return e / (1.0 + e)


class Receiver(object):
    """A 10x10 navigable room with geometry-less point referents, whose
    decoded spacing witness is exactly their world position."""

    def __init__(self, session):
        self.space = session.add_space()
        self.space.set_observability_capture(True)
        self.room = self.space.add_element()
        self.room.set_shared_id(100)
        self.room.set_polygon(capi.GeometryRole.NAVIGABLE, ROOM)
        actor_elem = self.space.add_element(parent=self.room)
        actor_elem.set_shared_id(900)
        self.actor = self.space.create_actor()
        self.actor.publish_embodied_pose(0.0, 0.0, 0.0, 0.0, 1)  # the driver's pose
        self.actor.add_anchor("root", 0, 0, 0)
        self.actor.add_standard_anchors()
        self.actor.set_element(actor_elem)

    def point(self, shared_id, x, y, z=0.0):
        e = self.space.add_element(parent=self.room)
        e.set_shared_id(shared_id)
        e.set_transform(translate(x, y, z))
        return e

    def surface(self, shared_id, x, y, z):
        """An affordance referent needs support geometry -- a 0.4 m
        square surface centred on (x, y) at height z."""
        e = self.space.add_element(parent=self.room)
        e.set_shared_id(shared_id)
        e.set_surface([[x - 0.2, y - 0.2, z], [x + 0.2, y - 0.2, z],
                       [x + 0.2, y + 0.2, z], [x - 0.2, y + 0.2, z]],
                      capi.GeometryRole.SURFACE)
        return e


def contact(ref, r, d, anchor="hand"):
    """An AffordancePose payload engaged at the support's chart origin
    (metric support distance d)."""
    # The count (the ABI's `estimated_time`): 1 in contact (v1; contact is
    # d < kContactEpsilonM = 1e-6, remote_update.h, which the ABI does not
    # export), else 2 (an approach). A count of 0 is no valid count: Progressive computes no weight
    # for it and it makes no request.
    return obs.RelationPayload(referent_shared_id=ref,
                               kernel_type=capi.KernelType.AFFORDANCE_POSE,
                               anchor_name=anchor, proximity=r,
                               support_distance_m=d, has_contact=True,
                               encoded_state=(0.0, 0.0, 0.0), chart_version=1,
                               estimated_time=1.0 if d < 1e-6 else 2.0)


def spacing(ref, r, t, anchor="root"):
    return obs.RelationPayload(referent_shared_id=ref,
                               kernel_type=capi.KernelType.SPACING,
                               anchor_name=anchor, proximity=r,
                               estimated_time=t)


class AbiTest(unittest.TestCase):
    def test_x1_layouts_and_versions(self):
        # Import already ran check_struct_layout (field-by-field offsets of
        # every mirrored struct); run it again explicitly so a failure is
        # attributed here.
        _api.check_struct_layout(_api.get_native().lib)
        self.assertEqual(obs.observability_version(), 1)
        self.assertEqual(obs.trace_format_version(), 10)  # format 10: + scope.reason inside_transfer_leaf
        self.assertEqual(capi.abi_version(), 4)
        self.assertEqual(capi.abi_minor_version(), 14)
        layout = capi.v2_abi_layout()
        self.assertEqual(layout["abi_minor"], 14)
        self.assertEqual(layout["realization_target_size"],
                         capi.sizeof(_obs_types.SB_STRUCT_REALIZATION_TARGET))
        self.assertEqual(capi.sizeof(_obs_types.SB_STRUCT_SCOPE_RECORD),
                         __import__("ctypes").sizeof(_obs_types.SbScopeRecord))


class PolicyTest(unittest.TestCase):
    def test_r3_5_declared_parameters(self):
        d = obs.default_policy_params()
        self.assertEqual(d.gamma, 1.0)  # v1 SoftAnnealing
        self.assertEqual((d.eta_min, d.eta_max), (1e-6, 100.0))
        self.assertEqual((d.epsilon_v, d.t_max, d.kappa), (0.01, 100.0, 0.2))
        with capi.Session() as session:
            r = Receiver(session)
            self.assertEqual(r.space.policy_params().gamma, 1.0)
            self.assertTrue(r.space.set_policy_params(gamma=2.0))
            self.assertEqual(r.space.policy_params().gamma, 2.0)
            self.assertFalse(r.space.set_policy_params(kappa=0.0))
            self.assertEqual(r.space.policy_params().kappa, 0.2)
            r.space.set_realization_policy(capi.RealizationPolicy.GREEDY)
            self.assertEqual(r.space.policy_params().policy,
                             int(capi.RealizationPolicy.GREEDY))

    def test_x2_r3_6_injected_series_drives_weights(self):
        with capi.Session() as session:
            r = Receiver(session)
            r.point(301, 2, 2)
            r.point(302, 8, 8)
            r.actor.inject_relation_payloads(
                [spacing(301, 0.9, 1.0), spacing(302, 0.5, 3.0)], 0.1)
            w = r.actor.realization_weights()
            w2 = progressive_second(1.0, 3.0, 1.0)
            self.assertAlmostEqual(w[1].weight, w2, places=12)
            self.assertEqual([x.is_driving for x in w], [1, 0])
            ch = r.actor.realization_channels()
            self.assertEqual(len(ch), 1)
            self.assertEqual(ch[0].driving_referent_shared_id, 301)
            self.assertEqual(ch[0].realized_referent_shared_id, 301)
            self.assertEqual(ch[0].realized_is_policy_argmax, 1)
            self.assertAlmostEqual(ch[0].eta, 1e-3, places=12)
            # Swap the series: the driving relation follows it.
            r.actor.inject_relation_payloads(
                [spacing(301, 0.9, 5.0), spacing(302, 0.5, 3.0)], 0.2)
            self.assertEqual(
                r.actor.realization_channels()[0].driving_referent_shared_id,
                302)
            with self.assertRaises(ValueError):
                r.actor.inject_relation_payloads([spacing(301, 1, 0)],
                                                 float("nan"))


class SolverTest(unittest.TestCase):
    def test_x1_witnesses_blend_snap_and_reach(self):
        with capi.Session() as session:
            r = Receiver(session)
            r.point(301, 2, 2)
            r.point(302, 8, 8)
            r.actor.inject_relation_payloads(
                [spacing(301, 0.9, 1.0), spacing(302, 0.5, 3.0)], 0.1)
            goals = r.actor.solver_goals()
            # n = 1 is contact, so that referent takes the whole weight. The first injection is the first
            # proposal, placed, which reads every position relation whatever its weight:
            # the zero-weight referent is listed too, with weight 0.
            self.assertEqual([(g.witness_x, g.witness_y) for g in goals],
                             [(2.0, 2.0), (8.0, 8.0)])
            self.assertEqual(goals[1].weight, 0.0)
            d = r.actor.solver_diagnostics()
            w2 = progressive_second(1.0, 3.0, 1.0)
            self.assertAlmostEqual(d.unconstrained_x, 2 + 6 * w2, places=9)
            self.assertEqual(d.snapped, 0)
            first_x, first_y = d.solved_x, d.solved_y
            # A later injection is a continuation (the pursued target is the
            # live rule, the placed answer is the first proposal's only). It has no update count,
            # so no request: the anchor stays at its current position, the sample (0, 0)
            # (v1's hold); its snap is measured from there.
            r.point(303, 12, 12)
            r.actor.inject_relation_payloads([spacing(303, 0.5, 1.0)], 0.2)
            d = r.actor.solver_diagnostics()
            self.assertEqual(d.snapped, 0)
            self.assertAlmostEqual(d.unconstrained_x, 12.0, places=9)
            self.assertAlmostEqual(d.unconstrained_y, 12.0, places=9)
            self.assertAlmostEqual(d.solved_x, 0.0, places=9, msg="the hold is the sample, not %r" % first_x)
            self.assertAlmostEqual(d.solved_y, 0.0, places=9, msg="the hold is the sample, not %r" % first_y)
            self.assertAlmostEqual(d.feasibility_snap_m, math.hypot(12.0, 12.0),
                                   places=6)
            # Hand contact at (9, 1), beyond the 0.9 m reach from the
            # root (2, 2): the target stays on the contact and is flagged.
            r.surface(310, 9, 1, 0.8)
            r.actor.inject_relation_payloads(
                [spacing(301, 1.0, 0.0), contact(310, 1.0, 0.0)], 0.3)
            targets = {t.anchor_name: t for t in r.actor.realization_targets_v2()}
            hand, root = targets["hand"], targets["root"]
            dist = math.hypot(9 - root.position_x, 1 - root.position_y)
            self.assertEqual(hand.beyond_reach, 1)
            self.assertAlmostEqual(hand.position_x, 9.0, places=9)
            self.assertAlmostEqual(hand.reach_excess_m, dist - 0.9, places=6)

    def test_r1_3_target_relation_evaluation(self):
        with capi.Session() as session:
            r = Receiver(session)
            r.surface(310, 5, 5, 0.8)
            e = r.space.evaluate_target_relation(
                contact(310, 1.0 / 1.15, 0.15), (5.05, 4.9, 1.2), 0.0)
            self.assertEqual(obs.disposition_name(e), "evaluated")
            # The payload carries no offset/height (a pre-pose-fit sender), so
            # both sides are scored with the proximity component.
            self.assertAlmostEqual(e.r_tgt, 1.0 / 1.4, places=9)
            self.assertAlmostEqual(e.r_src, 1.0 / 1.15, places=12)
            self.assertAlmostEqual(e.error, 1.0 / 1.4 - 1.0 / 1.15, places=9)
            # No offsets => formula version 0; support is the family's own
            # grounding (the payload names no part), not a fallback.
            self.assertEqual(e.r_formula_version, 0)
            self.assertEqual(e.support_basis, 0)
            # A pose-fit sender (offset/height supplied) is scored with the pose
            # fit on BOTH sides: position 0.5, height 0.3, yaw 0.2 (yaw
            # relative to the source's relative yaw, here 0 = candidate's).
            a4 = dataclasses.replace(
                contact(310, 1.0 / 1.15, 0.15), ap_offset_u_m=0.0,
                ap_offset_v_m=0.0, ap_height_m=0.15)
            e4 = r.space.evaluate_target_relation(a4, (5.05, 4.9, 1.2), 0.0)
            self.assertEqual(obs.disposition_name(e4), "evaluated")
            self.assertEqual(e4.r_formula_version, 1)
            self.assertAlmostEqual(e4.r_src, 0.5 + 0.3 / 1.15 + 0.2, places=9)
            # No named part: the support is the candidate's own projection,
            # so the in-plane offset is 0 (position fit 1); height 0.4.
            self.assertAlmostEqual(e4.r_tgt, 0.5 + 0.3 / 1.4 + 0.2,
                                   places=9)
            self.assertAlmostEqual(e4.error, e4.r_tgt - e4.r_src, places=12)
            missing = r.space.evaluate_target_relation(
                contact(999, 0.5, 1.0), (5, 5, 1), 0.0)
            self.assertEqual(obs.disposition_name(missing), "unmapped")


class RichAwaitingEngagementTest(unittest.TestCase):
    """An admitted affordance channel whose source anchor is not
    engaged reports disposition 5 on routine approach frames. The rich reader
    and the K1 serializer must carry it ("awaiting_engagement", Unity's
    RichContributorDisposition.AwaitingEngagement)."""

    def test_disposition_5_round_trips_through_the_k1_serializer(self):
        with capi.Session() as session:
            r = Receiver(session)
            r.point(301, 2, 2)
            r.surface(310, 3, 2, 0.8)
            with rich.RichRealization.activate(r.space, r.actor) as token:
                # Hand 0.3 m above the support: the gate is open (R = 1/1.3)
                # but the hand is not engaged (enter 0.05 m).
                r.actor.inject_relation_payloads(
                    [spacing(301, 1.0, 0.0), contact(310, 1.0 / 1.3, 0.3)],
                    0.1)
                snapshot = None
                while True:
                    polled = token.poll()
                    if polled is None:
                        break
                    snapshot = polled
                self.assertIsNotNone(snapshot)
                hand = [c for c in snapshot.channels if c.anchor == "hand"]
                self.assertEqual(len(hand), 1)
                self.assertFalse(hand[0].has_goal)
                self.assertEqual(hand[0].status, "converged")
                self.assertEqual(
                    [x.disposition for x in hand[0].contributors],
                    ["awaiting_engagement"])
                text = rich.to_rich_json(snapshot)
                # Unity's spelling (RichContributorDisposition.
                # AwaitingEngagement), byte-stable through read and re-write.
                self.assertIn('"disposition":"awaiting_engagement"', text)
                reread = rich.from_rich_json(text)
                self.assertEqual(
                    [x.disposition for c in reread.channels
                     if c.anchor == "hand" for x in c.contributors],
                    ["awaiting_engagement"])
                self.assertEqual(rich.to_rich_json(reread), text)


class SourceTest(unittest.TestCase):
    def build(self, session):
        space = session.add_space()
        space.set_observability_capture(True)
        room = space.add_element()
        room.set_shared_id(100)
        room.set_polygon(capi.GeometryRole.NAVIGABLE, ROOM)
        actor_elem = space.add_element(parent=room)
        actor_elem.set_shared_id(900)
        desk = space.add_element(parent=room)
        desk.set_shared_id(201)
        desk.set_tag("Desk")
        desk.set_box(capi.GeometryRole.OBSTACLE, (4, 4, 0), (6, 6, 1))
        chair = space.add_element(parent=room)
        chair.set_shared_id(202)
        chair.set_tag("Chair")
        chair.set_box(capi.GeometryRole.OBSTACLE, (8, 8, 0), (9, 9, 1))
        actor = space.create_actor()
        actor.publish_embodied_pose(0.0, 0.0, 0.0, 0.0, 1)  # the driver's pose
        actor.add_anchor("root", 0, 0, 0)
        actor.set_element(actor_elem)
        return space, room, actor_elem, desk, chair, actor

    def test_e1_7_scope_record(self):
        with capi.Session() as session:
            space, room, actor_elem, desk, chair, actor = self.build(session)
            actor.bind("root", capi.KernelType.SPACING)
            actor.bind("root", capi.KernelType.FACING)
            actor.bind("root", capi.KernelType.FACING)
            actor.update_state(2.0, 5.0, 0.0)
            records = actor.scope_records()
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0].navigable_element, room.id)
            self.assertEqual(records[0].deepest_containing, room.id)
            # D6: a tag binds only the AffordancePose family; a Facing
            # binding carries none, and no bind takes a scope.
            self.assertEqual(records[1].tag_filter, "")
            # Facing stays global: the chair is a referent along with the desk.
            self.assertEqual(records[2].explicit_scope, 0)
            reasons = [{c.element_id: c.reason_name
                        for c in actor.scope_candidates(i)} for i in range(3)]
            self.assertEqual(reasons[0][actor_elem.id], "actor_element")
            self.assertEqual(reasons[0][desk.id], "included")
            self.assertEqual(reasons[1][desk.id], "included")
            self.assertEqual(reasons[1][chair.id], "included")
            self.assertEqual(reasons[2][chair.id], "included")
            self.assertEqual(reasons[2][desk.id], "included")

    def test_x1_x2_relation_state_and_pure_evaluation(self):
        with capi.Session() as session:
            space, room, _, desk, _, actor = self.build(session)
            actor.bind("root", capi.KernelType.SPACING)
            actor.update_state(2.0, 5.0, 0.0)
            rows = {s.referent_element_id: s for s in actor.relation_states_v2()}
            v1 = {s.referent_element_id: s for s in actor.relation_states()}
            row = rows[desk.id]
            self.assertEqual(row.proximity, v1[desk.id].proximity)
            # 2 m to the desk face; room diagonal sqrt(200).
            self.assertAlmostEqual(row.support_distance_m, 2.0, places=6)
            self.assertAlmostEqual(row.source_scope_diameter_m,
                                   math.sqrt(200), places=6)
            e = space.evaluate_relation(capi.KernelType.SPACING,
                                        (2.0, 5.0, 0.0), 0.0, desk, room)
            self.assertEqual(e.proximity, row.proximity)
            self.assertEqual(e.encoded_state, row.encoded_state)
            self.assertEqual(e.navigable_element, room.id)

    def test_x1_world_transform(self):
        with capi.Session() as session:
            space = session.add_space()
            a = space.add_element()
            a.set_transform(translate(1, 2))
            b = space.add_element(parent=a)
            b.set_transform([[0, -1, 0, 3], [1, 0, 0, 0], [0, 0, 1, 1],
                             [0, 0, 0, 1]])
            w = b.world_transform()
            self.assertEqual((w[0][3], w[1][3], w[2][3]), (4.0, 2.0, 1.0))
            self.assertEqual(b.parent(), a)
            self.assertIsNone(a.parent())


class TraceTest(unittest.TestCase):
    def test_x3_r3_6_trace_through_capi(self):
        path = os.path.join(tempfile.mkdtemp(), "trace.jsonl")
        with capi.Session() as session:
            r = Receiver(session)
            r.point(301, 2, 2)
            r.point(302, 8, 8)
            r.space.set_trace(path)
            r.actor.inject_relation_payloads(
                [spacing(301, 0.9, 1.0), spacing(302, 0.5, 3.0)], 0.1)
            r.space.set_trace(None)
        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        kinds = [ev["ev"] for ev in lines]
        self.assertEqual(kinds[0], "meta")
        self.assertIn("policy", kinds)
        channel = [ev for ev in lines if ev["ev"] == "channel"]
        self.assertEqual(channel[0]["driving_referent"], 301)
        policy = [ev for ev in lines if ev["ev"] == "policy"][0]
        self.assertEqual(policy["gamma"], 1.0)


if __name__ == "__main__":
    unittest.main()
