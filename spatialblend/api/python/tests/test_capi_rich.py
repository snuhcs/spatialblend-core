# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Rich-realization capi wrappers and their JSON wire format.

Golden evidence:
- tests/data/rich_realization_unity/*.json are native captures exported by
  the Unity app's (snuhcs/spatialblend-unity) EditMode test
  FoundationRichRealizationNativeTests.
  NativeSourceReceiverCapturesResetTimestampedAndRetirementTransactions
  (Assets/SpatialBlend/Framework/Test/Restoration/FoundationRichRealizationTests.cs);
  the manifest's sha256 values are Unity's own.
- tests/vendor/motiongen_rich_realization.py is a verbatim copy of the
  MotionGen consumer (the motion generator is not public).

The tests replay the SAME native scenario through the Python capi and
require the Python serialization to be byte-identical to Unity's after the
values that differ between any two runs are translated RELATIONALLY: space
handles, the producer lifetime and the clock epoch must map one-to-one and
consistently across all three snapshots of the run (so a source/receiver
swap fails), and only the raw receipt/solve nanoseconds are taken as-is.

A second Unity capture (tests/data/rich_realization_unity_support, three
channels, from FoundationRichSurfaceNativeTests) cannot be replayed through
the capi: it depends on the Unity object layer (SpatialBlendObject
registration, float32 transforms). For it the tests check the serializer
alone: loading it and re-serializing is byte-identical, and MotionGen
parses both.
"""
import hashlib
import json
import math
import os
import re
import struct
import unittest

from spatialblend.api.python.capi import api as capi
from spatialblend.api.python.capi import rich
from spatialblend.api.python.tests.vendor import motiongen_rich_realization as mg

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "rich_realization_unity")

# Run-volatile values, by identity class. Values in one class must map
# one-to-one between the Unity run and ours; "ns" values are not identities.
_VOLATILE_CLASS = {
    "receiver_space_id": "space", "source_actor_space": "space",
    "relay_source_space": "space", "source_space": "space",
    "producer_lifetime": "lifetime", "native_clock_epoch": "clock_epoch",
    "native_receipt_nanoseconds": "ns", "native_solve_nanoseconds": "ns",
}
_VOLATILE_RE = re.compile(
    r'"(%s)":(\{"known":true,"value":)?"(-?\d+)"' %
    "|".join(_VOLATILE_CLASS))


def translate(ours, theirs, mapping):
    """Rewrite our volatile values into Unity's, checking that every
    identity class maps consistently and injectively (mapping is shared
    across the snapshots of one run). Returns the rewritten text."""
    a = list(_VOLATILE_RE.finditer(ours))
    b = list(_VOLATILE_RE.finditer(theirs))
    if [m.group(1) for m in a] != [m.group(1) for m in b]:
        raise AssertionError("volatile fields differ in order or presence")
    out, last = [], 0
    for ma, mb in zip(a, b):
        cls = _VOLATILE_CLASS[ma.group(1)]
        if cls != "ns":
            forward = mapping.setdefault(cls, {})
            backward = mapping.setdefault(cls + "^-1", {})
            ours_v, theirs_v = ma.group(3), mb.group(3)
            if forward.setdefault(ours_v, theirs_v) != theirs_v or \
                    backward.setdefault(theirs_v, ours_v) != ours_v:
                raise AssertionError(
                    "%s %s maps inconsistently (%s): %r vs %r" % (
                        cls, ma.group(1), ours_v, forward, backward))
        out.append(ours[last:ma.start(3)])
        out.append(mb.group(3))
        last = ma.end(3)
    out.append(ours[last:])
    return "".join(out)


# The binding revision changed a VALUE, not the schema: the vendored Unity capture predates
# the source binding revision, so its contributors report an unknown binding
# revision where the current core reports the source binding's revision. Rewrite exactly those
# two values back to the capture's (asserting ours are known), so every other
# byte is still compared. Re-capturing the Unity fixture removes this.
_BINDING_RE = re.compile(
    r'"binding_revision":\{"known":true,"value":"\d+"\}(.*?)'
    r'"binding_revision_known":true')


def pre381_binding_revision(ours, theirs):
    if '"binding_revision_known":true' in theirs:
        return ours  # a capture with the binding revision: compare as-is
    rewritten, n = _BINDING_RE.subn(
        lambda m: '"binding_revision":{"known":false}' + m.group(1) +
        '"binding_revision_known":false', ours)
    assert n == ours.count('"binding_revision_known"'), (
        "every contributor must now carry a known binding revision")
    return rewritten


# The exact observation (2, 2, 0) is admitted two doubles below 2 on x
# (0x1.ffffffffffffep+0) and one double above 2 on y. Both are 2^-51 away.
# That is floating-point roundoff through the solve, not a different pose.
_ADMITTED_OBSERVATION_ULPS = 2
_JSON_NUMBER = re.compile(
    r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _ulps_apart(a, b):
    if a == b:
        return 0
    if math.isnan(a) and math.isnan(b):
        return 0
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return 1 << 62
    ia = struct.unpack("<q", struct.pack("<d", float(a)))[0]
    ib = struct.unpack("<q", struct.pack("<d", float(b)))[0]
    return abs(ia - ib)


def _json_numbers(text):
    """(start, end, value) of every JSON number that is not inside a string."""
    spans = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == '"':
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        match = _JSON_NUMBER.match(text, i)
        if match and (text[i].isdigit() or text[i] == "-"):
            token = match.group(0)
            spans.append((match.start(), match.end(), float(token)))
            i = match.end()
            continue
        i += 1
    return spans


def align_admitted_observation_roundoff(ours, theirs):
    """Use the golden spelling where a number is within the admitted-pose
    tolerance. Every other byte stays ours, so the compare is still exact."""
    oa, ta = _json_numbers(ours), _json_numbers(theirs)
    if len(oa) != len(ta):
        return ours, []
    out, last, replaced = [], 0, []
    for (sb_, oe, ov), (ts, te, tv) in zip(oa, ta):
        out.append(ours[last:sb_])
        if ov != tv and _ulps_apart(ov, tv) <= _ADMITTED_OBSERVATION_ULPS:
            out.append(theirs[ts:te])
            replaced.append((ov, tv))
        else:
            out.append(ours[sb_:oe])
        last = oe
    out.append(ours[last:])
    return "".join(out), replaced


def v1_no_count_weight(label, theirs):
    """The Unity golden was captured while a lone relation took weight 1 whatever
    its count. Its timestamped frame is the source's FIRST observation, with no
    valid count yet: Progressive now computes no weight for it, as v1, so core reports the contributor's applied weight 0
    where the golden says 1. The receiver is still placed (the join reads every
    relation whatever its weight): the channel's goal, status,
    disposition and target weight are unchanged. Exactly this one number
    differs; everything else is compared byte for byte."""
    if label != "timestamped_complete":
        return theirs
    for field in ('"applied_weight":1,',):
        assert theirs.count(field) == 1, field
        theirs = theirs.replace(field, field[:-2] + "0,")
    return theirs


def unity(label):
    with open(os.path.join(DATA, label + ".json"), "rb") as f:
        return f.read()


def configure(space, room_id, object_id):
    room = space.add_element()
    room.set_shared_id(room_id)
    room.set_polygon(capi.GeometryRole.NAVIGABLE,
                     [(0, 0), (10, 0), (10, 10), (0, 10)])
    obstacle = space.add_element(parent=room)
    obstacle.set_shared_id(object_id)
    # Unity passes (4,0,4)-(6,1,6) Y-up, converted to Z-up by its wrapper.
    obstacle.set_box(capi.GeometryRole.OBSTACLE, (4, 4, 0), (6, 6, 1))
    return room


def run_unity_scenario():
    """NativeSourceReceiverCapturesResetTimestampedAndRetirementTransactions,
    call for call, through the Python capi. Returns the three exported
    snapshots and the polled copy of the complete one."""
    captured, replacement = [], []
    with capi.Session() as session:
        source = session.add_space()
        target = session.add_space()
        source_room = configure(source, 900000000000000001, 900000000000000003)
        target_room = configure(target, 900000000000000001, 900000000000000003)
        source_actor = source.create_actor()
        receiver_actor = target.create_actor()
        receiver_actor.publish_embodied_pose(0.0, 0.0, 0.0, 0.0, 1)
        source_actor.add_anchor("root", 0, 0, 0)
        receiver_actor.add_anchor("root", 0, 0, 0)
        source_actor.set_element(source_room)
        receiver_actor.set_element(target_room)
        source_actor.bind("root", capi.KernelType.SPACING)
        session.enable_actor_observation_frames()
        rich.register_on_rich_realization(target, captured.append)
        token = rich.RichRealization.activate(target, receiver_actor)

        epoch = source_actor.begin_observation_epoch("root")
        assert len(captured) == 1
        assert source_actor.apply_observation_frame(
            epoch, 0, 12.5, [("root", 2, 2, 0, 0)])
        assert len(captured) == 2
        polled = token.poll()
        assert not source_actor.apply_observation_frame(
            epoch, 1, 12.6, [("root", 2, 2, 0, 0),
                             ("unknown", 0, 0, 0, float("nan"))])
        assert token.poll() is None

        rich.register_on_rich_realization(target, replacement.append)
        source_actor.begin_observation_epoch("root")
        assert len(captured) == 2 and len(replacement) == 1
        token.retire()
        assert len(replacement) == 2
        assert token.release() == 1
        rich.register_on_rich_realization(target, None)
    return {
        "observation_reset_empty": captured[0],
        "timestamped_complete": captured[1],
        "activation_retirement_empty": replacement[1],
    }, polled


class RichWireFormatTest(unittest.TestCase):
    def test_number_format_matches_unity_roundtrip(self):
        # Unity mono's double.ToString("R") (see capi/_mono_number.py).
        cases = [(2.2204460492503131e-13, "2.2204460492503131E-13"),
                 (1.1960510015487671, "1.1960510015487671"),
                 # float32-derived exact binary ties at the 17th digit round
                 # half AWAY from zero (Python's ".17g" gives ...062 / ...062).
                 (0.99924087524414063, "0.99924087524414063"),
                 (90619585105019.0625, "90619585105019.063"),
                 # mono decides G15 with its own lossy parser:
                 (5.0867259815001304, "5.08672598150013"),
                 (3.6129, "3.6128999999999998"),
                 (-0.0, "0"),
                 (0.0, "0"), (2.0, "2"), (12.5, "12.5"),
                 (0.75, "0.75"), (-0.5, "-0.5"), (100.0, "100"),
                 (1e-4, "0.0001"), (1e-5, "1E-05"), (1.25e-7, "1.25E-07"),
                 (1e14, "100000000000000"), (1e15, "1E+15"),
                 (0.1, "0.1"), (1.0 / 3.0, "0.33333333333333331")]
        for value, expected in cases:
            self.assertEqual(rich._number(value), expected, repr(value))

    def test_number_format_matches_unity_mono_sample(self):
        # tests/data/mono_double_r_sample.tsv: 6017 doubles formatted by
        # Unity 6000.0.56f1's MonoBleedingEdge (tests/data/MonoDoubleR.cs,
        # `mono ref.exe out.tsv 1500`): fixed edge cases, float32 values
        # widened to double (frequent exact ties), random 64-bit patterns
        # (subnormals included) and random doubles in [-100, 100]. The same
        # generator at 100000 (399,555 values) matched with 0 mismatches
        # when this was written; the sample keeps the repo small.
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "mono_double_r_sample.tsv")
        mismatches, n = [], 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                bits, expected = line.rstrip("\n").split("\t")
                value = struct.unpack(">d", bytes.fromhex(bits))[0]
                n += 1
                got = rich._number(value)
                if got != expected:
                    mismatches.append((bits, expected, got))
        self.assertEqual(n, 6017)
        self.assertEqual(mismatches[:10], [], "%d of %d differ" % (
            len(mismatches), n))

    def test_string_escaping_matches_csharp(self):
        self.assertEqual(rich._string('a"b\\c\n\t\x01é'),
                         '"a\\"b\\\\c\\n\\t\\u0001é"')

    def test_vendored_unity_fixtures_are_intact_and_parse(self):
        with open(os.path.join(DATA, "manifest.json")) as f:
            manifest = json.load(f)
        self.assertEqual(manifest["schema"],
                         "spatialblend.rich_realization.native_manifest")
        for case in manifest["cases"]:
            payload = unity(case["label"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(),
                             case["sha256"])
            parsed = mg.parse_rich_realization(payload)
            self.assertEqual(parsed.header.snapshot_status,
                             case["snapshot_status"])


class RichNativeGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshots, cls.polled = run_unity_scenario()

    def test_k1_python_serialization_parses_with_motiongen(self):
        for label, snapshot in self.snapshots.items():
            text = rich.to_rich_json(snapshot)
            parsed = mg.parse_rich_realization(text)
            self.assertEqual(parsed.header.snapshot_status,
                             snapshot.header.snapshot_status, label)
            # MotionGen's own deterministic serializer round-trips it.
            self.assertEqual(
                mg.parse_rich_realization(mg.serialize_rich_realization(parsed)),
                parsed)

    def test_k1_python_serialization_is_byte_identical_to_unity(self):
        mapping = {}
        for label in ("observation_reset_empty", "timestamped_complete",
                      "activation_retirement_empty"):
            ours = rich.to_rich_json(self.snapshots[label])
            theirs = unity(label).decode("utf-8")
            theirs = v1_no_count_weight(label, theirs)
            ours = pre381_binding_revision(ours, theirs)
            ours = translate(ours, theirs, mapping)
            ours, replaced = align_admitted_observation_roundoff(ours, theirs)
            # The admitted (2, 2, 0) observation reaches the golden's roundoff
            # exactly through the single chart, so no number needs the
            # two-ulp tolerance.
            self.assertEqual(replaced, [], label)
            self.assertEqual(ours, theirs, label)
        # The run's two spaces (source, receiver) were both identified.
        self.assertEqual(len(mapping["space"]), 2)

    def test_relational_check_catches_a_source_receiver_swap(self):
        ours = rich.to_rich_json(self.snapshots["timestamped_complete"])
        theirs = unity("timestamped_complete").decode("utf-8")
        sources = re.findall(r'"source_actor_space":\{"known":true,'
                             r'"value":"(\d+)"', ours)
        receiver = re.search(r'"receiver_space_id":"(\d+)"', ours).group(1)
        swapped = ours.replace('"receiver_space_id":"%s"' % receiver,
                               '"receiver_space_id":"%s"' % sources[0])
        with self.assertRaises(AssertionError):
            translate(swapped, theirs, {})

    def test_poll_equals_callback(self):
        complete = self.snapshots["timestamped_complete"]
        self.assertEqual(self.polled, complete)
        self.assertEqual(complete.header.snapshot_status, "complete")
        self.assertEqual(complete.header.source_time_seconds.value, 12.5)
        channel = complete.channels[0]
        self.assertEqual((channel.anchor, channel.family),
                         ("root", "spacing"))
        # Same roundoff as the Unity golden: the exact input (2, 2, 0) is
        # admitted two ulps below 2 on x and one ulp above 2 on y.
        got = channel.admitted_position
        expected = (2.0, 2.0, 0.0)
        self.assertEqual(len(got), len(expected))
        for axis, (g, e) in zip("xyz", zip(got, expected)):
            self.assertLessEqual(
                _ulps_apart(g, e), _ADMITTED_OBSERVATION_ULPS,
                "admitted %s is more than %d ulps from %s" % (
                    axis, _ADMITTED_OBSERVATION_ULPS, e))


SUPPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                       "rich_realization_unity_support")


class RichSupportFixtureTest(unittest.TestCase):
    """Three channels (hand facing, hand affordance, root spacing), mapped
    support planes and 17-digit numbers. Serializer-only (see module doc)."""

    def test_multi_channel_capture_round_trips_byte_for_byte(self):
        with open(os.path.join(SUPPORT, "native-support-manifest.json")) as f:
            manifest = json.load(f)
        for case in manifest["cases"]:
            with open(os.path.join(SUPPORT, case["path"]), "rb") as f:
                payload = f.read()
            self.assertEqual(hashlib.sha256(payload).hexdigest(),
                             case["sha256"])
            snapshot = rich.from_rich_json(payload)
            self.assertEqual(len(snapshot.channels), 3)
            self.assertEqual(rich.to_rich_json(snapshot),
                             payload.decode("utf-8"))
            parsed = mg.parse_rich_realization(rich.to_rich_json(snapshot))
            self.assertEqual({(c.anchor, c.family) for c in parsed.channels},
                             {("hand", "facing"), ("hand", "affordance_pose"),
                              ("root", "spacing")})


DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                      "k1_rich_golden_driver")


class DriverGoldenTest(unittest.TestCase):
    """Goldens the e2e driver generated with the Unity serializer itself
    (the evaluation harness, not public: tests/fixtures/k1_rich_golden,
    cases 0-5 of 64; source.json names the serializer commit and runtime).
    Synthetic values that exercise the number formatter."""

    def test_driver_goldens_round_trip_byte_for_byte(self):
        with open(os.path.join(DRIVER, "cases.json")) as f:
            cases = json.load(f)
        self.assertEqual(len(cases), 6)
        for case in cases:
            with open(os.path.join(DRIVER, case["path"]), "rb") as f:
                payload = f.read()
            self.assertEqual(hashlib.sha256(payload).hexdigest(),
                             case["sha256"], case["path"])
            snapshot = rich.from_rich_json(payload)
            self.assertEqual(rich.to_rich_json(snapshot),
                             payload.decode("utf-8"), case["path"])
            mg.parse_rich_realization(payload)


class BindingRevisionK1Test(unittest.TestCase):
    """An engaged affordance contributor reaches the K1 wire with
    binding_revision_known true (MotionGen's strict admission needs it)."""

    def test_engaged_contributors_report_known_binding_revision(self):
        snapshots = []
        with capi.Session() as session:
            spaces = [session.add_space(), session.add_space()]
            rooms, actors = [], []
            seat_tops = ([(4, 4, .9), (6, 4, .9), (6, 6, .9), (4, 6, .9)],
                         [(2, 6, .4), (4, 6, .4), (4, 8, .4), (2, 8, .4)])
            for space, top in zip(spaces, seat_tops):
                room = space.add_element()
                room.set_shared_id(901)
                room.set_polygon(capi.GeometryRole.NAVIGABLE,
                                 [(0, 0), (10, 0), (10, 10), (0, 10)])
                seat = space.add_element(parent=room)
                seat.set_shared_id(902)
                seat.set_surface(top, capi.GeometryRole.SURFACE)
                actor = space.create_actor()
                actor.publish_embodied_pose(0.0, 0.0, 0.0, 0.0, 1)
                actor.add_anchor("root", 0, 0, 0)
                actor.add_anchor_with_position_dimension("body")
                actor.set_element(room)
                rooms.append(room)
                actors.append(actor)
            source, receiver = actors
            source.bind("root", capi.KernelType.SPACING)
            source.bind_affordance("body")
            rich.register_on_rich_realization(spaces[1], snapshots.append)
            token = rich.RichRealization.activate(spaces[1], receiver)
            for x in (5.0, 5.05):
                source.set_anchor_transform("body", 5, 4, 0.9, 0)  # touching the seat top
                source.update_state(x, 4, 0)
            token.retire()
            token.release()
            rich.register_on_rich_realization(spaces[1], None)
        complete = [s for s in snapshots
                    if s.header.snapshot_status == "complete"]
        self.assertTrue(complete)
        engaged = 0
        revisions = set()
        for snapshot in complete:
            parsed = mg.parse_rich_realization(rich.to_rich_json(snapshot))
            for channel in parsed.channels:
                for c in channel.contributors:
                    self.assertTrue(c.binding_revision_known)
                    self.assertTrue(c.binding_revision.known)
                    self.assertGreater(c.binding_revision.value, 0)
                    if channel.family == "affordance_pose":
                        revisions.add(c.binding_revision.value)
                        if (c.disposition == "included" and
                                c.has_mapped_support and
                                c.support_plane_to_world.known):
                            engaged += 1
        self.assertGreater(engaged, 0, "no engaged affordance contributor")
        self.assertEqual(len(revisions), 1)  # stable across ticks


if __name__ == "__main__":
    unittest.main()
