# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Tests of scope_context.py (spatialblend.scope-context v0): a hand-built case with known answers, the CLI, and (when
AFFORDANCE_EXPORT_DIR points at an export) every affordance case the exporter wrote."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scope_context  # noqa: E402

N = 6
IDENT = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def pose(ps, valid=None):
    q = [0, 0, 0, 1] * len(ps)
    return {"n": len(ps), "p": [c for p in ps for c in p], "q": q, "valid": valid or [1] * len(ps)}


def make_case():
    # a vertical board on the plane x = 2 (part frame: x_p -> world y, y_p -> world z, z_p -> world +x), polygon 1 x 0.5
    board_t = [0, 0, 1, 2.0,
               1, 0, 0, 5.0,
               0, 1, 0, 1.4,
               0, 0, 0, 1]
    poly = [[-0.5, -0.25], [0.5, -0.25], [0.5, 0.25], [-0.5, 0.25]]
    elems = [
        {"id": 1, "shared_id": 100, "name": "room", "kind": "room", "parent": None, "flags": [], "world_transform": IDENT, "footprint_local": poly},
        {"id": 10, "shared_id": 701, "name": "board", "kind": "obstacle", "parent": 1, "flags": [],
         "world_transform": [1, 0, 0, 2.0, 0, 1, 0, 5.0, 0, 0, 1, 0, 0, 0, 0, 1], "footprint_local": poly,
         "parts": [{"name": "board", "tag": "board", "world_transform": board_t, "footprint_local": poly, "origin": "core"}]},
        {"id": 11, "shared_id": 702, "name": "stool", "kind": "obstacle", "parent": 1, "flags": [], "world_transform": IDENT, "footprint_local": poly},
        {"id": 3, "shared_id": 110, "name": "actor", "kind": "person", "parent": 1, "flags": ["actor_element"], "world_transform": IDENT},
    ]
    # the hand: in front of the board at ticks 0-1 (x = 2.2), on the plane at ticks 2-4 (x = 2.0, y 5.1, z 1.45), away at 5
    hand = [[2.2, 5.1, 1.45], [2.1, 5.1, 1.45], [2.0, 5.1, 1.45], [2.0, 5.2, 1.5], [2.0, 5.2, 1.5], [2.3, 5.2, 1.5]]
    eng = [0, 0, 1, 1, 1, 0]
    scope = {"n": N, "mode": {"codes": [0] * N, "labels": ["OUTSIDE", "INTERIOR", "REPLAY"]}, "container": [1] * N, "transfer": [0] * N,
             "families": ["Spacing", "Facing", "AffordancePose"],
             "in_scope": {"offsets": [0, 3, 6, 9, 12, 15, 17],
                          # 3 rows per tick for ticks 0-4 (Spacing 701, Spacing 702, AffordancePose 701), 2 rows at tick 5
                          "family": [0, 0, 2] * 5 + [0, 0], "referent": [701, 702, 701] * 5 + [701, 702]},
             "dropped": {"offsets": [0] * (N + 1), "family": [], "referent": [], "reason": [], "reasons": []}}
    return {
        "format": "spatialblend.scenario-trace", "schema_version": "1.1.0",
        "provenance": {"producer": {"name": "test", "version": "1"}, "git_sha": "abc", "generated_at": "x", "schema_version": "1.1.0"},
        "case": {"id": "T-1", "suite": "affordance", "title": "t", "tags": {"input": "synthetic:affordance-tracks:1"}},
        "time": {"base": "tick", "n_ticks": N, "dt_s": 1 / 60, "t0_s": 0},
        "spaces": [{"id": "source", "role": "source", "elements": elems}],
        "actors": [{"id": "source", "space": "source", "element": 3, "role": "source_actor", "root_channel": "source.root"}],
        "channels": [
            {"id": "source.root", "type": "pose3d", "space": "source", "time_base": "tick", "data": pose([[1.0, 5.0, 0.0]] * N)},
            {"id": "source.scope", "type": "scope", "space": "source", "time_base": "tick", "data": scope},
            {"id": "source.anchor.hand_r", "type": "pose3d", "space": "source", "time_base": "tick", "data": pose(hand)},
            {"id": "source.anchor.hand_r.engaged", "type": "enum", "space": "source", "time_base": "tick",
             "data": {"n": N, "codes": eng, "labels": ["FREE", "ENGAGED"]}},
            {"id": "source.anchor.hand_r.part", "type": "enum", "space": "source", "time_base": "tick",
             "data": {"n": N, "codes": [0, 0, 1, 1, 1, 0], "labels": ["(none)", "board"]}},
        ],
    }


class ScopeContextTest(unittest.TestCase):
    def test_referents_and_parts(self):
        ctx = scope_context.extract(make_case())
        self.assertEqual(ctx["schema"], "spatialblend.scope-context")
        self.assertEqual(ctx["schema_version"], "0")
        self.assertEqual(ctx["case_id"], "T-1")
        self.assertEqual(ctx["actor"], "source")
        self.assertEqual(ctx["input"], "synthetic:affordance-tracks:1")
        self.assertEqual(sorted(ctx["referents"]), ["701", "702"])  # not the room, not the person
        part = ctx["referents"]["701"]["parts"][0]
        self.assertEqual(part["name"], "board")
        self.assertEqual(part["normal"], [1, 0, 0])  # the plane's +z, world +x
        self.assertEqual(ctx["referents"]["702"]["parts"], [])
        self.assertIn("ticks[].engaged[].referent", ctx["derived"])

    def test_ticks_scope_and_engagement(self):
        ctx = scope_context.extract(make_case())
        self.assertEqual(len(ctx["ticks"]), N)
        t0, t2, t5 = ctx["ticks"][0], ctx["ticks"][2], ctx["ticks"][5]
        self.assertEqual(t0["scope"]["mode"], "OUTSIDE")
        self.assertEqual(t0["scope"]["by_family"]["Spacing"], [701, 702])
        self.assertEqual(t0["scope"]["by_family"]["AffordancePose"], [701])
        self.assertEqual(t0["engaged"], [])
        self.assertEqual(len(t2["engaged"]), 1)
        e = t2["engaged"][0]
        self.assertEqual((e["anchor"], e["referent"], e["part"]), ("hand_r", 701, "board"))
        self.assertEqual(e["contact_p"], [2.0, 5.1, 1.45])
        self.assertEqual(e["source_height_m"], 0.0)
        self.assertEqual(t5["engaged"], [])
        self.assertEqual(t5["scope"]["by_family"]["AffordancePose"], [])
        self.assertEqual(ctx["ticks"][0]["anchors"]["hand_r"]["p"], [2.2, 5.1, 1.45])

    def test_embodiment_offsets_are_passed_through_not_applied(self):
        c = make_case()
        c["case"]["tags"]["embodiment_offsets"] = {"status": "provenance only", "m": {"hand_r": [0.1, 0.02, -0.03]}}
        ctx = scope_context.extract(c)
        self.assertEqual(ctx["embodiment_offsets"]["status"], "provenance only")
        self.assertNotIn("joint_p", ctx["ticks"][2]["anchors"]["hand_r"])
        self.assertEqual(ctx["ticks"][2]["anchors"]["hand_r"]["p"], [2.0, 5.1, 1.45])  # the contact point, untouched

    def test_is_deterministic_and_json(self):
        a = json.dumps(scope_context.extract(make_case()), sort_keys=True)
        b = json.dumps(scope_context.extract(make_case()), sort_keys=True)
        self.assertEqual(a, b)

    def test_invalid_anchor_ticks_carry_no_anchor(self):
        c = make_case()
        for ch in c["channels"]:
            if ch["id"] == "source.anchor.hand_r":
                ch["data"]["valid"] = [0, 0, 1, 1, 1, 0]
        ctx = scope_context.extract(c)
        self.assertNotIn("hand_r", ctx["ticks"][0]["anchors"])
        self.assertIn("hand_r", ctx["ticks"][2]["anchors"])

    def test_cli_writes_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = os.path.join(d, "case.json"), os.path.join(d, "ctx.json")
            with open(src, "w") as f:
                json.dump(make_case(), f)
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "scope_context.py"), src, "-o", out], check=True)
            with open(out) as f:
                self.assertEqual(json.load(f)["case_id"], "T-1")

    def test_refuses_other_formats(self):
        with self.assertRaises(ValueError):
            scope_context.extract({"format": "something else"})

    def test_exported_affordance_cases(self):
        d = os.environ.get("AFFORDANCE_EXPORT_DIR")
        if not d or not os.path.isdir(os.path.join(d, "cases")):
            self.skipTest("AFFORDANCE_EXPORT_DIR not set")
        n = 0
        for f in sorted(os.listdir(os.path.join(d, "cases"))):
            if not f.startswith("affordance__"):
                continue
            with open(os.path.join(d, "cases", f)) as fh:
                case = json.load(fh)
            ctx = scope_context.extract(case)
            uses = {c["id"][len("source.anchor."):] for c in case["channels"] if c["id"].startswith("source.anchor.") and c["id"].count(".") == 2}
            self.assertTrue(uses, f)
            for a in uses:
                eng = next(c for c in case["channels"] if c["id"] == f"source.anchor.{a}.engaged")["data"]["codes"]
                got = [i for i, t in enumerate(ctx["ticks"]) if any(e["anchor"] == a for e in t["engaged"])]
                self.assertEqual(got, [i for i, v in enumerate(eng) if v == 1], f"{f} {a}")
                for t in ctx["ticks"]:
                    for e in t["engaged"]:
                        self.assertIsNotNone(e["referent"], f"{f} tick {t['tick']}: no referent for {e['anchor']}")
            n += 1
        self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main()
