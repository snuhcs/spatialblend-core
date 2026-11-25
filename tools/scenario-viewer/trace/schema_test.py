# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Schema test: the producer's output must validate against spatialblend.scenario-trace v1.

1. the fixtures validate;
2. the validator rejects known breakages (a validator that accepts everything fails here);
3. the core exporter writes a fixed small set (the scope catalog and three matrix runs) and every file validates.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validate  # noqa: E402


def runfile(rel):
    root = os.path.join(os.environ.get("TEST_SRCDIR", ""), os.environ.get("TEST_WORKSPACE", "_main"))
    p = os.path.join(root, rel)
    return p if os.path.exists(p) else os.path.join(HERE, "..", "..", "..", rel)


SCHEMA_DIR = os.path.join(HERE, "schema")
FIXTURE = os.path.join(HERE, "fixtures", "synthetic-two-joint.json")
FIXTURE_110 = os.path.join(HERE, "fixtures", "synthetic-affordance.json")


class SchemaTest(unittest.TestCase):
    def test_fixture_validates(self):
        self.assertEqual(validate.validate_file(FIXTURE, SCHEMA_DIR), [])
        self.assertEqual(validate.validate_file(FIXTURE_110, SCHEMA_DIR), [])

    def _broken110(self, mutate):
        doc = json.load(open(FIXTURE_110))
        mutate(doc)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "broken.json")
            with open(p, "w") as f:
                json.dump(doc, f)
            return validate.validate_file(p, SCHEMA_DIR)

    def test_validator_rejects_110_breakage(self):
        def chan(doc, cid):
            return next(c for c in doc["channels"] if c["id"] == cid)

        def part(doc):
            return next(e for e in doc["spaces"][0]["elements"] if e.get("parts"))["parts"][0]
        cases = {
            "part transform not orthonormal": lambda d: part(d)["world_transform"].__setitem__(0, 2.0),
            "part transform bottom row": lambda d: part(d)["world_transform"].__setitem__(15, 2.0),
            "part with a short transform": lambda d: part(d)["world_transform"].pop(),
            "part footprint under 3 points": lambda d: part(d).__setitem__("footprint_local", [[0, 0], [1, 0]]),
            "part origin outside the enum": lambda d: part(d).__setitem__("origin", "guess"),
            "engaged code out of range": lambda d: chan(d, "source.anchor.hand_r.engaged")["data"]["codes"].__setitem__(0, 2),
            "anchor pose is not a pose": lambda d: chan(d, "source.anchor.hand_r").__setitem__("type", "scalar"),
            "unknown anchor name": lambda d: chan(d, "source.anchor.hand_r").__setitem__("id", "source.anchor.elbow"),
            "unknown anchor channel suffix": lambda d: chan(d, "source.anchor.hand_r.engaged").__setitem__("id", "source.anchor.hand_r.nope"),
            "skeleton parent after its child": lambda d: next(a for a in d["actors"] if a["skeleton"])["skeleton"]["joints"][0].__setitem__("parent", 1),
            "joint_count differs": lambda d: chan(d, "receiver.joints")["data"].__setitem__("joint_count", 3),
            "derived outside the enum": lambda d: chan(d, "receiver.joints").__setitem__("derived", "guess"),
            "expectation kind outside the enum": lambda d: d["expectations"][0].__setitem__("kind", "vibes"),
        }
        for name, m in cases.items():
            with self.subTest(name):
                self.assertTrue(self._broken110(m), f"validator accepted: {name}")

    def test_sidecar(self):
        good = {"format": "spatialblend.scenario-trace.sidecar", "schema_version": "1.1.0", "case_id": "T",
                "producer": {"name": "motiongen", "version": "0", "git_sha": "abc", "command": "x", "generated_at": "2026-10-19T00:00:00Z"},
                "actors": [{"id": "source", "skeleton": {"id": "j2", "joints": [{"name": "pelvis", "parent": -1}, {"name": "head", "parent": 0}],
                                                         "units": "m", "frame": "world z-up"}}],
                "channels": [{"id": "source.joints", "type": "joints", "space": "source", "actor": "source", "time_base": "tick",
                              "producer": {"name": "motiongen", "version": "0"}, "origin": "MotionGen output (not core output)",
                              "derived": "motiongen", "data": {"n": 2, "joint_count": 2, "p": [0.0] * 12, "q": [0.0, 0.0, 0.0, 1.0] * 4}}]}

        def run(mutate):
            doc = json.loads(json.dumps(good))
            mutate(doc)
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "x.sidecar.json")
                with open(p, "w") as f:
                    json.dump(doc, f)
                return validate.validate_file(p, SCHEMA_DIR)
        self.assertEqual(run(lambda d: None), [])
        for name, m in {
            "no producer.git_sha": lambda d: d["producer"].pop("git_sha"),
            "joints channel without a producer": lambda d: d["channels"][0].pop("producer"),
            "joints channel without derived": lambda d: d["channels"][0].pop("derived"),
            "joints channel without origin": lambda d: d["channels"][0].pop("origin"),
            "short p": lambda d: d["channels"][0]["data"]["p"].pop(),
            "joint_count differs from the skeleton": lambda d: d["channels"][0]["data"].__setitem__("joint_count", 3),
            "channel actor not in the sidecar": lambda d: d["channels"][0].__setitem__("actor", "nobody"),
            "skeleton units": lambda d: d["actors"][0]["skeleton"].__setitem__("units", "cm"),
        }.items():
            with self.subTest(name):
                self.assertTrue(run(m), f"validator accepted: {name}")

    def _broken(self, mutate):
        doc = json.load(open(FIXTURE))
        mutate(doc)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "broken.json")
            with open(p, "w") as f:
                json.dump(doc, f)
            return validate.validate_file(p, SCHEMA_DIR)

    def test_validator_rejects_breakage(self):
        def chan(doc, cid):
            return next(c for c in doc["channels"] if c["id"] == cid)
        cases = {
            "short pose column": lambda d: chan(d, "source.root")["data"]["p"].pop(),
            "non unit quaternion": lambda d: chan(d, "source.root")["data"]["q"].__setitem__(3, 5.0),
            "csr offsets not monotone": lambda d: chan(d, "source.scope")["data"]["in_scope"]["offsets"].__setitem__(3, 0),
            "missing required field": lambda d: d.pop("provenance"),
            "wrong version": lambda d: d.__setitem__("schema_version", "2.0.0"),
            "unknown actor root channel": lambda d: d["actors"][0].__setitem__("root_channel", "nope"),
            "enum code out of range": lambda d: chan(d, "receiver.proposal")["data"]["status"]["codes"].__setitem__(0, 99),
            "joints length": lambda d: chan(d, "receiver.joints")["data"]["p"].pop(),
            "event outside the case": lambda d: chan(d, "receiver.events")["data"]["events"].append({"tick": 999, "kind": "X"}),
        }
        for name, m in cases.items():
            with self.subTest(name):
                self.assertTrue(self._broken(m), f"validator accepted: {name}")

    def test_unknown_channel_type_is_not_an_error(self):
        def add(doc):
            doc["channels"].append({"id": "x.future", "type": "future_type", "space": None, "time_base": "tick", "data": {"anything": 1}})
        self.assertEqual(self._broken(add), [])

    def test_producer_output_validates(self):
        exe = runfile("spatialblend/sim/scenario/trace/scenario_trace_export")
        if not os.path.exists(exe):
            self.skipTest("needs the scenario-trace exporter, which is not part of this repository (an external producer)")
        catalog = runfile("spatialblend/space/testing/SCOPE_CASES.md")
        guards = runfile("spatialblend/sim/scenario/guards.csv")
        presets = runfile("spatialblend/sim/data/presets")
        with tempfile.TemporaryDirectory() as out:
            base = [exe, "--out", out, "--catalog", catalog, "--guards", guards, "--presets-dir", presets,
                    "--git-sha", "schema-test", "--command", "schema_test"]
            for suite, extra in (("scope-catalog", []), ("matrix-sample", ["--limit", "3"]), ("guards-table", []),
                                 ("affordance", ["--limit", "2"])):
                subprocess.run(base + ["--suite", suite] + extra, check=True)
            subprocess.run(base + ["--suite", "merge"], check=True)
            files = []
            for r, _, fs in os.walk(out):
                files += [os.path.join(r, f) for f in fs if f.endswith(".json")]
            self.assertGreater(len(files), 10)
            bad = []
            for f in files:
                bad += validate.validate_file(f, SCHEMA_DIR)
            self.assertEqual(bad, [])
            idx = json.load(open(os.path.join(out, "index.json")))
            self.assertTrue(any(c["suite"] == "scope-catalog" for c in idx["cases"]))
            self.assertTrue(any(c["suite"] == "matrix-sample" for c in idx["cases"]))
            aff = [c for c in idx["cases"] if c["suite"] == "affordance"]
            self.assertEqual(len(aff), 2)
            # the 1.1.0 content of an affordance case: parts, anchor channels, expectations of kind affordance
            with open(os.path.join(out, aff[0]["file"])) as f:
                case = json.load(f)
            self.assertEqual(case["schema_version"], "1.1.0")
            self.assertTrue(any(e.get("parts") for e in case["spaces"][0]["elements"]))
            ids = {c["id"] for c in case["channels"]}
            self.assertTrue({"source.anchor.pelvis", "source.anchor.pelvis.engaged", "receiver.anchor.pelvis.proposal"} <= ids)
            self.assertTrue(all(e["kind"] == "affordance" for e in case["expectations"]))
            import scope_context
            ctx = scope_context.extract(case)
            self.assertEqual(ctx["case_id"], case["case"]["id"])
            self.assertTrue(any(t["engaged"] for t in ctx["ticks"]))


if __name__ == "__main__":
    unittest.main()
