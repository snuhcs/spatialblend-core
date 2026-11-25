# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""ABI 4.7 embodiment registration through the ctypes package.

The spec is a FIXTURE (the numbers of the evaluation harness's (not public) SMPL-X sit-bone point, used only to exercise the schema)."""
import json
import math
import unittest

from spatialblend.api.python.capi import api as capi

SPEC = {
    "format": "spatialblend.embodiment", "schema_version": "1.0.0", "id": "fixture",
    "skeleton": {"joints": [{"name": "pelvis", "parent": -1}, {"name": "head", "parent": 0}]},
    "anchors": [
        {"name": "root", "bone": None, "families": ["Spacing"]},
        {"name": "pelvis", "bone": "pelvis", "offset": {"position": [0.0634, 0.0, -0.1860]},
         "families": ["AffordancePose"], "provenance": {"procedure": "fixture", "source": "test fixture"}}],
    "chain": {"rest_pose": "T", "provenance": {"procedure": "fixture", "source": "test fixture"},
              "joints": [{"name": "pelvis", "rest": {"position": [0, 0, 0], "rotation": [0, 0, 0, 1]}},
                         {"name": "head", "rest": {"position": [0.01, 0, 0.55], "rotation": [0, 0, 0, 1]}}]},
}


class EmbodimentCapiTest(unittest.TestCase):
    def setUp(self):
        self.session = capi.Session()
        self.space = self.session.add_space()
        self.actor = self.space.create_actor()

    def test_register_read_back_and_convert(self):
        self.assertIsNone(self.actor.embodiment_json())
        self.assertIsNone(self.actor.chain())
        self.actor.register_embodiment(json.dumps(SPEC))
        with self.assertRaises(ValueError):
            self.actor.register_embodiment(json.dumps(SPEC))   # once per actor
        back = json.loads(self.actor.embodiment_json())
        self.assertEqual(back["id"], "fixture")
        self.assertEqual(back["anchors"][1]["offset"]["position"], [0.0634, 0.0, -0.1860])
        chain = self.actor.chain()
        self.assertEqual([j["name"] for j in chain], ["pelvis", "head"])
        self.assertAlmostEqual(chain[1]["segment_length_m"], math.hypot(0.01, 0.55), places=15)
        self.assertEqual(self.actor.bone_id("head"), 1)

        yaw = 0.7
        q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        p, r = self.actor.anchor_pose_from_bone("pelvis", (1.5, -2.0, 0.636), q)
        self.assertAlmostEqual(p[0], 1.5 + math.cos(yaw) * 0.0634, places=14)
        self.assertAlmostEqual(p[2], 0.636 - 0.1860, places=14)
        bp, _ = self.actor.bone_pose_from_anchor("pelvis", p, r)
        for a, b in zip(bp, (1.5, -2.0, 0.636)):
            self.assertAlmostEqual(a, b, places=14)
        with self.assertRaises(ValueError):
            self.actor.anchor_pose_from_bone("root", (0, 0, 0), (0, 0, 0, 1))

    def test_bones_are_the_port(self):
        self.actor.register_embodiment(json.dumps(SPEC))
        pelvis_anchor = self.actor.anchor_id("pelvis")
        self.assertEqual(self.actor.publish_anchor_pose(pelvis_anchor, 1.0, 0.0, 0.0, 1.0, 1), -3)
        self.assertEqual(self.actor.publish_bone_pose(0, (1.0, 0.0, 0.6), (0, 0, 0, 1), 1.0, 1), 0)
        self.assertEqual(self.actor.publish_bone_pose(0, (1.0, 0.0, 0.6), (0, 0, 0, 1), 1.0, 1), -2)
        self.assertEqual(self.actor.publish_bone_pose(7, (0, 0, 0), (0, 0, 0, 1), 1.0, 2), -1)


if __name__ == "__main__":
    unittest.main()
