#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Writes the SYNTHETIC fixtures (not producer output; marked synthetic=true, never real data)."""
import json, math, os, sys
out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
N = 30
def qz(yaw): return [0, 0, math.sin(yaw / 2), math.cos(yaw / 2)]
def sq(x0, y0, x1, y1): return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
def elem(i, sid, name, kind, parent, ground, h=None, flags=None):
    return {"id": i, "shared_id": sid, "name": name, "kind": kind, "parent": parent, "flags": flags or [],
            "world_transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1], "footprint_local": ground, "ground": ground,
            "shape3d": {"extrude_height_m": h} if h else None}
def space(sid, role, s):
    return {"id": sid, "role": role, "navigable_root": 1,
            "free_space": {"outer": sq(0, 0, 10, 10), "holes": [sq(4, 4, 6, 6)], "origin": "synthetic"},
            "elements": [elem(1, 1, "room", "room", None, sq(0, 0, 10, s * 10)),
                         elem(10, 10, "table", "obstacle", 1, sq(4, 4, 6, 6), 0.75)]}
pos = [[1 + 0.1 * i, 2.0, 0.0] for i in range(N)]
def pose(ps, yawf):
    return {"n": N, "p": [c for p in ps for c in p], "q": [c for i in range(N) for c in qz(yawf(i))], "valid": [1] * N}
body = [[1.2 + 0.1 * i, 2.1, 0.0] for i in range(N)]
prop = [[1.0 + 0.1 * i, 2.0, 0.0] for i in range(N)]
doc = {
  "format": "spatialblend.scenario-trace", "schema_version": "1.0.0", "synthetic": True,
  "provenance": {"producer": {"name": "make_synthetic.py", "version": "1"}, "repo": "spatialblend-core", "git_sha": "synthetic",
                 "command": "make_synthetic.py", "generated_at": "1970-01-01T00:00:00Z", "schema_version": "1.0.0"},
  "case": {"id": "SYNTH-TWO-JOINT", "suite": "synthetic", "title": "Synthetic two-joint fixture (NOT real data)",
           "spec": {"rule": "Exercises every channel type and the reserved skeleton slot. Not producer output.", "level": "unit"}},
  "time": {"base": "tick", "n_ticks": N, "dt_s": 1 / 60, "t0_s": 0},
  "spaces": [space("source", "source", 1.0), space("receiver", "receiver", 1.0)],
  "correspondence": [{"source_space": "source", "receiver_space": "receiver", "source": 10, "receiver": 10, "shared_id": 10}],
  "actors": [
    {"id": "src", "space": "source", "element": 3, "role": "source_actor", "root_channel": "source.root",
     "body": {"capsule_radius_m": 0.2, "capsule_height_m": 1.7}, "skeleton": None, "joints_channel": None},
    {"id": "rx", "space": "receiver", "element": 3, "role": "receiver_body", "root_channel": "receiver.body",
     "body": {"capsule_radius_m": 0.2, "capsule_height_m": 1.7},
     "skeleton": {"format": "synthetic-2", "joints": ["pelvis", "head"], "parents": [-1, 0]}, "joints_channel": "receiver.joints"}],
  "channels": [
    {"id": "source.root", "type": "pose3d", "space": "source", "actor": "src", "time_base": "tick", "data": pose(pos, lambda i: 0.0)},
    {"id": "receiver.body", "type": "pose3d", "space": "receiver", "actor": "rx", "time_base": "tick", "data": pose(body, lambda i: 0.0)},
    {"id": "receiver.proposal", "type": "proposal", "space": "receiver", "actor": "rx", "time_base": "tick",
     "data": {"n": N, "status": {"codes": [1] * N, "labels": ["NONE", "EXACT", "INEXACT", "NO_SAMPLE", "NOT_CONVERGED"]},
              "cause": {"v": [1] + [0] * (N - 1), "bits": [{"bit": 0, "name": "FIRST_PROPOSAL"}]},
              "flags": {"v": [0] * N, "bits": [{"bit": 5, "name": "TRANSFER"}]},
              "p": [c for p in prop for c in p], "q": [c for i in range(N) for c in qz(0)], "valid": [1] * N, "err": {}}},
    {"id": "receiver.joints", "type": "joints", "space": "receiver", "actor": "rx", "time_base": "tick",
     "doc": "SYNTHETIC two-joint skeleton",
     "data": {"n": N, "joint_count": 2,
              "p": [c for i in range(N) for c in (body[i][0], body[i][1], 0.9, body[i][0], body[i][1], 1.6)],
              "q": [c for i in range(N) for c in (qz(0) + qz(0))]}},
    {"id": "source.scope", "type": "scope", "space": "source", "time_base": "tick",
     "data": {"n": N, "mode": {"codes": [0] * N, "labels": ["OUTSIDE", "INTERIOR", "REPLAY"]}, "container": [1] * N, "transfer": [0] * N,
              "families": ["Spacing", "Facing", "AffordancePose"],
              "in_scope": {"offsets": list(range(N + 1)), "family": [0] * N, "referent": [10] * N},
              "dropped": {"offsets": [0] * (N + 1), "family": [], "referent": [], "reason": [], "reasons": []}}},
    {"id": "receiver.relations", "type": "relation_set", "space": "receiver", "time_base": "tick",
     "data": {"n": N, "offsets": list(range(N + 1)), "families": ["Spacing", "Facing", "AffordancePose"], "family": [0] * N,
              "referent": [10] * N, "weight": [1.0] * N, "R_src": [0.5] * N, "d_m": [1.0] * N,
              "contact_p": [c for i in range(N) for c in (4.0, 4.0, 0.0)], "R_rx": [0.5] * N, "err_rx": [0.0] * N}},
    {"id": "source.J", "type": "scalar", "space": "receiver", "time_base": "tick", "data": {"n": N, "v": [0.1 / (i + 1) for i in range(N)]}},
    {"id": "receiver.events", "type": "event", "space": "receiver", "time_base": "tick",
     "data": {"events": [{"tick": 0, "kind": "FIRST_PROPOSAL"}, {"tick": 15, "kind": "SYNTHETIC_MARK"}]}}],
  "expectations": [{"id": "e1", "kind": "scope_mode", "tick": 0, "expected": "OUTSIDE"}],
  "results": [{"expectation": "e1", "actual": "OUTSIDE", "pass": True, "test": "synthetic"}],
  "metrics": {"row": {"frames": N}, "guards": []},
  "missing": ["everything: this is a synthetic fixture"]
}
json.dump(doc, open(os.path.join(out, "synthetic-two-joint.json"), "w"), separators=(",", ":"))
print("wrote", os.path.join(out, "synthetic-two-joint.json"))

# ---- schema 1.1.0 fixture: element parts, anchor channels, the 1.1.0 skeleton form, channel producer / derived ----
import copy
d2 = copy.deepcopy(doc)
d2["schema_version"] = "1.1.0"
d2["provenance"]["schema_version"] = "1.1.0"
d2["case"] = {"id": "SYNTH-AFFORDANCE", "suite": "synthetic", "title": "Synthetic affordance fixture (NOT real data)",
              "spec": {"rule": "Exercises schema 1.1.0: element parts, anchor channels, expectations of kind affordance, the skeleton form, channel producer and derived. Not producer output.", "level": "unit"}}
for sp in d2["spaces"]:
    for e in sp["elements"]:
        if e["name"] == "table":
            e["parts"] = [{"name": "top", "tag": "top", "origin": "scene-spec",
                           "world_transform": [1,0,0,5, 0,1,0,5, 0,0,1,0.75, 0,0,0,1],
                           "footprint_local": sq(-1, -1, 1, 1)}]
for a in d2["actors"]:
    if a["id"] == "rx":
        a["skeleton"] = {"id": "synthetic-2", "joints": [{"name": "pelvis", "parent": -1}, {"name": "head", "parent": 0}],
                         "units": "m", "frame": "world z-up"}
for c in d2["channels"]:
    if c["id"] == "receiver.joints":
        c["producer"] = {"name": "synthetic", "version": "1", "git_sha": "synthetic"}
        c["origin"] = "synthetic fixture"
        c["derived"] = "display-ik"
hand = [[5.0 + 0.001 * i, 5.0, 0.75 + 0.1 * max(0, 10 - i)] for i in range(N)]
d2["channels"] += [
    {"id": "source.anchor.hand_r", "type": "pose3d", "space": "source", "actor": "src", "time_base": "tick", "data": pose(hand, lambda i: 0.0)},
    {"id": "source.anchor.hand_r.engaged", "type": "enum", "space": "source", "time_base": "tick",
     "data": {"n": N, "codes": [1 if i >= 10 else 0 for i in range(N)], "labels": ["FREE", "ENGAGED"]}},
    {"id": "receiver.anchor.hand_r.proposal", "type": "pose3d", "space": "receiver", "actor": "rx", "time_base": "tick",
     "data": {"n": N, "p": [c for h in hand for c in h], "q": [c for i in range(N) for c in qz(0)], "valid": [1 if i >= 10 else 0 for i in range(N)]}},
    {"id": "receiver.anchor.hand_r.engaged", "type": "enum", "space": "receiver", "time_base": "tick",
     "data": {"n": N, "codes": [1 if i >= 10 else 0 for i in range(N)], "labels": ["FREE", "ENGAGED"]}},
]
d2["expectations"] = [{"id": "E4:hand_r", "kind": "affordance", "tick": 10, "expected": "within 1e-9 m of the oracle", "family": "AffordancePose", "text": "synthetic"}]
d2["results"] = [{"expectation": "E4:hand_r", "actual": {"text": "synthetic", "residue": 0.0, "unit": "mm"}, "pass": True, "test": "synthetic"}]
json.dump(d2, open(os.path.join(out, "synthetic-affordance.json"), "w"), separators=(",", ":"))
print("wrote", os.path.join(out, "synthetic-affordance.json"))
