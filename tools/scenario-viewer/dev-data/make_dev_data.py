#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Generates SYNTHETIC dev data for the viewer (not producer output). Writes dev-data/data/.

Every file is marked synthetic=true. The point is to exercise the viewer: plain 2D cases, scope/relations/
proposals, continuous channels, an unknown channel type, a failing expectation and guard, a provenance
SHA mismatch, a big table, and the two-joint skeleton fixture from tools/scenario-viewer/trace/fixtures.
"""
import json, math, os, random, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
FIX = os.path.join(HERE, "..", "trace", "fixtures", "synthetic-two-joint.json")
SHA = "dev0000000000000000000000000000000000000"
SCHEMA = "1.0.0"


def qz(yaw): return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
def box(x0, y0, x1, y1): return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
def r3(v): return round(v, 3)
def clamp(x, a, b): return max(a, min(b, x))
def prov(sha=SHA):
    return {"producer": {"name": "make_dev_data.py", "version": "1"}, "repo": "spatialblend-core", "git_sha": sha,
            "command": "make_dev_data.py", "generated_at": "2026-10-13T00:00:00Z", "schema_version": SCHEMA}


def elem(i, sid, name, kind, parent, ground, h=None, flags=None, local=None, wt=None):
    e = {"id": i, "shared_id": sid, "name": name, "kind": kind, "parent": parent, "flags": flags or [],
         "world_transform": wt or [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
         "footprint_local": local or ground, "shape3d": {"extrude_height_m": h} if h else None}
    if ground is not None: e["ground"] = ground
    return e


def make_spaces(W, Hh, sx, sy):
    """source (sx=sy=1) and receiver (scaled/moved layout) with the same shared ids."""
    def sp(role, ox, oy, k):
        els = [
            elem(1, 1, "living room", "room", None, box(0, 0, 10 * k, 8 * k)),
            elem(2, 2, "sofa", "object", 1, box(1 * k, 5.5 * k, 3.2 * k, 7 * k)),
            elem(3, 3, "table", "obstacle", 1, box(4.5 * k, 3 * k, 6 * k, 4.5 * k), h=0.75),
            elem(4, 4, "sofa zone", "interaction_region", 1, box(0.5 * k, 4.5 * k, 3.7 * k, 7.5 * k)),
            elem(5, 5, "tv", "object", 1, box(8.7 * k, 3 * k, 9.2 * k, 5 * k), h=0.6),
            elem(6, 6, "door", "surface", 1, box(4.5 * k, 7.6 * k, 5.5 * k, 8.0 * k), flags=["transfer_leaf"]),
            elem(8, 8, "wall", "border", 1, box(0, 0, 10 * k, 0.06)),
        ]
        if role == "source":
            els.append(elem(7, 7, "shelf", "object", 1, box(7 * k, 0.2, 9.5 * k, 0.8), h=1.9))
        else:  # receiver: no ground image exported for the shelf -> viewer falls back to footprint_local x world_transform
            els.append(elem(7, 7, "shelf", "object", 1, None, local=box(0, 0, 2.2, 0.5), wt=[1, 0, 0, 5.6, 0, 1, 0, 0.2, 0, 0, 1, 0, 0, 0, 0, 1]))
        holes = [box(4.5 * k, 3 * k, 6 * k, 4.5 * k), box(1 * k, 5.5 * k, 3.2 * k, 7 * k), box(8.7 * k, 3 * k, 9.2 * k, 5 * k)]
        return {"id": role, "role": role, "navigable_root": 1,
                "free_space": {"outer": box(0, 0, 10 * k, 8 * k), "holes": holes, "origin": "synthetic"}, "elements": els}
    return [sp("source", 0, 0, 1.0), sp("receiver", 0, 0, 0.8)]


def path_pts(n):
    keys = [(1.5, 1.5), (3.0, 2.0), (4.0, 5.2), (6.8, 5.4), (7.8, 3.9), (4.8, 6.4), (2.4, 4.9)]
    pts = []
    segs = len(keys) - 1
    for k in range(n):
        u = k / max(1, n - 1) * segs
        i = min(int(u), segs - 1); f = u - i
        a, b = keys[i], keys[i + 1]
        pts.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    return pts


def pose_ch(cid, space, actor, pts, yaws, valid=None, continuous=False, z=0.0):
    n = len(pts)
    ch = {"id": cid, "type": "pose3d", "space": space, "time_base": "tick",
          "data": {"n": n, "p": [c for i in range(n) for c in (r3(pts[i][0]), r3(pts[i][1]), z)],
                   "q": [round(c, 6) for i in range(n) for c in qz(yaws[i])], "valid": valid or [1] * n}}
    if actor: ch["actor"] = actor
    if continuous: ch["continuous"] = True
    return ch


def csr(rows_per_tick):
    off = [0]
    for r in rows_per_tick: off.append(off[-1] + len(r))
    return off


def nearest_on_box(p, b):
    return (clamp(p[0], b[0][0], b[2][0]), clamp(p[1], b[0][1], b[2][1]))


def main_case(cid, suite, n, title, rule, seed, sha=SHA, unknown=False, fail=True, continuous=False, with_proposal=True):
    rnd = random.Random(seed)
    pts = path_pts(n)
    yaws = []
    for i in range(n):
        a, b = pts[max(0, i - 1)], pts[min(n - 1, i + 1)]
        yaws.append(math.atan2(b[1] - a[1], b[0] - a[0]))
    spaces = make_spaces(10, 8, 1, 1)
    k = 0.8
    rpts = [(x * k + 0.06 * math.sin(i / 7), y * k + 0.05) for i, (x, y) in enumerate(pts)]
    prop_pts = [(x * k, y * k) for x, y in pts]
    modes = [0 if i < n * 0.25 else 1 if i < n * 0.83 else 2 for i in range(n)]
    tab, sofa, tv, shelf = box(4.5, 3, 6, 4.5), box(1, 5.5, 3.2, 7), box(8.7, 3, 9.2, 5), box(7, .2, 9.5, .8)
    tabk, sofak = box(4.5 * k, 3 * k, 6 * k, 4.5 * k), box(1 * k, 5.5 * k, 3.2 * k, 7 * k)
    in_rows, dr_rows, srel, rrel = [], [], [], []
    for i in range(n):
        ir, dr = [], []
        if modes[i] >= 1:
            ir += [(0, 3), (0, 2), (1, 5)]
            if i >= n * 0.5: ir.append((2, 2))
            dr += [(0, 7, 0)]
            if i % 20 < 10: dr.append((1, 3, 1))
        in_rows.append(ir); dr_rows.append(dr)
        sr, rr = [], []
        p = pts[i]
        if modes[i] >= 1:
            w = clamp((i - n * 0.25) / 15, 0, 1)
            c = nearest_on_box(p, tab); d = math.hypot(p[0] - c[0], p[1] - c[1])
            sr.append(dict(fam=0, ref=3, w=w, R=d / 2.0, d=d, contact=(c[0], c[1], 0.0), look=None, vert=None, ang=None, occ=None))
            rc = nearest_on_box((p[0] * k, p[1] * k), tabk)
            rr.append(dict(fam=0, ref=3, w=w, R=d / 2.0, Rrx=d / 2.0 + 0.05 * math.exp(-i / 30) * math.sin(i / 3), pt=(rc[0], rc[1], 0.0)))
            tvc = (8.95, 4.0, 1.0)
            ang = 0.15 * math.sin(i / 9) * math.exp(-(i - n * 0.25) / 40)
            sr.append(dict(fam=1, ref=5, w=clamp(w * 0.8, 0, 1), R=0.0, d=None, contact=None, look=tvc, vert=2 if i % 17 < 12 else -1, ang=ang, occ=1 if 0.55 * n < i < 0.66 * n else 0))
            rr.append(dict(fam=1, ref=5, w=clamp(w * 0.8, 0, 1), R=0.0, Rrx=ang * 0.4, pt=(8.95 * k, 4.0 * k, 1.0)))
            if i >= n * 0.5:
                c2 = nearest_on_box(p, sofa); d2 = math.hypot(p[0] - c2[0], p[1] - c2[1])
                sr.append(dict(fam=2, ref=2, w=clamp((i - n * 0.5) / 10, 0, 1), R=d2 / 3, d=d2, contact=(c2[0], c2[1], 0.45), look=None, vert=None, ang=None, occ=None))
                c2k = nearest_on_box((p[0] * k, p[1] * k), sofak)
                rr.append(dict(fam=2, ref=2, w=clamp((i - n * 0.5) / 10, 0, 1), R=d2 / 3, Rrx=d2 / 3 + 0.02, pt=(c2k[0], c2k[1], 0.45)))
        srel.append(sr); rrel.append(rr)

    def rel_data(rows, receiver):
        flat = [r for t in rows for r in t]
        o = csr(rows)
        d = {"n": n, "offsets": o, "families": ["Spacing", "Facing", "AffordancePose"], "family": [r["fam"] for r in flat],
             "referent": [r["ref"] for r in flat], "weight": [round(r["w"], 6) for r in flat]}
        if not receiver:
            d.update({"R_src": [round(r["R"], 6) for r in flat], "d_m": [None if r["d"] is None else r3(r["d"]) for r in flat],
                      "contact_p": [c for r in flat for c in (r["contact"] or (None, None, None))],
                      "look_at_p": [c for r in flat for c in (r["look"] or (None, None, None))],
                      "look_at_vertex": [r["vert"] for r in flat], "angle_err_rad": [None if r["ang"] is None else round(r["ang"], 6) for r in flat],
                      "occluded": [r["occ"] for r in flat]})
        else:
            d.update({"R_src": [round(r["R"], 6) for r in flat], "R_rx": [round(r["Rrx"], 6) for r in flat],
                      "err_rx": [round(abs(r["Rrx"] - r["R"]), 6) for r in flat],
                      "R_rx_body": [round(r["Rrx"] + 0.02, 6) for r in flat],
                      "rx_point_p": [r3(c) for r in flat for c in r["pt"]]})
        return d

    status_labels = ["NONE", "EXACT", "INEXACT", "NO_SAMPLE", "NOT_CONVERGED"]
    st = []
    valid = []
    for i in range(n):
        if 0.5 * n <= i < 0.5 * n + 3: st.append(3); valid.append(0)
        elif 0.35 * n <= i < 0.45 * n: st.append(2); valid.append(1)
        elif 0.75 * n <= i < 0.78 * n: st.append(4); valid.append(1)
        else: st.append(1); valid.append(1)
    cause = [1 if i == 0 else 2 if i == int(n * 0.37) else 4 if i == int(n * 0.58) else 0 for i in range(n)]
    flags = [32 if int(n * 0.83) <= i else 0 for i in range(n)]
    pp = [(prop_pts[i] if valid[i] else (0.0, 0.0)) for i in range(n)]
    J = [round(0.5 * math.exp(-(i % 25) / 6) + 0.01 * rnd.random(), 5) for i in range(n)]
    stepr = [round(0.9 + 0.2 * math.sin(i / 5) ** 2 + (0.8 if i == int(n * 0.58) else 0), 4) for i in range(n)]
    ev = [{"tick": 0, "kind": "SEND"}, {"tick": int(n * 0.25), "kind": "MODE_CHANGE", "data": {"to": "INTERIOR"}},
          {"tick": int(n * 0.37), "kind": "WEIGHT_SWITCH"}, {"tick": int(n * 0.58), "kind": "FACING_ROOT_SWITCH"},
          {"tick": int(n * 0.58), "kind": "GUARD_FAIL", "data": {"column": "step_ratio_max"}},
          {"tick": int(n * 0.83), "kind": "MODE_CHANGE", "data": {"to": "REPLAY"}}, {"tick": int(n * 0.83), "kind": "TRANSFER"},
          {"tick": int(n * 0.88), "kind": "TELEPORT_REQUEST"}, {"tick": int(n * 0.5), "kind": "STATUS_CHANGE", "data": {"to": "NO_SAMPLE"}}]
    body_after = [(prop_pts[i][0] + 0.03, prop_pts[i][1]) for i in range(n)]
    replay_valid = [1 if i >= n * 0.83 else 0 for i in range(n)]
    channels = [
        pose_ch("source.root", "source", "src", pts, yaws, continuous=continuous),
        {"id": "source.sent", "type": "enum", "space": "source", "time_base": "tick", "data": {"n": n, "codes": [0 if i % 4 else 1 for i in range(n)], "labels": ["SENT", "HELD"]}},
        {"id": "source.scope", "type": "scope", "space": "source", "time_base": "tick", "data": {
            "n": n, "mode": {"codes": modes, "labels": ["OUTSIDE", "INTERIOR", "REPLAY"]}, "container": [1 if m else None for m in modes],
            "transfer": [1 if i >= int(n * 0.83) else 0 for i in range(n)], "yaw_source": {"codes": [0] * n, "labels": ["BODY", "ROOT"]},
            "families": ["Spacing", "Facing", "AffordancePose"],
            "in_scope": {"offsets": csr(in_rows), "family": [f for t in in_rows for f, _ in t], "referent": [r for t in in_rows for _, r in t]},
            "dropped": {"offsets": csr(dr_rows), "family": [f for t in dr_rows for f, _, _ in t], "referent": [r for t in dr_rows for _, r, _ in t],
                        "reason": [q for t in dr_rows for _, _, q in t], "reasons": ["beyond horizon radius", "occluded by table"]}}},
        {"id": "source.relations", "type": "relation_set", "space": "source", "time_base": "tick", "origin": "synthetic", "data": rel_data(srel, False)},
        {"id": "receiver.relations", "type": "relation_set", "space": "receiver", "time_base": "tick", "origin": "synthetic", "data": rel_data(rrel, True)},
        pose_ch("receiver.body", "receiver", "rx", rpts, yaws),
        pose_ch("receiver.body_after", "receiver", None, body_after, yaws),
        pose_ch("receiver.replay", "receiver", None, prop_pts, yaws, valid=replay_valid),
        {"id": "receiver.proposal", "type": "proposal", "space": "receiver", "actor": "rx", "time_base": "tick", "data": {
            "n": n, "status": {"codes": st, "labels": status_labels},
            "cause": {"v": cause, "bits": [{"bit": 0, "name": "FIRST_PROPOSAL"}, {"bit": 1, "name": "WEIGHT_SWITCH"}, {"bit": 2, "name": "FACING_ROOT_SWITCH"}]},
            "flags": {"v": flags, "bits": [{"bit": 5, "name": "TRANSFER"}]},
            "p": [r3(c) for i in range(n) for c in (pp[i][0], pp[i][1], 0.0)], "q": [round(c, 6) for i in range(n) for c in qz(yaws[i])], "valid": valid,
            "err": {"spacing_target": [round(0.02 + 0.01 * math.sin(i / 6), 5) for i in range(n)], "facing_target": [round(0.1 * math.exp(-i / 50), 5) for i in range(n)], "scale_m": [0.8] * n}}},
        {"id": "receiver.solver.J", "type": "scalar", "space": "receiver", "time_base": "tick", "data": {"n": n, "v": J}},
        {"id": "receiver.solver.J_before", "type": "scalar", "space": "receiver", "time_base": "tick", "data": {"n": n, "v": [round(j * 2.1, 5) for j in J]}},
        {"id": "receiver.solver.iters", "type": "scalar", "space": "receiver", "time_base": "tick", "data": {"n": n, "v": [1 + (i % 7) for i in range(n)]}},
        {"id": "receiver.solver.stop", "type": "enum", "space": "receiver", "time_base": "tick", "data": {"n": n, "codes": [0 if i % 9 else 1 for i in range(n)], "labels": ["CONVERGED", "MAX_ITERS"]}},
        {"id": "receiver.step_ratio", "type": "scalar", "space": "receiver", "time_base": "tick", "data": {"n": n, "v": stepr}},
        {"id": "receiver.events", "type": "event", "space": "receiver", "time_base": "tick", "data": {"events": sorted(ev, key=lambda e: e["tick"])}},
        {"id": "case.metrics", "type": "metrics_row", "time_base": "tick", "data": {"row": {"frames": n, "step_ratio_max": round(max(stepr), 3)}}},
    ]
    if not with_proposal:
        channels = [c for c in channels if c["type"] != "proposal" and not c["id"].startswith("receiver.")]
        channels.append(pose_ch("receiver.body", "receiver", "rx", rpts, yaws))
    if unknown:
        channels.append({"id": "source.gaze_cone", "type": "gaze_cone_v2", "space": "source", "time_base": "tick", "doc": "a channel type from a later minor version", "data": {"n": n, "cone": [0.1] * n}})
    exp = [
        {"id": "e-mode", "kind": "scope_mode", "tick": int(n * 0.4), "expected": "INTERIOR"},
        {"id": "e-set", "kind": "scope_set", "tick": int(n * 0.4), "family": "Spacing", "expected": [3, 2]},
        {"id": "e-status", "kind": "status", "tick": int(n * 0.76), "expected": "EXACT"},
        {"id": "e-yaw", "kind": "yaw_source", "tick": 0, "expected": "BODY"},
        {"id": "e-step", "kind": "guard", "expected": {"column": "step_ratio_max", "max": 1.5}, "text": "step ratio stays under 1.5"},
        {"id": "e-note", "kind": "text", "text": "Door transfers the player to the replay mode"},
    ]
    res = [
        {"expectation": "e-mode", "actual": "INTERIOR", "pass": True, "test": "scope_test.Interior"},
        {"expectation": "e-set", "actual": [3, 2], "pass": True, "test": "scope_test.SpacingSet"},
        {"expectation": "e-status", "actual": "NOT_CONVERGED", "pass": False if fail else True, "test": "proposal_test.Status", "note": "solver stopped at the iteration cap"},
        {"expectation": "e-yaw", "actual": "BODY", "pass": True, "test": "scope_test.Yaw"},
        {"expectation": "e-step", "actual": round(max(stepr), 3), "pass": False if fail else True, "test": "guard_test.StepRatio"},
    ]
    return {
        "format": "spatialblend.scenario-trace", "schema_version": SCHEMA, "synthetic": True, "provenance": prov(sha),
        "case": {"id": cid, "suite": suite, "title": title, "spec": {"rule": rule, "citations": ["spec §3.2", "ABI note 14"], "source": "synthetic/make_dev_data.py", "level": "scenario", "plan_key": "P-" + cid},
                 "tags": {"families": ["Spacing", "Facing", "AffordancePose"], "modes": ["OUTSIDE", "INTERIOR", "REPLAY"], "scene": "living-room"}},
        "time": {"base": "tick", "n_ticks": n, "dt_s": 1 / 30, "t0_s": 0.0},
        "spaces": spaces,
        "correspondence": [{"source_space": "source", "receiver_space": "receiver", "source": i, "receiver": i, "shared_id": i} for i in (2, 3, 5, 6)],
        "actors": [
            {"id": "src", "space": "source", "element": None, "role": "source_actor", "root_channel": "source.root", "body": {"capsule_radius_m": 0.22, "capsule_height_m": 1.7}, "skeleton": None, "joints_channel": None},
            {"id": "rx", "space": "receiver", "element": None, "role": "receiver_body", "root_channel": "receiver.body", "body": {"capsule_radius_m": 0.22, "capsule_height_m": 1.7}, "skeleton": None, "joints_channel": None}],
        "channels": channels,
        "expectations": exp, "results": res,
        "metrics": {"row": {"frames": n, "step_ratio_max": round(max(stepr), 3), "spacing_err_max": 0.04, "facing_err_deg_p95": 3.2},
                    "guards": [{"column": "step_ratio_max", "bound": 1.5, "value": round(max(stepr), 3), "pass": False if fail else True},
                               {"column": "spacing_err_max", "bound": 0.1, "value": 0.04, "pass": True},
                               {"column": "facing_err_deg_p95", "bound": 5, "value": 3.2, "pass": True}]},
        "missing": ["receiver visibility witness: the core exposes no occlusion state for the receiver side (synthetic note)"],
    }


def enrich_parity(d):
    """additive conventions: parity.* scope, parity.agree / eval.* scalars, extra receiver poses, bounds, scope border"""
    n = d["time"]["n_ticks"]; ch = d["channels"]
    sc = next(c for c in ch if c["id"] == "source.scope")
    sc["data"]["border"] = [1 if i % 40 < 20 else 0 for i in range(n)]
    ch.append({"id": "parity.v1.scope", "type": "scope", "space": "source", "time_base": "tick", "data": {
        "n": n, "mode": {"codes": [None] * n, "labels": []}, "families": ["Spacing"],
        "in_scope": {"offsets": [i for i in range(n + 1)], "family": [0] * n, "referent": [3 if i % 2 else 2 for i in range(n)]},
        "dropped": {"offsets": [0] * (n + 1), "family": [], "referent": [], "reason": [], "reasons": []}}})
    ch.append({"id": "parity.agree", "type": "scalar", "space": "receiver", "time_base": "tick", "data": {"n": n, "v": [1 if i % 9 else 0 for i in range(n)]}})
    ch.append({"id": "eval.v1.R_receiver.100", "type": "scalar", "space": "receiver", "time_base": "tick", "data": {"n": n, "v": [round(0.5 + 0.1 * math.sin(i / 8), 4) for i in range(n)]}})
    ch.append({"id": "eval.source.R.100", "type": "scalar", "space": "receiver", "time_base": "tick", "data": {"n": n, "v": [round(0.5 + 0.1 * math.sin(i / 8 + 0.3), 4) for i in range(n)]}})
    base = next(c for c in ch if c["id"] == "receiver.body")["data"]
    for cid, dx in (("parity.v1.replay", 0.4), ("eval.v1.receiver_trace", -0.4)):
        ch.append({"id": cid, "type": "pose3d", "space": "receiver", "time_base": "tick", "data": {"n": n, "p": [round(v + (dx if j % 3 == 0 else 0), 3) for j, v in enumerate(base["p"])], "q": base["q"], "valid": [1 if i % 11 else 0 for i in range(n)]}})
    next(c for c in ch if c["id"] == "receiver.step_ratio")["bounds"] = [{"value": 1.5, "label": "owner bound 1.5"}]
    for c in ch:
        if c["id"] == "receiver.events": c["space"] = None

def enrich_compare(d):
    """eval.<engine>.receiver_body channels + R channels, for compare mode"""
    n = d["time"]["n_ticks"]; ch = d["channels"]
    base = next(c for c in ch if c["id"] == "receiver.body")["data"]
    ch.append({"id": "eval.source.R.3", "type": "scalar", "space": "source", "time_base": "tick", "data": {"n": n, "v": [round(0.6 + 0.1 * math.sin(i / 9), 4) for i in range(n)]}})
    ch.append({"id": "eval.v2.R_receiver.3", "type": "scalar", "space": "receiver", "time_base": "tick", "data": {"n": n, "v": [round(0.6 + 0.1 * math.sin(i / 9) + 0.01, 4) for i in range(n)]}})
    for eng, dx, amp in (("v1", 0.5, 0.15), ("v1eqw", -0.6, 0.3)):
        ch.append({"id": f"eval.{eng}.receiver_body", "type": "pose3d", "space": "receiver", "time_base": "tick", "data": {"n": n, "p": [round(v + (dx if j % 3 == 0 else 0), 3) for j, v in enumerate(base["p"])], "q": base["q"], "valid": [1] * n}})
        ch.append({"id": f"eval.{eng}.R_receiver.3", "type": "scalar", "space": "receiver", "time_base": "tick", "data": {"n": n, "v": [round(0.6 + 0.1 * math.sin(i / 9) + amp * abs(math.sin(i / 5)), 4) for i in range(n)]}})

def spec_only():
    return {"format": "spatialblend.scenario-trace", "schema_version": SCHEMA, "synthetic": True, "provenance": prov(),
            "case": {"id": "SPEC-ONLY", "suite": "scope-catalog", "title": "Spec-only case (no ticks, no spaces)", "spec": {"rule": "A rule with nothing to draw.", "level": "unit"}},
            "time": {"base": "tick", "n_ticks": 0, "dt_s": 1 / 60, "t0_s": 0}, "spaces": [], "actors": [], "channels": [],
            "expectations": [{"id": "x1", "kind": "text", "text": "documented only"}], "results": [{"expectation": "x1", "pass": None, "test": "none"}]}

def unit_case():
    d = main_case("UNIT-ONE-SPACE", "scope-catalog", 3, "Unit case: one space, poses not seconds", "dt_s = 1: ticks are pose indices.", 7, fail=False, with_proposal=False)
    d["time"]["dt_s"] = 1.0
    d["spaces"] = [d["spaces"][0]]; d["correspondence"] = []
    d["actors"] = [d["actors"][0]]
    d["channels"] = [c for c in d["channels"] if c["id"] in ("source.root", "source.scope", "source.relations", "source.sent")]
    for c in d["channels"]:
        if c["id"] == "source.scope": c["data"].pop("border", None)
    return d

def wr(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(doc, f, separators=(",", ":"))
    return os.path.getsize(path)


def main():
    if os.path.isdir(OUT): shutil.rmtree(OUT)
    cases = [
        ("scope-catalog", main_case("SC-INTERIOR-WALK", "scope-catalog", 120, "Interior walk with all three families", "Inside a room the source scope holds the container's referents per family; the receiver reproduces R within the guard.", 1)),
        ("scope-catalog", main_case("SC-FACING-FUTURE", "scope-catalog", 48, "Case with a channel type from the future", "Unknown channel types are listed and ignored.", 2, sha="dev1111111111111111111111111111111111111", unknown=True, fail=False)),
        ("scope-catalog", main_case("SC-CONTINUOUS", "scope-catalog", 12, "Continuous source root", "A channel marked continuous may be interpolated for display only, and is labelled.", 3, continuous=True, fail=False)),
        ("matrix", main_case("MX-0001", "matrix", 60, "Matrix sample 1", "Matrix sample with a trace.", 4, fail=False)),
        ("v1-comparison", main_case("V1P-0001", "v1-comparison", 40, "v1 parity sample", "v2 reproduces the v1 proposal.", 5, fail=False, with_proposal=False)),
    ]
    enrich_parity(cases[0][1])
    cmpc = main_case("CMP-TWO-ENGINES", "scope-catalog", 60, "Compare mode: v2, v1 and v1eqw", "Each engine has its own receiver body.", 11, fail=False)
    enrich_compare(cmpc); cases.append(("scope-catalog", cmpc))
    cases.append(("scope-catalog", spec_only())); cases.append(("scope-catalog", unit_case()))
    index_cases = []
    for suite, d in cases:
        f = f"cases/{d['case']['id']}.json"
        size = wr(os.path.join(OUT, f), d)
        res = d["results"]
        index_cases.append({"id": d["case"]["id"], "suite": suite, "title": d["case"]["title"], "rule": d["case"]["spec"]["rule"], "file": f, "bytes": size,
                            "git_sha": d["provenance"]["git_sha"], "has_trace": d["time"]["n_ticks"] > 0, "pass": None if not res or any(r["pass"] is None for r in res) else all(r["pass"] for r in res),
                            "families": ["Spacing", "Facing", "AffordancePose"], "modes": ["OUTSIDE", "INTERIOR", "REPLAY"], "statuses": ["EXACT", "INEXACT", "NO_SAMPLE", "NOT_CONVERGED"]})
    # a case with no assertion
    index_cases[4]["pass"] = None
    # synthetic two-joint skeleton fixture (copied, read-only source)
    shutil.copy(FIX, os.path.join(OUT, "cases", "SYNTH-TWO-JOINT.json"))
    sj = json.load(open(FIX))
    index_cases.append({"id": "SYNTH-TWO-JOINT", "suite": "synthetic", "title": sj["case"]["title"], "rule": sj["case"]["spec"]["rule"], "file": "cases/SYNTH-TWO-JOINT.json",
                        "bytes": os.path.getsize(FIX), "git_sha": "synthetic", "has_trace": True, "pass": True, "families": ["Spacing"], "modes": ["OUTSIDE"], "statuses": ["EXACT"]})
    # big table (5000 rows)
    rnd = random.Random(9)
    cols = ["case_id", "scene", "scenario", "mode", "status", "R_err_max", "step_ratio_max", "weight_switches", "pass"]
    rows = []
    for i in range(5000):
        mode = rnd.choice(["OUTSIDE", "INTERIOR", "REPLAY"]); stat = rnd.choice(["EXACT", "EXACT", "EXACT", "INEXACT", "NOT_CONVERGED"])
        err = round(abs(rnd.gauss(0.03, 0.03)), 4); sr = round(0.9 + abs(rnd.gauss(0.2, 0.25)), 3)
        rows.append(["MX-0001" if i == 7 else "", f"scene{i % 12}", f"scn{i % 40}", mode, stat, err, sr, rnd.randint(0, 3), err < 0.1 and sr < 1.5])
    tsize = wr(os.path.join(OUT, "tables", "matrix-summary-0.json"), {"format": "spatialblend.scenario-trace.table", "schema_version": SCHEMA, "synthetic": True, "provenance": prov(),
                                                                       "columns": cols, "rows": rows, "guards": {"R_err_max": {"max": 0.1}, "step_ratio_max": {"max": 1.5}}})
    ecols = ["case", "arm", "verdict", "note"]
    erows = [[f"E{i}", rnd.choice(["v1", "v1eqw", "source"]), rnd.choice(["agree", "differs"]), "a long free-text note about why " * 3] for i in range(40)]
    wr(os.path.join(OUT, "tables", "eval-1.json"), {"format": "spatialblend.scenario-trace.table", "schema_version": SCHEMA, "synthetic": True, "provenance": prov(), "columns": ecols, "rows": erows})
    index = {"format": "spatialblend.scenario-trace.index", "schema_version": SCHEMA, "provenance": prov(),
             "suites": [{"id": "scope-catalog", "title": "Scope catalog", "description": "One case per catalog rule (synthetic dev data).", "kind": "cases"},
                        {"id": "matrix", "title": "Matrix", "description": "Sampled matrix cases and the summary table.", "kind": "table"},
                        {"id": "v1-comparison", "title": "v1 / v2 parity", "kind": "cases"},
                        {"id": "synthetic", "title": "Synthetic fixtures", "description": "Fixtures that exercise the viewer. Never real data.", "kind": "cases"}],
             "cases": index_cases,
             "tables": [{"id": "matrix-summary-0", "suite": "matrix", "title": "Matrix summary shard 0", "file": "tables/matrix-summary-0.json", "columns": cols, "rows": len(rows), "source": "synthetic", "trace_case": "case_id"},
                        {"id": "eval-1", "suite": "scope-catalog", "title": "Eval text table", "file": "tables/eval-1.json", "columns": ecols, "rows": len(erows), "source": "synthetic"}],
             "excluded": ["dataset-derived scenarios: not exported (synthetic dev note)"], "missing": []}
    wr(os.path.join(OUT, "index.json"), index)
    print("wrote", OUT, len(index_cases), "cases, table bytes", tsize)


if __name__ == "__main__":
    main()
