#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""spatialblend.scope-context v0: the conditioning context of a scenario case, for a motion generator (stdlib only).

usage: scope_context.py CASE.json [--actor source|receiver] [-o OUT.json]

Reads one spatialblend.scenario-trace case file (schema 1.1.0, the `affordance` suite or any case that has the body-anchor
channels) and writes the scope context a motion generator conditions on:

  {schema:"spatialblend.scope-context", schema_version:"0", case_id, source_case{...}, actor, space, dt_s, n_ticks,
   input, derived[...], embodiment (the actor's registered embodiment, verbatim, or null),
   referents:{<shared_id>:{element_id, name, kind, world_transform[16], shape3d, parts:[{name, tag, world_transform[16],
                           footprint_local[[x,y]], normal[3]}]}},
   ticks:[{tick, root{p,q}, scope{mode, container, container_shared_id, referents[shared ids], by_family{Spacing,Facing,
           AffordancePose}}, anchors{<name>:{p,q}}, engaged:[{anchor, referent, part, contact_p, source_height_m}]}]}

It contains only values the core exported (or the scenario declared) and what follows from them by a transform:
`parts[].normal` is the third column of the part's world_transform (the plane's +z); `engaged[].referent` is the
AffordancePose referent in scope whose part of that name the anchor lies on (the core's own contact rule decided
that the anchor is engaged); `engaged[].source_height_m` is the anchor's height above that plane computed from the
exported anchor pose and the exported part transform (the exported poses are rounded to 1 mm, so it is at that
resolution; engagement itself means on or below the plane to the core's numerical zero). Every derived field is
listed in `derived`. `input` states the provenance of the case's source track (for the affordance suite:
synthetic:affordance-tracks:<version>); nothing is added to it.
"""
import argparse
import json
import math
import os
import sys

SCHEMA = "spatialblend.scope-context"
SCHEMA_VERSION = "0"
# The resolution the exporter rounds poses to (trace_recorder.cc kMm): a height below it is not resolvable.
POSE_RESOLUTION_M = 1e-3


def _chan(case):
    return {c["id"]: c for c in case["channels"]}


def _rot_inv_apply(m, p):
    """Apply the inverse of a rigid 4x4 row-major transform m to the point p: R^T (p - t)."""
    r = [m[0:3], m[4:7], m[8:11]]
    t = [m[3], m[7], m[11]]
    d = [p[i] - t[i] for i in range(3)]
    return [sum(r[k][i] * d[k] for k in range(3)) for i in range(3)]


def _inside(poly, x, y):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _pose(data, t):
    """(p, q, valid) of tick t of a pose3d channel."""
    ok = data.get("valid", [1] * data["n"])[t] != 0
    return data["p"][3 * t:3 * t + 3], data["q"][4 * t:4 * t + 4], ok


def extract(case, actor="source"):
    if case.get("format") != "spatialblend.scenario-trace":
        raise ValueError("not an spatialblend.scenario-trace case file")
    chans = _chan(case)
    space_id = {a["id"]: a["space"] for a in case["actors"]}.get(actor, actor)
    spaces = {s["id"]: s for s in case["spaces"]}
    if space_id not in spaces:
        raise ValueError(f"no space for actor {actor!r}")
    space = spaces[space_id]
    n = case["time"]["n_ticks"]
    elems = {e["id"]: e for e in space["elements"]}

    # ---- referents: every element with a shared id that is not the room or a person ----
    referents = {}
    for e in space["elements"]:
        if e.get("shared_id") is None or e["kind"] in ("room", "person"):
            continue
        parts = []
        for p in e.get("parts", []):
            m = p["world_transform"]
            parts.append({"name": p["name"], "tag": p.get("tag", p["name"]), "world_transform": m,
                          "footprint_local": p["footprint_local"], "normal": [m[2], m[6], m[10]]})
        referents[str(e["shared_id"])] = {"element_id": e["id"], "name": e["name"], "kind": e["kind"],
                                          "world_transform": e["world_transform"], "shape3d": e.get("shape3d"), "parts": parts}

    # ---- scope per tick (the source's resolved scope) ----
    scope = chans.get(f"{actor}.scope")
    fam_names = scope["data"]["families"] if scope else []
    ap_idx = fam_names.index("AffordancePose") if "AffordancePose" in fam_names else None

    def scope_at(t):
        if not scope:
            return {"mode": None, "container": None, "container_shared_id": None, "referents": [], "by_family": {}}
        d = scope["data"]
        code = d["mode"]["codes"][t]
        cont = d["container"][t]
        o = d["in_scope"]["offsets"]
        by = {f: [] for f in fam_names}
        for k in range(o[t], o[t + 1]):
            by[fam_names[d["in_scope"]["family"][k]]].append(d["in_scope"]["referent"][k])
        flat = sorted({r for v in by.values() for r in v})
        sid = elems[cont].get("shared_id") if cont is not None and cont in elems else None
        return {"mode": None if code is None else d["mode"]["labels"][code], "container": cont, "container_shared_id": sid,
                "referents": flat, "by_family": {f: sorted(set(v)) for f, v in by.items()}}

    # ---- anchors ----
    anchors = {}
    for cid, c in chans.items():
        prefix = f"{actor}.anchor."
        if cid.startswith(prefix) and c["type"] == "pose3d":
            name = cid[len(prefix):]
            if "." in name:
                continue
            anchors[name] = {"pose": c["data"],
                             "engaged": chans.get(cid + ".engaged", {}).get("data"),
                             "part": chans.get(cid + ".part", {}).get("data")}
    root = chans.get(f"{actor}.root") or chans.get(f"{actor}.body")

    emb = (case["case"].get("tags") or {}).get("embodiment_offsets")
    ticks = []
    for t in range(n):
        tk = {"tick": t}
        if root:
            p, q, ok = _pose(root["data"], t)
            tk["root"] = {"p": p, "q": q} if ok else None
        sc = scope_at(t)
        tk["scope"] = sc
        tk["anchors"] = {}
        tk["engaged"] = []
        for name, a in anchors.items():
            p, q, ok = _pose(a["pose"], t)
            if not ok:
                continue
            tk["anchors"][name] = {"p": p, "q": q}
            eng = a["engaged"]
            if not eng or eng["codes"][t] != 1:
                continue
            part_name = None
            if a["part"] and a["part"]["codes"][t] is not None:
                part_name = a["part"]["labels"][a["part"]["codes"][t]]
            # the AffordancePose referent in scope whose part of that name the anchor lies on
            best = None
            cands = sc["by_family"].get("AffordancePose", []) if ap_idx is not None else []
            for rid in cands:
                for part in referents.get(str(rid), {}).get("parts", []):
                    if part["name"] != part_name:
                        continue
                    lx, ly, lz = _rot_inv_apply(part["world_transform"], p)
                    if _inside(part["footprint_local"], lx, ly) and (best is None or abs(lz) < abs(best[1])):
                        best = (rid, lz)
            if best is None:
                tk["engaged"].append({"anchor": name, "referent": None, "part": part_name, "contact_p": p, "source_height_m": None})
            else:
                h = round(best[1] / POSE_RESOLUTION_M) * POSE_RESOLUTION_M
                tk["engaged"].append({"anchor": name, "referent": best[0], "part": part_name, "contact_p": p,
                                      "source_height_m": 0.0 if h == 0 else h})
        ticks.append(tk)

    prov = case.get("provenance", {})
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "case_id": case["case"]["id"],
        "source_case": {"suite": case["case"]["suite"], "schema_version": case["schema_version"], "git_sha": prov.get("git_sha"),
                        "producer": prov.get("producer")},
        "actor": actor,
        "space": space_id,
        "dt_s": case["time"]["dt_s"],
        "n_ticks": n,
        "input": (case["case"].get("tags") or {}).get("input"),
        "derived": ["referents.*.parts[].normal", "ticks[].engaged[].referent", "ticks[].engaged[].source_height_m"],
        # provenance only, passed through: no joint is computed from it (anchors and offsets are registered by the embodiment driver)
        "embodiment_offsets": emb,
        "pose_resolution_m": POSE_RESOLUTION_M,
        # The embodiment the core held for this actor (schema 1.1.0 actor.embodiment, verbatim): anchors, their bones and
        # offsets with provenance, the skeleton and the optional normalized chain. A motion generator reads its anchor and
        # contact points from here and nowhere else; null = the actor registered none.
        "embodiment": {a["id"]: a for a in case["actors"]}.get(actor, {}).get("embodiment"),
        "referents": referents,
        "ticks": ticks,
    }


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("--actor", default="source")
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)
    with open(a.case) as f:
        case = json.load(f)
    ctx = extract(case, a.actor)
    text = json.dumps(ctx, separators=(",", ":"))
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
