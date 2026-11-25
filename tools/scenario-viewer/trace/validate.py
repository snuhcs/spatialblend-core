#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Validator for spatialblend.scenario-trace v1 (1.0.x and 1.1.x) files (stdlib only).

usage: validate.py [--schema-dir DIR] PATH...   (PATH: a case file, index.json, a table file, a sidecar, or a directory)

Implements the JSON Schema subset the schemas use (type, required, properties, items, enum, const,
minimum, minItems, maxItems, pattern, $ref into the same file) and the semantic checks a schema cannot
express (column lengths against n, CSR offsets, references between channels, actors and elements).
Exit status 0 = every file valid.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _types(t):
    return t if isinstance(t, list) else [t]


def _is(v, t):
    if t == "object": return isinstance(v, dict)
    if t == "array": return isinstance(v, list)
    if t == "string": return isinstance(v, str)
    if t == "boolean": return isinstance(v, bool)
    if t == "null": return v is None
    if t == "integer": return isinstance(v, int) and not isinstance(v, bool)
    if t == "number": return isinstance(v, (int, float)) and not isinstance(v, bool)
    return False


def check(v, s, root, path, errs):
    if "$ref" in s:
        node = root
        for part in s["$ref"].lstrip("#/").split("/"):
            node = node[part]
        check(v, node, root, path, errs)
    if "const" in s and v != s["const"]:
        errs.append(f"{path}: expected {s['const']!r}, got {v!r}")
    if "enum" in s and v not in s["enum"]:
        errs.append(f"{path}: {v!r} not in {s['enum']}")
    if "type" in s and not any(_is(v, t) for t in _types(s["type"])):
        errs.append(f"{path}: expected {s['type']}, got {type(v).__name__}")
        return
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        errs.append(f"{path}: non-finite number")
    if "minimum" in s and _is(v, "number") and v < s["minimum"]:
        errs.append(f"{path}: {v} < {s['minimum']}")
    if "pattern" in s and isinstance(v, str) and not re.search(s["pattern"], v):
        errs.append(f"{path}: {v!r} does not match {s['pattern']}")
    if isinstance(v, dict):
        for r in s.get("required", []):
            if r not in v:
                errs.append(f"{path}: missing '{r}'")
        for k, sub in s.get("properties", {}).items():
            if k in v:
                check(v[k], sub, root, f"{path}.{k}", errs)
    if isinstance(v, list):
        if "minItems" in s and len(v) < s["minItems"]:
            errs.append(f"{path}: fewer than {s['minItems']} items")
        if "maxItems" in s and len(v) > s["maxItems"]:
            errs.append(f"{path}: more than {s['maxItems']} items")
        if "items" in s:
            for i, x in enumerate(v):
                check(x, s["items"], root, f"{path}[{i}]", errs)
                if len(errs) > 50: return


def load(p):
    with open(p) as f:
        return json.load(f)


def semantic(doc, types, errs):
    n = doc["time"]["n_ticks"]
    chans = {}
    for c in doc["channels"]:
        if c["id"] in chans: errs.append(f"channel id {c['id']} repeated")
        chans[c["id"]] = c
    spaces = {s["id"]: s for s in doc["spaces"]}
    for sid, s in spaces.items():
        ids = {e["id"] for e in s["elements"]}
        for e in s["elements"]:
            if e.get("parent") is not None and e["parent"] not in ids:
                errs.append(f"space {sid} element {e['id']}: parent {e['parent']} not in the space")
        if s.get("navigable_root") is not None and s["navigable_root"] not in ids:
            errs.append(f"space {sid}: navigable_root not an element")
    for sid, s in spaces.items():
        for e in s["elements"]:
            for p in e.get("parts", []):
                m = p["world_transform"]
                R = [m[0:3], m[4:7], m[8:11]]
                if abs(m[12]) > 1e-9 or abs(m[13]) > 1e-9 or abs(m[14]) > 1e-9 or abs(m[15] - 1) > 1e-9:
                    errs.append(f"space {sid} element {e['id']} part {p['name']}: world_transform bottom row is not 0 0 0 1")
                for i in range(3):
                    for k in range(3):
                        dot = sum(R[r][i] * R[r][k] for r in range(3))
                        if abs(dot - (1.0 if i == k else 0.0)) > 1e-4:
                            errs.append(f"space {sid} element {e['id']} part {p['name']}: rotation is not orthonormal"); break
                    else:
                        continue
                    break
                if len(p["footprint_local"]) < 3: errs.append(f"space {sid} element {e['id']} part {p['name']}: footprint has fewer than 3 points")
    for c in doc["channels"]:
        # 1.1.0 anchor channels: source.anchor.<name>[.engaged|.part], receiver.anchor.<name>.proposal|.engaged|.status|.part
        m = re.match(r"^(source|receiver)\.anchor\.([A-Za-z0-9_]+)(\.[a-z_]+)?$", c["id"])
        if m:
            want = {None: "pose3d", ".proposal": "pose3d", ".engaged": "enum", ".status": "enum", ".part": "enum"}.get(m.group(3))
            if want is None: errs.append(f"channel {c['id']}: unknown anchor channel suffix")
            elif c["type"] != want: errs.append(f"channel {c['id']}: expected type {want}, got {c['type']}")
            elif m.group(3) == ".engaged":
                if any(x not in (0, 1, None) for x in c["data"]["codes"]) or len(c["data"]["labels"]) != 2:
                    errs.append(f"channel {c['id']}: engaged codes must be 0/1 with two labels")
            if m.group(2) not in ("pelvis", "hand_r", "hand_l"): errs.append(f"channel {c['id']}: anchor name {m.group(2)!r} not in pelvis, hand_r, hand_l")
    for a in doc["actors"]:
        if a["space"] not in spaces: errs.append(f"actor {a['id']}: unknown space")
        rc = chans.get(a["root_channel"])
        if not rc or rc["type"] != "pose3d": errs.append(f"actor {a['id']}: root_channel is not a pose3d channel")
        sk = a.get("skeleton")
        if sk:
            js = sk.get("joints", [])
            if js and isinstance(js[0], dict):
                # 1.1.0: joints [{name, parent}], parent = a joint index or -1, parents before children
                for i, j in enumerate(js):
                    if not isinstance(j, dict) or "name" not in j or "parent" not in j:
                        errs.append(f"actor {a['id']}: skeleton joint {i} needs name and parent"); break
                    if not (-1 <= j["parent"] < len(js)) or j["parent"] >= i and j["parent"] != -1:
                        errs.append(f"actor {a['id']}: skeleton joint {i} ({j['name']}) has parent {j['parent']}"); break
                if len({j.get("name") for j in js if isinstance(j, dict)}) != len(js): errs.append(f"actor {a['id']}: skeleton joint names repeat")
            elif len(js) != len(sk.get("parents", [])):
                errs.append(f"actor {a['id']}: skeleton joints/parents differ")
            jc = chans.get(a.get("joints_channel") or "")
            if jc and jc["data"].get("joint_count") != len(js): errs.append(f"actor {a['id']}: joint_count differs")
        emb = a.get("embodiment")
        if emb is not None:
            errs.extend(_check_embodiment(f"actor {a['id']}", emb, a.get("skeleton")))
    for c in doc["channels"]:
        t, d, cid = c["type"], c["data"], c["id"]
        if t not in types["types"]:
            continue  # unknown types are listed by the viewer, not an error
        sch = {"$defs": types["$defs"]}
        e2 = []
        check(d, types["types"][t], sch, f"channel {cid}.data", e2)
        errs.extend(e2)
        if e2: continue
        L = lambda a, k: len(a)
        def need(name, arr, k):
            if arr is not None and len(arr) != k:
                errs.append(f"channel {cid}: {name} has {len(arr)} entries, expected {k}")
        if t == "pose3d":
            need("p", d["p"], 3 * d["n"]); need("q", d["q"], 4 * d["n"]); need("valid", d.get("valid"), d["n"])
            if d["n"] != n: errs.append(f"channel {cid}: n={d['n']} but case has {n} ticks")
            for i in range(d["n"]):
                if d.get("valid") and not d["valid"][i]: continue
                q = d["q"][4 * i:4 * i + 4]
                if None in q or abs(sum(x * x for x in q) - 1) > 1e-3: errs.append(f"channel {cid}: tick {i} quaternion not unit"); break
        elif t == "scalar":
            need("v", d["v"], d["n"])
        elif t == "vector":
            need("v", d["v"], d["n"] * d["dim"])
        elif t == "enum":
            need("codes", d["codes"], d["n"])
            for x in d["codes"]:
                if x is not None and not (0 <= x < len(d["labels"])): errs.append(f"channel {cid}: code {x} out of labels"); break
        elif t == "bitset":
            need("v", d["v"], d["n"])
        elif t == "event":
            for ev in d["events"]:
                if not (0 <= ev["tick"] < max(n, 1)): errs.append(f"channel {cid}: event tick {ev['tick']} outside the case"); break
        elif t == "scope":
            need("mode.codes", d["mode"]["codes"], d["n"])
            for part in ("mode", "yaw_source"):
                if part in d:
                    need(part + ".codes", d[part]["codes"], d["n"])
                    for x in d[part]["codes"]:
                        if x is not None and not (0 <= x < len(d[part]["labels"])): errs.append(f"channel {cid}: {part} code {x} out of labels"); break
            for part in ("in_scope", "dropped"):
                o = d[part]["offsets"]
                if len(o) != d["n"] + 1 or o[0] != 0 or any(o[i] > o[i + 1] for i in range(len(o) - 1)):
                    errs.append(f"channel {cid}: {part}.offsets is not a CSR over {d['n']} ticks"); continue
                for k in d[part]:
                    if k not in ("offsets", "reasons") and len(d[part][k]) != o[-1]:
                        errs.append(f"channel {cid}: {part}.{k} has {len(d[part][k])} rows, offsets say {o[-1]}")
        elif t == "relation_set":
            o = d["offsets"]
            if len(o) != d["n"] + 1 or o[0] != 0 or any(o[i] > o[i + 1] for i in range(len(o) - 1)):
                errs.append(f"channel {cid}: offsets is not a CSR over {d['n']} ticks"); continue
            rows = o[-1]
            for k, v in d.items():
                if k in ("n", "offsets", "families") or not isinstance(v, list): continue
                want = 3 * rows if k.endswith("_p") else rows
                if len(v) != want: errs.append(f"channel {cid}: column {k} has {len(v)} entries, expected {want}")
        elif t == "proposal":
            need("status.codes", d["status"]["codes"], d["n"]); need("p", d["p"], 3 * d["n"]); need("q", d["q"], 4 * d["n"])
            for x in d["status"]["codes"]:
                if x is not None and not (0 <= x < len(d["status"]["labels"])): errs.append(f"channel {cid}: status code {x} out of labels"); break
        elif t == "joints":
            need("p", d["p"], 3 * d["n"] * d["joint_count"]); need("q", d["q"], 4 * d["n"] * d["joint_count"])
    for r in doc.get("results", []):
        if r["expectation"] not in {x["id"] for x in doc.get("expectations", [])}:
            errs.append(f"result for unknown expectation {r['expectation']}")


def validate_sidecar(doc, sdir):
    """A sidecar (1.1.0): {format, schema_version, case_id, producer{name,version,git_sha,command,generated_at},
    actors[{id, skeleton}], channels[...]}. Channels carry their own `producer` and (joints) `derived`; they are merged into
    the case by a viewer, never into the core's own channels."""
    errs = []
    for k in ("schema_version", "case_id", "producer", "actors", "channels"):
        if k not in doc: errs.append(f"sidecar: missing '{k}'")
    if errs: return errs
    if not re.search(r"^1\.[0-9]+\.[0-9]+$", str(doc["schema_version"])): errs.append("sidecar: schema_version is not 1.x.y")
    for k in ("name", "version", "git_sha", "command", "generated_at"):
        if not isinstance(doc["producer"].get(k), str) or not doc["producer"].get(k): errs.append(f"sidecar: producer.{k} missing")
    types = load(os.path.join(sdir, "channel-types.schema.json"))
    skel = {}
    for a in doc["actors"]:
        sk = a.get("skeleton")
        if not isinstance(sk, dict) or not isinstance(sk.get("joints"), list) or not sk["joints"] or not isinstance(sk["joints"][0], dict):
            errs.append(f"sidecar actor {a.get('id')}: skeleton needs joints [{{name, parent}}]"); continue
        if sk.get("units") != "m" or sk.get("frame") != "world z-up": errs.append(f"sidecar actor {a['id']}: skeleton needs units 'm' and frame 'world z-up'")
        for i, j in enumerate(sk["joints"]):
            if not (-1 <= j.get("parent", -2) < i): errs.append(f"sidecar actor {a['id']}: joint {i} parent must be -1 or an earlier joint"); break
        skel[a["id"]] = sk
    for c in doc["channels"]:
        cid = c.get("id", "?")
        if c.get("type") != "joints": 
            continue
        if not isinstance(c.get("producer"), dict) or not c["producer"].get("name"): errs.append(f"sidecar channel {cid}: joints channels need producer.name")
        if not c.get("origin"): errs.append(f"sidecar channel {cid}: joints channels need origin")
        if c.get("derived") not in ("motiongen", "display-ik", "recorded"): errs.append(f"sidecar channel {cid}: derived must be motiongen, display-ik or recorded")
        d = c["data"]
        e2 = []
        check(d, types["types"]["joints"], {"$defs": types["$defs"]}, f"channel {cid}.data", e2)
        errs.extend(e2)
        if e2: continue
        if len(d["p"]) != 3 * d["n"] * d["joint_count"] or len(d["q"]) != 4 * d["n"] * d["joint_count"]:
            errs.append(f"sidecar channel {cid}: p/q length differs from n * joint_count")
        sk = skel.get(c.get("actor"))
        if sk is None: errs.append(f"sidecar channel {cid}: actor {c.get('actor')!r} is not in the sidecar's actors")
        elif sk and len(sk["joints"]) != d["joint_count"]: errs.append(f"sidecar channel {cid}: joint_count differs from the skeleton")
    return errs


def validate_file(path, sdir):
    errs = []
    try:
        doc = load(path)
    except Exception as e:
        return [f"{path}: unreadable json: {e}"]
    fmt = doc.get("format") if isinstance(doc, dict) else None
    if fmt == "spatialblend.scenario-trace":
        sch = load(os.path.join(sdir, "scenario-trace.schema.json"))
        types = load(os.path.join(sdir, "channel-types.schema.json"))
        check(doc, sch, sch, "$", errs)
        if not errs: semantic(doc, types, errs)
    elif fmt == "spatialblend.scenario-trace.index":
        sch = load(os.path.join(sdir, "index.schema.json"))
        check(doc, sch, sch, "$", errs)
        if not errs:
            ids = set()
            for c in doc["cases"]:
                if (c["suite"], c["id"]) in ids: errs.append(f"case {c['suite']}/{c['id']} repeated")
                ids.add((c["suite"], c["id"]))
                if not os.path.exists(os.path.join(os.path.dirname(path), c["file"])): errs.append(f"case file {c['file']} missing")
            for c in doc["cases"]:
                for sc in c.get("sidecars", []):
                    if not os.path.exists(os.path.join(os.path.dirname(path), sc["file"])): errs.append(f"case {c['id']}: sidecar file {sc['file']} missing")
            suites = {s["id"] for s in doc["suites"]}
            for c in doc["cases"]:
                if c["suite"] not in suites: errs.append(f"case {c['id']}: unknown suite {c['suite']}")
    elif fmt == "spatialblend.scenario-trace.sidecar":
        errs = validate_sidecar(doc, sdir)
    elif fmt == "spatialblend.scenario-trace.table":
        for k in ("columns", "rows", "schema_version"):
            if k not in doc: errs.append(f"table: missing {k}")
        if not errs:
            w = len(doc["columns"])
            for i, r in enumerate(doc["rows"]):
                if len(r) != w: errs.append(f"table row {i} has {len(r)} cells, expected {w}"); break
    else:
        errs.append(f"unknown format {fmt!r}")
    return [f"{path}: {e}" for e in errs]


def main(argv):
    sdir = os.path.join(HERE, "schema")
    paths = []
    i = 0
    while i < len(argv):
        if argv[i] == "--schema-dir": sdir = argv[i + 1]; i += 2; continue
        paths.append(argv[i]); i += 1
    files = []
    for p in paths:
        if os.path.isdir(p):
            for r, _, fs in os.walk(p):
                files += [os.path.join(r, f) for f in sorted(fs) if f.endswith(".json")]
        else:
            files.append(p)
    bad, ok = 0, 0
    for f in files:
        e = validate_file(f, sdir)
        if e:
            bad += 1
            print("\n".join(e[:20]))
        else:
            ok += 1
    print(f"scenario-trace validate: {ok} valid, {bad} invalid of {len(files)} files")
    return 1 if bad else 0



def _check_embodiment(where, emb, skeleton):
    """1.1.0 actor.embodiment: the registration the core held (spatialblend.embodiment 1.x, canonical JSON)."""
    errs = []
    if emb.get("format") != "spatialblend.embodiment": return [f"{where}: embodiment.format must be spatialblend.embodiment"]
    js = emb.get("skeleton", {}).get("joints", [])
    names = [j.get("name") for j in js]
    if not js or len(set(names)) != len(names): errs.append(f"{where}: embodiment skeleton needs unique joint names")
    for i, j in enumerate(js):
        if not (-1 <= j.get("parent", -2) < i): errs.append(f"{where}: embodiment joint {j.get('name')} has parent {j.get('parent')}")
    if skeleton and [j["name"] for j in skeleton.get("joints", [])] != names:
        errs.append(f"{where}: actor.skeleton differs from the embodiment skeleton")
    seen = set()
    for an in emb.get("anchors", []):
        n = an.get("name")
        if n in seen: errs.append(f"{where}: embodiment anchor {n} repeats")
        seen.add(n)
        if an.get("bone") is not None and an["bone"] not in names: errs.append(f"{where}: embodiment anchor {n} bone {an['bone']!r} not in the skeleton")
        if not set(an.get("families", [])) or not set(an["families"]) <= {"Spacing", "Facing", "AffordancePose"}:
            errs.append(f"{where}: embodiment anchor {n} families must be a non-empty subset of Spacing, Facing, AffordancePose")
        off = an.get("offset", {})
        nonid = any(off.get("position", [0, 0, 0])) or list(off.get("rotation", [0, 0, 0, 1])) != [0, 0, 0, 1]
        prov = an.get("provenance") or {}
        hex64 = lambda v: isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdefABCDEF" for c in v)
        if nonid and not (prov.get("procedure") and (hex64(prov.get("asset_sha256")) or (isinstance(prov.get("source"), str) and prov.get("source")))):
            errs.append(f"{where}: embodiment anchor {n} has a non-identity offset without provenance.procedure and either a 64-hex provenance.asset_sha256 or provenance.source (the core's rule)")
    ch = emb.get("chain")
    if ch is not None:
        cj = ch.get("joints", [])
        if [j.get("name") for j in cj] != names: errs.append(f"{where}: embodiment.chain joints must list the skeleton joints in order")
        if not (ch.get("provenance") or {}).get("procedure"): errs.append(f"{where}: embodiment.chain needs provenance.procedure")
        if (ch.get("normalization") or {}).get("up") != "z": errs.append(f"{where}: embodiment.chain.normalization must state z up")
    return errs

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
