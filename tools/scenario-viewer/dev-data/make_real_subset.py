#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Copies a few real producer case files into dev-data/real/data and writes a minimal index.json for them.
usage: make_real_subset.py SRC_EXPORT_DIR [N]   (dev-data/real/ is untracked scratch)"""
import json, os, shutil, sys, glob
src = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real", "data")
shutil.rmtree(out, ignore_errors=True); os.makedirs(out + "/cases")
files = sorted(glob.glob(src + "/cases/*.json"), key=os.path.getsize)
pick = [f for f in files if json.load(open(f))["time"]["n_ticks"] == 0][:1] + \
       [f for f in files if json.load(open(f))["time"]["dt_s"] >= 1][:2] + \
       [f for f in files if os.path.getsize(f) > 100000 and "border" not in open(f).read(2000)][:n]
pick = list(dict.fromkeys(pick))[:n]
cases = []
for f in pick:
    d = json.load(open(f)); shutil.copy(f, out + "/cases/")
    c = d["case"]
    cases.append({"id": c["id"], "suite": c["suite"], "title": c["title"], "rule": c.get("spec", {}).get("rule", ""), "file": "cases/" + os.path.basename(f),
                  "bytes": os.path.getsize(f), "git_sha": d["provenance"].get("git_sha"), "has_trace": d["time"]["n_ticks"] > 0, "pass": None})
d0 = json.load(open(pick[0]))
json.dump({"format": "spatialblend.scenario-trace.index", "schema_version": d0["schema_version"], "provenance": d0["provenance"],
           "suites": [{"id": s, "title": s} for s in sorted({c["suite"] for c in cases})], "cases": cases}, open(out + "/index.json", "w"))
print(len(cases), "cases ->", out)
