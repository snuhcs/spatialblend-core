#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Check the layering of //spatialblend (CONTRIBUTING.md describes the layout; this script is the rule).

    python3 tools/check_layering.py [--bazel CMD] [--includes]

Layers, top (depended on) to bottom (depends):

    base < geometry (primitives < path < context) < model (element, actor, link, embodiment < relation < realization)
         < solver < space < net < api/c < api/python < sim

Rules, checked on every direct edge of every rule under //spatialblend/... (`bazel query --output=xml`, select() unioned):

  1. A target depends only on its own layer or the layers above it. Inside geometry and model the sub-layers are ordered
     as well.
  2. Only testonly targets depend on a `testing/` package.
  3. Declared exceptions: testonly targets of api/python may depend on //spatialblend/sim:simulator (the simulator
     extension must sit beside the package's __init__.py) and //spatialblend/sim/data:presets (example data).

--includes also reports every `#include "spatialblend/..."` of a C++ rule that neither the rule itself nor a direct
dependency provides (what clang's layering_check enforces; macOS builds with it off). Headers no rule lists (generated
proto headers) are skipped. Reported only; --strict-includes makes them fail.

Exit status: 0 clean, 1 violations, 2 the query failed.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

SUB = {
    "geometry": {"spatialblend/geometry": 0, "spatialblend/geometry/path": 1, "spatialblend/geometry/context": 2},
    "model": {"spatialblend/model/element": 0, "spatialblend/model/actor": 0, "spatialblend/model/link": 0,
              "spatialblend/model/embodiment": 0, "spatialblend/model/relation": 1, "spatialblend/model/realization": 2,
              "spatialblend/model/testing": 0},
}
LAYERS = ["base", "geometry", "model", "solver", "space", "net", "api/c", "api/python", "sim"]
EXCEPTIONS = {("api/python", "//spatialblend/sim:simulator"), ("api/python", "//spatialblend/sim/data:presets")}
INCLUDE = re.compile(r'^\s*#\s*include\s+"(spatialblend/[^"]+)"', re.M)


def layer(pkg: str) -> tuple[int, int] | None:
    """(layer, sub-layer) of a package, or None outside //spatialblend."""
    if not pkg.startswith("spatialblend/"):
        return None
    rest = pkg[len("spatialblend/"):]
    for i, name in enumerate(LAYERS):
        if rest == name or rest.startswith(name + "/"):
            sub = 0
            if name in SUB:
                found = [v for k, v in SUB[name].items() if pkg == k or pkg.startswith(k + "/")]
                if not found and pkg != f"spatialblend/{name}":
                    raise SystemExit(f"package {pkg} has no {name} sub-layer: add it to tools/check_layering.py")
                sub = max(found, default=0)
            return i, sub
    raise SystemExit(f"package {pkg} is in no layer: add it to tools/check_layering.py")


def pkg_of(label: str) -> str:
    return label[2:].split(":", 1)[0]


def query(bazel: list[str]) -> ET.Element:
    out = subprocess.run(bazel + ["query", "kind(rule, //spatialblend/...)", "--output=xml", "--keep_going"],
                         capture_output=True, text=True)
    if out.returncode not in (0, 3) or not out.stdout.strip():
        sys.stderr.write(out.stderr)
        sys.exit(2)
    return ET.fromstring(out.stdout)


def labels(rule: ET.Element, *names: str) -> list[str]:
    found = []
    for lst in rule.findall("list"):
        if lst.get("name") in names:
            found += [e.get("value") for e in lst.findall("label")]
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bazel", default=os.environ.get("BAZEL", "bazel"), help="bazel command (default: $BAZEL or bazel)")
    ap.add_argument("--includes", action="store_true", help="also check #include lines against direct deps")
    ap.add_argument("--strict-includes", action="store_true", help="fail on an include without a direct dep too")
    args = ap.parse_args()
    root = query(shlex.split(args.bazel))
    rules = {r.get("name"): r for r in root.findall("rule")}
    # testonly: the attribute, tests and test suites (implicitly testonly), and the testing/ packages themselves.
    testonly = {name: any(b.get("name") == "testonly" and b.get("value") == "true" for b in r.findall("boolean"))
                or r.get("class", "").endswith("_test") or r.get("class") == "test_suite"
                or pkg_of(name).endswith("/testing")
                for name, r in rules.items()}
    problems, includes = [], []
    for name, r in sorted(rules.items()):
        src = layer(pkg_of(name))
        for dep in sorted({i.get("name") for i in r.findall("rule-input")}):
            if dep not in rules:  # source files and anything outside //spatialblend
                continue
            dst = layer(pkg_of(dep))
            if (LAYERS[src[0]], dep) in EXCEPTIONS and testonly[name]:
                continue
            if dst[0] > src[0] or (dst[0] == src[0] and dst[1] > src[1]):
                problems.append(f"upward: {name} ({LAYERS[src[0]]}) -> {dep} ({LAYERS[dst[0]]})")
            if pkg_of(dep).endswith("/testing") and not testonly[name]:
                problems.append(f"testing: non-testonly {name} -> {dep}")
    if args.includes:
        owners: dict[str, set[str]] = {}
        for name, r in rules.items():
            for f in labels(r, "hdrs", "srcs", "textual_hdrs"):
                if f.startswith("//spatialblend/"):
                    owners.setdefault(f[2:].replace(":", "/"), set()).add(name)
        for name, r in sorted(rules.items()):
            if not r.get("class", "").startswith("cc_"):
                continue
            direct = {name} | set(labels(r, "deps", "implementation_deps"))
            for f in labels(r, "srcs", "hdrs", "textual_hdrs"):
                path = f[2:].replace(":", "/")
                if not f.startswith("//spatialblend/") or not os.path.isfile(path):
                    continue
                for inc in INCLUDE.findall(open(path, encoding="utf-8", errors="replace").read()):
                    have = owners.get(inc)
                    if have and not (have & direct):
                        includes.append(f"include: {name} includes {inc} from {sorted(have)[0]} (not a direct dep)")
    for p in sorted(set(problems)) + sorted(set(includes)):
        print(p)
    print(f"check_layering: {len(rules)} rules, {len(set(problems))} layering violations"
          + (f", {len(set(includes))} includes without a direct dep" if args.includes else ""))
    return 1 if problems or (args.strict_includes and includes) else 0


if __name__ == "__main__":
    sys.exit(main())
