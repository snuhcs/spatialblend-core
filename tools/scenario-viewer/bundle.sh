#!/bin/sh
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

# Inlines every ES module into ONE page, index.bundle.html, for the case where the host's CSP refuses
# <script type="module" src=...> files. Same page contract as index.html (no doctype/html/head/body).
# The three.js <script> tag stays external (cdnjs, pinned); data/ is still fetched by relative URL.
#
# usage: ./bundle.sh [OUT.html]      (default: index.bundle.html next to this script)
# needs: sh + python3 (stdlib only)
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-"$here/index.bundle.html"}
command -v python3 >/dev/null 2>&1 || { echo "bundle.sh: python3 not found" >&2; exit 1; }
python3 - "$here" "$out" <<'PY'
import os, re, sys

root, out = sys.argv[1], sys.argv[2]
html = open(os.path.join(root, "index.html"), encoding="utf-8").read()

IMPORT = re.compile(r"""^import\s+(?:\*\s+as\s+(?P<ns>\w+)|\{(?P<names>[^}]*)\})\s+from\s+['"](?P<src>[^'"]+)['"];?[ \t]*$""", re.M)
EXPORT_DECL = re.compile(r"^export\s+(?P<kw>async\s+function|function|class|const|let)\s+(?P<name>\w+)", re.M)
EXPORT_LIST = re.compile(r"^export\s+\{(?P<names>[^}]*)\};?[ \t]*$", re.M)

mods, order = {}, []

def load(rel):
    rel = os.path.normpath(rel).replace(os.sep, "/")
    if rel in mods:
        return rel
    src = open(os.path.join(root, rel), encoding="utf-8").read()
    mods[rel] = None  # cycle guard
    deps = []
    def imp(m):
        dep = load(os.path.normpath(os.path.join(os.path.dirname(rel), m.group("src"))))
        if m.group("ns"):
            return "const %s = __m(%r);" % (m.group("ns"), dep)
        names = ", ".join(n.strip().replace(" as ", ": ") for n in m.group("names").split(",") if n.strip())
        return "const { %s } = __m(%r);" % (names, dep)
    body = IMPORT.sub(imp, src)
    exports = [m.group("name") for m in EXPORT_DECL.finditer(body)]
    for m in EXPORT_LIST.finditer(body):
        exports += [n.strip().split(" as ")[-1] for n in m.group("names").split(",") if n.strip()]
    body = EXPORT_LIST.sub("", body)
    body = EXPORT_DECL.sub(lambda m: m.group("kw") + " " + m.group("name"), body)
    if re.search(r"^\s*(import|export)\s", body, re.M):
        sys.exit("bundle.sh: unsupported import/export syntax left in " + rel)
    mods[rel] = "function () {\n'use strict';\n%s\nreturn { %s };\n}" % (body, ", ".join(exports))
    order.append(rel)
    return rel

entry = load("js/main.js")
code = ["const __defs = {}, __cache = {};",
        "function __m(id) { if (!(id in __cache)) __cache[id] = __defs[id](); return __cache[id]; }"]
for rel in order:
    code.append("__defs[%r] = %s;" % (rel, mods[rel]))
code.append("__m(%r);" % entry)
js = "\n".join(code).replace("</script", "<\\/script").replace("<!--", "<\\!--")

tag = '<script type="module" src="js/main.js"></script>'
if tag not in html:
    sys.exit("bundle.sh: index.html has no module script tag to replace")
page = html.replace(tag, "<script>\n" + js + "\n</script>")
open(out, "w", encoding="utf-8").write(page)
print("wrote %s (%d bytes, %d modules)" % (out, len(page.encode("utf-8")), len(order)))
PY
