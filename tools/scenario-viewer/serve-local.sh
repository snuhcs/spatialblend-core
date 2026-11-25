#!/usr/bin/env bash
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

# Serve the scenario viewer with one or more data dirs merged, on 127.0.0.1 only.
#   tools/scenario-viewer/serve-local.sh [-p PORT] [-s SIDECAR-DIR ...] <data-dir> [<data-dir> ...]
# A data dir holds index.json plus the files it lists (relative to index.json). Index fragments are merged
# (suites, cases, tables by id; excluded/missing concatenated; the first dir's provenance is kept). Files are
# symlinked into a temp dir, so nothing is copied and nothing leaves this machine. Ctrl-C stops it and removes
# the temp dir. It never opens a browser.
set -euo pipefail
PORT=8765
SIDECAR_DIRS=""
while [ "${1:-}" = "-p" ] || [ "${1:-}" = "-s" ]; do
  if [ "$1" = "-p" ]; then PORT="$2"; else SIDECAR_DIRS="$SIDECAR_DIRS$2:"; fi
  shift 2
done
[ "$#" -ge 1 ] || { sed -n 2,8p "$0"; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/scenario-viewer.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
ln -s "$HERE/index.html" "$TMP/index.html"
ln -s "$HERE/js" "$TMP/js"
[ -d "$HERE/embodiments" ] && ln -s "$HERE/embodiments" "$TMP/embodiments"
SIDECAR_DIRS="$SIDECAR_DIRS" python3 - "$TMP/data" "$@" <<'PY'
import json, os, sys
out, dirs = sys.argv[1], [os.path.abspath(d) for d in sys.argv[2:]]
os.makedirs(out)
merged = None
def key(kind, x):
    return x["id"]
for d in dirs:
    ip = os.path.join(d, "index.json")
    if not os.path.isfile(ip):
        sys.exit("no index.json in " + d)
    idx = json.load(open(ip))
    for sect in ("cases", "tables"):
        for e in idx.get(sect, []):
            # a case's sidecars (schema 1.1, index `sidecars: [{file}]`) are served next to its case file
            for f in [e["file"]] + [sc["file"] for sc in e.get("sidecars", []) if isinstance(sc, dict) and sc.get("file")]:
                src = os.path.join(d, f)
                if not os.path.isfile(src):
                    sys.exit("%s lists missing file %s" % (ip, f))
                dst = os.path.join(out, f)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.lexists(dst):
                    sys.exit("file path collision across data dirs: " + f)
                os.symlink(src, dst)
    if merged is None:
        merged = idx
        merged["provenance"] = dict(idx.get("provenance") or {}, merged_from=dirs)
        continue
    for sect in ("suites", "cases", "tables"):
        have = {x["id"] for x in merged.setdefault(sect, [])}
        for e in idx.get(sect, []):
            if e["id"] in have:
                sys.exit("duplicate %s id across data dirs: %s" % (sect, e["id"]))
            merged[sect].append(e)
    for sect in ("excluded", "missing"):
        merged.setdefault(sect, []).extend(idx.get(sect, []))
# -s DIR: a sidecar fragment (DIR/index-sidecars.json = {cases: {case id: [{file, producer}]}} + the sidecar files in DIR) is attached to the
# merged index as `sidecars` entries; the files are served under sidecars/ (the data dirs' index files are not edited)
for sd in [d for d in os.environ.get("SIDECAR_DIRS", "").split(":") if d]:
    sd = os.path.abspath(sd)
    frag = json.load(open(os.path.join(sd, "index-sidecars.json")))
    by_id = {c["id"]: c for c in merged.get("cases", [])}
    n = 0
    for cid, ents in frag.get("cases", {}).items():
        c = by_id.get(cid)
        if c is None: continue
        for sc in ents:
            src = os.path.join(sd, sc["file"])
            if not os.path.isfile(src): sys.exit("fragment lists missing sidecar " + sc["file"])
            dst = os.path.join(out, "sidecars", sc["file"]); os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.lexists(dst): os.symlink(src, dst)
            c.setdefault("sidecars", []).append({"file": "sidecars/" + sc["file"], "producer": sc.get("producer", "motiongen")}); n += 1
    print("sidecar fragment %s: %d sidecars attached" % (sd, n))
json.dump(merged, open(os.path.join(out, "index.json"), "w"))
print("merged %d dirs: %d cases, %d tables" % (len(dirs), len(merged.get("cases", [])), len(merged.get("tables", []))))
PY
echo "open http://127.0.0.1:$PORT/index.html   (Ctrl-C to stop)"
cd "$TMP"
python3 -m http.server "$PORT" --bind 127.0.0.1
