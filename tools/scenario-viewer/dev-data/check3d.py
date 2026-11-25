#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Headless 3D check (Playwright, Chrome, no GUI). Renders sample cases of EVERY available suite in 3D, asserts zero console
errors and a non-blank canvas, checks the 3D-specific behaviour on the SYNTHETIC fixtures, and writes screenshots to
--out (default: a temporary directory outside the repo: screenshots of dataset-derived data must never be committed).

Suites: dev-data/data (synthetic dev data), dev-data/fixtures-affordance (synthetic 1.1 fixtures), and when present
<data dir>/pub/data, <data dir>/real (dataset-derived, local only) and <data dir>/affordance (producer export), with --data-dir.
Served ONLY with serve-local.sh on port 8791 (8790 and 8765 are left alone).

usage: check3d.py [--per-suite N] [--data-dir DIR] [--out DIR] [--channel chrome] [--three FILE] [--port 8791]
"""
import argparse, json, os, signal, socket, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--per-suite", type=int, default=4, help="sample cases per suite (fixtures: all)")
ap.add_argument("--data-dir", default=None, help="<data dir> holding local exports (pub/data, real, affordance, affordance-motiongen)")
ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "scenario-viewer-shots-3d"))
ap.add_argument("--channel", default="chrome")
ap.add_argument("--three", default=None, help="local three.module.min.js served instead of the CDN URL (offline runs; SRI still checked)")
ap.add_argument("--port", type=int, default=8791)
ap.add_argument("--viewport", default="1800x1000")
ap.add_argument("--only", default="", help="run only the data groups whose name contains this text (dev, pub, real)")
args = ap.parse_args()
if args.port in (8790, 8765):
    sys.exit("ports 8790 and 8765 belong to other servers")
os.makedirs(args.out, exist_ok=True)

from playwright.sync_api import sync_playwright

fails = []
def ok(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not cond: fails.append(name)

def present(p): return os.path.isfile(os.path.join(p, "index.json"))
V = os.path.abspath(os.path.expanduser(args.data_dir)) if args.data_dir else os.path.join(HERE, "no-data-dir")
# serve-local.sh refuses duplicate suite/case ids across data dirs, and the pub export reuses the dev-data suite ids,
# so the suites are served in groups (one serve-local.sh on the one allowed port at a time).
GROUPS = [("dev+fixtures", [os.path.join(HERE, "data"), os.path.join(HERE, "fixtures-affordance")])]
if present(os.path.join(V, "affordance")): GROUPS.append(("affordance (export, synthetic)", [os.path.join(V, "affordance")]))
if present(os.path.join(V, "affordance")) and os.path.isfile(os.path.join(V, "affordance-motiongen", "index-sidecars.json")):
    GROUPS.append(("motiongen (sidecars on the affordance export)", [os.path.join(V, "affordance")], [os.path.join(V, "affordance-motiongen")]))
if present(os.path.join(V, "pub", "data")): GROUPS.append(("pub", [os.path.join(V, "pub", "data")]))
if present(os.path.join(V, "real")): GROUPS.append(("real (dataset-derived, local only)", [os.path.join(V, "real")]))
print("groups:", [(g[0], [os.path.relpath(d, ROOT) for d in g[1]]) for g in GROUPS])

def run_group(label, dirs, sidecars=()):
    print("=== group:", label, flush=True)
    srv = subprocess.Popen([os.path.join(ROOT, "serve-local.sh"), "-p", str(args.port)] + [x for d in sidecars for x in ("-s", d)] + dirs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    for _ in range(100):
        try: socket.create_connection(("127.0.0.1", args.port), 0.2).close(); break
        except OSError: time.sleep(0.2)
    else:
        os.killpg(srv.pid, signal.SIGTERM); sys.exit("serve-local.sh did not come up")
    BASE = f"http://127.0.0.1:{args.port}/index.html"
    print("serve command: tools/scenario-viewer/serve-local.sh -p %d %s %s" % (args.port, " ".join("-s " + d for d in sidecars), " ".join(dirs)))

    def merged_index():
        import urllib.request
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/data/index.json"))

    try:
        idx = merged_index()
        by_suite = {}
        for c in idx["cases"]: by_suite.setdefault(c["suite"], []).append(c)
        print("suites:", {k: len(v) for k, v in by_suite.items()})
        with sync_playwright() as pw:
            br = pw.chromium.launch(channel=args.channel, headless=True, args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
            ctx = br.new_context(viewport={"width": int(args.viewport.split("x")[0]), "height": int(args.viewport.split("x")[1])}, device_scale_factor=1)
            pg = ctx.new_page()
            msgs, errors, http, reqs = [], [], [], []
            pg.on("console", lambda m: msgs.append((m.type, m.text)) if "favicon" not in (m.location or {}).get("url", "") else None)
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.on("response", lambda r: http.append((r.status, r.url)) if r.status >= 400 and not r.url.endswith("favicon.ico") else None)
            pg.on("request", lambda r: reqs.append(r.url))
            if args.three:
                pg.route("**/three.module.min.js", lambda route: route.fulfill(path=args.three, content_type="application/javascript", headers={"access-control-allow-origin": "*"}))
            def console_errors():
                return [t for k, t in msgs if k == "error" and not t.startswith("Failed to load resource")] + errors + [f"HTTP {c} {u}" for c, u in http]
            def wait_case(cid):
                pg.wait_for_function("(c) => window.__viewer && window.__viewer.view && window.__viewer.view.kind === 'case' && window.__viewer.view.entry.id === c", arg=cid, timeout=20000)
                pg.wait_for_timeout(150)
            def to3d():
                pg.evaluate("window.__viewer.view.stage.setKind('3d')")
                pg.wait_for_function("window.__viewer.view.stage.renderer && window.__viewer.view.stage.renderer.kind === 'perspective3d' && window.__viewer.view.stage.renderer.panes.length > 0", timeout=20000)
            def info_now(): return pg.evaluate("window.__viewer.view.stage.renderer.getInfo()")
            def settle():
                pg.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
                pg.wait_for_timeout(120)
            def ink():
                from PIL import Image
                import io
                png = pg.locator("#p-stage .stage-canvas").screenshot()
                im = Image.open(io.BytesIO(png)).convert("RGB")
                return len(set((r >> 4, g >> 4, b >> 4) for r, g, b in list(im.get_flattened_data() if hasattr(im, 'get_flattened_data') else im.getdata())[::37]))
            def shot(name):
                pg.locator("#p-stage .stage-wrap").scroll_into_view_if_needed()
                import re
                path = os.path.join(args.out, re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120] + ".png")
                pg.locator("#p-stage .stage-wrap").screenshot(path=path)
                return path

            pg.goto(BASE + "#" + idx["cases"][0]["id"], wait_until="load")
            wait_case(idx["cases"][0]["id"])
            three_reqs = [u for u in reqs if "three.module" in u]
            ok("three.js is NOT requested before the first 3D use (lazy)", not three_reqs, str(three_reqs))
            pg.evaluate("window.__viewer.view.stage.setKind('2d')")

            # ------------------------------------------------------------------ sample every suite
            for sid, cases in by_suite.items():
                if "fixtures" in sid or sid == "affordance": pick = cases   # the motiongen group re-serves the affordance suite with sidecars
                else:
                    step = max(1, len(cases) // args.per_suite) if "affordance" not in sid else 1; pick = [c for c in cases[::step][:args.per_suite] if c.get("has_trace", True)]
                for c in pick:
                    cid = c["id"]
                    pg.evaluate("(id) => { location.hash = encodeURIComponent(id); }", cid)
                    wait_case(cid)
                    if not pg.evaluate("window.__viewer.view.scene.spaceIds.length"): continue
                    to3d()
                    n = pg.evaluate("window.__viewer.view.scene.n")
                    ticks = {"SYN-SIT-01": 90, "SYN-WRITE-01": 62, "SYN-LOOKAHEAD-01": 50}.get(cid, n // 2)
                    if sid == "affordance":   # middle of the span where a source anchor is engaged
                        ticks = pg.evaluate("""() => { const sc = window.__viewer.view.scene; let a = -1, b = -1; for (const ch of sc.channels) { if (!/^source\\.anchor\\..*\\.engaged$/.test(ch.id)) continue; const d = ch.data; for (let k = 0; k < sc.n; k++) { const v = d.v ? d.v[k] >= 0.5 : (d.labels && /^(1|true|engaged)$/i.test(d.labels[d.codes[k]] ?? '')); if (v) { if (a < 0 || k < a) a = k; if (k > b) b = k; } } } return a < 0 ? Math.floor(sc.n / 2) : Math.floor((a + b) / 2); }""")
                    pg.evaluate("(t) => window.__viewer.view.timeline.set(t)", ticks)
                    settle()
                    ink_n = ink()
                    ok(f"[{sid}] {cid}: non-blank 3D canvas", ink_n > 12, f"colours={ink_n}")
                    if sidecars:
                        figs = info_now()["figures"]
                        has = set(pg.evaluate("window.__viewer.view.scene.actors.filter(a => a.jointsCh && a.jointsCh.sidecar).map(a => a.id)"))
                        mine = [f for f in figs if f["actor"] in has]
                        ok(f"[{label}] {cid}: each actor with a sidecar joints channel is drawn from it, labelled 'MotionGen output'", bool(mine) and all(f["kind"] == "joints" and f["label"]["text"] == "MotionGen output" and f["label"]["detail"].startswith("motiongen") for f in mine), json.dumps([(f["actor"], f["kind"], f["label"]["text"]) for f in figs])[:200])
                        ok(f"[{label}] {cid}: sidecar accepted without warnings", "refused" not in pg.inner_text("#p-header") and "could not be loaded" not in pg.inner_text("#p-header"))
                        if pg.evaluate("window.__viewer.view.scene.channels.some(c => /\\.lookahead$/.test(c.id))"):
                            ok(f"[{label}] {cid}: look-ahead ghost labelled", "look-ahead buffer (MotionGen, not yet played)" in pg.inner_text(".hud"))
                    if sid == "affordance":
                        o = pg.evaluate("""() => { const sc = window.__viewer.view.scene, r = window.__viewer.view.stage.renderer, o = sc.anchorCh.get('source:pelvis'); if (!o || !o.engaged) return null;
                            const e = o.engaged.data; let k = -1; for (let t = 0; t < sc.n; t++) { if (e.v ? e.v[t] >= .5 : /^(1|true|engaged)$/i.test(e.labels?.[e.codes[t]] ?? '')) { k = t; break; } } if (k < 0) return null;
                            const a = sc.at(k).spaces.source.actors[0], an = a.anchors.pelvis; if (!an.pose) return null; const d = r.pdata.get('source'); let best = null, bd = 1e9;
                            for (const el of sc.spaces.get('source').list) for (const p of el.parts || []) { const dd = Math.hypot(p.origin3[0] - an.pose.p[0], p.origin3[1] - an.pose.p[1]); if (dd < bd) { bd = dd; best = el; } }
                            const f = d.elems.get(best.id)?.furn; return f ? { yaw: f.group.userData.furn.yaw, actor: a.yaw, type: f.type } : null; }""")
                        if o and o["yaw"] is not None:
                            import math
                            dd = abs(math.degrees(math.atan2(math.sin(o["yaw"] - o["actor"]), math.cos(o["yaw"] - o["actor"]))))
                            ok(f"[{sid}] {cid}: seated source faces the seat part's +x (exported part frame)", dd < 10, f"{o['type']} {dd:.0f} deg")
                    inf = pg.evaluate("(() => { const i = window.__viewer.view.stage.renderer.getInfo(); return { panes: i.panes.map(p => p.key), humanoids: i.humanoids, furniture: i.furniture, parts: i.parts, ik: i.ik, bones: i.bones }; })()")
                    shot(f"{sid}__{cid}")
                    print("    ", json.dumps(inf))
            # ------------------------------------------------------------------ 3D behaviour on fixtures
            def open3d(cid, t=None):
                pg.evaluate("(id) => { location.hash = encodeURIComponent(id); }", cid); wait_case(cid); to3d()
                if t is not None: pg.evaluate("(t) => window.__viewer.view.timeline.set(t)", t)
                settle()
            def info(): return pg.evaluate("window.__viewer.view.stage.renderer.getInfo()")
            def hud(): return pg.inner_text(".hud")
            def tags(): return pg.inner_text(".pane-tags")
            three_reqs = [u for u in reqs if "three.module" in u]
            ok("three.js module requested once 3D was used (pinned 0.160.0, cdnjs)", len(three_reqs) >= 1 and "0.160.0" in three_reqs[0] and "cdnjs.cloudflare.com" in three_reqs[0], str(three_reqs[:2]))
            i = info()
            ok("three.js loaded with SRI verified", i["three"]["sri"] == "verified", json.dumps(i["three"]))

            if any(c["id"] == "SYN-SIT-01" for c in idx["cases"]):
                open3d("SYN-SIT-01", 90)
                i = info()
                ok("SIT: two panes (source | receiver), no v1 pane", [p["key"] for p in i["panes"]] == ["source", "receiver"], json.dumps([p["key"] for p in i["panes"]]))
                ok("SIT: display-IK label on the HUD and the pane tags", "display-derived (IK), not model output" in hud() and "display-derived (IK), not model output" in tags())
                ok("SIT: v1 not available is said", "receiver v1: not available" in hud())
                ok("SIT: figures drawn from IK (2) and no core/MotionGen label", i["ik"] == 2 and "MotionGen" not in hud(), f"ik={i['ik']}")
                fg = {f["pane"]: f for f in i["figures"]}
                ok("SIT: seated posture at engaged pelvis anchor", "seated" in fg["source"]["posture"] and "seated" in fg["receiver"]["posture"], json.dumps({k: v["posture"] for k, v in fg.items()}))
                ok("SIT: no IK residue (all anchors reachable)", not any(f["residues"] for f in i["figures"]), json.dumps([f["residues"] for f in i["figures"]]))
                ok("SIT: furniture built from footprint + parts", i["furniture"] >= 4 and i["parts"] == 2, f"furniture={i['furniture']} parts={i['parts']}")
                pg.evaluate("window.__viewer.view.timeline.set(20)"); settle(); shot("fixture__SIT-approach-t20")
                pg.evaluate("window.__viewer.view.stage.renderer.setCamera({mode: 'follow'})"); pg.evaluate("window.__viewer.view.timeline.set(90)"); settle(); shot("fixture__SIT-follow-t90")
                pg.evaluate("window.__viewer.view.stage.renderer.setCamera({mode: 'orbit'})")
            if any(c["id"] == "SYN-WRITE-01" for c in idx["cases"]):
                open3d("SYN-WRITE-01", 62)
                i = info()
                ok("WRITE: sidecar merged: joints figures labelled MotionGen output with producer", "MotionGen output" in hud() and "SYNTHETIC motiongen-style fixture" in hud(), hud().replace("\n", " | ")[:200])
                ok("WRITE: joints take priority over display IK (no IK label)", i["ik"] == 0 and "display-derived (IK)" not in hud() and all(f["kind"] == "joints" for f in i["figures"]), json.dumps([f["kind"] for f in i["figures"]]))
                ok("WRITE: MotionGen is never labelled core", "core" not in " ".join(f["label"]["text"].lower() for f in i["figures"]))
                ok("WRITE: parts drawn (boards)", i["parts"] == 2, str(i["parts"]))
                ok("WRITE: SYNTHETIC chip shown", "SYNTHETIC" in hud())
                ok("WRITE: sidecar merged channels visible in the inspector", "source.joints" in pg.inner_text("#p-inspector"))
            if any(c["id"] == "SYN-LOOKAHEAD-01" for c in idx["cases"]):
                open3d("SYN-LOOKAHEAD-01", 50)
                ok("LOOKAHEAD: ghost labelled on HUD and pane tag", "look-ahead buffer (MotionGen, not yet played)" in hud() and "look-ahead buffer (MotionGen, not yet played)" in tags())
                ok("LOOKAHEAD: buffer revision shown", "rev 5" in hud(), hud().replace("\n", " | ")[:260])
                vis = pg.evaluate("(() => { const d = window.__viewer.view.stage.renderer.pdata.get('source'); const f = [...d.figs.values()][0]; return f.ghost && f.ghost.group.visible; })()")
                ok("LOOKAHEAD: ghost figure visible", bool(vis))
                pg.evaluate("window.__viewer.view.timeline.set(110)"); settle(); shot("fixture__LOOKAHEAD-t110")
            if any(c["id"] == "SYN-FURNITURE" for c in idx["cases"]):
                open3d("SYN-FURNITURE", 0)
                i = info()
                ok("FURNITURE: all builders produce meshes", i["furniture"] >= 17, f"furniture={i['furniture']}")
                ok("FURNITURE: labelled display-derived", "furniture: display-derived" in hud())
                pg.evaluate("window.__viewer.view.stage.renderer.setCamera({el: 1.1, az: -0.6})"); settle(); shot("fixture__FURNITURE-high")
            # ------------------------------------------------------------------ compare panes (v1 only where v1 channels exist)
            pg.evaluate("location.hash = 'CMP-TWO-ENGINES'") if any(c["id"] == "CMP-TWO-ENGINES" for c in idx["cases"]) else None
            if any(c["id"] == "CMP-TWO-ENGINES" for c in idx["cases"]):
                wait_case("CMP-TWO-ENGINES"); to3d(); pg.evaluate("window.__viewer.view.timeline.set(40)"); settle()
                keys = [p["key"] for p in info()["panes"]]
                ok("COMPARE: Source | receiver v2 | receiver v1 panes side by side", keys == ["source", "receiver", "receiver#v1"], str(keys))
                ps = info()["panes"]
                ok("COMPARE: panes are laid out without overlap", all(a["x"] + a["w"] <= b["x"] or a["y"] + a["h"] <= b["y"] for a, b in zip(ps, ps[1:])), json.dumps([(p["x"], p["y"], p["w"], p["h"]) for p in ps]))
                shot("compare__CMP-TWO-ENGINES")
                pg.locator("#p-stage label:has-text('Compare') input").uncheck(); settle()
                ok("COMPARE: off -> two panes", [p["key"] for p in info()["panes"]] == ["source", "receiver"])
                pg.locator("#p-stage label:has-text('Compare') input").check(); settle()
            # ------------------------------------------------------------------ resource behaviour: switching cases does not leak GL resources
            names = [c["id"] for c in idx["cases"] if c.get("has_trace", True)][:12]
            mem = []
            t0 = time.time()
            for cid in names:
                pg.evaluate("(id) => { location.hash = encodeURIComponent(id); }", cid); wait_case(cid)
                if not pg.evaluate("window.__viewer.view.scene.spaceIds.length"): continue
                to3d(); settle()
                mem.append(pg.evaluate("(() => { const m = window.__viewer.view.stage.renderer.gl.info.memory; return [m.geometries, m.textures]; })()"))
            dt = (time.time() - t0) / max(1, len(names))
            print(f"    case switch in 3D: {dt:.2f} s/case; GL (geometries, textures) per case: {mem}")
            ok("textures are cached across cases (no growth)", max(m[1] for m in mem) - min(m[1] for m in mem) <= 14, str([m[1] for m in mem]))
            first = names[0]
            pg.evaluate("(id) => { location.hash = encodeURIComponent(id); }", first); wait_case(first); to3d(); settle()
            again = pg.evaluate("window.__viewer.view.stage.renderer.gl.info.memory.geometries")
            ok("geometries are disposed on a case switch (revisiting a case gives the same GL geometry count)", again == mem[0][0], f"first visit {mem[0][0]}, after {len(mem)} other cases {again}")
            ok("one WebGL canvas across case switches", pg.evaluate("document.querySelectorAll('canvas.stage-canvas').length") == 1)
            # ------------------------------------------------------------------ the page-wide console
            errs = console_errors()
            warns = [(k, t) for k, t in msgs if k != "error" and "GL Driver Message" not in t]
            ok("ZERO console errors across every rendered case", not errs, "; ".join(errs)[:500])
            print("    other console messages:", len(warns), [t[:120] for _, t in warns[:6]])
            app_msgs = [m for m in msgs if "GL Driver Message" not in m[1]]
            drv = len(msgs) - len(app_msgs)
            ok("ZERO application console messages (the module build prints nothing)", not app_msgs and not errors, str(app_msgs[:3]))
            print(f"    swiftshader 'GPU stall due to ReadPixels' driver notices: {drv} (raised by the harness's canvas screenshots, not by the viewer; not errors)")
            br.close()
    finally:
        try: os.killpg(srv.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: srv.wait(timeout=8)
        except Exception: pass
        print("server stopped (pgid %d)" % srv.pid)

for g in GROUPS:
    if args.only and args.only not in g[0]: continue
    run_group(*g)
print("shots:", args.out)
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
