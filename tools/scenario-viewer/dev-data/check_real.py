#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Headless check of the viewer against REAL producer output copied by make_real_subset.py (dev-data/real/data).
Opens every case in 2D and 3D, scrubs, and fails on any console error. usage: check_real.py [--three FILE] [--data DIR]"""
import argparse, json, os, signal, socket, subprocess, sys, tempfile, time, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
ap = argparse.ArgumentParser(); ap.add_argument("--three"); ap.add_argument("--data", default=os.path.join(HERE, "real", "data")); ap.add_argument("--bundle", action="store_true"); ap.add_argument("--compare-only", action="store_true", help="only cases with eval.*.receiver_body"); ap.add_argument("--stride", type=int, default=1, help="every Nth case")
args = ap.parse_args()
from playwright.sync_api import sync_playwright
fails = []
def ok(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + (f"  [{d}]" if d else "")); 
    if not c: fails.append(n)
base = tempfile.mkdtemp(prefix="viewer-real-")
if args.bundle: subprocess.check_call([os.path.join(ROOT, "bundle.sh"), base + "/index.html"], stdout=subprocess.DEVNULL)
else:
    os.symlink(ROOT + "/index.html", base + "/index.html"); os.symlink(ROOT + "/js", base + "/js"); os.symlink(ROOT + "/embodiments", base + "/embodiments")
os.symlink(args.data, base + "/data")
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=base, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
idx = json.load(open(args.data + "/index.json"))
if args.compare_only:
    idx["cases"] = [c for c in idx["cases"] if any(ch["id"].startswith("eval.") and ch["id"].endswith(".receiver_body") for ch in json.load(open(args.data + "/" + c["file"]))["channels"])]
idx["cases"] = idx["cases"][::args.stride]
with sync_playwright() as pw:
    br = pw.chromium.launch(channel="chrome", headless=True, args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = br.new_page(viewport={"width": 1280, "height": 900}); logs = []
    pg.on("console", lambda m: logs.append(m.text) if m.type == "error" and not m.text.startswith("Failed to load resource") else None)
    pg.on("pageerror", lambda e: logs.append(str(e)))
    pg.on("response", lambda r: logs.append(f"HTTP {r.status} {r.url}") if r.status >= 400 and not r.url.endswith("favicon.ico") else None)
    if args.three: pg.route("**/three.min.js", lambda r: r.fulfill(path=args.three, content_type="application/javascript"))
    pg.goto(f"http://127.0.0.1:{port}/index.html")
    for c in idx["cases"]:
        pg.evaluate("(id) => { location.hash = encodeURIComponent(id); }", c["id"])
        pg.wait_for_function("(id) => window.__viewer && window.__viewer.view && window.__viewer.view.kind === 'case' && window.__viewer.view.entry.id === id", arg=c["id"], timeout=20000)
        n = pg.evaluate("window.__viewer.view.scene.n"); sp = pg.evaluate("window.__viewer.view.scene.spaceIds.length")
        res = []
        for kind in (["2d", "3d"] if sp else ["2d"]):
            pg.evaluate("(k) => window.__viewer.view.stage.setKind(k)", kind); pg.wait_for_timeout(250)
            for t in ([0, n // 2, max(0, n - 1)] if n else [0]):
                pg.evaluate("(t) => window.__viewer.view.timeline.set(t)", t); pg.wait_for_timeout(60)
            res.append(kind + ":" + pg.evaluate("window.__viewer.view.stage.renderer ? window.__viewer.view.stage.renderer.getInfo().kind : 'none'"))
        neng = pg.evaluate("window.__viewer.view.scene.engines.length")
        if neng and n:
            pg.evaluate("window.__viewer.view.stage.setKind('2d')"); pg.wait_for_timeout(150)
            pn = pg.evaluate("window.__viewer.view.stage.renderer.getInfo().panes.length")
            tog = pg.locator("#p-stage label:has-text('Compare') input")
            ok(f"  compare default ON, {1 + 1 + neng} panes ({c['id'][:36]})", tog.count() == 1 and tog.is_checked() and pn == 2 + neng and not logs, f"panes={pn}")
        if sp and n:
            pg.evaluate("(k) => { window.__viewer.view.stage.setKind('2d'); window.__viewer.view.stage.renderer.setCompare({ on: false }); window.__viewer.view.timeline.set(k); }", n // 2); pg.wait_for_timeout(120)
            want = pg.evaluate("(() => { const v = window.__viewer.view, r = v.stage.renderer; const S = v.scene.at(v.timeline.tick); return r.getInfo().panes.reduce((s, p) => s + S.spaces[p.space].actors.filter((a) => a.pose && !a.joints).length, 0); })()")
            got = pg.evaluate("window.__viewer.view.stage.renderer.getInfo().glyphs")
            det = pg.evaluate("(() => { const sc = window.__viewer.view.scene; const a = sc.at(Math.floor(sc.n / 3)).spaces[sc.spaceIds[0]].actors[0]; sc.at(Math.max(0, sc.n - 1)); const b = sc.at(Math.floor(sc.n / 3)).spaces[sc.spaceIds[0]].actors[0]; return JSON.stringify(a?.walk) === JSON.stringify(b?.walk); })()")
            ok(f"  glyph per actor + deterministic walk ({c['id'][:40]})", got == want and det and not logs, f"glyphs={got} actors={want}")
        if sp and n:
            pg.evaluate("window.__viewer.view.stage.setKind('2d')"); pg.evaluate("document.querySelector('#p-stage .stage-bar').querySelectorAll('button').forEach((b) => b.textContent === 'all' && b.click());")
            bad = []
            for t in sorted({0, n // 2, n - 1}):
                pg.evaluate("(k) => window.__viewer.view.timeline.set(k)", t); pg.wait_for_timeout(100)
                info = pg.evaluate("(() => { const i = window.__viewer.view.stage.renderer.getInfo(); return { labels: i.labels, g: i.glyphRects }; })()")
                L = info["labels"]
                for i, a in enumerate(L):
                    for b in L[i + 1:]:
                        if a["space"] == b["space"] and a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"] and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"]: bad.append(("label/label", t, a["text"], b["text"]))
                    for g in info["g"]:
                        if a["space"] == g["space"] and a["x"] < g["x"] + g["w"] and g["x"] < a["x"] + a["w"] and a["y"] < g["y"] + g["h"] and g["y"] < a["y"] + a["h"]: bad.append(("label/glyph", t, a["text"]))
            ok(f"  labels (mode all): no overlaps at first/mid/last tick ({c['id'][:40]})", not bad, str(bad[:2]))
            ae = pg.evaluate("(() => { const sc = window.__viewer.view.scene; const ids = new Set(sc.actorElements.map((e) => e.space + ':' + e.id)); const hit = []; for (const k of [0, Math.floor(sc.n / 2)]) { const S = sc.at(k); for (const sid of sc.spaceIds) for (const pe of S.spaces[sid].people) if (ids.has(sid + ':' + pe.id)) hit.push(pe.id); } return hit.length; })()")
            ok("  no person glyph at an actor element", ae == 0, str(ae))
            pg.evaluate("window.__viewer.view.stage.renderer.setCamera({ reset: true })"); pg.wait_for_timeout(120)
            fz = pg.evaluate("(() => { const r = window.__viewer.view.stage.renderer, i = r.getInfo(), sc = window.__viewer.view.scene; const f = i.panes.map((p) => sc.focus.get(p.space)); return { scale: i.scale, ext: Math.max(...f.map((x) => Math.max(x.w, x.h))), kinds: f.map((x) => x.kind) }; })()")
            moving = any("paths" in k for k in fz["kinds"])
            if moving and fz["ext"] <= 4.5 and not neng: ok(f"  default fit: glyph >= 40 px on a compact moving case", 0.45 * fz["scale"] >= 40, f"glyph={0.45 * fz['scale']:.0f}px ext={fz['ext']:.1f}m")
            elif moving: print(f"   info  fit glyph {0.45 * fz['scale']:.0f}px (metric) for focus extent {fz['ext']:.1f} m; shoulders floored at 28 px")
            pg.evaluate("window.__viewer.view.stage.setKind('2d')")
        head = pg.inner_text("#p-header")[:0]
        ok(f"{c['id']} (n={n}, spaces={sp})", "Could not open" not in pg.inner_text("#main") and not logs, " ".join(res) + (" ERR " + "; ".join(logs)[:200] if logs else ""))
        logs.clear()
    # time axis label for unit cases
    for c in idx["cases"]:
        d = json.load(open(args.data + "/" + c["file"]))
        if d["time"]["dt_s"] >= 1 and d["time"]["n_ticks"] > 0:
            pg.evaluate("(id) => { location.hash = encodeURIComponent(id); }", c["id"]); pg.wait_for_timeout(600)
            ok("unit case labels poses, not seconds", "pose" in pg.inner_text("#p-timeline") and " s" not in pg.inner_text("#p-timeline").split("\n")[0] , pg.inner_text("#p-timeline").split("\n")[1:3].__repr__()); break
    br.close()
srv.send_signal(signal.SIGTERM); srv.wait(5); shutil.rmtree(base, ignore_errors=True)
print("server killed", srv.pid); print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
