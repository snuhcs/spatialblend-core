#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Headless check of the viewer against dev-data (no GUI window). Needs python playwright + Chrome.

usage: check.py [--stage DIR] [--chrome-channel chrome]
Stages symlinks (index.html, js/, data/) in DIR, serves it on 127.0.0.1 with python's http.server, drives
headless Chrome, prints one PASS/FAIL line per check, kills the server by PID, exits non-zero on any FAIL.
"""
import argparse, json, os, shutil, signal, subprocess, sys, tempfile, time, socket

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--stage", default=None)
ap.add_argument("--channel", default="chrome")
ap.add_argument("--three", default=None, help="local three.module.min.js to serve in place of the cdnjs URL (offline runs; the SRI check still applies)")
ap.add_argument("--bundle", action="store_true", help="check index.bundle.html (built by bundle.sh) instead of index.html + js/")
ap.add_argument("--data", default=os.path.join(HERE, "data"))
args = ap.parse_args()

from playwright.sync_api import sync_playwright

fails = []
def ok(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond: fails.append(name)

def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

def stage(dirpath, with_data=True):
    if os.path.isdir(dirpath): shutil.rmtree(dirpath)
    os.makedirs(dirpath)
    if args.bundle:
        subprocess.check_call([os.path.join(ROOT, "bundle.sh"), os.path.join(dirpath, "index.html")], stdout=subprocess.DEVNULL)
    else:
        os.symlink(os.path.join(ROOT, "index.html"), os.path.join(dirpath, "index.html"))
        os.symlink(os.path.join(ROOT, "js"), os.path.join(dirpath, "js"))
    os.symlink(os.path.join(ROOT, "embodiments"), os.path.join(dirpath, "embodiments"))
    if with_data: os.symlink(args.data, os.path.join(dirpath, "data"))

def serve(dirpath):
    port = free_port()
    p = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=dirpath, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        try: socket.create_connection(("127.0.0.1", port), 0.2).close(); break
        except OSError: time.sleep(0.1)
    return p, port

base = args.stage or tempfile.mkdtemp(prefix="viewer-stage-")
stage(os.path.join(base, "site")); stage(os.path.join(base, "empty"), with_data=False)
srv, port = serve(os.path.join(base, "site"))
srv2, port2 = serve(os.path.join(base, "empty"))
print(f"servers: site pid={srv.pid} port={port}; empty pid={srv2.pid} port={port2}")

MOCK = r"""
(() => {
  const store = new Map();
  const listeners = new Set();
  window.__writes = [];
  const seed = JSON.parse(window.__SEED || '{}');
  for (const [k, v] of Object.entries(seed)) store.set(k, v);
  const snap = () => ({ docs: [...store].map(([id, d]) => ({ id, exists: true, data: () => d, metadata: {} })), size: store.size, empty: !store.size, docChanges: () => [], metadata: {} });
  const coll = { onSnapshot(next) { window.__subs = (window.__subs || 0) + 1; listeners.add(next); setTimeout(() => next(snap()), 5); return () => listeners.delete(next); },
    doc(id) { return { id, async set(d) { window.__writes.push({ id, d }); store.set(id, d); listeners.forEach((l) => l(snap())); } }; } };
  const db = { collection: (n) => { if (n !== 'verdicts') throw new Error('unexpected collection ' + n); return coll; } };
  const user = { id: async () => 'u-owner', can: async () => true, profiles: async (ids) => Object.fromEntries([].concat(ids).map((i) => [i, { id: i, name: 'Owner Name' }])) };
  window.claude = { use: async (n) => (n === 'db' ? db : n === 'user' ? user : null) };
})();
"""

# the committed SYNTHETIC fixtures must validate against the final schema (tools/scenario-viewer/trace/validate.py)
val = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "trace", "validate.py"), os.path.join(HERE, "fixtures-affordance")], capture_output=True, text=True)
ok("validate.py accepts every committed fixture (dev-data/fixtures-affordance)", val.returncode == 0, (val.stdout + val.stderr).strip().splitlines()[-1] if (val.stdout + val.stderr).strip() else "")

with sync_playwright() as pw:
    br = pw.chromium.launch(channel=args.channel, headless=True, args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    def new_page(url, w=1280, h=900, init=None, three=True):
        ctx = br.new_context(viewport={"width": w, "height": h}, device_scale_factor=1)
        pg = ctx.new_page()
        logs = {"console": [], "errors": [], "failed": [], "http": []}
        pg.on("console", lambda m: logs["console"].append((m.type, m.text)))
        pg.on("pageerror", lambda e: logs["errors"].append(str(e)))
        pg.on("response", lambda r: logs["http"].append((r.status, r.url)) if r.status >= 400 else None)
        pg.on("requestfailed", lambda r: logs["failed"].append(r.url))
        if args.three:
            pg.route("**/three.module.min.js", lambda route: route.fulfill(path=args.three, content_type="application/javascript", headers={"access-control-allow-origin": "*"}))
        if init: pg.add_init_script(init)
        pg.goto(url, wait_until="load")
        return pg, logs, ctx

    def errs(logs):
        # a 404 for /favicon.ico is the bare http.server having no icon; any other HTTP error counts
        bad = [t for k, t in logs["console"] if k == "error" and "fonts.g" not in t and not t.startswith("Failed to load resource")]
        http = [f"HTTP {c} {u}" for c, u in logs["http"] if not u.endswith("/favicon.ico")]
        return bad + logs["errors"] + http

    def wait_case(pg, cid=None, timeout=15000):
        pg.wait_for_function("(c) => window.__viewer && window.__viewer.view && window.__viewer.view.kind === 'case' && (!c || window.__viewer.view.entry.id === c)", arg=cid, timeout=timeout)
        pg.wait_for_timeout(250)

    def canvas_ink(pg, sel):
        return pg.evaluate("""(sel) => { const c = document.querySelector(sel); if (!c) return -1; const t = document.createElement('canvas'); t.width = c.width; t.height = c.height; const g = t.getContext('2d'); g.drawImage(c, 0, 0);
            const d = g.getImageData(0, 0, t.width, t.height).data; const set = new Set(); let n = 0; for (let i = 0; i < d.length; i += 40) { set.add((d[i] >> 4) * 256 + (d[i+1] >> 4) * 16 + (d[i+2] >> 4)); n++; } return set.size; }""", sel)

    # ---------------------------------------------------------------- first load: real case, 2D
    pg, logs, ctx = new_page(f"http://127.0.0.1:{port}/index.html")
    wait_case(pg)
    first = pg.evaluate("window.__viewer.view.entry.id")
    ok("first load shows the first case of the first suite", first == "SC-INTERIOR-WALK", first)
    ok("hash holds the bare case id", pg.evaluate("location.hash") == "#SC-INTERIOR-WALK", pg.evaluate("location.hash"))
    for _ in range(30):
        if canvas_ink(pg, ".stage-canvas") > 12: break
        pg.wait_for_timeout(200)
    ok("2D canvas has content (distinct colours)", canvas_ink(pg, ".stage-canvas") > 12, str(canvas_ink(pg, ".stage-canvas")))
    info = pg.evaluate("window.__viewer.view.stage.renderer.getInfo()")
    ok("2D: two panes side by side", len(info["panes"]) == 2 and info["panes"][0]["y"] == info["panes"][1]["y"], json.dumps(info["panes"]))
    ok("2D: same metric scale in both panes (locked)", info["lock"] is True)
    ok("case header shows the rule", "Inside a room" in pg.inner_text("#p-header"))
    ok("producer FAIL chip shown for the failing case", "FAIL" in pg.inner_text("#p-header"))
    ok("failing expectation row rendered red", pg.locator("#p-results tr.bad").count() == 2, str(pg.locator("#p-results tr.bad").count()))
    ok("guard column marked failing", pg.locator("#p-results .metric.bad").count() == 1)
    ok("verdict UI disabled with the published-page line", "Verdicts are recorded on the published page" in pg.inner_text("#p-verdict") and pg.locator("#p-verdict button:not([disabled])").count() == 0)
    ok("plots built from existing channels", pg.evaluate("window.__viewer.view.plots.count") >= 8, str(pg.evaluate("window.__viewer.view.plots.count")))

    # time / HUD / scope at tick 70 (INTERIOR, status change etc.)
    pg.evaluate("window.__viewer.view.timeline.set(45)")
    pg.wait_for_timeout(200)
    hud = pg.inner_text(".hud")
    ok("HUD shows mode badge and status chip at tick 45", "mode INTERIOR" in hud and "INEXACT" in hud and "null" not in hud, hud.replace("\n", " | "))
    ok("inspector lists channels at the tick", "receiver.proposal" in pg.inner_text("#p-inspector") and "tick 45" in pg.inner_text("#p-inspector"))
    pg.click("#p-inspector button[aria-pressed]")
    ok("raw JSON on demand", pg.locator("#p-inspector pre").count() == 1 and "receiver.relations" in pg.inner_text("#p-inspector pre"))
    pg.click("#p-inspector button[aria-pressed]")
    # pick + tooltip: dropped shelf (source space) while in INTERIOR
    ent = pg.evaluate("""() => { const r = window.__viewer.view.stage.renderer; const i = r.getInfo(); const p = i.panes[0]; const sc = window.__viewer.view.scene;
        const e = sc.spaces.get('source').elems.get(7); const c = e.centroid; const [sx, sy] = r._toScreen(r.panes[0], c[0], c[1]); return r.pick(sx, sy); }""")
    ok("2D pick returns the element under the cursor", ent and ent.get("kind") == "element" and ent.get("id") == 7, json.dumps(ent))
    pg.locator(".stage-wrap").scroll_into_view_if_needed(); pg.wait_for_timeout(100)
    box = pg.locator(".stage-wrap").bounding_box()
    pt = pg.evaluate("""() => { const r = window.__viewer.view.stage.renderer; const e = window.__viewer.view.scene.spaces.get('source').elems.get(7); const [x, y] = r._toScreen(r.panes[0], e.centroid[0], e.centroid[1]); return [x, y]; }""")
    pg.mouse.move(box["x"] + pt[0], box["y"] + pt[1]); pg.wait_for_timeout(250)
    tip = pg.inner_text(".tip") if pg.locator(".tip:not([hidden])").count() else ""
    ok("hover tooltip gives the drop reason", "beyond horizon radius" in tip, tip.replace("\n", " | "))
    ok("hover highlights the receiver counterpart", pg.evaluate("window.__viewer.view.stage.renderer.highlight && window.__viewer.view.stage.renderer.highlight.id") == 7)
    # layers
    pg.evaluate("window.__viewer.view.stage.state.layers.delete('relations')")
    pg.locator(".layers-menu > summary").click()
    pg.locator("input[data-layer='trail.source']").uncheck()
    ok("layer toggle applies", "trail.source" not in pg.evaluate("[...window.__viewer.view.stage.renderer.layers]"))
    pg.keyboard.press("Escape")
    # theme: dark via data-theme + redraw
    pg.click("#themeBtn"); pg.click("#themeBtn"); pg.wait_for_timeout(300)
    ok("theme toggle -> data-theme=dark", pg.evaluate("document.documentElement.getAttribute('data-theme')") == "dark")
    ok("canvas redrawn on theme change (theme re-read)", pg.evaluate("window.__viewer.view.stage.renderer.theme === null || window.__viewer.view.stage.renderer.theme.bg") not in ("#fbfcfd",))
    # 3D
    pg.click("#p-stage .seg button:has-text('3D')")
    pg.wait_for_function("window.__viewer.view.stage.renderer && window.__viewer.view.stage.renderer.kind === 'perspective3d' && window.__viewer.view.stage.renderer.panes.length > 0", timeout=20000)
    pg.wait_for_timeout(600)
    pg.evaluate("window.__viewer.view.stage.renderer.render()")   # counters are filled by a render; do not race the debounced one
    i3 = pg.evaluate("window.__viewer.view.stage.renderer.getInfo()")
    ok("3D renderer mounted", i3["kind"] == "perspective3d", json.dumps({k: i3[k] for k in i3 if k != 'panes'}))
    ok("3D canvas has content", canvas_ink(pg, ".stage-canvas") > 12 if False else pg.evaluate("""() => { const r = window.__viewer.view.stage.renderer; const gl = r.gl.getContext(); const W = gl.drawingBufferWidth, H = gl.drawingBufferHeight; const px = new Uint8Array(W * H * 4); gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px); const s = new Set(); for (let i = 0; i < px.length; i += 160) s.add((px[i] >> 4) * 256 + (px[i+1] >> 4) * 16 + (px[i+2] >> 4)); return s.size; }""") > 12)
    ok("3D: extruded footprints only where shape3d exists", i3["extruded"] >= 4 and i3["flat"] >= 4, f"extruded={i3['extruded']} flat={i3['flat']}")
    ok("3D: no stature stated -> a capsule marker per actor, no invented humanoid", i3["capsules"] == 2 and i3["humanoids"] == 0, f"capsules={i3['capsules']} humanoids={i3['humanoids']}")
    p3 = pg.evaluate("""() => { const r = window.__viewer.view.stage.renderer; const d = r.pdata.get('source'); const p = r.panes[0]; const S = window.__viewer.view.scene.at(r.tick); r._applyCamera(p, d, S);
        const a = S.spaces.source.actors[0].pose.p; const w = new d.camera.position.constructor(a[0], a[1], 1.0).project(d.camera); return [p.x + (w.x + 1) / 2 * p.w, p.y + (1 - w.y) / 2 * p.h]; }""")
    pk = pg.evaluate("(p) => window.__viewer.view.stage.renderer.pick(p[0], p[1])", p3)
    ok("3D pick returns the actor", pk and pk.get("kind") == "actor", json.dumps(pk))
    pg.evaluate("window.__viewer.view.timeline.set(80)"); pg.wait_for_timeout(300)
    ok("3D: time cursor shared (HUD tick)", "tick 80/" in pg.inner_text(".hud"))
    ok("no console errors (2D+3D, first case)", not errs(logs), "; ".join(errs(logs))[:300])
    warns = [t for k, t in logs["console"] if k == "warning"]
    print("   console warnings:", warns[:5] if warns else "none")
    ctx.close()

    # ---------------------------------------------------------------- every case: 2D then 3D, no errors
    pg, logs, ctx = new_page(f"http://127.0.0.1:{port}/index.html")
    wait_case(pg)
    idx = json.load(open(os.path.join(args.data, "index.json")))
    for c in idx["cases"]:
        pg.evaluate("(id) => { location.hash = encodeURIComponent(id); }", c["id"])
        wait_case(pg, c["id"])
        for kind in ("2d", "3d"):
            pg.evaluate("(k) => window.__viewer.view.stage.setKind(k)", kind)
            pg.wait_for_timeout(300)
            pg.evaluate("window.__viewer.view.timeline.set(Math.floor(window.__viewer.view.scene.n / 2))"); pg.wait_for_timeout(150)
            ok(f"case {c['id']} renders in {kind}", pg.evaluate("window.__viewer.view.stage.renderer.getInfo().kind") == ("perspective3d" if kind == "3d" else "topdown2d"))
    ok("no console errors across every case", not errs(logs), "; ".join(errs(logs))[:400])
    # synthetic skeleton
    pg.evaluate("location.hash = 'SYNTH-TWO-JOINT'"); wait_case(pg, "SYNTH-TWO-JOINT")
    pg.evaluate("window.__viewer.view.stage.setKind('3d')"); pg.wait_for_timeout(500)
    i3 = pg.evaluate("window.__viewer.view.stage.renderer.getInfo()")
    ok("3D draws skeleton bones from the joints channel (synthetic fixture)", i3["bones"] == 1 and i3["joints"] == 2, f"bones={i3['bones']} joints={i3['joints']}")
    ok("synthetic fixture is marked SYNTHETIC in the UI", "SYNTHETIC" in pg.inner_text("#p-header") and "SYNTHETIC" in pg.inner_text(".hud"))
    pg.evaluate("window.__viewer.view.stage.setKind('2d')"); pg.wait_for_timeout(300)
    # unknown channel type listed
    pg.evaluate("location.hash = 'SC-FACING-FUTURE'"); wait_case(pg, "SC-FACING-FUTURE")
    ok("unknown channel type listed in the inspector", "gaze_cone_v2" in pg.inner_text("#p-inspector") and "listed and ignored" in pg.inner_text("#p-inspector"))
    ok("provenance sha mismatch warns", "produced at dev111111" in pg.inner_text("#p-header"), "")
    # continuous channel: interpolation labelled
    pg.evaluate("location.hash = 'SC-CONTINUOUS'"); wait_case(pg, "SC-CONTINUOUS")
    pg.evaluate("window.__viewer.view.stage.setTime(3.5)"); pg.wait_for_timeout(150)
    ok("continuous channel interpolates only at fractional ticks, labelled", "INTERPOLATED" in pg.inner_text(".hud"))
    pg.evaluate("window.__viewer.view.stage.setTime(3)"); pg.wait_for_timeout(150)
    ok("integer tick: no interpolation", "INTERPOLATED" not in pg.inner_text(".hud"))
    # play/pause
    pg.evaluate("location.hash = 'SC-INTERIOR-WALK'"); wait_case(pg, "SC-INTERIOR-WALK")
    pg.click("#p-timeline button:has-text('Play')"); pg.wait_for_timeout(700)
    t_play = pg.evaluate("window.__viewer.view.timeline.tick")
    pg.click("#p-timeline button[aria-label='pause']")
    ok("play advances the tick, pause stops it", t_play > 1, str(t_play))
    t0 = pg.evaluate("window.__viewer.view.timeline.tick"); pg.click("#p-timeline button[aria-label='step forward one tick']")
    ok("step +1", pg.evaluate("window.__viewer.view.timeline.tick") == round(t0) + 1)
    ok("event marks listed", "WEIGHT_SWITCH" in pg.inner_text("#p-timeline") and "FACING_ROOT_SWITCH" in pg.inner_text("#p-timeline"))
    pg.evaluate("(() => { const v = window.__viewer.view; v.stage.setKind('2d'); v.timeline.set(20); })()")
    try: pg.wait_for_function("window.__viewer.view.stage.renderer.getInfo().glyphs > 0", timeout=5000)
    except Exception: pass
    gi = pg.evaluate("window.__viewer.view.stage.renderer.getInfo().glyphs")
    ok("2D glyph drawn per actor (source + receiver body)", gi == 2, str(gi))
    ok("glyph legend states size and walk are derived, not data", "glyph size, not data" in pg.inner_text("#p-stage") and "not a motion model" in pg.inner_text("#p-stage"))
    # additive conventions (parity/eval/extras/border/bounds, spec-only and unit cases)
    pg.evaluate("location.hash = 'SC-INTERIOR-WALK'"); wait_case(pg, "SC-INTERIOR-WALK")
    pg.evaluate("window.__viewer.view.timeline.set(31)"); pg.wait_for_timeout(200)
    mk = pg.evaluate("[...window.__viewer.view.scene.at(31).spaces.source.marks.values()].filter(m => m.parity.length).length")
    ok("parity.* scope drawn as a second marked set (not the main scope)", mk >= 1 and pg.evaluate("window.__viewer.view.scene.at(31).spaces.source.scope.channel") == "source.scope", str(mk))
    ok("extra receiver poses listed in the legend by channel id", "parity.v1.replay" in pg.inner_text("#p-stage") and "eval.v1.receiver_trace" in pg.inner_text("#p-stage"))
    plots = pg.evaluate("[...document.querySelectorAll('#p-plots .plot h3')].map(e => e.textContent)")
    ok("one panel per eval / parity group", any(t.startswith("eval scalars") for t in plots) and any(t.startswith("parity scalars") for t in plots), str(plots))
    ok("channel bounds drawn on the step-ratio plot", pg.evaluate("window.__viewer.view.plots && true") and "owner bound 1.5" in pg.evaluate("JSON.stringify(window.__viewer.view.scene.chById.get('receiver.step_ratio').bounds)"))
    ok("scope border solid vs dashed is carried per tick", pg.evaluate("[window.__viewer.view.scene.at(5).spaces.source.scope.border, window.__viewer.view.scene.at(25).spaces.source.scope.border]") == [1, 0])
    pg.evaluate("location.hash = 'CMP-TWO-ENGINES'"); wait_case(pg, "CMP-TWO-ENGINES"); pg.wait_for_timeout(300)
    pg.evaluate("window.__viewer.view.stage.setKind('2d')"); pg.wait_for_timeout(300)
    pn = pg.evaluate("window.__viewer.view.stage.renderer.getInfo().panes.map(p => p.space + ':' + p.engine)")
    ok("compare: toggle present and ON by default, 4 panes (source | v2 | v1 | v1eqw)", pg.locator("#p-stage label:has-text('Compare') input").is_checked() and pn == ["source:None", "receiver:v2", "receiver:v1", "receiver:v1eqw"] or pn == ["source:null", "receiver:v2", "receiver:v1", "receiver:v1eqw"], str(pn))
    pg.locator("#p-stage label:has-text('Overlay') input").check(); pg.wait_for_timeout(300)
    lb = pg.evaluate("window.__viewer.view.stage.renderer.getInfo().labels.map(l => l.text)")
    ok("compare overlay: v2-v1 distance label placed", any(t.startswith("v2-v1 ") for t in lb), str(lb)[:120])
    pg.locator("#p-stage label:has-text('Compare') input").uncheck(); pg.wait_for_timeout(300)
    ok("compare off: back to two panes", pg.evaluate("window.__viewer.view.stage.renderer.getInfo().panes.length") == 2)
    pg.locator("#p-stage label:has-text('Compare') input").check()
    ok("compare plots built (error + distance)", any("Relation error" in t for t in pg.evaluate("[...document.querySelectorAll('#p-plots .plot h3')].map(e => e.textContent)")) and any("Body distance" in t for t in pg.evaluate("[...document.querySelectorAll('#p-plots .plot h3')].map(e => e.textContent)")))
    ok("compare: no console errors", not errs(logs))
    pg.evaluate("location.hash = 'SPEC-ONLY'"); wait_case(pg, "SPEC-ONLY")
    ok("spec-only case: header + results, no stage, no errors", "spec only" in pg.inner_text("#p-timeline") and pg.locator("#p-stage .stage-wrap:visible").count() == 0 and "documented only" in pg.inner_text("#p-results") and not errs(logs))
    pg.evaluate("location.hash = 'UNIT-ONE-SPACE'"); wait_case(pg, "UNIT-ONE-SPACE")
    try: pg.wait_for_function("document.querySelector('#p-timeline').innerText.includes('pose 0/2')", timeout=5000)
    except Exception: pass
    ok("unit case (one space, dt_s=1) labels poses", "pose 0/2" in pg.inner_text("#p-timeline") and pg.evaluate("window.__viewer.view.stage.renderer.getInfo().panes.length") == 1, pg.inner_text("#p-timeline")[40:90].replace("\n", " "))
    pg.evaluate("window.__viewer.view.stage.setKind('3d')"); pg.wait_for_timeout(300)
    ok("unit case renders in 3D", pg.evaluate("window.__viewer.view.stage.renderer.getInfo().kind") == "perspective3d" and not errs(logs))
    pg.evaluate("location.hash = 'eval-1'"); pg.wait_for_function("window.__viewer.view && window.__viewer.view.kind === 'table'"); pg.wait_for_timeout(300)
    ok("text table renders as a plain table", pg.evaluate("window.__viewer.view.tv.info().rows") == 40)
    # browser filters
    pg.fill("#browserBody input[type=search]", "parity")
    ok("browser free-text filter", pg.locator("#caseList .case-row").count() == 1, str(pg.locator("#caseList .case-row").count()))
    pg.fill("#browserBody input[type=search]", "")
    pg.select_option("#browserBody select[aria-label='pass or fail']", "fail")
    ok("browser pass/fail filter", pg.locator("#caseList .case-row:not(:has(.chip:text('TABLE')))").count() == 1, str(pg.locator("#caseList .case-row").count()))
    pg.select_option("#browserBody select[aria-label='pass or fail']", "")
    ok("browser sections per suite", pg.locator("#caseList details.suite").count() == 4)
    # table suite
    pg.evaluate("location.hash = 'matrix-summary-0'"); pg.wait_for_function("window.__viewer.view && window.__viewer.view.kind === 'table'"); pg.wait_for_timeout(400)
    ti = pg.evaluate("window.__viewer.view.tv.info()")
    ok("table virtualized (5000 rows, few DOM rows)", ti["rows"] == 5000 and 0 < ti["domRows"] < 120, json.dumps(ti))
    pg.evaluate("document.querySelector('.tv-scroll').scrollTop = 26 * 3000"); pg.wait_for_timeout(300)
    ti2 = pg.evaluate("window.__viewer.view.tv.info()")
    ok("table scroll renders a different window", ti2["first"] > 2900, json.dumps(ti2))
    pg.fill(".tv-filters th, .tv-filters div:nth-child(4) input", "INTERIOR"); pg.wait_for_timeout(400)
    ti3 = pg.evaluate("window.__viewer.view.tv.info()")
    ok("table column filter", 0 < ti3["shown"] < 5000, json.dumps(ti3))
    pg.fill(".tv-filters div:nth-child(4) input", ""); pg.fill(".tv-filters div:nth-child(1) input", "MX-0001"); pg.wait_for_timeout(400)
    ok("table row with trace opens the case", pg.locator(".tv-row.trace").count() == 1)
    pg.locator(".tv-row.trace").click(); wait_case(pg, "MX-0001")
    ok("click on a trace row navigated to the case", pg.evaluate("location.hash") == "#MX-0001")
    ctx.close()

    # ---------------------------------------------------------------- sort
    pg, logs, ctx = new_page(f"http://127.0.0.1:{port}/index.html#matrix-summary-0")
    pg.wait_for_function("window.__viewer && window.__viewer.view && window.__viewer.view.kind === 'table'"); pg.wait_for_timeout(400)
    pg.click(".tv-head button:has-text('R_err_max')"); pg.wait_for_timeout(300); pg.click(".tv-head button:has-text('R_err_max')"); pg.wait_for_timeout(300)
    vals = pg.evaluate("[...document.querySelectorAll('.tv-row')].slice(0,5).map(r => parseFloat(r.children[5].textContent))")
    ok("table sort descending by a numeric column", vals == sorted(vals, reverse=True) and vals[0] > vals[-1], str(vals))
    ctx.close()

    # ---------------------------------------------------------------- 400 px, no horizontal scroll
    pg, logs, ctx = new_page(f"http://127.0.0.1:{port}/index.html", w=400, h=800)
    wait_case(pg)
    sw = pg.evaluate("document.scrollingElement.scrollWidth")
    ok("400px: no horizontal page scroll", sw <= 400, f"scrollWidth={sw}")
    i = pg.evaluate("window.__viewer.view.stage.renderer.getInfo()")
    ok("400px: panes stacked, stage usable", len(i["panes"]) == 2 and i["panes"][0]["x"] == i["panes"][1]["x"] and i["panes"][0]["h"] >= 150, json.dumps(i["panes"]))
    ok("400px: no console errors", not errs(logs), "; ".join(errs(logs))[:200])
    pg.screenshot(path=os.path.join(base, "shot-400.png"))
    ctx.close()

    # ---------------------------------------------------------------- verdicts with a mocked runtime
    seed = json.dumps({"scope-catalog__SC-FACING-FUTURE": {"verdict": "wrong", "note": "old", "case_id": "SC-FACING-FUTURE", "suite": "scope-catalog", "core_sha": "oldsha", "schema_version": "1.0.0", "by": "u-owner", "at": "2026-01-01T00:00:00Z"}})
    pg, logs, ctx = new_page(f"http://127.0.0.1:{port}/index.html", init="window.__SEED = " + json.dumps(seed) + ";" + MOCK)
    wait_case(pg); pg.wait_for_timeout(500)
    ok("verdict UI enabled with the db + user capabilities", pg.locator("#p-verdict button:not([disabled])").count() >= 3 and "published page" not in pg.inner_text("#p-verdict"))
    ok("subscribed exactly once", pg.evaluate("window.__subs") == 1, str(pg.evaluate("window.__subs")))
    ok("STALE verdict shown, not hidden", pg.locator("#caseList .chip.stale").count() == 1 and "STALE" in pg.locator("#caseList .chip.stale").inner_text())
    pg.click("#p-verdict button.v-ok"); pg.wait_for_timeout(100)
    ok("selecting a verdict writes nothing", pg.evaluate("window.__writes.length") == 0)
    pg.fill("#p-verdict textarea", "looks right"); pg.click("#p-verdict button:has-text('Save verdict')"); pg.wait_for_timeout(300)
    w = pg.evaluate("window.__writes")
    ok("one write per Save click, correct doc + fields", len(w) == 1 and w[0]["id"] == "scope-catalog__SC-INTERIOR-WALK" and w[0]["d"]["verdict"] == "ok" and w[0]["d"]["by"] == "u-owner" and w[0]["d"]["core_sha"].startswith("dev000") and w[0]["d"]["note"] == "looks right" and w[0]["d"]["schema_version"] == "1.0.0", json.dumps(w)[:300])
    ok("verdict chip appears in the browser", pg.locator("#caseList .case-row[data-id='SC-INTERIOR-WALK'] .chip.v-ok").count() == 1)
    ok("author resolved via user.profiles", "Owner Name" in pg.inner_text("#p-verdict"))
    pg.select_option("#browserBody select[aria-label='verdict']", "stale"); pg.wait_for_timeout(200)
    ok("verdict filter (stale)", pg.locator("#caseList .case-row").count() == 1)
    ok("no console errors (verdicts)", not errs(logs), "; ".join(errs(logs))[:200])
    ctx.close()

    # ---------------------------------------------------------------- missing data dir
    pg, logs, ctx = new_page(f"http://127.0.0.1:{port2}/index.html")
    pg.wait_for_selector("#main .panel h2"); 
    txt = pg.inner_text("#main")
    ok("missing data dir shows a clear message", "No data to show" in txt and "index.json" in txt, txt.replace("\n", " | ")[:160])
    ctx.close()
    br.close()

for p in (srv, srv2):
    p.send_signal(signal.SIGTERM); p.wait(timeout=5)
print(f"servers killed (pids {srv.pid}, {srv2.pid})")
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
