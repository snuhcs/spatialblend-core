// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// perspective3d: lit, shaded, textured 3D view (three.js 0.160.0 ES module, loaded lazily on first use).
// Same interface as topdown2d. One WebGL renderer + one canvas for all panes (viewport/scissor). Z is up (the schema's world frame).
//   furniture  : procedural, display-derived, from footprint + shape3d + element.parts (furniture.js)
//   people     : shaded humanoid; joints channel (MotionGen ...) else display IK from pose3d anchors (figure/pose.js)
//   cameras    : orbit (drag / wheel / keys) and follow (the actor of the pane)
// The viewer draws only what the case exports; display-only values are labelled in the HUD and the inspector.
import { readTheme, familyColor, watchTheme } from './theme.js';
import { visibleSpaces, layoutPanes, DEFAULT_LAYERS, extraKey } from './renderer.js';
import { clamp, statusClass, debounceRaf } from '../util.js';
import { loadThree, threeInfo } from './three-loader.js';
import { makeTextures } from './textures.js';
import { GeoCache, buildFurniture, DISPLAY as FDISPLAY } from './furniture.js';
import { FigureMesh, makeFigureGeos } from './figure3d.js';
import { figureFor, staticFigure, MARKER_CAPSULE } from '../figure/pose.js';
import { statureFor } from '../model/display.js';
import { jointMap, decorate as decorateFor } from '../figure/figure.js';

const KIND_RANK = { room: 0, region: 1, surface: 2, interaction_region: 3, object: 4, obstacle: 5, other: 4, border: 7 };
const kindFill = (th, k) => ({ room: th.kRoom, region: th.kRegion, surface: th.kSurface, object: th.kObject, obstacle: th.kObstacle, interaction_region: th.kIregion, border: th.kBorder }[k] || th.kObject);
const DECAL_KINDS = new Set(['region', 'interaction_region', 'surface']);

export class Perspective3D {
  constructor(THREE) {
    this.kind = 'perspective3d';
    this.T = THREE;
    this.layers = new Set(DEFAULT_LAYERS);
    this.scene = null; this.tick = 0; this.highlight = null;
    this.orbit = { az: -0.95, el: 0.75, dist: 20, pan: [0, 0] };
    this.lock = true; this.per = {}; this.mode = 'orbit';
    this.panes = []; this.pdata = new Map();
    this.userMoved = false; this.pointers = new Map();
    this.compare = { on: true, overlay: false };
    this.info = { bones: 0, joints: 0, extruded: 0, flat: 0, furniture: 0, parts: 0, humanoids: 0, ik: 0, capsules: 0 };
    this.figures = [];
    this._draw = debounceRaf(() => this.render());
  }

  /** Loads three.js (once per page, lazily) and returns a mounted renderer. */
  static async create(el) {
    const THREE = await loadThree();
    const r = new Perspective3D(THREE);
    r.mount(el);
    return r;
  }
  static get threeInfo() { return threeInfo; }

  mount(el) {
    const T = this.T;
    this.el = el;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'stage-canvas';
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute('role', 'img');
    this.canvas.setAttribute('aria-label', '3D view of the source and receiver spaces. Drag to orbit, shift-drag or right-drag to pan, wheel to zoom, arrow keys orbit, plus and minus zoom, 0 resets.');
    el.append(this.canvas);
    this.tags = document.createElement('div'); this.tags.className = 'pane-tags'; el.append(this.tags);
    this.gl = new T.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false, preserveDrawingBuffer: true });
    this.gl.shadowMap.enabled = true; this.gl.shadowMap.type = T.PCFSoftShadowMap;
    this.gl.toneMapping = T.ACESFilmicToneMapping; this.gl.toneMappingExposure = 1.0;
    this.raycaster = new T.Raycaster();
    const aniso = Math.min(8, this.gl.capabilities.getMaxAnisotropy?.() || 1);
    this.tex = makeTextures(T, aniso);
    this.geo = new GeoCache(T);
    this.figGeo = makeFigureGeos(T);
    this._geoY = new Map();
    this.ctx = {
      THREE: T, tex: this.tex, geo: this.geo, fig: this.figGeo,
      geoY: (r, L) => { const k = `${r.toFixed(3)}|${L.toFixed(3)}`; let g = this._geoY.get(k); if (!g) { g = new T.CylinderGeometry(r, r, Math.max(L, 1e-3), 8, 1); g.userData.shared = true; this._geoY.set(k, g); } return g; },
      roomCenter: (sp) => { const rm = sp.list.find((e) => e.kind === 'room' && e.centroid); if (rm) return rm.centroid; const b = this.scene.bounds.get(sp.id); return [b.cx, b.cy]; },
      kindColor: (k) => kindFill(this.theme || readTheme(), k),
    };
    this.ro = new ResizeObserver(() => this._resize());
    this.ro.observe(el);
    this._unTheme = watchTheme(() => { this.theme = null; this._rebuild(); });
    const c = this.canvas;
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener('pointerdown', (e) => this._down(e));
    c.addEventListener('pointermove', (e) => this._move(e));
    c.addEventListener('pointerup', (e) => this.pointers.delete(e.pointerId));
    c.addEventListener('pointercancel', (e) => this.pointers.delete(e.pointerId));
    c.addEventListener('wheel', (e) => { e.preventDefault(); const r = c.getBoundingClientRect(); this._zoom(Math.exp(e.deltaY * 0.0012), this._paneAt(e.clientX - r.left, e.clientY - r.top)); }, { passive: false });
    c.addEventListener('keydown', (e) => this._key(e));
    this._resize();
  }

  /** Detach from the page but keep the GL context, textures and geometry caches (the next case adopts it). */
  park() {
    try { this.ro?.disconnect(); } catch (e) { /* */ }
    this._disposeScenes(); this._disposeCaches(); this.scene = null; this.panes = []; this.figures = []; this.highlight = null; this.per = {};
    this.canvas?.remove(); this.tags?.remove(); this.tags?.replaceChildren();
  }
  adopt(el) {
    this.el = el; el.append(this.canvas, this.tags);
    this.ro = new ResizeObserver(() => this._resize()); this.ro.observe(el);
    this._resize();
  }

  dispose() {
    try { this.ro?.disconnect(); } catch (e) { /* */ }
    try { this._unTheme?.(); } catch (e) { /* */ }
    this._disposeScenes();
    try { this.tex?.dispose(); this.geo?.dispose(); this.figGeo?.dispose(); for (const g of this._geoY?.values() || []) g.dispose(); this._geoY?.clear(); } catch (e) { /* */ }
    try { this.gl?.dispose(); this.gl?.forceContextLoss?.(); } catch (e) { /* */ }
    this.canvas?.remove(); this.tags?.remove();
  }

  // ---- interface
  setScene(scene) { this.scene = scene; this.tick = 0; this.userMoved = false; this.per = {}; this._rebuild(); this._fit(); this._draw(); }
  setTime(t) { this.tick = t; this._draw(); }
  setLayers(set) { this.layers = new Set(set); this._layout(); this._draw(); }
  setHighlight(ent) { this.highlight = ent; this._draw(); }
  setCompare(c = {}) { const prev = this.compare.on; this.compare = { ...this.compare, ...c }; if (prev !== this.compare.on) { this._rebuild(); } this._draw(); }
  setStature() { this._draw(); }
  setCamera(c = {}) {
    if (c.mode) { this.mode = c.mode; this.userMoved = false; }
    if (c.lock !== undefined) {
      if (!c.lock && this.lock) for (const p of this.panes) this.per[p.key] = { ...this.orbit, pan: [...this.orbit.pan] };
      this.lock = !!c.lock;
    }
    if (c.reset || c.fit) { this.per = {}; this.userMoved = false; this._fit(); if (!this.lock) for (const p of this.panes) this.per[p.key] = { ...this.orbit, pan: [0, 0] }; }
    if (c.zoomBy) this._zoom(1 / c.zoomBy, this.panes[0]);
    if (c.az !== undefined) this.orbit.az = c.az;
    if (c.el !== undefined) this.orbit.el = c.el;
    this._draw();
  }
  getInfo() {
    return { kind: this.kind, panes: this.panes.map((p) => ({ space: p.space, engine: p.engine, key: p.key, x: p.x, y: p.y, w: p.w, h: p.h })), lock: this.lock, mode: this.mode,
      three: { ...threeInfo }, textures: this.tex.count(), geometries: this.geo.size, figures: this.figures, ...this.info };
  }

  pick(x, y) {
    const p = this._paneAt(x, y);
    if (!p || !this.scene) return null;
    const d = this.pdata.get(p.key);
    if (!d) return null;
    this._applyCamera(p, d, this.scene.at(this.tick));
    this.raycaster.setFromCamera({ x: ((x - p.x) / p.w) * 2 - 1, y: -((y - p.y) / p.h) * 2 + 1 }, d.camera);
    const list = d.pickables.filter((o) => { for (let q = o; q; q = q.parent) if (!q.visible) return false; return true; });
    const hits = this.raycaster.intersectObjects(list, true);
    for (const h of hits) { let o = h.object; while (o && !o.userData.entity) o = o.parent; if (o?.userData.entity) return o.userData.entity; }
    return null;
  }

  // ---- layout / camera
  _paneAt(x, y) { return this.panes.find((q) => x >= q.x && x <= q.x + q.w && y >= q.y && y <= q.y + q.h) || null; }
  _oc(p) { return this.lock ? this.orbit : (this.per[p.key] ||= { ...this.orbit, pan: [0, 0] }); }
  _resize() {
    const r = this.el.getBoundingClientRect();
    this.W = Math.max(50, Math.floor(r.width)); this.H = Math.max(50, Math.floor(r.height));
    this.gl.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.gl.setSize(this.W, this.H, false);
    this.canvas.style.width = this.W + 'px'; this.canvas.style.height = this.H + 'px';
    this._layout(); this._draw();
  }
  _specs() {
    const ids = visibleSpaces(this.scene, this.layers);
    const out = [];
    const rcv = this.scene.roleSpace('receiver');
    const v1 = this.scene.engines.find((e) => e.name === 'v1');
    for (const id of ids) {
      out.push({ space: id, engine: null, key: id });
      if (this.compare.on && id === rcv && v1) out.push({ space: id, engine: 'v1', key: id + '#v1' });
    }
    return out;
  }
  _layout() {
    if (!this.scene) { this.panes = []; return; }
    const specs = this._specs();
    this.panes = layoutPanes(specs.length, this.W, this.H, 6, this.scene.aspect).map((r, i) => ({ ...r, ...specs[i] }));
    this.tags.replaceChildren();
    for (const p of this.panes) {
      const t = document.createElement('div'); t.className = 'pane-tag'; t.style.left = p.x + 6 + 'px'; t.style.top = p.y + 6 + 'px';
      this.tags.append(t); p.tagEl = t;
    }
    if (!this.userMoved) this._fit();
  }
  _fit() {
    if (!this.scene) return;
    let m = 3;
    for (const f of this.scene.focus.values()) m = Math.max(m, f.w, f.h);   // the actors' paths + referents (the 2D default focus)
    const pa = this.panes.length ? Math.min(...this.panes.map((q) => q.w / q.h)) : 1;
    this.orbit = { az: -0.95, el: 0.78, dist: (m * 1.25) / Math.min(1, pa), pan: [0, 0] };
  }
  _focus(p, d, state) {
    // follow camera: the first actor of the pane (or the engine body); else the space centre
    const b = this.scene.focus.get(p.space) || this.scene.bounds.get(p.space);
    if (this.mode === 'follow') {
      const st = state.spaces[p.space];
      const pos = p.engine ? state.engines?.find((e) => e.name === p.engine)?.pose?.p : st?.actors.find((a) => a.pose)?.pose?.p;
      if (pos) return [pos[0], pos[1], 0.9];
    }
    return [b.cx, b.cy, 0.5];
  }
  _applyCamera(p, d, state) {
    const o = this._oc(p), foc = this._focus(p, d, state);
    const follow = this.mode === 'follow';
    const dist = follow ? Math.max(2.5, o.dist * 0.3) : o.dist;
    const tx = foc[0] + (follow ? 0 : o.pan[0]), ty = foc[1] + (follow ? 0 : o.pan[1]), tz = foc[2];
    d.camera.up.set(0, 0, 1);
    d.camera.position.set(tx + dist * Math.cos(o.el) * Math.cos(o.az), ty + dist * Math.cos(o.el) * Math.sin(o.az), tz + dist * Math.sin(o.el));
    d.camera.lookAt(tx, ty, tz);
    d.camera.aspect = p.w / p.h; d.camera.near = Math.max(0.05, dist / 200); d.camera.far = dist * 40 + 200;
    d.camera.updateProjectionMatrix(); d.camera.updateMatrixWorld();
  }
  _down(e) {
    this.canvas.setPointerCapture?.(e.pointerId);
    const r = this.canvas.getBoundingClientRect();
    this.pointers.set(e.pointerId, { x: e.clientX - r.left, y: e.clientY - r.top, button: e.button, shift: e.shiftKey });
  }
  _move(e) {
    const prev = this.pointers.get(e.pointerId);
    if (!prev) return;
    const r = this.canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    const p = this._paneAt(prev.x, prev.y);
    if (!p) return;
    this.userMoved = true;
    if (this.pointers.size === 2) {
      const other = [...this.pointers.entries()].find(([id]) => id !== e.pointerId)[1];
      const d0 = Math.hypot(prev.x - other.x, prev.y - other.y), d1 = Math.hypot(x - other.x, y - other.y);
      if (d0 > 4) this._zoom(d0 / d1, p);
    } else {
      const o = this._oc(p);
      if ((prev.button === 2 || prev.shift) && this.mode !== 'follow') {
        const k = o.dist * 0.0016;
        const rx = -Math.sin(o.az), ry = Math.cos(o.az), fx = -Math.cos(o.az), fy = -Math.sin(o.az);
        const dx = x - prev.x, dy = y - prev.y;
        o.pan = [o.pan[0] - rx * dx * k + fx * dy * k, o.pan[1] - ry * dx * k + fy * dy * k];
      } else { o.az -= (x - prev.x) * 0.008; o.el = clamp(o.el + (y - prev.y) * 0.008, 0.06, 1.55); }
      if (this.lock) this.orbit = o;
    }
    this.pointers.set(e.pointerId, { ...prev, x, y });
    this._draw();
  }
  _zoom(f, p) { const o = p ? this._oc(p) : this.orbit; o.dist = clamp(o.dist * f, 0.5, 2000); this.userMoved = true; this._draw(); }
  _key(e) {
    const o = this.orbit;
    if (e.key === 'ArrowLeft') o.az -= 0.08; else if (e.key === 'ArrowRight') o.az += 0.08;
    else if (e.key === 'ArrowUp') o.el = clamp(o.el + 0.06, 0.06, 1.55); else if (e.key === 'ArrowDown') o.el = clamp(o.el - 0.06, 0.06, 1.55);
    else if (e.key === '+' || e.key === '=') o.dist *= 0.85; else if (e.key === '-') o.dist *= 1.18;
    else if (e.key === '0') { this.setCamera({ reset: true }); e.preventDefault(); return; }
    else return;
    e.preventDefault(); this._draw();
  }

  // ---- scene building
  _disposeScenes() {
    for (const d of this.pdata.values()) {
      d.scene.traverse((o) => {
        if (o.geometry && !o.geometry.userData.shared) o.geometry.dispose();
        const m = o.material; if (m) (Array.isArray(m) ? m : [m]).forEach((x) => { if (!x.userData.shared) x.dispose?.(); });
      });
      d.light?.shadow?.map?.dispose?.();
      for (const f of d.figs.values()) { f.mesh.dispose(); f.ghost?.dispose(); }
    }
    this.pdata.clear();
  }

  /** Primitive geometries are cached within a case (a hall of identical chairs shares one) and freed at the next case switch. Textures and materials live for the session. */
  _disposeCaches() { this.geo.dispose(); for (const g of this._geoY.values()) g.dispose(); this._geoY.clear(); }

  _rebuild() {
    if (!this.scene || !this.gl) return;
    this._disposeScenes(); this._disposeCaches();
    this.theme = readTheme();
    this.info = { bones: 0, joints: 0, extruded: 0, flat: 0, furniture: 0, parts: 0, humanoids: 0, ik: 0, capsules: 0 };
    this._layoutSpecsForBuild();
    this._layout(); this._draw();
  }
  _layoutSpecsForBuild() {
    // build one pane scene per (space, engine) that can be shown, whatever the current visibility
    const rcv = this.scene.roleSpace('receiver'), v1 = this.scene.engines.find((e) => e.name === 'v1');
    this.info.extruded = this.info.flat = 0;
    const counted = new Set();
    for (const sp of this.scene.spaces.values()) {
      this.pdata.set(sp.id, this._buildSpace(sp, null, !counted.has(sp.id)));
      counted.add(sp.id);
      if (sp.id === rcv && v1 && this.compare.on) this.pdata.set(sp.id + '#v1', this._buildSpace(sp, 'v1', false));
    }
  }

  _buildSpace(sp, engine, count) {
    const T = this.T, th = this.theme, sc = this.scene, M = this.tex.M;
    const S = new T.Scene();
    const camera = new T.PerspectiveCamera(42, 1, 0.1, 500);
    const b = sc.bounds.get(sp.id);
    const d = { scene: S, camera, elems: new Map(), pickables: [], figs: new Map(), statics: [], ghosts: [], rels: [], extras: new Map(), trails: [], skel: new Map(), anchors: new Map(), parts: [], walls: [], engine, light: null };
    const col = (c) => new T.Color(c);

    // lights: hemisphere fill + one shadow-casting directional light fitted to the space
    S.add(new T.HemisphereLight(0xe6eefb, 0x8a7a66, 1.6));
    const span = Math.max(b.w, b.h, 4), sun = new T.DirectionalLight(0xfff1de, 2.6);
    sun.position.set(b.cx + span * 0.35, b.cy - span * 0.6, span * 1.1); sun.target.position.set(b.cx, b.cy, 0);
    sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
    const sh = sun.shadow.camera; const hw = span * 0.85; sh.left = -hw; sh.right = hw; sh.top = hw; sh.bottom = -hw; sh.near = 0.5; sh.far = span * 4;
    sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.03; sun.shadow.radius = 2;
    S.add(sun, sun.target); d.light = sun;

    // reference grid under everything (visible outside the rooms)
    {
      const pts = [], majors = [];
      const x0 = Math.floor(b.x0 - 1), x1 = Math.ceil(b.x1 + 1), y0 = Math.floor(b.y0 - 1), y1 = Math.ceil(b.y1 + 1);
      for (let x = x0; x <= x1; x++) (x % 5 === 0 ? majors : pts).push(x, y0, -0.004, x, y1, -0.004);
      for (let y = y0; y <= y1; y++) (y % 5 === 0 ? majors : pts).push(x0, y, -0.004, x1, y, -0.004);
      for (const [arr, c] of [[pts, th.grid], [majors, th.gridMajor]]) {
        const g = new T.BufferGeometry(); g.setAttribute('position', new T.Float32BufferAttribute(arr, 3));
        const l = new T.LineSegments(g, new T.LineBasicMaterial({ color: col(c) })); l.userData.layer = 'grid'; S.add(l);
      }
    }
    if (sp.free?.outer?.length >= 3) {
      const shape = new T.Shape(sp.free.outer.map((q) => new T.Vector2(q[0], q[1])));
      for (const h of sp.free.holes || []) shape.holes.push(new T.Path(h.map((q) => new T.Vector2(q[0], q[1]))));
      const m = new T.Mesh(new T.ShapeGeometry(shape), new T.MeshBasicMaterial({ color: col(th.fFill), transparent: true, opacity: 0.12, side: T.DoubleSide, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }));
      m.position.z = 0.003; m.userData.layer = 'free'; S.add(m);
    }
    const hasRoom = sp.list.some((e) => e.kind === 'room' && e.poly);
    if (!hasRoom) {
      const f = new T.Mesh(new T.PlaneGeometry(b.w + 2, b.h + 2), M.floor()); f.position.set(b.cx, b.cy, -0.002); f.receiveShadow = true; f.userData.layer = 'footprints';
      f.geometry.attributes.uv.array.forEach((_, i, a) => { a[i] *= (i % 2 ? b.h + 2 : b.w + 2); }); S.add(f);
    }

    let roomIdx = 0;
    for (const e of sp.list) {
      if (!e.poly) continue;
      const rank = KIND_RANK[e.kind] ?? 4;
      const ent = { space: sp.id, kind: 'element', id: e.id };
      let top = 0.02;
      const rec = { kind: e.kind, outline: null, halo: null, base: kindFill(th, e.kind), furn: null };
      if (e.kind === 'room') {
        const shape = new T.Shape(e.poly.map((q) => new T.Vector2(q[0], q[1])));
        const fl = new T.Mesh(new T.ShapeGeometry(shape), M.floor()); fl.position.z = 0.0008 * (roomIdx++); fl.receiveShadow = true; fl.userData.layer = 'footprints'; fl.userData.entity = ent; S.add(fl); d.pickables.push(fl);
        this._walls(T, S, M, e.poly, d);
        if (count) this.info.flat++;
      } else if (e.kind === 'border') {
        // border: drawn as a line only
      } else if (DECAL_KINDS.has(e.kind)) {
        const shape = new T.Shape(e.poly.map((q) => new T.Vector2(q[0], q[1])));
        const dm = new T.Mesh(new T.ShapeGeometry(shape), new T.MeshBasicMaterial({ color: col(kindFill(th, e.kind)), transparent: true, opacity: e.kind === 'surface' ? 0.55 : 0.28, side: T.DoubleSide, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3 }));
        dm.position.z = 0.004 + 0.001 * rank; dm.userData.layer = 'footprints'; dm.userData.entity = ent; S.add(dm); d.pickables.push(dm);
        rec.decal = dm; top = 0.03;
      } else {
        const f = buildFurniture(this.ctx, e, sp);
        if (f) {
          f.group.userData.layer = 'footprints'; f.group.userData.entity = ent; f.group.userData.furn = { type: f.type, by: f.by, note: f.note, yaw: f.yaw, id: e.id };
          S.add(f.group); d.pickables.push(f.group); rec.furn = f;
          const h = e.shape3d?.extrude_height_m;
          if (count) { this.info.furniture++; if (h > 0) this.info.extruded++; else this.info.flat++; }
          top = (f.dims?.h || h || 0.0) + 0.01;
          f.group.traverse((o) => { if (o.isMesh) o.userData.owner = f.group; });
        }
      }
      // halo decal (scope / hover tint) + outline
      if (e.kind !== 'room' && e.kind !== 'border') {
        const shape = new T.Shape(e.poly.map((q) => new T.Vector2(q[0], q[1])));
        const halo = new T.Mesh(new T.ShapeGeometry(shape), new T.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: T.DoubleSide, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 }));
        halo.position.z = 0.012; halo.userData.layer = 'footprints'; S.add(halo); rec.halo = halo;
      }
      const transfer = e.flags?.includes('transfer_leaf');
      const lm = transfer ? new T.LineDashedMaterial({ color: col(th.ink), dashSize: 0.15, gapSize: 0.1 }) : new T.LineBasicMaterial({ color: col(e.kind === 'border' ? th.kBorder : th.ink3) });
      const outline = new T.LineLoop(new T.BufferGeometry().setFromPoints(e.poly.map((q) => new T.Vector3(q[0], q[1], e.kind === 'border' ? 0.03 : 0.02))), lm); if (transfer) outline.computeLineDistances();
      outline.userData.layer = 'footprints'; S.add(outline); rec.outline = outline;
      const pl = new T.LineLoop(new T.BufferGeometry().setFromPoints(e.poly.map((q) => new T.Vector3(q[0], q[1], top + 0.03))), new T.LineDashedMaterial({ color: col(th.s3), dashSize: 0.1, gapSize: 0.08 }));
      pl.computeLineDistances(); pl.visible = false; S.add(pl); rec.parityLine = pl;
      d.elems.set(e.id, rec);
      // schema 1.1 parts: the exported polygon on its own plane (display overlay, layer "parts")
      for (const pt of e.parts || []) {
        const shape = new T.Shape(pt.local.map(([x, y]) => new T.Vector2(x, y)));
        const pm = new T.Mesh(new T.ShapeGeometry(shape), new T.MeshBasicMaterial({ color: col(th.cAfford), transparent: true, opacity: 0.32, side: T.DoubleSide, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -5, polygonOffsetUnits: -5 }));
        pm.geometry.translate(0, 0, 0.004);
        pm.matrixAutoUpdate = false; pm.matrix.set(...pt.m); pm.matrixWorldNeedsUpdate = true; pm.userData.layer = 'parts';
        const ol = new T.LineLoop(new T.BufferGeometry().setFromPoints(pt.local.map(([x, y]) => new T.Vector3(x, y, 0.006))), new T.LineBasicMaterial({ color: col(th.cAfford) }));
        ol.matrixAutoUpdate = false; ol.matrix.set(...pt.m); ol.matrixWorldNeedsUpdate = true; ol.userData.layer = 'parts';
        S.add(pm, ol); d.parts.push({ pm, ol, name: pt.name, el: e.id });
        if (count) this.info.parts++;
      }
    }
    // people (static person elements)
    for (const pr of sc.at(0).spaces[sp.id].people) {
      const fig = staticFigure(pr, statureFor(null));  // unknown stature -> capsule marker
      const fm = new FigureMesh(this.ctx, col(th.person));
      fm.update(fig); fm.group.userData.entity = { space: sp.id, kind: 'element', id: pr.id };
      S.add(fm.group); d.statics.push({ id: pr.id, fm, pr });
    }
    // actors
    if (!engine) {
      for (const a of sc.actors.filter((x) => x.space === sp.id)) {
        const rcv = a.role === 'receiver_body';
        const fm = new FigureMesh(this.ctx, col(rcv ? th.s2 : th.s1));
        fm.group.userData.entity = { space: sp.id, kind: 'actor', id: a.id };
        S.add(fm.group); d.pickables.push(fm.group);
        const f = { mesh: fm, ghost: null, a };
        if (a.lookahead_channel) { f.ghost = new FigureMesh(this.ctx, col(th.warn), { ghost: true }); S.add(f.ghost.group); }
        d.figs.set(a.id, f);
        if (a.jointsCh && a.skeleton) {
          const J = a.jointsCh.data.joint_count, jm = [], bm = [];
          const mat = () => new T.MeshBasicMaterial({ color: col(th.cFacing), depthTest: false, transparent: true });
          for (let i = 0; i < J; i++) {
            const j = new T.Mesh(this._tipGeo(), mat()); j.scale.setScalar(0.03); j.renderOrder = 11; j.userData.layer = 'bones'; j.frustumCulled = false;
            const bn = new T.Mesh(this.ctx.geoY(1, 1), mat()); bn.renderOrder = 10; bn.userData.layer = 'bones'; bn.frustumCulled = false;
            S.add(j, bn); jm.push(j); bm.push(bn);
          }
          const parents = a.skeleton.parents || (a.skeleton.joints || []).map((x) => (typeof x === 'object' ? x.parent : -1));
          d.skel.set(a.id, { jm, bm, J, parents });
        }
      }
    } else {
      const ra = sc.actors.find((x) => x.space === sp.id && x.role === 'receiver_body');
      const fm = new FigureMesh(this.ctx, col(th.eV1));
      fm.group.userData.entity = { space: sp.id, kind: 'actor', id: ra?.id ?? 'v1' };
      S.add(fm.group); d.figs.set('v1', { mesh: fm, ghost: null, a: ra || { body: null } });
    }
    // proposal ghosts (the receiver's proposed body pose), v2 pane only
    if (!engine) for (const ch of sc.channels.filter((c) => c.type === 'proposal' && (sc.chSpace(c) || sc.roleSpace('receiver')) === sp.id)) {
      const g = new T.Group(); g.userData.layer = 'proposal';
      const ra = sc.actors.find((a) => a.id === ch.actor) || sc.actors.find((a) => a.space === sp.id && a.role === 'receiver_body');
      const r = ra?.body?.capsule_radius_m > 0 ? ra.body.capsule_radius_m : MARKER_CAPSULE.r, H = ra?.body?.capsule_height_m > 0 ? ra.body.capsule_height_m : MARKER_CAPSULE.H; // exported capsule, else a display-only marker size (ASSETS.md)
      const geo = new T.CapsuleGeometry(r, Math.max(0.01, H - 2 * r), 3, 12); geo.rotateX(Math.PI / 2);
      const mat = new T.MeshBasicMaterial({ color: col(th.warn), transparent: true, opacity: 0.16, depthWrite: false });
      const m = new T.Mesh(geo, mat); m.position.z = H / 2;
      const wm = new T.Mesh(geo, new T.MeshBasicMaterial({ color: col(th.warn), wireframe: true, transparent: true, opacity: 0.4 })); wm.position.z = H / 2;
      const arrow = new T.Line(new T.BufferGeometry().setFromPoints([new T.Vector3(0, 0, H * 0.6), new T.Vector3(r * 2.6, 0, H * 0.6)]), new T.LineBasicMaterial({ color: col(th.warn) }));
      m.userData.entity = { space: sp.id, kind: 'proposal', index: d.ghosts.length };
      g.add(m, wm, arrow); S.add(g); d.pickables.push(m);
      const dash = new T.Line(new T.BufferGeometry().setFromPoints([new T.Vector3(), new T.Vector3(0, 0, 0.001)]), new T.LineDashedMaterial({ color: col(th.ink), dashSize: 0.12, gapSize: 0.08 }));
      dash.userData.layer = 'proposal'; S.add(dash);
      d.ghosts.push({ ch, group: g, mats: [mat, wm.material, arrow.material], dash, H });
    }
    // extra poses (anchor channels are drawn as anchor markers instead)
    if (!engine) sc.extraChannels.forEach((ch, i) => {
      if (sc.chSpace(ch) !== sp.id || sc.anchorIds.has(ch.id)) return;
      const m = new T.Mesh(i % 2 ? new T.TorusGeometry(0.22, 0.02, 8, 24) : new T.OctahedronGeometry(0.16), new T.MeshBasicMaterial({ color: col(th[extraKey(i)]), wireframe: i % 2 === 0 }));
      m.userData.layer = 'extras'; S.add(m); d.extras.set(ch.id, { mesh: m });
    });
    // trails
    const trails = (sc.trails.get(sp.id) || []).filter((t) => !engine || t.kind === 'body');
    for (const t of trails) {
      const pos = []; let last = null;
      for (const pt of t.pts) { const q = pt || last; if (q) { pos.push(q[0], q[1], q[2] + 0.015); last = q; } }
      if (pos.length < 6) continue;
      const c = t.kind === 'extra' ? th[extraKey(t.idx)] : t.kind === 'proposal' ? th.warn : t.kind === 'body' ? th.s2 : th.s1;
      const mk = (opacity) => { const g = new T.BufferGeometry(); g.setAttribute('position', new T.Float32BufferAttribute(pos, 3)); const l = new T.Line(g, new T.LineBasicMaterial({ color: col(c), transparent: true, opacity })); l.userData.layer = t.kind === 'extra' ? 'extras' : 'trail.' + t.kind; l.frustumCulled = false; S.add(l); return l; };
      const dim = mk(0.3), bright = mk(1);
      d.trails.push({ kind: t.kind, dim, bright, count: pos.length / 3, offsetFirstValid: t.pts.findIndex((p) => p) });
    }
    return d;
  }

  _walls(T, S, M, poly, d) {
    // One single-sided quad per wall segment, facing the inside of the room (back-face culled from outside).
    // Cutaway: a segment whose inside faces the camera AND lies between the camera and the view target is lowered to a curb,
    // so a concave or interior wall never hides the furniture behind it (display choice; the heights are in furniture.js DISPLAY).
    let area = 0; for (let i = 0; i < poly.length; i++) { const p = poly[i], q = poly[(i + 1) % poly.length]; area += p[0] * q[1] - q[0] * p[1]; }
    const pts = area < 0 ? [...poly].reverse() : poly, n = pts.length;
    let run = 0;
    for (let i = 0; i < n; i++) {
      const a = pts[i], b = pts[(i + 1) % n], L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (L < 1e-4) continue;
      const pos = [a[0], a[1], 0, b[0], b[1], 0, b[0], b[1], 1, a[0], a[1], 1];
      const uv = [run, 0, run + L, 0, run + L, 1, run, 1];  // v is scaled with the wall height below
      const g = new T.BufferGeometry(); g.setAttribute('position', new T.Float32BufferAttribute(pos, 3)); g.setAttribute('uv', new T.Float32BufferAttribute(uv, 2)); g.setIndex([0, 2, 1, 0, 3, 2]);
      g.setAttribute('normal', new T.Float32BufferAttribute([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], 3)); // replaced below
      const nx = -(b[1] - a[1]) / L, ny = (b[0] - a[0]) / L; // left of a->b = inside of a CCW polygon
      g.setAttribute('normal', new T.Float32BufferAttribute([nx, ny, 0, nx, ny, 0, nx, ny, 0, nx, ny, 0], 3));
      const w = new T.Mesh(g, M.wall()); w.receiveShadow = true; w.userData.layer = 'walls'; w.scale.z = FDISPLAY.wallH; w.frustumCulled = false;
      S.add(w); d.walls.push({ mesh: w, c: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], n: [nx, ny] });
      run += L;
    }
    const base = new T.LineLoop(new T.BufferGeometry().setFromPoints(pts.map((p) => new T.Vector3(p[0], p[1], 0.03))), new T.LineBasicMaterial({ color: 0x6b5a48 })); base.userData.layer = 'footprints'; S.add(base);
  }

  _tipGeo() { return (this._tip ||= (() => { const g = new this.T.SphereGeometry(1, 12, 10); g.userData.shared = true; return g; })()); }

  _relMeshes(d, n) {
    const T = this.T;
    while (d.rels.length < n) {
      const cyl = new T.Mesh(this.ctx.geoY(1, 1), new T.MeshBasicMaterial({ color: 0x888888 }));
      const tip = new T.Mesh(this._tipGeo(), new T.MeshBasicMaterial({ color: 0x888888 }));
      cyl.userData.layer = 'relations'; tip.userData.layer = 'relations';
      d.scene.add(cyl, tip); d.rels.push({ cyl, tip }); d.pickables.push(cyl, tip);
    }
  }

  // ---- per-tick update + render
  render() {
    if (!this.scene || !this.gl || !this.panes.length) return;
    const T = this.T, th = this.theme || (this.theme = readTheme());
    const state = this.scene.at(this.tick);
    const L = this.layers;
    this.gl.setScissorTest(false);
    this.gl.setClearColor(new T.Color(th.bg), 1);
    this.gl.clear();
    this.gl.setScissorTest(true);
    const hlKeys = new Set();
    const hl = this.highlight;
    if (hl?.kind === 'element') { hlKeys.add(`${hl.space}:${hl.id}`); for (const m of this.scene.mates(hl.space, hl.id)) hlKeys.add(`${m.space}:${m.id}`); }
    this.info.bones = 0; this.info.joints = 0; this.info.humanoids = 0; this.info.ik = 0; this.info.capsules = 0; this.figures = [];
    for (const p of this.panes) {
      const d = this.pdata.get(p.key);
      if (!d) continue;
      const st = state.spaces[p.space];
      this._applyCamera(p, d, state);
      this._updatePane(T, th, d, st, state, hlKeys, L, p);
      const y = this.H - p.y - p.h;
      this.gl.setViewport(p.x, y, p.w, p.h); this.gl.setScissor(p.x, y, p.w, p.h);
      this.gl.setClearColor(new T.Color(th.panel), 1);
      this.gl.render(d.scene, d.camera);
      if (p.tagEl) this._tag(p, st, state, d);
    }
    this.gl.setScissorTest(false);
  }

  _tag(p, st, state, d) {
    const role = this.scene.spaces.get(p.space).role;
    const head = `${p.space} · ${role}${p.engine ? ' · ' + p.engine : role === 'receiver' && this.compare.on && this.scene.engines.some((e) => e.name === 'v1') ? ' · v2' : ''}${st.scope?.mode ? ` · mode ${st.scope.mode}` : ''}`;
    const figs = this.figures.filter((f) => f.pane === p.key);
    const rows = [head, ...figs.flatMap((f) => [`${f.actor}: ${f.label.text}${f.label.kind === 'motiongen' && f.label.detail ? ' · ' + f.label.detail : ''}`, ...(f.look ? [`${f.actor}: look-ahead buffer (MotionGen, not yet played)${f.look.revision != null ? ' rev ' + f.look.revision : ''}`] : [])])];
    const key = rows.join('\n');
    if (p.tagKey === key) return;
    p.tagKey = key;
    p.tagEl.replaceChildren(...rows.map((r, i) => { const s = document.createElement('div'); s.textContent = r; if (i) s.className = 'tag-sub'; return s; }));
  }

  _updatePane(T, th, d, st, state, hlKeys, L, p) {
    d.scene.traverse((o) => { const ly = o.userData.layer; if (ly && ly !== 'bones') o.visible = L.has(ly); });
    // scope + highlight on footprints
    for (const [id, e] of d.elems) {
      const mk = st.marks.get(id);
      const scopeOn = L.has('scope') && mk;
      const fam = scopeOn && mk.inScope.length ? mk.inScope[0] : null;
      const cls = fam ? (fam.toLowerCase().startsWith('spac') ? 'spacing' : fam.toLowerCase().startsWith('fac') ? 'facing' : fam.toLowerCase().startsWith('aff') ? 'afford' : 'other') : null;
      let outline = e.kind === 'border' ? th.kBorder : th.ink3, halo = null, op = 0;
      if (cls) { halo = familyColor(th, cls); op = 0.35; outline = halo; }
      else if (scopeOn && mk.dropped.length) { halo = th.ink3; op = 0.18; }
      if (scopeOn && mk.container) outline = th.ink;
      if (hlKeys.has(`${p.space}:${id}`)) { halo = th.hl; op = 0.5; outline = th.hl; }
      if (e.halo) { e.halo.material.color.set(halo || 0xffffff); e.halo.material.opacity = op; e.halo.visible = L.has('footprints') && op > 0; }
      e.outline.material.color.set(outline);
      e.parityLine.visible = L.has('parity') && !!mk?.parity?.length;
    }
    for (const sf of d.statics) { sf.fm.update(staticFigure(sf.pr, statureFor(null))); sf.fm.group.visible = L.has('actors'); }
    // wall cutaway: facing the camera and between it and the view target -> a curb
    {
      const cp = d.camera.position, foc = this._focus(p, d, state), vx = foc[0] - cp.x, vy = foc[1] - cp.y, vl = Math.hypot(vx, vy) || 1;
      for (const w of d.walls) {
        const facing = (cp.x - w.c[0]) * w.n[0] + (cp.y - w.c[1]) * w.n[1] > 0;
        const between = ((w.c[0] - foc[0]) * vx + (w.c[1] - foc[1]) * vy) / vl < 0;
        const hgt = facing && between ? FDISPLAY.curbH : FDISPLAY.wallH;
        w.mesh.scale.z = hgt;
        const uv = w.mesh.geometry.attributes.uv; if (w.h !== hgt) { uv.setY(2, hgt); uv.setY(3, hgt); uv.needsUpdate = true; w.h = hgt; }
      }
    }
    // figures
    if (d.engine) {
      const eng = state.engines?.find((e) => e.name === d.engine);
      const f = d.figs.get('v1');
      const rcvA = st.actors.find((x) => x.role === 'receiver_body');
      if (f) {
        const a = { pose: eng?.pose || null, yaw: eng?.yaw, walk: eng?.walk, body: rcvA?.body || f.a?.body, anchors: {}, statureBase: rcvA?.statureBase || null };
        const fig = eng?.pose ? figureFor(a, { label: { text: 'v1 body: standing display figure at the v1 root', detail: 'root pose only (no joints, no anchors): not model output', kind: 'root' } }) : null;
        f.mesh.update(fig); f.mesh.group.visible = L.has('actors') && !!fig;
        if (fig?.kind === 'capsule') fig.label = { text: 'v1 body: ' + fig.label.text, detail: fig.label.detail, kind: 'unknown' };
        if (fig) { this.info.humanoids++; this.figures.push({ pane: p.key, actor: 'v1', label: fig.label, posture: fig.posture, stature: fig.stature, H: fig.H, kind: fig.kind, residues: {} }); }
      }
    } else for (const a of st.actors) {
      const o = d.figs.get(a.id);
      if (!o) continue;
      const fig = figureFor(a);
      const sk = d.skel.get(a.id);
      const generic = fig?.kind === 'generic';
      o.mesh.update(generic ? null : fig);
      o.mesh.group.visible = L.has('actors') && !!fig && !generic;
      if (o.ghost && fig?.kind === 'capsule') o.ghost.group.visible = false;
      if (fig) {
        this.info.humanoids += generic || fig.kind === 'capsule' ? 0 : 1; if (fig.kind === 'capsule') this.info.capsules++; if (fig.kind === 'ik') this.info.ik++;
        const rec = { pane: p.key, actor: a.id, label: fig.label, posture: fig.posture, stature: fig.stature, H: fig.H, kind: fig.kind, residues: fig.residues || {}, revision: a.jointsMeta?.revision ?? null, horizon: a.jointsMeta?.horizon ?? null, look: a.look ? { revision: a.look.revision, horizon: a.look.horizon } : null };
        this.figures.push(rec);
      }
      // look-ahead ghost: the not-yet-played MotionGen buffer, translucent
      if (o.ghost) {
        const J = a.look ? jointMap(a.skeleton, a.look.joints) : null;
        const H = fig?.H || 1;
        if (J) {
          const lf = { kind: 'joints', J, H, ...decorateFor(J, H) };
          o.ghost.update(lf); o.ghost.group.visible = L.has('actors');
        } else o.ghost.group.visible = false;
      }
      if (sk) {
        const on = a.joints && (generic ? L.has('skeleton') : L.has('bones3d'));
        const Y = new T.Vector3(0, 1, 0);
        sk.jm.forEach((m, i) => {
          const jt = on ? a.joints[i] : null, pj = on && sk.parents[i] >= 0 ? a.joints[sk.parents[i]] : null;
          m.visible = !!jt; const bn = sk.bm[i]; bn.visible = !!(jt && pj);
          if (jt) { m.position.set(...jt.p); this.info.joints++; }
          if (jt && pj) {
            const A = new T.Vector3(...pj.p), B = new T.Vector3(...jt.p), dir = B.clone().sub(A), len = dir.length();
            bn.position.copy(A).add(B).multiplyScalar(0.5); bn.scale.set(0.015, Math.max(len, 1e-4), 0.015);
            if (len > 1e-6) bn.quaternion.setFromUnitVectors(Y, dir.normalize());
            this.info.bones++;
          }
        });
      }
      // anchor markers (exported pose3d anchors)
      for (const [name, an] of Object.entries(a.anchors || {})) {
        let mk = d.anchors.get(a.id + ':' + name);
        if (!mk) {
          const g = new T.Group(); g.userData.layer = 'anchors';
          const s = new T.Mesh(new T.SphereGeometry(0.035, 14, 10), new T.MeshBasicMaterial({ color: new T.Color(th.cAfford), depthTest: false, transparent: true, opacity: 0.95 })); s.renderOrder = 12;
          const r = new T.Mesh(new T.TorusGeometry(0.075, 0.008, 8, 24), new T.MeshBasicMaterial({ color: new T.Color(th.hl), depthTest: false })); r.renderOrder = 12;
          g.add(s, r); d.scene.add(g); mk = { g, s, r }; d.anchors.set(a.id + ':' + name, mk);
        }
        mk.g.visible = L.has('anchors') && !!an.pose;
        if (an.pose) { mk.g.position.set(...an.pose.p); mk.r.visible = an.engaged === true; mk.s.material.opacity = an.engaged === false ? 0.55 : 0.95; }
      }
    }
    // proposals
    d.ghosts.forEach((g) => {
      const pr = st.proposals.find((x) => x.channel === g.ch.id);
      const on = L.has('proposal') && pr?.pose;
      g.group.visible = !!on; g.dash.visible = !!on;
      if (!on) return;
      g.group.position.set(...pr.pose.p); g.group.quaternion.set(...pr.pose.q);
      const c = { good: th.good, warn: th.warn, bad: th.bad, neutral: th.ink3 }[statusClass(pr.status)];
      g.mats.forEach((m) => m.color.set(c));
      if (pr.bodyPose) { g.dash.geometry.setFromPoints([new T.Vector3(...pr.bodyPose.p), new T.Vector3(...pr.pose.p)]); g.dash.computeLineDistances(); } else g.dash.visible = false;
    });
    for (const x of st.extras) {
      const o = d.extras.get(x.id);
      if (!o) continue;
      o.mesh.visible = L.has('extras') && !!x.pose;
      if (x.pose) { o.mesh.position.set(...x.pose.p); o.mesh.quaternion.set(...x.pose.q); o.mesh.position.z += 0.1; }
    }
    // relations: cylinders whose radius carries the weight
    const draw = d.engine ? [] : st.relations.filter((r) => r.kind !== 'none');
    this._relMeshes(d, draw.length);
    const Y = new T.Vector3(0, 1, 0);
    d.rels.forEach((rm, i) => {
      const r = draw[i];
      const on = r && L.has('relations');
      rm.cyl.visible = rm.tip.visible = !!on;
      if (!on) { rm.cyl.userData.entity = rm.tip.userData.entity = null; return; }
      const a = new T.Vector3(...r.a), b = new T.Vector3(...r.b), dir = b.clone().sub(a), len = dir.length();
      const w = isFinite(r.weight) && r.weight !== null ? clamp(r.weight, 0, 1) : 0.2, rad = 0.012 + 0.04 * w, c = familyColor(th, r.fc);
      rm.cyl.material.color.set(c); rm.tip.material.color.set(c);
      rm.cyl.position.copy(a).add(b).multiplyScalar(0.5); rm.cyl.scale.set(rad, Math.max(len, 1e-4), rad);
      if (len > 1e-6) rm.cyl.quaternion.setFromUnitVectors(Y, dir.normalize());
      rm.tip.position.copy(b); rm.tip.scale.setScalar(rad * 2.2 + 0.03);
      rm.cyl.userData.entity = rm.tip.userData.entity = { space: p.space, kind: 'relation', index: st.relations.indexOf(r) };
    });
    const k = Math.floor(state.tick);
    for (const t of d.trails) t.bright.geometry.setDrawRange(0, Math.max(0, Math.min(t.count, clamp(k + 1 - Math.max(0, t.offsetFirstValid), 0, t.count))));
  }
}

