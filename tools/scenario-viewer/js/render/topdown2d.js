// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// topdown2d: canvas renderer. Source and receiver panes side by side at the SAME metric scale.
import { readTheme, familyColor, watchTheme, engineColor } from './theme.js';
import { visibleSpaces, layoutPanes, DEFAULT_LAYERS, extraKey } from './renderer.js';
import { LabelPlacer } from './labels.js';
import { drawGlyph, drawWedge, drawFoot, glyphDims, glyphScale, headPos } from './glyph.js';
import { clamp, pointInPoly, withAlpha, quatForward, fmt, statusClass, debounceRaf } from '../util.js';

const KIND_RANK = { room: 0, region: 1, surface: 2, interaction_region: 3, object: 4, obstacle: 5, person: 6, other: 4, border: 7 };
const kindFill = (th, k) => ({ room: th.kRoom, region: th.kRegion, surface: th.kSurface, object: th.kObject, obstacle: th.kObstacle, interaction_region: th.kIregion, border: th.kBorder }[k] || th.kObject);
const statusColor = (th, label) => ({ good: th.good, warn: th.warn, bad: th.bad, neutral: th.ink3 }[statusClass(label)]);
const yawOfQ = (q) => { const f = quatForward(q); return Math.atan2(f[1], f[0]); };
const MAX_FIT_SCALE = 150;
const niceStep = (target) => { const p = 10 ** Math.floor(Math.log10(target)); for (const m of [1, 2, 5, 10]) if (m * p >= target) return m * p; return 10 * p; };

export class TopDown2D {
  constructor() {
    this.kind = 'topdown2d';
    this.layers = new Set(DEFAULT_LAYERS);
    this.scene = null; this.tick = 0; this.highlight = null;
    this.cam = { lock: true, scale: 40, pan: [0, 0], per: {} };
    this.userMoved = false;
    this.panes = [];
    this._draw = debounceRaf(() => this.draw());
    this.theme = null; this.hatch = null;
    this.pointers = new Map();
    this.labelMode = 'key';
    this.compare = { on: true, overlay: false };
    this.home = {};
    this._labels = [];
    this._glyphRects = [];
  }

  mount(el) {
    this.el = el;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'stage-canvas';
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute('role', 'img');
    this.canvas.setAttribute('aria-label', 'Top-down view of the source and receiver spaces. Drag to pan, wheel to zoom, arrow keys pan, plus and minus zoom, 0 resets.');
    el.append(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.ro = new ResizeObserver(() => this._resize());
    this.ro.observe(el);
    this._unTheme = watchTheme(() => { this.theme = null; this.hatch = null; this._draw(); });
    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => this._down(e));
    c.addEventListener('pointermove', (e) => this._move(e));
    c.addEventListener('pointerup', (e) => this._up(e));
    c.addEventListener('pointercancel', (e) => this._up(e));
    c.addEventListener('wheel', (e) => { e.preventDefault(); const r = c.getBoundingClientRect(); this._zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
    c.addEventListener('keydown', (e) => this._key(e));
    this._resize();
  }

  dispose() {
    try { this.ro?.disconnect(); } catch (e) { /* */ }
    try { this._unTheme?.(); } catch (e) { /* */ }
    this.canvas?.remove();
    this.scene = null;
  }

  // ---- interface
  setScene(scene) { this.scene = scene; this.tick = 0; this.userMoved = false; this.cam.per = {}; this._fit(); this._draw(); }
  setTime(t) { this.tick = t; this._draw(); }
  setLayers(set) { const prev = visibleSpaces(this.scene || { spaceIds: [] }, this.layers).join(); this.layers = new Set(set); if (this.scene && prev !== visibleSpaces(this.scene, this.layers).join()) { this._layout(); if (!this.userMoved) this._fit(); } this._draw(); }
  setCompare(o) { this.compare = { ...this.compare, ...o }; if (this.scene) { this._layout(); if (!this.userMoved) this._fit(); } this._draw(); }
  _cmp() { return !!(this.scene && this.compare.on && this.scene.engines.length); }
  setLabelMode(m) { this.labelMode = m; this._draw(); }
  setHighlight(ent) { this.highlight = ent; this._draw(); }
  setCamera(c = {}) {
    if (c.lock !== undefined && c.lock !== this.cam.lock) {
      if (!c.lock) for (const p of this.panes) this.cam.per[p.key ?? p.space] = { scale: this.cam.scale, pan: [...this.cam.pan] };
      this.cam.lock = !!c.lock;
    }
    if (c.reset || c.fit) { this.userMoved = false; this.cam.per = {}; if (!this.cam.lock) for (const p of this.panes) this.cam.per[p.key ?? p.space] = { scale: this.cam.scale, pan: [0, 0] }; this._fit(); }
    if (c.scale) this.cam.scale = c.scale;
    if (c.pan) this.cam.pan = [...c.pan];
    if (c.zoomBy) { const p = this.panes[0]; if (p) this._zoomAt(p.x + p.w / 2, p.y + p.h / 2, c.zoomBy); }
    this._draw();
  }
  getInfo() { return { glyphPx: glyphScale(glyphDims(null), this.cam.scale) * glyphDims(null).W, focus: this.scene ? Object.fromEntries([...this.scene.focus].map(([k, v]) => [k, v.kind])) : null, labels: this._labels, glyphRects: this._glyphRects, labelMode: this.labelMode, glyphs: this._glyphs ?? 0, kind: this.kind, panes: this.panes.map((p) => ({ space: p.space, engine: p.engine ?? null, x: p.x, y: p.y, w: p.w, h: p.h })), lock: this.cam.lock, scale: this.cam.scale }; }

  pick(x, y) {
    if (!this.scene) return null;
    const p = this.panes.find((q) => x >= q.x && x <= q.x + q.w && y >= q.y && y <= q.y + q.h);
    if (!p) return null;
    const st = this.scene.at(this.tick).spaces[p.space];
    const sp = this.scene.spaces.get(p.space);
    const [wx, wy] = this._toWorld(p, x, y);
    const sc = this._pc(p).scale;
    const near = (pt, r) => pt && Math.hypot(pt[0] - wx, pt[1] - wy) * sc <= r;
    for (const pr of st.proposals.map((q, i) => [q, i])) if (pr[0].pose && near(pr[0].pose.p, 12) && this.layers.has('proposal')) return { space: p.space, kind: 'proposal', index: pr[1] };
    if (this.layers.has('actors')) for (const a of st.actors) if (a.pose && near(a.pose.p, Math.max(10, (a.body?.capsule_radius_m ?? 0.2) * sc + 3))) return { space: p.space, kind: 'actor', id: a.id };
    if (this.layers.has('relations')) {
      let best = null, bd = 6;
      st.relations.forEach((r, i) => {
        if (r.kind === 'none') return;
        const d = this._distSeg(r.a, r.b, wx, wy) * sc;
        if (d < bd) { bd = d; best = { space: p.space, kind: 'relation', index: i }; }
      });
      if (best) return best;
    }
    let hit = null;
    for (const e of sp.list) if (e.poly && pointInPoly(wx, wy, e.poly)) {
      const r = KIND_RANK[e.kind] ?? 4;
      if (!hit || r > hit.r || (r === hit.r && e.area < hit.e.area)) hit = { e, r };
    }
    return hit ? { space: p.space, kind: 'element', id: hit.e.id } : null;
  }

  // ---- geometry helpers
  _distSeg(a, b, x, y) {
    const dx = b[0] - a[0], dy = b[1] - a[1], L = dx * dx + dy * dy;
    const t = L ? clamp(((x - a[0]) * dx + (y - a[1]) * dy) / L, 0, 1) : 0;
    return Math.hypot(a[0] + t * dx - x, a[1] + t * dy - y);
  }
  _pc(p) { return this.cam.lock ? this.cam : (this.cam.per[p.key ?? p.space] ||= { scale: this.cam.scale, pan: [0, 0] }); }
  _ctr(p) { const f = this.scene.focus.get(p.space) || this.scene.bounds.get(p.space); return [f.cx, f.cy]; }
  _toWorld(p, sx, sy) {
    const [ox, oy] = this._ctr(p), c = this._pc(p);
    return [ox + c.pan[0] + (sx - (p.x + p.w / 2)) / c.scale, oy + c.pan[1] - (sy - (p.y + p.h / 2)) / c.scale];
  }
  _toScreen(p, wx, wy) {
    const [ox, oy] = this._ctr(p), c = this._pc(p);
    return [p.x + p.w / 2 + (wx - ox - c.pan[0]) * c.scale, p.y + p.h / 2 - (wy - oy - c.pan[1]) * c.scale];
  }

  _resize() {
    const r = this.el.getBoundingClientRect();
    this.W = Math.max(50, Math.floor(r.width)); this.H = Math.max(50, Math.floor(r.height));
    this.dpr = Math.min(3, window.devicePixelRatio || 1);
    this.canvas.width = Math.round(this.W * this.dpr); this.canvas.height = Math.round(this.H * this.dpr);
    this.canvas.style.width = this.W + 'px'; this.canvas.style.height = this.H + 'px';
    this._layout();
    if (!this.userMoved) this._fit();
    this._draw();
  }
  _layout() {
    if (!this.scene) { this.panes = []; return; }
    const ids = visibleSpaces(this.scene, this.layers), cmp = this._cmp(), items = [];
    for (const id of ids) {
      if (cmp && this.scene.spaces.get(id).role === 'receiver') for (const eng of ['v2', ...this.scene.engines.map((e) => e.name)]) items.push({ space: id, engine: eng, key: `${id}:${eng}` });
      else items.push({ space: id, key: id });
    }
    this.panes = layoutPanes(items.length, this.W, this.H, 6, this.scene.aspect).map((r, i) => ({ ...r, ...items[i] }));
  }
  _fit() {
    if (!this.scene || !this.panes.length) return;
    let s = Infinity;
    for (const p of this.panes) { const f = this.scene.focus.get(p.space); s = Math.min(s, (p.w - 28) / f.w, (p.h - 46) / f.h); }
    // one metric scale for both panes: the smaller fit keeps both paths in view; capped so a stationary case does not zoom absurdly
    this.cam.scale = clamp(s, 0.5, MAX_FIT_SCALE); this.cam.pan = [0, 0];
  }
  _zoomAt(sx, sy, f) {
    const p = this.panes.find((q) => sx >= q.x && sx <= q.x + q.w && sy >= q.y && sy <= q.y + q.h) || this.panes[0];
    if (!p || !this.scene) return;
    this.userMoved = true;
    const c = this._pc(p), [ox, oy] = this._ctr(p);
    const wx = ox + c.pan[0] + (sx - (p.x + p.w / 2)) / c.scale, wy = oy + c.pan[1] - (sy - (p.y + p.h / 2)) / c.scale;
    const s2 = clamp(c.scale * f, 0.5, 4000);
    c.pan = [wx - ox - (sx - (p.x + p.w / 2)) / s2, wy - oy + (sy - (p.y + p.h / 2)) / s2];
    c.scale = s2;
    if (this.cam.lock) this.cam.scale = s2;
    this._draw();
  }
  _down(e) {
    this.canvas.setPointerCapture?.(e.pointerId);
    const r = this.canvas.getBoundingClientRect();
    this.pointers.set(e.pointerId, { x: e.clientX - r.left, y: e.clientY - r.top });
    this._pinch = null;
  }
  _move(e) {
    const r = this.canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const prev = this.pointers.get(e.pointerId);
    if (!prev) return;
    if (this.pointers.size === 1) {
      const p = this.panes.find((q) => prev.x >= q.x && prev.x <= q.x + q.w && prev.y >= q.y && prev.y <= q.y + q.h);
      if (p && this.scene) {
        this.userMoved = true;
        const c = this._pc(p);
        c.pan = [c.pan[0] - (x - prev.x) / c.scale, c.pan[1] + (y - prev.y) / c.scale];
        this._draw();
      }
    } else if (this.pointers.size === 2) {
      const other = [...this.pointers.entries()].find(([id]) => id !== e.pointerId)[1];
      const d0 = Math.hypot(prev.x - other.x, prev.y - other.y), d1 = Math.hypot(x - other.x, y - other.y);
      if (d0 > 4) this._zoomAt((x + other.x) / 2, (y + other.y) / 2, d1 / d0);
    }
    this.pointers.set(e.pointerId, { x, y });
  }
  _up(e) { this.pointers.delete(e.pointerId); }
  _key(e) {
    const step = 30 / this.cam.scale;
    const mv = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }[e.key];
    if (mv) { for (const p of this.panes) { const c = this._pc(p); c.pan = [c.pan[0] + mv[0], c.pan[1] + mv[1]]; if (this.cam.lock) break; } this.userMoved = true; e.preventDefault(); this._draw(); }
    else if (e.key === '+' || e.key === '=') { this.setCamera({ zoomBy: 1.25 }); e.preventDefault(); }
    else if (e.key === '-') { this.setCamera({ zoomBy: 0.8 }); e.preventDefault(); }
    else if (e.key === '0') { this.setCamera({ reset: true }); e.preventDefault(); }
  }

  // ---- drawing
  _hatchPattern(th) {
    if (this.hatch) return this.hatch;
    const c = document.createElement('canvas'); c.width = c.height = 8;
    const g = c.getContext('2d');
    g.strokeStyle = th.ink; g.lineWidth = 1.2; g.globalAlpha = 0.75;
    g.beginPath(); g.moveTo(-2, 10); g.lineTo(10, -2); g.moveTo(-2, 2); g.lineTo(2, -2); g.moveTo(6, 10); g.lineTo(10, 6); g.stroke();
    return (this.hatch = this.ctx.createPattern(c, 'repeat'));
  }

  draw() {
    const ctx = this.ctx;
    if (!ctx) return;
    const th = this.theme || (this.theme = readTheme());
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.W, this.H);
    if (!this.scene) return;
    const state = this.scene.at(this.tick);
    this._glyphs = 0; this._labels = []; this._glyphRects = [];
    for (const p of this.panes) this._drawPane(ctx, th, p, state);
  }

  _path(ctx, p, poly) {
    ctx.beginPath();
    poly.forEach((pt, i) => { const [x, y] = this._toScreen(p, pt[0], pt[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath();
  }

  _drawPane(ctx, th, p, state) {
    const L = this.layers, sp = this.scene.spaces.get(p.space);
    const cmp = this._cmp(), eng = p.engine && p.engine !== 'v2' ? p.engine : null;
    let st = state.spaces[p.space];
    if (eng) st = { ...st, actors: [], proposals: [], relations: [], extras: [], marks: new Map(), scope: null };
    else if (cmp) st = { ...st, extras: st.extras.filter((x) => !this._isEngineCh(x.id)) };
    const LM = this.labelMode;
    const lp = this._lp = new LabelPlacer(p);
    const lab = (mode, it) => { if (LM === 'none' || (mode === 'all' && LM !== 'all')) return; lp.add(it); };
    lp.block(p.x + 4, p.y + 4, 330, cmp && p.engine ? 44 : 24); lp.block(p.x + 4, p.y + p.h - 36, 150, 34);
    if (L.has('hud') && p.x + p.w >= this.W - 2 && p.y <= 1) lp.block(p.x + p.w - 250, p.y, 250, 86);
    const gblock = (x, y, r) => { lp.blockCircle(x, y, r); this._glyphRects.push({ x: x - r, y: y - r, w: 2 * r, h: 2 * r, space: p.space }); };
    const c = this._pc(p), b = this.scene.bounds.get(p.space);
    ctx.save();
    ctx.beginPath(); ctx.rect(p.x, p.y, p.w, p.h); ctx.clip();
    ctx.fillStyle = th.panel; ctx.fillRect(p.x, p.y, p.w, p.h);

    // grid (1 m when it is at least 14 px apart, else a nicer multiple), metre labels on the pane edge
    if (L.has('grid')) {
      const step = niceStep(14 / c.scale);
      const [wx0, wy1] = this._toWorld(p, p.x, p.y), [wx1, wy0] = this._toWorld(p, p.x + p.w, p.y + p.h);
      ctx.lineWidth = 1; ctx.font = `10px ${th.fontMono}`; ctx.textBaseline = 'top';
      for (let gx = Math.ceil(wx0 / step) * step; gx <= wx1; gx += step) {
        const major = Math.abs(Math.round(gx / step) % 5) === 0 || Math.abs(gx) < 1e-9;
        const [sx] = this._toScreen(p, gx, 0);
        ctx.strokeStyle = major ? th.gridMajor : th.grid; ctx.beginPath(); ctx.moveTo(Math.round(sx) + 0.5, p.y); ctx.lineTo(Math.round(sx) + 0.5, p.y + p.h); ctx.stroke();
        if (LM !== 'none' && (major || step * c.scale > 34)) { ctx.fillStyle = th.ink3; ctx.textAlign = 'left'; ctx.fillText(`${+gx.toFixed(2)}`, sx + 2, p.y + p.h - 12); }
      }
      for (let gy = Math.ceil(wy0 / step) * step; gy <= wy1; gy += step) {
        const major = Math.abs(Math.round(gy / step) % 5) === 0 || Math.abs(gy) < 1e-9;
        const [, sy] = this._toScreen(p, 0, gy);
        ctx.strokeStyle = major ? th.gridMajor : th.grid; ctx.beginPath(); ctx.moveTo(p.x, Math.round(sy) + 0.5); ctx.lineTo(p.x + p.w, Math.round(sy) + 0.5); ctx.stroke();
        if (LM !== 'none' && (major || step * c.scale > 34)) { ctx.fillStyle = th.ink3; ctx.textAlign = 'left'; ctx.fillText(`${+gy.toFixed(2)}`, p.x + 3, sy - 11); }
      }
    }

    // F: walkable fill with holes (even-odd), exactly as the core exported it
    if (L.has('free') && sp.free?.outer?.length >= 3) {
      ctx.beginPath();
      const add = (poly) => poly.forEach((pt, i) => { const [x, y] = this._toScreen(p, pt[0], pt[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      add(sp.free.outer); ctx.closePath();
      for (const h of sp.free.holes || []) { add(h); ctx.closePath(); }
      ctx.fillStyle = withAlpha(th.fFill, 0.16); ctx.fill('evenodd');
      ctx.strokeStyle = withAlpha(th.fFill, 0.85); ctx.lineWidth = 1.4; ctx.setLineDash([]); ctx.stroke();
    }

    const hl = this.highlight;
    const hlKeys = new Set();
    if (hl && hl.kind === 'element') {
      hlKeys.add(`${hl.space}:${hl.id}`);
      for (const m of this.scene.mates(hl.space, hl.id)) hlKeys.add(`${m.space}:${m.id}`);
    }

    // footprints by kind
    const scopeOn = L.has('scope');
    const list = [...sp.list].sort((a, z) => (KIND_RANK[a.kind] ?? 4) - (KIND_RANK[z.kind] ?? 4));
    if (L.has('footprints')) for (const e of list) {
      if (!e.poly) {
        if (e.centroid && !(e.kind === 'person' && L.has('actors'))) { const [x, y] = this._toScreen(p, e.centroid[0], e.centroid[1]); ctx.fillStyle = th.ink3; ctx.fillRect(x - 2, y - 2, 4, 4); }
        continue;
      }
      this._path(ctx, p, e.poly);
      const col = kindFill(th, e.kind);
      if (e.kind === 'border') { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash([]); ctx.stroke(); }
      else {
        ctx.fillStyle = withAlpha(col, e.kind === 'room' || e.kind === 'region' ? 0.35 : e.kind === 'interaction_region' ? 0.3 : 0.8);
        ctx.fill();
        ctx.strokeStyle = e.kind === 'interaction_region' ? col : th.ink3; ctx.lineWidth = e.kind === 'room' ? 1.6 : 1;
        ctx.setLineDash(e.kind === 'interaction_region' ? [6, 3] : []); ctx.stroke(); ctx.setLineDash([]);
      }
      if (e.flags?.includes('transfer_leaf')) { this._path(ctx, p, e.poly); ctx.fillStyle = this._hatchPattern(th); ctx.fill(); }
    }

    // scope overlay
    if (scopeOn) for (const e of list) {
      const mk = st.marks.get(e.id);
      if (!mk || !e.poly) continue;
      if (mk.dropped.length && !mk.inScope.length) {
        this._path(ctx, p, e.poly); ctx.fillStyle = withAlpha(th.panel, 0.62); ctx.fill();
        ctx.strokeStyle = th.ink3; ctx.lineWidth = 1; ctx.setLineDash([2, 3]); ctx.stroke(); ctx.setLineDash([]);
      }
      const fams = [...new Set(mk.inScope)];
      fams.forEach((f, i) => {
        const cls = f.toLowerCase().startsWith('spac') ? 'spacing' : f.toLowerCase().startsWith('fac') ? 'facing' : f.toLowerCase().startsWith('aff') ? 'afford' : 'other';
        this._path(ctx, p, e.poly);
        if (i === 0) { ctx.fillStyle = withAlpha(familyColor(th, cls), 0.18); ctx.fill(); }
        ctx.strokeStyle = familyColor(th, cls); ctx.lineWidth = 2.5 + 2.5 * (fams.length - 1 - i); ctx.setLineDash([]); ctx.stroke();
      });
      if (mk.container) { this._path(ctx, p, e.poly); ctx.strokeStyle = th.ink; ctx.lineWidth = 2.5; ctx.setLineDash(mk.containerBorder === 1 ? [] : [8, 4]); ctx.stroke(); ctx.setLineDash([]); }
      if (mk.parity?.length && L.has('parity')) { /* drawn below */ }
    }

    // parity scope sets (e.g. what v1 answers): second, dashed outline set, labelled with the channel id
    if (L.has('parity')) {
      for (const e of list) {
        const mk = st.marks.get(e.id);
        if (!mk?.parity?.length || !e.poly) continue;
        mk.parity.forEach((pid, i) => {
          this._path(ctx, p, e.poly); ctx.strokeStyle = th[extraKey(i + 2)]; ctx.lineWidth = 2; ctx.setLineDash([3, 3]); ctx.lineDashOffset = i * 3; ctx.stroke(); ctx.setLineDash([]); ctx.lineDashOffset = 0;
          { const [x, y] = this._toScreen(p, e.centroid[0], e.centroid[1]); lab('all', { text: pid, ax: x, ay: y, prio: 40, color: th[extraKey(i + 2)], inside: i === 0, r: 4 }); }
        });
      }
    }

    // correspondence highlight
    if (hlKeys.size) for (const e of sp.list) if (e.poly && hlKeys.has(`${p.space}:${e.id}`)) {
      this._path(ctx, p, e.poly); ctx.fillStyle = withAlpha(th.hl, 0.22); ctx.fill(); ctx.strokeStyle = th.hl; ctx.lineWidth = 3; ctx.setLineDash([]); ctx.stroke();
    }

    // one short name per element: kind + short id (full name on hover); never stacked
    if (L.has('footprints')) {
      for (const e of list) {
        if (!e.poly || e.kind === 'border') continue;
        if (Math.sqrt(e.area) * c.scale < 26 && LM !== 'all') continue;
        const [x, y] = this._toScreen(p, e.centroid[0], e.centroid[1]);
        lab('all', { text: `${e.kind} ${e.shared_id ?? e.id}`, ax: x, ay: y, prio: 30, color: th.ink2, font: `10px ${th.fontSans}`, inside: true, r: 4 });
      }
    }

    // trails
    const trails = eng ? [] : this.scene.trails.get(p.space) || [];
    for (const t of trails) {
      if (t.kind === 'extra' ? !L.has('extras') : !L.has('trail.' + t.kind)) continue;
      const col = t.kind === 'extra' ? th[extraKey(t.idx)] : t.kind === 'proposal' ? th.warn : t.kind === 'body' ? th.s2 : th.s1;
      for (const pass of [0, 1]) {
        ctx.beginPath(); let pen = false;
        const last = Math.floor(state.tick);
        t.pts.forEach((pt, k) => {
          if ((pass === 1 && k > last) || !pt) { pen = false; return; }
          const [x, y] = this._toScreen(p, pt[0], pt[1]);
          if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
        });
        ctx.strokeStyle = col; ctx.globalAlpha = pass ? 0.95 : 0.28; ctx.lineWidth = pass ? 2 : 1.2; ctx.setLineDash(t.kind === 'proposal' ? [4, 3] : []); ctx.stroke();
      }
      ctx.globalAlpha = 1; ctx.setLineDash([]);
    }

    // relations (weight = thickness)
    if (L.has('relations')) {
      let top = -1, tw = -Infinity;
      st.relations.forEach((r, i) => { if ((r.kind === 'contact' || r.kind === 'rx') && (r.weight ?? -1) > tw) { tw = r.weight ?? -1; top = i; } });
      this._top = top;
      let tf = -1, tfw = -Infinity;
      st.relations.forEach((r, i) => { if ((r.kind === 'look_at' || r.kind === 'rx') && r.fc === 'facing' && r.yaw != null && (r.weight ?? -1) > tfw) { tfw = r.weight ?? -1; tf = i; } });
      this._topFacing = tf;
      st.relations.forEach((r, ri) => this._drawRelation(ctx, th, p, r, ri, lab));
    }

    // extra poses: distinct hollow markers per channel (legend by id is in the stage), trails above
    if (L.has('extras')) for (const x of st.extras) if (x.pose) {
      const [sx, sy] = this._toScreen(p, x.pose.p[0], x.pose.p[1]);
      const col = th[extraKey(x.idx)], r = 7 + (x.idx % 3) * 2;
      if (/replay/.test(x.id)) {
        drawGlyph(ctx, sx, sy, yawOfQ(x.pose.q), glyphDims(null), c.scale, null, { fill: null, stroke: col, dash: [1.5, 3.5], lw: 2.2 });
        gblock(sx, sy, 14);
        lab('all', { text: x.label.replace(/^receiver\./, ''), ax: sx, ay: sy, prio: 55, color: col, r: 14 });
        continue;
      }
      ctx.beginPath();
      const sh = x.idx % 3;
      if (sh === 0) ctx.arc(sx, sy, r, 0, 7); else if (sh === 1) ctx.rect(sx - r, sy - r, 2 * r, 2 * r); else { ctx.moveTo(sx, sy - r - 1); ctx.lineTo(sx + r, sy + r); ctx.lineTo(sx - r, sy + r); ctx.closePath(); }
      ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.setLineDash([]); ctx.stroke();
      this._tick(ctx, p, x.pose, r * 1.8, col, 1.8);
      gblock(sx, sy, r + 3);
      lab('all', { text: x.label.replace(/^receiver\./, ''), ax: sx, ay: sy, prio: 55, color: col, r: r + 3 });
    }

    // proposals: dashed arrow body -> proposal, ghost glyph (dashed outline, no fill, no walk cycle)
    if (L.has('proposal')) st.proposals.forEach((pr) => {
      if (!pr.pose) return;
      const col = statusColor(th, pr.status);
      const [px, py] = this._toScreen(p, pr.pose.p[0], pr.pose.p[1]);
      if (pr.bodyPose) {
        const [bx, by] = this._toScreen(p, pr.bodyPose.p[0], pr.bodyPose.p[1]);
        const d = Math.hypot(px - bx, py - by);
        if (d > 6) {
          ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(px, py); ctx.strokeStyle = th.ink; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]);
          const ang = Math.atan2(py - by, px - bx);
          ctx.beginPath(); ctx.moveTo(px - 6 * Math.cos(ang - 0.45), py - 6 * Math.sin(ang - 0.45)); ctx.lineTo(px, py); ctx.lineTo(px - 6 * Math.cos(ang + 0.45), py - 6 * Math.sin(ang + 0.45)); ctx.stroke();
        }
      }
      const dims = glyphDims(pr.body);
      drawGlyph(ctx, px, py, yawOfQ(pr.pose.q), dims, c.scale, null, { fill: null, stroke: col, dash: [4, 3], lw: 2 });
      { const gr = Math.max(glyphScale(dims, c.scale) * dims.W * 0.6, 12); gblock(px, py, gr); lab('all', { text: `proposal ${pr.status ?? ''}`, ax: px, ay: py, prio: 60, color: col, r: gr }); }
    });

    // footprint marks (derived stride marks, default off)
    if (L.has('steps')) for (const a of st.actors) {
      const wt = this.scene.walk.get(a.id);
      if (!wt) continue;
      const dims = glyphDims(a.body), col = withAlpha(a.role === 'receiver_body' ? th.s2 : th.person, 0.5);
      for (const s of wt.steps) {
        if (s.k > state.k) break;
        const [x, y] = this._toScreen(p, s.x, s.y);
        drawFoot(ctx, x, y, s.yaw, dims, c.scale, s.side, col);
      }
    }

    // people: other persons neutral solid; actors: source solid, receiver body solid in its colour
    if (L.has('actors')) {
      for (const pe of st.people) {
        const [x, y] = this._toScreen(p, pe.pos[0], pe.pos[1]);
        drawGlyph(ctx, x, y, pe.yaw, glyphDims(null), c.scale, null, { fill: withAlpha(th.ink3, 0.3), headFill: th.ink3, footFill: th.ink3, stroke: th.ink3 });
        { const gr = Math.max(glyphScale(glyphDims(null), c.scale) * 0.3, 12); gblock(x, y, gr); lab('all', { text: pe.name || 'person', ax: x, ay: y, prio: 45, color: th.ink2, r: gr }); }
      }
      for (const a of st.actors) {
        if (!a.pose) continue;
        const [x, y] = this._toScreen(p, a.pose.p[0], a.pose.p[1]);
        const dims = glyphDims(a.body);
        const col = a.role === 'receiver_body' ? th.s2 : a.role === 'source_actor' ? th.person : th.ink3;
        if (L.has('gaze') && a.role !== 'other' && !a.joints) { const [hx, hy] = headPos(x, y, a.yaw, dims, c.scale); drawWedge(ctx, hx, hy, a.yaw, c.scale, col); }
        if (a.joints && a.skeleton) { ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fillStyle = col; ctx.fill(); }
        else this._glyphs++; drawGlyph(ctx, x, y, a.yaw, dims, c.scale, a.walk, { fill: withAlpha(col, 0.3), headFill: col, footFill: col, stroke: col, lw: 1.8 });
        { const gr = Math.max(glyphScale(dims, c.scale) * dims.W * 0.6, 12); gblock(x, y, gr); lab('key', { text: a.id, ax: x, ay: y, prio: 90, color: th.ink, r: gr }); }
        if (hl && hl.kind === 'actor' && hl.space === p.space && hl.id === a.id) { ctx.beginPath(); ctx.arc(x, y, 16, 0, 7); ctx.strokeStyle = th.hl; ctx.lineWidth = 3; ctx.stroke(); }
      }
    }
    if (L.has('skeleton')) for (const a of st.actors) {
      if (!a.joints || !a.skeleton) continue;
      const par = a.skeleton.parents || [];
      ctx.strokeStyle = th.cFacing; ctx.lineWidth = 2; ctx.setLineDash([]);
      a.joints.forEach((j, i) => {
        if (!j) return;
        const [x, y] = this._toScreen(p, j.p[0], j.p[1]);
        const pj = par[i] >= 0 ? a.joints[par[i]] : null;
        if (pj) { const [px, py] = this._toScreen(p, pj.p[0], pj.p[1]); ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(x, y); ctx.stroke(); }
        ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fillStyle = th.cFacing; ctx.fill();
      });
    }

    if (cmp && p.engine) this._drawCompare(ctx, th, p, c, state, st, lab, gblock, eng);
    this._labels.push(...lp.resolve(ctx, th).map((r) => ({ ...r, space: p.space })));

    // pane chrome: border, header (space id, role, scope mode badge), scale bar
    ctx.restore();
    ctx.save();
    ctx.strokeStyle = th.line2; ctx.lineWidth = 1; ctx.strokeRect(p.x + 0.5, p.y + 0.5, p.w - 1, p.h - 1);
    ctx.font = `600 11px ${th.fontMono}`; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
    const title = cmp && p.engine ? `${sp.id} · ${p.engine}` : `${sp.id} · ${sp.role}`;
    const tw = ctx.measureText(title).width;
    const ecol = cmp && p.engine ? engineColor(th, p.engine, this.scene.engines.findIndex((e) => e.name === p.engine)) : null;
    ctx.fillStyle = ecol || withAlpha(th.panel2, 0.92); ctx.fillRect(p.x + 6, p.y + 6, tw + 12, 18);
    ctx.fillStyle = ecol ? th.panel : th.ink; ctx.fillText(title, p.x + 12, p.y + 10);
    if (cmp && p.engine) {
      const ee = this.scene.engineErr(p.engine, state.k);
      const txt = ee ? `|R_src - R_rx| mean ${ee.mean.toFixed(3)} over ${ee.per.length} (${ee.origin})` : 'no per-relation data for this engine';
      ctx.font = `10px ${th.fontMono}`; const ew = Math.min(ctx.measureText(txt).width, p.w - 20);
      ctx.fillStyle = withAlpha(th.panel2, 0.92); ctx.fillRect(p.x + 6, p.y + 26, ew + 10, 15); ctx.fillStyle = ee ? th.ink : th.ink3; ctx.fillText(txt, p.x + 11, p.y + 29);
      ctx.font = `600 11px ${th.fontMono}`;
    }
    if (st.scope?.mode) {
      const mode = `mode ${st.scope.mode}${st.scope.transfer ? ' · transfer' : ''}`;
      const mw = ctx.measureText(mode).width;
      ctx.fillStyle = th.ink; ctx.fillRect(p.x + 12 + tw + 12, p.y + 6, mw + 12, 18);
      ctx.fillStyle = th.panel; ctx.fillText(mode, p.x + 18 + tw + 12, p.y + 10);
    }
    // scale bar (same metric scale in every pane unless unlocked)
    const bar = niceStep(70 / c.scale), bpx = bar * c.scale;
    const bx = p.x + 12, by = p.y + p.h - 26;
    ctx.strokeStyle = th.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(bx, by - 4); ctx.lineTo(bx, by); ctx.lineTo(bx + bpx, by); ctx.lineTo(bx + bpx, by - 4); ctx.stroke();
    ctx.font = `10px ${th.fontMono}`; ctx.fillStyle = th.ink; ctx.textBaseline = 'bottom'; ctx.fillText(`${bar} m`, bx + bpx + 6, by + 3);
    ctx.restore();
  }

  _isEngineCh(id) { return this.scene.engines.some((e) => e.ch.id === id) || (this.scene.v2Ch && this.scene.v2Ch.id === id && !this.scene.actors.some((a) => a.root_channel === id)); }

  /** compare panes: the v2 pane gets its body glyph (eval cases have no receiver actor) and the optional v1 overlay;
   * an engine pane gets only what its data has: body glyph, trail, per-referent lines and R. No invented proposals/scope. */
  _drawCompare(ctx, th, p, c, state, st, lab, gblock, eng) {
    const sc = this.scene, k = state.k, L = this.layers;
    const idx = (n) => sc.engines.findIndex((e) => e.name === n);
    const glyphOf = (bs, col, dash, name) => {
      if (!bs?.pose) return null;
      const [x, y] = this._toScreen(p, bs.pose.p[0], bs.pose.p[1]);
      const dims = glyphDims(null);
      this._glyphs++; drawGlyph(ctx, x, y, bs.yaw, dims, c.scale, bs.walk, { fill: dash ? null : withAlpha(col, 0.3), headFill: dash ? null : col, footFill: dash ? null : col, stroke: col, dash: dash ? [4, 3] : null, lw: 1.8 });
      const gr = Math.max(glyphScale(dims, c.scale) * dims.W * 0.6, 12); gblock(x, y, gr);
      lab('key', { text: name, ax: x, ay: y, prio: 88, color: col, r: gr });
      return [x, y];
    };
    if (!eng) {
      const hasActor = st.actors.some((a) => a.role === 'receiver_body');
      if (!hasActor && L.has('actors')) glyphOf(state.v2, engineColor(th, 'v2'), false, 'v2');
      if (this.compare.overlay && state.v2?.pose) {
        for (const e of state.engines) {
          if (!e.pose) continue;
          const col = engineColor(th, e.name, idx(e.name));
          const pt = glyphOf(e, col, true, e.name);
          const [vx, vy] = this._toScreen(p, state.v2.pose.p[0], state.v2.pose.p[1]);
          if (pt) {
            ctx.beginPath(); ctx.moveTo(vx, vy); ctx.lineTo(pt[0], pt[1]); ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]);
            const d = Math.hypot(e.pose.p[0] - state.v2.pose.p[0], e.pose.p[1] - state.v2.pose.p[1]);
            lab('key', { text: `v2-${e.name} ${d.toFixed(2)} m`, ax: (vx + pt[0]) / 2, ay: (vy + pt[1]) / 2, prio: 76, color: col, cands: (w, h) => LabelPlacer.lineCands(vx, vy, pt[0], pt[1], w, h), r: 2 });
          }
        }
      }
      return;
    }
    const E = sc.engines.find((x) => x.name === eng), es = state.engines.find((x) => x.name === eng);
    const col = engineColor(th, eng, idx(eng));
    if (L.has('trail.body')) for (const pass of [0, 1]) {
      ctx.beginPath(); let pen = false;
      E.pts.forEach((pt, i) => { if ((pass === 1 && i > k) || !pt) { pen = false; return; } const [x, y] = this._toScreen(p, pt[0], pt[1]); if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; } });
      ctx.strokeStyle = col; ctx.globalAlpha = pass ? 0.95 : 0.28; ctx.lineWidth = pass ? 2 : 1.2; ctx.setLineDash([]); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // referent lines: receiver footprints sharing the source referents' shared ids
    const refs = new Set(sc.evalRefs.map((r) => (Number.isFinite(+r) ? +r : r)));
    for (const r of state.spaces[sc.roleSpace('source')]?.relations || []) if (r.referent !== null && r.referent !== undefined) refs.add(r.referent);
    const body = es?.pose ? this._toScreen(p, es.pose.p[0], es.pose.p[1]) : null;
    for (const ref of refs) for (const e of sc.resolveReferent(p.space, ref)) {
      if (!e.poly) continue;
      this._path(ctx, p, e.poly); ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);
      if (!body) continue;
      const [cx, cy] = this._toScreen(p, e.centroid[0], e.centroid[1]);
      ctx.beginPath(); ctx.moveTo(body[0], body[1]); ctx.lineTo(cx, cy); ctx.strokeStyle = withAlpha(col, 0.55); ctx.lineWidth = 1; ctx.stroke();
      const R = sc.engineR(eng, ref, k);
      if (R !== null) lab('key', { text: `R ${R.toFixed(2)}`, ax: cx, ay: cy, prio: 70, color: col, inside: true, r: 4 });
    }
    glyphOf(es, col, false, eng);
  }

  _tick(ctx, p, pose, len, col, w = 2) {
    const f = quatForward(pose.q);
    const [x, y] = this._toScreen(p, pose.p[0], pose.p[1]);
    const m = Math.hypot(f[0], f[1]);
    if (m < 1e-6) return;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + (f[0] / m) * len, y - (f[1] / m) * len);
    ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash([]); ctx.stroke();
  }

  _drawRelation(ctx, th, p, r, ri, lab) {
    if (r.kind === 'none') return;
    const col = familyColor(th, r.fc);
    const w = isFinite(r.weight) && r.weight !== null ? 1 + 5 * clamp(r.weight, 0, 1) : 1.2;
    let [ax, ay] = this._toScreen(p, r.a[0], r.a[1]);
    const [bx, by] = this._toScreen(p, r.b[0], r.b[1]);
    const row = r.row, pxm = this._pc(p).scale;
    const gazeRay = (r.kind === 'look_at' || r.kind === 'rx') && r.fc === 'facing' && r.yaw !== null && r.yaw !== undefined;
    const rayFrom = gazeRay ? headPos(ax, ay, r.yaw, glyphDims(r.body), pxm) : null; // the ray starts at the head
    if (rayFrom) [ax, ay] = rayFrom;
    ctx.save();
    ctx.lineCap = 'round';
    ctx.strokeStyle = col; ctx.lineWidth = w;
    ctx.globalAlpha = r.weight === 0 ? 0.35 : 0.95;
    ctx.setLineDash(r.kind === 'look_at' && row.occluded === 1 ? [6, 4] : r.kind === 'rx' ? [3, 3] : []);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    let label = '';
    if (r.kind === 'contact') {
      // contact marker (Spacing: dot on the support; AffordancePose: diamond with a cross)
      if (r.fc === 'afford') { ctx.beginPath(); ctx.moveTo(bx, by - 7); ctx.lineTo(bx + 7, by); ctx.lineTo(bx, by + 7); ctx.lineTo(bx - 7, by); ctx.closePath(); ctx.fillStyle = withAlpha(col, 0.3); ctx.fill(); ctx.lineWidth = 2; ctx.stroke(); ctx.beginPath(); ctx.moveTo(bx - 3, by - 3); ctx.lineTo(bx + 3, by + 3); ctx.moveTo(bx + 3, by - 3); ctx.lineTo(bx - 3, by + 3); ctx.lineWidth = 1.5; ctx.stroke(); }
      else { ctx.beginPath(); ctx.arc(bx, by, 4.5, 0, 7); ctx.fillStyle = col; ctx.fill(); }
      label = [row.d_m != null ? `d ${fmt(row.d_m, 2)} m` : null, row.R_src != null ? `R ${fmt(row.R_src, 2)}` : null].filter(Boolean).join(' · ');
    } else if (r.kind === 'look_at') {
      const ang = Math.atan2(by - ay, bx - ax);
      ctx.beginPath(); ctx.moveTo(bx - 9 * Math.cos(ang - 0.4), by - 9 * Math.sin(ang - 0.4)); ctx.lineTo(bx, by); ctx.lineTo(bx - 9 * Math.cos(ang + 0.4), by - 9 * Math.sin(ang + 0.4)); ctx.lineWidth = 2; ctx.stroke();
      if (row.look_at_vertex !== null && row.look_at_vertex !== undefined && row.look_at_vertex >= 0) {
        ctx.beginPath(); ctx.rect(bx - 5, by - 5, 10, 10); ctx.fillStyle = th.panel; ctx.fill(); ctx.lineWidth = 2; ctx.stroke();
        label = `vertex ${row.look_at_vertex}`;
      } else label = 'hit point';
      if (row.angle_err_rad != null) label += ` · Δψ ${fmt((row.angle_err_rad * 180) / Math.PI, 1)}°`;
      if (row.occluded === 1) label += ' · occluded'; else if (row.occluded === 0) label += ' · visible';
    } else if (r.kind === 'rx') {
      ctx.beginPath(); ctx.arc(bx, by, 4.5, 0, 7); ctx.fillStyle = th.panel; ctx.fill(); ctx.lineWidth = 2; ctx.stroke();
      label = [row.R_rx != null ? `R_rx ${fmt(row.R_rx, 2)}` : null, row.err_rx != null ? `err (R_rx - R_src, signed) ${fmt(row.err_rx, 2)}` : null].filter(Boolean).join(' · ');
    }
    if (gazeRay && this.layers.has('gaze')) {
      // angle between where the body faces (wedge axis) and what the Facing relation targets (ray); derived for display
      const tgt = Math.atan2(ay - by, bx - ax), dth = Math.atan2(Math.sin(tgt - r.yaw), Math.cos(tgt - r.yaw));
      const R = Math.max(0.6 * pxm, 26);
      ctx.beginPath(); ctx.arc(ax, ay, R, -r.yaw, -(r.yaw + dth), dth > 0); ctx.lineWidth = 1.8; ctx.setLineDash([]); ctx.stroke();
      const mid = -(r.yaw + dth / 2);
      const tx = ax + Math.cos(mid) * (R + 12), ty = ay + Math.sin(mid) * (R + 12);
      lab(this._topFacing === ri ? 'key' : 'all', { text: `${(Math.abs(dth) * 180 / Math.PI).toFixed(1)}\u00b0`, ax: tx, ay: ty, prio: 85, color: col, font: `600 10px ${th.fontMono}`, r: 4 });
    }
    if (label && lab && (this._top === ri || (r.kind === 'look_at' && false))) lab('key', { text: label, ax: (ax + bx) / 2, ay: (ay + by) / 2, prio: 80, color: col, cands: (w, h) => LabelPlacer.lineCands(ax, ay, bx, by, w, h), r: 2 });
    else if (label && lab) lab('all', { text: label, ax: (ax + bx) / 2, ay: (ay + by) / 2, prio: 50 + 10 * clamp(r.weight ?? 0, 0, 1), color: col, cands: (w, h) => LabelPlacer.lineCands(ax, ay, bx, by, w, h), r: 2 });
    ctx.restore();
  }
}
