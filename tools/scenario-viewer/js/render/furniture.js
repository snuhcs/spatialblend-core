// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Procedural furniture meshes for the 3D view. DISPLAY-DERIVED: one mesh per element, chosen by element name / tag /
// part names, built from the exported footprint (`ground`), `shape3d.extrude_height_m` and `element.parts` (schema 1.1).
// Anything a case exports wins (footprint, height, part planes). Everything else (leg style, back height, the default
// heights of a type) is a display choice, authored in DISPLAY below, and the UI labels these meshes "display-derived".
// z is up. Textures are in metres (see textures.js), so nothing is stretched.

export const DISPLAY = {
  // heights used ONLY when the case exports neither shape3d nor a part plane (authored display values)
  seatH: { chair: 0.45, armchair: 0.42, stool: 0.65, bench: 0.45, sofa: 0.42, officechair: 0.5 },
  topH: { table: 0.75, desk: 0.74, counter: 0.9, shelf: 1.8 },
  board: { bottom: 0.9, height: 1.0, easelBottom: 0.8, flipBottom: 0.7 },
  backH: 0.45, slab: 0.04, leg: 0.04, wallH: 2.6, curbH: 0.35,
};

// name -> type, most specific first. `tags` and part names are used when the name says nothing.
const NAME_RULES = [
  [/office[\s_-]*chair|swivel|task[\s_-]*chair/, 'officechair'], [/arm[\s_-]*chair|lounge[\s_-]*chair|recliner/, 'armchair'],
  [/sofa|couch|loveseat/, 'sofa'], [/stool/, 'stool'], [/bench/, 'bench'], [/chair|seat/, 'chair'],
  [/white[\s_-]*board|whiteboard/, 'whiteboard'], [/black[\s_-]*board|chalk[\s_-]*board/, 'blackboard'],
  [/easel/, 'easel'], [/flip[\s_-]*chart|flipchart/, 'flipchart'],
  [/book[\s_-]*case|shelf|shelves|rack/, 'shelf'], [/counter|bar\b/, 'counter'], [/desk/, 'desk'], [/table/, 'table'],
  [/panel|button|switch/, 'panel'],
];
// Part TAG vocabulary (adopted by the scene exporter): seat, back, arm, writing_surface, keyboard_area, edge, shelf_surface,
// button_panel (the first exports also use the names board, desk_surface, top, shelf_top, button). The tags of an element's parts choose the
// family of builder FIRST; the element name only picks the member of that family; with no usable part tag the name decides, as before.
const TAG_FAMILY = { seat: 'seat', back: 'seat', arm: 'seat', writing_surface: 'board', board: 'board', keyboard_area: 'desk', desk_surface: 'desk',
  edge: 'top', top: 'top', shelf_surface: 'shelf', shelf_top: 'shelf', button_panel: 'panel', button: 'panel' };
const FAMILY_MEMBERS = {
  seat: [[/office[\s_-]*chair|swivel|task[\s_-]*chair/, 'officechair'], [/arm[\s_-]*chair|lounge|recliner/, 'armchair'], [/sofa|couch|loveseat/, 'sofa'], [/stool/, 'stool'], [/bench/, 'bench']], seatDefault: 'chair',
  board: [[/black[\s_-]*board|chalk/, 'blackboard'], [/easel/, 'easel'], [/flip[\s_-]*chart|flipchart/, 'flipchart']], boardDefault: 'whiteboard',
  desk: [[/desk/, 'desk']], deskDefault: 'table',
  top: [[/counter|bar\b/, 'counter'], [/desk/, 'desk']], topDefault: 'table',
  shelf: [], shelfDefault: 'shelf', panel: [], panelDefault: 'panel',
};
const partTagOf = (p) => String(p.tag || '').toLowerCase();

export function classify(e) {
  const name = String(e.name || '').toLowerCase();
  // 1. part tags (then part names) -> family -> member by name
  const fams = [];
  for (const p of e.parts || []) for (const k of [partTagOf(p), String(p.name || '').toLowerCase()]) if (TAG_FAMILY[k]) fams.push(TAG_FAMILY[k]);
  const fam = ['board', 'shelf', 'panel', 'desk', 'top', 'seat'].find((f) => fams.includes(f)) ?? null;   // an element with both seat and desk parts is a desk
  if (fam) {
    const hit = (FAMILY_MEMBERS[fam] || []).find(([re]) => re.test(name));
    return { type: hit ? hit[1] : FAMILY_MEMBERS[fam + 'Default'], by: 'part tag' };
  }
  // 2. the element name, then its tags / flags
  for (const [re, t] of NAME_RULES) if (re.test(name)) return { type: t, by: 'name' };
  const tags = [...(e.tags || []), ...(e.flags || [])].join(' ').toLowerCase();
  for (const [re, t] of NAME_RULES) if (re.test(tags)) return { type: t, by: 'tag' };
  return { type: null, by: null };
}

/** parts whose tag or name matches (tag first) */
export const partsMatching = (e, re) => {
  const ps = e.parts || [];
  return [...ps.filter((p) => re.test(partTagOf(p))), ...ps.filter((p) => !re.test(partTagOf(p)) && re.test(String(p.name || '').toLowerCase()))];
};

const TAU = Math.PI * 2;
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

/** minimum-area oriented box of a polygon: {cx, cy, ang, w, d} (w along the axis at angle `ang`, d across). */
export function obb(poly) {
  let best = null;
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const a = poly[i], b = poly[(i + 1) % n];
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    if (!Number.isFinite(ang) || (a[0] === b[0] && a[1] === b[1])) continue;
    const c = Math.cos(ang), s = Math.sin(ang);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of poly) { const x = p[0] * c + p[1] * s, y = -p[0] * s + p[1] * c; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const area = (x1 - x0) * (y1 - y0);
    if (!best || area < best.area - 1e-12) {
      const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      best = { area, ang, w: x1 - x0, d: y1 - y0, cx: mx * c - my * s, cy: mx * s + my * c };
    }
  }
  return best || { area: 0, ang: 0, w: 0.5, d: 0.5, cx: poly[0]?.[0] ?? 0, cy: poly[0]?.[1] ?? 0 };
}

/** Which exported thing decides which way a seat / counter / shelf faces. Priority: the `back` part (face away from it), the exported
 * `seat` part frame (+x), a non-identity element rotation (+x), else toward the room centre (a display guess, said so in the inspector). */
export function facingChoice(e, ob, roomC) {
  const partsBy = (re) => partsMatching(e, re);
  const back = partsBy(/^back|backrest/)[0];
  if (back) return { hint: Math.atan2(-(back.origin3[1] - ob.cy), -(back.origin3[0] - ob.cx)), by: 'back part' };
  const seat = partsBy(/^seat/)[0];
  if (seat) return { hint: Math.atan2(seat.m[4], seat.m[0]), by: 'seat part +x' };
  const m = e.world_transform;
  if (Array.isArray(m) && (Math.abs(m[1]) > 1e-6 || Math.abs(m[4]) > 1e-6)) return { hint: Math.atan2(m[4], m[0]), by: 'element rotation' };
  return { hint: Math.atan2(roomC[1] - ob.cy, roomC[0] - ob.cx), by: 'toward room centre (display guess)' };
}

/** What the inspector says about one element's procedural mesh (pure; no three.js). */
export function describeFurniture(e, roomC) {
  if (!e.poly || e.poly.length < 3 || ['room', 'border', 'region', 'interaction_region', 'surface'].includes(e.kind)) return null;
  const { type } = classify(e);
  if (!type) return { type: null, text: `extruded footprint${e.shape3d?.extrude_height_m > 0 ? ' (shape3d height)' : ' (flat: no height exported)'}; no facing` };
  if (['whiteboard', 'blackboard', 'easel', 'flipchart'].includes(type)) {
    const part = partsMatching(e, /board|write|writing_surface|draw|surface|paper|top/).find((p) => Math.abs(p.normal[2]) < 0.5) || (e.parts || []).find((p) => Math.abs(p.normal[2]) < 0.5);
    return { type, text: `procedural furniture (display-derived: ${type}); facing: ${part ? `exported part "${part.name}" (its plane and +z normal)` : 'no vertical part exported: authored size, toward room centre (display guess)'}` };
  }
  const ob = obb(e.poly), f = facingChoice(e, ob, roomC);
  return { type, text: `procedural furniture (display-derived: ${type}); facing: chosen by ${f.by}` };
}

/** Geometry caches shared by every case (box / cylinder primitives). `shared` geometries are never disposed on a case switch. */
export class GeoCache {
  constructor(THREE) { this.T = THREE; this.map = new Map(); }
  get size() { return this.map.size; }
  box(w, d, h) {
    const k = `b${w.toFixed(3)}|${d.toFixed(3)}|${h.toFixed(3)}`;
    let g = this.map.get(k);
    if (!g) {
      g = new this.T.BoxGeometry(w, d, h);
      const uv = g.attributes.uv, sc = [[d, h], [d, h], [w, h], [w, h], [w, d], [w, d]]; // px nx py ny pz nz
      for (let f = 0; f < 6; f++) for (let v = 0; v < 4; v++) { const i = f * 4 + v; uv.setXY(i, uv.getX(i) * sc[f][0], uv.getY(i) * sc[f][1]); }
      g.userData.shared = true; this.map.set(k, g);
    }
    return g;
  }
  cyl(r0, r1, h, seg = 24) {
    const k = `c${r0.toFixed(3)}|${r1.toFixed(3)}|${h.toFixed(3)}|${seg}`;
    let g = this.map.get(k);
    if (!g) { g = new this.T.CylinderGeometry(r0, r1, h, seg, 1); g.rotateX(Math.PI / 2); g.userData.shared = true; this.map.set(k, g); }
    return g;
  }
  dispose() { for (const g of this.map.values()) g.dispose(); this.map.clear(); }
}

class B {
  constructor(ctx, group) { this.ctx = ctx; this.T = ctx.THREE; this.g = group; this.M = ctx.tex.M; }
  mesh(geo, mat, x, y, z, shadow = true) {
    const m = new this.T.Mesh(geo, mat); m.position.set(x, y, z); m.castShadow = shadow; m.receiveShadow = true; this.g.add(m); return m;
  }
  box(mat, w, d, h, x, y, z0) { return this.mesh(this.ctx.geo.box(Math.max(w, 1e-3), Math.max(d, 1e-3), Math.max(h, 1e-3)), mat, x, y, z0 + h / 2); }
  cyl(mat, r, h, x, y, z0, r1 = r, seg = 24) { return this.mesh(this.ctx.geo.cyl(r, r1, Math.max(h, 1e-3), seg), mat, x, y, z0 + h / 2); }
  /** a thin cylinder between two points of this group's frame */
  strut(mat, a, b, r) {
    const T = this.T, A = new T.Vector3(...a), Bv = new T.Vector3(...b), d = Bv.clone().sub(A), L = d.length();
    const g = this.ctx.geoY(r, L); const m = new T.Mesh(g, mat);
    m.position.copy(A).add(Bv).multiplyScalar(0.5); m.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), d.normalize());
    m.castShadow = m.receiveShadow = true; this.g.add(m); return m;
  }
}

function legs(b, mat, w, d, h, inset, t = DISPLAY.leg) {
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) b.box(mat, t, t, h, sx * (w / 2 - inset), sy * (d / 2 - inset), 0);
}

// ---------------------------------------------------------------------------------------------------- builders (frame: +x forward, +y left, origin = footprint centre on the floor)
const BUILD = {
  chair(b, o) {
    const { w, d, Hs } = o, M = b.M, wood = M.wood(), th = DISPLAY.slab;
    b.box(wood, w, d, th, 0, 0, Hs - th);
    legs(b, wood, w, d, Hs - th, 0.03);
    b.box(wood, th, d * 0.94, DISPLAY.backH, -w / 2 + th / 2, 0, Hs);
    for (const sy of [-1, 1]) b.box(wood, 0.04, 0.04, DISPLAY.backH + Hs - th, -w / 2 + 0.03, sy * (d / 2 - 0.03), 0);
  },
  armchair(b, o) {
    const { w, d, Hs } = o, M = b.M, fab = M.fabric(0x8a5a44), wood = M.wood(0xcfae85);
    const arm = Math.min(0.12, d * 0.2), baseH = Hs - 0.12;
    b.box(wood, w, d, baseH, 0, 0, 0.06);
    b.box(fab, w - 0.1, d - 2 * arm, 0.12, 0.02, 0, baseH + 0.06 - 0.0);
    for (const sy of [-1, 1]) b.box(fab, w, arm, Hs + 0.2 - 0.06, 0, sy * (d / 2 - arm / 2), 0.06);
    b.box(fab, 0.14, d, DISPLAY.backH + Hs - 0.06, -w / 2 + 0.07, 0, 0.06);
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) b.box(wood, 0.04, 0.04, 0.06, sx * (w / 2 - 0.04), sy * (d / 2 - 0.04), 0);
  },
  stool(b, o) {
    const { w, d, Hs, round } = o, M = b.M, wood = M.wood(), th = DISPLAY.slab;
    if (round) {
      const r = Math.min(w, d) / 2;
      b.cyl(wood, r, th, 0, 0, Hs - th);
      for (let i = 0; i < 3; i++) { const a = (i * TAU) / 3 + 0.5; b.strut(M.metal(), [Math.cos(a) * r * 0.6, Math.sin(a) * r * 0.6, Hs - th], [Math.cos(a) * r * 0.95, Math.sin(a) * r * 0.95, 0.0], 0.014); }
    } else { b.box(wood, w, d, th, 0, 0, Hs - th); legs(b, wood, w, d, Hs - th, 0.03); }
  },
  bench(b, o) {
    const { w, d, Hs } = o, M = b.M, wood = M.wood(), th = 0.05;
    b.box(wood, w, d, th, 0, 0, Hs - th);
    for (const sy of [-1, 1]) b.box(wood, w * 0.9, th, Hs - th, 0, sy * (d / 2 - 0.1 - th / 2), 0);
  },
  sofa(b, o) {
    const { w, d, Hs } = o, M = b.M, fab = M.fabric(0x4f6a7a), wood = M.wood(0xb69870);
    const arm = Math.min(0.16, d * 0.18), baseH = Hs * 0.55;
    b.box(fab, w, d, baseH, 0, 0, 0.06);
    const n = Math.max(1, Math.round((d - 2 * arm) / 0.75)), cw = (d - 2 * arm) / n;
    for (let i = 0; i < n; i++) b.box(fab, w - 0.2, cw - 0.02, Hs - baseH - 0.06, 0.08, -(d - 2 * arm) / 2 + cw * (i + 0.5), baseH + 0.06);
    for (const sy of [-1, 1]) b.box(fab, w, arm, Hs + 0.22 - 0.06, 0, sy * (d / 2 - arm / 2), 0.06);
    b.box(fab, 0.2, d - 2 * arm, DISPLAY.backH + Hs - 0.06, -w / 2 + 0.1, 0, 0.06);
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) b.box(wood, 0.05, 0.05, 0.06, sx * (w / 2 - 0.08), sy * (d / 2 - 0.08), 0);
  },
  officechair(b, o) {
    const { w, d, Hs } = o, M = b.M, pl = M.plastic(0x2b3138), mt = M.metal(), r = Math.min(w, d) / 2 * 0.95, th = 0.08;
    b.cyl(pl, r, th, 0, 0, Hs - th);
    b.box(pl, 0.06, r * 1.7, 0.45, -r + 0.05, 0, Hs + 0.05);
    b.cyl(mt, 0.025, Hs - th - 0.05, 0, 0, 0.05);
    for (let i = 0; i < 5; i++) { const a = (i * TAU) / 5; const x = Math.cos(a) * r * 0.85, y = Math.sin(a) * r * 0.85; b.strut(mt, [0, 0, 0.07], [x, y, 0.05], 0.014); b.cyl(pl, 0.025, 0.04, x, y, 0.0); }
  },
  table(b, o) {
    const { w, d, Hs, round } = o, M = b.M, wood = M.wood(0xdcc09a), th = DISPLAY.slab;
    if (round) { const r = Math.min(w, d) / 2; b.cyl(wood, r, th, 0, 0, Hs - th); b.cyl(M.metal(), 0.05, Hs - th, 0, 0, 0); b.cyl(M.metal(), r * 0.45, 0.02, 0, 0, 0); }
    else { b.box(wood, w, d, th, 0, 0, Hs - th); legs(b, wood, w, d, Hs - th, 0.05, 0.06); }
  },
  counter(b, o) {
    const { w, d, Hs } = o, M = b.M;
    b.box(M.wood(0x9a7b57), w - 0.04, d - 0.06, Hs - 0.04, 0, 0, 0);
    b.box(M.wood(0xe0cfae), w, d, 0.04, 0, 0, Hs - 0.04);
  },
  shelf(b, o) {
    const { w, d, Hs } = o, M = b.M, wood = M.wood(0xb89a72), t = 0.03;
    for (const sy of [-1, 1]) b.box(wood, w, t, Hs, 0, sy * (d / 2 - t / 2), 0);
    b.box(wood, 0.015, d, Hs, -w / 2 + 0.0075, 0, 0);
    const n = Math.max(2, Math.round(Hs / 0.38));
    for (let i = 0; i <= n; i++) b.box(wood, w, d, t, 0, 0, Math.min(Hs - t, (Hs - t) * (i / n)));
  },
  panel(b, o) {
    const { w, d, Hs } = o, M = b.M;
    b.box(M.plastic(0xd9dde2), w, d, Math.max(0.02, Hs), 0, 0, 0);
    b.cyl(M.plastic(0xc0392b), Math.min(w, d) * 0.28, 0.01, 0, 0, Math.max(0.02, Hs));
  },
};

// vertical boards: built in the PART frame (plane z = 0, normal +z), so they sit exactly on the exported part plane
function boardInPartFrame(ctx, grp, poly, M4, kind, opts = {}) {
  const T = ctx.THREE, M = ctx.tex.M;
  const g = new T.Group(); g.matrixAutoUpdate = false; g.matrix.copy(M4); g.matrixWorldNeedsUpdate = true; grp.add(g);
  const b = new B(ctx, g);
  const shape = new T.Shape(poly.map(([x, y]) => new T.Vector2(x, y)));
  const geo = new T.ExtrudeGeometry(shape, { depth: 0.02, bevelEnabled: false }); geo.translate(0, 0, -0.02);
  const surf = kind === 'blackboard' ? M.blackboard() : M.whiteboard();
  const m = new T.Mesh(geo, surf); m.castShadow = true; m.receiveShadow = true; g.add(m);
  const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), fr = M.metal(), t = 0.03;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], c = poly[(i + 1) % poly.length], L = Math.hypot(c[0] - a[0], c[1] - a[1]);
    if (L < 1e-4) continue;
    const bar = new T.Mesh(ctx.geo.box(L, t, t), fr); bar.position.set((a[0] + c[0]) / 2, (a[1] + c[1]) / 2, -0.01); bar.rotation.z = Math.atan2(c[1] - a[1], c[0] - a[0]); bar.castShadow = true; g.add(bar);
  }
  const tray = new T.Mesh(ctx.geo.box((x1 - x0) * 0.6, 0.07, 0.025), fr); tray.position.set((x0 + x1) / 2, y0 + 0.02, 0.035); tray.castShadow = true; g.add(tray); // pen tray
  return { g, b, x0, x1, y0, y1 };
}

function partMatrix(T, m) { return new T.Matrix4().set(...m); }

/** Build the mesh group of one element. Returns {group, type, by, note, dims} or null when the element is only a decal/room. */
export function buildFurniture(ctx, e, sp) {
  const T = ctx.THREE, M = ctx.tex.M;
  const { type, by } = classify(e);
  const poly = e.poly;
  if (!poly || poly.length < 3) return null;
  const grp = new T.Group(); grp.userData.display = true;
  const ob = obb(poly);
  const roomC = ctx.roomCenter(sp);
  const hasShape = e.shape3d?.extrude_height_m > 0;
  const partsBy = (re) => partsMatching(e, re);
  const note = [];
  let dims = null;

  // ---- vertical boards (whiteboard, blackboard, easel, flip chart): on the exported board part when there is one
  if (type === 'whiteboard' || type === 'blackboard' || type === 'easel' || type === 'flipchart') {
    let part = partsBy(/board|write|writing_surface|draw|surface|paper|top/)[0] || (e.parts || [])[0] || null;
    let M4, lp;
    if (part && Math.abs(part.normal[2]) < 0.5) { M4 = partMatrix(T, part.m); lp = part.local; note.push('on part ' + part.name); }
    else {
      // no vertical part: a board of the footprint's long side, facing the room centre (authored display defaults)
      let len = Math.max(ob.w, ob.d), ang = ob.w >= ob.d ? ob.ang : ob.ang + Math.PI / 2;
      let u = [Math.cos(ang), Math.sin(ang)], n = [u[1], -u[0]];
      const c = [ob.cx, ob.cy];
      if ((roomC[0] - c[0]) * n[0] + (roomC[1] - c[1]) * n[1] < 0) { u = [-u[0], -u[1]]; n = [-n[0], -n[1]]; }
      const bottom = type === 'easel' ? DISPLAY.board.easelBottom : type === 'flipchart' ? DISPLAY.board.flipBottom : DISPLAY.board.bottom;
      let hgt = DISPLAY.board.height;
      if (type === 'flipchart' || type === 'easel') len = Math.min(len, 0.7);
      M4 = new T.Matrix4().makeBasis(new T.Vector3(u[0], u[1], 0), new T.Vector3(0, 0, 1), new T.Vector3(n[0], n[1], 0));
      M4.setPosition(c[0] - u[0] * len / 2, c[1] - u[1] * len / 2, bottom);
      lp = [[0, 0], [len, 0], [len, hgt], [0, hgt]];
      note.push('no board part exported: authored default size and height');
    }
    const r = boardInPartFrame(ctx, grp, lp, M4, type === 'blackboard' ? 'blackboard' : 'whiteboard');
    if (type === 'easel' || type === 'flipchart') {
      const bottomLocal = -M4.elements[14] / (M4.elements[5] || 1); // local y of the floor (part frame y is up for vertical boards)
      const mt = M.metal();
      const W = r.x1 - r.x0, cx = (r.x0 + r.x1) / 2;
      const rr = type === 'easel' ? 0.016 : 0.014;
      const fl = (x, z) => [x, bottomLocal, z];
      if (type === 'easel') {
        r.b.strut(mt, [r.x0 + W * 0.08, r.y0 + 0.1, 0.0], fl(r.x0 + W * 0.02, 0.22), rr);
        r.b.strut(mt, [r.x1 - W * 0.08, r.y0 + 0.1, 0.0], fl(r.x1 - W * 0.02, 0.22), rr);
        r.b.strut(mt, [cx, r.y1 - 0.05, -0.03], fl(cx, -0.62), rr);
      } else {
        for (const [x, z] of [[cx - 0.32, 0.26], [cx + 0.32, 0.26], [cx, -0.3]]) r.b.strut(mt, [cx, r.y0 + 0.02, -0.03], fl(x, z), rr);
        r.b.box(M.paper(), W * 0.96, 0.02, (r.y1 - r.y0) * 0.96, cx, 0, 0).position.set(cx, (r.y0 + r.y1) / 2, 0.012);
      }
    }
    dims = { w: r.x1 - r.x0, h: r.y1 - r.y0 };
    return { group: grp, type, by, note: note.join('; '), dims };
  }

  // ---- seated / table-like objects in the footprint frame
  const elongated = ['bench', 'sofa', 'counter', 'shelf'].includes(type);
  let { w, d } = ob, ang = ob.ang;
  if (elongated && w > d) { ang += Math.PI / 2; [w, d] = [d, w]; }
  const cands = [0, 1, 2, 3].map((k) => wrap(ang + (k * Math.PI) / 2)).filter((_, k) => !elongated || k % 2 === 0);
  const { hint, by: hintBy } = facingChoice(e, ob, roomC);
  let yaw = cands[0], bestD = Infinity;
  for (const c of cands) { const dd = Math.abs(wrap(c - hint)); if (dd < bestD) { bestD = dd; yaw = c; } }
  if (!elongated) { const k = Math.round(wrap(yaw - ang) / (Math.PI / 2)); if (Math.abs(k) % 2 === 1) [w, d] = [d, w]; }
  const round = poly.length >= 10 && Math.abs(w - d) < 0.25 * Math.max(w, d);
  note.push(`facing ${hintBy}`);

  const kind = type;
  const seatKinds = ['chair', 'armchair', 'stool', 'bench', 'sofa', 'officechair'];
  let Hs;
  if (seatKinds.includes(kind)) {
    const sp0 = partsBy(/^seat/)[0];
    if (sp0 && Number.isFinite(sp0.origin3[2]) && sp0.origin3[2] > 0.05) { Hs = sp0.origin3[2]; note.push('seat height from the seat part plane'); }
    else if (hasShape) { Hs = e.shape3d.extrude_height_m; note.push('height from shape3d'); }
    else { Hs = DISPLAY.seatH[kind]; note.push('authored default seat height'); }
  } else if (['table', 'desk', 'counter', 'shelf'].includes(kind)) {
    const tp = partsBy(/desk_surface|keyboard_area|shelf_surface|shelf_top|^top|^edge/)[0];
    if (tp && tp.origin3[2] > 0.05) { Hs = tp.origin3[2]; note.push('top height from the part plane'); }
    else if (hasShape) { Hs = e.shape3d.extrude_height_m; note.push('height from shape3d'); }
    else { Hs = DISPLAY.topH[kind]; note.push('authored default height'); }
  } else Hs = hasShape ? e.shape3d.extrude_height_m : 0.05;

  const g = new T.Group(); g.position.set(ob.cx, ob.cy, 0); g.rotation.z = yaw; grp.add(g);
  const b = new B(ctx, g);
  const builder = kind === 'desk' ? BUILD.table : BUILD[kind];
  if (builder) { builder(b, { w, d, Hs, round }); dims = { w, d, h: Hs }; }
  else {
    // fallback: extruded footprint (data only)
    const shape = new T.Shape(poly.map(([x, y]) => new T.Vector2(x, y)));
    const h = hasShape ? e.shape3d.extrude_height_m : 0.012;
    const geo = new T.ExtrudeGeometry(shape, { depth: h, bevelEnabled: false });
    const m = new T.Mesh(geo, M.generic(ctx.kindColor(e.kind))); m.castShadow = hasShape; m.receiveShadow = true; grp.add(m);
    grp.userData.ownGeo = true;
    return { group: grp, type: null, by: null, note: hasShape ? 'extruded footprint (shape3d)' : 'flat footprint (no height exported)', dims: { h } , own: true };
  }
  return { group: grp, type: kind, by, note: note.join('; '), dims, yaw };
}
