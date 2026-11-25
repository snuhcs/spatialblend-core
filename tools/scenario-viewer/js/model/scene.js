// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Scene model: the state of a case at tick t, derived ONLY by indexing the exported channels.
// No geometry is re-implemented. Nothing is interpolated unless channel.continuous is true, and then
// only at a fractional tick and flagged `interpolated` (display only).
import { clamp, isNum, polyArea, polyCentroid, familyClass, quatForward } from '../util.js';
import { G } from '../render/glyph.js';
import { SEG } from '../figure/anthro.js';

const yawOf = (q) => { const f = quatForward(q); return Math.atan2(f[1], f[0]); };
const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export const KNOWN_TYPES = new Set(['pose3d', 'scalar', 'vector', 'enum', 'bitset', 'event', 'scope', 'relation_set', 'proposal', 'joints', 'metrics_row']);
const REL_VEC_COLS = new Set(['contact_p', 'look_at_p', 'rx_point_p']);
const REL_SKIP = new Set(['n', 'offsets', 'families']);

const vec3 = (a, i) => {
  if (!a) return null;
  const x = a[3 * i], y = a[3 * i + 1], z = a[3 * i + 2];
  return isNum(x) && isNum(y) && isNum(z) ? [x, y, z] : null;
};
const quat4 = (a, i) => {
  if (!a) return null;
  const q = [a[4 * i], a[4 * i + 1], a[4 * i + 2], a[4 * i + 3]];
  return q.every(isNum) ? q : [0, 0, 0, 1];
};
const lerp = (a, b, f) => a + (b - a) * f;

/** pose at tick k (valid=0 -> null). frac>0 + continuous -> interpolated (flagged). */
function poseAt(ch, k, frac = 0) {
  const d = ch.data;
  if (!d || !d.n) return null;
  const at = (i) => {
    if (i < 0 || i >= d.n) return null;
    if (d.valid && d.valid[i] === 0) return null;
    const p = vec3(d.p, i);
    return p ? { p, q: quat4(d.q, i) } : null;
  };
  const a = at(k);
  if (!a) return null;
  if (ch.continuous && frac > 1e-6) {
    const b = at(k + 1);
    if (b) {
      let dot = a.q[0] * b.q[0] + a.q[1] * b.q[1] + a.q[2] * b.q[2] + a.q[3] * b.q[3];
      const s = dot < 0 ? -1 : 1;
      const q = a.q.map((v, i) => lerp(v, s * b.q[i], frac));
      const m = Math.hypot(...q) || 1;
      return { p: a.p.map((v, i) => lerp(v, b.p[i], frac)), q: q.map((v) => v / m), interpolated: true };
    }
  }
  return a;
}

function scalarAt(ch, k, frac = 0) {
  const v = ch.data?.v;
  if (!v || k < 0 || k >= v.length) return null;
  const a = v[k];
  if (!isNum(a)) return null;
  if (ch.continuous && frac > 1e-6 && isNum(v[k + 1])) return { v: lerp(a, v[k + 1], frac), interpolated: true };
  return { v: a };
}

const labelAt = (codes, labels, k) => {
  const c = codes?.[k];
  return c === null || c === undefined ? null : (labels?.[c] ?? `#${c}`);
};
const bitNames = (v, bits) => {
  if (!isNum(v)) return [];
  return (bits || []).filter((b) => Math.floor(v / 2 ** b.bit) % 2 === 1).map((b) => b.name);
};

function csrRows(offsets, k) {
  if (!offsets || k < 0 || k + 1 >= offsets.length) return [0, 0];
  return [offsets[k] ?? 0, offsets[k + 1] ?? 0];
}

function scopeAt(ch, k) {
  const d = ch.data;
  const fams = d.families || [];
  const [a, b] = csrRows(d.in_scope?.offsets, k);
  const inScope = [];
  for (let i = a; i < b; i++) inScope.push({ family: fams[d.in_scope.family[i]] ?? `#${d.in_scope.family[i]}`, referent: d.in_scope.referent[i] });
  const [c, e] = csrRows(d.dropped?.offsets, k);
  const dropped = [];
  for (let i = c; i < e; i++) {
    const r = d.dropped.reason?.[i];
    dropped.push({ family: fams[d.dropped.family[i]] ?? `#${d.dropped.family[i]}`, referent: d.dropped.referent[i], reason: r === null || r === undefined ? null : (d.dropped.reasons?.[r] ?? `#${r}`) });
  }
  return {
    mode: labelAt(d.mode?.codes, d.mode?.labels, k),
    modeCode: d.mode?.codes?.[k] ?? null,
    container: d.container?.[k] ?? null,
    transfer: d.transfer?.[k] ?? null,
    border: d.border?.[k] ?? null,
    yawSource: d.yaw_source ? labelAt(d.yaw_source.codes, d.yaw_source.labels, k) : null,
    inScope, dropped,
  };
}

function relationsAt(ch, k) {
  const d = ch.data;
  const [a, b] = csrRows(d.offsets, k);
  const total = d.offsets?.[d.offsets.length - 1] ?? 0;
  const rows = [];
  for (let i = a; i < b; i++) {
    const row = { i, family: d.families?.[d.family?.[i]] ?? `#${d.family?.[i]}`, referent: d.referent?.[i] ?? null };
    for (const [key, arr] of Object.entries(d)) {
      if (REL_SKIP.has(key) || key === 'family' || key === 'referent' || !Array.isArray(arr)) continue;
      if (REL_VEC_COLS.has(key) || (total > 0 && arr.length === total * 3)) row[key] = vec3(arr, i);
      else row[key] = arr[i] ?? null;
    }
    rows.push(row);
  }
  return rows;
}

function proposalAt(ch, k, frac) {
  const d = ch.data;
  const pose = poseAt({ data: d, continuous: ch.continuous }, k, frac);
  const err = {};
  for (const [name, arr] of Object.entries(d.err || {})) if (Array.isArray(arr)) err[name] = arr[k] ?? null;
  return {
    channel: ch.id,
    status: labelAt(d.status?.codes, d.status?.labels, k),
    cause: d.cause ? bitNames(d.cause.v?.[k], d.cause.bits) : [],
    causeRaw: d.cause?.v?.[k] ?? null,
    flags: d.flags ? bitNames(d.flags.v?.[k], d.flags.bits) : [],
    flagsRaw: d.flags?.v?.[k] ?? null,
    pose, err,
  };
}

function jointsAt(ch, k) {
  const d = ch.data;
  const J = d.joint_count || 0;
  if (!J || k < 0 || k >= d.n || !Array.isArray(d.p) || d.p.length < 3 * d.n * J) return null;
  const out = [];
  for (let j = 0; j < J; j++) {
    const p = vec3(d.p, k * J + j);
    out.push(p ? { p, q: quat4(d.q, k * J + j) } : null);
  }
  return out;
}

const mat4Apply = (m, x, y, z) => [m[0] * x + m[1] * y + m[2] * z + m[3], m[4] * x + m[5] * y + m[6] * z + m[7], m[8] * x + m[9] * y + m[10] * z + m[11]];
const isM16 = (m) => Array.isArray(m) && m.length === 16 && m.every(isNum);

/** schema 1.1 element.parts (contract 1): the part polygon lifted to world by its own 4x4 (plane = local z 0, normal = local +z). Unknown fields ignored. */
function readParts(e) {
  if (!Array.isArray(e.parts)) return [];
  const out = [];
  for (const p of e.parts) {
    if (!p || !isM16(p.world_transform) || !Array.isArray(p.footprint_local) || p.footprint_local.length < 3) continue;
    const m = p.world_transform;
    out.push({ name: p.name ?? '', tag: p.tag ?? '', origin: p.origin ?? null, m, local: p.footprint_local,
      world: p.footprint_local.map(([x, y]) => mat4Apply(m, x, y, 0)), normal: [m[2], m[6], m[10]], origin3: [m[3], m[7], m[11]] });
  }
  return out;
}

/** Label of a joints channel from its own fields (contract 3). MotionGen output is never labelled as core output. */
export function motionLabel(ch) {
  const pr = ch?.producer;
  const who = pr?.name ? `${pr.name}${pr.version ? ' ' + pr.version : ''}${/^[0-9a-f]{7,}$/i.test(String(pr.git_sha || '')) ? ' @' + String(pr.git_sha).slice(0, 8) : ''}` : null;
  const d = ch?.derived;
  if (ch?.sidecar) return { text: 'MotionGen output', detail: who || 'producer not stated', kind: 'motiongen' };   // merged from a MotionGen sidecar: always labelled so
  if (d === 'motiongen' || /motiongen/i.test(pr?.name || '')) return { text: 'MotionGen output', detail: who || 'producer not stated', kind: 'motiongen' };
  if (d === 'recorded') return { text: 'recorded motion', detail: who || ch?.origin || 'producer not stated', kind: 'recorded' };
  if (d === 'display-ik') return { text: 'display-derived (IK), not model output', detail: who || '', kind: 'display-ik' };
  return { text: 'joints channel', detail: [who, ch?.origin].filter(Boolean).join(' · ') || 'producer not stated', kind: 'other' };
}

const engagedAt = (ch, k) => {
  if (!ch) return null;
  const d = ch.data || {};
  if (Array.isArray(d.v)) return isNum(d.v[k]) ? d.v[k] >= 0.5 : null;
  if (Array.isArray(d.codes)) { const c = d.codes[k]; if (c === null || c === undefined) return null; const lab = d.labels?.[c]; return lab !== undefined ? /^(1|true|yes|engaged)$/i.test(String(lab)) : c === 1; }
  return null;
};
const perTick = (v, k) => (Array.isArray(v) ? v[k] ?? null : isNum(v) ? v : null);

/** The registered offset of the pelvis anchor from the pelvis joint: the case's own `embodiment` (anchors[{id, bone, offset}]), else the interim
 * display fallback file the loader attached. {offset:{forward,left,up}, origin} or null. */
export function sitboneOf(json) {
  const pick = (emb, label) => {
    const a = Array.isArray(emb?.anchors) ? emb.anchors.find((x) => x?.id === 'pelvis') : null;
    const o = a?.offset;
    const v = Array.isArray(o) ? { forward: o[0], left: o[1], up: o[2] } : o && typeof o === 'object' ? { forward: o.forward, left: o.left, up: o.up } : null;
    if (!v || ![v.forward, v.left, v.up].every((x) => Number.isFinite(x) && Math.abs(x) < 1)) return null;
    return { offset: v, origin: label };
  };
  return pick(json.embodiment, 'sit-bone offset registered by the case embodiment') || pick(json._embodimentFallback, 'sit-bone offset measured on the SMPL-X neutral body, interim display fallback: the case registers none');
}

const STATURE_GUARD_M = 3;   // display-safety upper bound for a stated stature (ASSETS.md, presentation values)

export function buildScene(json) {
  const time = json.time || {};
  const n = time.n_ticks ?? 0;
  const dt = time.dt_s ?? 0;
  const t0 = time.t0_s ?? 0;
  const channels = json.channels || [];
  const chById = new Map(channels.map((c) => [c.id, c]));
  const unknownChannels = channels.filter((c) => !KNOWN_TYPES.has(c.type));

  // ---- spaces & elements
  const spaces = new Map();
  const sharedIndex = new Map(); // shared_id -> [{space,id}]
  for (const sp of json.spaces || []) {
    const elems = new Map();
    for (const e of sp.elements || []) {
      let poly = null, polySource = null;
      if (Array.isArray(e.ground) && e.ground.length >= 3) { poly = e.ground; polySource = 'ground'; }
      else if (Array.isArray(e.footprint_local) && e.footprint_local.length >= 3 && Array.isArray(e.world_transform) && e.world_transform.length === 16) {
        const m = e.world_transform; // fallback only: the app's own 4x4 applied to the local footprint
        poly = e.footprint_local.map(([x, y]) => [m[0] * x + m[1] * y + m[3], m[4] * x + m[5] * y + m[7]]);
        polySource = 'footprint_local x world_transform (no ground image exported)';
      }
      const tr = Array.isArray(e.world_transform) && e.world_transform.length === 16 ? [e.world_transform[3], e.world_transform[7], e.world_transform[11]] : null;
      const el = { ...e, parts: readParts(e), poly, polySource, pos: tr, area: poly ? polyArea(poly) : 0, centroid: poly ? polyCentroid(poly) : tr ? [tr[0], tr[1]] : null };
      elems.set(e.id, el);
      if (e.shared_id !== null && e.shared_id !== undefined) {
        if (!sharedIndex.has(e.shared_id)) sharedIndex.set(e.shared_id, []);
        sharedIndex.get(e.shared_id).push({ space: sp.id, id: e.id });
      }
    }
    spaces.set(sp.id, { id: sp.id, role: sp.role || 'other', raw: sp, elems, list: [...elems.values()], free: sp.free_space || null });
  }
  const spaceIds = [...spaces.keys()];
  const roleSpace = (role) => [...spaces.values()].find((s) => s.role === role)?.id ?? null;

  const chSpace = (ch) => {
    if (ch.space && spaces.has(ch.space)) return ch.space;
    if (ch.id.startsWith('source')) return roleSpace('source');
    if (ch.id.startsWith('receiver')) return roleSpace('receiver');
    if (ch.actor) { const a = (json.actors || []).find((x) => x.id === ch.actor); if (a && spaces.has(a.space)) return a.space; }
    return null;
  };

  // ---- actors
  const sitbone = sitboneOf(json);
  const actors = (json.actors || []).map((a) => ({ ...a, rootCh: chById.get(a.root_channel) || null, jointsCh: a.joints_channel ? chById.get(a.joints_channel) || null : null }));
  const rootIds = new Set(actors.map((a) => a.root_channel));
  // schema 1.1 anchor channels (contract 2): source.anchor.<name>, receiver.anchor.<name>.proposal and the .engaged flags
  const anchorCh = new Map(); // `${role}:${name}` -> {pose, engaged}
  for (const ch of channels) {
    const m = /^(source|receiver)\.anchor\.([A-Za-z0-9_]+?)(\.proposal|\.engaged)?$/.exec(ch.id);
    if (!m) continue;
    const key = `${m[1]}:${m[2]}`;
    const o = anchorCh.get(key) || { role: m[1], name: m[2], pose: null, engaged: null };
    if (m[3] === '.engaged') o.engaged = ch;
    else if (ch.type === 'pose3d' && ((m[1] === 'source' && !m[3]) || (m[1] === 'receiver' && m[3] === '.proposal'))) o.pose = ch;
    anchorCh.set(key, o);
  }
  const anchorIds = new Set([...anchorCh.values()].flatMap((o) => [o.pose?.id, o.engaged?.id].filter(Boolean)));
  const actorsOf = (sid) => actors.filter((a) => a.space === sid);

  // ---- stature of each actor: stated by the case, else derived from its standing pelvis height / 0.530 (Winter Fig. 4.1 hip height), else unknown
  const statureBase = new Map();
  const derivedFrom = (actor) => {
    // only the SOURCE track states a standing height; receiver anchors are proposals (targets, e.g. the seat plane), not a standing pelvis
    if (spaces.get(actor.space)?.role === 'receiver') return null;
    const o = anchorCh.get('source:pelvis');
    if (!o?.pose?.data) return null;
    const d = o.pose.data, e = o.engaged?.data;
    let best = null;
    for (let k = 0; k < (d.n || 0); k++) {
      if (d.valid && d.valid[k] === 0) continue;
      if (e && engagedAt(o.engaged, k) !== false) continue;           // standing = valid and not engaged
      const z = d.p?.[3 * k + 2] - (actor.rootCh ? (poseAt(actor.rootCh, k)?.p[2] ?? 0) : 0);
      if (isNum(z) && z > 0 && (best === null || z > best)) best = z;
    }
    return best === null ? null : { z: best, ch: o.pose.id };
  };
  for (const a of actors) {
    const stated = a.body?.stature_m ?? a.stature_m;
    // display-safety guard (not an anthropometric claim): a stated stature outside (0, 3] m is 'not usable' and falls through to the derived / unknown path
    if (isNum(stated) && stated > 0 && stated <= STATURE_GUARD_M) { statureBase.set(a.id, { value: stated, origin: 'stated by the case (stature_m)', kind: 'stated' }); continue; }
    const dv = derivedFrom(a);
    if (dv) statureBase.set(a.id, { value: dv.z / SEG.hip, origin: `derived: standing pelvis ${dv.z.toFixed(3)} m (${dv.ch}, not engaged) / 0.530 (Winter Fig. 4.1 hip height)`, kind: 'derived' });
  }
  for (const a of actors) if (!statureBase.has(a.id)) {      // another actor of the case with a derived stature (same figure model): labelled as such
    const other = actors.find((x) => x.id !== a.id && statureBase.get(x.id)?.kind === 'derived');
    if (other) statureBase.set(a.id, { ...statureBase.get(other.id), origin: `derived from actor ${other.id}: ${statureBase.get(other.id).origin}`, kind: 'derived' });
  }

  // ---- correspondence
  const corr = json.correspondence || [];
  /** Counterparts of (space,id) in other spaces, via the exported correspondence, then shared_id. */
  function mates(sid, id) {
    const out = [];
    const add = (s, i, via) => { if (s !== sid && spaces.get(s)?.elems.has(i) && !out.some((o) => o.space === s && o.id === i)) out.push({ space: s, id: i, via }); };
    for (const c of corr) {
      const ss = c.source_space ?? roleSpace('source'), rs = c.receiver_space ?? roleSpace('receiver');
      if (ss === sid && c.source === id) add(rs, c.receiver, 'correspondence');
      if (rs === sid && c.receiver === id) add(ss, c.source, 'correspondence');
    }
    const sh = spaces.get(sid)?.elems.get(id)?.shared_id;
    if (sh !== null && sh !== undefined) for (const m of sharedIndex.get(sh) || []) add(m.space, m.id, 'shared_id');
    return out;
  }

  /** Elements of `sid` that a referent (shared id) denotes. Falls back to element id (flagged by the producer's choice). */
  function resolveReferent(sid, ref) {
    const sp = spaces.get(sid);
    if (!sp || ref === null || ref === undefined) return [];
    const bySh = sp.list.filter((e) => e.shared_id === ref);
    if (bySh.length) return bySh;
    const byId = sp.elems.get(ref);
    return byId ? [byId] : [];
  }

  // ---- events
  const events = [];
  for (const ch of channels) if (ch.type === 'event') for (const e of ch.data?.events || []) events.push({ ...e, channel: ch.id });
  events.sort((a, b) => a.tick - b.tick);
  const eventsByTick = new Map();
  for (const e of events) { if (!eventsByTick.has(e.tick)) eventsByTick.set(e.tick, []); eventsByTick.get(e.tick).push(e); }

  // ---- trails (full paths, per space)
  const trails = new Map();
  const pushTrail = (sid, entry) => { if (!trails.has(sid)) trails.set(sid, []); trails.get(sid).push(entry); };
  for (const a of actors) {
    if (!a.rootCh) continue;
    const pts = [];
    for (let k = 0; k < n; k++) pts.push(poseAt(a.rootCh, k)?.p ?? null);
    pushTrail(a.space, { id: a.id, kind: a.role === 'receiver_body' ? 'body' : 'source', pts });
  }
  for (const ch of channels) if (ch.type === 'proposal') {
    const sid = chSpace(ch) || roleSpace('receiver');
    const pts = [];
    for (let k = 0; k < n; k++) pts.push(poseAt({ data: ch.data }, k)?.p ?? null);
    if (sid) pushTrail(sid, { id: ch.id, kind: 'proposal', pts });
  }

  // extra pose channels (not an actor root): drawn as hollow markers + trails, legend by channel id
  const extraChannels = channels.filter((c) => c.type === 'pose3d' && !rootIds.has(c.id));
  extraChannels.forEach((ch, i) => {
    const sid = chSpace(ch);
    if (!sid) return;
    const pts = [];
    for (let k = 0; k < n; k++) pts.push(poseAt(ch, k)?.p ?? null);
    pushTrail(sid, { id: ch.id, kind: 'extra', idx: i, pts });
  });

  // derived walk cycle (display only): cumulative root path length (+ 0.2 m per radian of turning) / stride.
  // Precomputed per root channel so a tick always shows the same pose (no interpolation, scrub = play).
  const walk = new Map();
  const makeWalk = (rootCh) => {
    const phase = new Float32Array(n), amp = new Float32Array(n), steps = [];
    let last = null, lastYaw = 0, cum = 0, foot = 0;
    for (let k = 0; k < n; k++) {
      const pose = poseAt(rootCh, k);
      if (!pose) { phase[k] = cum / G.STRIDE; continue; }
      const yaw = yawOf(pose.q);
      let step = 0;
      if (last) { step = Math.hypot(pose.p[0] - last[0], pose.p[1] - last[1]) + G.YAW_PIVOT * Math.abs(wrapPi(yaw - lastYaw)); cum += step; }
      const before = Math.floor(phase[k - 1] ?? 0);
      phase[k] = cum / G.STRIDE;
      const v = dt > 0 && dt < 1 ? step / dt : step > 0.01 ? 1 : 0; // dt >= 1: ticks are poses
      amp[k] = G.SWING * clamp(v / 0.25, 0, 1);
      if (last && Math.floor(phase[k]) > before) steps.push({ k, x: pose.p[0], y: pose.p[1], yaw, side: (foot++ % 2) ? -1 : 1 });
      last = pose.p; lastYaw = yaw;
    }
    return { phase, amp, steps };
  };
  for (const a of actors) {
    if (!a.rootCh || a.jointsCh) continue;
    walk.set(a.id, makeWalk(a.rootCh));
  }

  // ---- compare mode: v1 / v1eqw ... bodies (eval.<engine>.receiver_body) next to the v2 body
  const v2Ch = actors.find((a) => a.role === 'receiver_body' && a.rootCh)?.rootCh || chById.get('receiver.body') || null;
  const engines = channels.filter((c) => c.type === 'pose3d' && /^eval\.[^.]+\.receiver_body$/.test(c.id)).map((c) => {
    const name = c.id.split('.')[1];
    const pts = [];
    for (let k = 0; k < n; k++) pts.push(poseAt(c, k)?.p ?? null);
    return { name, ch: c, pts, walk: makeWalk(c) };
  });
  const v2Walk = v2Ch ? makeWalk(v2Ch) : null;
  const scalarOf = (id) => { const c = chById.get(id); return c && c.type === 'scalar' ? c.data.v : null; };
  const srcR = new Map(), engR = new Map();
  for (const c of channels) {
    let m = /^eval\.source\.R\.(.+)$/.exec(c.id);
    if (m && c.type === 'scalar') srcR.set(m[1], c.data.v);
    m = /^eval\.([^.]+)\.R_receiver\.(.+)$/.exec(c.id);
    if (m && c.type === 'scalar' && m[1] !== 'source') { if (!engR.has(m[1])) engR.set(m[1], new Map()); engR.get(m[1]).set(m[2], c.data.v); }
  }
  const recvRel = channels.find((c) => c.type === 'relation_set' && chSpace(c) === roleSpace('receiver'));
  /** per-relation error of an engine vs the source at tick k: |R_src - R_rx|, with its origin; null = no data */
  function engineErr(name, k) {
    const er = engR.get(name);
    if (er && srcR.size) {
      const per = [];
      for (const [ref, v] of er) { const a = srcR.get(ref)?.[k], b = v[k]; if (isNum(a) && isNum(b)) per.push({ ref, err: Math.abs(a - b), R_src: a, R_rx: b }); }
      if (per.length) return { mean: per.reduce((t, x) => t + x.err, 0) / per.length, per, origin: `eval.${name}.R_receiver.* vs eval.source.R.*` };
    }
    if (name === 'v2' && recvRel) {
      const rows = relationsAt(recvRel, k).filter((r) => isNum(r.err_rx));
      if (rows.length) return { mean: rows.reduce((t, r) => t + Math.abs(r.err_rx), 0) / rows.length, per: rows.map((r) => ({ ref: r.referent, err: Math.abs(r.err_rx), R_rx: r.R_rx })), origin: 'receiver.relations err_rx' };
    }
    return null;
  }
  /** R of an engine at referent `ref`, tick k (for the referent lines) */
  const engineR = (name, ref, k) => { const v = engR.get(name)?.get(String(ref))?.[k]; return isNum(v) ? v : null; };
  const evalRefs = [...new Set([...srcR.keys(), ...[...engR.values()].flatMap((m) => [...m.keys()])])];
  // An actor's own registration element is NOT a person glyph: its static pose is the pose at export, not where the
  // actor is. New data: actor.element or flag `actor_element`. Old data (no actor.element exported): a person
  // element without a footprint named 'actor' in a space that has an actor.
  const isActorElement = (sp, e) => {
    if (e.flags?.includes('actor_element')) return true;
    const mine = actors.filter((a) => a.space === sp.id);
    if (mine.some((a) => a.element !== null && a.element !== undefined && a.element === e.id)) return true;
    return !actors.some((a) => a.element !== null && a.element !== undefined) && e.kind === 'person' && !e.poly && e.name === 'actor' && mine.length > 0;
  };
  const people = new Map(), actorElements = [];
  for (const sp of spaces.values()) {
    const list = [];
    for (const e of sp.list) {
      if (isActorElement(sp, e)) { actorElements.push({ space: sp.id, id: e.id, name: e.name, kind: e.kind, pos: e.pos }); continue; }
      if (e.kind !== 'person' || !e.pos) continue;
      const m = e.world_transform;
      list.push({ id: e.id, name: e.name, pos: e.pos, yaw: m && m.length === 16 ? Math.atan2(m[4], m[0]) : 0 });
    }
    people.set(sp.id, list);
  }

  // ---- bounds
  const bounds = new Map();
  for (const sp of spaces.values()) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    const take = (pt) => { if (!pt) return; x0 = Math.min(x0, pt[0]); x1 = Math.max(x1, pt[0]); y0 = Math.min(y0, pt[1]); y1 = Math.max(y1, pt[1]); };
    (sp.free?.outer || []).forEach(take);
    sp.list.forEach((e) => (e.poly || []).forEach(take));
    (trails.get(sp.id) || []).forEach((t) => t.pts.forEach(take));
    if (!isFinite(x0)) { x0 = -5; y0 = -5; x1 = 5; y1 = 5; }
    if (x1 - x0 < 2) { x0 -= 1; x1 += 1; }
    if (y1 - y0 < 2) { y0 -= 1; y1 += 1; }
    bounds.set(sp.id, { x0, y0, x1, y1, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w: x1 - x0, h: y1 - y0 });
  }

  // default view focus per space: the actors' root paths (source / body / proposal) plus the footprints of every
  // referent that is in scope at any tick; falls back to the referents, then to the room.
  const focus = new Map();
  const scopeRefs = new Set();
  for (const ch of channels) if (ch.type === 'scope' && !ch.id.startsWith('parity.')) for (const r of ch.data.in_scope?.referent || []) scopeRefs.add(r);
  for (const sp of spaces.values()) {
    const grow = (bb, x, y) => { bb.x0 = Math.min(bb.x0, x); bb.x1 = Math.max(bb.x1, x); bb.y0 = Math.min(bb.y0, y); bb.y1 = Math.max(bb.y1, y); };
    const mk = () => ({ x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity });
    const paths = mk(), refs = mk();
    for (const t of trails.get(sp.id) || []) if (t.kind !== 'extra') for (const pt of t.pts) if (pt) grow(paths, pt[0], pt[1]);
    for (const e of engines) if (chSpace(e.ch) === sp.id) for (const pt of e.pts) if (pt) grow(paths, pt[0], pt[1]);
    if (v2Ch && chSpace(v2Ch) === sp.id) for (let k = 0; k < n; k++) { const q = poseAt(v2Ch, k); if (q) grow(paths, q.p[0], q.p[1]); }
    for (const r of scopeRefs) for (const e of resolveReferent(sp.id, r)) for (const q of e.poly || []) grow(refs, q[0], q[1]);
    const ok = (b) => isFinite(b.x0), ext = (b) => (ok(b) ? Math.max(b.x1 - b.x0, b.y1 - b.y0) : 0);
    let bb = null, kind = 'room';
    if (ext(paths) >= 1 || (ok(paths) && !ok(refs))) {
      bb = { ...paths }; kind = 'paths';
      if (ok(refs)) { grow(bb, refs.x0, refs.y0); grow(bb, refs.x1, refs.y1); kind = 'paths+referents'; }
    } else if (ok(refs)) {
      bb = { ...refs }; kind = 'referents';
      if (ok(paths)) { grow(bb, paths.x0, paths.y0); grow(bb, paths.x1, paths.y1); }
    }
    if (bb && kind === 'paths' && ext(bb) < 1) bb = null; // stationary, no referents: the room
    const B = bounds.get(sp.id);
    if (!bb) { focus.set(sp.id, { cx: B.cx, cy: B.cy, w: B.w, h: B.h, kind: 'room' }); continue; }
    let w = bb.x1 - bb.x0, h = bb.y1 - bb.y0;
    const pad = Math.max(0.5, 0.15 * Math.max(w, h));
    w = Math.max(3, w + 2 * pad); h = Math.max(3, h + 2 * pad);
    focus.set(sp.id, { cx: (bb.x0 + bb.x1) / 2, cy: (bb.y0 + bb.y1) / 2, w, h, kind });
  }

  const bl = [...bounds.values()];
  const aspect = bl.length ? bl.reduce((a, b) => a + b.w / b.h, 0) / bl.length : 1.25;

  // ---- state at tick
  let lastKey = null, lastState = null;
  function at(tick) {
    const t = clamp(Number.isFinite(tick) ? tick : 0, 0, Math.max(0, n - 1));
    const k = Math.floor(t + 1e-9);
    const frac = k >= n - 1 ? 0 : t - k;
    const key = `${k}:${frac.toFixed(4)}`;
    if (key === lastKey) return lastState;
    let interpolated = false;
    const S = { tick: t, k, frac, t: t0 + t * dt, interpolated: false, spaces: {}, events: eventsByTick.get(k) || [] };
    for (const sp of spaces.values()) S.spaces[sp.id] = { id: sp.id, role: sp.role, people: people.get(sp.id) || [], actors: [], extras: [], proposals: [], scope: null, relations: [], marks: new Map() };

    for (const a of actors) {
      const st = S.spaces[a.space];
      if (!st) continue;
      const pose = a.rootCh ? poseAt(a.rootCh, k, frac) : null;
      if (pose?.interpolated) interpolated = true;
      let joints = null;
      if (a.jointsCh && a.jointsCh.type === 'joints') joints = jointsAt(a.jointsCh, k);
      const wt = walk.get(a.id);
      let look = null;
      const lch = a.lookahead_channel ? chById.get(a.lookahead_channel) : null;
      if (lch && lch.type === 'joints') {
        const hzRaw = perTick(lch.data?.horizon ?? lch.horizon, k), hz = hzRaw === null ? null : Math.max(0, Math.floor(hzRaw));
        const at = clamp(k + (hz ?? 0), 0, Math.max(0, (lch.data?.n ?? 1) - 1));
        const fr = jointsAt(lch, at);
        if (fr) look = { joints: fr, horizon: hz, revision: perTick(lch.data?.buffer_revision ?? lch.buffer_revision, k), tick: at, label: motionLabel(lch), channel: lch.id };
      }
      const anchors = {};
      for (const o of anchorCh.values()) {
        if (o.role !== (spaces.get(a.space)?.role === 'receiver' ? 'receiver' : 'source')) continue;
        anchors[o.name] = { pose: o.pose ? poseAt(o.pose, k) : null, engaged: engagedAt(o.engaged, k), channel: o.pose?.id ?? null, hasEngaged: !!o.engaged };
      }
      st.actors.push({ id: a.id, role: a.role, pose, yaw: pose ? yawOf(pose.q) : null, walk: wt && pose ? { phase: wt.phase[k], amp: wt.amp[k] } : null, body: a.body || null, skeleton: a.skeleton || null, joints, element: a.element ?? null,
        jointsMeta: joints && a.jointsCh ? { ...motionLabel(a.jointsCh), channel: a.jointsCh.id, revision: perTick(a.jointsCh.data?.buffer_revision ?? a.jointsCh.buffer_revision, k), horizon: perTick(a.jointsCh.data?.horizon ?? a.jointsCh.horizon, k) } : null,
        look, anchors, sitbone, statureBase: statureBase.get(a.id) || null });
    }

    for (const ch of channels) {
      if (ch.type === 'pose3d' && !rootIds.has(ch.id)) {
        const sid = chSpace(ch);
        if (!sid) continue;
        const pose = poseAt(ch, k, frac);
        if (pose?.interpolated) interpolated = true;
        S.spaces[sid].extras.push({ id: ch.id, label: ch.id, idx: extraChannels.indexOf(ch), pose });
      } else if (ch.type === 'proposal') {
        const sid = chSpace(ch) || roleSpace('receiver');
        if (!sid) continue;
        const pr = proposalAt(ch, k, frac);
        if (pr.pose?.interpolated) interpolated = true;
        const st = S.spaces[sid];
        const rb = (ch.actor && st.actors.find((x) => x.id === ch.actor)) || st.actors.find((x) => x.role === 'receiver_body') || st.actors[0] || null;
        pr.space = sid; pr.bodyPose = rb?.pose || null; pr.body = rb?.body || null; pr.actor = rb?.id ?? null;
        st.proposals.push(pr);
      } else if (ch.type === 'scope') {
        const sc = scopeAt(ch, k);
        sc.channel = ch.id;
        sc.space = chSpace(ch) || roleSpace('source');
        const parity = ch.id.startsWith('parity.');
        for (const sid of spaceIds) {
          const marks = S.spaces[sid].marks;
          const mark = (id) => { if (!marks.has(id)) marks.set(id, { inScope: [], dropped: [], container: false, containerBorder: null, parity: [] }); return marks.get(id); };
          if (parity) {
            for (const r of sc.inScope) for (const e of resolveReferent(sid, r.referent)) { const m = mark(e.id); if (!m.parity.includes(ch.id)) m.parity.push(ch.id); }
            continue;
          }
          for (const r of sc.inScope) for (const e of resolveReferent(sid, r.referent)) mark(e.id).inScope.push(r.family);
          for (const r of sc.dropped) for (const e of resolveReferent(sid, r.referent)) mark(e.id).dropped.push({ family: r.family, reason: r.reason });
          if (sc.container !== null && sc.container !== undefined) {
            if (sid === sc.space && spaces.get(sid).elems.has(sc.container)) { const m = mark(sc.container); m.container = true; m.containerBorder = sc.border; }
            else if (sc.space) for (const m of mates(sc.space, sc.container)) if (m.space === sid) { const mm = mark(m.id); mm.container = true; mm.containerBorder = sc.border; }
          }
        }
        if (parity) { (S.parityScopes ||= []).push(sc); continue; }
        const target = S.spaces[sc.space];
        if (target) target.scope = sc;
      }
    }

    // relation lines (resolved against the exported anchors only)
    for (const ch of channels) {
      if (ch.type !== 'relation_set') continue;
      const sid = chSpace(ch);
      if (!sid || !S.spaces[sid]) continue;
      const st = S.spaces[sid];
      const role = spaces.get(sid).role;
      const anchorActor = st.actors.find((a) => (role === 'receiver' ? a.role === 'receiver_body' : a.role === 'source_actor') && a.pose) || st.actors.find((a) => a.pose);
      const prop = st.proposals.find((p) => p.pose);
      for (const row of relationsAt(ch, k)) {
        const fc = familyClass(row.family);
        const base = { channel: ch.id, space: sid, family: row.family, fc, referent: row.referent, weight: row.weight ?? null, row };
        if (role === 'receiver') {
          // Facing: the ray starts at the embodied body (its head); other families at the proposal
          const bodyA = st.actors.find((x) => x.role === 'receiver_body' && x.pose) || anchorActor;
          const src = fc === 'facing' ? (bodyA?.pose || prop?.pose) : (prop?.pose || anchorActor?.pose);
          const a = src?.p;
          if (a && row.rx_point_p) st.relations.push({ ...base, kind: 'rx', a, b: row.rx_point_p, yaw: fc === 'facing' && bodyA?.pose ? bodyA.yaw : null, body: bodyA?.body || null });
          else st.relations.push({ ...base, kind: 'none' });
        } else {
          const a = anchorActor?.pose?.p;
          let drew = false;
          if (a && row.contact_p) { st.relations.push({ ...base, kind: 'contact', a, b: row.contact_p }); drew = true; }
          if (a && row.look_at_p) { st.relations.push({ ...base, kind: 'look_at', a, b: row.look_at_p, yaw: anchorActor?.yaw ?? null, body: anchorActor?.body || null }); drew = true; }
          if (!drew) st.relations.push({ ...base, kind: 'none' });
        }
      }
    }
    if (engines.length) {
      const bodyState = (ch, wt) => { const pose = ch ? poseAt(ch, k, 0) : null; return { pose, yaw: pose ? yawOf(pose.q) : null, walk: wt && pose ? { phase: wt.phase[k], amp: wt.amp[k] } : null }; };
      S.v2 = bodyState(v2Ch, v2Walk);
      S.engines = engines.map((e) => ({ name: e.name, ...bodyState(e.ch, e.walk) }));
    }
    S.interpolated = interpolated;
    lastKey = key; lastState = S;
    return S;
  }

  /** Raw per-tick slice of every channel (inspector + "raw JSON"). Unknown types are listed, not decoded. */
  function inspect(tick) {
    const k = clamp(Math.floor(tick + 1e-9), 0, Math.max(0, n - 1));
    return channels.map((ch) => {
      let slice;
      switch (ch.type) {
        case 'pose3d': { const d = ch.data; slice = { valid: d.valid ? d.valid[k] : 1, p: vec3(d.p, k), q: quat4(d.q, k) }; if (d.valid && d.valid[k] === 0) slice = { valid: 0 }; break; }
        case 'scalar': slice = { v: ch.data.v?.[k] ?? null }; break;
        case 'vector': { const dim = ch.data.dim; const v = (ch.data.v || []).slice(k * dim, k * dim + dim); slice = ch.data.names ? Object.fromEntries(ch.data.names.map((nm, i) => [nm, v[i] ?? null])) : { v }; break; }
        case 'enum': slice = { code: ch.data.codes?.[k] ?? null, label: labelAt(ch.data.codes, ch.data.labels, k) }; break;
        case 'bitset': { const v = ch.data.v?.[k] ?? null; slice = { v, set: bitNames(v, ch.data.bits) }; break; }
        case 'event': slice = { events_at_tick: (eventsByTick.get(k) || []).filter((e) => e.channel === ch.id), total: ch.data.events?.length ?? 0 }; break;
        case 'scope': slice = scopeAt(ch, k); break;
        case 'relation_set': slice = { rows: relationsAt(ch, k) }; break;
        case 'proposal': { const p = proposalAt(ch, k, 0); delete p.channel; slice = p; break; }
        case 'joints': { const names = actors.find((a) => a.joints_channel === ch.id)?.skeleton?.joints; slice = { joints: (jointsAt(ch, k) || []).map((j, i) => ({ name: names?.[i] ?? i, ...(j || { p: null }) })) }; break; }
        case 'metrics_row': slice = ch.data.row ?? {}; break;
        default: slice = { note: `unknown channel type "${ch.type}": listed, not decoded` };
      }
      return { id: ch.id, type: ch.type, space: ch.space ?? null, actor: ch.actor ?? null, unit: ch.unit, doc: ch.doc, continuous: !!ch.continuous, known: KNOWN_TYPES.has(ch.type), slice };
    });
  }

  return {
    json, n, dt, t0, channels, chById, unknownChannels,
    engines, v2Ch, engineErr, engineR, evalRefs, srcR, engR, focus, walk, actorElements, spaces, spaceIds, actors, correspondence: corr, bounds, trails, events, aspect,
    synthetic: !!json.synthetic,
    hasJoints: actors.some((a) => a.jointsCh && a.skeleton),
    extraChannels, anchorIds, anchorCh, timeUnit: dt >= 1 ? 'pose' : 's', chSpace, mates, resolveReferent, roleSpace, at, inspect,
    /** time label: seconds, or 'pose k' when dt_s >= 1 (unit cases are poses, not seconds) */
    timeLabel: (k) => (dt >= 1 ? `pose ${k}` : `t ${+(t0 + k * dt).toFixed(4)} s`),
    tickToSeconds: (k) => t0 + k * dt,
    headingOf: (pose) => (pose ? quatForward(pose.q) : null),
  };
}

/** Human description of a picked entity (tooltip). */
export function describeEntity(scene, state, ent) {
  if (!ent) return null;
  const lines = [];
  const sp = scene.spaces.get(ent.space);
  if (ent.kind === 'element') {
    const e = sp?.elems.get(ent.id);
    if (!e) return null;
    lines.push([`${e.name || 'element'} #${e.id}`, `${e.kind}${e.shared_id != null ? ` · shared ${e.shared_id}` : ''}`]);
    if (e.flags?.length) lines.push(['flags', e.flags.join(', ')]);
    const mk = state?.spaces[ent.space]?.marks.get(ent.id);
    if (mk) {
      if (mk.inScope.length) lines.push(['in scope', [...new Set(mk.inScope)].join(', ')]);
      for (const d of mk.dropped) lines.push(['dropped', `${d.family}: ${d.reason ?? 'reason not exported'}`]);
      if (mk.container) lines.push(['role', `container / entered referent (${mk.containerBorder === 1 ? 'border relation present' : mk.containerBorder === 0 ? 'no border relation' : 'border unknown'})`]);
      for (const pc of mk.parity || []) lines.push(['parity', `${pc} includes this`]);
    }
    const ms = scene.mates(ent.space, ent.id);
    if (ms.length) lines.push(['counterpart', ms.map((m) => `${m.space} #${m.id}`).join(', ')]);
    if (e.shape3d?.extrude_height_m != null) lines.push(['height', `${e.shape3d.extrude_height_m} m`]);
  } else if (ent.kind === 'actor') {
    const a = state?.spaces[ent.space]?.actors.find((x) => x.id === ent.id);
    lines.push([`actor ${ent.id}`, a?.role ?? '']);
    if (a?.pose) lines.push(['p', a.pose.p.map((v) => +v.toFixed(3)).join(', ')]);
  } else if (ent.kind === 'proposal') {
    const p = state?.spaces[ent.space]?.proposals[ent.index ?? 0];
    lines.push(['proposal', p?.status ?? 'no status']);
    if (p?.cause?.length) lines.push(['cause', p.cause.join(', ')]);
  } else if (ent.kind === 'relation') {
    const r = state?.spaces[ent.space]?.relations[ent.index];
    if (r) lines.push([`${r.family} #${r.referent}`, `w ${r.weight ?? '–'}`]);
  }
  return lines;
}
