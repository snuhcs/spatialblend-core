// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Inspector: every exported field of the current tick as key/value; raw JSON on demand; unknown channels listed.
import { h, fmt, isNum, safeJSON } from '../util.js';
import { figureFor } from '../figure/pose.js';
import { describeFurniture } from '../render/furniture.js';

function flat(v, prefix, out, depth = 0) {
  if (v === null || v === undefined) { out.push([prefix, '–']); return; }
  if (Array.isArray(v)) {
    if (v.every((x) => isNum(x) || x === null) && v.length <= 16) { out.push([prefix, `[${v.map((x) => fmt(x, 4)).join(', ')}]`]); return; }
    if (!v.length) { out.push([prefix, '[]']); return; }
    v.slice(0, 40).forEach((x, i) => flat(x, `${prefix}[${i}]`, out, depth + 1));
    if (v.length > 40) out.push([prefix, `… ${v.length - 40} more`]);
    return;
  }
  if (typeof v === 'object') {
    const es = Object.entries(v);
    if (!es.length) { out.push([prefix, '{}']); return; }
    for (const [k, x] of es) flat(x, prefix ? `${prefix}.${k}` : k, out, depth + 1);
    return;
  }
  out.push([prefix, typeof v === 'number' ? fmt(v, 5) : String(v)]);
}

export function createInspector() {
  const root = h('div', { class: 'insp' });
  let scene = null, tick = 0, raw = false;
  const rawBtn = h('button', { type: 'button', 'aria-pressed': 'false', onclick: () => { raw = !raw; rawBtn.setAttribute('aria-pressed', String(raw)); rawBtn.textContent = raw ? 'Hide raw JSON' : 'Show raw JSON'; render(); } }, 'Show raw JSON');
  const body = h('div', { class: 'insp-scroll' });
  root.append(h('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', marginBottom: '6px' } }, h('h3', null, 'Inspector'), rawBtn), body);

  function render() {
    body.replaceChildren();
    if (!scene) return;
    const k = Math.floor(tick);
    const ins = scene.inspect(tick);
    body.append(h('div', { class: 'mono' }, scene.n ? `${scene.timeUnit === 'pose' ? 'pose' : 'tick'} ${k} · ${scene.timeLabel(k)}` : 'no ticks (spec only): static content only'));
    const unknown = ins.filter((c) => !c.known);
    if (unknown.length) body.append(h('div', { class: 'chip warn', style: { display: 'block', whiteSpace: 'normal', margin: '6px 0', padding: '3px 6px' } }, `Unknown channel types, listed and ignored: ${unknown.map((c) => `${c.id} (${c.type})`).join(', ')}`));
    if (scene.actorElements.length) {
      const tip = "the actor's own registration element; its pose here is the pose at export, not where the actor is";
      body.append(h('div', { class: 'ch', title: tip }, h('div', { class: 'ct' }, h('span', { class: 'mono', style: { fontWeight: 600 } }, 'actor elements'), h('span', { class: 'chip', title: tip }, 'not drawn')),
        h('div', { class: 'note' }, tip),
        h('div', { class: 'kv' }, ...scene.actorElements.flatMap((e) => [h('div', { class: 'k' }, `${e.space} #${e.id}`), h('div', { class: 'v', title: tip }, `${e.name ?? ''} (${e.kind}) static pose ${e.pos ? e.pos.map((v) => +v.toFixed(3)).join(', ') : 'n/a'}`)]))));
    }
    // display figures (3D): where each person's pose comes from, and what the viewer derived (never model output unless labelled so)
    if (scene.n) {
      const S = scene.at(tick), rows = [];
      for (const sid of scene.spaceIds) for (const a of S.spaces[sid].actors) {
        const fig = figureFor(a);
        if (!fig) continue;
        rows.push([`${sid}/${a.id}`, `${fig.label.text}${fig.label.detail ? ' · ' + fig.label.detail : ''}`]);
        rows.push(['  pose source', fig.kind === 'joints' ? `exported joints channel ${a.jointsMeta?.channel ?? ''}${a.jointsMeta?.revision != null ? ' · buffer_revision ' + a.jointsMeta.revision : ''}${a.jointsMeta?.horizon != null ? ' · horizon ' + a.jointsMeta.horizon : ''}` : fig.kind === 'ik' ? 'display IK from the root pose and the exported pose3d anchors' : 'skeleton drawn as bones']);
        rows.push(['  posture', fig.posture]);
        rows.push(['  stature', fig.H > 0 ? `${fmt(fig.H, 3)} m · ${fig.stature.origin}` : fig.stature.origin]);
        const rs = Object.entries(fig.residues || {});
        rows.push(['  IK reach shortfall', rs.length ? rs.map(([n, mm]) => `${n} ${mm} mm`).join(', ') : fig.kind === 'ik' ? 'none (every exported anchor reached)' : 'n/a']);
        if (a.look) rows.push(['  look-ahead', `buffer (MotionGen, not yet played) at tick ${a.look.tick}${a.look.revision != null ? ' · rev ' + a.look.revision : ''}`]);
      }
      if (rows.length) body.append(h('div', { class: 'ch' }, h('div', { class: 'ct' }, h('span', { class: 'mono', style: { fontWeight: 600 } }, 'display figures'), h('span', { class: 'chip warn', title: 'display only; the trace is not modified' }, '3D · display')),
        h('div', { class: 'note' }, 'Figures are drawn by the viewer from what the case exports. IK figures are display-derived, not model output; MotionGen joints are shown as MotionGen output.'),
        h('div', { class: 'kv' }, ...rows.flatMap(([kk, vv]) => [h('div', { class: 'k' }, kk), h('div', { class: 'v' }, vv)]))));
    }
    // procedural furniture: what is display-derived and which exported thing chose its facing
    {
      const rows = [];
      for (const sp of scene.spaces.values()) {
        const rm = sp.list.find((e) => e.kind === 'room' && e.centroid), b = scene.bounds.get(sp.id), rc = rm ? rm.centroid : [b.cx, b.cy];
        for (const e of sp.list) { const d = describeFurniture(e, rc); if (d) rows.push([`${sp.id} · ${e.name || 'element'} #${e.id}`, d.text]); }
      }
      if (rows.length) body.append(h('div', { class: 'ch' }, h('div', { class: 'ct' }, h('span', { class: 'mono', style: { fontWeight: 600 } }, 'furniture (3D)'), h('span', { class: 'chip warn' }, 'display-derived')),
        h('div', { class: 'note' }, 'Footprint, height and parts are exported; leg and back styles and the facing rule are display choices.'),
        h('div', { class: 'kv' }, ...rows.flatMap(([kk, vv]) => [h('div', { class: 'k' }, kk), h('div', { class: 'v' }, vv)]))));
    }
    for (const c of ins) {
      const rows = []; flat(c.slice, '', rows);
      body.append(h('div', { class: 'ch' },
        h('div', { class: 'ct' }, h('span', { class: 'mono', style: { fontWeight: 600 } }, c.id), h('span', { class: 'chip' }, c.type), c.space ? h('span', { class: 'muted mono' }, c.space) : null, c.unit ? h('span', { class: 'muted' }, c.unit) : null, c.continuous ? h('span', { class: 'chip warn', title: 'producer marks this channel continuous; interpolation is display-only' }, 'continuous') : null),
        c.doc ? h('div', { class: 'note' }, c.doc) : null,
        h('div', { class: 'kv' }, ...rows.flatMap(([kk, vv]) => [h('div', { class: 'k' }, kk || 'value'), h('div', { class: 'v' }, vv)]))));
    }
    if (raw) body.append(h('pre', { tabindex: 0, 'aria-label': 'raw JSON of the current tick' }, safeJSON({ tick: k, channels: Object.fromEntries(ins.map((c) => [c.id, { type: c.type, ...c.slice }])) })));
  }
  const onDisp = () => { if (root.isConnected) render(); else window.removeEventListener('osv-display', onDisp); }; window.addEventListener('osv-display', onDisp);
  return { root, setScene(s) { scene = s; tick = 0; render(); }, setTick(t) { const nk = Math.floor(t); if (nk === Math.floor(tick) && scene) { tick = t; return; } tick = t; render(); } };
}
