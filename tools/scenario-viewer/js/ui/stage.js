// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// One stage, three toggles: renderer (2D / 3D), spaces (both side by side / one), lock. Plus layers, HUD, tooltip, legend.
import { h, statusClass, debounceRaf } from '../util.js';
import { TopDown2D } from '../render/topdown2d.js';
import { Perspective3D } from '../render/perspective3d.js';
import { ENGINE_VARS } from '../render/theme.js';
import { LAYERS, DEFAULT_LAYERS, extraKey, EXTRA_VARS } from '../render/renderer.js';
import { describeEntity } from '../model/scene.js';
import { figureFor } from '../figure/pose.js';
import { statureFor, DISPLAY } from '../model/display.js';

// the kind of view survives a case switch; the GL renderer is parked (not destroyed) so textures and geometry caches live on
let lastKind = '2d';
let parked = null;

export function createStage({ onStatus }) {
  const st = { kind: lastKind, loading: false, spaces: 'both', lock: true, layers: new Set(DEFAULT_LAYERS), scene: null, tick: 0, renderer: null, hover: null };
  const wrap = h('div', { class: 'stage-wrap' });
  const hud = h('div', { class: 'hud' });
  const tip = h('div', { class: 'tip', hidden: true });
  const bar = h('div', { class: 'stage-bar' });
  const legend = h('div', { class: 'legend' });
  const msg = h('div', { class: 'note', role: 'status' });
  const root = h('div', null, bar, wrap, msg, legend);
  const mount = h('div', { style: { position: 'absolute', inset: 0 } });
  wrap.append(mount, hud, tip);

  // controls
  const kindSeg = h('div', { class: 'seg', role: 'group', 'aria-label': 'renderer' });
  const b2 = h('button', { type: 'button', onclick: () => setKind('2d') }, '2D top-down');
  const b3 = h('button', { type: 'button', onclick: () => setKind('3d') }, '3D perspective');
  kindSeg.append(b2, b3);
  const spaceSeg = h('div', { class: 'seg', role: 'group', 'aria-label': 'spaces shown' });
  const lockLbl = h('label', null, h('input', { type: 'checkbox', checked: true, onchange: (e) => { st.lock = e.target.checked; st.renderer?.setCamera({ lock: st.lock }); } }), 'Lock views');
  let labelMode = 'key';
  try { const v = localStorage.getItem('osv.labels'); if (['all', 'key', 'none'].includes(v)) labelMode = v; } catch (e) { /* storage may be blocked */ }
  const labelSeg = h('div', { class: 'seg', role: 'group', 'aria-label': 'labels' });
  const labelBtns = ['all', 'key', 'none'].map((m) => h('button', { type: 'button', title: m === 'all' ? 'all names and relation labels' : m === 'key' ? 'actor names, the top-weight relation and the gaze angle' : 'no labels', 'aria-pressed': String(labelMode === m), onclick: () => setLabelMode(m) }, m));
  labelSeg.append(h('span', { class: 'note', style: { alignSelf: 'center', marginRight: '6px' } }, 'labels:'), ...labelBtns);
  function setLabelMode(m) {
    labelMode = m;
    try { localStorage.setItem('osv.labels', m); } catch (e) { /* ignore */ }
    labelBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(['all', 'key', 'none'][i] === m)));
    st.renderer?.setLabelMode?.(m);
  }
  const cmpWrap = h('span', { style: { display: 'inline-flex', gap: '10px' }, hidden: true });
  let compareOn = true, overlayOn = false;
  const cmpBox = h('input', { type: 'checkbox', checked: true, onchange: (e) => { compareOn = e.target.checked; ovBox.disabled = !compareOn; applyCompare(); } });
  const ovBox = h('input', { type: 'checkbox', onchange: (e) => { overlayOn = e.target.checked; applyCompare(); } });
  cmpWrap.append(h('label', null, cmpBox, 'Compare'), h('label', { title: 'draw the v1 / v1eqw body as a dashed glyph in the v2 pane' }, ovBox, 'Overlay v1 in v2 pane'));
  function applyCompare() {
    wrap.classList.toggle('compare', !!(compareOn && st.scene?.engines.length && st.kind === '2d'));
    camSeg.hidden = st.kind !== '3d'; statLbl.hidden = st.kind !== '3d';
    cmpWrap.hidden = !(st.scene?.engines.length || (st.kind === '3d' && st.scene?.roleSpace('receiver')));
    ovBox.parentElement.hidden = st.kind === '3d';
    st.renderer?.setCompare?.({ on: compareOn, overlay: overlayOn });
    updateLegend();
  }
  const camSeg = h('div', { class: 'seg', role: 'group', 'aria-label': '3D camera', hidden: true });
  const camBtns = [['orbit', 'Orbit'], ['follow', 'Follow']].map(([m, l]) => h('button', { type: 'button', 'aria-pressed': String(m === 'orbit'), onclick: () => { camBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(['orbit', 'follow'][i] === m))); st.renderer?.setCamera?.({ mode: m }); } }, l));
  camSeg.append(...camBtns);
  const statIn = h('input', { type: 'number', min: '0.5', max: '2.5', step: '0.05', placeholder: 'enter', 'aria-label': 'stature in metres', style: { width: '64px' }, onchange: (e) => { const v = parseFloat(e.target.value); DISPLAY.stature = v > 0 ? v : null; st.renderer?.setStature?.(); updateHud(); window.dispatchEvent(new Event('osv-display')); } });
  const statLbl = h('label', { hidden: true, title: 'Display parameter: the stature the IK figure is drawn at (segment lengths are fractions of it). Empty = the case stature (stated, or derived from its standing pelvis height), else no figure is drawn: a capsule marker and the label "stature unknown - enter a value".' }, 'stature m ', statIn);
  const resetBtn = h('button', { type: 'button', onclick: () => st.renderer?.setCamera({ reset: true }) }, 'Reset view');
  const layersMenu = h('details', { class: 'layers-menu' }, h('summary', null, 'Layers'));
  const pop = h('div', { class: 'pop' });
  for (const [id, label] of LAYERS) pop.append(h('label', null, h('input', { type: 'checkbox', checked: st.layers.has(id), dataset: { layer: id }, onchange: (e) => { e.target.checked ? st.layers.add(id) : st.layers.delete(id); applyLayers(); } }), label));
  layersMenu.append(pop);
  bar.append(kindSeg, spaceSeg, cmpWrap, camSeg, statLbl, labelSeg, lockLbl, resetBtn, layersMenu);

  function applyLayers() {
    const set = new Set([...st.layers].filter((x) => !x.startsWith('space:')));
    if (st.scene) {
      const ids = st.spaces === 'both' ? st.scene.spaceIds : [st.spaces];
      for (const id of ids) set.add('space:' + id);
    }
    hud.hidden = !st.layers.has('hud');
    st.renderer?.setLayers(set);
    updateHud();
  }

  function buildSpaceSeg() {
    spaceSeg.replaceChildren();
    if (!st.scene) return;
    const opts = [['both', 'Both'], ...st.scene.spaceIds.map((id) => [id, `${id} only`])];
    if (st.scene.spaceIds.length < 2) opts.shift();
    if (!opts.some((o) => o[0] === st.spaces)) st.spaces = opts[0][0];
    for (const [v, label] of opts) spaceSeg.append(h('button', { type: 'button', 'aria-pressed': String(st.spaces === v), onclick: () => { st.spaces = v; buildSpaceSeg(); applyLayers(); sizeWrap(); } }, label));
    spaceSeg.hidden = opts.length < 2;
  }

  function sizeWrap() {
    const n = st.spaces === 'both' && st.scene ? st.scene.spaceIds.length : 1;
    wrap.classList.toggle('stacked', n > 1 && wrap.clientWidth < 720);
  }

  let rtoken = 0;
  async function makeRenderer() {
    const tok = ++rtoken;
    if (st.renderer) { st.renderer.dispose(); st.renderer = null; }
    mount.replaceChildren();
    st.loading = true;
    try {
      if (st.kind === '3d') {
        if (parked) { st.renderer = parked; parked = null; st.renderer.adopt(mount); }
        else { msg.textContent = 'Loading the 3D view (three.js 0.160.0, first use) …'; st.renderer = await Perspective3D.create(mount); }
        if (tok !== rtoken) { st.renderer.dispose(); st.renderer = null; return; }
        msg.textContent = '';
      } else { st.renderer = new TopDown2D(); st.renderer.mount(mount); }
    } catch (e) {
      st.renderer?.dispose?.(); st.renderer = null; mount.replaceChildren();
      msg.textContent = `${st.kind === '3d' ? '3D' : '2D'} renderer unavailable: ${e.message || e}`;
      st.loading = false;
      if (st.kind === '3d') { st.kind = lastKind = '2d'; return makeRenderer(); }
      return;
    }
    st.loading = false;
    b2.setAttribute('aria-pressed', String(st.kind === '2d')); b3.setAttribute('aria-pressed', String(st.kind === '3d'));
    if (st.scene) { st.renderer.setScene(st.scene); }
    st.renderer.setCamera({ lock: st.lock });
    st.renderer.setLabelMode?.(labelMode);
    applyCompare();
    applyLayers();
    st.renderer.setTime(st.tick);
    wireHover();
    updateLegend();
    updateHud();
  }

  function setKind(k) {
    st.kind = lastKind = k; sizeWrap();
    if (st.renderer && st.renderer.kind === (k === '3d' ? 'perspective3d' : 'topdown2d')) return Promise.resolve();
    if (st.renderer?.kind === 'perspective3d' && k !== '3d') { st.renderer.park(); parked = st.renderer; st.renderer = null; }
    return makeRenderer();
  }

  // hover: pick, correspondence highlight, tooltip
  let hoverWired = false;
  function wireHover() {
    if (hoverWired) return; hoverWired = true;
    const c = mount;
    const move = debounceRaf(() => {
      if (!st.renderer || !st.pt) return;
      const ent = st.renderer.pick(st.pt.x, st.pt.y);
      const key = ent ? JSON.stringify(ent) : '';
      if (key !== st.hoverKey) { st.hoverKey = key; st.hover = ent; st.renderer.setHighlight(ent); }
      showTip(ent);
    });
    c.addEventListener('pointermove', (e) => { const r = wrap.getBoundingClientRect(); st.pt = { x: e.clientX - r.left, y: e.clientY - r.top }; move(); });
    c.addEventListener('pointerleave', () => { st.pt = null; st.hoverKey = ''; st.hover = null; st.renderer?.setHighlight(null); tip.hidden = true; });
  }
  function showTip(ent) {
    const lines = st.scene && ent ? describeEntity(st.scene, st.scene.at(st.tick), ent) : null;
    if (!lines?.length) { tip.hidden = true; return; }
    tip.replaceChildren(...lines.map(([k, v], i) => h('div', null, h('span', { class: i ? 'tk' : '', style: i ? null : { fontWeight: 600 } }, i ? k + ' ' : k), i ? v : v ? ` ${v}` : '')));
    tip.hidden = false;
    const w = wrap.clientWidth, x = Math.min(st.pt.x + 14, w - tip.offsetWidth - 6);
    tip.style.left = Math.max(4, x) + 'px'; tip.style.top = Math.max(4, st.pt.y + 14) + 'px';
  }

  function updateHud() {
    hud.replaceChildren();
    const sc = st.scene;
    if (!sc) return;
    const S = sc.at(st.tick);
    const r1 = h('div', { class: 'row' }), r2 = h('div', { class: 'row' }), r3 = h('div', { class: 'row' });
    if (sc.n) r1.append(h('span', { class: 'chip' }, `${sc.timeUnit === 'pose' ? 'pose' : 'tick'} ${S.k}/${Math.max(0, sc.n - 1)} · ${sc.timeLabel(S.k)}`));
    if (S.interpolated) r1.append(h('span', { class: 'chip warn', title: 'a channel marked continuous is interpolated between ticks' }, 'INTERPOLATED (display only)'));
    if (sc.synthetic) r1.append(h('span', { class: 'chip synthetic' }, sc.hasJoints ? 'SYNTHETIC · skeleton is a fixture' : 'SYNTHETIC'));
    for (const sid of sc.spaceIds) {
      const sp = S.spaces[sid];
      if (sp.scope?.mode) {
        r2.append(h('span', { class: 'chip spacing' }, `mode ${sp.scope.mode}`));
        if (sp.scope.transfer) r2.append(h('span', { class: 'chip afford' }, 'transfer'));
        if (sp.scope.yawSource) r2.append(h('span', { class: 'chip' }, `yaw ${sp.scope.yawSource}`));
      }
      for (const pr of sp.proposals) {
        const cls = statusClass(pr.status);
        r3.append(h('span', { class: `chip ${cls === 'neutral' ? '' : cls} solid`, title: `${pr.channel} status` }, pr.status ?? 'no status'));
        for (const t of pr.cause) r3.append(h('span', { class: 'chip', title: 'cause' }, t));
        for (const t of pr.flags) r3.append(h('span', { class: 'chip afford', title: 'flag' }, t));
      }
    }
    if (st.kind === '3d') {
      const r4 = h('div', { class: 'row' });
      const seen = new Set();
      for (const sid of sc.spaceIds) for (const a of S.spaces[sid].actors) {
        const fig = figureFor(a);
        if (!fig) continue;
        const k = fig.label.kind;
        r4.append(h('span', { class: `chip ${k === 'motiongen' || k === 'recorded' ? 'afford' : k === 'unknown' ? 'bad' : 'warn'} wrap`, title: `${fig.label.detail || ''}\n${fig.posture}` }, `${a.id}: ${fig.label.text}${fig.label.kind === 'motiongen' && fig.label.detail ? ' · ' + fig.label.detail : ''}`));
        if (Object.keys(fig.residues || {}).length) r4.append(h('span', { class: 'chip bad wrap', title: 'display IK could not reach the exported anchor' }, `IK short: ${Object.entries(fig.residues).map(([n, mm]) => `${n} ${mm} mm`).join(', ')}`));
        if (a.look) r4.append(h('span', { class: 'chip warn wrap' }, `look-ahead buffer (MotionGen, not yet played)${a.look.revision != null ? ' rev ' + a.look.revision : ''}`));
        seen.add(fig.kind === 'generic' ? '' : fig.stature.origin + '|' + (fig.H || 0).toFixed(3));
      }
      const r5 = h('div', { class: 'row' });
      for (const sx of seen) { if (!sx) continue; const [o, H] = sx.split('|'); r5.append(h('span', { class: `chip wrap${+H > 0 ? '' : ' bad'}`, title: 'display parameter' }, +H > 0 ? `stature ${(+H).toFixed(2)} m · ${o}` : o)); }
      r5.append(h('span', { class: 'chip wrap', title: 'furniture is built from the exported footprint, shape3d height and parts; the leg / back styles are display choices' }, 'furniture: display-derived'));
      if (sc.roleSpace('receiver') && compareOn && !sc.engines.some((e) => e.name === 'v1')) r5.append(h('span', { class: 'chip wrap' }, 'receiver v1: not available (no eval.v1.* body channel in this case)'));
      hud.append(r4, r5);
    }
    hud.append(r1, r2, r3);
  }

  function updateLegend() {
    legend.replaceChildren(
      h('span', null, h('span', { class: 'sw', style: { borderColor: 'var(--c-spacing)' } }), 'Spacing'),
      h('span', null, h('span', { class: 'sw', style: { borderColor: 'var(--c-facing)' } }), 'Facing'),
      h('span', null, h('span', { class: 'sw', style: { borderColor: 'var(--c-afford)' } }), 'AffordancePose'),
      h('span', { title: 'relation line thickness encodes the annealed weight' }, 'weight:', ' ', h('span', { class: 'sw w1', style: { borderColor: 'var(--ink-2)', marginLeft: '4px' } }), '0 ', h('span', { class: 'sw w3', style: { borderColor: 'var(--ink-2)' } }), '0.5 ', h('span', { class: 'sw w6', style: { borderColor: 'var(--ink-2)' } }), '1'),
      h('span', null, h('span', { class: 'sw dash', style: { borderColor: 'var(--ink)' } }), 'body to proposal / container outline'),
      ...(st.scene?.engines.length && compareOn && st.kind === '2d' ? ['v2', ...st.scene.engines.map((e) => e.name)].map((n) => h('span', { title: 'engine colour' }, h('span', { class: 'sw', style: { borderColor: `var(${ENGINE_VARS[n] || '--s3'})`, borderTopWidth: '5px' } }), `engine ${n}`)) : []),
      ...(st.scene?.extraChannels || []).filter((c) => !(compareOn && st.kind === '2d' && st.scene.engines.some((e) => e.ch.id === c.id)))
.map((c) => h('span', { title: 'extra pose channel (hollow marker + trail)' }, h('span', { class: 'sw', style: { borderColor: `var(${EXTRA_VARS[st.scene.extraChannels.indexOf(c) % 8]})`, borderTopStyle: 'dotted' } }), c.id)),
      st.kind === '2d' ? h('span', { class: 'note' }, 'glyph size, not data (shoulders 2 x body radius, else 0.45 m; min 28 px). Walk animation is derived from the root path for readability (stride 0.7 m, swing 0.25 m, +0.2 m per rad of turn); not a motion model. Wedge = where the body faces (+-30 deg, 1.5 m, display constant); ray = the Facing relation target; arc = angle between them (derived).') : null,
      h('span', { class: 'note' }, st.kind === '3d' ? 'dashed outline = transfer leaf · furniture and the humanoid are display-derived (procedural meshes from the exported footprint, height and parts; IK figure from exported anchors, never model output) · teal = affordance part, ring = engaged anchor · walls facing the camera are hidden' : 'hatch = transfer leaf · dim = dropped (hover for the reason) · hover an element for its counterpart'));
  }

  return {
    root,
    get renderer() { return st.renderer; },
    get state() { return st; },
    setScene(scene) {
      st.scene = scene;
      const empty = !scene.spaceIds.length;
      wrap.hidden = bar.hidden = legend.hidden = empty;
      msg.textContent = empty ? 'This case has no spatial content (spec only): see the results below.' : '';
      if (empty) { if (st.renderer?.kind === 'perspective3d') { st.renderer.park(); parked?.dispose?.(); parked = st.renderer; } else st.renderer?.dispose(); st.renderer = null; mount.replaceChildren(); hud.replaceChildren(); return; }
      if (st.spaces !== 'both' && !scene.spaces.has(st.spaces)) st.spaces = 'both';
      cmpWrap.hidden = !scene.engines.length; buildSpaceSeg(); sizeWrap();
      if (!st.renderer) { if (!st.loading) makeRenderer(); }
      else {
        updateLegend(); st.renderer.setScene(scene); st.renderer.setLabelMode?.(labelMode); cmpWrap.hidden = !scene.engines.length; applyCompare(); st.renderer.setCamera({ lock: st.lock }); applyLayers(); st.renderer.setTime(st.tick); }
      updateHud();
    },
    setTime(t) { st.tick = t; st.renderer?.setTime(t); updateHud(); },
    setKind,
    dispose() { if (st.renderer?.kind === 'perspective3d') { st.renderer.park(); parked?.dispose?.(); parked = st.renderer; st.renderer = null; } else st.renderer?.dispose(); },
  };
}
