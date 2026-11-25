// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Timeline: scrubber, play/pause, speed, step +-1, event marks (from event channels only).
import { h, clamp, eventClass } from '../util.js';
import { readTheme, familyColor, watchTheme } from '../render/theme.js';

export function createTimeline({ onTick }) {
  const S = { scene: null, tick: 0, playing: false, speed: 1, raf: 0, last: 0, theme: null };
  const play = h('button', { type: 'button', 'aria-label': 'play', onclick: () => toggle() }, '▶ Play');
  const prev = h('button', { type: 'button', 'aria-label': 'step back one tick', onclick: () => step(-1) }, '← 1');
  const next = h('button', { type: 'button', 'aria-label': 'step forward one tick', onclick: () => step(1) }, '1 →');
  const prevEv = h('button', { type: 'button', title: 'previous event', onclick: () => jumpEvent(-1) }, '◀ event');
  const nextEv = h('button', { type: 'button', title: 'next event', onclick: () => jumpEvent(1) }, 'event ▶');
  const range = h('input', { type: 'range', min: 0, max: 0, step: 1, value: 0, 'aria-label': 'tick', oninput: (e) => set(+e.target.value, true) });
  const speed = h('select', { 'aria-label': 'speed', onchange: (e) => { S.speed = +e.target.value; } }, ...[0.1, 0.25, 0.5, 1, 2, 4].map((v) => h('option', { value: v, selected: v === 1 }, `${v}×`)));
  const readout = h('span', { class: 'mono', style: { minWidth: '130px' } });
  const strip = h('canvas', { class: 'evstrip', 'aria-label': 'event marks; click to jump', title: 'event marks (click to jump)' });
  const kinds = h('div', { class: 'kinds' });
  const evline = h('div', { class: 'note', role: 'status' });
  const root = h('div', null,
    h('div', { class: 'tl-row' }, play, prev, range, next, speed, readout),
    h('div', { class: 'tl-row', style: { marginTop: '6px' } }, prevEv, nextEv, evline), strip, kinds);

  const maxTick = () => Math.max(0, (S.scene?.n ?? 1) - 1);
  function set(t, fromUser) {
    if (fromUser && S.playing) pause();
    S.tick = clamp(t, 0, maxTick());
    range.value = Math.floor(S.tick);
    refresh();
    onTick(S.tick);
  }
  function step(d) { pause(); set(Math.round(S.tick) + d); }
  function pause() { S.playing = false; cancelAnimationFrame(S.raf); play.textContent = '▶ Play'; play.setAttribute('aria-label', 'play'); S.tick = Math.floor(S.tick); }
  function toggle() {
    if (S.playing) { pause(); set(S.tick); return; }
    if (!S.scene) return;
    if (S.tick >= maxTick()) S.tick = 0;
    S.playing = true; S.last = performance.now(); play.textContent = '❚❚ Pause'; play.setAttribute('aria-label', 'pause');
    const loop = (now) => {
      if (!S.playing) return;
      const dt = (now - S.last) / 1000; S.last = now;
      const tps = S.speed / (S.scene.dt || 1 / 60);
      S.tick = Math.min(maxTick(), S.tick + dt * tps);
      range.value = Math.floor(S.tick); refresh(); onTick(S.tick);
      if (S.tick >= maxTick()) { pause(); set(maxTick()); return; }
      S.raf = requestAnimationFrame(loop);
    };
    S.raf = requestAnimationFrame(loop);
  }
  function jumpEvent(dir) {
    if (!S.scene) return;
    const k = Math.round(S.tick);
    const ticks = [...new Set(S.scene.events.map((e) => e.tick))].sort((a, b) => a - b);
    const t = dir > 0 ? ticks.find((x) => x > k) : [...ticks].reverse().find((x) => x < k);
    if (t !== undefined) { pause(); set(t); }
  }
  function refresh() {
    const sc = S.scene;
    const k = Math.floor(S.tick);
    readout.textContent = sc ? (sc.n ? `${sc.timeUnit === 'pose' ? 'pose' : 'tick'} ${k}/${maxTick()} · ${sc.timeLabel(k)}` : 'no ticks (spec only)') : '';
    const evs = sc?.events.filter((e) => e.tick === k) || [];
    evline.textContent = evs.length ? `events: ${evs.map((e) => e.kind).join(', ')}` : '';
    drawStrip();
  }
  function drawStrip() {
    const sc = S.scene;
    const th = S.theme || (S.theme = readTheme());
    const r = strip.getBoundingClientRect();
    const W = Math.max(40, Math.floor(r.width)), H = 26, dpr = Math.min(3, window.devicePixelRatio || 1);
    if (strip.width !== Math.round(W * dpr)) { strip.width = Math.round(W * dpr); strip.height = Math.round(H * dpr); }
    const ctx = strip.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = th.panel; ctx.fillRect(0, 0, W, H);
    if (!sc) return;
    const x = (t) => 4 + (t / Math.max(1, maxTick())) * (W - 8);
    for (const e of sc.events) {
      const cls = eventClass(e.kind);
      ctx.strokeStyle = { bad: th.bad, warn: th.warn, spacing: th.cSpacing, facing: th.cFacing, afford: th.cAfford, neutral: th.ink2 }[cls];
      ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(Math.round(x(e.tick)) + 0.5, 3); ctx.lineTo(Math.round(x(e.tick)) + 0.5, H - 3); ctx.stroke();
    }
    ctx.strokeStyle = th.ink; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x(S.tick), 0); ctx.lineTo(x(S.tick), H); ctx.stroke();
    ctx.fillStyle = th.ink; ctx.beginPath(); ctx.moveTo(x(S.tick) - 4, 0); ctx.lineTo(x(S.tick) + 4, 0); ctx.lineTo(x(S.tick), 5); ctx.fill();
  }
  strip.addEventListener('pointerdown', (e) => {
    if (!S.scene) return;
    const r = strip.getBoundingClientRect(), W = r.width;
    const t = ((e.clientX - r.left - 4) / (W - 8)) * Math.max(1, maxTick());
    let best = null, bd = 8;
    for (const ev of S.scene.events) { const d = Math.abs(4 + (ev.tick / Math.max(1, maxTick())) * (W - 8) - (e.clientX - r.left)); if (d < bd) { bd = d; best = ev; } }
    set(best ? best.tick : Math.round(t), true);
  });
  new ResizeObserver(() => drawStrip()).observe(strip);
  watchTheme(() => { S.theme = null; drawStrip(); });
  document.addEventListener('keydown', (e) => {
    const t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
    if (e.key === ' ' && !(t && t.tagName === 'BUTTON')) { e.preventDefault(); toggle(); }
    else if (e.key === ',' || e.key === '[') step(-1);
    else if (e.key === '.' || e.key === ']') step(1);
  });

  return {
    root,
    setScene(scene) {
      pause(); S.scene = scene; S.tick = 0;
      for (const b of [play, prev, next, prevEv, nextEv, range, speed]) b.disabled = !scene.n;
      const none = !scene.n;
      for (const b of [play, prev, next, prevEv, nextEv, range, speed]) b.disabled = none;
      range.max = maxTick(); range.value = 0;
      kinds.replaceChildren(...(() => {
        const cnt = new Map();
        for (const e of scene.events) cnt.set(e.kind, (cnt.get(e.kind) || 0) + 1);
        return cnt.size ? [...cnt].map(([k, n]) => h('span', { class: `chip ${eventClass(k) === 'neutral' ? '' : eventClass(k)}` }, `${k} ×${n}`)) : [h('span', { class: 'note' }, 'No event channel in this case.')];
      })());
      refresh();
    },
    get tick() { return S.tick; },
    set: (t) => set(t, true),
  };
}
