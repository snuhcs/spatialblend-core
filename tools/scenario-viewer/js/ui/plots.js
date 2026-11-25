// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Small time plots under the stage. Built only from channels that exist (model/series.js).
import { h, isNum, withAlpha, statusClass } from '../util.js';
import { readTheme, familyColor, watchTheme } from '../render/theme.js';
import { buildSeries } from '../model/series.js';

const PH = 88;

export function createPlots({ onTick }) {
  const root = h('div');
  let items = [], scene = null, tick = 0, theme = null;

  const colorOf = (th, s) => s.color ? th[s.color] : s.cls === 'neutral' ? th.s1 : familyColor(th, s.cls);

  function draw(it) {
    const th = theme || (theme = readTheme());
    const cv = it.canvas, r = cv.getBoundingClientRect();
    const W = Math.max(60, Math.floor(r.width)), dpr = Math.min(3, window.devicePixelRatio || 1);
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(it.h * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(it.h * dpr); cv.style.height = it.h + 'px'; }
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = th.panel; ctx.fillRect(0, 0, W, it.h);
    const n = scene.n, L = 38, R = 6, T = 6, B = it.strip ? 4 : 14;
    const x = (k) => L + (k / Math.max(1, n - 1)) * (W - L - R);
    ctx.strokeStyle = th.line; ctx.lineWidth = 1; ctx.strokeRect(L + 0.5, T + 0.5, W - L - R, it.h - T - B);
    // event ticks (faint)
    ctx.strokeStyle = withAlpha(th.ink3, 0.35); ctx.lineWidth = 1;
    for (const e of scene.events) { ctx.beginPath(); ctx.moveTo(Math.round(x(e.tick)) + 0.5, T); ctx.lineTo(Math.round(x(e.tick)) + 0.5, it.h - B); ctx.stroke(); }

    if (it.strip) {
      const labels = it.spec.labels;
      const codes = it.spec.codes;
      for (let k = 0; k < n; k++) {
        const c = codes[k];
        if (c === null || c === undefined) continue;
        const lbl = labels[c] ?? `#${c}`;
        const col = it.spec.status ? { good: th.good, warn: th.warn, bad: th.bad, neutral: th.ink3 }[statusClass(lbl)] : [th.s1, th.s2, th.s3, th.s4, th.cSpacing, th.cAfford][c % 6];
        ctx.fillStyle = col; ctx.globalAlpha = 0.8;
        ctx.fillRect(x(k) - (W - L - R) / Math.max(1, n - 1) / 2, T + 1, Math.max(1, (W - L - R) / Math.max(1, n - 1)), it.h - T - B - 2);
      }
      ctx.globalAlpha = 1;
    } else {
      let lo = Infinity, hi = -Infinity;
      for (const s of it.spec.series) for (const v of s.values) if (isNum(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      for (const b of it.spec.bounds || []) { lo = Math.min(lo, b.value); hi = Math.max(hi, b.value); }
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
      const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
      const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (it.h - T - B);
      ctx.font = `10px ${th.fontMono}`; ctx.fillStyle = th.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(+hi.toFixed(3), L - 4, T + 6); ctx.fillText(+lo.toFixed(3), L - 4, it.h - B - 6);
      for (const s of it.spec.series) {
        ctx.beginPath(); let pen = false;
        s.values.forEach((v, k) => { if (!isNum(v)) { pen = false; return; } if (pen) ctx.lineTo(x(k), y(v)); else { ctx.moveTo(x(k), y(v)); pen = true; } });
        ctx.strokeStyle = colorOf(th, s); ctx.globalAlpha = s.alpha ?? 1; ctx.lineWidth = 1.7; ctx.setLineDash(s.dash || []); ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
      }
      (it.spec.bounds || []).forEach((b, bi) => {
        const yy = y(b.value);
        ctx.strokeStyle = th.bad; ctx.lineWidth = 1.2; ctx.setLineDash([6, 3]); ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(W - R, yy); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = th.bad; ctx.textAlign = 'left'; ctx.fillText(b.label, L + 4 + bi * 0, yy - 7);
      });
    }
    // current-tick cursor
    const k = Math.floor(tick);
    ctx.strokeStyle = th.hl; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(x(k) + 0.5, T - 2); ctx.lineTo(x(k) + 0.5, it.h - B + 2); ctx.stroke();
    if (!it.strip) {
      ctx.fillStyle = th.ink3; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.font = `10px ${th.fontMono}`;
      const pu = scene.timeUnit === 'pose' ? 'pose #' : 'tick'; ctx.fillText(`${pu} 0`, L, it.h - B + 2); ctx.textAlign = 'right'; ctx.fillText(String(Math.max(0, n - 1)), W - R, it.h - B + 2);
    }
    // value readout at the cursor
    if (it.valEl && !it.strip) it.valEl.textContent = it.spec.series.map((s) => (isNum(s.values[k]) ? +s.values[k].toFixed(4) : '–')).join(' · ');
    if (it.valEl && it.strip) { const c = it.spec.codes[k]; it.valEl.textContent = c === null || c === undefined ? '–' : (it.spec.labels[c] ?? `#${c}`); }
  }

  function scrub(it, e) {
    const r = it.canvas.getBoundingClientRect(), L = 38, R = 6;
    const t = Math.round(((e.clientX - r.left - L) / (r.width - L - R)) * Math.max(1, scene.n - 1));
    onTick(Math.max(0, Math.min(scene.n - 1, t)));
  }

  function block(spec, strip) {
    const canvas = h('canvas', { 'aria-label': `${spec.title} over time; click or drag to set the tick` });
    const valEl = h('span', { class: 'mono muted' });
    const sw = strip ? null : h('div', { class: 'swatches' }, ...spec.series.map((s) => h('span', null, h('i', { style: { borderColor: colorOf(readTheme(), s), opacity: s.alpha ?? 1, borderTopStyle: s.dash?.length ? 'dashed' : 'solid' } }), s.name)));
    const el = h('div', { class: 'plot' }, h('div', { class: 'ph' }, h('h3', null, spec.title, spec.unit ? ` (${spec.unit})` : ''), valEl, spec.note ? h('span', { class: 'note' }, spec.note) : null), sw, canvas);
    const it = { spec, canvas, strip, h: strip ? 26 : PH, valEl, el, swEl: sw };
    let drag = false;
    canvas.addEventListener('pointerdown', (e) => { drag = true; canvas.setPointerCapture?.(e.pointerId); scrub(it, e); });
    canvas.addEventListener('pointermove', (e) => { if (drag) scrub(it, e); });
    canvas.addEventListener('pointerup', () => { drag = false; });
    canvas.addEventListener('pointercancel', () => { drag = false; });
    return it;
  }

  const ro = new ResizeObserver(() => items.forEach(draw));
  watchTheme(() => { theme = null; items.forEach((it) => { if (it.swEl) it.swEl.replaceWith((it.swEl = h('div', { class: 'swatches' }, ...it.spec.series.map((s) => h('span', null, h('i', { style: { borderColor: colorOf(readTheme(), s), opacity: s.alpha ?? 1, borderTopStyle: s.dash?.length ? 'dashed' : 'solid' } }), s.name))))); }); items.forEach(draw); });

  return {
    root,
    setScene(sc) {
      scene = sc; tick = 0; items.forEach((it) => ro.unobserve(it.canvas));
      root.replaceChildren(h('h3', { style: { marginBottom: '8px' } }, 'Time plots'));
      const { plots, strips } = buildSeries(sc);
      items = sc.n ? [...plots.map((p) => block(p, false)), ...strips.map((s) => block({ ...s, series: [] }, true))] : [];
      if (!sc.n) root.append(h('p', { class: 'note' }, 'This case has no ticks (spec only): nothing to plot.'));
      else if (!items.length) root.append(h('p', { class: 'note' }, 'No plottable channel in this case (no scalar, relation or proposal channel).'));
      for (const it of items) { root.append(it.el); ro.observe(it.canvas); }
      requestAnimationFrame(() => items.forEach(draw));
    },
    setTick(t) { tick = t; for (const it of items) draw(it); },
    get count() { return items.length; },
  };
}
