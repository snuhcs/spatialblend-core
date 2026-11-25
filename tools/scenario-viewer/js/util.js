// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Small shared helpers (no DOM state).
export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
export const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

export function fmt(v, d = 3) {
  if (v === null || v === undefined) return '–';
  if (typeof v === 'number') return Number.isFinite(v) ? String(+v.toFixed(d)) : String(v);
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  return String(v);
}

/** Tiny DOM builder: h('div', {class:'x', onclick:fn, dataset:{a:1}}, child, 'text'). */
export function h(tag, props, ...kids) {
  const e = document.createElement(tag);
  if (props) for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k === 'dataset') Object.assign(e.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) {
    if (c === null || c === undefined || c === false) continue;
    e.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return e;
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

/** Rotate vector v by unit quaternion q = [x,y,z,w]. */
export function quatRotate(q, v) {
  const [x, y, z, w] = q;
  const [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  return [vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)];
}
export const quatForward = (q) => quatRotate(q, [1, 0, 0]);

export function polyArea(poly) {
  let a = 0;
  for (let i = 0, n = poly.length; i < n; i++) {
    const p = poly[i], r = poly[(i + 1) % n];
    a += p[0] * r[1] - r[0] * p[1];
  }
  return Math.abs(a) / 2;
}
export function polyCentroid(poly) {
  let a = 0, cx = 0, cy = 0;
  for (let i = 0, n = poly.length; i < n; i++) {
    const p = poly[i], r = poly[(i + 1) % n];
    const c = p[0] * r[1] - r[0] * p[1];
    a += c; cx += (p[0] + r[0]) * c; cy += (p[1] + r[1]) * c;
  }
  if (Math.abs(a) < 1e-12) {
    const n = poly.length || 1;
    return [poly.reduce((s, p) => s + p[0], 0) / n, poly.reduce((s, p) => s + p[1], 0) / n];
  }
  return [cx / (3 * a), cy / (3 * a)];
}
export function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** '#rrggbb' + alpha -> 'rgba(...)'. Non-hex input is returned unchanged. */
export function withAlpha(hex, a) {
  const m = /^#([0-9a-f]{6})$/i.exec((hex || '').trim());
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
}

export function familyClass(label) {
  const s = String(label ?? '').toLowerCase();
  if (s.startsWith('spac')) return 'spacing';
  if (s.startsWith('fac')) return 'facing';
  if (s.startsWith('aff')) return 'afford';
  return 'other';
}

export function statusClass(label) {
  const s = String(label ?? '').toUpperCase();
  if (!s || s === 'NONE') return 'neutral';
  if (/^EXACT/.test(s)) return 'good';
  if (/INEXACT|APPROX|PARTIAL|CLAMP/.test(s)) return 'warn';
  if (/NO_|NOT_|FAIL|INVALID|ERR|REJECT/.test(s)) return 'bad';
  return 'neutral';
}

export function eventClass(kind) {
  const s = String(kind ?? '').toUpperCase();
  if (/GUARD|FAIL/.test(s)) return 'bad';
  if (/TELEPORT/.test(s)) return 'bad';
  if (/SWITCH/.test(s)) return 'warn';
  if (/MODE/.test(s)) return 'spacing';
  if (/TRANSFER/.test(s)) return 'afford';
  if (/STATUS/.test(s)) return 'facing';
  return 'neutral';
}

export function debounceRaf(fn) {
  let pending = false;
  return () => { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; fn(); }); };
}

export function safeJSON(v, space = 2) {
  try { return JSON.stringify(v, null, space); } catch (e) { return String(e); }
}
