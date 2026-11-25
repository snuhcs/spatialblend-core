// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Canvas colours come from the computed CSS tokens (never hard-coded in renderers).
const TOKENS = ['bg', 'panel', 'panel-2', 'ink', 'ink-2', 'ink-3', 'line', 'line-2', 'hl', 'grid', 'grid-major',
  'c-spacing', 'c-facing', 'c-afford', 'good', 'warn', 'bad', 'f-fill',
  'k-room', 'k-region', 'k-surface', 'k-object', 'k-obstacle', 'k-iregion', 'k-border', 'person',
  's1', 's2', 's3', 's4', 'e-v2', 'e-v1', 'e-v1eqw'];
const camel = (s) => s.replace(/-([a-z0-9])/g, (_, c) => c.toUpperCase());

export function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  const o = {};
  for (const t of TOKENS) o[camel(t)] = cs.getPropertyValue('--' + t).trim() || '#888888';
  o.fontMono = cs.getPropertyValue('--font-mono').trim() || 'monospace';
  o.fontSans = cs.getPropertyValue('--font-sans').trim() || 'sans-serif';
  o.reduceMotion = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
  return o;
}

export function familyColor(theme, cls) {
  return cls === 'spacing' ? theme.cSpacing : cls === 'facing' ? theme.cFacing : cls === 'afford' ? theme.cAfford : theme.ink2;
}

/** Calls cb whenever the effective theme may have changed. Returns an unsubscribe function. */
export function watchTheme(cb) {
  let mq = null, mo = null;
  const fire = () => requestAnimationFrame(cb);
  try {
    mq = window.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) mq.addEventListener('change', fire); else if (mq.addListener) mq.addListener(fire);
  } catch (e) { /* no matchMedia */ }
  try {
    mo = new MutationObserver(fire);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  } catch (e) { /* no MutationObserver */ }
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(fire).catch(() => {});
  return () => {
    try { if (mq) { if (mq.removeEventListener) mq.removeEventListener('change', fire); else if (mq.removeListener) mq.removeListener(fire); } } catch (e) { /* */ }
    try { if (mo) mo.disconnect(); } catch (e) { /* */ }
  };
}

/** engine colour: v2 = the receiver colour, v1 / v1eqw fixed tokens, any other engine from the series palette */
export function engineColor(theme, name, i = 0) {
  return name === 'v2' ? theme.eV2 : name === 'v1' ? theme.eV1 : name === 'v1eqw' ? theme.eV1eqw : [theme.s3, theme.s4, theme.cAfford, theme.cFacing][i % 4];
}
export const ENGINE_VARS = { v2: '--e-v2', v1: '--e-v1', v1eqw: '--e-v1eqw' };
