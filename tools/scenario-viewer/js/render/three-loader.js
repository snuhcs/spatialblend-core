// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Lazy loader for three.js: loaded on the FIRST use of the 3D view, never at page load.
// Exact version pinned (0.160.0, cdnjs ES-module build). The bytes are fetched with an SRI check (fetch `integrity`),
// then imported from a Blob URL, so a tampered or wrong file is refused. If the host forbids blob: scripts the same
// URL is imported directly (version-pinned, SRI not enforced) and `threeInfo.sri` says so. The module build prints no
// console message (the UMD build prints a deprecation warning).
export const THREE_VERSION = '0.160.0';
export const THREE_URL = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.module.min.js';
export const THREE_SRI = 'sha384-8/hf5tabFfIo0BPku31TKUvGIKpWGYaMwNtlSNqNyZzXgZCMd4lnOGWbJElML9dv';
export const threeInfo = { version: THREE_VERSION, url: THREE_URL, sri: 'not loaded', ms: null };

let pending = null;
let mod = null;

export const threeLoaded = () => mod;

export function loadThree() {
  if (mod) return Promise.resolve(mod);
  if (pending) return pending;
  pending = (async () => {
    const t0 = performance.now();
    let text;
    try {
      const res = await fetch(THREE_URL, { integrity: THREE_SRI, mode: 'cors', credentials: 'omit' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      text = await res.text();
    } catch (e) {
      throw new Error(`three.js ${THREE_VERSION} could not be loaded from ${new URL(THREE_URL).host} or failed its SRI check (${e.message || e}).`);
    }
    let m;
    try {
      const url = URL.createObjectURL(new Blob([text], { type: 'text/javascript' }));
      try { m = await import(/* @vite-ignore */ url); } finally { URL.revokeObjectURL(url); }
      threeInfo.sri = 'verified';
    } catch (e) {
      m = await import(/* @vite-ignore */ THREE_URL);
      threeInfo.sri = 'bytes verified, imported by URL (blob: scripts refused by the host)';
    }
    threeInfo.ms = Math.round(performance.now() - t0);
    mod = m;
    return m;
  })().catch((e) => { pending = null; throw e; });
  return pending;
}
