// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Procedural canvas textures + shared PBR materials. No external asset. Deterministic (seeded). Cached for the whole
// session (textures and materials are shared and are NOT disposed on a case switch; `disposeShared` frees them).
// One texture tile = 1 m x 1 m of surface unless noted, so geometry UVs are in metres and nothing is stretched.

function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function canvas(n) { const c = document.createElement('canvas'); c.width = c.height = n; return c; }

const painters = {
  // parquet: 1 m tile, 4 strips of 0.25 m x 1 m with slightly different tones and grain lines
  parquet(c) {
    const g = c.getContext('2d'), N = c.width, R = rng(11);
    const base = ['#b98d5d', '#c29767', '#ad8253', '#bf9462'];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 2; j++) {
        g.fillStyle = base[(i * 2 + j) % 4]; const x = (i * N) / 4, y = (j * N) / 2; g.fillRect(x, y, N / 4, N / 2);
        g.strokeStyle = 'rgba(80,50,25,0.18)'; g.lineWidth = 1;
        for (let k = 0; k < 14; k++) { const gx = x + R() * (N / 4); g.beginPath(); g.moveTo(gx, y); g.bezierCurveTo(gx + (R() - .5) * 8, y + N / 6, gx + (R() - .5) * 8, y + N / 3, gx + (R() - .5) * 4, y + N / 2); g.stroke(); }
        g.strokeStyle = 'rgba(40,25,10,0.55)'; g.strokeRect(x + .5, y + .5, N / 4 - 1, N / 2 - 1);
      }
    }
  },
  plaster(c) {
    const g = c.getContext('2d'), N = c.width, R = rng(5);
    g.fillStyle = '#e8e3da'; g.fillRect(0, 0, N, N);
    for (let i = 0; i < 2600; i++) { const v = 205 + R() * 40; g.fillStyle = `rgba(${v},${v - 4},${v - 12},0.35)`; g.fillRect(R() * N, R() * N, 2 + R() * 4, 2 + R() * 4); }
  },
  wood(c) { // light wood grain along x, 1 m tile
    const g = c.getContext('2d'), N = c.width, R = rng(23);
    g.fillStyle = '#c9a373'; g.fillRect(0, 0, N, N);
    for (let i = 0; i < 90; i++) { const y = R() * N, a = 0.06 + R() * 0.16; g.strokeStyle = `rgba(95,60,28,${a})`; g.lineWidth = 0.6 + R() * 1.8; g.beginPath(); g.moveTo(0, y); for (let x = 0; x <= N; x += 32) g.lineTo(x, y + Math.sin(x / 70 + i) * 3 + (R() - .5) * 1.2); g.stroke(); }
  },
  fabric(c) { // woven weave, tile 0.25 m (set by repeat)
    const g = c.getContext('2d'), N = c.width;
    g.fillStyle = '#b4b4b4'; g.fillRect(0, 0, N, N);
    const s = N / 16;
    for (let i = 0; i < 16; i++) for (let j = 0; j < 16; j++) { const o = (i + j) % 2; g.fillStyle = o ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)'; g.fillRect(i * s, j * s, s - 1, s - 1); }
  },
  whiteboard(c) { // 1 m tile: white surface, soft vertical sheen, faint grid of erased marks
    const g = c.getContext('2d'), N = c.width, R = rng(31);
    const gr = g.createLinearGradient(0, 0, N, N); gr.addColorStop(0, '#fbfbfa'); gr.addColorStop(0.5, '#f1f2f1'); gr.addColorStop(1, '#fafaf9'); g.fillStyle = gr; g.fillRect(0, 0, N, N);
    g.strokeStyle = 'rgba(90,90,100,0.06)'; g.lineWidth = 6; for (let i = 0; i < 6; i++) { g.beginPath(); g.moveTo(R() * N, R() * N); g.lineTo(R() * N, R() * N); g.stroke(); }
  },
  blackboard(c) {
    const g = c.getContext('2d'), N = c.width, R = rng(37);
    g.fillStyle = '#26382f'; g.fillRect(0, 0, N, N);
    for (let i = 0; i < 40; i++) { g.strokeStyle = `rgba(210,220,215,${0.02 + R() * 0.05})`; g.lineWidth = 8 + R() * 18; g.beginPath(); g.moveTo(R() * N, R() * N); g.lineTo(R() * N, R() * N); g.stroke(); }
  },
  metal(c) {
    const g = c.getContext('2d'), N = c.width, R = rng(41);
    g.fillStyle = '#a9adb2'; g.fillRect(0, 0, N, N);
    for (let i = 0; i < 160; i++) { const v = 150 + R() * 70; g.strokeStyle = `rgba(${v},${v},${v + 6},0.35)`; g.lineWidth = 1; const y = R() * N; g.beginPath(); g.moveTo(0, y); g.lineTo(N, y + (R() - .5) * 3); g.stroke(); }
  },
  carpet(c) {
    const g = c.getContext('2d'), N = c.width, R = rng(53);
    g.fillStyle = '#6d7c8c'; g.fillRect(0, 0, N, N);
    for (let i = 0; i < 6000; i++) { const v = 90 + R() * 70; g.fillStyle = `rgba(${v},${v + 12},${v + 26},0.4)`; g.fillRect(R() * N, R() * N, 2, 2); }
  },
};

export function makeTextures(THREE, maxAniso = 8) {
  const cache = new Map();
  const tex = (name, size = 512) => {
    let t = cache.get(name);
    if (!t) {
      const c = canvas(size); painters[name](c);
      t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = maxAniso; t.userData.shared = true;
      cache.set(name, t);
    }
    return t;
  };
  const mats = new Map();
  const mat = (key, make) => { let m = mats.get(key); if (!m) { m = make(); m.userData.shared = true; mats.set(key, m); } return m; };
  const std = (o) => new THREE.MeshStandardMaterial(o);
  const M = {
    floor: () => mat('floor', () => std({ map: tex('parquet'), roughness: 0.55, metalness: 0 })),
    carpet: () => mat('carpet', () => std({ map: tex('carpet'), roughness: 0.95, metalness: 0 })),
    wall: () => mat('wall', () => std({ map: tex('plaster'), roughness: 0.95, metalness: 0, side: THREE.FrontSide })),
    wood: (tint = 0xffffff) => mat('wood' + tint, () => std({ map: tex('wood'), color: tint, roughness: 0.6, metalness: 0 })),
    fabric: (color) => mat('fabric' + color, () => { const m = std({ map: tex('fabric'), color, roughness: 0.92, metalness: 0 }); return m; }),
    metal: () => mat('metal', () => std({ map: tex('metal'), roughness: 0.38, metalness: 0.85 })),
    plastic: (color) => mat('plastic' + color, () => std({ color, roughness: 0.5, metalness: 0.05 })),
    whiteboard: () => mat('whiteboard', () => std({ map: tex('whiteboard'), roughness: 0.25, metalness: 0 })),
    blackboard: () => mat('blackboard', () => std({ map: tex('blackboard'), roughness: 0.85, metalness: 0 })),
    paper: () => mat('paper', () => std({ color: 0xf4f1e8, roughness: 0.9, metalness: 0 })),
    screen: () => mat('screen', () => std({ color: 0x15181c, roughness: 0.2, metalness: 0.3 })),
    generic: (color) => mat('generic' + color, () => std({ map: tex('plaster'), color, roughness: 0.75, metalness: 0.02 })),
  };
  return {
    M, tex,
    dispose() { for (const t of cache.values()) t.dispose(); for (const m of mats.values()) m.dispose(); cache.clear(); mats.clear(); },
    count: () => ({ textures: cache.size, materials: mats.size }),
  };
}
