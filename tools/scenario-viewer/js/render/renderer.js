// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Renderer interface (both renderers implement exactly this; the stage talks only to it).
//
//   mount(el)            attach to a container element (renderer owns its canvas + pointer/camera input)
//   setScene(scene)      scene = buildScene(caseJson) from model/scene.js; resets the camera to fit
//   setTime(tick)        tick index (number; fractional ticks only matter for channels marked continuous)
//   setLayers(set)       Set<string>: overlay ids (see LAYERS) and 'space:<id>' for each visible space
//   pick(x, y)           x,y in CSS px relative to the mounted element -> entity | null
//                        entity = {space, kind:'element'|'actor'|'proposal'|'relation', id?, index?}
//   setCamera(cam)       {lock?:bool, fit?:true, reset?:true, zoomBy?:number}; 2D also {scale, pan:[x,y]}
//   dispose()            release listeners/GL resources
// The 3D renderer is created with `await Perspective3D.create(el)` (three.js is loaded lazily); it also takes setCompare({on}).
// Optional (both implement): setHighlight(entity|null)  correspondence hover highlight
//                            getInfo() -> {kind, panes:[{space,x,y,w,h}], skeletonDrawn?:number, ...}

export const LAYERS = [
  ['grid', 'Grid'], ['free', 'Free space F'], ['footprints', 'Footprints'],
  ['scope', 'Scope'], ['relations', 'Relations'], ['actors', 'Actors'], ['skeleton', 'Skeleton'],
  ['proposal', 'Proposal'], ['parity', 'Parity scope (dashed)'], ['extras', 'Extra poses + trails'],
  ['gaze', 'Gaze wedge + angle'], ['steps', 'Footprint marks (derived)', false], ['trail.source', 'Trail: source'], ['trail.body', 'Trail: body'], ['trail.proposal', 'Trail: proposal'],
  ['walls', 'Walls (3D)'], ['parts', 'Affordance parts (3D)'], ['anchors', 'Anchors (3D)'], ['bones3d', 'Bones over the 3D figure', false],
  ['hud', 'Status chips'],
];
export const DEFAULT_LAYERS = new Set(LAYERS.filter((l) => l[2] !== false).map((l) => l[0]));

export function visibleSpaces(scene, layers) {
  const flagged = scene.spaceIds.filter((id) => layers.has('space:' + id));
  return flagged.length ? flagged : scene.spaceIds;
}

/** Split a W x H box into n panes. `aspect` is the content's width/height: pick the split (side by side
 * or stacked) that gives the larger fitting scale. */
export function layoutPanes(n, W, H, gap = 6, aspect = 1.25) {
  if (n <= 0) return [];
  if (n === 1) return [{ x: 0, y: 0, w: W, h: H }];
  const fit = (w, h) => Math.min(w / aspect, h);
  const horizontal = fit((W - gap * (n - 1)) / n, H) >= fit(W, (H - gap * (n - 1)) / n);
  const out = [];
  for (let i = 0; i < n; i++) {
    out.push(horizontal
      ? { x: Math.round((i * (W + gap)) / n), y: 0, w: Math.floor((W - gap * (n - 1)) / n), h: H }
      : { x: 0, y: Math.round((i * (H + gap)) / n), w: W, h: Math.floor((H - gap * (n - 1)) / n) });
  }
  return out;
}

/** Colours for extra pose channels (hollow markers + trails), by channel index: theme keys and CSS vars. */
export const EXTRA_KEYS = ['s2', 's3', 's4', 's1', 'cAfford', 'cFacing', 'cSpacing', 'warn'];
export const EXTRA_VARS = ['--s2', '--s3', '--s4', '--s1', '--c-afford', '--c-facing', '--c-spacing', '--warn'];
export const extraKey = (i) => EXTRA_KEYS[((i % 8) + 8) % 8];
