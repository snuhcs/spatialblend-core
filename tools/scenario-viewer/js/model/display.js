// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Display-only parameters shared by the renderer, the HUD and the inspector (never written to a case).
export const DISPLAY = { stature: null };   // user override in metres (null = use the case)

/** Stature for drawing an IK figure: {value|null, origin, kind}. Order: set in the viewer > stated or derived by the case model
 * (`actor.statureBase`, see scene.js) > unknown (no default: the viewer draws a capsule marker and asks for a value). */
export function statureFor(actor) {
  if (DISPLAY.stature > 0) return { value: DISPLAY.stature, origin: 'set in the viewer (display parameter)', kind: 'user' };
  const b = actor?.statureBase;
  if (b?.value > 0) return b;
  return { value: null, origin: 'stature unknown - enter a value', kind: 'unknown' };
}
