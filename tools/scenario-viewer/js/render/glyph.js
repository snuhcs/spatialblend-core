// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Top-down character glyph. Glyph size and the walk animation are DISPLAY conventions, not data.
// Constants (shown in the legend): shoulder width = 2 x declared body radius (else 0.45 m), head diameter
// 0.2 m (else 0.22 x shoulder width), stride 0.7 m, foot swing +-0.25 m, gaze wedge +-30 deg x 1.5 m.
import { withAlpha } from '../util.js';

export const G = { SHOULDER: 0.45, HEAD_D: 0.2, STRIDE: 0.7, SWING: 0.25, YAW_PIVOT: 0.2, WEDGE_DEG: 30, WEDGE_LEN: 1.5, MIN_PX: 28 };

export function glyphDims(body) {
  const r = body?.capsule_radius_m;
  const W = r > 0 ? 2 * r : G.SHOULDER;
  return { W, hr: r > 0 ? W * 0.22 : G.HEAD_D / 2, g: W / G.SHOULDER };
}

/** metres -> pixels scale with the minimum pixel size applied (shoulders never below MIN_PX). */
export const glyphScale = (dims, pxPerM) => pxPerM * Math.max(1, G.MIN_PX / (dims.W * pxPerM));

/** screen position of the head for a glyph at (sx,sy) heading `yaw` (world, CCW) */
export function headPos(sx, sy, yaw, dims, pxPerM) {
  const sc = glyphScale(dims, pxPerM), hx = dims.W * 0.06;
  return [sx + Math.cos(yaw) * hx * sc, sy - Math.sin(yaw) * hx * sc];
}

/**
 * style: {fill:color|null, stroke:color, dash:[px]|null, alpha, feet:bool}
 * walk: {phase, amp} or null (feet together)
 */
export function drawGlyph(ctx, sx, sy, yaw, dims, pxPerM, walk, style) {
  const sc = glyphScale(dims, pxPerM), { W, hr, g } = dims;
  const lw = (style.lw ?? 1.6) / sc;
  ctx.save();
  ctx.translate(sx, sy); ctx.rotate(-yaw); ctx.scale(sc, -sc);
  ctx.lineWidth = lw; ctx.lineJoin = 'round';
  ctx.setLineDash(style.dash ? style.dash.map((d) => d / sc) : []);
  ctx.globalAlpha = style.alpha ?? 1;
  const paint = (fill = style.fill) => { if (fill) { ctx.fillStyle = fill; ctx.fill(); } ctx.strokeStyle = style.stroke; ctx.stroke(); };
  // feet: two small ovals under the shoulders, alternating fore/aft with the (derived) stride phase
  const off = walk ? walk.amp * g * Math.sin(2 * Math.PI * walk.phase) : 0;
  for (const side of [1, -1]) {
    ctx.beginPath(); ctx.ellipse(side * off, side * W * 0.25, 0.09 * g, 0.045 * g, 0, 0, 7); paint(style.footFill ?? style.fill);
  }
  // shoulders: rounded ellipse across the heading
  ctx.beginPath(); ctx.ellipse(0, 0, W * 0.24, W * 0.5, 0, 0, 7); paint();
  // head slightly forward, with a visor notch along the yaw
  const hx = W * 0.06;
  ctx.beginPath(); ctx.arc(hx, 0, hr, 0, 7); paint(style.headFill ?? style.fill);
  ctx.beginPath(); ctx.moveTo(hx + hr * 0.55, hr * 0.55); ctx.lineTo(hx + hr * 1.5, 0); ctx.lineTo(hx + hr * 0.55, -hr * 0.55); ctx.closePath();
  if (style.fill) { ctx.fillStyle = style.stroke; ctx.fill(); } ctx.strokeStyle = style.stroke; ctx.stroke();
  ctx.restore();
}

/** view wedge from the head along the yaw, fading out (display constants) */
export function drawWedge(ctx, hx, hy, yaw, pxPerM, color) {
  const L = Math.max(G.WEDGE_LEN * pxPerM, 30), a = (G.WEDGE_DEG * Math.PI) / 180;
  const gr = ctx.createRadialGradient(hx, hy, 0, hx, hy, L);
  gr.addColorStop(0, withAlpha(color, 0.42)); gr.addColorStop(1, withAlpha(color, 0));
  ctx.save(); ctx.beginPath(); ctx.moveTo(hx, hy); ctx.arc(hx, hy, L, -(yaw + a), -(yaw - a)); ctx.closePath(); ctx.fillStyle = gr; ctx.fill(); ctx.restore();
}

/** small footprint oval (derived stride mark) */
export function drawFoot(ctx, sx, sy, yaw, dims, pxPerM, side, color) {
  const sc = glyphScale(dims, pxPerM);
  ctx.save(); ctx.translate(sx, sy); ctx.rotate(-yaw); ctx.scale(sc, -sc);
  ctx.beginPath(); ctx.ellipse(0, side * dims.W * 0.25, 0.09 * dims.g, 0.045 * dims.g, 0, 0, 7); ctx.fillStyle = color; ctx.fill(); ctx.restore();
}
