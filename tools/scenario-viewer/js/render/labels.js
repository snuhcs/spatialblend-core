// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Greedy label placer for the 2D stage: several candidate positions per label, leader lines when a label
// sits away from its anchor, priorities, and the lowest-priority label is DROPPED when nothing fits.
// Obstacles (glyphs, pane chrome) and already placed labels are never overlapped.
import { withAlpha } from '../util.js';

const PAD = 2;
const hit = (a, b) => a.x < b.x + b.w + PAD && a.x + a.w + PAD > b.x && a.y < b.y + b.h + PAD && a.y + a.h + PAD > b.y;

export class LabelPlacer {
  constructor(pane) { this.pane = pane; this.obst = []; this.items = []; this.placed = []; this.dropped = []; }
  block(x, y, w, h) { this.obst.push({ x, y, w, h }); }
  blockCircle(cx, cy, r) { this.block(cx - r, cy - r, 2 * r, 2 * r); }
  /** it: {text, ax, ay, prio, color, font, r (anchor radius), cands?: (w,h)=>[[cx,cy],...] absolute centres, line?:[x0,y0,x1,y1]} */
  add(it) { this.items.push(it); }

  static pointCands(ax, ay, r, w, h) {
    const out = [];
    for (const k of [0, 1, 2]) {
      const d = r + 4 + k * 14;
      out.push([ax + d + w / 2, ay], [ax - d - w / 2, ay], [ax, ay - d - h / 2], [ax, ay + d + h / 2],
        [ax + d * 0.7 + w / 2, ay - d * 0.7 - h / 2], [ax - d * 0.7 - w / 2, ay - d * 0.7 - h / 2],
        [ax + d * 0.7 + w / 2, ay + d * 0.7 + h / 2], [ax - d * 0.7 - w / 2, ay + d * 0.7 + h / 2]);
    }
    return out;
  }
  /** perpendicular offsets from the midpoint (and nearby points) of a line, clear of the line itself */
  static lineCands(x0, y0, x1, y1, w, h) {
    const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy) || 1, nx = -dy / L, ny = dx / L;
    const base = Math.abs(nx) * w / 2 + Math.abs(ny) * h / 2 + 4;
    const out = [];
    for (const extra of [0, 12, 26]) for (const t of [0.5, 0.3, 0.7]) for (const s of [1, -1]) {
      const mx = x0 + dx * t, my = y0 + dy * t;
      out.push([mx + s * nx * (base + extra), my + s * ny * (base + extra)]);
    }
    return out;
  }

  resolve(ctx, th) {
    const P = this.pane, inside = (r) => r.x >= P.x + 2 && r.y >= P.y + 2 && r.x + r.w <= P.x + P.w - 2 && r.y + r.h <= P.y + P.h - 2;
    const order = [...this.items].sort((a, b) => b.prio - a.prio);
    for (const it of order) {
      ctx.font = it.font || `10px ${th.fontMono}`;
      const w = Math.ceil(ctx.measureText(it.text).width) + 8, h = 15;
      const cands = it.cands ? it.cands(w, h) : LabelPlacer.pointCands(it.ax, it.ay, it.r ?? 8, w, h);
      if (it.inside) cands.unshift([it.ax, it.ay]);
      let ok = null;
      for (const [cx, cy] of cands) {
        const r = { x: Math.round(cx - w / 2), y: Math.round(cy - h / 2), w, h };
        if (!inside(r)) continue;
        if (this.obst.some((o) => hit(r, o)) || this.placed.some((o) => hit(r, o.rect))) continue;
        ok = r; break;
      }
      if (!ok) { this.dropped.push(it); continue; }
      this.placed.push({ rect: ok, it });
    }
    for (const { rect: r, it } of this.placed) {
      // leader line when the label is not next to its anchor
      const nx = Math.max(r.x, Math.min(it.ax, r.x + r.w)), ny = Math.max(r.y, Math.min(it.ay, r.y + r.h));
      if (!it.inside && Math.hypot(nx - it.ax, ny - it.ay) > (it.r ?? 8) + 12) {
        ctx.beginPath(); ctx.moveTo(it.ax, it.ay); ctx.lineTo(nx, ny); ctx.strokeStyle = withAlpha(it.color, 0.6); ctx.lineWidth = 1; ctx.setLineDash([]); ctx.stroke();
      }
      ctx.fillStyle = withAlpha(th.panel, 0.9); ctx.fillRect(r.x, r.y, r.w, r.h);
      ctx.strokeStyle = withAlpha(it.color, 0.35); ctx.lineWidth = 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
      ctx.font = it.font || `10px ${th.fontMono}`; ctx.fillStyle = it.color; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(it.text, r.x + 4, r.y + r.h / 2 + 0.5);
    }
    return this.placed.map(({ rect, it }) => ({ ...rect, text: it.text }));
  }
}
