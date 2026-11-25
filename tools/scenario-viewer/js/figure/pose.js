// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Chooses the source of a person's pose at a tick and returns the figure to draw + the label the UI must show.
// Priority: joints channel (MotionGen etc.)  >  display IK from exported anchors  >  standing figure at the root.
// With no stature (none stated, none derivable, none entered) there is no figure: a capsule marker is returned instead.
import { statureFor } from '../model/display.js';
import { solveFigure, jointMap, decorate } from './figure.js';
import { SEG } from './anthro.js';

export const IK_LABEL = 'display-derived (IK), not model output';
export const UNKNOWN_LABEL = 'stature unknown - enter a value';
// Capsule marker size when the case exports no body capsule: a marker, NOT a body size (display-only, ASSETS.md)
export const MARKER_CAPSULE = { r: 0.2, H: 0.5 };

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

function ikLabel(anchors, hasRoot) {
  const names = Object.keys(anchors).filter((n) => anchors[n].pose);
  if (names.length) return { text: IK_LABEL, detail: `from exported anchors: ${names.join(', ')}`, kind: 'display-ik' };
  return { text: 'display figure (standing at the root; no joints, no anchors exported)', detail: hasRoot ? 'not model output' : '', kind: 'root' };
}

/** a = actor state from scene.at(): {pose, yaw, walk, body, skeleton, joints, jointsMeta, anchors, look, statureBase}. */
export function figureFor(a, opts = {}) {
  if (a.joints && a.jointsMeta) {
    const J = jointMap(a.skeleton, a.joints);
    if (J) {
      const leg = (dist(J.left_hip, J.left_knee) + dist(J.left_knee, J.left_ankle) + dist(J.right_hip, J.right_knee) + dist(J.right_knee, J.right_ankle)) / 2;
      const H = leg / (SEG.thigh + SEG.shank);
      const stature = { value: H, origin: 'derived from the joints: mean hip-knee-ankle length / 0.491 (Winter Fig. 4.1)', kind: 'derived' };
      return { kind: 'joints', J, ...decorate(J, H), H, stature, label: a.jointsMeta, residues: {}, posture: 'as exported' };
    }
    return { kind: 'generic', H: 0, stature: { value: null, origin: 'n/a', kind: 'n/a' }, label: a.jointsMeta, residues: {}, posture: 'skeleton drawn as bones (not the 22-joint SMPL-family names)' };
  }
  const st = statureFor(a);
  const anchors = a.anchors || {};
  const pel = anchors.pelvis?.pose ? { p: anchors.pelvis.pose.p, seated: anchors.pelvis.engaged === true } : null;
  const hands = { left: anchors.hand_l?.pose?.p || null, right: anchors.hand_r?.pose?.p || null };
  if (!a.pose && !pel) return null;
  if (!(st.value > 0)) {
    const exported = a.body?.capsule_radius_m > 0 && a.body?.capsule_height_m > 0;
    const cap = exported ? { r: a.body.capsule_radius_m, H: a.body.capsule_height_m, exported: true } : { ...MARKER_CAPSULE, exported: false };
    return { kind: 'capsule', capsule: cap, root: a.pose ? a.pose.p : [pel.p[0], pel.p[1], 0], yaw: a.pose ? a.yaw : 0, H: 0, stature: st,
      label: { text: UNKNOWN_LABEL, detail: exported ? 'capsule = exported body capsule' : 'capsule marker, display-only size (not a body)', kind: 'unknown' }, residues: {}, posture: 'capsule marker' };
  }
  const H = st.value;
  const root = a.pose ? a.pose.p : [pel.p[0], pel.p[1], 0];
  const yaw = a.pose ? a.yaw : 0;
  const fig = solveFigure({ root, yaw, H, pelvis: pel, hands, walk: opts.noWalk ? null : a.walk, sitbone: a.sitbone?.offset || null, sitboneOrigin: a.sitbone?.origin });
  return { kind: 'ik', ...fig, J: fig.joints, H, stature: st, label: opts.label || ikLabel(anchors, !!a.pose), forward: [Math.cos(yaw), Math.sin(yaw), 0] };
}

/** A person element (static). */
export function staticFigure(person, stature) {
  const root = [person.pos[0], person.pos[1], person.pos[2] || 0], yaw = person.yaw || 0;
  if (!(stature.value > 0)) return { kind: 'capsule', capsule: { ...MARKER_CAPSULE, exported: false }, root, yaw, H: 0, stature, label: { text: UNKNOWN_LABEL, detail: 'capsule marker, display-only size', kind: 'unknown' }, residues: {}, posture: 'capsule marker' };
  const H = stature.value;
  const fig = solveFigure({ root, yaw, H, pelvis: null, hands: {}, walk: null });
  return { kind: 'ik', ...fig, J: fig.joints, H, stature, label: { text: 'display figure (static person element)', detail: 'not model output', kind: 'root' }, forward: [Math.cos(yaw), Math.sin(yaw), 0] };
}
