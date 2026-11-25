// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Display figure: joint positions of a person for the 3D view. Pure math (no three.js, no DOM), z-up, metres.
// Two sources, never mixed in one tick:
//   * a `joints` channel (MotionGen sidecar, recorded motion ...): the exported joint positions are used as they are;
//   * display IK: a standing / seated figure placed from the root pose and the exported pose3d anchors
//     (source.anchor.<name>, receiver.anchor.<name>.proposal). That is DISPLAY-DERIVED, not model output.
import { SEG } from './anthro.js';

export const SMPL22 = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3',
  'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'];
export const SMPL22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19];

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => { const l = len(a); return l > 1e-9 ? mul(a, 1 / l) : [0, 0, 0]; };
const UP = [0, 0, 1];

/** Two-bone IK. root R, target T, lengths a (R->mid) and b (mid->end), `pole` = direction the mid joint bends toward.
 * Returns {mid, end, reach} where reach is the shortfall in metres (0 = target reached). */
export function twoBone(R, T, a, b, pole) {
  const v = sub(T, R);
  const dist = len(v);
  const maxD = a + b - 1e-6, minD = Math.abs(a - b) + 1e-6;
  const d = Math.min(maxD, Math.max(minD, dist));
  const u = dist > 1e-9 ? mul(v, 1 / dist) : [0, 0, -1];
  const along = (a * a - b * b + d * d) / (2 * d);
  const hh = Math.sqrt(Math.max(0, a * a - along * along));
  let p = sub(pole, mul(u, dot(pole, u)));
  if (len(p) < 1e-6) p = sub([1, 0, 0], mul(u, u[0])); // pole parallel to the limb: any perpendicular
  p = norm(p);
  const mid = add(add(R, mul(u, along)), mul(p, hh));
  const end = add(R, mul(u, d));
  return { mid, end, reach: Math.max(0, dist - maxD) };
}

/**
 * Display IK figure.
 *  in: {root:[x,y,z] feet point, yaw, H stature, pelvis:{p,seated}|null, hands:{left,right} (hand-end world points or null),
 *       walk:{phase,amp}|null}
 *  out: {joints:{name:[x,y,z]}, handEnd:{left,right}, residues:{name:mm}, posture, mode:'display-ik'}
 */
export function solveFigure(inp) {
  const H = inp.H, f = [Math.cos(inp.yaw), Math.sin(inp.yaw), 0], l = [-f[1], f[0], 0];
  const root = inp.root;
  const hasPelvis = !!inp.pelvis?.p;
  const seated = !!(hasPelvis && inp.pelvis.seated);
  // seated on the pelvis anchor (the sit-bone contact): the display pelvis joint is the anchor minus the measured offset (anthro.js)
  const SB = inp.sitbone || null;   // {forward, left, up} registered offset of the pelvis anchor from the pelvis joint (embodiment); null = none registered
  const pelvis = seated && SB ? sub(sub(sub(inp.pelvis.p, mul(f, SB.forward)), mul(l, SB.left)), [0, 0, SB.up])
    : seated ? inp.pelvis.p
    : hasPelvis ? inp.pelvis.p : add(root, [0, 0, SEG.hip * H]);
  const J = {}, residues = {};
  J.pelvis = pelvis;
  J.left_hip = add(pelvis, mul(l, SEG.hipHalf * H));
  J.right_hip = add(pelvis, mul(l, -SEG.hipHalf * H));
  const trunk = (SEG.shoulder - SEG.hip) * H, sh = SEG.shoulderHalf * H;
  const ua = SEG.upperArm * H, fa = SEG.forearm * H, hl = SEG.hand * H;
  const wk = inp.walk && !hasPelvis ? inp.walk : null;
  const sw = wk ? wk.amp * Math.sin(2 * Math.PI * wk.phase) : 0;
  const targets = [inp.hands?.left, inp.hands?.right].filter(Boolean);

  // upper body for a torso lean `th` (radians, forward toward direction dir): spine, neck, head, shoulders, arms.
  const upper = (th, dir) => {
    const U = {}, up = add(mul(UP, Math.cos(th)), mul(dir, Math.sin(th)));
    U.spine1 = add(pelvis, mul(up, trunk * 0.25)); U.spine2 = add(pelvis, mul(up, trunk * 0.5)); U.spine3 = add(pelvis, mul(up, trunk * 0.75));
    U.neck = add(pelvis, mul(up, trunk));
    U.left_shoulder = add(U.neck, mul(l, sh)); U.right_shoulder = add(U.neck, mul(l, -sh));
    U.left_collar = add(U.neck, mul(l, sh * 0.5)); U.right_collar = add(U.neck, mul(l, -sh * 0.5));
    U.head = add(U.neck, mul(up, (SEG.chin - SEG.shoulder) * H));
    const hand = {}, res = {}; let miss = 0;
    for (const [side, sgn] of [['left', 1], ['right', -1]]) {
      const S = U[side + '_shoulder'], T = inp.hands?.[side] || null;
      if (T) {
        // the exported anchor is the hand point (fingertip / contact): the wrist sits one hand length before it, along the forearm
        const pole = [-f[0] * 0.3, -f[1] * 0.3, -1];
        let arm = twoBone(S, T, ua, fa, pole);
        for (let i = 0; i < 4; i++) { const d = norm(sub(T, arm.mid)); arm = twoBone(S, sub(T, mul(d, hl)), ua, fa, pole); }
        const d = norm(sub(arm.end, arm.mid)), tip = add(arm.end, mul(d, hl));
        U[side + '_elbow'] = arm.mid; U[side + '_wrist'] = arm.end; hand[side] = tip;
        const m = len(sub(tip, T)); miss += m;
        if (m > 1e-3) res['hand_' + (side === 'left' ? 'l' : 'r')] = Math.round(m * 1000);
      } else {
        const swing = wk ? -sgn * sw : 0;
        const arm = twoBone(S, add(add(S, [0, 0, -(ua + fa)]), mul(f, swing)), ua, fa, f);
        U[side + '_elbow'] = arm.mid; U[side + '_wrist'] = arm.end;
        hand[side] = add(arm.end, mul(norm(sub(arm.end, arm.mid)), hl));
      }
    }
    return { U, hand, res, miss };
  };
  // Torso lean (display IK): when a hand target is out of reach, lean the trunk from the pelvis toward the targets, by the smallest angle
  // that reaches them (at most 90 degrees, i.e. horizontal); the residue is reported when even that falls short.
  let best = upper(0, f), lean = 0;
  if (targets.length && best.miss > 2e-3) {
    const c = targets.reduce((a, t) => add(a, t), [0, 0, 0]).map((v) => v / targets.length);
    const dir = norm([c[0] - pelvis[0], c[1] - pelvis[1], 0]);
    if (len(dir) > 0.5) {
      let cand = best;
      for (let deg = 2.5; deg <= 90; deg += 2.5) {
        const r = upper((deg * Math.PI) / 180, dir);
        if (r.miss < cand.miss - 1e-6) { cand = r; lean = deg; }
        if (r.miss <= 2e-3) break;
      }
      best = cand;
    }
  }
  Object.assign(J, best.U); Object.assign(residues, best.res);
  const floorZ = root[2];
  for (const [side, sgn] of [['left', 1], ['right', -1]]) {
    const hip = J[side + '_hip'];
    let ank;
    if (seated) ank = add(hip, mul(f, SEG.thigh * H));
    else ank = add(hip, mul(f, sgn * sw));
    ank = [ank[0], ank[1], floorZ + SEG.ankle * H];
    const leg = twoBone(hip, ank, SEG.thigh * H, SEG.shank * H, f);
    J[side + '_knee'] = leg.mid; J[side + '_ankle'] = leg.end;
    J[side + '_foot'] = add(leg.end, mul(f, SEG.footLength * H));
    if (leg.reach > 1e-4 && !wk) residues[side + '_foot'] = Math.round(leg.reach * 1000);
  }
  const posture = (seated ? (SB ? `seated (pelvis anchor engaged; pelvis joint = anchor - R(yaw)*(${SB.forward}, ${SB.left}, ${SB.up}) m: ${inp.sitboneOrigin || 'registered offset'})` : 'seated (pelvis anchor engaged; no offset registered: pelvis joint at the anchor)') : hasPelvis ? 'standing, pelvis at anchor' : 'standing at the root') + (lean ? `, torso lean ${lean} deg toward the hand targets` : '');
  return { joints: J, handEnd: best.hand, residues, posture, lean, mode: 'display-ik', yaw: inp.yaw };
}

/** Joint map from a joints-channel frame (array of {p}|null, in skeleton order). Returns null unless every SMPL22 name is present. */
export function jointMap(skeleton, frame) {
  const names = (skeleton?.joints || []).map((j) => (typeof j === 'string' ? j : j?.name));
  if (!frame || SMPL22.some((n) => !names.includes(n))) return null;
  const out = {};
  for (const n of SMPL22) { const j = frame[names.indexOf(n)]; if (!j?.p) return null; out[n] = j.p; }
  return out;
}

/** Hand-end points and body forward for a joint map (hand length from the displayed stature). */
export function decorate(J, H) {
  const handEnd = {};
  for (const side of ['left', 'right']) handEnd[side] = add(J[side + '_wrist'], mul(norm(sub(J[side + '_wrist'], J[side + '_elbow'])), SEG.hand * H));
  const lat = sub(J.left_hip, J.right_hip);
  const fwd = norm([-lat[1], lat[0], 0]).map((v) => -v); // up x (right - left) = forward
  return { handEnd, forward: fwd };
}
