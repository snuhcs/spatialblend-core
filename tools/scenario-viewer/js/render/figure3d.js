// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Shaded humanoid from a figure ({J: joint name -> [x,y,z], handEnd, forward, H}) made by js/figure/figure.js.
// Shape only: capsule-like tapered limbs, ellipsoid torso and head, box feet. Thickness is a display choice
// (anthro.js DISPLAY_SHAPE); segment LENGTHS come from the joints themselves.
import { SEG, DISPLAY_SHAPE as S } from '../figure/anthro.js';

const LIMBS = [ // [from, to, radius at from (x H), radius at to (x H)]
  ['left_hip', 'left_knee', 0.046, 0.034], ['right_hip', 'right_knee', 0.046, 0.034],
  ['left_knee', 'left_ankle', 0.034, 0.022], ['right_knee', 'right_ankle', 0.034, 0.022],
  ['left_shoulder', 'left_elbow', 0.03, 0.024], ['right_shoulder', 'right_elbow', 0.03, 0.024],
  ['left_elbow', 'left_wrist', 0.024, 0.018], ['right_elbow', 'right_wrist', 0.024, 0.018],
  ['neck', 'head', 0.03, 0.026],
  ['left_collar', 'left_shoulder', 0.032, 0.03], ['right_collar', 'right_shoulder', 0.032, 0.03],
];
const BALLS = [['left_knee', 0.034], ['right_knee', 0.034], ['left_elbow', 0.024], ['right_elbow', 0.024], ['left_shoulder', 0.032], ['right_shoulder', 0.032], ['left_wrist', 0.018], ['right_wrist', 0.018]];

export class FigureMesh {
  /** shared: {THREE, geo: {sphere, tcyl, box}} */
  constructor(ctx, color, opts = {}) {
    const T = ctx.THREE; this.T = T; this.ctx = ctx;
    this.group = new T.Group(); this.group.userData.layer = 'actors';
    this.mat = new T.MeshStandardMaterial({ color, roughness: 0.62, metalness: 0.02, transparent: !!opts.ghost, opacity: opts.ghost ? 0.32 : 1, depthWrite: !opts.ghost });
    this.skin = new T.MeshStandardMaterial({ color: opts.ghost ? color : 0xd9b08c, roughness: 0.7, metalness: 0, transparent: !!opts.ghost, opacity: opts.ghost ? 0.32 : 1, depthWrite: !opts.ghost });
    const g = ctx.fig;
    const mk = (geo, mat) => { const m = new T.Mesh(geo, mat); m.castShadow = !opts.ghost; m.receiveShadow = !opts.ghost; this.group.add(m); return m; };
    this.limbs = LIMBS.map(() => mk(g.tcyl, this.mat));
    this.balls = BALLS.map(() => mk(g.sphere, this.mat));
    this.hands = [mk(g.tcyl, this.skin), mk(g.tcyl, this.skin)];
    this.handBalls = [mk(g.sphere, this.skin), mk(g.sphere, this.skin)];
    this.feet = [mk(g.box, this.mat), mk(g.box, this.mat)];
    this.torso = mk(g.sphere, this.mat); this.pelvisM = mk(g.sphere, this.mat); this.head = mk(g.sphere, this.skin); this.shoulderBar = mk(g.tcyl, this.mat);
    this.hipBar = mk(g.tcyl, this.mat);
    this.group.traverse((o) => { o.frustumCulled = false; });
    this.up = new T.Vector3(0, 0, 1); this.Y = new T.Vector3(0, 1, 0);
  }

  _seg(m, a, b, r0, r1) {
    const T = this.T, A = new T.Vector3(...a), B = new T.Vector3(...b), d = B.clone().sub(A), L = d.length();
    m.visible = L > 1e-6;
    if (!m.visible) return;
    m.position.copy(A).add(B).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(this.Y, d.multiplyScalar(1 / L));
    m.scale.set(r0, L, r0); // tcyl: radiusTop 0.8 x bottom, unit height; r1 is carried by the taper ratio
    m.userData.r1 = r1;
  }

  /** fig: from figureFor(); returns nothing. Hides the figure when fig is null. */
  update(fig) {
    const T = this.T;
    this.updateCapsule(fig);
    const body = !!fig && fig.kind !== 'capsule';
    for (const c of this.group.children) if (c !== this.cap) c.visible = body;
    this.group.visible = !!fig;
    if (!body) return;
    const J = fig.J, H = fig.H, V = (n) => J[n];
    const f = new T.Vector3(...(fig.forward || [1, 0, 0])).setZ(0).normalize();
    const left = new T.Vector3(-f.y, f.x, 0);
    // limbs. For the taper we draw from the thicker end: scale radius = r0 and the geometry narrows to 0.8 r0.
    LIMBS.forEach(([a, b, r0, r1], i) => this._seg(this.limbs[i], V(a), V(b), r0 * H, r1 * H));
    BALLS.forEach(([n, r], i) => { this.balls[i].position.set(...V(n)); this.balls[i].scale.setScalar(r * H); });
    for (const [i, side] of [[0, 'left'], [1, 'right']]) {
      this._seg(this.hands[i], V(side + '_wrist'), fig.handEnd[side], 0.02 * H, 0.015 * H);
      this.handBalls[i].position.set(...fig.handEnd[side]); this.handBalls[i].scale.setScalar(0.017 * H);
      const ank = V(side + '_ankle'), fl = SEG.footLength * H, ft = SEG.ankle * H;
      const fb = this.feet[i]; fb.scale.set(fl, SEG.footBreadth * H, ft);
      const fw = new T.Vector3(...(fig.footDir?.[side] || [f.x, f.y, 0])).setZ(0).normalize();
      fb.position.set(ank[0] + fw.x * fl * 0.35, ank[1] + fw.y * fl * 0.35, ank[2] - ft / 2);
      fb.rotation.set(0, 0, Math.atan2(fw.y, fw.x));
    }
    // torso: an ellipsoid from the pelvis to the shoulder line, oriented by the body frame
    const P = new T.Vector3(...V('pelvis')), N = new T.Vector3(...V('neck')), mid = P.clone().add(N).multiplyScalar(0.5), len = N.distanceTo(P);
    const zax = N.clone().sub(P).normalize(), fo = f.clone().addScaledVector(zax, -f.dot(zax)).normalize(), lo = new T.Vector3().crossVectors(zax, fo);
    const basis = new T.Matrix4().makeBasis(fo, lo, zax);
    const q = new T.Quaternion().setFromRotationMatrix(basis);
    this.torso.position.copy(mid).addScaledVector(this.up, len * 0.04); this.torso.quaternion.copy(q);
    this.torso.scale.set(S.trunkHalfDepth * H, S.trunkHalfWidth * H, len * 0.6);
    this.pelvisM.position.copy(P).addScaledVector(this.up, 0.0); this.pelvisM.quaternion.copy(q); this.pelvisM.scale.set(S.trunkHalfDepth * H * 1.05, SEG.hipHalf * H * 1.1, SEG.hipHalf * H * 0.8);
    // head: ellipsoid centred at eye height along the neck -> head direction
    const hd = new T.Vector3(...V('head')), nd = hd.clone().sub(N).normalize();
    const hc = hd.clone().addScaledVector(nd, (SEG.eye - SEG.chin) * H * 0.9 + 0.0);
    this.head.position.copy(hc); this.head.quaternion.copy(q);
    this.head.scale.set(S.headRx * H * 0.95, S.headRx * H * 0.85, SEG.head * H * 0.5);
    // shoulder + hip bars (thin cylinders across)
    this._seg(this.shoulderBar, V('left_shoulder'), V('right_shoulder'), 0.03 * H, 0.03 * H);
    this._seg(this.hipBar, V('left_hip'), V('right_hip'), 0.04 * H, 0.04 * H);
  }

  /** capsule marker (no stature known): a translucent capsule at the root, size from the exported body capsule or a display-only marker */
  updateCapsule(fig) {
    const T = this.T;
    if (!this.cap) {
      this.capMat = new T.MeshStandardMaterial({ color: this.mat.color, roughness: 0.7, transparent: true, opacity: 0.55 });
      this.cap = new T.Mesh(new T.CapsuleGeometry(1, 1, 4, 14), this.capMat); this.cap.castShadow = true; this.group.add(this.cap);
    }
    this.cap.visible = !!fig && fig.kind === 'capsule';
    if (!this.cap.visible) return;
    const { r, H } = fig.capsule, g = this.cap.geometry;
    if (g.userData.r !== r || g.userData.H !== H) { g.dispose(); this.cap.geometry = new T.CapsuleGeometry(r, Math.max(0.01, H - 2 * r), 4, 14); this.cap.geometry.rotateX(Math.PI / 2); this.cap.geometry.userData = { r, H }; }
    this.cap.position.set(fig.root[0], fig.root[1], fig.root[2] + H / 2); this.cap.rotation.set(0, 0, fig.yaw);
  }

  dispose() { this.mat.dispose(); this.skin.dispose(); this.capMat?.dispose(); this.cap?.geometry.dispose(); }
}

export function makeFigureGeos(T) {
  const sphere = new T.SphereGeometry(1, 16, 12); sphere.userData.shared = true;
  const tcyl = new T.CylinderGeometry(0.8, 1, 1, 14, 1); tcyl.userData.shared = true; // narrow end is +y (distal)
  const box = new T.BoxGeometry(1, 1, 1); box.userData.shared = true;
  return { sphere, tcyl, box, dispose() { sphere.dispose(); tcyl.dispose(); box.dispose(); } };
}
