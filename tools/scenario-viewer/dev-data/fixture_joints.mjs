// Helper of make_fixtures_affordance.py: turns frames into 22-joint positions with the viewer's own figure solver
// (js/figure/figure.js). Used ONLY to author SYNTHETIC fixtures. stdin: JSON [{root,yaw,pelvis,hands,walk,H}] -> stdout: JSON [[x,y,z]*22 per frame]
import { solveFigure, SMPL22 } from '../js/figure/figure.js';
let buf = '';
process.stdin.on('data', (d) => { buf += d; });
process.stdin.on('end', () => {
  const frames = JSON.parse(buf);
  const out = frames.map((f) => {
    const r = solveFigure({ root: f.root, yaw: f.yaw, H: f.H, pelvis: f.pelvis, hands: f.hands || {}, walk: f.walk });
    return SMPL22.map((n) => r.joints[n].map((v) => Math.round(v * 1000) / 1000));
  });
  process.stdout.write(JSON.stringify(out));
});
