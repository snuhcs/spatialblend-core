// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Time-plot series, built ONLY from channels that exist. Each plot: {id,title,unit,note,series[],bound?,strips?}
// series: {name, key, cls('spacing'|'facing'|'afford'|'neutral'), dash, alpha, values:(number|null)[]}
import { isNum, familyClass } from '../util.js';

const csr = (ch) => {
  const d = ch.data, n = d.n ?? 0;
  const rows = [];
  for (let k = 0; k < n; k++) rows.push([d.offsets?.[k] ?? 0, d.offsets?.[k + 1] ?? 0]);
  return rows;
};

export function buildSeries(scene) {
  const n = scene.n;
  const plots = [];
  const nulls = () => new Array(n).fill(null);
  const rel = new Map(); // key -> {family, referent, R_src, R_rx, R_rx_body, weight, angle}
  const relChannels = scene.channels.filter((c) => c.type === 'relation_set');
  const slot = (fam, ref) => {
    const key = `${fam}#${ref}`;
    if (!rel.has(key)) rel.set(key, { key, family: fam, referent: ref, R_src: null, R_rx: null, R_rx_body: null, weight: null, angle: null });
    return rel.get(key);
  };
  const put = (o, col, k, v) => { if (!isNum(v)) return; if (!o[col]) o[col] = nulls(); o[col][k] = v; };
  for (const ch of relChannels) {
    const d = ch.data;
    const rows = csr(ch);
    rows.forEach(([a, b], k) => {
      for (let i = a; i < b; i++) {
        const o = slot(d.families?.[d.family?.[i]] ?? `#${d.family?.[i]}`, d.referent?.[i]);
        put(o, 'R_src', k, d.R_src?.[i]);
        put(o, 'R_rx', k, d.R_rx?.[i]);
        put(o, 'R_rx_body', k, d.R_rx_body?.[i]);
        if (!o.weight || o.weight[k] === null) put(o, 'weight', k, d.weight?.[i]);
        put(o, 'angle', k, isNum(d.angle_err_rad?.[i]) ? (d.angle_err_rad[i] * 180) / Math.PI : null);
      }
    });
  }
  const keys = [...rel.values()].slice(0, 8);
  const alphaFor = (i) => [1, 0.8, 0.62, 0.5, 0.4, 0.34, 0.3, 0.26][i] ?? 0.25;
  const relPlot = (id, title, unit, cols, note) => {
    const series = [];
    keys.forEach((o, i) => {
      for (const [col, dash, label] of cols) {
        if (o[col]) series.push({ name: `${o.family} #${o.referent} ${label}`, key: o.key, cls: familyClass(o.family), dash, alpha: alphaFor(i), values: o[col] });
      }
    });
    if (series.length) plots.push({ id, title, unit, note: note + (rel.size > keys.length ? ` (first ${keys.length} of ${rel.size} relations)` : ''), series });
  };
  relPlot('R', 'R per relation', '', [['R_src', [], 'src'], ['R_rx', [5, 3], 'rx'], ['R_rx_body', [1.5, 2.5], 'rx@body']], 'source R solid, receiver R dashed, receiver R at body dotted');
  relPlot('weight', 'Weights', '', [['weight', [], 'w']], 'annealed policy weight');

  // solver J and friends
  const scalars = scene.channels.filter((c) => c.type === 'scalar');
  const used = new Set();
  const sc = (c) => ({ values: c.data.v.map((v) => (isNum(v) ? v : null)) });
  const chBounds = (chs) => chs.flatMap((c) => (c.bounds || []).filter((b) => isNum(b.value)).map((b) => ({ value: b.value, label: b.label || String(b.value) })));
  const J = scalars.filter((c) => /(^|\.)J(_|$|\.)|(^|\.)J_before/.test(c.id));
  if (J.length) {
    J.forEach((c) => used.add(c.id));
    plots.push({ id: 'J', title: 'Solver J', unit: J[0].unit || '', note: J.map((c) => c.id).join(', '), bounds: chBounds(J),
      series: J.map((c, i) => ({ name: c.id, key: c.id, cls: 'neutral', color: `s${(i % 4) + 1}`, dash: i ? [5, 3] : [], alpha: 1, ...sc(c) })) });
  }

  // grouped scalars: parity.* and eval.* -> one panel per group, one line per channel, labelled by id
  for (const group of ['parity', 'eval']) {
    const g = scalars.filter((c) => c.id.startsWith(group + '.'));
    if (!g.length) continue;
    g.forEach((c) => used.add(c.id));
    plots.push({ id: group, title: `${group} scalars`, unit: '', note: `${g.length} channel${g.length > 1 ? 's' : ''}`, bounds: chBounds(g),
      series: g.map((c, i) => ({ name: c.id, key: c.id, cls: 'neutral', color: ['s1', 's2', 's3', 's4'][i % 4], dash: [[], [5, 3], [1.5, 2.5], [8, 3, 2, 3]][Math.floor(i / 4) % 4], alpha: 1, ...sc(c) })) });
  }

  // body-proposal distance (derived by the viewer from two exported poses; xy, metres)
  const prop = scene.channels.find((c) => c.type === 'proposal');
  const body = prop && scene.actors.find((a) => a.role === 'receiver_body' && a.rootCh)?.rootCh;
  const haveDist = scalars.some((c) => /body_proposal_dist/.test(c.id));
  if (prop && body && !haveDist) {
    const v = nulls();
    const bp = body.data, pp = prop.data;
    for (let k = 0; k < n; k++) {
      if (bp.valid && bp.valid[k] === 0) continue;
      if (pp.valid && pp.valid[k] === 0) continue;
      const dx = pp.p[3 * k] - bp.p[3 * k], dy = pp.p[3 * k + 1] - bp.p[3 * k + 1];
      if (isNum(dx) && isNum(dy)) v[k] = Math.hypot(dx, dy);
    }
    plots.push({ id: 'dist', title: 'Body to proposal', unit: 'm', note: 'xy distance, derived in the viewer from the two exported poses', series: [{ name: 'distance', key: 'dist', cls: 'neutral', color: 's2', dash: [], alpha: 1, values: v }] });
  }

  // step ratio against its bound
  const sr = scalars.find((c) => /step_ratio/.test(c.id));
  if (sr) {
    used.add(sr.id);
    const cb = chBounds([sr]);
    const g = (scene.json.metrics?.guards || []).find((x) => /step_ratio/.test(x.column) && isNum(x.bound));
    const bounds = cb.length ? cb : [{ value: g ? g.bound : 1.5, label: g ? `bound ${g.bound}` : 'bound 1.5' }];
    plots.push({ id: 'step', title: 'Step ratio', unit: sr.unit || '', note: cb.length ? 'bounds as exported on the channel' : g ? `bound ${g.bound} from the guard row` : 'bound 1.5 (fixed by the verification brief; none exported)',
      bounds,
      series: [{ name: sr.id, key: sr.id, cls: 'neutral', color: 's1', dash: [], alpha: 1, ...sc(sr) }] });
  }

  // yaw error (relations' angle_err_rad, degrees)
  const yawKeys = keys.filter((o) => o.angle);
  if (yawKeys.length) plots.push({ id: 'yaw', title: 'Yaw / angle error', unit: 'deg', note: 'angle_err_rad of relations, shown in degrees',
    series: yawKeys.map((o, i) => ({ name: `${o.family} #${o.referent}`, key: o.key, cls: familyClass(o.family), dash: [], alpha: alphaFor(i), values: o.angle })) });

  // proposal error columns
  if (prop && prop.data.err) {
    const cols = Object.entries(prop.data.err).filter(([, a]) => Array.isArray(a) && a.some(isNum));
    if (cols.length) plots.push({ id: 'err', title: 'Proposal error', unit: '', note: 'receiver.proposal err columns (R units; scale_m in metres)',
      series: cols.map(([name, a], i) => ({ name, key: name, cls: familyClass(name), color: familyClass(name) === 'other' ? `s${(i % 4) + 1}` : undefined, dash: [], alpha: 1, values: a.map((v) => (isNum(v) ? v : null)) })) });
  }

  // vector channels: one plot, a series per component
  for (const c of scene.channels.filter((x) => x.type === 'vector')) {
    const { dim, v, names } = c.data;
    const series = [];
    for (let j = 0; j < dim; j++) series.push({ name: `${c.id}.${names?.[j] ?? j}`, key: `${c.id}.${j}`, cls: 'neutral', color: `s${(j % 4) + 1}`, dash: [], alpha: 1, values: Array.from({ length: n }, (_, k) => (isNum(v[k * dim + j]) ? v[k * dim + j] : null)) });
    plots.push({ id: c.id, title: c.id, unit: c.unit || '', note: c.doc || 'vector channel', series });
  }

  // remaining scalars (iters, evals, ...)
  for (const c of scalars) if (!used.has(c.id)) {
    plots.push({ id: c.id, title: c.id, unit: c.unit || '', note: c.doc || '', bounds: chBounds([c]), series: [{ name: c.id, key: c.id, cls: 'neutral', color: 's1', dash: [], alpha: 1, ...sc(c) }] });
  }

  // compare mode: per-engine relation error over time, and the v2-engine body distance
  if (scene.engines.length) {
    const names = ['v2', ...scene.engines.map((e) => e.name)];
    const ck = (nm, i) => (nm === 'v2' ? 'eV2' : nm === 'v1' ? 'eV1' : nm === 'v1eqw' ? 'eV1eqw' : ['s3', 's4', 's1'][i % 3]);
    const errSeries = [], origins = new Set();
    names.forEach((nm, i) => {
      const v = nulls(); let any = false;
      for (let k = 0; k < n; k++) { const e = scene.engineErr(nm, k); if (e) { v[k] = e.mean; any = true; origins.add(`${nm}: ${e.origin}`); } }
      if (any) errSeries.push({ name: nm, key: 'err.' + nm, cls: 'neutral', color: ck(nm, i), dash: [], alpha: 1, values: v });
    });
    if (errSeries.length) plots.push({ id: 'cmp-err', title: 'Relation error vs source', unit: '|dR|', note: [...origins].join('; '), series: errSeries });
    const okp = (d, k) => d && (!d.valid || d.valid[k] !== 0) && isNum(d.p[3 * k]) && isNum(d.p[3 * k + 1]);
    const v2d = scene.v2Ch?.data;
    if (v2d) {
      const ds = [];
      scene.engines.forEach((e, i) => {
        const v = nulls(), ed = e.ch.data;
        for (let k = 0; k < n; k++) if (okp(v2d, k) && okp(ed, k)) v[k] = Math.hypot(v2d.p[3 * k] - ed.p[3 * k], v2d.p[3 * k + 1] - ed.p[3 * k + 1]);
        ds.push({ name: `v2 - ${e.name}`, key: 'bd.' + e.name, cls: 'neutral', color: ck(e.name, i), dash: [], alpha: 1, values: v });
      });
      plots.push({ id: 'cmp-dist', title: 'Body distance v2 to engine', unit: 'm', note: 'xy distance between the two exported body poses', series: ds });
    }
  }

  // categorical strips
  const strips = [];
  for (const c of scene.channels) {
    if (c.type === 'enum') strips.push({ id: c.id, title: c.id, codes: c.data.codes, labels: c.data.labels });
    if (c.type === 'proposal') strips.push({ id: `${c.id}.status`, title: `${c.id}.status`, codes: c.data.status?.codes || [], labels: c.data.status?.labels || [], status: true });
    if (c.type === 'scope' && c.data.mode?.labels?.length) strips.push({ id: `${c.id}.mode`, title: `${c.id}.mode`, codes: c.data.mode?.codes || [], labels: c.data.mode?.labels || [] });
  }
  return { plots, strips };
}
