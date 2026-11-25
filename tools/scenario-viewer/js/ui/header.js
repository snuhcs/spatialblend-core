// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Case header: rule, expectations vs producer results, metrics vs guards, provenance.
// pass / fail comes from the producer (index entry, results[], guards[]). Nothing is recomputed here.
import { h, fmt, isNum } from '../util.js';
import { passChip } from './browser.js';

const show = (v) => (v === null || v === undefined ? '–' : typeof v === 'object' ? JSON.stringify(v) : String(v));

export function renderHeader(root, { entry, scene, warnings, ds, onJump }) {
  const j = scene.json, c = j.case || {}, spec = c.spec || {};
  root.replaceChildren();
  const title = h('div', { class: 'hd-title' },
    h('h2', null, c.title || c.id), passChip(entry?.pass),
    j.synthetic ? h('span', { class: 'chip synthetic', title: 'Not producer output. A fixture for exercising the viewer.' }, 'SYNTHETIC') : null,
    h('span', { class: 'chip' }, c.suite), spec.level ? h('span', { class: 'chip' }, spec.level) : null);
  root.append(title, h('div', { class: 'mono muted' }, c.id));
  if (spec.rule) root.append(h('p', { class: 'rule' }, spec.rule));
  if (spec.citations?.length) root.append(h('div', { class: 'note' }, 'Cites: ', spec.citations.join(' · ')));
  if (spec.source) root.append(h('div', { class: 'note' }, 'Source: ', h('span', { class: 'mono' }, spec.source), spec.plan_key ? ` · plan ${spec.plan_key}` : ''));
  if (c.tags && Object.keys(c.tags).length) root.append(h('div', { style: { marginTop: '4px', display: 'flex', flexWrap: 'wrap', gap: '3px' } }, ...Object.entries(c.tags).map(([k, v]) => h('span', { class: 'chip wrap' }, `${k}: ${Array.isArray(v) ? v.join(',') : (v && typeof v === 'object') ? JSON.stringify(v) : v}`))));
  for (const w of warnings || []) root.append(h('div', { class: 'chip warn', style: { display: 'block', whiteSpace: 'normal', marginTop: '6px', lineHeight: '1.4', padding: '3px 6px' } }, w));

}

export function renderResults(root, { scene, ds, onJump }) {
  const j = scene.json;
  root.replaceChildren();
  // expectations vs results
  const ex = j.expectations || [], res = new Map((j.results || []).map((r) => [r.expectation, r]));
  if (ex.length) {
    const rows = ex.map((e) => {
      const r = res.get(e.id);
      const bad = r?.pass === false;
      return h('tr', { class: bad ? 'bad' : '' },
        h('td', null, e.id),
        h('td', null, e.kind, e.family ? ` · ${e.family}` : '', isNum(e.tick) ? h('button', { type: 'button', style: { marginLeft: '6px', minHeight: '20px', padding: '0 5px' }, title: 'jump to this tick', onclick: () => onJump?.(e.tick) }, `t${e.tick}`) : null),
        h('td', { class: 'mono' }, e.text && e.expected === undefined ? e.text : show(e.expected)),
        h('td', { class: 'mono' }, r ? show(r.actual) : '–'),
        h('td', null, r ? (r.pass === true ? h('span', { class: 'chip good' }, 'PASS') : r.pass === false ? h('span', { class: 'chip bad solid' }, 'FAIL') : h('span', { class: 'chip' }, 'N/E')) : h('span', { class: 'chip' }, 'NO RESULT')),
        h('td', { class: 'muted' }, [r?.test, r?.note].filter(Boolean).join(' · ')));
    });
    root.append(h('h3', { style: { marginTop: '10px' } }, 'Expected vs actual (producer results)'),
      h('div', { class: 'tbl-wrap' }, h('table', { class: 't' }, h('thead', null, h('tr', null, ...['id', 'kind', 'expected', 'actual', 'result', 'test'].map((x) => h('th', null, x)))), h('tbody', null, rows))));
  } else root.append(h('p', { class: 'note' }, 'This case carries no expectations.'));

  // metrics vs guards
  const m = j.metrics || {}, row = m.row || {}, guards = new Map((m.guards || []).map((g) => [g.column, g]));
  const cols = [...new Set([...Object.keys(row), ...guards.keys()])];
  if (cols.length) {
    root.append(h('h3', { style: { marginTop: '10px' } }, 'Metrics vs guards'));
    root.append(h('div', { class: 'metrics' }, ...cols.map((k) => {
      const g = guards.get(k);
      const cls = g ? (g.pass === false ? 'bad' : g.pass === true ? 'good' : '') : '';
      return h('div', { class: `metric ${cls}` },
        h('div', { class: 'mk' }, k),
        h('div', { class: 'mv' }, show(isNum(row[k]) ? +row[k].toFixed(4) : row[k] ?? g?.value)),
        g ? h('div', { class: 'mk' }, `bound ${show(g.bound)} · `, g.pass === false ? h('span', { style: { color: 'var(--bad)', fontWeight: 600 } }, 'GUARD FAIL') : g.pass === true ? 'guard ok' : 'guard not evaluated') : null);
    })));
  }

  const p = j.provenance || {};
  root.append(h('details', { class: 'sub' }, h('summary', null, 'Provenance'),
    h('div', { class: 'kv' }, ...[['producer', p.producer ? `${p.producer.name} ${p.producer.version}` : null], ['repo', p.repo], ['git_sha', p.git_sha], ['index git_sha', ds.coreSha], ['command', p.command], ['generated_at', p.generated_at], ['schema_version', j.schema_version]]
      .flatMap(([k, v]) => [h('div', { class: 'k' }, k), h('div', { class: 'v' }, show(v))]))));
  if (j.missing?.length) root.append(h('details', { class: 'sub' }, h('summary', null, `Producer-declared missing fields (${j.missing.length})`), h('ul', { style: { margin: '4px 0', paddingLeft: '18px' } }, ...j.missing.map((x) => h('li', null, x)))));
}
