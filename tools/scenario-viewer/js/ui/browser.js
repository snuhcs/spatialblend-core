// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Case browser: sections per suite, filters, id + rule line + producer pass chip + verdict chip.
import { h, clear, familyClass } from '../util.js';
import { verdictChip } from './verdicts.js';

const PAGE = 200;

export function passChip(p) {
  if (p === true) return h('span', { class: 'chip good', title: 'producer verdict: pass' }, 'PASS');
  if (p === false) return h('span', { class: 'chip bad solid', title: 'producer verdict: fail' }, 'FAIL');
  return h('span', { class: 'chip', title: 'the producer made no assertion for this case' }, 'N/A');
}

export function createBrowser({ ds, store, onOpen }) {
  const f = { text: '', family: '', mode: '', status: '', pass: '', verdict: '' };
  const shown = new Map(); // suite -> count shown
  let currentId = null;
  const uniq = (key) => [...new Set(ds.cases.flatMap((c) => c[key] || []))].sort();
  const opt = (label, vals) => [h('option', { value: '' }, label), ...vals.map((v) => h('option', { value: v }, v))];
  const sel = (key, label, vals, extra) => h('select', { 'aria-label': label, onchange: (e) => { f[key] = e.target.value; shown.clear(); renderList(); } }, opt(label, vals), extra);

  const filters = h('div', { class: 'filters' },
    h('input', { type: 'search', 'aria-label': 'search id, title, rule', placeholder: 'search id / rule', oninput: (e) => { f.text = e.target.value.trim().toLowerCase(); shown.clear(); renderList(); } }),
    sel('family', 'family', uniq('families')),
    sel('mode', 'mode', uniq('modes')),
    sel('status', 'status', uniq('statuses')),
    h('select', { 'aria-label': 'pass or fail', onchange: (e) => { f.pass = e.target.value; shown.clear(); renderList(); } },
      h('option', { value: '' }, 'pass / fail'), h('option', { value: 'pass' }, 'pass'), h('option', { value: 'fail' }, 'fail'), h('option', { value: 'none' }, 'no assertion')),
    h('select', { 'aria-label': 'verdict', onchange: (e) => { f.verdict = e.target.value; shown.clear(); renderList(); } },
      h('option', { value: '' }, 'verdict'), h('option', { value: 'none' }, 'unjudged'), h('option', { value: 'ok' }, 'ok'), h('option', { value: 'wrong' }, 'wrong'), h('option', { value: 'unclear' }, 'unclear'), h('option', { value: 'stale' }, 'stale')));
  const list = h('div', { id: 'caseList' });
  const root = h('div', null, filters, list);

  const match = (c) => {
    if (f.text && !`${c.id} ${c.title || ''} ${c.rule || ''}`.toLowerCase().includes(f.text)) return false;
    if (f.family && !(c.families || []).includes(f.family)) return false;
    if (f.mode && !(c.modes || []).includes(f.mode)) return false;
    if (f.status && !(c.statuses || []).includes(f.status)) return false;
    if (f.pass === 'pass' && c.pass !== true) return false;
    if (f.pass === 'fail' && c.pass !== false) return false;
    if (f.pass === 'none' && (c.pass === true || c.pass === false)) return false;
    if (f.verdict) {
      const v = store.get(c.suite, c.id);
      if (f.verdict === 'none' ? !!v : f.verdict === 'stale' ? !(v && v.stale) : !(v && v.verdict === f.verdict)) return false;
    }
    return true;
  };

  function renderList() {
    clear(list);
    const suites = [...ds.suites];
    for (const c of ds.cases) if (!suites.some((s) => s.id === c.suite)) suites.push({ id: c.suite, title: c.suite });
    for (const t of ds.tables) if (!suites.some((s) => s.id === t.suite)) suites.push({ id: t.suite, title: t.suite });
    let any = false;
    for (const s of suites) {
      const all = ds.cases.filter((c) => c.suite === s.id);
      const hits = all.filter(match);
      const tables = ds.tables.filter((t) => t.suite === s.id && (!f.text || `${t.id} ${t.title}`.toLowerCase().includes(f.text)) && !f.family && !f.mode && !f.status && !f.pass && !f.verdict);
      if (!all.length && !tables.length) continue;
      if (!hits.length && !tables.length && (f.text || f.family || f.mode || f.status || f.pass || f.verdict)) continue;
      any = true;
      const n = shown.get(s.id) || PAGE;
      const det = h('details', { class: 'suite', open: true },
        h('summary', null, `${s.title || s.id} `, h('span', { class: 'muted mono' }, `${hits.length}/${all.length}${tables.length ? ` + ${tables.length} table${tables.length > 1 ? 's' : ''}` : ''}`)));
      if (s.description) det.append(h('div', { class: 'desc' }, s.description));
      for (const t of tables) {
        det.append(h('button', { type: 'button', class: 'case-row', 'aria-current': String(currentId === t.id), dataset: { id: t.id }, onclick: () => onOpen(t, 'table') },
          h('span', { class: 'chips' }, h('span', { class: 'chip' }, 'TABLE')),
          h('span', { class: 'id' }, t.id), h('span', { class: 'rule' }, `${t.title ?? ''} · ${t.rows ?? '?'} rows`)));
      }
      for (const c of hits.slice(0, n)) {
        const vc = verdictChip(store, c.suite, c.id);
        det.append(h('button', { type: 'button', class: 'case-row', 'aria-current': String(currentId === c.id), dataset: { id: c.id }, onclick: () => onOpen(c, 'case') },
          h('span', { class: 'chips' }, passChip(c.pass), vc),
          h('span', { class: 'id' }, c.id, ' ', ...(c.families || []).map((x) => h('span', { class: `dot ${familyClass(x)}`, title: x }))),
          h('span', { class: 'rule', title: c.rule || c.title || '' }, c.rule || c.title || '')));
      }
      if (hits.length > n) det.append(h('button', { type: 'button', class: 'more', onclick: () => { shown.set(s.id, n + PAGE); renderList(); } }, `Show more (${hits.length - n} left)`));
      list.append(det);
    }
    if (!any) list.append(h('p', { class: 'note' }, 'No case matches these filters.'));
  }
  store.onChange(renderList);
  renderList();
  return { root, refresh: renderList, setCurrent(id) {
    currentId = id;
    for (const b of list.querySelectorAll('.case-row')) b.setAttribute('aria-current', String(b.dataset.id === id));
    list.querySelector('.case-row[aria-current="true"]')?.scrollIntoView?.({ block: 'nearest' });
  } };
}
