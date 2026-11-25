// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Matrix table view: sortable, filterable, virtualized (fixed row height, only visible rows are in the DOM).
import { h, isNum } from '../util.js';

const ROW_H = 26;

function parseFilter(s) {
  const m = /^\s*(>=|<=|!=|>|<|=)\s*(-?[0-9.eE+-]+)\s*$/.exec(s);
  if (m && Number.isFinite(+m[2])) { const v = +m[2]; return { num: (x) => isNum(x) && ({ '>': x > v, '<': x < v, '>=': x >= v, '<=': x <= v, '=': x === v, '!=': x !== v }[m[1]]) }; }
  const q = s.trim().toLowerCase();
  return q ? { sub: (x) => String(x ?? '').toLowerCase().includes(q) } : null;
}

export async function createTableView({ ds, entry, onOpenCase }) {
  const { json } = await ds.loadTable(entry);
  const cols = json.columns, rows = json.rows;
  const traceIdx = entry.trace_case ? cols.indexOf(entry.trace_case) : -1;
  const S = { sort: -1, dir: 1, global: '', col: new Array(cols.length).fill('') };
  let view = rows.map((_, i) => i);

  // column widths from header + a sample of values
  const widths = cols.map((c, i) => {
    let w = c.length * 7.4 + 30;
    for (let r = 0; r < Math.min(rows.length, 300); r++) w = Math.max(w, String(rows[r][i] ?? '').length * 7 + 16);
    return Math.min(300, Math.max(70, Math.round(w)));
  });
  const colsVar = widths.map((w) => w + 'px').join(' ');
  const total = widths.reduce((a, b) => a + b, 0);

  const count = h('span', { class: 'mono muted' });
  const scroller = h('div', { class: 'tv-scroll', tabindex: 0, 'aria-label': `${entry.title || entry.id} table` });
  const headRow = h('div', { class: 'tv-head', style: { '--cols': colsVar, width: total + 'px' } });
  const filtRow = h('div', { class: 'tv-head tv-filters', style: { '--cols': colsVar, width: total + 'px', top: '27px' } });
  const spacer = h('div', { style: { height: '0px', width: total + 'px', position: 'relative' } });
  const sortMark = (i) => (S.sort === i ? (S.dir > 0 ? ' ▲' : ' ▼') : '');
  const headBtns = cols.map((c, i) => h('button', { type: 'button', title: 'sort', onclick: () => { if (S.sort === i) { if (S.dir > 0) S.dir = -1; else { S.sort = -1; S.dir = 1; } } else { S.sort = i; S.dir = 1; } recompute(); } }, c));
  cols.forEach((c, i) => {
    headRow.append(h('div', null, headBtns[i]));
    filtRow.append(h('div', null, h('input', { type: 'text', 'aria-label': `filter ${c}`, placeholder: 'filter', oninput: (e) => { S.col[i] = e.target.value; clearTimeout(S._t); S._t = setTimeout(recompute, 120); } })));
  });
  // sticky header block
  const head = h('div', { style: { position: 'sticky', top: 0, zIndex: 3, width: total + 'px' } }, headRow, filtRow);
  headRow.style.position = 'relative'; filtRow.style.position = 'relative'; filtRow.style.top = '0';
  scroller.append(head, spacer);

  function recompute() {
    const fs = S.col.map(parseFilter), g = S.global.trim().toLowerCase();
    let idx = [];
    for (let r = 0; r < rows.length; r++) {
      const row = rows[r];
      let ok = true;
      for (let c = 0; c < cols.length && ok; c++) {
        const f = fs[c];
        if (f && !(f.num ? f.num(row[c]) : f.sub(row[c]))) ok = false;
      }
      if (ok && g && !row.some((v) => String(v ?? '').toLowerCase().includes(g))) ok = false;
      if (ok) idx.push(r);
    }
    if (S.sort >= 0) {
      const c = S.sort, d = S.dir;
      idx.sort((a, b) => {
        const x = rows[a][c], y = rows[b][c];
        if (x === y) return a - b;
        if (x === null || x === undefined) return 1;
        if (y === null || y === undefined) return -1;
        if (isNum(x) && isNum(y)) return (x - y) * d;
        return String(x).localeCompare(String(y), undefined, { numeric: true }) * d;
      });
    }
    view = idx;
    headBtns.forEach((b, i) => { b.textContent = cols[i] + sortMark(i); });
    count.textContent = `${view.length} of ${rows.length} rows`;
    spacer.style.height = view.length * ROW_H + 'px';
    scroller.scrollTop = 0;
    paint();
  }

  let first = 0, last = -1;
  function paint() {
    const top = scroller.scrollTop, hh = scroller.clientHeight;
    const a = Math.max(0, Math.floor(top / ROW_H) - 8), b = Math.min(view.length - 1, Math.ceil((top + hh) / ROW_H) + 8);
    spacer.replaceChildren();
    for (let i = a; i <= b; i++) {
      const row = rows[view[i]];
      const tc = traceIdx >= 0 && row[traceIdx] && ds.caseById.has(String(row[traceIdx])) ? String(row[traceIdx]) : null;
      const el = h('div', { class: `tv-row${tc ? ' trace' : ''}`, style: { '--cols': colsVar, top: i * ROW_H + 'px', width: total + 'px' }, tabindex: tc ? 0 : null, role: tc ? 'button' : null, title: tc ? `open trace case ${tc}` : null },
        ...row.map((v) => { const t = v === null || v === undefined ? '–' : typeof v === 'number' ? String(+v.toFixed(5)) : String(v); return h('div', { title: t.length > 24 ? t : null }, t); }));
      if (tc) {
        el.addEventListener('click', () => onOpenCase(tc));
        el.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpenCase(tc); } });
      }
      spacer.append(el);
    }
    first = a; last = b;
  }
  scroller.addEventListener('scroll', () => requestAnimationFrame(paint), { passive: true });
  new ResizeObserver(() => paint()).observe(scroller);

  const guards = json.guards && Object.keys(json.guards).length
    ? h('details', { class: 'sub' }, h('summary', null, 'Guards (as exported)'), h('pre', { class: 'mono', style: { overflow: 'auto', maxHeight: '160px' } }, JSON.stringify(json.guards, null, 1)))
    : null;
  const root = h('div', null,
    h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'baseline' } }, h('h2', null, entry.title || entry.id), h('span', { class: 'chip' }, 'TABLE'), h('span', { class: 'mono muted' }, entry.id)),
    entry.source ? h('div', { class: 'note' }, 'Source: ', h('span', { class: 'mono' }, entry.source)) : null,
    traceIdx >= 0 ? h('div', { class: 'note' }, `Rows with a value in "${entry.trace_case}" that names a trace case are marked ▶ and open that case.`) : h('div', { class: 'note' }, 'This table has no trace_case column; rows do not open cases.'),
    guards,
    h('div', { class: 'tv-ctl' }, h('input', { type: 'search', 'aria-label': 'filter all columns', placeholder: 'filter all columns', oninput: (e) => { S.global = e.target.value; clearTimeout(S._t); S._t = setTimeout(recompute, 120); } }), count,
      h('span', { class: 'note' }, 'Column filters: text, or >0.5  <=2  =1  !=0 for numbers.')),
    scroller);
  requestAnimationFrame(recompute);
  return { root, info: () => ({ rows: rows.length, shown: view.length, domRows: spacer.childElementCount, first, last }) };
}
