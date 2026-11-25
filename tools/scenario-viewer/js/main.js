// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Entry: wires data -> model -> renderer/ui. The URL keeps the open case (or table) id as a bare #token.
import { h, clear } from './util.js';
import { loadDataset, DataError } from './data/loader.js';
import { buildScene } from './model/scene.js';
import { VerdictStore, createVerdictPanel } from './ui/verdicts.js';
import { createBrowser } from './ui/browser.js';
import { renderHeader, renderResults } from './ui/header.js';
import { createStage } from './ui/stage.js';
import { createTimeline } from './ui/timeline.js';
import { createPlots } from './ui/plots.js';
import { createInspector } from './ui/inspector.js';
import { createTableView } from './ui/table.js';

const $ = (id) => document.getElementById(id);
const main = $('main'), banner = $('banner');

function say(text, kind = 'warn') {
  const box = h('div', { class: kind === 'err' ? 'err' : '' }, h('button', { type: 'button', class: 'x', 'aria-label': 'dismiss', onclick: () => box.remove() }, '×'), text);
  banner.append(box);
  return box;
}

// theme toggle: system -> light -> dark (sets data-theme, which the canvases watch)
(function theme() {
  const btn = $('themeBtn'), root = document.documentElement;
  const modes = ['system', 'light', 'dark'];
  let i = 0;
  btn.addEventListener('click', () => {
    i = (i + 1) % 3;
    if (modes[i] === 'system') root.removeAttribute('data-theme'); else root.setAttribute('data-theme', modes[i]);
    btn.textContent = `Theme: ${modes[i]}`;
  });
})();

// the browser panel is always open on wide screens, collapsible on a phone
(function browserPanel() {
  const d = $('browser'), mq = window.matchMedia('(min-width: 980px)');
  const sync = () => { if (mq.matches) d.open = true; };
  mq.addEventListener?.('change', sync); sync();
  if (!mq.matches) d.open = false;
})();

const app = { ds: null, store: new VerdictStore(), browser: null, token: 0, current: null, view: null };
window.__viewer = app; // debugging / headless checks

async function start() {
  app.store.init().catch((e) => console.error('verdict store', e));
  let ds;
  try { ds = await loadDataset(); }
  catch (e) {
    clear(main).append(h('div', { class: 'panel' }, h('h2', null, 'No data to show'), h('p', null, e instanceof DataError ? e.message : `Unexpected error: ${e?.message ?? e}`),
      h('p', { class: 'note' }, 'The viewer needs data/index.json (and the case files it lists) beside index.html. See README.md.')));
    $('browserSummary').textContent = 'Cases (none)';
    return;
  }
  app.ds = ds;
  app.store.coreSha = ds.coreSha; app.store.schemaVersion = ds.index.schema_version;
  const p = ds.index.provenance || {};
  $('provline').textContent = `${p.producer?.name ?? 'producer'} ${p.producer?.version ?? ''} · core ${(p.git_sha || 'unknown').slice(0, 10)} · schema ${ds.index.schema_version} · ${ds.cases.length} cases`;
  for (const w of ds.warnings) say(w);
  for (const x of ds.index.excluded || []) say(`Excluded by the producer: ${x}`);
  $('browserSummary').textContent = `Cases (${ds.cases.length}${ds.tables.length ? ` + ${ds.tables.length} tables` : ''})`;
  app.browser = createBrowser({ ds, store: app.store, onOpen: (entry) => go(entry.id) });
  $('browserBody').append(app.browser.root);
  window.addEventListener('hashchange', route);
  route();
}

const hashId = () => { try { return decodeURIComponent(location.hash.replace(/^#/, '')); } catch (e) { return ''; } };
function go(id) { const t = encodeURIComponent(id); if (location.hash === '#' + t) route(); else location.hash = t; }

function route() {
  const ds = app.ds;
  const id = hashId();
  let entry = id && (ds.caseById.get(id) || ds.tableById.get(id));
  if (!entry) {
    if (id) say(`"${id}" is not in the index; showing the first case instead.`);
    entry = ds.firstCase() || ds.tables[0];
    if (!entry) return;
    history.replaceState(null, '', '#' + encodeURIComponent(entry.id));
  }
  open(entry);
}

async function open(entry) {
  const token = ++app.token;
  const ds = app.ds;
  const isTable = ds.tableById.has(entry.id) && !ds.caseById.has(entry.id);
  app.current?.dispose?.(); app.current = null;
  app.browser.setCurrent(entry.id);
  const det = $('browser');
  if (!window.matchMedia('(min-width: 980px)').matches) det.open = false;
  clear(main).append(h('div', { class: 'loading' }, `Loading ${entry.id} …`));
  try {
    if (isTable) {
      const tv = await createTableView({ ds, entry, onOpenCase: (cid) => go(cid) });
      if (token !== app.token) return;
      clear(main).append(h('div', { class: 'panel' }, tv.root));
      app.view = { kind: 'table', tv };
      return;
    }
    const { json, warnings } = await ds.loadCase(entry);
    if (token !== app.token) return;
    const scene = buildScene(json);
    mountCase(entry, scene, warnings);
  } catch (e) {
    if (token !== app.token) return;
    console.error(e);
    clear(main).append(h('div', { class: 'panel' }, h('h2', null, `Could not open ${entry.id}`), h('p', null, e?.message ?? String(e))));
  }
}

function mountCase(entry, scene, warnings) {
  const header = h('section', { class: 'panel', id: 'p-header' });
  const verdictPanel = createVerdictPanel(app.store);
  const pv = h('section', { class: 'panel', id: 'p-verdict' }, verdictPanel.root);
  const stage = createStage({});
  const ps = h('section', { class: 'panel', id: 'p-stage' }, h('h3', { style: { marginBottom: '8px' } }, 'Stage'), stage.root);
  const timeline = createTimeline({ onTick: (t) => tickChanged(t) });
  const pt = h('section', { class: 'panel', id: 'p-timeline' }, h('h3', { style: { marginBottom: '8px' } }, 'Timeline'), timeline.root);
  const plots = createPlots({ onTick: (t) => timeline.set(t) });
  const pp = h('section', { class: 'panel', id: 'p-plots' }, plots.root);
  const inspector = createInspector();
  const pi = h('section', { class: 'panel', id: 'p-inspector' }, inspector.root);
  function tickChanged(t) { stage.setTime(t); plots.setTick(t); inspector.setTick(t); }
  const pr = h('section', { class: 'panel', id: 'p-results' });
  clear(main).append(h('div', { class: 'case-grid' }, header, ps, pt, pv, pr, pp, pi));
  renderHeader(header, { entry, scene, warnings, ds: app.ds });
  renderResults(pr, { scene, ds: app.ds, onJump: (t) => timeline.set(t) });
  verdictPanel.setCase({ suite: entry.suite ?? scene.json.case.suite, id: entry.id });
  stage.setScene(scene);
  timeline.setScene(scene);
  plots.setScene(scene);
  inspector.setScene(scene);
  app.view = { kind: 'case', entry, scene, stage, timeline, plots, inspector };
  app.current = { dispose: () => stage.dispose() };
}

start().catch((e) => { console.error(e); say(`Viewer failed to start: ${e?.message ?? e}`, 'err'); });
