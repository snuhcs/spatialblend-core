// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Owner verdicts via the artifact runtime `db` + `user` capabilities (declared at publish: {db:{}, user:{}}).
// Everything is guarded: with no `claude` global (local static server) or null capabilities the UI is
// disabled with "Verdicts are recorded on the published page" and nothing else changes.
import { h } from '../util.js';

export const DISABLED_LINE = 'Verdicts are recorded on the published page';
export const VERDICTS = ['ok', 'wrong', 'unclear'];

/** Document id: letters, digits and _ - . ~ : @ + only (db path grammar); other bytes become ~hex. */
export function verdictKey(suite, caseId) {
  const raw = `${suite}__${caseId}`;
  let out = '';
  for (const ch of raw) {
    if (/[A-Za-z0-9_\-.~:@+]/.test(ch)) out += ch;
    else for (const b of new TextEncoder().encode(ch)) out += '~' + b.toString(16).padStart(2, '0');
  }
  if (out.length > 190) {
    let hsh = 5381;
    for (let i = 0; i < out.length; i++) hsh = ((hsh * 33) ^ out.charCodeAt(i)) >>> 0;
    out = out.slice(0, 170) + '~~' + hsh.toString(16);
  }
  return out;
}

const withTimeout = (p, ms) => Promise.race([p, new Promise((res) => setTimeout(() => res(null), ms))]);

export class VerdictStore {
  constructor() {
    this.available = false;       // db + user + a user id, and not read-only
    this.reason = DISABLED_LINE;  // why it is disabled (shown verbatim)
    this.docs = new Map();        // doc id -> data
    this.names = new Map();       // user id -> display name (resolved via user.profiles)
    this.uid = null;
    this.coreSha = null;
    this.schemaVersion = null;
    this.listeners = new Set();
    this.error = null;
    this._asked = new Set();
    this.ready = false;
  }
  onChange(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  _emit() { for (const fn of this.listeners) { try { fn(); } catch (e) { console.error(e); } } }

  async init() {
    let db = null, user = null;
    try {
      const c = window.claude;
      if (c && typeof c.use === 'function') {
        db = await withTimeout(Promise.resolve(c.use('db')).catch(() => null), 12000);
        user = await withTimeout(Promise.resolve(c.use('user')).catch(() => null), 12000);
      }
    } catch (e) { db = user = null; }
    this._db = db; this._user = user;
    if (db && user) {
      try { this.uid = await user.id(); } catch (e) { this.uid = null; }
      if (this.uid) {
        let can = null;
        try { can = await user.can('data.write'); } catch (e) { can = null; }
        if (can === false) { this.available = false; this.reason = 'Viewing only: this access level cannot record verdicts'; }
        else this.available = true;
        // subscribe ONCE; never from render code
        try {
          this._unsub = db.collection('verdicts').onSnapshot((snap) => {
            this.docs.clear();
            for (const d of snap.docs) if (d.exists) this.docs.set(d.id, d.data());
            this._resolveNames();
            this._emit();
          }, (err) => {
            this.error = `Verdict list unavailable (${err?.code ?? 'error'})`;
            this._emit();
          });
        } catch (e) { this.error = `Verdict list unavailable (${e?.message ?? e})`; }
      }
    }
    this.ready = true;
    this._emit();
  }

  async _resolveNames() {
    if (!this._user) return;
    const ids = [...new Set([...this.docs.values()].map((d) => d?.by).filter((x) => typeof x === 'string'))].filter((i) => !this._asked.has(i));
    if (!ids.length) return;
    ids.forEach((i) => this._asked.add(i));
    try {
      const prof = await this._user.profiles(ids);
      for (const [id, p] of Object.entries(prof || {})) this.names.set(id, p.name || id);
      this._emit();
    } catch (e) { /* names are decoration; ids stay */ }
  }

  get(suite, caseId) {
    const data = this.docs.get(verdictKey(suite, caseId));
    if (!data) return null;
    const stale = !!(this.coreSha && data.core_sha !== this.coreSha);
    return { ...data, stale };
  }

  /** One write per deliberate Save click. */
  async set(suite, caseId, verdict, note) {
    if (!this.available || !this._db) throw new Error(this.reason);
    if (!VERDICTS.includes(verdict)) throw new Error('verdict must be ok, wrong or unclear');
    const data = { verdict, note: note || '', case_id: caseId, suite, core_sha: this.coreSha ?? null, schema_version: this.schemaVersion ?? null, by: this.uid, at: new Date().toISOString() };
    await this._db.collection('verdicts').doc(verdictKey(suite, caseId)).set(data);
    return data;
  }
  dispose() { try { this._unsub?.(); } catch (e) { /* */ } }
}

/** Chip for the case browser. */
export function verdictChip(store, suite, id) {
  const v = store.get(suite, id);
  if (!v) return null;
  return h('span', { class: `chip v-${v.verdict}${v.stale ? ' stale' : ''}`, title: v.stale ? `STALE: recorded at core ${String(v.core_sha).slice(0, 10)}, current ${String(store.coreSha).slice(0, 10)}` : `verdict: ${v.verdict}` }, v.verdict.toUpperCase(), v.stale ? ' STALE' : '');
}

/** The verdict panel for the open case. Re-renders from the store; does not subscribe to the db. */
export function createVerdictPanel(store) {
  const root = h('div', { class: 'verdict' });
  let cur = null;               // {suite, id}
  const drafts = new Map();     // key -> {verdict, note}
  let msg = '', busy = false;

  const render = () => {
    root.replaceChildren();
    root.append(h('h3', null, 'Owner verdict'));
    if (!cur) return;
    const key = verdictKey(cur.suite, cur.id);
    const stored = store.get(cur.suite, cur.id);
    const draft = drafts.get(key) || { verdict: stored?.verdict ?? null, note: stored?.note ?? '' };
    const enabled = store.available && !busy;
    if (!store.available) root.append(h('p', { class: 'note', style: { margin: '6px 0' } }, store.reason));
    const opts = h('div', { class: 'opts', role: 'group', 'aria-label': 'verdict' });
    for (const v of VERDICTS) {
      opts.append(h('button', { type: 'button', class: `v-${v}`, 'aria-pressed': String(draft.verdict === v), disabled: !enabled, onclick: () => { drafts.set(key, { ...draft, verdict: v }); msg = ''; render(); } }, v));
    }
    const ta = h('textarea', { 'aria-label': 'verdict note', placeholder: 'Note (what is right or wrong here)', disabled: !enabled, oninput: (e) => { const d = drafts.get(key) || { ...draft }; d.note = e.target.value; drafts.set(key, d); saveBtn.disabled = !canSave(); } });
    ta.value = draft.note;
    const canSave = () => { const d = drafts.get(key) || draft; return enabled && VERDICTS.includes(d.verdict) && (d.verdict !== stored?.verdict || (d.note || '') !== (stored?.note || '') || stored?.stale); };
    const saveBtn = h('button', { type: 'button', disabled: !canSave(), onclick: async () => {
      const d = drafts.get(key) || draft;
      busy = true; msg = 'Saving…'; render();
      try { await store.set(cur.suite, cur.id, d.verdict, d.note); drafts.delete(key); msg = 'Saved.'; }
      catch (e) { msg = `Not saved: ${e?.message ?? e?.code ?? e}`; }
      busy = false; render();
    } }, stored?.stale ? 'Re-record for current data' : 'Save verdict');
    root.append(opts, ta, h('div', { style: { marginTop: '6px', display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' } }, saveBtn, h('span', { class: 'note', role: 'status' }, msg)));
    if (stored) {
      const who = store.names.get(stored.by) || stored.by || 'unknown';
      root.append(h('p', { class: 'note', style: { margin: '8px 0 0' } },
        `Recorded: ${stored.verdict.toUpperCase()} by ${who} at ${stored.at ?? '?'}`,
        stored.stale ? h('span', { class: 'chip bad', style: { marginLeft: '6px' } }, `STALE (core ${String(stored.core_sha).slice(0, 10)} vs current ${String(store.coreSha).slice(0, 10)})`) : null));
    }
    if (store.error) root.append(h('p', { class: 'note' }, store.error));
  };
  store.onChange(() => { if (document.activeElement?.tagName === 'TEXTAREA' && root.contains(document.activeElement)) return; render(); });
  return { root, setCase(c) { cur = c; msg = ''; render(); } };
}
