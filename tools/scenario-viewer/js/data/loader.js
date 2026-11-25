// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Data layer: index + lazy case/table fetch, schema-version gate, provenance checks.
export class DataError extends Error {}

export const SUPPORTED_MAJOR = 1;
export const DATA_DIR = 'data/';

export function parseSemver(s) {
  const m = /^(\d+)\.(\d+)\.(\d+)/.exec(String(s ?? ''));
  return m ? { major: +m[1], minor: +m[2], patch: +m[3] } : null;
}

/** Accepts any 1.x; throws DataError otherwise. */
export function checkVersion(v, what) {
  const sv = parseSemver(v);
  if (!sv) throw new DataError(`${what}: missing or malformed schema_version (${JSON.stringify(v)}).`);
  if (sv.major !== SUPPORTED_MAJOR) {
    throw new DataError(`${what}: schema_version ${v} is not supported (this viewer reads ${SUPPORTED_MAJOR}.x).`);
  }
  return sv;
}

// Viewer safety limits for sidecars (not data values): a sidecar bigger than this, or a joints channel with more joints / samples
// than this, is refused with a warning instead of being parsed into memory.
export const LIMITS = { sidecarBytes: 64 * 1024 * 1024, joints: 512, samples: 20_000_000, ticks: 2_000_000 };

async function fetchJSON(url, what, maxBytes = 0) {
  let res;
  try { res = await fetch(url, { cache: 'no-cache' }); }
  catch (e) { throw new DataError(`${what}: network error while fetching ${url} (${e.message || e}).`); }
  if (!res.ok) throw new DataError(`${what}: ${url} returned HTTP ${res.status}.`);
  try {
    if (!maxBytes) return await res.json();
    const declared = +res.headers.get('content-length');
    if (declared > maxBytes) throw new DataError(`${what}: ${url} is ${declared} bytes, over the viewer limit of ${maxBytes}.`);
    const text = await res.text();
    if (text.length > maxBytes) throw new DataError(`${what}: ${url} is over the viewer limit of ${maxBytes} bytes.`);
    return JSON.parse(text);
  } catch (e) {
    if (e instanceof DataError) throw e;
    throw new DataError(`${what}: ${url} is not valid JSON (${e.message || e}).`);
  }
}

const isInt = (v) => Number.isInteger(v);
const isNumArr = (a, len) => Array.isArray(a) && a.length === len;

/** Why a sidecar joints channel is refused (null = fine). `nTicks` = the case's tick count. Pure. */
export function jointsProblem(ch, actorIds) {
  if (typeof ch.id !== 'string' || !ch.id || ch.id.length > 200) return 'bad channel id';
  const d = ch.data;
  if (!d || typeof d !== 'object') return 'no data object';
  if (!isInt(d.n) || d.n < 1 || d.n > LIMITS.ticks) return `n ${JSON.stringify(d.n)} is not an integer in 1..${LIMITS.ticks}`;
  if (!isInt(d.joint_count) || d.joint_count < 1 || d.joint_count > LIMITS.joints) return `joint_count ${JSON.stringify(d.joint_count)} is not an integer in 1..${LIMITS.joints}`;
  if (d.n * d.joint_count > LIMITS.samples) return `n x joint_count exceeds ${LIMITS.samples}`;
  if (!isNumArr(d.p, 3 * d.n * d.joint_count)) return `p must have 3*n*joint_count = ${3 * d.n * d.joint_count} numbers`;
  if (d.q !== undefined && !isNumArr(d.q, 4 * d.n * d.joint_count)) return `q must have 4*n*joint_count = ${4 * d.n * d.joint_count} numbers`;
  for (const k of ['horizon', 'buffer_revision']) {
    const v = d[k];
    if (v === undefined || v === null) continue;
    if (Array.isArray(v) ? v.length !== d.n || !v.every((x) => x === null || Number.isFinite(x)) : !Number.isFinite(v)) return `${k} must be a number or one number per tick`;
  }
  if (typeof ch.actor !== 'string' || !actorIds.has(ch.actor)) return `actor ${JSON.stringify(ch.actor)} is not an actor of this case`;
  return null;
}

/** horizon in whole ticks (floored, never negative) */
export const floorHorizon = (v) => (Array.isArray(v) ? v.map((x) => (x === null ? null : Math.max(0, Math.floor(x)))) : Math.max(0, Math.floor(v)));

export class Dataset {
  constructor(index, baseUrl) {
    this.index = index;
    this.baseUrl = baseUrl;          // URL of index.json; entry.file resolves against it
    this.warnings = [];
    this._cache = new Map();         // file -> parsed (LRU, small)
    this.coreSha = index.provenance?.git_sha || null;
    this.suites = index.suites || [];
    this.cases = index.cases || [];
    this.tables = index.tables || [];
    this.caseById = new Map(this.cases.map((c) => [c.id, c]));
    this.tableById = new Map(this.tables.map((t) => [t.id, t]));
  }

  url(file) { return new URL(file, this.baseUrl).href; }

  _remember(key, val) {
    this._cache.set(key, val);
    if (this._cache.size > 6) this._cache.delete(this._cache.keys().next().value);
    return val;
  }

  /** Lazy fetch of one case file. Returns {json, warnings:[]}. */
  async loadCase(entry) {
    const hit = this._cache.get(entry.file);
    if (hit) { this._cache.delete(entry.file); this._cache.set(entry.file, hit); return hit; }
    const json = await fetchJSON(this.url(entry.file), `case ${entry.id}`);
    if (json.format !== 'spatialblend.scenario-trace') {
      throw new DataError(`case ${entry.id}: format is ${JSON.stringify(json.format)}, expected "spatialblend.scenario-trace".`);
    }
    checkVersion(json.schema_version, `case ${entry.id}`);
    const warnings = [];
    const sha = json.provenance?.git_sha;
    if (this.coreSha && sha && sha !== this.coreSha && sha !== 'synthetic') {
      warnings.push(`Provenance: this case was produced at ${sha.slice(0, 12)} but the index is at ${this.coreSha.slice(0, 12)}.`);
    }
    if (entry.git_sha && sha && entry.git_sha !== sha && sha !== 'synthetic') {
      warnings.push(`Provenance: index lists ${entry.git_sha.slice(0, 12)} for this case, the file says ${sha.slice(0, 12)}.`);
    }
    if (json.case?.id && json.case.id !== entry.id) {
      warnings.push(`Case file id "${json.case.id}" differs from its index id "${entry.id}".`);
    }
    if (json.synthetic) warnings.push('SYNTHETIC fixture: not producer output, never real data.');
    for (const sc of entry.sidecars || []) await this._mergeSidecar(json, entry, sc, warnings);
    if (!json.embodiment && (json.channels || []).some((c) => /^(source|receiver)\.anchor\./.test(c.id))) json._embodimentFallback = await this._fallbackEmbodiment();
    return this._remember(entry.file, { json, warnings });
  }

  /** Contract 4: merge a sidecar (`spatialblend.scenario-trace.sidecar`, file relative to index.json) into the case. Channels keep their own
   * producer / derived / origin; an actor gets the sidecar's skeleton when it has none and the sidecar joints channel when it has no joints.
   * Unknown fields are ignored. A missing or malformed sidecar is a warning, never a failure. */
  async _mergeSidecar(json, entry, sc, warnings) {
    const file = sc?.file;
    if (!file || typeof file !== 'string') return;
    let side;
    try { side = await fetchJSON(this.url(file), `sidecar ${file}`, LIMITS.sidecarBytes); }
    catch (e) { warnings.push(`Sidecar ${file} could not be loaded (${e.message || e}); showing the case without it.`); return; }
    if (!side || side.format !== 'spatialblend.scenario-trace.sidecar') { warnings.push(`Sidecar ${file}: format is ${JSON.stringify(side?.format)}, expected "spatialblend.scenario-trace.sidecar"; ignored.`); return; }
    try { checkVersion(side.schema_version, `sidecar ${file}`); } catch (e) { warnings.push(`${e.message} Ignored.`); return; }
    if (side.case_id && side.case_id !== entry.id) { warnings.push(`Sidecar ${file} is for case "${side.case_id}", not "${entry.id}"; ignored.`); return; }
    const producer = side.producer || (sc.producer ? { name: sc.producer } : null);
    json.channels = json.channels || [];
    const have = new Set(json.channels.map((c) => c.id));
    const actorIds = new Set((json.actors || []).map((a) => a.id));
    for (const a of Array.isArray(side.actors) ? side.actors : []) {
      const act = (json.actors || []).find((x) => x.id === a?.id);
      const sk = a?.skeleton, js = sk?.joints;
      const okSk = Array.isArray(js) && js.length >= 1 && js.length <= LIMITS.joints && js.every((j) => typeof j?.name === 'string' && isInt(j.parent) && j.parent >= -1 && j.parent < js.length);
      if (act && sk && !okSk) warnings.push(`Sidecar ${file}: skeleton of actor ${a.id} is malformed; ignored.`);
      else if (act && okSk && !(act.skeleton && (act.skeleton.joints || []).length)) act.skeleton = sk;
    }
    const rejected = [];
    for (const ch of Array.isArray(side.channels) ? side.channels : []) {
      if (!ch || typeof ch !== 'object') continue;
      if (ch.type !== 'joints') { rejected.push(String(ch.id)); continue; }       // a sidecar carries joints only (contract 3 / 4)
      const why = jointsProblem(ch, actorIds);
      if (why) { warnings.push(`Sidecar ${file}: channel ${String(ch.id).slice(0, 80)} refused (${why}).`); continue; }
      if (have.has(ch.id)) { warnings.push(`Sidecar ${file}: channel ${ch.id} already exists in the case; not overwritten.`); continue; }
      const data = { ...ch.data };
      if (data.horizon !== undefined && data.horizon !== null) data.horizon = floorHorizon(data.horizon);
      const c = { ...ch, data, producer: ch.producer || producer, sidecar: file };   // `sidecar` marks the origin: the viewer labels it MotionGen output whatever `derived` says
      json.channels.push(c); have.add(c.id);
      const act = json.actors.find((x) => x.id === c.actor);
      if (/\.lookahead$/.test(c.id)) act.lookahead_channel = c.id;
      else if (!act.joints_channel || !(json.channels.find((x) => x.id === act.joints_channel)?.data?.n)) act.joints_channel = c.id;
    }
    if (rejected.length) warnings.push(`Sidecar ${file}: only joints channels are merged; refused ${rejected.length} other channel(s): ${rejected.slice(0, 8).join(', ')}${rejected.length > 8 ? ' ...' : ''}.`);
    (json.sidecars_merged ||= []).push({ file, producer });
  }

  /** Interim display fallback (embodiments/smplx.json) for cases that carry anchors but no `embodiment` registration. Fetched once; a failure only means no offset. */
  async _fallbackEmbodiment() {
    if (this._emb === undefined) {
      try { this._emb = await fetchJSON(new URL('embodiments/smplx.json', document.baseURI).href, 'embodiment fallback'); }
      catch (e) { this._emb = null; this.warnings.push(`Interim embodiment fallback not available (${e.message || e}); a seated figure is drawn with its pelvis at the anchor.`); }
    }
    return this._emb;
  }

  async loadTable(entry) {
    const hit = this._cache.get(entry.file);
    if (hit) return hit;
    const json = await fetchJSON(this.url(entry.file), `table ${entry.id}`);
    if (json.format !== 'spatialblend.scenario-trace.table') {
      throw new DataError(`table ${entry.id}: format is ${JSON.stringify(json.format)}, expected "spatialblend.scenario-trace.table".`);
    }
    checkVersion(json.schema_version, `table ${entry.id}`);
    if (!Array.isArray(json.columns) || !Array.isArray(json.rows)) throw new DataError(`table ${entry.id}: columns/rows missing.`);
    const warnings = [];
    const sha = json.provenance?.git_sha;
    if (this.coreSha && sha && sha !== this.coreSha) warnings.push(`Provenance: table produced at ${sha.slice(0, 12)}, index at ${this.coreSha.slice(0, 12)}.`);
    return this._remember(entry.file, { json, warnings });
  }

  /** First case of the first suite that has cases (suite order from index.suites). */
  firstCase() {
    for (const s of this.suites) {
      const c = this.cases.find((x) => x.suite === s.id);
      if (c) return c;
    }
    return this.cases[0] || null;
  }
}

export async function loadDataset(dir = DATA_DIR) {
  const base = new URL(dir + 'index.json', document.baseURI).href;
  let index;
  try { index = await fetchJSON(base, 'index'); }
  catch (e) {
    if (e instanceof DataError) {
      throw new DataError(`No data found. ${e.message} Put an export under "${dir}" next to index.html (see README: "how to view locally").`);
    }
    throw e;
  }
  if (!index || index.format !== 'spatialblend.scenario-trace.index') {
    throw new DataError(`${dir}index.json is not an spatialblend.scenario-trace.index (format = ${JSON.stringify(index?.format)}).`);
  }
  checkVersion(index.schema_version, 'index');
  const ds = new Dataset(index, base);
  if (!ds.cases.length && !ds.tables.length) {
    throw new DataError(`${dir}index.json lists no cases and no tables. The export is empty.`);
  }
  if (!ds.cases.length) ds.warnings.push('The index lists tables but no trace cases.');
  for (const m of index.missing || []) ds.warnings.push(`Producer reports missing: ${m}`);
  return ds;
}
