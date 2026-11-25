# SpatialBlend scenario viewer

The viewer half of the official scenario-verification tool. It reads `spatialblend.scenario-trace` v1.x files
(`tools/scenario-viewer/trace/SCHEMA.md`) and shows them: a top-down 2D stage (primary), a 3D perspective stage,
a timeline, small time plots, an inspector, the producer's expectations vs results, and the owner's
verdicts. It never recomputes geometry or pass/fail: everything drawn is a value the producer exported.

Plain ES modules, no build step, no dependency except three.js 0.160.0 (exact version, cdnjs ES-module build, SRI-checked),
which is loaded lazily on the first use of the 3D view and prints no console message (see "3D view" below). Viewer code is about 170 KB.

## Layout

```
index.html        the page (page contract: no doctype/html/head/body; <title>, <style>, then markup)
js/main.js        wiring + URL routing (#<case id>, a bare token)
js/data/          loader: index + lazy case/table fetch, schema-version gate (any 1.x), provenance warnings
js/model/         scene.js (state at tick t, indexing only), series.js (plot series from existing channels)
js/render/        renderer.js (interface + layer list), topdown2d.js, perspective3d.js, theme.js (CSS tokens)
js/ui/            browser, header (+results), stage, timeline, plots, inspector, table, verdicts
bundle.sh         inlines all modules into one index.bundle.html (CSP fallback)
dev-data/         make_dev_data.py (SYNTHETIC dev data), check.py (headless check)
trace/            the input contract: SCHEMA.md, schema/*.json, validate.py, fixtures/, scope_context.py
```

## Export

The input format is `spatialblend.scenario-trace` v1, specified in `trace/SCHEMA.md` (JSON Schemas in `trace/schema/`, checked by
`trace/validate.py`). The producers that export it are not part of this repository. The viewer needs a
directory `data/` beside `index.html` holding `index.json` and the files it lists. `file` entries resolve
relative to `index.json`.

## View locally

```
cd tools/scenario-viewer
ln -s /path/to/export data          # or: ln -s dev-data/data data   (synthetic dev data)
python3 -m http.server 8765 --bind 127.0.0.1
# open http://127.0.0.1:8765/index.html
```

Without `data/` the page says so in plain words instead of showing an empty shell. Locally there is no
`window.claude`, so the verdict UI is disabled with "Verdicts are recorded on the published page".

Headless check (no GUI window; needs python `playwright` and Chrome; it starts and kills its own servers):

```
python3 dev-data/make_dev_data.py                       # regenerate synthetic data (optional)
python3 dev-data/check.py [--bundle] [--three LOCAL_three.module.min.js]
python3 dev-data/check3d.py [--per-suite N] [--only dev|pub|real]   # 3D check of every suite; screenshots to <out dir>
```

## Publish (Artifact)

Publish `index.html` with capabilities `{db: {}, user: {}}`, and the other files beside it, keeping paths:
`js/**/*.js` and `data/**`. Module paths are relative, so nothing else changes. Limits: 250 files
and 60 MB per publish; case files are fetched lazily, one per opened case.

If the host's CSP refuses module scripts, run `./bundle.sh` and publish `index.bundle.html` as the page
instead (still with `data/`). three.js is fetched from cdnjs at runtime in both (dynamic `import()` of a Blob URL after an SRI-checked `fetch`).

## Owner verdicts

`claude.use("db")` and `claude.use("user")`; either may be `null` (local, signed out), then the verdict UI is
disabled and nothing else changes. Collection `verdicts`, one document per `<suite>__<case_id>` (characters
outside `A-Za-z0-9_-.~:@+` become `~hex`), fields
`{verdict: "ok"|"wrong"|"unclear", note, case_id, suite, core_sha, schema_version, by, at}`. One write per
deliberate click on Save; one `onSnapshot` subscription for the whole list (never from render); `by` is the
user id and names are resolved with `user.profiles`. A verdict whose `core_sha` differs from the index's
`provenance.git_sha` is shown as STALE (never hidden) and can be re-recorded.

## A new producer

Write files that follow `tools/scenario-viewer/trace/schema/`, then add the suite to `index.suites` and its cases to
`index.cases` (or its tables to `index.tables`). No viewer change is needed for existing channel types.
Unknown channel types are listed in the inspector and ignored. A new channel type is an additive schema
change (minor bump); teach the viewer by adding a branch in `model/scene.js` (`inspect`, `at`) and, if it
should be plotted, `model/series.js`.

Conventions the viewer relies on (all from SCHEMA.md): actor roots come from `actors[].root_channel`; a
channel belongs to `channel.space` (falls back to its id prefix `source.` / `receiver.`); the viewer draws a
`relation_set` of a source-role space from `contact_p` / `look_at_p` and of a receiver-role space from
`rx_point_p`; `scope.in_scope.referent` / `dropped.referent` and `relation_set.referent` are matched to
`element.shared_id` (falling back to `element.id`).

## Renderer interface

Both renderers (`topdown2d`, `perspective3d`) implement exactly this; the stage talks to nothing else.

| call | meaning |
|---|---|
| `mount(el)` | attach to a container; the renderer owns its canvas, pointer and keyboard input |
| `setScene(scene)` | `scene = buildScene(caseJson)`; resets the camera to fit |
| `setTime(tick)` | tick index; fractional ticks only matter for channels marked `continuous` |
| `setLayers(set)` | `Set<string>`: overlay ids from `render/renderer.js` plus `space:<id>` for each visible space |
| `pick(x, y)` | CSS px relative to the mount -> `{space, kind: element\|actor\|proposal\|relation, id?, index?}` or null |
| `setCamera(cam)` | `{lock?, reset?/fit?, zoomBy?}`; 2D also `{scale, pan}`, 3D also `{az, el}` |
| `dispose()` | release listeners and GL resources |

Also implemented by both: `setHighlight(entity)` (correspondence hover) and `getInfo()` (panes, camera lock,
counters used by the headless check). Canvas colours are read from the computed CSS tokens and the stage
redraws on `prefers-color-scheme` and `data-theme` changes.

## Channel to drawing

| channel | 2D / 3D | plot / other |
|---|---|---|
| `pose3d` (actor root) | disc + heading tick / capsule at the SE(3) root; trail | inspector |
| `pose3d` (other) | replay ring, hollow marker | inspector |
| `proposal` | ghost disc with heading, status colour, dashed arrow from body / ghost capsule + dashed line | status strip, error columns, body-to-proposal distance (derived from two exported poses), HUD status chip + cause tags |
| `scope` | in-scope outline per family, dropped dimmed (reason on hover), container dashed, mode badge | mode strip, HUD |
| `relation_set` | Spacing/AffordancePose line to `contact_p`, Facing ray to `look_at_p` (vertex marked, occluded dashed), receiver line to `rx_point_p`; thickness = weight | R, weight, yaw/angle error plots |
| `scalar` / `vector` | | line plot (J, step ratio against its bound, others) |
| `enum` / `bitset` | | colour strip / inspector |
| `event` | | timeline marks, inspector |
| `joints` | bones + joints (3D, and projected in 2D) when `actor.skeleton` exists | inspector |
| `metrics_row`, `metrics`, `expectations`, `results` | | case header and results panel |

## Additive conventions rendered (generic, by id or field)

- `scope` channels whose id starts with `parity.` (no mode needed): a second dashed outline set per
  referent, labelled with the channel id, layer `Parity scope`; they never replace the main scope.
- `scope.border` per tick: the container outline is solid when 1, dashed when 0 or absent (2D; 3D uses the
  container colour only).
- Scalars `parity.*` and `eval.*`: one plot panel per group, one line per channel, named by id.
- `pose3d` channels that are not an actor root (parity.v1.replay, eval.*.receiver_body, receiver.replay...):
  distinct hollow markers + trails per channel, legend by id under the stage, layer `Extra poses + trails`.
- `channel.bounds[{value,label}]`: dashed reference lines on that channel's plot (step ratio's 1.5).
- Cases with `n_ticks = 0` or no spaces show header + results only; one-space cases show one pane;
  `dt_s >= 1` labels the axis `pose #` instead of seconds. `event` channels may have `space: null`.
- Index tables with text cells (eval-*) and guards tables are plain virtualized tables.
- Referent ids are shared ids; `scope.container` is a local element id of the channel's space;
  `contact_p`/`look_at_p` are source-world, `rx_point_p` receiver-world; actor root is at the feet; ground
  polygons are z = 0.

Real-data check: `dev-data/make_real_subset.py EXPORT_DIR 10` copies ~10 real cases into the untracked
`dev-data/real/data` (with a minimal index) and `dev-data/check_real.py` opens each in 2D and 3D.

## Character glyph, walk cycle, gaze (2D; derived for readability, not data)

- Glyph at each actor root: shoulders (rounded ellipse across the heading), head slightly forward with a visor
  notch along the yaw, two feet. Size is a display convention: shoulder width = 2 x `actor.body` radius, else
  0.45 m; head diameter 0.2 m (else 0.22 x shoulders); never below 28 px, metric above. The stage legend says
  "glyph size, not data".
- Walk cycle: phase = (cumulative xy root path length + 0.2 m per radian of turning) / 0.7 m, precomputed per
  root channel, so a tick always shows the same pose (scrub, step and play agree; no interpolation). Feet
  swing +-0.25 m, scaled by speed, and stand together when the per-tick displacement is about 0. Constants live in
  `js/render/glyph.js`. The legend says "not a motion model". An actor with a joints channel keeps the skeleton and
  gets a root dot instead of the derived glyph.
- Gaze: a translucent wedge from the head along the actor yaw (+-30 deg, 1.5 m, display constants, fading). A
  Facing relation draws its ray from the head to `look_at_p` (source) or `rx_point_p` (receiver body) in the
  Facing colour, plus a small arc and label with the angle between wedge axis and ray (derived; the core's own
  `angle_err_rad` stays in the relation label). Layer `Gaze wedge + angle`.
- Roles: source actor solid; receiver body solid in the receiver colour; proposal = dashed ghost glyph, no fill, no
  walk; `*replay*` pose = dotted ghost; `person` elements that are not an actor's spawn element = neutral solid
  glyph. The dashed body-to-proposal arrow stays.
- Layer `Footprint marks (derived)` (default off): a foot mark every stride along the path, alternating L/R.
- Everything uses theme tokens and redraws on theme change. Nothing animates beyond the per-tick pose, so
  `prefers-reduced-motion` needs no special case. The 3D view draws a shaded humanoid instead (next section).

## Labels, default view, actor elements (2D)

- Labels go through a greedy placer (`js/render/labels.js`): several candidate positions, leader lines, priorities,
  the lowest priority is dropped when nothing fits; glyphs and pane chrome are obstacles, the Facing arc label is a
  placed label too. Toolbar `labels: all | key | none` (default key = actor names, top-weight relation d/R or
  R_rx/err, the gaze angle; persisted in localStorage when available). Elements get one name, `kind short-id`
  (full name on hover). Relation labels sit beside the line, never on it.
- Default view = the actors' root paths (source, body, proposal) plus the footprints of every referent in scope
  at any tick, padded, one metric scale for both panes (the smaller fit, capped at 150 px/m); falls back to the
  referents, then to the room. `Reset view` returns to it. Shoulders are floored at 28 px, so on long paths the
  glyph is larger than its metric size.
- An actor's own registration element (`actor.element`, flag `actor_element`; for old data a footprint-less person
  named `actor` in a space with an actor) is not drawn: its pose is the pose at export. It is listed in the
  inspector only.

## Compare mode (2D)

A case with any `eval.<engine>.receiver_body` pose3d channel (v1, v1eqw, ...) gets a `Compare` toggle, ON by default:
panes `source | receiver . v2 | receiver . v1 [| v1eqw]` at one metric scale (rows when narrow). The v2 pane keeps
everything it shows today; an engine pane shows only what its data has: the engine's body glyph (walk rule), its
trail, dashed footprint outlines and lines from the body to the receiver elements that share the source referents'
shared ids, and `R` per referent from `eval.<engine>.R_receiver.<ref>` when present. Nothing is invented (no v1
proposals, no scope). The pane header shows the engine's mean per-relation error `|R_src - R_rx|` with its origin
(`eval.<engine>.R_receiver.*` vs `eval.source.R.*`, or `receiver.relations err_rx` for v2) or "no per-relation data
for this engine". `Overlay v1 in v2 pane` adds each engine body as a dashed glyph in the v2 pane with a dashed line
and a placed distance label. Two plots: relation error per engine, and v2-engine body distance. Engine colours are
CSS tokens `--e-v2` (the receiver colour), `--e-v1`, `--e-v1eqw`, light and dark. The v2 body of an eval case
without a receiver actor is `receiver.body`. 3D and non-compare cases are unchanged.

## Known gaps

See the report that came with this tool; the main ones: `shape3d.mesh_asset` is ignored until a producer ships one,
table `guards` are shown as exported (their shape is not specified), and 3D has no text labels (hover gives names).

## 3D view (three.js 0.160.0, lit and textured)

Code: `js/render/perspective3d.js` (panes, cameras, scene), `three-loader.js` (lazy, SRI), `textures.js` (procedural canvas
textures and shared PBR materials), `furniture.js` (procedural furniture), `figure3d.js` (humanoid mesh); `js/figure/`
(`anthro.js` constants with their citation, `figure.js` display IK, pure math, `pose.js` chooses the source of a pose).
Assets, citations and every display constant: `ASSETS.md`.

- **Loading.** three.js is not requested at page load. The first click on "3D perspective" fetches
  `three.module.min.js` 0.160.0 from cdnjs with an SRI hash (`fetch` with `integrity`), imports it from a Blob URL, and keeps it for the
  session. The module build prints nothing to the console. One WebGL renderer and one canvas serve all panes; leaving 3D parks the renderer
  (context, textures kept), a case switch disposes that case's geometry, and every shared texture/material is cached for the session.
- **Scene.** Textured floor per `room` element, inward-facing walls (display height 2.6 m; a wall between the camera and the view target is
  lowered to a curb), hemisphere fill + one shadow-casting directional light, ACES tone mapping, PBR materials, procedural 1 m textures
  (UVs are in metres, nothing is stretched). Layers: `Walls (3D)`, `Affordance parts (3D)`, `Anchors (3D)`, `Bones over the 3D figure`.
- **Cameras.** Orbit (drag, shift/right-drag pan, wheel, arrows, `0` resets) and follow (the pane's actor). Views are locked by default.
- **Furniture** (display-derived; labelled so in the HUD). One mesh per element, chosen by the part TAG vocabulary first (`seat`, `back`, `arm`, `writing_surface`, `keyboard_area`, `edge`, `shelf_surface`, `button_panel`, plus the earlier names `board`, `desk_surface`, `top`, `shelf_top`, `button`), the element name only picking the member of that family (`stool`, `armchair`, `blackboard` ...); with no usable part tag the element name, then its tags, decide
  (`chair`, `armchair`, `stool`, `bench`, `sofa`, `office chair`, `table`/`desk`, `whiteboard`/`blackboard`, `easel`, `flip chart`, `shelf`,
  `counter`, `panel`); any other element is its extruded footprint (flat if no height is exported). Footprint = `ground`, height =
  `shape3d.extrude_height_m`, seat/top height = the exported part plane when there is one, boards sit on their exported part (`element.parts`,
  schema 1.1: the part's own 4x4, plane z = 0, normal +z). Heights used only when nothing is exported are authored display values
  (`furniture.js` DISPLAY, listed in `ASSETS.md`). Facing of a seat: the `back` part, else the exported `seat` part frame (+x), else the element rotation, else toward the room centre.
- **People.** A shaded humanoid. Source of the pose, in priority order, never mixed within a tick:
  1. a `joints` channel (22 SMPL-family names; merged from the case or from a sidecar, see below). The label comes from the channel's own
     `derived` / `producer` / `origin`: `derived: "motiongen"` (or a producer named motiongen) shows "MotionGen output" with the producer
     name/version, never "core"; `recorded` shows "recorded motion";
  2. display IK from exported `pose3d` anchors (`source.anchor.<name>`, `receiver.anchor.<name>.proposal`, `<name>` in pelvis, hand_r, hand_l;
     `.engaged` 0/1): pelvis placed at the anchor (seated when `.engaged` is 1), hands reached by two-bone IK, leaning the torso by the smallest angle that reaches them (the lean is shown in the posture line). Labelled **"display-derived (IK),
     not model output"** on the canvas HUD, in the pane tag and in the inspector, with any reach shortfall in mm. The trace is untouched;
  3. otherwise a standing figure at the actor root (labelled so).
  Segment lengths are fractions of stature from Winter, Fig. 4.1 (`ASSETS.md`). There is no default stature: it is the typed "stature m", else a
  stature stated by the case, else derived from the standing pelvis height / 0.530 (labelled so), else a capsule marker labelled "stature unknown -
  enter a value". When seated on an engaged pelvis anchor (the sit-bone contact) the display pelvis joint is
  the anchor minus R(yaw) x the registered sit-bone offset: read from the case's `embodiment` (`anchors[{id:"pelvis", offset}]`), else from the interim
  fallback `embodiments/smplx.json` (measured on the SMPL-X neutral body, see `ASSETS.md`; the label says "interim display fallback"); no constant lives in code.
  Publish `embodiments/` beside `index.html`. MotionGen joints figures are used as given.
- **Sidecars (contract 4).** (Only `joints` channels are merged; other channel types are refused with a warning that lists them. Everything
  merged from a sidecar is labelled "MotionGen output" whatever `derived` says. `n`, `joint_count`, array lengths, `horizon` and `buffer_revision`
  are validated, `horizon` is floored, and a sidecar over 64 MiB is refused.) An index case entry may list `sidecars: [{file, producer}]` (file relative to `index.json`). The loader merges the
  sidecar's channels into the case (keeping each channel's `producer`/`derived`/`origin`), gives an actor its skeleton when it has none and its
  `joints_channel` when it has none. Unknown fields are ignored; a missing or malformed sidecar is a banner warning, never a failure.
  `serve-local.sh` serves sidecar files next to the case files, and `-s DIR` attaches a sidecar fragment (`DIR/index-sidecars.json` + the files) to the merged index.
- **Look-ahead (MotionGen).** A joints channel whose id ends in `.lookahead` (same layout as `joints`) is drawn as a translucent ghost labelled
  "look-ahead buffer (MotionGen, not yet played)": at tick k the ghost is the lookahead channel at tick k + `horizon` (`data.horizon`, a
  number or one value per tick; 0 when absent); `data.buffer_revision` (number or per tick) is shown next to the label. Absent fields are ignored.
- **Repeated furniture.** Geometry and material caching is used instead of `InstancedMesh`: primitive geometries are cached per case (identical
  chairs share them) and textures and materials per session. A piece of furniture is several parts, so instancing would need one instanced mesh per
  part; with the element counts of the suites (tens per room) the draw-call count is not the limit.
- **Compare.** Panes side by side: Source | receiver v2 | receiver v1. The v1 pane exists only when the case has an `eval.v1.receiver_body`
  channel (it shows a standing display figure at the v1 root: no joints or anchors exist for v1); otherwise the HUD says "receiver v1: not
  available". The Compare checkbox removes the v1 pane. The viewer never computes pass/fail; the producer's verdicts are shown unchanged.
- **Check.** `dev-data/check3d.py` (Playwright, no GUI) serves with `serve-local.sh -p 8791` (suites with clashing ids are served in
  turn), renders sample cases of every suite in 3D (dev data, `dev-data/fixtures-affordance`, and with `--data-dir <data dir>` the local
  exports `<data dir>/pub/data`, `<data dir>/real`, `<data dir>/affordance` when present), asserts zero console errors and a non-blank canvas, checks lazy loading, SRI, the
  labels, the compare panes and GL resource behaviour, and writes screenshots to `--out` (default: a temporary directory; never into the repository:
  `real/` is dataset-derived). `dev-data/make_fixtures_affordance.py` regenerates the SYNTHETIC schema-1.1 fixtures (needs node).
