# spatialblend.scenario-trace, schema v1.1.0

The official format for verifying scenario behaviour. A producer (the core scenario harness today, MotionGen
later) writes it; the viewer in `tools/scenario-viewer/` reads it. The core's own computed values are the
single source of truth: a consumer never re-implements core geometry or recomputes pass/fail.

Files: `scenario-trace.schema.json` (case file), `channel-types.schema.json` (the closed channel-type
registry), `index.schema.json` (index). `validate.py` checks files against them (stdlib only).

Versioning: semver, as for the ABI. Additive change (new optional field, new channel type) = minor.
Breaking change = major. A reader accepts any `1.x` and ignores unknown fields and unknown channel types
(it lists them).

## Conventions

- World frame: right-handed, z-up. Metres, radians, seconds. Planar producers write z = their height and a
  yaw-only quaternion; consumers never assume planarity.
- Pose: position `[x,y,z]`, unit quaternion `[qx,qy,qz,qw]`.
- One time base per case: the tick index; `t = t0_s + tick * dt_s`. Every channel is on it (`time_base: "tick"`).
- JSON has no NaN: a value the producer did not export is `null` (scalars) or `valid:0` (poses). A viewer
  draws nothing for it.
- Numbers are rounded by the producer (mm for metres, 0.1 degree for angles; weights and R to 1e-6).
- Channels are column-wise arrays. Variable-length per-tick data (relations, scope members) is CSR:
  `offsets` has `n+1` entries, tick `t` owns rows `offsets[t] .. offsets[t+1]-1` of every column.

## Files

`index.json`
```
{format:"spatialblend.scenario-trace.index", schema_version, provenance, suites:[{id,title,description,source,kind}],
 cases:[{id,suite,title,rule,file,bytes,git_sha,has_trace,pass,families,modes,statuses}],
 tables:[{id,suite,title,file,columns,rows,source,trace_case}], excluded:[...], missing:[...]}
```
`pass` is the producer's verdict (true / false / null = no assertion). `trace_case` names the table column
holding the case id of a row that has a trace.

Table file (`format: "spatialblend.scenario-trace.table"`): `{schema_version, columns:[..], rows:[[..],..], guards?: {...}}`.
The matrix summary shards are tables.

Case file (`format: "spatialblend.scenario-trace"`):
```
schema_version, synthetic?, provenance{producer{name,version},repo,git_sha,command,generated_at,schema_version},
case{id,suite,title,spec{rule,citations,source,level,plan_key},tags},
time{base:"tick",n_ticks,dt_s,t0_s},
spaces[{id,role,navigable_root,free_space{outer,holes,origin},elements[...]}],
correspondence[{source_space,receiver_space,source,receiver,shared_id}],
actors[{id,space,element,role,root_channel,body,skeleton,joints_channel}],
channels[{id,type,space,actor,time_base,continuous,unit,doc,origin,data}],
expectations[{id,kind,tick,family,expected,text}], results[{expectation,actual,pass,test,note}],
metrics{row{...},guards[{column,bound,value,pass}]}, missing[...]
```
Element: `{id, shared_id, name, kind, parent, flags[], world_transform(4x4 row-major, as the app gave it),
footprint_local[[x,y]], ground[[x,y]] (the core's GroundFrame image), shape3d{extrude_height_m|mesh_asset}|null}`.
`kind` in region, object, obstacle, interaction_region, person, border, room, surface, other; `flags` include
`transfer_leaf`, `occluder`, `seat`, `actor_element` (the actor's own registration person element; its pose is the pose at export, not where the actor is; a viewer draws no person glyph at it). `shape3d` is null unless the scenario declares a height (never invented).

Actor: `root_channel` names a `pose3d` channel. `body` (capsule) and `skeleton` / `joints_channel` are
optional; the core leaves skeleton and joints empty (a MotionGen sidecar may add them, see 1.1.0 below).

## Channel registry (closed; unchanged in 1.1.0)

| type | data | meaning |
|---|---|---|
| `pose3d` | `n,p[3n],q[4n],valid[n]?` | an SE(3) stream |
| `scalar` | `n,v[n]` | one number per tick (null = none); `unit` on the channel |
| `vector` | `n,dim,v[n*dim],names?` | a small fixed vector per tick |
| `enum` | `n,codes[n],labels[]` | a categorical stream |
| `bitset` | `n,v[n],bits[{bit,name}]` | cause bits / flags |
| `event` | `events[{tick,kind,data?}]` | sparse marks |
| `scope` | `n,mode{codes,labels},container[n],transfer[n],yaw_source?,families[],in_scope{offsets,family,referent},dropped{offsets,family,referent,reason,reasons[]}` | the source's resolved scope per tick |
| `relation_set` | CSR; columns `family,referent,weight,R_src,d_m,contact_p,look_at_p,look_at_vertex,angle_err_rad,occluded,R_rx,err_rx,R_rx_body,rx_point_p` | the relations of the payload |
| `proposal` | `n,status{codes,labels},cause{v,bits},flags{v,bits},p[3n],q[4n],valid[n],err{...}` | the receiver proposal set (root) |
| `joints` | `n,joint_count,p[3*n*J],q[4*n*J]` | RESERVED: skeleton joints per tick, tick-major |
| `metrics_row` | `row{...}` | a static row |

`continuous: true` on a channel lets a viewer interpolate between ticks for display only (labelled); the
default is no interpolation. Channels the core exports (ids are stable):

`source.root` (pose3d), `source.sent` (enum: SENT/HELD, reason in `source.gate_reason`), `source.scope` (scope),
`source.relations` (relation_set, the payload as the source saw it), `receiver.body` (pose3d, the embodied
sample the proposal was computed from), `receiver.body_after` (pose3d, the driver's pose after the step),
`receiver.proposal` (proposal), `receiver.replay` (pose3d, valid only where the proposal is a replay),
`receiver.relations` (relation_set, receiver R / error), `receiver.solver.J`, `.J_before`, `.iters`, `.evals`
(scalar), `receiver.solver.stop` (enum), `receiver.step_ratio` (scalar), `receiver.events` (event: sends,
mode changes, WEIGHT_SWITCH, FACING_ROOT_SWITCH, TELEPORT_REQUEST, TRANSFER, status changes, guard failures).

## Identifiers, frames, conventions added in 1.0.0 (clarifications)

- `referent` columns (scope, relation_set) are SHARED ids; `scope.container` is a LOCAL element id of the channel's space.
- `contact_p` and `look_at_p` are in the source world; `rx_point_p` is in the receiver world.
- An actor root is the feet point (capsule base at the root z). `ground` polygons are on z = 0.
- `scope.border[t]` = 1 when the container is among the referents as its border relation.
- A channel may carry `bounds: [{value, label}]` (reference lines for a plot, e.g. the 1.5 step-ratio bound).
- Channels whose id starts with `parity.` or `eval.` are producer-specific comparison channels (for example
  `parity.v1.scope`, the set v1 answers; `parity.agree`; `eval.v1.receiver_body`). A viewer draws them generically
  by type; producers write them only in their comparison suites.
- Unit cases of the scope catalog have `dt_s = 1` (one tick per pose). A case may have no receiver space, one space,
  or `n_ticks = 0` (spec only; the results carry the test's verdict).
- A case's overall pass is the index entry's `pass` (true when every evaluated result passed, null when none).

## Expectations and results

`expectations` are structured assertions (`scope_set` with `family` and `expected` ids, `scope_mode`, `status`,
`yaw_source`, `guard` with a bound, `boolean`, `text`). `results` are the producer's evaluation: `actual`,
`pass` (true, false, or null = not evaluated) and `test` (the test that checked it). A viewer displays them and
never recomputes them.

## Added in 1.1.0 (additive: a 1.0.x reader ignores all of it)

Source: the AffordancePose scenarios (suite `affordance`) and the motion
generator that conditions on them (`scope_context.py`).

**Element parts.** `element.parts`: `[{name, tag, world_transform, footprint_local, origin}]`. A part is a named patch of
the element's spatial definition (an affordance part: a seat, a board, a desk surface). `world_transform` is the 4x4
row-major transform of the part frame to the world: the part's plane is its local `z = 0` and its normal is its local
`+z`, so a part may be horizontal, vertical or tilted. `footprint_local` is the part polygon in the part frame (round and
non-convex polygons allowed). `origin` is `core` when the exporter read the part from the core's `SpatialDefinition`
(the affordance suite) and `scene-spec` when it is the scenario's declaration.

**Part vocabulary.** Part `tag`s of the affordance suite are from a closed, declared vocabulary: `seat`, `back`, `arm`, `writing_surface`, `keyboard_area`, `edge`, `shelf_surface`, `button_panel`; `name` may carry an index. A consumer does not infer correspondence from equal names.

**Anchor channels.** Body anchors, `<name>` in `pelvis`, `hand_r`, `hand_l`:

| channel | type | meaning |
|---|---|---|
| `source.anchor.<name>` | `pose3d` | the anchor's world pose as the producer stated it; `valid` where it did (the affordance tracks are synthetic, the channel `origin` says so) |
| `source.anchor.<name>.engaged` | `enum` (`FREE`, `ENGAGED`) | the core's contact rule on the anchor's AffordancePose relation (`IsAffordanceContact`, `solver/engagement.h`) |
| `source.anchor.<name>.part` | `enum` | the named part of the engaged relation (label 0 = none) |
| `receiver.anchor.<name>.proposal` | `pose3d` | the receiver's proposal target for the anchor, `valid` only where the proposal carries a target (status EXACT or INEXACT). The receiver driver does not realize hand proposals |
| `receiver.anchor.<name>.engaged` | `enum` (`FREE`, `ENGAGED`) | `kAnchorFlagEngaged` of the anchor proposal |
| `receiver.anchor.<name>.status` | `enum` | the anchor proposal's status |
| `receiver.anchor.<name>.part` | `enum` | the sub-region name of the payload the target was grounded on |

**Skeleton and joints.** `actor.skeleton` is `{id, joints: [{name, parent}], units: "m", frame: "world z-up"}` with `parent` a
joint index or -1 and parents before their children (the 1.0.0 form, string `joints` with a parallel `parents` array, is still
accepted). The `joints` channel is as in the registry (`n, joint_count, p[3*n*J], q[4*n*J]`, tick-major), with the channel
fields `producer: {name, version, git_sha}`, `origin` and `derived`: `motiongen` (a MotionGen output, never core output),
`display-ik` (a viewer-side display pose) or `recorded`. The core exporter writes no skeleton (`actor.skeleton` is null) and never
a non-empty joints channel.

**Sidecars.** An index case entry may carry `sidecars: [{file, producer}]`. A sidecar file is
`{format: "spatialblend.scenario-trace.sidecar", schema_version, case_id, producer{name, version, git_sha, command, generated_at},
actors: [{id, skeleton}], channels: [...]}`; a viewer merges its channels into the case and keeps each channel's `producer`.
`validate.py` checks sidecars (skeleton form, joint counts, producer and derived on every joints channel).

**Expectations.** Expectation `kind: "affordance"`: the producer-owned expectations E1-E8 of the AffordancePose scenarios
one per anchor (`id` like `E4:hand_r`), `result.actual` an object
`{text, residue, unit}` with the exact residue. A viewer displays them and never recomputes them.

**Scope context.** `scope_context.py CASE.json` extracts `spatialblend.scope-context` v0, the conditioning context a motion
generator reads: the referents with their parts, and per tick the source's scope, the anchors and the engaged anchors. It adds
nothing the core did not export except the fields it lists under `derived`.

## Provenance

Every file carries producer name and version, repo, git SHA, command line, generation time and schema version.
A viewer warns when a case's `git_sha` differs from the index's.

## Honesty

A producer that does not know a field leaves it null or lists it under `missing` with the reason; it never
fills it in. Excluded data (for example dataset-derived) is listed in the index `excluded`.

## Added in 1.1.0 (embodiment registration, ABI 4.7; additive)

**`actor.embodiment`.** For an actor that registered an embodiment (`sb_register_embodiment_json`), the actor entry carries the registration the core held, verbatim
(`sb_get_embodiment_json`, canonical JSON): `{format: "spatialblend.embodiment", schema_version, id, units, frame, bone_frame,
skeleton: {joints: [{name, parent}]}, anchors: [{name, bone (joint name or null = the actor root), offset: {position [m],
rotation [x,y,z,w] in the bone's frame}, families: [Spacing|Facing|AffordancePose], provenance: {procedure, asset_sha256 | source, ...}}],
chain?: {normalization, rest_pose, provenance, limits_provenance?, joints: [{name, rest: {position, rotation}, limits?}]},
origin: "registered"}`. A non-identity offset always carries `provenance.procedure` and either a 64-hex `provenance.asset_sha256` or an explicit `provenance.source` (the core
refuses the registration otherwise). `actor.skeleton` is filled from the same registration (the joints CHANNEL is still empty:
the core writes no body stream). `chain` is the normalized kinematic chain: z-up, x-forward, y-left, right-handed, metres, radians,
absolute rest positions of each joint frame in the embodiment's own labelled rest pose, the first root joint at the origin; the
solver does not read it. Anchor channels `source.anchor.<name>` are the anchor poses the core composed from the bones and the
registered offsets. Viewers and motion generators read anchors, bones and contact offsets from here only; the scope context v0
carries it as `embodiment`.

**Changelog note.** The 1.1.0 fields above (`actor.embodiment`, the filled `actor.skeleton`, the scope context's `embodiment`) were added under 1.1.0 without a version bump: they are additive and a 1.1.0 reader that ignores unknown fields is unaffected. A reader that needs them checks for their presence.
