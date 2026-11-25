# SpatialBlend trace format (format 7)

The structured trace that `Space` writes: one JSON object per line
(JSON Lines, UTF-8), one event per line. This file is the normative
description of every event kind and field the core emits. It is read by
`spatialblend/api/c/trace_spec_conformance_test.cc`, which parses the field
tables below and checks every line of a traced run against them, so a field
added to the code without a row here (or the reverse) fails the test.

Covered: the trace is specified and versioned; it records the policy
parameters, the driving relation per channel and tick, the scope record,
witness points, the feasibility snap and the reach diagnostic, and the
mediator's paced state per tick (replaced by proposal in format 8). Source:
`spatialblend/space/space.cc`, `spatialblend/api/c/spatialblend_c.cc`
(`sb_set_trace`), `spatialblend/base/common/trace.h` (`kTraceFormatVersion`).

## Activation and routing

- `sb_set_trace(space, path)` binds one space to a file (append,
  line-buffered). Several spaces bound to the same path share one sink and
  their lines interleave; the `space` field tells them apart.
  `sb_set_trace(space, NULL)` disables that space.
- A space never bound falls back to the process-wide sink, activated by the
  environment: `SB_V2_TRACE_FILE=<path>` or `SB_V2_TRACE=1` (stderr).
  Environment-activated traces carry no `meta` line.
- Tracing off costs one atomic load per emit site; no JSON is built.

## Versioning

- The format version is an integer, `kTraceFormatVersion` in
  `base/common/trace.h`, also returned by `sb_get_trace_format_version()`. It is
  written in the `format` field of `meta` and `policy` lines.
- Any change to an event's field set, a field's type or its meaning bumps the
  version and adds a row to the history below.
- Consumers must ignore event kinds they do not know. Within a known kind
  the field set is exact for a given format version.
- Guard: the conformance test hashes the field tables below (kind, field,
  type) and compares the hash with the one pinned for the current
  `kTraceFormatVersion`. Changing an emitted field therefore needs three
  edits together: the code, this file, and a version bump with a newly
  pinned hash.

| Format | Change |
|---|---|
| 1 | `src`, `payload`, `target`, `relation`, `share`, `tick` (no version field on any line). |
| 2 | Adds `meta`, `policy`, `channel`, `goal`, `scope`. `payload` gains `sub_region`, `driving`. `target` gains `snap_m`, `snapped`, `requested_pos`, `reach_clamped`, `reach_m`. `relation` gains `referent`. |
| 3 | Adds `mediator`: one line per `sb_tick_avatar` with the paced pose, mode, remaining budget, course and teleport state. `target.snap_m` / `snapped` now report the free-space projection (walkable minus body-radius obstacle clearance) instead of the walkable-polygon clamp. |
| 4 | `scope.reason` gains `beyond_scope_radius`; `not_visible`, `unreachable`, `outside_near_field` and `gated_inactive` are now produced by the scope-gating stage instead of reserved. `payload.dist` is now also set for AffordancePose (anchor-to-support distance, the near-field measure) instead of `-1`. |
| 5 | `target` drops `requested_pos`, `reach_clamped`, `reach_m` (no reach clamp exists any more) and gains `beyond_reach`, `reach_excess_m`: an engaged anchor target is never moved by reach; one farther than the anchor's reach from the solved root is flagged. A `kFailed` realization now carries no root target but may carry engaged anchor targets. |
| 6 | Adds `qualify`: one line per seat-anchor entry-qualification sample, with the stillness speed now COUCH's smoothed estimator on trailing samples (`speed`, `window_s`, `baseline_s`, `span_s`, `sparse`) instead of the per-frame difference. |
| 7 | `target` gains `objective_before`, `objective_after`, `objective_iterations`, `objective_fallback`: the paper objective (SpatialBlend App. A eq. weighted, Σ w_p (R^tgt_p(S)-R^src_p)^2) at the goal-point initializer and after the bounded local refinement that now runs on top of it, root target only (same root-only scope as `snap_m` / `snapped`; anchor target lines carry the zero/false defaults). |
| 8 | Adds `proposal`: one line per receiver proposal (what core proposes for the root, with its cause bitmask and the identity/age of the embodied sample it was computed from). `mediator` is removed: core no longer paces (a driver trace carries the paced pose). |
| 9 | ABI 4: `qualify` is removed (the seat is geometric contact: no stillness qualification). `proposal` drops `budget_s` (pacing belongs to the driver), reports `engaged` as the geometric fact (no hysteresis) and gains `anchors`: one entry per relation-bearing anchor other than the root (pelvis, head, ...), the root being the line's own `pos` / `yaw` / `cause`. `scope.reason` drops `beyond_scope_radius`, `outside_near_field` and `gated_inactive` and gains `outside_parent_region` (family scope D1-D9, no radii or activation gate). |
| 10 | `scope.reason` gains `inside_transfer_leaf`: a Facing binding whose anchor pose is inside a containment-transfer element excludes that element and its non-Surface descendants (ScopeResolver::ExcludedFromFacingInsideTransfer). |

Consumers written for format 1 that match the substring `"referent":<id>`
must now also filter on `ev`, because `relation`, `goal`, `channel` and
`target` lines carry referent ids too.

A trace without any `meta` or `policy` line is format 1.

## Conventions

- **Frame:** Z-up world frame of the emitting space (`x` right, `y` forward,
  `z` up). Positions in metres, angles in radians, yaw about +Z.
- **Time `t`:** seconds on the emitting space's clock. A source space uses
  its observation time for timestamped frames and its accumulated update
  clock otherwise; a receiver uses its own clock, which advances by the real
  source-clock gap between deliveries. `t` is non-decreasing per space in file
  order. `meta` has no `t`.
- **`space`:** the emitting space's id. **`actor`:** the actor's shared id
  (the local element id when the actor's element has no shared id).
- **`referent`:** the referent's shared id as carried in the payload (the
  local element id when the element is not shared). **`*_elem` / scope
  ids:** local element ids of the emitting space, `-1` = none.
- **`kernel`:** integer, internal `RelationKernelType`: `0` facing,
  `1` spacing, `2` affordance pose. (This differs from the C enum
  `sb_kernel_type_t`, where spacing is 0 and facing is 1.)
- **`anchor`:** never empty; the root anchor is written `"root"`.
- **Channel:** one (kernel, anchor) pair. Weights compete only inside a
  channel.
- Types in the tables: `int` (JSON integer), `number` (integer or float),
  `bool`, `string`, `number[2]` / `number[3]` (fixed-length arrays),
  `array` (see the meaning column), `X|null`.

## Event order within one update

A **source** update (`sb_update_actor_state` or an observation frame) writes
`relation` lines for phase transitions, then one `scope` line per binding
with an active anchor (preceded once per revision by a `policy` line; written
after the payload is built so late drops are included), then one `src` line
when the update is propagated.

A **receiver** evaluation (a delivered peer update, or
`sb_inject_relation_payloads`) writes `payload` lines (one per received
payload, including unresolved ones), a `policy` line when the policy or
imminence revision changed since it was last traced, `relation` lines for
phase transitions, then -- when the evaluation reaches a solve -- one
`channel` line per channel, one `goal` line per relation that entered the
root blend, one `target` line per realized target, and a closing `tick`. A
receiver-input reevaluation (`sb_reevaluate_receiver_realization`) writes
nothing.

A **proposal** (every receiver solve, including one that cannot run for lack of
an embodied sample, and a release) writes one `proposal` line.

## Event kinds

### `ev=meta`

Written by `sb_set_trace(space, path)` when it binds a path. Describes the
format; has no `t`.

| Field | Type | Unit / frame | Meaning |
|---|---|---|---|
| `ev` | string | | `"meta"` |
| `space` | int | | space id |
| `format` | int | | trace format version (3) |
| `spec` | string | | path of this file in spatialblend-core |
| `frame` | string | | `"z_up"` |

### `ev=policy`

The receiver policy family and every declared parameter. Written by a
space the first time it traces after `sb_set_trace`, and again whenever its
policy revision changes (`sb_set_realization_policy`,
`sb_set_realization_policy_params`, `sb_set_anchor_gating`,
`sb_set_anchor_reach`). Receiver temperature schedule (SpatialBlend Eq. A6):
`eta = clip(eta_0 * (T_min + epsilon_t)^gamma, eta_min, eta_max)`. Source
imminence: `R_src = clip(1 - (1 - R)/kappa, 0, 1)`,
`T_p = clip((1 - R_src) / max(EMA_tau(dR_src/dt)+, epsilon_v), 0, t_max)`.

Production defaults: `gamma = 2` (the paper's schedule is `gamma = 1`; a
recorded deviation, see `ProductionRealizationPolicyConfig` in
`solver/realization_policy.h`), `eta_0 = 1`, `epsilon_t = 1e-3`,
`eta_min = 1e-6`, `eta_max = 100`, `progress_tau_s = 0.092517`,
`epsilon_v = 0.01`, `t_max = 100`, `kappa = 0.2`. With these values
`eta_min` equals `epsilon_t^2` and never binds, and `eta_max` binds only when
`T_min` exceeds about 10 s: the clip is effectively unbounded below.

| Field | Type | Unit / frame | Meaning |
|---|---|---|---|
| `ev` | string | | `"policy"` |
| `t` | number | s | space clock |
| `space` | int | | space id |
| `format` | int | | trace format version |
| `revision` | int | | the space's receiver policy revision (bumped by the family, the temperature schedule, anchor gating and reach) |
| `imminence_revision` | int | | the space's imminence revision (bumped only by the source-side constants) |
| `policy` | string | | `"progressive"`, `"uniform"` or `"greedy"` |
| `eta_0` | number | s | temperature scale |
| `gamma` | number | | annealing exponent |
| `epsilon_t` | number | s | T_min offset |
| `eta_min` | number | s | lower temperature clip |
| `eta_max` | number | s | upper temperature clip |
| `progress_tau_s` | number | s | EMA time constant of dR_src/dt |
| `epsilon_v` | number | 1/s | approach-rate floor |
| `t_max` | number | s | imminence cap |
| `kappa` | number | | engagement scale on R |

### `ev=scope`

Source side, one line per binding with an active anchor per evaluation.
Lists every element the kernel's scope policy produced and what
happened to it, in scope-policy order (an explicit leaf scope's
self-candidate last). An element absent from both lists was never in the raw
scope. The same record is readable through `sb_get_scope_records` /
`sb_get_scope_candidates`.

| Field | Type | Unit / frame | Meaning |
|---|---|---|---|
| `ev` | string | | `"scope"` |
| `t` | number | s | source time of the evaluation |
| `space` | int | | source space id |
| `actor` | int | | actor shared id |
| `anchor` | string | | binding anchor |
| `kernel` | int | | binding kernel |
| `explicit` | bool | | bound with `sb_bind_scoped` |
| `scope_root` | int | local element id | explicit root, or the actor's nearest containment ancestor |
| `deepest` | int | local element id | deepest containment element holding the anchor (`-1` when the scope is gone) |
| `navigable` | int | local element id | element whose navigable geometry normalized R (`-1` = none) |
| `included` | array | referent ids | evaluated referents, in scope-policy order |
| `excluded` | array | `[referent id, reason]` pairs | reason: see Vocabularies (`scope.reason`) |

### `ev=src`

Source side, one line per propagated update.

| Field | Type | Unit / frame | Meaning |
|---|---|---|---|
| `ev` | string | | `"src"` |
| `t` | number | s | source clock of the propagation |
| `space` | int | | source space id |
| `actor` | int | | actor shared id |
| `pos` | number[2] | m, world | root position |
| `yaw` | number | rad | root yaw |

### `ev=payload`

Receiver side, one line per received payload, after the policy has weighted
the resolved ones.

| Field | Type | Unit / frame | Meaning |
|---|---|---|---|
| `ev` | string | | `"payload"` |
| `t` | number | s | receiver clock |
| `space` | int | | receiver space id |
| `actor` | int | | actor shared id carried by the update |
| `referent` | int | | referent id carried by the payload |
| `anchor` | string | | subject anchor |
| `kernel` | int | | relation kernel |
| `R` | number | [0, 1] | source relation value |
| `dist` | number | m | source support distance (`-1` = unset) |
| `depth` | int | | tree depth of the resolved local referent (0 = unresolved) |
| `weight` | number | [0, 1] | policy weight (0 when unresolved) |
| `est_time` | number | s | imminence T_p carried by the payload |
| `via` | string | | `"linker"`, `"sharedid"` or `"unresolved"` |
| `sub_region` | string | | engaged sub-region name, `""` = whole boundary |
| `driving` | bool | | this payload is its channel's POLICY argmax (before the solver and the anchor gate; the realized driving relation is on `channel`) |

### `ev=channel`

Receiver side, one line per channel per evaluation that reaches a solve.
It names two relations: the POLICY argmax (`driving_*`, the
max-weight payload) and the REALIZED driving relation (`realized_*`): the
max-weight goal that entered the root blend (root channels), or the payload
of the target's referent when an anchor target survived the anchor gate (anchor
channels). They differ when the argmax was unresolvable, had zero weight, or
its anchor target was gated out.

| Field | Type | Unit / frame | Meaning |
|---|---|---|---|
| `ev` | string | | `"channel"` |
| `t` | number | s | receiver clock |
| `space` | int | | receiver space id |
| `actor` | int | | actor shared id |
| `anchor` | string | | channel anchor |
| `kernel` | int | | channel kernel |
| `policy` | string | | policy family used |
| `n` | int | | enabled payloads in the channel |
| `t_min` | number\|null | s | min valid T_p (progressive only) |
| `eta` | number\|null | s | temperature (progressive with n >= 2) |
| `driving_referent` | int\|null | | referent with the maximal weight (ties: first received) |
| `driving_weight` | number | [0, 1] | its weight (0 when none) |
| `driving_R` | number\|null | [0, 1] | its R |
| `driving_est_time` | number\|null | s | its T_p |
| `tied` | bool | | another payload has the same maximal weight |
| `realized_referent` | int\|null | | referent of the relation that drove the channel's realized target (null: no realized target) |
| `realized_weight` | number | [0, 1] | its weight (0 when none) |
| `realized_is_policy_argmax` | bool | | the realized driving relation is the policy argmax |

### `ev=relation`

One line per relation phase transition, on both source and receiver.

| Field | Type | Unit / frame | Meaning |
|---|---|---|---|
| `ev` | string | | `"relation"` |
| `t` | number | s | time of the transition |
| `space` | int | | emitting space id |
| `actor` | int | | actor shared id |
| `relation` | string | | `"<kernel name>:<anchor>"`, kernel name `facing`, `spacing` or `affordance_pose` |
| `phase` | string | | `"approach"` (to active or driving), `"engage"` (to contact), `"release"` (to dormant) |
| `tracker_phase` | string | | internal state entered: `dormant`, `active`, `driving`, `contact` |
| `R` | number | [0, 1] | relation value at the transition |
| `referent` | int | | the relation's referent |

### `ev=goal`

Receiver side, one line per relation that entered the root blend (non-zero
weight), with its decoded witness point before blending. Readable as
`sb_get_solver_goals`.

| Field | Type | Unit / frame | Meaning |
|---|---|---|---|
| `ev` | string | | `"goal"` |
| `t` | number | s | receiver clock |
| `space` | int | | receiver space id |
| `actor` | int | | actor shared id |
| `anchor` | string | | relation anchor |
| `kernel` | int | | relation kernel |
| `referent` | int | | referent id |
| `witness` | number[2] | m, world | point that alone preserves this relation's R |
| `weight` | number | [0, 1] | blend weight |
| `pos_goal` | bool | | contributes to the position blend |
| `yaw_goal` | bool | | contributes to the yaw blend |

### `ev=target`

Receiver side, one line per realized target (root first, then anchor
targets).

| Field | Type | Unit / frame | Meaning |
|---|---|---|---|
| `ev` | string | | `"target"` |
| `t` | number | s | receiver clock |
| `space` | int | | receiver space id |
| `actor` | int | | actor shared id |
| `anchor` | string | | target anchor |
| `pos` | number[3] | m, world | target position (`z = 0` unless `has_z`) |
| `has_z` | bool | | `pos[2]` is a real height (anchor targets) |
| `yaw` | number | rad | target yaw |
| `kernel` | int | | producing kernel |
| `referent` | int | | referent the target was computed against |
| `weight` | number | [0, 1] | target weight |
| `status` | string | | `"converged"`, `"fallback"` or `"failed"` |
| `snap_m` | number | m | root target: distance the feasibility clamp moved the blend (0 for anchor targets) |
| `snapped` | bool | | root target: the clamp fired |
| `beyond_reach` | bool | | anchor target: the engaged support lies beyond the anchor's reach from the solved root (the target is not moved) |
| `reach_excess_m` | number | m | horizontal distance beyond the reach (0 when within reach) |
| `objective_before` | number | | root target only: the paper objective Σ w_p (R^tgt_p(S)-R^src_p)^2 (SpatialBlend App. A eq. weighted) at the goal-point initializer, R^tgt_p evaluated with the same per-family evaluator as target evaluation; `0` for an anchor target |
| `objective_after` | number | | root target only: the objective of the state actually emitted -- the refined state when the bounded local search strictly improved on the initializer, else the initializer itself (so always `<= objective_before`); `0` for an anchor target |
| `objective_iterations` | int | | root target only: the local search's iteration count (`0` when no relation was evaluable, or the first step already failed to improve); `0` for an anchor target |
| `objective_fallback` | bool | | root target only: the search made no improving move, so the initializer was kept unchanged; `false` for an anchor target |

### `ev=tick`

Receiver side, closes each evaluation.

| Field | Type | Unit / frame | Meaning |
|---|---|---|---|
| `ev` | string | | `"tick"` |
| `t` | number | s | receiver clock |
| `space` | int | | receiver space id |
| `cost_us` | int | µs, wall clock | compute cost of the evaluation (the only wall-clock field) |

### `ev=share`

One line per resolved shared element (`sb_resolve_shared_element`).

| Field | Type | Unit / frame | Meaning |
|---|---|---|---|
| `ev` | string | | `"share"` |
| `t` | number | s | space clock |
| `space` | int | | resolving space id |
| `shared_id` | int | | shared id of the element |
| `received` | int | local element id | temporary element created on receipt |
| `resolved` | int | local element id | element the share resolved to |
| `resolution` | string | | `"add_as_new"` or `"link_to_existing"` |

### `ev=proposal`

Receiver side, one line per receiver proposal: written
when a source-triggered solve publishes the actor's proposal
(`sb_read_proposals`). Core proposes the relation-preserving pose of every
relation-bearing anchor and the attribution; the driver applies them within its
own kinetic bounds and publishes its poses back through the sampling ports, so
there is no paced pose in core and no core-side mediator trace.

Statuses: `exact` (an exact minimiser), `inexact` (best minimiser, residual
above epsilon), `hold` (infeasible: the target is the sampled pose),
`no_sample` (the driver has published no embodied pose: nothing was solved and
no pose is substituted), `none` (the source released every relation).

| Field | Type | Unit / frame | Meaning |
|---|---|---|---|
| `ev` | string | | `"proposal"` |
| `t` | number | s | receiver clock |
| `space` | int | | receiver space id |
| `actor` | int | | actor shared id |
| `status` | string | | proposal status (see above) |
| `source_seq` | int | | the source update this answers (`0` = hand-built) |
| `sample_seq` | int | | sequence of the embodied sample it was computed from (`0` = none); the sample AGE is wall-clock and is not traced (it is in `sb_read_proposals`) |
| `pos` | number[2] | m, world | proposed root position |
| `yaw` | number | rad | proposed root yaw |
| `cause` | int | | cause bitmask: why this proposal differs from the previous one (`SB_PROPOSAL_CAUSE_*`) |
| `engaged` | bool | | the root is on, or touches, a non-containment referent (a geometric fact, no hysteresis) |
| `anchors` | array | | the other relation-bearing anchors of this set, one object each: `anchor` (name), `id`, `status` (number, `sb_proposal_status_t`), `pos` (`[x, y, z]`), `yaw`, `cause`, `engaged` |
| `err_spacing_target` | number | R | max Spacing `|R_tgt - R_src|` at the target; `-1` = no Spacing relation |
| `err_spacing_sample` | number | R | the same at the SAMPLED embodied pose |

## Vocabularies

String values allowed per event field (checked by the conformance test).

| Value of | Values |
|---|---|
| `meta.frame` | `z_up` |
| `policy.policy` | `progressive`, `uniform`, `greedy` |
| `channel.policy` | `progressive`, `uniform`, `greedy` |
| `payload.via` | `linker`, `sharedid`, `unresolved` |
| `relation.phase` | `approach`, `engage`, `release` |
| `relation.tracker_phase` | `dormant`, `active`, `driving`, `contact` |
| `target.status` | `converged`, `fallback`, `failed` |
| `share.resolution` | `add_as_new`, `link_to_existing` |
| `proposal.status` | `none`, `no_sample`, `exact`, `inexact`, `hold` |
| `scope.reason` | `explicit_scope`, `tag`, `actor_element`, `observed_point_control`, `observed_point_ineligible`, `missing_element`, `surface_prefilter`, `observed_point_expired`, `not_visible`, `unreachable`, `outside_parent_region`, `inside_transfer_leaf` |

`scope.reason` is the second element of each `excluded` pair.
The scope-gating stage produces `not_visible` (Facing: the one ray to the
look-at contact is blocked) and `unreachable` (Spacing: no geodesic path in
the source free space to the referent's support). `outside_parent_region` (AffordancePose: the anchor is outside the referent's
parent object region). `inside_transfer_leaf` (Facing: the anchor pose is inside a containment-transfer
element, and the candidate is that element or one of its non-Surface descendants; format 10). `beyond_scope_radius`,
`outside_near_field` and `gated_inactive` are retired with the radii, the
near field and the activation gate; their
numbers (13, 10, 11) are not reused.
`observed_point_expired` marks a referent that was evaluated and then dropped
while building the payload because its observed facing point expired.

## Not yet traced

These are observable only by polling:
the target-side R^tgt of a candidate state.
