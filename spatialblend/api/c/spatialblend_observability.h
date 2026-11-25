// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SPATIALBLEND_OBSERVABILITY_H_
#define SPATIALBLEND_C_SPATIALBLEND_OBSERVABILITY_H_

/*
 * Observability and injection surface for component testing (see the
 * per-section notes below).
 *
 * Additive only: no existing struct or function changes. Every struct here
 * starts with {struct_size, version}; callers set struct_size = sizeof(T)
 * (and version = SB_OBSERVABILITY_VERSION on inputs) and every function
 * refuses a size it does not know (returns -1, or 0 for the int32 "ok"
 * functions). Offsets can be checked field by field with
 * sb_get_struct_field_offset(); sizes with sb_sizeof().
 *
 * Threading: every getter takes the space lock and copies values that the
 * latest evaluation recorded (never a recomputation), and none of them
 * marks the space's scene inputs as escaped (receiver input and observed
 * facing points keep working). Called from inside an sb_on_relation_event_t
 * callback (the space lock is held there) the getters return -1 / 0 without
 * touching the space; sb_get_relation_states_v2 alone is allowed there.
 *
 * Capture is opt-in (per-tick recording allocates): the space records scope
 * records, policy observations and solver diagnostics while it is traced, or
 * after sb_set_observability_capture(space, 1). Every getter below turns
 * capture on for its space, so the first read on an untraced space can be
 * empty and every later evaluation is recorded.
 *
 * String pointers in outputs (anchor_name, tag_filter, sub_region_name)
 * point into thread-local storage owned by the library and stay valid until
 * the next call of the SAME function on the same thread.
 *
 * Z-UP throughout (see spatialblend.h). Units: metres, radians, seconds.
 */

#include <stddef.h>
#include <stdint.h>

#include "spatialblend_type.h"

#if defined(_WIN32)
#if defined(SPATIALBLEND_BUILD_DLL)
#define SB_OBS_EXPORT __declspec(dllexport)
#else
#define SB_OBS_EXPORT __declspec(dllimport)
#endif
#else
#define SB_OBS_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Version of the structs and semantics in this header. */
#define SB_OBSERVABILITY_VERSION 1u
/* Capacity of the fixed encoded-state arrays below. Kernels today encode at
 * most 4 values; encoded_state_size reports the real length. */
#define SB_MAX_ENCODED_STATE 8

/* Struct ids for sb_sizeof() / sb_get_struct_field_offset(): the
 * enumerators SB_STRUCT_RELATION_STATE (3) .. SB_STRUCT_V2_ABI_LAYOUT (16) of
 * sb_struct_id_t in spatialblend_type.h. */

/** Turn per-evaluation observation capture on (1) or off (0) for `space`.
 * 1 on success, 0 for an unknown space. Tracing captures regardless. */
SB_OBS_EXPORT int32_t sb_set_observability_capture(sb_space_t space,
                                                   int32_t enabled);

/** Returns SB_OBSERVABILITY_VERSION. */
SB_OBS_EXPORT uint32_t sb_get_observability_version(void);
/** Returns the trace format version written by this build (the
 * "format" field of ev=meta / ev=policy; see api/c/trace.format.md). */
SB_OBS_EXPORT uint32_t sb_get_trace_format_version(void);

/**
 * Number of fields of a struct id (declaration order), or 0 if unknown.
 * Covers every id listed above plus the legacy 0-2.
 */
SB_OBS_EXPORT uint32_t sb_get_struct_field_count(sb_struct_id_t struct_id);
/** Byte offset of field `field` (declaration order), or UINT32_MAX. */
SB_OBS_EXPORT uint32_t sb_get_struct_field_offset(sb_struct_id_t struct_id,
                                                  uint32_t field);

/* ==========================================================================
 * Relation state with support, encoded state and normalizing scope.
 * ========================================================================== */

typedef struct {
  uint32_t struct_size;
  uint32_t version;
  const char* anchor_name;
  /* Named sub-region (affordance part) the support is grounded on; "" when
   * grounded on the whole boundary. */
  const char* sub_region_name;
  double proximity; /* R in [0, 1] */
  /* Absolute distance to the support (m); < 0 when the kernel leaves it
   * unset. */
  double support_distance_m;
  /* Diameter of the navigable scope that normalized R (m); <= 0 unset. */
  double source_scope_diameter_m;
  double contact_point_world_x;
  double contact_point_world_y;
  double contact_point_world_z;
  double normalized_angle;
  double normalized_radial_ratio;
  double normalized_height_ratio;
  /* Interior chart coordinate (box-UVW fallback frame) when has_interior. */
  double interior_u;
  double interior_v;
  double interior_w;
  /* GroundedRelation::encoded_state[0..encoded_state_size). */
  double encoded_state[SB_MAX_ENCODED_STATE];
  int64_t referent_element_id;
  int64_t referent_shared_id;
  int64_t binding_index;
  sb_kernel_type_t kernel_type;
  sb_normalized_strategy_t normalized_strategy;
  sb_relation_phase_t phase; /* tracker phase behind `active` */
  int32_t has_contact;
  int32_t active;
  int32_t has_interior;
  /* Real length of the encoded state; may exceed SB_MAX_ENCODED_STATE, in
   * which case only the first SB_MAX_ENCODED_STATE values are copied. */
  int32_t encoded_state_size;
  /* Radial surface-chart version of normalized_* (0 and 1 = the pure
   * radial chart, the only chart; 2, the deleted regularized chart, is rejected).
   * Occupies the former reserved slot (same size and offsets), so a
   * read-then-inject round trip carries the chart it was encoded with. */
  uint32_t chart_version;
  /* AffordancePose only -- the anchor's pose relative to the
   * support, beyond normalized_angle/normalized_radial_ratio above: its
   * in-plane offset from the support point (support-local u/v) and its
   * height above the support plane (also support-local). NAN = unknown (a
   * pre-pose-fit sender, or any other relation family). `proximity` above stays
   * the GEOMETRIC proximity component 1/(1+distance) (the only input to the
   * contact-phase hysteresis); the pose-fit R is computed by Eval_p
   * (sb_evaluate_target_relation r_src/r_tgt), from these offsets.
   * A caller may pass struct_size = offsetof(sb_relation_state_v2_t,
   * ap_offset_u_m) (the pre-ap_offset layout): rows are then struct_size bytes
   * apart, the three fields are not written. */
  double ap_offset_u_m;
  double ap_offset_v_m;
  double ap_height_m;
} sb_relation_state_v2_t;

/**
 * Superset of sb_get_relation_states(): same rows, same order.
 * @return total row count (may exceed cap; re-query), -1 on a bad
 *         struct_size / null out with cap > 0.
 */
SB_OBS_EXPORT int32_t sb_get_relation_states_v2(sb_space_t space,
                                                sb_actor_t actor,
                                                sb_relation_state_v2_t* out,
                                                int32_t cap,
                                                uint32_t struct_size);

/* ==========================================================================
 * Per-binding scope record.
 * ========================================================================== */

/* Why a referent the kernel's scope policy produced is in or out. Values 8, 9
 * and 14 come from the family scope gate: 8 the one ray to the
 * look-at contact is blocked (Facing), 9 no geodesic path in the source free
 * space (Spacing), 14 the anchor is outside the referent's parent object region
 * (AffordancePose). 10 (outside a near field), 11 (dormant) and 13 (beyond a
 * scope radius) are RETIRED with the near field, the activation gate and the
 * radii (ABI 4). Never renumbered. */
typedef enum {
  SB_SCOPE_INCLUDED = 0,
  SB_SCOPE_EXCLUDED_EXPLICIT_SCOPE = 1,
  SB_SCOPE_EXCLUDED_TAG = 2,
  SB_SCOPE_EXCLUDED_ACTOR_ELEMENT = 3,
  SB_SCOPE_EXCLUDED_OBSERVED_POINT_CONTROL = 4,
  SB_SCOPE_EXCLUDED_OBSERVED_POINT_INELIGIBLE = 5,
  SB_SCOPE_EXCLUDED_MISSING_ELEMENT = 6,
  SB_SCOPE_EXCLUDED_SURFACE_PREFILTER = 7,
  SB_SCOPE_EXCLUDED_NOT_VISIBLE = 8,
  SB_SCOPE_EXCLUDED_UNREACHABLE = 9,
  SB_SCOPE_EXCLUDED_OUTSIDE_NEAR_FIELD = 10,
  SB_SCOPE_GATED_INACTIVE = 11,
  /* Evaluated, then dropped at payload build because its observed facing
   * point expired in between (overwrites INCLUDED). */
  SB_SCOPE_EXCLUDED_OBSERVED_POINT_EXPIRED = 12,
  SB_SCOPE_EXCLUDED_BEYOND_SCOPE_RADIUS = 13
} sb_scope_reason_t;

typedef struct {
  uint32_t struct_size;
  uint32_t version;
  const char* anchor_name;
  const char* tag_filter; /* "" = no filter */
  double anchor_x;
  double anchor_y;
  double anchor_z;
  double anchor_yaw;
  double time_seconds; /* source phase time of the evaluation */
  int64_t binding_index;
  /* Scope root used: explicit root (none since ABI 4.2: always the actor's nearest
   * containment ancestor. */
  int64_t scope_root;
  /* Element passed to the kernel's scope policy (explicit root, or the
   * actor's own element). */
  int64_t kernel_scope_origin;
  int64_t deepest_containing;
  /* Element whose navigable geometry normalized R; 0 when none. */
  int64_t navigable_element;
  int64_t candidate_count;
  int64_t included_count;
  sb_kernel_type_t kernel_type;
  int32_t valid;          /* 0 until the binding was evaluated once */
  int32_t anchor_active;  /* 0: anchor inactive, no candidates evaluated */
  int32_t explicit_scope;
  int32_t scope_live;
  int32_t reserved;
} sb_scope_record_t;

typedef struct {
  uint32_t struct_size;
  uint32_t version;
  /* R of the evaluated relation; NaN unless it was evaluated: INCLUDED, or
   * evaluated then withheld (OUTSIDE_NEAR_FIELD, GATED_INACTIVE,
   * OBSERVED_POINT_EXPIRED). */
  double proximity;
  int64_t element_id;
  int64_t shared_id; /* equals element_id when the element is not shared */
  sb_scope_reason_t reason;
  int32_t reserved;
} sb_scope_candidate_t;

/**
 * One record per binding of `actor`, in binding order. A referent the scope
 * policy never produced does not appear among the candidates at all.
 * Candidates are in scope-policy order (an explicit leaf scope's
 * self-candidate last).
 * @return total binding count, -1 on bad struct_size / null out with cap > 0.
 */
SB_OBS_EXPORT int32_t sb_get_scope_records(sb_space_t space, sb_actor_t actor,
                                           sb_scope_record_t* out,
                                           int32_t cap, uint32_t struct_size);
/**
 * Candidates of binding `binding_index`, in scope-policy order.
 * @return total candidate count, -1 on bad arguments / unknown binding.
 */
SB_OBS_EXPORT int32_t sb_get_scope_candidates(sb_space_t space,
                                              sb_actor_t actor,
                                              int32_t binding_index,
                                              sb_scope_candidate_t* out,
                                              int32_t cap,
                                              uint32_t struct_size);

/* ==========================================================================
 * Declared policy parameters.
 * ========================================================================== */

typedef struct {
  uint32_t struct_size;
  uint32_t version;
  /* Receiver temperature schedule (SpatialBlend Eq. A6):
   *   eta = clip(eta_0 * (T_min + epsilon_t)^gamma, eta_min, eta_max). */
  double eta_0;
  double gamma;     /* production default 2 (paper: 1) -- recorded deviation */
  double epsilon_t;
  double eta_min;
  double eta_max;
  /* Source imminence constants (T_p = clip((1-R_src)/max(Rdot+, epsilon_v),
   * 0, t_max), R_src = clip(1 - (1-R)/kappa, 0, 1), Rdot EMA with tau). */
  double progress_tau_s;
  double epsilon_v;
  double t_max;
  double kappa;
  sb_realization_policy_t policy; /* ignored by the setter */
  int32_t reserved;
} sb_realization_policy_params_t;

/** Production defaults (what a fresh space uses). 1 ok, 0 bad size/null. */
SB_OBS_EXPORT int32_t sb_get_default_realization_policy_params(
    sb_realization_policy_params_t* out, uint32_t struct_size);
/** Effective parameters of `space` (policy = the resolved family). */
SB_OBS_EXPORT int32_t sb_get_realization_policy_params(
    sb_space_t space, sb_realization_policy_params_t* out,
    uint32_t struct_size);
/**
 * Set every numeric parameter at once (the family stays with
 * sb_set_realization_policy). Rejects (returns 0, changes nothing) any
 * non-finite value, eta_0 <= 0, gamma < 0, epsilon_t < 0, eta_min <= 0,
 * eta_max < eta_min, progress_tau_s <= 0, epsilon_v <= 0, t_max <= 0, or
 * kappa outside (0, 1]. Re-emits ev=policy on the next traced update.
 */
SB_OBS_EXPORT int32_t sb_set_realization_policy_params(
    sb_space_t space, const sb_realization_policy_params_t* params,
    uint32_t struct_size);

/* ==========================================================================
 * Receiver weights and the driving relation per channel.
 * A channel is one (kernel family, subject anchor) group; weights compete
 * only inside a channel.
 * ========================================================================== */

typedef struct {
  uint32_t struct_size;
  uint32_t version;
  const char* anchor_name; /* raw subject anchor; "" means root */
  const char* sub_region_name;
  double proximity;      /* R carried by the payload */
  double estimated_time; /* T_p (s) carried by the payload */
  double weight;         /* policy output in [0, 1] */
  double time_seconds;   /* receiver clock of the evaluation */
  int64_t referent_shared_id;
  /* Local element the referent resolved to (0 when resolved by shared id
   * without a linker mapping). */
  int64_t resolved_element_id;
  int64_t payload_index; /* index in the receiver's policy input */
  int64_t channel_index; /* index into sb_get_realization_channels(); -1 if
                            the payload is disabled */
  sb_kernel_type_t kernel_type;
  int32_t enabled;
  int32_t is_driving;          /* its channel's policy argmax */
  int32_t is_realized_driving; /* drove its channel's realized target */
} sb_realization_weight_t;

/* Two driving relations per channel:
 *  - driving_*: the POLICY argmax -- the max-weight payload, before the
 *    solver and the anchor gate run;
 *  - realized_*: the payload that actually drove the channel's realized
 *    target -- the max-weight goal that entered the root blend (root
 *    channels) or the payload of the target's referent when an anchor
 *    target survived the anchor gate (anchor channels). has_realized == 0 when
 *    the channel produced no realized target (gated out, unresolved). */
typedef struct {
  uint32_t struct_size;
  uint32_t version;
  const char* anchor_name;
  double time_seconds;
  double t_min;            /* NaN unless progressive with a valid T_p */
  double eta;              /* NaN unless progressive with >= 2 members */
  double driving_weight;   /* 0 when has_driving == 0 */
  double driving_proximity;
  double driving_estimated_time;
  double realized_weight;  /* 0 when has_realized == 0 */
  int64_t driving_referent_shared_id;
  int64_t driving_payload_index; /* -1 when has_driving == 0 */
  int64_t realized_referent_shared_id;
  int64_t realized_payload_index; /* -1 when has_realized == 0 */
  int64_t candidate_count;
  sb_kernel_type_t kernel_type;
  sb_realization_policy_t policy;
  int32_t has_driving;
  /* Another member carries the same maximal weight; the driving row is then
   * the lowest payload index. */
  int32_t driving_tied;
  int32_t has_realized;
  int32_t realized_is_policy_argmax;
} sb_realization_channel_t;

/** Latest receiver evaluation of `actor`. Returns the total count, 0 if the
 * actor never received a relation update, -1 on bad struct_size. */
SB_OBS_EXPORT int32_t sb_get_realization_weights(sb_space_t space,
                                                 sb_actor_t actor,
                                                 sb_realization_weight_t* out,
                                                 int32_t cap,
                                                 uint32_t struct_size);
SB_OBS_EXPORT int32_t sb_get_realization_channels(
    sb_space_t space, sb_actor_t actor, sb_realization_channel_t* out,
    int32_t cap, uint32_t struct_size);

/* ==========================================================================
 * Root solve -- witness points before blending, feasibility snap.
 * ========================================================================== */

typedef struct {
  uint32_t struct_size;
  uint32_t version;
  double time_seconds;
  /* Weighted blend of the position witnesses before the feasibility clamp
   * (valid when has_unconstrained). */
  double unconstrained_x;
  double unconstrained_y;
  double solved_x;
  double solved_y;
  double solved_yaw;
  /* |solved - unconstrained| (m). */
  double feasibility_snap_m;
  int64_t goal_count;
  int32_t has_solve;
  int32_t has_unconstrained;
  int32_t snapped; /* the clamp moved the candidate at least once */
  int32_t status;  /* 0 converged, 1 fallback, 2 failed */
} sb_solver_diagnostics_t;

typedef struct {
  uint32_t struct_size;
  uint32_t version;
  const char* anchor_name;
  /* The relation's decoded witness point (the point that alone preserves
   * its R) before blending. */
  double witness_x;
  double witness_y;
  double weight;
  int64_t referent_shared_id;
  /* Index into sb_get_realization_weights() rows. */
  int64_t payload_index;
  sb_kernel_type_t kernel_type;
  int32_t position_goal;
  int32_t yaw_goal;
  int32_t reserved;
} sb_solver_goal_t;

/** 1 when the actor has a root solve on record, 0 otherwise / bad input. */
SB_OBS_EXPORT int32_t sb_get_solver_diagnostics(sb_space_t space,
                                                sb_actor_t actor,
                                                sb_solver_diagnostics_t* out,
                                                uint32_t struct_size);
/** Goals that entered the root blend (non-zero weight). Returns the total
 * count, -1 on bad struct_size. */
SB_OBS_EXPORT int32_t sb_get_solver_goals(sb_space_t space, sb_actor_t actor,
                                          sb_solver_goal_t* out, int32_t cap,
                                          uint32_t struct_size);

/* ==========================================================================
 * Realized targets with the reach diagnostic.
 *
 * Reach never moves a target any more (a moved contact target is not the
 * contact). An engaged body-anchor target farther than the anchor's reach
 * (sb_set_anchor_reach) from the solved root is FLAGGED: beyond_reach = 1
 * and reach_excess_m = horizontal distance beyond the reach. requested_x/y
 * equal position_x/y (kept for layout compatibility).
 * ========================================================================== */

typedef struct {
  uint32_t struct_size;
  uint32_t version;
  const char* anchor_name;
  double arrival_time;
  double position_x;
  double position_y;
  double position_z;
  double yaw;
  double weight;
  /* == position_x/y (no reach clamp exists any more). */
  double requested_x;
  double requested_y;
  /* Horizontal distance of an engaged target beyond the anchor's reach from
   * the solved root; 0 when within reach. */
  double reach_excess_m;
  int64_t referent_id;
  sb_kernel_type_t kernel_type;
  int32_t has_z;
  /* 1 when the engaged target lies beyond the anchor's reach (not moved). */
  int32_t beyond_reach;
  int32_t reserved;
} sb_realization_target_v2_t;

/** Current realization result of `actor` (root first). Total count, or -1. */
SB_OBS_EXPORT int32_t sb_get_realization_targets_v2(
    sb_space_t space, sb_actor_t actor, sb_realization_target_v2_t* out,
    int32_t cap, uint32_t struct_size);

/* ==========================================================================
 * Injection.
 * ========================================================================== */

/* One relation as a receiver sees it (RelationPayload). */
typedef struct {
  uint32_t struct_size;
  uint32_t version;
  const char* anchor_name;     /* NULL or "" = root */
  const char* sub_region_name; /* NULL or "" = whole boundary */
  double proximity;
  double estimated_time; /* the injected scheduling count, in updates (read as the payload's update count) */
  double support_distance_m;
  double source_scope_diameter_m;
  double contact_point_local_x;
  double contact_point_local_y;
  double contact_point_world_x;
  double contact_point_world_y;
  double contact_point_world_z;
  double normalized_angle;
  double normalized_radial_ratio;
  double normalized_height_ratio;
  double interior_u;
  double interior_v;
  double interior_w;
  double encoded_state[SB_MAX_ENCODED_STATE];
  /* Receiver-local referent by shared id (resolved like any peer payload). */
  int64_t referent_shared_id;
  sb_kernel_type_t kernel_type;
  sb_normalized_strategy_t normalized_strategy;
  /* 0..SB_MAX_ENCODED_STATE. 0 is a valid, explicit choice, not an error: the
   * payload carries no source state, `proximity` alone is not trusted as a real
   * R^src measurement (it is indistinguishable from a payload that never went
   * through Extract_p), so the payload is left out of the solver's exactness
   * report (J over the position terms) and DecodeTargetPoint treats an empty
   * encoded_state as its own case (the bare support point). Encode the real
   * source state, as Extract_p produces it, to take part in it. */
  int32_t encoded_state_size; /* 0..SB_MAX_ENCODED_STATE */
  int32_t has_contact;
  int32_t has_interior;
  int32_t enabled; /* 0 = disabled contributor (no policy mass) */
  /* Radial surface-chart version of the normalized contact: 0 = a
   * pre-versioning sender, 1 = the current chart (both pure radial; 2, the
   * deleted regularized chart, is rejected).
   * A caller may pass struct_size = offsetof(sb_relation_payload_t,
   * chart_version) (the pre-chart_version layout); chart_version is then 0. */
  uint32_t chart_version;
  int32_t reserved;
  /* Identities, as a peer payload carries them, so a receiver
   * resolves the referent and the source's normalizing scope exactly as the
   * realizer does. Each group is used only when its has_* flag is set:
   *  - referent_source_*: the source-side referent handle (ElementLinker);
   *  - scope_shared_id: the shared id of the source's normalizing scope;
   *  - scope_source_*: that scope's source-side handle (ElementLinker).
   * Accepted struct sizes: sizeof(sb_relation_payload_t); the size before the binding fields,
   * offsetof(sb_relation_payload_t, source_binding_ordinal) (no binding
   * revision); and the pre-chart_version size offsetof(sb_relation_payload_t,
   * chart_version) (no chart version, no identities). */
  int64_t referent_source_space_id;
  int64_t referent_source_element_id;
  int64_t scope_shared_id;
  int64_t scope_source_space_id;
  int64_t scope_source_element_id;
  int32_t has_referent_source;
  int32_t has_scope_shared_id;
  int32_t has_scope_source;
  int32_t reserved2;
  /* The SOURCE binding this relation was extracted under -- its
   * ordinal on the source actor and its revision. When
   * has_binding_revision is set, the rich contributors of this relation
   * report binding_revision_known with this value; otherwise unknown. */
  uint64_t source_binding_ordinal;
  uint64_t binding_revision;
  int32_t has_binding_revision;
  /* The R formula `proximity` and the ap_* fields below were
   * encoded under (formerly the reserved3 slot; same size and offset, so a
   * zeroed or pre-pose-fit caller reads 0). 0 = proximity only (a pre-pose-fit sender:
   * ap_* are ignored whatever they hold). 1 = AffordancePose pose fit: the
   * ap_* fields are valid. Set it to 1 together with the ap_* fields; a
   * struct that does not set it is scored with `proximity` on both sides,
   * never a pose-fit R_tgt against a proximity R_src. */
  int32_t r_formula_version;
  /* AffordancePose only -- the anchor's pose relative to the
   * support, beyond normalized_angle/normalized_radial_ratio above: its
   * in-plane offset from the support point (support-local u/v) and its
   * height above the support plane (also support-local). Read only when
   * r_formula_version == 1; NAN = unknown. A caller may pass
   * struct_size = offsetof(sb_relation_payload_t, ap_offset_u_m) (the
   * pre-ap_offset layout, which still carries source_binding_ordinal and
   * binding_revision); these three then read NAN and the payload is a
   * formula-version-0 sender: Eval_p scores BOTH sides with `proximity`. */
  double ap_offset_u_m;
  double ap_offset_v_m;
  double ap_height_m;
} sb_relation_payload_t;

/**
 * Drive `actor`'s receiver with one fixed relation snapshot at
 * `time_seconds`, exactly as if it had arrived from a peer (policy, solver,
 * targets, trace, phase events all run), without any source extractor.
 * count == 0 is an authoritative empty snapshot (releases targets).
 * @return 1 on success, 0 on bad arguments (unknown actor, bad struct_size,
 *         non-finite time, encoded_state_size out of range, bad kernel).
 */
SB_OBS_EXPORT int32_t sb_inject_relation_payloads(
    sb_space_t space, sb_actor_t actor, const sb_relation_payload_t* payloads,
    int32_t count, uint32_t struct_size, double time_seconds);

typedef struct {
  uint32_t struct_size;
  uint32_t version;
  double anchor_x;
  double anchor_y;
  double anchor_z;
  double anchor_yaw;
  int64_t referent;
  /* 0: no navigable scope (Euclidean, unnormalized). Otherwise the deepest
   * containment element under scope_root containing the anchor is found and
   * the nearest Navigable ancestor of it normalizes R, exactly as the
   * binding pipeline does:
   *  - explicit_scope = 1 (retired in ABI 4.2: never set): scope_root is the bound root and
   *    the navigable search stops at it;
   *  - explicit_scope = 0 (owner-derived sb_bind/sb_bind_affordance): scope_root
   *    must be the actor element's nearest containment ancestor, and the
   *    search may continue above it to the tree root. */
  int64_t scope_root;
  sb_kernel_type_t kernel_type;
  int32_t position_dimension; /* 0 or 2: planar anchor; 3: 3-D anchor */
  int32_t explicit_scope;
  int32_t reserved;
} sb_relation_query_t;

typedef struct {
  uint32_t struct_size;
  uint32_t version;
  const char* sub_region_name;
  double proximity;
  double support_distance_m;
  double source_scope_diameter_m;
  double contact_point_world_x;
  double contact_point_world_y;
  double contact_point_world_z;
  double normalized_angle;
  double normalized_radial_ratio;
  double normalized_height_ratio;
  double interior_u;
  double interior_v;
  double interior_w;
  double encoded_state[SB_MAX_ENCODED_STATE];
  int64_t deepest_containing;
  int64_t navigable_element;
  sb_normalized_strategy_t normalized_strategy;
  int32_t encoded_state_size;
  int32_t has_contact;
  int32_t has_interior;
  /* AffordancePose only; NAN = unknown/not this family. A caller
   * may pass out->struct_size = offsetof(sb_relation_evaluation_t,
   * ap_offset_u_m) (the pre-ap_offset layout): the three fields are then not
   * written and out->struct_size is left as passed. `proximity` is the
   * geometric component, not the pose-fit R. */
  double ap_offset_u_m;
  double ap_offset_v_m;
  double ap_height_m;
} sb_relation_evaluation_t;

/**
 * Pure Extract_p: GroundingRule::ComputeSupport + RelationKernel::Encode for
 * one (kernel, anchor pose, referent, scope) -- the same calls the binding
 * pipeline makes, with no binding, phase or propagation side effect.
 * @return 1 ok, 0 on bad arguments / unknown elements.
 */
SB_OBS_EXPORT int32_t sb_evaluate_relation(sb_space_t space,
                                           const sb_relation_query_t* query,
                                           sb_relation_evaluation_t* out);

/* ==========================================================================
 * Target-side relation evaluation Eval_p.
 * ========================================================================== */

/* Outcome of sb_evaluate_target_relation (never renumbered). */
typedef enum {
  SB_TARGET_EVAL_EVALUATED = 0,
  SB_TARGET_EVAL_UNMAPPED = 1,            /* no receiver counterpart */
  SB_TARGET_EVAL_NO_SUPPORT_GEOMETRY = 2, /* counterpart has no geometry */
  SB_TARGET_EVAL_PART_NOT_FOUND = 3,      /* the source's part is missing */
  SB_TARGET_EVAL_UNSUPPORTED_SUPPORT = 4, /* geometry cannot carry it */
  SB_TARGET_EVAL_INVALID_INPUT = 5
} sb_target_eval_disposition_t;

/* Which support sb_evaluate_target_relation scored r_tgt against. */
typedef enum {
  SB_TARGET_SUPPORT_FAMILY_GROUNDING = 0, /* the family's own grounding */
  SB_TARGET_SUPPORT_MAPPED = 1,           /* the corresponding mapped support */
  SB_TARGET_SUPPORT_CANDIDATE_PROJECTION_FALLBACK = 2 /* degraded, see reason */
} sb_target_support_basis_t;

typedef struct {
  uint32_t struct_size;
  uint32_t version;
  /* Why the disposition is not EVALUATED ("" otherwise), or -- when
   * disposition is EVALUATED but support_basis is
   * SB_TARGET_SUPPORT_CANDIDATE_PROJECTION_FALLBACK -- why the mapped
   * support was unavailable. */
  const char* reason;
  const char* sub_region_name;
  /* R^tgt_p(t; S): the relation-preservation score on the receiver support
   * at S, under the formula named by r_formula_version. */
  double r_tgt;
  /* R^src_p(t): the same formula applied to the source payload. */
  double r_src;
  /* r_tgt - r_src. */
  double error;
  double support_distance_m;
  /* Diameter of the normalizing scope used (<= 0: none). */
  double normalizing_scope_diameter_m;
  double contact_point_world_x;
  double contact_point_world_y;
  double contact_point_world_z;
  /* The source support decoded onto the receiver referent (valid when
   * has_mapped_support). */
  double mapped_support_x;
  double mapped_support_y;
  double mapped_support_z;
  double encoded_state[SB_MAX_ENCODED_STATE];
  int64_t local_referent;        /* receiver element; 0 when unmapped */
  int64_t normalizing_scope;     /* receiver element; 0 when none */
  sb_target_eval_disposition_t disposition;
  int32_t encoded_state_size;
  int32_t has_contact;
  int32_t has_mapped_support;
  /* AffordancePose only; NAN = unknown/not this family. The
   * candidate's pose relative to the support it was scored against. */
  double ap_offset_u_m;
  double ap_offset_v_m;
  double ap_height_m;
  /* Which support r_tgt was scored against (sb_target_support_basis_t).
   * SB_TARGET_SUPPORT_CANDIDATE_PROJECTION_FALLBACK is a degraded score (the
   * mapped support could not be decoded; `reason` says why) and must not be
   * read as a faithful Eval_p. */
  sb_target_support_basis_t support_basis;
  /* The R formula r_src and r_tgt were BOTH scored under: 0 =
   * proximity component (every family but AffordancePose, and a pre-pose-fit
   * AffordancePose sender whose ap_* offsets are NAN), 1 = AffordancePose
   * pose fit: 0.5 position + 0.3 height + 0.2 yaw, yaw relative to the
   * source's relative yaw (dropped from both when unknown). */
  int32_t r_formula_version;
} sb_target_relation_evaluation_t;
/* A caller may pass out->struct_size = offsetof(sb_target_relation_evaluation_t,
 * ap_offset_u_m) (the pre-ap_offset layout) or offsetof(..., support_basis) (the
 * ap_offset layout without the basis/formula fields): fields at or beyond
 * struct_size are not written and out->struct_size is left as passed. */

/**
 * Eval_p (SpatialBlend eq. eval; relation_evaluator.h): ground `payload` on
 * this space's corresponding referent (resolved like any peer payload) and
 * evaluate it with the SOURCE family's kernel and grounding rule at the
 * candidate anchor pose given by `candidate` (anchor_x/y/z, anchor_yaw;
 * candidate->kernel_type and ->referent are ignored -- the payload's are
 * used). The normalizing context follows the one rule shared with the
 * realizer's decode (ResolveSpacingNormalization); candidate->scope_root,
 * when non-zero, overrides it with that navigable element.
 * payload_struct_size is sizeof(sb_relation_payload_t) or its pre-chart_version size.
 * @return 1 when `out` was written (including non-EVALUATED dispositions),
 *         0 on bad arguments (sizes, null pointers, unknown space).
 */
SB_OBS_EXPORT int32_t sb_evaluate_target_relation(
    sb_space_t space, const sb_relation_payload_t* payload,
    uint32_t payload_struct_size, const sb_relation_query_t* candidate,
    sb_target_relation_evaluation_t* out);

/* ==========================================================================
 * Element tree readback.
 * ========================================================================== */

/** World transform (row-major 4x4, the sb_set_element_transform layout):
 * the composition of the ancestors' local transforms. 1 ok, 0 unknown. */
SB_OBS_EXPORT int32_t sb_get_element_world_transform(sb_space_t space,
                                                     sb_element_t elem,
                                                     double out_row_major[16]);
/** Parent of `elem` (0 for a root). 1 ok, 0 unknown element. */
SB_OBS_EXPORT int32_t sb_get_element_parent(sb_space_t space,
                                            sb_element_t elem,
                                            sb_element_t* out_parent);

#ifdef __cplusplus
}
#endif

#endif  /* SPATIALBLEND_C_SPATIALBLEND_OBSERVABILITY_H_ */
