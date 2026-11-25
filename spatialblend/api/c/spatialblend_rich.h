// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SPATIALBLEND_RICH_H_
#define SPATIALBLEND_C_SPATIALBLEND_RICH_H_

// Additive rich realization boundary.  This header intentionally does not
// alter the base ABI's sb_realization_target_t, sb_on_targets_t, or descriptor.

#include <stdint.h>

#include "spatialblend_type.h"

#ifndef SB_EXPORT
#if defined(_WIN32)
#if defined(SPATIALBLEND_BUILD_DLL)
#define SB_EXPORT __declspec(dllexport)
#else
#define SB_EXPORT __declspec(dllimport)
#endif
#else
#define SB_EXPORT __attribute__((visibility("default")))
#endif
#endif  // SB_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

#define SB_RICH_REALIZATION_FEATURE_VERSION 1u
#define SB_RICH_REALIZATION_VALIDITY_FEATURE_VERSION 1u
#define SB_RICH_REALIZATION_CLOCK_ID 1u
#define SB_RICH_REALIZATION_UNKNOWN_POLICY_REVISION (1u << 0)
#define SB_RICH_REALIZATION_UNKNOWN_TREE_REVISION (1u << 1)
#define SB_RICH_REALIZATION_UNKNOWN_COUNTERPART_REVISION (1u << 2)
#define SB_RICH_REALIZATION_UNKNOWN_SUPPORT_REVISION (1u << 3)
#define SB_RICH_FAILURE_NONE 0u
#define SB_RICH_FAILURE_LIMIT_EXCEEDED 1u

// Requested-domain bits for the separately negotiated validity query.  The
// first four match the historical header masks. Receiver input is deliberately
// validity-only: Stage A reports it unchecked rather than silently claiming
// that policy/link-only validity covers current actor pose/anchor state.
#define SB_RICH_VALIDITY_DOMAIN_POLICY (1u << 0)
#define SB_RICH_VALIDITY_DOMAIN_TREE (1u << 1)
#define SB_RICH_VALIDITY_DOMAIN_COUNTERPART (1u << 2)
#define SB_RICH_VALIDITY_DOMAIN_SUPPORT (1u << 3)
#define SB_RICH_VALIDITY_DOMAIN_RECEIVER_INPUT (1u << 4)
#define SB_RICH_VALIDITY_DOMAIN_ALL \
  (SB_RICH_VALIDITY_DOMAIN_POLICY | SB_RICH_VALIDITY_DOMAIN_TREE | \
   SB_RICH_VALIDITY_DOMAIN_COUNTERPART | SB_RICH_VALIDITY_DOMAIN_SUPPORT | \
   SB_RICH_VALIDITY_DOMAIN_RECEIVER_INPUT)

typedef uint64_t sb_rich_realization_token_t;

typedef enum {
  SB_RICH_RESULT_OK = 1,
  SB_RICH_RESULT_NO_NEW_SNAPSHOT = 0,
  SB_RICH_RESULT_INVALID_ARGUMENT = -1,
  SB_RICH_RESULT_UNKNOWN_TOKEN = -2,
  SB_RICH_RESULT_BUFFER_TOO_SMALL = -3,
  SB_RICH_RESULT_UNSUPPORTED_VERSION = -4,
  SB_RICH_RESULT_UNAVAILABLE = -5,
  SB_RICH_RESULT_STALE_SNAPSHOT = -6,
} sb_rich_realization_result_t;

typedef enum {
  SB_RICH_SNAPSHOT_COMPLETE = 1,
  SB_RICH_SNAPSHOT_EMPTY = 2,
  SB_RICH_SNAPSHOT_FAILED = 3,
} sb_rich_snapshot_status_t;
typedef enum {
  SB_RICH_SOURCE_LEGACY_UNKNOWN = 0,
  SB_RICH_SOURCE_TIMESTAMPED_OBSERVATION = 1,
  SB_RICH_SOURCE_OBSERVATION_RESET = 2,
} sb_rich_source_timing_mode_t;
typedef enum {
  SB_RICH_FAMILY_FACING = 0,
  SB_RICH_FAMILY_SPACING = 1,
  SB_RICH_FAMILY_AFFORDANCE_POSE = 2
} sb_rich_family_t;
typedef enum {
  SB_RICH_SOLVER_CONVERGED = 0,
  /* Also carries a solve that ran out of its evaluation budget (SolverStatus::kBudgetHit): its best feasible incumbent is published,
   * and it is NOT converged. The ABI has no separate value for it yet; the C observation does not expose
   * budget_hit yet, and the proposal's status for it is INEXACT, as for any non-exact answer. A solve that hit the iteration safety cap
   * (SolverStatus::kNotConverged) reports FAILED. */
  SB_RICH_SOLVER_FALLBACK = 1,
  SB_RICH_SOLVER_FAILED = 2
} sb_rich_solver_status_t;
typedef enum {
  SB_RICH_GEOMETRY_NOT_APPLICABLE = 0,
  SB_RICH_GEOMETRY_EXACT_SUPPORT = 1,
  SB_RICH_GEOMETRY_BLENDED_SUPPORT = 2,
  /* The channel's EXACT support lies beyond the anchor's reach
   * from the solved root. The target is not moved (reach no longer projects
   * a contact); reach_projected = 1, reach_displacement_m = the excess. */
  SB_RICH_GEOMETRY_REACH_PROJECTED = 3,
  SB_RICH_GEOMETRY_LEGACY_CENTER_NON_CONTACT = 4,
  SB_RICH_GEOMETRY_UNSUPPORTED = 5
} sb_rich_geometry_t;
typedef enum {
  SB_RICH_GOAL_NONE = 0,
  SB_RICH_GOAL_ROOT_POSITION = 1,
  SB_RICH_GOAL_FACING_DIRECTION = 2,
  SB_RICH_GOAL_SUPPORT_POSE = 3,
  SB_RICH_GOAL_SOFT_GEOMETRIC_POSE = 4
} sb_rich_goal_kind_t;
typedef enum {
  SB_RICH_DISPOSITION_INCLUDED = 0,
  SB_RICH_DISPOSITION_DISABLED = 1,
  SB_RICH_DISPOSITION_UNMAPPED = 2,
  SB_RICH_DISPOSITION_GROUNDING_FAILED = 3,
  SB_RICH_DISPOSITION_UNSUPPORTED = 4,
  /* Admitted, but the source anchor is not engaged on its support;
   * the anchor follows the body. A normal state, not a failure. */
  SB_RICH_DISPOSITION_AWAITING_ENGAGEMENT = 5
} sb_rich_disposition_t;
typedef enum {
  SB_RICH_GROUNDING_EXPLICIT_SUB_REGION = 0,
  SB_RICH_GROUNDING_WHOLE_PLANAR_BOUNDARY = 1,
  SB_RICH_GROUNDING_LEGACY_CENTER_FALLBACK = 2
} sb_rich_grounding_mode_t;

// Negotiated separately from ABI2.  `descriptor_size` and every offset/size
// are supplied so managed bindings can reject an incompatible native build
// before dereferencing flat arrays.
typedef struct {
  uint32_t descriptor_size;
  uint32_t feature_version;
  uint32_t snapshot_size;
  uint32_t channel_size;
  uint32_t contributor_size;
  uint32_t copy_size;
  uint32_t snapshot_channels_offset;
  uint32_t snapshot_contributors_offset;
  uint32_t snapshot_utf8_offset;
  uint32_t snapshot_channel_count_offset;
  uint32_t snapshot_contributor_count_offset;
  uint32_t snapshot_utf8_bytes_offset;
  uint32_t copy_channels_offset;
  uint32_t copy_contributors_offset;
  uint32_t copy_utf8_offset;
  uint32_t copy_header_offset;
  uint32_t channel_anchor_offset_offset;
  uint32_t channel_contributor_offset_offset;
  uint32_t channel_has_z_offset;
  uint32_t channel_requested_position_offset;
  uint32_t contributor_reason_offset_offset;
  uint32_t contributor_support_identifier_offset;
  uint32_t contributor_support_plane_to_world_offset;
  uint32_t max_channels;
  uint32_t max_contributors;
  uint32_t max_utf8_bytes;
  uint32_t coordinate_convention;  // 1 = Z-up, matrices are row-major.
  uint64_t monotonic_clock_id;
  uint64_t monotonic_clock_epoch;
} sb_rich_realization_feature_t;

// Fixed-width header for one frozen native transaction.  `receiver_*` are C
// handles for the activated receiver. `source_time_seconds` remains in the
// declared source clock; receiver ingress/solve are raw nanoseconds in the
// identified native monotonic clock/epoch and are never converted on poll.
typedef struct {
  uint64_t producer_lifetime;
  uint64_t revision;
  uint64_t monotonic_clock_id;
  uint64_t monotonic_clock_epoch;
  uint64_t receiver_ingress_monotonic_ns;
  uint64_t receiver_solve_monotonic_ns;
  uint64_t source_epoch;
  // Relation-delivery sequence; resets consume this sequence too. It is not
  // the source caller's observation-frame sequence.
  int64_t source_sequence;
  // The original timestamped observation-frame sequence. Known only for an
  // actual source sample, never inferred for legacy/reset/control updates.
  int64_t source_observation_sequence;
  int64_t source_actor_space;
  int64_t source_actor_element;
  int64_t relay_source_space;
  int64_t receiver_space;
  int64_t receiver_actor;
  uint64_t policy_revision;
  uint64_t tree_revision;
  uint64_t counterpart_revision;
  uint64_t support_revision;
  double source_time_seconds;
  double source_clock_seconds;
  uint32_t status;
  uint32_t source_timing_mode;
  uint32_t source_time_known;
  uint32_t source_clock_known;
  uint32_t source_observation_sequence_known;
  uint32_t known_revision_mask;
  uint32_t unknown_revision_mask;
  uint32_t failure_code;
} sb_rich_realization_header_t;

// Flat channel row.  anchor bytes are `utf8[anchor_offset:anchor_offset+
// anchor_length]`; there is no NUL or implicit encoding conversion.
typedef struct {
  uint32_t anchor_offset;
  uint32_t anchor_length;
  uint32_t contributor_offset;
  uint32_t contributor_count;
  int32_t family;
  int32_t solver_status;
  int32_t geometry;
  int32_t goal_kind;
  uint32_t has_goal;
  uint32_t has_z;
  uint32_t has_requested_position;
  uint32_t reach_projected;  // exact support beyond reach (not moved)
  int64_t referent_id;
  double arrival_seconds;
  double position_x;
  double position_y;
  double position_z;
  double yaw;
  double weight;
  double reach_displacement_m;
  double requested_position_x;
  double requested_position_y;
} sb_rich_realization_channel_t;

typedef struct {
  int64_t referent_id;
  int64_t source_space;
  int64_t source_element;
  int64_t resolved_local_element;
  uint64_t binding_revision;
  uint32_t snapshot_ordinal;
  int32_t disposition;
  int32_t grounding_mode;
  uint32_t has_mapped_support;
  uint32_t chart_version_known;
  uint32_t binding_revision_known;
  uint32_t reason_offset;
  uint32_t reason_length;
  uint32_t support_identifier_offset;
  uint32_t support_identifier_length;
  double applied_weight;
  double mapped_support_x;
  double mapped_support_y;
  double mapped_support_z;
  double support_plane_to_world_row_major[16];
} sb_rich_realization_contributor_t;

// Callback view is borrowed only for the duration of sb_on_rich_realization_t.
// Array/string pointers point into one immutable, C-owned frozen copy.
typedef struct {
  sb_rich_realization_header_t header;
  const sb_rich_realization_channel_t* channels;
  const sb_rich_realization_contributor_t* contributors;
  const uint8_t* utf8;
  uint32_t channel_count;
  uint32_t contributor_count;
  uint32_t utf8_bytes;
  uint32_t reserved;
} sb_rich_realization_snapshot_t;

// Poll copy contract.  Caller initializes struct_size, array pointers and
// capacities.  On BUFFER_TOO_SMALL the required counts/bytes are reported but
// no snapshot arrays are written and the poll cursor is not advanced.
typedef struct {
  uint32_t struct_size;
  uint32_t channel_capacity;
  uint32_t contributor_capacity;
  uint32_t utf8_capacity;
  sb_rich_realization_channel_t* channels;
  sb_rich_realization_contributor_t* contributors;
  uint8_t* utf8;
  sb_rich_realization_header_t header;
  uint32_t required_channels;
  uint32_t required_contributors;
  uint32_t required_utf8_bytes;
  uint32_t reserved;
} sb_rich_realization_copy_t;

typedef void (*sb_on_rich_realization_t)(
    const sb_rich_realization_snapshot_t* snapshot, void* user_data);

// Independently negotiated from rich feature-v1 so v1 snapshot layouts remain
// frozen. All offsets below are independently queryable before use.
typedef struct {
  uint32_t descriptor_size;
  uint32_t feature_version;
  uint32_t validity_size;
  uint32_t supported_domain_mask;
  uint32_t validity_producer_lifetime_offset;
  uint32_t validity_revision_offset;
  uint32_t validity_captured_known_mask_offset;
  uint32_t validity_current_known_mask_offset;
  uint32_t validity_changed_mask_offset;
  uint32_t validity_unchecked_mask_offset;
  uint32_t validity_valid_offset;
  uint32_t reserved;
} sb_rich_realization_validity_feature_t;

// A comparison of one frozen rich revision with the receiver's current
// state. `captured_*` are copied from the historical solve; current fields are
// zero when their current_known bit is clear. `valid` is one only when every
// requested domain is captured-known, current-known, checked, and unchanged.
typedef struct {
  uint64_t producer_lifetime;
  uint64_t revision;
  uint64_t captured_policy_revision;
  uint64_t captured_tree_revision;
  uint64_t captured_counterpart_revision;
  uint64_t captured_support_revision;
  uint64_t current_policy_revision;
  uint64_t current_tree_revision;
  uint64_t current_counterpart_revision;
  uint64_t current_support_revision;
  uint32_t requested_domain_mask;
  uint32_t captured_known_mask;
  uint32_t current_known_mask;
  uint32_t changed_mask;
  uint32_t unchecked_mask;
  uint32_t valid;
} sb_rich_realization_validity_t;

// Field-query ABI gate: indices are declaration order (zero-based) within
// each struct, including pointer and array members as one field.  Consumers
// compare every field offset and field count to their local mirror; UINT32_MAX
// means an invalid struct/index.  This avoids a partial descriptor silently
// blessing a managed layout with a later-field mismatch.
typedef enum {
  SB_RICH_LAYOUT_FEATURE = 1,
  SB_RICH_LAYOUT_HEADER = 2,
  SB_RICH_LAYOUT_CHANNEL = 3,
  SB_RICH_LAYOUT_CONTRIBUTOR = 4,
  SB_RICH_LAYOUT_SNAPSHOT = 5,
  SB_RICH_LAYOUT_COPY = 6
} sb_rich_layout_struct_t;

typedef enum {
  SB_RICH_VALIDITY_LAYOUT_FEATURE = 1,
  SB_RICH_VALIDITY_LAYOUT_RESULT = 2
} sb_rich_validity_layout_struct_t;

SB_EXPORT uint32_t sb_get_rich_realization_feature(
    sb_rich_realization_feature_t* feature, uint32_t feature_size);
SB_EXPORT uint32_t sb_get_rich_realization_field_count(uint32_t struct_kind);
SB_EXPORT uint32_t sb_get_rich_realization_field_offset(uint32_t struct_kind,
                                                        uint32_t field_index);
SB_EXPORT uint32_t sb_get_rich_realization_validity_feature(
    sb_rich_realization_validity_feature_t* feature, uint32_t feature_size);
SB_EXPORT uint32_t sb_get_rich_realization_validity_field_count(
    uint32_t struct_kind);
SB_EXPORT uint32_t sb_get_rich_realization_validity_field_offset(
    uint32_t struct_kind, uint32_t field_index);
SB_EXPORT sb_rich_realization_token_t sb_activate_rich_realization(
    sb_space_t space, sb_actor_t actor, uint32_t requested_feature_version);
SB_EXPORT void sb_retire_rich_realization(sb_rich_realization_token_t token);
// Releases a retired token and its frozen terminal snapshot. Active tokens
// reject release; callers must retire first.  This bounds registry retention.
SB_EXPORT int32_t
sb_release_rich_realization(sb_rich_realization_token_t token);
SB_EXPORT void sb_register_on_rich_realization(
    sb_space_t space, sb_on_rich_realization_t callback, void* user_data);
SB_EXPORT int32_t sb_poll_rich_realization(sb_rich_realization_token_t token,
                                           sb_rich_realization_copy_t* out);
// Does not poll, refresh, or advance a token cursor. `revision` must identify
// the token's currently frozen snapshot, otherwise STALE_SNAPSHOT is returned.
// UNKNOWN/UNCHECKED requested domains return UNAVAILABLE with diagnostic masks.
SB_EXPORT int32_t sb_get_rich_realization_validity(
    sb_rich_realization_token_t token, uint64_t revision,
    uint32_t requested_domain_mask, sb_rich_realization_validity_t* out,
    uint32_t out_size);
SB_EXPORT uint64_t sb_read_rich_monotonic_ns(void);
SB_EXPORT double sb_read_rich_monotonic_seconds(void);

#ifdef __cplusplus
}
#endif

#endif  // SPATIALBLEND_C_SPATIALBLEND_RICH_H_
