// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SPATIALBLEND_SOURCE_EVIDENCE_H_
#define SPATIALBLEND_C_SPATIALBLEND_SOURCE_EVIDENCE_H_

#include "spatialblend_rich.h"

#ifdef __cplusplus
extern "C" {
#endif

// Independent optional feature. Does not change ABI2, rich or actor-frame ABI.
#define SB_SOURCE_EVIDENCE_FEATURE_VERSION 1u
#define SB_SOURCE_EVIDENCE_MAX_UTF8 2097152u

typedef struct {
  uint32_t descriptor_size, feature_version;
  uint32_t header_size, anchor_size, relation_size, node_size, copy_size;
  uint32_t max_anchors, max_relations, max_path_depth, max_nodes, max_points;
  uint32_t max_name_bytes, max_utf8_bytes, coordinate_convention;
  uint64_t max_age_ns, monotonic_clock_id, monotonic_clock_epoch;
} sb_source_evidence_feature_t;

typedef struct {
  uint32_t struct_size, feature_version;
  sb_space_t space;
  sb_actor_t actor;
  sb_element_t actor_element;
  int64_t actor_shared;  // Actual shared id; invalid stays invalid.
  uint64_t epoch;
  int64_t sequence;
  double source_time_seconds;
  int64_t receipt_monotonic_ns;  // Original admission receipt, not replay time.
  uint64_t monotonic_clock_id, monotonic_clock_epoch;
  uint64_t tree_revision, support_revision;
} sb_source_evidence_header_t;

typedef struct {
  uint32_t name_offset, name_length;
  uint32_t observed, locomotion_root, position_dimension;
  double position_x, position_y, position_z,
      yaw;  // Z-up metres, Z yaw radians.
} sb_source_evidence_anchor_t;

typedef struct {
  // These are a single native selection: binding_ordinal identifies the
  // evaluated binding and anchor_ordinal selects its declared anchor row. The
  // anchor name is read only from that row; consumers must not infer it from
  // kernel, relation order, a local id, or a repeated source sequence.
  uint32_t binding_ordinal, anchor_ordinal;
  int32_t kernel;
  sb_element_t explicit_scope;
  uint32_t filter_offset, filter_length;
  double proximity, normalized_weight;
  int32_t
      phase;  // 0 Dormant, 1 Active, 2 Driving, 3 Contact; not seat semantics.
  uint32_t phase_identity_unambiguous;
  int64_t phase_referent_id;  // Generic phase key may use a local-id fallback.
  uint32_t path_offset, path_count;
  uint32_t has_projection, bounded_planar_support;
  double projected_x, projected_y, projected_z, anchor_support_distance;
  double support_to_world_row_major[16];
  uint32_t point_offset, point_count;  // Points, not doubles, in boundary_xy.
  uint32_t sub_region_offset, sub_region_length;
  double support_angle, support_radial_ratio;
} sb_source_evidence_relation_t;

typedef struct {
  sb_element_t element, parent;
  int64_t shared;
  uint32_t tag_offset, tag_length, geometry_role;
  double world_transform_row_major[16];
} sb_source_evidence_node_t;

typedef struct {
  uint32_t struct_size;
  uint32_t anchor_capacity, relation_capacity, node_capacity, point_capacity,
      utf8_capacity;
  sb_source_evidence_anchor_t* anchors;
  sb_source_evidence_relation_t* relations;
  sb_source_evidence_node_t* nodes;
  double* boundary_xy;
  uint8_t* utf8;
  sb_source_evidence_header_t header;
  uint32_t anchor_count, relation_count, node_count, point_count, utf8_bytes;
} sb_source_evidence_copy_t;

typedef enum {
  SB_SOURCE_EVIDENCE_OK = 1,
  SB_SOURCE_EVIDENCE_UNAVAILABLE = 0,
  SB_SOURCE_EVIDENCE_INVALID_ARGUMENT = -1,
  SB_SOURCE_EVIDENCE_STALE = -2,
  SB_SOURCE_EVIDENCE_BUFFER_TOO_SMALL = -3,
  SB_SOURCE_EVIDENCE_BUSY = -4,
} sb_source_evidence_result_t;

// Kinds 1..6: feature, header, anchor, relation, node, copy. Fields in
// declaration order; invalid kind/index -> 0/UINT32_MAX. Short descriptor
// probes do not write.
SB_EXPORT uint32_t sb_get_source_evidence_feature(
    sb_source_evidence_feature_t* out, uint32_t size);
SB_EXPORT uint32_t sb_get_source_evidence_field_count(uint32_t kind);
SB_EXPORT uint32_t sb_get_source_evidence_field_offset(uint32_t kind,
                                                       uint32_t index);

// Pins exact session/space ownership; both session and Space locks are
// try-locks. Requires an existing source epoch. Future frames only; no replay
// or actuation. Disable/reset/retirement revoke current copies. No old-epoch
// disable affects a replacement epoch. All functions return
// sb_source_evidence_result_t.
SB_EXPORT int32_t sb_try_enable_source_evidence(
    sb_session_t session, sb_space_t space, sb_actor_t actor, uint32_t version,
    uint64_t epoch, uint32_t enabled);

// Capacity=0/null is a probe. BUFFER_TOO_SMALL writes only header/count
// metadata, never any caller array prefix. All other failures clear
// header/counts when copy_size is sufficient; pointers/capacities and array
// contents are unchanged. Caller owns all declared extents and must not overlap
// buffers with each other or the copy struct. Strings are bounded valid UTF-8
// slices without NUL. Retry with the SAME epoch/sequence/max_age. The read
// copies a frozen admitted value, never later geometry. Old/expired/mutated
// context is unavailable/stale. OK is a detached as-of-query value, not a lease
// or a physical contact claim.
SB_EXPORT int32_t sb_copy_source_evidence(
    sb_session_t session, sb_space_t space, sb_actor_t actor, uint32_t version,
    uint64_t epoch, int64_t sequence, int64_t maximum_age_ns,
    sb_source_evidence_copy_t* copy, uint32_t copy_size);

#ifdef __cplusplus
}
#endif
#endif  // SPATIALBLEND_C_SPATIALBLEND_SOURCE_EVIDENCE_H_
