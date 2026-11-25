// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SPATIALBLEND_RECEIVER_APPROACH_CONTEXT_H_
#define SPATIALBLEND_C_SPATIALBLEND_RECEIVER_APPROACH_CONTEXT_H_

// Additive receiver approach-context ingress.  This admits explicit receiver-local data only; it
// neither computes a route nor accepts source/Unity/controller input.
#include "spatialblend_rich.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SB_RECEIVER_APPROACH_CONTEXT_FEATURE_VERSION 1u
#define SB_RECEIVER_APPROACH_CONTEXT_MAX_POLYGONS 64u
#define SB_RECEIVER_APPROACH_CONTEXT_MAX_VERTICES_PER_POLYGON 256u
#define SB_RECEIVER_APPROACH_CONTEXT_MAX_TOTAL_VERTICES 32768u
#define SB_RECEIVER_APPROACH_CONTEXT_COORDINATE_CONVENTION 1u
#define SB_RECEIVER_APPROACH_CONTEXT_AFFINE_CONVENTION 1u

typedef struct {
  uint32_t descriptor_size, feature_version;
  uint32_t registration_size, polygon_size, destination_size;
  uint32_t max_polygons, max_vertices_per_polygon, max_total_vertices;
  uint32_t max_root_name_bytes, coordinate_convention, affine_convention;
  uint32_t required_receiver_input_version, native_clock_domain;
} sb_receiver_approach_context_feature_t;

// Local XY vertices map to the declared receiver Space Z-up metre basis by
// local_to_space_row_major (a 3x3 row-major affine matrix).  vertices_xy has
// exactly vertex_count pairs and remains caller-owned for the duration only.
typedef struct {
  uint32_t struct_size;
  uint32_t vertex_count;
  uint64_t nav_element_id;
  uint64_t region_id;
  uint32_t coordinate_convention;
  uint32_t affine_convention;
  double local_to_space_row_major[9];
  const double* vertices_xy;
} sb_receiver_approach_polygon_t;

typedef struct {
  uint32_t struct_size;
  sb_space_t space;
  uint64_t region_id;
  sb_element_t object;
  uint64_t affordance_id;
  uint64_t support_revision;
  sb_receiver_approach_polygon_t local_support;
  uint32_t coordinate_convention;
  uint32_t reserved;
  double safe_approach_pose_row_major[16];
} sb_receiver_approach_destination_t;

typedef struct {
  uint32_t struct_size;
  uint32_t feature_version;
  uint64_t expected_registration_generation;
  uint64_t expected_receiver_input_epoch;
  uint64_t expected_receiver_input_sequence;
  uint64_t geometry_revision;
  int64_t requested_valid_until_ns;
  const uint8_t* navigation_root_utf8;
  uint32_t navigation_root_bytes;
  uint32_t coordinate_convention;
  double navigation_root_pose_row_major[16];
  const sb_receiver_approach_polygon_t* navigable_regions;
  uint32_t navigable_region_count;
  const sb_receiver_approach_polygon_t* obstacle_regions;
  uint32_t obstacle_region_count;
  double footprint_radius_m;
  double required_clearance_m;
  sb_receiver_approach_destination_t destination;
} sb_receiver_approach_context_registration_t;

typedef enum {
  SB_RECEIVER_APPROACH_CONTEXT_OK = 1,
  SB_RECEIVER_APPROACH_CONTEXT_UNAVAILABLE = 0,
  SB_RECEIVER_APPROACH_CONTEXT_INVALID_ARGUMENT = -1,
  SB_RECEIVER_APPROACH_CONTEXT_UNSUPPORTED_VERSION = -2,
  SB_RECEIVER_APPROACH_CONTEXT_STALE = -3,
  SB_RECEIVER_APPROACH_CONTEXT_EXPIRED = -4,
  SB_RECEIVER_APPROACH_CONTEXT_REPLACED = -5,
  SB_RECEIVER_APPROACH_CONTEXT_BUSY = -6,
  SB_RECEIVER_APPROACH_CONTEXT_MISSING_NAVIGATION_ROOT = -7,
  SB_RECEIVER_APPROACH_CONTEXT_BODY_ONLY_INPUT = -8,
  SB_RECEIVER_APPROACH_CONTEXT_MISSING_GEOMETRY = -9,
  SB_RECEIVER_APPROACH_CONTEXT_INVALID_GEOMETRY = -10,
  SB_RECEIVER_APPROACH_CONTEXT_MISSING_POLICY = -11,
  SB_RECEIVER_APPROACH_CONTEXT_MISSING_DESTINATION = -12,
} sb_receiver_approach_context_result_t;

// Short feature probes do not write.  Kinds 1..4 are feature, registration,
// polygon and destination; fields follow declaration order.
SB_EXPORT uint32_t sb_get_receiver_approach_context_feature(
    sb_receiver_approach_context_feature_t* out, uint32_t out_size);
SB_EXPORT uint32_t sb_get_receiver_approach_context_field_count(uint32_t kind);
SB_EXPORT uint32_t sb_get_receiver_approach_context_field_offset(uint32_t kind,
                                                                  uint32_t index);

// Atomically deep-copies a complete registration through the current admitted
// receiver input.  No input pointers are retained.  The native steady clock
// domain is selected by Core; callers cannot provide source/replay time.
SB_EXPORT int32_t sb_register_receiver_approach_context(
    sb_session_t session, sb_space_t space, sb_actor_t receiver_actor,
    uint32_t feature_version,
    const sb_receiver_approach_context_registration_t* registration,
    uint32_t registration_size, uint64_t* out_context_revision);

#ifdef __cplusplus
}
#endif

#endif  // SPATIALBLEND_C_SPATIALBLEND_RECEIVER_APPROACH_CONTEXT_H_
