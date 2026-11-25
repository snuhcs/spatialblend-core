// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SPATIALBLEND_RECEIVER_APPROACH_FEASIBILITY_H_
#define SPATIALBLEND_C_SPATIALBLEND_RECEIVER_APPROACH_FEASIBILITY_H_

#include "spatialblend_rich.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SB_RECEIVER_APPROACH_FEASIBILITY_FEATURE_VERSION 1u
#define SB_RECEIVER_APPROACH_FEASIBILITY_MAX_ROUTE_POINTS 16384u

// Mirrors the receiver-local feasibility outcome carried in header.feasibility_status.
typedef enum {
  SB_C3_FEASIBLE = 0, SB_C3_NO_CONTEXT = 1, SB_C3_STALE = 2,
  SB_C3_EXPIRED = 3, SB_C3_REPLACED = 4, SB_C3_BUSY = 5,
  SB_C3_START_OUTSIDE = 6, SB_C3_GOAL_OUTSIDE = 7, SB_C3_BOUNDARY = 8,
  SB_C3_DISCONNECTED = 9, SB_C3_BLOCKED = 10, SB_C3_INSUFFICIENT_CLEARANCE = 11,
  SB_C3_INVALID_GEOMETRY = 12,
} sb_receiver_approach_c3_status_t;

typedef struct {
  uint32_t descriptor_size, feature_version, header_size, max_route_points;
  uint32_t coordinate_convention, route_point_stride, native_clock_domain;
  uint32_t required_context_version;
} sb_receiver_approach_feasibility_feature_t;

typedef struct {
  uint32_t struct_size, feature_version;
  int32_t feasibility_status;
  uint32_t coordinate_convention;
  sb_session_t session;
  sb_space_t space;
  sb_actor_t receiver_actor;
  uint64_t registration_session_id, registration_generation;
  uint64_t receiver_input_epoch, receiver_input_sequence, receiver_input_revision;
  uint64_t geometry_revision, support_revision, context_revision;
  uint64_t calculation_revision, owner_lifetime, native_clock_domain;
  int64_t issued_at_ns, valid_until_ns;
  uint32_t route_point_count, route_point_stride;
  double start_x, start_y, safe_approach_x, safe_approach_y, route_length_m;
} sb_receiver_approach_feasibility_header_t;

typedef enum {
  SB_RECEIVER_APPROACH_FEASIBILITY_OK = 1,
  SB_RECEIVER_APPROACH_FEASIBILITY_UNAVAILABLE = 0,
  SB_RECEIVER_APPROACH_FEASIBILITY_INVALID_ARGUMENT = -1,
  SB_RECEIVER_APPROACH_FEASIBILITY_UNSUPPORTED_VERSION = -2,
  SB_RECEIVER_APPROACH_FEASIBILITY_STALE = -3,
  SB_RECEIVER_APPROACH_FEASIBILITY_EXPIRED = -4,
  SB_RECEIVER_APPROACH_FEASIBILITY_REPLACED = -5,
  SB_RECEIVER_APPROACH_FEASIBILITY_BUSY = -6,
  SB_RECEIVER_APPROACH_FEASIBILITY_RETRY_CHANGED = -7,
  SB_RECEIVER_APPROACH_FEASIBILITY_BUFFER_TOO_SMALL = -8,
} sb_receiver_approach_feasibility_result_t;

SB_EXPORT uint32_t sb_get_receiver_approach_feasibility_feature(
    sb_receiver_approach_feasibility_feature_t* out, uint32_t out_size);
SB_EXPORT uint32_t sb_get_receiver_approach_feasibility_field_count(uint32_t kind);
SB_EXPORT uint32_t sb_get_receiver_approach_feasibility_field_offset(uint32_t kind,
                                                                      uint32_t index);

// expected_calculation_revision=0 computes and freezes a new snapshot.
// A nonzero value copies that exact frozen snapshot only.  Capacity=0/null is
// a probe; BUFFER_TOO_SMALL writes header/count only and never route prefix.
SB_EXPORT int32_t sb_copy_receiver_approach_feasibility(
    sb_session_t session, sb_space_t space, sb_actor_t receiver_actor,
    uint32_t feature_version, uint64_t context_revision,
    uint64_t expected_calculation_revision, double* route_xy,
    uint32_t route_point_capacity,
    sb_receiver_approach_feasibility_header_t* header, uint32_t header_size);

#ifdef __cplusplus
}
#endif
#endif
