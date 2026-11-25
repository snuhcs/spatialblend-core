// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SPATIALBLEND_PLANAR_SUPPORT_H_
#define SPATIALBLEND_C_SPATIALBLEND_PLANAR_SUPPORT_H_

#include "spatialblend_rich.h"

#ifdef __cplusplus
extern "C" {
#endif

// Additive, independently negotiated numeric read. Rich-v1 and ABI2 unchanged.
#define SB_PLANAR_SUPPORT_FEATURE_VERSION 1u
#define SB_PLANAR_SUPPORT_MAX_POINTS 65536u
// (u,v,0,1) -> Z-up world through the affine row-major matrix. This defines
// the queried geometry frame, NOT the absent rich-v1 source chart version.
#define SB_PLANAR_SUPPORT_XY_AFFINE_CONVENTION 1u

typedef struct {
  uint32_t descriptor_size;
  uint32_t feature_version;
  uint32_t header_size;
  uint32_t max_points;
  uint32_t coordinate_convention;  // 1: Z-up metres.
  uint32_t plane_convention;
} sb_planar_support_feature_t;

typedef struct {
  uint32_t struct_size;
  uint32_t feature_version;
  uint32_t definition_kind;  // 1: Surface, 2: Navigable; whole boundary only.
  uint32_t geometry_role;    // sb_geometry_role_t; not semantic attestation.
  sb_space_t space;
  sb_element_t element;
  uint64_t tree_revision;
  uint64_t support_revision;
  uint32_t point_count;
  uint32_t plane_convention;
  double plane_to_world_row_major[16];
} sb_planar_support_header_t;

typedef enum {
  SB_PLANAR_SUPPORT_OK = 1,
  SB_PLANAR_SUPPORT_UNAVAILABLE = 0,
  SB_PLANAR_SUPPORT_INVALID_ARGUMENT = -1,
  SB_PLANAR_SUPPORT_STALE = -2,
  SB_PLANAR_SUPPORT_BUFFER_TOO_SMALL = -3,
  SB_PLANAR_SUPPORT_BUSY = -4,
} sb_planar_support_result_t;

SB_EXPORT uint32_t sb_get_planar_support_feature(
    sb_planar_support_feature_t* feature, uint32_t feature_size);
// kind 1=feature, 2=header. Unknown kind/index -> 0/UINT32_MAX respectively.
SB_EXPORT uint32_t sb_get_planar_support_field_count(uint32_t kind);
SB_EXPORT uint32_t sb_get_planar_support_field_offset(uint32_t kind,
                                                    uint32_t index);

// Caller owns header and capacity*2 doubles. Snapshot/actor admission is NOT
// implied: validate rich currentness before/after this read separately. Expected
// scene revisions are mandatory and compared while locked. The session pins
// exact native lifetime. A try-lock avoids waiting under session/solve callbacks.
//
// Capacity=0/null is a size probe. BUFFER_TOO_SMALL writes header only, including
// point_count; it never partially writes coordinates. Retry with the SAME
// expected revisions; mutation yields STALE, not a mixed-size/new geometry read.
// OK gives detached values as of the query, not a lease. Borrowed/escaped scene
// inputs, unsupported geometry, missing/retiring session or object -> UNAVAILABLE.
SB_EXPORT int32_t sb_copy_planar_support(
    sb_session_t session, sb_space_t space, sb_element_t element,
    uint32_t feature_version, uint64_t expected_tree_revision,
    uint64_t expected_support_revision, double* boundary_xy,
    uint32_t point_capacity, sb_planar_support_header_t* header,
    uint32_t header_size);

#ifdef __cplusplus
}
#endif

#endif  // SPATIALBLEND_C_SPATIALBLEND_PLANAR_SUPPORT_H_
