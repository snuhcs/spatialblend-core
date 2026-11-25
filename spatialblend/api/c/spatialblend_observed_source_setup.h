// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SPATIALBLEND_OBSERVED_SOURCE_SETUP_H_
#define SPATIALBLEND_C_SPATIALBLEND_OBSERVED_SOURCE_SETUP_H_

// Additive observed-source setup boundary.  This is deliberately limited to opening fresh
// source observation epochs and source-anchor point leases.  It accepts no
// frame, receiver, realization, transport, or authority data.
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
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SB_OBSERVED_SOURCE_SETUP_FEATURE_VERSION 1u
#define SB_OBSERVED_SOURCE_SETUP_LAYOUT_FEATURE 1u
#define SB_OBSERVED_SOURCE_SETUP_LAYOUT_PARTICIPANT 2u
#define SB_OBSERVED_SOURCE_SETUP_LAYOUT_POINT 3u
#define SB_OBSERVED_SOURCE_SETUP_LAYOUT_ACTOR_RESULT 4u
#define SB_OBSERVED_SOURCE_SETUP_LAYOUT_POINT_RESULT 5u

typedef struct {
  uint32_t descriptor_size;
  uint32_t feature_version;
  uint32_t participant_size;
  uint32_t point_size;
  uint32_t actor_result_size;
  uint32_t point_result_size;
  uint32_t max_participants;
  uint32_t max_points;
  uint32_t max_anchor_name_bytes;
  uint32_t reserved;
  int64_t max_age_ns;
  uint32_t coordinate_convention;
  uint32_t reserved2;
} sb_observed_source_setup_feature_t;

// Every row declares its originating C Space explicitly.  This prevents a
// numerically coincident actor/element handle from another Space being treated
// as local to this transaction.
typedef struct {
  uint32_t struct_size;
  uint32_t feature_version;
  sb_space_t space;
  sb_actor_t actor;
  const char* root_anchor_name;
  // Zero requires a fresh actor with no active observation epoch.  A nonzero
  // value permits replacement only when it exactly matches a legacy epoch.
  // Setup-owned epochs are never replaceable through this entry point.
  uint64_t expected_prior_epoch;
} sb_observed_source_setup_participant_t;

typedef struct {
  uint32_t struct_size;
  uint32_t feature_version;
  sb_space_t space;
  sb_element_t element;
  sb_actor_t producer;
  const char* anchor_name;
  int64_t maximum_age_ns;
} sb_observed_source_setup_point_t;

typedef struct {
  sb_actor_t actor;
  uint64_t epoch;
} sb_observed_source_setup_actor_result_t;

typedef struct {
  sb_element_t element;
  uint64_t lease;
} sb_observed_source_setup_point_result_t;

// Copy-safe descriptor: a null/short output is a size probe and no partial
// descriptor is written.
SB_EXPORT uint32_t sb_get_observed_source_setup_feature(
    sb_observed_source_setup_feature_t* out_feature, uint32_t out_size);
// The managed adapter must negotiate both size and member offsets before
// passing these structs across the ABI.  Unknown layout/member values return
// UINT32_MAX; the count for an unknown layout is zero.
SB_EXPORT uint32_t sb_get_observed_source_setup_field_count(uint32_t layout);
SB_EXPORT uint32_t sb_get_observed_source_setup_field_offset(uint32_t layout,
                                                               uint32_t field);

// Opens a bounded fresh observed-source setup as one transaction.  The complete request,
// including duplicates, roots, anchors, ownership, point shape, point age and
// declared Space, is validated before Core changes any actor clock, evidence,
// lease, or scene input revision.  Outputs are written only on success.
// This and the retire/release transitions return 0 without mutation if called
// from a relation callback, which runs under the current Space solve lock.
//
// A participant is fresh only when `expected_prior_epoch` is zero; otherwise
// Core replaces precisely the declared legacy epoch.  A point producer must
// name exactly one participant in this call.  `out_*_capacity` must be at
// least the matching request count.  The returned token can retire this setup
// only while no participant frame and no point publication was admitted.
SB_EXPORT int32_t sb_begin_observed_source_setup(
    sb_space_t space,
    const sb_observed_source_setup_participant_t* participants,
    uint32_t participant_count,
    const sb_observed_source_setup_point_t* points, uint32_t point_count,
    sb_observed_source_setup_t* out_setup,
    sb_observed_source_setup_actor_result_t* out_actors,
    uint32_t out_actor_capacity,
    sb_observed_source_setup_point_result_t* out_points,
    uint32_t out_point_capacity);

// Retires every epoch and lease owned by this setup only if none has admitted
// evidence.  Success is idempotent.  If later input was admitted, this returns
// 0 and leaves all state intact rather than claiming rollback.
SB_EXPORT int32_t sb_retire_observed_source_setup(
    sb_space_t space, sb_observed_source_setup_t setup);

// Releases only this token's registry capability; it never changes admitted
// Core observations, epochs, or point leases. An untouched active setup is
// deliberately unreleasable because its token is still the sole cancellation
// capability. A retired setup, a setup with any admitted frame/publication,
// or a setup made terminal/stale by Core may be released. A released token is
// no longer idempotently retireable.
SB_EXPORT int32_t sb_release_observed_source_setup(
    sb_observed_source_setup_t setup);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPATIALBLEND_C_SPATIALBLEND_OBSERVED_SOURCE_SETUP_H_
