// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SPATIALBLEND_OBSERVED_COLLABORATOR_H_
#define SPATIALBLEND_C_SPATIALBLEND_OBSERVED_COLLABORATOR_H_

// Additive boundary for coherent observed-collaborator input.  It carries
// only declared source-anchor observations and bounded source-point lifetime
// data.  It has no receiver pose, realization result, socket, room identity,
// or authority grant.
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

#define SB_OBSERVED_COLLABORATOR_FEATURE_VERSION 1u
#define SB_OBSERVED_COLLABORATOR_LAYOUT_FEATURE 1u
#define SB_OBSERVED_COLLABORATOR_LAYOUT_FRAME 2u
#define SB_OBSERVED_COLLABORATOR_LAYOUT_PUBLICATION 3u

typedef struct {
  uint32_t descriptor_size;
  uint32_t feature_version;
  uint32_t frame_size;
  uint32_t observation_size;
  uint32_t publication_size;
  uint32_t max_frames;
  uint32_t max_observations_per_frame;
  uint32_t max_publications;
  uint32_t max_anchor_name_bytes;
  uint64_t max_age_ns;
  uint32_t coordinate_convention;
  uint32_t reserved;
} sb_observed_collaborator_feature_t;

// `observations` is copied before this call returns.  Each row must have the
// exact negotiated `observation_size`; no stride or retained caller memory is
// accepted.  The source epoch/sequence remain the sole provenance clock.
typedef struct {
  uint32_t struct_size;
  uint32_t feature_version;
  sb_actor_t actor;
  uint64_t epoch;
  int64_t sequence;
  double time_seconds;
  const sb_actor_observation_t* observations;
  uint32_t observation_count;
  uint32_t observation_size;
} sb_observed_collaborator_frame_t;

// A source point is registered separately to one producer anchor.  This row
// never accepts a point coordinate or source epoch/sequence: Core derives
// both from the matching current frame before evaluating any subject.
typedef struct {
  uint32_t struct_size;
  uint32_t feature_version;
  sb_element_t element;
  uint64_t lease;
  int64_t observed_at_ns;
  int64_t valid_until_ns;
} sb_observed_collaborator_publication_t;

// Descriptor is copy-safe: a null/short destination is a size probe and no
// partial descriptor is written.  Coordinate convention 1 is Core Z-up.
SB_EXPORT uint32_t sb_get_observed_collaborator_feature(
    sb_observed_collaborator_feature_t* out_feature, uint32_t out_size);
SB_EXPORT uint32_t sb_get_observed_collaborator_field_count(uint32_t layout);
SB_EXPORT uint32_t sb_get_observed_collaborator_field_offset(uint32_t layout,
                                                               uint32_t field);

// Starts one bounded, source-anchor-only point lease. `element` must be a
// valid independently owned point element; it cannot be an actor element.
// A successful lease is current only until retirement/replacement or expiry.
SB_EXPORT int32_t sb_begin_observed_source_anchor_point(
    sb_space_t space, sb_element_t element, sb_actor_t producer,
    const char* anchor_name, int64_t maximum_age_ns, uint64_t* out_lease);
SB_EXPORT int32_t sb_retire_observed_source_anchor_point(
    sb_space_t space, sb_element_t element, uint64_t lease);

// Validates and copies every frame and publication before changing the Space.
// On success all source points are derived and published before any batch
// subject is solved. A malformed, duplicate, stale, expired, self-owned, or
// unavailable member rejects the entire batch without a prefix commit.
// Observed-collaborator mutation is also rejected (0, without state change) from a relation
// callback, which runs under the current Space solve lock.
SB_EXPORT int32_t sb_apply_observed_collaborator_batch(
    sb_space_t space, const sb_observed_collaborator_frame_t* frames,
    uint32_t frame_count,
    const sb_observed_collaborator_publication_t* publications,
    uint32_t publication_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPATIALBLEND_C_SPATIALBLEND_OBSERVED_COLLABORATOR_H_
