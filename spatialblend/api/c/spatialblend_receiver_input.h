// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SPATIALBLEND_RECEIVER_INPUT_H_
#define SPATIALBLEND_C_SPATIALBLEND_RECEIVER_INPUT_H_

// Separately negotiated external receiver-pose input. ABI2, rich feature-v1,
// validity-v1 and rich JSON-v1 remain unchanged. This is not a complete body
// source, calibrated rig/DOF admission, contact certificate, or motion plan.
#include "spatialblend_rich.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SB_RECEIVER_INPUT_FEATURE_VERSION 1u

typedef struct {
  const uint8_t* name_utf8;
  uint32_t name_bytes;
  uint32_t available;  // Exactly 0 or 1, even for the required root row.
  double world_pose_row_major[16];  // Finite rigid receiver-world Z-up pose.
} sb_receiver_anchor_input_t;

typedef struct {
  uint32_t struct_size;
  uint32_t feature_version;
  uint32_t anchor_count;
  uint32_t anchor_stride;
  uint64_t epoch;
  uint64_t sequence;
  uint64_t monotonic_clock_id;
  uint64_t monotonic_clock_epoch;
  int64_t observed_at_ns;
  int64_t valid_until_ns;
  const sb_receiver_anchor_input_t* anchors;
} sb_receiver_input_frame_t;

typedef struct {
  uint64_t epoch;
  uint64_t revision;
  uint64_t sequence;
  uint64_t rig_revision;
  uint64_t calibration_revision;
  int64_t observed_at_ns;
  int64_t valid_until_ns;
  uint32_t known;
  uint32_t reserved;
} sb_receiver_input_stamp_t;

typedef struct {
  uint64_t producer_lifetime;
  uint64_t revision;
  sb_receiver_input_stamp_t captured;
  sb_receiver_input_stamp_t current;
  uint32_t changed;
  uint32_t valid;
} sb_receiver_input_validity_t;

typedef struct {
  uint32_t descriptor_size;
  uint32_t feature_version;
  uint32_t anchor_size;
  uint32_t frame_size;
  uint32_t stamp_size;
  uint32_t validity_size;
  uint32_t max_anchors;
  uint32_t max_name_bytes;
  uint32_t coordinate_convention;  // 1 = Z-up, row-major rigid transforms.
  uint32_t reserved;
  uint64_t max_age_ns;
  uint64_t monotonic_clock_id;
  uint64_t monotonic_clock_epoch;
} sb_receiver_input_feature_t;

typedef enum {
  SB_RECEIVER_INPUT_LAYOUT_FEATURE = 1,
  SB_RECEIVER_INPUT_LAYOUT_ANCHOR = 2,
  SB_RECEIVER_INPUT_LAYOUT_FRAME = 3,
  SB_RECEIVER_INPUT_LAYOUT_STAMP = 4,
  SB_RECEIVER_INPUT_LAYOUT_VALIDITY = 5,
} sb_receiver_input_layout_t;

// Size probe does not write a short buffer. Field indices follow declaration
// order in the selected struct; an invalid kind/index returns 0/UINT32_MAX.
SB_EXPORT uint32_t sb_get_receiver_input_feature(
    sb_receiver_input_feature_t* feature, uint32_t feature_size);
SB_EXPORT uint32_t sb_get_receiver_input_field_count(uint32_t struct_kind);
SB_EXPORT uint32_t sb_get_receiver_input_field_offset(uint32_t struct_kind,
                                                      uint32_t field_index);

// All functions below return sb_rich_realization_result_t values. Begin owns
// a new nonzero epoch; it captures the actor's complete current anchor layout.
// The root must be unparented, zero-offset, Position2D and not Position3D.
// Full rigid input poses retain XYZ, but Core locomotion is planar.
// This path only emits geometric goals: even configured legacy Core
// controllers cannot actuate it. Runtime body/DOF admission remains separate.
// Source-observation/published-source actors cannot silently become receivers.
// After Begin, source setters are rejected/no-op; the driver's embodied pose is published by the
// receiver-input frames themselves (sb_publish_anchor_pose is refused: -3). Receiver mode ends at actor destruction,
// not at Clear. Structural rebinding/layout changes require another Begin.
// Rig/calibration revisions are nonzero caller identities, NOT proof that
// either calibration has been reviewed or that the poses were measured.
SB_EXPORT int32_t sb_begin_receiver_input(
    sb_space_t space, sb_actor_t actor, uint32_t feature_version,
    const uint8_t* root_name_utf8, uint32_t root_name_bytes,
    uint64_t rig_revision, uint64_t calibration_revision, uint64_t* epoch);

// One atomic full frame: every configured anchor exactly once, no implicit
// retained offsets/poses. Unavailable root is an accepted loss event. Poses
// must be rigid even when unavailable. Timestamp uses the calibrated native
// clock from this descriptor (read with sb_read_rich_monotonic_ns); it cannot
// be source replay time or a timestamp restamped at submission. Expiry must
// be after now and at most max_age_ns after observation. No propagation,
// source-clock change, pose write, rich callback or new rich revision occurs.
// Buffers must be allocated to their declared extents; arbitrary pointers are
// not safely validated. C session lifetime retains its single-owner contract.
SB_EXPORT int32_t sb_submit_receiver_input(
    sb_space_t space, sb_actor_t actor, const sb_receiver_input_frame_t* frame,
    uint32_t frame_size);
SB_EXPORT int32_t sb_clear_receiver_input(sb_space_t space, sb_actor_t actor,
                                          uint64_t epoch);

// Nonmutating comparison of the frozen rich token+revision's input sidecar
// against current input. Does not poll or advance the rich cursor. OK can
// report valid=0 for a newer frame; UNAVAILABLE reports missing/expired or
// uncertified input. Every scene/policy/support/DOF/actuator dependency must
// still be checked independently. Validity-v1 receiver-input stays unchecked.
SB_EXPORT int32_t sb_get_receiver_input_validity(
    sb_rich_realization_token_t token, uint64_t rich_revision,
    sb_receiver_input_validity_t* out, uint32_t out_size);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // SPATIALBLEND_C_SPATIALBLEND_RECEIVER_INPUT_H_
