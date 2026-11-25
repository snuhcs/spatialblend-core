// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SPATIALBLEND_RECEIVER_REEVALUATION_H_
#define SPATIALBLEND_C_SPATIALBLEND_RECEIVER_REEVALUATION_H_

#include "spatialblend_rich.h"

#ifdef __cplusplus
extern "C" {
#endif

// Independent negotiation; no existing receiver/rich/validity-v1 layout changes.
#define SB_RECEIVER_REEVALUATION_FEATURE_VERSION 1u
#define SB_RECEIVER_REEVALUATION_PRESERVES_SOURCE_CAUSE (1u << 0)
#define SB_RECEIVER_REEVALUATION_EXACT_INPUT_ONCE (1u << 1)
#define SB_RECEIVER_REEVALUATION_TARGET_ONLY (1u << 2)
#define SB_RECEIVER_REEVALUATION_LAYOUT_FEATURE 1u

typedef struct {
  uint32_t descriptor_size;
  uint32_t feature_version;
  uint32_t required_receiver_input_version;
  uint32_t required_rich_version;
  uint32_t guarantees;
  uint32_t reserved;
} sb_receiver_reevaluation_feature_t;

// Short/null probe does not write. Field indices follow declaration order.
SB_EXPORT uint32_t sb_get_receiver_reevaluation_feature(
    sb_receiver_reevaluation_feature_t* feature, uint32_t feature_size);
SB_EXPORT uint32_t sb_get_receiver_reevaluation_field_count(uint32_t kind);
SB_EXPORT uint32_t sb_get_receiver_reevaluation_field_offset(uint32_t kind,
                                                           uint32_t index);

// Explicit rich-only local evaluation; submit and poll remain non-solving.
// Requires this exact active token, a known live external input with the
// expected nonzero epoch/sequence, and an admitted source packet retained in
// this rich/input lifetime. OK emits a new rich revision; NO_NEW_SNAPSHOT means
// this input was already evaluated, including empty or failed geometry.
// STALE_SNAPSHOT means another input is current; UNAVAILABLE means no live
// input/source/activation. No legacy target callbacks, source publication,
// source clock/sequence changes, or display/body writes occur.
// Original source causal fields AND receiver_ingress_monotonic_ns are kept.
// Only solve time, input stamp, current geometry/revisions can advance. This
// does not renew source freshness: adapters must still apply the ORIGINAL
// ingress-age/deadline policy. The source packet has no implicit Core TTL.
// Single-owner C session lifetime rules still apply; relation-callback reentry
// is rejected before locks. Rich callbacks run after native locks are released.
SB_EXPORT int32_t sb_reevaluate_receiver_realization(
    sb_rich_realization_token_t token, uint32_t feature_version,
    uint64_t expected_epoch, uint64_t expected_sequence);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // SPATIALBLEND_C_SPATIALBLEND_RECEIVER_REEVALUATION_H_
