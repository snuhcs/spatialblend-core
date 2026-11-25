// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SPATIALBLEND_CONTAINMENT_TRANSFER_H_
#define SPATIALBLEND_C_SPATIALBLEND_CONTAINMENT_TRANSFER_H_

// Containment transfer (part of the scope layer; ABI 4). Independent optional feature.
//
// A Containment element flagged with sb_set_element_containment_transfer, on both
// sides and with congruent shapes, is a transfer LEAF (whether it carries an object
// does not matter). While an actor's root is
// inside it, the leaf IS the actor's whole scope, so the empty-scope mapping maps
// the root through the leaf's chart onto the receiver's counterpart. That
// mapping is the containment transfer -- there is no second mechanism, no
// contained pose, no blend band and no HANDOVER state beside the root proposal
// (no side channels). Facing is untouched (it stays
// global).
//
// What survives is the FACT that the actor is inside a transfer element: the
// SB_ANCHOR_PROPOSAL_FLAG_TRANSFER of the root proposal (ABI 4.3, recomputed on
// every update) and the CONTAIN_ENTER / CONTAIN_EXIT cause bits of the jump
// update. ABI 3's sb_get_actor_containment (state, alpha and the mapped contained
// pose) and ABI 4's contained event (sb_register_on_containment_event) are removed. ABI 4.4 adds sb_get_source_transfer.

#include "spatialblend_rich.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SB_CONTAINMENT_TRANSFER_FEATURE_VERSION 4u

typedef struct {
  uint32_t descriptor_size, feature_version;
  uint32_t coordinate_convention;  // 1: Z-up metres, yaw radians about Z.
  uint32_t reserved;
} sb_containment_transfer_feature_t;

// kinds: 1 = feature. Fields in declaration order; invalid kind/index -> 0 /
// UINT32_MAX. Short descriptor probes do not write.
SB_EXPORT uint32_t sb_get_containment_transfer_feature(
    sb_containment_transfer_feature_t* out, uint32_t size);
SB_EXPORT uint32_t sb_get_containment_transfer_field_count(uint32_t kind);
SB_EXPORT uint32_t sb_get_containment_transfer_field_offset(uint32_t kind,
                                                            uint32_t index);

// Element-local polygon. points_xy holds num_points (x, y) pairs in the
// ELEMENT's own frame, so containment follows later sb_set_element_transform
// calls on the element or its ancestors. (ABI 3.1: sb_set_element_polygon reads
// element-local points too, except NAVIGABLE on a root; this setter is an alias.) `role` must be a valid sb_geometry_role_t other than NAVIGABLE and
// SURFACE; a room stays world-absolute. Returns 1 on success, 0 on invalid
// input or inside a relation/containment callback.
SB_EXPORT int32_t sb_set_element_local_polygon(
    sb_space_t space, sb_element_t elem, sb_geometry_role_t role,
    const double* points_xy, int64_t num_points);

// Flag (enabled != 0) or clear a Containment element for containment transfer:
// declare the region for replay. Declare it on BOTH sides. The element must
// already have a Containment-role definition (a local polygon, box or collider);
// returns 1 on success, 0 when it does not, for an unknown element, and inside a
// callback. Clearing returns 1 for any known element.
//
// A transfer leaf is a region declared on both sides whose shapes are equivalent
// in the two spaces: congruent under a rigid transform (rotation and translation,
// no reflection, the geometry epsilon relative to the footprint). The declaration is
// never refused for its shape: equivalence is judged on every update, as part of the
// receiver's SB_ANCHOR_PROPOSAL_FLAG_TRANSFER, so a late counterpart, a geometry edit
// or a counterpart in another process is picked up on the next update, and a
// non-congruent one leaves the flag clear. Whether the region carries an object does
// not matter.
SB_EXPORT int32_t sb_set_element_containment_transfer(sb_space_t space,
                                                      sb_element_t elem,
                                                      int32_t enabled);

// ABI 4.3: there is NO contained event. The receiver's contained state is the per-update
// SB_ANCHOR_PROPOSAL_FLAG_TRANSFER of the root proposal (sb_read_proposals); an app reads
// it together with the root's SB_ANCHOR_PROPOSAL_FLAG_ARRIVED and derives enter / leave
// from the frame-to-frame change.

// ABI 4.4: the SOURCE-side transfer fact. Returns 1 and the shared id of the transfer
// leaf the source actor's root was inside at its latest update, 0 (id untouched) when it
// was inside none, and -1 for an unknown actor, a null out pointer, or inside a callback.
// It is what that update computed (the single judgment, in core); nothing else is stored.
SB_EXPORT int32_t sb_get_source_transfer(sb_space_t space, sb_actor_t actor,
                                         int64_t* leaf_shared_id);

#ifdef __cplusplus
}
#endif
#endif  // SPATIALBLEND_C_SPATIALBLEND_CONTAINMENT_TRANSFER_H_
