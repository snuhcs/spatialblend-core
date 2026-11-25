// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_CONTAINMENT_TRANSFER_H_
#define SPATIALBLEND_SPACE_CONTAINMENT_TRANSFER_H_

// Containment transfer (part of the scope layer).
//
// A Containment element flagged for transfer (Space::SetElementContainmentTransfer)
// is a transfer LEAF when both spaces declare it and its shapes are congruent under
// a rigid transform (whether it carries an object does not matter). While an actor's root is inside it, the leaf IS the actor's whole
// scope, so the empty-scope mapping maps the root
// through the leaf's chart onto the receiver's counterpart. That mapping is the
// containment transfer; there is no second mechanism. Core therefore sends and keeps
// no contained pose, no blend weight and no HANDOVER state (no side
// channels).
//
// What survives is the FACT that the actor is inside a transfer leaf: the identity of
// the leaf on the wire (ContainmentObservation), the CONTAIN_ENTER / CONTAIN_EXIT
// cause bits of the jump update, and the per-update transfer flag of the receiver's
// root proposal (kAnchorFlagTransfer), which is recomputed on every update and
// stored nowhere. There is no contained event.

#include <cstdint>
#include <string>

#include "spatialblend/base/common/types.h"

namespace spatialblend {

// Optional wire field of RemoteRelationUpdate. Present only while the source
// root is inside a transfer element; absent means "not contained" (and is how an
// old peer's update decodes).
struct ContainmentObservation {
  // Identity of the containing element, same triple the referents use.
  shared_element_id_t scope_shared_id = kInvalidSharedElementId;
  space_id_t scope_source_space_id = kInvalidSpaceId;
  element_id_t scope_source_element_id = kInvalidElementId;

  bool operator==(const ContainmentObservation& o) const {
    return scope_shared_id == o.scope_shared_id &&
           scope_source_space_id == o.scope_source_space_id &&
           scope_source_element_id == o.scope_source_element_id;
  }
};

// ABI 4.14 ("replay through the chart" as its own output): one source anchor's pose in the frame of
// the transfer leaf that contains it, read from that anchor's OWN sample (a head turned against the root carries its own
// yaw; there is no head offset). Sent for every active bound anchor whose pose is inside a transfer leaf; the receiver
// maps it through its counterpart of the leaf and exposes it beside the Spacing / Facing answers (sb_read_anchor_chart).
// Pitch and roll stay the anchor's own pass-through (as on its proposal): a leaf turns about the vertical only.
struct AnchorChartObservation {
  std::string anchor_name;                    // the source anchor ("" or "root" is the root)
  shared_element_id_t scope_shared_id = kInvalidSharedElementId;  // the leaf, as ContainmentObservation names it
  space_id_t scope_source_space_id = kInvalidSpaceId;
  element_id_t scope_source_element_id = kInvalidElementId;
  double x = 0.0, y = 0.0, z = 0.0;           // the anchor in the leaf's element frame (Z-up, metres)
  double yaw = 0.0;                           // the anchor's yaw less the leaf's world yaw (radians)

  bool operator==(const AnchorChartObservation& o) const {
    return anchor_name == o.anchor_name && scope_shared_id == o.scope_shared_id &&
           scope_source_space_id == o.scope_source_space_id && scope_source_element_id == o.scope_source_element_id &&
           x == o.x && y == o.y && z == o.z && yaw == o.yaw;
  }
};

// Per-actor last observed contained identity. For a local (source) actor it is
// computed from the local root pose (the source sends a tick on which it changed);
// for a remote actor proxy it is the latest received observation, which the cause
// bits compare against the previous proposal.
struct ActorContainment {
  bool contained = false;
  shared_element_id_t scope_shared_id = kInvalidSharedElementId;
  // Identity of the containing element at the SOURCE ({space, element}); used
  // for transition detection.
  space_id_t scope_source_space_id = kInvalidSpaceId;
  element_id_t scope_source_element_id = kInvalidElementId;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_CONTAINMENT_TRANSFER_H_
