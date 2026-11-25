// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_RELATION_STATE_H_
#define SPATIALBLEND_SPACE_RELATION_STATE_H_

#include <Eigen/Dense>
#include <limits>
#include <string>

#include "spatialblend/base/common/types.h"
#include "spatialblend/geometry/polygon_coordinate_mapper.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/space/relation_event.h"

namespace spatialblend {

// A read-only snapshot of one (anchor, referent) relation's currently
// grounded state (relation-state query — prerequisite for the Python
// relation oracle and for real tracked hands).
//
// Populated directly from the owning Binding's latest_matches (see
// Space::UpdateActorState) — this is a PULL query over already-propagated
// numbers, never a recomputation, so it always agrees with whatever was just
// sent over the wire in the same frame.
struct RelationState {
  std::string anchor_name;
  RelationKernelType kernel_type = RelationKernelType::kAvatarFacing;

  // Referent identity: local element id (always valid while the element is
  // still in this space's tree) and shared id (stable across spaces; equals
  // referent_element_id when the referent element isn't shared).
  element_id_t referent_element_id = kInvalidElementId;
  shared_element_id_t referent_shared_id = kInvalidSharedElementId;

  // Normalized proximity: 0 = far/disengaged, 1 = fully engaged. Same value
  // as GroundedRelation::proximity for this match (called R).
  double proximity = 0.0;

  // Support (contact) info, copied from Binding::MatchResult::eval.support:
  bool has_contact = false;
  // Contact point on the referent boundary, WORLD frame.
  Eigen::Vector3d contact_point_world = Eigen::Vector3d::Zero();
  // Polygon-normalized contact position (radial or MVC encoding depending on
  // referent boundary convexity). mvc_weights is intentionally NOT surfaced
  // through the C API (variable-length, doesn't fit a POD struct) but is
  // available here and via the Python binding.
  NormalizedPolygonPosition contact_normalized;

  // Whether RelationPhaseTracker currently considers this relation engaged
  // beyond the dormant floor (kActive/kDriving/kContact), i.e. not kDormant.
  bool active = false;

  // Extensions (sb_relation_state_v2_t), copied from the same match:
  // the owning binding (index into the actor's bindings, in bind order),
  size_t binding_index = 0;
  // the tracker phase behind `active`,
  RelationPhase phase = RelationPhase::kDormant;
  // the named sub-region the support is grounded on (empty = whole boundary),
  std::string sub_region_name;
  // the kernel's encoded state (GroundedRelation::encoded_state),
  Eigen::VectorXd encoded_state;
  // the absolute support distance (m; < 0 unset) and the diameter of the
  // scope that normalized R (m; <= 0 unset),
  double support_distance_m = -1.0;
  double source_scope_diameter_m = -1.0;
  // and the interior chart coordinate when the anchor is inside the referent.
  bool has_interior = false;
  Eigen::Vector3d interior_uvw = Eigen::Vector3d::Zero();
  // AffordancePose only -- the anchor's pose relative to the
  // support (GroundedRelation::ap_offset_u_m/ap_offset_v_m/ap_height_m).
  // NaN = unknown/not this family.
  double ap_offset_u_m = std::numeric_limits<double>::quiet_NaN();
  double ap_offset_v_m = std::numeric_limits<double>::quiet_NaN();
  double ap_height_m = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_RELATION_STATE_H_
