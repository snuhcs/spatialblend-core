// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_RELATION_AVATAR_AFFORDANCE_POSE_KERNEL_H_
#define SPATIALBLEND_RELATION_AVATAR_AFFORDANCE_POSE_KERNEL_H_

#include <algorithm>
#include <optional>

#include "spatialblend/model/element/element.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/scope_resolver.h"

namespace spatialblend {

// AffordancePose has TWO distinct quantities, deliberately split.
//
//  1. GroundedRelation::proximity = 1 / (1 + anchor-to-support distance): the
//     GEOMETRIC proximity component. It is the only thing the activation /
//     Driving / Contact hysteresis, scheduling and the relation weights read
//     (RelationPhaseConfig contact_proximity 0.95 / contact_release 0.85), so
//     "in contact" keeps its original geometric meaning (a pelvis releases at
//     ~0.18 m, a hand at any yaw can still enter Contact). Unchanged by the pose fit,
//     and byte-identical to what a pre-pose-fit sender emits.
//
//  2. R, the POSE FIT (AffordancePoseR below): in-plane offset, height above
//     the support plane and yaw, each normalized to [0, 1] with 1 = exact and
//     combined with the declared weights. R is the relation-preservation
//     score: it is what Eval_p (EvaluateTargetRelation r_src / r_tgt / error)
//     and the metrics built on it compare. R never drives phase hysteresis.
//
// R's inputs are carried on the payload as ap_offset_u_m / ap_offset_v_m /
// ap_height_m (support-local); the formula version is implied by their
// presence (AffordancePoseRFormulaVersion). A pre-pose-fit sender (offsets NaN) has
// R_src = proximity under formula version 0; a receiver then scores BOTH
// sides with the proximity component only (never a new-formula R_tgt against
// an old-formula R_src).
//
// Seat front. The encoded relative yaw is
// `anchor_yaw - part_yaw`, measured from the +x axis of the affordance PART frame (the seat part, for a sit), and
// the receiver's target yaw is its own part's +x plus that relative yaw. Both apps must therefore author the
// part frame of corresponding seats with the same meaning of +x (the seat's front); core never infers a front.
//
// Declared weights (sum to 1, so R = 1 iff every fit term = 1).
inline constexpr double kApPositionWeight = 0.5;
inline constexpr double kApHeightWeight = 0.3;
inline constexpr double kApYawWeight = 0.2;

// R formula versions. 0 = proximity-only (pre-pose-fit payload: no offset/height).
// 1 = pose fit (AffordancePoseR).
inline constexpr int kApRFormulaProximityOnly = 0;
inline constexpr int kApRFormulaPoseFitV1 = 1;

// The R formula version `gr` was encoded under: kApRFormulaPoseFitV1 iff all
// of ap_offset_u_m / ap_offset_v_m / ap_height_m are finite, else
// kApRFormulaProximityOnly.
int AffordancePoseRFormulaVersion(const GroundedRelation& gr);

// R (formula version 1): a declared monotone combination of the anchor's
// position, height and yaw fit relative to the support. `yaw_error_rad` is the
// anchor's relative yaw minus the SOURCE's relative yaw (the source's own
// R_src passes 0: its yaw is its own reference); nullopt = the source's
// relative yaw is unknown, and the yaw term then does not contribute (the
// position and height weights are renormalized to sum to 1). Callers must
// pass the same availability for R_src and R_tgt.
double AffordancePoseR(double offset_u_m, double offset_v_m, double height_m,
                       std::optional<double> yaw_error_rad);

class AvatarAffordancePoseKernel : public RelationKernel {
 public:
  GroundedRelation Encode(const Eigen::Vector3d& anchor_world_position,
                          double anchor_world_yaw,
                          const Support& support) const override;

  double Compare(const GroundedRelation& source,
                 const GroundedRelation& target) const override;

  SatisfactionMetric ComputeSatisfactionMetric(
      const GroundedRelation& current,
      const GroundedRelation* previous,
      const ActorMotionState& motion) const override;

  RelationKernelType GetType() const override {
    return RelationKernelType::kAvatarAffordancePose;
  }

  std::vector<element_id_t> ResolveScope(
      const Eigen::Vector3d& actor_position,
      element_id_t actor_element_id,
      const ElementTree& tree,
      element_id_t cached_deepest_containing = kInvalidElementId)
      const override {
    // AffordancePose evaluates from any distance (approach with low proximity)
    // but engagement increases when actor enters parent boundary.
    // Scope = all descendants of actor's root element (tag filter applied later).
    // One shared implementation (ScopeResolver::ResolveProximal), which also drops every element with
    // no shared id (D7, SP-D7-03).
    return ScopeResolver::ResolveProximal(actor_position, actor_element_id, tree,
                                          cached_deepest_containing);
  }

  std::vector<StateType> GetDynamicStates() const override {
    std::vector<StateType> result;
    result.push_back(Is3DPosition() ? StateType::kPosition3D : StateType::kPosition2D);
    result.push_back(Is3DRotation() ? StateType::kRotation3D : StateType::kRotationYaw);
    return result;
  }
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_RELATION_AVATAR_AFFORDANCE_POSE_KERNEL_H_
