// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_RELATION_AVATAR_FACING_KERNEL_H_
#define SPATIALBLEND_RELATION_AVATAR_FACING_KERNEL_H_

#include <algorithm>

#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/scope_resolver.h"

namespace spatialblend {

class AvatarFacingKernel : public RelationKernel {
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
    return RelationKernelType::kAvatarFacing;
  }

  std::vector<StateType> GetDynamicStates() const override {
    return Is3DRotation() ? std::vector<StateType>{StateType::kRotation3D}
                          : std::vector<StateType>{StateType::kRotationYaw};
  }

  std::vector<StateType> GetStaticStates() const override {
    return Is3DPosition() ? std::vector<StateType>{StateType::kPosition3D}
                          : std::vector<StateType>{StateType::kPosition2D};
  }

  // Facing scope is every element the anchor could look at, at ANY
  // hierarchy depth -- not only the actor's subtree. The structural scope here
  // is the whole tree minus the actor's own element and the elements it sits
  // in (its ancestors: you do not face your container). Line of sight is then
  // enforced per candidate by the scope gate (relation/scope_gate.h,
  // IsReferentVisible), after the explicit-scope and tag filters.
  //
  // `actor_element_id` is the explicit scope root when the binding is scoped
  // (Space clips the result to that subtree).
  std::vector<element_id_t> ResolveScope(
      const Eigen::Vector3d& /*actor_position*/,
      element_id_t actor_element_id,
      const ElementTree& tree,
      element_id_t /*cached_deepest_containing*/ = kInvalidElementId)
      const override {
    // One shared implementation (ScopeResolver::ResolveVisible): every element but the actor's own
    // and its ancestors, and never an element with no shared id (D7, SP-D7-03 / FA-D7-01).
    return ScopeResolver::ResolveVisible(actor_element_id, tree);
  }
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_RELATION_AVATAR_FACING_KERNEL_H_
