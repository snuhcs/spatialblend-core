// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_RELATION_AVATAR_SPACING_KERNEL_H_
#define SPATIALBLEND_RELATION_AVATAR_SPACING_KERNEL_H_

#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/scope_resolver.h"

namespace spatialblend {

class AvatarSpacingKernel : public RelationKernel {
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
    return RelationKernelType::kAvatarSpacing;
  }

  std::vector<StateType> GetDynamicStates() const override {
    return Is3DPosition() ? std::vector<StateType>{StateType::kPosition3D}
                          : std::vector<StateType>{StateType::kPosition2D};
  }

  std::vector<StateType> GetStaticStates() const override {
    return Is3DRotation() ? std::vector<StateType>{StateType::kRotation3D}
                          : std::vector<StateType>{StateType::kRotationYaw};
  }

  std::vector<element_id_t> ResolveScope(
      const Eigen::Vector3d& actor_position,
      element_id_t actor_element_id,
      const ElementTree& tree,
      element_id_t cached_deepest_containing = kInvalidElementId)
      const override {
    element_id_t scope = (cached_deepest_containing != kInvalidElementId)
                             ? cached_deepest_containing
                             : FindDeepestContaining(
                                   actor_position, actor_element_id, tree);
    if (scope == kInvalidElementId) {
      return {};
    }
    // One shared implementation (ScopeResolver): the deepest container's
    // objects, plus the container itself while the actor is inside it.
    return ScopeResolver::Resolve(ScopePolicy::kReachable, actor_position,
                                  actor_element_id, tree, scope);
  }
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_RELATION_AVATAR_SPACING_KERNEL_H_
