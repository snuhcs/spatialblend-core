// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_RELATION_KERNEL_FACTORY_H_
#define SPATIALBLEND_RELATION_KERNEL_FACTORY_H_

#include <memory>

#include "spatialblend/model/relation/affordance_pose_grounding_rule.h"
#include "spatialblend/model/relation/avatar_affordance_pose_kernel.h"
#include "spatialblend/model/relation/avatar_facing_kernel.h"
#include "spatialblend/model/relation/avatar_spacing_kernel.h"
#include "spatialblend/model/relation/facing_grounding_rule.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/spacing_grounding_rule.h"

namespace spatialblend {

struct RelationComponents {
  std::unique_ptr<RelationKernel> kernel;
  std::unique_ptr<GroundingRule> grounding_rule;
};

class KernelFactory {
 public:
  static RelationComponents Create(
      RelationKernelType type,
      const std::vector<StateType>& dof = {StateType::kPosition2D,
                                            StateType::kRotationYaw}) {
    RelationComponents result;
    switch (type) {
      case RelationKernelType::kAvatarFacing:
        result.kernel = std::make_unique<AvatarFacingKernel>();
        result.grounding_rule = std::make_unique<FacingGroundingRule>();
        break;
      case RelationKernelType::kAvatarSpacing:
        result.kernel = std::make_unique<AvatarSpacingKernel>();
        result.grounding_rule = std::make_unique<SpacingGroundingRule>();
        break;
      case RelationKernelType::kAvatarAffordancePose:
        result.kernel = std::make_unique<AvatarAffordancePoseKernel>();
        result.grounding_rule = std::make_unique<AffordancePoseGroundingRule>();
        break;
    }
    if (result.kernel) {
      result.kernel->SetDOF(dof);
    }
    return result;
  }
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_RELATION_KERNEL_FACTORY_H_
