// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_RELATION_AFFORDANCE_POSE_GROUNDING_RULE_H_
#define SPATIALBLEND_RELATION_AFFORDANCE_POSE_GROUNDING_RULE_H_

#include "spatialblend/model/relation/grounding_rule.h"

namespace spatialblend {

class AffordancePoseGroundingRule : public GroundingRule {
 public:
  Support ComputeSupport(
      const Element& referent, const ElementTree& tree,
      const Eigen::Vector3d& anchor_pos, double anchor_yaw,
      const NavigableSpatialDefinition* navigable_def = nullptr) const override;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_RELATION_AFFORDANCE_POSE_GROUNDING_RULE_H_
