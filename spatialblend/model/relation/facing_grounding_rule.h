// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_RELATION_FACING_GROUNDING_RULE_H_
#define SPATIALBLEND_RELATION_FACING_GROUNDING_RULE_H_

#include <optional>

#include <Eigen/Dense>

#include "spatialblend/model/relation/grounding_rule.h"

namespace spatialblend {

class FacingGroundingRule : public GroundingRule {
 public:
  Support ComputeSupport(
      const Element& referent, const ElementTree& tree,
      const Eigen::Vector3d& anchor_pos, double anchor_yaw,
      const NavigableSpatialDefinition* navigable_def = nullptr) const override;

  // The anchor position of the source's previous evaluation of this binding (unset on the
  // first). The support reports whether the path from it to the current anchor crossed the
  // referent's boundary (Support::look_at_boundary_crossed). The only state the rule reads.
  void SetPreviousAnchorPosition(const std::optional<Eigen::Vector3d>& previous) {
    previous_anchor_ = previous;
  }

 private:
  std::optional<Eigen::Vector3d> previous_anchor_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_RELATION_FACING_GROUNDING_RULE_H_
