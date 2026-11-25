// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_RELATION_SPACING_GROUNDING_RULE_H_
#define SPATIALBLEND_RELATION_SPACING_GROUNDING_RULE_H_

#include <cstdint>
#include <vector>

#include "spatialblend/model/relation/grounding_rule.h"

namespace spatialblend {

// The referent's OWN footprints (v1: reaching a target is entering its
// affordance volume): the obstacle-role footprints of the referent, of what it carries, and of the objects it stands in
// up to the navigable it is in. They never block the final leg onto the referent's contact. One rule for both sides: the
// source's nearest contact (SpacingGroundingRule, PickGeodesicContact) and the receiver's mapped contact
// (relation_evaluator.cc, relation_solver.cc).
std::vector<int64_t> OwnFootprints(const ElementTree& tree, const Element& referent);

class SpacingGroundingRule : public GroundingRule {
 public:
  Support ComputeSupport(
      const Element& referent, const ElementTree& tree,
      const Eigen::Vector3d& anchor_pos, double anchor_yaw,
      const NavigableSpatialDefinition* navigable_def = nullptr) const override;

 private:
  // Search-mode cache (only used while skip_chart_encoding is set; see
  // GroundingRule). The referent's world transform and its inverse do not
  // depend on the candidate pose, and the paper-objective search asks for
  // the support at thousands of candidates of ONE relation on an immutable
  // tree, so they are computed once per (tree, referent). The rule instance
  // is owned by one prepared relation of one Solve() call (single thread,
  // tree unchanged for its lifetime), which is what makes this safe.
  mutable const ElementTree* cache_tree_ = nullptr;
  mutable element_id_t cache_referent_ = kInvalidElementId;
  mutable Eigen::Matrix4d cache_to_world_ = Eigen::Matrix4d::Identity();
  mutable Eigen::Matrix4d cache_to_local_ = Eigen::Matrix4d::Identity();
  mutable bool cache_segment_ = false;  // the cached referent's ground image is a segment
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_RELATION_SPACING_GROUNDING_RULE_H_
