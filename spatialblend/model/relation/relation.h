// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_RELATION_RELATION_H_
#define SPATIALBLEND_RELATION_RELATION_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/relation/grounded_relation.h"
#include "spatialblend/model/relation/grounding_rule.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/support.h"

namespace spatialblend {

struct RelationEvaluationResult {
  GroundedRelation grounded;
  Support support;
};

class Relation {
 public:
  Relation(const std::string& name, const std::string& subject_anchor_name,
           element_id_t referent_element_id,
           std::unique_ptr<GroundingRule> grounding_rule,
           std::unique_ptr<RelationKernel> kernel);

  const std::string& GetName() const;
  const std::string& GetSubjectAnchorName() const;
  element_id_t GetReferentElementId() const;
  RelationKernelType GetKernelType() const;

  RelationEvaluationResult Evaluate(
      const Eigen::Vector3d& anchor_world_position,
      double anchor_world_yaw, const ElementTree& tree,
      const NavigableSpatialDefinition* navigable_def = nullptr) const;

  absl::StatusOr<RelationEvaluationResult> TryEvaluate(
      const Eigen::Vector3d& anchor_world_position,
      double anchor_world_yaw, const ElementTree& tree,
      const NavigableSpatialDefinition* navigable_def = nullptr) const;

  double Compare(const GroundedRelation& source,
                 const GroundedRelation& target) const;

 private:
  std::string name_;
  std::string subject_anchor_name_;
  element_id_t referent_element_id_;
  std::unique_ptr<GroundingRule> grounding_rule_;
  std::unique_ptr<RelationKernel> kernel_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_RELATION_RELATION_H_
