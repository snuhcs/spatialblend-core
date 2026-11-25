// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/relation.h"

#include "absl/status/status.h"

namespace spatialblend {

Relation::Relation(const std::string& name,
                   const std::string& subject_anchor_name,
                   element_id_t referent_element_id,
                   std::unique_ptr<GroundingRule> grounding_rule,
                   std::unique_ptr<RelationKernel> kernel)
    : name_(name),
      subject_anchor_name_(subject_anchor_name),
      referent_element_id_(referent_element_id),
      grounding_rule_(std::move(grounding_rule)),
      kernel_(std::move(kernel)) {}

const std::string& Relation::GetName() const { return name_; }
const std::string& Relation::GetSubjectAnchorName() const {
  return subject_anchor_name_;
}
element_id_t Relation::GetReferentElementId() const {
  return referent_element_id_;
}
RelationKernelType Relation::GetKernelType() const {
  return kernel_->GetType();
}

RelationEvaluationResult Relation::Evaluate(
    const Eigen::Vector3d& anchor_world_position,
    double anchor_world_yaw, const ElementTree& tree,
    const NavigableSpatialDefinition* navigable_def) const {
  auto result = TryEvaluate(anchor_world_position, anchor_world_yaw, tree,
                            navigable_def);
  return *std::move(result);
}

absl::StatusOr<RelationEvaluationResult> Relation::TryEvaluate(
    const Eigen::Vector3d& anchor_world_position,
    double anchor_world_yaw, const ElementTree& tree,
    const NavigableSpatialDefinition* navigable_def) const {
  const Element* referent = tree.GetElement(referent_element_id_);
  if (!referent) {
    return absl::NotFoundError("Referent element not found in tree");
  }

  Support support =
      grounding_rule_->ComputeSupport(*referent, tree, anchor_world_position, anchor_world_yaw, navigable_def);
  GroundedRelation grounded =
      kernel_->Encode(anchor_world_position, anchor_world_yaw, support);

  RelationEvaluationResult eval_result;
  eval_result.grounded = std::move(grounded);
  eval_result.support = std::move(support);
  return eval_result;
}

double Relation::Compare(const GroundedRelation& source,
                         const GroundedRelation& target) const {
  return kernel_->Compare(source, target);
}

}  // namespace spatialblend
