// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_RELATION_GROUNDING_RULE_H_
#define SPATIALBLEND_RELATION_GROUNDING_RULE_H_

#include <Eigen/Dense>

#include "spatialblend/model/element/element.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/relation/support.h"

namespace spatialblend {

class GroundingRule {
 public:
  virtual ~GroundingRule() = default;

  // Performance hook for callers that read only the relation VALUE (its
  // proximity) of the kernel the support feeds, never the support's chart
  // encoding: a rule that honours it skips the normalized-position chart of
  // the contact point (Support::contact_normalized, a polygon-chart encode
  // plus a nearest-boundary-point query per call). Everything the kernel's
  // proximity depends on -- contact point, containment (has_interior),
  // navigable scope -- is computed either way. Default false: every
  // existing caller keeps the full support. The paper-objective search
  // (relation_solver.cc) evaluates thousands of candidate poses per solve
  // and needs only R.
  bool skip_chart_encoding = false;

  // The caller measures the relation against a contact of its own (the mapped
  // source contact), so the geodesic-nearest support contact is not needed:
  // a rule that honours it skips that ring-field query. Containment is
  // computed either way. Default false.
  bool skip_nearest_contact = false;

  virtual Support ComputeSupport(
      const Element& referent, const ElementTree& tree,
      const Eigen::Vector3d& anchor_pos,
      double anchor_yaw,
      const NavigableSpatialDefinition* navigable_def = nullptr) const = 0;
};

class DefaultGroundingRule : public GroundingRule {
 public:
  Support ComputeSupport(
      const Element& referent, const ElementTree& tree,
      const Eigen::Vector3d& anchor_pos,
      double anchor_yaw,
      const NavigableSpatialDefinition* navigable_def = nullptr) const override {
    Support support;
    support.world_transform = tree.GetWorldTransform(referent.GetId());
    auto* spatial_def = referent.GetDeclaredSpatialDefinition();
    if (spatial_def) {
      support.boundary = spatial_def->GetBoundary();
    }
    support.referent_element_id = referent.GetId();
    support.navigable_def = navigable_def;
    return support;
  }
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_RELATION_GROUNDING_RULE_H_
