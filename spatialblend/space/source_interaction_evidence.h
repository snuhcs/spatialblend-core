// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_SOURCE_INTERACTION_EVIDENCE_H_
#define SPATIALBLEND_SPACE_SOURCE_INTERACTION_EVIDENCE_H_

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "spatialblend/base/common/common.h"
#include "spatialblend/base/common/types.h"
#include "spatialblend/model/element/geometry_role.h"
#include "spatialblend/space/relation_event.h"

namespace spatialblend {

// Owned, value-only source evidence. No borrowed Actor, Binding, geometry, or
// generated receiver goal is retained. A returned value is historical after
// the copy; it is not an actuator lease or a physical tracking certificate.
struct SourceInteractionEvidence {
  static constexpr size_t kMaxAnchors = 128;
  static constexpr size_t kMaxRelations = 256;
  static constexpr size_t kMaxPathDepth = 64;
  static constexpr size_t kMaxPathNodes = 8192;
  static constexpr size_t kMaxBoundaryPoints = 65536;
  static constexpr size_t kMaxNameBytes = 128;
  static constexpr int64_t kMaximumAgeNs = 1000000000;

  struct Anchor {
    std::string name;
    bool observed = false;
    bool locomotion_root = false;
    int position_dimension = 0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    double yaw = 0.0;
  };
  struct HierarchyNode {
    element_id_t element = kInvalidElementId;
    element_id_t parent = kInvalidElementId;
    shared_element_id_t shared = kInvalidSharedElementId;
    std::string tag;
    GeometryRole geometry_role = GeometryRole::kUnspecified;
    Eigen::Matrix4d world_transform = Eigen::Matrix4d::Identity();
  };
  struct Relation {
    size_t binding_ordinal = 0;
    size_t anchor_ordinal = 0;
    RelationKernelType kernel = RelationKernelType::kAvatarFacing;
    element_id_t explicit_scope = kInvalidElementId;
    std::string tag_filter;
    double proximity = 0.0;
    double normalized_weight = 0.0;
    RelationPhase phase = RelationPhase::kDormant;
    // Generic lifecycle keys use a public shared-id fallback. Ambiguous keys
    // must not be interpreted as a selected semantic contact.
    bool phase_identity_unambiguous = false;
    shared_element_id_t phase_referent_id = kInvalidSharedElementId;
    // Ordered from hierarchy root through the exact local referent.
    std::vector<HierarchyNode> path;
    bool has_projection = false;
    bool bounded_planar_support = false;
    Eigen::Vector3d projected_point = Eigen::Vector3d::Zero();
    double anchor_support_distance = 0.0;
    Eigen::Matrix4d support_to_world = Eigen::Matrix4d::Identity();
    std::vector<Eigen::Vector2d> support_boundary;
    std::string sub_region;
    double support_angle = 0.0;
    double support_radial_ratio = 0.0;
  };

  space_id_t space = kInvalidSpaceId;
  spatialblend::entity_id_t actor = spatialblend::kInvalidEntityId;
  element_id_t actor_element = kInvalidElementId;
  shared_element_id_t actor_shared = kInvalidSharedElementId;
  uint64_t epoch = 0;
  int64_t sequence = -1;
  double source_time_seconds = 0.0;
  // Original admission-attempt receipt in ReceiverInputNowNs's process-local
  // clock. Source replay time is not a freshness clock; upstream age is
  // unknown.
  int64_t receipt_monotonic_ns = 0;
  uint64_t tree_revision = 0;
  uint64_t support_revision = 0;
  std::vector<Anchor> anchors;
  std::vector<Relation> relations;
};

}  // namespace spatialblend
#endif  // SPATIALBLEND_SPACE_SOURCE_INTERACTION_EVIDENCE_H_
