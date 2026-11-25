// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_ELEMENT_ELEMENT_H_
#define SPATIALBLEND_ELEMENT_ELEMENT_H_

#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "spatialblend/base/common/common.h"
#include "spatialblend/base/common/types.h"
#include <optional>

#include "spatialblend/model/element/affordance_pose.h"
#include "spatialblend/model/element/geometry_role.h"
#include "spatialblend/model/element/spatial_definition.h"

namespace spatialblend {

class Element {
 public:
  explicit Element(element_id_t id);

  element_id_t GetId() const;

  void SetSharedId(shared_element_id_t shared_id);
  shared_element_id_t GetSharedId() const;

  void SetParentId(element_id_t parent_id);
  element_id_t GetParentId() const;
  void AddChildId(element_id_t child_id);
  void RemoveChildId(element_id_t child_id);
  const std::vector<element_id_t>& GetChildIds() const;
  bool IsRoot() const;
  bool IsLeaf() const;

  void SetLocalTransform(const Eigen::Matrix4d& transform);
  const Eigen::Matrix4d& GetLocalTransform() const;

  // Three definitions (a parent expands to contain its children;
  // bridges are only for walkable connectivity). Each consumer reads exactly one:
  //  - DECLARED (GetDeclaredSpatialDefinition): what the app registered. Support,
  //    contact, frame, chart, height and interior consumers: the extension never maps
  //    anything.
  //  - SCOPE (GetScopeDefinition, read through relation_kernel.h FootprintHolds and
  //    ScopeFootprintRegion): the declared region united with its region children's
  //    scope regions, WITHOUT the bridges and corridors across a gap. The scope descent,
  //    the containment tree and every scope/leaf query.
  //  - WALKABLE (GetSpatialDefinition, the effective definition): the scope region plus
  //    the bridges and corridors that keep it one connected walkable region
  //    (element/scope_extension.h). The navigable free space and path planning.
  // Each is the declared one unless region children that stick out of this region
  // extend it (containment_layout.h); the Space installs the extended definitions and
  // keeps the declared one. SetSpatialDefinition sets all three (no extension).
  //
  // Thread safety: the Space swaps the walkable and scope definitions from its background
  // worker under the space lock. A reader of an escaped tree WITHOUT that lock
  // (ElementTree* from Space::GetElementTree: the Python binding, the simulator) MUST
  // use the *Shared() accessors: they load the pointer atomically and own the
  // definition while it is used. The raw pointers are for readers holding the space
  // lock (the replaced definition is kept alive only until the next swap, a grace for a
  // raw pointer read just before one, not a guarantee).
  void SetSpatialDefinition(std::shared_ptr<SpatialDefinition> def);
  SpatialDefinition* GetSpatialDefinition() const;
  std::shared_ptr<SpatialDefinition> GetSpatialDefinitionShared() const {
    return std::atomic_load(&spatial_definition_);
  }
  // The definition the app registered, which the containment layout checks and the
  // scope extension starts from.
  SpatialDefinition* GetDeclaredSpatialDefinition() const {
    return declared_definition_.get();
  }
  std::shared_ptr<SpatialDefinition> GetDeclaredSpatialDefinitionShared() const {
    return declared_definition_;
  }
  // Installs the walkable definition (nullptr: back to the declared one). The
  // declared definition is unchanged.
  void SetEffectiveSpatialDefinition(std::shared_ptr<SpatialDefinition> def) {
    retired_definition_ = std::atomic_exchange(
        &spatial_definition_, def ? std::move(def) : declared_definition_);
  }
  // The scope region (see above); nullptr: back to the declared one.
  void SetScopeDefinition(std::shared_ptr<SpatialDefinition> def) {
    retired_scope_definition_ = std::atomic_exchange(
        &scope_definition_, def ? std::move(def) : declared_definition_);
  }
  SpatialDefinition* GetScopeDefinition() const {
    return std::atomic_load(&scope_definition_).get();
  }
  std::shared_ptr<SpatialDefinition> GetScopeDefinitionShared() const {
    return std::atomic_load(&scope_definition_);
  }
  bool HasScopeExtension() const {
    return std::atomic_load(&spatial_definition_) != declared_definition_ ||
           std::atomic_load(&scope_definition_) != declared_definition_;
  }
  GeometryRole GetGeometryRole() const;
  bool HasGeometryRole(GeometryRole role) const;

  void SetGroundingEntityId(spatialblend::entity_id_t entity_id);
  spatialblend::entity_id_t GetGroundingEntityId() const;

  void SetTag(const std::string& tag);
  const std::string& GetTag() const;

  // Containment transfer (c/spatialblend_containment_transfer.h): while an
  // actor's root is contained in this element, the root's pose is sent in this
  // element's frame and mapped through the receiver's counterpart. Opt-in;
  // default false leaves every relation path untouched.
  void SetContainmentTransfer(bool enabled) { containment_transfer_ = enabled; }
  bool HasContainmentTransfer() const { return containment_transfer_; }

  // An affordance pose of the PARENT object (element/affordance_pose.h): set on an interaction-role child, whose tag
  // names the pose. nullptr when this element carries none.
  void SetAffordancePose(const AffordancePoseTemplate& pose) { affordance_pose_ = pose; }
  void ClearAffordancePose() { affordance_pose_.reset(); }
  const AffordancePoseTemplate* GetAffordancePose() const {
    return affordance_pose_ ? &*affordance_pose_ : nullptr;
  }

  // Named anchor points — local transforms relative to this element.
  // Used by grounding rules to compute relation-specific support points.
  void AddAnchor(const std::string& name, const Eigen::Matrix4d& local_transform);
  const Eigen::Matrix4d* GetAnchor(const std::string& name) const;
  const std::unordered_map<std::string, Eigen::Matrix4d>& GetAnchors() const;

 private:
  element_id_t id_;
  shared_element_id_t shared_id_ = kInvalidSharedElementId;

  element_id_t parent_id_ = kInvalidElementId;
  std::vector<element_id_t> child_ids_;

  Eigen::Matrix4d local_transform_ = Eigen::Matrix4d::Identity();
  std::shared_ptr<SpatialDefinition> spatial_definition_;   // effective
  std::shared_ptr<SpatialDefinition> declared_definition_;  // as registered
  std::shared_ptr<SpatialDefinition> retired_definition_;  // the last replaced effective
  std::shared_ptr<SpatialDefinition> scope_definition_;     // scope (no bridges)
  std::shared_ptr<SpatialDefinition> retired_scope_definition_;

  spatialblend::entity_id_t grounding_entity_id_ = spatialblend::kInvalidEntityId;
  std::string tag_;
  bool containment_transfer_ = false;
  std::optional<AffordancePoseTemplate> affordance_pose_;
  std::unordered_map<std::string, Eigen::Matrix4d> anchors_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_ELEMENT_ELEMENT_H_
