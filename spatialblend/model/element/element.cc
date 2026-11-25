// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/element.h"

namespace spatialblend {

Element::Element(element_id_t id) : id_(id) {}

element_id_t Element::GetId() const { return id_; }

void Element::SetSharedId(shared_element_id_t shared_id) {
  shared_id_ = shared_id;
}
shared_element_id_t Element::GetSharedId() const { return shared_id_; }

void Element::SetParentId(element_id_t parent_id) { parent_id_ = parent_id; }
element_id_t Element::GetParentId() const { return parent_id_; }

void Element::AddChildId(element_id_t child_id) {
  child_ids_.push_back(child_id);
}

void Element::RemoveChildId(element_id_t child_id) {
  child_ids_.erase(
      std::remove(child_ids_.begin(), child_ids_.end(), child_id),
      child_ids_.end());
}

const std::vector<element_id_t>& Element::GetChildIds() const {
  return child_ids_;
}

bool Element::IsRoot() const { return parent_id_ == kInvalidElementId; }
bool Element::IsLeaf() const { return child_ids_.empty(); }

void Element::SetLocalTransform(const Eigen::Matrix4d& transform) {
  local_transform_ = transform;
}

const Eigen::Matrix4d& Element::GetLocalTransform() const {
  return local_transform_;
}

void Element::SetSpatialDefinition(std::shared_ptr<SpatialDefinition> def) {
  declared_definition_ = def;
  retired_scope_definition_ = std::atomic_exchange(&scope_definition_, def);
  retired_definition_ = std::atomic_exchange(&spatial_definition_, std::move(def));
}

SpatialDefinition* Element::GetSpatialDefinition() const {
  return std::atomic_load(&spatial_definition_).get();
}

// The role is the declared definition's (an extension keeps its role).
GeometryRole Element::GetGeometryRole() const {
  return declared_definition_ ? declared_definition_->GetGeometryRole()
                              : GeometryRole::kUnspecified;
}

bool Element::HasGeometryRole(GeometryRole role) const {
  return declared_definition_ && declared_definition_->GetGeometryRole() == role;
}

void Element::SetGroundingEntityId(spatialblend::entity_id_t entity_id) {
  grounding_entity_id_ = entity_id;
}

spatialblend::entity_id_t Element::GetGroundingEntityId() const {
  return grounding_entity_id_;
}

void Element::SetTag(const std::string& tag) { tag_ = tag; }

const std::string& Element::GetTag() const { return tag_; }

void Element::AddAnchor(const std::string& name,
                        const Eigen::Matrix4d& local_transform) {
  anchors_[name] = local_transform;
}

const Eigen::Matrix4d* Element::GetAnchor(const std::string& name) const {
  auto it = anchors_.find(name);
  if (it == anchors_.end()) return nullptr;
  return &it->second;
}

const std::unordered_map<std::string, Eigen::Matrix4d>&
Element::GetAnchors() const {
  return anchors_;
}

}  // namespace spatialblend
