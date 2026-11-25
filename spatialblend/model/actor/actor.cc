// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/actor/actor.h"

#include "spatialblend/base/math/util.h"

namespace spatialblend {

Actor::Actor(entity_id_t id) : id_(id) {}

double Actor::GetRotation() const {
  return math::ToDegrees(math::RotationMatrixToEuler(transform_.block<3, 3>(0, 0)).z());
}

void Actor::AddAnchor(const SubjectAnchor& anchor) {
  anchor_index_[anchor.name] = anchors_.size();
  anchors_.push_back(anchor);
}

void Actor::RemoveAnchor(const std::string& name) {
  auto it = anchor_index_.find(name);
  if (it == anchor_index_.end()) return;

  size_t idx = it->second;
  anchors_.erase(anchors_.begin() + idx);
  anchor_index_.erase(it);

  anchor_index_.clear();
  for (size_t i = 0; i < anchors_.size(); ++i) {
    anchor_index_[anchors_[i].name] = i;
  }
}

const SubjectAnchor* Actor::GetAnchor(const std::string& name) const {
  auto it = anchor_index_.find(name);
  if (it == anchor_index_.end()) return nullptr;
  return &anchors_[it->second];
}

SubjectAnchor* Actor::GetAnchor(const std::string& name) {
  auto it = anchor_index_.find(name);
  if (it == anchor_index_.end()) return nullptr;
  return &anchors_[it->second];
}

const std::vector<SubjectAnchor>& Actor::GetAnchors() const {
  return anchors_;
}

void Actor::SetElementId(element_id_t element_id) {
  element_id_ = element_id;
}

element_id_t Actor::GetElementId() const { return element_id_; }

void Actor::SetAnchorWorldPose(const std::string& anchor_name,
                               const Eigen::Vector3d& position, double yaw,
                               int32_t orientation_type, double pitch, double roll) {
  AnchorWorldPoseOverride& o = anchor_world_pose_overrides_[anchor_name];
  o.position = position;
  o.yaw = yaw;
  o.orientation_type = orientation_type;
  o.pitch = pitch;
  o.roll = roll;
}

void Actor::ClearAnchorWorldPose(const std::string& anchor_name) {
  anchor_world_pose_overrides_.erase(anchor_name);
}

const AnchorWorldPoseOverride* Actor::GetAnchorWorldPoseOverride(
    const std::string& anchor_name) const {
  auto it = anchor_world_pose_overrides_.find(anchor_name);
  return it != anchor_world_pose_overrides_.end() ? &it->second : nullptr;
}

void Actor::SetAnchorActive(const std::string& anchor_name, bool active) {
  if (active) {
    inactive_anchors_.erase(anchor_name);
  } else {
    inactive_anchors_.insert(anchor_name);
  }
}

bool Actor::IsAnchorActive(const std::string& anchor_name) const {
  return inactive_anchors_.count(anchor_name) == 0;
}

}  // namespace spatialblend
