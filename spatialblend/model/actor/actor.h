// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_ACTOR_ACTOR_H_
#define SPATIALBLEND_ACTOR_ACTOR_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "spatialblend/base/common/common.h"
#include "spatialblend/base/common/types.h"
#include "spatialblend/model/actor/subject_anchor.h"

namespace spatialblend {

// A per-anchor world-pose override ("anchor liveness"):
// SubjectAnchor::local_offset is a fixed offset
// from the actor root, fine for the simulator but wrong for a real tracked
// hand. When set for an anchor, grounding uses this WORLD pose directly
// instead of deriving the anchor's position from actor_transform *
// local_offset.
struct AnchorWorldPoseOverride {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  double yaw = 0.0;
  // The anchor's orientation beyond yaw (sb_orientation_type_t; pitch and roll in radians), when the source tracks it
  // (a bone-bound anchor): a pass-through of this anchor's proposal on the receiver, as the root's is.
  int32_t orientation_type = 0;
  double pitch = 0.0, roll = 0.0;
};

// A tracked participant: an id, the root pose and its named anchors.
class Actor {
 public:
  explicit Actor(entity_id_t id = kInvalidEntityId);

  entity_id_t GetId() const { return id_; }

  // The root pose (world). GetWorldTransform is the same matrix (kept for the callers that read it by that name).
  void SetTransform(const Eigen::Matrix4d& transform) { transform_ = transform; }
  const Eigen::Matrix4d& GetTransform() const { return transform_; }
  const Eigen::Matrix4d& GetWorldTransform() const { return transform_; }
  // The root's horizontal position (x, y) and yaw in DEGREES, [0, 360) (Z-up).
  Eigen::Vector2d GetPosition() const { return transform_.block<2, 1>(0, 3); }
  double GetRotation() const;

  void AddAnchor(const SubjectAnchor& anchor);
  void RemoveAnchor(const std::string& name);
  const SubjectAnchor* GetAnchor(const std::string& name) const;
  SubjectAnchor* GetAnchor(const std::string& name);
  const std::vector<SubjectAnchor>& GetAnchors() const;

  void SetElementId(element_id_t element_id);
  element_id_t GetElementId() const;

  // Per-anchor world-pose override (see AnchorWorldPoseOverride above). The
  // "root" anchor is exempt from this in practice: Space::UpdateActorState
  // always drives root_pos from the live actor transform for scope
  // resolution/containment, regardless of any override stored here (see
  // Space::SetAnchorWorldPose doc comment).
  void SetAnchorWorldPose(const std::string& anchor_name,
                          const Eigen::Vector3d& position, double yaw,
                          int32_t orientation_type = 0, double pitch = 0.0, double roll = 0.0);
  void ClearAnchorWorldPose(const std::string& anchor_name);
  const AnchorWorldPoseOverride* GetAnchorWorldPoseOverride(
      const std::string& anchor_name) const;

  // Anchors are active by default for backwards compatibility. Inactive
  // tracked anchors are omitted from relation evaluation and wire snapshots.
  void SetAnchorActive(const std::string& anchor_name, bool active);
  bool IsAnchorActive(const std::string& anchor_name) const;

 private:
  entity_id_t id_;
  Eigen::Matrix4d transform_ = Eigen::Matrix4d::Identity();
  std::vector<SubjectAnchor> anchors_;
  std::unordered_map<std::string, size_t> anchor_index_;
  element_id_t element_id_ = kInvalidElementId;
  std::unordered_map<std::string, AnchorWorldPoseOverride>
      anchor_world_pose_overrides_;
  std::unordered_set<std::string> inactive_anchors_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_ACTOR_ACTOR_H_
