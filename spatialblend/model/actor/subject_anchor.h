// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_ACTOR_SUBJECT_ANCHOR_H_
#define SPATIALBLEND_ACTOR_SUBJECT_ANCHOR_H_

#include <Eigen/Dense>
#include <string>
#include <vector>
#include "spatialblend/base/common/state_type.h"

namespace spatialblend {

struct SubjectAnchor {
  SubjectAnchor() = default;
  SubjectAnchor(const std::string& name,
                const Eigen::Vector3d& local_offset = Eigen::Vector3d::Zero(),
                std::vector<StateType> dof = {StateType::kPosition2D,
                                               StateType::kRotationYaw},
                std::string parent = "")
      : name(name),
        local_offset(local_offset),
        dof(std::move(dof)),
        parent(std::move(parent)) {}

  Eigen::Vector3d GetWorldPosition(
      const Eigen::Matrix4d& actor_transform) const {
    Eigen::Vector4d local(local_offset.x(), local_offset.y(),
                          local_offset.z(), 1.0);
    Eigen::Vector4d world = actor_transform * local;
    return world.head<3>();
  }

  Eigen::Vector2d GetWorldPosition2D(
      const Eigen::Matrix4d& actor_transform) const {
    return GetWorldPosition(actor_transform).head<2>();
  }

  bool HasPosition3D() const {
    for (auto s : dof) {
      if (s == StateType::kPosition3D) return true;
    }
    return false;
  }

  bool HasRotation3D() const {
    for (auto s : dof) {
      if (s == StateType::kRotation3D) return true;
    }
    return false;
  }

  std::string name;
  Eigen::Vector3d local_offset = Eigen::Vector3d::Zero();
  std::vector<StateType> dof = {StateType::kPosition2D, StateType::kRotationYaw};
  // Name of the parent anchor in the same actor's anchor set.
  // Empty = attached directly to the actor root transform.
  std::string parent;
};

// Canonical body anchor set. Local offsets are
// nominal defaults for a standing adult; runtime pose updates override them.
inline std::vector<SubjectAnchor> StandardAnchors() {
  using ST = StateType;
  return {
      SubjectAnchor("root", {0, 0, 0}, {ST::kPosition2D, ST::kRotationYaw}),
      SubjectAnchor("hips", {0, 0, 0.95}, {ST::kPosition3D, ST::kRotationYaw},
                    "root"),
      SubjectAnchor("chest", {0, 0, 1.35}, {ST::kPosition3D, ST::kRotationYaw},
                    "hips"),
      SubjectAnchor("head", {0, 0, 1.65}, {ST::kPosition3D, ST::kRotation3D},
                    "chest"),
      SubjectAnchor("hand_l", {-0.2, 0.3, 1.1},
                    {ST::kPosition3D, ST::kRotation3D}, "chest"),
      SubjectAnchor("hand_r", {0.2, 0.3, 1.1},
                    {ST::kPosition3D, ST::kRotation3D}, "chest"),
  };
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_ACTOR_SUBJECT_ANCHOR_H_
