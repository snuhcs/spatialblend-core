// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SIMULATOR_ACTOR_HANDLE_H_
#define SPATIALBLEND_SIMULATOR_ACTOR_HANDLE_H_

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "spatialblend/base/common/common.h"
#include "spatialblend/model/actor/actor.h"
#include "spatialblend/model/actor/subject_anchor.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/space/builder/element_handle.h"
#include "spatialblend/space/space.h"

namespace spatialblend {

// Callback for registering dirty actor updates with a simulator.
using DirtyActorCallback = std::function<void(
    Space*, spatialblend::entity_id_t, const Eigen::Vector2d&, double)>;

class ActorHandle {
 public:
  ActorHandle(Space* space, spatialblend::entity_id_t id,
              DirtyActorCallback on_dirty = nullptr)
      : space_(space), id_(id), on_dirty_(std::move(on_dirty)) {}

  ActorHandle& WithAnchor(const std::string& name, double x, double y,
                           double z) {
    space_->GetActor(id_)->AddAnchor(
        SubjectAnchor(name, Eigen::Vector3d(x, y, z)));
    return *this;
  }

  ActorHandle& InScope(const ElementHandle& elem) {
    space_->GetActor(id_)->SetElementId(elem.id());
    return *this;
  }

  void Bind(const std::string& anchor, RelationKernelType kernel_type,
            const std::string& tag_filter = "") {
    space_->Bind(id_, anchor, kernel_type, tag_filter);
  }

  void MoveTo(const Eigen::Vector2d& pos, double yaw) {
    pending_position_ = pos;
    pending_yaw_ = yaw;
    dirty_ = true;
    if (on_dirty_) {
      on_dirty_(space_, id_, pos, yaw);
    }
  }

  void MoveTo(const Eigen::Vector2d& pos) { MoveTo(pos, 0.0); }

  // The pose as the space shows it: a source actor's own pose, or the
  // driver's latest published sample for a receiver avatar.
  Eigen::Vector2d GetPosition() const {
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    double yaw = 0.0;
    space_->CopyActorPose(id_, &position, &yaw);
    return position;
  }

  double GetYaw() const {
    Eigen::Vector2d position;
    double yaw = 0.0;
    space_->CopyActorPose(id_, &position, &yaw);
    return yaw;
  }

  const RealizationResult* GetTargets() const {
    return space_->GetRealizationTargets(id_);
  }

  Space* GetSpace() const { return space_; }
  spatialblend::entity_id_t id() const { return id_; }

  bool IsDirty() const { return dirty_; }

  const Eigen::Vector2d& GetPendingPosition() const {
    return pending_position_;
  }

  double GetPendingYaw() const { return pending_yaw_; }

  void ClearDirty() { dirty_ = false; }

 private:
  Space* space_;
  spatialblend::entity_id_t id_;
  DirtyActorCallback on_dirty_;
  Eigen::Vector2d pending_position_ = Eigen::Vector2d::Zero();
  double pending_yaw_ = 0.0;
  bool dirty_ = false;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SIMULATOR_ACTOR_HANDLE_H_
