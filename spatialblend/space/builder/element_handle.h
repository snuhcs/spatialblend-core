// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SIMULATOR_ELEMENT_HANDLE_H_
#define SPATIALBLEND_SIMULATOR_ELEMENT_HANDLE_H_

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/space/space.h"

namespace spatialblend {

class ElementHandle {
 public:
  ElementHandle(Space* space, element_id_t id,
                std::shared_ptr<int64_t> next_id)
      : space_(space), id_(id), next_id_(std::move(next_id)) {}

  ElementHandle& WithPolygon(const std::vector<Eigen::Vector2d>& points) {
    auto polygon = std::make_shared<Polygon2D>(points);
    auto nav_def = std::make_shared<NavigableSpatialDefinition>(polygon);
    space_->GetElementTree()->GetElement(id_)->SetSpatialDefinition(nav_def);
    return *this;
  }

  ElementHandle& WithBoundary(const std::vector<Eigen::Vector2d>& points) {
    return WithPolygon(points);
  }

  ElementHandle& WithSharedId(int64_t id) {
    space_->GetElementTree()->GetElement(id_)->SetSharedId(id);
    return *this;
  }

  ElementHandle& At(double x, double y) {
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform(0, 3) = x;
    transform(1, 3) = y;
    space_->GetElementTree()->GetElement(id_)->SetLocalTransform(transform);
    return *this;
  }

  ElementHandle& Tagged(const std::string& tag) {
    space_->GetElementTree()->GetElement(id_)->SetTag(tag);
    return *this;
  }

  ElementHandle AddChild(const std::string& name) {
    int64_t new_id = ++(*next_id_);
    (void)space_->GetElementTree()->AddElement(
        static_cast<element_id_t>(new_id), id_);
    space_->GetElementTree()->GetElement(static_cast<element_id_t>(new_id))
        ->SetTag(name);
    return ElementHandle(space_, static_cast<element_id_t>(new_id), next_id_);
  }

  Eigen::Vector2d Position() const {
    Eigen::Matrix4d world = space_->GetElementTree()->GetWorldTransform(id_);
    return Eigen::Vector2d(world(0, 3), world(1, 3));
  }

  element_id_t id() const { return id_; }

 private:
  Space* space_;
  element_id_t id_;
  std::shared_ptr<int64_t> next_id_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SIMULATOR_ELEMENT_HANDLE_H_
