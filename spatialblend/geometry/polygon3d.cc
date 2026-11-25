// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/geometry/polygon3d.h"

namespace spatialblend {

Polygon3D::Polygon3D(const std::vector<Eigen::Vector3d>& vertices)
    : vertices_(vertices) {
  ComputeDerivedData();
}

const std::vector<Eigen::Vector3d>& Polygon3D::GetVertices() const {
  return vertices_;
}

Eigen::Vector3d Polygon3D::GetNormal() const { return normal_; }
Eigen::Vector3d Polygon3D::GetCenter() const { return center_; }
Eigen::Matrix4d Polygon3D::GetLocalFrame() const { return local_frame_; }
const Eigen::Matrix4d& Polygon3D::GetLocalFrameInverse() const {
  return local_frame_inverse_;
}

Eigen::Vector2d Polygon3D::ProjectToLocal2D(
    const Eigen::Vector3d& world_point) const {
  Eigen::Vector4d local =
      local_frame_inverse_ *
      Eigen::Vector4d(world_point.x(), world_point.y(), world_point.z(), 1.0);
  return Eigen::Vector2d(local.x(), local.y());
}

std::shared_ptr<spatialblend::Polygon2D> Polygon3D::Get2DBoundary() const {
  std::call_once(boundary_2d_flag_, [this]() {
    std::vector<Eigen::Vector2d> points_2d;
    points_2d.reserve(vertices_.size());
    for (const auto& v : vertices_) {
      points_2d.push_back(ProjectToLocal2D(v));
    }
    cached_boundary_2d_ = std::make_shared<spatialblend::Polygon2D>(points_2d);
  });
  return cached_boundary_2d_;
}

void Polygon3D::ComputeDerivedData() {
  center_ = Eigen::Vector3d::Zero();
  for (const auto& v : vertices_) {
    center_ += v;
  }
  if (!vertices_.empty()) {
    center_ /= static_cast<double>(vertices_.size());
  }

  if (vertices_.size() >= 3) {
    Eigen::Vector3d e1 = vertices_[1] - vertices_[0];
    Eigen::Vector3d e2 = vertices_[2] - vertices_[0];
    normal_ = e1.cross(e2).normalized();
  } else {
    normal_ = Eigen::Vector3d::UnitZ();
  }

  Eigen::Vector3d x_axis;
  if (vertices_.size() >= 2) {
    Eigen::Vector3d edge = vertices_[1] - vertices_[0];
    x_axis = (edge - edge.dot(normal_) * normal_).normalized();
  } else {
    Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
    if (std::abs(normal_.dot(up)) > 0.9) {
      up = Eigen::Vector3d::UnitX();
    }
    x_axis = normal_.cross(up).normalized();
  }
  Eigen::Vector3d y_axis = normal_.cross(x_axis).normalized();

  local_frame_ = Eigen::Matrix4d::Identity();
  local_frame_.block<3, 1>(0, 0) = x_axis;
  local_frame_.block<3, 1>(0, 1) = y_axis;
  local_frame_.block<3, 1>(0, 2) = normal_;
  local_frame_.block<3, 1>(0, 3) = center_;

  local_frame_inverse_ = local_frame_.inverse();
}

}  // namespace spatialblend
