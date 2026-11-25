// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_ELEMENT_POLYGON3D_H_
#define SPATIALBLEND_ELEMENT_POLYGON3D_H_

#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <vector>

#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {

class Polygon3D {
 public:
  explicit Polygon3D(const std::vector<Eigen::Vector3d>& vertices);

  const std::vector<Eigen::Vector3d>& GetVertices() const;
  Eigen::Vector3d GetNormal() const;
  Eigen::Vector3d GetCenter() const;
  Eigen::Matrix4d GetLocalFrame() const;
  const Eigen::Matrix4d& GetLocalFrameInverse() const;
  Eigen::Vector2d ProjectToLocal2D(const Eigen::Vector3d& world_point) const;
  std::shared_ptr<spatialblend::Polygon2D> Get2DBoundary() const;

 private:
  std::vector<Eigen::Vector3d> vertices_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d center_;
  Eigen::Matrix4d local_frame_;
  Eigen::Matrix4d local_frame_inverse_;

  void ComputeDerivedData();

  // Cached 2D boundary projection (built once on first access)
  mutable std::shared_ptr<Polygon2D> cached_boundary_2d_;
  mutable std::once_flag boundary_2d_flag_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_ELEMENT_POLYGON3D_H_
