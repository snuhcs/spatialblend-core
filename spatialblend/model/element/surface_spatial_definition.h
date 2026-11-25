// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_ELEMENT_SURFACE_SPATIAL_DEFINITION_H_
#define SPATIALBLEND_ELEMENT_SURFACE_SPATIAL_DEFINITION_H_

#include <memory>

#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/model/element/spatial_definition.h"

namespace spatialblend {

// SpatialDefinition for planar 3D surfaces (tabletops, board surfaces, seats).
// Wraps a Polygon3D and provides projection, normal, and distance queries.
// TODO: Rename file to surface_spatial_definition.h in a follow-up.
class SurfaceSpatialDefinition : public SpatialDefinition {
 public:
  explicit SurfaceSpatialDefinition(std::shared_ptr<Polygon3D> surface);

  // Core interface
  bool Contains(const Eigen::Vector2d& point) const override;
  bool IsSurfaceDefinition() const override { return true; }
  std::shared_ptr<spatialblend::Polygon2D> GetBoundary() const override;
  const Eigen::Matrix4d& GetLocalTransform() const override;
  Bound2D GetBounds() const override;

  // Geometric query overrides
  bool Contains3D(const Eigen::Vector3d& point) const override;
  Eigen::Vector3d GetClosestPoint(const Eigen::Vector3d& from) const override;
  double GetDistance(const Eigen::Vector3d& from) const override;
  Eigen::Vector3d ProjectOnto(const Eigen::Vector3d& point) const override;
  Eigen::Vector3d GetNormal(
      const Eigen::Vector3d& at_point = Eigen::Vector3d::Zero()) const override;
  Eigen::Matrix4d GetTransform() const override;

  // Surface-specific
  std::shared_ptr<Polygon3D> GetSurface() const;
  Eigen::Matrix4d GetSurfaceLocalFrame() const;

 private:
  std::shared_ptr<Polygon3D> surface_;
  std::shared_ptr<spatialblend::Polygon2D> boundary_2d_;        // Local-frame 2D boundary
  std::shared_ptr<spatialblend::Polygon2D> world_boundary_2d_;  // World-frame XY footprint
  Eigen::Matrix4d local_transform_ = Eigen::Matrix4d::Identity();

  mutable Bound2D cached_bounds_;
  mutable bool bounds_computed_ = false;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_ELEMENT_SURFACE_SPATIAL_DEFINITION_H_
