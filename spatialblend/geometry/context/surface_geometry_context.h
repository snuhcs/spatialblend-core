// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_GEOMETRY_SURFACE_GEOMETRY_CONTEXT_H_
#define SPATIALBLEND_GEOMETRY_SURFACE_GEOMETRY_CONTEXT_H_

#include <memory>

#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/geometry/context/geometry_context.h"
#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {

// Precomputed geometry for planar 3D surfaces (tabletops, seats, boards).
// Owns the 3D surface, its 2D boundary projection, and local frame.
class SurfaceGeometryContext : public GeometryContext {
 public:
  static std::shared_ptr<SurfaceGeometryContext> Create(
      std::shared_ptr<Polygon3D> surface);

  // GeometryContext interface
  bool Contains(const Eigen::Vector2d& point) const override;
  Eigen::Vector2d GetNearestPoint(const Eigen::Vector2d& from) const override;
  double GetDistance(const Eigen::Vector2d& from) const override;
  Eigen::Vector3d ProjectOnto(const Eigen::Vector3d& point) const override;

  // Surface-specific
  Eigen::Vector3d GetNormal() const;
  Eigen::Matrix4d GetLocalFrame() const;
  const Polygon3D& GetSurface() const;
  std::shared_ptr<Polygon2D> GetBoundary2D() const;

  // 3D distance: distance from a 3D point to the surface plane
  double GetDistance3D(const Eigen::Vector3d& from) const;

 private:
  explicit SurfaceGeometryContext(std::shared_ptr<Polygon3D> surface);

  std::shared_ptr<Polygon3D> surface_;
  std::shared_ptr<Polygon2D> boundary_2d_;        // Local-frame 2D boundary
  std::shared_ptr<Polygon2D> world_boundary_2d_;  // World-frame XY footprint
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_GEOMETRY_SURFACE_GEOMETRY_CONTEXT_H_
