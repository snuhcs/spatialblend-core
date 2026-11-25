// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/surface_spatial_definition.h"

namespace spatialblend {

SurfaceSpatialDefinition::SurfaceSpatialDefinition(
    std::shared_ptr<Polygon3D> surface)
    : SpatialDefinition(GeometryRole::kSurface),
      surface_(std::move(surface)) {
  if (surface_) {
    boundary_2d_ = surface_->Get2DBoundary();
    // Build world-frame 2D footprint (XY projection of 3D vertices)
    // for scope-resolution containment checks.
    const auto& verts = surface_->GetVertices();
    std::vector<Eigen::Vector2d> world_pts;
    world_pts.reserve(verts.size());
    for (const auto& v : verts) {
      world_pts.push_back(v.head<2>());
    }
    world_boundary_2d_ = std::make_shared<Polygon2D>(world_pts);
  }
}

bool SurfaceSpatialDefinition::Contains(const Eigen::Vector2d& point) const {
  // World-frame 2D containment (used by scope resolution).
  return world_boundary_2d_ && world_boundary_2d_->Contains(point);
}

std::shared_ptr<spatialblend::Polygon2D>
SurfaceSpatialDefinition::GetBoundary() const {
  return boundary_2d_;
}

const Eigen::Matrix4d& SurfaceSpatialDefinition::GetLocalTransform() const {
  return local_transform_;
}

Bound2D SurfaceSpatialDefinition::GetBounds() const {
  if (bounds_computed_) return cached_bounds_;
  if (!surface_) return cached_bounds_;
  // Compute world-frame 2D bounds from the 3D vertices directly
  const auto& vertices = surface_->GetVertices();
  if (vertices.empty()) return cached_bounds_;

  cached_bounds_.min = vertices[0].head<2>();
  cached_bounds_.max = vertices[0].head<2>();
  for (const auto& v : vertices) {
    cached_bounds_.min = cached_bounds_.min.cwiseMin(v.head<2>());
    cached_bounds_.max = cached_bounds_.max.cwiseMax(v.head<2>());
  }
  bounds_computed_ = true;
  return cached_bounds_;
}

bool SurfaceSpatialDefinition::Contains3D(const Eigen::Vector3d& point) const {
  if (!boundary_2d_ || !surface_) return false;
  // Transform world 3D point to surface's local frame
  Eigen::Vector4d world_pt(point.x(), point.y(), point.z(), 1.0);
  Eigen::Vector4d local_pt = surface_->GetLocalFrameInverse() * world_pt;
  // Check 2D containment in local frame + distance to surface plane
  constexpr double kSurfaceTolerance = 0.5;  // 0.5m tolerance
  bool in_boundary = boundary_2d_->Contains(
      Eigen::Vector2d(local_pt.x(), local_pt.y()));
  bool near_surface = std::abs(local_pt.z()) < kSurfaceTolerance;
  return in_boundary && near_surface;
}

std::shared_ptr<Polygon3D> SurfaceSpatialDefinition::GetSurface() const {
  return surface_;
}

Eigen::Matrix4d SurfaceSpatialDefinition::GetSurfaceLocalFrame() const {
  return surface_ ? surface_->GetLocalFrame() : Eigen::Matrix4d::Identity();
}

Eigen::Vector3d SurfaceSpatialDefinition::GetClosestPoint(
    const Eigen::Vector3d& from) const {
  if (!surface_) return from;
  // Project onto the surface plane, then clamp to boundary
  return ProjectOnto(from);
}

double SurfaceSpatialDefinition::GetDistance(
    const Eigen::Vector3d& from) const {
  if (!surface_) return 0.0;
  Eigen::Vector3d projected = ProjectOnto(from);
  return (from - projected).norm();
}

Eigen::Vector3d SurfaceSpatialDefinition::ProjectOnto(
    const Eigen::Vector3d& point) const {
  if (!surface_) return point;
  // Project onto the surface plane in the surface's local frame, then clamp
  // to the bounded polygon footprint.
  Eigen::Matrix4d frame = surface_->GetLocalFrame();
  const Eigen::Matrix4d& inv = surface_->GetLocalFrameInverse();
  Eigen::Vector4d local = inv * Eigen::Vector4d(point.x(), point.y(),
                                                point.z(), 1.0);
  local.z() = 0.0;
  Eigen::Vector2d local_xy(local.x(), local.y());
  if (boundary_2d_ && !boundary_2d_->GetPoints().empty() &&
      !boundary_2d_->Contains(local_xy)) {
    local_xy = boundary_2d_->GetNearestPoint(local_xy);
  }
  local.x() = local_xy.x();
  local.y() = local_xy.y();
  Eigen::Vector4d world = frame * local;
  return world.head<3>();
}

Eigen::Vector3d SurfaceSpatialDefinition::GetNormal(
    const Eigen::Vector3d& /*at_point*/) const {
  if (!surface_) return Eigen::Vector3d(0, 0, 1);
  return surface_->GetNormal();
}

Eigen::Matrix4d SurfaceSpatialDefinition::GetTransform() const {
  return surface_ ? surface_->GetLocalFrame() : Eigen::Matrix4d::Identity();
}

}  // namespace spatialblend
