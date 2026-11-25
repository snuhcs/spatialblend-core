// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/geometry/context/surface_geometry_context.h"

namespace spatialblend {

SurfaceGeometryContext::SurfaceGeometryContext(
    std::shared_ptr<Polygon3D> surface)
    : surface_(std::move(surface)) {
  if (surface_) {
    boundary_2d_ = surface_->Get2DBoundary();
    // Build world-frame 2D footprint (XY projection of 3D vertices)
    const auto& verts = surface_->GetVertices();
    std::vector<Eigen::Vector2d> world_pts;
    world_pts.reserve(verts.size());
    for (const auto& v : verts) {
      world_pts.push_back(v.head<2>());
    }
    world_boundary_2d_ = std::make_shared<Polygon2D>(world_pts);
  }
}

std::shared_ptr<SurfaceGeometryContext> SurfaceGeometryContext::Create(
    std::shared_ptr<Polygon3D> surface) {
  return std::shared_ptr<SurfaceGeometryContext>(
      new SurfaceGeometryContext(std::move(surface)));
}

bool SurfaceGeometryContext::Contains(const Eigen::Vector2d& point) const {
  // World-frame 2D containment (XY footprint of the 3D surface).
  return world_boundary_2d_ && world_boundary_2d_->Contains(point);
}

Eigen::Vector2d SurfaceGeometryContext::GetNearestPoint(
    const Eigen::Vector2d& from) const {
  if (!boundary_2d_ || !surface_) return from;
  // Transform world 2D point to surface's local 2D frame
  Eigen::Vector4d world_pt(from.x(), from.y(), 0.0, 1.0);
  Eigen::Vector4d local_pt = surface_->GetLocalFrameInverse() * world_pt;
  Eigen::Vector2d local_nearest = boundary_2d_->GetNearestPoint(
      Eigen::Vector2d(local_pt.x(), local_pt.y()));
  // Transform back to world frame
  Eigen::Vector4d local_result(local_nearest.x(), local_nearest.y(), 0.0, 1.0);
  Eigen::Vector4d world_result = surface_->GetLocalFrame() * local_result;
  return world_result.head<2>();
}

double SurfaceGeometryContext::GetDistance(
    const Eigen::Vector2d& from) const {
  if (!boundary_2d_ || !surface_) return 0.0;
  // Transform world 2D point to surface's local 2D frame
  Eigen::Vector4d world_pt(from.x(), from.y(), 0.0, 1.0);
  Eigen::Vector4d local_pt = surface_->GetLocalFrameInverse() * world_pt;
  return boundary_2d_->GetDistance(
      Eigen::Vector2d(local_pt.x(), local_pt.y()));
}

Eigen::Vector3d SurfaceGeometryContext::ProjectOnto(
    const Eigen::Vector3d& point) const {
  if (!surface_) return point;
  Eigen::Matrix4d frame = surface_->GetLocalFrame();
  const Eigen::Matrix4d& inv = surface_->GetLocalFrameInverse();
  Eigen::Vector4d local = inv * Eigen::Vector4d(point.x(), point.y(),
                                                 point.z(), 1.0);
  local.z() = 0.0;
  Eigen::Vector4d world = frame * local;
  return world.head<3>();
}

Eigen::Vector3d SurfaceGeometryContext::GetNormal() const {
  if (!surface_) return Eigen::Vector3d(0, 0, 1);
  return surface_->GetNormal();
}

Eigen::Matrix4d SurfaceGeometryContext::GetLocalFrame() const {
  return surface_ ? surface_->GetLocalFrame() : Eigen::Matrix4d::Identity();
}

const Polygon3D& SurfaceGeometryContext::GetSurface() const {
  return *surface_;
}

std::shared_ptr<Polygon2D> SurfaceGeometryContext::GetBoundary2D() const {
  return boundary_2d_;
}

double SurfaceGeometryContext::GetDistance3D(
    const Eigen::Vector3d& from) const {
  Eigen::Vector3d projected = ProjectOnto(from);
  return (from - projected).norm();
}

}  // namespace spatialblend
