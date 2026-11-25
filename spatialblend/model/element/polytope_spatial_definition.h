// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_ELEMENT_POLYTOPE_SPATIAL_DEFINITION_H_
#define SPATIALBLEND_ELEMENT_POLYTOPE_SPATIAL_DEFINITION_H_

#include <Eigen/Dense>

#include <memory>
#include <vector>

#include "spatialblend/model/element/spatial_definition.h"

namespace spatialblend {

// Convex 3D volume spatial definition. The box constructor is the first
// concrete path used by clients; the vertices/faces constructor keeps the core
// representation geometry-driven rather than type-driven.
//
// The geometry role defaults to kUnspecified: an OBSTACLE
// role constrains root feasibility, geodesic distance and mediator courses,
// so a box becomes an obstacle only when a caller declares it (the C ABI's
// sb_set_element_box / sb_set_element_polytope always pass a role).
class PolytopeSpatialDefinition : public SpatialDefinition {
 public:
  PolytopeSpatialDefinition(
      const Eigen::Vector3d& min_corner,
      const Eigen::Vector3d& max_corner,
      GeometryRole geometry_role = GeometryRole::kUnspecified);

  PolytopeSpatialDefinition(
      std::vector<Eigen::Vector3d> vertices,
      std::vector<std::vector<int>> face_indices,
      GeometryRole geometry_role = GeometryRole::kUnspecified);

  bool Contains(const Eigen::Vector2d& point) const override;
  bool Contains3D(const Eigen::Vector3d& point) const override;
  std::shared_ptr<Polygon2D> GetBoundary() const override;
  const Eigen::Matrix4d& GetLocalTransform() const override;
  Bound2D GetBounds() const override;

  Eigen::Vector3d GetClosestPoint(const Eigen::Vector3d& from) const override;
  double GetDistance(const Eigen::Vector3d& from) const override;
  Eigen::Vector3d ProjectOnto(const Eigen::Vector3d& point) const override;
  Eigen::Vector3d GetNormal(
      const Eigen::Vector3d& at_point = Eigen::Vector3d::Zero()) const override;
  Eigen::Matrix4d GetTransform() const override;

  // See SpatialDefinition::RequiresElementWorldTransform. This definition's
  // vertices are stored verbatim in the owning element's local frame (e.g.
  // LOCAL half-extents around the element origin), and `local_transform_`
  // below is a permanently-Identity field with no setter (this class has no
  // reference to the owning Element/ElementTree to keep it in sync), so
  // `GetTransform()` can never reflect the element's real world placement.
  bool RequiresElementWorldTransform() const override { return true; }

  const std::vector<Eigen::Vector3d>& GetVertices() const;
  const std::vector<std::vector<int>>& GetFaceIndices() const;
  Eigen::Vector3d GetMinCorner() const;
  Eigen::Vector3d GetMaxCorner() const;
  Eigen::Vector3d GetCenter() const;

 private:
  struct Face {
    std::vector<int> indices;
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    double offset = 0.0;
  };

  static std::vector<Eigen::Vector3d> BuildBoxVertices(
      const Eigen::Vector3d& min_corner,
      const Eigen::Vector3d& max_corner);
  static std::vector<std::vector<int>> BuildBoxFaces();

  void Initialize();
  void ComputeBounds();
  void ComputeFaces();
  void ComputeBoundary();

  std::vector<Eigen::Vector3d> vertices_;
  std::vector<std::vector<int>> face_indices_;
  std::vector<Face> faces_;
  std::shared_ptr<Polygon2D> boundary_2d_;
  Eigen::Vector3d min_corner_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d max_corner_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d center_ = Eigen::Vector3d::Zero();
  Bound2D bounds_;
  Eigen::Matrix4d local_transform_ = Eigen::Matrix4d::Identity();
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_ELEMENT_POLYTOPE_SPATIAL_DEFINITION_H_
