// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/polytope_spatial_definition.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace spatialblend {

namespace {

constexpr double kEpsilon = 1e-9;

double Cross2D(const Eigen::Vector2d& origin,
               const Eigen::Vector2d& a,
               const Eigen::Vector2d& b) {
  Eigen::Vector2d oa = a - origin;
  Eigen::Vector2d ob = b - origin;
  return oa.x() * ob.y() - oa.y() * ob.x();
}

std::vector<Eigen::Vector2d> ConvexHull2D(
    std::vector<Eigen::Vector2d> points) {
  std::sort(points.begin(), points.end(),
            [](const Eigen::Vector2d& lhs, const Eigen::Vector2d& rhs) {
              if (lhs.x() == rhs.x()) return lhs.y() < rhs.y();
              return lhs.x() < rhs.x();
            });
  points.erase(
      std::unique(points.begin(), points.end(),
                  [](const Eigen::Vector2d& lhs,
                     const Eigen::Vector2d& rhs) {
                    return (lhs - rhs).squaredNorm() < kEpsilon * kEpsilon;
                  }),
      points.end());

  if (points.size() <= 2) return points;

  std::vector<Eigen::Vector2d> lower;
  for (const auto& p : points) {
    while (lower.size() >= 2 &&
           Cross2D(lower[lower.size() - 2], lower.back(), p) <= kEpsilon) {
      lower.pop_back();
    }
    lower.push_back(p);
  }

  std::vector<Eigen::Vector2d> upper;
  for (auto it = points.rbegin(); it != points.rend(); ++it) {
    while (upper.size() >= 2 &&
           Cross2D(upper[upper.size() - 2], upper.back(), *it) <= kEpsilon) {
      upper.pop_back();
    }
    upper.push_back(*it);
  }

  lower.pop_back();
  upper.pop_back();
  lower.insert(lower.end(), upper.begin(), upper.end());
  return lower;
}

Eigen::Vector3d ClosestPointOnTriangle(const Eigen::Vector3d& point,
                                       const Eigen::Vector3d& a,
                                       const Eigen::Vector3d& b,
                                       const Eigen::Vector3d& c) {
  const Eigen::Vector3d ab = b - a;
  const Eigen::Vector3d ac = c - a;
  const Eigen::Vector3d ap = point - a;
  const double d1 = ab.dot(ap);
  const double d2 = ac.dot(ap);
  if (d1 <= 0.0 && d2 <= 0.0) return a;

  const Eigen::Vector3d bp = point - b;
  const double d3 = ab.dot(bp);
  const double d4 = ac.dot(bp);
  if (d3 >= 0.0 && d4 <= d3) return b;

  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double v = d1 / (d1 - d3);
    return a + v * ab;
  }

  const Eigen::Vector3d cp = point - c;
  const double d5 = ab.dot(cp);
  const double d6 = ac.dot(cp);
  if (d6 >= 0.0 && d5 <= d6) return c;

  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double w = d2 / (d2 - d6);
    return a + w * ac;
  }

  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return b + w * (c - b);
  }

  const double denom = 1.0 / (va + vb + vc);
  const double v = vb * denom;
  const double w = vc * denom;
  return a + ab * v + ac * w;
}

Eigen::Vector3d NearestVertex(const std::vector<Eigen::Vector3d>& vertices,
                              const Eigen::Vector3d& point) {
  if (vertices.empty()) return point;
  double best_distance_sq = std::numeric_limits<double>::max();
  Eigen::Vector3d best = vertices[0];
  for (const auto& vertex : vertices) {
    double distance_sq = (point - vertex).squaredNorm();
    if (distance_sq < best_distance_sq) {
      best_distance_sq = distance_sq;
      best = vertex;
    }
  }
  return best;
}

}  // namespace

PolytopeSpatialDefinition::PolytopeSpatialDefinition(
    const Eigen::Vector3d& min_corner,
    const Eigen::Vector3d& max_corner,
    GeometryRole geometry_role)
    : SpatialDefinition(geometry_role),
      vertices_(BuildBoxVertices(min_corner, max_corner)),
      face_indices_(BuildBoxFaces()) {
  Initialize();
}

PolytopeSpatialDefinition::PolytopeSpatialDefinition(
    std::vector<Eigen::Vector3d> vertices,
    std::vector<std::vector<int>> face_indices,
    GeometryRole geometry_role)
    : SpatialDefinition(geometry_role),
      vertices_(std::move(vertices)),
      face_indices_(std::move(face_indices)) {
  Initialize();
}

bool PolytopeSpatialDefinition::Contains(const Eigen::Vector2d& point) const {
  return boundary_2d_ && boundary_2d_->Contains(point);
}

bool PolytopeSpatialDefinition::Contains3D(
    const Eigen::Vector3d& point) const {
  if (faces_.empty()) return false;
  for (const auto& face : faces_) {
    if (face.normal.dot(point) - face.offset > kEpsilon) {
      return false;
    }
  }
  return true;
}

std::shared_ptr<Polygon2D> PolytopeSpatialDefinition::GetBoundary() const {
  return boundary_2d_;
}

const Eigen::Matrix4d& PolytopeSpatialDefinition::GetLocalTransform() const {
  return local_transform_;
}

Bound2D PolytopeSpatialDefinition::GetBounds() const {
  return bounds_;
}

Eigen::Vector3d PolytopeSpatialDefinition::GetClosestPoint(
    const Eigen::Vector3d& from) const {
  if (vertices_.empty()) return from;
  if (faces_.empty()) return NearestVertex(vertices_, from);

  if (Contains3D(from)) {
    double best_distance = std::numeric_limits<double>::max();
    Eigen::Vector3d best = from;
    for (const auto& face : faces_) {
      const double signed_distance = face.normal.dot(from) - face.offset;
      const double distance_to_face = std::abs(signed_distance);
      if (distance_to_face < best_distance) {
        best_distance = distance_to_face;
        best = from - signed_distance * face.normal;
      }
    }
    return best;
  }

  double best_distance_sq = std::numeric_limits<double>::max();
  Eigen::Vector3d best = vertices_[0];
  for (const auto& face : faces_) {
    if (face.indices.size() < 3) continue;
    const Eigen::Vector3d& a = vertices_[face.indices[0]];
    for (size_t i = 1; i + 1 < face.indices.size(); ++i) {
      const Eigen::Vector3d& b = vertices_[face.indices[i]];
      const Eigen::Vector3d& c = vertices_[face.indices[i + 1]];
      Eigen::Vector3d candidate = ClosestPointOnTriangle(from, a, b, c);
      double distance_sq = (from - candidate).squaredNorm();
      if (distance_sq < best_distance_sq) {
        best_distance_sq = distance_sq;
        best = candidate;
      }
    }
  }
  if (best_distance_sq == std::numeric_limits<double>::max()) {
    return NearestVertex(vertices_, from);
  }
  return best;
}

double PolytopeSpatialDefinition::GetDistance(
    const Eigen::Vector3d& from) const {
  return (from - GetClosestPoint(from)).norm();
}

Eigen::Vector3d PolytopeSpatialDefinition::ProjectOnto(
    const Eigen::Vector3d& point) const {
  return GetClosestPoint(point);
}

Eigen::Vector3d PolytopeSpatialDefinition::GetNormal(
    const Eigen::Vector3d& at_point) const {
  if (faces_.empty()) return Eigen::Vector3d::UnitZ();

  double best_distance = std::numeric_limits<double>::max();
  Eigen::Vector3d normal = faces_[0].normal;
  for (const auto& face : faces_) {
    const double distance = std::abs(face.normal.dot(at_point) - face.offset);
    if (distance < best_distance) {
      best_distance = distance;
      normal = face.normal;
    }
  }
  return normal;
}

Eigen::Matrix4d PolytopeSpatialDefinition::GetTransform() const {
  return local_transform_;
}

const std::vector<Eigen::Vector3d>&
PolytopeSpatialDefinition::GetVertices() const {
  return vertices_;
}

const std::vector<std::vector<int>>&
PolytopeSpatialDefinition::GetFaceIndices() const {
  return face_indices_;
}

Eigen::Vector3d PolytopeSpatialDefinition::GetMinCorner() const {
  return min_corner_;
}

Eigen::Vector3d PolytopeSpatialDefinition::GetMaxCorner() const {
  return max_corner_;
}

Eigen::Vector3d PolytopeSpatialDefinition::GetCenter() const {
  return center_;
}

std::vector<Eigen::Vector3d> PolytopeSpatialDefinition::BuildBoxVertices(
    const Eigen::Vector3d& min_corner,
    const Eigen::Vector3d& max_corner) {
  Eigen::Vector3d lo = min_corner.cwiseMin(max_corner);
  Eigen::Vector3d hi = min_corner.cwiseMax(max_corner);
  return {
      {lo.x(), lo.y(), lo.z()},
      {hi.x(), lo.y(), lo.z()},
      {hi.x(), hi.y(), lo.z()},
      {lo.x(), hi.y(), lo.z()},
      {lo.x(), lo.y(), hi.z()},
      {hi.x(), lo.y(), hi.z()},
      {hi.x(), hi.y(), hi.z()},
      {lo.x(), hi.y(), hi.z()},
  };
}

std::vector<std::vector<int>> PolytopeSpatialDefinition::BuildBoxFaces() {
  return {
      {0, 1, 2, 3},
      {4, 7, 6, 5},
      {0, 4, 5, 1},
      {1, 5, 6, 2},
      {2, 6, 7, 3},
      {3, 7, 4, 0},
  };
}

void PolytopeSpatialDefinition::Initialize() {
  ComputeBounds();
  ComputeFaces();
  ComputeBoundary();
}

void PolytopeSpatialDefinition::ComputeBounds() {
  if (vertices_.empty()) {
    min_corner_ = Eigen::Vector3d::Zero();
    max_corner_ = Eigen::Vector3d::Zero();
    center_ = Eigen::Vector3d::Zero();
    bounds_ = {};
    return;
  }

  min_corner_ = vertices_[0];
  max_corner_ = vertices_[0];
  center_ = Eigen::Vector3d::Zero();
  for (const auto& vertex : vertices_) {
    min_corner_ = min_corner_.cwiseMin(vertex);
    max_corner_ = max_corner_.cwiseMax(vertex);
    center_ += vertex;
  }
  center_ /= static_cast<double>(vertices_.size());
  bounds_.min = min_corner_.head<2>();
  bounds_.max = max_corner_.head<2>();
}

void PolytopeSpatialDefinition::ComputeFaces() {
  faces_.clear();
  for (const auto& indices : face_indices_) {
    if (indices.size() < 3) continue;
    bool valid = true;
    for (int index : indices) {
      if (index < 0 ||
          static_cast<size_t>(index) >= vertices_.size()) {
        valid = false;
        break;
      }
    }
    if (!valid) continue;

    const Eigen::Vector3d& a = vertices_[indices[0]];
    const Eigen::Vector3d& b = vertices_[indices[1]];
    const Eigen::Vector3d& c = vertices_[indices[2]];
    Eigen::Vector3d normal = (b - a).cross(c - a);
    double norm = normal.norm();
    if (norm < kEpsilon) continue;
    normal /= norm;

    Face face;
    face.indices = indices;
    for (int index : indices) {
      face.center += vertices_[index];
    }
    face.center /= static_cast<double>(indices.size());

    if (normal.dot(center_ - face.center) > 0.0) {
      normal = -normal;
    }
    face.normal = normal;
    face.offset = face.normal.dot(face.center);
    faces_.push_back(std::move(face));
  }
}

void PolytopeSpatialDefinition::ComputeBoundary() {
  std::vector<Eigen::Vector2d> projected;
  projected.reserve(vertices_.size());
  for (const auto& vertex : vertices_) {
    projected.push_back(vertex.head<2>());
  }
  boundary_2d_ =
      std::make_shared<Polygon2D>(ConvexHull2D(std::move(projected)));
}

}  // namespace spatialblend
