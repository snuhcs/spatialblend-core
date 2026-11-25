// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/polygon2d.h
#ifndef SPATIALBLEND_GEOMETRY_POLYGON2D_H_
#define SPATIALBLEND_GEOMETRY_POLYGON2D_H_

#include <Eigen/Dense>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace spatialblend {

namespace internal {

// Edge data structure for efficient nearest point queries via R-tree.
struct EdgeData {
  Eigen::Vector2d p1;
  Eigen::Vector2d p2;
  size_t ring_index;
  size_t edge_index_in_ring;
};

}  // namespace internal

// 2D polygon with Boost.Geometry backend.
// Supports containment, nearest point, distance, segment intersection.
// Uses Boost.Geometry internally but exposes Eigen::Vector2d interface.
//
// Spatial queries (GetNearestPoint, IntersectsSegment, RayBoundaryIntersection)
// use a lazily-built R-tree index for O(log n) performance.
class Polygon2D {
 public:
  Polygon2D() = default;
  explicit Polygon2D(const std::vector<Eigen::Vector2d>& points);

  // Copy constructor — copies polygon data; indices are lazily rebuilt.
  Polygon2D(const Polygon2D& other);

  // Move constructor — moves polygon data; indices are lazily rebuilt.
  Polygon2D(Polygon2D&& other) noexcept;

  // Copy assignment — copies polygon data; indices are lazily rebuilt.
  Polygon2D& operator=(const Polygon2D& other);

  // Move assignment — moves polygon data; indices are lazily rebuilt.
  Polygon2D& operator=(Polygon2D&& other) noexcept;

  // Queries
  bool Contains(const Eigen::Vector2d& point,
                bool include_boundary = true) const;
  Eigen::Vector2d GetNearestPoint(const Eigen::Vector2d& point) const;
  double GetDistance(const Eigen::Vector2d& point) const;
  bool IntersectsSegment(const Eigen::Vector2d& p1,
                         const Eigen::Vector2d& p2) const;

  // Properties
  const std::vector<Eigen::Vector2d>& GetPoints() const;
  // Components after the largest. Empty for a single ring.
  const std::vector<std::vector<Eigen::Vector2d>>& ExtraRings() const {
    return extra_rings_;
  }
  Eigen::Vector2d GetCenter() const;
  double GetArea() const;
  // Area of the intersection with `other` (0 for disjoint or edge-touching).
  double IntersectionArea(const Polygon2D& other) const;
  size_t NumPoints() const;
  // Outer rings after a buffer that splits. 0 for an empty polygon (an erosion
  // that removed the whole region). 1 for an ordinary simple polygon.
  size_t NumComponents() const;
  // True if the (corrected) outer ring is convex. Backed by
  // boost::geometry::is_convex — matches spatialblend::BasePolygon2D::IsConvex.
  // A region of more than one component is not convex.
  bool IsConvex() const;

  // Ray casting
  std::optional<Eigen::Vector2d> RayBoundaryIntersection(
      const Eigen::Vector2d& origin,
      const Eigen::Vector2d& direction) const;

  // Operations
  Polygon2D Inflate(double distance) const;

  // Shortest distance between the two regions (every component of each); 0 when
  // they touch or overlap, +inf when either is empty.
  double DistanceTo(const Polygon2D& other) const;
  // The region common to both (every component). Empty when they do not overlap.
  Polygon2D Intersection(const Polygon2D& other) const;
  // The union of the regions: the largest component is the primary ring, further
  // components are ExtraRings(). A pocket the union encloses stays a hole
  // (Holes()): Contains() excludes it.
  static Polygon2D Union(const std::vector<Polygon2D>& parts);
  // The inner rings (holes) of every component. GetPoints()/ExtraRings() are the
  // outer rings; Contains(), GetArea() and IntersectionArea() respect the holes.
  std::vector<std::vector<Eigen::Vector2d>> Holes() const;
  // The region (holes included) under p -> a * p + t.
  Polygon2D Transformed(const Eigen::Matrix2d& a, const Eigen::Vector2d& t) const;
  // Whether every component is a valid simple polygon (no self-intersection).
  bool IsValid() const;
  // A valid polygon of the same region, for an input that is invalid only by
  // features of zero area: repeated vertices, spikes (an edge that doubles back on
  // itself) and zero-width slits (a keyhole cut joining a hole to the outer ring).
  // Real holes are kept. Empty (NumComponents() == 0) when no such repair exists:
  // a ring that crosses itself encloses parts of opposite orientation, so the repair
  // would change its area, and it is refused.
  Polygon2D Repaired() const;

  // Pre-build spatial indices for predictable query latency.
  // By default, indices are lazily built on first query. Call this method
  // during initialization to avoid latency spikes during real-time operations.
  void EnsureEdgeIndex() const;

 private:
  friend struct Polygon2DAccess;  // polygon2d.cc: builds a Polygon2D from Boost results

  std::vector<Eigen::Vector2d> points_;
  // Outer rings of every component after the largest (points_ / impl_->bg_poly).
  // Empty when the polygon is a single ring. Indexed with the primary ring so
  // containment, nearest-point and segment tests see the whole region.
  std::vector<std::vector<Eigen::Vector2d>> extra_rings_;

  // Forward-declared pimpl for Boost.Geometry internals
  struct Impl;
  std::shared_ptr<Impl> impl_;

  // R-tree spatial index (lazily built, mutable for const query methods)
  struct SpatialIndex;
  mutable std::shared_ptr<SpatialIndex> spatial_index_;
  mutable std::mutex index_mutex_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_GEOMETRY_POLYGON2D_H_
