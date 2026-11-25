// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_GEOMETRY_RADIAL_SURFACE_CHART_H_
#define SPATIALBLEND_GEOMETRY_RADIAL_SURFACE_CHART_H_

#include <cstdint>

#include "Eigen/Core"
#include "absl/status/statusor.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/geometry/polygon_coordinate_mapper.h"

namespace spatialblend {

// A radial chart for a valid convex polygon or a strictly star-shaped simple
// polygon. This is intentionally independent of PolygonCoordinateMapper: the
// latter retains MVC for general concave topology. Surface-chart vertices are
// radial only when every direction from the selected origin has one bounded
// polygon boundary hit. This planar chart accepts only kRadial packets with
// empty MVC weights and height_ratio exactly zero (including signed zero).
//
// The chart is pure radial, as in v1: a point at distance r along direction
// theta has ratio rho = r / B(theta), with B the origin-to-boundary distance
// along theta, and Decode places ratio rho at rho * B(theta) on the receiver.
// The angle is never altered. Encode inverts Decode exactly. (The
// core regularization no longer exists.)
//
// The chart version travels with the packet
// (NormalizedPolygonPosition::chart_version, SupportProto field 17). 0 and 1
// are this chart; version 2 (the deleted regularized chart) and newer are
// rejected.
class RadialSurfaceChart {
 public:
  static constexpr uint32_t kChartVersion = 1;

  static absl::StatusOr<NormalizedPolygonPosition> Encode(
      const Eigen::Vector2d& point, const Polygon2D& polygon,
      const Eigen::Vector2d& forward_direction);

  static absl::StatusOr<Eigen::Vector2d> Decode(
      const NormalizedPolygonPosition& normalized, const Polygon2D& polygon,
      const Eigen::Vector2d& forward_direction);

  // Decode with the chart's RADIUS (the ratio in `normalized`) along
  // the ray from the chart origin through `boundary_point` (a point of this
  // polygon's boundary) instead of along `normalized.angle`. This is the radial
  // chart with its angle taken from a boundary-to-boundary (arc-length) map of
  // the source's boundary point onto this polygon, the receiver-convexity
  // chart (NormalizedPolygonPosition::boundary_arclen): for the same polygon
  // the boundary point is the encode-side ray hit and the result is the plain
  // Decode. The ratio and version are unchanged.
  static absl::StatusOr<Eigen::Vector2d> DecodeTowardBoundaryPoint(
      const NormalizedPolygonPosition& normalized, const Polygon2D& polygon,
      const Eigen::Vector2d& boundary_point);

  // Convex polygons use their area centroid. Concave polygons use the strict
  // interior centroid of the kernel, obtained by clipping an enclosing box by
  // the orientation-aware interior halfplanes of every edge.
  static absl::StatusOr<Eigen::Vector2d> Origin(const Polygon2D& polygon);
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_GEOMETRY_RADIAL_SURFACE_CHART_H_
