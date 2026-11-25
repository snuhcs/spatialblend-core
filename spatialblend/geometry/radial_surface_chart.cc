// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/geometry/radial_surface_chart.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "absl/status/status.h"

namespace spatialblend {
namespace {

constexpr double kRoundoffUlps = 128.0;

double Cross(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
  return a.x() * b.y() - a.y() * b.x();
}

double CoordinateScale(const std::vector<Eigen::Vector2d>& vertices) {
  double scale = 1.0;
  for (const auto& vertex : vertices) {
    scale = std::max(scale, std::abs(vertex.x()));
    scale = std::max(scale, std::abs(vertex.y()));
  }
  return scale;
}

double GeometryScale(const std::vector<Eigen::Vector2d>& vertices) {
  double scale = 1.0;
  const Eigen::Vector2d reference = vertices.front();
  for (const auto& vertex : vertices) {
    scale = std::max(scale, (vertex - reference).norm());
  }
  return scale;
}

double Tolerance(double scale) {
  return kRoundoffUlps * std::numeric_limits<double>::epsilon() * scale;
}

struct ValidPolygon {
  std::vector<Eigen::Vector2d> vertices;
  double scale;
  // Local topology/ray tolerance: never grows with a world translation.
  double local_length_tolerance;
  double area_tolerance;
  // Only for snapping an otherwise-outside input back to a represented edge.
  double canonical_boundary_tolerance;
  int orientation;
  bool convex;
};

bool OnSegment(const Eigen::Vector2d& point, const Eigen::Vector2d& a,
               const Eigen::Vector2d& b, double tolerance) {
  const Eigen::Vector2d edge = b - a;
  const double length = edge.norm();
  if (length <= tolerance) return false;
  if (std::abs(Cross(edge, point - a)) > tolerance * length) return false;
  return (point - a).dot(point - b) <= tolerance * length;
}

double PairCrossTolerance(const Eigen::Vector2d& first,
                          const Eigen::Vector2d& second) {
  return kRoundoffUlps * std::numeric_limits<double>::epsilon() *
         std::max(first.norm() * second.norm(),
                  std::numeric_limits<double>::min());
}

int PairOrientation(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                    const Eigen::Vector2d& c) {
  const Eigen::Vector2d first = b - a;
  const Eigen::Vector2d second = c - a;
  const double value = Cross(first, second);
  const double tolerance = PairCrossTolerance(first, second);
  if (value > tolerance) return 1;
  if (value < -tolerance) return -1;
  return 0;
}

bool AabbOverlaps(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                  const Eigen::Vector2d& c, const Eigen::Vector2d& d) {
  return std::max(std::min(a.x(), b.x()), std::min(c.x(), d.x())) <=
             std::min(std::max(a.x(), b.x()), std::max(c.x(), d.x())) &&
         std::max(std::min(a.y(), b.y()), std::min(c.y(), d.y())) <=
             std::min(std::max(a.y(), b.y()), std::max(c.y(), d.y()));
}

bool SegmentsIntersect(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                       const Eigen::Vector2d& c, const Eigen::Vector2d& d) {
  const int ab_c = PairOrientation(a, b, c);
  const int ab_d = PairOrientation(a, b, d);
  const int cd_a = PairOrientation(c, d, a);
  const int cd_b = PairOrientation(c, d, b);
  // Do not let a tiny local crossing disappear beneath the overall ring's
  // area tolerance. Near-collinear cases whose segment boxes overlap are
  // intentionally rejected as ambiguous rather than admitted as "simple".
  if ((ab_c == 0 || ab_d == 0 || cd_a == 0 || cd_b == 0) &&
      AabbOverlaps(a, b, c, d)) {
    return true;
  }
  return ab_c != ab_d && cd_a != cd_b;
}

absl::StatusOr<ValidPolygon> Validate(const Polygon2D& polygon) {
  std::vector<Eigen::Vector2d> vertices = polygon.GetPoints();
  if (vertices.size() < 3) {
    return absl::InvalidArgumentError("Polygon needs at least three vertices");
  }
  for (const auto& vertex : vertices) {
    if (!vertex.allFinite()) {
      return absl::InvalidArgumentError("Polygon vertices must be finite");
    }
  }
  const double scale = GeometryScale(vertices);
  // Local topology/ray calculations use only geometric extent. Absolute world
  // coordinate precision is reserved for the explicitly bounded outside-to-
  // boundary canonicalization path, never for radial displacement or u.
  const double local_length_tolerance = Tolerance(scale);
  const double area_tolerance = local_length_tolerance * scale;
  const double canonical_boundary_tolerance =
      Tolerance(std::max(scale, CoordinateScale(vertices)));
  // Polygon2D preserves its input ring, while common geometry producers emit
  // a conventional exactly repeated closing vertex. Accept only that exact
  // representation; a near-closing final point is a real edge and is checked
  // below rather than silently changing topology.
  if (vertices.size() > 3 &&
      vertices.front().x() == vertices.back().x() &&
      vertices.front().y() == vertices.back().y()) {
    vertices.pop_back();
  }
  if (vertices.size() < 3) {
    return absl::InvalidArgumentError(
        "Polygon needs at least three distinct vertices");
  }
  const size_t count = vertices.size();
  for (size_t i = 0; i < count; ++i) {
    if ((vertices[(i + 1) % count] - vertices[i]).norm() <=
        local_length_tolerance) {
      return absl::InvalidArgumentError(
          "Polygon has repeated or zero-length edges");
    }
  }
  // Translation-stable signed area and orientation.
  const Eigen::Vector2d reference = vertices.front();
  double twice_area = 0.0;
  for (size_t i = 0; i < count; ++i) {
    twice_area +=
        Cross(vertices[i] - reference, vertices[(i + 1) % count] - reference);
  }
  if (!std::isfinite(twice_area) || std::abs(twice_area) <= area_tolerance) {
    return absl::InvalidArgumentError("Polygon area is degenerate");
  }
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = i + 1; j < count; ++j) {
      if (j == i || (j + 1) % count == i || (i + 1) % count == j) continue;
      if (SegmentsIntersect(vertices[i], vertices[(i + 1) % count], vertices[j],
                            vertices[(j + 1) % count])) {
        return absl::InvalidArgumentError("Polygon must be simple");
      }
    }
  }
  bool convex = true;
  const int winding = twice_area > 0.0 ? 1 : -1;
  for (size_t i = 0; i < count; ++i) {
    const double turn =
        Cross(vertices[(i + 1) % count] - vertices[i],
              vertices[(i + 2) % count] - vertices[(i + 1) % count]);
    if (winding * turn < -area_tolerance) convex = false;
  }
  return ValidPolygon{std::move(vertices),
                      scale,
                      local_length_tolerance,
                      area_tolerance,
                      canonical_boundary_tolerance,
                      winding,
                      convex};
}

Eigen::Vector2d AreaCentroid(const ValidPolygon& polygon) {
  const Eigen::Vector2d reference = polygon.vertices.front();
  double twice_area = 0.0;
  Eigen::Vector2d numerator = Eigen::Vector2d::Zero();
  for (size_t i = 0; i < polygon.vertices.size(); ++i) {
    const Eigen::Vector2d a = polygon.vertices[i] - reference;
    const Eigen::Vector2d b =
        polygon.vertices[(i + 1) % polygon.vertices.size()] - reference;
    const double cross = Cross(a, b);
    twice_area += cross;
    numerator += (a + b) * cross;
  }
  return reference + numerator / (3.0 * twice_area);
}

bool StrictlyInsideHalfplanes(const Eigen::Vector2d& point,
                              const ValidPolygon& polygon) {
  for (size_t i = 0; i < polygon.vertices.size(); ++i) {
    const Eigen::Vector2d edge =
        polygon.vertices[(i + 1) % polygon.vertices.size()] -
        polygon.vertices[i];
    if (polygon.orientation * Cross(edge, point - polygon.vertices[i]) <=
        polygon.area_tolerance) {
      return false;
    }
  }
  return true;
}

std::vector<Eigen::Vector2d> ClipByInteriorHalfplane(
    const std::vector<Eigen::Vector2d>& subject, const Eigen::Vector2d& a,
    const Eigen::Vector2d& b, int orientation, double area_tolerance) {
  std::vector<Eigen::Vector2d> output;
  if (subject.empty()) return output;
  const Eigen::Vector2d edge = b - a;
  auto value = [&](const Eigen::Vector2d& point) {
    return orientation * Cross(edge, point - a);
  };
  Eigen::Vector2d previous = subject.back();
  double previous_value = value(previous);
  bool previous_inside = previous_value >= -area_tolerance;
  for (const Eigen::Vector2d& current : subject) {
    const double current_value = value(current);
    const bool current_inside = current_value >= -area_tolerance;
    if (current_inside != previous_inside) {
      const double denominator = previous_value - current_value;
      if (std::abs(denominator) > area_tolerance) {
        const double t = previous_value / denominator;
        output.push_back(previous + t * (current - previous));
      }
    }
    if (current_inside) output.push_back(current);
    previous = current;
    previous_value = current_value;
    previous_inside = current_inside;
  }
  return output;
}

bool PointInPolygonOrBoundary(const Eigen::Vector2d& point,
                              const ValidPolygon& polygon) {
  bool inside = false;
  for (size_t i = 0, j = polygon.vertices.size() - 1;
       i < polygon.vertices.size(); j = i++) {
    const Eigen::Vector2d& a = polygon.vertices[j];
    const Eigen::Vector2d& b = polygon.vertices[i];
    // Exact boundary membership is part of the domain. A near-outside point is
    // handled separately by CanonicalizePoint's explicitly bounded snap.
    if (OnSegment(point, a, b, 0.0)) return true;
    const bool straddles = (a.y() > point.y()) != (b.y() > point.y());
    if (straddles) {
      const double x =
          a.x() + (point.y() - a.y()) * (b.x() - a.x()) / (b.y() - a.y());
      if (point.x() < x) inside = !inside;
    }
  }
  return inside;
}

Eigen::Vector2d NearestBoundaryPoint(const Eigen::Vector2d& point,
                                     const ValidPolygon& polygon) {
  Eigen::Vector2d nearest = polygon.vertices.front();
  double best = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < polygon.vertices.size(); ++i) {
    const Eigen::Vector2d a = polygon.vertices[i];
    const Eigen::Vector2d edge =
        polygon.vertices[(i + 1) % polygon.vertices.size()] - a;
    const double t =
        std::clamp((point - a).dot(edge) / edge.squaredNorm(), 0.0, 1.0);
    const Eigen::Vector2d candidate = a + t * edge;
    const double distance = (point - candidate).norm();
    if (distance < best) {
      best = distance;
      nearest = candidate;
    }
  }
  return nearest;
}

absl::StatusOr<Eigen::Vector2d> CanonicalizePoint(const Eigen::Vector2d& point,
                                                  const ValidPolygon& polygon) {
  if (!point.allFinite())
    return absl::InvalidArgumentError("Point must be finite");
  if (PointInPolygonOrBoundary(point, polygon)) return point;
  const Eigen::Vector2d nearest = NearestBoundaryPoint(point, polygon);
  if ((nearest - point).norm() <= polygon.canonical_boundary_tolerance)
    return nearest;
  return absl::InvalidArgumentError("Point is outside the polygon");
}

absl::StatusOr<double> RayBoundaryDistance(const Eigen::Vector2d& origin,
                                           const Eigen::Vector2d& direction,
                                           const ValidPolygon& polygon) {
  double best = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < polygon.vertices.size(); ++i) {
    const Eigen::Vector2d a = polygon.vertices[i];
    const Eigen::Vector2d edge =
        polygon.vertices[(i + 1) % polygon.vertices.size()] - a;
    const double denominator = Cross(direction, edge);
    if (std::abs(denominator) <= polygon.local_length_tolerance) continue;
    const double t = Cross(a - origin, edge) / denominator;
    const double u = Cross(a - origin, direction) / denominator;
    // u is a dimensionless segment parameter: using a world-length epsilon
    // here admits ray/line extension hits after large translations.
    constexpr double kSegmentParameterTolerance =
        kRoundoffUlps * std::numeric_limits<double>::epsilon();
    if (t > polygon.local_length_tolerance &&
        u >= -kSegmentParameterTolerance &&
        u <= 1.0 + kSegmentParameterTolerance) {
      best = std::min(best, t);
    }
  }
  if (!std::isfinite(best))
    return absl::InvalidArgumentError("Ray misses polygon boundary");
  return best;
}

absl::StatusOr<Eigen::Vector2d> OriginForValidated(
    const ValidPolygon& polygon) {
  if (polygon.convex) {
    const Eigen::Vector2d origin = AreaCentroid(polygon);
    if (!origin.allFinite() || !StrictlyInsideHalfplanes(origin, polygon)) {
      return absl::InvalidArgumentError(
          "Convex polygon has no strict chart origin");
    }
    return origin;
  }
  const Eigen::Vector2d center = AreaCentroid(polygon);
  const double extent = polygon.scale * 4.0;
  std::vector<Eigen::Vector2d> kernel = {
      center + Eigen::Vector2d(-extent, -extent),
      center + Eigen::Vector2d(extent, -extent),
      center + Eigen::Vector2d(extent, extent),
      center + Eigen::Vector2d(-extent, extent)};
  for (size_t i = 0; i < polygon.vertices.size(); ++i) {
    kernel = ClipByInteriorHalfplane(
        kernel, polygon.vertices[i],
        polygon.vertices[(i + 1) % polygon.vertices.size()],
        polygon.orientation, polygon.area_tolerance);
    if (kernel.size() < 3) {
      return absl::InvalidArgumentError(
          "Concave polygon has an empty star kernel");
    }
  }
  double kernel_twice_area = 0.0;
  const Eigen::Vector2d kernel_reference = kernel.front();
  for (size_t i = 0; i < kernel.size(); ++i) {
    kernel_twice_area +=
        Cross(kernel[i] - kernel_reference,
              kernel[(i + 1) % kernel.size()] - kernel_reference);
  }
  if (!std::isfinite(kernel_twice_area) ||
      std::abs(kernel_twice_area) <= polygon.area_tolerance) {
    return absl::InvalidArgumentError(
        "Concave polygon has no strict two-dimensional kernel");
  }
  ValidPolygon kernel_polygon{std::move(kernel),
                              polygon.scale,
                              polygon.local_length_tolerance,
                              polygon.area_tolerance,
                              polygon.canonical_boundary_tolerance,
                              1,
                              true};
  const Eigen::Vector2d origin = AreaCentroid(kernel_polygon);
  if (!origin.allFinite() || !StrictlyInsideHalfplanes(origin, polygon) ||
      !PointInPolygonOrBoundary(origin, polygon)) {
    return absl::InvalidArgumentError(
        "Concave polygon has no strict chart origin");
  }
  return origin;
}

}  // namespace

absl::StatusOr<Eigen::Vector2d> RadialSurfaceChart::Origin(
    const Polygon2D& polygon) {
  auto validated = Validate(polygon);
  if (!validated.ok()) return validated.status();
  return OriginForValidated(*validated);
}

absl::StatusOr<NormalizedPolygonPosition> RadialSurfaceChart::Encode(
    const Eigen::Vector2d& point, const Polygon2D& polygon,
    const Eigen::Vector2d& forward_direction) {
  auto validated = Validate(polygon);
  if (!validated.ok()) return validated.status();
  if (!forward_direction.allFinite() ||
      (forward_direction.x() == 0.0 && forward_direction.y() == 0.0)) {
    return absl::InvalidArgumentError(
        "Forward direction must be finite and nonzero");
  }
  auto canonical = CanonicalizePoint(point, *validated);
  if (!canonical.ok()) return canonical.status();
  auto origin = OriginForValidated(*validated);
  if (!origin.ok()) return origin.status();
  const Eigen::Vector2d delta = *canonical - *origin;
  NormalizedPolygonPosition result;
  result.strategy = NormalizedPolygonPosition::Strategy::kRadial;
  result.height_ratio = 0.0;
  result.chart_version = kChartVersion;
  if (delta.x() == 0.0 && delta.y() == 0.0) return result;
  const double delta_distance = std::hypot(delta.x(), delta.y());
  const Eigen::Vector2d direction = delta / delta_distance;
  const double forward_angle =
      std::atan2(forward_direction.y(), forward_direction.x());
  result.angle = std::atan2(direction.y(), direction.x()) - forward_angle;
  while (result.angle > M_PI) result.angle -= 2.0 * M_PI;
  while (result.angle < -M_PI) result.angle += 2.0 * M_PI;
  auto boundary_distance = RayBoundaryDistance(*origin, direction, *validated);
  if (!boundary_distance.ok()) return boundary_distance.status();
  result.radial_ratio = delta_distance / *boundary_distance;
  if (result.radial_ratio < 0.0 ||
      result.radial_ratio >
          1.0 + validated->local_length_tolerance / *boundary_distance) {
    return absl::InvalidArgumentError("Point is outside radial chart domain");
  }
  // A bounded endpoint roundoff residue is canonicalized; no material clamp.
  if (result.radial_ratio > 1.0) result.radial_ratio = 1.0;
  return result;
}

absl::StatusOr<Eigen::Vector2d> RadialSurfaceChart::Decode(
    const NormalizedPolygonPosition& normalized, const Polygon2D& polygon,
    const Eigen::Vector2d& forward_direction) {
  auto validated = Validate(polygon);
  if (!validated.ok()) return validated.status();
  if (normalized.strategy != NormalizedPolygonPosition::Strategy::kRadial ||
      !std::isfinite(normalized.angle) ||
      !std::isfinite(normalized.radial_ratio) ||
      !std::isfinite(normalized.height_ratio) ||
      !normalized.mvc_weights.empty() || normalized.height_ratio != 0.0 ||
      normalized.angle < -M_PI ||
      normalized.angle > M_PI || normalized.radial_ratio < 0.0 ||
      normalized.radial_ratio > 1.0) {
    return absl::InvalidArgumentError("Invalid radial surface-chart packet");
  }
  if (!forward_direction.allFinite() ||
      (forward_direction.x() == 0.0 && forward_direction.y() == 0.0)) {
    return absl::InvalidArgumentError(
        "Forward direction must be finite and nonzero");
  }
  // Chart version (wire field): 0 and 1 are the pure radial chart. The
  // regularized version 2 was deleted; it and anything newer
  // is rejected rather than silently misread.
  if (normalized.chart_version > kChartVersion)
    return absl::InvalidArgumentError("Unknown radial surface-chart version");
  auto origin = OriginForValidated(*validated);
  if (!origin.ok()) return origin.status();
  if (normalized.radial_ratio == 0.0) return *origin;
  const double absolute_angle =
      std::atan2(forward_direction.y(), forward_direction.x()) +
      normalized.angle;
  const Eigen::Vector2d direction(std::cos(absolute_angle),
                                  std::sin(absolute_angle));
  auto boundary_distance = RayBoundaryDistance(*origin, direction, *validated);
  if (!boundary_distance.ok()) return boundary_distance.status();
  double radius = normalized.radial_ratio * *boundary_distance;
  const Eigen::Vector2d point = *origin + direction * radius;
  if (!point.allFinite()) {
    return absl::InvalidArgumentError(
        "Decoded point is outside radial chart domain");
  }
  return CanonicalizePoint(point, *validated);
}

absl::StatusOr<Eigen::Vector2d> RadialSurfaceChart::DecodeTowardBoundaryPoint(
    const NormalizedPolygonPosition& normalized, const Polygon2D& polygon,
    const Eigen::Vector2d& boundary_point) {
  auto validated = Validate(polygon);
  if (!validated.ok()) return validated.status();
  if (normalized.strategy != NormalizedPolygonPosition::Strategy::kRadial ||
      !std::isfinite(normalized.radial_ratio) ||
      !normalized.mvc_weights.empty() || normalized.radial_ratio < 0.0 ||
      normalized.radial_ratio > 1.0 || !boundary_point.allFinite()) {
    return absl::InvalidArgumentError("Invalid radial surface-chart packet");
  }
  if (normalized.chart_version > kChartVersion)
    return absl::InvalidArgumentError("Unknown radial surface-chart version");
  auto origin = OriginForValidated(*validated);
  if (!origin.ok()) return origin.status();
  if (normalized.radial_ratio == 0.0) return *origin;
  const Eigen::Vector2d to_boundary = boundary_point - *origin;
  if (to_boundary.norm() <= validated->local_length_tolerance)
    return absl::InvalidArgumentError("Boundary point at the chart origin");
  const Eigen::Vector2d direction = to_boundary.normalized();
  auto boundary_distance = RayBoundaryDistance(*origin, direction, *validated);
  if (!boundary_distance.ok()) return boundary_distance.status();
  double radius = normalized.radial_ratio * *boundary_distance;
  const Eigen::Vector2d point = *origin + direction * radius;
  if (!point.allFinite())
    return absl::InvalidArgumentError(
        "Decoded point is outside radial chart domain");
  return CanonicalizePoint(point, *validated);
}

}  // namespace spatialblend
