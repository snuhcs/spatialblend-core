// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/geometry/polygon_coordinate_mapper.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>

#include "absl/status/statusor.h"

namespace spatialblend {

namespace {

// Normalize angle to [-π, π]
double NormalizeAngle(double angle) {
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

}  // namespace

Eigen::Vector2d PolygonCoordinateMapper::ComputeCentroid(
    const Polygon2D& polygon) {
  // v2 Polygon2D is single-ring; its GetPoints() is the outer ring.
  const auto& pts = polygon.GetPoints();
  if (pts.size() < 3) {
    // Fallback: vertex average
    if (pts.empty()) return Eigen::Vector2d::Zero();
    Eigen::Vector2d sum = Eigen::Vector2d::Zero();
    for (const auto& p : pts) sum += p;
    return sum / static_cast<double>(pts.size());
  }

  int n = static_cast<int>(pts.size());

  // Area-weighted centroid via shoelace formula
  double signed_area = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  for (int i = 0; i < n; ++i) {
    int j = (i + 1) % n;
    double cross = pts[i].x() * pts[j].y() - pts[j].x() * pts[i].y();
    signed_area += cross;
    cx += (pts[i].x() + pts[j].x()) * cross;
    cy += (pts[i].y() + pts[j].y()) * cross;
  }
  signed_area *= 0.5;
  if (std::abs(signed_area) < 1e-12) {
    // Degenerate polygon, fallback to vertex average
    Eigen::Vector2d sum = Eigen::Vector2d::Zero();
    for (const auto& p : pts) sum += p;
    return sum / static_cast<double>(n);
  }
  double factor = 1.0 / (6.0 * signed_area);
  return Eigen::Vector2d(cx * factor, cy * factor);
}

std::vector<Eigen::Vector2d> PolygonCoordinateMapper::ResampleBoundary(
    const Polygon2D& polygon, const Eigen::Vector2d& forward_direction,
    int num_samples) {
  const auto& pts = polygon.GetPoints();
  if (pts.size() < 3 || num_samples < 2) {
    return {};
  }

  int n = static_cast<int>(pts.size());

  // Compute centroid and find start point via ray intersection
  Eigen::Vector2d centroid = ComputeCentroid(polygon);
  auto start_opt = polygon.RayBoundaryIntersection(centroid, forward_direction);
  if (!start_opt.has_value()) {
    // Fallback: use the vertex closest to the forward direction
    start_opt = pts[0];
  }
  Eigen::Vector2d start_point = start_opt.value();

  // Build the ordered edge list starting from the edge containing start_point
  // Find which edge contains start_point
  int start_edge = -1;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < n; ++i) {
    int j = (i + 1) % n;
    Eigen::Vector2d edge = pts[j] - pts[i];
    double edge_len_sq = edge.squaredNorm();
    if (edge_len_sq < 1e-18) continue;
    double t = (start_point - pts[i]).dot(edge) / edge_len_sq;
    t = std::clamp(t, 0.0, 1.0);
    Eigen::Vector2d projected = pts[i] + t * edge;
    double dist = (start_point - projected).norm();
    if (dist < min_dist) {
      min_dist = dist;
      start_edge = i;
    }
  }
  if (start_edge < 0) return {};

  // Compute total perimeter
  double total_perimeter = 0.0;
  for (int i = 0; i < n; ++i) {
    int j = (i + 1) % n;
    total_perimeter += (pts[j] - pts[i]).norm();
  }
  if (total_perimeter < 1e-12) return {};

  double spacing = total_perimeter / num_samples;

  // Walk the boundary from start_point, placing samples at equal spacing
  std::vector<Eigen::Vector2d> samples;
  samples.reserve(num_samples);
  samples.push_back(start_point);

  // Current position along the boundary
  int current_edge = start_edge;
  double remaining_on_edge =
      (pts[(current_edge + 1) % n] - start_point).norm();
  // If start_t is very close to 1.0, we're at the end of the edge
  if (remaining_on_edge < 1e-12) {
    current_edge = (current_edge + 1) % n;
    remaining_on_edge = (pts[(current_edge + 1) % n] - pts[current_edge]).norm();
  }

  Eigen::Vector2d current_pos = start_point;

  for (int s = 1; s < num_samples; ++s) {
    double distance_to_walk = spacing;
    while (distance_to_walk > 1e-12) {
      int next_vertex = (current_edge + 1) % n;
      if (remaining_on_edge >= distance_to_walk - 1e-12) {
        // Place sample on this edge
        Eigen::Vector2d edge_dir =
            (pts[next_vertex] - current_pos).normalized();
        current_pos = current_pos + edge_dir * distance_to_walk;
        remaining_on_edge -= distance_to_walk;
        distance_to_walk = 0.0;
      } else {
        // Move to end of this edge
        distance_to_walk -= remaining_on_edge;
        current_edge = next_vertex;
        current_pos = pts[current_edge];
        int next_next = (current_edge + 1) % n;
        remaining_on_edge = (pts[next_next] - pts[current_edge]).norm();
      }
    }
    samples.push_back(current_pos);
  }

  return samples;
}

std::vector<double> PolygonCoordinateMapper::ComputeMVCWeights(
    const Eigen::Vector2d& point,
    const std::vector<Eigen::Vector2d>& polygon_vertices) {
  int n = static_cast<int>(polygon_vertices.size());
  std::vector<double> weights(n, 0.0);
  if (n < 3) return weights;

  // Mean Value Coordinates (Floater 2003)
  std::vector<Eigen::Vector2d> s(n);
  std::vector<double> r(n);
  for (int i = 0; i < n; ++i) {
    s[i] = polygon_vertices[i] - point;
    r[i] = s[i].norm();
    // If point is on a vertex, return 1.0 for that vertex
    if (r[i] < 1e-10) {
      std::fill(weights.begin(), weights.end(), 0.0);
      weights[i] = 1.0;
      return weights;
    }
  }

  std::vector<double> tan_half(n);
  for (int i = 0; i < n; ++i) {
    int next = (i + 1) % n;
    double dot = s[i].dot(s[next]);
    double cross = s[i].x() * s[next].y() - s[i].y() * s[next].x();
    double cos_alpha = dot / (r[i] * r[next]);
    // Clamp for numerical stability
    cos_alpha = std::clamp(cos_alpha, -1.0, 1.0);
    double alpha = std::acos(cos_alpha);
    // Check if point is on edge
    if (std::abs(cross) < 1e-10 && cos_alpha < -1.0 + 1e-10) {
      // Point is on the edge between vertex i and next
      double d_i = r[i];
      double d_next = r[next];
      double total = d_i + d_next;
      std::fill(weights.begin(), weights.end(), 0.0);
      weights[i] = d_next / total;
      weights[next] = d_i / total;
      return weights;
    }
    tan_half[i] = std::tan(alpha / 2.0);
  }

  double weight_sum = 0.0;
  for (int i = 0; i < n; ++i) {
    int prev = (i - 1 + n) % n;
    weights[i] = (tan_half[prev] + tan_half[i]) / r[i];
    weight_sum += weights[i];
  }

  if (std::abs(weight_sum) < 1e-12) {
    // Fallback: uniform weights
    std::fill(weights.begin(), weights.end(), 1.0 / n);
    return weights;
  }

  for (int i = 0; i < n; ++i) {
    weights[i] /= weight_sum;
  }

  return weights;
}

absl::StatusOr<NormalizedPolygonPosition>
PolygonCoordinateMapper::EncodeRadial(
    const Eigen::Vector2d& point, const Polygon2D& polygon,
    const Eigen::Vector2d& forward_direction) {
  Eigen::Vector2d centroid = ComputeCentroid(polygon);
  Eigen::Vector2d delta = point - centroid;

  NormalizedPolygonPosition result;
  result.strategy = NormalizedPolygonPosition::Strategy::kRadial;
  result.height_ratio = 0.0;

  double delta_norm = delta.norm();
  if (delta_norm < 1e-12) {
    // Point is at centroid
    result.angle = 0.0;
    result.radial_ratio = 0.0;
    return result;
  }

  // Angle relative to forward direction
  double point_angle = std::atan2(delta.y(), delta.x());
  double fwd_angle =
      std::atan2(forward_direction.y(), forward_direction.x());
  result.angle = NormalizeAngle(point_angle - fwd_angle);

  // Ray from centroid through point to find boundary distance
  Eigen::Vector2d direction = delta.normalized();
  auto boundary_hit = polygon.RayBoundaryIntersection(centroid, direction);
  if (!boundary_hit.has_value()) {
    return absl::InternalError(
        "RayBoundaryIntersection failed from centroid - polygon may be "
        "degenerate");
  }
  double boundary_dist = (boundary_hit.value() - centroid).norm();
  if (boundary_dist < 1e-12) {
    return absl::InternalError("Boundary distance is zero");
  }
  result.radial_ratio = delta_norm / boundary_dist;
  return result;
}

absl::StatusOr<Eigen::Vector2d> PolygonCoordinateMapper::DecodeRadial(
    const NormalizedPolygonPosition& normalized, const Polygon2D& polygon,
    const Eigen::Vector2d& forward_direction) {
  Eigen::Vector2d centroid = ComputeCentroid(polygon);

  if (normalized.radial_ratio < 1e-12) {
    return centroid;
  }

  // Reconstruct absolute direction from angle + forward_direction
  double fwd_angle =
      std::atan2(forward_direction.y(), forward_direction.x());
  double abs_angle = normalized.angle + fwd_angle;
  Eigen::Vector2d direction(std::cos(abs_angle), std::sin(abs_angle));

  // Ray from centroid in that direction
  auto boundary_hit = polygon.RayBoundaryIntersection(centroid, direction);
  if (!boundary_hit.has_value()) {
    return absl::InternalError(
        "RayBoundaryIntersection failed during decode - target polygon may be "
        "degenerate");
  }
  double boundary_dist = (boundary_hit.value() - centroid).norm();

  return centroid + direction * (normalized.radial_ratio * boundary_dist);
}

absl::StatusOr<NormalizedPolygonPosition> PolygonCoordinateMapper::EncodeMVC(
    const Eigen::Vector2d& point, const Polygon2D& polygon,
    const Eigen::Vector2d& forward_direction, int num_samples) {
  auto resampled = ResampleBoundary(polygon, forward_direction, num_samples);
  if (resampled.empty()) {
    return absl::InternalError("Failed to resample polygon boundary");
  }

  NormalizedPolygonPosition result;
  result.strategy = NormalizedPolygonPosition::Strategy::kMVC;
  result.height_ratio = 0.0;
  result.angle = 0.0;
  result.radial_ratio = 0.0;
  result.mvc_weights = ComputeMVCWeights(point, resampled);
  return result;
}

absl::StatusOr<Eigen::Vector2d> PolygonCoordinateMapper::DecodeMVC(
    const NormalizedPolygonPosition& normalized, const Polygon2D& polygon,
    const Eigen::Vector2d& forward_direction, int num_samples) {
  auto resampled = ResampleBoundary(polygon, forward_direction, num_samples);
  if (resampled.empty()) {
    return absl::InternalError(
        "Failed to resample target polygon boundary");
  }

  int n = static_cast<int>(resampled.size());
  if (static_cast<int>(normalized.mvc_weights.size()) != n) {
    return absl::InvalidArgumentError(
        "MVC weight count does not match target polygon sample count");
  }

  Eigen::Vector2d result = Eigen::Vector2d::Zero();
  for (int i = 0; i < n; ++i) {
    result += normalized.mvc_weights[i] * resampled[i];
  }
  return result;
}

absl::StatusOr<Eigen::Vector2d>
PolygonCoordinateMapper::CanonicalizeDomainPoint(const Eigen::Vector2d& point,
                                                  const Polygon2D& polygon) {
  if (!point.allFinite()) {
    return absl::InvalidArgumentError("Point must be finite");
  }
  if (polygon.NumPoints() < 3) {
    return absl::InvalidArgumentError("Polygon needs at least three vertices");
  }

  // Geometry enters this mapper as double after matrix composition and
  // projection. Preserve ordinary strict containment, but permit only the
  // residual expected from that double arithmetic to snap back to the exact
  // authored edge. This is intentionally not a Polygon2D::Contains policy.
  double scale = std::max(1.0, std::max(std::abs(point.x()), std::abs(point.y())));
  for (const Eigen::Vector2d& vertex : polygon.GetPoints()) {
    if (!vertex.allFinite()) {
      return absl::InvalidArgumentError("Polygon vertices must be finite");
    }
    scale = std::max(scale, std::max(std::abs(vertex.x()), std::abs(vertex.y())));
  }
  // Polygon2D::GetArea() is absolute area. Compare twice that area to the
  // grounding contract's |twice_area| threshold before a nearest-edge query:
  // a collinear or repeated ring has no chart to canonicalize.
  const double twice_area = 2.0 * polygon.GetArea();
  if (!std::isfinite(twice_area) || twice_area <= 1e-12) {
    return absl::InvalidArgumentError("Polygon area is too small for a chart");
  }
  if (polygon.Contains(point, /*include_boundary=*/true)) return point;

  constexpr double kBoundaryRoundoffUlps = 128.0;
  const double tolerance =
      kBoundaryRoundoffUlps * std::numeric_limits<double>::epsilon() * scale;
  const Eigen::Vector2d nearest = polygon.GetNearestPoint(point);
  if (!nearest.allFinite()) {
    return absl::InvalidArgumentError("Polygon boundary is not finite");
  }
  if ((point - nearest).norm() <= tolerance) return nearest;
  return absl::InvalidArgumentError("Point is outside the polygon");
}

namespace {

// Cumulative perimeter coordinate of `p`'s projection on the ring, plus the
// ring's total perimeter. Returns false for a degenerate ring.
bool ArcCoordinate(const std::vector<Eigen::Vector2d>& pts,
                   const Eigen::Vector2d& p, double* arc, double* perimeter) {
  const int n = static_cast<int>(pts.size());
  if (n < 3) return false;
  double total = 0.0, best = std::numeric_limits<double>::max(), best_arc = 0.0;
  double cum = 0.0;
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector2d a = pts[i], b = pts[(i + 1) % n];
    const Eigen::Vector2d e = b - a;
    const double len = e.norm();
    if (len > 1e-12) {
      const double t = std::clamp((p - a).dot(e) / (len * len), 0.0, 1.0);
      const double dist = (p - (a + t * e)).norm();
      if (dist < best - 1e-12) {
        best = dist;
        best_arc = cum + t * len;
      }
    }
    cum += len;
    total = cum;
  }
  if (total < 1e-12) return false;
  *arc = best_arc;
  *perimeter = total;
  return true;
}

}  // namespace

absl::StatusOr<double> PolygonCoordinateMapper::EncodeBoundaryArcLength(
    const Eigen::Vector2d& point, const Polygon2D& polygon,
    const Eigen::Vector2d& forward_direction) {
  if (!point.allFinite() || !forward_direction.allFinite() ||
      forward_direction.norm() < 1e-12)
    return absl::InvalidArgumentError("Invalid arc-length chart input");
  const auto& pts = polygon.GetPoints();
  const Eigen::Vector2d centroid = ComputeCentroid(polygon);
  auto start = polygon.RayBoundaryIntersection(centroid, forward_direction);
  if (!start.has_value()) start = pts.empty() ? Eigen::Vector2d::Zero() : pts[0];
  double a0 = 0, ap = 0, total = 0;
  if (!ArcCoordinate(pts, *start, &a0, &total) ||
      !ArcCoordinate(pts, point, &ap, &total))
    return absl::InvalidArgumentError("Polygon has no boundary chart");
  double f = std::fmod((ap - a0) / total, 1.0);
  if (f < 0.0) f += 1.0;
  return f;
}

absl::StatusOr<Eigen::Vector2d> PolygonCoordinateMapper::DecodeBoundaryArcLength(
    double arclen, const Polygon2D& polygon,
    const Eigen::Vector2d& forward_direction) {
  if (!std::isfinite(arclen) || !forward_direction.allFinite() ||
      forward_direction.norm() < 1e-12)
    return absl::InvalidArgumentError("Invalid arc-length chart input");
  const auto& pts = polygon.GetPoints();
  const int n = static_cast<int>(pts.size());
  const Eigen::Vector2d centroid = ComputeCentroid(polygon);
  auto start = polygon.RayBoundaryIntersection(centroid, forward_direction);
  if (!start.has_value()) start = n > 0 ? pts[0] : Eigen::Vector2d::Zero();
  double a0 = 0, total = 0;
  if (!ArcCoordinate(pts, *start, &a0, &total))
    return absl::InvalidArgumentError("Polygon has no boundary chart");
  double f = std::fmod(arclen, 1.0);
  if (f < 0.0) f += 1.0;
  double target = std::fmod(a0 + f * total, total);
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector2d a = pts[i], b = pts[(i + 1) % n];
    const double len = (b - a).norm();
    if (target <= len || i == n - 1) {
      const double t = len > 1e-12 ? std::clamp(target / len, 0.0, 1.0) : 0.0;
      return Eigen::Vector2d(a + t * (b - a));
    }
    target -= len;
  }
  return pts[0];
}

namespace {

struct OffsetSample {
  Eigen::Vector2d p;
  double g;  // boundary arc-length of the generating contact
  // Arc samples (about a convex vertex): vertex index, its position, the angle
  // of the sample's normal. -1 = a point of a straight offset edge / reflex
  // intersection.
  int arc = -1;
  Eigen::Vector2d center = Eigen::Vector2d::Zero();
  double ang = 0.0;
};

double DistanceToRing(const std::vector<Eigen::Vector2d>& ring,
                      const Eigen::Vector2d& p) {
  const int n = static_cast<int>(ring.size());
  double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector2d a = ring[i], e = ring[(i + 1) % n] - a;
    const double len2 = e.squaredNorm();
    const double t =
        len2 < 1e-24 ? 0.0 : std::clamp((p - a).dot(e) / len2, 0.0, 1.0);
    best = std::min(best, (p - (a + t * e)).norm());
  }
  return best;
}

Eigen::Vector2d Rot(const Eigen::Vector2d& v, double a) {
  const double c = std::cos(a), s = std::sin(a);
  return Eigen::Vector2d(c * v.x() - s * v.y(), s * v.x() + c * v.y());
}

// Closed offset curve (see PolygonCoordinateMapper::EncodeOffsetFraction),
// rotated to begin at the offset point generated at/after `start_point`, plus
// its cumulative chord length. Empty if the ring is degenerate.
bool BuildOffsetCurve(const std::vector<Eigen::Vector2d>& ring,
                      const Eigen::Vector2d& start_point, double d,
                      std::vector<Eigen::Vector2d>* curve,
                      std::vector<double>* cum, double* total) {
  const int n = static_cast<int>(ring.size());
  curve->clear();
  cum->clear();
  if (n < 3 || !(d >= 0.0)) return false;
  double area2 = 0.0;
  std::vector<double> len(n), at(n + 1, 0.0);
  std::vector<Eigen::Vector2d> nrm(n, Eigen::Vector2d::Zero());
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector2d &a = ring[i], &b = ring[(i + 1) % n];
    area2 += a.x() * b.y() - b.x() * a.y();
    len[i] = (b - a).norm();
    at[i + 1] = at[i] + len[i];
  }
  if (std::fabs(area2) < 1e-12 || at[n] < 1e-12) return false;
  const double sign = area2 > 0.0 ? 1.0 : -1.0;
  for (int i = 0; i < n; ++i) {
    if (len[i] < 1e-12) continue;
    const Eigen::Vector2d e = ring[(i + 1) % n] - ring[i];
    nrm[i] = Eigen::Vector2d(sign * e.y() / len[i], -sign * e.x() / len[i]);
  }
  const auto prev_normal = [&](int i) {
    for (int g = 1; g <= n; ++g) {
      const int k = ((i - g) % n + n) % n;
      if (len[k] >= 1e-12) return nrm[k];
    }
    return Eigen::Vector2d(0, 0);
  };
  // Arc coordinate of the start point on the ring.
  double a0 = 0.0, best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < n; ++i) {
    if (len[i] < 1e-12) continue;
    const Eigen::Vector2d a = ring[i], e = ring[(i + 1) % n] - a;
    const double t = std::clamp((start_point - a).dot(e) / (len[i] * len[i]),
                                0.0, 1.0);
    const double dist = (start_point - (a + t * e)).norm();
    if (dist < best - 1e-12) {
      best = dist;
      a0 = at[i] + t * len[i];
    }
  }
  const double tol = 1e-4 * d + 1e-7;
  std::vector<OffsetSample> raw;
  constexpr double kArcStep = M_PI / 180.0;  // 1 degree
  for (int k = 0; k < n; ++k) {
    if (len[k] < 1e-12) continue;
    // Vertex k: join of the previous edge's offset and this edge's.
    const Eigen::Vector2d np = prev_normal(k), nc = nrm[k];
    const double turn = std::remainder(
        std::atan2(nc.y(), nc.x()) - std::atan2(np.y(), np.x()), 2.0 * M_PI);
    if (d > 0.0 && np.norm() > 0.5) {
      if (turn * sign > 1e-9) {  // convex: an arc about the vertex
        const int m = std::max(1, static_cast<int>(std::ceil(std::fabs(turn) /
                                                              kArcStep)));
        const double a_prev = std::atan2(np.y(), np.x());
        for (int i = 0; i <= m; ++i)
          raw.push_back({ring[k] + d * Rot(np, turn * i / m), at[k], k, ring[k],
                         a_prev + turn * i / m});
      } else if (turn * sign < -1e-9) {  // reflex: the offset lines meet
        const double den = 1.0 + np.dot(nc);
        if (den > 1e-9)
          raw.push_back({ring[k] + d * (np + nc) / den, at[k]});
      }
    }
    // Edge k's offset segment, sampled.
    const Eigen::Vector2d e = ring[(k + 1) % n] - ring[k];
    const int m = std::clamp(
        static_cast<int>(std::ceil(len[k] / std::max(1e-3, std::min(0.01, d / 50.0 + 1e-3)))),
        1, 400);
    for (int i = 0; i <= m; ++i) {
      const double t = static_cast<double>(i) / m;
      raw.push_back({ring[k] + t * e + d * nc, at[k] + t * len[k]});
    }
  }
  // Cull the shadow of pockets (samples nearer than d to the ring), with the
  // cull boundary located exactly: a sample-granular cull moves the curve by a
  // sample spacing (an arc step, d * 1 degree) whenever a sample toggles.
  const auto valid = [&](const Eigen::Vector2d& p) {
    return d <= 0.0 || DistanceToRing(ring, p) >= d - tol;
  };
  const auto between = [&](const OffsetSample& a, const OffsetSample& b,
                           double t) {
    OffsetSample o = a;
    if (a.arc >= 0 && a.arc == b.arc) {
      o.ang = a.ang + t * (b.ang - a.ang);
      o.p = a.center + d * Eigen::Vector2d(std::cos(o.ang), std::sin(o.ang));
    } else {
      o.p = a.p + t * (b.p - a.p);
      o.g = a.g + t * (b.g - a.g);
    }
    return o;
  };
  std::vector<char> ok(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) ok[i] = valid(raw[i].p) ? 1 : 0;
  std::vector<OffsetSample> kept;
  for (size_t i = 0; i < raw.size(); ++i) {
    const size_t j = (i + 1) % raw.size();
    if (ok[i]) kept.push_back(raw[i]);
    if (ok[i] != ok[j] && !(raw[i].p - raw[j].p).isZero(1e-12)) {
      // Bisect for the transition between the valid and the culled sample.
      double lo = 0.0, hi = 1.0;
      const bool start_valid = ok[i];
      for (int it = 0; it < 40; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (valid(between(raw[i], raw[j], mid).p) == start_valid)
          lo = mid;
        else
          hi = mid;
      }
      kept.push_back(between(raw[i], raw[j], start_valid ? lo : hi));
    }
  }
  if (kept.size() < 3) return false;
  // Rotate to begin at the offset point generated by the start contact, EXACTLY:
  // the first kept sample generated at/after it, and, when the sample before it
  // is on the same straight offset edge (chord == generating-arc difference),
  // the interpolated point between them. A sample-granular start jumps by a
  // sample spacing (an arc step, d * 1 degree) whenever the culling changes,
  // which for a large stand-off is centimetres of target step.
  size_t first = 0;
  for (size_t i = 0; i < kept.size(); ++i) {
    if (kept[i].g >= a0 - 1e-9) {
      first = i;
      break;
    }
  }
  const size_t before = (first + kept.size() - 1) % kept.size();
  if (kept[first].g > a0 + 1e-9 && kept[before].g < a0 - 1e-9) {
    const double dg = kept[first].g - kept[before].g;
    const double chord = (kept[first].p - kept[before].p).norm();
    if (dg > 1e-12 && std::fabs(chord - dg) <= 1e-6 + 1e-3 * dg)
    {
      curve->push_back(kept[before].p +
                       (a0 - kept[before].g) / dg *
                           (kept[first].p - kept[before].p));
    }
  }
  for (size_t i = 0; i < kept.size(); ++i) {
    curve->push_back(kept[(first + i) % kept.size()].p);
  }
  cum->assign(curve->size() + 1, 0.0);
  for (size_t i = 0; i < curve->size(); ++i) {
    const size_t j = (i + 1) % curve->size();
    (*cum)[i + 1] = (*cum)[i] + ((*curve)[j] - (*curve)[i]).norm();
  }
  *total = cum->back();
  return *total > 1e-12;
}

// Closed-form offset curve of a CONVEX ring. There is no pocket to cull, so the
// curve is the translated edges joined by the vertex arcs, and the polyline
// BuildOffsetCurve walks (1-degree arc chords, edges as straight lines) is known
// in closed form: this struct IS that polyline, with its vertices computed by the
// same expressions, but every query (nearest point, arc-length position, point at
// a fraction, ray hit) is O(n) with O(1) work per piece instead of building and
// scanning ~360 samples with a distance-to-ring test and a 40-step bisection per
// cull transition. The curve is parameterised by chord length from the offset
// point generated at/after `start_point`.
struct ConvexOffsetCurve {
  struct Piece {
    bool arc = false;
    Eigen::Vector2d p0 = Eigen::Vector2d::Zero();      // edge: start point
    Eigen::Vector2d dir = Eigen::Vector2d::Zero();     // edge: unit direction
    Eigen::Vector2d center = Eigen::Vector2d::Zero();  // arc: the vertex
    Eigen::Vector2d n0 = Eigen::Vector2d::Zero();      // arc: first unit normal
    double a0 = 0.0;      // arc: angle of n0
    double sweep = 0.0;   // arc: signed turn
    int m = 1;            // arc: chords
    double chord = 0.0;   // arc: chord length
    double len = 0.0;     // chord length of the whole piece
    double s0 = 0.0;      // natural-order position of the piece start
  };
  std::vector<Piece> pieces;
  double d = 0.0;
  double total = 0.0;
  double s_start = 0.0;  // natural-order position of the chart origin

  // Vertex i (0..m) of an arc piece, as BuildOffsetCurve computes it.
  Eigen::Vector2d ArcVertex(const Piece& pc, int i) const {
    return pc.center + d * Rot(pc.n0, pc.sweep * i / pc.m);
  }
  Eigen::Vector2d At(const Piece& pc, double local) const {
    if (!pc.arc) return pc.p0 + pc.dir * local;
    const int i = std::clamp(
        pc.chord > 1e-15 ? static_cast<int>(local / pc.chord) : 0, 0, pc.m - 1);
    const Eigen::Vector2d a = ArcVertex(pc, i), b = ArcVertex(pc, i + 1);
    const double t = pc.chord > 1e-15
                         ? std::clamp((local - i * pc.chord) / pc.chord, 0.0, 1.0)
                         : 0.0;
    return a + t * (b - a);
  }
  // Natural-order position -> chart position in [0, total).
  double ToChart(double s_nat) const {
    double s = std::fmod(s_nat - s_start, total);
    return s < 0.0 ? s + total : s;
  }
  double FromChart(double s_chart) const {
    return std::fmod(s_start + s_chart, total);
  }
};

enum class ConvexBuild { kOk, kNotConvex, kDegenerate };

ConvexBuild BuildConvexOffsetCurve(const std::vector<Eigen::Vector2d>& ring,
                                   const Eigen::Vector2d& start_point,
                                   double d, ConvexOffsetCurve* out) {
  const int n = static_cast<int>(ring.size());
  out->pieces.clear();
  if (n < 3 || !(d >= 0.0)) return ConvexBuild::kDegenerate;
  double area2 = 0.0;
  std::vector<double> len(n), at(n + 1, 0.0);
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector2d &a = ring[i], &b = ring[(i + 1) % n];
    area2 += a.x() * b.y() - b.x() * a.y();
    len[i] = (b - a).norm();
    at[i + 1] = at[i] + len[i];
  }
  if (std::fabs(area2) < 1e-12 || at[n] < 1e-12) return ConvexBuild::kDegenerate;
  const double sign = area2 > 0.0 ? 1.0 : -1.0;
  std::vector<Eigen::Vector2d> nrm(n, Eigen::Vector2d::Zero());
  for (int i = 0; i < n; ++i) {
    if (len[i] < 1e-12) continue;
    const Eigen::Vector2d e = ring[(i + 1) % n] - ring[i];
    nrm[i] = Eigen::Vector2d(sign * e.y() / len[i], -sign * e.x() / len[i]);
  }
  const auto prev_index = [&](int i) {
    for (int g = 1; g <= n; ++g) {
      const int k = ((i - g) % n + n) % n;
      if (len[k] >= 1e-12) return k;
    }
    return -1;
  };
  constexpr double kArcStep = M_PI / 180.0;  // 1 degree, as BuildOffsetCurve
  out->d = d;
  std::vector<double> arc_s(n, -1.0), edge_s(n, -1.0);  // natural positions
  double s = 0.0;
  for (int k = 0; k < n; ++k) {
    if (len[k] < 1e-12) continue;
    const int pk = prev_index(k);
    const Eigen::Vector2d np = nrm[pk], nc = nrm[k];
    const double turn = std::remainder(
        std::atan2(nc.y(), nc.x()) - std::atan2(np.y(), np.x()), 2.0 * M_PI);
    if (turn * sign < -1e-9) return ConvexBuild::kNotConvex;  // reflex vertex
    arc_s[k] = s;
    if (d > 0.0 && np.norm() > 0.5 && turn * sign > 1e-9) {
      ConvexOffsetCurve::Piece pc;
      pc.arc = true;
      pc.center = ring[k];
      pc.n0 = np;
      pc.a0 = std::atan2(np.y(), np.x());
      pc.sweep = turn;
      pc.m = std::max(1, static_cast<int>(std::ceil(std::fabs(turn) / kArcStep)));
      pc.chord = 2.0 * d * std::sin(0.5 * std::fabs(turn) / pc.m);
      pc.len = pc.m * pc.chord;
      pc.s0 = s;
      out->pieces.push_back(pc);
      s += pc.len;
    }
    ConvexOffsetCurve::Piece pe;
    pe.p0 = ring[k] + d * nc;
    pe.dir = (ring[(k + 1) % n] - ring[k]) / len[k];
    pe.len = len[k];
    pe.s0 = s;
    edge_s[k] = s;
    out->pieces.push_back(pe);
    s += pe.len;
  }
  out->total = s;
  if (out->pieces.empty() || !(s > 1e-12)) return ConvexBuild::kDegenerate;
  // Chart origin: the offset point generated by the start contact.
  double a0 = 0.0, best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < n; ++i) {
    if (len[i] < 1e-12) continue;
    const Eigen::Vector2d a = ring[i], e = ring[(i + 1) % n] - a;
    const double t = std::clamp((start_point - a).dot(e) / (len[i] * len[i]),
                                0.0, 1.0);
    const double dist = (start_point - (a + t * e)).norm();
    if (dist < best - 1e-12) {
      best = dist;
      a0 = at[i] + t * len[i];
    }
  }
  out->s_start = 0.0;
  for (int k = 0; k < n; ++k) {
    if (len[k] < 1e-12) continue;
    // The first edge whose far end is at/after a0 (the last one takes a0 that
    // wrapped past the perimeter).
    if (at[k + 1] < a0 - 1e-9 && k + 1 < n) continue;
    const double on_edge = a0 - at[k];
    if (on_edge <= 1e-9) {
      out->s_start = arc_s[k];  // at the vertex: the start of its arc
    } else if (at[k + 1] - a0 <= 1e-9) {
      int nk = (k + 1) % n;
      while (len[nk] < 1e-12) nk = (nk + 1) % n;
      out->s_start = arc_s[nk];
    } else {
      out->s_start = edge_s[k] + on_edge;  // interior of the edge
    }
    break;
  }
  return ConvexBuild::kOk;
}

// Closest point of a segment [a, b] to p: distance and parameter in [0, 1].
double SegmentNearest(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                      const Eigen::Vector2d& p, double* t_out) {
  const Eigen::Vector2d e = b - a;
  const double l2 = e.squaredNorm();
  const double t = l2 < 1e-24 ? 0.0 : std::clamp((p - a).dot(e) / l2, 0.0, 1.0);
  *t_out = t;
  return (p - (a + t * e)).norm();
}

// Chart position (chord length from the origin) and distance of the point of
// the curve nearest to `p`.
void ConvexNearest(const ConvexOffsetCurve& c, const Eigen::Vector2d& p,
                   double* best_dist, double* best_s) {
  *best_dist = std::numeric_limits<double>::infinity();
  *best_s = 0.0;
  for (const auto& pc : c.pieces) {
    if (!pc.arc) {
      double t = 0.0;
      const double dist = SegmentNearest(pc.p0, pc.p0 + pc.dir * pc.len, p, &t);
      if (dist < *best_dist - 1e-12) {
        *best_dist = dist;
        *best_s = c.ToChart(pc.s0 + t * pc.len);
      }
      continue;
    }
    const Eigen::Vector2d r = p - pc.center;
    if (r.norm() < 1e-15) continue;
    const double rel = std::remainder(std::atan2(r.y(), r.x()) - pc.a0, 2.0 * M_PI);
    const double u = pc.sweep > 0.0 ? rel : -rel;
    if (u < -1e-9 || u > std::fabs(pc.sweep) + 1e-9) continue;  // a neighbour's end
    const int i0 = std::clamp(static_cast<int>(u / (std::fabs(pc.sweep) / pc.m)), 0,
                              pc.m - 1);
    for (int i = std::max(0, i0 - 1); i <= std::min(pc.m - 1, i0 + 1); ++i) {
      double t = 0.0;
      const double dist =
          SegmentNearest(c.ArcVertex(pc, i), c.ArcVertex(pc, i + 1), p, &t);
      if (dist < *best_dist - 1e-12) {
        *best_dist = dist;
        *best_s = c.ToChart(pc.s0 + (i + t) * pc.chord);
      }
    }
  }
}

// First crossing (t > 0) of the ray origin + t u with the curve; +inf if none.
double ConvexRayHit(const ConvexOffsetCurve& c, const Eigen::Vector2d& origin,
                    const Eigen::Vector2d& u) {
  double best = std::numeric_limits<double>::infinity();
  // origin + t u = a + s (b - a), t > 0, s in [0, 1].
  const auto hit_segment = [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    const Eigen::Vector2d e = b - a;
    const double den = u.x() * (-e.y()) - u.y() * (-e.x());
    if (std::fabs(den) < 1e-14) return;
    const Eigen::Vector2d w = a - origin;
    const double t = (w.x() * (-e.y()) - w.y() * (-e.x())) / den;
    const double sp = (u.x() * w.y() - u.y() * w.x()) / den;
    if (t > 1e-12 && sp >= -1e-9 && sp <= 1.0 + 1e-9 && t < best) best = t;
  };
  for (const auto& pc : c.pieces) {
    if (!pc.arc) {
      hit_segment(pc.p0, pc.p0 + pc.dir * pc.len);
      continue;
    }
    // The chords lie inside the arc's circle: a ray that misses the circle misses
    // them, and otherwise they are near the circle's two crossings.
    const Eigen::Vector2d w = origin - pc.center;
    const double b = u.dot(w);
    const double disc = b * b - (w.squaredNorm() - c.d * c.d);
    if (disc < 0.0) continue;
    const double sq = std::sqrt(disc);
    const double step = std::fabs(pc.sweep) / pc.m;
    for (const double t : {-b - sq, -b + sq}) {
      const Eigen::Vector2d r = w + t * u;
      const double rel = std::remainder(std::atan2(r.y(), r.x()) - pc.a0, 2.0 * M_PI);
      const double un = pc.sweep > 0.0 ? rel : -rel;
      const int i0 = std::clamp(static_cast<int>(std::floor(un / step)), 0, pc.m - 1);
      // The circle crossing may be an arc end away: test the neighbours too.
      for (int i = std::max(0, i0 - 1); i <= std::min(pc.m - 1, i0 + 1); ++i)
        hit_segment(c.ArcVertex(pc, i), c.ArcVertex(pc, i + 1));
    }
  }
  return best;
}

}  // namespace

std::optional<Eigen::Vector2d> PolygonCoordinateMapper::DecodeOffsetRay(
    const std::vector<Eigen::Vector2d>& ring, double distance,
    const Eigen::Vector2d& origin, const Eigen::Vector2d& direction) {
  if (!origin.allFinite() || !direction.allFinite() ||
      direction.norm() < 1e-12 || ring.size() < 3)
    return std::nullopt;
  const Eigen::Vector2d u = direction.normalized();
  ConvexOffsetCurve cv;
  const ConvexBuild kind = BuildConvexOffsetCurve(ring, ring[0], distance, &cv);
  if (kind == ConvexBuild::kDegenerate) return std::nullopt;
  if (kind == ConvexBuild::kOk) {
    const double best = ConvexRayHit(cv, origin, u);
    if (!std::isfinite(best)) return std::nullopt;
    return Eigen::Vector2d(origin + best * u);
  }
  std::vector<Eigen::Vector2d> curve;
  std::vector<double> cum;
  double total = 0.0;
  if (!BuildOffsetCurve(ring, ring[0], distance, &curve, &cum, &total))
    return std::nullopt;
  double best_t = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < curve.size(); ++i) {
    const Eigen::Vector2d a = curve[i], e = curve[(i + 1) % curve.size()] - a;
    // origin + t u = a + s e, t > 0, s in [0, 1].
    const double den = u.x() * (-e.y()) - u.y() * (-e.x());
    if (std::fabs(den) < 1e-14) continue;
    const Eigen::Vector2d w = a - origin;
    const double t = (w.x() * (-e.y()) - w.y() * (-e.x())) / den;
    const double s = (u.x() * w.y() - u.y() * w.x()) / den;
    if (t > 1e-12 && s >= -1e-9 && s <= 1.0 + 1e-9 && t < best_t) best_t = t;
  }
  if (!std::isfinite(best_t)) return std::nullopt;
  return Eigen::Vector2d(origin + best_t * u);
}

std::optional<Eigen::Vector2d> PolygonCoordinateMapper::ForwardHit(
    const Polygon2D& polygon, const Eigen::Vector2d& forward_direction) {
  if (const auto hit = polygon.RayBoundaryIntersection(
          ComputeCentroid(polygon), forward_direction))
    return hit;
  // A polygon the centroid ray misses (the centroid of a U lies in its open
  // cavity): the first vertex, as ResampleBoundary does.
  const auto& pts = polygon.GetPoints();
  if (pts.empty()) return std::nullopt;
  return pts.front();
}

absl::StatusOr<double> PolygonCoordinateMapper::EncodeOffsetFraction(
    const std::vector<Eigen::Vector2d>& ring, const Eigen::Vector2d& start_point,
    const Eigen::Vector2d& position, double distance) {
  if (!position.allFinite()) return absl::InvalidArgumentError("No offset curve");
  ConvexOffsetCurve cv;
  const ConvexBuild kind = BuildConvexOffsetCurve(ring, start_point, distance, &cv);
  if (kind == ConvexBuild::kDegenerate)
    return absl::InvalidArgumentError("No offset curve");
  if (kind == ConvexBuild::kOk) {
    double dist = 0.0, pos = 0.0;
    ConvexNearest(cv, position, &dist, &pos);
    double f = std::fmod(pos / cv.total, 1.0);
    if (f < 0.0) f += 1.0;
    return f;
  }
  std::vector<Eigen::Vector2d> curve;
  std::vector<double> cum;
  double total = 0.0;
  if (!BuildOffsetCurve(ring, start_point, distance, &curve, &cum, &total))
    return absl::InvalidArgumentError("No offset curve");
  double best = std::numeric_limits<double>::infinity(), pos = 0.0;
  for (size_t i = 0; i < curve.size(); ++i) {
    const Eigen::Vector2d a = curve[i], e = curve[(i + 1) % curve.size()] - a;
    const double l2 = e.squaredNorm();
    const double t = l2 < 1e-24 ? 0.0
                                : std::clamp((position - a).dot(e) / l2, 0.0, 1.0);
    const double dist = (position - (a + t * e)).norm();
    if (dist < best - 1e-12) {
      best = dist;
      pos = cum[i] + t * (cum[i + 1] - cum[i]);
    }
  }
  double f = std::fmod(pos / total, 1.0);
  if (f < 0.0) f += 1.0;
  return f;
}

absl::StatusOr<Eigen::Vector2d> PolygonCoordinateMapper::DecodeOffsetFraction(
    const std::vector<Eigen::Vector2d>& ring, const Eigen::Vector2d& start_point,
    double fraction, double distance) {
  if (!std::isfinite(fraction)) return absl::InvalidArgumentError("No offset curve");
  ConvexOffsetCurve cv;
  const ConvexBuild kind = BuildConvexOffsetCurve(ring, start_point, distance, &cv);
  if (kind == ConvexBuild::kDegenerate)
    return absl::InvalidArgumentError("No offset curve");
  if (kind == ConvexBuild::kOk) {
    double f = std::fmod(fraction, 1.0);
    if (f < 0.0) f += 1.0;
    const double s_nat = cv.FromChart(f * cv.total);
    for (size_t i = 0; i < cv.pieces.size(); ++i) {
      const auto& pc = cv.pieces[i];
      if (s_nat <= pc.s0 + pc.len || i + 1 == cv.pieces.size())
        return cv.At(pc, std::clamp(s_nat - pc.s0, 0.0, pc.len));
    }
  }
  std::vector<Eigen::Vector2d> curve;
  std::vector<double> cum;
  double total = 0.0;
  if (!BuildOffsetCurve(ring, start_point, distance, &curve, &cum, &total))
    return absl::InvalidArgumentError("No offset curve");
  double f = std::fmod(fraction, 1.0);
  if (f < 0.0) f += 1.0;
  const double target = f * total;
  for (size_t i = 0; i < curve.size(); ++i) {
    const double seg = cum[i + 1] - cum[i];
    if (target <= cum[i + 1] || i + 1 == curve.size()) {
      const double t = seg > 1e-12 ? std::clamp((target - cum[i]) / seg, 0.0, 1.0)
                                   : 0.0;
      const Eigen::Vector2d a = curve[i], b = curve[(i + 1) % curve.size()];
      return Eigen::Vector2d(a + t * (b - a));
    }
  }
  return curve.front();
}

absl::StatusOr<double> PolygonCoordinateMapper::EncodeRadialArcLength(
    const Eigen::Vector2d& point, const Polygon2D& polygon,
    const Eigen::Vector2d& forward_direction) {
  if (!point.allFinite()) return absl::InvalidArgumentError("Invalid point");
  const Eigen::Vector2d centroid = ComputeCentroid(polygon);
  const Eigen::Vector2d delta = point - centroid;
  if (delta.norm() < 1e-12) return 0.0;
  const auto hit = polygon.RayBoundaryIntersection(centroid, delta.normalized());
  if (!hit.has_value())
    return absl::InternalError("RayBoundaryIntersection failed from centroid");
  return EncodeBoundaryArcLength(*hit, polygon, forward_direction);
}

absl::StatusOr<Eigen::Vector2d> PolygonCoordinateMapper::DecodeStarArcLength(
    double arclen, double ratio, const Polygon2D& polygon,
    const Eigen::Vector2d& forward_direction) {
  if (!std::isfinite(ratio))
    return absl::InvalidArgumentError("Invalid radial ratio");
  auto boundary = DecodeBoundaryArcLength(arclen, polygon, forward_direction);
  if (!boundary.ok()) return boundary.status();
  const double r = std::clamp(ratio, 0.0, 1.0);
  const Eigen::Vector2d centroid = ComputeCentroid(polygon);
  Eigen::Vector2d p = centroid + r * (*boundary - centroid);
  if (r < 1.0 && !polygon.Contains(p, /*include_boundary=*/true))
    p = polygon.GetNearestPoint(p);
  return p;
}

absl::StatusOr<NormalizedPolygonPosition> PolygonCoordinateMapper::Encode(
    const Eigen::Vector2d& point, const Polygon2D& polygon,
    const Eigen::Vector2d& forward_direction, int num_samples) {
  auto canonical = CanonicalizeDomainPoint(point, polygon);
  if (!canonical.ok()) return canonical.status();
  if (!forward_direction.allFinite() || forward_direction.norm() < 1e-12) {
    return absl::InvalidArgumentError("Forward direction is zero-length");
  }

  if (polygon.IsConvex()) {
    auto encoded = EncodeRadial(*canonical, polygon, forward_direction);
    if (encoded.ok())
      encoded->radial_ratio = std::clamp(encoded->radial_ratio, 0.0, 1.0);
    return encoded;
  }
  return EncodeMVC(*canonical, polygon, forward_direction, num_samples);
}

absl::StatusOr<Eigen::Vector2d> PolygonCoordinateMapper::Decode(
    const NormalizedPolygonPosition& normalized, const Polygon2D& polygon,
    const Eigen::Vector2d& forward_direction, int num_samples) {
  if (!forward_direction.allFinite() || forward_direction.norm() < 1e-12) {
    return absl::InvalidArgumentError("Forward direction is zero-length");
  }

  if (normalized.strategy == NormalizedPolygonPosition::Strategy::kRadial) {
    return DecodeRadial(normalized, polygon, forward_direction);
  }
  return DecodeMVC(normalized, polygon, forward_direction, num_samples);
}

absl::StatusOr<Eigen::Vector2d> PolygonCoordinateMapper::MapPosition(
    const Eigen::Vector2d& point, const Polygon2D& source_polygon,
    const Eigen::Vector2d& source_forward, const Polygon2D& target_polygon,
    const Eigen::Vector2d& target_forward, int num_samples) {
  auto canonical = CanonicalizeDomainPoint(point, source_polygon);
  if (!canonical.ok()) return canonical.status();
  if (!source_forward.allFinite() || source_forward.norm() < 1e-12 ||
      !target_forward.allFinite() || target_forward.norm() < 1e-12) {
    return absl::InvalidArgumentError("Forward direction is zero-length");
  }

  // If either polygon is non-convex, use MVC for both
  bool use_mvc = !source_polygon.IsConvex() || !target_polygon.IsConvex();

  NormalizedPolygonPosition normalized;
  if (use_mvc) {
    auto encode_result =
        EncodeMVC(*canonical, source_polygon, source_forward, num_samples);
    if (!encode_result.ok()) return encode_result.status();
    normalized = std::move(encode_result.value());
    return DecodeMVC(normalized, target_polygon, target_forward, num_samples);
  } else {
    auto encode_result =
        EncodeRadial(*canonical, source_polygon, source_forward);
    if (!encode_result.ok()) return encode_result.status();
    encode_result->radial_ratio =
        std::clamp(encode_result->radial_ratio, 0.0, 1.0);
    normalized = std::move(encode_result.value());
    return DecodeRadial(normalized, target_polygon, target_forward);
  }
}


std::vector<Eigen::Vector2d> InsetRing(const std::vector<Eigen::Vector2d>& ring, double h) {
  const size_t n = ring.size();
  std::vector<Eigen::Vector2d> out(n, Eigen::Vector2d::Zero());
  if (n < 3) return ring;
  double area2 = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const Eigen::Vector2d &a = ring[i], &b = ring[(i + 1) % n];
    area2 += a.x() * b.y() - b.x() * a.y();
  }
  std::vector<Eigen::Vector2d> inward(n, Eigen::Vector2d::Zero());
  for (size_t i = 0; i < n; ++i) {
    const Eigen::Vector2d e = ring[(i + 1) % n] - ring[i];
    if (e.squaredNorm() < 1e-18) continue;
    const Eigen::Vector2d nl = Eigen::Vector2d(-e.y(), e.x()).normalized();
    inward[i] = area2 > 0.0 ? nl : Eigen::Vector2d(-nl);
  }
  for (size_t j = 0; j < n; ++j) {
    const Eigen::Vector2d& n_prev = inward[(j + n - 1) % n];
    const Eigen::Vector2d& n_next = inward[j];
    const double denom = 1.0 + n_prev.dot(n_next);
    const Eigen::Vector2d m =
        denom > 1e-9 ? Eigen::Vector2d((n_prev + n_next) / denom) : n_next;
    out[j] = ring[j] + h * m;
  }
  return out;
}

std::optional<Eigen::Vector2d> BoundaryPointOfInset(const std::vector<Eigen::Vector2d>& ring,
                                                    const Eigen::Vector2d& p) {
  const size_t n = ring.size();
  if (n < 3) return std::nullopt;
  double best = std::numeric_limits<double>::infinity();
  size_t best_k = 0;
  bool best_is_vertex = false;
  for (size_t k = 0; k < n; ++k) {
    const Eigen::Vector2d &a = ring[k], &b = ring[(k + 1) % n];
    const Eigen::Vector2d e = b - a;
    const double len2 = e.squaredNorm();
    if (len2 < 1e-18) continue;
    const double t = (p - a).dot(e) / len2;
    const double d = (p - (a + std::clamp(t, 0.0, 1.0) * e)).norm();
    if (d < best) {
      best = d;
      best_k = k;
      best_is_vertex = t <= 0.0 || t >= 1.0;
    }
  }
  if (!std::isfinite(best)) return std::nullopt;
  const std::vector<Eigen::Vector2d> inset = InsetRing(ring, best);
  const Eigen::Vector2d &v0 = inset[best_k], &v1 = inset[(best_k + 1) % n];
  const Eigen::Vector2d e = v1 - v0;
  const double len2 = e.squaredNorm();
  if (len2 < 1e-18) return std::nullopt;
  const double t = (p - v0).dot(e) / len2;
  // A closest feature that is a vertex is a mitered vertex of the inset (a convex corner
  // seen from inside: t = 0 or 1) -- or a reflex vertex, where the inset edge does not
  // carry p (t outside [0, 1] by more than round-off).
  if (t < -1e-9 || t > 1.0 + 1e-9) return std::nullopt;
  (void)best_is_vertex;
  const double tc = std::clamp(t, 0.0, 1.0);
  return ring[best_k] + tc * (ring[(best_k + 1) % n] - ring[best_k]);
}
}  // namespace spatialblend
