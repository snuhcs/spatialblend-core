// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/geometry/planar_frame_chart.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "absl/status/status.h"

namespace spatialblend {

absl::StatusOr<SurfaceChartKind> SurfaceChartKindFromInt(int32_t v) {
  switch (v) {
    case 0: return SurfaceChartKind::kRadial;
    case 1: return SurfaceChartKind::kFrameStretch;
    case 2: return SurfaceChartKind::kFrameContain;
    default: return absl::InvalidArgumentError("unknown surface chart kind");
  }
}

absl::StatusOr<FrameExtent> PlanarFrameChart::ExtentOf(const Polygon2D& boundary) {
  std::vector<Eigen::Vector2d> pts = boundary.GetPoints();
  if (pts.size() >= 2 && (pts.front() - pts.back()).norm() == 0.0) pts.pop_back();  // closed ring
  if (pts.size() != 4) return absl::InvalidArgumentError("frame chart needs a 4-vertex rectangle");
  FrameExtent e;
  e.x0 = e.y0 = std::numeric_limits<double>::infinity();
  e.x1 = e.y1 = -std::numeric_limits<double>::infinity();
  for (const auto& p : pts) {
    if (!p.allFinite()) return absl::InvalidArgumentError("non-finite rectangle vertex");
    e.x0 = std::min(e.x0, p.x());
    e.x1 = std::max(e.x1, p.x());
    e.y0 = std::min(e.y0, p.y());
    e.y1 = std::max(e.y1, p.y());
  }
  const double w = e.Width(), h = e.Height();
  if (!(w > 0.0) || !(h > 0.0)) return absl::InvalidArgumentError("degenerate rectangle");
  // Every vertex a distinct corner of the bounding box: an axis-aligned rectangle, in any vertex order. The tolerance
  // is rounding of the authoring frame (1e-9 of the extent), not a design threshold.
  const double tol = 1e-9 * std::max(w, h);
  bool corner[4] = {false, false, false, false};
  for (const auto& p : pts) {
    const bool lx = std::abs(p.x() - e.x0) <= tol, hx = std::abs(p.x() - e.x1) <= tol;
    const bool ly = std::abs(p.y() - e.y0) <= tol, hy = std::abs(p.y() - e.y1) <= tol;
    if (!(lx || hx) || !(ly || hy)) return absl::InvalidArgumentError("part boundary is not an axis-aligned rectangle");
    corner[(hx ? 1 : 0) + (hy ? 2 : 0)] = true;
  }
  if (!(corner[0] && corner[1] && corner[2] && corner[3]))
    return absl::InvalidArgumentError("part boundary is not an axis-aligned rectangle");
  // A ring, not a bow-tie: consecutive vertices share exactly one coordinate (every edge axis-aligned).
  for (size_t i = 0; i < 4; ++i) {
    const Eigen::Vector2d d = pts[(i + 1) % 4] - pts[i];
    if ((std::abs(d.x()) <= tol) == (std::abs(d.y()) <= tol))
      return absl::InvalidArgumentError("part boundary is not an axis-aligned rectangle ring");
  }
  return e;
}

absl::StatusOr<FrameCoord> PlanarFrameChart::Encode(const Eigen::Vector2d& point, double normal_m,
                                                    const Polygon2D& part) {
  if (!point.allFinite() || !std::isfinite(normal_m)) return absl::InvalidArgumentError("non-finite point");
  auto e = ExtentOf(part);
  if (!e.ok()) return e.status();
  FrameCoord c;
  c.u = (point.x() - e->x0) / e->Width();
  c.v = (point.y() - e->y0) / e->Height();
  c.normal_m = normal_m;
  c.src_width_m = e->Width();
  c.src_height_m = e->Height();
  return c;
}

absl::StatusOr<Eigen::Vector3d> PlanarFrameChart::Decode(const FrameCoord& c, const Polygon2D& target,
                                                         SurfaceChartKind kind) {
  if (!IsFrameChart(kind)) return absl::InvalidArgumentError("not a frame chart");
  if (!std::isfinite(c.u) || !std::isfinite(c.v) || !std::isfinite(c.normal_m) || !(c.src_width_m > 0.0) ||
      !(c.src_height_m > 0.0) || !std::isfinite(c.src_width_m) || !std::isfinite(c.src_height_m))
    return absl::InvalidArgumentError("invalid frame coordinate");
  auto e = ExtentOf(target);
  if (!e.ok()) return e.status();
  const double cx = 0.5 * (e->x0 + e->x1), cy = 0.5 * (e->y0 + e->y1);
  if (kind == SurfaceChartKind::kFrameStretch)
    return Eigen::Vector3d(e->x0 + c.u * e->Width(), e->y0 + c.v * e->Height(), c.normal_m);
  const double s = std::min(e->Width() / c.src_width_m, e->Height() / c.src_height_m);
  return Eigen::Vector3d(cx + (c.u - 0.5) * c.src_width_m * s, cy + (c.v - 0.5) * c.src_height_m * s, c.normal_m);
}

absl::StatusOr<Eigen::Vector3d> PlanarFrameChart::Map(const Eigen::Vector2d& point, double normal_m,
                                                      const Polygon2D& source, const Polygon2D& target,
                                                      SurfaceChartKind kind) {
  auto c = Encode(point, normal_m, source);
  if (!c.ok()) return c.status();
  return Decode(*c, target, kind);
}

}  // namespace spatialblend
