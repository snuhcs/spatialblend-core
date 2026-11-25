// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/visibility_graph.cc
#include "spatialblend/geometry/visibility_graph.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace spatialblend {

VisibilityGraph VisibilityGraph::Build(const Polygon2D& polygon) {
  VisibilityGraph graph;
  graph.polygon = std::make_shared<Polygon2D>(polygon);

  const auto& points = polygon.GetPoints();
  if (points.size() < 3) {
    return graph;
  }

  // Deduplicate vertices
  constexpr double kDuplicateTol = 1e-9;
  for (const auto& pt : points) {
    if (!std::isfinite(pt.x()) || !std::isfinite(pt.y())) {
      continue;
    }
    bool dup = false;
    for (const auto& existing : graph.vertices) {
      if ((existing - pt).norm() < kDuplicateTol) {
        dup = true;
        break;
      }
    }
    if (!dup) {
      graph.vertices.push_back(pt);
    }
  }

  const size_t n = graph.vertices.size();
  graph.adjacency.resize(n);

  // Build a set of polygon edges for fast lookup.
  // Polygon edges (adjacent vertex pairs in points_) are always visible and
  // must NOT be blocked by IntersectsSegment even though that function may
  // return true for coincident segments.
  const auto& raw_pts = polygon.GetPoints();
  const size_t np = raw_pts.size();

  // Helper: check if (vi, vj) is a polygon boundary edge.
  auto is_polygon_edge = [&](const Eigen::Vector2d& vi,
                              const Eigen::Vector2d& vj) -> bool {
    for (size_t k = 0; k < np; ++k) {
      const Eigen::Vector2d& a = raw_pts[k];
      const Eigen::Vector2d& b = raw_pts[(k + 1) % np];
      if (((a - vi).norm() < 1e-9 && (b - vj).norm() < 1e-9) ||
          ((a - vj).norm() < 1e-9 && (b - vi).norm() < 1e-9)) {
        return true;
      }
    }
    return false;
  };

  // For each pair of vertices, check line-of-sight:
  // A segment has clear LOS if:
  //   (a) It is a polygon boundary edge (always visible), OR
  //   (b) It does not cross the polygon boundary AND its midpoint is inside.
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      const Eigen::Vector2d& vi = graph.vertices[i];
      const Eigen::Vector2d& vj = graph.vertices[j];

      bool visible = false;
      if (is_polygon_edge(vi, vj)) {
        // Polygon boundary edges are always mutually visible.
        visible = true;
      } else {
        // For non-edge pairs: use a slightly shrunk segment to avoid false
        // positives when endpoints are polygon vertices sitting exactly on the
        // boundary (which makes IntersectsSegment return true even for valid
        // internal diagonals).
        Eigen::Vector2d dir = vj - vi;
        double seg_len = dir.norm();
        if (seg_len < 1e-12) {
          visible = false;
        } else {
          constexpr double kEps = 1e-6;
          double shrink = std::min(kEps, seg_len * 0.01);
          Eigen::Vector2d unit = dir / seg_len;
          Eigen::Vector2d vis = vi + unit * shrink;
          Eigen::Vector2d vjs = vj - unit * shrink;

          if (!polygon.IntersectsSegment(vis, vjs)) {
            // Midpoint must be inside the polygon (handles concave regions).
            Eigen::Vector2d mid = 0.5 * (vi + vj);
            visible = polygon.Contains(mid, /*include_boundary=*/true);
          }
        }
      }

      if (visible) {
        graph.adjacency[i].push_back(j);
        graph.adjacency[j].push_back(i);
      }
    }
  }

  return graph;
}

}  // namespace spatialblend
