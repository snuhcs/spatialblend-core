// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/convex_decomposition.cc
#include "spatialblend/geometry/convex_decomposition.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "CDT.h"

namespace spatialblend {

namespace {

// Cross product of vectors (b-a) x (c-a)
double Cross2D(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
               const Eigen::Vector2d& c) {
  return (b.x() - a.x()) * (c.y() - a.y()) -
         (b.y() - a.y()) * (c.x() - a.x());
}

// Check if polygon vertices form a convex polygon (all cross products same sign)
bool IsConvexPolygon(const std::vector<Eigen::Vector2d>& pts) {
  int n = static_cast<int>(pts.size());
  if (n < 3) return false;
  if (n == 3) return true;

  int sign = 0;
  for (int i = 0; i < n; ++i) {
    double cross = Cross2D(pts[i], pts[(i + 1) % n], pts[(i + 2) % n]);
    if (cross != 0.0) {
      int s = (cross > 0) ? 1 : -1;
      if (sign == 0) {
        sign = s;
      } else if (sign != s) {
        return false;
      }
    }
  }
  return true;
}

// Centroid of a triangle
Eigen::Vector2d TriangleCentroid(const Eigen::Vector2d& a,
                                  const Eigen::Vector2d& b,
                                  const Eigen::Vector2d& c) {
  return (a + b + c) / 3.0;
}

// Polygon signed area (positive = counter-clockwise)
double SignedArea(const std::vector<Eigen::Vector2d>& pts) {
  double area = 0.0;
  int n = static_cast<int>(pts.size());
  for (int i = 0; i < n; ++i) {
    const auto& a = pts[i];
    const auto& b = pts[(i + 1) % n];
    area += (a.x() * b.y() - b.x() * a.y());
  }
  return area / 2.0;
}

// Check if merging cell_a and cell_b (which share an edge) produces a convex
// polygon. Returns the merged vertex list if convex, empty otherwise.
std::vector<Eigen::Vector2d> TryMergeCells(
    const std::vector<Eigen::Vector2d>& a_pts,
    const std::vector<Eigen::Vector2d>& b_pts) {
  int n_a = static_cast<int>(a_pts.size());
  int n_b = static_cast<int>(b_pts.size());
  const double kEps = 1e-9;

  // Find the shared edge: two consecutive vertices in a that both appear in b
  int shared_a_start = -1, shared_a_end = -1;

  for (int i = 0; i < n_a; ++i) {
    int j = (i + 1) % n_a;
    int bi = -1, bj = -1;
    for (int k = 0; k < n_b; ++k) {
      if ((a_pts[i] - b_pts[k]).squaredNorm() < kEps) bi = k;
      if ((a_pts[j] - b_pts[k]).squaredNorm() < kEps) bj = k;
    }
    if (bi >= 0 && bj >= 0) {
      shared_a_start = i;
      shared_a_end = j;
      break;
    }
  }

  if (shared_a_start < 0) return {};  // No shared edge found

  // Find the corresponding indices in b
  // shared_a_end == shared_b_start, shared_a_start == shared_b_end
  int shared_b_end = -1;    // corresponds to a_pts[shared_a_end]
  int shared_b_start = -1;  // corresponds to a_pts[shared_a_start]
  for (int k = 0; k < n_b; ++k) {
    if ((a_pts[shared_a_end] - b_pts[k]).squaredNorm() < kEps)
      shared_b_end = k;
    if ((a_pts[shared_a_start] - b_pts[k]).squaredNorm() < kEps)
      shared_b_start = k;
  }
  if (shared_b_end < 0 || shared_b_start < 0) return {};

  // Build merged polygon:
  // Traverse a from shared_a_end for n_a-1 steps (excludes shared_a_start).
  // Then traverse b from shared_b_start for n_b-1 steps (excludes shared_b_end).
  std::vector<Eigen::Vector2d> merged;
  merged.reserve(n_a + n_b - 2);

  for (int step = 0; step < n_a - 1; ++step) {
    int idx = (shared_a_end + step) % n_a;
    merged.push_back(a_pts[idx]);
  }
  for (int step = 0; step < n_b - 1; ++step) {
    int idx = (shared_b_start + step) % n_b;
    merged.push_back(b_pts[idx]);
  }

  if (static_cast<int>(merged.size()) < 3) return {};
  if (IsConvexPolygon(merged)) return merged;
  return {};
}

}  // namespace

ConvexCellPartition ConvexCellPartition::Build(const Polygon2D& polygon) {
  ConvexCellPartition result;
  result.source_polygon = std::make_shared<Polygon2D>(polygon);

  const auto& points = polygon.GetPoints();
  int n = static_cast<int>(points.size());
  if (n < 3) return result;

  // Step 1: CDT triangulation
  CDT::Triangulation<double> cdt;

  // Deduplicate coincident vertices before inserting. Input polygons are often
  // CLOSED rings (last vertex == first) and may carry other near-coincident
  // vertices; CDT throws a "duplicate vertex" error if two inserted vertices
  // collapse to the same location. Collapse within kDuplicateTol and remap the
  // boundary edges onto the unique indices (matching visibility_graph.cc).
  constexpr double kDuplicateTol = 1e-9;
  std::vector<CDT::V2d<double>> cdt_vertices;
  cdt_vertices.reserve(n);
  std::vector<int> remap(n, -1);  // original index -> unique vertex index
  for (int i = 0; i < n; ++i) {
    const auto& p = points[i];
    int found = -1;
    for (int k = 0; k < static_cast<int>(cdt_vertices.size()); ++k) {
      double dx = cdt_vertices[k].x - p.x();
      double dy = cdt_vertices[k].y - p.y();
      if (dx * dx + dy * dy < kDuplicateTol * kDuplicateTol) {
        found = k;
        break;
      }
    }
    if (found < 0) {
      found = static_cast<int>(cdt_vertices.size());
      cdt_vertices.push_back(CDT::V2d<double>(p.x(), p.y()));
    }
    remap[i] = found;
  }

  // Need at least a triangle's worth of distinct vertices.
  if (cdt_vertices.size() < 3) {
    ConvexCell cell;
    cell.polygon = polygon;
    cell.index = 0;
    result.cells.push_back(std::move(cell));
    result.adjacency.resize(1);
    return result;
  }

  cdt.insertVertices(cdt_vertices);

  // Insert constrained edges (polygon boundary), remapped onto unique vertices
  // and skipping degenerate edges whose endpoints collapsed together.
  std::vector<CDT::Edge> edges;
  edges.reserve(n);
  for (int i = 0; i < n; ++i) {
    int a = remap[i];
    int b = remap[(i + 1) % n];
    if (a == b) continue;
    edges.push_back(CDT::Edge(static_cast<CDT::VertInd>(a),
                              static_cast<CDT::VertInd>(b)));
  }
  cdt.insertEdges(edges);
  cdt.eraseOuterTriangles();

  // Step 2: Collect triangles inside the polygon
  // (eraseOuterTriangles already removed exterior triangles)
  struct TriCell {
    std::vector<Eigen::Vector2d> pts;
  };

  std::vector<TriCell> tri_cells;
  const auto& cdt_tris = cdt.triangles;
  const auto& cdt_verts = cdt.vertices;

  for (const auto& tri : cdt_tris) {
    CDT::VertInd i0 = tri.vertices[0];
    CDT::VertInd i1 = tri.vertices[1];
    CDT::VertInd i2 = tri.vertices[2];

    Eigen::Vector2d p0(cdt_verts[i0].x, cdt_verts[i0].y);
    Eigen::Vector2d p1(cdt_verts[i1].x, cdt_verts[i1].y);
    Eigen::Vector2d p2(cdt_verts[i2].x, cdt_verts[i2].y);

    // Filter: centroid must be strictly inside the polygon
    Eigen::Vector2d centroid = TriangleCentroid(p0, p1, p2);
    if (!polygon.Contains(centroid, false)) continue;

    // Ensure CCW order
    if (Cross2D(p0, p1, p2) < 0) std::swap(p1, p2);

    TriCell cell;
    cell.pts = {p0, p1, p2};
    tri_cells.push_back(std::move(cell));
  }

  if (tri_cells.empty()) {
    // Fallback: return the entire polygon as one cell
    ConvexCell cell;
    cell.polygon = polygon;
    cell.index = 0;
    result.cells.push_back(std::move(cell));
    result.adjacency.resize(1);
    return result;
  }

  // Step 3: Hertel-Mehlhorn greedy merge
  std::vector<std::vector<Eigen::Vector2d>> cell_pts;
  cell_pts.reserve(tri_cells.size());
  for (auto& tc : tri_cells) {
    cell_pts.push_back(std::move(tc.pts));
  }

  bool merged_any = true;
  while (merged_any) {
    merged_any = false;
    for (int i = 0; i < static_cast<int>(cell_pts.size()); ++i) {
      for (int j = i + 1; j < static_cast<int>(cell_pts.size()); ++j) {
        auto merged = TryMergeCells(cell_pts[i], cell_pts[j]);
        if (!merged.empty()) {
          cell_pts[i] = std::move(merged);
          cell_pts.erase(cell_pts.begin() + j);
          merged_any = true;
          break;
        }
      }
      if (merged_any) break;
    }
  }

  // Step 4: Build ConvexCell objects with correct winding
  result.cells.reserve(cell_pts.size());
  for (size_t i = 0; i < cell_pts.size(); ++i) {
    ConvexCell cell;
    // Ensure CCW winding (positive area)
    if (SignedArea(cell_pts[i]) < 0) {
      std::reverse(cell_pts[i].begin(), cell_pts[i].end());
    }
    cell.polygon = Polygon2D(cell_pts[i]);
    cell.index = i;
    result.cells.push_back(std::move(cell));
  }

  // Step 5: Build adjacency from shared edges
  int nc = static_cast<int>(result.cells.size());
  result.adjacency.resize(nc);

  const double kEps = 1e-9;
  for (int i = 0; i < nc; ++i) {
    const auto& pi = result.cells[i].polygon.GetPoints();
    int ni = static_cast<int>(pi.size());
    for (int j = i + 1; j < nc; ++j) {
      const auto& pj = result.cells[j].polygon.GetPoints();
      int nj = static_cast<int>(pj.size());
      bool adjacent = false;
      for (int ei = 0; ei < ni && !adjacent; ++ei) {
        const auto& a = pi[ei];
        const auto& b = pi[(ei + 1) % ni];
        for (int ej = 0; ej < nj && !adjacent; ++ej) {
          const auto& c = pj[ej];
          const auto& d = pj[(ej + 1) % nj];
          if (((a - c).squaredNorm() < kEps && (b - d).squaredNorm() < kEps) ||
              ((a - d).squaredNorm() < kEps && (b - c).squaredNorm() < kEps)) {
            adjacent = true;
          }
        }
      }
      if (adjacent) {
        result.adjacency[i].push_back(j);
        result.adjacency[j].push_back(i);
      }
    }
  }

  return result;
}

int ConvexCellPartition::FindCellForPoint(const Eigen::Vector2d& point) const {
  for (const auto& cell : cells) {
    if (cell.polygon.Contains(point)) {
      return static_cast<int>(cell.index);
    }
  }
  return -1;
}

}  // namespace spatialblend
