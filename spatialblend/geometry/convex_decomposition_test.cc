// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/convex_decomposition_test.cc
#include "spatialblend/geometry/convex_decomposition.h"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {
namespace test {

namespace {

// Helper: check that a polygon is convex (CCW winding assumed)
bool IsConvex(const Polygon2D& poly) {
  const auto& pts = poly.GetPoints();
  int n = static_cast<int>(pts.size());
  if (n < 3) return false;
  if (n == 3) return true;
  int sign = 0;
  for (int i = 0; i < n; ++i) {
    const auto& a = pts[i];
    const auto& b = pts[(i + 1) % n];
    const auto& c = pts[(i + 2) % n];
    double cross =
        (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
    if (std::abs(cross) > 1e-9) {
      int s = (cross > 0) ? 1 : -1;
      if (sign == 0)
        sign = s;
      else if (sign != s)
        return false;
    }
  }
  return true;
}

// Helper: check that a set of cells covers the given polygon
// by sampling a grid of points inside the polygon and checking each is in
// some cell.
bool CellsCoverPolygon(const ConvexCellPartition& partition,
                       const Polygon2D& polygon) {
  // Use the bounding box of the polygon to generate test points
  const auto& pts = polygon.GetPoints();
  double min_x = pts[0].x(), max_x = pts[0].x();
  double min_y = pts[0].y(), max_y = pts[0].y();
  for (const auto& p : pts) {
    min_x = std::min(min_x, p.x());
    max_x = std::max(max_x, p.x());
    min_y = std::min(min_y, p.y());
    max_y = std::max(max_y, p.y());
  }

  int samples = 10;
  int covered = 0, total_inside = 0;
  for (int i = 0; i <= samples; ++i) {
    for (int j = 0; j <= samples; ++j) {
      double x = min_x + (max_x - min_x) * i / samples;
      double y = min_y + (max_y - min_y) * j / samples;
      Eigen::Vector2d p(x, y);
      if (polygon.Contains(p, false)) {
        ++total_inside;
        if (partition.FindCellForPoint(p) >= 0) {
          ++covered;
        }
      }
    }
  }
  return (total_inside == 0) || (covered == total_inside);
}

}  // namespace

TEST(ConvexDecompositionTest, Triangle) {
  // A triangle is already convex — should produce exactly one cell (or two
  // triangles at most).
  Polygon2D poly({
      {0.0, 0.0},
      {4.0, 0.0},
      {2.0, 3.0},
  });
  auto partition = ConvexCellPartition::Build(poly);
  EXPECT_GE(partition.cells.size(), 1u);
  // All cells should be convex
  for (const auto& cell : partition.cells) {
    EXPECT_TRUE(IsConvex(cell.polygon))
        << "Cell " << cell.index << " is not convex";
  }
}

TEST(ConvexDecompositionTest, Square) {
  Polygon2D poly({
      {0.0, 0.0},
      {4.0, 0.0},
      {4.0, 4.0},
      {0.0, 4.0},
  });
  auto partition = ConvexCellPartition::Build(poly);
  EXPECT_GE(partition.cells.size(), 1u);
  for (const auto& cell : partition.cells) {
    EXPECT_TRUE(IsConvex(cell.polygon))
        << "Cell " << cell.index << " is not convex";
  }
}

TEST(ConvexDecompositionTest, ClosedRingWithDuplicateVertex) {
  // Real preset walkable rings are CLOSED (last vertex repeats the first).
  // The raw duplicate previously made CDT throw "duplicate vertex"; Build must
  // dedup it and still produce a valid covering triangulation.
  Polygon2D poly({
      {0.0, 0.0},
      {4.0, 0.0},
      {4.0, 4.0},
      {0.0, 4.0},
      {0.0, 0.0},  // closing duplicate of vertex 0
  });
  auto partition = ConvexCellPartition::Build(poly);
  EXPECT_GE(partition.cells.size(), 1u);
  for (const auto& cell : partition.cells) {
    EXPECT_TRUE(IsConvex(cell.polygon))
        << "Cell " << cell.index << " is not convex";
  }
  EXPECT_TRUE(CellsCoverPolygon(partition, poly));
}

TEST(ConvexDecompositionTest, NearCoincidentVertex) {
  // A vertex within kDuplicateTol of another must not crash CDT.
  Polygon2D poly({
      {0.0, 0.0},
      {4.0, 0.0},
      {4.0, 4.0},
      {0.0, 4.0},
      {1e-12, 1e-12},  // effectively coincident with vertex 0
  });
  auto partition = ConvexCellPartition::Build(poly);
  EXPECT_GE(partition.cells.size(), 1u);
  for (const auto& cell : partition.cells) {
    EXPECT_TRUE(IsConvex(cell.polygon))
        << "Cell " << cell.index << " is not convex";
  }
}

TEST(ConvexDecompositionTest, AllCellsConvex) {
  // L-shape polygon (non-convex)
  Polygon2D poly({
      {0.0, 0.0},
      {4.0, 0.0},
      {4.0, 2.0},
      {2.0, 2.0},
      {2.0, 4.0},
      {0.0, 4.0},
  });
  auto partition = ConvexCellPartition::Build(poly);
  ASSERT_GE(partition.cells.size(), 1u);
  for (const auto& cell : partition.cells) {
    EXPECT_TRUE(IsConvex(cell.polygon))
        << "Cell " << cell.index << " is not convex";
  }
}

TEST(ConvexDecompositionTest, LShapeMultipleCells) {
  // L-shape should require at least 2 cells
  Polygon2D poly({
      {0.0, 0.0},
      {4.0, 0.0},
      {4.0, 2.0},
      {2.0, 2.0},
      {2.0, 4.0},
      {0.0, 4.0},
  });
  auto partition = ConvexCellPartition::Build(poly);
  EXPECT_GE(partition.cells.size(), 2u);
}

TEST(ConvexDecompositionTest, CoverageSquare) {
  Polygon2D poly({
      {0.0, 0.0},
      {4.0, 0.0},
      {4.0, 4.0},
      {0.0, 4.0},
  });
  auto partition = ConvexCellPartition::Build(poly);
  EXPECT_TRUE(CellsCoverPolygon(partition, poly));
}

TEST(ConvexDecompositionTest, CoverageLShape) {
  Polygon2D poly({
      {0.0, 0.0},
      {4.0, 0.0},
      {4.0, 2.0},
      {2.0, 2.0},
      {2.0, 4.0},
      {0.0, 4.0},
  });
  auto partition = ConvexCellPartition::Build(poly);
  EXPECT_TRUE(CellsCoverPolygon(partition, poly));
}

TEST(ConvexDecompositionTest, FindCellForPointInside) {
  Polygon2D poly({
      {0.0, 0.0},
      {4.0, 0.0},
      {4.0, 4.0},
      {0.0, 4.0},
  });
  auto partition = ConvexCellPartition::Build(poly);
  Eigen::Vector2d interior(2.0, 2.0);
  int cell_idx = partition.FindCellForPoint(interior);
  EXPECT_GE(cell_idx, 0);
}

TEST(ConvexDecompositionTest, FindCellForPointOutside) {
  Polygon2D poly({
      {0.0, 0.0},
      {4.0, 0.0},
      {4.0, 4.0},
      {0.0, 4.0},
  });
  auto partition = ConvexCellPartition::Build(poly);
  Eigen::Vector2d outside(10.0, 10.0);
  int cell_idx = partition.FindCellForPoint(outside);
  EXPECT_EQ(cell_idx, -1);
}

TEST(ConvexDecompositionTest, AdjacentCellsShareEdge) {
  // For L-shape with multiple cells, adjacent cells should share an edge
  Polygon2D poly({
      {0.0, 0.0},
      {4.0, 0.0},
      {4.0, 2.0},
      {2.0, 2.0},
      {2.0, 4.0},
      {0.0, 4.0},
  });
  auto partition = ConvexCellPartition::Build(poly);
  ASSERT_GE(partition.cells.size(), 2u);

  // Check that each adjacency pair shares at least two vertices
  const double kEps = 1e-9;
  for (size_t i = 0; i < partition.cells.size(); ++i) {
    for (size_t j : partition.adjacency[i]) {
      if (j <= i) continue;  // Check each pair once
      const auto& pi = partition.cells[i].polygon.GetPoints();
      const auto& pj = partition.cells[j].polygon.GetPoints();
      int shared = 0;
      for (const auto& vi : pi) {
        for (const auto& vj : pj) {
          if ((vi - vj).squaredNorm() < kEps) ++shared;
        }
      }
      EXPECT_GE(shared, 2)
          << "Adjacent cells " << i << " and " << j
          << " share fewer than 2 vertices";
    }
  }
}

TEST(ConvexDecompositionTest, IndicesMatchPosition) {
  Polygon2D poly({
      {0.0, 0.0},
      {4.0, 0.0},
      {4.0, 4.0},
      {0.0, 4.0},
  });
  auto partition = ConvexCellPartition::Build(poly);
  for (size_t i = 0; i < partition.cells.size(); ++i) {
    EXPECT_EQ(partition.cells[i].index, i);
  }
}

TEST(ConvexDecompositionTest, FindCellForPointLShape) {
  Polygon2D poly({
      {0.0, 0.0},
      {4.0, 0.0},
      {4.0, 2.0},
      {2.0, 2.0},
      {2.0, 4.0},
      {0.0, 4.0},
  });
  auto partition = ConvexCellPartition::Build(poly);

  // Point in bottom-right area
  EXPECT_GE(partition.FindCellForPoint({3.0, 1.0}), 0);
  // Point in top-left area
  EXPECT_GE(partition.FindCellForPoint({1.0, 3.0}), 0);
  // Point outside
  EXPECT_EQ(partition.FindCellForPoint({5.0, 5.0}), -1);
  EXPECT_EQ(partition.FindCellForPoint({3.0, 3.0}), -1);
}

}  // namespace test
}  // namespace spatialblend
