// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/visibility_graph_test.cc
#include "spatialblend/geometry/visibility_graph.h"

#include <gtest/gtest.h>

#include <vector>

#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {
namespace test {

using Eigen::Vector2d;

// ===== Helper polygons =====

std::vector<Vector2d> MakeSquare() {
  return {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
}

// L-shaped polygon (non-convex): concave at the inner corner (5,5)
std::vector<Vector2d> MakeLShape() {
  return {{0, 0}, {10, 0}, {10, 5}, {5, 5}, {5, 10}, {0, 10}};
}

std::vector<Vector2d> MakeTriangle() {
  return {{0, 0}, {10, 0}, {5, 10}};
}

// ===== Tests =====

TEST(VisibilityGraphTest, SquareHasCorrectVertexCount) {
  Polygon2D poly(MakeSquare());
  auto graph = VisibilityGraph::Build(poly);

  EXPECT_TRUE(graph.IsValid());
  EXPECT_EQ(graph.vertices.size(), 4u);
  EXPECT_EQ(graph.adjacency.size(), 4u);
}

TEST(VisibilityGraphTest, TriangleHasCorrectVertexCount) {
  Polygon2D poly(MakeTriangle());
  auto graph = VisibilityGraph::Build(poly);

  EXPECT_TRUE(graph.IsValid());
  EXPECT_EQ(graph.vertices.size(), 3u);
}

TEST(VisibilityGraphTest, SquareAllVerticesMutuallyVisible) {
  // In a convex polygon (square), every pair of vertices should be visible.
  Polygon2D poly(MakeSquare());
  auto graph = VisibilityGraph::Build(poly);

  ASSERT_TRUE(graph.IsValid());
  const size_t n = graph.vertices.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      // Check both directions
      bool ij_found = false;
      bool ji_found = false;
      for (size_t nb : graph.adjacency[i]) {
        if (nb == j) ij_found = true;
      }
      for (size_t nb : graph.adjacency[j]) {
        if (nb == i) ji_found = true;
      }
      EXPECT_TRUE(ij_found) << "vertex " << i << " should see vertex " << j;
      EXPECT_TRUE(ji_found) << "vertex " << j << " should see vertex " << i;
    }
  }
}

TEST(VisibilityGraphTest, TriangleAllVerticesVisible) {
  Polygon2D poly(MakeTriangle());
  auto graph = VisibilityGraph::Build(poly);

  ASSERT_TRUE(graph.IsValid());
  ASSERT_EQ(graph.vertices.size(), 3u);

  // All pairs visible in triangle
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i + 1; j < 3; ++j) {
      bool ij_found = false;
      for (size_t nb : graph.adjacency[i]) {
        if (nb == j) ij_found = true;
      }
      EXPECT_TRUE(ij_found) << "triangle vertex " << i << " should see " << j;
    }
  }
}

TEST(VisibilityGraphTest, LShapeVertexCountIsCorrect) {
  Polygon2D poly(MakeLShape());
  auto graph = VisibilityGraph::Build(poly);

  EXPECT_TRUE(graph.IsValid());
  EXPECT_EQ(graph.vertices.size(), 6u);
}

TEST(VisibilityGraphTest, LShapeConcaveCornerBlocksVisibility) {
  // The L-shape polygon is {0,0},{10,0},{10,5},{5,5},{5,10},{0,10}.
  // The vertex (5,10) at the top of the inner notch and the vertex (10,5)
  // at the right of the inner notch: the segment between them has midpoint
  // (7.5, 7.5), which is OUTSIDE the L (x>5 and y>5). So they should NOT
  // be mutually visible.
  Polygon2D poly(MakeLShape());
  auto graph = VisibilityGraph::Build(poly);

  ASSERT_TRUE(graph.IsValid());

  // Find indices of (5,10) and (10,5)
  size_t idx_5_10 = SIZE_MAX, idx_10_5 = SIZE_MAX;
  for (size_t i = 0; i < graph.vertices.size(); ++i) {
    if ((graph.vertices[i] - Vector2d(5, 10)).norm() < 1e-6) idx_5_10 = i;
    if ((graph.vertices[i] - Vector2d(10, 5)).norm() < 1e-6) idx_10_5 = i;
  }
  ASSERT_NE(idx_5_10, SIZE_MAX) << "vertex (5,10) should exist";
  ASSERT_NE(idx_10_5, SIZE_MAX) << "vertex (10,5) should exist";

  // Segment from (5,10) to (10,5) passes through (7.5, 7.5) which is outside.
  bool found = false;
  for (size_t nb : graph.adjacency[idx_5_10]) {
    if (nb == idx_10_5) found = true;
  }
  EXPECT_FALSE(found) << "(5,10) and (10,5) should NOT be mutually visible "
                         "because their midpoint is outside the L-shape";
}

TEST(VisibilityGraphTest, AdjacencyIsSymmetric) {
  Polygon2D poly(MakeLShape());
  auto graph = VisibilityGraph::Build(poly);

  ASSERT_TRUE(graph.IsValid());
  const size_t n = graph.vertices.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j : graph.adjacency[i]) {
      ASSERT_LT(j, n);
      bool found = false;
      for (size_t k : graph.adjacency[j]) {
        if (k == i) {
          found = true;
          break;
        }
      }
      EXPECT_TRUE(found) << "adjacency must be symmetric: " << i << " -> " << j
                         << " but not reverse";
    }
  }
}

TEST(VisibilityGraphTest, EmptyPolygonReturnsInvalid) {
  Polygon2D poly;
  auto graph = VisibilityGraph::Build(poly);
  EXPECT_FALSE(graph.IsValid());
  EXPECT_EQ(graph.vertices.size(), 0u);
}

TEST(VisibilityGraphTest, PolygonStoredInGraph) {
  Polygon2D poly(MakeSquare());
  auto graph = VisibilityGraph::Build(poly);

  EXPECT_TRUE(graph.IsValid());
  ASSERT_NE(graph.polygon, nullptr);
  EXPECT_EQ(graph.polygon->NumPoints(), poly.NumPoints());
}

}  // namespace test
}  // namespace spatialblend
