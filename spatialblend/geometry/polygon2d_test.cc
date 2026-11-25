// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/polygon2d_test.cc
#include "spatialblend/geometry/polygon2d.h"

#include <gtest/gtest.h>

namespace spatialblend {
namespace test {

using Eigen::Vector2d;

// ===== Helper polygons =====

// Simple 10x10 square at origin
std::vector<Vector2d> MakeSquare() {
  return {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
}

// L-shaped polygon (non-convex)
std::vector<Vector2d> MakeLShape() {
  return {{0, 0}, {10, 0}, {10, 5}, {5, 5}, {5, 10}, {0, 10}};
}

// Triangle
std::vector<Vector2d> MakeTriangle() {
  return {{0, 0}, {10, 0}, {5, 10}};
}

// ===== Construction =====

TEST(Polygon2DTest, ConstructFromPoints) {
  Polygon2D poly(MakeSquare());
  EXPECT_EQ(poly.NumPoints(), 4);
}

TEST(Polygon2DTest, GetPoints) {
  Polygon2D poly(MakeSquare());
  EXPECT_EQ(poly.GetPoints().size(), 4);
}

TEST(Polygon2DTest, GetCenter) {
  Polygon2D poly(MakeSquare());
  auto center = poly.GetCenter();
  EXPECT_NEAR(center.x(), 5.0, 0.1);
  EXPECT_NEAR(center.y(), 5.0, 0.1);
}

TEST(Polygon2DTest, GetArea) {
  Polygon2D poly(MakeSquare());
  EXPECT_NEAR(poly.GetArea(), 100.0, 0.1);
}

TEST(Polygon2DTest, TriangleArea) {
  Polygon2D poly(MakeTriangle());
  EXPECT_NEAR(poly.GetArea(), 50.0, 0.1);
}

// ===== Containment =====

TEST(Polygon2DTest, ContainsPointInside) {
  Polygon2D poly(MakeSquare());
  EXPECT_TRUE(poly.Contains(Vector2d(5, 5)));
}

TEST(Polygon2DTest, ContainsPointOutside) {
  Polygon2D poly(MakeSquare());
  EXPECT_FALSE(poly.Contains(Vector2d(15, 5)));
}

TEST(Polygon2DTest, ContainsPointOnBoundary) {
  Polygon2D poly(MakeSquare());
  // On boundary — include_boundary=true by default
  EXPECT_TRUE(poly.Contains(Vector2d(0, 5)));
  EXPECT_TRUE(poly.Contains(Vector2d(10, 5)));
}

TEST(Polygon2DTest, ContainsPointOnVertex) {
  Polygon2D poly(MakeSquare());
  EXPECT_TRUE(poly.Contains(Vector2d(0, 0)));
  EXPECT_TRUE(poly.Contains(Vector2d(10, 10)));
}

TEST(Polygon2DTest, ContainsExcludeBoundary) {
  Polygon2D poly(MakeSquare());
  EXPECT_TRUE(poly.Contains(Vector2d(5, 5), /*include_boundary=*/false));
  // On boundary with include_boundary=false → false
  EXPECT_FALSE(poly.Contains(Vector2d(0, 5), /*include_boundary=*/false));
}

TEST(Polygon2DTest, ContainsNonConvex) {
  Polygon2D poly(MakeLShape());
  // Inside the L
  EXPECT_TRUE(poly.Contains(Vector2d(2, 2)));
  EXPECT_TRUE(poly.Contains(Vector2d(2, 8)));
  // Outside the L (in the concave notch)
  EXPECT_FALSE(poly.Contains(Vector2d(8, 8)));
}

TEST(Polygon2DTest, ContainsNearBoundary) {
  Polygon2D poly(MakeSquare());
  // Just inside
  EXPECT_TRUE(poly.Contains(Vector2d(0.001, 0.001)));
  // Just outside
  EXPECT_FALSE(poly.Contains(Vector2d(-0.001, -0.001)));
}

// ===== Nearest Point =====

TEST(Polygon2DTest, NearestPointFromOutside) {
  Polygon2D poly(MakeSquare());
  // Point to the right of the square → nearest is on right edge
  auto nearest = poly.GetNearestPoint(Vector2d(15, 5));
  EXPECT_NEAR(nearest.x(), 10.0, 0.1);
  EXPECT_NEAR(nearest.y(), 5.0, 0.1);
}

TEST(Polygon2DTest, NearestPointFromCorner) {
  Polygon2D poly(MakeSquare());
  // Point diagonally outside → nearest is corner
  auto nearest = poly.GetNearestPoint(Vector2d(15, 15));
  EXPECT_NEAR(nearest.x(), 10.0, 0.1);
  EXPECT_NEAR(nearest.y(), 10.0, 0.1);
}

TEST(Polygon2DTest, NearestPointFromInside) {
  Polygon2D poly(MakeSquare());
  // Point inside → nearest is on boundary
  auto nearest = poly.GetNearestPoint(Vector2d(9, 5));
  EXPECT_NEAR(nearest.x(), 10.0, 0.1);
  EXPECT_NEAR(nearest.y(), 5.0, 0.5);
}

// ===== Distance =====

TEST(Polygon2DTest, DistanceFromOutside) {
  Polygon2D poly(MakeSquare());
  double dist = poly.GetDistance(Vector2d(15, 5));
  EXPECT_NEAR(dist, 5.0, 0.1);
}

TEST(Polygon2DTest, DistanceFromInside) {
  Polygon2D poly(MakeSquare());
  // Distance to boundary from inside
  double dist = poly.GetDistance(Vector2d(5, 5));
  EXPECT_NEAR(dist, 5.0, 0.1);
}

TEST(Polygon2DTest, DistanceFromBoundary) {
  Polygon2D poly(MakeSquare());
  double dist = poly.GetDistance(Vector2d(10, 5));
  EXPECT_NEAR(dist, 0.0, 0.1);
}

// ===== Segment Intersection =====

TEST(Polygon2DTest, SegmentCrossesThrough) {
  Polygon2D poly(MakeSquare());
  // Segment crosses from outside to outside through polygon
  EXPECT_TRUE(poly.IntersectsSegment(Vector2d(-1, 5), Vector2d(11, 5)));
}

TEST(Polygon2DTest, SegmentInsideToOutside) {
  Polygon2D poly(MakeSquare());
  // Segment starts inside, exits
  EXPECT_TRUE(poly.IntersectsSegment(Vector2d(5, 5), Vector2d(15, 5)));
}

TEST(Polygon2DTest, SegmentEntirelyInside) {
  Polygon2D poly(MakeSquare());
  // Segment entirely inside → does NOT intersect boundary
  EXPECT_FALSE(poly.IntersectsSegment(Vector2d(2, 2), Vector2d(8, 8)));
}

TEST(Polygon2DTest, SegmentEntirelyOutside) {
  Polygon2D poly(MakeSquare());
  EXPECT_FALSE(poly.IntersectsSegment(Vector2d(15, 0), Vector2d(15, 10)));
}

TEST(Polygon2DTest, SegmentTouchesBoundary) {
  Polygon2D poly(MakeSquare());
  // Segment runs along boundary edge
  EXPECT_TRUE(poly.IntersectsSegment(Vector2d(-1, 0), Vector2d(11, 0)));
}

TEST(Polygon2DTest, SegmentNonConvex) {
  Polygon2D poly(MakeLShape());
  // Segment through the concave notch area — should cross boundary twice
  EXPECT_TRUE(poly.IntersectsSegment(Vector2d(2, 7), Vector2d(8, 7)));
}

// ===== Inflate =====

TEST(Polygon2DTest, InflateGrows) {
  Polygon2D poly(MakeSquare());
  auto inflated = poly.Inflate(1.0);
  // Original doesn't contain (11, 5), inflated should
  EXPECT_FALSE(poly.Contains(Vector2d(11, 5)));
  EXPECT_TRUE(inflated.Contains(Vector2d(10.5, 5)));
}

TEST(Polygon2DTest, InflatePreservesCenter) {
  Polygon2D poly(MakeSquare());
  auto inflated = poly.Inflate(2.0);
  EXPECT_TRUE(inflated.Contains(Vector2d(5, 5)));
}

// An erosion that removes the polygon is empty. Returning the original would
// claim a domain the body cannot occupy.
TEST(Polygon2DTest, ErosionOfVanishedRegionIsEmpty) {
  Polygon2D poly(MakeSquare());  // 10x10
  auto gone = poly.Inflate(-20.0);
  EXPECT_EQ(gone.NumComponents(), 0u);
  EXPECT_EQ(gone.NumPoints(), 0u);
  EXPECT_FALSE(gone.Contains(Vector2d(5, 5)));
  EXPECT_DOUBLE_EQ(gone.GetArea(), 0.0);
}

// A thin neck that erodes away leaves both rooms. v1 kept every component of
// the buffer; dropping all but the first would hide a whole room.
TEST(Polygon2DTest, ErosionKeepsSplitComponents) {
  // Two 2x2 blocks joined by a corridor of height 0.2.
  std::vector<Vector2d> barbell = {
      {0, 0},   {2, 0},   {2, 0.9}, {4, 0.9}, {4, 0},   {6, 0},
      {6, 2},   {4, 2},   {4, 1.1}, {2, 1.1}, {2, 2},   {0, 2},
  };
  Polygon2D poly(barbell);
  ASSERT_EQ(poly.NumComponents(), 1u);
  auto split = poly.Inflate(-0.15);
  EXPECT_EQ(split.NumComponents(), 2u);
  EXPECT_TRUE(split.Contains(Vector2d(1, 1)));
  EXPECT_TRUE(split.Contains(Vector2d(5, 1)));
  EXPECT_FALSE(split.Contains(Vector2d(3, 1)));
}

// ===== Edge Cases =====

TEST(Polygon2DTest, EmptyPolygon) {
  Polygon2D poly(std::vector<Vector2d>{});
  EXPECT_EQ(poly.NumPoints(), 0);
  EXPECT_FALSE(poly.Contains(Vector2d(0, 0)));
}

TEST(Polygon2DTest, SinglePoint) {
  Polygon2D poly({{5, 5}});
  EXPECT_EQ(poly.NumPoints(), 1);
}

TEST(Polygon2DTest, TwoPoints) {
  Polygon2D poly({{0, 0}, {10, 0}});
  EXPECT_EQ(poly.NumPoints(), 2);
}

// ===== RayBoundaryIntersection =====

TEST(Polygon2DTest, RayBoundaryIntersectionRight) {
  Polygon2D poly({{0, 0}, {10, 0}, {10, 10}, {0, 10}});
  auto result = poly.RayBoundaryIntersection(
      Eigen::Vector2d(5, 5), Eigen::Vector2d(1, 0));
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->x(), 10.0, 0.1);
  EXPECT_NEAR(result->y(), 5.0, 0.1);
}

TEST(Polygon2DTest, RayBoundaryIntersectionUp) {
  Polygon2D poly({{0, 0}, {10, 0}, {10, 10}, {0, 10}});
  auto result = poly.RayBoundaryIntersection(
      Eigen::Vector2d(5, 5), Eigen::Vector2d(0, 1));
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->x(), 5.0, 0.1);
  EXPECT_NEAR(result->y(), 10.0, 0.1);
}

TEST(Polygon2DTest, RayBoundaryIntersectionMiss) {
  Polygon2D poly({{0, 0}, {10, 0}, {10, 10}, {0, 10}});
  auto result = poly.RayBoundaryIntersection(
      Eigen::Vector2d(20, 5), Eigen::Vector2d(1, 0));
  EXPECT_FALSE(result.has_value());
}

TEST(Polygon2DTest, RayBoundaryIntersectionDiagonal) {
  Polygon2D poly({{0, 0}, {10, 0}, {10, 10}, {0, 10}});
  Eigen::Vector2d dir(1, 1);
  dir.normalize();
  auto result = poly.RayBoundaryIntersection(
      Eigen::Vector2d(5, 5), dir);
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->x(), 10.0, 0.1);
  EXPECT_NEAR(result->y(), 10.0, 0.1);
}

// NavMesh exports: a spike (an edge that doubles back) and a keyhole (a zero-width slit cut from
// the outer ring to a hole) are invalid to Boost only by features of zero area. Repaired() keeps
// the region (the hole stays a hole); a ring that crosses itself is refused.
TEST(Polygon2DTest, RepairRemovesSpikesAndKeyholeSlitsAndKeepsTheHole) {
  const Polygon2D spike({{0, 0}, {4, 0}, {4, 2}, {6, 2}, {4, 2}, {4, 4}, {0, 4}});
  EXPECT_FALSE(spike.IsValid());
  const Polygon2D fixed = spike.Repaired();
  ASSERT_TRUE(fixed.IsValid());
  EXPECT_NEAR(fixed.GetArea(), 16.0, 1e-9);
  EXPECT_FALSE(fixed.Contains({5, 2}));

  // The outer ring counter-clockwise; the slit enters at (5,10) and walks the hole clockwise.
  const Polygon2D keyhole({{0, 0}, {10, 0}, {10, 10}, {5, 10}, {5, 7}, {7, 7}, {7, 3}, {3, 3},
                           {3, 7}, {5, 7}, {5, 10}, {0, 10}});
  EXPECT_FALSE(keyhole.IsValid());
  const Polygon2D opened = keyhole.Repaired();
  ASSERT_TRUE(opened.IsValid());
  EXPECT_NEAR(opened.GetArea(), 100.0 - 16.0, 1e-9);
  EXPECT_FALSE(opened.Contains({5, 5})) << "the hole stays a hole";
  EXPECT_TRUE(opened.Contains({1, 1}));

  const Polygon2D bowtie({{0, 0}, {2, 2}, {2, 0}, {0, 2}});
  EXPECT_EQ(bowtie.Repaired().NumComponents(), 0u);
}

}  // namespace test
}  // namespace spatialblend
