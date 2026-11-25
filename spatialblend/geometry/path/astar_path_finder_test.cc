// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/path/astar_path_finder_test.cc
#include "spatialblend/geometry/path/astar_path_finder.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/geometry/visibility_graph.h"
#include "spatialblend/geometry/path/discrete_path.h"

namespace spatialblend {
namespace test {

using Eigen::Vector2d;

// ===== Helper polygons =====

// 10x10 square at origin (open, convex)
Polygon2D MakeSquarePoly() {
  return Polygon2D({{0, 0}, {10, 0}, {10, 10}, {0, 10}});
}

// L-shape: {0,0},{10,0},{10,5},{5,5},{5,10},{0,10}
// The concave notch is at upper-right. A path from (1,9) to (9,1) must go
// around the inner corner at (5,5).
Polygon2D MakeLShapePoly() {
  return Polygon2D({{0, 0}, {10, 0}, {10, 5}, {5, 5}, {5, 10}, {0, 10}});
}

// ===== Tests =====

TEST(AStarPathFinderTest, StraightLineInOpenSquare) {
  auto poly = MakeSquarePoly();
  auto graph = VisibilityGraph::Build(poly);
  AStarPathFinder finder(graph);

  Vector2d start(1.0, 1.0);
  Vector2d goal(9.0, 9.0);
  auto [path, dist] = finder.PlanPath(start, goal);

  EXPECT_GT(path.NumPoints(), 0u);
  // Euclidean distance is the lower bound.
  double eucl = (goal - start).norm();
  EXPECT_GE(dist + 1e-9, eucl) << "path distance should be >= Euclidean";
  EXPECT_NEAR(dist, path.GetLength(), 1e-9);
}

TEST(AStarPathFinderTest, StartEqualsGoalDistanceIsZero) {
  auto poly = MakeSquarePoly();
  auto graph = VisibilityGraph::Build(poly);
  AStarPathFinder finder(graph);

  Vector2d pt(5.0, 5.0);
  auto [path, dist] = finder.PlanPath(pt, pt);

  // Either the path is trivial with 0 length, or start==goal gives 0.
  EXPECT_NEAR(dist, 0.0, 1e-9);
}

TEST(AStarPathFinderTest, PathLengthGeEuclidean) {
  auto poly = MakeSquarePoly();
  auto graph = VisibilityGraph::Build(poly);
  AStarPathFinder finder(graph);

  Vector2d start(0.5, 0.5);
  Vector2d goal(9.5, 9.5);
  auto [path, dist] = finder.PlanPath(start, goal);

  double eucl = (goal - start).norm();
  EXPECT_GE(dist + 1e-9, eucl);
}

TEST(AStarPathFinderTest, PathAroundLShapeCorner) {
  // The L-shaped polygon {0,0},{10,0},{10,5},{5,5},{5,10},{0,10} has a
  // concave notch at the upper-right. A straight line from (1,9) to (9,4)
  // passes through the point (5.8, 6.0) which is OUTSIDE the polygon
  // (x>5 and y>5), so A* must route around the inner corner at (5,5).
  auto poly = MakeLShapePoly();
  auto graph = VisibilityGraph::Build(poly);
  AStarPathFinder finder(graph);

  Vector2d start(1.0, 9.0);  // top-left arm
  Vector2d goal(9.0, 4.0);   // bottom-right arm

  auto [path, dist] = finder.PlanPath(start, goal);

  EXPECT_GT(path.NumPoints(), 0u);
  EXPECT_LT(dist, std::numeric_limits<double>::infinity());

  // Path length must be at least Euclidean.
  double eucl = (goal - start).norm();
  EXPECT_GE(dist + 1e-9, eucl);

  // Geodesic distance around the concave corner must be strictly greater
  // than the Euclidean (straight-line) distance.
  EXPECT_GT(dist, eucl + 1e-6)
      << "Path around L-shape corner should be longer than straight line";
}

TEST(AStarPathFinderTest, PathWaypointsInsidePolygon) {
  auto poly = MakeLShapePoly();
  auto graph = VisibilityGraph::Build(poly);
  AStarPathFinder finder(graph);

  Vector2d start(1.0, 9.0);
  Vector2d goal(9.0, 4.0);
  auto [path, dist] = finder.PlanPath(start, goal);

  ASSERT_GT(path.NumPoints(), 0u);
  for (const auto& pt : path.GetPoints()) {
    EXPECT_TRUE(poly.Contains(pt, /*include_boundary=*/true))
        << "Path point (" << pt.x() << ", " << pt.y()
        << ") should be inside polygon";
  }
}

TEST(AStarPathFinderTest, GetDistanceMatchesPlanPath) {
  auto poly = MakeSquarePoly();
  auto graph = VisibilityGraph::Build(poly);
  AStarPathFinder finder(graph);

  Vector2d start(1.0, 1.0);
  Vector2d goal(9.0, 9.0);

  auto [path, plan_dist] = finder.PlanPath(start, goal);
  double get_dist = finder.GetDistance(start, goal);

  EXPECT_NEAR(plan_dist, get_dist, 1e-9);
}

TEST(AStarPathFinderTest, PathWithMultipleWaypoints) {
  auto poly = MakeLShapePoly();
  auto graph = VisibilityGraph::Build(poly);
  AStarPathFinder finder(graph);

  Vector2d start(1.0, 9.0);
  Vector2d goal(9.0, 4.0);
  auto [path, dist] = finder.PlanPath(start, goal);

  // Must go through at least one intermediate waypoint (the corner).
  EXPECT_GE(path.NumPoints(), 2u);

  // Verify path length consistency.
  EXPECT_NEAR(dist, path.GetLength(), 1e-9);
}

TEST(AStarPathFinderTest, NonConvexGeodesicGtEuclidean) {
  // Larger L-shape to make the effect more pronounced.
  Polygon2D poly({{0, 0}, {20, 0}, {20, 8}, {8, 8}, {8, 20}, {0, 20}});
  auto graph = VisibilityGraph::Build(poly);
  AStarPathFinder finder(graph);

  Vector2d start(1.0, 18.0);
  Vector2d goal(18.0, 1.0);

  auto [path, geodesic_dist] = finder.PlanPath(start, goal);
  double euclidean_dist = (goal - start).norm();

  EXPECT_LT(geodesic_dist, std::numeric_limits<double>::infinity())
      << "Should find a path";
  EXPECT_GT(geodesic_dist, euclidean_dist + 1e-6)
      << "Geodesic must be longer than Euclidean in non-convex polygon";
}

}  // namespace test
}  // namespace spatialblend
