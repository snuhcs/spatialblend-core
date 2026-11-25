// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/context/obstacle_free_space_test.cc
//
// Geometry contract: the free space of a point body
// with clearance r inside a walkable polygon with interior obstacles. Every
// oracle below is analytic (axis-aligned box distance), independent of the
// implementation's segment/polygon distance code.
#include "spatialblend/geometry/obstacle_free_space.h"

#include <array>
#include <cmath>
#include <random>

#include <boost/multiprecision/cpp_int.hpp>
#include <limits>
#include <queue>
#include <string>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "spatialblend/geometry/context/navigable_geometry_context.h"

namespace spatialblend {
namespace {

constexpr double kRadius = 0.2;

struct Box {
  double x0, y0, x1, y1;
  // Analytic distance from p to the box (0 inside).
  double Distance(const Eigen::Vector2d& p) const {
    const double cx = 0.5 * (x0 + x1), cy = 0.5 * (y0 + y1);
    const double hx = 0.5 * (x1 - x0), hy = 0.5 * (y1 - y0);
    const double dx = std::max(0.0, std::abs(p.x() - cx) - hx);
    const double dy = std::max(0.0, std::abs(p.y() - cy) - hy);
    return std::hypot(dx, dy);
  }
  std::shared_ptr<const Polygon2D> Polygon() const {
    return std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
        {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});
  }
};

std::shared_ptr<const Polygon2D> Room(double w, double h) {
  return std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {w, 0}, {w, h}, {0, h}});
}

// Independent exact certificate for the randomized convex-box tests below.
// Rational half-space clipping uses the input doubles' exact binary values;
// there is no union polygon, tangent test, floating intersection parameter or
// point-sampling rule shared with the implementation under test.
using Exact = boost::multiprecision::cpp_rational;
using ExactPoint = std::array<Exact, 2>;

ExactPoint ExactCoordinates(const Eigen::Vector2d& p) {
  return {Exact(p.x()), Exact(p.y())};
}

struct ConvexBoxOracle {
  std::vector<Eigen::Vector2d> vertices;
  std::vector<ExactPoint> exact_vertices;
  Eigen::Vector2d lo, hi;
  double boundary_distance[4][4] = {};

  explicit ConvexBoxOracle(const std::vector<Eigen::Vector2d>& points)
      : vertices(points), lo(points.front()), hi(points.front()) {
    for (const auto& p : points) {
      exact_vertices.push_back(ExactCoordinates(p));
      lo = lo.cwiseMin(p);
      hi = hi.cwiseMax(p);
    }
    // These are counterclockwise convex quadrilaterals, generated independently
    // of Polygon2D's correction/union code.
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        boundary_distance[i][j] = i == j ? 0.0 : std::numeric_limits<double>::infinity();
    for (int i = 0; i < 4; ++i) {
      const int j = (i + 1) % 4;
      boundary_distance[i][j] = boundary_distance[j][i] = (vertices[i] - vertices[j]).norm();
    }
    for (int k = 0; k < 4; ++k)
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          boundary_distance[i][j] = std::min(boundary_distance[i][j],
                                            boundary_distance[i][k] + boundary_distance[k][j]);
  }

  bool EntersInterior(const Eigen::Vector2d& from, const Eigen::Vector2d& to) const {
    if ((from.cwiseMax(to).array() < lo.array()).any() ||
        (from.cwiseMin(to).array() > hi.array()).any()) return false;
    const auto a = ExactCoordinates(from), b = ExactCoordinates(to);
    const Exact dx = b[0] - a[0], dy = b[1] - a[1];
    Exact lower = 0, upper = 1;
    for (size_t i = 0; i < exact_vertices.size(); ++i) {
      const auto& p = exact_vertices[i];
      const auto& q = exact_vertices[(i + 1) % exact_vertices.size()];
      const Exact ex = q[0] - p[0], ey = q[1] - p[1];
      const Exact initial = ex * (a[1] - p[1]) - ey * (a[0] - p[0]);
      const Exact slope = ex * dy - ey * dx;
      if (slope == 0) {
        if (initial <= 0) return false;
      } else {
        const Exact crossing = -initial / slope;
        if (slope > 0) lower = std::max(lower, crossing);
        else upper = std::min(upper, crossing);
      }
      if (upper <= lower) return false;
    }
    return upper > lower;
  }

  // A taut string around this one convex obstacle is a lower bound on any
  // path avoiding the complete union, including a union with sealed seams.
  double DistanceLowerBound(const Eigen::Vector2d& from, const Eigen::Vector2d& to) const {
    if (!EntersInterior(from, to)) return (to - from).norm();
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 4; ++i) {
      if (EntersInterior(from, vertices[i])) continue;
      for (int j = 0; j < 4; ++j) {
        if (EntersInterior(vertices[j], to)) continue;
        best = std::min(best, (from - vertices[i]).norm() + boundary_distance[i][j] +
                              (vertices[j] - to).norm());
      }
    }
    return best;
  }
};

// Minimum analytic clearance along a polyline, sampled every 5 mm, skipping
// the first/last `skip_m` metres (explicit snap legs).
double MinClearance(const std::vector<Eigen::Vector2d>& pts, const Box& box) {
  double best = std::numeric_limits<double>::infinity();
  for (size_t i = 1; i < pts.size(); ++i) {
    const Eigen::Vector2d a = pts[i - 1], b = pts[i];
    const int n = std::max(1, static_cast<int>((b - a).norm() / 0.005));
    for (int k = 0; k <= n; ++k) {
      best = std::min(best, box.Distance(a + (b - a) * (double(k) / n)));
    }
  }
  return best;
}

bool InsideRoom(const std::vector<Eigen::Vector2d>& pts, double w, double h) {
  for (const auto& p : pts) {
    if (p.x() < -1e-9 || p.y() < -1e-9 || p.x() > w + 1e-9 ||
        p.y() > h + 1e-9)
      return false;
  }
  return true;  // the room is convex: vertices inside => segments inside
}

// An obstacle between start and goal with a free detour. The path keeps
// >= r from the obstacle on every sampled point and stays in the room.
TEST(ObstacleFreeSpaceTest, R2_1_DetourKeepsBodyRadiusClearance) {
  const Box box{4, 0, 6, 6};  // attached to the south wall; detour over y=6
  ObstacleFreeSpace fs(Room(10, 10), {{7, box.Polygon()}}, kRadius);

  const Eigen::Vector2d a(2, 2), b(8, 2);
  auto path = fs.FindPath(a, b);
  ASSERT_TRUE(path.found);
  EXPECT_DOUBLE_EQ(path.from_snap_m, 0.0);
  EXPECT_DOUBLE_EQ(path.to_snap_m, 0.0);
  EXPECT_GE(MinClearance(path.points, box), kRadius - 1e-9);
  EXPECT_TRUE(InsideRoom(path.points, 10, 10));
  // Analytic bounds: at least the taut string around the raw corners, at
  // most the taut string around the miter-inflated corners.
  const double lower = 2 * std::hypot(2.0, 4.0) + 2.0;
  const double upper = 2 * std::hypot(1.8, 4.2) + 2.4 + 1e-3;
  EXPECT_GE(path.length, lower);
  EXPECT_LE(path.length, upper);
}

// The contact field is the same length as a fresh FindPath: straight when
// the segment is free, otherwise the visibility-graph geodesic. A second
// query to the same contact must hit the cached field and still match.
TEST(ObstacleFreeSpaceTest, GeodesicDistanceMatchesFindPath) {
  const Box box{4, 0, 6, 6};
  ObstacleFreeSpace fs(Room(10, 10), {{7, box.Polygon()}}, kRadius);
  const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> pairs = {
      {{2, 2}, {8, 2}},
      {{1, 8}, {9, 1}},
      {{2, 2}, {3, 3}},
      {{5, 5}, {8, 8}},
      {{0.2, 0.2}, {9.5, 9.5}},
  };
  for (const auto& [a, b] : pairs) {
    const auto path = fs.FindPath(a, b);
    const double once = fs.GeodesicDistance(a, b);
    const double again = fs.GeodesicDistance(a, b);
    if (!path.found) {
      EXPECT_FALSE(std::isfinite(once));
      EXPECT_FALSE(std::isfinite(again));
      continue;
    }
    EXPECT_NEAR(once, path.length, 1e-6) << a.transpose() << " -> " << b.transpose();
    EXPECT_NEAR(again, path.length, 1e-6);
  }
  // Reversed pairs build the field from the other contact.
  for (const auto& [a, b] : pairs) {
    const auto path = fs.FindPath(b, a);
    const double d = fs.GeodesicDistance(b, a);
    if (!path.found) {
      EXPECT_FALSE(std::isfinite(d));
    } else {
      EXPECT_NEAR(d, path.length, 1e-6);
    }
  }

  // Several footprints, including a start inside one: the field length is
  // the per-query geodesic, not a tangent shortcut.
  std::vector<FreeSpaceObstacle> many;
  for (int i = 0; i < 8; ++i) {
    const Eigen::Vector2d c(2.0 + (i % 4) * 2.0, 2.0 + (i / 4) * 5.0);
    many.push_back({i + 1, std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
                               c + Eigen::Vector2d(-0.4, -0.4),
                               c + Eigen::Vector2d(0.4, -0.4),
                               c + Eigen::Vector2d(0.4, 0.4),
                               c + Eigen::Vector2d(-0.4, 0.4)})});
  }
  ObstacleFreeSpace cluttered(Room(10, 10), many, 0.0);
  const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> cluttered_pairs = {
      {{0.5, 0.5}, {9.5, 9.5}},
      {{0.5, 9.5}, {9.5, 0.5}},
      {{2.0, 2.0}, {8.0, 7.0}},
      {{1.0, 5.0}, {9.0, 5.0}},
  };
  for (const auto& [a, b] : cluttered_pairs) {
    const auto path = cluttered.FindPath(a, b);
    const double d = cluttered.GeodesicDistance(a, b);
    ASSERT_TRUE(path.found) << a.transpose() << " -> " << b.transpose();
    EXPECT_NEAR(d, path.length, 1e-6) << a.transpose() << " -> " << b.transpose();
  }

  // The ring field is the same contact distance as the per-query search.
  const Polygon2D target({{8.2, 8.2}, {9.2, 8.2}, {9.2, 9.2}, {8.2, 9.2}});
  for (const Eigen::Vector2d& a :
       {Eigen::Vector2d(0.5, 0.5), Eigen::Vector2d(0.5, 9.5),
        Eigen::Vector2d(5.0, 0.4), Eigen::Vector2d(2.0, 2.0)}) {
    double uncached = 0.0, cached = 0.0;
    cluttered.ClosestOnRingUncached(a, target, &uncached);
    cluttered.ClosestOnRing(a, target, &cached);
    EXPECT_NEAR(cached, uncached, 1e-4) << a.transpose();
    const double again = [&] {
      double v = 0.0;
      cluttered.ClosestOnRing(a, target, &v);
      return v;
    }();
    EXPECT_NEAR(again, uncached, 1e-4);
  }
}

// A path from a point to ITSELF has length 0 and exists, wherever the point is: in free
// space, on the clearance ring, inside the ring, inside the footprint. (A body seated exactly
// on its proposal inside a scaled seat footprint was reported unreachable -- the zero-length
// segment of the snapped endpoint was rejected -- so the receiver proposed HOLD while every
// error was 0.)
// A contact on a footprint is inside its clearance ring, so the goal snaps to
// the nearest free point. That snap is a function of the contact alone and is
// cached across the many from-points one solve measures. A cached answer is
// the one a fresh free space computes.
TEST(ObstacleFreeSpaceTest, GeodesicDistanceToAContactOnAFootprintIsCachedExactly) {
  const Box box{4, 0, 6, 6};
  ObstacleFreeSpace warm(Room(10, 10), {{7, box.Polygon()}}, kRadius);
  const Eigen::Vector2d contact(5.0, 3.0);  // on the footprint
  const std::vector<Eigen::Vector2d> froms = {
      {2, 2}, {1, 8}, {8, 2}, {9, 9}, {3.3, 4.1}, {2, 2}};
  for (const auto& from : froms) {
    ObstacleFreeSpace fresh(Room(10, 10), {{7, box.Polygon()}}, kRadius);
    const double expected = fresh.GeodesicDistance(from, contact);
    const double cached = warm.GeodesicDistance(from, contact);
    ASSERT_TRUE(std::isfinite(expected)) << from.transpose();
    EXPECT_DOUBLE_EQ(cached, expected) << from.transpose();
  }
}

TEST(ObstacleFreeSpaceTest, APathFromAPointToItselfAlwaysExists) {
  const Box box{4, 4, 6, 6};
  ObstacleFreeSpace fs(Room(10, 10), {{7, box.Polygon()}}, kRadius);
  const auto free_pt = fs.NearestFree(Eigen::Vector2d(5, 5));  // inside the footprint
  ASSERT_TRUE(free_pt.has_value());
  const std::vector<Eigen::Vector2d> pts = {
      {1, 1},                    // free
      *free_pt,                  // on the clearance ring
      {3.9, 5.0},                // inside the ring
      {5.0, 5.0},                // inside the footprint
      {4.0, 4.0},                // a footprint corner
      {4.0 - kRadius, 5.0},      // exactly on the ring
  };
  for (const auto& p : pts) {
    const auto r = fs.FindPath(p, p);
    EXPECT_TRUE(r.found) << p.transpose();
  }
}

// Two points on the same wall (both on one walkable boundary edge, e.g. a
// tracked anchor and a support snapped onto it) are connected along the wall.
// Previously only a whole vertex-to-vertex edge counted as "on the boundary",
// so the collinear sub-segment read as a crossing and the pair as
// unreachable (+inf geodesic).
TEST(ObstacleFreeSpaceTest, R2_1_PointsOnOneWallAreConnected) {
  auto room = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  auto ctx = NavigableGeometryContext::Create(room, 0.0, {}, 0.0);
  auto r = ctx->QueryReachability({4, 0}, {1, 0}, /*radius=*/0.0);
  EXPECT_TRUE(r.reachable);
  EXPECT_NEAR(r.geodesic_distance_m, 3.0, 1e-9);
  EXPECT_DOUBLE_EQ(r.from_snap_m, 0.0);
  EXPECT_DOUBLE_EQ(r.to_snap_m, 0.0);
}

// Same along a wall made of two collinear walkable edges (a vertex in the
// middle of the wall): the segment crosses their shared vertex.
TEST(ObstacleFreeSpaceTest, R2_1_PointsAlongCollinearEdgesAreConnected) {
  auto room = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {0, 0}, {10, 0}, {20, 0}, {20, 20}, {0, 20}});
  auto ctx = NavigableGeometryContext::Create(room, 0.0, {}, 0.0);
  auto r = ctx->QueryReachability({4, 0}, {15, 0}, /*radius=*/0.0);
  EXPECT_TRUE(r.reachable);
  EXPECT_NEAR(r.geodesic_distance_m, 11.0, 1e-9);
}

// An anchor inside an obstacle that is flush against a thin sealing wall
// must snap out on ITS side of the wall, not through the wall to the nearer
// free point on the other side.
//   room [0,10]^2, wall [5,5.1]x[0,10] (splits the room), couch [4.2,5]x[2,3]
//   flush against the wall; anchor (4.99, 2.5) inside the couch.
TEST(ObstacleFreeSpaceTest, R2_1_SnapDoesNotCrossAThinWall) {
  auto room = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {0, 0}, {10, 0}, {10, 10}, {0, 10}});
  const Box wall{5.0, 0.0, 5.1, 10.0};
  const Box couch{4.2, 2.0, 5.0, 3.0};
  auto ctx = NavigableGeometryContext::Create(
      room, 0.0, {{1, wall.Polygon()}, {2, couch.Polygon()}}, 0.0);
  const Eigen::Vector2d anchor(4.99, 2.5);
  auto left = ctx->QueryReachability(anchor, {2, 2}, /*radius=*/0.0);
  EXPECT_TRUE(left.reachable) << "the anchor's own side must stay reachable";
  auto right = ctx->QueryReachability(anchor, {8, 2}, /*radius=*/0.0);
  EXPECT_FALSE(right.reachable) << "snapped through the wall";
}


// No detour exists (the obstacle spans the room) -> explicit
// "no path", never a straight line through the obstacle.
TEST(ObstacleFreeSpaceTest, R2_1_NoDetourIsExplicitlyDisconnected) {
  const Box wall{4, 0, 6, 10};
  ObstacleFreeSpace fs(Room(10, 10), {{7, wall.Polygon()}}, kRadius);
  EXPECT_TRUE(fs.IsFree({2, 5}));
  EXPECT_TRUE(fs.IsFree({8, 5}));
  auto path = fs.FindPath({2, 5}, {8, 5});
  EXPECT_FALSE(path.found);
  EXPECT_TRUE(path.points.empty());
}

TEST(ObstacleFreeSpaceTest, R2_1_FreeMeansOutsideClearanceRing) {
  const Box box{4, 4, 6, 6};
  ObstacleFreeSpace fs(Room(10, 10), {{7, box.Polygon()}}, kRadius);
  EXPECT_FALSE(fs.IsFree({5, 5}));        // inside
  EXPECT_FALSE(fs.IsFree({5, 6.1}));      // in the ring
  EXPECT_TRUE(fs.IsFree({5, 6.25}));      // outside the ring
  EXPECT_FALSE(fs.IsFree({11, 5}));       // outside the room
  EXPECT_TRUE(fs.IsFree({5, 5}, {7}));    // exempt footprint
  EXPECT_TRUE(fs.IsFree({5, 6.1}, {7}));  // exempt ring
}

TEST(ObstacleFreeSpaceTest, R2_1_NearestFreeProjectsToTheClearanceBoundary) {
  const Box box{4, 4, 6, 6};
  ObstacleFreeSpace fs(Room(10, 10), {{7, box.Polygon()}}, kRadius);
  auto p = fs.NearestFree({5, 5.9});
  ASSERT_TRUE(p.has_value());
  EXPECT_TRUE(fs.IsFree(*p));
  EXPECT_NEAR(box.Distance(*p), kRadius, 1e-3);
  EXPECT_NEAR(p->x(), 5.0, 1e-6);  // straight out of the nearest (north) side
  EXPECT_GT(p->y(), 6.0);

  // Near a wall and the obstacle at once: both constraints hold.
  const Box corner{0.1, 4, 2, 6};
  ObstacleFreeSpace fs2(Room(10, 10), {{8, corner.Polygon()}}, kRadius);
  auto q = fs2.NearestFree({0.05, 5});
  ASSERT_TRUE(q.has_value());
  EXPECT_TRUE(fs2.IsFree(*q));
  EXPECT_GE(corner.Distance(*q), kRadius - 1e-9);
}

// Approaching an obstacle's own footprint is not detoured around that same
// footprint (the approached referent is exempt from the straight test).
TEST(ObstacleFreeSpaceTest, DirectApproachToOwnFootprintIsStraight) {
  const Box box{4, 0, 6, 6};
  ObstacleFreeSpace fs(Room(10, 10), {{7, box.Polygon()}}, kRadius);
  auto path = fs.FindPath({5, 9}, {5, 6});
  ASSERT_TRUE(path.found);
  EXPECT_NEAR(path.length, 3.0, 1e-9);
}

// The navigable context routes around obstacles once they are attached, and
// exposes the reachability query used by scope gating.
TEST(ObstacleFreeSpaceTest, R2_1_NavigableContextGeodesicAndReachability) {
  const Box box{4, 0, 6, 6};
  auto room = std::make_shared<Polygon2D>(*Room(10, 10));
  auto plain = NavigableGeometryContext::Create(room, 0.0);
  EXPECT_NEAR(plain->GetGeodesicDistance({2, 2}, {8, 2}), 6.0, 1e-6);

  auto ctx = NavigableGeometryContext::Create(
      room, 0.0, {{7, box.Polygon()}}, kRadius);
  const double geo = ctx->GetGeodesicDistance({2, 2}, {8, 2});
  EXPECT_GE(geo, 2 * std::hypot(2.0, 4.0) + 2.0);
  auto path = ctx->GetDiscretePath({2, 2}, {8, 2});
  EXPECT_GT(MinClearance(path.GetPoints(), box), 0.0);  // around, not through

  // The point geodesic is point connectivity around the RAW footprints
  // (independent of the body radius); the body-radius reachability detours
  // further out.
  auto point_reach = ctx->QueryReachability({2, 2}, {8, 2}, /*radius=*/0.0);
  EXPECT_TRUE(point_reach.reachable);
  EXPECT_NEAR(point_reach.geodesic_distance_m, geo, 1e-9);
  auto reach = ctx->QueryReachability({2, 2}, {8, 2});
  EXPECT_TRUE(reach.reachable);
  EXPECT_GT(reach.geodesic_distance_m, geo);
  EXPECT_GE(MinClearance(reach.path.GetPoints(), box), kRadius - 1e-9);

  const Box wall{4, 0, 6, 10};
  auto sealed = NavigableGeometryContext::Create(
      room, 0.0, {{9, wall.Polygon()}}, kRadius);
  auto blocked = sealed->QueryReachability({2, 5}, {8, 5});
  EXPECT_FALSE(blocked.reachable);
  EXPECT_FALSE(std::isfinite(blocked.geodesic_distance_m));
  // An unreachable geodesic is +inf, never a Euclidean fallback.
  EXPECT_FALSE(std::isfinite(sealed->GetGeodesicDistance({2, 5}, {8, 5})));
}

// A straight segment that lies in F and ends on the walkable boundary is
// direct. The previous form of this test started on the footprint corner and
// passed only because FindPath opened that obstacle's clearance ring whenever
// an endpoint sat in it. Endpoint membership and path search now use the same
// F, so a point on the footprint is snapped onto F first.
TEST(ObstacleFreeSpaceTest, SegmentToBoundaryPointIsDirect) {
  const Box chair{1.75, 0.75, 2.25, 1.25};
  auto room = std::make_shared<Polygon2D>(*Room(20, 20));
  ObstacleFreeSpace fs(room, {{2, chair.Polygon()}}, kRadius);
  const Eigen::Vector2d corner(2.25, 1.25), dir(0.6, 0.8);
  auto hit = room->RayBoundaryIntersection(corner, dir);
  ASSERT_TRUE(hit.has_value());
  const Eigen::Vector2d start = corner + dir * (kRadius + 1e-3);
  ASSERT_TRUE(fs.IsFree(start));
  auto path = fs.FindPath(start, *hit);
  ASSERT_TRUE(path.found);
  EXPECT_EQ(path.points.size(), 2u);
  EXPECT_NEAR(path.length, (*hit - start).norm(), 1e-6);

  auto from_footprint = fs.FindPath(corner, *hit);
  ASSERT_TRUE(from_footprint.found);
  EXPECT_GT(from_footprint.from_snap_m, 0.0);
  EXPECT_TRUE(fs.IsFree(from_footprint.from_free));
  EXPECT_FALSE(fs.IsFree(corner));
}

// ---------------------------------------------------------------------------
// Regression cases, kept as permanent tests.
// ---------------------------------------------------------------------------

double PolylineClearance(const std::vector<Eigen::Vector2d>& pts,
                         const Box& box) {
  return MinClearance(pts, box);
}

// Standing inside a table's clearance ring must not let the
// direct segment pass THROUGH that table.
TEST(ObstacleFreeSpaceTest, R2_1_Review1_DirectSegmentNeverCrossesFootprint) {
  const Box table{4, 3, 6, 7};
  ObstacleFreeSpace fs(Room(10, 10), {{1, table.Polygon()}}, kRadius);
  auto r = fs.FindPath({3.9, 5}, {7, 5});
  ASSERT_TRUE(r.found);
  EXPECT_GT(PolylineClearance(r.points, table), 0.0)
      << "path crosses the table footprint";
  // Only the snap leg from the ring start may be closer than the radius.
  std::vector<Eigen::Vector2d> body(r.points.begin() + 1, r.points.end());
  EXPECT_GE(PolylineClearance(body, table), kRadius - 1e-9);
}

std::shared_ptr<const Polygon2D> Poly(std::vector<Eigen::Vector2d> v) {
  return std::make_shared<Polygon2D>(v);
}

// Two diamond tips 0.7 m apart with clearance 0.3 leave a 0.1 m
// free gap; the exact free space is connected and FindPath must find it,
// with a near-straight length (not a huge detour).
TEST(ObstacleFreeSpaceTest, R2_1_Review2_GapBetweenDiamondTips) {
  for (double shift : {0.0, -1.0}) {
    SCOPED_TRACE(shift);
    auto room = Poly({{0, 0}, {10, 0}, {10, 4}, {0, 4}});
    const double cx = 5 + shift;
    auto up = Poly({{cx, 2.35}, {cx + 3, 5.35}, {cx, 8.35}, {cx - 3, 5.35}});
    auto lo = Poly({{cx, 1.65}, {cx + 3, -1.35}, {cx, -4.35}, {cx - 3, -1.35}});
    ObstacleFreeSpace fs(room, {{1, up}, {2, lo}}, 0.3);
    const Eigen::Vector2d a(cx - 3 + 0.0, 1.5), m(cx, 2.0), b(8, 1.8);
    ASSERT_TRUE(fs.IsFree(a));
    ASSERT_TRUE(fs.IsFree(b));
    ASSERT_TRUE(fs.SegmentFree(a, m) && fs.SegmentFree(m, b));
    auto r = fs.FindPath(a, b);
    ASSERT_TRUE(r.found) << "exact free space connected but not found";
    EXPECT_LE(r.length, (m - a).norm() + (b - m).norm() + 1e-6);
    for (size_t i = 1; i < r.points.size(); ++i)
      EXPECT_TRUE(fs.SegmentFree(r.points[i - 1], r.points[i]));
  }
}

TEST(ObstacleFreeSpaceTest, R2_1_Review3_LShapeReflexNearObstacle) {
  auto room = Poly({{0, 0}, {10, 0}, {10, 4}, {4, 4}, {4, 10}, {0, 10}});
  ObstacleFreeSpace fs(room, {{1, Poly({{4.1, 4.1}, {5, 4.1}, {5, 5}, {4.1, 5}})}},
                       kRadius);
  auto r = fs.FindPath({8, 2}, {2, 8});
  ASSERT_TRUE(r.found);
  for (size_t i = 1; i < r.points.size(); ++i)
    EXPECT_TRUE(fs.SegmentFree(r.points[i - 1], r.points[i]));
}

TEST(ObstacleFreeSpaceTest, R2_1_Review4_OverlappingObstaclesPocket) {
  ObstacleFreeSpace fs(Room(10, 10),
                       {{1, Poly({{3, 2}, {7, 2}, {7, 3}, {3, 3}})},
                        {2, Poly({{3, 2.5}, {4, 2.5}, {4, 8}, {3, 8}})},
                        {3, Poly({{6, 2.5}, {7, 2.5}, {7, 8}, {6, 8}})}},
                       kRadius);
  auto r = fs.FindPath({5, 5}, {5, 1});
  ASSERT_TRUE(r.found);
  EXPECT_GT(r.length, 8.0);
}

// Grid oracle (independent of the visibility graph): Dijkstra on a fine
// 16-connected grid of points whose clearance exceeds the radius by a margin
// (so every grid move is a truly free segment). Completeness: grid-connected
// => FindPath finds a path. Near-optimality: FindPath is no longer than the
// grid path (which over-estimates the optimum). Validity: every path segment
// passes the exact clearance test.
struct GridOracle {
  double length = std::numeric_limits<double>::infinity();
};

GridOracle GridShortestPath(const ObstacleFreeSpace& fs,
                            const std::vector<Box>& boxes, double w, double h,
                            double radius, Eigen::Vector2d a,
                            Eigen::Vector2d b, double step) {
  const int nx = static_cast<int>(w / step) + 1;
  const int ny = static_cast<int>(h / step) + 1;
  const double margin = 0.03 + step;
  auto clear = [&](const Eigen::Vector2d& p) {
    for (const auto& bx : boxes)
      if (bx.Distance(p) < radius + margin) return false;
    return p.x() >= 0 && p.y() >= 0 && p.x() <= w && p.y() <= h;
  };
  std::vector<char> ok(nx * ny);
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) ok[j * nx + i] = clear({i * step, j * step});
  auto idx = [&](const Eigen::Vector2d& p) {
    int i = static_cast<int>(std::lround(p.x() / step));
    int j = static_cast<int>(std::lround(p.y() / step));
    return j * nx + i;
  };
  const int s = idx(a), t = idx(b);
  GridOracle out;
  if (!ok[s] || !ok[t]) return out;
  std::vector<double> dist(nx * ny, std::numeric_limits<double>::infinity());
  using Q = std::pair<double, int>;
  std::priority_queue<Q, std::vector<Q>, std::greater<Q>> q;
  dist[s] = 0;
  q.push({0, s});
  const int moves[16][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1},
                            {1, -1}, {-1, 1}, {-1, -1}, {2, 1}, {2, -1},
                            {-2, 1}, {-2, -1}, {1, 2}, {1, -2}, {-1, 2},
                            {-1, -2}};
  while (!q.empty()) {
    auto [d, u] = q.top();
    q.pop();
    if (u == t) break;
    if (d > dist[u]) continue;
    const int ui = u % nx, uj = u / nx;
    for (const auto& m : moves) {
      const int vi = ui + m[0], vj = uj + m[1];
      if (vi < 0 || vj < 0 || vi >= nx || vj >= ny) continue;
      const int v = vj * nx + vi;
      if (!ok[v]) continue;
      if (std::abs(m[0]) + std::abs(m[1]) == 3 &&
          !ok[(uj + m[1] / 2) * nx + (ui + m[0] / 2)])
        continue;
      const double nd = d + step * std::hypot(m[0], m[1]);
      if (nd < dist[v]) {
        dist[v] = nd;
        q.push({nd, v});
      }
    }
  }
  out.length = dist[t] + (a - Eigen::Vector2d((s % nx) * step, (s / nx) * step)).norm() +
               (b - Eigen::Vector2d((t % nx) * step, (t / nx) * step)).norm();
  return out;
}

TEST(ObstacleFreeSpaceTest, R2_1_GridOracleCompletenessAndNearOptimality) {
  struct Case {
    const char* name;
    std::vector<Box> boxes;
    Eigen::Vector2d a, b;
  };
  const std::vector<Case> cases = {
      {"gap_0.6", {{4, 0, 6, 4.7}, {4, 5.3, 6, 10}}, {2, 5}, {8, 5}},
      {"gap_0.3_closed", {{4, 0, 6, 4.85}, {4, 5.15, 6, 10}}, {2, 5}, {8, 5}},
      {"overlap_L", {{3, 3, 7, 4}, {6, 3, 7, 8}}, {5, 6}, {8.5, 2}},
      {"table_chairs",
       {{4, 4, 6, 6}, {4.6, 3.1, 5.4, 3.7}, {4.6, 6.3, 5.4, 6.9},
        {3.1, 4.6, 3.7, 5.4}, {6.3, 4.6, 6.9, 5.4}},
       {1, 5}, {9, 5}},
      {"corridor_maze",
       {{2, 0, 2.5, 8}, {5, 2, 5.5, 10}, {7.5, 0, 8, 8}},
       {1, 1}, {9, 9}},
  };
  const double step = 0.025;
  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    std::vector<FreeSpaceObstacle> obs;
    for (size_t i = 0; i < c.boxes.size(); ++i)
      obs.push_back({static_cast<int64_t>(i), c.boxes[i].Polygon()});
    ObstacleFreeSpace fs(Room(10, 10), obs, kRadius);
    auto grid = GridShortestPath(fs, c.boxes, 10, 10, kRadius, c.a, c.b, step);
    auto r = fs.FindPath(c.a, c.b);
    if (std::isfinite(grid.length)) {
      ASSERT_TRUE(r.found) << "grid-connected but FindPath disconnected";
      EXPECT_LE(r.length, grid.length + 1e-6);
      // And not absurdly shorter than a grid path with a SMALLER margin
      // would allow: the grid over-estimates by at most ~3% + margins.
      EXPECT_GE(r.length, 0.9 * grid.length - 0.5);
    }
    if (r.found) {
      for (size_t i = 1; i < r.points.size(); ++i)
        EXPECT_TRUE(fs.SegmentFree(r.points[i - 1], r.points[i]))
            << r.points[i - 1].transpose() << " -> " << r.points[i].transpose();
      for (const auto& bx : c.boxes)
        EXPECT_GE(MinClearance(r.points, bx), kRadius - 1e-9);
    }
    if (std::string(c.name) == "gap_0.3_closed") {
      // 0.3 m < 2 r: the walls seal the room; both oracles agree.
      EXPECT_FALSE(std::isfinite(grid.length));
      EXPECT_FALSE(r.found);
    } else {
      EXPECT_TRUE(std::isfinite(grid.length)) << "oracle case must connect";
    }
  }
}

// ---------------------------------------------------------------------------
// Regression cases (door band, thin walls, rotated gaps).
// ---------------------------------------------------------------------------

// Door of width w between two wall slabs; start/goal need a bend. Whenever
// the exact test passes a polyline through the door, the search must find a
// path; the search is conservative only for doors narrower than
// 2 r / cos(pi/64) * 1.0001 (+2e-6) ~ 0.40053 m at r = 0.2.
TEST(ObstacleFreeSpaceTest, R2_1_ReReview_NarrowDoors) {
  const double r = 0.2;
  for (double w : {0.401, 0.405, 0.409, 0.412, 0.42, 0.5}) {
    SCOPED_TRACE(w);
    const double y0 = 5 - w / 2, y1 = 5 + w / 2;
    ObstacleFreeSpace fs(Room(10, 10),
                         {{1, Poly({{4, 0}, {4.2, 0}, {4.2, y0}, {4, y0}})},
                          {2, Poly({{4, y1}, {4.2, y1}, {4.2, 10}, {4, 10}})}},
                         r);
    const Eigen::Vector2d a(2, 2), d1(3.7, 5), d2(4.5, 5), b(8, 8);
    ASSERT_TRUE(fs.IsFree(d1) && fs.IsFree(d2) && fs.SegmentFree(a, d1) &&
                fs.SegmentFree(d1, d2) && fs.SegmentFree(d2, b));
    auto res = fs.FindPath(a, b);
    ASSERT_TRUE(res.found);
    EXPECT_LE(res.length, (d1 - a).norm() + (d2 - d1).norm() +
                              (b - d2).norm() + 1e-9);
    for (size_t i = 1; i < res.points.size(); ++i)
      EXPECT_TRUE(fs.SegmentFree(res.points[i - 1], res.points[i]));
  }
  // Never more permissive than the exact test: a door narrower than 2 r.
  ObstacleFreeSpace closed(
      Room(10, 10),
      {{1, Poly({{4, 0}, {4.2, 0}, {4.2, 4.80005}, {4, 4.80005}})},
       {2, Poly({{4, 5.19995}, {4.2, 5.19995}, {4.2, 10}, {4, 10}})}},
      r);
  EXPECT_FALSE(closed.FindPath({2, 2}, {8, 8}).found);
}

TEST(ObstacleFreeSpaceTest, R2_1_ReReview_ThinWallRadius0) {
  // 1 mm thick wall with a 2 cm door, radius 0 (the point geodesic).
  ObstacleFreeSpace fs(Room(10, 10),
                       {{1, Poly({{5, 0}, {5.001, 0}, {5.001, 4.99}, {5, 4.99}})},
                        {2, Poly({{5, 5.01}, {5.001, 5.01}, {5.001, 10}, {5, 10}})}},
                       0.0);
  auto r = fs.FindPath({2, 1}, {8, 1});
  ASSERT_TRUE(r.found);
  EXPECT_GT(r.length, 2 * std::hypot(3.0, 3.99) - 1e-6);  // via the door
  ObstacleFreeSpace sealed(Room(10, 10),
                           {{1, Poly({{5, 0}, {5.001, 0}, {5.001, 10}, {5, 10}})}},
                           0.0);
  EXPECT_FALSE(sealed.FindPath({2, 1}, {8, 9}).found);
}

TEST(ObstacleFreeSpaceTest, R2_1_ReReview_RotatedGap) {
  // Two 45-degree rotated slabs with parallel faces g apart.
  const double s2 = std::sqrt(0.5);
  const Eigen::Vector2d n(s2, s2), t(-s2, s2), c(5, 5);
  for (double g : {0.405, 0.41, 0.45}) {
    SCOPED_TRACE(g);
    auto slab = [&](double off) {
      const Eigen::Vector2d o = c + n * off;
      const Eigen::Vector2d d = n * (off > 0 ? 1.0 : -1.0);
      return Poly({o - t * 3, o + t * 3, o + t * 3 + d, o - t * 3 + d});
    };
    ObstacleFreeSpace fs(Room(10, 10), {{1, slab(g / 2)}, {2, slab(-g / 2)}},
                         0.2);
    const Eigen::Vector2d p0 = c - t * 2.0, p1 = c + t * 2.0;
    ASSERT_TRUE(fs.SegmentFree(p0, p1));  // straight along the slot
    auto r = fs.FindPath(p0, p1);
    ASSERT_TRUE(r.found);
    EXPECT_NEAR(r.length, 4.0, 1e-9);
  }
}

// Grid oracle with a margin tight enough to see the door band: a 0.405 m
// door (0.005 m of exact free width at r = 0.2) in a 1 m room; BFS on a
// 0.5 mm grid (clearance r + 1 mm) connects, and so must FindPath, with the
// straight-through length.
TEST(ObstacleFreeSpaceTest, R2_1_ReReview_GridOracleDoorBand) {
  const double r = 0.2, w = 0.405, step = 0.0005, margin = 0.001;
  const double x0 = 0.5 - w / 2, x1 = 0.5 + w / 2;
  const std::vector<Box> walls = {{0, 0.45, x0, 0.55}, {x1, 0.45, 1, 0.55}};
  std::vector<FreeSpaceObstacle> obs;
  for (size_t i = 0; i < walls.size(); ++i)
    obs.push_back({static_cast<int64_t>(i), walls[i].Polygon()});
  ObstacleFreeSpace fs(Room(1, 1), obs, r);
  const int nx = static_cast<int>(1.0 / step) + 1;
  std::vector<char> ok(nx * nx, 0), seen(nx * nx, 0);
  for (int j = 0; j < nx; ++j)
    for (int i = 0; i < nx; ++i) {
      const Eigen::Vector2d p(i * step, j * step);
      bool free = true;
      for (const auto& bx : walls)
        if (bx.Distance(p) < r + margin) free = false;
      ok[j * nx + i] = free;
    }
  const Eigen::Vector2d a(0.5, 0.2), b(0.5, 0.8);
  auto idx = [&](const Eigen::Vector2d& p) {
    return static_cast<int>(std::lround(p.y() / step)) * nx +
           static_cast<int>(std::lround(p.x() / step));
  };
  std::vector<int> queue = {idx(a)};
  seen[idx(a)] = 1;
  for (size_t q = 0; q < queue.size(); ++q) {
    const int u = queue[q], ui = u % nx, uj = u / nx;
    const int nb[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (const auto& d : nb) {
      const int vi = ui + d[0], vj = uj + d[1];
      if (vi < 0 || vj < 0 || vi >= nx || vj >= nx) continue;
      const int v = vj * nx + vi;
      if (ok[v] && !seen[v]) {
        seen[v] = 1;
        queue.push_back(v);
      }
    }
  }
  ASSERT_TRUE(seen[idx(b)]) << "oracle: the door is open";
  auto res = fs.FindPath(a, b);
  ASSERT_TRUE(res.found) << "grid-connected through the door band";
  EXPECT_NEAR(res.length, 0.6, 1e-6);
}

// NearestFree must consider every candidate, including the
// configuration-space boundary, and keep the nearest. Brute-force oracle: a
// 1 cm grid of exactly-free points around the query.
double BruteForceNearestFree(const ObstacleFreeSpace& fs,
                             const Eigen::Vector2d& p, double reach,
                             double step) {
  double best = std::numeric_limits<double>::infinity();
  const int n = static_cast<int>(reach / step);
  for (int i = -n; i <= n; ++i)
    for (int j = -n; j <= n; ++j) {
      const Eigen::Vector2d c = p + Eigen::Vector2d(i, j) * step;
      if ((c - p).norm() < best && fs.IsFree(c)) best = (c - p).norm();
    }
  return best;
}

TEST(ObstacleFreeSpaceTest, R2_1_DeltaReview_NearestFreeTwoRings) {
  ObstacleFreeSpace fs(Room(10, 10),
                       {{1, Poly({{4, 4}, {4.85, 4}, {4.85, 4.6}, {4, 4.6}})},
                        {2, Poly({{5.15, 4}, {6, 4}, {6, 4.6}, {5.15, 4.6}})}},
                       kRadius);
  const Eigen::Vector2d p(5.0, 4.5);
  auto q = fs.NearestFree(p);
  ASSERT_TRUE(q.has_value());
  EXPECT_TRUE(fs.IsFree(*q));
  const double oracle = BruteForceNearestFree(fs, p, 2.0, 0.01);
  EXPECT_LE((*q - p).norm(), oracle + 0.01) << q->transpose();
  EXPECT_LT((*q - p).norm(), 0.3);  // not the 4.5 m walkable boundary
}

TEST(ObstacleFreeSpaceTest, R2_1_DeltaReview_NearestFreeCornerWallObstacle) {
  ObstacleFreeSpace fs(Room(10, 10),
                       {{1, Poly({{0.25, 3}, {2, 3}, {2, 5}, {0.25, 5}})}},
                       kRadius);
  const Eigen::Vector2d p(0.1, 4.0);
  auto q = fs.NearestFree(p);
  ASSERT_TRUE(q.has_value());
  EXPECT_TRUE(fs.IsFree(*q));
  EXPECT_LE((*q - p).norm(), BruteForceNearestFree(fs, p, 2.5, 0.01) + 0.01);
}

// Random points in multi-obstacle layouts (overlapping rings, wall contact,
// narrow gaps) against the brute-force oracle.
TEST(ObstacleFreeSpaceTest, R2_1_DeltaReview_NearestFreeRandomOracle) {
  const std::vector<std::vector<Box>> layouts = {
      {{4, 4, 4.85, 4.6}, {5.15, 4, 6, 4.6}, {4.5, 4.8, 5.5, 5.4}},
      {{0.1, 3, 2, 5}, {2.2, 3, 3, 5}, {0, 5.2, 3, 5.6}},
      {{3, 3, 7, 3.5}, {3, 3.8, 3.5, 7}, {6.5, 3.8, 7, 7}, {3.9, 6.9, 6.1, 7.3}},
  };
  uint32_t seed = 12345;
  auto rnd = [&]() {
    seed = seed * 1664525u + 1013904223u;
    return (seed >> 8) / static_cast<double>(1u << 24);
  };
  int checked = 0;
  for (const auto& boxes : layouts) {
    std::vector<FreeSpaceObstacle> obs;
    for (size_t i = 0; i < boxes.size(); ++i)
      obs.push_back({static_cast<int64_t>(i), boxes[i].Polygon()});
    ObstacleFreeSpace fs(Room(10, 10), obs, kRadius);
    for (int k = 0; k < 40; ++k) {
      // Sample near the obstacles, where constraints bind.
      const Box& b = boxes[k % boxes.size()];
      const Eigen::Vector2d p(b.x0 - 0.3 + rnd() * (b.x1 - b.x0 + 0.6),
                              b.y0 - 0.3 + rnd() * (b.y1 - b.y0 + 0.6));
      if (p.x() < 0 || p.y() < 0 || p.x() > 10 || p.y() > 10) continue;
      auto q = fs.NearestFree(p);
      ASSERT_TRUE(q.has_value());
      EXPECT_TRUE(fs.IsFree(*q));
      const double oracle = BruteForceNearestFree(fs, p, 2.0, 0.01);
      // Oracle grid step (0.01 * sqrt(2) / 2) + buffer approximation slack.
      EXPECT_LE((*q - p).norm(), oracle + 0.008 + 1e-3 * kRadius)
          << "p=" << p.transpose() << " got " << q->transpose();
      ++checked;
    }
  }
  EXPECT_GT(checked, 100);
}

// F does not extend outside the navigable domain just because a footprint is
// exempt. An empty erosion is an empty domain, distinct from a search that
// found no candidate in a domain that still has area.
TEST(ObstacleFreeSpace, ExemptFootprintOutsideWalkableIsNotFree) {
  Box seat{12, 4, 14, 6};  // entirely outside the 10x10 room
  ObstacleFreeSpace fs(Room(10, 10), {{7, seat.Polygon()}}, kRadius);
  EXPECT_FALSE(fs.IsFree({13, 5}, /*exempt=*/{7}));
  EXPECT_FALSE(fs.SegmentFree({9, 5}, {13, 5}, /*exempt=*/{7}));
}

TEST(ObstacleFreeSpace, EmptyErosionIsDomainEmptyNotSearchMiss) {
  auto sliver = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {0, 0}, {0.1, 0}, {0.1, 0.1}, {0, 0.1}});
  auto gone = std::make_shared<Polygon2D>(sliver->Inflate(-1.0));
  ASSERT_EQ(gone->NumComponents(), 0u);
  ObstacleFreeSpace fs(gone, {}, 0.0);
  EXPECT_TRUE(fs.DomainEmpty());
  ObstacleFreeSpace::NearestFreeStatus status =
      ObstacleFreeSpace::NearestFreeStatus::kAlreadyFree;
  EXPECT_FALSE(fs.NearestFree({0, 0}, {}, {}, nullptr, &status).has_value());
  EXPECT_EQ(status, ObstacleFreeSpace::NearestFreeStatus::kDomainEmpty);
  auto path = fs.FindPath({0, 0}, {1, 0});
  EXPECT_FALSE(path.found);
  EXPECT_TRUE(path.domain_empty);
  EXPECT_FALSE(path.search_found_none);
}

// The contact is the geodesic nearest point of the ring, not the Euclidean
// nearest. A barrier between the anchor and the ring's near side makes those
// two points differ; the selected point is the one the free path reaches.
TEST(ObstacleFreeSpaceTest, ClosestOnRingIsGeodesicNotEuclidean) {
  auto room = Room(10, 10);
  const Box barrier{0, 4, 7, 6};
  ObstacleFreeSpace fs(room, {{1, barrier.Polygon()}}, /*clearance=*/0.0);
  const Polygon2D ring(std::vector<Eigen::Vector2d>{
      {3, 1}, {4, 1}, {4, 2}, {3, 2}});
  const Eigen::Vector2d anchor(3.5, 8);
  const Eigen::Vector2d eu = ring.GetNearestPoint(anchor);
  EXPECT_NEAR(eu.x(), 3.5, 1e-9);
  EXPECT_NEAR(eu.y(), 2.0, 1e-9);
  double length = 0.0;
  const Eigen::Vector2d contact = fs.ClosestOnRing(anchor, ring, &length);
  EXPECT_GT((contact - eu).norm(), 0.1);
  EXPECT_NEAR(ring.GetDistance(contact), 0.0, 1e-6);
  EXPECT_TRUE(std::isfinite(length));
  auto via_contact = fs.FindPath(anchor, contact);
  auto via_eu = fs.FindPath(anchor, eu);
  ASSERT_TRUE(via_contact.found);
  ASSERT_TRUE(via_eu.found);
  EXPECT_LT(via_contact.length, via_eu.length - 0.1);

  ObstacleFreeSpace open(room, {}, 0.0);
  const Eigen::Vector2d seen = open.ClosestOnRing(anchor, ring);
  EXPECT_NEAR((seen - eu).norm(), 0.0, 1e-6);
}

// v1's approach direction is the direction of the final straight leg onto the
// contact. An anchor that sees the contact starts the leg itself; one that goes
// round a barrier starts it at the last waypoint of the geodesic, the corner it
// turned at. The leg is free and, with the path to its start, is the geodesic.
TEST(ObstacleFreeSpaceTest, ClosestOnRingReportsWhereTheFinalLegStarts) {
  auto room = Room(10, 10);
  const Box barrier{0, 4, 7, 6};
  ObstacleFreeSpace fs(room, {{1, barrier.Polygon()}}, /*clearance=*/0.0);
  const Polygon2D ring(std::vector<Eigen::Vector2d>{
      {3, 1}, {4, 1}, {4, 2}, {3, 2}});

  // Behind the barrier: the geodesic turns at its east end.
  const Eigen::Vector2d anchor(3.5, 8);
  double length = 0.0;
  Eigen::Vector2d origin = anchor;
  const Eigen::Vector2d contact = fs.ClosestOnRing(anchor, ring, &length, &origin);
  EXPECT_GT((origin - anchor).norm(), 1.0) << "the anchor does not see the contact";
  // The leg is the geodesic from its start (straight), and the path to its start
  // plus the leg is the whole geodesic.
  EXPECT_NEAR(fs.GeodesicDistance(origin, contact), (contact - origin).norm(), 1e-6);
  EXPECT_NEAR(fs.GeodesicDistance(anchor, origin) + (contact - origin).norm(),
              length, 1e-6);
  EXPECT_GT(fs.GeodesicDistance(anchor, contact), (anchor - contact).norm() + 0.5)
      << "the barrier is in the way";

  // In the open the anchor sees the contact: the leg starts at the anchor.
  const Eigen::Vector2d seeing(2.0, 0.2);
  Eigen::Vector2d seeing_origin(-1, -1);
  const Eigen::Vector2d c2 = fs.ClosestOnRing(seeing, ring, nullptr, &seeing_origin);
  EXPECT_TRUE(fs.SegmentFree(seeing, c2));
  EXPECT_NEAR((seeing_origin - seeing).norm(), 0.0, 1e-12);
}

// Footprints that touch (a chair against a desk: two boxes sharing an edge) are each an obstacle of the configuration
// space, and the geodesic past them is continuous. AF-DESK-01@d90 (the affordance catalogue): the chair (key 11) and the
// box beside it (key 12) share the edge x = 4.3; the configuration space was built from a union of their hair-buffered
// footprints that lost both, so the visibility graph had none of their corners while SegmentFree still blocked them,
// and the geodesic from a point below the chair's corner to a contact inside the chair (its own footprint, open) jumped
// by 0.50 m across the hair band of x = 4.3, between points 1e-13 m apart.
TEST(ObstacleFreeSpace, TouchingFootprintsKeepTheirCornersAndTheGeodesicIsContinuous) {
  // The scene's footprints with their exact coordinates (as the scene transforms produce them): which of them
  // bg::buffer drops from a multipolygon whose members touch depends on the last bits (5.075 written as a literal is a
  // different double, and the old code then happened to keep them).
  const auto poly = [](std::vector<Eigen::Vector2d> pts) { return std::make_shared<Polygon2D>(std::move(pts)); };
  const auto desk = poly({{5.0749999999999993, 2.9250000000000003}, {4.625, 2.9250000000000003},
                          {4.625, 2.4750000000000001}, {5.0749999999999993, 2.4750000000000001}});
  const auto chair = poly({{3.7000000000000002, 2.2000000000000002}, {4.2999999999999998, 2.2000000000000002},
                           {4.2999999999999998, 3.7999999999999998}, {3.7000000000000002, 3.7999999999999998}});
  const auto side = poly({{4.2999999999999998, 3.2000000000000002}, {4.8999999999999995, 3.2000000000000002},
                          {4.8999999999999995, 3.7999999999999998}, {4.2999999999999998, 3.7999999999999998}});
  ObstacleFreeSpace fs(Room(8.0, 6.0), {{10, desk}, {11, chair}, {12, side}}, 0.0);
  const Eigen::Vector2d goal(4.2893859368188769, 2.8029646693651991);  // the mapped contact, 10.6 mm inside the chair
  const std::vector<int64_t> own = {11};
  // The analytic geodesic for a start below the chair, left of its right edge: round the corner (4.3, 2.2), then up the
  // edge to the contact's foot, then the foot's depth: |start - corner| + |corner - foot| + depth, within the hair.
  const Eigen::Vector2d corner(4.3, 2.2), foot(4.3, goal.y());
  double prev = std::numeric_limits<double>::quiet_NaN();
  double lo = std::numeric_limits<double>::infinity(), hi = -lo;
  for (int k = -40; k <= 40; ++k) {
    const Eigen::Vector2d start(4.3 + k * 1e-7, 1.2513465907);
    const double d = fs.GeodesicDistance(start, goal, own);
    ASSERT_TRUE(std::isfinite(d)) << k;
    lo = std::min(lo, d);
    hi = std::max(hi, d);
    const double oracle = (start - corner).norm() + (corner - foot).norm() + (foot - goal).norm();
    EXPECT_LE(d, oracle + 1e-5) << "start x = 4.3 + " << k << "e-7: the route does not round the chair's corner";
    if (std::isfinite(prev)) EXPECT_NEAR(d, prev, 1e-5) << "start x = 4.3 + " << k << "e-7";
    prev = d;
  }
  EXPECT_LT(hi - lo, 1e-4) << "the geodesic jumps across the hair band of the chair's edge";
}

// A footprint that is not a valid polygon for the union (zero area: a collinear ring) still blocks, as SegmentFree blocks
// it: its ring is buffered on its own. Unioning it would drop it, and the geodesic across it would be inf.
TEST(ObstacleFreeSpace, ADegenerateFootprintStillBlocksTheGeodesic) {
  const auto wall = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{{4, 1}, {4, 5}, {4, 3}});  // zero area
  const Box other{1, 1, 2, 2};
  // Around the segment x = 4, y in [1, 5]: over its top end, (3, 3) -> (4, 5) -> (5, 3) at clearance 0.
  ObstacleFreeSpace c0(Room(8.0, 6.0), {{1, wall}, {2, other.Polygon()}}, 0.0);
  EXPECT_FALSE(c0.SegmentFree({3.0, 3.0}, {5.0, 3.0}));
  EXPECT_NEAR(c0.GeodesicDistance({3.0, 3.0}, {5.0, 3.0}), 2.0 * std::hypot(1.0, 2.0), 1e-5);
  // With a clearance ring the route rounds the end's ring: longer than at clearance 0, finite.
  ObstacleFreeSpace c2(Room(8.0, 6.0), {{1, wall}, {2, other.Polygon()}}, kRadius);
  const double d2 = c2.GeodesicDistance({3.0, 3.0}, {5.0, 3.0});
  EXPECT_TRUE(std::isfinite(d2));
  EXPECT_GT(d2, 2.0 * std::hypot(1.0, 2.0));
  EXPECT_LT(d2, 2.0 * std::hypot(1.0, 2.0 + kRadius) + 2.0 * kRadius);
  // A self-intersecting ring (a bowtie) blocks too.
  const auto bowtie = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{{3.8, 1}, {4.2, 5}, {4.2, 1}, {3.8, 5}});
  ObstacleFreeSpace bt(Room(8.0, 6.0), {{1, bowtie}, {2, other.Polygon()}}, kRadius);
  EXPECT_TRUE(std::isfinite(bt.GeodesicDistance({3.0, 3.0}, {5.0, 3.0})));
}

// At clearance 0 a ringed footprint and a bare one that touch are one obstacle of the configuration space (both are
// buffered by the same hair, so two buffers would overlap in a sliver, and the geodesic would be inf everywhere).
TEST(ObstacleFreeSpace, AtClearanceZeroARingedAndABareFootprintThatTouchAreBothBlocked) {
  // The AF-DESK-01 footprints with their exact coordinates (the box beside the chair bare).
  const auto poly = [](std::vector<Eigen::Vector2d> pts) { return std::make_shared<Polygon2D>(std::move(pts)); };
  const auto desk = poly({{5.0749999999999993, 2.9250000000000003}, {4.625, 2.9250000000000003},
                          {4.625, 2.4750000000000001}, {5.0749999999999993, 2.4750000000000001}});
  const auto chair = poly({{3.7000000000000002, 2.2000000000000002}, {4.2999999999999998, 2.2000000000000002},
                           {4.2999999999999998, 3.7999999999999998}, {3.7000000000000002, 3.7999999999999998}});
  const auto side = poly({{4.2999999999999998, 3.2000000000000002}, {4.8999999999999995, 3.2000000000000002},
                          {4.8999999999999995, 3.7999999999999998}, {4.2999999999999998, 3.7999999999999998}});
  ObstacleFreeSpace fs(Room(8.0, 6.0), {{10, desk}, {11, chair}, {12, side}}, 0.0, /*bare_keys=*/{12});
  const Eigen::Vector2d goal(4.3, 4.5);
  double prev = std::numeric_limits<double>::quiet_NaN();
  Eigen::Vector2d prev_s = Eigen::Vector2d::Zero();
  for (int k = 0; k <= 400; ++k) {
    const Eigen::Vector2d s(3.5 + 1.7 * k / 400.0, 1.5);
    const double d = fs.GeodesicDistance(s, goal);
    ASSERT_TRUE(std::isfinite(d)) << "start " << s.transpose();
    // The block is in the way (the straight line crosses it): no shorter than the straight line, and no longer than the
    // feasible path round its left side, through its corners (3.7, 2.2) and (3.7, 3.8).
    const Eigen::Vector2d lo(3.7, 2.2), hi(3.7, 3.8);
    EXPECT_GT(d, (s - goal).norm()) << "start " << s.transpose();
    EXPECT_LE(d, (s - lo).norm() + (lo - hi).norm() + (hi - goal).norm() + 1e-5) << "start " << s.transpose();
    if (std::isfinite(prev)) EXPECT_LE(std::fabs(d - prev), (s - prev_s).norm() + 1e-6) << "start " << s.transpose();
    prev = d;
    prev_s = s;
  }
}

}  // namespace

// ---- Stage 2 interface of the active-piece solver: every route, and the cone of F ----

// The routes of GeodesicRoutes: the shortest is the distance GeodesicDistance returns, to the bit, and a
// symmetric pair of detours around a box is listed as two routes whose error intervals overlap.
TEST(ObstacleFreeSpaceRoutes, ShortestRouteIsTheGeodesicAndASymmetricPairTies) {
  const Box box{4.0, 4.0, 6.0, 6.0};
  ObstacleFreeSpace fs(Room(10, 10), {{1, box.Polygon()}}, /*clearance=*/0.0);
  // Diagonal through the box's centre: the two detours (round the lower-right and the upper-left corner) tie.
  const Eigen::Vector2d from(2.0, 2.0), to(8.0, 8.0);
  const auto routes = fs.GeodesicRoutes(from, to);
  ASSERT_GE(routes.size(), 2u);
  EXPECT_EQ(routes[0].value, fs.GeodesicDistance(from, to));
  EXPECT_LE(routes[1].value - routes[0].value, routes[0].err + routes[1].err);
  EXPECT_GE(routes[0].legs, 2);
  EXPECT_GT(routes[0].err, 0.0);
  // The first waypoints differ: the two routes leave in different directions.
  EXPECT_GT((routes[0].first - routes[1].first).norm(), 1.0);
}

TEST(ObstacleFreeSpaceRoutes, AVisiblePairHasOneRouteAndNoAlternative) {
  ObstacleFreeSpace fs(Room(10, 10), {}, 0.0);
  const auto routes = fs.GeodesicRoutes({1, 1}, {4, 5}, /*slack=*/100.0);
  ASSERT_EQ(routes.size(), 1u);
  EXPECT_EQ(routes[0].node, -1);
  EXPECT_DOUBLE_EQ(routes[0].value, 5.0);
  EXPECT_EQ(routes[0].legs, 1);
}

// A slack lists the routes within it and no others; the slack-free list is a prefix of it.
TEST(ObstacleFreeSpaceRoutes, SlackListsExactlyTheRoutesWithinIt) {
  const Box box{4.0, 4.0, 6.0, 6.0};
  ObstacleFreeSpace fs(Room(10, 10), {{1, box.Polygon()}}, 0.0);
  const Eigen::Vector2d from(2.0, 3.0), to(8.0, 7.0);
  const auto all = fs.GeodesicRoutes(from, to, /*slack=*/100.0);
  const auto few = fs.GeodesicRoutes(from, to, /*slack=*/0.0);
  ASSERT_GE(all.size(), few.size());
  for (size_t i = 0; i < few.size(); ++i) EXPECT_EQ(few[i].node, all[i].node);
  const double slack = 0.5 * (all.back().value - all.front().value);
  const auto mid = fs.GeodesicRoutes(from, to, slack);
  for (const auto& r : mid) EXPECT_LE(r.value - all.front().value, slack + 1e-9);
  for (const auto& r : all)
    if (r.value - all.front().value <= slack - 1e-9) {
      bool found = false;
      for (const auto& m : mid) found = found || m.node == r.node;
      EXPECT_TRUE(found) << "route " << r.node << " within the slack is missing";
    }
}

// The cone of F at x, exactly.
double ConeMeasure(const ObstacleFreeSpace::FeasibleCone& c) {
  double m = 0.0;
  for (const auto& a : c.arcs) m += a.second - a.first;
  return m;
}

// The hair is a feasibility offset, not a length: every length is measured from the
// endpoints' feet on F's boundary (the snap less the hair), so a point on a footprint's boundary is at distance 0 from
// itself, its snap is 0, and a point inside is reported at its depth below the boundary, not depth + hair.
TEST(ObstacleFreeSpaceHair, ThePathFromAPointOnAFootprintToItselfHasLengthZeroAndAnInsidePointItsDepth) {
  const Box box{4.0, 4.0, 6.0, 6.0};
  ObstacleFreeSpace fs(Room(10, 10), {{1, box.Polygon()}}, 0.0);
  // The foot is computed from the hair-inflated free point: its depth carries the rounding of those coordinates, a few
  // units in the last place of |p| (gamma_8 |p|_inf bounds the projection, the difference and the norm), not the hair.
  const auto rounding = [](const Eigen::Vector2d& p) {
    constexpr double u = 1.1102230246251565e-16;
    return 8 * u / (1 - 8 * u) * std::max(1.0, p.cwiseAbs().maxCoeff());
  };
  for (const Eigen::Vector2d p : {Eigen::Vector2d(5.0, 4.0), Eigen::Vector2d(6.0, 5.0), Eigen::Vector2d(4.0, 4.0),
                                  Eigen::Vector2d(5.0, 6.0)}) {
    EXPECT_LE(fs.GeodesicDistance(p, p), 2 * rounding(p)) << p.transpose();
    const auto path = fs.FindPath(p, p);
    ASSERT_TRUE(path.found) << p.transpose();
    EXPECT_LE(path.length, 2 * rounding(p)) << p.transpose();
    EXPECT_LE(path.from_snap_m, rounding(p));
    EXPECT_LE(path.to_snap_m, rounding(p));
  }
  const double depth = 1e-3;
  const auto inside = fs.FindPath(Eigen::Vector2d(5.0, 4.0 + depth), Eigen::Vector2d(1.0, 5.0));
  ASSERT_TRUE(inside.found);
  EXPECT_NEAR(inside.from_snap_m, depth, 1e-15) << "the depth below the boundary, not depth + hair";
}

// Continuity across a footprint's boundary: from (5, 4 + depth) beside a box spanning 4..6 to
// (1, 5), the distance is 1-Lipschitz in the endpoint: at depth 0, 1e-12 inside and 5e-7 outside it moves by no more
// than the endpoint does (plus the rounding of the lengths).
TEST(ObstacleFreeSpaceHair, TheDistanceIsContinuousAcrossAFootprintBoundary) {
  const Box box{4.0, 4.0, 6.0, 6.0};
  ObstacleFreeSpace fs(Room(10, 10), {{1, box.Polygon()}}, 0.0);
  const Eigen::Vector2d to(1.0, 5.0);
  const double d0 = fs.GeodesicDistance(Eigen::Vector2d(5.0, 4.0), to);
  for (const double dy : {1e-12, -5e-7, 1e-9, -1e-12, 5e-7}) {
    const double d = fs.GeodesicDistance(Eigen::Vector2d(5.0, 4.0 + dy), to);
    EXPECT_LE(std::fabs(d - d0), std::fabs(dy) + 1e-14) << "dy = " << dy;
  }
}

// The same across a CLEARANCE RING (a body radius around the footprint): the ring is F's boundary there, and the
// c-space anchor of a point in the ring's buffer band is an attachment offset, not a length.
TEST(ObstacleFreeSpaceHair, TheDistanceIsContinuousAcrossAClearanceRing) {
  const Box box{4.0, 4.0, 6.0, 6.0};
  ObstacleFreeSpace fs(Room(10, 10), {{1, box.Polygon()}}, kRadius);
  const Eigen::Vector2d to(1.0, 5.0);
  const double y0 = 4.0 - kRadius;  // on the ring below the box's bottom edge
  const double d0 = fs.GeodesicDistance(Eigen::Vector2d(5.0, y0), to);
  ASSERT_TRUE(std::isfinite(d0));
  for (const double dy : {1e-12, -1e-12, 1e-9, -1e-9, 5e-7, -5e-7, 1e-4, -1e-4}) {
    const double d = fs.GeodesicDistance(Eigen::Vector2d(5.0, y0 + dy), to);
    EXPECT_LE(std::fabs(d - d0), std::fabs(dy) + 1e-14) << "dy = " << dy;
    const auto path = fs.FindPath(Eigen::Vector2d(5.0, y0 + dy), to);
    ASSERT_TRUE(path.found);
    EXPECT_NEAR(path.length, d, 1e-12) << "dy = " << dy;
  }
}

// GeodesicDistance, FindPath and GeodesicRoutes measure the same length (the shortest route's value), whether an
// endpoint is free, on a footprint's boundary or inside it.
TEST(ObstacleFreeSpaceHair, TheThreeQueriesAgreeOnTheLength) {
  const Box box{4.0, 4.0, 6.0, 6.0};
  for (const double clearance : {0.0, kRadius}) {
  ObstacleFreeSpace fs(Room(10, 10), {{1, box.Polygon()}}, clearance);
  const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> pairs = {
      {{1.0, 5.0}, {9.0, 5.0}},        // round the box, both free
      {{1.0, 5.0}, {6.0, 5.0}},        // to its far face (on the boundary)
      {{5.0, 4.0}, {5.0, 6.0}},        // boundary to boundary, round it
      {{5.0, 4.001}, {9.0, 5.0}},      // from inside
      {{6.0, 5.0}, {4.0, 4.0}},        // face to corner
  };
  for (const auto& [a, b] : pairs) {
    const double g = fs.GeodesicDistance(a, b);
    const auto path = fs.FindPath(a, b);
    ASSERT_TRUE(path.found);
    const auto routes = fs.GeodesicRoutes(a, b, 0.0);
    ASSERT_FALSE(routes.empty()) << a.transpose() << " -> " << b.transpose();
    EXPECT_NEAR(path.length, g, 1e-12) << a.transpose() << " -> " << b.transpose();
    EXPECT_NEAR(routes.front().value, g, 1e-12) << a.transpose() << " -> " << b.transpose();
  }
  }  // with and without a clearance ring (the contacts then lie inside the ring: their feet are on it)
}

TEST(ObstacleFreeSpaceCone, InteriorWallCornerAndReflexCorner) {
  // L-shaped room: (0,0) (4,0) (4,2) (2,2) (2,4) (0,4); the reflex corner is (2,2).
  auto lroom = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{{0, 0}, {4, 0}, {4, 2}, {2, 2}, {2, 4}, {0, 4}});
  ObstacleFreeSpace fs(lroom, {}, 0.0);
  const double tol = ObstacleFreeSpace::kBoundaryHairM;
  const auto inside = fs.TangentConeAt({1.0, 1.0}, {}, {}, tol);
  EXPECT_TRUE(inside.Full());
  EXPECT_TRUE(inside.facets.empty());
  const auto wall = fs.TangentConeAt({2.0, 0.0}, {}, {}, tol);  // on the bottom edge: the upper half plane
  EXPECT_NEAR(ConeMeasure(wall), M_PI, 1e-12);
  EXPECT_TRUE(wall.Contains(M_PI / 2));
  EXPECT_TRUE(wall.Contains(0.0));
  EXPECT_TRUE(wall.Contains(M_PI));
  EXPECT_FALSE(wall.Contains(3 * M_PI / 2));
  ASSERT_EQ(wall.facets.size(), 1u);
  EXPECT_NEAR(wall.facets[0].normal.y(), -1.0, 1e-15);
  const auto corner = fs.TangentConeAt({4.0, 0.0}, {}, {}, tol);  // convex corner: a quarter turn
  EXPECT_NEAR(ConeMeasure(corner), M_PI / 2, 1e-12);
  EXPECT_TRUE(corner.Contains(3 * M_PI / 4));
  EXPECT_FALSE(corner.Contains(M_PI / 4));
  EXPECT_EQ(corner.facets.size(), 2u);
  const auto reflex = fs.TangentConeAt({2.0, 2.0}, {}, {}, tol);  // reflex corner: three quarters
  EXPECT_NEAR(ConeMeasure(reflex), 3 * M_PI / 2, 1e-12);
  EXPECT_FALSE(reflex.Contains(M_PI / 4));  // the open quadrant (0, 90 degrees) is the notch
  EXPECT_TRUE(reflex.Contains(0.0));        // along the edge towards (4, 2)
  EXPECT_TRUE(reflex.Contains(M_PI / 2));   // along the edge towards (2, 4)
  EXPECT_TRUE(reflex.Contains(M_PI));
  EXPECT_TRUE(reflex.Contains(3 * M_PI / 2));
}

TEST(ObstacleFreeSpaceCone, ClearanceRingAndFootprint) {
  const Box box{4.0, 4.0, 6.0, 6.0};
  ObstacleFreeSpace fs(Room(10, 10), {{1, box.Polygon()}}, kRadius);
  const double tol = ObstacleFreeSpace::kBoundaryHairM;
  // On the ring beside the box's left edge: x = 4 - r, y = 5. Free directions: away from the box (half plane).
  const auto side = fs.TangentConeAt({4.0 - kRadius, 5.0}, {}, {}, tol);
  EXPECT_NEAR(ConeMeasure(side), M_PI, 1e-12);
  EXPECT_TRUE(side.Contains(M_PI));
  EXPECT_FALSE(side.Contains(0.0));
  ASSERT_EQ(side.facets.size(), 1u);
  EXPECT_EQ(side.facets[0].kind, ObstacleFreeSpace::Facet::kRingEdge);
  EXPECT_NEAR(side.facets[0].normal.x(), 1.0, 1e-15);
  // On the arc round the box's corner (4, 4): the point at 225 degrees, radius r.
  const Eigen::Vector2d on_arc = Eigen::Vector2d(4.0, 4.0) + kRadius * Eigen::Vector2d(std::cos(5 * M_PI / 4), std::sin(5 * M_PI / 4));
  const auto arc = fs.TangentConeAt(on_arc, {}, {}, tol);
  EXPECT_NEAR(ConeMeasure(arc), M_PI, 1e-12);
  ASSERT_EQ(arc.facets.size(), 1u);
  EXPECT_EQ(arc.facets[0].kind, ObstacleFreeSpace::Facet::kRingArc);
  EXPECT_DOUBLE_EQ(arc.facets[0].radius, kRadius);
  // The tangent to the arc at 225 degrees points along 135 / 315 degrees.
  EXPECT_TRUE(arc.Contains(3 * M_PI / 4));
  EXPECT_TRUE(arc.Contains(7 * M_PI / 4));
  EXPECT_FALSE(arc.Contains(M_PI / 4));
  // A point beyond the ring is interior; the same point with the obstacle exempt is interior too.
  EXPECT_TRUE(fs.TangentConeAt({1.0, 1.0}, {}, {}, tol).Full());
  EXPECT_TRUE(fs.TangentConeAt({4.0 - kRadius, 5.0}, {1}, {}, tol).Full());
  // Ring-exempt: the footprint itself bounds F. At the box's corner (4,4) the free directions are the
  // three quarters outside it.
  const auto bare = fs.TangentConeAt({4.0, 4.0}, {}, {1}, tol);
  EXPECT_NEAR(ConeMeasure(bare), 3 * M_PI / 2, 1e-12);
  EXPECT_FALSE(bare.Contains(M_PI / 4));
  EXPECT_TRUE(bare.Contains(5 * M_PI / 4));
  EXPECT_EQ(bare.facets.size(), 2u);
}

// A body in contact: a point on the c-space boundary of a footprint edge
// (the edge offset by the hair), nudged across it by 1e-14..1e-12 (inside
// PointInRing's on-boundary band), keeps the distance of the free point
// 1e-10 outside: the geodesic is 1-Lipschitz there. (A roundoff-
// only Tangent bound: 1,530 of ~71k such points jumped, up to 1.23 m.)
TEST(ObstacleFreeSpace, GeodesicsAreLipschitzForABodyInContact) {
  const auto rbox = [](Eigen::Vector2d c, double hx, double hy, double yaw) {
    const Eigen::Vector2d ex(std::cos(yaw), std::sin(yaw)), ey(-std::sin(yaw), std::cos(yaw));
    std::vector<Eigen::Vector2d> p;
    for (const Eigen::Vector2d s : {Eigen::Vector2d(-1, -1), Eigen::Vector2d(1, -1),
                                    Eigen::Vector2d(1, 1), Eigen::Vector2d(-1, 1)})
      p.push_back(c + s.x() * hx * ex + s.y() * hy * ey);
    return std::make_shared<Polygon2D>(p);
  };
  int checks = 0, violations = 0;
  double worst = 0.0;
  for (int n : {10, 30}) {
    for (int seed = 1; seed <= 3; ++seed) {
      std::mt19937 rng(seed);
      std::uniform_real_distribution<double> u(0, 1);
      std::vector<FreeSpaceObstacle> obs;
      for (int i = 0; i < n; ++i) {
        const double cx = 0.5 + 9 * u(rng);
        const double cy = 0.5 + 9 * u(rng);
        const double h1 = 0.1 + 0.3 * u(rng);
        const double h2 = 0.1 + 0.3 * u(rng);
        const double yaw = 3 * u(rng);
        obs.push_back({i + 1, rbox({cx, cy}, h1, h2, yaw)});
      }
      ObstacleFreeSpace fs(Room(10, 10), obs, 0.0);
      std::vector<Eigen::Vector2d> goals;
      while (goals.size() < 2) {
        const double gx = 0.05 + 9.9 * u(rng);
        const double gy = 0.05 + 9.9 * u(rng);
        if (fs.IsFree({gx, gy})) goals.push_back({gx, gy});
      }
      for (const auto& o : obs) {
        const auto& pts = o.footprint->GetPoints();
        for (size_t k = 0; k < pts.size(); ++k) {
          const Eigen::Vector2d a = pts[k], b = pts[(k + 1) % pts.size()];
          const Eigen::Vector2d t = (b - a).normalized();
          const Eigen::Vector2d out(t.y(), -t.x());  // CCW box: the outward normal
          for (double f : {0.13, 0.5, 0.87}) {
            const Eigen::Vector2d m = a + f * (b - a) + ObstacleFreeSpace::kBoundaryHairM * out;
            if (!fs.IsFree(m + 1e-9 * out) || fs.IsFree(m - 1e-6 * out)) continue;  // not a lone contact
            const Eigen::Vector2d ref = m + 1e-10 * out;
            for (const auto& g : goals) {
              const double dref = fs.GeodesicDistance(ref, g);
              if (!std::isfinite(dref)) continue;
              for (double off : {-1e-12, -3e-13, -1e-13, -3e-14, -1e-14, 0.0, 1e-14, 1e-12}) {
                const Eigen::Vector2d q = m + off * out;
                if (!fs.IsFree(q) || !fs.SegmentFree(q, ref)) continue;
                const double d = fs.GeodesicDistance(q, g);
                ++checks;
                const double excess = std::fabs(d - dref) - (q - ref).norm();
                if (!(excess <= 1e-12 * (1 + dref))) {
                  ++violations;
                  if (std::isfinite(d)) worst = std::max(worst, excess);
                }
              }
            }
          }
        }
      }
    }
  }
  EXPECT_GT(checks, 1000);
  EXPECT_EQ(violations, 0) << "worst excess " << worst << " m over " << checks << " contact points";
}

// Frozen geometry from ma_ring6|ma_back_and_forth|crowded, frame 52 (Linux).
// These are the actual footprint coordinates, not a seed: the scene's RNG argument-evaluation order differed across
// compilers. A valid first hop crossed a tangent-admission boundary of the hair-buffered graph while remaining visible
// against the original footprint. Its admission changed d by 4.803953e-7 m between points only 9.0e-13 m apart.
TEST(ObstacleFreeSpace, CrowdedGeodesicsAreLipschitzAcrossRouteAdmission) {
  const auto poly = [](std::vector<Eigen::Vector2d> pts) { return std::make_shared<Polygon2D>(std::move(pts)); };
  const auto walkable = poly({{0, 0}, {8, 0}, {8, 5.2000000000000002},
                              {6.4000000000000004, 5.2000000000000002}, {6.4000000000000004, 7}, {0, 7}});
  const std::vector<FreeSpaceObstacle> obstacles = {
      {10, poly({{6.4499999999999993, 2.8999999999999999}, {6.4499999999999993, 3.8999999999999999},
                 {5.75, 3.8999999999999999}, {5.75, 2.8999999999999999}})},
      {11, poly({{5.5230127018922195, 5.2111404610069636}, {4.6569872981077802, 5.7111404610069636},
                 {4.4769872981077805, 5.399371315644566}, {5.3430127018922198, 4.899371315644566}})},
      {12, poly({{3.1005255888325767, 5.2247302994931895}, {2.8805255888325769, 5.6057814771583425},
                 {2.4994744111674239, 5.3857814771583419}, {2.7194744111674236, 5.0047302994931888}})},
      {13, poly({{1.9499999999999997, 3.6500000000000004}, {1.4499999999999997, 3.6500000000000004},
                 {1.4499999999999997, 3.1500000000000004}, {1.9499999999999997, 3.1500000000000004}})},
      {14, poly({{2.7084936490538896, 1.8362504626203449}, {2.4584936490538891, 1.4032377607281257},
                 {2.8915063509461083, 1.1532377607281257}, {3.1415063509461087, 1.5862504626203449}})},
      {15, poly({{4.4971796769724488, 0.76565445823614853}, {5.8828203230275511, 1.5656544582361476},
                 {5.5028203230275512, 2.223833765112321}, {4.1171796769724489, 1.4238337651123221}})},
      {16, poly({{0.60205511526072009, 0.52474299126391633}, {1.1752570087360836, 0.70205511526072018},
                 {0.99794488473928, 1.2752570087360837}, {0.4247429912639164, 1.0979448847392799}})},
      {17, poly({{2.8693000314347157, 3.4735094584320141}, {3.1473751903820713, 2.9979472237138705},
                 {3.5931336401440492, 3.2585952372029023}, {3.3150584811966937, 3.734157471921046}})},
      {18, poly({{0.90552140037659501, 5.2755097953595023}, {1.6261616971663679, 5.6530652477532843},
                 {1.3397655616408219, 6.1997097069870879}, {0.61912526485104902, 5.8221542545933058}})},
      {19, poly({{4.9336496935116489, 2.9033550360853568}, {4.9444172740281829, 3.6147617796768743},
                 {4.3943778771085302, 3.6230869657212152}, {4.3836102965919963, 2.9116802221296978}})},
  };
  ObstacleFreeSpace fs(walkable, obstacles, /*clearance=*/0.0);
  struct Contact {
    int64_t own;
    Eigen::Vector2d point;
    double oracle_distance;
  };
  // Regenerate with: python3 tools/geodesic_fixture_oracle.py
  // The independent graph uses exact binary-rational predicates and intersections, then 80-digit decimal lengths.
  // Contact 13 uses its fixed nearest boundary foot (1.95, 3.4) plus the 0.06 m depth. Other contacts are on boundaries.
  const std::vector<Contact> contacts = {
      {12, {2.7274999999999983, 5.0093638753660814}, 2.7193711974469059063},
      {13, {1.8899999999999997, 3.3999999999999999}, 3.1993709965846313296},
      {14, {2.742500000000001, 1.8166168867474504}, 3.0570486941664467656},
  };
  const Eigen::Vector2d stopped(4.9429852207169187, 3.5144917199490444);
  // The middle pair is the measured failure. The outer samples bracket the same boundary by 10 nm so that a change
  // in floating-point evaluation cannot move the admission comparison out of the fixture's sampled interval.
  const std::vector<Eigen::Vector2d> samples = {
      {4.9429852307158688, 3.514491719804135},
      {4.9429852207168183, 3.5144917199490457},
      {4.9429852207159186, 3.5144917199490591},
      {4.9429852107179686, 3.5144917200939538},
  };
  for (const auto& contact : contacts) {
    SCOPED_TRACE(contact.own);
    const std::vector<int64_t> own = {contact.own};
    const double at_stop_distance = fs.GeodesicDistance(stopped, contact.point, own);
    // The independent oracle measures the raw-domain lower bound. The base's
    // hair-offset graph is deliberately retained, including its longer route.
    EXPECT_GE(at_stop_distance + 2e-13, contact.oracle_distance);
    EXPECT_LE(at_stop_distance, contact.oracle_distance + 16 * ObstacleFreeSpace::kBoundaryHairM);
    const auto at_stop = fs.GeodesicRoutes(stopped, contact.point, /*slack=*/0.0, own);
    ASSERT_FALSE(at_stop.empty());
    const Eigen::Vector2d gradient = (stopped - at_stop.front().first).normalized();
    // There is a unique first hop here (the next route is at least 0.04 m longer). Check each derivative against the
    // scalar distance, away from any tie, rather than just checking that two route consumers agree with one another.
    constexpr double h = 1e-6;
    for (int axis = 0; axis < 2; ++axis) {
      const Eigen::Vector2d step = h * Eigen::Vector2d::Unit(axis);
      ASSERT_TRUE(fs.SegmentFree(stopped - step, stopped + step));
      const double fd = (fs.GeodesicDistance(stopped + step, contact.point, own) -
                         fs.GeodesicDistance(stopped - step, contact.point, own)) / (2.0 * h);
      EXPECT_NEAR(fd, gradient[axis], 2e-8) << "axis " << axis;
    }
    std::vector<double> distances;
    for (size_t i = 0; i < samples.size(); ++i) {
      SCOPED_TRACE(i);
      ASSERT_TRUE(fs.IsFree(samples[i]));
      const double d = fs.GeodesicDistance(samples[i], contact.point, own);
      ASSERT_TRUE(std::isfinite(d));
      distances.push_back(d);
      const double rounding = 64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, d);
      const auto routes = fs.GeodesicRoutes(samples[i], contact.point, /*slack=*/0.0, own);
      ASSERT_FALSE(routes.empty());
      EXPECT_NEAR(routes.front().value, d, rounding);
      if (i > 0) {
        ASSERT_TRUE(fs.SegmentFree(samples[i - 1], samples[i]));
        EXPECT_LE(std::fabs(d - distances[i - 1]), (samples[i] - samples[i - 1]).norm() + rounding)
            << "a geodesic distance is 1-Lipschitz along a free segment";
      }
    }
    // The field's previous winning hop is only a search hint: reversing query order cannot change any distance.
    for (size_t i = samples.size(); i-- > 0;)
      EXPECT_DOUBLE_EQ(fs.GeodesicDistance(samples[i], contact.point, own), distances[i]);
  }
}

// A regression case. The bottom edge of
// b27 crosses b2's protruding corner. Rounded intersection parameters in the
// old union visibility test missed both crossings and admitted the chord.
TEST(ObstacleFreeSpaceMetric, OverlappingRotatedBoxesDoNotAdmitAUnionBoundaryShortcut) {
  using V = Eigen::Vector2d;
  const std::vector<V> b2 = {{1.9983397740535951, 4.3954640574203383},
      {2.0288882955108405, 4.723390439782448}, {1.6183228797003517, 4.7616373363088771},
      {1.5877743582431063, 4.4337109539467674}};
  const std::vector<V> b27 = {{2.2493761768193705, 4.5652252814970451},
      {1.9393910806244168, 5.1073121407981708}, {1.5886794214242841, 4.9067623847055701},
      {1.898664517619238, 4.3646755254044445}};
  ObstacleFreeSpace fs(Room(10, 10), {{3, std::make_shared<Polygon2D>(b2)},
                                    {28, std::make_shared<Polygon2D>(b27)}}, 0.0);
  const V from = b27[3], to = b27[0];
  const ConvexBoxOracle protrusion(b2);
  ASSERT_TRUE(protrusion.EntersInterior(from, to));
  ASSERT_FALSE(fs.SegmentFree(from, to));
  // Independent raw-domain taut string: these two legs skirt b2[0].
  const double raw_reference = (from - b2[0]).norm() + (b2[0] - to).norm();
  EXPECT_NEAR(raw_reference, 0.40737014796500287, 2e-16);
  const double distance = fs.GeodesicDistance(from, to);
  EXPECT_GE(distance, raw_reference - 2e-13);
  // Preserve the base architecture's hair-offset detour, rather than the
  // shorter raw-domain metric of the rejected separate-domain implementation.
  EXPECT_NEAR(distance, 0.407370551217980, 2e-12);
  EXPECT_LE(distance, raw_reference + 1e-5);
  EXPECT_NEAR(fs.GeodesicDistance(to, from), distance, 2e-12);
  std::vector<V> reversed_b2(b2.rbegin(), b2.rend()), reversed_b27(b27.rbegin(), b27.rend());
  ObstacleFreeSpace reversed(Room(10, 10), {{3, std::make_shared<Polygon2D>(reversed_b2)},
                                          {28, std::make_shared<Polygon2D>(reversed_b27)}}, 0.0);
  EXPECT_NEAR(reversed.GeodesicDistance(from, to), distance, 2e-12);
  EXPECT_NEAR(reversed.GeodesicDistance(to, from), distance, 2e-12);
  const auto routes = fs.GeodesicRoutes(from, to);
  ASSERT_FALSE(routes.empty());
  EXPECT_NEAR(routes.front().value, distance, 2e-12);

  // Also exercise the free endpoint pair that originally failed realization.
  const V a(1, 4), b(3, 4.9);
  const auto path = fs.FindPath(a, b);
  ASSERT_TRUE(path.found);
  EXPECT_NEAR(path.length, fs.GeodesicDistance(a, b), 2e-12);
  const ConvexBoxOracle other(b27);
  for (size_t i = 1; i < path.points.size(); ++i) {
    EXPECT_FALSE(protrusion.EntersInterior(path.points[i - 1], path.points[i]));
    EXPECT_FALSE(other.EntersInterior(path.points[i - 1], path.points[i]));
    EXPECT_TRUE(fs.SegmentFree(path.points[i - 1], path.points[i]));
  }
}

// probe2.cc's overlapping scenes, with invariant assertions instead of timing
// output. Draw x and y separately, and exercise both coordinate orders so that
// Clang/GCC's old argument-evaluation order cannot hide the reported failures.
TEST(ObstacleFreeSpaceMetric, RandomOverlappingScenesHaveCertifiedCoursesAndConsistentDistances) {
  using V = Eigen::Vector2d;
  const V goal(9.95, 9.95);
  for (int count : {30, 60, 120}) for (int seed = 1; seed <= 5; ++seed) {
    SCOPED_TRACE(::testing::Message() << "boxes=" << count << " seed=" << seed);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<FreeSpaceObstacle> obstacles;
    std::vector<ConvexBoxOracle> oracles;
    for (int i = 0; i < count; ++i) {
      const double cx = 0.5 + 9 * uniform(rng), cy = 0.5 + 9 * uniform(rng);
      const double hx = 0.1 + 0.3 * uniform(rng), hy = 0.1 + 0.3 * uniform(rng);
      const double yaw = 3 * uniform(rng);
      const V ex(std::cos(yaw), std::sin(yaw)), ey(-std::sin(yaw), std::cos(yaw));
      std::vector<V> points;
      for (const V& sign : {V(-1, -1), V(1, -1), V(1, 1), V(-1, 1)})
        points.push_back(V(cx, cy) + sign.x() * hx * ex + sign.y() * hy * ey);
      obstacles.push_back({i + 1, std::make_shared<Polygon2D>(points)});
      oracles.emplace_back(points);
    }
    ObstacleFreeSpace fs(Room(10, 10), obstacles, 0.0);
    fs.GeodesicDistance({0.05, 0.05}, goal);
    fs.FindPath({0.05, 0.05}, goal);
    for (int query = 0; query < 100; ++query) {
      const double x = 0.05 + 9.9 * uniform(rng), y = 0.05 + 9.9 * uniform(rng);
      for (const V& from : {V(x, y), V(y, x)}) {
        SCOPED_TRACE(::testing::Message() << "query=" << query << " from=" << from.transpose());
        if (!fs.IsFree(from)) continue;
        const double distance = fs.GeodesicDistance(from, goal);
        const auto path = fs.FindPath(from, goal);
        ASSERT_EQ(std::isfinite(distance), path.found);
        if (!path.found) continue;
        EXPECT_NEAR(path.length, distance, 2e-12);
        EXPECT_EQ(path.points.front(), from);
        EXPECT_EQ(path.points.back(), goal);
        double course_length = 0.0;
        for (const auto& oracle : oracles) {
          ASSERT_FALSE(oracle.EntersInterior(from, from));
          EXPECT_GE(distance + 2e-12, oracle.DistanceLowerBound(from, goal));
        }
        for (size_t i = 1; i < path.points.size(); ++i) {
          const auto& a = path.points[i - 1];
          const auto& b = path.points[i];
          EXPECT_TRUE(fs.SegmentFree(a, b));
          for (const auto& oracle : oracles) EXPECT_FALSE(oracle.EntersInterior(a, b));
          course_length += (b - a).norm();
        }
        // Endpoint feet can shorten the reported metric by feasibility hair;
        // a centimetre-short metric cannot borrow a safe executable detour.
        const double offset_bound = 2.0 * path.points.size() * ObstacleFreeSpace::kBoundaryHairM;
        EXPECT_LE(course_length, distance + offset_bound + 2e-12);
        EXPECT_GE(course_length + 2e-12, distance);
      }
    }
  }
}

// Reviewer probes12/13: rotating a valid room with a flush furnishing must
// not lose the room's hole or make its free interior unavailable. Reproduce
// all 200 angles; the query stays below the central obstacle and is straight.
TEST(ObstacleFreeSpace, RotatedRoomsWithFlushFurnitureKeepTheirFreeInterior) {
  using V = Eigen::Vector2d;
  const V centre(2.5, 2.0);
  for (int i = 0; i < 200; ++i) {
    const double yaw = 0.01 + i * 0.0157;
    SCOPED_TRACE(yaw);
    const auto rotate = [&](const V& p) {
      const V d = p - centre;
      return V(centre + V(std::cos(yaw) * d.x() - std::sin(yaw) * d.y(),
                          std::sin(yaw) * d.x() + std::cos(yaw) * d.y()));
    };
    const auto box = [&](double x0, double y0, double x1, double y1) {
      std::vector<V> points = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
      for (auto& p : points) p = rotate(p);
      return std::make_shared<Polygon2D>(points);
    };
    std::vector<FreeSpaceObstacle> obstacles = {
        {1, box(-0.1, -0.1, 5.1, 0.0)}, {2, box(5.0, -0.1, 5.1, 4.1)},
        {3, box(-0.1, 4.0, 5.1, 4.1)}, {4, box(-0.1, -0.1, 0.0, 4.1)},
        {5, box(0.3, 0.0, 1.1, 0.6)}, {6, box(2.2, 1.5, 2.8, 2.5)}};
    ObstacleFreeSpace fs(box(-0.3, -0.3, 5.3, 4.3), obstacles, 0.0);
    const V from = rotate(V(1.5, 1.0)), to = rotate(V(3.5, 1.0));
    ASSERT_TRUE(fs.IsFree(from));
    ASSERT_TRUE(fs.IsFree(to));
    ASSERT_TRUE(fs.SegmentFree(from, to));
    EXPECT_NEAR(fs.GeodesicDistance(from, to), 2.0, 2e-12);
    const auto path = fs.FindPath(from, to);
    ASSERT_TRUE(path.found);
    EXPECT_NEAR(path.length, 2.0, 2e-12);
  }
}

// Reviewer probe14, with a fixed draw order on every compiler. The local
// geodesic is 1-Lipschitz whenever the connecting segment remains free.
TEST(ObstacleFreeSpace, OverlappingScenesRemainLipschitzAtSeveralResolutions) {
  using V = Eigen::Vector2d;
  for (int count : {30, 60}) for (int seed = 1; seed <= 4; ++seed) {
    SCOPED_TRACE(::testing::Message() << "boxes=" << count << " seed=" << seed);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<FreeSpaceObstacle> obstacles;
    for (int i = 0; i < count; ++i) {
      const double cx = 0.5 + 9 * uniform(rng), cy = 0.5 + 9 * uniform(rng);
      const double hx = 0.1 + 0.3 * uniform(rng), hy = 0.1 + 0.3 * uniform(rng);
      const double yaw = 3 * uniform(rng);
      const V ex(std::cos(yaw), std::sin(yaw)), ey(-std::sin(yaw), std::cos(yaw));
      std::vector<V> points;
      for (const V& sign : {V(-1, -1), V(1, -1), V(1, 1), V(-1, 1)})
        points.push_back(V(cx, cy) + sign.x() * hx * ex + sign.y() * hy * ey);
      obstacles.push_back({i + 1, std::make_shared<Polygon2D>(points)});
    }
    ObstacleFreeSpace fs(Room(10, 10), obstacles, 0.0);
    const V goal(9.95, 9.95);
    for (int query = 0; query < 300; ++query) {
      const double x = 0.05 + 9.9 * uniform(rng), y = 0.05 + 9.9 * uniform(rng);
      const V from(x, y);
      if (!fs.IsFree(from)) continue;
      const double distance = fs.GeodesicDistance(from, goal);
      if (!std::isfinite(distance)) continue;
      for (double step : {1e-12, 1e-9, 1e-6, 1e-3}) {
        const double angle = 2 * M_PI * uniform(rng);
        const V to = from + step * V(std::cos(angle), std::sin(angle));
        if (!fs.IsFree(to) || !fs.SegmentFree(from, to)) continue;
        const double other = fs.GeodesicDistance(to, goal);
        ASSERT_TRUE(std::isfinite(other));
        EXPECT_LE(std::fabs(other - distance), (to - from).norm() + 1e-13 * (1 + distance));
      }
    }
  }
}

// Preserve the base's seam policy: two hair-offset footprints seal a gap
// below two hairs. At a larger gap both scalar distance and course connect.
}  // namespace spatialblend
