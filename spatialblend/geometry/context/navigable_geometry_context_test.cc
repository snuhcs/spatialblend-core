// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/context/navigable_geometry_context_test.cc
#include "spatialblend/geometry/context/navigable_geometry_context.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <memory>

#include "gtest/gtest.h"
#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {
namespace {

// 20 m square with its upper-right quarter-by-quarter notch cut out (scenario room_L).
std::shared_ptr<Polygon2D> LRoom() {
  return std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 15}, {15, 15}, {15, 20}, {0, 20}});
}

// A geodesic is the SHORTEST path: for two points that see each other it is the
// straight segment. (The cell-corridor funnel chose its corridor by cell-centroid
// distance, so a visible pair got the long way round: 16.6 m for two points 1.36 m
// apart in the L-shaped room, and 49% of the visible pairs of this grid were wrong.)
TEST(NavigableGeometryContextTest, VisiblePairGeodesicIsTheStraightSegment) {
  auto poly = LRoom();
  auto ctx = NavigableGeometryContext::Create(poly);
  const Eigen::Vector2d a(8.13967, 9.48804), b(9.5, 9.488);
  EXPECT_NEAR(ctx->GetGeodesicDistance(a, b), (a - b).norm(), 1e-9);
  int checked = 0, bad = 0;
  for (double x0 = 0.5; x0 < 20; x0 += 1.7)
    for (double y0 = 0.5; y0 < 20; y0 += 1.9)
      for (double x1 = 0.7; x1 < 20; x1 += 2.3)
        for (double y1 = 0.3; y1 < 20; y1 += 1.3) {
          const Eigen::Vector2d p(x0, y0), q(x1, y1);
          if (!poly->Contains(p, true) || !poly->Contains(q, true)) continue;
          bool visible = true;
          for (int k = 1; k < 200 && visible; ++k)
            visible = poly->Contains(p + (q - p) * (k / 200.0), true);
          // (Sampling cannot see a segment that clips the reflex vertex by a hair.)
          const Eigen::Vector2d d = q - p, v(15, 15);
          const double t = std::clamp((v - p).dot(d) / d.squaredNorm(), 0.0, 1.0);
          if (!visible || (p + t * d - v).norm() < 0.1) continue;
          ++checked;
          if (std::fabs(ctx->GetGeodesicDistance(p, q) - (p - q).norm()) > 1e-9) {
            if (!bad++) ADD_FAILURE() << "first bad pair (" << p.x() << "," << p.y() << ") -> (" << q.x() << ","
                                      << q.y() << "): " << ctx->GetGeodesicDistance(p, q) << " vs " << (p - q).norm();
          }
        }
  EXPECT_GT(checked, 1000);
  EXPECT_EQ(bad, 0) << bad << " of " << checked << " visible pairs are not straight";
}

// Around the reflex corner the shortest path bends at the corner (15, 15).
TEST(NavigableGeometryContextTest, GeodesicBendsAtTheReflexVertex) {
  auto ctx = NavigableGeometryContext::Create(LRoom());
  const Eigen::Vector2d a(19, 12), b(12, 19);
  const double expected = (Eigen::Vector2d(15, 15) - a).norm() + (b - Eigen::Vector2d(15, 15)).norm();
  EXPECT_NEAR(ctx->GetGeodesicDistance(a, b), expected, 1e-9);
  const DiscretePath path = ctx->GetDiscretePath(a, b);
  ASSERT_EQ(path.NumPoints(), 3u);
  EXPECT_NEAR((path.GetPoints()[1] - Eigen::Vector2d(15, 15)).norm(), 0.0, 1e-9);
}

// ONE free space. Two boxes meet diagonally across a room, their facing corners (2, 1) and
// (2.3, 1.3) 0.424 m apart. A body of radius 0.2 passes between them: the exact round
// clearance rings of the two corners are 0.4 m apart at their nearest, 0.024 m short of
// touching. A polygon miter of the footprints (each inflated by 0.2 with sharp corners)
// reaches 0.283 m diagonally from a corner, and the two mitered corners overlap: the same
// pair of points was free, the clamp's NearestFree left them in place, and the reachability
// ask (under an exemption set unrelated to these boxes) called the far side sealed.
class DiagonalGapTest : public ::testing::Test {
 protected:
  static std::shared_ptr<Polygon2D> Box(double x0, double y0, double x1, double y1) {
    return std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});
  }
  void SetUp() override {
    auto room = Box(0, 0, 4, 2.3);
    std::vector<FreeSpaceObstacle> obstacles = {
        {1, Box(1.0, 0.0, 2.0, 1.0)},    // A: floor to y = 1, corner (2, 1)
        {2, Box(2.3, 1.3, 3.3, 2.3)},    // B: ceiling down to y = 1.3, corner (2.3, 1.3)
        {3, Box(3.6, 0.0, 3.9, 0.3)}};   // C: far from the gap
    ctx_ = NavigableGeometryContext::Create(room, 0.0, obstacles, 0.2);
  }
  std::shared_ptr<NavigableGeometryContext> ctx_;
  const Eigen::Vector2d left_{0.5, 1.0}, right_{3.0, 0.5}, gap_{2.15, 1.15};
};

TEST_F(DiagonalGapTest, TheGapIsFreeAndConnectsTheTwoSides) {
  // Feasibility and the clamp: the middle of the gap is free (0.212 m from both corners).
  const auto& fs = ctx_->FreeSpace();
  EXPECT_TRUE(fs.IsFree(gap_));
  const auto near = fs.NearestFree(gap_);
  ASSERT_TRUE(near.has_value());
  EXPECT_EQ((*near - gap_).norm(), 0.0);
  EXPECT_TRUE(fs.IsFree(left_));
  EXPECT_TRUE(fs.IsFree(right_));
  // Reachability with no exemption.
  EXPECT_TRUE(ctx_->QueryReachability(left_, right_).reachable);
}

TEST_F(DiagonalGapTest, ReachabilityUnderAnExemptionIsTheSameFreeSpace) {
  // Under an exemption set that does not touch the two boxes (C is exempt), reachability
  // must answer exactly what it answers with none, and what feasibility says of the gap.
  const auto plain = ctx_->QueryReachability(left_, right_);
  const auto under_exempt = ctx_->QueryReachabilityUnder(left_, right_, {3}, {});
  const auto under_ring_exempt = ctx_->QueryReachabilityUnder(left_, right_, {}, {3});
  ASSERT_TRUE(plain.reachable);
  EXPECT_TRUE(under_exempt.reachable) << "the gap the exact ring leaves open is sealed";
  EXPECT_TRUE(under_ring_exempt.reachable) << "the gap the exact ring leaves open is sealed";
  // ... and the path it finds stays in the exact free space (every point keeps the ring).
  const auto& fs = ctx_->FreeSpace();
  const auto& pts = under_exempt.path.GetPoints();
  for (size_t i = 1; i < pts.size(); ++i)
    EXPECT_TRUE(fs.SegmentFree(pts[i - 1], pts[i], {3}, {}))
        << "leg " << i << " leaves the free space";
}

// The mirror image: a gap the exact ring really closes (0.38 m between the corners < 2 x 0.2)
// is closed under an exemption too.
TEST_F(DiagonalGapTest, AClosedGapIsClosedUnderAnExemptionToo) {
  auto room = Box(0, 0, 4, 2.3);
  std::vector<FreeSpaceObstacle> obstacles = {
      {1, Box(1.0, 0.0, 2.0, 1.0)}, {2, Box(2.27, 1.27, 3.3, 2.3)}, {3, Box(3.6, 0.0, 3.9, 0.3)}};
  auto ctx = NavigableGeometryContext::Create(room, 0.0, obstacles, 0.2);
  EXPECT_FALSE(ctx->QueryReachability(left_, right_).reachable);
  EXPECT_FALSE(ctx->QueryReachabilityUnder(left_, right_, {3}, {}).reachable);
}

}  // namespace
}  // namespace spatialblend
