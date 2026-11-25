// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/path/bezier_course_test.cc
#include "spatialblend/geometry/path/bezier_course.h"

#include <cmath>
#include <gtest/gtest.h>

namespace spatialblend {
namespace {

using V = Eigen::Vector2d;

// With no direction given, a straight course is its chord: the arc length is
// the chord length and the point after an arc is on the line.
TEST(BezierCourseTest, StraightCourseIsItsChord) {
  const auto c = BezierCourse::Build({V(0, 0), V(3, 4)}, std::nullopt, std::nullopt);
  ASSERT_TRUE(c.has_value());
  EXPECT_NEAR(c->Length(), 5.0, 1e-12);
  const V p = c->PointAfter(2.0);
  EXPECT_NEAR(p.x(), 1.2, 1e-9);
  EXPECT_NEAR(p.y(), 1.6, 1e-9);
  EXPECT_NEAR((c->PointAfter(99.0) - V(3, 4)).norm(), 0.0, 1e-12);
  EXPECT_NEAR((c->PointAfter(-1.0) - V(0, 0)).norm(), 0.0, 1e-12);
}

// A start heading along the chord and an end direction along the chord keep the
// straight course exact (the identity case: the mapped entering direction of an
// identical copy is the chord's direction).
TEST(BezierCourseTest, DirectionsAlongTheChordKeepItStraight) {
  const auto c = BezierCourse::Build({V(0, 0), V(6, 0)}, V(1, 0), V(1, 0));
  ASSERT_TRUE(c.has_value());
  EXPECT_NEAR(c->Length(), 6.0, 1e-12);
  EXPECT_NEAR(c->PointAfter(2.0).y(), 0.0, 1e-12);
  EXPECT_NEAR(c->PointAfter(2.0).x(), 2.0, 1e-9);
}

// The course arrives along the end direction: the last stretch of the curve
// points along it, and the curve ends on the goal.
TEST(BezierCourseTest, ArrivesAlongTheEndDirection) {
  const auto c = BezierCourse::Build({V(0, 0), V(4, 0)}, std::nullopt, V(0, 1));
  ASSERT_TRUE(c.has_value());
  EXPECT_GT(c->Length(), 4.0);  // a bent course is longer than its chord
  const V end = c->PointAfter(c->Length());
  EXPECT_NEAR((end - V(4, 0)).norm(), 0.0, 1e-9);
  const V near_end = c->PointAfter(c->Length() - 1e-4);
  const V dir = (end - near_end).normalized();
  EXPECT_NEAR(dir.dot(V(0, 1)), 1.0, 1e-4);
}

// Interior waypoints take the mean of the adjacent segment directions, so the
// course is continuous in direction there.
TEST(BezierCourseTest, TangentIsContinuousAtAnInteriorWaypoint) {
  const auto c = BezierCourse::Build({V(0, 0), V(2, 0), V(2, 2)}, std::nullopt,
                                     std::nullopt);
  ASSERT_TRUE(c.has_value());
  constexpr int kPer = 4096;
  const auto s = c->Samples(kPer);
  // The sample at index kPer is the shared waypoint (2, 0).
  ASSERT_GT(s.size(), static_cast<size_t>(kPer) + 2);
  EXPECT_NEAR((s[kPer] - V(2, 0)).norm(), 0.0, 1e-12);
  const V in = (s[kPer] - s[kPer - 1]).normalized();
  const V out = (s[kPer + 1] - s[kPer]).normalized();
  EXPECT_GT(in.dot(out), 1.0 - 1e-6);
}

TEST(BezierCourseTest, DegenerateCourseHasNoCurve) {
  EXPECT_FALSE(BezierCourse::Build({V(1, 1)}, std::nullopt, std::nullopt));
  EXPECT_FALSE(BezierCourse::Build({V(1, 1), V(1, 1)}, std::nullopt, std::nullopt));
}

}  // namespace
}  // namespace spatialblend
