// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/testing/test_driver.h"

#include <gtest/gtest.h>

namespace spatialblend {
namespace driver {
namespace {

TEST(BoundedDriver, StraightLineIsSpeedBounded) {
  Pose pose;
  Pose target;
  target.position = Eigen::Vector2d(10, 0);
  const Pose next = Step(pose, target, {}, Bounds{}, 0.1);
  EXPECT_NEAR(next.position.x(), 0.2, 1e-12);
  EXPECT_NEAR(next.position.y(), 0.0, 1e-12);
}

TEST(BoundedDriver, ArrivesExactlyAndDoesNotOvershoot) {
  Pose pose;
  Pose target;
  target.position = Eigen::Vector2d(0.05, 0);
  const Pose next = Step(pose, target, {}, Bounds{}, 1.0);
  EXPECT_NEAR(next.position.x(), 0.05, 1e-12);
}

TEST(BoundedDriver, FollowsTheCourseAroundACorner) {
  Pose pose;
  Pose target;
  target.position = Eigen::Vector2d(1, 1);
  const std::vector<Eigen::Vector2d> course = {{0, 0}, {1, 0}, {1, 1}};
  Bounds b;
  b.max_linear_speed = 1.5;
  const Pose next = Step(pose, target, course, b, 1.0);
  EXPECT_NEAR(next.position.x(), 1.0, 1e-12);
  EXPECT_NEAR(next.position.y(), 0.5, 1e-12);
}

TEST(BoundedDriver, YawTurnsAlongTheShortArcAtTheBound) {
  Pose pose;
  pose.yaw = M_PI - 0.1;
  Pose target;
  target.yaw = -M_PI + 0.1;  // 0.2 rad away through +pi
  Bounds b;
  b.max_angular_speed = 0.1;
  const Pose next = Step(pose, target, {}, b, 1.0);
  EXPECT_NEAR(WrapAngle(next.yaw - pose.yaw), 0.1, 1e-12);
}

TEST(BoundedDriver, ZeroDtIsIdentity) {
  Pose pose;
  pose.position = Eigen::Vector2d(1, 2);
  Pose target;
  target.position = Eigen::Vector2d(5, 5);
  const Pose next = Step(pose, target, {}, Bounds{}, 0.0);
  EXPECT_EQ(next.position, pose.position);
}

}  // namespace
}  // namespace driver
}  // namespace spatialblend
