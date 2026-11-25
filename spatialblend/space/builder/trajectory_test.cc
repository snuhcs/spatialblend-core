// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/builder/trajectory.h"

#include <cmath>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

namespace spatialblend {
namespace {

TEST(TrajectoryTest, AddWaypointAndQuery) {
  Trajectory traj;
  traj.AddWaypoint(0.0, Eigen::Vector2d(0.0, 0.0), 0.0);
  traj.AddWaypoint(4.0, Eigen::Vector2d(4.0, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(traj.Duration(), 4.0);
}

TEST(TrajectoryTest, PositionInterpolation) {
  Trajectory traj;
  traj.AddWaypoint(0.0, Eigen::Vector2d(0.0, 0.0), 0.0);
  traj.AddWaypoint(2.0, Eigen::Vector2d(4.0, 2.0), 0.0);

  Eigen::Vector2d mid = traj.PositionAt(1.0);
  EXPECT_NEAR(mid.x(), 2.0, 1e-9);
  EXPECT_NEAR(mid.y(), 1.0, 1e-9);
}

TEST(TrajectoryTest, YawInterpolation) {
  Trajectory traj;
  // From 0 to -3pi/4: shortest path crosses -pi, going to 5pi/4 equivalent.
  // Use a simpler case: from 0 to pi/2 halfway should be pi/4.
  traj.AddWaypoint(0.0, Eigen::Vector2d(0.0, 0.0), 0.0);
  traj.AddWaypoint(2.0, Eigen::Vector2d(0.0, 0.0), M_PI / 2.0);

  double mid_yaw = traj.YawAt(1.0);
  EXPECT_NEAR(mid_yaw, M_PI / 4.0, 1e-9);

  // Shortest path: from pi to -pi+epsilon should wrap correctly.
  // From 3pi/4 to -3pi/4 (shortest is -pi/2 delta, going via pi boundary).
  Trajectory traj2;
  traj2.AddWaypoint(0.0, Eigen::Vector2d(0.0, 0.0), 3.0 * M_PI / 4.0);
  traj2.AddWaypoint(2.0, Eigen::Vector2d(0.0, 0.0), -3.0 * M_PI / 4.0);
  // delta via remainder: remainder(-3pi/4 - 3pi/4, 2pi) = remainder(-3pi/2, 2pi) = pi/2
  // midpoint: 3pi/4 + 0.5 * pi/2 = 3pi/4 + pi/4 = pi
  double mid_yaw2 = traj2.YawAt(1.0);
  EXPECT_NEAR(mid_yaw2, M_PI, 1e-9);
}

TEST(TrajectoryTest, StraightLineFactory) {
  Trajectory traj = Trajectory::StraightLine(
      Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(4.0, 0.0), 4.0);

  EXPECT_DOUBLE_EQ(traj.Duration(), 4.0);

  Eigen::Vector2d mid = traj.PositionAt(2.0);
  EXPECT_NEAR(mid.x(), 2.0, 1e-9);
  EXPECT_NEAR(mid.y(), 0.0, 1e-9);

  // Yaw should be atan2(0, 4) = 0
  EXPECT_NEAR(traj.YawAt(0.0), 0.0, 1e-9);
}

TEST(TrajectoryTest, ClampsBeyondDuration) {
  Trajectory traj;
  traj.AddWaypoint(0.0, Eigen::Vector2d(1.0, 2.0), 0.5);
  traj.AddWaypoint(3.0, Eigen::Vector2d(7.0, 8.0), 1.0);

  // Beyond end should return last position
  Eigen::Vector2d beyond = traj.PositionAt(10.0);
  EXPECT_NEAR(beyond.x(), 7.0, 1e-9);
  EXPECT_NEAR(beyond.y(), 8.0, 1e-9);

  // Before start should return first position
  Eigen::Vector2d before = traj.PositionAt(-1.0);
  EXPECT_NEAR(before.x(), 1.0, 1e-9);
  EXPECT_NEAR(before.y(), 2.0, 1e-9);
}

}  // namespace
}  // namespace spatialblend
