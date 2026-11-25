// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/testing/test_driver.h
//
// TEST DOUBLE. Core has no driver role: driving is the application's job (the
// Unity avatar controller). This is the minimal stand-in the scenario harness,
// the simulator and the tests use to publish embodied poses. Each tick it
//   - heads toward the latest proposal along the traversable course within the
//     position bound,
//   - rotates toward the proposal yaw within the yaw bound (short arc), or toward its
//     direction of motion when the proposal leaves the yaw unspecified,
//   - and the caller publishes the result.
// Nothing else: no placement, no contained lock, no engagement teleport, no
// per-cause branching. The function is pure ((pose, target, course, bounds, dt)
// -> pose); the pose belongs to the caller. The bounds are the ONE declared
// config below (`Bounds`), defaulting to the v1 prefab values (2 m/s, 150 deg/s).
#ifndef SPATIALBLEND_SIMULATOR_SCENARIO_TEST_DRIVER_H_
#define SPATIALBLEND_SIMULATOR_SCENARIO_TEST_DRIVER_H_

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

namespace spatialblend {
namespace driver {

struct Bounds {
  double max_linear_speed = 2.0;                    // m/s (v1 prefab)
  double max_angular_speed = 150.0 * M_PI / 180.0;  // rad/s (v1 prefab)
};

struct Pose {
  Eigen::Vector2d position = Eigen::Vector2d::Zero();
  double yaw = 0.0;
};

inline double WrapAngle(double a) { return std::remainder(a, 2.0 * M_PI); }

// Advance `pose` toward `target` by `dt` seconds. `course` is the traversable
// polyline from the pose to the target position (first point = pose.position);
// empty or one point = straight line.
inline Pose Step(const Pose& pose, const Pose& target,
                 const std::vector<Eigen::Vector2d>& course,
                 const Bounds& bounds, double dt, bool yaw_specified = true) {
  Pose next = pose;
  dt = std::max(0.0, dt);
  double budget = bounds.max_linear_speed * dt;
  Eigen::Vector2d at = pose.position;
  const auto advance_to = [&](const Eigen::Vector2d& goal) {
    const Eigen::Vector2d d = goal - at;
    const double len = d.norm();
    if (len <= budget) {
      at = goal;
      budget -= len;
      return true;
    }
    at += d * (budget / len);
    budget = 0.0;
    return false;
  };
  if (course.size() >= 2) {
    for (size_t i = 1; i < course.size(); ++i)
      if (!advance_to(course[i])) break;
  } else {
    advance_to(target.position);
  }
  next.position = at;
  // An UNSPECIFIED yaw (v1 behaviour): face the direction of
  // motion, and keep the current yaw when stationary.
  double goal_yaw = target.yaw;
  if (!yaw_specified) {
    const Eigen::Vector2d moved = next.position - pose.position;
    goal_yaw = moved.norm() > 1e-9 ? std::atan2(moved.y(), moved.x()) : pose.yaw;
  }
  const double turn = WrapAngle(goal_yaw - pose.yaw);
  const double limit = bounds.max_angular_speed * dt;
  next.yaw = WrapAngle(pose.yaw + std::clamp(turn, -limit, limit));
  return next;
}

}  // namespace driver
}  // namespace spatialblend

#endif  // SPATIALBLEND_SIMULATOR_SCENARIO_TEST_DRIVER_H_
