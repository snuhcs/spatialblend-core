// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/actor/actor_motion_tracker.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace spatialblend {

TEST(ActorMotionTrackerTest, EqualTimeCorrectionReseedsNextDerivativeOnly) {
  ActorMotionTracker tracker;
  tracker.Update(Eigen::Vector2d(0.0, 0.0), 0.0, 0.0);  // genuine t=0 seed
  tracker.Update(Eigen::Vector2d(10.0, 0.0), 0.0, 0.0);  // correction
  EXPECT_DOUBLE_EQ(tracker.GetState().linear_speed, 0.0);
  tracker.Update(Eigen::Vector2d(10.0, 0.0), 0.0, 0.1);
  EXPECT_DOUBLE_EQ(tracker.GetState().linear_speed, 0.0);
  EXPECT_DOUBLE_EQ(tracker.GetState().angular_speed, 0.0);
}
namespace test {

// kMotionFloor in relation_kernel.h: below this EMA-smoothed speed the actor is
// treated as "still" and frame-delta progress is rejected as jitter. The motion
// tracker is the producer of that signal, so its output for a stationary-but-
// jittering source MUST stay below the floor — otherwise the imminence gate
// opens on noise and the clone oscillates (the SpatialBlend clone-chatter bug).
constexpr double kMotionFloor = 0.05;
constexpr double kDt = 0.033;  // matches Space::update_interval_ (30 fps).

// A stationary VR source jitters a few mm per frame in random directions. The
// per-frame SPEED magnitude |Δp|/dt is therefore a persistently positive value
// (~2.5 mm / 33 ms ≈ 0.076 m/s), well above kMotionFloor. Smoothing that
// rectified magnitude does NOT remove the bias (the mean of a non-negative
// signal is non-zero), so the gate stays open on a still source. The correct
// signal is the magnitude of the smoothed *signed* velocity, which is zero-mean
// under jitter. This test pins that property at the tracker boundary.
TEST(ActorMotionTrackerTest, StationaryJitterReadsAsStill) {
  ActorMotionTracker tracker;
  // Zero-mean 4-phase box jitter; each consecutive step has |Δp| = d*sqrt(2).
  // Pick d so the rectified per-frame speed is ~0.076 m/s (clearly > floor).
  const double d = (0.0025 / std::sqrt(2.0));  // ~1.77 mm
  const Eigen::Vector2d base(10.0, -8.0);
  const Eigen::Vector2d offsets[4] = {{d, 0}, {0, d}, {-d, 0}, {0, -d}};
  for (int i = 0; i < 200; ++i) {
    tracker.Update(base + offsets[i % 4], 0.0, kDt);
  }
  // Sanity: the raw rectified speed really is above the floor (so this is a
  // meaningful test, not trivially passing because the jitter is tiny).
  EXPECT_GT(0.0025 / kDt, kMotionFloor);
  // The reported motion speed must read as "still" despite that rectified speed.
  EXPECT_LT(tracker.GetState().linear_speed, kMotionFloor);
}

// Yaw jitter (small, zero-mean) must likewise not register as turning.
TEST(ActorMotionTrackerTest, StationaryYawJitterReadsAsStill) {
  ActorMotionTracker tracker;
  const double a = 0.004;  // ~0.23 deg shake; |Δyaw|/dt ≈ 0.24 rad/s > floor.
  for (int i = 0; i < 200; ++i) {
    tracker.Update(Eigen::Vector2d(1.0, 1.0), (i % 2 == 0) ? a : -a, kDt);
  }
  EXPECT_GT((2 * a) / kDt, kMotionFloor);
  EXPECT_LT(tracker.GetState().angular_speed, kMotionFloor);
}

// A genuine slow walk in a consistent direction MUST still clear the floor —
// the fix rejects jitter without suppressing real (even slow) motion.
TEST(ActorMotionTrackerTest, GenuineMotionIsDetected) {
  ActorMotionTracker tracker;
  const double step = 0.003;  // 3 mm/frame in +x => 0.091 m/s sustained.
  Eigen::Vector2d p(0.0, 0.0);
  for (int i = 0; i < 100; ++i) {
    p.x() += step;
    tracker.Update(p, 0.0, kDt);
  }
  EXPECT_GT(tracker.GetState().linear_speed, kMotionFloor);
  EXPECT_NEAR(tracker.GetState().linear_speed, step / kDt, 5e-3);
}

// A genuine steady turn MUST still clear the floor.
TEST(ActorMotionTrackerTest, GenuineTurnIsDetected) {
  ActorMotionTracker tracker;
  const double dyaw = 0.01;  // 0.01 rad/frame => 0.30 rad/s sustained.
  double yaw = 0.0;
  for (int i = 0; i < 100; ++i) {
    yaw += dyaw;
    tracker.Update(Eigen::Vector2d(2.0, 2.0), yaw, kDt);
  }
  EXPECT_GT(tracker.GetState().angular_speed, kMotionFloor);
}

TEST(ActorMotionTrackerTest, PhysicalEmaAgreesAtCommonTenthSecondPrefix) {
  auto run = [](double dt, int steps) {
    ActorMotionTracker tracker;
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    constexpr double kSpeed = 0.6;
    constexpr double kYawRate = 0.4;
    tracker.Update(position, 0.0, dt);  // seed at t=0
    for (int step = 0; step < steps; ++step) {
      position.x() += kSpeed * dt;
      tracker.Update(position, kYawRate * (step + 1) * dt, dt);
    }
    return tracker.GetState();
  };

  const ActorMotionState at_30_hz = run(1.0 / 30.0, 3);
  const ActorMotionState at_60_hz = run(1.0 / 60.0, 6);
  constexpr double kTau = 0.147879;
  const double retained = 1.0 - std::exp(-0.1 / kTau);
  EXPECT_NEAR(at_30_hz.linear_speed, 0.6 * retained, 1e-6);
  EXPECT_NEAR(at_60_hz.linear_speed, 0.6 * retained, 1e-6);
  EXPECT_NEAR(at_30_hz.angular_speed, 0.4 * retained, 1e-6);
  EXPECT_NEAR(at_60_hz.angular_speed, 0.4 * retained, 1e-6);
}

TEST(ActorMotionTrackerTest, InvalidSamplesDoNotSeedOrContaminateState) {
  ActorMotionTracker tracker;
  tracker.Update(Eigen::Vector2d::Zero(), 0.0,
                 std::numeric_limits<double>::quiet_NaN());
  tracker.Update(Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity()),
                 0.0, 0.033);
  EXPECT_DOUBLE_EQ(tracker.GetState().linear_speed, 0.0);
  tracker.Update(Eigen::Vector2d::Zero(), 0.0, 0.033);  // valid seed
  tracker.Update(Eigen::Vector2d(0.033, 0.0), 0.0, 0.033);
  EXPECT_TRUE(std::isfinite(tracker.GetState().linear_speed));
  EXPECT_GT(tracker.GetState().linear_speed, 0.0);
}

}  // namespace test
}  // namespace spatialblend
