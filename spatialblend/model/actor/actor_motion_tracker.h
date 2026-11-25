// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_ACTOR_MOTION_TRACKER_H_
#define SPATIALBLEND_SPACE_ACTOR_MOTION_TRACKER_H_

#include <Eigen/Dense>

namespace spatialblend {

struct ActorMotionState {
  // m/s: magnitude of the EMA-smoothed SIGNED velocity. This is zero-mean under
  // random jitter (opposing frames cancel), so a stationary-but-jittering source
  // reads ~0 — unlike a smoothed speed magnitude, whose positive bias survives.
  double linear_speed = 0.0;
  double angular_speed = 0.0;  // rad/s: |EMA-smoothed signed yaw rate|.
  Eigen::Vector2d velocity = Eigen::Vector2d::Zero();  // EMA-smoothed signed velocity.
};

class ActorMotionTracker {
 public:
  void Update(const Eigen::Vector2d& position, double yaw, double dt);
  // Starts a new temporal lifetime without deriving velocity across a gap.
  void Reset();
  const ActorMotionState& GetState() const { return state_; }

 private:
  Eigen::Vector2d prev_position_ = Eigen::Vector2d::Zero();
  double prev_yaw_ = 0.0;
  bool has_previous_ = false;
  ActorMotionState state_;
  Eigen::Vector2d smoothed_velocity_ = Eigen::Vector2d::Zero();
  double smoothed_yaw_rate_ = 0.0;
  // Smoothing factor for the SIGNED velocity / yaw-rate EMA. Lower = stronger
  // jitter rejection, slightly more lag detecting genuine motion. 0.2 keeps a
  // stationary source's coherent speed well under kMotionFloor (relation_kernel.h)
  // for few-mm/frame jitter while still detecting a slow walk within ~0.15 s.
  // Physical EMA time constant calibrated so alpha=.2 at the legacy 33 ms
  // tick: -0.033 / log(0.8). Keep the numeric value C++17 constexpr-safe.
  static constexpr double kSmoothingTauSeconds = 0.147879;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_ACTOR_MOTION_TRACKER_H_
