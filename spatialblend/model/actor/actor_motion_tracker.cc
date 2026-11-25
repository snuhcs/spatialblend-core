// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/actor/actor_motion_tracker.h"

#include <cmath>

namespace spatialblend {

void ActorMotionTracker::Reset() {
  prev_position_.setZero();
  prev_yaw_ = 0.0;
  has_previous_ = false;
  state_ = ActorMotionState{};
  smoothed_velocity_.setZero();
  smoothed_yaw_rate_ = 0.0;
}

void ActorMotionTracker::Update(const Eigen::Vector2d& position, double yaw,
                                double dt) {
  if (!std::isfinite(dt) || dt < 0.0 || !std::isfinite(position.x()) ||
      !std::isfinite(position.y()) || !std::isfinite(yaw)) {
    return;
  }
  if (!has_previous_) {
    prev_position_ = position;
    prev_yaw_ = yaw;
    has_previous_ = true;
    return;
  }
  // Equal-time corrections change the next interval's baseline but never
  // manufacture/decay a current rate.
  if (dt <= 0.0) {
    prev_position_ = position;
    prev_yaw_ = yaw;
    return;
  }
  // Smooth the SIGNED velocity / yaw rate, then take the magnitude — NOT the
  // other way around. The magnitude of a smoothed signed velocity is zero-mean
  // under random jitter (opposing frames cancel), so a stationary source reads
  // ~0. Smoothing the rectified magnitude |Δp|/dt instead retains the jitter's
  // positive bias (a few mm/frame ≈ 0.076 m/s), which kept the imminence gate
  // (relation_kernel.h) open on a still source and oscillated the clone — the
  // SpatialBlend clone-chatter bug.
  Eigen::Vector2d raw_velocity = (position - prev_position_) / dt;
  double raw_yaw_rate = std::remainder(yaw - prev_yaw_, 2 * M_PI) / dt;
  const double alpha = 1.0 - std::exp(-dt / kSmoothingTauSeconds);
  smoothed_velocity_ =
      (1.0 - alpha) * smoothed_velocity_ + alpha * raw_velocity;
  smoothed_yaw_rate_ =
      (1.0 - alpha) * smoothed_yaw_rate_ + alpha * raw_yaw_rate;
  state_.velocity = smoothed_velocity_;
  state_.linear_speed = smoothed_velocity_.norm();
  state_.angular_speed = std::abs(smoothed_yaw_rate_);
  prev_position_ = position;
  prev_yaw_ = yaw;
}

}  // namespace spatialblend
