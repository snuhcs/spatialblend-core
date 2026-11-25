// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/builder/trajectory.h"

#include <algorithm>
#include <cmath>

namespace spatialblend {

void Trajectory::AddWaypoint(double time, Eigen::Vector2d position,
                              double yaw) {
  Waypoint wp{time, position, yaw};
  auto it = std::lower_bound(waypoints_.begin(), waypoints_.end(), wp,
                             [](const Waypoint& a, const Waypoint& b) {
                               return a.time < b.time;
                             });
  waypoints_.insert(it, wp);
}

Eigen::Vector2d Trajectory::PositionAt(double t) const {
  if (waypoints_.empty()) return Eigen::Vector2d::Zero();
  if (t <= waypoints_.front().time) return waypoints_.front().position;
  if (t >= waypoints_.back().time) return waypoints_.back().position;

  // Find the first waypoint with time > t
  auto it = std::upper_bound(waypoints_.begin(), waypoints_.end(), t,
                             [](double val, const Waypoint& wp) {
                               return val < wp.time;
                             });
  const Waypoint& next = *it;
  const Waypoint& prev = *(it - 1);

  double alpha = (t - prev.time) / (next.time - prev.time);
  return prev.position + alpha * (next.position - prev.position);
}

double Trajectory::YawAt(double t) const {
  if (waypoints_.empty()) return 0.0;
  if (t <= waypoints_.front().time) return waypoints_.front().yaw;
  if (t >= waypoints_.back().time) return waypoints_.back().yaw;

  auto it = std::upper_bound(waypoints_.begin(), waypoints_.end(), t,
                             [](double val, const Waypoint& wp) {
                               return val < wp.time;
                             });
  const Waypoint& next = *it;
  const Waypoint& prev = *(it - 1);

  double alpha = (t - prev.time) / (next.time - prev.time);
  double delta = std::remainder(next.yaw - prev.yaw, 2.0 * M_PI);
  return prev.yaw + alpha * delta;
}

double Trajectory::Duration() const {
  if (waypoints_.size() < 2) return 0.0;
  return waypoints_.back().time - waypoints_.front().time;
}

// static
Trajectory Trajectory::StraightLine(Eigen::Vector2d from, Eigen::Vector2d to,
                                     double duration) {
  Trajectory traj;
  double yaw = std::atan2(to.y() - from.y(), to.x() - from.x());
  traj.AddWaypoint(0.0, from, yaw);
  traj.AddWaypoint(duration, to, yaw);
  return traj;
}

// static
Trajectory Trajectory::FromWaypoints(
    const std::vector<std::tuple<double, Eigen::Vector2d, double>>& waypoints) {
  Trajectory traj;
  for (const auto& [time, position, yaw] : waypoints) {
    traj.AddWaypoint(time, position, yaw);
  }
  return traj;
}

}  // namespace spatialblend
