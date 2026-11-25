// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SIMULATOR_TRAJECTORY_H_
#define SPATIALBLEND_SIMULATOR_TRAJECTORY_H_

#include <Eigen/Dense>
#include <tuple>
#include <vector>

namespace spatialblend {

class Trajectory {
 public:
  void AddWaypoint(double time, Eigen::Vector2d position, double yaw);

  Eigen::Vector2d PositionAt(double t) const;
  double YawAt(double t) const;
  double Duration() const;

  static Trajectory StraightLine(Eigen::Vector2d from, Eigen::Vector2d to,
                                  double duration);
  static Trajectory FromWaypoints(
      const std::vector<std::tuple<double, Eigen::Vector2d, double>>& waypoints);

 private:
  struct Waypoint {
    double time;
    Eigen::Vector2d position;
    double yaw;
  };
  std::vector<Waypoint> waypoints_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SIMULATOR_TRAJECTORY_H_
