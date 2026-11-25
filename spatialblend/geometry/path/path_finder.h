// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_PATH_PATH_FINDER_H_
#define SPATIALBLEND_PATH_PATH_FINDER_H_

#include <Eigen/Dense>
#include <utility>

#include "spatialblend/geometry/path/discrete_path.h"

namespace spatialblend {

class PathFinder {
 public:
  virtual ~PathFinder() = default;
  virtual std::pair<DiscretePath, double> PlanPath(
      const Eigen::Vector2d& start, const Eigen::Vector2d& goal) const = 0;
  virtual double GetDistance(
      const Eigen::Vector2d& start, const Eigen::Vector2d& goal) const = 0;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_PATH_PATH_FINDER_H_
