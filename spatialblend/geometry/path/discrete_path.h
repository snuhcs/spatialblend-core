// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_PATH_DISCRETE_PATH_H_
#define SPATIALBLEND_PATH_DISCRETE_PATH_H_

#include <Eigen/Dense>
#include <vector>

namespace spatialblend {

class DiscretePath {
 public:
  DiscretePath() = default;

  void AddPoint(const Eigen::Vector2d& point);
  const std::vector<Eigen::Vector2d>& GetPoints() const;
  double GetLength() const;
  size_t NumPoints() const;

 private:
  std::vector<Eigen::Vector2d> points_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_PATH_DISCRETE_PATH_H_
