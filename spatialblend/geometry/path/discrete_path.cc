// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/path/discrete_path.cc
#include "spatialblend/geometry/path/discrete_path.h"

namespace spatialblend {

void DiscretePath::AddPoint(const Eigen::Vector2d& point) {
  points_.push_back(point);
}

const std::vector<Eigen::Vector2d>& DiscretePath::GetPoints() const {
  return points_;
}

double DiscretePath::GetLength() const {
  double length = 0.0;
  for (size_t i = 1; i < points_.size(); ++i) {
    length += (points_[i] - points_[i - 1]).norm();
  }
  return length;
}

size_t DiscretePath::NumPoints() const {
  return points_.size();
}

}  // namespace spatialblend
