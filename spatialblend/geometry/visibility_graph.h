// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_GEOMETRY_VISIBILITY_GRAPH_H_
#define SPATIALBLEND_GEOMETRY_VISIBILITY_GRAPH_H_

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {

struct VisibilityGraph {
  std::vector<Eigen::Vector2d> vertices;
  std::vector<std::vector<size_t>> adjacency;
  std::shared_ptr<Polygon2D> polygon;

  static VisibilityGraph Build(const Polygon2D& polygon);
  bool IsValid() const { return !vertices.empty(); }
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_GEOMETRY_VISIBILITY_GRAPH_H_
