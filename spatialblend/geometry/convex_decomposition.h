// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/convex_decomposition.h
#ifndef SPATIALBLEND_GEOMETRY_CONVEX_DECOMPOSITION_H_
#define SPATIALBLEND_GEOMETRY_CONVEX_DECOMPOSITION_H_

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {

struct ConvexCell {
  Polygon2D polygon;
  size_t index;
};

struct ConvexCellPartition {
  std::vector<ConvexCell> cells;
  std::vector<std::vector<size_t>> adjacency;
  std::shared_ptr<Polygon2D> source_polygon;

  static ConvexCellPartition Build(const Polygon2D& polygon);
  int FindCellForPoint(const Eigen::Vector2d& point) const;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_GEOMETRY_CONVEX_DECOMPOSITION_H_
