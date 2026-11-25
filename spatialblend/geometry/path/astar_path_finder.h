// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_PATH_ASTAR_PATH_FINDER_H_
#define SPATIALBLEND_PATH_ASTAR_PATH_FINDER_H_

#include <Eigen/Dense>
#include <utility>
#include <vector>

#include "spatialblend/geometry/visibility_graph.h"
#include "spatialblend/geometry/path/discrete_path.h"
#include "spatialblend/geometry/path/path_finder.h"

namespace spatialblend {

// A* path finder operating on a precomputed VisibilityGraph.
//
// PlanPath adds start and goal as temporary vertices, connects them via
// line-of-sight to graph vertices, then runs A* with Euclidean heuristic.
class AStarPathFinder : public PathFinder {
 public:
  explicit AStarPathFinder(const VisibilityGraph& graph);
  ~AStarPathFinder() override = default;

  std::pair<DiscretePath, double> PlanPath(
      const Eigen::Vector2d& start,
      const Eigen::Vector2d& goal) const override;

  double GetDistance(const Eigen::Vector2d& start,
                     const Eigen::Vector2d& goal) const override;

 private:
  VisibilityGraph graph_;

  // Returns true if segment (p1, p2) has clear line-of-sight inside polygon.
  bool IsLineOfSightClear(const Eigen::Vector2d& p1,
                          const Eigen::Vector2d& p2) const;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_PATH_ASTAR_PATH_FINDER_H_
