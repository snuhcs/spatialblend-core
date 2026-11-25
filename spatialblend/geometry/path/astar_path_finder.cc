// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/path/astar_path_finder.cc
#include "spatialblend/geometry/path/astar_path_finder.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <vector>

namespace spatialblend {

AStarPathFinder::AStarPathFinder(const VisibilityGraph& graph)
    : graph_(graph) {}

bool AStarPathFinder::IsLineOfSightClear(const Eigen::Vector2d& p1,
                                         const Eigen::Vector2d& p2) const {
  if (!graph_.polygon) return false;

  // Both endpoints must be inside (or on boundary of) the polygon.
  if (!graph_.polygon->Contains(p1, /*include_boundary=*/true)) return false;
  if (!graph_.polygon->Contains(p2, /*include_boundary=*/true)) return false;

  // Check LOS using interior samples to avoid false positives when endpoints
  // are exactly on the polygon boundary (which causes IntersectsSegment to
  // return true even for valid segments).
  //
  // Strategy: test a segment slightly shrunk from both ends. If the shrunk
  // segment and additional interior samples are all inside the polygon, LOS
  // is clear.
  Eigen::Vector2d dir = p2 - p1;
  double len = dir.norm();
  if (len < 1e-12) {
    return true;  // Same point: trivially clear.
  }

  // Use the shrunk segment (epsilon away from each endpoint).
  constexpr double kEps = 1e-6;
  double shrink = std::min(kEps, len * 0.01);
  Eigen::Vector2d unit = dir / len;
  Eigen::Vector2d p1s = p1 + unit * shrink;
  Eigen::Vector2d p2s = p2 - unit * shrink;

  // If the shrunk segment is degenerate (very short), just check midpoint.
  if ((p2s - p1s).norm() < 1e-12) {
    Eigen::Vector2d mid = 0.5 * (p1 + p2);
    return graph_.polygon->Contains(mid, /*include_boundary=*/true);
  }

  // The shrunk segment must not cross the polygon boundary.
  if (graph_.polygon->IntersectsSegment(p1s, p2s)) return false;

  // Check multiple interior samples for concave regions.
  for (int k = 1; k <= 4; ++k) {
    double t = static_cast<double>(k) / 5.0;
    Eigen::Vector2d sample = p1 + t * dir;
    if (!graph_.polygon->Contains(sample, /*include_boundary=*/true)) {
      return false;
    }
  }

  return true;
}

std::pair<DiscretePath, double> AStarPathFinder::PlanPath(
    const Eigen::Vector2d& start, const Eigen::Vector2d& goal) const {
  // Build an augmented graph: static vertices + start + goal.
  const size_t static_n = graph_.vertices.size();
  const size_t start_idx = static_n;
  const size_t goal_idx = static_n + 1;
  const size_t total_n = static_n + 2;

  // Graph representation: positions and adjacency.
  std::vector<Eigen::Vector2d> positions;
  positions.reserve(total_n);
  for (const auto& v : graph_.vertices) positions.push_back(v);
  positions.push_back(start);
  positions.push_back(goal);

  std::vector<std::vector<size_t>> adj(total_n);
  // Copy static adjacency.
  for (size_t i = 0; i < static_n; ++i) {
    adj[i] = graph_.adjacency[i];
  }

  // Connect start and goal to static vertices via LOS.
  for (size_t i = 0; i < static_n; ++i) {
    if (IsLineOfSightClear(start, graph_.vertices[i])) {
      adj[start_idx].push_back(i);
      adj[i].push_back(start_idx);
    }
    if (IsLineOfSightClear(goal, graph_.vertices[i])) {
      adj[goal_idx].push_back(i);
      adj[i].push_back(goal_idx);
    }
  }

  // Direct start -> goal connection.
  if (IsLineOfSightClear(start, goal)) {
    adj[start_idx].push_back(goal_idx);
    adj[goal_idx].push_back(start_idx);
  }

  // A* search.
  std::vector<double> g_cost(total_n, std::numeric_limits<double>::infinity());
  std::vector<double> h_cost(total_n, 0.0);
  std::vector<size_t> parent(total_n, SIZE_MAX);

  g_cost[start_idx] = 0.0;
  h_cost[start_idx] = (start - goal).norm();

  auto f_cost = [&](size_t idx) { return g_cost[idx] + h_cost[idx]; };

  // Min-heap: (f_cost, index)
  using Entry = std::pair<double, size_t>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open_set;
  open_set.push({f_cost(start_idx), start_idx});

  std::vector<bool> closed(total_n, false);

  while (!open_set.empty()) {
    auto [f, current] = open_set.top();
    open_set.pop();

    if (closed[current]) continue;
    closed[current] = true;

    if (current == goal_idx) {
      // Reconstruct path.
      std::vector<size_t> path_indices;
      size_t idx = goal_idx;
      while (idx != SIZE_MAX) {
        path_indices.push_back(idx);
        idx = parent[idx];
      }
      std::reverse(path_indices.begin(), path_indices.end());

      DiscretePath path;
      for (size_t pi : path_indices) {
        path.AddPoint(positions[pi]);
      }
      double dist = g_cost[goal_idx];
      return {path, dist};
    }

    for (size_t neighbor : adj[current]) {
      if (closed[neighbor]) continue;
      double tentative_g =
          g_cost[current] + (positions[current] - positions[neighbor]).norm();
      if (tentative_g < g_cost[neighbor]) {
        g_cost[neighbor] = tentative_g;
        h_cost[neighbor] = (positions[neighbor] - goal).norm();
        parent[neighbor] = current;
        open_set.push({f_cost(neighbor), neighbor});
      }
    }
  }

  // No path found: return empty path with infinite distance.
  return {DiscretePath(), std::numeric_limits<double>::infinity()};
}

double AStarPathFinder::GetDistance(const Eigen::Vector2d& start,
                                    const Eigen::Vector2d& goal) const {
  auto [path, dist] = PlanPath(start, goal);
  return dist;
}

}  // namespace spatialblend
