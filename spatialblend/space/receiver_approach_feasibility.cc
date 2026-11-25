// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/space.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>

namespace spatialblend {
namespace {

constexpr double kEpsilon = 1e-8;
constexpr double kSegmentSampleM = .025;
constexpr size_t kMaxGridAxis = 128;

struct Polygon {
  std::vector<Eigen::Vector2d> points;
};

enum class EndpointState { kFree, kOutside, kBoundary, kBlocked, kNarrow };

Polygon InSpaceBasis(const ReceiverApproachPolygon& input) {
  Polygon result;
  result.points.reserve(input.vertices.size());
  for (const auto& vertex : input.vertices) {
    const Eigen::Vector3d point =
        input.local_to_space * Eigen::Vector3d(vertex.x(), vertex.y(), 1);
    result.points.push_back(point.head<2>());
  }
  return result;
}

double Cross(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
             const Eigen::Vector2d& p) {
  return (b.x() - a.x()) * (p.y() - a.y()) -
         (b.y() - a.y()) * (p.x() - a.x());
}

bool OnBoundary(const Polygon& polygon, const Eigen::Vector2d& point) {
  for (size_t i = 0; i < polygon.points.size(); ++i) {
    const auto& a = polygon.points[i];
    const auto& b = polygon.points[(i + 1) % polygon.points.size()];
    if (std::abs(Cross(a, b, point)) <= kEpsilon &&
        point.x() >= std::min(a.x(), b.x()) - kEpsilon &&
        point.x() <= std::max(a.x(), b.x()) + kEpsilon &&
        point.y() >= std::min(a.y(), b.y()) - kEpsilon &&
        point.y() <= std::max(a.y(), b.y()) + kEpsilon)
      return true;
  }
  return false;
}

bool Contains(const Polygon& polygon, const Eigen::Vector2d& point) {
  if (OnBoundary(polygon, point)) return true;
  bool inside = false;
  for (size_t i = 0, j = polygon.points.size() - 1; i < polygon.points.size();
       j = i++) {
    const auto& a = polygon.points[j];
    const auto& b = polygon.points[i];
    if ((a.y() > point.y()) != (b.y() > point.y()) &&
        point.x() < (b.x() - a.x()) * (point.y() - a.y()) /
                            (b.y() - a.y()) +
                        a.x())
      inside = !inside;
  }
  return inside;
}

double DistanceToSegment(const Eigen::Vector2d& point, const Eigen::Vector2d& a,
                         const Eigen::Vector2d& b) {
  const Eigen::Vector2d edge = b - a;
  const double length_squared = edge.squaredNorm();
  if (length_squared <= kEpsilon) return (point - a).norm();
  const double t = std::clamp((point - a).dot(edge) / length_squared, 0.0, 1.0);
  return (point - (a + t * edge)).norm();
}

double BoundaryDistance(const Polygon& polygon, const Eigen::Vector2d& point) {
  double result = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < polygon.points.size(); ++i)
    result = std::min(result, DistanceToSegment(
                                  point, polygon.points[i],
                                  polygon.points[(i + 1) % polygon.points.size()]));
  return result;
}

ReceiverApproachFeasibilityStatus MapContextStatus(
    ReceiverApproachContextStatus status) {
  switch (status) {
    case ReceiverApproachContextStatus::kAvailable:
      return ReceiverApproachFeasibilityStatus::kFeasible;
    case ReceiverApproachContextStatus::kExpired:
      return ReceiverApproachFeasibilityStatus::kExpired;
    case ReceiverApproachContextStatus::kReplaced:
      return ReceiverApproachFeasibilityStatus::kReplaced;
    case ReceiverApproachContextStatus::kBusy:
      return ReceiverApproachFeasibilityStatus::kBusy;
    case ReceiverApproachContextStatus::kStale:
      return ReceiverApproachFeasibilityStatus::kStale;
    default:
      return ReceiverApproachFeasibilityStatus::kNoContext;
  }
}

EndpointState ClassifyEndpoint(const Eigen::Vector2d& point,
                               const std::vector<Polygon>& navigable,
                               const std::vector<Polygon>& obstacles,
                               double clearance) {
  bool in_navigation = false;
  bool on_boundary = false;
  double nav_distance = std::numeric_limits<double>::infinity();
  for (const auto& polygon : navigable) {
    in_navigation |= Contains(polygon, point);
    on_boundary |= OnBoundary(polygon, point);
    if (Contains(polygon, point))
      nav_distance = std::min(nav_distance, BoundaryDistance(polygon, point));
  }
  if (!in_navigation) return EndpointState::kOutside;
  if (on_boundary) return EndpointState::kBoundary;
  for (const auto& obstacle : obstacles) {
    if (Contains(obstacle, point)) return EndpointState::kBlocked;
    if (BoundaryDistance(obstacle, point) <= clearance + kEpsilon)
      return EndpointState::kNarrow;
  }
  return nav_distance <= clearance + kEpsilon ? EndpointState::kNarrow
                                                : EndpointState::kFree;
}

bool IsFree(const Eigen::Vector2d& point, const std::vector<Polygon>& navigable,
            const std::vector<Polygon>& obstacles, double clearance) {
  return ClassifyEndpoint(point, navigable, obstacles, clearance) ==
         EndpointState::kFree;
}

bool SegmentFree(const Eigen::Vector2d& from, const Eigen::Vector2d& to,
                 const std::vector<Polygon>& navigable,
                 const std::vector<Polygon>& obstacles, double clearance) {
  const double length = (to - from).norm();
  const size_t samples = std::max<size_t>(2, static_cast<size_t>(
                                            std::ceil(length / kSegmentSampleM)));
  for (size_t i = 0; i <= samples; ++i) {
    const double t = static_cast<double>(i) / samples;
    if (!IsFree((1.0 - t) * from + t * to, navigable, obstacles, clearance))
      return false;
  }
  return true;
}

ReceiverApproachFeasibilityStatus EndpointStatus(EndpointState state,
                                                  bool start) {
  switch (state) {
    case EndpointState::kOutside:
      return start ? ReceiverApproachFeasibilityStatus::kStartOutsidePermittedRegion
                   : ReceiverApproachFeasibilityStatus::kGoalOutsidePermittedRegion;
    case EndpointState::kBoundary:
      return ReceiverApproachFeasibilityStatus::kEndpointOnBoundary;
    case EndpointState::kBlocked:
      return ReceiverApproachFeasibilityStatus::kBlocked;
    case EndpointState::kNarrow:
      return ReceiverApproachFeasibilityStatus::kInsufficientClearance;
    case EndpointState::kFree:
      return ReceiverApproachFeasibilityStatus::kFeasible;
  }
  return ReceiverApproachFeasibilityStatus::kInvalidGeometry;
}

double RouteLength(const std::vector<Eigen::Vector2d>& route) {
  double length = 0;
  for (size_t i = 1; i < route.size(); ++i) length += (route[i] - route[i - 1]).norm();
  return length;
}

}  // namespace

ReceiverApproachFeasibilityStatus Space::TryComputeReceiverApproachFeasibility(
    spatialblend::entity_id_t receiver_actor, uint64_t context_revision,
    ReceiverApproachFeasibilitySnapshot* out) const {
  if (!out) return ReceiverApproachFeasibilityStatus::kNoContext;
  *out = {};
  ReceiverApproachContext context;
  const auto context_status = TryCopyReceiverApproachContext(
      receiver_actor, context_revision, &context);
  if (context_status != ReceiverApproachContextStatus::kAvailable) {
    out->status = MapContextStatus(context_status);
    return out->status;
  }

  ReceiverApproachFeasibilitySnapshot value;
  value.context = context;
  value.start = context.navigation_root.pose.block<2, 1>(0, 3);
  value.safe_approach = context.destination.safe_approach_pose.block<2, 1>(0, 3);
  std::vector<Polygon> navigable;
  std::vector<Polygon> obstacles;
  for (const auto& polygon : context.navigable_regions)
    navigable.push_back(InSpaceBasis(polygon));
  for (const auto& polygon : context.obstacle_regions)
    obstacles.push_back(InSpaceBasis(polygon));
  const double clearance = context.clearance.footprint_radius_m +
                           context.clearance.required_clearance_m;
  auto finish = [&](ReceiverApproachFeasibilityStatus status) {
    // A feasibility result is usable only if its original receipt remains
    // exactly current after planning; polling never renews the receipt.
    ReceiverApproachContext verified;
    const auto verified_status = TryCopyReceiverApproachContext(
        receiver_actor, context_revision, &verified);
    if (verified_status != ReceiverApproachContextStatus::kAvailable) {
      *out = {};
      out->status = MapContextStatus(verified_status);
      return out->status;
    }
    value.status = status;
    {
      std::lock_guard<std::mutex> snapshot_lock(
          receiver_approach_feasibility_mutex_);
      if (next_receiver_approach_feasibility_revision_ == 0) {
        *out = {};
        out->status = ReceiverApproachFeasibilityStatus::kStale;
        return out->status;
      }
      value.calculation_revision = next_receiver_approach_feasibility_revision_++;
      value.owner_lifetime = value.context.registration_generation;
      receiver_approach_feasibility_snapshots_[receiver_actor] = value;
    }
    *out = value;
    return out->status;
  };

  const auto start_state = ClassifyEndpoint(value.start, navigable, obstacles, clearance);
  if (start_state != EndpointState::kFree) return finish(EndpointStatus(start_state, true));
  const auto goal_state =
      ClassifyEndpoint(value.safe_approach, navigable, obstacles, clearance);
  if (goal_state != EndpointState::kFree) return finish(EndpointStatus(goal_state, false));
  if (SegmentFree(value.start, value.safe_approach, navigable, obstacles, clearance)) {
    value.route = {value.start, value.safe_approach};
    value.route_length_m = RouteLength(value.route);
    return finish(ReceiverApproachFeasibilityStatus::kFeasible);
  }

  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  for (const auto& polygon : navigable) {
    for (const auto& point : polygon.points) {
      min_x = std::min(min_x, point.x());
      min_y = std::min(min_y, point.y());
      max_x = std::max(max_x, point.x());
      max_y = std::max(max_y, point.y());
    }
  }
  // Quarter-clearance lattice is conservative at every sampled segment while
  // still representing a real passage whose free width is just over 2r.
  const double grid_step = std::max(.05, clearance * .25);
  const int x_count = static_cast<int>(std::floor((max_x - min_x) / grid_step)) + 1;
  const int y_count = static_cast<int>(std::floor((max_y - min_y) / grid_step)) + 1;
  if (x_count <= 0 || y_count <= 0 || x_count > static_cast<int>(kMaxGridAxis) ||
      y_count > static_cast<int>(kMaxGridAxis))
    return finish(ReceiverApproachFeasibilityStatus::kInvalidGeometry);

  std::vector<Eigen::Vector2d> nodes{value.start, value.safe_approach};
  std::unordered_map<uint64_t, size_t> grid_nodes;
  const auto grid_key = [](int x, int y) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
           static_cast<uint32_t>(y);
  };
  for (int y = 0; y < y_count; ++y) {
    for (int x = 0; x < x_count; ++x) {
      const Eigen::Vector2d point(min_x + x * grid_step, min_y + y * grid_step);
      if (!IsFree(point, navigable, obstacles, clearance)) continue;
      grid_nodes.emplace(grid_key(x, y), nodes.size());
      nodes.push_back(point);
    }
  }
  std::vector<std::vector<std::pair<size_t, double>>> graph(nodes.size());
  auto add_edge = [&](size_t a, size_t b) {
    if (!SegmentFree(nodes[a], nodes[b], navigable, obstacles, clearance)) return;
    const double distance = (nodes[a] - nodes[b]).norm();
    graph[a].push_back({b, distance});
    graph[b].push_back({a, distance});
  };
  for (int y = 0; y < y_count; ++y) {
    for (int x = 0; x < x_count; ++x) {
      const auto current = grid_nodes.find(grid_key(x, y));
      if (current == grid_nodes.end()) continue;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if ((dx == 0 && dy == 0) || x + dx < 0 || y + dy < 0 ||
              x + dx >= x_count || y + dy >= y_count)
            continue;
          const auto neighbor = grid_nodes.find(grid_key(x + dx, y + dy));
          if (neighbor != grid_nodes.end() && current->second < neighbor->second)
            add_edge(current->second, neighbor->second);
        }
      }
    }
  }
  // Endpoint visibility is explicitly checked; no source or inferred pose can
  // create a grid edge.
  for (size_t node = 2; node < nodes.size(); ++node) {
    add_edge(0, node);
    add_edge(1, node);
  }

  using QueueEntry = std::pair<double, size_t>;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<>> open;
  std::vector<double> distance(nodes.size(), std::numeric_limits<double>::infinity());
  std::vector<size_t> parent(nodes.size(), nodes.size());
  distance[0] = 0;
  open.push({0, 0});
  while (!open.empty()) {
    const auto [cost, current] = open.top();
    open.pop();
    if (cost != distance[current]) continue;
    for (const auto& [next, edge] : graph[current]) {
      if (cost + edge >= distance[next]) continue;
      distance[next] = cost + edge;
      parent[next] = current;
      open.push({distance[next], next});
    }
  }
  if (!std::isfinite(distance[1])) {
    bool direct_hits_obstacle = false;
    const Eigen::Vector2d delta = value.safe_approach - value.start;
    for (int i = 1; i < 100; ++i) {
      const Eigen::Vector2d sample = value.start + (i / 100.0) * delta;
      for (const auto& obstacle : obstacles) direct_hits_obstacle |= Contains(obstacle, sample);
    }
    return finish(direct_hits_obstacle ? ReceiverApproachFeasibilityStatus::kBlocked
                                       : ReceiverApproachFeasibilityStatus::kDisconnected);
  }
  for (size_t current = 1; current != nodes.size(); current = parent[current])
    value.route.push_back(nodes[current]);
  std::reverse(value.route.begin(), value.route.end());
  // Greedy visibility compression preserves the same conservative segment
  // oracle, so no grid artifact becomes a penetrating shortcut.
  std::vector<Eigen::Vector2d> compact;
  for (size_t begin = 0; begin < value.route.size();) {
    compact.push_back(value.route[begin]);
    if (begin + 1 == value.route.size()) break;
    size_t next = value.route.size() - 1;
    while (next > begin + 1 && !SegmentFree(value.route[begin], value.route[next],
                                             navigable, obstacles, clearance))
      --next;
    if (next == begin) return finish(ReceiverApproachFeasibilityStatus::kInvalidGeometry);
    begin = next;
  }
  value.route = std::move(compact);
  value.route_length_m = RouteLength(value.route);
  return finish(ReceiverApproachFeasibilityStatus::kFeasible);
}

ReceiverApproachFeasibilityStatus
Space::TryCopyReceiverApproachFeasibilitySnapshot(
    spatialblend::entity_id_t receiver_actor, uint64_t calculation_revision,
    ReceiverApproachFeasibilitySnapshot* out) const {
  if (!out || calculation_revision == 0)
    return ReceiverApproachFeasibilityStatus::kNoContext;
  *out = {};
  ReceiverApproachFeasibilitySnapshot value;
  {
    std::unique_lock<std::mutex> lock(receiver_approach_feasibility_mutex_,
                                      std::try_to_lock);
    if (!lock.owns_lock()) return ReceiverApproachFeasibilityStatus::kBusy;
    const auto found = receiver_approach_feasibility_snapshots_.find(receiver_actor);
    if (found == receiver_approach_feasibility_snapshots_.end())
      return ReceiverApproachFeasibilityStatus::kNoContext;
    if (found->second.calculation_revision != calculation_revision)
      return ReceiverApproachFeasibilityStatus::kReplaced;
    value = found->second;
  }
  ReceiverApproachContext verified;
  const auto status = TryCopyReceiverApproachContext(
      receiver_actor, value.context.context_revision, &verified);
  if (status != ReceiverApproachContextStatus::kAvailable)
    return MapContextStatus(status);
  *out = std::move(value);
  return out->status;
}

}  // namespace spatialblend
