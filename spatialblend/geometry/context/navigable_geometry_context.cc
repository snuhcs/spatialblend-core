// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/geometry/context/navigable_geometry_context.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace spatialblend {

NavigableGeometryContext::NavigableGeometryContext(
    std::shared_ptr<Polygon2D> polygon, double agent_radius)
    : polygon_(std::move(polygon)), agent_radius_(agent_radius) {}

std::shared_ptr<NavigableGeometryContext> NavigableGeometryContext::Create(
    std::shared_ptr<Polygon2D> polygon, double agent_radius) {
  return std::shared_ptr<NavigableGeometryContext>(
      new NavigableGeometryContext(std::move(polygon), agent_radius));
}

std::shared_ptr<NavigableGeometryContext> NavigableGeometryContext::Create(
    std::shared_ptr<Polygon2D> polygon, double agent_radius,
    std::vector<FreeSpaceObstacle> obstacles, double obstacle_clearance) {
  auto ctx = std::shared_ptr<NavigableGeometryContext>(
      new NavigableGeometryContext(std::move(polygon), agent_radius));
  ctx->obstacles_ = std::move(obstacles);
  ctx->obstacle_clearance_ =
      std::isfinite(obstacle_clearance) && obstacle_clearance > 0.0
          ? obstacle_clearance
          : 0.0;
  return ctx;
}

const ObstacleFreeSpace& NavigableGeometryContext::FreeSpace(
    double radius) const {
  if (!(radius >= 0.0)) radius = obstacle_clearance_;
  std::lock_guard<std::mutex> lock(free_space_mutex_);
  for (const auto& [r, fs] : free_spaces_) {
    if (r == radius) return *fs;
  }
  std::shared_ptr<const Polygon2D> walk = polygon_;
  if (polygon_ && agent_radius_ > 0.0) {
    walk = std::make_shared<Polygon2D>(polygon_->Inflate(-agent_radius_));
  }
  free_spaces_.emplace_back(
      radius, std::make_unique<ObstacleFreeSpace>(walk, obstacles_, radius));
  return *free_spaces_.back().second;
}

const ObstacleFreeSpace& NavigableGeometryContext::FreeSpaceUnder(
    const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt, double radius) const {
  if (exempt.empty() && ring_exempt.empty()) return FreeSpace(radius);
  if (!(radius >= 0.0)) radius = obstacle_clearance_;
  auto key_of = [](std::vector<int64_t> v) { std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end()); return v; };
  const auto key = std::make_tuple(key_of(exempt), key_of(ring_exempt), radius);
  std::lock_guard<std::mutex> lock(free_space_mutex_);
  auto it = under_.find(key);
  if (it == under_.end()) {
    std::shared_ptr<const Polygon2D> walk = polygon_;
    if (polygon_ && agent_radius_ > 0.0)
      walk = std::make_shared<Polygon2D>(polygon_->Inflate(-agent_radius_));
    // The SAME free space as every other query (feasibility, the clamp, FreeSpace): the
    // raw footprints with the exact round clearance ring, the exempt footprints removed
    // and the ring-exempt ones kept without their ring. (It used to inflate the
    // footprints by a polygon miter and search with clearance 0: the mitered corner
    // reaches r / cos(theta / 2) > r from the footprint, so a gap the exact ring leaves
    // open at a corner was sealed here and the body "unreachable" at a point the clamp
    // had just declared free.)
    std::vector<FreeSpaceObstacle> obs;
    for (const auto& o : obstacles_) {
      if (std::binary_search(std::get<0>(key).begin(), std::get<0>(key).end(), o.key)) continue;
      obs.push_back(o);
    }
    it = under_.emplace(key, std::make_unique<ObstacleFreeSpace>(walk, std::move(obs), radius,
                                                                 std::get<1>(key))).first;
  }
  return *it->second;
}

std::optional<Eigen::Vector2d> NavigableGeometryContext::NearestFreeUnder(
    const Eigen::Vector2d& point, const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt) const {
  return FreeSpaceUnder(exempt, ring_exempt).NearestFree(point);
}

NavigableGeometryContext::Reachability
NavigableGeometryContext::QueryReachabilityUnder(
    const Eigen::Vector2d& from, const Eigen::Vector2d& to,
    const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt, double radius) const {
  if (exempt.empty() && ring_exempt.empty()) return QueryReachability(from, to, radius);
  const ObstacleFreeSpace* fs = &FreeSpaceUnder(exempt, ring_exempt, radius);
  Reachability r;
  auto found = fs->FindPath(from, to);
  if (!found.found) return r;
  r.reachable = true;
  r.geodesic_distance_m = found.length;
  r.from_snap_m = found.from_snap_m;
  r.to_snap_m = found.to_snap_m;
  for (const auto& p : found.points) r.path.AddPoint(p);
  return r;
}

NavigableGeometryContext::Reachability
NavigableGeometryContext::QueryReachability(const Eigen::Vector2d& from,
                                            const Eigen::Vector2d& to,
                                            double radius) const {
  Reachability r;
  auto found = FreeSpace(radius).FindPath(from, to);
  if (!found.found) return r;
  r.reachable = true;
  r.geodesic_distance_m = found.length;
  r.from_snap_m = found.from_snap_m;
  r.to_snap_m = found.to_snap_m;
  for (const auto& p : found.points) r.path.AddPoint(p);
  return r;
}

const ObstacleFreeSpace& NavigableGeometryContext::GetPointSpace() const {
  std::lock_guard<std::mutex> lock(free_space_mutex_);
  if (!point_space_)
    point_space_ = std::make_unique<ObstacleFreeSpace>(
        polygon_, std::vector<FreeSpaceObstacle>{}, 0.0);
  return *point_space_;
}

bool NavigableGeometryContext::Contains(const Eigen::Vector2d& point) const {
  return polygon_ && polygon_->Contains(point);
}

Eigen::Vector2d NavigableGeometryContext::GetNearestPoint(
    const Eigen::Vector2d& from) const {
  if (!polygon_) return from;
  return polygon_->GetNearestPoint(from);
}

double NavigableGeometryContext::GetDistance(
    const Eigen::Vector2d& from) const {
  if (!polygon_) return 0.0;
  return polygon_->GetDistance(from);
}

Eigen::Vector3d NavigableGeometryContext::ProjectOnto(
    const Eigen::Vector3d& point) const {
  if (!polygon_) return point;
  if (polygon_->Contains(point.head<2>())) return point;
  Eigen::Vector2d nearest = polygon_->GetNearestPoint(point.head<2>());
  return Eigen::Vector3d(nearest.x(), nearest.y(), point.z());
}

Eigen::Vector2d NavigableGeometryContext::ClosestGeodesicOnRing(
    const Eigen::Vector2d& from, const Polygon2D& ring, double* length,
    Eigen::Vector2d* final_leg_origin, const std::vector<int64_t>& open) const {
  if (final_leg_origin) *final_leg_origin = from;
  if (!polygon_) {
    const Eigen::Vector2d p = ring.NumPoints() >= 1 ? ring.GetNearestPoint(from) : from;
    if (length) *length = (p - from).norm();
    return p;
  }
  const ObstacleFreeSpace& fs =
      obstacles_.empty() ? GetPointSpace() : FreeSpace(0.0);
  return fs.ClosestOnRing(from, ring, length, final_leg_origin, open);
}

double NavigableGeometryContext::GetGeodesicDistance(
    const Eigen::Vector2d& from, const Eigen::Vector2d& to) const {
  if (!polygon_) return (from - to).norm();

  // Obstacle-aware (the point geodesic): point connectivity around the RAW
  // obstacle footprints -- deliberately independent of any body radius, so
  // a source's relation value does not depend on the sender's local body
  // radius. +inf when the endpoints are not connected (unreachable).
  // Length only: the free space keeps a Dijkstra field per contact, so a
  // solve that moves the trial point does not re-plan. The path itself
  // stays on FindPath (QueryReachability).
  if (!obstacles_.empty()) return FreeSpace(0.0).GeodesicDistance(from, to);

  // No footprints: the shortest path of the walkable polygon itself. Exact: the
  // straight segment when the two points see each other, else the shortest path
  // through the polygon's reflex vertices. (A cell-corridor funnel chose its
  // corridor by cell-centroid distance, so a visible pair got a path around the
  // room: 16.6 m for two points 1.36 m apart in an L-shaped room.)
  return GetPointSpace().GeodesicDistance(from, to);
}

double NavigableGeometryContext::GetGeodesicDistance(const Eigen::Vector2d& from, const Eigen::Vector2d& to,
                                                    const std::vector<int64_t>& open) const {
  if (!polygon_) return (from - to).norm();
  if (!obstacles_.empty()) return FreeSpace(0.0).GeodesicDistance(from, to, open);
  return GetPointSpace().GeodesicDistance(from, to);
}

std::vector<ObstacleFreeSpace::RoutePiece> NavigableGeometryContext::GeodesicRoutes(
    const Eigen::Vector2d& from, const Eigen::Vector2d& to, double slack, const std::vector<int64_t>& open) const {
  if (!polygon_) return {};
  if (!obstacles_.empty()) return FreeSpace(0.0).GeodesicRoutes(from, to, slack, open);
  return GetPointSpace().GeodesicRoutes(from, to, slack);
}

std::vector<ObstacleFreeSpace::RoutePiece>
NavigableGeometryContext::GeodesicRoutes(const Eigen::Vector2d& from,
                                         const Eigen::Vector2d& to, double slack) const {
  if (!polygon_) return {};
  if (!obstacles_.empty()) return FreeSpace(0.0).GeodesicRoutes(from, to, slack);
  return GetPointSpace().GeodesicRoutes(from, to, slack);
}

std::vector<ObstacleFreeSpace::RoutePiece> NavigableGeometryContext::ContactRoutes(
    const Eigen::Vector2d& from, const std::vector<const std::vector<Eigen::Vector2d>*>& rings, double slack) const {
  if (!polygon_) return {};
  if (!obstacles_.empty()) return FreeSpace(0.0).ContactRoutes(from, rings, slack);
  return GetPointSpace().ContactRoutes(from, rings, slack);
}

std::vector<ObstacleFreeSpace::RoutePiece>
NavigableGeometryContext::GeodesicRoutesUnder(
    const Eigen::Vector2d& from, const Eigen::Vector2d& to,
    const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt, double slack) const {
  if (!polygon_) return {};
  return FreeSpaceUnder(exempt, ring_exempt).GeodesicRoutes(from, to, slack);
}

ObstacleFreeSpace::FeasibleCone NavigableGeometryContext::TangentConeAt(
    const Eigen::Vector2d& x, const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt, double tol) const {
  if (!polygon_) {
    ObstacleFreeSpace::FeasibleCone c;
    c.arcs = {{0.0, 6.283185307179586}};
    return c;
  }
  return FreeSpace().TangentConeAt(x, exempt, ring_exempt, tol);
}

DiscretePath NavigableGeometryContext::GetDiscretePath(
    const Eigen::Vector2d& from, const Eigen::Vector2d& to) const {
  if (!polygon_) {
    DiscretePath path;
    path.AddPoint(from);
    path.AddPoint(to);
    return path;
  }

  if (!obstacles_.empty()) {
    // Same radius-0 free space as GetGeodesicDistance (the decode walks the
    // path the source's geodesic was measured along).
    auto found = FreeSpace(0.0).FindPath(from, to);
    DiscretePath path;
    if (found.found) {
      for (const auto& p : found.points) path.AddPoint(p);
    } else {
      path.AddPoint(from);
      path.AddPoint(to);
    }
    return path;
  }

  const auto found = GetPointSpace().FindPath(from, to);
  DiscretePath path;
  if (found.found) {
    for (const auto& p : found.points) path.AddPoint(p);
  } else {
    path.AddPoint(from);
    path.AddPoint(to);
  }
  return path;
}

const VisibilityGraph& NavigableGeometryContext::GetVisibilityGraph() const {
  std::lock_guard<std::mutex> lock(mutex_);
  EnsureVisibilityGraph();
  return *vg_;
}

const ConvexCellPartition& NavigableGeometryContext::GetConvexCells() const {
  std::lock_guard<std::mutex> lock(mutex_);
  EnsureConvexCells();
  return *cells_;
}

const Polygon2D& NavigableGeometryContext::GetPolygon() const {
  return *polygon_;
}

void NavigableGeometryContext::EnsureVisibilityGraph() const {
  if (!vg_ && polygon_) {
    vg_ = std::make_unique<VisibilityGraph>(
        VisibilityGraph::Build(*polygon_));
  }
}

void NavigableGeometryContext::EnsureConvexCells() const {
  if (!cells_ && polygon_) {
    cells_ = std::make_shared<ConvexCellPartition>(
        ConvexCellPartition::Build(*polygon_));
  }
}

void NavigableGeometryContext::EnsureAStar() const {
  EnsureVisibilityGraph();
  if (!astar_pf_ && vg_) {
    astar_pf_ = std::make_unique<AStarPathFinder>(*vg_);
  }
}

}  // namespace spatialblend
