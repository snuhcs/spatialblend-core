// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_GEOMETRY_NAVIGABLE_GEOMETRY_CONTEXT_H_
#define SPATIALBLEND_GEOMETRY_NAVIGABLE_GEOMETRY_CONTEXT_H_

#include <map>
#include <tuple>
#include <memory>
#include <limits>
#include <mutex>
#include <vector>

#include "spatialblend/geometry/convex_decomposition.h"
#include "spatialblend/geometry/context/geometry_context.h"
#include "spatialblend/geometry/obstacle_free_space.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/geometry/visibility_graph.h"
#include "spatialblend/geometry/path/astar_path_finder.h"
#include "spatialblend/geometry/path/discrete_path.h"

namespace spatialblend {

// Precomputed geometry for navigable areas (rooms).
// Owns visibility graph, convex cells, and pathfinders.
// Built once from a Polygon2D; cached via shared_ptr.
class NavigableGeometryContext : public GeometryContext {
 public:
  static std::shared_ptr<NavigableGeometryContext> Create(
      std::shared_ptr<Polygon2D> polygon, double agent_radius = 0.0);
  // Obstacle-aware context: `obstacles` are world-frame
  // footprints that the root must keep `obstacle_clearance` (body radius)
  // away from. Geodesic distance and discrete paths then route through the
  // free space (see ObstacleFreeSpace); with no obstacles the behavior is
  // exactly the walkable-only context above.
  static std::shared_ptr<NavigableGeometryContext> Create(
      std::shared_ptr<Polygon2D> polygon, double agent_radius,
      std::vector<FreeSpaceObstacle> obstacles, double obstacle_clearance);

  // GeometryContext interface
  bool Contains(const Eigen::Vector2d& point) const override;
  Eigen::Vector2d GetNearestPoint(const Eigen::Vector2d& from) const override;
  double GetDistance(const Eigen::Vector2d& from) const override;
  Eigen::Vector3d ProjectOnto(const Eigen::Vector3d& point) const override;

  // Navigable-specific queries
  double GetGeodesicDistance(const Eigen::Vector2d& from,
                             const Eigen::Vector2d& to) const;
  // The same with the goal's own footprints open on the final leg (ObstacleFreeSpace::GeodesicDistance `open_keys`).
  double GetGeodesicDistance(const Eigen::Vector2d& from, const Eigen::Vector2d& to,
                             const std::vector<int64_t>& open) const;
  // Point of `ring` minimizing the same geodesic GetGeodesicDistance measures
  // (clearance 0 when obstacles exist, otherwise the walkable itself).
  // `final_leg_origin` receives where the final straight leg onto the ring
  // starts (see ObstacleFreeSpace::ClosestOnRing).
  // `open`: obstacle keys of the goal's own footprint (see ObstacleFreeSpace::ClosestOnRing).
  Eigen::Vector2d ClosestGeodesicOnRing(
      const Eigen::Vector2d& from, const Polygon2D& ring,
      double* length = nullptr,
      Eigen::Vector2d* final_leg_origin = nullptr,
      const std::vector<int64_t>& open = {}) const;
  DiscretePath GetDiscretePath(const Eigen::Vector2d& from,
                                const Eigen::Vector2d& to) const;
  // Every route GetGeodesicDistance chooses among (same free space), see
  // ObstacleFreeSpace::GeodesicRoutes. Empty with no polygon.
  std::vector<ObstacleFreeSpace::RoutePiece> GeodesicRoutes(
      const Eigen::Vector2d& from, const Eigen::Vector2d& to, double slack = 0.0) const;
  std::vector<ObstacleFreeSpace::RoutePiece> GeodesicRoutes(
      const Eigen::Vector2d& from, const Eigen::Vector2d& to, double slack, const std::vector<int64_t>& open) const;
  // The same in the free space QueryReachabilityUnder searches.
  std::vector<ObstacleFreeSpace::RoutePiece> ContactRoutes(
      const Eigen::Vector2d& from, const std::vector<const std::vector<Eigen::Vector2d>*>& rings, double slack = 0.0) const;
  std::vector<ObstacleFreeSpace::RoutePiece> GeodesicRoutesUnder(
      const Eigen::Vector2d& from, const Eigen::Vector2d& to,
      const std::vector<int64_t>& exempt,
      const std::vector<int64_t>& ring_exempt, double slack = 0.0) const;
  // The cone of directions in which the free space continues from x (the free space
  // under the exemptions QueryReachabilityUnder searches), see ObstacleFreeSpace::TangentConeAt.
  ObstacleFreeSpace::FeasibleCone TangentConeAt(const Eigen::Vector2d& x, const std::vector<int64_t>& exempt,
                                                const std::vector<int64_t>& ring_exempt, double tol) const;

  // --- Free space of the root (walkable minus obstacle clearance) ---------
  // The walkable is eroded by agent_radius when that is > 0 (callers that
  // pass an already-eroded polygon keep agent_radius = 0).
  // Free space for a body of radius `radius` (< 0: the configured obstacle
  // clearance). Built lazily once per distinct radius and cached.
  const ObstacleFreeSpace& FreeSpace(double radius = -1.0) const;
  bool HasObstacles() const { return !obstacles_.empty(); }
  double GetObstacleClearance() const { return obstacle_clearance_; }

  // Reachability query (scope gating, feasibility): is there a
  // collision-free path for a body of radius GetObstacleClearance() from
  // `from` to `to`? Endpoints that are not in free space (e.g. a point on a
  // referent's own footprint) are snapped to the nearest free point; the snap
  // distances are reported so a caller can bound them (a referent sealed
  // inside an obstacle ring snaps far, or into a disconnected pocket).
  struct Reachability {
    bool reachable = false;
    // Length of the free path including the snap legs; +inf when !reachable.
    double geodesic_distance_m = std::numeric_limits<double>::infinity();
    double from_snap_m = 0.0;
    double to_snap_m = 0.0;
    DiscretePath path;  // empty when !reachable
  };
  //
  // Single source of truth for reachability (scope gating's
  // IsNavigablyReachable should call this). `radius` < 0 uses the configured
  // obstacle clearance (the body radius the Space synced); 0 is point
  // connectivity around the raw footprints.
  Reachability QueryReachability(const Eigen::Vector2d& from,
                                 const Eigen::Vector2d& to,
                                 double radius = -1.0) const;
  // The same query in the free space UNDER an exemption set (the one IsFree /
  // NearestFree use): footprints in `exempt` are removed, footprints in `ring_exempt`
  // keep their footprint but lose the clearance ring, every other footprint keeps the
  // ring. Feasibility and reachability then ask ONE free space.
  // The free space under an exemption set, and the nearest pose in it. Its configuration
  // space is built for that set, so its nearest point is exact where the query-time
  // exemptions of NearestFree leave a corner of two constraints (a ring-exempt footprint
  // and another obstacle's ring) out of the candidates.
  // `radius` < 0: the configured obstacle clearance; otherwise the clearance ring of a body of that radius (a
  // driver's course keeps its body clear, the core's own free space has clearance 0).
  const ObstacleFreeSpace& FreeSpaceUnder(const std::vector<int64_t>& exempt,
                                          const std::vector<int64_t>& ring_exempt, double radius = -1.0) const;
  std::optional<Eigen::Vector2d> NearestFreeUnder(
      const Eigen::Vector2d& point, const std::vector<int64_t>& exempt,
      const std::vector<int64_t>& ring_exempt) const;
  Reachability QueryReachabilityUnder(const Eigen::Vector2d& from,
                                      const Eigen::Vector2d& to,
                                      const std::vector<int64_t>& exempt,
                                      const std::vector<int64_t>& ring_exempt, double radius = -1.0) const;

  // Access precomputed structures
  const VisibilityGraph& GetVisibilityGraph() const;
  const ConvexCellPartition& GetConvexCells() const;
  const Polygon2D& GetPolygon() const;

 private:
  NavigableGeometryContext(std::shared_ptr<Polygon2D> polygon,
                            double agent_radius);

  std::shared_ptr<Polygon2D> polygon_;
  double agent_radius_;
  std::vector<FreeSpaceObstacle> obstacles_;
  double obstacle_clearance_ = 0.0;
  mutable std::vector<std::pair<double, std::unique_ptr<ObstacleFreeSpace>>>
      free_spaces_;
  mutable std::mutex free_space_mutex_;
  // The raw walkable with no obstacles and no clearance: the exact shortest path
  // of a room without footprints (see GetGeodesicDistance).
  mutable std::unique_ptr<ObstacleFreeSpace> point_space_;
  mutable std::map<std::tuple<std::vector<int64_t>, std::vector<int64_t>, double>,
                   std::unique_ptr<ObstacleFreeSpace>> under_;
  const ObstacleFreeSpace& GetPointSpace() const;

  // Lazy-built, thread-safe
  mutable std::unique_ptr<VisibilityGraph> vg_;
  mutable std::shared_ptr<ConvexCellPartition> cells_;
  mutable std::unique_ptr<AStarPathFinder> astar_pf_;
  mutable std::mutex mutex_;

  void EnsureVisibilityGraph() const;
  void EnsureConvexCells() const;
  void EnsureAStar() const;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_GEOMETRY_NAVIGABLE_GEOMETRY_CONTEXT_H_
