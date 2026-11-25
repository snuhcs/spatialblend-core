// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_ELEMENT_NAVIGABLE_SPATIAL_DEFINITION_H_
#define SPATIALBLEND_ELEMENT_NAVIGABLE_SPATIAL_DEFINITION_H_

#include <memory>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

#include "spatialblend/model/element/spatial_definition.h"
#include "spatialblend/geometry/context/navigable_geometry_context.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/geometry/path/discrete_path.h"

namespace spatialblend {

class NavigableSpatialDefinition : public SpatialDefinition {
 public:
  explicit NavigableSpatialDefinition(
      std::shared_ptr<Polygon2D> walkable,
      std::shared_ptr<Polygon2D> wall = nullptr,
      double agent_radius = 0.0,
      GeometryRole geometry_role = GeometryRole::kNavigable);

  // Core interface
  bool Contains(const Eigen::Vector2d& point) const override;
  std::shared_ptr<Polygon2D> GetBoundary() const override;
  const Eigen::Matrix4d& GetLocalTransform() const override;
  Bound2D GetBounds() const override;

  // Canonical interior chart (SpatialDefinition override): box-UVW for box-like
  // walkables (affine-exact), Mean Value Coordinates for irregular/non-convex
  // ones (shape-faithful). Always also fills the universal uvw fallback so a
  // cross-kind partner degrades gracefully. Decode clamps back into the boundary
  // (an MVC convex-combination can otherwise exit a non-convex polygon).
  NormalizedInterior EncodeInterior(
      const Eigen::Vector3d& world_point) const override;
  Eigen::Vector3d DecodeInterior(const NormalizedInterior& c) const override;

  // Element-local polygon (sb_set_element_local_polygon). When true the
  // polygon is stored verbatim in the owning ELEMENT's local frame and every
  // caller with tree access lifts through the element world transform (see
  // SpatialDefinition::RequiresElementWorldTransform). Default false keeps the
  // historical world-absolute reading. Intended for Containment-role regions;
  // a kNavigable room must stay world-absolute (obstacle sync, free space).
  void SetElementLocal(bool element_local) { element_local_ = element_local; }
  bool IsElementLocal() const { return element_local_; }
  bool RequiresElementWorldTransform() const override { return element_local_; }

  // Geometry access
  std::shared_ptr<Polygon2D> GetWalkablePolygon() const;
  std::shared_ptr<Polygon2D> GetWall() const;

  void SetAgentRadius(double radius);
  double GetAgentRadius() const;

  // --- Obstacles -----------------------------------------------------------
  // World-frame obstacle footprints inside this navigable area and the body
  // radius the root must keep from them. The Space keeps these in sync with
  // its GeometryRole::kObstacle elements (Space::SyncNavigableObstaclesLocked);
  // standalone callers may set them directly. Contains() stays walkable-only
  // (it is containment/scope, not feasibility); feasibility, geodesic
  // distance and paths use the obstacle-aware free space below.
  // Returns true if the obstacle set changed (the context is then rebuilt).
  bool SetObstacles(std::vector<FreeSpaceObstacle> obstacles,
                    double clearance);
  const std::vector<FreeSpaceObstacle>& GetObstacles() const;
  // Builds the radius-0 (geodesic) and body-radius (feasibility / course)
  // free spaces and their visibility graphs now, so the cost lands on the
  // update that changed the geometry rather than on a later query.
  // Cost: ~30 ms at 60 box obstacles (see geometry_mediation_benchmark_test);
  // every obstacle transform change pays it again.
  void PrewarmFreeSpaces() const;
  double GetObstacleClearance() const;

  // Whether a pose is in the walkable region by the free space's own boundary tolerance (a
  // rounding error outside the boundary is on it). Contains() above is the exact containment
  // test; feasibility (IsFree, NearestFree, the solve's projection) uses this one.
  bool IsInWalkable(const Eigen::Vector2d& point) const;

  // Root feasibility: inside the walkable and >= clearance from every
  // obstacle whose key is not in `exempt` (an exempt obstacle's own footprint
  // is feasible too, e.g. the seat the source is sitting on). A
  // `ring_exempt` obstacle keeps its footprint but drops the clearance ring
  // (e.g. the seat the source is approaching). See ObstacleFreeSpace.
  bool IsFree(const Eigen::Vector2d& point,
              const std::vector<int64_t>& exempt = {},
              const std::vector<int64_t>& ring_exempt = {}) const;
  // The nearest pose in F under the exemption set, from the free space built for that set
  // (exact where NearestFree's query-time exemptions miss a corner of two constraints).
  std::optional<Eigen::Vector2d> NearestFreeUnder(
      const Eigen::Vector2d& point, const std::vector<int64_t>& exempt,
      const std::vector<int64_t>& ring_exempt) const;
  // Every point of [a, b] is free under the same exemptions as IsFree.
  bool SegmentFree(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                   const std::vector<int64_t>& exempt = {},
                   const std::vector<int64_t>& ring_exempt = {}) const;
  std::optional<Eigen::Vector2d> NearestFree(
      const Eigen::Vector2d& point, const std::vector<int64_t>& exempt = {},
      const std::vector<int64_t>& ring_exempt = {}, int64_t* branch = nullptr,
      ObstacleFreeSpace::NearestFreeStatus* status = nullptr) const;
  // Shortest-geodesic point of `ring` in the same metric as GetGeodesicDistance.
  // `final_leg_origin` receives where the final straight leg onto the ring
  // starts (see ObstacleFreeSpace::ClosestOnRing).
  // `open`: obstacle keys of the goal's own footprint, which never blocks a leg onto the ring.
  Eigen::Vector2d ClosestGeodesicOnRing(
      const Eigen::Vector2d& from, const Polygon2D& ring,
      double* length = nullptr,
      Eigen::Vector2d* final_leg_origin = nullptr,
      const std::vector<int64_t>& open = {}) const;

  // Reachability query for scope gating: collision-free connectivity
  // of the free space between two points for the configured body radius.
  // See NavigableGeometryContext::QueryReachability for snap semantics.
  // Single source of truth for reachability; `radius` < 0 = the synced body
  // radius, 0 = point connectivity around raw footprints.
  NavigableGeometryContext::Reachability QueryReachability(
      const Eigen::Vector2d& from, const Eigen::Vector2d& to,
      double radius = -1.0) const;
  // Reachability in the free space under the exemption sets IsFree / NearestFree use.
  NavigableGeometryContext::Reachability QueryReachability(
      const Eigen::Vector2d& from, const Eigen::Vector2d& to,
      const std::vector<int64_t>& exempt,
      const std::vector<int64_t>& ring_exempt, double radius = -1.0) const;
  // Convenience: reachable, with both endpoint snaps <= max_snap_m.
  bool IsReachable(const Eigen::Vector2d& from, const Eigen::Vector2d& to,
                   double max_snap_m = std::numeric_limits<double>::infinity(),
                   double radius = -1.0) const;

  // Geometry context (lazy, cached)
  std::shared_ptr<NavigableGeometryContext> GetGeometryContext() const;

  // Geometric query overrides (delegate to context)
  Eigen::Vector3d GetClosestPoint(const Eigen::Vector3d& from) const override;
  double GetDistance(const Eigen::Vector3d& from) const override;
  Eigen::Vector3d ProjectOnto(const Eigen::Vector3d& point) const override;
  double GetGeodesicDistance(const Eigen::Vector3d& from,
                             const Eigen::Vector3d& to) const override;
  // The same with the goal's own footprints open on the final leg (the referent's: OwnFootprints).
  double GetGeodesicDistance(const Eigen::Vector2d& from, const Eigen::Vector2d& to,
                             const std::vector<int64_t>& open) const;

  // Path planning (delegate to context)
  DiscretePath GetDiscretePath(const Eigen::Vector2d& start,
                                const Eigen::Vector2d& goal) const;

  // Every route GetGeodesicDistance chooses among (see
  // ObstacleFreeSpace::GeodesicRoutes); `under` selects the free space of
  // QueryReachability(exempt, ring_exempt) instead.
  std::vector<ObstacleFreeSpace::RoutePiece> GeodesicRoutes(
      const Eigen::Vector2d& from, const Eigen::Vector2d& to, double slack = 0.0) const;
  std::vector<ObstacleFreeSpace::RoutePiece> GeodesicRoutes(
      const Eigen::Vector2d& from, const Eigen::Vector2d& to, double slack, const std::vector<int64_t>& open) const;
  std::vector<ObstacleFreeSpace::RoutePiece> ContactRoutes(
      const Eigen::Vector2d& from, const std::vector<const std::vector<Eigen::Vector2d>*>& rings, double slack = 0.0) const;
  std::vector<ObstacleFreeSpace::RoutePiece> GeodesicRoutesUnder(
      const Eigen::Vector2d& from, const Eigen::Vector2d& to,
      const std::vector<int64_t>& exempt,
      const std::vector<int64_t>& ring_exempt, double slack = 0.0) const;

  // The cone of directions in which the free space continues from x (see
  // ObstacleFreeSpace::TangentConeAt), under the exemptions of the solve.
  ObstacleFreeSpace::FeasibleCone TangentConeAt(const Eigen::Vector2d& x, const std::vector<int64_t>& exempt,
                                                 const std::vector<int64_t>& ring_exempt, double tol) const;

  // Pre-build spatial indices for predictable query latency.
  // Called during construction to avoid first-frame latency spikes.
  void PreBuildIndices() const;

 private:
  std::shared_ptr<Polygon2D> walkable_;
  std::shared_ptr<Polygon2D> wall_;
  std::vector<Polygon2D> holes_;  // the walkable's holes (not walkable)
  double agent_radius_;
  bool element_local_ = false;
  std::vector<FreeSpaceObstacle> obstacles_;
  double obstacle_clearance_ = 0.0;
  Eigen::Matrix4d local_transform_ = Eigen::Matrix4d::Identity();

  mutable std::shared_ptr<NavigableGeometryContext> context_;
  mutable std::mutex context_mutex_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_ELEMENT_NAVIGABLE_SPATIAL_DEFINITION_H_
