// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/relation_scope.h
//
// THE SCOPE A RELATION IS SOLVED IN: its free space F and the distance over F, one interface for
// every scope geometry. The solve asks the scope; it never asks what kind of space it is. v1 has the same structure: one
// SpaceGeometry serves every relation's distance and feasibility (GetDistance, GetDiscretePath, IsStateValid; v1
// measures every Euclidean relation through GetDiscretePath and projects through the walkable's GetNearestPoint), and
// where no walkable is registered the same calls answer the straight line.
//
// The geodesic is the general computation: the shortest path in F. Where the straight segment lies in F (always, in a
// convex F) it IS the geodesic, by construction of the shortest path, and the route list is that one segment: the
// distance is smooth there and no alternative route ties with it.
//
// Implementations: WalkableScope (a navigable floor: a polygon with obstacle footprints and clearance rings, under a
// solve's exemptions), UnboundedScope (no free-space boundary: F is the whole space), PoseRegionScope (the region volume
// V of an object's affordance-pose SD, affordance_pose_sd.h).
#ifndef SPATIALBLEND_SPACE_RELATION_SCOPE_H_
#define SPATIALBLEND_SPACE_RELATION_SCOPE_H_

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/obstacle_free_space.h"

namespace spatialblend {

// The geometry types a scope answers in. Dimension 2 uses the free-space library's own (the route pieces, the facets and
// the cone as closed arcs of directions); other dimensions use the same fields.
template <int Dim>
struct ScopeGeometryTypes {
  using Vec = Eigen::Matrix<double, Dim, 1>;
  struct Route {
    int64_t node = -1;
    Vec first = Vec::Zero();  // the first waypoint of the route after the query point
    double value = std::numeric_limits<double>::infinity();
    int legs = 1;
    double err = 0.0;
    Vec contact = Vec::Zero();
    int edge = -1;
    bool foot_vertex = false;
    bool flat_foot = false;
  };
  struct Facet {
    Vec normal = Vec::Zero();  // unit, pointing out of F
    double radius = 0.0;
  };
  struct Cone {
    std::vector<Facet> facets;  // the faces of F through the point: the cone is {v : n . v <= 0 for every facet}
    bool Full() const { return facets.empty(); }
    bool Empty() const { return false; }
  };
};
template <>
struct ScopeGeometryTypes<2> {
  using Vec = Eigen::Vector2d;
  using Route = ObstacleFreeSpace::RoutePiece;
  using Facet = ObstacleFreeSpace::Facet;
  using Cone = ObstacleFreeSpace::FeasibleCone;
};

template <int Dim>
struct ScopeReach {
  using Vec = Eigen::Matrix<double, Dim, 1>;
  bool reachable = false;
  double length = std::numeric_limits<double>::infinity();  // the geodesic, including the snap legs
  double from_snap_m = 0.0;
  double to_snap_m = 0.0;
  std::vector<Vec> path;  // the course, from `from` to `to`; empty when not reachable
};

template <int Dim>
class RelationScope {
 public:
  using Types = ScopeGeometryTypes<Dim>;
  using Vec = typename Types::Vec;
  using Route = typename Types::Route;
  using Cone = typename Types::Cone;
  using Reach = ScopeReach<Dim>;
  using NearestFreeStatus = ObstacleFreeSpace::NearestFreeStatus;

  virtual ~RelationScope() = default;

  // Feasibility.
  virtual bool IsFree(const Vec& x) const = 0;
  virtual bool SegmentFree(const Vec& a, const Vec& b) const = 0;
  // The nearest pose in F (nullopt: none); `status` says how it was found.
  virtual std::optional<Vec> NearestFree(const Vec& x, NearestFreeStatus* status = nullptr) const = 0;
  // The same, from the free space built for the solve's exemptions (exact at the corner of two of them).
  virtual std::optional<Vec> NearestFreeUnder(const Vec& x) const = 0;
  // A direction at x projected onto the tangent cone of F there.
  virtual Vec ProjectDirection(const Vec& x, const Vec& v) const = 0;
  // The tangent cone of F at x: the faces of F within `tol` of x.
  virtual Cone TangentConeAt(const Vec& x, double tol) const = 0;

  // The distance over F.
  // The geodesic from `from` to `to` in F, and its course.
  virtual Reach Reachability(const Vec& from, const Vec& to) const = 0;
  // The course to a contact whose own footprints (`open`, the referent's) do not block the final leg onto it: the walkable
  // course, with the open measure as its length (GeodesicRoutesOpen's shortest). Default: Reachability.
  virtual Reach ReachabilityOpen(const Vec& from, const Vec& to, const std::vector<int64_t>& open) const {
    (void)open;
    return Reachability(from, to);
  }
  // The routes the geodesic chooses among whose length is within `slack` of the shortest: in the free space of the
  // solve's exemptions (Under), at radius 0, and to the nearest point of a set of rings. One straight route has no
  // alternative: the list is empty there (the distance is smooth).
  virtual std::vector<Route> GeodesicRoutesUnder(const Vec& from, const Vec& to, double slack) const = 0;
  virtual std::vector<Route> GeodesicRoutes(const Vec& from, const Vec& to, double slack) const = 0;
  // The routes to a contact whose own footprints (`open`, the referent's) do not block the final leg onto it (the
  // measure of a spacing relation's mapped contact, as the source's nearest contact). Default: as GeodesicRoutes.
  virtual std::vector<Route> GeodesicRoutesOpen(const Vec& from, const Vec& to, double slack,
                                                const std::vector<int64_t>& open) const {
    (void)open;
    return GeodesicRoutes(from, to, slack);
  }
  virtual std::vector<Route> ContactRoutes(const Vec& from, const std::vector<const std::vector<Vec>*>& rings,
                                           double slack) const = 0;
  // The most segments a shortest path in F can have (a segment per node of the visibility graph); 0 when every
  // shortest path is one segment.
  virtual double RouteSegmentBound() const = 0;
  // The obstacle clearance the free space keeps (the body radius it was eroded by); 0 when none.
  virtual double ObstacleClearance() const = 0;
  // The diameter of F: no step of a solve in it is longer. +inf for an unbounded F.
  virtual double Diameter() const = 0;
  // Whether F has a boundary at all. It selects no computation: it only names the solve's feasibility outcome
  // (FEASIBLE inside a bounded F, UNCONSTRAINED without one).
  virtual bool Bounded() const = 0;
};

// No free-space boundary: F is the whole space, and the geodesic is the straight segment.
template <int Dim>
class UnboundedScope final : public RelationScope<Dim> {
 public:
  using Base = RelationScope<Dim>;
  using typename Base::Cone;
  using typename Base::NearestFreeStatus;
  using typename Base::Reach;
  using typename Base::Route;
  using typename Base::Vec;
  bool IsFree(const Vec&) const override { return true; }
  bool SegmentFree(const Vec&, const Vec&) const override { return true; }
  std::optional<Vec> NearestFree(const Vec& x, NearestFreeStatus* status = nullptr) const override {
    if (status) *status = NearestFreeStatus::kAlreadyFree;
    return x;
  }
  std::optional<Vec> NearestFreeUnder(const Vec& x) const override { return x; }
  Vec ProjectDirection(const Vec&, const Vec& v) const override { return v; }
  Cone TangentConeAt(const Vec&, double) const override { return FullCone(); }
  Reach Reachability(const Vec& from, const Vec& to) const override {
    Reach r;
    r.reachable = true;
    r.length = (to - from).norm();
    r.path = {from, to};
    return r;
  }
  std::vector<Route> GeodesicRoutesUnder(const Vec&, const Vec&, double) const override { return {}; }
  std::vector<Route> GeodesicRoutes(const Vec&, const Vec&, double) const override { return {}; }
  std::vector<Route> ContactRoutes(const Vec&, const std::vector<const std::vector<Vec>*>&, double) const override {
    return {};
  }
  double RouteSegmentBound() const override { return 0.0; }
  double ObstacleClearance() const override { return 0.0; }
  double Diameter() const override { return std::numeric_limits<double>::infinity(); }
  bool Bounded() const override { return false; }

 private:
  static Cone FullCone();
};
template <int Dim>
typename UnboundedScope<Dim>::Cone UnboundedScope<Dim>::FullCone() {
  return Cone{};
}
template <>
inline UnboundedScope<2>::Cone UnboundedScope<2>::FullCone() {
  Cone c;
  c.arcs = {{0.0, 2.0 * M_PI}};
  return c;
}

// A navigable floor under one solve's exemptions: `exempt` obstacles are open entirely, `ring_exempt` ones keep their
// footprint and drop their clearance ring (relation_solver.cc, GatherEngagedFootprints).
class WalkableScope final : public RelationScope<2> {
 public:
  WalkableScope(const NavigableSpatialDefinition& nav, std::vector<int64_t> exempt, std::vector<int64_t> ring_exempt)
      : nav_(nav), exempt_(std::move(exempt)), ring_exempt_(std::move(ring_exempt)) {}
  bool IsFree(const Vec& x) const override { return nav_.IsFree(x, exempt_, ring_exempt_); }
  bool SegmentFree(const Vec& a, const Vec& b) const override { return nav_.SegmentFree(a, b, exempt_, ring_exempt_); }
  std::optional<Vec> NearestFree(const Vec& x, NearestFreeStatus* status = nullptr) const override {
    return nav_.NearestFree(x, exempt_, ring_exempt_, nullptr, status);
  }
  std::optional<Vec> NearestFreeUnder(const Vec& x) const override {
    return nav_.NearestFreeUnder(x, exempt_, ring_exempt_);
  }
  // The tangent cone is a union of closed arcs of directions: v inside an arc is unchanged; otherwise it is projected
  // onto the nearer end direction of an arc (its component along it, never backwards).
  Vec ProjectDirection(const Vec& x, const Vec& v) const override {
    const Cone cone = TangentConeAt(x, ObstacleFreeSpace::kBoundaryHairM);
    if (cone.Full() || !(v.norm() > 0.0)) return v;
    double th = std::atan2(v.y(), v.x());
    if (th < 0.0) th += 2.0 * M_PI;
    if (cone.Contains(th)) return v;
    Vec best = Vec::Zero();
    for (const auto& arc : cone.arcs)
      for (const double end : {arc.first, arc.second}) {
        const Vec e(std::cos(end), std::sin(end));
        const double along = v.dot(e);
        if (along > 0.0 && (along * e).squaredNorm() > best.squaredNorm()) best = along * e;
      }
    return best;
  }
  Cone TangentConeAt(const Vec& x, double tol) const override {
    return nav_.TangentConeAt(x, exempt_, ring_exempt_, tol);
  }
  Reach Reachability(const Vec& from, const Vec& to) const override {
    const auto q = nav_.QueryReachability(from, to, exempt_, ring_exempt_);
    Reach r;
    r.reachable = q.reachable;
    r.length = q.geodesic_distance_m;
    r.from_snap_m = q.from_snap_m;
    r.to_snap_m = q.to_snap_m;
    if (q.reachable) r.path = q.path.GetPoints();
    return r;
  }
  std::vector<Route> GeodesicRoutesUnder(const Vec& from, const Vec& to, double slack) const override {
    return nav_.GeodesicRoutesUnder(from, to, exempt_, ring_exempt_, slack);
  }
  std::vector<Route> GeodesicRoutes(const Vec& from, const Vec& to, double slack) const override {
    return nav_.GeodesicRoutes(from, to, slack);
  }
  std::vector<Route> GeodesicRoutesOpen(const Vec& from, const Vec& to, double slack,
                                        const std::vector<int64_t>& open) const override {
    return nav_.GeodesicRoutes(from, to, slack, open);
  }
  Reach ReachabilityOpen(const Vec& from, const Vec& to, const std::vector<int64_t>& open) const override {
    // The walkable course is the scope's (through F, which a footprint's boundary is not in); its LENGTH is the open
    // measure (the referent's own footprints touched by the final leg), the distance the relation's term measures.
    Reach r = Reachability(from, to);
    if (!r.reachable) return r;
    const auto routes = nav_.GeodesicRoutes(from, to, 0.0, open);
    if (!routes.empty() && std::isfinite(routes.front().value)) r.length = routes.front().value;
    return r;
  }
  std::vector<Route> ContactRoutes(const Vec& from, const std::vector<const std::vector<Vec>*>& rings,
                                   double slack) const override {
    return nav_.ContactRoutes(from, rings, slack);
  }
  double RouteSegmentBound() const override {
    const auto ctx = nav_.GetGeometryContext();
    return ctx ? static_cast<double>(ctx->FreeSpace(0.0).GraphNodeCount()) : 0.0;
  }
  double ObstacleClearance() const override { return nav_.GetObstacleClearance(); }
  double Diameter() const override {
    const Bound2D b = nav_.GetBounds();
    return (b.max - b.min).norm();
  }
  bool Bounded() const override { return true; }

 private:
  const NavigableSpatialDefinition& nav_;
  std::vector<int64_t> exempt_, ring_exempt_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_RELATION_SCOPE_H_
