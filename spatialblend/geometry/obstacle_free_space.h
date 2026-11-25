// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/obstacle_free_space.h
#ifndef SPATIALBLEND_GEOMETRY_OBSTACLE_FREE_SPACE_H_
#define SPATIALBLEND_GEOMETRY_OBSTACLE_FREE_SPACE_H_

#include <algorithm>
#include <Eigen/Dense>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {

// One obstacle footprint in the walkable's (world) frame. `key` is an opaque
// caller identity (the Space uses the obstacle element's local id) so queries
// can exempt specific obstacles (e.g. the seat the source is sitting on).
struct FreeSpaceObstacle {
  int64_t key = 0;
  std::shared_ptr<const Polygon2D> footprint;
};

// Free space of a point body (the root anchor) inside a walkable polygon with
// interior obstacles:
//
//   F = { p : p in walkable  AND  dist(p, O_i) >= clearance for every O_i }
//
// The walkable polygon is taken as given (callers that want wall clearance
// erode it first, as the transfer harness does; NavigableSpatialDefinition
// erodes by its agent_radius). Obstacles are raw footprints; the clearance
// (body radius) is applied here, exactly (segment-to-polygon distance), so the
// body DISC never enters an obstacle. With clearance 0 the free space is the
// walkable minus the obstacle interiors (paths may touch a footprint).
//
// Paths: shortest paths in a reduced visibility graph. Its nodes are the
// vertices of the configuration-space polygon  walkable \ U buffer(O_i, r')
// computed with Boost.Geometry (r' = clearance / cos(pi/64) * 1.0001 + a hair,
// a 64-gon circumscribing every clearance disc; see kArcPoints in the .cc), so
// the node set includes the intersection points of overlapping clearance rings
// and of rings with walls -- every gap the exact free space leaves open, down
// to 2 (r' - r) (r' - r is about 0.13% of the clearance plus a hair), is represented.
// A narrower gap is closed for the graph although the exact straight-segment
// test can pass along it (a known limit). Edges
// are bitangent candidates validated by the EXACT segment test, so every edge
// keeps >= clearance.
//
// A path whose endpoints are not free starts/ends with an explicit straight
// "snap" leg to the nearest free point; the result reports both snap
// distances so callers can decide what an endpoint inside an obstacle means
// (e.g. a referent's own footprint).
//
// Immutable after construction; all queries are const and thread-safe.
class ObstacleFreeSpace {
 public:
  // The offset, in metres, at which the free space places a point it moves off an
  // obstacle's footprint: a point ON a footprint's boundary is its own nearest free
  // point up to this hair. A snap of at most this is "on F", not a snap.
  static constexpr double kBoundaryHairM = 1e-6;
  // `bare_keys`: obstacles of this free space that carry NO clearance ring (their footprint
  // interior stays excluded, nothing around it): the free space of a body that stands on or
  // touches them. It is the same space as the query-time `ring_exempt` of IsFree /
  // SegmentFree / NearestFree, fixed at construction so the visibility graph (reachability,
  // paths) is built over it too: feasibility, the clamp and reachability are then one
  // geometric representation, the exact round clearance ring everywhere.
  ObstacleFreeSpace(std::shared_ptr<const Polygon2D> walkable,
                    std::vector<FreeSpaceObstacle> obstacles,
                    double clearance,
                    std::vector<int64_t> bare_keys = {});

  double clearance() const { return clearance_; }

  // Whether `p` is in the walkable region, with the free space's own boundary tolerance (a
  // point a rounding error outside the boundary is ON it): the walkable test IsFree and
  // the searches use, public so a caller deciding "this pose is in the space" agrees with
  // them. Contains() of the spatial definition is exact and stays the containment/scope test.
  bool IsInWalkable(const Eigen::Vector2d& p) const { return InWalkable(p); }
  bool HasObstacles() const { return !obstacles_.empty(); }
  const std::vector<FreeSpaceObstacle>& obstacles() const {
    return obstacles_;
  }

  // Distance from p to the nearest (non-exempt) obstacle footprint; 0 when p
  // is inside one; +inf when there is none.
  double ObstacleDistance(const Eigen::Vector2d& p,
                          const std::vector<int64_t>& exempt = {}) const;

  // True iff p is in the walkable polygon and keeps >= clearance from every
  // non-exempt obstacle. Engagement exemptions open a footprint or its
  // clearance ring; they do not extend F outside the navigable domain. For a
  // `ring_exempt` obstacle only its footprint INTERIOR is excluded (no
  // clearance ring): the root may stand as close to it as the source does,
  // never on it.
  bool IsFree(const Eigen::Vector2d& p,
              const std::vector<int64_t>& exempt = {},
              const std::vector<int64_t>& ring_exempt = {}) const;

  // Nearest free point to p (p itself when free).
  //
  // `status`, when given, distinguishes a projection from the two null results:
  // kDomainEmpty means the navigable region itself is empty (an erosion that
  // removed it); kSearchFoundNone means this search found no free candidate
  // and does not prove the domain is empty.
  //
  // `branch`, when given, receives the identity of the constraint the answer lies
  // on: the obstacle pushed out of (and, for a point INSIDE its footprint, the
  // footprint edge it is nearest to), a wall, or an edge of the configuration-space
  // boundary; 0 when p is free. The nearest-point map is continuous inside one
  // branch and can jump only where the branch changes (the medial axis of the free
  // space), so a caller can name that event.
  enum class NearestFreeStatus {
    kAlreadyFree = 0,
    kProjected = 1,
    kSearchFoundNone = 2,
    kDomainEmpty = 3,
  };
  std::optional<Eigen::Vector2d> NearestFree(
      const Eigen::Vector2d& p, const std::vector<int64_t>& exempt = {},
      const std::vector<int64_t>& ring_exempt = {},
      int64_t* branch = nullptr,
      NearestFreeStatus* status = nullptr) const;

  // True when the navigable polygon was eroded away (no component left). An
  // absent walkable is unconstrained, not empty.
  bool DomainEmpty() const;

  // True iff every point of segment [a, b] is inside the walkable polygon and
  // keeps >= clearance from every non-exempt obstacle; for a `ring_exempt`
  // obstacle the segment may enter its clearance ring but never its
  // footprint interior.
  bool SegmentFree(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                   const std::vector<int64_t>& exempt = {},
                   const std::vector<int64_t>& ring_exempt = {}) const;

  struct PathResult {
    // False when the free space does not connect the (snapped) endpoints.
    bool found = false;
    // Set with !found: the navigable region is empty, or the search found no
    // free endpoint. They are different facts.
    bool domain_empty = false;
    bool search_found_none = false;
    // from, [from_free], waypoints..., [to_free], to. Empty when !found.
    std::vector<Eigen::Vector2d> points;
    double length = std::numeric_limits<double>::infinity();
    Eigen::Vector2d from_free = Eigen::Vector2d::Zero();
    Eigen::Vector2d to_free = Eigen::Vector2d::Zero();
    double from_snap_m = 0.0;
    double to_snap_m = 0.0;
  };

  // Shortest free path from `from` to `to` in the same F as IsFree / SegmentFree.
  // The straight segment is used when it is free. Otherwise the endpoints are
  // snapped to the nearest free point and joined through the visibility graph.
  // A null snap is domain_empty or search_found_none, not a path.
  PathResult FindPath(const Eigen::Vector2d& from,
                      const Eigen::Vector2d& to) const;

  // Length of FindPath(from, to). The visibility-graph distances from `to`
  // are built once (Dijkstra from the contact, then a straight leg at the
  // query) and kept while this free space and the contact are unchanged.
  // The free space is dropped when the geometry changes, so the cache key
  // is the geometry revision plus the contact.
  // `open_keys`: obstacle keys of the goal's own footprints (a referent's), which never block the final leg onto the
  // goal (the same rule as ClosestOnRing's `own`, SegmentReachesContact). Empty: every footprint blocks.
  double GeodesicDistance(const Eigen::Vector2d& from,
                          const Eigen::Vector2d& to,
                          const std::vector<int64_t>& open_keys = {}) const;

  // Point of `ring` at the shortest free-space path from `from`. The last
  // segment may end on the ring. A straight segment that reaches the ring is
  // optimal, because a geodesic is at least the Euclidean length; otherwise
  // the optimum is a foot from a visibility-graph node onto a ring edge.
  // `length`, when given, receives that path length (+inf when none).
  // `final_leg_origin`, when given, receives where the final straight leg onto
  // the ring starts: `from` itself when it sees the contact, else the last
  // waypoint of the geodesic (a visibility-graph node, or the free point `from`
  // snaps to). v1 takes the approach direction from this leg.
  //
  // `open`: keys of obstacles whose footprint is the GOAL's own (an element's contact region lies
  // inside its own footprint: a seat part inside its chair). Such a footprint never blocks a leg
  // onto the ring: reaching the contact is entering the footprint (v1: reaching a target is entering
  // its affordance volume). It still blocks every other path.
  Eigen::Vector2d ClosestOnRing(const Eigen::Vector2d& from,
                                const Polygon2D& ring,
                                double* length = nullptr,
                                Eigen::Vector2d* final_leg_origin = nullptr,
                                const std::vector<int64_t>& open = {}) const;

  // The per-query search ClosestOnRing replaces. Lengths match within
  // numerical tolerance; the field is built once per ring.
  Eigen::Vector2d ClosestOnRingUncached(const Eigen::Vector2d& from,
                                        const Polygon2D& ring,
                                        double* length = nullptr,
                                        const std::vector<int64_t>& open = {}) const;

  // One route that GeodesicDistance(from, to) chooses among: the straight leg
  // (node = -1) or the hop through visibility-graph node `node`. `first` is the
  // first waypoint after `from` (the gradient of the route's length at `from` is
  // unit(from - first)), `value` its total length (snap legs included) and `legs`
  // the number of straight segments it is made of. The arithmetic error of `value`
  // is `err` = gamma_{legs+3} * value (Higham, Accuracy and Stability of Numerical
  // Algorithms, 2nd ed., sec. 4.2: a segment length has relative error gamma_4 and the
  // sum of `legs` non-negative terms adds gamma_{legs-1}).
  struct RoutePiece {
    int64_t node = -1;
    Eigen::Vector2d first = Eigen::Vector2d::Zero();
    double value = std::numeric_limits<double>::infinity();
    int legs = 1;
    double err = 0.0;
    // For a route to a contact on a boundary (ContactRoutes): the contact point, the index of the boundary edge it lies
    // on, and whether it is a vertex of that boundary. A direct (straight) route to a contact in the INTERIOR of an
    // edge has a distance with no curvature along the edge (the distance to a line): `flat_foot`.
    Eigen::Vector2d contact = Eigen::Vector2d::Zero();
    int edge = -1;
    bool foot_vertex = false;
    bool flat_foot = false;
  };
  // The routes GeodesicDistance(from, to) chooses among whose length is within
  // `slack` of the shortest, shortest first (so [0].value == the distance). With
  // slack = 0 these are the routes the shortest ties with to double rounding at
  // most; a larger slack lists the routes a step of length slack / 2 could make
  // the shortest. Empty when the distance is infinite.
  std::vector<RoutePiece> GeodesicRoutes(const Eigen::Vector2d& from,
                                         const Eigen::Vector2d& to,
                                         double slack = 0.0,
                                         const std::vector<int64_t>& open_keys = {}) const;
  // The same for a relation that measures the geodesic to the NEAREST point of a polygon's boundary (a border
  // relation): one piece per (boundary edge, route), the contact of each being the nearest point of that edge, so a tie
  // of two edges (the medial axis of the polygon) lists both. Shortest first; empty when no edge is reachable.
  std::vector<RoutePiece> ContactRoutes(const Eigen::Vector2d& from,
                                        const std::vector<const std::vector<Eigen::Vector2d>*>& rings,
                                        double slack = 0.0) const;

  // The cone of directions in which F continues from x, exactly: a closed union
  // of angular intervals built from the walkable's edges and vertices and the
  // obstacles' clearance rings (or footprint edges and vertices when the ring
  // is exempt) that pass within `tol` of x. Nothing is probed or sampled. The
  // facets that bound it are listed with their outward unit normals.
  struct Facet {
    enum Kind { kWallEdge, kWallVertex, kRingArc, kRingEdge, kFootprintEdge, kFootprintVertex };
    Kind kind = kWallEdge;
    Eigen::Vector2d normal = Eigen::Vector2d::Zero();  // unit, pointing out of F
    Eigen::Vector2d tangent = Eigen::Vector2d::Zero(); // unit, normal rotated +90 degrees
    double radius = 0.0;                               // > 0: a circular arc of this radius
    int64_t obstacle_key = 0;                          // 0 for the walkable
  };
  struct FeasibleCone {
    // Closed arcs [lo, hi] of directions in radians, 0 <= lo <= hi <= 2 pi; the
    // full circle is the single arc [0, 2 pi].
    std::vector<std::pair<double, double>> arcs;
    std::vector<Facet> facets;
    bool Full() const { return arcs.size() == 1 && arcs[0].first <= 0.0 && arcs[0].second >= 6.283185307179586; }
    bool Empty() const { return arcs.empty(); }
    bool Contains(double angle) const;
  };
  FeasibleCone TangentConeAt(const Eigen::Vector2d& x, const std::vector<int64_t>& exempt,
                             const std::vector<int64_t>& ring_exempt, double tol) const;

  // Number of visibility-graph nodes / edges (builds the graph; for
  // diagnostics and benchmarks).
  size_t GraphNodeCount() const;
  size_t GraphEdgeCount() const;

 private:
  // NearestFree, optionally restricted to snap legs that do not pass through
  // an obstacle p is not inside (FindPath endpoint snapping: an anchor inside
  // a couch flush against a thin wall must come out on its own side). Falls
  // back to the unrestricted nearest point when no such candidate exists.
  std::optional<Eigen::Vector2d> NearestFreeImpl(
      const Eigen::Vector2d& p, const std::vector<int64_t>& exempt,
      const std::vector<int64_t>& ring_exempt, bool avoid_crossing,
      int64_t* branch = nullptr,
      NearestFreeStatus* status = nullptr) const;

  struct Node {
    Eigen::Vector2d p;
    Eigen::Vector2d prev, next;  // ring neighbours (bitangent test)
    // Tangent() inputs derived once per node, so the per-query pruning loop
    // (it runs over every node) does no subtraction or square root.
    Eigen::Vector2d to_prev, to_next;  // prev - p, next - p
    // Inputs of each determinant's own bound (Tangent): the L1 norm of the
    // edge vector, the largest coordinate magnitude of its ends, its length.
    double prev_l1 = 0.0, next_l1 = 0.0, prev_scale = 0.0, next_scale = 0.0;
    double prev_len = 0.0, next_len = 0.0;
    Node(const Eigen::Vector2d& p_, const Eigen::Vector2d& prev_,
         const Eigen::Vector2d& next_)
        : p(p_), prev(prev_), next(next_), to_prev(prev_ - p_),
          to_next(next_ - p_), prev_l1(to_prev.cwiseAbs().sum()),
          next_l1(to_next.cwiseAbs().sum()),
          prev_scale(std::max(p_.cwiseAbs().maxCoeff(), prev_.cwiseAbs().maxCoeff())),
          next_scale(std::max(p_.cwiseAbs().maxCoeff(), next_.cwiseAbs().maxCoeff())),
          prev_len(to_prev.norm()), next_len(to_next.norm()) {}
  };

  bool ViolatesClearance(double distance) const;
  bool InWalkable(const Eigen::Vector2d& p) const;
  bool SegmentInWalkable(const Eigen::Vector2d& a,
                         const Eigen::Vector2d& b) const;
  // Segment stays in the walkable and does not enter an obstacle interior.
  // The endpoint may lie on a footprint boundary (a contact). With a positive
  // clearance the open segment keeps that clearance.
  // Obstacles in `open` do not block it (the goal's own footprint).
  bool SegmentTouchesContact(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                             const std::vector<int64_t>& own) const;
  bool SegmentReachesContact(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                             const std::vector<int64_t>& open = {}) const;
  bool IsExempt(size_t i, const std::vector<int64_t>& exempt) const;
  std::vector<int64_t> KeysAround(const Eigen::Vector2d& p) const;
  // A line from `from` through node n does not cut into the blocked side at
  // n (both ring neighbours on one side): only such edges can lie on a
  // shortest path.
  static bool Tangent(const Node& n, const Eigen::Vector2d& from);

  std::shared_ptr<const Polygon2D> walkable_;
  std::vector<FreeSpaceObstacle> obstacles_;
  std::vector<Eigen::Vector2d> obstacle_lo_, obstacle_hi_;  // bbox + clearance
  std::vector<Polygon2D> touch_;  // footprint grown by a hair (ring-exempt)
  std::vector<char> bare_;        // per obstacle: no clearance ring (constructor `bare_keys`)
  double clearance_ = 0.0;
  double buffer_radius_ = 0.0;  // configuration-space buffer vertex radius
  bool walkable_convex_ = false;  // walkable polygon is convex (fast paths)
  std::vector<bool> obstacle_in_walkable_;  // footprint inside a convex walkable

  PathResult FindPathUncached(const Eigen::Vector2d& from,
                              const Eigen::Vector2d& to) const;
  // Endpoint joined to the visibility graph: a short free leg out of the
  // clearance band, then bitangent pruning. Same attachment as FindPath.
  struct EndpointAttach {
    Eigen::Vector2d anchor = Eigen::Vector2d::Zero();
    double lead = 0.0;
    bool prune = false;
  };
  EndpointAttach AttachEndpoint(const Eigen::Vector2d& e) const;
  // Shortest-path distance from each visibility-graph node to one contact.
  struct GoalField {
    Eigen::Vector2d goal_free = Eigen::Vector2d::Zero();
    Eigen::Vector2d goal_end = Eigen::Vector2d::Zero();  // the goal's foot on F, where the last leg ends (FootOnF)
    Eigen::Vector2d goal = Eigen::Vector2d::Zero();      // the goal itself: the distances run to it
    std::vector<int64_t> open;                            // the goal's own footprints (never block the final leg)
    std::vector<double> dist;
    std::vector<int> legs;  // straight segments of node i's route to the goal
    std::vector<double> cell_min;
    // Last winning hop for this field. The Jacobian sample and the line
    // search trial are neighbours, so the next query starts from it.
    mutable size_t hop_hint = static_cast<size_t>(-1);
  };
  GoalField BuildGoalField(const Eigen::Vector2d& goal_free, const Eigen::Vector2d& goal_end,
                           const Eigen::Vector2d& goal, const std::vector<int64_t>& open_keys) const;
  // Pointer into goal_fields_. Stable until the next GoalFieldFor that
  // inserts. The query uses it immediately and does not re-enter.
  const GoalField* GoalFieldFor(const Eigen::Vector2d& goal_free, const Eigen::Vector2d& goal_end,
                                const Eigen::Vector2d& goal, const std::vector<int64_t>& open_keys) const;

  // Geodesic from each visibility-graph node to a support ring, plus the
  // contact that realises it. The ring is fixed for a solve; the trial
  // point only pays a straight leg onto the field.
  struct RingField {
    std::vector<Eigen::Vector2d> key;
    std::vector<int64_t> open_key;  // the goal's own footprints (see ClosestOnRing)
    std::vector<double> dist;
    std::vector<Eigen::Vector2d> foot;
    // The node whose own straight leg reaches foot[i]: the last waypoint of
    // node i's geodesic to the ring (i itself when its leg is direct).
    std::vector<size_t> last;
    std::vector<double> cell_min;
    // Last winning hop. One field per referent, kept across Gauss-Newton
    // iterations; each referent keeps its own hop.
    mutable size_t hop_hint = static_cast<size_t>(-1);
  };
  RingField BuildRingField(
      const std::vector<const std::vector<Eigen::Vector2d>*>& rings,
      const std::vector<int64_t>& open) const;
  const RingField* RingFieldFor(
      const std::vector<const std::vector<Eigen::Vector2d>*>& rings,
      const std::vector<int64_t>& open) const;
  void EnsureNodeGrid() const;
  // Smallest free straight leg onto a node field. n on failure.
  size_t BestHop(const EndpointAttach& at, const std::vector<double>& dist,
                 const std::vector<double>& cell_min, double* f,
                 size_t* hint) const;
  struct MemoEntry {
    Eigen::Vector2d from, to;
    PathResult result;
  };
  static constexpr size_t kMemoSize = 16;
  mutable std::vector<MemoEntry> memo_;
  mutable std::mutex memo_mutex_;
  static constexpr size_t kGoalFieldCap = 64;
  mutable std::vector<GoalField> goal_fields_;
  // NearestFreeImpl(to, avoid_crossing) of a goal contact, by exact contact.
  struct GoalSnapEntry {
    Eigen::Vector2d to;
    std::optional<Eigen::Vector2d> snap;
  };
  std::optional<Eigen::Vector2d> GoalSnap(const Eigen::Vector2d& to) const;
  mutable std::vector<GoalSnapEntry> goal_snaps_;
  static constexpr size_t kRingFieldCap = 16;
  mutable std::vector<RingField> ring_fields_;
  mutable std::vector<char> query_skip_;
  mutable bool node_grid_ready_ = false;
  mutable double grid_cell_ = 1.0;
  mutable Eigen::Vector2d grid_origin_ = Eigen::Vector2d::Zero();
  mutable int grid_nx_ = 0;
  mutable int grid_ny_ = 0;
  mutable std::vector<std::vector<size_t>> node_grid_;

  // Lazily built configuration-space boundary and visibility graph.
  void EnsureGraph() const;
  mutable std::once_flag graph_once_;
  mutable std::vector<Node> nodes_;
  mutable std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>
      cspace_edges_;  // boundary segments of walkable \ U buffers
  mutable std::vector<Eigen::Vector2d> cspace_edge_lo_, cspace_edge_hi_;
  struct CspacePolygon {
    std::vector<Eigen::Vector2d> outer;
    std::vector<std::vector<Eigen::Vector2d>> inners;
    // Bounding boxes (filled once the polygons are built): a point outside a
    // ring's box is outside the ring, so InCspace skips the O(ring) test.
    Eigen::Vector2d outer_lo, outer_hi;
    std::vector<Eigen::Vector2d> inner_lo, inner_hi;
  };
  mutable std::vector<CspacePolygon> cspace_polys_;
  bool InCspace(const Eigen::Vector2d& p) const;
  std::optional<Eigen::Vector2d> NearestCspacePoint(
      const Eigen::Vector2d& p) const;
  mutable std::vector<std::vector<std::pair<size_t, double>>> adjacency_;
  mutable size_t edge_count_ = 0;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_GEOMETRY_OBSTACLE_FREE_SPACE_H_
