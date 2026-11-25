// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/navigable_spatial_definition.h"

#include <limits>

#include <cmath>

#include "spatialblend/geometry/polygon_coordinate_mapper.h"

namespace spatialblend {

namespace {

// Forward convention for interior charts: +Y is the element's functional front
// in its local frame (matches the grounding rules). This MUST be honored by the
// corresponding element in the other space, or the map mirrors/rotates.
const Eigen::Vector2d kInteriorForward(0.0, 1.0);

// Fraction of the local-frame axis-aligned bounding box that the walkable area
// fills. ~1 ⇒ box-like (box-UVW is affine-exact); lower ⇒ irregular (use MVC).
double BoxFillRatio(const std::vector<Eigen::Vector2d>& pts) {
  if (pts.size() < 3) return 1.0;
  Eigen::Vector2d mn = pts[0], mx = pts[0];
  double area2 = 0.0;
  for (size_t i = 0; i < pts.size(); ++i) {
    mn = mn.cwiseMin(pts[i]);
    mx = mx.cwiseMax(pts[i]);
    const Eigen::Vector2d& a = pts[i];
    const Eigen::Vector2d& b = pts[(i + 1) % pts.size()];
    area2 += a.x() * b.y() - b.x() * a.y();  // shoelace
  }
  double box = (mx.x() - mn.x()) * (mx.y() - mn.y());
  if (box < 1e-9) return 1.0;
  return std::abs(area2) * 0.5 / box;
}

}  // namespace

NavigableSpatialDefinition::NavigableSpatialDefinition(
    std::shared_ptr<Polygon2D> walkable,
    std::shared_ptr<Polygon2D> wall,
    double agent_radius,
    GeometryRole geometry_role)
    : SpatialDefinition(geometry_role),
      walkable_(std::move(walkable)),
      wall_(std::move(wall)),
      agent_radius_(agent_radius) {
  if (walkable_)
    for (auto& ring : walkable_->Holes()) holes_.emplace_back(std::move(ring));
  PreBuildIndices();
}

bool NavigableSpatialDefinition::Contains(
    const Eigen::Vector2d& point) const {
  return walkable_ && walkable_->Contains(point);
}

std::shared_ptr<Polygon2D> NavigableSpatialDefinition::GetBoundary() const {
  return walkable_;
}

const Eigen::Matrix4d& NavigableSpatialDefinition::GetLocalTransform() const {
  return local_transform_;
}

NormalizedInterior NavigableSpatialDefinition::EncodeInterior(
    const Eigen::Vector3d& world_point) const {
  // Always populate the universal box-UVW fallback.
  NormalizedInterior n = EncodeBoxUVW(world_point);
  if (!walkable_) return n;

  // Box-like walkables keep the exact box-UVW chart; only irregular shapes pay
  // for the richer (and approximate) MVC channel.
  constexpr double kBoxLikeThreshold = 0.9;
  if (BoxFillRatio(walkable_->GetPoints()) >= kBoxLikeThreshold) return n;

  // Irregular: add Mean Value Coordinates over the local-frame boundary. The
  // sample count is a fixed shared constant (kDefaultNumSamples) so the receiver
  // resamples identically — never negotiated.
  Eigen::Vector3d local =
      (GetTransform().inverse() * world_point.homogeneous()).head<3>();
  // Force the MVC channel for a non-box interior: resample the boundary exactly
  // as DecodeMVC does, then compute Mean Value Coordinates against those samples
  // so the receiver reconstructs identically.
  auto samples = PolygonCoordinateMapper::ResampleBoundary(
      *walkable_, kInteriorForward,
      PolygonCoordinateMapper::kDefaultNumSamples);
  auto weights = PolygonCoordinateMapper::ComputeMVCWeights(
      Eigen::Vector2d(local.x(), local.y()), samples);
  if (!weights.empty()) {
    n.kind = NormalizedInterior::Kind::kMVC;
    n.weights = std::move(weights);
  }
  return n;
}

Eigen::Vector3d NavigableSpatialDefinition::DecodeInterior(
    const NormalizedInterior& c) const {
  if (c.kind == NormalizedInterior::Kind::kMVC && !c.weights.empty() &&
      walkable_) {
    NormalizedPolygonPosition np;
    np.strategy = NormalizedPolygonPosition::Strategy::kMVC;
    np.mvc_weights = c.weights;
    // Decode rejects a weight/sample-count mismatch; fall back to uvw then.
    auto dec = PolygonCoordinateMapper::Decode(
        np, *walkable_, kInteriorForward,
        PolygonCoordinateMapper::kDefaultNumSamples);
    if (dec.ok()) {
      // Containment clamp: an MVC convex-combination of boundary samples can
      // land in a concavity / outside a non-convex polygon. Project back in.
      Eigen::Vector2d local2(dec->x(), dec->y());
      if (!walkable_->Contains(local2)) {
        local2 = walkable_->GetNearestPoint(local2);
      }
      return (GetTransform() *
              Eigen::Vector4d(local2.x(), local2.y(), 0.0, 1.0))
          .head<3>();
    }
  }
  return DecodeBoxUVW(c);  // universal fallback
}

Bound2D NavigableSpatialDefinition::GetBounds() const {
  Bound2D bounds;
  if (!walkable_) return bounds;
  const auto& points = walkable_->GetPoints();
  if (points.empty()) return bounds;

  bounds.min = points[0];
  bounds.max = points[0];
  for (const auto& p : points) {
    bounds.min = bounds.min.cwiseMin(p);
    bounds.max = bounds.max.cwiseMax(p);
  }
  return bounds;
}

std::shared_ptr<Polygon2D> NavigableSpatialDefinition::GetWalkablePolygon() const {
  return walkable_;
}

std::shared_ptr<Polygon2D> NavigableSpatialDefinition::GetWall() const {
  return wall_;
}

void NavigableSpatialDefinition::SetAgentRadius(double radius) {
  std::lock_guard<std::mutex> lock(context_mutex_);
  agent_radius_ = radius;
  context_.reset();
}

double NavigableSpatialDefinition::GetAgentRadius() const {
  return agent_radius_;
}

std::shared_ptr<NavigableGeometryContext>
NavigableSpatialDefinition::GetGeometryContext() const {
  std::lock_guard<std::mutex> lock(context_mutex_);
  if (!context_ && walkable_) {
    // A hole of the walkable (a pocket a scope extension encloses, element/scope_extension.h)
    // is not walkable: the free space treats it as an obstacle. Its keys are below any
    // element's id, so no exemption names one.
    std::vector<FreeSpaceObstacle> obstacles = obstacles_;
    int64_t hole_key = std::numeric_limits<int64_t>::min();
    for (auto& hole : walkable_->Holes())
      obstacles.push_back({hole_key++, std::make_shared<Polygon2D>(std::move(hole))});
    context_ = obstacles.empty()
                   ? NavigableGeometryContext::Create(walkable_, agent_radius_)
                   : NavigableGeometryContext::Create(walkable_, agent_radius_,
                                                      obstacles,
                                                      obstacle_clearance_);
  }
  return context_;
}

namespace {
bool SamePolygon(const Polygon2D* a, const Polygon2D* b) {
  if (a == b) return true;
  if (!a || !b) return false;
  const auto& pa = a->GetPoints();
  const auto& pb = b->GetPoints();
  if (pa.size() != pb.size()) return false;
  for (size_t i = 0; i < pa.size(); ++i) {
    if ((pa[i] - pb[i]).squaredNorm() > 1e-18) return false;
  }
  return true;
}
}  // namespace

bool NavigableSpatialDefinition::SetObstacles(
    std::vector<FreeSpaceObstacle> obstacles, double clearance) {
  std::lock_guard<std::mutex> lock(context_mutex_);
  if (!std::isfinite(clearance) || clearance < 0.0) clearance = 0.0;
  bool same = clearance == obstacle_clearance_ &&
              obstacles.size() == obstacles_.size();
  for (size_t i = 0; same && i < obstacles.size(); ++i) {
    same = obstacles[i].key == obstacles_[i].key &&
           SamePolygon(obstacles[i].footprint.get(),
                       obstacles_[i].footprint.get());
  }
  if (same) return false;
  obstacles_ = std::move(obstacles);
  obstacle_clearance_ = clearance;
  context_.reset();
  return true;
}

void NavigableSpatialDefinition::PrewarmFreeSpaces() const {
  auto ctx = GetGeometryContext();
  if (!ctx || !ctx->HasObstacles()) return;
  ctx->FreeSpace(0.0).GraphNodeCount();
  ctx->FreeSpace().GraphNodeCount();
}

const std::vector<FreeSpaceObstacle>& NavigableSpatialDefinition::GetObstacles()
    const {
  return obstacles_;
}

double NavigableSpatialDefinition::GetObstacleClearance() const {
  return obstacle_clearance_;
}

bool NavigableSpatialDefinition::IsInWalkable(const Eigen::Vector2d& point) const {
  if (!walkable_) return false;
  // A hole of the walkable (a pocket a scope extension keeps out) is not walkable, as IsFree and
  // Contains read it; its boundary is (the boundary tolerance of the free space).
  for (const Polygon2D& hole : holes_)
    if (hole.Contains(point, /*include_boundary=*/false)) return false;
  auto ctx = GetGeometryContext();
  if (!ctx) return walkable_->Contains(point);
  return ctx->FreeSpace().IsInWalkable(point);
}

bool NavigableSpatialDefinition::IsFree(
    const Eigen::Vector2d& point, const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt) const {
  auto ctx = GetGeometryContext();
  if (!ctx) return true;
  return ctx->FreeSpace().IsFree(point, exempt, ring_exempt);
}

std::optional<Eigen::Vector2d> NavigableSpatialDefinition::NearestFreeUnder(
    const Eigen::Vector2d& point, const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt) const {
  auto ctx = GetGeometryContext();
  if (!ctx) return point;
  return ctx->NearestFreeUnder(point, exempt, ring_exempt);
}

bool NavigableSpatialDefinition::SegmentFree(
    const Eigen::Vector2d& a, const Eigen::Vector2d& b,
    const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt) const {
  auto ctx = GetGeometryContext();
  if (!ctx) return true;
  return ctx->FreeSpace().SegmentFree(a, b, exempt, ring_exempt);
}

std::optional<Eigen::Vector2d> NavigableSpatialDefinition::NearestFree(
    const Eigen::Vector2d& point, const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt, int64_t* branch,
    ObstacleFreeSpace::NearestFreeStatus* status) const {
  if (branch) *branch = 0;
  auto ctx = GetGeometryContext();
  if (!ctx) return point;
  return ctx->FreeSpace().NearestFree(point, exempt, ring_exempt, branch,
                                         status);
}

Eigen::Vector2d NavigableSpatialDefinition::ClosestGeodesicOnRing(
    const Eigen::Vector2d& from, const Polygon2D& ring, double* length,
    Eigen::Vector2d* final_leg_origin, const std::vector<int64_t>& open) const {
  if (final_leg_origin) *final_leg_origin = from;
  auto ctx = GetGeometryContext();
  if (!ctx) {
    const Eigen::Vector2d p =
        ring.NumPoints() >= 1 ? ring.GetNearestPoint(from) : from;
    if (length) *length = (p - from).norm();
    return p;
  }
  return ctx->ClosestGeodesicOnRing(from, ring, length, final_leg_origin, open);
}

NavigableGeometryContext::Reachability
NavigableSpatialDefinition::QueryReachability(const Eigen::Vector2d& from,
                                              const Eigen::Vector2d& to,
                                              double radius) const {
  auto ctx = GetGeometryContext();
  if (!ctx) {
    NavigableGeometryContext::Reachability r;
    r.reachable = true;
    r.geodesic_distance_m = (to - from).norm();
    r.path.AddPoint(from);
    r.path.AddPoint(to);
    return r;
  }
  return ctx->QueryReachability(from, to, radius);
}

NavigableGeometryContext::Reachability
NavigableSpatialDefinition::QueryReachability(const Eigen::Vector2d& from,
                                              const Eigen::Vector2d& to,
                                              const std::vector<int64_t>& exempt,
                                              const std::vector<int64_t>& ring_exempt,
                                              double radius) const {
  auto ctx = GetGeometryContext();
  if (!ctx) return QueryReachability(from, to, -1.0);
  return ctx->QueryReachabilityUnder(from, to, exempt, ring_exempt, radius);
}

bool NavigableSpatialDefinition::IsReachable(const Eigen::Vector2d& from,
                                             const Eigen::Vector2d& to,
                                             double max_snap_m,
                                             double radius) const {
  auto r = QueryReachability(from, to, radius);
  return r.reachable && r.from_snap_m <= max_snap_m &&
         r.to_snap_m <= max_snap_m;
}

Eigen::Vector3d NavigableSpatialDefinition::GetClosestPoint(
    const Eigen::Vector3d& from) const {
  auto ctx = GetGeometryContext();
  if (!ctx) return from;
  Eigen::Vector2d nearest = ctx->GetNearestPoint(from.head<2>());
  return Eigen::Vector3d(nearest.x(), nearest.y(), from.z());
}

double NavigableSpatialDefinition::GetDistance(
    const Eigen::Vector3d& from) const {
  auto ctx = GetGeometryContext();
  if (!ctx) return 0.0;
  return ctx->GetDistance(from.head<2>());
}

Eigen::Vector3d NavigableSpatialDefinition::ProjectOnto(
    const Eigen::Vector3d& point) const {
  auto ctx = GetGeometryContext();
  if (!ctx) return point;
  return ctx->ProjectOnto(point);
}

double NavigableSpatialDefinition::GetGeodesicDistance(const Eigen::Vector2d& from, const Eigen::Vector2d& to,
                                                       const std::vector<int64_t>& open) const {
  auto ctx = GetGeometryContext();
  if (!ctx) return (from - to).norm();
  return ctx->GetGeodesicDistance(from, to, open);
}

std::vector<ObstacleFreeSpace::RoutePiece> NavigableSpatialDefinition::GeodesicRoutes(
    const Eigen::Vector2d& from, const Eigen::Vector2d& to, double slack, const std::vector<int64_t>& open) const {
  auto ctx = GetGeometryContext();
  if (!ctx) return {};
  return ctx->GeodesicRoutes(from, to, slack, open);
}

double NavigableSpatialDefinition::GetGeodesicDistance(
    const Eigen::Vector3d& from, const Eigen::Vector3d& to) const {
  auto ctx = GetGeometryContext();
  if (!ctx) return (from - to).norm();
  return ctx->GetGeodesicDistance(from.head<2>(), to.head<2>());
}

DiscretePath NavigableSpatialDefinition::GetDiscretePath(
    const Eigen::Vector2d& start, const Eigen::Vector2d& goal) const {
  auto ctx = GetGeometryContext();
  if (!ctx) {
    DiscretePath path;
    path.AddPoint(start);
    path.AddPoint(goal);
    return path;
  }
  return ctx->GetDiscretePath(start, goal);
}

std::vector<ObstacleFreeSpace::RoutePiece>
NavigableSpatialDefinition::GeodesicRoutes(const Eigen::Vector2d& from,
                                           const Eigen::Vector2d& to, double slack) const {
  auto ctx = GetGeometryContext();
  if (!ctx) return {};
  return ctx->GeodesicRoutes(from, to, slack);
}

std::vector<ObstacleFreeSpace::RoutePiece> NavigableSpatialDefinition::ContactRoutes(
    const Eigen::Vector2d& from, const std::vector<const std::vector<Eigen::Vector2d>*>& rings, double slack) const {
  auto ctx = GetGeometryContext();
  if (!ctx) return {};
  return ctx->ContactRoutes(from, rings, slack);
}

std::vector<ObstacleFreeSpace::RoutePiece>
NavigableSpatialDefinition::GeodesicRoutesUnder(
    const Eigen::Vector2d& from, const Eigen::Vector2d& to,
    const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt, double slack) const {
  auto ctx = GetGeometryContext();
  if (!ctx) return {};
  return ctx->GeodesicRoutesUnder(from, to, exempt, ring_exempt, slack);
}

ObstacleFreeSpace::FeasibleCone NavigableSpatialDefinition::TangentConeAt(
    const Eigen::Vector2d& x, const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt, double tol) const {
  auto ctx = GetGeometryContext();
  if (!ctx) {
    ObstacleFreeSpace::FeasibleCone c;
    c.arcs = {{0.0, 6.283185307179586}};
    return c;
  }
  return ctx->TangentConeAt(x, exempt, ring_exempt, tol);
}

void NavigableSpatialDefinition::PreBuildIndices() const {
  if (walkable_) {
    walkable_->EnsureEdgeIndex();
  }
}

}  // namespace spatialblend
