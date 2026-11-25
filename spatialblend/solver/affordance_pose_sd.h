// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/affordance_pose_sd.h
//
// The INNER level of the relation solve: room -> object spatial definition (SD) -> affordance poses.
// The object's affordance poses (element/affordance_pose.h, carried by its interaction-role children) are instantiated
// for ONE embodiment on ONE side, and a body anchor (the pelvis for a chair, the hand for a board) is related to them
// exactly as Spacing relates the root to the room's objects:
//
//   * support    each pose's support is the NEAREST POINT of its instance to the anchor (for `seated` the nearest point
//                of the seat surface; for `stand` the standing anchor point over the actor's root), so the distance is
//                Spacing's: the metric distance to the referent's nearest point;
//   * scale      R = 1 - d / D, D the diameter of the SD's region volume V (below), as Spacing normalizes by the
//                diameter of its scope. One D per SD: no per-pose span, no extra height coordinate;
//   * scope      the poses are in scope exactly while the anchor is inside V (proximal, SP-D3-01 one level down).
//
// The region volume V of the SD is the box, in the object's frame, over the object's own footprint and its pose
// children's footprints, from the object's floor (its frame's z = 0) up to the highest pose instance. It is the free
// space F of the inner solve. Nothing here is a constant of the core: every height comes from the scene (parts) or from
// the embodiment's registration (the standing height).
#ifndef SPATIALBLEND_SPACE_AFFORDANCE_POSE_SD_H_
#define SPATIALBLEND_SPACE_AFFORDANCE_POSE_SD_H_

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "spatialblend/base/common/types.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/geometry/planar_frame_chart.h"
#include "spatialblend/geometry/polygon_coordinate_mapper.h"
#include "spatialblend/model/embodiment/embodiment.h"
#include "spatialblend/solver/relation_scope.h"

namespace spatialblend {

// The support set of one pose instance, world frame.
struct PoseSupport {
  enum class Kind { kPoint, kPlanarPolygon };
  Kind kind = Kind::kPoint;
  Eigen::Vector3d point = Eigen::Vector3d::Zero();  // kPoint
  // kPlanarPolygon: the plane (origin, unit normal, in-plane orthonormal basis u, v) and the ring in (u, v).
  Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d u = Eigen::Vector3d::UnitX(), v = Eigen::Vector3d::UnitY();
  std::vector<Eigen::Vector2d> ring;
  // The part frame -> world (the frame the template is stated in): the part's own frame for kPart, the actor root's
  // position and heading (yaw) for kActorRoot.
  Eigen::Matrix4d part_to_world = Eigen::Matrix4d::Identity();
  // kPlanarPolygon: the part's boundary in its own frame and how a point of it is read on another space's part.
  std::shared_ptr<Polygon2D> boundary;
  SurfaceChartKind chart = SurfaceChartKind::kRadial;
  std::string part;
};

// The distance from x to the support set and its exact derivatives (the nearest point moves with x; the gradient is
// exact by the envelope theorem, the Hessian is the curvature of the distance to the nearest FEATURE: 0 over the
// polygon's face, (I - g g^T - t t^T) / d to an edge of direction t, (I - g g^T) / d to a vertex or a point).
// `feature`: 0 the face (or the point), 1 + k edge k, 1000 + k vertex k: the branch of the distance (FOLD attribution).
// At d = 0 the gradient is the face normal (polygon) or 0 (point).
double PoseSupportDistance(const PoseSupport& s, const Eigen::Vector3d& x, Eigen::Vector3d* grad = nullptr,
                           Eigen::Matrix3d* hess = nullptr, int* feature = nullptr,
                           Eigen::Vector3d* nearest = nullptr);

struct PoseInstance {
  element_id_t element = kInvalidElementId;  // the interaction child carrying the pose
  std::string tag;
  PoseSupport support;
};

struct PoseSD {
  element_id_t object = kInvalidElementId;
  std::string anchor;
  std::vector<PoseInstance> poses;
  // V: the object's frame (its world transform) and the box in it.
  Eigen::Matrix4d object_to_world = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d world_to_object = Eigen::Matrix4d::Identity();
  Eigen::Vector3d lo = Eigen::Vector3d::Zero(), hi = Eigen::Vector3d::Zero();
  double scale_m = 0.0;  // D, the diameter of V

  bool Contains(const Eigen::Vector3d& world) const;
  Eigen::Vector3d Project(const Eigen::Vector3d& world) const;
  // The direction `v` projected onto the tangent cone of V at `world` (the faces V's point lies on block motion out).
  Eigen::Vector3d ProjectDirection(const Eigen::Vector3d& world, const Eigen::Vector3d& v) const;
  const PoseInstance* Find(const std::string& tag) const;
};

// True when `object` has an interaction child carrying an affordance pose of `anchor`.
bool HasAffordancePoses(const ElementTree& tree, element_id_t object, const std::string& anchor);

// Instantiates the affordance poses of `anchor` in `object`'s SD for `embodiment`, with the actor's root at
// `root_world`. Errors: no pose of the anchor; a pose child that is not interaction-role; a named part that does not
// exist or is degenerate; a kActorRoot pose for an embodiment that registered no standing height for the anchor (no
// default exists); a degenerate V.
// `root_yaw`: the actor root's heading, the frame of the kActorRoot poses.
absl::StatusOr<PoseSD> InstantiatePoseSD(const ElementTree& tree, element_id_t object, const std::string& anchor,
                                         const Embodiment* embodiment, const Eigen::Vector3d& root_world,
                                         double root_yaw = 0.0);

// THE CONTACT CHART of a pose (one level down: the radial part of a relation is the ratio
// pursuit, the tangential part is the referent chart's mapping of the source pose, never pursuit history).
// Source: the anchor's nearest point of the pose's support, read in the support's own frame (a part's chart
// coordinates; the root frame for a point), and the unit direction from it to the anchor in that frame.
struct PoseContact {
  bool on_part = false;  // kPlanarPolygon: `local` / `chart` are the nearest point in the part's chart
  Eigen::Vector2d local = Eigen::Vector2d::Zero();
  NormalizedPolygonPosition chart;
  Eigen::Vector3d dir = Eigen::Vector3d::UnitZ();  // anchor - nearest, unit, in the support frame (the normal at d = 0)
  double distance = 0.0;
};
absl::StatusOr<PoseContact> EncodePoseContact(const PoseSupport& s, const Eigen::Vector3d& anchor);
// Receiver: the source's contact read on THIS side's instance of the pose (the part chart, or the point itself), and
// the source's direction carried into this support's frame. `src_boundary` / `src_chart`: the source part's boundary
// and chart kind (a frame chart needs the source boundary; the two sides must declare the same kind).
struct MappedPoseContact {
  Eigen::Vector3d point = Eigen::Vector3d::Zero();  // c: the mapped contact, world
  Eigen::Vector3d dir = Eigen::Vector3d::UnitZ();   // u: the mapped direction, world, unit
};
absl::StatusOr<MappedPoseContact> MapPoseContact(const PoseSupport& rcv, const PoseContact& src,
                                                 const Polygon2D* src_boundary, SurfaceChartKind src_chart);

// V as a relation scope (relation_scope.h). V is a box, convex: a segment is in V exactly when its ends are, the
// straight segment is the geodesic, and the nearest pose is the clamp of the object-frame coordinates.
class PoseRegionScope final : public RelationScope<3> {
 public:
  explicit PoseRegionScope(const PoseSD& v) : v_(v) {}
  bool IsFree(const Vec& x) const override { return v_.Contains(x); }
  bool SegmentFree(const Vec& a, const Vec& b) const override { return v_.Contains(a) && v_.Contains(b); }
  std::optional<Vec> NearestFree(const Vec& x, NearestFreeStatus* status = nullptr) const override {
    if (status) *status = v_.Contains(x) ? NearestFreeStatus::kAlreadyFree : NearestFreeStatus::kProjected;
    return v_.Project(x);
  }
  std::optional<Vec> NearestFreeUnder(const Vec& x) const override { return v_.Project(x); }
  Vec ProjectDirection(const Vec& x, const Vec& v) const override { return v_.ProjectDirection(x, v); }
  Cone TangentConeAt(const Vec& x, double tol) const override;
  Reach Reachability(const Vec& from, const Vec& to) const override;
  std::vector<Route> GeodesicRoutesUnder(const Vec&, const Vec&, double) const override { return {}; }
  std::vector<Route> GeodesicRoutes(const Vec&, const Vec&, double) const override { return {}; }
  std::vector<Route> ContactRoutes(const Vec&, const std::vector<const std::vector<Vec>*>&, double) const override {
    return {};
  }
  double RouteSegmentBound() const override { return 0.0; }
  double ObstacleClearance() const override { return 0.0; }
  double Diameter() const override { return v_.scale_m; }
  bool Bounded() const override { return true; }

 private:
  const PoseSD& v_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_AFFORDANCE_POSE_SD_H_
