// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/affordance_pose_sd.cc
#include "spatialblend/solver/affordance_pose_sd.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "spatialblend/model/element/affordance_pose.h"
#include "spatialblend/model/element/geometry_role.h"
#include "spatialblend/model/element/spatial_definition.h"
#include "spatialblend/geometry/radial_surface_chart.h"

namespace spatialblend {

namespace {

// Even-odd point-in-ring test in the plane's (u, v).
bool InRing(const std::vector<Eigen::Vector2d>& ring, const Eigen::Vector2d& q) {
  bool in = false;
  for (size_t i = 0, j = ring.size() - 1; i < ring.size(); j = i++) {
    const Eigen::Vector2d& a = ring[i];
    const Eigen::Vector2d& b = ring[j];
    if ((a.y() > q.y()) != (b.y() > q.y()) &&
        q.x() < (b.x() - a.x()) * (q.y() - a.y()) / (b.y() - a.y()) + a.x())
      in = !in;
  }
  return in;
}

Eigen::Vector3d Apply(const Eigen::Matrix4d& t, const Eigen::Vector3d& p) {
  return (t * Eigen::Vector4d(p.x(), p.y(), p.z(), 1.0)).head<3>();
}

// The frame a spatial definition's boundary polygon is stored in.
Eigen::Matrix4d BoundaryToWorld(const ElementTree& tree, const Element& e) {
  const SpatialDefinition* def = e.GetDeclaredSpatialDefinition();
  return def->RequiresElementWorldTransform() ? tree.GetWorldTransform(e.GetId()) : def->GetTransform();
}

}  // namespace

double PoseSupportDistance(const PoseSupport& s, const Eigen::Vector3d& x, Eigen::Vector3d* grad,
                           Eigen::Matrix3d* hess, int* feature, Eigen::Vector3d* nearest) {
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const auto to_point = [&](const Eigen::Vector3d& c, int feat, const Eigen::Vector3d* edge_dir) {
    const Eigen::Vector3d r = x - c;
    const double d = r.norm();
    if (nearest) *nearest = c;
    if (feature) *feature = feat;
    if (d > 0.0) {
      const Eigen::Vector3d g = r / d;
      if (grad) *grad = g;
      if (hess) {
        Eigen::Matrix3d h = I - g * g.transpose();
        if (edge_dir) h -= (*edge_dir) * edge_dir->transpose();
        *hess = h / d;
      }
    } else {
      if (grad) *grad = Eigen::Vector3d::Zero();
      if (hess) *hess = Eigen::Matrix3d::Zero();
    }
    return d;
  };
  if (s.kind == PoseSupport::Kind::kPoint) return to_point(s.point, 0, nullptr);
  const double h = (x - s.origin).dot(s.normal);
  const Eigen::Vector3d q = x - h * s.normal;
  const Eigen::Vector2d q2((q - s.origin).dot(s.u), (q - s.origin).dot(s.v));
  if (InRing(s.ring, q2)) {
    if (nearest) *nearest = q;
    if (feature) *feature = 0;
    if (grad) *grad = h >= 0.0 ? s.normal : Eigen::Vector3d(-s.normal);
    if (hess) *hess = Eigen::Matrix3d::Zero();
    return std::fabs(h);
  }
  // Outside the face: the nearest point of the boundary (an edge's interior or a vertex).
  double best = std::numeric_limits<double>::infinity();
  Eigen::Vector3d best_c = s.origin;
  int best_feat = 0;
  Eigen::Vector3d best_dir = Eigen::Vector3d::Zero();
  bool best_edge = false;
  const size_t n = s.ring.size();
  const auto lift = [&](const Eigen::Vector2d& p) { return Eigen::Vector3d(s.origin + p.x() * s.u + p.y() * s.v); };
  for (size_t k = 0; k < n; ++k) {
    const Eigen::Vector3d a = lift(s.ring[k]), b = lift(s.ring[(k + 1) % n]);
    const Eigen::Vector3d ab = b - a;
    const double len2 = ab.squaredNorm();
    double t = len2 > 0.0 ? (x - a).dot(ab) / len2 : 0.0;
    int feat;
    bool edge = false;
    if (t <= 0.0) {
      t = 0.0;
      feat = 1000 + static_cast<int>(k);
    } else if (t >= 1.0) {
      t = 1.0;
      feat = 1000 + static_cast<int>((k + 1) % n);
    } else {
      feat = 1 + static_cast<int>(k);
      edge = true;
    }
    const Eigen::Vector3d c = a + t * ab;
    const double d = (x - c).norm();
    if (d < best) {
      best = d;
      best_c = c;
      best_feat = feat;
      best_edge = edge;
      best_dir = len2 > 0.0 ? Eigen::Vector3d(ab / std::sqrt(len2)) : Eigen::Vector3d::Zero();
    }
  }
  return to_point(best_c, best_feat, best_edge ? &best_dir : nullptr);
}

bool PoseSD::Contains(const Eigen::Vector3d& world) const {
  const Eigen::Vector3d l = Apply(world_to_object, world);
  return (l.array() >= lo.array()).all() && (l.array() <= hi.array()).all();
}

Eigen::Vector3d PoseSD::Project(const Eigen::Vector3d& world) const {
  // V is a box in the object's (rigid) frame: the Euclidean projection is the clamp of the local coordinates.
  const Eigen::Vector3d l = Apply(world_to_object, world);
  if ((l.array() >= lo.array()).all() && (l.array() <= hi.array()).all()) return world;
  return Apply(object_to_world, l.cwiseMax(lo).cwiseMin(hi));
}

Eigen::Vector3d PoseSD::ProjectDirection(const Eigen::Vector3d& world, const Eigen::Vector3d& v) const {
  const Eigen::Vector3d l = Apply(world_to_object, world);
  const Eigen::Matrix3d r = object_to_world.block<3, 3>(0, 0);
  Eigen::Vector3d vl = r.transpose() * v;
  // A face holds the point when its coordinate is the bound to the representable resolution of the coordinate.
  const double res = 4.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, l.cwiseAbs().maxCoeff());
  for (int k = 0; k < 3; ++k) {
    if (l[k] <= lo[k] + res && vl[k] < 0.0) vl[k] = 0.0;
    if (l[k] >= hi[k] - res && vl[k] > 0.0) vl[k] = 0.0;
  }
  return r * vl;
}

PoseRegionScope::Cone PoseRegionScope::TangentConeAt(const Vec& x, double tol) const {
  const Eigen::Vector3d l = Apply(v_.world_to_object, x);
  const Eigen::Matrix3d r = v_.object_to_world.block<3, 3>(0, 0);
  Cone cone;
  for (int k = 0; k < 3; ++k) {
    if (l[k] <= v_.lo[k] + tol) cone.facets.push_back({Vec(-r.col(k)), 0.0});
    if (l[k] >= v_.hi[k] - tol) cone.facets.push_back({Vec(r.col(k)), 0.0});
  }
  return cone;
}

PoseRegionScope::Reach PoseRegionScope::Reachability(const Vec& from, const Vec& to) const {
  // Both ends are brought into V (the snap legs), and the straight segment joins them: V is convex.
  const Vec a = v_.Project(from), b = v_.Project(to);
  Reach r;
  r.reachable = true;
  r.from_snap_m = (from - a).norm();
  r.to_snap_m = (to - b).norm();
  r.length = r.from_snap_m + (b - a).norm() + r.to_snap_m;
  // The course: the snap legs (as the walkable's course has them, obstacle_free_space.cc FindPath) and the segment.
  if (r.from_snap_m > 1e-12) r.path.push_back(from);
  r.path.push_back(a);
  r.path.push_back(b);
  if (r.to_snap_m > 1e-12) r.path.push_back(to);
  return r;
}

const PoseInstance* PoseSD::Find(const std::string& tag) const {
  for (const auto& p : poses)
    if (p.tag == tag) return &p;
  return nullptr;
}

bool HasAffordancePoses(const ElementTree& tree, element_id_t object, const std::string& anchor) {
  const Element* obj = tree.GetElement(object);
  if (!obj) return false;
  for (element_id_t c : obj->GetChildIds()) {
    const Element* child = tree.GetElement(c);
    if (child && child->GetAffordancePose() && child->GetAffordancePose()->subject_anchor == anchor) return true;
  }
  return false;
}

absl::StatusOr<PoseSD> InstantiatePoseSD(const ElementTree& tree, element_id_t object, const std::string& anchor,
                                         const Embodiment* embodiment, const Eigen::Vector3d& root_world,
                                         double root_yaw) {
  const Element* obj = tree.GetElement(object);
  if (!obj) return absl::NotFoundError("object not found");
  const SpatialDefinition* def = obj->GetDeclaredSpatialDefinition();
  if (!def || !def->GetBoundary()) return absl::FailedPreconditionError("object has no spatial definition");
  PoseSD sd;
  sd.object = object;
  sd.anchor = anchor;
  sd.object_to_world = tree.GetWorldTransform(object);
  sd.world_to_object = sd.object_to_world.inverse();

  Eigen::Vector2d xy_lo = Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector2d xy_hi = -xy_lo;
  const auto add_footprint = [&](const Element& e) {
    const SpatialDefinition* d = e.GetDeclaredSpatialDefinition();
    if (!d || !d->GetBoundary()) return;
    const Eigen::Matrix4d to_obj = sd.world_to_object * BoundaryToWorld(tree, e);
    for (const auto& p : d->GetBoundary()->GetPoints()) {
      const Eigen::Vector3d l = Apply(to_obj, Eigen::Vector3d(p.x(), p.y(), 0.0));
      xy_lo = xy_lo.cwiseMin(l.head<2>());
      xy_hi = xy_hi.cwiseMax(l.head<2>());
    }
  };
  add_footprint(*obj);
  double z_hi = -std::numeric_limits<double>::infinity();
  const auto note_z = [&](const Eigen::Vector3d& world) { z_hi = std::max(z_hi, Apply(sd.world_to_object, world).z()); };

  for (element_id_t c : obj->GetChildIds()) {
    const Element* child = tree.GetElement(c);
    if (!child || !child->GetAffordancePose() || child->GetAffordancePose()->subject_anchor != anchor) continue;
    if (!child->HasGeometryRole(GeometryRole::kInteractionRegion))
      return absl::FailedPreconditionError("affordance pose '" + child->GetTag() +
                                           "' is not on an interaction-role child");
    const AffordancePoseTemplate& t = *child->GetAffordancePose();
    PoseInstance inst;
    inst.element = c;
    inst.tag = child->GetTag();
    if (t.support == AffordancePoseTemplate::Support::kActorRoot) {
      const EmbodimentAnchor* a = embodiment ? embodiment->FindAnchor(anchor) : nullptr;
      if (!a || !a->has_standing)
        return absl::FailedPreconditionError("affordance pose '" + inst.tag + "': the embodiment registers no standing height for '" +
                                             anchor + "'");
      inst.support.kind = PoseSupport::Kind::kPoint;
      inst.support.point = root_world + Eigen::Vector3d(0.0, 0.0, a->standing_height_m);
      Eigen::Matrix4d f = Eigen::Matrix4d::Identity();
      f.block<3, 3>(0, 0) = Eigen::AngleAxisd(root_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
      f.block<3, 1>(0, 3) = inst.support.point;
      inst.support.part_to_world = f;
      note_z(inst.support.point);
    } else {
      const SubRegion* sub = def->GetSubRegion(t.part);
      if (!sub || !sub->boundary || sub->boundary->GetPoints().size() < 3)
        return absl::NotFoundError("affordance pose '" + inst.tag + "': part '" + t.part + "' not found");
      const Eigen::Matrix4d part_to_world = BoundaryToWorld(tree, *obj) * sub->local_transform;
      PoseSupport& s = inst.support;
      s.kind = PoseSupport::Kind::kPlanarPolygon;
      s.part_to_world = part_to_world;
      s.boundary = sub->boundary;
      s.chart = sub->chart;
      s.part = t.part;
      s.origin = part_to_world.block<3, 1>(0, 3);
      const Eigen::Vector3d ax = part_to_world.block<3, 1>(0, 0), ay = part_to_world.block<3, 1>(0, 1);
      s.normal = ax.cross(ay).normalized();
      s.u = ax.normalized();
      s.v = s.normal.cross(s.u);
      for (const auto& p : sub->boundary->GetPoints()) {
        const Eigen::Vector3d w = Apply(part_to_world, Eigen::Vector3d(p.x(), p.y(), 0.0));
        s.ring.emplace_back((w - s.origin).dot(s.u), (w - s.origin).dot(s.v));
        note_z(w);
      }
    }
    add_footprint(*child);
    sd.poses.push_back(std::move(inst));
  }
  if (sd.poses.empty()) return absl::NotFoundError("the object has no affordance pose of '" + anchor + "'");
  sd.lo = Eigen::Vector3d(xy_lo.x(), xy_lo.y(), 0.0);
  sd.hi = Eigen::Vector3d(xy_hi.x(), xy_hi.y(), z_hi);
  if (!(sd.hi.array() > sd.lo.array()).all()) return absl::FailedPreconditionError("degenerate SD region volume");
  sd.scale_m = (sd.hi - sd.lo).norm();
  return sd;
}

absl::StatusOr<PoseContact> EncodePoseContact(const PoseSupport& s, const Eigen::Vector3d& anchor) {
  PoseContact c;
  Eigen::Vector3d nearest;
  c.distance = PoseSupportDistance(s, anchor, nullptr, nullptr, nullptr, &nearest);
  const Eigen::Matrix3d r = s.part_to_world.block<3, 3>(0, 0);
  const Eigen::Vector3d off = anchor - nearest;
  // The direction in the support frame (rigid part frames: the rotation's inverse is its transpose). At d = 0 it is the
  // support's normal (up for a seat and for the standing point).
  c.dir = off.norm() > 0.0 ? Eigen::Vector3d(r.transpose() * off / off.norm()) : Eigen::Vector3d::UnitZ();
  if (s.kind == PoseSupport::Kind::kPlanarPolygon) {
    if (!s.boundary) return absl::FailedPreconditionError("part without a boundary");
    const Eigen::Vector4d l = s.part_to_world.inverse() * Eigen::Vector4d(nearest.x(), nearest.y(), nearest.z(), 1.0);
    c.on_part = true;
    c.local = l.head<2>();
    auto chart = RadialSurfaceChart::Encode(c.local, *s.boundary, Eigen::Vector2d(0, 1));
    if (!chart.ok()) return chart.status();
    c.chart = *chart;
  }
  return c;
}

absl::StatusOr<MappedPoseContact> MapPoseContact(const PoseSupport& rcv, const PoseContact& src,
                                                 const Polygon2D* src_boundary, SurfaceChartKind src_chart) {
  MappedPoseContact m;
  const Eigen::Matrix3d r = rcv.part_to_world.block<3, 3>(0, 0);
  const Eigen::Vector3d dir = r * src.dir;
  if (!(dir.norm() > 0.0)) return absl::InvalidArgumentError("degenerate pose direction");
  m.dir = dir.normalized();
  if (rcv.kind == PoseSupport::Kind::kPoint) {
    m.point = rcv.point;
    return m;
  }
  if (!src.on_part || !rcv.boundary) return absl::FailedPreconditionError("the pose's contact has no part chart");
  if (src_chart != rcv.chart) return absl::FailedPreconditionError("matched parts declare different surface charts");
  Eigen::Vector3d local;
  if (IsFrameChart(rcv.chart)) {
    if (!src_boundary) return absl::FailedPreconditionError("frame chart needs the source part boundary");
    auto mapped = PlanarFrameChart::Map(src.local, 0.0, *src_boundary, *rcv.boundary, rcv.chart);
    if (!mapped.ok()) return mapped.status();
    local = *mapped;
  } else {
    auto d2 = RadialSurfaceChart::Decode(src.chart, *rcv.boundary, Eigen::Vector2d(0, 1));
    if (!d2.ok()) return d2.status();
    local = Eigen::Vector3d(d2->x(), d2->y(), 0.0);
  }
  m.point = Apply(rcv.part_to_world, local);
  return m;
}

}  // namespace spatialblend
