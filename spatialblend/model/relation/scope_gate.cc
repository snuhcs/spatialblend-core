// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/scope_gate.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/geometry/obstacle_free_space.h"

namespace spatialblend {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// World-XY footprint of an element's spatial definition, honoring the frame
// contract of SpatialDefinition::RequiresElementWorldTransform (same rule the
// grounding rules use).
std::vector<Eigen::Vector2d> WorldFootprint(const ElementTree& tree,
                                            element_id_t id,
                                            const SpatialDefinition& def) {
  std::vector<Eigen::Vector2d> out;
  auto boundary = def.GetBoundary();
  if (!boundary) return out;
  const Eigen::Matrix4d to_world = def.RequiresElementWorldTransform()
                                       ? tree.GetWorldTransform(id)
                                       : def.GetTransform();
  out.reserve(boundary->GetPoints().size());
  for (const auto& p : boundary->GetPoints()) {
    out.push_back((to_world * Eigen::Vector4d(p.x(), p.y(), 0.0, 1.0)).head<2>());
  }
  return out;
}

// Top of an element in world Z: a polytope's or surface's highest vertex;
// `footprint_top` above its placement for a footprint-only definition.
double WorldTop(const ElementTree& tree, element_id_t id,
                const SpatialDefinition& def, double footprint_top) {
  if (const auto* poly = dynamic_cast<const PolytopeSpatialDefinition*>(&def)) {
    const Eigen::Matrix4d w = tree.GetWorldTransform(id);
    double top = -kInf;
    for (const auto& v : poly->GetVertices()) {
      top = std::max(top, (w * Eigen::Vector4d(v.x(), v.y(), v.z(), 1.0)).z());
    }
    if (std::isfinite(top)) return top;
  }
  if (const auto* surf = dynamic_cast<const SurfaceSpatialDefinition*>(&def)) {
    double top = -kInf;
    if (auto poly3 = surf->GetSurface()) {
      for (const auto& v : poly3->GetVertices()) top = std::max(top, v.z());
    }
    if (std::isfinite(top)) return top;  // surface vertices are world-frame
  }
  return tree.GetWorldTransform(id)(2, 3) + footprint_top;
}

bool PointInRing(const std::vector<Eigen::Vector2d>& ring,
                 const Eigen::Vector2d& p) {
  bool inside = false;
  const size_t n = ring.size();
  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    const auto& a = ring[i];
    const auto& b = ring[j];
    if ((a.y() > p.y()) != (b.y() > p.y())) {
      const double x = a.x() + (p.y() - a.y()) * (b.x() - a.x()) / (b.y() - a.y());
      if (p.x() < x) inside = !inside;
    }
  }
  return inside;
}

double Cross(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
  return a.x() * b.y() - a.y() * b.x();
}

// Parameters t in [0,1] along p->q at which it meets segment a->b (touching
// included; a collinear overlap reports its two ends).
void SegmentHits(const Eigen::Vector2d& p, const Eigen::Vector2d& q,
                 const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                 std::vector<double>* ts) {
  constexpr double kEps = 1e-12;
  const Eigen::Vector2d r = q - p;
  const Eigen::Vector2d s = b - a;
  const double denom = Cross(r, s);
  const Eigen::Vector2d ap = a - p;
  if (std::abs(denom) < kEps) {
    if (std::abs(Cross(ap, r)) > kEps) return;  // parallel, disjoint
    const double rr = r.squaredNorm();
    if (rr < kEps) return;
    double t0 = ap.dot(r) / rr;
    double t1 = (b - p).dot(r) / rr;
    if (t0 > t1) std::swap(t0, t1);
    if (t1 < 0.0 || t0 > 1.0) return;
    ts->push_back(std::max(t0, 0.0));
    ts->push_back(std::min(t1, 1.0));
    return;
  }
  const double t = Cross(ap, s) / denom;
  const double u = Cross(ap, r) / denom;
  if (t >= -1e-9 && t <= 1.0 + 1e-9 && u >= -1e-9 && u <= 1.0 + 1e-9) {
    ts->push_back(std::clamp(t, 0.0, 1.0));
  }
}

// True when the sightline eye -> target crosses a ring edge (closed ring,
// last edge wraps) at a point where the occluder (height `top`) rises above
// the sightline.
bool Blocks(const std::vector<Eigen::Vector2d>& ring, double top,
            const Eigen::Vector3d& eye, const Eigen::Vector3d& target) {
  const Eigen::Vector2d p = eye.head<2>();
  const Eigen::Vector2d q = target.head<2>();
  const Eigen::Vector2d lo = p.cwiseMin(q), hi = p.cwiseMax(q);
  std::vector<double> ts;
  const size_t n = ring.size();
  for (size_t i = 0; i < n; ++i) {
    const Eigen::Vector2d& a = ring[i];
    const Eigen::Vector2d& b = ring[(i + 1) % n];
    // Cheap reject: edge bounding box disjoint from the sightline's.
    if (std::max(a.x(), b.x()) < lo.x() || std::min(a.x(), b.x()) > hi.x() ||
        std::max(a.y(), b.y()) < lo.y() || std::min(a.y(), b.y()) > hi.y())
      continue;
    SegmentHits(p, q, a, b, &ts);
  }
  for (double t : ts) {
    const double sight_z = eye.z() + t * (target.z() - eye.z());
    if (top > sight_z + 1e-9) return true;
  }
  return false;
}

}  // namespace

OcclusionScene::OcclusionScene(const ElementTree& tree,
                               std::unordered_map<element_id_t, double> body_tops)
    : tree_(tree), body_tops_(std::move(body_tops)) {
  for (element_id_t id : tree.GetAllIds()) {
    const Element* e = tree.GetElement(id);
    const SpatialDefinition* def = e ? e->GetDeclaredSpatialDefinition() : nullptr;
    if (!def) continue;
    if (e->GetGeometryRole() == GeometryRole::kObstacle) {
      Occluder o;
      o.owner = id;
      o.owner_ancestors = tree.GetAncestorIds(id);
      o.ring = WorldFootprint(tree, id, *def);
      o.top = WorldTop(tree, id, *def, kFootprintObstacleHeightM);
      if (o.ring.size() >= 2) occluders_.push_back(std::move(o));
    }
    if (const auto* nav = dynamic_cast<const NavigableSpatialDefinition*>(def)) {
      if (auto wall = nav->GetWall(); wall && wall->GetPoints().size() >= 2) {
        Occluder o;  // walls: stored in world frame, full height, no exemptions
        o.ring = wall->GetPoints();
        o.top = kInf;
        o.eye_exempt_capable = false;
        occluders_.push_back(std::move(o));
      }
    }
  }
}

bool OcclusionScene::IsReferentVisible(
    const Eigen::Vector3d& eye_in, element_id_t referent_id,
    const Eigen::Vector3d* contact_point_world) const {
  const Element* referent = tree_.GetElement(referent_id);
  if (!referent) return false;
  Eigen::Vector3d eye = eye_in;
  if (eye.z() < kMinSightlineHeightM) eye.z() = kDefaultSightlineHeightM;

  // Occluders relevant to this referent: not itself, not an ancestor (it
  // sits in/on them), not a descendant (its own parts), not enclosing the eye.
  const std::vector<element_id_t> ref_ancestors =
      tree_.GetAncestorIds(referent_id);
  std::vector<const Occluder*> active;
  active.reserve(occluders_.size());
  for (const auto& o : occluders_) {
    if (o.eye_exempt_capable) {
      if (o.owner == referent_id) continue;
      if (std::find(ref_ancestors.begin(), ref_ancestors.end(), o.owner) !=
          ref_ancestors.end())
        continue;
      if (std::find(o.owner_ancestors.begin(), o.owner_ancestors.end(),
                    referent_id) != o.owner_ancestors.end())
        continue;
      if (o.ring.size() >= 3 && PointInRing(o.ring, eye.head<2>())) continue;
    }
    active.push_back(&o);
  }
  if (active.empty()) return true;

  // ONE ray from the eye to the look-at contact
  // (SpatialBlend: "a ray-cast line-of-sight test from the subject anchor"). With no
  // contact the look-at point is the referent's own position at its top. There is no
  // search over the footprint for a visible sample: a partly hidden referent whose
  // contact is hidden is not visible.
  const SpatialDefinition* rdef = referent->GetDeclaredSpatialDefinition();
  const Eigen::Matrix4d rworld = tree_.GetWorldTransform(referent_id);
  // The contact is a point of the referent's footprint boundary; the ray ends at it
  // at the referent's top height (a floor-level end would be blocked by anything
  // between that stands taller than the sightline's floor end). With no declared
  // geometry the top is the body the caller gave the element (a person), else its
  // placement.
  const auto body = body_tops_.find(referent_id);
  const double top = rdef ? WorldTop(tree_, referent_id, *rdef, 0.0)
                          : rworld(2, 3) + (body != body_tops_.end() ? body->second : 0.0);
  const Eigen::Vector3d target =
      contact_point_world
          ? Eigen::Vector3d(contact_point_world->x(), contact_point_world->y(),
                            std::max(top, contact_point_world->z()))
          : Eigen::Vector3d(rworld(0, 3), rworld(1, 3), top);
  const Eigen::Vector2d s = target.head<2>();
  for (const Occluder* o : active) {
    // A finite obstacle enclosing the look-at point is what the referent rests on.
    if (o->eye_exempt_capable && o->ring.size() >= 3 &&
        std::isfinite(o->top) && PointInRing(o->ring, s))
      continue;
    if (Blocks(o->ring, o->top, eye, target)) return false;
  }
  return true;
}

bool IsReferentVisible(const ElementTree& tree, const Eigen::Vector3d& eye,
                       element_id_t referent_id,
                       const Eigen::Vector3d* contact_point_world) {
  return OcclusionScene(tree).IsReferentVisible(eye, referent_id,
                                                contact_point_world);
}

bool IsReferentReachable(const NavigableSpatialDefinition& navigable,
                         const Eigen::Vector3d& from,
                         const Eigen::Vector3d& to) {
  // D8: a geodesic path in the source free space F from the anchor to the
  // referent's support. Radius 0: point connectivity around the raw obstacle
  // footprints -- the same free space the source geodesic uses
  // (NavigableGeometryContext::GetGeodesicDistance), so a referent is in scope
  // exactly when its spacing R is finite, independent of the sender's body
  // radius. The anchor may be anywhere (a tracked anchor over an edge or inside
  // furniture is not a reason to lose scope); the approach point is the
  // geodesic-nearest contact, which lies ON F, so it is not snapped: a contact
  // flush with the free-space edge is reachable by definition, and one that
  // is not on F (behind a wall, sealed by an obstacle ring) is not. There is no
  // snap tolerance (the former 0.25 m is deleted); the free space's own boundary
  // hair is the only slack.
  const NavigableGeometryContext::Reachability r = navigable.QueryReachability(
      from.head<2>(), to.head<2>(), /*radius=*/0.0);
  return r.reachable && r.to_snap_m <= ObstacleFreeSpace::kBoundaryHairM;
}

ScopeGateVerdict FamilyScopeGateVerdict(
    RelationKernelType family, const ElementTree& tree,
    element_id_t referent_id, const Eigen::Vector3d& anchor_world_position,
    const Support& support, const NavigableSpatialDefinition* navigable,
    const OcclusionScene* occlusion) {
  switch (family) {
    case RelationKernelType::kAvatarFacing: {
      std::optional<OcclusionScene> local;
      if (!occlusion) occlusion = &local.emplace(tree);
      bool visible;
      if (!support.has_contact) {
        visible =
            occlusion->IsReferentVisible(anchor_world_position, referent_id);
      } else {
        // FacingGroundingRule fills contact_point_local only (the same point
        // AvatarFacingKernel::Encode bears toward), so lift it to world here.
        const Eigen::Vector3d contact_world =
            (support.world_transform *
             Eigen::Vector4d(support.contact_point_local.x(),
                             support.contact_point_local.y(), 0.0, 1.0))
                .head<3>();
        visible = occlusion->IsReferentVisible(anchor_world_position,
                                               referent_id, &contact_world);
      }
      return visible ? ScopeGateVerdict::kInScope
                     : ScopeGateVerdict::kNotVisible;
    }
    case RelationKernelType::kAvatarSpacing: {
      // D8. With no navigable there is no free space to judge a path in.
      if (!navigable) return ScopeGateVerdict::kInScope;
      // The anchor stands on or in the referent (its footprint contains it): the
      // geodesic to its support is 0 by containment, so a path trivially exists.
      // (This is a fact of the geometry, not a gate bypass: the contact then is the
      // nearest edge, reached from where the anchor already is.)
      if (support.has_interior) return ScopeGateVerdict::kInScope;
      // The source geodesic the kernel measured to the chosen contact IS the path in
      // the source F (SpacingGroundingRule: ClosestGeodesicOnRing). Infinite = sealed.
      if (support.geodesic_length_m >= 0.0)
        return std::isfinite(support.geodesic_length_m)
                   ? ScopeGateVerdict::kInScope
                   : ScopeGateVerdict::kUnreachable;
      // No geodesic was measured (a point referent, such as a person with no
      // footprint): a path from the anchor to its position must exist in F.
      const Eigen::Vector3d target = support.has_contact
                                         ? support.contact_point_world
                                         : support.GetPosition3D();
      return IsReferentReachable(*navigable, anchor_world_position, target)
                 ? ScopeGateVerdict::kInScope
                 : ScopeGateVerdict::kUnreachable;
    }
    case RelationKernelType::kAvatarAffordancePose: {
      // The scope is the parent object's region
      // (v1 affordance volume): the anchor lies inside the footprint of the
      // nearest ancestor that declares one. No distance, no near field.
      const Element* referent = tree.GetElement(referent_id);
      if (!referent) return ScopeGateVerdict::kInScope;
      for (element_id_t up = referent->GetParentId(); up != kInvalidElementId;) {
        const Element* parent = tree.GetElement(up);
        if (!parent) break;
        if (const SpatialDefinition* def = parent->GetDeclaredSpatialDefinition()) {
          const std::vector<Eigen::Vector2d> ring = WorldFootprint(tree, up, *def);
          // A parent whose footprint has no area (a vertical surface such as a
          // whiteboard projects to a line) declares no region on the floor: the next
          // ancestor's region is the one that holds the anchor.
          double twice_area = 0.0;
          for (size_t i = 0; i < ring.size(); ++i)
            twice_area += Cross(ring[i], ring[(i + 1) % ring.size()]);
          // The parent's whole SCOPE region (relation_kernel.h FootprintHolds): every component
          // (a child across a gap) and hole, no bridge or corridor.
          if (ring.size() >= 3 && std::abs(twice_area) > 0.0)
            return FootprintHolds(*parent, tree, anchor_world_position)
                       ? ScopeGateVerdict::kInScope
                       : ScopeGateVerdict::kOutsideParentRegion;
        }
        up = parent->GetParentId();
      }
      return ScopeGateVerdict::kInScope;  // no ancestor declares a region
    }
  }
  return ScopeGateVerdict::kInScope;
}

bool PassesFamilyScopeGate(RelationKernelType family, const ElementTree& tree,
                           element_id_t referent_id,
                           const Eigen::Vector3d& anchor_world_position,
                           const Support& support,
                           const NavigableSpatialDefinition* navigable,
                           const OcclusionScene* occlusion) {
  return FamilyScopeGateVerdict(family, tree, referent_id,
                                anchor_world_position, support, navigable,
                                occlusion) == ScopeGateVerdict::kInScope;
}

}  // namespace spatialblend
