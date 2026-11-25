// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/relation_solver.cc
#include "spatialblend/solver/relation_solver.h"
#include "spatialblend/solver/nd_trust_region.h"
#include "spatialblend/solver/nd_trust_region_loop.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <unordered_set>
#include <string>

#include <Eigen/Eigenvalues>

#include "absl/status/status.h"
#include "spatialblend/base/common/state_type.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/geometry/polygon_coordinate_mapper.h"
#include "spatialblend/geometry/path/bezier_course.h"
#include "spatialblend/geometry/planar_frame_chart.h"
#include "spatialblend/geometry/radial_surface_chart.h"
#include "spatialblend/model/relation/grounding_rule.h"
#include "spatialblend/model/relation/kernel_factory.h"
#include "spatialblend/model/relation/scope_resolver.h"
#include "spatialblend/model/relation/spacing_grounding_rule.h"
#include "spatialblend/solver/engagement.h"
#include "spatialblend/solver/relation_evaluator.h"
#include "spatialblend/solver/relation_scope.h"
#include "spatialblend/solver/scope_pursuit.h"

namespace spatialblend {

namespace {

bool HasInvalidWeight(const std::vector<double>& weights) {
  return std::any_of(weights.begin(), weights.end(), [](double weight) {
    return !std::isfinite(weight) || weight < 0.0;
  });
}

double WeightAt(const std::vector<double>& weights, size_t index) {
  return (!weights.empty() && index < weights.size()) ? weights[index] : 1.0;
}

// The source offset, a point in the support frame, read on the receiver
// polygon. Convex pairs use the radial chart with its ratio left unclamped, so
// a point outside the source polygon is the same fraction of the receiver ray
// and a scale is that ray's length. If either polygon is non-convex and the
// point lies in the source, the chart is MVC (v1: MVC when either polygon is
// non-convex). There is no test that the two polygons are the same shape.
absl::StatusOr<Eigen::Vector2d> MapThroughSupportChart(
    const Eigen::Vector2d& point, const Polygon2D& source,
    const Polygon2D& target) {
  const Eigen::Vector2d forward(0, 1);
  if (!source.IsConvex() || !target.IsConvex()) {
    if (auto mapped = PolygonCoordinateMapper::MapPosition(
            point, source, forward, target, forward);
        mapped.ok())
      return mapped;
  }
  auto chart = PolygonCoordinateMapper::EncodeRadial(point, source, forward);
  if (!chart.ok()) return chart.status();
  return PolygonCoordinateMapper::Decode(*chart, target, forward);
}

// The one correspondence between two supports (v1 PolygonCoordinateMapper): a
// point of the source polygon read on the receiver polygon. Both the mapped
// contact c_p and the approach pose a_p go through it. A receiver whose polygon
// is the source's, vertex for vertex, is the identity map.
absl::StatusOr<Eigen::Vector2d> MapSupportPoint(const Eigen::Vector2d& point,
                                                const Polygon2D& source,
                                                const Polygon2D& target) {
  const auto& sp = source.GetPoints();
  const auto& tp = target.GetPoints();
  bool same_shape = sp.size() == tp.size();
  for (size_t i = 0; same_shape && i < tp.size(); ++i)
    same_shape = (sp[i] - tp[i]).norm() < 1e-9;
  if (same_shape) return point;
  return MapThroughSupportChart(point, source, target);
}

// Decode the support/contact point from a payload onto the target referent.
// When the source payload has a contact point (e.g. "left edge of Board_A"),
// this maps that normalized polygon position onto the target referent's
// boundary ("left edge of Board_B").
//
// A missing counterpart is an explicit outcome, never a silent
// centre target. The only origin-valued support is a payload whose SOURCE
// support was itself the referent origin (has_contact == false: the source
// referent had no boundary to ground on), which decodes exactly. A source
// contact that the target cannot reproduce (no spatial definition, no
// boundary, a requested part the target lacks, or an undecodable chart) is an
// error the caller must report.
struct SupportDecodeInfo {
  // The receiver boundary is vertex-for-vertex the source's (an identical copy,
  // frame-wise): the source's own direction is already exact there.
  bool same_shape_as_source = false;
};

absl::StatusOr<Eigen::Vector2d> DecodeSupportPoint(
    const RelationPayload& payload, const Element& target_referent,
    const ElementTree& tree, SupportDecodeInfo* info = nullptr) {
  if (info) *info = SupportDecodeInfo();
  Eigen::Matrix4d world_t = tree.GetWorldTransform(target_referent.GetId());
  Eigen::Vector2d referent_pos(world_t(0, 3), world_t(1, 3));
  if (!payload.support.has_contact) return referent_pos;

  auto* spatial_def = target_referent.GetDeclaredSpatialDefinition();
  if (!spatial_def)
    return absl::FailedPreconditionError(
        "target referent has no support geometry");
  // Cross-space injective map: if the source support
  // was grounded on a named sub-region, decode onto the same-named sub-region
  // of the target and lift back through its local transform. A named part the
  // target lacks is reported, never re-targeted to the whole boundary.
  std::shared_ptr<Polygon2D> boundary;
  Eigen::Matrix4d sub_transform = Eigen::Matrix4d::Identity();
  bool used_sub_region = false;
  if (!payload.support.sub_region_name.empty()) {
    const SubRegion* sub =
        spatial_def->GetSubRegion(payload.support.sub_region_name);
    if (!sub || !sub->boundary)
      return absl::NotFoundError("target support sub-region not found");
    boundary = sub->boundary;
    sub_transform = sub->local_transform;
    used_sub_region = true;
  }
  if (!boundary) boundary = spatial_def->GetBoundary();
  if (!boundary)
    return absl::FailedPreconditionError(
        "target referent has no support boundary");
  Eigen::Vector2d forward(0, 1);
  // One chart (v1 PolygonCoordinateMapper): the source contact read on the receiver
  // polygon, radial, or MVC when either polygon is non-convex. The payload's chart is
  // inverted on the SOURCE polygon to the contact point, which the same mapping that
  // places the approach pose then carries onto the receiver. An identical copy is
  // the identity map, so its own contact is exact.
  const NormalizedPolygonPosition& chart = payload.support.contact_normalized;
  const auto& src_boundary = payload.support.boundary;
  const auto& rcv_pts = boundary->GetPoints();
  bool same_shape = src_boundary && src_boundary->GetPoints().size() == rcv_pts.size();
  for (size_t i = 0; same_shape && i < rcv_pts.size(); ++i)
    same_shape = (src_boundary->GetPoints()[i] - rcv_pts[i]).norm() < 1e-9;
  const bool exact_copy_contact = same_shape && !boundary->IsConvex() &&
                                  payload.support.has_contact &&
                                  !payload.support.has_interior;
  absl::StatusOr<Eigen::Vector2d> decoded;
  if (exact_copy_contact) {
    decoded = payload.support.contact_point_local;
  } else if (src_boundary) {
    absl::StatusOr<Eigen::Vector2d> contact =
        PolygonCoordinateMapper::Decode(chart, *src_boundary, forward);
    decoded = contact.ok() ? MapSupportPoint(*contact, *src_boundary, *boundary)
                           : contact;
  } else {
    // A legacy payload without the source polygon: its own chart on the receiver.
    decoded = PolygonCoordinateMapper::Decode(chart, *boundary, forward);
  }
  if (info) {
    info->same_shape_as_source = same_shape;
  }
  if (!decoded.ok()) return decoded.status();
  // The decode side of the frame composition:
  // GetBoundary()/sub-region boundaries are stored in whatever frame the
  // referent's SpatialDefinition (or, for a sub-region, that definition's OWN
  // frame) considers "local" — RequiresElementWorldTransform()==true
  // (PolytopeSpatialDefinition) means that frame is the owning ELEMENT's
  // local frame, with no internal mechanism to reflect the element's
  // placement in the tree, so the decoded point must be lifted through the
  // tree-derived world_t here, exactly mirroring
  // SpacingGroundingRule::ComputeSupport's encode-side pattern.
  // RequiresElementWorldTransform()==false (Navigable/Surface) means the
  // definition's own frame is already world-absolute, so no further lift is
  // needed.
  const bool lift_to_world = spatial_def->RequiresElementWorldTransform();
  // The frame the boundary polygon is stored in, lifted to world. A Surface keeps
  // its polygon in its OWN frame, GetTransform() (its queries are world-in /
  // world-out and the encode side maps the anchor through GetTransform().
  // inverse(), SpacingGroundingRule::ComputeSupport); the decode returned that
  // point unlifted, so a Surface referent's support decode landed in the
  // surface-local frame (near the world origin) and was hidden by a distance
  // guard in the interior blend, since removed. A Navigable that is not
  // element-local is world-absolute and stays unlifted.
  const bool is_surface =
      dynamic_cast<const SurfaceSpatialDefinition*>(spatial_def) != nullptr;
  const Eigen::Matrix4d lift_t =
      lift_to_world ? world_t
                    : (is_surface ? Eigen::Matrix4d(spatial_def->GetTransform())
                                  : Eigen::Matrix4d::Identity());
  const auto to_world = [&](const Eigen::Vector2d& p) -> Eigen::Vector2d {
    const Eigen::Vector4d local4(p.x(), p.y(), 0.0, 1.0);
    if (used_sub_region) {
      // sub_transform maps the sub-region's own local frame into the OWNING
      // SpatialDefinition's own frame (see SubRegion's doc comment) — not
      // necessarily world space yet.
      const Eigen::Vector4d lifted = lift_t * (sub_transform * local4);
      return Eigen::Vector2d(lifted.x(), lifted.y());
    }
    const Eigen::Vector4d lifted = lift_t * local4;
    return Eigen::Vector2d(lifted.x(), lifted.y());
  };
  return to_world(*decoded);
}

}  // namespace

const NavigableSpatialDefinition* CorrespondingScope(
    const RelationPayload& payload, const ElementTree& tree,
    element_id_t* scope_element) {
  auto navigable_of = [&](element_id_t id) -> const NavigableSpatialDefinition* {
    const Element* el = tree.GetElement(id);
    return el ? dynamic_cast<const NavigableSpatialDefinition*>(
                    el->GetSpatialDefinition())
              : nullptr;
  };
  auto found = [&](element_id_t id) {
    if (scope_element) *scope_element = id;
    return navigable_of(id);
  };
  if (payload.resolved_scope_element_id != kInvalidElementId &&
      navigable_of(payload.resolved_scope_element_id))
    return found(payload.resolved_scope_element_id);
  // A receiver that already resolved the scope applied its own identity mode
  // (linker-only never falls back to shared ids); only a payload that no
  // receiver resolved (direct solver API use) is matched by shared id here.
  if (!payload.scope_resolved_by_receiver &&
      payload.scope_shared_id != kInvalidSharedElementId) {
    if (const Element* el = tree.GetElementBySharedId(payload.scope_shared_id);
        el && navigable_of(el->GetId()))
      return found(el->GetId());
  }
  // The structural correspondent hangs off the referent as the receiver
  // resolved it (a linker-only receiver never admits a payload without a
  // linker-resolved referent, so the shared id here is its own resolution).
  const Element* referent =
      payload.resolved_local_element_id != kInvalidElementId
          ? tree.GetElement(payload.resolved_local_element_id)
          : tree.GetElementBySharedId(payload.referent_shared_id);
  if (!referent) return nullptr;
  for (element_id_t e = referent->GetParentId(); e != kInvalidElementId;) {
    const Element* el = tree.GetElement(e);
    if (!el) break;
    if (navigable_of(e)) return found(e);
    e = el->GetParentId();
  }
  return nullptr;
}

// The distance a position term measures. A container's border relation measures its
// depth below the nearest border point. Every other relation measures the geodesic
// length of the course to the mapped contact, the length the one-relation step walks
// (v1: the geodesic distance to the target). It does not stop at 0 inside the
// referent, where R = 1 and support_distance_m is 0: the objective would drop by the
// whole distance to the contact across the referent's boundary, a cliff the pursuit
// cannot descend.
double PursuitDistance(const GroundedRelation& gr, bool container_self) {
  return container_self ? gr.support_distance_m : gr.contact_distance_m;
}

// What a position term reports against: in the pursuit the distance above; when the
// solve places or snaps, the term reports the SOURCE relation (d* = the source's
// support distance on the receiver's scope), so it is read as the relation is, in
// support distance (0 inside the referent, R = 1).
double TermDistance(const GroundedRelation& gr, bool container_self, bool pursuit) {
  return pursuit ? PursuitDistance(gr, container_self) : gr.support_distance_m;
}

SpacingNormalization ResolveSpacingNormalization(const RelationPayload& payload,
                                                 const ElementTree& tree) {
  SpacingNormalization out;
  const auto& gr = payload.grounded_relation;
  if (gr.source_scope_diameter_m > 1e-6) {
    out.mode = SpacingNormalization::Mode::kScoped;
  } else if (gr.support_distance_m >= 0.0) {
    out.mode = SpacingNormalization::Mode::kUnscoped;
    return out;  // no context on either side
  } else {
    out.mode = SpacingNormalization::Mode::kLegacyScoped;
  }
  out.scope = CorrespondingScope(payload, tree, &out.scope_element);
  if (out.scope) {
    const Bound2D b = out.scope->GetBounds();
    out.diameter_m = (b.max - b.min).norm();
  } else {
    out.scope_element = kInvalidElementId;
  }
  return out;
}

namespace {

// The approach pose a_p of a relation: the source anchor's offset in the support
// frame, read on the receiver's same-id support through the one chart, then lifted
// to world (v1's approach point plus offset). A referent whose boundary is not in
// the element-local frame, or a named part, has no chart: the offset is lifted
// rigidly.
std::optional<Eigen::Vector2d> MapAnchorPoint(const Eigen::Vector2d& anchor_local,
                                              const RelationPayload& payload,
                                              const Element& referent,
                                              const ElementTree& tree) {
  Eigen::Vector2d local = anchor_local;
  const auto* def = referent.GetDeclaredSpatialDefinition();
  const bool chart = def && def->RequiresElementWorldTransform() &&
                     payload.support.sub_region_name.empty() &&
                     payload.support.boundary && def->GetBoundary();
  if (chart) {
    if (auto mapped = MapSupportPoint(local, *payload.support.boundary,
                                      *def->GetBoundary());
        mapped.ok() && mapped->allFinite())
      local = *mapped;
  }
  // A Surface keeps its boundary in its own frame, not the element frame the
  // offset is in (a world-absolute Navigable is charted in the element frame, see
  // ContainmentTransfer.WorldAbsoluteRegionIsChartedInTheElementFrame), so there
  // is no point of the offset's frame to put through the chart, and the element
  // origin is not the support: lifting the offset from the receiver element's
  // origin lands the body at the source's raw coordinates, not on the receiver's
  // seat. The support is what corresponds, so the offset is carried from the
  // mapped contact: the contact read through the chart, plus the source's
  // contact-to-anchor vector lifted rigidly.
  if (dynamic_cast<const SurfaceSpatialDefinition*>(def) &&
      payload.support.sub_region_name.empty() && payload.support.has_contact) {
    SupportDecodeInfo unused;
    const auto contact = DecodeSupportPoint(payload, referent, tree, &unused);
    if (contact.ok() && contact->allFinite()) {
      const Eigen::Matrix4d to_world = tree.GetWorldTransform(referent.GetId());
      // An engaged source (inside the referent, or touching it: the one contact
      // epsilon, as SourceEngaged) IS at the support point the chart encoded (its
      // own position), so it has no offset.
      const Eigen::Vector2d offset =
          (payload.support.has_interior ||
           (payload.grounded_relation.support_distance_m >= 0.0 &&
            payload.grounded_relation.support_distance_m < kContactEpsilonM))
              ? Eigen::Vector2d::Zero().eval()
              : Eigen::Vector2d(anchor_local - payload.support.contact_point_local);
      const Eigen::Vector2d lifted_offset =
          to_world.block<2, 2>(0, 0) * offset;
      return *contact + lifted_offset;
    }
  }
  const Eigen::Vector4d lifted =
      tree.GetWorldTransform(referent.GetId()) *
      Eigen::Vector4d(local.x(), local.y(), 0.0, 1.0);
  if (!lifted.allFinite()) return std::nullopt;
  return Eigen::Vector2d(lifted.x(), lifted.y());
}

std::optional<Eigen::Vector2d> ApproachPose(const RelationPayload& payload,
                                            const Element& referent,
                                            const ElementTree& tree) {
  const auto& es = payload.grounded_relation.encoded_state;
  if (es.size() < 2) return std::nullopt;
  return MapAnchorPoint(Eigen::Vector2d(es(0), es(1)), payload, referent, tree);
}

// The mapped entering direction: v1's approach direction is the direction of the
// source's final straight leg onto the contact (EuclideanND::BuildPathToBoundary:
// last waypoint to the boundary point), stored in the target's local frame. The
// receiver reads that leg through the same chart as the contact and the approach
// pose: the start of the leg (the source anchor when it sees the contact, else the
// last waypoint of its geodesic) mapped like the approach pose, ending at the
// mapped contact. Nullopt when the leg has no direction (the source is at the
// contact).
std::optional<Eigen::Vector2d> MappedEnteringDirection(
    const RelationPayload& payload, const Element& referent,
    const ElementTree& tree, const Eigen::Vector2d& mapped_contact_world) {
  const auto& es = payload.grounded_relation.encoded_state;
  Eigen::Vector2d origin_local;
  if (payload.support.has_approach_origin) {
    origin_local = payload.support.approach_origin_local;
  } else if (es.size() >= 2) {
    origin_local = Eigen::Vector2d(es(0), es(1));
  } else {
    return std::nullopt;
  }
  const auto origin = MapAnchorPoint(origin_local, payload, referent, tree);
  if (!origin) return std::nullopt;
  const Eigen::Vector2d leg = mapped_contact_world - *origin;
  if (!(leg.norm() > 1e-9)) return std::nullopt;
  return leg.normalized();
}

// The source is inside its referent, or at contact with it: decided on the metric
// the kernel measured, not on a threshold of the normalized proximity (see
// DecodeTargetPoint). A container-self relation with the actor inside carries its
// depth below the border as the metric distance, but the actor is inside. Contact is
// the one contact epsilon of the count and engagement (kContactEpsilonM, strict <).
bool SourceEngaged(const RelationPayload& payload) {
  return (payload.support.container_self && payload.support.has_interior) ||
         (payload.grounded_relation.support_distance_m >= 0.0 &&
          payload.grounded_relation.support_distance_m < kContactEpsilonM);
}

// v1's inside-volume snap: the source is inside its referent or at contact with
// it. A real source then has a finite update count (the contact override makes it
// one; arriving at contact makes it d/(d - 0)), so a payload that claims contact
// with no count at all is not a request. A container's border relation carries its
// depth, not a distance to contact, so the source being inside it is the whole
// condition.
bool SnapsToChart(const RelationPayload& payload) {
  return SourceEngaged(payload) &&
         (std::isfinite(payload.update_count) || payload.support.container_self);
}

// Decode the mapped source contact onto the receiver support. That point is the
// geometry R is measured against. The actor pose is sample ⊕ δ from the
// continuation solve, not a stand-off along an offset curve or a path walk.
// `branch_out` stays 0: the contact chart has one rule. Interior and
// container-self poses still transfer through the referent's own chart.
absl::StatusOr<Eigen::Vector2d> DecodeTargetPoint(
    const RelationPayload& payload, const Element& target_referent,
    const ElementTree& tree, bool* exact_copy_out = nullptr,
    int64_t* branch_out = nullptr) {
  if (branch_out) *branch_out = 0;
  Eigen::Matrix4d world_t = tree.GetWorldTransform(target_referent.GetId());
  SupportDecodeInfo sinfo;
  auto decoded_support =
      DecodeSupportPoint(payload, target_referent, tree, &sinfo);
  if (exact_copy_out) *exact_copy_out = false;
  if (!decoded_support.ok()) return decoded_support.status();
  const Eigen::Vector2d support_point = *decoded_support;

  // Preserve the source's relation value (R_tgt = R_src, Eq 8). For spacing the
  // encoded state holds the subject anchor in the SUPPORT frame — the source's
  // offset from its support. Reapplying that offset at the target support
  // places the avatar at the same support-relative state, so it mirrors the
  // source's approach LEVEL instead of being driven onto the referent
  // (R_tgt -> 1). An engaged source (offset 0) reduces to the support point.
  if (payload.kernel_type == RelationKernelType::kAvatarSpacing &&
      payload.grounded_relation.encoded_state.size() >= 2) {
    const auto& es = payload.grounded_relation.encoded_state;
    Eigen::Vector2d local(es(0), es(1));

    // Engaged / INSIDE the referent (R_src ≈ 1): the avatar is AT the support
    // point — for a containment anchor (a relation to the room/object the actor
    // is inside) that support point is the actor's mapped interior position, so
    // there is no approach offset to add. Returning early also avoids the
    // no-scale fallback below double-counting `local` (support_point + local),
    // which for an inside relation would push the avatar off to ~2x its
    // position (e.g. interior (11,11) → (22,22) clamped to a room corner).
    //
    // "Engaged" is decided on the metric the kernel actually measured, not on
    // a threshold of the NORMALIZED proximity: R = 1 - d/D saturates against
    // 1 - 1e-3 whenever d < 1e-3 * D, which in a 20 m room (D = 28 m) is every
    // source up to ~2.8 cm OUTSIDE the referent edge. Treating that band as
    // engaged returned the zero-stand-off support point, so the receiver S0
    // sat on the edge, its R^tgt disagreed with R^src by more than the refinement's
    // improvement epsilon, and the paper-objective compass search wandered over the flat
    // R level set to a point up to 0.55 m away (the scene-level Spacing step
    // outside a region edge). A metric stand-off d > 0 is an approach and goes
    // through the stand-off decode below; only d == 0 (inside) or a legacy
    // payload with no metric falls back to the proximity threshold.
    // A container-self relation with the actor inside (support.container_self)
    // carries its depth below the border as the metric distance, but the actor is
    // INSIDE: its pose is the interior chart, not a stand-off.
    const bool source_engaged = SourceEngaged(payload);
    if (source_engaged) {
      // Within-element pose: reconstruct the interior position with the TARGET
      // referent's own canonical chart (box-UVW / MVC, type-agnostic), so it
      // transfers faithfully across a differently shaped/typed corresponding
      // element. Falls back to the support point when no interior was encoded
      // (e.g. legacy payloads) or the referent has no spatial definition.
      // A non-convex RECEIVER maps the interior with the same star/arc-length
      // chart as its boundary (`support_point`, the chart decode of the source
      // position): the box-UVW interior chart lands in the notch of an L and
      // disagrees with the boundary chart by up to the referent's extent, and
      // the containment-transfer pose of the same element uses this chart too.
      // An identical copy of an element-local referent: the source's own pose
      // relative to the referent, lifted through the receiver's placement,
      // exactly (the interior chart of a non-convex source is a sampled MVC).
      if (sinfo.same_shape_as_source && payload.support.has_interior)
        if (const auto* idef = target_referent.GetDeclaredSpatialDefinition())
          if (idef->RequiresElementWorldTransform()) {
            const Eigen::Vector4d w4 =
                world_t * Eigen::Vector4d(local.x(), local.y(), 0.0, 1.0);
            if (exact_copy_out) *exact_copy_out = true;
            return Eigen::Vector2d(w4.x(), w4.y());
          }
      // Inside the referent the pose is the approach pose: the anchor's offset read
      // through the same chart as the contact (v1's polygon coordinate mapping).
      // The offset is in the ELEMENT frame (the kernel encodes it through the
      // element's world transform). A Surface keeps its boundary in its own
      // frame, so for it the element-frame offset is not a point of the
      // boundary, and lifting it rigidly puts the receiver at the source's raw
      // coordinates instead of on its own seat. The inside
      // pose is then `support_point`: the anchor's own boundary-frame position,
      // which the grounding rule encoded and the decode above mapped and lifted.
      const bool surface_frame = dynamic_cast<const SurfaceSpatialDefinition*>(
                                     target_referent.GetDeclaredSpatialDefinition()) != nullptr;
      if (payload.support.has_interior && !surface_frame) {
        if (const auto approach = ApproachPose(payload, target_referent, tree)) {
          if (exact_copy_out && sinfo.same_shape_as_source) *exact_copy_out = true;
          return *approach;
        }
      }
      return support_point;
    }

    // The mapped contact is the target geometry R is measured against. The
    // actor's position is the continuation solve (sample ⊕ δ), not a stand-off
    // guessed from an offset curve, an offset ray, or a path walk.
    return support_point;
  }

  return support_point;
}

}  // namespace

// Map StateType to indices in the state array.
std::vector<int> GetStateIndices(StateType type) {
  switch (type) {
    case StateType::kPosition2D:
      return {kStateX, kStateY};
    case StateType::kPosition3D:
      return {kStateX, kStateY, kStateZ};
    case StateType::kRotationYaw:
      return {kStateYaw};
    case StateType::kRotation3D:
      return {kStateYaw, kStatePitch, kStateRoll};
  }
  return {};
}

// Convenience overload: uniform weights.
SolverResult RelationSolver::Solve(
    const Eigen::Vector2d& current_position, double current_yaw,
    const std::vector<RelationPayload>& payloads,
    const ElementTree& target_tree,
    const NavigableSpatialDefinition* navigable_def,
    const ReferentResolver* resolver) const {
  return Solve(current_position, current_yaw, payloads, {}, target_tree,
               navigable_def, resolver);
}

// Primary solve: geodesic goal-based solver.
//
// Instead of Ceres encoded-state matching, this solver:
// 1. Decodes a target point per payload from the support's contact mapping
// 2. Determines position/yaw goals from each kernel's dynamic states
// 3. Solves position via iterative gradient descent with ClampToPolygon
// 4. Solves yaw via weighted circular mean of bearings to target points
namespace {

// Obstacle footprints the root may approach or occupy.
//
// Every obstacle keeps the full body-radius clearance EXCEPT the referent the
// source is actually engaging, by the absolute criterion of engagement.h:
//
// ON (footprint open): the source root is inside the referent's footprint
// (spacing support.has_interior -- the case DecodeTargetPoint decodes to an
// interior pose). Only then may the root target lie on the footprint;
// approach proximity never grants it (the previous "prox >= 0.5 =>
// footprint + 0.5 m" rule did, and pulled the root onto seats the source
// only stood or squatted next to).
//
// IN CONTACT (ring open): the source root touches the referent's edge (support
// distance 0, R == 1). The body-radius ring of that referent is dropped -- the
// root may come as close to its edge as the source does -- but its footprint
// stays closed. No radius: F changes with the relation's own contact, never with
// a threshold (one F).
//
// Containment scopes (navigable-role referents such as the room) are
// skipped: R = 1 there only means "inside the room" and must not exempt the
// obstacles it contains. The referent's obstacle-role ancestors (below the
// navigable scope) and descendants share its status, so a seat modelled as
// object + surface elements opens as one footprint.
void GatherEngagedFootprints(const std::vector<RelationPayload>& payloads,
                             const ElementTree& tree,
                             const ReferentResolver& resolve,
                             std::vector<int64_t>* on,
                             std::vector<int64_t>* in_contact) {
  for (const auto& p : payloads) {
    if (p.kernel_type != RelationKernelType::kAvatarSpacing) continue;
    std::vector<int64_t>* out = IsOnReferent(p)             ? on
                                : IsEngaged(p) ? in_contact
                                                           : nullptr;
    if (!out) continue;
    const Element* ref = (p.resolved_local_element_id != kInvalidElementId)
                             ? tree.GetElement(p.resolved_local_element_id)
                             : resolve.Resolve(p.referent_shared_id, tree);
    if (!ref || !ref->GetDeclaredSpatialDefinition()) continue;
    if (ref->HasGeometryRole(GeometryRole::kNavigable)) continue;
    for (element_id_t id : tree.GetSubtreeIds(ref->GetId())) out->push_back(id);
    for (element_id_t e = ref->GetParentId(); e != kInvalidElementId;) {
      const Element* anc = tree.GetElement(e);
      if (!anc || anc->HasGeometryRole(GeometryRole::kNavigable)) break;
      out->push_back(e);
      e = anc->GetParentId();
    }
  }
}

}  // namespace

double FacingBearingConditioning(const Element& referent,
                                 const ElementTree& tree,
                                 const Eigen::Vector2d& position,
                                 double distance_to_look_at,
                                 double yaw_offset) {
  // The footprint, distance, and away fades are retired. Occlusion is the
  // scope gate; a look-at inside the body is the yaw stage's clearance check.
  (void)referent;
  (void)tree;
  (void)position;
  (void)distance_to_look_at;
  (void)yaw_offset;
  return 1.0;
}

namespace {

// Wraps to (-pi, pi], used only by the paper-objective search's yaw
// candidates below (the position/yaw solves above use BlendYaw's short-arc
// interpolation instead and do not need this).
double WrapAngle(double a) { return std::atan2(std::sin(a), std::cos(a)); }


// A polygon's rings (its outer ring and its extra rings) lifted to the world through `lift`, the frame it is stored in.
void AppendLiftedRings(const Eigen::Matrix4d& lift, const Polygon2D& poly,
                       std::vector<std::vector<Eigen::Vector2d>>* out) {
  const auto lift_ring = [&](const std::vector<Eigen::Vector2d>& pts) {
    std::vector<Eigen::Vector2d> w;
    w.reserve(pts.size());
    for (const auto& pt : pts) {
      const Eigen::Vector4d q = lift * Eigen::Vector4d(pt.x(), pt.y(), 0.0, 1.0);
      w.emplace_back(q.x(), q.y());
    }
    return w;
  };
  out->push_back(lift_ring(poly.GetPoints()));
  for (const auto& extra : poly.ExtraRings()) out->push_back(lift_ring(extra));
}

SolverResult SolveImpl(const Eigen::Vector2d& embodied_position,
                       double current_yaw,
                       const std::vector<RelationPayload>& payloads,
                       const std::vector<double>& weights,
                       const ElementTree& target_tree,
                       const NavigableSpatialDefinition* navigable_def,
                       bool use_step_progress,
                       const ReferentResolver* resolver,
                       const RelationSolver::Config& config,
                       std::optional<double> heading = std::nullopt,
                       const SolverWarmStart& warm_start_in = {}) {
  // The merged solver is idempotent (weighted-centroid + navigable clamp); the
  // progressive per-step mode is upstream-only and unused on the live Solve()
  // path (space.cc realizes via Solve, not SolveProgressive).
  (void)use_step_progress;
  ReferentResolver default_resolver;
  const ReferentResolver& resolve = resolver ? *resolver : default_resolver;
  SolverResult result;
  result.position = embodied_position;
  result.yaw = current_yaw;
  // How well the proposed position is known: the representable resolution of its coordinates for an analytic answer, and
  // the displacement the stop rule treats as no move (U2: sqrt(tau_F)(1 + |x|)) for one the descent ended at. The Facing
  // yaw is read at that position, so this is the uncertainty its bearings carry (the yaw stage).
  double position_resolution_m = -1.0;

  if (HasInvalidWeight(weights)) {
    result.status = SolverStatus::kFailed;
    return result;
  }

  if (payloads.empty()) {
    result.status = SolverStatus::kConverged;
    result.total_residual = 0.0;
    return result;
  }

  // Step 1: Compute goals from payloads
  struct GoalEntry {
    size_t payload_index = 0;  // index into `payloads` (for Eval_p reuse)
    Eigen::Vector2d target_position = Eigen::Vector2d::Zero();
    bool has_position_goal = false;
    bool has_yaw_goal = false;
    double weight = 1.0;
    // Source's facing offset (bearing - yaw) toward the referent, so the
    // avatar reproduces the same relative orientation instead of pointing
    // straight at the target.
    double yaw_offset = 0.0;
    // Receiver referent of this goal (facing footprint conditioning).
    const Element* referent = nullptr;
    // The decoded point is the source's own pose (DecodeTargetPoint's
    // exact_copy_out): the same shape, the stand-off not rescaled.
    bool exact_copy = false;
  };
  std::vector<GoalEntry> entries;
  // Enabled POSITION relations (non-Facing) that produced no goal. Facing
  // only orients: its drops are reported per relation and never fail the
  // solve.
  size_t enabled_dropped = 0;

  for (size_t i = 0; i < payloads.size(); ++i) {
    const auto& p = payloads[i];
    // Prefer the linker-resolved local id (handle-based cross-space identity);
    // fall back to the pluggable referent resolver (default: exact shared-id
    // match). The two compose — linker first, resolver second.
    const Element* referent =
        (p.resolved_local_element_id != kInvalidElementId)
            ? target_tree.GetElement(p.resolved_local_element_id)
            : resolve.Resolve(p.referent_shared_id, target_tree);
    const double w = WeightAt(weights, i);
    // Every payload gets an explicit outcome.
    auto drop = [&](SolverRelationDisposition d, std::string reason) {
      result.relations.push_back({i, d, std::move(reason)});
    };
    // A policy weight of zero disables this relation. Do not turn policy
    // intent into a synthetic epsilon constraint. Two exceptions:
    //  - The PLACED answer (a first proposal, a source exception) reads every enabled relation whatever its weight
    //    (weight only breaks its ties; Facing as well as Spacing), since on the
    //    first update no count is valid yet and Progressive weights are then all 0.
    //  - A position relation with no valid count is not disabled but makes NO REQUEST (v1: no plan, the entity stays
    //    where it is): Progressive computes no weight for it, and the pursuit omits it (RatioRequest), so an
    //    anchor whose relations all lack a count holds at its sample. Only a relation the policy weighed 0 although
    //    its count is valid (a Greedy loser) is disabled.
    const bool placed_reads = warm_start_in.absolute && p.enabled;
    const bool position_no_request =
        p.enabled && p.kernel_type != RelationKernelType::kAvatarFacing && !RatioRequest(p.update_count);
    if (w == 0.0 && !placed_reads && !position_no_request) {
      // Keep an idle Facing channel alive without changing the legacy
      // disabled-relation/yaw flags.
      if (p.enabled && referent &&
          p.kernel_type == RelationKernelType::kAvatarFacing &&
          !RatioRequest(p.update_count))
        result.yaw_active = true;
      drop(SolverRelationDisposition::kDisabled, "zero policy weight");
      continue;
    }
    // The root solve (position + yaw) is Spacing + Facing ONLY. An
    // AffordancePose relation is a per-body-anchor world pose, realized by
    // SolveActorTargets / GroundAffordanceTarget; it must never drag the
    // root. The live path already filters it
    // out before calling Solve() (Space::OnRemoteUpdate, `root_payloads`);
    // a direct Solve() call enforces the same contract itself, so the
    // paper-objective position stage can never be handed one. It is
    // reported as an explicit omission, never a position goal, and never
    // counts as a dropped POSITION relation (it does not fail the solve).
    if (p.kernel_type == RelationKernelType::kAvatarAffordancePose) {
      drop(SolverRelationDisposition::kUnsupported,
           "affordance-pose is realized per anchor, not by the root solve");
      continue;
    }
    if (!referent) {
      if (p.kernel_type != RelationKernelType::kAvatarFacing)
        ++enabled_dropped;
      drop(SolverRelationDisposition::kUnmapped,
           "referent unresolved on receiver");
      continue;
    }

    auto components = KernelFactory::Create(p.kernel_type);
    if (!components.kernel) {
      if (p.kernel_type != RelationKernelType::kAvatarFacing)
        ++enabled_dropped;
      drop(SolverRelationDisposition::kUnsupported, "unknown relation family");
      continue;
    }

    // Decode target point from support. A support the receiver cannot
    // reproduce is an explicit omission, never a centre target.
    bool exact_copy = false;
    int64_t decode_branch = 0;
    auto decoded = DecodeTargetPoint(p, *referent, target_tree, &exact_copy, &decode_branch);
    if (!decoded.ok()) {
      if (p.kernel_type != RelationKernelType::kAvatarFacing)
        ++enabled_dropped;
      drop(SolverRelationDisposition::kGroundingFailed,
           std::string(decoded.status().message()));
      continue;
    }
    Eigen::Vector2d target_pos = *decoded;
    for (const auto& ov : warm_start_in.decode_overrides)
      if (ov.first == i && p.kernel_type == RelationKernelType::kAvatarSpacing) target_pos = ov.second;

    // Determine position/yaw goals from kernel's dynamic states
    auto dynamic = components.kernel->GetDynamicStates();
    bool needs_position = false;
    bool needs_yaw = false;
    for (auto st : dynamic) {
      if (st == StateType::kPosition2D || st == StateType::kPosition3D)
        needs_position = true;
      if (st == StateType::kRotationYaw || st == StateType::kRotation3D)
        needs_yaw = true;
    }

    // Facing is an ORIENTATION relation: it must set yaw ONLY, never position.
    // Its decoded target point is still used below as the bearing reference for
    // the yaw solve (entry.target_position), but it must not enter the position
    // solve. A facing/room containment anchor (prox=1 at the room, weight
    // ~0.97) was competing with the engaged object's spacing relation for
    // POSITION and dragging the clone off the object it was at onto a
    // room-relative spot
    // ("table in apt0 -> TV in apt1"). Position now comes purely from the
    // spacing/affordance kernels; facing only orients.
    if (p.kernel_type == RelationKernelType::kAvatarFacing) {
      needs_position = false;
    }

    GoalEntry entry;
    entry.payload_index = i;
    entry.target_position = target_pos;
    entry.has_position_goal = needs_position;
    entry.has_yaw_goal = needs_yaw;
    entry.weight = w;
    entry.referent = referent;
    entry.exact_copy = exact_copy;
    // For facing relations, encoded_state[0] is the source's bearing offset
    // (bearing-to-referent minus the source's yaw). Reproducing it lets the
    // avatar keep the same orientation relative to its referent rather than
    // staring directly at it.
    if (needs_yaw && p.kernel_type == RelationKernelType::kAvatarFacing &&
        p.grounded_relation.encoded_state.size() > 0) {
      entry.yaw_offset = p.grounded_relation.encoded_state(0);
    }
    entries.push_back(entry);
    result.goals.push_back({i, target_pos, w, needs_position, needs_yaw});
    result.goals.back().decode_branch = decode_branch;
    result.relations.push_back({i, SolverRelationDisposition::kUsed, ""});
  }

  // Status semantics (aligned with geometry-mediation): kFailed = no
  // position target can be produced, hold in place -- only when enabled
  // position relations existed and none produced a goal. Partial drops, and
  // any Facing drop, are reported per relation in result.relations, never as
  // a blanket status.
  const bool has_position_goal =
      std::any_of(entries.begin(), entries.end(),
                  [](const GoalEntry& e) { return e.has_position_goal; });
  result.position_active = has_position_goal;
  if (enabled_dropped > 0 && !has_position_goal) {
    // D7: the source scope is authoritative. A referent
    // the receiver cannot map or ground drops; when every position referent drops
    // there is nothing to map, which is the empty-scope case, not a hold. The
    // result names it so the proposal says "nothing proposed" and the driver keeps
    // its body.
    result.status = SolverStatus::kConverged;
    result.no_grounded_referent = true;
    result.total_residual = 0.0;
    return result;
  }
  if (entries.empty()) {
    result.status = SolverStatus::kConverged;
    result.total_residual = 0.0;
    return result;
  }

  // Source discontinuity (a teleport or reset of the source) is a DECLARED
  // source-side event (SolverWarmStart::cold), not a threshold on how far the
  // source's pose moved. The embodied seed and the body's basin are then not
  // continued: the new pose's nearest exact point and the body's are mirrors.
  result.source_discontinuity = warm_start_in.cold;
  // (No seed is dropped on a discontinuity: the embodied sample is where a
  // multi-referent descent starts and where the Facing sign root is anchored,
  // whatever the source did; v1 had no global search to fall back on.)
  const SolverWarmStart& warm_start = warm_start_in;

  // `evaluations_used` counts the relation evaluations of this call (reported as
  // SolverResult::objective_evaluations). `objective_fallback` starts true
  // ("nothing improved": nothing is refined any more) and stays true.
  int evaluations_used = 0;
  // The solve's EVALUATION budget (Config::solve_budget_evals; a safety net, not a stop rule). The count is the solver's own
  // relation-evaluation counter (per position term per evaluation of J, plus the sample's; derivatives and route listings are not counted, see kP1SolveBudgetEvals), so the same solve is cut at the same place on any host under any load. Sticky: once a check has seen the
  // budget spent, every later check answers true, so the solve ends by one rule.
  const bool budget_on = config.solve_budget_evals > 0;
  bool budget_tripped = false;
  const auto over_budget = [&]() {
    if (!budget_on) return false;
    if (!budget_tripped && evaluations_used >= config.solve_budget_evals) budget_tripped = true;
    return budget_tripped;
  };
  TargetEvaluationOptions eval_options;
  eval_options.resolver = &resolve;
  result.objective_fallback = true;

  // Step 2: Solve position (gradient descent with ClampToPolygon)
  std::vector<const GoalEntry*> position_goals;
  for (const auto& e : entries) {
    if (e.has_position_goal) position_goals.push_back(&e);
  }

  // The root target must lie in the receiver's
  // FREE space -- inside the walkable, >= body radius from every obstacle
  // footprint -- except the footprint of a referent the source root is
  // actually on. Hoisted above the position-goal branch (rather than
  // block-local) so the paper-objective refinement below can re-use the same exemption
  // set and free-space/reachability gate for its own candidate positions.
  std::vector<int64_t> exempt, ring_exempt;
  GatherEngagedFootprints(payloads, target_tree, resolve, &exempt,
                          &ring_exempt);
  result.free_exempt = exempt;
  result.free_ring_exempt = ring_exempt;
  // The scope the root's relations are solved in: F and the geodesic over it (relation_scope.h). The solve below asks
  // only the scope; a space with no navigable floor is the unbounded scope, whose geodesic is the straight segment.
  const UnboundedScope<2> unbounded_scope;
  std::optional<WalkableScope> walkable_scope;
  if (navigable_def) walkable_scope.emplace(*navigable_def, exempt, ring_exempt);
  const RelationScope<2>& scope =
      walkable_scope ? static_cast<const RelationScope<2>&>(*walkable_scope) : unbounded_scope;
  // The embodied sample can lie outside F while the room contains it: the body went
  // where the exemptions allowed (a footprint the source engaged) and the source
  // released it, or the geometry moved under it. The solve is posed from the nearest
  // pose in F (the projection onto the constraint, as every other answer is), so the
  // proposal brings the body back into F instead of holding it where it is. A sample
  // outside the walkable is projected the same way (as v1 does: v1's GetNearestPoint projects the start into
  // the walkable set before it solves): the SOLVER's start point is the nearest pose in F, so
  // every iterate and the budget's incumbent are in F by construction. The embodied sample
  // itself is not changed. Only when F has no pose to project to does the pursuit run from
  // the sample (D7: no hold).
  Eigen::Vector2d current_position = embodied_position;
  // (In the walkable by the free space's own boundary tolerance: a sample a rounding error
  // outside the boundary, as a body that walked along a wall ends up, is ON it and is
  // projected, not left where it stands.)
  if (!scope.IsFree(embodied_position)) {
    // Under the solve's exemptions, from the free space built for them: exact at the
    // corner of an engaged footprint and a neighbour's ring, where the query-time
    // exemptions of NearestFree leave candidates out.
    auto nearest = scope.NearestFreeUnder(embodied_position);
    if (!nearest || !scope.IsFree(*nearest)) nearest = scope.NearestFree(embodied_position);
    if (nearest) {
      current_position = *nearest;
      result.feasibility = FeasibilityOutcome::kProjected;
    }
  }
  result.solve_origin = current_position;
  {
    // The clamp's constraint set (order-independent: each id mixed through a hash).
    uint64_t h = 1469598103934665603ull;
    const auto mix = [&](int64_t id, uint64_t salt) {
      uint64_t x = static_cast<uint64_t>(id) * 0x9E3779B97F4A7C15ull + salt;
      x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
      h += x;
    };
    for (int64_t id : exempt) mix(id, 1);
    for (int64_t id : ring_exempt) mix(id, 2);
    result.clamp_exempt_hash = h;
  }

  if (!position_goals.empty()) {
    // S0, the centroid of the decoded goal points, is recorded for attribution.
    double w_sum = 0.0;
    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    for (const auto* g : position_goals) {
      centroid += g->target_position * g->weight;
      w_sum += g->weight;
    }
    if (w_sum > 1e-9) centroid /= w_sum;
    const Eigen::Vector2d s0 = (w_sum > 1e-9) ? centroid : current_position;
    result.unconstrained_position = s0;
    result.has_unconstrained_position = true;

    // Position follows v1:
    //  - the first sample and a declared source discontinuity are PLACED at
    //    the approach pose, the source offset through the same-id support
    //    chart (v1's approach point plus offset);
    //  - a relation whose source is inside its referent, or at contact, is
    //    SNAPPED to the chart pose (v1's inside-volume snap), and the contact
    //    guarantee (n = 1, one-hot weight) leaves nothing else to weigh;
    //  - otherwise the receiver PURSUES each relation's mapped contact c_p by
    //    1/n of its course, where n is the source's remaining update count.
    //    A relation with no finite n (a static source) issues no request, so
    //    the receiver holds. One relation steps along its course; several are
    //    v1's StepSync on the distance-reduction residuals.
    // There is no level-set selection rule: no step picks a point of a curve of
    // exact answers, because no step asks for one.
    const bool placed = warm_start.absolute;

    struct PosTerm {
      PreparedTargetRelation prepared;
      double weight;
      double d_star;
      double d_sample;
      double update_count;
      // Relations this term stands for (identical terms are merged).
      int multiplicity = 1;
      // The mapped contact c_p of a relation whose source is ENGAGED on an object
      // (inside it, or at its contact): the source's own state mapped through the
      // chart, the pose the old snap answered (DecodeTargetPoint). The kernel
      // measures an inside source against the referent's nearest edge, which is
      // another point, so the term measures the geodesic to c_p itself.
      std::optional<Eigen::Vector2d> goal;
    };
    std::vector<PosTerm> pos_terms;

    // The relations the receiver pursues: every spacing relation with a finite
    // update count. Placement and snap instead report the source relation
    // (d* = the source distance on the receiver scope) at the pose they answer.
    //
    // A source that is inside its referent, or at its contact, has the update count
    // 1 of v1's contact override: the ratio relation reaches its mapped contact c_p
    // (d* = 0), and the policy's weights collapse onto it when it is the nearest to
    // contact (v1's annealing is one-hot at n = 1; Greedy takes the minimum). When the
    // relation that drives (the largest weight) is an engaged OBJECT, the chart pose
    // is the minimiser of J, so it is answered as such: exactly, without a descent,
    // and from any sample, in whatever component of F it lies.
    //
    // That holds only while the engaged relations carry ALL the weight. A relation
    // outside them that has weight and a request (a finite update count; a static
    // source issues none, so it is no term of J) pulls the minimiser off the chart
    // pose, and answering the engaged referent anyway overrides the weights: under
    // Uniform, whose relations keep one weight for the whole solve, the receiver arrived at the engaged referent whatever
    // the others asked. Then every relation is a term of J, the engaged ones at n = 1
    // (the weighted minimiser of the c_p goals under every policy).
    //
    // A CONTAINER's border relation is not an object relation: its depth below the
    // border cannot say where the actor is inside, and its answer is the mapping of
    // that interior position through the container's chart, not a minimiser of J. It keeps driving the snap as before.
    std::vector<const GoalEntry*> engaged;
    const GoalEntry* driving = nullptr;
    for (const auto* g : position_goals)
      if (!driving || g->weight > driving->weight) driving = g;
    const auto snappable = [&](const GoalEntry& g) {
      const RelationPayload& p = payloads[g.payload_index];
      return p.kernel_type == RelationKernelType::kAvatarSpacing && SnapsToChart(p);
    };
    if (driving && !placed && snappable(*driving)) {
      const bool border_drives = payloads[driving->payload_index].support.container_self;
      bool engaged_carries_all_weight = true;
      for (const auto* g : position_goals) {
        if (!(g->weight > 0.0)) continue;
        const RelationPayload& p = payloads[g->payload_index];
        if (snappable(*g))
          engaged.push_back(g);
        else if (std::isfinite(p.update_count) && p.update_count > 0.0)
          engaged_carries_all_weight = false;
      }
      if (!border_drives && !engaged_carries_all_weight) engaged.clear();
    }
    const bool snapped_to_chart = !placed && !engaged.empty();

    // An object relation whose source is engaged on it (n = 1 by the contact
    // override). It is an ordinary term of J: d* = 0, the receiver's whole course
    // to c_p, weighed with the other relations by the policy's weights.
    const auto engaged_object = [&](const RelationPayload& p) {
      return p.kernel_type == RelationKernelType::kAvatarSpacing &&
             !p.support.container_self && SourceEngaged(p);
    };
    // The geodesic length of the course from `from` to `to` in F (the length the
    // one-relation step walks), the straight line where F does not connect them
    // (the course falls back to it).
    const auto goal_distance = [&](const Eigen::Vector2d& from,
                                   const Eigen::Vector2d& to) {
      const auto reach = scope.Reachability(from, to);
      if (reach.reachable && std::isfinite(reach.length)) return reach.length;
      return (to - from).norm();
    };

    for (const auto& e : entries) {
      if (!e.has_position_goal) continue;
      const RelationPayload& p = payloads[e.payload_index];
      if (p.grounded_relation.encoded_state.size() == 0) continue;
      auto prep = PrepareTargetRelation(p, target_tree, eval_options);
      if (!prep.ok()) continue;
      const SpacingNormalization nrm = ResolveSpacingNormalization(p, target_tree);
      const double d_scope = p.grounded_relation.source_scope_diameter_m;
      const double d_src = p.grounded_relation.support_distance_m;
      double d_new = d_src;
      if (d_src >= 0.0 && nrm.mode != SpacingNormalization::Mode::kUnscoped &&
          nrm.diameter_m > 1e-6 && d_scope > 1e-6)
        d_new = d_src * (nrm.diameter_m / d_scope);
      if (placed || snapped_to_chart) {
        if (!(d_src >= 0.0)) continue;
        pos_terms.push_back({std::move(prep), e.weight, d_new, d_new, 1.0});
        continue;
      }
      if (!RatioRequest(p.update_count)) {
        // n = ∞: the term is omitted. The ratio does not move a static source.
        continue;
      }
      std::optional<Eigen::Vector2d> goal;
      if (engaged_object(p)) goal = e.target_position;
      double d_sample;
      if (goal) {
        d_sample = goal_distance(current_position, *goal);
      } else {
        TargetAnchorPose sample_anchor;
        sample_anchor.position =
            Eigen::Vector3d(current_position.x(), current_position.y(), 0.0);
        sample_anchor.yaw = current_yaw;
        ++evaluations_used;
        const TargetRelationEvaluation at_sample = EvaluatePreparedTargetRelation(
            prep, sample_anchor, target_tree, /*fast_distance=*/false,
            /*nearest_support_contact=*/p.support.container_self);
        d_sample = PursuitDistance(at_sample.relation, p.support.container_self);
        if (!at_sample.ok()) continue;
      }
      if (!(d_sample >= 0.0)) continue;
      const double d0 = d_sample;
      const double d_star = RatioTargetDistance(d0, p.update_count);
      result.ratio_intents.push_back({d0, p.update_count, e.weight, -1.0, e.payload_index});
      if (goal) {
        result.ratio_intents.back().engaged_goal = true;
        result.ratio_intents.back().goal = *goal;
      }
      pos_terms.push_back(
          {std::move(prep), e.weight, d_star, d0, p.update_count, 1, goal});
    }
    // Terms that are the same function of the pose are one term: the same
    // referent seen from the same anchor with the same target distance. Their
    // residuals are equal at every pose, so Σ w e² = (Σ w) e², and the
    // minimiser is unchanged. Several bindings of one relation (an app that
    // binds it per scope root) reach the solver as copies, and each copy would
    // otherwise cost a full geodesic query at every Jacobian sample and trial.
    {
      std::vector<PosTerm> merged;
      merged.reserve(pos_terms.size());
      for (auto& term : pos_terms) {
        const auto same = [&](const PosTerm& m) {
          return m.prepared.kernel_type == term.prepared.kernel_type &&
                 m.prepared.anchor_name == term.prepared.anchor_name &&
                 m.prepared.local_referent_id ==
                     term.prepared.local_referent_id &&
                 m.prepared.normalizing_scope_id ==
                     term.prepared.normalizing_scope_id &&
                 m.prepared.payload_for_eval->support.container_self ==
                     term.prepared.payload_for_eval->support.container_self &&
                 m.goal.has_value() == term.goal.has_value() &&
                 (!m.goal || *m.goal == *term.goal) && m.d_star == term.d_star;
        };
        const auto it = std::find_if(merged.begin(), merged.end(), same);
        if (it == merged.end()) {
          merged.push_back(std::move(term));
        } else {
          it->weight += term.weight;
          it->multiplicity += term.multiplicity;
        }
      }
      pos_terms = std::move(merged);
    }
    // The objective reads the geodesic length, not the contact chart. Skip
    // the chart encode on every Gauss-Newton sample. The per-referent ring
    // field still runs, and its hop is reused across iterations.
    for (auto& term : pos_terms) {
      if (auto* rule = term.prepared.components_for_eval.grounding_rule.get()) {
        rule->skip_chart_encoding = true;
        // Every evaluation here measures to the mapped contact, so the nearest
        // support contact is never read.
        if (term.prepared.has_mapped_support &&
            !term.prepared.payload_for_eval->support.container_self)
          rule->skip_nearest_contact = true;
      }
    }
    // The residual is already metres (geodesic distance); there is no R-to-metre
    // scale.

    // e = d(pose, c_p) − d*. J = sum w e^2. The distance is measured to the
    // mapped source contact, the point R is scored against and Eval reports.
    auto eval_errors = [&](const Eigen::Vector2d& pos, bool fast,
                           std::vector<double>* errors) {
      TargetAnchorPose a;
      a.position = Eigen::Vector3d(pos.x(), pos.y(), 0.0);
      a.yaw = current_yaw;
      double j_r = 0.0;
      int n = 0;
      if (errors) errors->assign(pos_terms.size(), 0.0);
      for (size_t i = 0; i < pos_terms.size(); ++i) {
        ++evaluations_used;
        if (pos_terms[i].goal && !(placed || snapped_to_chart)) {
          // An engaged object measures the course to its c_p.
          const double err = goal_distance(pos, *pos_terms[i].goal) - pos_terms[i].d_star;
          if (errors) (*errors)[i] = err;
          j_r += pos_terms[i].weight * err * err;
          n += pos_terms[i].multiplicity;
          continue;
        }
        // A container's border relation measures its depth below the nearest
        // border point; every other relation measures to the mapped contact.
        const TargetRelationEvaluation ev = EvaluatePreparedTargetRelation(
            pos_terms[i].prepared, a, target_tree, fast,
            /*nearest_support_contact=*/
            pos_terms[i].prepared.payload_for_eval->support.container_self);
        const double dist = TermDistance(
            ev.relation, pos_terms[i].prepared.payload_for_eval->support.container_self,
            !(placed || snapped_to_chart));
        if (!ev.ok() || !(dist >= 0.0)) {
          if (errors) (*errors)[i] = std::numeric_limits<double>::quiet_NaN();
          continue;
        }
        const double err = dist - pos_terms[i].d_star;
        if (errors) (*errors)[i] = err;
        j_r += pos_terms[i].weight * err * err;
        n += pos_terms[i].multiplicity;
      }
      return std::pair<double, int>(j_r, n);
    };

    auto in_f = [&](const Eigen::Vector2d& p) {
      return scope.IsFree(p);
    };
    // The derivatives of a relation's own distance (the pursuit's terms, scope_pursuit.h): analytic as in v1
    // (gradient = -first_segment / seg_len), d = |x - p1| + (fixed rest of
    // the course), p1 the first waypoint: grad d = (x - p1) / |x - p1|, hess d = (I - u u^T) / |x - p1|.
    using TermDeriv = ScopeTermDeriv<2>;
    const auto leg_deriv = [](const Eigen::Vector2d& x, const Eigen::Vector2d& p1) {
      TermDeriv out;
      out.analytic = true;
      const double r = (x - p1).norm();
      out.r = r;
      // v1's kEpsilon: a zero-length first segment has zero gradient.
      if (r > 1e-9) {
        out.g = (x - p1) / r;
        out.h = (Eigen::Matrix2d::Identity() - out.g * out.g.transpose()) / r;
      }
      return out;
    };
    // The first waypoint of a course that is not the iterate itself (a path that starts with a repeat of the pose, as
    // on the boundary, has a zero first leg and a gradient of 0 where the distance changes at rate 1).
    const auto first_far = [](const std::vector<Eigen::Vector2d>& pts, const Eigen::Vector2d& x,
                              const Eigen::Vector2d& fallback) {
      for (size_t q = 1; q < pts.size(); ++q)
        if ((pts[q] - x).norm() > 1e-9) return pts[q];
      return fallback;
    };
    // The relation's own distance at a pose of the pursuit (NaN: not evaluable), and its derivatives.
    const auto term_distance = [&](size_t i, const Eigen::Vector2d& pos, bool fast) {
      TargetAnchorPose a;
      a.position = Eigen::Vector3d(pos.x(), pos.y(), 0.0);
      a.yaw = current_yaw;
      // A container's border relation measures its depth below the nearest
      // border point; every other relation measures to the mapped contact.
      const TargetRelationEvaluation ev = EvaluatePreparedTargetRelation(
          pos_terms[i].prepared, a, target_tree, fast,
          /*nearest_support_contact=*/
          pos_terms[i].prepared.payload_for_eval->support.container_self);
      const double dist = TermDistance(
          ev.relation, pos_terms[i].prepared.payload_for_eval->support.container_self, /*pursuit=*/true);
      return ev.ok() ? dist : std::numeric_limits<double>::quiet_NaN();
    };
    const auto term_deriv = [&](size_t i, const Eigen::Vector2d& pos) -> TermDeriv {
      TargetAnchorPose a;
      a.position = Eigen::Vector3d(pos.x(), pos.y(), 0.0);
      a.yaw = current_yaw;
      const bool container_self = pos_terms[i].prepared.payload_for_eval->support.container_self;
      const TargetRelationEvaluation ev = EvaluatePreparedTargetRelation(
          pos_terms[i].prepared, a, target_tree, /*fast=*/false, /*nearest_support_contact=*/container_self);
      if (!ev.ok()) return TermDeriv{};
      const double dist = TermDistance(ev.relation, container_self, /*pursuit=*/true);
      if (dist == 0.0) {
        TermDeriv zero;
        zero.analytic = true;
        // Inside the referent the distance is clamped to 0 (a set with an interior: zero derivatives); on the
        // contact point of a spacing relation (no interior) it is the vertex of a cone.
        if (!ev.support.has_interior && !container_self && ev.relation.contact_distance_m == 0.0) zero.vertex = true;
        return zero;
      }
      const Support& sup = ev.support;
      if (dist == ev.relation.contact_distance_m && sup.navigable_def && !sup.fast_distance) {
        const Eigen::Vector2d to = (sup.has_contact ? sup.contact_point_world : sup.GetPosition3D()).head<2>();
        // A spacing relation's mapped contact is measured with the referent's own footprints open on the final leg
        // (the evaluator, OwnFootprints): its first waypoint and segments are the shortest such route's.
        const bool mapped_spacing = !container_self && ev.kernel_type == RelationKernelType::kAvatarSpacing &&
                                    pos_terms[i].prepared.has_mapped_support && pos_terms[i].prepared.referent_for_eval;
        std::vector<int64_t> own;
        if (mapped_spacing) own = OwnFootprints(target_tree, *pos_terms[i].prepared.referent_for_eval);
        TermDeriv td;
        if (!own.empty()) {
          const auto routes = sup.navigable_def->GeodesicRoutes(pos, to, 0.0, own);
          if (routes.empty()) return TermDeriv{};
          td = leg_deriv(pos, routes.front().first);
          td.legs = std::max(1, routes.front().legs);
        } else {
          const auto path = sup.navigable_def->GetDiscretePath(pos, to);
          const auto& pts = path.GetPoints();
          td = leg_deriv(pos, first_far(pts, pos, to));
          td.legs = std::max<int>(1, static_cast<int>(pts.size()) - 1);
        }
        td.route_kind = 2;
        td.route_to = to;
        td.route_open = own;
        // The mapped contact of a spacing relation is a FIXED point (the evaluator overrides the support's contact
        // with it); every other contact is the grounding rule's nearest point of the referent, which moves.
        td.moving_contact = !(!container_self && ev.kernel_type == RelationKernelType::kAvatarSpacing &&
                              pos_terms[i].prepared.has_mapped_support);
        if (container_self && sup.has_interior && sup.boundary && sup.has_boundary_lift) {
          // route_kind 3: a border relation of the container the iterate stands in: the geodesic to the nearest point
          // of this polygon's boundary, so its pieces are the polygon's edges.
          td.route_kind = 3;
          const std::shared_ptr<Polygon2D> border = sup.boundary;
          const Eigen::Matrix4d lift = sup.boundary_lift;
          td.rings = [border, lift]() {
            std::vector<std::vector<Eigen::Vector2d>> world_rings;
            AppendLiftedRings(lift, *border, &world_rings);
            return world_rings;
          };
        } else if (td.moving_contact && !container_self && sup.has_boundary_lift && pos_terms[i].prepared.referent_for_eval &&
                   pos_terms[i].prepared.referent_for_eval->GetDeclaredSpatialDefinition()) {
          // route_kind 4: a spacing relation whose contact is the NEAREST point of the referent's boundary: its pieces
          // are the referent's boundary edges (its named parts' edges when it has parts), lifted by the support's lift.
          td.route_kind = 4;
          const SpatialDefinition* contact_def = pos_terms[i].prepared.referent_for_eval->GetDeclaredSpatialDefinition();
          const Eigen::Matrix4d lift = sup.boundary_lift;
          td.rings = [contact_def, lift]() {
            std::vector<std::vector<Eigen::Vector2d>> world_rings;
            // The grounding rule picks the nearest contact over the referent's named parts when it has them (each
            // part's boundary in the frame of its own transform), else over its whole boundary (PickGeodesicContact).
            bool any_part = false;
            for (const auto& sub : contact_def->GetSubRegions()) {
              if (!sub.boundary || sub.boundary->GetPoints().size() < 2) continue;
              any_part = true;
              AppendLiftedRings(lift * sub.local_transform, *sub.boundary, &world_rings);
            }
            if (!any_part) {
              if (const auto boundary = contact_def->GetBoundary())
                if (boundary->GetPoints().size() >= 2) AppendLiftedRings(lift, *boundary, &world_rings);
            }
            return world_rings;
          };
        }
        return td;
      }
      return TermDeriv{};
    };

    // A point referent has no polygon: its approach point is the decoded contact.
    auto has_polygon = [&](const GoalEntry& e) {
      if (e.referent == nullptr) return false;
      const RelationPayload& pose_payload = payloads[e.payload_index];
      if (pose_payload.support.boundary) return true;
      const auto* def = e.referent->GetDeclaredSpatialDefinition();
      return def && def->GetBoundary();
    };

    // Where the receiver ends up when it does not pursue.
    //  placed: the approach pose a_p of the closest relation: the source
    //          offset through the support chart, lifted to world.
    //  snap:   the weighted mean of the engaged relations' decoded chart poses.
    Eigen::Vector2d answer = current_position;
    bool pursue = !placed && !snapped_to_chart;
    if (placed) {
      // v1 places the body at the approach point of the closest relation (the
      // one the source is most about to reach), not at a mean of poses that
      // belong to different objects. Ties go to the larger weight.
      const GoalEntry* nearest = nullptr;
      double nearest_d = std::numeric_limits<double>::infinity();
      for (const auto* g : position_goals) {
        if (g->referent == nullptr) continue;
        const double d = payloads[g->payload_index].grounded_relation.support_distance_m;
        const double dd = d >= 0.0 ? d : std::numeric_limits<double>::max();
        if (!nearest || dd < nearest_d || (dd == nearest_d && g->weight > nearest->weight)) {
          nearest = g;
          nearest_d = dd;
        }
      }
      answer = s0;
      if (nearest) {
        const GoalEntry& g = *nearest;
        Eigen::Vector2d pose = g.target_position;
        if (has_polygon(g) || position_goals.size() > 1)
          if (const auto approach =
                  ApproachPose(payloads[g.payload_index], *g.referent, target_tree))
            pose = *approach;
        answer = pose;
      }
    } else if (snapped_to_chart) {
      Eigen::Vector2d snap = Eigen::Vector2d::Zero();
      double snap_w = 0.0;
      for (const auto* g : engaged) {
        snap += g->target_position * g->weight;
        snap_w += g->weight;
      }
      answer = snap_w > 1e-9 ? Eigen::Vector2d(snap / snap_w) : s0;
    }

    if (!pursue) {
      // The answer is a pose, not a step: it is clamped into F and reported
      // against the source relation. The sample is not an input.
      if (!in_f(answer)) {
        if (const auto nearest = scope.NearestFree(answer)) answer = *nearest;
      }
      std::optional<FeasibilityOutcome> blocked;
      if (!in_f(answer)) {
        // No free pose answers the mapped relation: the answer stays the embodied
        // sample and the diagnostic says why. The status is not a hold (D7).
        blocked = FeasibilityOutcome::kSampleOutside;
        {
          ObstacleFreeSpace::NearestFreeStatus st =
              ObstacleFreeSpace::NearestFreeStatus::kSearchFoundNone;
          scope.NearestFree(answer, &st);
          if (st == ObstacleFreeSpace::NearestFreeStatus::kDomainEmpty)
            blocked = FeasibilityOutcome::kNoFreeSpace;
        }
        answer = current_position;
      }
      std::vector<double> e_here;
      const auto [j_here, n_here] = eval_errors(answer, /*fast=*/false, &e_here);
      double max_abs = 0.0;
      int n_ok = 0;
      for (double e : e_here) {
        if (!std::isfinite(e)) continue;
        max_abs = std::max(max_abs, std::fabs(e));
        ++n_ok;
      }
      result.position = answer;
      result.absolute_placement = placed;
      result.chart_snap = snapped_to_chart;
      result.has_objective_seed = true;
      result.objective_seed_position = answer;
      result.objective_before += j_here;
      result.objective_after += j_here;
      result.objective_relation_count += n_here;
      result.objective_exact =
          n_ok > 0 && max_abs <= config.objective_improve_epsilon_m;
      result.objective_fallback = false;
      result.feasibility = blocked ? *blocked
                           : scope.Bounded() ? FeasibilityOutcome::kFeasible
                                             : FeasibilityOutcome::kUnconstrained;
      result.feasibility_snap_m = (answer - s0).norm();
    } else {
      // The pursuit of the root's relations in their scope: the one live path of every anchor (scope_pursuit.h).
      ScopePursuitInput<2> pin;
      pin.scope = &scope;
      pin.start = current_position;
      pin.gn_safety_cap = config.gn_safety_cap;
      pin.exact_eps_m = config.objective_improve_epsilon_m;
      pin.budget_evals = config.solve_budget_evals;
      pin.warm_cold = warm_start.cold;
      for (size_t i = 0; i < pos_terms.size(); ++i) {
        const PosTerm& pt = pos_terms[i];
        ScopeTerm<2> t;
        t.weight = pt.weight;
        t.d_star = pt.d_star;
        t.d_sample = pt.d_sample;
        t.update_count = pt.update_count;
        t.multiplicity = pt.multiplicity;
        t.goal = pt.goal;
        t.course_goal = pt.goal ? *pt.goal
                        : pt.prepared.has_mapped_support
                            ? Eigen::Vector2d(pt.prepared.mapped_support.head<2>())
                            : position_goals.front()->target_position;
        // A container's border relation measures its depth below the border, not the length of a course to a contact,
        // so its one-relation step stays the relation's own distance over n along the discrete course.
        t.course_relation = !(pt.prepared.payload_for_eval && pt.prepared.payload_for_eval->support.container_self);
        if (pos_terms.size() == 1 && t.course_relation && !pt.goal && pt.prepared.has_mapped_support &&
            pt.prepared.payload_for_eval && pt.prepared.referent_for_eval)
          t.entering = MappedEnteringDirection(*pt.prepared.payload_for_eval, *pt.prepared.referent_for_eval,
                                               target_tree, pt.prepared.mapped_support.head<2>());
        if (!pt.goal && t.course_relation && pt.prepared.has_mapped_support && pt.prepared.referent_for_eval &&
            pt.prepared.payload_for_eval &&
            pt.prepared.payload_for_eval->kernel_type == RelationKernelType::kAvatarSpacing)
          t.open = OwnFootprints(target_tree, *pt.prepared.referent_for_eval);
        t.distance = [&term_distance, i](const Eigen::Vector2d& pos, bool fast) { return term_distance(i, pos, fast); };
        t.deriv = [&term_deriv, i](const Eigen::Vector2d& pos) { return term_deriv(i, pos); };
        pin.terms.push_back(std::move(t));
      }
      ScopePursuitResult<2> pr;
      pr.status = result.status;
      pr.gn_stop = result.gn_stop;
      pr.feasibility = result.feasibility;
      pr.objective_before = result.objective_before;
      pr.objective_after = result.objective_after;
      pr.objective_relation_count = result.objective_relation_count;
      pr.objective_fallback = result.objective_fallback;
      pr.objective_exact = result.objective_exact;
      pr.budget_hit_iteration = result.budget_hit_iteration;
      pr.chi_descent_taken = result.chi_descent_taken;
      pr.chi_descent_tries = result.chi_descent_tries;
      pr.fd_terms = result.fd_terms;
      pr.projection_excursions = result.projection_excursions;
      pr.retract_cap_hits = result.retract_cap_hits;
      pr.route_list_queries = result.route_list_queries;
      PursueInScope<2>(pin, pr, evaluations_used, budget_tripped, position_resolution_m);
      result.position = pr.position;
      result.objective_seed_position = pr.objective_seed_position;
      result.has_objective_seed = pr.has_objective_seed;
      result.no_request = pr.no_request;
      result.objective_stationary = pr.objective_stationary;
      result.objective_exact = pr.objective_exact;
      result.objective_fallback = pr.objective_fallback;
      result.objective_warm_start_used = pr.objective_warm_start_used;
      result.budget_hit = pr.budget_hit;
      result.budget_incumbent_is_warm_start = pr.budget_incumbent_is_warm_start;
      result.budget_hit_iteration = pr.budget_hit_iteration;
      result.iterations = pr.iterations;
      result.objective_iterations = pr.objective_iterations;
      result.objective_relation_count = pr.objective_relation_count;
      result.chi_descent_taken = pr.chi_descent_taken;
      result.chi_descent_tries = pr.chi_descent_tries;
      result.fd_terms = pr.fd_terms;
      result.projection_excursions = pr.projection_excursions;
      result.retract_cap_hits = pr.retract_cap_hits;
      result.route_list_queries = pr.route_list_queries;
      result.incumbent_objective = pr.incumbent_objective;
      result.objective_before = pr.objective_before;
      result.objective_after = pr.objective_after;
      result.status = pr.status;
      result.gn_stop = pr.gn_stop;
      result.feasibility = pr.feasibility;
      if (pr.one_relation_step) {
        // What an observer needs to recompute the step from its inputs.
        for (auto& intent : result.ratio_intents) {
          intent.step_m = pr.step_m;
          intent.course = pr.course;
          intent.course_relation = pr.course_relation;
          if (pr.course_relation) {
            intent.smoothed = pr.smoothed;
            intent.has_entering = pin.terms.front().entering.has_value();
            if (pin.terms.front().entering) intent.entering = *pin.terms.front().entering;
          }
        }
      }
      result.feasibility_snap_m = (result.position - s0).norm();
    }
  }

  result.position_requested = result.position_active && !result.no_request;

  // Step 3: Solve yaw (weighted circular mean of bearings to target points)
  std::vector<const GoalEntry*> yaw_goals;
  for (const auto& e : entries) {
    if (e.has_yaw_goal) yaw_goals.push_back(&e);
  }

  // UNSPECIFIED YAW (v1 behaviour; there is no source-heading
  // fallback). A Facing is well defined only where its bearing is: its look-at point does
  // not coincide with the body (D2: the direction from the body to its own position is
  // arbitrary).
  // Distance, footprint, and away fades do not drop a Facing. The yaw is the circular
  // mean of the well-defined bearings, weighted w*c. With none left the yaw is UNSPECIFIED:
  // the flag says so, the value carries no meaning (the held yaw is echoed), and the driver
  // faces its direction of motion (and keeps its yaw when stationary).
  // D2: a look-at is in-body only when it COINCIDES with
  // the body (v1: delta < kEpsilon), where the direction to it is undefined. The
  // receiver's body radius is not a scope and does not drop a Facing.
  double sin_sum = 0.0, cos_sum = 0.0;
  double target_sin_sum = 0.0, target_cos_sum = 0.0;  // the same mean of the UNSTEPPED targets (the definition's tie)
  bool facing_no_request = false;                      // a Facing without a valid count: the yaw holds
  std::map<const GoalEntry*, double> stepped_yaw;     // a Facing's ratio step, when it is not the target itself (n > 1)
  std::map<const GoalEntry*, double> stepped_turn;    // ... and the signed, unwrapped yaw turn it requests
  double turn_weight = 0.0;                            // the weight of the relations' requested turns
  bool long_way_requested = false;                     // a weighted stepping Facing requests a turn of pi or more
  std::vector<std::pair<double, double>> stepped_requests;  // (m, requested turn) of each weighted long-way stepper
  {
    std::vector<const GoalEntry*> kept;
    for (const auto* g : yaw_goals) {
      const Eigen::Vector2d to_target = g->target_position - result.position;
      const double c = g->referent
                           ? FacingBearingConditioning(*g->referent, target_tree,
                                                       result.position, to_target.norm(),
                                                       g->yaw_offset)
                           : 1.0;
      auto& wg = result.goals[g - entries.data()];
      wg.yaw_conditioning = c;
      if (LookAtInBody(to_target.norm()) || c <= 0.0) {
        // Left out of the circular mean: it enters with conditioning 0 (the reported c is the
        // weight the bearing entered the mean with, so a consumer that recomputes the mean
        // from the goals gets the solver's own).
        wg.yaw_conditioning = 0.0;
        wg.yaw_excluded = true;
        continue;
      }
      // A Facing with no weight carries no yaw (and is not in the referent set: the
      // signature drops weight <= 0 exactly, so the flag and its cause flip together),
      // except in a PLACED answer, which reads every relation whatever its weight.
      // The facing direction is a ratio too: on a continuation the yaw follows the ratio
      // continuation, as the position does: a Facing makes a request only with a weight and a valid count, and a
      // Facing without a valid count is no request (the yaw holds, as the v1 hold). This differs from v1, whose yaw is
      // PLACED every update (v1's ComputeWeightedTargetYaw faces the targets directly).
      const RelationPayload& yp = payloads[g->payload_index];
      if (!warm_start_in.absolute && !RatioRequest(yp.update_count)) {
        facing_no_request = true;
        continue;
      }
      if (!(g->weight > 0.0) && !warm_start_in.absolute) continue;
      kept.push_back(g);
      // The target: the yaw that reproduces the source's relative orientation, (bearing - yaw) = offset, i.e.
      // yaw = bearing - offset.
      const double oriented = std::atan2(to_target.y(), to_target.x()) - g->yaw_offset;
      wg.yaw_bearing = oriented;
      // The step: 1/n of the receiver's OWN remaining offset, the angle from its current yaw to its look-at point (the
      // relation's contact, the counterpart of the source's: the source's count n is v1's on ITS offset to its look-at,
      // so closing 1/n of the receiver's offset to the mapped look-at is the position's rule in angle, and a receiver
      // that is the source's copy stays its copy). n = 1, and a placed answer, take the target yaw itself (bearing -
      // source offset; the source is aligned there, |offset| below the contact epsilon in degrees), so n = 1 reproduces
      // the placed yaw. A remaining offset of pi is a tie (left and right equally short), within the one contact epsilon
      // read in degrees (the "aligned" test of FacingUpdateCount); it turns the way the receiver last turned, and with no
      // previous turn keeps remainder()'s own sign.
      // The source counts its offset to ONE look-at point at two poses: its previous sent pose and this one (the
      // bearing changes with its position as well as its yaw). The receiver's own offset is therefore read at ITS pose
      // before this step (the embodied position and yaw), and the stepped offset is reached at the answer's position:
      // the yaw is the bearing from the answer to the look-at less the stepped offset. A receiver that is the source's
      // copy then stays its copy while both translate. Where the look-at coincides with the body before the step the
      // offset there is undefined (D2), as the source's is when it is co-located (its count is then 1): the target yaw
      // itself.
      const Eigen::Vector2d to_target_before = g->target_position - embodied_position;
      const bool before_in_body = LookAtInBody(to_target_before.norm());
      // A count below 1 is the source turning PAST alignment within this update (its offset crossed zero): the source
      // arrived, so the receiver arrives on this update too, at its target yaw (the paper's
      // formula is the reference, v1 approximates it). Its fraction is 1: the step is not extrapolated past the
      // receiver's own remaining offset by 1/n (which turned a lagging body by its whole lag times 1/n). This is the
      // count itself, not a cap: n <= 1 is arrival, n > 1 a step of 1/n. (The position ratio needs no such rule: a
      // distance cannot cross 0, so SourceUpdateCount is never below 1, and RatioTargetDistance already lands on 0.)
      const bool arrives = !(yp.update_count > 1.0);
      const double f = warm_start_in.absolute || before_in_body || arrives ? 1.0 : 1.0 / yp.update_count;
      if (f != 1.0) {
        // The receiver reproduces the source's motion: its OWN offset o = bearing - yaw to its mapped
        // look-at (read at its pose before the step) travels in the source's rotation direction s
        // (RelationPayload::facing_rotation_sign), and its remaining turn is the branch of that travel, d0 + 2 pi k,
        // nearest the source's own remaining turn (whether the source closes, the short way, or opens, the long way
        // round, as v1's angular proximity). The source's remaining turn at its old offset is R_old = n |rotation|, and
        // at its current offset (the payload's state) R_now = R_old - |rotation|, so R_old = R_now n / (n - 1), with
        // n > 1 here (f != 1). A receiver that is the source's copy has o = the source's old offset (branch 0).
        // k may be negative: a receiver a little past its look-at while the source is nearly done
        // turns back the short way. With k >= 0 it turned a whole extra revolution, 1/n of it per update (a source head
        // turning ~100 degrees, its receiver ~460).
        // A step needs the source's rotation sign and state: a count with no rotation sign is no rotation the receiver
        // can reproduce (FacingUpdateCount returns no valid count when the source does not rotate), so a payload
        // without one makes no request and the yaw holds, as for any Facing without a valid count.
        const int s = yp.facing_rotation_sign;
        if (s == 0 || yp.grounded_relation.encoded_state.size() == 0) {
          kept.pop_back();
          facing_no_request = true;
          continue;
        }
        const double o =
            std::remainder(std::atan2(to_target_before.y(), to_target_before.x()) - current_yaw, 2.0 * M_PI);
        const auto travel = [](double offset, int dir) {  // the angle `offset` travels in direction dir back to 0
          return std::fmod(std::fmod(-dir * offset, 2.0 * M_PI) + 2.0 * M_PI, 2.0 * M_PI);
        };
        const double n = yp.update_count;
        const double source_remaining = travel(yp.grounded_relation.encoded_state(0), s) * n / (n - 1.0);
        const double d0 = travel(o, s);
        const double k = std::round((source_remaining - d0) / (2.0 * M_PI));
        const double offset_move = s * (d0 + 2.0 * M_PI * k);  // the receiver offset's change over the remaining turn
        stepped_yaw[g] = std::atan2(to_target.y(), to_target.x()) - (o + offset_move * f);
        result.yaw_requested_turn_max = std::max(result.yaw_requested_turn_max, std::fabs(offset_move * f));
        // The signed turn it requests: the bearing's turn from the pose before the step to the answer, less the offset's
        // move (a yaw change of t moves the offset by -t), unwrapped.
        stepped_turn[g] = std::remainder(std::atan2(to_target.y(), to_target.x()) -
                                             std::atan2(to_target_before.y(), to_target_before.x()),
                                         2.0 * M_PI) -
                          offset_move * f;
      }
    }
    // The mean DIRECTION does not depend on the scale of the weights, so the sums are taken
    // with the weights relative to the largest: a Facing whose weight is 1e-320 (the nearest
    // referent has taken all of it) then does not lose its direction to the underflow of
    // w * cos(b), which rounded the circular mean to a multiple of 90 degrees.
    double w_max = 0.0;
    for (const auto* g : kept) w_max = std::max(w_max, g->weight);
    // An ARRIVING Facing (n <= 1) turns the whole remaining way to its target, the shortest arc, not
    // the source's turn direction; so does a placed answer.
    for (const auto* g : kept) {
      const auto& wg = result.goals[g - entries.data()];
      // A placed answer whose Facings all weigh 0 (the first update: no count yet) reads them alike: weight only breaks
      // ties, and there is nothing to break.
      const double rel = w_max > 0.0 ? g->weight / w_max : 1.0;
      const double m = rel * wg.yaw_conditioning;
      // Unit vector of the oriented bearing from the look-at components, so
      // two opposite axis-aligned bearings cancel to the zero vector. sin/cos
      // of atan2 do not: sin(pi) is not 0.
      const Eigen::Vector2d to_target = g->target_position - result.position;
      const double r = to_target.norm();
      const double bc = r > 0.0 ? to_target.x() / r : 1.0;
      const double bs = r > 0.0 ? to_target.y() / r : 0.0;
      const double co = std::cos(g->yaw_offset);
      const double so = std::sin(g->yaw_offset);
      target_cos_sum += m * (bc * co + bs * so);
      target_sin_sum += m * (bs * co - bc * so);
      // The requested turn (ABI 4.12, SolverResult::yaw_turn): only a LONG-way stepping Facing (|turn| >= pi) carries a
      // branch of its own. A short-way stepping Facing, like an arriving or placed one, requests a shortest arc to its
      // own answer (branch 0): measured against the answer's arc, round((t - base) / 2 pi), a short request near a half
      // turn read as +-1 (+183 degrees at n = 1.01 for a -177 degree shortest arc, -175 at n = 1.00).
      if (const auto tt = stepped_turn.find(g); tt != stepped_turn.end() && m > 0.0 && std::fabs(tt->second) >= M_PI) {
        long_way_requested = true;
        stepped_requests.emplace_back(m, tt->second);
      }
      turn_weight += m;
      if (const auto st = stepped_yaw.find(g); st != stepped_yaw.end()) {
        // n > 1: the stepped yaw's own direction.
        cos_sum += m * std::cos(st->second);
        sin_sum += m * std::sin(st->second);
      } else {
        // The target itself (n = 1, or placed): its unit vector from the look-at components, so
        // n = 1 reproduces the placed yaw bit for bit.
        cos_sum += m * (bc * co + bs * so);
        sin_sum += m * (bs * co - bc * so);
      }
    }
    yaw_goals = std::move(kept);
  }
  result.yaw_specified = !yaw_goals.empty();
  // The circular resultant (sin_sum, cos_sum) = sum m_i (cos b_i, sin b_i), m_i = (w_i / w_max) c_i, b_i the oriented bearing
  // to look-at point i, has a direction only if it is distinguishable from zero. It is computed from the unit vectors
  // (to_target / r) and the offsets' cos and sin: each product bc co + bs so carries gamma_4 (Higham, Accuracy and
  // Stability of Numerical Algorithms, 2nd ed., Lemma 3.1; a unit vector component has gamma_3, the two products and the
  // sum add one rounding each), the weight scaling one more, and the sum of n terms gamma_n of its magnitude: so
  // |computed - exact| <= gamma_{n+6} sum m_i. The position it is read at is known to position_resolution_m, and a
  // bearing turns by at most |dx| / r_i when the point moves by dx: that adds sum (m_i / r_i) position_resolution_m. A
  // resultant not above their sum has no direction the computation can name.
  result.facing_tie = false;
  bool stepped_tie = false;
  double resultant_scale = 0.0;
  if (!yaw_goals.empty()) {
    double w_top = 0.0;
    for (const auto* g : yaw_goals) w_top = std::max(w_top, g->weight);
    double m_sum = 0.0, m_over_r = 0.0;
    for (const auto* g : yaw_goals) {
      const auto& wg = result.goals[g - entries.data()];
      const double m = (g->weight / w_top) * wg.yaw_conditioning;
      const double r = (g->target_position - result.position).norm();
      m_sum += m;
      if (r > 0.0) m_over_r += m / r;
    }
    constexpr double ku = 1.1102230246251565e-16;
    const double kk = static_cast<double>(yaw_goals.size()) + 6.0;
    const double pos_res = position_resolution_m >= 0.0
                               ? position_resolution_m
                               : 2.220446049250313e-16 * std::max(1.0, result.position.cwiseAbs().maxCoeff());
    const double bound = kk * ku / (1.0 - kk * ku) * m_sum + m_over_r * pos_res;
    const double rn = std::hypot(sin_sum, cos_sum);
    // The definition's tie is the mean of the relations' TARGETS (unstepped): opposing bearings are a singularity of
    // the definition whatever step the ratio takes towards them. The yaw's own ambiguity
    // is the stepped mean's (stepped_tie): only then is a middle chosen.
    const double rn_target = std::hypot(target_sin_sum, target_cos_sum);
    resultant_scale = m_sum;
    result.facing_resultant = m_sum > 0.0 ? rn_target / m_sum : 0.0;
    result.facing_resultant_bound = m_sum > 0.0 ? bound / m_sum : 0.0;
    result.facing_tie = m_sum > 0.0 && rn_target <= bound;
    stepped_tie = m_sum > 0.0 && rn <= bound;
  }
  (void)resultant_scale;
  result.yaw_specified = !yaw_goals.empty();
  if (result.yaw_specified) {
    // Exactly antipodal: the resultant has no direction, so there are two
    // middles. They are the perpendiculars of the heaviest bearing. The one
    // nearer the embodied yaw is the continuation. The resultant counts as zero
    // when it is within the error interval of its own computation (facing_tie
    // above; exact zero is the case of an empty interval).
    if (stepped_tie) {
      const GoalEntry* heavy = yaw_goals.front();
      for (const auto* g : yaw_goals)
        if (g->weight > heavy->weight) heavy = g;
      const auto st = stepped_yaw.find(heavy);
      const double axis = st != stepped_yaw.end() ? st->second : result.goals[heavy - entries.data()].yaw_bearing;
      const double a = std::remainder(axis + M_PI / 2.0, 2.0 * M_PI);
      const double b = std::remainder(axis - M_PI / 2.0, 2.0 * M_PI);
      const auto ang = [](double u, double v) {
        return std::fabs(std::remainder(u - v, 2.0 * M_PI));
      };
      // Of the two middles, the one the receiver
      // reaches turning the way it last turned (the sign of the step from its current yaw equals its turn sign); with no
      // previous turn, or both on one side, the nearer one.
      const double sa = std::remainder(a - current_yaw, 2.0 * M_PI), sb = std::remainder(b - current_yaw, 2.0 * M_PI);
      const int ta = (sa > 0.0) - (sa < 0.0), tb = (sb > 0.0) - (sb < 0.0);
      // The tie at n = 1: while the receiver is pursuing (some Facing steps by 1/n, n > 1) a
      // tie keeps its current turn direction; on the arrival update (every Facing at n = 1, or a placed answer) a tie
      // takes the nearer middle. A receiver already at a middle stays there either way.
      const bool arriving = stepped_yaw.empty();
      if (ta == 0 || tb == 0)
        result.yaw = ta == 0 ? a : b;
      else if (!arriving && warm_start_in.turn_sign != 0 && ta != tb &&
               (ta == warm_start_in.turn_sign || tb == warm_start_in.turn_sign))
        result.yaw = ta == warm_start_in.turn_sign ? a : b;
      else
        result.yaw = ang(a, current_yaw) <= ang(b, current_yaw) ? a : b;
    } else {
      result.yaw = std::atan2(sin_sum, cos_sum);
      // The relations' TARGETS tie (opposing bearings: the definition has two middles, a half turn apart) while the
      // stepped mean has a direction: the receiver takes the middle it reaches turning the way it last turned (it
      // must not swing back and forth between the two).
      // (Only while pursuing: on the arrival update, n = 1 everywhere, the rule takes the nearer middle, which
      // the stepped mean already is.)
      if (result.facing_tie && !warm_start_in.absolute && warm_start_in.turn_sign != 0 && !stepped_yaw.empty()) {
        const double step = std::remainder(result.yaw - current_yaw, 2.0 * M_PI);
        const double other = std::remainder(result.yaw + M_PI, 2.0 * M_PI);
        const double other_step = std::remainder(other - current_yaw, 2.0 * M_PI);
        const int sign = (step > 0.0) - (step < 0.0), other_sign = (other_step > 0.0) - (other_step < 0.0);
        if (sign == -warm_start_in.turn_sign && other_sign == warm_start_in.turn_sign) result.yaw = other;
      }
    }
  }

  // The ratio continuation's own residual (the yaw against each relation's step target).
  result.yaw_stepped = result.yaw_specified && !stepped_yaw.empty();
  if (result.yaw_stepped) {
    double worst = 0.0;
    for (const auto* g : yaw_goals) {
      const auto st = stepped_yaw.find(g);
      const double y = st != stepped_yaw.end() ? st->second : result.goals[g - entries.data()].yaw_bearing;
      worst = std::max(worst, std::fabs(std::remainder(result.yaw - y, 2.0 * M_PI)));
    }
    result.yaw_step_residual = worst;
  }
  result.yaw_requested = !yaw_goals.empty();
  result.yaw_active =
      result.yaw_active || result.yaw_requested || facing_no_request;
  // No Facing request but a Facing without a valid count: the yaw holds at the anchor's current yaw (the
  // v1 hold, as the position's). A disabled Facing (weight 0 with a valid count) does not hold it.
  if (!result.yaw_specified && facing_no_request) {
    result.yaw = current_yaw;
    result.yaw_specified = true;
  }
  // ABI 4.12: the signed, unwrapped turn from the current yaw that the answered yaw is (the proposal carries the
  // turn, not only an absolute yaw). It is congruent to the yaw (current_yaw + yaw_turn = yaw, modulo 2 pi): the
  // shortest arc, unless the weighted vote of the requests' branches below (each long-way stepping Facing its own,
  // every other Facing branch 0) is a majority off it, which may exceed pi while a Facing reproduces a source turning
  // away the long way round; an arrival is the shortest arc. A hold is 0; a placed answer (first
  // proposal, source exception) is the shortest arc. Without a specified yaw it is 0.
  // Only a stepping Facing that itself requests pi or more takes the turn off the shortest arc. Each other request is
  // a shortest arc to its own answer, and their LINEAR mean is not the arc to the CIRCULAR mean of the answers: several
  // arriving Facings whose targets lie on both sides of the receiver's yaw (one source look-at line mapped to look-ats
  // in different directions on the receiver) averaged arcs of +161 and -74 degrees to +83, and the branch nearest
  // that turned the head +191 degrees to a yaw 169 degrees the other way.
  if (result.yaw_specified) {
    const double base = std::remainder(result.yaw - current_yaw, 2.0 * M_PI);
    result.yaw_turn = base;
    if (!warm_start_in.absolute && turn_weight > 0.0 && long_way_requested) {
      // The branch is the weighted mean of each long-way stepping Facing's OWN branch relative to the answer,
      // round((t_i - base) / 2 pi), with every other Facing (arriving, or stepping the short way) counting branch 0
      // (its request is the shortest arc): one Facing alone gets its own branch, as before; a long-way Facing
      // outweighed by arrivals on both sides of the yaw does not let their linear mean decide.
      double branch_sum = 0.0;
      for (const auto& [m, t] : stepped_requests) branch_sum += m * std::round((t - base) / (2.0 * M_PI));
      // A non-zero branch only beyond one half: an exact half (two Facings of equal weight, one long-way, one arriving)
      // is no majority, and std::round's half-away-from-zero took the long branch (279 degrees for requests of 191 and
      // -7). Halves round toward zero.
      const double mean = branch_sum / turn_weight;
      const double branch = std::copysign(std::ceil(std::fabs(mean) - 0.5), mean);
      result.yaw_turn = base + 2.0 * M_PI * branch;
    }
  }

  // The Facing sign (FACING_ROOT_SWITCH): the sign of the signed bearing offset
  // theta_tgt of the heaviest well-conditioned Facing term at the answered yaw, 0
  // inside the deadband (the yaw is decoded
  // like the position, the conditioned circular mean above, with no J search, no
  // root scan and no compass polish; the signed offset the source holds already
  // fixes which of the two mirror roots of the even R the mean follows). Reported
  // only; it never moves the yaw.
  if (!yaw_goals.empty()) {
    const GoalEntry* heavy = nullptr;
    for (const auto* g : yaw_goals) {
      const RelationPayload& p = payloads[g->payload_index];
      if (p.kernel_type != RelationKernelType::kAvatarFacing) continue;
      if (p.grounded_relation.encoded_state.size() == 0) continue;
      const Eigen::Vector2d to_target = g->target_position - result.position;
      const double c =
          g->referent ? FacingBearingConditioning(*g->referent, target_tree,
                                                  result.position,
                                                  to_target.norm(),
                                                  g->yaw_offset)
                     : 1.0;
      if (c <= 1e-9) continue;  // excluded: keeps the heading fallback
      if (!heavy || g->weight > heavy->weight) heavy = g;
    }
    if (heavy) {
      auto prep = PrepareTargetRelation(payloads[heavy->payload_index], target_tree,
                                        eval_options);
      if (prep.ok()) {
        if (prep.components_for_eval.grounding_rule)
          prep.components_for_eval.grounding_rule->skip_chart_encoding = true;
        TargetAnchorPose a;
        a.position = Eigen::Vector3d(result.position.x(), result.position.y(), 0.0);
        a.yaw = result.yaw;
        const TargetRelationEvaluation ev =
            EvaluatePreparedTargetRelation(prep, a, target_tree);
        // No deadband: the sign is the sign of the signed offset, and 0 only when it is exactly 0.
        // A band around zero has no citation in the papers or in v1 (v1's Angular1D takes the sign
        // of the exact shortest delta), so a +-theta root
        // switch inside it would go unreported.
        if (ev.ok() && ev.relation.encoded_state.size() >= 1) {
          const double th = std::remainder(ev.relation.encoded_state(0), 2.0 * M_PI);
          if (th != 0.0) result.facing_sign = th > 0 ? 1 : -1;
        }
      }
    }
  }
  result.objective_evaluations = evaluations_used;

  // Compute total residual. A loop-guard hit keeps kNotConverged: it is not a finished minimiser.
  if (result.status != SolverStatus::kFailed &&
      result.status != SolverStatus::kNotConverged &&
      result.status != SolverStatus::kBudgetHit)
    result.status = SolverStatus::kConverged;
  result.total_residual = 0.0;
  for (const auto* g : position_goals) {
    double d = (result.position - g->target_position).norm();
    result.total_residual += g->weight * d * d;
  }

  return result;
}

}  // namespace

SolverResult RelationSolver::Solve(
    const Eigen::Vector2d& current_position, double current_yaw,
    const std::vector<RelationPayload>& payloads,
    const std::vector<double>& weights, const ElementTree& target_tree,
    const NavigableSpatialDefinition* navigable_def,
    const ReferentResolver* resolver, std::optional<double> heading,
    const SolverWarmStart& warm_start) const {
  return SolveImpl(current_position, current_yaw, payloads, weights,
                   target_tree, navigable_def, /*use_step_progress=*/false,
                   resolver, config_, heading, warm_start);
}

absl::StatusOr<GroundedAffordanceTarget> RelationSolver::GroundAffordanceTarget(
    const RelationPayload& payload, const ElementTree& tree) const {
  if (payload.kernel_type != RelationKernelType::kAvatarAffordancePose)
    return absl::InvalidArgumentError("payload is not an affordance relation");
  if (payload.resolved_local_element_id == kInvalidElementId &&
      payload.referent_shared_id == kInvalidSharedElementId)
    return absl::InvalidArgumentError(
        "affordance payload has no referent identity");
  const Element* referent =
      payload.resolved_local_element_id != kInvalidElementId
          ? tree.GetElement(payload.resolved_local_element_id)
          : tree.GetElementBySharedId(payload.referent_shared_id);
  if (!referent) return absl::NotFoundError("target referent not found");

  GroundedAffordanceTarget target;
  target.local_element_id = referent->GetId();
  target.referent_id = payload.referent_shared_id;
  // The binding the source extracted this relation under.
  target.binding_revision = payload.binding_revision;
  target.binding_revision_known = payload.binding_revision_known;
  const double relative_yaw =
      payload.grounded_relation.encoded_state.size() >= 3
          ? payload.grounded_relation.encoded_state(2)
          : 0.0;
  if (!std::isfinite(relative_yaw))
    return absl::InvalidArgumentError("invalid affordance relative yaw");
  const SpatialDefinition* def = referent->GetDeclaredSpatialDefinition();
  // A referent without support geometry has no surface to put
  // the anchor on. Its origin is not a contact, so the relation is reported
  // as a grounding failure instead of producing a target at the centre (the
  // retired kLegacyCenterFallback mode).
  if (!def)
    return absl::FailedPreconditionError(
        "target referent has no support geometry");
  if (!payload.support.has_contact)
    return absl::FailedPreconditionError(
        "target geometry requires a contact chart");

  const NormalizedPolygonPosition& chart = payload.support.contact_normalized;
  if (chart.strategy != NormalizedPolygonPosition::Strategy::kRadial)
    return absl::UnimplementedError("MVC affordance charts are unsupported");
  if (!std::isfinite(chart.angle) || !std::isfinite(chart.radial_ratio) ||
      chart.angle < -M_PI || chart.angle > M_PI || chart.radial_ratio < 0.0 ||
      chart.radial_ratio > 1.0)
    return absl::InvalidArgumentError("invalid affordance support chart");

  std::shared_ptr<Polygon2D> boundary;
  Eigen::Matrix4d support_in_def = Eigen::Matrix4d::Identity();
  if (!payload.support.sub_region_name.empty()) {
    const SubRegion* sub = def->GetSubRegion(payload.support.sub_region_name);
    if (!sub || !sub->boundary)
      return absl::NotFoundError("target support sub-region not found");
    boundary = sub->boundary;
    support_in_def = sub->local_transform;
  } else if (dynamic_cast<const SurfaceSpatialDefinition*>(def) ||
             dynamic_cast<const NavigableSpatialDefinition*>(def)) {
    boundary = def->GetBoundary();
    target.mode = AffordanceGroundingMode::kWholePlanarBoundary;
  } else {
    return absl::FailedPreconditionError(
        "unannotated volume has no support plane");
  }
  if (!boundary || boundary->GetPoints().size() < 3)
    return absl::InvalidArgumentError("target support boundary is degenerate");
  // Affordance E/R share a strict radial domain and deterministic kernel
  // origin. Legacy spacing/facing and MVC mapper paths remain unchanged.
  // A patch whose chart is a frame chart (SubRegion::chart) reads the source contact in the rectangle's (u,v) instead,
  // plus the source anchor's normal offset: the same mapping a stroke or any other point of the patch goes through
  // (sb_part_chart_*). Both spaces declare the same kind for matched patches; a mismatch is reported.
  absl::StatusOr<Eigen::Vector3d> decoded3 = absl::InternalError("unset");
  {
    auto kind = SurfaceChartKindFromInt(payload.support.sub_region_chart);
    if (!kind.ok()) return kind.status();
    const SubRegion* declared =
        payload.support.sub_region_name.empty() ? nullptr : def->GetSubRegion(payload.support.sub_region_name);
    const SurfaceChartKind receiver_kind = declared ? declared->chart : SurfaceChartKind::kRadial;
    if (receiver_kind != *kind)
      return absl::FailedPreconditionError("matched sub-regions declare different surface charts");
    if (IsFrameChart(*kind)) {
      if (!payload.support.boundary)
        return absl::FailedPreconditionError("frame chart needs the source part boundary");
      const double normal = std::isfinite(payload.grounded_relation.ap_height_m)
                                ? payload.grounded_relation.ap_height_m
                                : 0.0;
      decoded3 = PlanarFrameChart::Map(payload.support.contact_point_local, normal, *payload.support.boundary,
                                       *boundary, *kind);
    } else {
      auto d2 = RadialSurfaceChart::Decode(chart, *boundary, Eigen::Vector2d(0, 1));
      if (!d2.ok()) return d2.status();
      decoded3 = Eigen::Vector3d(d2->x(), d2->y(), 0.0);
    }
  }
  if (!decoded3.ok()) return decoded3.status();
  const Eigen::Matrix4d def_to_world =
      def->RequiresElementWorldTransform()
          ? tree.GetWorldTransform(referent->GetId())
          : def->GetTransform();
  const Eigen::Matrix4d support_to_world = def_to_world * support_in_def;
  target.support_plane_to_world = support_to_world;
  // The chart version is NOT surfaced as chart_version_known: rich
  // K1 schema v1 cannot carry a known chart version. It is exported through
  // sb_relation_state_v2_t instead.
  target.support_identifier = payload.support.sub_region_name.empty()
                                  ? "whole-planar-boundary"
                                  : payload.support.sub_region_name;
  target.position = (support_to_world * Eigen::Vector4d(decoded3->x(), decoded3->y(), decoded3->z(), 1)).head<3>();
  target.yaw =
      std::atan2(support_to_world(1, 0), support_to_world(0, 0)) + relative_yaw;
  if (!target.position.allFinite() || !std::isfinite(target.yaw))
    return absl::InvalidArgumentError("invalid grounded affordance target");
  return target;
}

// Two-stage realizer. Stage A groups payloads by
// (kernel family, subject anchor) and solves each group independently for its
// own state component; Stage B composes in dependency order. The structural
// guarantee: no group's payloads ever enter another group's solve, so an
// affordance relation (whose dynamic states include position AND yaw) can
// never drag root — for any current or future relation family.
RealizationResult RelationSolver::SolveActorTargets(
    const Eigen::Vector2d& current_position, double current_yaw,
    const std::vector<RelationPayload>& payloads,
    const std::vector<double>& weights, const ElementTree& target_tree,
    const NavigableSpatialDefinition* navigable_def,
    const SolverWarmStart& warm_start) const {
  RealizationResult result;
  result.status = SolverStatus::kConverged;
  if (payloads.empty()) return result;
  if (HasInvalidWeight(weights)) {
    result.status = SolverStatus::kFailed;
    return result;
  }

  bool has_enabled_relation = false;
  for (size_t i = 0; i < payloads.size(); ++i) {
    if (WeightAt(weights, i) > 0.0) {
      has_enabled_relation = true;
      break;
    }
  }
  // An all-zero policy is an explicit release, not a request for a fabricated
  // root hold target.
  if (!has_enabled_relation) return result;

  std::vector<size_t> spacing_idx;
  std::vector<size_t> facing_idx;
  std::map<std::string, std::vector<size_t>> affordance_groups;
  for (size_t i = 0; i < payloads.size(); ++i) {
    switch (payloads[i].kernel_type) {
      case RelationKernelType::kAvatarSpacing:
        spacing_idx.push_back(i);
        break;
      case RelationKernelType::kAvatarFacing:
        facing_idx.push_back(i);
        break;
      case RelationKernelType::kAvatarAffordancePose:
        // The inner affordance-pose terms are the anchor's own N-D solve (Space::BuildAnchorProposalsLocked), not a
        // grounded contact.
        if (!IsInnerPosePayload(payloads[i])) affordance_groups[payloads[i].subject_anchor_name].push_back(i);
        break;
    }
  }

  auto subset = [&](const std::vector<size_t>& idx,
                    std::vector<RelationPayload>& sub_payloads,
                    std::vector<double>& sub_weights) {
    sub_payloads.reserve(idx.size());
    sub_weights.reserve(idx.size());
    for (size_t i : idx) {
      sub_payloads.push_back(payloads[i]);
      if (!weights.empty() && i < weights.size())
        sub_weights.push_back(weights[i]);
    }
    if (sub_weights.size() != sub_payloads.size()) sub_weights.clear();
  };

  // Root position ← spacing group only. With an empty group this returns the
  // current (commanded) pose, so root holds still when nothing drives it. The
  // navigable-clamp relaxation around engaged footprints is derived inside
  // SolveImpl from these spacing payloads.
  std::vector<RelationPayload> spacing_payloads;
  std::vector<double> spacing_weights;
  subset(spacing_idx, spacing_payloads, spacing_weights);
  SolverResult pos_result =
      Solve(current_position, current_yaw, spacing_payloads, spacing_weights,
            target_tree, navigable_def, nullptr, std::nullopt, warm_start);

  // Yaw ← facing group, bearings evaluated AT the composed root position —
  // the one legitimate cross-group coupling, made explicit here.
  std::vector<RelationPayload> facing_payloads;
  std::vector<double> facing_weights;
  subset(facing_idx, facing_payloads, facing_weights);
  SolverResult yaw_result =
      Solve(pos_result.position, current_yaw, facing_payloads, facing_weights,
            target_tree, navigable_def, nullptr,
            ReferentRelativeHeading(spacing_payloads, spacing_weights,
                                    target_tree),
            warm_start);

  // The realization status comes from the POSITION solve only; the
  // separate facing solve never fails it (a missing facing referent leaves
  // the yaw with the heading fallback) and its drops are reported below.
  result.status = pos_result.status;
  for (const auto& r : pos_result.relations)
    result.relations.push_back({spacing_idx[r.payload_index], r.disposition,
                                r.reason});
  for (const auto& r : yaw_result.relations)
    result.relations.push_back({facing_idx[r.payload_index], r.disposition,
                                r.reason});

  RealizationTarget root;
  root.anchor_name = "root";
  root.position = pos_result.position;
  root.yaw = yaw_result.yaw;
  root.kernel_type = RelationKernelType::kAvatarSpacing;
  root.weight = 1.0;
  result.targets.push_back(std::move(root));

  // One world pose per affordance anchor group, blended within the group only.
  // Decoded directly from the target support (not gradient descent): position
  // via a verified target support chart, including that support's Z and yaw
  // plus the source's support-relative yaw.
  // Deliberately NOT clamped to the navigable polygon — a drawable surface or
  // seat is not walkable; feasibility is the reach clamp (later step).
  bool had_grounding_error = false;
  for (const auto& [anchor_name, idx] : affordance_groups) {
    double weight_sum = 0.0;
    Eigen::Vector2d pos_sum = Eigen::Vector2d::Zero();
    double z_sum = 0.0, sin_sum = 0.0, cos_sum = 0.0, arrival_sum = 0.0;
    shared_element_id_t dominant_referent = kInvalidSharedElementId;
    double dominant_weight = -1.0;
    size_t used = 0;

    for (size_t i : idx) {
      const auto& p = payloads[i];
      const double w = WeightAt(weights, i);
      if (w == 0.0) continue;
      auto grounded = GroundAffordanceTarget(p, target_tree);
      if (!grounded.ok()) {
        had_grounding_error = true;
        continue;
      }

      pos_sum += w * grounded->position.head<2>();
      z_sum += w * grounded->position.z();
      sin_sum += w * std::sin(grounded->yaw);
      cos_sum += w * std::cos(grounded->yaw);
      arrival_sum += p.estimated_time;
      weight_sum += w;
      ++used;
      if (w > dominant_weight) {
        dominant_weight = w;
        dominant_referent = p.referent_shared_id;
      }
    }
    if (used == 0 || weight_sum <= 1e-9) continue;

    RealizationTarget target;
    target.anchor_name = anchor_name;
    target.position = pos_sum / weight_sum;
    target.z = z_sum / weight_sum;
    target.has_z = true;
    target.yaw = std::atan2(sin_sum, cos_sum);
    target.kernel_type = RelationKernelType::kAvatarAffordancePose;
    target.referent_id = dominant_referent;
    target.weight = weight_sum;
    target.arrival_time = arrival_sum / static_cast<double>(used);
    result.targets.push_back(std::move(target));
  }

  if (had_grounding_error && result.status == SolverStatus::kConverged) {
    result.status = SolverStatus::kFallback;
  }

  return result;
}

// Step-synchronized progressive solve for controlled avatars (upstream). Each
// position relation becomes a next-step target by reducing its distance by
// d_current / estimated_steps, blended with the annealed weights. Not on the
// live realization path (space.cc realizes via Solve); retained for callers /
// tests that want per-step pacing inside the solver.
SolverResult RelationSolver::SolveProgressive(
    const Eigen::Vector2d& current_position, double current_yaw,
    const std::vector<RelationPayload>& payloads,
    const std::vector<double>& weights, const ElementTree& target_tree,
    const NavigableSpatialDefinition* navigable_def,
    const ReferentResolver* resolver) const {
  return SolveImpl(current_position, current_yaw, payloads, weights,
                   target_tree, navigable_def, /*use_step_progress=*/true,
                   resolver, config_);
}

RealizationResult RelationSolver::SolveTargets(
    const Eigen::Vector2d& current_position, double current_yaw,
    const std::vector<RelationPayload>& payloads,
    const std::vector<double>& weights, const ElementTree& target_tree,
    double step_duration, const NavigableSpatialDefinition* navigable_def,
    double bucket_size) const {
  RealizationResult result;

  if (HasInvalidWeight(weights)) {
    result.status = SolverStatus::kFailed;
    return result;
  }

  if (payloads.empty()) {
    result.status = SolverStatus::kConverged;
    return result;
  }

  // Group payload indices by bucket key = floor(arrival_time / bucket_size).
  std::map<int, std::vector<size_t>> buckets;
  for (size_t i = 0; i < payloads.size(); ++i) {
    double arrival_time = payloads[i].estimated_time;
    int key = static_cast<int>(std::floor(arrival_time / bucket_size));
    buckets[key].push_back(i);
  }

  SolverStatus overall_status = SolverStatus::kConverged;

  for (const auto& [key, indices] : buckets) {
    // Build sub-payloads and sub-weights for this bucket.
    std::vector<RelationPayload> sub_payloads;
    std::vector<double> sub_weights;
    sub_payloads.reserve(indices.size());
    sub_weights.reserve(indices.size());

    double arrival_time_sum = 0.0;
    double weight_sum = 0.0;

    for (size_t idx : indices) {
      const double w = WeightAt(weights, idx);
      if (w == 0.0) continue;
      sub_payloads.push_back(payloads[idx]);
      sub_weights.push_back(w);
      arrival_time_sum += payloads[idx].estimated_time;
      weight_sum += w;
    }

    if (sub_payloads.empty()) continue;

    SolverResult solver_result =
        Solve(current_position, current_yaw, sub_payloads, sub_weights,
              target_tree, navigable_def);

    if (solver_result.status != SolverStatus::kConverged) {
      overall_status = solver_result.status;
    }

    // Use first payload's metadata for the target descriptor.
    const RelationPayload& first = sub_payloads[0];

    RealizationTarget target;
    target.arrival_time =
        arrival_time_sum / static_cast<double>(sub_payloads.size());
    target.anchor_name = first.subject_anchor_name;
    target.position = solver_result.position;
    target.yaw = solver_result.yaw;
    target.kernel_type = first.kernel_type;
    target.referent_id = first.referent_shared_id;
    target.weight = weight_sum;

    result.targets.push_back(target);
  }

  result.status = overall_status;
  return result;
}

// The legacy Ceres cost functor that lived here under `#if 0` is replaced by
// the live target-side evaluator EvaluateTargetRelation
// (space/relation_evaluator.h).


double AffordanceSupportDistance(const RelationPayload& payload) {
  if (payload.kernel_type != RelationKernelType::kAvatarAffordancePose)
    return std::numeric_limits<double>::infinity();
  const double metric = payload.grounded_relation.support_distance_m;
  if (std::isfinite(metric) && metric >= 0.0) return metric;
  // Legacy payload (no metric distance): exact inverse of R = 1/(1+d).
  const double r = payload.grounded_relation.proximity;
  if (!std::isfinite(r) || r <= 0.0)
    return std::numeric_limits<double>::infinity();
  return std::max(0.0, 1.0 / std::min(r, 1.0) - 1.0);
}

double BlendYaw(double from, double to, double alpha) {
  alpha = std::clamp(alpha, 0.0, 1.0);
  return std::remainder(from + alpha * std::remainder(to - from, 2.0 * M_PI),
                        2.0 * M_PI);
}

std::optional<double> ReferentRelativeHeading(
    const std::vector<RelationPayload>& payloads,
    const std::vector<double>& weights, const ElementTree& target_tree,
    const ReferentResolver* resolver) {
  ReferentResolver default_resolver;
  const ReferentResolver& resolve = resolver ? *resolver : default_resolver;
  // The DRIVING Spacing relation (highest policy weight; declaration order
  // on a tie) supplies the heading. Averaging several referents is wrong when
  // the receiver re-oriented them differently: near-opposite headings would
  // cancel into an arbitrary direction.
  std::optional<double> heading;
  double best_weight = 0.0;
  for (size_t i = 0; i < payloads.size(); ++i) {
    const auto& p = payloads[i];
    if (p.kernel_type != RelationKernelType::kAvatarSpacing) continue;
    const double w = WeightAt(weights, i);
    if (!std::isfinite(w) || w <= best_weight) continue;
    // AvatarSpacingKernel::Encode: [x, y, rel_yaw] or [x, y, z, rel_yaw, ..].
    const auto& es = p.grounded_relation.encoded_state;
    const int yaw_index = es.size() == 3 ? 2 : es.size() == 6 ? 3 : -1;
    if (yaw_index < 0 || !std::isfinite(es(yaw_index))) continue;
    const Element* referent =
        p.resolved_local_element_id != kInvalidElementId
            ? target_tree.GetElement(p.resolved_local_element_id)
            : resolve.Resolve(p.referent_shared_id, target_tree);
    if (!referent) continue;
    const Eigen::Matrix4d world = target_tree.GetWorldTransform(referent->GetId());
    heading = std::remainder(std::atan2(world(1, 0), world(0, 0)) + es(yaw_index),
                             2.0 * M_PI);
    best_weight = w;
  }
  return heading;
}

absl::StatusOr<Eigen::Vector2d> DecodeMappedSupportPoint(
    const RelationPayload& payload, const Element& target_referent,
    const ElementTree& target_tree) {
  return DecodeSupportPoint(payload, target_referent, target_tree);
}

Eigen::Vector2d ClampToReach(const Eigen::Vector2d& point,
                             const Eigen::Vector2d& center, double radius) {
  Eigen::Vector2d offset = point - center;
  double dist = offset.norm();
  if (radius <= 0.0 || dist <= radius || dist <= 1e-12) {
    return point;
  }
  return center + offset * (radius / dist);
}

}  // namespace spatialblend
