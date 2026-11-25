// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/avatar_spacing_kernel.h"

#include <algorithm>
#include <cmath>

#include "spatialblend/model/element/navigable_spatial_definition.h"

namespace spatialblend {

GroundedRelation AvatarSpacingKernel::Encode(
    const Eigen::Vector3d& anchor_world_position, double anchor_world_yaw,
    const Support& support) const {
  // Relative position in support frame (for solver comparison)
  const Eigen::Matrix4d inv = support.has_world_transform_inverse
                                  ? support.world_transform_inverse
                                  : Eigen::Matrix4d(support.world_transform.inverse());
  Eigen::Vector4d local =
      inv * Eigen::Vector4d(anchor_world_position.x(),
                            anchor_world_position.y(),
                            anchor_world_position.z(), 1.0);

  double support_yaw = support.GetYaw();
  double relative_yaw = std::remainder(anchor_world_yaw - support_yaw,
                                       2 * M_PI);

  // Distance for proximity = GEODESIC distance through the navigable scope to the
  // nearest point on the referent's EDGE (contact_point_world). This matches v1
  // (geodesic-to-closest-edge): path-aware reachability, so distance/proximity
  // fall as the actor nears an object's boundary and rise when obstacles are in
  // the way — instead of a Euclidean-to-contact value that stays large mid-room.
  double distance;
  if (support.has_contact && support.navigable_def && !support.fast_distance) {
    // An unreachable referent is OUT OF SCOPE (scope_gate.h,
    // IsReferentReachable) and this kernel adds no Euclidean fallback of its
    // own: a +inf geodesic scores R = 0. NOTE: until the geometry-mediation
    // rebase, NavigableGeometryContext::GetGeodesicDistance itself still
    // falls back to Euclidean when no path is found; that branch makes it
    // return +inf for unreachable targets.
    // The grounding rule already walked the per-referent ring field to this
    // contact. A second point field would repeat that plan for every Jacobian
    // sample and every line-search trial.
    distance = support.geodesic_length_m >= 0.0
                   ? support.geodesic_length_m
                   : support.navigable_def->GetGeodesicDistance(
                         anchor_world_position, support.contact_point_world);
  } else if (support.has_contact) {
    // Performance: support.fast_distance takes this branch
    // even when a navigable_def IS available -- see Support::fast_distance.
    // Straight-line to the same contact point the geodesic branch above
    // targets, just without the visibility-graph pathfind. Never taken for
    // the value that decides search acceptance (relation_solver.cc always
    // re-evaluates the winning candidate with fast_distance=false first).
    // Also the genuine no-navigable-scope case: Euclidean to the nearest
    // edge point either way.
    Eigen::Vector2d anchor_local(local.x(), local.y());
    // An anchor exactly ON the boundary is at distance 0 (R = 1), as the
    // geodesic branch above reports; it is not "at the contact point, so
    // measure to the centre". The old centre fallback made the fast value
    // jump from ~0 to the half-extent for an anchor within 1 nm of the edge,
    // which put the paper-objective search's seed (a decode that lands on the edge) tens
    // of cm off R and made its answer jump between ticks.
    distance = (anchor_local - support.contact_point_local).norm();
    // Metric scale: the
    // local-frame difference above is in the referent's own units, but the
    // geodesic branch (and so the exact acceptance value) measures METRES in
    // the world frame. A referent whose world transform carries scale made
    // fast != exact by that factor. When this is the search's fast proxy next
    // to a navigable scope, measure the same Euclidean segment in the world
    // frame (planar, like the geodesic and ReferentBoundaryDistance): exact
    // for any scale, anisotropic included, so fast == exact wherever the
    // geodesic is the straight segment. (With no scope the exact value is
    // this same local-frame distance, so it stays unscaled to agree with it.)
    if (support.fast_distance && support.navigable_def) {
      distance = (anchor_world_position.head<2>() -
                  support.contact_point_world.head<2>())
                     .norm();
    }
  } else if (support.navigable_def && !support.fast_distance) {
    // No contact point: geodesic to the referent center.
    distance = support.navigable_def->GetGeodesicDistance(
        anchor_world_position, support.GetPosition3D());
  } else if (support.navigable_def) {
    // fast_distance: straight-line to the referent center.
    Eigen::Vector3d center = support.GetPosition3D();
    distance = (anchor_world_position - center).norm();
  } else {
    // Euclidean fallback.
    distance = std::sqrt(local.x() * local.x() + local.y() * local.y());
  }

  // Check if actor is inside the referent's boundary (leaf grounding).
  // support.has_interior is set by SpacingGroundingRule::ComputeSupport,
  // which already performs this containment check in the referent's correct
  // boundary frame (world for NavigableSpatialDefinition, the surface's own
  // local frame for SurfaceSpatialDefinition, and the element's local frame
  // for PolytopeSpatialDefinition — see
  // SpatialDefinition::RequiresElementWorldTransform). Recomputing it here
  // against the raw WORLD anchor_world_position would reproduce that same
  // frame bug for any referent whose boundary is not already world-frame
  // (e.g. a box/Polytope referent), so reuse the grounding rule's result
  // instead of re-deriving it from possibly-mismatched frames.
  bool inside_boundary = support.has_interior;

  // R = 1 - clip(distance / scope_diameter): the approach level is normalized by
  // the extent of the element the actor is CONTAINED IN (its active scope) —
  // support.navigable_def is the deepest containing navigable (the room when the
  // actor is in the room, an object's region when inside one). The decoder
  // denormalizes by the RECEIVER avatar's containing scope (see
  // RelationSolver::DecodeTargetPoint), so under identical geometry the
  // stand-off round-trips. (Known asymmetry: if the sender is inside an object
  // but the receiver avatar is outside, the two scopes differ — the user flagged
  // this; not handled here.)
  double proximity;
  double scope_diameter = 0.0;
  // Outside a container the body is outside this scope. The border relation
  // is the value at the body's projection onto the border: depth 0, R = 1.
  // R does not continue past 1.
  const bool border_projection = support.container_self && !inside_boundary;
  if (inside_boundary && support.container_self && support.navigable_def &&
      std::isfinite(distance)) {
    Bound2D b = support.navigable_def->GetBounds();
    scope_diameter = (b.max - b.min).norm();
    proximity = (scope_diameter > 1e-6)
                    ? 1.0 - std::clamp(distance / scope_diameter, 0.0, 1.0)
                    : 1.0 / (1.0 + distance);
  } else if (inside_boundary || border_projection) {
    proximity = 1.0;
  } else if (support.navigable_def) {
    Bound2D b = support.navigable_def->GetBounds();
    scope_diameter = (b.max - b.min).norm();
    proximity = (scope_diameter > 1e-6)
                    ? 1.0 - std::clamp(distance / scope_diameter, 0.0, 1.0)
                    : 1.0 / (1.0 + distance);
  } else {
    proximity = 1.0 / (1.0 + distance);
  }

  // Scale-invariant engagement distance in metres (Hall's proxemics): the
  // ABSOLUTE boundary distance, independent of scope extent. 0 when inside.
  // Used by the realizer's commitment weighting AND as the exact
  // metric standoff the approach-branch decode denormalizes, paired with
  // source_scope_diameter_m below.
  if (!inside_boundary && !std::isfinite(distance)) proximity = 0.0;
  const bool depth_inside =
      inside_boundary && support.container_self && support.navigable_def;
  // On the border projection the depth is 0, the same value as a pose that
  // is already on the border. The outside distance is not a depth past 1.
  double support_distance_m =
      (border_projection || (inside_boundary && !depth_inside))
          ? 0.0
          : (std::isfinite(distance) ? distance : -1.0);
  // The scope diameter proximity was actually normalized by (0 = unset: no
  // navigable scope at encode time, or inside/engaged where distance-based
  // decode does not apply). See GroundedRelation::source_scope_diameter_m.
  double source_scope_diameter_m =
      ((inside_boundary && !depth_inside) || scope_diameter <= 1e-6)
          ? -1.0
          : scope_diameter;

  if (Is3DPosition()) {
    double relative_pitch = 0.0;
    double relative_roll = 0.0;

    GroundedRelation gr;
    gr.encoded_state = Eigen::VectorXd(6);
    gr.encoded_state << local.x(), local.y(), local.z(),
                        relative_yaw, relative_pitch, relative_roll;
    gr.proximity = proximity;
    gr.support_distance_m = support_distance_m;
    gr.contact_distance_m = std::isfinite(distance) ? distance : -1.0;
    gr.source_scope_diameter_m = source_scope_diameter_m;
    return gr;
  }

  GroundedRelation gr;
  gr.encoded_state = Eigen::VectorXd(3);
  gr.encoded_state << local.x(), local.y(), relative_yaw;
  gr.proximity = proximity;
  gr.support_distance_m = support_distance_m;
  gr.contact_distance_m = std::isfinite(distance) ? distance : -1.0;
  gr.source_scope_diameter_m = source_scope_diameter_m;
  return gr;
}

double AvatarSpacingKernel::Compare(const GroundedRelation& source,
                                    const GroundedRelation& target) const {
  return source.DistanceTo(target);
}

SatisfactionMetric AvatarSpacingKernel::ComputeSatisfactionMetric(
    const GroundedRelation& current,
    const GroundedRelation* previous,
    const ActorMotionState& motion) const {
  int pos_dim = Is3DPosition() ? 3 : 2;
  double d_current = current.encoded_state.head(pos_dim).norm();

  // Trust directional progress only when sustainedly moving (EMA-smoothed linear
  // speed above floor); otherwise a still/jittering source emits a thrashing
  // estimated_time that oscillates the clone. See EstimateTimeToSatisfaction.
  bool has_prev = previous && previous->encoded_state.size() >= pos_dim;
  double d_previous =
      has_prev ? previous->encoded_state.head(pos_dim).norm() : 0.0;
  double t = EstimateTimeToSatisfaction(d_current, d_previous, has_prev,
                                        motion.linear_speed);
  return {d_current, t};
}

}  // namespace spatialblend
