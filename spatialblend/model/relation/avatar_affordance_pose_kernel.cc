// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/avatar_affordance_pose_kernel.h"

#include <algorithm>
#include <cmath>

#include "spatialblend/geometry/polygon_coordinate_mapper.h"

namespace spatialblend {

namespace {
constexpr double kApWeightSumError =
    kApPositionWeight + kApHeightWeight + kApYawWeight - 1.0;
static_assert(kApWeightSumError > -1e-12 && kApWeightSumError < 1e-12,
             "AffordancePose fit weights must sum to 1 (R = 1 iff every "
             "fit term = 1)");
}  // namespace

int AffordancePoseRFormulaVersion(const GroundedRelation& gr) {
  return std::isfinite(gr.ap_offset_u_m) && std::isfinite(gr.ap_offset_v_m) &&
                 std::isfinite(gr.ap_height_m)
             ? kApRFormulaPoseFitV1
             : kApRFormulaProximityOnly;
}

// R as a declared monotone combination of the anchor's position,
// height and yaw fit relative to the support, each in [0, 1] with 1 = exact:
//   position (in-plane offset from the support point): 0.5 -- the primary
//     geometric fit ("on this patch, not elsewhere on it").
//   height (above/below the support plane): 0.3 -- resting on the surface vs
//     hovering above it.
//   yaw (relative to the SOURCE's relative yaw): 0.2 -- a real but secondary
//     refinement. Scored against the source's relative yaw, not the patch's
//     own frame: the patch frame's yaw is arbitrary, the source's is the
//     condition being preserved. Unknown => the term is dropped.
// The paper (SpatialBlend section 5) leaves the formula open but requires full
// satisfaction to mean the POSE, not just proximity.
double AffordancePoseR(double offset_u_m, double offset_v_m, double height_m,
                       std::optional<double> yaw_error_rad) {
  const double position_fit = 1.0 / (1.0 + std::hypot(offset_u_m, offset_v_m));
  const double height_fit = 1.0 / (1.0 + std::abs(height_m));
  if (!yaw_error_rad.has_value() || !std::isfinite(*yaw_error_rad)) {
    return (kApPositionWeight * position_fit + kApHeightWeight * height_fit) /
           (kApPositionWeight + kApHeightWeight);
  }
  const double yaw_fit = 0.5 * (1.0 + std::cos(*yaw_error_rad));
  return kApPositionWeight * position_fit + kApHeightWeight * height_fit +
         kApYawWeight * yaw_fit;
}

GroundedRelation AvatarAffordancePoseKernel::Encode(
    const Eigen::Vector3d& anchor_world_position, double anchor_world_yaw,
    const Support& support) const {
  // Transform anchor into support-local frame
  Eigen::Matrix4d inv = support.world_transform.inverse();
  Eigen::Vector4d local =
      inv * Eigen::Vector4d(anchor_world_position.x(),
                            anchor_world_position.y(),
                            anchor_world_position.z(), 1.0);
  Eigen::Vector2d local_2d(local.x(), local.y());

  // Relative yaw
  double support_yaw = support.GetYaw();
  double relative_yaw =
      std::remainder(anchor_world_yaw - support_yaw, 2 * M_PI);

  // If GroundingRule already computed a normalized contact, use it directly
  if (support.has_contact && support.boundary) {
    // Distance from the anchor to the referent's support geometry, in world
    // frame. NOTE: this is deliberately NOT (local_2d - contact_point_local)
    // — both of those are the anchor's own position re-expressed in the
    // support frame (see AffordancePoseGroundingRule::ComputeSupport), so
    // that difference is always ~0 by construction and previously pinned
    // proximity to 1.0 regardless of anchor position.
    // contact_point_world is the actual
    // projection of the anchor onto the referent's geometry (e.g. straight
    // down onto a tabletop plane, or onto a navigable floor boundary), so
    // this distance genuinely shrinks as the anchor approaches the support.
    double anchor_dist =
        (anchor_world_position - support.contact_point_world).norm();

    GroundedRelation gr;
    if (Is3DRotation()) {
      double relative_pitch = 0.0;
      double relative_roll = 0.0;
      gr.encoded_state = Eigen::VectorXd(5);
      gr.encoded_state << support.contact_normalized.angle,
                          support.contact_normalized.radial_ratio,
                          relative_yaw, relative_pitch, relative_roll;
    } else {
      gr.encoded_state = Eigen::VectorXd(3);
      gr.encoded_state << support.contact_normalized.angle,
                          support.contact_normalized.radial_ratio,
                          relative_yaw;
    }
    // The anchor's pose relative to the support -- its in-plane
    // offset from the support point and its height above the support
    // plane -- decomposing what anchor_dist above mixes into one scalar.
    // Measured relative to the CONTACT point's own local coordinates, not
    // the support frame's origin: the whole-boundary grounding path (no
    // named sub-region) places that origin at the referent's own local
    // origin, which need not lie on the surface (e.g. a box's bottom, not
    // its top face) -- only the named-sub-region path's local_transform
    // (Pose(0, 0, top, 0)) puts the origin AT the surface. Re-deriving the
    // contact point's local Z here keeps height correct in both cases.
    const Eigen::Vector4d contact_local =
        inv * Eigen::Vector4d(support.contact_point_world.x(),
                              support.contact_point_world.y(),
                              support.contact_point_world.z(), 1.0);
    // Over the part's footprint the nearest support point IS the anchor's own
    // projection: the in-plane offset is 0 by definition (decided by the grounding
    // rule where the part's frame is exact), not a difference of two round-tripped
    // coordinates. On a named part the plane is the part frame's z = 0, so the
    // height is the anchor's own local z.
    if (support.anchor_over_footprint) {
      gr.ap_offset_u_m = 0.0;
      gr.ap_offset_v_m = 0.0;
    } else {
      gr.ap_offset_u_m = local_2d.x() - contact_local.x();
      gr.ap_offset_v_m = local_2d.y() - contact_local.y();
    }
    gr.ap_height_m = support.sub_region_name.empty()
                         ? local.z() - contact_local.z()
                         : local.z();
    // proximity stays the GEOMETRIC component (1 / (1 + anchor_dist)): the
    // contact-phase hysteresis and weights read it, exactly as before the pose fit.
    // The pose-fit R is NOT stored; it is AffordancePoseR over the offsets
    // above plus a yaw reference (see the header), evaluated by Eval_p.
    gr.proximity = 1.0 / (1.0 + anchor_dist);
    // The metric anchor-to-support distance (trace payload.dist, relation
    // evidence); engagement is contact, not a threshold on it. An anchor over
    // the part's footprint and on or below its plane is IN the support, the
    // same fact as a Spacing source inside its referent: its distance is 0
    // (the count reads the relation's distance, and
    // a pelvis sinking into the seat is contact, not receding).
    gr.support_distance_m =
        (support.anchor_over_footprint && gr.ap_height_m <= 0.0) ? 0.0 : anchor_dist;
    return gr;
  }

  // If boundary available, encode as normalized polygon coordinates
  if (support.boundary) {
    Eigen::Vector2d forward(0, 1);  // Y-forward in local frame
    auto normalized = PolygonCoordinateMapper::Encode(
        local_2d, *support.boundary, forward);
    if (normalized.ok()) {
      // Distance from bound anchor to support center (contact proximity)
      double anchor_dist = local_2d.norm();

      GroundedRelation gr;
      if (Is3DRotation()) {
        double relative_pitch = 0.0;
        double relative_roll = 0.0;
        gr.encoded_state = Eigen::VectorXd(5);
        gr.encoded_state << normalized->angle, normalized->radial_ratio,
                            relative_yaw, relative_pitch, relative_roll;
      } else {
        gr.encoded_state = Eigen::VectorXd(3);
        gr.encoded_state << normalized->angle, normalized->radial_ratio,
                            relative_yaw;
      }
      // Proximity: how close the bound anchor is to the affordance contact point
      gr.proximity = 1.0 / (1.0 + anchor_dist);
      gr.support_distance_m = anchor_dist;  // metric distance to the support
      return gr;
    }
  }

  // Fallback: raw local coords (no boundary)
  double anchor_dist = local_2d.norm();

  GroundedRelation gr;
  if (Is3DRotation()) {
    double relative_pitch = 0.0;
    double relative_roll = 0.0;
    gr.encoded_state = Eigen::VectorXd(5);
    gr.encoded_state << local_2d.x(), local_2d.y(), relative_yaw,
                        relative_pitch, relative_roll;
  } else {
    gr.encoded_state = Eigen::VectorXd(3);
    gr.encoded_state << local_2d.x(), local_2d.y(), relative_yaw;
  }
  gr.proximity = 1.0 / (1.0 + anchor_dist);
  gr.support_distance_m = anchor_dist;  // metric distance to the support
  return gr;
}

double AvatarAffordancePoseKernel::Compare(
    const GroundedRelation& source, const GroundedRelation& target) const {
  return source.DistanceTo(target);
}

SatisfactionMetric AvatarAffordancePoseKernel::ComputeSatisfactionMetric(
    const GroundedRelation& current,
    const GroundedRelation* previous,
    const ActorMotionState& motion) const {
  double d_current = current.encoded_state.norm();

  // Trust directional progress only when sustainedly moving (EMA-smoothed linear
  // + angular speed above floor); otherwise a still/jittering source emits a
  // thrashing estimated_time that oscillates the clone. See
  // EstimateTimeToSatisfaction.
  bool has_prev =
      previous && previous->encoded_state.size() == current.encoded_state.size();
  double d_previous = has_prev ? previous->encoded_state.norm() : 0.0;
  double t = EstimateTimeToSatisfaction(d_current, d_previous, has_prev,
                                        motion.linear_speed +
                                            motion.angular_speed);
  return {d_current, t};
}

}  // namespace spatialblend
