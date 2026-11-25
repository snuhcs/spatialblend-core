// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_RELATION_GROUNDED_RELATION_H_
#define SPATIALBLEND_RELATION_GROUNDED_RELATION_H_

#include <Eigen/Dense>
#include <limits>

namespace spatialblend {

struct GroundedRelation {
  Eigen::VectorXd encoded_state;

  // Normalized proximity: 0 = far/disengaged, 1 = fully engaged.
  // Kernel-defined. Extent-RELATIVE: normalized by the actor's containing scope,
  // so it transfers across differently sized spaces (position/standoff decode
  // depends on this). Used by the solver to weight this relation.
  double proximity = 0.0;

  // Absolute boundary distance in metres (0 when the actor is inside/engaged).
  // < 0 means "unset". Used by the realizer's commitment weighting AND
  // as the exact metric standoff for approach-branch decode: see
  // RelationSolver::DecodeTargetPoint and source_scope_diameter_m below.
  double support_distance_m = -1.0;

  // Spacing: the metric geodesic length from the anchor to the relation's contact point,
  // whether or not the anchor is inside the referent (support_distance_m is 0 there).
  // It is the length of the course to the contact, a continuous function of the anchor's
  // position; the pursuit measures it. < 0 means "unset" or unreachable.
  double contact_distance_m = -1.0;

  // The diameter (bounding-box diagonal) of the SOURCE actor's containing
  // navigable scope used to normalize `proximity` at encode time (see
  // AvatarSpacingKernel::Encode) -- the same scope support_distance_m was
  // measured against. Paired with support_distance_m, this lets the decoder
  // (RelationSolver::DecodeTargetPoint) reconstruct the target standoff as
  // support_distance_m * (target_scope_diameter / source_scope_diameter_m):
  // the exact metric distance when the target scope is the SAME size as the
  // source (identity/same-scale rooms -- reduces to an exact round trip) and
  // otherwise the R-preserving (Eq 8) normalized-distance rule, matching the
  // proximity-based reconstruction it supersedes. <= 0 means "unset" (legacy
  // payload, or no containing navigable scope at encode time); the decoder
  // falls back to reconstructing distance from `proximity` alone.
  double source_scope_diameter_m = -1.0;

  // AffordancePose only: the anchor's pose relative to the
  // support, beyond the [angle, radial_ratio] support point already in
  // encoded_state -- its in-plane offset from that support point (in the
  // support's own local frame, u/v) and its height above the support plane
  // (also support-local). NaN = unknown: a pre-pose-fit sender, or any other
  // relation family. All three finite = AffordancePose R formula version 1
  // (AffordancePoseRFormulaVersion, avatar_affordance_pose_kernel.h);
  // `proximity` stays the geometric 1/(1+distance) component either way, and
  // the pose-fit R is evaluated by Eval_p. Appended field (the existing
  // pattern of support_distance_m / source_scope_diameter_m): a pre-pose-fit receiver simply
  // never reads these.
  double ap_offset_u_m = std::numeric_limits<double>::quiet_NaN();
  double ap_offset_v_m = std::numeric_limits<double>::quiet_NaN();
  double ap_height_m = std::numeric_limits<double>::quiet_NaN();

  double DistanceTo(const GroundedRelation& other) const {
    if (encoded_state.size() != other.encoded_state.size()) {
      return std::numeric_limits<double>::infinity();
    }
    return (encoded_state - other.encoded_state).norm();
  }
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_RELATION_GROUNDED_RELATION_H_
