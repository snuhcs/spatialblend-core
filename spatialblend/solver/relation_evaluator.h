// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/relation_evaluator.h
//
// Target-side relation evaluation Eval_p (the SpatialBlend paper, Section 5):
//
//     R^tgt_p(t; S) = Eval_p(A^tgt(S), D^tgt_{r_p})
//
// For a received relation payload and a CANDIDATE receiver state S (the pose
// of the payload's subject anchor), ground the relation on the corresponding
// receiver referent and evaluate it with the SAME kernel and grounding rule
// the source used in Extract_p (KernelFactory), so R^tgt is the code
// definition of R applied on the receiver. This is what the realizer's ideal
// (eq. ideal: R^tgt_p(S*) = R^src_p for every active p) is measured against,
// and what RE@E / MRE are computed from.
//
// Normalizing-context rule (ONE rule with the decode that chooses S). The
// source normalizes R by the navigable scope containing the subject anchor
// and names it in the payload (scope_shared_id / scope_source_* handle). The
// target uses `options.normalizing_scope` when given, else:
//   * Spacing: ResolveSpacingNormalization (relation_solver.h) -- the
//     receiver counterpart of the source scope (CorrespondingScope: the
//     receiver-resolved counterpart, else the structural correspondent, the
//     navigable enclosing the referent's parent) when the source was scoped;
//     no scope when the source had none (R = 1/(1+d) on both sides); the
//     structural correspondent for a legacy payload that carries neither the
//     scope diameter nor the metric distance;
//   * Facing / AffordancePose: CorrespondingScope (their R does not depend on
//     it; it is only handed to the grounding rule as the source did).
// DecodeTargetPoint inverts R in exactly these modes, so a realizer output
// evaluates back to R^src.
//
// A referent that cannot be grounded is an explicit disposition (no
// counterpart, no support geometry, part not found); nothing is evaluated
// against a referent origin standing in for a missing support.
#ifndef SPATIALBLEND_SPACE_RELATION_EVALUATOR_H_
#define SPATIALBLEND_SPACE_RELATION_EVALUATOR_H_

#include <Eigen/Dense>
#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "spatialblend/base/common/types.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/relation/grounded_relation.h"
#include "spatialblend/model/relation/kernel_factory.h"
#include "spatialblend/model/relation/support.h"
#include "spatialblend/solver/referent_resolver.h"
#include "spatialblend/model/realization/remote_update.h"

namespace spatialblend {

class NavigableSpatialDefinition;

enum class TargetEvaluationDisposition {
  kEvaluated,
  // No receiver counterpart for the payload's referent.
  kUnmapped,
  // The counterpart has no support geometry but the source relation was
  // grounded on geometry (always the case for AffordancePose).
  kNoSupportGeometry,
  // The source support was a named part the counterpart does not have.
  kPartNotFound,
  // The counterpart's geometry cannot carry this family's support (e.g. an
  // unannotated volume for AffordancePose, an invalid chart).
  kUnsupportedSupport,
  // Malformed payload or candidate state (non-finite pose, missing anchor).
  kInvalidInput,
};

const char* TargetEvaluationDispositionName(TargetEvaluationDisposition d);

// Which support R^tgt was scored against. Never silent: a fallback is a
// distinct, exposed value (and carries a `reason`), not a quiet substitution.
enum class TargetSupportBasis {
  // The family's own grounding rule at the candidate (Spacing, Facing, and an
  // AffordancePose payload with no named part). The corresponding-support rule does not apply.
  kFamilyGrounding = 0,
  // AffordancePose with a named part: the CORRESPONDING target support, the
  // RadialSurfaceChart map M of the source support point, fixed independent of
  // the candidate's position. The only basis that implements Eval_p.
  kMappedSupport = 1,
  // AffordancePose with a named part, but the mapped support could not be
  // decoded (e.g. a legacy payload with no chart): R^tgt was scored against
  // the support the CANDIDATE itself projects onto -- the earlier behavior,
  // which scores a hand at the wrong end of the patch as R = 1. The score is
  // NOT a faithful Eval_p; callers should treat it as degraded.
  kCandidateProjectionFallback = 2,
};

// Pose of one subject anchor in the candidate receiver state S (world frame,
// Z-up, yaw about +Z in radians).
struct TargetAnchorPose {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  double yaw = 0.0;
};

struct TargetEvaluationOptions {
  // Explicit normalizing scope element (must carry a
  // NavigableSpatialDefinition); overrides the shared rule above.
  element_id_t normalizing_scope = kInvalidElementId;
  // Referent resolution after the payload's linker-resolved local id
  // (default: exact shared-id match).
  const ReferentResolver* resolver = nullptr;
};

struct TargetRelationEvaluation {
  TargetEvaluationDisposition disposition =
      TargetEvaluationDisposition::kInvalidInput;
  std::string reason;
  RelationKernelType kernel_type = RelationKernelType::kAvatarFacing;
  std::string anchor_name;
  shared_element_id_t referent_id = kInvalidSharedElementId;
  element_id_t local_referent_id = kInvalidElementId;
  // Normalizing context actually used (kInvalidElementId / -1 when none).
  element_id_t normalizing_scope_id = kInvalidElementId;
  double normalizing_scope_diameter_m = -1.0;
  // Support grounded on the receiver referent used to score R^tgt (same
  // family grounding rule as the source; AffordancePose is restricted to the
  // source's part). For AffordancePose with a named part, this is the
  // CORRESPONDING target support -- the RadialSurfaceChart map M of the
  // source support point -- not wherever the candidate happens to project.
  Support support;
  // Which support `support` is. kCandidateProjectionFallback means the
  // mapped support was unavailable; `reason` then says why even though the
  // disposition is kEvaluated.
  TargetSupportBasis support_basis = TargetSupportBasis::kFamilyGrounding;
  // Diagnostic only, never used for r_tgt/error: where the candidate's
  // own anchor projects onto the same named part (the earlier behavior).
  // Lets a caller see how far the candidate is from ITS OWN projection vs.
  // the fixed corresponding support in `support` above.
  bool has_candidate_projection = false;
  Support candidate_projected_support;
  // Kernel encoding at S. relation.proximity is the geometric proximity
  // component; R^tgt below is the relation-preservation score.
  GroundedRelation relation;
  // The R formula both r_src and r_tgt were scored under
  // (kApRFormulaProximityOnly / kApRFormulaPoseFitV1,
  // avatar_affordance_pose_kernel.h). AffordancePose with a payload that
  // carries offset/height (and a target that can encode them) is scored with
  // the pose fit R (position, height, yaw relative to the SOURCE's relative
  // yaw); everything else -- other families, a pre-pose-fit sender -- uses the
  // proximity component on BOTH sides. Never mixed.
  int r_formula_version = 0;
  double r_tgt = 0.0;
  double r_src = 0.0;
  // r_tgt - r_src.
  double error = 0.0;
  // AffordancePose only: the candidate anchor TOUCHES the receiver's
  // counterpart surface (engagement.h IsAffordanceContact against the support
  // the candidate itself projects onto). The arrival of a seat.
  bool anchor_touches_support = false;
  // The source support decoded onto the receiver referent (the point the
  // realizer targets), when decodable. World frame; z is meaningful only for
  // AffordancePose.
  bool has_mapped_support = false;
  Eigen::Vector3d mapped_support = Eigen::Vector3d::Zero();

  bool ok() const {
    return disposition == TargetEvaluationDisposition::kEvaluated;
  }
};

// Eval_p for one payload at one candidate anchor pose.
TargetRelationEvaluation EvaluateTargetRelation(
    const RelationPayload& payload, const TargetAnchorPose& anchor,
    const ElementTree& receiver_tree,
    const TargetEvaluationOptions& options = {});

// The pose-INDEPENDENT parts
// of Eval_p for one payload -- referent resolution, KernelFactory
// construction, normalizing-scope resolution, R^src, and the mapped source
// support (DecodeMappedSupportPoint / GroundAffordanceTarget, a geodesic /
// chart computation that never depends on the candidate pose). A bounded
// local search evaluates the same relation at O(iterations x candidates)
// poses; redoing all of this on every candidate was the dominant per-tick
// cost (3-9ms, p95 16-18ms, over a 2.5ms budget).
// Callers that only need R^tgt in a hot loop should call
// EvaluatePreparedTargetRelation with this prepared once per relation,
// per search, rather than calling EvaluateTargetRelation per candidate.
struct PreparedTargetRelation {
  TargetEvaluationDisposition disposition =
      TargetEvaluationDisposition::kInvalidInput;
  std::string reason;
  RelationKernelType kernel_type = RelationKernelType::kAvatarFacing;
  std::string anchor_name;
  shared_element_id_t referent_id = kInvalidSharedElementId;
  element_id_t local_referent_id = kInvalidElementId;
  element_id_t normalizing_scope_id = kInvalidElementId;
  double normalizing_scope_diameter_m = -1.0;
  double r_src = 0.0;
  bool has_mapped_support = false;
  Eigen::Vector3d mapped_support = Eigen::Vector3d::Zero();

  // The CORRESPONDING target support of an AffordancePose relation with
  // a named part -- the RadialSurfaceChart map M of the source support
  // point, decoded against the target part. It depends only on the payload,
  // the referent and the tree, never on the candidate, so it is resolved
  // once here (`corresponding_status` records why it could not be, so the
  // per-candidate evaluation can fall back EXPLICITLY, see
  // TargetSupportBasis::kCandidateProjectionFallback).
  Support corresponding_support;
  absl::Status corresponding_status;

  bool ok() const {
    return disposition == TargetEvaluationDisposition::kEvaluated;
  }

  // Opaque to callers outside relation_evaluator.cc; only
  // EvaluatePreparedTargetRelation reads these.
  const Element* referent_for_eval = nullptr;
  const NavigableSpatialDefinition* scope_for_eval = nullptr;
  bool affordance_for_eval = false;
  const RelationPayload* payload_for_eval = nullptr;
  RelationComponents components_for_eval;
};

PreparedTargetRelation PrepareTargetRelation(
    const RelationPayload& payload, const ElementTree& receiver_tree,
    const TargetEvaluationOptions& options = {});

// Eval_p at one candidate pose, reusing everything `prepared` already
// resolved. Requires `prepared.ok()`; still redoes the pose-DEPENDENT work
// (support computation and the kernel encode -- e.g. Spacing's nearest-
// boundary-point support genuinely moves with the candidate).
//
// `fast_distance` (performance): forwarded to
// Support::fast_distance (see support.h). AvatarSpacing's R depends on a
// GEODESIC distance to the referent's contact point -- a visibility-graph
// pathfind that dominated the paper-objective search's per-tick cost
// (measured: 96%+ of the search's instructions, one full pathfind per
// candidate). Pass true ONLY for a search's internal candidate-ranking
// calls, which need a cheap proxy to pick a descent direction, never for a
// value that decides acceptance -- always re-evaluate the winning candidate
// with the default (false, exact geodesic) before comparing it to the
// incumbent. Every existing caller keeps exact (false) behavior by default.
TargetRelationEvaluation EvaluatePreparedTargetRelation(
    const PreparedTargetRelation& prepared, const TargetAnchorPose& anchor,
    const ElementTree& receiver_tree, bool fast_distance = false,
    bool nearest_support_contact = false);

// Eval_p for every payload of a candidate state S given as anchor-name ->
// pose (an empty payload anchor name means "root"). A payload whose anchor is
// absent from `state` is kInvalidInput. Output order matches `payloads`.
std::vector<TargetRelationEvaluation> EvaluateTargetRelations(
    const std::vector<RelationPayload>& payloads,
    const std::unordered_map<std::string, TargetAnchorPose>& state,
    const ElementTree& receiver_tree,
    const TargetEvaluationOptions& options = {});

// ARRIVED for a Facing relation.
//
// v1's definition (Angular1D::
// ComputeExpectedInteractionTimeDetailed, "If already aligned"): the receiver is aligned when the
// SIGNED shortest angular delta to the target yaw is within math::Constants::kEpsilon
// (1e-6, applied to a yaw in DEGREES). It is an ANGLE and it is signed:
//   * no metre tolerance on the unitless R = (1 + cos theta) / 2, whose slope is 0 at alignment
//     (the old test accepted +-3.6 degrees near alignment but 0.11 degrees at 90);
//   * no mirror: R is even in theta, so -theta has the same R and was accepted. The signed offsets
//     are compared, and +theta is not -theta.
// Here the relation's signed offset is encoded_state(0), the bearing to the look-at contact minus the
// anchor's yaw. The target yaw v1 aligns to is the one at which the receiver's offset equals the
// source's, so the arrival error is theta_tgt(sample) - theta_src, wrapped.
// The one contact epsilon (remote_update.h kContactEpsilonM), read in degrees, as FacingUpdateCount's "aligned".
constexpr double kFacingArrivalRad = kContactEpsilonM * M_PI / 180.0;

// The signed angular error of a Facing relation at the sampled pose, in radians (wrapped to
// [-pi, pi]); nullopt when either side carries no signed offset.
inline std::optional<double> FacingAngularError(const RelationPayload& payload,
                                                const TargetRelationEvaluation& at_sample) {
  if (payload.grounded_relation.encoded_state.size() < 1 || at_sample.relation.encoded_state.size() < 1)
    return std::nullopt;
  const double src = payload.grounded_relation.encoded_state(0);
  const double tgt = at_sample.relation.encoded_state(0);
  if (!std::isfinite(src) || !std::isfinite(tgt)) return std::nullopt;
  return std::remainder(tgt - src, 2.0 * M_PI);
}

inline bool FacingArrived(const RelationPayload& payload, const TargetRelationEvaluation& at_sample) {
  const auto error = FacingAngularError(payload, at_sample);
  return error.has_value() && std::fabs(*error) < kFacingArrivalRad;
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_RELATION_EVALUATOR_H_
