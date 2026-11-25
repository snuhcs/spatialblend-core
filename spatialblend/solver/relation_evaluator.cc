// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/relation_evaluator.cc
#include "spatialblend/solver/relation_evaluator.h"

#include <cmath>
#include <limits>
#include <optional>

#include "spatialblend/base/common/state_type.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/radial_surface_chart.h"
#include "spatialblend/model/relation/avatar_affordance_pose_kernel.h"
#include "spatialblend/model/relation/kernel_factory.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/spacing_grounding_rule.h"
#include "spatialblend/solver/relation_solver.h"

namespace spatialblend {

namespace {

// The state DOF the SOURCE binding encoded with, inferred from the encoded
// state's size, so the target kernel is configured like the source kernel.
std::vector<StateType> DofForPayload(const RelationPayload& payload) {
  const auto n = payload.grounded_relation.encoded_state.size();
  std::vector<StateType> dof = {StateType::kPosition2D,
                                StateType::kRotationYaw};
  switch (payload.kernel_type) {
    case RelationKernelType::kAvatarFacing:
      if (n == 3) dof[1] = StateType::kRotation3D;
      break;
    case RelationKernelType::kAvatarSpacing:
      if (n == 6) dof = {StateType::kPosition3D, StateType::kRotation3D};
      break;
    case RelationKernelType::kAvatarAffordancePose:
      if (n == 5) dof[1] = StateType::kRotation3D;
      break;
  }
  return dof;
}

bool UsableTransform(const Eigen::Matrix4d& m) {
  const double det = m.determinant();
  return m.allFinite() && std::isfinite(det) && std::abs(det) > 1e-12;
}

// AffordancePoseGroundingRule restricted to the source's named part: the
// candidate anchor projected into that part's patch (nearest boundary point
// when outside), exactly as the rule grounds the part it selects.
absl::Status GroundOnNamedPart(const Element& referent,
                               const ElementTree& tree,
                               const std::string& part_name,
                               const Eigen::Vector3d& anchor,
                               Support* support) {
  const SpatialDefinition* def = referent.GetDeclaredSpatialDefinition();
  const SubRegion* sub = def->GetSubRegion(part_name);
  if (!sub || !sub->boundary)
    return absl::NotFoundError("target support sub-region not found");
  if (!RadialSurfaceChart::Origin(*sub->boundary).ok() ||
      !UsableTransform(sub->local_transform))
    return absl::FailedPreconditionError("target part has no radial chart");
  const bool local_frame = def->RequiresElementWorldTransform();
  const Eigen::Matrix4d boundary_to_world =
      local_frame ? tree.GetWorldTransform(referent.GetId())
                  : def->GetTransform();
  const Eigen::Matrix4d patch_to_world =
      boundary_to_world * sub->local_transform;
  if (!UsableTransform(patch_to_world))
    return absl::FailedPreconditionError("target part frame is degenerate");
  const Eigen::Vector4d patch_anchor =
      patch_to_world.inverse() * anchor.homogeneous();
  Eigen::Vector2d local = patch_anchor.head<2>();
  const bool inside = sub->boundary->Contains(local);
  if (!inside) local = sub->boundary->GetNearestPoint(local);
  support->anchor_over_footprint = inside;
  support->boundary = sub->boundary;
  support->sub_region_name = sub->name;
  support->contact_point_local = local;
  support->contact_point_world =
      (patch_to_world * Eigen::Vector4d(local.x(), local.y(), 0.0, 1.0))
          .head<3>();
  support->world_transform = patch_to_world;
  auto chart = RadialSurfaceChart::Encode(local, *sub->boundary,
                                          Eigen::Vector2d(0, 1));
  if (chart.ok()) {
    support->contact_normalized = *chart;
    support->has_contact = true;
  }
  return absl::OkStatus();
}

// The CORRESPONDING target support -- the receiver-side point that the
// RadialSurfaceChart map M assigns to the source support point -- restricted
// to the source's named part, mirroring RelationSolver::
// GroundAffordanceTarget's decode. Unlike GroundOnNamedPart (which re-derives
// a support from wherever the CANDIDATE currently is), this support is fixed
// by the source's own recorded chart coordinates and does not move with the
// candidate: a hand at the wrong end of the same patch scores against the
// true corresponding point, not its own position.
absl::Status GroundOnMappedSupport(const Element& referent,
                                   const ElementTree& tree,
                                   const RelationPayload& payload,
                                   Support* support) {
  if (!payload.support.has_contact)
    return absl::FailedPreconditionError(
        "source payload carries no contact chart to map");
  const NormalizedPolygonPosition& chart = payload.support.contact_normalized;
  if (chart.strategy != NormalizedPolygonPosition::Strategy::kRadial)
    return absl::UnimplementedError("MVC affordance charts are unsupported");
  const SpatialDefinition* def = referent.GetDeclaredSpatialDefinition();
  const SubRegion* sub = def->GetSubRegion(payload.support.sub_region_name);
  if (!sub || !sub->boundary)
    return absl::NotFoundError("target support sub-region not found");
  if (!RadialSurfaceChart::Origin(*sub->boundary).ok() ||
      !UsableTransform(sub->local_transform))
    return absl::FailedPreconditionError("target part has no radial chart");
  auto decoded =
      RadialSurfaceChart::Decode(chart, *sub->boundary, Eigen::Vector2d(0, 1));
  if (!decoded.ok()) return decoded.status();
  const bool local_frame = def->RequiresElementWorldTransform();
  const Eigen::Matrix4d boundary_to_world =
      local_frame ? tree.GetWorldTransform(referent.GetId())
                  : def->GetTransform();
  const Eigen::Matrix4d patch_to_world =
      boundary_to_world * sub->local_transform;
  if (!UsableTransform(patch_to_world))
    return absl::FailedPreconditionError("target part frame is degenerate");
  support->boundary = sub->boundary;
  support->sub_region_name = sub->name;
  support->contact_point_local = *decoded;
  support->contact_point_world =
      (patch_to_world * Eigen::Vector4d(decoded->x(), decoded->y(), 0.0, 1.0))
          .head<3>();
  support->world_transform = patch_to_world;
  support->contact_normalized = chart;  // M: same normalized chart, decoded
                                        // against the TARGET's own boundary.
  support->has_contact = true;
  return absl::OkStatus();
}

// Relative yaw (encoded_state[2]) of an AffordancePose relation, if present.
std::optional<double> RelativeYaw(const GroundedRelation& gr) {
  if (gr.encoded_state.size() < 3 || !std::isfinite(gr.encoded_state[2]))
    return std::nullopt;
  return gr.encoded_state[2];
}

// Relation-preservation scores r_src / r_tgt. See
// TargetRelationEvaluation::r_formula_version. Both sides always use the same
// formula and the same yaw availability.
void ScorePreservation(const RelationPayload& payload,
                       TargetRelationEvaluation* out) {
  const GroundedRelation& src = payload.grounded_relation;
  const GroundedRelation& tgt = out->relation;
  if (payload.kernel_type == RelationKernelType::kAvatarAffordancePose &&
      AffordancePoseRFormulaVersion(src) == kApRFormulaPoseFitV1 &&
      AffordancePoseRFormulaVersion(tgt) == kApRFormulaPoseFitV1) {
    // Yaw is scored relative to the SOURCE's relative yaw, not the patch's
    // own frame. Unknown on either side => yaw contributes to neither R.
    const std::optional<double> src_yaw = RelativeYaw(src);
    const std::optional<double> tgt_yaw = RelativeYaw(tgt);
    std::optional<double> src_err, tgt_err;
    if (src_yaw && tgt_yaw) {
      src_err = 0.0;  // the source's yaw is its own reference
      tgt_err = std::remainder(*tgt_yaw - *src_yaw, 2.0 * M_PI);
    }
    out->r_src = AffordancePoseR(src.ap_offset_u_m, src.ap_offset_v_m,
                                 src.ap_height_m, src_err);
    out->r_tgt = AffordancePoseR(tgt.ap_offset_u_m, tgt.ap_offset_v_m,
                                 tgt.ap_height_m, tgt_err);
    out->r_formula_version = kApRFormulaPoseFitV1;
  } else {
    // Pre-pose-fit sender (R_src is the old proximity), or a family / support with
    // no pose decomposition: the proximity component on both sides.
    out->r_src = src.proximity;
    out->r_tgt = tgt.proximity;
    out->r_formula_version = kApRFormulaProximityOnly;
  }
  out->error = out->r_tgt - out->r_src;
}

TargetEvaluationDisposition DispositionFor(const absl::Status& status) {
  switch (status.code()) {
    case absl::StatusCode::kNotFound:
      return TargetEvaluationDisposition::kPartNotFound;
    case absl::StatusCode::kInvalidArgument:
      return TargetEvaluationDisposition::kInvalidInput;
    default:
      return TargetEvaluationDisposition::kUnsupportedSupport;
  }
}

}  // namespace

const char* TargetEvaluationDispositionName(TargetEvaluationDisposition d) {
  switch (d) {
    case TargetEvaluationDisposition::kEvaluated:
      return "evaluated";
    case TargetEvaluationDisposition::kUnmapped:
      return "unmapped";
    case TargetEvaluationDisposition::kNoSupportGeometry:
      return "no_support_geometry";
    case TargetEvaluationDisposition::kPartNotFound:
      return "part_not_found";
    case TargetEvaluationDisposition::kUnsupportedSupport:
      return "unsupported_support";
    case TargetEvaluationDisposition::kInvalidInput:
      return "invalid_input";
  }
  return "unknown";
}

PreparedTargetRelation PrepareTargetRelation(
    const RelationPayload& payload, const ElementTree& tree,
    const TargetEvaluationOptions& options) {
  PreparedTargetRelation out;
  out.kernel_type = payload.kernel_type;
  out.anchor_name =
      payload.subject_anchor_name.empty() ? "root" : payload.subject_anchor_name;
  out.referent_id = payload.referent_shared_id;
  out.r_src = payload.grounded_relation.proximity;
  out.payload_for_eval = &payload;

  // Note: unlike EvaluateTargetRelation's `fail`, this one must not return
  // PreparedTargetRelation BY VALUE -- it holds a move-only
  // RelationComponents (owning the kernel/grounding-rule unique_ptrs), and
  // `out` here is accessed through the lambda's reference capture, not a
  // local of the lambda itself, so "return out;" from inside the lambda
  // would need to COPY it (deleted). Set the fields and let each call site
  // do its own `return out;` -- a genuine named-local return, eligible for
  // NRVO/move.
  auto fail = [&](TargetEvaluationDisposition d, std::string reason) {
    out.disposition = d;
    out.reason = std::move(reason);
  };

  // Referent: the linker-resolved local id first, then the resolver.
  const Element* referent = nullptr;
  if (payload.resolved_local_element_id != kInvalidElementId) {
    referent = tree.GetElement(payload.resolved_local_element_id);
  } else if (payload.referent_shared_id != kInvalidSharedElementId) {
    ReferentResolver default_resolver;
    referent = (options.resolver ? *options.resolver : default_resolver)
                   .Resolve(payload.referent_shared_id, tree);
  }
  if (!referent) {
    fail(TargetEvaluationDisposition::kUnmapped,
        "referent unresolved on receiver");
    return out;
  }
  out.local_referent_id = referent->GetId();
  out.referent_for_eval = referent;

  out.components_for_eval =
      KernelFactory::Create(payload.kernel_type, DofForPayload(payload));
  if (!out.components_for_eval.kernel || !out.components_for_eval.grounding_rule) {
    fail(TargetEvaluationDisposition::kInvalidInput,
        "unknown relation family");
    return out;
  }

  // Normalizing context: an explicit option, else the rule shared with the
  // decode (ResolveSpacingNormalization / CorrespondingScope).
  element_id_t scope_id = options.normalizing_scope;
  if (scope_id == kInvalidElementId) {
    if (payload.kernel_type == RelationKernelType::kAvatarSpacing) {
      scope_id = ResolveSpacingNormalization(payload, tree).scope_element;
    } else {
      CorrespondingScope(payload, tree, &scope_id);
    }
  }
  const NavigableSpatialDefinition* scope = nullptr;
  if (scope_id != kInvalidElementId) {
    const Element* scope_el = tree.GetElement(scope_id);
    scope = scope_el ? dynamic_cast<const NavigableSpatialDefinition*>(
                           scope_el->GetSpatialDefinition())
                     : nullptr;
    if (!scope) {
      fail(TargetEvaluationDisposition::kInvalidInput,
          "normalizing scope has no navigable definition");
      return out;
    }
    out.normalizing_scope_id = scope_id;
    const Bound2D b = scope->GetBounds();
    out.normalizing_scope_diameter_m = (b.max - b.min).norm();
  }
  out.scope_for_eval = scope;

  const SpatialDefinition* def = referent->GetDeclaredSpatialDefinition();
  const bool affordance =
      payload.kernel_type == RelationKernelType::kAvatarAffordancePose;
  out.affordance_for_eval = affordance;
  // A source support grounded on geometry cannot be evaluated against a
  // geometryless counterpart (AffordancePose always needs a surface).
  if (!def && (affordance || payload.support.has_contact)) {
    fail(TargetEvaluationDisposition::kNoSupportGeometry,
        "target referent has no support geometry");
    return out;
  }

  // The corresponding target support (candidate-independent, so it
  // belongs here, not in the per-candidate evaluation).
  if (affordance && !payload.support.sub_region_name.empty()) {
    out.corresponding_support.referent_element_id = referent->GetId();
    out.corresponding_support.navigable_def = scope;
    out.corresponding_status = GroundOnMappedSupport(
        *referent, tree, payload, &out.corresponding_support);
  }

  // The mapped source support on this referent (what the realizer targets).
  // Pose-independent: never re-derived per candidate.
  if (affordance) {
    RelationSolver solver;
    RelationPayload resolved = payload;
    resolved.resolved_local_element_id = referent->GetId();
    auto target = solver.GroundAffordanceTarget(resolved, tree);
    if (target.ok()) {
      out.has_mapped_support = true;
      out.mapped_support = target->position;
    }
  } else {
    auto mapped = DecodeMappedSupportPoint(payload, *referent, tree);
    if (mapped.ok()) {
      out.has_mapped_support = true;
      out.mapped_support = Eigen::Vector3d(mapped->x(), mapped->y(), 0.0);
    }
  }

  out.disposition = TargetEvaluationDisposition::kEvaluated;
  return out;
}

// `nearest_support_contact`: measure the spacing distance to the
// geodesic-nearest support contact (the grounding rule's contact). The
// default keeps the mapped source contact, which is the point R^tgt is
// scored against for relation-error reporting.
TargetRelationEvaluation EvaluatePreparedTargetRelation(
    const PreparedTargetRelation& prepared, const TargetAnchorPose& anchor,
    const ElementTree& tree, bool fast_distance, bool nearest_support_contact) {
  TargetRelationEvaluation out;
  out.disposition = prepared.disposition;
  out.reason = prepared.reason;
  out.kernel_type = prepared.kernel_type;
  out.anchor_name = prepared.anchor_name;
  out.referent_id = prepared.referent_id;
  out.local_referent_id = prepared.local_referent_id;
  out.normalizing_scope_id = prepared.normalizing_scope_id;
  out.normalizing_scope_diameter_m = prepared.normalizing_scope_diameter_m;
  out.r_src = prepared.r_src;
  out.has_mapped_support = prepared.has_mapped_support;
  out.mapped_support = prepared.mapped_support;

  auto fail = [&](TargetEvaluationDisposition d, std::string reason) {
    out.disposition = d;
    out.reason = std::move(reason);
    return out;
  };

  if (!prepared.ok())
    return out;  // Prepare already recorded disposition/reason.
  if (!anchor.position.allFinite() || !std::isfinite(anchor.yaw))
    return fail(TargetEvaluationDisposition::kInvalidInput,
                "candidate anchor pose is not finite");

  const Element& referent = *prepared.referent_for_eval;
  const RelationComponents& components = prepared.components_for_eval;

  const RelationPayload& payload = *prepared.payload_for_eval;
  Support support;
  bool has_candidate_projection = false;
  Support candidate_projected_support;
  if (prepared.affordance_for_eval && !payload.support.sub_region_name.empty()) {
    // Diagnostic only: where the candidate's own anchor projects onto
    // the same patch. Never used for r_tgt/error below.
    candidate_projected_support.referent_element_id = referent.GetId();
    candidate_projected_support.navigable_def = prepared.scope_for_eval;
    has_candidate_projection =
        GroundOnNamedPart(referent, tree, payload.support.sub_region_name,
                          anchor.position, &candidate_projected_support)
            .ok();

    // Score the candidate against the CORRESPONDING target support --
    // the RadialSurfaceChart map M of the source support point -- resolved
    // once in PrepareTargetRelation, fixed independently of the candidate.
    if (prepared.corresponding_status.ok()) {
      support = prepared.corresponding_support;
      out.support_basis = TargetSupportBasis::kMappedSupport;
    } else {
      if (!has_candidate_projection)
        return fail(DispositionFor(prepared.corresponding_status),
                    std::string(prepared.corresponding_status.message()));
      // Old-sender compatibility: a payload that cannot decode a mapped
      // support (e.g. a legacy sender missing chart fields) is scored
      // against the candidate's own projection rather than failing
      // outright -- but EXPLICITLY: support_basis says so and `reason`
      // records why. It is not a faithful Eval_p (the earlier behavior).
      support = candidate_projected_support;
      out.support_basis = TargetSupportBasis::kCandidateProjectionFallback;
      out.reason = "mapped support unavailable (" +
                   std::string(prepared.corresponding_status.message()) +
                   "); scored against the candidate's own projection";
    }
  } else {
    support = components.grounding_rule->ComputeSupport(
        referent, tree, anchor.position, anchor.yaw, prepared.scope_for_eval);
    // Spacing R is the geodesic to the mapped source contact, a fixed point
    // of the receiver support. A candidate does not choose a new contact.
    if (!nearest_support_contact &&
        payload.kernel_type == RelationKernelType::kAvatarSpacing &&
        prepared.has_mapped_support) {
      support.contact_point_world = prepared.mapped_support;
      support.has_contact = true;
      // The ring field measured the geodesic to the nearest contact. The distance to this other contact is its own
      // length, measured as the source measures its own: the referent's own
      // footprints do not block the final leg onto its contact (OwnFootprints), so identical rooms give identical
      // distances.
      support.geodesic_length_m =
          prepared.scope_for_eval && !fast_distance
              ? prepared.scope_for_eval->GetGeodesicDistance(Eigen::Vector2d(anchor.position.head<2>()),
                                                             Eigen::Vector2d(prepared.mapped_support.head<2>()),
                                                             OwnFootprints(tree, referent))
              : -1.0;
      out.support_basis = TargetSupportBasis::kMappedSupport;
    }
  }
  if (prepared.affordance_for_eval && payload.support.has_contact &&
      !support.has_contact)
    return fail(TargetEvaluationDisposition::kUnsupportedSupport,
                "target geometry does not carry the affordance support");

  // AffordancePose: does the CANDIDATE anchor touch a part of the receiver's
  // counterpart surface (engagement.h IsAffordanceContact)? Judged against the
  // support the candidate itself projects onto, not the mapped contact: touching
  // is a fact about the anchor and the surface, wherever on it.
  if (prepared.affordance_for_eval) {
    const Support& touched =
        has_candidate_projection ? candidate_projected_support : support;
    if (touched.boundary) {
      const GroundedRelation g =
          components.kernel->Encode(anchor.position, anchor.yaw, touched);
      out.anchor_touches_support = IsAffordanceContact(g);
    }
  }

  support.container_self = payload.support.container_self;
  support.fast_distance = fast_distance;
  out.relation = components.kernel->Encode(anchor.position, anchor.yaw, support);
  out.support = std::move(support);
  // r_src / r_tgt / error under one R formula (ScorePreservation).
  ScorePreservation(payload, &out);
  out.has_candidate_projection = has_candidate_projection;
  if (has_candidate_projection)
    out.candidate_projected_support = std::move(candidate_projected_support);
  out.disposition = TargetEvaluationDisposition::kEvaluated;
  return out;
}

TargetRelationEvaluation EvaluateTargetRelation(
    const RelationPayload& payload, const TargetAnchorPose& anchor,
    const ElementTree& tree, const TargetEvaluationOptions& options) {
  // EvaluateTargetRelation is Prepare+EvaluateAtPose
  // composed for a single call, so a caller with many candidate poses per
  // relation (the paper-objective search) can call PrepareTargetRelation
  // once and EvaluatePreparedTargetRelation per candidate instead, without
  // the two code paths drifting apart.
  const PreparedTargetRelation prepared =
      PrepareTargetRelation(payload, tree, options);
  if (!prepared.ok()) {
    TargetRelationEvaluation out;
    out.disposition = prepared.disposition;
    out.reason = prepared.reason;
    out.kernel_type = prepared.kernel_type;
    out.anchor_name = prepared.anchor_name;
    out.referent_id = prepared.referent_id;
    out.r_src = prepared.r_src;
    return out;
  }
  return EvaluatePreparedTargetRelation(prepared, anchor, tree);
}

std::vector<TargetRelationEvaluation> EvaluateTargetRelations(
    const std::vector<RelationPayload>& payloads,
    const std::unordered_map<std::string, TargetAnchorPose>& state,
    const ElementTree& tree, const TargetEvaluationOptions& options) {
  std::vector<TargetRelationEvaluation> out;
  out.reserve(payloads.size());
  for (const auto& payload : payloads) {
    const std::string anchor = payload.subject_anchor_name.empty()
                                   ? "root"
                                   : payload.subject_anchor_name;
    const auto it = state.find(anchor);
    if (it == state.end()) {
      TargetRelationEvaluation missing;
      missing.kernel_type = payload.kernel_type;
      missing.anchor_name = anchor;
      missing.referent_id = payload.referent_shared_id;
      missing.r_src = payload.grounded_relation.proximity;
      missing.disposition = TargetEvaluationDisposition::kInvalidInput;
      missing.reason = "candidate state has no pose for anchor";
      out.push_back(std::move(missing));
      continue;
    }
    out.push_back(EvaluateTargetRelation(payload, it->second, tree, options));
  }
  return out;
}

}  // namespace spatialblend
