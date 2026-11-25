// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_SCOPE_RECORD_H_
#define SPATIALBLEND_SPACE_SCOPE_RECORD_H_

#include <Eigen/Dense>
#include <limits>
#include <vector>

#include "spatialblend/base/common/types.h"

namespace spatialblend {

// Why each referent is in or out of a binding's scope.
//
// Recorded by the source-side binding pipeline (Space::EvaluateActorStateLocked)
// once per binding per evaluation, only while observability capture is on
// (tracing active for the space, or an observability getter was called; see
// Space::EnableObservabilityCapture). A referent that the kernel's scope
// policy never produced is ABSENT from the record ("never in scope"); every
// element the policy did produce is listed with the first filter that removed
// it, or kIncluded when it was evaluated. Candidates are in SCOPE-POLICY
// ORDER (the order ResolveScope returned them; an explicit leaf scope's
// self-candidate, which the policy does not return, comes last). Outcomes
// decided after evaluation (kExcludedObservedPointExpired at payload build)
// overwrite kIncluded in place. Observation only: recording never changes
// which referents are evaluated or delivered.
//
// Values are part of the C ABI (sb_scope_reason_t mirrors them 1:1); append
// new reasons, never renumber.
enum class ScopeReason : int {
  kIncluded = 0,
  // Produced by the kernel's scope policy but outside the explicit
  // sb_bind_scoped subtree.
  kExcludedExplicitScope = 1,
  // Binding tag filter did not match.
  kExcludedTag = 2,
  // An actor's own element (actors are subjects, never referents).
  kExcludedActorElement = 3,
  // Observed facing point skipped by a legacy/control update (only actual
  // source frames may use observed points).
  kExcludedObservedPointControl = 4,
  // Observed facing point not a live candidate for this actor/kernel.
  kExcludedObservedPointIneligible = 5,
  // Element vanished between scope resolution and evaluation.
  kExcludedMissingElement = 6,
  // Surface farther than the 20 m bounding-box prefilter.
  kExcludedSurfacePrefilter = 7,
  // Scope gating, relation/scope_gate.h:
  // Facing: no line of sight from the anchor.
  kExcludedNotVisible = 8,
  // Spacing: no collision-free walkable path to the approach point.
  kExcludedUnreachable = 9,
  // 10 (outside near-field) and 11 (gated inactive) are RETIRED with the
  // activation gate; the numbers are not reused.
  // Evaluated (was kIncluded), then dropped while building the propagated
  // payload because its observed facing point expired in between.
  kExcludedObservedPointExpired = 12,
  // 13 (beyond the scope radius) is RETIRED with the radii.
  // AffordancePose: the anchor is outside the referent's parent object region (D3).
  kExcludedOutsideParentRegion = 14,
  // Facing (a scope rule of the relation definition): the anchor pose is inside a containment-transfer
  // element, and this element is that element or one of its non-Surface descendants (ScopeResolver::
  // ExcludedFromFacingInsideTransfer).
  kExcludedInsideTransferLeaf = 15,
};

inline const char* ScopeReasonName(ScopeReason reason) {
  switch (reason) {
    case ScopeReason::kIncluded:
      return "included";
    case ScopeReason::kExcludedExplicitScope:
      return "explicit_scope";
    case ScopeReason::kExcludedTag:
      return "tag";
    case ScopeReason::kExcludedActorElement:
      return "actor_element";
    case ScopeReason::kExcludedObservedPointControl:
      return "observed_point_control";
    case ScopeReason::kExcludedObservedPointIneligible:
      return "observed_point_ineligible";
    case ScopeReason::kExcludedMissingElement:
      return "missing_element";
    case ScopeReason::kExcludedSurfacePrefilter:
      return "surface_prefilter";
    case ScopeReason::kExcludedNotVisible:
      return "not_visible";
    case ScopeReason::kExcludedUnreachable:
      return "unreachable";
    case ScopeReason::kExcludedObservedPointExpired:
      return "observed_point_expired";
    case ScopeReason::kExcludedOutsideParentRegion:
      return "outside_parent_region";
    case ScopeReason::kExcludedInsideTransferLeaf:
      return "inside_transfer_leaf";
  }
  return "unknown";
}

struct ScopeCandidate {
  element_id_t element_id = kInvalidElementId;
  // Shared id captured under the space lock at record time (the local
  // element id when the element is not shared).
  int64_t shared_or_local_id = -1;
  ScopeReason reason = ScopeReason::kIncluded;
  // R of the evaluated relation; NaN unless the candidate was evaluated.
  double proximity = std::numeric_limits<double>::quiet_NaN();
};

struct ScopeRecord {
  // False until the binding has been evaluated at least once.
  bool valid = false;
  // False when the binding's anchor was inactive (no candidates evaluated).
  bool anchor_active = false;
  bool explicit_scope = false;
  bool scope_live = false;
  Eigen::Vector3d anchor_position = Eigen::Vector3d::Zero();
  double anchor_yaw = 0.0;
  // Time of the evaluation (the source's phase time).
  double time_seconds = 0.0;
  // Scope root actually used: the explicit sb_bind_scoped root, or the
  // actor's nearest containment ancestor.
  element_id_t scope_root = kInvalidElementId;
  // The element the kernel's ResolveScope was called with (explicit root, or
  // the actor's own element for owner-derived scope).
  element_id_t kernel_scope_origin = kInvalidElementId;
  element_id_t deepest_containing = kInvalidElementId;
  // Element whose NavigableSpatialDefinition normalized this binding
  // (kInvalidElementId when none).
  element_id_t navigable_element = kInvalidElementId;
  // Raw scope (kernel policy output) in policy order, each with its outcome.
  std::vector<ScopeCandidate> candidates;

  // Resets every field but keeps the candidate buffer's capacity, so a
  // steady-state tick allocates nothing.
  void Reset() {
    std::vector<ScopeCandidate> keep = std::move(candidates);
    keep.clear();
    *this = ScopeRecord();
    candidates = std::move(keep);
  }

  void Add(element_id_t id, ScopeReason reason, int64_t shared_or_local_id) {
    ScopeCandidate c;
    c.element_id = id;
    c.shared_or_local_id = shared_or_local_id;
    c.reason = reason;
    candidates.push_back(c);
  }

  // Update an already-listed candidate (e.g. a later gating stage). Returns
  // false when `id` is not in the record.
  bool Mark(element_id_t id, ScopeReason reason) {
    for (auto& c : candidates) {
      if (c.element_id == id) {
        c.reason = reason;
        return true;
      }
    }
    return false;
  }

  bool SetProximity(element_id_t id, double proximity) {
    for (auto& c : candidates) {
      if (c.element_id == id) {
        c.proximity = proximity;
        return true;
      }
    }
    return false;
  }

  size_t IncludedCount() const {
    size_t n = 0;
    for (const auto& c : candidates) n += c.reason == ScopeReason::kIncluded;
    return n;
  }
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_SCOPE_RECORD_H_
