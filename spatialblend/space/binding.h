// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_BINDING_H_
#define SPATIALBLEND_SPACE_BINDING_H_

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "spatialblend/base/common/types.h"
#include "spatialblend/model/relation/grounded_relation.h"
#include "spatialblend/model/relation/grounding_rule.h"
#include "spatialblend/model/relation/relation.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/scope_resolver.h"
#include "spatialblend/space/scope_record.h"
#include "spatialblend/space/relation_phase_tracker.h"

namespace spatialblend {

struct Binding {
  static constexpr size_t kInvalidOrdinal = std::numeric_limits<size_t>::max();

  Binding(const std::string& anchor_name_in, RelationKernelType kernel_type_in,
          const std::string& tag_filter_in,
          std::unique_ptr<GroundingRule> grounding_rule_in,
          std::unique_ptr<RelationKernel> kernel_in, size_t binding_ordinal_in,
          size_t anchor_ordinal_in)
      : anchor_name(anchor_name_in),
        kernel_type(kernel_type_in),
        tag_filter(tag_filter_in),
        grounding_rule(std::move(grounding_rule_in)),
        kernel(std::move(kernel_in)),
        binding_ordinal(binding_ordinal_in),
        anchor_ordinal(anchor_ordinal_in) {}

  // Movable, not copyable (due to unique_ptr members)
  Binding(Binding&&) = default;
  Binding& operator=(Binding&&) = default;
  Binding(const Binding&) = delete;
  Binding& operator=(const Binding&) = delete;

  std::string anchor_name;
  RelationKernelType kernel_type;
  // Tag filter: exact case-sensitive match against Element::GetTag().
  // Empty string = match all elements (no filtering).
  // Elements with empty tags never match a non-empty filter.
  std::string tag_filter;

  // kInvalidElementId preserves legacy owner/containment-derived scope.
  // A non-invalid id is an explicit hard subtree boundary.
  element_id_t explicit_scope_root = kInvalidElementId;

  std::unique_ptr<GroundingRule> grounding_rule;
  std::unique_ptr<RelationKernel> kernel;

  // Captured at native bind time. Owned source evidence must retain this
  // association instead of resolving an anchor name or relation position in a
  // later capture. Pointer-returning binding APIs make certified copies
  // unavailable before external mutation can matter.
  size_t binding_ordinal = kInvalidOrdinal;
  size_t anchor_ordinal = kInvalidOrdinal;
  // Identity of this binding's definition. A binding never changes
  // after creation, so a space-wide counter assigned at creation (never
  // reused, 0 = unassigned) identifies exactly the definition a relation was
  // extracted under; clearing and re-binding an actor yields new revisions.
  uint64_t revision = 0;

  struct MatchResult {
    element_id_t element_id = kInvalidElementId;
    RelationEvaluationResult eval;
    // Source shared id copied from a remote payload. Local evaluation leaves
    // this invalid and the query derives the phase key from the local element.
    // A LinkToExisting referent keeps its own shared id, which is not the
    // source id EmitRelationTransitions keys the phase on.
    shared_element_id_t referent_shared_id = kInvalidSharedElementId;
    // An inner affordance-pose relation (RelationPayload::affordance_pose_tag): `element_id` is the pose's
    // interaction child (the progress baseline's key), `pose_object` the object whose SD holds it (the referent sent).
    std::string pose_tag;
    element_id_t pose_object = kInvalidElementId;
  };
  std::vector<MatchResult> latest_matches;

  // Each referent's grounded relation as of the last update this binding SENT (the progress
  // baseline of the update count n). The source-side movement gate sends only some ticks, and
  // v1 measures progress against the last ACCEPTED state, not the previous tick
  // (Space::ShouldProcessStateUpdate keeps the entity's state when a change is below the gate;
  // EuclideanND / Angular1D compute n against that state). So the baseline moves only when an
  // update leaves the space (Space::EvaluateActorStateLocked), never per tick. A referent that
  // was not in the last sent update has no baseline: it holds (n infinite) until the next one.
  std::unordered_map<element_id_t, GroundedRelation> sent_relations;
  // The anchor position of the last update this binding sent. A Facing reports whether the path
  // from it to the anchor now crossed a referent's boundary (Support::look_at_boundary_crossed):
  // the receiver sees only the sent updates, so the path to attribute a yaw step to is the one
  // between two sent poses. Cleared when the anchor goes inactive.
  std::optional<Eigen::Vector3d> sent_anchor_position;
  // The anchor yaw of the last update this binding sent. With sent_anchor_position it is the
  // previous POSE a Facing's progress is measured from: the previous bearing offset is the one
  // from that pose to the CURRENT look-at contact, so the count compares one point at two poses
  // (a look-at contact that moved to another vertex since the last send is not a rotation).
  // Cleared with sent_anchor_position.
  std::optional<double> sent_anchor_yaw;
  // The anchor position of this binding's latest evaluation; it becomes sent_anchor_position
  // when the evaluation's update is sent.
  std::optional<Eigen::Vector3d> latest_anchor_position;
  // The anchor yaw of this binding's latest evaluation; it becomes sent_anchor_yaw likewise.
  std::optional<double> latest_anchor_yaw;

  // EMA of the per-referent source-side progress (proximity) derivative, for
  // the Progressive policy's imminence estimate Ṙ⁺ (SpatialBlend
  // app:policy-instantiation, eq. smoothed_derivative). Keyed by referent
  // element.
  std::unordered_map<element_id_t, double> smoothed_progress_rate;
  // Per-binding/referent source elapsed time. This must not use Space's global
  // clock because unrelated actor updates may advance it between these samples.
  std::unordered_map<element_id_t, double> last_progress_sample_seconds;
  // Presence is separate from the value: t=0 is a genuine recorded sample.
  std::unordered_set<element_id_t> has_progress_sample;

  // The Spacing scope of the latest evaluation (mode, container, entered referents): set for a
  // Spacing binding every evaluation, empty for the other families. Observation only.
  std::optional<SpacingScope> spacing_scope;

  // Observation: the latest evaluation's scope record (raw scope, each
  // candidate's filter outcome). Written by the source pipeline only; never
  // read back by it.
  ScopeRecord scope_record;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_BINDING_H_
