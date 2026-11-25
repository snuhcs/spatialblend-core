// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/space.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>

#include "spatialblend/base/util/logger.h"
#include "spatialblend/base/common/trace.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/model/relation/facing_grounding_rule.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/scope_gate.h"
#include "spatialblend/solver/affordance_pose_sd.h"
#include "spatialblend/solver/realization_policy.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {

namespace {

// Whether the softmax weights moved S0 beyond what the approach points' own
// motion did (WEIGHT_SWITCH). `prev` and `cur` are the position goals
// of two consecutive solves with the same referent set. Pure: S0 is the
// weighted point sum(w a)/sum(w); the tolerance is the geometry epsilon the
// relation layer uses (1e-9 m). A goal that is in one list only has weight 0 in the
// other (a goal enters a solve with weight > 0; one whose weight reached 0 is not
// there), and a weight-0 goal contributes nothing to either sum -- the lists are
// compared over the union, never by their sizes.
bool WeightsAddedS0Motion(const std::vector<ProposalGoal>& prev,
                          const std::vector<ProposalGoal>& cur) {
  struct Term { Eigen::Vector2d p_prev, p_cur; double w_prev = 0.0, w_now = 0.0; };
  std::vector<std::pair<std::pair<shared_element_id_t, int>, Term>> terms;
  const auto at = [&](const ProposalGoal& g) -> Term& {
    for (auto& t : terms)
      if (t.first.first == g.referent && t.first.second == g.kernel) return t.second;
    terms.push_back({{g.referent, g.kernel}, Term{}});
    return terms.back().second;
  };
  for (const auto& g : prev) { Term& t = at(g); t.p_prev = g.point; t.p_cur = g.point; t.w_prev = g.weight; }
  for (const auto& g : cur) {
    bool known = false;
    for (const auto& q : prev)
      if (q.referent == g.referent && q.kernel == g.kernel) known = true;
    Term& t = at(g);
    if (!known) t.p_prev = g.point;
    t.p_cur = g.point;
    t.w_now = g.weight;
  }
  Eigen::Vector2d s_prev = Eigen::Vector2d::Zero(), s_geo = s_prev, s_now = s_prev;
  double w_prev = 0.0, w_now = 0.0;
  for (const auto& [key, t] : terms) {
    s_prev += t.w_prev * t.p_prev;
    s_geo += t.w_prev * t.p_cur;  // previous weight, current point
    s_now += t.w_now * t.p_cur;
    w_prev += t.w_prev;
    w_now += t.w_now;
  }
  // (S0 is a weighted MEAN: it does not depend on the scale of the weights, so there is no
  // threshold on their sum, only that they are positive.)
  if (!(w_prev > 0.0) || !(w_now > 0.0)) return false;
  s_prev /= w_prev;
  s_geo /= w_prev;
  s_now /= w_now;
  constexpr double kEpsilon = 1e-9;  // the relation layer's geometry epsilon
  return (s_now - s_prev).norm() > (s_geo - s_prev).norm() + kEpsilon;
}


// The same question for the yaw: with the Facing relations unchanged, did the weights
// (and the bearings' conditioning) move the circular mean beyond the bearings' own motion?
// Over the union of the two lists, as WeightsAddedS0Motion: a Facing in one list only has
// weight 0 in the other and contributes nothing.
bool WeightsAddedYawMotion(const std::vector<ProposalYawGoal>& prev,
                           const std::vector<ProposalYawGoal>& cur) {
  struct Term { double b_prev = 0.0, b_cur = 0.0, m_prev = 0.0, m_now = 0.0; };
  std::vector<std::pair<std::pair<shared_element_id_t, int>, Term>> terms;
  const auto at = [&](const ProposalYawGoal& g) -> Term& {
    for (auto& t : terms)
      if (t.first.first == g.referent && t.first.second == g.kernel) return t.second;
    terms.push_back({{g.referent, g.kernel}, Term{}});
    return terms.back().second;
  };
  for (const auto& g : prev) {
    Term& t = at(g);
    t.b_prev = t.b_cur = g.bearing;
    t.m_prev = g.weight * g.conditioning;
  }
  for (const auto& g : cur) {
    bool known = false;
    for (const auto& q : prev)
      if (q.referent == g.referent && q.kernel == g.kernel) known = true;
    Term& t = at(g);
    if (!known) t.b_prev = g.bearing;
    t.b_cur = g.bearing;
    t.m_now = g.weight * g.conditioning;
  }
  double sp_x = 0, sp_y = 0, sg_x = 0, sg_y = 0, sn_x = 0, sn_y = 0;
  for (const auto& [key, t] : terms) {
    sp_x += t.m_prev * std::cos(t.b_prev); sp_y += t.m_prev * std::sin(t.b_prev);
    sg_x += t.m_prev * std::cos(t.b_cur);  sg_y += t.m_prev * std::sin(t.b_cur);
    sn_x += t.m_now * std::cos(t.b_cur);   sn_y += t.m_now * std::sin(t.b_cur);
  }
  const auto mag = [](double x, double y) { return std::hypot(x, y); };
  // The yaw is the DIRECTION of the weighted sum, which does not depend on the scale of the
  // weights (a Facing whose weight is 1e-9 still carries the yaw, as the solver has it): it is
  // undefined only where the sum is exactly zero.
  if (mag(sp_x, sp_y) == 0.0 || mag(sg_x, sg_y) == 0.0 || mag(sn_x, sn_y) == 0.0) return false;
  const double y_prev = std::atan2(sp_y, sp_x), y_geo = std::atan2(sg_y, sg_x), y_now = std::atan2(sn_y, sn_x);
  constexpr double kEpsilon = 1e-9;
  return std::fabs(std::remainder(y_now - y_prev, 2 * M_PI)) >
         std::fabs(std::remainder(y_geo - y_prev, 2 * M_PI)) + kEpsilon;
}

double WorldYawOf(const Eigen::Matrix4d& m) {
  return std::atan2(m(1, 0), m(0, 0));
}


// SPEC S3 (api/c/trace.format.md): stringify a kernel type / relation phase
// for the structured trace's "relation" ev. Lowercase snake_case names; the
// integer `kernel` field on other ev kinds is the raw enum value (Facing=0,
// Spacing=1, AffordancePose=2 — the parser translates by name).
const char* KernelTraceName(RelationKernelType k) {
  switch (k) {
    case RelationKernelType::kAvatarFacing:
      return "facing";
    case RelationKernelType::kAvatarSpacing:
      return "spacing";
    case RelationKernelType::kAvatarAffordancePose:
      return "affordance_pose";
  }
  return "unknown";
}

// Internal RelationPhaseTracker state name (the tracker's own vocabulary), preserved
// verbatim on every ev=relation line's "tracker_phase" field so translating
// to SPEC S3's phase vocabulary (below) never loses information.
const char* TrackerPhaseName(RelationPhase p) {
  switch (p) {
    case RelationPhase::kDormant:
      return "dormant";
    case RelationPhase::kActive:
      return "active";
    case RelationPhase::kDriving:
      return "driving";
    case RelationPhase::kContact:
      return "contact";
  }
  return "unknown";
}

// SPEC S3 (api/c/trace.format.md): translate a phase TRANSITION's `to`
// state to the spec's three-value "phase" vocabulary. See the mapping table
// in spatialblend/base/common/trace.h (mirrored in api/c/trace.format.md) for the
// full rationale; identical here by construction:
//   to == kDormant                -> "release"  (relation ended)
//   to == kActive or kDriving      -> "approach" (relation newly forming)
//   to == kContact                 -> "engage"   (grounded contact reached)
const char* SpecPhaseTraceName(RelationPhase to) {
  switch (to) {
    case RelationPhase::kDormant:
      return "release";
    case RelationPhase::kActive:
      return "approach";
    case RelationPhase::kDriving:
      return "approach";
    case RelationPhase::kContact:
      return "engage";
  }
  return "unknown";
}

constexpr int kHeartbeatInterval = 30;  // every 30 frames (~1 second at 30fps)

// Which relation of each channel actually drove a realized target.
//  - Root channels (spacing/facing blended into the root solve): the
//    max-weight channel payload among the goals that entered the blend
//    (resolvable, non-zero weight), provided the realization has targets.
//  - Anchor channels (affordance pose; rich facing): the channel payload of
//    the target's referent (the solver's dominant referent) when a target for
//    that (anchor, family) survived the anchor gate; none when it was gated out.
// Ties break toward the lowest payload index, like the policy argmax.
void AnnotateRealizedDriving(
    const std::vector<RelationPayload>& payloads,
    const std::vector<double>& weights,
    const Space::SolverObservation* solve,
    const RealizationResult* realization,
    std::vector<RealizationChannelDiagnostics>* channels) {
  auto weight_at = [&](size_t i) { return i < weights.size() ? weights[i] : 0.0; };
  std::vector<bool> in_root_blend(payloads.size(), false);
  const bool root_realized =
      solve && realization && !realization->targets.empty();
  if (root_realized) {
    for (const auto& goal : solve->solve.goals) {
      if (goal.payload_index < solve->goal_payload_index.size()) {
        const size_t idx = solve->goal_payload_index[goal.payload_index];
        if (idx < in_root_blend.size()) in_root_blend[idx] = true;
      }
    }
  }
  for (auto& channel : *channels) {
    channel.has_realized = false;
    channel.realized_index = 0;
    channel.realized_weight = 0.0;
    auto consider = [&](size_t idx) {
      const double w = weight_at(idx);
      if (!(w > 0.0)) return;
      if (!channel.has_realized || w > channel.realized_weight) {
        channel.has_realized = true;
        channel.realized_index = idx;
        channel.realized_weight = w;
      }
    };
    for (size_t idx : channel.payload_indices)
      if (in_root_blend[idx]) consider(idx);
    if (channel.has_realized || !realization) continue;
    const std::string anchor =
        channel.anchor_name.empty() ? "root" : channel.anchor_name;
    for (const auto& target : realization->targets) {
      if (target.kernel_type != channel.kernel_type ||
          target.anchor_name != anchor ||
          target.referent_id == kInvalidSharedElementId)
        continue;
      for (size_t idx : channel.payload_indices)
        if (payloads[idx].referent_shared_id == target.referent_id)
          consider(idx);
    }
  }
}

}  // namespace

// static
const std::vector<Binding> Space::kEmptyBindings;

void Space::CleanupRemovedElements(
    const std::vector<element_id_t>& removed_ids) {
  std::unordered_set<element_id_t> removed(removed_ids.begin(),
                                           removed_ids.end());
  for (element_id_t id : removed_ids) {
    InvalidateObservedFacingConsumersLocked(id);
    observed_facing_points_.erase(id);
  }
  for (auto& [actor_id, actor] : actors_) {
    if (removed.count(actor->GetElementId()) == 0) continue;
    for (const SubjectAnchor& anchor : actor->GetAnchors()) {
      actor->ClearAnchorWorldPose(anchor.name);
    }
    actor->SetElementId(kInvalidElementId);
    InvalidateReceiverInputLocked(actor_id);
    phase_tracker_.ClearOwner(PhaseOwner(actor_id));
    source_relations_.erase(actor_id);
    source_evidence_enabled_.erase(actor_id);
    source_evidence_phase_dirty_.erase(actor_id);
    source_evidence_.erase(actor_id);
    observed_facing_dependencies_.erase(actor_id);
    bindings_.erase(actor_id);
    realization_results_.erase(actor_id);
    realization_snapshots_.erase(actor_id);
    previous_weights_.erase(actor_id);
    solver_warm_states_.erase(actor_id);
    proposal_attribution_.erase(actor_id);
    receiver_actor_seconds_.erase(actor_id);
    latency_trackers_.erase(actor_id);
    receiver_actor_updates_.erase(actor_id);
    motion_trackers_.erase(actor_id);
    actor_update_seconds_.erase(actor_id);
    {
      ReceiverProposal released_proposal;
      released_proposal.status = ProposalStatus::kNone;
      PublishProposalLocked(actor_id, std::move(released_proposal));
    }
  }
  // Bindings for surviving actors can still cache a removed referent.
  for (auto& [actor_id, bindings] : bindings_) {
    const auto old = source_relations_.find(actor_id);
    if (old != source_relations_.end()) {
      std::vector<SourceRelationObservation> remaining;
      for (const auto& relation : old->second)
        if (removed.count(relation.referent) == 0)
          remaining.push_back(relation);
      // Removal is a control at the last source time, never a fresh body
      // sample.
      ReconcileSourceRelationPhasesLocked(actor_id, remaining,
                                          actor_update_seconds_[actor_id]);
    }
    // A removed element that was a candidate changes the family's set. The
    // scope argument is not a root, so this does not depend on one.
    bool candidate_removed = false;
    for (auto& binding : bindings) {
      for (const auto& match : binding.latest_matches) {
        if (removed.count(match.element_id) != 0) {
          candidate_removed = true;
          break;
        }
      }
      binding.latest_matches.erase(
          std::remove_if(binding.latest_matches.begin(),
                         binding.latest_matches.end(),
                         [&](const Binding::MatchResult& match) {
                           return removed.count(match.element_id) != 0;
                         }),
          binding.latest_matches.end());
      for (element_id_t id : removed) {
        binding.sent_relations.erase(id);
        binding.smoothed_progress_rate.erase(id);
        binding.last_progress_sample_seconds.erase(id);
        binding.has_progress_sample.erase(id);
      }
    }
    if (candidate_removed) anchor_observation_dirty_.insert(actor_id);
  }
}

void Space::PreserveIndependentChildren(element_id_t removed_root) {
  const Element* removed = element_tree_.GetElement(removed_root);
  if (removed == nullptr) return;
  const element_id_t new_parent = removed->GetParentId();
  const auto children = element_tree_.GetDirectChildIds(removed_root);
  for (element_id_t child_id : children) {
    const Eigen::Matrix4d world = element_tree_.GetWorldTransform(child_id);
    if (!element_tree_.ReparentElement(child_id, new_parent).ok()) continue;
    AdvanceSceneInputsLocked(true, true);
    Eigen::Matrix4d parent_world = Eigen::Matrix4d::Identity();
    if (new_parent != kInvalidElementId) {
      parent_world = element_tree_.GetWorldTransform(new_parent);
    }
    SetElementTransformLocked(child_id, parent_world.inverse() * world);
  }
}

namespace {
double ResolveDeadbandEnv(const char* variable, double fallback) {
  if (const char* value = std::getenv(variable)) {
    const double parsed = std::atof(value);
    if (parsed >= 0.0) return parsed;
  }
  return fallback;
}
}  // namespace

// Bootstrap boundary for the movement deadbands. Resolving them on first use
// let the first update observe whatever the process environment happened to
// hold at that moment, and made them untestable without environment plumbing.
void Space::ResolveBootstrapPolicy() {
  bootstrap_min_propagation_move_ = ResolveDeadbandEnv("SB_V2_MIN_MOVE", kDefaultMinMoveM);
  bootstrap_min_propagation_yaw_ = ResolveDeadbandEnv("SB_V2_MIN_YAW", kDefaultMinYawRad);
  min_propagation_move_ = bootstrap_min_propagation_move_;
  min_propagation_yaw_ = bootstrap_min_propagation_yaw_;
}

Space::Space() { ResolveBootstrapPolicy(); }

Space::Space(std::function<int64_t()> receiver_input_clock)
    : receiver_input_clock_(receiver_input_clock ? std::move(receiver_input_clock)
                                                 : ReceiverInputNowNs) {
  ResolveBootstrapPolicy();
}

spatialblend::entity_id_t Space::CreateActor(
    const std::vector<SubjectAnchor>& anchors) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  spatialblend::entity_id_t id = next_actor_id_++;
  auto actor = std::make_unique<Actor>(id);
  for (const auto& anchor : anchors) {
    actor->AddAnchor(anchor);
  }
  actors_[id] = std::move(actor);
  {
    std::lock_guard<std::mutex> mailbox_lock(mailboxes_mutex_);
    mailboxes_[id] = std::make_shared<ActorMailbox>();
  }
  return id;
}

bool Space::DestroyActor(spatialblend::entity_id_t actor_id) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto* actor = FindActorLocked(actor_id);
  if (!actor) return false;
  auto* element = element_tree_.GetElement(actor->GetElementId());
  if (element && element->GetGroundingEntityId() == actor_id)
    element->SetGroundingEntityId(spatialblend::kInvalidEntityId);
  actors_.erase(actor_id);
  actor_body_radius_.erase(actor_id);
  observed_facing_dependencies_.erase(actor_id);
  PruneObservedFacingPointsLocked();
  receiver_inputs_.erase(actor_id);
  receiver_approach_contexts_.erase(actor_id);
  receiver_solve_inputs_.erase(actor_id);
  retained_receiver_evaluations_.erase(actor_id);
  rich_realization_actors_.erase(actor_id);
  // Destruction removes the external input mode with the actor.
  AdvanceSceneInputsLocked(true, false);
  phase_tracker_.ClearOwner(PhaseOwner(actor_id));
  source_relations_.erase(actor_id);
  source_evidence_enabled_.erase(actor_id);
  source_evidence_phase_dirty_.erase(actor_id);
  source_evidence_.erase(actor_id);
  borrowed_binding_actors_.erase(actor_id);
  bindings_.erase(actor_id);
  actor_containment_.erase(actor_id);
  realization_results_.erase(actor_id);
  placed_solve_.erase(actor_id);
  for (auto it = yaw_track_.begin(); it != yaw_track_.end();)
    it = it->first.first == actor_id ? yaw_track_.erase(it) : std::next(it);
  realization_snapshots_.erase(actor_id);
  previous_weights_.erase(actor_id);
  solver_warm_states_.erase(actor_id);
  proposal_attribution_.erase(actor_id);
  pending_source_exception_.erase(actor_id);
  latency_trackers_.erase(actor_id);
  receiving_actors_.erase(actor_id);
  receiver_actor_seconds_.erase(actor_id);
  receiver_actor_updates_.erase(actor_id);
  motion_trackers_.erase(actor_id);
  actor_update_seconds_.erase(actor_id);
  observation_clocks_.erase(actor_id);
  last_observation_propagated_time_.erase(actor_id);
  has_observation_propagated_time_.erase(actor_id);
  {
    std::lock_guard<std::mutex> mailbox_lock(mailboxes_mutex_);
    mailboxes_.erase(actor_id);
  }
  heartbeat_counters_.erase(actor_id);
  quiet_since_send_.erase(actor_id);
  last_propagated_pose_.erase(actor_id);
  last_propagated_orientation_.erase(actor_id);
  source_orientation_.erase(actor_id);
  next_propagate_sequence_.erase(actor_id);
  anchor_observation_dirty_.erase(actor_id);
  // Watermarks identify remote sources, not a local receiver actor. Retaining
  // them preserves duplicate suppression for every surviving actor.
  return true;
}

Actor* Space::GetActor(spatialblend::entity_id_t actor_id) {
  scene_inputs_escaped_.store(true);
  return FindActorLocked(actor_id);
}

const Actor* Space::GetActor(spatialblend::entity_id_t actor_id) const {
  scene_inputs_escaped_.store(true);
  return FindActorLocked(actor_id);
}

Actor* Space::FindActorLocked(spatialblend::entity_id_t actor_id) {
  auto it = actors_.find(actor_id);
  return it != actors_.end() ? it->second.get() : nullptr;
}

const Actor* Space::FindActorLocked(spatialblend::entity_id_t actor_id) const {
  auto it = actors_.find(actor_id);
  return it != actors_.end() ? it->second.get() : nullptr;
}

namespace {
const std::string kNoTag;
}  // namespace

Binding* Space::Bind(spatialblend::entity_id_t actor_id,
                     const std::string& anchor_name,
                     RelationKernelType kernel_type,
                     const std::string& tag_filter) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  borrowed_binding_actors_.insert(actor_id);
  return CreateBindingLocked(actor_id, anchor_name, kernel_type,
                              kInvalidElementId, tag_filter);
}

Binding* Space::CreateBindingLocked(spatialblend::entity_id_t actor_id,
                                    const std::string& anchor_name,
                                    RelationKernelType kernel_type,
                                    element_id_t scope_root,
                                    const std::string& tag_filter) {
  // ABI 4.7: an anchor a registered embodiment declares serves only the families it declared. An anchor the
  // registration does not mention, and an actor without one, bind exactly as before.
  if (const auto mailbox = MailboxFor(actor_id)) {
    std::lock_guard<std::mutex> alock(mailbox->anchors_mutex);
    if (mailbox->embodiment) {
      if (const EmbodimentAnchor* declared = mailbox->embodiment->FindAnchor(anchor_name)) {
        const uint32_t bit = kernel_type == RelationKernelType::kAvatarSpacing ? kFamilySpacing
                             : kernel_type == RelationKernelType::kAvatarFacing ? kFamilyFacing
                                                                                : kFamilyAffordancePose;
        if (!(declared->families & bit)) return nullptr;
      }
    }
  }
  // Look up anchor DOF
  std::vector<StateType> dof = {StateType::kPosition2D,
                                StateType::kRotationYaw};
  auto* actor = FindActorLocked(actor_id);
  if (actor) {
    const SubjectAnchor* anchor = actor->GetAnchor(anchor_name);
    if (anchor) dof = anchor->dof;
  }

  // A tag is the AffordancePose binding (which
  // affordance kind an anchor binds to, e.g. pelvis -> Sittable). It never narrows
  // Spacing or Facing scope, which belongs to the family: for those the argument is
  // ignored, as scope_root is, and the binding carries no tag.
  const std::string& effective_tag =
      kernel_type == RelationKernelType::kAvatarAffordancePose ? tag_filter
                                                               : kNoTag;

  auto components = KernelFactory::Create(kernel_type, dof);
  if (!components.kernel) return nullptr;

  // This association is captured at bind time, while both the actor's declared
  // anchor chart and its native binding vector are under the Space lock.  The
  // source-evidence capture path must never recover it later by matching a
  // name, a kernel, or the order in which relations happened to be delivered.
  size_t anchor_ordinal = Binding::kInvalidOrdinal;
  if (actor) {
    const auto& anchors = actor->GetAnchors();
    for (size_t i = 0; i < anchors.size(); ++i) {
      if (anchors[i].name == anchor_name) {
        anchor_ordinal = i;
        break;
      }
    }
  }
  auto& actor_bindings = bindings_[actor_id];
  Binding binding(anchor_name, kernel_type, effective_tag,
                  std::move(components.grounding_rule),
                  std::move(components.kernel), actor_bindings.size(),
                  anchor_ordinal);
  // Scope comes from the relation family. A caller-supplied root is ignored.
  (void)scope_root;
  binding.explicit_scope_root = kInvalidElementId;
  binding.revision = next_binding_revision_++;
  source_evidence_.erase(actor_id);
  actor_bindings.push_back(std::move(binding));
  return &actor_bindings.back();
}

Binding* Space::BindScoped(spatialblend::entity_id_t actor_id,
                           const std::string& anchor_name,
                           RelationKernelType kernel_type,
                           element_id_t scope_root,
                           const std::string& tag_filter) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (kernel_type != RelationKernelType::kAvatarSpacing &&
      kernel_type != RelationKernelType::kAvatarFacing &&
      kernel_type != RelationKernelType::kAvatarAffordancePose) {
    return nullptr;
  }
  Actor* actor = FindActorLocked(actor_id);
  if (!actor || !actor->GetAnchor(anchor_name) ||
      scope_root == kInvalidElementId ||
      !element_tree_.GetElement(scope_root)) {
    return nullptr;
  }
  // scope_root is checked so a missing element fails the call. It does not
  // select candidates: Spacing is the containment tree plus its border,
  // Facing is every visible referent, AffordancePose stays proximal.
  borrowed_binding_actors_.insert(actor_id);
  return CreateBindingLocked(actor_id, anchor_name, kernel_type,
                              kInvalidElementId, tag_filter);
}

bool Space::BindOwned(spatialblend::entity_id_t actor_id,
                       const std::string& anchor_name,
                       RelationKernelType kernel_type, element_id_t scope_root,
                       const std::string& tag_filter) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  const auto* actor = FindActorLocked(actor_id);
  if (!actor || !actor->GetAnchor(anchor_name) ||
      (scope_root != kInvalidElementId && !element_tree_.GetElement(scope_root)))
    return false;
  return CreateBindingLocked(actor_id, anchor_name, kernel_type, scope_root,
                              tag_filter) != nullptr;
}

const std::vector<Binding>& Space::GetBindings(
    spatialblend::entity_id_t actor_id) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  // A const Binding still contains mutable kernel/rule pointers.
  borrowed_binding_actors_.insert(actor_id);
  auto it = bindings_.find(actor_id);
  return it != bindings_.end() ? it->second : kEmptyBindings;
}

shared_element_id_t Space::GetActorSharedIdForEvents(
    spatialblend::entity_id_t actor_id) const {
  auto it = actors_.find(actor_id);
  if (it == actors_.end()) return kInvalidSharedElementId;
  element_id_t actor_elem_id = it->second->GetElementId();
  const Element* actor_elem = element_tree_.GetElement(actor_elem_id);
  return (actor_elem && actor_elem->GetSharedId() != kInvalidSharedElementId)
             ? actor_elem->GetSharedId()
             : static_cast<shared_element_id_t>(actor_elem_id);
}

void Space::UpdateActorState(spatialblend::entity_id_t actor_id,
                             const Eigen::Vector2d& position, double yaw) {
  UpdateActorState(actor_id, position, yaw, SourceOrientation{});
}

void Space::UpdateActorState(spatialblend::entity_id_t actor_id,
                             const Eigen::Vector2d& position, double yaw,
                             const SourceOrientation& orientation) {
  std::vector<PendingPropagation> pending;
  {
    std::unique_lock<std::mutex> lock(space_mutex_);
    if (receiver_inputs_.count(actor_id)) return;
    source_orientation_[actor_id] = orientation;
    InvalidateSourceEvidencePhaseLocked(actor_id);
    EvaluateActorStateLocked(actor_id, position, yaw, false, 0.0,
                             update_interval_, &pending);
    // Hand off from the state guard to the delivery guard while still holding
    // the former, then release it before dispatching. Propagate runs the
    // receiving space's OnRemoteUpdate under that space's own mutex.
    std::unique_lock<std::mutex> delivery(propagation_mutex_);
    lock.unlock();
    DeliverPropagations(pending);
  }
}

void Space::DeliverPropagations(
    const std::vector<PendingPropagation>& pending_list) {
  for (const PendingPropagation& pending : pending_list) {
  if (!pending.has_send || pending.hub == nullptr) continue;
  pending.hub->Propagate(pending.update);

  // SPEC S3 ev=src: the source actor's propagated pose, one line per
  // propagated update. It follows the send it describes, as it did inline.
  if (TraceSink::ForSpace(pending.space_id).Active()) {
    nlohmann::ordered_json ev;
    ev["ev"] = "src";
    ev["t"] = pending.clock;
    ev["space"] = pending.space_id;
    ev["actor"] = pending.update.actor_shared_id;
    ev["pos"] = {pending.position.x(), pending.position.y()};
    ev["yaw"] = pending.yaw;
    TraceSink::ForSpace(pending.space_id).Emit(ev);
  }
  }
}

void Space::EvaluateActorStateLocked(spatialblend::entity_id_t actor_id,
                                     const Eigen::Vector2d& position,
                                     double yaw, bool timestamped_observation,
                                     double source_time, double elapsed_seconds,
                                     std::vector<PendingPropagation>* pending,
                                     bool source_observation_sequence_known) {
  SyncEmbodiedMirrorsLocked();
  auto* actor = FindActorLocked(actor_id);
  if (!actor) return;
  // This tick counts as quiet only if the movement gate below skips it
  // for the deadband alone (see quiet_since_send_).
  const bool quiet_before_tick = quiet_since_send_[actor_id];
  quiet_since_send_[actor_id] = false;
  PruneObservedFacingPointsLocked();
  SyncNavigableObstaclesLocked();

  if (timestamped_observation) {
    // A timestamp is an actor-local input clock. The diagnostic space clock is
    // only a monotonic watermark; interleaved actors cannot add time to it.
    space_clock_ = std::max(space_clock_, source_time);
    actor_update_seconds_[actor_id] = source_time;
    if (actor->IsAnchorActive(observation_clocks_[actor_id].root)) {
      motion_trackers_[actor_id].Update(position, yaw, elapsed_seconds);
    }
  } else {
    // Preserve the legacy call-timed path unchanged.
    space_clock_ += update_interval_;
    actor_update_seconds_[actor_id] += update_interval_;
    motion_trackers_[actor_id].Update(position, yaw, update_interval_);
  }

  // Set actor transform
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform(0, 3) = position.x();
  transform(1, 3) = position.y();
  double cos_yaw = std::cos(yaw);
  double sin_yaw = std::sin(yaw);
  transform(0, 0) = cos_yaw;
  transform(0, 1) = -sin_yaw;
  transform(1, 0) = sin_yaw;
  transform(1, 1) = cos_yaw;
  actor->SetTransform(transform);

  // Mirror the pose onto the actor's element when the actor occupies a leaf
  // element (one without its own spatial region — e.g. a user/avatar element,
  // not a room). This keeps the element's position current so that *other*
  // actors evaluate relations against this actor's actual position rather
  // than the element's origin. Rooms are containers the actor moves within,
  // so elements that define a spatial region are left in place.
  {
    element_id_t aeid = actor->GetElementId();
    Element* ae = element_tree_.GetElement(aeid);
    if (!receiver_inputs_.count(actor_id) && ae && !ae->GetSpatialDefinition() &&
        element_tree_.GetDirectChildIds(aeid).empty()) {
      SetElementTransformLocked(aeid, transform, /*actor_pose=*/true);
    }
  }

  // Containment transfer: the identity of the flagged element that contains the
  // root (nullopt otherwise). Evaluated for every update so the contained state
  // does not depend on bindings or a hub.
  bool containment_transitioned = false;
  const std::optional<ContainmentObservation> containment_obs =
      UpdateSourceContainmentLocked(actor_id, position,
                                    &containment_transitioned);
  const std::string observed_root_anchor =
      observation_clocks_.count(actor_id) ? observation_clocks_[actor_id].root
                                          : std::string();
  const auto is_root_anchor = [&](const std::string& name) {
    return name.empty() || name == "root" ||
           (!observed_root_anchor.empty() && name == observed_root_anchor);
  };

  // Evaluate bindings (new Bind API)
  auto bind_it = bindings_.find(actor_id);
  if (bind_it == bindings_.end()) {
  }
  if (bind_it != bindings_.end()) {
    // Skip binding evaluation if the actor has no element assigned.
    element_id_t actor_elem_id = actor->GetElementId();
    if (actor_elem_id == kInvalidElementId) return;

    // Every actor-owned element. The subject's own element is not a referent.
    // Another actor is the collaborator, a referent the way v1 treated another
    // person: Facing toward that person is a relation.
    std::unordered_set<element_id_t> actor_element_ids;
    for (const auto& [other_id, other_actor] : actors_) {
      element_id_t oeid = other_actor->GetElementId();
      if (oeid != kInvalidElementId) actor_element_ids.insert(oeid);
    }

    // Stable actor identity for relation events: shared id when the actor's
    // element is shared, otherwise the local element id (single-space setups).
    shared_element_id_t actor_shared_for_events =
        GetActorSharedIdForEvents(actor_id);

    // Every scope descent starts at the tree's navigable root. The
    // actor's element is its person leaf, wherever it hangs: it does not start or bound a descent.
    const element_id_t scope_root =
        ScopeResolver::NavigableRoot(element_tree_, actor_elem_id);

    bool scoped_snapshot_lost = false;
    // ABI 4.14: each active bound anchor's pose in the transfer leaf holding it (one per anchor name).
    std::vector<AnchorChartObservation> anchor_charts;
    std::vector<SourceRelationObservation> source_relations;
    // Scope records are captured only while someone observes
    // (tracing on, or an observability getter enabled capture), so an
    // unobserved tick does no extra work.
    const bool capture = ObservabilityCaptureActiveLocked();
    // Facing occluders, built lazily once for this evaluation.
    std::optional<OcclusionScene> occlusion;
    // Lifecycle rows for a binding's matches (source side).
    // Weight is the in-group proximity share so a grouped set of
    // candidate referents sums to 1.
    const auto append_source_rows = [&](const Binding& binding) {
      double proximity_sum = 0.0;
      for (const auto& m : binding.latest_matches) {
        proximity_sum += m.eval.grounded.proximity;
      }
      for (const auto& m : binding.latest_matches) {
        const Element* ref_elem = element_tree_.GetElement(m.element_id);
        if (!ref_elem) continue;
        RelationEvent id;
        id.actor_shared_id = actor_shared_for_events;
        id.anchor_name = binding.anchor_name;
        id.referent_shared_id =
            (ref_elem->GetSharedId() != kInvalidSharedElementId)
                ? ref_elem->GetSharedId()
                : static_cast<shared_element_id_t>(m.element_id);
        id.kernel_type = binding.kernel_type;
        double proximity = m.eval.grounded.proximity;
        double weight = proximity_sum > 1e-12 ? proximity / proximity_sum : 0.0;
        source_relations.push_back({m.element_id, id, proximity, weight});
      }
      return proximity_sum;
    };
    const double actor_time = actor_update_seconds_[actor_id];
    for (auto& binding : bind_it->second) {
      binding.latest_matches.clear();
      // Observation only -- recorded at each stage below, never read.
      ScopeRecord& scope_record = binding.scope_record;
      scope_record.Reset();
      scope_record.valid = capture;
      scope_record.time_seconds =
          timestamped_observation ? source_time : space_clock_;
      auto scope_add = [&](element_id_t id, ScopeReason reason) {
        if (capture) scope_record.Add(id, reason, SharedOrLocalIdLocked(id));
      };

      if (!actor->IsAnchorActive(binding.anchor_name)) {
        // An inactive anchor (explicit, or omitted by the observation frame) is
        // an authoritative omission at once: there is no tracking-gap dwell.
        // Leaving it out produces an empty
        // snapshot when it was the only binding, releasing receiver targets.
        // An inactive observation begins a new relation sample lifetime when
        // reactivated. Do not derive a rate across its absence.
        binding.sent_relations.clear();
        binding.sent_anchor_position.reset();
        binding.sent_anchor_yaw.reset();
        binding.smoothed_progress_rate.clear();
        binding.last_progress_sample_seconds.clear();
        binding.has_progress_sample.clear();
        continue;
      }

      // Each binding owns its scope pose. A tracked hand can be in a different
      // region than the root, so containment, navigation, broadphase and
      // candidate enumeration must all use this exact anchor rather than a
      // root pose subsequently overridden only for Encode().
      Eigen::Vector3d anchor_pos(position.x(), position.y(), 0.0);
      double anchor_yaw = yaw;
      const SubjectAnchor* anchor = actor->GetAnchor(binding.anchor_name);
      const AnchorWorldPoseOverride* anchor_override =
          actor->GetAnchorWorldPoseOverride(binding.anchor_name);
      if (anchor_override) {
        anchor_pos = anchor_override->position;
        anchor_yaw = anchor_override->yaw;
      } else if (anchor) {
        anchor_pos = anchor->GetWorldPosition(actor->GetTransform());
      }
      // The transfer leaf holding this anchor's pose (the same descent as the root's transfer fact,
      // UpdateSourceContainmentLocked). It bounds the Facing scope (ExcludedFromFacingInsideTransfer) and gives the
      // anchor's chart observation (ABI 4.14): the anchor's OWN pose in the leaf's frame.
      element_id_t anchor_leaf = kInvalidElementId;
      if (containment_transfer_used_ && element_tree_.GetElement(scope_root))
        anchor_leaf = DescendScope(Eigen::Vector3d(anchor_pos.x(), anchor_pos.y(), 0.0), scope_root, element_tree_,
                                   /*choose_overlap=*/true, /*skip_transparent=*/true)
                          .transfer_leaf;
      if (const Element* leaf_el = anchor_leaf == kInvalidElementId ? nullptr : element_tree_.GetElement(anchor_leaf);
          leaf_el && leaf_el->GetSharedId() != kInvalidSharedElementId &&
          std::none_of(anchor_charts.begin(), anchor_charts.end(),
                       [&](const AnchorChartObservation& c) { return c.anchor_name == binding.anchor_name; })) {
        const Eigen::Matrix4d leaf_world = element_tree_.GetWorldTransform(anchor_leaf);
        const Eigen::Vector4d local = leaf_world.inverse() * Eigen::Vector4d(anchor_pos.x(), anchor_pos.y(),
                                                                             anchor_pos.z(), 1.0);
        AnchorChartObservation c;
        c.anchor_name = binding.anchor_name;
        c.scope_shared_id = leaf_el->GetSharedId();
        c.scope_source_space_id = space_id_;
        c.scope_source_element_id = anchor_leaf;
        c.x = local.x();
        c.y = local.y();
        c.z = local.z();
        c.yaw = std::remainder(anchor_yaw - WorldYawOf(leaf_world), 2.0 * M_PI);
        anchor_charts.push_back(std::move(c));
      }

      // Scope is the family's. Spacing descends the containment tree and
      // adds the container's border. Facing is global. AffordancePose is
      // the proximal subtree. An app-supplied root does not change this.
      const element_id_t selected_scope = scope_root;
      const bool scope_is_live =
          element_tree_.GetElement(selected_scope) != nullptr;
      // The containment tree membership of the anchor is ONE definition for every family
      // (relation/relation_kernel.h DescendScope): the deepest referent whose footprint
      // holds it, a flagged containment-transfer element being a LEAF that is the scope
      // whatever it carries below it. For Spacing the same
      // descent also decides the mode and the referents (ScopeResolver::ResolveSpacing).
      std::optional<SpacingScope> spacing_scope;
      if (scope_is_live && binding.kernel_type == RelationKernelType::kAvatarSpacing) {
        // Every other person in the same containment tree is a Spacing referent while
        // the anchor is outside every referent; inside one the people inside it stay in
        // its interior and the others drop out with the siblings.
        const std::vector<element_id_t> people(actor_element_ids.begin(),
                                               actor_element_ids.end());
        spacing_scope = ScopeResolver::ResolveSpacing(anchor_pos, selected_scope,
                                                      element_tree_, people, actor_elem_id);
      }
      element_id_t deepest_containing =
          !scope_is_live ? kInvalidElementId
          : spacing_scope ? spacing_scope->container
                          : FindDeepestContaining(anchor_pos, selected_scope, element_tree_);
      binding.spacing_scope = spacing_scope;
      const NavigableSpatialDefinition* navigable_def = nullptr;
      element_id_t navigable_element = kInvalidElementId;
      for (element_id_t e = deepest_containing; e != kInvalidElementId;) {
        const auto* scope_elem = element_tree_.GetElement(e);
        if (!scope_elem) break;
        if (scope_elem->HasGeometryRole(GeometryRole::kNavigable)) {
          navigable_def = dynamic_cast<const NavigableSpatialDefinition*>(
              scope_elem->GetSpatialDefinition());
          if (navigable_def) {
            navigable_element = e;
            break;
          }
        }
        e = scope_elem->GetParentId();
      }

      auto scoped_ids = !scope_is_live ? std::vector<element_id_t>{}
                        : spacing_scope ? spacing_scope->referents
                                        : binding.kernel->ResolveScope(
                                              anchor_pos, actor_elem_id,
                                              element_tree_, deepest_containing);
      const std::vector<element_id_t> raw_scope_ids =
          capture ? scoped_ids : std::vector<element_id_t>{};
      scope_record.anchor_active = true;
      scope_record.explicit_scope = false;
      scope_record.scope_live = scope_is_live;
      scope_record.anchor_position = anchor_pos;
      scope_record.anchor_yaw = anchor_yaw;
      scope_record.scope_root = selected_scope;
      scope_record.kernel_scope_origin = actor_elem_id;
      scope_record.deepest_containing = deepest_containing;
      scope_record.navigable_element = navigable_element;

      // The tag is the AffordancePose binding (which affordance kind the
      // anchor binds to). It never narrows Spacing or Facing scope.
      std::vector<element_id_t> filtered_ids;
      for (element_id_t eid : scoped_ids) {
        if (binding.tag_filter.empty()) {
          filtered_ids.push_back(eid);
          continue;
        }
        const Element* elem = element_tree_.GetElement(eid);
        if (elem && elem->GetTag() == binding.tag_filter) {
          filtered_ids.push_back(eid);
        } else {
          scope_add(eid, ScopeReason::kExcludedTag);
        }
      }
      for (element_id_t elem_id : filtered_ids) {
        // The subject's own element is not a referent. Its scene descendants
        // are: a room assigned as the actor element still contains objects.
        // A peer actor's element is the collaborator.
        if (elem_id == actor_elem_id) {
          scope_add(elem_id, ScopeReason::kExcludedActorElement);
          continue;
        }
        // Facing inside a transfer leaf: the leaf and what it carries are not look-at targets, its
        // Surface descendants are (ScopeResolver's rule).
        if (binding.kernel_type == RelationKernelType::kAvatarFacing &&
            ExcludedFromFacingInsideTransfer(element_tree_, anchor_leaf, elem_id)) {
          scope_add(elem_id, ScopeReason::kExcludedInsideTransferLeaf);
          continue;
        }
        // A legacy update/control must not re-publish held/generated subject
        // state as new observed-point intent. Only actual source frames use
        // these referents; controls may still withdraw their previous matches.
        if (observed_facing_points_.count(elem_id) &&
            (!timestamped_observation || !source_observation_sequence_known)) {
          scope_add(elem_id, ScopeReason::kExcludedObservedPointControl);
          continue;
        }
        if (!IsObservedFacingCandidateLocked(elem_id, actor_id,
                                               binding.kernel_type,
                                               receiver_input_clock_())) {
          scope_add(elem_id, ScopeReason::kExcludedObservedPointIneligible);
          continue;
        }

        const auto* elem = element_tree_.GetElement(elem_id);
        if (!elem) {
          scope_add(elem_id, ScopeReason::kExcludedMissingElement);
          continue;
        }
        // An affordance pose's interaction child is its OBJECT's inner scope (the poses are related through the object,
        // below), never a contact referent of its own.
        if (binding.kernel_type == RelationKernelType::kAvatarAffordancePose && elem->GetAffordancePose()) {
          scope_add(elem_id, ScopeReason::kExcludedTag);
          continue;
        }

        // Evaluate — pass navigable_def for geodesic distance
        if (auto* facing_rule = dynamic_cast<FacingGroundingRule*>(binding.grounding_rule.get()))
          facing_rule->SetPreviousAnchorPosition(binding.sent_anchor_position);
        Support support = binding.grounding_rule->ComputeSupport(
            *elem, element_tree_, anchor_pos, anchor_yaw, navigable_def);
        if (navigable_def) support.navigable_element_id = navigable_element;
        // The deepest container as a referent (ScopeResolver::ResolveReachable):
        // the relation to the border the actor is inside of. The scope is the
        // container's effective region, which its children extend
        // (element/scope_extension.h); the support is its declared one. A pose in
        // that extension is in the container's scope but outside the object: its
        // relation to the container is the ordinary outside one (the distance to
        // the declared edge), never the border's depth 0 (R = 1), which would put
        // the receiver on the edge.
        if (spacing_scope && spacing_scope->border && elem_id == deepest_containing &&
            support.has_interior)
          support.container_self = true;
        // Facing is global: one ray to the look-at contact. Spacing is the
        // containment tree plus every person; its gate is the source-F
        // geodesic path. AffordancePose is the parent object's region.
        if (binding.kernel_type == RelationKernelType::kAvatarFacing &&
            !occlusion) {
          occlusion.emplace(element_tree_, ActorBodyTopsLocked());  // once per evaluation
        }
        const ScopeGateVerdict verdict = FamilyScopeGateVerdict(
            binding.kernel_type, element_tree_, elem_id, anchor_pos, support,
            navigable_def, occlusion ? &*occlusion : nullptr);
        if (verdict != ScopeGateVerdict::kInScope) {
          scope_add(elem_id,
                    verdict == ScopeGateVerdict::kNotVisible
                        ? ScopeReason::kExcludedNotVisible
                    : verdict == ScopeGateVerdict::kOutsideParentRegion
                        ? ScopeReason::kExcludedOutsideParentRegion
                        : ScopeReason::kExcludedUnreachable);
          continue;
        }
        GroundedRelation grounded =
            binding.kernel->Encode(anchor_pos, anchor_yaw, support);

        if (capture) {
          scope_add(elem_id, ScopeReason::kIncluded);
          scope_record.candidates.back().proximity = grounded.proximity;
        }

        Binding::MatchResult match;
        match.element_id = elem_id;
        match.eval.grounded = std::move(grounded);
        match.eval.support = std::move(support);
        binding.latest_matches.push_back(std::move(match));

        // The inner level: room -> object SD -> affordance poses. While the anchor is inside the
        // object's SD region volume V, it is related to each of the object's affordance poses, instantiated for THIS
        // actor's embodiment, exactly as Spacing relates the root to the room's objects: d = the distance to the pose's
        // nearest point (its count n is the receiver's ratio continuation), and the CONTACT CHART of that nearest point
        // (the pose's part chart, or the root frame for the standing point) with the direction from it to the anchor:
        // the receiver's tangential freedom is that chart's image of the source pose, never pursuit history (the
        // contact chart, one level down). Outside V there is no inner relation (proximal scope).
        if (binding.kernel_type == RelationKernelType::kAvatarAffordancePose &&
            HasAffordancePoses(element_tree_, elem_id, binding.anchor_name)) {
          const SubjectAnchor* root_anchor = actor->GetAnchor("root");
          const Eigen::Vector3d root_world =
              root_anchor ? root_anchor->GetWorldPosition(actor->GetTransform())
                          : Eigen::Vector3d(position.x(), position.y(), 0.0);
          const std::shared_ptr<const Embodiment> embodiment = EmbodimentOf(actor_id);
          const auto sd = InstantiatePoseSD(element_tree_, elem_id, binding.anchor_name, embodiment.get(), root_world, yaw);
          if (sd.ok() && sd->Contains(anchor_pos)) {
            for (const PoseInstance& pose : sd->poses) {
              const auto contact = EncodePoseContact(pose.support, anchor_pos);
              if (!contact.ok()) continue;
              const double d = contact->distance;
              Binding::MatchResult pm;
              pm.element_id = pose.element;
              pm.pose_tag = pose.tag;
              pm.pose_object = elem_id;
              GroundedRelation& gr = pm.eval.grounded;
              // encoded_state: the direction from the contact to the anchor, in the support's own frame.
              gr.encoded_state = Eigen::VectorXd(3);
              gr.encoded_state << contact->dir.x(), contact->dir.y(), contact->dir.z();
              gr.support_distance_m = d;
              gr.contact_distance_m = d;
              gr.source_scope_diameter_m = sd->scale_m;
              gr.proximity = 1.0 - std::clamp(d / sd->scale_m, 0.0, 1.0);
              Support& sup = pm.eval.support;
              if (contact->on_part) {
                sup.has_contact = true;
                sup.contact_point_local = contact->local;
                sup.contact_normalized = contact->chart;
                sup.boundary = pose.support.boundary;
                sup.sub_region_name = pose.support.part;
                sup.sub_region_chart = static_cast<int32_t>(pose.support.chart);
              }
              if (capture) {
                scope_add(pose.element, ScopeReason::kIncluded);
                scope_record.candidates.back().proximity = gr.proximity;
              }
              binding.latest_matches.push_back(std::move(pm));
            }
          }
        }
      }

      // Report candidates in the family's scope order.
      if (capture && scope_record.candidates.size() > 1) {
        std::unordered_map<element_id_t, size_t> order;
        order.reserve(raw_scope_ids.size());
        for (size_t i = 0; i < raw_scope_ids.size(); ++i)
          order.emplace(raw_scope_ids[i], i);
        auto rank = [&](const ScopeCandidate& c) {
          auto it = order.find(c.element_id);
          return it != order.end() ? it->second : raw_scope_ids.size();
        };
        std::stable_sort(scope_record.candidates.begin(),
                         scope_record.candidates.end(),
                         [&](const ScopeCandidate& a, const ScopeCandidate& b) {
                           return rank(a) < rank(b);
                         });
      }

      const double proximity_sum = append_source_rows(binding);

      // Every in-scope relation is realized: there is no activation gate.
      (void)proximity_sum;

      binding.latest_anchor_position = anchor_pos;
      binding.latest_anchor_yaw = anchor_yaw;

      // The progress baseline (sent_relations) moves only when an update is sent, below.
      std::unordered_set<element_id_t> current_elements;
      for (const auto& m : binding.latest_matches) {
        current_elements.insert(m.element_id);
      }
      for (auto it = binding.smoothed_progress_rate.begin();
           it != binding.smoothed_progress_rate.end();) {
        if (current_elements.count(it->first) == 0) {
          binding.last_progress_sample_seconds.erase(it->first);
          binding.has_progress_sample.erase(it->first);
          it = binding.smoothed_progress_rate.erase(it);
        } else {
          ++it;
        }
      }
    }

    // Reconcile the entire actor snapshot first: one absent binding must not
    // reset another still-present binding's phase. This also works without a
    // Hub, where sent_relations is not populated (nothing is sent).
    const double phase_time =
        timestamped_observation ? source_time : space_clock_;
    ReconcileSourceRelationPhasesLocked(actor_id, source_relations, phase_time);
    // A clear/control may withdraw rows, but the surviving held body rows are
    // not new source evidence and must not advance an action lifecycle.
    if (!timestamped_observation || source_observation_sequence_known)
      for (const auto& relation : source_relations)
        EmitRelationTransitions(relation.identity, relation.proximity,
                                relation.weight, phase_time,
                                PhaseOwner(actor_id));

    // Auto-propagate through hub (with delta detection)
    if (hub_) {
      element_id_t actor_elem_id = actor->GetElementId();
      const Element* actor_elem = element_tree_.GetElement(actor_elem_id);
      // Identity is the actor element handle ({space, actor_elem_id}), resolved
      // on the receiver via the ElementLinker — a valid shared_id is no longer
      // required to propagate.
      if (actor_elem) {
        // --- Movement gate (mirrors v1 Space::ShouldProcessStateUpdate)
        // ------- Propagate relation updates only when the actor has actually
        // moved beyond a deadband since the pose we last propagated (compared
        // against the last *propagated* pose, so slow sub-deadband drift still
        // eventually fires) — or on the periodic heartbeat. A stationary VR
        // source jitters a few mm/frame; re-grounding and propagating that
        // jitter is what oscillated the clone. Gating on the SOURCE pose — not
        // on per-payload proximity/estimated_time deltas, which themselves
        // jitter — makes a still source emit ZERO updates, so the clone holds
        // its pose BY CONSTRUCTION rather than being smoothed after the fact.
        // The heartbeat still refreshes late joiners / dropped packets; its
        // identical payloads re-solve idempotently, so no jump.
        // Both deadbands were resolved at construction. A negative value can
        // only arrive from SetMovementGate, and it still means "reset to the
        // bootstrap value" — resolved from the retained copy, not the
        // environment, so no update path reads getenv.
        if (min_propagation_move_ < 0.0) {
          min_propagation_move_ = bootstrap_min_propagation_move_;
        }
        if (min_propagation_yaw_ < 0.0) {
          min_propagation_yaw_ = bootstrap_min_propagation_yaw_;
        }
        int& heartbeat = heartbeat_counters_[actor_id];
        heartbeat++;
        bool force_heartbeat = (heartbeat >= kHeartbeatInterval);
        if (timestamped_observation) {
          force_heartbeat =
              !has_observation_propagated_time_.count(actor_id) ||
              source_time - last_observation_propagated_time_[actor_id] >= 1.0;
        }
        bool moved = true;  // first propagation for this actor always sends
        auto lp_it = last_propagated_pose_.find(actor_id);
        if (lp_it != last_propagated_pose_.end()) {
          double dpos = (position - lp_it->second.first).norm();
          double dyaw =
              std::abs(std::remainder(yaw - lp_it->second.second, 2 * M_PI));
          moved =
              (dpos >= min_propagation_move_) || (dyaw >= min_propagation_yaw_);
        }
        SourceOrientation orientation_now;
        if (const auto oi = source_orientation_.find(actor_id);
            oi != source_orientation_.end())
          orientation_now = oi->second;
        // A pitch/roll change alone is a reason to send (the deadband is a
        // position/yaw movement gate).
        if (orientation_now != last_propagated_orientation_[actor_id]) moved = true;

        RemoteRelationUpdate update;
        update.source_space_id = space_id_;
        update.timing = timestamped_observation
                            ? RelationUpdateTiming::kTimestampedObservation
                            : RelationUpdateTiming::kLegacy;
        if (timestamped_observation) {
          update.source_epoch = observation_clocks_[actor_id].epoch;
          update.source_time = source_time;
          // Only ApplyActorObservationFrame commits a new canonical sample.
          // Anchor clears carry epoch/time control context but must not reuse
          // the last sample sequence as though they were an observation.
          if (source_observation_sequence_known) {
            update.source_observation_sequence =
                observation_clocks_[actor_id].sequence;
            update.source_observation_sequence_known = true;
          }
        }
        update.actor_shared_id = actor_elem->GetSharedId();
        // Source actor handle for linker-based resolution on the receiver.
        update.actor_source_space_id = space_id_;
        update.actor_source_element_id = actor_elem_id;
        update.step_duration = update_interval_;
        update.source_orientation_type = orientation_now.type;
        update.source_pitch = orientation_now.pitch;
        update.source_roll = orientation_now.roll;
        // THIS ACTOR's own elapsed source time as of this call (advanced by its own ticks at the top of UpdateActorState,
        // or its timestamp) -- lets the receiver compute the TRUE real-time gap since its last delivery for this actor, instead
        // of assuming every delivery is exactly update_interval_ apart (false across a movement-gate hold resolved only by the
        // heartbeat, up to ~1.0s later). See remote_update.h's `source_clock` doc. Not space_clock_: that diagnostic clock also
        // advances by the deltas of the updates this space RECEIVES, so between two peers that both send, each stamp carried
        // the other's increments back and both clocks grew geometrically (a source clock of 22706 s after 96 s),
        // which ran the receiver's 5 s latency limit out within frames and teleported its avatar at every new engagement.
        update.source_clock = actor_time;
        update.containment = containment_obs;
        update.anchor_charts = anchor_charts;


        // The relations of this tick's payloads, one per payload: the progress baseline of the
        // next update if this one is sent.
        std::vector<std::pair<Binding*, const Binding::MatchResult*>> sent_now;
        for (auto& binding : bind_it->second) {
          const auto& sent_rows = binding.latest_matches;
          for (const auto& match : sent_rows) {
            const auto* ref_elem = element_tree_.GetElement(match.element_id);
            if (!ref_elem) {
              if (capture)
                binding.scope_record.Mark(match.element_id,
                                          ScopeReason::kExcludedMissingElement);
              continue;
            }
            // Identity is the referent handle ({space, match.element_id}),
            // resolved on the receiver via the ElementLinker; a valid shared_id
            // is no longer required to reference a referent.

            // An inner affordance-pose relation names its OBJECT as the referent, and the pose by its tag.
            if (!match.pose_tag.empty()) {
              ref_elem = element_tree_.GetElement(match.pose_object);
              if (!ref_elem) continue;
            }
            RelationPayload payload;
            payload.referent_shared_id = ref_elem->GetSharedId();
            payload.affordance_pose_tag = match.pose_tag;
            payload.subject_anchor_name = binding.anchor_name;
            // The subject anchor's orientation beyond yaw: the root's is the actor's
            // declared orientation, another anchor's is its own tracked one (a bone-bound anchor), else yaw only.
            payload.has_subject_orientation = true;
            if (is_root_anchor(binding.anchor_name)) {
              if (const auto oi = source_orientation_.find(actor_id); oi != source_orientation_.end())
                payload.subject_orientation = oi->second;
            } else if (const AnchorWorldPoseOverride* o = actor->GetAnchorWorldPoseOverride(binding.anchor_name)) {
              payload.subject_orientation.type = o->orientation_type;
              payload.subject_orientation.pitch = o->pitch;
              payload.subject_orientation.roll = o->roll;
            }
            payload.grounded_relation = match.eval.grounded;
            payload.support = match.eval.support;
            payload.kernel_type = binding.kernel_type;
            // The binding identity + revision. A binding exposed as a
            // mutable pointer (Bind/BindScoped/GetBindings) may have been
            // edited after creation, so its revision is not certified.
            payload.source_binding_ordinal = binding.binding_ordinal;
            payload.binding_revision = binding.revision;
            payload.binding_revision_known =
                binding.revision != 0 &&
                binding.binding_ordinal != Binding::kInvalidOrdinal &&
                borrowed_binding_actors_.count(actor_id) == 0;
            const auto observed = observed_facing_points_.find(match.element_id);
            if (observed != observed_facing_points_.end()) {
              if (!IsObservedFacingPointLiveLocked(match.element_id,
                                                    receiver_input_clock_())) {
                if (capture)
                  binding.scope_record.Mark(
                      match.element_id,
                      ScopeReason::kExcludedObservedPointExpired);
                continue;
              }
              payload.observed_facing_point = true;
              payload.observed_logical_actor = observed->second.logical_actor;
              payload.observed_source_point = observed->second.source_point;
              payload.observed_source_point_lease = observed->second.source_point_lease;
              payload.observed_valid_until_ns = observed->second.valid_until_ns;
            }
            // Carry the source-side referent handle so the receiver
            // can resolve via ElementLinker (SB_V2_REFERENT_RESOLVE=linker).
            payload.referent_source_space_id = space_id_;
            payload.referent_source_element_id = match.pose_tag.empty() ? match.element_id : match.pose_object;
            // Identity of the scope that normalized R.
            if (const Element* scope_el = element_tree_.GetElement(
                    match.eval.support.navigable_element_id)) {
              payload.scope_source_space_id = space_id_;
              payload.scope_source_element_id = scope_el->GetId();
              payload.scope_shared_id = scope_el->GetSharedId();
            }

            // Compute satisfaction metric from kernel + actor motion
            const GroundedRelation* prev = nullptr;
            auto prev_it = binding.sent_relations.find(match.element_id);
            if (prev_it != binding.sent_relations.end()) {
              prev = &prev_it->second;
            }
            if (payload.kernel_type == RelationKernelType::kAvatarFacing &&
                match.eval.grounded.encoded_state.size() > 0) {
              // v1 Angular1D: the count of updates to alignment, from the
              // bearing offset's own progress. Progress is the rotation toward ONE point: the
              // previous offset is the CURRENT look-at contact's, seen from the previous sent
              // pose (the kernel's own encoding of this support at that pose), not the sent
              // offset, which was measured to the look-at contact of that time. When the
              // contact moved to another vertex since the send, the two sent offsets are to
              // two points and their difference is no rotation of the body (a spin through the
              // vertex switch read as opening when it closes).
              const auto& now_es = match.eval.grounded.encoded_state;
              const bool has_prev = prev && prev->encoded_state.size() == now_es.size() &&
                                    binding.sent_anchor_position && binding.sent_anchor_yaw;
              const GroundedRelation before_state =
                  has_prev ? binding.kernel->Encode(*binding.sent_anchor_position,
                                                    *binding.sent_anchor_yaw, match.eval.support)
                           : match.eval.grounded;
              const double now = now_es(0);
              const double before = before_state.encoded_state(0);
              // The look-at distance is the last component of the Facing state (bearing[, elevation], distance).
              const Eigen::Index last = now_es.size() - 1;
              const double distance_before = last >= 1 ? before_state.encoded_state(last) : 1.0;
              payload.update_count = FacingUpdateCount(before, now, distance_before);
              payload.facing_rotation_sign = FacingRotationSign(before, now);
              payload.facing_opening = FacingOpening(before, now, distance_before);
            } else {
              payload.update_count = SourceUpdateCount(
                  prev ? prev->support_distance_m
                       : match.eval.grounded.support_distance_m,
                  match.eval.grounded.support_distance_m);
            }
            // Scheduling reads update_count (v1 SoftAnnealing). The source
            // does not write a κ-clipped time-to-contact into estimated_time.

            // The baseline for the next update's progress is this relation, but only once
            // this update is actually sent (below): a gated tick sends nothing.
            sent_now.emplace_back(&binding, &match);

            update.payloads.push_back(std::move(payload));
          }
        }

        const bool anchor_observation_changed =
            anchor_observation_dirty_.count(actor_id) != 0;
        // Propagate an authoritative empty snapshot for an observed-anchor
        // transition too; normal root movement with no relations remains quiet.
        // When the last in-scope relation leaves scope, send one
        // authoritative empty snapshot so the receiver releases its target
        // instead of holding the stale one.
        const bool snapshot_emptied = update.payloads.empty() &&
                                      nonempty_snapshot_sent_.count(actor_id);
        // A contained root, or the transition out of containment, is itself
        // something to send even when every scene-level relation is dormant:
        // the receiver learns the contained pose, or that it ended.
        // A source-app exception is itself a reason to send: it goes out
        // through the movement gate and the heartbeat, with the relations it
        // describes (an empty snapshot would release the receiver's targets, so
        // the exception never turns a no-relation tick into one).
        const bool exception_pending =
            pending_source_exception_.count(actor_id) != 0;
        bool will_propagate =
            (timestamped_observation || moved || force_heartbeat ||
             anchor_observation_changed || scoped_snapshot_lost ||
             snapshot_emptied || containment_transitioned ||
             exception_pending) &&
            (!update.payloads.empty() || anchor_observation_changed ||
             scoped_snapshot_lost || snapshot_emptied ||
             containment_obs.has_value() || containment_transitioned ||
             exception_pending);
        if (will_propagate) {
          // Stamp a per-actor monotonically increasing delivery counter
          // right before the update actually leaves this space, so a
          // receiver (Space::OnRemoteUpdate) can recognize an exact-replay
          // delivery of THIS update (see remote_update.h's `sequence` doc).
          // Only assigned here (not earlier, unconditionally) so a
          // movement-gated/heartbeat-skipped tick that never propagates
          // doesn't burn a sequence number.
          update.sequence = ++next_propagate_sequence_[actor_id];
          // The progress baseline is the state at the last SENT update (v1: the last accepted
          // state): n of the next update is the progress made since this one, however many
          // gated ticks lie between. Every binding that was evaluated holds exactly the
          // relations of this update (an absent referent leaves no baseline).
          for (auto& binding : bind_it->second) {
            if (!actor->IsAnchorActive(binding.anchor_name)) continue;
            binding.sent_relations.clear();
            binding.sent_anchor_position = binding.latest_anchor_position;
            binding.sent_anchor_yaw = binding.latest_anchor_yaw;
          }
          for (const auto& [binding, match] : sent_now)
            binding->sent_relations[match->element_id] = match->eval.grounded;
          // A declared teleport/reset of this source rides the update it
          // describes (a source-side event, replacing the 1 m jump test).
          // A source-app exception rides the update it describes. The first
          // update a source actor propagates is its join, itself an exception.
          if (const auto pending = pending_source_exception_.find(actor_id);
              pending != pending_source_exception_.end()) {
            update.has_exception = true;
            update.exception_id = pending->second.id;
            update.exception_payload = std::move(pending->second.payload);
            pending_source_exception_.erase(pending);
          } else if (update.sequence == 1) {
            update.has_exception = true;
            update.exception_id = kExceptionJoin;
          }
          // Assert the interval this send closes was a pure deadband
          // hold, closed by a pure heartbeat: every skipped tick since the
          // last send was a deadband skip, and this send itself is due to
          // the heartbeat alone (nothing moved or changed on this tick).
          // Never for a timestamped sample, which is sent every tick.
          update.quiet_since_last_send =
              quiet_before_tick && !timestamped_observation && !moved &&
              !anchor_observation_changed && !scoped_snapshot_lost &&
              !snapshot_emptied && !containment_transitioned;
          quiet_since_send_[actor_id] = !timestamped_observation;
          heartbeat = 0;
          if (update.payloads.empty()) {
            nonempty_snapshot_sent_.erase(actor_id);
          } else {
            nonempty_snapshot_sent_.insert(actor_id);
          }
          if (timestamped_observation) {
            last_observation_propagated_time_[actor_id] = source_time;
            has_observation_propagated_time_.insert(actor_id);
          }
          last_propagated_pose_[actor_id] = {position, yaw};
          last_propagated_orientation_[actor_id] = orientation_now;
          anchor_observation_dirty_.erase(actor_id);

          // Staged last, after every local mutation this branch makes, so the
          // published state is complete before a peer can observe it. The
          // trace inputs are captured here because they are read outside the
          // guard, where another caller may already have moved them on.
          if (pending != nullptr) {
            PendingPropagation staged;
            staged.hub = hub_;
            staged.space_id = space_id_;
            staged.clock = space_clock_;
            staged.position = position;
            staged.yaw = yaw;
            staged.update = std::move(update);
            staged.has_send = true;
            pending->push_back(std::move(staged));
          }
        } else if (!timestamped_observation && !moved && !force_heartbeat &&
                   !anchor_observation_changed && !scoped_snapshot_lost &&
                   !snapshot_emptied && !update.payloads.empty()) {
          // Skipped by the movement deadband alone, with relations to send:
          // the quiet interval since the last send continues.
          quiet_since_send_[actor_id] = quiet_before_tick;
        }
      }
    }

    // ev=scope: each active binding's raw scope and every candidate's
    // final outcome, written after payload build so late drops
    // (observed_point_expired) are included.
    if (capture && TraceSink::ForSpace(space_id_).Active()) {
      auto& sink = TraceSink::ForSpace(space_id_);
      for (const auto& binding : bind_it->second) {
        const ScopeRecord& record = binding.scope_record;
        if (!record.valid || !record.anchor_active) continue;
        EmitPolicyTraceIfChangedLocked(record.time_seconds);
        nlohmann::ordered_json ev;
        ev["ev"] = "scope";
        ev["t"] = record.time_seconds;
        ev["space"] = space_id_;
        ev["actor"] = actor_shared_for_events;
        ev["anchor"] =
            binding.anchor_name.empty() ? "root" : binding.anchor_name;
        ev["kernel"] = static_cast<int>(binding.kernel_type);
        ev["explicit"] = record.explicit_scope;
        ev["scope_root"] = record.scope_root;
        ev["deepest"] = record.deepest_containing;
        ev["navigable"] = record.navigable_element;
        nlohmann::ordered_json included = nlohmann::ordered_json::array();
        nlohmann::ordered_json excluded = nlohmann::ordered_json::array();
        for (const auto& c : record.candidates) {
          if (c.reason == ScopeReason::kIncluded) {
            included.push_back(c.shared_or_local_id);
          } else {
            excluded.push_back({c.shared_or_local_id, ScopeReasonName(c.reason)});
          }
        }
        ev["included"] = std::move(included);
        ev["excluded"] = std::move(excluded);
        sink.Emit(ev);
      }
    }
  }
}

void Space::ResetObservationStateLocked(spatialblend::entity_id_t actor_id) {
  Actor* actor = FindActorLocked(actor_id);
  if (!actor) return;
  phase_tracker_.ClearOwner(PhaseOwner(actor_id));
  source_relations_.erase(actor_id);
  source_evidence_enabled_.erase(actor_id);
  source_evidence_phase_dirty_.erase(actor_id);
  source_evidence_.erase(actor_id);
  for (const SubjectAnchor& anchor : actor->GetAnchors()) {
    actor->ClearAnchorWorldPose(anchor.name);
    actor->SetAnchorActive(anchor.name, false);
  }
  if (auto bindings_it = bindings_.find(actor_id);
      bindings_it != bindings_.end()) {
    for (Binding& binding : bindings_it->second) {
      binding.latest_matches.clear();
      binding.sent_relations.clear();
      binding.smoothed_progress_rate.clear();
      binding.last_progress_sample_seconds.clear();
      binding.has_progress_sample.clear();
    }
  }
  previous_weights_.erase(actor_id);
  solver_warm_states_.erase(actor_id);
  ClearActorContainmentLocked(actor_id);
  // Reset is an authoritative cancellation, not an unknown result. Keep a
  // known empty snapshot so downstream dispatch can call managed controllers
  // with an empty target list instead of silently retaining their old target.
  RealizationResult released;
  released.status = SolverStatus::kConverged;
  realization_results_[actor_id] = std::move(released);
  CaptureRealizationSnapshot(actor_id, {}, {}, {}, {},
                             realization_results_[actor_id],
                             CaptureSceneInputStampLocked());
  proposal_attribution_.erase(actor_id);
  {
    ReceiverProposal released_proposal;
    released_proposal.status = ProposalStatus::kNone;
    PublishProposalLocked(actor_id, std::move(released_proposal));
  }
  receiver_actor_seconds_.erase(actor_id);
  latency_trackers_.erase(actor_id);
  receiver_actor_updates_.erase(actor_id);
  motion_trackers_.erase(actor_id);
  actor_update_seconds_.erase(actor_id);
  heartbeat_counters_.erase(actor_id);
  quiet_since_send_.erase(actor_id);
  last_propagated_pose_.erase(actor_id);
  anchor_observation_dirty_.erase(actor_id);
  last_observation_propagated_time_.erase(actor_id);
  has_observation_propagated_time_.erase(actor_id);
}

bool Space::BeginActorObservationEpoch(spatialblend::entity_id_t actor_id,
                                       const std::string& root_anchor,
                                       uint64_t* epoch) {
  if (!epoch) return false;
  std::unique_lock<std::mutex> lock(space_mutex_);
  if (receiver_inputs_.count(actor_id)) return false;
  Actor* actor = FindActorLocked(actor_id);
  const SubjectAnchor* root = actor ? actor->GetAnchor(root_anchor) : nullptr;
  const bool horizontal =
      root &&
      std::find(root->dof.begin(), root->dof.end(), StateType::kPosition2D) !=
          root->dof.end() &&
      std::find(root->dof.begin(), root->dof.end(), StateType::kPosition3D) ==
          root->dof.end();
  if (!root || !horizontal || !root->parent.empty() ||
      !root->local_offset.isZero(0.0)) {
    return false;
  }

  ObservationClock& clock = observation_clocks_[actor_id];
  const auto fence = retired_observation_epoch_fences_.find(actor_id);
  const uint64_t fenced_epoch = fence == retired_observation_epoch_fences_.end()
                                    ? 0
                                    : fence->second;
  const uint64_t previous_epoch = std::max(clock.epoch, fenced_epoch);
  if (previous_epoch == std::numeric_limits<uint64_t>::max()) return false;
  const uint64_t next_epoch = previous_epoch + 1;
  // A direct legacy begin deliberately supersedes an old setup capability.
  // Its token will then fail truthful retirement rather than erase this epoch.
  observed_source_setup_epochs_.erase(actor_id);
  ResetObservationStateLocked(actor_id);
  clock.epoch = next_epoch;
  clock.sequence = -1;
  clock.time = 0.0;
  clock.has_sample = false;
  clock.root = root_anchor;
  *epoch = next_epoch;

  // A reset is a control envelope: it cancels receiver output without
  // pretending that a source-time sample or controller tick occurred.
  std::vector<PendingPropagation> pending;
  if (hub_) {
    const Element* element = element_tree_.GetElement(actor->GetElementId());
    if (element) {
      RemoteRelationUpdate reset;
      reset.timing = RelationUpdateTiming::kObservationReset;
      reset.source_epoch = next_epoch;
      reset.source_space_id = space_id_;
      reset.actor_shared_id = element->GetSharedId();
      reset.actor_source_space_id = space_id_;
      reset.actor_source_element_id = actor->GetElementId();
      reset.sequence = ++next_propagate_sequence_[actor_id];
      // Staged rather than sent here, for the same reason as the source path:
      // Propagate runs the receiving space's OnRemoteUpdate under that space's
      // own mutex. A reset carries no ev=src trace, so only the send is staged.
      PendingPropagation staged;
      staged.hub = hub_;
      staged.space_id = space_id_;
      staged.update = std::move(reset);
      staged.has_send = true;
      pending.push_back(std::move(staged));
    }
  }
  std::unique_lock<std::mutex> delivery(propagation_mutex_);
  lock.unlock();
  DeliverPropagations(pending);
  return true;
}

bool Space::ApplyActorObservationFrame(
    spatialblend::entity_id_t actor_id, uint64_t epoch, int64_t sequence,
    double time_seconds, const std::vector<ActorObservation>& observations) {
  // Preserve admission-attempt receipt even across the wrapper's frame copy.
  const int64_t receipt = receiver_input_clock_();
  return ApplyActorObservationBatchWithReceipt(
      {{actor_id, epoch, sequence, time_seconds, observations}}, {}, receipt);
}

bool Space::ValidateActorObservationFrameLocked(
    const ActorObservationFrame& frame) const {
  const auto actor_id = frame.actor;
  const auto epoch = frame.epoch;
  const auto sequence = frame.sequence;
  const double time_seconds = frame.time_seconds;
  if (receiver_inputs_.count(actor_id)) return false;
  const Actor* actor = FindActorLocked(actor_id);
  auto clock_it = observation_clocks_.find(actor_id);
  if (!actor || clock_it == observation_clocks_.end() ||
      clock_it->second.epoch != epoch || sequence < 0 ||
      sequence <= clock_it->second.sequence || !std::isfinite(time_seconds) ||
      time_seconds < 0.0 ||
      (clock_it->second.has_sample && time_seconds < clock_it->second.time)) {
    return false;
  }
  const SubjectAnchor* root = actor->GetAnchor(clock_it->second.root);
  const bool horizontal =
      root &&
      std::find(root->dof.begin(), root->dof.end(), StateType::kPosition2D) !=
          root->dof.end() &&
      std::find(root->dof.begin(), root->dof.end(), StateType::kPosition3D) ==
          root->dof.end();
  if (!root || !horizontal || !root->parent.empty() ||
      !root->local_offset.isZero(0.0)) {
    return false;
  }

  std::unordered_set<std::string> names;
  const auto embodiment = EmbodimentOf(actor_id);
  for (const ActorObservation& observation : frame.observations) {
    // A bone-bound anchor is derived from its bone: a stated pose is a second source for one fact (refused).
    if (!observation.from_bones && IsBoneBound(embodiment, observation.anchor_name)) return false;
    if (!actor->GetAnchor(observation.anchor_name) ||
        !names.insert(observation.anchor_name).second ||
        !std::isfinite(observation.position.x()) ||
        !std::isfinite(observation.position.y()) ||
        !std::isfinite(observation.position.z()) ||
        !std::isfinite(observation.yaw)) {
      return false;
    }
  }
  return true;
}

Space::PreparedActorObservation Space::CommitActorObservationFrameLocked(
    const ActorObservationFrame& frame, int64_t receipt) {
  const auto actor_id = frame.actor;
  source_evidence_.erase(actor_id);
  Actor* actor = FindActorLocked(actor_id);
  auto clock_it = observation_clocks_.find(actor_id);
  // Entire-frame validation finished: mutate availability and overrides as a
  // single committed snapshot. Omitted root keeps the native transform only.
  Eigen::Vector2d reference_position = actor->GetPosition();
  double reference_yaw = actor->GetRotation() * M_PI / 180.0;
  if (!std::isfinite(reference_position.x()) ||
      !std::isfinite(reference_position.y())) {
    reference_position.setZero();
  }
  if (!std::isfinite(reference_yaw)) reference_yaw = 0.0;
  for (const SubjectAnchor& anchor : actor->GetAnchors()) {
    actor->ClearAnchorWorldPose(anchor.name);
    actor->SetAnchorActive(anchor.name, false);
  }
  for (const ActorObservation& observation : frame.observations) {
    Eigen::Vector3d position = observation.position;
    if (observation.anchor_name == clock_it->second.root) {
      position.z() = 0.0;  // declared horizontal root projects finite Z away
      reference_position = position.head<2>();
      reference_yaw = observation.yaw;
    }
    actor->SetAnchorWorldPose(observation.anchor_name, position,
                              observation.yaw);
    actor->SetAnchorActive(observation.anchor_name, true);
  }
  if (!actor->IsAnchorActive(clock_it->second.root)) {
    // A later root sample is a temporal re-entry, not velocity across a gap.
    motion_trackers_.erase(actor_id);
  }

  const double elapsed = clock_it->second.has_sample
                             ? frame.time_seconds - clock_it->second.time
                             : 0.0;
  clock_it->second.sequence = frame.sequence;
  clock_it->second.time = frame.time_seconds;
  clock_it->second.has_sample = true;
  clock_it->second.receipt_monotonic_ns = receipt;
  clock_it->second.anchors_from_committed_frame = true;
  anchor_observation_dirty_.insert(actor_id);
  // Prepare every batch member's pose before any binding evaluation/callback.
  // EvaluateActorStateLocked repeats this same pose, without a tree revision.
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform(0, 3) = reference_position.x();
  transform(1, 3) = reference_position.y();
  transform(0, 0) = transform(1, 1) = std::cos(reference_yaw);
  transform(0, 1) = -std::sin(reference_yaw);
  transform(1, 0) = std::sin(reference_yaw);
  actor->SetTransform(transform);
  auto* element = element_tree_.GetElement(actor->GetElementId());
  if (element && !element->GetSpatialDefinition() &&
      element->GetChildIds().empty())
    SetElementTransformLocked(element->GetId(), transform, /*actor_pose=*/true);
  return {reference_position, reference_yaw, elapsed};
}

bool Space::ClearActorObservationAnchor(spatialblend::entity_id_t actor_id,
                                        uint64_t epoch,
                                        const std::string& anchor_name) {
  std::vector<PendingPropagation> pending;
  std::unique_lock<std::mutex> lock(space_mutex_);
  if (receiver_inputs_.count(actor_id)) return false;
  Actor* actor = FindActorLocked(actor_id);
  auto clock_it = observation_clocks_.find(actor_id);
  if (!actor || !actor->GetAnchor(anchor_name) ||
      clock_it == observation_clocks_.end() ||
      clock_it->second.epoch != epoch) {
    return false;
  }
  // Already absent is an idempotent control no-op, including immediately
  // after Begin. In particular it must not create a synthetic receipt.
  if (!actor->IsAnchorActive(anchor_name)) return true;
  source_evidence_.erase(actor_id);
  actor->ClearAnchorWorldPose(anchor_name);
  actor->SetAnchorActive(anchor_name, false);
  if (anchor_name == clock_it->second.root) motion_trackers_.erase(actor_id);
  if (auto bindings_it = bindings_.find(actor_id);
      bindings_it != bindings_.end()) {
    for (Binding& binding : bindings_it->second) {
      if (binding.anchor_name != anchor_name) continue;
      binding.sent_relations.clear();
      binding.smoothed_progress_rate.clear();
      binding.last_progress_sample_seconds.clear();
      binding.has_progress_sample.clear();
    }
  }
  anchor_observation_dirty_.insert(actor_id);
  Eigen::Vector2d position = actor->GetPosition();
  double yaw = actor->GetRotation() * M_PI / 180.0;
  if (!std::isfinite(position.x()) || !std::isfinite(position.y()))
    position.setZero();
  if (!std::isfinite(yaw)) yaw = 0.0;
  EvaluateActorStateLocked(actor_id, position, yaw, true, clock_it->second.time,
                           0.0, &pending);
  // Same handoff as UpdateActorState: take the delivery guard while the state
  // guard is still held, release the state guard, then dispatch.
  std::unique_lock<std::mutex> delivery(propagation_mutex_);
  lock.unlock();
  DeliverPropagations(pending);
  return true;
}

bool Space::GetActorObservationClock(spatialblend::entity_id_t actor_id,
                                     uint64_t* epoch, int64_t* sequence,
                                     double* time, bool* has_sample) const {
  if (!epoch || !sequence || !time || !has_sample) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!FindActorLocked(actor_id)) return false;
  auto it = observation_clocks_.find(actor_id);
  if (it == observation_clocks_.end() || it->second.epoch == 0) return false;
  *epoch = it->second.epoch;
  *sequence = it->second.sequence;
  *time = it->second.time;
  *has_sample = it->second.has_sample;
  return true;
}

space_id_t Space::GetSpaceId() const { return space_id_; }
void Space::SetSpaceId(space_id_t id) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (space_id_ == id) return;
  source_evidence_.clear();
  source_evidence_enabled_.clear();
  source_evidence_phase_dirty_.clear();
  // Changing identity cannot temporarily hide and then revive an old point
  // capability if the caller later restores the same numeric space ID.
  for (auto& [element, point] : observed_facing_points_) {
    if (point.retired) continue;
    point.active = false;
    point.retired = true;
    AdvanceSceneInputsLocked(true, true);
    InvalidateObservedFacingConsumersLocked(element);
  }
  for (const auto& [actor_id, input] : receiver_inputs_)
    InvalidateReceiverInputLocked(actor_id);
  receiver_approach_contexts_.clear();
  space_id_ = id;
}

void Space::SetUpdateInterval(double seconds) {
  if (!std::isfinite(seconds) || seconds <= 0.0) return;
  update_interval_ = seconds;
}

void Space::SetRelationSolverConfig(const RelationSolver::Config& config) {
  solver_ = RelationSolver(config);
}

void Space::SetWeightSmoothing(double alpha) {
  weight_smoothing_alpha_ = std::clamp(alpha, 1e-6, 1.0);
}

void Space::SetReferentResolveMode(ReferentResolveMode mode) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (mode != ReferentResolveMode::kLinker &&
      mode != ReferentResolveMode::kSharedId &&
      mode != ReferentResolveMode::kLinkerOnly) {
    return;
  }
  if (referent_resolve_mode_ != mode) ++counterpart_revision_;
  referent_resolve_mode_ = mode;
  referent_resolve_mode_env_resolved_ = true;  // pre-empt the env lazy-init
}

void Space::SetRealizationPolicy(RealizationPolicyType type) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!IsValidRealizationPolicy(type)) return;
  if (realization_policy_type_ != type) ++policy_revision_;
  realization_policy_type_ = type;
  realization_policy_env_resolved_ = true;  // pre-empt the env lazy-init
}

void Space::SetMinMove(double meters) {
  min_propagation_move_ = std::max(0.0, meters);
}

void Space::SetMinYaw(double radians) {
  min_propagation_yaw_ = std::max(0.0, radians);
}


void Space::SetAnchorReach(const std::string& anchor_name, double radius) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (anchor_name.empty() || !std::isfinite(radius)) return;
  radius = std::max(0.0, radius);
  const auto it = anchor_reach_radii_.find(anchor_name);
  if (it != anchor_reach_radii_.end() && it->second == radius) return;
  anchor_reach_radii_[anchor_name] = radius;
  ++policy_revision_;
}

bool Space::IsValidRealizationPolicy(RealizationPolicyType type) {
  return type == RealizationPolicyType::kProgressive ||
         type == RealizationPolicyType::kUniform ||
         type == RealizationPolicyType::kGreedy;
}

void Space::ResolveRealizationPolicyLocked() {
  if (realization_policy_env_resolved_) return;
  realization_policy_env_resolved_ = true;
  RealizationPolicyType resolved = realization_policy_type_;
  if (const char* e = std::getenv("SB_V2_REALIZATION_POLICY")) {
    const std::string policy(e);
    if (policy == "uniform") resolved = RealizationPolicyType::kUniform;
    if (policy == "greedy") resolved = RealizationPolicyType::kGreedy;
    if (policy == "progressive") resolved = RealizationPolicyType::kProgressive;
  }
  if (resolved != realization_policy_type_) {
    realization_policy_type_ = resolved;
    ++policy_revision_;
  }
}

void Space::ObserveCounterpartGraphLocked(uint64_t graph_revision) {
  if (observed_counterpart_hub_ != hub_ ||
      !has_observed_linker_graph_revision_ ||
      observed_linker_graph_revision_ != graph_revision) {
    ++counterpart_revision_;
    observed_counterpart_hub_ = hub_;
    observed_linker_graph_revision_ = graph_revision;
    has_observed_linker_graph_revision_ = true;
  }
}

void Space::ResolveReferentResolveModeLocked() {
  if (referent_resolve_mode_env_resolved_) return;
  referent_resolve_mode_env_resolved_ = true;
  ReferentResolveMode resolved = referent_resolve_mode_;
  if (const char* e = std::getenv("SB_V2_REFERENT_RESOLVE")) {
    const std::string mode(e);
    if (mode == "sharedid") resolved = ReferentResolveMode::kSharedId;
    if (mode == "linker") resolved = ReferentResolveMode::kLinker;
    if (mode == "linker_only") resolved = ReferentResolveMode::kLinkerOnly;
  }
  if (resolved != referent_resolve_mode_) {
    referent_resolve_mode_ = resolved;
    ++counterpart_revision_;
  }
}

bool Space::CaptureCounterpartRevisionLocked(uint64_t graph_revision_before,
                                             uint64_t graph_revision_after,
                                             uint64_t* revision) {
  ResolveReferentResolveModeLocked();
  if (!revision || !hub_ || has_custom_referent_resolver_ ||
      referent_resolve_mode_ == ReferentResolveMode::kSharedId ||
      graph_revision_before != graph_revision_after) {
    return false;
  }
  ObserveCounterpartGraphLocked(graph_revision_after);
  *revision = counterpart_revision_;
  return true;
}

bool Space::CurrentCounterpartRevisionLocked(uint64_t* revision) {
  ResolveReferentResolveModeLocked();
  if (!revision || !hub_ || has_custom_referent_resolver_ ||
      referent_resolve_mode_ == ReferentResolveMode::kSharedId) {
    return false;
  }
  ObserveCounterpartGraphLocked(hub_->GetLinker().GetGraphRevision());
  *revision = counterpart_revision_;
  return true;
}


bool Space::SourceRetiredLocked(space_id_t space, element_id_t element) const {
  return element != kInvalidElementId &&
         retired_sources_.count(element_handle_t{space, element}) != 0;
}

// The receiver's identity resolution of one incoming payload (shared by
// OnRemoteUpdate and EvaluateTargetRelation): the referent's and the source
// normalizing scope's local counterparts through this hub's linker (never
// trusting sender-provided receiver-local scratch state), then the scope by
// shared id unless linker-only.
void Space::ResolveReceiverIdentitiesLocked(RelationPayload* payload) const {
  const bool translate_linker =
      hub_ && referent_resolve_mode_ != ReferentResolveMode::kSharedId;
  payload->resolved_local_element_id = kInvalidElementId;
  payload->referent_retired = false;
  if (translate_linker &&
      payload->referent_source_element_id != kInvalidElementId) {
    payload->resolved_local_element_id = hub_->GetLinker().Translate(
        element_handle_t{payload->referent_source_space_id,
                         payload->referent_source_element_id},
        space_id_);
  }
  // A source whose link ended (unshared or unlinked) and was not resolved
  // anew has no counterpart here: the shared id it carries must not find the
  // element that used to stand for it. Whatever the resolve mode.
  if (payload->resolved_local_element_id == kInvalidElementId &&
      SourceRetiredLocked(payload->referent_source_space_id,
                          payload->referent_source_element_id)) {
    payload->referent_retired = true;
  }
  payload->resolved_scope_element_id = kInvalidElementId;
  payload->scope_resolved_by_receiver = true;
  if (translate_linker &&
      payload->scope_source_element_id != kInvalidElementId) {
    payload->resolved_scope_element_id = hub_->GetLinker().Translate(
        element_handle_t{payload->scope_source_space_id,
                         payload->scope_source_element_id},
        space_id_);
  }
  const bool scope_retired =
      payload->resolved_scope_element_id == kInvalidElementId &&
      SourceRetiredLocked(payload->scope_source_space_id,
                          payload->scope_source_element_id);
  if (payload->resolved_scope_element_id == kInvalidElementId &&
      !scope_retired &&
      referent_resolve_mode_ != ReferentResolveMode::kLinkerOnly &&
      payload->scope_shared_id != kInvalidSharedElementId) {
    if (const Element* scope_el =
            GetReferentResolver().Resolve(payload->scope_shared_id,
                                          element_tree_))
      payload->resolved_scope_element_id = scope_el->GetId();
  }
}

bool Space::EvaluateTargetRelation(const RelationPayload& payload,
                                   const TargetAnchorPose& anchor,
                                   element_id_t normalizing_scope,
                                   TargetRelationEvaluation* out) const {
  if (!out || !anchor.position.allFinite() || !std::isfinite(anchor.yaw))
    return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  // Resolve exactly as a delivered payload would be (linker handles, scope
  // counterpart, linker-only strictness), so the ABI path evaluates against
  // the same referent and normalizing context as the realizer.
  RelationPayload resolved = payload;
  ResolveReceiverIdentitiesLocked(&resolved);
  const bool live_link =
      resolved.resolved_local_element_id != kInvalidElementId &&
      element_tree_.GetElement(resolved.resolved_local_element_id) != nullptr;
  // A retired source has no counterpart: nothing to evaluate against.
  if (resolved.referent_retired && !live_link) return false;
  if (referent_resolve_mode_ == ReferentResolveMode::kLinkerOnly &&
      !live_link) {
    resolved.resolved_local_element_id = kInvalidElementId;
    resolved.referent_shared_id = kInvalidSharedElementId;  // -> unmapped
  }
  TargetEvaluationOptions options;
  options.normalizing_scope = normalizing_scope;
  options.resolver = &GetReferentResolver();
  *out = spatialblend::EvaluateTargetRelation(resolved, anchor, element_tree_,
                                              options);
  // r_src / r_tgt / error come from the evaluator (one R formula on both
  // sides); do not overwrite r_src with the proximity component here.
  return true;
}

double Space::ActorSecondsLocked(spatialblend::entity_id_t actor_id) const {
  const auto it = receiver_actor_seconds_.find(actor_id);
  return it != receiver_actor_seconds_.end() ? it->second.seconds : 0.0;
}

bool Space::InvalidPolicyWeights(const std::vector<double>& weights) {
  return std::any_of(weights.begin(), weights.end(), [](double w) {
    return !std::isfinite(w) || w < 0.0;
  });
}



double Space::GetAnchorReach(const std::string& anchor_name) const {
  auto it = anchor_reach_radii_.find(anchor_name);
  return (it != anchor_reach_radii_.end()) ? it->second : default_anchor_reach_;
}

void Space::AppendAnchorTargets(spatialblend::entity_id_t actor_id,
                                const Eigen::Vector2d& root_position,
                                double root_yaw,
                                const std::vector<RelationPayload>& payloads,
                                const std::vector<double>& weights,
                                const NavigableSpatialDefinition* navigable_def,
                                RealizationResult& realization) {
  // Performance: only the AffordancePose anchor targets (and the kFallback a
  // grounding error raises) are read from SolveActorTargets' result; its
  // composed root position/yaw, and every Spacing/Facing relation it would
  // solve to get them, are discarded (the caller already solved and owns
  // the root pose). Affordance groups are independent of the other families
  // (the affordance block of SolveActorTargets
  // reads neither the root position nor the root yaw), so hand it the
  // AffordancePose payloads only -- and nothing at all when there are none.
  std::vector<RelationPayload> anchor_payloads;
  std::vector<double> anchor_weights;
  for (size_t i = 0; i < payloads.size(); ++i) {
    if (payloads[i].kernel_type != RelationKernelType::kAvatarAffordancePose)
      continue;
    anchor_payloads.push_back(payloads[i]);
    if (i < weights.size()) anchor_weights.push_back(weights[i]);
  }
  if (anchor_weights.size() != anchor_payloads.size()) anchor_weights.clear();
  RealizationResult candidates;
  if (!anchor_payloads.empty()) {
    candidates = solver_.SolveActorTargets(root_position, root_yaw,
                                           anchor_payloads, anchor_weights,
                                           element_tree_, navigable_def);
  }
  // Only an invalid policy vector invalidates anchor targets. A failed root
  // position or facing solve never suppresses engaged body-anchor targets:
  // each AffordancePose group is grounded independently.
  if (InvalidPolicyWeights(weights)) {
    realization.status = SolverStatus::kFailed;
    return;
  }
  if (candidates.status == SolverStatus::kFallback &&
      realization.status == SolverStatus::kConverged) {
    realization.status = SolverStatus::kFallback;
  }
  // A PLACED answer (the join, a source exception) reads the relations whatever their weight (weight only
  // breaks ties), as the realization producer's contributors already do (placed_solve_): on the join frame no relation
  // has a valid count yet, so the policy weighs every one 0, and the anchors that carry AffordancePose relations are
  // still decided below (engaged on the mapped support the source touches, or awaiting engagement). SolveActorTargets
  // blends only weighted groups, so such an anchor had no candidate and was reported as an unsupported release, and its
  // engaged support was skipped for its weight. On a continuation a relation weighed 0 stays disabled.
  const bool placed = placed_solve_.count(actor_id) != 0 && placed_solve_.at(actor_id);
  for (const auto& p : anchor_payloads) {
    if (!placed) break;
    if (!p.enabled || IsInnerPosePayload(p) || p.subject_anchor_name.empty()) continue;
    bool listed = false;
    for (const auto& t : candidates.targets)
      listed = listed || (t.kernel_type == RelationKernelType::kAvatarAffordancePose &&
                          t.anchor_name == p.subject_anchor_name);
    if (listed) continue;
    RealizationTarget t;
    t.anchor_name = p.subject_anchor_name;
    t.kernel_type = RelationKernelType::kAvatarAffordancePose;
    t.weight = 0.0;
    candidates.targets.push_back(std::move(t));
  }

  for (auto& tgt : candidates.targets) {
    if (tgt.kernel_type != RelationKernelType::kAvatarAffordancePose) continue;
    const std::string& anchor_name = tgt.anchor_name;

    // An empty anchor is invalid.  A literal "root" is NOT a physical-role
    // declaration: applications may bind a body anchor under that name.
    // Controller isolation below keeps affordance targets out of locomotion.
    if (anchor_name.empty()) {
      realization.status = SolverStatus::kFallback;
      continue;
    }
    const auto receiver = receiver_inputs_.find(actor_id);
    if (receiver != receiver_inputs_.end()) {
      Eigen::Matrix4d pose;
      if (!CurrentReceiverInputLocked(actor_id).known ||
          !receiver->second.CopyPose(anchor_name, &pose)) {
        realization.status = SolverStatus::kFallback;
        continue;
      }
    }
    // The target is the source support mapped onto the receiver ONLY while the
    // source anchor TOUCHES its support surface (engagement
    // is geometric, engagement.h IsAffordanceContact). No distance
    // threshold, no stillness, no dwell, no hysteresis. The engaged support is the
    // contributor the source anchor is actually on (smallest support distance),
    // not a blend. Outside contact the anchor has no support target and follows the
    // actor's body. The engaged target is never moved by the reach limit: a
    // clamped point is not the contact (up to 1.05 m of target/contact
    // disagreement when the clamp was centred on the solved root). Reach is
    // reported instead (beyond_reach).
    // A referent whose counterpart offers no affordance is not in
    // AffordancePose scope. It is skipped, here as in SolveActorTargets; it never fails
    // the anchor. The engaged support is the nearest engaged referent that grounds.
    size_t engaged_index = payloads.size();
    double engaged_distance = std::numeric_limits<double>::infinity();
    absl::StatusOr<GroundedAffordanceTarget> contact =
        absl::InternalError("no engaged referent");
    for (size_t i = 0; i < payloads.size(); ++i) {
      const auto& p = payloads[i];
      if (p.kernel_type != RelationKernelType::kAvatarAffordancePose ||
          IsInnerPosePayload(p) ||  // the inner level is the anchor's own solve, never a contact
          p.subject_anchor_name != anchor_name || !p.enabled || !IsEngaged(p) ||
          !(placed || (i < weights.size() && weights[i] > 0.0)))
        continue;
      const double d = AffordanceSupportDistance(p);
      if (!(d < engaged_distance || engaged_index >= payloads.size())) continue;
      auto grounded = solver_.GroundAffordanceTarget(p, element_tree_);
      if (!grounded.ok()) continue;
      engaged_distance = d;
      engaged_index = i;
      contact = std::move(grounded);
    }
    if (engaged_index >= payloads.size()) {
      // The source is not on a support the receiver can ground: a normal state, not a
      // failure.
      realization.awaiting_engagement.push_back(anchor_name);
      continue;
    }
    tgt.position = contact->position.head<2>();
    tgt.z = contact->position.z();
    tgt.has_z = true;
    tgt.yaw = contact->yaw;
    tgt.referent_id = payloads[engaged_index].referent_shared_id;
    tgt.engaged_support = true;
    const double reach = GetAnchorReach(anchor_name);
    const double from_root = (tgt.position - root_position).norm();
    if (reach > 0.0 && from_root > reach) {
      tgt.beyond_reach = true;
      tgt.reach_excess_m = from_root - reach;
    }
    realization.targets.push_back(std::move(tgt));
  }
}

void Space::AppendFacingTargets(spatialblend::entity_id_t actor_id,
                                const std::vector<RelationPayload>& payloads,
                                const std::vector<double>& weights,
                                const NavigableSpatialDefinition* navigable_def,
                                RealizationResult& realization) {
  Actor* actor = FindActorLocked(actor_id);
  if (!actor) return;
  std::map<std::string, std::vector<size_t>> groups;
  for (size_t i = 0; i < payloads.size(); ++i) {
    if (payloads[i].kernel_type == RelationKernelType::kAvatarFacing) {
      groups[payloads[i].subject_anchor_name.empty()
                 ? "root"
                 : payloads[i].subject_anchor_name]
          .push_back(i);
    }
  }
  for (const auto& [anchor_name, indices] : groups) {
    bool has_enabled = false;
    for (size_t i : indices) {
      if (i < weights.size() && weights[i] > 0.0) has_enabled = true;
    }
    if (!has_enabled) continue;
    Eigen::Vector3d anchor_position;
    double anchor_yaw = 0.0;
    const auto receiver = receiver_inputs_.find(actor_id);
    if (receiver != receiver_inputs_.end()) {
      Eigen::Matrix4d pose;
      if (!CurrentReceiverInputLocked(actor_id).known ||
          !receiver->second.CopyPose(anchor_name, &pose) ||
          !ReceiverPoseYaw(pose, &anchor_yaw)) {
        realization.status = SolverStatus::kFallback;
        continue;
      }
      anchor_position = pose.block<3, 1>(0, 3);
    } else {
      const SubjectAnchor* anchor = actor->GetAnchor(anchor_name);
      if (!anchor || !actor->IsAnchorActive(anchor_name)) {
        realization.status = SolverStatus::kFallback;
        continue;
      }
      anchor_yaw = actor->GetRotation() * M_PI / 180.0;
      if (const auto* override = actor->GetAnchorWorldPoseOverride(anchor_name)) {
        anchor_position = override->position;
        anchor_yaw = override->yaw;
      } else {
        anchor_position = anchor->GetWorldPosition(actor->GetTransform());
      }
    }
    std::vector<RelationPayload> group_payloads;
    std::vector<double> group_weights;
    for (size_t i : indices) {
      group_payloads.push_back(payloads[i]);
      group_weights.push_back(i < weights.size() ? weights[i] : 1.0);
    }
    // The source's own (referent-relative) heading is the fallback
    // when every facing bearing is conditioned out.
    const SolverResult solved = solver_.Solve(
        anchor_position.head<2>(), anchor_yaw, group_payloads, group_weights,
        element_tree_, navigable_def, &GetReferentResolver(),
        ReferentRelativeHeading(payloads, weights, element_tree_,
                                &GetReferentResolver()));
    if (solved.status == SolverStatus::kFailed) {
      realization.status = SolverStatus::kFailed;
      continue;
    }
    RealizationTarget target;
    target.anchor_name = anchor_name;
    target.position = anchor_position.head<2>();
    target.z = anchor_position.z();
    target.has_z = true;
    target.yaw = solved.yaw;
    target.kernel_type = RelationKernelType::kAvatarFacing;
    target.weight = 1.0;
    realization.targets.push_back(std::move(target));
  }
}

void Space::CaptureRealizationSnapshot(
    spatialblend::entity_id_t actor_id,
    const std::vector<RelationPayload>& incoming_payloads,
    const std::vector<int>& resolved_index,
    const std::vector<RelationPayload>& payloads,
    const std::vector<double>& weights,
    const RealizationResult& candidate_realization,
    const SceneInputStamp& scene_inputs,
    const RemoteRelationUpdate* source_update,
    int64_t receiver_ingress_monotonic_ns, bool counterpart_revision_known,
    uint64_t counterpart_revision,
    const ReceiverInputStamp& receiver_input) {
  auto coherent_receiver_input = receiver_input;
  const auto current_receiver = CurrentReceiverInputLocked(actor_id);
  coherent_receiver_input.known =
      receiver_input.known && current_receiver.known &&
      receiver_input.epoch == current_receiver.epoch &&
      receiver_input.revision == current_receiver.revision;
  const bool expired = receiver_input.known && !coherent_receiver_input.known;
  if (expired) {
    ClearReceiverTargetsLocked(actor_id);
    realization_results_[actor_id].status = SolverStatus::kFailed;
  }
  const auto& realization =
      expired ? realization_results_.at(actor_id) : candidate_realization;
  if (receiver_inputs_.count(actor_id))
    receiver_solve_inputs_[actor_id] = coherent_receiver_input;
  const auto retained = retained_receiver_evaluations_.find(actor_id);
  if (retained != retained_receiver_evaluations_.end())
    retained->second.last_evaluated_input = coherent_receiver_input;
  if (!rich_realization_enabled_ && !rich_realization_actors_.count(actor_id))
    return;
  ActorRealizationSnapshot snapshot;
  auto old = realization_snapshots_.find(actor_id);
  snapshot.revision =
      old == realization_snapshots_.end() ? 1 : old->second.revision + 1;
  snapshot.solve_status = realization.status;
  // Expiry can change while the solve mutex is held. Certify only the
  // still-live frame actually consumed, never a reset/control snapshot.
  snapshot.receiver_input = coherent_receiver_input;
  ResolveRealizationPolicyLocked();
  snapshot.policy_revision = policy_revision_;
  snapshot.known_revision_mask = kRealizationPolicyRevision;
  // The solver must have consumed one coherent scene input, not merely see a
  // new revision at the end. Legacy direct realization can itself mirror a
  // leaf pose during the solve; that mixed-input transaction is uncertified.
  const auto current_scene = CaptureSceneInputStampLocked();
  if (scene_inputs.known && current_scene.known &&
      scene_inputs.tree == current_scene.tree &&
      scene_inputs.support == current_scene.support) {
    snapshot.tree_revision = scene_inputs.tree;
    snapshot.support_revision = scene_inputs.support;
    snapshot.known_revision_mask |= kRealizationTreeRevision |
                                    kRealizationSupportRevision;
  } else {
    snapshot.unknown_revision_mask |= kRealizationTreeRevision |
                                      kRealizationSupportRevision;
  }
  if (counterpart_revision_known) {
    snapshot.counterpart_revision = counterpart_revision;
    snapshot.known_revision_mask |= kRealizationCounterpartRevision;
  } else {
    snapshot.unknown_revision_mask |= kRealizationCounterpartRevision;
  }
  if (source_update) {
    snapshot.source_timing_mode =
        source_update->timing == RelationUpdateTiming::kTimestampedObservation
            ? ActorRealizationSnapshot::SourceTimingMode::kTimestamped
        : source_update->timing == RelationUpdateTiming::kObservationReset
            ? ActorRealizationSnapshot::SourceTimingMode::kReset
            : ActorRealizationSnapshot::SourceTimingMode::kUnknownLegacy;
    snapshot.source_epoch = source_update->source_epoch;
    snapshot.source_sequence = source_update->sequence;
    snapshot.source_observation_sequence_known =
        source_update->timing ==
            RelationUpdateTiming::kTimestampedObservation &&
        source_update->source_observation_sequence_known;
    if (snapshot.source_observation_sequence_known) {
      snapshot.source_observation_sequence =
          source_update->source_observation_sequence;
    }
    snapshot.source_time_known =
        source_update->timing == RelationUpdateTiming::kTimestampedObservation;
    if (snapshot.source_time_known)
      snapshot.source_time_seconds = source_update->source_time;
    snapshot.source_clock_known = source_update->source_clock > 0.0;
    if (snapshot.source_clock_known)
      snapshot.source_clock_seconds = source_update->source_clock;
    snapshot.source_space_id =
        source_update->actor_source_space_id != kInvalidSpaceId
            ? source_update->actor_source_space_id
            : source_update->source_space_id;
    snapshot.relay_source_space_id = source_update->source_space_id;
    snapshot.source_actor_element_id = source_update->actor_source_element_id;
    snapshot.receiver_ingress_monotonic_ns = receiver_ingress_monotonic_ns;
    snapshot.receiver_solve_monotonic_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
  }
  std::map<std::pair<std::string, int>, RealizationChannel> channels;
  for (size_t ordinal = 0; ordinal < incoming_payloads.size(); ++ordinal) {
    const auto& incoming = incoming_payloads[ordinal];
    const int admitted =
        ordinal < resolved_index.size() ? resolved_index[ordinal] : -1;
    const auto& payload = admitted >= 0 ? payloads[admitted] : incoming;
    const std::string anchor = payload.subject_anchor_name.empty()
                                   ? "root"
                                   : payload.subject_anchor_name;
    auto& channel = channels[{anchor, static_cast<int>(payload.kernel_type)}];
    if (channel.anchor_name.empty()) {
      channel.anchor_name = anchor;
      channel.family = payload.kernel_type;
      channel.status = SolverStatus::kConverged;
    }
    RealizationContributor contributor;
    contributor.referent_id = payload.referent_shared_id;
    contributor.source_space_id = payload.referent_source_space_id;
    contributor.source_element_id = payload.referent_source_element_id;
    contributor.resolved_local_element_id =
        admitted >= 0 ? payload.resolved_local_element_id : kInvalidElementId;
    contributor.snapshot_ordinal = static_cast<uint32_t>(ordinal);
    // The source binding revision rides every contributor.
    contributor.binding_revision = payload.binding_revision;
    contributor.binding_revision_known = payload.binding_revision_known;
    contributor.applied_weight =
        admitted >= 0 && static_cast<size_t>(admitted) < weights.size()
            ? weights[admitted]
            : 0.0;
    const bool usable =
        admitted >= 0 &&
        (payload.resolved_local_element_id != kInvalidElementId
             ? element_tree_.GetElement(payload.resolved_local_element_id) !=
                   nullptr
             : element_tree_.GetElementBySharedId(payload.referent_shared_id) !=
                   nullptr);
    if (usable && contributor.resolved_local_element_id == kInvalidElementId) {
      contributor.resolved_local_element_id =
          element_tree_.GetElementBySharedId(payload.referent_shared_id)
              ->GetId();
    }
    if (!usable) {
      contributor.disposition = RealizationDisposition::kUnmapped;
      contributor.reason =
          admitted < 0 &&
                  referent_resolve_mode_ == ReferentResolveMode::kLinkerOnly
              ? "strict linker-only referent rejected"
              : "referent unresolved on receiver";
    } else if (!payload.enabled || !std::isfinite(contributor.applied_weight) || contributor.applied_weight < 0.0 ||
               (contributor.applied_weight == 0.0 &&
                (RatioRequest(payload.update_count) || placed_solve_.count(actor_id) == 0 ||
                 !placed_solve_.at(actor_id)))) {
      // Disabled: a relation the policy weighed 0. A relation without a valid count weighs 0 as well (Progressive
      // computes no weight for it, as v1) and makes no request; only a PLACED answer (a
      // join) reads it whatever its weight, so there it is included, with weight 0.
      contributor.disposition = RealizationDisposition::kDisabled;
      contributor.reason = "non-positive policy weight";
    } else if (IsInnerPosePayload(payload)) {
      // A term of the anchor's inner solve (the object's affordance pose of this tag), not a mapped contact.
      contributor.disposition = RealizationDisposition::kIncluded;
      contributor.reason = "inner affordance pose '" + payload.affordance_pose_tag + "'";
    } else if (payload.kernel_type ==
               RelationKernelType::kAvatarAffordancePose) {
      if (payload.referent_shared_id == kInvalidSharedElementId &&
          payload.resolved_local_element_id == kInvalidElementId) {
        contributor.disposition = RealizationDisposition::kUnmapped;
        contributor.reason = "missing resolved referent binding";
      } else {
        auto grounded = solver_.GroundAffordanceTarget(payload, element_tree_);
        if (grounded.ok()) {
          contributor.disposition = RealizationDisposition::kIncluded;
          contributor.has_mapped_support = true;
          contributor.mapped_support = grounded->position;
          contributor.grounding_mode = grounded->mode;
          contributor.support_plane_to_world = grounded->support_plane_to_world;
          contributor.support_identifier = grounded->support_identifier;
          contributor.chart_version_known = grounded->chart_version_known;
          contributor.binding_revision = grounded->binding_revision;
          contributor.binding_revision_known =
              grounded->binding_revision_known;
        } else {
          contributor.disposition = RealizationDisposition::kGroundingFailed;
          contributor.reason = std::string(grounded.status().message());
          channel.status = SolverStatus::kFallback;
          channel.geometry = RealizationGeometry::kUnsupported;
        }
      }
    } else {
      contributor.disposition = RealizationDisposition::kIncluded;
    }
    channel.contributors.push_back(std::move(contributor));
  }
  for (auto& [key, channel] : channels) {
    bool has_enabled_contributor = false;
    for (const auto& c : channel.contributors) {
      has_enabled_contributor |=
          c.disposition == RealizationDisposition::kIncluded;
    }
    for (const auto& target : realization.targets) {
      if (target.anchor_name == channel.anchor_name &&
          target.kernel_type == channel.family && has_enabled_contributor) {
        channel.target = target;
        channel.has_goal = true;
        if (channel.family == RelationKernelType::kAvatarAffordancePose) {
          std::vector<const RealizationContributor*> included;
          for (const auto& c : channel.contributors) {
            if (c.disposition == RealizationDisposition::kIncluded &&
                c.has_mapped_support)
              included.push_back(&c);
          }
          const bool legacy =
              !channel.contributors.empty() && !included.empty() &&
              std::all_of(
                  included.begin(), included.end(),
                  [](const RealizationContributor* c) {
                    return c->has_mapped_support &&
                           c->grounding_mode ==
                               AffordanceGroundingMode::kLegacyCenterFallback;
                  });
          // An engaged target is the exact support of the one relation
          // the source anchor is on, whatever else is in the channel.
          const bool exact = target.engaged_support || included.size() == 1;
          channel.geometry =
              legacy ? RealizationGeometry::kLegacyCenterNonContact
              : exact ? RealizationGeometry::kExactSupport
                      : RealizationGeometry::kBlendedSupport;
          if (!exact) {
            // A blend has no asserted dominant contact identity.
            channel.target.referent_id = kInvalidSharedElementId;
          }
          // Reach never moves a target; an engaged support beyond the
          // anchor's reach is FLAGGED through the (formerly projection)
          // reach fields so the executor sees an unreachable target.
          if (channel.target.beyond_reach) {
            channel.reach_displacement_m = channel.target.reach_excess_m;
            channel.reach_projected = true;
          }
          if (channel.reach_projected) {
            channel.geometry = RealizationGeometry::kReachProjected;
            channel.goal_kind = exact ? RealizationGoalKind::kSupportPose
                                      : RealizationGoalKind::kSoftGeometricPose;
          } else if (legacy) {
            channel.goal_kind = RealizationGoalKind::kSoftGeometricPose;
          } else if (exact) {
            channel.goal_kind = RealizationGoalKind::kSupportPose;
          } else {
            channel.goal_kind = RealizationGoalKind::kSoftGeometricPose;
          }
        } else {
          channel.geometry = RealizationGeometry::kNotApplicable;
          channel.goal_kind =
              channel.family == RelationKernelType::kAvatarSpacing
                  ? RealizationGoalKind::kRootPosition
                  : RealizationGoalKind::kFacingDirection;
        }
        break;
      }
    }
    const bool awaiting =
        !channel.has_goal &&
        channel.family == RelationKernelType::kAvatarAffordancePose &&
        std::find(realization.awaiting_engagement.begin(),
                  realization.awaiting_engagement.end(),
                  channel.anchor_name) != realization.awaiting_engagement.end();
    if (awaiting) {
      // Admitted, but the source anchor does not touch its support surface
      // (yet, or any more). The anchor follows the body; this is not a failure.
      const std::string reason = "source anchor not engaged on its support";
      for (auto& c : channel.contributors) {
        if (c.disposition == RealizationDisposition::kIncluded) {
          c.disposition = RealizationDisposition::kAwaitingEngagement;
          c.reason = reason;
        }
      }
    } else if (!channel.has_goal) {
      // A valid but non-admitted/gating-failed channel is an explicit release,
      // never permission to retain an older richer goal.
      for (auto& c : channel.contributors) {
        if (c.disposition == RealizationDisposition::kIncluded) {
          c.disposition = RealizationDisposition::kUnsupported;
          c.reason = "no admitted realization target";
        }
      }
      if (channel.status == SolverStatus::kConverged)
        channel.status = SolverStatus::kFallback;
    }
    for (const auto& c : channel.contributors) {
      if (c.disposition == RealizationDisposition::kUnmapped ||
          c.disposition == RealizationDisposition::kGroundingFailed ||
          c.disposition == RealizationDisposition::kUnsupported) {
        if (channel.status == SolverStatus::kConverged)
          channel.status = SolverStatus::kFallback;
      }
    }
    snapshot.channels.push_back(std::move(channel));
  }
  // Rich contributor construction can itself take time. Recheck immediately
  // before publishing, not just before the root/anchor solve or header build.
  if (snapshot.receiver_input.known) {
    const auto live = CurrentReceiverInputLocked(actor_id);
    if (!live.known || live.epoch != snapshot.receiver_input.epoch ||
        live.revision != snapshot.receiver_input.revision) {
      snapshot.receiver_input.known = false;
      snapshot.solve_status = SolverStatus::kFailed;
      for (auto& channel : snapshot.channels) {
        channel.has_goal = false;
        channel.target = {};
        channel.status = SolverStatus::kFailed;
        channel.goal_kind = RealizationGoalKind::kNone;
        channel.geometry = RealizationGeometry::kUnsupported;
        for (auto& contributor : channel.contributors) {
          if (contributor.disposition == RealizationDisposition::kIncluded) {
            contributor.disposition = RealizationDisposition::kUnsupported;
            contributor.reason = "receiver input expired before publication";
          }
        }
      }
      ClearReceiverTargetsLocked(actor_id);
      realization_results_[actor_id].status = SolverStatus::kFailed;
      receiver_solve_inputs_[actor_id] = snapshot.receiver_input;
    }
  }
  realization_snapshots_[actor_id] = std::move(snapshot);
}

void Space::SetMovementGate(double min_move_m, double min_yaw_rad) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  min_propagation_move_ = min_move_m;
  min_propagation_yaw_ = min_yaw_rad;
}

void Space::SetAnchorWorldPose(spatialblend::entity_id_t actor_id,
                               const std::string& anchor_name,
                               const Eigen::Vector3d& position, double yaw) {
  if (IsBoneBound(EmbodimentOf(actor_id), anchor_name)) return;  // derived from its bone
  SetAnchorWorldPoseImpl(actor_id, anchor_name, position, yaw);
}

bool Space::SetBoneAnchorWorldPose(spatialblend::entity_id_t actor_id, const std::string& anchor_name,
                                   const Eigen::Vector3d& position, double yaw, int32_t orientation_type,
                                   double pitch, double roll) {
  if (!IsBoneBound(EmbodimentOf(actor_id), anchor_name)) return false;
  SetAnchorWorldPoseImpl(actor_id, anchor_name, position, yaw, orientation_type, pitch, roll);
  return true;
}

void Space::SetAnchorWorldPoseImpl(spatialblend::entity_id_t actor_id,
                                   const std::string& anchor_name,
                                   const Eigen::Vector3d& position, double yaw, int32_t orientation_type,
                                   double pitch, double roll) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (receiver_inputs_.count(actor_id)) return;
  auto* actor = FindActorLocked(actor_id);
  if (!actor || !actor->GetAnchor(anchor_name)) return;
  InvalidateSourceEvidencePhaseLocked(actor_id);
  actor->SetAnchorWorldPose(anchor_name, position, yaw, orientation_type, pitch, roll);
  if (auto clock = observation_clocks_.find(actor_id); clock != observation_clocks_.end())
    clock->second.anchors_from_committed_frame = false;
  anchor_observation_dirty_.insert(actor_id);
}

void Space::ClearAnchorWorldPose(spatialblend::entity_id_t actor_id,
                                 const std::string& anchor_name) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (receiver_inputs_.count(actor_id)) return;
  auto* actor = FindActorLocked(actor_id);
  if (!actor || !actor->GetAnchor(anchor_name)) return;
  InvalidateSourceEvidencePhaseLocked(actor_id);
  actor->ClearAnchorWorldPose(anchor_name);
  if (auto clock = observation_clocks_.find(actor_id); clock != observation_clocks_.end())
    clock->second.anchors_from_committed_frame = false;
  anchor_observation_dirty_.insert(actor_id);
}

bool Space::SetAnchorActive(spatialblend::entity_id_t actor_id,
                            const std::string& anchor_name, bool active) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (receiver_inputs_.count(actor_id)) return false;
  auto* actor = FindActorLocked(actor_id);
  if (!actor || !actor->GetAnchor(anchor_name)) return false;
  if (actor->IsAnchorActive(anchor_name) == active) return true;
  InvalidateSourceEvidencePhaseLocked(actor_id);
  actor->SetAnchorActive(anchor_name, active);
  if (auto clock = observation_clocks_.find(actor_id); clock != observation_clocks_.end())
    clock->second.anchors_from_committed_frame = false;
  anchor_observation_dirty_.insert(actor_id);
  return true;
}

std::vector<RelationState> Space::GetRelationStates(
    spatialblend::entity_id_t actor_id) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return GetRelationStatesLocked(actor_id);
}

std::vector<RelationState> Space::GetRelationStatesFromCallback(
    spatialblend::entity_id_t actor_id) const {
  return GetRelationStatesLocked(actor_id);
}

std::vector<RelationState> Space::GetRelationStatesLocked(
    spatialblend::entity_id_t actor_id) const {
  std::vector<RelationState> out;
  auto bind_it = bindings_.find(actor_id);
  if (bind_it == bindings_.end()) return out;

  shared_element_id_t actor_shared_for_events =
      GetActorSharedIdForEvents(actor_id);

  for (size_t binding_index = 0; binding_index < bind_it->second.size();
       ++binding_index) {
    const Binding& binding = bind_it->second[binding_index];
    for (const auto& m : binding.latest_matches) {
      if (!IsObservedFacingCandidateLocked(m.element_id, actor_id,
                                             binding.kernel_type,
                                             receiver_input_clock_())) continue;
      const Element* ref_elem = element_tree_.GetElement(m.element_id);

      RelationState rs;
      rs.anchor_name = binding.anchor_name;
      rs.kernel_type = binding.kernel_type;
      rs.referent_element_id = m.element_id;
      // A received row stores the source referent shared id, which is the
      // phase-tracker key. The local element of a LinkToExisting referent
      // has a different shared id, so looking the phase up on that id
      // reports DORMANT while the event is ACTIVE.
      if (m.referent_shared_id != kInvalidSharedElementId) {
        rs.referent_shared_id = m.referent_shared_id;
      } else if (ref_elem &&
                 ref_elem->GetSharedId() != kInvalidSharedElementId) {
        rs.referent_shared_id = ref_elem->GetSharedId();
      } else {
        rs.referent_shared_id = static_cast<shared_element_id_t>(m.element_id);
      }
      // Read straight off the already-propagated match — no recomputation.
      rs.proximity = m.eval.grounded.proximity;
      rs.has_contact = m.eval.support.has_contact;
      rs.contact_point_world = m.eval.support.contact_point_world;
      rs.contact_normalized = m.eval.support.contact_normalized;

      // "active" reads the SAME phase-tracker key EmitRelationTransitions
      // populated this frame (identity fields only; from/to/timestamp unused
      // by GetPhase).
      RelationEvent identity;
      identity.actor_shared_id = actor_shared_for_events;
      identity.anchor_name = binding.anchor_name;
      identity.referent_shared_id = rs.referent_shared_id;
      identity.kernel_type = binding.kernel_type;
      rs.phase =
          phase_tracker_.GetPhaseForOwner(identity, PhaseOwner(actor_id));
      // "active" <=> the relation is in scope and realized (there is no
      // activation gate).
      rs.active = rs.phase != RelationPhase::kDormant;
      rs.binding_index = binding_index;
      rs.sub_region_name = m.eval.support.sub_region_name;
      rs.encoded_state = m.eval.grounded.encoded_state;
      rs.support_distance_m = m.eval.grounded.support_distance_m;
      rs.source_scope_diameter_m = m.eval.grounded.source_scope_diameter_m;
      rs.has_interior = m.eval.support.has_interior;
      rs.interior_uvw = m.eval.support.interior.uvw;
      rs.ap_offset_u_m = m.eval.grounded.ap_offset_u_m;
      rs.ap_offset_v_m = m.eval.grounded.ap_offset_v_m;
      rs.ap_height_m = m.eval.grounded.ap_height_m;

      out.push_back(std::move(rs));
    }
  }
  return out;
}

void Space::SetHub(SpaceHub* hub) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (hub_ == hub) return;
  for (const auto& retained : retained_receiver_evaluations_)
    ClearReceiverTargetsLocked(retained.first);
  retained_receiver_evaluations_.clear();
  hub_ = hub;
  observed_counterpart_hub_ = nullptr;
  has_observed_linker_graph_revision_ = false;
  ++counterpart_revision_;
}

spatialblend::entity_id_t Space::FindActorForRemoteUpdate(
    const RemoteRelationUpdate& update) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return FindActorForRemoteUpdateLocked(update);
}

spatialblend::entity_id_t Space::FindActorForRemoteUpdateLocked(
    const RemoteRelationUpdate& update) const {
  element_id_t linked_actor_elem = kInvalidElementId;
  if (hub_ && update.actor_source_element_id != kInvalidElementId) {
    linked_actor_elem = hub_->GetLinker().Translate(
        element_handle_t{update.actor_source_space_id,
                         update.actor_source_element_id},
        space_id_);
  }
  // A source actor whose link ended and was not resolved anew drives nothing:
  // its shared id must not find the actor that used to stand for it.
  if (linked_actor_elem == kInvalidElementId &&
      SourceRetiredLocked(update.actor_source_space_id,
                          update.actor_source_element_id)) {
    return spatialblend::kInvalidEntityId;
  }
  for (const auto& [id, actor] : actors_) {
    const element_id_t elem_id = actor->GetElementId();
    if (elem_id == kInvalidElementId) continue;
    if (linked_actor_elem != kInvalidElementId) {
      if (elem_id == linked_actor_elem) return id;
      continue;  // A resolved link is authoritative over shared-id fallback.
    }
    const Element* elem = element_tree_.GetElement(elem_id);
    if (elem && elem->GetSharedId() == update.actor_shared_id) return id;
  }
  return spatialblend::kInvalidEntityId;
}

void Space::RecordRemoteBindingMatchesLocked(
    spatialblend::entity_id_t actor_id,
    const std::vector<RelationPayload>& payloads) {
  auto bind_it = bindings_.find(actor_id);
  if (bind_it == bindings_.end()) return;

  const auto referent_of = [&](const RelationPayload& payload) -> const Element* {
    if (payload.resolved_local_element_id != kInvalidElementId) {
      if (const Element* resolved =
              element_tree_.GetElement(payload.resolved_local_element_id)) {
        return resolved;
      }
    }
    if (referent_resolve_mode_ == ReferentResolveMode::kLinkerOnly) return nullptr;
    if (payload.referent_shared_id == kInvalidSharedElementId) return nullptr;
    return GetReferentResolver().Resolve(payload.referent_shared_id,
                                         element_tree_);
  };
  const auto in_scope = [&](const Binding& binding, element_id_t element) {
    if (binding.explicit_scope_root == kInvalidElementId) return true;
    for (element_id_t id = element; id != kInvalidElementId;) {
      if (id == binding.explicit_scope_root) return true;
      const Element* node = element_tree_.GetElement(id);
      if (!node) return false;
      id = node->GetParentId();
    }
    return false;
  };

  for (auto& binding : bind_it->second) {
    std::vector<Binding::MatchResult> matches;
    std::unordered_set<element_id_t> seen_elements;
    std::unordered_set<shared_element_id_t> seen_shared;
    for (const auto& payload : payloads) {
      if (payload.kernel_type != binding.kernel_type) continue;
      // An omitted anchor name is the root anchor, matching the root-payload
      // filter in OnRemoteUpdate.
      if (payload.subject_anchor_name.empty()) {
        if (binding.anchor_name != "root") continue;
      } else if (payload.subject_anchor_name != binding.anchor_name) {
        continue;
      }
      const Element* referent = referent_of(payload);
      if (!referent || !in_scope(binding, referent->GetId())) continue;
      if (!binding.tag_filter.empty() &&
          binding.kernel_type == RelationKernelType::kAvatarAffordancePose &&
          referent->GetTag() != binding.tag_filter) {
        continue;
      }
      if (!payload.enabled) continue;
      const element_id_t element_id = referent->GetId();
      const shared_element_id_t shared_id =
          payload.referent_shared_id != kInvalidSharedElementId
              ? payload.referent_shared_id
              : referent->GetSharedId();
      if (!seen_elements.insert(element_id).second) continue;
      if (shared_id != kInvalidSharedElementId &&
          !seen_shared.insert(shared_id).second) {
        continue;
      }
      Binding::MatchResult match;
      match.element_id = element_id;
      match.referent_shared_id = payload.referent_shared_id;
      match.eval.grounded = payload.grounded_relation;
      match.eval.support = payload.support;
      matches.push_back(std::move(match));
    }
    // Every delivery is authoritative for every binding, including one this
    // snapshot does not mention. An empty snapshot, a disabled anchor, or a
    // referent that no longer resolves clears the previous rows.
    binding.latest_matches = std::move(matches);
  }
}

void Space::OnRemoteUpdate(const RemoteRelationUpdate& update) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  // A malformed time field must not poison receipt clocks, sequence fences, or
  // phase state. source_clock==0 remains the documented legacy sentinel.
  if (!std::isfinite(update.source_clock) ||
      !std::isfinite(update.step_duration)) {
    return;
  }
  // Legacy receipt accounting is intentionally retained. Timestamped input is
  // different: admission must finish before it can become observable through
  // the receipt ring/callback path.
  constexpr size_t kMaxReceivedUpdates = 100;
  auto record_received = [&]() {
    if (received_updates_.size() >= kMaxReceivedUpdates) {
      received_updates_.erase(received_updates_.begin());
    }
    received_updates_.push_back(update);
    ++received_update_count_;
  };
  const bool observation_update =
      update.timing == RelationUpdateTiming::kTimestampedObservation ||
      update.timing == RelationUpdateTiming::kObservationReset;
  if (update.timing == RelationUpdateTiming::kLegacy) record_received();
  if (!observation_update && update.timing != RelationUpdateTiming::kLegacy) {
    return;
  }

  // Resolve before any reset/new-epoch snapshot as well as the normal solve.
  // Otherwise a lazy shared-ID mode could falsely certify the default linker.
  ResolveReferentResolveModeLocked();
  // Linker lookups below are individually thread-safe but need not all see
  // the same map.  Fence the entire receiver resolution/solve transaction;
  // a graph mutation between these reads makes the captured counterpart stamp
  // explicitly unknown rather than claiming a coherent mixed map.
  const uint64_t linker_revision_before =
      hub_ ? hub_->GetLinker().GetGraphRevision() : 0;
  const auto scene_inputs_before = CaptureSceneInputStampLocked();
  const spatialblend::entity_id_t actor_id =
      FindActorForRemoteUpdateLocked(update);
  Actor* local_actor = FindActorLocked(actor_id);

  if (!local_actor) return;
  // Captured once for an update that has reached this receiver/actor.  It is
  // threaded unchanged to the snapshot produced by the actual solve; polling
  // and callback dispatch never read the clock again for that snapshot.
  const int64_t receiver_ingress_monotonic_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();

  if (observation_update) {
    if (update.source_epoch == 0 || !std::isfinite(update.source_time) ||
        update.source_time < 0.0 || update.sequence <= 0) {
      return;
    }
    // A known source observation sequence denotes an actual timestamped
    // sample, never a reset/control event. Unknown remains compatible with
    // older in-process timestamped producers.
    if (update.source_observation_sequence_known &&
        (update.timing != RelationUpdateTiming::kTimestampedObservation ||
         update.source_observation_sequence < 0)) {
      return;
    }
    // Prefer the complete source actor handle. A source without one uses the
    // explicit invalid-element fallback namespace, never the legacy key.
    const ObservationRemoteKey observation_key =
        update.actor_source_element_id != kInvalidElementId
            ? ObservationRemoteKey{update.actor_source_space_id,
                                   update.actor_source_element_id,
                                   kInvalidSharedElementId}
            : ObservationRemoteKey{update.source_space_id, kInvalidElementId,
                                   update.actor_shared_id};
    const uint64_t accepted_epoch =
        last_remote_observation_epoch_[observation_key];
    const int64_t accepted_sequence =
        last_remote_observation_sequence_[observation_key];
    if (update.source_epoch < accepted_epoch ||
        update.sequence <= accepted_sequence) {
      return;
    }
    const bool new_epoch = update.source_epoch > accepted_epoch;
    // Reset begins a distinct replay lifetime; accepting a same-epoch reset
    // would erase the time fence and permit a later timestamp regression.
    if (update.timing == RelationUpdateTiming::kObservationReset &&
        !new_epoch) {
      return;
    }
    if (update.timing == RelationUpdateTiming::kTimestampedObservation &&
        !new_epoch && has_remote_observation_time_.count(observation_key) &&
        update.source_time < last_remote_observation_time_[observation_key]) {
      return;
    }
    if (update.timing == RelationUpdateTiming::kObservationReset || new_epoch) {
      InvalidateSourceEvidencePhaseLocked(actor_id);
      retained_receiver_evaluations_.erase(actor_id);
      // No contained root survives a reset or a new epoch.
      ClearActorContainmentLocked(actor_id);
      // A reset is a cancellation control event: no synthetic phase time or
      // source-clock tick. It establishes the next temporal lifetime.
      previous_weights_.erase(actor_id);
      solver_warm_states_.erase(actor_id);
      proposal_attribution_.erase(actor_id);
        phase_tracker_.ClearOwner(PhaseOwner(actor_id));
      receiver_actor_seconds_.erase(actor_id);
      // The engagement timers belong to the clock epoch they were latched in.
      latency_trackers_.erase(actor_id);
      receiver_actor_updates_.erase(actor_id);
      // A receiver reset must be observable as a known empty target result.
      // Erasing this entry makes the C dispatch path skip cancellation.
      RealizationResult released;
      released.status = SolverStatus::kConverged;
      realization_results_[actor_id] = std::move(released);
      uint64_t counterpart_revision = 0;
      const bool counterpart_known = CaptureCounterpartRevisionLocked(
          linker_revision_before,
          hub_ ? hub_->GetLinker().GetGraphRevision() : 0,
          &counterpart_revision);
      CaptureRealizationSnapshot(actor_id, {}, {}, {}, {},
                                 realization_results_[actor_id],
                                 scene_inputs_before, &update,
                                 receiver_ingress_monotonic_ns,
                                 counterpart_known, counterpart_revision);
      {
        ReceiverProposal released_proposal;
        released_proposal.status = ProposalStatus::kNone;
        released_proposal.source_sequence = update.sequence;
        PublishProposalLocked(actor_id, std::move(released_proposal));
      }
      last_remote_observation_epoch_[observation_key] = update.source_epoch;
      last_remote_observation_sequence_[observation_key] = update.sequence;
      has_remote_observation_time_.erase(observation_key);
      if (update.timing == RelationUpdateTiming::kObservationReset) {
        record_received();
        return;
      }
    }
    if (update.source_epoch != last_remote_observation_epoch_[observation_key])
      return;
    last_remote_observation_sequence_[observation_key] = update.sequence;
    InvalidateSourceEvidencePhaseLocked(actor_id);
    last_remote_observation_time_[observation_key] = update.source_time;
    has_remote_observation_time_.insert(observation_key);
    // This is a watermark only, not a sum of unrelated actor deltas.
    space_clock_ = std::max(space_clock_, update.source_time);
    record_received();
  }

  // Dedupe an exact-replay delivery of an update this space already
  // applied -- observed ~1/10 runs under Transport::kGrpcTwoSessions, an
  // extra duplicate OnRemoteUpdate call for the very first delivered update
  // right after sb_connect_space's bootstrap/Subscribe-stream-settle window
  // (see unity_twin.h's Transport doc comment, and remote_update.h's `sequence` field for why a
  // content-based key isn't needed here: the source stamps a per-actor
  // monotonically increasing counter once, before it ever leaves the
  // source space, so a second delivery of the SAME update carries the SAME
  // sequence value). sequence == 0 is the "not assigned" sentinel (a
  // hand-built update, e.g. in a unit test) and always bypasses this check.
  if (!observation_update && update.sequence > 0) {
    auto key = std::make_pair(update.source_space_id, update.actor_shared_id);
    int64_t& last_seen = last_remote_sequence_[key];
    if (update.sequence <= last_seen) {
      return;  // exact replay of an already-applied update: no-op.
    }
    last_seen = update.sequence;
  }

  // Advance this space's own local clock on a genuine remote update too
  // -- otherwise a receiving space whose only actor is a remotely-driven
  // clone (never locally calling UpdateActorState) would have a clock frozen
  // at 0.0: see space_clock_'s doc comment in space.h for the full rationale.
  //
  // step_duration is NOT a real elapsed-time measurement -- Space::
  // UpdateActorState (source side) always sets it to the constant
  // update_interval_ (~0.033s), on EVERY call, regardless of whether this
  // particular delivery followed immediately from the last one or arrived
  // only after the movement-gate deadband held the source still for up to
  // kHeartbeatInterval frames (~1.0s) before the heartbeat forced a resend.
  // Advancing space_clock_ by step_duration therefore advances it by
  // delivery-count * update_interval_ regardless of real time elapsed --
  // under a heartbeat gap this badly under-runs real time (e.g. 39
  // deliveries * 0.033s = 1.287s of peer clock for a 6.0s real trajectory). Use the REAL elapsed source-side time
  // instead: the delta between this update's source_clock and the last
  // source_clock seen for this (source_space_id, actor_shared_id) key.
  // Falls back to step_duration (then update_interval_) when source_clock
  // can't yield a valid delta -- unset (<= 0.0, e.g. a hand-built update
  // that never populates it), first delivery for this key (no prior
  // source_clock to diff against), or non-monotonic (a stale/out-of-order
  // delivery, which should never regress the clock backward). This does
  // not change the movement-gate/heartbeat mechanism itself (its
  // contract: max real gap between propagations stays bounded by one
  // heartbeat interval) -- only how the RECEIVER accounts for the real time
  // that gap actually spanned.
  if (!observation_update) {
    InvalidateSourceEvidencePhaseLocked(actor_id);
    double delta = -1.0;
    if (update.source_clock > 0.0) {
      auto src_key =
          std::make_pair(update.source_space_id, update.actor_shared_id);
      double& last_source_clock = last_source_clock_[src_key];
      if (last_source_clock > 0.0 && update.source_clock > last_source_clock) {
        delta = update.source_clock - last_source_clock;
      }
      if (update.source_clock > last_source_clock) {
        last_source_clock = update.source_clock;
      }
    }
    if (delta <= 0.0) {
      delta = (update.step_duration > 0.0) ? update.step_duration
                                           : update_interval_;
    }
    space_clock_ += delta;
  }

  if (receiver_inputs_.count(actor_id) &&
      (rich_realization_enabled_ || rich_realization_actors_.count(actor_id))) {
    RetainedReceiverEvaluation retained;
    retained.update = update;
    // A declared discontinuity is consumed by the evaluation of the update that
    // carried it; a re-evaluation of the retained copy must not replay it (the
    // driver would teleport again).
    retained.update.has_exception = false;
    retained.update.exception_payload.clear();
    retained.original_ingress_ns = receiver_ingress_monotonic_ns;
    retained.receiver_element = local_actor->GetElementId();
    retained.requires_actor_link = hub_ &&
        update.actor_source_element_id != kInvalidElementId &&
        hub_->GetLinker().Translate(
            {update.actor_source_space_id, update.actor_source_element_id},
            space_id_) == retained.receiver_element;
    // Accepted newer/empty packets replace old intent even if solving fails.
    retained_receiver_evaluations_[actor_id] = std::move(retained);
  }
  EvaluateRemoteRelationLocked(update, actor_id, receiver_ingress_monotonic_ns,
                               false, linker_revision_before);
}

void Space::EvaluateRemoteRelationLocked(
    const RemoteRelationUpdate& update, spatialblend::entity_id_t actor_id,
    int64_t receiver_ingress_monotonic_ns, bool receiver_reevaluation,
    uint64_t source_actor_resolution_revision, bool carries_containment) {
  ResolveReferentResolveModeLocked();
  PruneObservedFacingPointsLocked();
  SyncNavigableObstaclesLocked();
  if (FindActorLocked(actor_id)) receiving_actors_.insert(actor_id);
  SyncEmbodiedMirrorsLocked();
  // A source delivery has already selected its actor. Its fence must begin
  // BEFORE that lookup, not at this factored solver entry. Reevaluation
  // resolves/checks the actor below and therefore owns a new before-sample.
  const uint64_t linker_revision_before = receiver_reevaluation
      ? (hub_ ? hub_->GetLinker().GetGraphRevision() : 0)
      : source_actor_resolution_revision;
  const auto scene_inputs_before = CaptureSceneInputStampLocked();
  auto* local_actor = FindActorLocked(actor_id);
  if (!local_actor) return;
  if (receiver_reevaluation) {
    const auto retained = retained_receiver_evaluations_.find(actor_id);
    const bool actor_link_live = retained != retained_receiver_evaluations_.end() &&
        (!retained->second.requires_actor_link ||
         (hub_ && hub_->GetLinker().Translate(
             {update.actor_source_space_id, update.actor_source_element_id},
             space_id_) == retained->second.receiver_element));
    if (retained == retained_receiver_evaluations_.end() || !actor_link_live ||
        local_actor->GetElementId() != retained->second.receiver_element ||
        FindActorForRemoteUpdateLocked(update) != actor_id) {
      retained_receiver_evaluations_.erase(actor_id);
      ClearReceiverTargetsLocked(actor_id);
      // Keep the historical revision monotonic within the active rich token,
      // but invalidate its source identity through the counterpart domain.
      ++counterpart_revision_;
      return;
    }
  }

  // SPEC S3 ev=tick: measure the substantive compute cost of this update
  // (referent resolution through the solver) when tracing is active. No
  // chrono calls at all when inactive, per spec's perf contract.
  const bool trace_active = !receiver_reevaluation &&
                           TraceSink::ForSpace(space_id_).Active();
  // The receiver actor's OWN source time (stillness and handover):
  // the sample time of a timestamped observation; for a legacy delivery the
  // source's elapsed clock at this delivery (source_clock: real time
  // across movement-gated gaps and heartbeats); without it, this actor's own
  // accumulated step_duration. Never the receiver's shared space clock, which
  // other actors' deliveries advance.
  if (!receiver_reevaluation) {
    ReceiverCadence& cadence = receiver_actor_updates_[actor_id];
    ++cadence.updates;
    cadence.timestamped =
        update.timing == RelationUpdateTiming::kTimestampedObservation;
    cadence.epoch = cadence.timestamped ? update.source_epoch : 0;
    cadence.quiet = update.quiet_since_last_send;
    cadence.sequence_known = update.sequence > 0;
    cadence.sequence_gap = update.sequence > 0 && cadence.last_sequence > 0 &&
                           update.sequence != cadence.last_sequence + 1;
    if (update.sequence > 0) cadence.last_sequence = update.sequence;
    // One clock source per actor epoch. The epoch is the clock domain (timestamped or legacy, and the
    // source epoch). Its source is latched by the epoch's first update and never switched: mixing
    // sources (a tick-accumulated start, then the sender's own clock at ~100 s) would jump the
    // timeline, and every timer latched on it with it. A new epoch starts a new timeline, so the
    // engagement timers latched on the old one end with it.
    const bool timestamped =
        update.timing == RelationUpdateTiming::kTimestampedObservation &&
        std::isfinite(update.source_time);
    ReceiverActorClock& clock = receiver_actor_seconds_[actor_id];
    if (clock.source == ReceiverActorClock::Source::kUnset ||
        clock.timestamped != timestamped || clock.epoch != cadence.epoch) {
      if (clock.source != ReceiverActorClock::Source::kUnset)
        latency_trackers_.erase(actor_id);
      clock = ReceiverActorClock{};
      clock.timestamped = timestamped;
      clock.epoch = cadence.epoch;
      if (timestamped) {
        clock.source = ReceiverActorClock::Source::kSampleTime;
      } else if (update.source_clock > 0.0 && std::isfinite(update.source_clock)) {
        clock.source = ReceiverActorClock::Source::kSourceClock;
      } else {
        clock.source = ReceiverActorClock::Source::kSteps;
      }
    }
    const bool has_step =
        std::isfinite(update.step_duration) && update.step_duration > 0.0;
    if (clock.source == ReceiverActorClock::Source::kSampleTime) {
      clock.seconds = update.source_time;
    } else if (clock.source == ReceiverActorClock::Source::kSourceClock &&
               update.source_clock > 0.0 && std::isfinite(update.source_clock)) {
      clock.seconds = std::max(clock.seconds, update.source_clock);
    } else if (has_step) {
      clock.seconds += update.step_duration;
    } else {
      // No time source at all: count updates at a nominal 30 Hz, so the
      // release dwell and the stillness hold still expire (after
      // ceil(dwell * 30) updates) instead of holding forever.
      clock.seconds += kClocklessUpdateSeconds;
    }
  }
  std::chrono::steady_clock::time_point tick_start;
  if (trace_active) tick_start = std::chrono::steady_clock::now();

  // Preserve every received payload for trace/accounting, but in linker-only
  // mode pass only explicitly linked referents downstream. This is a safety
  // boundary for full-scene retention: an unlinked object with a coincident
  // local dataset ID stays visible but cannot win policy, gating, or solving.
  std::vector<RelationPayload> incoming_payloads(update.payloads);
  std::vector<RelationPayload> payloads;
  // Per payload: the receiver has a counterpart of its referent (see the weights below).
  std::vector<bool> payload_has_counterpart;
  std::vector<int> resolved_index(incoming_payloads.size(), -1);
  payloads.reserve(incoming_payloads.size());
  std::vector<ObservedFacingDependency> point_dependencies;
  // A rich activation is a stricter causal-consumer boundary than ABI2's
  // legacy target stream. kInvalidSharedElementId is a sentinel, never a
  // receiver lookup key: historical shared-id lookup can otherwise select an
  // unrelated unshared element. A live ElementLinker correspondence remains
  // sufficient even when the two endpoints deliberately have no shared id.
  const bool rich_actor_active =
      rich_realization_enabled_ || rich_realization_actors_.count(actor_id) ||
      receiver_inputs_.count(actor_id);
  for (size_t i = 0; i < incoming_payloads.size(); ++i) {
    RelationPayload payload = incoming_payloads[i];
    // Receiver-local scratch state never authorizes a strict live mapping.
    // Re-derive it through this hub for every delivery.
    // Never trust sender-provided receiver-local scratch state.
    ResolveReceiverIdentitiesLocked(&payload);
    const bool has_live_explicit_link =
        payload.resolved_local_element_id != kInvalidElementId &&
        element_tree_.GetElement(payload.resolved_local_element_id) != nullptr;
    if (rich_actor_active &&
        payload.referent_shared_id == kInvalidSharedElementId &&
        !has_live_explicit_link) {
      continue;  // Preserve this raw row in the rich unmapped ledger.
    }
    if (referent_resolve_mode_ == ReferentResolveMode::kLinkerOnly &&
        !has_live_explicit_link) {
      continue;  // raw packet remains visible only in trace/accounting
    }
    if (payload.referent_retired && !has_live_explicit_link) {
      continue;  // its link ended: no counterpart, not even by shared id
    }
    const Element* referent = has_live_explicit_link
        ? element_tree_.GetElement(payload.resolved_local_element_id)
        : GetReferentResolver().Resolve(payload.referent_shared_id, element_tree_);
    const auto point = referent ? observed_facing_points_.find(referent->GetId())
                                : observed_facing_points_.end();
    if (payload.observed_facing_point || point != observed_facing_points_.end()) {
      const int64_t now = receiver_input_clock_();
      if (!payload.observed_facing_point || point == observed_facing_points_.end() ||
          payload.observed_logical_actor != point->second.logical_actor ||
          !(payload.observed_source_point == point->second.source_point) ||
          payload.observed_source_point_lease != point->second.source_point_lease ||
          payload.observed_valid_until_ns <= now ||
          !IsObservedFacingCandidateLocked(referent->GetId(), actor_id,
                                             payload.kernel_type, now)) continue;
      // Pin exact target point generation/revision and both original deadlines.
      point_dependencies.push_back({referent->GetId(), point->second.lease,
          point->second.revision,
          std::min(payload.observed_valid_until_ns, point->second.valid_until_ns)});
      auto& phase = point_dependencies.back().phase_identity;
      phase.actor_shared_id = update.actor_shared_id;
      phase.anchor_name = payload.subject_anchor_name;
      phase.referent_shared_id = payload.referent_shared_id;
      phase.kernel_type = payload.kernel_type;
      payload.resolved_local_element_id = referent->GetId();
    }
    resolved_index[i] = static_cast<int>(payloads.size());
    payload_has_counterpart.push_back(referent != nullptr);
    payloads.push_back(std::move(payload));
  }
  // Authoritative omission/rejection withdraws point history too, including
  // receive-only actors that never had local source bindings.
  for (const auto& old : observed_facing_dependencies_[actor_id]) {
    const bool retained = std::any_of(point_dependencies.begin(), point_dependencies.end(),
        [&](const ObservedFacingDependency& next) {
          return old.element == next.element && old.lease == next.lease &&
                 old.revision == next.revision &&
                 old.phase_identity.anchor_name == next.phase_identity.anchor_name &&
                 old.phase_identity.referent_shared_id == next.phase_identity.referent_shared_id &&
                 old.phase_identity.kernel_type == next.phase_identity.kernel_type;
        });
    if (!retained) phase_tracker_.ClearRelationForOwner(old.phase_identity, PhaseOwner(actor_id));
  }
  observed_facing_dependencies_[actor_id] = std::move(point_dependencies);

  // Look up the navigable spatial definition by walking up from the actor's
  // element to the nearest ancestor (e.g. the room) that defines one. This
  // constrains the solver to the walkable polygon, matching v1's
  // StepSyncSolver::ClampToPolygon behavior. Walking parents lets an actor
  // reference its own per-actor element (required so multiple avatars match
  // distinct shared_ids) while still inheriting the room's walkable boundary.
  const NavigableSpatialDefinition* navigable_def = nullptr;
  element_id_t nav_elem_id = local_actor->GetElementId();
  while (nav_elem_id != kInvalidElementId) {
    const Element* nav_elem = element_tree_.GetElement(nav_elem_id);
    if (!nav_elem) break;
    if (nav_elem->GetSpatialDefinition()) {
      navigable_def = dynamic_cast<const NavigableSpatialDefinition*>(
          nav_elem->GetSpatialDefinition());
      if (navigable_def) break;
    }
    nav_elem_id = nav_elem->GetParentId();
  }

  // Trace retains the received-payload denominator, including rejected strict
  // linker-only entries. They have depth/weight zero and never enter policy.
  std::vector<int> payload_depths(incoming_payloads.size(), 0);
  for (size_t i = 0; i < incoming_payloads.size(); ++i) {
    const int resolved = resolved_index[i];
    const RelationPayload* payload =
        resolved >= 0 ? &payloads[resolved] : &incoming_payloads[i];
    const Element* ref = nullptr;
    if (payload->resolved_local_element_id != kInvalidElementId) {
      ref = element_tree_.GetElement(payload->resolved_local_element_id);
    } else if (referent_resolve_mode_ != ReferentResolveMode::kLinkerOnly) {
      ref = element_tree_.GetElementBySharedId(payload->referent_shared_id);
    }
    int depth = 0;
    for (element_id_t e = ref ? ref->GetId() : kInvalidElementId;
         e != kInvalidElementId;) {
      const Element* el = element_tree_.GetElement(e);
      if (!el) break;
      ++depth;
      e = el->GetParentId();
    }
    payload_depths[i] = depth;
  }
  // Resolve the realization policy family once (lazy, env-driven
  // default), unless SetRealizationPolicy already pre-empted it. Default
  // kProgressive is the default policy_config.
  ResolveRealizationPolicyLocked();

  // Realization weights: the progressive policy (softmax over
  // seconds-to-arrival, temperature decaying toward one-hot on the approached
  // referent) so the blend commits to a single object instead of a multi-object
  // centroid.
  // The parameters are a declared per-space config
  // (ProductionRealizationPolicyConfig: γ = 2, a recorded deviation from the
  // paper's γ = 1 -- see realization_policy.h), settable via
  // SetRealizationPolicyParams and traced as ev=policy.
  RealizationPolicyConfig policy_config = realization_policy_config_;
  policy_config.type = realization_policy_type_;
  // A payload referent with no counterpart on this receiver takes no part in the
  // weight normalisation, so the present terms sum to 1. v1 does the same: SoftAnnealingPlanner::
  // BuildTargets skips a target whose entity is absent from the receiver BEFORE its steps enter the
  // weights, which are computed over the present targets. The absent payload keeps weight 0 (it has no
  // approach point or look-at to ground anyway).
  std::vector<double> weights(payloads.size(), 0.0);
  if (std::none_of(payload_has_counterpart.begin(), payload_has_counterpart.end(), [](bool b) { return b; })) {
    // Nothing maps on this receiver (the proposal is NONE, never a hold): no payload is present, so
    // nothing is normalised away; the policy sees the payloads as before.
    weights = ComputeRealizationWeights(payloads, policy_config);
  } else {
    std::vector<size_t> present;
    std::vector<RelationPayload> present_payloads;
    for (size_t i = 0; i < payloads.size(); ++i) {
      if (!payload_has_counterpart[i]) continue;
      present.push_back(i);
      present_payloads.push_back(payloads[i]);
    }
    const std::vector<double> present_weights =
        ComputeRealizationWeights(present_payloads, policy_config);
    for (size_t k = 0; k < present.size() && k < present_weights.size(); ++k)
      weights[present[k]] = present_weights[k];
  }
  previous_weights_[actor_id] = weights;
  // Record the policy's inputs, weights and per-channel policy
  // argmax (observation only; derived from `weights`, never recomputed).
  // Captured only while observed (tracing on or a getter enabled capture).
  // The REALIZED driving relation is filled in after the solve and the
  // anchor gate (AnnotateRealizedDriving below).
  const bool capture = ObservabilityCaptureActiveLocked();
  PolicyObservation* policy_obs = nullptr;
  if (capture) {
    policy_obs = &policy_observations_[actor_id];
    policy_obs->valid = true;
    policy_obs->time_seconds = space_clock_;
    policy_obs->config = policy_config;
    policy_obs->payloads = payloads;
    policy_obs->weights = weights;
    policy_obs->channels =
        DescribeRealizationChannels(payloads, weights, policy_config);
  } else {
    auto stale = policy_observations_.find(actor_id);
    if (stale != policy_observations_.end()) stale->second.valid = false;
  }
  if (auto stale = solver_observations_.find(actor_id);
      stale != solver_observations_.end())
    stale->second.valid = false;
  std::vector<bool> payload_is_driving;
  if (policy_obs) {
    payload_is_driving.assign(payloads.size(), false);
    for (const auto& channel : policy_obs->channels) {
      if (channel.has_driving) payload_is_driving[channel.driving_index] = true;
    }
  }

  // The root pose must be solved from spacing/facing payloads ONLY — an
  // affordance-pose payload (hand/pelvis anchor relation) must never drag
  // root position or yaw (the groups are solved independently). root_payloads is a
  // filtered view; `payloads`/`weights` (full, unfiltered) are still used
  // below for trace/phase-tracker emission and for the per-anchor targets
  // (AppendAnchorTargets does its own grouping via SolveActorTargets).
  std::vector<RelationPayload> root_payloads;
  std::vector<double> root_weights;
  std::vector<size_t> root_payload_index;  // root_payloads[k] = payloads[...]
  const auto receiver = receiver_inputs_.find(actor_id);
  const bool external_receiver = receiver != receiver_inputs_.end();
  const std::string root_anchor_name = RootAnchorNameLocked(actor_id);
  root_payloads.reserve(payloads.size());
  root_weights.reserve(payloads.size());
  // A relation of another anchor is that anchor's (BuildAnchorProposalsLocked), whatever its kernel and for every actor
  // (the anchor is a parameter of the relation definition).
  // An unnamed (or "root") source anchor is the legacy root: it is this actor's root unless a receiver input names
  // its root otherwise (an anonymous relation never aliases a named receiver root).
  const auto of_root = [&](const RelationPayload& p) {
    if (p.subject_anchor_name == root_anchor_name) return true;
    return !external_receiver && (p.subject_anchor_name.empty() || p.subject_anchor_name == "root");
  };
  for (size_t i = 0; i < payloads.size(); ++i) {
    if (payloads[i].kernel_type == RelationKernelType::kAvatarAffordancePose ||
        (rich_actor_active &&
         payloads[i].kernel_type == RelationKernelType::kAvatarFacing) ||
        !of_root(payloads[i])) {
      continue;
    }
    root_payloads.push_back(payloads[i]);
    root_weights.push_back((i < weights.size()) ? weights[i] : 1.0);
    root_payload_index.push_back(i);
  }

  // SPEC S3 ev=payload: one line per incoming payload, after weights are
  // computed (weight/depth are only known here).
  if (trace_active) {
    auto& sink = TraceSink::ForSpace(space_id_);
    for (size_t i = 0; i < incoming_payloads.size(); ++i) {
      const auto& p = incoming_payloads[i];
      const int resolved = resolved_index[i];
      nlohmann::ordered_json ev;
      ev["ev"] = "payload";
      ev["t"] = space_clock_;
      ev["space"] = space_id_;
      ev["actor"] = update.actor_shared_id;
      ev["referent"] = p.referent_shared_id;
      // SPEC S3: never emit an empty anchor name -- "root" is the
      // convention for the actor's root anchor (see space.h's
      // SetAnchorWorldPose comment); an empty subject_anchor_name here would
      // otherwise mean the same thing but violate the spec's "anchor is
      // always a non-empty name" contract for consumers (a consumer may key
      // frames by anchor name).
      ev["anchor"] =
          p.subject_anchor_name.empty() ? "root" : p.subject_anchor_name;
      ev["kernel"] = static_cast<int>(p.kernel_type);
      ev["R"] = p.grounded_relation.proximity;
      ev["dist"] = p.grounded_relation.support_distance_m;
      ev["depth"] = payload_depths[i];
      ev["weight"] =
          (resolved >= 0 && static_cast<size_t>(resolved) < weights.size())
              ? weights[resolved]
              : 0.0;
      ev["est_time"] = p.estimated_time;
      ev["via"] =
          resolved < 0 ? "unresolved"
          : payloads[resolved].resolved_local_element_id != kInvalidElementId
              ? "linker"
              : "sharedid";
      // Trace format 2: engaged part and whether this payload is
      // its channel's POLICY argmax (the realized driving relation, after
      // gating, is on ev=channel).
      ev["sub_region"] = p.support.sub_region_name;
      ev["driving"] = resolved >= 0 &&
                      static_cast<size_t>(resolved) < payload_is_driving.size() &&
                      payload_is_driving[resolved];
      sink.Emit(ev);
    }
    // Policy type and parameters, once per revision.
    EmitPolicyTraceIfChangedLocked(space_clock_);
  }

  // Query rows must exist before EmitRelationTransitions: the C callback
  // reads them under the space lock. R and the referent come from the
  // payload; the phase is the tracker update inside the emit. Dormant rows
  // are reported beside the realized ones and are not solver input.
  {
    RecordRemoteBindingMatchesLocked(actor_id, payloads);
  }

  // Emit relation lifecycle events target-side: feed the
  // incoming payloads' proximity and realization weight into the phase tracker.
  for (size_t i = 0; !receiver_reevaluation && i < payloads.size(); ++i) {
    const auto& p = payloads[i];
    RelationEvent id;
    // New local observation sessions can have invalid/identical shared ids.
    // Receiver phase state must remain actor-local in that case.
    id.actor_shared_id = update.actor_shared_id;
    id.anchor_name = p.subject_anchor_name;
    id.referent_shared_id = p.referent_shared_id;
    id.kernel_type = p.kernel_type;
    double proximity = p.grounded_relation.proximity;
    double weight = (i < weights.size()) ? weights[i] : 0.0;
    EmitRelationTransitions(
        id, proximity, weight,
        update.timing == RelationUpdateTiming::kTimestampedObservation
            ? update.source_time
            : space_clock_,
        PhaseOwner(actor_id));
  }

  // Containment transfer, receiver side. The source says only WHICH transfer
  // element its root is inside (the identity triple); there is no pose to map,
  // no blend and no HANDOVER. The transfer leaf is the actor's scope, so the
  // empty-scope mapping maps the root and this state survives only
  // as the CONTAIN_ENTER / CONTAIN_EXIT cause bits and the transfer flag.
  {
    ActorContainment next;
    if (carries_containment && update.containment) {
      const ContainmentObservation& c = *update.containment;
      next.contained = true;
      next.scope_shared_id = c.scope_shared_id;
      next.scope_source_space_id = c.scope_source_space_id;
      next.scope_source_element_id = c.scope_source_element_id;
    }
    // InjectRelationPayloads carries scene-level payloads only: the packet says
    // nothing about containment, so it must not end a contained state.
    if (carries_containment) SetActorContainmentLocked(actor_id, next);
  }
  if (payloads.empty()) {
    // A strict-mode all-unlinked update (or an explicit empty authoritative
    // snapshot) releases stale root and anchor targets rather than preserving
    // the last relation indefinitely.
    previous_weights_[actor_id].clear();
    solver_warm_states_.erase(actor_id);
    // A release ends the temporal lifetime of the proposals: the next relation
    // starts a fresh attribution, and a source that let go cancels its timers.
    proposal_attribution_.erase(actor_id);
    latency_trackers_.erase(actor_id);
    receiver_actor_seconds_.erase(actor_id);
    receiver_actor_updates_.erase(actor_id);
    RealizationResult released;
    released.status = SolverStatus::kConverged;
    realization_results_[actor_id] = std::move(released);
    uint64_t counterpart_revision = 0;
    const bool counterpart_known = CaptureCounterpartRevisionLocked(
        linker_revision_before,
        hub_ ? hub_->GetLinker().GetGraphRevision() : 0,
        &counterpart_revision);
    CaptureRealizationSnapshot(
        actor_id, incoming_payloads, resolved_index, payloads, weights,
        realization_results_[actor_id], scene_inputs_before, &update,
        receiver_ingress_monotonic_ns,
        counterpart_known, counterpart_revision);
    // Empty source/control geometry remains uncertified as body evidence.
    // It still consumes one receiver evaluation, so retrying the same input
    // cannot create an unbounded sequence of empty snapshots.
    const auto retained = retained_receiver_evaluations_.find(actor_id);
    if (retained != retained_receiver_evaluations_.end())
      retained->second.last_evaluated_input = CurrentReceiverInputLocked(actor_id);
    {
      // The source released every relation: nothing to propose (the driver
      // keeps its body). Sample identity is reported for staleness only.
      ReceiverProposal released_proposal;
      released_proposal.status = ProposalStatus::kNone;
      released_proposal.source_sequence = update.sequence;
      released_proposal.source_shared_id = update.actor_shared_id;
      std::vector<ActorEvent> events;
      // A source-app exception rides this update even though it carries no relation: the
      // source raised it at its interaction, and core propagates it at once.
      // It is queued before the set it belongs to
      // is published, as on the solve path.
      if (update.has_exception && !receiver_reevaluation) {
        ActorEvent event;
        event.kind = ActorEvent::Kind::kSourceException;
        event.revision = proposal_revision_ + 1;
        event.source_sequence = update.sequence;
        event.source_shared_id = update.actor_shared_id;
        event.exception_id = update.exception_id;
        event.payload = update.exception_payload;
        events.push_back(std::move(event));
        released_proposal.cause |= kCauseSourceException;
      }
      PublishProposalLocked(actor_id, std::move(released_proposal),
                            std::move(events));
    }
    return;
  }

  // The embodied pose is SAMPLED from the actor's port: the
  // driver is its only writer, core keeps no receiver pose and
  // never substitutes one. Without a valid sample there is nothing to continue
  // from and the proposal says so explicitly (kNoSample); the next source
  // update after the driver publishes solves.
  ReceiverInputStamp receiver_input;
  EmbodiedSample sample;
  const auto actor_mailbox = MailboxFor(actor_id);
  if (actor_mailbox) sample = actor_mailbox->port.Sample();
  const int64_t solve_monotonic_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  bool receiver_input_live = true;
  if (external_receiver) {
    receiver_input = CurrentReceiverInputLocked(actor_id);
    receiver_input_live = receiver_input.known;
  }
  const auto make_proposal = [&](ProposalStatus status) {
    ReceiverProposal p;
    p.status = status;
    p.anchor_id = kRootAnchorId;
    p.anchor_name = root_anchor_name;
    p.source_sequence = update.sequence;
    p.source_shared_id = update.actor_shared_id;
    p.request_sample = sample;
    // The source's orientation beyond yaw: a pass-through on the root anchor.
    p.orientation_type = update.source_orientation_type;
    p.pitch = update.source_pitch;
    p.roll = update.source_roll;
    p.z = sample.valid ? sample.z : 0.0;
    p.flags = sample.valid ? kAnchorFlagSampled : 0u;
    p.sample_valid = sample.valid;
    if (sample.valid) {
      p.sample_sequence = sample.sequence;
      p.sample_epoch = sample.epoch;
      p.sample_timestamp_s = sample.timestamp_s;
      p.sample_age_s =
          std::max(0.0, (solve_monotonic_ns - sample.published_ns) * 1e-9);
    }
    return p;
  };
  if (!sample.valid || !receiver_input_live) {
    // ONE proposal per solve: the NO_SAMPLE proposal below is the only one
    // published (a NONE first would read as "released" to the driver).
    ClearReceiverTargetsLocked(actor_id, /*preserve_history=*/false,
                               /*publish_released=*/false);
    auto& failed = realization_results_[actor_id];
    failed.status = SolverStatus::kFailed;
    uint64_t counterpart_revision = 0;
    const bool counterpart_known = CaptureCounterpartRevisionLocked(
        linker_revision_before,
        hub_ ? hub_->GetLinker().GetGraphRevision() : 0,
        &counterpart_revision);
    CaptureRealizationSnapshot(
        actor_id, incoming_payloads, resolved_index, payloads, weights,
        failed, scene_inputs_before, &update, receiver_ingress_monotonic_ns,
        counterpart_known, counterpart_revision);
    ReceiverProposal none = make_proposal(ProposalStatus::kNoSample);
    none.sample_valid = false;
    // The root has no sample: its own proposal says so, and every other anchor is still answered from its own sample
    // (one path per anchor: the root's sample gates only the root).
    {
      const auto prior_attr = proposal_attribution_.find(actor_id);
      const bool actor_first = prior_attr == proposal_attribution_.end() || !prior_attr->second.AnyPrev();
      BuildAnchorProposalsLocked(actor_id, update, payloads, weights, failed, navigable_def, root_anchor_name,
                                 receiver_reevaluation, update.has_exception || actor_first, /*root_cause=*/0, &none);
    }
    std::vector<ActorEvent> events;
    const bool placed_anchor = std::any_of(
        none.anchors.begin(), none.anchors.end(), [](const AnchorProposal& a) {
          return a.placement && (a.position_requested || a.yaw_requested);
        });
    if (!receiver_reevaluation && (update.has_exception || placed_anchor)) {
      ActorEvent event;
      event.kind = ActorEvent::Kind::kSourceException;
      event.revision = proposal_revision_ + 1;
      event.source_sequence = update.sequence;
      event.source_shared_id = update.actor_shared_id;
      event.exception_id =
          update.has_exception ? update.exception_id : kExceptionJoin;
      if (update.has_exception) event.payload = update.exception_payload;
      events.push_back(std::move(event));
    }
    PublishProposalLocked(actor_id, std::move(none), std::move(events));
    return;
  }
  const Eigen::Vector2d current_pos(sample.x, sample.y);
  const double current_yaw = sample.yaw;

  // Continuation from the embodied pose: the tracking pass, the restarts and
  // the Facing sign rule are seeded at the sample, never at core's previous
  // answer. A source-app exception (a teleport, the first join, ...) ends the
  // continuation: the absolute relation is solved.
  SolverWarmStart warm_start;
  warm_start.has_position = true;
  warm_start.position = current_pos;
  warm_start.has_yaw = true;
  warm_start.yaw = current_yaw;
  // `cold` only echoes that this proposal follows NO_SAMPLE or a declared source
  // discontinuity: the rules are the same (v1 split: one referent answers with
  // its approach point, several descend from the embodied sample; no global
  // search exists to fall back on).
  // The first proposal is the ACTOR's (its join): no anchor of it has proposed before. It is placed, as a source
  // exception is, for every anchor (one first-proposal rule).
  const auto prior = proposal_attribution_.find(actor_id);
  const bool first_proposal = prior == proposal_attribution_.end() || !prior->second.AnyPrev();
  const AnchorCauseState* prior_root = nullptr;
  if (prior != proposal_attribution_.end())
    if (const auto it = prior->second.anchors.find(kRootAnchorId); it != prior->second.anchors.end())
      prior_root = &it->second;
  const bool embodied_reset_now =
      prior_root && prior_root->has_prev &&
      (prior_root->sample_epoch != sample.epoch ||
       ((sample.flags & kEmbodiedTeleported) != 0 && prior_root->teleport_sequence != sample.sequence));
  // Cold marks a first proposal, a declared discontinuity, or an embodied
  // reset. Continuation still starts at the embodied sample. The first
  // sample and a declared discontinuity are placed at the approach pose; an
  // embodied reset is not, it continues the ratio.
  warm_start.cold =
      update.has_exception || first_proposal || embodied_reset_now;
  // The live rule of EVERY anchor, the root included: core proposes
  // the PURSUED target, the ratio continuation of each relation (RatioTargetDistance: 1/n of the receiver's own
  // distance per update), and the driver realizes it within its own kinematic bounds. The placed answer (the referent
  // chart's image of the source pose) is kept only where a destination is explicitly requested:
  // the first proposal (arrival placement) and a source exception (a teleport, a join).
  warm_start.absolute = update.has_exception || first_proposal;
  placed_solve_[actor_id] = warm_start.absolute;
  // The root's relations in the room's scope, by the solve every anchor's room-scope relations take
  // (SolveRoomScopeLocked: the continuation from the sample, the leaf-relative yaw). No relation issues a request (every
  // update count is invalid: a pause, a static source): the SOLVE answers at its origin (the sample, brought into F
  // when it lies outside), a position with NO_REQUEST (as v1: no valid count is no
  // request and no plan, so the entity stays where it is). The PUBLISHED anchor keeps its retained pending target
  // (RetainIdlePositionLocked below: no request is neither a sampled-position hold nor a release).
  SolverResult solve_result = SolveRoomScopeLocked(sample, root_payloads, root_weights, navigable_def,
                                                   warm_start.absolute, warm_start.cold, actor_id, kRootAnchorId);
  {
    SolverWarmState& counters = solver_warm_states_[actor_id];
    ++counters.solves;
    if (solve_result.objective_warm_start_used) ++counters.warm_started_solves;
  }
  const double scene_yaw = solve_result.yaw;  // before the containment blend
  const bool solve_result_yaw_specified = solve_result.yaw_specified;
  // The anchor targets (AffordancePose / Seat) are absolute poses solved and
  // reach-clamped against the root: they are placed from the scene solve plus
  // the containment blend.
  SolverResult anchor_basis = solve_result;
  const Eigen::Vector2d scene_position = solve_result.position;
  if (capture) {
    SolverObservation& observed = solver_observations_[actor_id];
    observed.valid = true;
    observed.time_seconds = space_clock_;
    observed.solve = solve_result;
    observed.scene_position = scene_position;
    observed.scene_yaw = scene_yaw;
    observed.goal_payload_index = root_payload_index;
  }

  RealizationResult realization;
  realization.status = solve_result.status;
  // No position referent maps on the receiver: no root target, the proposal
  // is NONE, and the engaged body-anchor targets are still realized (they do
  // not depend on the root mapping). Only an invalid policy vector yields
  // no targets at all.
  const bool no_referent = solve_result.no_grounded_referent;
  const bool not_converged =
      solve_result.status == SolverStatus::kNotConverged;
  ReceiverProposal proposal = make_proposal(ProposalStatus::kNone);
  // A budget hit (the per-solve evaluation-count safety net) publishes its incumbent as an inexact proposal; the solve is not
  // converged, and the harness counts the frame as n_budget_hit.
  if (solve_result.status == SolverStatus::kConverged ||
      solve_result.status == SolverStatus::kFallback ||
      solve_result.status == SolverStatus::kBudgetHit || no_referent ||
      not_converged) {
    RealizationTarget target;
    // The target's arrival_time stays 0: pacing belongs to the driver
    // (there is no arrival budget).
    target.arrival_time = 0.0;
    // Receiver-input roots are explicitly named; "root" is only the legacy
    // default.
    target.anchor_name = root_anchor_name;
    target.position = solve_result.position;
    target.yaw = solve_result.yaw;
    target.kernel_type = RelationKernelType::kAvatarSpacing;
    target.weight = 1.0;
    realization.targets.push_back(target);

    // Per-anchor affordance targets (hysteresis-gated, reach-clamped),
    // appended AFTER root.
    AppendAnchorTargets(actor_id, anchor_basis.position, anchor_basis.yaw,
                        payloads, weights, navigable_def, realization);
    // The root yaw is the Facing yaw. There is no seat authority over it:
    // the seat is the pelvis anchor's own
    // proposal, and the driver / IK realizes it.
    const double root_yaw = scene_yaw;
    if (rich_actor_active) {
      AppendFacingTargets(actor_id, payloads, weights, navigable_def,
                          realization);
    }
    // No referent maps: no root target is reported, but the
    // engaged body-anchor targets are. A loop-guard hit is not an
    // answer either: the proposal carries the last trial as a diagnostic.
    if (no_referent || not_converged)
      realization.targets.erase(realization.targets.begin());
    realization_results_[actor_id] = realization;

    // ---- the proposal --------------------------------------------------
    proposal.status = no_referent     ? ProposalStatus::kNone
                      : not_converged ? ProposalStatus::kNotConverged
                                      : ProposalStatus::kInexact;
    proposal.x = solve_result.position.x();
    proposal.y = solve_result.position.y();
    // With no referent to map the pose fields are the embodied sample (position and
    // yaw); the status says nothing is proposed.
    proposal.yaw = no_referent ? current_yaw : root_yaw;
    proposal.yaw_turn = no_referent ? 0.0 : solve_result.yaw_turn;  // ABI 4.12 (from the same sample yaw)
    if (!no_referent && !not_converged) {
      proposal.placement = warm_start.absolute;
      proposal.position_active = solve_result.position_active;
      proposal.yaw_active = solve_result.yaw_active;
      proposal.position_requested = solve_result.position_requested;
      proposal.yaw_requested = solve_result.yaw_requested;
    }
    proposal.z = sample.z;
    proposal.anchor_id = kRootAnchorId;
    proposal.anchor_name = root_anchor_name;
    RetainIdlePositionLocked(actor_id, &proposal);
    // No yaw is specified when nothing authors one: no Facing relation entered
    // the solve. The driver then faces its direction of motion (v1: yaw
    // followed the path tangent). Containment does not author a yaw either.
    if (!solve_result_yaw_specified) proposal.flags |= kAnchorFlagYawUnspecified;
    // No new position request when the source is static, receding or not
    // closing (n = infinity). The solve starts at the current sample, while
    // the snapshot retains an active unfinished target selected above. No
    // POSITION_UNSPECIFIED flag is introduced; NONE still releases authority.
    // The root anchor is engaged while the source root is on, or touches, a
    // non-containment referent (engagement.h): a geometric fact, no hysteresis.
    for (size_t k = 0; k < root_payloads.size(); ++k) {
      const RelationPayload& rp = root_payloads[k];
      if (!rp.enabled || root_weights[k] <= 0.0 || !IsEngaged(rp)) continue;
      if (rp.kernel_type != RelationKernelType::kAvatarSpacing) continue;
      const Element* ref =
          rp.resolved_local_element_id != kInvalidElementId
              ? element_tree_.GetElement(rp.resolved_local_element_id)
              : GetReferentResolver().Resolve(rp.referent_shared_id,
                                              element_tree_);
      if (!ref || ref->HasGeometryRole(GeometryRole::kNavigable)) continue;
      proposal.flags |= kAnchorFlagEngaged;
      break;
    }
    // The driving referent of the root's solve (sb_anchor_proposal_t::referent_shared_id): the
    // heaviest root relation in it (Spacing or Facing, positive weight), the first of a tie, the
    // same rule as every other anchor's room-scope solve (anchor_proposals.cc). 0 when nothing
    // maps (no_referent).
    if (!no_referent) {
      double heaviest = 0.0;
      for (size_t k = 0; k < root_payloads.size(); ++k) {
        const RelationPayload& rp = root_payloads[k];
        if (!rp.enabled || !(root_weights[k] > heaviest)) continue;
        if (rp.kernel_type != RelationKernelType::kAvatarSpacing &&
            rp.kernel_type != RelationKernelType::kAvatarFacing)
          continue;
        heaviest = root_weights[k];
        proposal.referent_shared_id = rp.referent_shared_id;
      }
    }
    // Relation error at the target and at the SAMPLED pose, and whether each relation is
    // satisfied at the sampled pose (the ARRIVED terms, below).
    //  * An interaction the source holds (a Spacing contact, engagement.h) is arrived when its own
    //    relation is satisfied at the sample: the error in metres (R error times that relation's
    //    scope diameter) within the solver's declared tolerance.
    //  * A Facing relation is arrived by v1's definition (relation_evaluator.h kFacingArrivalRad):
    //    the signed angle, no tolerance borrowed from a length.
    // Whether the OTHER positive-weight relations are satisfied is not part of an interaction's
    // arrival: under a policy that weights several referents at once (Uniform) no pose satisfies
    // them all, and a body at its contact would never arrive.
    // The relation error is a measurement of how well each relation is reproduced, so it is reported for EVERY enabled
    // Spacing / Facing relation with a counterpart, whatever its weight this update: a relation the policy weighs 0 (no
    // valid count: a source that spins in place makes no Spacing request) is still a relation the receiver reproduces
    // or not. Whether a relation DROVE the solve is what the arrival and the status read, from the positive-weight
    // relations only (driving_facing_err below).
    bool contact_seen = false, contact_arrived = true;
    bool other_seen = false, other_arrived = true;
    double driving_facing_err = -1.0;  // the Facing error over the positive-weight relations (the status' input)
    {
      TargetEvaluationOptions eval_options;
      eval_options.resolver = &GetReferentResolver();
      const double arrival_eps_m = solver_.config().objective_improve_epsilon_m;
      double heaviest = -1.0;
      for (size_t k = 0; k < root_payloads.size(); ++k) {
        const RelationPayload& rp = root_payloads[k];
        const bool spacing = rp.kernel_type == RelationKernelType::kAvatarSpacing;
        const bool facing = rp.kernel_type == RelationKernelType::kAvatarFacing;
        if ((!spacing && !facing) || !rp.enabled) continue;
        const bool driving = root_weights[k] > 0.0;
        const PreparedTargetRelation prepared =
            PrepareTargetRelation(rp, element_tree_, eval_options);
        if (!prepared.ok()) continue;
        // The height is the embodied sample's own, the port being the source of truth: the root anchor's Facing is
        // evaluated where the root anchor is, as the source evaluated it. There is no invented
        // eye height. (v1 has none either: Angular1D reads kPosition2D and the yaw.) A Facing
        // bound to another anchor (the head) is reported by that anchor's own proposal at that
        // anchor's own sample (anchor_proposals.cc). Spacing is planar: z = 0 as on the source.
        const TargetAnchorPose at_target{
            Eigen::Vector3d(proposal.x, proposal.y, spacing ? 0.0 : proposal.z),
            proposal.yaw};
        const TargetAnchorPose at_sample{
            Eigen::Vector3d(current_pos.x(), current_pos.y(),
                            spacing ? 0.0 : sample.z),
            current_yaw};
        const auto et = EvaluatePreparedTargetRelation(prepared, at_target,
                                                       element_tree_);
        const auto es = EvaluatePreparedTargetRelation(prepared, at_sample,
                                                       element_tree_);
        double& err_t = spacing ? proposal.err_spacing_target
                                : proposal.err_facing_target;
        double& err_s = spacing ? proposal.err_spacing_sample
                                : proposal.err_facing_sample;
        if (et.ok()) err_t = std::max(err_t, std::fabs(et.r_tgt - et.r_src));
        if (es.ok()) err_s = std::max(err_s, std::fabs(es.r_tgt - es.r_src));
        if (!driving) continue;
        if (facing && et.ok()) driving_facing_err = std::max(driving_facing_err, std::fabs(et.r_tgt - et.r_src));
        if (es.ok()) {
          bool arrived_k = false;
          if (spacing) {
            const double scale_k = prepared.normalizing_scope_diameter_m > 1e-6
                                       ? prepared.normalizing_scope_diameter_m
                                       : 1.0;
            arrived_k = std::fabs(es.r_tgt - es.r_src) * scale_k <= arrival_eps_m;
          } else {
            arrived_k = FacingArrived(rp, es);
          }
          if (IsTimedSpacingContact(rp)) {
            contact_seen = true;
            contact_arrived = contact_arrived && arrived_k;
          } else {
            other_seen = true;
            other_arrived = other_arrived && arrived_k;
          }
        }
        if (spacing && root_weights[k] > heaviest &&
            prepared.normalizing_scope_diameter_m > 0.0) {
          heaviest = root_weights[k];
          proposal.err_scale_m = prepared.normalizing_scope_diameter_m;
        }
      }
      // A position stage reports its own residual: within 1 mm is EXACT, and
      // a fraction the local step cannot close — including a cold solve that
      // stays in its basin — is INEXACT. R_tgt − R_src is not that residual.
      // With no position intent, facing keeps the angular R report.
      if (!no_referent && !not_converged) {
        if (solve_result.objective_relation_count > 0) {
          proposal.status = solve_result.objective_exact
                                ? ProposalStatus::kExact
                                : ProposalStatus::kInexact;
        } else {
          const double eps_m = solver_.config().objective_improve_epsilon_m;
          const double scale =
              proposal.err_scale_m > 1e-6 ? proposal.err_scale_m : 1.0;
          bool saw = false;
          bool exact = true;
          auto admit = [&](double err) {
            if (!(err >= 0.0)) return;
            saw = true;
            if (err * scale > eps_m) exact = false;
          };
          if (solve_result.yaw_stepped) {
            // A ratio-continuation yaw is exact when it reaches its step targets (as a position stage
            // is when it reaches d*), within the Facing arrival angle (the one contact epsilon in degrees), not when it
            // reproduces the source's offset at once.
            saw = true;
            exact = solve_result.yaw_step_residual < kFacingArrivalRad;
          } else {
            admit(driving_facing_err);
          }
          proposal.status = (saw && exact) ? ProposalStatus::kExact
                                           : ProposalStatus::kInexact;
        }
      }
    }
    // ARRIVED: the SAMPLED root already satisfies what the source holds. This, not the
    // target's exactness, is what stops the latency timer. With a Spacing contact engaged that is
    // the contact (every engaged one); with none, every positive-weight root relation.
    {
      const bool arrived = contact_seen ? contact_arrived : (other_seen && other_arrived);
      if (arrived && !no_referent && !not_converged)
        proposal.flags |= kAnchorFlagArrived;
    }
    // ---- attribution: why this proposal differs from the previous one ----
    ProposalAttribution& attr = proposal_attribution_[actor_id];
    AnchorCauseState& root_state = attr.anchors[kRootAnchorId];
    AnchorCauseFacts facts = SolveCauseFacts(solve_result, root_payloads, root_weights);
    // Contained = inside a transfer element (the fact; the mapping is the
    // empty-scope mapping of the leaf, which is this actor's scope).
    // The source's containment observation names the source's root; any anchor's containment relation counts too.
    bool contained_now = false, observed_contained = false;
    shared_element_id_t contain_scope = kInvalidSharedElementId;
    if (const auto ac = actor_containment_.find(actor_id);
        ac != actor_containment_.end() && ac->second.contained) {
      contained_now = true;
      observed_contained = true;
      contain_scope = ac->second.scope_shared_id;
    } else if (const RelationPayload* cr = ContainmentRelationLocked(root_payloads, root_weights, nullptr)) {
      contained_now = true;
      contain_scope = cr->referent_shared_id;
    }
    // The transfer flag (ABI 4.3): this
    // update's root proposal IS the transfer mapping. The source root is inside a transfer
    // leaf (the source-side point test, the observation above), this space has that leaf's
    // counterpart declared a transfer leaf, AND the two shapes are congruent under a rigid
    // transform at this update (the source's footprint rides the leaf payload; the existing
    // geometry epsilon, no other). Recomputed on every solve against the CURRENT element
    // set and stored nowhere, so a counterpart that appears late or a geometry edit is
    // picked up on the next update, across processes, and a missing, undeclared or
    // non-congruent counterpart leaves it clear. Nothing is proposed when no referent maps,
    // so the flag is clear then too.
    // (The flag is the source's own transfer observation: the source names a transfer leaf it is inside, which a
    // containment relation alone does not say.)
    if (observed_contained && !no_referent) {
      const Element* counterpart =
          GetReferentResolver().Resolve(contain_scope, element_tree_);
      if (counterpart && counterpart->HasContainmentTransfer()) {
        // The leaf's SHAPE is its authored polygon in its own element frame, as the source's
        // payload carries it: placement (a rotation that an app hands over as single-precision
        // floats, say) is not part of the shape and must not enter the comparison.
        // The chart is the object's own: both sides' DECLARED shapes, whatever extends their scopes.
        const SpatialDefinition* counterpart_def = counterpart->GetDeclaredSpatialDefinition();
        const std::shared_ptr<Polygon2D> mine =
            counterpart_def ? counterpart_def->GetBoundary() : nullptr;
        for (size_t k = 0; mine && k < root_payloads.size(); ++k) {
          const RelationPayload& rp = root_payloads[k];
          if (rp.referent_shared_id != contain_scope || !rp.support.boundary)
            continue;
          if (FootprintsCongruent(*rp.support.boundary, *mine))
            proposal.flags |= kAnchorFlagTransfer;
          break;
        }
      }
    }
    ApplyAnchorChartLocked(update, root_anchor_name, /*is_root=*/true, &proposal);
    facts.sampled = true;
    facts.sample_epoch = sample.epoch;
    facts.sample_sequence = sample.sequence;
    facts.teleported = (sample.flags & kEmbodiedTeleported) != 0;
    facts.stuck = proposal.status == ProposalStatus::kNone || proposal.status == ProposalStatus::kNoSample;
    facts.engaged = (proposal.flags & kAnchorFlagEngaged) != 0;
    facts.contained = contained_now;
    facts.contain_scope = contain_scope;
    facts.exception = update.has_exception;
    // Why this proposal differs from the previous one: the rule of every anchor (AttributeAnchorCauses).
    const uint32_t cause = AttributeAnchorCauses(&root_state, facts, geometry_revision_, !receiver_reevaluation);
    proposal.cause = cause;
  } else {
    // An invalid policy vector: no targets at all (rich snapshots must not
    // attach current contributors to an old target).
    RealizationResult failed;
    failed.status = solve_result.status;
    realization_results_[actor_id] = std::move(failed);
  }
  // The other relation-bearing anchors (pelvis, head, ...): one proposal each,
  // from the same solve and the same update. Then the latency timers of the
  // interactions the source holds, and the exception it raised.
  // Every anchor is answered whatever the root's proposal is (one path per anchor).
  BuildAnchorProposalsLocked(actor_id, update, payloads, weights, realization_results_[actor_id], navigable_def,
                             root_anchor_name, receiver_reevaluation, warm_start.absolute, proposal.cause,
                             &proposal);
  std::vector<ActorEvent> events;
  if (!receiver_reevaluation) {
    // Stage the entire revision locally. PublishProposalLocked stores its
    // snapshot, then atomically exposes every lifecycle marker and immutable
    // target together, before invoking in-process observers.
    const uint64_t revision = proposal_revision_ + 1;
    // The receiver's first placed set is its join, whichever source update it answers: the source's first update
    // carries SB_EXCEPTION_JOIN, but a receiver whose body had no sample then, or whose space joined later, is placed
    // by a later update, and the event is how its app realizes a placement (a teleport to the set).
    const bool placed_join = !update.has_exception && warm_start.absolute &&
                             (proposal.status == ProposalStatus::kExact || proposal.status == ProposalStatus::kInexact);
    if (update.has_exception || placed_join) {
      ActorEvent event;
      event.kind = ActorEvent::Kind::kSourceException;
      event.revision = revision;
      event.source_sequence = update.sequence;
      event.source_shared_id = update.actor_shared_id;
      event.exception_id = update.has_exception ? update.exception_id : kExceptionJoin;
      if (update.has_exception) event.payload = update.exception_payload;
      events.push_back(std::move(event));
    }
    EvaluateInteractionLatencyLocked(actor_id, update, payloads, weights,
                                     root_anchor_name, proposal, revision,
                                     &events);
  }
  PublishProposalLocked(actor_id, std::move(proposal), std::move(events));

  if (auto it = realization_results_.find(actor_id);
      it != realization_results_.end()) {
    uint64_t counterpart_revision = 0;
    const bool counterpart_known = CaptureCounterpartRevisionLocked(
        linker_revision_before,
        hub_ ? hub_->GetLinker().GetGraphRevision() : 0,
        &counterpart_revision);
    CaptureRealizationSnapshot(actor_id, incoming_payloads, resolved_index,
                               payloads, weights, it->second,
                               scene_inputs_before, &update,
                               receiver_ingress_monotonic_ns,
                               counterpart_known, counterpart_revision,
                               receiver_input);
  }

  // SPEC S3 ev=target: one line per realized target (root + any per-anchor
  // targets), every OnRemoteUpdate call while active — not jump-gated, so a
  // clone parked at the centroid is visible too (the old JUMP line was silent
  // when nothing moved).
  // ev=goal: each root relation's witness point before blending.
  const auto solver_obs = solver_observations_.find(actor_id);
  const bool has_solver_obs =
      solver_obs != solver_observations_.end() && solver_obs->second.valid;

  // The realized driving relation per channel, decided after the root
  // solve dropped unresolvable/zero-weight goals and after the anchor
  // gate and reach clamp decided which anchor targets exist.
  if (policy_obs) {
    const auto rit = realization_results_.find(actor_id);
    AnnotateRealizedDriving(
        payloads, weights,
        has_solver_obs ? &solver_obs->second : nullptr,
        rit != realization_results_.end() ? &rit->second : nullptr,
        &policy_obs->channels);
  }
  // ev=channel: one line per (family, anchor) channel per update: the
  // policy argmax and temperature, and the relation that actually drove the
  // realized target.
  if (trace_active && policy_obs) {
    auto& sink = TraceSink::ForSpace(space_id_);
    for (const auto& channel : policy_obs->channels) {
      nlohmann::ordered_json ev;
      ev["ev"] = "channel";
      ev["t"] = space_clock_;
      ev["space"] = space_id_;
      ev["actor"] = update.actor_shared_id;
      ev["anchor"] =
          channel.anchor_name.empty() ? "root" : channel.anchor_name;
      ev["kernel"] = static_cast<int>(channel.kernel_type);
      ev["policy"] = RealizationPolicyName(policy_config.type);
      ev["n"] = channel.payload_indices.size();
      ev["t_min"] = std::isfinite(channel.t_min)
                        ? nlohmann::ordered_json(channel.t_min)
                        : nlohmann::ordered_json(nullptr);
      ev["eta"] = std::isfinite(channel.eta)
                      ? nlohmann::ordered_json(channel.eta)
                      : nlohmann::ordered_json(nullptr);
      if (channel.has_driving) {
        const RelationPayload& d = payloads[channel.driving_index];
        ev["driving_referent"] = d.referent_shared_id;
        ev["driving_weight"] = channel.driving_weight;
        ev["driving_R"] = d.grounded_relation.proximity;
        ev["driving_est_time"] = d.estimated_time;
      } else {
        ev["driving_referent"] = nullptr;
        ev["driving_weight"] = 0.0;
        ev["driving_R"] = nullptr;
        ev["driving_est_time"] = nullptr;
      }
      ev["tied"] = channel.driving_tied;
      if (channel.has_realized) {
        ev["realized_referent"] =
            payloads[channel.realized_index].referent_shared_id;
        ev["realized_weight"] = channel.realized_weight;
      } else {
        ev["realized_referent"] = nullptr;
        ev["realized_weight"] = 0.0;
      }
      ev["realized_is_policy_argmax"] =
          channel.has_realized && channel.has_driving &&
          channel.realized_index == channel.driving_index;
      sink.Emit(ev);
    }
  }
  if (trace_active && has_solver_obs) {
    auto& sink = TraceSink::ForSpace(space_id_);
    for (const auto& goal : solver_obs->second.solve.goals) {
      if (goal.payload_index >= root_payloads.size()) continue;
      const RelationPayload& p = root_payloads[goal.payload_index];
      nlohmann::ordered_json ev;
      ev["ev"] = "goal";
      ev["t"] = space_clock_;
      ev["space"] = space_id_;
      ev["actor"] = update.actor_shared_id;
      ev["anchor"] =
          p.subject_anchor_name.empty() ? "root" : p.subject_anchor_name;
      ev["kernel"] = static_cast<int>(p.kernel_type);
      ev["referent"] = p.referent_shared_id;
      ev["witness"] = {goal.witness.x(), goal.witness.y()};
      ev["weight"] = goal.weight;
      ev["pos_goal"] = goal.position_goal;
      ev["yaw_goal"] = goal.yaw_goal;
      sink.Emit(ev);
    }
  }

  if (trace_active) {
    auto rit = realization_results_.find(actor_id);
    if (rit != realization_results_.end()) {
      const char* status_str =
          (rit->second.status == SolverStatus::kConverged)   ? "converged"
          : (rit->second.status == SolverStatus::kFallback)  ? "fallback"
          : (rit->second.status == SolverStatus::kBudgetHit) ? "budget_hit"
                                                             : "failed";
      auto& sink = TraceSink::ForSpace(space_id_);
      for (const auto& tg : rit->second.targets) {
        nlohmann::ordered_json ev;
        ev["ev"] = "target";
        ev["t"] = space_clock_;
        ev["space"] = space_id_;
        ev["actor"] = update.actor_shared_id;
        // SPEC S3: same empty->"root" normalization as the ev=payload
        // block above -- defensive, since both current call sites that
        // populate RealizationTarget::anchor_name already hardcode "root"
        // for the root target, but nothing should ever emit "".
        ev["anchor"] = tg.anchor_name.empty() ? "root" : tg.anchor_name;
        if (tg.has_z) {
          ev["pos"] = {tg.position.x(), tg.position.y(), tg.z};
        } else {
          ev["pos"] = {tg.position.x(), tg.position.y(), 0.0};
        }
        ev["has_z"] = tg.has_z;
        ev["yaw"] = tg.yaw;
        ev["kernel"] = static_cast<int>(tg.kernel_type);
        ev["referent"] = tg.referent_id;
        ev["weight"] = tg.weight;
        ev["status"] = status_str;
        // Trace format 2: feasibility snap (root target only, from the
        // root solve); format 5: reach diagnostic (anchor targets).
        const bool is_root_target = &tg == &rit->second.targets.front() &&
                                    tg.kernel_type ==
                                        RelationKernelType::kAvatarSpacing &&
                                    !tg.has_z;
        const bool snapped = is_root_target && has_solver_obs &&
                             solver_obs->second.solve.feasibility_snapped;
        ev["snap_m"] = (is_root_target && has_solver_obs)
                           ? solver_obs->second.solve.feasibility_snap_m
                           : 0.0;
        ev["snapped"] = snapped;
        ev["beyond_reach"] = tg.beyond_reach;
        ev["reach_excess_m"] = tg.reach_excess_m;
        // Trace format 7: the paper
        // objective at the goal-point initializer and after the bounded
        // local refinement, root target only (same root-only scope as
        // snap_m / snapped above).
        ev["objective_before"] = (is_root_target && has_solver_obs)
                                     ? solver_obs->second.solve.objective_before
                                     : 0.0;
        ev["objective_after"] = (is_root_target && has_solver_obs)
                                    ? solver_obs->second.solve.objective_after
                                    : 0.0;
        ev["objective_iterations"] =
            (is_root_target && has_solver_obs)
                ? solver_obs->second.solve.objective_iterations
                : 0;
        ev["objective_fallback"] =
            is_root_target && has_solver_obs &&
            solver_obs->second.solve.objective_fallback;
        sink.Emit(ev);
      }
    }
  }

  // SPEC S3 ev=tick: compute cost of this update (referent resolution through
  // the solver / target trace above), in microseconds.
  if (trace_active) {
    auto elapsed = std::chrono::steady_clock::now() - tick_start;
    int64_t cost_us =
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    nlohmann::ordered_json ev;
    ev["ev"] = "tick";
    ev["t"] = space_clock_;
    ev["space"] = space_id_;
    ev["cost_us"] = cost_us;
    TraceSink::ForSpace(space_id_).Emit(ev);
  }
}

const std::vector<RemoteRelationUpdate>& Space::GetReceivedUpdates() const {
  return received_updates_;
}

Space::ReceivedUpdatesSnapshot Space::GetReceivedUpdatesSnapshot() const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return {received_update_count_, received_updates_};
}

uint64_t Space::GetReceivedUpdateCount() const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return received_update_count_;
}

int64_t Space::GetLastEmittedSequence(spatialblend::entity_id_t actor_id) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto it = next_propagate_sequence_.find(actor_id);
  return it != next_propagate_sequence_.end() ? it->second : 0;
}

int64_t Space::GetLastAppliedRemoteSequence(
    space_id_t source_space_id,
    shared_element_id_t source_actor_shared_id) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto it = last_remote_sequence_.find(
      std::make_pair(source_space_id, source_actor_shared_id));
  return it != last_remote_sequence_.end() ? it->second : 0;
}

bool Space::GetActorWireIdentity(
    spatialblend::entity_id_t actor_id, space_id_t* source_space_id,
    shared_element_id_t* source_actor_shared_id) const {
  if (!source_space_id || !source_actor_shared_id) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto actor_it = actors_.find(actor_id);
  if (actor_it == actors_.end()) return false;
  const Element* elem =
      element_tree_.GetElement(actor_it->second->GetElementId());
  if (!elem) return false;
  *source_space_id = space_id_;
  // This deliberately does not substitute a local element id: it mirrors the
  // actor_shared_id field written onto RemoteRelationUpdate exactly.
  *source_actor_shared_id = elem->GetSharedId();
  return true;
}

void Space::RegisterOnRelationEvent(
    std::function<void(const RelationEvent&)> cb) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  on_relation_event_ = std::move(cb);
}

// ---------------------------------------------------------------------------
// Containment transfer (space/containment_transfer.h)
// ---------------------------------------------------------------------------


bool Space::SetElementContainmentTransfer(element_id_t id, bool enabled) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto* element = element_tree_.GetElement(id);
  if (!element || observed_facing_points_.count(id)) return false;
  if (element->HasContainmentTransfer() != enabled) {
    element->SetContainmentTransfer(enabled);
    if (enabled) containment_transfer_used_ = true;
    AdvanceSceneInputsLocked(true, false);
  }
  return true;
}

bool Space::IsElementContainmentTransfer(element_id_t id) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  const auto* element = element_tree_.GetElement(id);
  return element && element->HasContainmentTransfer();
}

bool Space::CopySourceTransferLeaf(spatialblend::entity_id_t actor_id,
                                   shared_element_id_t* leaf_shared_id) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!FindActorLocked(actor_id)) return false;
  *leaf_shared_id = kInvalidSharedElementId;
  const auto it = actor_containment_.find(actor_id);
  // Only what this actor's own latest update computed: a source actor's entry names this
  // space as the source; a remote proxy's entry (a received observation) does not count.
  if (it != actor_containment_.end() && it->second.contained &&
      it->second.scope_source_space_id == space_id_)
    *leaf_shared_id = it->second.scope_shared_id;
  return true;
}

bool Space::SetActorContainmentLocked(spatialblend::entity_id_t actor_id,
                                      const ActorContainment& next) {
  ActorContainment& slot = actor_containment_[actor_id];
  const bool was = slot.contained;
  const bool same_scope =
      slot.scope_source_space_id == next.scope_source_space_id &&
      slot.scope_source_element_id == next.scope_source_element_id;
  const bool transitioned = was != next.contained ||
                            (was && next.contained && !same_scope);
  slot = next;
  return transitioned;
}

void Space::ClearActorContainmentLocked(spatialblend::entity_id_t actor_id) {
  actor_containment_.erase(actor_id);
}

std::optional<ContainmentObservation> Space::UpdateSourceContainmentLocked(
    spatialblend::entity_id_t actor_id, const Eigen::Vector2d& position,
    bool* transitioned) {
  *transitioned = false;
  if (!containment_transfer_used_) {
    actor_containment_.erase(actor_id);
    return std::nullopt;
  }
  const Actor* actor = FindActorLocked(actor_id);
  if (!actor || actor->GetElementId() == kInvalidElementId)
    return std::nullopt;
  const element_id_t scope_root =
      ScopeResolver::NavigableRoot(element_tree_, actor->GetElementId());
  ActorContainment next;
  std::optional<ContainmentObservation> obs;
  // The source-side transfer fact is the scope layer's own: the deepest flagged element on
  // the containment chain of the root (DescendScope), the same descent every scope uses.
  const element_id_t frame =
      DescendScope(Eigen::Vector3d(position.x(), position.y(), 0.0), scope_root, element_tree_,
                   /*choose_overlap=*/true, /*skip_transparent=*/true)
          .transfer_leaf;
  if (const Element* frame_el =
          frame == kInvalidElementId ? nullptr : element_tree_.GetElement(frame)) {
    ContainmentObservation o;
    o.scope_shared_id = frame_el->GetSharedId();
    o.scope_source_space_id = space_id_;
    o.scope_source_element_id = frame;
    next.contained = true;
    next.scope_shared_id = o.scope_shared_id;
    next.scope_source_space_id = space_id_;
    next.scope_source_element_id = frame;
    obs = o;
  }
  *transitioned = SetActorContainmentLocked(actor_id, next);
  return obs;
}

void Space::SetReferentResolver(std::shared_ptr<ReferentResolver> resolver) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (referent_resolver_ == resolver) return;
  referent_resolver_ = std::move(resolver);
  has_custom_referent_resolver_ = referent_resolver_ != nullptr;
  // Custom resolution receives a const tree with mutable nested geometry.
  // Removing the resolver later cannot revoke references it might retain.
  if (has_custom_referent_resolver_) scene_inputs_escaped_.store(true);
  ++counterpart_revision_;
}

const ReferentResolver& Space::GetReferentResolver() const {
  if (!referent_resolver_) {
    referent_resolver_ = std::make_shared<ReferentResolver>();
  }
  return *referent_resolver_;
}

bool Space::IsActorReferentLocked(
    element_id_t element_id,
    const std::unordered_set<element_id_t>& actor_element_ids) const {
  if (observed_facing_points_.count(element_id)) return true;
  if (actor_element_ids.count(element_id)) return true;
  for (element_id_t a : element_tree_.GetAncestorIds(element_id)) {
    if (actor_element_ids.count(a)) return true;
  }
  return false;
}

absl::Status Space::SetRelationPhaseConfig(RelationKernelType family,
                                           const RelationPhaseConfig& config) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return phase_tracker_.SetConfig(family, config);
}

RelationPhaseConfig Space::GetRelationPhaseConfig(
    RelationKernelType family) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return phase_tracker_.GetConfig(family);
}

std::string Space::PhaseOwner(spatialblend::entity_id_t actor_id) const {
  return "space:" + std::to_string(space_id_) + ":" + std::to_string(actor_id);
}

void Space::ReconcileSourceRelationPhasesLocked(
    spatialblend::entity_id_t actor_id,
    const std::vector<SourceRelationObservation>& current, double timestamp) {
  auto& previous = source_relations_[actor_id];
  const auto owner = PhaseOwner(actor_id);
  // Legacy public shared ids need not be unique. Several local rows may own
  // the same phase key; losing one must not reset a still-continuous row.
  const auto phase_key = [](const RelationEvent& event) {
    return std::make_tuple(event.actor_shared_id, event.anchor_name,
                           event.referent_shared_id, event.kernel_type);
  };
  using SourceIdentity = decltype(phase_key(RelationEvent{}));
  std::set<std::pair<element_id_t, SourceIdentity>> current_rows;
  for (const auto& next : current)
    current_rows.emplace(next.referent, phase_key(next.identity));
  std::set<SourceIdentity> continuing;
  for (const auto& old : previous)
    if (current_rows.count({old.referent, phase_key(old.identity)}))
      continuing.insert(phase_key(old.identity));
  std::vector<RelationEvent> withdrawals;
  for (const auto& old : previous) {
    if (continuing.count(phase_key(old.identity))) continue;
    RelationEvent withdrawal = old.identity;
    withdrawal.from = phase_tracker_.GetPhaseForOwner(old.identity, owner);
    withdrawal.to = RelationPhase::kDormant;
    withdrawal.timestamp = timestamp;
    phase_tracker_.ClearRelationForOwner(old.identity, owner);
    if (withdrawal.from != RelationPhase::kDormant)
      withdrawals.push_back(std::move(withdrawal));
  }
  previous = current;
  // A withdrawal terminates the exact relation immediately. Feeding zero
  // proximity through the ordinary hysteresis would leave Contact/Driving
  // alive for extra invented samples. No source clock/sequence is advanced.
  DispatchRelationEvents(withdrawals, 0.0);
}

void Space::EmitRelationTransitions(const RelationEvent& identity,
                                    double proximity, double weight,
                                    double timestamp,
                                    const std::string& owner) {
  // Called with space_mutex_ already held. The registered handler must not
  // re-enter this Space (the mutex is non-recursive).
  auto events =
      owner.empty()
          ? phase_tracker_.Update(identity, proximity, weight, timestamp)
          : phase_tracker_.UpdateForOwner(identity, owner, proximity, weight,
                                          timestamp);
  DispatchRelationEvents(events, proximity);
}

void Space::DispatchRelationEvents(const std::vector<RelationEvent>& events,
                                   double proximity) {
  if (on_relation_event_) {
    for (const auto& ev : events) on_relation_event_(ev);
  }
  // SPEC S3 ev=relation: one line per phase transition (not per observation —
  // phase_tracker_.Update only returns entries when `to` != `from`).
  if (!events.empty() && TraceSink::ForSpace(space_id_).Active()) {
    auto& sink = TraceSink::ForSpace(space_id_);
    for (const auto& re : events) {
      nlohmann::ordered_json ev;
      ev["ev"] = "relation";
      ev["t"] = re.timestamp;
      ev["space"] = space_id_;
      ev["actor"] = re.actor_shared_id;
      // SPEC S3: same empty->"root" normalization as the ev=payload /
      // ev=target anchor fields above -- the anchor name embedded in this
      // "relation" id string must never be empty either (e.g. "spacing:"
      // instead of "spacing:root").
      ev["relation"] = std::string(KernelTraceName(re.kernel_type)) + ":" +
                       (re.anchor_name.empty() ? "root" : re.anchor_name);
      // Trace format 2: which referent this relation line is about.
      ev["referent"] = re.referent_shared_id;
      ev["phase"] = SpecPhaseTraceName(re.to);
      ev["tracker_phase"] = TrackerPhaseName(re.to);
      ev["R"] = proximity;
      sink.Emit(ev);
    }
  }
}

void Space::SetBodyRadius(double radius_m) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!std::isfinite(radius_m) || radius_m < 0.0) return;
  body_radius_ = radius_m;
}

double Space::BodyRadiusOfLocked(spatialblend::entity_id_t actor_id) const {
  const auto it = actor_body_radius_.find(actor_id);
  return it != actor_body_radius_.end() ? it->second : body_radius_;
}

double Space::DriverClearanceLocked(spatialblend::entity_id_t actor_id) const {
  const double r = BodyRadiusOfLocked(actor_id);
  return std::isfinite(r) && r > 0.0 ? r : 0.0;
}

double Space::GetBodyRadius(spatialblend::entity_id_t actor_id) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return BodyRadiusOfLocked(actor_id);
}

double Space::GetBodyRadius() const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return body_radius_;
}





const NavigableSpatialDefinition* Space::FindActorNavigableLocked(
    spatialblend::entity_id_t actor_id) const {
  const Actor* actor = FindActorLocked(actor_id);
  if (!actor) return nullptr;
  // Same lookup as the receiver solve: nearest ancestor (inclusive) of the
  // actor's element that carries a NavigableSpatialDefinition.
  for (element_id_t e = actor->GetElementId(); e != kInvalidElementId;) {
    const Element* el = element_tree_.GetElement(e);
    if (!el) break;
    if (const auto* nav = dynamic_cast<const NavigableSpatialDefinition*>(
            el->GetSpatialDefinition())) {
      return nav;
    }
    e = el->GetParentId();
  }
  return nullptr;
}

std::vector<Eigen::Vector2d> Space::QueryCourse(
    spatialblend::entity_id_t actor_id, const Eigen::Vector2d& from,
    const Eigen::Vector2d& to) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!from.allFinite() || !to.allFinite()) return {};
  SyncNavigableObstaclesLocked();
  const NavigableSpatialDefinition* nav = FindActorNavigableLocked(actor_id);
  if (!nav) return {from, to};
  // A DRIVER SERVICE, never part of the core's proposal or solve (body clearance belongs to the
  // driver; the core's F has clearance 0). The course of a body of this actor's registered radius, called by the
  // driver with its own radius. A goal inside the body's clearance ring ends the course at the nearest free point:
  // the body stops there, it does not walk into the ring.
  const auto reach = nav->QueryReachability(from, to, DriverClearanceLocked(actor_id));
  if (!reach.reachable) return {};
  std::vector<Eigen::Vector2d> course = reach.path.GetPoints();
  if (reach.to_snap_m > 1e-12 && course.size() > 2) course.pop_back();
  return course;
}

bool Space::IsFreeUnder(spatialblend::entity_id_t actor_id, const Eigen::Vector2d& point,
                        const std::vector<int64_t>& exempt,
                        const std::vector<int64_t>& ring_exempt) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!point.allFinite()) return false;
  SyncNavigableObstaclesLocked();
  const NavigableSpatialDefinition* nav = FindActorNavigableLocked(actor_id);
  return !nav || nav->IsFree(point, exempt, ring_exempt);
}

std::vector<ObstacleFreeSpace::RoutePiece> Space::QueryRoutesUnder(
    spatialblend::entity_id_t actor_id, const Eigen::Vector2d& from, const Eigen::Vector2d& to,
    const std::vector<int64_t>& exempt, const std::vector<int64_t>& ring_exempt, double slack) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!from.allFinite() || !to.allFinite()) return {};
  SyncNavigableObstaclesLocked();
  const NavigableSpatialDefinition* nav = FindActorNavigableLocked(actor_id);
  if (!nav) return {};
  return nav->GeodesicRoutesUnder(from, to, exempt, ring_exempt, slack);
}

std::vector<ObstacleFreeSpace::RoutePiece> Space::QueryRoutes(spatialblend::entity_id_t actor_id, const Eigen::Vector2d& from,
                                                              const Eigen::Vector2d& to, double slack) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!from.allFinite() || !to.allFinite()) return {};
  SyncNavigableObstaclesLocked();
  const NavigableSpatialDefinition* nav = FindActorNavigableLocked(actor_id);
  if (!nav) return {};
  return nav->GeodesicRoutes(from, to, slack);
}

std::vector<ObstacleFreeSpace::RoutePiece> Space::QueryContactRoutes(
    spatialblend::entity_id_t actor_id, const Eigen::Vector2d& from,
    const std::vector<std::vector<Eigen::Vector2d>>& rings, double slack) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!from.allFinite()) return {};
  SyncNavigableObstaclesLocked();
  const NavigableSpatialDefinition* nav = FindActorNavigableLocked(actor_id);
  if (!nav) return {};
  std::vector<const std::vector<Eigen::Vector2d>*> ptrs;
  for (const auto& r : rings) ptrs.push_back(&r);
  return nav->ContactRoutes(from, ptrs, slack);
}

ObstacleFreeSpace::FeasibleCone Space::QueryConeUnder(
    spatialblend::entity_id_t actor_id, const Eigen::Vector2d& point, const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt, double tol) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  ObstacleFreeSpace::FeasibleCone full;
  full.arcs = {{0.0, 6.283185307179586}};
  if (!point.allFinite()) return full;
  SyncNavigableObstaclesLocked();
  const NavigableSpatialDefinition* nav = FindActorNavigableLocked(actor_id);
  if (!nav) return full;
  return nav->TangentConeAt(point, exempt, ring_exempt, tol);
}

std::vector<Eigen::Vector2d> Space::QueryCourseUnder(
    spatialblend::entity_id_t actor_id, const Eigen::Vector2d& from,
    const Eigen::Vector2d& to, const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!from.allFinite() || !to.allFinite()) return {};
  SyncNavigableObstaclesLocked();
  const NavigableSpatialDefinition* nav = FindActorNavigableLocked(actor_id);
  if (!nav) return {from, to};
  // A DRIVER SERVICE (see QueryCourse): the proposal and the solve never call it, and the core's F has clearance 0.
  const auto reach = nav->QueryReachability(from, to, exempt, ring_exempt, DriverClearanceLocked(actor_id));
  if (!reach.reachable) return {};
  std::vector<Eigen::Vector2d> course = reach.path.GetPoints();
  if (reach.to_snap_m > 1e-12 && course.size() > 2) course.pop_back();
  return course;
}

std::optional<Eigen::Vector2d> Space::QueryNearestClearPoint(spatialblend::entity_id_t actor_id,
                                                             const Eigen::Vector2d& point,
                                                             const std::vector<int64_t>& exempt,
                                                             const std::vector<int64_t>& ring_exempt) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!point.allFinite()) return std::nullopt;
  SyncNavigableObstaclesLocked();
  const NavigableSpatialDefinition* nav = FindActorNavigableLocked(actor_id);
  if (!nav) return point;
  const auto ctx = nav->GetGeometryContext();
  if (!ctx) return point;
  return ctx->FreeSpaceUnder(exempt, ring_exempt, DriverClearanceLocked(actor_id)).NearestFree(point);
}

void Space::SyncNavigableObstaclesLocked() {
  // Cheap path: Space-mediated scene mutations advance the tree/support
  // revisions; when both are unchanged (and no raw tree pointer escaped),
  // nothing can have moved.
  const SceneInputStamp stamp = CaptureSceneInputStampLocked();
  if (obstacle_sync_valid_ && stamp.known && stamp.tree == obstacle_sync_tree_ &&
      stamp.support == obstacle_sync_support_)
    return;
  Element* root = element_tree_.GetRoot();
  if (!root) return;

  // Fingerprint of everything the obstacle geometry depends on (element ids,
  // definition identity, world placement). An unchanged fingerprint skips
  // the footprint rebuild -- e.g. the per-tick actor-leaf transform update
  // that advances the tree revision without touching any obstacle.
  std::vector<double> fingerprint;
  std::vector<element_id_t> obstacle_ids;
  std::vector<std::pair<element_id_t, NavigableSpatialDefinition*>> navs;
  for (element_id_t id : element_tree_.GetSubtreeIds(root->GetId())) {
    Element* el = element_tree_.GetElement(id);
    if (!el) continue;
    SpatialDefinition* def = el->GetSpatialDefinition();
    if (!def) continue;
    if (el->HasGeometryRole(GeometryRole::kObstacle)) {
      obstacle_ids.push_back(id);
      fingerprint.push_back(1.0);
      fingerprint.push_back(static_cast<double>(id));
      fingerprint.push_back(
          static_cast<double>(reinterpret_cast<uintptr_t>(def)));
      const Eigen::Matrix4d m = def->RequiresElementWorldTransform()
                                    ? element_tree_.GetWorldTransform(id)
                                    : def->GetTransform();
      for (int k = 0; k < 16; ++k) fingerprint.push_back(m(k));
    } else if (el->HasGeometryRole(GeometryRole::kNavigable)) {
      if (auto* nav = dynamic_cast<NavigableSpatialDefinition*>(def)) {
        navs.push_back({id, nav});
        fingerprint.push_back(2.0);
        fingerprint.push_back(static_cast<double>(id));
        fingerprint.push_back(
            static_cast<double>(reinterpret_cast<uintptr_t>(nav)));
        // The obstacles a navigable owns exclude its ancestors, so its ancestry is part of the sync state (a
        // reparent changes it without moving any obstacle).
        for (element_id_t anc : element_tree_.GetAncestorIds(id))
          fingerprint.push_back(static_cast<double>(anc));
        fingerprint.push_back(-1.0);
      }
    }
  }
  obstacle_sync_valid_ = true;
  obstacle_sync_tree_ = stamp.tree;
  obstacle_sync_support_ = stamp.support;
  if (fingerprint == obstacle_fingerprint_) return;
  obstacle_fingerprint_ = std::move(fingerprint);

  // The same footprints and the same per-navigable rule the scope-extension worker applies
  // (scope_extension.h), so an extension it installs holds these obstacles already.
  const std::vector<ObstacleFootprint> obstacles = CollectObstacleFootprints(element_tree_);
  bool changed = false;
  for (auto& [nav_id, nav] : navs) {
    std::vector<FreeSpaceObstacle> own =
        ObstaclesWithin(obstacles, nav->GetBounds(), element_tree_.GetAncestorIds(nav_id));
    if (nav->SetObstacles(std::move(own), /*clearance=*/0.0)) {
      changed = true;
      ++obstacle_sync_rebuilds_;
      // Pay the free-space / graph build on this (changing) update, not on a
      // later steady-state query. Moving obstacle-role elements therefore
      // cost one rebuild per frame they move (~30 ms at 60 boxes): fine for
      // static scanned furniture, not for continuously animated obstacles.
      nav->PrewarmFreeSpaces();
    }
  }
  (void)changed;
}

const RealizationResult* Space::GetRealizationTargets(
    spatialblend::entity_id_t actor_id) const {
  if (!AreObservedFacingDependenciesLiveLocked(actor_id, receiver_input_clock_()))
    return nullptr;
  // This remains a legacy single-threaded borrowed view. An external-input
  // receiver must not expose stale actionable goals even during idle time.
  if (receiver_inputs_.count(actor_id)) {
    const auto current = CurrentReceiverInputLocked(actor_id);
    const auto solved = receiver_solve_inputs_.find(actor_id);
    if (!current.known || solved == receiver_solve_inputs_.end() ||
        !solved->second.known || solved->second.epoch != current.epoch ||
        solved->second.revision != current.revision)
      return nullptr;
  }
  auto it = realization_results_.find(actor_id);
  return it != realization_results_.end() ? &it->second : nullptr;
}

void Space::SetRichRealizationEnabled(bool enabled) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  rich_realization_enabled_ = enabled;
  if (!enabled) {
    realization_snapshots_.clear();
    retained_receiver_evaluations_.clear();
  }
}

void Space::SetActorRichRealizationEnabled(spatialblend::entity_id_t actor_id,
                                           bool enabled) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (enabled) {
    if (rich_realization_actors_.insert(actor_id).second)
      retained_receiver_evaluations_.erase(actor_id);
  } else {
    rich_realization_actors_.erase(actor_id);
    realization_snapshots_.erase(actor_id);
    retained_receiver_evaluations_.erase(actor_id);
  }
}

bool Space::CopyActorRealizationSnapshot(spatialblend::entity_id_t actor_id,
                                         ActorRealizationSnapshot* out) const {
  if (!out) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  const auto it = realization_snapshots_.find(actor_id);
  if (it == realization_snapshots_.end()) return false;
  *out = it->second;
  return true;
}

bool Space::GetActorRealizationValidity(spatialblend::entity_id_t actor_id,
                                        uint64_t revision,
                                        uint32_t requested_domains,
                                        ActorRealizationValidity* out) {
  if (!out) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  PruneObservedFacingPointsLocked();
  const auto snapshot_it = realization_snapshots_.find(actor_id);
  if (snapshot_it == realization_snapshots_.end() ||
      snapshot_it->second.revision != revision) {
    return false;
  }
  ResolveRealizationPolicyLocked();
  const ActorRealizationSnapshot& snapshot = snapshot_it->second;
  ActorRealizationValidity value;
  value.revision = revision;
  value.captured_known_mask = snapshot.known_revision_mask;
  const auto scene = CaptureSceneInputStampLocked();
  if (scene.known) {
    value.tree_revision = scene.tree;
    value.support_revision = scene.support;
    value.current_known_mask |= kRealizationTreeRevision |
                                kRealizationSupportRevision;
    if ((snapshot.known_revision_mask & kRealizationTreeRevision) != 0 &&
        snapshot.tree_revision != scene.tree)
      value.changed_mask |= kRealizationTreeRevision;
    if ((snapshot.known_revision_mask & kRealizationSupportRevision) != 0 &&
        snapshot.support_revision != scene.support)
      value.changed_mask |= kRealizationSupportRevision;
  } else {
    value.unchecked_mask |= kRealizationTreeRevision |
                            kRealizationSupportRevision;
  }
  value.policy_revision = policy_revision_;
  value.current_known_mask |= kRealizationPolicyRevision;
  if ((snapshot.known_revision_mask & kRealizationPolicyRevision) != 0 &&
      snapshot.policy_revision != policy_revision_) {
    value.changed_mask |= kRealizationPolicyRevision;
  }
  uint64_t current_counterpart_revision = 0;
  if (CurrentCounterpartRevisionLocked(&current_counterpart_revision)) {
    value.current_known_mask |= kRealizationCounterpartRevision;
    value.counterpart_revision = current_counterpart_revision;
    if ((snapshot.known_revision_mask & kRealizationCounterpartRevision) != 0 &&
        snapshot.counterpart_revision != current_counterpart_revision) {
      value.changed_mask |= kRealizationCounterpartRevision;
    }
  }
  *out = value;
  return true;
}

const ActorRealizationSnapshot* Space::GetActorRealizationSnapshot(
    spatialblend::entity_id_t actor_id) const {
  auto it = realization_snapshots_.find(actor_id);
  return it != realization_snapshots_.end() ? &it->second : nullptr;
}

void Space::ShareElement(element_id_t elem_id, const std::string& user_data) {
  SpaceHub* hub = nullptr;
  ElementEvent event;
  {
    std::lock_guard<std::mutex> lock(space_mutex_);
    if (!hub_ || observed_facing_points_.count(elem_id)) return;

    const Element* elem = element_tree_.GetElement(elem_id);
    if (!elem) return;
    // No shared_id requirement: the element is identified by its source handle
    // ({space_id_, elem_id}); the receiver links to it on resolve. shared_id,
    // if present, rides along only as a legacy fallback.

    event.type = ElementEvent::kAdded;
    event.shared_id = elem->GetSharedId();
    event.tag = elem->GetTag();
    event.source_space_id = space_id_;
    event.source_element_id = elem_id;
    event.user_data = user_data;
    locally_shared_elements_.insert(elem_id);

    // Find parent's shared_id
    element_id_t parent_id = kInvalidElementId;
    auto ancestors = element_tree_.GetAncestorIds(elem_id);
    if (!ancestors.empty()) {
      parent_id = ancestors.front();  // immediate parent
    }
    if (parent_id != kInvalidElementId) {
      const Element* parent = element_tree_.GetElement(parent_id);
      if (parent) {
        event.parent_shared_id = parent->GetSharedId();
      }
    }

    hub = hub_;
  }
  // Peer dispatch is synchronous for local hubs and may re-enter this source.
  hub->PropagateElementEvent(event, space_id_);
}

void Space::UnshareElement(element_id_t elem_id) {
  SpaceHub* hub = nullptr;
  ElementEvent event;
  {
    std::lock_guard<std::mutex> lock(space_mutex_);
    if (!hub_) return;

    const Element* elem = element_tree_.GetElement(elem_id);
    if (!elem) return;
    event.type = ElementEvent::kRemoved;
    event.shared_id = elem->GetSharedId();
    event.source_space_id = space_id_;
    event.source_element_id = elem_id;
    locally_shared_elements_.erase(elem_id);

    hub = hub_;
  }
  hub->PropagateElementEvent(event, space_id_);
}

bool Space::RemoveLocalElementSubtree(element_id_t elem_id) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  const Element* root = element_tree_.GetElement(elem_id);
  if (!root || root->GetParentId() == kInvalidElementId) return false;

  const auto subtree = element_tree_.GetSubtreeIds(elem_id);
  if (subtree.empty()) return false;
  auto is_local_only = [&](element_id_t id) {
    if (locally_shared_elements_.count(id) != 0) return false;
    if (pending_share_source_.count(id) != 0) return false;
    for (const auto& entry : remote_owned_replicas_) {
      if (entry.second == id) return false;
    }
    if (hub_ && (hub_->GetLinker().IsSource(element_handle_t{space_id_, id}) ||
                 hub_->GetLinker().IsTarget(element_handle_t{space_id_, id}))) {
      return false;
    }
    return true;
  };
  for (element_id_t id : subtree) {
    if (!is_local_only(id)) return false;
  }
  // A local child nested in received/shared furniture cannot be retired via
  // this local-only path either: its ancestor owns the cross-space lifetime.
  for (element_id_t ancestor : element_tree_.GetAncestorIds(elem_id)) {
    if (!is_local_only(ancestor)) return false;
  }

  if (!element_tree_.RemoveElement(elem_id).ok()) return false;
  AdvanceSceneInputsLocked(true, true);
  CleanupRemovedElements(subtree);
  return true;
}

void Space::OnElementEvent(const ElementEvent& event) {
  // Element id of a newly added element, surfaced to on_element_added_ after
  // the lock is released so the callback can re-enter the space safely.
  std::vector<std::pair<element_id_t, ElementEvent>> added;
  std::vector<element_id_t> removed_ids;
  struct LinkRemoval {
    element_id_t local;
    space_id_t peer;
    LinkRemovedReason reason;
  };
  std::vector<LinkRemoval> link_removed;
  {
    std::lock_guard<std::mutex> lock(space_mutex_);

    if (event.type == ElementEvent::kAdded) {
      AddReceivedElementLocked(event, &added);
    } else if (event.type == ElementEvent::kRemoved) {
      const element_handle_t source_handle{event.source_space_id,
                                           event.source_element_id};
      ClearRetainedEvaluationsForSourceLocked(source_handle, event.shared_id);
      // What this space held for the source before it goes: a link to a local
      // element, a pending (unresolved) temp, or an owned replica. Only a
      // source this space knew is retired, so an unrelated unshare changes
      // nothing here.
      std::optional<element_handle_t> linked_target;
      if (hub_ && event.source_element_id != kInvalidElementId) {
        linked_target = hub_->GetLinker().GetTarget(source_handle, space_id_);
      }
      // Only a resolved link is retired: a pending temp or a replica without
      // one leaves nothing that could be mistaken for the source's counterpart.
      const bool known_source = linked_target.has_value();
      // Clear an unresolved received share for this exact source. Its temp is
      // ours to remove; a shared-id lookup could delete unrelated furniture.
      for (auto it = pending_share_source_.begin();
           it != pending_share_source_.end();) {
        if (it->second == source_handle) {
          PreserveIndependentChildren(it->first);
          const auto subtree = element_tree_.GetSubtreeIds(it->first);
          if (!subtree.empty() && element_tree_.RemoveElement(it->first).ok()) {
            AdvanceSceneInputsLocked(true, true);
            CleanupRemovedElements(subtree);
            removed_ids.push_back(it->first);
          }
          it = pending_share_source_.erase(it);
        } else {
          ++it;
        }
      }
      auto owned = remote_owned_replicas_.find(source_handle);
      if (owned != remote_owned_replicas_.end()) {
        const element_id_t local_id = owned->second;
        PreserveIndependentChildren(local_id);
        const auto subtree = element_tree_.GetSubtreeIds(local_id);
        if (!subtree.empty() && element_tree_.RemoveElement(local_id).ok()) {
          AdvanceSceneInputsLocked(true, true);
          CleanupRemovedElements(subtree);
          removed_ids.push_back(local_id);
        }
        remote_owned_replicas_.erase(owned);
      }
      if (hub_ && event.source_element_id != kInvalidElementId) {
        // The source no longer exists remotely. Remove this space's end of its
        // correspondence in every local outcome, including LinkToExisting where
        // no replica was owned locally. Only this space's pair: in a hub shared
        // by several local spaces the others remove their own, and each is told.
        if (linked_target) {
          hub_->GetLinker().RemoveLink(source_handle, *linked_target);
          // A LinkToExisting target survives the unshare: the app releases its
          // counterpart. (A removed AddAsNew replica was reported above.)
          if (element_tree_.GetElement(linked_target->element_id) != nullptr) {
            link_removed.push_back({linked_target->element_id,
                                    event.source_space_id,
                                    LinkRemovedReason::kPeerUnshared});
          }
        }
        if (known_source && retired_sources_.insert(source_handle).second) {
          ++counterpart_revision_;
        }
      }
    } else if (event.type == ElementEvent::kUnlinked) {
      // A receiver dropped its link to a source this space shares. Only the
      // sharer acts; the share itself stays live.
      if (event.source_space_id == space_id_ &&
          locally_shared_elements_.count(event.source_element_id) != 0) {
        link_removed.push_back({event.source_element_id, event.peer_space_id,
                                LinkRemovedReason::kPeerUnlinked});
      }
    }
  }

  OnElementAddedFn on_added;
  OnElementRemovedFn on_removed;
  OnLinkRemovedFn on_link_removed;
  {
    std::lock_guard<std::mutex> lock(space_mutex_);
    on_added = on_element_added_;
    on_removed = on_element_removed_;
    on_link_removed = on_link_removed_;
  }
  if (on_added) {
    for (const auto& [id, e] : added) on_added(id, e.shared_id, e.tag, e.user_data);
  }
  if (on_removed) {
    for (element_id_t removed_id : removed_ids) {
      on_removed(removed_id);
    }
  }
  if (on_link_removed) {
    for (const LinkRemoval& removal : link_removed) {
      on_link_removed(removal.local, removal.peer, removal.reason);
    }
  }
}

void Space::AddReceivedElementLocked(const ElementEvent& event,
                                     std::vector<std::pair<element_id_t, ElementEvent>>* added) {
  // Idempotency: if the hub has already recorded a link from this source
  // handle to this space, the element was resolved in a prior share and
  // must not be duplicated.
  const element_handle_t source_handle{event.source_space_id, event.source_element_id};
  const bool pending = std::any_of(
      pending_share_source_.begin(), pending_share_source_.end(),
      [&](const auto& entry) { return entry.second == source_handle; });
  if (hub_ && event.source_element_id != kInvalidElementId &&
      (pending || hub_->GetLinker().GetTarget(source_handle, space_id_).has_value())) {
    return;  // already linked: no second element, no callback
  }

  // The parent is the element with the source parent's shared id. When this
  // space has none (the source's parent is not shared here, or has no shared
  // id), the element goes under the room rather than becoming a parentless
  // root of its own, which no containment check reaches
  // (FindContainmentViolation checks against the parent and its children).
  // A space without a room keeps received elements as roots, as before.
  const Element* named = event.parent_shared_id != kInvalidSharedElementId
                             ? element_tree_.GetElementBySharedId(event.parent_shared_id)
                             : nullptr;
  const Element* room = RoomLocked();
  const element_id_t parent_id =
      named ? named->GetId() : room ? room->GetId() : kInvalidElementId;
  if (!named && room && TraceSink::ForSpace(space_id_).Active()) {
    nlohmann::ordered_json ev;
    ev["ev"] = "recv_parent_fallback";
    ev["space"] = space_id_;
    ev["shared_id"] = event.shared_id;
    ev["parent_shared_id"] = event.parent_shared_id;
    ev["room"] = parent_id;
    TraceSink::ForSpace(space_id_).Emit(ev);
  }

  // A bootstrap copies the host's ids verbatim, which include the ids the
  // host gave its own received elements; never hand out one in use.
  while (element_tree_.GetElement(next_element_id_) != nullptr) {
    ++next_element_id_;
  }
  const element_id_t new_id = next_element_id_++;
  if (!element_tree_.AddElement(new_id, parent_id).ok()) return;
  AdvanceSceneInputsLocked(true, true);
  Element* elem = element_tree_.GetElement(new_id);
  if (!elem) return;
  elem->SetSharedId(event.shared_id);
  if (!event.tag.empty()) {
    elem->SetTag(event.tag);
  }
  // Record source handle so ResolveSharedElement can create the link.
  if (event.source_element_id != kInvalidElementId) {
    // Shared anew: the source is not retired any more (an app that
    // never resolves relies on shared-id matching, as before).
    retired_sources_.erase(source_handle);
    pending_share_source_[new_id] = source_handle;
  }
  added->emplace_back(new_id, event);
}

void Space::ClearRetainedEvaluationsForSourceLocked(
    const element_handle_t& source, shared_element_id_t shared_id) {
  for (auto it = retained_receiver_evaluations_.begin();
       it != retained_receiver_evaluations_.end();) {
    const auto& packet = it->second.update;
    const bool matches = packet.actor_source_element_id != kInvalidElementId
        ? packet.actor_source_space_id == source.space_id &&
              packet.actor_source_element_id == source.element_id
        : packet.source_space_id == source.space_id &&
              packet.actor_shared_id != kInvalidSharedElementId &&
              packet.actor_shared_id == shared_id;
    if (!matches) { ++it; continue; }
    ClearReceiverTargetsLocked(it->first);
    ++counterpart_revision_;
    it = retained_receiver_evaluations_.erase(it);
  }
}

int Space::UnlinkElement(element_id_t elem_id) {
  SpaceHub* hub = nullptr;
  std::vector<ElementEvent> events;
  {
    std::lock_guard<std::mutex> lock(space_mutex_);
    if (!hub_ || element_tree_.GetElement(elem_id) == nullptr) return 0;
    const element_handle_t local{space_id_, elem_id};
    for (const element_handle_t& source : hub_->GetLinker().GetSourcesOf(local)) {
      // Never a source's own space: a link is between two spaces.
      if (source.space_id == space_id_) continue;
      if (!hub_->GetLinker().RemoveLink(source, local)) continue;
      // An AddAsNew replica is released into local ownership, not deleted.
      auto owned = remote_owned_replicas_.find(source);
      if (owned != remote_owned_replicas_.end() && owned->second == elem_id) {
        remote_owned_replicas_.erase(owned);
      }
      retired_sources_.insert(source);
      ClearRetainedEvaluationsForSourceLocked(source, kInvalidSharedElementId);
      ++counterpart_revision_;
      ElementEvent event;
      event.type = ElementEvent::kUnlinked;
      event.shared_id = kInvalidSharedElementId;
      event.source_space_id = source.space_id;
      event.source_element_id = source.element_id;
      event.peer_space_id = space_id_;
      events.push_back(std::move(event));
    }
    hub = hub_;
  }
  // Peer dispatch is synchronous for local hubs and may re-enter this space.
  for (const ElementEvent& event : events) {
    hub->PropagateElementEvent(event, space_id_);
  }
  return static_cast<int>(events.size());
}

void Space::AdoptBootstrapShare(element_id_t element_id,
                                const ElementEvent& event) {
  if (event.type != ElementEvent::kAdded) return;
  bool occupied = false;
  {
    std::lock_guard<std::mutex> lock(space_mutex_);
    Element* elem = element_tree_.GetElement(element_id);
    if (!elem) return;
    // One pending source per element: a second share the host stands for with
    // this same element is received as a live one would be, onto its own.
    occupied = pending_share_source_.count(element_id) != 0;
  }
  if (occupied) {
    OnElementEvent(event);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(space_mutex_);
    if (event.source_element_id != kInvalidElementId) {
      const element_handle_t source{event.source_space_id,
                                    event.source_element_id};
      const bool pending = std::any_of(
          pending_share_source_.begin(), pending_share_source_.end(),
          [&](const auto& entry) { return entry.second == source; });
      if (pending || (hub_ && hub_->GetLinker().GetTarget(source, space_id_))) {
        return;
      }
      pending_share_source_[element_id] = source;
    }
  }
  OnElementAddedFn on_added;
  {
    std::lock_guard<std::mutex> lock(space_mutex_);
    on_added = on_element_added_;
  }
  if (on_added) on_added(element_id, event.shared_id, event.tag, event.user_data);
}

element_id_t Space::LocalElementForSource(const element_handle_t& source) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (source.space_id == space_id_) {
    return locally_shared_elements_.count(source.element_id) != 0
               ? source.element_id
               : kInvalidElementId;
  }
  for (const auto& entry : pending_share_source_) {
    if (entry.second == source) return entry.first;
  }
  auto owned = remote_owned_replicas_.find(source);
  if (owned != remote_owned_replicas_.end()) return owned->second;
  if (hub_) {
    if (auto target = hub_->GetLinker().GetTarget(source, space_id_)) {
      return target->element_id;
    }
  }
  return kInvalidElementId;
}

void Space::SetOnElementAdded(OnElementAddedFn fn) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  on_element_added_ = std::move(fn);
}

void Space::SetOnElementRemoved(OnElementRemovedFn fn) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  on_element_removed_ = std::move(fn);
}

void Space::SetOnLinkRemoved(OnLinkRemovedFn fn) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  on_link_removed_ = std::move(fn);
}

void Space::ResolveSharedElement(element_id_t received_id,
                                 element_id_t resolved_id) {
  std::lock_guard<std::mutex> lock(space_mutex_);

  auto it = pending_share_source_.find(received_id);
  if (it == pending_share_source_.end()) {
    return;  // Unknown received_id — already resolved or never created.
  }

  element_handle_t source_handle = it->second;
  element_id_t target_id;

  // Capture the shared_id BEFORE any RemoveElement below — the received temp
  // element is the only place it's recorded (SPEC S3 ev=share's shared_id).
  shared_element_id_t shared_id = kInvalidSharedElementId;
  if (const Element* received = element_tree_.GetElement(received_id)) {
    shared_id = received->GetSharedId();
  }
  const char* resolution =
      (resolved_id != received_id) ? "link_to_existing" : "add_as_new";

  if (resolved_id != received_id) {
    // LinkToExisting: remove the temp received element (no duplicate). The
    // cross-space identity is the link recorded below — the local element keeps
    // its own internal id; correspondence lives in the ElementLinker, not in a
    // matching shared id.
    if (!element_tree_.GetElement(resolved_id)) return;
    PreserveIndependentChildren(received_id);
    const auto subtree = element_tree_.GetSubtreeIds(received_id);
    if (!element_tree_.RemoveElement(received_id).ok()) return;
    AdvanceSceneInputsLocked(true, true);
    CleanupRemovedElements(subtree);
    target_id = resolved_id;
  } else {
    // AddAsNew: keep the temp element as the canonical local element.
    target_id = received_id;
    remote_owned_replicas_[source_handle] = received_id;
  }

  // Record the cross-space link in the hub's ElementLinker. This link IS the
  // identity: OnRemoteUpdate resolves referents (and actors) through
  // it.
  if (hub_) {
    hub_->GetLinker().CreateLink(source_handle,
                                 element_handle_t{space_id_, target_id});
  }
  // Resolved anew: the source has a counterpart again.
  retired_sources_.erase(source_handle);

  pending_share_source_.erase(it);

  // SPEC S3 ev=share: one line per resolved share, so the trace shows how
  // received_id -> resolved_id mapped (dedup vs new).
  if (TraceSink::ForSpace(space_id_).Active()) {
    nlohmann::ordered_json ev;
    ev["ev"] = "share";
    ev["t"] = space_clock_;
    ev["space"] = space_id_;
    ev["shared_id"] = shared_id;
    ev["received"] = received_id;
    ev["resolved"] = target_id;
    ev["resolution"] = resolution;
    TraceSink::ForSpace(space_id_).Emit(ev);
  }
}



// ---------------------------------------------------------------------------
// Observability and injection.
// ---------------------------------------------------------------------------

bool Space::SetRealizationPolicyParams(const RealizationPolicyConfig& policy,
                                       const ImminenceConfig& imminence) {
  if (!IsValidRealizationPolicyConfig(policy) ||
      !IsValidImminenceConfig(imminence)) {
    return false;
  }
  std::lock_guard<std::mutex> lock(space_mutex_);
  RealizationPolicyConfig next = policy;
  next.type = realization_policy_config_.type;  // family: SetRealizationPolicy
  const RealizationPolicyConfig& cur = realization_policy_config_;
  // The receiver schedule is a solve input (rich policy validity domain);
  // the imminence constants only change what a SOURCE writes into future
  // payloads, so they must not invalidate receiver snapshots.
  const bool receiver_changed =
      cur.eta_0 != next.eta_0 || cur.decay_gamma != next.decay_gamma ||
      cur.epsilon != next.epsilon || cur.eta_min != next.eta_min ||
      cur.eta_max != next.eta_max;
  const bool source_changed =
      imminence_config_.progress_tau_s != imminence.progress_tau_s ||
      imminence_config_.epsilon_v != imminence.epsilon_v ||
      imminence_config_.t_max != imminence.t_max ||
      imminence_config_.engage_kappa != imminence.engage_kappa;
  realization_policy_config_ = next;
  imminence_config_ = imminence;
  if (receiver_changed) ++policy_revision_;
  if (source_changed) ++imminence_revision_;
  return true;
}

RealizationPolicyConfig Space::GetRealizationPolicyParams() const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  // Resolving the env default here would mutate state from a const getter;
  // report the family the next update will use.
  RealizationPolicyConfig out = realization_policy_config_;
  out.type = realization_policy_type_;
  if (!realization_policy_env_resolved_) {
    if (const char* e = std::getenv("SB_V2_REALIZATION_POLICY")) {
      const std::string policy(e);
      if (policy == "uniform") out.type = RealizationPolicyType::kUniform;
      if (policy == "greedy") out.type = RealizationPolicyType::kGreedy;
      if (policy == "progressive")
        out.type = RealizationPolicyType::kProgressive;
    }
  }
  return out;
}

ImminenceConfig Space::GetImminenceParams() const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return imminence_config_;
}

void Space::EmitPolicyTraceIfChangedLocked(double t) {
  auto& sink = TraceSink::ForSpace(space_id_);
  if (!sink.Active() || (traced_policy_revision_ == policy_revision_ &&
                         traced_imminence_revision_ == imminence_revision_))
    return;
  traced_policy_revision_ = policy_revision_;
  traced_imminence_revision_ = imminence_revision_;
  nlohmann::ordered_json ev;
  ev["ev"] = "policy";
  ev["t"] = t;
  ev["space"] = space_id_;
  ev["format"] = kTraceFormatVersion;
  ev["revision"] = policy_revision_;
  ev["imminence_revision"] = imminence_revision_;
  ev["policy"] = RealizationPolicyName(realization_policy_type_);
  ev["eta_0"] = realization_policy_config_.eta_0;
  ev["gamma"] = realization_policy_config_.decay_gamma;
  ev["epsilon_t"] = realization_policy_config_.epsilon;
  ev["eta_min"] = realization_policy_config_.eta_min;
  ev["eta_max"] = realization_policy_config_.eta_max;
  ev["progress_tau_s"] = imminence_config_.progress_tau_s;
  ev["epsilon_v"] = imminence_config_.epsilon_v;
  ev["t_max"] = imminence_config_.t_max;
  ev["kappa"] = imminence_config_.engage_kappa;
  sink.Emit(ev);
}

void Space::OnTraceRebound() {
  std::lock_guard<std::mutex> lock(space_mutex_);
  traced_policy_revision_ = 0;
  traced_imminence_revision_ = 0;
}

bool Space::ObservabilityCaptureActiveLocked() const {
  return observability_capture_.load(std::memory_order_relaxed) ||
         TraceSink::ForSpace(space_id_).Active();
}

int64_t Space::SharedOrLocalIdLocked(element_id_t id) const {
  const Element* e = element_tree_.GetElement(id);
  return (e && e->GetSharedId() != kInvalidSharedElementId)
             ? static_cast<int64_t>(e->GetSharedId())
             : static_cast<int64_t>(id);
}

std::vector<Space::ScopeObservation> Space::GetScopeRecords(
    spatialblend::entity_id_t actor_id) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  std::vector<ScopeObservation> out;
  auto it = bindings_.find(actor_id);
  if (it == bindings_.end()) return out;
  out.reserve(it->second.size());
  for (size_t i = 0; i < it->second.size(); ++i) {
    const Binding& b = it->second[i];
    ScopeObservation o;
    o.binding_index = i;
    o.anchor_name = b.anchor_name;
    o.tag_filter = b.tag_filter;
    o.kernel_type = b.kernel_type;
    o.record = b.scope_record;
    out.push_back(std::move(o));
  }
  return out;
}

bool Space::CopyPolicyObservation(spatialblend::entity_id_t actor_id,
                                  PolicyObservation* out) const {
  if (!out) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto it = policy_observations_.find(actor_id);
  if (it == policy_observations_.end() || !it->second.valid) return false;
  *out = it->second;
  return true;
}

bool Space::CopySolverWarmState(spatialblend::entity_id_t actor_id,
                                SolverWarmState* out) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto it = solver_warm_states_.find(actor_id);
  if (it == solver_warm_states_.end()) return false;
  if (out) *out = it->second;
  return true;
}

bool Space::CopySolverObservation(spatialblend::entity_id_t actor_id,
                                  SolverObservation* out) const {
  if (!out) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto it = solver_observations_.find(actor_id);
  if (it == solver_observations_.end() || !it->second.valid) return false;
  *out = it->second;
  return true;
}

bool Space::InjectRelationPayloads(spatialblend::entity_id_t actor_id,
                                   std::vector<RelationPayload> payloads,
                                   double time_seconds) {
  if (!std::isfinite(time_seconds)) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  Actor* actor = FindActorLocked(actor_id);
  if (!actor) return false;
  RemoteRelationUpdate update;
  update.timing = RelationUpdateTiming::kLegacy;
  update.source_space_id = kInvalidSpaceId;
  // Receiver-side identity only: the phase tracker keys on it. No source
  // handle is attached, so no linker resolution is attempted for the actor.
  const Element* actor_elem = element_tree_.GetElement(actor->GetElementId());
  update.actor_shared_id =
      (actor_elem && actor_elem->GetSharedId() != kInvalidSharedElementId)
          ? actor_elem->GetSharedId()
          : static_cast<shared_element_id_t>(actor->GetElementId());
  update.step_duration = update_interval_;
  update.source_clock = time_seconds;
  for (auto& p : payloads) {
    // Never trust caller-supplied receiver-local scratch or source handles.
    p.resolved_local_element_id = kInvalidElementId;
    p.referent_source_space_id = kInvalidSpaceId;
    p.referent_source_element_id = kInvalidElementId;
    p.observed_facing_point = false;
  }
  update.payloads = std::move(payloads);
  space_clock_ = std::max(space_clock_, time_seconds);
  const uint64_t linker_revision_before =
      hub_ ? hub_->GetLinker().GetGraphRevision() : 0;
  const int64_t ingress_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  EvaluateRemoteRelationLocked(update, actor_id, ingress_ns, false,
                               linker_revision_before,
                               /*carries_containment=*/false);
  return true;
}

bool Space::EvaluateRelation(RelationKernelType kernel_type,
                             const Eigen::Vector3d& anchor_position,
                             double anchor_yaw, element_id_t referent,
                             element_id_t scope_root, bool explicit_scope,
                             const std::vector<StateType>& dof,
                             RelationEvaluation* out) const {
  if (!out || !anchor_position.allFinite() || !std::isfinite(anchor_yaw)) {
    return false;
  }
  auto components = KernelFactory::Create(kernel_type, dof);
  if (!components.kernel || !components.grounding_rule) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  const Element* elem = element_tree_.GetElement(referent);
  if (!elem) return false;
  RelationEvaluation result;
  const NavigableSpatialDefinition* navigable_def = nullptr;
  if (scope_root != kInvalidElementId) {
    if (!element_tree_.GetElement(scope_root)) return false;
    result.deepest_containing =
        FindDeepestContaining(anchor_position, scope_root, element_tree_);
    // Mirrors the binding pipeline's navigable search: explicit scopes stop
    // at their root, owner-derived scopes may borrow any ancestor.
    for (element_id_t e = result.deepest_containing; e != kInvalidElementId;) {
      const auto* scope_elem = element_tree_.GetElement(e);
      if (!scope_elem) break;
      if (scope_elem->HasGeometryRole(GeometryRole::kNavigable)) {
        navigable_def = dynamic_cast<const NavigableSpatialDefinition*>(
            scope_elem->GetSpatialDefinition());
        if (navigable_def) {
          result.navigable_element = e;
          break;
        }
      }
      if (explicit_scope && e == scope_root) break;
      e = scope_elem->GetParentId();
    }
  }
  result.support = components.grounding_rule->ComputeSupport(
      *elem, element_tree_, anchor_position, anchor_yaw, navigable_def);
  result.grounded =
      components.kernel->Encode(anchor_position, anchor_yaw, result.support);
  *out = std::move(result);
  return true;
}

bool Space::GetElementWorldTransform(element_id_t elem_id,
                                     Eigen::Matrix4d* out) const {
  if (!out) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!element_tree_.GetElement(elem_id)) return false;
  *out = element_tree_.GetWorldTransform(elem_id);
  return true;
}

bool Space::CopyRealizationResult(spatialblend::entity_id_t actor_id,
                                  RealizationResult* out) const {
  if (!out) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  const RealizationResult* result = GetRealizationTargets(actor_id);
  if (!result) return false;
  *out = *result;
  return true;
}

bool Space::GetElementParent(element_id_t elem_id,
                             element_id_t* parent) const {
  if (!parent) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  const Element* elem = element_tree_.GetElement(elem_id);
  if (!elem) return false;
  *parent = elem->GetParentId();
  return true;
}

// ---- embodied-pose sampling port and proposal mailbox ----------------------

std::shared_ptr<Space::ActorMailbox> Space::MailboxFor(
    spatialblend::entity_id_t actor_id) const {
  std::lock_guard<std::mutex> lock(mailboxes_mutex_);
  const auto it = mailboxes_.find(actor_id);
  return it == mailboxes_.end() ? nullptr : it->second;
}

EmbodiedPublishResult Space::PublishEmbodiedPose(
    spatialblend::entity_id_t actor_id, const EmbodiedPoseInput& pose,
    double timestamp_s, uint64_t sequence) {
  // Wait-free with respect to a solve: space_mutex_ is never taken.
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox) return EmbodiedPublishResult::kInvalid;
  if (mailbox->input_owned.load()) return EmbodiedPublishResult::kRefused;
  const int64_t now_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  return mailbox->port.Publish(pose, timestamp_s, sequence, now_ns);
}

EmbodiedPublishResult Space::PublishReceiverInputPose(
    spatialblend::entity_id_t actor_id, const EmbodiedPoseInput& pose,
    double timestamp_s, uint64_t sequence) {
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox) return EmbodiedPublishResult::kInvalid;
  const int64_t now_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  return mailbox->port.Publish(pose, timestamp_s, sequence, now_ns);
}

bool Space::SampleEmbodiedPose(spatialblend::entity_id_t actor_id,
                               EmbodiedSample* out) const {
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox || !out) return false;
  *out = mailbox->port.Sample();
  return true;
}

bool Space::ReadProposal(spatialblend::entity_id_t actor_id,
                         ReceiverProposal* out) const {
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox || !out) return false;
  if (!mailbox->proposal.Load(out)) *out = ReceiverProposal{};
  return true;
}

void Space::RetainIdlePositionLocked(spatialblend::entity_id_t actor_id,
                                     AnchorProposal* anchor) {
  if (!anchor->position_active || anchor->position_requested ||
      anchor->placement)
    return;
  const auto mailbox = MailboxFor(actor_id);
  ReceiverProposal previous;
  if (!mailbox || !mailbox->proposal.Load(&previous)) return;
  const auto* old = previous.Find(anchor->anchor_id);
  if (!old || !old->position_active) return;
  anchor->x = old->x;
  anchor->y = old->y;
  anchor->z = old->z;
}

void Space::PublishProposalLocked(spatialblend::entity_id_t actor_id,
                                  ReceiverProposal proposal,
                                  std::vector<ActorEvent> events) {
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox) return;
  ReceiverProposal previous;
  const bool had_previous = mailbox->proposal.Load(&previous);
  bool publish = !had_previous || !events.empty();
  const bool deliver = mailbox->target_events_enabled.load();
  const uint64_t revision = proposal_revision_ + 1;
  std::vector<ActorEvent> target_events;
  const auto event_for = [&](const AnchorProposal& a, ActorEvent::Kind kind,
                             uint32_t channels) {
    ActorEvent e;
    e.kind = kind;
    e.revision = revision;
    e.source_sequence = proposal.source_sequence;
    e.source_shared_id = proposal.source_shared_id;
    e.anchor_id = a.anchor_id;
    e.x = a.x;
    e.y = a.y;
    e.z = a.z;
    e.yaw = a.yaw;
    e.pitch = a.pitch;
    e.roll = a.roll;
    e.orientation_type = a.orientation_type;
    e.channels = channels;
    e.sample_epoch = a.request_sample.epoch;
    e.sample_sequence = a.request_sample.sequence;
    e.sample_yaw = a.request_sample.yaw;
    e.yaw_turn = a.yaw_turn;
    e.proposal_flags = a.flags;
    e.proposal_cause = a.cause;
    e.facing_yaw = a.facing_yaw;
    e.proposal_status = a.status;
    e.request_flags =
        a.placement && kind != ActorEvent::Kind::kTargetRelease ? 1u : 0u;
    return e;
  };
  const auto reconcile = [&](AnchorProposal& a) {
    const AnchorProposal* old =
        had_previous ? previous.Find(a.anchor_id) : nullptr;
    // A new sample identity is not withdrawal of an unfinished request. In
    // particular, feedback after applying a queued placement must not cancel a
    // valid ordinary goal decoded later in that same drain.
    const bool reset = a.placement;
    const bool sample_reset =
        (a.cause & kCauseEmbodiedReset) != 0 ||
        (old && old->request_sample.valid && a.request_sample.valid &&
         old->request_sample.epoch != a.request_sample.epoch);
    const uint32_t released =
        old ? ((old->position_active && (!a.position_active || reset) ? 1u
                                                                      : 0u) |
               (old->yaw_active && (!a.yaw_active || reset) ? 2u : 0u))
            : 0u;
    if (released) {
      publish = true;
      if (deliver)
        target_events.push_back(
            event_for(a, ActorEvent::Kind::kTargetRelease, released));
    }
    const auto actionable = [](ProposalStatus status) {
      return status == ProposalStatus::kExact ||
             status == ProposalStatus::kInexact;
    };
    if (a.position_requested || a.yaw_requested || reset || sample_reset ||
        (old && (old->position_active != a.position_active ||
                 old->yaw_active != a.yaw_active ||
                 (old->status != a.status &&
                  (!actionable(old->status) || !actionable(a.status))) ||
                 old->orientation_type != a.orientation_type ||
                 old->pitch != a.pitch || old->roll != a.roll)))
      publish = true;
    if (deliver && a.position_requested)
      target_events.push_back(
          event_for(a, ActorEvent::Kind::kPositionRequest, 1));
    if (deliver && a.yaw_requested)
      target_events.push_back(event_for(a, ActorEvent::Kind::kYawRequest, 2));
  };
  reconcile(proposal);
  for (auto& a : proposal.anchors) reconcile(a);
  if (had_previous) {
    for (const auto* old : previous.All()) {
      if (proposal.Find(old->anchor_id)) continue;
      const uint32_t channels =
          (old->position_active ? 1u : 0u) | (old->yaw_active ? 2u : 0u);
      if (!channels) continue;
      publish = true;
      if (deliver)
        target_events.push_back(
            event_for(*old, ActorEvent::Kind::kTargetRelease, channels));
    }
  }
  if (!publish)
    return;  // diagnostics were captured by the solve, not by a fabricated hold
             // publication
  proposal.revision = ++proposal_revision_;
  // Lifecycle markers precede releases and their OWN immutable poses. Snapshot
  // readiness and batch visibility are guaranteed before the first
  // poll/observer; the queue mutex is never held across callbacks.
  std::stable_partition(target_events.begin(), target_events.end(),
                        [](const ActorEvent& event) {
                          return event.kind == ActorEvent::Kind::kTargetRelease;
                        });
  std::vector<ActorEvent> batch;
  for (auto& event : events)
    if (event.kind == ActorEvent::Kind::kSourceException)
      batch.push_back(std::move(event));
  for (auto& event : target_events) batch.push_back(std::move(event));
  for (auto& event : events)
    if (event.kind != ActorEvent::Kind::kSourceException)
      batch.push_back(std::move(event));
  auto shared = std::make_shared<const ReceiverProposal>(std::move(proposal));
  mailbox->proposal.Store(shared);
  AppendActorEvents(actor_id, std::move(batch));
  // ev=proposal (trace format 8): what core proposed and why.
  if (TraceSink::ForSpace(space_id_).Active()) {
    const ReceiverProposal& p = *shared;
    const char* status = p.status == ProposalStatus::kNoSample     ? "no_sample"
                         : p.status == ProposalStatus::kExact       ? "exact"
                         : p.status == ProposalStatus::kInexact     ? "inexact"
                         : p.status == ProposalStatus::kNotConverged
                             ? "not_converged"
                                                                    : "none";
    nlohmann::ordered_json ev;
    ev["ev"] = "proposal";
    ev["t"] = space_clock_;
    ev["space"] = space_id_;
    ev["actor"] = GetActorSharedIdForEvents(actor_id);
    ev["status"] = status;
    ev["source_seq"] = p.source_sequence;
    ev["sample_seq"] = p.sample_sequence;
    ev["pos"] = {p.x, p.y};
    ev["yaw"] = p.yaw;
    ev["cause"] = p.cause;
    ev["engaged"] = p.engaged();
    ev["err_spacing_target"] = p.err_spacing_target;
    ev["err_spacing_sample"] = p.err_spacing_sample;
    // The other relation-bearing anchors of this set (always present: empty
    // when the root is the only one).
    {
      nlohmann::ordered_json anchors = nlohmann::ordered_json::array();
      for (const auto& a : p.anchors) {
        nlohmann::ordered_json one;
        one["anchor"] = a.anchor_name;
        one["id"] = a.anchor_id;
        one["status"] = static_cast<int>(a.status);
        one["pos"] = {a.x, a.y, a.z};
        one["yaw"] = a.yaw;
        one["cause"] = a.cause;
        one["engaged"] = a.engaged();
        anchors.push_back(std::move(one));
      }
      ev["anchors"] = std::move(anchors);
    }
    TraceSink::ForSpace(space_id_).Emit(ev);
  }
  for (const auto& entry : proposal_observers_) entry.second(actor_id, *shared);
  // An observer (an in-process driver) may have published a pose: bring the
  // element-tree mirror up to date before the lock is released.
  if (!proposal_observers_.empty()) SyncEmbodiedMirrorsLocked();
}

uint64_t Space::AddProposalObserver(ProposalObserver observer) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  const uint64_t id = next_proposal_observer_id_++;
  proposal_observers_.emplace_back(id, std::move(observer));
  return id;
}

void Space::RemoveProposalObserver(uint64_t id) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  proposal_observers_.erase(
      std::remove_if(proposal_observers_.begin(), proposal_observers_.end(),
                     [id](const auto& e) { return e.first == id; }),
      proposal_observers_.end());
}


// The receiver avatar's Actor transform and leaf element are a DERIVED mirror of
// the driver's port, keyed to the sample (epoch, sequence): other local actors
// evaluate their relations against the avatar's live position through the
// element tree, so the tree must show the sampled pose. This is the only
// writer of a receiver avatar's transform; it copies the sample verbatim (no
// integration, no extrapolation) and skips actors whose pose is a source feed
// (UpdateActorState / observation frames).
void Space::SyncEmbodiedMirrorsLocked() {
  std::vector<std::pair<spatialblend::entity_id_t, std::shared_ptr<ActorMailbox>>>
      boxes;
  {
    std::lock_guard<std::mutex> lock(mailboxes_mutex_);
    boxes.assign(mailboxes_.begin(), mailboxes_.end());
  }
  for (const auto& [actor_id, mailbox] : boxes) {
    const EmbodiedSample sample = mailbox->port.Sample();
    if (!sample.valid) continue;
    auto& mirrored = mirrored_sample_[actor_id];
    if (mirrored.first == sample.epoch && mirrored.second == sample.sequence)
      continue;
    Actor* actor = FindActorLocked(actor_id);
    if (!actor || !receiving_actors_.count(actor_id) ||
        actor_update_seconds_.count(actor_id) ||
        observation_clocks_.count(actor_id) || receiver_inputs_.count(actor_id))
      continue;
    mirrored = {sample.epoch, sample.sequence};
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 3) = sample.x;
    t(1, 3) = sample.y;
    t(0, 0) = t(1, 1) = std::cos(sample.yaw);
    t(0, 1) = -std::sin(sample.yaw);
    t(1, 0) = std::sin(sample.yaw);
    actor->SetTransform(t);
    Element* leaf = element_tree_.GetElement(actor->GetElementId());
    if (leaf && !leaf->GetSpatialDefinition() &&
        element_tree_.GetDirectChildIds(leaf->GetId()).empty())
      SetElementTransformLocked(leaf->GetId(), t, /*actor_pose=*/true);
  }
}

// A person is a Facing referent like any visible element. Its element carries no
// declared geometry (the actor's pose moves it, SyncEmbodiedMirrorsLocked / UpdateActorState), so the sightline's end
// would be the floor point under the person and anything between taller than that ray (a table between two people)
// would hide them. The person's body is its actor's anchors: its top is the highest anchor (the head) above the
// element. One rule for every actor, source or received; no anchor name is read.
std::unordered_map<element_id_t, double> Space::ActorBodyTopsLocked() const {
  std::unordered_map<element_id_t, double> tops;
  for (const auto& [id, actor] : actors_) {
    if (!actor) continue;
    const Element* e = element_tree_.GetElement(actor->GetElementId());
    if (!e || e->GetDeclaredSpatialDefinition()) continue;
    double top = 0.0;
    for (const SubjectAnchor& a : actor->GetAnchors()) top = std::max(top, a.local_offset.z());
    if (top > 0.0) tops[e->GetId()] = std::max(tops[e->GetId()], top);
  }
  return tops;
}

// ---- the room scope: one solve for every anchor's Spacing / Facing relations ----------

SolverResult Space::SolveRoomScopeLocked(const EmbodiedSample& sample, const std::vector<RelationPayload>& payloads,
                                         const std::vector<double>& weights,
                                         const NavigableSpatialDefinition* navigable_def, bool placed, bool cold,
                                         spatialblend::entity_id_t actor_id, uint32_t anchor_id) {
  // Continuation from the anchor's embodied sample: the tracking pass, the restarts and the Facing sign rule are seeded
  // there, never at core's previous answer. `placed`: the actor's first proposal or a source exception (the absolute
  // relation is solved); `cold` also marks the anchor's own driver discontinuity.
  SolverWarmStart warm;
  warm.has_position = true;
  warm.position = Eigen::Vector2d(sample.x, sample.y);
  warm.has_yaw = true;
  warm.yaw = sample.yaw;
  warm.cold = cold;
  warm.absolute = placed;
  YawTrack& track = yaw_track_[{actor_id, anchor_id}];
  warm.turn_sign = track.turn_sign;
  SolverResult r = solver_.Solve(warm.position, sample.yaw, payloads, weights, element_tree_, navigable_def,
                                 &GetReferentResolver(), std::nullopt, warm);
  // The receiver's own proposed turn on a continuation sets its turn direction; a placed answer does not. The
  // direction is the turn it requested (yaw_turn, ABI 4.12), not the shortest arc to its yaw: after a long-way turn the
  // arc points the other way, and the next tie took the middle behind the receiver (a half turn back).
  if (!placed && r.yaw_specified) {
    const double step = r.yaw_turn;
    if (step != 0.0) track.turn_sign = step > 0.0 ? 1 : -1;
  }
  // For every anchor: inside a containment-transfer leaf the
  // anchor's yaw is the Facing yaw when a Facing is authored, else the leaf-relative yaw of its containment relation
  // (LeafRelativeYawLocked). It is never YAW_UNSPECIFIED while the anchor is inside a leaf the receiver maps.
  double leaf_yaw = 0.0;
  if (!r.no_grounded_referent && LeafRelativeYawLocked(payloads, weights, &leaf_yaw) &&
      ResolveRootYawSource(r.yaw_specified, /*in_leaf=*/true) == RootYawSource::kLeafRelative) {
    r.yaw = leaf_yaw;
    r.yaw_specified = true;
    r.yaw_active = true;
    r.yaw_requested = r.position_requested;
    r.yaw_turn = std::remainder(leaf_yaw - sample.yaw, 2.0 * M_PI);  // the leaf-relative yaw: its shortest arc
  }
  return r;
}

// ---- containment, a relation of any anchor ----------------------------------------

const RelationPayload* Space::ContainmentRelationLocked(const std::vector<RelationPayload>& payloads,
                                                        const std::vector<double>& weights,
                                                        const Element** leaf_out) const {
  const RelationPayload* best = nullptr;
  const Element* best_leaf = nullptr;
  int best_depth = -1;
  for (size_t k = 0; k < payloads.size(); ++k) {
    const RelationPayload& rp = payloads[k];
    const double w = k < weights.size() ? weights[k] : 1.0;
    if (!rp.enabled || !(w > 0.0) || rp.kernel_type != RelationKernelType::kAvatarSpacing || !rp.support.has_interior)
      continue;
    const Element* leaf = rp.resolved_local_element_id != kInvalidElementId
                              ? element_tree_.GetElement(rp.resolved_local_element_id)
                              : GetReferentResolver().Resolve(rp.referent_shared_id, element_tree_);
    if (!leaf || !leaf->HasContainmentTransfer()) continue;
    // The deepest leaf the anchor is inside (as the source's DescendScope picks its transfer leaf).
    int depth = 0;
    for (element_id_t e = leaf->GetParentId(); e != kInvalidElementId;) {
      const Element* parent = element_tree_.GetElement(e);
      if (!parent) break;
      ++depth;
      e = parent->GetParentId();
    }
    if (!best || depth > best_depth) {
      best = &rp;
      best_leaf = leaf;
      best_depth = depth;
    }
  }
  if (best && leaf_out) *leaf_out = best_leaf;
  return best;
}

void Space::ApplyAnchorChartLocked(const RemoteRelationUpdate& update, const std::string& name, bool is_root,
                                   AnchorProposal* ap) const {
  ap->has_chart = false;
  ap->chart_verified = false;
  for (const auto& c : update.anchor_charts) {
    const bool match = is_root ? (c.anchor_name.empty() || c.anchor_name == "root" || c.anchor_name == name)
                               : (!name.empty() && c.anchor_name == name);
    if (!match) continue;
    const Element* counterpart = GetReferentResolver().Resolve(c.scope_shared_id, element_tree_);
    if (!counterpart || !counterpart->HasContainmentTransfer()) return;
    // The counterpart's element frame is the leaf's chart on this side (the transfer's own mapping: congruent copies
    // under a rigid transform are authored in corresponding element frames).
    const Eigen::Matrix4d world = element_tree_.GetWorldTransform(counterpart->GetId());
    const Eigen::Vector4d p = world * Eigen::Vector4d(c.x, c.y, c.z, 1.0);
    ap->has_chart = true;
    ap->chart_leaf_shared_id = c.scope_shared_id;
    ap->chart_x = p.x();
    ap->chart_y = p.y();
    ap->chart_z = p.z();
    ap->chart_yaw = std::remainder(WorldYawOf(world) + c.yaw, 2.0 * M_PI);
    // Pitch and roll: the anchor's own pass-through, as its proposal carries it (a leaf turns about the vertical only).
    ap->chart_orientation_type = ap->orientation_type;
    ap->chart_pitch = ap->pitch;
    ap->chart_roll = ap->roll;
    // Verified as the root's transfer flag is: the source's footprint of the leaf rode this update (a relation to the
    // leaf carries it) and is congruent with the counterpart's.
    // Declared shapes on both sides: the scope extension never enters the chart.
    const SpatialDefinition* def = counterpart->GetDeclaredSpatialDefinition();
    const std::shared_ptr<Polygon2D> mine = def ? def->GetBoundary() : nullptr;
    for (const auto& rp : update.payloads)
      if (mine && rp.referent_shared_id == c.scope_shared_id && rp.support.boundary) {
        ap->chart_verified = FootprintsCongruent(*rp.support.boundary, *mine);
        break;
      }
    return;
  }
}

bool Space::LeafRelativeYawLocked(const std::vector<RelationPayload>& payloads, const std::vector<double>& weights,
                                  double* yaw) const {
  const Element* leaf = nullptr;
  const RelationPayload* rp = ContainmentRelationLocked(payloads, weights, &leaf);
  if (!rp || !leaf || rp->grounded_relation.encoded_state.size() < 3) return false;
  *yaw = std::remainder(WorldYawOf(element_tree_.GetWorldTransform(leaf->GetId())) + rp->grounded_relation.encoded_state(2),
                        2.0 * M_PI);
  return true;
}

// ---- the cause bits: one rule for every anchor ----------------------------------

Space::AnchorCauseFacts Space::SolveCauseFacts(const SolverResult& r, const std::vector<RelationPayload>& payloads,
                                               const std::vector<double>& weights) {
  AnchorCauseFacts f;
  f.solved = true;
  double driving_weight = -1.0;
  // A Facing whose look-at point lies inside the receiver's body clearance is not in the yaw
  // (SolverGoalWitness::yaw_excluded): it leaves the referent set, and re-enters it, as a relation does.
  std::unordered_set<size_t> yaw_excluded;
  for (const auto& g : r.goals)
    if (g.yaw_excluded) yaw_excluded.insert(g.payload_index);
  for (size_t k = 0; k < payloads.size(); ++k) {
    const RelationPayload& rp = payloads[k];
    const double w = k < weights.size() ? weights[k] : 1.0;
    if (!rp.enabled || w <= 0.0) continue;
    if (yaw_excluded.count(k)) continue;
    f.signature.emplace_back(rp.referent_shared_id, static_cast<int>(rp.kernel_type));
    if (w > driving_weight) {
      driving_weight = w;
      f.driving_referent = rp.referent_shared_id;
      f.driving_kernel = static_cast<int>(rp.kernel_type);
    }
  }
  std::sort(f.signature.begin(), f.signature.end());
  // The position goals of this solve (the inputs of S0), keyed like `signature`, and its yaw inputs.
  for (const auto& g : r.goals) {
    if (!g.position_goal || g.payload_index >= payloads.size()) continue;
    const RelationPayload& rp = payloads[g.payload_index];
    f.goals.push_back({rp.referent_shared_id, static_cast<int>(rp.kernel_type), g.witness, g.weight, g.decode_branch});
  }
  for (const auto& g : r.goals) {
    if (!g.yaw_goal || g.payload_index >= payloads.size()) continue;
    const RelationPayload& rp = payloads[g.payload_index];
    f.yaw_goals.push_back({rp.referent_shared_id, static_cast<int>(rp.kernel_type), g.weight, g.yaw_conditioning,
                           g.yaw_bearing, rp.support.look_at_vertex, rp.support.look_at_boundary_crossed});
  }
  f.witnesses = r.goals;
  f.facing_sign = r.facing_sign;
  f.facing_tie = r.facing_tie;
  f.position = r.position;
  f.clamp_branch = r.clamp_branch;
  f.clamp_exempt_hash = r.clamp_exempt_hash;
  f.progress = !r.ratio_intents.empty();
  f.chart_snap = r.chart_snap;
  f.no_request = r.no_request;
  return f;
}

uint32_t Space::AttributeAnchorCauses(AnchorCauseState* state, const AnchorCauseFacts& now, uint64_t geometry_revision,
                                      bool commit) {
  AnchorCauseState& prev = *state;
  uint32_t cause = 0;
  if (!prev.has_prev) cause |= kCauseFirstProposal;
  // The driver's own discontinuity of this anchor's sample: a new epoch, or a teleport not yet attributed.
  if (prev.has_prev && now.sampled &&
      (prev.sample_epoch != now.sample_epoch || (now.teleported && prev.teleport_sequence != now.sample_sequence)))
    cause |= kCauseEmbodiedReset;
  if (prev.has_prev) {
    // Both are reported when both happened (a referent-set change that also moves the driving relation is two causes).
    if (now.signature != prev.signature) cause |= kCauseReferentSetChange;
    if (now.driving_referent != prev.driving_referent || now.driving_kernel != prev.driving_kernel ||
        now.driving_tag != prev.driving_tag)
      cause |= kCauseWeightSwitch;
    // WEIGHT_SWITCH is also raised when the softmax weights moved S0 by more than the sources' own motion did:
    // with the referent set unchanged, a weight change that adds motion beyond the points' own is a cause.
    if (now.signature == prev.signature &&
        (WeightsAddedS0Motion(prev.goals, now.goals) || WeightsAddedYawMotion(prev.yaw_goals, now.yaw_goals)))
      cause |= kCauseWeightSwitch;
    // A referent that drives S0 (weight > 0 in both solves) decoded its approach point by another rule: the decode is
    // continuous inside one rule, not across.
    for (const auto& c : now.goals)
      for (const auto& p : prev.goals)
        if (p.referent == c.referent && p.kernel == c.kernel && p.weight > 0.0 && c.weight > 0.0 &&
            p.decode_branch != c.decode_branch)
          cause |= kCauseDecodeBranch;
    if (now.stuck != prev.stuck) cause |= kCauseStatusChange;
    // The receiver's own geometry/structure edits move the answer without the source moving.
    if (geometry_revision != prev.geometry_revision) cause |= kCauseSceneChange;
    if (now.engaged && !prev.engaged) cause |= kCauseSeatEngage;
    if (!now.engaged && prev.engaged) cause |= kCauseSeatRelease;
    if (now.contained && !prev.contained) cause |= kCauseContainEnter;
    if (!now.contained && prev.contained) cause |= kCauseContainExit;
    // The contained frame itself changed: the previous frame's containment ended and the next one began.
    if (now.contained && prev.contained && now.contain_scope != prev.contain_scope)
      cause |= kCauseContainEnter | kCauseContainExit;
    // The clamp of S0 moved from one constraint branch to another (both clamped).
    if (prev.clamp_branch != 0 && now.clamp_branch != 0 && prev.clamp_branch != now.clamp_branch)
      cause |= kCauseClampBranch;
    // The free space itself changed: the source engaged or released a referent whose footprint the anchor may then
    // stand in.
    if (prev.clamp_exempt_hash != now.clamp_exempt_hash) cause |= kCauseClampBranch;
    if (prev.facing_sign != 0 && now.facing_sign != 0 && prev.facing_sign != now.facing_sign)
      cause |= kCauseFacingRootSwitch;
  }
  // A tie of the Facing bearings the arithmetic cannot resolve: the update that enters or leaves the tie differs from
  // its neighbour by the tie alone.
  if (now.facing_tie || (prev.has_prev && prev.facing_tie)) cause |= kCauseFacingTie;
  // A pass through a look-at point: the bearing to a point has a pole at the point; a path that comes within the input
  // data's resolution (float32, relation/support.h) of it has passed through it as far as the data knows.
  if (prev.has_prev && prev.has_position && now.solved) {
    const Eigen::Vector2d a = prev.position, b = now.position;
    const double e_in = static_cast<double>(std::numeric_limits<float>::epsilon()) *
                        std::max(1.0, std::max(std::fabs(b.x()), std::fabs(b.y())));
    for (const auto& g : now.witnesses) {
      if (!g.yaw_goal || g.yaw_excluded || !(g.yaw_conditioning > 0.0) || !(g.weight > 0.0)) continue;
      const Eigen::Vector2d ab = b - a;
      const double len2 = ab.squaredNorm();
      const double t = len2 > 0.0 ? std::clamp((g.witness - a).dot(ab) / len2, 0.0, 1.0) : 0.0;
      if ((g.witness - (a + t * ab)).norm() <= e_in) {
        cause |= kCauseFacingTie;
        break;
      }
    }
  }
  // The least-angle look-at rule's own discontinuities (a different silhouette
  // vertex in the vertex regime, a pass through the boundary), declared as the sign-root switch.
  if (prev.has_prev) {
    for (const auto& c : now.yaw_goals)
      for (const auto& p : prev.yaw_goals)
        if (p.referent == c.referent && p.kernel == c.kernel && p.weight > 0.0 && c.weight > 0.0 &&
            (c.boundary_crossed ||
             (p.look_at_vertex >= 0 && c.look_at_vertex >= 0 && p.look_at_vertex != c.look_at_vertex)))
          cause |= kCauseFacingRootSwitch;
  }
  if (now.exception) cause |= kCauseSourceException;
  // A finite source count is a ratio intent: the bit only says the intent was admitted.
  if (now.progress) cause |= kCauseProgress;
  if (now.chart_snap) cause |= kCauseChartSnap;
  if (now.no_request) cause |= kCauseNoRequest;
  if (commit) {
    prev.has_prev = true;
    if (now.sampled) {
      prev.sample_epoch = now.sample_epoch;
      if (now.teleported) prev.teleport_sequence = now.sample_sequence;
    }
    prev.signature = now.signature;
    prev.driving_referent = now.driving_referent;
    prev.driving_kernel = now.driving_kernel;
    prev.driving_tag = now.driving_tag;
    prev.stuck = now.stuck;
    prev.engaged = now.engaged;
    prev.contained = now.contained;
    prev.contain_scope = now.contained ? now.contain_scope : kInvalidSharedElementId;
    prev.geometry_revision = geometry_revision;
    prev.facing_sign = now.facing_sign;
    prev.facing_tie = now.facing_tie;
    prev.position = now.position;
    prev.has_position = now.solved;
    prev.clamp_branch = now.clamp_branch;
    prev.clamp_exempt_hash = now.clamp_exempt_hash;
    prev.goals = now.goals;
    prev.yaw_goals = now.yaw_goals;
  }
  return cause;
}

}  // namespace spatialblend

