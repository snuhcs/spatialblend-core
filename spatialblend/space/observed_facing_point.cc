// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <atomic>
#include <limits>

#include "spatialblend/space/space.h"

namespace spatialblend {
namespace {

uint64_t IssueObservedFacingLease() {
  // Space IDs can be reused while a peer still retains an old counterpart.
  // This capability is process-local and must not repeat across Space objects.
  static std::atomic<uint64_t> last{0};
  uint64_t current = last.load(std::memory_order_relaxed);
  while (current != std::numeric_limits<uint64_t>::max()) {
    if (last.compare_exchange_weak(current, current + 1,
                                   std::memory_order_relaxed))
      return current + 1;
  }
  return 0;
}

}  // namespace

bool Space::BeginObservedFacingPoint(
    element_id_t id, spatialblend::entity_id_t producer, const std::string& anchor,
    ObservedFacingPointKind kind, int64_t maximum_age_ns, uint64_t* lease,
    element_handle_t source_point, uint64_t source_point_lease) {
  if (!lease) return false;
  *lease = 0;
  std::lock_guard<std::mutex> lock(space_mutex_);
  const auto* element = element_tree_.GetElement(id);
  const auto* actor = FindActorLocked(producer);
  const auto* owner =
      actor ? element_tree_.GetElement(actor->GetElementId()) : nullptr;
  const auto old = observed_facing_points_.find(id);
  if (scene_inputs_escaped_.load() || maximum_age_ns <= 0 ||
      maximum_age_ns > ReceiverInputState::kMaxAgeNs || !element || !owner ||
      !actor->GetAnchor(anchor) || anchor.empty() ||
      owner->GetSharedId() == kInvalidSharedElementId ||
      element->GetSpatialDefinition() || !element->GetChildIds().empty() ||
      !element->GetAnchors().empty() ||
      (old != observed_facing_points_.end() && !old->second.retired))
    return false;
  if (kind != ObservedFacingPointKind::kSourceAnchor &&
      kind != ObservedFacingPointKind::kReceiverCounterpart)
    return false;
  const bool source = observation_clocks_.count(producer) != 0;
  if (source != (kind == ObservedFacingPointKind::kSourceAnchor)) return false;
  if (source) {
    if (space_id_ == kInvalidSpaceId || source_point_lease != 0 ||
        !(source_point == element_handle_t{}))
      return false;
  } else if (source_point.space_id == kInvalidSpaceId ||
             source_point.element_id == kInvalidElementId ||
             source_point_lease == 0) {
    return false;
  }
  // No generated Actor transform may move this referent through ancestry.
  for (element_id_t ancestor = id; ancestor != kInvalidElementId;) {
    for (const auto& [actor_id, other] : actors_)
      if (other->GetElementId() == ancestor) return false;
    const auto* parent = element_tree_.GetElement(ancestor);
    ancestor = parent ? parent->GetParentId() : kInvalidElementId;
  }
  ObservedFacingPoint value;
  value.kind = kind;
  value.producer = producer;
  value.producer_element = owner->GetId();
  value.logical_actor = owner->GetSharedId();
  value.referent_event_id = element->GetSharedId() == kInvalidSharedElementId
                                ? id
                                : element->GetSharedId();
  value.anchor = anchor;
  value.lease = IssueObservedFacingLease();
  if (value.lease == 0) return false;
  value.maximum_age_ns = maximum_age_ns;
  value.source_point = source ? element_handle_t{space_id_, id} : source_point;
  value.source_point_lease = source ? value.lease : source_point_lease;
  *lease = value.lease;
  observed_facing_points_[id] = std::move(value);
  AdvanceSceneInputsLocked(true, true);
  return true;
}

bool Space::BeginObservedSourceSetup(
    const std::vector<ObservedSourceSetupParticipant>& participants,
    const std::vector<ObservedSourceSetupPoint>& points,
    std::vector<ObservedSourceSetupActorResult>* actor_results,
    std::vector<ObservedSourceSetupPointResult>* point_results) {
  if (!actor_results || !point_results || participants.empty() || points.empty())
    return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (scene_inputs_escaped_.load()) return false;

  std::unordered_set<spatialblend::entity_id_t> actor_ids;
  std::unordered_map<spatialblend::entity_id_t, const ObservedSourceSetupParticipant*>
      participant_by_actor;
  std::unordered_map<spatialblend::entity_id_t, uint64_t> next_epochs;
  for (const auto& participant : participants) {
    Actor* actor = FindActorLocked(participant.actor);
    const SubjectAnchor* root =
        actor ? actor->GetAnchor(participant.root_anchor) : nullptr;
    const bool horizontal =
        root && std::find(root->dof.begin(), root->dof.end(),
                          StateType::kPosition2D) != root->dof.end() &&
        std::find(root->dof.begin(), root->dof.end(),
                  StateType::kPosition3D) == root->dof.end();
    const auto clock = observation_clocks_.find(participant.actor);
    const auto setup_epoch =
        observed_source_setup_epochs_.find(participant.actor);
    if (!actor || participant.root_anchor.empty() ||
        !actor_ids.insert(participant.actor).second || !root || !horizontal ||
        !root->parent.empty() || !root->local_offset.isZero(0.0) ||
        // Fresh setup and declared legacy transition are intentionally
        // distinct.  A setup-owned epoch is never a replaceable prior state.
        (participant.expected_prior_epoch == 0
             ? clock != observation_clocks_.end()
             : clock == observation_clocks_.end() ||
                   clock->second.epoch != participant.expected_prior_epoch ||
                   setup_epoch != observed_source_setup_epochs_.end()))
      return false;
    // An individually opened source point is also observed-source state, even when its
    // epoch came from the legacy begin API.  Do not replace that epoch behind
    // an unknown point lease; its owner must retire it explicitly first.
    for (const auto& [element, point] : observed_facing_points_)
      if (point.producer == participant.actor && !point.retired) return false;
    const auto fence = retired_observation_epoch_fences_.find(participant.actor);
    const uint64_t prior = fence == retired_observation_epoch_fences_.end()
                               ? 0
                               : fence->second;
    const uint64_t current =
        clock == observation_clocks_.end() ? 0 : clock->second.epoch;
    const uint64_t previous = std::max(prior, current);
    if (previous == std::numeric_limits<uint64_t>::max()) return false;
    participant_by_actor.emplace(participant.actor, &participant);
    next_epochs.emplace(participant.actor, previous + 1);
  }

  std::unordered_set<element_id_t> element_ids;
  std::vector<ObservedFacingPoint> staged_points;
  staged_points.reserve(points.size());
  for (const auto& request : points) {
    const auto participant = participant_by_actor.find(request.producer);
    const auto* element = element_tree_.GetElement(request.element);
    const auto* actor = FindActorLocked(request.producer);
    const auto* owner =
        actor ? element_tree_.GetElement(actor->GetElementId()) : nullptr;
    const auto old = observed_facing_points_.find(request.element);
    if (participant == participant_by_actor.end() || !element || !actor ||
        !owner || request.anchor.empty() || !actor->GetAnchor(request.anchor) ||
        !element_ids.insert(request.element).second ||
        request.maximum_age_ns <= 0 ||
        request.maximum_age_ns > ReceiverInputState::kMaxAgeNs ||
        owner->GetSharedId() == kInvalidSharedElementId ||
        element->GetSpatialDefinition() || !element->GetChildIds().empty() ||
        !element->GetAnchors().empty() ||
        (old != observed_facing_points_.end() && !old->second.retired))
      return false;
    // A point referent cannot be actor-owned, including through ancestry.
    for (element_id_t ancestor = request.element;
         ancestor != kInvalidElementId;) {
      for (const auto& [actor_id, other] : actors_)
        if (other->GetElementId() == ancestor) return false;
      const auto* parent = element_tree_.GetElement(ancestor);
      ancestor = parent ? parent->GetParentId() : kInvalidElementId;
    }
    ObservedFacingPoint value;
    value.kind = ObservedFacingPointKind::kSourceAnchor;
    value.producer = request.producer;
    value.producer_element = owner->GetId();
    value.logical_actor = owner->GetSharedId();
    value.referent_event_id = element->GetSharedId() == kInvalidSharedElementId
                                  ? request.element
                                  : element->GetSharedId();
    value.anchor = request.anchor;
    value.maximum_age_ns = request.maximum_age_ns;
    value.source_point = element_handle_t{space_id_, request.element};
    staged_points.push_back(std::move(value));
  }

  // Reserve every process-local capability before touching Space state.  A
  // lease exhaustion is therefore a pre-mutation rejection, never a prefix.
  for (auto& value : staged_points) {
    value.lease = IssueObservedFacingLease();
    if (value.lease == 0) return false;
    value.source_point_lease = value.lease;
  }

  // All rejection paths are above.  The following mutations install a fresh,
  // unsampled epoch and inactive lease for every member as one source setup.
  std::vector<ObservedSourceSetupActorResult> staged_actor_results;
  std::vector<ObservedSourceSetupPointResult> staged_point_results;
  staged_actor_results.reserve(participants.size());
  staged_point_results.reserve(points.size());
  for (const auto& participant : participants) {
    ResetObservationStateLocked(participant.actor);
    const uint64_t epoch = next_epochs.at(participant.actor);
    ObservationClock& clock = observation_clocks_[participant.actor];
    clock.epoch = epoch;
    clock.sequence = -1;
    clock.time = 0.0;
    clock.receipt_monotonic_ns = 0;
    clock.anchors_from_committed_frame = false;
    clock.has_sample = false;
    clock.root = participant.root_anchor;
    observed_source_setup_epochs_[participant.actor] = epoch;
    staged_actor_results.push_back({participant.actor, epoch});
  }
  for (size_t index = 0; index < staged_points.size(); ++index) {
    auto& value = staged_points[index];
    observed_facing_points_[points[index].element] = value;
    staged_point_results.push_back({points[index].element, value.lease});
  }
  AdvanceSceneInputsLocked(true, true);
  *actor_results = std::move(staged_actor_results);
  *point_results = std::move(staged_point_results);
  return true;
}

bool Space::RetireObservedSourceSetup(
    const std::vector<ObservedSourceSetupActorResult>& actors,
    const std::vector<ObservedSourceSetupPointResult>& points) {
  if (actors.empty() || points.empty()) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  std::unordered_set<spatialblend::entity_id_t> actor_ids;
  std::unordered_set<element_id_t> point_ids;
  // First prove that cancellation would not erase admitted evidence.  No
  // state changes occur until all members pass this exact capability check.
  for (const auto& actor_result : actors) {
    const auto clock = observation_clocks_.find(actor_result.actor);
    if (!actor_ids.insert(actor_result.actor).second ||
        clock == observation_clocks_.end() ||
        clock->second.epoch != actor_result.epoch || clock->second.has_sample ||
        clock->second.sequence != -1)
      return false;
  }
  for (const auto& point_result : points) {
    const auto point = observed_facing_points_.find(point_result.element);
    if (!point_ids.insert(point_result.element).second ||
        point == observed_facing_points_.end() || point->second.retired ||
        point->second.lease != point_result.lease || point->second.active ||
        point->second.revision != 0)
      return false;
  }
  for (const auto& actor_result : actors) {
    ResetObservationStateLocked(actor_result.actor);
    observation_clocks_.erase(actor_result.actor);
    retired_observation_epoch_fences_[actor_result.actor] = actor_result.epoch;
    observed_source_setup_epochs_.erase(actor_result.actor);
  }
  for (const auto& point_result : points) {
    auto& point = observed_facing_points_.at(point_result.element);
    point.active = false;
    point.retired = true;
  }
  AdvanceSceneInputsLocked(true, true);
  return true;
}

Space::ObservedSourceSetupReleaseState
Space::GetObservedSourceSetupReleaseState(
    const std::vector<ObservedSourceSetupActorResult>& actors,
    const std::vector<ObservedSourceSetupPointResult>& points) const {
  if (actors.empty() || points.empty())
    return ObservedSourceSetupReleaseState::kTerminal;
  std::lock_guard<std::mutex> lock(space_mutex_);
  std::unordered_set<spatialblend::entity_id_t> actor_ids;
  std::unordered_set<element_id_t> point_ids;
  bool admitted = false;
  for (const auto& actor_result : actors) {
    const auto clock = observation_clocks_.find(actor_result.actor);
    const auto setup_epoch =
        observed_source_setup_epochs_.find(actor_result.actor);
    if (!actor_ids.insert(actor_result.actor).second ||
        clock == observation_clocks_.end() ||
        setup_epoch == observed_source_setup_epochs_.end() ||
        clock->second.epoch != actor_result.epoch ||
        setup_epoch->second != actor_result.epoch)
      return ObservedSourceSetupReleaseState::kTerminal;
    admitted = admitted || clock->second.has_sample ||
               clock->second.sequence != -1;
  }
  for (const auto& point_result : points) {
    const auto point = observed_facing_points_.find(point_result.element);
    if (!point_ids.insert(point_result.element).second ||
        point == observed_facing_points_.end() || point->second.retired ||
        point->second.lease != point_result.lease)
      return ObservedSourceSetupReleaseState::kTerminal;
    admitted = admitted || point->second.active || point->second.revision != 0;
  }
  return admitted ? ObservedSourceSetupReleaseState::kAdmitted
                  : ObservedSourceSetupReleaseState::kUnadmitted;
}

bool Space::IsObservedFacingPointLiveLocked(element_id_t id,
                                            int64_t now) const {
  const auto found = observed_facing_points_.find(id);
  if (found == observed_facing_points_.end()) return false;
  const auto& point = found->second;
  const auto* actor = FindActorLocked(point.producer);
  const auto* element = element_tree_.GetElement(id);
  const auto* owner =
      actor ? element_tree_.GetElement(actor->GetElementId()) : nullptr;
  if (scene_inputs_escaped_.load() || !point.active || point.retired ||
      now < point.observed_at_ns || now >= point.valid_until_ns || !element ||
      !owner || owner->GetId() != point.producer_element ||
      owner->GetSharedId() != point.logical_actor ||
      element->GetSpatialDefinition() || !element->GetChildIds().empty() ||
      !element->GetAnchors().empty() ||
      element_tree_.GetWorldTransform(id) != point.committed_world)
    return false;
  const auto clock = observation_clocks_.find(point.producer);
  if (point.kind == ObservedFacingPointKind::kReceiverCounterpart)
    return clock == observation_clocks_.end();
  if (!(point.source_point == element_handle_t{space_id_, id})) return false;
  const auto* pose = actor->GetAnchorWorldPoseOverride(point.anchor);
  return clock != observation_clocks_.end() && clock->second.has_sample &&
         clock->second.anchors_from_committed_frame &&
         clock->second.epoch == point.source_epoch &&
         clock->second.sequence == point.source_sequence &&
         actor->IsAnchorActive(point.anchor) && pose &&
         pose->position == point.point;
}

bool Space::IsObservedFacingCandidateLocked(element_id_t id,
                                            spatialblend::entity_id_t actor_id,
                                            RelationKernelType kernel,
                                            int64_t now) const {
  const auto point = observed_facing_points_.find(id);
  if (point == observed_facing_points_.end()) return true;
  const auto* actor = FindActorLocked(actor_id);
  const auto* owner =
      actor ? element_tree_.GetElement(actor->GetElementId()) : nullptr;
  return kernel == RelationKernelType::kAvatarFacing && owner &&
         owner->GetSharedId() != kInvalidSharedElementId &&
         owner->GetSharedId() != point->second.logical_actor &&
         IsObservedFacingPointLiveLocked(id, now);
}

bool Space::CommitObservedFacingPoint(element_id_t id, uint64_t lease,
                                      uint64_t source_epoch,
                                      int64_t source_sequence,
                                      const Eigen::Vector3d& world_point,
                                      int64_t observed_at_ns,
                                      int64_t valid_until_ns) {
  return CommitObservedFacingPoints(
      {{id, lease, source_epoch, source_sequence, world_point, observed_at_ns,
        valid_until_ns}});
}

bool Space::PrepareObservedFacingPointLocked(
    const ObservedFacingPointUpdate& update, int64_t now,
    PreparedObservedFacingPoint* prepared,
    const ActorObservationFrame* prospective_frame,
    int64_t admission_receipt) const {
  const auto id = update.element;
  const auto lease = update.lease;
  const auto source_epoch = update.source_epoch;
  const auto source_sequence = update.source_sequence;
  const auto& world_point = update.world_point;
  const auto observed_at_ns = update.observed_at_ns;
  const auto valid_until_ns = update.valid_until_ns;
  auto found = observed_facing_points_.find(id);
  if (found == observed_facing_points_.end() || lease == 0 ||
      found->second.lease != lease || found->second.retired ||
      scene_inputs_escaped_.load() || !world_point.allFinite() ||
      source_epoch == 0 || source_sequence < 0 || observed_at_ns < 0 ||
      observed_at_ns > now || valid_until_ns <= now ||
      valid_until_ns <= observed_at_ns)
    return false;
  const auto& point = found->second;
  const auto* element = element_tree_.GetElement(id);
  const auto* actor = FindActorLocked(point.producer);
  const auto* owner =
      actor ? element_tree_.GetElement(actor->GetElementId()) : nullptr;
  if (!element || !owner || owner->GetId() != point.producer_element ||
      owner->GetSharedId() != point.logical_actor ||
      valid_until_ns - observed_at_ns > point.maximum_age_ns ||
      observed_at_ns < point.observed_at_ns ||
      source_epoch < point.source_epoch ||
      (source_epoch == point.source_epoch &&
       source_sequence <= point.source_sequence) ||
      point.revision == std::numeric_limits<uint64_t>::max())
    return false;
  const auto clock = observation_clocks_.find(point.producer);
  if (point.kind == ObservedFacingPointKind::kSourceAnchor) {
    if (!(point.source_point == element_handle_t{space_id_, id})) return false;
    if (prospective_frame) {
      // The caller already validated the entire frame. Value and original
      // clock are still checked here, before any batch member is admitted.
      if (prospective_frame->actor != point.producer ||
          prospective_frame->epoch != source_epoch ||
          prospective_frame->sequence != source_sequence ||
          observed_at_ns > admission_receipt ||
          clock == observation_clocks_.end())
        return false;
      const auto observation =
          std::find_if(prospective_frame->observations.begin(),
                       prospective_frame->observations.end(),
                       [&](const ActorObservation& row) {
                         return row.anchor_name == point.anchor;
                       });
      if (observation == prospective_frame->observations.end()) return false;
      Eigen::Vector3d position = observation->position;
      if (point.anchor == clock->second.root) position.z() = 0.0;
      if (position != world_point) return false;
    } else {
      const auto* pose = actor->GetAnchorWorldPoseOverride(point.anchor);
      if (clock == observation_clocks_.end() || !clock->second.has_sample ||
          !clock->second.anchors_from_committed_frame ||
          clock->second.epoch != source_epoch ||
          clock->second.sequence != source_sequence ||
          !actor->IsAnchorActive(point.anchor) || !pose ||
          pose->position != world_point ||
          observed_at_ns > clock->second.receipt_monotonic_ns)
        return false;
    }
  } else if (clock != observation_clocks_.end()) {
    return false;
  }
  Eigen::Matrix4d world = Eigen::Matrix4d::Identity();
  world.block<3, 1>(0, 3) = world_point;
  const element_id_t parent_id = element->GetParentId();
  const Eigen::Matrix4d parent =
      parent_id == kInvalidElementId
          ? Eigen::Matrix4d::Identity()
          : element_tree_.GetWorldTransform(parent_id);
  const Eigen::Matrix4d local = parent.inverse() * world;
  if (!local.allFinite()) return false;
  prepared->update = update;
  prepared->local_transform = local;
  prepared->continuous =
      IsObservedFacingPointLiveLocked(id, now) &&
      point.source_epoch == source_epoch &&
      point.source_sequence < std::numeric_limits<int64_t>::max() &&
      source_sequence == point.source_sequence + 1;
  return true;
}

void Space::CommitObservedFacingPointLocked(
    const PreparedObservedFacingPoint& prepared) {
  const auto& update = prepared.update;
  auto& point = observed_facing_points_.at(update.element);
  element_tree_.GetElement(update.element)
      ->SetLocalTransform(prepared.local_transform);
  point.committed_world = element_tree_.GetWorldTransform(update.element);
  point.point = update.world_point;
  point.source_epoch = update.source_epoch;
  point.source_sequence = update.source_sequence;
  point.observed_at_ns = update.observed_at_ns;
  point.valid_until_ns = update.valid_until_ns;
  point.active = true;
  ++point.revision;
  AdvanceSceneInputsLocked(true, true);
  InvalidateObservedFacingConsumersLocked(update.element, prepared.continuous);
}

bool Space::CommitObservedFacingPoints(
    const std::vector<ObservedFacingPointUpdate>& updates) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  const int64_t now = receiver_input_clock_();
  if (updates.empty()) return false;
  std::unordered_set<element_id_t> elements;
  std::vector<PreparedObservedFacingPoint> prepared(updates.size());
  for (size_t i = 0; i < updates.size(); ++i) {
    if (!elements.insert(updates[i].element).second ||
        !PrepareObservedFacingPointLocked(updates[i], now, &prepared[i]))
      return false;
  }
  for (const auto& point : prepared) CommitObservedFacingPointLocked(point);
  // No actor input, generated state, phase/sample clock or relation delivery.
  return true;
}

bool Space::ApplyActorObservationBatch(
    const std::vector<ActorObservationFrame>& frames,
    const std::vector<ObservedFacingPointPublication>& publications) {
  // Waiting for the solve lock counts against every original receipt window.
  const int64_t receipt = receiver_input_clock_();
  return ApplyActorObservationBatchWithReceipt(frames, publications, receipt);
}

bool Space::ApplyActorObservationBatchWithReceipt(
    std::vector<ActorObservationFrame> frames,
    std::vector<ObservedFacingPointPublication> publications,
    int64_t receipt) {
  // Own the rows across delivery callbacks. Capture receipt in the public
  // entry, so neither these defensive copies nor lock waiting renew input.
  std::vector<PendingPropagation> pending;
  std::unique_lock<std::mutex> lock(space_mutex_);
  const int64_t now = receiver_input_clock_();
  if (frames.empty()) return false;
  std::unordered_map<spatialblend::entity_id_t, const ActorObservationFrame*>
      inputs;
  std::unordered_set<element_id_t> owners;
  for (const auto& frame : frames) {
    if (!inputs.emplace(frame.actor, &frame).second ||
        !ValidateActorObservationFrameLocked(frame))
      return false;
    const auto owner = FindActorLocked(frame.actor)->GetElementId();
    // Two subjects must not overwrite the same leaf pose between evaluations.
    if (owner != kInvalidElementId && !owners.insert(owner).second)
      return false;
  }
  std::unordered_set<element_id_t> elements;
  std::vector<PreparedObservedFacingPoint> points;
  for (const auto& publication : publications) {
    const auto point = observed_facing_points_.find(publication.element);
    if (!elements.insert(publication.element).second ||
        point == observed_facing_points_.end() ||
        point->second.kind != ObservedFacingPointKind::kSourceAnchor)
      return false;
    const auto frame = inputs.find(point->second.producer);
    if (frame == inputs.end()) return false;
    const auto& rows = frame->second->observations;
    const auto row = std::find_if(
        rows.begin(), rows.end(), [&](const ActorObservation& input) {
          return input.anchor_name == point->second.anchor;
        });
    if (row == rows.end()) return false;
    Eigen::Vector3d position = row->position;
    if (point->second.anchor == observation_clocks_.at(frame->first).root)
      position.z() = 0.0;
    PreparedObservedFacingPoint prepared;
    if (!PrepareObservedFacingPointLocked(
            {publication.element, publication.lease, frame->second->epoch,
             frame->second->sequence, position, publication.observed_at_ns,
             publication.valid_until_ns},
            now, &prepared, frame->second, receipt))
      return false;
    points.push_back(std::move(prepared));
  }

  // No mutating call or callback occurred before this point. Continuity was
  // sampled against the old live input, not the half-admitted new frame.
  // An opt-in starts a source-only phase lifetime. Legacy/generated/remote
  // writes break that lifetime; they cannot warm Contact's hysteresis for the
  // next genuine source frame. Reset only after the ENTIRE batch validates,
  // without callbacks or clock/sample advancement during member commits.
  for (const auto& frame : frames) {
    if (source_evidence_phase_dirty_.erase(frame.actor)) {
      phase_tracker_.ClearOwner(PhaseOwner(frame.actor));
      source_relations_.erase(frame.actor);
    }
  }
  std::vector<PreparedActorObservation> admitted;
  admitted.reserve(frames.size());
  for (const auto& frame : frames)
    admitted.push_back(CommitActorObservationFrameLocked(frame, receipt));
  for (const auto& point : points) CommitObservedFacingPointLocked(point);
  // Omitted publications cannot keep a producer's superseded anchor alive.
  PruneObservedFacingPointsLocked();
  for (size_t i = 0; i < frames.size(); ++i) {
    EvaluateActorStateLocked(frames[i].actor, admitted[i].position,
                             admitted[i].yaw, true, frames[i].time_seconds,
                             admitted[i].elapsed, &pending, true);
  }
  // Freeze after every member has been evaluated, under the same source lock.
  // A later read copies these owned values; it never re-queries old geometry.
  for (const auto& frame : frames) CaptureSourceInteractionEvidenceLocked(frame.actor);
  // Same handoff as the other entry points: take the delivery guard while the
  // state guard is still held, release the state guard, then dispatch the
  // batch in the order its sequence numbers were stamped.
  std::unique_lock<std::mutex> delivery(propagation_mutex_);
  lock.unlock();
  DeliverPropagations(pending);
  return true;
}

bool Space::AreObservedFacingDependenciesLiveLocked(
    spatialblend::entity_id_t actor, int64_t now) const {
  const auto dependencies = observed_facing_dependencies_.find(actor);
  if (dependencies == observed_facing_dependencies_.end()) return true;
  for (const auto& dependency : dependencies->second) {
    const auto point = observed_facing_points_.find(dependency.element);
    if (point == observed_facing_points_.end() ||
        point->second.lease != dependency.lease ||
        point->second.revision != dependency.revision ||
        now >= dependency.valid_until_ns ||
        !IsObservedFacingPointLiveLocked(dependency.element, now))
      return false;
  }
  return true;
}

void Space::InvalidateObservedFacingConsumersLocked(element_id_t id,
                                                    bool preserve_history) {
  const auto point = observed_facing_points_.find(id);
  const int64_t now = receiver_input_clock_();
  for (auto it = observed_facing_dependencies_.begin();
       it != observed_facing_dependencies_.end();) {
    const bool used = std::any_of(it->second.begin(), it->second.end(),
                                  [id](const ObservedFacingDependency& value) {
                                    return value.element == id;
                                  });
    if (!used) {
      ++it;
      continue;
    }
    const bool continuous =
        preserve_history && point != observed_facing_points_.end() &&
        std::all_of(
            it->second.begin(), it->second.end(),
            [now](const auto& dep) { return now < dep.valid_until_ns; });
    // A receiver-body refresh must not reuse old subject intent against this
    // changed peer. Preserve only history, until a fresh source delivery.
    retained_receiver_evaluations_.erase(it->first);
    ClearReceiverTargetsLocked(it->first, continuous);
    if (continuous) {
      // Keep phase identity, not the old actionable solve. The old source
      // deadline is never extended by a newly received counterpart point.
      for (auto& dependency : it->second) {
        if (dependency.element != id) continue;
        dependency.revision = point->second.revision;
        dependency.valid_until_ns =
            std::min(dependency.valid_until_ns, point->second.valid_until_ns);
      }
      ++it;
    } else {
      for (const auto& dependency : it->second)
        phase_tracker_.ClearRelationForOwner(dependency.phase_identity,
                                             PhaseOwner(it->first));
      it = observed_facing_dependencies_.erase(it);
    }
  }
  for (auto& [actor, bindings] : bindings_) {
    for (auto& binding : bindings) {
      const auto point = observed_facing_points_.find(id);
      if (!preserve_history && point != observed_facing_points_.end()) {
        RelationEvent identity;
        identity.actor_shared_id = GetActorSharedIdForEvents(actor);
        identity.anchor_name = binding.anchor_name;
        identity.referent_shared_id = point->second.referent_event_id;
        identity.kernel_type = binding.kernel_type;
        phase_tracker_.ClearRelationForOwner(identity, PhaseOwner(actor));
      }
      binding.latest_matches.erase(
          std::remove_if(binding.latest_matches.begin(),
                         binding.latest_matches.end(),
                         [id](const Binding::MatchResult& m) {
                           return m.element_id == id;
                         }),
          binding.latest_matches.end());
      if (!preserve_history) {
        binding.sent_relations.erase(id);
        binding.smoothed_progress_rate.erase(id);
        binding.last_progress_sample_seconds.erase(id);
        binding.has_progress_sample.erase(id);
      }
    }
  }
}

bool Space::ClearObservedFacingPoint(element_id_t id, uint64_t lease) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto found = observed_facing_points_.find(id);
  if (found == observed_facing_points_.end() || lease == 0 ||
      found->second.lease != lease || found->second.retired)
    return false;
  if (found->second.active) {
    found->second.active = false;
    AdvanceSceneInputsLocked(true, true);
    InvalidateObservedFacingConsumersLocked(id);
  }
  return true;
}

bool Space::RetireObservedFacingPoint(element_id_t id, uint64_t lease) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto found = observed_facing_points_.find(id);
  if (found == observed_facing_points_.end() || lease == 0 ||
      found->second.lease != lease)
    return false;
  if (!found->second.retired) {
    found->second.active = false;
    found->second.retired = true;
    AdvanceSceneInputsLocked(true, true);
    InvalidateObservedFacingConsumersLocked(id);
  }
  return true;
}

void Space::PruneObservedFacingPointsLocked() {
  const int64_t now = receiver_input_clock_();
  for (auto& [id, point] : observed_facing_points_) {
    if (!point.active || IsObservedFacingPointLiveLocked(id, now)) continue;
    point.active = false;
    AdvanceSceneInputsLocked(true, true);
    InvalidateObservedFacingConsumersLocked(id);
  }
  // The source-side receipt may expire before the independently live local
  // counterpart. Its original deadline must not be refreshed on receipt/solve.
  for (auto it = observed_facing_dependencies_.begin();
       it != observed_facing_dependencies_.end();) {
    if (AreObservedFacingDependenciesLiveLocked(it->first, now)) {
      ++it;
      continue;
    }
    for (const auto& dependency : it->second)
      phase_tracker_.ClearRelationForOwner(dependency.phase_identity,
                                           PhaseOwner(it->first));
    ClearReceiverTargetsLocked(it->first);
    AdvanceSceneInputsLocked(true, true);
    it = observed_facing_dependencies_.erase(it);
  }
}

void Space::PruneObservedFacingPoints() {
  std::lock_guard<std::mutex> lock(space_mutex_);
  PruneObservedFacingPointsLocked();
}

void Space::WithdrawObservedFacingPointsBelowLocked(element_id_t ancestor) {
  for (element_id_t id : element_tree_.GetSubtreeIds(ancestor)) {
    auto point = observed_facing_points_.find(id);
    if (point == observed_facing_points_.end() || !point->second.active)
      continue;
    point->second.active = false;
    AdvanceSceneInputsLocked(true, true);
    InvalidateObservedFacingConsumersLocked(id);
  }
}

}  // namespace spatialblend
