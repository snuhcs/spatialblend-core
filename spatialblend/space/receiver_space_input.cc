// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <algorithm>

#include "spatialblend/space/space.h"

namespace spatialblend {

ReceiverInputStamp Space::CurrentReceiverInputLocked(
    spatialblend::entity_id_t actor_id) const {
  const auto it = receiver_inputs_.find(actor_id);
  if (it == receiver_inputs_.end()) return {};
  auto stamp = it->second.Stamp(receiver_input_clock_());
  if (scene_inputs_escaped_.load()) stamp.known = false;
  return stamp;
}

void Space::ClearReceiverTargetsLocked(spatialblend::entity_id_t actor_id,
                                       bool preserve_history,
                                       bool publish_released) {
  realization_results_[actor_id] = {};
  receiver_solve_inputs_.erase(actor_id);
  // Retain the historical rich snapshot for a nonmutating invalidation query.
  if (!preserve_history) {
    proposal_attribution_.erase(actor_id);
    previous_weights_.erase(actor_id);
  }
  if (!publish_released) return;
  ReceiverProposal released;
  released.status = ProposalStatus::kNone;
  PublishProposalLocked(actor_id, std::move(released));
}

void Space::InvalidateReceiverInputLocked(spatialblend::entity_id_t actor_id) {
  retained_receiver_evaluations_.erase(actor_id);
  const auto it = receiver_inputs_.find(actor_id);
  if (it == receiver_inputs_.end()) return;
  it->second.Invalidate();
  ClearReceiverTargetsLocked(actor_id);
}

bool Space::BeginReceiverInputEpoch(spatialblend::entity_id_t actor_id,
                                    const std::string& root_anchor,
                                    uint64_t rig_revision,
                                    uint64_t calibration_revision,
                                    uint64_t* epoch) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  const auto* actor = FindActorLocked(actor_id);
  if (!actor || !element_tree_.GetElement(actor->GetElementId()) ||
      scene_inputs_escaped_.load() || observation_clocks_.count(actor_id) ||
      (next_propagate_sequence_.count(actor_id) &&
       next_propagate_sequence_.at(actor_id) > 0))
    return false;
  const auto* root = actor->GetAnchor(root_anchor);
  // The port is owned by the receiver-input stream from the START of Begin: a
  // direct publish that races this call is refused, not interleaved with the
  // stream's frames. Restored when Begin does not go through.
  const auto mailbox = MailboxFor(actor_id);
  const bool was_input_owned = mailbox && mailbox->input_owned.exchange(true);
  const auto disown = [&] {
    if (mailbox && !was_input_owned) mailbox->input_owned.store(false);
  };
  if (!root || !root->parent.empty() || !root->local_offset.isZero(0.0) ||
      std::find(root->dof.begin(), root->dof.end(), StateType::kPosition2D) ==
          root->dof.end() ||
      std::find(root->dof.begin(), root->dof.end(), StateType::kPosition3D) !=
          root->dof.end()) {
    disown();
    return false;
  }
  std::vector<std::string> layout;
  for (const auto& anchor : actor->GetAnchors()) layout.push_back(anchor.name);
  const auto old = receiver_inputs_.find(actor_id);
  if (old != receiver_inputs_.end()) {
    if (!old->second.Begin(root_anchor, layout, rig_revision,
                           calibration_revision, epoch)) {
      disown();
      return false;
    }
  } else {
    ReceiverInputState next;
    if (!next.Begin(root_anchor, layout, rig_revision, calibration_revision,
                    epoch)) {
      disown();
      return false;
    }
    receiver_inputs_[actor_id] = std::move(next);
  }
  retained_receiver_evaluations_.erase(actor_id);
  // The stream takes the port: whatever an earlier direct writer left (its
  // epoch, its sequence) must not make the stream's frames stale, so the port
  // starts empty. The first frame republishes the sample.
  if (mailbox) mailbox->port.Reset();
  ClearReceiverTargetsLocked(actor_id);
  return true;
}

bool Space::ApplyReceiverInputFrame(
    spatialblend::entity_id_t actor_id, uint64_t epoch, uint64_t sequence,
    int64_t observed_at_ns, int64_t valid_until_ns,
    const std::vector<ReceiverAnchorInput>& anchors) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  const auto it = receiver_inputs_.find(actor_id);
  if (it == receiver_inputs_.end() || scene_inputs_escaped_.load())
    return false;
  if (!it->second.Commit(epoch, sequence, observed_at_ns, valid_until_ns,
                         anchors, receiver_input_clock_()))
    return false;
  if (!CurrentReceiverInputLocked(actor_id).known)
    ClearReceiverTargetsLocked(actor_id);
  // The committed frame IS the driver's embodied pose report: its root anchor
  // is forwarded into the actor's sampling port (the single place the solve
  // reads the receiver pose from). The rig's other anchors stay in the frame.
  Eigen::Matrix4d root_pose;
  double root_yaw = 0.0;
  if (it->second.CopyPose(it->second.root_anchor(), &root_pose) &&
      ReceiverPoseYaw(root_pose, &root_yaw)) {
    EmbodiedPoseInput pose;
    pose.x = root_pose(0, 3);
    pose.y = root_pose(1, 3);
    pose.yaw = root_yaw;
    pose.epoch = epoch;
    // A refusal is surfaced: the frame is not a usable pose report then.
    if (PublishReceiverInputPose(actor_id, pose, observed_at_ns * 1e-9,
                                 sequence) != EmbodiedPublishResult::kOk)
      return false;
  }
  // No source clock or relation propagation.
  return true;
}

bool Space::ClearReceiverInputFrame(spatialblend::entity_id_t actor_id,
                                    uint64_t epoch) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  const auto it = receiver_inputs_.find(actor_id);
  if (it == receiver_inputs_.end() || !it->second.Clear(epoch)) return false;
  ClearReceiverTargetsLocked(actor_id);
  return true;
}

Space::ReceiverReevaluationResult Space::ReevaluateReceiverRealization(
    spatialblend::entity_id_t actor_id, uint64_t expected_epoch,
    uint64_t expected_sequence) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!receiver_inputs_.count(actor_id) ||
      (!rich_realization_enabled_ && !rich_realization_actors_.count(actor_id)))
    return ReceiverReevaluationResult::kUnavailable;
  const auto input = CurrentReceiverInputLocked(actor_id);
  if (expected_epoch == 0 || expected_sequence == 0 ||
      input.epoch != expected_epoch || input.sequence != expected_sequence)
    return ReceiverReevaluationResult::kStaleInput;
  if (!input.known) return ReceiverReevaluationResult::kUnavailable;
  const auto retained = retained_receiver_evaluations_.find(actor_id);
  if (retained == retained_receiver_evaluations_.end())
    return ReceiverReevaluationResult::kUnavailable;
  const auto& last = retained->second.last_evaluated_input;
  if (last.known && last.epoch == input.epoch &&
      last.revision == input.revision && last.sequence == input.sequence)
    return ReceiverReevaluationResult::kNoNewInput;
  const auto previous = realization_snapshots_.find(actor_id);
  const uint64_t previous_revision =
      previous == realization_snapshots_.end() ? 0 : previous->second.revision;
  // Own the packet across possible retainer invalidation inside evaluation.
  const auto packet = retained->second.update;
  const auto ingress = retained->second.original_ingress_ns;
  EvaluateRemoteRelationLocked(packet, actor_id, ingress, true);
  const auto current = realization_snapshots_.find(actor_id);
  return current != realization_snapshots_.end() &&
                 current->second.revision > previous_revision
             ? ReceiverReevaluationResult::kEvaluated
             : ReceiverReevaluationResult::kUnavailable;
}

bool Space::GetActorReceiverInputValidity(
    spatialblend::entity_id_t actor_id, uint64_t realization_revision,
    ActorReceiverInputValidity* out) const {
  if (!out) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  const auto snapshot = realization_snapshots_.find(actor_id);
  if (snapshot == realization_snapshots_.end() ||
      snapshot->second.revision != realization_revision)
    return false;
  ActorReceiverInputValidity value;
  value.revision = realization_revision;
  value.captured = snapshot->second.receiver_input;
  value.current = CurrentReceiverInputLocked(actor_id);
  value.changed = value.captured.epoch != value.current.epoch ||
                  value.captured.revision != value.current.revision;
  value.valid = value.captured.known && value.current.known && !value.changed;
  *out = value;
  return true;
}

}  // namespace spatialblend
