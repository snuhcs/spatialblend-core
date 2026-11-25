// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/anchor_proposals.cc
//
// ABI 4: the per-anchor embodied
// ports, the pose proposal of every relation-bearing anchor other than the root,
// the actor event queue (source exceptions, teleport requests) and the
// interaction latency timers. The root anchor's solve stays in
// Space::EvaluateRemoteRelationLocked.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>

#include "spatialblend/solver/affordance_pose_sd.h"
#include "spatialblend/solver/engagement.h"
#include "spatialblend/solver/scope_pursuit.h"
#include "spatialblend/space/space.h"

namespace spatialblend {

namespace {

int64_t SteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

// ---- anchors and their ports ------------------------------------------------

uint32_t Space::InternAnchor(ActorMailbox& box, const std::string& name,
                             const std::string& root_name) {
  std::lock_guard<std::mutex> lock(box.anchors_mutex);
  box.anchor_names[0] = root_name;
  if (name.empty() || name == root_name) return kRootAnchorId;
  for (size_t i = 1; i < box.anchor_names.size(); ++i)
    if (box.anchor_names[i] == name) return static_cast<uint32_t>(i);
  box.anchor_names.push_back(name);
  box.anchor_ports.push_back(nullptr);
  return static_cast<uint32_t>(box.anchor_names.size() - 1);
}

// The actor's root anchor: the one its own state carries (the receiver input names it; an actor of this space carries
// it as its "root" anchor, or, when it has none of that name, as its first anchor).
std::string Space::RootAnchorNameLocked(spatialblend::entity_id_t actor_id) const {
  const auto it = receiver_inputs_.find(actor_id);
  if (it != receiver_inputs_.end()) return it->second.root_anchor();
  if (const Actor* actor = FindActorLocked(actor_id)) {
    const auto& anchors = actor->GetAnchors();
    if (!anchors.empty() && !actor->GetAnchor("root")) return anchors.front().name;
  }
  return std::string("root");
}

bool Space::AnchorIdFor(spatialblend::entity_id_t actor_id, const std::string& name,
                        uint32_t* id) {
  if (!id) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!FindActorLocked(actor_id)) return false;
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox) return false;
  *id = InternAnchor(*mailbox, name, RootAnchorNameLocked(actor_id));
  return true;
}

bool Space::AnchorNameOf(spatialblend::entity_id_t actor_id, uint32_t anchor_id,
                         std::string* name) const {
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox || !name) return false;
  std::lock_guard<std::mutex> lock(mailbox->anchors_mutex);
  if (anchor_id >= mailbox->anchor_names.size()) return false;
  *name = mailbox->anchor_names[anchor_id];
  return true;
}

EmbodiedPublishResult Space::PublishAnchorPose(
    spatialblend::entity_id_t actor_id, uint32_t anchor_id,
    const EmbodiedPoseInput& pose, double timestamp_s, uint64_t sequence) {
  // Like the root's port: space_mutex_ is never taken, and nothing waits for a
  // solve. The driver is each port's only writer.
  if (anchor_id == kRootAnchorId)
    return PublishEmbodiedPose(actor_id, pose, timestamp_s, sequence);
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox) return EmbodiedPublishResult::kInvalid;
  if (mailbox->input_owned.load()) return EmbodiedPublishResult::kRefused;
  EmbodiedPort* port = nullptr;
  {
    std::lock_guard<std::mutex> lock(mailbox->anchors_mutex);
    if (anchor_id >= mailbox->anchor_names.size())
      return EmbodiedPublishResult::kInvalid;
    // ABI 4.7: a registered, bone-bound anchor is derived from its bone; the bone's port is the single writer.
    if (mailbox->embodiment) {
      const EmbodimentAnchor* declared = mailbox->embodiment->FindAnchor(mailbox->anchor_names[anchor_id]);
      if (declared && declared->bone >= 0) return EmbodiedPublishResult::kRefused;
    }
    if (!mailbox->anchor_ports[anchor_id])
      mailbox->anchor_ports[anchor_id] = std::make_unique<EmbodiedPort>();
    port = mailbox->anchor_ports[anchor_id].get();
  }
  return port->Publish(pose, timestamp_s, sequence, SteadyNowNs());
}

bool Space::SampleAnchorPose(spatialblend::entity_id_t actor_id, uint32_t anchor_id,
                             EmbodiedSample* out) const {
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox || !out) return false;
  if (anchor_id == kRootAnchorId) {
    *out = mailbox->port.Sample();
    return true;
  }
  const EmbodiedPort* port = nullptr;
  const EmbodimentAnchor* declared = nullptr;
  const BonePort* bone_port = nullptr;
  std::shared_ptr<const Embodiment> embodiment;
  {
    std::lock_guard<std::mutex> lock(mailbox->anchors_mutex);
    if (anchor_id >= mailbox->anchor_names.size()) return false;
    if (mailbox->embodiment) {
      declared = mailbox->embodiment->FindAnchor(mailbox->anchor_names[anchor_id]);
      if (declared && declared->bone >= 0) {
        embodiment = mailbox->embodiment;  // keeps `declared` alive
        bone_port = mailbox->bone_ports[declared->bone].get();
      }
    }
    if (!bone_port) port = mailbox->anchor_ports[anchor_id].get();
  }
  if (bone_port) {
    // ABI 4.7: the anchor is its bone's latest sample composed with the registered offset. Nothing is stored,
    // integrated or substituted: no bone sample is no anchor sample.
    const BoneSample bone = bone_port->Sample();
    *out = EmbodiedSample{};
    if (!bone.valid) return true;
    RigidTransform bone_pose;
    bone_pose.position = Eigen::Vector3d(bone.p[0], bone.p[1], bone.p[2]);
    bone_pose.rotation = Eigen::Quaterniond(bone.q[3], bone.q[0], bone.q[1], bone.q[2]).normalized();
    const RigidTransform anchor = ComposeAnchor(*declared, bone_pose);
    const EulerZyx euler = EulerFromRotation(anchor.rotation);
    out->valid = true;
    out->x = anchor.position.x();
    out->y = anchor.position.y();
    out->z = anchor.position.z();
    out->yaw = euler.yaw;
    out->pitch = euler.pitch;
    out->roll = euler.roll;
    out->orientation_type = 2;  // SB_ORIENTATION_YAW_PITCH_ROLL
    out->timestamp_s = bone.timestamp_s;
    out->sequence = bone.sequence;
    out->epoch = bone.epoch;
    out->flags = bone.flags;
    out->published_ns = bone.published_ns;
    return true;
  }
  *out = port ? port->Sample() : EmbodiedSample{};
  return true;
}

// ---- embodiment registration (ABI 4.7) ---------------------------------------

Space::RegisterEmbodimentResult Space::RegisterEmbodiment(spatialblend::entity_id_t actor_id,
                                                          const Embodiment& embodiment) {
  return RegisterEmbodiment(actor_id, embodiment, nullptr);
}

Space::RegisterEmbodimentResult Space::RegisterEmbodiment(spatialblend::entity_id_t actor_id,
                                                          const Embodiment& embodiment,
                                                          std::string* conflict) {
  if (!embodiment.Validate(nullptr)) return RegisterEmbodimentResult::kInvalid;
  std::vector<SubjectAnchor> missing;
  {
    std::lock_guard<std::mutex> lock(space_mutex_);
    const Actor* actor = FindActorLocked(actor_id);
    if (!actor) return RegisterEmbodimentResult::kInvalid;
    const auto mailbox = MailboxFor(actor_id);
    if (!mailbox) return RegisterEmbodimentResult::kInvalid;
    if (mailbox->input_owned.load()) return RegisterEmbodimentResult::kRefused;
    // Declared families are enforced at bind; a binding made BEFORE the registration must not be able to violate
    // them silently, so the registration is refused with the first conflict named (order-independent outcome).
    if (const auto bound = bindings_.find(actor_id); bound != bindings_.end()) {
      for (const Binding& b : bound->second) {
        const EmbodimentAnchor* declared = embodiment.FindAnchor(b.anchor_name);
        if (!declared) continue;
        const uint32_t bit = b.kernel_type == RelationKernelType::kAvatarSpacing ? kFamilySpacing
                             : b.kernel_type == RelationKernelType::kAvatarFacing ? kFamilyFacing
                                                                                  : kFamilyAffordancePose;
        if (!(declared->families & bit)) {
          if (conflict)
            *conflict = b.anchor_name + "@" +
                        (bit == kFamilySpacing ? "Spacing" : bit == kFamilyFacing ? "Facing" : "AffordancePose");
          return RegisterEmbodimentResult::kBindingConflict;
        }
      }
    }
    {
      std::lock_guard<std::mutex> alock(mailbox->anchors_mutex);
      if (mailbox->embodiment) return RegisterEmbodimentResult::kRefused;
      // A bone-bound anchor that already has a driver-published port would have two writers.
      for (const EmbodimentAnchor& a : embodiment.anchors) {
        if (a.bone < 0) continue;
        for (size_t i = 1; i < mailbox->anchor_names.size(); ++i)
          if (mailbox->anchor_names[i] == a.name && mailbox->anchor_ports[i])
            return RegisterEmbodimentResult::kRefused;
      }
      mailbox->embodiment = std::make_shared<const Embodiment>(embodiment);
      if (embodiment.body.present) {
        // Per actor: this actor's free space is eroded by its own embodiment's radius.
        actor_body_radius_[actor_id] = embodiment.body.floor_footprint_radius_m;
      }
      mailbox->bone_ports.clear();
      for (size_t i = 0; i < embodiment.joints.size(); ++i)
        mailbox->bone_ports.push_back(std::make_unique<BonePort>());
    }
    // The registered anchors exist as subject anchors too (as sb_add_anchor would add them: legacy default
    // degrees of freedom, no fixed offset -- their world pose always arrives from a bone), so bindings and
    // the observation path see them. An anchor the app already added is kept.
    for (const EmbodimentAnchor& a : embodiment.anchors)
      if (!actor->GetAnchor(a.name)) missing.emplace_back(a.name);
  }
  if (!missing.empty()) AddActorAnchors(actor_id, missing, /*reject_duplicates=*/false);
  return RegisterEmbodimentResult::kOk;
}

std::shared_ptr<const Embodiment> Space::EmbodimentOf(spatialblend::entity_id_t actor_id) const {
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox) return nullptr;
  std::lock_guard<std::mutex> lock(mailbox->anchors_mutex);
  return mailbox->embodiment;
}

EmbodiedPublishResult Space::PublishBonePose(spatialblend::entity_id_t actor_id, int bone_id,
                                             const BonePoseInput& pose, double timestamp_s,
                                             uint64_t sequence) {
  // Like the anchor ports: space_mutex_ is never taken, and nothing waits for a solve.
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox) return EmbodiedPublishResult::kInvalid;
  if (mailbox->input_owned.load()) return EmbodiedPublishResult::kRefused;
  BonePort* port = nullptr;
  {
    std::lock_guard<std::mutex> lock(mailbox->anchors_mutex);
    if (bone_id < 0 || static_cast<size_t>(bone_id) >= mailbox->bone_ports.size())
      return EmbodiedPublishResult::kInvalid;
    port = mailbox->bone_ports[bone_id].get();
  }
  // The caller's rotation is normalised once, here; a zero-length quaternion is not a rotation.
  BonePoseInput in = pose;
  Eigen::Quaterniond q(in.q[3], in.q[0], in.q[1], in.q[2]);
  if (!std::isfinite(q.norm()) || q.norm() == 0.0) return EmbodiedPublishResult::kInvalid;
  q.normalize();
  in.q[0] = q.x();
  in.q[1] = q.y();
  in.q[2] = q.z();
  in.q[3] = q.w();
  return port->Publish(in, timestamp_s, sequence, SteadyNowNs());
}

bool Space::SampleBonePose(spatialblend::entity_id_t actor_id, int bone_id, BoneSample* out) const {
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox || !out) return false;
  const BonePort* port = nullptr;
  {
    std::lock_guard<std::mutex> lock(mailbox->anchors_mutex);
    if (bone_id < 0 || static_cast<size_t>(bone_id) >= mailbox->bone_ports.size()) return false;
    port = mailbox->bone_ports[bone_id].get();
  }
  *out = port->Sample();
  return true;
}

bool Space::ComposeBoneObservations(spatialblend::entity_id_t actor_id,
                                    const std::vector<std::pair<std::string, RigidTransform>>& bones,
                                    std::vector<ActorObservation>* out) const {
  if (!out) return false;
  const std::shared_ptr<const Embodiment> e = EmbodimentOf(actor_id);
  if (!e) return false;
  std::vector<const RigidTransform*> by_bone(e->joints.size(), nullptr);
  for (const auto& named : bones) {
    const int id = e->BoneId(named.first);
    if (id < 0 || by_bone[id]) return false;  // an unknown or repeated bone
    by_bone[id] = &named.second;
  }
  for (const EmbodimentAnchor& a : e->anchors) {
    if (a.bone < 0 || !by_bone[a.bone]) continue;
    const RigidTransform anchor = ComposeAnchor(a, *by_bone[a.bone]);
    const EulerZyx euler = EulerFromRotation(anchor.rotation);
    ActorObservation o{a.name, anchor.position, euler.yaw, /*from_bones=*/true};
    o.orientation_type = 2;  // SB_ORIENTATION_YAW_PITCH_ROLL: the bone carries the full rotation
    o.pitch = euler.pitch;
    o.roll = euler.roll;
    out->push_back(std::move(o));
  }
  return true;
}

bool Space::SampleAnchorLocked(spatialblend::entity_id_t actor_id,
                               const std::string& name,
                               const std::string& root_name,
                               EmbodiedSample* out) const {
  *out = EmbodiedSample{};
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox) return false;
  if (const auto receiver = receiver_inputs_.find(actor_id);
      receiver != receiver_inputs_.end()) {
    // An external receiver's anchors are sampled from its stamped input frame.
    Eigen::Matrix4d pose;
    double yaw = 0.0;
    const auto stamp = CurrentReceiverInputLocked(actor_id);
    if (!stamp.known ||
        !receiver->second.CopyPose(name.empty() ? root_name : name, &pose) ||
        !ReceiverPoseYaw(pose, &yaw))
      return false;
    out->valid = true;
    out->x = pose(0, 3);
    out->y = pose(1, 3);
    out->z = pose(2, 3);
    out->yaw = yaw;
    out->epoch = stamp.epoch;
    out->sequence = stamp.sequence;
    return true;
  }
  const uint32_t id = InternAnchor(*mailbox, name, root_name);
  if (!SampleAnchorPose(actor_id, id, out)) return false;
  return out->valid;
}

// ---- events -----------------------------------------------------------------

void Space::AppendActorEvents(spatialblend::entity_id_t actor_id,
                              std::vector<ActorEvent> events) {
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox || events.empty()) return;
  const auto ordinary_motion = [](const ActorEvent& e) {
    return (e.kind == ActorEvent::Kind::kPositionRequest ||
            e.kind == ActorEvent::Kind::kYawRequest) &&
           !(e.request_flags & 1u);
  };
  std::lock_guard<std::mutex> lock(mailbox->events_mutex);
  // The whole revision becomes visible together. One consumer drains without a
  // budget, grouping until a revision boundary or a true empty queue. No
  // producer may append more events with this revision after this unlocks.
  for (auto& event : events) {
    event.target_delivery = mailbox->target_events_enabled.load();
    mailbox->events.erase(
        std::remove_if(mailbox->events.begin(), mailbox->events.end(),
                       [&](const ActorEvent& old) {
                         if (!ordinary_motion(old)) return false;
                         if (event.kind == ActorEvent::Kind::kSourceException)
                           return true;
                         if (event.anchor_id != old.anchor_id) return false;
                         if (event.kind == ActorEvent::Kind::kTeleportRequest ||
                             (event.request_flags & 1u))
                           return true;
                         if (event.kind == ActorEvent::Kind::kTargetRelease)
                           return (event.channels & old.channels) != 0;
                         return ordinary_motion(event) &&
                                old.kind == event.kind;
                       }),
        mailbox->events.end());
    mailbox->events.push_back(std::move(event));
  }
}

bool Space::SetTargetEventDelivery(spatialblend::entity_id_t actor_id,
                                   bool enabled) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox) return false;
  {
    std::lock_guard<std::mutex> events_lock(mailbox->events_mutex);
    if (mailbox->target_events_enabled.load() == enabled)
      return true;  // no reset on repeated enable
    // A historical legacy exception has no immutable pose batch. Never
    // substitute today's snapshot or discard it.
    if (enabled &&
        std::any_of(mailbox->events.begin(), mailbox->events.end(),
                    [](const ActorEvent& event) {
                      return event.kind == ActorEvent::Kind::kSourceException &&
                             !event.target_delivery;
                    }))
      return false;
    mailbox->target_events_enabled.store(enabled);
  }
  // Disabling changes future publication only. Already queued lifecycle groups
  // and releases must survive.
  if (!enabled) return true;
  // A subscriber (including a reloaded driver) starts at a cancellation
  // boundary, not at an old sample-relative turn. The next source solve is a
  // fresh placement, with a new immutable origin. Existing lifecycle events
  // stay ordered.
  ReceiverProposal previous;
  mailbox->proposal.Load(&previous);
  previous.revision = ++proposal_revision_;
  std::vector<ActorEvent> events;
  {
    // The latest snapshot can omit an anchor withdrawn while disabled. Its
    // queued lifecycle poses or the consumer's latched goal still need release.
    // Anchor IDs are interned for the actor's lifetime. Do not nest this lock
    // with the event lock or hold it across publication.
    std::lock_guard<std::mutex> anchors_lock(mailbox->anchors_mutex);
    for (size_t id = 0; id < mailbox->anchor_names.size(); ++id) {
      ActorEvent release;
      release.kind = ActorEvent::Kind::kTargetRelease;
      release.revision = previous.revision;
      release.anchor_id = static_cast<uint32_t>(id);
      release.channels = 3;
      events.push_back(std::move(release));
    }
  }
  mailbox->proposal.Store(
      std::make_shared<const ReceiverProposal>(std::move(previous)));
  AppendActorEvents(actor_id, std::move(events));
  proposal_attribution_.erase(actor_id);
  return true;
}

bool Space::PollActorEvent(spatialblend::entity_id_t actor_id, ActorEvent* out) {
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox || !out) return false;
  std::lock_guard<std::mutex> lock(mailbox->events_mutex);
  if (mailbox->events.empty()) return false;
  *out = std::move(mailbox->events.front());
  mailbox->events.pop_front();
  return true;
}

int Space::PollActorEventAfterSolve(spatialblend::entity_id_t actor_id,
                                    ActorEvent* out) {
  // This synchronization is part of the supported C polling contract, not an
  // incidental HasActor side effect. Ordinary request production is
  // solve-locked and uses the current port sample; none is deferred/replayed
  // later.
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!out || !FindActorLocked(actor_id)) return -1;
  return PollActorEvent(actor_id, out) ? 1 : 0;
}

bool Space::RaiseSourceException(spatialblend::entity_id_t actor_id,
                                 uint32_t exception_id, const void* payload,
                                 size_t payload_size) {
  if (payload_size > kMaxExceptionPayloadBytes) return false;
  if (payload_size > 0 && !payload) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (!FindActorLocked(actor_id)) return false;
  PendingException pending;
  pending.id = exception_id;
  if (payload_size > 0) {
    const auto* bytes = static_cast<const uint8_t*>(payload);
    pending.payload.assign(bytes, bytes + payload_size);
  }
  pending_source_exception_[actor_id] = std::move(pending);
  return true;
}

void Space::SetInteractionLatencyPolicy(const InteractionLatencyPolicy& policy) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  latency_policy_ = policy;
}

InteractionLatencyPolicy Space::GetInteractionLatencyPolicy() const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return latency_policy_;
}

// ---- the proposals of the other relation-bearing anchors ---------------------

void Space::BuildAnchorProposalsLocked(
    spatialblend::entity_id_t actor_id, const RemoteRelationUpdate& update,
    const std::vector<RelationPayload>& payloads,
    const std::vector<double>& weights, const RealizationResult& realization,
    const NavigableSpatialDefinition* navigable_def,
    const std::string& root_name, bool receiver_reevaluation, bool placed,
    uint32_t root_cause, ReceiverProposal* set) {
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox) return;
  const auto weight_at = [&](size_t i) {
    return i < weights.size() ? weights[i] : 1.0;
  };
  // A relation the anchor answers: weighed by the policy, or without a valid count, which is NO REQUEST (v1: no plan,
  // the entity stays where it is) and not a disabled relation: Progressive computes no weight for it, and
  // the anchor's solve holds at its sample. Only a relation weighed 0 although its count is valid (a
  // Greedy loser) is left out.
  const auto answered = [&](size_t i) {
    return weight_at(i) > 0.0 || !RatioRequest(payloads[i].update_count);
  };
  // A source anchor named "root" (or unnamed) is the receiver actor's root, whatever its root anchor is called, unless
  // a receiver input names its root (an anonymous relation never aliases a named receiver root).
  const bool named_root = receiver_inputs_.count(actor_id) != 0;
  const auto is_root_name = [&](const std::string& name) {
    return name == root_name || (!named_root && (name.empty() || name == "root"));
  };

  // Which anchors bear a relation in this update (other than the root).
  std::map<uint32_t, std::string> anchors;
  const auto add = [&](const std::string& name) {
    if (is_root_name(name) || name.empty()) return;  // an unnamed anchor is the root's id; it is never another anchor
    anchors[InternAnchor(*mailbox, name, root_name)] = name;
  };
  for (const auto& t : realization.targets)
    if (t.kernel_type == RelationKernelType::kAvatarAffordancePose)
      add(t.anchor_name);
  for (const auto& name : realization.awaiting_engagement) add(name);
  for (size_t i = 0; i < payloads.size(); ++i) {
    const auto& p = payloads[i];
    if (((p.kernel_type == RelationKernelType::kAvatarFacing ||
          p.kernel_type == RelationKernelType::kAvatarSpacing ||
          p.kernel_type == RelationKernelType::kAvatarAffordancePose) && p.enabled && answered(i)) ||
        (IsInnerPosePayload(p) && p.enabled))  // an inner channel whose weights are all 0 is still answered (no request)
      add(p.subject_anchor_name);
  }
  ProposalAttribution& attr = proposal_attribution_[actor_id];
  const double eps_m = solver_.config().objective_improve_epsilon_m;
  TargetEvaluationOptions eval_options;
  eval_options.resolver = &GetReferentResolver();

  std::map<uint32_t, AnchorCauseState> next_states;  // this solve's anchors (an anchor absent from it starts afresh)
  for (const auto& [id, name] : anchors) {
    AnchorProposal ap;
    ap.anchor_id = id;
    // The relation solve of this anchor's Spacing / Facing relations, when they were solved (the causes read it).
    std::optional<SolverResult> anchor_solve;
    std::vector<RelationPayload> anchor_group;
    std::vector<double> anchor_group_weights;
    ap.anchor_name = name;
    EmbodiedSample sample;
    const bool sampled = SampleAnchorLocked(actor_id, name, root_name, &sample);
    if (sampled) ap.flags |= kAnchorFlagSampled;
    ap.request_sample = sample;

    // The anchor's relations.
    std::vector<size_t> ap_idx, facing_idx, spacing_idx, inner;
    std::vector<std::pair<shared_element_id_t, int>> signature;
    for (size_t i = 0; i < payloads.size(); ++i) {
      const auto& p = payloads[i];
      if ((is_root_name(p.subject_anchor_name) ? root_name : p.subject_anchor_name) != name || !p.enabled) continue;
      // The inner relations are kept whatever their weight: all-zero weights are the inner channel's "no request".
      if (IsInnerPosePayload(p)) inner.push_back(i);
      if (!answered(i)) continue;
      if (p.kernel_type == RelationKernelType::kAvatarAffordancePose)
        ap_idx.push_back(i);
      else if (p.kernel_type == RelationKernelType::kAvatarFacing)
        facing_idx.push_back(i);
      else if (p.kernel_type == RelationKernelType::kAvatarSpacing)
        spacing_idx.push_back(i);
      else
        continue;
      signature.emplace_back(p.referent_shared_id,
                             static_cast<int>(p.kernel_type));
    }
    std::sort(signature.begin(), signature.end());

    const RealizationTarget* engaged_target = nullptr;
    for (const auto& t : realization.targets)
      if (t.kernel_type == RelationKernelType::kAvatarAffordancePose &&
          t.anchor_name == name && t.engaged_support) {
        engaged_target = &t;
        break;
      }

    if (engaged_target) {
      // The pelvis (or hand) anchor TOUCHES its surface at the source: the
      // proposal is the mapped contact, the seat pose. The driver / IK realizes it.
      ap.flags |= kAnchorFlagEngaged;
      ap.x = engaged_target->position.x();
      ap.y = engaged_target->position.y();
      ap.z = engaged_target->has_z ? engaged_target->z : 0.0;
      ap.yaw = engaged_target->yaw;
      if (sampled) ap.yaw_turn = std::remainder(ap.yaw - sample.yaw, 2.0 * M_PI);  // ABI 4.12: the pose's shortest arc
      ap.referent_shared_id = engaged_target->referent_id;
      // The target IS the mapped contact: an exact minimiser. Arrival is the flag.
      ap.status = ProposalStatus::kExact;
      // The engaged payload: the one the target was grounded on.
      for (size_t i : ap_idx) {
        const auto& p = payloads[i];
        if (p.referent_shared_id != engaged_target->referent_id ||
            !IsEngaged(p))
          continue;
        const PreparedTargetRelation prepared =
            PrepareTargetRelation(p, element_tree_, eval_options);
        if (!prepared.ok()) break;
        const auto et = EvaluatePreparedTargetRelation(
            prepared,
            TargetAnchorPose{Eigen::Vector3d(ap.x, ap.y, ap.z), ap.yaw},
            element_tree_);
        if (et.ok())
          ap.err_pose_target = std::fabs(et.r_tgt - et.r_src);
        if (sampled) {
          const auto es = EvaluatePreparedTargetRelation(
              prepared,
              TargetAnchorPose{Eigen::Vector3d(sample.x, sample.y, sample.z),
                               sample.yaw},
              element_tree_);
          if (es.ok()) {
            ap.err_pose_sample = std::fabs(es.r_tgt - es.r_src);
            // Arrived: the SAMPLED anchor touches the receiver's counterpart
            // surface (the same geometric fact that engaged the source).
            if (es.anchor_touches_support) ap.flags |= kAnchorFlagArrived;
          }
        }
        break;
      }
    } else if (!inner.empty()) {
      // The INNER level: the source anchor is inside its object's SD but not in contact. Its relations
      // are pursued by the one live path of every anchor (scope_pursuit.h, the root's), in the scope
      // the poses are defined in, the SD's region volume V:
      //   * each pose relation is a term w (d(x) - d*)^2, d the geodesic over V (the straight segment: V is convex) to
      //     its MAPPED CONTACT c (the source's contact read through the pose's chart on THIS side's instance: its part
      //     chart, or the standing point over the proposed root), as the root's terms measure to their mapped contacts;
      //   * d* = RatioTargetDistance(d(x), n), x = the anchor's CURRENT position, its sample in this update (as
      //     v1);
      //   * one relation walks min(L/n, L) of its course, several are minimised over V (the root's rules). The chart rule
      //     for the free component is not applied here, as it is not on the root.
      // No term with a valid count is no request: the solve answers at the anchor's current position (v1), a position
      // with NO_REQUEST, and the published anchor keeps its retained pending target.
      ap.flags |= kAnchorFlagInner;
      // One SD: the object of the heaviest inner relation (the first when none has weight).
      size_t heaviest = inner.front();
      for (size_t i : inner)
        if (weight_at(i) > weight_at(heaviest)) heaviest = i;
      const RelationPayload& hp = payloads[heaviest];
      const auto object_of = [&](const RelationPayload& p) -> const Element* {
        return p.resolved_local_element_id != kInvalidElementId
                   ? element_tree_.GetElement(p.resolved_local_element_id)
                   : GetReferentResolver().Resolve(p.referent_shared_id, element_tree_);
      };
      const Element* object = object_of(hp);
      EmbodiedSample root_sample;
      const bool root_sampled = SampleAnchorLocked(actor_id, root_name, root_name, &root_sample);
      if (!sampled || !root_sampled) {
        ap.status = ProposalStatus::kNoSample;  // core never substitutes a pose
      } else if (!object) {
        ap.status = ProposalStatus::kNone;  // no counterpart object (D7)
      } else {
        // The actor-root poses (`stand`) are over the root THIS solve proposes, in its heading: the pelvis moves with the
        // root in the same update, and the root's own causes are carried with it (below).
        Eigen::Vector3d root_world(root_sample.x, root_sample.y, root_sample.z);
        double root_yaw = root_sample.yaw;
        if ((set->status == ProposalStatus::kExact || set->status == ProposalStatus::kInexact) &&
            set->position_specified())
          root_world = Eigen::Vector3d(set->x, set->y, set->z);
        if (set->yaw_specified()) root_yaw = set->yaw;
        const std::shared_ptr<const Embodiment> embodiment = EmbodimentOf(actor_id);
        const auto sd = InstantiatePoseSD(element_tree_, object->GetId(), name, embodiment.get(), root_world, root_yaw);
        // The pursuit's origin: the anchor's current position, its sample.
        const Eigen::Vector3d x_now(sample.x, sample.y, sample.z);
        // The anchor's relations in its scope, the SD's region volume V (PoseRegionScope), pursued by the one live path of
        // every anchor (scope_pursuit.h, the root's): each relation is a term w (d(x) - d*)^2, d the geodesic over V
        // to the relation's MAPPED CONTACT c (the source's contact read through the pose's chart on THIS side's
        // instance: its part chart, or the standing point over the proposed root), d* = RatioTargetDistance(d(sample), n).
        // The pursuit starts at the anchor's current position, its sample, brought into V when it lies outside.
        const PoseRegionScope* scope_ptr = nullptr;
        std::optional<PoseRegionScope> region_scope;
        if (sd.ok()) {
          region_scope.emplace(*sd);
          scope_ptr = &*region_scope;
        }
        Eigen::Vector3d start = x_now;
        if (scope_ptr && !scope_ptr->IsFree(start))
          if (const auto onto = scope_ptr->NearestFree(start)) start = *onto;
        std::vector<ScopeTerm<3>> terms;
        std::vector<ScopeTerm<3>>
            diagnostic_terms;  // grounded relations, including idle ones
        std::vector<std::pair<std::string, double>> term_tags;  // (tag, weight) of each term
        std::vector<int> term_kind;                             // AffordancePoseTemplate::Support of each term
        std::vector<double> term_src_d;                         // the source's own distance of each term's relation
        bool any_counterpart = false;
        if (sd.ok()) {
          for (size_t i : inner) {
            const RelationPayload& p = payloads[i];
            if (object_of(p) != object) continue;
            const PoseInstance* inst = sd->Find(p.affordance_pose_tag);
            if (!inst || !(p.grounded_relation.support_distance_m >= 0.0)) continue;  // no counterpart pose (D7)
            any_counterpart = true;
            const double w = weight_at(i);
            // The source's contact, read on this side's instance. Ground idle
            // terms for diagnostics without admitting them into the pursuit.
            PoseContact src;
            src.on_part = p.support.has_contact;
            src.local = p.support.contact_point_local;
            src.chart = p.support.contact_normalized;
            const auto& es = p.grounded_relation.encoded_state;
            if (es.size() >= 3) src.dir = Eigen::Vector3d(es(0), es(1), es(2));
            auto src_kind = SurfaceChartKindFromInt(p.support.sub_region_chart);
            if (!src_kind.ok()) continue;
            const auto mapped = MapPoseContact(inst->support, src, p.support.boundary.get(), *src_kind);
            if (!mapped.ok()) continue;  // the contact cannot be reproduced here: the term drops
            const Eigen::Vector3d c = mapped->point;
            const auto reach = scope_ptr->Reachability(start, c);
            const double d0 = reach.reachable && std::isfinite(reach.length) ? reach.length : (c - start).norm();
            ScopeTerm<3> t;
            t.weight = w;
            t.d_sample = d0;
            t.d_star = RatioTargetDistance(d0, p.update_count);
            t.update_count = p.update_count;
            t.goal = c;
            t.course_goal = c;
            diagnostic_terms.push_back(t);
            // The placed answer reads every relation whatever its weight;
            // ordinary pursuit still admits only genuine weighted requests.
            if (!placed && (!(w > 0.0) || !RatioRequest(p.update_count)))
              continue;
            terms.push_back(std::move(t));
            term_tags.emplace_back(p.affordance_pose_tag, w);
            term_src_d.push_back(p.grounded_relation.support_distance_m);
            const Element* pe = element_tree_.GetElement(inst->element);
            term_kind.push_back(pe && pe->GetAffordancePose() ? static_cast<int>(pe->GetAffordancePose()->support) : 0);
          }
        }
        if (!any_counterpart) {
          ap.status = ProposalStatus::kNone;
        } else if (terms.empty()) {
          // No request: the solve answers at the anchor's current position (its sample, brought into V), a position
          // with NO_REQUEST (as v1: no plan, the entity stays where it is); the
          // retention below replaces it with the retained pending target.
          ap.x = start.x();
          ap.y = start.y();
          ap.z = start.z();
          ap.yaw = sample.yaw;
          ap.flags |= kAnchorFlagYawUnspecified;
          ap.status = ProposalStatus::kInexact;
          ap.referent_shared_id = hp.referent_shared_id;
          ap.inner.valid = true;
          ap.inner.no_request = true;
          ap.inner.object_shared_id = hp.referent_shared_id;
          ap.inner.scale_m = sd->scale_m;
          ap.position_active = true;
          ap.placement = placed;
          RetainIdlePositionLocked(actor_id, &ap);
          // Retention is snapshot selection, not a new solve. Score the final
          // returned pose against this update's grounded terms, and keep the
          // sample errors at the actual sample. No weighted request means J=0.
          if (!diagnostic_terms.empty()) {
            double et = 0.0, es = 0.0;
            const Eigen::Vector3d target(ap.x, ap.y, ap.z);
            for (const auto& t : diagnostic_terms) {
              const auto at = scope_ptr->Reachability(target, *t.goal);
              const auto as = scope_ptr->Reachability(x_now, *t.goal);
              et = std::max(et, std::fabs(at.length - t.d_star));
              es = std::max(es, std::fabs(as.length - t.d_star));
            }
            ap.err_scale_m = sd->scale_m;
            ap.err_pose_target = et / sd->scale_m;
            ap.err_pose_sample = es / sd->scale_m;
          }
        } else {
          ScopePursuitInput<3> pin;
          pin.scope = scope_ptr;
          pin.start = start;
          pin.terms = terms;
          pin.gn_safety_cap = solver_.config().gn_safety_cap;
          pin.exact_eps_m = eps_m;
          pin.budget_evals = solver_.config().solve_budget_evals;
          ScopePursuitResult<3> r;
          int evaluations = 0;
          bool budget_tripped = false;
          double position_resolution_m = -1.0;
          if (placed) {
            // The actor's first proposal, or a source exception: PLACED, as every anchor is (the root's rule, v1's
            // approach point): the mapped contact of the relation the source is most about to reach (the smallest source
            // distance; ties to the larger weight), brought into V.
            size_t nearest = 0;
            for (size_t k = 1; k < terms.size(); ++k)
              if (term_src_d[k] < term_src_d[nearest] ||
                  (term_src_d[k] == term_src_d[nearest] && term_tags[k].second > term_tags[nearest].second))
                nearest = k;
            r.position = *scope_ptr->NearestFree(*terms[nearest].goal);
            r.objective_exact = scope_ptr->IsFree(*terms[nearest].goal);
            r.no_request = false;
          } else {
            PursueInScope<3>(pin, r, evaluations, budget_tripped, position_resolution_m);
          }
          size_t driving = 0;
          for (size_t k = 1; k < term_tags.size(); ++k)
            if (term_tags[k].second > term_tags[driving].second) driving = k;
          ap.x = r.position.x();
          ap.y = r.position.y();
          ap.z = r.position.z();
          // The inner terms are distances: they do not orient the anchor. Its yaw is the sample's, unspecified.
          ap.yaw = sample.yaw;
          ap.flags |= kAnchorFlagYawUnspecified;
          ap.status = r.budget_hit                            ? ProposalStatus::kInexact
                      : r.status == SolverStatus::kNotConverged ? ProposalStatus::kNotConverged
                      : r.objective_exact                       ? ProposalStatus::kExact
                                                                : ProposalStatus::kInexact;
          ap.referent_shared_id = hp.referent_shared_id;
          const PoseSD& v = *sd;
          ap.err_scale_m = v.scale_m;
          double et = 0.0, es = 0.0;
          for (const ScopeTerm<3>& t : terms) {
            const auto at = scope_ptr->Reachability(r.position, *t.goal);
            const auto as = scope_ptr->Reachability(x_now, *t.goal);
            et = std::max(et, std::fabs(at.length - t.d_star));
            es = std::max(es, std::fabs(as.length - t.d_star));
          }
          ap.err_pose_target = et / v.scale_m;
          ap.err_pose_sample = es / v.scale_m;
          ap.inner.valid = true;
          ap.inner.object_shared_id = hp.referent_shared_id;
          ap.inner.scale_m = v.scale_m;
          ap.inner.j = r.objective_after;
          ap.inner.iterations = r.iterations;
          ap.inner.driving_tag = term_tags[driving].first;
          for (size_t k = 0; k < terms.size(); ++k)
            if (term_kind[k] == static_cast<int>(AffordancePoseTemplate::Support::kActorRoot))
              ap.inner.carries_root = true;
        }
      }
    } else if (!ap_idx.empty()) {
      // Bound, but the source anchor does not touch its surface: no target. The
      // anchor follows the body.
      ap.status = ProposalStatus::kNone;
    } else if (!spacing_idx.empty() || !facing_idx.empty()) {
      // The anchor's relations in the room's scope (Spacing and Facing; the root's are the same, EvaluateRemoteRelation):
      // pursued by the same solve with the same continuation (SolveRoomScopeLocked), from the anchor's own sample. Its
      // height is the sample's. With no Spacing relation nothing moves its position: it is the sample with the yaw it
      // should have. Without a sample there is nothing to answer (core never substitutes a pose).
      if (!sampled) {
        ap.status = ProposalStatus::kNoSample;
      } else {
        std::vector<RelationPayload> group;
        std::vector<double> group_weights;
        double heaviest = -1.0;
        for (const auto* idx : {&spacing_idx, &facing_idx})
          for (size_t i : *idx) {
            group.push_back(payloads[i]);
            group_weights.push_back(weight_at(i));
            if (weight_at(i) > heaviest) {
              heaviest = weight_at(i);
              ap.referent_shared_id = payloads[i].referent_shared_id;
            }
          }
        // The anchor's own driver discontinuity continues cold, as the root's does.
        bool embodied_reset = false;
        if (const auto prev = attr.anchors.find(id); prev != attr.anchors.end() && prev->second.has_prev)
          embodied_reset = prev->second.sample_epoch != sample.epoch ||
                           ((sample.flags & kEmbodiedTeleported) != 0 && prev->second.teleport_sequence != sample.sequence);
        const SolverResult solved =
            SolveRoomScopeLocked(sample, group, group_weights, navigable_def, placed, placed || embodied_reset,
                                 actor_id, id);
        anchor_solve = solved;
        anchor_group = group;
        anchor_group_weights = group_weights;
        ap.x = sample.x;
        ap.y = sample.y;
        ap.z = sample.z;
        ap.yaw = sample.yaw;
        if (solved.status == SolverStatus::kFailed || solved.no_grounded_referent) {
          ap.status = ProposalStatus::kNone;  // no referent maps on the receiver (D7)
        } else {
          if (!spacing_idx.empty()) {
            ap.x = solved.position.x();
            ap.y = solved.position.y();
          }
          ap.yaw = solved.yaw_specified ? solved.yaw : sample.yaw;
          ap.yaw_turn = solved.yaw_specified ? solved.yaw_turn : 0.0;  // ABI 4.12 (from the same sample yaw)
          if (!solved.yaw_specified) ap.flags |= kAnchorFlagYawUnspecified;
          ap.placement = placed;
          ap.position_active = solved.position_active;
          ap.position_requested = solved.position_requested;
          RetainIdlePositionLocked(actor_id, &ap);
          // The relation errors at the target and at the sample, and ARRIVED, by family: Spacing in metres within the
          // solver's tolerance, Facing by v1's definition (relation_evaluator.h kFacingArrivalRad, the SIGNED angle).
          bool saw = false, exact = true, saw_sample = false, arrived_all = true;
          for (size_t k = 0; k < group.size(); ++k) {
            const PreparedTargetRelation prepared = PrepareTargetRelation(group[k], element_tree_, eval_options);
            if (!prepared.ok()) continue;
            const bool spacing = group[k].kernel_type == RelationKernelType::kAvatarSpacing;
            const auto et = EvaluatePreparedTargetRelation(
                prepared,
                TargetAnchorPose{
                    Eigen::Vector3d(ap.x, ap.y, spacing ? 0.0 : ap.z), ap.yaw},
                element_tree_);
            const auto es = EvaluatePreparedTargetRelation(
                prepared,
                TargetAnchorPose{Eigen::Vector3d(sample.x, sample.y,
                                                 spacing ? 0.0 : sample.z),
                                 sample.yaw},
                element_tree_);
            const double scale = spacing && prepared.normalizing_scope_diameter_m > 1e-6
                                     ? prepared.normalizing_scope_diameter_m
                                     : 1.0;
            double& err_t = spacing ? ap.err_spacing_target : ap.err_facing_target;
            double& err_s = spacing ? ap.err_spacing_sample : ap.err_facing_sample;
            if (et.ok()) {
              const double e = std::fabs(et.r_tgt - et.r_src);
              err_t = std::max(err_t, e);
              saw = true;
              if (e * scale > eps_m) exact = false;
            }
            if (es.ok()) {
              const double e = std::fabs(es.r_tgt - es.r_src);
              err_s = std::max(err_s, e);
              saw_sample = true;
              if (spacing ? e * scale > eps_m : !FacingArrived(group[k], es)) arrived_all = false;
            }
            if (spacing && prepared.normalizing_scope_diameter_m > 0.0) ap.err_scale_m = prepared.normalizing_scope_diameter_m;
          }
          ap.status = solved.status == SolverStatus::kNotConverged ? ProposalStatus::kNotConverged
                      : solved.status == SolverStatus::kBudgetHit  ? ProposalStatus::kInexact
                      : (saw && exact)                             ? ProposalStatus::kExact
                                                                   : ProposalStatus::kInexact;
          if (saw_sample && arrived_all) ap.flags |= kAnchorFlagArrived;
        }
      }
    }

    if (ap.status == ProposalStatus::kExact ||
        ap.status == ProposalStatus::kInexact) {
      ap.placement = placed;
      if (engaged_target) {
        ap.position_active = ap.yaw_active = true;
        ap.position_requested = true;
        // An unsampled engaged contact is still an absolute pose snapshot.
        // Only placement may deliver its yaw without an accepted own-anchor
        // origin; an ordinary signed turn must correspond to that sample.
        ap.yaw_requested = placed || sampled;
      } else if (ap.inner.valid) {
        ap.position_active = true;
        ap.position_requested = !ap.inner.no_request;
      } else if (anchor_solve) {
        ap.position_active = anchor_solve->position_active;
        ap.yaw_active = anchor_solve->yaw_active;
        ap.position_requested = anchor_solve->position_requested;
        ap.yaw_requested = anchor_solve->yaw_requested;
      }
    }

    // Seated facing: while the anchor is on its support or in its object's SD, the head and upper
    // body follow the direction core's Facing gives the actor, not the user's raw head yaw. The proposal carries that
    // direction (the root's Facing yaw) when the root has one.
    if ((ap.flags & (kAnchorFlagEngaged | kAnchorFlagInner)) != 0 && set->yaw_specified() &&
        (ap.status == ProposalStatus::kExact || ap.status == ProposalStatus::kInexact)) {
      ap.flags |= kAnchorFlagFacingYaw;
      ap.facing_yaw = set->yaw;
    }

    // Pitch and roll are the anchor's own: the pass-through of the source anchor's
    // orientation beyond yaw, from its heaviest relation that carries it, as the root's is the actor's.
    {
      double best_w = -1.0;
      for (size_t i = 0; i < payloads.size(); ++i) {
        const auto& p = payloads[i];
        if (!p.enabled || !p.has_subject_orientation) continue;
        if ((is_root_name(p.subject_anchor_name) ? root_name : p.subject_anchor_name) != name) continue;
        if (weight_at(i) > best_w) {
          best_w = weight_at(i);
          ap.orientation_type = p.subject_orientation.type;
          ap.pitch = p.subject_orientation.pitch;
          ap.roll = p.subject_orientation.roll;
        }
      }
    }
    // Why this anchor's proposal differs from its previous one: the rule of every anchor (AttributeAnchorCauses).
    AnchorCauseFacts facts = anchor_solve ? SolveCauseFacts(*anchor_solve, anchor_group, anchor_group_weights)
                                          : AnchorCauseFacts{};
    if (!anchor_solve) facts.signature = signature;
    facts.sampled = sampled;
    facts.sample_epoch = sample.epoch;
    facts.sample_sequence = sample.sequence;
    facts.teleported = (sample.flags & kEmbodiedTeleported) != 0;
    facts.driving_tag = ap.inner.driving_tag;
    facts.stuck = ap.status == ProposalStatus::kNone || ap.status == ProposalStatus::kNoSample;
    facts.engaged = (ap.flags & kAnchorFlagEngaged) != 0;
    facts.exception = update.has_exception;
    if (anchor_solve)
      if (const RelationPayload* cr = ContainmentRelationLocked(anchor_group, anchor_group_weights, nullptr)) {
        facts.contained = true;
        facts.contain_scope = cr->referent_shared_id;
      }
    if (ap.inner.valid) {
      facts.progress = !ap.inner.no_request;
      facts.no_request = ap.inner.no_request;
    }
    AnchorCauseState state;
    if (const auto prev = attr.anchors.find(id); prev != attr.anchors.end()) state = prev->second;
    uint32_t cause = AttributeAnchorCauses(&state, facts, geometry_revision_, /*commit=*/true);
    // The scope order: an anchor whose scope is instantiated over the root's proposal
    // of this update (the actor-root poses, `stand`) moves with it, and carries its causes (its first proposal excepted,
    // which is the root's own).
    if (ap.inner.carries_root) cause |= root_cause & ~kCauseFirstProposal;
    ap.cause = cause;
    next_states[id] = std::move(state);
    ApplyAnchorChartLocked(update, name, /*is_root=*/false, &ap);
    set->anchors.push_back(std::move(ap));
  }
  if (!receiver_reevaluation) {
    for (auto it = attr.anchors.begin(); it != attr.anchors.end();)
      it = it->first == kRootAnchorId ? std::next(it) : attr.anchors.erase(it);
    for (auto& [aid, st] : next_states) attr.anchors[aid] = std::move(st);
  }
}

// ---- the latency timers -------------------------------------------------------

void Space::EvaluateInteractionLatencyLocked(
    spatialblend::entity_id_t actor_id, const RemoteRelationUpdate& update,
    const std::vector<RelationPayload>& payloads,
    const std::vector<double>& weights, const std::string& root_name,
    const ReceiverProposal& set, uint64_t revision,
    std::vector<ActorEvent>* events) {
  if (set.status == ProposalStatus::kNone) {
    // No referent maps on this receiver: the source's engagement has no counterpart here, so it
    // has ended for this target. Its timers are
    // cancelled, and a later engagement starts a new one: a timer latched before the gap must
    // never carry across it.
    latency_trackers_.erase(actor_id);
    return;
  }
  const auto mailbox = MailboxFor(actor_id);
  if (!mailbox) return;
  // The interactions the source holds engaged: Spacing contact (R == 1) and an
  // AffordancePose anchor touching its surface. Facing never starts a timer. The
  // container the source is inside is the scope, not an interaction.
  std::vector<InteractionObservation> engaged;
  for (size_t i = 0; i < payloads.size(); ++i) {
    const auto& p = payloads[i];
    if (!p.enabled || (i < weights.size() && !(weights[i] > 0.0))) continue;
    InteractionKind kind;
    if (IsTimedSpacingContact(p)) {
      kind = InteractionKind::kSpacingContact;
    } else if (p.kernel_type == RelationKernelType::kAvatarAffordancePose &&
               !IsInnerPosePayload(p) && IsEngaged(p)) {
      kind = InteractionKind::kAffordancePoseEngagement;
    } else {
      continue;
    }
    InteractionObservation o;
    o.source_shared_id = update.actor_shared_id;
    o.anchor_id = InternAnchor(*mailbox, p.subject_anchor_name, root_name);
    o.referent_shared_id = p.referent_shared_id;
    o.kind = kind;
    // Arrived: the target app's SAMPLED anchor already satisfies the relation (L2).
    const AnchorProposal* anchor = set.Find(o.anchor_id);
    o.arrived = anchor && anchor->arrived();
    engaged.push_back(o);
  }
  const double now = ActorSecondsLocked(actor_id);
  for (const InteractionExpiry& e :
       latency_trackers_[actor_id].Update(now, engaged, latency_policy_)) {
    // Core asks THIS target app, which has not arrived, to teleport.
    const AnchorProposal* anchor = set.Find(e.observation.anchor_id);
    if (!anchor) continue;
    ActorEvent event;
    event.kind = ActorEvent::Kind::kTeleportRequest;
    event.revision = revision;
    event.source_sequence = update.sequence;
    event.source_shared_id = e.observation.source_shared_id;
    event.interaction_kind = e.observation.kind;
    event.anchor_id = e.observation.anchor_id;
    event.x = anchor->x;
    event.y = anchor->y;
    event.z = anchor->z;
    event.yaw = anchor->yaw;
    event.pitch = anchor->pitch;
    event.roll = anchor->roll;
    event.orientation_type = anchor->orientation_type;
    events->push_back(std::move(event));
  }
}

}  // namespace spatialblend
