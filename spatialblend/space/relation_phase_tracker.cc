// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/relation_phase_tracker.h"

#include <cmath>

#include "absl/strings/str_cat.h"

namespace spatialblend {
void RelationPhaseTracker::ClearRelationForOwner(const RelationEvent& identity,
                                                 const std::string& owner) {
  phases_.erase(Key(identity, owner));
}

RelationPhase StepRelationPhase(RelationPhaseState* state, double proximity,
                                double weight,
                                const RelationPhaseConfig& c) {
  RelationPhase next = state->phase;
  switch (state->phase) {
    case RelationPhase::kDormant:
      next = RelationPhase::kActive;  // observed = in scope = active
      break;
    case RelationPhase::kActive:
      if (proximity >= c.contact_proximity) next = RelationPhase::kContact;
      else if (weight >= c.driving_weight) next = RelationPhase::kDriving;
      break;
    case RelationPhase::kDriving:
      if (proximity >= c.contact_proximity) next = RelationPhase::kContact;
      else if (weight < c.driving_release) next = RelationPhase::kActive;
      break;
    case RelationPhase::kContact:
      if (proximity < c.contact_release) next = RelationPhase::kDriving;
      break;
  }
  return state->phase = next;
}

RelationPhaseConfig DefaultRelationPhaseConfig(RelationKernelType) {
  return RelationPhaseConfig{};
}

absl::Status ValidateRelationPhaseConfig(const RelationPhaseConfig& c) {
  const double values[] = {c.driving_weight, c.driving_release,
                           c.contact_proximity, c.contact_release};
  for (double v : values) {
    if (!std::isfinite(v)) {
      return absl::InvalidArgumentError("phase thresholds must be finite");
    }
  }
  if (c.driving_release > c.driving_weight) {
    return absl::InvalidArgumentError("need driving_release <= driving_weight");
  }
  if (c.contact_release > c.contact_proximity) {
    return absl::InvalidArgumentError(
        "need contact_release <= contact_proximity");
  }
  return absl::OkStatus();
}

RelationPhaseTracker::RelationPhaseTracker() {
  configs_[FamilyIndex(RelationKernelType::kAvatarFacing)] =
      DefaultRelationPhaseConfig(RelationKernelType::kAvatarFacing);
  configs_[FamilyIndex(RelationKernelType::kAvatarSpacing)] =
      DefaultRelationPhaseConfig(RelationKernelType::kAvatarSpacing);
  configs_[FamilyIndex(RelationKernelType::kAvatarAffordancePose)] =
      DefaultRelationPhaseConfig(RelationKernelType::kAvatarAffordancePose);
}

RelationPhaseTracker::RelationPhaseTracker(RelationPhaseConfig config) {
  configs_.fill(config);
}

size_t RelationPhaseTracker::FamilyIndex(RelationKernelType family) {
  switch (family) {
    case RelationKernelType::kAvatarFacing:
      return 0;
    case RelationKernelType::kAvatarSpacing:
      return 1;
    case RelationKernelType::kAvatarAffordancePose:
      return 2;
  }
  return 0;
}

absl::Status RelationPhaseTracker::SetConfig(RelationKernelType family,
                                             const RelationPhaseConfig& config) {
  absl::Status status = ValidateRelationPhaseConfig(config);
  if (!status.ok()) return status;
  configs_[FamilyIndex(family)] = config;
  return absl::OkStatus();
}

const RelationPhaseConfig& RelationPhaseTracker::GetConfig(
    RelationKernelType family) const {
  return configs_[FamilyIndex(family)];
}

std::string RelationPhaseTracker::OwnerForActor(
    shared_element_id_t actor_shared_id) {
  return absl::StrCat("shared:", actor_shared_id);
}

std::string RelationPhaseTracker::Key(const RelationEvent& identity,
                                      const std::string& owner) {
  return absl::StrCat(owner, "|", identity.anchor_name, "|",
                      identity.referent_shared_id, "|",
                      static_cast<int>(identity.kernel_type));
}

std::vector<RelationEvent> RelationPhaseTracker::Update(
    const RelationEvent& identity, double proximity, double weight,
    double timestamp) {
  return UpdateForOwner(identity, OwnerForActor(identity.actor_shared_id),
                        proximity, weight, timestamp);
}

std::vector<RelationEvent> RelationPhaseTracker::UpdateForOwner(
    const RelationEvent& identity, const std::string& owner,
    double proximity, double weight, double timestamp) {
  const std::string key = Key(identity, owner);
  RelationPhaseState& state = phases_[key];  // new: dormant, no event
  const RelationPhase current = state.phase;
  const RelationPhase next =
      StepRelationPhase(&state, proximity, weight,
                        configs_[FamilyIndex(identity.kernel_type)]);

  std::vector<RelationEvent> events;
  if (next != current) {
    RelationEvent ev = identity;
    ev.from = current;
    ev.to = next;
    ev.timestamp = timestamp;
    events.push_back(ev);
  }
  return events;
}

RelationPhase RelationPhaseTracker::GetPhase(
    const RelationEvent& identity) const {
  return GetPhaseForOwner(identity, OwnerForActor(identity.actor_shared_id));
}

RelationPhase RelationPhaseTracker::GetPhaseForOwner(
    const RelationEvent& identity, const std::string& owner) const {
  auto it = phases_.find(Key(identity, owner));
  return (it != phases_.end()) ? it->second.phase : RelationPhase::kDormant;
}

void RelationPhaseTracker::ClearActor(shared_element_id_t actor_shared_id) {
  ClearOwner(OwnerForActor(actor_shared_id));
}

void RelationPhaseTracker::ClearOwner(const std::string& owner) {
  const std::string prefix = absl::StrCat(owner, "|");
  for (auto it = phases_.begin(); it != phases_.end();) {
    if (it->first.rfind(prefix, 0) == 0) {
      it = phases_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace spatialblend
