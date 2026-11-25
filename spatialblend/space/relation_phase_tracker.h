// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_RELATION_PHASE_TRACKER_H_
#define SPATIALBLEND_SPACE_RELATION_PHASE_TRACKER_H_

#include <array>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "spatialblend/space/relation_event.h"

namespace spatialblend {

// Thresholds for the relation lifecycle state machine.
// Enter/leave pairs give hysteresis so the DRIVING and CONTACT labels don't
// flicker at a boundary.
//
// There is no activation gate: a relation in the family's scope is realized
// (the scope definitions D1-D9). There is no R dormant gate, no AffordancePose
// near field, no tracking-gap dwell and no scope radius. A relation enters kActive when it first
// enters scope and returns to kDormant when it leaves scope (the caller clears it).
struct RelationPhaseConfig {
  double driving_weight = 0.50;      // weight to enter kDriving
  double driving_release = 0.40;     // weight to leave kDriving
  double contact_proximity = 0.95;   // R to enter kContact
  double contact_release = 0.85;     // R to leave kContact
};

// Per-relation phase state.
struct RelationPhaseState {
  RelationPhase phase = RelationPhase::kDormant;
};

// One step of the lifecycle state machine (shared by RelationPhaseTracker and
// the per-(binding, element) state in Space). Returns the new phase and updates
// `state` in place.
RelationPhase StepRelationPhase(RelationPhaseState* state, double proximity,
                                double weight,
                                const RelationPhaseConfig& config);

// Declared per-family defaults. They are the same for every family.
RelationPhaseConfig DefaultRelationPhaseConfig(RelationKernelType family);

// OK when every enter/release pair is ordered for hysteresis and finite:
// driving_release <= driving_weight, contact_release <= contact_proximity.
absl::Status ValidateRelationPhaseConfig(const RelationPhaseConfig& config);

// Tracks the lifecycle phase of each (actor, anchor, referent, kernel) relation
// and emits a RelationEvent on every transition. The chain
// kDormant↔kActive↔kDriving↔kContact advances at most one step per Update,
// except that a grounded relation (proximity ≥ contact_proximity) may enter
// kContact directly from kActive even when its weight is low. A relation
// enters kActive the moment it is observed (it is in scope).
class RelationPhaseTracker {
 public:
  // Per-family declared defaults (DefaultRelationPhaseConfig).
  RelationPhaseTracker();
  // One config applied to every family (legacy / tests).
  explicit RelationPhaseTracker(RelationPhaseConfig config);

  // Per-family thresholds. SetConfig rejects an invalid config
  // (ValidateRelationPhaseConfig) and keeps the previous one. Takes effect on
  // the next Update; existing phases are kept.
  absl::Status SetConfig(RelationKernelType family,
                         const RelationPhaseConfig& config);
  const RelationPhaseConfig& GetConfig(RelationKernelType family) const;

  // Feed one relation observation; returns events for any transition.
  // `identity` carries the four id fields (from/to ignored).
  std::vector<RelationEvent> Update(
      const RelationEvent& identity, double proximity, double weight,
      double timestamp);
  // Same public event identity, with a private owner namespace for sessions
  // whose shared actor ids are absent, duplicated, or otherwise non-unique.
  std::vector<RelationEvent> UpdateForOwner(
      const RelationEvent& identity, const std::string& owner,
      double proximity, double weight, double timestamp);

  RelationPhase GetPhase(const RelationEvent& identity) const;

  // Drops every anchor/referent/kernel phase for one actor while retaining
  // independent actors' temporal lifetimes.
  void ClearActor(shared_element_id_t actor_shared_id);
  void ClearOwner(const std::string& owner);
  // Authoritative disappearance starts a new lifetime for this relation only.
  void ClearRelationForOwner(const RelationEvent& identity, const std::string& owner);
  RelationPhase GetPhaseForOwner(const RelationEvent& identity,
                                 const std::string& owner) const;

 private:
  static std::string OwnerForActor(shared_element_id_t actor_shared_id);
  static std::string Key(const RelationEvent& identity,
                         const std::string& owner);

  static size_t FamilyIndex(RelationKernelType family);

  std::array<RelationPhaseConfig, 3> configs_;
  std::unordered_map<std::string, RelationPhaseState> phases_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_RELATION_PHASE_TRACKER_H_
