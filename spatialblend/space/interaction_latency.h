// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/interaction_latency.h
//
// The per-interaction maximum-latency policy and its timers (L1-L4).
//
// The receiver's driver keeps following the latest proposals within its own
// kinematic bounds, so the target can lag a source that is already at its
// interaction. Core holds, per interaction kind, the longest lag it accepts. When
// that time has run out and a target app has not arrived, core asks that app to
// teleport (an SB_ACTOR_EVENT_TELEPORT_REQUEST carrying the target pose).
//
//   L1  The timer starts when the SOURCE relation is engaged: Spacing contact
//       (R == 1) or an AffordancePose anchor touching its surface. Facing never
//       starts one. It is latched once per engagement.
//   L2  A target has ARRIVED when its SAMPLED anchor already satisfies the relation
//       (AnchorProposal::arrived()): R_tgt at the sample within the solver's declared
//       tolerance of the source's R (1 - eps at contact) for the root, touching the
//       surface for an AffordancePose anchor. No new epsilon. It is NOT the solver's
//       EXACT: that says the TARGET is an exact minimiser, which it is at contact
//       wherever the body is.
//   L3  If the source's engagement ends first, the request is cancelled.
//   L4  5000 ms for both kinds: v1's interaction timeout
//       default, kDefaultTimeoutMs, which
//       is the effective deployed value because v1's policy field defaults to
//       INT_MAX (= use kDefaultTimeoutMs) and v1 used one value for every kind.
//       DefaultInteractionLatencyPolicy() is the single place they are written.
#ifndef SPATIALBLEND_SPACE_INTERACTION_LATENCY_H_
#define SPATIALBLEND_SPACE_INTERACTION_LATENCY_H_

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

namespace spatialblend {

enum class InteractionKind : int32_t {
  kSpacingContact = 0,
  kAffordancePoseEngagement = 1,
};
constexpr int kInteractionKindCount = 2;

// The longest lag core accepts for each interaction kind, in seconds of SOURCE
// time. An empty entry means no limit: the timer is tracked and never acts.
struct InteractionLatencyPolicy {
  std::array<std::optional<double>, kInteractionKindCount> max_latency_s{};

  std::optional<double> For(InteractionKind kind) const {
    return max_latency_s[static_cast<size_t>(kind)];
  }
};

// L4. v1's interaction timeout default (interaction_manager.h kDefaultTimeoutMs =
// 5000 ms), for both kinds. The values live here, nowhere else.
inline InteractionLatencyPolicy DefaultInteractionLatencyPolicy() {
  InteractionLatencyPolicy policy;
  policy.max_latency_s[static_cast<size_t>(InteractionKind::kSpacingContact)] = 5.0;
  policy.max_latency_s[static_cast<size_t>(
      InteractionKind::kAffordancePoseEngagement)] = 5.0;
  return policy;
}

// One engaged source interaction, as seen by one receiver actor in one solve.
struct InteractionObservation {
  int64_t source_shared_id = 0;  // the source actor that holds the engagement
  uint32_t anchor_id = 0;        // the receiver anchor that realizes it
  int64_t referent_shared_id = 0;
  InteractionKind kind = InteractionKind::kSpacingContact;
  bool arrived = false;  // the target's SAMPLED anchor satisfies the relation
};

// A target that has not arrived within the policy's limit: core asks it to teleport.
struct InteractionExpiry {
  InteractionObservation observation;
  double started_s = 0.0;
  double now_s = 0.0;
};

// The timers of ONE receiver actor.
class InteractionLatencyTracker {
 public:
  // `engaged` lists every interaction the source holds engaged in this solve.
  // An interaction that was engaged before and is not listed now has ended: its
  // timer (and any pending request) is cancelled. Returns the interactions that
  // expire in this solve, each at most once per engagement.
  std::vector<InteractionExpiry> Update(
      double now_s, const std::vector<InteractionObservation>& engaged,
      const InteractionLatencyPolicy& policy) {
    std::map<Key, Entry> next;
    std::vector<InteractionExpiry> expired;
    for (const auto& o : engaged) {
      const Key key = KeyOf(o);
      if (next.count(key)) continue;
      Entry e;
      const auto it = entries_.find(key);
      if (it != entries_.end()) {
        e = it->second;  // latched: the start time is never taken again
      } else {
        e.started_s = now_s;
      }
      if (o.arrived) e.arrived = true;
      if (!e.arrived && !e.requested) {
        const std::optional<double> limit = policy.For(o.kind);
        if (limit.has_value() && now_s - e.started_s >= *limit) {
          e.requested = true;
          expired.push_back({o, e.started_s, now_s});
        }
      }
      next.emplace(key, e);
    }
    entries_ = std::move(next);
    return expired;
  }

  // Test and trace access: the latched start time of an engagement.
  std::optional<double> StartedAt(const InteractionObservation& o) const {
    const auto it = entries_.find(KeyOf(o));
    if (it == entries_.end()) return std::nullopt;
    return it->second.started_s;
  }
  bool Arrived(const InteractionObservation& o) const {
    const auto it = entries_.find(KeyOf(o));
    return it != entries_.end() && it->second.arrived;
  }
  size_t size() const { return entries_.size(); }
  void Clear() { entries_.clear(); }

 private:
  using Key = std::tuple<int64_t, uint32_t, int64_t, int32_t>;
  struct Entry {
    double started_s = 0.0;
    bool arrived = false;    // the target's sampled anchor arrived: the timer stopped
    bool requested = false;  // the teleport request was raised
  };
  static Key KeyOf(const InteractionObservation& o) {
    return {o.source_shared_id, o.anchor_id, o.referent_shared_id,
            static_cast<int32_t>(o.kind)};
  }
  std::map<Key, Entry> entries_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_INTERACTION_LATENCY_H_
