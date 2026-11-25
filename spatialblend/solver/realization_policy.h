// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/realization_policy.h
//
// Stateless realization policy: computes per-payload solver weights. The solver
// is fully transparent to the policy — it receives a weight vector and optimizes
// the weighted objective (Eq 8). The policy is the only component that decides
// which relation drives reconstruction.
//
// Weights are computed per kernel-type group so different relation families
// (spacing, facing, affordance) do not share a weight budget (yaw-only facing
// would otherwise starve positional spacing).
//
// POLICY: ComputeAnnealedWeights — v1 SoftAnnealing (Case A, the runtime
// default). The scheduling quantity is the update count n, not a κ-clipped
// proximity. v1 does not form R_eng = clip(1 − (1 − R) / κ, 0, 1) and does
// not turn that into a time-to-contact. η_eff = η_0 (n_min − 1 + ε)^γ, and
// the weights are softmax(−(n − n_min) / η_eff), one-hot as n_min → 1.

#ifndef SPATIALBLEND_SPACE_REALIZATION_POLICY_H_
#define SPATIALBLEND_SPACE_REALIZATION_POLICY_H_

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "spatialblend/solver/soft_annealing_weights.h"
#include "spatialblend/model/realization/remote_update.h"

namespace spatialblend {

// ===========================================================================
// v1's battle-tested PROGRESSIVE WEIGHTED ANNEALING, brought over.
//
// Annealed-softmax weights keyed on the update count n. As the approached
// referent nears contact (n → 1), the temperature decays and the distribution
// collapses to one-hot, so the blend commits to a single object. A referent
// with no valid intent (non-finite n, the static or receding case) takes no
// weight when a peer has one.
//
// Reuses v1's exact implementation (ComputeSoftAnnealingWeights, solver/soft_annealing_weights.h),
// per (kernel_type, subject_anchor) group, on payload.update_count.
// Defaults: eta=1, decay on, gamma=1, epsilon=1e-3. A non-positive or
// non-finite count is no valid intent and takes no weight.
// ===========================================================================
// v1 SoftAnnealing (Case A): η_eff = η_0 (n_min − 1 + ε)^γ, then
// softmax(−(n − n_min) / η_eff). There is no η clip and no κ-clipped
// engagement score. eta_min / eta_max are unused by this schedule.
struct AnnealWeightConfig {
  double eta = 1.0;         // η_0: overall temperature scale
  double decay_gamma = 1.0; // γ: annealing rate
  double epsilon_t = 1e-3;  // ε_T: avoids degeneracy near T_min = 0
  double eta_min = 0.05;    // sharpest usable temperature
  double eta_max = 10.0;    // most diffuse usable temperature (bounds flatness)
};

// Weighting policy family, as v1's planners. Progressive (the default) is v1 SoftAnnealing; Greedy is v1
// NearestFirst, weight 1 on the intent with the fewest updates (v1's InteractionTimeline order); Uniform is v1
// MinProximityError's 1/N over every target whatever its count.
enum class RealizationPolicyType { kProgressive, kUniform, kGreedy };

// Configuration for the progressive realization policy.
// v1 SoftAnnealing: η_eff = η_0 (n_min − 1 + ε)^γ. The runtime uses these
// defaults (γ = 1, decay on). eta_min / eta_max are not applied.
struct RealizationPolicyConfig {
  RealizationPolicyType type = RealizationPolicyType::kProgressive;
  double eta_0 = 1.0;          // Base temperature scale
  double decay_gamma = 1.0;    // Annealing exponent (γ)
  double epsilon = 1e-3;       // Numerical stability (ε_T)
  double eta_min = 1e-6;       // Minimum temperature
  double eta_max = 100.0;      // Maximum temperature
};

// The realization policy parameters the runtime actually uses, with
// their provenance.
//
//  γ = 1, decay on. That is v1 SoftAnnealing. The earlier γ = 2 deviation
//  and the κ-clipped engagement score are not part of this schedule.
//
//  η_min and η_max stay on the struct so a caller can still name them. The
//  schedule does not clip. AnnealWeightConfig's bounds are likewise unused.
inline RealizationPolicyConfig ProductionRealizationPolicyConfig() {
  return RealizationPolicyConfig{};
}

// Declared imminence constants. Scheduling does not read them: v1 anneals
// the update count, and the source does not form
//   R_eng = clip(1 − (1 − proximity) / κ, 0, 1)
//   T_p   = clip((1 − R_eng) / max(Ṙ⁺, ε_v), 0, T_max).
// The struct remains so a caller can still observe the declared values.
struct ImminenceConfig {
  // EMA time constant (s) of the progress derivative; alpha = 0.3 at the
  // legacy 33 ms tick: -0.033 / log(0.7).
  double progress_tau_s = 0.092517;
  // Approach-rate floor ε_v (1/s).
  double epsilon_v = 0.01;
  // Imminence cap T_max (s) = 1 / ε_v.
  double t_max = 100.0;
  // Proxemic engagement scale κ: D_engage = κ · scope_extent (Hall social
  // distance / typical room ≈ 0.2).
  double engage_kappa = 0.2;
};

inline bool IsValidRealizationPolicyConfig(const RealizationPolicyConfig& c) {
  return std::isfinite(c.eta_0) && c.eta_0 > 0.0 &&
         std::isfinite(c.decay_gamma) && c.decay_gamma >= 0.0 &&
         std::isfinite(c.epsilon) && c.epsilon >= 0.0 &&
         std::isfinite(c.eta_min) && c.eta_min > 0.0 &&
         std::isfinite(c.eta_max) && c.eta_max >= c.eta_min;
}

inline bool IsValidImminenceConfig(const ImminenceConfig& c) {
  return std::isfinite(c.progress_tau_s) && c.progress_tau_s > 0.0 &&
         std::isfinite(c.epsilon_v) && c.epsilon_v > 0.0 &&
         std::isfinite(c.t_max) && c.t_max > 0.0 &&
         std::isfinite(c.engage_kappa) && c.engage_kappa > 0.0 &&
         c.engage_kappa <= 1.0;
}

inline const char* RealizationPolicyName(RealizationPolicyType type) {
  switch (type) {
    case RealizationPolicyType::kProgressive:
      return "progressive";
    case RealizationPolicyType::kUniform:
      return "uniform";
    case RealizationPolicyType::kGreedy:
      return "greedy";
  }
  return "unknown";
}

inline bool HasValidEstimatedTime(double seconds) {
  return std::isfinite(seconds) && seconds >= 0.0;
}

// A scheduling count is a positive finite update count. Zero, negative, and
// non-finite values are no valid intent (v1 skips steps <= 0).
inline bool HasValidUpdateCount(double count) {
  return std::isfinite(count) && count > 0.0;
}

// The scheduling score is the update count, in updates. A relation without a
// valid count has no score; the time-to-satisfaction in seconds is not a
// count and is not substituted for one.
inline double SchedulingScore(const RelationPayload& payload) {
  if (HasValidUpdateCount(payload.update_count)) return payload.update_count;
  return std::numeric_limits<double>::quiet_NaN();
}

// A weight channel: (kernel family, subject anchor, level). The level separates an anchor's INNER affordance-pose
// relations (room -> object SD -> poses, RelationPayload::affordance_pose_tag) from its contact relations: the two
// levels are two solves and never share a weight budget, as the families do not.
using ChannelKey = std::tuple<int, std::string, int>;
inline ChannelKey ChannelKeyOf(const RelationPayload& p) {
  return ChannelKey{static_cast<int>(p.kernel_type), p.subject_anchor_name, p.affordance_pose_tag.empty() ? 0 : 1};
}

// v1 SoftAnnealing over one group's update counts: a relation without a valid count
// takes weight 0 (v1 BuildTargets skips it), and a group with no valid count computes no weights at all, every
// weight stays 0: v1 sends no request and plans nothing. No uniform fallback, for every family.
inline void AssignSoftAnnealingWeights(
    const std::vector<RelationPayload>& payloads,
    const std::vector<size_t>& ids, double eta, double decay_gamma,
    double epsilon, std::vector<double>& weights) {
  if (ids.empty()) return;
  std::vector<size_t> valid;
  std::vector<double> steps;
  valid.reserve(ids.size());
  steps.reserve(ids.size());
  for (size_t idx : ids) {
    const double count = SchedulingScore(payloads[idx]);
    if (!HasValidUpdateCount(count)) {
      weights[idx] = 0.0;
      continue;
    }
    valid.push_back(idx);
    steps.push_back(count);
  }
  if (valid.empty()) return;  // no valid count: no weights (v1), every weight is already 0
  const std::vector<double> annealed = spatialblend::ComputeSoftAnnealingWeights(
      steps, eta, /*enable_decay=*/true, decay_gamma, epsilon);
  for (size_t k = 0; k < valid.size(); ++k) weights[valid[k]] = annealed[k];
}

inline std::vector<double> ComputeAnnealedWeights(
    const std::vector<RelationPayload>& payloads,
    const AnnealWeightConfig& cfg = {}) {
  const size_t n = payloads.size();
  if (n == 0) return {};

  // Group by (kernel family, subject anchor) so spacing/facing/affordance never
  // share a temperature budget.
  std::map<ChannelKey, std::vector<size_t>> groups;
  for (size_t i = 0; i < n; ++i) {
    groups[ChannelKeyOf(payloads[i])].push_back(i);
  }

  std::vector<double> weights(n, 0.0);
  for (const auto& [key, idx] : groups) {
    AssignSoftAnnealingWeights(payloads, idx, cfg.eta, cfg.decay_gamma,
                               cfg.epsilon_t, weights);
  }
  return weights;
}

inline std::vector<double> ComputeRealizationWeights(
    const std::vector<RelationPayload>& payloads,
    const RealizationPolicyConfig& config = {}) {
  const size_t n = payloads.size();
  if (n == 0) return {};
  // A lone relation is no special case: Progressive weighs it 1 exactly when its count is valid (v1 SoftAnnealing over
  // one valid count), and 0 otherwise (no request, as v1).

  // Group by (kernel family, subject anchor) so different relation families on
  // the same actor part never share a weight budget.
  std::map<ChannelKey, std::vector<size_t>> groups;
  for (size_t i = 0; i < n; ++i) {
    if (!payloads[i].enabled) continue;
    groups[ChannelKeyOf(payloads[i])].push_back(i);
  }

  auto greedy_best = [&](size_t a, size_t b) {
    const double a_time = SchedulingScore(payloads[a]);
    const double b_time = SchedulingScore(payloads[b]);
    const bool a_finite = HasValidUpdateCount(a_time);
    const bool b_finite = HasValidUpdateCount(b_time);

    if (a_finite && !b_finite) return true;
    if (!a_finite && b_finite) return false;

    if (!a_finite) {
      if (payloads[a].referent_shared_id != payloads[b].referent_shared_id) {
        return payloads[a].referent_shared_id < payloads[b].referent_shared_id;
      }
      return a < b;
    }

    if (a_time < b_time) return true;
    if (a_time > b_time) return false;
    if (payloads[a].referent_shared_id != payloads[b].referent_shared_id) {
      return payloads[a].referent_shared_id < payloads[b].referent_shared_id;
    }
    return a < b;
  };

  auto progressive = [&](const std::vector<size_t>& ids,
                         std::vector<double>& weights) {
    AssignSoftAnnealingWeights(payloads, ids, config.eta_0, config.decay_gamma,
                               config.epsilon, weights);
  };

  // Baseline policies, including deterministic greedy tie-breaks.
  if (config.type == RealizationPolicyType::kUniform) {
    std::vector<double> weights(n, 0.0);
    for (const auto& group : groups) {
      const auto& ids = group.second;
      const double uniform = 1.0 / static_cast<double>(ids.size());
      for (size_t idx : ids) {
        weights[idx] = uniform;
      }
    }
    return weights;
  }
  if (config.type == RealizationPolicyType::kGreedy) {
    std::vector<double> weights(n, 0.0);
    for (const auto& group : groups) {
      const auto& ids = group.second;
      if (ids.empty()) continue;
      size_t best = ids[0];
      for (size_t idx : ids) {
        if (greedy_best(idx, best)) {
          best = idx;
        }
      }
      weights[best] = 1.0;
    }
    return weights;
  }

  std::vector<double> weights(n, 0.0);
  for (const auto& group : groups) {
    const auto& ids = group.second;
    progressive(ids, weights);
  }
  return weights;
}

// ---------------------------------------------------------------------------
// Per-channel policy diagnostics (the policy's driving relation; the
// realized one is annotated by Space after gating).
//
// A channel is one (kernel family, subject anchor) group -- exactly the
// groups ComputeRealizationWeights normalizes over (enabled payloads only).
// The DRIVING relation of a channel is its argmax-weight payload; ties are
// broken by the lowest payload index (the order the receiver saw them) and
// flagged via `driving_tied`. Pure: reads the weights the policy produced, it
// never recomputes them, so the diagnostics cannot disagree with the solve.
struct RealizationChannelDiagnostics {
  RelationKernelType kernel_type = RelationKernelType::kAvatarFacing;
  std::string anchor_name;
  std::vector<size_t> payload_indices;  // enabled members, ascending
  // Progressive only (NaN otherwise, or when no member has a valid T_p):
  double t_min = std::numeric_limits<double>::quiet_NaN();
  double eta = std::numeric_limits<double>::quiet_NaN();
  // POLICY argmax: what the weights prefer, before the solver and the
  // anchor gate run.
  bool has_driving = false;
  size_t driving_index = 0;  // index into the payload vector
  double driving_weight = 0.0;
  bool driving_tied = false;
  // REALIZED driving relation: filled by Space after the solve and the
  // anchor gate (see AnnotateRealizedDriving in space.cc). False when the channel
  // produced no realized target (e.g. gated out, or every goal unresolved).
  bool has_realized = false;
  size_t realized_index = 0;
  double realized_weight = 0.0;
};

inline std::vector<RealizationChannelDiagnostics> DescribeRealizationChannels(
    const std::vector<RelationPayload>& payloads,
    const std::vector<double>& weights,
    const RealizationPolicyConfig& config) {
  std::map<ChannelKey, std::vector<size_t>> groups;
  for (size_t i = 0; i < payloads.size(); ++i) {
    if (!payloads[i].enabled) continue;
    groups[ChannelKeyOf(payloads[i])].push_back(i);
  }
  std::vector<RealizationChannelDiagnostics> out;
  out.reserve(groups.size());
  for (const auto& [key, ids] : groups) {
    RealizationChannelDiagnostics d;
    d.kernel_type = static_cast<RelationKernelType>(std::get<0>(key));
    d.anchor_name = std::get<1>(key);
    d.payload_indices = ids;
    if (config.type == RealizationPolicyType::kProgressive) {
      double t_min = std::numeric_limits<double>::infinity();
      for (size_t idx : ids) {
        const double score = SchedulingScore(payloads[idx]);
        if (HasValidUpdateCount(score)) t_min = std::min(t_min, score);
      }
      // A single-member channel is weighted 1 without a temperature. η is
      // v1's η_eff = η_0 (n_min − 1 + ε)^γ, unclipped.
      if (std::isfinite(t_min)) {
        d.t_min = t_min;
        if (ids.size() > 1) {
          d.eta = config.eta_0 *
                  std::pow(t_min - 1.0 + config.epsilon, config.decay_gamma);
        }
      }
    }
    for (size_t idx : ids) {
      const double w = idx < weights.size() ? weights[idx] : 0.0;
      if (!(w > 0.0)) continue;
      if (!d.has_driving || w > d.driving_weight) {
        d.has_driving = true;
        d.driving_index = idx;
        d.driving_weight = w;
        d.driving_tied = false;
      } else if (w == d.driving_weight) {
        d.driving_tied = true;
      }
    }
    out.push_back(std::move(d));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Per-anchor hysteresis gating.
//
// Affordance-anchor targets (e.g. a hand target on a tabletop) must not
// flicker on/off as the driving proximity signal bounces around a single
// threshold. UpdateAnchorGate applies an EMA to the raw weight*proximity
// signal and a Schmitt trigger (enter/exit) on top of the smoothed value, so
// a target that is "active" stays active until the signal has clearly
// dropped, and one that is "inactive" stays inactive until it has clearly
// risen. State is per (actor, anchor) and owned by the caller (Space) —
// this function is pure and holds no global state.
struct AnchorGateState {
  double ema = 0.0;
  bool active = false;
  bool initialized = false;
};

struct AnchorGateConfig {
  double enter = 0.6;
  double exit = 0.4;
  double alpha = 0.3;
};

// Updates `state` in place from one new raw sample and returns whether the
// anchor should be considered active (i.e. its target should be emitted)
// this frame. Cold-start seeds the EMA to the first raw sample so a single
// strong frame can gate on immediately rather than being damped toward 0.
inline bool UpdateAnchorGate(AnchorGateState& state, double raw_value,
                              const AnchorGateConfig& config = AnchorGateConfig()) {
  if (!state.initialized) {
    state.ema = raw_value;
    state.initialized = true;
  } else {
    state.ema = (1.0 - config.alpha) * state.ema + config.alpha * raw_value;
  }

  if (!state.active && state.ema >= config.enter) {
    state.active = true;
  } else if (state.active && state.ema <= config.exit) {
    state.active = false;
  }
  return state.active;
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_REALIZATION_POLICY_H_
