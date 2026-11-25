// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// C ABI contract tests for spatialblend_observability.h.
// Each test names the contract it checks.
// Fixtures are small and analytic: every asserted number is derived in the
// comment next to it, not read back from the implementation.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "spatialblend/api/c/c_test_embodied.h"
#include "spatialblend/model/realization/remote_update.h"
#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/api/c/spatialblend_receiver_input.h"

namespace {

constexpr double kTol = 1e-9;

template <typename T>
T Init() {
  T value;
  std::memset(&value, 0, sizeof(T));
  value.struct_size = static_cast<uint32_t>(sizeof(T));
  value.version = SB_OBSERVABILITY_VERSION;
  return value;
}

std::string TempPath(const char* stem) {
  const char* dir = std::getenv("TEST_TMPDIR");
  std::string path = std::string(dir ? dir : "/tmp") + "/" + stem;
  std::remove(path.c_str());
  return path;
}

std::vector<nlohmann::json> ReadTrace(const std::string& path) {
  std::vector<nlohmann::json> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) out.push_back(nlohmann::json::parse(line));
  }
  return out;
}

void Translate(sb_space_t space, sb_element_t elem, double x, double y,
               double z) {
  const double m[16] = {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1};
  sb_set_element_transform(space, elem, m);
}

// Receiver fixture: a 10x10 navigable room with geometry-less referents,
// whose decoded spacing witness is exactly their world position.
struct Receiver {
  sb_session_t session = 0;
  sb_space_t space = 0;
  sb_element_t room = 0;
  sb_element_t actor_elem = 0;
  sb_actor_t actor = 0;

  Receiver() {
    session = sb_create_session();
    space = sb_add_space(session);
    EXPECT_EQ(sb_set_observability_capture(space, 1), 1);
    const double room_poly[] = {0, 0, 10, 0, 10, 10, 0, 10};
    room = sb_add_element(space, 0);
    sb_set_element_shared_id(space, room, 100);
    sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, room_poly, 4);
    actor_elem = sb_add_element(space, room);
    sb_set_element_shared_id(space, actor_elem, 900);
    actor = sb_create_actor(space);
    spatialblend_c_test::PublishStartPose(space, actor);
    sb_add_anchor(space, actor, "root", 0, 0, 0, nullptr);
    sb_add_standard_anchors(space, actor);
    sb_set_actor_element(space, actor, actor_elem);
  }
  ~Receiver() { sb_destroy_session(session); }

  // An affordance referent needs support geometry (a
  // geometryless referent is a grounding failure, never a target at its
  // origin). A 0.4 m square surface centred on (x, y) at height z.
  sb_element_t Surface(int64_t shared_id, double x, double y, double z) {
    sb_element_t e = sb_add_element(space, room);
    sb_set_element_shared_id(space, e, shared_id);
    const double v[] = {x - 0.2, y - 0.2, z, x + 0.2, y - 0.2, z,
                        x + 0.2, y + 0.2, z, x - 0.2, y + 0.2, z};
    sb_set_element_surface(space, e, v, 4, SB_GEOMETRY_SURFACE);
    return e;
  }

  sb_element_t Point(int64_t shared_id, double x, double y, double z = 0) {
    sb_element_t e = sb_add_element(space, room);
    sb_set_element_shared_id(space, e, shared_id);
    Translate(space, e, x, y, z);
    return e;
  }
};

sb_relation_payload_t Payload(int64_t referent, sb_kernel_type_t kernel,
                              const char* anchor, double proximity,
                              double estimated_time);

// An AffordancePose payload engaged at the support's chart origin
// (engaged = metric support distance within the anchor's enter distance).
sb_relation_payload_t Contact(int64_t referent, const char* anchor,
                              double proximity, double support_distance_m) {
  // The count (the ABI's `estimated_time`): 1 in contact (v1), else 2 (an approach). A count of 0 is no valid count:
  // Progressive computes no weight for it and it makes no request.
  sb_relation_payload_t p = Payload(referent, SB_KERNEL_AFFORDANCE_POSE,
                                    anchor, proximity,
                                    support_distance_m < spatialblend::kContactEpsilonM ? 1.0 : 2.0);
  p.has_contact = 1;
  p.normalized_strategy = SB_NORMALIZED_RADIAL;
  p.chart_version = 1;
  p.support_distance_m = support_distance_m;
  p.encoded_state_size = 3;
  return p;
}

sb_relation_payload_t Payload(int64_t referent, sb_kernel_type_t kernel,
                              const char* anchor, double proximity,
                              double estimated_time) {
  auto p = Init<sb_relation_payload_t>();
  p.referent_shared_id = referent;
  p.kernel_type = kernel;
  p.anchor_name = anchor;
  p.proximity = proximity;
  p.estimated_time = estimated_time;
  p.support_distance_m = -1;
  p.source_scope_diameter_m = -1;
  p.enabled = 1;
  return p;
}

std::vector<sb_realization_weight_t> Weights(const Receiver& r) {
  const int32_t n = sb_get_realization_weights(
      r.space, r.actor, nullptr, 0, sizeof(sb_realization_weight_t));
  std::vector<sb_realization_weight_t> out(std::max(n, 0),
                                           Init<sb_realization_weight_t>());
  if (n > 0) {
    EXPECT_EQ(sb_get_realization_weights(r.space, r.actor, out.data(), n,
                                         sizeof(sb_realization_weight_t)),
              n);
  }
  return out;
}

std::vector<sb_realization_channel_t> Channels(const Receiver& r) {
  const int32_t n = sb_get_realization_channels(
      r.space, r.actor, nullptr, 0, sizeof(sb_realization_channel_t));
  std::vector<sb_realization_channel_t> out(std::max(n, 0),
                                            Init<sb_realization_channel_t>());
  if (n > 0) {
    EXPECT_EQ(sb_get_realization_channels(r.space, r.actor, out.data(), n,
                                          sizeof(sb_realization_channel_t)),
              n);
  }
  return out;
}

// v1 SoftAnnealing weight of the later of two members.
// η_eff = η_0 (n_min − 1 + ε)^γ. Below ε the minimum takes the whole weight.
double ProgressiveSecond(double n_min, double n2, double gamma) {
  const double eta = std::pow(n_min - 1.0 + 1e-3, gamma);
  const double safe = std::max(eta, 1e-6);
  if (safe < 1e-3) return 0.0;
  const double e = std::exp(-(n2 - n_min) / safe);
  return e / (1.0 + e);
}

}  // namespace

// ---------------------------------------------------------------------------
// ABI: sizes, versions, per-field offsets.
// ---------------------------------------------------------------------------

TEST(ObservabilityAbiTest, SizesAndOffsetsMatchTheHeader) {
  EXPECT_EQ(sb_get_observability_version(), SB_OBSERVABILITY_VERSION);
  EXPECT_EQ(sb_get_trace_format_version(), 10u);  // format 10: + scope.reason inside_transfer_leaf
  EXPECT_EQ(sb_sizeof(SB_STRUCT_RELATION_STATE), sizeof(sb_relation_state_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_RELATION_STATE_V2),
            sizeof(sb_relation_state_v2_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_SCOPE_RECORD), sizeof(sb_scope_record_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_SCOPE_CANDIDATE), sizeof(sb_scope_candidate_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_REALIZATION_POLICY_PARAMS),
            sizeof(sb_realization_policy_params_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_REALIZATION_WEIGHT),
            sizeof(sb_realization_weight_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_REALIZATION_CHANNEL),
            sizeof(sb_realization_channel_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_SOLVER_DIAGNOSTICS),
            sizeof(sb_solver_diagnostics_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_SOLVER_GOAL), sizeof(sb_solver_goal_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_RELATION_PAYLOAD),
            sizeof(sb_relation_payload_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_RELATION_QUERY), sizeof(sb_relation_query_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_RELATION_EVALUATION),
            sizeof(sb_relation_evaluation_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_REALIZATION_TARGET_V2),
            sizeof(sb_realization_target_v2_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_V2_ABI_LAYOUT), sizeof(sb_v2_abi_layout_t));
  // Legacy ids unchanged.
  EXPECT_EQ(sb_sizeof(SB_STRUCT_REALIZATION_TARGET),
            sizeof(sb_realization_target_t));

  // Offsets are declaration-ordered, strictly increasing, and the last field
  // ends inside the struct.
  for (int id : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23,
                 24, 25}) {
    const auto sid = static_cast<sb_struct_id_t>(id);
    const uint32_t n = sb_get_struct_field_count(sid);
    ASSERT_GT(n, 0u) << "struct id " << id;
    uint32_t prev = 0;
    for (uint32_t f = 0; f < n; ++f) {
      const uint32_t off = sb_get_struct_field_offset(sid, f);
      ASSERT_NE(off, UINT32_MAX);
      if (f > 0) EXPECT_GT(off, prev) << "struct id " << id << " field " << f;
      prev = off;
    }
    EXPECT_LT(prev, sb_sizeof(sid)) << "struct id " << id;
    EXPECT_EQ(sb_get_struct_field_offset(sid, n), UINT32_MAX);
  }
  EXPECT_EQ(sb_get_struct_field_offset(SB_STRUCT_SCOPE_CANDIDATE, 5),
            offsetof(sb_scope_candidate_t, reason));
  EXPECT_EQ(sb_get_struct_field_count(static_cast<sb_struct_id_t>(99)), 0u);

  // The legacy layout descriptor is untouched by this surface.
  sb_v2_abi_layout_t layout{};
  EXPECT_EQ(sb_get_v2_abi_layout(&layout, sizeof(layout)), sizeof(layout));
  EXPECT_EQ(layout.realization_target_size, sizeof(sb_realization_target_t));
}

TEST(ObservabilityAbiTest, WrongStructSizeIsRejectedWithoutWriting) {
  Receiver r;
  sb_realization_weight_t w = Init<sb_realization_weight_t>();
  EXPECT_EQ(sb_get_realization_weights(r.space, r.actor, &w, 1,
                                       sizeof(sb_realization_weight_t) - 8),
            -1);
  sb_realization_policy_params_t p = Init<sb_realization_policy_params_t>();
  EXPECT_EQ(sb_get_realization_policy_params(r.space, &p, sizeof(p) + 8), 0);
  auto payload = Payload(1, SB_KERNEL_SPACING, "root", 1, 0);
  EXPECT_EQ(sb_inject_relation_payloads(r.space, r.actor, &payload, 1,
                                        sizeof(payload) - 4, 0.1),
            0);
  EXPECT_EQ(sb_inject_relation_payloads(r.space, 9999, &payload, 1,
                                        sizeof(payload), 0.1),
            0);
  EXPECT_EQ(sb_inject_relation_payloads(r.space, r.actor, &payload, 1,
                                        sizeof(payload), NAN),
            0);
}

// ---------------------------------------------------------------------------
// Declared policy parameters.
// ---------------------------------------------------------------------------

TEST(PolicyParamsTest, R3_5_DefaultsAreDeclaredAndSettable) {
  auto d = Init<sb_realization_policy_params_t>();
  ASSERT_EQ(sb_get_default_realization_policy_params(&d, sizeof(d)), 1);
  // Production values. γ = 1 is v1 SoftAnnealing.
  EXPECT_DOUBLE_EQ(d.gamma, 1.0);
  EXPECT_DOUBLE_EQ(d.eta_0, 1.0);
  EXPECT_DOUBLE_EQ(d.epsilon_t, 1e-3);
  EXPECT_DOUBLE_EQ(d.eta_min, 1e-6);
  EXPECT_DOUBLE_EQ(d.eta_max, 100.0);
  EXPECT_DOUBLE_EQ(d.progress_tau_s, 0.092517);
  EXPECT_DOUBLE_EQ(d.epsilon_v, 0.01);
  EXPECT_DOUBLE_EQ(d.t_max, 100.0);
  EXPECT_DOUBLE_EQ(d.kappa, 0.2);
  EXPECT_EQ(d.policy, SB_REALIZATION_POLICY_PROGRESSIVE);

  Receiver r;
  auto p = Init<sb_realization_policy_params_t>();
  ASSERT_EQ(sb_get_realization_policy_params(r.space, &p, sizeof(p)), 1);
  EXPECT_DOUBLE_EQ(p.gamma, 1.0);

  p.gamma = 1.0;
  p.kappa = 0.5;
  ASSERT_EQ(sb_set_realization_policy_params(r.space, &p, sizeof(p)), 1);
  auto q = Init<sb_realization_policy_params_t>();
  ASSERT_EQ(sb_get_realization_policy_params(r.space, &q, sizeof(q)), 1);
  EXPECT_DOUBLE_EQ(q.gamma, 1.0);
  EXPECT_DOUBLE_EQ(q.kappa, 0.5);

  // Invalid values change nothing.
  auto bad = q;
  bad.eta_max = bad.eta_min / 2;
  EXPECT_EQ(sb_set_realization_policy_params(r.space, &bad, sizeof(bad)), 0);
  bad = q;
  bad.kappa = 0;
  EXPECT_EQ(sb_set_realization_policy_params(r.space, &bad, sizeof(bad)), 0);
  bad = q;
  bad.gamma = NAN;
  EXPECT_EQ(sb_set_realization_policy_params(r.space, &bad, sizeof(bad)), 0);
  auto after = Init<sb_realization_policy_params_t>();
  ASSERT_EQ(sb_get_realization_policy_params(r.space, &after, sizeof(after)),
            1);
  EXPECT_DOUBLE_EQ(after.gamma, 1.0);
  EXPECT_DOUBLE_EQ(after.eta_max, q.eta_max);

  // The family stays owned by sb_set_realization_policy.
  sb_set_realization_policy(r.space, SB_REALIZATION_POLICY_GREEDY);
  ASSERT_EQ(sb_get_realization_policy_params(r.space, &after, sizeof(after)),
            1);
  EXPECT_EQ(after.policy, SB_REALIZATION_POLICY_GREEDY);
}

// ---------------------------------------------------------------------------
// Payload injection drives the policy deterministically; the
// driving relation is the per-channel argmax.
// ---------------------------------------------------------------------------

TEST(PayloadInjectionTest, X2_R3_6_InjectedTimeSeriesDrivesWeights) {
  Receiver r;
  r.Point(301, 2, 2);
  r.Point(302, 8, 8);

  // Two same-channel spacing relations, T_1 = 1 s, T_2 = 3 s.
  sb_relation_payload_t in[2] = {
      Payload(301, SB_KERNEL_SPACING, "root", 0.9, 1.0),
      Payload(302, SB_KERNEL_SPACING, "root", 0.5, 3.0)};
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 2, sizeof(in[0]),
                                        0.1),
            1);

  auto w = Weights(r);
  ASSERT_EQ(w.size(), 2u);
  // Progressive, γ = 1: η = (n_min − 1 + ε)^γ. n_min = 1 collapses to one-hot.
  const double w2 = ProgressiveSecond(1.0, 3.0, 1.0);
  EXPECT_NEAR(w[1].weight, w2, kTol);
  EXPECT_NEAR(w[0].weight, 1.0 - w2, kTol);
  EXPECT_EQ(w[0].referent_shared_id, 301);
  EXPECT_EQ(w[0].is_driving, 1);
  EXPECT_EQ(w[1].is_driving, 0);
  EXPECT_EQ(w[0].channel_index, w[1].channel_index);
  EXPECT_DOUBLE_EQ(w[0].estimated_time, 1.0);
  EXPECT_DOUBLE_EQ(w[1].proximity, 0.5);

  auto ch = Channels(r);
  ASSERT_EQ(ch.size(), 1u);
  EXPECT_EQ(ch[0].policy, SB_REALIZATION_POLICY_PROGRESSIVE);
  EXPECT_EQ(ch[0].has_driving, 1);
  EXPECT_EQ(ch[0].driving_referent_shared_id, 301);
  EXPECT_EQ(ch[0].driving_payload_index, 0);
  EXPECT_EQ(ch[0].candidate_count, 2);
  EXPECT_DOUBLE_EQ(ch[0].t_min, 1.0);
  EXPECT_NEAR(ch[0].eta, 1e-3, kTol);
  EXPECT_EQ(ch[0].driving_tied, 0);

  // Setting γ = 1 explicitly keeps the same v1 temperature.
  auto p = Init<sb_realization_policy_params_t>();
  ASSERT_EQ(sb_get_realization_policy_params(r.space, &p, sizeof(p)), 1);
  p.gamma = 1.0;
  ASSERT_EQ(sb_set_realization_policy_params(r.space, &p, sizeof(p)), 1);
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 2, sizeof(in[0]),
                                        0.2),
            1);
  w = Weights(r);
  EXPECT_NEAR(w[1].weight, ProgressiveSecond(1.0, 3.0, 1.0), kTol);
  EXPECT_NEAR(Channels(r)[0].eta, 1e-3, kTol);

  // Reversing the time series moves the driving relation.
  in[0].estimated_time = 5.0;
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 2, sizeof(in[0]),
                                        0.3),
            1);
  ch = Channels(r);
  EXPECT_EQ(ch[0].driving_referent_shared_id, 302);
  EXPECT_EQ(ch[0].driving_payload_index, 1);

  // Greedy: one-hot on the smaller T_p.
  sb_set_realization_policy(r.space, SB_REALIZATION_POLICY_GREEDY);
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 2, sizeof(in[0]),
                                        0.4),
            1);
  w = Weights(r);
  EXPECT_DOUBLE_EQ(w[0].weight, 0.0);
  EXPECT_DOUBLE_EQ(w[1].weight, 1.0);
  ch = Channels(r);
  EXPECT_EQ(ch[0].policy, SB_REALIZATION_POLICY_GREEDY);
  EXPECT_TRUE(std::isnan(ch[0].eta));

  // Uniform: a tie, broken by the first payload and flagged.
  sb_set_realization_policy(r.space, SB_REALIZATION_POLICY_UNIFORM);
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 2, sizeof(in[0]),
                                        0.5),
            1);
  ch = Channels(r);
  EXPECT_DOUBLE_EQ(ch[0].driving_weight, 0.5);
  EXPECT_EQ(ch[0].driving_payload_index, 0);
  EXPECT_EQ(ch[0].driving_tied, 1);
}

TEST(PayloadInjectionTest, R3_4_R3_6_ChannelsAreSeparatedByAnchorAndFamily) {
  Receiver r;
  r.Point(301, 2, 2);
  r.Point(302, 8, 8);
  sb_relation_payload_t in[3] = {
      Payload(301, SB_KERNEL_SPACING, "root", 0.9, 1.0),
      Payload(302, SB_KERNEL_SPACING, "root", 0.5, 3.0),
      // A lone relation on another anchor: its own channel, weight 1 (in contact: v1's count 1; a count of 0 is no
      // valid count, no weight).
      Payload(302, SB_KERNEL_AFFORDANCE_POSE, "hand", 1.0, 1.0)};
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 3, sizeof(in[0]),
                                        0.1),
            1);
  auto ch = Channels(r);
  ASSERT_EQ(ch.size(), 2u);
  int hand = std::string(ch[0].anchor_name) == "hand" ? 0 : 1;
  EXPECT_EQ(ch[hand].kernel_type, SB_KERNEL_AFFORDANCE_POSE);
  EXPECT_EQ(ch[hand].driving_referent_shared_id, 302);
  EXPECT_DOUBLE_EQ(ch[hand].driving_weight, 1.0);
  EXPECT_EQ(ch[1 - hand].driving_referent_shared_id, 301);
  auto w = Weights(r);
  EXPECT_NEAR(w[0].weight, 1.0 - ProgressiveSecond(1.0, 3.0, 1.0), kTol);
}

// ---------------------------------------------------------------------------
// Witness points before blending, unconstrained blend, feasibility snap.
// ---------------------------------------------------------------------------

TEST(SolverDiagnosticsTest, X1_WitnessesBlendAndSnap) {
  Receiver r;
  r.Point(301, 2, 2);
  r.Point(302, 8, 8);
  sb_relation_payload_t in[2] = {
      Payload(301, SB_KERNEL_SPACING, "root", 0.9, 1.0),
      Payload(302, SB_KERNEL_SPACING, "root", 0.5, 3.0)};
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 2, sizeof(in[0]),
                                        0.1),
            1);
  auto diag = Init<sb_solver_diagnostics_t>();
  ASSERT_EQ(sb_get_solver_diagnostics(r.space, r.actor, &diag, sizeof(diag)),
            1);
  // n = 1 is contact, so that referent takes the whole weight. The first injection is the first
  // proposal, PLACED, and placement reads every position relation whatever its weight:
  // the zero-weight referent is a goal too, with weight 0 (it breaks no tie here and moves no blend).
  ASSERT_EQ(diag.goal_count, 2);
  std::vector<sb_solver_goal_t> goals(2, Init<sb_solver_goal_t>());
  ASSERT_EQ(sb_get_solver_goals(r.space, r.actor, goals.data(), 2,
                                sizeof(sb_solver_goal_t)),
            2);
  EXPECT_EQ(goals[1].referent_shared_id, 302);
  EXPECT_EQ(goals[1].weight, 0.0);
  // A geometry-less referent's witness is its world position.
  EXPECT_NEAR(goals[0].witness_x, 2.0, kTol);
  EXPECT_NEAR(goals[0].witness_y, 2.0, kTol);
  EXPECT_EQ(goals[0].referent_shared_id, 301);
  EXPECT_EQ(goals[0].position_goal, 1);
  // The contact takes the whole weight, so the blend is its own point.
  const double w2 = ProgressiveSecond(1.0, 3.0, 1.0);
  EXPECT_NEAR(goals[0].weight, 1.0 - w2, kTol);
  EXPECT_EQ(diag.has_unconstrained, 1);
  EXPECT_NEAR(diag.unconstrained_x, 2.0 + 6.0 * w2, 1e-9);
  EXPECT_NEAR(diag.unconstrained_y, 2.0 + 6.0 * w2, 1e-9);
  EXPECT_EQ(diag.snapped, 0);
  EXPECT_NEAR(diag.feasibility_snap_m, 0.0, 1e-9);
  // This test's Payload() helper never sets encoded_state_size (it stays
  // 0), so these payloads never went through a real Extract_p and are
  // explicitly EXCLUDED from the paper-objective refinement (an empty
  // encoded_state opts out, since `proximity` alone cannot be
  // trusted as a real R^src measurement) -- `solved_x/y` is therefore
  // exactly the centroid `unconstrained_x/y`, unchanged by the refinement.
  EXPECT_NEAR(diag.solved_x, diag.unconstrained_x, 1e-6);
  const double first_x = diag.solved_x, first_y = diag.solved_y;

  // A later injection is a continuation (the pursued target is the live rule; the placed
  // answer is the first proposal's only). This payload has no update count, so it issues no request: the anchor
  // stays at its current position, the sample (0, 0) (v1's hold), and its snap is measured
  // from there; the outside witness is still the unconstrained point. It is not a teleport.
  r.Point(303, 12, 12);
  auto out = Payload(303, SB_KERNEL_SPACING, "root", 0.5, 1.0);
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, &out, 1,
                                        sizeof(out), 0.2),
            1);
  ASSERT_EQ(sb_get_solver_diagnostics(r.space, r.actor, &diag, sizeof(diag)),
            1);
  EXPECT_NEAR(diag.unconstrained_x, 12.0, kTol);
  EXPECT_NEAR(diag.unconstrained_y, 12.0, kTol);
  EXPECT_EQ(diag.snapped, 0);
  EXPECT_NEAR(diag.solved_x, 0.0, 1e-9) << "the hold is the sample, not the first answer (" << first_x << ")";
  EXPECT_NEAR(diag.solved_y, 0.0, 1e-9) << "the hold is the sample, not the first answer (" << first_y << ")";
  EXPECT_NEAR(diag.feasibility_snap_m, std::hypot(12.0, 12.0), 1e-6);
}

// The variant of X1_WitnessesBlendAndSnap whose payloads carry a real
// encoded_state, so they ENTER the paper-objective refinement (an empty
// encoded_state opts a payload out, which is all the test above exercises).
//
// Geometry: a 10 x 10 room (diagonal D = 10 sqrt 2), geometry-less referents
// at (2, 2) and (8, 8), whose R^tgt(S) = 1 - |S - c| / D. Sources R^src =
// 0.9 and 0.5 ask for distances 0.1 D = sqrt 2 from the first and 0.5 D =
// 5 sqrt 2 from the second; those sum to |c1 - c2| = 6 sqrt 2, so the exact
// relation-preserving state is the single point (3, 3) on the segment
// between them with J = 0 -- for ANY policy weights. The goal-point centroid
// (2 + 6 w2 on both axes) is not that point, so only the refinement reaches
// it: this is a counterexample through the C ABI.
TEST(SolverDiagnosticsTest, X1_PopulatedEncodedStateReachesTheExactState) {
  Receiver r;
  r.Point(301, 2, 2);
  r.Point(302, 8, 8);
  sb_relation_payload_t in[2] = {
      Payload(301, SB_KERNEL_SPACING, "root", 0.9, 1.0),
      Payload(302, SB_KERNEL_SPACING, "root", 0.5, 3.0)};
  for (auto& p : in) {
    p.encoded_state_size = 2;  // (0, 0): the support-frame offset
    p.encoded_state[0] = 0.0;
    p.encoded_state[1] = 0.0;
  }
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 2, sizeof(in[0]),
                                        0.1),
            1);
  auto diag = Init<sb_solver_diagnostics_t>();
  ASSERT_EQ(sb_get_solver_diagnostics(r.space, r.actor, &diag, sizeof(diag)),
            1);
  // The earlier decode is unchanged: witnesses at the referents, blended.
  const double w2 = ProgressiveSecond(1.0, 3.0, 1.0);
  EXPECT_EQ(diag.goal_count, 2) << "the placed first proposal reads the zero-weight relation too";
  EXPECT_NEAR(diag.unconstrained_x, 2.0 + 6.0 * w2, 1e-9);
  EXPECT_NEAR(diag.unconstrained_y, 2.0 + 6.0 * w2, 1e-9);
  // There is NO refinement: the answer is S0, the
  // weighted point of the decoded approach points, clamped to free space. Here the
  // centroid is inside the room, so solved == unconstrained, and the relations are
  // only approximately preserved (J(centroid) is reported, not minimised).
  const double D = std::hypot(10.0, 10.0);
  auto objective = [&](double x, double y) {
    const double r1 = 1.0 - std::hypot(x - 2.0, y - 2.0) / D;
    const double r2 = 1.0 - std::hypot(x - 8.0, y - 8.0) / D;
    return (1.0 - w2) * (r1 - 0.9) * (r1 - 0.9) +
           w2 * (r2 - 0.5) * (r2 - 0.5);
  };
  EXPECT_NEAR(diag.solved_x, diag.unconstrained_x, 1e-9);
  EXPECT_NEAR(diag.solved_y, diag.unconstrained_y, 1e-9);
  EXPECT_GT(objective(diag.solved_x, diag.solved_y), 1e-6)
      << "the centroid does not preserve the relations and is not refined";
}

// Reach never moves an engaged target; one beyond the anchor's
// reach from the solved root is flagged (formerly: clamped to the reach).
TEST(SolverDiagnosticsTest, X1_BeyondReachIsObservable) {
  Receiver r;
  r.Point(301, 2, 2);
  r.Surface(310, 9, 1, 0.8);
  sb_relation_payload_t in[2] = {
      Payload(301, SB_KERNEL_SPACING, "root", 1.0, 0.0),
      Contact(310, "hand", 1.0, 0.0)};
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 2, sizeof(in[0]),
                                        0.1),
            1);
  const int32_t n = sb_get_realization_targets_v2(
      r.space, r.actor, nullptr, 0, sizeof(sb_realization_target_v2_t));
  ASSERT_GE(n, 2);
  std::vector<sb_realization_target_v2_t> t(n,
                                            Init<sb_realization_target_v2_t>());
  ASSERT_EQ(sb_get_realization_targets_v2(r.space, r.actor, t.data(), n,
                                          sizeof(sb_realization_target_v2_t)),
            n);
  EXPECT_STREQ(t[0].anchor_name, "root");
  EXPECT_EQ(t[0].beyond_reach, 0);
  const sb_realization_target_v2_t* hand = nullptr;
  for (const auto& x : t)
    if (std::string(x.anchor_name) == "hand") hand = &x;
  ASSERT_NE(hand, nullptr);
  // Root at (2, 2); the contact (9, 1) is sqrt(50) m away, beyond the 0.9 m
  // default hand reach: the target stays ON the contact and is flagged.
  const double dist =
      std::hypot(9.0 - t[0].position_x, 1.0 - t[0].position_y);
  EXPECT_EQ(hand->beyond_reach, 1);
  EXPECT_NEAR(hand->position_x, 9.0, 1e-9);
  EXPECT_NEAR(hand->position_y, 1.0, 1e-9);
  EXPECT_NEAR(hand->requested_x, 9.0, 1e-9);
  EXPECT_NEAR(hand->reach_excess_m, dist - 0.9, 1e-6);
  EXPECT_EQ(hand->has_z, 1);
  EXPECT_NEAR(hand->position_z, 0.8, 1e-9);
}

// The target-side Eval_p through the C ABI -- R^tgt, R^src,
// error, disposition, and the pre-chart_version payload layout.
TEST(TargetEvaluationTest, R1_3_EvaluatesAtACandidateState) {
  Receiver r;
  r.Surface(310, 5, 5, 0.8);
  // The source hand was 0.15 m above its support: R^src = 1/1.15.
  sb_relation_payload_t p = Contact(310, "hand", 1.0 / 1.15, 0.15);
  sb_relation_query_t q = Init<sb_relation_query_t>();
  q.anchor_x = 5.05;
  q.anchor_y = 4.9;
  q.anchor_z = 1.2;  // 0.4 m above the surface
  sb_target_relation_evaluation_t out =
      Init<sb_target_relation_evaluation_t>();
  ASSERT_EQ(sb_evaluate_target_relation(r.space, &p, sizeof(p), &q, &out), 1);
  EXPECT_EQ(out.disposition, SB_TARGET_EVAL_EVALUATED);
  EXPECT_NEAR(out.r_tgt, 1.0 / 1.4, 1e-9);
  EXPECT_NEAR(out.r_src, 1.0 / 1.15, 1e-12);
  EXPECT_NEAR(out.error, 1.0 / 1.4 - 1.0 / 1.15, 1e-9);
  EXPECT_NEAR(out.contact_point_world_z, 0.8, 1e-9);
  EXPECT_EQ(out.has_mapped_support, 1);
  EXPECT_NEAR(out.mapped_support_x, 5.0, 1e-9);  // chart origin
  EXPECT_NE(out.local_referent, 0);
  // The layout without chart_version is accepted.
  out = Init<sb_target_relation_evaluation_t>();
  p.struct_size =
      static_cast<uint32_t>(offsetof(sb_relation_payload_t, chart_version));
  ASSERT_EQ(sb_evaluate_target_relation(r.space, &p, p.struct_size, &q, &out),
            1);
  EXPECT_EQ(out.disposition, SB_TARGET_EVAL_EVALUATED);
  // A referent the receiver lacks is an explicit disposition.
  p = Contact(999, "hand", 0.5, 1.0);
  out = Init<sb_target_relation_evaluation_t>();
  ASSERT_EQ(sb_evaluate_target_relation(r.space, &p, sizeof(p), &q, &out), 1);
  EXPECT_EQ(out.disposition, SB_TARGET_EVAL_UNMAPPED);
  EXPECT_STRNE(out.reason, "");
  // Bad sizes are refused.
  EXPECT_EQ(sb_evaluate_target_relation(r.space, &p, 12, &q, &out), 0);
  EXPECT_EQ(sb_sizeof(SB_STRUCT_TARGET_RELATION_EVALUATION),
            sizeof(sb_target_relation_evaluation_t));
  EXPECT_EQ(sb_get_struct_field_offset(SB_STRUCT_TARGET_RELATION_EVALUATION,
                                       4),
            offsetof(sb_target_relation_evaluation_t, r_tgt));
}

// ---------------------------------------------------------------------------
// Source-side scope record and relation state v2, and
// sb_evaluate_relation against the binding pipeline and analytic values.
// ---------------------------------------------------------------------------

struct Source {
  sb_session_t session = 0;
  sb_space_t space = 0;
  sb_element_t room = 0, actor_elem = 0, desk = 0, chair = 0;
  sb_actor_t actor = 0;
  Source() {
    session = sb_create_session();
    space = sb_add_space(session);
    EXPECT_EQ(sb_set_observability_capture(space, 1), 1);
    const double room_poly[] = {0, 0, 10, 0, 10, 10, 0, 10};
    room = sb_add_element(space, 0);
    sb_set_element_shared_id(space, room, 100);
    sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, room_poly, 4);
    actor_elem = sb_add_element(space, room);
    sb_set_element_shared_id(space, actor_elem, 900);
    desk = sb_add_element(space, room);
    sb_set_element_shared_id(space, desk, 201);
    sb_set_element_tag(space, desk, "Desk");
    sb_set_element_box(space, desk, SB_GEOMETRY_OBSTACLE, 4, 4, 0, 6, 6, 1);
    chair = sb_add_element(space, room);
    sb_set_element_shared_id(space, chair, 202);
    sb_set_element_tag(space, chair, "Chair");
    sb_set_element_box(space, chair, SB_GEOMETRY_OBSTACLE, 8, 8, 0, 9, 9, 1);
    actor = sb_create_actor(space);
    spatialblend_c_test::PublishStartPose(space, actor);
    sb_add_anchor(space, actor, "root", 0, 0, 0, nullptr);
    sb_set_actor_element(space, actor, actor_elem);
  }
  ~Source() { sb_destroy_session(session); }

  std::vector<sb_scope_candidate_t> Candidates(int32_t binding) {
    const int32_t n = sb_get_scope_candidates(space, actor, binding, nullptr, 0,
                                              sizeof(sb_scope_candidate_t));
    EXPECT_GE(n, 0);
    std::vector<sb_scope_candidate_t> out(std::max(n, 0),
                                          Init<sb_scope_candidate_t>());
    if (n > 0)
      sb_get_scope_candidates(space, actor, binding, out.data(), n,
                              sizeof(sb_scope_candidate_t));
    return out;
  }
};

const sb_scope_candidate_t* Find(const std::vector<sb_scope_candidate_t>& v,
                                 sb_element_t id) {
  for (const auto& c : v)
    if (c.element_id == id) return &c;
  return nullptr;
}

TEST(ScopeRecordTest, E1_7_ReasonsPerReferent) {
  Source s;
  sb_bind(s.space, s.actor, "root", SB_KERNEL_SPACING);          // 0
  sb_bind(s.space, s.actor, "root", SB_KERNEL_FACING);
  ASSERT_EQ(sb_bind(s.space, s.actor, "root", SB_KERNEL_FACING),
            1);
  sb_update_actor_state(s.space, s.actor, 2.0, 5.0, 0.0);

  std::vector<sb_scope_record_t> rec(3, Init<sb_scope_record_t>());
  ASSERT_EQ(sb_get_scope_records(s.space, s.actor, rec.data(), 3,
                                 sizeof(sb_scope_record_t)),
            3);
  // Binding 0: owner-derived spacing scope = the room's direct children.
  EXPECT_EQ(rec[0].valid, 1);
  EXPECT_EQ(rec[0].anchor_active, 1);
  EXPECT_EQ(rec[0].explicit_scope, 0);
  EXPECT_EQ(rec[0].scope_root, s.room);
  EXPECT_EQ(rec[0].kernel_scope_origin, s.actor_elem);
  EXPECT_EQ(rec[0].deepest_containing, s.room);
  EXPECT_EQ(rec[0].navigable_element, s.room);
  EXPECT_DOUBLE_EQ(rec[0].anchor_x, 2.0);
  EXPECT_DOUBLE_EQ(rec[0].anchor_y, 5.0);
  EXPECT_STREQ(rec[0].anchor_name, "root");
  auto c0 = s.Candidates(0);
  ASSERT_NE(Find(c0, s.actor_elem), nullptr);
  EXPECT_EQ(Find(c0, s.actor_elem)->reason, SB_SCOPE_EXCLUDED_ACTOR_ELEMENT);
  ASSERT_NE(Find(c0, s.desk), nullptr);
  EXPECT_EQ(Find(c0, s.desk)->reason, SB_SCOPE_INCLUDED);
  EXPECT_EQ(Find(c0, s.desk)->shared_id, 201);
  EXPECT_EQ(Find(c0, s.chair)->reason, SB_SCOPE_INCLUDED);
  EXPECT_EQ(rec[0].included_count, 2);
  EXPECT_EQ(rec[0].candidate_count, static_cast<int64_t>(c0.size()));
  EXPECT_TRUE(std::isnan(Find(c0, s.actor_elem)->proximity));
  EXPECT_FALSE(std::isnan(Find(c0, s.desk)->proximity));

  // Binding 1: the tag "Chair" no longer narrows a
  // Facing binding (it is only the AffordancePose binding), so the binding carries no
  // tag and the desk is evaluated along with the chair. (It used to be excluded by tag.)
  EXPECT_STREQ(rec[1].tag_filter, "");
  auto c1 = s.Candidates(1);
  ASSERT_NE(Find(c1, s.desk), nullptr);
  EXPECT_EQ(Find(c1, s.desk)->reason, SB_SCOPE_INCLUDED);
  ASSERT_NE(Find(c1, s.chair), nullptr);
  EXPECT_EQ(Find(c1, s.chair)->reason, SB_SCOPE_INCLUDED);

  // Binding 2: the scope id is ignored. Facing stays global, so the chair
  // is a referent along with the desk.
  EXPECT_EQ(rec[2].explicit_scope, 0);
  EXPECT_EQ(rec[2].scope_root, s.room);
  auto c2 = s.Candidates(2);
  ASSERT_NE(Find(c2, s.desk), nullptr);
  EXPECT_EQ(Find(c2, s.desk)->reason, SB_SCOPE_INCLUDED);
  ASSERT_NE(Find(c2, s.chair), nullptr);
  EXPECT_EQ(Find(c2, s.chair)->reason, SB_SCOPE_INCLUDED);

  EXPECT_EQ(sb_get_scope_candidates(s.space, s.actor, 3, nullptr, 0,
                                    sizeof(sb_scope_candidate_t)),
            -1);
}

TEST(ScopeRecordTest, E1_7_InactiveAnchorRecordsNoCandidates) {
  Source s;
  sb_bind(s.space, s.actor, "root", SB_KERNEL_SPACING);
  ASSERT_EQ(sb_set_anchor_active(s.space, s.actor, "root", 0), 1);
  sb_update_actor_state(s.space, s.actor, 2.0, 5.0, 0.0);
  auto rec = Init<sb_scope_record_t>();
  ASSERT_EQ(sb_get_scope_records(s.space, s.actor, &rec, 1, sizeof(rec)), 1);
  EXPECT_EQ(rec.valid, 1);
  EXPECT_EQ(rec.anchor_active, 0);
  EXPECT_EQ(rec.candidate_count, 0);
}

TEST(RelationStateV2Test, X1_SupportEncodedStateAndScopeDiameter) {
  Source s;
  sb_bind(s.space, s.actor, "root", SB_KERNEL_SPACING);
  sb_update_actor_state(s.space, s.actor, 2.0, 5.0, 0.0);

  const int32_t n = sb_get_relation_states_v2(s.space, s.actor, nullptr, 0,
                                              sizeof(sb_relation_state_v2_t));
  ASSERT_EQ(n, 2);
  std::vector<sb_relation_state_v2_t> v2(n, Init<sb_relation_state_v2_t>());
  ASSERT_EQ(sb_get_relation_states_v2(s.space, s.actor, v2.data(), n,
                                      sizeof(sb_relation_state_v2_t)),
            n);
  std::vector<sb_relation_state_t> v1(n);
  ASSERT_EQ(sb_get_relation_states(s.space, s.actor, v1.data(), n), n);

  const sb_relation_state_v2_t* desk = nullptr;
  for (int i = 0; i < n; ++i) {
    // v2 is a superset of v1, row for row.
    EXPECT_EQ(v2[i].referent_element_id, v1[i].referent_element_id);
    EXPECT_DOUBLE_EQ(v2[i].proximity, v1[i].proximity);
    EXPECT_EQ(v2[i].active, v1[i].active);
    EXPECT_EQ(v2[i].binding_index, 0);
    if (v2[i].referent_element_id == s.desk) desk = &v2[i];
  }
  ASSERT_NE(desk, nullptr);
  // Anchor (2, 5), desk box [4, 6]^2: nearest boundary point (4, 5) at 2 m;
  // the room's bounding-box diagonal is sqrt(200) m.
  EXPECT_NEAR(desk->support_distance_m, 2.0, 1e-6);
  EXPECT_NEAR(desk->source_scope_diameter_m, std::sqrt(200.0), 1e-6);
  EXPECT_NEAR(desk->proximity, 1.0 - 2.0 / std::sqrt(200.0), 1e-6);
  EXPECT_GE(desk->encoded_state_size, 2);
  EXPECT_STREQ(desk->sub_region_name, "");
  EXPECT_EQ(desk->has_contact, 1);
  EXPECT_NEAR(desk->contact_point_world_x, 4.0, 1e-6);
  EXPECT_NEAR(desk->contact_point_world_y, 5.0, 1e-6);
}

TEST(EvaluateRelationTest, X2_PureEvaluationEqualsTheBindingPipeline) {
  Source s;
  sb_bind(s.space, s.actor, "root", SB_KERNEL_SPACING);
  sb_bind(s.space, s.actor, "root", SB_KERNEL_FACING);
  sb_update_actor_state(s.space, s.actor, 2.0, 5.0, 0.0);
  const int32_t n = sb_get_relation_states_v2(s.space, s.actor, nullptr, 0,
                                              sizeof(sb_relation_state_v2_t));
  std::vector<sb_relation_state_v2_t> st(n, Init<sb_relation_state_v2_t>());
  ASSERT_EQ(sb_get_relation_states_v2(s.space, s.actor, st.data(), n,
                                      sizeof(sb_relation_state_v2_t)),
            n);

  int compared = 0;
  for (const auto& row : st) {
    if (row.referent_element_id != s.desk) continue;
    auto q = Init<sb_relation_query_t>();
    q.anchor_x = 2.0;
    q.anchor_y = 5.0;
    q.anchor_yaw = 0.0;
    q.referent = s.desk;
    q.scope_root = s.room;
    q.kernel_type = row.kernel_type;
    auto e = Init<sb_relation_evaluation_t>();
    ASSERT_EQ(sb_evaluate_relation(s.space, &q, &e), 1);
    // Same ComputeSupport + Encode as the pipeline: bit-identical.
    EXPECT_EQ(e.proximity, row.proximity);
    EXPECT_EQ(e.support_distance_m, row.support_distance_m);
    EXPECT_EQ(e.source_scope_diameter_m, row.source_scope_diameter_m);
    ASSERT_EQ(e.encoded_state_size, row.encoded_state_size);
    for (int k = 0; k < e.encoded_state_size && k < SB_MAX_ENCODED_STATE; ++k)
      EXPECT_EQ(e.encoded_state[k], row.encoded_state[k]);
    EXPECT_EQ(e.navigable_element, s.room);
    EXPECT_EQ(e.deepest_containing, s.room);
    if (row.kernel_type == SB_KERNEL_FACING) {
      // Heading +x from (2, 5) hits the desk face: facing R = 1.
      EXPECT_NEAR(e.proximity, 1.0, 1e-9);
    } else {
      EXPECT_NEAR(e.proximity, 1.0 - 2.0 / std::sqrt(200.0), 1e-6);
    }
    ++compared;
  }
  EXPECT_EQ(compared, 2);

  // Analytic spacing values at another pose, no binding involved:
  // anchor (1, 5) is 3 m from the desk face.
  auto q = Init<sb_relation_query_t>();
  q.anchor_x = 1.0;
  q.anchor_y = 5.0;
  q.referent = s.desk;
  q.scope_root = s.room;
  q.kernel_type = SB_KERNEL_SPACING;
  auto e = Init<sb_relation_evaluation_t>();
  ASSERT_EQ(sb_evaluate_relation(s.space, &q, &e), 1);
  EXPECT_NEAR(e.support_distance_m, 3.0, 1e-6);
  EXPECT_NEAR(e.proximity, 1.0 - 3.0 / std::sqrt(200.0), 1e-6);
  // Pure: evaluating did not touch the binding's state.
  auto rec = Init<sb_scope_record_t>();
  ASSERT_EQ(sb_get_scope_records(s.space, s.actor, &rec, 1, sizeof(rec)), 2);
  EXPECT_DOUBLE_EQ(rec.anchor_x, 2.0);

  // Bad inputs.
  q.referent = 424242;
  EXPECT_EQ(sb_evaluate_relation(s.space, &q, &e), 0);
  q.referent = s.desk;
  q.position_dimension = 4;
  EXPECT_EQ(sb_evaluate_relation(s.space, &q, &e), 0);
}

// ---------------------------------------------------------------------------
// World transforms of tree elements.
// ---------------------------------------------------------------------------

TEST(ElementTreeReadbackTest, X1_WorldTransformComposesAncestors) {
  Receiver r;
  sb_element_t a = r.Point(501, 1, 2, 0);
  sb_element_t b = sb_add_element(r.space, a);
  // Local: 90 deg yaw + translation (3, 0, 1).
  const double local[16] = {0, -1, 0, 3, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1};
  sb_set_element_transform(r.space, b, local);
  double w[16];
  ASSERT_EQ(sb_get_element_world_transform(r.space, b, w), 1);
  // World = T(1,2,0) * [R(90) | (3,0,1)] -> translation (4, 2, 1).
  EXPECT_NEAR(w[3], 4.0, kTol);
  EXPECT_NEAR(w[7], 2.0, kTol);
  EXPECT_NEAR(w[11], 1.0, kTol);
  EXPECT_NEAR(w[0], 0.0, kTol);
  EXPECT_NEAR(w[1], -1.0, kTol);
  sb_element_t parent = -5;
  ASSERT_EQ(sb_get_element_parent(r.space, b, &parent), 1);
  EXPECT_EQ(parent, a);
  ASSERT_EQ(sb_get_element_parent(r.space, r.room, &parent), 1);
  EXPECT_EQ(parent, 0);
  EXPECT_EQ(sb_get_element_world_transform(r.space, 777777, w), 0);
}

// ---------------------------------------------------------------------------
// The trace carries policy parameters and the driving
// relation per channel and tick.
// ---------------------------------------------------------------------------

TEST(TraceFieldsTest, R3_5_R3_6_PolicyAndDrivingRelationAreTraced) {
  const std::string path = TempPath("obs_policy_trace.jsonl");
  Receiver r;
  r.Point(301, 2, 2);
  r.Point(302, 8, 8);
  sb_set_trace(r.space, path.c_str());
  sb_relation_payload_t in[2] = {
      Payload(301, SB_KERNEL_SPACING, "root", 0.9, 1.0),
      Payload(302, SB_KERNEL_SPACING, "root", 0.5, 3.0)};
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 2, sizeof(in[0]),
                                        0.1),
            1);
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 2, sizeof(in[0]),
                                        0.2),
            1);
  auto p = Init<sb_realization_policy_params_t>();
  sb_get_realization_policy_params(r.space, &p, sizeof(p));
  p.gamma = 2.0;
  ASSERT_EQ(sb_set_realization_policy_params(r.space, &p, sizeof(p)), 1);
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 2, sizeof(in[0]),
                                        0.3),
            1);
  sb_set_trace(r.space, nullptr);

  const auto lines = ReadTrace(path);
  ASSERT_FALSE(lines.empty());
  EXPECT_EQ(lines.front()["ev"], "meta");
  EXPECT_EQ(lines.front()["format"], 10);
  std::vector<nlohmann::json> policy, channel, payload_driving;
  for (const auto& ev : lines) {
    if (ev["ev"] == "policy") policy.push_back(ev);
    if (ev["ev"] == "channel") channel.push_back(ev);
    if (ev["ev"] == "payload" && ev["driving"] == true)
      payload_driving.push_back(ev);
  }
  // One policy line for the initial revision, one after the gamma change.
  ASSERT_EQ(policy.size(), 2u);
  EXPECT_EQ(policy[0]["policy"], "progressive");
  EXPECT_DOUBLE_EQ(policy[0]["gamma"].get<double>(), 1.0);
  EXPECT_DOUBLE_EQ(policy[0]["eta_max"].get<double>(), 100.0);
  EXPECT_DOUBLE_EQ(policy[0]["kappa"].get<double>(), 0.2);
  EXPECT_DOUBLE_EQ(policy[1]["gamma"].get<double>(), 2.0);
  // One channel line per tick, naming the driving referent.
  ASSERT_EQ(channel.size(), 3u);
  for (const auto& c : channel) {
    EXPECT_EQ(c["driving_referent"], 301);
    EXPECT_EQ(c["n"], 2);
    EXPECT_EQ(c["anchor"], "root");
    EXPECT_EQ(c["kernel"], 1);  // internal enum: spacing
  }
  EXPECT_NEAR(channel[0]["eta"].get<double>(), 1e-3, kTol);
  EXPECT_NEAR(channel[2]["eta"].get<double>(), 1e-6, kTol);
  EXPECT_NEAR(channel[0]["driving_weight"].get<double>(),
              1.0 - ProgressiveSecond(1.0, 3.0, 1.0), kTol);
  ASSERT_EQ(payload_driving.size(), 3u);
  for (const auto& pl : payload_driving) EXPECT_EQ(pl["referent"], 301);
  std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Covered: no escape, locking, realized driving, scope order, parity,
// revision scoping.
// ---------------------------------------------------------------------------

TEST(ObservabilityNoEscapeTest, ReceiverInputStillWorksAfterEveryGetter) {
  // Regression: reading the scope record once used to mark the space's scene
  // inputs as escaped, which permanently disabled receiver input.
  Source s;
  sb_bind(s.space, s.actor, "root", SB_KERNEL_SPACING);
  sb_update_actor_state(s.space, s.actor, 2.0, 5.0, 0.0);
  sb_element_t rec_elem = sb_add_element(s.space, s.room);
  const sb_actor_t rec = sb_create_actor(s.space);
  spatialblend_c_test::PublishStartPose(s.space, rec);
  sb_add_anchor(s.space, rec, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(s.space, rec, rec_elem);

  std::vector<sb_scope_record_t> r(4, Init<sb_scope_record_t>());
  std::vector<sb_scope_candidate_t> c(8, Init<sb_scope_candidate_t>());
  std::vector<sb_relation_state_v2_t> st(8, Init<sb_relation_state_v2_t>());
  std::vector<sb_realization_weight_t> w(8, Init<sb_realization_weight_t>());
  std::vector<sb_realization_channel_t> ch(8,
                                           Init<sb_realization_channel_t>());
  std::vector<sb_solver_goal_t> g(8, Init<sb_solver_goal_t>());
  std::vector<sb_realization_target_v2_t> t(
      8, Init<sb_realization_target_v2_t>());
  auto d = Init<sb_solver_diagnostics_t>();
  auto p = Init<sb_realization_policy_params_t>();
  auto q = Init<sb_relation_query_t>();
  auto e = Init<sb_relation_evaluation_t>();
  q.anchor_x = 2;
  q.anchor_y = 5;
  q.referent = s.desk;
  q.scope_root = s.room;
  q.kernel_type = SB_KERNEL_SPACING;
  double m[16];
  sb_element_t parent = 0;
  EXPECT_GT(sb_get_scope_records(s.space, s.actor, r.data(), 4, sizeof(r[0])),
            0);
  EXPECT_GT(sb_get_scope_candidates(s.space, s.actor, 0, c.data(), 8,
                                    sizeof(c[0])),
            0);
  EXPECT_GT(sb_get_relation_states_v2(s.space, s.actor, st.data(), 8,
                                      sizeof(st[0])),
            0);
  sb_get_realization_weights(s.space, s.actor, w.data(), 8, sizeof(w[0]));
  sb_get_realization_channels(s.space, s.actor, ch.data(), 8, sizeof(ch[0]));
  sb_get_solver_goals(s.space, s.actor, g.data(), 8, sizeof(g[0]));
  sb_get_solver_diagnostics(s.space, s.actor, &d, sizeof(d));
  sb_get_realization_targets_v2(s.space, s.actor, t.data(), 8, sizeof(t[0]));
  EXPECT_EQ(sb_get_realization_policy_params(s.space, &p, sizeof(p)), 1);
  EXPECT_EQ(sb_set_realization_policy_params(s.space, &p, sizeof(p)), 1);
  EXPECT_EQ(sb_evaluate_relation(s.space, &q, &e), 1);
  EXPECT_EQ(sb_get_element_world_transform(s.space, s.desk, m), 1);
  EXPECT_EQ(sb_get_element_parent(s.space, s.desk, &parent), 1);
  std::vector<sb_relation_state_t> v1(8);
  sb_get_relation_states(s.space, s.actor, v1.data(), 8);

  uint64_t epoch = 0;
  EXPECT_EQ(sb_begin_receiver_input(s.space, rec, 1,
                                    reinterpret_cast<const uint8_t*>("root"),
                                    4, 1, 1, &epoch),
            SB_RICH_RESULT_OK);
  EXPECT_NE(epoch, 0u);
}

TEST(PayloadInjectionTest, R3_6_RealizedDrivingSkipsUnresolvableArgmax) {
  Receiver r;
  r.Point(302, 8, 8);
  // 999 does not exist here, so it takes no part in the weights (as in v1, which skips a target absent from
  // the receiver before its steps enter the weights): 302 is the
  // policy argmax and drives the realized root target.
  sb_relation_payload_t in[2] = {
      Payload(999, SB_KERNEL_SPACING, "root", 0.9, 2.0),
      Payload(302, SB_KERNEL_SPACING, "root", 0.5, 4.0)};
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 2, sizeof(in[0]),
                                        0.1),
            1);
  auto ch = Channels(r);
  ASSERT_EQ(ch.size(), 1u);
  EXPECT_EQ(ch[0].driving_referent_shared_id, 302);
  ASSERT_EQ(ch[0].has_realized, 1);
  EXPECT_EQ(ch[0].realized_referent_shared_id, 302);
  EXPECT_EQ(ch[0].realized_payload_index, 1);
  EXPECT_EQ(ch[0].realized_is_policy_argmax, 1);
  auto w = Weights(r);
  EXPECT_EQ(w[0].is_driving, 0);
  EXPECT_EQ(w[0].is_realized_driving, 0);
  EXPECT_EQ(w[1].is_driving, 1);
  EXPECT_EQ(w[1].is_realized_driving, 1);
}

// An anchor whose source hand does not TOUCH its support has no realized
// driving relation (it follows the body); it is realized on the very solve it
// touches. (The retired contract: a weight*R EMA gate, entered at 0.6.)
TEST(PayloadInjectionTest, R3_6_NotTouchingAnchorHasNoRealizedDriving) {
  Receiver r;
  r.Point(301, 2, 2);
  // The hand support needs geometry and a touching contact.
  r.Surface(310, 2.5, 2, 0.8);
  // The hand is 0.3 m from its support: not touching, no hand target.
  sb_relation_payload_t in[2] = {
      Payload(301, SB_KERNEL_SPACING, "root", 1.0, 0.0),
      Contact(310, "hand", 0.3, 0.3)};
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 2, sizeof(in[0]),
                                        0.1),
            1);
  auto ch = Channels(r);
  ASSERT_EQ(ch.size(), 2u);
  const auto& hand = std::string(ch[0].anchor_name) == "hand" ? ch[0] : ch[1];
  EXPECT_EQ(hand.has_driving, 1);
  EXPECT_EQ(hand.driving_referent_shared_id, 310);
  EXPECT_EQ(hand.has_realized, 0);
  EXPECT_EQ(hand.realized_payload_index, -1);
  // Touching realizes it at once: no EMA to cross.
  in[1].proximity = 1.0;
  in[1].support_distance_m = 0.0;
  in[1].estimated_time = 1.0;  // in contact: v1's count 1
  for (double t : {0.2})
    ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, in, 2,
                                          sizeof(in[0]), t),
              1);
  ch = Channels(r);
  const auto& hand2 = std::string(ch[0].anchor_name) == "hand" ? ch[0] : ch[1];
  EXPECT_EQ(hand2.has_realized, 1);
  EXPECT_EQ(hand2.realized_referent_shared_id, 310);
  EXPECT_EQ(hand2.realized_is_policy_argmax, 1);
}

// The ABI Eval_p resolves the referent and the source's
// normalizing scope exactly as the realizer does, through the identities the
// payload carries -- the case of differing hierarchies (the receiver's
// table sits in a navigable zone the source does not have).
TEST(TargetEvaluationTest, R1_3_ScopeAndReferentIdentitiesMatchTheRealizer) {
  const sb_session_t session = sb_create_session();
  const sb_space_t src = sb_add_space(session);
  const sb_space_t rcv = sb_add_space(session);
  const double room_poly[] = {0, 0, 10, 0, 10, 8, 0, 8};
  const double zone_poly[] = {2, 0.5, 8, 0.5, 8, 6.5, 2, 6.5};
  const sb_element_t src_room = sb_add_element(src, 0);
  sb_set_element_shared_id(src, src_room, 100);
  sb_set_element_polygon(src, src_room, SB_GEOMETRY_NAVIGABLE, room_poly, 4);
  const sb_element_t src_table = sb_add_element(src, src_room);
  sb_set_element_shared_id(src, src_table, 200);
  sb_set_element_box(src, src_table, SB_GEOMETRY_OBSTACLE, 4, 3, 0, 6, 5, 0.8);
  const sb_element_t rcv_room = sb_add_element(rcv, 0);
  sb_set_element_shared_id(rcv, rcv_room, 100);
  sb_set_element_polygon(rcv, rcv_room, SB_GEOMETRY_NAVIGABLE, room_poly, 4);
  const sb_element_t rcv_zone = sb_add_element(rcv, rcv_room);
  sb_set_element_shared_id(rcv, rcv_zone, 150);
  sb_set_element_polygon(rcv, rcv_zone, SB_GEOMETRY_NAVIGABLE, zone_poly, 4);
  const sb_element_t rcv_table = sb_add_element(rcv, rcv_zone);
  sb_set_element_shared_id(rcv, rcv_table, 200);
  sb_set_element_box(rcv, rcv_table, SB_GEOMETRY_OBSTACLE, 4, 3, 0, 6, 5, 0.8);

  // Source Extract_p at the anchor (5, 1): spacing to the table, normalized
  // by the source room.
  sb_relation_query_t q = Init<sb_relation_query_t>();
  q.anchor_x = 5;
  q.anchor_y = 1;
  q.referent = src_table;
  q.scope_root = src_room;
  q.kernel_type = SB_KERNEL_SPACING;
  sb_relation_evaluation_t ev = Init<sb_relation_evaluation_t>();
  ASSERT_EQ(sb_evaluate_relation(src, &q, &ev), 1);
  ASSERT_NEAR(ev.proximity, 1.0 - 2.0 / std::hypot(10.0, 8.0), 1e-9);
  sb_relation_payload_t p =
      Payload(200, SB_KERNEL_SPACING, "root", ev.proximity, 0.0);
  p.support_distance_m = ev.support_distance_m;
  p.source_scope_diameter_m = ev.source_scope_diameter_m;
  p.has_contact = ev.has_contact;
  p.normalized_angle = ev.normalized_angle;
  p.normalized_radial_ratio = ev.normalized_radial_ratio;
  p.encoded_state_size = ev.encoded_state_size;
  for (int k = 0; k < ev.encoded_state_size; ++k)
    p.encoded_state[k] = ev.encoded_state[k];

  sb_relation_query_t candidate = Init<sb_relation_query_t>();
  candidate.anchor_x = 5;
  candidate.anchor_y = 1;
  auto evaluate = [&](const sb_relation_payload_t& payload) {
    sb_target_relation_evaluation_t out =
        Init<sb_target_relation_evaluation_t>();
    EXPECT_EQ(sb_evaluate_target_relation(rcv, &payload, sizeof(payload),
                                          &candidate, &out),
              1);
    return out;
  };
  // Without the scope identity: the structural correspondent (the zone).
  auto anonymous = evaluate(p);
  ASSERT_EQ(anonymous.disposition, SB_TARGET_EVAL_EVALUATED);
  EXPECT_EQ(anonymous.normalizing_scope, rcv_zone);
  EXPECT_GT(std::abs(anonymous.r_tgt - ev.proximity), 1e-3);
  // With the source scope's shared id: its counterpart, R^tgt = R^src.
  sb_relation_payload_t named = p;
  named.has_scope_shared_id = 1;
  named.scope_shared_id = 100;
  auto scoped = evaluate(named);
  EXPECT_EQ(scoped.normalizing_scope, rcv_room);
  EXPECT_NEAR(scoped.r_tgt, ev.proximity, 1e-9);
  EXPECT_NEAR(scoped.error, 0.0, 1e-9);

  // Linker-only receiver: identities by handle. The referent's shared id is
  // deliberately wrong; the scope has only its source handle.
  const sb_support_correspondence_t links[2] = {
      {src, src_room, rcv, rcv_room}, {src, src_table, rcv, rcv_table}};
  ASSERT_NE(sb_register_support_correspondences(session, links, 2), 0);
  sb_set_referent_resolve_mode(rcv, SB_REFERENT_RESOLVE_LINKER_ONLY);
  sb_relation_payload_t handles = p;
  handles.referent_shared_id = 999;
  handles.has_referent_source = 1;
  handles.referent_source_space_id = src;
  handles.referent_source_element_id = src_table;
  handles.has_scope_source = 1;
  handles.scope_source_space_id = src;
  handles.scope_source_element_id = src_room;
  auto linked = evaluate(handles);
  ASSERT_EQ(linked.disposition, SB_TARGET_EVAL_EVALUATED);
  EXPECT_EQ(linked.local_referent, rcv_table);
  EXPECT_EQ(linked.normalizing_scope, rcv_room);
  EXPECT_NEAR(linked.r_tgt, ev.proximity, 1e-9);
  // Linker-only with a scope shared id but no scope handle: no shared-id
  // fallback for the scope (structural zone), and a referent without a
  // handle is unmapped.
  sb_relation_payload_t shared_scope = handles;
  shared_scope.has_scope_source = 0;
  shared_scope.has_scope_shared_id = 1;
  shared_scope.scope_shared_id = 100;
  EXPECT_EQ(evaluate(shared_scope).normalizing_scope, rcv_zone);
  sb_relation_payload_t no_handle = named;
  EXPECT_EQ(evaluate(no_handle).disposition, SB_TARGET_EVAL_UNMAPPED);
  sb_destroy_session(session);
}

// sb_relation_state_v2_t carries the chart version: a read-then-inject round
// trip near the seat centre is exact. The chart is pure radial, so a
// version-0 reader (pre-chart_version) decodes the same point.
TEST(RelationStateV2Test, ChartVersionRoundTripsNearTheSeatCentre) {
  Source s;
  const sb_element_t sofa = sb_add_element(s.space, s.room);
  sb_set_element_shared_id(s.space, sofa, 310);
  const double top[] = {4.1, 6.7, 0.45, 5.9, 6.7, 0.45,
                        5.9, 7.3, 0.45, 4.1, 7.3, 0.45};  // 1.8 x 0.6 seat
  sb_set_element_surface(s.space, sofa, top, 4, SB_GEOMETRY_SURFACE);
  ASSERT_EQ(sb_add_anchor_with_position_dimension(s.space, s.actor, "hand", 0,
                                                  0, 0, nullptr, 3),
            1);
  sb_bind(s.space, s.actor, "hand", SB_KERNEL_AFFORDANCE_POSE);
  // The hand ON the seat (touching), 0.06 m / 0.02 m off its centre (5, 7).
  sb_set_anchor_transform(s.space, s.actor, "hand", 5.06, 7.02, 0.45, 0);
  sb_update_actor_state(s.space, s.actor, 5.0, 6.0, 0.0);
  std::vector<sb_relation_state_v2_t> rows(
      8, Init<sb_relation_state_v2_t>());
  const int32_t n = sb_get_relation_states_v2(
      s.space, s.actor, rows.data(), 8, sizeof(sb_relation_state_v2_t));
  const sb_relation_state_v2_t* hand = nullptr;
  for (int32_t i = 0; i < n; ++i)
    if (rows[i].referent_element_id == sofa) hand = &rows[i];
  ASSERT_NE(hand, nullptr);
  EXPECT_EQ(hand->chart_version, 1u);
  ASSERT_LT(hand->normalized_radial_ratio, 0.3);

  // Inject the read row into an identical receiver (the Receiver fixture's
  // room plus the same seat surface).
  auto inject = [&](uint32_t chart_version) {
    Receiver r;
    const sb_element_t seat = sb_add_element(r.space, r.room);
    sb_set_element_shared_id(r.space, seat, 310);
    sb_set_element_surface(r.space, seat, top, 4, SB_GEOMETRY_SURFACE);
    sb_relation_payload_t p =
        Contact(310, "hand", hand->proximity, hand->support_distance_m);
    p.normalized_angle = hand->normalized_angle;
    p.normalized_radial_ratio = hand->normalized_radial_ratio;
    p.chart_version = chart_version;
    EXPECT_EQ(sb_inject_relation_payloads(r.space, r.actor, &p, 1, sizeof(p),
                                          0.1),
              1);
    std::vector<sb_realization_target_v2_t> t(
        4, Init<sb_realization_target_v2_t>());
    const int32_t m = sb_get_realization_targets_v2(
        r.space, r.actor, t.data(), 4, sizeof(sb_realization_target_v2_t));
    for (int32_t i = 0; i < m; ++i)
      if (std::string(t[i].anchor_name) == "hand")
        return std::make_pair(t[i].position_x, t[i].position_y);
    ADD_FAILURE() << "no hand target";
    return std::make_pair(0.0, 0.0);
  };
  const auto versioned = inject(hand->chart_version);
  EXPECT_NEAR(versioned.first, 5.06, 1e-9);
  EXPECT_NEAR(versioned.second, 7.02, 1e-9);
  // Dropping the version (a pre-chart_version reader) decodes the same pure radial point.
  const auto legacy = inject(0);
  EXPECT_NEAR(legacy.first, 5.06, 1e-9);
  EXPECT_NEAR(legacy.second, 7.02, 1e-9);
}

TEST(ScopeRecordTest, E1_7_CandidatesFollowScopePolicyOrder) {
  Source s;
  // D6: only an AffordancePose binding carries a tag, so the raw/tagged comparison
  // is made on that family.
  sb_bind(s.space, s.actor, "root", SB_KERNEL_AFFORDANCE_POSE);   // raw
  sb_bind_affordance(s.space, s.actor, "root", "Chair");
  sb_update_actor_state(s.space, s.actor, 2.0, 5.0, 0.0);
  auto raw = s.Candidates(0);
  auto tagged = s.Candidates(1);
  ASSERT_EQ(raw.size(), tagged.size());
  bool saw_tag_exclusion_before_inclusion = false;
  for (size_t i = 0; i < raw.size(); ++i) {
    EXPECT_EQ(raw[i].element_id, tagged[i].element_id) << i;
    if (tagged[i].reason == SB_SCOPE_EXCLUDED_TAG)
      for (size_t k = i + 1; k < tagged.size(); ++k)
        saw_tag_exclusion_before_inclusion |=
            tagged[k].reason == SB_SCOPE_INCLUDED;
  }
  // The desk (tag-excluded) precedes the chair (included) in policy order,
  // which stage-by-stage recording would have reversed.
  EXPECT_TRUE(saw_tag_exclusion_before_inclusion);
}

TEST(ScopeRecordTest, E1_7_CaptureIsOptIn) {
  // No trace, no capture: nothing is recorded (and nothing allocated).
  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  const double poly[] = {0, 0, 10, 0, 10, 10, 0, 10};
  const sb_element_t room = sb_add_element(space, 0);
  sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, poly, 4);
  const sb_element_t desk = sb_add_element(space, room);
  sb_set_element_box(space, desk, SB_GEOMETRY_OBSTACLE, 4, 4, 0, 6, 6, 1);
  const sb_element_t ae = sb_add_element(space, room);
  const sb_actor_t actor = sb_create_actor(space);
  spatialblend_c_test::PublishStartPose(space, actor);
  sb_add_anchor(space, actor, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(space, actor, ae);
  sb_bind(space, actor, "root", SB_KERNEL_SPACING);
  sb_update_actor_state(space, actor, 2, 5, 0);
  auto rec = Init<sb_scope_record_t>();
  // The first read enables capture; the evaluation before it was not
  // recorded.
  ASSERT_EQ(sb_get_scope_records(space, actor, &rec, 1, sizeof(rec)), 1);
  EXPECT_EQ(rec.valid, 0);
  sb_update_actor_state(space, actor, 2.1, 5, 0);
  ASSERT_EQ(sb_get_scope_records(space, actor, &rec, 1, sizeof(rec)), 1);
  EXPECT_EQ(rec.valid, 1);
  EXPECT_GT(rec.candidate_count, 0);
  sb_destroy_session(session);
}

// sb_evaluate_relation parity beyond the flat room: nested regions, owner-derived vs explicit
// scope, and a 3-D anchor.
TEST(EvaluateRelationTest, X2_ParityNestedExplicitOwnerAnd3D) {
  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  sb_set_observability_capture(space, 1);
  const double room_poly[] = {0, 0, 20, 0, 20, 20, 0, 20};
  const sb_element_t room = sb_add_element(space, 0);
  sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, room_poly, 4);
  // A non-navigable containment region nested in the room, with a nested
  // navigable sub-region inside it.
  const double region_poly[] = {1, 1, 11, 1, 11, 11, 1, 11};
  const sb_element_t region = sb_add_element(space, room);
  sb_set_element_polygon(space, region, SB_GEOMETRY_CONTAINMENT, region_poly,
                         4);
  const double sub_poly[] = {2, 2, 8, 2, 8, 8, 2, 8};
  const sb_element_t sub = sb_add_element(space, region);
  sb_set_element_polygon(space, sub, SB_GEOMETRY_NAVIGABLE, sub_poly, 4);
  const sb_element_t desk = sb_add_element(space, region);
  sb_set_element_shared_id(space, desk, 201);
  sb_set_element_box(space, desk, SB_GEOMETRY_OBSTACLE, 9, 4, 0, 10, 6, 1);
  const sb_element_t cup = sb_add_element(space, sub);
  sb_set_element_shared_id(space, cup, 301);
  const double cm[16] = {1, 0, 0, 5, 0, 1, 0, 5, 0, 0, 1, 0.9, 0, 0, 0, 1};
  sb_set_element_transform(space, cup, cm);
  const sb_element_t ae = sb_add_element(space, room);
  const sb_actor_t actor = sb_create_actor(space);
  spatialblend_c_test::PublishStartPose(space, actor);
  sb_add_anchor(space, actor, "root", 0, 0, 0, nullptr);
  ASSERT_EQ(sb_add_anchor_with_position_dimension(space, actor, "hand", 0, 0,
                                                  0, nullptr, 3),
            1);
  sb_set_actor_element(space, actor, ae);
  sb_bind(space, actor, "root", SB_KERNEL_SPACING);                 // 0
  ASSERT_EQ(sb_bind(space, actor, "root", SB_KERNEL_SPACING),
            1);
  sb_bind(space, actor, "hand", SB_KERNEL_AFFORDANCE_POSE);        // 2
  sb_set_anchor_transform(space, actor, "hand", 4.5, 5.0, 1.1, 0.3);
  sb_update_actor_state(space, actor, 4.0, 5.0, 0.0);

  const int32_t n = sb_get_relation_states_v2(space, actor, nullptr, 0,
                                              sizeof(sb_relation_state_v2_t));
  std::vector<sb_relation_state_v2_t> st(n, Init<sb_relation_state_v2_t>());
  ASSERT_EQ(sb_get_relation_states_v2(space, actor, st.data(), n,
                                      sizeof(sb_relation_state_v2_t)),
            n);
  std::vector<sb_scope_record_t> rec(3, Init<sb_scope_record_t>());
  ASSERT_EQ(sb_get_scope_records(space, actor, rec.data(), 3,
                                 sizeof(sb_scope_record_t)),
            3);
  // Nested: the anchor (4, 5) is in room > region > sub; the owner-derived
  // binding normalizes over the nested navigable sub-region.
  EXPECT_EQ(rec[0].deepest_containing, sub);
  EXPECT_EQ(rec[0].navigable_element, sub);

  // Every row of `binding` (whatever referents its scope policy produced)
  // must equal the pure evaluation.
  auto check = [&](int64_t binding, sb_element_t scope_root,
                   int32_t explicit_scope, int32_t dim, double x, double y,
                   double z, double yaw) {
    int rows = 0;
    for (const auto& s : st) {
      if (s.binding_index != binding) continue;
      ++rows;
      const sb_relation_state_v2_t* row = &s;
      const sb_element_t referent = row->referent_element_id;
      // The container the anchor is inside is a spacing referent too (the
      // container contract), with a CONTAINER-SELF relation: its
      // depth below the border is derived from the source's own support, which
      // the pure (anchor, referent) query does not carry. Parity is asserted for
      // the object referents, whose evaluation is a function of (anchor, referent).
      if (referent == rec[binding].deepest_containing ||
          referent == rec[binding].navigable_element) {
        continue;
      }
    auto q = Init<sb_relation_query_t>();
    q.anchor_x = x;
    q.anchor_y = y;
    q.anchor_z = z;
    q.anchor_yaw = yaw;
    q.referent = referent;
    q.scope_root = scope_root;
    q.kernel_type = row->kernel_type;
    q.position_dimension = dim;
    q.explicit_scope = explicit_scope;
    auto e = Init<sb_relation_evaluation_t>();
    ASSERT_EQ(sb_evaluate_relation(space, &q, &e), 1);
    EXPECT_EQ(e.proximity, row->proximity) << "binding " << binding;
    EXPECT_EQ(e.support_distance_m, row->support_distance_m);
    EXPECT_EQ(e.source_scope_diameter_m, row->source_scope_diameter_m);
    ASSERT_EQ(e.encoded_state_size, row->encoded_state_size);
    for (int k = 0; k < e.encoded_state_size && k < SB_MAX_ENCODED_STATE; ++k)
      EXPECT_EQ(e.encoded_state[k], row->encoded_state[k]);
    EXPECT_EQ(e.navigable_element, rec[binding].navigable_element);
    EXPECT_EQ(e.deepest_containing, rec[binding].deepest_containing);
    }
    EXPECT_GT(rows, 0) << "binding " << binding << " produced no relation";
  };
  // Owner-derived: scope root = the actor element's containment ancestor.
  check(0, room, 0, 2, 4.0, 5.0, 0.0, 0.0);
  // Explicit scope at the non-navigable region: the navigable search stops
  // at the region's root... but the anchor is inside the nested navigable
  // sub-region, which lies below it, so both find `sub`.
  check(1, region, 1, 2, 4.0, 5.0, 0.0, 0.0);
  // 3-D hand anchor, owner-derived affordance pose on the cup.
  check(2, room, 0, 3, 4.5, 5.0, 1.1, 0.3);

  // Where the two scope rules differ: from (9.5, 10.5) -- inside the region
  // but outside the nested navigable -- an explicit region scope finds no
  // navigable (search stops at the region), the owner rule borrows the room.
  auto q = Init<sb_relation_query_t>();
  q.anchor_x = 9.5;
  q.anchor_y = 10.5;
  q.referent = desk;
  q.scope_root = region;
  q.kernel_type = SB_KERNEL_SPACING;
  q.explicit_scope = 1;
  auto e = Init<sb_relation_evaluation_t>();
  ASSERT_EQ(sb_evaluate_relation(space, &q, &e), 1);
  EXPECT_EQ(e.deepest_containing, region);
  EXPECT_EQ(e.navigable_element, 0);
  q.explicit_scope = 0;
  ASSERT_EQ(sb_evaluate_relation(space, &q, &e), 1);
  EXPECT_EQ(e.navigable_element, room);
  // ... and the explicit binding agrees with the explicit rule there.
  sb_update_actor_state(space, actor, 9.5, 10.5, 0.0);
  ASSERT_EQ(sb_get_scope_records(space, actor, rec.data(), 3,
                                 sizeof(sb_scope_record_t)),
            3);
  // An explicit scope root does not change the family's
  // navigable. Outside the nested sub-region the binding still uses the room.
  EXPECT_EQ(rec[1].navigable_element, room);
  EXPECT_EQ(rec[0].navigable_element, room);
  sb_destroy_session(session);
}

TEST(TraceFieldsTest, R3_5_ImminenceChangeDoesNotBumpPolicyRevision) {
  const std::string path = TempPath("obs_revision_trace.jsonl");
  Receiver r;
  r.Point(301, 2, 2);
  sb_set_trace(r.space, path.c_str());
  auto in = Payload(301, SB_KERNEL_SPACING, "root", 0.9, 1.0);
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, &in, 1, sizeof(in),
                                        0.1),
            1);
  auto p = Init<sb_realization_policy_params_t>();
  sb_get_realization_policy_params(r.space, &p, sizeof(p));
  p.kappa = 0.3;  // source-only
  ASSERT_EQ(sb_set_realization_policy_params(r.space, &p, sizeof(p)), 1);
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, &in, 1, sizeof(in),
                                        0.2),
            1);
  p.gamma = 2.0;  // receiver schedule; the default is already 1
  ASSERT_EQ(sb_set_realization_policy_params(r.space, &p, sizeof(p)), 1);
  ASSERT_EQ(sb_inject_relation_payloads(r.space, r.actor, &in, 1, sizeof(in),
                                        0.3),
            1);
  sb_set_trace(r.space, nullptr);
  std::vector<nlohmann::json> policy;
  for (const auto& ev : ReadTrace(path))
    if (ev["ev"] == "policy") policy.push_back(ev);
  ASSERT_EQ(policy.size(), 3u);
  EXPECT_EQ(policy[1]["revision"], policy[0]["revision"]);
  EXPECT_EQ(policy[1]["imminence_revision"].get<int>(),
            policy[0]["imminence_revision"].get<int>() + 1);
  EXPECT_EQ(policy[2]["revision"].get<int>(),
            policy[1]["revision"].get<int>() + 1);
  std::remove(path.c_str());
}


// ---------------------------------------------------------------------------
// Every path the header promises to keep accepting at the pre-pose-fit
// struct size (spatialblend_observability.h), including output structs.
// ---------------------------------------------------------------------------

namespace {
constexpr uint32_t kPreA4State = static_cast<uint32_t>(
    offsetof(sb_relation_state_v2_t, ap_offset_u_m));
constexpr uint32_t kPreA4Evaluation = static_cast<uint32_t>(
    offsetof(sb_relation_evaluation_t, ap_offset_u_m));
constexpr uint32_t kPreA4TargetEvaluation = static_cast<uint32_t>(
    offsetof(sb_target_relation_evaluation_t, ap_offset_u_m));
constexpr uint32_t kPreA3TargetEvaluation = static_cast<uint32_t>(
    offsetof(sb_target_relation_evaluation_t, support_basis));
constexpr unsigned char kSentinel = 0xAB;

bool TailUntouched(const std::vector<unsigned char>& buf, size_t from) {
  for (size_t i = from; i < buf.size(); ++i)
    if (buf[i] != kSentinel) return false;
  return true;
}
}  // namespace

TEST(PreA4LayoutTest, RelationStatesV2RowsStrideByTheCallersStructSize) {
  static_assert(kPreA4State < sizeof(sb_relation_state_v2_t), "A4 grew it");
  Source s;
  sb_bind(s.space, s.actor, "root", SB_KERNEL_SPACING);
  sb_update_actor_state(s.space, s.actor, 2.0, 5.0, 0.0);
  const int32_t n = sb_get_relation_states_v2(s.space, s.actor, nullptr, 0,
                                              kPreA4State);
  ASSERT_EQ(n, 2);
  std::vector<sb_relation_state_v2_t> full(n, Init<sb_relation_state_v2_t>());
  ASSERT_EQ(sb_get_relation_states_v2(s.space, s.actor, full.data(), n,
                                      sizeof(sb_relation_state_v2_t)),
            n);
  std::vector<unsigned char> buf(static_cast<size_t>(n) * kPreA4State + 64,
                                 kSentinel);
  ASSERT_EQ(sb_get_relation_states_v2(
                s.space, s.actor,
                reinterpret_cast<sb_relation_state_v2_t*>(buf.data()), n,
                kPreA4State),
            n);
  for (int32_t i = 0; i < n; ++i) {
    sb_relation_state_v2_t row;
    std::memset(&row, 0, sizeof(row));
    std::memcpy(&row, buf.data() + static_cast<size_t>(i) * kPreA4State,
                kPreA4State);
    EXPECT_EQ(row.struct_size, kPreA4State) << "row " << i;
    EXPECT_EQ(row.referent_element_id, full[i].referent_element_id);
    EXPECT_EQ(row.proximity, full[i].proximity);
    EXPECT_EQ(row.encoded_state_size, full[i].encoded_state_size);
    EXPECT_EQ(row.chart_version, full[i].chart_version);
  }
  EXPECT_TRUE(TailUntouched(buf, static_cast<size_t>(n) * kPreA4State))
      << "the pre-A4 call wrote past n rows of its own struct size";
  // An arbitrary other size is still refused.
  EXPECT_EQ(sb_get_relation_states_v2(s.space, s.actor, nullptr, 0,
                                      kPreA4State + 8),
            -1);
}

TEST(PreA4LayoutTest, EvaluateRelationWritesOnlyTheCallersLayout) {
  Source s;
  auto q = Init<sb_relation_query_t>();
  q.anchor_x = 2.0;
  q.anchor_y = 5.0;
  q.referent = s.desk;
  q.scope_root = s.room;
  q.kernel_type = SB_KERNEL_SPACING;
  auto full = Init<sb_relation_evaluation_t>();
  ASSERT_EQ(sb_evaluate_relation(s.space, &q, &full), 1);
  EXPECT_EQ(full.struct_size, sizeof(sb_relation_evaluation_t));

  std::vector<unsigned char> buf(sizeof(sb_relation_evaluation_t) + 32,
                                 kSentinel);
  auto* old = reinterpret_cast<sb_relation_evaluation_t*>(buf.data());
  old->struct_size = kPreA4Evaluation;
  ASSERT_EQ(sb_evaluate_relation(s.space, &q, old), 1);
  sb_relation_evaluation_t seen;
  std::memset(&seen, 0, sizeof(seen));
  std::memcpy(&seen, buf.data(), kPreA4Evaluation);
  EXPECT_EQ(seen.struct_size, kPreA4Evaluation);
  EXPECT_EQ(seen.proximity, full.proximity);
  EXPECT_EQ(seen.support_distance_m, full.support_distance_m);
  EXPECT_EQ(seen.encoded_state_size, full.encoded_state_size);
  EXPECT_EQ(seen.has_contact, full.has_contact);
  EXPECT_EQ(seen.has_interior, full.has_interior);
  EXPECT_TRUE(TailUntouched(buf, kPreA4Evaluation));
  // A size that is neither layout is refused.
  old->struct_size = kPreA4Evaluation + 8;
  EXPECT_EQ(sb_evaluate_relation(s.space, &q, old), 0);
}

TEST(PreA4LayoutTest, EvaluateTargetRelationAcceptsOldPayloadAndOutputSizes) {
  Receiver r;
  r.Surface(310, 5, 5, 0.8);
  sb_relation_query_t q = Init<sb_relation_query_t>();
  q.anchor_x = 5.05;
  q.anchor_y = 4.9;
  q.anchor_z = 1.2;  // 0.4 m above the surface
  // A pre-pose-fit sender: the payload struct ends before ap_offset_u_m.
  const uint32_t pre_a4_payload = static_cast<uint32_t>(
      offsetof(sb_relation_payload_t, ap_offset_u_m));
  sb_relation_payload_t old_payload = Contact(310, "hand", 1.0 / 1.15, 0.15);
  old_payload.struct_size = pre_a4_payload;

  auto full = Init<sb_target_relation_evaluation_t>();
  ASSERT_EQ(sb_evaluate_target_relation(r.space, &old_payload,
                                        pre_a4_payload, &q, &full),
            1);
  EXPECT_EQ(full.disposition, SB_TARGET_EVAL_EVALUATED);
  // Formula version 0: BOTH sides are the proximity component.
  EXPECT_EQ(full.r_formula_version, 0);
  EXPECT_EQ(full.support_basis, SB_TARGET_SUPPORT_FAMILY_GROUNDING);
  EXPECT_NEAR(full.r_src, 1.0 / 1.15, 1e-12);
  EXPECT_NEAR(full.r_tgt, 1.0 / 1.4, 1e-9);
  EXPECT_NEAR(full.error, 1.0 / 1.4 - 1.0 / 1.15, 1e-9);

  // Old OUTPUT layouts: only that prefix is written.
  for (const uint32_t out_size : {kPreA4TargetEvaluation,
                                  kPreA3TargetEvaluation}) {
    SCOPED_TRACE(out_size);
    std::vector<unsigned char> buf(sizeof(sb_target_relation_evaluation_t) +
                                       32,
                                   kSentinel);
    auto* out = reinterpret_cast<sb_target_relation_evaluation_t*>(buf.data());
    out->struct_size = out_size;
    ASSERT_EQ(sb_evaluate_target_relation(r.space, &old_payload,
                                          pre_a4_payload, &q, out),
              1);
    sb_target_relation_evaluation_t seen;
    std::memset(&seen, 0, sizeof(seen));
    std::memcpy(&seen, buf.data(), out_size);
    EXPECT_EQ(seen.struct_size, out_size);
    EXPECT_EQ(seen.disposition, SB_TARGET_EVAL_EVALUATED);
    EXPECT_EQ(seen.r_tgt, full.r_tgt);
    EXPECT_EQ(seen.r_src, full.r_src);
    EXPECT_EQ(seen.has_mapped_support, 1);
    EXPECT_TRUE(TailUntouched(buf, out_size));
  }
  // Neither layout: refused.
  auto bad = Init<sb_target_relation_evaluation_t>();
  bad.struct_size = kPreA4TargetEvaluation + 8;
  EXPECT_EQ(sb_evaluate_target_relation(r.space, &old_payload, pre_a4_payload,
                                        &q, &bad),
            0);
  // Current payload layout WITHOUT r_formula_version (zero-initialised
  // offsets) is a formula-version-0 sender too: zeros never read as
  // "exactly on the support".
  sb_relation_payload_t zeroed = Contact(310, "hand", 1.0 / 1.15, 0.15);
  full = Init<sb_target_relation_evaluation_t>();
  ASSERT_EQ(sb_evaluate_target_relation(r.space, &zeroed, sizeof(zeroed), &q,
                                        &full),
            1);
  EXPECT_EQ(full.r_formula_version, 0);
  EXPECT_NEAR(full.r_src, 1.0 / 1.15, 1e-12);
  EXPECT_NEAR(full.r_tgt, 1.0 / 1.4, 1e-9);
  // A pose-fit sender (formula version 1 + offsets) is scored with the pose fit on
  // both sides: position 0.5, height 0.3, yaw 0.2 (candidate yaw = the
  // source's relative yaw, encoded_state[2] = 0).
  sb_relation_payload_t a4 = Contact(310, "hand", 1.0 / 1.15, 0.15);
  a4.r_formula_version = 1;
  a4.ap_offset_u_m = 0.0;
  a4.ap_offset_v_m = 0.0;
  a4.ap_height_m = 0.15;
  full = Init<sb_target_relation_evaluation_t>();
  ASSERT_EQ(sb_evaluate_target_relation(r.space, &a4, sizeof(a4), &q, &full),
            1);
  EXPECT_EQ(full.r_formula_version, 1);
  EXPECT_NEAR(full.r_src, 0.5 + 0.3 / 1.15 + 0.2, 1e-9);
  // The payload names no part, so the support is the family's own grounding
  // at the candidate (its projection onto the surface): in-plane offset 0
  // (position fit 1), height 0.4 -> 0.3/1.4, yaw exact -> 0.2. (The mapped
  // support applies to a NAMED part; see the support-basis test below.)
  EXPECT_NEAR(full.r_tgt, 0.5 + 0.3 / 1.4 + 0.2, 1e-9);
  EXPECT_NEAR(full.error, full.r_tgt - full.r_src, 1e-12);
  EXPECT_EQ(sb_sizeof(SB_STRUCT_TARGET_RELATION_EVALUATION),
            sizeof(sb_target_relation_evaluation_t));
}

// The fallback to the candidate's own projection is exposed through the
// ABI as support_basis + reason, never silent.
TEST(TargetEvaluationTest, A3_SupportBasisExposesTheMappedSupportFallback) {
  Receiver r;
  const sb_element_t surface = r.Surface(310, 5, 5, 0.8);
  const double patch[] = {4.8, 4.8, 5.2, 4.8, 5.2, 5.2, 4.8, 5.2};
  const double at_top[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.8, 0, 0, 0, 1};
  sb_add_sub_region(r.space, surface, "top", "Sittable", patch, 4, at_top);

  sb_relation_query_t q = Init<sb_relation_query_t>();
  q.anchor_x = 5.05;
  q.anchor_y = 4.9;
  q.anchor_z = 1.2;
  // A payload with the source's chart: scored against the MAPPED support.
  sb_relation_payload_t mapped = Contact(310, "hand", 1.0 / 1.15, 0.15);
  mapped.sub_region_name = "top";
  auto out = Init<sb_target_relation_evaluation_t>();
  ASSERT_EQ(sb_evaluate_target_relation(r.space, &mapped, sizeof(mapped), &q,
                                        &out),
            1);
  ASSERT_EQ(out.disposition, SB_TARGET_EVAL_EVALUATED) << out.reason;
  EXPECT_EQ(out.support_basis, SB_TARGET_SUPPORT_MAPPED);
  EXPECT_STREQ(out.reason, "");

  // The same part named, but no contact chart to map: an explicit degraded
  // fallback to the candidate's own projection, with the reason.
  sb_relation_payload_t legacy = mapped;
  legacy.has_contact = 0;
  out = Init<sb_target_relation_evaluation_t>();
  ASSERT_EQ(sb_evaluate_target_relation(r.space, &legacy, sizeof(legacy), &q,
                                        &out),
            1);
  EXPECT_EQ(out.disposition, SB_TARGET_EVAL_EVALUATED);
  EXPECT_EQ(out.support_basis, SB_TARGET_SUPPORT_CANDIDATE_PROJECTION_FALLBACK);
  EXPECT_STRNE(out.reason, "");
}
