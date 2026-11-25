// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Unit tests for the anneal-based realization policy.

#include "spatialblend/solver/realization_policy.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace spatialblend {
namespace {


// ===========================================================================
// ComputeAnnealedWeights — v1's progressive weighted annealing in v2.
// ===========================================================================
namespace {
RelationPayload AnnealStepsPayload(shared_element_id_t ref, double update_count,
                                   const std::string& anchor = "root") {
  RelationPayload p;
  p.referent_shared_id = ref;
  p.kernel_type = RelationKernelType::kAvatarSpacing;
  p.subject_anchor_name = anchor;
  // The scheduling quantity is the update count, not a time-to-contact.
  p.update_count = update_count;
  return p;
}
}  // namespace

TEST(AnnealPolicy, CommitsToApproachedReferent) {
  // A is being approached (small steps); B is receding (huge steps, the value
  // space.cc assigns when not closing). Annealing must put ~all weight on A.
  auto w = ComputeAnnealedWeights(
      {AnnealStepsPayload(300, 3.0), AnnealStepsPayload(301, 5000.0)});
  ASSERT_EQ(w.size(), 2u);
  EXPECT_GT(w[0], 0.95) << "approached referent should win";
  EXPECT_LT(w[1], 0.05) << "receding referent should be dropped";
}

TEST(AnnealPolicy, OneHotAtContact) {
  // Temperature decays as the nearest approaches contact → one-hot.
  auto w = ComputeAnnealedWeights(
      {AnnealStepsPayload(300, 1.05), AnnealStepsPayload(301, 8.0)});
  ASSERT_EQ(w.size(), 2u);
  EXPECT_GT(w[0], 0.99);
}

TEST(AnnealPolicy, ParkedFallbackOrdersByDistanceNearestWins) {
  // With no closing intent the count is large. Annealing still favors the
  // smaller count.
  RelationPayload near_obj = AnnealStepsPayload(300, 8.3 / 1e-4);   // nearest
  RelationPayload far_obj = AnnealStepsPayload(301, 13.4 / 1e-4);   // farthest
  auto w = ComputeAnnealedWeights({near_obj, far_obj});
  ASSERT_EQ(w.size(), 2u);
  EXPECT_GT(w[0], w[1]) << "nearest object must lead under the parked fallback";
}

TEST(AnnealPolicy, SeparateGroupsPerKernelAnchor) {
  // Spacing and facing must not share a temperature budget — each group
  // normalizes to 1 independently.
  RelationPayload spacing = AnnealStepsPayload(300, 2.0);
  RelationPayload facing = AnnealStepsPayload(301, 2.0, "chest");
  facing.kernel_type = RelationKernelType::kAvatarFacing;
  auto w = ComputeAnnealedWeights({spacing, facing});
  ASSERT_EQ(w.size(), 2u);
  EXPECT_NEAR(w[0], 1.0, 1e-9) << "sole member of spacing group";
  EXPECT_NEAR(w[1], 1.0, 1e-9) << "sole member of facing group";
}

namespace {
RelationPayload MakePayload(shared_element_id_t referent_shared_id,
                           double update_count,
                           RelationKernelType kernel_type,
                           std::string anchor) {
  RelationPayload p;
  p.referent_shared_id = referent_shared_id;
  p.kernel_type = kernel_type;
  p.subject_anchor_name = std::move(anchor);
  p.update_count = update_count;
  return p;
}

double SumOf(const std::vector<double>& weights, const std::vector<size_t>& idx) {
  double sum = 0.0;
  for (size_t i : idx) {
    sum += weights[i];
  }
  return sum;
}
}  // namespace

TEST(RealizationPolicyTest, EmptyPayloadsReturnEmpty) {
  std::vector<RealizationPolicyType> types = {
      RealizationPolicyType::kUniform,
      RealizationPolicyType::kGreedy,
      RealizationPolicyType::kProgressive,
  };

  for (auto policy : types) {
    RealizationPolicyConfig cfg;
    cfg.type = policy;
    EXPECT_TRUE(ComputeRealizationWeights({}, cfg).empty());
  }
}

TEST(RealizationPolicyTest, ProgressiveUsesSecondsScheduleWithoutStepOffset) {
  std::vector<RelationPayload> payloads = {
      MakePayload(1, 2.0, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(2, 3.0, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(3, 4.0, RelationKernelType::kAvatarSpacing, "root"),
  };
  const auto weights = ComputeRealizationWeights(payloads);
  ASSERT_EQ(weights.size(), 3u);
  // v1 SoftAnnealing: n_min=2, η_eff = (n_min − 1 + ε)^γ = 1.001.
  const double q = std::exp(-1.0 / 1.001);
  const double denominator = 1.0 + q + q * q;
  EXPECT_NEAR(weights[0], 1.0 / denominator, 1e-12);
  EXPECT_NEAR(weights[1], q / denominator, 1e-12);
  EXPECT_NEAR(weights[2], q * q / denominator, 1e-12);
}

TEST(RealizationPolicyTest, ProgressiveRejectsInvalidSecondsWithoutNanWeights) {
  std::vector<RelationPayload> payloads = {
      MakePayload(1, 0.25, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(2, -1.0, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(3, std::numeric_limits<double>::quiet_NaN(),
                  RelationKernelType::kAvatarSpacing, "root"),
  };
  const auto weights = ComputeRealizationWeights(payloads);
  ASSERT_EQ(weights.size(), 3u);
  EXPECT_NEAR(weights[0], 1.0, 1e-12);
  EXPECT_DOUBLE_EQ(weights[1], 0.0);
  EXPECT_DOUBLE_EQ(weights[2], 0.0);
  for (double weight : weights) EXPECT_TRUE(std::isfinite(weight));
}

TEST(RealizationPolicyTest, GreedyTreatsNegativeTimeAsInvalid) {
  std::vector<RelationPayload> payloads = {
      MakePayload(1, -1.0, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(2, 2.0, RelationKernelType::kAvatarSpacing, "root"),
  };
  RealizationPolicyConfig config;
  config.type = RealizationPolicyType::kGreedy;
  const auto weights = ComputeRealizationWeights(payloads, config);
  EXPECT_DOUBLE_EQ(weights[0], 0.0);
  EXPECT_DOUBLE_EQ(weights[1], 1.0);
}

TEST(RealizationPolicyTest, ProgressiveMatchesV1SoftAnnealing) {
  // Counts are update counts (n >= 1). γ = 1 is the v1 runtime schedule;
  // the earlier γ = 2 seconds schedule is not the quantity.
  std::vector<RelationPayload> payloads = {
      MakePayload(1, 2.0, RelationKernelType::kAvatarSpacing, "navigation"),
      MakePayload(2, 3.0, RelationKernelType::kAvatarSpacing, "navigation"),
  };
  RealizationPolicyConfig config = ProductionRealizationPolicyConfig();

  const auto weights = ComputeRealizationWeights(payloads, config);
  const double eta = std::pow(2.0 - 1.0 + config.epsilon, config.decay_gamma);
  const double q = std::exp(-1.0 / eta);
  EXPECT_NEAR(weights[0], 1.0 / (1.0 + q), 1e-12);
  EXPECT_NEAR(weights[1], q / (1.0 + q), 1e-12);
}

TEST(RealizationPolicyTest, UniformPolicyPerPairKernelAnchorAndHandIsolation) {
  std::vector<RelationPayload> payloads = {
      MakePayload(11, 1.0, RelationKernelType::kAvatarSpacing, "left_hand"),
      MakePayload(12, 9.0, RelationKernelType::kAvatarSpacing, "left_hand"),
      MakePayload(13, 5.0, RelationKernelType::kAvatarSpacing, "right_hand"),
      MakePayload(14, 7.0, RelationKernelType::kAvatarSpacing, "right_hand"),
  };
  RealizationPolicyConfig cfg;
  cfg.type = RealizationPolicyType::kUniform;

  const auto w = ComputeRealizationWeights(payloads, cfg);
  ASSERT_EQ(w.size(), 4u);
  EXPECT_NEAR(SumOf(w, {0, 1}), 1.0, 1e-12);
  EXPECT_NEAR(SumOf(w, {2, 3}), 1.0, 1e-12);
  EXPECT_NEAR(w[0], 0.5, 1e-12);
  EXPECT_NEAR(w[1], 0.5, 1e-12);
  EXPECT_NEAR(w[2], 0.5, 1e-12);
  EXPECT_NEAR(w[3], 0.5, 1e-12);

  payloads[0].update_count = 123.0;
  payloads[1].update_count = 456.0;
  auto w_perturbed = ComputeRealizationWeights(payloads, cfg);
  ASSERT_EQ(w_perturbed.size(), 4u);
  EXPECT_NEAR(w_perturbed[2], w[2], 1e-12);
  EXPECT_NEAR(w_perturbed[3], w[3], 1e-12);
}

TEST(RealizationPolicyTest,
     GreedyPolicyPerPairKernelAnchorAndDeterministicTieBreak) {
  std::vector<RelationPayload> payloads = {
      MakePayload(3, 10.0, RelationKernelType::kAvatarFacing, "left_hand"),
      MakePayload(1, 10.0, RelationKernelType::kAvatarFacing, "left_hand"),
      MakePayload(20, 0.5, RelationKernelType::kAvatarFacing, "right_hand"),
      MakePayload(21, 1.5, RelationKernelType::kAvatarFacing, "right_hand"),
  };
  RealizationPolicyConfig cfg;
  cfg.type = RealizationPolicyType::kGreedy;

  auto w = ComputeRealizationWeights(payloads, cfg);
  ASSERT_EQ(w.size(), 4u);
  EXPECT_NEAR(SumOf(w, {0, 1}), 1.0, 1e-12);
  EXPECT_NEAR(SumOf(w, {2, 3}), 1.0, 1e-12);
  EXPECT_NEAR(w[1], 1.0, 1e-12);  // smaller referent id wins tie.
  EXPECT_NEAR(w[0], 0.0, 1e-12);
  EXPECT_NEAR(w[2], 1.0, 1e-12);
  EXPECT_NEAR(w[3], 0.0, 1e-12);

  payloads[0].update_count = 100.0;
  payloads[1].update_count = 100.0;
  auto w_perturbed = ComputeRealizationWeights(payloads, cfg);
  ASSERT_EQ(w_perturbed.size(), 4u);
  EXPECT_NEAR(w_perturbed[2], 1.0, 1e-12);
  EXPECT_NEAR(w_perturbed[3], 0.0, 1e-12);
}

TEST(RealizationPolicyTest, GreedyPolicyHandlesNonfiniteTieInputs) {
  std::vector<RelationPayload> payloads = {
      MakePayload(99, std::numeric_limits<double>::infinity(),
                  RelationKernelType::kAvatarSpacing, "left_hand"),
      MakePayload(42, std::numeric_limits<double>::infinity(),
                  RelationKernelType::kAvatarSpacing, "left_hand"),
      MakePayload(200, 1.0, RelationKernelType::kAvatarFacing, "right_hand"),
      MakePayload(201, 2.0, RelationKernelType::kAvatarFacing, "right_hand"),
  };
  RealizationPolicyConfig cfg;
  cfg.type = RealizationPolicyType::kGreedy;

  auto w = ComputeRealizationWeights(payloads, cfg);
  ASSERT_EQ(w.size(), 4u);
  EXPECT_NEAR(SumOf(w, {0, 1}), 1.0, 1e-12);
  EXPECT_NEAR(w[1], 1.0, 1e-12);
  EXPECT_NEAR(SumOf(w, {2, 3}), 1.0, 1e-12);
  EXPECT_NEAR(w[2], 1.0, 1e-12);
}

TEST(RealizationPolicyTest,
     ProgressivePolicyPerPairKernelAnchorAndMixedFamilies) {
  std::vector<RelationPayload> payloads = {
      MakePayload(31, 5.0, RelationKernelType::kAvatarSpacing, "left_hand"),
      MakePayload(32, 6.0, RelationKernelType::kAvatarSpacing, "left_hand"),
      MakePayload(33, 9.0, RelationKernelType::kAvatarFacing, "left_hand"),
      MakePayload(34, 2.0, RelationKernelType::kAvatarFacing, "left_hand"),
      MakePayload(35, 8.0, RelationKernelType::kAvatarSpacing, "right_hand"),
      MakePayload(36, 10.0, RelationKernelType::kAvatarSpacing, "right_hand"),
  };
  RealizationPolicyConfig cfg;  // kProgressive default.

  const auto w = ComputeRealizationWeights(payloads, cfg);
  ASSERT_EQ(w.size(), 6u);
  EXPECT_NEAR(SumOf(w, {0, 1}), 1.0, 1e-12);
  EXPECT_NEAR(SumOf(w, {2, 3}), 1.0, 1e-12);
  EXPECT_NEAR(SumOf(w, {4, 5}), 1.0, 1e-12);

  payloads[4].update_count = 1000.0;
  payloads[5].update_count = 1001.0;
  auto w_perturbed = ComputeRealizationWeights(payloads, cfg);
  ASSERT_EQ(w_perturbed.size(), 6u);
  EXPECT_NEAR(w_perturbed[0], w[0], 1e-12);
  EXPECT_NEAR(w_perturbed[1], w[1], 1e-12);
  EXPECT_NEAR(w_perturbed[2], w[2], 1e-12);
  EXPECT_NEAR(w_perturbed[3], w[3], 1e-12);
}

TEST(RealizationPolicyTest, ProgressivePolicyMakesNoFacingRequestForAnInvalidGroup) {
  std::vector<RelationPayload> payloads = {
      MakePayload(101, std::numeric_limits<double>::infinity(),
                  RelationKernelType::kAvatarFacing, "left_hand"),
      MakePayload(102, std::numeric_limits<double>::infinity(),
                  RelationKernelType::kAvatarFacing, "left_hand"),
      MakePayload(103, 3.0, RelationKernelType::kAvatarSpacing, "left_hand"),
      MakePayload(104, 8.0, RelationKernelType::kAvatarSpacing, "left_hand"),
  };
  RealizationPolicyConfig cfg;
  cfg.type = RealizationPolicyType::kProgressive;

  auto w = ComputeRealizationWeights(payloads, cfg);
  ASSERT_EQ(w.size(), 4u);
  // v1 plans nothing for a Facing group with no valid angular count: no yaw
  // request, not a uniform mean.
  EXPECT_EQ(w[0], 0.0);
  EXPECT_EQ(w[1], 0.0);
  EXPECT_NEAR(SumOf(w, {2, 3}), 1.0, 1e-12);
  EXPECT_GT(w[2], w[3]);
}

// ===========================================================================
// Declared parameters. Channel diagnostics (driving relation).
// ===========================================================================

TEST(RealizationPolicyParamsTest, R3_5_ProductionDeviationIsDeclared) {
  const RealizationPolicyConfig paper;
  const RealizationPolicyConfig production = ProductionRealizationPolicyConfig();
  EXPECT_DOUBLE_EQ(paper.decay_gamma, 1.0);
  // Production is v1 SoftAnnealing. The γ = 2 deviation is retired.
  EXPECT_DOUBLE_EQ(production.decay_gamma, 1.0);
  EXPECT_DOUBLE_EQ(production.eta_0, paper.eta_0);
  EXPECT_DOUBLE_EQ(production.epsilon, paper.epsilon);
  EXPECT_DOUBLE_EQ(production.eta_min, paper.eta_min);
  EXPECT_DOUBLE_EQ(production.eta_max, paper.eta_max);
  EXPECT_TRUE(IsValidRealizationPolicyConfig(production));
  RealizationPolicyConfig bad = production;
  bad.eta_max = bad.eta_min / 2;
  EXPECT_FALSE(IsValidRealizationPolicyConfig(bad));
  const ImminenceConfig imminence;
  EXPECT_TRUE(IsValidImminenceConfig(imminence));
  EXPECT_DOUBLE_EQ(imminence.t_max, 1.0 / imminence.epsilon_v);
}

TEST(RealizationChannelDiagnosticsTest, R3_6_DrivingIsArgmaxPerChannel) {
  const std::vector<RelationPayload> payloads = {
      MakePayload(10, 3.0, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(11, 1.0, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(12, 0.5, RelationKernelType::kAvatarAffordancePose, "hand"),
      MakePayload(13, 2.0, RelationKernelType::kAvatarAffordancePose, "hand"),
  };
  RealizationPolicyConfig config = ProductionRealizationPolicyConfig();
  const auto weights = ComputeRealizationWeights(payloads, config);
  const auto channels = DescribeRealizationChannels(payloads, weights, config);
  ASSERT_EQ(channels.size(), 2u);
  for (const auto& c : channels) {
    ASSERT_TRUE(c.has_driving);
    EXPECT_FALSE(c.driving_tied);
    double max_w = 0.0;
    for (size_t i : c.payload_indices) max_w = std::max(max_w, weights[i]);
    EXPECT_DOUBLE_EQ(c.driving_weight, max_w);
    if (c.kernel_type == RelationKernelType::kAvatarSpacing) {
      EXPECT_EQ(c.driving_index, 1u);  // n = 1 beats n = 3
      EXPECT_DOUBLE_EQ(c.t_min, 1.0);
      EXPECT_NEAR(c.eta, 1e-3, 1e-12);  // η_eff = (1 − 1 + ε)^1
    } else {
      EXPECT_EQ(c.anchor_name, "hand");
      EXPECT_EQ(c.driving_index, 2u);
      EXPECT_DOUBLE_EQ(c.t_min, 0.5);
    }
  }

  config.type = RealizationPolicyType::kUniform;
  const auto uniform = DescribeRealizationChannels(
      payloads, ComputeRealizationWeights(payloads, config), config);
  for (const auto& c : uniform) {
    EXPECT_TRUE(c.driving_tied);
    EXPECT_EQ(c.driving_index, c.payload_indices.front());
    EXPECT_TRUE(std::isnan(c.eta));
    EXPECT_TRUE(std::isnan(c.t_min));
  }

  // A disabled payload belongs to no channel and cannot drive.
  std::vector<RelationPayload> disabled = payloads;
  disabled[1].enabled = false;
  config.type = RealizationPolicyType::kProgressive;
  const auto d = DescribeRealizationChannels(
      disabled, ComputeRealizationWeights(disabled, config), config);
  for (const auto& c : d) {
    for (size_t i : c.payload_indices) EXPECT_NE(i, 1u);
    if (c.kernel_type == RelationKernelType::kAvatarSpacing) {
      EXPECT_EQ(c.driving_index, 0u);
      EXPECT_DOUBLE_EQ(c.driving_weight, 1.0);
      EXPECT_TRUE(std::isnan(c.eta));  // single member: no temperature
    }
  }
}

}  // namespace
// ---- The three policies behave as v1's planners. ----
namespace {
constexpr double kNoCount = std::numeric_limits<double>::infinity();
std::vector<double> WeightsOf(RealizationPolicyType type, const std::vector<RelationPayload>& ps) {
  RealizationPolicyConfig c;
  c.type = type;
  return ComputeRealizationWeights(ps, c);
}
}  // namespace

// Progressive = v1 SoftAnnealing: the valid counts are annealed as v1's
// ComputeSoftAnnealingWeights (eta_0 = 1, gamma = 1, decay on, epsilon = 1e-3), and a relation with no valid count
// takes weight 0 (v1 BuildTargets skips it).
TEST(RealizationPolicyV1, ProgressiveIsV1SoftAnnealingOverTheValidCounts) {
  // Hand-computed from v1's formula (soft annealing, eta_0 = 1, gamma = 1, epsilon = 1e-3):
  //   s_min = 1.999, eta_eff = eta_0 (s_min - 1 + epsilon)^gamma = 1.999 - 1 + 0.001 = 1 (exact in binary64);
  //   w_i = exp(-(s_i - s_min) / eta_eff) / sum: (1, e^-1) / (1 + e^-1) = (e / (e + 1), 1 / (e + 1))
  //       = (0.7310585786300049, 0.2689414213699951).
  // The invalid counts (no count, a negative one) take no weight: v1 BuildTargets skips them.
  const std::vector<RelationPayload> ps = {
      MakePayload(1, 1.999, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(2, 2.999, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(3, kNoCount, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(4, -1.0, RelationKernelType::kAvatarSpacing, "root")};
  const auto w = WeightsOf(RealizationPolicyType::kProgressive, ps);
  ASSERT_EQ(w.size(), 4u);
  EXPECT_NEAR(w[0], 0.7310585786300049, 1e-15);
  EXPECT_NEAR(w[1], 0.2689414213699951, 1e-15);
  EXPECT_EQ(w[2], 0.0);
  EXPECT_EQ(w[3], 0.0);
}

// v1's other branches, hand-computed with gamma != 1 (weight.cc ComputeSoftAnnealingWeights, read, not run):
//   eta_eff = eta_0 (s_min - 1 + eps)^gamma; safe = max(eta_eff, eps^2); safe < eps -> one-hot on the first minimum;
//   otherwise w_i = exp(-(s_i - s_min) / safe) / sum.
std::vector<double> ProgressiveWith(double eta_0, double gamma, double eps, const std::vector<double>& counts) {
  std::vector<RelationPayload> ps;
  for (size_t i = 0; i < counts.size(); ++i)
    ps.push_back(MakePayload(static_cast<shared_element_id_t>(i + 1), counts[i], RelationKernelType::kAvatarSpacing,
                             "root"));
  RealizationPolicyConfig c;
  c.type = RealizationPolicyType::kProgressive;
  c.eta_0 = eta_0;
  c.decay_gamma = gamma;
  c.epsilon = eps;
  return ComputeRealizationWeights(ps, c);
}

TEST(RealizationPolicyV1Golden, ProgressiveOneHotBranchByHand) {
  // gamma = 2, eps = 1e-3, {1, 1.0005}: eta_eff = 1 (1 - 1 + 1e-3)^2 = 1e-6; safe = max(1e-6, 1e-6) = 1e-6 < 1e-3:
  // one-hot on the first minimum, (1, 0). (The softmax would have given 1 / (1 + e^-500), not exactly 1.)
  EXPECT_EQ(ProgressiveWith(1.0, 2.0, 1e-3, {1.0, 1.0005}), (std::vector<double>{1.0, 0.0}));
  // A tie at the minimum: the FIRST minimum takes it, (0, 1, 0) for {1.5, 1, 1}.
  EXPECT_EQ(ProgressiveWith(1.0, 2.0, 1e-3, {1.5, 1.0, 1.0}), (std::vector<double>{0.0, 1.0, 0.0}));
}

TEST(RealizationPolicyV1Golden, ProgressiveEpsilonSquaredClampByHand) {
  // eps = 2, gamma = 2, eta_0 = 0.1, {1, 3}: eta_eff = 0.1 (1 - 1 + 2)^2 = 0.4; safe = max(0.4, eps^2 = 4) = 4, not below
  // eps, so the softmax at 4: (1, e^(-2/4)) / (1 + e^-0.5) = (0.6224593312018546, 0.37754066879814546).
  const auto w = ProgressiveWith(0.1, 2.0, 2.0, {1.0, 3.0});
  ASSERT_EQ(w.size(), 2u);
  EXPECT_NEAR(w[0], 0.6224593312018546, 1e-15);
  EXPECT_NEAR(w[1], 0.37754066879814546, 1e-15);
}

TEST(RealizationPolicyV1Golden, ProgressiveGammaTwoByHand) {
  // gamma = 2, eps = 1e-3, {2.999, 6.999}: s_min - 1 + eps = 2.999 - 1 + 0.001 = 2 (exact in binary64), eta_eff = 2^2 = 4;
  // the softmax at 4: (1, e^(-(6.999 - 2.999)/4)) / sum, with 6.999 - 2.999 = 3.9999999999999996 in binary64 (not 4: the
  // exponent differs from -1 by 1e-16, inside the comparison's 1e-15), so (e / (e + 1), 1 / (e + 1)) =
  // (0.7310585786300049, 0.2689414213699951).
  const auto w = ProgressiveWith(1.0, 2.0, 1e-3, {2.999, 6.999});
  ASSERT_EQ(w.size(), 2u);
  EXPECT_NEAR(w[0], 0.7310585786300049, 1e-15);
  EXPECT_NEAR(w[1], 0.2689414213699951, 1e-15);
}

// Progressive with NO valid count computes no weight (v1 sends no request): every weight
// is 0, for a group and for a lone relation alike.
TEST(RealizationPolicyV1, ProgressiveWithNoValidCountComputesNoWeight) {
  EXPECT_EQ(WeightsOf(RealizationPolicyType::kProgressive,
                      {MakePayload(1, kNoCount, RelationKernelType::kAvatarSpacing, "root"),
                       MakePayload(2, -1.0, RelationKernelType::kAvatarSpacing, "root")}),
            (std::vector<double>{0.0, 0.0}));
  EXPECT_EQ(WeightsOf(RealizationPolicyType::kProgressive,
                      {MakePayload(1, kNoCount, RelationKernelType::kAvatarSpacing, "root")}),
            (std::vector<double>{0.0}));
  EXPECT_EQ(WeightsOf(RealizationPolicyType::kProgressive,
                      {MakePayload(1, 4.0, RelationKernelType::kAvatarSpacing, "root")}),
            (std::vector<double>{1.0}))
      << "a lone relation with a valid count: SoftAnnealing over one count is 1";
}

// Greedy = v1 NearestFirst: weight 1 on the intent with the fewest updates
// (InteractionTimeline orders the intents by count), 0 on the others.
TEST(RealizationPolicyV1, GreedyIsV1NearestFirst) {
  const std::vector<RelationPayload> ps = {
      MakePayload(1, 5.0, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(2, 2.0, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(3, kNoCount, RelationKernelType::kAvatarSpacing, "root")};
  const auto w = WeightsOf(RealizationPolicyType::kGreedy, ps);
  EXPECT_EQ(w, (std::vector<double>{0.0, 1.0, 0.0}));
}

// Uniform = v1 MinProximityError: 1/N over every target, whatever its count.
TEST(RealizationPolicyV1, UniformIsV1MinProximityErrorWeights) {
  const std::vector<RelationPayload> ps = {
      MakePayload(1, 5.0, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(2, kNoCount, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(3, 2.0, RelationKernelType::kAvatarSpacing, "root"),
      MakePayload(4, 1.0, RelationKernelType::kAvatarSpacing, "root")};
  const auto w = WeightsOf(RealizationPolicyType::kUniform, ps);
  for (double x : w) EXPECT_DOUBLE_EQ(x, 0.25);
}

// Progressive is the default, and the policy is selectable by name as before (the env var and the C ABI map these).
TEST(RealizationPolicyV1, ProgressiveIsTheDefaultAndEveryPolicyIsNamed) {
  EXPECT_EQ(RealizationPolicyConfig{}.type, RealizationPolicyType::kProgressive);
  EXPECT_STREQ(RealizationPolicyName(RealizationPolicyType::kProgressive), "progressive");
  EXPECT_STREQ(RealizationPolicyName(RealizationPolicyType::kUniform), "uniform");
  EXPECT_STREQ(RealizationPolicyName(RealizationPolicyType::kGreedy), "greedy");
}

// Goldens computed by v1's own planner code (NearestFirstPlanner::GetLastWeights over
// InteractionTimeline::FilterToValidSteps and MinProximityErrorPlanner::GetLastWeights over a ProximityVector,
// printed %.17g). The Progressive values are NOT generated:
// v1's ComputeSoftAnnealingWeights is the same weight.cc the code under test calls, so they are computed by
// hand from v1's formula, the arithmetic written beside each.
//   greedy {1:5, 2:2, 3:inf}        -> id2=1 id1=0          (id3 filtered out: no valid count)
//   greedy {7:3, 3:3, 5:4}          -> id3=1 id7=0 id5=0    (a tie goes to the lower id: the timeline's order)
//   greedy {1:1, 2:7.5, 3:1.5}      -> id1=1 id3=0 id2=0
//   uniform {1,2,3}                 -> 0.33333333333333331 each
TEST(RealizationPolicyV1Golden, GreedyMatchesV1NearestFirst) {
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(WeightsOf(RealizationPolicyType::kGreedy,
                      {MakePayload(1, 5.0, RelationKernelType::kAvatarSpacing, "root"),
                       MakePayload(2, 2.0, RelationKernelType::kAvatarSpacing, "root"),
                       MakePayload(3, inf, RelationKernelType::kAvatarSpacing, "root")}),
            (std::vector<double>{0.0, 1.0, 0.0}));
  EXPECT_EQ(WeightsOf(RealizationPolicyType::kGreedy,
                      {MakePayload(7, 3.0, RelationKernelType::kAvatarSpacing, "root"),
                       MakePayload(3, 3.0, RelationKernelType::kAvatarSpacing, "root"),
                       MakePayload(5, 4.0, RelationKernelType::kAvatarSpacing, "root")}),
            (std::vector<double>{0.0, 1.0, 0.0}))
      << "v1 breaks a tie by the lower target id";
  EXPECT_EQ(WeightsOf(RealizationPolicyType::kGreedy,
                      {MakePayload(1, 1.0, RelationKernelType::kAvatarSpacing, "root"),
                       MakePayload(2, 7.5, RelationKernelType::kAvatarSpacing, "root"),
                       MakePayload(3, 1.5, RelationKernelType::kAvatarSpacing, "root")}),
            (std::vector<double>{1.0, 0.0, 0.0}));
}

TEST(RealizationPolicyV1Golden, UniformMatchesV1MinProximityError) {
  const auto w = WeightsOf(RealizationPolicyType::kUniform,
                           {MakePayload(1, 5.0, RelationKernelType::kAvatarSpacing, "root"),
                            MakePayload(2, kNoCount, RelationKernelType::kAvatarSpacing, "root"),
                            MakePayload(3, 2.0, RelationKernelType::kAvatarSpacing, "root")});
  ASSERT_EQ(w.size(), 3u);
  for (double x : w) EXPECT_EQ(x, 0.33333333333333331);
}

TEST(RealizationPolicyV1Golden, ProgressiveIsV1SoftAnnealingByHand) {
  // {1, 7.5, 1.5}: s_min = 1 (at contact), eta_eff = 1 (1 - 1 + 1e-3)^1 = 1e-3; max(eta_eff, eps^2) = 1e-3, not below
  // eps, so the softmax: (e^0, e^(-6.5/1e-3), e^(-0.5/1e-3)) = (1, e^-6500, e^-500) = (1, 0, 7.124576406741286e-218)
  // in binary64; sum = 1 + 7.1e-218 = 1, so the weights are those numbers.
  auto w = WeightsOf(RealizationPolicyType::kProgressive,
                     {MakePayload(1, 1.0, RelationKernelType::kAvatarSpacing, "root"),
                      MakePayload(2, 7.5, RelationKernelType::kAvatarSpacing, "root"),
                      MakePayload(3, 1.5, RelationKernelType::kAvatarSpacing, "root")});
  ASSERT_EQ(w.size(), 3u);
  EXPECT_EQ(w[0], 1.0);
  EXPECT_EQ(w[1], 0.0);
  EXPECT_NEAR(w[2], 7.124576406741286e-218, 1e-230);
  // {1.999, 1.999, 2.999}: eta_eff = 1.999 - 1 + 0.001 = 1; (1, 1, e^-1) / (2 + e^-1):
  //   1 / 2.3678794411714423 = 0.4223187982515182 (twice), 0.36787944117144233 / 2.3678794411714423 = 0.15536240349696362.
  w = WeightsOf(RealizationPolicyType::kProgressive,
                {MakePayload(1, 1.999, RelationKernelType::kAvatarSpacing, "root"),
                 MakePayload(2, 1.999, RelationKernelType::kAvatarSpacing, "root"),
                 MakePayload(3, 2.999, RelationKernelType::kAvatarSpacing, "root")});
  ASSERT_EQ(w.size(), 3u);
  EXPECT_NEAR(w[0], 0.4223187982515182, 1e-15);
  EXPECT_NEAR(w[1], 0.4223187982515182, 1e-15);
  EXPECT_NEAR(w[2], 0.15536240349696362, 1e-15);
}


}  // namespace spatialblend
