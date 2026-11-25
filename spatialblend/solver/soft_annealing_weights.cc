// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/solver/soft_annealing_weights.h"

#include <algorithm>
#include <cmath>

namespace spatialblend {

std::vector<double> ComputeSoftAnnealingWeights(
    const std::vector<double>& steps,
    double eta,
    bool enable_decay,
    double decay_gamma,
    double epsilon) {
  const size_t n = steps.size();
  if (n == 0) return {};
  if (n == 1) return {1.0};

  std::vector<double> weights(n, 0.0);

  // Find minimum for numerical stability and decay computation
  size_t i_star = 0;
  double s_min = steps[0];
  for (size_t i = 1; i < n; ++i) {
    if (steps[i] < s_min) {
      s_min = steps[i];
      i_star = i;
    }
  }

  // Guard: if s_min is not finite (all steps are infinity or NaN), the softmax
  // subtraction (steps[i] - s_min) produces NaN. Return equal weights since
  // all targets are equally unreachable.
  if (!std::isfinite(s_min)) {
    double uniform = 1.0 / static_cast<double>(n);
    return std::vector<double>(n, uniform);
  }

  // Compute effective temperature
  // With decay: η_eff = η_0 * (s_min - 1 + ε)^γ
  // As s_min → 1, η_eff → η_0 * ε^γ ≈ 0, making softmax → one-hot
  double eta_eff = eta;
  if (enable_decay) {
    double decay_factor = std::pow(s_min - 1.0 + epsilon, decay_gamma);
    eta_eff = eta * decay_factor;
  }
  double safe_eta = std::max(eta_eff, epsilon * epsilon);  // Ensure positive

  // At very low effective temperature, return one-hot for numerical stability
  if (safe_eta < epsilon) {
    weights[i_star] = 1.0;
    return weights;
  }

  // Compute softmax: w_i = exp(-s_i / η_eff) / Σ
  // Use exp(-(s_i - s_min) / η_eff) for numerical stability
  double sum = 0.0;

  for (size_t i = 0; i < n; ++i) {
    weights[i] = std::exp(-(steps[i] - s_min) / safe_eta);
    sum += weights[i];
  }

  // Normalize
  if (sum > 0.0 && std::isfinite(sum)) {
    for (size_t i = 0; i < n; ++i) {
      weights[i] /= sum;
    }
  } else {
    // Fallback: reset all weights and set one-hot at minimum.
    // This handles edge cases where exp() produced NaN values.
    std::fill(weights.begin(), weights.end(), 0.0);
    weights[i_star] = 1.0;
  }

  return weights;
}

}  // namespace spatialblend
