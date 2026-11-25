// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SOLVER_SOFT_ANNEALING_WEIGHTS_H_
#define SPATIALBLEND_SOLVER_SOFT_ANNEALING_WEIGHTS_H_

#include <vector>

namespace spatialblend {

// The progressive weighted annealing of the v1 soft-annealing planner, kept verbatim for the
// realizer (realization_policy.h) and the scenario tools.
//
// Compute softmax weights with optional temperature decay (Case A).
//
// This implements soft dynamic annealing where the distribution is controlled
// by a temperature parameter η that optionally decays as contact approaches.
//
// Algorithm:
// 1. If enable_decay: η_eff = η_0 * (s_min - 1 + ε)^γ
//    Else: η_eff = η_0
// 2. w_i = exp(-s_i / η_eff) / Σ exp(-s_j / η_eff)
//
// With decay enabled, as s_min → 1:
//   η_eff → η_0 * ε^γ ≈ 0 → softmax → one-hot
//
// Args:
//   steps: Vector of remaining steps per target (TTA)
//   eta: Base temperature parameter (η_0)
//   enable_decay: If true, temperature decays as s_min → 1
//   decay_gamma: Exponent for decay (γ), higher = faster sharpening near contact
//   epsilon: Numerical stability constant
//
// Returns: Normalized weights that sum to 1.0
std::vector<double> ComputeSoftAnnealingWeights(
    const std::vector<double>& steps,
    double eta = 1.0,
    bool enable_decay = true,
    double decay_gamma = 1.0,
    double epsilon = 1e-3);

}  // namespace spatialblend

#endif  // SPATIALBLEND_SOLVER_SOFT_ANNEALING_WEIGHTS_H_
