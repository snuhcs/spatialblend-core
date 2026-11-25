// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// The harness's local-minimum oracle of a multi-relation answer, shared by the scenario runner and the core integration
// suite (moved verbatim from runner.cc).
#ifndef SPATIALBLEND_SIMULATOR_SCENARIO_LOCAL_MINIMUM_H_
#define SPATIALBLEND_SIMULATOR_SCENARIO_LOCAL_MINIMUM_H_

#include <Eigen/Dense>
#include <functional>
#include <string>

#include "spatialblend/space/space.h"

namespace spatialblend {
namespace scn {

// Whether the multi-relation answer is a local minimum of the objective it solved,
// J(x) = sum_i w_i (d_i(x) - d*_i)^2 with d*_i = d_i0 (1 - 1/n_i): the harness evaluates
// J itself through the relation evaluator at the answer and around it, and the answer fails
// if a point of F lower than it by MORE THAN THE RESOLUTION OF J exists. Nothing is read from
// the solver's stationary flag.
//
// The resolution R(x) of J(x), derived (u = 2^-53 the double unit roundoff, gamma_k = k u / (1 - k u),
// Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., Lemma 3.1; eps_f = 2^-23 the
// float epsilon, 1.2e-7):
//  1. A distance is a sum of s non-negative segment lengths. A length sqrt(dx^2 + dy^2) of two
//     differences is evaluated with at most 4 roundings, so it carries a relative error <= gamma_4;
//     the recursive sum of s non-negative terms adds gamma_{s-1} (Higham sec. 4.2, eq. 4.4,
//     sum |x_i| = d because the terms do not change sign). So |d~ - d| <= gamma_{s+3} d.
//     s is NOT derived here: it is the segments of the longest geodesic. kRouteSegments = 128
//     assumes a route wraps at most two obstacles' rings of kArcPoints = 64 vertices
//     (geometry/obstacle_free_space.cc). This term is below 1e-13 d, six orders under (3), so a
//     tenfold different s changes no verdict (checked: SCENARIO quick, s = 1280, same counts).
//  2. e = d~ - d*: the subtraction adds one rounding, u |e| <= u (d + d*). So the bound on e is
//     E_dbl = gamma_{s+3} d + u (d + d*).
//  3. The resolution of the POSE. The geodesic distance is 1-Lipschitz in the pose (|d(x) - d(x')|
//     <= |x - x'|), and the pose is known to the data's precision, a float epsilon of its
//     magnitude: eps_f max(1, |x|_inf), the same quantity the solver stops at
//     (relation_solver.cc below_precision; the data's precision, relation/support.h:24-26: the
//     apps hand the core single-precision matrices). So d is known to E_data = eps_f max(1, |x|_inf),
//     and J cannot be resolved below it: this is the precision the system defines, not a tolerance
//     chosen here. (The factor 1 in max(1, .) is the solver's own, inherited, not derived.)
//  4. E = E_data + E_dbl. A term w e^2 is evaluated with 2 roundings (e*e, w*.): |w e~^2 - w e^2| <=
//     w (2 |e| E + E^2) + gamma_2 w e^2; their sum over n terms adds gamma_{n-1} J (Higham 4.2).
//     R(x) is that total. The test is J(x0) - J(x) > R(x0) + R(x).
//  5. A RISE is tested without E_data (double precision only): the data's precision decides whether
//     a lower point is resolvable, not whether J climbed on the way to it. J rising by more than
//     the double-precision R along the segment to a lower point is a ridge (below); J rising in
//     every direction at 1 micrometre is a kink (the classifier).
// Two things are assumptions, not derivations: s (above, dominated) and that the pose magnitude
// stands for the scale of the data (3). Sensitivity, SCENARIO quick (988 rows): E_data x2 and
// s x10 leave n_min_fail at 0; E_data x0.5 gives 2 frames, x0.25 gives 22.
//
// A lower neighbour is either a DESCENT (J falls monotonically along the segment to it:
// the answer is not a minimum) or lies across a RIDGE (J first rises along the segment: the
// geodesic distance is the minimum over routes, so J has a concave kink where two routes tie,
// and the solver stays in the route basin the sample is in; no basin crossing, as v1).
// The segment is sampled at sixteenths, and nearer the answer at
// 1/64, 1/256 and 1/1024 of it.
enum class MinKind { kMinimum, kRidge, kNotMinimum, kUnchecked };
MinKind LocalMinimumOfJ(Space* tgt, const Space::PolicyObservation& po,
                     const Space::SolverObservation& so, const Eigen::Vector2d& answer,
                     const std::function<bool(const Eigen::Vector2d&)>& in_f,
                     bool* micro_descent = nullptr, double* micro_min_dj = nullptr,
                     std::string* profile = nullptr,
                     const std::function<double(const Eigen::Vector2d&, const Eigen::Vector2d&)>& goal_distance = nullptr);

}  // namespace scn
}  // namespace spatialblend

#endif  // SPATIALBLEND_SIMULATOR_SCENARIO_LOCAL_MINIMUM_H_
