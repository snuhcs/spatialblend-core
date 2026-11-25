// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/nd_trust_region_loop.h
//
// The OUTER loop of the relation solve (one solver whose state dimension follows the scope). Every
// anchor's pursuit runs it through scope_pursuit.h (PursueInScope), whose hooks read the scope and the terms:
//
//   ProjectIterate()       an iterate a rounding error outside F is projected onto it
//   NoteIncumbent()        the budget's incumbent: the lowest-J iterate in F
//   OverBudget()           the evaluation budget (sticky)
//   Assemble()             e, the derivatives, the model (half gradient g, half Hessian A, Gauss-Newton part N), the
//                          accuracy R of J (j_res), the rounding of g (g_tol) and tau_F = R / (1 + |J|)
//   Chi()                  U3's measure: the rate of the steepest first-order descent of J that stays in F
//   ChiDescent()           a step along chi's direction when U1 + U2 hold and U3 does not (the length taken, 0 = none; -1 = J refuted the
//                          direction before it could gain J's accuracy R, a minimiser to that accuracy)
//   Subproblem(model, D)   the trust-region subproblem (nd_trust_region.h)
//   MakeTrial(step)        the trial point of a step: in F (the root's active-set faces and route ties; a box's projection)
//   EvalJ(y)               J and the number of terms evaluated at y
//   Retract(trial, jt)     the root's chord retraction of an inadequate step (a no-op for a convex F)
//
// The loop itself is the root's, moved here unchanged (relation_solver_parity_test: bit-identical root answers):
// Algorithm 4.1 (Nocedal & Wright 2006) with acceptance eta = 0 and its radius update, the combined stop U1 + U2 + U3
// (Gill, Murray & Wright 1981, sec. 8.2.3.2) with tau_F from the accuracy of J, the collapse of the radius to the
// representable resolution of x (NOT converged unless U3 holds there), the evaluation budget, and the two
// counts of unresolved steps (kTrMaxUnresolvedNormal / kTrMaxUnresolvedChi).
#ifndef SPATIALBLEND_SPACE_ND_TRUST_REGION_LOOP_H_
#define SPATIALBLEND_SPACE_ND_TRUST_REGION_LOOP_H_

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "spatialblend/solver/nd_trust_region.h"

namespace spatialblend {

// kPrecisionChiRefuted: the radius collapsed with chi above U3's threshold, and the descent along chi's direction was refuted at
// J's accuracy (TrChiRefutedAtResolution): a minimiser to that accuracy, told apart from a U3 precision stop.
enum class TrLoopStop { kNone, kNoDampedStep, kNoDampedStepOnBoundary, kPrecision, kPrecisionChiRefuted };

// The refutation of a descent along chi's direction at J's accuracy R. chi is the slope of J's linearisation (U3's measure), so
// along its direction J can fall by at most about chi * s over a length s. If J, measured at x + s v + pull, rose by more than R
// above J at x + pull (the same pull, without the step: j_base) at a length s_rise with chi * s_rise <= R, the descent chi
// measures ends at a kink of J before it can gain anything J resolves: no resolved decrease exists along it. The rise is taken
// from j_base, never from J(x): a pull that alone raises J (on a face of F, where J presses into the wall) is not a rise along v.
inline bool TrChiRefutedAtResolution(double chi, double s_rise, double j_rise, double j_base, double j_res) {
  return chi >= 0.0 && s_rise > 0.0 && j_res > 0.0 && j_rise - j_base > j_res && chi * s_rise <= j_res;
}

// Policy counts (see relation_solver.cc, "Chains of steps J cannot resolve").
constexpr int kTrMaxUnresolvedNormal = 1;
constexpr int kTrMaxUnresolvedChi = 3;
constexpr double kTrEta = 0.0;

template <int Dim>
struct TrIterModel {
  Eigen::Matrix<double, Dim, 1> g;
  Eigen::Matrix<double, Dim, Dim> A, N;
  int nterms = 0;
  double j_res = 0.0, g_tol = 0.0, tau_f = 0.0;
};

template <int Dim>
struct TrTrial {
  bool abort = false;  // no trial can be formed (the iterate is not in F): the trial loop ends without a step
  Eigen::Matrix<double, Dim, 1> trial;
  bool projected = false;  // an active-set (face) step: a step of another model than the unconstrained one
  bool have_pred = false;  // `pred` is the predicted reduction of the model that proposed the trial
  double pred = 0.0;
};

// The caller's solve state, by reference (the loop reads and writes it in place).
template <int Dim>
struct TrLoopVars {
  Eigen::Matrix<double, Dim, 1>& x;
  double& j;
  double& delta;
  double delta_max;
  double& last_dj;
  double& last_dx;
  bool& last_constrained;
  int& chi_streak;
  int& normal_streak;
  int& iterations;
  bool& stationary;
  bool& stopped;
  TrLoopStop& stop;
  bool& not_converged;
  int& budget_hit_iteration;
};

// The model's predicted reduction J(x) - m(x + p) = -2 g.p - p.A.p (g and A are the halves). Dim = 2 in the scalar form
// the root has always evaluated it in.
template <int Dim>
inline double TrModelReduction(const Eigen::Matrix<double, Dim, 1>& g, const Eigen::Matrix<double, Dim, Dim>& A,
                               const Eigen::Matrix<double, Dim, 1>& p) {
  return -2.0 * g.dot(p) - p.dot(A * p);
}
template <>
inline double TrModelReduction<2>(const Eigen::Vector2d& g, const Eigen::Matrix2d& A, const Eigen::Vector2d& p) {
  return -2.0 * (g(0) * p.x() + g(1) * p.y()) -
         (A(0, 0) * p.x() * p.x() + 2.0 * A(0, 1) * p.x() * p.y() + A(1, 1) * p.y() * p.y());
}

// Hooks from lambdas (in the order of the list in the header comment).
template <class P, class N, class O, class A, class C, class D, class S, class T, class E, class R>
struct TrLambdaHooks {
  P project;
  N note;
  O over;
  A assemble;
  C chi;
  D chi_descent;
  S subproblem;
  T make_trial;
  E eval_j;
  R retract;
  void ProjectIterate() { project(); }
  void NoteIncumbent() { note(); }
  bool OverBudget() { return over(); }
  auto Assemble() { return assemble(); }
  double Chi() { return chi(); }
  double ChiDescent() { return chi_descent(); }
  template <class M>
  auto Subproblem(const M& m, double delta) { return subproblem(m, delta); }
  template <class St>
  auto MakeTrial(const St& st) { return make_trial(st); }
  template <class V>
  std::pair<double, int> EvalJ(const V& y) { return eval_j(y); }
  template <class V>
  bool Retract(V* trial, double* jt) { return retract(trial, jt); }
};
template <class P, class N, class O, class A, class C, class D, class S, class T, class E, class R>
TrLambdaHooks<P, N, O, A, C, D, S, T, E, R> MakeTrLoopHooks(P p, N n, O o, A a, C c, D d, S s, T t, E e, R r) {
  return {p, n, o, a, c, d, s, t, e, r};
}

template <int Dim, class Hooks>
void RunTrustRegionLoop(TrLoopVars<Dim>& s, Hooks& h, int safety_cap) {
  using Vec = Eigen::Matrix<double, Dim, 1>;
  constexpr double kEps = 2.220446049250313e-16;  // double epsilon
  Vec& x = s.x;
  double& j = s.j;
  for (; s.iterations < safety_cap; ++s.iterations) {
    h.ProjectIterate();
    // The budget, at the outer iteration boundary: the iterate here has been judged by J and is the incumbent if it is the
    // lowest feasible one. A check that tripped inside the previous iteration (an inner loop cut short) lands here too.
    h.NoteIncumbent();
    if (h.OverBudget()) {
      s.budget_hit_iteration = s.iterations;
      break;
    }
    const TrIterModel<Dim> m = h.Assemble();
    const double xres = kEps * std::max(1.0, x.cwiseAbs().maxCoeff());  // the representable resolution of x
    bool accepted = false;
    bool radius_collapsed = false;
    const double tau_f = m.tau_f;
    const double j_res = m.j_res;
    // The budget, once the iteration's derivatives are assembled (the step search below is the expensive part): the iterate
    // is the incumbent; the next boundary check ends the solve.
    if (h.OverBudget()) continue;
    if (s.iterations > 0 && s.last_dj < tau_f * (1.0 + std::fabs(j)) && s.last_dx < std::sqrt(tau_f) * (1.0 + x.norm())) {
      if (h.Chi() <= std::cbrt(tau_f) * (1.0 + std::fabs(j))) {
        s.stop = TrLoopStop::kPrecision;
        s.stationary = true;
        s.stopped = true;
        break;
      }
      // U1 and U2 hold (the last step gained and moved nothing J and x resolve) but U3 does not, so a descent that J resolves
      // exists and the model steps cannot find it: step along chi's direction.
      // A step taken (> 0) continues; none (0) and a refutation at J's accuracy (-1) both go on to the model steps below,
      // whose collapse judges U3 again (a refutation is acted on only there).
      if (const double taken_len = h.ChiDescent(); taken_len > 0.0) continue;
    }
    bool budget_cut = false;  // the trial loop was cut by the budget: no stop rule below is judged on it
    for (;;) {
      if (h.OverBudget()) {
        budget_cut = true;
        break;
      }
      // U3 at machine resolution: the radius has collapsed below the representable resolution of x, so no step in the
      // model's reach moves x.
      if (s.delta <= xres) {
        s.stop = s.last_constrained ? TrLoopStop::kNoDampedStepOnBoundary : TrLoopStop::kNoDampedStep;
        radius_collapsed = true;
        break;
      }
      // The exact solution of the trust-region subproblem (rank aware).
      const TrStep<Dim> tro = h.Subproblem(m, s.delta);
      const bool radius_active = tro.radius_active;
      // A model step below the representable resolution of x moves nothing (U2 at machine resolution); with U3 met (and
      // so U1: the step changes nothing) the solve has converged.
      if (tro.p.norm() <= xres && h.Chi() <= std::cbrt(tau_f) * (1.0 + std::fabs(j))) {
        s.stop = TrLoopStop::kPrecision;
        s.stationary = true;
        s.stopped = true;
        radius_collapsed = false;
        break;
      }
      const TrTrial<Dim> t = h.MakeTrial(tro);
      if (t.abort) break;
      Vec trial = t.trial;
      const bool projected = t.projected;
      Vec taken = trial - x;
      double jt = j;
      int nt = 0;
      if (taken.norm() > 0.0) std::tie(jt, nt) = h.EvalJ(trial);
      const double predicted_plain =
          taken.norm() > 0.0 ? (t.have_pred ? t.pred : TrModelReduction<Dim>(m.g, m.A, taken)) : 0.0;
      if (!projected && taken.norm() > 0.0 && nt > 0 &&
          !(predicted_plain > 0.0 && jt < j && (j - jt) / predicted_plain >= 0.75)) {
        if (h.Retract(&trial, &jt)) taken = trial - x;
      }
      double rho = -1.0;
      if (taken.norm() > 0.0 && nt > 0 && jt < j) {
        const double predicted = predicted_plain;
        // The ratio of actual to predicted reduction (N&W 2006, Algorithm 4.1). Where the model predicts no gain and J fell,
        // the ratio is +infinity. A PROJECTED step is a step of a different model (the face's): accepted because J fell
        // (rho > eta = 0), with the radius left as it is (rho = 1/2 stands for any rho in [1/4, 3/4]).
        rho = predicted > 0.0 ? (j - jt) / predicted : (projected ? 0.5 : std::numeric_limits<double>::infinity());
        // A ratio of two reductions that are both within J's accuracy R measures rounding, not the model: it may not cut
        // the radius.
        if (predicted > 0.0 && predicted <= j_res) rho = std::max(rho, 0.5);
        // A projected step whose decrease is within R and that moves more than U2's displacement is no success.
        if (projected && j - jt <= j_res &&
            (taken.norm() >= std::sqrt(tau_f) * (1.0 + x.norm()) || s.normal_streak >= kTrMaxUnresolvedNormal))
          rho = -1.0;
      } else if (!projected && s.normal_streak < kTrMaxUnresolvedNormal && taken.norm() > 0.0 && nt > 0 &&
                 predicted_plain > 0.0 && predicted_plain <= j_res && jt - j <= j_res) {
        // A step whose model reduction is itself within R cannot be refuted by J: it is taken on the model's prediction.
        rho = 1.0;
      }
      // Algorithm 4.1, the radius update. A step of the region's length: x + p - x loses the bits below the resolution of
      // x, so the comparison is made to that resolution.
      const double len_tol = 4.0 * kEps * std::max(1.0, x.cwiseAbs().maxCoeff());
      const bool at_radius = radius_active || taken.norm() >= s.delta - len_tol;
      s.delta = TrRadiusUpdate(rho, at_radius, s.delta, s.delta_max);
      if (rho > kTrEta) {
        s.chi_streak = 0;
        s.normal_streak = (j - jt > j_res) ? 0 : s.normal_streak + 1;
        s.last_dj = j - jt;
        s.last_dx = taken.norm();
        x = trial;
        j = jt;
        accepted = true;
        break;
      }
    }
    if (!accepted && budget_cut) continue;  // the boundary check at the top of the loop ends the solve
    if (!accepted) {
      if (s.stop == TrLoopStop::kNone)
        s.stop = s.last_constrained ? TrLoopStop::kNoDampedStepOnBoundary : TrLoopStop::kNoDampedStep;
      // A radius that collapsed to the resolution of x is a solve that failed to find a descent, not a minimiser: it
      // counts as NOT converged (U3 holding at the collapse is a converged solve).
      if (radius_collapsed && h.Chi() > std::cbrt(tau_f) * (1.0 + std::fabs(j))) {
        const double taken_len = h.ChiDescent();
        if (taken_len > 0.0) {
          s.stop = TrLoopStop::kNone;
          s.delta = std::min(s.delta_max, std::max(s.delta, 2.0 * taken_len));
          s.last_constrained = false;
          continue;
        }
        if (taken_len < 0.0) {
          // J refuted chi's direction before it could gain R (ChiDescent's -1): U3 holds to J's accuracy.
          s.stop = TrLoopStop::kPrecisionChiRefuted;
          radius_collapsed = false;
        } else {
          s.not_converged = true;
        }
      } else {
        radius_collapsed = false;
      }
      s.stationary = !radius_collapsed;
      s.stopped = !radius_collapsed;
      break;
    }
  }
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_ND_TRUST_REGION_LOOP_H_
