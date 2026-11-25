// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/testing/local_minimum.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "spatialblend/space/testing/geom.h"
#include "spatialblend/solver/relation_evaluator.h"
#include "spatialblend/solver/relation_solver.h"

namespace spatialblend {
namespace scn {

MinKind LocalMinimumOfJ(Space* tgt, const Space::PolicyObservation& po,
                     const Space::SolverObservation& so, const Eigen::Vector2d& answer,
                     const std::function<bool(const Eigen::Vector2d&)>& in_f,
                     bool* micro_descent, double* micro_min_dj,
                     std::string* profile,
                     const std::function<double(const Eigen::Vector2d&, const Eigen::Vector2d&)>& goal_distance) {
  struct Term {
    PreparedTargetRelation prepared;
    double weight, d_star;
    bool border;
    bool engaged = false;  // measures the course length to `goal` (RatioIntent::engaged_goal)
    Vec2 goal = Vec2::Zero();
  };
  const ElementTree& tree = *tgt->GetElementTree();
  std::vector<Term> terms;
  for (const auto& q : so.solve.ratio_intents) {
    if (q.payload_index >= so.goal_payload_index.size()) return MinKind::kUnchecked;
    const size_t pi = so.goal_payload_index[q.payload_index];
    if (pi >= po.payloads.size()) return MinKind::kUnchecked;
    const RelationPayload& p = po.payloads[pi];
    auto prep = PrepareTargetRelation(p, tree);
    if (!prep.ok()) return MinKind::kUnchecked;
    const double d_star = std::max(0.0, q.sample_distance_m * (1.0 - 1.0 / q.update_count));
    terms.push_back({std::move(prep), q.weight, d_star, p.support.container_self, q.engaged_goal, q.goal});
  }
  if (terms.empty()) return MinKind::kUnchecked;
  // Higham's gamma_k (u = 2^-53). kRouteSegments is the assumed upper bound of the segments a
  // geodesic distance is summed from: a route wraps at most the rings of two obstacles, each
  // a kArcPoints = 64-gon (geometry/obstacle_free_space.cc).
  constexpr double kU = 1.1102230246251565e-16;
  constexpr int kRouteSegments = 128;
  const auto gamma = [&](int k) { return k * kU / (1.0 - k * kU); };
  const double gamma_route = gamma(kRouteSegments + 3);  // item 1: gamma_{s+3}
  const double gamma_terms = gamma(static_cast<int>(terms.size()));
  const auto j_at = [&](const Vec2& x, bool* ok, std::vector<double>* dists = nullptr, double* rounding = nullptr,
                        bool with_data = true) {
    double j = 0.0;
    double r = 0.0;
    // E_data of the derivation above, item 3: the resolution of the pose, eps_f max(1, |x|_inf).
    const double data_err = !with_data ? 0.0 : static_cast<double>(std::numeric_limits<float>::epsilon()) *
                            std::max(1.0, std::max(std::fabs(x.x()), std::fabs(x.y())));
    if (dists) dists->clear();
    *ok = true;
    for (const auto& t : terms) {
      TargetAnchorPose a;
      a.position = Eigen::Vector3d(x.x(), x.y(), 0.0);
      a.yaw = 0.0;
      double dist;
      if (t.engaged && goal_distance) {
        // An engaged object measures the course to its mapped contact (relation_solver.cc, goal_distance).
        dist = goal_distance(x, t.goal);
        if (!(dist >= 0.0)) { *ok = false; return 0.0; }
      } else {
        const auto ev = EvaluatePreparedTargetRelation(t.prepared, a, tree, /*fast_distance=*/false,
                                                       /*nearest_support_contact=*/t.border);
        // The distance the solver's pursuit measures (see PursuitDistance in relation_solver.cc).
        dist = t.border ? ev.relation.support_distance_m : ev.relation.contact_distance_m;
        if (!ev.ok() || !(dist >= 0.0)) { *ok = false; return 0.0; }
      }
      const double e = dist - t.d_star;
      if (dists) dists->push_back(dist);
      j += t.weight * e * e;
      const double big_e = data_err + gamma_route * dist + kU * (dist + t.d_star);  // E of item 4
      r += t.weight * (2.0 * std::fabs(e) * big_e + big_e * big_e + gamma(2) * e * e);
    }
    if (rounding) *rounding = r + gamma_terms * j;
    return j;
  };
  bool ok = true;
  double r0 = 0.0, r0d = 0.0;  // R with the data's precision (a lower point must beat it), and without (a rise is a rise)
  const double j0 = j_at(answer, &ok, nullptr, &r0);
  if (ok) j_at(answer, &ok, nullptr, &r0d, /*with_data=*/false);
  if (!ok) return MinKind::kUnchecked;
  // The data's precision at the answer (relation/support.h:24-26): the solver stops when its step
  // is below it, so a point within it of the minimum is converged. Whether the answer is within
  // it is read here from J itself: the undamped Newton step h = -H^-1 g of J at the answer, from
  // CENTRAL differences of J at this harness's own step (below), must not exceed it (the criterion of Madsen, Nielsen & Tingleff 2004,
  // sec. 3.2, eq. 3.14b, evaluated independently of the solver's analytic derivatives). A
  // 1 micrometre neighbour test is not a test of that: a point 1.3 micrometres from the minimum has
  // lower neighbours at 1 micrometre. Where the stencil leaves F (the answer is on the boundary)
  // the neighbours are tested at twice the precision: no neighbour at 2 delta from a point delta
  // from an isotropic minimum is lower.
  const double data_precision_m =
      static_cast<double>(std::numeric_limits<float>::epsilon()) * std::max(1.0, std::max(std::fabs(answer.x()), std::fabs(answer.y())));
  constexpr double kDataPrecisionM = 1e-6;  // the radius the kink classifier samples at
  {
    // The step of the central differences of J: this harness's own, independent of the solver (which
    // differentiates analytically). A central difference of J has a rounding error ~ eps_A / h and a
    // truncation error ~ h^2 |J'''| / 6, least at h ~ eps_A^(1/3) in the units of the problem (Gill, Murray &
    // Wright, Practical Optimization, 1981, sec. 8.6; Gill, Murray, Saunders & Wright, SIAM J. Sci. Stat.
    // Comput. 4, 1983), with eps_A the relative error of a distance gamma_{s+3} (item 1 above) and the pose
    // magnitude the scale.
    const double kS = std::cbrt(gamma(kRouteSegments + 3)) * std::max(1.0, std::max(std::fabs(answer.x()), std::fabs(answer.y())));
    bool stencil_free = true, all_ok = true;
    double jv[3][3];
    for (int ix = -1; ix <= 1 && stencil_free && all_ok; ++ix)
      for (int iy = -1; iy <= 1; ++iy) {
        const Vec2 q = answer + kS * Vec2(ix, iy);
        if (!in_f(q)) { stencil_free = false; break; }
        bool qok = true;
        jv[ix + 1][iy + 1] = (ix == 0 && iy == 0) ? j0 : j_at(q, &qok);
        if (!qok) { all_ok = false; break; }
      }
    if (stencil_free && all_ok) {
      const double g0 = (jv[2][1] - jv[0][1]) / (2 * kS), g1 = (jv[1][2] - jv[1][0]) / (2 * kS);
      const double h00 = (jv[2][1] - 2 * j0 + jv[0][1]) / (kS * kS);
      const double h11 = (jv[1][2] - 2 * j0 + jv[1][0]) / (kS * kS);
      const double h01 = (jv[2][2] - jv[2][0] - jv[0][2] + jv[0][0]) / (4 * kS * kS);
      const double det = h00 * h11 - h01 * h01;
      if (det > 0.0 && h00 > 0.0) {
        const double hx = -(h11 * g0 - h01 * g1) / det, hy = -(-h01 * g0 + h00 * g1) / det;
        const double hn = std::sqrt(hx * hx + hy * hy);
        // The criterion of MNT eq. 3.14b in its own form, ||h|| <= eps (||x|| + eps) with the 2-norm of
        // the pose and eps the float epsilon, and the Newton step is a claim the harness CHECKS:
        // J at x + h must be lower than J at the answer by more than the rounding bound (a
        // nearly singular H, a valley, predicts steps J does not reward). Beyond the stencil the
        // quadratic model is not read: a Newton step longer than the stencil is left to the
        // neighbour tests below.
        const double eps = static_cast<double>(std::numeric_limits<float>::epsilon());
        if (hn > eps * (answer.norm() + eps) && hn < kS) {
          bool ho = true;
          double rh = 0.0;
          const double jh = j_at(answer + Vec2(hx, hy), &ho, nullptr, &rh);
          // A descent is not a ridge: if J rises above J(answer) anywhere between the answer and x + h
          // (a route tie: the geodesic distance is the minimum over routes, J has a concave kink
          // there and the solver stays in the route basin it is in, see kRidge below), the lower
          // point lies across it.
          bool ridge = false;
          if (ho) {
            // A rise is a rise at double precision: the data's precision only decides whether a LOWER
            // point is resolvable, not whether J climbed on the way to it.
            std::vector<double> fr = {1.0 / 1024, 1.0 / 256, 1.0 / 64};
            for (int s = 1; s < 16; ++s) fr.push_back(s / 16.0);
            for (double f : fr) {
              bool so_ = true;
              double rsd = 0.0;
              const double js_ = j_at(answer + Vec2(hx, hy) * f, &so_, nullptr, &rsd, /*with_data=*/false);
              if (so_ && js_ - j0 > r0d + rsd) { ridge = true; break; }
            }
          }
          if (ridge) return MinKind::kRidge;
          if (ho && in_f(answer + Vec2(hx, hy)) && j0 - jh > r0 + rh) {
            if (micro_descent) *micro_descent = true;
            if (micro_min_dj) *micro_min_dj = jh - j0;
            return MinKind::kNotMinimum;
          }
        }
      }
    } else if (all_ok) {
      for (int k = 0; k < 8; ++k) {
        const Vec2 nb = answer + 2.0 * data_precision_m * Vec2(std::cos(k * M_PI / 4), std::sin(k * M_PI / 4));
        if (!in_f(nb)) continue;
        bool nok = true;
        double rn = 0.0;
        const double jn = j_at(nb, &nok, nullptr, &rn);
        if (nok && j0 - jn > r0 + rn) {
          if (micro_descent) *micro_descent = true;
          if (micro_min_dj) *micro_min_dj = jn - j0;
          return MinKind::kNotMinimum;
        }
      }
    }
  }
  constexpr double kStepM = 1e-3;
  for (int k = 0; k < 8; ++k) {
    const Vec2 nb = answer + kStepM * Vec2(std::cos(k * M_PI / 4), std::sin(k * M_PI / 4));
    if (!in_f(nb)) continue;
    bool nok = true;
    double rn = 0.0;
    const double jn = j_at(nb, &nok, nullptr, &rn);
    if (!nok) continue;
    if (j0 - jn > r0 + rn) {
      // The ridge can start closer to the answer than a sixteenth of the neighbour's distance
      // (the solver stops within its parameter tolerance of the tie), so the segment is also
      // sampled at 1/1024, 1/256 and 1/64 of it.
      std::vector<double> fractions = {1.0 / 1024, 1.0 / 256, 1.0 / 64};
      for (int s = 1; s < 16; ++s) fractions.push_back(static_cast<double>(s) / 16.0);
      for (double f : fractions) {
        bool sok = true;
        const Vec2 mid = answer + (nb - answer) * f;
        if (!in_f(mid)) continue;
        double rs = 0.0;
        const double js = j_at(mid, &sok, nullptr, &rs, /*with_data=*/false);
        if (sok && js - j0 > r0d + rs) return MinKind::kRidge;
      }
      // Is the failure a kink? A neighbour 1 mm away is lower, but is J lowered at the data's
      // precision (1 micrometre, above) in ANY direction? Directions: 32 (11.25 degrees apart): for
      // a differentiable J with gradient g some direction d has d.g <= -|g| cos(5.625 degrees) =
      // -0.995 |g|, so a smooth descent loses at most 0.5% of its rate to the sampling, far
      // below the rounding bound R that every test here is made against. If none lowers J, J
      // is not differentiable here (a route tie: the geodesic distance is the minimum over
      // routes; all one-sided directional derivatives >= 0, Clarke stationary) with a cliff between
      // the point and the lower neighbour. If one does, a smooth descent was left.
      if (micro_descent) {
        constexpr double kMicroM = kDataPrecisionM;
        double best = 1e300, best_margin = 1e300;
        for (int k = 0; k < 32; ++k) {
          const Vec2 q = answer + kMicroM * Vec2(std::cos(k * M_PI / 16), std::sin(k * M_PI / 16));
          if (!in_f(q)) continue;
          bool qok = true;
          double rq = 0.0;
          const double jq = j_at(q, &qok, nullptr, &rq, /*with_data=*/false);
          if (qok) { best = std::min(best, jq - j0); best_margin = std::min(best_margin, jq - j0 + r0d + rq); }
        }
        if (micro_min_dj) *micro_min_dj = best;
        *micro_descent = best_margin < 0.0;
        if (profile) {
          // dJ along the direction whose 1 mm neighbour is lowest: the cliff, if there is one.
          int bk = 0;
          double bj = 1e300;
          for (int k = 0; k < 32; ++k) {
            const Vec2 q = answer + 1e-3 * Vec2(std::cos(k * M_PI / 16), std::sin(k * M_PI / 16));
            if (!in_f(q)) continue;
            bool qok = true;
            const double jq = j_at(q, &qok);
            if (qok && jq < bj) { bj = jq; bk = k; }
          }
          char b[256];
          std::string s = "profile dJ(delta) along the lowest 1 mm direction:";
          for (double dd : {1e-6, 1e-5, 1e-4, 3e-4, 1e-3}) {
            const Vec2 q = answer + dd * Vec2(std::cos(bk * M_PI / 16), std::sin(bk * M_PI / 16));
            bool qok = true;
            const double jq = j_at(q, &qok);
            std::snprintf(b, sizeof b, " %.0e:%.3e", dd, qok ? jq - j0 : 0.0);
            s += b;
          }
          // The relation distances at the answer and at the points either side of the cliff:
          // the term whose distance jumps is the one whose geodesic is the minimum over routes.
          for (double dd : {0.0, 3e-4, 1e-3}) {
            const Vec2 q = answer + dd * Vec2(std::cos(bk * M_PI / 16), std::sin(bk * M_PI / 16));
            bool qok = true;
            std::vector<double> ds;
            j_at(q, &qok, &ds);
            std::snprintf(b, sizeof b, " | d_i at %.0e:", dd);
            s += b;
            for (double dv : ds) { std::snprintf(b, sizeof b, " %.6f", dv); s += b; }
          }
          *profile = s;
        }
      }
      return MinKind::kNotMinimum;
    }
  }
  return MinKind::kMinimum;
}

}  // namespace scn
}  // namespace spatialblend
