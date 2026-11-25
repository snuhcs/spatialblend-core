// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/scope_pursuit.h
//
// THE PURSUIT OF AN ANCHOR IN ITS SCOPE (one live path for every anchor; the anchor is a parameter
// of the relation definition). The relations of one anchor, each a term w (d(x) - d*)^2 with d the relation's own
// distance (the geodesic over the scope to its mapped contact), are pursued from the anchor's current position in the
// scope's free space F: one relation walks min(L/n, L) of its course (v1 BaseSingleTargetPlanner), several are
// minimised by the trust-region Newton method of nd_trust_region_loop.h with the active-piece hooks below (route ties,
// the exact tangent cone, the active-set face steps, the chord retraction, the descent along chi).
//
// The algorithm does not depend on the anchor, nor on the kind of space: the scope answers every geometric question
// (relation_scope.h). It depends on the dimension of the scope only where the plane has a closed form or a structure of
// its own: the 2 x 2 eigendecomposition (SymEig2), the cone as arcs of directions, the curved facets of clearance rings,
// the one tangent of a tie curve. Those are the planar instances of the general computations and keep their
// arithmetic (relation_solver_parity_test is bit-identical); the general instances serve every other dimension.
#ifndef SPATIALBLEND_SPACE_SCOPE_PURSUIT_H_
#define SPATIALBLEND_SPACE_SCOPE_PURSUIT_H_

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "spatialblend/geometry/path/bezier_course.h"
#include "spatialblend/solver/nd_trust_region.h"
#include "spatialblend/solver/nd_trust_region_loop.h"
#include "spatialblend/solver/relation_scope.h"
#include "spatialblend/solver/relation_solver.h"

namespace spatialblend {

// Chords per Bezier segment when the smoothed course is checked against the
// free space (the segment test is exact per chord).
constexpr int kCurveChecksPerSegment = 32;

namespace scope_pursuit_internal {

// least value is at an end or at -G / |G|.
template <int Dim>
struct ChiTermT {
  double c = 0.0;
  std::vector<Eigen::Matrix<double, Dim, 1>> g;
};
inline double ChiOverArcs(const std::vector<ChiTermT<2>>& terms, double constant,
                   const std::vector<std::pair<double, double>>& arcs, Eigen::Vector2d* best_dir = nullptr) {
  const auto value_at = [&](double th, Eigen::Vector2d* sel_sum) {
    const Eigen::Vector2d v(std::cos(th), std::sin(th));
    double tot = constant;
    Eigen::Vector2d G = Eigen::Vector2d::Zero();
    for (const auto& t : terms) {
      if (t.g.empty()) continue;
      size_t bi = 0;
      double bv = t.g[0].dot(v);
      for (size_t a = 1; a < t.g.size(); ++a) {
        const double dv = t.g[a].dot(v);
        if (dv < bv) { bv = dv; bi = a; }
      }
      tot += t.c * bv;
      G += t.c * t.g[bi];
    }
    if (sel_sum) *sel_sum = G;
    return tot;
  };
  double best = std::numeric_limits<double>::infinity();
  double best_th = 0.0;
  std::vector<double> cross;
  for (const auto& t : terms)
    for (size_t a = 0; a < t.g.size(); ++a)
      for (size_t b = a + 1; b < t.g.size(); ++b) {
        const Eigen::Vector2d d = t.g[a] - t.g[b];
        if (d.norm() == 0.0) continue;
        const double th = std::atan2(d.x(), -d.y());  // perpendicular to d
        for (const double c : {th, th + M_PI}) {
          double w = std::fmod(c, 2.0 * M_PI);
          if (w < 0.0) w += 2.0 * M_PI;
          cross.push_back(w);
        }
      }
  std::sort(cross.begin(), cross.end());
  for (const auto& arc : arcs) {
    std::vector<double> cuts = {arc.first, arc.second};
    for (const double c : cross) {
      if (c > arc.first && c < arc.second) cuts.push_back(c);
      if (c + 2.0 * M_PI > arc.first && c + 2.0 * M_PI < arc.second) cuts.push_back(c + 2.0 * M_PI);
    }
    std::sort(cuts.begin(), cuts.end());
    if (!(arc.second > arc.first)) {
      const double val = value_at(arc.first, nullptr);
      if (val < best) { best = val; best_th = arc.first; }
      continue;
    }
    for (size_t k = 0; k + 1 < cuts.size(); ++k) {
      const double lo = cuts[k], hi = cuts[k + 1];
      Eigen::Vector2d G;
      value_at(0.5 * (lo + hi), &G);
      for (const double th : {lo, hi}) {
        const double val = value_at(th, nullptr);
        if (val < best) { best = val; best_th = th; }
      }
      if (hi > lo && G.norm() > 0.0) {
        const double th = std::atan2(-G.y(), -G.x());
        for (const double cand : {th, th + 2.0 * M_PI, th - 2.0 * M_PI})
          if (cand > lo && cand < hi) {
            const double val = value_at(cand, nullptr);
            if (val < best) { best = val; best_th = cand; }
          }
      }
    }
  }
  if (best_dir) *best_dir = Eigen::Vector2d(std::cos(best_th), std::sin(best_th));
  return std::max(0.0, -best);
}

// The first t in (0, 1) at which route b becomes shorter than route a along the segment x + t p (2 when it never does
// there). Each route is a fixed rest of length plus the straight first leg from the moving point to its first waypoint,
// f(t) = |x + t p - first| + rest, so f_a = f_b is a quadratic in t after the root is isolated and squared.
template <int Dim>
double FirstCrossing(const Eigen::Matrix<double, Dim, 1>& x, const Eigen::Matrix<double, Dim, 1>& p,
                     const typename ScopeGeometryTypes<Dim>::Route& a, const typename ScopeGeometryTypes<Dim>::Route& b) {
  const Eigen::Matrix<double, Dim, 1> u = x - a.first, w = x - b.first;
  const double rest_a = a.value - u.norm(), rest_b = b.value - w.norm();
  const auto h = [&](double t) { return ((x + t * p - a.first).norm() + rest_a) - ((x + t * p - b.first).norm() + rest_b); };
  const double delta = rest_b - rest_a;  // |u + t p| = |w + t p| + delta
  const double A0 = u.squaredNorm() - w.squaredNorm() - delta * delta;
  const double B0 = 2.0 * p.dot(u - w);
  std::vector<double> roots;
  if (delta == 0.0) {
    const double l0 = u.squaredNorm() - w.squaredNorm();
    if (B0 != 0.0) roots.push_back(-l0 / B0);
  } else {
    const double a2 = 4.0 * delta * delta * p.squaredNorm() - B0 * B0;
    const double a1 = 8.0 * delta * delta * w.dot(p) - 2.0 * A0 * B0;
    const double a0 = 4.0 * delta * delta * w.squaredNorm() - A0 * A0;
    if (a2 == 0.0) {
      if (a1 != 0.0) roots.push_back(-a0 / a1);
    } else {
      const double disc = a1 * a1 - 4.0 * a2 * a0;
      if (disc >= 0.0) {
        const double sq = std::sqrt(disc);
        roots.push_back((-a1 + sq) / (2.0 * a2));
        roots.push_back((-a1 - sq) / (2.0 * a2));
      }
    }
  }
  std::vector<double> bounds = {0.0};
  std::sort(roots.begin(), roots.end());
  for (const double r : roots)
    if (r > 0.0 && r < 1.0) bounds.push_back(r);
  bounds.push_back(1.0);
  for (size_t k = 0; k + 1 < bounds.size(); ++k)
    if (h(0.5 * (bounds[k] + bounds[k + 1])) > 0.0) return bounds[k] > 0.0 ? bounds[k] : 2.0;
  return 2.0;
}

// chi over the tangent cone of F. The plane: the cone is its arcs (the full circle when F does not bound the point), and
// ChiOverArcs is exact. More dimensions: the cone is {v : n . v <= 0} over its facets, and for each choice of one route
// per term J'(x; v) = constant + G . v is linear, so its least value over the unit directions of the cone is attained
// at the normalized projection of -G onto a face of the cone (a subset of facets held as equalities), or on an extreme
// ray; every such point is enumerated. The least over the choices is exact where every term with more than one route
// has c >= 0 (the minimum over its routes); a term with c < 0 and several routes would make J' a maximum over them, and
// the least over choices is then a lower bound, so chi is never under-reported (no scope of more than two dimensions
// lists alternative routes today: relation_scope.h).
template <int Dim>
double ChiOverCone(const std::vector<ChiTermT<Dim>>& terms, double constant,
                   const typename ScopeGeometryTypes<Dim>::Cone& cone, Eigen::Matrix<double, Dim, 1>* best_dir) {
  using Vec = Eigen::Matrix<double, Dim, 1>;
  const auto& fs = cone.facets;
  const int nf = static_cast<int>(fs.size());
  const double tol = 4.0 * 2.220446049250313e-16;
  const auto in_cone = [&](const Vec& v) {
    for (const auto& f : fs)
      if (f.normal.dot(v) > tol) return false;
    return true;
  };
  double best = std::numeric_limits<double>::infinity();
  Vec best_v = Vec::Zero();
  const auto least_of = [&](const Vec& G) {
    for (int mask = 0; mask < (1 << nf); ++mask) {
      Eigen::MatrixXd Nm(0, Dim);
      for (int k = 0; k < nf; ++k)
        if (mask & (1 << k)) {
          Nm.conservativeResize(Nm.rows() + 1, Eigen::NoChange);
          Nm.row(Nm.rows() - 1) = fs[k].normal.transpose();
        }
      Eigen::MatrixXd basis;
      if (Nm.rows() == 0) {
        basis = Eigen::MatrixXd::Identity(Dim, Dim);
      } else {
        const Eigen::FullPivLU<Eigen::MatrixXd> lu(Nm);
        if (lu.rank() == Dim) continue;
        basis = Eigen::HouseholderQR<Eigen::MatrixXd>(lu.kernel()).householderQ() *
                Eigen::MatrixXd::Identity(Dim, Dim - lu.rank());
      }
      const Vec d = Vec(-(basis * (basis.transpose() * Eigen::VectorXd(G))));
      std::vector<Vec> cands;
      if (d.norm() > 0.0) cands.push_back(Vec(d / d.norm()));
      for (int c = 0; c < basis.cols(); ++c) {
        const Vec b = Vec(basis.col(c));
        cands.push_back(b);
        cands.push_back(Vec(-b));
      }
      for (const Vec& v : cands) {
        if (!in_cone(v)) continue;
        const double val = constant + G.dot(v);
        if (val < best) {
          best = val;
          best_v = v;
        }
      }
    }
  };
  std::vector<size_t> choice(terms.size(), 0);
  for (;;) {
    Vec G = Vec::Zero();
    for (size_t t = 0; t < terms.size(); ++t)
      if (!terms[t].g.empty()) G += terms[t].c * terms[t].g[choice[t]];
    least_of(G);
    size_t k = 0;
    for (; k < terms.size(); ++k) {
      if (++choice[k] < terms[k].g.size()) break;
      choice[k] = 0;
    }
    if (k == terms.size()) break;
  }
  if (best_dir) *best_dir = best_v;
  return std::isfinite(best) ? std::max(0.0, -best) : 0.0;
}
template <>
inline double ChiOverCone<2>(const std::vector<ChiTermT<2>>& terms, double constant,
                             const ScopeGeometryTypes<2>::Cone& cone, Eigen::Vector2d* best_dir) {
  std::vector<std::pair<double, double>> arcs = {{0.0, 2.0 * M_PI}};
  if (!cone.arcs.empty()) arcs = cone.arcs;
  return ChiOverArcs(terms, constant, arcs, best_dir);
}

// The eigendecomposition the model reads (the largest eigenpair and the resolution tol): the plane's closed form, or
// the self-adjoint solver with nd_trust_region.h's resolution.
template <int Dim>
struct SymEigT {
  double lo = 0.0, hi = 0.0;
  Eigen::Matrix<double, Dim, 1> vlo = Eigen::Matrix<double, Dim, 1>::Zero(), vhi = Eigen::Matrix<double, Dim, 1>::Zero();
  double tol = 0.0;
};
template <int Dim>
SymEigT<Dim> SymEigD(const Eigen::Matrix<double, Dim, Dim>& A) {
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Dim, Dim>> es(A);
  SymEigT<Dim> r;
  r.lo = es.eigenvalues()[0];
  r.hi = es.eigenvalues()[Dim - 1];
  r.vlo = es.eigenvectors().col(0);
  r.vhi = es.eigenvectors().col(Dim - 1);
  r.tol = TrGamma(8.0) * A.cwiseAbs().sum();
  return r;
}
template <>
inline SymEigT<2> SymEigD<2>(const Eigen::Matrix2d& A) {
  const SymEig e = SymEig2(A(0, 0), A(0, 1), A(1, 1));
  SymEigT<2> r;
  r.lo = e.lo;
  r.hi = e.hi;
  r.vlo = e.vlo;
  r.vhi = e.vhi;
  r.tol = e.tol;
  return r;
}

// p.K.p (the plane in the scalar form the root solve has always evaluated it in).
template <int Dim>
double QuadForm(const Eigen::Matrix<double, Dim, Dim>& K, const Eigen::Matrix<double, Dim, 1>& p) {
  return p.dot(K * p);
}
template <>
inline double QuadForm<2>(const Eigen::Matrix2d& K, const Eigen::Vector2d& p) {
  return K(0, 0) * p.x() * p.x() + 2.0 * K(0, 1) * p.x() * p.y() + K(1, 1) * p.y() * p.y();
}

// The unit direction opposite the gradient accumulated so far (a term standing on its contact moves along the descent
// of the rest); the first axis when there is none.
template <int Dim>
Eigen::Matrix<double, Dim, 1> DescentUnit(const Eigen::Matrix<double, Dim, 1>& g) {
  const double gr = g.norm();
  return gr > 0.0 ? Eigen::Matrix<double, Dim, 1>(-g / gr) : Eigen::Matrix<double, Dim, 1>::UnitX();
}
template <>
inline Eigen::Vector2d DescentUnit<2>(const Eigen::Vector2d& g) {
  const double gr = std::hypot(g(0), g(1));
  return gr > 0.0 ? Eigen::Vector2d(-g(0) / gr, -g(1) / gr) : Eigen::Vector2d(1.0, 0.0);
}

}  // namespace scope_pursuit_internal

// The derivatives of a term's distance at a pose (see PursueInScope).
template <int Dim>
struct ScopeTermDeriv {
  using Vec = Eigen::Matrix<double, Dim, 1>;
  using Mat = Eigen::Matrix<double, Dim, Dim>;
  bool analytic = false;
  double r = 0.0;  // the length of the first leg of the course (for the gradient's error bound)
  // The iterate stands on the end of the course (the contact point itself): d = |x - c| is a cone there, its gradient is
  // the unit vector of the direction of motion, which no derivative gives. The solver takes the direction of the
  // model's descent (see the term loop).
  bool vertex = false;
  // The contact point is the NEAREST point of the referent's boundary (it moves with the pose): the gradient is exact by
  // the envelope theorem, the Hessian (I - u u^T) / r holds only for a FIXED point, so it is differenced.
  bool moving_contact = false;
  Vec g = Vec::Zero();
  Mat h = Mat::Zero();
  // How the distance is a minimum over routes (the pieces the stationarity measure and the tie scan list): route_kind 0
  // none (smooth there), 1 the geodesic to `route_to` in the scope's free space under its exemptions, 2 the radius-0
  // geodesic to `route_to`, 3 and 4 the nearest point of a set of rings (`rings`: a container's border, a referent's
  // boundary or its named parts).
  int route_kind = 0;
  Vec route_to = Vec::Zero();
  // route_kind 2: the contact's own footprints (the referent's), open on the final leg onto `route_to` (a spacing
  // relation's mapped contact is measured so, as the source's nearest contact is; OwnFootprints).
  std::vector<int64_t> route_open;
  // The straight segments the course is made of (0: not known), for the arithmetic error of the distance.
  int legs = 0;
  std::function<std::vector<std::vector<Vec>>()> rings;
};

// One relation of the anchor as the pursuit sees it.
template <int Dim>
struct ScopeTerm {
  using Vec = Eigen::Matrix<double, Dim, 1>;
  double weight = 0.0;
  double d_star = 0.0;    // the distance the relation is to have after this update (the ratio continuation)
  double d_sample = 0.0;  // its distance at the anchor's current position
  double update_count = 1.0;
  int multiplicity = 1;
  // A term measured as the geodesic over the scope to this point (an engaged object's mapped contact; a pose's mapped
  // contact one level down). Without it the term is the relation's own distance (`distance`, `deriv`).
  std::optional<Vec> goal;
  // The end of the one-relation course, its kind and the direction it arrives along.
  Vec course_goal = Vec::Zero();
  bool course_relation = true;  // false: a container's border relation (its step is its own distance over n)
  std::optional<Vec> entering;
  // A spacing relation's mapped contact: the referent's own footprints, open on the final leg onto it (its course is
  // the one its distance measures, RelationScope::ReachabilityOpen).
  std::vector<int64_t> open;
  std::function<double(const Vec&, bool fast)> distance;  // NaN or < 0: not evaluable at the pose
  std::function<ScopeTermDeriv<Dim>(const Vec&)> deriv;     // analytic = false: differenced
};

template <int Dim>
struct ScopePursuitInput {
  using Vec = Eigen::Matrix<double, Dim, 1>;
  const RelationScope<Dim>* scope = nullptr;
  Vec start = Vec::Zero();  // the anchor's current position (its sample, in F when F has a pose for it)
  std::vector<ScopeTerm<Dim>> terms;
  int gn_safety_cap = 10000;
  double exact_eps_m = 1e-3;
  int budget_evals = 0;  // <= 0: no budget
  bool warm_cold = false;
};

template <int Dim>
struct ScopePursuitResult {
  using Vec = Eigen::Matrix<double, Dim, 1>;
  Vec position = Vec::Zero();
  Vec objective_seed_position = Vec::Zero();
  bool has_objective_seed = false, no_request = false, objective_stationary = false, objective_exact = false;
  bool objective_fallback = true, objective_warm_start_used = false;
  bool budget_hit = false, budget_incumbent_is_warm_start = false;
  int budget_hit_iteration = -1, iterations = 0, objective_iterations = 0, objective_relation_count = 0;
  int chi_descent_taken = 0, chi_descent_tries = 0, fd_terms = 0, projection_excursions = 0, retract_cap_hits = 0;
  int route_list_queries = 0;
  double incumbent_objective = 0.0, objective_before = 0.0, objective_after = 0.0;
  SolverStatus status = SolverStatus::kConverged;
  SolverResult::GnStop gn_stop = SolverResult::GnStop::kNone;
  FeasibilityOutcome feasibility = FeasibilityOutcome::kUnconstrained;
  // The one-relation course step (v1 CalculateDistanceToMove), for an observer to recompute.
  bool one_relation_step = false;
  double step_m = -1.0;
  std::vector<Vec> course;
  bool course_relation = false, smoothed = false;
};

// The pursuit. `evaluations` counts the relation evaluations (per term per evaluation of J), `budget_tripped` is set
// when the evaluation budget runs out, `position_resolution_m` receives the last iterate's resolution (U2's displacement).
template <int Dim>
void PursueInScope(const ScopePursuitInput<Dim>& in, ScopePursuitResult<Dim>& result, int& evaluations,
                   bool& budget_tripped, double& position_resolution_m) {
  using Vec = Eigen::Matrix<double, Dim, 1>;
  using Mat = Eigen::Matrix<double, Dim, Dim>;
  using Route = typename ScopeGeometryTypes<Dim>::Route;
  using scope_pursuit_internal::FirstCrossing;
  const RelationScope<Dim>& scope = *in.scope;
  const std::vector<ScopeTerm<Dim>>& terms = in.terms;
  const Vec start = in.start;
  const bool budget_on = in.budget_evals > 0;
  const auto over_budget = [&]() {
    if (!budget_on) return false;
    if (!budget_tripped && evaluations >= in.budget_evals) budget_tripped = true;
    return budget_tripped;
  };
    // The geodesic length of the course from `from` to `to` in F (the length the
    // one-relation step walks), the straight line where F does not connect them
    // (the course falls back to it).
    const auto goal_distance = [&](const Vec& from, const Vec& to) {
      const auto reach = scope.Reachability(from, to);
      if (reach.reachable && std::isfinite(reach.length)) return reach.length;
      return (to - from).norm();
    };
    // e = d(pose) - d*. J = sum w e^2.
    auto eval_errors = [&](const Vec& pos, bool fast, std::vector<double>* errors) {
      double j_r = 0.0;
      int n = 0;
      if (errors) errors->assign(terms.size(), 0.0);
      for (size_t i = 0; i < terms.size(); ++i) {
        ++evaluations;
        if (terms[i].goal) {
          // An engaged object measures the course to its c_p.
          const double err = goal_distance(pos, *terms[i].goal) - terms[i].d_star;
          if (errors) (*errors)[i] = err;
          j_r += terms[i].weight * err * err;
          n += terms[i].multiplicity;
          continue;
        }
        const double dist = terms[i].distance(pos, fast);
        if (!(dist >= 0.0)) {
          if (errors) (*errors)[i] = std::numeric_limits<double>::quiet_NaN();
          continue;
        }
        const double err = dist - terms[i].d_star;
        if (errors) (*errors)[i] = err;
        j_r += terms[i].weight * err * err;
        n += terms[i].multiplicity;
      }
      return std::pair<double, int>(j_r, n);
    };

    auto in_f = [&](const Vec& p) {
      return scope.IsFree(p);
    };
    // The derivatives of a term's distance d(pos) with respect to the pose, analytic as in v1
    // (gradient = -first_segment / seg_len). The distance is the
    // length of the taut course to a fixed point p: d = |x - p1| + (fixed rest of the course), p1 the first
    // waypoint, so within the course's cell (the sequence of waypoints does not change)
    //   grad d = (x - p1) / |x - p1|,   hess d = (I - u u^T) / |x - p1|,   u = grad d.
    // A distance clamped to 0 (inside the referent) has zero derivatives. The same course the kernel measured
    // is walked: NavigableSpatialDefinition::GetDiscretePath uses the free space of GetGeodesicDistance
    // (navigable_geometry_context.cc), the engaged goal's course is QueryReachability's. `analytic` is false for
    // a term this does not cover (see fd_term below): the Euclidean fallback of the kernel in a referent's local
    // frame, which depends on the referent's transform, and a course snapped onto F at its start.
    using TermDeriv = ScopeTermDeriv<Dim>;
    const auto leg_deriv = [](const Vec& x, const Vec& p1) {
      TermDeriv out;
      out.analytic = true;
      const double r = (x - p1).norm();
      out.r = r;
      // v1's kEpsilon: a zero-length first segment has zero gradient.
      if (r > 1e-9) {
        out.g = (x - p1) / r;
        out.h = (Mat::Identity() - out.g * out.g.transpose()) / r;
      }
      return out;
    };
    // The first waypoint of a course that is not the iterate itself (a path that starts with a repeat of the pose, as
    // on the boundary, has a zero first leg and a gradient of 0 where the distance changes at rate 1).
    const auto first_far = [](const std::vector<Vec>& pts, const Vec& x,
                              const Vec& fallback) {
      for (size_t q = 1; q < pts.size(); ++q)
        if ((pts[q] - x).norm() > 1e-9) return pts[q];
      return fallback;
    };
    const auto term_deriv = [&](const Vec& pos, size_t i) -> TermDeriv {
      if (terms[i].goal) {
        const Vec& goal = *terms[i].goal;
        {
          const auto reach = scope.Reachability(pos, goal);
          if (reach.reachable && std::isfinite(reach.length)) {
            if (reach.from_snap_m > ObstacleFreeSpace::kBoundaryHairM) return TermDeriv{};
            const auto& pts = reach.path;
            if (reach.length <= 1e-9) { TermDeriv v; v.analytic = true; v.vertex = true; return v; }
            TermDeriv td = leg_deriv(pos, first_far(pts, pos, goal));
            td.route_kind = 1;
            td.route_to = goal;
            td.legs = std::max<int>(1, static_cast<int>(pts.size()) - 1);
            return td;
          }
        }
        if ((pos - goal).norm() <= 1e-9) { TermDeriv v; v.analytic = true; v.vertex = true; return v; }
        TermDeriv td = leg_deriv(pos, goal);  // the straight line, the course's fallback
        return td;
      }
      return terms[i].deriv(pos);
    };
    // A term without an analytic derivative keeps a forward difference. Interval (Gill, Murray, Saunders & Wright,
    // "Computing forward-difference intervals for numerical optimization", SIAM J. Sci. Stat. Comput. 4 (1983)):
    // h = 2 sqrt(eps_A / |phi''|), eps_A the bound on the error of one evaluation and phi'' the curvature of the
    // term. eps_A is gamma_{s+3} d (Higham, Lemma 3.1 and sec. 4.2; s = 128 segments) and |phi''| <= 1/d for a
    // distance d (the curvature of a distance function is at most 1/d), so h = 2 d sqrt(gamma_131). The
    // difference points into F (forward when the forward point is free, else backward). No curvature term is
    // formed for such a term (Gauss-Newton only). SolverResult::fd_terms counts them.
    const auto fd_term_deriv = [&](const Vec& pos, size_t i, double e0) {
      TermDeriv out;
      const double u = 1.1102230246251565e-16;
      const double d = std::max(e0 + terms[i].d_star,
                                static_cast<double>(std::numeric_limits<float>::epsilon()) * std::max(1.0, pos.cwiseAbs().maxCoeff()));  // the data's precision floors a zero distance
      const double h = 2.0 * d * std::sqrt(131.0 * u / (1.0 - 131.0 * u));
      for (int ax = 0; ax < Dim; ++ax) {
        Vec dir = Vec::Zero();
        dir[ax] = 1.0;
        const double sign = (in_f(pos + h * dir) || !in_f(pos - h * dir)) ? 1.0 : -1.0;
        std::vector<double> e1;
        eval_errors(pos + sign * h * dir, /*fast=*/false, &e1);
        out.g[ax] = std::isfinite(e1[i]) ? (e1[i] - e0) / (sign * h) : 0.0;
      }
      return out;
    };

    // D7: there is no HOLD. A goal the sample cannot reach inside F is pursued as
    // far as F allows (the course below falls back to the straight segment and
    // `retract` stops where it leaves F); it is not a reason to withhold the answer.
    // The last point of [from, to] that is in F.
    const auto retract = [&](const Vec& from,
                             const Vec& to) -> Vec {
      if (in_f(to)) return to;
      if (!in_f(from)) return from;
      Vec lo = from, hi = to;
      for (int bisect = 0; bisect < 60 && (hi - lo).norm() > 0.0; ++bisect) {
        const Vec mid = 0.5 * (lo + hi);
        if (in_f(mid))
          lo = mid;
        else
          hi = mid;
      }
      return lo;
    };

    // The farthest point of the straight segment [from, to] that the body reaches along
    // it without leaving F: the segment's free prefix. `retract` above answers "is the
    // end in F"; this answers "can the body walk there in a straight line".
    const auto free_prefix = [&](const Vec& from,
                                 const Vec& to) -> Vec {
      if (!in_f(from)) return from;
      if (scope.SegmentFree(from, to)) return to;
      Vec lo = from, hi = to;
      for (int bisect = 0; bisect < 60 && (hi - lo).norm() > 0.0; ++bisect) {
        const Vec mid = 0.5 * (lo + hi);
        if (scope.SegmentFree(from, mid))
          lo = mid;
        else
          hi = mid;
      }
      return lo;
    };

    // A move from a to b, both in F, is walkable when the straight segment is free, or when
    // F connects them by a path that exceeds the chord by less than the body radius. Going
    // round the rounded corner of a clearance ring (or along a boundary the polygonal
    // configuration space and the exact ring place a fraction of a millimetre apart)
    // exceeds the chord by less than the radius of that ring; getting to the other side of
    // an obstacle exceeds it by at least the obstacle's width and two rings.
    const auto walkable = [&](const Vec& a, const Vec& b) {
      if (scope.SegmentFree(a, b)) return true;
      const auto way = scope.Reachability(a, b);
      return way.reachable && way.length <= (b - a).norm() + scope.ObstacleClearance();
    };

      const auto [j_sample, n_sample] =
          eval_errors(start, /*fast=*/false, nullptr);
      result.has_objective_seed = true;
      result.objective_seed_position = start;
      result.objective_warm_start_used = !in.warm_cold;

      Vec x = start;
      double j = j_sample;
      result.no_request = terms.empty();
      int gn_iters = 0;
      bool stationary = false;
      bool stopped = false;
      bool incumbent_outside_f = false;  // a budget hit with no iterate in F: the answer is the sample, outside F
      // The best feasible incumbent of this solve (the budget's answer): the lowest-J iterate IN F seen. The warm start (the sample) is
      // the incumbent only when it is in F; a sample outside F (outside the walkable, so not projected above) is a candidate of last
      // resort, taken only when no iterate in F was seen, and then the result says so (FeasibilityOutcome::kSampleOutside), it is
      // never labelled feasible.
      const bool warm_feasible = in_f(x);
      Vec best_x = x;
      double best_j = warm_feasible ? j : std::numeric_limits<double>::infinity();
      bool best_is_warm = true;
      bool best_feasible = warm_feasible;
      const auto note_incumbent = [&]() {
        if (j < best_j && in_f(x)) {
          best_x = x;
          best_j = j;
          best_is_warm = false;
          best_feasible = true;
        }
      };
      if (terms.size() == 1) {
        // One relation: v1's single-target step (BaseSingleTargetPlanner::
        // CalculateDeltaState). The receiver walks min(L/n, L) of its own
        // smoothed course to the mapped contact, L the length of that course, n
        // the source's remaining update count. The course is the geodesic path
        // smoothed into a spline that leaves along the body's heading and arrives
        // along the mapped entering direction.
        const ScopeTerm<Dim>& term = terms.front();
        Vec goal = term.goal ? *term.goal : term.course_goal;
        std::vector<Vec> course;
        bool course_in_f = false;  // the course is a path through F, not the straight fallback
        double reach_length = -1.0;
        {
          const auto reach = term.open.empty() ? scope.Reachability(start, goal)
                                               : scope.ReachabilityOpen(start, goal, term.open);
          if (reach.reachable) {
            course = reach.path;
            course_in_f = course.size() >= 2;
            reach_length = reach.length;
          }
        }
        if (course.size() < 2) course = {start, goal};
        double length = 0.0;
        for (size_t i = 1; i < course.size(); ++i)
          length += (course[i] - course[i - 1]).norm();
        // The course's LENGTH is the scope's geodesic (measured from the endpoints' feet on F's boundary); its points are
        // the walkable polyline, which may exceed it by the free space's boundary hairs (a feasibility offset, not a
        // length: ObstacleFreeSpace::FindPath). L is the geodesic.
        if (course_in_f && reach_length >= 0.0) length = reach_length;
        // A container's border relation measures its depth below the border, not the
        // length of a course to a contact (see eval_errors), so its step stays the
        // relation's own distance over n along the discrete course; the smoothed
        // course is the object relations' (their distance IS the course to the
        // mapped contact).
        const bool course_relation = term.course_relation;
        const std::optional<Vec> entering = term.entering;
        // v1 smooths the course whenever the generator succeeds, and keeps the
        // discrete course when it declines. v1 clamps the control points into
        // the convex cell of each segment; the v2 cells carry no obstacle
        // footprints, so the curve is checked against the free space itself and
        // the discrete course stands when the curve leaves it.
        // Only the entering direction is used. v1
        // also seeds the start tangent from the body's yaw; that couples the
        // proposal to the embodied heading (a body facing away takes a curved
        // first step), and on the full matrix it raised the peak speed ratio of
        // the proposal stream from 273 to 763 and the infeasible proposals from
        // 690 to 710 without moving a single identity result, so the course
        // leaves along its own first segment.
        std::optional<BezierCourseT<Dim>> curve;
        if (course_relation) curve = BezierCourseT<Dim>::Build(course, std::nullopt, entering);
        if (curve) {
          const std::vector<Vec> chords = curve->Samples(kCurveChecksPerSegment);
          for (size_t i = 1; i < chords.size() && curve; ++i)
            if (!scope.SegmentFree(chords[i - 1], chords[i]))
              curve.reset();
        }
        const double total = curve ? curve->Length() : length;
        const double arc =
            course_relation
                ? std::min(total / term.update_count, total)
                : std::min(term.d_sample / term.update_count, length);
        Vec reached = course.back();
        if (curve) {
          reached = curve->PointAfter(arc);
        } else if (arc < length) {
          double walked = 0.0;
          for (size_t i = 1; i < course.size(); ++i) {
            const double leg = (course[i] - course[i - 1]).norm();
            if (walked + leg >= arc && leg > 0.0) {
              reached = course[i - 1] + (course[i] - course[i - 1]) *
                                            ((arc - walked) / leg);
              break;
            }
            walked += leg;
          }
        }
        // What an observer needs to recompute the step from its inputs.
        result.one_relation_step = true;
        result.step_m = arc;
        result.course = course;
        result.course_relation = course_relation;
        result.smoothed = curve.has_value();
        // A course through F is walkable end to end. With none (the goal is not reachable
        // from the sample in F, e.g. across a gap narrower than the body) the receiver
        // pursues the straight line as far as F allows: the free prefix of the segment, so
        // the answer is a pose the body can walk to.
        x = course_in_f ? retract(start, reached) : free_prefix(start, reached);
        j = eval_errors(x, /*fast=*/false, nullptr).first;
        stationary = true;
        stopped = true;
      } else {
        // Several relations: the objective J = sum w (d - d*)^2 over the relations' geodesic distances, descended
        // from the embodied sample by a trust-region Newton method with the exact Hessian (below). v1's StepSync
        // ran damped Gauss-Newton with a fixed schedule and a 10-iteration cap; this replaces both
        // with a trust-region Newton method and a resolution-only stop. The 1 mm objective
        // tolerance only decides the reported EXACT; it is not a stop rule. Acceptance uses the exact kernel.
        //
        // Iterate to convergence. in.gn_safety_cap is not a convergence budget and does
        // not change the answer; it only stops a loop that never meets a stop rule below. Reaching it is
        // NOT_CONVERGED (a defect, counted in n_not_converged), not a minimiser. The stop is the combined test of
        // Gill, Murray & Wright (1981), sec. 8.2.3.2, U1 + U2 + U3 together (below). A trust radius that collapses to the
        // representable resolution of x ends the solve NOT converged.
        // Trust-region Newton with the exact Hessian (Nocedal & Wright, Numerical Optimization, 2nd ed. 2006,
        // Algorithm 4.1: N&W 2006 Alg. 4.1, constants from memory, line not verified (the book was not available);
        // the subproblem min g.p + p.H.p / 2, |p| <= Delta is solved EXACTLY, sec. 4.3 / More &
        // Sorensen 1983, in closed form in 2-D from the eigendecomposition of H, including the hard case and
        // negative curvature). Acceptance eta = 0 (the book: eta in [0, 1/4)); radius update as Algorithm 4.1:
        // rho < 1/4: Delta <- Delta / 4; rho > 3/4 and |p| = Delta: Delta <- min(2 Delta, Delta_max); else
        // unchanged. Delta_max (the book's overall bound) is the diameter of the navigable scope, Delta_0 =
        // Delta_max.
        constexpr double kEps = 2.220446049250313e-16;  // double epsilon
        // The accuracy of J as computed (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., Lemma 3.1 and
        // sec. 4.2; u = 2^-53, gamma_k = k u / (1 - k u)); the same derivation as the harness's resolution of J (runner.cc,
        // LocalMinimumOfJ), here without the data-precision term (the inputs' rounding perturbs the problem once, the
        // same at every x; it does not limit the comparison of J at two points).
        // A distance of s segments is evaluated with relative error gamma_{s+3}; e = d - d* adds one rounding, so
        // |e~ - e| <= E = gamma_{s+3} d + u (d + d*); a term w e^2 carries w (2 |e| E + E^2) + gamma_2 w e^2 and the sum
        // of n terms adds gamma_n J.
        constexpr double kU = 1.1102230246251565e-16;
        const auto gamma_k = [&](double k) { return k * kU / (1.0 - k * kU); };
        // The route's own segment count s (the evaluator knows the route, geometry/obstacle_free_space.h RoutePiece), not an
        // assumed bound: a term without a route list takes the course it was differentiated along, and one without a
        // course the most any shortest path can have, one segment per node of the visibility graph and one more.
        const auto eval_err = [&](double e, double d_star, int legs) {
          const double d = std::fabs(e + d_star);
          return gamma_k(legs + 3) * d + kU * (d + d_star);
        };
        // Delta_max, the book's overall bound: the diameter of the navigable scope (no step of a solve in it is
        // longer); with no scope, the sum of the relations' distances at the sample.
        double delta_max = scope.Diameter();
        if (!(std::isfinite(delta_max) && delta_max > 0.0)) delta_max = 0.0;
        if (!(delta_max > 0.0))
          for (const auto& t : terms) delta_max += std::max(t.d_sample, 0.0);
        if (!(delta_max > 0.0)) delta_max = std::numeric_limits<double>::min();
        double delta = delta_max;
        double last_dj = std::numeric_limits<double>::infinity(), last_dx = last_dj;
        bool last_constrained = false;  // the last step tried was an active-set (boundary) step
        // ---- The active-piece directional solver ----
        constexpr double kHairM = ObstacleFreeSpace::kBoundaryHairM;
        // The routes of a term at xx whose length is within `slack` of the shortest (slack 0: those it ties with to
        // rounding). Empty for a term with no route list (a vertex, a term the course does not describe): smooth there.
        const auto pieces_of = [&](const TermDeriv& td, const Vec& xx, double slack) {
          std::vector<Route> out;
          if (!td.analytic || td.vertex) return out;
          ++result.route_list_queries;
          if (td.route_kind == 1) {
            out = scope.GeodesicRoutesUnder(xx, td.route_to, slack);
          } else if (td.route_kind == 2) {
            out = td.route_open.empty() ? scope.GeodesicRoutes(xx, td.route_to, slack)
                                        : scope.GeodesicRoutesOpen(xx, td.route_to, slack, td.route_open);
          } else if ((td.route_kind == 3 || td.route_kind == 4) && td.rings) {
            // The boundary's rings in the world, from the term (the polygon of its contact, lifted).
            const std::vector<std::vector<Vec>> world_rings = td.rings();
            std::vector<const std::vector<Vec>*> rings;
            for (const auto& r : world_rings) rings.push_back(&r);
            out = scope.ContactRoutes(xx, rings, slack);
          }
          return out;
        };
        const auto unit_of = [](const Route& pc, const Vec& xx) {
          const Vec d = xx - pc.first;
          const double r = d.norm();
          return r > 1e-9 ? Vec(d / r) : Vec(Vec::Zero());
        };
        // The Hessian of a route's length at xx: (I - g g^T) / r for the straight first leg to a fixed waypoint, and 0 for a
        // route straight to the nearest point of a boundary edge (the distance to a line has no curvature along it).
        const auto piece_hess = [&](const Route& pc, const Vec& xx) {
          const Vec g = unit_of(pc, xx);
          const double r = (xx - pc.first).norm();
          if (pc.flat_foot || !(r > 1e-9)) return Mat(Mat::Zero());
          return Mat((Mat::Identity() - g * g.transpose()) / r);
        };
        // Co-active: the length intervals [value -+ err] of route a and of the shortest overlap (no gap
        // constant, no widening by a displacement). A route that is not co-active is shorter or longer by more than the
        // arithmetic can resolve and J is smooth in it there.
        const auto is_coactive = [&](const Route& a, const Route& shortest) {
          return a.value - shortest.value <= a.err + shortest.err;
        };
        // The visibility graph's node count (builds the graph once; it bounds the segments of any route). It is the radius-0 graph's: for a
        // route_kind 1 term (a goal reached under the solve's exemptions, whose free space is another graph) it is not that graph's count, only
        // the margin chi's listing keeps; the routes it lists are filtered by is_coactive, which uses each route's own error.
        const auto list_nodes = [&]() -> double {
          return scope.RouteSegmentBound();
        };
        // chi(xx): the rate of the steepest first-order descent of J that stays in F, over the exact tangent cone.
        const auto chi_at = [&](const Vec& xx, const std::vector<double>& e,
                                const std::vector<TermDeriv>& effs, Vec* dir = nullptr,
                                std::vector<std::vector<Route>>* lists = nullptr) {
          std::vector<scope_pursuit_internal::ChiTermT<Dim>> ct;
          if (lists) lists->assign(terms.size(), {});
          double constant = 0.0;
          for (size_t i = 0; i < terms.size(); ++i) {
            if (!std::isfinite(e[i])) continue;
            const double w = std::max(terms[i].weight, 0.0);
            const double c = 2.0 * w * e[i];
            const TermDeriv& td = effs[i];
            if (td.vertex) {  // d = |x - contact|: the rate is |v| = 1 in every direction
              constant += c;
              continue;
            }
            scope_pursuit_internal::ChiTermT<Dim> t;
            t.c = c;
            // The listing keeps the routes within the arithmetic's margin of the shortest, as GeodesicRoutes does: a bound on the difference
            // of two lengths' errors, 2 gamma_{nodes + 3} d (no route has more segments than the graph has nodes). The ones whose
            // intervals do not overlap are not active (chi counts only those that do) but are what the descent looks for: a tie a few
            // 1e-14 m away. ContactRoutes drops them unless it is given this slack.
            const double d_i = std::fabs(e[i] + terms[i].d_star);
            auto pc = pieces_of(td, xx, lists ? 2.0 * gamma_k(list_nodes() + 3) * d_i : 0.0);
            for (const auto& a : pc)
              if (is_coactive(a, pc[0])) t.g.push_back(unit_of(a, xx));
            if (lists) (*lists)[i] = std::move(pc);
            if (t.g.empty()) t.g.push_back(td.g);
            ct.push_back(std::move(t));
          }
          return scope_pursuit_internal::ChiOverCone<Dim>(ct, constant, scope.TangentConeAt(xx, kHairM), dir);
        };
        // The trust-region subproblem min g.p + p.H.p / 2, |p| <= dlt, solved exactly in 2-D (Nocedal & Wright 2006,
        // sec. 4.3; More & Sorensen 1983), over a Jacobian that may be rank deficient. Rank (Higham, Accuracy and
        // Stability of Numerical Algorithms, 2nd ed.): the Gauss-Newton matrix N = J^T W J is formed from the unit
        // gradients with an error of at most gamma_{m+13} tr(N) in the 2-norm (m terms: each entry is a sum of m
        // products of two unit-vector components, each component carrying gamma_5, the product gamma_2, the sum
        // gamma_{m+1}) and its eigenvalues are computed with at most SymEig::tol (SymEig2). An eigenvalue of N not above
        // the sum is zero as far as the arithmetic knows, and the gradient J^T W e of J has no component along its
        // eigenvector (it lies in the range of J^T): there is no first-order information there, and the exact Hessian's
        // curvature along it is residual curvature (small), whose Newton step -c / m amplifies the rounding of c. The step
        // is then the minimum-norm Gauss-Newton step in the range; the null direction is followed only along negative
        // curvature that the Hessian's own eigenvalue tolerance resolves.
        struct TrOut {
          Vec p = Vec::Zero();
          bool radius_active = false;
          bool rank_def = false;
        };
        // A component of the gradient not above the rounding uncertainty of the gradient is zero (gtol, below): g is a sum of
        // w e (grad d), each e carrying the evaluation error E of its distance, the representable resolution of the pose
        // (a residual cannot be set finer than |grad d| = 1 times it) and each unit vector gamma_5, so
        // |dg| <= sum w (E + xres + gamma_5 |e|) in any direction. The Newton step -c / m along an eigendirection with such a
        // component, and a curvature m that small, amplifies that uncertainty into a step; the component is dropped.
        // The subproblem is the shared N-D step engine (nd_trust_region.h) at Dim = 2, its closed form: the same
        // arithmetic this solve has always used.
        const auto tr_solve = [&](const Vec& gv_in, const Mat& A, const Mat& N, int nterms, double dlt,
                                  double gtol) -> TrOut {
          const TrStep<Dim> st = SolveTrSubproblem<Dim>(
              gv_in, A, N, nterms, dlt, gtol, [&](const Vec& pa, const Vec& pb) {
                return eval_errors(x + pa, /*fast=*/false, nullptr).first <=
                       eval_errors(x + pb, /*fast=*/false, nullptr).first;
              });
          TrOut out;
          out.p = st.p;
          out.radius_active = st.radius_active;
          out.rank_def = st.rank_def;
          return out;
        };
        // Chains of steps J cannot resolve. A step whose decrease is R or less is a step the arithmetic cannot tell from none; a chain of them is
        // a walk (a face slide whose length halved at every iteration, 5e-11 m, 2.5e-11 m, ..., each gaining less than R, then a landing on a
        // route tie one ulp at a time, to the iteration cap: ma_meeting_room|ma_back_and_forth). Two counts bound the chains. What each does,
        // exactly (this is what the code does, not a general rule):
        //   * normal_unresolved_streak counts the accepted loop steps whose decrease is R or less in a row (a step above R resets it). While it
        //     is at kMaxUnresolvedNormal, a PROJECTED step with a decrease of R or less is rejected, and a non-projected step is not taken on
        //     the model because it is within R (the branch that accepts a step whose predicted reduction is R or less). The model's own
        //     non-projected steps are not limited by it: they are accepted as the trust region accepts any step that lowers J.
        //   * chi_unresolved_streak counts the descents along chi's direction that gained R or less (a landing on a tie, a search, a search
        //     again from the point reached); at kMaxUnresolvedChi the next is refused. ANY accepted loop step resets it.
        // Applying the first count to every branch was tried and is wrong here: ma_meeting_room|ma_back_and_forth then collapses its radius 1e-10 m
        // from a vertex of F with chi 7e-4 (the slides bisect towards the vertex; the descent that reaches it needs the interleaved resets).
        // The limits are POLICY COUNTS: neither derived nor from a paper (the watchdog of Chamberlain, Powell, Lemarechal & Pedersen 1982 is the nearest
        // published idea; the paper was not available to check, so none is claimed). Their basis is measured insensitivity, below. They never
        // certify convergence: they only let the solver keep iterating, and a solve they end is not called converged by them. Any change to
        // them needs the same measurement. Sensitivity, measured: on the quick level and on the 41 slow
        // rows that failed at earlier candidates, (N, C) in {(1,1), (1,2), (1,3), (1,4), (2,2), (2,3)} give the same counts (0 not converged,
        // 0 n_min_fail, and no row differing in any count). The four full-level rows that were left open with C = 2 (occluded|passby_mid,
        // look_passby, gaze_switch_walk at 1.4 m/s 15 deg; occluded|teleport|aniso0.8x1.3 2.8 m/s 255 deg) converge with C = 3 and not with C = 1
        // or 2 (N = 1 or 2): C = 3 is the smallest value that does, and is what is used.
        constexpr int kMaxUnresolvedChi = kTrMaxUnresolvedChi;
        int chi_unresolved_streak = 0;
        int normal_unresolved_streak = 0;
        // ---- The shared outer loop (nd_trust_region_loop.h) with this solve's planar F as its hooks ----
        // The iteration's model and the quantities the hooks share (assigned by `assemble` each iteration).
        constexpr int kRetractSteps = 60;
        std::vector<double> e0;
        // The model: H half the Hessian of J, Nm its Gauss-Newton part J^T W J, gacc half the gradient (each symmetric matrix
        // is accumulated on and above its diagonal and mirrored).
        Mat H = Mat::Zero(), Nm = Mat::Zero();
        Vec gacc = Vec::Zero();
        double j_res = 0.0, g_tol = 0.0, tau_f = 0.0, xres = 0.0;
        std::vector<TermDeriv> tds, eff;
        scope_pursuit_internal::SymEigT<Dim> heig{}, nsym{};
        Vec gvec = Vec::Zero(), u1_hard = Vec::Zero();
        Vec step = Vec::Zero();
        double g_u3_memo = -1.0;
        Vec g_u3_dir = Vec::Zero();
        std::vector<std::vector<Route>> g_u3_lists;
          const auto g_u3_fn = [&]() {
            if (g_u3_memo >= 0.0) return g_u3_memo;
            g_u3_memo = chi_at(x, e0, eff, &g_u3_dir, &g_u3_lists);
            return g_u3_memo;
          };
          const auto chord_retract = [&](Vec y, double* jy_out) {
            std::vector<double> ey;
            double jy = eval_errors(y, /*fast=*/false, &ey).first;
            if (nsym.hi > 0.0) {
              for (int it = 0;; ++it) {
                if (it >= kRetractSteps) {
                  ++result.retract_cap_hits;
                  break;
                }
                if (over_budget()) break;  // the best point so far, which J has judged
                Vec gch = Vec::Zero();
                for (size_t i = 0; i < terms.size(); ++i) {
                  if (!std::isfinite(ey[i]) || eff[i].vertex) continue;
                  gch += std::max(terms[i].weight, 0.0) * ey[i] * eff[i].g;
                }
                const Vec pn = -(u1_hard.dot(gch) / nsym.hi) * u1_hard;
                if (!(pn.norm() > kEps * std::max(1.0, y.cwiseAbs().maxCoeff()))) break;
                const Vec y2 = y + pn;
                if (!y2.allFinite() || !in_f(y2) || !walkable(y, y2)) break;
                std::vector<double> e2;
                const double j2 = eval_errors(y2, /*fast=*/false, &e2).first;
                if (!(j2 < jy)) break;
                const bool worth = jy - j2 > j_res;
                y = y2;
                jy = j2;
                ey = std::move(e2);
                if (!worth) break;
              }
            }
            if (jy_out) *jy_out = jy;
            return y;
          };
          const auto chi_descent = [&]() {
            ++result.chi_descent_tries;
            if (chi_unresolved_streak >= kMaxUnresolvedChi) return 0.0;
            if (over_budget()) return 0.0;
            const Vec v = g_u3_dir;
            // On a face of F the direction chi measured runs along it, and a point constructed on the face's line is accepted or rejected by the
            // last bit of the membership test (see the face slides): the trial points are pulled into F by the rounding of that test, along
            // the facets' inward normals.
            Vec inward = Vec::Zero();
            {
              for (const auto& f : scope.TangentConeAt(x, kHairM).facets) inward -= f.normal;
              if (inward.norm() > 0.0) inward *= 2.0 * gamma_k(8) * std::max(1.0, x.cwiseAbs().maxCoeff()) / inward.norm();
            }
            const double rx = std::sqrt(tau_f) * (1.0 + x.norm());
            const double s0 = 4.0 * kEps * std::max(1.0, x.cwiseAbs().maxCoeff());
            // First the nearest route tie along the direction, within U2's displacement: a route that is longer by a few 1e-14 m (the
            // length intervals do not overlap) overtakes there, J's minimum along the line is the tie, and landing ON it is what
            // makes the routes co-active (chi 0). FirstCrossing is the closed form of where two routes' lengths are equal along
            // the segment; the point is taken if J has not risen by more than R.
            if (v.norm() > 0.0) {
              const Vec p = rx * v;
              double tc = 2.0;
              // (the route lists chi was just built from: they hold the routes within the arithmetic's margin of the shortest, the
              // ones that are not co-active included, so no second walk of the route grid)
              for (size_t i = 0; i < g_u3_lists.size(); ++i) {
                const auto& pc = g_u3_lists[i];
                for (size_t b = 1; b < pc.size(); ++b) tc = std::min(tc, FirstCrossing(x, p, pc[0], pc[b]));
              }
              // (a tie closer than x can move, within its representable resolution, is not a point to land on: x + step is x)
              if (tc > 0.0 && tc < 1.0 && (x + tc * p) != x) {
                const Vec y = x + tc * p + inward;
                if (in_f(y) && walkable(x, y)) {
                  const double jy = eval_errors(y, /*fast=*/false, nullptr).first;
                  if (jy <= j + j_res) {
                    last_dj = j - jy;
                    last_dx = tc * rx;
                    const double taken_len = tc * rx;
                    chi_unresolved_streak = last_dj > j_res ? 0 : chi_unresolved_streak + 1;
                    x = y;
                    j = jy;
                    ++result.chi_descent_taken;
                    return taken_len;
                  }
                }
              }
            }
            // The trial points are x + s v + inward, so a rise along v is measured from J(x + inward), the same pull without the
            // step along v. On a face of F the pull alone can raise J by more than R (J's gradient pressing into the wall); that
            // rise says nothing about the descent along v. Measured from J(x), the search ended at its first length on the pull's
            // rise, and the solve at a collapsed radius was NOT_CONVERGED with a real descent along v (ma_two_zones|ma_circle_cluster|
            // rearranged at 0.6 m/s, gate 0.05: 36 frames; the pull raised J by 8.2e-18 with R 5.7e-18, chi 5.5e-5, while the same
            // direction lowered J by 1.8e-12 at 1.1e-8 m). A pulled point outside F gives no base: the rise is measured from J(x).
            double j_base = j;
            bool base_ok = true;
            if (v.norm() > 0.0 && inward.norm() > 0.0) {
              const Vec xb = x + inward;
              base_ok = in_f(xb);
              if (base_ok) j_base = eval_errors(xb, /*fast=*/false, nullptr).first;
            }
            const double j_ceiling = (base_ok ? std::max(j, j_base) : j) + j_res;
            double best_s = 0.0, best_j = std::numeric_limits<double>::infinity();
            double s_rise = 0.0;  // the first length at which J rose by more than R above the base (the search's end), 0 if it did not
            double j_rise = 0.0;  // J there
            for (double s = s0; v.norm() > 0.0 && s <= std::max(rx, 2.0 * s0); s *= 2.0) {
              if (over_budget()) break;
              const Vec y = x + s * v + inward;
              if (!in_f(y) || !walkable(x, y)) break;
              const double jy = eval_errors(y, /*fast=*/false, nullptr).first;
              if (jy > j_ceiling) {
                s_rise = s;
                j_rise = jy;
                break;
              }
              if (jy <= best_j) {
                best_j = jy;
                best_s = s;
              }
            }
            if (!(best_s > 0.0 && best_j <= j)) {
              // Refuted at J's resolution (TrChiRefutedAtResolution), the rise measured from the same base as the search's. A
              // pulled point outside F gives no base, and the rule does not fire.
              if (s_rise > 0.0 && g_u3_memo >= 0.0 && base_ok &&
                  TrChiRefutedAtResolution(g_u3_memo, s_rise, j_rise, j_base, j_res))
                return -1.0;
              return 0.0;
            }
            last_dj = j - best_j;
            last_dx = best_s;
            chi_unresolved_streak = last_dj > j_res ? 0 : chi_unresolved_streak + 1;
            x = x + best_s * v + inward;
            j = best_j;
            ++result.chi_descent_taken;
            return best_s;
          };
        const auto project_iterate = [&]() {
          // The objective lives on F. An iterate a rounding error outside it (an accepted
          // boundary step lands on the wall's hair) is projected onto F, as the sample is at the
          // start of the solve, instead of ending the descent where it stands.
          if (!in_f(x)) {
            if (const auto onto = scope.NearestFree(x);
                onto && in_f(*onto)) {
              // An excursion beyond the free space's boundary hair is not a rounding error:
              // counted (SolverResult::projection_excursions) and reported by the harness.
              if ((*onto - x).norm() > ObstacleFreeSpace::kBoundaryHairM) ++result.projection_excursions;
              last_dx = std::max(last_dx, (*onto - x).norm());  // the projection is part of the step that reached x
              x = *onto;
              j = eval_errors(x, /*fast=*/false, nullptr).first;
            }
          }
        };
        const auto assemble = [&]() {
          e0.clear();
          // The kernel itself is evaluated (the fast distance chart is a different function, and a step of
          // that function is not a step of J). Once per iterate: a rejected step leaves the iterate where
          // it is, so the damping loop below reuses it.
          eval_errors(x, /*fast=*/false, &e0);
          // Gradient and Hessian of J = sum w e^2, e = d - d*. Half the gradient is g = sum w e grad d and
          // the Hessian of J / 2 is sum w grad d grad d^T (Gauss-Newton) + sum w e hess d (the residuals'
          // curvature; Nocedal & Wright, Numerical Optimization, 2nd ed., sec. 10.1). Gauss-Newton keeps the
          // first term only and converges linearly wherever the residuals at the solution are not small
          // (sec. 10.2); the full Hessian restores the quadratic rate (sec. 3.3, Theorem 3.5). The
          // derivatives of each distance are analytic (term_deriv above): no difference interval.
          H.setZero();
          gacc.setZero();
          Nm.setZero();  // the Gauss-Newton part J^T W J
          j_res = 0.0;                             // R, the accuracy of J as computed (above)
          g_tol = 0.0;                             // the rounding uncertainty of the gradient, sum w (E + gamma_5 |e|)
          tds.assign(terms.size(), TermDeriv{});
          eff.assign(terms.size(), TermDeriv{});  // the derivatives the model is built from (after the vertex and difference cases)
          for (size_t i = 0; i < terms.size(); ++i)
            if (std::isfinite(e0[i])) tds[i] = term_deriv(x, i);
          // The Hessian of a term whose contact point moves with the pose has no closed form here: it is the
          // derivative of the (analytic) gradient, differenced. Interval (Gill, Murray, Saunders & Wright 1983):
          // h = 2 sqrt(eps_A / |phi''|) with eps_A = 3u the rounding of the unit vector (gamma_3, u = 2^-53) and
          // |phi''| <= 1/r for the curvature of a distance at distance r (r the first leg), so h = 2 sqrt(3 u r);
          // forward, into F. The terms differenced are counted with the others (SolverResult::fd_terms).
          for (size_t i = 0; i < terms.size(); ++i) {
            if (!std::isfinite(e0[i]) || !tds[i].analytic || !tds[i].moving_contact) continue;
            const double u = 1.1102230246251565e-16;
            const double h = 2.0 * std::sqrt(3.0 * u * std::max(tds[i].r, 1e-12));
            Mat hm = Mat::Zero();
            for (int ax = 0; ax < Dim; ++ax) {
              Vec dir = Vec::Zero();
              dir[ax] = 1.0;
              const double sgn = (in_f(x + h * dir) || !in_f(x - h * dir)) ? 1.0 : -1.0;
              const TermDeriv t2 = term_deriv(x + sgn * h * dir, i);
              if (!t2.analytic) continue;
              hm.col(ax) = (t2.g - tds[i].g) / (sgn * h);
            }
            tds[i].h = 0.5 * (hm + hm.transpose());
            ++result.fd_terms;
          }
          // Terms at the vertex of their cone come after the others: their gradient is the unit vector of the
          // direction of the descent of the rest (moving away from the vertex raises d at rate 1 in any
          // direction, and e = -d* <= 0 there, so J falls along the descent of the rest and the term adds to it).
          for (int pass = 0; pass < 2; ++pass)
          for (size_t i = 0; i < terms.size(); ++i) {
            if (!std::isfinite(e0[i])) continue;
            if ((pass == 1) != tds[i].vertex) continue;
            TermDeriv td = tds[i];
            if (td.vertex) {
              td.g = scope_pursuit_internal::DescentUnit<Dim>(gacc);
            }
            const bool had_route = td.analytic;
            if (!td.analytic) {
              td = fd_term_deriv(x, i, e0[i]);
              ++result.fd_terms;
            }
            eff[i] = td;
            const double w = std::max(terms[i].weight, 0.0);
            {
              // The accuracy R(x) of J(x) (see eval_err): the sum of w (2 |e| E + E^2 + gamma_2 e^2), then gamma_n J.
              // The segments of the course the distance was summed over (term_deriv counts them from the course itself; the
              // evaluator's route list of the same course has the same count), not a second walk of the route grid.
              int legs_i = had_route && td.legs > 0 ? td.legs : 1;
              if (!had_route) {
                // A term without a course (differenced): the most any shortest path has, a segment per node of the graph.
                legs_i = 1 + static_cast<int>(scope.RouteSegmentBound());
              }
              const double e_i = eval_err(e0[i], terms[i].d_star, legs_i);
              j_res += w * (2.0 * std::fabs(e0[i]) * e_i + e_i * e_i + gamma_k(2) * e0[i] * e0[i]);
              // ... and the pose itself is representable to kEps max(1, |x|) only: a residual cannot be resolved finer than
              // |grad d| times that, and grad d is a unit vector.
              g_tol += w * (e_i + kEps * std::max(1.0, x.cwiseAbs().maxCoeff()) + gamma_k(5) * std::fabs(e0[i]));
            }
            for (int r = 0; r < Dim; ++r)
              for (int q = r; q < Dim; ++q) Nm(r, q) += w * td.g[r] * td.g[q];
            for (int r = 0; r < Dim; ++r) gacc[r] += w * e0[i] * td.g[r];
            for (int r = 0; r < Dim; ++r)
              for (int q = r; q < Dim; ++q) H(r, q) += w * td.g[r] * td.g[q] + w * e0[i] * td.h(r, q);
          }
          j_res += gamma_k(static_cast<double>(terms.size())) * j;
          // The Hessian's eigendecomposition is exact to u |H| (a symmetric 2 x 2, Eigen's closed form).
          for (int r = 0; r < Dim; ++r)
            for (int q = 0; q < r; ++q) {
              H(r, q) = H(q, r);
              Nm(r, q) = Nm(q, r);
            }
          heig = scope_pursuit_internal::SymEigD<Dim>(H);
          gvec = gacc;
          xres = kEps * std::max(1.0, x.cwiseAbs().maxCoeff());  // the representable resolution of x
          // The combined termination test of Gill, Murray & Wright, Practical Optimization (1981), sec. 8.2.3.2, all
          // three together. THE FORMS ARE NOT VERIFIED AGAINST THE BOOK (no faithful text
          // found); they are as follows:
          //   U1  F_{k-1} - F_k  <  tau_F (1 + |F_k|)
          //   U2  |x_{k-1} - x_k|  <  sqrt(tau_F) (1 + |x_k|)
          //   U3  |g_k|  <=  cbrt(tau_F) (1 + |F_k|)
          // with tau_F (1 + |F_k|) = R, the accuracy of J as computed (derived above), so tau_F = R / (1 + |J|).
          // U1 is never used alone. Tested on the accepted step that reached this iterate.
          tau_f = j_res / (1.0 + std::fabs(j));
          position_resolution_m = std::sqrt(tau_f) * (1.0 + x.norm());
          // U3 is on chi(x), the rate of the steepest first-order descent of J that stays in F (chi_at above): J is a
          // minimum over routes, so at a tie there is no gradient, and F's boundary is its exact tangent cone, not a
          // probe. Where one route is active and x is interior, chi = |2 g|. Evaluated when the other parts pass.
          g_u3_memo = -1.0;
          g_u3_dir = Vec::Zero();  // the direction of steepest first-order descent that chi measured
          g_u3_lists.clear();  // the route lists chi was built from, by term
          // A curved valley the quadratic model cannot follow (a heavily weighted term whose level set the step leaves at
          // second order: kappa s^2 / 2, costing w kappa^2 s^4 / 4, so the model, which holds that residual fixed, is good for
          // steps of only (gain / (w kappa^2))^(1/3) however flat the valley is: the linear convergence of Gauss-Newton /
          // LM on a singular, nearly zero-residual problem, Nocedal & Wright 2006 sec. 10.2). The standard remedy is a
          // retraction: a step the model's ratio calls inadequate (rho < 3/4, Algorithm 4.1) is also tried restored to the
          // principal direction's level set, by Gauss-Newton steps along u1 alone with this iterate's gradients (a chord),
          // while J keeps falling by more than its accuracy. The lower of the two is kept; J judges.
          // The chord is a root find on the principal coordinate, a candidate generator inside ONE trial step: the outer loop is what
          // converges. A chord that contracts the coordinate by one half per step (the rate bisection guarantees) resolves it to x's
          // representable resolution in the number of halvings a bisection of a double needs, 60 (the bound free_prefix and retract use
          // above). A chord still falling after that is not contracting (a creeping zero-residual valley, where J falls by more than R
          // at every step for ever: 10,000 steps per trial, 16 s in one frame of ma_scatter8): it is abandoned with the best point so
          // far, which J has judged, and the hit is counted (SolverResult::retract_cap_hits). It is not the solve's failure: whether
          // the solve converged is U1 + U2 + U3, which the outer loop tests.
          // The premise of 60, plainly: it is the number of halvings the bisections in this file use (free_prefix, retract: `bisect < 60`, enough for
          // a double's 53 mantissa digits and a few exponent steps). The chord does NOT guarantee that it halves anything: this is a bound on the
          // work of one trial, a heuristic, not a convergence criterion. Reaching it abandons the chord with the best point so far.
          nsym = scope_pursuit_internal::SymEigD<Dim>(Nm);
          u1_hard = nsym.vhi;
          // Stage 3 of the active-piece solver: where chi, the exact rate of the steepest first-order descent, is above the U3 threshold,
          // step ALONG the direction chi measured and accept only a point at which J has not risen. chi counts the routes whose
          // length intervals overlap; a route that is not (it is longer by a few 1e-14 m, say) overtakes within a distance too small for
          // J to see the descent (chi s < R), and beyond it the slope turns up. So the search starts at the representable resolution
          // of x and the length doubles (the trust region's own growth, Algorithm 4.1) while J has not risen by more than its accuracy R
          // (a rise of R or less is not resolved, U1), up to the displacement U2 treats as no move, rx: the largest length J does not
          // refute, with the lowest J among them. A point at which J is lower, or equal within R, is taken: a descent that J cannot
          // see is the one the model (chi) reports. Returns the length taken (0: none).
          TrIterModel<Dim> m;
          m.g = gvec;
          m.A = H;
          m.N = Nm;
          m.nterms = static_cast<int>(terms.size());
          m.j_res = j_res;
          m.g_tol = g_tol;
          m.tau_f = tau_f;
          return m;
        };
        const auto subproblem = [&](const TrIterModel<Dim>& m, double dlt) {
          const TrOut tro = tr_solve(m.g, H, Nm, m.nterms, dlt, g_tol);
          TrStep<Dim> st;
          st.p = tro.p;
          st.radius_active = tro.radius_active;
          st.rank_def = tro.rank_def;
          return st;
        };
        const auto make_trial = [&](const TrStep<Dim>& tro) -> TrTrial<Dim> {
            bool radius_active = tro.radius_active;
            (void)radius_active;
            step = tro.p;
            Vec trial = x + step;
            bool projected = false;  // the step is a projected (active-set) step
            // A step that lands in F but cannot be walked there (it crosses an obstacle's
            // footprint or ring) is a step out of F as far as the body is concerned.
            if (!in_f(trial) || (in_f(x) && !walkable(x, trial))) {
              if (!in_f(x)) { TrTrial<Dim> out; out.abort = true; return out; }
              // The damped step leaves F, so the constrained step lies on the
              // boundary of F: the minimiser of the same damped model over the
              // boundary line (an active-set step). Stopping where the straight
              // step enters F, or at the projection of the trial, advances along a
              // boundary that is not aligned with the Hessian's soft direction by
              // micrometres per iteration, so the iterate crawls and ends wherever a
              // relative-decrease rule fires, not at a minimiser of J over F.
              ObstacleFreeSpace::NearestFreeStatus st =
                  ObstacleFreeSpace::NearestFreeStatus::kSearchFoundNone;
              auto snapped = scope.NearestFree(trial, &st);
              bool on_boundary = false;
              last_constrained = false;
              // The trial is out of F (its nearest free point is the snapped one), or in F
              // across an obstacle the segment cannot cross (no snap: the boundary is read
              // at the exit point alone).
              const bool trial_snapped = snapped && in_f(*snapped) && (trial - *snapped).norm() > 0.0;
              if (trial_snapped || (in_f(trial) && !walkable(x, trial))) {
                // n points out of F at the boundary point nearest the trial, t is the
                // boundary's tangent there. The model's minimiser on the line
                // n.(h) = c (c: how far the boundary is from the iterate along n):
                // h = c n + s t, with A = H + lambda I and A h + g = mu n.
                // The line passes through where the step leaves F (the exact boundary):
                // the nearest free point of the configuration space is only as exact as
                // its polygonal approximation of the clearance rings, which is up to a
                // few millimetres inside F.
                // The boundary of F at the exit point is read exactly: TangentConeAt lists every constraint (a wall
                // edge or vertex, an obstacle's clearance ring, arc or offset edge, or a bare footprint edge or vertex) that
                // passes within the free space's own boundary hair of it, each with its outward unit normal. The iterate may
                // stand at a VERTEX of F, where the facet the step crosses is not the one it can slide along: every facet
                // there is a candidate face. Each gives its active-set step (the model's minimiser on the face, below,
                // backtracked while J does not fall) and the step that lowers J most, evaluated, is taken. Nothing is probed.
                const Vec exit1 = free_prefix(x, trial);
                const Vec gv = gacc;
                // The face's one-dimensional problem has its own multiplier (0 when the slide fits, else the radius bound
                // c^2 + s^2 <= Delta^2 below), so it uses the model's Hessian H, not H + lambda I of the full-space step.
                const Mat a = H;
                // A facet that is an arc of a clearance ring has its radius (Facet::radius): the face is then the arc itself,
                // and its curvature enters the reduced model below.
                struct Face {
                  Vec n;
                  double radius;
                };
                std::vector<Face> faces;
                for (const auto& f : scope.TangentConeAt(exit1, kHairM).facets)
                  faces.push_back({f.normal, f.radius});
                if (faces.empty() && trial_snapped) faces.push_back({Vec((trial - *snapped).normalized()), 0.0});
                bool have_step = false;
                double best_j = j;
                Vec best_y = x;
                // A slide along a face or a ray of F's cone, backtracked: the model's minimiser along a face is the point the
                // curved level sets of the other terms make J rise at (the second-order change of a heavily weighted residual,
                // kappa s^2 / 2), so a slide that does not lower J is shortened by the trust-region's own factor 1/4
                // (Nocedal & Wright 2006, Algorithm 4.1) until it does, or until it is below the resolution of x.
                const auto backtrack = [&](double slide0, double cap, const auto& make_y, Vec* y_out, double* jy_out) {
                  double sl = slide0;
                  Vec y = x;
                  double jy = j;
                  // A slide that cannot be taken (its end is not walkable, or it does not move) is shortened like one that
                  // does not lower J.
                  for (;;) {
                    if (over_budget()) break;
                    y = make_y(sl);
                    if ((y - x).norm() > 0.0) {
                      jy = eval_errors(y, /*fast=*/false, nullptr).first;
                      if (jy < j) break;
                    }
                    if (!(std::fabs(sl) * 0.25 > xres)) break;
                    sl *= 0.25;
                  }
                  // J along the face is a smooth curve: from the first slide that lowers it, move in the direction in which it
                  // keeps falling, by the trust region's own factors (Algorithm 4.1: the radius is divided by 4 when the model
                  // over-promised and doubled when it under-promised): shorter (the model's curvature along the face was too
                  // small) or longer, up to the face's radius bound `cap` (it was too large: the creeping valley whose Newton
                  // slide is 1e-12 m where 1e-5 m still lowers J), and keep the lowest.
                  if ((y - x).norm() > 0.0 && jy < j) {
                    for (const double factor : {2.0, 0.25}) {
                      for (;;) {
                        if (over_budget()) break;
                        const double sl2 = sl * factor;
                        if (!(std::fabs(sl2) > xres) || std::fabs(sl2) > cap) break;
                        const Vec y2 = make_y(sl2);
                        if (!((y2 - x).norm() > 0.0)) break;
                        const double j2 = eval_errors(y2, /*fast=*/false, nullptr).first;
                        if (!(j2 < jy)) break;
                        sl = sl2;
                        y = y2;
                        jy = j2;
                      }
                    }
                  }
                  *y_out = y;
                  *jy_out = jy;
                };
                // The point a slide starts from is the face's own line pulled INTO F by the rounding of the membership test. An
                // iterate rests where the straight step leaves F: the last point the free space accepts, bisected to its
                // tolerance (kClearanceTol), so a point constructed on the face's line a rounding either side of it is accepted or
                // rejected by the last bit of a distance (the slide then snaps 1e-6 m away, or crawls). Pulled in by nu, twice
                // the rounding of the distance the test evaluates. The test is the orientation tests of the ring (exact for the sign) and then
                // the point-to-polygon distance of Boost.Geometry compared with clearance - kClearanceTol. gamma_8 L is an ESTIMATE of that
                // distance's rounding for operands of magnitude L, not an itemisation of Boost's code (Higham, Accuracy and Stability of
                // Numerical Algorithms, 2nd ed., Lemma 3.1 and sec. 3.1-3.3, gamma_k = k u / (1 - k u)): a point-to-segment distance is two
                // coordinate differences (1 each), two dot products for the foot (gamma_3), the foot and the difference to it (gamma_2), the two
                // squares and their sum (gamma_3) and the root (1), about 8-10 operations of that order. What fixes the size is measurement, and the
                // factor 2 and the order gamma_8 are a POLICY COUNT, not a derivation; like the unresolved-step counts it never certifies anything and
                // a change needs the same measurement: 1e-14 m and 1e-12 m behave alike on the rows that failed, 1e-10 m does not. A constructed face point is accepted by the
                // test and the slide's own length decides. nu is a rounding, not a tolerance: a larger offset (1e-9 m, the free space's
                // hair 1e-6 m) lowers J by more than a creeping valley's gain per step.
                const double face_pull = 2.0 * gamma_k(8) * std::max(1.0, x.cwiseAbs().maxCoeff());
                const auto project = [&](Vec y) {
                  if (!in_f(y)) {
                    // The boundary curves away from its tangent line: the move is the projection of y, and it counts as a
                    // move of this step only when it is walkable (NearestFree of a point across a thin obstacle is the
                    // far side).
                    const auto onto = scope.NearestFree(y);
                    y = x;
                    if (onto && in_f(*onto) && walkable(x, *onto)) y = *onto;
                  } else if (!walkable(x, y)) {
                    y = x;
                  }
                  return y;
                };
                if constexpr (Dim == 2) {
                for (const Face& face : faces) {
                  if (over_budget()) break;  // the candidates so far are judged by J; the best of them is taken
                  const Vec n = face.n;
                  const Vec tang(-n.y(), n.x());
                  const double c = n.dot(exit1 - x);
                  const double along = tang.dot(a * tang);
                  const double gt_eff = tang.dot(gv) + c * tang.dot(a * n);
                  // The face is the curve x(s) = x_face + s t + (kappa s^2 / 2) n (n out of F, kappa = 1 / radius of an arc, 0 for a
                  // straight facet; an arc of a clearance ring curves towards its obstacle). To second order
                  // J(x(s)) = J + 2 g.t s + (t.H.t + kappa g.n) s^2, with g.n the half gradient's normal part at the face (it is
                  // negative where the gradient presses the iterate into the wall, so the wall's curvature then LOWERS the
                  // curvature along the face: on a ring it can cancel it, and a slide that ignores that moves by 1e-6 m a step).
                  const double kappa = face.radius > 0.0 ? 1.0 / face.radius : 0.0;
                  const double g_n = n.dot(gv) + c * n.dot(a * n);
                  const double along_eff = along + kappa * g_n;
                  // The reduced model on the active face is one-dimensional: its trust-region step is the Newton
                  // slide where the reduced curvature along the face is positive and the slide is inside the
                  // region, else the descent direction to the region's boundary (negative curvature is followed).
                  double slide = 0.0;
                  if (n.norm() == 0.0) continue;
                  // The face subproblem has its own radius constraint c^2 + s^2 <= Delta^2 (the point is c n + s t).
                  const double s_max = std::sqrt(std::max(delta * delta - c * c, 0.0));
                  // A gradient along the face not above its rounding uncertainty is zero, and a curvature not below minus the
                  // eigensolver's tolerance is not negative: the slide is then zero (the move to the face is still taken).
                  const bool unresolved = std::fabs(gt_eff) <= g_tol && along_eff > -heig.tol;
                  if (unresolved) slide = 0.0;
                  else if (along_eff > 0.0 && std::fabs(gt_eff / along_eff) <= s_max) slide = -gt_eff / along_eff;
                  else slide = (gt_eff <= 0.0 ? 1.0 : -1.0) * s_max;
                  Vec y;
                  double jy;
                  backtrack(slide, s_max,
                            [&](double sl) {
                              if (face.radius > 0.0) {
                                // The arc itself: centre C = q + radius n beyond the boundary point q = x + c n, moved along it by
                                // arc length sl in the direction of t (the rotation by -sl / radius about C).
                                const Vec q = x + c * n;
                                const Vec cen = q + face.radius * n;
                                const double ang = -sl / face.radius;
                                const Vec v = q - face_pull * n - cen;
                                return project(cen + Vec(v.x() * std::cos(ang) - v.y() * std::sin(ang),
                                                                     v.x() * std::sin(ang) + v.y() * std::cos(ang)));
                              }
                              return project(x + (c - face_pull) * n + sl * tang);
                            },
                            &y, &jy);
                  if ((y - x).norm() <= 0.0) continue;  // a candidate that does not move is not a step
                  if (!have_step || jy < best_j) {
                    have_step = true;
                    best_j = jy;
                    best_y = y;
                  }
                }
                {
                  // The tangent cone of F at the iterate, exactly (TangentConeAt): the directions in which F continues. Its
                  // arcs' end directions are those of the adjacent facets (a half circle on a wall, a quarter at a corner;
                  // the 0 / 2 pi seam of an arc that continues across it is not an end). The projected step along each (the
                  // model's 1-D minimiser along the ray, inside the region, backtracked) is a candidate: at a vertex the
                  // facet the unconstrained step crosses is not the one that descends.
                  const auto cone = scope.TangentConeAt(x, kHairM);
                  if (!cone.Full() && !cone.arcs.empty()) {
                    const double two_pi = 2.0 * M_PI;
                    const bool seam = cone.arcs.front().first <= 0.0 && cone.arcs.back().second >= two_pi && cone.arcs.size() >= 2;
                    std::vector<double> ends;
                    for (size_t k = 0; k < cone.arcs.size(); ++k) {
                      if (!(seam && k == 0)) ends.push_back(cone.arcs[k].first);
                      if (!(seam && k + 1 == cone.arcs.size())) ends.push_back(cone.arcs[k].second);
                    }
                    for (const double th : ends) {
                      const Vec t(std::cos(th), std::sin(th));
                      const double along = t.dot(a * t), gt = t.dot(gv);
                      double slide = 0.0;
                      if (std::fabs(gt) <= g_tol && along > -heig.tol) slide = 0.0;
                      else if (along > 0.0) slide = std::min(delta, std::max(0.0, -gt / along));
                      else if (gt < 0.0) slide = delta;
                      if (!(slide > 0.0)) continue;
                      Vec y;
                      double jy;
                      backtrack(slide, delta, [&](double sl) { return project(x + sl * t); }, &y, &jy);
                      if ((y - x).norm() <= 0.0) continue;
                      if (!have_step || jy < best_j) {
                        have_step = true;
                        best_j = jy;
                        best_y = y;
                      }
                    }
                  }
                }
                } else {
                for (const Face& face : faces) {
                  // More than two dimensions: a face of F is a flat facet (a box face of V; relation_scope.h), and the
                  // reduced model on it lives in its tangent space, of Dim - 1 dimensions: the face's own trust-region
                  // subproblem (the point is c n + s, s in the tangent space, c^2 + |s|^2 <= Delta^2), solved exactly
                  // (nd_trust_region.h), and the slide is backtracked along its direction as in the plane.
                  if (over_budget()) break;
                  const Vec n = face.n;
                  if (n.norm() == 0.0 || face.radius > 0.0) continue;  // (no scope of more than two dimensions has a curved facet)
                  const double c = n.dot(exit1 - x);
                  const Eigen::FullPivLU<Eigen::Matrix<double, 1, Dim>> lu(n.transpose());
                  const Eigen::MatrixXd B = Eigen::HouseholderQR<Eigen::MatrixXd>(lu.kernel())
                                                .householderQ() * Eigen::MatrixXd::Identity(Dim, Dim - 1);
                  const Eigen::VectorXd gr = B.transpose() * Eigen::VectorXd(gv + c * (a * n));
                  const Eigen::MatrixXd Ar0 = B.transpose() * Eigen::MatrixXd(a) * B;
                  const Eigen::MatrixXd Ar = 0.5 * (Ar0 + Ar0.transpose());
                  const double s_max = std::sqrt(std::max(delta * delta - c * c, 0.0));
                  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> are(Ar);
                  const bool unresolved = gr.norm() <= g_tol && are.eigenvalues()[0] > -heig.tol;
                  Vec dir = Vec::Zero();
                  double slide = 0.0;
                  if (!unresolved && s_max > 0.0) {
                    const auto sub = nd_internal::SolveDense(
                        gr, Ar, s_max, g_tol, [&](const Eigen::VectorXd& pa, const Eigen::VectorXd& pb) {
                          return eval_errors(Vec(x + (c - face_pull) * n + B * pa), /*fast=*/false, nullptr).first <=
                                 eval_errors(Vec(x + (c - face_pull) * n + B * pb), /*fast=*/false, nullptr).first;
                        });
                    const Vec sv = B * sub.p;
                    slide = sv.norm();
                    if (slide > 0.0) dir = sv / slide;
                  }
                  Vec y;
                  double jy;
                  backtrack(slide, s_max, [&](double sl) { return project(x + (c - face_pull) * n + sl * dir); }, &y, &jy);
                  if ((y - x).norm() <= 0.0) continue;  // a candidate that does not move is not a step
                  if (!have_step || jy < best_j) {
                    have_step = true;
                    best_j = jy;
                    best_y = y;
                  }
                }
                {
                  // More than two dimensions: the extreme rays of F's tangent cone, each the line where Dim - 1 of its facets
                  // meet (in the plane, the arcs' end directions above), taken in either sense when it lies in the cone. The
                  // projected step along each is a candidate, as in the plane.
                  const auto cone = scope.TangentConeAt(x, kHairM);
                  const auto& fs = cone.facets;
                  std::vector<Vec> rays;
                  const auto in_cone = [&](const Vec& v) {
                    for (const auto& f : fs)
                      if (f.normal.dot(v) > 4.0 * kEps) return false;
                    return true;
                  };
                  std::vector<int> pick(Dim - 1, 0);
                  const int nf = static_cast<int>(fs.size());
                  const std::function<void(int, int)> choose = [&](int k, int from) {
                    if (k == Dim - 1) {
                      Eigen::Matrix<double, Dim - 1, Dim> nm;
                      for (int r = 0; r < Dim - 1; ++r) nm.row(r) = fs[pick[r]].normal.transpose();
                      const Eigen::FullPivLU<Eigen::Matrix<double, Dim - 1, Dim>> lu(nm);
                      if (lu.rank() != Dim - 1) return;
                      const Vec t = Vec(lu.kernel().col(0)).normalized();
                      for (const Vec& cand : {Vec(t), Vec(-t)})
                        if (in_cone(cand)) rays.push_back(cand);
                      return;
                    }
                    for (int q = from; q < nf; ++q) {
                      pick[k] = q;
                      choose(k + 1, q + 1);
                    }
                  };
                  if (!cone.Full()) choose(0, 0);
                  for (const Vec& t : rays) {
                    const double along = t.dot(a * t), gt = t.dot(gv);
                    double slide = 0.0;
                    if (std::fabs(gt) <= g_tol && along > -heig.tol) slide = 0.0;
                    else if (along > 0.0) slide = std::min(delta, std::max(0.0, -gt / along));
                    else if (gt < 0.0) slide = delta;
                    if (!(slide > 0.0)) continue;
                    Vec y;
                    double jy;
                    backtrack(slide, delta, [&](double sl) { return project(x + sl * t); }, &y, &jy);
                    if ((y - x).norm() <= 0.0) continue;
                    if (!have_step || jy < best_j) {
                      have_step = true;
                      best_j = jy;
                      best_y = y;
                    }
                  }
                }
                }
                {
                  // The model's own step, cut where it leaves F, searched along and RETRACTED to the principal level set after
                  // every trial slide (retract above): the feasible part of a direction that is not a facet's. It is the
                  // soft direction of a valley that crosses the wall at a small angle, which the facet slides can only follow
                  // as a staircase (a slide along the wall, a step back off it): the retraction keeps the heavy residual at its
                  // level set, so the valley's own slope sets the step, not its curvature.
                  const double len = (exit1 - x).norm();
                  if (len > 0.0) {
                    const Vec dir = (exit1 - x) / len;
                    Vec y;
                    double jy;
                    backtrack(len, len,
                              [&](double sl) {
                                double jr = 0.0;
                                const Vec yr = chord_retract(project(x + sl * dir), &jr);
                                return in_f(yr) && walkable(x, yr) ? yr : x;
                              },
                              &y, &jy);
                    if ((y - x).norm() > 0.0 && (!have_step || jy < best_j)) {
                      have_step = true;
                      best_j = jy;
                      best_y = y;
                    }
                  }
                }
                if (!have_step) {
                  // No active-set candidate has a step: the step stopped where it leaves F (the constraint may be
                  // inactive: the gradient may point into the interior). Used only then: a candidate that rises
                  // is rejected by the region and retried shorter, while a stub of the free prefix that lowers J by
                  // rounding would end the solve at a point from which a free direction descends (the corner of
                  // ma_meeting_room|ma_tour|permuted, frame 60).
                  const Vec y0 = exit1;
                  if ((y0 - x).norm() > 0.0 && walkable(x, y0)) {
                    best_y = y0;
                    have_step = true;
                  }
                }
                if (have_step) {
                  last_constrained = true;
                  trial = best_y;
                  on_boundary = true;
                  projected = true;
                }
              }
              if (!on_boundary) {
                trial = free_prefix(x, (snapped && in_f(*snapped)) ? *snapped : trial);
              }
            }
            // The model step is a step of the region: the projected step is kept inside it (on the segment from x).
            if (!projected && (trial - x).norm() > delta) trial = x + delta * (trial - x).normalized();
            // ---- Routes that could become the shortest within this step's reach (a tie, or one about to be) ----
            // A route b more than 2 |p| longer than the shortest cannot catch up within a step p (the gradients are unit
            // vectors, so their difference is at most 2): only terms that list another route within that reach are
            // nonsmooth in this step. For those the step is built over the pieces, not from the one the evaluator happens to
            // hold: (1) the region's step of each choice of the routes, as it is and cut where another route overtakes,
            // (2) for a term with e < 0, where squaring the minimum makes J a MAXIMUM over its routes (a valley), the
            // Newton step along the curve where two routes are equal (SQP on d_a = d_b, Nocedal & Wright 2006 sec. 18).
            // Every candidate is judged by J itself and the lowest is taken; the rest of the iteration (the ratio, the
            // radius) is unchanged.
            bool have_pred = false;
            double pred_value = 0.0;
            if (!projected && step.norm() > 0.0) {
              const double reach = 2.0 * step.norm();
              struct TieTerm {
                size_t i;
                std::vector<Route> pc;
                std::vector<size_t> act;  // the pieces co-active with the shortest (length intervals overlap); always holds 0
              };
              std::vector<TieTerm> ties;
              for (size_t i = 0; i < terms.size(); ++i) {
                if (!std::isfinite(e0[i])) continue;
                auto pc = pieces_of(eff[i], x, reach);
                if (pc.size() < 2) continue;
                std::vector<size_t> act = {0};
                for (size_t a = 1; a < pc.size(); ++a)
                  if (is_coactive(pc[a], pc[0])) act.push_back(a);
                ties.push_back({i, std::move(pc), std::move(act)});
              }
              if (!ties.empty()) {
                struct Cand {
                  Vec y;
                  double jy, pred;
                };
                std::optional<Cand> best;
                const auto consider = [&](const Vec& y, double pred) {
                  if (!y.allFinite() || !((y - x).norm() > 0.0) || !in_f(y) || !walkable(x, y)) return;
                  const double jy = eval_errors(y, /*fast=*/false, nullptr).first;
                  if (!best || jy < best->jy) best = Cand{y, jy, pred};
                };
                struct Mod {
                  Vec G = Vec::Zero();
                  Mat K = Mat::Zero(), Nn = Mat::Zero();
                  double pred(const Vec& pp) const {
                    return -2.0 * G.dot(pp) - scope_pursuit_internal::QuadForm<Dim>(K, pp);
                  }
                };
                const auto model_of = [&](const std::vector<size_t>& choice) {
                  Mod m;
                  for (size_t i = 0; i < terms.size(); ++i) {
                    if (!std::isfinite(e0[i])) continue;
                    Vec gi = eff[i].g;
                    Mat hi = eff[i].h;
                    for (size_t t = 0; t < ties.size(); ++t)
                      if (ties[t].i == i) {
                        const auto& pcs = ties[t].pc[ties[t].act[choice[t]]];
                        gi = unit_of(pcs, x);
                        hi = piece_hess(pcs, x);
                      }
                    const double w = std::max(terms[i].weight, 0.0);
                    m.G += w * e0[i] * gi;
                    for (int r = 0; r < Dim; ++r)
                      for (int q = r; q < Dim; ++q) {
                        m.K(r, q) += w * gi[r] * gi[q] + w * e0[i] * hi(r, q);
                        m.Nn(r, q) += w * gi[r] * gi[q];
                      }
                  }
                  for (int r = 0; r < Dim; ++r)
                    for (int q = 0; q < r; ++q) {
                      m.K(r, q) = m.K(q, r);
                      m.Nn(r, q) = m.Nn(q, r);
                    }
                  return m;
                };
                {  // the trial as it stands, judged with the unconstrained model's own prediction
                  const Vec p0 = trial - x;
                  consider(trial, TrModelReduction<Dim>(gacc, H, p0));
                }
                // (1) every choice among the CO-ACTIVE routes of the tie terms (a product over the terms that have more than one: the
                // length intervals of the pieces overlap, so J there is the minimum over them and each choice is a model of it). A
                // route that is within reach of the step but not co-active is no choice: it enters as the point where the chosen
                // route is overtaken (the clip by FirstCrossing below) and, for a term with e < 0, as the SQP step onto its tie curve.
                std::vector<size_t> choice(ties.size(), 0);
                for (;;) {
                  if (over_budget()) break;
                  const Mod m = model_of(choice);
                  const TrOut ts = tr_solve(m.G, m.K, m.Nn, static_cast<int>(terms.size()), delta, g_tol);
                  if (ts.p.norm() > 0.0) {
                    consider(x + ts.p, m.pred(ts.p));
                    double tc = 2.0;
                    for (size_t t = 0; t < ties.size(); ++t)
                      for (size_t b = 0; b < ties[t].pc.size(); ++b)
                        if (b != ties[t].act[choice[t]]) tc = std::min(tc, FirstCrossing(x, ts.p, ties[t].pc[ties[t].act[choice[t]]], ties[t].pc[b]));
                    if (tc > 0.0 && tc < 1.0) consider(x + tc * ts.p, m.pred(tc * ts.p));
                  }
                  size_t k = 0;
                  for (; k < ties.size(); ++k) {
                    if (++choice[k] < ties[k].act.size()) break;
                    choice[k] = 0;
                  }
                  if (k == ties.size()) break;
                }
                // (2) the Newton step along the curve where two routes of a term with e < 0 are equal. In the plane the tie set is a
                // curve with one tangent; in more dimensions it is a surface, whose step is not formed here (a scope of more than
                // two dimensions lists no alternative routes: relation_scope.h), so the candidate is the plane's only.
                if constexpr (Dim == 2) {
                  const std::vector<size_t> primary(ties.size(), 0);
                  const Mod m = model_of(primary);
                  for (size_t t = 0; t < ties.size(); ++t) {
                    if (!(e0[ties[t].i] < 0.0)) continue;
                    for (size_t b = 1; b < ties[t].pc.size(); ++b) {
                      const auto& pa = ties[t].pc[0];
                      const auto& pb = ties[t].pc[b];
                      const Vec ga = unit_of(pa, x), gb = unit_of(pb, x);
                      const Mat ha = piece_hess(pa, x), hb = piece_hess(pb, x);
                      const Vec gc = ga - gb;
                      const double gc2 = gc.squaredNorm();
                      if (!(gc2 > 0.0)) continue;
                      const double c0 = pa.value - pb.value;
                      const double lam = -m.G.dot(gc) / gc2;
                      Mat W = m.K;
                      W(0, 0) = m.K(0, 1) * 0.0 + m.K(0, 0);
                      W += lam * (ha - hb);
                      const Vec pr = -c0 * gc / gc2;
                      const Vec tg(-gc.y(), gc.x());
                      const Vec tu = tg / tg.norm();
                      const double tgrad = tu.dot(m.G), twt = tu.dot(W * tu), twp = tu.dot(W * pr);
                      double sl = 0.0;
                      if (twt > 0.0) sl = -(tgrad + twp) / twt;
                      else sl = ((tgrad + twp) <= 0.0 ? 1.0 : -1.0) * delta;
                      Vec pm = pr + sl * tu;
                      if (pr.norm() >= delta) {
                        pm = pr * (delta / pr.norm());
                      } else if (pm.norm() > delta) {
                        const double sm = std::sqrt(std::max(0.0, delta * delta - pr.squaredNorm()));
                        pm = pr + (sl >= 0.0 ? sm : -sm) * tu;
                      }
                      consider(x + pm, m.pred(pm));
                    }
                  }
                }
                if (best && best->jy < j) {
                  trial = best->y;
                  have_pred = true;
                  pred_value = best->pred;
                }
              }
            }
            TrTrial<Dim> out;
            out.trial = trial;
            out.projected = projected;
            out.have_pred = have_pred;
            out.pred = pred_value;
            return out;
        };
        const auto eval_j = [&](const Vec& y) { return eval_errors(y, /*fast=*/false, nullptr); };
        const auto retract_hook = [&](Vec* trial_io, double* jt_io) {
              Vec& trial = *trial_io;
              double& jt = *jt_io;
              double jr = 0.0;
              const Vec yr = chord_retract(trial, &jr);
              if ((yr - trial).norm() > 0.0 && in_f(yr) && walkable(x, yr) && jr < jt) {
                trial = yr;
                jt = jr;
                return true;
              }
              return false;
        };
        auto hooks = MakeTrLoopHooks(project_iterate, note_incumbent, over_budget, assemble, g_u3_fn, chi_descent,
                                     subproblem, make_trial, eval_j, retract_hook);
        TrLoopStop loop_stop = TrLoopStop::kNone;
        bool loop_not_converged = false;
        TrLoopVars<Dim> vars{x, j, delta, delta_max, last_dj, last_dx, last_constrained, chi_unresolved_streak,
                           normal_unresolved_streak, gn_iters, stationary, stopped, loop_stop, loop_not_converged,
                           result.budget_hit_iteration};
        RunTrustRegionLoop<Dim>(vars, hooks, in.gn_safety_cap);
        switch (loop_stop) {
          case TrLoopStop::kNone: break;
          case TrLoopStop::kNoDampedStep: result.gn_stop = SolverResult::GnStop::kNoDampedStep; break;
          case TrLoopStop::kNoDampedStepOnBoundary: result.gn_stop = SolverResult::GnStop::kNoDampedStepOnBoundary; break;
          case TrLoopStop::kPrecision: result.gn_stop = SolverResult::GnStop::kPrecision; break;
          // U3 held to J's accuracy by a refuted chi descent: reported as the precision stop it is (no separate public status).
          case TrLoopStop::kPrecisionChiRefuted: result.gn_stop = SolverResult::GnStop::kPrecision; break;
        }
        if (loop_not_converged) result.status = SolverStatus::kNotConverged;
      }
      if (budget_tripped) {
        // The budget ran out (at the loop boundary or inside an inner loop): the answer is the incumbent, however the rest of
        // the iteration ended. A solve whose stop rule was met at the same moment is still a budget hit (conservative).
        note_incumbent();
        x = best_x;
        j = best_feasible ? best_j : j_sample;  // no iterate in F: the sample, with its own J
        result.incumbent_objective = j;
        incumbent_outside_f = !best_feasible;
        result.budget_hit = true;
        result.budget_incumbent_is_warm_start = best_is_warm;
        stationary = false;
        stopped = false;
      }
      result.position = x;
      result.objective_stationary = stationary;
      result.iterations = gn_iters;
      result.objective_iterations = gn_iters;
      result.objective_before += j_sample;
      result.objective_after += j;
      result.objective_relation_count += n_sample;
      if (result.budget_hit) {
        result.status = SolverStatus::kBudgetHit;
        result.gn_stop = SolverResult::GnStop::kBudget;
        result.objective_exact = false;
      } else if (!stopped) {
        // The safety cap fired. This point is not a minimiser.
        result.status = SolverStatus::kNotConverged;
        result.gn_stop = SolverResult::GnStop::kCap;
        result.objective_exact = false;
      } else {
        double max_abs = 0.0;
        int n_final = 0;
        std::vector<double> e_final;
        eval_errors(x, /*fast=*/false, &e_final);
        for (double e : e_final) {
          if (!std::isfinite(e)) continue;
          max_abs = std::max(max_abs, std::fabs(e));
          ++n_final;
        }
        result.objective_exact =
            n_final > 0 && max_abs <= in.exact_eps_m;
        result.objective_fallback = false;
      }
      if (result.feasibility != FeasibilityOutcome::kProjected &&
          result.feasibility != FeasibilityOutcome::kNoFreeSpace &&
          result.feasibility != FeasibilityOutcome::kSearchFoundNone) {
        result.feasibility = incumbent_outside_f ? FeasibilityOutcome::kSampleOutside
                             : scope.Bounded()   ? FeasibilityOutcome::kFeasible
                                                 : FeasibilityOutcome::kUnconstrained;
      }

}

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_SCOPE_PURSUIT_H_
