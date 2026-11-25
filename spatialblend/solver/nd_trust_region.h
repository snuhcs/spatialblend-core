// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/nd_trust_region.h
//
// The shared step engine of the relation solve, generalized to N dimensions (the state dimension follows the
// subject). One solve is the trust-region Newton method on
// J = sum_i w_i (d_i(x) - d*_i)^2 (Nocedal & Wright, Numerical Optimization, 2nd ed. 2006, Algorithm 4.1), and its
// subproblem
//     min_p  g.p + p.A.p / 2,   |p| <= Delta
// is solved exactly from the eigendecomposition of A (N&W sec. 4.3; More & Sorensen 1983), over a Gauss-Newton matrix
// N = J^T W J that may be rank deficient. The state is x in R^Dim: the planar root position (Dim = 2) and a body
// anchor's position (Dim = 3) are two instances of the same algorithm.
//
// Dim = 2 is the closed form the root solve has always used (SymEig2): its arithmetic is unchanged, so the root's
// answers are bit-identical to the solve before the generalization (relation_solver_parity_test). Every other Dim uses
// the same algorithm through Eigen's self-adjoint eigendecomposition; for Dim = 2 that general path agrees with the
// closed form to the eigensolver's rounding (nd_trust_region_test), which is what makes the closed form the 2-D
// instance and not a separate solver.
#ifndef SPATIALBLEND_SPACE_ND_TRUST_REGION_H_
#define SPATIALBLEND_SPACE_ND_TRUST_REGION_H_

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <limits>

namespace spatialblend {

// Unit roundoff u = 2^-53 and gamma_k = k u / (1 - k u) (Higham, Accuracy and Stability of Numerical Algorithms,
// 2nd ed., Lemma 3.1).
constexpr double kTrUnitRoundoff = 1.1102230246251565e-16;
inline double TrGamma(double k) { return k * kTrUnitRoundoff / (1.0 - k * kTrUnitRoundoff); }

// The eigen-decomposition of the symmetric 2 x 2 matrix [[a, b], [b, d]] in closed form: with t = a + d,
// s = sqrt((a - d)^2 + 4 b^2), lambda = (t -+ s) / 2, the principal axis at angle atan2(2 b, a - d) / 2. Each lambda is
// computed with an absolute error of at most gamma_8 (|a| + |d| + 2 |b|) (t and a - d carry one rounding each, the three
// squares and the sum of non-negative terms add gamma_3, the root halves the relative error and adds one rounding,
// the final sum and halving two more; Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., sec. 3.1/4.2),
// whatever the sign of the entries. That bound is also the resolution at which two eigenvalues, or an eigenvalue and
// zero, are distinguished: SymEig::tol.
struct SymEig {
  double lo, hi;
  Eigen::Vector2d vlo, vhi;
  double tol;
};
inline SymEig SymEig2(double a, double b, double d) {
  constexpr double u = kTrUnitRoundoff;
  SymEig r;
  const double t = a + d, df = a - d;
  const double s = std::sqrt(df * df + 4.0 * b * b);
  r.hi = 0.5 * (t + s);
  r.lo = 0.5 * (t - s);
  const double th = 0.5 * std::atan2(2.0 * b, df);
  r.vhi = Eigen::Vector2d(std::cos(th), std::sin(th));
  r.vlo = Eigen::Vector2d(-std::sin(th), std::cos(th));
  r.tol = (8.0 * u / (1.0 - 8.0 * u)) * (std::fabs(a) + std::fabs(d) + 2.0 * std::fabs(b));
  return r;
}

template <int Dim>
struct TrStep {
  Eigen::Matrix<double, Dim, 1> p;  // sized and zeroed by the solver
  bool radius_active = false;
  bool rank_def = false;
};

namespace nd_internal {

// The exact subproblem over a dense symmetric A (any dimension), full rank in the Gauss-Newton sense: the secular
// equation |p(lambda)| = Delta, p(lambda) = -sum_k c_k / (m_k + lambda) v_k over the eigenpairs (m_k, v_k) of A, with
// the hard case (g orthogonal to the eigenvector of the smallest eigenvalue). Components of g along an eigenvector not
// above `gtol` are zero (see SolveTrSubproblem). `pick(pa, pb)` chooses between the two completions of the hard case
// (same model value; the caller's J judges).
inline TrStep<Eigen::Dynamic> SolveDense(const Eigen::VectorXd& g, const Eigen::MatrixXd& A, double dlt, double gtol,
                                         const std::function<bool(const Eigen::VectorXd&, const Eigen::VectorXd&)>& pick) {
  TrStep<Eigen::Dynamic> out;
  const int n = static_cast<int>(g.size());
  out.p = Eigen::VectorXd::Zero(n);
  if (n == 0) return out;
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
  const Eigen::VectorXd m = es.eigenvalues();  // ascending
  const Eigen::MatrixXd V = es.eigenvectors();
  Eigen::VectorXd c = V.transpose() * g;
  for (int k = 0; k < n; ++k)
    if (std::fabs(c[k]) <= gtol) c[k] = 0.0;
  const auto p_of = [&](double lam) {
    Eigen::VectorXd pp = Eigen::VectorXd::Zero(n);
    for (int k = 0; k < n; ++k)
      if (m[k] + lam > 0.0) pp -= (c[k] / (m[k] + lam)) * V.col(k);
    return pp;
  };
  if (m[0] > 0.0 && p_of(0.0).norm() <= dlt) {
    out.p = p_of(0.0);
    return out;
  }
  const double lam_lo = std::max(0.0, -m[0]);
  // The hard case: every component along the smallest eigenvalue's eigenvectors is 0 and the rest is inside.
  bool hard = m[0] <= 0.0;
  double rest = 0.0;
  for (int k = 0; k < n && hard; ++k) {
    if (m[k] + lam_lo > 0.0) {
      rest += (c[k] / (m[k] + lam_lo)) * (c[k] / (m[k] + lam_lo));
    } else if (c[k] != 0.0) {
      hard = false;
    }
  }
  if (hard && rest < dlt * dlt) {
    Eigen::VectorXd base = Eigen::VectorXd::Zero(n);
    for (int k = 0; k < n; ++k)
      if (m[k] + lam_lo > 0.0) base -= (c[k] / (m[k] + lam_lo)) * V.col(k);
    const double tau = std::sqrt(dlt * dlt - rest);
    const Eigen::VectorXd pa = base + tau * V.col(0), pb = base - tau * V.col(0);
    out.p = pick(pa, pb) ? pa : pb;
  } else {
    const double gn = c.norm();
    double lo = lam_lo, hi = std::max(lam_lo, gn / dlt - m[0]);
    bool root_bracketed = false;  // some representable lambda > -m0 has |p(lambda)| > Delta
    for (int it = 0; it < 200; ++it) {
      const double mid = 0.5 * (lo + hi);
      if (!(mid > lo && mid < hi)) break;
      const bool feasible_shift = m[0] + mid > 0.0;
      if (p_of(mid).norm() > dlt || !feasible_shift) {
        lo = mid;
        if (feasible_shift) root_bracketed = true;
      } else {
        hi = mid;
      }
    }
    out.p = p_of(hi);
    // With m0 <= 0 and no representable lambda > -m0 at which |p(lambda)| exceeds Delta (the component along the
    // smallest eigenvector is too small for the secular equation to have a root the arithmetic can name), the problem
    // IS the hard case to this precision: the step is completed along that eigenvector to the boundary, the two
    // completions judged by J.
    // The same holds when the root lies between two adjacent doubles (the bracket closed with |p(hi)| < Delta < |p(lo)|):
    // the boundary point is then the completion on the side the gradient's component sets (-sign c0).
    if (m[0] <= 0.0 && out.p.squaredNorm() < dlt * dlt) {
      const Eigen::VectorXd base = out.p - out.p.dot(V.col(0)) * V.col(0);
      const double tau = std::sqrt(std::max(0.0, dlt * dlt - base.squaredNorm()));
      const Eigen::VectorXd pa = base + tau * V.col(0), pb = base - tau * V.col(0);
      if (root_bracketed && c[0] != 0.0)
        out.p = c[0] < 0.0 ? pa : pb;
      else
        out.p = pick(pa, pb) ? pa : pb;
    }
  }
  out.radius_active = true;
  return out;
}

// The eigenvalue resolution of a dense symmetric matrix.
//   n = 2: DERIVED. SymEig2's closed form computes each eigenvalue with an absolute error of at most
//          gamma_8 (|a| + |d| + 2 |b|) = gamma_8 sum |A_ij| (see SymEig2).
//   n > 2: NOT DERIVED. The same expression gamma_8 sum |A_ij| is used unchanged as a stated choice. Eigen's
//          SelfAdjointEigenSolver (Householder tridiagonalization + implicit symmetric QR) is backward stable, its
//          eigenvalues within c(n) u |A|_2 of the exact ones (Golub & Van Loan, Matrix Computations, sec. 8.3), but
//          c(n) for this implementation is not stated anywhere this code can cite, so no bound is claimed for n > 2.
inline double DenseEigTol(const Eigen::MatrixXd& A) { return TrGamma(8.0) * A.cwiseAbs().sum(); }

}  // namespace nd_internal

// The trust-region subproblem of one iteration. `g` is half the gradient of J, `A` half its Hessian (the exact one:
// Gauss-Newton plus the residuals' curvature), `N` the Gauss-Newton part, `nterms` the number of terms N was summed over
// (its rounding is gamma_{nterms + 13} tr(N)), `dlt` the radius, `gtol` the rounding uncertainty of g. Rank: an
// eigenvalue of N not above its rounding is zero, and g has no component along its eigenvector (g lies in the range of
// J^T); the step is then the exact subproblem restricted to the range, and a null direction is followed only along
// negative curvature that A's own eigenvalue resolution resolves.
template <int Dim>
TrStep<Dim> SolveTrSubproblem(
    const Eigen::Matrix<double, Dim, 1>& gv_in, const Eigen::Matrix<double, Dim, Dim>& A,
    const Eigen::Matrix<double, Dim, Dim>& N, int nterms, double dlt, double gtol,
    const std::function<bool(const Eigen::Matrix<double, Dim, 1>&, const Eigen::Matrix<double, Dim, 1>&)>& pick) {
  using Vec = Eigen::Matrix<double, Dim, 1>;
  TrStep<Dim> out;
  const int n = static_cast<int>(gv_in.size());
  out.p = Vec::Zero(n);
  const auto model = [&](const Vec& pp) { return gv_in.dot(pp) + 0.5 * pp.dot(A * pp); };
  const auto pick_dyn = [&](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    return pick(Vec(a), Vec(b));
  };
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ne{Eigen::MatrixXd(N)};
  const double trn = N.trace();
  const double tau_n = nd_internal::DenseEigTol(N) + TrGamma(nterms + 13) * trn;
  int null_count = 0;
  for (int k = 0; k < n; ++k)
    if (ne.eigenvalues()[k] <= tau_n) ++null_count;
  if (trn > 0.0 && null_count > 0) {
    out.rank_def = true;
    const int r = n - null_count;
    const Eigen::MatrixXd U = ne.eigenvectors().rightCols(r);    // the range (ascending order: the largest last)
    const Eigen::MatrixXd Z = ne.eigenvectors().leftCols(null_count);
    // The null space carries no first-order information (g lies in the range of J^T), only A's curvature. The part of
    // it the subproblem follows is the NEGATIVE-curvature eigenspace of the reduced Hessian Z^T A Z (eigenvalues below
    // A's own resolution): it is a property of the null SPACE, not of the basis Z the eigensolver of N happened to
    // return. The step is then the EXACT subproblem (SolveDense: the secular equation and More & Sorensen's hard case)
    // on S = range(U) + that eigenspace, with A restricted to S; in the hard case the two completions along the
    // eigenvector are judged by J (`pick`). With no negative curvature there S is the range.
    const double atol = nd_internal::DenseEigTol(Eigen::MatrixXd(A));
    const Eigen::MatrixXd Az = Z.transpose() * Eigen::MatrixXd(A) * Z;
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ze{0.5 * (Az + Az.transpose())};
    int neg = 0;
    while (neg < null_count && ze.eigenvalues()[neg] < -atol) ++neg;
    Eigen::MatrixXd Q(n, r + neg);
    Q.leftCols(r) = U;
    if (neg > 0) Q.rightCols(neg) = Z * ze.eigenvectors().leftCols(neg);
    if (r + neg > 0) {
      const Eigen::VectorXd gq = Q.transpose() * Eigen::VectorXd(gv_in);
      const Eigen::MatrixXd Aq0 = Q.transpose() * Eigen::MatrixXd(A) * Q;
      const Eigen::MatrixXd Aq = 0.5 * (Aq0 + Aq0.transpose());
      const auto sub = nd_internal::SolveDense(
          gq, Aq, dlt, gtol, [&](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
            return pick_dyn(Q * a, Q * b);
          });
      out.p = Vec(Q * sub.p);
      out.radius_active = sub.radius_active;
    }
    (void)model;
    return out;
  }
  const auto full = nd_internal::SolveDense(Eigen::VectorXd(gv_in), Eigen::MatrixXd(A), dlt, gtol, pick_dyn);
  out.p = Vec(full.p);
  out.radius_active = full.radius_active;
  return out;
}

// Dim = 2: the closed form (unchanged arithmetic of the root solve; see the header comment).
template <>
inline TrStep<2> SolveTrSubproblem<2>(const Eigen::Vector2d& gv_in, const Eigen::Matrix2d& A, const Eigen::Matrix2d& N,
                                      int nterms, double dlt, double gtol,
                                      const std::function<bool(const Eigen::Vector2d&, const Eigen::Vector2d&)>& pick) {
  const double a00 = A(0, 0), a01 = A(0, 1), a11 = A(1, 1);
  const double m00 = N(0, 0), m01 = N(0, 1), m11 = N(1, 1);
  TrStep<2> out;
  out.p = Eigen::Vector2d::Zero();
  Eigen::Vector2d gv = gv_in;
  const SymEig he = SymEig2(a00, a01, a11);
  const SymEig ne = SymEig2(m00, m01, m11);
  const double trn = m00 + m11;
  const double tau_n = ne.tol + TrGamma(nterms + 13) * trn;
  if (trn > 0.0 && ne.lo <= tau_n) {
    out.rank_def = true;
    Eigen::Vector2d u1 = ne.hi > tau_n ? ne.vhi : Eigen::Vector2d(Eigen::Vector2d::Zero());
    const Eigen::Vector2d u0 = ne.hi > tau_n ? ne.vlo : ne.vhi;
    const auto model = [&](const Eigen::Vector2d& pp) {
      return gv.dot(pp) + 0.5 * (a00 * pp.x() * pp.x() + 2.0 * a01 * pp.x() * pp.y() + a11 * pp.y() * pp.y());
    };
    Eigen::Vector2d best = Eigen::Vector2d::Zero();
    if (u1.norm() > 0.0) {
      double gg = u1.dot(gv);
      if (std::fabs(gg) <= gtol) gg = 0.0;
      const double hh = a00 * u1.x() * u1.x() + 2.0 * a01 * u1.x() * u1.y() + a11 * u1.y() * u1.y();
      if (hh > 0.0 && std::fabs(gg) / hh <= dlt) {
        best = -(gg / hh) * u1;
      } else if (hh > 0.0) {
        best = -(gg >= 0.0 ? 1.0 : -1.0) * dlt * u1;
        out.radius_active = true;
      } else if (gg != 0.0 || hh < 0.0) {
        best = (gg <= 0.0 ? 1.0 : -1.0) * dlt * u1;
        out.radius_active = true;
      }
    }
    const double hn = a00 * u0.x() * u0.x() + 2.0 * a01 * u0.x() * u0.y() + a11 * u0.y() * u0.y();
    if (hn < -he.tol) {
      const double gu0 = std::fabs(u0.dot(gv)) <= gtol ? 0.0 : u0.dot(gv);
      const Eigen::Vector2d pn = (gu0 <= 0.0 ? 1.0 : -1.0) * dlt * u0;
      if (model(pn) < model(best)) {
        best = pn;
        out.radius_active = true;
      }
    }
    out.p = best;
    return out;
  }
  const double m0 = he.lo, m1 = he.hi;
  const Eigen::Vector2d v0 = he.vlo, v1 = he.vhi;
  double c0 = v0.dot(gv), c1 = v1.dot(gv);
  if (std::fabs(c0) <= gtol) c0 = 0.0;
  if (std::fabs(c1) <= gtol) c1 = 0.0;
  const double gn = std::hypot(c0, c1);
  const auto p_of = [&](double lam) {
    Eigen::Vector2d pp = Eigen::Vector2d::Zero();
    if (m0 + lam > 0.0) pp -= (c0 / (m0 + lam)) * v0;
    if (m1 + lam > 0.0) pp -= (c1 / (m1 + lam)) * v1;
    return pp;
  };
  if (m0 > 0.0 && p_of(0.0).norm() <= dlt) {
    out.p = p_of(0.0);  // the Newton step is inside the region
    return out;
  }
  const double lam_lo = std::max(0.0, -m0);
  const double rest = (m1 + lam_lo > 0.0) ? (c1 / (m1 + lam_lo)) * (c1 / (m1 + lam_lo))
                                          : std::numeric_limits<double>::infinity();
  if (m0 <= 0.0 && c0 == 0.0 && rest < dlt * dlt) {
    // The hard case (More & Sorensen 1983; N&W sec. 4.3): g is orthogonal to the eigenvector of the smallest
    // eigenvalue, lambda = -m0, and the step is completed along that eigenvector to the boundary. Either sign
    // gives the same model value; the one whose J is lower is taken.
    const Eigen::Vector2d base = (m1 + lam_lo > 0.0) ? Eigen::Vector2d(-(c1 / (m1 + lam_lo)) * v1)
                                                     : Eigen::Vector2d(Eigen::Vector2d::Zero());
    const double tau = std::sqrt(dlt * dlt - rest);
    const Eigen::Vector2d pa = base + tau * v0, pb = base - tau * v0;
    out.p = pick(pa, pb) ? pa : pb;
  } else {
    // |p(lambda)| = Delta for lambda > max(0, -m0): |p| is decreasing, bracketed by lambda <= |g| / Delta - m0.
    double lo = lam_lo, hi = std::max(lam_lo, gn / dlt - m0);
    for (int it = 0; it < 200; ++it) {
      const double mid = 0.5 * (lo + hi);
      if (!(mid > lo && mid < hi)) break;
      if (p_of(mid).norm() > dlt || !(m0 + mid > 0.0)) lo = mid; else hi = mid;
    }
    out.p = p_of(hi);
  }
  out.radius_active = true;
  return out;
}

// The general path at any Dim, also for Dim = 2 (tests compare it with the closed form above).
template <int Dim>
TrStep<Dim> SolveTrSubproblemGeneral(
    const Eigen::Matrix<double, Dim, 1>& g, const Eigen::Matrix<double, Dim, Dim>& A,
    const Eigen::Matrix<double, Dim, Dim>& N, int nterms, double dlt, double gtol,
    const std::function<bool(const Eigen::Matrix<double, Dim, 1>&, const Eigen::Matrix<double, Dim, 1>&)>& pick) {
  // The dynamic-size instantiation is the general body; it is not specialized.
  const Eigen::VectorXd gd = g;
  const Eigen::MatrixXd Ad = A, Nd = N;
  const auto r = SolveTrSubproblem<Eigen::Dynamic>(
      gd, Ad, Nd, nterms, dlt, gtol,
      [&](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        return pick(Eigen::Matrix<double, Dim, 1>(a), Eigen::Matrix<double, Dim, 1>(b));
      });
  TrStep<Dim> out;
  out.p = r.p;
  out.radius_active = r.radius_active;
  out.rank_def = r.rank_def;
  return out;
}

// Algorithm 4.1's radius update (N&W 2006): rho < 1/4: Delta / 4; rho > 3/4 and the step on the boundary:
// min(2 Delta, Delta_max); else unchanged.
inline double TrRadiusUpdate(double rho, bool on_boundary, double delta, double delta_max) {
  if (rho < 0.25) return delta / 4.0;
  if (rho > 0.75 && on_boundary) return std::min(2.0 * delta, delta_max);
  return delta;
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_ND_TRUST_REGION_H_
