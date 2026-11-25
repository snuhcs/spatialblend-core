// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// The shared step engine in N dimensions (nd_trust_region.h) and the pursuit in a scope of any dimension (scope_pursuit.h).
#include "spatialblend/solver/nd_trust_region.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <random>

#include "spatialblend/solver/scope_pursuit.h"

namespace spatialblend {
namespace {

// A J-free tie break: both completions have the same model value; the test picks the first.
template <int Dim>
bool First(const Eigen::Matrix<double, Dim, 1>&, const Eigen::Matrix<double, Dim, 1>&) {
  return true;
}

// The brute-force minimum of the model g.p + p.A.p / 2 over the ball |p| <= dlt restricted to span(Q) (dense sampling
// of the ball, then a local polish): an oracle independent of the eigendecompositions.
template <int Dim>
double BruteMin(const Eigen::Matrix<double, Dim, 1>& g, const Eigen::Matrix<double, Dim, Dim>& A, double dlt,
                const Eigen::MatrixXd& Q) {
  using Vec = Eigen::Matrix<double, Dim, 1>;
  const int k = static_cast<int>(Q.cols());
  const auto model = [&](const Eigen::VectorXd& c) {
    const Vec p = Vec(Q * c);
    return g.dot(p) + 0.5 * p.dot(A * p);
  };
  std::mt19937 rng(11);
  std::normal_distribution<double> N01;
  std::uniform_real_distribution<double> U01(0.0, 1.0);
  double best = 0.0;  // p = 0
  Eigen::VectorXd bc = Eigen::VectorXd::Zero(k);
  for (int s = 0; s < 40000; ++s) {
    Eigen::VectorXd c(k);
    for (int i = 0; i < k; ++i) c[i] = N01(rng);
    c *= dlt * std::pow(U01(rng), 1.0 / k) / c.norm();
    if (s % 2) c *= dlt / c.norm();  // half on the sphere
    const double v = model(c);
    if (v < best) { best = v; bc = c; }
  }
  for (double h = 0.05 * dlt; h > 1e-12; h *= 0.5)  // coordinate polish inside the ball
    for (int it = 0; it < 50; ++it) {
      bool moved = false;
      for (int i = 0; i < k; ++i)
        for (const double sgn : {1.0, -1.0}) {
          Eigen::VectorXd c = bc;
          c[i] += sgn * h;
          if (c.norm() > dlt) c *= dlt / c.norm();
          const double v = model(c);
          if (v < best) { best = v; bc = c; moved = true; }
        }
      if (!moved) break;
    }
  return best;
}

// Random 2-D problems (indefinite, rank-deficient Gauss-Newton matrices, steps inside and on the radius). With N of
// full rank, or rank deficient with no negative curvature across its null space, the 2-D closed form IS the general
// algorithm at Dim = 2 (the same step to the eigensolvers' rounding, up to the +-p tie of a zero gradient component).
// Rank deficient WITH negative curvature across the null space, they differ: the general algorithm solves the exact
// subproblem on range + the negative null eigenspace (the whole plane in 2-D), the root's closed form takes the better
// of the range step and the null direction at full radius. The general step is then the brute-force minimum, never
// worse than the closed form's (the closed form is kept for the root: root parity is bit-identical).
TEST(NdTrustRegion, TheTwoDimensionalClosedFormAndTheGeneralAlgorithm) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  int same = 0, rank_def = 0, on_radius = 0, ties = 0, joint = 0;
  for (int k = 0; k < 2000; ++k) {
    // N = sum of 1..3 rank-one unit-gradient terms; A = N + residual curvature.
    const int terms = 1 + k % 3;
    Eigen::Matrix2d N = Eigen::Matrix2d::Zero();
    Eigen::Vector2d g = Eigen::Vector2d::Zero();
    for (int t = 0; t < terms; ++t) {
      const double gx = U(rng);  // drawn in order (argument evaluation order is unspecified)
      const double gy = U(rng);
      Eigen::Vector2d gi(gx, gy);
      gi.normalize();
      const double w = 0.5 + std::fabs(U(rng)), e = U(rng);
      N += w * gi * gi.transpose();
      g += w * e * gi;
    }
    Eigen::Matrix2d C;
    C << U(rng), U(rng), 0, U(rng);
    C(1, 0) = C(0, 1);
    const Eigen::Matrix2d A = N + 0.3 * C;
    const double dlt = 0.05 + 2.0 * std::fabs(U(rng));
    const auto a = SolveTrSubproblem<2>(g, A, N, terms, dlt, 0.0, First<2>);
    const auto b = SolveTrSubproblemGeneral<2>(g, A, N, terms, dlt, 0.0, First<2>);
    EXPECT_EQ(a.rank_def, b.rank_def) << k;
    if (a.rank_def != b.rank_def) continue;
    const auto model = [&](const Eigen::Vector2d& p) { return g.dot(p) + 0.5 * p.dot(A * p); };
    bool neg_null = false;
    if (a.rank_def) {
      const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> ne(N);
      const Eigen::Vector2d z = ne.eigenvectors().col(0);
      neg_null = z.dot(A * z) < -TrGamma(8.0) * A.cwiseAbs().sum();
    }
    if (neg_null) {
      ++joint;
      EXPECT_LE(model(b.p), model(a.p) + 1e-12 * (1.0 + std::fabs(model(a.p)))) << k;
      // At least as low as the brute-force search (whose own resolution is ~1e-6 here), and as low as it to that.
      const double brute = BruteMin<2>(g, A, dlt, Eigen::MatrixXd::Identity(2, 2));
      EXPECT_LE(model(b.p), brute + 1e-12) << k;
      EXPECT_NEAR(model(b.p), brute, 1e-6 * (1.0 + dlt)) << k;
    } else if ((a.p - b.p).norm() > 1e-9 * (1.0 + dlt)) {
      // +p and -p both exact along an eigenvector with a zero gradient component (same model value and length), or the
      // near-hard case, where |p(lambda)| = Delta is so steep at lambda -> -m0 that the root fixes p only to about
      // sqrt(u) |p| in either path.
      ++ties;
      EXPECT_NEAR(model(a.p), model(b.p), 1e-7 * (1.0 + std::fabs(model(a.p)))) << k;
      EXPECT_NEAR(a.p.norm(), b.p.norm(), 1e-7 * (1.0 + dlt)) << k;
    } else {
      EXPECT_EQ(a.radius_active, b.radius_active) << k;
      ++same;
    }
    if (a.rank_def) ++rank_def;
    if (a.radius_active) ++on_radius;
  }
  EXPECT_GT(same, 1500);
  EXPECT_LT(ties, 150);
  EXPECT_GT(rank_def, 100);
  EXPECT_GT(on_radius, 100);
  EXPECT_GT(joint, 10);
  std::printf("same %d rank-deficient %d on the radius %d sign ties / near-hard %d joint (negative null curvature) %d\n", same,
              rank_def, on_radius, ties, joint);
}

// 3-D, one weighted term: N has rank 1 and a TWO-dimensional null space carrying negative curvature in a direction that
// is no eigenvector the solver of N would return. The step is the exact subproblem on range + the negative null
// eigenspace (brute force agrees), and it is a property of the problem, not of a basis: rotating the whole problem
// rotates the step (J's tie break `pick` is made rotation-covariant here).
TEST(NdTrustRegion, ATwoDimensionalNullSpaceIsHandledExactlyAndIndependentlyOfItsBasis) {
  const Eigen::Vector3d n1 = Eigen::Vector3d(1, 2, 2).normalized();
  const Eigen::Matrix3d N = 2.0 * n1 * n1.transpose();
  const Eigen::Vector3d z = Eigen::Vector3d(2, -1, 0).normalized();  // in the null space (z.n1 = 0)
  Eigen::Matrix3d A = N - 0.7 * z * z.transpose() + 0.2 * Eigen::Matrix3d::Identity();
  A = 0.5 * (A + A.transpose());
  const Eigen::Vector3d g = 0.6 * n1;
  const double dlt = 0.8;
  const Eigen::Vector3d c(0.3, -0.5, 0.8);
  const auto pick = [&](const Eigen::Vector3d& pa, const Eigen::Vector3d& pb) { return pa.dot(c) >= pb.dot(c); };
  // g_tol: the rounding of g, as every solve passes it (a component of g below it is zero).
  const double gtol = 1e-14;
  const auto st = SolveTrSubproblem<3>(g, A, N, 1, dlt, gtol, pick);
  ASSERT_TRUE(st.rank_def);
  const auto model = [&](const Eigen::Vector3d& p) { return g.dot(p) + 0.5 * p.dot(A * p); };
  // The null space's negative eigenspace is span(z) (curvature -0.5 there, +0.2 across): S = span(n1, z).
  Eigen::MatrixXd Q(3, 2);
  Q << n1, z;
  EXPECT_LE(model(st.p), BruteMin<3>(g, A, dlt, Q) + 1e-12);
  EXPECT_NEAR(model(st.p), BruteMin<3>(g, A, dlt, Q), 1e-4);
  EXPECT_NEAR(st.p.norm(), dlt, 1e-9);                      // negative curvature: on the boundary
  EXPECT_NEAR(st.p.dot(n1.cross(z).normalized()), 0.0, 1e-9);  // nothing along the positively curved null direction
  // Rotated problem: the rotated step.
  const Eigen::Matrix3d R = (Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 1, 0).normalized()) *
                             Eigen::AngleAxisd(-1.1, Eigen::Vector3d::UnitZ())).toRotationMatrix();
  const auto pick_r = [&](const Eigen::Vector3d& pa, const Eigen::Vector3d& pb) { return pa.dot(R * c) >= pb.dot(R * c); };
  const auto sr = SolveTrSubproblem<3>(R * g, R * A * R.transpose(), R * N * R.transpose(), 1, dlt, gtol, pick_r);
  EXPECT_NEAR((sr.p - R * st.p).norm(), 0.0, 1e-9);
}

// 3-D hard case (More & Sorensen 1983): N of full rank, A indefinite, g orthogonal to the eigenvector of A's smallest
// eigenvalue and the shifted step inside the ball: the step is completed along that eigenvector to the boundary, and is
// the brute-force minimum.
TEST(NdTrustRegion, TheThreeDimensionalHardCaseIsExact) {
  const Eigen::Matrix3d V = Eigen::AngleAxisd(0.4, Eigen::Vector3d(1, -2, 1).normalized()).toRotationMatrix();
  const Eigen::Matrix3d A = V * Eigen::Vector3d(-0.5, 1.0, 2.0).asDiagonal() * V.transpose();
  const Eigen::Matrix3d N = V * Eigen::Vector3d(0.3, 1.0, 2.0).asDiagonal() * V.transpose();
  const Eigen::Vector3d g = V * Eigen::Vector3d(0.0, 0.2, -0.3);  // no component along the -0.5 eigenvector
  const double dlt = 1.0;
  const auto model = [&](const Eigen::Vector3d& p) { return g.dot(p) + 0.5 * p.dot(A * p); };
  const auto pick = [&](const Eigen::Vector3d& pa, const Eigen::Vector3d& pb) { return model(pa) <= model(pb); };
  const auto st = SolveTrSubproblem<3>(g, A, N, 3, dlt, 0.0, pick);
  EXPECT_FALSE(st.rank_def);
  EXPECT_TRUE(st.radius_active);
  EXPECT_NEAR(st.p.norm(), dlt, 1e-12);
  EXPECT_LE(model(st.p), BruteMin<3>(g, A, dlt, Eigen::MatrixXd::Identity(3, 3)) + 1e-12);
  EXPECT_NEAR(model(st.p), BruteMin<3>(g, A, dlt, Eigen::MatrixXd::Identity(3, 3)), 1e-4);
  // lambda = 0.5: the part off the hard eigenvector is -(c_k / (m_k + 0.5)).
  const Eigen::Vector3d off = V.transpose() * st.p;
  EXPECT_NEAR(off[1], -0.2 / 1.5, 1e-12);
  EXPECT_NEAR(off[2], 0.3 / 2.5, 1e-12);
}

// Three dimensions with the third decoupled: the planar part of the step is the 2-D step, and the third is the 1-D one.
TEST(NdTrustRegion, ADecoupledThirdAxisLeavesThePlanarStepUnchanged) {
  Eigen::Matrix2d A2;
  A2 << 2.0, 0.3, 0.3, 1.0;
  const Eigen::Vector2d g2(0.4, -0.2);
  Eigen::Matrix3d A3 = Eigen::Matrix3d::Zero();
  A3.block<2, 2>(0, 0) = A2;
  A3(2, 2) = 4.0;
  const Eigen::Vector3d g3(0.4, -0.2, 0.8);
  const auto p2 = SolveTrSubproblem<2>(g2, A2, A2, 2, 10.0, 0.0, First<2>);
  const auto p3 = SolveTrSubproblem<3>(g3, A3, A3, 3, 10.0, 0.0, First<3>);
  EXPECT_NEAR((p3.p.head<2>() - p2.p).norm(), 0.0, 1e-12);
  EXPECT_NEAR(p3.p.z(), -0.2, 1e-12);  // -g / a
  EXPECT_FALSE(p3.radius_active);
}

// The pursuit in 3-D (the one path of every anchor, scope_pursuit.h): two distance terms to points whose target distances
// meet on a circle; from the start the trust-region solve lands on the circle at the point nearest the start, with the
// gradient resolved, in an unbounded scope.
TEST(ScopePursuit, TwoPointTermsInThreeDimensionsLandOnTheirCircleNearestTheStart) {
  using V3 = Eigen::Vector3d;
  const auto point_term = [](const V3& c, double d_star, double w) {
    ScopeTerm<3> t;
    t.weight = w;
    t.d_star = d_star;
    t.goal = c;
    t.course_goal = c;
    return t;
  };
  // |x - a| = |x - b| = 5 with a = 0, b = (6,0,0): the circle x = 3, y^2 + z^2 = 16.
  const UnboundedScope<3> scope;
  ScopePursuitInput<3> in;
  in.scope = &scope;
  in.start = V3(2.5, 4.0, 4.0);
  in.terms = {point_term(V3::Zero(), 5.0, 1.0), point_term(V3(6, 0, 0), 5.0, 1.0)};
  for (auto& t : in.terms) t.d_sample = (in.start - *t.goal).norm();
  ScopePursuitResult<3> r;
  int evaluations = 0;
  bool budget_tripped = false;
  double resolution = -1.0;
  PursueInScope<3>(in, r, evaluations, budget_tripped, resolution);
  EXPECT_TRUE(r.objective_stationary);
  EXPECT_TRUE(r.objective_exact);
  EXPECT_NEAR(r.position.x(), 3.0, 1e-6);
  EXPECT_NEAR(std::hypot(r.position.y(), r.position.z()), 4.0, 1e-6);
  // The tangential freedom is not invented: the answer lies on the start's own side of the circle (the y = z plane).
  EXPECT_NEAR(r.position.y(), r.position.z(), 1e-6);
}

// A box scope (the unit cube, convex): one relation to a point above it, n = 1, is pursued along its course as far as
// F allows (the root's rule: the free part of the segment from the start), and the answer is in F.
class UnitBoxScope final : public RelationScope<3> {
 public:
  bool IsFree(const Vec& x) const override { return (x.array() >= 0.0).all() && (x.array() <= 1.0).all(); }
  bool SegmentFree(const Vec& a, const Vec& b) const override { return IsFree(a) && IsFree(b); }
  std::optional<Vec> NearestFree(const Vec& x, NearestFreeStatus* st = nullptr) const override {
    if (st) *st = IsFree(x) ? NearestFreeStatus::kAlreadyFree : NearestFreeStatus::kProjected;
    return Vec(x.cwiseMax(Vec::Zero()).cwiseMin(Vec::Ones()));
  }
  std::optional<Vec> NearestFreeUnder(const Vec& x) const override { return NearestFree(x); }
  Vec ProjectDirection(const Vec& x, const Vec& v) const override {
    Vec o = v;
    for (int k = 0; k < 3; ++k) {
      if (x[k] <= 0.0 && o[k] < 0) o[k] = 0;
      if (x[k] >= 1.0 && o[k] > 0) o[k] = 0;
    }
    return o;
  }
  Cone TangentConeAt(const Vec& x, double tol) const override {
    Cone c;
    for (int k = 0; k < 3; ++k) {
      if (x[k] <= tol) c.facets.push_back({Vec(-Vec::Unit(k)), 0.0});
      if (x[k] >= 1.0 - tol) c.facets.push_back({Vec(Vec::Unit(k)), 0.0});
    }
    return c;
  }
  Reach Reachability(const Vec& from, const Vec& to) const override {
    const Vec a = *NearestFree(from), b = *NearestFree(to);
    Reach r;
    r.reachable = true;
    r.from_snap_m = (from - a).norm();
    r.to_snap_m = (to - b).norm();
    r.length = r.from_snap_m + (b - a).norm() + r.to_snap_m;
    if (r.from_snap_m > 1e-12) r.path.push_back(from);
    r.path.push_back(a);
    r.path.push_back(b);
    if (r.to_snap_m > 1e-12) r.path.push_back(to);
    return r;
  }
  std::vector<Route> GeodesicRoutesUnder(const Vec&, const Vec&, double) const override { return {}; }
  std::vector<Route> GeodesicRoutes(const Vec&, const Vec&, double) const override { return {}; }
  std::vector<Route> ContactRoutes(const Vec&, const std::vector<const std::vector<Vec>*>&, double) const override {
    return {};
  }
  double RouteSegmentBound() const override { return 0.0; }
  double ObstacleClearance() const override { return 0.0; }
  double Diameter() const override { return std::sqrt(3.0); }
  bool Bounded() const override { return true; }
};
TEST(ScopePursuit, ABoxScopeIsPursuedAsFarAsItAllows) {
  using V3 = Eigen::Vector3d;
  const UnitBoxScope scope;
  ScopePursuitInput<3> in;
  in.scope = &scope;
  in.start = V3(0.2, 0.7, 0.1);
  ScopeTerm<3> t;
  t.weight = 1.0;
  t.d_star = 0.0;
  t.update_count = 1.0;
  t.goal = V3(0.5, 0.5, 2.0);
  t.course_goal = *t.goal;
  t.d_sample = scope.Reachability(in.start, *t.goal).length;
  in.terms = {t};
  ScopePursuitResult<3> r;
  int evaluations = 0;
  bool budget_tripped = false;
  double resolution = -1.0;
  PursueInScope<3>(in, r, evaluations, budget_tripped, resolution);
  EXPECT_TRUE(scope.IsFree(r.position));
  // The segment from the start to the goal leaves the cube through its top face, z = 1.
  const double s = (1.0 - 0.1) / (2.0 - 0.1);
  EXPECT_NEAR((r.position - (in.start + s * (*t.goal - in.start))).norm(), 0.0, 1e-9);
}

// The refutation of a chi descent at J's accuracy R (TrChiRefutedAtResolution). The numbers are the macOS arm64 collapse of
// occluded|passby_mid|uniform1.3|global|noflag|1.4|15, frame 43: chi 4.92e-6 against U3's 2.39e-6, R = 1.36e-17, J rising past R
// within s = 8.5e-14 m of an interior point (no pull): chi * s = 4.2e-19 <= R, refuted.
TEST(ChiRefutation, AnInteriorRiseWithinRBeforeChiCanGainRIsARefutation) {
  const double j = 1.0545600286914274e-05, r = 1.36e-17;
  EXPECT_TRUE(TrChiRefutedAtResolution(4.92e-6, 8.5e-14, j + 2.0 * r, j, r));
}

// On an active face of F the trial points are pulled inward by 2 gamma_8 max(1, |x|), of the order of (here more than) the search's
// first length s0 = 4 eps max(1, |x|) (1.06e-14 m at |x| = 12). Where J's gradient presses into the wall,
// the pull ALONE raises J by more than R, so the first trial already reads a rise above J(x) at s = s0. Measured from J(x), the
// test would pass for any chi up to R / s0 (1.3e-3, some 500 times U3's threshold): a point with a real tangential descent would
// stop as a minimiser. Measured from J(x + pull), the same trial shows no rise along v, and the rule does not fire.
TEST(ChiRefutation, APullIntoTheFaceThatAloneRaisesJIsNotARefutation) {
  const double j = 1.0e-5, r = 1.36e-17, s0 = 1.06e-14, chi = 1.0e-3;
  const double j_pull = j + 5.0 * r;          // J(x + pull): the pull alone raises J by 5 R
  const double j_trial = j_pull - 1.0e-18;    // J(x + s0 v + pull): along v, J fell
  ASSERT_GT(j_trial, j + r) << "the trial reads a rise above J(x): the case the rule must not take";
  ASSERT_LE(chi * s0, r);
  EXPECT_FALSE(TrChiRefutedAtResolution(chi, s0, j_trial, j_pull, r));
  // and a rise along v ABOVE the pulled base, within R of what chi can gain, is still one
  EXPECT_TRUE(TrChiRefutedAtResolution(chi, s0, j_pull + 2.0 * r, j_pull, r));
}

// A descent chi can resolve before J rises (chi * s_rise > R) is not refuted, however J rose after it; nor is a search that never
// saw a rise (s_rise 0) or a rise within R.
TEST(ChiRefutation, AResolvableDescentOrNoRiseIsNotARefutation) {
  const double j = 1.0e-5, r = 1.36e-17;
  EXPECT_FALSE(TrChiRefutedAtResolution(4.92e-6, 1.0e-10, j + 2.0 * r, j, r));  // chi * s = 4.9e-16 > R
  EXPECT_FALSE(TrChiRefutedAtResolution(4.92e-6, 0.0, j + 2.0 * r, j, r));      // no rise seen
  EXPECT_FALSE(TrChiRefutedAtResolution(4.92e-6, 8.5e-14, j + 0.5 * r, j, r));  // a rise within R
}

}  // namespace
}  // namespace spatialblend
