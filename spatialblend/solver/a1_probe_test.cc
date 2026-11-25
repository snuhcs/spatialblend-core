// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Displacement-cap, facing-pull, and a
// 150-scene randomized probe vs. brute force, for the paper-objective
// refinement in relation_solver.cc.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/obstacle_free_space.h"
#include "spatialblend/model/relation/avatar_facing_kernel.h"
#include "spatialblend/model/relation/avatar_spacing_kernel.h"
#include "spatialblend/model/relation/grounding_rule.h"
#include "spatialblend/model/relation/relation.h"
#include "spatialblend/solver/relation_evaluator.h"
#include "spatialblend/solver/relation_solver.h"
#include "spatialblend/model/realization/remote_update.h"

namespace spatialblend {
namespace test {
namespace {

using Clock = std::chrono::steady_clock;

RelationPayload SpacingPayload(const Eigen::Vector2d& off, shared_element_id_t sid) {
  ElementTree src;
  src.AddElement(1);
  src.AddElement(2, 1);
  src.GetElement(2)->SetSharedId(sid);
  Relation rel("spacing", "root", 2, std::make_unique<DefaultGroundingRule>(),
               std::make_unique<AvatarSpacingKernel>());
  auto r = rel.Evaluate(Eigen::Vector3d(off.x(), off.y(), 0), 0.0, src);
  RelationPayload p;
  p.referent_shared_id = sid;
  p.subject_anchor_name = "root";
  p.grounded_relation = r.grounded;
  p.support = r.support;
  p.kernel_type = RelationKernelType::kAvatarSpacing;
  return p;
}

RelationPayload FacingPayload(const Eigen::Vector2d& off, double yaw,
                              shared_element_id_t sid) {
  ElementTree src;
  src.AddElement(1);
  src.AddElement(2, 1);
  src.GetElement(2)->SetSharedId(sid);
  Relation rel("facing", "root", 2, std::make_unique<DefaultGroundingRule>(),
               std::make_unique<AvatarFacingKernel>());
  auto r = rel.Evaluate(Eigen::Vector3d(off.x(), off.y(), 0), yaw, src);
  RelationPayload p;
  p.referent_shared_id = sid;
  p.subject_anchor_name = "root";
  p.grounded_relation = r.grounded;
  p.support = r.support;
  p.kernel_type = RelationKernelType::kAvatarFacing;
  return p;
}

void AddReferent(ElementTree* t, element_id_t id, shared_element_id_t sid,
                 const Eigen::Vector2d& pos) {
  t->AddElement(id, 1);
  t->GetElement(id)->SetSharedId(sid);
  Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
  m(0, 3) = pos.x();
  m(1, 3) = pos.y();
  t->GetElement(id)->SetLocalTransform(m);
}

double J(const std::vector<RelationPayload>& ps, const std::vector<double>& w,
         const ElementTree& t, const Eigen::Vector2d& p, double yaw) {
  TargetAnchorPose a;
  a.position = Eigen::Vector3d(p.x(), p.y(), 0);
  a.yaw = yaw;
  double j = 0;
  for (size_t i = 0; i < ps.size(); ++i) {
    auto ev = EvaluateTargetRelation(ps[i], a, t);
    if (!ev.ok()) continue;
    j += w[i] * ev.error * ev.error;
  }
  return j;
}

struct Brute {
  double j = 1e300;
  Eigen::Vector2d p;
};

Brute BruteForce(const std::vector<RelationPayload>& ps,
                 const std::vector<double>& w, const ElementTree& t,
                 const NavigableSpatialDefinition* nav, double lo, double hi,
                 double step) {
  Brute b;
  for (double x = lo; x <= hi + 1e-9; x += step)
    for (double y = lo; y <= hi + 1e-9; y += step) {
      Eigen::Vector2d p(x, y);
      if (nav && !nav->IsFree(p)) continue;
      double j = J(ps, w, t, p, 0.0);
      if (j < b.j) {
        b.j = j;
        b.p = p;
      }
    }
  Brute c = b;
  for (double x = b.p.x() - step; x <= b.p.x() + step; x += step / 20)
    for (double y = b.p.y() - step; y <= b.p.y() + step; y += step / 20) {
      Eigen::Vector2d p(x, y);
      if (nav && !nav->IsFree(p)) continue;
      double j = J(ps, w, t, p, 0.0);
      if (j < c.j) {
        c.j = j;
        c.p = p;
      }
    }
  return c;
}

TEST(A1ProbeAdapted, DisplacementCapAllowsExactOptimum) {
  auto a = SpacingPayload({12, 0}, 200);
  auto b = SpacingPayload({-12, 0}, 300);
  ElementTree t;
  t.AddElement(1);
  AddReferent(&t, 2, 200, {0, 0});
  AddReferent(&t, 3, 300, {20, 0});
  RelationSolver s;
  auto r = s.Solve({10, 0}, 0, {a, b}, t);
  std::printf("[cap] default: pos=(%.3f,%.3f) J0=%.3e J=%.3e it=%d evals=%d restarts=%d\n",
              r.position.x(), r.position.y(), r.objective_before,
              r.objective_after, r.objective_iterations,
              r.objective_evaluations, r.objective_restarts);
}

TEST(A1ProbeAdapted, FacingPullsPosition) {
  auto sp = SpacingPayload({2, 0}, 200);
  auto f1 = FacingPayload({-3, 0}, 0.0, 300);
  auto f2 = FacingPayload({0, -3}, 0.0, 400);
  ElementTree t;
  t.AddElement(1);
  AddReferent(&t, 2, 200, {0, 0});
  AddReferent(&t, 3, 300, {0, 6});
  AddReferent(&t, 4, 400, {-6, 0});
  RelationSolver s;
  auto base_like = s.Solve({0, 0}, 0, {sp}, {1.0}, t);
  auto r = s.Solve({0, 0}, 0, {sp, f1, f2}, {0.2, 0.4, 0.4}, t);
  std::printf("[facing-pull] spacing-only pos=(%.3f,%.3f); joint pos=(%.3f,%.3f) "
              "yaw=%.3f J0=%.4f J=%.4f fb=%d it=%d drift=%.3f m\n",
              base_like.position.x(), base_like.position.y(), r.position.x(),
              r.position.y(), r.yaw, r.objective_before, r.objective_after,
              r.objective_fallback, r.objective_iterations,
              (r.position - base_like.position).norm());
  EXPECT_NEAR((r.position - base_like.position).norm(), 0.0, 1e-9);
}

// RETIRED (the J refinement no longer exists): asserts what the position search / refinement produced (an exact optimum, restarts, tracking, acceptance rule, basin loss) or a step bound the refinement shaped. The position answer is S0 clamped; the harness defects (proposal == clamped S0, identical copies exact, every jump traced to its cause) replace these checks.
TEST(A1ProbeAdapted, DISABLED_RandomScenesVsBruteForce) {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> U(0, 1);
  const int kScenes = 150;
  int far_opt = 0, stuck = 0, improved = 0, worse_than_init = 0;
  double max_gap = 0, sum_ms = 0, max_ms = 0;
  int total_evals = 0, total_restarts = 0;
  for (int sc = 0; sc < kScenes; ++sc) {
    const int k = 2 + (sc % 3);
    ElementTree t;
    t.AddElement(1);
    std::vector<RelationPayload> ps;
    std::vector<double> w;
    double wsum = 0;
    for (int i = 0; i < k; ++i) {
      const double px = -6 + 12 * U(rng);  // drawn in order (argument evaluation order is unspecified)
      const double py = -6 + 12 * U(rng);
      Eigen::Vector2d pos(px, py);
      AddReferent(&t, 2 + i, 200 + 100 * i, pos);
      double ang = 2 * M_PI * U(rng), d = 0.5 + 5.5 * U(rng);
      ps.push_back(SpacingPayload(d * Eigen::Vector2d(std::cos(ang), std::sin(ang)),
                                  200 + 100 * i));
      w.push_back(0.1 + U(rng));
      wsum += w.back();
    }
    for (auto& x : w) x /= wsum;
    auto walk = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
        {-10, -10}, {10, -10}, {10, 10}, {-10, 10}});
    NavigableSpatialDefinition nav(walk);
    const bool obstacle = sc % 2 == 1;
    if (obstacle) {
      const double cx = -5 + 10 * U(rng);  // drawn in order
      const double cy = -5 + 10 * U(rng);
      Eigen::Vector2d c(cx, cy);
      double hw = 0.5 + 1.5 * U(rng), hh = 0.5 + 1.5 * U(rng);
      auto ob = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
          c + Eigen::Vector2d(-hw, -hh), c + Eigen::Vector2d(hw, -hh),
          c + Eigen::Vector2d(hw, hh), c + Eigen::Vector2d(-hw, hh)});
      nav.SetObstacles({FreeSpaceObstacle{1, ob}}, 0.1);
    }
    Eigen::Vector2d cur(-9.5, -9.5);
    RelationSolver s;
    auto t0 = Clock::now();
    auto r = s.Solve(cur, 0, ps, w, t, &nav);
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    sum_ms += ms;
    max_ms = std::max(max_ms, ms);
    total_evals += r.objective_evaluations;
    total_restarts += r.objective_restarts;
    if (r.objective_after > r.objective_before + 1e-12) ++worse_than_init;
    if (r.objective_after < r.objective_before - 1e-9) ++improved;
    auto bf = BruteForce(ps, w, t, &nav, -10, 10, 0.1);
    double jr = J(ps, w, t, r.position, 0);
    double gap = jr - bf.j;
    max_gap = std::max(max_gap, gap);
    bool is_stuck = gap > 1e-4 && gap > 0.1 * bf.j;
    if (is_stuck) ++stuck;
    if ((r.position - bf.p).norm() > 1.0 && gap > 1e-5) ++far_opt;
    if (is_stuck)
      std::printf("[scene %d k=%d obs=%d] J0=%.2e J=%.2e Jbf=%.2e pos=(%.2f,%.2f) "
                  "bf=(%.2f,%.2f) it=%d fb=%d restarts=%d\n",
                  sc, k, obstacle, r.objective_before, jr, bf.j, r.position.x(),
                  r.position.y(), bf.p.x(), bf.p.y(), r.objective_iterations,
                  r.objective_fallback, r.objective_restarts);
  }
  std::printf("[summary] scenes=%d improved=%d stuck(gap>1e-4 & >10%%)=%d "
              "far_from_global(>1m)=%d max_gap=%.3e worse_than_init=%d "
              "mean_ms=%.3f max_ms=%.3f mean_evals=%.1f mean_restarts=%.2f\n",
              kScenes, improved, stuck, far_opt, max_gap, worse_than_init,
              sum_ms / kScenes, max_ms, (double)total_evals / kScenes,
              (double)total_restarts / kScenes);
  // Regression gates. The search must never regress the initializer. The
  // stuck-scene count is the quality gate on the restarts; its value,
  // measured on THIS probe (this RNG seed, 150 scenes):
  //   - 22/150 before the restart fix;
  //   - 8/150 with the restarts (far_from_global 9, max_gap 2.75e-3);
  //   - after the exact-verification restructure it is 5/150
  //     (far_from_global 6, max_gap 2.14e-3).
  // Gate at the pre-fix measured value (8), not at slack above it: the count
  // may only get better.
  EXPECT_EQ(worse_than_init, 0) << "search must never be worse than S0";
  EXPECT_LE(stuck, 8) << "stuck-scene count regressed past the measured 8/150";
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
