// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/solver_layer_test.cc
//
// Layer 2 (solver), isolated. A pure function of (relation payloads, weights,
// receiver scene, embodied sample): the mapped contact of each relation, S0 =
// the softmax-weighted centroid of those contacts (the cold initializer only),
// the continuation sample ⊕ δ inside F, and the yaw the conditioned circular
// mean of the Facing referents' decoded directions.
// No Space, no time, no scope resolution, no driver.
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/obstacle_free_space.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/relation/avatar_facing_kernel.h"
#include "spatialblend/model/relation/avatar_spacing_kernel.h"
#include "spatialblend/model/relation/grounding_rule.h"
#include "spatialblend/model/relation/relation.h"
#include "spatialblend/solver/relation_solver.h"

namespace spatialblend {
namespace test {
namespace {

using Vec2 = Eigen::Vector2d;

// A scene root with two boards (shared ids 200, 300). Each board's local frame is
// a similarity transform: rotation `theta`, uniform `scale`, then `shift`.
struct Scene {
  double theta = 0.0, scale = 1.0;
  Vec2 shift = Vec2::Zero();
  Vec2 b200 = Vec2::Zero(), b300 = Vec2::Zero();  // board positions BEFORE the transform
};

Eigen::Matrix4d Similarity(const Scene& s, const Vec2& p) {
  const double c = std::cos(s.theta), sn = std::sin(s.theta);
  Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
  m(0, 0) = s.scale * c;
  m(0, 1) = -s.scale * sn;
  m(1, 0) = s.scale * sn;
  m(1, 1) = s.scale * c;
  m(0, 3) = s.scale * (c * p.x() - sn * p.y()) + s.shift.x();
  m(1, 3) = s.scale * (sn * p.x() + c * p.y()) + s.shift.y();
  return m;
}

Vec2 Apply(const Scene& s, const Vec2& p) {
  const Eigen::Matrix4d m = Similarity(s, Vec2::Zero());
  return Vec2(m(0, 0) * p.x() + m(0, 1) * p.y() + m(0, 3),
              m(1, 0) * p.x() + m(1, 1) * p.y() + m(1, 3));
}

ElementTree MakeTree(const Scene& s,
                     std::shared_ptr<NavigableSpatialDefinition> room = nullptr) {
  ElementTree tree;
  tree.AddElement(1);
  if (room) tree.GetElement(1)->SetSpatialDefinition(room);
  tree.AddElement(2, 1);
  tree.GetElement(2)->SetSharedId(200);
  tree.GetElement(2)->SetLocalTransform(Similarity(s, s.b200));
  tree.AddElement(3, 1);
  tree.GetElement(3)->SetSharedId(300);
  tree.GetElement(3)->SetLocalTransform(Similarity(s, s.b300));
  return tree;
}

RelationPayload MakePayload(RelationKernelType kind, const Vec2& actor, double yaw,
                            const ElementTree& source, element_id_t referent,
                            shared_element_id_t shared_id,
                            const NavigableSpatialDefinition* nav = nullptr) {
  const bool spacing = kind == RelationKernelType::kAvatarSpacing;
  std::unique_ptr<RelationKernel> kernel;
  if (spacing) kernel = std::make_unique<AvatarSpacingKernel>();
  else kernel = std::make_unique<AvatarFacingKernel>();
  Relation rel(spacing ? "spacing" : "facing", spacing ? "root" : "chest", referent,
               std::make_unique<DefaultGroundingRule>(), std::move(kernel));
  const auto result =
      rel.Evaluate(Eigen::Vector3d(actor.x(), actor.y(), spacing ? 0.0 : 1.3), yaw,
                   source, nav);
  RelationPayload p;
  p.referent_shared_id = shared_id;
  p.subject_anchor_name = spacing ? "root" : "chest";
  p.grounded_relation = result.grounded;
  p.support = result.support;
  p.kernel_type = kind;
  // A Facing request with count 1: the yaw reaches its target this update, the placed yaw these tests are about. The
  // yaw is a ratio continuation: without a valid count it holds, and with n > 1 it turns
  // 1/n of the receiver's own offset.
  if (!spacing) p.update_count = 1.0;
  return p;
}

// The approach points of a two-relation solve: witness k belongs to payload k.
std::vector<Vec2> Witnesses(const SolverResult& r, size_t n) {
  std::vector<Vec2> w(n, Vec2::Constant(std::nan("")));
  for (const auto& g : r.goals)
    if (g.position_goal && g.payload_index < n) w[g.payload_index] = g.witness;
  return w;
}

// Source: identical boards, the actor at fixed offsets from each.
const Vec2 kActor200(3.0, 1.0);   // local offset from board 200
const Vec2 kActor300(-2.0, -1.0);  // local offset from board 300

struct Fixture {
  Scene src;
  ElementTree src_tree;
  RelationPayload spacing_a, spacing_b;
  // The source's scope: a square room. A similarity transform that rescales the
  // receiver's room by the same factor makes the stand-off exact (D_t / D_s).
  std::shared_ptr<NavigableSpatialDefinition> src_nav =
      std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(
          std::vector<Vec2>{{-40, -40}, {40, -40}, {40, 40}, {-40, 40}}));
  Fixture() {
    src.b200 = Vec2(0, 0);
    src.b300 = Vec2(6, 0);
    src_tree = MakeTree(src, src_nav);
    spacing_a = MakePayload(RelationKernelType::kAvatarSpacing,
                            src.b200 + kActor200, 0.0, src_tree, 2, 200, src_nav.get());
    spacing_b = MakePayload(RelationKernelType::kAvatarSpacing,
                            src.b300 + kActor300, 0.0, src_tree, 3, 300, src_nav.get());
  }
  // The receiver's room under the similarity (scale s, rotation by a multiple of 90
  // degrees so the axis-aligned scope extent scales by exactly s).
  static std::shared_ptr<NavigableSpatialDefinition> TargetNav(const Scene& t) {
    std::vector<Vec2> pts;
    for (const Vec2& c : std::vector<Vec2>{{-40, -40}, {40, -40}, {40, 40}, {-40, 40}})
      pts.push_back(Apply(t, c));
    return std::make_shared<NavigableSpatialDefinition>(
        std::make_shared<Polygon2D>(pts));
  }
};

}  // namespace

// --- the decoded contact: the mapped support, not the source stand-off ---------
// The witness is the source contact mapped onto the receiver support. The actor
// pose that preserves R is the continuation (sample ⊕ δ), not that contact.

class ExactDecode : public ::testing::TestWithParam<std::tuple<double, double>> {};

TEST_P(ExactDecode, MappedContactAndContinuationPreserveTheSource) {
  const auto [theta, scale] = GetParam();
  Fixture f;
  Scene tgt;
  tgt.theta = theta;
  tgt.scale = scale;
  tgt.shift = Vec2(10.0, -4.0);
  tgt.b200 = f.src.b200;
  tgt.b300 = f.src.b300;
  const auto tgt_nav = Fixture::TargetNav(tgt);
  const ElementTree tgt_tree = MakeTree(tgt, tgt_nav);

  RelationSolver solver;
  const SolverResult r = solver.Solve(Vec2(0, 0), 0.0, {f.spacing_a, f.spacing_b},
                                      {1.0, 1.0}, tgt_tree, tgt_nav.get());
  const auto w = Witnesses(r, 2);
  const Vec2 contact_a = Apply(tgt, f.src.b200);
  const Vec2 contact_b = Apply(tgt, f.src.b300);
  EXPECT_NEAR((w[0] - contact_a).norm(), 0.0, 1e-6)
      << "theta=" << theta << " scale=" << scale;
  EXPECT_NEAR((w[1] - contact_b).norm(), 0.0, 1e-6)
      << "theta=" << theta << " scale=" << scale;

  // One spacing relation, started at the source pose carried into the receiver
  // frame: the residual is zero, so the proposal is that pose.
  const Vec2 carried = Apply(tgt, f.src.b200 + kActor200);
  const SolverResult held = solver.Solve(carried, 0.0, {f.spacing_a}, {1.0},
                                         tgt_tree, tgt_nav.get());
  EXPECT_NEAR((held.position - carried).norm(), 0.0, 1e-3)
      << "theta=" << theta << " scale=" << scale;
}

INSTANTIATE_TEST_SUITE_P(
    IdenticalAndSimilarity, ExactDecode,
    ::testing::Values(std::make_tuple(0.0, 1.0),          // identical copy
                      std::make_tuple(M_PI / 2, 1.0),     // rotated copy
                      std::make_tuple(0.0, 1.5),          // uniform scale
                      std::make_tuple(-M_PI / 2, 0.6)));  // rotated and scaled

// --- S0 is the hand-computed softmax-weighted point ----------------------------

TEST(S0, IsTheWeightedPointOfTheApproachPoints) {
  Fixture f;
  Scene tgt = f.src;
  tgt.theta = 0.4;
  tgt.shift = Vec2(2.0, 3.0);
  const ElementTree tgt_tree = MakeTree(tgt);
  RelationSolver solver;
  const std::vector<double> weights = {0.3, 0.7};
  const SolverResult r = solver.Solve(Vec2(0, 0), 0.0, {f.spacing_a, f.spacing_b},
                                      weights, tgt_tree);
  const auto a = Witnesses(r, 2);
  const Vec2 hand = (0.3 * a[0] + 0.7 * a[1]) / 1.0;
  // S0 is recorded. It is not the answer: the answer continues from the sample.
  EXPECT_NEAR((r.unconstrained_position - hand).norm(), 0.0, 1e-12);
  EXPECT_GT((r.position - hand).norm(), 1.0);
  const SolverResult r2 = solver.Solve(Vec2(0, 0), 0.0, {f.spacing_a, f.spacing_b},
                                       {3.0, 7.0}, tgt_tree);
  EXPECT_NEAR((r2.unconstrained_position - hand).norm(), 0.0, 1e-12);
}

// --- weight sensitivity: dS0 = sum dw_i (a_i - S0) / sum w' ---------

TEST(S0, WeightSensitivity) {
  Fixture f;
  Scene tgt = f.src;
  tgt.shift = Vec2(1.0, 1.0);
  const ElementTree tgt_tree = MakeTree(tgt);
  RelationSolver solver;
  const std::vector<double> w0 = {0.3, 0.7}, w1 = {0.5, 0.9};
  const SolverResult r0 = solver.Solve(Vec2(0, 0), 0.0, {f.spacing_a, f.spacing_b}, w0, tgt_tree);
  const SolverResult r1 = solver.Solve(Vec2(0, 0), 0.0, {f.spacing_a, f.spacing_b}, w1, tgt_tree);
  const auto a = Witnesses(r0, 2);
  Vec2 predicted = Vec2::Zero();
  for (int i = 0; i < 2; ++i)
    predicted += (w1[i] - w0[i]) * (a[i] - r0.unconstrained_position);
  predicted /= (w1[0] + w1[1]);
  // The centroid moves with the weights. The proposal is the continuation.
  EXPECT_NEAR(((r1.unconstrained_position - r0.unconstrained_position) - predicted).norm(),
              0.0, 1e-12);
  // The approach points do not depend on the weights.
  const auto a1 = Witnesses(r1, 2);
  EXPECT_NEAR((a1[0] - a[0]).norm() + (a1[1] - a[1]).norm(), 0.0, 1e-12);
}

// --- the clamp is NearestFree(S0) and nothing else -------------------------------

TEST(Clamp, IsNearestFreeOfS0) {
  Fixture f;
  Scene tgt = f.src;
  tgt.shift = Vec2(8.0, 8.0);
  const ElementTree tgt_tree = MakeTree(tgt);
  NavigableSpatialDefinition nav(std::make_shared<Polygon2D>(std::vector<Vec2>{
      {-20, -20}, {40, -20}, {40, 40}, {-20, 40}}));
  RelationSolver solver;
  const SolverResult free_r =
      solver.Solve(Vec2(0, 0), 0.0, {f.spacing_a, f.spacing_b}, {1.0, 1.0}, tgt_tree);
  // An obstacle over S0: the answer is the nearest free point to S0.
  const Vec2 s0 = free_r.unconstrained_position;
  nav.SetObstacles(
      {FreeSpaceObstacle{1, std::make_shared<Polygon2D>(std::vector<Vec2>{
                                {s0.x() - 1, s0.y() - 1}, {s0.x() + 1, s0.y() - 1},
                                {s0.x() + 1, s0.y() + 1}, {s0.x() - 1, s0.y() + 1}})}},
      /*clearance=*/0.1);
  const SolverResult r = solver.Solve(Vec2(s0.x() + 5, s0.y()), 0.0,
                                      {f.spacing_a, f.spacing_b}, {1.0, 1.0},
                                      tgt_tree, &nav);
  EXPECT_NEAR((r.unconstrained_position - s0).norm(), 0.0, 1e-12);
  // The sample sits 5 m to the +x of S0. The proposal continues from there and
  // stays in F. It is not NearestFree(S0), which would cross the obstacle.
  EXPECT_TRUE(nav.IsFree(r.position));
  EXPECT_GT(r.position.x(), s0.x());
  const auto snapped = nav.NearestFree(s0);
  ASSERT_TRUE(snapped.has_value());
  EXPECT_GT((r.position - *snapped).norm(), 0.5);
}

// --- rigid-transform equivariance of the whole answer -----------------------------

TEST(S0, RigidTransformEquivariance) {
  Fixture f;
  Scene a;
  a.b200 = f.src.b200;
  a.b300 = f.src.b300;
  Scene b = a;
  b.theta = 0.7;
  b.shift = Vec2(10.0, -4.0);
  const ElementTree ta = MakeTree(a), tb = MakeTree(b);
  RelationSolver solver;
  const std::vector<double> w = {0.4, 0.6};
  const SolverResult ra = solver.Solve(Vec2(0, 0), 0.0, {f.spacing_a, f.spacing_b}, w, ta);
  const SolverResult rb = solver.Solve(Apply(b, Vec2(0, 0)), 0.0,
                                       {f.spacing_a, f.spacing_b}, w, tb);
  EXPECT_NEAR((rb.unconstrained_position - Apply(b, ra.unconstrained_position)).norm(),
              0.0, 1e-9);
  EXPECT_NEAR(rb.objective_before, ra.objective_before, 1e-9);
  // The continuation step transforms with the scene. The axis-aligned
  // finite-difference Gauss-Newton stops a fraction of a step off the exact
  // rotate; a missing transform is metres, not this residual.
  EXPECT_NEAR((rb.position - Apply(b, ra.position)).norm(), 0.0, 0.1);
}

// --- yaw: each Facing referent decodes a direction; yaw = their circular mean ----

namespace {
double Wrap(double a) { return std::remainder(a, 2.0 * M_PI); }
}  // namespace

class ExactYaw : public ::testing::TestWithParam<std::tuple<double, double>> {};

TEST_P(ExactYaw, FacingDirectionIsTheSourceDirectionInTheReceiverFrame) {
  const auto [theta, scale] = GetParam();
  Fixture f;
  const double src_yaw = 0.3;
  const Vec2 src_actor(3.0, 6.0);
  const RelationPayload fa = MakePayload(RelationKernelType::kAvatarFacing, src_actor,
                                         src_yaw, f.src_tree, 2, 200);
  const RelationPayload fb = MakePayload(RelationKernelType::kAvatarFacing, src_actor,
                                         src_yaw, f.src_tree, 3, 300);
  Scene tgt;
  tgt.theta = theta;
  tgt.scale = scale;
  tgt.shift = Vec2(10.0, -4.0);
  tgt.b200 = f.src.b200;
  tgt.b300 = f.src.b300;
  const ElementTree tgt_tree = MakeTree(tgt);
  RelationSolver solver;
  // The receiver stands where the source stands, carried by the same similarity:
  // the source's yaw is rotated by theta.
  const SolverResult r = solver.Solve(Apply(tgt, src_actor), src_yaw, {fa, fb},
                                      {0.25, 0.75}, tgt_tree);
  EXPECT_NEAR(Wrap(r.yaw - (src_yaw + theta)), 0.0, 1e-9)
      << "theta=" << theta << " scale=" << scale;
  // A Facing-only solve has no position terms: nothing to report as inexact.
  EXPECT_EQ(r.objective_relation_count, 0);
  EXPECT_FALSE(r.has_objective_seed && !r.objective_exact && r.objective_relation_count > 0);
}

INSTANTIATE_TEST_SUITE_P(
    IdenticalAndSimilarity, ExactYaw,
    ::testing::Values(std::make_tuple(0.0, 1.0), std::make_tuple(0.9, 1.0),
                      std::make_tuple(0.0, 1.7), std::make_tuple(-2.2, 0.5)));

TEST(Yaw, IsTheWeightedCircularMeanOfTheDecodedDirections) {
  Fixture f;
  const Vec2 src_actor(3.0, 6.0);
  // Facing both boards (offsets of +-27 degrees: both well-conditioned).
  const double src_yaw = -M_PI / 2;
  const RelationPayload fa = MakePayload(RelationKernelType::kAvatarFacing, src_actor,
                                         src_yaw, f.src_tree, 2, 200);
  const RelationPayload fb = MakePayload(RelationKernelType::kAvatarFacing, src_actor,
                                         src_yaw, f.src_tree, 3, 300);
  // The receiver scene moved ONLY board 300: the two decoded directions disagree.
  Scene tgt = f.src;
  tgt.b300 = Vec2(-4.0, 9.0);
  const ElementTree tgt_tree = MakeTree(tgt);
  RelationSolver solver;
  const Vec2 at = src_actor;
  const std::vector<double> w = {0.2, 0.8};
  const SolverResult r = solver.Solve(at, src_yaw, {fa, fb}, w, tgt_tree);
  const SolverResult only_a = solver.Solve(at, src_yaw, {fa, fb}, {1.0, 1e-30}, tgt_tree);
  const SolverResult only_b = solver.Solve(at, src_yaw, {fa, fb}, {1e-30, 1.0}, tgt_tree);
  const double ya = only_a.yaw, yb = only_b.yaw;
  EXPECT_GT(std::fabs(Wrap(ya - yb)), 0.1) << "the directions must disagree";
  const double expected = std::atan2(0.2 * std::sin(ya) + 0.8 * std::sin(yb),
                                     0.2 * std::cos(ya) + 0.8 * std::cos(yb));
  EXPECT_NEAR(Wrap(r.yaw - expected), 0.0, 1e-9);
}

// A solve with no Facing relation specifies no yaw (the driver faces its direction
// of motion, as v1 did); with a Facing relation the yaw is specified, even when
// every bearing is conditioned out (the heading fallback is still an answer).
TEST(Yaw, IsUnspecifiedWithoutAFacingReferent) {
  Fixture f;
  Scene tgt = f.src;
  tgt.shift = Vec2(3.0, 1.0);
  const ElementTree tgt_tree = MakeTree(tgt);
  RelationSolver solver;
  const SolverResult spacing_only =
      solver.Solve(Vec2(0, 0), 0.7, {f.spacing_a, f.spacing_b}, {1.0, 1.0}, tgt_tree);
  EXPECT_FALSE(spacing_only.yaw_specified);
  EXPECT_NEAR(Wrap(spacing_only.yaw - 0.7), 0.0, 1e-12) << "yaw echoes the held yaw";

  const RelationPayload facing = MakePayload(RelationKernelType::kAvatarFacing,
                                             Vec2(3.0, 6.0), -M_PI / 2, f.src_tree, 2, 200);
  const SolverResult with_facing = solver.Solve(
      Vec2(0, 0), 0.7, {f.spacing_a, facing}, {1.0, 1.0}, tgt_tree);
  EXPECT_TRUE(with_facing.yaw_specified);
}

// A Facing whose decoded look-at point COINCIDES with the body has no defined bearing: it
// is left out of the yaw; with none left the yaw is unspecified (the driver faces its
// direction of motion). "In-body" is the v1 coincidence test only (delta < kEpsilon),
// not the receiver's declared body radius.
TEST(Yaw, ALookAtPointInsideTheBodyHasNoBearing) {
  Fixture f;
  const Vec2 src_actor(3.0, 6.0);
  const double src_yaw = -M_PI / 2;
  const RelationPayload fa = MakePayload(RelationKernelType::kAvatarFacing, src_actor,
                                         src_yaw, f.src_tree, 2, 200);
  const RelationPayload fb = MakePayload(RelationKernelType::kAvatarFacing, src_actor,
                                         src_yaw, f.src_tree, 3, 300);
  Scene tgt = f.src;
  tgt.b300 = Vec2(-4.0, 9.0);
  const ElementTree tgt_tree = MakeTree(tgt);
  NavigableSpatialDefinition nav(std::make_shared<Polygon2D>(std::vector<Vec2>{
      {-40, -40}, {40, -40}, {40, 40}, {-40, 40}}));
  nav.SetObstacles({}, /*clearance=*/0.2);
  RelationSolver solver;
  const SolverResult probe = solver.Solve(src_actor, src_yaw, {fa, fb}, {0.5, 0.5}, tgt_tree, &nav);
  ASSERT_EQ(probe.goals.size(), 2u);
  const Vec2 look_b = probe.goals[1].witness;

  // Standing 0.5 m from the look-at point of b: both Facings are in the yaw.
  const SolverResult outside = solver.Solve(look_b + Vec2(0.5, 0), src_yaw, {fa, fb}, {0.5, 0.5}, tgt_tree, &nav);
  EXPECT_FALSE(outside.goals[0].yaw_excluded);
  EXPECT_FALSE(outside.goals[1].yaw_excluded);
  EXPECT_TRUE(outside.yaw_specified);

  // Standing 0.05 m from it, well inside the declared 0.2 m body clearance: b still has a
  // bearing (D2), both Facings are in the yaw.
  const Vec2 inside_at = look_b + Vec2(0.05, 0);
  const SolverResult inside = solver.Solve(inside_at, src_yaw, {fa, fb}, {0.5, 0.5}, tgt_tree, &nav);
  EXPECT_FALSE(inside.goals[0].yaw_excluded);
  EXPECT_FALSE(inside.goals[1].yaw_excluded);
  EXPECT_TRUE(inside.yaw_specified);

  // Standing exactly on it: the look-at coincides with the body, b has no bearing, the
  // yaw is a's alone.
  const Vec2 on_it = look_b;
  const SolverResult coincident = solver.Solve(on_it, src_yaw, {fa, fb}, {0.5, 0.5}, tgt_tree, &nav);
  EXPECT_FALSE(coincident.goals[0].yaw_excluded);
  EXPECT_TRUE(coincident.goals[1].yaw_excluded);
  EXPECT_TRUE(coincident.yaw_specified);
  const SolverResult only_a = solver.Solve(on_it, src_yaw, {fa}, {1.0}, tgt_tree, &nav);
  EXPECT_NEAR(Wrap(coincident.yaw - only_a.yaw), 0.0, 1e-12);

  // With b alone and coincident with the body no Facing remains: the yaw is unspecified.
  const SolverResult none = solver.Solve(on_it, 0.7, {fb}, {1.0}, tgt_tree, &nav);
  EXPECT_TRUE(none.goals[0].yaw_excluded);
  EXPECT_FALSE(none.yaw_specified);
  // (v1 behaviour: the value carries no meaning -- the held yaw)
  EXPECT_NEAR(Wrap(none.yaw - 0.7), 0.0, 1e-12);

  // No receiver clearance and no distance fade: a 0.12 m look-at keeps its bearing, with
  // or without a declared body clearance.
  const Vec2 near_at = look_b + Vec2(0.12, 0);
  const SolverResult near_nav = solver.Solve(near_at, 0.7, {fb}, {1.0}, tgt_tree, &nav);
  EXPECT_FALSE(near_nav.goals[0].yaw_excluded);
  EXPECT_TRUE(near_nav.yaw_specified);
  const SolverResult no_nav = solver.Solve(near_at, 0.7, {fb}, {1.0}, tgt_tree);
  EXPECT_FALSE(no_nav.goals[0].yaw_excluded);
  EXPECT_TRUE(no_nav.yaw_specified);
}

// Reachability is judged in the same free space as feasibility. A footprint the source stands
// on is exempt: points inside it are free, and connected to each other (a body seated exactly
// on its proposal); without the exemption they are not a path (the footprint seals the room).
TEST(Clamp, ReachabilityUnderAnExemptionIsInTheSameFreeSpaceAsFeasibility) {
  NavigableSpatialDefinition nav(std::make_shared<Polygon2D>(std::vector<Vec2>{
      {-1, -1}, {1, -1}, {1, 1}, {-1, 1}}));
  // a footprint covering the whole room
  nav.SetObstacles({FreeSpaceObstacle{7, std::make_shared<Polygon2D>(std::vector<Vec2>{
                                             {-2, -2}, {2, -2}, {2, 2}, {-2, 2}})}},
                   /*clearance=*/0.2);
  const Vec2 a(-0.3, 0.1), b(0.4, -0.2);
  EXPECT_FALSE(nav.IsFree(a));
  EXPECT_TRUE(nav.IsFree(a, {7}));
  EXPECT_FALSE(nav.QueryReachability(a, b).reachable);
  EXPECT_TRUE(nav.QueryReachability(a, b, {7}, {}).reachable);
  EXPECT_TRUE(nav.QueryReachability(a, a, {7}, {}).reachable);  // sample == target
}

}  // namespace test
}  // namespace spatialblend
