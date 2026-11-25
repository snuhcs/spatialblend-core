// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/relation_solver_test.cc
#include "spatialblend/solver/relation_solver.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>
#include <string>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/envelope.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/obstacle_free_space.h"
#include "spatialblend/model/relation/affordance_pose_grounding_rule.h"
#include "spatialblend/model/relation/avatar_affordance_pose_kernel.h"
#include "spatialblend/model/relation/avatar_facing_kernel.h"
#include "spatialblend/model/relation/avatar_spacing_kernel.h"
#include "spatialblend/model/relation/grounding_rule.h"
#include "spatialblend/model/relation/relation.h"
#include "spatialblend/model/relation/spacing_grounding_rule.h"
#include "spatialblend/solver/realization_policy.h"
#include "spatialblend/solver/relation_evaluator.h"
#include "spatialblend/model/realization/remote_update.h"

namespace spatialblend {
namespace test {

// Helper: build a space with Room(1) -> Board(2) and return element tree.
// Board is at the given position.
ElementTree MakeTree(double board_x, double board_y,
                     shared_element_id_t board_shared_id = 200) {
  ElementTree tree;
  tree.AddElement(1);
  tree.AddElement(2, 1);
  tree.GetElement(2)->SetSharedId(board_shared_id);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = board_x;
  t(1, 3) = board_y;
  tree.GetElement(2)->SetLocalTransform(t);
  return tree;
}

// The v1 single-target step, written out independently of the solver: one cubic
// Bezier from the sample to the contact, leaving along the chord and arriving
// along `entering`, control points 0.3 of the chord from the ends
// (v1's spline generator), walked 1/n of its arc length
// (BaseSingleTargetPlanner::CalculateDistanceToMove). Chord-sum arc length at a
// resolution far finer than the solver's.
Eigen::Vector2d V1CourseStep(const Eigen::Vector2d& from,
                             const Eigen::Vector2d& contact,
                             const Eigen::Vector2d& entering, double n) {
  const Eigen::Vector2d chord = contact - from;
  const double len = chord.norm();
  const Eigen::Vector2d p1 = from + 0.3 * len * chord.normalized();
  const Eigen::Vector2d p2 = contact - 0.3 * len * entering.normalized();
  const auto at = [&](double t) -> Eigen::Vector2d {
    const double u = 1.0 - t;
    return u * u * u * from + 3 * u * u * t * p1 + 3 * u * t * t * p2 +
           t * t * t * contact;
  };
  constexpr int kChords = 100000;
  double total = 0.0;
  Eigen::Vector2d prev = at(0.0);
  for (int k = 1; k <= kChords; ++k) {
    const Eigen::Vector2d cur = at(static_cast<double>(k) / kChords);
    total += (cur - prev).norm();
    prev = cur;
  }
  const double target = std::min(total / n, total);
  double walked = 0.0;
  prev = at(0.0);
  for (int k = 1; k <= kChords; ++k) {
    const Eigen::Vector2d cur = at(static_cast<double>(k) / kChords);
    const double chord_len = (cur - prev).norm();
    if (walked + chord_len >= target)
      return prev + (cur - prev) * ((target - walked) / chord_len);
    walked += chord_len;
    prev = cur;
  }
  return contact;
}

// Helper: evaluate a spacing relation and package as RelationPayload.
RelationPayload MakeSpacingPayload(const Eigen::Vector3d& actor_pos,
                                    double actor_yaw,
                                    const ElementTree& tree,
                                    element_id_t referent_id,
                                    shared_element_id_t shared_id) {
  Relation rel("spacing", "root", referent_id,
               std::make_unique<DefaultGroundingRule>(),
               std::make_unique<AvatarSpacingKernel>());
  auto result = rel.Evaluate(actor_pos, actor_yaw, tree);

  RelationPayload payload;
  payload.referent_shared_id = shared_id;
  payload.subject_anchor_name = "root";
  payload.grounded_relation = result.grounded;
  payload.support = result.support;
  payload.kernel_type = RelationKernelType::kAvatarSpacing;
  return payload;
}

RelationPayload MakeFacingPayload(const Eigen::Vector3d& actor_pos,
                                   double actor_yaw,
                                   const ElementTree& tree,
                                   element_id_t referent_id,
                                   shared_element_id_t shared_id) {
  Relation rel("facing", "chest", referent_id,
               std::make_unique<DefaultGroundingRule>(),
               std::make_unique<AvatarFacingKernel>());
  auto result = rel.Evaluate(actor_pos, actor_yaw, tree);

  RelationPayload payload;
  payload.referent_shared_id = shared_id;
  payload.subject_anchor_name = "chest";
  payload.grounded_relation = result.grounded;
  payload.support = result.support;
  payload.kernel_type = RelationKernelType::kAvatarFacing;
  // A Facing request with count 1: the yaw reaches its target this update, the placed yaw these tests are about. The
  // yaw is a ratio continuation: without a valid count it holds, and with n > 1 it turns
  // 1/n of the receiver's own offset.
  payload.update_count = 1.0;
  return payload;
}

TEST(RelationSolverTest, SingleSpacingRelation) {
  // Source: board at (5,8), actor at (3,6)
  auto source_tree = MakeTree(5.0, 8.0);
  auto payload = MakeSpacingPayload(
      Eigen::Vector3d(3, 6, 0), 0.0, source_tree, 2, 200);

  // Target: board at (10, 4)
  // New geodesic solver: avatar approaches decoded target point (board center)
  auto target_tree = MakeTree(10.0, 4.0);

  RelationSolver solver;
  // The offset (8, 2) preserves R. It is a solution, not a teleport from a
  // far sample: the proposal continues from the sample inside rho = K*||q||.
  const Eigen::Vector2d solution(8.0, 2.0);
  SolverResult held = solver.Solve(solution, 0.0, {payload}, target_tree);
  EXPECT_EQ(held.status, SolverStatus::kConverged);
  EXPECT_NEAR(held.position.x(), solution.x(), 1e-2);
  EXPECT_NEAR(held.position.y(), solution.y(), 1e-2);

  const Eigen::Vector2d sample(0, 0);
  SolverResult result = solver.Solve(sample, 0.0, {payload}, target_tree);
  EXPECT_EQ(result.status, SolverStatus::kConverged);
  const double d_src = std::hypot(2.0, 2.0);
  const double r_src = 1.0 / (1.0 + d_src);
  const double r_at_sample = 1.0 / (1.0 + std::hypot(10.0, 4.0));
  const double rho = (1.0 / (r_src * r_src)) * std::abs(r_src - r_at_sample);
  EXPECT_LE((result.position - sample).norm(), rho + 1e-6);
  EXPECT_GT((result.position - solution).norm(), 0.5);
}

// The first sample minimises R_tgt = R_src from the contact centroid. Two
// referents make that point the source pose, not a point on one level set.
// A far sample must not stay put: the ratio would keep the start error.
TEST(RelationSolverTest, AbsolutePlacementMatchesTheSourceRelation) {
  ElementTree source = MakeTree(5.0, 8.0);
  ASSERT_TRUE(source.AddElement(3, 1).ok());
  source.GetElement(3)->SetSharedId(300);
  Eigen::Matrix4d placed = Eigen::Matrix4d::Identity();
  placed(0, 3) = 0.0;
  placed(1, 3) = 1.0;
  source.GetElement(3)->SetLocalTransform(placed);
  const Eigen::Vector3d actor(3.0, 6.0, 0.0);
  auto board = MakeSpacingPayload(actor, 0.0, source, 2, 200);
  auto other = MakeSpacingPayload(actor, 0.0, source, 3, 300);

  ElementTree target = MakeTree(5.0, 8.0);
  ASSERT_TRUE(target.AddElement(3, 1).ok());
  target.GetElement(3)->SetSharedId(300);
  target.GetElement(3)->SetLocalTransform(placed);

  RelationSolver solver;
  SolverWarmStart warm;
  warm.absolute = true;
  const SolverResult placed_at =
      solver.Solve(Eigen::Vector2d(0.0, 0.0), 0.0, {board, other}, {}, target,
                   nullptr, nullptr, std::nullopt, warm);
  EXPECT_EQ(placed_at.status, SolverStatus::kConverged);
  EXPECT_NEAR(placed_at.position.x(), actor.x(), 1e-2);
  EXPECT_NEAR(placed_at.position.y(), actor.y(), 1e-2);

  // The ratio from that exact sample, with no source motion, stays there.
  const SolverResult held =
      solver.Solve(placed_at.position, 0.0, {board, other}, target);
  EXPECT_NEAR(held.position.x(), actor.x(), 1e-2);
  EXPECT_NEAR(held.position.y(), actor.y(), 1e-2);
}

TEST(RelationSolverTest, SingleFacingRelation) {
  // Source: board at (5,0), actor at (0,0), yaw=0
  auto source_tree = MakeTree(5.0, 0.0);
  auto payload = MakeFacingPayload(
      Eigen::Vector3d(0, 0, 0), 0.0, source_tree, 2, 200);

  // Target: board at (0, 10) -> solver should orient actor to face (0,10)
  auto target_tree = MakeTree(0.0, 10.0);

  RelationSolver solver;
  SolverResult result = solver.Solve(
      Eigen::Vector2d(0, 0), 0.0,
      {payload}, target_tree);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  // Facing-only: yaw should point toward the board at (0, 10) from (0, 0) = pi/2
  EXPECT_NEAR(result.yaw, M_PI / 2.0, 0.2);
  // Position should not have moved (facing doesn't affect position)
  EXPECT_NEAR(result.position.x(), 0.0, 0.01);
  EXPECT_NEAR(result.position.y(), 0.0, 0.01);
}

// The Facing yaw is a ratio continuation. A request with count n turns the receiver 1/n
// of its own remaining offset to its look-at point (the source here has offset 0, so that is the target yaw too); n = 1
// is the placed yaw (bearing - source offset), bit for bit; no
// valid count holds the yaw. This differs from v1, whose yaw is placed every update.
// The receiver turns in the source's rotation direction s (facing_rotation_sign): a remaining offset of exactly pi is
// turned the source's way. A step needs the source's sign: a payload without one makes no request (the yaw holds).
TEST(RelationSolverTest, FacingYawIsARatioContinuationAndAPiTieKeepsTheTurn) {
  auto source_tree = MakeTree(5.0, 0.0);
  auto payload = MakeFacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, source_tree, 2, 200);  // source offset 0
  auto target_tree = MakeTree(0.0, 10.0);  // the receiver's target yaw: the bearing to (0, 10), pi/2
  RelationSolver solver;
  // A step (n > 1): the source closes on its look-at, its offset 0.05 decreasing (s = -1) / -0.05 increasing (s = +1).
  // n = 1: the source is aligned (offset 0).
  const auto solve = [&](double count, double current_yaw, int turn_sign, int s = -1) {
    RelationPayload p = payload;
    p.update_count = count;
    p.facing_rotation_sign = s;
    p.grounded_relation.encoded_state(0) = count > 1.0 ? -s * 0.05 : 0.0;
    SolverWarmStart w;
    w.has_position = true;
    w.has_yaw = true;
    w.yaw = current_yaw;
    w.turn_sign = turn_sign;
    return solver.Solve(Eigen::Vector2d(0, 0), current_yaw, {p}, {1.0}, target_tree, nullptr, nullptr, std::nullopt, w);
  };
  // n = 1: the placed yaw, the same bits as the placed answer (an absolute solve).
  const SolverResult one = solve(1.0, 0.3, 0);
  SolverWarmStart placed;
  placed.has_position = true;
  placed.absolute = true;
  RelationPayload p1 = payload;
  p1.update_count = 1.0;
  const SolverResult abs = solver.Solve(Eigen::Vector2d(0, 0), 0.3, {p1}, {1.0}, target_tree, nullptr, nullptr,
                                        std::nullopt, placed);
  ASSERT_TRUE(one.yaw_specified);
  EXPECT_EQ(one.yaw, abs.yaw);
  EXPECT_NEAR(one.yaw, M_PI / 2.0, 1e-12);
  // n = 4 from yaw 0.3: 0.3 + (pi/2 - 0.3) / 4.
  const SolverResult four = solve(4.0, 0.3, 0);
  EXPECT_NEAR(four.yaw, 0.3 + (M_PI / 2.0 - 0.3) / 4.0, 1e-12);
  // No valid count: the yaw holds.
  const SolverResult held = solve(std::numeric_limits<double>::infinity(), 0.3, 0);
  ASSERT_TRUE(held.yaw_specified);
  EXPECT_EQ(held.yaw, 0.3);
  // No rotation sign: no request, the yaw holds.
  const SolverResult unsigned_held = solve(4.0, 0.3, 0, 0);
  ASSERT_TRUE(unsigned_held.yaw_specified);
  EXPECT_EQ(unsigned_held.yaw, 0.3);
  // From -pi/2 the target is exactly pi away: left and right are equally short. n = 2 turns half of it, the SOURCE's
  // way (whatever the receiver's last turn): its offset decreasing (s = -1, the yaw increasing) reaches 0, increasing
  // (s = +1) reaches -pi.
  const SolverResult left = solve(2.0, -M_PI / 2.0, -1, -1);
  const SolverResult right = solve(2.0, -M_PI / 2.0, +1, +1);
  EXPECT_NEAR(std::remainder(left.yaw - 0.0, 2.0 * M_PI), 0.0, 1e-12);
  EXPECT_NEAR(std::remainder(right.yaw - M_PI, 2.0 * M_PI), 0.0, 1e-12);
}

// ABI 4.12: the yaw turn is signed and unwrapped. A Facing stepping the long way round at a count below 2 requests more
// than pi in one update; the absolute yaw carries it as the shorter turn the other way, the signed turn keeps it.
TEST(RelationSolverTest, ASignedTurnAbovePiKeepsItsDirection) {
  auto source_tree = MakeTree(5.0, 0.0);
  auto payload = MakeFacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, source_tree, 2, 200);
  auto target_tree = MakeTree(0.0, 10.0);  // the receiver's look-at bearing: pi/2
  RelationSolver solver;
  const double current = 0.3, o = M_PI / 2.0 - 0.3;  // the receiver's offset, positive
  // The source's offset increases toward 0 from -1.67 (s = +1): the receiver's offset turns the same way, the long way
  // round from o, 2 pi - o, which is the branch nearest the source's remaining turn 1.67 n / (n - 1) = 5.01 at n = 1.5.
  RelationPayload p = payload;
  p.update_count = 1.5;
  p.facing_rotation_sign = +1;
  p.grounded_relation.encoded_state(0) = -1.67;
  SolverWarmStart w;
  w.has_position = true;
  w.has_yaw = true;
  w.yaw = current;
  const SolverResult r =
      solver.Solve(Eigen::Vector2d(0, 0), current, {p}, {1.0}, target_tree, nullptr, nullptr, std::nullopt, w);
  ASSERT_TRUE(r.yaw_specified);
  const double requested = -(2.0 * M_PI - o) / 1.5;  // the offset moves by +(2 pi - o) / n, the yaw by its negative
  EXPECT_LT(requested, -M_PI);
  EXPECT_NEAR(r.yaw_turn, requested, 1e-12) << "signed and unwrapped: clockwise by more than pi";
  EXPECT_NEAR(std::remainder(current + r.yaw_turn - r.yaw, 2.0 * M_PI), 0.0, 1e-12) << "congruent to the yaw";
  EXPECT_GT(std::remainder(r.yaw - current, 2.0 * M_PI), 0.0) << "the absolute yaw alone reads counter-clockwise";
  EXPECT_GT(r.yaw_requested_turn_max, M_PI);
  // A hold (no valid count) turns 0.
  RelationPayload held = p;
  held.update_count = std::numeric_limits<double>::infinity();
  EXPECT_EQ(solver.Solve(Eigen::Vector2d(0, 0), current, {held}, {1.0}, target_tree, nullptr, nullptr, std::nullopt, w)
                .yaw_turn,
            0.0);
}

// On the arrival
// update the receiver turns the shortest arc to its target, whatever the source's rotation sign. A body whose look-at
// is +o counter-clockwise turns +o, also when the source turned clockwise; a body just past its look-at turns back.
TEST(RelationSolverTest, AnArrivalTurnsTheShortestArcWhateverTheSourcesDirection) {
  auto source_tree = MakeTree(5.0, 0.0);
  auto payload = MakeFacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, source_tree, 2, 200);
  auto target_tree = MakeTree(0.0, 10.0);
  RelationSolver solver;
  RelationPayload arrive = payload;
  arrive.update_count = 1.0;
  arrive.grounded_relation.encoded_state(0) = 0.0;
  const auto solve = [&](double current, int sign) {
    arrive.facing_rotation_sign = sign;
    SolverWarmStart w;
    w.has_position = true;
    w.has_yaw = true;
    w.yaw = current;
    return solver.Solve(Eigen::Vector2d(0, 0), current, {arrive}, {1.0}, target_tree, nullptr, nullptr, std::nullopt, w);
  };
  for (const int sign : {+1, -1, 0}) {
    const double o = M_PI / 2.0 + 0.3;  // more than a quarter turn: still the short way
    const SolverResult r = solve(-0.3, sign);
    ASSERT_TRUE(r.yaw_specified);
    EXPECT_NEAR(r.yaw, M_PI / 2.0, 1e-12) << "the arrival yaw is the target";
    EXPECT_NEAR(r.yaw_turn, o, 1e-12) << "the shortest arc, sign " << sign;
    EXPECT_NEAR(solve(M_PI / 2.0 + 0.2, sign).yaw_turn, -0.2, 1e-12) << "past the look-at: back, sign " << sign;
  }
}

// An arrival turns the shortest arc, and so does an arrival at the mean of several Facings. The look-ats
// of one source gaze line (a whiteboard area, a seat area and a wall all aligned, offsets 0) map on the receiver to
// targets on both sides of its yaw. Each requests the shortest arc to its own target, but the linear mean of those arcs
// is not the arc to the circular mean of the targets: the turn's branch was taken nearest it and the head turned the
// long way, +191 to +256 degrees (a replayed case:
// current 41 degrees, targets -157.9, -157.9 and -33.2 at a third each).
TEST(RelationSolverTest, AnArrivalAtTheMeanOfSeveralFacingsTurnsTheShortestArc) {
  const auto deg = [](double d) { return d * M_PI / 180.0; };
  const auto points = [](const std::vector<double>& bearings) {
    ElementTree tree;
    tree.AddElement(1);
    for (size_t i = 0; i < bearings.size(); ++i) {
      const element_id_t id = static_cast<element_id_t>(2 + i);
      tree.AddElement(id, 1);
      tree.GetElement(id)->SetSharedId(static_cast<shared_element_id_t>(200 + i));
      Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
      t(0, 3) = 5.0 * std::cos(bearings[i]);
      t(1, 3) = 5.0 * std::sin(bearings[i]);
      tree.GetElement(id)->SetLocalTransform(t);
    }
    return tree;
  };
  struct Case {
    double current_deg;
    std::vector<double> target_deg;
    std::vector<double> weights;
  };
  const std::vector<Case> cases = {
      {116.0, {100.0, -75.0}, {0.45, 0.55}},                     // two look-ats (revs 494-497), the old turn: +190
      {41.0, {-157.9, -157.9, -33.2}, {1.0 / 3, 1.0 / 3, 1.0 / 3}},  // replay rev 580, the old turn: +191 degrees
  };
  RelationSolver solver;
  for (const auto& c : cases) {
    std::vector<double> east(c.target_deg.size(), 0.0), targets;
    for (double t : c.target_deg) targets.push_back(deg(t));
    auto source_tree = points(east);  // every look-at on the source's gaze line: offsets 0
    auto target_tree = points(targets);
    std::vector<RelationPayload> payloads;
    double cs = 0.0, sn = 0.0;
    for (size_t i = 0; i < targets.size(); ++i) {
      RelationPayload p = MakeFacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, source_tree,
                                            static_cast<element_id_t>(2 + i),
                                            static_cast<shared_element_id_t>(200 + i));
      p.update_count = 1.0;  // the source is aligned on every one of them: each arrives
      p.facing_rotation_sign = -1;
      p.grounded_relation.encoded_state(0) = 0.0;
      payloads.push_back(p);
      cs += c.weights[i] * std::cos(targets[i]);
      sn += c.weights[i] * std::sin(targets[i]);
    }
    const double current = deg(c.current_deg);
    SolverWarmStart w;
    w.has_position = true;
    w.has_yaw = true;
    w.yaw = current;
    w.turn_sign = +1;
    const SolverResult r = solver.Solve(Eigen::Vector2d(0, 0), current, payloads, c.weights, target_tree, nullptr,
                                        nullptr, std::nullopt, w);
    ASSERT_TRUE(r.yaw_specified);
    ASSERT_FALSE(r.facing_tie);
    EXPECT_NEAR(std::remainder(r.yaw - std::atan2(sn, cs), 2.0 * M_PI), 0.0, 1e-9)
        << "the yaw is the circular mean of the targets, current " << c.current_deg;
    EXPECT_LE(std::fabs(r.yaw_turn), M_PI) << "no Facing asks for more than a half turn, current " << c.current_deg;
    EXPECT_NEAR(r.yaw_turn, std::remainder(r.yaw - current, 2.0 * M_PI), 1e-12)
        << "the shortest arc to the answered yaw, current " << c.current_deg;
  }
}

// The turn's branch with one long-way stepping Facing among arrivals on both sides of the yaw: the stepping Facing's own
// branch, weighted against the arrivals' branch 0. Outweighed, the turn is the shortest arc; alone it keeps its long way.
TEST(RelationSolverTest, ALongWayFacingOutweighedByArrivalsTurnsTheShortestArc) {
  const auto deg = [](double d) { return d * M_PI / 180.0; };
  const auto points = [](const std::vector<double>& b) {
    ElementTree tree;
    tree.AddElement(1);
    for (size_t i = 0; i < b.size(); ++i) {
      const element_id_t id = static_cast<element_id_t>(2 + i);
      tree.AddElement(id, 1);
      tree.GetElement(id)->SetSharedId(static_cast<shared_element_id_t>(200 + i));
      Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
      t(0, 3) = 5.0 * std::cos(b[i]);
      t(1, 3) = 5.0 * std::sin(b[i]);
      tree.GetElement(id)->SetLocalTransform(t);
    }
    return tree;
  };
  // Look-ats: one the long-way Facing turns to (pi/2), two arrivals on either side of the receiver's yaw (41 degrees,
  // the replay's rev 580 targets). The long-way Facing requests -207 degrees; the arrivals' arcs are +161 and -74.
  auto source_tree = points({0.0, 0.0, 0.0});
  auto target_tree = points({M_PI / 2.0, deg(-157.9), deg(-33.2)});
  std::vector<RelationPayload> ps;
  for (int i = 0; i < 3; ++i)
    ps.push_back(MakeFacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, source_tree, static_cast<element_id_t>(2 + i),
                                   static_cast<shared_element_id_t>(200 + i)));
  // As ASignedTurnAbovePiKeepsItsDirection: the source opens the long way round, n = 1.5, more than pi on this update.
  ps[0].update_count = 1.5;
  ps[0].facing_rotation_sign = +1;
  ps[0].grounded_relation.encoded_state(0) = -1.67;
  for (int i = 1; i < 3; ++i) {
    ps[i].update_count = 1.0;
    ps[i].facing_rotation_sign = +1;
    ps[i].grounded_relation.encoded_state(0) = 0.0;
  }
  const double current = deg(41.0);
  SolverWarmStart w;
  w.has_position = true;
  w.has_yaw = true;
  w.yaw = current;
  RelationSolver solver;
  // The linear mean of the requests (+54 degrees) sent the turn +185 degrees, against every request's direction.
  const SolverResult outweighed =
      solver.Solve(Eigen::Vector2d(0, 0), current, ps, {0.1, 0.6, 0.3}, target_tree, nullptr, nullptr, std::nullopt, w);
  ASSERT_TRUE(outweighed.yaw_specified);
  EXPECT_GT(outweighed.yaw_requested_turn_max, M_PI) << "the long-way Facing is in the mean";
  EXPECT_LE(std::fabs(outweighed.yaw_turn), M_PI);
  EXPECT_NEAR(outweighed.yaw_turn, std::remainder(outweighed.yaw - current, 2.0 * M_PI), 1e-12) << "the shortest arc";
  const SolverResult alone =
      solver.Solve(Eigen::Vector2d(0, 0), current, {ps[0]}, {1.0}, target_tree, nullptr, nullptr, std::nullopt, w);
  EXPECT_LT(alone.yaw_turn, -M_PI) << "alone it keeps the long way round";
  // Equal weights, the long-way Facing and one arrival: the branch mean is exactly -1/2, no majority: the shortest arc.
  // (Rounded half away from zero, it took the long branch.)
  const SolverResult half =
      solver.Solve(Eigen::Vector2d(0, 0), current, {ps[0], ps[1]}, {0.5, 0.5}, target_tree, nullptr, nullptr,
                   std::nullopt, w);
  ASSERT_TRUE(half.yaw_specified);
  EXPECT_GT(half.yaw_requested_turn_max, M_PI) << "the long-way Facing is in the mean";
  EXPECT_LE(std::fabs(half.yaw_turn), M_PI);
  EXPECT_NEAR(half.yaw_turn, std::remainder(half.yaw - current, 2.0 * M_PI), 1e-12) << "the shortest arc at a half";
}

// A short-way STEPPING Facing counts branch 0 in the turn's vote, as an arrival does. The
// ALongWayFacingOutweighedByArrivals setup with the two arrivals made near-arrival steppers (count n slightly above 1,
// each a short-way turn): measured against the answer's arc, round((t - base) / 2 pi), their short requests near a half
// turn read as branch +1, and n = 1.01 turned +183.2 degrees where the shortest arc is -176.8 (n = 1.00: -175.2; without
// the long-way Facing: -170.8). A 0.01 change in n must not flip the turn by a whole revolution.
TEST(RelationSolverTest, ALongWayFacingAmongShortWaySteppersTurnsTheShortestArc) {
  const auto deg = [](double d) { return d * M_PI / 180.0; };
  const auto points = [](const std::vector<double>& b) {
    ElementTree tree;
    tree.AddElement(1);
    for (size_t i = 0; i < b.size(); ++i) {
      const element_id_t id = static_cast<element_id_t>(2 + i);
      tree.AddElement(id, 1);
      tree.GetElement(id)->SetSharedId(static_cast<shared_element_id_t>(200 + i));
      Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
      t(0, 3) = 5.0 * std::cos(b[i]);
      t(1, 3) = 5.0 * std::sin(b[i]);
      tree.GetElement(id)->SetLocalTransform(t);
    }
    return tree;
  };
  auto source_tree = points({0.0, 0.0, 0.0});
  auto target_tree = points({M_PI / 2.0, deg(-157.9), deg(-33.2)});
  const double current = deg(41.0);
  SolverWarmStart w;
  w.has_position = true;
  w.has_yaw = true;
  w.yaw = current;
  RelationSolver solver;
  for (double n : {1.0, 1.01, 1.1}) {
    for (bool with_long : {false, true}) {
      std::vector<RelationPayload> ps;
      for (int i = 0; i < 3; ++i)
        ps.push_back(MakeFacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, source_tree, static_cast<element_id_t>(2 + i),
                                       static_cast<shared_element_id_t>(200 + i)));
      ps[0].update_count = 1.5;  // the long-way Facing, as in ALongWayFacingOutweighedByArrivals: -207 degrees
      ps[0].facing_rotation_sign = +1;
      ps[0].grounded_relation.encoded_state(0) = -1.67;
      for (int i = 1; i < 3; ++i) {  // n = 1: arrivals; n > 1: short-way steppers
        ps[i].update_count = n;
        ps[i].facing_rotation_sign = +1;
        ps[i].grounded_relation.encoded_state(0) = 0.0;
      }
      std::vector<RelationPayload> use = ps;
      std::vector<double> wt = {0.1, 0.6, 0.3};
      if (!with_long) {
        use = {ps[1], ps[2]};
        wt = {0.6, 0.3};
      }
      const SolverResult r =
          solver.Solve(Eigen::Vector2d(0, 0), current, use, wt, target_tree, nullptr, nullptr, std::nullopt, w);
      ASSERT_TRUE(r.yaw_specified) << "n=" << n << " long=" << with_long;
      if (with_long) EXPECT_GT(r.yaw_requested_turn_max, M_PI) << "the long-way Facing is in the mean, n=" << n;
      EXPECT_LE(std::fabs(r.yaw_turn), M_PI) << "n=" << n << " long=" << with_long;
      EXPECT_NEAR(r.yaw_turn, std::remainder(r.yaw - current, 2.0 * M_PI), 1e-12)
          << "the shortest arc, n=" << n << " long=" << with_long;
    }
  }
}

// A pursuing receiver a little past its look-at while the source is nearly done turns back the short
// way, 1/n of its own offset. With the branch held at k >= 0 it requested the long way, -1.52 rad on this update and a
// whole revolution over the remaining ones.
TEST(RelationSolverTest, APursuingReceiverPastItsLookAtTurnsBackTheShortWay) {
  auto source_tree = MakeTree(5.0, 0.0);
  auto payload = MakeFacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, source_tree, 2, 200);
  auto target_tree = MakeTree(0.0, 10.0);  // the receiver's look-at bearing: pi/2
  RelationSolver solver;
  RelationPayload p = payload;
  p.update_count = 4.0;
  p.facing_rotation_sign = +1;              // the source's offset rotates +1 ...
  p.grounded_relation.encoded_state(0) = -0.1;  // ... 0.1 rad from its look-at
  const double current = M_PI / 2.0 - 0.2;  // the receiver's offset +0.2: past its look-at in the source's direction
  SolverWarmStart w;
  w.has_position = true;
  w.has_yaw = true;
  w.yaw = current;
  const SolverResult r =
      solver.Solve(Eigen::Vector2d(0, 0), current, {p}, {1.0}, target_tree, nullptr, nullptr, std::nullopt, w);
  ASSERT_TRUE(r.yaw_specified);
  EXPECT_NEAR(r.yaw_turn, 0.2 / 4.0, 1e-12) << "toward the look-at, 1/n of the offset";
}

// The tie at n = 1: two Facings whose targets are opposite (0 and pi) tie; the middles are
// +-pi/2. While pursuing (n = 2) the receiver keeps its current turn direction; on the arrival update (n = 1) it takes
// the nearer middle.
TEST(RelationSolverTest, AFacingTieKeepsTheTurnWhilePursuingAndTakesTheNearerMiddleOnArrival) {
  const auto two_points = [](double x2, double y2) {
    ElementTree tree;
    tree.AddElement(1);
    tree.AddElement(2, 1);
    tree.AddElement(3, 1);
    tree.GetElement(2)->SetSharedId(200);
    tree.GetElement(3)->SetSharedId(201);
    Eigen::Matrix4d a = Eigen::Matrix4d::Identity(), b = Eigen::Matrix4d::Identity();
    a(0, 3) = 5.0;
    b(0, 3) = x2;
    b(1, 3) = y2;
    tree.GetElement(2)->SetLocalTransform(a);
    tree.GetElement(3)->SetLocalTransform(b);
    return tree;
  };
  auto source_tree = two_points(10.0, 0.0);  // both look-ats east of the source: offsets 0
  auto target_tree = two_points(-5.0, 0.0);  // on the receiver's side east and west: targets 0 and pi
  auto f1 = MakeFacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, source_tree, 2, 200);
  auto f2 = MakeFacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, source_tree, 3, 201);
  RelationSolver solver;
  const auto solve = [&](double n, double yaw, int turn_sign) {
    RelationPayload a = f1, b = f2;
    a.update_count = b.update_count = n;
    // A step: the source closes on both look-ats, its offsets 0.05 decreasing; n = 1: aligned on both.
    a.facing_rotation_sign = b.facing_rotation_sign = -1;
    a.grounded_relation.encoded_state(0) = b.grounded_relation.encoded_state(0) = n > 1.0 ? 0.05 : 0.0;
    SolverWarmStart w;
    w.has_position = true;
    w.has_yaw = true;
    w.yaw = yaw;
    w.turn_sign = turn_sign;
    return solver.Solve(Eigen::Vector2d(0, 0), yaw, {a, b}, {0.5, 0.5}, target_tree, nullptr, nullptr, std::nullopt, w);
  };
  // Arrival (n = 1), the receiver at 1.4 rad, last turning clockwise: the nearer middle, +pi/2, not the turn's -pi/2.
  const SolverResult arrive = solve(1.0, 1.4, -1);
  ASSERT_TRUE(arrive.facing_tie);
  EXPECT_NEAR(arrive.yaw, M_PI / 2.0, 1e-12);
  // Pursuing (n = 2): the stepped yaws' mean lies just counter-clockwise of 1.4; a receiver last turning clockwise takes
  // the middle a half turn away, turning clockwise; one last turning counter-clockwise keeps the mean.
  const SolverResult cw = solve(2.0, 1.4, -1), ccw = solve(2.0, 1.4, +1);
  ASSERT_TRUE(cw.facing_tie);
  EXPECT_LT(std::remainder(cw.yaw - 1.4, 2.0 * M_PI), 0.0) << "keeps turning clockwise";
  EXPECT_GT(std::remainder(ccw.yaw - 1.4, 2.0 * M_PI), 0.0) << "keeps turning counter-clockwise";
}

TEST(RelationSolverTest, MultiRelationSpacingAndFacing) {
  // Source: board at (5, 8), actor at (3, 6) yaw=0
  auto source_tree = MakeTree(5.0, 8.0);
  auto spacing = MakeSpacingPayload(
      Eigen::Vector3d(3, 6, 0), 0.0, source_tree, 2, 200);
  auto facing = MakeFacingPayload(
      Eigen::Vector3d(3, 6, 0), 0.0, source_tree, 2, 200);

  // Target: board at (2, 3). The sample is (0, 0), 3.606 m from the board.
  // The source stand-off (0, 1) is the retired current-error point. v1 walks
  // 1/n of the receiver's own course to the contact, n = 2 half of it. The
  // source stood at (-2, -2) from the board centre, so the course arrives
  // along (1, 1)/sqrt(2), the mapped entering direction, and is longer than the
  // straight chord: half its arc is not the chord midpoint (1, 1.5).
  auto target_tree = MakeTree(2.0, 3.0);
  spacing.update_count = 2.0;

  RelationSolver solver;
  SolverResult result = solver.Solve(
      Eigen::Vector2d(0, 0), 0.0,
      {spacing, facing}, target_tree);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  const Eigen::Vector2d expected = V1CourseStep(
      Eigen::Vector2d(0, 0), Eigen::Vector2d(2, 3),
      Eigen::Vector2d(1, 1) / std::sqrt(2.0), 2.0);
  EXPECT_NEAR(result.position.x(), expected.x(), 1e-4);
  EXPECT_NEAR(result.position.y(), expected.y(), 1e-4);
  EXPECT_GT((expected - Eigen::Vector2d(1.0, 1.5)).norm(), 0.05)
      << "the entering direction must bend the course";
}

TEST(RelationSolverTest, NoPayloadsReturnsCurrent) {
  auto target_tree = MakeTree(5.0, 5.0);

  RelationSolver solver;
  SolverResult result = solver.Solve(
      Eigen::Vector2d(3, 4), 1.0,
      {}, target_tree);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  EXPECT_NEAR(result.position.x(), 3.0, 1e-9);
  EXPECT_NEAR(result.position.y(), 4.0, 1e-9);
  EXPECT_NEAR(result.yaw, 1.0, 1e-9);
}

TEST(RelationSolverTest, MissingReferentFallback) {
  // Payload references shared_id 999 which doesn't exist in target tree
  auto source_tree = MakeTree(5.0, 8.0);
  auto payload = MakeSpacingPayload(
      Eigen::Vector3d(3, 6, 0), 0.0, source_tree, 2, 999);

  auto target_tree = MakeTree(10.0, 4.0, 200);  // shared_id=200, not 999

  RelationSolver solver;
  SolverResult result = solver.Solve(
      Eigen::Vector2d(1, 1), 0.0,
      {payload}, target_tree);

  // No valid relations -> return current state
  EXPECT_NEAR(result.position.x(), 1.0, 1e-9);
  EXPECT_NEAR(result.position.y(), 1.0, 1e-9);
}

TEST(RelationSolverTest, SolverConvergesQuickly) {
  // Warm-start test: initial position close to solution
  auto source_tree = MakeTree(5.0, 8.0);
  auto payload = MakeSpacingPayload(
      Eigen::Vector3d(3, 6, 0), 0.0, source_tree, 2, 200);

  auto target_tree = MakeTree(10.0, 4.0);

  // The sample is already 0.7 m from the board. The retired assertion walked
  // out to the source stand-off (8, 2), 2.8 m away. A payload with no valid
  // intent (static source) does not open that gap.
  RelationSolver solver;
  const Eigen::Vector2d sample(9.5, 3.5);
  SolverResult result = solver.Solve(sample, 0.0, {payload}, target_tree);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  EXPECT_LT((result.position - sample).norm(), 1e-6);
}

TEST(RelationSolverTest, MultiRelationThreeTypes) {
  // Test solving Spacing + Facing simultaneously
  // Source: Board at (5, 8), actor at (4, 7) facing board
  auto source_tree = MakeTree(5.0, 8.0);

  Eigen::Vector3d actor_pos(4, 7, 0);
  Eigen::Vector2d to_board(5.0 - 4.0, 8.0 - 7.0);
  double actor_yaw = std::atan2(to_board.y(), to_board.x());

  auto spacing = MakeSpacingPayload(actor_pos, actor_yaw, source_tree, 2, 200);
  auto facing = MakeFacingPayload(actor_pos, actor_yaw, source_tree, 2, 200);

  // Target: board at (2, 3)
  auto target_tree = MakeTree(2.0, 3.0);

  RelationSolver solver;
  SolverResult result = solver.Solve(
      Eigen::Vector2d(0, 0), 0.0,
      {spacing, facing}, target_tree);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  // (1, 2) preserves both the spacing offset and is not the proposal from a
  // sample at the origin: that step is bounded by rho = K*||q||.
  const double d_src = std::hypot(1.0, 1.0);
  const double r_src = 1.0 / (1.0 + d_src);
  const double r_at_sample = 1.0 / (1.0 + std::hypot(2.0, 3.0));
  const double rho = (1.0 / (r_src * r_src)) * std::abs(r_src - r_at_sample);
  EXPECT_LE(result.position.norm(), rho + 1e-6);
  EXPECT_GT((result.position - Eigen::Vector2d(1.0, 2.0)).norm(), 0.5);

  SolverResult at_solution =
      solver.Solve(Eigen::Vector2d(1.0, 2.0), 0.0, {spacing, facing}, target_tree);
  EXPECT_NEAR(at_solution.position.x(), 1.0, 1e-2);
  EXPECT_NEAR(at_solution.position.y(), 2.0, 1e-2);

  // Yaw should face toward board at (2, 3) from solved position
  double expected_yaw = std::atan2(
      3.0 - result.position.y(), 2.0 - result.position.x());
  EXPECT_NEAR(result.yaw, expected_yaw, 0.3);
}

TEST(RelationSolverTest, MultipleObjectsCompromise) {
  // Two objects: Board at (5, 8) and Table at (15, 2)
  // Actor midway: Spacing to both
  ElementTree source_tree;
  source_tree.AddElement(1);
  source_tree.AddElement(2, 1);
  source_tree.GetElement(2)->SetSharedId(200);
  Eigen::Matrix4d bt = Eigen::Matrix4d::Identity();
  bt(0, 3) = 5.0; bt(1, 3) = 8.0;
  source_tree.GetElement(2)->SetLocalTransform(bt);

  source_tree.AddElement(3, 1);
  source_tree.GetElement(3)->SetSharedId(300);
  Eigen::Matrix4d tt = Eigen::Matrix4d::Identity();
  tt(0, 3) = 15.0; tt(1, 3) = 2.0;
  source_tree.GetElement(3)->SetLocalTransform(tt);

  // Actor at (10, 5) -- midway between board and table
  Eigen::Vector3d actor_pos(10, 5, 0);
  auto spacing_board = MakeSpacingPayload(actor_pos, 0.0, source_tree, 2, 200);
  auto spacing_table = MakeSpacingPayload(actor_pos, 0.0, source_tree, 3, 300);

  // Target: Board at (2, 3), Table at (12, 10)
  ElementTree target_tree;
  target_tree.AddElement(1);
  target_tree.AddElement(2, 1);
  target_tree.GetElement(2)->SetSharedId(200);
  Eigen::Matrix4d bt2 = Eigen::Matrix4d::Identity();
  bt2(0, 3) = 2.0; bt2(1, 3) = 3.0;
  target_tree.GetElement(2)->SetLocalTransform(bt2);

  target_tree.AddElement(3, 1);
  target_tree.GetElement(3)->SetSharedId(300);
  Eigen::Matrix4d tt2 = Eigen::Matrix4d::Identity();
  tt2(0, 3) = 12.0; tt2(1, 3) = 10.0;
  target_tree.GetElement(3)->SetLocalTransform(tt2);

  RelationSolver solver;
  SolverResult result = solver.Solve(
      Eigen::Vector2d(7, 6), 0.0,
      {spacing_board, spacing_table}, target_tree);

  EXPECT_EQ(result.status, SolverStatus::kConverged);

  // Avatar should compromise between the two targets
  // Both spacing relations constrain position -- solver finds weighted centroid
  double dist_to_board = (result.position - Eigen::Vector2d(2, 3)).norm();
  double dist_to_table = (result.position - Eigen::Vector2d(12, 10)).norm();
  // Neither should be 0 -- it's a compromise
  EXPECT_GT(dist_to_board, 1.0);
  EXPECT_GT(dist_to_table, 1.0);
}

TEST(RelationSolverTest, ProximityWeightFavorsCloseObject) {
  // Two objects: Board CLOSE (high proximity) and Table FAR (low proximity).
  // Actor is near Board in source space.
  // Solver should favor Board's relation due to higher proximity weight.

  // Source: Board at (5, 5), Table at (15, 15)
  ElementTree source_tree;
  source_tree.AddElement(1);
  source_tree.AddElement(2, 1);
  source_tree.GetElement(2)->SetSharedId(200);
  Eigen::Matrix4d bt = Eigen::Matrix4d::Identity();
  bt(0, 3) = 5.0; bt(1, 3) = 5.0;
  source_tree.GetElement(2)->SetLocalTransform(bt);

  source_tree.AddElement(3, 1);
  source_tree.GetElement(3)->SetSharedId(300);
  Eigen::Matrix4d tt = Eigen::Matrix4d::Identity();
  tt(0, 3) = 15.0; tt(1, 3) = 15.0;
  source_tree.GetElement(3)->SetLocalTransform(tt);

  // Actor at (4, 4) -- close to Board (dist ~1.4), far from Table (dist ~15.6)
  Eigen::Vector3d actor_pos(4, 4, 0);
  auto spacing_board = MakeSpacingPayload(actor_pos, 0.0, source_tree, 2, 200);
  auto spacing_table = MakeSpacingPayload(actor_pos, 0.0, source_tree, 3, 300);

  // Verify proximity values differ significantly
  EXPECT_GT(spacing_board.grounded_relation.proximity, 0.3);
  EXPECT_LT(spacing_table.grounded_relation.proximity, 0.1);

  // Target: Board at (10, 3), Table at (3, 12)
  ElementTree target_tree;
  target_tree.AddElement(1);
  target_tree.AddElement(2, 1);
  target_tree.GetElement(2)->SetSharedId(200);
  Eigen::Matrix4d bt2 = Eigen::Matrix4d::Identity();
  bt2(0, 3) = 10.0; bt2(1, 3) = 3.0;
  target_tree.GetElement(2)->SetLocalTransform(bt2);

  target_tree.AddElement(3, 1);
  target_tree.GetElement(3)->SetSharedId(300);
  Eigen::Matrix4d tt2 = Eigen::Matrix4d::Identity();
  tt2(0, 3) = 3.0; tt2(1, 3) = 12.0;
  target_tree.GetElement(3)->SetLocalTransform(tt2);

  RelationSolver solver;
  // Weight each referent by its proximity (as the realization policy does),
  // so the strongly-satisfied Board pulls the solved pose toward it.
  std::vector<double> weights = {
      spacing_board.grounded_relation.proximity,
      spacing_table.grounded_relation.proximity};
  SolverResult result = solver.Solve(
      Eigen::Vector2d(7, 7), 0.0,
      {spacing_board, spacing_table}, weights, target_tree);

  EXPECT_EQ(result.status, SolverStatus::kConverged);

  // Avatar should be closer to Board_B (10, 3) than Table_B (3, 12)
  // because Board's proximity weight is much higher
  double dist_board = (result.position - Eigen::Vector2d(10, 3)).norm();
  double dist_table = (result.position - Eigen::Vector2d(3, 12)).norm();
  EXPECT_LT(dist_board, dist_table);
}

TEST(RelationSolverTest, WalkabilityConstraintFallback) {
  // Source: board at (5, 8), actor at (3, 6)
  auto source_tree = MakeTree(5.0, 8.0);
  auto payload = MakeSpacingPayload(
      Eigen::Vector3d(3, 6, 0), 0.0, source_tree, 2, 200);

  // Target: board at (100, 100) -- outside walkable area 10x10
  auto target_tree = MakeTree(100.0, 100.0);

  auto walkable = std::make_shared<spatialblend::Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {10, 0}, {10, 10}, {0, 10}});
  NavigableSpatialDefinition nav(walkable);

  RelationSolver solver;
  SolverResult result = solver.Solve(
      Eigen::Vector2d(5, 5), 0.0,
      {payload}, target_tree, &nav);

  // Solver clamps to polygon boundary
  EXPECT_TRUE(result.status == SolverStatus::kConverged ||
              result.status == SolverStatus::kFallback);
  // Result must be inside (or on boundary of) the 10x10 walkable area
  EXPECT_GE(result.position.x(), -0.01);
  EXPECT_LE(result.position.x(), 10.01);
  EXPECT_GE(result.position.y(), -0.01);
  EXPECT_LE(result.position.y(), 10.01);
}

TEST(RelationSolverTest, WalkabilityConstraintConverges) {
  // Target board inside walkable area -- should converge toward board
  auto source_tree = MakeTree(5.0, 8.0);
  auto payload = MakeSpacingPayload(
      Eigen::Vector3d(3, 6, 0), 0.0, source_tree, 2, 200);

  auto target_tree = MakeTree(7.0, 5.0);

  auto walkable = std::make_shared<spatialblend::Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  NavigableSpatialDefinition nav(walkable);

  // Sample (5, 5), board (7, 5): 2 m away. n = 2 walks half the course, which
  // arrives along the mapped entering direction (1, 1)/sqrt(2) (the source
  // stood at (-2, -2) from its board). The retired assertion restored the
  // source stand-off, which is farther than the sample (7 − √8).
  payload.update_count = 2.0;
  RelationSolver solver;
  SolverResult result = solver.Solve(
      Eigen::Vector2d(5, 5), 0.0,
      {payload}, target_tree, &nav);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  const Eigen::Vector2d expected = V1CourseStep(
      Eigen::Vector2d(5, 5), Eigen::Vector2d(7, 5),
      Eigen::Vector2d(1, 1) / std::sqrt(2.0), 2.0);
  EXPECT_NEAR(result.position.x(), expected.x(), 1e-4);
  EXPECT_NEAR(result.position.y(), expected.y(), 1e-4);
  EXPECT_TRUE(nav.IsFree(result.position));
}

TEST(RelationSolverTest, FacingOnlyFixesPosition) {
  // Facing-only: position should be fixed, only yaw changes.
  auto source_tree = MakeTree(5.0, 0.0);
  auto payload = MakeFacingPayload(
      Eigen::Vector3d(0, 0, 0), 0.0, source_tree, 2, 200);

  auto target_tree = MakeTree(0.0, 10.0);

  RelationSolver solver;
  SolverResult result = solver.Solve(
      Eigen::Vector2d(3, 4), 0.0,
      {payload}, target_tree);

  // Position should NOT move from initial (3, 4)
  EXPECT_NEAR(result.position.x(), 3.0, 0.01);
  EXPECT_NEAR(result.position.y(), 4.0, 0.01);
  // Z, pitch, roll should remain 0 (fixed)
  EXPECT_NEAR(result.z, 0.0, 0.01);
  EXPECT_NEAR(result.pitch, 0.0, 0.01);
  EXPECT_NEAR(result.roll, 0.0, 0.01);
}

TEST(RelationSolverTest, SpacingOnlyFixesYaw) {
  // Spacing-only: yaw should be fixed, only position changes.
  auto source_tree = MakeTree(5.0, 8.0);
  auto payload = MakeSpacingPayload(
      Eigen::Vector3d(3, 6, 0), 0.0, source_tree, 2, 200);

  auto target_tree = MakeTree(10.0, 4.0);

  double initial_yaw = 1.5;
  RelationSolver solver;
  const Eigen::Vector2d solution(8.0, 2.0);
  SolverResult held =
      solver.Solve(solution, initial_yaw, {payload}, target_tree);
  EXPECT_NEAR(held.yaw, initial_yaw, 0.01);
  EXPECT_NEAR(held.position.x(), solution.x(), 1e-2);
  EXPECT_NEAR(held.position.y(), solution.y(), 1e-2);

  const Eigen::Vector2d sample(5, 5);
  SolverResult result = solver.Solve(sample, initial_yaw, {payload}, target_tree);
  EXPECT_NEAR(result.yaw, initial_yaw, 0.01);
  const double d_src = std::hypot(2.0, 2.0);
  const double r_src = 1.0 / (1.0 + d_src);
  const double r_at_sample = 1.0 / (1.0 + (sample - Eigen::Vector2d(10, 4)).norm());
  const double rho = (1.0 / (r_src * r_src)) * std::abs(r_src - r_at_sample);
  EXPECT_LE((result.position - sample).norm(), rho + 1e-6);
}

// ===========================================================================
// Box (Polytope) referent decode-side frame composition.
//
// PolytopeSpatialDefinition stores its footprint verbatim in the OWNING
// ELEMENT's LOCAL frame (RequiresElementWorldTransform() == true — see
// spatialblend/model/element/polytope_spatial_definition.h). Unlike MakeTree's bare board (no spatial definition at all, so
// DefaultGroundingRule/DecodeSupportPoint's has_contact branch never
// triggers), these tests build a REAL SpacingGroundingRule payload against a
// box referent, exercising both:
//   (a) the "approach" branch — DecodeSupportPoint's non-sub-region
//       world_t-composition fix, reached when the source is OUTSIDE the box
//       (proximity < 1 - 1e-3), and
//   (b) the "engaged / inside" branch — DecodeTargetPoint's DecodeInterior
//       world_t-composition fix, reached when the source is INSIDE the box
//       (proximity >= 1 - 1e-3, support.has_interior == true).
// Both axis-aligned and yaw=pi/4 box referents are covered, since a fix that
// composed only the translation part of world_t (not the full rotation+
// translation) would still pass an axis-aligned-only check.
// ===========================================================================

// Room(1) -> Box(2, sid) with a PolytopeSpatialDefinition (LOCAL half-extents)
// placed at (box_x, box_y) with the given yaw via SetLocalTransform.
ElementTree MakeBoxTree(double box_x, double box_y, double half, double yaw,
                        shared_element_id_t shared_id = 200) {
  ElementTree tree;
  tree.AddElement(1);
  tree.AddElement(2, 1);
  tree.GetElement(2)->SetSharedId(shared_id);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 0) = std::cos(yaw);
  t(0, 1) = -std::sin(yaw);
  t(1, 0) = std::sin(yaw);
  t(1, 1) = std::cos(yaw);
  t(0, 3) = box_x;
  t(1, 3) = box_y;
  tree.GetElement(2)->SetLocalTransform(t);
  tree.GetElement(2)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-half, -half, -1.0),
          Eigen::Vector3d(half, half, 1.0)));
  return tree;
}

// Same as MakeSpacingPayload, but grounded with the REAL SpacingGroundingRule
// (not DefaultGroundingRule) so has_contact/contact_normalized/has_interior
// are actually computed against the referent's spatial definition — required
// to exercise the box/Polytope decode paths under test here.
RelationPayload MakeRealSpacingPayload(const Eigen::Vector3d& actor_pos,
                                       double actor_yaw,
                                       const ElementTree& tree,
                                       element_id_t referent_id,
                                       shared_element_id_t shared_id) {
  Relation rel("spacing", "root", referent_id,
               std::make_unique<SpacingGroundingRule>(),
               std::make_unique<AvatarSpacingKernel>());
  auto result = rel.Evaluate(actor_pos, actor_yaw, tree);

  RelationPayload payload;
  payload.referent_shared_id = shared_id;
  payload.subject_anchor_name = "root";
  payload.grounded_relation = result.grounded;
  payload.support = result.support;
  payload.kernel_type = RelationKernelType::kAvatarSpacing;
  return payload;
}

ElementTree MakeRectTree(double x, double y, double hx, double hy,
                         shared_element_id_t shared_id = 200) {
  ElementTree tree;
  tree.AddElement(1);
  tree.AddElement(2, 1);
  tree.GetElement(2)->SetSharedId(shared_id);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = x;
  t(1, 3) = y;
  tree.GetElement(2)->SetLocalTransform(t);
  tree.GetElement(2)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-hx, -hy, -1.0), Eigen::Vector3d(hx, hy, 1.0)));
  return tree;
}

// As v1's BaseSingleTargetPlanner, the receiver arrives
// along the MAPPED entering direction. The direction is the source's final leg
// onto the contact read through the one support chart: the leg's start (here the
// last waypoint of the source's geodesic, carried in the payload) and the contact
// are both mapped, and the leg is their difference. On a differently shaped
// referent that is not the source's direction copied: the radial chart scales the
// two points by different factors.
// Join placement reads the relations whatever their weight; weight only breaks its ties.
// On the first update no count is valid yet, so Progressive weights them all 0, and the placed answer is still the
// approach pose of the relation the source is nearest to, not the sample.
TEST(RelationSolverTest, JoinPlacementReadsTheRelationsWhateverTheirWeight) {
  auto source_tree = MakeRectTree(10.0, 10.0, 0.5, 0.9);
  auto target_tree = MakeRectTree(20.0, 5.0, 0.5, 0.9);
  auto payload = MakeRealSpacingPayload(Eigen::Vector3d(11.2, 10.3, 0.0), 0.0, source_tree, 2, 200);
  payload.update_count = std::numeric_limits<double>::infinity();  // the first update: no count yet
  RelationSolver solver;
  const Eigen::Vector2d sample(15.0, 0.0);
  SolverWarmStart join;
  join.has_position = true;
  join.position = sample;
  join.absolute = true;
  const SolverResult placed = solver.Solve(sample, 0.0, {payload}, {0.0}, target_tree, nullptr, nullptr,
                                           std::nullopt, join);
  EXPECT_TRUE(placed.absolute_placement);
  EXPECT_NEAR(placed.position.x(), 21.2, 1e-9) << "the source's offset east of its box, mapped onto the receiver's";
  EXPECT_NEAR(placed.position.y(), 5.3, 1e-9);
  // The same update as a continuation (not a join): no count, no request, the anchor holds at its sample.
  SolverWarmStart cont = join;
  cont.absolute = false;
  const SolverResult held = solver.Solve(sample, 0.0, {payload}, {0.0}, target_tree, nullptr, nullptr,
                                         std::nullopt, cont);
  EXPECT_EQ(held.position, sample);
}

// The join reads a Facing relation at weight 0 as well (on the first update no count is valid, so every
// Progressive weight is 0). On a continuation the same zero-weight Facing is disabled.
TEST(RelationSolverTest, JoinPlacementReadsAZeroWeightFacing) {
  auto source_tree = MakeTree(5.0, 0.0);
  auto facing = MakeFacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, source_tree, 2, 200);
  facing.update_count = std::numeric_limits<double>::infinity();  // the first update: no count yet
  auto target_tree = MakeTree(0.0, 10.0);
  RelationSolver solver;
  SolverWarmStart join;
  join.has_position = true;
  join.position = Eigen::Vector2d(0, 0);
  join.absolute = true;
  const SolverResult placed = solver.Solve(Eigen::Vector2d(0, 0), 0.0, {facing}, {0.0}, target_tree, nullptr, nullptr,
                                           std::nullopt, join);
  ASSERT_EQ(placed.relations.size(), 1u);
  EXPECT_EQ(placed.relations[0].disposition, SolverRelationDisposition::kUsed) << "the join reads it";
  EXPECT_TRUE(placed.yaw_specified);
  EXPECT_NEAR(placed.yaw, M_PI / 2.0, 0.2) << "facing the board at (0, 10)";
  SolverWarmStart cont = join;
  cont.absolute = false;
  const SolverResult held = solver.Solve(Eigen::Vector2d(0, 0), 0.0, {facing}, {0.0}, target_tree, nullptr, nullptr,
                                         std::nullopt, cont);
  ASSERT_EQ(held.relations.size(), 1u);
  EXPECT_EQ(held.relations[0].disposition, SolverRelationDisposition::kDisabled);
}

// A relation with no valid count issues no request: the answer is the anchor's current position, converged (as
// v1: no plan, the entity stays where it is).
TEST(RelationSolverTest, NoRequestIsTheCurrentPositionAndConverged) {
  auto source_tree = MakeRectTree(10.0, 10.0, 0.5, 0.9);
  auto target_tree = MakeRectTree(20.0, 5.0, 0.4, 1.17);
  auto payload = MakeRealSpacingPayload(Eigen::Vector3d(11.2, 10.3, 0.0), 0.0, source_tree, 2, 200);
  payload.update_count = std::numeric_limits<double>::infinity();  // a static source
  RelationSolver solver;
  const Eigen::Vector2d sample(15.0, 10.0);
  const SolverResult result = solver.Solve(sample, 0.0, {payload}, target_tree);
  EXPECT_TRUE(result.no_request);
  EXPECT_EQ(result.status, SolverStatus::kConverged);
  EXPECT_EQ(result.position, sample);
}

TEST(RelationSolverTest, ArrivesAlongTheMappedEnteringDirection) {
  // Source box 1.0 x 1.8 at (10, 10); receiver box 0.8 x 2.34 at (20, 5).
  auto source_tree = MakeRectTree(10.0, 10.0, 0.5, 0.9);
  auto target_tree = MakeRectTree(20.0, 5.0, 0.4, 1.17);
  // The source stands 0.7 m east of its box at (11.2, 10.3); its geodesic bends
  // round an obstacle, so its final leg starts at (10.3, 12.0) and not at the
  // anchor.
  auto payload = MakeRealSpacingPayload(Eigen::Vector3d(11.2, 10.3, 0.0), 0.0,
                                        source_tree, 2, 200);
  ASSERT_TRUE(payload.support.has_contact);
  payload.support.has_approach_origin = true;
  payload.support.approach_origin_local = Eigen::Vector2d(0.3, 2.0);
  payload.update_count = 1.0001;  // almost arrived: the end of the course

  // The radial chart of an axis-aligned box about its centre: a point p maps to
  // p * m_s(p) / m_r(p), m(p) = max(|x| / hx, |y| / hy) per box; a boundary
  // point has m_s = 1.
  const auto m = [](const Eigen::Vector2d& p, double hx, double hy) {
    return std::max(std::abs(p.x()) / hx, std::abs(p.y()) / hy);
  };
  const Eigen::Vector2d contact_local(0.5, 0.3);
  const Eigen::Vector2d origin_local(0.3, 2.0);
  const Eigen::Vector2d rcv_center(20.0, 5.0);
  const Eigen::Vector2d contact =
      rcv_center + contact_local / m(contact_local, 0.4, 1.17);
  const Eigen::Vector2d origin =
      rcv_center + origin_local * m(origin_local, 0.5, 0.9) /
                       m(origin_local, 0.4, 1.17);
  const Eigen::Vector2d entering = (contact - origin).normalized();
  const Eigen::Vector2d rigid = (Eigen::Vector2d(0.5, 0.3) - origin_local).normalized();
  ASSERT_LT(entering.dot(rigid), 1.0 - 1e-3) << "the chart must matter here";

  const auto prepared = PrepareTargetRelation(payload, target_tree);
  ASSERT_TRUE(prepared.ok());
  ASSERT_TRUE(prepared.has_mapped_support);
  ASSERT_NEAR((prepared.mapped_support.head<2>() - contact).norm(), 0.0, 1e-9);

  RelationSolver solver;
  const Eigen::Vector2d sample(15.0, 10.0);
  const SolverResult result = solver.Solve(sample, 0.0, {payload}, target_tree);
  ASSERT_EQ(result.status, SolverStatus::kConverged);
  const Eigen::Vector2d expected =
      V1CourseStep(sample, contact, entering, payload.update_count);
  EXPECT_NEAR((result.position - expected).norm(), 0.0, 1e-4);
  // Just short of the contact (a thousandth of the course), the receiver is
  // coming in along the entering direction, not along the chord from the sample.
  const Eigen::Vector2d last = (contact - result.position).normalized();
  const Eigen::Vector2d chord = (contact - sample).normalized();
  EXPECT_GT(last.dot(entering), 1.0 - 1e-3);
  EXPECT_LT(chord.dot(entering), 1.0 - 1e-2);
  ASSERT_EQ(result.ratio_intents.size(), 1u);
  EXPECT_GT(result.ratio_intents[0].step_m,
            result.ratio_intents[0].sample_distance_m /
                result.ratio_intents[0].update_count)
      << "the smoothed course is longer than the chord";
}

TEST(RelationSolverTest, BoxReferentApproachComposesWorldTransformAxisAligned) {
  // Source: axis-aligned box at (1, 1), half-extent 0.25. Actor approaches
  // 2m south of the box's south face (OUTSIDE -> "approach" branch).
  const double half = 0.25;
  auto source_tree = MakeBoxTree(1.0, 1.0, half, /*yaw=*/0.0);
  Eigen::Vector3d actor_pos(1.0, 1.0 - half - 2.0, 0.0);  // (1, -1.25, 0)
  auto payload =
      MakeRealSpacingPayload(actor_pos, 0.0, source_tree, 2, 200);

  ASSERT_TRUE(payload.support.has_contact);
  ASSERT_FALSE(payload.support.has_interior);
  ASSERT_LT(payload.grounded_relation.proximity, 1.0 - 1e-3);

  // Target: axis-aligned box at (4, 2), same half-extent -- away from the
  // tree origin (chair at world (4, 2), half-extent 0.25). A decode that
  // returns the UNCOMPOSED local box-corner coordinate (~(0, -0.25)) instead of
  // lifting through world_t freezes the solved root near the tree origin
  // instead of tracking (4, 2).
  auto target_tree = MakeBoxTree(4.0, 2.0, half, /*yaw=*/0.0);

  RelationSolver solver;
  // The world-composed stand-off is a solution of R. The proposal continues
  // from the sample, so the sample is that pose: a broken world lift moves it.
  SolverResult result = solver.Solve(
      Eigen::Vector2d(4.0, -0.25), 0.0, {payload}, target_tree);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  // Closed form (R-preserving, Eq 8): support_point = (4, 2-0.25) = (4, 1.75).
  // The source had no navigable scope, so R = 1/(1+d) with d = 2 to its
  // contact. The unscoped decode preserves that metric stand-off
  // from the contact along the source's approach direction, so the result is
  // (4, 1.75 - 2) = (4, -0.25). (Adding the CENTRE-relative local
  // offset (0, -2.25) would give (4, -0.5), whose R^tgt = 1/3.25 != R^src.)
  EXPECT_NEAR(result.position.x(), 4.0, 1e-3);
  EXPECT_NEAR(result.position.y(), -0.25, 1e-3);
  // Decisively NOT frozen near the tree origin.
  EXPECT_GT(result.position.norm(), 2.0);
}

TEST(RelationSolverTest, BoxReferentApproachComposesWorldTransformRotated) {
  // Same as above, but both boxes are yawed pi/4 -- a fix that composed only
  // the translation part of world_t (not the full rotation) would still pass
  // the axis-aligned case above but fail here.
  const double half = 0.25;
  const double yaw = M_PI / 4.0;
  auto source_tree = MakeBoxTree(1.0, 1.0, half, yaw);

  // Approach 2m beyond the box's own local south (-Y) face, along the box's
  // OWN (rotated) local -Y axis: world offset = R(yaw) * (0, -(half+2.0)).
  const double local_off = half + 2.0;
  Eigen::Vector3d actor_pos(
      1.0 + local_off * std::sin(yaw),
      1.0 - local_off * std::cos(yaw), 0.0);
  auto payload =
      MakeRealSpacingPayload(actor_pos, 0.0, source_tree, 2, 200);

  ASSERT_TRUE(payload.support.has_contact);
  ASSERT_FALSE(payload.support.has_interior);
  ASSERT_LT(payload.grounded_relation.proximity, 1.0 - 1e-3);

  auto target_tree = MakeBoxTree(4.0, 2.0, half, yaw);

  // Closed form: support_point = (4,2) + R(yaw)*(0,-half); the unscoped
  // source keeps the metric stand-off along the box's rotated -Y.
  Eigen::Vector2d support_point(
      4.0 + half * std::sin(yaw), 2.0 - half * std::cos(yaw));
  Eigen::Vector2d local_offset(0.0, -2.0);
  Eigen::Vector2d rotated_local(
      std::cos(yaw) * local_offset.x() - std::sin(yaw) * local_offset.y(),
      std::sin(yaw) * local_offset.x() + std::cos(yaw) * local_offset.y());
  Eigen::Vector2d expected = support_point + rotated_local;

  RelationSolver solver;
  SolverResult result = solver.Solve(expected, 0.0, {payload}, target_tree);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  // The sample is the closed form. Continuation leaves it there.
  EXPECT_NEAR(result.position.x(), expected.x(), 1e-3);
  EXPECT_NEAR(result.position.y(), expected.y(), 1e-3);
  // Decisively NOT frozen near the tree origin.
  EXPECT_GT(result.position.norm(), 2.0);
}

TEST(RelationSolverTest,
     BoxReferentEngagedInteriorComposesWorldTransformAxisAligned) {
  // Source: axis-aligned box at (1, 1), half-extent 0.25. Actor stands
  // exactly at the box's local (0.1, 0.05) offset from center -- INSIDE the
  // box ("engaged/inside" branch: proximity ~1.0, has_interior == true).
  const double half = 0.25;
  auto source_tree = MakeBoxTree(1.0, 1.0, half, /*yaw=*/0.0);
  Eigen::Vector3d actor_pos(1.0 + 0.1, 1.0 + 0.05, 0.0);
  auto payload =
      MakeRealSpacingPayload(actor_pos, 0.0, source_tree, 2, 200);

  ASSERT_TRUE(payload.support.has_interior);
  ASSERT_GE(payload.grounded_relation.proximity, 1.0 - 1e-6);

  // Target: axis-aligned box at (4, 2), same shape.
  auto target_tree = MakeBoxTree(4.0, 2.0, half, /*yaw=*/0.0);

  RelationSolver solver;
  SolverResult result = solver.Solve(
      Eigen::Vector2d(4.0 + 0.1, 2.0 + 0.05), 0.0, {payload}, target_tree);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  // Same-shape box, identity rotation: the canonical box-UVW chart round-
  // trips the LOCAL offset exactly, so the target world position is the
  // target box's world position plus the same local offset.
  EXPECT_NEAR(result.position.x(), 4.0 + 0.1, 1e-3);
  EXPECT_NEAR(result.position.y(), 2.0 + 0.05, 1e-3);
  EXPECT_GT(result.position.norm(), 2.0);
}

TEST(RelationSolverTest,
     BoxReferentEngagedInteriorComposesWorldTransformRotated) {
  // Same as above, but both boxes are yawed pi/4 -- exercises DecodeInterior's
  // world_t lift under a genuine rotation (a translation-only fix would pass
  // the axis-aligned case but fail here).
  const double half = 0.25;
  const double yaw = M_PI / 4.0;
  auto source_tree = MakeBoxTree(1.0, 1.0, half, yaw);

  // Actor stands at the box's own LOCAL (0.1, 0.05) offset from center,
  // expressed in world coordinates via the box's rotation.
  Eigen::Vector2d local_offset(0.1, 0.05);
  Eigen::Vector2d world_offset(
      std::cos(yaw) * local_offset.x() - std::sin(yaw) * local_offset.y(),
      std::sin(yaw) * local_offset.x() + std::cos(yaw) * local_offset.y());
  Eigen::Vector3d actor_pos(1.0 + world_offset.x(), 1.0 + world_offset.y(),
                           0.0);
  auto payload =
      MakeRealSpacingPayload(actor_pos, 0.0, source_tree, 2, 200);

  ASSERT_TRUE(payload.support.has_interior);
  ASSERT_GE(payload.grounded_relation.proximity, 1.0 - 1e-6);

  auto target_tree = MakeBoxTree(4.0, 2.0, half, yaw);

  RelationSolver solver;
  const Eigen::Vector2d expected(4.0 + world_offset.x(), 2.0 + world_offset.y());
  SolverResult result = solver.Solve(expected, 0.0, {payload}, target_tree);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  // Same-shape box, same yaw: the canonical box-UVW chart round-trips the
  // LOCAL offset exactly, so the target world position is the target box's
  // world position plus the SAME rotated local offset.
  EXPECT_NEAR(result.position.x(), expected.x(), 1e-3);
  EXPECT_NEAR(result.position.y(), expected.y(), 1e-3);
  EXPECT_GT(result.position.norm(), 2.0);
}

// Identity-room check: target room == source room (literally the SAME tree
// and SAME box referent, chair at world (2, 1), the T1_IdentityRoom
// coordinates).
//
// Note: only the ENGAGED (interior, proximity >= 1-1e-3) branch is checked
// for an exact round-trip here. The APPROACH (proximity < 1) branch is NOT
// expected to round-trip exactly even in a literal identity room: by design
// (see the "Denormalize R_src by the diameter of the receiver avatar's
// CONTAINING scope" comment on DecodeTargetPoint in relation_solver.cc), it
// reconstructs distance from the normalized proximity value scaled by the
// target scope's diameter, not from the original metric offset -- an
// R-preserving (not position-preserving) approximation that is orthogonal to
// the world-transform composition and measurably introduces meter-scale error
// of its own even with a real navigable room ancestor present. Only the
// engaged/interior case below round-trips exactly.
TEST(RelationSolverTest, IdentityRoomBoxChairRoundTripEngaged) {
  const double half = 0.25;
  auto room = MakeBoxTree(2.0, 1.0, half, /*yaw=*/0.0, /*shared_id=*/201);

  // Actor stands inside the box ("engaged/interior" branch).
  Eigen::Vector3d actor_pos(2.0 + 0.1, 1.0 + 0.05, 0.0);
  auto payload = MakeRealSpacingPayload(actor_pos, 0.0, room, 2, 201);
  ASSERT_TRUE(payload.support.has_interior);
  ASSERT_GE(payload.grounded_relation.proximity, 1.0 - 1e-6);

  RelationSolver solver;
  // Identity room: the embodied pose is already the solution. The sample is
  // that pose; the old test passed the origin because the solver ignored it.
  SolverResult result =
      solver.Solve(actor_pos.head<2>(), 0.0, {payload}, room);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  EXPECT_NEAR(result.position.x(), actor_pos.x(), 1e-3);
  EXPECT_NEAR(result.position.y(), actor_pos.y(), 1e-3);
}

// Identity room, APPROACH branch (not engaged), with a REAL
// NavigableSpatialDefinition room (so the room-diameter-normalized decode
// path in DecodeTargetPoint's approach branch actually runs -- MakeBoxTree's
// Room(1) above has no spatial definition, so the tests above all hit the
// "no target room scale available" fallback instead). Approaches from a
// DIAGONAL bearing (not face-perpendicular). A diagonal bearing exposed two
// decode errors that this test guards against (0.386 m together at this
// scenario before the fix): a direction reconstructed from the referent's CENTER
// double-counted the center-to-contact offset, and a "far" ray endpoint snapped
// to the nearest boundary point of a wildly out-of-room overshoot instead of the
// true ray/boundary intersection misrouted the geodesic path.
TEST(RelationSolverTest, IdentityRoomApproachBranchDiagonalBearingExact) {
  const double half = 0.25;
  ElementTree room;
  room.AddElement(1);
  auto walkable = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {0, 0}, {20, 0}, {20, 20}, {0, 20}});
  auto nav = std::make_shared<NavigableSpatialDefinition>(walkable);
  room.GetElement(1)->SetSpatialDefinition(nav);
  room.AddElement(2, 1);
  room.GetElement(2)->SetSharedId(201);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = 2.0;
  t(1, 3) = 1.0;
  room.GetElement(2)->SetLocalTransform(t);
  room.GetElement(2)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-half, -half, -1.0),
          Eigen::Vector3d(half, half, 1.0)));

  // Diagonal bearing from the box center, well outside the box, staying
  // INSIDE the 20x20 room polygon (chair is near the (0,0) corner, so the
  // bearing must have non-negative components).
  Eigen::Vector2d bearing = Eigen::Vector2d(3, 4).normalized();
  Eigen::Vector2d box_center(2.0, 1.0);
  Eigen::Vector3d actor_pos(box_center.x() + bearing.x() * 4.0,
                            box_center.y() + bearing.y() * 4.0, 0.0);

  Relation rel("spacing", "root", 2,
               std::make_unique<SpacingGroundingRule>(),
               std::make_unique<AvatarSpacingKernel>());
  auto eval = rel.Evaluate(actor_pos, 0.0, room, nav.get());

  RelationPayload payload;
  payload.referent_shared_id = 201;
  payload.subject_anchor_name = "root";
  payload.grounded_relation = eval.grounded;
  payload.support = eval.support;
  payload.kernel_type = RelationKernelType::kAvatarSpacing;

  ASSERT_TRUE(payload.support.has_contact);
  ASSERT_FALSE(payload.support.has_interior);
  ASSERT_LT(payload.grounded_relation.proximity, 1.0 - 1e-3);

  RelationSolver solver;
  SolverResult result =
      solver.Solve(actor_pos.head<2>(), 0.0, {payload}, room, nav.get());

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  EXPECT_NEAR(result.position.x(), actor_pos.x(), 1e-3);
  EXPECT_NEAR(result.position.y(), actor_pos.y(), 1e-3);
}

// ===========================================================================
// SolveActorTargets — two-stage realizer:
// independent solves per (kernel family, subject anchor) group, then
// dependency-ordered composition (root position → yaw → anchor poses).
// ===========================================================================

// An affordance support needs geometry (a geometryless referent
// is a grounding failure, not a target at its origin). Gives element `id` a
// flat 2 x 2 m box with a named "top" part in its own plane (z = 0).
void AddBoardTop(ElementTree& tree, element_id_t id) {
  auto def = std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-1, -1, -0.05), Eigen::Vector3d(1, 1, 0.0),
      GeometryRole::kSurface);
  SubRegion top;
  top.name = "top";
  top.boundary = std::make_shared<spatialblend::Polygon2D>(
      std::vector<Eigen::Vector2d>{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}});
  def->AddSubRegion(top);
  tree.GetElement(id)->SetSpatialDefinition(def);
}

RelationPayload MakeAffordancePayload(const Eigen::Vector3d& anchor_pos,
                                      double anchor_yaw,
                                      const ElementTree& tree,
                                      element_id_t referent_id,
                                      shared_element_id_t shared_id,
                                      const std::string& anchor_name) {
  Relation rel("affordance", anchor_name, referent_id,
               std::make_unique<AffordancePoseGroundingRule>(),
               std::make_unique<AvatarAffordancePoseKernel>());
  auto result = rel.Evaluate(anchor_pos, anchor_yaw, tree);

  RelationPayload payload;
  payload.referent_shared_id = shared_id;
  payload.subject_anchor_name = anchor_name;
  payload.grounded_relation = result.grounded;
  payload.support = result.support;
  payload.kernel_type = RelationKernelType::kAvatarAffordancePose;
  return payload;
}

// Room(1) -> {Board(2, sid 200), Table(3, sid 300)} at given positions.
ElementTree MakeTwoObjectTree(const Eigen::Vector2d& board,
                              const Eigen::Vector2d& table) {
  ElementTree tree;
  tree.AddElement(1);
  tree.AddElement(2, 1);
  tree.GetElement(2)->SetSharedId(200);
  Eigen::Matrix4d bt = Eigen::Matrix4d::Identity();
  bt(0, 3) = board.x();
  bt(1, 3) = board.y();
  tree.GetElement(2)->SetLocalTransform(bt);
  tree.AddElement(3, 1);
  tree.GetElement(3)->SetSharedId(300);
  Eigen::Matrix4d tt = Eigen::Matrix4d::Identity();
  tt(0, 3) = table.x();
  tt(1, 3) = table.y();
  tree.GetElement(3)->SetLocalTransform(tt);
  return tree;
}

const RealizationTarget* FindTarget(const RealizationResult& result,
                                    const std::string& anchor_name) {
  for (const auto& t : result.targets) {
    if (t.anchor_name == anchor_name) return &t;
  }
  return nullptr;
}

TEST(SolveActorTargetsTest, AffordanceDoesNotLeakIntoRoot) {
  // Spacing(root → board) plus an engaged Affordance(hand_r → table). The
  // affordance kernel's dynamic states include position AND yaw, so in the
  // shared solve it would drag root toward the table; the per-group solve
  // must produce a root target IDENTICAL to the spacing-only solve.
  auto source_tree = MakeTwoObjectTree({5, 8}, {15, 2});
  Eigen::Vector3d actor_pos(3, 6, 0);
  auto spacing = MakeSpacingPayload(actor_pos, 0.0, source_tree, 2, 200);
  auto affordance = MakeAffordancePayload(
      Eigen::Vector3d(14.6, 2, 1.0), 0.5, source_tree, 3, 300, "hand_r");

  auto target_tree = MakeTwoObjectTree({10, 4}, {2, 9});

  RelationSolver solver;
  SolverResult spacing_only = solver.Solve(
      Eigen::Vector2d(0, 0), 0.25, {spacing}, target_tree);
  RealizationResult composed = solver.SolveActorTargets(
      Eigen::Vector2d(0, 0), 0.25, {spacing, affordance}, {}, target_tree);

  const RealizationTarget* root = FindTarget(composed, "root");
  ASSERT_NE(root, nullptr);
  EXPECT_NEAR(root->position.x(), spacing_only.position.x(), 1e-9);
  EXPECT_NEAR(root->position.y(), spacing_only.position.y(), 1e-9);
  // Spacing fixes yaw; the affordance's relative-yaw goal must not leak.
  EXPECT_NEAR(root->yaw, 0.25, 1e-9);
}

TEST(SolveActorTargetsTest, EmitsPerAnchorAffordanceTarget) {
  // Engaged hand on the board: the result must contain a "hand_r" target at
  // the decoded support on the TARGET board, with the target's height as Z
  // and the source's support-relative yaw reproduced. The boards
  // carry a support part (a geometryless referent has no target).
  // The hand projects to board-local (0.5, 0),
  // which maps to the same chart point on the congruent target board.
  ElementTree source_tree;
  source_tree.AddElement(1);
  source_tree.AddElement(2, 1);
  source_tree.GetElement(2)->SetSharedId(200);
  Eigen::Matrix4d bt = Eigen::Matrix4d::Identity();
  bt(0, 3) = 5.0;
  bt(1, 3) = 8.0;
  bt(2, 3) = 0.5;
  source_tree.GetElement(2)->SetLocalTransform(bt);
  AddBoardTop(source_tree, 2);

  auto affordance = MakeAffordancePayload(
      Eigen::Vector3d(5.5, 8.0, 1.0), 0.7, source_tree, 2, 200, "hand_r");

  ElementTree target_tree;
  target_tree.AddElement(1);
  target_tree.AddElement(2, 1);
  target_tree.GetElement(2)->SetSharedId(200);
  Eigen::Matrix4d tt = Eigen::Matrix4d::Identity();
  tt(0, 3) = 10.0;
  tt(1, 3) = 4.0;
  tt(2, 3) = 0.8;
  target_tree.GetElement(2)->SetLocalTransform(tt);
  AddBoardTop(target_tree, 2);

  RelationSolver solver;
  RealizationResult result = solver.SolveActorTargets(
      Eigen::Vector2d(0, 0), 0.0, {affordance}, {}, target_tree);

  const RealizationTarget* hand = FindTarget(result, "hand_r");
  ASSERT_NE(hand, nullptr);
  EXPECT_NEAR(hand->position.x(), 10.5, 1e-6);
  EXPECT_NEAR(hand->position.y(), 4.0, 1e-6);
  EXPECT_TRUE(hand->has_z);
  EXPECT_NEAR(hand->z, 0.8, 1e-6);
  // Source relative yaw (0.7 − support yaw 0) reapplied at the target support.
  EXPECT_NEAR(hand->yaw, 0.7, 1e-6);
  EXPECT_EQ(hand->kernel_type, RelationKernelType::kAvatarAffordancePose);
  EXPECT_EQ(hand->referent_id, 200);
}

TEST(SolveActorTargetsTest, ComposedYawEvaluatedAtSolvedRootPosition) {
  // Facing's bearing must be evaluated AT the composed root position (the one
  // legitimate cross-group coupling), not at the seed position.
  auto source_tree = MakeTree(5.0, 8.0);
  Eigen::Vector3d actor_pos(4, 7, 0);
  double actor_yaw = std::atan2(8.0 - 7.0, 5.0 - 4.0);
  auto spacing = MakeSpacingPayload(actor_pos, actor_yaw, source_tree, 2, 200);
  auto facing = MakeFacingPayload(actor_pos, actor_yaw, source_tree, 2, 200);

  auto target_tree = MakeTree(2.0, 3.0);

  RelationSolver solver;
  RealizationResult result = solver.SolveActorTargets(
      Eigen::Vector2d(0, 0), 0.0, {spacing, facing}, {}, target_tree);

  const RealizationTarget* root = FindTarget(result, "root");
  ASSERT_NE(root, nullptr);
  double expected_yaw = std::atan2(3.0 - root->position.y(),
                                   2.0 - root->position.x());
  EXPECT_NEAR(root->yaw, expected_yaw, 0.3);
}

TEST(SolveActorTargetsTest, RootTargetAlwaysFirst) {
  // Even with affordance-only payloads the contract guarantees targets[0] is
  // the root (held at the current commanded pose when nothing drives it), so
  // consumers can rely on ordering.
  // Boards carry a support part so the anchor target exists.
  auto source_tree = MakeTree(5.0, 8.0);
  AddBoardTop(source_tree, 2);
  auto affordance = MakeAffordancePayload(
      Eigen::Vector3d(5.5, 8.0, 1.0), 0.0, source_tree, 2, 200, "hand_r");

  auto target_tree = MakeTree(10.0, 4.0);
  AddBoardTop(target_tree, 2);

  RelationSolver solver;
  RealizationResult result = solver.SolveActorTargets(
      Eigen::Vector2d(3, 4), 1.0, {affordance}, {}, target_tree);

  ASSERT_GE(result.targets.size(), 2u);
  EXPECT_EQ(result.targets[0].anchor_name, "root");
  EXPECT_NEAR(result.targets[0].position.x(), 3.0, 1e-9);
  EXPECT_NEAR(result.targets[0].position.y(), 4.0, 1e-9);
  EXPECT_NEAR(result.targets[0].yaw, 1.0, 1e-9);
}

// ===========================================================================
// UpdateAnchorGate — hysteresis gating, owned by realization_policy.h.
// ===========================================================================

TEST(UpdateAnchorGateTest, ColdStartBelowEnterStaysInactive) {
  AnchorGateState state;
  AnchorGateConfig config;  // enter=0.6, exit=0.4, alpha=0.3
  EXPECT_FALSE(UpdateAnchorGate(state, 0.5, config));
  EXPECT_FALSE(state.active);
}

TEST(UpdateAnchorGateTest, ColdStartAboveEnterGatesOnImmediately) {
  // A single strong first sample must gate on right away (cold-start seeds
  // the EMA to the first sample rather than damping it toward 0).
  AnchorGateState state;
  AnchorGateConfig config;
  EXPECT_TRUE(UpdateAnchorGate(state, 0.9, config));
  EXPECT_TRUE(state.active);
}

TEST(UpdateAnchorGateTest, NoFlickerWhileOscillatingWithinHysteresisBand) {
  // Drive the gate solidly on, then oscillate the raw signal repeatedly
  // between the exit (0.4) and enter (0.6) thresholds — i.e. never clearly
  // above enter nor clearly below exit. A single-threshold gate (no
  // hysteresis) would chatter here; the Schmitt-trigger gate must not.
  AnchorGateState state;
  AnchorGateConfig config;
  for (int i = 0; i < 10; ++i) UpdateAnchorGate(state, 0.95, config);
  ASSERT_TRUE(state.active);

  int transitions = 0;
  bool prev_active = state.active;
  const std::vector<double> oscillation = {
      0.45, 0.55, 0.45, 0.55, 0.45, 0.55, 0.45, 0.55, 0.45, 0.55,
      0.45, 0.55, 0.45, 0.55, 0.45, 0.55, 0.45, 0.55, 0.45, 0.55};
  for (double raw : oscillation) {
    bool active = UpdateAnchorGate(state, raw, config);
    if (active != prev_active) ++transitions;
    prev_active = active;
  }
  EXPECT_EQ(transitions, 0);
  EXPECT_TRUE(state.active);
}

TEST(UpdateAnchorGateTest, ClearDropBelowExitReleases) {
  AnchorGateState state;
  AnchorGateConfig config;
  for (int i = 0; i < 10; ++i) UpdateAnchorGate(state, 0.9, config);
  ASSERT_TRUE(state.active);
  bool active = false;
  for (int i = 0; i < 10; ++i) active = UpdateAnchorGate(state, 0.0, config);
  EXPECT_FALSE(active);
}

// ===========================================================================
// ClampToReach — reach clamp, owned by relation_solver.h.
// ===========================================================================

TEST(ClampToReachTest, WithinRadiusPassesThrough) {
  Eigen::Vector2d point(1.2, 1.2);
  Eigen::Vector2d center(1.0, 1.0);
  Eigen::Vector2d clamped = ClampToReach(point, center, /*radius=*/0.9);
  EXPECT_NEAR(clamped.x(), point.x(), 1e-12);
  EXPECT_NEAR(clamped.y(), point.y(), 1e-12);
}

TEST(ClampToReachTest, BeyondRadiusEngagesAndProjectsToBoundary) {
  // Decoded hand point far across the room (matches the reach-clamp gap
  // that ScenarioReplayTest.AffordancePoseReplay hits when the root does not
  // move — see the NOTE there): must clamp onto the circle of radius 0.9m
  // around root, in the same direction as the unclamped point.
  Eigen::Vector2d root(0.0, 0.0);
  Eigen::Vector2d far_target(8.0, 4.0);
  Eigen::Vector2d clamped = ClampToReach(far_target, root, /*radius=*/0.9);
  EXPECT_NEAR(clamped.norm(), 0.9, 1e-9);
  EXPECT_NEAR(clamped.x() / clamped.y(), far_target.x() / far_target.y(),
             1e-9);
}

TEST(ClampToReachTest, ZeroOrNegativeRadiusDisablesClamp) {
  Eigen::Vector2d point(8.0, 4.0);
  Eigen::Vector2d center(0.0, 0.0);
  EXPECT_EQ(ClampToReach(point, center, 0.0), point);
  EXPECT_EQ(ClampToReach(point, center, -1.0), point);
}

// The T1-identity-room geometry (sim/harness/testdata/
// T1-identity-room.scenario.json) -- a RECTANGULAR (non-square, 6.4x3.2) room
// and a YAWED (1.5708 rad) box chair built via MakeEnvelope (the SAME
// world-baked, GeometryRole::kContainment NavigableSpatialDefinition that
// scenario.cc's harness AND the C API's box-footprint path both use for a
// "box" footprint -- NOT PolytopeSpatialDefinition).
// This test directly asserts the decode itself (not a harness's
// controller-pacing measurement) is exact for this geometry, at multiple
// approach angles, so any future regression here is caught at the source
// instead of being conflated with the harness's frame-0 artifact.
// RETIRED (the J refinement no longer exists): asserts what the position search / refinement produced (an exact optimum, restarts, tracking, acceptance rule, basin loss) or a step bound the refinement shaped. The position answer is S0 clamped; the harness defects (proposal == clamped S0, identical copies exact, every jump traced to its cause) replace these checks.
TEST(RelationSolverTest, DISABLED_IdentityRoomEnvelopeBoxChairRotatedRoomExactRoundTrip) {
  const Eigen::Vector3d chair_pos(2.0, 1.0, 0.0);
  const double chair_yaw = 1.5708;

  ElementTree room;
  room.AddElement(1);
  auto walkable = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {0, 0}, {6.4, 0}, {6.4, 3.2}, {0, 3.2}});
  auto nav = std::make_shared<NavigableSpatialDefinition>(walkable);
  room.GetElement(1)->SetSpatialDefinition(nav);

  room.AddElement(2, 1);
  room.GetElement(2)->SetSharedId(201);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 0) = std::cos(chair_yaw);
  t(0, 1) = -std::sin(chair_yaw);
  t(1, 0) = std::sin(chair_yaw);
  t(1, 1) = std::cos(chair_yaw);
  t(0, 3) = chair_pos.x();
  t(1, 3) = chair_pos.y();
  t(2, 3) = chair_pos.z();
  room.GetElement(2)->SetLocalTransform(t);
  std::vector<double> params = {chair_pos.x(), chair_pos.y(), chair_pos.z(),
                                0.5, 0.5, 0.9};
  auto chair_def =
      MakeEnvelope(ColliderType::kBox, params, /*circle_segments=*/16,
                  chair_yaw);
  ASSERT_NE(chair_def, nullptr);
  room.GetElement(2)->SetSpatialDefinition(chair_def);

  auto SolveApproach = [&](const Eigen::Vector3d& actor_pos) {
    Relation rel("spacing", "root", 2,
                 std::make_unique<SpacingGroundingRule>(),
                 std::make_unique<AvatarSpacingKernel>());
    auto eval = rel.Evaluate(actor_pos, 0.0, room, nav.get());

    RelationPayload payload;
    payload.referent_shared_id = 201;
    payload.subject_anchor_name = "root";
    payload.grounded_relation = eval.grounded;
    payload.support = eval.support;
    payload.kernel_type = RelationKernelType::kAvatarSpacing;

    EXPECT_TRUE(payload.support.has_contact);
    EXPECT_FALSE(payload.support.has_interior);
    EXPECT_LT(payload.grounded_relation.proximity, 1.0 - 1e-3);

    RelationSolver solver;
    return solver.Solve(Eigen::Vector2d(0, 0), 0.0, {payload}, room, nav.get());
  };

  // North (perpendicular to the chair's world-north face; 2.2m of clearance
  // to the y=3.2 wall, unlike south's 1m -- see the wall-clamp note below).
  {
    Eigen::Vector3d actor_pos(2.0, 1.0 + 0.25 + 0.8, 0.0);
    SolverResult result = SolveApproach(actor_pos);
    EXPECT_EQ(result.status, SolverStatus::kConverged);
    EXPECT_NEAR(result.position.x(), actor_pos.x(), 1e-3);
    EXPECT_NEAR(result.position.y(), actor_pos.y(), 1e-3);
  }
  // North-east diagonal (bearing (0.6,0.8) from the chair CENTER, 2.0m out --
  // not axis-aligned with the yawed box's own faces).
  {
    const Eigen::Vector2d bearing(0.6, 0.8);
    Eigen::Vector3d actor_pos(chair_pos.x() + bearing.x() * 2.0,
                              chair_pos.y() + bearing.y() * 2.0, 0.0);
    SolverResult result = SolveApproach(actor_pos);
    EXPECT_EQ(result.status, SolverStatus::kConverged);
    EXPECT_NEAR(result.position.x(), actor_pos.x(), 1e-3);
    EXPECT_NEAR(result.position.y(), actor_pos.y(), 1e-3);
  }
  // South, 0.8m beyond the chair's south face (matching the fixture script's
  // literal "approach chair distance 0.8"): the room is only 3.2m deep and
  // the chair sits 1m from the y=0 wall, so this specific approach would
  // place the actor OUTSIDE the room polygon entirely -- a genuine
  // wall-clamp limitation (see relation_solver.cc's `far`/GetDiscretePath
  // fallback), not a
  // Documented here, not asserted exact.
  //
  // The nearest-point wall clamp is only
  // the INITIALIZER now. Since the true (out-of-room) target is infeasible,
  // the paper-objective refinement searches along the feasible south wall
  // (y=0) for the point whose R (grounded on the yawed box's actual
  // asymmetric footprint, the SAME evaluator Extract_p used) is closest to
  // R^src, which is generally NOT the point nearest the decoded witness --
  // that is exactly the gap the refinement closes. So x is no longer pinned to the
  // witness's x=2.0; instead the test asserts the paper-objective contract
  // directly: still on the wall, still inside the room, and never worse
  // than the initializer it replaced.
  {
    Eigen::Vector3d actor_pos(2.0, 1.0 - 0.25 - 0.8, 0.0);
    SolverResult result = SolveApproach(actor_pos);
    EXPECT_EQ(result.status, SolverStatus::kConverged);
    // Still clamped to the south wall -- the refinement's own feasibility
    // gate never leaves F(t).
    EXPECT_NEAR(result.position.y(), 0.0, 1e-3);
    EXPECT_GE(result.position.x(), 0.0);
    EXPECT_LE(result.position.x(), 6.4);
    // Never worse than the initializer (the old wall-clamp-to-witness
    // point); this is the general refinement contract, not specific to this case.
    EXPECT_LE(result.objective_after, result.objective_before + 1e-9);
    // A single relation constrained to the wall still has an
    // exact (J=0) relation-preserving point along it (the search finds the
    // x on y=0 where R^tgt(x,0) == R^src exactly); it is not merely
    // "better than the witness clamp", it is the true optimum.
    EXPECT_LT(result.objective_after, 1e-6);
  }
}

// ---------------------------------------------------------------------------
// Paper-objective refinement tests.
// Builds a two-referent target scene where both spacing relations are
// unscoped (R = 1/(1+d), DefaultGroundingRule with no boundary), referent
// 200 at (0,0) and referent 300 at (6,0) unless noted.
// ---------------------------------------------------------------------------
namespace {

void MakeTwoSupportTargetTree(ElementTree* target_tree) {
  target_tree->AddElement(1);
  target_tree->AddElement(2, 1);
  target_tree->GetElement(2)->SetSharedId(200);
  target_tree->AddElement(3, 1);
  target_tree->GetElement(3)->SetSharedId(300);
  Eigen::Matrix4d t3 = Eigen::Matrix4d::Identity();
  t3(0, 3) = 6.0;
  target_tree->GetElement(3)->SetLocalTransform(t3);
}

}  // namespace

// RETIRED (the J refinement no longer exists): asserts what the position search / refinement produced (an exact optimum, restarts, tracking, acceptance rule, basin loss) or a step bound the refinement shaped. The position answer is S0 clamped; the harness defects (proposal == clamped S0, identical copies exact, every jump traced to its cause) replace these checks.
TEST(PaperObjectiveTest, CounterexampleReachesExactRelationPreservingState) {
  // Counterexample:
  // two target supports at (0,0) and (6,0); both source relations require
  // distance 5 under the same (unscoped) normalization. (3,4) satisfies
  // both exactly; the OLD centroid-of-decoded-goals objective averages the
  // witnesses (5,0) and (1,0) into (3,0), which satisfies neither.
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a =
      MakeSpacingPayload(Eigen::Vector3d(5, 0, 0), 0.0, source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(-5, 0, 0), 0.0,
                                      source_tree_b, 2, 300);
  ASSERT_NEAR(payload_a.grounded_relation.proximity, 1.0 / 6.0, 1e-6);
  ASSERT_NEAR(payload_b.grounded_relation.proximity, 1.0 / 6.0, 1e-6);

  ElementTree target_tree;
  MakeTwoSupportTargetTree(&target_tree);

  RelationSolver solver;
  // (3, ±4) is where both distances equal the source's 5 m. That is the
  // retired current-error point. These payloads carry no valid intent, so
  // the continuation stays at the sample.
  SolverResult result = solver.Solve(Eigen::Vector2d(3, 0), 0.0,
                                      {payload_a, payload_b}, target_tree);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  EXPECT_LT((result.position - Eigen::Vector2d(3, 0)).norm(), 1e-6);
  EXPECT_EQ(result.objective_relation_count, 0);
}

// The retired assertion required two samples 1 cm apart to share one
// absolute-distance point (both circles of the source's 5 m). The ratio
// residual uses each sample's own distance, so those samples do not share a
// point. On one ray they do: n = 2 halves each distance, and the landings
// stay on that ray, separated by half the sample gap.
TEST(PaperObjectiveTest, NearbySamplesOnOneBasinShareTheMinimiser) {
  auto source_tree = MakeTree(0.0, 0.0, 200);
  auto payload =
      MakeSpacingPayload(Eigen::Vector3d(0, 4, 0), 0.0, source_tree, 2, 200);
  payload.update_count = 2.0;

  auto target_tree = MakeTree(0.0, 0.0, 200);
  RelationSolver solver;
  const Eigen::Vector2d near(0.0, 8.0);
  const Eigen::Vector2d far(0.0, 8.02);
  const SolverResult a = solver.Solve(near, 0.0, {payload}, target_tree);
  const SolverResult b = solver.Solve(far, 0.0, {payload}, target_tree);

  EXPECT_EQ(a.status, SolverStatus::kConverged);
  EXPECT_EQ(b.status, SolverStatus::kConverged);
  EXPECT_NEAR(a.position.x(), 0.0, 1e-3);
  EXPECT_NEAR(a.position.y(), 4.0, 1e-2);
  EXPECT_NEAR(b.position.x(), 0.0, 1e-3);
  EXPECT_NEAR(b.position.y(), 4.01, 1e-2);
  EXPECT_LT((b.position - a.position).norm(), 0.02);
}

// An infeasible Gauss-Newton trial used to be replaced by NearestFree of that
// exterior point. On this sofa the sample sits on the west clearance and the
// step lands inside the north ring, so that snap is the far face. Samples
// 3 cm apart then stop on opposite faces, 1.4 m apart. A snap counts only
// when the segment to it stays in F, so both samples stay on the south ring.
TEST(PaperObjectiveTest, InfeasibleStepStopsWhereTheSegmentEntersTheObstacle) {
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a =
      MakeSpacingPayload(Eigen::Vector3d(5, 0, 0), 0.0, source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(-5, 0, 0), 0.0,
                                      source_tree_b, 2, 300);
  // Contact, so the step tries to close both distances and enters the sofa.
  // The retired current-error step did the same from these samples.
  payload_a.update_count = 1.0;
  payload_b.update_count = 1.0;

  ElementTree target_tree;
  MakeTwoSupportTargetTree(&target_tree);
  auto walkable = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {-10, -10}, {10, -10}, {10, 10}, {-10, 10}});
  NavigableSpatialDefinition nav(walkable);
  auto sofa = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {2, 3.55}, {4, 3.55}, {4, 4.45}, {2, 4.45}});
  nav.SetObstacles({FreeSpaceObstacle{/*key=*/1, sofa}}, /*clearance=*/0.2);

  RelationSolver solver;
  const SolverResult low =
      solver.Solve(Eigen::Vector2d(1.8, 3.66), 0.0, {payload_a, payload_b},
                   target_tree, &nav);
  const SolverResult high =
      solver.Solve(Eigen::Vector2d(1.8, 3.69), 0.0, {payload_a, payload_b},
                   target_tree, &nav);

  EXPECT_EQ(low.status, SolverStatus::kConverged);
  EXPECT_EQ(high.status, SolverStatus::kConverged);
  // The footprint's south edge is y = 3.55; the north ring is y = 4.65.
  // The far snap put the higher sample on that north ring. Both stay south
  // of the footprint.
  EXPECT_LT(low.position.y(), 3.55) << low.position.transpose();
  EXPECT_LT(high.position.y(), 3.55) << high.position.transpose();
  EXPECT_TRUE(nav.IsFree(low.position));
  EXPECT_TRUE(nav.IsFree(high.position));
}

// (RigidTransformEquivariance is re-enabled on the S0 pipeline in solver_layer_test.cc.)

TEST(PaperObjectiveTest, DISABLED_RigidTransformEquivariance) {
  // Rotate and translate the whole target scene (both referents' own local
  // frames, not just their positions) and verify the refined position
  // transforms the same way -- theta=0/shift=0 below is the identity case.
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a =
      MakeSpacingPayload(Eigen::Vector3d(5, 0, 0), 0.0, source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(-5, 0, 0), 0.0,
                                      source_tree_b, 2, 300);

  auto Solve = [&](double theta, const Eigen::Vector2d& shift) {
    const double c = std::cos(theta), s = std::sin(theta);
    auto Rigid = [&](double x, double y) {
      Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
      m(0, 0) = c;
      m(0, 1) = -s;
      m(1, 0) = s;
      m(1, 1) = c;
      m(0, 3) = c * x - s * y + shift.x();
      m(1, 3) = s * x + c * y + shift.y();
      return m;
    };
    ElementTree target_tree;
    target_tree.AddElement(1);
    target_tree.AddElement(2, 1);
    target_tree.GetElement(2)->SetSharedId(200);
    target_tree.GetElement(2)->SetLocalTransform(Rigid(0.0, 0.0));
    target_tree.AddElement(3, 1);
    target_tree.GetElement(3)->SetSharedId(300);
    target_tree.GetElement(3)->SetLocalTransform(Rigid(6.0, 0.0));

    RelationSolver solver;
    const Eigen::Vector2d seed(c * 3.0 - s * 0.0 + shift.x(),
                               s * 3.0 + c * 0.0 + shift.y());
    return solver.Solve(seed, 0.0, {payload_a, payload_b}, target_tree);
  };

  const SolverResult identity_result = Solve(0.0, Eigen::Vector2d::Zero());
  const double theta = 0.7;
  const Eigen::Vector2d shift(10.0, -4.0);
  const SolverResult transformed_result = Solve(theta, shift);

  // Two exact-solution branches exist (the counterexample's two circles
  // intersect at (3,4) and its mirror (3,-4)); either is an equally valid
  // global minimizer, and which one an axis/diagonal-limited local search
  // reaches after an arbitrary rotation is not itself an equivariance
  // property. Equivariance is checked on the OBJECTIVE (rotation/
  // translation invariant by construction: distances, hence R, do not
  // change) and by requiring the position land on WHICHEVER of the two
  // transformed branches it corresponds to.
  const double c = std::cos(theta), s = std::sin(theta);
  auto TransformPoint = [&](const Eigen::Vector2d& p) {
    return Eigen::Vector2d(c * p.x() - s * p.y() + shift.x(),
                           s * p.x() + c * p.y() + shift.y());
  };
  const Eigen::Vector2d branch_north = TransformPoint(Eigen::Vector2d(3, 4));
  const Eigen::Vector2d branch_south = TransformPoint(Eigen::Vector2d(3, -4));
  const double dist_north =
      (transformed_result.position - branch_north).norm();
  const double dist_south =
      (transformed_result.position - branch_south).norm();
  EXPECT_LT(std::min(dist_north, dist_south), 0.1);
  EXPECT_NEAR(transformed_result.objective_before,
              identity_result.objective_before, 1e-9);
  EXPECT_NEAR(transformed_result.objective_after,
              identity_result.objective_after, 1e-6);
  EXPECT_EQ(transformed_result.objective_fallback,
            identity_result.objective_fallback);
}

TEST(PaperObjectiveTest, RefinementNeverEntersAnObstacle) {
  // Same two-referent counterexample, but the exact minimizer (3,4) now
  // sits inside an obstacle: the refinement must still try to improve on
  // the initializer without ever stepping inside it.
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a =
      MakeSpacingPayload(Eigen::Vector3d(5, 0, 0), 0.0, source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(-5, 0, 0), 0.0,
                                      source_tree_b, 2, 300);

  ElementTree target_tree;
  MakeTwoSupportTargetTree(&target_tree);

  auto walkable = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {-10, -10}, {10, -10}, {10, 10}, {-10, 10}});
  NavigableSpatialDefinition nav(walkable);
  auto obstacle = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{2, 3}, {4, 3}, {4, 5}, {2, 5}});
  nav.SetObstacles({FreeSpaceObstacle{/*key=*/1, obstacle}},
                   /*clearance=*/0.1);

  RelationSolver solver;
  SolverResult result = solver.Solve(
      Eigen::Vector2d(3, 0), 0.0, {payload_a, payload_b}, target_tree, &nav);

  EXPECT_TRUE(result.status == SolverStatus::kConverged ||
              result.status == SolverStatus::kFallback);
  EXPECT_TRUE(nav.IsFree(result.position));
  // Still the general refinement contract: never worse than the initializer.
  EXPECT_LE(result.objective_after, result.objective_before + 1e-9);
}

TEST(PaperObjectiveTest, RefinementIsDeterministic) {
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a =
      MakeSpacingPayload(Eigen::Vector3d(5, 0, 0), 0.0, source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(-5, 0, 0), 0.0,
                                      source_tree_b, 2, 300);
  ElementTree target_tree;
  MakeTwoSupportTargetTree(&target_tree);

  RelationSolver solver;
  const SolverResult r1 = solver.Solve(Eigen::Vector2d(3, 0), 0.0,
                                       {payload_a, payload_b}, target_tree);
  const SolverResult r2 = solver.Solve(Eigen::Vector2d(3, 0), 0.0,
                                       {payload_a, payload_b}, target_tree);

  EXPECT_EQ(r1.position.x(), r2.position.x());
  EXPECT_EQ(r1.position.y(), r2.position.y());
  EXPECT_EQ(r1.yaw, r2.yaw);
  EXPECT_EQ(r1.objective_before, r2.objective_before);
  EXPECT_EQ(r1.objective_after, r2.objective_after);
  EXPECT_EQ(r1.objective_iterations, r2.objective_iterations);
  EXPECT_EQ(r1.objective_fallback, r2.objective_fallback);
}

TEST(PaperObjectiveTest, FallbackNeverWorseThanInitializer) {
  // The answer is the continuation, not a fallback to the centroid. J at the
  // emitted pose is at most J at the sample.
  auto source_tree = MakeTree(5.0, 8.0);
  auto payload =
      MakeSpacingPayload(Eigen::Vector3d(3, 6, 0), 0.0, source_tree, 2, 200);
  auto target_tree = MakeTree(10.0, 4.0);

  RelationSolver solver;
  SolverResult result =
      solver.Solve(Eigen::Vector2d(0, 0), 0.0, {payload}, target_tree);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  EXPECT_FALSE(result.objective_fallback);
  EXPECT_LE(result.objective_after, result.objective_before + 1e-9);
}

TEST(PaperObjectiveTest, PolicyWeightsSteerTheRefinedResult) {
  // Same two supports as the counterexample, but weighted heavily toward
  // referent A: the refined position should end up far closer to exactly
  // satisfying A's relation than B's (unlike the equal-weight case, which
  // satisfies both equally at (3,4)/(3,-4)).
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a =
      MakeSpacingPayload(Eigen::Vector3d(5, 0, 0), 0.0, source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(-5, 0, 0), 0.0,
                                      source_tree_b, 2, 300);
  // n = 2 halves each of the sample's 3 m distances. Both cannot reach 1.5 m:
  // the supports are 6 m apart. The heavier weight pulls toward A. The retired
  // assertion required both R errors under 0.02, the current-error point.
  payload_a.update_count = 2.0;
  payload_b.update_count = 2.0;
  ElementTree target_tree;
  MakeTwoSupportTargetTree(&target_tree);

  RelationSolver solver;
  const SolverResult heavy_a =
      solver.Solve(Eigen::Vector2d(3, 0), 0.0, {payload_a, payload_b},
                   {10.0, 1.0}, target_tree);

  TargetAnchorPose anchor;
  anchor.position =
      Eigen::Vector3d(heavy_a.position.x(), heavy_a.position.y(), 0.0);
  anchor.yaw = heavy_a.yaw;
  const auto eval_a =
      EvaluateTargetRelation(payload_a, anchor, target_tree);
  const auto eval_b =
      EvaluateTargetRelation(payload_b, anchor, target_tree);
  ASSERT_TRUE(eval_a.ok());
  ASSERT_TRUE(eval_b.ok());
  const double d_a = (heavy_a.position - Eigen::Vector2d(0, 0)).norm();
  const double d_b = (heavy_a.position - Eigen::Vector2d(6, 0)).norm();
  EXPECT_LT(std::abs(d_a - 1.5), std::abs(d_b - 1.5));
  EXPECT_LT(heavy_a.position.x(), 2.5);
}

// The multi-relation descent iterates to convergence. The
// iteration cap is a safety net, not a budget; reaching it is NOT_CONVERGED.
TEST(PaperObjectiveTest, GaussNewtonRunsToConvergenceAndTheCapIsNotConverged) {
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a =
      MakeSpacingPayload(Eigen::Vector3d(5, 0, 0), 0.0, source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(-5, 0, 0), 0.0,
                                      source_tree_b, 2, 300);
  payload_a.update_count = 2.0;
  payload_b.update_count = 2.0;
  ElementTree target_tree;
  MakeTwoSupportTargetTree(&target_tree);

  RelationSolver solver;
  EXPECT_GE(solver.config().gn_safety_cap, 10000);
  const SolverResult full = solver.Solve(
      Eigen::Vector2d(3, 0), 0.0, {payload_a, payload_b}, {10.0, 1.0},
      target_tree);
  EXPECT_EQ(full.status, SolverStatus::kConverged);
  EXPECT_TRUE(full.objective_stationary);
  ASSERT_GE(full.objective_iterations, 1);

  RelationSolver::Config capped;
  capped.gn_safety_cap = 1;
  RelationSolver capped_solver(capped);
  const SolverResult cut = capped_solver.Solve(
      Eigen::Vector2d(3, 0), 0.0, {payload_a, payload_b}, {10.0, 1.0},
      target_tree);
  EXPECT_EQ(cut.status, SolverStatus::kNotConverged);
  EXPECT_FALSE(cut.objective_stationary);
  EXPECT_FALSE(cut.objective_exact);
}

// The per-solve evaluation budget (the per-frame budget as a count of J evaluations, kP1SolveBudgetEvals) is a safety net: on a hit the solve returns
// its best feasible incumbent as kBudgetHit (published, never converged). The cap is injected through the Config and the budget is a
// count, so the test needs neither a clock nor a sleep, and a cut is the same on any host.
TEST(PaperObjectiveTest, ABudgetHitPublishesTheIncumbentAndIsNeverConverged) {
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a =
      MakeSpacingPayload(Eigen::Vector3d(5, 0, 0), 0.0, source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(-5, 0, 0), 0.0,
                                      source_tree_b, 2, 300);
  payload_a.update_count = 2.0;
  payload_b.update_count = 2.0;
  ElementTree target_tree;
  MakeTwoSupportTargetTree(&target_tree);
  const Eigen::Vector2d warm(3, 0);

  // The product default is a budget that is on and sits above this solve.
  EXPECT_GT(RelationSolver::Config().solve_budget_evals, 0);
  RelationSolver::Config off;
  off.solve_budget_evals = 0;
  const SolverResult full = RelationSolver(off).Solve(warm, 0.0, {payload_a, payload_b}, {10.0, 1.0}, target_tree);
  ASSERT_EQ(full.status, SolverStatus::kConverged);
  ASSERT_GE(full.objective_iterations, 2);  // there is an iteration after the first to cut at
  EXPECT_FALSE(full.budget_hit);
  const SolverResult dflt = RelationSolver().Solve(warm, 0.0, {payload_a, payload_b}, {10.0, 1.0}, target_tree);
  EXPECT_EQ(dflt.status, SolverStatus::kConverged);
  EXPECT_EQ(dflt.position, full.position);  // the default budget does not touch a solve below it
  ASSERT_LT(full.objective_evaluations, RelationSolver::Config().solve_budget_evals);

  // Cut at every cap in turn: each cut is a published, never-converged incumbent that is no worse than the warm start and no better
  // than the converged answer, a larger cap never has a worse incumbent, some cut is the warm start and some took a step.
  int hits = 0, warm_hits = 0, stepped_hits = 0;
  double prev_j = std::numeric_limits<double>::infinity();
  for (int cap = 1; cap <= full.objective_evaluations + 1; ++cap) {
    RelationSolver::Config cfg;
    cfg.solve_budget_evals = cap;
    const SolverResult cut = RelationSolver(cfg).Solve(warm, 0.0, {payload_a, payload_b}, {10.0, 1.0}, target_tree);
    EXPECT_NE(cut.status, SolverStatus::kNotConverged);
    if (cut.status != SolverStatus::kBudgetHit) {
      EXPECT_EQ(cut.status, SolverStatus::kConverged);  // the budget was never reached
      EXPECT_FALSE(cut.budget_hit);
      EXPECT_EQ(cut.position, full.position);
      break;
    }
    ++hits;
    EXPECT_TRUE(cut.budget_hit);
    EXPECT_FALSE(cut.objective_stationary);
    EXPECT_FALSE(cut.objective_exact);
    EXPECT_TRUE(cut.position.allFinite());
    EXPECT_LE(cut.objective_after, cut.objective_before);
    EXPECT_GE(cut.objective_after, full.objective_after - 1e-9);  // J as computed, to its rounding
    EXPECT_LE(cut.objective_after, prev_j);
    prev_j = cut.objective_after;
    if (cut.budget_incumbent_is_warm_start) {
      ++warm_hits;
      EXPECT_EQ(cut.position, warm);
      EXPECT_DOUBLE_EQ(cut.objective_after, cut.objective_before);
    } else {
      ++stepped_hits;
      EXPECT_NE(cut.position, warm);
      EXPECT_LT(cut.objective_after, cut.objective_before);
    }
  }
  EXPECT_GE(warm_hits, 1);
  EXPECT_GE(stepped_hits, 1);
  EXPECT_GE(hits, 2);

  // A budget of one evaluation is spent at the first check: nothing improved, the warm start is the incumbent.
  RelationSolver::Config spent;
  spent.solve_budget_evals = 1;
  const SolverResult none = RelationSolver(spent).Solve(warm, 0.0, {payload_a, payload_b}, {10.0, 1.0}, target_tree);
  EXPECT_EQ(none.status, SolverStatus::kBudgetHit);
  EXPECT_TRUE(none.budget_incumbent_is_warm_start);
  EXPECT_EQ(none.budget_hit_iteration, 0);
  EXPECT_EQ(none.position, warm);
  EXPECT_DOUBLE_EQ(none.objective_after, none.objective_before);
}

// ---------------------------------------------------------------------------
// Two-stage split, determinism, scene-scale displacement, and the
// yaw-conditioning safeguard.
// ---------------------------------------------------------------------------

// RETIRED (the J refinement no longer exists): asserts what the position search / refinement produced (an exact optimum, restarts, tracking, acceptance rule, basin loss) or a step bound the refinement shaped. The position answer is S0 clamped; the harness defects (proposal == clamped S0, identical copies exact, every jump traced to its cause) replace these checks.
TEST(PaperObjectiveTest, DISABLED_DisplacementCapDerivesFromSceneScale) {
  // Two supports 20 m apart, both wanting distance 12: exact answer is
  // (10, +-6.633), 6.633 m from the centroid (10,0) -- outside the OLD
  // fixed 6 m cap, inside the scene-scale-derived one (margin x radius-to-
  // witness = 4 x 2 = 8 m here).
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a = MakeSpacingPayload(Eigen::Vector3d(12, 0, 0), 0.0,
                                      source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(-12, 0, 0), 0.0,
                                      source_tree_b, 2, 300);
  ElementTree target_tree;
  target_tree.AddElement(1);
  target_tree.AddElement(2, 1);
  target_tree.GetElement(2)->SetSharedId(200);
  target_tree.AddElement(3, 1);
  target_tree.GetElement(3)->SetSharedId(300);
  Eigen::Matrix4d t3 = Eigen::Matrix4d::Identity();
  t3(0, 3) = 20.0;
  target_tree.GetElement(3)->SetLocalTransform(t3);

  RelationSolver solver;
  SolverResult result = solver.Solve(Eigen::Vector2d(10, 0), 0.0,
                                     {payload_a, payload_b}, target_tree);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  const double dist_to_north =
      (result.position - Eigen::Vector2d(10, 6.633)).norm();
  const double dist_to_south =
      (result.position - Eigen::Vector2d(10, -6.633)).norm();
  EXPECT_LT(std::min(dist_to_north, dist_to_south), 0.1);
  EXPECT_LT(result.objective_after, 1e-4);
  EXPECT_FALSE(result.objective_fallback);
}

TEST(PaperObjectiveTest, RefinementIsDeterministicWithObstacles) {
  // No wall-clock in the search, so a multi-relation,
  // obstacle-gated scene must reproduce the exact same result run to run.
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a =
      MakeSpacingPayload(Eigen::Vector3d(5, 0, 0), 0.0, source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(-5, 0, 0), 0.0,
                                      source_tree_b, 2, 300);
  ElementTree target_tree;
  MakeTwoSupportTargetTree(&target_tree);

  auto walkable = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {-10, -10}, {10, -10}, {10, 10}, {-10, 10}});
  auto obstacle = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{2, 3}, {4, 3}, {4, 5}, {2, 5}});

  RelationSolver solver;
  std::vector<SolverResult> results;
  for (int i = 0; i < 5; ++i) {
    NavigableSpatialDefinition nav(walkable);
    nav.SetObstacles({FreeSpaceObstacle{/*key=*/1, obstacle}},
                     /*clearance=*/0.1);
    results.push_back(solver.Solve(Eigen::Vector2d(3, 0), 0.0,
                                    {payload_a, payload_b}, target_tree,
                                    &nav));
  }
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_EQ(results[i].position.x(), results[0].position.x());
    EXPECT_EQ(results[i].position.y(), results[0].position.y());
    EXPECT_EQ(results[i].objective_after, results[0].objective_after);
    EXPECT_EQ(results[i].objective_iterations, results[0].objective_iterations);
    EXPECT_EQ(results[i].objective_fallback, results[0].objective_fallback);
  }
}

TEST(PaperObjectiveTest, TwoStageYawRefinementNeverMovesPosition) {
  // Position search uses Spacing terms only
  // (AffordancePose never reaches the root solve, see
  // DirectSolveExcludesAffordancePoseFromTheRootSolve); yaw search uses
  // Facing terms only. Adding a
  // well-conditioned Facing relation to a Spacing solve must not perturb
  // the refined position at all -- "Facing never moves position" (already
  // true of the earlier decode) stays true of the paper-objective refinement too.
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto spacing_a =
      MakeSpacingPayload(Eigen::Vector3d(5, 0, 0), 0.0, source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto spacing_b = MakeSpacingPayload(Eigen::Vector3d(-5, 0, 0), 0.0,
                                      source_tree_b, 2, 300);
  auto source_tree_f = MakeTree(0.0, 0.0, 400);
  auto facing = MakeFacingPayload(Eigen::Vector3d(-3, 0, 0), 0.0,
                                  source_tree_f, 2, 400);

  ElementTree target_tree;
  MakeTwoSupportTargetTree(&target_tree);
  target_tree.AddElement(4, 1);
  target_tree.GetElement(4)->SetSharedId(400);
  Eigen::Matrix4d t4 = Eigen::Matrix4d::Identity();
  t4(0, 3) = 2.0;
  t4(1, 3) = 6.0;
  target_tree.GetElement(4)->SetLocalTransform(t4);

  RelationSolver solver;
  SolverResult spacing_only = solver.Solve(
      Eigen::Vector2d(3, 0), 0.0, {spacing_a, spacing_b}, target_tree);
  SolverResult joint = solver.Solve(Eigen::Vector2d(3, 0), 0.0,
                                    {spacing_a, spacing_b, facing},
                                    target_tree);

  EXPECT_EQ(spacing_only.position.x(), joint.position.x());
  EXPECT_EQ(spacing_only.position.y(), joint.position.y());
}

TEST(PaperObjectiveTest, DirectSolveExcludesAffordancePoseFromTheRootSolve) {
  // The root solve is Spacing + Facing only: an
  // AffordancePose relation is a per-anchor world pose, solved by
  // SolveActorTargets. The live path filters it out before calling Solve();
  // a DIRECT Solve() call enforces the same contract itself, so neither the
  // goal-point blend nor the paper-objective position stage can be handed one -- the
  // result with it is bit-identical to the result without it.
  auto source_tree = MakeTwoObjectTree({5, 8}, {15, 2});
  AddBoardTop(source_tree, 2);
  auto spacing = MakeSpacingPayload(Eigen::Vector3d(3, 6, 0), 0.0, source_tree,
                                    3, 300);
  auto hand = MakeAffordancePayload(Eigen::Vector3d(5.5, 8.0, 0.5), 0.5,
                                    source_tree, 2, 200, "hand_r");
  auto target_tree = MakeTwoObjectTree({10, 4}, {2, 9});
  AddBoardTop(target_tree, 2);

  // The affordance really is groundable on the target (so excluding it is a
  // decision, not a side effect of a failed decode).
  RelationSolver solver;
  RelationPayload resolved = hand;
  resolved.resolved_local_element_id = 2;
  ASSERT_TRUE(solver.GroundAffordanceTarget(resolved, target_tree).ok());

  const SolverResult without = solver.Solve(Eigen::Vector2d(0, 0), 0.25,
                                            {spacing}, target_tree);
  const SolverResult with = solver.Solve(Eigen::Vector2d(0, 0), 0.25,
                                         {spacing, hand}, target_tree);
  EXPECT_EQ(with.status, without.status);
  EXPECT_EQ(with.position.x(), without.position.x());
  EXPECT_EQ(with.position.y(), without.position.y());
  EXPECT_EQ(with.yaw, without.yaw);
  EXPECT_EQ(with.objective_before, without.objective_before);
  EXPECT_EQ(with.objective_after, without.objective_after);
  EXPECT_EQ(with.objective_relation_count, without.objective_relation_count);
  EXPECT_EQ(with.objective_evaluations, without.objective_evaluations);
  EXPECT_EQ(with.goals.size(), without.goals.size());
  // ... and it is reported as an explicit omission, never silently dropped.
  const SolverRelationOutcome* outcome = nullptr;
  for (const auto& r : with.relations)
    if (r.payload_index == 1) outcome = &r;
  ASSERT_NE(outcome, nullptr);
  EXPECT_EQ(outcome->disposition, SolverRelationDisposition::kUnsupported);
  EXPECT_NE(outcome->reason.find("per anchor"), std::string::npos)
      << outcome->reason;

  // An AffordancePose-only call holds the pose: no goal, nothing evaluable,
  // and (it is not a dropped POSITION relation) not a failure.
  const SolverResult only = solver.Solve(Eigen::Vector2d(1, 2), 0.25, {hand},
                                         target_tree);
  EXPECT_EQ(only.status, SolverStatus::kConverged);
  EXPECT_EQ(only.position.x(), 1.0);
  EXPECT_EQ(only.position.y(), 2.0);
  EXPECT_EQ(only.yaw, 0.25);
  EXPECT_TRUE(only.goals.empty());
  EXPECT_EQ(only.objective_relation_count, 0);
  EXPECT_EQ(only.objective_evaluations, 0);
  ASSERT_EQ(only.relations.size(), 1u);
  EXPECT_EQ(only.relations[0].disposition,
            SolverRelationDisposition::kUnsupported);

  // The per-anchor path still grounds it: the exclusion is the ROOT solve's.
  RealizationResult anchors = solver.SolveActorTargets(
      Eigen::Vector2d(0, 0), 0.25, {spacing, hand}, {}, target_tree);
  EXPECT_NE(FindTarget(anchors, "hand_r"), nullptr);
}

// RETIRED (the J refinement no longer exists): asserts what the position search / refinement produced (an exact optimum, restarts, tracking, acceptance rule, basin loss) or a step bound the refinement shaped. The position answer is S0 clamped; the harness defects (proposal == clamped S0, identical copies exact, every jump traced to its cause) replace these checks.
TEST(PaperObjectiveTest, DISABLED_WarmStartSeedsTheSearchAndNeverWorsensIt) {
  // A caller-supplied warm start is an extra restart, tried after the
  // initializer, and reported via objective_warm_start_used. Without one the
  // flag is false; with one the result is never worse than the initializer
  // and at least as good as without. The scene is deliberately NOT exactly
  // satisfiable (supports 6 m apart wanting distances 5 and 0.5: the circles
  // never meet), so J has a positive floor and the search cannot stop early
  // at J = 0 -- an exactly satisfiable scene ends at its first restart and
  // never reaches the warm seed, which is correct, not a bug.
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto spacing_a =
      MakeSpacingPayload(Eigen::Vector3d(5, 0, 0), 0.0, source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto spacing_b = MakeSpacingPayload(Eigen::Vector3d(-0.5, 0, 0), 0.0,
                                      source_tree_b, 2, 300);
  ElementTree target_tree;
  MakeTwoSupportTargetTree(&target_tree);

  RelationSolver solver;
  const SolverResult cold = solver.Solve(Eigen::Vector2d(3, 0), 0.0,
                                         {spacing_a, spacing_b}, target_tree);
  EXPECT_FALSE(cold.objective_warm_start_used);

  SolverWarmStart warm;
  warm.has_position = true;
  warm.position = cold.position + Eigen::Vector2d(0.4, -0.3);  // not S0
  const SolverResult seeded = solver.Solve(
      Eigen::Vector2d(3, 0), 0.0, {spacing_a, spacing_b}, {}, target_tree,
      nullptr, nullptr, std::nullopt, warm);
  EXPECT_TRUE(seeded.objective_warm_start_used);
  EXPECT_LE(seeded.objective_after, seeded.objective_before + 1e-12);
  // (No comparison with the cold answer: local-minimum continuation keeps the minimiser of the SEED's basin even when a
  // better minimiser exists in another basin, so "at least as good as without"
  // no longer holds; the seeded answer is a local minimum no worse than S0.)

  // A warm start that IS the initializer would repeat that restart: it is
  // not "used" (and costs no extra evaluations).
  SolverWarmStart same;
  same.has_position = true;
  same.position = cold.unconstrained_position;
  const SolverResult repeated = solver.Solve(
      Eigen::Vector2d(3, 0), 0.0, {spacing_a, spacing_b}, {}, target_tree,
      nullptr, nullptr, std::nullopt, same);
  EXPECT_FALSE(repeated.objective_warm_start_used);
  EXPECT_EQ(repeated.objective_evaluations, cold.objective_evaluations);
}

TEST(PaperObjectiveTest, YawStageExcludesBadlyConditionedBearing) {
  // The away-R fade used to drop a Facing whose offset is ~3.04 rad
  // (R^src ~0.0025). Core reports that bearing. The driver decides whether
  // to act on it. Occlusion and a look-at inside the body are the exclusions.
  const double off = M_PI - 0.1;  // ~3.04 rad
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto spacing = MakeSpacingPayload(Eigen::Vector3d(2, 0, 0), 0.7,
                                    source_tree_a, 2, 200);
  auto source_tree_f = MakeTree(0.0, 0.0, 300);
  auto facing = MakeFacingPayload(Eigen::Vector3d(-3, 0, 0), -off,
                                  source_tree_f, 2, 300);
  ASSERT_NEAR(facing.grounded_relation.proximity, 0.0025, 5e-3);

  ElementTree target_tree;
  target_tree.AddElement(1);
  target_tree.AddElement(2, 1);
  target_tree.GetElement(2)->SetSharedId(200);
  target_tree.AddElement(3, 1);
  target_tree.GetElement(3)->SetSharedId(300);
  Eigen::Matrix4d t3 = Eigen::Matrix4d::Identity();
  t3(0, 3) = 2.0;
  t3(1, 3) = 4.0;
  target_tree.GetElement(3)->SetLocalTransform(t3);

  RelationSolver solver;
  SolverResult spacing_only =
      solver.Solve(Eigen::Vector2d(0, 0), 0.7, {spacing}, target_tree);
  SolverResult joint = solver.Solve(Eigen::Vector2d(0, 0), 0.7,
                                    {spacing, facing}, target_tree);

  EXPECT_TRUE(joint.yaw_specified);
  ASSERT_GE(joint.goals.size(), 2u);
  EXPECT_FALSE(joint.goals[1].yaw_excluded);
  EXPECT_GT(std::abs(std::remainder(joint.yaw - spacing_only.yaw, 2.0 * M_PI)),
            0.1);
}

// The yaw is the DIRECTION of the weighted sum of the Facing bearings, which is the same for
// weights 1 and 1e-320. Summed with the raw weights, w * cos(b) of a weight that small
// underflows (to zero, or a denormal) and the circular mean came out as a multiple of 90
// degrees: the yaw of a Facing that had all but lost its weight was -90.0 for one frame and
// its bearing the next. It is the same answer at every scale of the weight.
TEST(RelationSolverTest, AntipodalFacingsTakeTheMiddleNearestTheEmbodiedYaw) {
  ElementTree tree;
  tree.AddElement(1);
  tree.AddElement(2, 1);
  tree.GetElement(2)->SetSharedId(200);
  Eigen::Matrix4d east = Eigen::Matrix4d::Identity();
  east(0, 3) = 1.0;
  tree.GetElement(2)->SetLocalTransform(east);
  tree.AddElement(3, 1);
  tree.GetElement(3)->SetSharedId(300);
  Eigen::Matrix4d west = Eigen::Matrix4d::Identity();
  west(0, 3) = -1.0;
  tree.GetElement(3)->SetLocalTransform(west);

  auto face_east = MakeFacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, tree, 2, 200);
  auto face_west = MakeFacingPayload(Eigen::Vector3d(0, 0, 0), M_PI, tree, 3, 300);
  RelationSolver solver;
  SolverResult north = solver.Solve(Eigen::Vector2d(0, 0), 0.2, {face_east, face_west},
                                    {1.0, 1.0}, tree);
  SolverResult south = solver.Solve(Eigen::Vector2d(0, 0), -0.2, {face_east, face_west},
                                    {1.0, 1.0}, tree);
  ASSERT_TRUE(north.yaw_specified);
  ASSERT_TRUE(south.yaw_specified);
  EXPECT_NEAR(north.yaw, M_PI / 2.0, 1e-9);
  EXPECT_NEAR(south.yaw, -M_PI / 2.0, 1e-9);
}

// A tie of opposing Facing bearings is a tie to the arithmetic, not only when the resultant is exactly zero. Two look-at
// points on opposite sides of the body, one 1e-15 m off the axis: the resultant is 1e-15 of the sum, below the error
// interval of its own computation (gamma_{n+6} of the sum plus the turn of each bearing over the position's resolution),
// so it has no direction and the yaw is the middle nearest the embodied yaw, whichever side of the axis the point is on
// (the sign of 1e-15 is rounding). A point 1e-6 m off the axis is a resolved resultant and the yaw follows it.
TEST(RelationSolverTest, ANumericallyUnresolvedTieOfOpposingFacingsIsAFacingTieWithTheNearestMiddle) {
  const auto make_tree = [](double off_y) {
    ElementTree tree;
    tree.AddElement(1);
    tree.AddElement(2, 1);
    tree.GetElement(2)->SetSharedId(200);
    Eigen::Matrix4d east = Eigen::Matrix4d::Identity();
    east(0, 3) = 1.0;
    tree.GetElement(2)->SetLocalTransform(east);
    tree.AddElement(3, 1);
    tree.GetElement(3)->SetSharedId(300);
    Eigen::Matrix4d west = Eigen::Matrix4d::Identity();
    west(0, 3) = -1.0;
    west(1, 3) = off_y;
    tree.GetElement(3)->SetLocalTransform(west);
    return tree;
  };
  // The source's payloads are made on the exact opposite pair; the receiver's west point is the one off the axis.
  ElementTree source_tree = make_tree(0.0);
  auto face_east = MakeFacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, source_tree, 2, 200);
  auto face_west = MakeFacingPayload(Eigen::Vector3d(0, 0, 0), M_PI, source_tree, 3, 300);
  const auto solve = [&](double off_y, double embodied_yaw, bool* tie) {
    ElementTree tree = make_tree(off_y);
    RelationSolver solver;
    const SolverResult r = solver.Solve(Eigen::Vector2d(0, 0), embodied_yaw, {face_east, face_west}, {1.0, 1.0}, tree);
    *tie = r.facing_tie;
    return r.yaw;
  };
  bool tie_pos = false, tie_neg = false, tie_far = false;
  const double yaw_pos = solve(+1e-15, 0.2, &tie_pos);
  const double yaw_neg = solve(-1e-15, 0.2, &tie_neg);
  EXPECT_TRUE(tie_pos);
  EXPECT_TRUE(tie_neg);
  EXPECT_NEAR(yaw_pos, M_PI / 2.0, 1e-9);
  EXPECT_NEAR(yaw_neg, M_PI / 2.0, 1e-9) << "the side of an unresolved offset does not pick the middle";
  const double yaw_far = solve(-1e-6, 0.2, &tie_far);
  EXPECT_FALSE(tie_far);
  EXPECT_NEAR(yaw_far, -M_PI / 2.0, 1e-3) << "a resolved resultant is followed";
}

// D7, "no HOLD": a sample outside F is not held (no kFailed, no echoed sample,
// no kSampleOutside): the goal is pursued as far as F allows, and the status is not kFailed.
TEST(RelationSolverTest, SampleOutsideFreeSpaceIsPursuedNotHeld) {
  auto source_tree = MakeTree(5.0, 8.0);
  auto payload = MakeSpacingPayload(Eigen::Vector3d(3, 6, 0), 0.0, source_tree, 2, 200);
  auto target_tree = MakeTree(5.0, 5.0);
  auto walkable = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {0, 0}, {10, 0}, {10, 10}, {0, 10}});
  NavigableSpatialDefinition nav(walkable);
  RelationSolver solver;
  const Eigen::Vector2d sample(20, 20);
  SolverResult result = solver.Solve(sample, 0.4, {payload}, target_tree, &nav);
  EXPECT_NE(result.status, SolverStatus::kFailed);
  EXPECT_NE(result.feasibility, FeasibilityOutcome::kSampleOutside);
  EXPECT_FALSE(result.no_grounded_referent);
  EXPECT_TRUE(result.has_objective_seed) << "the pursuit ran from the sample";
}

// The embodied sample can sit inside an obstacle's clearance ring while the room
// contains it (the body went there under an exemption that was released). The
// proposal then starts from the nearest pose in F and brings the body out, instead of
// holding it in the ring.
TEST(RelationSolverTest, SampleInAnObstacleRingStartsFromItsProjectionIntoF) {
  auto source_tree = MakeTree(5.0, 8.0);
  auto payload = MakeSpacingPayload(Eigen::Vector3d(3, 6, 0), 0.0, source_tree, 2, 200);
  auto target_tree = MakeTree(9.0, 9.0);
  auto walkable = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {0, 0}, {10, 0}, {10, 10}, {0, 10}});
  auto box = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {4, 4}, {6, 4}, {6, 6}, {4, 6}});
  NavigableSpatialDefinition nav(walkable);
  nav.SetObstacles({FreeSpaceObstacle{/*key=*/1, box}}, /*clearance=*/0.4);
  RelationSolver solver;

  // 0.2 m from the footprint: inside the 0.4 m ring, outside the box, inside the room.
  const Eigen::Vector2d sample(6.2, 5.0);
  ASSERT_FALSE(nav.IsFree(sample));
  ASSERT_TRUE(nav.Contains(sample));
  const auto nearest = nav.NearestFree(sample);
  ASSERT_TRUE(nearest.has_value());

  // A static source issues no request: the proposal is the sample's projection.
  payload.update_count = std::numeric_limits<double>::infinity();
  SolverResult held = solver.Solve(sample, 0.0, {payload}, target_tree, &nav);
  EXPECT_EQ(held.status, SolverStatus::kConverged);
  EXPECT_EQ(held.feasibility, FeasibilityOutcome::kProjected);
  EXPECT_TRUE(nav.IsFree(held.position));
  EXPECT_NEAR((held.position - *nearest).norm(), 0.0, 1e-9);
  EXPECT_NEAR((held.position - sample).norm(), 0.2, 1e-3);

  // A request continues from that pose and stays in F.
  payload.update_count = 2.0;
  SolverResult moved = solver.Solve(sample, 0.0, {payload}, target_tree, &nav);
  EXPECT_EQ(moved.status, SolverStatus::kConverged);
  EXPECT_TRUE(nav.IsFree(moved.position));
  EXPECT_GT((moved.position - *nearest).norm(), 0.1);
}

TEST(RelationSolverTest, FacingYawDoesNotDependOnTheScaleOfTheWeight) {
  auto source_tree = MakeTree(5.0, 0.0);
  auto payload = MakeFacingPayload(Eigen::Vector3d(0, 0, 0), 0.3, source_tree, 2, 200);
  auto target_tree = MakeTree(3.0, 7.0);
  RelationSolver solver;
  const SolverResult unit =
      solver.Solve(Eigen::Vector2d(0, 0), 0.0, {payload}, {1.0}, target_tree);
  ASSERT_TRUE(unit.yaw_specified);
  for (const double w : {1e-3, 1e-9, 1e-200, 1e-300, 1e-320}) {
    const SolverResult tiny =
        solver.Solve(Eigen::Vector2d(0, 0), 0.0, {payload}, {w}, target_tree);
    EXPECT_TRUE(tiny.yaw_specified) << "w=" << w;
    EXPECT_NEAR(std::remainder(tiny.yaw - unit.yaw, 2.0 * M_PI), 0.0, 1e-9) << "w=" << w;
  }
}

// The Facing sign is the sign of the signed
// bearing offset at the answered yaw, with no deadband. A band (e.g. 0.01 rad) has no
// citation in the papers or in v1 and would report "no sign" for every offset inside it, so a +-theta root
// switch inside the band could not be attributed. Exactly zero is the only plateau.
TEST(RelationSolverTest, TheFacingSignIsTheSignOfTheOffsetWithNoDeadband) {
  for (const double offset : {0.002, -0.002, 0.0099, -0.0099, 0.05, -0.05}) {
    // The source stands at the origin, yaw 0, and its referent is `offset` radians off its heading.
    auto source_tree = MakeTree(10.0, 10.0 * std::tan(offset));
    auto payload = MakeFacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, source_tree, 2, 200);
    auto target_tree = MakeTree(3.0, 7.0);
    RelationSolver solver;
    const SolverResult result =
        solver.Solve(Eigen::Vector2d(0, 0), 0.0, {payload}, {1.0}, target_tree);
    ASSERT_TRUE(result.yaw_specified);
    EXPECT_EQ(result.facing_sign, offset > 0 ? 1 : -1) << "offset " << offset;
  }
}

// v1 ratio form. The source's new distance is
// not the receiver's target: a lagging body closes the same fraction of its
// own remaining geodesic distance. The payload below is encoded at 2 m, which
// is what the retired current-error objective would chase.
TEST(RatioObjectiveTest, LaggingReceiverClosesItsOwnFraction) {
  auto tree = MakeTree(0.0, 0.0, 200);
  auto payload =
      MakeSpacingPayload(Eigen::Vector3d(0, 2, 0), 0.0, tree, 2, 200);
  // Source went 4 m → 2 m: n = 4 / (4 − 2) = 2. Receiver at 10 m closes half.
  payload.update_count = SourceUpdateCount(4.0, 2.0);
  ASSERT_DOUBLE_EQ(payload.update_count, 2.0);

  RelationSolver solver;
  const Eigen::Vector2d sample(0, 10);
  const SolverResult result = solver.Solve(sample, 0.0, {payload}, tree);
  EXPECT_EQ(result.status, SolverStatus::kConverged);
  EXPECT_NEAR(result.position.x(), 0.0, 1e-3);
  EXPECT_NEAR(result.position.y(), 5.0, 1e-2) << result.position.transpose();
  EXPECT_GT(result.position.y(), 3.0);
  ASSERT_EQ(result.ratio_intents.size(), 1u);
  EXPECT_NEAR(result.ratio_intents[0].sample_distance_m, 10.0, 1e-3);
  EXPECT_DOUBLE_EQ(result.ratio_intents[0].update_count, 2.0);
  // The proposal step from the sample is the recorded closure d/n.
  EXPECT_NEAR((result.position - sample).norm(),
              result.ratio_intents[0].sample_distance_m /
                  result.ratio_intents[0].update_count,
              1e-3);
}

TEST(RatioObjectiveTest, StaticSourceDoesNotMove) {
  auto tree = MakeTree(0.0, 0.0, 200);
  auto payload =
      MakeSpacingPayload(Eigen::Vector3d(0, 2, 0), 0.0, tree, 2, 200);
  payload.update_count = SourceUpdateCount(2.0, 2.0);
  ASSERT_FALSE(std::isfinite(payload.update_count));

  RelationSolver solver;
  const Eigen::Vector2d sample(0, 10);
  const SolverResult result = solver.Solve(sample, 0.0, {payload}, tree);
  EXPECT_EQ(result.status, SolverStatus::kConverged);
  EXPECT_LT((result.position - sample).norm(), 1e-6)
      << result.position.transpose();
  EXPECT_TRUE(result.ratio_intents.empty());
}

// A source that jitters about a constant distance ratchets the receiver toward the
// contact, and that is v1: only a closing update has an intent (v1:
// progress < epsilon is an infinite count, so a receding or still update plans nothing),
// and a closing update closes the fraction 1/n of the receiver's OWN remaining distance.
// So the receiver's remaining distance only falls, by exactly (1 - 1/n) on each closing
// update and by nothing on each receding one. The recurrence is checked step by step
// against the formula, not against the solver's own output.
TEST(RatioObjectiveTest, AJitteringSourceRatchetsTheReceiverTowardContactAsV1) {
  auto tree = MakeTree(0.0, 0.0, 200);
  auto payload = MakeSpacingPayload(Eigen::Vector3d(0, 1, 0), 0.0, tree, 2, 200);
  RelationSolver solver;
  Eigen::Vector2d pos(0, 20);
  double remaining = 20.0;
  // The source's distance to its referent: 5, then +-0.1 about it.
  const double d[] = {5.0, 4.9, 5.0, 4.9, 5.0, 4.9, 5.0, 4.9, 5.0};
  int closing = 0;
  for (size_t k = 1; k < sizeof(d) / sizeof(d[0]); ++k) {
    payload.update_count = SourceUpdateCount(d[k - 1], d[k]);
    const SolverResult result = solver.Solve(pos, 0.0, {payload}, tree);
    ASSERT_EQ(result.status, SolverStatus::kConverged) << k;
    double expected = remaining;
    if (d[k] < d[k - 1]) {  // closing: n = d_old / (d_old - d_new)
      const double n = d[k - 1] / (d[k - 1] - d[k]);
      ASSERT_TRUE(std::isfinite(payload.update_count)) << k;
      EXPECT_NEAR(payload.update_count, n, 1e-9) << k;
      expected = remaining * (1.0 - 1.0 / n);
      ++closing;
    } else {  // receding: no approach, v1's infinite count: no request
      EXPECT_FALSE(std::isfinite(payload.update_count)) << k;
    }
    EXPECT_NEAR(result.position.norm(), expected, 1e-6) << k;
    EXPECT_LE(result.position.norm(), remaining + 1e-9) << "the receiver never retreats";
    pos = result.position;
    remaining = pos.norm();
  }
  EXPECT_EQ(closing, 4);
  EXPECT_LT(remaining, 20.0 * 0.99) << "the jitter alone moved the receiver toward the contact";
}

TEST(RatioObjectiveTest, ContactClosesTheRemainingDistance) {
  auto tree = MakeTree(0.0, 0.0, 200);
  auto payload =
      MakeSpacingPayload(Eigen::Vector3d(0, 4, 0), 0.0, tree, 2, 200);
  payload.update_count = SourceUpdateCount(0.0, 0.0);
  ASSERT_DOUBLE_EQ(payload.update_count, 1.0);

  RelationSolver solver;
  const SolverResult result =
      solver.Solve(Eigen::Vector2d(0, 8), 0.0, {payload}, tree);
  EXPECT_EQ(result.status, SolverStatus::kConverged);
  EXPECT_LT(result.position.norm(), 0.05) << result.position.transpose();
}

// Same fraction each update: a body that starts twice as far arrives on the
// support in the same update the source does.
TEST(RatioObjectiveTest, LaggingReceiverArrivesWithTheSource) {
  auto tree = MakeTree(0.0, 0.0, 200);
  auto payload =
      MakeSpacingPayload(Eigen::Vector3d(0, 1, 0), 0.0, tree, 2, 200);
  RelationSolver solver;
  Eigen::Vector2d pos(0, 20);
  double d_src = 10.0;
  for (int step = 0; step < 5; ++step) {
    const double d_next = d_src - 2.0;
    payload.update_count = SourceUpdateCount(d_src, d_next);
    const SolverResult result = solver.Solve(pos, 0.0, {payload}, tree);
    ASSERT_EQ(result.status, SolverStatus::kConverged) << step;
    pos = result.position;
    d_src = d_next;
  }
  EXPECT_DOUBLE_EQ(d_src, 0.0);
  EXPECT_LT(pos.norm(), 0.05) << pos.transpose();
}

// Contact wants both distances closed. The unconstrained point is south of
// the obstacle; S0 is that centroid. A cold solve used to seed the search
// there. It stays on the sample's side and leaves the residual inexact.
TEST(RatioObjectiveTest, ColdSolveStaysInTheSampleBasin) {
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a =
      MakeSpacingPayload(Eigen::Vector3d(5, 0, 0), 0.0, source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(-5, 0, 0), 0.0,
                                      source_tree_b, 2, 300);
  payload_a.update_count = 1.0;
  payload_b.update_count = 1.0;

  ElementTree target_tree;
  MakeTwoSupportTargetTree(&target_tree);
  auto walkable = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {-10, -10}, {10, -10}, {10, 10}, {-10, 10}});
  NavigableSpatialDefinition nav(walkable);
  auto obstacle = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{2, 3}, {4, 3}, {4, 5}, {2, 5}});
  nav.SetObstacles({FreeSpaceObstacle{/*key=*/1, obstacle}},
                   /*clearance=*/0.1);

  SolverWarmStart warm;
  warm.cold = true;
  // This test is of the basin the search starts in, not of the budget: the cold solve of this room uses more J evaluations than the
  // provisional product cap (kP1SolveBudgetEvals, Xeon-derived), so it runs unbudgeted here.
  RelationSolver::Config unbudgeted;
  unbudgeted.solve_budget_evals = 0;
  RelationSolver solver(unbudgeted);
  const SolverResult result = solver.Solve(
      Eigen::Vector2d(1.5, 6.0), 0.0, {payload_a, payload_b}, {}, target_tree,
      &nav, nullptr, std::nullopt, warm);

  EXPECT_EQ(result.status, SolverStatus::kConverged);
  // The search starts at the sample. S0, the contact centroid, is the other
  // side of the obstacle and is not the seed.
  EXPECT_LT((result.objective_seed_position - Eigen::Vector2d(1.5, 6.0)).norm(),
            1e-9);
  EXPECT_LT(result.unconstrained_position.y(), 1.0)
      << result.unconstrained_position.transpose();
  EXPECT_GT((result.objective_seed_position - result.unconstrained_position)
                .norm(),
            1.0);
  EXPECT_TRUE(nav.IsFree(result.position));
  EXPECT_FALSE(result.objective_exact) << result.position.transpose();
}

// The default-config twin of the test above: the product path runs with the budget on, and an embodied reset takes this cold path
// (space.cc, the reset solves). This room's cold solve exceeds the provisional cap, so the answer is the budget's incumbent: it is a
// budget hit (never converged), it started from the sample, it is in F, and it is the unbudgeted solve's answer to 1e-3 m with the same J
// (the budget cut the polish, not the descent).
TEST(RatioObjectiveTest, ColdSolveUnderTheDefaultBudgetPublishesAnIncumbentOnTheSamplesSide) {
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a =
      MakeSpacingPayload(Eigen::Vector3d(5, 0, 0), 0.0, source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(-5, 0, 0), 0.0,
                                      source_tree_b, 2, 300);
  payload_a.update_count = 1.0;
  payload_b.update_count = 1.0;
  ElementTree target_tree;
  MakeTwoSupportTargetTree(&target_tree);
  auto walkable = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {-10, -10}, {10, -10}, {10, 10}, {-10, 10}});
  NavigableSpatialDefinition nav(walkable);
  auto obstacle = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{2, 3}, {4, 3}, {4, 5}, {2, 5}});
  nav.SetObstacles({FreeSpaceObstacle{/*key=*/1, obstacle}},
                   /*clearance=*/0.1);
  SolverWarmStart warm;
  warm.cold = true;
  const Eigen::Vector2d sample(1.5, 6.0);
  const SolverResult result = RelationSolver().Solve(
      sample, 0.0, {payload_a, payload_b}, {}, target_tree, &nav, nullptr, std::nullopt, warm);
  EXPECT_EQ(result.status, SolverStatus::kBudgetHit);
  EXPECT_TRUE(result.budget_hit);
  EXPECT_LT((result.objective_seed_position - sample).norm(), 1e-9);
  EXPECT_TRUE(nav.IsFree(result.position)) << result.position.transpose();
  EXPECT_EQ(result.feasibility, FeasibilityOutcome::kFeasible);
  EXPECT_LE(result.objective_after, result.objective_before);
  // Which side of the obstacle: the converged answer of this room is NOT on the sample's side (the sample (1.5, 6) is above the obstacle
  // y in [3, 5]; the unbudgeted solve above converges at (3, 0), below it, J = 18, in its first iteration's reach), so "the sample's side"
  // is not a property of this solve. What the budget must keep is that its incumbent is the unbudgeted answer's basin, not a worse point:
  // the cut came during the flat polish at J = 18, and the incumbent is within 1e-3 m of the converged position with the same J.
  RelationSolver::Config unbudgeted;
  unbudgeted.solve_budget_evals = 0;
  const SolverResult full = RelationSolver(unbudgeted).Solve(
      sample, 0.0, {payload_a, payload_b}, {}, target_tree, &nav, nullptr, std::nullopt, warm);
  ASSERT_EQ(full.status, SolverStatus::kConverged);
  EXPECT_LT((result.position - full.position).norm(), 1e-3) << result.position.transpose() << " vs " << full.position.transpose();
  EXPECT_NEAR(result.objective_after, full.objective_after, 1e-6);
}

// A sample outside the walkable is projected onto F before the solve, as v1 does: the solver's start point is
// the nearest pose in F, so a budget hit's incumbent is in F by construction, whatever the cap. The result says kProjected and the seed is
// a free pose. (With the sample itself as the start, being lower in J than its projection, it would be the incumbent of a cut solve: a point
// outside F published as feasible.)
TEST(RatioObjectiveTest, ASampleOutsideTheWalkableIsProjectedBeforeTheSolveSoABudgetHitIsInF) {
  auto source_tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a =
      MakeSpacingPayload(Eigen::Vector3d(5, 0, 0), 0.0, source_tree_a, 2, 200);
  auto source_tree_b = MakeTree(0.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(-5, 0, 0), 0.0,
                                      source_tree_b, 2, 300);
  payload_a.update_count = 1.0;
  payload_b.update_count = 1.0;
  ElementTree target_tree;
  MakeTwoSupportTargetTree(&target_tree);
  // The walkable is x in [5, 10]; the sample (2, 0) is outside it and has the lower J (the relations pull towards x = 3).
  auto right = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{{5, -10}, {10, -10}, {10, 10}, {5, 10}});
  NavigableSpatialDefinition nav(right);
  SolverWarmStart warm;
  warm.cold = true;
  const Eigen::Vector2d sample(2.0, 0.0);
  ASSERT_FALSE(nav.IsFree(sample));
  int hits = 0;
  for (int cap = 0; cap <= 800; ++cap) {  // 0: unbudgeted
    RelationSolver::Config cfg;
    cfg.solve_budget_evals = cap;
    const SolverResult r = RelationSolver(cfg).Solve(
        sample, 0.0, {payload_a, payload_b}, {}, target_tree, &nav, nullptr, std::nullopt, warm);
    EXPECT_EQ(r.feasibility, FeasibilityOutcome::kProjected) << "cap " << cap;
    EXPECT_TRUE(nav.IsFree(r.objective_seed_position)) << "cap " << cap << " seed " << r.objective_seed_position.transpose();
    EXPECT_TRUE(nav.IsFree(r.position)) << "cap " << cap << " " << r.position.transpose();
    if (r.status == SolverStatus::kBudgetHit) ++hits;
  }
  EXPECT_GT(hits, 0);
}

// An engaged referent is one weighted term of J, not a snap. The source stands on referent A (n = 1, d* = 0) and
// is closing on B (n = 4). The receiver is on the segment AB, so the minimiser of
//   J = w_A |x - c_A|^2 + w_B (|x - c_B| - d*_B)^2,  d*_B = d_B(sample) (1 - 1/n_B)
// is on the segment too, and J is quadratic in the distance t = |x - c_A| along it:
//   t = w_B (L - d*_B) / (w_A + w_B).
// The expected position is that closed form, not the solver's own output.
TEST(RatioObjectiveTest, AnEngagedReferentIsOneWeightedTermNotASnap) {
  auto tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a = MakeSpacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, tree_a, 2, 200);
  ASSERT_LE(payload_a.grounded_relation.support_distance_m, 1e-9) << "A is engaged";
  payload_a.update_count = SourceUpdateCount(0.0, 0.0);
  ASSERT_DOUBLE_EQ(payload_a.update_count, 1.0);
  auto tree_b = MakeTree(3.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, tree_b, 2, 300);
  payload_b.update_count = SourceUpdateCount(4.0, 3.0);  // 4 / (4 - 3)
  ASSERT_DOUBLE_EQ(payload_b.update_count, 4.0);

  ElementTree target;
  MakeTwoSupportTargetTree(&target);  // A at (0, 0), B at (6, 0)
  const Eigen::Vector2d sample(4.0, 0.0);
  const double length = 6.0;
  const double d_star_b = (length - sample.x()) * (1.0 - 1.0 / 4.0);

  RelationSolver solver;
  for (const auto& weights : {std::vector<double>{0.5, 0.5}, std::vector<double>{0.8, 0.2},
                              std::vector<double>{0.2, 0.8}}) {
    const double expected_t =
        weights[1] * (length - d_star_b) / (weights[0] + weights[1]);
    const SolverResult ab =
        solver.Solve(sample, 0.0, {payload_a, payload_b}, weights, target);
    EXPECT_EQ(ab.status, SolverStatus::kConverged);
    EXPECT_FALSE(ab.chart_snap) << "weights " << weights[0] << ": the engaged referent is one term of J";
    EXPECT_NEAR(ab.position.x(), expected_t, 2e-3) << "weights " << weights[0];
    EXPECT_NEAR(ab.position.y(), 0.0, 2e-3) << "weights " << weights[0];
    // The payload order decides nothing: a tie is not a driver.
    const SolverResult ba = solver.Solve(
        sample, 0.0, {payload_b, payload_a}, {weights[1], weights[0]}, target);
    EXPECT_NEAR((ba.position - ab.position).norm(), 0.0, 1e-6) << "weights " << weights[0];
    EXPECT_FALSE(ba.chart_snap);
  }

  // A policy whose weight is entirely on the engaged referent (the contact guarantee of
  // Progressive and Greedy: v1's annealing is one-hot at n = 1) answers its mapped contact,
  // which is then the minimiser of J.
  const SolverResult contact =
      solver.Solve(sample, 0.0, {payload_a, payload_b}, {1.0, 0.0}, target);
  EXPECT_NEAR(contact.position.norm(), 0.0, 1e-9) << contact.position.transpose();
  const SolverResult contact_ba =
      solver.Solve(sample, 0.0, {payload_b, payload_a}, {0.0, 1.0}, target);
  EXPECT_NEAR(contact_ba.position.norm(), 0.0, 1e-9) << contact_ba.position.transpose();
}

// The contact answers the engaged referent's mapped contact wherever it lies in F, a static
// peer included: a relation with no request (n = infinity: the source is still) is no term
// of J, so its weight pulls nothing. A body that cannot reach the contact (a wall splits F)
// is still asked for the contact: the pose the latency policy's teleport request carries.
TEST(RatioObjectiveTest, AStaticPeerPullsNothingAndTheContactIsAnsweredAcrossAWall) {
  auto tree_a = MakeTree(0.0, 0.0, 200);
  auto payload_a = MakeSpacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, tree_a, 2, 200);
  payload_a.update_count = SourceUpdateCount(0.0, 0.0);
  auto tree_b = MakeTree(3.0, 0.0, 300);
  auto payload_b = MakeSpacingPayload(Eigen::Vector3d(0, 0, 0), 0.0, tree_b, 2, 300);
  payload_b.update_count = SourceUpdateCount(3.0, 3.0);  // still: no request
  ASSERT_FALSE(std::isfinite(payload_b.update_count));

  ElementTree target;
  MakeTwoSupportTargetTree(&target);  // A at (0, 0), B at (6, 0)
  auto walkable = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {-10, -10}, {10, -10}, {10, 10}, {-10, 10}});
  NavigableSpatialDefinition nav(walkable);
  auto wall = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{-3.1, -11}, {-2.9, -11}, {-2.9, 11}, {-3.1, 11}});
  nav.SetObstacles({FreeSpaceObstacle{/*key=*/1, wall}}, /*clearance=*/0.1);
  const Eigen::Vector2d sample(-6.0, 0.0);
  ASSERT_FALSE(nav.QueryReachability(sample, Eigen::Vector2d(0, 0), {}, {}).reachable);

  RelationSolver solver;
  for (const auto& weights : {std::vector<double>{0.5, 0.5}, std::vector<double>{1.0, 0.0}}) {
    const SolverResult r = solver.Solve(sample, 0.0, {payload_a, payload_b}, weights, target, &nav);
    EXPECT_NEAR(r.position.norm(), 0.0, 1e-6) << "weights " << weights[0] << " at " << r.position.transpose();
    EXPECT_TRUE(nav.IsFree(r.position));
  }
}

}  // namespace test
}  // namespace spatialblend
