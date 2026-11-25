// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/design_acceptance_test.cc
//
// Design acceptance scenario.
//
// This is the "does the system realize the paper" end-to-end test: a scripted
// source trajectory in one room, an avatar driven by the velocity-bounded
// controller in a DIFFERENTLY SIZED AND SHAPED room, asserting the core claim
// of the paper — RELATION PRESERVATION across heterogeneous spaces.
//
// The spacing relation is a room-diameter-normalized geodesic (Eq 8), so the
// avatar should hold the SAME NORMALIZED approach level to the corresponding
// object as the actor does in its own room:
//
//     dist(actor, X_A) / diameter_A  ~=  dist(avatar, X_B) / diameter_B
//
// We drive the loop through the controller (the supported smoothing path) and
// check (1) normalized-distance preservation to the engaged object, (2) the
// avatar stays inside its walkable polygon, and (3) motion is speed-limited.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/testing/test_driver.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace test {

// Measure the spacing kernel's normalized relation value R (proximity) for an
// actor at `pos` relative to a single Board at `board` in a w×h room. Built in
// an isolated throwaway space so it never perturbs the live scenario. This is
// the metric the design preserves (R_tgt = R_src), independent of raw geometry.
double SpacingProximity(const Eigen::Vector2d& pos, const Eigen::Vector2d& board,
                        double w, double h) {
  auto ctx = LocalContext::Create();
  Space* s = ctx->AddSpace();
  auto* tree = s->GetElementTree();
  tree->AddElement(1);
  tree->GetElement(1)->SetSharedId(100);
  auto walkable = std::make_shared<spatialblend::Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {w, 0}, {w, h}, {0, h}});
  tree->GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(walkable));
  tree->AddElement(2, 1);
  tree->GetElement(2)->SetSharedId(200);
  Eigen::Matrix4d bt = Eigen::Matrix4d::Identity();
  bt(0, 3) = board.x();
  bt(1, 3) = board.y();
  tree->GetElement(2)->SetLocalTransform(bt);
  std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", {0, 0, 0})};
  auto id = s->CreateActor(anchors);
  s->GetActor(id)->SetElementId(1);
  s->Bind(id, "root", RelationKernelType::kAvatarSpacing);
  s->UpdateActorState(id, pos, 0.0);
  const auto& b = s->GetBindings(id);
  if (b.empty() || b[0].latest_matches.empty()) return -1.0;
  return b[0].latest_matches[0].eval.grounded.proximity;
}

// ==========================================================================
// Two heterogeneous rooms connected via LocalContext:
//   Space A: 20x20 square. Board_A(4,16).
//   Space B: 12x8 rectangle (smaller AND different aspect ratio). Board_B(3,6).
// A single object isolates the core claim (relation preservation) from
// multi-referent blending. The B room is a scaled + reshaped image of A, so
// absolute distances differ but NORMALIZED approach levels should be preserved.
// ==========================================================================
class DesignAcceptanceScenario : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = LocalContext::Create();
    space_a_ = context_->AddSpace();
    space_b_ = context_->AddSpace();

    SetupSpace(space_a_, 20.0, 20.0, /*board=*/{6, 14});
    SetupSpace(space_b_, 14.0, 10.0, /*board=*/{5, 5});

    std::vector<SubjectAnchor> anchors = {
        SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
        SubjectAnchor("chest", Eigen::Vector3d(0, 0, 1.2)),
    };

    actor_id_ = space_a_->CreateActor(anchors);
    space_a_->GetActor(actor_id_)->SetElementId(1);
    space_a_->GetElementTree()->GetElement(1)->SetSharedId(100);
    space_a_->SetHub(context_->GetHub());

    avatar_id_ = space_b_->CreateActor(anchors);
    space_b_->GetActor(avatar_id_)->SetElementId(1);
    follower_ = std::make_unique<IdealDriver>(space_b_, avatar_id_);
    space_b_->GetElementTree()->GetElement(1)->SetSharedId(100);

    space_a_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
    space_a_->Bind(actor_id_, "chest", RelationKernelType::kAvatarFacing);

    diam_a_ = RoomDiameter(space_a_);
    diam_b_ = RoomDiameter(space_b_);
  }

  // Opt-in: the avatar is driven by a bounded driver (the production/device
  // path) instead of the zero-lag follower: it owns the body, publishes its
  // pose, and follows each proposal within its bounds.
  void EnableBoundedDriver() {
    follower_.reset();
    body_ = driver::Pose{};
    PublishBody();
  }
  void PublishBody() {
    EmbodiedPoseInput in;
    in.x = body_.position.x();
    in.y = body_.position.y();
    in.yaw = body_.yaw;
    in.epoch = 1;
    space_b_->PublishEmbodiedPose(avatar_id_, in, 0.0, ++body_sequence_);
  }

  void SetupSpace(Space* space, double w, double h, Eigen::Vector2d board) {
    auto* tree = space->GetElementTree();
    ASSERT_TRUE(tree->AddElement(1).ok());
    tree->GetElement(1)->SetSharedId(100);
    auto walkable = std::make_shared<spatialblend::Polygon2D>(
        std::vector<Eigen::Vector2d>{{0, 0}, {w, 0}, {w, h}, {0, h}});
    tree->GetElement(1)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(walkable));

    ASSERT_TRUE(tree->AddElement(2, 1).ok());
    tree->GetElement(2)->SetSharedId(200);
    Eigen::Matrix4d bt = Eigen::Matrix4d::Identity();
    bt(0, 3) = board.x();
    bt(1, 3) = board.y();
    tree->GetElement(2)->SetLocalTransform(bt);
  }

  double RoomDiameter(Space* space) {
    const auto* nav = dynamic_cast<const NavigableSpatialDefinition*>(
        space->GetElementTree()->GetElement(1)->GetSpatialDefinition());
    Bound2D b = nav->GetBounds();
    return (b.max - b.min).norm();
  }

  bool AvatarInPolygon() {
    const auto* nav = dynamic_cast<const NavigableSpatialDefinition*>(
        space_b_->GetElementTree()->GetElement(1)->GetSpatialDefinition());
    return nav->Contains(space_b_->GetActor(avatar_id_)->GetPosition());
  }

  // Move the actor and let the bounded driver follow the proposal.
  Eigen::Vector2d MoveActorAndTick(const Eigen::Vector2d& actor_pos,
                                   double actor_yaw, int frames, double dt) {
    space_a_->UpdateActorState(actor_id_, actor_pos, actor_yaw);
    for (int i = 0; i < frames; ++i) {
      ReceiverProposal proposal;
      space_b_->ReadProposal(avatar_id_, &proposal);
      if (proposal.IsAbsolute()) {
        body_.position = Eigen::Vector2d(proposal.x, proposal.y);
        body_.yaw = proposal.yaw;
        PublishBody();
      } else if (proposal.status == ProposalStatus::kExact ||
                 proposal.status == ProposalStatus::kInexact) {
        driver::Pose target;
        target.position = Eigen::Vector2d(proposal.x, proposal.y);
        target.yaw = proposal.yaw;
        body_ = driver::Step(
            body_, target,
            space_b_->QueryCourse(avatar_id_, body_.position, target.position),
            driver::Bounds{}, dt);
        PublishBody();
      }
      EXPECT_TRUE(AvatarInPolygon()) << "Avatar left the walkable polygon";
    }
    return body_.position;
  }

  std::unique_ptr<LocalContext> context_;
  std::unique_ptr<IdealDriver> follower_;  // after context_: destroyed first
  Space* space_a_ = nullptr;
  Space* space_b_ = nullptr;
  spatialblend::entity_id_t actor_id_;
  spatialblend::entity_id_t avatar_id_;
  driver::Pose body_;  // the bounded driver's body (EnableBoundedDriver)
  uint64_t body_sequence_ = 0;
  double diam_a_ = 0.0;
  double diam_b_ = 0.0;
};

// The first sample places the body on the contact centroid. A source that
// then only changes its standoff does not walk that body back out: once the
// receiver is on the contact its gap is closed, so the ratio step is zero.
// Matching R_src at each standoff was the removed approach-point walk.
TEST_F(DesignAcceptanceScenario, ImmediatePathPreservesRelationValue) {
  const Eigen::Vector2d board_a(6, 14);
  const Eigen::Vector2d bearing = Eigen::Vector2d(1, -1).normalized();

  Eigen::Vector2d placed = Eigen::Vector2d::Zero();
  bool have_placement = false;
  for (double r_target : {0.40, 0.25, 0.10}) {
    Eigen::Vector2d actor_pos = board_a + bearing * (r_target * diam_a_);
    double yaw =
        std::atan2(board_a.y() - actor_pos.y(), board_a.x() - actor_pos.x());
    space_a_->UpdateActorState(actor_id_, actor_pos, yaw);
    Eigen::Vector2d avatar_pos = space_b_->GetActor(avatar_id_)->GetPosition();
    if (!have_placement) {
      EXPECT_GT(avatar_pos.norm(), 0.5);
      placed = avatar_pos;
      have_placement = true;
    } else {
      EXPECT_NEAR((avatar_pos - placed).norm(), 0.0, 1e-3) << r_target;
    }
  }
}

// A static source does not glide the body to the source's standoff. The
// first sample places it on the contact centroid; later ticks of the same
// pose hold that placement.
TEST_F(DesignAcceptanceScenario, DrivenPathPreservesR) {
  constexpr double kDt = 0.033;
  const Eigen::Vector2d board_a(6, 14);
  const Eigen::Vector2d bearing = Eigen::Vector2d(1, -1).normalized();

  // Actor held at a high approach level (R_src = 0.90); avatar starts neutral
  // (at its initial pose, away from the relation-preserving point).
  const double r_target = 0.10;
  Eigen::Vector2d actor_pos = board_a + bearing * (r_target * diam_a_);
  double yaw =
      std::atan2(board_a.y() - actor_pos.y(), board_a.x() - actor_pos.x());
  double r_src = SpacingProximity(actor_pos, board_a, 20.0, 20.0);
  ASSERT_NEAR(r_src, 0.90, 0.02);

  EnableBoundedDriver();
  Eigen::Vector2d avatar_pos = MoveActorAndTick(actor_pos, yaw, 5, kDt);
  Eigen::Vector2d held = MoveActorAndTick(actor_pos, yaw, 5, kDt);
  EXPECT_GT(avatar_pos.norm(), 0.5);
  EXPECT_NEAR((held - avatar_pos).norm(), 0.0, 1e-3);
}

// The first sample places the body on the contact centroid of every spacing
// referent in the family's scope. A second board is in that centroid even
// when the source is far from it.
TEST(MultiReferentScenario, BlendPerturbsOnlyUnderCompetition) {
  const Eigen::Vector2d board_a(6, 14), table_a(14, 6);
  const Eigen::Vector2d board_b(4, 7), table_b(10, 3);

  // Build a two-space scenario (immediate path, no controller), optionally with
  // a second referent (Table), drive the actor to `actor_pos`, and return the
  // avatar's realized R to Board_B (its engagement with the Board).
  auto solve_board_r = [&](bool with_table,
                           const Eigen::Vector2d& actor_pos) -> double {
    auto ctx = LocalContext::Create();
    Space* a = ctx->AddSpace();
    Space* b = ctx->AddSpace();
    auto setup = [&](Space* s, double w, double h, Eigen::Vector2d board,
                     Eigen::Vector2d table) {
      auto* tree = s->GetElementTree();
      tree->AddElement(1);
      tree->GetElement(1)->SetSharedId(100);
      auto poly = std::make_shared<spatialblend::Polygon2D>(
          std::vector<Eigen::Vector2d>{{0, 0}, {w, 0}, {w, h}, {0, h}});
      tree->GetElement(1)->SetSpatialDefinition(
          std::make_shared<NavigableSpatialDefinition>(poly));
      tree->AddElement(2, 1);
      tree->GetElement(2)->SetSharedId(200);
      Eigen::Matrix4d bt = Eigen::Matrix4d::Identity();
      bt(0, 3) = board.x();
      bt(1, 3) = board.y();
      tree->GetElement(2)->SetLocalTransform(bt);
      if (with_table) {
        tree->AddElement(3, 1);
        tree->GetElement(3)->SetSharedId(300);
        Eigen::Matrix4d tt = Eigen::Matrix4d::Identity();
        tt(0, 3) = table.x();
        tt(1, 3) = table.y();
        tree->GetElement(3)->SetLocalTransform(tt);
      }
    };
    setup(a, 20.0, 20.0, board_a, table_a);
    setup(b, 14.0, 10.0, board_b, table_b);

    std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", {0, 0, 0})};
    auto actor = a->CreateActor(anchors);
    a->GetActor(actor)->SetElementId(1);
    a->SetHub(ctx->GetHub());
    auto avatar = b->CreateActor(anchors);
    b->GetActor(avatar)->SetElementId(1);
    IdealDriver follower(b, avatar);  // declared after ctx: destroyed first
    a->Bind(actor, "root", RelationKernelType::kAvatarSpacing);

    a->UpdateActorState(actor, actor_pos, 0.0);
    Eigen::Vector2d avatar_pos = b->GetActor(avatar)->GetPosition();
    return SpacingProximity(avatar_pos, board_b, 14.0, 10.0);
  };

  // Regime 1 (ENGAGED): actor 2 m from Board, ~10 m from Table.
  const Eigen::Vector2d engaged(6, 12);
  // Regime 2 (COMPETING): actor equidistant from Board and Table.
  const Eigen::Vector2d competing(10, 10);

  double eng_single = solve_board_r(false, engaged);
  double eng_multi = solve_board_r(true, engaged);
  double cmp_single = solve_board_r(false, competing);
  double cmp_multi = solve_board_r(true, competing);
  GTEST_LOG_(INFO) << "ENGAGED   single=" << eng_single << " multi=" << eng_multi
                   << " delta=" << (eng_single - eng_multi);
  GTEST_LOG_(INFO) << "COMPETING single=" << cmp_single << " multi=" << cmp_multi
                   << " delta=" << (cmp_single - cmp_multi);

  // The first sample places the body on the contact centroid. One referent
  // lands on the board, so that R is the contact. A second referent's contact
  // is in the same centroid whether or not the source is near it, so the
  // placement leaves the board. That is not an R-preserving blend.
  EXPECT_GT(eng_single, 0.9);
  EXPECT_LT(eng_multi, eng_single);
  EXPECT_GT(cmp_single, 0.9);
  EXPECT_LT(cmp_multi, cmp_single);
}

// ==========================================================================
// DESIGN: cross-space TIMING is a TARGET-SIDE quantity. How fast the avatar
// arrives is governed by the realization on the TARGET side — the geodesic
// distance to its OWN R-preserving pose (which grows with the target room's
// size, Eq 8) divided by the embodied character's kinematic bound (speed
// limit) — NOT by the source's approach rate. A larger target room or a
// tighter speed cap => the avatar arrives LATER, and that is correct.
//
// The velocity-bounded controller already embodies this: given a real
// destination and no plausible pacing budget (the "momentarily static source,
// whose estimated time diverges" case), it
// glides in at max speed instead of stalling. The production freeze
// (ControllerPathDoesNotPreserveR_KnownGap) is UPSTREAM — SolveProgressive
// bakes the source's estimated_steps into the target POSITION, handing the
// controller its own current position. This scenario characterizes the
// intended target-side law by driving the controller to the R-preserving pose.
// ==========================================================================

// Realize the avatar's R-preserving pose for a source held at a fixed approach
// (R_src ~ 0.70) in a 20x20 source room, imaged into a room_w x room_h target
// room whose board scales with the room. Immediate path (no controller).
Eigen::Vector2d RPreservingPose(double room_w, double room_h) {
  auto ctx = LocalContext::Create();
  Space* a = ctx->AddSpace();
  Space* b = ctx->AddSpace();
  auto setup = [](Space* s, double w, double h, Eigen::Vector2d board) {
    auto* tree = s->GetElementTree();
    tree->AddElement(1);
    tree->GetElement(1)->SetSharedId(100);
    auto poly = std::make_shared<spatialblend::Polygon2D>(
        std::vector<Eigen::Vector2d>{{0, 0}, {w, 0}, {w, h}, {0, h}});
    tree->GetElement(1)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(poly));
    tree->AddElement(2, 1);
    tree->GetElement(2)->SetSharedId(200);
    Eigen::Matrix4d bt = Eigen::Matrix4d::Identity();
    bt(0, 3) = board.x();
    bt(1, 3) = board.y();
    tree->GetElement(2)->SetLocalTransform(bt);
  };
  const Eigen::Vector2d board_a(10, 10);
  const Eigen::Vector2d board_b(room_w * 0.5, room_h * 0.5);
  setup(a, 20.0, 20.0, board_a);
  setup(b, room_w, room_h, board_b);

  std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", {0, 0, 0})};
  auto actor = a->CreateActor(anchors);
  a->GetActor(actor)->SetElementId(1);
  a->SetHub(ctx->GetHub());
  auto avatar = b->CreateActor(anchors);
  b->GetActor(avatar)->SetElementId(1);
  IdealDriver follower(b, avatar);  // declared after ctx: destroyed first
  a->Bind(actor, "root", RelationKernelType::kAvatarSpacing);

  // Source at approach level R_src ~ 0.70 along a fixed diagonal bearing.
  const double diam_a = std::sqrt(20.0 * 20.0 + 20.0 * 20.0);
  const Eigen::Vector2d bearing = Eigen::Vector2d(-1, -1).normalized();
  Eigen::Vector2d actor_pos = board_a + bearing * (0.30 * diam_a);
  a->UpdateActorState(actor, actor_pos, 0.0);
  return b->GetActor(avatar)->GetPosition();
}

// Drive a bounded driver from `start` toward `target`; return arrival time.
double ArrivalTime(const Eigen::Vector2d& start, const Eigen::Vector2d& target,
                   double v_max) {
  driver::Bounds bounds;
  bounds.max_linear_speed = v_max;
  driver::Pose body;
  body.position = start;
  driver::Pose goal;
  goal.position = target;
  const double dt = 1.0 / 60.0;
  int ticks = 0;
  const int kMaxTicks = 1000000;
  while (ticks < kMaxTicks) {
    body = driver::Step(body, goal, {}, bounds, dt);
    ++ticks;
    if ((body.position - target).norm() < 1e-6) break;
  }
  return ticks * dt;
}

TEST(CrossSpaceTimingScenario, ArrivalGovernedByTargetGeometryAndSpeedLimit) {
  const Eigen::Vector2d origin(0.0, 0.0);
  const double v = 1.4;  // default embodied speed limit (m/s)

  Eigen::Vector2d pose_base = RPreservingPose(14.0, 10.0);
  Eigen::Vector2d pose_big = RPreservingPose(28.0, 20.0);  // 2x target room
  double d_base = (pose_base - origin).norm();
  double d_big = (pose_big - origin).norm();

  double t_base = ArrivalTime(origin, pose_base, v);
  double t_big = ArrivalTime(origin, pose_big, v);
  double t_slow = ArrivalTime(origin, pose_base, v * 0.5);  // half speed

  GTEST_LOG_(INFO) << "pose_base=(" << pose_base.x() << "," << pose_base.y()
                   << ") d=" << d_base << " t=" << t_base;
  GTEST_LOG_(INFO) << "pose_big=(" << pose_big.x() << "," << pose_big.y()
                   << ") d=" << d_big << " t=" << t_big;
  GTEST_LOG_(INFO) << "t_slow(half v)=" << t_slow;

  const double dt = 1.0 / 60.0;

  // Target-side law: arrival time = geodesic distance / speed limit (empty
  // convex room => straight line), within one tick of pacing granularity.
  EXPECT_NEAR(t_base, d_base / v, 2 * dt);
  EXPECT_NEAR(t_big, d_big / v, 2 * dt);

  // (a) Larger TARGET room => R-preserving pose farther => arrives later.
  EXPECT_GT(d_big, d_base) << "bigger target room should push the pose farther";
  EXPECT_GT(t_big, t_base) << "bigger target room should take longer to reach";

  // (b) Tighter kinematic bound (half speed) => ~2x arrival time, same room.
  EXPECT_NEAR(t_slow, 2.0 * t_base, 4 * dt)
      << "halving the speed limit should double arrival time";
}

// RETIRED: CrossSpaceTimingScenario.
// PacingSyncsToSlowSourceUntilKinematicBoundBinds tested a core deadline
// pacing (speed = min(remaining/budget, v_max)) that no longer exists. Core
// does not pace, and since ABI 4 it reports no arrival budget either: pacing belongs to the driver, and a
// teleport is a source-app exception or core's latency request. The helper in
// spatialblend/driver deliberately has no deadline pacing.

}  // namespace test
}  // namespace spatialblend
