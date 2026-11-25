// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/sim/simulator.h"

#include <cmath>

#include <gtest/gtest.h>

#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/space/builder/kernel_shorthands.h"

namespace spatialblend {
namespace test {

namespace {
// Scenario replay tests use immediate (teleport) mode so the avatar reaches
// the solver target each frame — the tests validate spatial correctness, not
// motion smoothness.
const Simulator::RunConfig kImmediate{
    .max_linear_speed = 1.4,
    .max_angular_speed = 2.0,
    .use_velocity_bounded = false};
}  // namespace

// ---------------------------------------------------------------------------
// 1. DiningScenarioReplay
//    Source 30x30 room with Chair at (15,12), Target 12x12 with Chair at (6,4).
//    Actor walks from (3,3) towards Chair_A over 5 seconds.
// ---------------------------------------------------------------------------
TEST(ScenarioReplayTest, DiningScenarioReplay) {
  auto sim = Simulator::Create();

  // Source space (30x30)
  auto sa = sim->AddSpace();
  auto room_a = sa.AddElement("Room")
                    .WithPolygon({{0, 0}, {30, 0}, {30, 30}, {0, 30}})
                    .WithSharedId(100);
  auto chair_a = room_a.AddChild("Chair")
                     .At(15, 12)
                     .WithBoundary({{14, 11}, {16, 11}, {16, 13}, {14, 13}})
                     .WithSharedId(200);
  auto actor =
      sa.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_a);
  actor.Bind("root", Spacing());

  // Target space (12x12)
  auto sb = sim->AddSpace();
  auto room_b = sb.AddElement("Room")
                    .WithPolygon({{0, 0}, {12, 0}, {12, 12}, {0, 12}})
                    .WithSharedId(100);
  auto chair_b = room_b.AddChild("Chair")
                     .At(6, 4)
                     .WithBoundary({{5, 3}, {7, 3}, {7, 5}, {5, 5}})
                     .WithSharedId(200);
  auto avatar =
      sb.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_b);

  // Actor walks from (3,3) to Chair_A position (15,12) over 5 seconds
  auto traj = Trajectory::StraightLine({3, 3}, {15, 12}, 5.0);
  auto result = sim->Run(actor, avatar, traj, 0.033, kImmediate);

  // Geodesic solver may place avatar near the target immediately.
  // Travel might be small if solver converges at the decoded target point.
  EXPECT_LT(result.FinalDistanceTo({6, 4}), 8.0);
}

// ---------------------------------------------------------------------------
// 2. PresentationScenarioReplay
//    Source 20x20: Board(5,15), Table(15,5).
//    Target 15x15: Board(10,3), Table(3,10).
//    Actor walks to Board_A, binds spacing + facing.
//    Avatar should end closer to Board_B than Table_B.
// ---------------------------------------------------------------------------
TEST(ScenarioReplayTest, PresentationScenarioReplay) {
  auto sim = Simulator::Create();

  // Source space (20x20)
  auto sa = sim->AddSpace();
  auto room_a = sa.AddElement("Room")
                    .WithPolygon({{0, 0}, {20, 0}, {20, 20}, {0, 20}})
                    .WithSharedId(100);
  auto board_a = room_a.AddChild("Board")
                     .At(5, 15)
                     .WithBoundary({{3, 13}, {7, 13}, {7, 17}, {3, 17}})
                     .WithSharedId(200);
  auto table_a = room_a.AddChild("Table")
                     .At(15, 5)
                     .WithBoundary({{13, 3}, {17, 3}, {17, 7}, {13, 7}})
                     .WithSharedId(300);
  auto actor =
      sa.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_a);
  actor.Bind("root", Spacing());
  actor.Bind("root", Facing());

  // Target space (15x15)
  auto sb = sim->AddSpace();
  auto room_b = sb.AddElement("Room")
                    .WithPolygon({{0, 0}, {15, 0}, {15, 15}, {0, 15}})
                    .WithSharedId(100);
  auto board_b = room_b.AddChild("Board")
                     .At(5, 12)
                     .WithBoundary({{3, 10}, {7, 10}, {7, 14}, {3, 14}})
                     .WithSharedId(200);
  auto table_b = room_b.AddChild("Table")
                     .At(12, 2)
                     .WithBoundary({{10, 0.5}, {14, 0.5}, {14, 3.5}, {10, 3.5}})
                     .WithSharedId(300);
  auto avatar =
      sb.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_b);

  // Actor walks from near Table_A all the way to Board_A
  auto traj = Trajectory::StraightLine({15, 5}, {5, 15}, 5.0);
  auto result = sim->Run(actor, avatar, traj, 0.033, kImmediate);

  ASSERT_FALSE(result.avatar_positions.empty());
  auto final_pos = result.avatar_positions.back();
  Eigen::Vector2d board_b_pos(5, 12);
  Eigen::Vector2d table_b_pos(12, 2);
  double dist_to_board = (final_pos - board_b_pos).norm();
  double dist_to_table = (final_pos - table_b_pos).norm();
  // Avatar should end closer to Board_B (same shared ID as Board_A, the target)
  EXPECT_LT(dist_to_board, dist_to_table)
      << "Avatar at (" << final_pos.x() << ", " << final_pos.y()
      << ") dist_to_board=" << dist_to_board
      << " dist_to_table=" << dist_to_table;
}

// ---------------------------------------------------------------------------
// 3. WalkingTourReplay
//    Actor visits 3 objects in sequence via waypoints.
// ---------------------------------------------------------------------------
TEST(ScenarioReplayTest, WalkingTourReplay) {
  auto sim = Simulator::Create();

  // Source space (30x30) with Board, Table, Chair
  auto sa = sim->AddSpace();
  auto room_a = sa.AddElement("Room")
                    .WithPolygon({{0, 0}, {30, 0}, {30, 30}, {0, 30}})
                    .WithSharedId(100);
  room_a.AddChild("Board")
      .At(5, 25)
      .WithBoundary({{3, 23}, {7, 23}, {7, 27}, {3, 27}})
      .WithSharedId(200);
  room_a.AddChild("Table")
      .At(25, 25)
      .WithBoundary({{23, 23}, {27, 23}, {27, 27}, {23, 27}})
      .WithSharedId(300);
  room_a.AddChild("Chair")
      .At(15, 5)
      .WithBoundary({{14, 4}, {16, 4}, {16, 6}, {14, 6}})
      .WithSharedId(400);
  auto actor =
      sa.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_a);
  actor.Bind("root", Spacing());

  // Target space (20x20)
  auto sb = sim->AddSpace();
  auto room_b = sb.AddElement("Room")
                    .WithPolygon({{0, 0}, {20, 0}, {20, 20}, {0, 20}})
                    .WithSharedId(100);
  room_b.AddChild("Board")
      .At(3, 17)
      .WithBoundary({{1, 15}, {5, 15}, {5, 19}, {1, 19}})
      .WithSharedId(200);
  room_b.AddChild("Table")
      .At(17, 17)
      .WithBoundary({{15, 15}, {19, 15}, {19, 19}, {15, 19}})
      .WithSharedId(300);
  room_b.AddChild("Chair")
      .At(10, 3)
      .WithBoundary({{9, 2}, {11, 2}, {11, 4}, {9, 4}})
      .WithSharedId(400);
  auto avatar =
      sb.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_b);

  // Actor visits Board -> Table -> Chair via waypoints
  auto traj = Trajectory::FromWaypoints({
      {0.0, {2, 2}, 0.0},
      {3.0, {5, 25}, 0.0},   // Board
      {6.0, {25, 25}, 0.0},  // Table
      {9.0, {15, 5}, 0.0},   // Chair
  });
  auto result = sim->Run(actor, avatar, traj, 0.1, kImmediate);

  EXPECT_GT(result.num_frames, 0);
  EXPECT_GT(result.TotalAvatarTravel(), 2.0);
}

// ---------------------------------------------------------------------------
// 4. AsymmetricRoomSizeReplay
//    Source 30x30, Target 12x12. Actor traverses full source room.
//    Avatar must stay inside the target polygon at all times.
// ---------------------------------------------------------------------------
TEST(ScenarioReplayTest, AsymmetricRoomSizeReplay) {
  auto sim = Simulator::Create();

  // Source space (30x30)
  auto sa = sim->AddSpace();
  auto room_a = sa.AddElement("Room")
                    .WithPolygon({{0, 0}, {30, 0}, {30, 30}, {0, 30}})
                    .WithSharedId(100);
  room_a.AddChild("Marker")
      .At(25, 25)
      .WithBoundary({{24, 24}, {26, 24}, {26, 26}, {24, 26}})
      .WithSharedId(200);
  auto actor =
      sa.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_a);
  actor.Bind("root", Spacing());

  // Target space (12x12)
  auto sb = sim->AddSpace();
  auto room_b = sb.AddElement("Room")
                    .WithPolygon({{0, 0}, {12, 0}, {12, 12}, {0, 12}})
                    .WithSharedId(100);
  room_b.AddChild("Marker")
      .At(10, 10)
      .WithBoundary({{9, 9}, {11, 9}, {11, 11}, {9, 11}})
      .WithSharedId(200);
  auto avatar =
      sb.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_b);

  // Actor traverses from one corner to the opposite
  auto traj = Trajectory::StraightLine({2, 2}, {28, 28}, 5.0);
  auto result = sim->Run(actor, avatar, traj, 0.033, kImmediate);

  Polygon2D target_polygon({{0, 0}, {12, 0}, {12, 12}, {0, 12}});
  EXPECT_TRUE(result.AvatarAlwaysInPolygon(target_polygon));
}

// ---------------------------------------------------------------------------
// 5. ContactSideReplay
//    Two runs: actor approaches Board from LEFT vs RIGHT.
//    Final avatar positions should differ.
// ---------------------------------------------------------------------------
TEST(ScenarioReplayTest, ContactSideReplay) {
  // Helper lambda to build a simulator with Board at (5, 15) in source
  // and Board at (10, 3) in target.
  auto build_sim = []() {
    auto sim = Simulator::Create();

    auto sa = sim->AddSpace();
    auto room_a = sa.AddElement("Room")
                      .WithPolygon({{0, 0}, {20, 0}, {20, 20}, {0, 20}})
                      .WithSharedId(100);
    room_a.AddChild("Board")
        .At(5, 15)
        .WithBoundary({{3, 13}, {7, 13}, {7, 17}, {3, 17}})
        .WithSharedId(200);
    auto actor =
        sa.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_a);
    actor.Bind("root", Spacing());

    auto sb = sim->AddSpace();
    auto room_b = sb.AddElement("Room")
                      .WithPolygon({{0, 0}, {15, 0}, {15, 15}, {0, 15}})
                      .WithSharedId(100);
    room_b.AddChild("Board")
        .At(10, 3)
        .WithBoundary({{8, 1}, {12, 1}, {12, 5}, {8, 5}})
        .WithSharedId(200);
    auto avatar =
        sb.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_b);

    struct SimSetup {
      std::unique_ptr<Simulator> sim;
      ActorHandle actor;
      ActorHandle avatar;
    };
    return SimSetup{std::move(sim), std::move(actor), std::move(avatar)};
  };

  // Run 1: approach Board from the left
  auto setup_left = build_sim();
  auto traj_left = Trajectory::StraightLine({1, 15}, {3, 15}, 2.0);
  auto result_left =
      setup_left.sim->Run(setup_left.actor, setup_left.avatar, traj_left, 0.1,
                          kImmediate);

  // Run 2: approach Board from the right
  auto setup_right = build_sim();
  auto traj_right = Trajectory::StraightLine({9, 15}, {7, 15}, 2.0);
  auto result_right = setup_right.sim->Run(
      setup_right.actor, setup_right.avatar, traj_right, 0.1, kImmediate);

  // The two approaches should both produce valid positions near Board_B.
  // With the geodesic solver, contact-side differentiation depends on the
  // grounding rule's contact_normalized mapping.  Both should end near Board_B.
  ASSERT_FALSE(result_left.avatar_positions.empty());
  ASSERT_FALSE(result_right.avatar_positions.empty());
  auto left_final = result_left.avatar_positions.back();
  auto right_final = result_right.avatar_positions.back();
  // Both should be in the vicinity of Board_B at (10, 3).
  EXPECT_LT((left_final - Eigen::Vector2d(10, 3)).norm(), 10.0);
  EXPECT_LT((right_final - Eigen::Vector2d(10, 3)).norm(), 10.0);
}

// ---------------------------------------------------------------------------
// 6. AffordancePoseReplay
//    Source 20x20: Table(10,15) with boundary.
//    Target 15x15: Table(8,4) with boundary.
//    Actor has "hand" anchor at (0.4, 0, 0.9) and binds AffordancePose.
//    Actor walks to Table, avatar should end near target Table.
// ---------------------------------------------------------------------------
// Asserts a per-anchor "hand" affordance target decoded onto the target
// furniture (with height). Produced by the two-stage SolveActorTargets
// realizer, wired into OnRemoteUpdate: a hysteresis-gated,
// reach-clamped RealizationTarget is appended per (AffordancePose, anchor)
// group after the root target, restoring what this test originally covered
// before the upstream single-root-Solve merge retired the per-anchor path.
TEST(ScenarioReplayTest, AffordancePoseReplay) {
  auto sim = Simulator::Create();

  auto sa = sim->AddSpace();
  auto room_a = sa.AddElement("Room")
                    .WithPolygon({{0, 0}, {20, 0}, {20, 20}, {0, 20}})
                    .WithSharedId(100);
  // NOTE: no .At(...) here, deliberately. NavigableSpatialDefinition's
  // Contains/ProjectOnto operate directly on the given polygon (already
  // absolute world coordinates below) with no composition of the element
  // tree's own world transform (element/navigable_spatial_definition.cc has
  // no such composition). AffordancePose's per-anchor "where" decode
  // (DecodeSupportPoint / contact_normalized) DOES read the tree's world
  // transform for contact_point_local (affordance_pose_grounding_rule.cc).
  // Combining a non-identity .At(...) with an already-absolute boundary
  // therefore feeds two disagreeing frames into that decode — exactly the
  // latent inconsistency flagged as "not
  // exercised by existing tests ... likely relevant to the anchor-specific
  // target emission." This is the first test to combine AffordancePose with
  // a boundary-owning referent, so it is the first to hit it. Leaving the
  // tree transform at its identity default (by omitting .At) keeps both
  // frames in agreement without touching relation/ (owned by a concurrent
  // agent); Spacing-only tests elsewhere in this file are unaffected by
  // this same combination because Spacing's root-position solve never
  // consults contact_normalized.
  auto table_a = room_a.AddChild("Table")
                     .WithBoundary({{9, 14}, {11, 14}, {11, 16}, {9, 16}})
                     .WithSharedId(200);
  auto actor = sa.AddActor()
                   .WithAnchor("root", 0, 0, 0)
                   .WithAnchor("hand", 0.4, 0, 0.9)
                   .InScope(room_a);
  actor.Bind("hand", AffordancePose());

  auto sb = sim->AddSpace();
  auto room_b = sb.AddElement("Room")
                    .WithPolygon({{0, 0}, {15, 0}, {15, 15}, {0, 15}})
                    .WithSharedId(100);
  // See the matching NOTE on table_a above — same reasoning.
  auto table_b = room_b.AddChild("Table")
                     .WithBoundary({{7, 3}, {9, 3}, {9, 5}, {7, 5}})
                     .WithSharedId(200);
  auto avatar = sb.AddActor()
                    .WithAnchor("root", 0, 0, 0)
                    .WithAnchor("hand", 0.4, 0, 0.9)
                    .InScope(room_b);
  // NOTE: the default "hand" reach radius (~0.9m, a realistic arm's
  // length — see AppendAnchorTargets's ClampToReach in space.cc) is by
  // design meant to keep an anchor target physically near the avatar's own
  // (unmoved) root. This test binds AffordancePose only — no Spacing — so
  // the root deliberately never approaches Table_B (see the
  // TotalAvatarTravel assertion below); at default reach the resulting hand
  // target would therefore always clamp down near the stationary root,
  // regardless of whether the cross-space decode onto Table_B landed in the
  // right place. That decode correctness (not the clamp) is what this
  // legacy scenario replay is restoring/covering, so the reach is widened
  // here to exceed both rooms' diagonals. The clamp engaging (and gating
  // hysteresis, and the trace's "anchor" field) is covered by dedicated
  // tests in space/relation_solver_test.cc (ClampToReachTest,
  // UpdateAnchorGateTest) and simulator/anchor_realization_test.cc.
  sb.GetSpace()->SetAnchorReach("hand", 25.0);

  // Actor walks from (5,5) to Table_A at (10,15)
  auto traj = Trajectory::StraightLine({5, 5}, {10, 15}, 5.0);
  auto result = sim->Run(actor, avatar, traj, 0.033, kImmediate);

  EXPECT_GT(result.num_frames, 0);
  // Two-stage realizer: an AffordancePose binding drives the HAND
  // anchor target, never root locomotion. With no spacing relation bound,
  // the avatar root must NOT chase the table.
  EXPECT_LT(result.TotalAvatarTravel(), 0.5);
  // The hand target must appear and land on Table_B's support, with height.
  const RealizationTarget* hand = nullptr;
  for (auto it = result.targets.rbegin(); it != result.targets.rend(); ++it) {
    for (const auto& t : *it) {
      if (t.anchor_name == "hand") {
        hand = &t;
        break;
      }
    }
    if (hand) break;
  }
  ASSERT_NE(hand, nullptr) << "no hand anchor target was ever emitted";
  EXPECT_LT((hand->position - Eigen::Vector2d(8, 4)).norm(), 2.0);
  EXPECT_TRUE(hand->has_z);
  // Avatar should stay inside target room
  Polygon2D target_room({{0, 0}, {15, 0}, {15, 15}, {0, 15}});
  EXPECT_TRUE(result.AvatarAlwaysInPolygon(target_room));
}

// ---------------------------------------------------------------------------
// 7. AllKernelsReplay
//    Source has 3 objects, actor binds Spacing(root), Facing(chest),
//    AffordancePose(hand). Walks toward object A. Avatar should converge.
// ---------------------------------------------------------------------------
TEST(ScenarioReplayTest, AllKernelsReplay) {
  auto sim = Simulator::Create();

  auto sa = sim->AddSpace();
  auto room_a = sa.AddElement("Room")
                    .WithPolygon({{0, 0}, {20, 0}, {20, 20}, {0, 20}})
                    .WithSharedId(100);
  room_a.AddChild("Sofa").At(5, 15).WithSharedId(200)
      .WithBoundary({{3, 14}, {7, 14}, {7, 16}, {3, 16}});
  room_a.AddChild("Table").At(15, 5).WithSharedId(300)
      .WithBoundary({{14, 4}, {16, 4}, {16, 6}, {14, 6}});
  auto actor = sa.AddActor()
                   .WithAnchor("root", 0, 0, 0)
                   .WithAnchor("chest", 0, 0, 1.2)
                   .WithAnchor("hand", 0.4, 0, 0.9)
                   .InScope(room_a);
  actor.Bind("root", Spacing());
  actor.Bind("chest", Facing());
  actor.Bind("hand", AffordancePose());

  auto sb = sim->AddSpace();
  auto room_b = sb.AddElement("Room")
                    .WithPolygon({{0, 0}, {15, 0}, {15, 15}, {0, 15}})
                    .WithSharedId(100);
  room_b.AddChild("Sofa").At(10, 3).WithSharedId(200)
      .WithBoundary({{8, 2}, {12, 2}, {12, 4}, {8, 4}});
  room_b.AddChild("Table").At(3, 10).WithSharedId(300)
      .WithBoundary({{2, 9}, {4, 9}, {4, 11}, {2, 11}});
  auto avatar = sb.AddActor()
                    .WithAnchor("root", 0, 0, 0)
                    .WithAnchor("chest", 0, 0, 1.2)
                    .WithAnchor("hand", 0.4, 0, 0.9)
                    .InScope(room_b);

  // Actor walks toward Sofa_A
  auto traj = Trajectory::StraightLine({10, 10}, {5, 15}, 5.0);
  auto result = sim->Run(actor, avatar, traj, 0.033);

  EXPECT_GT(result.num_frames, 0);
  // Avatar should stay inside target room
  Polygon2D target_room({{0, 0}, {15, 0}, {15, 15}, {0, 15}});
  EXPECT_TRUE(result.AvatarAlwaysInPolygon(target_room));
  // Should have finite positions (no NaN)
  for (const auto& p : result.avatar_positions) {
    EXPECT_TRUE(std::isfinite(p.x()));
    EXPECT_TRUE(std::isfinite(p.y()));
  }
  // Max speed bounded
  EXPECT_LE(result.MaxSpeed(), 1.4 * 1.1);
}

}  // namespace test
}  // namespace spatialblend
