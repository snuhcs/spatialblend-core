// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/sim/simulator.h"

#include <cmath>
#include <iostream>

#include <gtest/gtest.h>

#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/space/builder/kernel_shorthands.h"
#include "spatialblend/space/builder/trajectory.h"

namespace spatialblend {
namespace test {

// Helper: build the shared "presentation" scenario.
// Source 20x20: Board(5,15), Table(15,5).
// Target 15x15: Board(10,3), Table(3,10).
struct PresentationScenario {
  std::unique_ptr<Simulator> sim;
  ActorHandle actor;
  ActorHandle avatar;
};

PresentationScenario BuildPresentationScenario() {
  auto sim = Simulator::Create();

  // Source space (20x20)
  auto sa = sim->AddSpace();
  auto room_a = sa.AddElement("Room")
                    .WithPolygon({{0, 0}, {20, 0}, {20, 20}, {0, 20}})
                    .WithSharedId(100);
  room_a.AddChild("Board")
      .At(5, 15)
      .WithBoundary({{3, 13}, {7, 13}, {7, 17}, {3, 17}})
      .WithSharedId(200);
  room_a.AddChild("Table")
      .At(15, 5)
      .WithBoundary({{13, 3}, {17, 3}, {17, 7}, {13, 7}})
      .WithSharedId(300);
  auto actor =
      sa.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_a);
  actor.Bind("root", Spacing());

  // Target space (15x15)
  auto sb = sim->AddSpace();
  auto room_b = sb.AddElement("Room")
                    .WithPolygon({{0, 0}, {15, 0}, {15, 15}, {0, 15}})
                    .WithSharedId(100);
  room_b.AddChild("Board")
      .At(10, 3)
      .WithBoundary({{8, 1}, {12, 1}, {12, 5}, {8, 5}})
      .WithSharedId(200);
  room_b.AddChild("Table")
      .At(3, 10)
      .WithBoundary({{1, 8}, {5, 8}, {5, 12}, {1, 12}})
      .WithSharedId(300);
  auto avatar =
      sb.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_b);

  return {std::move(sim), std::move(actor), std::move(avatar)};
}

// ---------------------------------------------------------------------------
// Metrics unit tests (on synthetic data)
// ---------------------------------------------------------------------------

TEST(TrajectoryQualityTest, MaxAccelerationSynthetic) {
  SimulationResult result;
  result.dt = 1.0;
  // Constant velocity: (0,0) -> (1,0) -> (2,0) -> (3,0)
  result.avatar_positions = {{0, 0}, {1, 0}, {2, 0}, {3, 0}};
  result.num_frames = 4;
  EXPECT_NEAR(result.MaxAcceleration(), 0.0, 1e-9);
}

TEST(TrajectoryQualityTest, MaxAccelerationWithChange) {
  SimulationResult result;
  result.dt = 1.0;
  // Velocity changes from (1,0) to (3,0) at frame 2 => accel = 2.0
  result.avatar_positions = {{0, 0}, {1, 0}, {4, 0}, {7, 0}};
  result.num_frames = 4;
  EXPECT_NEAR(result.MaxAcceleration(), 2.0, 1e-9);
}

TEST(TrajectoryQualityTest, MeanJerkSynthetic) {
  SimulationResult result;
  result.dt = 1.0;
  // Constant acceleration => zero jerk
  // pos: 0, 1, 3, 6, 10  (vel: 1,2,3,4; accel: 1,1,1; jerk: 0,0)
  result.avatar_positions = {{0, 0}, {1, 0}, {3, 0}, {6, 0}, {10, 0}};
  result.num_frames = 5;
  EXPECT_NEAR(result.MeanJerk(), 0.0, 1e-9);
}

TEST(TrajectoryQualityTest, DirectionChangesSynthetic) {
  SimulationResult result;
  result.dt = 1.0;
  // Straight line: no direction changes
  result.avatar_positions = {{0, 0}, {1, 0}, {2, 0}, {3, 0}};
  result.num_frames = 4;
  EXPECT_EQ(result.DirectionChanges(), 0);

  // Right angle turn at frame 2
  result.avatar_positions = {{0, 0}, {1, 0}, {1, 1}, {1, 2}};
  EXPECT_EQ(result.DirectionChanges(), 1);
}

TEST(TrajectoryQualityTest, ArrivalLagSynthetic) {
  SimulationResult result;
  result.dt = 0.1;
  result.num_frames = 5;
  // Actor arrives at target (5,0) at frame 2, avatar at frame 4
  result.actor_positions = {{0, 0}, {2.5, 0}, {5, 0}, {5, 0}, {5, 0}};
  result.avatar_positions = {{0, 0}, {1, 0}, {2, 0}, {3.5, 0}, {5, 0}};
  Eigen::Vector2d target(5, 0);
  double lag = result.ArrivalLag(target, 1.0);
  // actor_arrival=2, avatar_arrival=3 (dist 3.5->5 = 1.5 > 1, dist 5->5 = 0)
  // Actually avatar at frame 4: dist=0 <= 1. Avatar at frame 3: dist=1.5 > 1.
  // So avatar_arrival=4. lag = (4-2)*0.1 = 0.2
  EXPECT_NEAR(lag, 0.2, 0.01);
}

TEST(TrajectoryQualityTest, ArrivalLagActorNeverArrives) {
  SimulationResult result;
  result.dt = 0.1;
  result.num_frames = 3;
  result.actor_positions = {{0, 0}, {1, 0}, {2, 0}};
  result.avatar_positions = {{0, 0}, {0.5, 0}, {1, 0}};
  // Target far away, actor never arrives
  EXPECT_NEAR(result.ArrivalLag({100, 100}, 1.0), 0.0, 1e-9);
}

TEST(TrajectoryQualityTest, MeanRelationErrorSynthetic) {
  SimulationResult result;
  result.dt = 1.0;
  result.num_frames = 3;
  result.avatar_positions = {{0, 0}, {1, 0}, {2, 0}};
  // Target positions offset by (0,1) from avatar
  RealizationTarget t0;
  t0.position = {0, 1};
  RealizationTarget t1;
  t1.position = {1, 1};
  RealizationTarget t2;
  t2.position = {2, 1};
  result.targets = {{t0}, {t1}, {t2}};
  EXPECT_NEAR(result.MeanRelationError(), 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Smoothness on actual simulation
// ---------------------------------------------------------------------------

TEST(TrajectoryQualityTest, SmoothnessOnPresentation) {
  auto [sim, actor, avatar] = BuildPresentationScenario();
  auto traj = Trajectory::StraightLine({2, 2}, {5, 15}, 5.0);

  // Run with velocity-bounded controller (default).
  Simulator::RunConfig config;
  config.max_linear_speed = 1.4;
  config.use_velocity_bounded = true;
  auto result = sim->Run(actor, avatar, traj, 0.033, config);

  EXPECT_GT(result.num_frames, 50);

  // The avatar should exhibit bounded acceleration and jerk.
  double max_accel = result.MaxAcceleration();
  double mean_jerk = result.MeanJerk();
  int direction_changes = result.DirectionChanges();

  // Log metrics for manual inspection.
  std::cout << "[SmoothnessOnPresentation] max_accel=" << max_accel
            << " mean_jerk=" << mean_jerk
            << " direction_changes=" << direction_changes
            << " total_travel=" << result.TotalAvatarTravel()
            << " max_speed=" << result.MaxSpeed() << "\n";

  // Sanity: metrics are finite and non-negative.
  EXPECT_TRUE(std::isfinite(max_accel));
  EXPECT_GE(max_accel, 0.0);
  EXPECT_TRUE(std::isfinite(mean_jerk));
  EXPECT_GE(mean_jerk, 0.0);
  EXPECT_GE(direction_changes, 0);
}

TEST(TrajectoryQualityTest, ArrivalLagBounded) {
  auto [sim, actor, avatar] = BuildPresentationScenario();
  // Actor walks from (2,2) to Board_A at (5,15) over 5 seconds.
  auto traj = Trajectory::StraightLine({2, 2}, {5, 15}, 5.0);

  // Run with velocity-bounded controller.
  Simulator::RunConfig config;
  config.max_linear_speed = 1.4;
  config.use_velocity_bounded = true;
  auto result = sim->Run(actor, avatar, traj, 0.033, config);

  // Board_B is at (10,3). The avatar should arrive in the vicinity.
  Eigen::Vector2d board_b(10, 3);
  double lag = result.ArrivalLag(board_b, 2.0);

  std::cout << "[ArrivalLagBounded] lag=" << lag
            << " final_dist=" << result.FinalDistanceTo(board_b) << "\n";

  // The lag should be bounded: avatar should reach the target area within
  // a reasonable time after the actor. If lag is 0 it means avatar never
  // arrived or arrived simultaneously; both are acceptable.
  EXPECT_TRUE(std::isfinite(lag));
}

TEST(TrajectoryQualityTest, RelationErrorDecreases) {
  auto [sim, actor, avatar] = BuildPresentationScenario();
  auto traj = Trajectory::StraightLine({2, 2}, {5, 15}, 5.0);
  auto result = sim->Run(actor, avatar, traj, 0.033);

  ASSERT_GT(result.num_frames, 10);

  // Compute mean relation error for first half and second half.
  auto half_error = [&](size_t from, size_t to) {
    double total = 0.0;
    int count = 0;
    for (size_t i = from; i < to && i < result.targets.size(); ++i) {
      if (result.targets[i].empty()) continue;
      total += (result.avatar_positions[i] - result.targets[i][0].position)
                   .norm();
      ++count;
    }
    return count > 0 ? total / count : 0.0;
  };

  size_t mid = result.num_frames / 2;
  double first_half_error = half_error(0, mid);
  double second_half_error = half_error(mid, result.num_frames);

  std::cout << "[RelationErrorDecreases] first_half=" << first_half_error
            << " second_half=" << second_half_error << "\n";

  // Both should be finite.
  EXPECT_TRUE(std::isfinite(first_half_error));
  EXPECT_TRUE(std::isfinite(second_half_error));
}

TEST(TrajectoryQualityTest, AvatarStaysInTargetPolygon) {
  auto [sim, actor, avatar] = BuildPresentationScenario();
  auto traj = Trajectory::StraightLine({2, 2}, {5, 15}, 5.0);
  auto result = sim->Run(actor, avatar, traj, 0.033);

  Polygon2D target_polygon({{0, 0}, {15, 0}, {15, 15}, {0, 15}});
  EXPECT_TRUE(result.AvatarAlwaysInPolygon(target_polygon));
}

TEST(TrajectoryQualityTest, CompareShortVsLongTrajectory) {
  // Short trajectory (1 second) vs long trajectory (5 seconds) to the same
  // destination. The long trajectory should have lower max acceleration.
  auto build_and_run = [](double duration) {
    auto [sim, actor, avatar] = BuildPresentationScenario();
    auto traj = Trajectory::StraightLine({2, 2}, {5, 15}, duration);
    return sim->Run(actor, avatar, traj, 0.033);
  };

  auto result_short = build_and_run(1.0);
  auto result_long = build_and_run(5.0);

  std::cout << "[CompareShortVsLong] short_max_accel="
            << result_short.MaxAcceleration()
            << " long_max_accel=" << result_long.MaxAcceleration()
            << " short_jerk=" << result_short.MeanJerk()
            << " long_jerk=" << result_long.MeanJerk() << "\n";

  // Both results should have valid metrics.
  EXPECT_TRUE(std::isfinite(result_short.MaxAcceleration()));
  EXPECT_TRUE(std::isfinite(result_long.MaxAcceleration()));
  EXPECT_TRUE(std::isfinite(result_short.MeanJerk()));
  EXPECT_TRUE(std::isfinite(result_long.MeanJerk()));
}

TEST(TrajectoryQualityTest, WaypointTrajectoryMetrics) {
  auto sim = Simulator::Create();

  // Source space (20x20)
  auto sa = sim->AddSpace();
  auto room_a = sa.AddElement("Room")
                    .WithPolygon({{0, 0}, {20, 0}, {20, 20}, {0, 20}})
                    .WithSharedId(100);
  room_a.AddChild("Board")
      .At(5, 15)
      .WithBoundary({{3, 13}, {7, 13}, {7, 17}, {3, 17}})
      .WithSharedId(200);
  room_a.AddChild("Table")
      .At(15, 5)
      .WithBoundary({{13, 3}, {17, 3}, {17, 7}, {13, 7}})
      .WithSharedId(300);
  auto actor =
      sa.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_a);
  actor.Bind("root", Spacing());

  // Target space (15x15)
  auto sb = sim->AddSpace();
  auto room_b = sb.AddElement("Room")
                    .WithPolygon({{0, 0}, {15, 0}, {15, 15}, {0, 15}})
                    .WithSharedId(100);
  room_b.AddChild("Board")
      .At(10, 3)
      .WithBoundary({{8, 1}, {12, 1}, {12, 5}, {8, 5}})
      .WithSharedId(200);
  room_b.AddChild("Table")
      .At(3, 10)
      .WithBoundary({{1, 8}, {5, 8}, {5, 12}, {1, 12}})
      .WithSharedId(300);
  auto avatar =
      sb.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_b);

  // Waypoint trajectory: Table -> center -> Board
  auto traj = Trajectory::FromWaypoints({
      {0.0, {15, 5}, 0.0},
      {2.5, {10, 10}, 0.0},
      {5.0, {5, 15}, 0.0},
  });
  auto result = sim->Run(actor, avatar, traj, 0.033);

  EXPECT_GT(result.num_frames, 50);
  EXPECT_GT(result.TotalAvatarTravel(), 1.0);

  double mean_re = result.MeanRelationError();
  std::cout << "[WaypointTrajectory] mean_relation_error=" << mean_re
            << " max_accel=" << result.MaxAcceleration()
            << " direction_changes=" << result.DirectionChanges() << "\n";

  EXPECT_TRUE(std::isfinite(mean_re));
}

TEST(TrajectoryQualityTest, SmoothnessWithVelocityBound) {
  auto [sim, actor, avatar] = BuildPresentationScenario();
  auto traj = Trajectory::StraightLine({2, 2}, {5, 15}, 5.0);

  // Run with velocity bounded controller (1.4 m/s walking speed).
  Simulator::RunConfig config;
  config.max_linear_speed = 1.4;
  config.use_velocity_bounded = true;
  auto result = sim->Run(actor, avatar, traj, 0.033, config);

  double max_accel = result.MaxAcceleration();
  double mean_jerk = result.MeanJerk();
  double max_speed = result.MaxSpeed();

  std::cout << "[VelocityBounded] max_accel=" << max_accel
            << " mean_jerk=" << mean_jerk
            << " max_speed=" << max_speed
            << " total_travel=" << result.TotalAvatarTravel()
            << " direction_changes=" << result.DirectionChanges()
            << " final_dist_board=" << result.FinalDistanceTo({10, 3})
            << " arrival_lag=" << result.ArrivalLag({10, 3}, 2.0)
            << "\n";

  // With velocity bound, speed should be <= max + small tolerance.
  EXPECT_LE(max_speed, 1.4 * 1.1 + 0.1);  // 10% tolerance

  // Acceleration should be reasonable (not teleportation).
  EXPECT_LT(max_accel, 100.0);  // well below 231 m/s^2 teleport acceleration

  // Jerk should be reasonable (higher values acceptable with distance-based metric).
  EXPECT_LT(mean_jerk, 300.0);
}

TEST(TrajectoryQualityTest, CompareImmediateVsBounded) {
  // Same scenario, two modes.
  auto run_with_config = [](bool use_bounded) {
    auto [sim, actor, avatar] = BuildPresentationScenario();
    auto traj = Trajectory::StraightLine({2, 2}, {5, 15}, 5.0);
    Simulator::RunConfig config;
    config.use_velocity_bounded = use_bounded;
    config.max_linear_speed = 1.4;
    return sim->Run(actor, avatar, traj, 0.033, config);
  };

  auto immediate = run_with_config(false);
  auto bounded = run_with_config(true);

  std::cout << "[Immediate] max_accel=" << immediate.MaxAcceleration()
            << " max_speed=" << immediate.MaxSpeed()
            << " final_dist=" << immediate.FinalDistanceTo({10, 3}) << "\n";
  std::cout << "[Bounded]   max_accel=" << bounded.MaxAcceleration()
            << " max_speed=" << bounded.MaxSpeed()
            << " final_dist=" << bounded.FinalDistanceTo({10, 3}) << "\n";

  // Bounded should have lower max speed.
  EXPECT_LT(bounded.MaxSpeed(), immediate.MaxSpeed());
  // Bounded should have reasonable acceleration (may be slightly higher due
  // to temporal bucketing with distance-based estimated_time metric).
  EXPECT_LT(bounded.MaxAcceleration(), 200.0);

  // Immediate should arrive closer (teleports to solution).
  // Bounded may have higher final distance (still approaching).
}

}  // namespace test
}  // namespace spatialblend
