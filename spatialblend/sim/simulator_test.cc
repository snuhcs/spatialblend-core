// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/sim/simulator.h"

#include <cmath>

#include <gtest/gtest.h>

#include "spatialblend/space/builder/kernel_shorthands.h"
#include "spatialblend/space/builder/trajectory.h"

namespace spatialblend {
namespace test {

TEST(SimulatorTest, CreateAndAddSpaces) {
  auto sim = Simulator::Create();
  auto sa = sim->AddSpace();
  auto sb = sim->AddSpace();
  EXPECT_NE(sa.GetSpace(), sb.GetSpace());
}

TEST(SimulatorTest, TickPropagates) {
  auto sim = Simulator::Create();
  auto sa = sim->AddSpace();
  auto room_a = sa.AddElement("Room")
                    .WithPolygon({{0, 0}, {20, 0}, {20, 20}, {0, 20}})
                    .WithSharedId(100);
  room_a.AddChild("Board").At(5, 15).WithSharedId(200);
  auto actor =
      sa.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_a);
  actor.Bind("root", Spacing());

  auto sb = sim->AddSpace();
  auto room_b = sb.AddElement("Room")
                    .WithPolygon({{0, 0}, {15, 0}, {15, 15}, {0, 15}})
                    .WithSharedId(100);
  room_b.AddChild("Board").At(10, 3).WithSharedId(200);
  auto avatar =
      sb.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_b);

  actor.MoveTo({5, 5}, 0.0);
  sim->Tick(0.033);

  auto pos = avatar.GetPosition();
  EXPECT_TRUE(std::isfinite(pos.x()));
  EXPECT_TRUE(std::isfinite(pos.y()));
}

TEST(SimulatorTest, RunReturnsResult) {
  auto sim = Simulator::Create();

  auto sa = sim->AddSpace();
  auto room_a = sa.AddElement("Room")
                    .WithPolygon({{0, 0}, {20, 0}, {20, 20}, {0, 20}})
                    .WithSharedId(100);
  room_a.AddChild("Board").At(5, 15).WithSharedId(200);
  auto actor =
      sa.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_a);
  actor.Bind("root", Spacing());

  auto sb = sim->AddSpace();
  auto room_b = sb.AddElement("Room")
                    .WithPolygon({{0, 0}, {15, 0}, {15, 15}, {0, 15}})
                    .WithSharedId(100);
  room_b.AddChild("Board").At(10, 3).WithSharedId(200);
  auto avatar =
      sb.AddActor().WithAnchor("root", 0, 0, 0).InScope(room_b);

  auto traj = Trajectory::StraightLine({3, 3}, {5, 15}, 2.0);
  auto result = sim->Run(actor, avatar, traj, 0.1);

  EXPECT_GT(result.num_frames, 0);
  EXPECT_EQ(static_cast<int>(result.actor_positions.size()),
            result.num_frames);
  EXPECT_EQ(static_cast<int>(result.avatar_positions.size()),
            result.num_frames);
}

TEST(SimulatorTest, MetricsWork) {
  SimulationResult result;
  result.dt = 1.0;
  result.avatar_positions = {{0, 0}, {3, 0}, {3, 4}};
  result.num_frames = 3;

  EXPECT_NEAR(result.TotalAvatarTravel(), 7.0, 0.01);
  EXPECT_NEAR(result.FinalDistanceTo({3, 4}), 0.0, 0.01);
  EXPECT_NEAR(result.MaxSpeed(), 4.0, 0.01);
}

TEST(SimulatorTest, AvatarAlwaysInPolygon) {
  SimulationResult result;
  result.dt = 1.0;
  result.avatar_positions = {{1, 1}, {5, 5}, {9, 9}};
  result.num_frames = 3;

  Polygon2D inside_polygon({{0, 0}, {10, 0}, {10, 10}, {0, 10}});
  EXPECT_TRUE(result.AvatarAlwaysInPolygon(inside_polygon));

  Polygon2D small_polygon({{0, 0}, {2, 0}, {2, 2}, {0, 2}});
  EXPECT_FALSE(result.AvatarAlwaysInPolygon(small_polygon));
}

TEST(SimulatorTest, FinalDistanceToEmpty) {
  SimulationResult result;
  EXPECT_NEAR(result.FinalDistanceTo({1, 1}), 0.0, 0.01);
}

}  // namespace test
}  // namespace spatialblend
