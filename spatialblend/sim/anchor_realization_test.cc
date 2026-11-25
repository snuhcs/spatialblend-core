// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/sim/anchor_realization_test.cc
//
// End-to-end coverage for per-anchor realization targets that is easiest
// to express through the Simulator's cross-space replay harness (as opposed
// to the direct RelationSolver::SolveActorTargets unit tests in
// space/relation_solver_test.cc, which cover the root-bit-identical, hand-z,
// and hysteresis/reach-clamp *mechanisms* in isolation). This file covers:
//   - the structured trace ("target" JSONL lines) carrying an "anchor" field
//     per emitted RealizationTarget, per anchor.

#include "spatialblend/sim/simulator.h"

#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include "nlohmann/json.hpp"

#include "spatialblend/base/common/trace.h"
#include "spatialblend/space/builder/kernel_shorthands.h"

namespace spatialblend {
namespace {

const Simulator::RunConfig kImmediate{
    .max_linear_speed = 1.4,
    .max_angular_speed = 2.0,
    .use_velocity_bounded = false};

class AnchorRealizationTraceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TraceSink::Instance().ResetForTest();
    unsetenv("SB_V2_TRACE");
    unsetenv("SB_V2_TRACE_FILE");
  }
  void TearDown() override {
    TraceSink::Instance().ResetForTest();
    unsetenv("SB_V2_TRACE");
    unsetenv("SB_V2_TRACE_FILE");
    if (!path_.empty()) {
      std::remove(path_.c_str());
      path_.clear();
    }
  }

  std::string TempPath() {
    path_ = "/tmp/spatialblend_anchor_trace_test_" +
            std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
            "_" + std::to_string(reinterpret_cast<uintptr_t>(this)) +
            ".jsonl";
    return path_;
  }

  std::string path_;
};

TEST_F(AnchorRealizationTraceTest, TargetLinesCarryAnchorFieldPerAnchor) {
  TraceSink::Instance().SetPath(TempPath());
  ASSERT_TRUE(TraceSink::Instance().Active());

  auto sim = Simulator::Create();

  auto sa = sim->AddSpace();
  auto room_a = sa.AddElement("Room")
                    .WithPolygon({{0, 0}, {20, 0}, {20, 20}, {0, 20}})
                    .WithSharedId(100);
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
  auto table_b = room_b.AddChild("Table")
                     .WithBoundary({{7, 3}, {9, 3}, {9, 5}, {7, 5}})
                     .WithSharedId(200);
  auto avatar = sb.AddActor()
                    .WithAnchor("root", 0, 0, 0)
                    .WithAnchor("hand", 0.4, 0, 0.9)
                    .InScope(room_b);
  // Widen reach so the hand target is not clamped down near the (unmoved)
  // root before it reaches the trace sink — see the matching NOTE in
  // scenario_replay_test.cc's AffordancePoseReplay. Irrelevant to what this
  // test checks (the "anchor" field's presence), but keeps the emitted
  // "hand" target's position sane rather than incidentally clamped.
  sb.GetSpace()->SetAnchorReach("hand", 25.0);

  auto traj = Trajectory::StraightLine({5, 5}, {10, 15}, 5.0);
  auto result = sim->Run(actor, avatar, traj, 0.033, kImmediate);
  ASSERT_GT(result.num_frames, 0);

  TraceSink::Instance().Disable();

  std::ifstream in(path_);
  ASSERT_TRUE(in.good());

  bool saw_root_target = false;
  bool saw_hand_target = false;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    nlohmann::json parsed = nlohmann::json::parse(line);
    if (parsed.value("ev", "") != "target") continue;
    ASSERT_TRUE(parsed.contains("anchor"))
        << "target trace line missing \"anchor\" field: " << line;
    std::string anchor = parsed["anchor"].get<std::string>();
    if (anchor == "root") saw_root_target = true;
    if (anchor == "hand") saw_hand_target = true;
  }
  EXPECT_TRUE(saw_root_target) << "no root-anchor target line was traced";
  EXPECT_TRUE(saw_hand_target) << "no hand-anchor target line was traced";
}

}  // namespace
}  // namespace spatialblend
