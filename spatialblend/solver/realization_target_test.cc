// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/realization_target_test.cc
#include "spatialblend/solver/realization_target.h"

#include <gtest/gtest.h>

#include "spatialblend/base/common/types.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/relation/avatar_facing_kernel.h"
#include "spatialblend/model/relation/avatar_spacing_kernel.h"
#include "spatialblend/model/relation/grounding_rule.h"
#include "spatialblend/model/relation/relation.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/solver/relation_solver.h"
#include "spatialblend/model/realization/remote_update.h"

namespace spatialblend {
namespace {

// Helpers copied from relation_solver_test.cc

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

TEST(RealizationTargetTest, DefaultConstruction) {
  RealizationTarget target;
  EXPECT_EQ(target.arrival_time, 0.0);
  EXPECT_TRUE(target.anchor_name.empty());
  EXPECT_EQ(target.position, Eigen::Vector2d::Zero());
  EXPECT_EQ(target.yaw, 0.0);
  EXPECT_EQ(target.kernel_type, RelationKernelType::kAvatarFacing);
  EXPECT_EQ(target.referent_id, kInvalidSharedElementId);
  EXPECT_EQ(target.weight, 0.0);
}

TEST(RealizationTargetTest, FieldAssignment) {
  RealizationTarget target;
  target.arrival_time = 1.5;
  target.anchor_name = "anchor_a";
  target.position = Eigen::Vector2d(3.0, 4.0);
  target.yaw = 1.57;
  target.kernel_type = RelationKernelType::kAvatarSpacing;
  target.referent_id = 42;
  target.weight = 0.8;

  EXPECT_EQ(target.arrival_time, 1.5);
  EXPECT_EQ(target.anchor_name, "anchor_a");
  EXPECT_EQ(target.position, Eigen::Vector2d(3.0, 4.0));
  EXPECT_DOUBLE_EQ(target.yaw, 1.57);
  EXPECT_EQ(target.kernel_type, RelationKernelType::kAvatarSpacing);
  EXPECT_EQ(target.referent_id, 42);
  EXPECT_DOUBLE_EQ(target.weight, 0.8);
}

TEST(RealizationResultTest, DefaultConstruction) {
  RealizationResult result;
  EXPECT_TRUE(result.targets.empty());
  EXPECT_EQ(result.status, SolverStatus::kConverged);
}

TEST(RealizationResultTest, MultipleTargets) {
  RealizationResult result;

  RealizationTarget t0;
  t0.arrival_time = 0.0;
  t0.anchor_name = "start";
  t0.position = Eigen::Vector2d(0.0, 0.0);
  t0.yaw = 0.0;
  t0.weight = 1.0;

  RealizationTarget t1;
  t1.arrival_time = 2.0;
  t1.anchor_name = "mid";
  t1.position = Eigen::Vector2d(1.0, 2.0);
  t1.yaw = 0.5;
  t1.weight = 0.5;

  RealizationTarget t2;
  t2.arrival_time = 5.0;
  t2.anchor_name = "end";
  t2.position = Eigen::Vector2d(3.0, 4.0);
  t2.yaw = 1.0;
  t2.weight = 0.9;

  result.targets = {t0, t1, t2};
  result.status = SolverStatus::kConverged;

  ASSERT_EQ(result.targets.size(), 3u);
  EXPECT_EQ(result.targets[0].anchor_name, "start");
  EXPECT_EQ(result.targets[1].arrival_time, 2.0);
  EXPECT_EQ(result.targets[2].position, Eigen::Vector2d(3.0, 4.0));
  EXPECT_EQ(result.status, SolverStatus::kConverged);
}

TEST(RealizationResultTest, FallbackStatus) {
  RealizationResult result;
  result.status = SolverStatus::kFallback;
  EXPECT_EQ(result.status, SolverStatus::kFallback);
}

TEST(RealizationResultTest, FailedStatus) {
  RealizationResult result;
  result.status = SolverStatus::kFailed;
  EXPECT_EQ(result.status, SolverStatus::kFailed);
}

// SolveTargets tests

TEST(SolveTargetsTest, EmptyPayloadsProducesEmptyResult) {
  auto target_tree = MakeTree(5.0, 5.0);
  RelationSolver solver;
  RealizationResult result = solver.SolveTargets(
      Eigen::Vector2d(0, 0), 0.0,
      {}, {}, target_tree, /*step_duration=*/1.0);

  EXPECT_TRUE(result.targets.empty());
  EXPECT_EQ(result.status, SolverStatus::kConverged);
}

TEST(SolveTargetsTest, SolveTargetsProducesTargets) {
  // 1 payload with estimated_time=3.0, step_duration=1.0 → arrival_time≈3.0, 1 target
  auto source_tree = MakeTree(5.0, 5.0);
  auto payload = MakeSpacingPayload(
      Eigen::Vector3d(3, 3, 0), 0.0, source_tree, 2, 200);
  payload.estimated_time = 3.0;

  auto target_tree = MakeTree(10.0, 10.0);
  RelationSolver solver;
  RealizationResult result = solver.SolveTargets(
      Eigen::Vector2d(0, 0), 0.0,
      {payload}, {1.0}, target_tree, /*step_duration=*/1.0);

  ASSERT_EQ(result.targets.size(), 1u);
  EXPECT_NEAR(result.targets[0].arrival_time, 3.0, 1e-9);
  EXPECT_EQ(result.targets[0].anchor_name, "root");
  EXPECT_EQ(result.targets[0].kernel_type, RelationKernelType::kAvatarSpacing);
  EXPECT_EQ(result.targets[0].referent_id, 200);
}

TEST(SolveTargetsTest, GroupsSimilarArrivalTimes) {
  // 2 payloads with estimated_time 2.0 and 2.3, step_duration=1.0, bucket=0.5
  // → both land in bucket floor(2.0/0.5)=4 and floor(2.3/0.5)=4 → 1 target
  auto source_tree = MakeTree(5.0, 5.0);

  auto p1 = MakeSpacingPayload(
      Eigen::Vector3d(3, 3, 0), 0.0, source_tree, 2, 200);
  p1.estimated_time = 2.0;

  auto p2 = MakeSpacingPayload(
      Eigen::Vector3d(3, 3, 0), 0.0, source_tree, 2, 200);
  p2.estimated_time = 2.3;

  auto target_tree = MakeTree(10.0, 10.0);
  RelationSolver solver;
  RealizationResult result = solver.SolveTargets(
      Eigen::Vector2d(0, 0), 0.0,
      {p1, p2}, {1.0, 1.0}, target_tree, /*step_duration=*/1.0,
      /*navigable_def=*/nullptr, /*bucket_size=*/0.5);

  ASSERT_EQ(result.targets.size(), 1u);
  // Mean arrival_time = (2.0 + 2.3) / 2 = 2.15
  EXPECT_NEAR(result.targets[0].arrival_time, 2.15, 0.01);
  // Combined weight = 1.0 + 1.0 = 2.0
  EXPECT_NEAR(result.targets[0].weight, 2.0, 1e-9);
}

TEST(SolveTargetsTest, SplitsDifferentArrivalTimes) {
  // 2 payloads with estimated_time 1.0 and 10.0, step_duration=1.0, bucket=0.5
  // → floor(1.0/0.5)=2 vs floor(10.0/0.5)=20 → 2 separate targets
  auto source_tree = MakeTree(5.0, 5.0);

  auto p1 = MakeSpacingPayload(
      Eigen::Vector3d(3, 3, 0), 0.0, source_tree, 2, 200);
  p1.estimated_time = 1.0;

  auto p2 = MakeSpacingPayload(
      Eigen::Vector3d(3, 3, 0), 0.0, source_tree, 2, 200);
  p2.estimated_time = 10.0;

  auto target_tree = MakeTree(10.0, 10.0);
  RelationSolver solver;
  RealizationResult result = solver.SolveTargets(
      Eigen::Vector2d(0, 0), 0.0,
      {p1, p2}, {1.0, 1.0}, target_tree, /*step_duration=*/1.0,
      /*navigable_def=*/nullptr, /*bucket_size=*/0.5);

  ASSERT_EQ(result.targets.size(), 2u);
  // Targets are sorted by bucket key (std::map), so [0] = earlier arrival
  EXPECT_NEAR(result.targets[0].arrival_time, 1.0, 1e-9);
  EXPECT_NEAR(result.targets[1].arrival_time, 10.0, 1e-9);
}

}  // namespace
}  // namespace spatialblend
