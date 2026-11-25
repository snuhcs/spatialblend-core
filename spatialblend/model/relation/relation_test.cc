// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/relation.h"

#include <gtest/gtest.h>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/relation/grounding_rule.h"

namespace spatialblend {
namespace test {

class TestPositionKernel : public RelationKernel {
 public:
  GroundedRelation Encode(const Eigen::Vector3d& anchor_pos,
                          double anchor_yaw,
                          const Support& support) const override {
    Eigen::Matrix4d inv = support.world_transform.inverse();
    Eigen::Vector4d local =
        inv * Eigen::Vector4d(anchor_pos.x(), anchor_pos.y(),
                              anchor_pos.z(), 1.0);
    GroundedRelation gr;
    gr.encoded_state = Eigen::VectorXd(2);
    gr.encoded_state << local.x(), local.y();
    return gr;
  }

  RelationKernelType GetType() const override {
    return RelationKernelType::kAvatarFacing;  // arbitrary, test-only
  }

  SatisfactionMetric ComputeSatisfactionMetric(
      const GroundedRelation& current,
      const GroundedRelation* /*previous*/,
      const ActorMotionState& motion) const override {
    double remaining = current.encoded_state.norm();
    double rate = std::max(motion.linear_speed, 0.01);
    return {remaining, remaining / rate};
  }

  std::vector<StateType> GetDynamicStates() const override {
    return {StateType::kPosition2D};
  }

  std::vector<element_id_t> ResolveScope(
      const Eigen::Vector3d& /*actor_position*/, element_id_t actor_element_id,
      const ElementTree& tree,
      element_id_t /*cached_deepest_containing*/ = kInvalidElementId)
      const override {
    return tree.GetDirectChildIds(actor_element_id);
  }
};

class RelationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tree_.AddElement(1);
    tree_.AddElement(2, 1);

    Eigen::Matrix4d board_t = Eigen::Matrix4d::Identity();
    board_t(0, 3) = 5.0;
    board_t(1, 3) = 8.0;
    tree_.GetElement(2)->SetLocalTransform(board_t);
  }

  ElementTree tree_;
};

TEST_F(RelationTest, Create) {
  auto rule = std::make_unique<DefaultGroundingRule>();
  auto kernel = std::make_unique<TestPositionKernel>();

  Relation rel("test_pos", "root", 2, std::move(rule), std::move(kernel));

  EXPECT_EQ(rel.GetName(), "test_pos");
  EXPECT_EQ(rel.GetSubjectAnchorName(), "root");
  EXPECT_EQ(rel.GetReferentElementId(), 2);
}

TEST_F(RelationTest, Evaluate) {
  auto rule = std::make_unique<DefaultGroundingRule>();
  auto kernel = std::make_unique<TestPositionKernel>();

  Relation rel("test_pos", "root", 2, std::move(rule), std::move(kernel));

  auto result = rel.Evaluate(Eigen::Vector3d(3, 6, 0), 0.0, tree_);
  ASSERT_EQ(result.grounded.encoded_state.size(), 2);
  EXPECT_NEAR(result.grounded.encoded_state(0), -2.0, 1e-9);
  EXPECT_NEAR(result.grounded.encoded_state(1), -2.0, 1e-9);
}

TEST_F(RelationTest, Compare) {
  auto rule = std::make_unique<DefaultGroundingRule>();
  auto kernel = std::make_unique<TestPositionKernel>();

  Relation rel("test_pos", "root", 2, std::move(rule), std::move(kernel));

  auto source_result = rel.Evaluate(Eigen::Vector3d(3, 6, 0), 0.0, tree_);
  auto target_result = rel.Evaluate(Eigen::Vector3d(3.5, 6.5, 0), 0.0, tree_);

  double residual = rel.Compare(source_result.grounded, target_result.grounded);
  EXPECT_NEAR(residual, std::sqrt(0.5), 1e-9);
}

TEST_F(RelationTest, TryEvaluateMissingElement) {
  auto rule = std::make_unique<DefaultGroundingRule>();
  auto kernel = std::make_unique<TestPositionKernel>();

  Relation rel("test", "root", 999, std::move(rule), std::move(kernel));

  auto result = rel.TryEvaluate(Eigen::Vector3d::Zero(), 0.0, tree_);
  EXPECT_FALSE(result.ok());
}

TEST_F(RelationTest, CrossSpaceTransfer) {
  auto source_result = [&] {
    auto rule = std::make_unique<DefaultGroundingRule>();
    auto kernel = std::make_unique<TestPositionKernel>();
    Relation rel("pos", "root", 2, std::move(rule), std::move(kernel));
    return rel.Evaluate(Eigen::Vector3d(3, 7, 0), 0.0, tree_);
  }();

  ElementTree target_tree;
  target_tree.AddElement(10);
  target_tree.AddElement(20, 10);
  Eigen::Matrix4d tgt_board = Eigen::Matrix4d::Identity();
  tgt_board(0, 3) = 2.0;
  tgt_board(1, 3) = 3.0;
  target_tree.GetElement(20)->SetLocalTransform(tgt_board);

  auto rule2 = std::make_unique<DefaultGroundingRule>();
  auto kernel2 = std::make_unique<TestPositionKernel>();
  Relation target_rel("pos", "root", 20, std::move(rule2), std::move(kernel2));
  auto target_result = target_rel.Evaluate(Eigen::Vector3d(0, 2, 0), 0.0,
                                           target_tree);

  double residual = target_rel.Compare(source_result.grounded, target_result.grounded);
  EXPECT_NEAR(residual, 0.0, 1e-9);
}

TEST_F(RelationTest, EvaluateReturnsSupport) {
  auto rule = std::make_unique<DefaultGroundingRule>();
  auto kernel = std::make_unique<TestPositionKernel>();

  Relation rel("test_pos", "root", 2, std::move(rule), std::move(kernel));

  auto result = rel.Evaluate(Eigen::Vector3d(3, 6, 0), 0.0, tree_);

  EXPECT_NEAR(result.support.world_transform(0, 3), 5.0, 1e-9);
  EXPECT_NEAR(result.support.world_transform(1, 3), 8.0, 1e-9);
  EXPECT_EQ(result.support.referent_element_id, 2);
}

}  // namespace test
}  // namespace spatialblend
