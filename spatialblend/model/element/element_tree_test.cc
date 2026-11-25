// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/element_tree.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace spatialblend {
namespace test {

class ElementTreeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(tree_.AddElement(1).ok());
    ASSERT_TRUE(tree_.AddElement(2, 1).ok());
    ASSERT_TRUE(tree_.AddElement(3, 2).ok());
    ASSERT_TRUE(tree_.AddElement(4, 3).ok());
    ASSERT_TRUE(tree_.AddElement(5, 2).ok());
    ASSERT_TRUE(tree_.AddElement(6, 5).ok());
  }

  ElementTree tree_;
};

TEST_F(ElementTreeTest, GetElement) {
  auto* elem = tree_.GetElement(3);
  ASSERT_NE(elem, nullptr);
  EXPECT_EQ(elem->GetParentId(), 2);
}

TEST_F(ElementTreeTest, GetElementNotFound) {
  EXPECT_EQ(tree_.GetElement(999), nullptr);
}

TEST_F(ElementTreeTest, DuplicateIdFails) {
  auto status = tree_.AddElement(1);
  EXPECT_FALSE(status.ok());
}

TEST_F(ElementTreeTest, MissingParentFails) {
  auto status = tree_.AddElement(99, 888);
  EXPECT_FALSE(status.ok());
}

TEST_F(ElementTreeTest, GetRoot) {
  auto* root = tree_.GetRoot();
  ASSERT_NE(root, nullptr);
  EXPECT_EQ(root->GetId(), 1);
}

TEST_F(ElementTreeTest, Size) {
  EXPECT_EQ(tree_.Size(), 6);
}

TEST_F(ElementTreeTest, GetSubtree) {
  auto subtree = tree_.GetSubtreeIds(2);
  EXPECT_EQ(subtree.size(), 5);
}

TEST_F(ElementTreeTest, GetDirectChildren) {
  auto children = tree_.GetDirectChildIds(2);
  ASSERT_EQ(children.size(), 2);
}

TEST_F(ElementTreeTest, GetAncestors) {
  auto ancestors = tree_.GetAncestorIds(4);
  ASSERT_EQ(ancestors.size(), 3);
  EXPECT_EQ(ancestors[0], 3);
  EXPECT_EQ(ancestors[1], 2);
  EXPECT_EQ(ancestors[2], 1);
}

TEST_F(ElementTreeTest, RemoveElement) {
  auto status = tree_.RemoveElement(3);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(tree_.GetElement(3), nullptr);
  EXPECT_EQ(tree_.GetElement(4), nullptr);
  auto children = tree_.GetDirectChildIds(2);
  ASSERT_EQ(children.size(), 1);
  EXPECT_EQ(children[0], 5);
}

TEST_F(ElementTreeTest, ReparentElement) {
  auto status = tree_.ReparentElement(5, 1);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(tree_.GetElement(5)->GetParentId(), 1);
  EXPECT_EQ(tree_.GetDirectChildIds(2).size(), 1);
}

TEST_F(ElementTreeTest, ReparentCycleDetection) {
  auto status = tree_.ReparentElement(1, 4);
  EXPECT_FALSE(status.ok());
}

TEST_F(ElementTreeTest, GetWorldTransform) {
  Eigen::Matrix4d region_t = Eigen::Matrix4d::Identity();
  region_t(0, 3) = 5.0;
  tree_.GetElement(2)->SetLocalTransform(region_t);

  Eigen::Matrix4d board_t = Eigen::Matrix4d::Identity();
  board_t(1, 3) = 3.0;
  tree_.GetElement(3)->SetLocalTransform(board_t);

  Eigen::Matrix4d world = tree_.GetWorldTransform(3);
  EXPECT_DOUBLE_EQ(world(0, 3), 5.0);
  EXPECT_DOUBLE_EQ(world(1, 3), 3.0);
}

TEST_F(ElementTreeTest, GetElementsBySharedId) {
  tree_.GetElement(3)->SetSharedId(100);
  tree_.GetElement(5)->SetSharedId(200);

  auto results = tree_.GetElementsBySharedId(100);
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0]->GetId(), 3);
}

TEST_F(ElementTreeTest, GetElementsBySharedIdNotFound) {
  auto results = tree_.GetElementsBySharedId(999);
  EXPECT_TRUE(results.empty());
}

TEST_F(ElementTreeTest, GetElementBySharedIdSingular) {
  tree_.GetElement(3)->SetSharedId(100);
  auto* elem = tree_.GetElementBySharedId(100);
  ASSERT_NE(elem, nullptr);
  EXPECT_EQ(elem->GetId(), 3);
}

TEST_F(ElementTreeTest, GetElementBySharedIdNotFound) {
  auto* elem = tree_.GetElementBySharedId(999);
  EXPECT_EQ(elem, nullptr);
}

// A shared_id resolves to exactly one element per space, by the declared
// smallest-element-id rule, whatever the registration order.
TEST_F(ElementTreeTest, UniqueIdentity_H_2_DuplicateResolvesByDeclaredRule) {
  tree_.GetElement(5)->SetSharedId(100);
  tree_.GetElement(3)->SetSharedId(100);
  tree_.GetElement(4)->SetSharedId(100);
  EXPECT_EQ(tree_.GetElementsBySharedId(100).size(), 3u);
  for (int i = 0; i < 3; ++i) {
    ASSERT_NE(tree_.GetElementBySharedId(100), nullptr);
    EXPECT_EQ(tree_.GetElementBySharedId(100)->GetId(), 3);
  }
  // Clearing the winner hands the identity to the next holder by the rule.
  tree_.GetElement(3)->SetSharedId(kInvalidSharedElementId);
  EXPECT_EQ(tree_.GetElementBySharedId(100)->GetId(), 4);
}

TEST_F(ElementTreeTest, GetAllIdsCoversEveryDepthAscending) {
  auto ids = tree_.GetAllIds();
  EXPECT_EQ(ids.size(), tree_.Size());
  EXPECT_TRUE(std::is_sorted(ids.begin(), ids.end()));
}

}  // namespace test
}  // namespace spatialblend
