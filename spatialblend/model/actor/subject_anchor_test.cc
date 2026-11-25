// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/actor/subject_anchor.h"

#include <gtest/gtest.h>

namespace spatialblend {
namespace test {

TEST(SubjectAnchorTest, DefaultConstruction) {
  SubjectAnchor anchor;
  EXPECT_TRUE(anchor.name.empty());
  EXPECT_EQ(anchor.local_offset, Eigen::Vector3d::Zero());
}

TEST(SubjectAnchorTest, ConstructWithNameAndOffset) {
  SubjectAnchor anchor("chest", Eigen::Vector3d(0, 0, 1.2));
  EXPECT_EQ(anchor.name, "chest");
  EXPECT_DOUBLE_EQ(anchor.local_offset.z(), 1.2);
}

TEST(SubjectAnchorTest, WorldPositionIdentityTransform) {
  SubjectAnchor anchor("chest", Eigen::Vector3d(0, 0, 1.2));
  Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
  auto world = anchor.GetWorldPosition(identity);
  EXPECT_DOUBLE_EQ(world.x(), 0.0);
  EXPECT_DOUBLE_EQ(world.y(), 0.0);
  EXPECT_DOUBLE_EQ(world.z(), 1.2);
}

TEST(SubjectAnchorTest, WorldPositionTranslated) {
  SubjectAnchor anchor("chest", Eigen::Vector3d(0, 0, 1.2));
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = 5.0;
  t(1, 3) = 3.0;
  auto world = anchor.GetWorldPosition(t);
  EXPECT_DOUBLE_EQ(world.x(), 5.0);
  EXPECT_DOUBLE_EQ(world.y(), 3.0);
  EXPECT_DOUBLE_EQ(world.z(), 1.2);
}

TEST(SubjectAnchorTest, WorldPosition2D) {
  SubjectAnchor anchor("root", Eigen::Vector3d(0, 0, 0));
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = 7.0;
  t(1, 3) = 2.0;
  auto pos2d = anchor.GetWorldPosition2D(t);
  EXPECT_DOUBLE_EQ(pos2d.x(), 7.0);
  EXPECT_DOUBLE_EQ(pos2d.y(), 2.0);
}

TEST(SubjectAnchorTest, DefaultDOFIs2D) {
  SubjectAnchor anchor("root");
  EXPECT_FALSE(anchor.HasPosition3D());
  EXPECT_FALSE(anchor.HasRotation3D());
}

TEST(SubjectAnchorTest, ExplicitDOF3D) {
  SubjectAnchor anchor("hand_right", Eigen::Vector3d(0.3, 0, 1.0),
                        {StateType::kPosition3D, StateType::kRotation3D});
  EXPECT_TRUE(anchor.HasPosition3D());
  EXPECT_TRUE(anchor.HasRotation3D());
}

TEST(SubjectAnchorTest, MixedDOF) {
  SubjectAnchor anchor("hips", Eigen::Vector3d(0, 0, 0.9),
                        {StateType::kPosition3D, StateType::kRotationYaw});
  EXPECT_TRUE(anchor.HasPosition3D());
  EXPECT_FALSE(anchor.HasRotation3D());
}

TEST(SubjectAnchorTest, ParentDefaultsEmptyAndIsSettable) {
  SubjectAnchor plain("root");
  EXPECT_TRUE(plain.parent.empty());

  SubjectAnchor hand("hand_l", Eigen::Vector3d(-0.2, 0.0, 1.1),
                     {StateType::kPosition3D, StateType::kRotation3D}, "chest");
  EXPECT_EQ(hand.parent, "chest");
}

TEST(SubjectAnchorTest, StandardAnchorsCanonicalSet) {
  auto anchors = StandardAnchors();
  ASSERT_EQ(anchors.size(), 6u);
  auto find = [&](const std::string& n) -> const SubjectAnchor* {
    for (const auto& a : anchors)
      if (a.name == n) return &a;
    return nullptr;
  };
  ASSERT_NE(find("root"), nullptr);
  EXPECT_TRUE(find("root")->parent.empty());
  EXPECT_FALSE(find("root")->HasPosition3D());
  ASSERT_NE(find("hips"), nullptr);
  EXPECT_EQ(find("hips")->parent, "root");
  ASSERT_NE(find("chest"), nullptr);
  EXPECT_EQ(find("chest")->parent, "hips");
  for (const char* n : {"head", "hand_l", "hand_r"}) {
    ASSERT_NE(find(n), nullptr) << n;
    EXPECT_EQ(find(n)->parent, "chest");
    EXPECT_TRUE(find(n)->HasPosition3D());
    EXPECT_TRUE(find(n)->HasRotation3D());
  }
}

}  // namespace test
}  // namespace spatialblend
