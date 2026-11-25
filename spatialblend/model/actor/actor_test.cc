// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/actor/actor.h"

#include <gtest/gtest.h>

namespace spatialblend {
namespace test {

TEST(ActorTest, KeepsItsId) {
  Actor actor(1);
  EXPECT_EQ(actor.GetId(), 1);
}

TEST(ActorTest, AddAndGetAnchors) {
  Actor actor(1);
  actor.AddAnchor(SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)));
  actor.AddAnchor(SubjectAnchor("chest", Eigen::Vector3d(0, 0, 1.2)));

  EXPECT_EQ(actor.GetAnchors().size(), 2);

  auto* root = actor.GetAnchor("root");
  ASSERT_NE(root, nullptr);
  EXPECT_DOUBLE_EQ(root->local_offset.z(), 0.0);

  auto* chest = actor.GetAnchor("chest");
  ASSERT_NE(chest, nullptr);
  EXPECT_DOUBLE_EQ(chest->local_offset.z(), 1.2);
}

TEST(ActorTest, GetAnchorNotFound) {
  Actor actor(1);
  EXPECT_EQ(actor.GetAnchor("nonexistent"), nullptr);
}

TEST(ActorTest, RemoveAnchor) {
  Actor actor(1);
  actor.AddAnchor(SubjectAnchor("root"));
  actor.AddAnchor(SubjectAnchor("chest"));
  actor.RemoveAnchor("root");

  EXPECT_EQ(actor.GetAnchors().size(), 1);
  EXPECT_EQ(actor.GetAnchor("root"), nullptr);
  EXPECT_NE(actor.GetAnchor("chest"), nullptr);
}

TEST(ActorTest, AnchorWorldPosition) {
  Actor actor(1);
  actor.AddAnchor(SubjectAnchor("chest", Eigen::Vector3d(0, 0, 1.2)));

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform(0, 3) = 5.0;
  transform(1, 3) = 3.0;
  actor.SetTransform(transform);

  auto* anchor = actor.GetAnchor("chest");
  Eigen::Vector3d world = anchor->GetWorldPosition(actor.GetTransform());
  EXPECT_DOUBLE_EQ(world.x(), 5.0);
  EXPECT_DOUBLE_EQ(world.y(), 3.0);
  EXPECT_DOUBLE_EQ(world.z(), 1.2);
}

TEST(ActorTest, ElementBackReference) {
  Actor actor(1);
  EXPECT_EQ(actor.GetElementId(), kInvalidElementId);
  actor.SetElementId(10);
  EXPECT_EQ(actor.GetElementId(), 10);
}

}  // namespace test
}  // namespace spatialblend
