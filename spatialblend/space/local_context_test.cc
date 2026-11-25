// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/local_context.h"

#include <gtest/gtest.h>

namespace spatialblend {
namespace test {

TEST(LocalContextTest, CreateAndAddSpaces) {
  auto context = LocalContext::Create();
  auto* space_a = context->AddSpace();
  auto* space_b = context->AddSpace();

  ASSERT_NE(space_a, nullptr);
  ASSERT_NE(space_b, nullptr);
  EXPECT_NE(space_a->GetSpaceId(), space_b->GetSpaceId());
  EXPECT_NE(space_a->GetSpaceId(), kInvalidSpaceId);
}

TEST(LocalContextTest, GetHub) {
  auto context = LocalContext::Create();
  EXPECT_NE(context->GetHub(), nullptr);
}

TEST(LocalContextTest, SpacesConnectedToHub) {
  auto context = LocalContext::Create();
  context->AddSpace();
  context->AddSpace();
  EXPECT_EQ(context->GetHub()->GetConnectedSpaceCount(), 2);
}

TEST(LocalContextTest, EndToEndRelationTransfer) {
  auto context = LocalContext::Create();
  auto* space_a = context->AddSpace();
  auto* space_b = context->AddSpace();

  // Space A: Room(1) -> Board(2), board at (5, 8)
  auto* tree_a = space_a->GetElementTree();
  tree_a->AddElement(1);
  tree_a->AddElement(2, 1);
  tree_a->GetElement(1)->SetSharedId(100);
  tree_a->GetElement(2)->SetSharedId(200);
  Eigen::Matrix4d board_a = Eigen::Matrix4d::Identity();
  board_a(0, 3) = 5.0;
  board_a(1, 3) = 8.0;
  tree_a->GetElement(2)->SetLocalTransform(board_a);

  // Space B: Room(10) -> Board(20), board at (2, 3)
  auto* tree_b = space_b->GetElementTree();
  tree_b->AddElement(10);
  tree_b->AddElement(20, 10);
  tree_b->GetElement(10)->SetSharedId(100);
  tree_b->GetElement(20)->SetSharedId(200);
  Eigen::Matrix4d board_b = Eigen::Matrix4d::Identity();
  board_b(0, 3) = 2.0;
  board_b(1, 3) = 3.0;
  tree_b->GetElement(20)->SetLocalTransform(board_b);

  // Create actor in Space A
  std::vector<SubjectAnchor> anchors = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))};
  auto actor_id = space_a->CreateActor(anchors);
  FollowProposals(space_a, actor_id);
  auto* actor = space_a->GetActor(actor_id);
  actor->SetElementId(1);

  // Bind spacing relation to Board via Children query
  space_a->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);

  // Connect hub for auto-propagation
  space_a->SetHub(context->GetHub());

  // Update actor state — auto-propagates via hub
  space_a->UpdateActorState(actor_id, Eigen::Vector2d(3, 6), 0.0);

  // Verify Space B received update
  auto& received = space_b->GetReceivedUpdates();
  ASSERT_GE(received.size(), 1u);
  EXPECT_EQ(received[0].actor_shared_id, 100);
  ASSERT_GE(received[0].payloads.size(), 1u);
  EXPECT_EQ(received[0].payloads[0].referent_shared_id, 200);

  // Actor at (3,6), Board at (5,8) → relative (-2, -2) in support frame
  const auto& gr = received[0].payloads[0].grounded_relation;
  EXPECT_NEAR(gr.encoded_state(0), -2.0, 1e-6);
  EXPECT_NEAR(gr.encoded_state(1), -2.0, 1e-6);
}

TEST(LocalContextTest, EndToEndSolverAdaptsAvatar) {
  auto context = LocalContext::Create();
  auto* space_a = context->AddSpace();
  auto* space_b = context->AddSpace();

  // Space A: Room(1) -> Board(2) at (5, 8)
  auto* tree_a = space_a->GetElementTree();
  tree_a->AddElement(1);
  tree_a->AddElement(2, 1);
  tree_a->GetElement(1)->SetSharedId(100);
  tree_a->GetElement(2)->SetSharedId(200);
  Eigen::Matrix4d board_a = Eigen::Matrix4d::Identity();
  board_a(0, 3) = 5.0;
  board_a(1, 3) = 8.0;
  tree_a->GetElement(2)->SetLocalTransform(board_a);

  // Space B: Room(10) -> Board(20) at (2, 3)
  auto* tree_b = space_b->GetElementTree();
  tree_b->AddElement(10);
  tree_b->AddElement(20, 10);
  tree_b->GetElement(10)->SetSharedId(100);
  tree_b->GetElement(20)->SetSharedId(200);
  Eigen::Matrix4d board_b = Eigen::Matrix4d::Identity();
  board_b(0, 3) = 2.0;
  board_b(1, 3) = 3.0;
  tree_b->GetElement(20)->SetLocalTransform(board_b);

  // Actor in Space A
  std::vector<SubjectAnchor> anchors = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))};
  auto actor_a_id = space_a->CreateActor(anchors);
  FollowProposals(space_a, actor_a_id);
  auto* actor_a = space_a->GetActor(actor_a_id);
  actor_a->SetElementId(1);

  // Avatar in Space B (same shared ID via element)
  auto actor_b_id = space_b->CreateActor(anchors);
  FollowProposals(space_b, actor_b_id);
  auto* actor_b = space_b->GetActor(actor_b_id);
  actor_b->SetElementId(10);

  // Bind spacing relation in Space A using Children query
  space_a->Bind(actor_a_id, "root", RelationKernelType::kAvatarSpacing);

  // Connect hub for auto-propagation
  space_a->SetHub(context->GetHub());

  // User moves in Space A to (3, 6) — offset (-2,-2) from Board_A(5,8) —
  // auto-propagates to Space B.
  Eigen::Vector2d src_pos(3, 6);
  space_a->UpdateActorState(actor_a_id, src_pos, 0.0);

  // The source is placed once and held. The first sample puts the receiver
  // on the contact centroid, which for this one board is Board_B itself.
  // It does not keep the source's (-2, -2) offset: that offset was the
  // removed walk to the approach point.
  Eigen::Vector2d avatar_pos = actor_b->GetPosition();
  EXPECT_NEAR(avatar_pos.x(), 2.0, 1e-6);
  EXPECT_NEAR(avatar_pos.y(), 3.0, 1e-6);
}

}  // namespace test
}  // namespace spatialblend
