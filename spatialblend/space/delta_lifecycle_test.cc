// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/delta_lifecycle_test.cc
//
// Tests for delta-based relation propagation and element lifecycle events.

#include <gtest/gtest.h>

#include <cmath>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace test {

// ============================================================================
// Helper: set up two spaces connected via hub with a shared room and object
// ============================================================================
struct TwoSpaceFixture {
  SpaceHub hub;
  Space space_a;
  Space space_b;

  // Element IDs in space_a
  static constexpr element_id_t kRoomId = 1;
  static constexpr element_id_t kObjectId = 2;
  static constexpr shared_element_id_t kRoomSharedId = 100;
  static constexpr shared_element_id_t kObjectSharedId = 200;

  spatialblend::entity_id_t actor_a = 0;
  spatialblend::entity_id_t actor_b = 0;

  void SetUp() {
    space_a.SetSpaceId(1);
    space_b.SetSpaceId(2);
    hub.Connect(&space_a);
    hub.Connect(&space_b);
    space_a.SetHub(&hub);
    space_b.SetHub(&hub);

    // Space A: room with navigable polygon + object
    auto* tree_a = space_a.GetElementTree();
    tree_a->AddElement(kRoomId);
    tree_a->GetElement(kRoomId)->SetSharedId(kRoomSharedId);
    auto room_poly = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 20}, {0, 20}});
    tree_a->GetElement(kRoomId)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(room_poly));

    tree_a->AddElement(kObjectId, kRoomId);
    tree_a->GetElement(kObjectId)->SetSharedId(kObjectSharedId);
    Eigen::Matrix4d obj_t = Eigen::Matrix4d::Identity();
    obj_t(0, 3) = 10.0;
    obj_t(1, 3) = 10.0;
    tree_a->GetElement(kObjectId)->SetLocalTransform(obj_t);
    auto obj_poly = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{8, 8}, {12, 8}, {12, 12}, {8, 12}});
    tree_a->GetElement(kObjectId)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(obj_poly));

    // Space B: mirror room + object
    auto* tree_b = space_b.GetElementTree();
    tree_b->AddElement(kRoomId);
    tree_b->GetElement(kRoomId)->SetSharedId(kRoomSharedId);
    tree_b->GetElement(kRoomId)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(room_poly));

    tree_b->AddElement(kObjectId, kRoomId);
    tree_b->GetElement(kObjectId)->SetSharedId(kObjectSharedId);
    tree_b->GetElement(kObjectId)->SetLocalTransform(obj_t);
    tree_b->GetElement(kObjectId)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(obj_poly));

    // Create actor in space_a bound to object
    actor_a = space_a.CreateActor({SubjectAnchor("root")});
    space_a.GetActor(actor_a)->SetElementId(kRoomId);
    space_a.Bind(actor_a, "root", RelationKernelType::kAvatarSpacing);

    // Create actor in space_b (receiver)
    actor_b = space_b.CreateActor({SubjectAnchor("root")});
    space_b.GetActor(actor_b)->SetElementId(kRoomId);
  }
};

// ============================================================================
// Delta propagation tests
// ============================================================================

TEST(DeltaPropagationTest, OnlyPropagatesOnChange) {
  TwoSpaceFixture f;
  f.SetUp();

  // First update: always propagates
  f.space_a.UpdateActorState(f.actor_a, {5.0, 5.0}, 0.0);
  size_t count_after_first = f.space_b.GetReceivedUpdates().size();
  EXPECT_EQ(count_after_first, 1);

  // Repeat with same position: should NOT propagate (no significant change)
  f.space_a.UpdateActorState(f.actor_a, {5.0, 5.0}, 0.0);
  f.space_a.UpdateActorState(f.actor_a, {5.0, 5.0}, 0.0);
  EXPECT_EQ(f.space_b.GetReceivedUpdates().size(), count_after_first);
}

TEST(DeltaPropagationTest, PropagatesOnSignificantChange) {
  TwoSpaceFixture f;
  f.SetUp();

  // First update
  f.space_a.UpdateActorState(f.actor_a, {5.0, 5.0}, 0.0);
  size_t count = f.space_b.GetReceivedUpdates().size();
  EXPECT_EQ(count, 1);

  // Move significantly (more than 1cm)
  f.space_a.UpdateActorState(f.actor_a, {6.0, 5.0}, 0.0);
  EXPECT_GT(f.space_b.GetReceivedUpdates().size(), count);
}

TEST(DeltaPropagationTest, HeartbeatPropagatesAfterInterval) {
  TwoSpaceFixture f;
  f.SetUp();

  // First update
  f.space_a.UpdateActorState(f.actor_a, {5.0, 5.0}, 0.0);
  size_t count_after_first = f.space_b.GetReceivedUpdates().size();
  EXPECT_EQ(count_after_first, 1);

  // Send 29 updates with same position (no change) — should not propagate
  for (int i = 0; i < 29; ++i) {
    f.space_a.UpdateActorState(f.actor_a, {5.0, 5.0}, 0.0);
  }
  // The 30th call should trigger heartbeat (counter starts at 0 after first
  // propagation, increments to 30 after 30 calls)
  size_t count_before_heartbeat = f.space_b.GetReceivedUpdates().size();
  // The heartbeat interval is 30, so after 30 unchanged calls we get one more
  EXPECT_GE(count_before_heartbeat, count_after_first);

  // One more to ensure heartbeat triggers if it hasn't yet
  f.space_a.UpdateActorState(f.actor_a, {5.0, 5.0}, 0.0);
  EXPECT_GT(f.space_b.GetReceivedUpdates().size(), count_after_first);
}

// A stationary source that jitters a few cm (below the movement deadband) must
// produce ZERO propagation — this is the fundamental fix for the SpatialBlend
// clone-chatter: with no propagation the receiver never re-solves, so the clone
// holds its pose by construction instead of oscillating on the jittered input.
TEST(DeltaPropagationTest, JitterBelowDeadbandDoesNotPropagate) {
  TwoSpaceFixture f;
  f.SetUp();

  f.space_a.UpdateActorState(f.actor_a, {5.0, 5.0}, 0.0);
  const size_t base = f.space_b.GetReceivedUpdates().size();
  EXPECT_EQ(base, 1u);

  // Sub-deadband jitter (~2 cm, < the 5 cm default) around the same point. Stay
  // under the heartbeat interval so the only thing that could propagate is the
  // jitter itself.
  const double j = 0.02;
  for (int i = 0; i < 20; ++i) {
    Eigen::Vector2d p(5.0 + ((i % 2) ? j : -j), 5.0 + ((i % 3) ? j : -j));
    f.space_a.UpdateActorState(f.actor_a, p, 0.0);
  }
  EXPECT_EQ(f.space_b.GetReceivedUpdates().size(), base);
}

// The gate compares against the last *propagated* pose (not the last frame), so
// a series of sub-deadband steps that accumulate past the deadband must
// eventually propagate — slow drift is real movement, jitter-in-place is not.
TEST(DeltaPropagationTest, AccumulatedDriftCrossesDeadband) {
  TwoSpaceFixture f;
  f.SetUp();

  f.space_a.UpdateActorState(f.actor_a, {5.0, 5.0}, 0.0);
  const size_t base = f.space_b.GetReceivedUpdates().size();

  f.space_a.UpdateActorState(f.actor_a, {5.02, 5.0}, 0.0);  // 2 cm: below
  f.space_a.UpdateActorState(f.actor_a, {5.04, 5.0}, 0.0);  // 4 cm cumulative
  EXPECT_EQ(f.space_b.GetReceivedUpdates().size(), base);

  f.space_a.UpdateActorState(f.actor_a, {5.06, 5.0}, 0.0);  // 6 cm >= 5 cm
  EXPECT_GT(f.space_b.GetReceivedUpdates().size(), base);
}

// ============================================================================
// Element lifecycle tests
// ============================================================================

TEST(ElementLifecycleTest, ShareElementPropagates) {
  SpaceHub hub;
  Space space_a;
  Space space_b;

  space_a.SetSpaceId(1);
  space_b.SetSpaceId(2);
  hub.Connect(&space_a);
  hub.Connect(&space_b);
  space_a.SetHub(&hub);
  space_b.SetHub(&hub);

  // Set up room in both spaces
  auto* tree_a = space_a.GetElementTree();
  tree_a->AddElement(1);
  tree_a->GetElement(1)->SetSharedId(100);

  auto* tree_b = space_b.GetElementTree();
  tree_b->AddElement(1);
  tree_b->GetElement(1)->SetSharedId(100);

  // Add object to space_a
  tree_a->AddElement(2, 1);
  tree_a->GetElement(2)->SetSharedId(200);
  tree_a->GetElement(2)->SetTag("chair");

  // Share element
  space_a.ShareElement(2);

  // Space B should now have an element with shared_id 200
  const Element* shared_elem = tree_b->GetElementBySharedId(200);
  ASSERT_NE(shared_elem, nullptr);
  EXPECT_EQ(shared_elem->GetSharedId(), 200);
  EXPECT_EQ(shared_elem->GetTag(), "chair");
}

TEST(ElementLifecycleTest, UnshareElementPropagates) {
  SpaceHub hub;
  Space space_a;
  Space space_b;

  space_a.SetSpaceId(1);
  space_b.SetSpaceId(2);
  hub.Connect(&space_a);
  hub.Connect(&space_b);
  space_a.SetHub(&hub);
  space_b.SetHub(&hub);

  // Set up room in both spaces
  auto* tree_a = space_a.GetElementTree();
  tree_a->AddElement(1);
  tree_a->GetElement(1)->SetSharedId(100);

  auto* tree_b = space_b.GetElementTree();
  tree_b->AddElement(1);
  tree_b->GetElement(1)->SetSharedId(100);

  // Add and share element
  tree_a->AddElement(2, 1);
  tree_a->GetElement(2)->SetSharedId(200);
  tree_a->GetElement(2)->SetTag("chair");
  space_a.ShareElement(2);

  // Verify it exists in space_b
  ASSERT_NE(tree_b->GetElementBySharedId(200), nullptr);

  // Unshare
  space_a.UnshareElement(2);

  // Should be gone from space_b
  EXPECT_EQ(tree_b->GetElementBySharedId(200), nullptr);
}

TEST(ElementLifecycleTest, ShareElementWithCorrectParent) {
  SpaceHub hub;
  Space space_a;
  Space space_b;

  space_a.SetSpaceId(1);
  space_b.SetSpaceId(2);
  hub.Connect(&space_a);
  hub.Connect(&space_b);
  space_a.SetHub(&hub);
  space_b.SetHub(&hub);

  // Set up room in both spaces
  auto* tree_a = space_a.GetElementTree();
  tree_a->AddElement(1);
  tree_a->GetElement(1)->SetSharedId(100);

  auto* tree_b = space_b.GetElementTree();
  tree_b->AddElement(1);
  tree_b->GetElement(1)->SetSharedId(100);

  // Add object under room in space_a and share
  tree_a->AddElement(2, 1);
  tree_a->GetElement(2)->SetSharedId(200);
  space_a.ShareElement(2);

  // In space_b, the new element should be a child of the room
  const Element* shared_elem = tree_b->GetElementBySharedId(200);
  ASSERT_NE(shared_elem, nullptr);
  // Verify its parent is the room (id=1 in space_b, shared_id=100)
  element_id_t parent_id = shared_elem->GetParentId();
  EXPECT_NE(parent_id, kInvalidElementId);
  const Element* parent = tree_b->GetElement(parent_id);
  ASSERT_NE(parent, nullptr);
  EXPECT_EQ(parent->GetSharedId(), 100);
}

}  // namespace test
}  // namespace spatialblend
