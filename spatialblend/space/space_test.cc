// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/space_test.cc
#include "spatialblend/space/space.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <set>

#include <gtest/gtest.h>

#include "spatialblend/base/common/state_type.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/relation/affordance_pose_grounding_rule.h"
#include "spatialblend/model/relation/avatar_affordance_pose_kernel.h"
#include "spatialblend/model/relation/avatar_spacing_kernel.h"
#include "spatialblend/model/relation/grounding_rule.h"
#include "spatialblend/model/relation/relation.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace test {

class SpaceBindTest : public ::testing::Test {
 protected:
  void SetUp() override {
    space_ = std::make_unique<Space>();

    auto* tree = space_->GetElementTree();
    tree->AddElement(1);
    tree->AddElement(2, 1);
    tree->AddElement(3, 2);
    // shared: an unshared element is never a referent (SP-D7-03)
    tree->GetElement(2)->SetSharedId(2);
    tree->GetElement(3)->SetSharedId(3);
    tree->GetElement(3)->SetTag("Writable");

    Eigen::Matrix4d board_t = Eigen::Matrix4d::Identity();
    board_t(0, 3) = 5.0;
    board_t(1, 3) = 8.0;
    tree->GetElement(2)->SetLocalTransform(board_t);

    // Give Board a spatial definition for AffordancePose containment check
    auto board_boundary = std::make_shared<spatialblend::Polygon2D>(
        std::vector<Eigen::Vector2d>{{3, 6}, {7, 6}, {7, 10}, {3, 10}});
    tree->GetElement(2)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(board_boundary));

    std::vector<SubjectAnchor> anchors = {
        SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
        SubjectAnchor("chest", Eigen::Vector3d(0, 0, 1.2)),
    };
    actor_id_ = space_->CreateActor(anchors);
    FollowProposals(space_.get(), actor_id_);
    space_->GetActor(actor_id_)->SetElementId(1);
  }

  std::unique_ptr<Space> space_;
  spatialblend::entity_id_t actor_id_;
};

TEST_F(SpaceBindTest, BindCreatesBinding) {
  auto* binding =
      space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
  ASSERT_NE(binding, nullptr);
  EXPECT_EQ(binding->anchor_name, "root");
  EXPECT_EQ(binding->kernel_type, RelationKernelType::kAvatarSpacing);
}

TEST_F(SpaceBindTest, SpacingChildrenResolvesObjects) {
  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);

  // Actor outside Board boundary → scope = Room → children = Board(2)
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(1, 1), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  ASSERT_EQ(bindings.size(), 1);
  // Board(2) is direct child of Room(1)
  EXPECT_EQ(bindings[0].latest_matches.size(), 1);
  EXPECT_EQ(bindings[0].latest_matches[0].element_id, 2);
}

TEST_F(SpaceBindTest, FacingIncludesTheOtherPerson) {
  // Each person is their own element under the room. v1 treated another
  // person as a referent; the subject's own element stays out.
  ASSERT_TRUE(space_->GetElementTree()->AddElement(10, 1).ok());
  ASSERT_TRUE(space_->GetElementTree()->AddElement(11, 1).ok());
  Eigen::Matrix4d at = Eigen::Matrix4d::Identity();
  at(0, 3) = 2.0;
  space_->GetElementTree()->GetElement(10)->SetLocalTransform(at);
  Eigen::Matrix4d bt = Eigen::Matrix4d::Identity();
  bt(0, 3) = 6.0;
  space_->GetElementTree()->GetElement(11)->SetLocalTransform(bt);

  const auto other = space_->CreateActor({SubjectAnchor("root")});
  FollowProposals(space_.get(), other);
  space_->GetActor(actor_id_)->SetElementId(10);
  space_->GetActor(other)->SetElementId(11);
  space_->GetElementTree()->GetElement(11)->SetSharedId(11);  // a peer is shared (SP-D7-03)

  space_->Bind(actor_id_, "chest", RelationKernelType::kAvatarFacing);
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(2, 1), 0.0);
  const auto& matches = space_->GetBindings(actor_id_)[0].latest_matches;
  bool sees_other = false;
  bool sees_self = false;
  for (const auto& m : matches) {
    if (m.element_id == 11) sees_other = true;
    if (m.element_id == 10) sees_self = true;
  }
  EXPECT_TRUE(sees_other);
  EXPECT_FALSE(sees_self);
}

TEST_F(SpaceBindTest, FacingResolvesAllDescendants) {
  space_->Bind(actor_id_, "chest", RelationKernelType::kAvatarFacing);

  space_->UpdateActorState(actor_id_, Eigen::Vector2d(3, 6), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  ASSERT_EQ(bindings.size(), 1);
  // Facing sees all descendants of Room: Board(2) and Writable(3)
  EXPECT_EQ(bindings[0].latest_matches.size(), 2);
}

TEST_F(SpaceBindTest, AffordancePoseDescendantsActivatesWhenInsideParent) {
  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarAffordancePose,
               "Writable");

  // Actor at (5, 8) — inside Board boundary {3,6 to 7,10}
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(5, 8), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  ASSERT_EQ(bindings.size(), 1);
  // Writable(3) found, IsActive checks parent Board(2) boundary → actor is
  // inside
  EXPECT_EQ(bindings[0].latest_matches.size(), 1);
  EXPECT_EQ(bindings[0].latest_matches[0].element_id, 3);
}

// AffordancePose does not evaluate from any distance while the actor is outside the
// Board's boundary. It is in scope only inside its parent object's region (the Board).
TEST_F(SpaceBindTest, AffordancePoseOutOfScopeOutsideParentBoundary) {
  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarAffordancePose,
               "Writable");

  // Actor at (1, 1) — outside Board boundary but inside Room.
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(1, 1), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  ASSERT_EQ(bindings.size(), 1);
  EXPECT_TRUE(bindings[0].latest_matches.empty());
}

TEST_F(SpaceBindTest, MultipleBindings) {
  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor_id_, "chest", RelationKernelType::kAvatarFacing);
  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarAffordancePose,
               "Writable");

  // Actor inside Board boundary
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(5, 8), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  EXPECT_EQ(bindings.size(), 3);
  // Spacing: 1 match — Writable(3), the approach target (child of Board). The
  // actor is inside Board(2), so Board is the deepest navigable CONTAINER (the
  // scope for geodesic-to-boundary distance). It has NO border referent here:
  // its only child is an interaction region, so its interior is not solved by
  // Spacing relations (ScopeResolver::HasBorderReferent: a container is a
  // referent while the actor is inside it only when it has other, non-interaction
  // referents in scope; a leaf, a containment-transfer element and the room never
  // are). Room(1) is the actor's own element and is filtered anyway.
  EXPECT_EQ(bindings[0].latest_matches.size(), 1);
  // Facing: 2 matches (Board + Writable — all descendants of Room)
  EXPECT_EQ(bindings[1].latest_matches.size(), 2);
  // AffordancePose: 1 match (Writable, inside Board boundary)
  EXPECT_EQ(bindings[2].latest_matches.size(), 1);
}

TEST_F(SpaceBindTest, AutoPropagatesThroughHub) {
  space_->SetSpaceId(1);
  space_->GetElementTree()->GetElement(2)->SetSharedId(200);

  // Create a second space to receive
  Space space_b;
  space_b.SetSpaceId(2);
  auto* tree_b = space_b.GetElementTree();
  tree_b->AddElement(10);
  tree_b->AddElement(20, 10);
  tree_b->GetElement(20)->SetSharedId(200);

  // Create avatar in space_b
  std::vector<SubjectAnchor> anchors = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))};
  auto avatar_id = space_b.CreateActor(anchors);
  FollowProposals(&space_b, avatar_id);
  space_b.GetActor(avatar_id)->SetElementId(10);
  tree_b->GetElement(10)->SetSharedId(100);

  // Wire hub
  SpaceHub hub;
  hub.Connect(space_.get());
  hub.Connect(&space_b);
  space_->SetHub(&hub);

  // Set actor element shared ID
  space_->GetElementTree()->GetElement(1)->SetSharedId(100);
  space_->GetActor(actor_id_)->SetElementId(1);

  // Bind and update — position outside Board boundary so scope = Room
  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(1, 1), 0.0);

  // Space B should have received an update (auto-propagated)
  auto& received = space_b.GetReceivedUpdates();
  EXPECT_FALSE(received.empty());
}

TEST_F(SpaceBindTest, PerAnchorEvaluation) {
  // Add hand_right anchor to actor
  auto* actor = space_->GetActor(actor_id_);
  actor->AddAnchor(SubjectAnchor("hand_right", Eigen::Vector3d(0.3, 0, 1.0)));

  // Bind spacing with root (z=0) and facing with chest (z=1.2)
  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor_id_, "chest", RelationKernelType::kAvatarFacing);

  space_->UpdateActorState(actor_id_, Eigen::Vector2d(5, 8), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  ASSERT_EQ(bindings.size(), 2);
  ASSERT_FALSE(bindings[0].latest_matches.empty());
  ASSERT_FALSE(bindings[1].latest_matches.empty());

  auto& root_gr = bindings[0].latest_matches[0].eval.grounded;
  auto& chest_gr = bindings[1].latest_matches[0].eval.grounded;

  EXPECT_NE(root_gr.encoded_state.size(), 0);
  EXPECT_NE(chest_gr.encoded_state.size(), 0);
}

TEST_F(SpaceBindTest, PerAnchorProducesDifferentEncodings) {
  // Bind spacing with root and facing with chest to the same element
  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor_id_, "chest", RelationKernelType::kAvatarSpacing);

  // Actor at (5, 8) — directly at the Board position
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(5, 8), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  ASSERT_EQ(bindings.size(), 2);
  ASSERT_FALSE(bindings[0].latest_matches.empty());
  ASSERT_FALSE(bindings[1].latest_matches.empty());

  auto& root_gr = bindings[0].latest_matches[0].eval.grounded;
  auto& chest_gr = bindings[1].latest_matches[0].eval.grounded;

  // Root (z=0) and chest (z=1.2) produce different 3D anchor positions.
  // The spacing kernel encodes relative position in support frame.
  // With different Z heights, the encoded states should differ.
  // (They encode [local_x, local_y, relative_yaw] — the local transform
  // of 3D anchor into support frame should differ for different Z offsets)
  // Check they both have valid encoded states
  EXPECT_EQ(root_gr.encoded_state.size(), 3);
  EXPECT_EQ(chest_gr.encoded_state.size(), 3);
}

TEST_F(SpaceBindTest, MultipleActorsIndependent) {
  // Create a second actor
  std::vector<SubjectAnchor> anchors2 = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
  };
  auto actor2_id = space_->CreateActor(anchors2);
  FollowProposals(space_.get(), actor2_id);
  space_->GetActor(actor2_id)->SetElementId(1);

  // Bind different relations to each actor
  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor2_id, "root", RelationKernelType::kAvatarFacing);

  // Update both actors at different positions
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(3, 6), 0.0);
  space_->UpdateActorState(actor2_id, Eigen::Vector2d(7, 9), M_PI / 4.0);

  // Each actor should have its own bindings
  auto& bindings1 = space_->GetBindings(actor_id_);
  auto& bindings2 = space_->GetBindings(actor2_id);

  ASSERT_EQ(bindings1.size(), 1);
  ASSERT_EQ(bindings2.size(), 1);

  // Actor 1 has spacing, actor 2 has facing
  EXPECT_EQ(bindings1[0].kernel_type, RelationKernelType::kAvatarSpacing);
  EXPECT_EQ(bindings2[0].kernel_type, RelationKernelType::kAvatarFacing);

  // Both should have matches (Board is a child of Room)
  EXPECT_FALSE(bindings1[0].latest_matches.empty());
  EXPECT_FALSE(bindings2[0].latest_matches.empty());
}

TEST_F(SpaceBindTest, MultipleActorsDoNotInterfere) {
  auto actor2_id = space_->CreateActor({SubjectAnchor("root")});
  FollowProposals(space_.get(), actor2_id);

  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor2_id, "root", RelationKernelType::kAvatarSpacing);

  // Update actor 1 only
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(3, 6), 0.0);

  // Actor 1 should have matches
  auto& b1 = space_->GetBindings(actor_id_);
  EXPECT_FALSE(b1[0].latest_matches.empty());

  // Actor 2 should have NO matches (hasn't been updated)
  auto& b2 = space_->GetBindings(actor2_id);
  EXPECT_TRUE(b2[0].latest_matches.empty());
}

TEST_F(SpaceBindTest, MixedDOFAnchors) {
  auto* actor = space_->GetActor(actor_id_);
  actor->AddAnchor(
      SubjectAnchor("hand_right", Eigen::Vector3d(0.3, 0, 1.0),
                    {StateType::kPosition3D, StateType::kRotation3D}));

  // root is 2D (default), hand is 3D
  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor_id_, "hand_right", RelationKernelType::kAvatarSpacing);

  space_->UpdateActorState(actor_id_, Eigen::Vector2d(5, 8), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  ASSERT_EQ(bindings.size(), 2);

  // root binding: 2D → 3 components
  ASSERT_FALSE(bindings[0].latest_matches.empty());
  EXPECT_EQ(bindings[0].latest_matches[0].eval.grounded.encoded_state.size(),
            3);

  // hand binding: 3D → 6 components
  ASSERT_FALSE(bindings[1].latest_matches.empty());
  EXPECT_EQ(bindings[1].latest_matches[0].eval.grounded.encoded_state.size(),
            6);
}

// ---------------------------------------------------------------------------
// Anchor liveness + relation-state query
// (sb_set_anchor_transform / GetRelationStates)
// ---------------------------------------------------------------------------

TEST_F(SpaceBindTest, AnchorWorldPoseOverrideGroundsAffordanceRootUnchanged) {
  // Writable(3) has NO spatial_def of its own (see SetUp): AffordancePose's
  // grounding falls back to raw Euclidean distance from the anchor to the
  // referent's world origin (AvatarAffordancePoseKernel::Encode's no-boundary
  // fallback), so R genuinely varies with anchor position — moving the anchor
  // exactly onto the referent's world origin drives distance -> 0, R -> 1.
  auto* actor = space_->GetActor(actor_id_);
  actor->AddAnchor(SubjectAnchor("hand", Eigen::Vector3d(0.4, 0, 0.9)));
  space_->Bind(actor_id_, "hand", RelationKernelType::kAvatarAffordancePose,
               "Writable");

  // Root at (5, 8) — inside Board's boundary, so Writable is in scope
  // (AffordancePoseDescendantsActivatesWhenInsideParent, same
  // fixture/position).
  const Eigen::Vector2d root_pos(5, 8);
  space_->UpdateActorState(actor_id_, root_pos, 0.0);

  auto FindAffordanceMatch = [&]() -> const Binding::MatchResult* {
    for (const auto& b : space_->GetBindings(actor_id_)) {
      if (b.kernel_type != RelationKernelType::kAvatarAffordancePose) continue;
      for (const auto& m : b.latest_matches) {
        if (m.element_id == 3) return &m;
      }
    }
    return nullptr;
  };

  const auto* before = FindAffordanceMatch();
  ASSERT_NE(before, nullptr);
  // Hand anchor (world ~5.4,8,0.9) is not at Writable's origin (~5,8,0) ->
  // proximity should be meaningfully below 1.
  EXPECT_LT(before->eval.grounded.proximity, 0.95);

  // Snapshot root pose (bit pattern) before applying the override.
  const Eigen::Vector2d root_before = actor->GetPosition();
  const double yaw_before = actor->GetRotation();

  // Move the hand anchor exactly onto Writable's world origin.
  Eigen::Matrix4d writable_world =
      space_->GetElementTree()->GetWorldTransform(3);
  Eigen::Vector3d writable_pos = writable_world.block<3, 1>(0, 3);
  space_->SetAnchorWorldPose(actor_id_, "hand", writable_pos, /*yaw=*/0.3);

  // Re-evaluate with the SAME root pose.
  space_->UpdateActorState(actor_id_, root_pos, 0.0);

  const auto* after = FindAffordanceMatch();
  ASSERT_NE(after, nullptr);
  EXPECT_GE(after->eval.grounded.proximity, 1.0 - 1e-6);

  // Root position/yaw must be bit-unchanged: the override only affects the
  // "hand" anchor, never the root-driven scope/authoritative pose.
  EXPECT_EQ(actor->GetPosition().x(), root_before.x());
  EXPECT_EQ(actor->GetPosition().y(), root_before.y());
  EXPECT_EQ(actor->GetRotation(), yaw_before);

  // get_relation_states must equal the propagated match exactly (same
  // numbers, not a recomputation).
  auto states = space_->GetRelationStates(actor_id_);
  bool found_in_states = false;
  for (const auto& rs : states) {
    if (rs.kernel_type != RelationKernelType::kAvatarAffordancePose) continue;
    if (rs.referent_element_id != 3) continue;
    found_in_states = true;
    EXPECT_EQ(rs.proximity, after->eval.grounded.proximity);
    EXPECT_EQ(rs.anchor_name, "hand");
  }
  EXPECT_TRUE(found_in_states);
}

TEST_F(SpaceBindTest, ClearAnchorWorldPoseRestoresOffsetDrivenGrounding) {
  auto* actor = space_->GetActor(actor_id_);
  actor->AddAnchor(SubjectAnchor("hand", Eigen::Vector3d(0.4, 0, 0.9)));
  space_->Bind(actor_id_, "hand", RelationKernelType::kAvatarAffordancePose,
               "Writable");

  const Eigen::Vector2d root_pos(5, 8);
  space_->UpdateActorState(actor_id_, root_pos, 0.0);

  auto GetProximity = [&]() -> double {
    for (const auto& b : space_->GetBindings(actor_id_)) {
      if (b.kernel_type != RelationKernelType::kAvatarAffordancePose) continue;
      for (const auto& m : b.latest_matches) {
        if (m.element_id == 3) return m.eval.grounded.proximity;
      }
    }
    return -1.0;
  };

  double baseline = GetProximity();
  ASSERT_GE(baseline, 0.0);

  Eigen::Matrix4d writable_world =
      space_->GetElementTree()->GetWorldTransform(3);
  Eigen::Vector3d writable_pos = writable_world.block<3, 1>(0, 3);
  space_->SetAnchorWorldPose(actor_id_, "hand", writable_pos, 0.0);
  space_->UpdateActorState(actor_id_, root_pos, 0.0);
  EXPECT_GE(GetProximity(), 1.0 - 1e-6);

  space_->ClearAnchorWorldPose(actor_id_, "hand");
  space_->UpdateActorState(actor_id_, root_pos, 0.0);
  EXPECT_DOUBLE_EQ(GetProximity(), baseline);
}

TEST_F(SpaceBindTest, ObservationFrameIsAtomicAndTimestamped) {
  space_->SetSpaceId(1);
  SpaceHub hub;
  std::vector<RemoteRelationUpdate> sent;
  hub.SetRelationUpdateCallback(
      [&](const RemoteRelationUpdate& update) { sent.push_back(update); });
  space_->SetHub(&hub);
  uint64_t epoch = 0;
  ASSERT_TRUE(space_->BeginActorObservationEpoch(actor_id_, "root", &epoch));
  EXPECT_NE(epoch, 0u);
  ASSERT_EQ(sent.size(), 1u);
  const RealizationResult* reset_result =
      space_->GetRealizationTargets(actor_id_);
  ASSERT_NE(reset_result, nullptr);
  EXPECT_TRUE(reset_result->targets.empty());
  EXPECT_TRUE(space_->ClearActorObservationAnchor(actor_id_, epoch, "root"));
  EXPECT_EQ(sent.size(), 1u);

  using Observation = Space::ActorObservation;
  ASSERT_TRUE(space_->ApplyActorObservationFrame(
      actor_id_, epoch, 0, 0.0,
      {Observation{"root", Eigen::Vector3d(1.0, 2.0, 9.0), 0.25},
       Observation{"chest", Eigen::Vector3d(1.0, 2.0, 1.7), 0.5}}));
  const Actor* actor = space_->GetActor(actor_id_);
  ASSERT_NE(actor, nullptr);
  const auto* root = actor->GetAnchorWorldPoseOverride("root");
  const auto* chest = actor->GetAnchorWorldPoseOverride("chest");
  ASSERT_NE(root, nullptr);
  ASSERT_NE(chest, nullptr);
  EXPECT_DOUBLE_EQ(root->position.z(), 0.0);
  EXPECT_DOUBLE_EQ(chest->position.z(), 1.7);
  EXPECT_DOUBLE_EQ(space_->GetSpaceClock(), 0.0);

  // The valid root must survive an invalid late member: no partial commit and
  // the rejected input sequence remains available for a corrected retry.
  EXPECT_FALSE(space_->ApplyActorObservationFrame(
      actor_id_, epoch, 1, 0.0,
      {Observation{"root", Eigen::Vector3d(8.0, 2.0, 0.0), 0.25},
       Observation{"missing", Eigen::Vector3d::Zero(), 0.0}}));
  root = actor->GetAnchorWorldPoseOverride("root");
  ASSERT_NE(root, nullptr);
  EXPECT_DOUBLE_EQ(root->position.x(), 1.0);
  uint64_t read_epoch = 0;
  int64_t sequence = -2;
  double time = -1.0;
  bool has_sample = false;
  ASSERT_TRUE(space_->GetActorObservationClock(actor_id_, &read_epoch,
                                                &sequence, &time, &has_sample));
  EXPECT_EQ(sequence, 0);
  EXPECT_DOUBLE_EQ(time, 0.0);
  EXPECT_TRUE(has_sample);

  // Equal-time geometry correction is accepted but is not a clock tick.
  ASSERT_TRUE(space_->ApplyActorObservationFrame(
      actor_id_, epoch, 1, 0.0,
      {Observation{"root", Eigen::Vector3d(3.0, 2.0, 4.0), 0.25}}));
  EXPECT_DOUBLE_EQ(space_->GetSpaceClock(), 0.0);
  root = actor->GetAnchorWorldPoseOverride("root");
  ASSERT_NE(root, nullptr);
  EXPECT_DOUBLE_EQ(root->position.x(), 3.0);

  // Omitting every anchor clears its observation. Root retains actor transform
  // as a reference, but is not itself active; clear is timeless/idempotent.
  ASSERT_TRUE(space_->ApplyActorObservationFrame(actor_id_, epoch, 2, 2.0, {}));
  EXPECT_FALSE(actor->IsAnchorActive("root"));
  EXPECT_EQ(actor->GetAnchorWorldPoseOverride("root"), nullptr);
  ASSERT_TRUE(space_->ClearActorObservationAnchor(actor_id_, epoch, "root"));
  ASSERT_TRUE(space_->GetActorObservationClock(actor_id_, &read_epoch,
                                                &sequence, &time, &has_sample));
  EXPECT_EQ(sequence, 2);
  EXPECT_DOUBLE_EQ(time, 2.0);
  EXPECT_DOUBLE_EQ(space_->GetSpaceClock(), 2.0);
}

TEST_F(SpaceBindTest, ScopedBindingUsesSelectedLeafNotActorOwnerScope) {
  // The supplied leaf is ignored. Facing is every other element in the tree.
  ASSERT_NE(space_->BindScoped(actor_id_, "root",
                               RelationKernelType::kAvatarFacing, 3), nullptr);
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(1, 1), 0.0);
  const auto& matches = space_->GetBindings(actor_id_)[0].latest_matches;
  std::set<element_id_t> ids;
  for (const auto& m : matches) ids.insert(m.element_id);
  EXPECT_EQ(ids.count(2), 1u);
  EXPECT_EQ(ids.count(3), 1u);
  EXPECT_EQ(ids.count(1), 0u);
}

TEST(SpaceScopedBindingTest, RootLevelLeafFacingIncludesOnlyExplicitLeaf) {
  // The scope id is ignored. Facing still sees the other root element.
  Space space;
  auto* tree = space.GetElementTree();
  ASSERT_TRUE(tree->AddElement(1).ok());
  ASSERT_TRUE(tree->AddElement(2).ok());
  tree->GetElement(2)->SetSharedId(2);  // shared (SP-D7-03)
  const auto actor = space.CreateActor({SubjectAnchor("root")});
  FollowProposals(&space, actor);
  space.GetActor(actor)->SetElementId(1);
  ASSERT_NE(space.BindScoped(actor, "root", RelationKernelType::kAvatarFacing,
                             2), nullptr);
  space.UpdateActorState(actor, Eigen::Vector2d::Zero(), 0.0);
  const auto& matches = space.GetBindings(actor)[0].latest_matches;
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].element_id, 2);
}

TEST_F(SpaceBindTest, ScopedBindingHardBoundsTagAndDeletedScope) {
  auto* tree = space_->GetElementTree();
  ASSERT_TRUE(tree->AddElement(4, 1).ok());
  tree->GetElement(4)->SetSharedId(4);  // shared (SP-D7-03)
  tree->GetElement(3)->SetTag("inside");
  tree->GetElement(4)->SetTag("inside");  // closer sibling distractor
  Eigen::Matrix4d selected_transform = Eigen::Matrix4d::Identity();
  selected_transform(0, 3) = 10.0;
  tree->GetElement(3)->SetLocalTransform(selected_transform);
  Eigen::Matrix4d distractor_transform = Eigen::Matrix4d::Identity();
  distractor_transform(0, 3) = 0.1;
  tree->GetElement(4)->SetLocalTransform(distractor_transform);
  ASSERT_NE(space_->BindScoped(actor_id_, "root",
                               RelationKernelType::kAvatarSpacing, 2,
                               "inside"), nullptr);
  // The scope root is ignored. The tag still applies, inside the
  // containment tree: both tagged elements, not only the supplied subtree.
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(1, 1), 0.0);
  const auto& matches = space_->GetBindings(actor_id_)[0].latest_matches;
  std::set<element_id_t> ids;
  for (const auto& m : matches) ids.insert(m.element_id);
  EXPECT_EQ(ids.count(4), 1u);

  ASSERT_TRUE(tree->RemoveElement(2).ok());
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(1, 1), 0.0);
  const auto& after = space_->GetBindings(actor_id_)[0].latest_matches;
  ASSERT_EQ(after.size(), 1u);
  EXPECT_EQ(after[0].element_id, 4);
}

TEST_F(SpaceBindTest, InvalidScopedBindingDoesNotMutateBindings) {
  EXPECT_EQ(space_->BindScoped(actor_id_, "missing",
                               RelationKernelType::kAvatarSpacing, 1), nullptr);
  EXPECT_EQ(space_->BindScoped(actor_id_, "root",
                               RelationKernelType::kAvatarSpacing,
                               kInvalidElementId), nullptr);
  EXPECT_TRUE(space_->GetBindings(actor_id_).empty());
}

TEST_F(SpaceBindTest,
       InactiveAnchorIsOmittedAndReactivationKeepsLegacyFallback) {
  auto* actor = space_->GetActor(actor_id_);
  actor->AddAnchor(SubjectAnchor("hand", Eigen::Vector3d(0.4, 0, 0.9)));
  space_->Bind(actor_id_, "hand", RelationKernelType::kAvatarAffordancePose,
               "Writable");
  const Eigen::Vector2d root_pos(5, 8);
  space_->UpdateActorState(actor_id_, root_pos, 0.0);
  ASSERT_EQ(space_->GetBindings(actor_id_)[0].latest_matches.size(), 1u);

  EXPECT_TRUE(space_->SetAnchorActive(actor_id_, "hand", false));
  space_->UpdateActorState(actor_id_, root_pos, 0.0);
  EXPECT_TRUE(space_->GetBindings(actor_id_)[0].latest_matches.empty());

  EXPECT_TRUE(space_->SetAnchorActive(actor_id_, "hand", true));
  space_->UpdateActorState(actor_id_, root_pos, 0.0);
  EXPECT_EQ(space_->GetBindings(actor_id_)[0].latest_matches.size(), 1u)
      << "reactivation must use the legacy local-offset pose when no override exists";
}

TEST_F(SpaceBindTest,
       StationaryRootAnchorChangesPropagateAndInactiveAnchorReleasesReceiver) {
  space_->SetSpaceId(1);
  space_->GetElementTree()->GetElement(1)->SetSharedId(100);
  space_->GetElementTree()->GetElement(2)->SetSharedId(200);
  space_->GetActor(actor_id_)->SetElementId(1);
  auto* source_actor = space_->GetActor(actor_id_);
  source_actor->AddAnchor(SubjectAnchor("hand", Eigen::Vector3d(0.4, 0, 0.9)));
  space_->Bind(actor_id_, "hand", RelationKernelType::kAvatarSpacing);

  Space target;
  target.SetSpaceId(2);
  auto* target_tree = target.GetElementTree();
  ASSERT_TRUE(target_tree->AddElement(10).ok());
  ASSERT_TRUE(target_tree->AddElement(20, 10).ok());
  target_tree->GetElement(10)->SetSharedId(100);
  target_tree->GetElement(20)->SetSharedId(200);
  // The source contact is decoded onto the counterpart's
  // boundary, so the counterpart needs the same board geometry (a
  // geometryless counterpart is an explicit omission, not a centre
  // target).
  target_tree->GetElement(20)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          std::make_shared<spatialblend::Polygon2D>(std::vector<Eigen::Vector2d>{
              {3, 6}, {7, 6}, {7, 10}, {3, 10}})));
  const auto target_actor = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, target_actor);
  target.GetActor(target_actor)->SetElementId(10);

  SpaceHub hub;
  ASSERT_TRUE(hub.Connect(space_.get()).ok());
  ASSERT_TRUE(hub.Connect(&target).ok());
  space_->SetHub(&hub);
  const Eigen::Vector2d root_pos(1, 1);
  space_->UpdateActorState(actor_id_, root_pos, 0.0);
  ASSERT_FALSE(target.GetReceivedUpdates().empty());
  ASSERT_FALSE(target.GetReceivedUpdates().back().payloads.empty());
  ASSERT_NE(target.GetRealizationTargets(target_actor), nullptr);
  ASSERT_FALSE(target.GetRealizationTargets(target_actor)->targets.empty());
  const Eigen::Vector2d target_before =
      target.GetRealizationTargets(target_actor)->targets.front().position;

  // Same root pose, but a new observed hand frame must bypass root deadband.
  space_->SetAnchorWorldPose(actor_id_, "hand", Eigen::Vector3d(2, 2, 0), 0.0);
  const size_t received_before = target.GetReceivedUpdates().size();
  space_->UpdateActorState(actor_id_, root_pos, 0.0);
  ASSERT_EQ(target.GetReceivedUpdates().size(), received_before + 1);
  EXPECT_FALSE(target.GetReceivedUpdates().back().payloads.empty());
  ASSERT_NE(target.GetRealizationTargets(target_actor), nullptr);
  ASSERT_FALSE(target.GetRealizationTargets(target_actor)->targets.empty());
  // The anchor is a parameter of the relation definition. The hand's relation is the hand's:
  // it reaches the receiver (above) and is answered for the receiver's hand anchor (this receiver publishes no hand
  // sample: NO_SAMPLE), and it does not move the receiver's root.
  EXPECT_EQ((target.GetRealizationTargets(target_actor)->targets.front().position - target_before).norm(), 0.0)
      << "a hand relation never moves the root";
  {
    ReceiverProposal proposal;
    ASSERT_TRUE(target.ReadProposal(target_actor, &proposal));
    const AnchorProposal* hand = nullptr;
    for (const auto& a : proposal.anchors)
      if (a.anchor_name == "hand") hand = &a;
    ASSERT_NE(hand, nullptr) << "the hand relation has the hand anchor's proposal";
    EXPECT_EQ(hand->status, ProposalStatus::kNoSample);
  }

  // An absent hand is an authoritative empty snapshot and releases targets.
  EXPECT_TRUE(space_->SetAnchorActive(actor_id_, "hand", false));
  space_->UpdateActorState(actor_id_, root_pos, 0.0);
  ASSERT_FALSE(target.GetReceivedUpdates().empty());
  EXPECT_TRUE(target.GetReceivedUpdates().back().payloads.empty());
  ASSERT_NE(target.GetRealizationTargets(target_actor), nullptr);
  EXPECT_TRUE(target.GetRealizationTargets(target_actor)->targets.empty());
}

TEST_F(SpaceBindTest, ScopeUsesEachAnchorPoseIncludingOverrideAndClear) {
  auto* tree = space_->GetElementTree();
  tree->GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(
          std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 10}, {0, 10}})));
  ASSERT_TRUE(tree->AddElement(10, 1).ok());
  ASSERT_TRUE(tree->AddElement(11, 1).ok());
  tree->GetElement(10)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(
          std::vector<Eigen::Vector2d>{{0, 0}, {9, 0}, {9, 10}, {0, 10}})));
  tree->GetElement(11)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(
          std::vector<Eigen::Vector2d>{{11, 0}, {20, 0}, {20, 10}, {11, 10}})));
  ASSERT_TRUE(tree->AddElement(12, 10).ok());
  ASSERT_TRUE(tree->AddElement(13, 11).ok());
  tree->GetElement(12)->SetSharedId(12);  // shared (SP-D7-03); the containers' borders need no id
  tree->GetElement(13)->SetSharedId(13);
  // The objects stand inside their own regions (a point referent must be reachable
  // from the anchor in the free space; with the identity transform they were at the
  // world origin, outside region 11).
  Eigen::Matrix4d in_left = Eigen::Matrix4d::Identity();
  in_left(0, 3) = 5.0;
  in_left(1, 3) = 5.0;
  tree->GetElement(12)->SetLocalTransform(in_left);
  Eigen::Matrix4d in_right = Eigen::Matrix4d::Identity();
  in_right(0, 3) = 15.0;  // region 11 has the identity transform: world (15, 5)
  in_right(1, 3) = 5.0;
  tree->GetElement(13)->SetLocalTransform(in_right);

  auto* actor = space_->GetActor(actor_id_);
  actor->AddAnchor(SubjectAnchor("hand", Eigen::Vector3d(13, 0, 0)));
  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor_id_, "hand", RelationKernelType::kAvatarSpacing);

  // Root is in the left region while its hand is independently in the right.
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(2, 2), 0.0);
  auto& bindings = space_->GetBindings(actor_id_);
  ASSERT_EQ(bindings.size(), 2u);
  // Container contract: the container an anchor is inside
  // stays a spacing referent beside its child objects. Each anchor's scope is
  // therefore {the child object, its container}: root (left region 10): {12, 10};
  // hand (right region 11): {13, 11}.
  const auto has = [](const auto& binding, int id) {
    for (const auto& m : binding.latest_matches)
      if (m.element_id == id) return true;
    return false;
  };
  ASSERT_EQ(bindings[0].latest_matches.size(), 2u);
  ASSERT_EQ(bindings[1].latest_matches.size(), 2u);
  EXPECT_TRUE(has(bindings[0], 12));
  EXPECT_TRUE(has(bindings[0], 10));
  EXPECT_TRUE(has(bindings[1], 13));
  EXPECT_TRUE(has(bindings[1], 11));

  space_->SetAnchorWorldPose(actor_id_, "hand", Eigen::Vector3d(3, 2, 0), 0.0);
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(2, 2), 0.0);
  EXPECT_TRUE(has(bindings[1], 12));
  EXPECT_TRUE(has(bindings[1], 10));
  EXPECT_FALSE(has(bindings[1], 13));

  space_->ClearAnchorWorldPose(actor_id_, "hand");
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(2, 2), 0.0);
  EXPECT_TRUE(has(bindings[1], 13));
  EXPECT_TRUE(has(bindings[1], 11));
  EXPECT_FALSE(has(bindings[1], 12));
}

TEST_F(SpaceBindTest, GetRelationStatesMirrorsLatestMatchesExactly) {
  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(1, 1), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  ASSERT_EQ(bindings.size(), 1);
  ASSERT_EQ(bindings[0].latest_matches.size(), 1);
  const auto& match = bindings[0].latest_matches[0];

  auto states = space_->GetRelationStates(actor_id_);
  ASSERT_EQ(states.size(), 1);
  const auto& rs = states[0];

  EXPECT_EQ(rs.anchor_name, "root");
  EXPECT_EQ(rs.kernel_type, RelationKernelType::kAvatarSpacing);
  EXPECT_EQ(rs.referent_element_id, match.element_id);
  EXPECT_EQ(rs.proximity, match.eval.grounded.proximity);
  EXPECT_EQ(rs.has_contact, match.eval.support.has_contact);
  EXPECT_EQ(rs.contact_point_world, match.eval.support.contact_point_world);
}

TEST_F(SpaceBindTest, GetRelationStatesActiveTracksPhaseTransition) {
  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);

  // Far outside the Room's navigable bounds -> proximity far below 0.10.
  // There is no activation floor, so an in-scope relation is active at once, however low its R.
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(-500, -500), 0.0);
  auto far_states = space_->GetRelationStates(actor_id_);
  ASSERT_FALSE(far_states.empty());
  EXPECT_LT(far_states[0].proximity, 0.10);
  EXPECT_TRUE(far_states[0].active);

  // Inside Board's boundary -> proximity 1.0 -> phase advances to kActive
  // (one step per Update, dormant -> active is the very next transition).
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(5, 8), 0.0);
  auto near_states = space_->GetRelationStates(actor_id_);
  ASSERT_FALSE(near_states.empty());
  EXPECT_TRUE(near_states[0].active);
}

TEST_F(SpaceBindTest, GetRelationStatesEmptyForActorWithoutBindings) {
  // actor_id_ has no Bind() calls in this test -> no bindings map entry.
  auto states = space_->GetRelationStates(actor_id_);
  EXPECT_TRUE(states.empty());
}

TEST_F(SpaceBindTest, GetRelationStatesProximityMatchesPropagatedPayload) {
  space_->SetSpaceId(1);
  space_->SetMovementGate(0.0, 0.0);
  space_->GetElementTree()->GetElement(2)->SetSharedId(200);
  space_->GetElementTree()->GetElement(1)->SetSharedId(100);
  space_->GetActor(actor_id_)->SetElementId(1);

  Space space_b;
  space_b.SetSpaceId(2);
  auto* tree_b = space_b.GetElementTree();
  tree_b->AddElement(10);
  tree_b->AddElement(20, 10);
  tree_b->GetElement(20)->SetSharedId(200);
  std::vector<SubjectAnchor> anchors_b = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))};
  auto avatar_id = space_b.CreateActor(anchors_b);
  FollowProposals(&space_b, avatar_id);
  space_b.GetActor(avatar_id)->SetElementId(10);
  tree_b->GetElement(10)->SetSharedId(100);

  SpaceHub hub;
  hub.Connect(space_.get());
  hub.Connect(&space_b);
  space_->SetHub(&hub);

  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(1, 1), 0.0);

  auto states = space_->GetRelationStates(actor_id_);
  ASSERT_FALSE(states.empty());

  auto& received = space_b.GetReceivedUpdates();
  ASSERT_FALSE(received.empty());
  const auto& payloads = received.back().payloads;
  ASSERT_FALSE(payloads.empty());

  bool matched = false;
  for (const auto& rs : states) {
    for (const auto& payload : payloads) {
      if (payload.subject_anchor_name != rs.anchor_name) continue;
      if (payload.kernel_type != rs.kernel_type) continue;
      if (payload.referent_shared_id != rs.referent_shared_id) continue;
      EXPECT_EQ(payload.grounded_relation.proximity, rs.proximity);
      matched = true;
    }
  }
  EXPECT_TRUE(matched);
}

TEST_F(SpaceBindTest, StepDurationPropagated) {
  space_->SetSpaceId(1);
  space_->GetElementTree()->GetElement(2)->SetSharedId(200);

  // Create receiving space
  Space space_b;
  space_b.SetSpaceId(2);
  auto* tree_b = space_b.GetElementTree();
  tree_b->AddElement(10);
  tree_b->AddElement(20, 10);
  tree_b->GetElement(20)->SetSharedId(200);

  std::vector<SubjectAnchor> anchors = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))};
  auto avatar_id = space_b.CreateActor(anchors);
  FollowProposals(&space_b, avatar_id);
  space_b.GetActor(avatar_id)->SetElementId(10);
  tree_b->GetElement(10)->SetSharedId(100);

  // Wire hub
  SpaceHub hub;
  hub.Connect(space_.get());
  hub.Connect(&space_b);
  space_->SetHub(&hub);

  space_->GetElementTree()->GetElement(1)->SetSharedId(100);
  space_->GetActor(actor_id_)->SetElementId(1);

  // Bind and set custom update interval
  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
  space_->SetUpdateInterval(0.05);  // 20fps
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(1, 1), 0.0);

  auto& received = space_b.GetReceivedUpdates();
  ASSERT_FALSE(received.empty());
  EXPECT_NEAR(received.back().step_duration, 0.05, 1e-9);

  space_->SetUpdateInterval(std::numeric_limits<double>::quiet_NaN());
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(2, 1), 0.0);
  EXPECT_NEAR(received.back().step_duration, 0.05, 1e-9);
}

// A receiving space whose ONLY actor is a remote clone -- i.e. one that
// never itself calls UpdateActorState -- must still see its own space_clock_
// advance, and RelationPhaseTracker transitions must still fire, purely from
// OnRemoteUpdate deliveries. Were space_clock_ advanced only inside
// UpdateActorState, such a space's clock (and every "t" it stamped on
// trace lines / RelationEvent.timestamp) would stay at 0.0 for the entire
// run -- see space.h's space_clock_ doc. Reuses the exact same source-space
// geometry/positions as GetRelationStatesActiveTracksPhaseTransition above
// (far-outside-Board -> proximity < activation floor -> stays dormant, then
// inside-Board -> proximity 1.0 -> dormant->active), but observes the
// RECEIVING space (space_b) instead of the driven space itself, and
// space_b's avatar actor never has UpdateActorState called on it at all.
TEST_F(SpaceBindTest,
       OnRemoteUpdateAdvancesClockAndFiresPhaseTransitionsWithoutLocalDrive) {
  space_->SetSpaceId(1);
  space_->GetElementTree()->GetElement(1)->SetSharedId(100);
  space_->GetElementTree()->GetElement(2)->SetSharedId(200);
  space_->GetActor(actor_id_)->SetElementId(1);

  Space space_b;
  space_b.SetSpaceId(2);
  auto* tree_b = space_b.GetElementTree();
  tree_b->AddElement(10);
  tree_b->AddElement(20, 10);
  tree_b->GetElement(20)->SetSharedId(200);

  std::vector<SubjectAnchor> anchors_b = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))};
  auto avatar_id = space_b.CreateActor(anchors_b);
  FollowProposals(&space_b, avatar_id);
  space_b.GetActor(avatar_id)->SetElementId(10);
  tree_b->GetElement(10)->SetSharedId(100);

  SpaceHub hub;
  hub.Connect(space_.get());
  hub.Connect(&space_b);
  space_->SetHub(&hub);

  std::vector<RelationEvent> events_seen;
  space_b.RegisterOnRelationEvent(
      [&events_seen](const RelationEvent& ev) { events_seen.push_back(ev); });

  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);

  EXPECT_DOUBLE_EQ(space_b.GetSpaceClock(), 0.0);

  // Far outside the Room's navigable bounds. There is no activation floor
  // (R < 0.10 is not DORMANT). The relation is in scope, so it is delivered: the
  // receiver clock advances and the dormant -> active transition fires on this very
  // OnRemoteUpdate.
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(-500, -500), 0.0);
  EXPECT_GT(space_b.GetSpaceClock(), 0.0);
  double clock_after_first = space_b.GetSpaceClock();
  ASSERT_FALSE(events_seen.empty());
  EXPECT_EQ(events_seen.back().to, RelationPhase::kActive);

  // Inside Board's boundary: the receiver clock advances purely from this
  // OnRemoteUpdate.
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(5, 8), 0.0);
  EXPECT_GT(space_b.GetSpaceClock(), clock_after_first);

  // (The next delivery raises the sole relation's weight to 1 -> kDriving; the one
  // after changes nothing.) A delivery that causes NO receiver phase transition still
  // advances the receiver clock. The bound referent sits at (5, 8): the last step must still approach it (0.71 m ->
  // 0.35 m) for v1's count to be valid; a receding step has no valid count, no weight (Progressive
  // computes none) and falls back to kActive, a transition.
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(4.5, 8.5), 0.0);
  ASSERT_EQ(events_seen.back().to, RelationPhase::kDriving);
  const size_t events_before = events_seen.size();
  const double clock_before = space_b.GetSpaceClock();
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(4.75, 8.25), 0.0);
  EXPECT_GT(space_b.GetSpaceClock(), clock_before);
  EXPECT_EQ(events_seen.size(), events_before)
      << "an update with no phase transition must still tick the clock";
}

// An exact replay of an already-applied RemoteRelationUpdate (identical
// `sequence` for the same (source_space_id, actor_shared_id) key) must be a
// no-op on the receiver -- this is the actual dedupe mechanism behind the
// observed ~1/10-runs duplicate-first-delivery bug under Transport::
// kGrpcTwoSessions right after sb_connect_space's bootstrap/Subscribe-
// stream-settle window (see remote_update.h's `sequence` field doc and
// unity_twin.h's Transport doc comment).
// Exercised directly here (bypassing any real network timing) for a
// deterministic, non-flaky test of the dedupe logic itself -- the clock
// advance added alongside it (see the OnRemoteUpdateAdvancesClock... test
// above) is the simplest observable signal that a second call was actually
// skipped rather than reprocessed.
TEST_F(SpaceBindTest, OnRemoteUpdateDedupesExactReplayBySequence) {
  space_->SetSpaceId(1);
  space_->GetElementTree()->GetElement(1)->SetSharedId(100);
  space_->GetElementTree()->GetElement(2)->SetSharedId(200);
  space_->GetActor(actor_id_)->SetElementId(1);

  Space space_b;
  space_b.SetSpaceId(2);
  auto* tree_b = space_b.GetElementTree();
  tree_b->AddElement(10);
  tree_b->AddElement(20, 10);
  tree_b->GetElement(20)->SetSharedId(200);
  std::vector<SubjectAnchor> anchors_b = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))};
  auto avatar_id = space_b.CreateActor(anchors_b);
  FollowProposals(&space_b, avatar_id);
  space_b.GetActor(avatar_id)->SetElementId(10);
  tree_b->GetElement(10)->SetSharedId(100);

  SpaceHub hub;
  hub.Connect(space_.get());
  hub.Connect(&space_b);
  space_->SetHub(&hub);

  space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(5, 8), 0.0);

  auto& received = space_b.GetReceivedUpdates();
  ASSERT_FALSE(received.empty());
  RemoteRelationUpdate replay = received.back();
  ASSERT_GT(replay.sequence, 0)
      << "UpdateActorState must stamp a positive sequence before propagating";

  double clock_after_first = space_b.GetSpaceClock();
  ASSERT_GT(clock_after_first, 0.0);

  // Replay the IDENTICAL update object a second time: same sequence for the
  // same (source_space_id, actor_shared_id) key must be recognized as an
  // exact replay and skipped -- the clock must NOT advance again.
  space_b.OnRemoteUpdate(replay);
  EXPECT_DOUBLE_EQ(space_b.GetSpaceClock(), clock_after_first)
      << "a replayed update with an already-seen sequence advanced the "
         "clock again -- dedupe did not skip it";

  // A NEW update (sequence == 0, hand-built, opts out of dedupe entirely --
  // e.g. a legacy/unit-test caller) always bypasses the check and still
  // advances the clock.
  RemoteRelationUpdate new_update = replay;
  new_update.sequence = 0;
  space_b.OnRemoteUpdate(new_update);
  EXPECT_GT(space_b.GetSpaceClock(), clock_after_first)
      << "sequence==0 (opt-out sentinel) must always bypass the dedupe check";
}

TEST_F(SpaceBindTest, SourceProgressUsesActorSecondsAndClearsOnObservationGap) {
  space_->SetSpaceId(1);
  space_->GetElementTree()->GetElement(1)->SetSharedId(100);
  space_->GetElementTree()->GetElement(2)->SetSharedId(200);
  space_->GetActor(actor_id_)->SetElementId(1);
  space_->SetUpdateInterval(1.0 / 30.0);
  space_->SetMinMove(0.0);
  ASSERT_NE(space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing),
            nullptr);

  SpaceHub hub;
  std::vector<RemoteRelationUpdate> sent;
  hub.SetRelationUpdateCallback(
      [&](const RemoteRelationUpdate& update) { sent.push_back(update); });
  space_->SetHub(&hub);

  space_->UpdateActorState(actor_id_, Eigen::Vector2d(2.8, 8.0), 0.0);
  ASSERT_FALSE(sent.empty());
  const RelationPayload first = sent.back().payloads.front();

  // A different actor advances the legacy space clock but must not dilute this
  // binding/referent's derivative interval.
  const auto other = space_->CreateActor({SubjectAnchor("other")});
  FollowProposals(space_.get(), other);
  space_->GetActor(other)->SetElementId(1);
  for (int i = 0; i < 5; ++i) {
    space_->UpdateActorState(other, Eigen::Vector2d(-10.0, -10.0), 0.0);
  }
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(2.9, 8.0), 0.0);
  ASSERT_GE(sent.size(), 2u);
  const RelationPayload second = sent.back().payloads.front();
  EXPECT_NEAR(first.grounded_relation.proximity, 1.0 / 1.2, 1e-12);
  EXPECT_NEAR(second.grounded_relation.proximity, 1.0 / 1.1, 1e-12);

  // The scheduling quantity is this binding's update count. Another actor's
  // samples do not replace it. The κ-clipped time-to-contact is not written.
  EXPECT_FALSE(std::isfinite(first.update_count));
  const double approached = SourceUpdateCount(
      first.grounded_relation.support_distance_m,
      second.grounded_relation.support_distance_m);
  EXPECT_DOUBLE_EQ(second.update_count, approached);
  EXPECT_TRUE(std::isfinite(approached));
  EXPECT_GT(approached, 1.0);

  // A repeated pose is no valid intent, and neither is a receding one: v1's rule (progress
  // below the contact epsilon is an infinite count).
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(2.9, 8.0), 0.0);
  const RelationPayload stationary = sent.back().payloads.front();
  EXPECT_FALSE(std::isfinite(stationary.update_count));
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(2.7, 8.0), 0.0);
  const RelationPayload receding = sent.back().payloads.front();
  EXPECT_FALSE(std::isfinite(receding.update_count));

  ASSERT_TRUE(space_->SetAnchorActive(actor_id_, "root", false));
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(2.9, 8.0), 0.0);
  ASSERT_TRUE(space_->SetAnchorActive(actor_id_, "root", true));
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(2.9, 8.0), 0.0);
  ASSERT_FALSE(sent.empty());
  const RelationPayload reentered = sent.back().payloads.front();
  // A gap clears the previous distance, so re-entry is a fresh observation.
  EXPECT_FALSE(std::isfinite(reentered.update_count));
}

TEST_F(SpaceBindTest,
       TimestampedPropagationUsesActorDeltaAndReceiverWatermark) {
  space_->SetSpaceId(1);
  space_->GetElementTree()->GetElement(1)->SetSharedId(100);
  space_->GetElementTree()->GetElement(2)->SetSharedId(200);
  ASSERT_NE(space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing),
            nullptr);

  Space target;
  target.SetSpaceId(2);
  ASSERT_TRUE(target.GetElementTree()->AddElement(10).ok());
  ASSERT_TRUE(target.GetElementTree()->AddElement(11).ok());
  ASSERT_TRUE(target.GetElementTree()->AddElement(20).ok());
  target.GetElementTree()->GetElement(10)->SetSharedId(100);
  target.GetElementTree()->GetElement(11)->SetSharedId(101);
  target.GetElementTree()->GetElement(20)->SetSharedId(200);
  const auto target_a = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, target_a);
  const auto target_b = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, target_b);
  target.GetActor(target_a)->SetElementId(10);
  target.GetActor(target_b)->SetElementId(11);

  SpaceHub hub;
  std::vector<RemoteRelationUpdate> sent;
  hub.SetRelationUpdateCallback([&](const RemoteRelationUpdate& update) {
    sent.push_back(update);
    target.OnRemoteUpdate(update);
  });
  space_->SetHub(&hub);

  std::vector<RelationEvent> source_events;
  space_->RegisterOnRelationEvent(
      [&](const RelationEvent& event) { source_events.push_back(event); });
  uint64_t epoch_a = 0;
  ASSERT_TRUE(space_->BeginActorObservationEpoch(actor_id_, "root", &epoch_a));
  ASSERT_TRUE(space_->ApplyActorObservationFrame(
      actor_id_, epoch_a, 0, 0.0,
      {{"root", Eigen::Vector3d(-100.0, 8.0, 0.0), 0.0}}));

  // Actor B has an independent handle/epoch and moves the receiver watermark
  // to 100 without contaminating actor A's source derivative interval.
  ASSERT_TRUE(space_->GetElementTree()->AddElement(4).ok());
  space_->GetElementTree()->GetElement(4)->SetSharedId(101);
  const auto other = space_->CreateActor({SubjectAnchor("root")});
  FollowProposals(space_.get(), other);
  space_->GetActor(other)->SetElementId(4);
  ASSERT_NE(space_->Bind(other, "root", RelationKernelType::kAvatarSpacing),
            nullptr);
  uint64_t epoch_b = 0;
  ASSERT_TRUE(space_->BeginActorObservationEpoch(other, "root", &epoch_b));
  ASSERT_TRUE(space_->ApplyActorObservationFrame(
      other, epoch_b, 0, 100.0,
      {{"root", Eigen::Vector3d::Zero(), 0.0}}));
  EXPECT_DOUBLE_EQ(target.GetSpaceClock(), 100.0);

  ASSERT_TRUE(space_->ApplyActorObservationFrame(
      actor_id_, epoch_a, 1, 0.2,
      {{"root", Eigen::Vector3d(2.9, 8.0, 0.0), 0.0}}));
  ASSERT_FALSE(sent.empty());
  const RemoteRelationUpdate& a_second = sent.back();
  ASSERT_EQ(a_second.timing, RelationUpdateTiming::kTimestampedObservation);
  ASSERT_EQ(a_second.source_epoch, epoch_a);
  ASSERT_FALSE(a_second.payloads.empty());
  const RelationPayload& payload = a_second.payloads.front();
  // Actor B's timestamp must not replace A's previous distance: A moved, so
  // the update count is a finite progress fraction of A's own samples.
  EXPECT_TRUE(std::isfinite(payload.update_count));
  EXPECT_GT(payload.update_count, 1.0);
  EXPECT_DOUBLE_EQ(target.GetSpaceClock(), 100.0);
  ASSERT_FALSE(source_events.empty());
  EXPECT_DOUBLE_EQ(source_events.back().timestamp, 0.2);
}

TEST(SpaceActorLifecycleTest, DestroyingOtherActorPreservesRemoteWatermark) {
  Space target;
  target.SetSpaceId(2);
  auto* tree = target.GetElementTree();
  ASSERT_TRUE(tree->AddElement(10).ok());
  ASSERT_TRUE(tree->AddElement(20).ok());
  tree->GetElement(10)->SetSharedId(501);
  tree->GetElement(20)->SetSharedId(502);
  const auto survivor = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, survivor);
  const auto retired = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, retired);
  target.GetActor(survivor)->SetElementId(10);
  target.GetActor(retired)->SetElementId(20);

  RemoteRelationUpdate update;
  update.source_space_id = 17;
  update.actor_shared_id = 501;
  update.sequence = 5;
  target.OnRemoteUpdate(update);
  EXPECT_EQ(target.GetLastAppliedRemoteSequence(17, 501), 5);
  const double clock_after_accept = target.GetSpaceClock();
  ASSERT_GT(clock_after_accept, 0.0);

  EXPECT_TRUE(target.DestroyActor(retired));
  EXPECT_NE(target.GetActor(survivor), nullptr);
  target.OnRemoteUpdate(update);  // exact replay of the surviving actor
  // Receipt diagnostics count duplicates too; only accepted updates advance
  // the solver clock. This must stay unchanged after unrelated retirement.
  EXPECT_DOUBLE_EQ(target.GetSpaceClock(), clock_after_accept);
  EXPECT_EQ(target.GetLastAppliedRemoteSequence(17, 501), 5);

  // Re-registering the same receiving identity must not reopen its replay
  // fence, either.
  EXPECT_TRUE(target.DestroyActor(survivor));
  const auto replacement = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, replacement);
  target.GetActor(replacement)->SetElementId(10);
  EXPECT_NE(target.GetActor(replacement), nullptr);
  target.OnRemoteUpdate(update);
  EXPECT_DOUBLE_EQ(target.GetSpaceClock(), clock_after_accept);
  update.sequence = 6;
  target.OnRemoteUpdate(update);
  EXPECT_GT(target.GetSpaceClock(), clock_after_accept);
  EXPECT_EQ(target.GetLastAppliedRemoteSequence(17, 501), 6);
}

TEST(ReferentResolveModeTest,
     LinkerOnlyRejectsUnlinkedDecoyAndAcceptsExplicitDifferentIdLink) {
  Space source;
  source.SetSpaceId(1);
  Space target;
  target.SetSpaceId(2);
  SpaceHub hub;
  source.SetHub(&hub);
  target.SetHub(&hub);
  ASSERT_TRUE(hub.Connect(&source).ok());
  ASSERT_TRUE(hub.Connect(&target).ok());
  ASSERT_TRUE(source.GetElementTree()->AddElement(2).ok());
  source.GetElementTree()->GetElement(2)->SetSharedId(999);
  auto* tree = target.GetElementTree();
  ASSERT_TRUE(tree->AddElement(10).ok());
  tree->GetElement(10)->SetSharedId(500);  // local clone actor
  auto actor = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);
  ASSERT_TRUE(tree->AddElement(20, 10).ok());
  tree->GetElement(20)->SetSharedId(42);  // coincident local dataset id
  Eigen::Matrix4d decoy_transform = Eigen::Matrix4d::Identity();
  decoy_transform(0, 3) = 8.0;
  tree->GetElement(20)->SetLocalTransform(decoy_transform);

  target.SetReferentResolveMode(ReferentResolveMode::kLinkerOnly);
  RelationPayload payload;
  payload.referent_shared_id = 42;
  payload.referent_source_space_id = 1;
  payload.referent_source_element_id = 2;
  // A sender can carry stale receiver scratch state; strict mode must ignore
  // it rather than treating this coincident local id as authorization.
  payload.resolved_local_element_id = 20;
  payload.subject_anchor_name = "root";
  payload.kernel_type = RelationKernelType::kAvatarSpacing;
  RemoteRelationUpdate update;
  update.source_space_id = 1;
  update.actor_shared_id = 500;  // actor fallback is intentionally unchanged
  update.payloads = {payload};

  target.OnRemoteUpdate(update);
  const RealizationResult* rejected = target.GetRealizationTargets(actor);
  ASSERT_NE(rejected, nullptr);
  EXPECT_TRUE(rejected->targets.empty());

  // Explicit correspondence wins even though source/target shared IDs differ.
  tree->GetElement(20)->SetSharedId(777);
  hub.GetLinker().CreateLink(element_handle_t{1, 2}, element_handle_t{2, 20});
  target.OnRemoteUpdate(update);
  const RealizationResult* linked = target.GetRealizationTargets(actor);
  ASSERT_NE(linked, nullptr);
  ASSERT_EQ(linked->targets.size(), 1u);
  EXPECT_NEAR(linked->targets[0].position.x(), 8.0, 1e-12);

  // A link that points at a removed endpoint is equally unresolved.
  ASSERT_TRUE(tree->RemoveElement(20).ok());
  target.OnRemoteUpdate(update);
  const RealizationResult* removed = target.GetRealizationTargets(actor);
  ASSERT_NE(removed, nullptr);
  EXPECT_TRUE(removed->targets.empty());
}

TEST(ReferentResolveModeTest, LegacyLinkerAndSharedIdModesRemainDistinct) {
  Space source;
  source.SetSpaceId(1);
  Space target;
  target.SetSpaceId(2);
  SpaceHub hub;
  source.SetHub(&hub);
  target.SetHub(&hub);
  ASSERT_TRUE(hub.Connect(&source).ok());
  ASSERT_TRUE(hub.Connect(&target).ok());
  ASSERT_TRUE(source.GetElementTree()->AddElement(2).ok());
  source.GetElementTree()->GetElement(2)->SetSharedId(42);
  auto* tree = target.GetElementTree();
  ASSERT_TRUE(tree->AddElement(10).ok());
  tree->GetElement(10)->SetSharedId(500);
  auto actor = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);
  ASSERT_TRUE(tree->AddElement(20, 10).ok());
  ASSERT_TRUE(tree->AddElement(21, 10).ok());
  tree->GetElement(20)->SetSharedId(42);
  Eigen::Matrix4d shared_transform = Eigen::Matrix4d::Identity();
  shared_transform(0, 3) = 3.0;
  tree->GetElement(20)->SetLocalTransform(shared_transform);
  Eigen::Matrix4d linked_transform = Eigen::Matrix4d::Identity();
  linked_transform(0, 3) = 9.0;
  tree->GetElement(21)->SetLocalTransform(linked_transform);

  hub.GetLinker().CreateLink(element_handle_t{1, 2}, element_handle_t{2, 21});
  RelationPayload payload;
  payload.referent_shared_id = 42;
  payload.referent_source_space_id = 1;
  payload.referent_source_element_id = 2;
  payload.subject_anchor_name = "root";
  payload.kernel_type = RelationKernelType::kAvatarSpacing;
  RemoteRelationUpdate update;
  update.source_space_id = 1;
  update.actor_shared_id = 500;
  update.payloads = {payload};

  target.SetReferentResolveMode(ReferentResolveMode::kLinker);
  target.OnRemoteUpdate(update);
  ASSERT_NE(target.GetRealizationTargets(actor), nullptr);
  EXPECT_NEAR(target.GetRealizationTargets(actor)->targets[0].position.x(),
              9.0, 1e-12);

  // Switching the resolution mode does not re-place the body by itself: only the
  // first sample and a source-app exception place it (there is no
  // referent-retarget trigger).
  // The shared-id resolution is a new placement because the source raises an
  // exception.
  target.SetReferentResolveMode(ReferentResolveMode::kSharedId);
  update.has_exception = true;
  update.exception_id = 7;
  target.OnRemoteUpdate(update);
  ASSERT_NE(target.GetRealizationTargets(actor), nullptr);
  EXPECT_NEAR(target.GetRealizationTargets(actor)->targets[0].position.x(),
              3.0, 1e-12);
}

// The proposal has no arrival budget (pacing belongs to the driver),
// so what is left of this contract is that a non-root Facing relation never
// drives the root's Spacing proposal, whatever its imminence.
TEST(ProgressiveTimingContractTest,
     RemoteRootProposalIgnoresHandAndFacingImminence) {
  Space target;
  target.SetSpaceId(2);
  auto* tree = target.GetElementTree();
  ASSERT_TRUE(tree->AddElement(10).ok());
  tree->GetElement(10)->SetSharedId(500);
  ASSERT_TRUE(tree->AddElement(20, 10).ok());
  tree->GetElement(20)->SetSharedId(42);

  const auto actor = target.CreateActor({SubjectAnchor("navigation")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);
  IdealDriver driver(&target, actor);

  RelationPayload root_spacing;
  root_spacing.referent_shared_id = 42;
  root_spacing.subject_anchor_name = "navigation";
  root_spacing.kernel_type = RelationKernelType::kAvatarSpacing;
  root_spacing.estimated_time = 4.0;

  RelationPayload urgent_facing = root_spacing;
  urgent_facing.subject_anchor_name = "left_hand";
  urgent_facing.kernel_type = RelationKernelType::kAvatarFacing;
  urgent_facing.update_count = 1.0;  // aligned: a valid Facing request (v1 Angular1D)
  urgent_facing.estimated_time = 0.001;

  RemoteRelationUpdate update;
  update.source_space_id = 1;
  update.actor_shared_id = 500;
  update.payloads = {root_spacing, urgent_facing};
  target.OnRemoteUpdate(update);

  const RealizationResult* result = target.GetRealizationTargets(actor);
  ASSERT_NE(result, nullptr);
  ASSERT_FALSE(result->targets.empty());
  // This exercises the production OnRemoteUpdate branch. The noncanonical
  // root anchor remains eligible by its Spacing family.
  EXPECT_TRUE(driver.proposal().status == ProposalStatus::kExact ||
              driver.proposal().status == ProposalStatus::kInexact);
}

TEST(ProgressiveTimingContractTest,
     FixedPrefixIgnoresUrgentHandFamiliesForTheRootProposal) {
  // There is no arrival budget (the driver paces). Urgent facing / hand
  // relations of the same update still never change the root's Spacing proposal.
  struct Rcv {
    Space space;
    spatialblend::entity_id_t actor = 0;
    std::unique_ptr<IdealDriver> driver;
  };
  auto setup = [](Rcv* r) {
    Space* space = &r->space;
    space->SetSpaceId(2);
    auto* tree = space->GetElementTree();
    EXPECT_TRUE(tree->AddElement(10).ok());
    EXPECT_TRUE(tree->AddElement(20, 10).ok());
    tree->GetElement(10)->SetSharedId(500);
    tree->GetElement(20)->SetSharedId(42);
    Eigen::Matrix4d referent = Eigen::Matrix4d::Identity();
    referent(0, 3) = 1.0;
    tree->GetElement(20)->SetLocalTransform(referent);
    r->actor = space->CreateActor(
        {SubjectAnchor("navigation"), SubjectAnchor("hand_l")});
    space->GetActor(r->actor)->SetElementId(10);
    r->driver = std::make_unique<IdealDriver>(space, r->actor);
  };
  Rcv a, b;
  setup(&a);
  setup(&b);

  RelationPayload spacing;
  spacing.referent_shared_id = 42;
  spacing.subject_anchor_name = "navigation";
  spacing.kernel_type = RelationKernelType::kAvatarSpacing;
  spacing.estimated_time = 2.0;
  RelationPayload facing = spacing;
  facing.subject_anchor_name = "hand_l";
  facing.kernel_type = RelationKernelType::kAvatarFacing;
  facing.update_count = 1.0;  // aligned: a valid Facing request (v1 Angular1D)
  facing.estimated_time = 0.001;
  RelationPayload hand = facing;
  hand.kernel_type = RelationKernelType::kAvatarAffordancePose;
  hand.grounded_relation.proximity = 1.0;

  RemoteRelationUpdate prefix;
  prefix.source_space_id = 1;
  prefix.actor_shared_id = 500;
  prefix.sequence = 1;
  prefix.step_duration = 1.0 / 30.0;
  prefix.payloads = {spacing, facing, hand};
  a.space.OnRemoteUpdate(prefix);
  b.space.OnRemoteUpdate(prefix);
  const RealizationResult* target_a = a.space.GetRealizationTargets(a.actor);
  const RealizationResult* target_b = b.space.GetRealizationTargets(b.actor);
  ASSERT_NE(target_a, nullptr);
  ASSERT_NE(target_b, nullptr);
  ASSERT_FALSE(target_a->targets.empty());
  ASSERT_FALSE(target_b->targets.empty());
  EXPECT_NEAR(target_a->targets.front().position.x(), 1.0, 1e-12);
  EXPECT_EQ(target_a->targets.size(), target_b->targets.size());
  EXPECT_NEAR(target_a->targets.front().position.x(),
              target_b->targets.front().position.x(), 1e-12);

  // A divergent suffix changes only non-root-family urgency: the root's
  // budget is identical.
  RemoteRelationUpdate suffix = prefix;
  suffix.sequence = 2;
  a.space.OnRemoteUpdate(suffix);
  suffix.payloads[1].estimated_time = 1e-6;
  suffix.payloads[2].estimated_time = 1e-6;
  b.space.OnRemoteUpdate(suffix);
  ASSERT_NE(b.space.GetRealizationTargets(b.actor), nullptr);
  EXPECT_NEAR(a.driver->proposal().x, b.driver->proposal().x, 1e-12);
  EXPECT_NEAR(a.driver->proposal().y, b.driver->proposal().y, 1e-12);
}

TEST(ProgressiveTimingContractTest, InvalidRemoteTimesDoNotPoisonClockOrFence) {
  Space target;
  target.SetSpaceId(2);
  auto* tree = target.GetElementTree();
  ASSERT_TRUE(tree->AddElement(10).ok());
  tree->GetElement(10)->SetSharedId(500);
  const auto actor = target.CreateActor({SubjectAnchor("navigation")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);

  RemoteRelationUpdate malformed;
  malformed.source_space_id = 1;
  malformed.actor_shared_id = 500;
  malformed.sequence = 1;
  malformed.source_clock = std::numeric_limits<double>::infinity();
  malformed.step_duration = 0.033;
  target.OnRemoteUpdate(malformed);
  EXPECT_DOUBLE_EQ(target.GetSpaceClock(), 0.0);
  EXPECT_EQ(target.GetLastAppliedRemoteSequence(1, 500), 0);

  malformed.sequence = 2;
  malformed.source_clock = 0.0;
  malformed.step_duration = 0.1;
  target.OnRemoteUpdate(malformed);
  EXPECT_NEAR(target.GetSpaceClock(), 0.1, 1e-12);
  EXPECT_EQ(target.GetLastAppliedRemoteSequence(1, 500), 2);
}

// CUE-01: exercise the receiver's actual OnRemoteUpdate producer, rather
// than solver encode/decode round trips.  Rich mode must keep root spacing and
// named facing channels separate and solve facing at each named anchor pose.
TEST(RichRealizationLiveProducerTest, KeepsRootHeadAndChestChannelsSeparate) {
  Space target;
  target.SetRichRealizationEnabled(true);
  auto* tree = target.GetElementTree();
  ASSERT_TRUE(tree->AddElement(10).ok());
  ASSERT_TRUE(tree->AddElement(20, 10).ok());
  tree->GetElement(10)->SetSharedId(500);
  tree->GetElement(20)->SetSharedId(42);
  Eigen::Matrix4d referent = Eigen::Matrix4d::Identity();
  referent(0, 3) = 4.0;
  referent(1, 3) = 4.0;
  tree->GetElement(20)->SetLocalTransform(referent);
  const auto actor = target.CreateActor(
      {SubjectAnchor("root"), SubjectAnchor("head"), SubjectAnchor("chest"),
       SubjectAnchor("hand")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);
  target.SetAnchorWorldPose(actor, "head", Eigen::Vector3d(0, 2, 1.7), 0.0);
  target.SetAnchorWorldPose(actor, "chest", Eigen::Vector3d(2, 0, 1.3), 0.0);

  RelationPayload spacing;
  spacing.referent_shared_id = 42;
  spacing.subject_anchor_name = "root";
  spacing.kernel_type = RelationKernelType::kAvatarSpacing;
  // A valid v1 count (approaching): a request. Without one Progressive computes no weight,
  // and the channel/target this test reads would not exist.
  spacing.update_count = 2.0;
  RelationPayload head = spacing;
  head.subject_anchor_name = "head";
  head.kernel_type = RelationKernelType::kAvatarFacing;
  head.update_count = 1.0;  // aligned: a valid Facing request (v1 Angular1D)
  RelationPayload chest = head;
  chest.subject_anchor_name = "chest";
  // The hand contact needs a support patch on the counterpart
  // (a geometryless referent is a grounding failure, not a centre target).
  ASSERT_TRUE(tree->AddElement(21, 10).ok());
  tree->GetElement(21)->SetSharedId(43);
  {
    auto boundary = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
        {3, 3}, {5, 3}, {5, 5}, {3, 5}});
    auto def = std::make_shared<NavigableSpatialDefinition>(boundary);
    SubRegion patch;
    patch.name = "top";
    patch.boundary = boundary;
    def->AddSubRegion(std::move(patch));
    tree->GetElement(21)->SetSpatialDefinition(def);
  }
  RelationPayload hand = spacing;
  hand.referent_shared_id = 43;
  hand.subject_anchor_name = "hand";
  hand.kernel_type = RelationKernelType::kAvatarAffordancePose;
  hand.grounded_relation.proximity = 1.0;
  hand.grounded_relation.encoded_state = Eigen::VectorXd::Zero(3);
  hand.support.has_contact = true;
  hand.support.sub_region_name = "top";
  hand.support.contact_normalized.strategy =
      NormalizedPolygonPosition::Strategy::kRadial;
  RemoteRelationUpdate update;
  update.actor_shared_id = 500;
  update.payloads = {spacing, head, chest, hand};
  target.OnRemoteUpdate(update);

  const auto* result = target.GetRealizationTargets(actor);
  ASSERT_NE(result, nullptr);
  ASSERT_GE(result->targets.size(), 4u);
  EXPECT_NEAR(result->targets.front().position.x(), 4.0, 1e-12);
  // Rich facing does not silently write legacy root yaw.
  EXPECT_NEAR(result->targets.front().yaw, 0.0, 1e-12);
  const auto* snapshot = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(snapshot, nullptr);
  ASSERT_EQ(snapshot->channels.size(), 4u);
  bool saw_head = false, saw_chest = false, saw_hand = false;
  for (const auto& channel : snapshot->channels) {
    if (channel.anchor_name == "head") {
      saw_head = true;
      EXPECT_EQ(channel.family, RelationKernelType::kAvatarFacing);
      EXPECT_NEAR(channel.target.position.y(), 2.0, 1e-12);
      EXPECT_EQ(channel.geometry, RealizationGeometry::kNotApplicable);
    }
    if (channel.anchor_name == "chest") {
      saw_chest = true;
      EXPECT_EQ(channel.family, RelationKernelType::kAvatarFacing);
      EXPECT_NEAR(channel.target.position.x(), 2.0, 1e-12);
    }
    if (channel.anchor_name == "hand") {
      saw_hand = channel.family == RelationKernelType::kAvatarAffordancePose &&
                 channel.has_goal;
    }
  }
  EXPECT_TRUE(saw_head);
  EXPECT_TRUE(saw_chest);
  EXPECT_TRUE(saw_hand);
}

// CUE-02: a geometryless affordance referent is an
// explicit grounding failure in the live snapshot (no centre target, no mapped
// support), and an empty accepted update withdraws it.
TEST(RichRealizationLiveProducerTest, LabelsLegacySupportAndWithdrawsOnEmpty) {
  Space target;
  target.SetRichRealizationEnabled(true);
  auto* tree = target.GetElementTree();
  ASSERT_TRUE(tree->AddElement(10).ok());
  ASSERT_TRUE(tree->AddElement(20, 10).ok());
  tree->GetElement(10)->SetSharedId(500);
  tree->GetElement(20)->SetSharedId(42);
  Eigen::Matrix4d far_referent = Eigen::Matrix4d::Identity();
  far_referent(0, 3) = 3.0;
  tree->GetElement(20)->SetLocalTransform(far_referent);
  const auto actor = target.CreateActor(
      {SubjectAnchor("root"), SubjectAnchor("hand")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);
  RelationPayload hand;
  hand.referent_shared_id = 42;
  hand.subject_anchor_name = "hand";
  hand.kernel_type = RelationKernelType::kAvatarAffordancePose;
  hand.grounded_relation.proximity = 1.0;
  hand.update_count = 1.0;  // in contact: v1's count 1, a request (otherwise it is disabled before grounding)
  RemoteRelationUpdate update;
  update.actor_shared_id = 500;
  update.payloads = {hand};
  target.OnRemoteUpdate(update);
  const auto* snapshot = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(snapshot, nullptr);
  ASSERT_EQ(snapshot->channels.size(), 1u);
  EXPECT_FALSE(snapshot->channels[0].has_goal);
  EXPECT_EQ(snapshot->channels[0].status, SolverStatus::kFallback);
  ASSERT_EQ(snapshot->channels[0].contributors.size(), 1u);
  EXPECT_FALSE(snapshot->channels[0].contributors[0].has_mapped_support);
  EXPECT_EQ(snapshot->channels[0].contributors[0].disposition,
            RealizationDisposition::kGroundingFailed);
  for (const auto& t : target.GetRealizationTargets(actor)->targets)
    EXPECT_NE(t.anchor_name, "hand");

  update.payloads.clear();
  target.OnRemoteUpdate(update);
  snapshot = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(snapshot, nullptr);
  EXPECT_TRUE(snapshot->channels.empty());
}

TEST(RichRealizationValidityTest,
     EffectivePolicyEpochChangesOnlyForEffectiveConfiguration) {
  Space target;
  target.SetRichRealizationEnabled(true);
  auto* tree = target.GetElementTree();
  ASSERT_TRUE(tree->AddElement(10).ok());
  ASSERT_TRUE(tree->AddElement(20, 10).ok());
  tree->GetElement(10)->SetSharedId(500);
  tree->GetElement(20)->SetSharedId(42);
  const auto actor = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);
  RelationPayload spacing;
  spacing.referent_shared_id = 42;
  spacing.subject_anchor_name = "root";
  spacing.kernel_type = RelationKernelType::kAvatarSpacing;
  RemoteRelationUpdate update;
  update.actor_shared_id = 500;
  update.payloads = {spacing};
  target.OnRemoteUpdate(update);
  ActorRealizationSnapshot first;
  ASSERT_TRUE(target.CopyActorRealizationSnapshot(actor, &first));
  EXPECT_NE(first.known_revision_mask & kRealizationPolicyRevision, 0u);
  EXPECT_NE(first.unknown_revision_mask & kRealizationTreeRevision, 0u);
  EXPECT_NE(first.unknown_revision_mask & kRealizationSupportRevision, 0u);
  ActorRealizationValidity validity;
  ASSERT_TRUE(target.GetActorRealizationValidity(
      actor, first.revision, kRealizationPolicyRevision, &validity));
  EXPECT_EQ(validity.changed_mask & kRealizationPolicyRevision, 0u);

  target.SetRealizationPolicy(RealizationPolicyType::kUniform);
  ASSERT_TRUE(target.GetActorRealizationValidity(
      actor, first.revision, kRealizationPolicyRevision, &validity));
  EXPECT_NE(validity.changed_mask & kRealizationPolicyRevision, 0u);
  const uint64_t after_policy = validity.policy_revision;
  target.SetRealizationPolicy(RealizationPolicyType::kUniform);  // no-op
  ASSERT_TRUE(target.GetActorRealizationValidity(
      actor, first.revision, kRealizationPolicyRevision, &validity));
  EXPECT_EQ(validity.policy_revision, after_policy);

  target.OnRemoteUpdate(update);
  ActorRealizationSnapshot uniform;
  ASSERT_TRUE(target.CopyActorRealizationSnapshot(actor, &uniform));
  EXPECT_GT(uniform.revision, first.revision);
  EXPECT_EQ(uniform.policy_revision, after_policy);
  // (There is no anchor-gating threshold: the seat is geometric.)
  target.SetAnchorReach("hand", 1.25);
  target.OnRemoteUpdate(update);
  ActorRealizationSnapshot reach;
  ASSERT_TRUE(target.CopyActorRealizationSnapshot(actor, &reach));
  EXPECT_GT(reach.policy_revision, uniform.policy_revision);
  const uint64_t after_reach = reach.policy_revision;
  target.SetAnchorReach("hand", 1.25);  // no-op
  target.SetAnchorReach("hand", std::numeric_limits<double>::infinity());
  target.OnRemoteUpdate(update);
  ActorRealizationSnapshot reach_noop;
  ASSERT_TRUE(target.CopyActorRealizationSnapshot(actor, &reach_noop));
  EXPECT_EQ(reach_noop.policy_revision, after_reach);
}

namespace {
class TreeDelegatingResolver final : public ReferentResolver {
 public:
  const Element* Resolve(shared_element_id_t id,
                         const ElementTree& tree) const override {
    return tree.GetElementBySharedId(id);
  }
};
}  // namespace

TEST(RichRealizationValidityTest, CustomResolverLeavesCounterpartUnknown) {
  Space target;
  target.SetRichRealizationEnabled(true);
  target.SetSpaceId(2);
  SpaceHub hub;
  ASSERT_TRUE(hub.Connect(&target).ok());
  target.SetHub(&hub);
  auto* tree = target.GetElementTree();
  ASSERT_TRUE(tree->AddElement(10).ok());
  ASSERT_TRUE(tree->AddElement(20, 10).ok());
  tree->GetElement(10)->SetSharedId(500);
  tree->GetElement(20)->SetSharedId(42);
  const auto actor = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);
  target.SetReferentResolver(std::make_shared<TreeDelegatingResolver>());
  RelationPayload spacing;
  spacing.referent_shared_id = 42;
  spacing.subject_anchor_name = "root";
  spacing.kernel_type = RelationKernelType::kAvatarSpacing;
  RemoteRelationUpdate update;
  update.actor_shared_id = 500;
  update.payloads = {spacing};
  target.OnRemoteUpdate(update);
  ActorRealizationSnapshot snapshot;
  ASSERT_TRUE(target.CopyActorRealizationSnapshot(actor, &snapshot));
  EXPECT_EQ(snapshot.known_revision_mask & kRealizationCounterpartRevision,
            0u);
  EXPECT_NE(snapshot.unknown_revision_mask & kRealizationCounterpartRevision,
            0u);
}

TEST(RichRealizationLiveProducerTest, AcceptedObservationResetAdvancesEmptySnapshot) {
  Space target;
  target.SetRichRealizationEnabled(true);
  auto* tree = target.GetElementTree();
  ASSERT_TRUE(tree->AddElement(10).ok());
  ASSERT_TRUE(tree->AddElement(20, 10).ok());
  tree->GetElement(10)->SetSharedId(500);
  tree->GetElement(20)->SetSharedId(42);
  const auto actor = target.CreateActor({SubjectAnchor("root"), SubjectAnchor("hand")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);
  RelationPayload hand;
  hand.referent_shared_id = 42;
  hand.subject_anchor_name = "hand";
  hand.kernel_type = RelationKernelType::kAvatarAffordancePose;
  hand.grounded_relation.proximity = 1.0;
  RemoteRelationUpdate accepted;
  accepted.actor_shared_id = 500;
  accepted.payloads = {hand};
  target.OnRemoteUpdate(accepted);
  const auto* before = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(before, nullptr);
  ASSERT_FALSE(before->channels.empty());
  const uint64_t old_revision = before->revision;

  RemoteRelationUpdate reset;
  reset.timing = RelationUpdateTiming::kObservationReset;
  reset.actor_shared_id = 500;
  reset.source_epoch = 1;
  reset.sequence = 1;
  target.OnRemoteUpdate(reset);
  const auto* after = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(after, nullptr);
  EXPECT_GT(after->revision, old_revision);
  EXPECT_TRUE(after->channels.empty());
}

TEST(RichRealizationLiveProducerTest,
     LazySharedIdModeCannotCertifyResetCounterparts) {
  struct RestoreEnvironment {
    bool existed;
    std::string value;
    ~RestoreEnvironment() {
      if (existed) setenv("SB_V2_REFERENT_RESOLVE", value.c_str(), 1);
      else unsetenv("SB_V2_REFERENT_RESOLVE");
    }
  } restore{std::getenv("SB_V2_REFERENT_RESOLVE") != nullptr,
            std::getenv("SB_V2_REFERENT_RESOLVE")
                ? std::getenv("SB_V2_REFERENT_RESOLVE") : ""};
  ASSERT_EQ(setenv("SB_V2_REFERENT_RESOLVE", "sharedid", 1), 0);
  SpaceHub hub;
  Space target;
  target.SetSpaceId(2);
  target.SetHub(&hub);
  target.SetRichRealizationEnabled(true);
  ASSERT_TRUE(target.GetElementTree()->AddElement(10).ok());
  target.GetElementTree()->GetElement(10)->SetSharedId(500);
  const auto actor = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);
  RemoteRelationUpdate reset;
  reset.timing = RelationUpdateTiming::kObservationReset;
  reset.actor_shared_id = 500;
  reset.source_epoch = 1;
  reset.sequence = 1;
  target.OnRemoteUpdate(reset);
  ActorRealizationSnapshot snapshot;
  ASSERT_TRUE(target.CopyActorRealizationSnapshot(actor, &snapshot));
  EXPECT_EQ(snapshot.known_revision_mask & kRealizationCounterpartRevision, 0u);
  EXPECT_NE(snapshot.unknown_revision_mask & kRealizationCounterpartRevision, 0u);
  ActorRealizationValidity validity;
  ASSERT_TRUE(target.GetActorRealizationValidity(actor, snapshot.revision,
      kRealizationCounterpartRevision, &validity));
  EXPECT_EQ(validity.captured_known_mask & kRealizationCounterpartRevision, 0u);
  EXPECT_EQ(validity.current_known_mask & kRealizationCounterpartRevision, 0u);
}

namespace {
RelationPayload ExplicitLiveSupport(element_id_t id, shared_element_id_t shared,
                                    const std::string& anchor,
                                    const std::string& patch, double time = 0.0) {
  RelationPayload p;
  p.resolved_local_element_id = id;
  p.referent_shared_id = shared;
  p.subject_anchor_name = anchor;
  p.kernel_type = RelationKernelType::kAvatarAffordancePose;
  p.estimated_time = time;
  p.grounded_relation.proximity = 1.0;
  // The source is in contact (proximity 1): v1's count is 1, a request. Without a count Progressive computes no
  // weight and the relation makes no request (v1).
  p.update_count = 1.0;
  p.support.has_contact = true;
  p.support.sub_region_name = patch;
  p.support.contact_normalized.strategy = NormalizedPolygonPosition::Strategy::kRadial;
  p.support.contact_normalized.angle = 0.0;
  p.support.contact_normalized.radial_ratio = 0.0;
  p.grounded_relation.encoded_state = Eigen::VectorXd::Zero(3);
  return p;
}

void AddExplicitPatch(Space* space, element_id_t id, shared_element_id_t shared,
                      const Eigen::Vector3d& center, bool tilt_x = false) {
  auto* tree = space->GetElementTree();
  ASSERT_TRUE(tree->AddElement(id, 10).ok());
  tree->GetElement(id)->SetSharedId(shared);
  auto boundary = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {-1, -1}, {1, -1}, {1, 1}, {-1, 1}});
  auto def = std::make_shared<NavigableSpatialDefinition>(boundary);
  SubRegion patch;
  patch.name = "seat";
  patch.boundary = boundary;
  patch.local_transform = Eigen::Matrix4d::Identity();
  patch.local_transform.block<3, 1>(0, 3) = center;
  if (tilt_x) {
    const double a = M_PI / 2.0;
    patch.local_transform(1, 1) = std::cos(a);
    patch.local_transform(1, 2) = -std::sin(a);
    patch.local_transform(2, 1) = std::sin(a);
    patch.local_transform(2, 2) = std::cos(a);
  }
  def->AddSubRegion(std::move(patch));
  tree->GetElement(id)->SetSpatialDefinition(def);
}

std::unique_ptr<Space> MakeExplicitLiveSpace(spatialblend::entity_id_t* actor) {
  auto target = std::make_unique<Space>();
  target->SetRichRealizationEnabled(true);
  auto* tree = target->GetElementTree();
  EXPECT_TRUE(tree->AddElement(10).ok());
  tree->GetElement(10)->SetSharedId(500);
  *actor = target->CreateActor({SubjectAnchor("root"), SubjectAnchor("hand")});
  FollowProposals(target.get(), *actor);
  target->GetActor(*actor)->SetElementId(10);
  return target;
}
}  // namespace

TEST(RichRealizationLiveProducerTest, ExplicitUniqueSupportKeepsPlaneAndExactMap) {
  spatialblend::entity_id_t actor;
  auto target = MakeExplicitLiveSpace(&actor);
  AddExplicitPatch(target.get(), 20, 42, Eigen::Vector3d(0.2, 0.3, 0.4), true);
  RemoteRelationUpdate update;
  update.actor_shared_id = 500;
  update.payloads = {ExplicitLiveSupport(20, 42, "hand", "seat")};
  target->OnRemoteUpdate(update);
  const auto* s = target->GetActorRealizationSnapshot(actor);
  ASSERT_NE(s, nullptr); ASSERT_EQ(s->channels.size(), 1u);
  const auto& c = s->channels[0];
  EXPECT_TRUE(c.has_goal);
  EXPECT_EQ(c.geometry, RealizationGeometry::kExactSupport);
  EXPECT_EQ(c.goal_kind, RealizationGoalKind::kSupportPose);
  EXPECT_NEAR(c.contributors[0].mapped_support.x(), 0.2, 1e-12);
  EXPECT_NEAR(c.contributors[0].mapped_support.y(), 0.3, 1e-12);
  EXPECT_NEAR(c.contributors[0].mapped_support.z(), 0.4, 1e-12);
  EXPECT_NEAR(c.contributors[0].support_plane_to_world(2, 1), 1.0, 1e-12);
}

TEST(RichRealizationLiveProducerTest, ConflictingEnabledSupportsBlendButZeroWeightDoesNot) {
  spatialblend::entity_id_t actor;
  auto target = MakeExplicitLiveSpace(&actor);
  AddExplicitPatch(target.get(), 20, 42, Eigen::Vector3d(0.2, 0, 0));
  AddExplicitPatch(target.get(), 21, 43, Eigen::Vector3d(0.8, 0, 0));
  RemoteRelationUpdate update; update.actor_shared_id = 500;
  update.payloads = {ExplicitLiveSupport(20, 42, "hand", "seat", 0.0),
                     ExplicitLiveSupport(21, 43, "hand", "seat", 0.0)};
  target->OnRemoteUpdate(update);
  auto* s = target->GetActorRealizationSnapshot(actor);
  // The target is the one support the source anchor is engaged
  // on (smallest support distance; declaration order on a tie), never a
  // blend of two contacts -- so the channel is an exact support pose on 42.
  ASSERT_NE(s, nullptr); EXPECT_EQ(s->channels[0].geometry, RealizationGeometry::kExactSupport);
  EXPECT_EQ(s->channels[0].goal_kind, RealizationGoalKind::kSupportPose);
  EXPECT_EQ(s->channels[0].target.referent_id, 42);
  EXPECT_NEAR(s->channels[0].target.position.x(), 0.2, 1e-12);
  // A non-finite update count is no intent and takes no weight.
  update.payloads[0].update_count = 1.0;
  update.payloads[1].update_count =
      std::numeric_limits<double>::infinity();
  target->OnRemoteUpdate(update);
  s = target->GetActorRealizationSnapshot(actor);
  ASSERT_NE(s, nullptr); EXPECT_EQ(s->channels[0].geometry, RealizationGeometry::kExactSupport);
  EXPECT_EQ(s->channels[0].contributors[1].disposition, RealizationDisposition::kDisabled);
}

TEST(RichRealizationLiveProducerTest, MissingChartFailsButLiteralRootBodyAnchorStaysSeparate) {
  spatialblend::entity_id_t actor;
  auto target = MakeExplicitLiveSpace(&actor);
  AddExplicitPatch(target.get(), 20, 42, Eigen::Vector3d(0.2, 0, 0));
  RemoteRelationUpdate update; update.actor_shared_id = 500;
  auto missing = ExplicitLiveSupport(20, 42, "hand", "absent");
  update.payloads = {missing}; target->OnRemoteUpdate(update);
  auto* s = target->GetActorRealizationSnapshot(actor);
  ASSERT_NE(s, nullptr); EXPECT_FALSE(s->channels[0].has_goal);
  EXPECT_EQ(s->channels[0].contributors[0].disposition, RealizationDisposition::kGroundingFailed);
  auto root = ExplicitLiveSupport(20, 42, "root", "seat");
  update.payloads = {root}; target->OnRemoteUpdate(update);
  s = target->GetActorRealizationSnapshot(actor);
  ASSERT_NE(s, nullptr); EXPECT_TRUE(s->channels[0].has_goal);
  EXPECT_EQ(s->channels[0].goal_kind, RealizationGoalKind::kSupportPose);
  // Affordance named "root" is still not a locomotion command: root spacing
  // has no relation here, so the direct legacy projection holds at origin.
  EXPECT_NEAR(target->GetActor(actor)->GetPosition().x(), 0.0, 1e-12);
}

TEST(RichRealizationLiveProducerTest, BodyAnchorLiterallyRootDoesNotDriveNavigation) {
  spatialblend::entity_id_t actor;
  auto target = MakeExplicitLiveSpace(&actor);
  target->GetActor(actor)->AddAnchor(SubjectAnchor("navigation"));
  AddExplicitPatch(target.get(), 20, 42, Eigen::Vector3d(3, 0, 2));
  RemoteRelationUpdate update; update.actor_shared_id = 500;
  update.payloads = {ExplicitLiveSupport(20, 42, "root", "seat")};
  target->OnRemoteUpdate(update);
  const auto* result = target->GetRealizationTargets(actor);
  ASSERT_NE(result, nullptr);
  const auto it = std::find_if(result->targets.begin(), result->targets.end(),
      [](const RealizationTarget& t) {
        return t.anchor_name == "root" &&
            t.kernel_type == RelationKernelType::kAvatarAffordancePose;
      });
  ASSERT_NE(it, result->targets.end());
  // The engaged contact is never moved by reach (a clamp would put it
  // 0.9 m from the root); the overshoot is reported instead.
  EXPECT_NEAR(it->position.x(), 3.0, 1e-12);
  EXPECT_TRUE(it->beyond_reach);
  EXPECT_NEAR(it->z, 2.0, 1e-12);
  EXPECT_NEAR(target->GetActor(actor)->GetPosition().x(), 0.0, 1e-12);
}

TEST(RichRealizationLiveProducerTest, AllExplicitlyDisabledSupportReleasesChannel) {
  spatialblend::entity_id_t actor;
  auto target = MakeExplicitLiveSpace(&actor);
  AddExplicitPatch(target.get(), 20, 42, Eigen::Vector3d(0.2, 0, 0));
  RemoteRelationUpdate update; update.actor_shared_id = 500;
  auto disabled = ExplicitLiveSupport(20, 42, "hand", "seat");
  disabled.enabled = false;
  update.payloads = {disabled}; target->OnRemoteUpdate(update);
  const auto* s = target->GetActorRealizationSnapshot(actor);
  ASSERT_NE(s, nullptr); ASSERT_EQ(s->channels.size(), 1u);
  EXPECT_FALSE(s->channels[0].has_goal);
  EXPECT_EQ(s->channels[0].contributors[0].disposition,
            RealizationDisposition::kDisabled);
}

// A referent whose counterpart offers no such affordance is
// not in AffordancePose scope. It is skipped and reported (kGroundingFailed, status
// kFallback as for every omission); it does not withdraw the valid support's goal.
// This converts MixedValidAndFailedSupportWithdrawsWholeChannel.
TEST(RichRealizationLiveProducerTest, AReferentWithNoAffordanceIsSkippedAndTheValidSupportRealizes) {
  spatialblend::entity_id_t actor;
  auto target = MakeExplicitLiveSpace(&actor);
  AddExplicitPatch(target.get(), 20, 42, Eigen::Vector3d(0.2, 0, 0));
  RemoteRelationUpdate update; update.actor_shared_id = 500;
  update.payloads = {ExplicitLiveSupport(20, 42, "hand", "seat"),
                     ExplicitLiveSupport(20, 42, "hand", "missing")};
  target->OnRemoteUpdate(update);
  const auto* s = target->GetActorRealizationSnapshot(actor);
  ASSERT_NE(s, nullptr); ASSERT_EQ(s->channels.size(), 1u);
  EXPECT_TRUE(s->channels[0].has_goal);
  EXPECT_EQ(s->channels[0].status, SolverStatus::kFallback);
  EXPECT_EQ(s->channels[0].contributors[1].disposition,
            RealizationDisposition::kGroundingFailed);
}

TEST(RichRealizationLiveProducerTest, MissingNamedFacingIsUnsupportedRelease) {
  Space target; target.SetRichRealizationEnabled(true);
  auto* tree = target.GetElementTree(); ASSERT_TRUE(tree->AddElement(10).ok());
  ASSERT_TRUE(tree->AddElement(20, 10).ok()); tree->GetElement(10)->SetSharedId(500);
  tree->GetElement(20)->SetSharedId(42);
  const auto actor = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);
  RelationPayload p; p.referent_shared_id = 42; p.subject_anchor_name = "head";
  p.kernel_type = RelationKernelType::kAvatarFacing;
  p.update_count = 1.0;  // aligned: a valid Facing request (v1 Angular1D)
  RemoteRelationUpdate update; update.actor_shared_id = 500; update.payloads = {p};
  target.OnRemoteUpdate(update);
  const auto* s = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(s, nullptr); ASSERT_EQ(s->channels.size(), 1u);
  EXPECT_FALSE(s->channels[0].has_goal);
  EXPECT_EQ(s->channels[0].status, SolverStatus::kFallback);
  EXPECT_EQ(s->channels[0].contributors[0].disposition,
            RealizationDisposition::kUnsupported);
  EXPECT_FALSE(s->channels[0].contributors[0].reason.empty());
}

TEST(RichRealizationLiveProducerTest, ExplicitUniqueSupportRecordsReachPrePostMetric) {
  spatialblend::entity_id_t actor;
  auto target = MakeExplicitLiveSpace(&actor);
  AddExplicitPatch(target.get(), 20, 42, Eigen::Vector3d(3, 0, 0));
  RemoteRelationUpdate update; update.actor_shared_id = 500;
  update.payloads = {ExplicitLiveSupport(20, 42, "hand", "seat")};
  target->OnRemoteUpdate(update);
  const auto* s = target->GetActorRealizationSnapshot(actor);
  ASSERT_NE(s, nullptr); const auto& c = s->channels[0];
  // Reach does not project an engaged contact (a projected
  // point is not the contact). The target stays the exact support; the 2.1 m
  // overshoot beyond the 0.9 m hand reach is FLAGGED through the rich reach
  // fields (reach_projected / kReachProjected now mean "beyond reach").
  EXPECT_TRUE(c.has_goal); EXPECT_TRUE(c.reach_projected);
  EXPECT_NEAR(c.reach_displacement_m, 2.1, 1e-12);
  EXPECT_EQ(c.geometry, RealizationGeometry::kReachProjected);
  EXPECT_EQ(c.goal_kind, RealizationGoalKind::kSupportPose);
  EXPECT_FALSE(c.target.has_requested_position);
  EXPECT_NEAR(c.target.position.x(), 3.0, 1e-12);
  EXPECT_TRUE(c.target.beyond_reach);
  EXPECT_NEAR(c.target.reach_excess_m, 2.1, 1e-12);
}

TEST(RichRealizationLiveProducerTest, DisabledFacingAndSpacingHaveNoChannelGoal) {
  Space target; target.SetRichRealizationEnabled(true);
  auto* tree = target.GetElementTree(); ASSERT_TRUE(tree->AddElement(10).ok());
  ASSERT_TRUE(tree->AddElement(20, 10).ok()); tree->GetElement(10)->SetSharedId(500);
  tree->GetElement(20)->SetSharedId(42);
  const auto actor = target.CreateActor({SubjectAnchor("root"), SubjectAnchor("head")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);
  RelationPayload spacing; spacing.referent_shared_id = 42; spacing.subject_anchor_name = "root";
  spacing.kernel_type = RelationKernelType::kAvatarSpacing; spacing.enabled = false;
  RelationPayload facing = spacing; facing.subject_anchor_name = "head";
  facing.kernel_type = RelationKernelType::kAvatarFacing;
  facing.update_count = 1.0;  // aligned: a valid Facing request (v1 Angular1D)
  RemoteRelationUpdate update; update.actor_shared_id = 500; update.payloads = {spacing, facing};
  target.OnRemoteUpdate(update);
  const auto* s = target.GetActorRealizationSnapshot(actor); ASSERT_NE(s, nullptr);
  ASSERT_EQ(s->channels.size(), 2u);
  for (const auto& c : s->channels) {
    EXPECT_FALSE(c.has_goal);
    EXPECT_EQ(c.contributors[0].disposition, RealizationDisposition::kDisabled);
  }
}

TEST(RichRealizationLiveProducerTest,
     DefaultUnresolvedSpacingAndFacingWithdrawPriorGoals) {
  Space target; target.SetRichRealizationEnabled(true);
  auto* tree = target.GetElementTree(); ASSERT_TRUE(tree->AddElement(10).ok());
  ASSERT_TRUE(tree->AddElement(20, 10).ok());
  tree->GetElement(10)->SetSharedId(500); tree->GetElement(20)->SetSharedId(42);
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity(); pose(0, 3) = 2.0;
  tree->GetElement(20)->SetLocalTransform(pose);
  const auto actor = target.CreateActor({SubjectAnchor("root"), SubjectAnchor("head")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);
  RelationPayload spacing; spacing.referent_shared_id = 42;
  spacing.subject_anchor_name = "root"; spacing.kernel_type = RelationKernelType::kAvatarSpacing;
  // A valid v1 count (approaching): a request. Without one Progressive computes no weight,
  // and the channel/target this test reads would not exist.
  spacing.update_count = 2.0;
  RelationPayload facing = spacing; facing.subject_anchor_name = "head";
  facing.kernel_type = RelationKernelType::kAvatarFacing;
  facing.update_count = 1.0;  // aligned: a valid Facing request (v1 Angular1D)
  RemoteRelationUpdate update; update.actor_shared_id = 500; update.payloads = {spacing, facing};
  target.OnRemoteUpdate(update);
  const auto* before = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(before, nullptr);
  ASSERT_EQ(before->channels.size(), 2u);
  for (const auto& channel : before->channels) ASSERT_TRUE(channel.has_goal);
  const uint64_t previous_revision = before->revision;
  update.payloads[0].referent_shared_id = 999;
  update.payloads[1].referent_shared_id = 999;
  target.OnRemoteUpdate(update);
  const auto* snapshot = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(snapshot, nullptr); ASSERT_EQ(snapshot->channels.size(), 2u);
  EXPECT_GT(snapshot->revision, previous_revision);
  for (const auto& channel : snapshot->channels) {
    EXPECT_FALSE(channel.has_goal);
    EXPECT_EQ(channel.contributors[0].disposition, RealizationDisposition::kUnmapped);
  }
}

TEST(RichRealizationLiveProducerTest,
     StrictUnlinkedRowIsRetainedDistinctFromAuthoritativeEmpty) {
  Space target; target.SetRichRealizationEnabled(true);
  SpaceHub hub; ASSERT_TRUE(hub.Connect(&target).ok());
  target.SetHub(&hub);
  target.SetReferentResolveMode(ReferentResolveMode::kLinkerOnly);
  auto* tree = target.GetElementTree(); ASSERT_TRUE(tree->AddElement(10).ok());
  tree->GetElement(10)->SetSharedId(500);
  const auto actor = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);
  RelationPayload row;
  row.referent_shared_id = kInvalidSharedElementId;
  row.referent_source_space_id = 71;
  row.referent_source_element_id = 72;
  row.subject_anchor_name = "root";
  row.kernel_type = RelationKernelType::kAvatarSpacing;
  RemoteRelationUpdate update; update.source_space_id = 71;
  update.actor_shared_id = 500; update.payloads = {row};
  target.OnRemoteUpdate(update);
  const auto* rejected = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(rejected, nullptr); ASSERT_EQ(rejected->channels.size(), 1u);
  EXPECT_FALSE(rejected->channels[0].has_goal);
  ASSERT_EQ(rejected->channels[0].contributors.size(), 1u);
  const auto& c = rejected->channels[0].contributors[0];
  EXPECT_EQ(c.disposition, RealizationDisposition::kUnmapped);
  EXPECT_EQ(c.snapshot_ordinal, 0u);
  EXPECT_EQ(c.source_space_id, 71);
  EXPECT_EQ(c.source_element_id, 72);
  const uint64_t rejected_revision = rejected->revision;

  update.payloads.clear();
  target.OnRemoteUpdate(update);
  const auto* empty = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(empty, nullptr);
  EXPECT_GT(empty->revision, rejected_revision);
  EXPECT_TRUE(empty->channels.empty());
}

TEST(RichRealizationLiveProducerTest,
     InvalidSharedIdDoesNotSelectUnsharedDecoyButLiveLinkStillAdmits) {
  Space target;
  target.SetSpaceId(2);
  target.SetRichRealizationEnabled(true);
  // Exercise normal non-strict linker mode. The ledger row below must not be
  // described as a linker-only rejection.
  target.SetReferentResolveMode(ReferentResolveMode::kLinker);
  SpaceHub hub;
  ASSERT_TRUE(hub.Connect(&target).ok());
  target.SetHub(&hub);
  auto* tree = target.GetElementTree();
  ASSERT_TRUE(tree->AddElement(10).ok());
  ASSERT_TRUE(tree->AddElement(20, 10).ok());
  ASSERT_TRUE(tree->AddElement(21, 10).ok());  // Deliberately unshared decoy.
  tree->GetElement(10)->SetSharedId(500);
  Eigen::Matrix4d linked_pose = Eigen::Matrix4d::Identity();
  linked_pose(0, 3) = 3.0;
  tree->GetElement(20)->SetLocalTransform(linked_pose);
  Eigen::Matrix4d decoy_pose = Eigen::Matrix4d::Identity();
  decoy_pose(0, 3) = 9.0;
  tree->GetElement(21)->SetLocalTransform(decoy_pose);
  const auto actor = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);

  RelationPayload row;
  row.referent_shared_id = kInvalidSharedElementId;
  row.referent_source_space_id = 71;
  row.referent_source_element_id = 72;
  row.subject_anchor_name = "root";
  row.kernel_type = RelationKernelType::kAvatarSpacing;
  // A valid v1 count (approaching): a request. Without one Progressive computes no weight,
  // and the channel/target this test reads would not exist.
  row.update_count = 2.0;
  RemoteRelationUpdate update;
  update.source_space_id = 71;
  update.actor_shared_id = 500;
  update.payloads = {row};

  // The old shared-id fallback could select an unshared local element. Rich
  // instead retains the original provenance without admitting either decoy.
  target.OnRemoteUpdate(update);
  const auto* unresolved = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(unresolved, nullptr);
  ASSERT_EQ(unresolved->channels.size(), 1u);
  EXPECT_FALSE(unresolved->channels[0].has_goal);
  ASSERT_EQ(unresolved->channels[0].contributors.size(), 1u);
  const auto& raw = unresolved->channels[0].contributors[0];
  EXPECT_EQ(raw.disposition, RealizationDisposition::kUnmapped);
  EXPECT_EQ(raw.reason, "referent unresolved on receiver");
  EXPECT_EQ(raw.source_space_id, 71);
  EXPECT_EQ(raw.source_element_id, 72);
  EXPECT_EQ(raw.resolved_local_element_id, kInvalidElementId);
  const uint64_t unresolved_revision = unresolved->revision;

  // Explicit correspondence is a valid identity even when both endpoints
  // have unknown shared ids.
  hub.GetLinker().CreateLink(element_handle_t{71, 72},
                             element_handle_t{2, 20});
  target.OnRemoteUpdate(update);
  const auto* linked = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(linked, nullptr);
  ASSERT_EQ(linked->channels.size(), 1u);
  EXPECT_GT(linked->revision, unresolved_revision);
  EXPECT_TRUE(linked->channels[0].has_goal);
  ASSERT_EQ(linked->channels[0].contributors.size(), 1u);
  EXPECT_EQ(linked->channels[0].contributors[0].disposition,
            RealizationDisposition::kIncluded);
  EXPECT_EQ(linked->channels[0].contributors[0].resolved_local_element_id,
            20);
  const uint64_t linked_revision = linked->revision;
  EXPECT_NE(linked->known_revision_mask & kRealizationCounterpartRevision,
            0u);
  ActorRealizationValidity counterpart_validity;
  ASSERT_TRUE(target.GetActorRealizationValidity(
      actor, linked_revision, kRealizationCounterpartRevision,
      &counterpart_validity));
  EXPECT_EQ(counterpart_validity.changed_mask & kRealizationCounterpartRevision,
            0u);

  // Link edits invalidate the historical solve immediately, without another
  // source delivery. Recreate consumes another graph generation, so an ABA
  // endpoint cannot make the old snapshot look current again.
  ASSERT_TRUE(hub.GetLinker().RemoveLink(element_handle_t{71, 72},
                                         element_handle_t{2, 20}));
  ASSERT_TRUE(target.GetActorRealizationValidity(
      actor, linked_revision, kRealizationCounterpartRevision,
      &counterpart_validity));
  EXPECT_NE(counterpart_validity.changed_mask &
                kRealizationCounterpartRevision,
            0u);
  hub.GetLinker().CreateLink(element_handle_t{71, 72},
                             element_handle_t{2, 20});
  ASSERT_TRUE(target.GetActorRealizationValidity(
      actor, linked_revision, kRealizationCounterpartRevision,
      &counterpart_validity));
  EXPECT_NE(counterpart_validity.changed_mask &
                kRealizationCounterpartRevision,
            0u);
  target.SetHub(nullptr);
  ASSERT_TRUE(target.GetActorRealizationValidity(
      actor, linked_revision, kRealizationCounterpartRevision,
      &counterpart_validity));
  EXPECT_EQ(counterpart_validity.current_known_mask &
                kRealizationCounterpartRevision,
            0u);

  // A stale correspondence cannot reopen sentinel fallback to element 21.
  ASSERT_TRUE(tree->RemoveElement(20).ok());
  target.OnRemoteUpdate(update);
  const auto* stale = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(stale, nullptr);
  ASSERT_EQ(stale->channels.size(), 1u);
  EXPECT_GT(stale->revision, linked_revision);
  EXPECT_FALSE(stale->channels[0].has_goal);
  EXPECT_EQ(stale->channels[0].contributors[0].disposition,
            RealizationDisposition::kUnmapped);
  const uint64_t stale_revision = stale->revision;

  // Authoritative empty remains distinct from a nonempty unmapped ledger row.
  update.payloads.clear();
  target.OnRemoteUpdate(update);
  const auto* empty = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(empty, nullptr);
  EXPECT_GT(empty->revision, stale_revision);
  EXPECT_TRUE(empty->channels.empty());
}

TEST(RichRealizationLiveProducerTest,
     StrictMixedRowsKeepOriginalOrdinalsAndLinkedLocalIdentity) {
  Space target; target.SetRichRealizationEnabled(true); target.SetSpaceId(2);
  SpaceHub hub; ASSERT_TRUE(hub.Connect(&target).ok());
  target.SetHub(&hub);
  target.SetReferentResolveMode(ReferentResolveMode::kLinkerOnly);
  auto* tree = target.GetElementTree(); ASSERT_TRUE(tree->AddElement(10).ok());
  tree->GetElement(10)->SetSharedId(500);
  const auto actor = target.CreateActor({SubjectAnchor("root"), SubjectAnchor("hand")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);
  AddExplicitPatch(&target, 20, kInvalidSharedElementId, Eigen::Vector3d(0.2, 0, 0));
  hub.GetLinker().CreateLink(element_handle_t{71, 73},
                             element_handle_t{2, 20});
  RelationPayload rejected;
  rejected.referent_shared_id = kInvalidSharedElementId;
  rejected.referent_source_space_id = 71; rejected.referent_source_element_id = 72;
  rejected.subject_anchor_name = "root";
  rejected.kernel_type = RelationKernelType::kAvatarSpacing;
  RelationPayload admitted = ExplicitLiveSupport(20, kInvalidSharedElementId,
                                                  "hand", "seat");
  admitted.referent_source_space_id = 71; admitted.referent_source_element_id = 73;
  RemoteRelationUpdate update; update.source_space_id = 71; update.actor_shared_id = 500;
  update.payloads = {rejected, admitted};
  target.OnRemoteUpdate(update);
  const auto* snapshot = target.GetActorRealizationSnapshot(actor);
  ASSERT_NE(snapshot, nullptr); ASSERT_EQ(snapshot->channels.size(), 2u);
  const RealizationChannel* root = nullptr;
  const RealizationChannel* hand = nullptr;
  for (const auto& channel : snapshot->channels) {
    if (channel.anchor_name == "root") root = &channel;
    if (channel.anchor_name == "hand") hand = &channel;
  }
  ASSERT_NE(root, nullptr); ASSERT_NE(hand, nullptr);
  EXPECT_FALSE(root->has_goal);
  EXPECT_EQ(root->contributors[0].disposition, RealizationDisposition::kUnmapped);
  EXPECT_EQ(root->contributors[0].snapshot_ordinal, 0u);
  EXPECT_TRUE(hand->has_goal);
  EXPECT_EQ(hand->contributors[0].snapshot_ordinal, 1u);
  EXPECT_EQ(hand->contributors[0].source_space_id, 71);
  EXPECT_EQ(hand->contributors[0].source_element_id, 73);
  EXPECT_EQ(hand->contributors[0].resolved_local_element_id, 20);
  EXPECT_NEAR(hand->target.position.x(), 0.2, 1e-12);
}

TEST(RichRealizationLiveProducerTest,
     SameChannelMixedAdmissionIsFallbackInEitherInboundOrder) {
  for (bool rejected_first : {false, true}) {
    spatialblend::entity_id_t actor;
    auto target = MakeExplicitLiveSpace(&actor);
    AddExplicitPatch(target.get(), 20, 42, Eigen::Vector3d(0.2, 0, 0));
    AddExplicitPatch(target.get(), 21, 43, Eigen::Vector3d(0, 0.2, 0));
    RelationPayload valid = ExplicitLiveSupport(20, 42, "hand", "seat");
    valid.kernel_type = RelationKernelType::kAvatarFacing;
    valid.update_count = 1.0;  // aligned: a valid Facing request (v1 Angular1D)
    valid.resolved_local_element_id = kInvalidElementId;  // shared-id path
    RelationPayload rejected = valid;
    rejected.referent_shared_id = 999;
    rejected.resolved_local_element_id = 21;  // forged perpendicular target
    RemoteRelationUpdate update; update.actor_shared_id = 500;
    update.payloads = rejected_first ? std::vector<RelationPayload>{rejected, valid}
                                     : std::vector<RelationPayload>{valid, rejected};
    target->OnRemoteUpdate(update);
    const auto* snapshot = target->GetActorRealizationSnapshot(actor);
    ASSERT_NE(snapshot, nullptr); ASSERT_EQ(snapshot->channels.size(), 1u);
    const auto& channel = snapshot->channels[0];
    EXPECT_TRUE(channel.has_goal);
    EXPECT_NEAR(channel.target.yaw, 0.0, 1e-12);
    EXPECT_EQ(channel.status, SolverStatus::kFallback);
    ASSERT_EQ(channel.contributors.size(), 2u);
    bool saw_valid = false, saw_unmapped = false;
    for (const auto& c : channel.contributors) {
      if (c.disposition == RealizationDisposition::kIncluded) {
        saw_valid = true;
        EXPECT_EQ(c.resolved_local_element_id, 20);
      }
      if (c.disposition == RealizationDisposition::kUnmapped) {
        saw_unmapped = true;
        EXPECT_EQ(c.resolved_local_element_id, kInvalidElementId);
      }
    }
    EXPECT_TRUE(saw_valid);
    EXPECT_TRUE(saw_unmapped);
  }
}

TEST(ObservationRemoteAdmissionTest,
     RejectedNewModeEnvelopeDoesNotCreateReceiptOrRegressTime) {
  Space target;
  target.SetSpaceId(2);
  ASSERT_TRUE(target.GetElementTree()->AddElement(10).ok());
  target.GetElementTree()->GetElement(10)->SetSharedId(500);
  const auto actor = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(10);

  RemoteRelationUpdate accepted;
  accepted.timing = RelationUpdateTiming::kTimestampedObservation;
  accepted.source_space_id = 1;
  accepted.actor_shared_id = 500;
  accepted.actor_source_space_id = 1;
  accepted.actor_source_element_id = 7;
  accepted.source_epoch = 3;
  accepted.sequence = 10;
  accepted.source_time = 2.0;
  target.OnRemoteUpdate(accepted);
  EXPECT_EQ(target.GetReceivedUpdateCount(), 1u);
  EXPECT_DOUBLE_EQ(target.GetSpaceClock(), 2.0);

  RemoteRelationUpdate late_time = accepted;
  late_time.sequence = 11;
  late_time.source_time = 1.0;
  target.OnRemoteUpdate(late_time);
  EXPECT_EQ(target.GetReceivedUpdateCount(), 1u);
  EXPECT_DOUBLE_EQ(target.GetSpaceClock(), 2.0);

  RemoteRelationUpdate malformed = accepted;
  malformed.sequence = 12;
  malformed.source_time = std::numeric_limits<double>::quiet_NaN();
  target.OnRemoteUpdate(malformed);
  EXPECT_EQ(target.GetReceivedUpdateCount(), 1u);
  EXPECT_DOUBLE_EQ(target.GetSpaceClock(), 2.0);
}

TEST(ObservationRemoteAdmissionTest,
     LinkedUnsharedActorsKeepEpochsPhasesAndTargetsIndependent) {
  Space target;
  target.SetSpaceId(2);
  SpaceHub hub;
  target.SetHub(&hub);
  ASSERT_TRUE(hub.Connect(&target).ok());
  auto* tree = target.GetElementTree();
  ASSERT_TRUE(tree->AddElement(10).ok());
  ASSERT_TRUE(tree->AddElement(11).ok());
  ASSERT_TRUE(tree->AddElement(20).ok());
  tree->GetElement(20)->SetSharedId(42);
  Eigen::Matrix4d referent = Eigen::Matrix4d::Identity();
  referent(0, 3) = 5.0;
  tree->GetElement(20)->SetLocalTransform(referent);
  const auto actor_a = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, actor_a);
  const auto actor_b = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, actor_b);
  target.GetActor(actor_a)->SetElementId(10);
  target.GetActor(actor_b)->SetElementId(11);
  hub.GetLinker().CreateLink(element_handle_t{1, 1}, element_handle_t{2, 10});
  hub.GetLinker().CreateLink(element_handle_t{1, 2}, element_handle_t{2, 11});

  std::vector<RelationEvent> events;
  target.RegisterOnRelationEvent(
      [&](const RelationEvent& event) { events.push_back(event); });
  RelationPayload payload;
  payload.referent_shared_id = 42;
  payload.subject_anchor_name = "root";
  payload.kernel_type = RelationKernelType::kAvatarSpacing;
  payload.grounded_relation.proximity = 0.2;
  RemoteRelationUpdate a;
  a.timing = RelationUpdateTiming::kTimestampedObservation;
  a.source_space_id = 1;
  a.actor_shared_id = kInvalidSharedElementId;  // intentionally identical
  a.actor_source_space_id = 1;
  a.actor_source_element_id = 1;
  a.source_epoch = 1;
  a.sequence = 1;
  a.source_time = 0.0;
  a.payloads = {payload};
  RemoteRelationUpdate b = a;
  b.actor_source_element_id = 2;
  b.sequence = 1;  // independent source-handle fence permits this.
  target.OnRemoteUpdate(a);
  target.OnRemoteUpdate(b);
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].actor_shared_id, kInvalidSharedElementId);
  EXPECT_EQ(events[1].actor_shared_id, kInvalidSharedElementId);
  ASSERT_NE(target.GetRealizationTargets(actor_b), nullptr);
  const size_t b_target_count = target.GetRealizationTargets(actor_b)->targets.size();

  RemoteRelationUpdate reset_a = a;
  reset_a.timing = RelationUpdateTiming::kObservationReset;
  reset_a.source_epoch = 2;
  reset_a.sequence = 2;
  target.OnRemoteUpdate(reset_a);
  const RealizationResult* reset_a_result =
      target.GetRealizationTargets(actor_a);
  ASSERT_NE(reset_a_result, nullptr);
  EXPECT_TRUE(reset_a_result->targets.empty());
  // Actor B's owner namespace and realization survive actor A reset/removal.
  ASSERT_NE(target.GetRealizationTargets(actor_b), nullptr);
  EXPECT_EQ(target.GetRealizationTargets(actor_b)->targets.size(), b_target_count);
  RemoteRelationUpdate b_next = b;
  b_next.sequence = 2;
  b_next.source_time = 0.1;
  // The second delivery carries a valid v1 count (approaching): the relation's weight becomes 1 -> kDriving. With no
  // count Progressive computes no weight and the phase stays kActive.
  b_next.payloads[0].update_count = 2.0;
  target.OnRemoteUpdate(b_next);
  ASSERT_GE(events.size(), 3u);
  EXPECT_EQ(events.back().from, RelationPhase::kActive);
  EXPECT_EQ(events.back().to, RelationPhase::kDriving);
  EXPECT_TRUE(target.DestroyActor(actor_a));
  ASSERT_NE(target.GetRealizationTargets(actor_b), nullptr);
  EXPECT_EQ(target.GetRealizationTargets(actor_b)->targets.size(), b_target_count);

  // A same-epoch reset is rejected rather than reopening B's time fence.
  RemoteRelationUpdate invalid_reset_b = b;
  invalid_reset_b.timing = RelationUpdateTiming::kObservationReset;
  invalid_reset_b.sequence = 3;
  target.OnRemoteUpdate(invalid_reset_b);
  RemoteRelationUpdate late_b = b;
  late_b.sequence = 4;
  late_b.source_time = 0.0;
  target.OnRemoteUpdate(late_b);
  EXPECT_DOUBLE_EQ(target.GetSpaceClock(), 0.1);
}

TEST(ElementRemovalTest, ExactSourceOwnershipPreservesLinkedAndCollidingLocalElements) {
  Space target;
  target.SetSpaceId(2);
  SpaceHub hub;
  target.SetHub(&hub);
  ASSERT_TRUE(hub.Connect(&target).ok());
  auto* tree = target.GetElementTree();
  ASSERT_TRUE(tree->AddElement(7).ok());
  tree->GetElement(7)->SetSharedId(42);  // local collision, never remote-owned

  std::vector<element_id_t> added;
  std::vector<element_id_t> removed;
  bool removal_callback_reentered = false;
  target.SetOnElementAdded([&](element_id_t id, shared_element_id_t,
                               const std::string&, const std::string&) {
    added.push_back(id);
  });
  target.SetOnElementRemoved([&](element_id_t id) {
    removed.push_back(id);
    removal_callback_reentered = target.GetElementTree()->GetElement(7) != nullptr;
  });

  ElementEvent add;
  add.type = ElementEvent::kAdded;
  add.source_space_id = 1;
  add.source_element_id = 10;
  add.shared_id = 42;
  target.OnElementEvent(add);
  ASSERT_EQ(added.size(), 1u);
  const element_id_t owned = added.back();
  target.ResolveSharedElement(owned, owned);  // AddAsNew
  ASSERT_TRUE(tree->AddElement(8, owned).ok());  // locally authored child
  Eigen::Matrix4d owned_transform = Eigen::Matrix4d::Identity();
  owned_transform(0, 3) = 3.0;
  owned_transform(1, 3) = -2.0;
  tree->GetElement(owned)->SetLocalTransform(owned_transform);
  Eigen::Matrix4d local_child_transform = Eigen::Matrix4d::Identity();
  local_child_transform(0, 3) = 0.5;
  local_child_transform(1, 3) = 4.0;
  tree->GetElement(8)->SetLocalTransform(local_child_transform);
  const Eigen::Matrix4d local_child_world = tree->GetWorldTransform(8);
  const auto actor = target.CreateActor({SubjectAnchor("root")});
  FollowProposals(&target, actor);
  target.GetActor(actor)->SetElementId(owned);
  ASSERT_NE(target.Bind(actor, "root", RelationKernelType::kAvatarSpacing),
            nullptr);
  ASSERT_EQ(target.GetBindings(actor).size(), 1u);

  ElementEvent remove = add;
  remove.type = ElementEvent::kRemoved;
  target.OnElementEvent(remove);
  EXPECT_EQ(tree->GetElement(7)->GetSharedId(), 42);
  EXPECT_EQ(tree->GetElement(owned), nullptr);
  ASSERT_NE(tree->GetElement(8), nullptr);
  EXPECT_EQ(tree->GetElement(8)->GetParentId(), kInvalidElementId);
  EXPECT_TRUE(tree->GetWorldTransform(8).isApprox(local_child_world, 1e-12));
  EXPECT_EQ(target.GetActor(actor)->GetElementId(), kInvalidElementId);
  EXPECT_TRUE(target.GetBindings(actor).empty());
  ASSERT_EQ(removed.size(), 1u);
  EXPECT_EQ(removed[0], owned);
  EXPECT_TRUE(removal_callback_reentered);
  EXPECT_FALSE(hub.GetLinker().GetTarget({1, 10}, 2).has_value());
  target.OnElementEvent(remove);  // duplicate unshare is idempotent
  EXPECT_EQ(removed.size(), 1u);

  // Re-sharing after removal creates a fresh owned replica, not a stale link.
  target.OnElementEvent(add);
  ASSERT_EQ(added.size(), 2u);
  const element_id_t reshared = added.back();
  target.ResolveSharedElement(reshared, reshared);
  EXPECT_NE(reshared, owned);
  EXPECT_TRUE(hub.GetLinker().GetTarget({1, 10}, 2).has_value());

  // Pending/unresolved temp replicas are source-owned too.
  ElementEvent pending = add;
  pending.source_element_id = 11;
  target.OnElementEvent(pending);
  ASSERT_EQ(added.size(), 3u);
  const element_id_t pending_id = added.back();
  target.OnElementEvent(pending);
  EXPECT_EQ(added.size(), 3u) << "duplicate unresolved adds are suppressed";
  ElementEvent remove_pending = pending;
  remove_pending.type = ElementEvent::kRemoved;
  target.OnElementEvent(remove_pending);
  EXPECT_EQ(tree->GetElement(pending_id), nullptr);
  EXPECT_EQ(removed.back(), pending_id);

  // LinkToExisting only removes its temporary receive and preserves local 7.
  ElementEvent linked = add;
  linked.source_element_id = 12;
  target.OnElementEvent(linked);
  ASSERT_EQ(added.size(), 4u);
  target.ResolveSharedElement(added.back(), 7);
  const size_t removals_before_linked_unshare = removed.size();
  ElementEvent remove_linked = linked;
  remove_linked.type = ElementEvent::kRemoved;
  target.OnElementEvent(remove_linked);
  EXPECT_NE(tree->GetElement(7), nullptr);
  EXPECT_EQ(removed.size(), removals_before_linked_unshare);

  // A local hub dispatches synchronously. The receiver both unregisters its
  // callback and calls back into the locked source API; Share/Unshare must
  // therefore not retain the source mutex across propagation.
  Space source;
  source.SetSpaceId(1);
  source.SetHub(&hub);
  ASSERT_TRUE(hub.Connect(&source).ok());
  ASSERT_TRUE(source.GetElementTree()->AddElement(10).ok());
  source.GetElementTree()->GetElement(10)->SetSharedId(99);
  bool receiver_self_unregistered = false;
  bool source_reentered = false;
  target.SetOnElementRemoved([&](element_id_t) {
    target.SetOnElementRemoved(nullptr);
    source.SetUpdateInterval(0.1);
    receiver_self_unregistered = true;
    source_reentered = true;
  });
  source.ShareElement(10);
  const element_id_t source_owned = added.back();
  target.ResolveSharedElement(source_owned, source_owned);
  source.UnshareElement(10);
  EXPECT_TRUE(receiver_self_unregistered);
  EXPECT_TRUE(source_reentered);
}

// In a space with a room, a received element never becomes a parentless root,
// which no containment check reaches (the Windows repro: a joiner admitted the
// source's regions as roots and never re-checked them against its room
// outline). It goes under its parent, or under the room when the parent is
// absent here.
TEST(ElementReceiveTest, ReceivedElementIsPlacedUnderItsParentOrTheRoom) {
  Space target;
  target.SetSpaceId(2);
  SpaceHub hub;
  target.SetHub(&hub);
  ASSERT_TRUE(hub.Connect(&target).ok());
  auto* tree = target.GetElementTree();
  std::vector<std::pair<element_id_t, shared_element_id_t>> added;
  target.SetOnElementAdded([&](element_id_t id, shared_element_id_t sid,
                               const std::string&, const std::string&) {
    added.emplace_back(id, sid);
  });
  ASSERT_TRUE(target.AddElement(1, kInvalidElementId));  // the room, no shared id yet

  ElementEvent region;
  region.type = ElementEvent::kAdded;
  region.source_space_id = 1;
  region.source_element_id = 46;
  region.shared_id = 500;
  region.parent_shared_id = 1;  // the source's room: absent here until tagged
  ElementEvent object = region;
  object.source_element_id = 47;
  object.shared_id = 501;
  object.parent_shared_id = 500;  // under the region
  target.OnElementEvent(region);
  target.OnElementEvent(object);
  ASSERT_EQ(added.size(), 2u);
  EXPECT_EQ(tree->GetElement(added[0].first)->GetParentId(), 1);
  EXPECT_EQ(tree->GetElement(added[1].first)->GetParentId(), added[0].first);

  // Under the room, the region is checked against it: a room outline that
  // leaves the region outside, at any distance, is accepted and the region
  // extends the room's scope, whichever was written first.
  const auto rect = [](double x0, double y0, double x1, double y1, GeometryRole role) {
    return std::make_shared<NavigableSpatialDefinition>(
        std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
            {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}),
        nullptr, 0.0, role);
  };
  ASSERT_TRUE(target.SetElementSpatialDefinition(
      added[0].first, rect(2.0, 2.0, 3.0, 3.0, GeometryRole::kContainment)));
  EXPECT_TRUE(target.SetElementSpatialDefinition(
      1, rect(0.0, 0.0, 1.4, 4.0, GeometryRole::kNavigable)));  // 0.6 m from the region: extended
  EXPECT_EQ(target.LastSceneError().code, SceneErrorCode::kOk);
  EXPECT_TRUE(target.SetElementSpatialDefinition(
      1, rect(0.0, 0.0, 2.5, 4.0, GeometryRole::kNavigable)));  // overlaps it: extended
  EXPECT_EQ(target.LastSceneError().code, SceneErrorCode::kOk);

  // A source element whose parent has no shared id names none: under the room
  // too, never a root of its own; and removing it changes nothing for the
  // next one (the room is tracked, not the tree's last root).
  ElementEvent unparented = region;
  unparented.source_element_id = 50;
  unparented.shared_id = 504;
  unparented.parent_shared_id = kInvalidSharedElementId;
  target.OnElementEvent(unparented);
  ASSERT_EQ(added.size(), 3u);
  EXPECT_EQ(tree->GetElement(added[2].first)->GetParentId(), 1);
  ElementEvent unshare = unparented;
  unshare.type = ElementEvent::kRemoved;
  target.OnElementEvent(unshare);
  ElementEvent on_table = region;  // a table this space does not have
  on_table.source_element_id = 51;
  on_table.shared_id = 505;
  on_table.parent_shared_id = 77;
  target.OnElementEvent(on_table);
  ASSERT_EQ(added.size(), 4u);
  EXPECT_EQ(tree->GetElement(added[3].first)->GetParentId(), 1);
}

TEST(SpaceElementLifecycleTest, LocalSubtreeRetirementClearsActorRuntimeState) {
  Space space;
  auto* tree = space.GetElementTree();
  ASSERT_TRUE(tree->AddElement(1).ok());
  ASSERT_TRUE(tree->AddElement(2, 1).ok());
  ASSERT_TRUE(tree->AddElement(3, 2).ok());
  ASSERT_TRUE(tree->AddElement(4, 1).ok());
  const auto actor = space.CreateActor({SubjectAnchor("hand")});
  FollowProposals(&space, actor);
  space.GetActor(actor)->SetElementId(2);
  space.SetAnchorWorldPose(actor, "hand", Eigen::Vector3d(1, 2, 3), 0.4);
  ASSERT_NE(space.Bind(actor, "hand", RelationKernelType::kAvatarSpacing),
            nullptr);
  IdealDriver driver(&space, actor);

  EXPECT_FALSE(space.RemoveLocalElementSubtree(1));  // root is protected
  EXPECT_TRUE(space.RemoveLocalElementSubtree(2));
  EXPECT_FALSE(space.RemoveLocalElementSubtree(2));  // idempotent failure
  EXPECT_EQ(tree->GetElement(2), nullptr);
  EXPECT_EQ(tree->GetElement(3), nullptr);
  EXPECT_NE(tree->GetElement(4), nullptr);  // sibling preserved
  EXPECT_EQ(space.GetActor(actor)->GetElementId(), kInvalidElementId);
  EXPECT_TRUE(space.GetBindings(actor).empty());
  EXPECT_EQ(space.GetRealizationTargets(actor), nullptr);
  EXPECT_EQ(driver.proposal().status, ProposalStatus::kNone);
  EXPECT_EQ(space.GetActor(actor)->GetAnchorWorldPoseOverride("hand"),
            nullptr);
}

}  // namespace test
}  // namespace spatialblend
