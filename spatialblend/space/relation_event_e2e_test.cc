// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/relation_event_e2e_test.cc
//
// End-to-end test for relation lifecycle events:
// as an actor engages an affordance, the Space emits phase-transition events
// through the registered callback (kDormant → kActive → ... → kContact).

#include <gtest/gtest.h>

#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/relation_event.h"

namespace spatialblend {
namespace test {

class RelationEventE2ETest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = LocalContext::Create();
    space_ = context_->AddSpace();
    auto* tree = space_->GetElementTree();

    // Room (1): 10x10 navigable area.
    tree->AddElement(1);
    tree->GetElement(1)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(
            std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
                {0, 0}, {10, 0}, {10, 10}, {0, 10}})));

    // Affordance (2): a small "Seat" patch near (0.5, 0.5).
    tree->AddElement(2, 1);
    tree->GetElement(2)->SetTag("Seat");
    tree->GetElement(2)->SetSharedId(7001);  // MA-1 / SP-D7-03: an unshared affordance is a referent of no relation
    tree->GetElement(2)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(
            std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
                {0, 0}, {1, 0}, {1, 1}, {0, 1}})));

    std::vector<SubjectAnchor> anchors = {SubjectAnchor("root")};
    actor_id_ = space_->CreateActor(anchors);
    space_->GetActor(actor_id_)->SetElementId(1);

    space_->Bind(actor_id_, "root", RelationKernelType::kAvatarAffordancePose,
                 "Seat");
  }

  std::unique_ptr<LocalContext> context_;
  Space* space_ = nullptr;
  spatialblend::entity_id_t actor_id_;
};

TEST_F(RelationEventE2ETest, ApproachAndEngageEmitsLifecycle) {
  std::vector<RelationEvent> seen;
  space_->RegisterOnRelationEvent(
      [&](const RelationEvent& e) { seen.push_back(e); });

  // Walk the actor from (5,5) toward the affordance at (0.5,0.5) in 10 steps.
  for (int i = 0; i <= 10; ++i) {
    double a = i / 10.0;
    space_->UpdateActorState(
        actor_id_,
        Eigen::Vector2d(5 * (1 - a) + 0.5 * a, 5 * (1 - a) + 0.5 * a),
        /*yaw=*/0.0);
  }

  ASSERT_FALSE(seen.empty());
  EXPECT_EQ(seen.front().to, RelationPhase::kActive);
  EXPECT_EQ(seen.back().to, RelationPhase::kContact);
  // The identity fields are carried through on every event.
  EXPECT_EQ(seen.front().anchor_name, "root");
  EXPECT_EQ(seen.front().kernel_type,
            RelationKernelType::kAvatarAffordancePose);
}

}  // namespace test
}  // namespace spatialblend
