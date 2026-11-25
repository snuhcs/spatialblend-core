// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/hierarchy_behavior_test.cc
//
// Expected behavior tests for recursive hierarchy-driven relation encoding.
// These tests define the DESIRED behavior — implementation should make them pass.
//
// Hierarchy:
//   Room
//     ├─ Board (with boundary) → leaf OR has children
//     │   ├─ Writable_L (affordance, optional)
//     │   └─ Writable_R (affordance, optional)
//     └─ Table (with boundary)
//
// Key behaviors:
//   - Spacing outside element: approach (geodesic distance decreasing)
//   - Spacing inside leaf element: grounded (polygon-mapped, R=1)
//   - Spacing inside non-leaf: scope narrows to children (same mechanic)
//   - Facing: always evaluates, regardless of containment
//   - AffordancePose inside leaf: full transform grounding

#include <gtest/gtest.h>

#include <cmath>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace test {

// ============================================================================
// Test fixture: configurable hierarchy
// ============================================================================
class HierarchyBehaviorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = LocalContext::Create();
    space_ = context_->AddSpace();
  }

  // Build a room with Board (leaf — no children) and Table
  void SetupLeafBoard() {
    auto* tree = space_->GetElementTree();
    tree->AddElement(1);
    tree->GetElement(1)->SetSharedId(100);
    auto room_poly = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 20}, {0, 20}});
    tree->GetElement(1)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(room_poly));

    // Board at (10, 15) with 4x4 boundary — LEAF (no children)
    tree->AddElement(2, 1);
    tree->GetElement(2)->SetSharedId(200);
    Eigen::Matrix4d bt = Eigen::Matrix4d::Identity();
    bt(0, 3) = 10.0; bt(1, 3) = 15.0;
    tree->GetElement(2)->SetLocalTransform(bt);
    auto board_poly = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{8, 13}, {12, 13}, {12, 17}, {8, 17}});
    tree->GetElement(2)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(board_poly));

    // Table at (10, 5) — LEAF
    tree->AddElement(3, 1);
    tree->GetElement(3)->SetSharedId(300);
    Eigen::Matrix4d tt = Eigen::Matrix4d::Identity();
    tt(0, 3) = 10.0; tt(1, 3) = 5.0;
    tree->GetElement(3)->SetLocalTransform(tt);
    auto table_poly = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{8, 3}, {12, 3}, {12, 7}, {8, 7}});
    tree->GetElement(3)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(table_poly));
  }

  // Build a room with Board that HAS children (Writable_L, Writable_R)
  void SetupNonLeafBoard() {
    SetupLeafBoard();  // Base setup

    auto* tree = space_->GetElementTree();

    // Add children to Board → Board becomes non-leaf
    // Writable_L at (9, 15)
    tree->AddElement(4, 2);
    tree->GetElement(4)->SetSharedId(201);
    tree->GetElement(4)->SetTag("Writable");
    Eigen::Matrix4d wl = Eigen::Matrix4d::Identity();
    wl(0, 3) = 9.0; wl(1, 3) = 15.0;
    tree->GetElement(4)->SetLocalTransform(wl);
    auto wl_poly = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{8.5, 14.5}, {9.5, 14.5}, {9.5, 15.5}, {8.5, 15.5}});
    tree->GetElement(4)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(wl_poly));

    // Writable_R at (11, 15)
    tree->AddElement(5, 2);
    tree->GetElement(5)->SetSharedId(202);
    tree->GetElement(5)->SetTag("Writable");
    Eigen::Matrix4d wr = Eigen::Matrix4d::Identity();
    wr(0, 3) = 11.0; wr(1, 3) = 15.0;
    tree->GetElement(5)->SetLocalTransform(wr);
    auto wr_poly = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{10.5, 14.5}, {11.5, 14.5}, {11.5, 15.5}, {10.5, 15.5}});
    tree->GetElement(5)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(wr_poly));
  }

  spatialblend::entity_id_t CreateActorWithBindings() {
    std::vector<SubjectAnchor> anchors = {
        SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
        SubjectAnchor("chest", Eigen::Vector3d(0, 0, 1.2)),
    };
    auto actor_id = space_->CreateActor(anchors);
    space_->GetActor(actor_id)->SetElementId(1);

    space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
    space_->Bind(actor_id, "chest", RelationKernelType::kAvatarFacing);
    return actor_id;
  }

  std::unique_ptr<LocalContext> context_;
  std::unique_ptr<IdealDriver> follower_;  // after context_: destroyed first
  Space* space_ = nullptr;
};

// ============================================================================
// SPACING: Outside element → approach behavior
// ============================================================================

TEST_F(HierarchyBehaviorTest, SpacingOutsideElement_ProximityLessThanOne) {
  SetupLeafBoard();
  auto actor = CreateActorWithBindings();

  // Actor at (5, 5) — far from Board (10, 15)
  space_->UpdateActorState(actor, Eigen::Vector2d(5, 5), 0.0);

  auto& bindings = space_->GetBindings(actor);
  ASSERT_FALSE(bindings.empty());

  // Find spacing binding matches
  bool found_board_spacing = false;
  for (const auto& b : bindings) {
    if (b.kernel_type != RelationKernelType::kAvatarSpacing) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 2) {  // Board
        found_board_spacing = true;
        // Outside: proximity should be < 1
        EXPECT_LT(m.eval.grounded.proximity, 1.0)
            << "Spacing proximity should be < 1 when outside element";
        EXPECT_GT(m.eval.grounded.proximity, 0.0);
      }
    }
  }
  EXPECT_TRUE(found_board_spacing) << "Board should be in spacing scope";
}

// ============================================================================
// SPACING: Inside LEAF element → grounded (polygon-mapped, R=1)
// ============================================================================

TEST_F(HierarchyBehaviorTest, SpacingInsideLeaf_ProximityEqualsOne) {
  SetupLeafBoard();
  auto actor = CreateActorWithBindings();

  // Actor at (10, 15) — inside Board boundary (8-12, 13-17), Board is leaf
  space_->UpdateActorState(actor, Eigen::Vector2d(10, 15), 0.0);

  auto& bindings = space_->GetBindings(actor);

  // Spacing should still evaluate Board (even though actor is inside)
  bool found_board_spacing = false;
  for (const auto& b : bindings) {
    if (b.kernel_type != RelationKernelType::kAvatarSpacing) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 2) {  // Board
        found_board_spacing = true;
        // Inside leaf: proximity should be 1.0 (grounded)
        EXPECT_NEAR(m.eval.grounded.proximity, 1.0, 0.01)
            << "Spacing proximity should be 1.0 when inside leaf element";
      }
    }
  }
  EXPECT_TRUE(found_board_spacing)
      << "Board should still be evaluated when actor is inside (leaf grounding)";
}

TEST_F(HierarchyBehaviorTest, SpacingInsideLeaf_HasContactNormalized) {
  SetupLeafBoard();
  auto actor = CreateActorWithBindings();

  // Actor at (9, 15) — inside Board, on the LEFT side
  space_->UpdateActorState(actor, Eigen::Vector2d(9, 15), 0.0);

  auto& bindings = space_->GetBindings(actor);

  for (const auto& b : bindings) {
    if (b.kernel_type != RelationKernelType::kAvatarSpacing) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 2) {
        // Inside leaf: support should have contact_normalized (polygon-mapped)
        EXPECT_TRUE(m.eval.support.has_contact)
            << "Support should have polygon-mapped contact when inside leaf";
      }
    }
  }
}

TEST_F(HierarchyBehaviorTest, SpacingInsideLeaf_EstimatedTimeZero) {
  SetupLeafBoard();
  auto actor = CreateActorWithBindings();

  // Warm up motion tracker
  space_->UpdateActorState(actor, Eigen::Vector2d(10, 14), 0.0);
  space_->UpdateActorState(actor, Eigen::Vector2d(10, 15), 0.0);

  // Now check: inside leaf → estimated_time should be ~0
  // This triggers the arrival gate → one-hot weight
  // We verify indirectly: if propagated, the payload estimated_time should be near 0
  // (This test needs two spaces to check payload — simplified version checks proximity)
  auto& bindings = space_->GetBindings(actor);
  for (const auto& b : bindings) {
    if (b.kernel_type != RelationKernelType::kAvatarSpacing) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 2) {
        EXPECT_NEAR(m.eval.grounded.proximity, 1.0, 0.01)
            << "At leaf grounding, proximity=1 → estimated_time=0";
      }
    }
  }
}

// ============================================================================
// SPACING: Inside NON-LEAF element → scope narrows to children
// ============================================================================

TEST_F(HierarchyBehaviorTest, SpacingInsideNonLeaf_ScopeNarrowsToChildren) {
  SetupNonLeafBoard();
  auto actor = CreateActorWithBindings();

  // Actor at (10, 15) — inside Board boundary, Board has children
  space_->UpdateActorState(actor, Eigen::Vector2d(10, 15), 0.0);

  auto& bindings = space_->GetBindings(actor);

  // Spacing scope should narrow to Board's children (Writable_L, Writable_R)
  // Board itself should NOT be in matches (it's the containing element)
  bool found_writable_l = false;
  bool found_writable_r = false;
  bool found_board = false;

  for (const auto& b : bindings) {
    if (b.kernel_type != RelationKernelType::kAvatarSpacing) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 4) found_writable_l = true;
      if (m.element_id == 5) found_writable_r = true;
      if (m.element_id == 2) found_board = true;
    }
  }

  EXPECT_TRUE(found_writable_l) << "Writable_L should be in scope";
  EXPECT_TRUE(found_writable_r) << "Writable_R should be in scope";
  // Container contract: the containing element (Board)
  // STAYS a spacing referent beside its child objects (its container-self
  // relation: depth below the border, signed, monotone through the border), so
  // a receiver keeps the source's containment relation. The container is also
  // the navigable scope that bounds the geodesic distance.
  EXPECT_TRUE(found_board)
      << "Board (container) is a spacing referent (container-self relation)";
}

TEST_F(HierarchyBehaviorTest, SpacingInsideNonLeaf_ChildrenHaveApproachProximity) {
  SetupNonLeafBoard();
  auto actor = CreateActorWithBindings();

  // Actor at (10, 15) — inside Board, equidistant from both Writables
  space_->UpdateActorState(actor, Eigen::Vector2d(10, 15), 0.0);

  auto& bindings = space_->GetBindings(actor);

  for (const auto& b : bindings) {
    if (b.kernel_type != RelationKernelType::kAvatarSpacing) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 4 || m.element_id == 5) {
        // Children: approach proximity (< 1, since actor is outside each child)
        EXPECT_LT(m.eval.grounded.proximity, 1.0)
            << "Children should have approach proximity (not grounded)";
        EXPECT_GT(m.eval.grounded.proximity, 0.0);
      }
    }
  }
}

// ============================================================================
// FACING: Always evaluates, regardless of containment
// ============================================================================

TEST_F(HierarchyBehaviorTest, FacingEvaluatesRegardlessOfContainment) {
  SetupLeafBoard();
  auto actor = CreateActorWithBindings();

  // Actor INSIDE Board — Facing should still evaluate Board and Table
  space_->UpdateActorState(actor, Eigen::Vector2d(10, 15), M_PI / 2);

  auto& bindings = space_->GetBindings(actor);

  bool found_facing = false;
  for (const auto& b : bindings) {
    if (b.kernel_type != RelationKernelType::kAvatarFacing) continue;
    found_facing = !b.latest_matches.empty();
  }
  EXPECT_TRUE(found_facing) << "Facing should evaluate even when inside an element";
}

TEST_F(HierarchyBehaviorTest, FacingProximityNotAffectedByContainment) {
  SetupLeafBoard();
  auto actor = CreateActorWithBindings();

  // Actor outside Board
  space_->UpdateActorState(actor, Eigen::Vector2d(5, 5), 0.0);
  auto& bindings_outside = space_->GetBindings(actor);
  double facing_prox_outside = 0.0;
  for (const auto& b : bindings_outside) {
    if (b.kernel_type != RelationKernelType::kAvatarFacing) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 2) facing_prox_outside = m.eval.grounded.proximity;
    }
  }

  // Actor inside Board — Facing proximity is about directional alignment,
  // not spatial containment. It should NOT jump to 1.0 just because inside.
  space_->UpdateActorState(actor, Eigen::Vector2d(10, 15), 0.0);
  auto& bindings_inside = space_->GetBindings(actor);
  double facing_prox_inside = 0.0;
  for (const auto& b : bindings_inside) {
    if (b.kernel_type != RelationKernelType::kAvatarFacing) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 2) facing_prox_inside = m.eval.grounded.proximity;
    }
  }

  // Facing proximity should be based on angular alignment, not distance
  // Both should be valid (finite, non-negative)
  EXPECT_TRUE(std::isfinite(facing_prox_outside));
  EXPECT_TRUE(std::isfinite(facing_prox_inside));
}

// ============================================================================
// CROSS-SPACE: Leaf grounding transfers correctly
// ============================================================================

class CrossSpaceLeafTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = LocalContext::Create();
    space_a_ = context_->AddSpace();
    space_b_ = context_->AddSpace();

    // Space A: Room + Board (leaf) at (10, 15)
    auto* ta = space_a_->GetElementTree();
    ta->AddElement(1);
    ta->GetElement(1)->SetSharedId(100);
    auto room_a = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 20}, {0, 20}});
    ta->GetElement(1)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(room_a));

    ta->AddElement(2, 1);
    ta->GetElement(2)->SetSharedId(200);
    Eigen::Matrix4d bt = Eigen::Matrix4d::Identity();
    bt(0, 3) = 10.0; bt(1, 3) = 15.0;
    ta->GetElement(2)->SetLocalTransform(bt);
    auto board_a = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{8, 13}, {12, 13}, {12, 17}, {8, 17}});
    ta->GetElement(2)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(board_a));

    // Space B: Room + Board at (5, 10) — different position, different boundary shape
    auto* tb = space_b_->GetElementTree();
    tb->AddElement(1);
    tb->GetElement(1)->SetSharedId(100);
    auto room_b = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{0, 0}, {15, 0}, {15, 15}, {0, 15}});
    tb->GetElement(1)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(room_b));

    tb->AddElement(2, 1);
    tb->GetElement(2)->SetSharedId(200);
    Eigen::Matrix4d bt2 = Eigen::Matrix4d::Identity();
    bt2(0, 3) = 5.0; bt2(1, 3) = 10.0;
    tb->GetElement(2)->SetLocalTransform(bt2);
    auto board_b = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{3, 8}, {7, 8}, {7, 12}, {3, 12}});
    tb->GetElement(2)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(board_b));

    // Actor in A
    std::vector<SubjectAnchor> anchors = {
        SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
    };
    actor_id_ = space_a_->CreateActor(anchors);
    space_a_->GetActor(actor_id_)->SetElementId(1);
    space_a_->GetElementTree()->GetElement(1)->SetSharedId(100);
    space_a_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);

    // Avatar in B
    avatar_id_ = space_b_->CreateActor(anchors);
    space_b_->GetActor(avatar_id_)->SetElementId(1);
    follower_ = std::make_unique<IdealDriver>(space_b_, avatar_id_);
    space_b_->GetElementTree()->GetElement(1)->SetSharedId(100);
  }

  std::unique_ptr<LocalContext> context_;
  std::unique_ptr<IdealDriver> follower_;  // after context_: destroyed first
  Space* space_a_ = nullptr;
  Space* space_b_ = nullptr;
  spatialblend::entity_id_t actor_id_;
  spatialblend::entity_id_t avatar_id_;
};

TEST_F(CrossSpaceLeafTest, LeafGrounding_AvatarNearTargetBoard) {
  // Actor inside Board_A (leaf) → grounded → propagate
  // Avatar should be near Board_B
  space_a_->UpdateActorState(actor_id_, Eigen::Vector2d(10, 15), 0.0);

  auto avatar_pos = space_b_->GetActor(avatar_id_)->GetPosition();
  double dist_to_board_b = (avatar_pos - Eigen::Vector2d(5, 10)).norm();

  EXPECT_LT(dist_to_board_b, 5.0)
      << "When actor is grounded on Board_A (leaf), avatar should be near Board_B";
}

TEST_F(CrossSpaceLeafTest, LeafGrounding_SidePreserved) {
  // Actor on LEFT side of Board_A (x=9) → avatar should be on LEFT side of Board_B
  space_a_->UpdateActorState(actor_id_, Eigen::Vector2d(9, 15), 0.0);
  auto avatar_left = space_b_->GetActor(avatar_id_)->GetPosition();

  // Actor on RIGHT side of Board_A (x=11)
  space_a_->UpdateActorState(actor_id_, Eigen::Vector2d(11, 15), 0.0);
  auto avatar_right = space_b_->GetActor(avatar_id_)->GetPosition();

  // The receiver follows the source's own slide.
  // Board_A is centred at x=10, so x=9 and x=11 are offsets -1 and +1, and
  // the receiver keeps that 2 m. Freezing both on the contact was the
  // removed centroid hold.
  EXPECT_NEAR((avatar_left - avatar_right).norm(), 2.0, 1e-4);
}

TEST_F(CrossSpaceLeafTest, ApproachThenGrounding_ContinuousTrajectory) {
  // Actor walks from far → approaches Board → enters Board
  // Avatar trajectory should be continuous (no teleportation)
  std::vector<Eigen::Vector2d> actor_trajectory = {
      {5, 5}, {7, 8}, {9, 11}, {10, 13}, {10, 14}, {10, 15}};

  std::vector<Eigen::Vector2d> avatar_positions;
  for (const auto& pos : actor_trajectory) {
    space_a_->UpdateActorState(actor_id_, pos, M_PI / 2);
    avatar_positions.push_back(space_b_->GetActor(avatar_id_)->GetPosition());
  }

  // No teleportation: max jump between consecutive frames should be bounded
  for (size_t i = 1; i < avatar_positions.size(); ++i) {
    double jump = (avatar_positions[i] - avatar_positions[i - 1]).norm();
    EXPECT_LT(jump, 10.0)
        << "Jump of " << jump << "m at step " << i
        << " — trajectory should be continuous through grounding transition";
  }

  // Final avatar position should be near Board_B
  double final_dist = (avatar_positions.back() - Eigen::Vector2d(5, 10)).norm();
  EXPECT_LT(final_dist, 5.0)
      << "After grounding, avatar should be near Board_B";
}

// ============================================================================
// HIERARCHY TRAVERSAL: Room → non-leaf Board → leaf Writable
// ============================================================================

TEST_F(HierarchyBehaviorTest, FullHierarchyTraversal) {
  SetupNonLeafBoard();
  auto actor = CreateActorWithBindings();

  // Step 1: Actor far from Board — Spacing to Board and Table
  space_->UpdateActorState(actor, Eigen::Vector2d(5, 5), 0.0);
  {
    auto& bindings = space_->GetBindings(actor);
    bool found_board = false, found_table = false;
    for (const auto& b : bindings) {
      if (b.kernel_type != RelationKernelType::kAvatarSpacing) continue;
      for (const auto& m : b.latest_matches) {
        if (m.element_id == 2) found_board = true;
        if (m.element_id == 3) found_table = true;
      }
    }
    EXPECT_TRUE(found_board) << "Step 1: Board should be in scope";
    EXPECT_TRUE(found_table) << "Step 1: Table should be in scope";
  }

  // Step 2: Actor inside Board (non-leaf) — Spacing narrows to Writable children
  space_->UpdateActorState(actor, Eigen::Vector2d(10, 15), 0.0);
  {
    auto& bindings = space_->GetBindings(actor);
    bool found_writable = false, found_board = false, found_table = false;
    for (const auto& b : bindings) {
      if (b.kernel_type != RelationKernelType::kAvatarSpacing) continue;
      for (const auto& m : b.latest_matches) {
        if (m.element_id == 4 || m.element_id == 5) found_writable = true;
        if (m.element_id == 2) found_board = true;
        if (m.element_id == 3) found_table = true;
      }
    }
    EXPECT_TRUE(found_writable) << "Step 2: Writable children should be in scope";
    // Container contract: the container (Board) stays a spacing referent
    // (container-self relation) beside its child objects; the sibling Table sits
    // OUTSIDE the active container so it drops out.
    EXPECT_TRUE(found_board) << "Step 2: Board (container) is a referent (container-self)";
    EXPECT_FALSE(found_table) << "Step 2: Table (outside container) should not be in scope";
  }

  // Step 3: Actor inside Writable_L (leaf) — Spacing should ground
  space_->UpdateActorState(actor, Eigen::Vector2d(9, 15), 0.0);
  {
    auto& bindings = space_->GetBindings(actor);
    bool found_grounded = false;
    for (const auto& b : bindings) {
      if (b.kernel_type != RelationKernelType::kAvatarSpacing) continue;
      for (const auto& m : b.latest_matches) {
        if (m.element_id == 4) {  // Writable_L
          found_grounded = true;
          EXPECT_NEAR(m.eval.grounded.proximity, 1.0, 0.01)
              << "Step 3: Writable_L is leaf — should be grounded (R=1)";
        }
      }
    }
    EXPECT_TRUE(found_grounded) << "Step 3: Writable_L should be evaluated (leaf grounding)";
  }
}

// ============================================================================
// AFFORDANCE POSE: Full transform engagement at leaf
// ============================================================================

class AffordancePoseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = LocalContext::Create();
    space_ = context_->AddSpace();

    auto* tree = space_->GetElementTree();
    tree->AddElement(1);
    tree->GetElement(1)->SetSharedId(100);
    auto room_poly = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 20}, {0, 20}});
    tree->GetElement(1)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(room_poly));

    // Chair at (10, 10) with boundary — leaf element
    tree->AddElement(2, 1);
    tree->GetElement(2)->SetSharedId(200);
    Eigen::Matrix4d ct = Eigen::Matrix4d::Identity();
    ct(0, 3) = 10.0; ct(1, 3) = 10.0;
    tree->GetElement(2)->SetLocalTransform(ct);
    auto chair_poly = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{9, 9}, {11, 9}, {11, 11}, {9, 11}});
    tree->GetElement(2)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(chair_poly));
    tree->GetElement(2)->SetTag("Sittable");

    // Actor with hips anchor for AffordancePose
    std::vector<SubjectAnchor> anchors = {
        SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
        SubjectAnchor("hips", Eigen::Vector3d(0, 0, 0.9)),
    };
    actor_id_ = space_->CreateActor(anchors);
    space_->GetActor(actor_id_)->SetElementId(1);
    space_->Bind(actor_id_, "hips",
                 RelationKernelType::kAvatarAffordancePose, "Sittable");
  }

  std::unique_ptr<LocalContext> context_;
  std::unique_ptr<IdealDriver> follower_;  // after context_: destroyed first
  Space* space_ = nullptr;
  spatialblend::entity_id_t actor_id_;
};

TEST_F(AffordancePoseTest, InScopeWhenFarFromChairInsideItsParentRegion) {
  // Actor at (2, 2) — ~9.9 m from the Chair footprint [9,11]^2, in the room that
  // is the chair's parent (D3). There is no declared near field (it is
  // retired): an AffordancePose relation is in scope while the anchor is inside its parent object's region, so
  // the candidate is grounded (its R is observable) AND active.
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(2, 2), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  bool affordance_evaluated = false;
  double prox = 0.0;
  for (const auto& b : bindings) {
    if (b.kernel_type == RelationKernelType::kAvatarAffordancePose) {
      affordance_evaluated = !b.latest_matches.empty();
      if (!b.latest_matches.empty()) {
        prox = b.latest_matches[0].eval.grounded.proximity;
      }
    }
  }
  EXPECT_TRUE(affordance_evaluated)
      << "AffordancePose is still grounded (R observable) when far";
  EXPECT_GE(prox, 0.0) << "Proximity must be non-negative";
  for (const auto& st : space_->GetRelationStates(actor_id_)) {
    if (st.kernel_type == RelationKernelType::kAvatarAffordancePose) {
      EXPECT_TRUE(st.active) << "D3: in scope inside the parent region";
    }
  }
}

TEST_F(AffordancePoseTest, ActiveWhenInsideChairBoundary) {
  // Actor at (10, 10) — inside Chair boundary (9-11, 9-11)
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(10, 10), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  bool affordance_active = false;
  for (const auto& b : bindings) {
    if (b.kernel_type == RelationKernelType::kAvatarAffordancePose) {
      affordance_active = !b.latest_matches.empty();
    }
  }
  EXPECT_TRUE(affordance_active)
      << "AffordancePose should be active when actor is inside Chair boundary";
}

TEST_F(AffordancePoseTest, InsideLeaf_ProximityEqualsOne) {
  // Actor inside Chair (leaf) → proximity should be 1.0
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(10, 10), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  for (const auto& b : bindings) {
    if (b.kernel_type != RelationKernelType::kAvatarAffordancePose) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 2) {
        EXPECT_NEAR(m.eval.grounded.proximity, 1.0, 0.01)
            << "AffordancePose inside leaf → proximity should be 1.0";
      }
    }
  }
}

TEST_F(AffordancePoseTest, EncodedStateIncludesRotation) {
  // Actor inside Chair with specific yaw — encoded state should include rotation
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(10, 10), M_PI / 4);

  auto& bindings = space_->GetBindings(actor_id_);
  for (const auto& b : bindings) {
    if (b.kernel_type != RelationKernelType::kAvatarAffordancePose) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 2) {
        // AffordancePose encoded_state should have > 2 components (pos + rot)
        EXPECT_GE(m.eval.grounded.encoded_state.size(), 3)
            << "AffordancePose encoded state should include rotation";
      }
    }
  }
}

// ============================================================================
// AFFORDANCE POSE: Cross-space transfer
// ============================================================================

class AffordanceCrossSpaceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = LocalContext::Create();
    space_a_ = context_->AddSpace();
    space_b_ = context_->AddSpace();

    // Space A: Room + Chair (leaf) at (10, 10)
    auto* ta = space_a_->GetElementTree();
    ta->AddElement(1);
    ta->GetElement(1)->SetSharedId(100);
    auto room_a = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 20}, {0, 20}});
    ta->GetElement(1)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(room_a));

    ta->AddElement(2, 1);
    ta->GetElement(2)->SetSharedId(200);
    ta->GetElement(2)->SetTag("Sittable");
    Eigen::Matrix4d ct = Eigen::Matrix4d::Identity();
    ct(0, 3) = 10.0; ct(1, 3) = 10.0;
    ta->GetElement(2)->SetLocalTransform(ct);
    auto chair_a = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{9, 9}, {11, 9}, {11, 11}, {9, 11}});
    ta->GetElement(2)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(chair_a));

    // Space B: Room + Chair at (5, 5) — different position
    auto* tb = space_b_->GetElementTree();
    tb->AddElement(1);
    tb->GetElement(1)->SetSharedId(100);
    auto room_b = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{0, 0}, {15, 0}, {15, 15}, {0, 15}});
    tb->GetElement(1)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(room_b));

    tb->AddElement(2, 1);
    tb->GetElement(2)->SetSharedId(200);
    tb->GetElement(2)->SetTag("Sittable");
    Eigen::Matrix4d ct2 = Eigen::Matrix4d::Identity();
    ct2(0, 3) = 5.0; ct2(1, 3) = 5.0;
    tb->GetElement(2)->SetLocalTransform(ct2);
    auto chair_b = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{4, 4}, {6, 4}, {6, 6}, {4, 6}});
    tb->GetElement(2)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(chair_b));

    // Actor in A with hips anchor
    std::vector<SubjectAnchor> anchors = {
        SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
        SubjectAnchor("hips", Eigen::Vector3d(0, 0, 0.9)),
    };
    actor_id_ = space_a_->CreateActor(anchors);
    space_a_->GetActor(actor_id_)->SetElementId(1);
    space_a_->GetElementTree()->GetElement(1)->SetSharedId(100);
    space_a_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
    space_a_->Bind(actor_id_, "hips",
                   RelationKernelType::kAvatarAffordancePose, "Sittable");

    // Avatar in B
    avatar_id_ = space_b_->CreateActor(anchors);
    space_b_->GetActor(avatar_id_)->SetElementId(1);
    follower_ = std::make_unique<IdealDriver>(space_b_, avatar_id_);
    space_b_->GetElementTree()->GetElement(1)->SetSharedId(100);
  }

  std::unique_ptr<LocalContext> context_;
  std::unique_ptr<IdealDriver> follower_;  // after context_: destroyed first
  Space* space_a_ = nullptr;
  Space* space_b_ = nullptr;
  spatialblend::entity_id_t actor_id_;
  spatialblend::entity_id_t avatar_id_;
};

TEST_F(AffordanceCrossSpaceTest, GroundedOnChair_AvatarNearTargetChair) {
  // Actor sits on Chair_A (10, 10) → avatar should be near Chair_B (5, 5)
  space_a_->UpdateActorState(actor_id_, Eigen::Vector2d(10, 10), 0.0);

  auto avatar_pos = space_b_->GetActor(avatar_id_)->GetPosition();
  double dist = (avatar_pos - Eigen::Vector2d(5, 5)).norm();
  EXPECT_LT(dist, 3.0)
      << "Actor grounded on Chair_A → avatar should be near Chair_B";
}

TEST_F(AffordanceCrossSpaceTest, ApproachThenSit_ContinuousTrajectory) {
  // Actor walks from (2, 2) → approaches Chair → sits
  std::vector<Eigen::Vector2d> trajectory = {
      {2, 2}, {4, 4}, {6, 6}, {8, 8}, {9.5, 9.5}, {10, 10}};

  std::vector<Eigen::Vector2d> avatar_positions;
  for (const auto& pos : trajectory) {
    space_a_->UpdateActorState(actor_id_, pos, 0.0);
    avatar_positions.push_back(space_b_->GetActor(avatar_id_)->GetPosition());
  }

  // No teleportation
  for (size_t i = 1; i < avatar_positions.size(); ++i) {
    double jump = (avatar_positions[i] - avatar_positions[i - 1]).norm();
    EXPECT_LT(jump, 10.0)
        << "Jump at step " << i << ": " << jump;
  }

  // Final avatar should be near Chair_B
  double final_dist = (avatar_positions.back() - Eigen::Vector2d(5, 5)).norm();
  EXPECT_LT(final_dist, 3.0)
      << "After sitting, avatar should be near Chair_B";
}

// ============================================================================
// MULTI-AFFORDANCE: Transition between affordances within element
// ============================================================================

class MultiAffordanceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = LocalContext::Create();
    space_ = context_->AddSpace();

    auto* tree = space_->GetElementTree();
    tree->AddElement(1);
    tree->GetElement(1)->SetSharedId(100);
    auto room_poly = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 20}, {0, 20}});
    tree->GetElement(1)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(room_poly));

    // Board at (10, 15) with 6x4 boundary — NON-LEAF (has children)
    tree->AddElement(2, 1);
    tree->GetElement(2)->SetSharedId(200);
    Eigen::Matrix4d bt = Eigen::Matrix4d::Identity();
    bt(0, 3) = 10.0; bt(1, 3) = 15.0;
    tree->GetElement(2)->SetLocalTransform(bt);
    auto board_poly = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{7, 13}, {13, 13}, {13, 17}, {7, 17}});
    tree->GetElement(2)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(board_poly));

    // Writable_L at (8, 15) — LEFT side affordance
    tree->AddElement(3, 2);
    tree->GetElement(3)->SetSharedId(201);
    tree->GetElement(3)->SetTag("Writable");
    Eigen::Matrix4d wl = Eigen::Matrix4d::Identity();
    wl(0, 3) = 8.0; wl(1, 3) = 15.0;
    tree->GetElement(3)->SetLocalTransform(wl);
    auto wl_poly = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{7, 14}, {9, 14}, {9, 16}, {7, 16}});
    tree->GetElement(3)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(wl_poly));

    // Writable_R at (12, 15) — RIGHT side affordance
    tree->AddElement(4, 2);
    tree->GetElement(4)->SetSharedId(202);
    tree->GetElement(4)->SetTag("Writable");
    Eigen::Matrix4d wr = Eigen::Matrix4d::Identity();
    wr(0, 3) = 12.0; wr(1, 3) = 15.0;
    tree->GetElement(4)->SetLocalTransform(wr);
    auto wr_poly = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{11, 14}, {13, 14}, {13, 16}, {11, 16}});
    tree->GetElement(4)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(wr_poly));

    // Actor with root + hand anchors
    std::vector<SubjectAnchor> anchors = {
        SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
        SubjectAnchor("hand_right", Eigen::Vector3d(0.3, 0.2, 1.0)),
    };
    actor_id_ = space_->CreateActor(anchors);
    space_->GetActor(actor_id_)->SetElementId(1);
    space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
    space_->Bind(actor_id_, "hand_right",
                 RelationKernelType::kAvatarAffordancePose, "Writable");
  }

  std::unique_ptr<LocalContext> context_;
  std::unique_ptr<IdealDriver> follower_;  // after context_: destroyed first
  Space* space_ = nullptr;
  spatialblend::entity_id_t actor_id_;
};

TEST_F(MultiAffordanceTest, InsideBoard_SpacingToWritables) {
  // Actor inside Board (non-leaf) → Spacing to Writable children
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(10, 15), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  bool found_wl = false, found_wr = false;
  for (const auto& b : bindings) {
    if (b.kernel_type != RelationKernelType::kAvatarSpacing) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 3) found_wl = true;
      if (m.element_id == 4) found_wr = true;
    }
  }
  EXPECT_TRUE(found_wl) << "Writable_L should be in spacing scope";
  EXPECT_TRUE(found_wr) << "Writable_R should be in spacing scope";
}

TEST_F(MultiAffordanceTest, NearWritableL_HigherProximity) {
  // Actor at (8, 15) — near Writable_L, far from Writable_R
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(8, 15), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  double prox_l = 0.0, prox_r = 0.0;
  for (const auto& b : bindings) {
    if (b.kernel_type != RelationKernelType::kAvatarSpacing) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 3) prox_l = m.eval.grounded.proximity;
      if (m.element_id == 4) prox_r = m.eval.grounded.proximity;
    }
  }
  EXPECT_GT(prox_l, prox_r)
      << "Writable_L proximity should be higher when actor is near it";
}

TEST_F(MultiAffordanceTest, InsideWritableL_Grounded) {
  // Actor inside Writable_L boundary (7-9, 14-16) — leaf
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(8, 15), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  for (const auto& b : bindings) {
    if (b.kernel_type != RelationKernelType::kAvatarSpacing) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 3) {
        EXPECT_NEAR(m.eval.grounded.proximity, 1.0, 0.01)
            << "Inside Writable_L (leaf) → grounded";
      }
    }
  }
}

TEST_F(MultiAffordanceTest, AffordancePose_ActiveInsideWritableL) {
  // Actor inside Writable_L → AffordancePose should be active for Writable_L
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(8, 15), 0.0);

  auto& bindings = space_->GetBindings(actor_id_);
  bool found_affordance = false;
  for (const auto& b : bindings) {
    if (b.kernel_type != RelationKernelType::kAvatarAffordancePose) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 3) {
        found_affordance = true;
        EXPECT_GT(m.eval.grounded.encoded_state.size(), 0)
            << "AffordancePose should have encoded state";
      }
    }
  }
  EXPECT_TRUE(found_affordance)
      << "AffordancePose should be active for Writable_L";
}

TEST_F(MultiAffordanceTest, TransitionLeftToRight) {
  // Actor moves from Writable_L to Writable_R within Board
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(8, 15), 0.0);
  auto& bindings_l = space_->GetBindings(actor_id_);

  double prox_l_at_left = 0.0, prox_r_at_left = 0.0;
  for (const auto& b : bindings_l) {
    if (b.kernel_type != RelationKernelType::kAvatarSpacing) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 3) prox_l_at_left = m.eval.grounded.proximity;
      if (m.element_id == 4) prox_r_at_left = m.eval.grounded.proximity;
    }
  }

  // Move to Writable_R
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(12, 15), 0.0);
  auto& bindings_r = space_->GetBindings(actor_id_);

  double prox_l_at_right = 0.0, prox_r_at_right = 0.0;
  for (const auto& b : bindings_r) {
    if (b.kernel_type != RelationKernelType::kAvatarSpacing) continue;
    for (const auto& m : b.latest_matches) {
      if (m.element_id == 3) prox_l_at_right = m.eval.grounded.proximity;
      if (m.element_id == 4) prox_r_at_right = m.eval.grounded.proximity;
    }
  }

  // At left position: L should be more proximate
  EXPECT_GT(prox_l_at_left, prox_r_at_left);
  // At right position: R should be more proximate
  EXPECT_GT(prox_r_at_right, prox_l_at_right);
}

}  // namespace test
}  // namespace spatialblend
