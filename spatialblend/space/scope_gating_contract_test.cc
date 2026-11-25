// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/scope_gating_contract_test.cc
//
// Contract tests for these component requirements of the core:
//   E1-1  Facing scope = line of sight, any hierarchy depth
//   E1-2  Spacing scope = walkable reachability, no Euclidean
//   E1-3  AffordancePose dormant until a declared near-field
//         RETIRED: scope is the parent object's region (D3)
//   E1-5  an inactive relation has zero influence on realization
//         RETIRED: there is no activation gate
//   E1-6  per-family configurable, observable thresholds
//         the driving / contact labels only
//   D1-D9 the family scope definitions (testing/SCOPE_CASES.md, "Decisions")
//   H-2   a shared_id resolves to exactly one element per space
//
// Every oracle is computed here from the fixture geometry (segment vs
// rectangle, point in notch, point-to-square distance), never from the core.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace test {
namespace {

std::shared_ptr<Polygon2D> Rect(double x0, double y0, double x1, double y1) {
  return std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});
}

void AddFootprint(ElementTree* tree, element_id_t id, element_id_t parent,
                  std::shared_ptr<Polygon2D> poly, GeometryRole role,
                  shared_element_id_t shared = kInvalidSharedElementId) {
  ASSERT_TRUE(tree->AddElement(id, parent).ok());
  tree->GetElement(id)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(poly, nullptr, 0.0, role));
  // every element of these worlds is shared unless it says otherwise (an unshared element is never a
  // referent: SP-D7-03)
  tree->GetElement(id)->SetSharedId(shared != kInvalidSharedElementId ? shared : id);
}

void AddBox(ElementTree* tree, element_id_t id, element_id_t parent, double cx,
            double cy, double half, double height, const std::string& tag = "") {
  ASSERT_TRUE(tree->AddElement(id, parent).ok());
  tree->GetElement(id)->SetSharedId(id);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = cx;
  t(1, 3) = cy;
  tree->GetElement(id)->SetLocalTransform(t);
  tree->GetElement(id)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-half, -half, 0.0), Eigen::Vector3d(half, half, height),
          GeometryRole::kObstacle));  // furniture volume (occludes, blocks)
  if (!tag.empty()) tree->GetElement(id)->SetTag(tag);
}

// Box of half-extents (hx, hy) and height h at (cx, cy); Obstacle role.
void AddBoxXY(ElementTree* tree, element_id_t id, element_id_t parent,
              double cx, double cy, double hx, double hy, double h,
              const std::string& tag = "") {
  ASSERT_TRUE(tree->AddElement(id, parent).ok());
  tree->GetElement(id)->SetSharedId(id);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = cx;
  t(1, 3) = cy;
  tree->GetElement(id)->SetLocalTransform(t);
  tree->GetElement(id)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-hx, -hy, 0.0), Eigen::Vector3d(hx, hy, h),
          GeometryRole::kObstacle));
  if (!tag.empty()) tree->GetElement(id)->SetTag(tag);
}

// Independent oracle: does segment p->q cross the axis-aligned rectangle?
// (Liang-Barsky clipping.)
bool SegmentHitsRect(Eigen::Vector2d p, Eigen::Vector2d q, double x0, double y0,
                     double x1, double y1) {
  double t0 = 0.0, t1 = 1.0;
  const Eigen::Vector2d d = q - p;
  const double pv[4] = {-d.x(), d.x(), -d.y(), d.y()};
  const double qv[4] = {p.x() - x0, x1 - p.x(), p.y() - y0, y1 - p.y()};
  for (int i = 0; i < 4; ++i) {
    if (std::abs(pv[i]) < 1e-12) {
      if (qv[i] < 0) return false;
    } else {
      const double r = qv[i] / pv[i];
      if (pv[i] < 0) t0 = std::max(t0, r);
      else t1 = std::min(t1, r);
      if (t0 > t1) return false;
    }
  }
  return true;
}

bool IsActive(const std::vector<RelationState>& states, element_id_t referent,
              RelationKernelType kernel) {
  for (const auto& s : states) {
    if (s.referent_element_id == referent && s.kernel_type == kernel &&
        s.active) {
      return true;
    }
  }
  return false;
}

// ===========================================================================
// E1-1: Facing by line of sight, any depth.
//
//   Building(1) ─┬─ RoomA(2) [0,20]x[0,20]  <- actor element
//                │    ├─ Chair(3) 1x1x1 box at (15,10)
//                │    └─ Wall(4)  obstacle volume [10,11]x[3,15], 2.5 m
//                └─ RoomB(5) [20,40]x[0,20]
//                     └─ Lamp(6) 1x1x1.5 box at (25,3)
// The chair is inside the actor's subtree but behind the wall; the lamp is
// in another subtree but in plain view.
// ===========================================================================
TEST(ScopeGatingContractTest, ScopeGating_E1_1_OccludedReferentNeverActive) {
  auto context = LocalContext::Create();
  Space* space = context->AddSpace();
  auto* tree = space->GetElementTree();
  AddFootprint(tree, 1, kInvalidElementId, Rect(0, 0, 40, 20),
               GeometryRole::kContainment);
  AddFootprint(tree, 2, 1, Rect(0, 0, 20, 20), GeometryRole::kNavigable);
  AddBox(tree, 3, 2, 15, 10, 0.5, 1.0);
  AddBoxXY(tree, 4, 2, 10.5, 9, 0.5, 6, 2.5);
  AddFootprint(tree, 5, 1, Rect(20, 0, 40, 20), GeometryRole::kNavigable);
  AddBox(tree, 6, 5, 25, 3, 0.5, 1.5);

  auto actor = space->CreateActor({SubjectAnchor("root", {0, 0, 0}),
                                   SubjectAnchor("chest", {0, 0, 1.2})});
  FollowProposals(space, actor);
  space->GetActor(actor)->SetElementId(2);
  space->Bind(actor, "chest", RelationKernelType::kAvatarFacing);

  int frames = 0;
  for (double x = 3.0; x <= 6.0 + 1e-9; x += 0.25) {
    const Eigen::Vector2d eye(x, 2.0);
    // Oracle: every sightline to the chair's footprint crosses the wall; the
    // sightline to the lamp's centre does not.
    for (double cx : {14.5, 15.5})
      for (double cy : {9.5, 10.5})
        ASSERT_TRUE(SegmentHitsRect(eye, {cx, cy}, 10, 3, 11, 15));
    ASSERT_FALSE(SegmentHitsRect(eye, {25, 3}, 10, 3, 11, 15));

    space->UpdateActorState(actor, eye, 0.0);  // facing +x: both ahead
    const auto states = space->GetRelationStates(actor);
    EXPECT_FALSE(IsActive(states, 3, RelationKernelType::kAvatarFacing))
        << "occluded chair active at x=" << x;
    EXPECT_TRUE(IsActive(states, 6, RelationKernelType::kAvatarFacing))
        << "visible lamp in another subtree not active at x=" << x;
    ++frames;
  }
  EXPECT_GT(frames, 10);

  // Control: without the wall the same chair is in scope and active, so the
  // exclusion above is the occlusion, not the bearing.
  ASSERT_TRUE(tree->RemoveElement(4).ok());
  space->UpdateActorState(actor, Eigen::Vector2d(4.0, 2.0), 0.0);
  EXPECT_TRUE(IsActive(space->GetRelationStates(actor), 3,
                       RelationKernelType::kAvatarFacing));
}

// ===========================================================================
// E1-2: Spacing by walkable reachability.
//
//   Room(1) walkable L = [0,10]^2 minus the walled-off notch [6,10]x[6,10]
//     ├─ SeatOpen(2)      box at (8,2)     (right arm, reachable)
//     ├─ SeatSealed(3)    box at (8,8)     (in the notch)
//     ├─ SealedRegion(4)  [6.5,9.5]^2      (the notch region)
//     └─ OpenRegion(5)    [1,6]x[6,9.5]    (sibling region, reachable)
//          └─ SeatInRegion(6) box at (2.5,8)
// ===========================================================================
TEST(ScopeGatingContractTest, ScopeGating_E1_2_SealedRegionSeatNeverActive) {
  auto context = LocalContext::Create();
  Space* space = context->AddSpace();
  auto* tree = space->GetElementTree();
  ASSERT_TRUE(tree->AddElement(1).ok());
  tree->GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
              {0, 0}, {10, 0}, {10, 6}, {6, 6}, {6, 10}, {0, 10}})));
  AddBox(tree, 2, 1, 8, 2, 0.25, 0.45, "Seat");
  AddBox(tree, 3, 1, 8, 8, 0.25, 0.45, "Seat");
  AddFootprint(tree, 4, 1, Rect(6.5, 6.5, 9.5, 9.5), GeometryRole::kContainment);
  AddFootprint(tree, 5, 1, Rect(1, 6, 6, 9.5), GeometryRole::kContainment);
  AddBox(tree, 6, 5, 2.5, 8, 0.25, 0.45, "Seat");

  auto actor = space->CreateActor({SubjectAnchor("root", {0, 0, 0})});
  FollowProposals(space, actor);
  space->GetActor(actor)->SetElementId(1);
  space->Bind(actor, "root", RelationKernelType::kAvatarSpacing);

  // Oracle for the notch: x > 6 && y > 6 is off the walkable area.
  auto sealed = [](const Eigen::Vector2d& p) { return p.x() > 6 && p.y() > 6; };
  ASSERT_TRUE(sealed({8, 8}));
  ASSERT_FALSE(sealed({8, 2}));

  for (double y = 2.0; y <= 5.0 + 1e-9; y += 0.5) {
    space->UpdateActorState(actor, Eigen::Vector2d(2.0, y), 0.0);
    const auto states = space->GetRelationStates(actor);
    EXPECT_FALSE(IsActive(states, 3, RelationKernelType::kAvatarSpacing))
        << "sealed seat active at y=" << y;
    EXPECT_FALSE(IsActive(states, 4, RelationKernelType::kAvatarSpacing))
        << "sealed region active at y=" << y;
    for (const auto& s : states) {
      EXPECT_NE(s.referent_element_id, 3) << "sealed seat must be out of scope";
      EXPECT_NE(s.referent_element_id, 4) << "sealed region must be out of scope";
    }
    EXPECT_TRUE(IsActive(states, 2, RelationKernelType::kAvatarSpacing));
    EXPECT_TRUE(IsActive(states, 5, RelationKernelType::kAvatarSpacing));
  }

  // Entering the sibling region narrows scope to its seat, which is reachable.
  space->UpdateActorState(actor, Eigen::Vector2d(2.5, 6.5), 0.0);
  EXPECT_TRUE(IsActive(space->GetRelationStates(actor), 6,
                       RelationKernelType::kAvatarSpacing));
}

// ===========================================================================
// E1-3 / E1-6: AffordancePose near-field in metres, configurable per family.
//
//   Room(1) [0,60]^2 ── Chair(2) footprint [24,26]^2, tag Sittable
// The hips anchor walks along y=25 at x = 26 + d, so its distance to the
// chair footprint is exactly d (the anchor projects horizontally).
// ===========================================================================
class NearFieldContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = LocalContext::Create();
    space_ = context_->AddSpace();
    auto* tree = space_->GetElementTree();
    AddFootprint(tree, 1, kInvalidElementId, Rect(0, 0, 60, 60),
                 GeometryRole::kNavigable);
    AddFootprint(tree, 2, 1, Rect(24, 24, 26, 26), GeometryRole::kNavigable);
    tree->GetElement(2)->SetTag("Sittable");
    actor_ = space_->CreateActor({SubjectAnchor("root", {0, 0, 0}),
                                  SubjectAnchor("hips", {0, 0, 0.9})});
    FollowProposals(space_, actor_);
    space_->GetActor(actor_)->SetElementId(1);
    space_->Bind(actor_, "hips", RelationKernelType::kAvatarAffordancePose,
                 "Sittable");
  }

  bool ActiveAt(double d) {
    space_->UpdateActorState(actor_, Eigen::Vector2d(26.0 + d, 25.0), 0.0);
    return IsActive(space_->GetRelationStates(actor_), 2,
                    RelationKernelType::kAvatarAffordancePose);
  }

  std::unique_ptr<LocalContext> context_;
  Space* space_ = nullptr;
  spatialblend::entity_id_t actor_ = 0;
};

// Retired contracts: the AffordancePose 1.5 / 2.0 m
// near field (E1-3: InactiveOutsideNearField, EnterAndReleaseAtDeclaredDistances), its
// 0.3 s release dwell (ReleaseAfterDeclaredDwell, SingleFrameGlitchKeepsSeatTarget), and
// the configurable near-field thresholds (E1-6 SettingThresholdsMovesFrames). The scope
// of an AffordancePose relation is the parent object's region (D3), not a distance.
TEST_F(NearFieldContractTest, ScopeGating_D3_InScopeAnywhereInsideTheParentRegion) {
  // The chair's parent is the room; the anchor is anywhere inside it: in scope, at
  // 0.5 m or at 20 m alike (under the near field it was active only within 1.5 m).
  for (double d : {0.5, 2.0, 9.0, 20.0}) {
    EXPECT_TRUE(ActiveAt(d)) << "d=" << d;
  }
}

// The chair sits in an Area inside the room. Standing in the room but outside the Area is
// outside the chair's parent region: out of scope at once (no dwell), and the scope record
// says why. Stepping into the Area brings it into scope.
TEST(ScopeGatingContractTest, ScopeGating_D3_OutsideTheParentRegionIsOutOfScope) {
  // Scope catalog case SP-D3-01 (spatialblend/space/testing/SCOPE_CASES.md).
  auto context = LocalContext::Create();
  Space* space = context->AddSpace();
  space->EnableObservabilityCapture(true);
  auto* tree = space->GetElementTree();
  AddFootprint(tree, 1, kInvalidElementId, Rect(0, 0, 60, 60), GeometryRole::kNavigable);
  AddFootprint(tree, 2, 1, Rect(30, 30, 40, 40), GeometryRole::kContainment);  // the Area
  AddFootprint(tree, 3, 2, Rect(34, 34, 36, 36), GeometryRole::kNavigable);    // chair
  tree->GetElement(3)->SetTag("Sittable");
  auto actor = space->CreateActor({SubjectAnchor("root", {0, 0, 0}),
                                   SubjectAnchor("hips", {0, 0, 0.9})});
  FollowProposals(space, actor);
  space->GetActor(actor)->SetElementId(1);
  space->Bind(actor, "hips", RelationKernelType::kAvatarAffordancePose, "Sittable");

  space->UpdateActorState(actor, Eigen::Vector2d(20.0, 20.0), 0.0);  // room, not the Area
  EXPECT_FALSE(IsActive(space->GetRelationStates(actor), 3,
                        RelationKernelType::kAvatarAffordancePose));
  bool recorded = false;
  for (const auto& c : space->GetBindings(actor)[0].scope_record.candidates)
    if (c.element_id == 3) {
      recorded = true;
      EXPECT_EQ(c.reason, ScopeReason::kExcludedOutsideParentRegion);
    }
  EXPECT_TRUE(recorded);

  space->UpdateActorState(actor, Eigen::Vector2d(32.0, 32.0), 0.0);  // inside the Area
  EXPECT_TRUE(IsActive(space->GetRelationStates(actor), 3,
                       RelationKernelType::kAvatarAffordancePose));
}

// A chair in ANOTHER room is not in scope: the parent region must contain the anchor.
TEST(ScopeGatingContractTest, ScopeGating_D3_ChairInAnotherRoomNotInScope) {
  auto context = LocalContext::Create();
  Space* space = context->AddSpace();
  auto* tree = space->GetElementTree();
  AddFootprint(tree, 1, kInvalidElementId, Rect(0, 0, 100, 40), GeometryRole::kContainment);
  AddFootprint(tree, 2, 1, Rect(0, 0, 40, 40), GeometryRole::kNavigable);
  AddFootprint(tree, 3, 1, Rect(60, 0, 100, 40), GeometryRole::kNavigable);
  AddFootprint(tree, 4, 2, Rect(10, 10, 12, 12), GeometryRole::kNavigable);  // this room
  AddFootprint(tree, 5, 3, Rect(70, 10, 72, 12), GeometryRole::kNavigable);  // the other
  tree->GetElement(4)->SetTag("Sittable");
  tree->GetElement(5)->SetTag("Sittable");
  auto actor = space->CreateActor({SubjectAnchor("root", {0, 0, 0}),
                                   SubjectAnchor("hips", {0, 0, 0.9})});
  FollowProposals(space, actor);
  space->GetActor(actor)->SetElementId(2);
  space->Bind(actor, "hips", RelationKernelType::kAvatarAffordancePose, "Sittable");
  space->UpdateActorState(actor, Eigen::Vector2d(20.0, 20.0), 0.0);
  const auto states = space->GetRelationStates(actor);
  EXPECT_TRUE(IsActive(states, 4, RelationKernelType::kAvatarAffordancePose));
  EXPECT_FALSE(IsActive(states, 5, RelationKernelType::kAvatarAffordancePose));
}

// E1-6: the per-family config keeps the driving / contact labels; the activation pair,
// the near field and the scope radii are not part of it.
TEST_F(NearFieldContractTest, Thresholds_E1_6_SettingThresholdsIsObservable) {
  RelationPhaseConfig strict =
      space_->GetRelationPhaseConfig(RelationKernelType::kAvatarAffordancePose);
  strict.driving_weight = 0.9;
  strict.driving_release = 0.8;
  ASSERT_TRUE(space_->SetRelationPhaseConfig(
                        RelationKernelType::kAvatarAffordancePose, strict)
                  .ok());
  EXPECT_DOUBLE_EQ(space_->GetRelationPhaseConfig(
                             RelationKernelType::kAvatarAffordancePose)
                       .driving_weight,
                   0.9);
  // Per family: Spacing is unchanged.
  EXPECT_DOUBLE_EQ(
      space_->GetRelationPhaseConfig(RelationKernelType::kAvatarSpacing)
          .driving_weight,
      0.50);
  // Invalid thresholds are rejected and the previous ones kept.
  RelationPhaseConfig bad = strict;
  bad.driving_release = 0.95;
  EXPECT_FALSE(space_->SetRelationPhaseConfig(
                         RelationKernelType::kAvatarAffordancePose, bad)
                   .ok());
  EXPECT_DOUBLE_EQ(space_->GetRelationPhaseConfig(
                             RelationKernelType::kAvatarAffordancePose)
                       .driving_release,
                   0.8);
}

// ===========================================================================
// E1-5 (RETIRED): "an inactive relation has zero
// influence on realization". There is no activation gate: every relation in the family's
// scope is sent. What stays true, and is tested here, is that an in-scope AffordancePose
// relation the source anchor does not touch contributes no target.
//
// Source A and receiver B share Room(100) [0,60]^2 with a seat T(300) footprint
// [24,26]^2 and a same-tag seat D(400). The hips anchor stands at (26.5, 25): outside
// both footprints, so neither is touched. The same walk is run with and without D.
// ===========================================================================
struct E15Run {
  std::set<shared_element_id_t> sent_referents;
  std::vector<RealizationTarget> final_targets;
  int empty_snapshots = 0;
};

E15Run RunSeatWalk(bool with_distractor) {
  setenv("SB_V2_MIN_MOVE", "0", 1);
  E15Run run;
  auto context = LocalContext::Create();
  Space* a = context->AddSpace();
  Space* b = context->AddSpace();
  for (Space* s : {a, b}) {
    auto* tree = s->GetElementTree();
    AddFootprint(tree, 1, kInvalidElementId, Rect(0, 0, 60, 60),
                 GeometryRole::kNavigable, 100);
    AddFootprint(tree, 2, 1, Rect(24, 24, 26, 26), GeometryRole::kNavigable, 300);
    tree->GetElement(2)->SetTag("Sittable");
    if (s == b || with_distractor) {
      AddFootprint(tree, 3, 1, Rect(30, 24.5, 31, 25.5),
                   GeometryRole::kNavigable, 400);
      tree->GetElement(3)->SetTag("Sittable");
    }
  }
  const std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", {0, 0, 0}),
                                              SubjectAnchor("hips", {0, 0, 0.0})};
  auto actor = a->CreateActor(anchors);
  FollowProposals(a, actor);
  a->GetActor(actor)->SetElementId(1);
  a->SetHub(context->GetHub());
  auto avatar = b->CreateActor(anchors);
  FollowProposals(b, avatar);
  b->GetActor(avatar)->SetElementId(1);
  a->Bind(actor, "hips", RelationKernelType::kAvatarAffordancePose, "Sittable");
  context->GetHub()->SetRelationUpdateCallback(
      [&run](const RemoteRelationUpdate& u) {
        if (u.payloads.empty()) ++run.empty_snapshots;
        for (const auto& p : u.payloads) run.sent_referents.insert(p.referent_shared_id);
      });

  for (int i = 0; i < 30; ++i) {
    a->UpdateActorState(actor, Eigen::Vector2d(26.5, 25.0 + 0.001 * i), 0.0);
  }
  if (const RealizationResult* r = b->GetRealizationTargets(avatar)) {
    run.final_targets = r->targets;
  }
  context->GetHub()->SetRelationUpdateCallback(nullptr);
  return run;
}

TEST(ScopeGatingContractTest, ScopeGating_E1_5_UntouchedSeatsAreSentButAddNoTarget) {
  const E15Run with = RunSeatWalk(true);
  const E15Run without = RunSeatWalk(false);

  // Both seats are in scope (their parent region, the room, holds the anchor): both are
  // sent.
  EXPECT_EQ(with.sent_referents.count(400), 1u);
  EXPECT_EQ(with.sent_referents.count(300), 1u);
  // Neither is touched, so neither adds an affordance target: the targets equal the
  // run without the second seat.
  for (const auto& t : with.final_targets)
    EXPECT_NE(t.kernel_type, RelationKernelType::kAvatarAffordancePose);
  ASSERT_EQ(with.final_targets.size(), without.final_targets.size());
  ASSERT_FALSE(with.final_targets.empty());
  for (size_t i = 0; i < with.final_targets.size(); ++i) {
    EXPECT_NEAR((with.final_targets[i].position -
                 without.final_targets[i].position).norm(), 0.0, 1e-9);
    EXPECT_NEAR(with.final_targets[i].yaw, without.final_targets[i].yaw, 1e-9);
    EXPECT_NEAR(with.final_targets[i].weight, without.final_targets[i].weight,
                1e-9);
  }
}

// When the last in-scope relation leaves scope the receiver gets one empty
// authoritative snapshot, so no stale target remains (was: "all dormant").
TEST(ScopeGatingContractTest, ScopeGating_E1_5_LeavingScopeReleasesReceiver) {
  setenv("SB_V2_MIN_MOVE", "0", 1);
  auto context = LocalContext::Create();
  Space* a = context->AddSpace();
  Space* b = context->AddSpace();
  for (Space* s : {a, b}) {
    auto* tree = s->GetElementTree();
    AddFootprint(tree, 1, kInvalidElementId, Rect(0, 0, 60, 60),
                 GeometryRole::kNavigable, 100);
    AddFootprint(tree, 2, 1, Rect(24, 24, 26, 26), GeometryRole::kNavigable, 300);
    tree->GetElement(2)->SetTag("Sittable");
  }
  const std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", {0, 0, 0}),
                                              SubjectAnchor("hips", {0, 0, 0.9})};
  auto actor = a->CreateActor(anchors);
  FollowProposals(a, actor);
  a->GetActor(actor)->SetElementId(1);
  a->SetHub(context->GetHub());
  auto avatar = b->CreateActor(anchors);
  FollowProposals(b, avatar);
  b->GetActor(avatar)->SetElementId(1);
  a->Bind(actor, "hips", RelationKernelType::kAvatarAffordancePose, "Sittable");

  std::vector<size_t> sizes;
  context->GetHub()->SetRelationUpdateCallback(
      [&sizes](const RemoteRelationUpdate& u) { sizes.push_back(u.payloads.size()); });
  a->UpdateActorState(actor, Eigen::Vector2d(26.5, 25), 0.0);  // in the room
  ASSERT_FALSE(sizes.empty());
  EXPECT_EQ(sizes.back(), 1u);

  // Out of the room (x > 60): no container holds the anchor, so nothing is in scope.
  // The release is immediate and exactly one empty snapshot is sent.
  a->UpdateActorState(actor, Eigen::Vector2d(65.0, 25), 0.0);
  EXPECT_EQ(sizes.back(), 0u) << "release must be propagated";
  const RealizationResult* released = b->GetRealizationTargets(avatar);
  ASSERT_NE(released, nullptr);
  for (const auto& t : released->targets) {
    EXPECT_NE(t.kernel_type, RelationKernelType::kAvatarAffordancePose)
        << "stale affordance target survived the release";
  }
  const size_t sends = sizes.size();
  a->UpdateActorState(actor, Eigen::Vector2d(66.0, 25), 0.0);
  EXPECT_EQ(sizes.size(), sends) << "only one empty snapshot per release";
  context->GetHub()->SetRelationUpdateCallback(nullptr);
}

// ===========================================================================
// H-2 through the Space API (what sb_set_element_shared_id calls).
// ===========================================================================
// Review probe (E1-5 x H-2): two seats sharing shared id 42 -- A at 0.5 m,
// B at 20 m. Rows are keyed by (binding, element), never by the shared id, so
// the duplicates stay independent: both are in scope (no activation gate)
// and both are sent as their own rows.
TEST(ScopeGatingContractTest, ScopeGating_E1_5_DuplicateSharedIdRowsPerElement) {
  setenv("SB_V2_MIN_MOVE", "0", 1);
  auto context = LocalContext::Create();
  Space* a = context->AddSpace();
  Space* b = context->AddSpace();
  auto* tree = a->GetElementTree();
  AddFootprint(tree, 1, kInvalidElementId, Rect(0, 0, 60, 60),
               GeometryRole::kNavigable, 100);
  AddFootprint(tree, 2, 1, Rect(24, 24, 26, 26), GeometryRole::kNavigable, 42);
  AddFootprint(tree, 3, 1, Rect(46, 24, 48, 26), GeometryRole::kNavigable, 42);
  tree->GetElement(2)->SetTag("Sittable");
  tree->GetElement(3)->SetTag("Sittable");
  AddFootprint(b->GetElementTree(), 1, kInvalidElementId, Rect(0, 0, 60, 60),
               GeometryRole::kNavigable, 100);
  const std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", {0, 0, 0}),
                                              SubjectAnchor("hips", {0, 0, 0})};
  auto actor = a->CreateActor(anchors);
  FollowProposals(a, actor);
  a->GetActor(actor)->SetElementId(1);
  a->SetHub(context->GetHub());
  a->Bind(actor, "hips", RelationKernelType::kAvatarAffordancePose, "Sittable");
  std::vector<element_id_t> sent;
  context->GetHub()->SetRelationUpdateCallback(
      [&](const RemoteRelationUpdate& u) {
        sent.clear();
        for (const auto& p : u.payloads) sent.push_back(p.referent_source_element_id);
      });
  for (int i = 0; i < 10; ++i) {
    a->UpdateActorState(actor, {26.5, 25.0}, 0.0);  // 0.5 m from A, 19.5 from B
    const auto states = a->GetRelationStates(actor);
    EXPECT_TRUE(IsActive(states, 2, RelationKernelType::kAvatarAffordancePose))
        << "frame " << i;
    EXPECT_TRUE(IsActive(states, 3, RelationKernelType::kAvatarAffordancePose));
    ASSERT_EQ(sent.size(), 2u) << "frame " << i;
    std::sort(sent.begin(), sent.end());
    EXPECT_EQ(sent[0], 2);
    EXPECT_EQ(sent[1], 3);
  }
  context->GetHub()->SetRelationUpdateCallback(nullptr);
}

// ===========================================================================
// Facing scope: visible, with no angular
// cutoff. The object-distance radius is OFF by default (SpatialBlend: visible
// "regardless of distance"); a PP-style scenario may opt into a declared
// radius (here 3.66 m = 12 ft) via SetRelationPhaseConfig.
//
// Typical living room [0,5]x[0,4], chest at (2.5, 2, 1.2), heading +y:
//   sofa (2.5,0.5), coffee table (2.5,1.3), TV (2.5,3.9), armchair (0.6,2),
//   bookshelf (4.8,2), floor lamp (0.3,3.7), side table (4.6,0.4)
// plus a low side cabinet (6.5,0.35), ~3.96 m from the chest and clear of
// every occluder's line of sight.
// ===========================================================================
namespace {
struct FacingScopeFixture {
  std::unique_ptr<LocalContext> context = LocalContext::Create();
  Space* space = context->AddSpace();
  ElementTree* tree = space->GetElementTree();
  spatialblend::entity_id_t actor;
  struct Item { element_id_t id; double cx, cy, hx, hy, h; };
  const std::vector<Item> items = {
      {2, 2.5, 0.5, 1.0, 0.45, 0.8},  {3, 2.5, 1.3, 0.5, 0.3, 0.45},
      {4, 2.5, 3.9, 0.6, 0.05, 1.2},  {5, 0.6, 2.0, 0.4, 0.4, 0.9},
      {6, 4.8, 2.0, 0.2, 0.6, 2.0},   {7, 0.3, 3.7, 0.15, 0.15, 1.6},
      {8, 4.6, 0.4, 0.25, 0.25, 0.6}, {9, 6.5, 0.35, 0.3, 0.3, 1.0}};

  FacingScopeFixture() {
    AddFootprint(tree, 1, kInvalidElementId, Rect(0, 0, 12, 4),
                 GeometryRole::kNavigable);
    for (const auto& it : items) AddBoxXY(tree, it.id, 1, it.cx, it.cy, it.hx, it.hy, it.h);
    actor = space->CreateActor({SubjectAnchor("root", {0, 0, 0}),
                                SubjectAnchor("chest", {0, 0, 1.2})});
    FollowProposals(space, actor);
    // A sibling of the items, not their ancestor.
    EXPECT_TRUE(tree->AddElement(20, 1).ok());
    space->GetActor(actor)->SetElementId(20);
  }

  // Bind and extract.
  void BindAndUpdate() {
    space->Bind(actor, "chest", RelationKernelType::kAvatarFacing);
    space->UpdateActorState(actor, {2.5, 2.0}, M_PI / 2);
  }

  // Point-to-rectangle distance from (2.5, 2) <= radius (radius <= 0 = all).
  // No item lies between the chest and another item above 1.2 m, so every
  // in-range item is visible.
  std::set<element_id_t> InRange(double radius) const {
    std::set<element_id_t> in_range;
    for (const auto& it : items) {
      const double dx = std::max(0.0, std::abs(2.5 - it.cx) - it.hx);
      const double dy = std::max(0.0, std::abs(2.0 - it.cy) - it.hy);
      if (radius <= 0.0 || std::hypot(dx, dy) <= radius) in_range.insert(it.id);
    }
    return in_range;
  }
};
}  // namespace

TEST(ScopeGatingContractTest,
     ScopeGating_A2_FacingDefaultUnboundedNoAngularCutoff) {
  FacingScopeFixture f;
  f.BindAndUpdate();
  // Off by default, so the 8.5 m cabinet stays in scope too.
  const std::set<element_id_t> in_range = f.InRange(/*radius=*/0.0);
  ASSERT_EQ(in_range.size(), f.items.size());
  const auto states = f.space->GetRelationStates(f.actor);
  int active = 0;
  for (const auto& st : states) {
    EXPECT_TRUE(in_range.count(st.referent_element_id))
        << "element " << st.referent_element_id << " unexpectedly out of scope";
    if (st.active) ++active;
  }
  // The sofa is directly behind the actor (heading +y, sofa at -y): R ~ 0,
  // still active -- the realization policy, not an R floor, weighs it.
  EXPECT_TRUE(IsActive(states, 2, RelationKernelType::kAvatarFacing));
  EXPECT_EQ(active, static_cast<int>(in_range.size()));
  RecordProperty("facing_active_relations", active);
  std::cout << "[dilution] typical room, unbounded: " << active
            << " facing relations active of " << f.items.size() << " objects\n";
}

// There is no max_scope_distance_m field for Facing;
// ScopeGating_A2_FacingDefaultUnboundedNoAngularCutoff above is the whole contract.

// ===========================================================================
// E1-2, REAL obstacle-sealed seat: obstacle-aware free space via
// NavigableSpatialDefinition::QueryReachability, with Space syncing the
// Obstacle elements into the navigable. Also covers that the
// sealed seat is never scored by a Euclidean geodesic fallback.
//
//   Room(1) [0,10]^2; Seat(2) box at (7,7); four 1 m wall volumes forming a
//   closed ring [5.5,8.5]^2 (0.2 m thick) around it. Anchor at (2,2).
// ===========================================================================
TEST(ScopeGatingContractTest,
     ScopeGating_E1_2_ObstacleRingSealsSeat) {
  auto context = LocalContext::Create();
  Space* space = context->AddSpace();
  auto* tree = space->GetElementTree();
  AddFootprint(tree, 1, kInvalidElementId, Rect(0, 0, 10, 10),
               GeometryRole::kNavigable);
  AddBoxXY(tree, 2, 1, 7, 7, 0.3, 0.3, 0.45, "Seat");
  AddBoxXY(tree, 3, 1, 7, 5.6, 1.5, 0.1, 1.0);   // south wall
  AddBoxXY(tree, 4, 1, 7, 8.4, 1.5, 0.1, 1.0);   // north wall
  AddBoxXY(tree, 5, 1, 5.6, 7, 0.1, 1.5, 1.0);   // west wall
  AddBoxXY(tree, 6, 1, 8.4, 7, 0.1, 1.5, 1.0);   // east wall
  AddBoxXY(tree, 7, 1, 3, 8, 0.3, 0.3, 0.45, "Seat");  // open seat (control)
  auto actor = space->CreateActor({SubjectAnchor("root", {0, 0, 0})});
  FollowProposals(space, actor);
  space->GetActor(actor)->SetElementId(1);
  space->Bind(actor, "root", RelationKernelType::kAvatarSpacing);
  space->EnableObservabilityCapture(true);
  space->UpdateActorState(actor, {2, 2}, 0.0);
  const auto states = space->GetRelationStates(actor);
  for (const auto& st : states) {
    EXPECT_NE(st.referent_element_id, 2) << "sealed seat must be out of scope";
  }
  EXPECT_TRUE(IsActive(states, 7, RelationKernelType::kAvatarSpacing));

  // Independent oracle: the ring encloses the seat on all four sides (every
  // wall spans the full [5.5, 8.5] side), so no path exists; the sealed
  // seat's source geodesic is +inf (not a Euclidean ~6.4 m).
  const auto* room = dynamic_cast<const NavigableSpatialDefinition*>(
      tree->GetElement(1)->GetSpatialDefinition());
  ASSERT_NE(room, nullptr);
  EXPECT_TRUE(std::isinf(room->GetGeodesicDistance({2, 2, 0}, {6.7, 6.7, 0})));
  EXPECT_TRUE(std::isfinite(room->GetGeodesicDistance({2, 2, 0}, {2.7, 7.7, 0})));

  // The exclusion is recorded with its reason.
  const auto& record = space->GetBindings(actor)[0].scope_record;
  bool recorded = false;
  for (const auto& c : record.candidates)
    if (c.element_id == 2) {
      recorded = true;
      EXPECT_EQ(c.reason, ScopeReason::kExcludedUnreachable);
    }
  EXPECT_TRUE(recorded);
}

// The scope record for the gating reasons that remain: not visible (D1), outside the parent object
// region (D3). The beyond-radius, outside-near-field and gated-inactive reasons are
// gone with the radii, the near field and the activation gate.
TEST(ScopeGatingContractTest, ScopeRecord_E1_7_GatingReasons) {
  auto context = LocalContext::Create();
  Space* space = context->AddSpace();
  space->EnableObservabilityCapture(true);
  auto* tree = space->GetElementTree();
  AddFootprint(tree, 1, kInvalidElementId, Rect(0, 0, 20, 20),
               GeometryRole::kNavigable);
  AddBoxXY(tree, 2, 1, 3.0, 2.0, 0.2, 0.2, 1.0, "Thing");   // visible, 1 m
  AddBoxXY(tree, 3, 1, 2.0, 4.0, 0.2, 0.2, 1.0, "Thing");   // behind wall
  AddBoxXY(tree, 4, 1, 2.0, 3.0, 1.0, 0.1, 2.5);            // wall
  AddBoxXY(tree, 5, 1, 9.0, 2.0, 0.2, 0.2, 1.0, "Thing");   // 6.8 m away
  AddFootprint(tree, 6, 1, Rect(4.5, 1.5, 5.5, 2.5), GeometryRole::kNavigable);
  tree->GetElement(6)->SetTag("Sittable");                  // seat, 2.5 m
  auto actor = space->CreateActor({SubjectAnchor("root", {0, 0, 0}),
                                   SubjectAnchor("chest", {0, 0, 1.2})});
  FollowProposals(space, actor);
  ASSERT_TRUE(tree->AddElement(9, 1).ok());
  space->GetActor(actor)->SetElementId(9);
  space->Bind(actor, "chest", RelationKernelType::kAvatarFacing);
  space->Bind(actor, "root", RelationKernelType::kAvatarAffordancePose,
              "Sittable");
  space->UpdateActorState(actor, {2.0, 2.0}, 0.0);

  auto reason_of = [&](size_t b, element_id_t id) {
    for (const auto& c : space->GetBindings(actor)[b].scope_record.candidates)
      if (c.element_id == id) return static_cast<int>(c.reason);
    return -1;
  };
  EXPECT_EQ(reason_of(0, 2), static_cast<int>(ScopeReason::kIncluded));
  EXPECT_EQ(reason_of(0, 3), static_cast<int>(ScopeReason::kExcludedNotVisible));
  // 6.8 m away and visible. Facing has no distance cap.
  EXPECT_EQ(reason_of(0, 5), static_cast<int>(ScopeReason::kIncluded));
  // Seat 2.5 m away, in the room that holds the anchor: in scope (D3; it was withheld
  // by the 1.5 m near field).
  EXPECT_EQ(reason_of(1, 6), static_cast<int>(ScopeReason::kIncluded));
  for (const auto& c : space->GetBindings(actor)[1].scope_record.candidates)
    if (c.element_id == 6) EXPECT_FALSE(std::isnan(c.proximity));
}

// There is no release dwell: a relation leaves with its scope.

// A missing anchor observation (the frame omits the anchor: a tracking dropout) withdraws
// the seat AT ONCE. There is no tracking-gap dwell holding the seat's realization through a
// short dropout; the observed source state is truthful ("a body sample missing from the current frame cannot borrow the
// previous contact"). An explicit deactivation withdraws at once as well.
class ObservationGapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("SB_V2_MIN_MOVE", "0", 1);
    context_ = LocalContext::Create();
    a_ = context_->AddSpace();
    b_ = context_->AddSpace();
    for (Space* s : {a_, b_}) {
      auto* tree = s->GetElementTree();
      AddFootprint(tree, 1, kInvalidElementId, Rect(0, 0, 60, 60),
                   GeometryRole::kNavigable, 100);
      AddFootprint(tree, 2, 1, Rect(24, 24, 26, 26), GeometryRole::kNavigable,
                   300);
      tree->GetElement(2)->SetTag("Sittable");
    }
    const std::vector<SubjectAnchor> anchors = {
        SubjectAnchor("root", {0, 0, 0}), SubjectAnchor("hips", {0, 0, 0})};
    actor_ = a_->CreateActor(anchors);
    FollowProposals(a_, actor_);
    a_->GetActor(actor_)->SetElementId(1);
    a_->SetHub(context_->GetHub());
    avatar_ = b_->CreateActor(anchors);
    FollowProposals(b_, avatar_);
    b_->GetActor(avatar_)->SetElementId(1);
    // The seated hips touch this floor-level seat footprint (z = 0).
    a_->Bind(actor_, "hips", RelationKernelType::kAvatarAffordancePose,
             "Sittable");
    context_->GetHub()->SetRelationUpdateCallback(
        [this](const RemoteRelationUpdate& u) {
          seat_sent_ = false;
          for (const auto& p : u.payloads)
            if (p.referent_shared_id == 300) seat_sent_ = true;
        });
    ASSERT_TRUE(a_->BeginActorObservationEpoch(actor_, "root", &epoch_));
    for (int i = 0; i < 3; ++i) Frame(true);  // seated: Active -> Contact
    ASSERT_EQ(PublicPhase(), RelationPhase::kContact);
    ASSERT_TRUE(ReceiverHasSeat());
  }
  void TearDown() override {
    context_->GetHub()->SetRelationUpdateCallback(nullptr);
  }

  void Frame(bool with_hips) {
    std::vector<Space::ActorObservation> obs = {{"root", {25, 25, 0}, 0}};
    if (with_hips) obs.push_back({"hips", {25, 25, 0.0}, 0});
    ASSERT_TRUE(a_->ApplyActorObservationFrame(actor_, epoch_, sequence_,
                                               sequence_ / 30.0, obs));
    ++sequence_;
  }
  // The public lifecycle phase of the seat relation (empty while the anchor is absent).
  RelationPhase PublicPhase() {
    for (const auto& r : a_->GetRelationStates(actor_))
      if (r.referent_element_id == 2) return r.phase;
    return RelationPhase::kDormant;
  }
  bool ReceiverHasSeat() {
    const RealizationResult* r = b_->GetRealizationTargets(avatar_);
    if (!r) return false;
    for (const auto& t : r->targets)
      if (t.kernel_type == RelationKernelType::kAvatarAffordancePose)
        return true;
    return false;
  }

  std::unique_ptr<LocalContext> context_;
  Space* a_ = nullptr;
  Space* b_ = nullptr;
  spatialblend::entity_id_t actor_ = 0, avatar_ = 0;
  uint64_t epoch_ = 0;
  int64_t sequence_ = 0;
  bool seat_sent_ = false;
};

TEST_F(ObservationGapTest, ScopeGating_D7_TrackingGapReleasesAtOnce) {
  // The first frame without the hips withdraws the seat: not sent, no receiver target,
  // no relation row (the observed state is not borrowed from the previous sample).
  Frame(false);
  EXPECT_FALSE(seat_sent_) << "the seat must not be held through a gap";
  EXPECT_FALSE(ReceiverHasSeat());
  EXPECT_EQ(PublicPhase(), RelationPhase::kDormant);
  EXPECT_TRUE(a_->GetRelationStates(actor_).empty());
  // The anchor returns: the seat is observed and sent again.
  Frame(true);
  EXPECT_TRUE(seat_sent_);
}

TEST_F(ObservationGapTest, ScopeGating_E1_3_ExplicitDeactivationWithdrawsAtOnce) {
  ASSERT_TRUE(a_->SetAnchorActive(actor_, "hips", false));
  Frame(false);
  EXPECT_FALSE(seat_sent_);
  EXPECT_EQ(PublicPhase(), RelationPhase::kDormant);
  EXPECT_FALSE(ReceiverHasSeat());
  // An explicit clear of an observed anchor is explicit too.
  Frame(true);
  Frame(true);
  ASSERT_TRUE(seat_sent_);
  ASSERT_TRUE(a_->ClearActorObservationAnchor(actor_, epoch_, "hips"));
  EXPECT_FALSE(seat_sent_);
  EXPECT_EQ(PublicPhase(), RelationPhase::kDormant);
}

// Another avatar's part is a Facing referent at any distance, like an object (there is no
// object or actor radius). Its element itself is never a referent
// (actors are subjects, space.cc), so the referent is a part under it -- here the remote
// avatar's head, 8 m away.
TEST(ScopeGatingContractTest, ScopeGating_E1_1_RemoteAvatarAtAnyDistance) {
  auto context = LocalContext::Create();
  Space* space = context->AddSpace();
  auto* tree = space->GetElementTree();
  AddFootprint(tree, 1, kInvalidElementId, Rect(0, 0, 20, 20),
               GeometryRole::kNavigable);
  ASSERT_TRUE(tree->AddElement(60, 1).ok());           // remote avatar element
  AddBoxXY(tree, 61, 60, 10.0, 2.0, 0.1, 0.1, 1.8, "Head");  // its head, 8 m
  AddBoxXY(tree, 62, 1, 2.0, 10.0, 0.1, 0.1, 1.8, "Head");   // an object, 8 m
  auto remote = space->CreateActor({SubjectAnchor("root", {0, 0, 0})});
  FollowProposals(space, remote);
  space->GetActor(remote)->SetElementId(60);
  auto actor = space->CreateActor({SubjectAnchor("root", {0, 0, 0}),
                                   SubjectAnchor("chest", {0, 0, 1.2})});
  FollowProposals(space, actor);
  ASSERT_TRUE(tree->AddElement(70, 1).ok());
  space->GetActor(actor)->SetElementId(70);
  space->Bind(actor, "chest", RelationKernelType::kAvatarFacing);
  space->UpdateActorState(actor, {2.0, 2.0}, 0.0);
  const auto states = space->GetRelationStates(actor);
  EXPECT_TRUE(IsActive(states, 61, RelationKernelType::kAvatarFacing))
      << "8 m remote avatar must stay active";
  EXPECT_TRUE(IsActive(states, 62, RelationKernelType::kAvatarFacing))
      << "a visible object stays in scope at any distance";
}

TEST(ScopeGatingContractTest, UniqueIdentity_H_2_DuplicateResolvesByDeclaredRule) {
  auto context = LocalContext::Create();
  Space* space = context->AddSpace();
  ASSERT_TRUE(space->AddElement(1));
  ASSERT_TRUE(space->AddElement(2, 1));
  ASSERT_TRUE(space->AddElement(3, 1));
  // Registered in descending id order: resolution still picks element 2.
  ASSERT_TRUE(space->SetElementSharedId(3, 500));
  ASSERT_TRUE(space->SetElementSharedId(2, 500));
  for (int i = 0; i < 3; ++i) EXPECT_EQ(space->FindElementBySharedId(500), 2);
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
