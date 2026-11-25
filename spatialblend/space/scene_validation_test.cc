// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Scene validation (element/containment_layout.h) at the Space registration API.
#include <gtest/gtest.h>

#include <chrono>
#include <atomic>
#include <future>
#include <stdexcept>
#include <thread>
#include <memory>
#include <string>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/model/element/scope_extension.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/model/relation/spacing_grounding_rule.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/scope_gate.h"
#include "spatialblend/model/relation/scope_resolver.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace test {
namespace {

std::shared_ptr<SpatialDefinition> Region(double x0, double y0, double x1, double y1) {
  return std::make_shared<NavigableSpatialDefinition>(
      std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
          {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}),
      nullptr, 0.0, GeometryRole::kContainment);
}

class SceneValidation : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = LocalContext::Create();
    space_ = context_->AddSpace();
    ASSERT_TRUE(space_->AddElement(1));  // room
    ASSERT_TRUE(space_->SetElementSpatialDefinition(
        1, std::make_shared<NavigableSpatialDefinition>(
               std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
                   {0, 0}, {20, 0}, {20, 20}, {0, 20}}))));
    ASSERT_TRUE(space_->AddElement(2, 1));
    ASSERT_TRUE(space_->AddElement(3, 1));
  }
  // Whether the element's SCOPE region holds the point (FootprintHolds: the declared region and its
  // children's scope regions, no bridge or corridor).
  bool Holds(element_id_t id, const Eigen::Vector2d& p) {
    const ElementTree& tree = *space_->GetElementTree();
    return FootprintHolds(*tree.GetElement(id), tree, Eigen::Vector3d(p.x(), p.y(), 0));
  }
  // Whether its WALKABLE region (the effective definition: the scope plus the bridges and corridors)
  // holds the point.
  bool Walks(element_id_t id, const Eigen::Vector2d& p) {
    const ElementTree& tree = *space_->GetElementTree();
    const Element& e = *tree.GetElement(id);
    return FootprintHoldsIn(e.GetSpatialDefinition(), e, tree, Eigen::Vector3d(p.x(), p.y(), 0));
  }
  std::shared_ptr<LocalContext> context_;
  Space* space_ = nullptr;
};

std::shared_ptr<SpatialDefinition> Box(double x0, double y0, double x1, double y1) {
  return std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(x0, y0, 0), Eigen::Vector3d(x1, y1, 1), GeometryRole::kContainment);
}

Eigen::Matrix4d At(double x, double y) {
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = x;
  t(1, 3) = y;
  return t;
}

TEST_F(SceneValidation, OverlappingSiblingsAreRejectedNamingBoth) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2, 2, 8, 8)));
  EXPECT_FALSE(space_->SetElementSpatialDefinition(3, Region(5, 2, 12, 8)));
  const auto err = space_->LastSceneError();
  EXPECT_EQ(err.code, SceneErrorCode::kContainmentOverlap);
  EXPECT_NE(err.message.find("element 2"), std::string::npos) << err.message;
  EXPECT_NE(err.message.find("element 3"), std::string::npos) << err.message;
  // Not registered, and nothing falls back at solve time.
  EXPECT_EQ(space_->GetElementTree()->GetElement(3)->GetSpatialDefinition(), nullptr);
}

TEST_F(SceneValidation, TouchingEdgesAreAccepted) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2, 2, 8, 8)));
  EXPECT_TRUE(space_->SetElementSpatialDefinition(3, Region(8, 2, 12, 8)));
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk);
  // Touching at a single corner too.
  ASSERT_TRUE(space_->AddElement(4, 1));
  EXPECT_TRUE(space_->SetElementSpatialDefinition(4, Region(12, 8, 15, 11)));
}

// A nearby child that sticks out of its parent is accepted and extends the parent's scope.
TEST_F(SceneValidation, NearbyChildThatSticksOutExtendsTheParentScope) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2, 2, 8, 8)));
  ASSERT_TRUE(space_->AddElement(4, 2));
  EXPECT_TRUE(space_->SetElementSpatialDefinition(4, Region(6, 4, 9, 6)));
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk);
  ASSERT_TRUE(space_->WaitForScopeExtension());
  const ElementTree& tree = *space_->GetElementTree();
  const Element* parent = tree.GetElement(2);
  EXPECT_TRUE(parent->HasScopeExtension());
  // The declared region is what the app registered; the effective one is the union.
  EXPECT_FALSE(parent->GetDeclaredSpatialDefinition()->Contains({8.5, 5}));
  EXPECT_TRUE(Holds(2, {8.5, 5}));
  EXPECT_NEAR(parent->GetSpatialDefinition()->GetBoundary()->GetArea(), 36.0 + 2.0, 1e-9);
  // A point in the extension resolves to the parent's scope (and into the child below it).
  const ScopeDescent d = DescendScope({8.5, 5, 0}, 1, tree);
  EXPECT_EQ(d.container, 4);
  EXPECT_FALSE(Holds(2, {8.5, 7}));  // outside both
  EXPECT_EQ(FindDeepestContaining({8.5, 7, 0}, 1, tree), 1);
}

// The NavMesh carve band: a child separated from the parent's ring by a small gap is
// bridged, so the parent's region (and the walkable the path planner uses) stays connected.
TEST_F(SceneValidation, ChildAcrossACarveGapIsBridgedIntoOneConnectedRegion) {
  // The room's walkable ring ends at x = 20; the child's stand-off band starts 0.2 m
  // beyond it (the NavMesh carve band), and reaches 2 m past the ring.
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(20.2, 5, 22, 7)));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  const ElementTree& tree = *space_->GetElementTree();
  const auto* room = dynamic_cast<const NavigableSpatialDefinition*>(
      tree.GetElement(1)->GetSpatialDefinition());
  ASSERT_NE(room, nullptr);
  ASSERT_TRUE(tree.GetElement(1)->HasScopeExtension());
  // One connected walkable region, not two islands: the band is filled.
  EXPECT_EQ(room->GetWalkablePolygon()->NumComponents(), 1u);
  // The band is walkable, not in the room's scope (bridges are only walkable).
  for (double y : {5.0, 5.5, 6.0, 6.5, 7.0}) {
    EXPECT_TRUE(Walks(1, {20.1, y})) << y;
    EXPECT_FALSE(Holds(1, {20.1, y})) << y;
  }
  EXPECT_TRUE(Holds(1, {21.0, 6})) << "the child is in the room's scope";
  EXPECT_FALSE(Walks(1, {20.1, 4.5}));  // nothing farther than the gap from both is added
  EXPECT_FALSE(Walks(1, {20.1, 8.0}));
  // The bridge resolves to the room; the band beyond it to the child.
  EXPECT_EQ(FindDeepestContaining({20.1, 6, 0}, 1, tree), 1);
  EXPECT_EQ(FindDeepestContaining({21.0, 6, 0}, 1, tree), 2);
  // Path planning sees the extended walkable: the child's band is reachable from the room.
  EXPECT_TRUE(room->IsReachable({10, 6}, {21.5, 6}, 1e-6, 0.0));
}

// A far child extends its parent too, whatever the gap: bridged to the parent's nearest point, one connected region.
TEST_F(SceneValidation, FarChildExtendsTheParentScopeAcrossTheGap) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2, 2, 8, 8)));
  ASSERT_TRUE(space_->AddElement(4, 2));
  // 4 m beyond the parent (x 12..13).
  EXPECT_TRUE(space_->SetElementSpatialDefinition(4, Region(12, 4, 13, 6)));
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk);
  ASSERT_TRUE(space_->WaitForScopeExtension());
  const ElementTree& tree = *space_->GetElementTree();
  ASSERT_TRUE(tree.GetElement(2)->HasScopeExtension());
  EXPECT_EQ(tree.GetElement(2)->GetSpatialDefinition()->GetBoundary()->NumComponents(), 1u);
  EXPECT_EQ(tree.GetElement(2)->GetScopeDefinition()->GetBoundary()->NumComponents(), 2u)
      << "the scope is the region and the child, no corridor";
  EXPECT_TRUE(Holds(2, {12.5, 5}));  // the child
  for (double x : {8.5, 10.0, 11.5}) {  // the bridge: walkable, not scope
    EXPECT_TRUE(Walks(2, {x, 5})) << x;
    EXPECT_FALSE(Holds(2, {x, 5})) << x;
  }
  EXPECT_FALSE(Walks(2, {14.5, 5}));  // beyond the child: nothing added
  // The bridge is a corridor as wide as the child (y 4..6), not
  // the points within the 4 m gap of both regions (a slab reaching y 0..10).
  EXPECT_FALSE(Walks(2, {10, 9.5}));
  EXPECT_FALSE(Walks(2, {10, 6.5}));
  EXPECT_FALSE(Walks(2, {10, 3.5}));
  EXPECT_EQ(DescendScope({12.5, 5, 0}, 1, tree).container, 4);
  EXPECT_EQ(FindDeepestContaining({10, 5, 0}, 1, tree), 1) << "the corridor is the room's";
}

TEST_F(SceneValidation, AnExtensionMustNotOverlapAFootprintOutsideItsLine) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2, 2, 8, 8)));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(3, Region(8, 2, 12, 8)));  // touches 2
  ASSERT_TRUE(space_->AddElement(4, 2));
  // Sticks out of 2 into its sibling 3: the extended regions would overlap.
  EXPECT_FALSE(space_->SetElementSpatialDefinition(4, Region(7, 4, 9, 6)));
  const auto err = space_->LastSceneError();
  EXPECT_EQ(err.code, SceneErrorCode::kContainmentOverlap);
  EXPECT_NE(err.message.find("element 4"), std::string::npos) << err.message;
  EXPECT_NE(err.message.find("element 3"), std::string::npos) << err.message;
}

TEST_F(SceneValidation, NestingAndParentEdgeContactAreAccepted) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2, 2, 8, 8)));
  ASSERT_TRUE(space_->AddElement(4, 2));
  EXPECT_TRUE(space_->SetElementSpatialDefinition(4, Region(2, 2, 5, 5)));  // shares edges
  ASSERT_TRUE(space_->AddElement(5, 4));
  EXPECT_TRUE(space_->SetElementSpatialDefinition(5, Region(3, 3, 4, 4)));
}

TEST_F(SceneValidation, MovingAnElementIntoAnotherIsRejectedAndNotApplied) {
  // Polytope regions keep their geometry in the element's local frame.
  const auto box = [](double x0, double y0, double x1, double y1) {
    return std::make_shared<PolytopeSpatialDefinition>(
        Eigen::Vector3d(x0, y0, 0), Eigen::Vector3d(x1, y1, 1),
        GeometryRole::kContainment);
  };
  const auto at_x = [](double x) {
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 3) = x;
    return t;
  };
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, box(0, 0, 4, 4)));
  ASSERT_TRUE(space_->SetElementTransform(3, at_x(5.0)));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(3, box(0, 0, 4, 4)));
  EXPECT_FALSE(space_->SetElementTransform(3, at_x(2.0)));  // overlaps element 2
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kContainmentOverlap);
  EXPECT_EQ(space_->GetElementTree()->GetElement(3)->GetLocalTransform(), at_x(5.0));
  EXPECT_TRUE(space_->SetElementTransform(3, at_x(4.0)));  // edge contact
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk);
}

// A child that extends its parent, as a movable box (element-local geometry).
class ScopeExtension : public SceneValidation {
 protected:
  void SetUp() override {
    SceneValidation::SetUp();
    ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2, 2, 8, 8)));
    ASSERT_TRUE(space_->AddElement(4, 2));
    ASSERT_TRUE(space_->SetElementTransform(4, At(7, 4)));
    ASSERT_TRUE(space_->SetElementSpatialDefinition(4, Box(0, 0, 2, 2)));  // x 7..9
    ASSERT_TRUE(space_->WaitForScopeExtension());
    ASSERT_TRUE(Holds(2, {8.5, 5}));
  }
};

// Moved 1 m beyond
// the parent, the child is applied and the extension follows it, bridged across the gap.
TEST_F(ScopeExtension, MovingTheChildFarExtendsAcrossTheGap) {
  const uint64_t revision = space_->GetScopeExtensionRevision();
  EXPECT_TRUE(space_->SetElementTransform(4, At(9, 4)));  // x 9..11: 1 m beyond the parent
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk);
  EXPECT_EQ(space_->GetElementTree()->GetElement(4)->GetLocalTransform(), At(9, 4));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_GT(space_->GetScopeExtensionRevision(), revision);
  EXPECT_TRUE(Holds(2, {10, 5}));   // the child
  EXPECT_TRUE(Walks(2, {8.5, 5}));  // the bridge across the 1 m gap: walkable, not scope
  EXPECT_FALSE(Holds(2, {8.5, 5}));
  EXPECT_FALSE(Walks(2, {8.5, 2.5}));  // nothing farther than the gap from both is added
  EXPECT_FALSE(Walks(2, {11.5, 5}));
}

TEST_F(ScopeExtension, MovingANonActorChildUpdatesTheScope) {
  ASSERT_TRUE(space_->SetElementTransform(4, At(4, 7)));  // y 7..9: sticks out at the top
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_FALSE(Holds(2, {8.5, 5}));  // the old extension is gone
  EXPECT_TRUE(Holds(2, {5, 8.5}));
  // Moving back inside shrinks the scope to the declared region.
  ASSERT_TRUE(space_->SetElementTransform(4, At(4, 4)));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_FALSE(space_->GetElementTree()->GetElement(2)->HasScopeExtension());
  EXPECT_FALSE(Holds(2, {5, 8.5}));
}

TEST_F(ScopeExtension, RemovingTheChildShrinksTheScopeBack) {
  ASSERT_TRUE(space_->RemoveLocalElementSubtree(4));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_FALSE(space_->GetElementTree()->GetElement(2)->HasScopeExtension());
  EXPECT_FALSE(Holds(2, {8.5, 5}));
  EXPECT_TRUE(Holds(2, {7.5, 5}));
}

TEST_F(ScopeExtension, AnActorNeverExtendsAParent) {
  const auto actor = space_->CreateActor({SubjectAnchor("root")});
  // An actor's element with a region inside the parent, then moved so it sticks out.
  ASSERT_TRUE(space_->AddElement(6, 2));
  ASSERT_TRUE(space_->SetElementTransform(6, At(3, 3)));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(6, Box(0, 0, 1, 1)));
  ASSERT_TRUE(space_->SetActorElement(actor, 6));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  const uint64_t revision = space_->GetScopeExtensionRevision();
  // 0.2 m out: an actor's element is not checked against its parent (its pose never is either)
  // and never extends it.
  EXPECT_TRUE(space_->SetElementTransform(6, At(1.8, 3)));
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk);
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_EQ(space_->GetScopeExtensionRevision(), revision);
  EXPECT_FALSE(Holds(2, {1.9, 3.5}));
  // An actor moving inside the parent does not reshape it.
  ASSERT_TRUE(space_->SetElementTransform(6, At(4, 3)));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_EQ(space_->GetScopeExtensionRevision(), revision);
  // The rule is the element's kind: the extending box made an actor's element stops extending.
  ASSERT_TRUE(space_->SetActorElement(actor, 4));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_FALSE(space_->GetElementTree()->GetElement(2)->HasScopeExtension());
  EXPECT_FALSE(Holds(2, {8.5, 5}));
}

TEST_F(ScopeExtension, QueriesDuringARebuildReadThePreviousScope) {
  std::promise<void> entered, release;
  std::shared_future<void> gate = release.get_future().share();
  bool first = true;
  space_->SetScopeExtensionBuildHookForTesting([&, gate] {
    if (first) {
      first = false;
      entered.set_value();
      gate.wait();
    }
  });
  const uint64_t revision = space_->GetScopeExtensionRevision();
  ASSERT_TRUE(space_->SetElementTransform(4, At(4, 7)));  // the new pose is applied at once
  entered.get_future().wait();
  // The rebuild is running: the scope is the last complete one, whole.
  EXPECT_EQ(space_->GetElementTree()->GetElement(4)->GetLocalTransform(), At(4, 7));
  EXPECT_TRUE(Holds(2, {8.5, 5}));
  EXPECT_FALSE(Holds(2, {5, 8.5}));
  EXPECT_EQ(space_->GetScopeExtensionRevision(), revision);
  release.set_value();
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_GT(space_->GetScopeExtensionRevision(), revision);
  EXPECT_FALSE(Holds(2, {8.5, 5}));
  EXPECT_TRUE(Holds(2, {5, 8.5}));
  space_->SetScopeExtensionBuildHookForTesting(nullptr);
}

TEST_F(ScopeExtension, ABurstOfMovesCoalescesAndEndsAtTheLatestPose) {
  std::promise<void> entered, release;
  std::shared_future<void> gate = release.get_future().share();
  bool first = true;
  space_->SetScopeExtensionBuildHookForTesting([&, gate] {
    if (first) {
      first = false;
      entered.set_value();
      gate.wait();
    }
  });
  const uint64_t builds = space_->GetScopeExtensionBuildCount();
  ASSERT_TRUE(space_->SetElementTransform(4, At(7, 2.0)));
  entered.get_future().wait();
  // A drag: many moves while the first rebuild runs.
  for (int i = 1; i <= 50; ++i) ASSERT_TRUE(space_->SetElementTransform(4, At(7, 2.0 + 0.08 * i)));
  release.set_value();
  ASSERT_TRUE(space_->WaitForScopeExtension());
  // The blocked rebuild and ONE more from the latest state, not 51.
  EXPECT_LE(space_->GetScopeExtensionBuildCount() - builds, 2u);
  // The final scope matches the latest pose (y 6..8).
  EXPECT_TRUE(Holds(2, {8.5, 7.5}));
  EXPECT_FALSE(Holds(2, {8.5, 3}));
  space_->SetScopeExtensionBuildHookForTesting(nullptr);
}

TEST_F(ScopeExtension, ConcurrentMovesAndQueriesAreSafe) {
  // An actor inside the parent: its course is planned in the parent's (effective) walkable,
  // which the worker swaps while this thread plans.
  const auto actor = space_->CreateActor({SubjectAnchor("root")});
  ASSERT_TRUE(space_->AddElement(7, 2));
  ASSERT_TRUE(space_->SetActorElement(actor, 7));
  std::atomic<bool> stop{false};
  std::thread reader([&] {
    while (!stop.load()) {
      EXPECT_GE(space_->QueryCourse(actor, {3, 3}, {4, 4}).size(), 2u);
      (void)space_->QueryCourse(actor, {3, 3}, {8.5, 5});
      (void)space_->GetScopeExtensionRevision();
    }
  });
  for (int i = 0; i < 200; ++i) {
    ASSERT_TRUE(space_->SetElementTransform(4, At(7, 2.0 + 0.02 * (i % 100))));
    if (i % 50 == 0) (void)space_->WaitForScopeExtension();
  }
  stop.store(true);
  reader.join();
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_TRUE(Holds(2, {8.5, 2.0 + 0.02 * 99 + 1.0}));
}

// Support, contact, frame and height consumers read the DECLARED definition. An
// extended Surface parent and an extended Polytope parent ground exactly as unextended: the
// support boundary is the declared one and the contact lies on the declared edge (not on the
// extension, not near the world origin).
TEST_F(SceneValidation, ExtendedSurfaceAndPolytopeParentsGroundOnTheirDeclaredShape) {
  // A horizontal surface region (x 12..16, y 12..15 at z 0.75) and a box region (x 2..4, y 12..14).
  auto surface = std::make_shared<SurfaceSpatialDefinition>(std::make_shared<Polygon3D>(
      std::vector<Eigen::Vector3d>{{12, 12, 0.75}, {16, 12, 0.75}, {16, 15, 0.75}, {12, 15, 0.75}}));
  surface->SetGeometryRole(GeometryRole::kContainment);
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, surface));
  ASSERT_TRUE(space_->SetElementTransform(3, At(2, 12)));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(3, Box(0, 0, 2, 2)));
  // A child of each, 0.2 m past its +x edge.
  ASSERT_TRUE(space_->AddElement(4, 2));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(4, Region(16.2, 13, 17, 14)));
  ASSERT_TRUE(space_->AddElement(5, 3));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(5, Region(4.2, 12.5, 5, 13.5)));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  const ElementTree& tree = *space_->GetElementTree();
  struct Case {
    element_id_t id;
    Eigen::Vector3d anchor;
    Eigen::Vector2d declared_contact;
    Eigen::Vector2d in_extension;
  };
  for (const Case& c : {Case{2, {18, 13.5, 0}, {16, 13.5}, {16.1, 13.5}},
                        Case{3, {6, 13, 0}, {4, 13}, {4.1, 13}}}) {
    SCOPED_TRACE(c.id);
    const Element& e = *tree.GetElement(c.id);
    ASSERT_TRUE(e.HasScopeExtension());
    EXPECT_TRUE(Walks(c.id, c.in_extension)) << "the walkable is bridged";
    EXPECT_FALSE(Holds(c.id, c.in_extension)) << "the bridge is not in the scope";
    EXPECT_TRUE(Holds(c.id, c.in_extension + Eigen::Vector2d(0.4, 0))) << "the child is";
    const Support s = SpacingGroundingRule().ComputeSupport(e, tree, c.anchor, 0.0);
    ASSERT_TRUE(s.boundary);
    EXPECT_EQ(s.boundary->GetPoints(), e.GetDeclaredSpatialDefinition()->GetBoundary()->GetPoints());
    EXPECT_NEAR(s.contact_point_world.x(), c.declared_contact.x(), 1e-6);
    EXPECT_NEAR(s.contact_point_world.y(), c.declared_contact.y(), 1e-6);
  }
}

// A pocket the union encloses stays out of the scope and the walkable.
TEST_F(SceneValidation, APocketTheUnionEnclosesIsNotAddedToTheScope) {
  // A U-shaped room open at the top (the pocket x 3..7, y 3..10) and a child across its mouth.
  ASSERT_TRUE(space_->SetElementSpatialDefinition(
      1, std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(
             std::vector<Eigen::Vector2d>{{0, 0}, {10, 0}, {10, 10}, {7, 10}, {7, 3}, {3, 3}, {3, 10}, {0, 10}}))));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2.8, 9.6, 7.2, 10.4)));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  const ElementTree& tree = *space_->GetElementTree();
  ASSERT_TRUE(tree.GetElement(1)->HasScopeExtension());
  const auto* room = dynamic_cast<const NavigableSpatialDefinition*>(tree.GetElement(1)->GetSpatialDefinition());
  ASSERT_NE(room, nullptr);
  EXPECT_EQ(room->GetWalkablePolygon()->Holes().size(), 1u);
  EXPECT_FALSE(Holds(1, {5, 5})) << "far from the child: not in scope";
  EXPECT_FALSE(Holds(1, {5, 9.0}));
  EXPECT_TRUE(Holds(1, {5, 10.2})) << "the child itself";
  EXPECT_TRUE(Holds(1, {1, 1}));
  // Walkable reads agree: IsInWalkable, as IsFree and the scope.
  EXPECT_FALSE(room->IsInWalkable({5, 5}));
  EXPECT_FALSE(room->IsFree({5, 5}));
  EXPECT_TRUE(room->IsInWalkable({1, 1}));
  EXPECT_TRUE(room->IsInWalkable({5, 10.2}));
  // The free space keeps it out too: a goal in the pocket is not reached inside it.
  const auto reach = room->QueryReachability({1, 1}, {5, 5}, 0.0);
  EXPECT_TRUE(!reach.reachable || reach.to_snap_m > 0.5);
}

// Across a 20 m gap the bridge is a corridor along the segment
// between the nearest points, as wide as the child across it; the walkable grows by the child
// and that corridor only (not the points within 20 m of both regions).
TEST_F(SceneValidation, AFarChildIsJoinedByACorridorNotASlab) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(40, 9, 41, 11)));
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk);
  ASSERT_TRUE(space_->WaitForScopeExtension());
  const ElementTree& tree = *space_->GetElementTree();
  ASSERT_TRUE(tree.GetElement(1)->HasScopeExtension());
  const auto* room = dynamic_cast<const NavigableSpatialDefinition*>(tree.GetElement(1)->GetSpatialDefinition());
  ASSERT_NE(room, nullptr);
  const auto walkable = room->GetWalkablePolygon();
  EXPECT_EQ(walkable->NumComponents(), 1u);
  // The room, the child and the corridor (20 m by the child's 2 m, plus the slack).
  const double strip = (20.0 + 2 * kBridgeSlackM) * (2.0 + 2 * kBridgeSlackM);
  EXPECT_LE(walkable->GetArea(), 400.0 + 2.0 + strip + 1e-6);
  EXPECT_GE(walkable->GetArea(), 400.0 + 2.0 + 20.0 * 2.0 - 1e-6);
  for (double x : {20.5, 25.0, 30.0, 35.0, 39.5}) {
    EXPECT_FALSE(Holds(1, {x, 10})) << x << ": the corridor is walkable, not scope";
    EXPECT_TRUE(Walks(1, {x, 10})) << x;
    EXPECT_TRUE(room->IsInWalkable({x, 10})) << x;
  }
  EXPECT_TRUE(Holds(1, {40.5, 10})) << "the child";
  for (const Eigen::Vector2d p : {Eigen::Vector2d(30, -10), Eigen::Vector2d(30, 28), Eigen::Vector2d(30, 8.5),
                                  Eigen::Vector2d(30, 11.5), Eigen::Vector2d(41.5, 10)}) {
    EXPECT_FALSE(Walks(1, p)) << p.transpose();
    EXPECT_FALSE(room->IsInWalkable(p)) << p.transpose();
  }
  EXPECT_TRUE(room->IsReachable({10, 10}, {40.5, 10}, 1e-6, 0.0));
}

// The corridor is as wide as the child across it, at most kScopeExtensionCorridorMaxWidthM: a
// long band far away is reached by a corridor, not joined by a slab as wide as the band.
TEST_F(SceneValidation, AFarCorridorIsNoWiderThanTheMaximum) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(30, -5, 31, 25)));  // 30 m long
  ASSERT_TRUE(space_->WaitForScopeExtension());
  const ElementTree& tree = *space_->GetElementTree();
  const auto* room = dynamic_cast<const NavigableSpatialDefinition*>(tree.GetElement(1)->GetSpatialDefinition());
  ASSERT_NE(room, nullptr);
  EXPECT_EQ(room->GetWalkablePolygon()->NumComponents(), 1u);
  const double strip =
      (10.0 + 2 * kBridgeSlackM) * (kScopeExtensionCorridorMaxWidthM + 2 * kBridgeSlackM);
  EXPECT_LE(room->GetWalkablePolygon()->GetArea(), 400.0 + 30.0 + strip + 1e-6);
  EXPECT_FALSE(Holds(1, {25, 2})) << "not a slab across the band's length";
  EXPECT_FALSE(Holds(1, {25, 18}));
}

// A far child fills no hole of the parent's declared geometry (the
// pocket reach does not scale with the gap).
TEST_F(SceneValidation, AFarChildLeavesTheParentsOwnHoleAHole) {
  // A keyhole slit from the top edge to the hole x 14..16, y 9..11.
  const std::vector<Eigen::Vector2d> ring = {{0, 0},  {20, 0},  {20, 20}, {15, 20}, {15, 11}, {16, 11},
                                             {16, 9}, {14, 9},  {14, 11}, {15, 11}, {15, 20}, {0, 20}};
  ASSERT_TRUE(space_->SetElementSpatialDefinition(
      1, std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(ring))));
  ASSERT_FALSE(Holds(1, {15, 10}));
  EXPECT_TRUE(space_->SetElementSpatialDefinition(2, Region(30, 9, 31, 11)));
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk);
  ASSERT_TRUE(space_->WaitForScopeExtension());
  const ElementTree& tree = *space_->GetElementTree();
  ASSERT_TRUE(tree.GetElement(1)->HasScopeExtension());
  const auto* room = dynamic_cast<const NavigableSpatialDefinition*>(tree.GetElement(1)->GetSpatialDefinition());
  ASSERT_NE(room, nullptr);
  EXPECT_FALSE(Holds(1, {15, 10})) << "the declared hole stays a hole";
  EXPECT_FALSE(room->IsInWalkable({15, 10}));
  EXPECT_EQ(room->GetWalkablePolygon()->Holes().size(), 1u);
  EXPECT_TRUE(Walks(1, {25, 10})) << "the corridor";
  EXPECT_FALSE(Holds(1, {25, 10})) << "the corridor is not in the scope";
  EXPECT_TRUE(Holds(1, {30.5, 10})) << "the child";
}

// The bridge is checked against the parent's other descendants
// too: a corridor across a sibling region or an obstacle is rejected, naming both.
TEST_F(SceneValidation, ABridgeAcrossASiblingOrAnObstacleIsRejected) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2, 2, 8, 8)));
  ASSERT_TRUE(space_->AddElement(5, 2));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(5, Region(10, 3, 11, 7)));  // extends 2 itself
  ASSERT_TRUE(space_->AddElement(4, 2));
  // Its corridor (x 8..13, y 4..6) crosses its sibling 5.
  EXPECT_FALSE(space_->SetElementSpatialDefinition(4, Region(13, 4, 14, 6)));
  auto err = space_->LastSceneError();
  EXPECT_EQ(err.code, SceneErrorCode::kContainmentOverlap);
  EXPECT_NE(err.message.find("element 4"), std::string::npos) << err.message;
  EXPECT_NE(err.message.find("element 5"), std::string::npos) << err.message;
  EXPECT_EQ(space_->GetElementTree()->GetElement(4)->GetSpatialDefinition(), nullptr);
  // An obstacle of the parent in the corridor's way: rejected too.
  ASSERT_TRUE(space_->RemoveLocalElementSubtree(5));
  ASSERT_TRUE(space_->AddElement(6, 2));
  ASSERT_TRUE(space_->SetElementTransform(6, At(10, 4.5)));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(
      6, std::make_shared<PolytopeSpatialDefinition>(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0.5, 1, 1),
                                                     GeometryRole::kObstacle)));
  EXPECT_FALSE(space_->SetElementSpatialDefinition(4, Region(13, 4, 14, 6)));
  err = space_->LastSceneError();
  EXPECT_EQ(err.code, SceneErrorCode::kContainmentOverlap);
  EXPECT_NE(err.message.find("element 4"), std::string::npos) << err.message;
  EXPECT_NE(err.message.find("element 6"), std::string::npos) << err.message;
  // Clear of the corridor, the same child extends its parent.
  ASSERT_TRUE(space_->SetElementTransform(6, At(10, 7)));
  EXPECT_TRUE(space_->SetElementSpatialDefinition(4, Region(13, 4, 14, 6)));
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk);
}

// Neighbouring stand-off bands (two pieces of furniture side by side, both across the carve band
// from the ring) only touch each other's bridge in the slack: both are accepted.
TEST_F(SceneValidation, NeighbouringBandsAcrossTheCarveGapAreAccepted) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(20.2, 5, 22, 7)));
  EXPECT_TRUE(space_->SetElementSpatialDefinition(3, Region(20.2, 7, 22, 9)));
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk) << space_->LastSceneError().message;
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_TRUE(Walks(1, {20.1, 6}));
  EXPECT_TRUE(Walks(1, {20.1, 8}));
  EXPECT_TRUE(Holds(1, {21, 6}));
  EXPECT_TRUE(Holds(1, {21, 8}));
}

// The extension and its bridge must not overlap a footprint outside the line.
TEST_F(SceneValidation, ABridgeIntoAnotherFootprintIsRejected) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2, 2, 8, 8)));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(3, Region(8, 2, 8.2, 8)));  // a strip touching 2
  ASSERT_TRUE(space_->AddElement(4, 2));
  // 0.3 m past 2's edge: its own footprint is clear of 3, its bridge across the gap is not.
  EXPECT_FALSE(space_->SetElementSpatialDefinition(4, Region(8.3, 4, 9, 6)));
  const auto err = space_->LastSceneError();
  EXPECT_EQ(err.code, SceneErrorCode::kContainmentOverlap);
  EXPECT_NE(err.message.find("element 4"), std::string::npos) << err.message;
  EXPECT_NE(err.message.find("element 3"), std::string::npos) << err.message;
}

TEST_F(SceneValidation, AFootprintPlacedInsideAnExistingBridgeIsRejected) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2, 2, 8, 8)));
  ASSERT_TRUE(space_->AddElement(4, 2));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(4, Region(8.3, 4, 9, 6)));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  // In the gap between 2 and 4 (the bridge), touching neither.
  EXPECT_FALSE(space_->SetElementSpatialDefinition(3, Region(8.05, 4.5, 8.25, 5.5)));
  const auto err = space_->LastSceneError();
  EXPECT_EQ(err.code, SceneErrorCode::kContainmentOverlap);
  EXPECT_NE(err.message.find("element 4"), std::string::npos) << err.message;
  EXPECT_NE(err.message.find("element 3"), std::string::npos) << err.message;
  // And inside the extended region itself.
  EXPECT_FALSE(space_->SetElementSpatialDefinition(3, Region(8.5, 4.5, 8.8, 5.5)));
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kContainmentOverlap);
}

// A self-intersecting region that sticks out is refused, not repaired.
TEST_F(SceneValidation, ASelfIntersectingChildThatSticksOutIsRefused) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2, 2, 8, 8)));
  ASSERT_TRUE(space_->AddElement(4, 2));
  auto bowtie = std::make_shared<NavigableSpatialDefinition>(
      std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{{7, 4}, {9, 6}, {9, 4}, {7, 6}}), nullptr, 0.0,
      GeometryRole::kContainment);
  EXPECT_FALSE(space_->SetElementSpatialDefinition(4, bowtie));
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kInvalidArgument);
  EXPECT_NE(space_->LastSceneError().message.find("element 4"), std::string::npos);
}

// Which was registered first does not matter for an actor's element.
TEST_F(SceneValidation, AnActorsRegionNeverExtendsWhicheverIsRegisteredFirst) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2, 2, 8, 8)));
  const auto actor = space_->CreateActor({SubjectAnchor("root")});
  // Actor first, then the region that sticks out.
  ASSERT_TRUE(space_->AddElement(4, 2));
  ASSERT_TRUE(space_->SetActorElement(actor, 4));
  EXPECT_TRUE(space_->SetElementSpatialDefinition(4, Region(7, 4, 9, 6)));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_FALSE(space_->GetElementTree()->GetElement(2)->HasScopeExtension());
  // Region first (it extends), then the actor: the extension is removed.
  ASSERT_TRUE(space_->SetActorElement(actor, kInvalidElementId));
  ASSERT_TRUE(space_->AddElement(5, 2));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(5, Region(4, 7, 5, 9)));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  ASSERT_TRUE(Holds(2, {4.5, 8.5}));
  ASSERT_TRUE(space_->SetActorElement(actor, 5));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_FALSE(Holds(2, {4.5, 8.5}));
  // The rule is the kind, not the history: element 4 is no longer an actor's element, so it extends.
  EXPECT_TRUE(Holds(2, {8.5, 5}));
}

// A reader of the escaped tree (the Python binding, the simulator) reads the
// definition pointer without the space lock while the worker swaps it: it always reads a whole
// definition, and one it just read stays alive while it uses it (run under ThreadSanitizer).
TEST_F(ScopeExtension, AnEscapedTreeReaderAlwaysSeesAWholeDefinition) {
  const Element* parent = space_->GetElementTree()->GetElement(2);
  std::atomic<bool> stop{false};
  std::atomic<int> reads{0};
  std::thread reader([&] {
    while (!stop.load()) {
      const SpatialDefinition* def = parent->GetSpatialDefinition();  // raw, as the readers use it
      ASSERT_NE(def, nullptr);
      const auto boundary = def->GetBoundary();
      ASSERT_TRUE(boundary);
      EXPECT_TRUE(boundary->Contains({5, 5})) << "the declared region is in every complete scope";
      reads.fetch_add(1);
    }
  });
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(space_->SetElementTransform(4, At(7, 2.0 + 0.04 * (i % 50))));
    if (i % 10 == 0) ASSERT_TRUE(space_->WaitForScopeExtension());
  }
  ASSERT_TRUE(space_->WaitForScopeExtension());
  stop.store(true);
  reader.join();
  EXPECT_GT(reads.load(), 0);
}

// A parent expands to contain its children; the bridges and corridors across a
// gap only connect the walkable. The Spacing scope of a pose (ScopeResolver::ResolveSpacing) in a
// gap is the room's, with the room's referents; a pose inside the extending child resolves through
// the parent into the child. A TABLE (2) with a Chess child (4) sticking out, a Whiteboard (3).
class ScopeAcrossAGap : public SceneValidation {
 protected:
  void Build(const std::shared_ptr<SpatialDefinition>& chess) {
    ASSERT_TRUE(space_->SetElementSharedId(1, 100));
    ASSERT_TRUE(space_->SetElementSharedId(2, 300));
    ASSERT_TRUE(space_->SetElementSharedId(3, 400));
    ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2, 2, 8, 8)));
    ASSERT_TRUE(space_->SetElementSpatialDefinition(3, Region(14, 14, 16, 16)));
    ASSERT_TRUE(space_->AddElement(4, 2));
    ASSERT_TRUE(space_->SetElementSharedId(4, 200));
    ASSERT_TRUE(space_->SetElementSpatialDefinition(4, chess));
    EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk);
    ASSERT_TRUE(space_->WaitForScopeExtension());
    ASSERT_TRUE(space_->GetElementTree()->GetElement(2)->HasScopeExtension());
  }
  SpacingScope At(double x, double y) {
    return ScopeResolver::ResolveSpacing({x, y, 0}, 1, *space_->GetElementTree());
  }
  static bool Has(const SpacingScope& s, element_id_t id) {
    return std::find(s.referents.begin(), s.referents.end(), id) != s.referents.end();
  }
};

TEST_F(ScopeAcrossAGap, ASourceInTheGapKeepsTheRoomsReferents) {
  Build(Region(8.3, 4, 9, 6));  // 0.3 m out: the carve-band bridge
  ASSERT_TRUE(Walks(2, {8.15, 5})) << "the gap is walkable";
  const SpacingScope s = At(8.15, 5);
  EXPECT_EQ(s.container, 1) << "the gap is the room's scope";
  EXPECT_EQ(s.mode, SpacingScopeMode::kOutside);
  EXPECT_TRUE(Has(s, 2)) << "the TABLE";
  EXPECT_TRUE(Has(s, 3)) << "the Whiteboard, the room's other referent";
  EXPECT_FALSE(s.border);
}

TEST_F(ScopeAcrossAGap, ASourceInTheExtendingChildResolvesThroughTheParent) {
  Build(Region(8.3, 4, 9, 6));
  const ScopeDescent d = DescendScope({8.6, 5, 0}, 1, *space_->GetElementTree(), true, true);
  EXPECT_EQ(d.container, 4);
  EXPECT_EQ(At(8.6, 5).container, 4);
  EXPECT_TRUE(Holds(2, {8.6, 5})) << "the TABLE's scope contains its child";
  EXPECT_FALSE(Holds(3, {8.6, 5}));
}

TEST_F(ScopeAcrossAGap, ASourceInACorridorKeepsTheRoomsReferents) {
  Build(Region(12, 4, 13, 6));  // 4 m out: a corridor
  ASSERT_TRUE(Walks(2, {10, 5})) << "the corridor is walkable";
  const SpacingScope s = At(10, 5);
  EXPECT_EQ(s.container, 1);
  EXPECT_EQ(s.mode, SpacingScopeMode::kOutside);
  EXPECT_TRUE(Has(s, 2));
  EXPECT_TRUE(Has(s, 3));
  // Inside the far child: through the parent.
  EXPECT_EQ(At(12.5, 5).container, 4);
  EXPECT_TRUE(Holds(2, {12.5, 5}));
}

// On the computation itself.
TEST(ScopeExtensionUnit, AnExtendedBoxKeepsItsHeightBandAndItsClosestPointAgreesWithItsDistance) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  tree.GetElement(1)->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(2, 2, 1), GeometryRole::kContainment));
  ASSERT_TRUE(tree.AddElement(2, 1).ok());
  tree.GetElement(2)->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(1.5, 0.5, 0), Eigen::Vector3d(2.5, 1.5, 1), GeometryRole::kContainment));
  ScopeExtensionInput input = CaptureScopeExtensionInput(tree);
  ScopeExtensionCache cache;
  const ScopeExtensionResult first = ComputeScopeExtensions(input, &cache);
  ASSERT_TRUE(InstallScopeExtensions(&tree, input, first));
  const SpatialDefinition& ext = *tree.GetElement(1)->GetSpatialDefinition();
  EXPECT_TRUE(ext.Contains3D({2.2, 1.0, 0.5}));
  EXPECT_FALSE(ext.Contains3D({2.2, 1.0, 3.0})) << "above the box's height band";
  EXPECT_EQ(ext.GetDistance({2.2, 1.0, 0.0}), 0.0);
  EXPECT_EQ(ext.GetClosestPoint({2.2, 1.0, 0.0}), Eigen::Vector3d(2.2, 1.0, 0.0));
  // Unchanged inputs: the same definition, nothing rebuilt.
  input = CaptureScopeExtensionInput(tree);
  const ScopeExtensionResult second = ComputeScopeExtensions(input, &cache);
  EXPECT_EQ(cache.reused, 1u);
  EXPECT_EQ(second.extended.at(1), first.extended.at(1));
  EXPECT_FALSE(InstallScopeExtensions(&tree, input, second));
}

// A parent ring that NavMesh exports with a spike or a keyhole slit (invalid
// to Boost only by zero-area features) is repaired, identically for admission and the build: the
// nearby child extends it, the keyhole's hole stays a hole. A parent ring that crosses itself
// cannot be repaired: the child is refused, naming both.
TEST_F(SceneValidation, ASpikeOrKeyholeParentRingIsRepairedAndExtended) {
  const std::vector<std::vector<Eigen::Vector2d>> rings = {
      {{0, 0}, {20, 0}, {20, 10}, {23, 10}, {20, 10}, {20, 20}, {0, 20}},  // a spike at y = 10
      {{0, 0}, {20, 0}, {20, 20}, {10, 20}, {10, 14}, {14, 14}, {14, 6}, {6, 6}, {6, 14}, {10, 14},
       {10, 20}, {0, 20}}};  // a keyhole slit from the top edge to the hole x 6..14, y 6..14
  for (size_t k = 0; k < rings.size(); ++k) {
    SCOPED_TRACE(k);
    SetUp();
    ASSERT_FALSE(Polygon2D(rings[k]).IsValid());
    ASSERT_TRUE(space_->SetElementSpatialDefinition(
        1, std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(rings[k]))));
    EXPECT_TRUE(space_->SetElementSpatialDefinition(2, Region(20.2, 3, 22, 5)));
    EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk);
    ASSERT_TRUE(space_->WaitForScopeExtension());
    const ElementTree& tree = *space_->GetElementTree();
    ASSERT_TRUE(tree.GetElement(1)->HasScopeExtension()) << "the child extends the repaired ring";
    EXPECT_TRUE(Walks(1, {20.1, 4})) << "bridged";
    EXPECT_FALSE(Holds(1, {20.1, 4})) << "the bridge is not in the scope";
    EXPECT_TRUE(Holds(1, {21, 4}));
    const auto* room = dynamic_cast<const NavigableSpatialDefinition*>(tree.GetElement(1)->GetSpatialDefinition());
    ASSERT_NE(room, nullptr);
    EXPECT_EQ(room->GetWalkablePolygon()->NumComponents(), 1u);
    if (k == 1) EXPECT_FALSE(Holds(1, {10, 10})) << "the keyhole's hole stays a hole";
  }
  SetUp();
  ASSERT_TRUE(space_->SetElementSpatialDefinition(
      1, std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(
             std::vector<Eigen::Vector2d>{{0, 0}, {20, 20}, {20, 0}, {0, 20}}))));  // crosses itself
  EXPECT_FALSE(space_->SetElementSpatialDefinition(2, Region(19.5, 3, 21, 5)));
  const auto err = space_->LastSceneError();
  EXPECT_EQ(err.code, SceneErrorCode::kInvalidArgument);
  EXPECT_NE(err.message.find("element 2"), std::string::npos) << err.message;
  EXPECT_NE(err.message.find("element 1"), std::string::npos) << err.message;
}

// The worker gives an extended navigable the obstacles the solve-thread sync
// would give it for the EXTENDED bounds, so installing the extension rebuilds no free space on a
// caller's thread.
TEST_F(SceneValidation, InstallingAnExtensionCausesNoInlineFreeSpaceRebuild) {
  // An obstacle beyond the room's ring (x 21..21.3) that only the extended bounds reach.
  ASSERT_TRUE(space_->SetElementTransform(3, At(21, 9)));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(
      3, std::make_shared<PolytopeSpatialDefinition>(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0.3, 0.3, 1),
                                                     GeometryRole::kObstacle)));
  (void)space_->QueryCourse(spatialblend::kInvalidEntityId, {1, 1}, {2, 2});  // the sync, settled
  const uint64_t rebuilds = space_->GetObstacleSyncRebuildCountForTesting();
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(20.2, 3, 22, 5)));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  (void)space_->QueryCourse(spatialblend::kInvalidEntityId, {1, 1}, {2, 2});  // the sync after the install
  EXPECT_EQ(space_->GetObstacleSyncRebuildCountForTesting(), rebuilds);
  const auto* room = dynamic_cast<const NavigableSpatialDefinition*>(
      space_->GetElementTree()->GetElement(1)->GetSpatialDefinition());
  ASSERT_NE(room, nullptr);
  ASSERT_TRUE(space_->GetElementTree()->GetElement(1)->HasScopeExtension());
  bool holds_obstacle = false;
  for (const auto& o : room->GetObstacles()) holds_obstacle |= o.key == 3;
  EXPECT_TRUE(holds_obstacle) << "the extended bounds reach it, as the sync computes";
}

// Releasing an actor's element always succeeds. The
// released region extends its parent by the normal rule (at any distance) and its
// next declaration or move is validated as any element's.
TEST_F(SceneValidation, ReleasingAnActorsElementAlwaysSucceeds) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2, 2, 8, 8)));
  const auto far_actor = space_->CreateActor({SubjectAnchor("root")});
  const auto near_actor = space_->CreateActor({SubjectAnchor("root")});
  ASSERT_TRUE(space_->AddElement(4, 2));
  ASSERT_TRUE(space_->SetActorElement(far_actor, 4));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(4, Region(9.5, 4, 10.5, 6)));  // 1.5 m out: an actor's
  ASSERT_TRUE(space_->AddElement(5, 2));
  ASSERT_TRUE(space_->SetActorElement(near_actor, 5));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(5, Region(4, 7, 5, 9)));  // sticks out, nearby
  EXPECT_TRUE(space_->SetActorElement(far_actor, kInvalidElementId));
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk);
  EXPECT_EQ(space_->GetActor(far_actor)->GetElementId(), kInvalidElementId);
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_TRUE(Holds(2, {10, 5})) << "released 1.5 m out: it extends its parent";
  EXPECT_TRUE(Walks(2, {8.75, 5})) << "bridged across the gap";
  EXPECT_FALSE(Holds(2, {8.75, 5})) << "the bridge is not in the scope";
  // Its next move is validated as any element's: 2 m out, it extends the parent there.
  EXPECT_TRUE(space_->SetElementTransform(4, At(0.5, 0)));
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk);
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_TRUE(Holds(2, {10.5, 5}));
  // Released nearby: it extends its parent by the normal rule.
  EXPECT_FALSE(Holds(2, {4.5, 8.5}));
  EXPECT_TRUE(space_->SetActorElement(near_actor, kInvalidElementId));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_TRUE(Holds(2, {4.5, 8.5}));
}

// An actor placed in the room (the room is its element, as the C API and the
// Unity Foundation tests do it) does not make the room's furniture an actor's.
TEST_F(SceneValidation, AnActorPlacedInTheRoomLeavesTheParentRuleToItsFurniture) {
  const auto actor = space_->CreateActor({SubjectAnchor("root")});
  ASSERT_TRUE(space_->SetActorElement(actor, 1));
  // Not the actor's body: the furniture extends the room by the parent rule, near or far.
  EXPECT_TRUE(space_->SetElementSpatialDefinition(3, Region(40, 4, 41, 6)));  // 20 m out
  EXPECT_EQ(space_->LastSceneError().code, SceneErrorCode::kOk);
  // Nearby, clear of the far one's corridor (y 4..6: a footprint in it is an overlap).
  EXPECT_TRUE(space_->SetElementSpatialDefinition(2, Region(20.2, 12, 22, 14)));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_TRUE(space_->GetElementTree()->GetElement(1)->HasScopeExtension());
  EXPECT_TRUE(Holds(1, {21, 13}));
  EXPECT_TRUE(Holds(1, {40.5, 5}));
  EXPECT_TRUE(Walks(1, {30, 5}));  // the bridge: walkable
  EXPECT_FALSE(Holds(1, {30, 5}));  // not scope
  // The actor's BODY below its own (non-region) element is still the actor's.
  ASSERT_TRUE(space_->AddElement(6, 1));
  ASSERT_TRUE(space_->AddElement(7, 6));
  ASSERT_TRUE(space_->SetActorElement(actor, 6));
  EXPECT_TRUE(IsActorBound(*space_->GetElementTree(), 7));
  EXPECT_FALSE(IsActorBound(*space_->GetElementTree(), 2));
}

// The AffordancePose parent-region gate excludes a point in a hole of the
// parent's (extended) region, as the footprint test does.
TEST_F(SceneValidation, TheParentRegionGateExcludesAHoleOfTheRegion) {
  ASSERT_TRUE(space_->SetElementSpatialDefinition(
      1, std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(
             std::vector<Eigen::Vector2d>{{0, 0}, {10, 0}, {10, 10}, {7, 10}, {7, 3}, {3, 3}, {3, 10}, {0, 10}}))));
  ASSERT_TRUE(space_->SetElementSpatialDefinition(2, Region(2.8, 9.6, 7.2, 10.4)));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  ASSERT_TRUE(space_->AddElement(6, 1));  // an affordance referent of the room
  const ElementTree& tree = *space_->GetElementTree();
  const Support support;
  EXPECT_EQ(FamilyScopeGateVerdict(RelationKernelType::kAvatarAffordancePose, tree, 6, {5, 5, 0}, support, nullptr),
            ScopeGateVerdict::kOutsideParentRegion)
      << "in the pocket: not in the region";
  EXPECT_EQ(FamilyScopeGateVerdict(RelationKernelType::kAvatarAffordancePose, tree, 6, {1, 1, 0}, support, nullptr),
            ScopeGateVerdict::kInScope);
}

// A scope region of several components (a child across a gap,
// no bridge) is read whole by every reader: the AffordancePose parent-region gate holds an anchor in
// ANY component, and the v1 normalized distance takes the centroid and radius of the whole region.
// Both a small parent with a larger child and the reverse.
TEST_F(SceneValidation, AMultiComponentScopeIsReadWhole) {
  struct Case {
    std::shared_ptr<SpatialDefinition> parent, child;
    Eigen::Vector2d parent_centre, child_centre, gap;
  };
  for (const Case& c : {Case{Region(5, 5, 6, 6), Region(6.4, 4, 9, 7), {5.5, 5.5}, {7.7, 5.5}, {6.2, 5.5}},
                        Case{Region(2, 2, 8, 8), Region(8.4, 4, 9, 5), {5, 5}, {8.7, 4.5}, {8.2, 4.5}}}) {
    SCOPED_TRACE(c.parent_centre.transpose());
    SetUp();
    ASSERT_TRUE(space_->SetElementSpatialDefinition(2, c.parent));
    ASSERT_TRUE(space_->AddElement(4, 2));
    ASSERT_TRUE(space_->SetElementSpatialDefinition(4, c.child));
    ASSERT_TRUE(space_->AddElement(5, 2));  // an affordance referent of the parent
    ASSERT_TRUE(space_->WaitForScopeExtension());
    const ElementTree& tree = *space_->GetElementTree();
    const Element& parent = *tree.GetElement(2);
    ASSERT_TRUE(parent.HasScopeExtension());
    const Polygon2D region = FootprintImageRegion(parent, tree);
    ASSERT_EQ(region.NumComponents(), 2u) << "no bridge joins the scope";
    const Support support;
    const auto gate = [&](const Eigen::Vector2d& p) {
      return FamilyScopeGateVerdict(RelationKernelType::kAvatarAffordancePose, tree, 5, {p.x(), p.y(), 0}, support,
                                    nullptr);
    };
    EXPECT_EQ(gate(c.parent_centre), ScopeGateVerdict::kInScope) << "the parent's own centre";
    EXPECT_EQ(gate(c.child_centre), ScopeGateVerdict::kInScope) << "inside the extending child";
    EXPECT_EQ(gate(c.gap), ScopeGateVerdict::kOutsideParentRegion) << "the gap is not in the scope";
    // The v1 distance over the whole region: area-weighted centroid, radius over every vertex.
    const Polygon2D a(*parent.GetDeclaredSpatialDefinition()->GetBoundary());
    const Polygon2D b(*tree.GetElement(4)->GetDeclaredSpatialDefinition()->GetBoundary());
    const Eigen::Vector2d centre = (a.GetArea() * a.GetCenter() + b.GetArea() * b.GetCenter()) / (a.GetArea() + b.GetArea());
    double radius = 0.0;
    for (const Polygon2D* q : {&a, &b})
      for (const Eigen::Vector2d& v : q->GetPoints()) radius = std::max(radius, (v - centre).norm());
    EXPECT_NEAR(V1NormalizedDistanceOf(parent, tree, c.parent_centre), (c.parent_centre - centre).norm() / radius,
                1e-6);
  }
}

// A build that throws keeps the previous scope; the next one installs.
TEST_F(ScopeExtension, AFailedBuildKeepsThePreviousScope) {
  const uint64_t revision = space_->GetScopeExtensionRevision();
  std::atomic<bool> fail{true};
  space_->SetScopeExtensionBuildHookForTesting([&fail] {
    if (fail.load()) throw std::runtime_error("injected build failure");
  });
  ASSERT_TRUE(space_->SetElementTransform(4, At(4, 7)));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_EQ(space_->GetScopeExtensionRevision(), revision);
  EXPECT_TRUE(Holds(2, {8.5, 5})) << "the previous, complete scope";
  EXPECT_FALSE(Holds(2, {5, 8.5}));
  fail.store(false);
  ASSERT_TRUE(space_->SetElementTransform(4, At(4, 7.1)));
  ASSERT_TRUE(space_->WaitForScopeExtension());
  EXPECT_GT(space_->GetScopeExtensionRevision(), revision);
  EXPECT_TRUE(Holds(2, {5, 8.5}));
  EXPECT_FALSE(Holds(2, {8.5, 5}));
  space_->SetScopeExtensionBuildHookForTesting(nullptr);
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
