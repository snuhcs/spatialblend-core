// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/relation/avatar_affordance_pose_kernel.h"
#include "spatialblend/model/relation/avatar_facing_kernel.h"
#include "spatialblend/model/relation/avatar_spacing_kernel.h"

namespace spatialblend {
namespace test {

namespace {

// Helper: check if a vector contains a given id.
bool Contains(const std::vector<element_id_t>& ids, element_id_t id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

// Helper: build a polygon containing (5,5).
std::shared_ptr<Polygon2D> MakePolygonContaining5x5() {
  return std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {10, 0}, {10, 10}, {0, 10}});
}

// Helper: build a smaller polygon containing (5,5).
std::shared_ptr<Polygon2D> MakeSmallPolygonContaining5x5() {
  return std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{3, 3}, {7, 3}, {7, 7}, {3, 7}});
}

}  // namespace

// --- Spacing kernel scope tests ---

TEST(ResolveScopeTest, SpacingReturnsChildrenOfContainer) {
  // Tree: Room(1) -> Board(2), Table(3)
  // Room has polygon containing (5,5)
  // Actor at (5,5) in Room -> scope = {Board, Table}
  ElementTree tree;
  tree.AddElement(1);
  tree.GetElement(1)->SetSharedId(1);
  tree.AddElement(2, 1);
  tree.GetElement(2)->SetSharedId(2);
  tree.AddElement(3, 1);

  tree.GetElement(3)->SetSharedId(3);
  auto nav = std::make_shared<NavigableSpatialDefinition>(
      MakePolygonContaining5x5());
  tree.GetElement(1)->SetSpatialDefinition(nav);

  AvatarSpacingKernel kernel;
  auto scope = kernel.ResolveScope(Eigen::Vector3d(5, 5, 0), 1, tree);

  EXPECT_EQ(scope.size(), 2u);
  EXPECT_TRUE(Contains(scope, 2));
  EXPECT_TRUE(Contains(scope, 3));
}

TEST(ResolveScopeTest, SpacingNarrowsToDeepestContaining) {
  // Tree: Room(1) -> Area(2) -> Table(3)
  // Area has polygon containing (5,5)
  // Actor at (5,5) starting from Room -> scope = Area's children = {Table}
  ElementTree tree;
  tree.AddElement(1);
  tree.GetElement(1)->SetSharedId(1);
  tree.AddElement(2, 1);
  tree.GetElement(2)->SetSharedId(2);
  tree.AddElement(3, 2);

  tree.GetElement(3)->SetSharedId(3);
  auto nav = std::make_shared<NavigableSpatialDefinition>(
      MakeSmallPolygonContaining5x5());
  tree.GetElement(2)->SetSpatialDefinition(nav);

  AvatarSpacingKernel kernel;
  auto scope = kernel.ResolveScope(Eigen::Vector3d(5, 5, 0), 1, tree);

  EXPECT_EQ(scope.size(), 1u);
  EXPECT_TRUE(Contains(scope, 3));
}

TEST(ResolveScopeTest, SpacingDoesNotTargetInteractionRegionChildren) {
  // Tree: Room(1) -> Object(2) -> InteractionRegion(3)
  // Actor inside Object and InteractionRegion -> spacing should target Object,
  // not its affordance/contact region child.
  ElementTree tree;
  tree.AddElement(1);
  tree.GetElement(1)->SetSharedId(1);
  tree.AddElement(2, 1);
  tree.GetElement(2)->SetSharedId(2);
  tree.AddElement(3, 2);

  tree.GetElement(3)->SetSharedId(3);
  auto object_region = std::make_shared<NavigableSpatialDefinition>(
      MakeSmallPolygonContaining5x5(), nullptr, 0.0,
      GeometryRole::kContainment);
  tree.GetElement(2)->SetSpatialDefinition(object_region);

  auto interaction_region = std::make_shared<NavigableSpatialDefinition>(
      MakeSmallPolygonContaining5x5(), nullptr, 0.0,
      GeometryRole::kInteractionRegion);
  tree.GetElement(3)->SetSpatialDefinition(interaction_region);

  AvatarSpacingKernel kernel;
  auto scope = kernel.ResolveScope(Eigen::Vector3d(5, 5, 0), 1, tree);

  ASSERT_EQ(scope.size(), 1u);
  EXPECT_TRUE(Contains(scope, 2));
  EXPECT_FALSE(Contains(scope, 3));
}

// --- Facing kernel scope tests ---

TEST(ResolveScopeTest, FacingReturnsAllDescendants) {
  // Tree: Room(1) -> Board(2) -> Writable(3)
  // Actor element = Room -> scope = {Board, Writable}
  ElementTree tree;
  tree.AddElement(1);
  tree.GetElement(1)->SetSharedId(1);
  tree.AddElement(2, 1);
  tree.GetElement(2)->SetSharedId(2);
  tree.AddElement(3, 2);

  tree.GetElement(3)->SetSharedId(3);
  AvatarFacingKernel kernel;
  auto scope = kernel.ResolveScope(Eigen::Vector3d(5, 5, 0), 1, tree);

  EXPECT_EQ(scope.size(), 2u);
  EXPECT_TRUE(Contains(scope, 2));
  EXPECT_TRUE(Contains(scope, 3));
}

TEST(ResolveScopeTest, FacingFallsBackToParentForLeaf) {
  // Tree: Room(1) -> Board(2), Chair(3)
  // Actor element = Board (leaf) -> subtree of Board minus Board = empty
  // Fallback: parent (Room) subtree minus Room = {Board, Chair}
  ElementTree tree;
  tree.AddElement(1);
  tree.GetElement(1)->SetSharedId(1);
  tree.AddElement(2, 1);
  tree.GetElement(2)->SetSharedId(2);
  tree.AddElement(3, 1);

  tree.GetElement(3)->SetSharedId(3);
  AvatarFacingKernel kernel;
  auto scope = kernel.ResolveScope(Eigen::Vector3d(5, 5, 0), 2, tree);

  // Should include siblings (and self, since parent subtree includes Board)
  EXPECT_FALSE(scope.empty());
  EXPECT_TRUE(Contains(scope, 3));
}

// Facing scope is not limited to the actor's subtree. An element in
// another subtree (here: another room, a sibling of the actor's room under a
// common building root) is a structural candidate; line of sight decides.
TEST(ResolveScopeTest, ScopeGating_E1_1_FacingStructuralScopeIsAnyDepth) {
  // Building(1) -> RoomA(2) -> Chair(4); Building(1) -> RoomB(3) -> Lamp(5)
  ElementTree tree;
  tree.AddElement(1);
  tree.GetElement(1)->SetSharedId(1);
  tree.AddElement(2, 1);
  tree.GetElement(2)->SetSharedId(2);
  tree.AddElement(3, 1);
  tree.GetElement(3)->SetSharedId(3);
  tree.AddElement(4, 2);
  tree.GetElement(4)->SetSharedId(4);
  tree.AddElement(5, 3);

  tree.GetElement(5)->SetSharedId(5);
  AvatarFacingKernel kernel;
  auto scope = kernel.ResolveScope(Eigen::Vector3d(5, 5, 0), 2, tree);
  EXPECT_TRUE(Contains(scope, 4));
  EXPECT_TRUE(Contains(scope, 5)) << "other subtree must be a candidate";
  EXPECT_TRUE(Contains(scope, 3));
  EXPECT_FALSE(Contains(scope, 2)) << "the actor's own element";
  EXPECT_FALSE(Contains(scope, 1)) << "an element the actor sits in";
}

// --- AffordancePose kernel scope tests ---

TEST(ResolveScopeTest, AffordancePoseReturnsDescendantsOfDeepest) {
  // Tree: Room(1) -> Object(2) -> Affordance(3)
  // Object has polygon containing (5,5)
  // Actor at (5,5) starting from Room -> deepest = Object -> scope = {Affordance}
  ElementTree tree;
  tree.AddElement(1);
  tree.GetElement(1)->SetSharedId(1);
  tree.AddElement(2, 1);
  tree.GetElement(2)->SetSharedId(2);
  tree.AddElement(3, 2);

  tree.GetElement(3)->SetSharedId(3);
  auto nav = std::make_shared<NavigableSpatialDefinition>(
      MakeSmallPolygonContaining5x5());
  tree.GetElement(2)->SetSpatialDefinition(nav);

  AvatarAffordancePoseKernel kernel;
  auto scope = kernel.ResolveScope(Eigen::Vector3d(5, 5, 0), 1, tree);

  EXPECT_EQ(scope.size(), 1u);
  EXPECT_TRUE(Contains(scope, 3));
}

// --- Invalid actor element tests ---

TEST(ResolveScopeTest, SpacingInvalidActorElementReturnsEmpty) {
  ElementTree tree;
  tree.AddElement(1);

  tree.GetElement(1)->SetSharedId(1);
  AvatarSpacingKernel kernel;
  auto scope = kernel.ResolveScope(
      Eigen::Vector3d(5, 5, 0), kInvalidElementId, tree);

  EXPECT_TRUE(scope.empty());
}

TEST(ResolveScopeTest, FacingInvalidActorElementReturnsEmpty) {
  ElementTree tree;
  tree.AddElement(1);

  tree.GetElement(1)->SetSharedId(1);
  AvatarFacingKernel kernel;
  auto scope = kernel.ResolveScope(
      Eigen::Vector3d(5, 5, 0), kInvalidElementId, tree);

  EXPECT_TRUE(scope.empty());
}

TEST(ResolveScopeTest, AffordancePoseInvalidActorElementReturnsEmpty) {
  ElementTree tree;
  tree.AddElement(1);

  tree.GetElement(1)->SetSharedId(1);
  AvatarAffordancePoseKernel kernel;
  auto scope = kernel.ResolveScope(
      Eigen::Vector3d(5, 5, 0), kInvalidElementId, tree);

  EXPECT_TRUE(scope.empty());
}

}  // namespace test
}  // namespace spatialblend
