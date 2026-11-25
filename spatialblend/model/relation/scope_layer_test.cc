// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Layer 1 (scope), layer-isolated: a pure function of (tree, anchor pose). No
// Space, no solver, no time. Encodes the current rule: the deepest containing element is the scope; Spacing referents are
// its direct non-interaction children plus, where the border-referent rule holds,
// the container itself; the rule lives in ScopeResolver::HasBorderReferent.
#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <vector>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/scope_resolver.h"

namespace spatialblend {
namespace test {
namespace {

enum : element_id_t {
  kRoom = 1, kRegion = 2, kObjA = 3, kObjB = 4, kTransfer = 5, kLeaf = 6, kAffordance = 7
};

std::shared_ptr<PolytopeSpatialDefinition> Box(double x0, double y0, double x1,
                                               double y1, GeometryRole role) {
  return std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(x0, y0, 0.0), Eigen::Vector3d(x1, y1, 1.0), role);
}

// Room (navigable, 0..20) -> Region (containment 5..10) -> {ObjA, ObjB, Affordance
// (interaction region)}; Transfer (containment 12..14, a frame-scope element);
// Leaf (containment 15..17, no objects).
void BuildTree(ElementTree* tree) {
  ASSERT_TRUE(tree->AddElement(kRoom).ok());
  tree->GetElement(kRoom)->SetSharedId(kRoom);
  tree->GetElement(kRoom)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(
          std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 20}, {0, 20}})));
  ASSERT_TRUE(tree->AddElement(kRegion, kRoom).ok());
  tree->GetElement(kRegion)->SetSharedId(kRegion);
  tree->GetElement(kRegion)->SetSpatialDefinition(Box(5, 5, 10, 10, GeometryRole::kContainment));
  ASSERT_TRUE(tree->AddElement(kObjA, kRegion).ok());
  tree->GetElement(kObjA)->SetSharedId(kObjA);
  tree->GetElement(kObjA)->SetSpatialDefinition(Box(6, 6, 7, 7, GeometryRole::kObstacle));
  ASSERT_TRUE(tree->AddElement(kObjB, kRegion).ok());
  tree->GetElement(kObjB)->SetSharedId(kObjB);
  tree->GetElement(kObjB)->SetSpatialDefinition(Box(8, 8, 9, 9, GeometryRole::kObstacle));
  ASSERT_TRUE(tree->AddElement(kAffordance, kRegion).ok());
  tree->GetElement(kAffordance)->SetSharedId(kAffordance);
  tree->GetElement(kAffordance)->SetSpatialDefinition(
      Box(6, 8, 7, 9, GeometryRole::kInteractionRegion));
  ASSERT_TRUE(tree->AddElement(kTransfer, kRoom).ok());
  tree->GetElement(kTransfer)->SetSharedId(kTransfer);
  tree->GetElement(kTransfer)->SetSpatialDefinition(Box(12, 5, 14, 10, GeometryRole::kContainment));
  tree->GetElement(kTransfer)->SetContainmentTransfer(true);
  ASSERT_TRUE(tree->AddElement(kLeaf, kRoom).ok());
  tree->GetElement(kLeaf)->SetSharedId(kLeaf);
  tree->GetElement(kLeaf)->SetSpatialDefinition(Box(15, 5, 17, 10, GeometryRole::kContainment));
}

std::set<element_id_t> Referents(const ElementTree& tree, double x, double y) {
  const Eigen::Vector3d a(x, y, 0.0);
  const auto ids = ScopeResolver::ResolveReachable(
      a, kRoom, tree, FindDeepestContaining(a, kRoom, tree));
  return std::set<element_id_t>(ids.begin(), ids.end());
}

// The hand-written table: which set a pose must give.
std::set<element_id_t> Expected(double x, double y) {
  const auto in = [&](double x0, double y0, double x1, double y1) {
    return x > x0 && x < x1 && y > y0 && y < y1;
  };
  if (in(6, 6, 7, 7)) return {kObjA};                      // inside an object: it, replayed
  if (in(8, 8, 9, 9)) return {kObjB};
  if (in(5, 5, 10, 10)) return {kObjA, kObjB, kRegion};  // objects + the border
  if (in(12, 5, 14, 10)) return {kTransfer};               // transfer leaf: itself
  if (in(15, 5, 17, 10)) return {kLeaf};                   // leaf: itself
  return {kRegion, kTransfer, kLeaf};                      // the room: its children
}

}  // namespace

TEST(ScopeExactSet, Grid) {
  ElementTree tree;
  BuildTree(&tree);
  int samples = 0;
  for (double x = 0.025; x < 20.0; x += 0.05)
    for (double y = 0.025; y < 20.0; y += 0.05) {
      // Skip the 5 mm band at an edge, where the polygon test is ambiguous.
      const auto near_edge = [&](double v, double e) { return std::fabs(v - e) < 0.01; };
      if (near_edge(x, 5) || near_edge(x, 10) || near_edge(x, 12) || near_edge(x, 14) ||
          near_edge(x, 15) || near_edge(x, 17) || near_edge(y, 5) || near_edge(y, 10) ||
          near_edge(x, 6) || near_edge(x, 7) || near_edge(x, 8) || near_edge(x, 9) ||
          near_edge(y, 6) || near_edge(y, 7) || near_edge(y, 8) || near_edge(y, 9))
        continue;
      ++samples;
      EXPECT_EQ(Referents(tree, x, y), Expected(x, y)) << "at " << x << "," << y;
    }
  EXPECT_GT(samples, 100000);
}

TEST(BorderReferent, EntersOnTheCrossingFrame) {
  ElementTree tree;
  BuildTree(&tree);
  // Frame k: just outside the region; frame k+1: just inside. The set changes on
  // exactly this frame (no hysteresis, no transition code), and the container is
  // an ordinary child before and the border after.
  const auto before = Referents(tree, 4.99, 7.5);
  const auto after = Referents(tree, 5.01, 7.5);
  EXPECT_EQ(before, (std::set<element_id_t>{kRegion, kTransfer, kLeaf}));
  EXPECT_EQ(after, (std::set<element_id_t>{kObjA, kObjB, kRegion}));
  EXPECT_TRUE(ScopeResolver::HasBorderReferent(tree, kRegion));
  // And on the way out (frame k+1 -> k+2): back to the room's children.
  EXPECT_EQ(Referents(tree, 10.01, 7.5),
            (std::set<element_id_t>{kRegion, kTransfer, kLeaf}));
}

TEST(BorderReferent, TheRule) {
  ElementTree tree;
  BuildTree(&tree);
  EXPECT_TRUE(ScopeResolver::HasBorderReferent(tree, kRegion))
      << "interior solved by relations: it has objects";
  EXPECT_FALSE(ScopeResolver::HasBorderReferent(tree, kRoom)) << "the room";
  EXPECT_FALSE(ScopeResolver::HasBorderReferent(tree, kTransfer))
      << "a frame-scope element: containment transfer carries the actor";
  EXPECT_FALSE(ScopeResolver::HasBorderReferent(tree, kLeaf)) << "a leaf is itself";
  EXPECT_FALSE(ScopeResolver::HasBorderReferent(tree, kObjA)) << "an object, no children";
  EXPECT_FALSE(ScopeResolver::HasBorderReferent(tree, 999)) << "no such element";
}

// A flagged region that carries an object (Chess, Whiteboard) is still a transfer leaf. It has no
// border referent and is the whole scope of an actor inside it; the object it carries is
// not a referent below it. Cleared, it is an ordinary scope container again.
TEST(BorderReferent, AFlaggedRegionThatCarriesAnObjectIsStillALeaf) {
  ElementTree tree;
  BuildTree(&tree);
  ASSERT_TRUE(tree.AddElement(8, kTransfer).ok());
  tree.GetElement(8)->SetSharedId(8);
  tree.GetElement(8)->SetSpatialDefinition(Box(12.5, 6, 13, 7, GeometryRole::kObstacle));
  EXPECT_FALSE(ScopeResolver::HasBorderReferent(tree, kTransfer))
      << "flagged, it carries its object and is a leaf";
  EXPECT_EQ(Referents(tree, 12.75, 6.5), (std::set<element_id_t>{kTransfer}))
      << "the leaf alone is the scope";
  tree.GetElement(kTransfer)->SetContainmentTransfer(false);
  EXPECT_TRUE(ScopeResolver::HasBorderReferent(tree, kTransfer));
  EXPECT_EQ(Referents(tree, 13.5, 9), (std::set<element_id_t>{8, kTransfer}))
      << "unflagged, an ordinary scope container: its object and its border";
  EXPECT_EQ(Referents(tree, 12.75, 6.5), (std::set<element_id_t>{8}))
      << "inside the object it carries: the object alone, replayed";
}

TEST(BorderReferent, InteractionRegionsAreNeverSpacingReferents) {
  ElementTree tree;
  BuildTree(&tree);
  EXPECT_EQ(Referents(tree, 6.5, 8.5).count(kAffordance), 0u);
}

}  // namespace test
}  // namespace spatialblend
