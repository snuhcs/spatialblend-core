// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/scope_cases_test.cc
//
// The scope case catalog as tests. SCOPE_CASES.md lists every scope rule as a named case; this file
// is its UNIT level: each case builds the geometry and asserts the exact scope set (and the mode)
// of the scope layer, independent of the solver (no hub, no receiver, no proposal). The source
// side runs the real Space pipeline up to the in-scope matches (the structural scope, the
// family gates, the people), which is the whole scope layer; nothing after it.
//
// Naming: every case id ("SP-INSIDE-02") appears in its test name with '-' written '_'
// ("ScopeCases.SP_INSIDE_02_..."). CatalogCoverage.EveryCaseHasATest cross-checks the catalog
// against the registered tests and fails when a case has none.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "spatialblend/model/element/containment_layout.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/model/element/sub_region.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/relation/scope_gate.h"
#include "spatialblend/model/relation/scope_resolver.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/solver/affordance_pose_sd.h"
#include "spatialblend/model/embodiment/embodiment.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace test {
namespace {

using Ids = std::set<element_id_t>;

std::shared_ptr<Polygon2D> Rect(double x0, double y0, double x1, double y1) {
  return std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});
}

Eigen::Matrix4d At(double x, double y) {
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = x;
  t(1, 3) = y;
  return t;
}

// An axis box [x0,x1]x[y0,y1], height h, as a Polytope in the element's local frame.
void Box(ElementTree* tree, element_id_t id, element_id_t parent, double x0, double y0, double x1,
         double y1, GeometryRole role, double h = 0.8) {
  ASSERT_TRUE(tree->AddElement(id, parent).ok());
  // Every element of these worlds is shared (D7: an unshared element is never a referent, SP-D7-03).
  tree->GetElement(id)->SetSharedId(id);
  // World coordinates in, the parent-relative local transform out.
  const double px = parent == kInvalidElementId ? 0.0 : tree->GetWorldTransform(parent)(0, 3);
  const double py = parent == kInvalidElementId ? 0.0 : tree->GetWorldTransform(parent)(1, 3);
  tree->GetElement(id)->SetLocalTransform(At(0.5 * (x0 + x1) - px, 0.5 * (y0 + y1) - py));
  tree->GetElement(id)->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-0.5 * (x1 - x0), -0.5 * (y1 - y0), 0.0),
      Eigen::Vector3d(0.5 * (x1 - x0), 0.5 * (y1 - y0), h), role));
}

// Element ids of the standard world.
enum : element_id_t {
  kRoom = 1,
  kR1 = 2, kO1 = 3, kO2 = 4, kAff1 = 5,   // region with two objects and an interaction region
  kR2 = 6,                                // empty region (no children)
  kL1 = 7, kOL = 8,                       // flagged transfer leaf carrying an object
  kT1 = 9, kC1 = 10,                      // free-standing non-region referents (table, chair)
  kA = 11, kB = 12,                       // non-region footprints that touch (they may not overlap)
  kR3 = 13, kR3a = 14, kR3aO = 15,        // nested regions
  kR4 = 16, kR4I = 17,                    // region with only an interaction child
  kSelf = 20,                             // the subject's own element
};

// The standard world, a 30 x 30 m room:
//   R1 (2,2)-(8,8) containment: O1 (3,3)-(4,4), O2 (6,6)-(7,7) obstacles, Aff1 (3,5)-(4,6) interaction
//   R2 (10,2)-(14,8) containment, empty
//   L1 (16,2)-(19,8) containment, flagged: carries OL (17,4)-(18,5)
//   T1 (2,12)-(5,14) and C1 (6,12)-(7,14) obstacles; A (10,12)-(13,15) and B (12,12)-(15,15) overlap
//   R3 (2,16)-(8,22) containment: R3a (3,17)-(6,21) containment: R3aO (4,18)-(5,19) obstacle
//   R4 (22,2)-(26,8) containment with an interaction-region child R4I (23,3)-(24,4)
struct World {
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space* space = ctx->AddSpace();
  spatialblend::entity_id_t actor = 0;
  // `actor_parent`: where the subject's own element sits in the tree. It is a person leaf and the
  // scope never depends on it (SP-TREE-04); the default is directly under the room.
  explicit World(element_id_t actor_parent = kRoom) {
    space->EnableObservabilityCapture(true);
    auto* t = space->GetElementTree();
    EXPECT_TRUE(t->AddElement(kRoom).ok());
    t->GetElement(kRoom)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 30, 30)));
    Box(t, kR1, kRoom, 2, 2, 8, 8, GeometryRole::kContainment);
    Box(t, kO1, kR1, 3, 3, 4, 4, GeometryRole::kObstacle);
    Box(t, kO2, kR1, 6, 6, 7, 7, GeometryRole::kObstacle);
    Box(t, kAff1, kR1, 3, 5, 4, 6, GeometryRole::kInteractionRegion);
    Box(t, kR2, kRoom, 10, 2, 14, 8, GeometryRole::kContainment);
    Box(t, kL1, kRoom, 16, 2, 19, 8, GeometryRole::kContainment);
    Box(t, kOL, kL1, 17, 4, 18, 5, GeometryRole::kObstacle);
    Box(t, kT1, kRoom, 2, 12, 5, 14, GeometryRole::kObstacle);
    Box(t, kC1, kRoom, 6, 12, 7, 14, GeometryRole::kObstacle);
    Box(t, kA, kRoom, 10, 12, 13, 15, GeometryRole::kObstacle);
    Box(t, kB, kRoom, 13, 12, 16, 15, GeometryRole::kObstacle);  // touches A along x = 13
    Box(t, kR3, kRoom, 2, 16, 8, 22, GeometryRole::kContainment);
    Box(t, kR3a, kR3, 3, 17, 6, 21, GeometryRole::kContainment);
    Box(t, kR3aO, kR3a, 4, 18, 5, 19, GeometryRole::kObstacle);
    Box(t, kR4, kRoom, 22, 2, 26, 8, GeometryRole::kContainment);
    Box(t, kR4I, kR4, 23, 3, 24, 4, GeometryRole::kInteractionRegion);
    EXPECT_TRUE(space->SetElementContainmentTransfer(kL1, true));
    EXPECT_TRUE(t->AddElement(kSelf, actor_parent).ok());
    actor = space->CreateActor({SubjectAnchor("root", {0, 0, 0}), SubjectAnchor("chest", {0, 0, 1.2})});
    FollowProposals(space, actor);
    space->GetActor(actor)->SetElementId(kSelf);
    space->Bind(actor, "root", RelationKernelType::kAvatarSpacing);
    space->Bind(actor, "chest", RelationKernelType::kAvatarFacing);
  }
  ElementTree* Tree() { return space->GetElementTree(); }
  // A person: another actor whose element sits at (x, y) under `parent`.
  void Person(element_id_t id, element_id_t parent, double x, double y) {
    ASSERT_TRUE(Tree()->AddElement(id, parent).ok());
    Tree()->GetElement(id)->SetSharedId(id);
    Tree()->GetElement(id)->SetLocalTransform(
        At(x - Tree()->GetWorldTransform(parent)(0, 3), y - Tree()->GetWorldTransform(parent)(1, 3)));
    const auto other = space->CreateActor({SubjectAnchor("root", {0, 0, 0})});
    FollowProposals(space, other);
    space->GetActor(other)->SetElementId(id);
  }
  void Go(double x, double y) { space->UpdateActorState(actor, Eigen::Vector2d(x, y), 0.0); }
  Ids Matched(size_t binding) const {
    Ids out;
    for (const auto& m : space->GetBindings(actor)[binding].latest_matches) out.insert(m.element_id);
    return out;
  }
  Ids Spacing() const { return Matched(0); }
  Ids Facing() const { return Matched(1); }
  element_id_t Container() const {
    return space->GetBindings(actor)[0].scope_record.deepest_containing;
  }
};

// The three modes of SCOPE_CASES.md: OUTSIDE (no referent entered), INTERIOR (a) and REPLAY (b).
std::string ModeOf(const World& w) {
  const auto& s = w.space->GetBindings(w.actor)[0].spacing_scope;
  return s ? SpacingScopeModeName(s->mode) : "none";
}

// ---------------------------------------------------------------------------------------------
// Spacing: the containment tree and the border referent
// ---------------------------------------------------------------------------------------------

TEST(ScopeCases, SP_TREE_01_PoseInTheRoomScopesTheRoomsChildren) {
  World w;
  w.Go(1, 26);
  EXPECT_EQ(w.Spacing(), (Ids{kR1, kR2, kL1, kT1, kC1, kA, kB, kR3, kR4}));
  EXPECT_EQ(w.Container(), kRoom);
  EXPECT_EQ(ModeOf(w), "OUTSIDE");
}

TEST(ScopeCases, SP_TREE_02_NestedContainersScopeTheDeepestOne) {
  World w;
  w.Go(3.2, 20);  // in R3a, outside its object
  EXPECT_EQ(w.Spacing(), (Ids{kR3aO, kR3a}));
  EXPECT_EQ(w.Container(), kR3a);
  EXPECT_EQ(ModeOf(w), "INTERIOR");
  w.Go(7, 20);  // in R3, outside R3a
  EXPECT_EQ(w.Spacing(), (Ids{kR3a, kR3}));
  EXPECT_EQ(w.Container(), kR3);
  EXPECT_EQ(ModeOf(w), "INTERIOR");
  w.Go(4.5, 18.5);  // in the object of R3a: the object is the container
  EXPECT_EQ(w.Spacing(), (Ids{kR3aO}));
  EXPECT_EQ(w.Container(), kR3aO);
  EXPECT_EQ(ModeOf(w), "REPLAY");
}

TEST(ScopeCases, SP_TREE_03_InteractionRegionsAreNeverSpacingReferents) {
  World w;
  w.Go(3.5, 5.5);  // inside Aff1, inside R1
  EXPECT_EQ(w.Spacing(), (Ids{kO1, kO2, kR1}));
  EXPECT_EQ(w.Container(), kR1);
  EXPECT_EQ(ModeOf(w), "INTERIOR");
}

// The actor element is the subject's person
// leaf, never a region, and the scope descent starts at the tree's navigable root. Where the leaf sits in
// the tree (under the room, under a region, under a transfer leaf) does not move the scope: a pose has
// the same scope whatever its person leaf hangs from. (Descending from the leaf's nearest containment
// ancestor would start inside that ancestor and never test it: under a leaf the scope of
// a pose in the open would be the leaf and no transfer would ever be flagged for a pose inside it.)
TEST(ScopeCases, SP_TREE_04_TheActorElementIsAPersonLeafAndTheScopeDoesNotDependOnWhereItHangs) {
  struct Pose { double x, y; Ids set; const char* mode; element_id_t container; };
  const std::vector<Pose> poses = {
      {1, 26, {kR1, kR2, kL1, kT1, kC1, kA, kB, kR3, kR4}, "OUTSIDE", kRoom},
      {5, 4, {kO1, kO2, kR1}, "INTERIOR", kR1},
      {17.5, 6, {kL1}, "REPLAY", kL1},
      {3.2, 20, {kR3aO, kR3a}, "INTERIOR", kR3a},
      {12, 5, {kR2}, "REPLAY", kR2},
  };
  for (const element_id_t parent : {kRoom, kR1, kL1, kR3a, kR2}) {
    World w(parent);
    for (const Pose& p : poses) {
      w.Go(p.x, p.y);
      EXPECT_EQ(w.Spacing(), p.set) << "actor element under " << parent << ", pose (" << p.x << "," << p.y << ")";
      EXPECT_EQ(ModeOf(w), p.mode) << "actor element under " << parent << ", pose (" << p.x << "," << p.y << ")";
      EXPECT_EQ(w.Container(), p.container) << "actor element under " << parent << ", pose (" << p.x << "," << p.y << ")";
    }
    // The transfer flag follows the source root, not the leaf's place in the tree.
    w.Go(17.5, 6);
    ASSERT_TRUE(w.space->GetBindings(w.actor)[0].spacing_scope.has_value());
    EXPECT_EQ(w.space->GetBindings(w.actor)[0].spacing_scope->transfer_leaf, kL1) << "under " << parent;
    w.Go(1, 26);
    EXPECT_EQ(w.space->GetBindings(w.actor)[0].spacing_scope->transfer_leaf, kInvalidElementId)
        << "under " << parent;
  }
}

TEST(ScopeCases, SP_BORDER_01_AContainerWithAnObjectHasABorderReferent) {
  World w;
  EXPECT_TRUE(ScopeResolver::HasBorderReferent(*w.Tree(), kR1));
  EXPECT_TRUE(ScopeResolver::HasBorderReferent(*w.Tree(), kR3));
}

TEST(ScopeCases, SP_BORDER_02_TheRoomHasNoBorderReferent) {
  World w;
  EXPECT_FALSE(ScopeResolver::HasBorderReferent(*w.Tree(), kRoom));
}

TEST(ScopeCases, SP_BORDER_03_ATransferLeafHasNoBorderReferent) {
  World w;
  EXPECT_FALSE(ScopeResolver::HasBorderReferent(*w.Tree(), kL1));
  w.space->SetElementContainmentTransfer(kL1, false);
  EXPECT_TRUE(ScopeResolver::HasBorderReferent(*w.Tree(), kL1)) << "unflagged it carries OL";
}

TEST(ScopeCases, SP_BORDER_04_AContainerWithoutChildrenHasNoBorderReferent) {
  World w;
  EXPECT_FALSE(ScopeResolver::HasBorderReferent(*w.Tree(), kR2));
  w.Go(12, 5);
  EXPECT_EQ(w.Spacing(), (Ids{kR2})) << "the empty container is its own referent";
}

TEST(ScopeCases, SP_BORDER_05_ContainerWithOnlyInteractionChildrenHasNoBorderReferent) {
  World w;
  EXPECT_FALSE(ScopeResolver::HasBorderReferent(*w.Tree(), kR4));
  w.Go(24.5, 5);
  EXPECT_EQ(w.Spacing(), (Ids{kR4}));
  EXPECT_EQ(w.Container(), kR4);
}

TEST(ScopeCases, SP_BORDER_06_AnElementWithoutAFootprintHasNoBorderReferent) {
  World w;
  EXPECT_FALSE(ScopeResolver::HasBorderReferent(*w.Tree(), kSelf));
  EXPECT_FALSE(ScopeResolver::HasBorderReferent(*w.Tree(), 999));
}

// ---------------------------------------------------------------------------------------------
// Spacing: the transfer leaf (D5)
// ---------------------------------------------------------------------------------------------

TEST(ScopeCases, SP_LEAF_01_TheTransferLeafIsTheWholeScopeAndTheObjectItCarriesIsNotBelowIt) {
  World w;
  w.Go(16.5, 7);  // in L1, away from OL
  EXPECT_EQ(w.Spacing(), (Ids{kL1}));
  EXPECT_EQ(w.Container(), kL1);
  EXPECT_EQ(ModeOf(w), "REPLAY");
  w.Go(17.5, 4.5);  // inside the carried object: still the leaf
  EXPECT_EQ(w.Spacing(), (Ids{kL1}));
  EXPECT_EQ(w.Container(), kL1);
  EXPECT_EQ(ModeOf(w), "REPLAY");
}

TEST(ScopeCases, SP_LEAF_02_ALeafTransfersOnlyBetweenCongruentFootprints) {
  const auto a = Rect(0, 0, 3, 6);
  const auto moved = Rect(10, 5, 13, 11);
  const auto rotated = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{20, 20}, {20, 23}, {14, 23}, {14, 20}});
  const auto other = Rect(0, 0, 3, 6.5);
  EXPECT_TRUE(FootprintsCongruent(*a, *moved));
  EXPECT_TRUE(FootprintsCongruent(*a, *rotated)) << "rotation + translation";
  EXPECT_FALSE(FootprintsCongruent(*a, *other)) << "a different shape does not transfer";
}

// ---------------------------------------------------------------------------------------------
// Spacing: source-side validity (D7), reachability (D8), people (D9)
// ---------------------------------------------------------------------------------------------

// A walkable L with a sealed notch; the referent sits in the notch.
struct LWorld {
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space* space = ctx->AddSpace();
  spatialblend::entity_id_t actor = 0;
  explicit LWorld(bool referent_in_notch) {
    auto* t = space->GetElementTree();
    EXPECT_TRUE(t->AddElement(1).ok());
    t->GetElement(1)->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(
        std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
            {0, 0}, {10, 0}, {10, 6}, {6, 6}, {6, 10}, {0, 10}})));
    if (referent_in_notch)
      Box(t, 2, 1, 7.5, 7.5, 8.5, 8.5, GeometryRole::kObstacle);
    else
      Box(t, 2, 1, 7.5, 1.5, 8.5, 2.5, GeometryRole::kObstacle);
    EXPECT_TRUE(t->AddElement(20, 1).ok());
    actor = space->CreateActor({SubjectAnchor("root", {0, 0, 0})});
    FollowProposals(space, actor);
    space->GetActor(actor)->SetElementId(20);
    space->Bind(actor, "root", RelationKernelType::kAvatarSpacing);
    space->UpdateActorState(actor, Eigen::Vector2d(2, 2), 0.0);
  }
  Ids Spacing() const {
    Ids out;
    for (const auto& m : space->GetBindings(actor)[0].latest_matches) out.insert(m.element_id);
    return out;
  }
};

TEST(ScopeCases, SP_D7_01_AReferentTheSourceCannotReachIsNotInTheSourcesScope) {
  LWorld sealed(true), open(false);
  EXPECT_EQ(sealed.Spacing(), Ids{}) << "decided on the source side, before anything is sent";
  EXPECT_EQ(open.Spacing(), (Ids{2}));
}

TEST(ScopeCases, SP_D7_02_WhenEveryReferentDropsTheScopeIsEmptyAndSaysSo) {
  LWorld sealed(true);
  const auto& cands = sealed.space->GetBindings(sealed.actor)[0].scope_record.candidates;
  (void)cands;  // the record is captured only while observed; the matches are the contract
  EXPECT_TRUE(sealed.Spacing().empty()) << "an empty scope: the receiver's empty-scope mapping, no HOLD";
}

// An obstacle with no shared id in the standard world (an unshared wall, pillar or clutter).
void AddUnshared(World* w) {
  auto* t = w->Tree();
  Box(t, 50, kRoom, 20, 12, 22, 14, GeometryRole::kObstacle);
  t->GetElement(50)->SetSharedId(kInvalidSharedElementId);
}

TEST(ScopeCases, SP_D7_03_AnUnsharedElementIsNeverAReferent) {
  World w;
  AddUnshared(&w);
  // an unshared object inside the empty region R2: no interior entity
  Box(w.Tree(), 51, kR2, 11, 3, 12, 4, GeometryRole::kObstacle);
  w.Tree()->GetElement(51)->SetSharedId(kInvalidSharedElementId);
  w.Go(1, 26);
  EXPECT_EQ(w.Spacing(), (Ids{kR1, kR2, kL1, kT1, kC1, kA, kB, kR3, kR4})) << "U (50) is no referent";
  EXPECT_EQ(ModeOf(w), "OUTSIDE");
  w.Go(12, 5);
  EXPECT_EQ(w.Spacing(), Ids{kR2});
  EXPECT_EQ(ModeOf(w), "REPLAY");
  // the same obstacle WITH a shared id is a referent
  World shared;
  Box(shared.Tree(), 50, kRoom, 20, 12, 22, 14, GeometryRole::kObstacle);
  shared.Go(1, 26);
  EXPECT_EQ(shared.Spacing().count(50), 1u);
  // AffordancePose: the proximal subtree drops it too
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  Box(&tree, 2, 1, 2, 2, 8, 8, GeometryRole::kContainment);
  Box(&tree, 3, 2, 4, 4, 5, 5, GeometryRole::kInteractionRegion, 0.5);
  Box(&tree, 4, 2, 6, 6, 7, 7, GeometryRole::kInteractionRegion, 0.5);
  tree.GetElement(4)->SetSharedId(kInvalidSharedElementId);
  const auto ids = ScopeResolver::ResolveProximal(Eigen::Vector3d(3, 3, 0), 1, tree, 2);
  EXPECT_EQ(std::count(ids.begin(), ids.end(), 3), 1);
  EXPECT_EQ(std::count(ids.begin(), ids.end(), 4), 0);
}

// An unshared element is never a referent, also as Spacing's ENTERED referent (SP-D7-05): an unshared element that carries nothing shared is TRANSPARENT to the Spacing scope. It is exactly as if it
// did not exist: the descent neither enters nor stops in it, and the scope of a pose inside it is the scope of the same pose in the tree
// with that element deleted (the nearest enclosing container that exists for the scope, its OWN objects and border), not a replay of the
// ancestor alone. It stays an obstacle (SP-D7-03, F is built from the tree) and a shared element below an unshared region keeps that
// region as its container with its border (space_test pins it).
TEST(ScopeCases, SP_D7_04_AnUnsharedLeafTheRootStandsInIsAsIfItWereDeleted) {
  // SP-D7-04 (with SP-D7-05): the same world with and without the unshared leaf.
  World w;
  AddUnshared(&w);             // U (50): (20,12)-(22,14), an obstacle leaf, no shared id
  World deleted;               // the same world with U deleted
  for (const auto& pose : {std::pair<double, double>{21, 13}, {20.1, 12.1}, {1, 26}, {12, 5}}) {
    w.Go(pose.first, pose.second);
    deleted.Go(pose.first, pose.second);
    EXPECT_EQ(w.Spacing(), deleted.Spacing()) << pose.first << "," << pose.second;
    EXPECT_EQ(ModeOf(w), ModeOf(deleted)) << pose.first << "," << pose.second;
    EXPECT_EQ(w.Container(), deleted.Container()) << pose.first << "," << pose.second;
  }
  w.Go(21, 13);                // the root stands inside U
  EXPECT_EQ(w.Spacing().count(50), 0u) << "the unshared leaf the root stands in is no Spacing referent";
  EXPECT_EQ(w.Spacing(), (Ids{kR1, kR2, kL1, kT1, kC1, kA, kB, kR3, kR4})) << "the room's own scope, not a replay of the room";
  EXPECT_EQ(ModeOf(w), "OUTSIDE");

  // The shared control: the same leaf WITH a shared id is the replay scope, unchanged.
  World shared;
  Box(shared.Tree(), 50, kRoom, 20, 12, 22, 14, GeometryRole::kObstacle);
  shared.Go(21, 13);
  EXPECT_EQ(shared.Spacing(), Ids{50});
  EXPECT_EQ(ModeOf(shared), "REPLAY");
}

TEST(ScopeCases, SP_D7_04_TheResolverSkipsAnUnsharedContainerThatCarriesNothingSharedInEveryBranch) {
  // SP-D7-04, straight on the resolver, in a tree whose room has NO shared id: an unshared region U holding one shared object, an
  // unshared leaf L, and a flagged unshared leaf F. L and F carry nothing shared, so they do not exist for the scope: a pose in them has
  // the room's own scope (its shared objects), the scope of the tree with them deleted (the equivalence tests are SP-D7-05).
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  tree.GetElement(1)->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 30, 30)));
  Box(&tree, 2, 1, 2, 2, 8, 8, GeometryRole::kContainment);       // U: an unshared region
  tree.GetElement(2)->SetSharedId(kInvalidSharedElementId);
  Box(&tree, 3, 2, 3, 3, 4, 4, GeometryRole::kObstacle);          // a shared object of U
  Box(&tree, 4, 1, 10, 2, 14, 8, GeometryRole::kObstacle);        // L: an unshared leaf
  tree.GetElement(4)->SetSharedId(kInvalidSharedElementId);
  Box(&tree, 5, 1, 16, 2, 19, 8, GeometryRole::kContainment);     // F: a flagged unshared leaf
  tree.GetElement(5)->SetSharedId(kInvalidSharedElementId);
  tree.GetElement(5)->SetContainmentTransfer(true);
  Box(&tree, 6, 1, 20, 2, 24, 8, GeometryRole::kObstacle);        // a shared leaf: the control

  // The unshared region: its shared object is the scope and its border stays (the container border is unaffected).
  const auto in_u = ScopeResolver::ResolveSpacing(Eigen::Vector3d(5, 5, 0), 1, tree);
  EXPECT_EQ(in_u.container, 2);
  EXPECT_EQ(in_u.mode, SpacingScopeMode::kInterior);
  EXPECT_EQ(in_u.referents, (std::vector<element_id_t>{3, 2}));
  EXPECT_TRUE(in_u.border);
  // The unshared leaf and the flagged unshared leaf are skipped: the room's own scope, the shared leaf 6 (OUTSIDE).
  for (const double x : {12.0, 17.0}) {
    const auto s = ScopeResolver::ResolveSpacing(Eigen::Vector3d(x, 5, 0), 1, tree);
    EXPECT_EQ(s.referents, (std::vector<element_id_t>{6})) << x;
    EXPECT_EQ(s.mode, SpacingScopeMode::kOutside) << x;
    EXPECT_EQ(s.container, 1) << x;
    EXPECT_EQ(s.transfer_leaf, kInvalidElementId) << x << ": an unshared flagged leaf is no transfer leaf of the scope";
  }
  // The shared control.
  EXPECT_EQ(ScopeResolver::ResolveSpacing(Eigen::Vector3d(22, 5, 0), 1, tree).referents,
            (std::vector<element_id_t>{6}));

  // ResolveReachable with the container given (the kernel's ResolveScope): the fallback, the flagged branch, the border.
  const Eigen::Vector3d at(0, 0, 0);
  EXPECT_EQ(ScopeResolver::ResolveReachable(at, kInvalidElementId, tree, 4), (std::vector<element_id_t>{6})) << "unshared leaf: the room's";
  EXPECT_EQ(ScopeResolver::ResolveReachable(at, kInvalidElementId, tree, 5), (std::vector<element_id_t>{6})) << "unshared flagged leaf: the room's";
  EXPECT_EQ(ScopeResolver::ResolveReachable(at, kInvalidElementId, tree, 2), (std::vector<element_id_t>{3, 2}))
      << "unshared region: the shared object and the container border, unchanged";
  EXPECT_EQ(ScopeResolver::ResolveReachable(at, kInvalidElementId, tree, 6), (std::vector<element_id_t>{6}))
      << "shared leaf: the fallback is the leaf";

  // The two scope-root branches no pose in the tree above reaches: a flagged scope root (D5), and a root with no object at all (the
  // degenerate fallback). Unshared: no referent; with a shared id: the root itself, as before.
  for (const bool shared : {false, true}) {
    ElementTree flagged;
    ASSERT_TRUE(flagged.AddElement(1).ok());
    if (shared) flagged.GetElement(1)->SetSharedId(1);
    flagged.GetElement(1)->SetContainmentTransfer(true);
    const auto f = ScopeResolver::ResolveSpacing(Eigen::Vector3d(1, 1, 0), 1, flagged);
    EXPECT_EQ(f.mode, SpacingScopeMode::kOutside);
    EXPECT_EQ(f.referents, shared ? std::vector<element_id_t>{1} : std::vector<element_id_t>{}) << "flagged scope root, shared=" << shared;
    ElementTree bare;
    ASSERT_TRUE(bare.AddElement(1).ok());
    if (shared) bare.GetElement(1)->SetSharedId(1);
    const auto b = ScopeResolver::ResolveSpacing(Eigen::Vector3d(1, 1, 0), 1, bare);
    EXPECT_EQ(b.mode, SpacingScopeMode::kOutside);
    EXPECT_EQ(b.referents, shared ? std::vector<element_id_t>{1} : std::vector<element_id_t>{}) << "root with no object, shared=" << shared;
  }
}

// SP-D7-05: the equivalence, on the resolver. One tree twice: with the unshared elements, and with them deleted (every id of the shared
// elements is the same in both). The scope of every pose is identical field by field.
//   room (1, shared, navigable)
//    +- S (2, shared region)
//    |   +- U1 (3, UNSHARED region, nothing shared below) > U2 (4, UNSHARED leaf)
//    |   +- M (5, shared object of S, far from U1)
//    +- F (6, UNSHARED flagged leaf)
//    +- V1 (7, UNSHARED region) > V2 (8, UNSHARED leaf)       (directly under the room)
//    +- O (9, shared OBSTACLE) > U3 (10, UNSHARED leaf)       (the descent enters an obstacle too)
//    +- W (11, UNSHARED region) > X (12, shared leaf)          (an unshared region holding something shared is NOT transparent)
//    +- Y1 (13, UNSHARED region) > Y2 (14, UNSHARED region) > Z (15, shared obstacle)   (something shared nested BELOW Y1, none in it)
void BuildTransparencyTree(ElementTree* tree, bool with_unshared) {
  ASSERT_TRUE(tree->AddElement(1).ok());
  tree->GetElement(1)->SetSharedId(1);
  tree->GetElement(1)->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 50, 30)));
  Box(tree, 2, 1, 2, 2, 14, 10, GeometryRole::kContainment);   // S
  Box(tree, 5, 2, 10, 3, 12, 5, GeometryRole::kObstacle);      // M
  Box(tree, 9, 1, 32, 2, 38, 10, GeometryRole::kObstacle);     // O
  Box(tree, 11, 1, 40, 2, 48, 10, GeometryRole::kContainment); // W
  tree->GetElement(11)->SetSharedId(kInvalidSharedElementId);
  Box(tree, 12, 11, 42, 3, 44, 5, GeometryRole::kObstacle);    // X
  if (!with_unshared) return;
  Box(tree, 13, 1, 2, 14, 12, 24, GeometryRole::kContainment);  // Y1
  tree->GetElement(13)->SetSharedId(kInvalidSharedElementId);
  Box(tree, 14, 13, 4, 16, 10, 22, GeometryRole::kContainment); // Y2
  tree->GetElement(14)->SetSharedId(kInvalidSharedElementId);
  Box(tree, 15, 14, 5, 17, 8, 20, GeometryRole::kObstacle);     // Z
  const auto unshared = [&](element_id_t id) { tree->GetElement(id)->SetSharedId(kInvalidSharedElementId); };
  Box(tree, 3, 2, 3, 3, 8, 9, GeometryRole::kContainment);     // U1
  unshared(3);
  Box(tree, 4, 3, 4, 4, 6, 6, GeometryRole::kObstacle);        // U2
  unshared(4);
  Box(tree, 6, 1, 16, 2, 19, 8, GeometryRole::kContainment);   // F
  unshared(6);
  tree->GetElement(6)->SetContainmentTransfer(true);
  Box(tree, 7, 1, 22, 2, 30, 10, GeometryRole::kContainment);  // V1
  unshared(7);
  Box(tree, 8, 7, 24, 4, 26, 6, GeometryRole::kObstacle);      // V2
  unshared(8);
  Box(tree, 10, 9, 33, 3, 35, 5, GeometryRole::kObstacle);     // U3
  unshared(10);
}

TEST(ScopeCases, SP_D7_05_TheScopeInsideAnUnsharedContainerIsTheScopeOfTheTreeWithItDeleted) {
  // SP-D7-05
  ElementTree with, without;
  BuildTransparencyTree(&with, true);
  BuildTransparencyTree(&without, false);
  struct Pose { double x, y; const char* what; std::vector<element_id_t> referents; SpacingScopeMode mode; element_id_t container; };
  const std::vector<Pose> poses = {
      {5, 5, "in U2 (unshared in unshared in S)", {5, 2}, SpacingScopeMode::kInterior, 2},
      {7, 8, "in U1 outside U2", {5, 2}, SpacingScopeMode::kInterior, 2},
      {25, 5, "in V2 (unshared in unshared in the room)", {2, 9}, SpacingScopeMode::kOutside, 1},
      {23, 9, "in V1", {2, 9}, SpacingScopeMode::kOutside, 1},
      {17, 5, "in the unshared FLAGGED leaf F", {2, 9}, SpacingScopeMode::kOutside, 1},
      {34, 4, "in U3 (unshared in a shared obstacle)", {9}, SpacingScopeMode::kReplay, 9},
      {11, 4, "in the shared leaf M (control)", {5}, SpacingScopeMode::kReplay, 5},
      {13, 8, "in S, outside its children (control)", {5, 2}, SpacingScopeMode::kInterior, 2},
      {43, 4, "in X, a shared leaf of an UNSHARED region (control: W is not transparent)", {12}, SpacingScopeMode::kReplay, 12},
      {3, 15, "in Y1 outside Y2 (nothing shared in it, something shared nested below)", {2, 9}, SpacingScopeMode::kOutside, 1},
      {41, 8, "in W outside X (control: W keeps its shared object and its border)", {12, 11}, SpacingScopeMode::kInterior, 11},
  };
  for (const Pose& pose : poses) {
    const Eigen::Vector3d at(pose.x, pose.y, 0);
    const auto a = ScopeResolver::ResolveSpacing(at, 1, with), b = ScopeResolver::ResolveSpacing(at, 1, without);
    EXPECT_EQ(a.referents, b.referents) << pose.what;
    EXPECT_EQ(a.mode, b.mode) << pose.what;
    EXPECT_EQ(a.container, b.container) << pose.what;
    EXPECT_EQ(a.entered, b.entered) << pose.what;
    EXPECT_EQ(a.transfer_leaf, b.transfer_leaf) << pose.what;
    EXPECT_EQ(a.border, b.border) << pose.what;
    EXPECT_EQ(a.referents, pose.referents) << pose.what;
    EXPECT_EQ(a.mode, pose.mode) << pose.what;
    EXPECT_EQ(a.container, pose.container) << pose.what;
  }
  // Z, shared and nested below two unshared regions, is still the replay scope of a pose inside it (no deleted-tree counterpart: Z is gone there).
  const auto in_z = ScopeResolver::ResolveSpacing(Eigen::Vector3d(6, 18, 0), 1, with);
  EXPECT_EQ(in_z.container, 15);
  EXPECT_EQ(in_z.referents, (std::vector<element_id_t>{15}));
  EXPECT_EQ(in_z.mode, SpacingScopeMode::kReplay);
  // ResolveReachable with the container given (the kernel's ResolveScope): an unshared container with nothing shared resolves at the
  // nearest ancestor that exists for the scope, the same answer as for that ancestor in the tree without it.
  const Eigen::Vector3d at(0, 0, 0);
  const std::vector<std::pair<element_id_t, element_id_t>> given = {{4, 2}, {3, 2}, {6, 1}, {8, 1}, {7, 1}, {10, 9}, {13, 1}};
  for (const auto& [unshared, ancestor] : given)
    EXPECT_EQ(ScopeResolver::ResolveReachable(at, kInvalidElementId, with, unshared),
              ScopeResolver::ResolveReachable(at, kInvalidElementId, without, ancestor))
        << "container " << unshared << " resolves as " << ancestor;
  EXPECT_EQ(ScopeResolver::ResolveReachable(at, kInvalidElementId, with, 14), (std::vector<element_id_t>{15, 14}))
      << "an unshared region that holds a shared object keeps it and its border";
  EXPECT_EQ(ScopeResolver::ResolveReachable(at, kInvalidElementId, with, 11), (std::vector<element_id_t>{12, 11}))
      << "an unshared region that holds a shared element keeps it and its border";
}

// The scope of a pose, field by field, against the scope of the same pose in the equivalent tree (SP-D7-05).
void ExpectSameScope(const SpacingScope& a, const SpacingScope& b, const std::string& what) {
  EXPECT_EQ(a.referents, b.referents) << what;
  EXPECT_EQ(a.mode, b.mode) << what;
  EXPECT_EQ(a.container, b.container) << what;
  EXPECT_EQ(a.entered, b.entered) << what;
  EXPECT_EQ(a.transfer_leaf, b.transfer_leaf) << what;
  EXPECT_EQ(a.border, b.border) << what;
}

// SP-D7-05, corner 1: a person (a shared element with no footprint) directly under an unshared container is no object of it and does not
// make it a scope. The tree twice: the person under the unshared region U, and the person under the room with U deleted. People are
// Spacing referents at any depth (D9), so the scope is the same either way.
TEST(ScopeCases, SP_D7_05_APersonUnderAnUnsharedContainerDoesNotMakeItAScope) {
  // SP-D7-05
  for (const bool under_unshared : {true, false}) {
    ElementTree tree;
    ASSERT_TRUE(tree.AddElement(1).ok());
    tree.GetElement(1)->SetSharedId(1);
    tree.GetElement(1)->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 30, 30)));
    Box(&tree, 2, 1, 20, 2, 26, 8, GeometryRole::kObstacle);  // M: a shared object of the room
    if (under_unshared) {
      Box(&tree, 3, 1, 2, 2, 10, 10, GeometryRole::kContainment);  // U: unshared, holds only the person
      tree.GetElement(3)->SetSharedId(kInvalidSharedElementId);
    }
    ASSERT_TRUE(tree.AddElement(4, under_unshared ? 3 : 1).ok());  // the person: shared, no footprint, at (5,5)
    tree.GetElement(4)->SetSharedId(4);
    const double px = under_unshared ? tree.GetWorldTransform(3)(0, 3) : 0.0, py = under_unshared ? tree.GetWorldTransform(3)(1, 3) : 0.0;
    tree.GetElement(4)->SetLocalTransform(At(5 - px, 5 - py));
    const auto scope = ScopeResolver::ResolveSpacing(Eigen::Vector3d(6, 6, 0), 1, tree, {4});
    EXPECT_EQ(scope.container, 1) << under_unshared;
    EXPECT_EQ(scope.mode, SpacingScopeMode::kOutside) << under_unshared;
    EXPECT_EQ(scope.referents, (std::vector<element_id_t>{2, 4})) << under_unshared << ": the room's objects and the person";
  }
}

// SP-D7-05, corner 2: an unshared FLAGGED element has no counterpart, so no receiver can map it: the flag acts on nothing. It is an
// ordinary unshared container, exactly the tree with the flag cleared (a region that holds a shared object keeps it and its border).
TEST(ScopeCases, SP_D7_05_AnUnsharedFlaggedLeafHoldingASharedObjectIsAnOrdinaryContainer) {
  // SP-D7-05
  const auto build = [](ElementTree* tree, bool flagged) {
    ASSERT_TRUE(tree->AddElement(1).ok());
    tree->GetElement(1)->SetSharedId(1);
    tree->GetElement(1)->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 30, 30)));
    Box(tree, 2, 1, 2, 2, 12, 12, GeometryRole::kContainment);  // F: unshared (flagged or not)
    tree->GetElement(2)->SetSharedId(kInvalidSharedElementId);
    tree->GetElement(2)->SetContainmentTransfer(flagged);
    Box(tree, 3, 2, 4, 4, 6, 6, GeometryRole::kObstacle);       // X: shared, in F
  };
  ElementTree flagged, plain;
  build(&flagged, true);
  build(&plain, false);
  for (const auto& pose : {std::pair<double, double>{9, 9}, {5, 5}, {20, 20}}) {
    const Eigen::Vector3d at(pose.first, pose.second, 0);
    ExpectSameScope(ScopeResolver::ResolveSpacing(at, 1, flagged), ScopeResolver::ResolveSpacing(at, 1, plain),
                    "pose " + std::to_string(pose.first));
  }
  const auto in_f = ScopeResolver::ResolveSpacing(Eigen::Vector3d(9, 9, 0), 1, flagged);
  EXPECT_EQ(in_f.referents, (std::vector<element_id_t>{3, 2})) << "F holds a shared object: it and the border";
  EXPECT_EQ(in_f.mode, SpacingScopeMode::kInterior);
  EXPECT_EQ(in_f.transfer_leaf, kInvalidElementId);
  EXPECT_EQ(ScopeResolver::ResolveSpacing(Eigen::Vector3d(5, 5, 0), 1, flagged).referents, (std::vector<element_id_t>{3}));
  // The shared control: a flagged SHARED leaf is still the leaf (D5).
  ElementTree shared;
  build(&shared, true);
  shared.GetElement(2)->SetSharedId(2);
  const auto leaf = ScopeResolver::ResolveSpacing(Eigen::Vector3d(9, 9, 0), 1, shared);
  EXPECT_EQ(leaf.referents, (std::vector<element_id_t>{2}));
  EXPECT_EQ(leaf.transfer_leaf, 2);
}

// SP-D7-05, corner 3: where footprints overlap, the V1 overlap choice must not hand the pose to an unshared container that cannot be a
// scope when a shared holder is also there. The tree twice: with the unshared E (and what it nests), and with E deleted.
TEST(ScopeCases, SP_D7_05_AnEmptyContainerNeverBeatsASharedOverlappingSibling) {
  // SP-D7-05
  const auto build = [](ElementTree* tree, bool with_e) {
    ASSERT_TRUE(tree->AddElement(1).ok());
    tree->GetElement(1)->SetSharedId(1);
    tree->GetElement(1)->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 40, 40)));
    Box(tree, 2, 1, 15, 15, 30, 30, GeometryRole::kObstacle);  // S: shared, overlaps E
    if (!with_e) return;
    Box(tree, 3, 1, 20, 20, 24, 24, GeometryRole::kContainment);  // E: unshared, centred on the pose
    tree->GetElement(3)->SetSharedId(kInvalidSharedElementId);
    Box(tree, 4, 3, 20.2, 20.2, 21, 21, GeometryRole::kContainment);  // E2: unshared, away from the pose
    tree->GetElement(4)->SetSharedId(kInvalidSharedElementId);
    Box(tree, 5, 4, 20.3, 20.3, 20.8, 20.8, GeometryRole::kObstacle);  // Z: shared, nested below, not holding the pose
  };
  ElementTree with, without;
  build(&with, true);
  build(&without, false);
  const Eigen::Vector3d at(22, 22, 0);
  const auto a = ScopeResolver::ResolveSpacing(at, 1, with), b = ScopeResolver::ResolveSpacing(at, 1, without);
  ExpectSameScope(a, b, "overlap");
  EXPECT_EQ(a.container, 2);
  EXPECT_EQ(a.referents, (std::vector<element_id_t>{2}));
}

// No shared ancestor at all: nothing is shared anywhere, and the scope STAYS EMPTY (the receiver's empty-scope mapping, SP-D7-02).
TEST(ScopeCases, SP_D7_05_WithNothingSharedAnywhereTheScopeStaysEmpty) {
  // SP-D7-05
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());  // the room, UNSHARED
  tree.GetElement(1)->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 30, 30)));
  Box(&tree, 2, 1, 2, 2, 10, 10, GeometryRole::kContainment);  // U1 (unshared) > U2 (unshared)
  tree.GetElement(2)->SetSharedId(kInvalidSharedElementId);
  Box(&tree, 3, 2, 4, 4, 6, 6, GeometryRole::kObstacle);
  tree.GetElement(3)->SetSharedId(kInvalidSharedElementId);
  EXPECT_TRUE(ScopeResolver::ResolveSpacing(Eigen::Vector3d(5, 5, 0), 1, tree).referents.empty());
  EXPECT_TRUE(ScopeResolver::ResolveSpacing(Eigen::Vector3d(8, 8, 0), 1, tree).referents.empty());
  EXPECT_TRUE(ScopeResolver::ResolveReachable(Eigen::Vector3d(0, 0, 0), kInvalidElementId, tree, 3).empty());
  EXPECT_TRUE(ScopeResolver::ResolveReachable(Eigen::Vector3d(0, 0, 0), kInvalidElementId, tree, 2).empty());
}

TEST(ScopeCases, FA_D7_01_AnUnsharedElementIsNeverAFacingTerm) {
  World w;
  AddUnshared(&w);
  w.Go(1, 26);
  EXPECT_EQ(w.Facing().count(50), 0u);
  EXPECT_GE(w.Facing().size(), 8u) << "every shared element is still a Facing referent";
  EXPECT_EQ(w.Facing().count(kT1), 1u);
  World shared;
  Box(shared.Tree(), 50, kRoom, 20, 12, 22, 14, GeometryRole::kObstacle);
  shared.Go(1, 26);
  EXPECT_EQ(shared.Facing().count(50), 1u);
}

TEST(ScopeCases, SP_D8_01_ReachabilityIsAGeodesicPathInTheSourceFreeSpace) {
  // Euclidean-near but sealed (8 m away in a straight line, no path): out. A far referent with a
  // path: in. No snap tolerance either way.
  LWorld sealed(true), open(false);
  EXPECT_EQ(sealed.Spacing().count(2), 0u);
  EXPECT_EQ(open.Spacing().count(2), 1u);
  auto room = std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {10, 0}, {10, 6}, {6, 6}, {6, 10}, {0, 10}}));
  EXPECT_TRUE(IsReferentReachable(*room, Eigen::Vector3d(2, 2, 0), Eigen::Vector3d(6.0, 8, 0)));
  EXPECT_FALSE(IsReferentReachable(*room, Eigen::Vector3d(2, 2, 0), Eigen::Vector3d(6.1, 8, 0)));
}

TEST(ScopeCases, SP_PEOPLE_01_EveryPersonInTheSameTreeIsAReferentAtAnyDepth) {
  World w;
  w.Person(30, kR3a, 3.5, 20);  // a person two levels down
  w.Go(1, 26);
  EXPECT_EQ(w.Spacing().count(30), 1u);
  EXPECT_EQ(w.Spacing().count(kSelf), 0u) << "the subject itself is never a referent";
  // A person in another tree is not.
  ASSERT_TRUE(w.Tree()->AddElement(40).ok());
  ASSERT_TRUE(w.Tree()->AddElement(41, 40).ok());
  w.Tree()->GetElement(41)->SetLocalTransform(At(5, 25));
  const auto other = w.space->CreateActor({SubjectAnchor("root", {0, 0, 0})});
  FollowProposals(w.space, other);
  w.space->GetActor(other)->SetElementId(41);
  w.Go(1, 26);
  EXPECT_EQ(w.Spacing().count(41), 0u);
}

// ---------------------------------------------------------------------------------------------
// Spacing: inside a referent
// ---------------------------------------------------------------------------------------------

TEST(ScopeCases, SP_INSIDE_01_InsideARegionTheSiblingsDropOut) {
  World w;
  w.Go(5, 4);  // inside R1, outside its objects
  const Ids s = w.Spacing();
  for (element_id_t sib : {kR2, kL1, kT1, kC1, kA, kB, kR3, kR4})
    EXPECT_EQ(s.count(sib), 0u) << "sibling " << sib << " is out of scope";
}

TEST(ScopeCases, SP_INSIDE_02_AReferentWithInteriorEntitiesIsItsOwnInteriorPlusBorder) {
  World w;
  w.Go(5, 4);
  EXPECT_EQ(w.Spacing(), (Ids{kO1, kO2, kR1})) << "the objects inside it and the border relation";
  EXPECT_EQ(w.Container(), kR1) << "the entered referent is the container";
  EXPECT_EQ(ModeOf(w), "INTERIOR");
}

TEST(ScopeCases, SP_INSIDE_03_ATransferLeafIsReplayedAtTheMappedPosition) {
  World w;
  w.Go(17.5, 6);
  EXPECT_EQ(w.Spacing(), (Ids{kL1}));
  EXPECT_EQ(ModeOf(w), "REPLAY");
}

TEST(ScopeCases, SP_INSIDE_04_AnEmptyObjectIsReplayedAtTheMappedPosition) {
  World w;
  w.Go(12, 5);  // R2: no interior entities, no leaf flag
  EXPECT_EQ(w.Spacing(), (Ids{kR2}));
  EXPECT_EQ(w.Container(), kR2);
  EXPECT_EQ(ModeOf(w), "REPLAY");
}

TEST(ScopeCases, SP_INSIDE_10_TheScopeOfAnEmptyObjectDoesNotDependOnTheReceiversShape) {
  // The scope is decided on the source side, from the source's own geometry: the same scope and mode
  // as SP-INSIDE-04, whatever the receiver's footprint is. The receiver half (the replay pose
  // through the chart of two non-congruent footprints) is the scenario level.
  World w;
  w.Go(10.4, 3.1);  // R2, off its centre line
  EXPECT_EQ(w.Spacing(), (Ids{kR2}));
  EXPECT_EQ(ModeOf(w), "REPLAY");
}

TEST(ScopeCases, SP_INSIDE_11_AnUprightPolygonHasNoGroundFootprintAndIsNotEntered) {
  World w;
  auto* t = w.Tree();
  Box(t, 30, kRoom, 20, 20, 24, 24, GeometryRole::kObstacle);  // a table
  // Its tabletop plane as the scene-geometry adapter registers it: an element-local Surface polygon
  // under Rx(-90) (Unity) taken into the Z-up frame, in float: the polygon stands upright.
  ASSERT_TRUE(t->AddElement(31, 30).ok());
  t->GetElement(31)->SetSharedId(31);
  Eigen::Matrix4d upright = Eigen::Matrix4d::Identity();
  const double c = static_cast<float>(std::cos(-M_PI / 2));
  upright(1, 1) = c; upright(1, 2) = 1.0;
  upright(2, 1) = -1.0; upright(2, 2) = c;
  t->GetElement(31)->SetLocalTransform(upright);
  auto plane = std::make_shared<NavigableSpatialDefinition>(Rect(-2, -2, 2, 2), nullptr, 0.0,
                                                           GeometryRole::kSurface);
  plane->SetElementLocal(true);
  t->GetElement(31)->SetSpatialDefinition(plane);
  w.Go(22.7, 21.3);  // inside the table, 1.3 m from the plane's strip
  EXPECT_EQ(w.Container(), 30) << "inside the table; the point is not inside an upright polygon";
  EXPECT_EQ(ModeOf(w), "INTERIOR");
  EXPECT_EQ(w.Spacing(), (Ids{31, 30}));
}

TEST(ScopeCases, SP_D8_02_AnElementsOwnFootprintNeverBlocksContactWithIt) {
  World w;
  auto* t = w.Tree();
  // A chair: an Obstacle footprint (+-0.25) whose seat part (+-0.2) lies INSIDE that footprint.
  Box(t, 30, kRoom, 24.75, 24.75, 25.25, 25.25, GeometryRole::kObstacle);
  auto def = std::make_shared<PolytopeSpatialDefinition>(Eigen::Vector3d(-0.25, -0.25, 0.0),
                                                         Eigen::Vector3d(0.25, 0.25, 0.45),
                                                         GeometryRole::kObstacle);
  SubRegion seat;
  seat.name = "seat";
  seat.tag = "seat";
  seat.local_transform = At(0, 0);
  seat.local_transform(2, 3) = 0.45;
  seat.boundary = Rect(-0.2, -0.2, 0.2, 0.2);
  def->AddSubRegion(seat);
  t->GetElement(30)->SetSpatialDefinition(def);
  w.Go(22, 25);  // outside it, in the open
  EXPECT_EQ(w.Spacing().count(30), 1u) << "the seat is reached by entering its chair's footprint";
}

TEST(ScopeCases, SP_INSIDE_12_AnUprightPolygonIsNeitherInsideNorAnInteriorReferent) {
  // The same upright plane however the app's matrix was rounded: exactly upright (a zero ground area), with
  // the noise of a double build of Rx(-90) (cos = 6.1e-17), of Unity's single-precision matrix (-4.4e-8) and
  // in between. Its ground image is a SEGMENT in every case, decided on the polygon's image in world
  // coordinates by the one tolerance of the data's precision (support.h GroundImageIsSegment), so the
  // referent is the same whatever the rounding: a point is not inside it,
  // not even ON the strip the plane stands on, and its Spacing support is that strip, the contact the
  // nearest point of the segment (for an exactly upright plane and a noisy one alike).
  const double kNoise[] = {0.0, static_cast<double>(static_cast<float>(std::cos(-M_PI / 2))), 1e-12,
                           static_cast<double>(static_cast<float>(-4.371139e-08)), 9e-7};
  std::vector<Eigen::Vector3d> contact_of_first;  // the contacts at c = 0, to compare the others against
  for (const double c : kNoise) {
    World w;
    auto* t = w.Tree();
    Box(t, 30, kRoom, 20, 20, 24, 24, GeometryRole::kObstacle);
    ASSERT_TRUE(t->AddElement(31, 30).ok());
    t->GetElement(31)->SetSharedId(31);
    Eigen::Matrix4d upright = Eigen::Matrix4d::Identity();
    upright(1, 1) = c; upright(1, 2) = 1.0;
    upright(2, 1) = -1.0; upright(2, 2) = c;
    t->GetElement(31)->SetLocalTransform(upright);
    auto plane = std::make_shared<NavigableSpatialDefinition>(Rect(-2, -2, 2, 2), nullptr, 0.0,
                                                             GeometryRole::kSurface);
    plane->SetElementLocal(true);
    t->GetElement(31)->SetSpatialDefinition(plane);
    // The strip is y = 22 (the plane's translation) from x = 20 to 24: a point ON it, and points off it.
    EXPECT_FALSE(FootprintHolds(*t->GetElement(31), *t, Eigen::Vector3d(22.0, 22.0, 0.0))) << "noise " << c;
    EXPECT_FALSE(FootprintHolds(*t->GetElement(31), *t, Eigen::Vector3d(22.7, 22.0, 0.0))) << "noise " << c;
    size_t at = 0;
    for (const double y : {21.3, 22.0, 22.5, 22.8}) {
      w.Go(22.7, y);
      EXPECT_EQ(w.Container(), 30) << "noise " << c << " y " << y << ": inside the table, never in its plane";
      bool seen = false;
      for (const auto& m : w.space->GetBindings(w.actor)[0].latest_matches) {
        if (m.element_id != 31) continue;
        seen = true;
        EXPECT_FALSE(m.eval.support.has_interior) << "noise " << c << " y " << y;
        EXPECT_GT(m.eval.grounded.support_distance_m, 0.0) << "noise " << c << " y " << y;
        EXPECT_TRUE(m.eval.support.has_contact) << "noise " << c << " y " << y << ": the strip is the support";
        // The contact is the nearest point of the strip (22.7 inside its x range [20, 24], y = 22), to the
        // precision of the data, and the same for every rounding.
        EXPECT_NEAR(m.eval.support.contact_point_world.x(), 22.7, 1e-4) << "noise " << c << " y " << y;
        EXPECT_NEAR(m.eval.support.contact_point_world.y(), 22.0, 1e-4) << "noise " << c << " y " << y;
        if (c == 0.0) {
          contact_of_first.push_back(m.eval.support.contact_point_world);
        } else if (at < contact_of_first.size()) {
          EXPECT_LT((m.eval.support.contact_point_world - contact_of_first[at]).norm(), 1e-4)
              << "noise " << c << " y " << y << ": the same referent as the exactly upright plane";
        }
      }
      EXPECT_TRUE(seen) << "noise " << c << " y " << y << ": the plane stays an object of its table";
      ++at;
    }
  }
}

TEST(ScopeCases, SP_INSIDE_05_InsideAnObjectFootprintTheSiblingsDropOutToo) {
  World w;
  w.Go(3, 13);  // inside the table T1 (an obstacle footprint, not a region)
  EXPECT_EQ(w.Spacing(), (Ids{kT1})) << "the table, replayed; every sibling is out of scope";
  EXPECT_EQ(w.Container(), kT1);
  EXPECT_EQ(ModeOf(w), "REPLAY");
  w.Go(6.5, 13);  // inside the chair C1
  EXPECT_EQ(w.Spacing(), (Ids{kC1}));
  EXPECT_EQ(ModeOf(w), "REPLAY");
}

// Sibling footprints may only touch (there is no v1-style choice of one referent among
// overlapping non-region footprints): registration refuses an overlap (see
// SP_OVERLAP_01), so a pose is in at most one sibling footprint. A and B touch along x = 13.
TEST(ScopeCases, SP_INSIDE_06_TouchingFootprintsAreEnteredOneAtATime) {
  World w;
  w.Go(12.2, 13.4);  // in A only
  EXPECT_EQ(w.Spacing(), (Ids{kA}));
  EXPECT_EQ(w.Container(), kA);
  EXPECT_EQ(ModeOf(w), "REPLAY");
  w.Go(13.8, 13.4);  // in B only
  EXPECT_EQ(w.Spacing(), (Ids{kB}));
  EXPECT_EQ(w.Container(), kB);
  EXPECT_EQ(ModeOf(w), "REPLAY");
}

TEST(ScopeCases, SP_INSIDE_07_ContactWithoutEnteringKeepsEverySibling) {
  World w;
  w.Go(1.99, 13);  // 1 cm outside T1's edge
  EXPECT_EQ(w.Spacing(), (Ids{kR1, kR2, kL1, kT1, kC1, kA, kB, kR3, kR4}));
  EXPECT_EQ(ModeOf(w), "OUTSIDE");
}

TEST(ScopeCases, SP_INSIDE_08_PeopleOutsideTheEnteredReferentDropOutWithTheSiblings) {
  World w;
  w.Person(30, kRoom, 1, 26);  // outside every referent
  w.Go(3, 13);                 // inside the table
  EXPECT_EQ(w.Spacing().count(30), 0u);
  w.Go(1, 26);
  EXPECT_EQ(w.Spacing().count(30), 1u);
}

TEST(ScopeCases, SP_INSIDE_09_InInteriorModeThePeopleInsideItStayAndTheOthersDrop) {
  World w;
  w.Person(30, kRoom, 7.5, 3);  // inside R1
  w.Person(31, kRoom, 1, 26);   // outside
  w.Go(5, 4);
  EXPECT_EQ(w.Spacing(), (Ids{kO1, kO2, kR1, 30}));
  EXPECT_EQ(ModeOf(w), "INTERIOR");
}

TEST(ScopeCases, SP_EXIT_01_LeavingAReferentRestoresTheSiblings) {
  World w;
  w.Go(3, 13);
  EXPECT_EQ(w.Spacing(), (Ids{kT1}));
  w.Go(3, 14.5);  // one step out of T1
  EXPECT_EQ(w.Spacing(), (Ids{kR1, kR2, kL1, kT1, kC1, kA, kB, kR3, kR4}));
  EXPECT_EQ(ModeOf(w), "OUTSIDE");
  w.Go(5, 4);  // into R1, then out
  EXPECT_EQ(w.Spacing(), (Ids{kO1, kO2, kR1}));
  w.Go(9, 4);
  EXPECT_EQ(w.Spacing(), (Ids{kR1, kR2, kL1, kT1, kC1, kA, kB, kR3, kR4}));
}

TEST(ScopeCases, SP_OVERLAP_01_OverlappingSiblingRegionsAreRefused) {
  World w;
  Box(w.Tree(), 50, kRoom, 9, 1, 12, 5, GeometryRole::kContainment);  // overlaps R2 (10,2)-(14,8)
  const auto v = FindContainmentViolation(*w.Tree(), 50);
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->code, SceneErrorCode::kContainmentOverlap);
  // Non-region footprints (furniture) may not overlap either; touching edges are fine.
  EXPECT_FALSE(FindContainmentViolation(*w.Tree(), kA).has_value());
  EXPECT_FALSE(FindContainmentViolation(*w.Tree(), kB).has_value());
  // A touch that differs by one float step (1e-6 m at x = 23) is within the data's precision (a length
  // derived from the float epsilon, not a ratio of areas): accepted; a real overlap of 1 cm is refused.
  Box(w.Tree(), 52, kRoom, 20, 20, 23, 23, GeometryRole::kObstacle);
  Box(w.Tree(), 53, kRoom, 23.0 - 1e-6, 20, 26, 23, GeometryRole::kObstacle);
  EXPECT_FALSE(FindContainmentViolation(*w.Tree(), 53).has_value());
  Box(w.Tree(), 54, kRoom, 26.0 - 1e-2, 20, 28, 23, GeometryRole::kObstacle);
  EXPECT_TRUE(FindContainmentViolation(*w.Tree(), 54).has_value());
  Box(w.Tree(), 51, kRoom, 11, 12, 14, 15, GeometryRole::kObstacle);  // overlaps A and B
  const auto f = FindContainmentViolation(*w.Tree(), 51);
  ASSERT_TRUE(f.has_value());
  EXPECT_EQ(f->code, SceneErrorCode::kContainmentOverlap);
}

// ---------------------------------------------------------------------------------------------
// AffordancePose: proximal scope (D3)
// ---------------------------------------------------------------------------------------------

TEST(ScopeCases, SP_D3_01_AffordancePoseIsInScopeInsideTheParentObjectsRegionOnly) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  Box(&tree, 2, 1, 2, 2, 8, 8, GeometryRole::kContainment);            // the object's region
  Box(&tree, 3, 2, 4, 4, 5, 5, GeometryRole::kInteractionRegion, 0.5);  // its affordance
  Support s;
  EXPECT_EQ(FamilyScopeGateVerdict(RelationKernelType::kAvatarAffordancePose, tree, 3,
                                   Eigen::Vector3d(7, 7, 0), s, nullptr),
            ScopeGateVerdict::kInScope);
  EXPECT_EQ(FamilyScopeGateVerdict(RelationKernelType::kAvatarAffordancePose, tree, 3,
                                   Eigen::Vector3d(9, 9, 0), s, nullptr),
            ScopeGateVerdict::kOutsideParentRegion);
}

// ---------------------------------------------------------------------------------------------
// Facing
// ---------------------------------------------------------------------------------------------

TEST(ScopeCases, FA_GLOBAL_01_FacingScopeIsEveryVisibleElementAtAnyDepthAndDistance) {
  World w;
  w.Go(1, 26);
  const Ids f = w.Facing();
  for (element_id_t id : {kR1, kO1, kO2, kAff1, kR2, kL1, kOL, kT1, kC1, kA, kB, kR3, kR3a, kR3aO, kR4, kR4I})
    EXPECT_EQ(f.count(id), 1u) << "element " << id;
  EXPECT_EQ(f.count(kSelf), 0u) << "the subject's own element";
  EXPECT_EQ(f.count(kRoom), 0u) << "the container the subject is in is not faced";
}

TEST(ScopeCases, FA_GLOBAL_02_InsideAReferentFacingIsNotNarrowed) {
  World w;
  w.Go(1, 26);
  const Ids outside = w.Facing();
  w.Go(5, 4);  // inside R1
  EXPECT_EQ(w.Facing().count(kT1), 1u);
  EXPECT_EQ(w.Facing().count(kL1), 1u);
  w.Go(17.5, 6);  // inside the transfer leaf
  EXPECT_EQ(w.Facing().count(kT1), 1u);
  w.Go(3, 13);  // inside the table: its own footprint does not occlude the eye (it holds it)
  EXPECT_GT(w.Facing().size(), 8u);
  (void)outside;
}

// D1: ONE ray from the eye to the look-at contact, on the source side. A full-height wall between
// the eye and the contact occludes; a low obstacle does not.
TEST(ScopeCases, FA_OCC_01_AWallOnTheRayToTheLookAtContactOccludes) {
  World w;
  Box(w.Tree(), 60, kRoom, 20, 20.0, 24, 20.2, GeometryRole::kObstacle, 2.5);  // a wall
  Box(w.Tree(), 61, kRoom, 21, 24, 23, 26, GeometryRole::kObstacle, 1.0);       // a cabinet behind it
  w.Go(22, 17);  // eye south of the wall, the cabinet north of it
  EXPECT_EQ(w.Facing().count(61), 0u) << "occluded: out of the source's scope";
  EXPECT_EQ(w.Facing().count(60), 1u) << "the wall itself is seen";
  w.Go(22, 28);  // north of the cabinet: nothing between
  EXPECT_EQ(w.Facing().count(61), 1u);
}

TEST(ScopeCases, FA_OCC_02_ALowObstacleDoesNotOcclude) {
  World w;
  Box(w.Tree(), 60, kRoom, 20, 20.0, 24, 20.2, GeometryRole::kObstacle, 0.3);  // below eye height
  Box(w.Tree(), 61, kRoom, 21, 24, 23, 26, GeometryRole::kObstacle, 1.0);
  w.Go(22, 17);
  EXPECT_EQ(w.Facing().count(61), 1u);
}

TEST(ScopeCases, FA_BODY_01_ALookAtIsInBodyOnlyWhenItCoincidesWithTheBody) {
  // D2 (v1: delta < kEpsilon). The scope layer owns the predicate; the solver reads it.
  EXPECT_TRUE(LookAtInBody(0.0));
  EXPECT_TRUE(LookAtInBody(5e-10));
  EXPECT_FALSE(LookAtInBody(2e-9));
  EXPECT_FALSE(LookAtInBody(1e-3)) << "a millimetre away is not in the body";
  EXPECT_FALSE(LookAtInBody(0.25)) << "the body radius is not a scope";
}

TEST(ScopeCases, FA_PEER_01_APeerActorIsAFacingReferent) {
  World w;
  w.Person(30, kRoom, 20, 25);
  w.Go(1, 26);
  EXPECT_EQ(w.Facing().count(30), 1u) << "Facing toward the collaborator works";
  EXPECT_EQ(w.Facing().count(kSelf), 0u);
}

TEST(ScopeCases, FA_NOTAG_01_TagsNarrowNeitherFacingNorSpacing) {
  World w;
  w.Tree()->GetElement(kT1)->SetTag("Table");
  w.Tree()->GetElement(kC1)->SetTag("Chair");
  const auto actor2 = w.space->CreateActor({SubjectAnchor("root", {0, 0, 0}), SubjectAnchor("chest", {0, 0, 1.2})});
  FollowProposals(w.space, actor2);
  w.Tree()->AddElement(31, kRoom);
  w.space->GetActor(actor2)->SetElementId(31);
  w.space->Bind(actor2, "root", RelationKernelType::kAvatarSpacing, "Chair");
  w.space->Bind(actor2, "chest", RelationKernelType::kAvatarFacing, "Chair");
  w.space->UpdateActorState(actor2, Eigen::Vector2d(1, 26), 0.0);
  Ids sp, fa;
  for (const auto& m : w.space->GetBindings(actor2)[0].latest_matches) sp.insert(m.element_id);
  for (const auto& m : w.space->GetBindings(actor2)[1].latest_matches) fa.insert(m.element_id);
  EXPECT_EQ(sp.count(kT1), 1u);
  EXPECT_EQ(fa.count(kT1), 1u);
  EXPECT_TRUE(w.space->GetBindings(actor2)[0].tag_filter.empty());
  EXPECT_TRUE(w.space->GetBindings(actor2)[1].tag_filter.empty());
}

TEST(ScopeCases, FA_LEAFYAW_01_InsideATransferLeafWithNoFacingTheYawIsLeafRelative) {
  // Facing is global, so inside the leaf its scope is not narrowed (FA-GLOBAL-02); the yaw
  // source is a function of whether any Facing is valid and whether the source is in a leaf.
  EXPECT_EQ(ResolveRootYawSource(/*a_facing_is_valid=*/true, /*in_leaf=*/true), RootYawSource::kFacing)
      << "an authored Facing owns the yaw inside the leaf";
  EXPECT_EQ(ResolveRootYawSource(false, true), RootYawSource::kLeafRelative)
      << "never unspecified while the source is inside a leaf the receiver maps";
  // The leaf the source is in is the scope layer's own fact.
  World w;
  w.Go(17.5, 6);
  const auto& s = w.space->GetBindings(w.actor)[0].spacing_scope;
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ(s->transfer_leaf, kL1);
  // ... and with a Facing valid there (the world has referents in sight) the Facing owns the yaw.
  EXPECT_FALSE(w.Facing().empty());
  EXPECT_EQ(ResolveRootYawSource(!w.Facing().empty(), s->transfer_leaf != kInvalidElementId),
            RootYawSource::kFacing);
  EXPECT_EQ(ResolveRootYawSource(false, s->transfer_leaf != kInvalidElementId),
            RootYawSource::kLeafRelative);
}

TEST(ScopeCases, FA_YAW_01_NoValidFacingLeavesTheYawUnspecified) {
  EXPECT_EQ(ResolveRootYawSource(false, false), RootYawSource::kUnspecified);
  // A world whose only other element is the subject's own: the Facing scope is empty, so no
  // Facing is valid and (outside any leaf) the yaw source is UNSPECIFIED.
  auto ctx = LocalContext::Create();
  Space* space = ctx->AddSpace();
  auto* t = space->GetElementTree();
  ASSERT_TRUE(t->AddElement(1).ok());
  t->GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 30, 30)));
  ASSERT_TRUE(t->AddElement(20, 1).ok());
  const auto actor = space->CreateActor({SubjectAnchor("chest", {0, 0, 1.2})});
  FollowProposals(space, actor);
  space->GetActor(actor)->SetElementId(20);
  space->Bind(actor, "chest", RelationKernelType::kAvatarFacing);
  space->UpdateActorState(actor, Eigen::Vector2d(15, 5), 0.0);
  EXPECT_TRUE(space->GetBindings(actor)[0].latest_matches.empty());
  EXPECT_EQ(ResolveRootYawSource(!space->GetBindings(actor)[0].latest_matches.empty(), false),
            RootYawSource::kUnspecified);
}

// ---------------------------------------------------------------------------------------------
// AffordancePose, one level down: the object's affordance poses
// ---------------------------------------------------------------------------------------------

// A chair at (5, 5) with a 0.45 m seat part, and its `stand` (in front) and `seated` pelvis poses on interaction-role
// children; one actor whose pelvis is bound to the chair. `standing` < 0: the embodiment registers no standing height.
struct ChairWorld {
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space* space = ctx->AddSpace();
  spatialblend::entity_id_t actor = 0;
  static constexpr element_id_t kChair = 30, kStandPose = 31, kSeatedPose = 32, kChairSelf = 33;
  explicit ChairWorld(double standing) {
    auto* t = space->GetElementTree();
    EXPECT_TRUE(t->AddElement(kRoom).ok());
    t->GetElement(kRoom)->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 10, 10)));
    EXPECT_TRUE(t->AddElement(kChair, kRoom).ok());
    t->GetElement(kChair)->SetSharedId(kChair);
    t->GetElement(kChair)->SetTag("Sittable");
    t->GetElement(kChair)->SetLocalTransform(At(5, 5));
    auto def = std::make_shared<PolytopeSpatialDefinition>(Eigen::Vector3d(-0.25, -0.22, 0),
                                                           Eigen::Vector3d(0.25, 0.22, 0.45), GeometryRole::kObstacle);
    SubRegion top;
    top.name = "seat";
    top.local_transform = Eigen::Matrix4d::Identity();
    top.local_transform(2, 3) = 0.45;
    top.boundary = Rect(-0.25, -0.22, 0.25, 0.22);
    def->AddSubRegion(top);
    t->GetElement(kChair)->SetSpatialDefinition(def);
    const auto pose = [&](element_id_t id, const char* tag, double y0, double y1, AffordancePoseTemplate tpl) {
      EXPECT_TRUE(t->AddElement(id, kChair).ok());
      t->GetElement(id)->SetTag(tag);
      t->GetElement(id)->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-0.25, y0, 0), Eigen::Vector3d(0.25, y1, 0.1), GeometryRole::kInteractionRegion));
      t->GetElement(id)->SetAffordancePose(tpl);
    };
    pose(kStandPose, "stand", -0.6, -0.22, {"pelvis", AffordancePoseTemplate::Support::kActorRoot, ""});
    pose(kSeatedPose, "seated", -0.22, 0.22, {"pelvis", AffordancePoseTemplate::Support::kPart, "seat"});
    EXPECT_TRUE(t->AddElement(kChairSelf, kRoom).ok());
    actor = space->CreateActor({SubjectAnchor("root", {0, 0, 0}), SubjectAnchor("pelvis", {0, 0, 0.9})});
    space->GetActor(actor)->SetElementId(kChairSelf);
    Embodiment e;
    e.id = "fixture";
    e.joints = {{"hips", -1}};
    EmbodimentAnchor pelvis;
    pelvis.name = "pelvis";
    pelvis.families = kFamilyAffordancePose;
    if (standing > 0) {
      pelvis.has_standing = true;
      pelvis.standing_height_m = standing;
      pelvis.standing_provenance_json = R"({"procedure":"fixture","source":"scope_cases_test"})";
    }
    e.anchors = {pelvis};
    EXPECT_EQ(space->RegisterEmbodiment(actor, e), Space::RegisterEmbodimentResult::kOk);
    EXPECT_NE(space->BindScoped(actor, "pelvis", RelationKernelType::kAvatarAffordancePose, kChair, "Sittable"),
              nullptr);
  }
  // The pelvis at `pelvis`, the root under it on the floor.
  void Go(const Eigen::Vector3d& pelvis) {
    space->SetAnchorWorldPose(actor, "pelvis", pelvis, 0.0);
    space->UpdateActorState(actor, pelvis.head<2>(), 0.0);
  }
  // element id -> (pose tag, support distance) of the pelvis binding's matches.
  std::map<element_id_t, std::pair<std::string, double>> Matches() const {
    std::map<element_id_t, std::pair<std::string, double>> out;
    for (const auto& m : space->GetBindings(actor)[0].latest_matches)
      out[m.element_id] = {m.pose_tag, m.eval.grounded.support_distance_m};
    return out;
  }
};

TEST(ScopeCases, SP_AFF_01_TheObjectsPosesAreItsInteractionChildrenInstantiatedForEachEmbodiment) {
  ChairWorld w(0.9);
  const ElementTree& t = *w.space->GetElementTree();
  Embodiment a, b;
  a.id = b.id = "x";
  a.joints = b.joints = {{"hips", -1}};
  EmbodimentAnchor pa;
  pa.name = "pelvis";
  pa.families = kFamilyAffordancePose;
  pa.has_standing = true;
  pa.standing_provenance_json = R"({"procedure":"fixture","source":"scope_cases_test"})";
  EmbodimentAnchor pb = pa;
  pa.standing_height_m = 0.88;
  pb.standing_height_m = 0.96;
  a.anchors = {pa};
  b.anchors = {pb};
  const Eigen::Vector3d root(5.0, 4.6, 0.0);
  const auto sa = InstantiatePoseSD(t, ChairWorld::kChair, "pelvis", &a, root);
  const auto sb = InstantiatePoseSD(t, ChairWorld::kChair, "pelvis", &b, root);
  ASSERT_TRUE(sa.ok());
  ASSERT_TRUE(sb.ok());
  EXPECT_EQ(sa->Find("stand")->support.point.z(), 0.88);
  EXPECT_EQ(sb->Find("stand")->support.point.z(), 0.96);
  EXPECT_EQ(sa->Find("seated")->element, ChairWorld::kSeatedPose);
  EXPECT_EQ(sa->Find("stand")->element, ChairWorld::kStandPose);
}

TEST(ScopeCases, SP_AFF_02_ThePosesAreInScopeExactlyWhileTheAnchorIsInsideTheObjectsSD) {
  ChairWorld w(0.9);
  w.Go({5.0, 3.0, 0.9});  // far in front of the chair: outside V
  for (const auto& [id, m] : w.Matches()) EXPECT_TRUE(m.first.empty()) << id;
  w.Go({5.0, 4.6, 0.9});  // standing in the stand region, pelvis at the standing height: on V's top face, inside
  auto m = w.Matches();
  ASSERT_EQ(m.count(ChairWorld::kStandPose), 1u);
  ASSERT_EQ(m.count(ChairWorld::kSeatedPose), 1u);
  EXPECT_EQ(m[ChairWorld::kStandPose].first, "stand");
  EXPECT_EQ(m[ChairWorld::kStandPose].second, 0.0);  // the standing pelvis over the root IS the stand pose
  EXPECT_NEAR(m[ChairWorld::kSeatedPose].second, std::hypot(4.78 - 4.6, 0.9 - 0.45), 1e-12);  // nearest seat point
  w.Go({5.0, 4.6, 0.91});  // 1 cm above the highest pose: outside V
  for (const auto& [id, mm] : w.Matches()) EXPECT_TRUE(mm.first.empty()) << id;
}

TEST(ScopeCases, SP_AFF_03_APoseElementIsNoReferentAndDoesNotMakeItsObjectAContainer) {
  ChairWorld w(0.9);
  w.Go({5.0, 5.0, 0.7});  // over the seat, inside the chair's footprint
  const auto m = w.Matches();
  ASSERT_EQ(m.count(ChairWorld::kChair), 1u) << "the chair stays the pelvis's contact referent";
  EXPECT_TRUE(m.at(ChairWorld::kChair).first.empty());
  // The pose children appear only as the chair's inner relations, never as contact referents of their own.
  EXPECT_EQ(m.at(ChairWorld::kStandPose).first, "stand");
  EXPECT_EQ(m.at(ChairWorld::kSeatedPose).first, "seated");
  EXPECT_NEAR(m.at(ChairWorld::kSeatedPose).second, 0.7 - 0.45, 1e-12);
}

TEST(ScopeCases, SP_AFF_04_AnEmbodimentWithoutAStandingHeightHasNoInnerRelations) {
  ChairWorld w(-1.0);
  w.Go({5.0, 5.0, 0.7});
  const auto m = w.Matches();
  EXPECT_EQ(m.count(ChairWorld::kChair), 1u);  // the contact relation is unchanged
  for (const auto& [id, mm] : m) EXPECT_TRUE(mm.first.empty()) << "no default standing height: " << id;
}

// ---------------------------------------------------------------------------------------------
// The catalog and the tests agree
// ---------------------------------------------------------------------------------------------

std::string ReadFile(const std::string& path) {
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::set<std::string> RegisteredCaseIds() {
  std::set<std::string> ids;
  const auto* unit = ::testing::UnitTest::GetInstance();
  for (int i = 0; i < unit->total_test_suite_count(); ++i) {
    const auto* suite = unit->GetTestSuite(i);
    if (std::string(suite->name()) != "ScopeCases") continue;
    for (int j = 0; j < suite->total_test_count(); ++j) {
      std::string n = suite->GetTestInfo(j)->name();
      // SP_INSIDE_02_Description -> SP-INSIDE-02
      static const std::regex re("^((?:SP|FA)_[A-Z0-9]+_[0-9]+)_");
      std::smatch m;
      if (std::regex_search(n, m, re)) {
        std::string id = m[1];
        std::replace(id.begin(), id.end(), '_', '-');
        ids.insert(id);
      }
    }
  }
  return ids;
}

TEST(CatalogCoverage, EveryCaseHasATest) {
  const std::string md = ReadFile("spatialblend/space/testing/SCOPE_CASES.md");
  ASSERT_FALSE(md.empty()) << "SCOPE_CASES.md missing from the test's data";
  std::set<std::string> catalog;
  const std::regex row("\\n\\| ((?:SP|FA)-[A-Z0-9]+-[0-9]+) \\|");
  for (auto it = std::sregex_iterator(md.begin(), md.end(), row); it != std::sregex_iterator(); ++it)
    catalog.insert((*it)[1]);
  ASSERT_GT(catalog.size(), 20u);
  const std::set<std::string> tests = RegisteredCaseIds();
  for (const auto& id : catalog)
    EXPECT_EQ(tests.count(id), 1u) << "catalog case " << id << " has no ScopeCases test";
  for (const auto& id : tests)
    EXPECT_EQ(catalog.count(id), 1u) << "test " << id << " is not in SCOPE_CASES.md";
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
