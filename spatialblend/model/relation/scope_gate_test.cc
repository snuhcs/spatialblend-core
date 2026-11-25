// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Contract tests for the per-family geometric scope gates (Facing line of sight,
// Spacing reachability).
// Oracles are analytic: straight sightlines against axis-aligned boxes, and
// a walkable L-shape whose notch is walled off.

#include "spatialblend/model/relation/scope_gate.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {
namespace test {
namespace {

std::shared_ptr<Polygon2D> Rect(double x0, double y0, double x1, double y1) {
  return std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});
}

void AddFootprint(ElementTree* tree, element_id_t id, element_id_t parent,
                  std::shared_ptr<Polygon2D> poly, GeometryRole role) {
  ASSERT_TRUE(tree->AddElement(id, parent).ok());
  tree->GetElement(id)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(poly, nullptr, 0.0, role));
}

// Box polytope in the element's local frame, placed at (cx, cy).
void AddBox(ElementTree* tree, element_id_t id, element_id_t parent, double cx,
            double cy, double hx, double hy, double height,
            GeometryRole role) {
  ASSERT_TRUE(tree->AddElement(id, parent).ok());
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = cx;
  t(1, 3) = cy;
  tree->GetElement(id)->SetLocalTransform(t);
  tree->GetElement(id)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-hx, -hy, 0.0), Eigen::Vector3d(hx, hy, height),
          role));
}

// Room(1) [0,20]x[-10,10]; actor sits at (1, 0) at chest height 1.2.
class VisibilityGateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    AddFootprint(&tree_, 1, kInvalidElementId, Rect(0, -10, 20, 10),
                 GeometryRole::kNavigable);
    // Referent: a 1 m tall cabinet at (10, 0), 1x1 footprint.
    AddBox(&tree_, 2, 1, 10, 0, 0.5, 0.5, 1.0, GeometryRole::kObstacle);
  }
  ElementTree tree_;
  const Eigen::Vector3d eye_{1.0, 0.0, 1.2};
};

TEST_F(VisibilityGateTest, ScopeGating_E1_1_UnobstructedReferentVisible) {
  EXPECT_TRUE(IsReferentVisible(tree_, eye_, 2));
}

TEST_F(VisibilityGateTest, ScopeGating_E1_1_FullHeightWallOccludes) {
  // A 2.5 m wall volume spanning y in [-3, 3] at x in [4, 5]: every
  // sightline from (1,0) to the cabinet footprint (|y| <= 0.5 at x ~10)
  // crosses it below its top.
  AddBox(&tree_, 3, 1, 4.5, 0, 0.5, 3.0, 2.5, GeometryRole::kObstacle);
  EXPECT_FALSE(IsReferentVisible(tree_, eye_, 2));
}

// ONE ray, to the look-at contact: visibility is the single ray, so a referent whose
// contact is hidden is not visible. The wall covers y in [-3, 0.1]: the cabinet's origin
// (10, 0) is hidden, its +y corner contact (9.5, 0.5) is in view (the sightline passes
// x = 4.5 at y ~ 0.21 > 0.1).
TEST_F(VisibilityGateTest, ScopeGating_D1_OneRayToTheLookAtContact) {
  AddBox(&tree_, 3, 1, 4.5, -1.45, 0.5, 1.55, 2.5, GeometryRole::kObstacle);
  EXPECT_FALSE(IsReferentVisible(tree_, eye_, 2));  // no contact: the origin ray
  const Eigen::Vector3d visible_contact(9.5, 0.5, 1.0);
  const Eigen::Vector3d hidden_contact(9.5, -0.5, 1.0);
  EXPECT_TRUE(IsReferentVisible(tree_, eye_, 2, &visible_contact));
  EXPECT_FALSE(IsReferentVisible(tree_, eye_, 2, &hidden_contact));
}

TEST_F(VisibilityGateTest, ScopeGating_E1_1_LowObstacleDoesNotOcclude) {
  // A 0.5 m coffee table between: the sightline from 1.2 m to the 1.0 m top
  // is above 1.1 m where it crosses, so the table does not block.
  AddBox(&tree_, 3, 1, 5, 0, 0.5, 2.0, 0.5, GeometryRole::kObstacle);
  EXPECT_TRUE(IsReferentVisible(tree_, eye_, 2));
  // A 2 m bookshelf in the same place does.
  AddBox(&tree_, 4, 1, 5, 0, 0.5, 2.0, 2.0, GeometryRole::kObstacle);
  EXPECT_FALSE(IsReferentVisible(tree_, eye_, 2));
}

TEST_F(VisibilityGateTest, ScopeGating_E1_1_OccluderContainingEyeIgnored) {
  // The actor stands inside a tall booth footprint: its own enclosure does
  // not hide the room from it.
  AddBox(&tree_, 3, 1, 1, 0, 0.8, 0.8, 2.0, GeometryRole::kObstacle);
  EXPECT_TRUE(IsReferentVisible(tree_, eye_, 2));
}

TEST_F(VisibilityGateTest, ScopeGating_E1_1_OwnPartsAndParentsDoNotOcclude) {
  // A seat part (child of the cabinet) is not hidden by the cabinet it
  // belongs to.
  AddBox(&tree_, 5, 2, 0, 0, 0.3, 0.3, 0.5, GeometryRole::kInteractionRegion);
  EXPECT_TRUE(IsReferentVisible(tree_, eye_, 5));
}

TEST_F(VisibilityGateTest, ScopeGating_E1_1_FlatFootprintNeverOccludes) {
  // An Obstacle-role footprint without height (a rug-like Navigable
  // footprint) has kFootprintObstacleHeightM = 0 and cannot block sight.
  AddFootprint(&tree_, 3, 1, Rect(4, -3, 5, 3), GeometryRole::kObstacle);
  EXPECT_TRUE(IsReferentVisible(tree_, eye_, 2));
}

TEST_F(VisibilityGateTest, ScopeGating_E1_1_DoorGapBetweenWallVolumesIsSeeThrough) {
  // A wall at x in [4,5] with a 1.2 m doorway at y in [-0.6, 0.6], modelled
  // as two Obstacle volumes: the cabinet is seen through the doorway.
  AddBox(&tree_, 3, 1, 4.5, -4.8, 0.5, 4.2, 2.5, GeometryRole::kObstacle);
  AddBox(&tree_, 4, 1, 4.5, 4.8, 0.5, 4.2, 2.5, GeometryRole::kObstacle);
  EXPECT_TRUE(IsReferentVisible(tree_, eye_, 2));
}

TEST_F(VisibilityGateTest, ScopeGating_E1_1_WallRingBlocksItsDoorway_Limitation) {
  // Documented limitation: a declared wall polygon is a closed ring, so it
  // cannot express a doorway -- the same wall declared as one ring blocks
  // what the door gap above lets through.
  ASSERT_TRUE(tree_.AddElement(6, 1).ok());
  tree_.GetElement(6)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          Rect(0, -10, 20, 10), Rect(4, -9, 5, 9), 0.0,
          GeometryRole::kContainment));
  EXPECT_FALSE(IsReferentVisible(tree_, eye_, 2));
}

TEST_F(VisibilityGateTest, ScopeGating_E1_1_DeclaredWallPolygonOccludes) {
  // Walls declared on a navigable (NavigableSpatialDefinition wall polygon)
  // occlude like obstacles, whatever element carries them.
  ASSERT_TRUE(tree_.AddElement(6, 1).ok());
  tree_.GetElement(6)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          Rect(6, -8, 8, 8), Rect(7, -8, 7.2, 8), 0.0,
          GeometryRole::kContainment));
  EXPECT_FALSE(IsReferentVisible(tree_, eye_, 2));
}

TEST_F(VisibilityGateTest, ScopeGating_E1_1_FloorAnchorUsesSightHeight) {
  // A root anchor at z=0 is lifted to kDefaultSightlineHeightM, so the low
  // table (0.5 m) still does not hide the cabinet.
  AddBox(&tree_, 3, 1, 5, 0, 0.5, 2.0, 0.5, GeometryRole::kObstacle);
  EXPECT_TRUE(IsReferentVisible(tree_, Eigen::Vector3d(1, 0, 0), 2));
}

// Facing dispatch goes through the gate with the support's contact point.
TEST_F(VisibilityGateTest, ScopeGating_E1_1_FamilyDispatch) {
  AddBox(&tree_, 3, 1, 4.5, 0, 0.5, 3.0, 2.5, GeometryRole::kObstacle);
  Support support;
  support.world_transform = tree_.GetWorldTransform(2);
  EXPECT_FALSE(PassesFamilyScopeGate(RelationKernelType::kAvatarFacing, tree_,
                                     2, eye_, support, nullptr));
  // AffordancePose is never gated geometrically (D3: its scope is the parent
  // object's region, structural, with no distance gate).
  EXPECT_TRUE(PassesFamilyScopeGate(RelationKernelType::kAvatarAffordancePose,
                                    tree_, 2, eye_, support, nullptr));
  // A prebuilt scene gives the same answer.
  const OcclusionScene scene(tree_);
  EXPECT_FALSE(PassesFamilyScopeGate(RelationKernelType::kAvatarFacing, tree_,
                                     2, eye_, support, nullptr, &scene));
}

// Retired: the Facing scope radius (3.66 m / 12 ft) with its
// ReferentFootprintDistance. Scope is the
// family's, not an app radius: the cabinet 8.5 m from the eye is in scope.
TEST_F(VisibilityGateTest, ScopeGating_E1_1_NoDistanceCutoff) {
  Support support;
  support.world_transform = tree_.GetWorldTransform(2);
  EXPECT_TRUE(PassesFamilyScopeGate(RelationKernelType::kAvatarFacing, tree_,
                                    2, eye_, support, nullptr));
}

// L-shaped walkable [0,10]^2 minus the walled-off notch [6,10]x[6,10].
std::shared_ptr<NavigableSpatialDefinition> LRoom() {
  return std::make_shared<NavigableSpatialDefinition>(
      std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
          {0, 0}, {10, 0}, {10, 6}, {6, 6}, {6, 10}, {0, 10}}));
}

TEST(ReachabilityGateTest, ScopeGating_E1_2_WalkableTargetReachable) {
  auto room = LRoom();
  const Eigen::Vector3d from(2, 2, 0);
  EXPECT_TRUE(IsReferentReachable(*room, from, Eigen::Vector3d(3, 3, 0)));
  // Around the inner corner: both arms are one connected walkable area.
  EXPECT_TRUE(IsReferentReachable(*room, from, Eigen::Vector3d(9, 2, 0)));
  EXPECT_TRUE(IsReferentReachable(*room, from, Eigen::Vector3d(2, 9, 0)));
}

TEST(ReachabilityGateTest, ScopeGating_E1_2_SealedTargetUnreachable) {
  auto room = LRoom();
  const Eigen::Vector3d from(2, 2, 0);
  // (8, 8) is 2 m from the nearest walkable point: behind the notch walls.
  // It is only ~8.5 m away in a straight line -- a Euclidean judgement would
  // have kept it; the gate must not.
  EXPECT_FALSE(IsReferentReachable(*room, from, Eigen::Vector3d(8, 8, 0)));
  EXPECT_FALSE(IsReferentReachable(*room, from, Eigen::Vector3d(6.5, 8, 0)));
}

// No snap tolerance. The geodesic-nearest contact lies ON F; a point exactly on the edge is reachable by
// definition, a point past it is not.
TEST(ReachabilityGateTest, ScopeGating_D8_NoSnapTolerance) {
  auto room = LRoom();
  EXPECT_TRUE(IsReferentReachable(*room, Eigen::Vector3d(2, 2, 0),
                                  Eigen::Vector3d(6.0, 8, 0)));  // on the edge
  EXPECT_FALSE(IsReferentReachable(*room, Eigen::Vector3d(2, 2, 0),
                                   Eigen::Vector3d(6.1, 8, 0)));  // 0.1 m past it
}

TEST(ReachabilityGateTest, ScopeGating_E1_2_SpacingDispatch) {
  auto room = LRoom();
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  // The referent: a seat footprint (a referent without geometry is not
  // gated -- it has no approach point).
  tree.GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(Rect(7.5, 7.5, 8.5, 8.5)));
  Support sealed;
  sealed.has_contact = true;
  sealed.contact_point_world = Eigen::Vector3d(8, 8, 0);
  EXPECT_FALSE(PassesFamilyScopeGate(RelationKernelType::kAvatarSpacing, tree,
                                     1, Eigen::Vector3d(2, 2, 0), sealed,
                                     room.get()));
  // D8: the kernel's own geodesic to the contact is the path. Infinite = sealed.
  sealed.geodesic_length_m = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(PassesFamilyScopeGate(RelationKernelType::kAvatarSpacing, tree,
                                     1, Eigen::Vector3d(2, 2, 0), sealed,
                                     room.get()));
  Support open;
  open.has_contact = true;
  open.contact_point_world = Eigen::Vector3d(3, 3, 0);
  open.geodesic_length_m = std::sqrt(2.0);
  EXPECT_TRUE(PassesFamilyScopeGate(RelationKernelType::kAvatarSpacing, tree,
                                    1, Eigen::Vector3d(2, 2, 0), open,
                                    room.get()));
  // The actor standing inside the referent: the geodesic to it is 0 by containment.
  sealed.has_interior = true;
  EXPECT_TRUE(PassesFamilyScopeGate(RelationKernelType::kAvatarSpacing, tree,
                                    1, Eigen::Vector3d(2, 2, 0), sealed,
                                    room.get()));
}

// The gate judges reachability in the navigable the kernel encodes against,
// and only when the anchor stands on it.
TEST(ReachabilityGateTest, ScopeGating_E1_2_GateUsesEncoderNavigable) {
  auto room = LRoom();
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  // The referent: a seat footprint (a referent without geometry is not
  // gated -- it has no approach point).
  tree.GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(Rect(7.5, 7.5, 8.5, 8.5)));
  Support notch;
  notch.has_contact = true;
  notch.contact_point_world = Eigen::Vector3d(8, 8, 0);
  EXPECT_FALSE(PassesFamilyScopeGate(RelationKernelType::kAvatarSpacing, tree,
                                     1, Eigen::Vector3d(2, 2, 0), notch,
                                     room.get()));
  // D8: the "anchor off the navigable -> not gated" bypass is deleted. A referent with
  // no measured geodesic (a point) is judged by a path from wherever the anchor is: the
  // sealed notch stays unreachable.
  EXPECT_FALSE(PassesFamilyScopeGate(RelationKernelType::kAvatarSpacing, tree,
                                     1, Eigen::Vector3d(-5, -5, 0), notch,
                                     room.get()));
  // With no navigable there is no free space to judge a path in.
  EXPECT_TRUE(PassesFamilyScopeGate(RelationKernelType::kAvatarSpacing, tree,
                                    1, Eigen::Vector3d(2, 2, 0), notch,
                                    nullptr));
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
