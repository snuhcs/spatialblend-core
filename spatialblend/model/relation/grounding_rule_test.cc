// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/model/element/sub_region.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/model/relation/affordance_pose_grounding_rule.h"
#include "spatialblend/model/relation/avatar_spacing_kernel.h"
#include "spatialblend/model/relation/facing_grounding_rule.h"
#include "spatialblend/model/relation/spacing_grounding_rule.h"

namespace spatialblend {
namespace test {

class SpacingGroundingRuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tree_.AddElement(1);
    tree_.AddElement(2, 1);

    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 3) = 5.0;
    t(1, 3) = 5.0;
    tree_.GetElement(2)->SetLocalTransform(t);

    auto boundary = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{3, 3}, {7, 3}, {7, 7}, {3, 7}});
    tree_.GetElement(2)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(boundary));
  }

  ElementTree tree_;
};

TEST_F(SpacingGroundingRuleTest, ComputesContactPoint) {
  SpacingGroundingRule rule;
  auto support = rule.ComputeSupport(
      *tree_.GetElement(2), tree_,
      Eigen::Vector3d(0, 5, 0), 0.0);

  EXPECT_TRUE(support.has_contact);
  // Nearest boundary point to (0, 5) should be near (3, 5) — left edge
  EXPECT_NEAR(support.contact_point_local.x(), -2.0, 1.0);  // local frame relative to (5,5)
}

TEST_F(SpacingGroundingRuleTest, HasBoundary) {
  SpacingGroundingRule rule;
  auto support = rule.ComputeSupport(
      *tree_.GetElement(2), tree_,
      Eigen::Vector3d(0, 5, 0), 0.0);

  EXPECT_NE(support.boundary, nullptr);
  EXPECT_EQ(support.referent_element_id, 2);
}

TEST_F(SpacingGroundingRuleTest, SetsNavigableDef) {
  SpacingGroundingRule rule;
  auto nav = std::make_shared<NavigableSpatialDefinition>(
      std::make_shared<Polygon2D>(
          std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 20}, {0, 20}}));
  auto support = rule.ComputeSupport(
      *tree_.GetElement(2), tree_,
      Eigen::Vector3d(0, 5, 0), 0.0, nav.get());

  EXPECT_EQ(support.navigable_def, nav.get());
}

// -------------------------------------------------------------------------
// Regression coverage for the box/Polytope referent frame bug.
//
// A PolytopeSpatialDefinition stores its vertices verbatim in the OWNING
// ELEMENT's local frame (LOCAL half-extents around the element's own origin,
// per the sb_set_element_collider / sb_set_element_box contract) — see
// SpatialDefinition::RequiresElementWorldTransform. This is unlike
// NavigableSpatialDefinition (world-absolute) and SurfaceSpatialDefinition
// (self-contained, own derived frame). Before the fix,
// SpacingGroundingRule::ComputeSupport queried this local-frame geometry
// directly with the raw WORLD anchor position, silently treating a box
// placed away from the tree origin as if it sat at the world origin.
//
// Confirmed real-world repro (chair box at world (2, 1), half-extents 0.25,
// room diagonal ~7.155m): standing exactly at the chair produced proximity
// ~0.7339 instead of the correct ~1.0. These tests pin down the fixed
// behavior with concrete numeric assertions (not the pre-fix broken values,
// since the fix is already applied to the source under test).
class SpacingGroundingRulePolytopeTest : public ::testing::Test {
 protected:
  // 2D rotation about Z (yaw), matching the convention used throughout this
  // codebase (see AvatarSpacingKernelTest.EncodeRelativeYaw): (0,0)=cos,
  // (0,1)=-sin, (1,0)=sin, (1,1)=cos.
  static Eigen::Matrix4d MakeTransform(double x, double y, double yaw) {
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 0) = std::cos(yaw);
    t(0, 1) = -std::sin(yaw);
    t(1, 0) = std::sin(yaw);
    t(1, 1) = std::cos(yaw);
    t(0, 3) = x;
    t(1, 3) = y;
    return t;
  }

  // 5x5 room; diagonal = 5*sqrt(2) ~= 7.0710678.
  static std::shared_ptr<NavigableSpatialDefinition> MakeRoom() {
    auto boundary = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{0, 0}, {5, 0}, {5, 5}, {0, 5}});
    return std::make_shared<NavigableSpatialDefinition>(boundary);
  }

  // Box referent: LOCAL half-extents (0.25 x 0.25 x 1.0) around the
  // element's own origin — placed in the tree via SetLocalTransform.
  static std::shared_ptr<PolytopeSpatialDefinition> MakeBox() {
    return std::make_shared<PolytopeSpatialDefinition>(
        Eigen::Vector3d(-0.25, -0.25, -1.0), Eigen::Vector3d(0.25, 0.25, 1.0));
  }

  static double RoomDiagonal() { return std::sqrt(5.0 * 5.0 + 5.0 * 5.0); }

  ElementTree tree_;
};

TEST_F(SpacingGroundingRulePolytopeTest, ActorAtBoxCenterIsFullyProximate) {
  tree_.AddElement(1);
  tree_.AddElement(2, 1);
  tree_.GetElement(2)->SetLocalTransform(MakeTransform(2.0, 1.0, 0.0));
  tree_.GetElement(2)->SetSpatialDefinition(MakeBox());

  auto room = MakeRoom();
  SpacingGroundingRule rule;
  auto support = rule.ComputeSupport(*tree_.GetElement(2), tree_,
                                      Eigen::Vector3d(2.0, 1.0, 0.0), 0.0,
                                      room.get());

  AvatarSpacingKernel kernel;
  auto relation = kernel.Encode(Eigen::Vector3d(2.0, 1.0, 0.0), 0.0, support);

  // Standing exactly at the chair must resolve to (near-)full proximity.
  // Before the fix this produced ~0.7339 (broken world/local frame mismatch:
  // the box's local vertices, centered near the tree origin, were queried
  // with the raw world anchor (2, 1, 0) instead of the local-frame anchor
  // (0, 0, 0)).
  EXPECT_GE(relation.proximity, 1.0 - 1e-6);
}

TEST_F(SpacingGroundingRulePolytopeTest,
       ProximityMatchesClosedFormAtTwoMetersAxisAligned) {
  tree_.AddElement(1);
  tree_.AddElement(2, 1);
  tree_.GetElement(2)->SetLocalTransform(MakeTransform(2.0, 1.0, 0.0));
  tree_.GetElement(2)->SetSpatialDefinition(MakeBox());

  auto room = MakeRoom();

  // 2m beyond the box's +X face along the box's own local +X axis (== world
  // +X here, since yaw=0): the box's right face is at world x = 2.0 + 0.25.
  Eigen::Vector3d actor_pos(2.25 + 2.0, 1.0, 0.0);

  SpacingGroundingRule rule;
  auto support = rule.ComputeSupport(*tree_.GetElement(2), tree_, actor_pos,
                                      0.0, room.get());
  AvatarSpacingKernel kernel;
  auto relation = kernel.Encode(actor_pos, 0.0, support);

  // Closed form: proximity = 1 - geodesic_distance_to_nearest_edge / D, where
  // D is the room's bounding diagonal. With a clear line of sight (no
  // obstacles) geodesic distance equals the straight-line distance (2.0m).
  double expected_proximity = 1.0 - 2.0 / RoomDiagonal();
  EXPECT_NEAR(relation.proximity, expected_proximity, 1e-6);
}

TEST_F(SpacingGroundingRulePolytopeTest,
       ProximityMatchesClosedFormAtTwoMetersRotatedBox) {
  tree_.AddElement(1);
  tree_.AddElement(2, 1);
  const double yaw = M_PI / 4.0;
  tree_.GetElement(2)->SetLocalTransform(MakeTransform(2.0, 1.0, yaw));
  tree_.GetElement(2)->SetSpatialDefinition(MakeBox());

  auto room = MakeRoom();

  // 2m beyond the box's +X face, measured along the BOX'S OWN (rotated)
  // local +X axis: world offset = R(yaw) * (0.25 + 2.0, 0).
  const double local_offset = 0.25 + 2.0;
  Eigen::Vector3d actor_pos(2.0 + local_offset * std::cos(yaw),
                             1.0 + local_offset * std::sin(yaw), 0.0);

  SpacingGroundingRule rule;
  auto support = rule.ComputeSupport(*tree_.GetElement(2), tree_, actor_pos,
                                      0.0, room.get());
  AvatarSpacingKernel kernel;
  auto relation = kernel.Encode(actor_pos, 0.0, support);

  // Same closed form as the axis-aligned case: rotation is an isometry, so
  // the geodesic (here, straight-line) distance to the nearest edge is still
  // exactly 2.0m. A frame bug that ignores the referent's yaw would break
  // this case even if it happened to work for yaw=0.
  double expected_proximity = 1.0 - 2.0 / RoomDiagonal();
  EXPECT_NEAR(relation.proximity, expected_proximity, 1e-6);
}

// v1's approach direction is the direction of the actor's final geodesic leg onto
// the contact. The support carries where that leg starts, in the frame the anchor
// offset is encoded in: the anchor itself when it sees the contact, else the last
// waypoint of its geodesic (here the east end of a barrier).
TEST(SpacingGroundingApproachOrigin, IsTheStartOfTheFinalGeodesicLeg) {
  ElementTree tree;
  tree.AddElement(1);
  tree.AddElement(2, 1);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = 3.5;
  t(1, 3) = 1.5;
  tree.GetElement(2)->SetLocalTransform(t);
  tree.GetElement(2)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-0.5, -0.5, 0.0), Eigen::Vector3d(0.5, 0.5, 1.0)));
  NavigableSpatialDefinition nav(std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {10, 0}, {10, 10}, {0, 10}}));
  nav.SetObstacles(
      {FreeSpaceObstacle{1, std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
                                {0, 4}, {7, 4}, {7, 6}, {0, 6}})}},
      /*clearance=*/0.0);
  SpacingGroundingRule rule;

  // Behind the barrier: the geodesic turns at (7, 4), the barrier's south-east
  // corner; the box centre is (3.5, 1.5), so that is (3.5, 2.5) in its frame.
  const auto behind = rule.ComputeSupport(*tree.GetElement(2), tree,
                                          Eigen::Vector3d(3.5, 8.0, 0), 0.0, &nav);
  ASSERT_TRUE(behind.has_approach_origin);
  EXPECT_NEAR(behind.approach_origin_local.x(), 3.5, 1e-5);
  EXPECT_NEAR(behind.approach_origin_local.y(), 2.5, 1e-5);

  // East of the box with nothing between: the anchor sees the contact, so the
  // leg starts at the anchor, (3, 0) from the box centre.
  const auto seeing = rule.ComputeSupport(*tree.GetElement(2), tree,
                                          Eigen::Vector3d(6.5, 1.5, 0), 0.0, &nav);
  ASSERT_TRUE(seeing.has_approach_origin);
  EXPECT_NEAR(seeing.approach_origin_local.x(), 3.0, 1e-9);
  EXPECT_NEAR(seeing.approach_origin_local.y(), 0.0, 1e-9);
}

class FacingGroundingRuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tree_.AddElement(1);
    tree_.AddElement(2, 1);

    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 3) = 10.0;
    t(1, 3) = 5.0;
    tree_.GetElement(2)->SetLocalTransform(t);

    // Board boundary: 4x4 around (10, 5)
    auto boundary = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{8, 3}, {12, 3}, {12, 7}, {8, 7}});
    tree_.GetElement(2)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(boundary));
  }

  ElementTree tree_;
};

TEST_F(FacingGroundingRuleTest, ContactIsInHeadingDirection) {
  FacingGroundingRule rule;
  // Actor at (0, 5) facing right (yaw=0 → heading +X)
  auto support = rule.ComputeSupport(
      *tree_.GetElement(2), tree_,
      Eigen::Vector3d(0, 5, 0), 0.0);

  EXPECT_TRUE(support.has_contact);
  // Heading +X from (0,5) should hit the left boundary (x=8) of the board
  // In local frame (relative to board at 10,5): x = 8-10 = -2
}

TEST_F(FacingGroundingRuleTest, ContactChangesWithYaw) {
  FacingGroundingRule rule;
  // Actor at (10, 0) facing up (yaw=pi/2 → heading +Y)
  auto support = rule.ComputeSupport(
      *tree_.GetElement(2), tree_,
      Eigen::Vector3d(10, 0, 0), M_PI / 2.0);

  EXPECT_TRUE(support.has_contact);
  // Heading +Y from (10,0) should hit the bottom boundary (y=3) of the board
}

// Even when the gaze ray does NOT hit the referent (the actor is looking AWAY),
// facing must still ENCODE a normalized look-at (the nearest boundary point), so
// the target side can ring-map it onto its own (differently shaped) referent and
// reconstruct WHERE the actor was looking. Before this fix the ray-miss fallback
// left contact_normalized at its default, so the look-at could not transfer and
// degraded to the referent center. This is facing's symmetric encode/decode,
// mirroring the spacing support transfer.
TEST_F(FacingGroundingRuleTest, RayMissStillEncodesNormalizedLookAt) {
  FacingGroundingRule rule;
  // Actor at (0,4) facing LEFT (yaw=pi -> heading -X), away from the board at +X, so the
  // gaze ray misses. The look-at is the boundary point at the least angle from the gaze:
  // the board's silhouette vertex the turned-away gaze is nearest in angle, (8,7) (bearing
  // 20.6 deg from the anchor, 159.4 deg from the heading; (8,3) is at 172.9 deg). In the
  // tree-relative frame (board at (10,5)): x = 8-10 = -2, y = 7-5 = 2.
  auto support = rule.ComputeSupport(
      *tree_.GetElement(2), tree_, Eigen::Vector3d(0, 4, 0), M_PI);
  ASSERT_TRUE(support.has_contact);
  EXPECT_NEAR(support.contact_point_local.x(), -2.0, 1e-9);
  EXPECT_NEAR(support.contact_point_local.y(), 2.0, 1e-9);

  // The vertex sits exactly ON the (convex) board boundary, so encoding must
  // succeed via the radial strategy with radial_ratio == 1.0 (boundary).
  // A concrete numeric check (unlike comparing against another
  // default-constructed NormalizedPolygonPosition, whose scalar fields are
  // uninitialized on a default build and so make an unreliable oracle).
  EXPECT_EQ(support.contact_normalized.strategy,
            spatialblend::NormalizedPolygonPosition::Strategy::kRadial);
  EXPECT_NEAR(support.contact_normalized.radial_ratio, 1.0, 1e-6);
}

// The least-angle contact is a silhouette VERTEX when the
// gaze misses, and the vertex minimiser is discontinuous at the one heading where two vertices
// tie. The support names the vertex (look_at_vertex) so the definition's own discontinuity can
// be declared: -1 while the gaze hits the referent, a ring index while it misses, and a
// different index on the other side of the tie.
TEST_F(FacingGroundingRuleTest, TheSilhouetteVertexIsNamedAndChangesAcrossTheTie) {
  FacingGroundingRule rule;
  const auto at = [&](double yaw) {
    return rule.ComputeSupport(*tree_.GetElement(2), tree_, Eigen::Vector3d(0, 5, 0), yaw);
  };
  EXPECT_EQ(at(0.0).look_at_vertex, -1) << "the gaze hits the board";
  EXPECT_EQ(at(0.1).look_at_vertex, -1);
  const auto before = at(M_PI - 0.05), after = at(M_PI + 0.05);
  EXPECT_GE(before.look_at_vertex, 0);
  EXPECT_GE(after.look_at_vertex, 0);
  EXPECT_NE(before.look_at_vertex, after.look_at_vertex)
      << "the two silhouette vertices swap at the tie heading";
  EXPECT_EQ(at(M_PI - 0.05).look_at_vertex, before.look_at_vertex) << "stateless";
  // The vertex is the contact: the same ring point on its own side.
  EXPECT_GT((before.contact_point_local - after.contact_point_local).norm(), 1.0);
}

TEST_F(FacingGroundingRuleTest, TheSupportSaysWhetherTheAnchorsPathCrossedTheBoundary) {
  FacingGroundingRule rule;
  const auto crossed = [&](const Eigen::Vector3d& from, const Eigen::Vector3d& to) {
    rule.SetPreviousAnchorPosition(from);
    return rule.ComputeSupport(*tree_.GetElement(2), tree_, to, 0.0).look_at_boundary_crossed;
  };
  // The board is x 8..12, y 3..7.
  EXPECT_TRUE(crossed({7.0, 5, 0}, {9.0, 5, 0})) << "outside to inside";
  EXPECT_TRUE(crossed({9.0, 5, 0}, {7.0, 5, 0})) << "inside to outside";
  EXPECT_TRUE(crossed({7.0, 5, 0}, {13.0, 5, 0})) << "through it in one step: both ends outside";
  EXPECT_FALSE(crossed({9.0, 5, 0}, {11.0, 5, 0})) << "inside throughout";
  EXPECT_FALSE(crossed({0.0, 5, 0}, {7.0, 5, 0})) << "outside throughout";
  EXPECT_FALSE(crossed({0.0, 0, 0}, {7.0, 1, 0})) << "outside, passing below it";
  rule.SetPreviousAnchorPosition(std::nullopt);
  EXPECT_FALSE(rule.ComputeSupport(*tree_.GetElement(2), tree_, {9.0, 5, 0}, 0.0)
                   .look_at_boundary_crossed) << "no previous evaluation";
}

// An edge running parallel to the gaze. The boundary point CLOSEST (in distance) to the gaze
// ray is then ambiguous along that whole edge, and which end of it is closer flips as the
// heading turns through the parallel: the contact jumped by the length of the edge (here
// 3 m) for a milliradian of turn. The contact at the least ANGLE from the gaze is the same
// vertex on both sides of the parallel.
TEST_F(FacingGroundingRuleTest, LookAtContactIsContinuousAcrossAnEdgeParallelToTheGaze) {
  FacingGroundingRule rule;
  // Actor at (9,0), below the board whose bottom edge (y = 3, x in [8,12]) the +X gaze
  // runs along.
  double worst = 0.0;
  Eigen::Vector2d prev(0, 0);
  bool first = true;
  for (double yaw = -0.3; yaw <= 0.3; yaw += 0.001) {
    auto support = rule.ComputeSupport(*tree_.GetElement(2), tree_, Eigen::Vector3d(9, 0, 0), yaw);
    ASSERT_TRUE(support.has_contact);
    if (!first) worst = std::max(worst, (support.contact_point_local - prev).norm());
    prev = support.contact_point_local;
    first = false;
  }
  EXPECT_LT(worst, 0.05) << "the contact jumped by " << worst << " m for 1 mrad of heading";
}

// The look-at contact is continuous in the heading. It used to be the ray's hit point when the
// ray hit and the boundary point NEAREST THE ANCHOR when it missed: turning the gaze across the
// silhouette of the referent moved the contact by the whole width of the board in one step
// (a source-side look-at jump the receiver then decoded faithfully). The contact is the point
// of the boundary closest to the gaze ray: the hit point when there is one (distance 0), the
// silhouette point when the ray just misses, so the two agree where they meet.
TEST_F(FacingGroundingRuleTest, LookAtContactIsContinuousInTheHeading) {
  FacingGroundingRule rule;
  double worst = 0.0;
  Eigen::Vector2d prev(0, 0);
  bool first = true;
  for (double yaw = -0.6; yaw <= 0.6; yaw += 0.001) {
    auto support = rule.ComputeSupport(*tree_.GetElement(2), tree_, Eigen::Vector3d(0, 5, 0), yaw);
    ASSERT_TRUE(support.has_contact);
    if (!first) worst = std::max(worst, (support.contact_point_local - prev).norm());
    prev = support.contact_point_local;
    first = false;
  }
  // 1 mrad of heading moves the contact by at most ~ range * 1e-3 = 0.01 m on the boundary
  EXPECT_LT(worst, 0.05) << "the contact jumped by " << worst << " m for 1 mrad of heading";
}

// Regression: with the referent's own frame correctly distinguished from the
// tree's world transform, a genuine ray-HIT must be found (not fall through
// to the nearest-point fallback) whenever the actor is truly looking at the
// referent — including when the referent's boundary is offset from the
// tree's placement transform (as in this fixture, where the board's world
// transform sits at (10,5) but its boundary vertices are independently
// world-absolute at (8-12, 3-7)).
TEST_F(FacingGroundingRuleTest, StraightOnApproachHitsExpectedEdgePoint) {
  FacingGroundingRule rule;
  // Actor at (0,5) facing RIGHT (yaw=0 -> heading +X), directly at the board.
  auto support = rule.ComputeSupport(
      *tree_.GetElement(2), tree_, Eigen::Vector3d(0, 5, 0), 0.0);
  ASSERT_TRUE(support.has_contact);

  EXPECT_NEAR(support.contact_point_local.x(), -2.0, 1e-9);
  EXPECT_NEAR(support.contact_point_local.y(), 0.0, 1e-9);
  EXPECT_EQ(support.contact_normalized.strategy,
            spatialblend::NormalizedPolygonPosition::Strategy::kRadial);
  EXPECT_NEAR(support.contact_normalized.radial_ratio, 1.0, 1e-6);
}

// Regression coverage mirroring SpacingGroundingRulePolytopeTest above: the
// same LOCAL-frame PolytopeSpatialDefinition must also be handled correctly
// by FacingGroundingRule's ray-boundary intersection, not just Spacing's
// nearest-point query.
class FacingGroundingRulePolytopeTest : public ::testing::Test {
 protected:
  ElementTree tree_;
};

TEST_F(FacingGroundingRulePolytopeTest, TheSupportSaysWhetherThePathCrossedTheBox) {
  tree_.AddElement(1);
  tree_.AddElement(2, 1);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = 2.0;
  t(1, 3) = 1.0;
  tree_.GetElement(2)->SetLocalTransform(t);
  tree_.GetElement(2)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-0.25, -0.25, 0.7), Eigen::Vector3d(0.25, 0.25, 0.8),
          GeometryRole::kSurface));
  FacingGroundingRule rule;
  const auto crossed = [&](double from, double to) {
    rule.SetPreviousAnchorPosition(Eigen::Vector3d(from, 1.0, 0.0));
    return rule.ComputeSupport(*tree_.GetElement(2), tree_, Eigen::Vector3d(to, 1.0, 0.0), M_PI)
        .look_at_boundary_crossed;
  };
  EXPECT_TRUE(crossed(2.3, 2.2)) << "the box is x 1.75..2.25 (the element's local frame)";
  EXPECT_TRUE(crossed(2.3, 1.7)) << "through it";
  EXPECT_FALSE(crossed(2.3, 2.26));
}

TEST_F(FacingGroundingRulePolytopeTest, GazeRayHitsBoxEdgeInLocalFrame) {
  tree_.AddElement(1);
  tree_.AddElement(2, 1);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = 2.0;
  t(1, 3) = 1.0;
  tree_.GetElement(2)->SetLocalTransform(t);
  tree_.GetElement(2)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-0.25, -0.25, -1.0),
          Eigen::Vector3d(0.25, 0.25, 1.0)));

  FacingGroundingRule rule;
  // Actor at (2, -1.25) facing +Y (yaw=pi/2): straight at the box's bottom
  // (-Y) face, which sits at world y = 1 - 0.25 = 0.75.
  auto support = rule.ComputeSupport(*tree_.GetElement(2), tree_,
                                      Eigen::Vector3d(2.0, -1.25, 0.0),
                                      M_PI / 2.0);
  ASSERT_TRUE(support.has_contact);

  // Before the fix, the box's LOCAL vertices (centered near the tree origin)
  // would have been queried directly against the raw world anchor/heading,
  // so the ray would not correctly hit the translated box's face.
  EXPECT_NEAR(support.contact_point_local.x(), 0.0, 1e-9);
  EXPECT_NEAR(support.contact_point_local.y(), -0.25, 1e-9);
  EXPECT_EQ(support.contact_normalized.strategy,
            spatialblend::NormalizedPolygonPosition::Strategy::kRadial);
  EXPECT_NEAR(support.contact_normalized.radial_ratio, 1.0, 1e-6);
}

class AffordancePoseGroundingRuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tree_.AddElement(1);
    tree_.AddElement(2, 1);
    tree_.AddElement(3, 2);

    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 3) = 5.0;
    t(1, 3) = 5.0;
    tree_.GetElement(3)->SetLocalTransform(t);

    // Tabletop surface
    auto surface = std::make_shared<Polygon3D>(
        std::vector<Eigen::Vector3d>{{4, 4, 1}, {6, 4, 1}, {6, 6, 1}, {4, 6, 1}});
    tree_.GetElement(3)->SetSpatialDefinition(
        std::make_shared<SurfaceSpatialDefinition>(surface));
  }

  ElementTree tree_;
};

TEST_F(AffordancePoseGroundingRuleTest, ProjectsAnchorOntoSurface) {
  AffordancePoseGroundingRule rule;
  // Anchor at surface center (5, 5)
  auto support = rule.ComputeSupport(
      *tree_.GetElement(3), tree_,
      Eigen::Vector3d(5, 5, 2), 0.0);

  EXPECT_TRUE(support.has_contact);
  // Contact should be near center in local frame → (0, 0)
  EXPECT_NEAR(support.contact_point_local.norm(), 0.0, 0.5);
}

TEST_F(AffordancePoseGroundingRuleTest, HasNormalizedContact) {
  AffordancePoseGroundingRule rule;
  auto support = rule.ComputeSupport(
      *tree_.GetElement(3), tree_,
      Eigen::Vector3d(5, 5, 2), 0.0);

  EXPECT_TRUE(support.has_contact);
  EXPECT_NE(support.boundary, nullptr);
}

TEST_F(AffordancePoseGroundingRuleTest, GroundsOntoNearestSubRegion) {
  // Fresh referent at the origin (identity transform → local == world) with a
  // 2x2 boundary split into a left and a right patch.
  tree_.AddElement(20, 1);
  auto boundary = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {2, 0}, {2, 2}, {0, 2}});
  tree_.GetElement(20)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(boundary));
  auto* def = tree_.GetElement(20)->GetSpatialDefinition();

  SubRegion left;
  left.name = "left_patch";
  left.boundary = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {1, 0}, {1, 2}, {0, 2}});
  def->AddSubRegion(left);
  SubRegion right;
  right.name = "right_patch";
  right.boundary = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{1, 0}, {2, 0}, {2, 2}, {1, 2}});
  def->AddSubRegion(right);

  AffordancePoseGroundingRule rule;
  // Anchor at (1.8, 1.0) lands inside the right patch.
  auto support = rule.ComputeSupport(
      *tree_.GetElement(20), tree_, Eigen::Vector3d(1.8, 1.0, 0.0), 0.0);
  EXPECT_EQ(support.sub_region_name, "right_patch");
  EXPECT_TRUE(support.has_contact);
}

TEST_F(AffordancePoseGroundingRuleTest, NoSubRegionsKeepsWholeBoundary) {
  AffordancePoseGroundingRule rule;
  auto support = rule.ComputeSupport(
      *tree_.GetElement(3), tree_, Eigen::Vector3d(5, 5, 2), 0.0);
  EXPECT_TRUE(support.sub_region_name.empty());
}

// Container depth for a Polytope (local-frame) container: an actor on the floor
// inside the box is ON its bottom face, so the 3D closest point would be the actor
// itself and the depth below the border 0. The contact is the planar nearest edge
// point, so the depth is the real distance to the border.
TEST(PolytopeContainerDepthTest, DepthBelowTheBorderIsNotZero) {
  ElementTree tree;
  tree.AddElement(1);
  tree.AddElement(2, 1);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = 5.0;
  t(1, 3) = 5.0;
  tree.GetElement(2)->SetLocalTransform(t);
  tree.GetElement(2)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-2, -2, 0), Eigen::Vector3d(2, 2, 1),
          GeometryRole::kContainment));
  NavigableSpatialDefinition room(std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 20}, {0, 20}}));

  SpacingGroundingRule rule;
  auto support = rule.ComputeSupport(*tree.GetElement(2), tree,
                                     Eigen::Vector3d(6.0, 5.0, 0.0), 0.0, &room);
  ASSERT_TRUE(support.has_contact);
  EXPECT_NEAR(support.contact_point_world.x(), 7.0, 1e-9);
  EXPECT_NEAR(support.contact_point_world.y(), 5.0, 1e-9);
  ASSERT_TRUE(support.has_border_point);
  support.container_self = true;
  AvatarSpacingKernel kernel;
  const auto gr = kernel.Encode(Eigen::Vector3d(6.0, 5.0, 0.0), 0.0, support);
  EXPECT_NEAR(gr.support_distance_m, 1.0, 1e-9);
}

}  // namespace test
}  // namespace spatialblend
