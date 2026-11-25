// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/affordance_pose_grounding_rule.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/model/element/sub_region.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend::test {

static Eigen::Matrix4d Transform(double x, double y, double z, double yaw) {
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 0) = std::cos(yaw);
  t(0, 1) = -std::sin(yaw);
  t(1, 0) = std::sin(yaw);
  t(1, 1) = std::cos(yaw);
  t(0, 3) = x;
  t(1, 3) = y;
  t(2, 3) = z;
  return t;
}

static std::shared_ptr<Polygon3D> MakeBakedSurface(double tx, double ty,
                                                   double tz, double yaw) {
  const double z = 0.8;

  Eigen::Vector4d p0(0.0, 0.0, z, 1.0);
  Eigen::Vector4d p1(2.0, 0.0, z, 1.0);
  Eigen::Vector4d p2(2.0, 1.0, z, 1.0);
  Eigen::Vector4d p3(0.0, 1.0, z, 1.0);

  Eigen::Matrix4d xform = Transform(tx, ty, 0.0, yaw);
  p0 = xform * p0;
  p1 = xform * p1;
  p2 = xform * p2;
  p3 = xform * p3;

  return std::make_shared<Polygon3D>(
      std::vector<Eigen::Vector3d>{{p0.x(), p0.y(), p0.z() + tz},
                                   {p1.x(), p1.y(), p1.z() + tz},
                                   {p2.x(), p2.y(), p2.z() + tz},
                                   {p3.x(), p3.y(), p3.z() + tz}});
}

TEST(AffordancePoseGroundingRuleTest,
     ProjectsBakedSurfaceWithoutComposingTreeTransform) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  ASSERT_TRUE(tree.AddElement(2, 1).ok());

  tree.GetElement(2)->SetLocalTransform(Transform(10.0, 15.0, 0.0, M_PI / 4.0));
  tree.GetElement(2)->SetSpatialDefinition(
      std::make_shared<SurfaceSpatialDefinition>(
          MakeBakedSurface(3.0, 7.0, 0.0, M_PI / 2.0)));

  AffordancePoseGroundingRule rule;
  const Eigen::Vector3d anchor_world(2.5, 7.2, 1.6);
  Support support =
      rule.ComputeSupport(*tree.GetElement(2), tree, anchor_world, 0.0);

  ASSERT_TRUE(support.has_contact);
  EXPECT_NEAR(support.contact_point_world.x(), 2.5, 1e-9);
  EXPECT_NEAR(support.contact_point_world.y(), 7.2, 1e-9);
  EXPECT_NEAR(support.contact_point_world.z(), 0.8, 1e-9);
  EXPECT_NEAR(support.GetYaw(), M_PI / 2.0, 1e-9);
}

TEST(AffordancePoseGroundingRuleTest,
     UsesPatchLocalFrameForNamedSubRegionSelection) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  ASSERT_TRUE(tree.AddElement(2, 1).ok());

  tree.GetElement(2)->SetLocalTransform(
      Transform(100.0, 40.0, 0.0, M_PI / 3.0));
  auto surface = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      {0, 0, 0.8}, {4, 0, 0.8}, {4, 2, 0.8}, {0, 2, 0.8}});
  auto def = std::make_shared<SurfaceSpatialDefinition>(surface);

  SubRegion left;
  left.name = "left";
  // Polygon3D's chart is centered at (2, 1, .8), so patch transforms are
  // relative to that chart, not the source vertices' old XY coordinates.
  left.local_transform = Transform(-2.0, -1.0, 0.0, 0.0);
  left.boundary = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {2, 0}, {2, 2}, {0, 2}});
  def->AddSubRegion(left);

  SubRegion right;
  right.name = "right";
  right.local_transform = Transform(0.0, -1.0, 0.0, 0.0);
  right.boundary = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {2, 0}, {2, 2}, {0, 2}});
  def->AddSubRegion(right);

  tree.GetElement(2)->SetSpatialDefinition(def);

  AffordancePoseGroundingRule rule;
  Support support = rule.ComputeSupport(*tree.GetElement(2), tree,
                                        Eigen::Vector3d(3.2, 1.0, 1.5), 0.0);

  ASSERT_TRUE(support.has_contact);
  EXPECT_EQ(support.sub_region_name, "right");
  EXPECT_NEAR(support.contact_point_world.x(), 3.2, 1e-9);
  EXPECT_NEAR(support.contact_point_world.y(), 1.0, 1e-9);
  EXPECT_NEAR(support.contact_point_world.z(), 0.8, 1e-9);
  EXPECT_NEAR(support.GetYaw(), 0.0, 1e-9);
}

TEST(AffordancePoseGroundingRuleTest, NoSpatialDefLeavesContactUnresolved) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());

  AffordancePoseGroundingRule rule;
  Support support = rule.ComputeSupport(*tree.GetElement(1), tree,
                                        Eigen::Vector3d(1, 2, 3), 0.0);

  EXPECT_FALSE(support.has_contact);
  EXPECT_EQ(support.sub_region_name, "");
  EXPECT_EQ(support.boundary, nullptr);
}

TEST(AffordancePoseGroundingRuleTest, DegenerateSurfaceBoundaryFailsEncoding) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());

  // Degenerate 3-point collinear polygon cannot produce a valid normalized
  // position; Encode() must fail and we must not report a fake contact.
  auto degenerate = std::make_shared<Polygon3D>(
      std::vector<Eigen::Vector3d>{{0, 0, 0.8}, {1, 0, 0.8}, {2, 0, 0.8}});
  tree.GetElement(1)->SetSpatialDefinition(
      std::make_shared<SurfaceSpatialDefinition>(degenerate));

  AffordancePoseGroundingRule rule;
  Support support = rule.ComputeSupport(*tree.GetElement(1), tree,
                                        Eigen::Vector3d(1, 5, 1.0), 0.0);

  EXPECT_FALSE(support.has_contact);
  EXPECT_EQ(support.contact_point_local.x(), 0.0);
  EXPECT_EQ(support.contact_point_local.y(), 0.0);
}

TEST(AffordancePoseGroundingRuleTest, ProjectsOutOfBoundsAnchorToSurfaceEdge) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());

  auto surface = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      {0, 0, 0.8}, {2, 0, 0.8}, {2, 1, 0.8}, {0, 1, 0.8}});
  tree.GetElement(1)->SetSpatialDefinition(
      std::make_shared<SurfaceSpatialDefinition>(surface));

  AffordancePoseGroundingRule rule;
  Support support = rule.ComputeSupport(*tree.GetElement(1), tree,
                                        Eigen::Vector3d(1.5, 2.7, 1.6), 0.0);

  ASSERT_TRUE(support.has_contact);
  EXPECT_NEAR(support.contact_point_world.x(), 1.5, 1e-9);
  EXPECT_NEAR(support.contact_point_world.y(), 1.0, 1e-9);
}

TEST(AffordancePoseGroundingRuleTest,
     GroundsTransformedSeatBoundaryAfterProjectionRoundoff) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  const double yaw = .49;
  const double tx = 13.7;
  const double ty = -4.2;
  const double tz = .35;
  tree.GetElement(1)->SetSpatialDefinition(
      std::make_shared<SurfaceSpatialDefinition>(MakeBakedSurface(tx, ty, tz, yaw)));

  // The anchor is infinitesimally past the transformed +X edge. Projection
  // must clamp it to the authored boundary, and the resulting local chart
  // coordinate must remain admissible to Encode.
  const Eigen::Vector4d beyond =
      Transform(tx, ty, 0.0, yaw) * Eigen::Vector4d(2.0 + 1e-12, .5, .8, 1.0);
  AffordancePoseGroundingRule rule;
  Support support = rule.ComputeSupport(
      *tree.GetElement(1), tree,
      Eigen::Vector3d(beyond.x(), beyond.y(), beyond.z() + tz + .4), 0.0);

  ASSERT_TRUE(support.has_contact);
  EXPECT_NEAR(support.contact_normalized.radial_ratio, 1.0, 1e-9);
  EXPECT_NEAR(support.contact_point_world.z(), .8 + tz, 1e-9);
}

TEST(AffordancePoseGroundingRuleTest,
     EncodesBakedLSeatAroundVisibilityKernelNotOutsideAreaCentroid) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  // Baked surface vertices already own the geometry frame. This unrelated
  // tree transform must not be applied to either contact or chart heading.
  tree.GetElement(1)->SetLocalTransform(Transform(90, 40, 12, -.8));
  const Eigen::Matrix4d frame = Transform(8, -4, .65, .4);
  const auto world = [&frame](double x, double y) -> Eigen::Vector3d {
    return (frame * Eigen::Vector4d(x, y, 0, 1)).head<3>();
  };
  auto surface = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      world(0, 0), world(3, 0), world(3, 1), world(1, 1), world(1, 3),
      world(0, 3)});
  tree.GetElement(1)->SetSpatialDefinition(
      std::make_shared<SurfaceSpatialDefinition>(surface));

  // Independent E oracle: the kernel is [0,1]^2 with centre (.5,.5),
  // whereas the ordinary area centroid (1.1,1.1) is in the missing notch.
  // Forward is the authored +Y axis; both arm rays extend 2.5 from centre.
  struct Sample { double x, y, angle, radius; };
  const Sample samples[] = {{.5, .5, 0, 0},
                            {1.75, .5, -M_PI / 2, .5},
                            {.5, 1.75, 0, .5},
                            {3, .5, -M_PI / 2, 1},
                            {.5, 3, 0, 1},
                            {1, 1, -M_PI / 4, 1}};
  AffordancePoseGroundingRule rule;
  for (const auto& sample : samples) {
    SCOPED_TRACE(::testing::Message() << sample.x << "," << sample.y);
    const Eigen::Vector3d expected = world(sample.x, sample.y);
    const auto support = rule.ComputeSupport(
        *tree.GetElement(1), tree, expected + Eigen::Vector3d(0, 0, .2), 0);
    ASSERT_TRUE(support.has_contact);
    EXPECT_EQ(support.contact_normalized.strategy,
              NormalizedPolygonPosition::Strategy::kRadial);
    EXPECT_NEAR(support.contact_normalized.radial_ratio, sample.radius, 1e-10);
    // Projection/inverse-frame roundoff may turn the mathematical centre
    // into a tiny nonzero delta, whose direction is intentionally undefined.
    if (sample.radius != 0)
      EXPECT_NEAR(support.contact_normalized.angle, sample.angle, 1e-10);
    EXPECT_TRUE(support.contact_point_world.isApprox(expected, 1e-10));
    EXPECT_NEAR(support.GetYaw(), .4, 1e-10);
  }
}

TEST(AffordancePoseGroundingRuleTest,
     RejectsNonStarSeatWithoutInventingCentroidOrMVCContact) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  // The U's opposing inner walls require kernel x <= 1 and x >= 2.
  // It is a valid simple polygon, but cannot use one strict radial chart.
  auto surface = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      {0, 0, .7}, {3, 0, .7}, {3, 3, .7}, {2, 3, .7},
      {2, 1, .7}, {1, 1, .7}, {1, 3, .7}, {0, 3, .7}});
  tree.GetElement(1)->SetSpatialDefinition(
      std::make_shared<SurfaceSpatialDefinition>(surface));
  AffordancePoseGroundingRule rule;
  const auto support = rule.ComputeSupport(*tree.GetElement(1), tree,
                                          Eigen::Vector3d(.5, .5, 1.2), 0);
  EXPECT_FALSE(support.has_contact);
}

TEST(AffordancePoseGroundingRuleTest,
     RejectsAllInvalidPatchesWithoutFallingBackToWholeSurface) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  auto surface = std::make_shared<Polygon3D>(
      std::vector<Eigen::Vector3d>{{0, 0, 0.8}, {2, 0, 0.8}, {2, 1, 0.8},
                                   {0, 1, 0.8}});
  auto def = std::make_shared<SurfaceSpatialDefinition>(surface);
  SubRegion invalid;
  invalid.name = "bad";
  invalid.boundary = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {1, 0}, {2, 0}});
  def->AddSubRegion(std::move(invalid));
  tree.GetElement(1)->SetSpatialDefinition(def);

  AffordancePoseGroundingRule rule;
  Support support = rule.ComputeSupport(*tree.GetElement(1), tree,
                                        Eigen::Vector3d(1, 0.5, 1.5), 0.0);
  EXPECT_FALSE(support.has_contact);
  EXPECT_EQ(support.boundary, nullptr);
}

TEST(AffordancePoseGroundingRuleTest, RejectsNonFiniteAnchor) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  tree.GetElement(1)->SetSpatialDefinition(
      std::make_shared<SurfaceSpatialDefinition>(
          std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
              {0, 0, 0.8}, {2, 0, 0.8}, {2, 1, 0.8}, {0, 1, 0.8}})));

  AffordancePoseGroundingRule rule;
  Support support = rule.ComputeSupport(
      *tree.GetElement(1), tree,
      Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0.5, 1.5),
      0.0);
  EXPECT_FALSE(support.has_contact);
  EXPECT_EQ(support.boundary, nullptr);
}

TEST(AffordancePoseGroundingRuleTest, RejectsSingularElementLocalChart) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  Eigen::Matrix4d singular = Eigen::Matrix4d::Identity();
  singular(0, 0) = 0.0;
  tree.GetElement(1)->SetLocalTransform(singular);
  tree.GetElement(1)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(1, 1, 1)));

  AffordancePoseGroundingRule rule;
  Support support = rule.ComputeSupport(*tree.GetElement(1), tree,
                                        Eigen::Vector3d(0, 0, 0), 0.0);
  EXPECT_FALSE(support.has_contact);
  EXPECT_EQ(support.boundary, nullptr);
}

TEST(AffordancePoseGroundingRuleTest,
     GeometrylessFallbackKeepsReferentWorldTransform) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  tree.GetElement(1)->SetLocalTransform(Transform(4.0, -2.0, 0.7, 0.3));

  AffordancePoseGroundingRule rule;
  Support support = rule.ComputeSupport(*tree.GetElement(1), tree,
                                        Eigen::Vector3d(4, -2, 0.7), 0.0);
  EXPECT_FALSE(support.has_contact);
  EXPECT_EQ(support.boundary, nullptr);
  EXPECT_NEAR(support.GetPosition3D().x(), 4.0, 1e-12);
  EXPECT_NEAR(support.GetPosition3D().y(), -2.0, 1e-12);
  EXPECT_NEAR(support.GetPosition3D().z(), 0.7, 1e-12);
  EXPECT_NEAR(support.GetYaw(), 0.3, 1e-12);
}

}  // namespace spatialblend::test
