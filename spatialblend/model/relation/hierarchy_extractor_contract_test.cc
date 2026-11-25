// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/model/testing/hierarchy_extractor_fixture.h"
#include "spatialblend/model/relation/affordance_pose_grounding_rule.h"
#include "spatialblend/model/relation/avatar_affordance_pose_kernel.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/scope_resolver.h"

namespace spatialblend::test {
namespace {

bool Contains(const std::vector<element_id_t>& ids, element_id_t id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

void ExpectIds(const std::vector<element_id_t>& actual,
               const std::vector<element_id_t>& expected) {
  EXPECT_EQ(actual, expected);
}

TEST(ExtractorContractTest, ResolvesTransformedContainmentHierarchy) {
  HierarchyExtractorFixture fixture;

  EXPECT_EQ(FindDeepestContaining(
                fixture.SourcePointInRegionOutsideChairWorld(),
                HierarchyExtractorFixture::kSourceRoot, fixture.source_tree),
            HierarchyExtractorFixture::kSourceRegion);
  EXPECT_EQ(FindDeepestContaining(fixture.SourceSeatCenterWorld(),
                                  HierarchyExtractorFixture::kSourceRoot,
                                  fixture.source_tree),
            HierarchyExtractorFixture::kSourceChair);
  EXPECT_EQ(FindDeepestContaining(fixture.SourcePointOutsideRegionWorld(),
                                  HierarchyExtractorFixture::kSourceRoot,
                                  fixture.source_tree),
            HierarchyExtractorFixture::kSourceRoot);

  // These are hierarchy candidate sets only; they do not establish line of
  // sight or navigable-path reachability.
  ExpectIds(ScopeResolver::Resolve(ScopePolicy::kReachable,
                                   fixture.SourcePointOutsideRegionWorld(),
                                   HierarchyExtractorFixture::kSourceRoot,
                                   fixture.source_tree),
            {HierarchyExtractorFixture::kSourceRegion});
  ExpectIds(ScopeResolver::Resolve(
                ScopePolicy::kReachable,
                fixture.SourcePointInRegionOutsideChairWorld(),
                HierarchyExtractorFixture::kSourceRoot, fixture.source_tree),
            // The chair, and the region's border referent: the region's interior
            // is solved by relations (it has the chair), so while the actor is
            // inside it the region is a referent too (ScopeResolver::HasBorderReferent).
            {HierarchyExtractorFixture::kSourceChair,
             HierarchyExtractorFixture::kSourceRegion});
  ExpectIds(ScopeResolver::Resolve(
                ScopePolicy::kReachable, fixture.SourceSeatCenterWorld(),
                HierarchyExtractorFixture::kSourceRoot, fixture.source_tree),
            {HierarchyExtractorFixture::kSourceChair});
  const auto region_scope = ScopeResolver::Resolve(
      ScopePolicy::kProximal, fixture.SourcePointInRegionOutsideChairWorld(),
      HierarchyExtractorFixture::kSourceRoot, fixture.source_tree);
  EXPECT_TRUE(Contains(region_scope, HierarchyExtractorFixture::kSourceChair));
  EXPECT_TRUE(Contains(region_scope, HierarchyExtractorFixture::kSourceSeatA));

  const auto chair_scope = ScopeResolver::Resolve(
      ScopePolicy::kProximal, fixture.SourceSeatCenterWorld(),
      HierarchyExtractorFixture::kSourceRoot, fixture.source_tree);
  ASSERT_EQ(chair_scope.size(), 1u);
  EXPECT_EQ(chair_scope.front(), HierarchyExtractorFixture::kSourceSeatA);
}

TEST(ExtractorContractTest, KeepsBakedNavigableContainmentInWorldFrame) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  ASSERT_TRUE(tree.AddElement(2, 1).ok());
  // The tree placement is intentionally unrelated to the baked world polygon.
  tree.GetElement(2)->SetLocalTransform(
      HierarchyExtractorFixture::Transform(50.0, -30.0, 0.0, M_PI / 3.0));
  auto boundary = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {1.0, 2.0}, {5.0, 2.0}, {5.0, 3.0}, {1.0, 3.0}});
  tree.GetElement(2)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(boundary, nullptr, 0.0,
                                                   GeometryRole::kContainment));

  EXPECT_EQ(FindDeepestContaining(Eigen::Vector3d(4.5, 2.5, 0.0), 1, tree), 2);
  EXPECT_EQ(FindDeepestContaining(Eigen::Vector3d(4.5, 3.5, 0.0), 1, tree), 1);
}

TEST(ExtractorContractTest, RotatesLocalNonSquareContainmentBeforeQuery) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  ASSERT_TRUE(tree.AddElement(2, 1).ok());
  tree.GetElement(2)->SetLocalTransform(
      HierarchyExtractorFixture::Transform(10.0, 5.0, 0.0, M_PI / 2.0));
  // A non-square local rectangle makes the yaw conversion observable.
  tree.GetElement(2)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-2.0, -0.5, -1.0), Eigen::Vector3d(2.0, 0.5, 1.0),
          GeometryRole::kContainment));

  // World (10, 6.5) is local (1.5, 0): inside only after the inverse yaw.
  EXPECT_EQ(FindDeepestContaining(Eigen::Vector3d(10.0, 6.5, 0.0), 1, tree), 2);
  // World (9.25, 5) is local (0, .75): outside the narrow local Y extent.
  EXPECT_EQ(FindDeepestContaining(Eigen::Vector3d(9.25, 5.0, 0.0), 1, tree), 1);
}

TEST(ExtractorContractTest, ExtractsAnalyticPolytopeSupportForDistinctAnchors) {
  HierarchyExtractorFixture fixture;
  AffordancePoseGroundingRule rule;
  AvatarAffordancePoseKernel kernel;
  const Element& seat =
      *fixture.source_tree.GetElement(HierarchyExtractorFixture::kSourceSeatA);

  const Eigen::Vector3d contact_anchor = fixture.SourceSeatTopWorld();
  const Eigen::Vector3d elevated_anchor =
      contact_anchor + Eigen::Vector3d(0.0, 0.0, 0.30);
  const double anchor_yaw = M_PI / 2.0 + 0.25;
  const Support contact = rule.ComputeSupport(seat, fixture.source_tree,
                                              contact_anchor, anchor_yaw);
  const Support elevated = rule.ComputeSupport(seat, fixture.source_tree,
                                               elevated_anchor, anchor_yaw);
  const GroundedRelation at_contact =
      kernel.Encode(contact_anchor, anchor_yaw, contact);
  const GroundedRelation above_seat =
      kernel.Encode(elevated_anchor, anchor_yaw, elevated);

  ASSERT_TRUE(contact.has_contact);
  EXPECT_NEAR(contact.contact_point_world.x(), 10.05, 1e-12);
  EXPECT_NEAR(contact.contact_point_world.y(), 6.10, 1e-12);
  EXPECT_NEAR(contact.contact_point_world.z(), 0.55, 1e-12);
  EXPECT_NEAR(contact.contact_point_local.x(), 0.10, 1e-12);
  EXPECT_NEAR(contact.contact_point_local.y(), -0.05, 1e-12);
  EXPECT_NEAR(at_contact.proximity, 1.0, 1e-12);
  EXPECT_NEAR(above_seat.proximity, 1.0 / 1.30, 1e-12);
  ASSERT_EQ(at_contact.encoded_state.size(), 3);
  EXPECT_NEAR(at_contact.encoded_state[2], 0.25, 1e-12);
}

TEST(ExtractorContractTest,
     ProjectsBakedSurfaceContactWithoutTreeFrameComposition) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  ASSERT_TRUE(tree.AddElement(2, 1).ok());
  // This transform must not move the Surface's already world-absolute vertices.
  tree.GetElement(2)->SetLocalTransform(
      HierarchyExtractorFixture::Transform(100.0, 100.0, 0.0, M_PI / 2.0));
  auto surface = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      {1.0, 2.0, 0.8}, {2.0, 2.0, 0.8}, {2.0, 3.0, 0.8}, {1.0, 3.0, 0.8}});
  tree.GetElement(2)->SetSpatialDefinition(
      std::make_shared<SurfaceSpatialDefinition>(surface));

  const Eigen::Vector3d anchor(1.5, 2.5, 1.1);
  AffordancePoseGroundingRule rule;
  AvatarAffordancePoseKernel kernel;
  const Support support =
      rule.ComputeSupport(*tree.GetElement(2), tree, anchor, 0.0);
  const GroundedRelation relation = kernel.Encode(anchor, 0.0, support);

  EXPECT_NEAR(support.contact_point_world.x(), 1.5, 1e-12);
  EXPECT_NEAR(support.contact_point_world.y(), 2.5, 1e-12);
  EXPECT_NEAR(support.contact_point_world.z(), 0.8, 1e-12);
  EXPECT_NEAR(relation.proximity, 1.0 / 1.30, 1e-12);
}

}  // namespace
}  // namespace spatialblend::test
