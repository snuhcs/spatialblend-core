// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "spatialblend/model/testing/hierarchy_extractor_fixture.h"

namespace spatialblend::test {
namespace {

TEST(HierarchyContractTest, ComposesParentTransformsForSeatSupport) {
  HierarchyExtractorFixture fixture;

  const Eigen::Matrix4d world = fixture.source_tree.GetWorldTransform(
      HierarchyExtractorFixture::kSourceSeatA);
  // Region (10,5, yaw +90) rotates the chair's local +X to world +Y.
  EXPECT_NEAR(world(0, 3), 10.0, 1e-12);
  EXPECT_NEAR(world(1, 3), 6.0, 1e-12);
  EXPECT_NEAR(world(2, 3), 0.45, 1e-12);
  // Production tree composition of local (.10, -.05, .10) must match the
  // independently calculated +90-degree analytic support coordinate.
  const Eigen::Vector4d local_support(0.10, -0.05, 0.10, 1.0);
  const Eigen::Vector3d actual_support = (world * local_support).head<3>();
  const Eigen::Vector3d expected_support = fixture.SourceSeatTopWorld();
  EXPECT_NEAR(actual_support.x(), expected_support.x(), 1e-12);
  EXPECT_NEAR(actual_support.y(), expected_support.y(), 1e-12);
  EXPECT_NEAR(actual_support.z(), expected_support.z(), 1e-12);
}

TEST(HierarchyContractTest, UsesOnlyExplicitCrossCategorySeatCorrespondence) {
  HierarchyExtractorFixture fixture;
  const auto source_chair =
      fixture.source_tree.GetElement(HierarchyExtractorFixture::kSourceChair);
  const auto target_sofa =
      fixture.target_tree.GetElement(HierarchyExtractorFixture::kTargetSofa);
  ASSERT_NE(source_chair, nullptr);
  ASSERT_NE(target_sofa, nullptr);
  EXPECT_EQ(source_chair->GetTag(), "chair");
  EXPECT_EQ(target_sofa->GetTag(), "sofa");
  EXPECT_NE(HierarchyExtractorFixture::kSourceSeatA,
            HierarchyExtractorFixture::kTargetSeatA);

  EXPECT_EQ(fixture.linker.Translate({HierarchyExtractorFixture::kSourceSpace,
                                      HierarchyExtractorFixture::kSourceSeatA},
                                     HierarchyExtractorFixture::kTargetSpace),
            HierarchyExtractorFixture::kTargetSeatA);
  EXPECT_EQ(fixture.linker.Translate({HierarchyExtractorFixture::kSourceSpace,
                                      HierarchyExtractorFixture::kSourceChair},
                                     HierarchyExtractorFixture::kTargetSpace),
            kInvalidElementId);
  EXPECT_FALSE(fixture.linker
                   .GetSource({HierarchyExtractorFixture::kTargetSpace,
                               HierarchyExtractorFixture::kTargetSeatB})
                   .has_value());
  const Eigen::Vector3d actual_a =
      fixture.target_tree
          .GetWorldTransform(HierarchyExtractorFixture::kTargetSeatA)
          .block<3, 1>(0, 3);
  const Eigen::Vector3d actual_b =
      fixture.target_tree
          .GetWorldTransform(HierarchyExtractorFixture::kTargetSeatB)
          .block<3, 1>(0, 3);
  const Eigen::Vector3d expected_a = fixture.TargetSeatACenterWorld();
  const Eigen::Vector3d expected_b = fixture.TargetSeatBCenterWorld();
  EXPECT_NEAR(actual_a.x(), expected_a.x(), 1e-12);
  EXPECT_NEAR(actual_a.y(), expected_a.y(), 1e-12);
  EXPECT_NEAR(actual_a.z(), expected_a.z(), 1e-12);
  EXPECT_NEAR(actual_b.x(), expected_b.x(), 1e-12);
  EXPECT_NEAR(actual_b.y(), expected_b.y(), 1e-12);
  EXPECT_NEAR(actual_b.z(), expected_b.z(), 1e-12);
  EXPECT_NEAR((actual_a - actual_b).norm(), 1.0, 1e-12);
}

TEST(HierarchyContractTest, DeletionRequiresTreeAndLinkLifecycleHandling) {
  HierarchyExtractorFixture fixture;
  ASSERT_TRUE(
      fixture.target_tree.RemoveElement(HierarchyExtractorFixture::kTargetSeatA)
          .ok());
  EXPECT_EQ(
      fixture.target_tree.GetElement(HierarchyExtractorFixture::kTargetSeatA),
      nullptr);

  // ElementLinker is deliberately a bare handle map. It cannot validate a
  // scene tree, so tree deletion alone leaves a stale handle until its own
  // lifecycle API removes it.
  EXPECT_EQ(fixture.linker.Translate({HierarchyExtractorFixture::kSourceSpace,
                                      HierarchyExtractorFixture::kSourceSeatA},
                                     HierarchyExtractorFixture::kTargetSpace),
            HierarchyExtractorFixture::kTargetSeatA);
  EXPECT_TRUE(
      fixture.linker.RemoveTarget({HierarchyExtractorFixture::kTargetSpace,
                                   HierarchyExtractorFixture::kTargetSeatA}));
  EXPECT_EQ(fixture.linker.Translate({HierarchyExtractorFixture::kSourceSpace,
                                      HierarchyExtractorFixture::kSourceSeatA},
                                     HierarchyExtractorFixture::kTargetSpace),
            kInvalidElementId);
}

}  // namespace
}  // namespace spatialblend::test
