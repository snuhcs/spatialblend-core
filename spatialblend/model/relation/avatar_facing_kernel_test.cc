// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/avatar_facing_kernel.h"

#include <gtest/gtest.h>

#include "spatialblend/model/relation/support.h"

namespace spatialblend {
namespace test {

TEST(AvatarFacingKernelTest, FacingDirectlyAtTarget) {
  AvatarFacingKernel kernel;

  Support support;
  support.world_transform(0, 3) = 5.0;

  auto gr = kernel.Encode(Eigen::Vector3d(0, 0, 0), 0.0, support);

  ASSERT_EQ(gr.encoded_state.size(), 2);
  EXPECT_NEAR(gr.encoded_state(0), 0.0, 1e-6);
  EXPECT_NEAR(gr.encoded_state(1), 5.0, 1e-6);
}

TEST(AvatarFacingKernelTest, TargetToTheLeft) {
  AvatarFacingKernel kernel;

  Support support;
  support.world_transform(1, 3) = 5.0;

  auto gr = kernel.Encode(Eigen::Vector3d(0, 0, 0), 0.0, support);

  EXPECT_NEAR(gr.encoded_state(0), M_PI / 2.0, 1e-6);
  EXPECT_NEAR(gr.encoded_state(1), 5.0, 1e-6);
}

TEST(AvatarFacingKernelTest, FacingAway) {
  AvatarFacingKernel kernel;

  Support support;
  support.world_transform(0, 3) = 5.0;

  auto gr = kernel.Encode(Eigen::Vector3d(0, 0, 0), M_PI, support);

  EXPECT_NEAR(std::abs(gr.encoded_state(0)), M_PI, 1e-6);
}

TEST(AvatarFacingKernelTest, CompareIdentical) {
  AvatarFacingKernel kernel;

  GroundedRelation a, b;
  a.encoded_state = Eigen::VectorXd(2);
  a.encoded_state << 0.3, 5.0;
  b.encoded_state = a.encoded_state;

  EXPECT_NEAR(kernel.Compare(a, b), 0.0, 1e-9);
}

TEST(AvatarFacingKernelTest, CompareDifferentBearing) {
  AvatarFacingKernel kernel;

  GroundedRelation a, b;
  a.encoded_state = Eigen::VectorXd(2);
  a.encoded_state << 0.0, 5.0;
  b.encoded_state = Eigen::VectorXd(2);
  b.encoded_state << 1.0, 5.0;

  EXPECT_GT(kernel.Compare(a, b), 0.0);
}

TEST(AvatarFacingKernelTest, PreservedAcrossSpaces) {
  AvatarFacingKernel kernel;

  Support src_support;
  src_support.world_transform(0, 3) = 10.0;
  auto src_gr = kernel.Encode(Eigen::Vector3d(5, 0, 0), 0.0, src_support);

  Support tgt_support;
  tgt_support.world_transform(0, 3) = 3.0;
  auto tgt_gr = kernel.Encode(Eigen::Vector3d(-2, 0, 0), 0.0, tgt_support);

  EXPECT_NEAR(kernel.Compare(src_gr, tgt_gr), 0.0, 1e-9);
}

TEST(AvatarFacingKernelTest, ProximityFacingDirectly) {
  AvatarFacingKernel kernel;
  Support support;
  support.world_transform(0, 3) = 5.0;

  // Actor at origin, yaw=0, target at (5,0) → bearing_offset=0
  // proximity = (1+cos(0))/2 = 1.0
  auto gr = kernel.Encode(Eigen::Vector3d(0, 0, 0), 0.0, support);
  EXPECT_NEAR(gr.proximity, 1.0, 0.01);
}

TEST(AvatarFacingKernelTest, ProximityFacingAway) {
  AvatarFacingKernel kernel;
  Support support;
  support.world_transform(0, 3) = 5.0;

  // Actor facing opposite direction (yaw=pi) → bearing_offset=±pi
  // proximity = (1+cos(pi))/2 = 0.0
  auto gr = kernel.Encode(Eigen::Vector3d(0, 0, 0), M_PI, support);
  EXPECT_NEAR(gr.proximity, 0.0, 0.01);
}

TEST(AvatarFacingKernelTest, ProximityPerpendicular) {
  AvatarFacingKernel kernel;
  Support support;
  support.world_transform(0, 3) = 5.0;

  // Actor facing perpendicular (yaw=pi/2) → bearing_offset=pi/2
  // proximity = (1+cos(pi/2))/2 = 0.5
  auto gr = kernel.Encode(Eigen::Vector3d(0, 0, 0), M_PI / 2.0, support);
  EXPECT_NEAR(gr.proximity, 0.5, 0.01);
}

TEST(AvatarFacingKernelTest, Encode3DWithElevation) {
  AvatarFacingKernel kernel;
  kernel.SetDOF({StateType::kPosition3D, StateType::kRotation3D});

  Support support;
  support.world_transform(0, 3) = 5.0;  // target at x=5
  support.world_transform(2, 3) = 5.0;  // target at z=5 (above)

  // Actor at origin, facing +X (yaw=0)
  auto gr = kernel.Encode(Eigen::Vector3d(0, 0, 0), 0.0, support);

  // 3D: [bearing_offset, elevation_offset, distance_3d]
  ASSERT_EQ(gr.encoded_state.size(), 3);
  // elevation should be positive (looking up to reach z=5)
  EXPECT_GT(gr.encoded_state(1), 0.0);
  // 3D distance should be > 2D distance
  double dist_3d = gr.encoded_state(2);
  EXPECT_GT(dist_3d, 5.0);  // sqrt(25+25) ≈ 7.07
}

TEST(AvatarFacingKernelTest, Encode2DStillHas2Components) {
  AvatarFacingKernel kernel;
  // Default 2D DOF

  Support support;
  support.world_transform(0, 3) = 5.0;

  auto gr = kernel.Encode(Eigen::Vector3d(0, 0, 0), 0.0, support);

  ASSERT_EQ(gr.encoded_state.size(), 2);
}

}  // namespace test
}  // namespace spatialblend
