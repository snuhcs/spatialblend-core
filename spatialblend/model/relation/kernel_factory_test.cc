// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/kernel_factory.h"

#include <gtest/gtest.h>

namespace spatialblend {
namespace test {

TEST(KernelFactoryTest, CreateFacing) {
  auto components = KernelFactory::Create(RelationKernelType::kAvatarFacing);
  ASSERT_NE(components.kernel, nullptr);
  ASSERT_NE(components.grounding_rule, nullptr);
  EXPECT_EQ(components.kernel->GetType(), RelationKernelType::kAvatarFacing);
}

TEST(KernelFactoryTest, CreateSpacing) {
  auto components = KernelFactory::Create(RelationKernelType::kAvatarSpacing);
  ASSERT_NE(components.kernel, nullptr);
  ASSERT_NE(components.grounding_rule, nullptr);
  EXPECT_EQ(components.kernel->GetType(), RelationKernelType::kAvatarSpacing);
}

TEST(KernelFactoryTest, CreateAffordancePose) {
  auto components = KernelFactory::Create(RelationKernelType::kAvatarAffordancePose);
  ASSERT_NE(components.kernel, nullptr);
  ASSERT_NE(components.grounding_rule, nullptr);
  EXPECT_EQ(components.kernel->GetType(), RelationKernelType::kAvatarAffordancePose);
}

TEST(KernelFactoryTest, CreateWithDOF3D) {
  auto components = KernelFactory::Create(
      RelationKernelType::kAvatarSpacing,
      {StateType::kPosition3D, StateType::kRotation3D});
  ASSERT_NE(components.kernel, nullptr);
  EXPECT_TRUE(components.kernel->Is3DPosition());
  auto dynamic = components.kernel->GetDynamicStates();
  ASSERT_EQ(dynamic.size(), 1);
  EXPECT_EQ(dynamic[0], StateType::kPosition3D);
}

TEST(KernelFactoryTest, DefaultDOFIs2D) {
  auto components = KernelFactory::Create(RelationKernelType::kAvatarFacing);
  ASSERT_NE(components.kernel, nullptr);
  EXPECT_FALSE(components.kernel->Is3DPosition());
  auto dynamic = components.kernel->GetDynamicStates();
  ASSERT_EQ(dynamic.size(), 1);
  EXPECT_EQ(dynamic[0], StateType::kRotationYaw);
}

}  // namespace test
}  // namespace spatialblend
