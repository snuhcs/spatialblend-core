// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/support.h"

#include <gtest/gtest.h>

namespace spatialblend {
namespace test {

TEST(SupportTest, DefaultConstruction) {
  Support support;
  EXPECT_TRUE(support.world_transform.isIdentity());
  EXPECT_EQ(support.boundary, nullptr);
  EXPECT_EQ(support.referent_element_id, kInvalidElementId);
}

TEST(SupportTest, ConstructWithValues) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform(0, 3) = 5.0;
  transform(1, 3) = 8.0;

  Support support;
  support.world_transform = transform;
  support.referent_element_id = 42;

  EXPECT_DOUBLE_EQ(support.world_transform(0, 3), 5.0);
  EXPECT_EQ(support.referent_element_id, 42);
}

TEST(SupportTest, GetPosition2D) {
  Support support;
  support.world_transform(0, 3) = 3.0;
  support.world_transform(1, 3) = 7.0;

  auto pos = support.GetPosition2D();
  EXPECT_DOUBLE_EQ(pos.x(), 3.0);
  EXPECT_DOUBLE_EQ(pos.y(), 7.0);
}

TEST(SupportTest, GetForwardDirection2D) {
  Support support;
  auto dir = support.GetForwardDirection2D();
  EXPECT_NEAR(dir.x(), 0.0, 1e-9);
  EXPECT_NEAR(dir.y(), 1.0, 1e-9);
}

TEST(SupportTest, GetForwardDirection2DRotated) {
  Support support;
  double angle = M_PI / 2.0;
  support.world_transform(0, 0) = std::cos(angle);
  support.world_transform(0, 1) = -std::sin(angle);
  support.world_transform(1, 0) = std::sin(angle);
  support.world_transform(1, 1) = std::cos(angle);

  auto dir = support.GetForwardDirection2D();
  EXPECT_NEAR(dir.x(), -1.0, 1e-9);
  EXPECT_NEAR(dir.y(), 0.0, 1e-9);
}

TEST(SupportTest, GetYaw) {
  Support support;
  EXPECT_NEAR(support.GetYaw(), 0.0, 1e-9);

  double angle = M_PI / 2.0;
  support.world_transform(0, 0) = std::cos(angle);
  support.world_transform(0, 1) = -std::sin(angle);
  support.world_transform(1, 0) = std::sin(angle);
  support.world_transform(1, 1) = std::cos(angle);
  EXPECT_NEAR(support.GetYaw(), M_PI / 2.0, 1e-9);
}

}  // namespace test
}  // namespace spatialblend
