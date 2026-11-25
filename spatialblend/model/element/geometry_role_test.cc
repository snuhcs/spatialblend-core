// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/geometry_role.h"

#include <gtest/gtest.h>

namespace spatialblend {
namespace test {

TEST(GeometryRoleTest, EnumValues) {
  EXPECT_NE(GeometryRole::kNavigable, GeometryRole::kContainment);
  EXPECT_NE(GeometryRole::kSurface, GeometryRole::kInteractionRegion);
}

TEST(GeometryRoleTest, ToString) {
  EXPECT_STREQ(GeometryRoleToString(GeometryRole::kUnspecified),
               "Unspecified");
  EXPECT_STREQ(GeometryRoleToString(GeometryRole::kNavigable), "Navigable");
  EXPECT_STREQ(GeometryRoleToString(GeometryRole::kContainment),
               "Containment");
  EXPECT_STREQ(GeometryRoleToString(GeometryRole::kObstacle), "Obstacle");
  EXPECT_STREQ(GeometryRoleToString(GeometryRole::kSurface), "Surface");
  EXPECT_STREQ(GeometryRoleToString(GeometryRole::kInteractionRegion),
               "InteractionRegion");
}

TEST(GeometryRoleTest, ContainmentRoles) {
  EXPECT_TRUE(IsContainmentRole(GeometryRole::kNavigable));
  EXPECT_TRUE(IsContainmentRole(GeometryRole::kContainment));
  EXPECT_FALSE(IsContainmentRole(GeometryRole::kInteractionRegion));
  EXPECT_FALSE(IsContainmentRole(GeometryRole::kSurface));
}

TEST(GeometryRoleTest, InteractionRoles) {
  EXPECT_TRUE(IsInteractionRole(GeometryRole::kInteractionRegion));
  EXPECT_FALSE(IsInteractionRole(GeometryRole::kSurface));
  EXPECT_FALSE(IsInteractionRole(GeometryRole::kContainment));
  EXPECT_FALSE(IsInteractionRole(GeometryRole::kNavigable));
}

}  // namespace test
}  // namespace spatialblend
