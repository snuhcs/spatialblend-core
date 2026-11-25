// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/base/common/types.h"

#include <gtest/gtest.h>

namespace spatialblend {
namespace test {

TEST(TypesTest, ElementIdDefaults) {
  element_id_t id = kInvalidElementId;
  EXPECT_EQ(id, -1);
}

TEST(TypesTest, SharedElementIdDefaults) {
  shared_element_id_t id = kInvalidSharedElementId;
  EXPECT_EQ(id, -1);
}

TEST(TypesTest, TypesAreDistinct) {
  element_id_t local_id = 42;
  shared_element_id_t shared_id = 42;
  EXPECT_EQ(local_id, 42);
  EXPECT_EQ(shared_id, 42);
}

TEST(TypesTest, SpaceIdDefaults) {
  space_id_t id = kInvalidSpaceId;
  EXPECT_EQ(id, -1);
}

}  // namespace test
}  // namespace spatialblend
