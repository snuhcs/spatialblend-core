// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/grounded_relation.h"

#include <gtest/gtest.h>

namespace spatialblend {
namespace test {

TEST(GroundedRelationTest, DefaultConstruction) {
  GroundedRelation gr;
  EXPECT_EQ(gr.encoded_state.size(), 0);
}

TEST(GroundedRelationTest, SetEncodedState) {
  GroundedRelation gr;
  gr.encoded_state = Eigen::VectorXd(3);
  gr.encoded_state << 1.0, 2.0, 0.5;
  EXPECT_EQ(gr.encoded_state.size(), 3);
  EXPECT_DOUBLE_EQ(gr.encoded_state(0), 1.0);
}

TEST(GroundedRelationTest, DistanceToIdentical) {
  GroundedRelation a;
  a.encoded_state = Eigen::VectorXd(2);
  a.encoded_state << 3.0, 4.0;

  GroundedRelation b;
  b.encoded_state = a.encoded_state;

  EXPECT_DOUBLE_EQ(a.DistanceTo(b), 0.0);
}

TEST(GroundedRelationTest, DistanceToDifferent) {
  GroundedRelation a;
  a.encoded_state = Eigen::VectorXd(2);
  a.encoded_state << 0.0, 0.0;

  GroundedRelation b;
  b.encoded_state = Eigen::VectorXd(2);
  b.encoded_state << 3.0, 4.0;

  EXPECT_DOUBLE_EQ(a.DistanceTo(b), 5.0);
}

TEST(GroundedRelationTest, DistanceToSizeMismatch) {
  GroundedRelation a;
  a.encoded_state = Eigen::VectorXd(2);
  a.encoded_state << 1.0, 2.0;

  GroundedRelation b;
  b.encoded_state = Eigen::VectorXd(3);
  b.encoded_state << 1.0, 2.0, 3.0;

  EXPECT_GT(a.DistanceTo(b), 1e8);
}

TEST(GroundedRelationTest, ProximityDefaultsToZero) {
  GroundedRelation gr;
  EXPECT_DOUBLE_EQ(gr.proximity, 0.0);
}

TEST(GroundedRelationTest, ProximityCanBeSet) {
  GroundedRelation gr;
  gr.proximity = 0.75;
  EXPECT_DOUBLE_EQ(gr.proximity, 0.75);
}

TEST(GroundedRelationTest, DistanceToIgnoresProximity) {
  // DistanceTo only compares encoded_state, not proximity
  GroundedRelation a, b;
  a.encoded_state = Eigen::VectorXd(2);
  a.encoded_state << 1.0, 2.0;
  a.proximity = 0.9;

  b.encoded_state = Eigen::VectorXd(2);
  b.encoded_state << 1.0, 2.0;
  b.proximity = 0.1;

  // Same encoded state → distance 0, even though proximities differ
  EXPECT_DOUBLE_EQ(a.DistanceTo(b), 0.0);
}

}  // namespace test
}  // namespace spatialblend
