// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/realization/remote_update.h"

#include <gtest/gtest.h>

namespace spatialblend {
namespace test {

TEST(RelationPayloadTest, DefaultConstruction) {
  RelationPayload payload;
  EXPECT_EQ(payload.referent_shared_id, kInvalidSharedElementId);
  EXPECT_TRUE(payload.subject_anchor_name.empty());
}

TEST(RelationPayloadTest, SetFields) {
  RelationPayload payload;
  payload.referent_shared_id = 100;
  payload.subject_anchor_name = "chest";
  payload.kernel_type = RelationKernelType::kAvatarFacing;
  payload.grounded_relation.encoded_state = Eigen::VectorXd(2);
  payload.grounded_relation.encoded_state << 0.5, 3.0;

  EXPECT_EQ(payload.referent_shared_id, 100);
  EXPECT_EQ(payload.subject_anchor_name, "chest");
  EXPECT_EQ(payload.kernel_type, RelationKernelType::kAvatarFacing);
  EXPECT_EQ(payload.grounded_relation.encoded_state.size(), 2);
}

TEST(RemoteRelationUpdateTest, DefaultConstruction) {
  RemoteRelationUpdate update;
  EXPECT_EQ(update.source_space_id, kInvalidSpaceId);
  EXPECT_EQ(update.actor_shared_id, kInvalidSharedElementId);
  EXPECT_TRUE(update.payloads.empty());
}

TEST(RemoteRelationUpdateTest, AddPayloads) {
  RemoteRelationUpdate update;
  update.source_space_id = 1;
  update.actor_shared_id = 42;

  RelationPayload p;
  p.referent_shared_id = 100;
  p.subject_anchor_name = "root";
  p.kernel_type = RelationKernelType::kAvatarSpacing;
  update.payloads.push_back(std::move(p));

  EXPECT_EQ(update.payloads.size(), 1);
  EXPECT_EQ(update.payloads[0].referent_shared_id, 100);
}

}  // namespace test
}  // namespace spatialblend
