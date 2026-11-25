// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/space_hub.h"

#include <gtest/gtest.h>

#include "spatialblend/space/space.h"

namespace spatialblend {
namespace test {

TEST(SpaceHubTest, ConnectSpaces) {
  SpaceHub hub;
  Space space_a;
  Space space_b;
  space_a.SetSpaceId(1);
  space_b.SetSpaceId(2);

  EXPECT_TRUE(hub.Connect(&space_a).ok());
  EXPECT_TRUE(hub.Connect(&space_b).ok());
  EXPECT_EQ(hub.GetConnectedSpaceCount(), 2);
}

TEST(SpaceHubTest, DisconnectSpace) {
  SpaceHub hub;
  Space space_a;
  space_a.SetSpaceId(1);
  hub.Connect(&space_a);
  hub.Disconnect(&space_a);
  EXPECT_EQ(hub.GetConnectedSpaceCount(), 0);
}

TEST(SpaceHubTest, PropagateUpdate) {
  SpaceHub hub;
  Space space_a;
  Space space_b;
  space_a.SetSpaceId(1);
  space_b.SetSpaceId(2);
  hub.Connect(&space_a);
  hub.Connect(&space_b);

  RemoteRelationUpdate update;
  update.source_space_id = 1;
  update.actor_shared_id = 42;

  RelationPayload payload;
  payload.referent_shared_id = 100;
  payload.subject_anchor_name = "root";
  payload.kernel_type = RelationKernelType::kAvatarSpacing;
  payload.grounded_relation.encoded_state = Eigen::VectorXd(3);
  payload.grounded_relation.encoded_state << 1.0, 2.0, 0.0;
  update.payloads.push_back(std::move(payload));

  hub.Propagate(update);

  auto& received = space_b.GetReceivedUpdates();
  ASSERT_EQ(received.size(), 1);
  EXPECT_EQ(received[0].actor_shared_id, 42);
  EXPECT_EQ(received[0].payloads.size(), 1);

  EXPECT_TRUE(space_a.GetReceivedUpdates().empty());
}

TEST(SpaceHubTest, AllocateSpaceIdIsFreshAndSkipsConnectedSpaces) {
  SpaceHub hub;
  Space held;
  held.SetSpaceId(2);  // numbered by something other than this hub
  ASSERT_TRUE(hub.Connect(&held).ok());
  EXPECT_TRUE(hub.HasSpaceId(2));
  EXPECT_FALSE(hub.HasSpaceId(1));

  const space_id_t first = hub.AllocateSpaceId();
  const space_id_t second = hub.AllocateSpaceId();
  const space_id_t third = hub.AllocateSpaceId();
  EXPECT_EQ(first, 1);
  EXPECT_EQ(second, 3) << "2 is held by a connected space";
  EXPECT_EQ(third, 4);
  EXPECT_NE(first, held.GetSpaceId());
}

}  // namespace test
}  // namespace spatialblend
