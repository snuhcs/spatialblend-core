// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/space.h"

#include <algorithm>

#include <gtest/gtest.h>

#include <Eigen/Geometry>

namespace spatialblend {
namespace {

using Status = ReceiverApproachFeasibilityStatus;

ReceiverAnchorInput Pose(const std::string& name, double x, double y) {
  ReceiverAnchorInput result;
  result.name = name;
  result.available = true;
  result.world_pose.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, 0);
  return result;
}

ReceiverApproachPolygon Polygon(uint64_t id, std::vector<Eigen::Vector2d> vertices) {
  ReceiverApproachPolygon result;
  result.nav_element_id = id;
  result.region_id = 30;
  result.vertices = std::move(vertices);
  return result;
}

class ReceiverApproachFeasibilityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    space.SetSpaceId(8);
    ASSERT_TRUE(space.AddElement(10));
    actor = space.CreateActor({SubjectAnchor("navigation_root"),
                               SubjectAnchor("pelvis")});
    ASSERT_TRUE(space.SetActorElement(actor, 10));
    ASSERT_TRUE(space.BeginReceiverInputEpoch(actor, "navigation_root", 3, 5,
                                               &epoch));
    Commit(-3, 0, 1);
  }

  void Commit(double root_x, double root_y, uint64_t sequence) {
    ASSERT_TRUE(space.ApplyReceiverInputFrame(
        actor, epoch, sequence, now, now + 10000,
        {Pose("navigation_root", root_x, root_y), Pose("pelvis", root_x, root_y)}));
    sequence_ = sequence;
    root_x_ = root_x;
    root_y_ = root_y;
  }

  ReceiverApproachContextRegistration Base() const {
    ReceiverApproachContextRegistration value;
    value.space_id = 8;
    value.session_id = 91;
    value.receiver_actor = actor;
    value.expected_receiver_input_epoch = epoch;
    value.expected_receiver_input_sequence = sequence_;
    value.geometry_revision = 11;
    value.requested_valid_until_ns = now + 5000;
    value.navigation_root.anchor_name = "navigation_root";
    value.navigation_root.pose = Pose("navigation_root", root_x_, root_y_).world_pose;
    // Explicit translated/scaled local navigation chart: local [-2,2]^2 maps
    // to the receiver Space-basis [-4,4]^2, independently of any support.
    auto nav = Polygon(100, {{-2, -1.5}, {2, -1.5}, {2, 1.5}, {-2, 1.5}});
    nav.local_to_space << 2, 0, 0, 0, 2, 0, 0, 0, 1;
    value.navigable_regions = {nav};
    value.obstacle_regions = {
        Polygon(101, {{-.4, -2}, {.4, -2}, {.4, 2}, {-.4, 2}})};
    value.clearance.footprint_radius_m = .25;
    value.clearance.required_clearance_m = .15;
    value.destination.space_id = 8;
    value.destination.region_id = 30;
    value.destination.object_id = 40;
    value.destination.affordance_id = 50;
    value.destination.support_revision = 12;
    value.destination.local_support =
        Polygon(102, {{-.5, -.5}, {.5, -.5}, {.5, .5}, {-.5, .5}});
    value.destination.safe_approach_pose = Pose("approach", 3, 0).world_pose;
    return value;
  }

  uint64_t Register(ReceiverApproachContextRegistration registration) {
    uint64_t revision = 0;
    EXPECT_EQ(ReceiverApproachContextStatus::kAvailable,
              space.RegisterReceiverApproachContext(registration, &revision));
    return revision;
  }

  int64_t now = 100;
  Space space{[this] { return now; }};
  spatialblend::entity_id_t actor = spatialblend::kInvalidEntityId;
  uint64_t epoch = 0;
  uint64_t sequence_ = 0;
  double root_x_ = 0;
  double root_y_ = 0;
};

TEST_F(ReceiverApproachFeasibilityTest,
       ObstacleRouteStartsAtObservedRootEndsAtSafeApproachAndStaysClear) {
  const uint64_t revision = Register(Base());
  ReceiverApproachFeasibilitySnapshot result;
  ASSERT_EQ(Status::kFeasible,
            space.TryComputeReceiverApproachFeasibility(actor, revision, &result));
  ASSERT_GE(result.route.size(), 3u);  // obstacle prevents a direct fallback.
  EXPECT_EQ(Eigen::Vector2d(-3, 0), result.route.front());
  EXPECT_EQ(Eigen::Vector2d(3, 0), result.route.back());
  EXPECT_EQ(revision, result.context.context_revision);
  EXPECT_EQ(11u, result.context.geometry_revision);
  EXPECT_EQ(12u, result.context.support_revision);
  EXPECT_GT(result.route_length_m, 6.0);
  // Independent scalar segment oracle: the implementation's route must not
  // merely have safe waypoints; every sampled segment keeps the 0.4m margin.
  for (size_t i = 1; i < result.route.size(); ++i) {
    for (int sample = 0; sample <= 100; ++sample) {
      const Eigen::Vector2d point = result.route[i - 1] +
                                    (sample / 100.0) *
                                        (result.route[i] - result.route[i - 1]);
      const Eigen::Vector2d nearest(std::clamp(point.x(), -.4, .4),
                                    std::clamp(point.y(), -2.0, 2.0));
      EXPECT_GT((point - nearest).norm(), .4 - 1e-6);
      EXPECT_LT(std::abs(point.y()), 3.0);
    }
  }
}

TEST_F(ReceiverApproachFeasibilityTest, TypedEndpointAndContextRejections) {
  ReceiverApproachFeasibilitySnapshot result;
  EXPECT_EQ(Status::kStale,
            space.TryComputeReceiverApproachFeasibility(actor, 99, &result));

  Commit(-4, 0, 2);  // exact permitted-region boundary, not a navigable root.
  const uint64_t boundary = Register(Base());
  EXPECT_EQ(Status::kEndpointOnBoundary,
            space.TryComputeReceiverApproachFeasibility(actor, boundary, &result));

  Commit(-3, 0, 3);
  auto narrow_registration = Base();
  narrow_registration.expected_registration_generation = 1;
  narrow_registration.clearance.footprint_radius_m = 3.1;
  const uint64_t narrow = Register(narrow_registration);
  EXPECT_EQ(Status::kInsufficientClearance,
            space.TryComputeReceiverApproachFeasibility(actor, narrow, &result));
}

TEST_F(ReceiverApproachFeasibilityTest,
       DisconnectedAndFullyBlockedGeometryNeverUseStraightLineFallback) {
  auto disconnected = Base();
  disconnected.navigable_regions = {
      Polygon(100, {{-4, -2}, {-1, -2}, {-1, 2}, {-4, 2}}),
      Polygon(103, {{1, -2}, {4, -2}, {4, 2}, {1, 2}})};
  disconnected.obstacle_regions = {Polygon(101, {{20, 20}, {21, 20}, {20, 21}})};
  const uint64_t disconnected_revision = Register(disconnected);
  ReceiverApproachFeasibilitySnapshot result;
  EXPECT_EQ(Status::kDisconnected, space.TryComputeReceiverApproachFeasibility(
                                      actor, disconnected_revision, &result));
  EXPECT_TRUE(result.route.empty());

  auto blocked = Base();
  blocked.obstacle_regions = {
      Polygon(101, {{-.4, -3}, {.4, -3}, {.4, 3}, {-.4, 3}})};
  blocked.expected_registration_generation = 1;
  const uint64_t blocked_revision = Register(blocked);
  EXPECT_EQ(Status::kBlocked,
            space.TryComputeReceiverApproachFeasibility(actor, blocked_revision, &result));
  EXPECT_TRUE(result.route.empty());
}

TEST_F(ReceiverApproachFeasibilityTest,
       InputChangeFencesThePriorFeasibilityReceiptBeforeExecution) {
  const uint64_t revision = Register(Base());
  now += 1;
  Commit(-3, 0, 2);
  ReceiverApproachFeasibilitySnapshot result;
  EXPECT_EQ(Status::kStale,
            space.TryComputeReceiverApproachFeasibility(actor, revision, &result));
  EXPECT_TRUE(result.route.empty());
}

TEST_F(ReceiverApproachFeasibilityTest,
       GeometryAndSupportReplacementFencePriorFeasibilityReceipt) {
  const uint64_t first = Register(Base());
  auto replacement = Base();
  replacement.expected_registration_generation = 1;
  replacement.geometry_revision = 13;
  replacement.destination.support_revision = 14;
  const uint64_t second = Register(replacement);
  ReceiverApproachFeasibilitySnapshot result;
  EXPECT_EQ(Status::kReplaced,
            space.TryComputeReceiverApproachFeasibility(actor, first, &result));
  EXPECT_EQ(Status::kFeasible,
            space.TryComputeReceiverApproachFeasibility(actor, second, &result));
  EXPECT_EQ(13u, result.context.geometry_revision);
  EXPECT_EQ(14u, result.context.support_revision);
}

}  // namespace
}  // namespace spatialblend
