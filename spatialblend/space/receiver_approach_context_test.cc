// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/space.h"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <functional>

namespace spatialblend {
namespace {

using Status = ReceiverApproachContextStatus;

ReceiverAnchorInput Pose(const std::string& name, double x, double y,
                        double yaw = 0) {
  ReceiverAnchorInput result;
  result.name = name;
  result.available = true;
  result.world_pose.block<3, 3>(0, 0) =
      Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  result.world_pose.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, 0);
  return result;
}

ReceiverApproachPolygon Polygon(uint64_t id,
                                std::vector<Eigen::Vector2d> vertices,
                                uint64_t region_id = 30) {
  ReceiverApproachPolygon polygon;
  polygon.nav_element_id = id;
  polygon.region_id = region_id;
  polygon.vertices = std::move(vertices);
  return polygon;
}

class ReceiverApproachContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    space.SetSpaceId(8);
    ASSERT_TRUE(space.AddElement(10));
    actor = space.CreateActor({SubjectAnchor("navigation_root"),
                               SubjectAnchor("pelvis")});
    ASSERT_TRUE(space.SetActorElement(actor, 10));
    ASSERT_TRUE(
        space.BeginReceiverInputEpoch(actor, "navigation_root", 3, 5, &epoch));
    Commit(1);
  }

  void Commit(uint64_t sequence) {
    ASSERT_TRUE(space.ApplyReceiverInputFrame(
        actor, epoch, sequence, now, now + 1000,
        {Pose("navigation_root", 2, 3, 0.4), Pose("pelvis", 2, 3)}));
    current_sequence = sequence;
  }

  ReceiverApproachContextRegistration Complete() const {
    ReceiverApproachContextRegistration value;
    value.space_id = 8;
    value.session_id = 91;
    value.receiver_actor = actor;
    value.expected_receiver_input_epoch = epoch;
    value.expected_receiver_input_sequence = current_sequence;
    value.geometry_revision = 11;
    value.requested_valid_until_ns = now + 2000;  // must cap at input expiry.
    value.navigation_root.anchor_name = "navigation_root";
    value.navigation_root.pose = Pose("navigation_root", 2, 3, 0.4).world_pose;
    auto nav = Polygon(100, {{-1, -1}, {2, -1}, {2, 1}, {-1, 1}});
    nav.local_to_space << 2, 0, 4, 0, 2, -3, 0, 0, 1;
    value.navigable_regions = {nav};
    value.obstacle_regions = {Polygon(101, {{20, 20}, {21, 20}, {20, 21}})};
    value.clearance.footprint_radius_m = .3;
    value.clearance.required_clearance_m = .1;
    value.destination.space_id = 8;
    value.destination.region_id = 30;
    value.destination.object_id = 40;
    value.destination.affordance_id = 50;
    value.destination.support_revision = 12;
    value.destination.local_support =
        Polygon(102, {{-.5, -.5}, {.5, -.5}, {.5, .5}, {-.5, .5}});
    // (6, -1) lies inside the translated/scaled navigation polygon.
    value.destination.safe_approach_pose = Pose("safe", 6, -1, 0.2).world_pose;
    return value;
  }

  int64_t now = 100;
  Space space{[this] {
    auto callback = std::move(on_clock_read);
    on_clock_read = nullptr;
    if (callback) callback();
    return now;
  }};
  spatialblend::entity_id_t actor = spatialblend::kInvalidEntityId;
  uint64_t epoch = 0;
  uint64_t current_sequence = 0;
  std::function<void()> on_clock_read;
};

TEST_F(ReceiverApproachContextTest,
       CompleteRegistrationCopiesIndependentNativeInputsAndCapsValidity) {
  auto registration = Complete();
  uint64_t revision = 0;
  EXPECT_EQ(Status::kAvailable,
            space.RegisterReceiverApproachContext(registration, &revision));
  ASSERT_NE(0u, revision);

  ReceiverApproachContext copied;
  EXPECT_EQ(Status::kAvailable,
            space.TryCopyReceiverApproachContext(actor, revision, &copied));
  EXPECT_EQ(revision, copied.context_revision);
  EXPECT_EQ(1u, copied.registration_generation);
  EXPECT_EQ(8, copied.space_id);
  EXPECT_EQ(91u, copied.session_id);
  EXPECT_EQ(actor, copied.receiver_actor);
  EXPECT_EQ(epoch, copied.receiver_input.epoch);
  EXPECT_EQ(1u, copied.receiver_input.sequence);
  EXPECT_EQ(11u, copied.geometry_revision);
  EXPECT_EQ(12u, copied.support_revision);
  EXPECT_EQ(now + 1000, copied.valid_until_ns);
  EXPECT_EQ(1u, copied.navigable_regions.size());
  EXPECT_EQ(2.0, copied.navigable_regions[0].local_to_space(0, 0));
  EXPECT_EQ(4.0, copied.navigable_regions[0].local_to_space(0, 2));
  EXPECT_EQ(-3.0, copied.navigable_regions[0].local_to_space(1, 2));
  registration.navigable_regions[0].vertices[0] = {99, 99};
  EXPECT_NE(registration.navigable_regions[0].vertices[0],
            copied.navigable_regions[0].vertices[0]);
}

TEST_F(ReceiverApproachContextTest, TypedNegativeRegistrationOutcomes) {
  auto registration = Complete();
  registration.navigation_root.anchor_name.clear();
  EXPECT_EQ(Status::kMissingNavigationRoot,
            space.RegisterReceiverApproachContext(registration));
  registration = Complete();
  registration.navigation_root.kind = ReceiverApproachRootKind::kPelvisOrBodyOnly;
  EXPECT_EQ(Status::kBodyOnlyInput, space.RegisterReceiverApproachContext(registration));
  registration = Complete();
  // A matching pelvis pose still cannot be relabeled as the independently
  // declared navigation root for this receiver-input epoch.
  registration.navigation_root.anchor_name = "pelvis";
  registration.navigation_root.pose = Pose("pelvis", 2, 3).world_pose;
  EXPECT_EQ(Status::kMissingNavigationRoot,
            space.RegisterReceiverApproachContext(registration));
  registration = Complete();
  registration.navigable_regions.clear();
  EXPECT_EQ(Status::kMissingGeometry, space.RegisterReceiverApproachContext(registration));
  registration = Complete();
  registration.obstacle_regions[0].vertices = {{0, 0}, {1, 1}, {0, 1}, {1, 0}};
  EXPECT_EQ(Status::kInvalidGeometry, space.RegisterReceiverApproachContext(registration));
  registration = Complete();
  registration.clearance.footprint_radius_m = 0;
  EXPECT_EQ(Status::kMissingPolicy, space.RegisterReceiverApproachContext(registration));
  registration = Complete();
  registration.clearance.required_clearance_m = -1;
  EXPECT_EQ(Status::kMissingPolicy, space.RegisterReceiverApproachContext(registration));
  registration = Complete();
  registration.destination.safe_approach_pose = Pose("outside", 99, 99).world_pose;
  EXPECT_EQ(Status::kMissingDestination,
            space.RegisterReceiverApproachContext(registration));
  registration = Complete();
  registration.destination.space_id = 9;
  EXPECT_EQ(Status::kMissingDestination,
            space.RegisterReceiverApproachContext(registration));
  registration = Complete();
  registration.space_id = 9;
  EXPECT_EQ(Status::kStale, space.RegisterReceiverApproachContext(registration));
  registration = Complete();
  registration.obstacle_regions[0].region_id = 31;
  EXPECT_EQ(Status::kInvalidGeometry,
            space.RegisterReceiverApproachContext(registration));
  registration = Complete();
  registration.destination.local_support.vertices.clear();
  EXPECT_EQ(Status::kMissingDestination,
            space.RegisterReceiverApproachContext(registration));
}

TEST_F(ReceiverApproachContextTest,
       InputSequenceAndEpochInvalidateWithoutReplacingCurrentRegistration) {
  auto registration = Complete();
  uint64_t revision = 0;
  ASSERT_EQ(Status::kAvailable,
            space.RegisterReceiverApproachContext(registration, &revision));
  now += 1;
  Commit(2);
  ReceiverApproachContext copied;
  EXPECT_EQ(Status::kStale,
            space.TryCopyReceiverApproachContext(actor, revision, &copied));

  uint64_t next_epoch = 0;
  ASSERT_TRUE(space.BeginReceiverInputEpoch(actor, "navigation_root", 4, 6,
                                            &next_epoch));
  epoch = next_epoch;
  now += 1;
  Commit(1);
  EXPECT_EQ(Status::kStale,
            space.TryCopyReceiverApproachContext(actor, revision, &copied));
}

TEST_F(ReceiverApproachContextTest,
       InputExpiryReplacementAndGenerationFenceDoNotReviveOldContext) {
  auto registration = Complete();
  uint64_t first = 0;
  ASSERT_EQ(Status::kAvailable,
            space.RegisterReceiverApproachContext(registration, &first));
  now += 1000;
  ReceiverApproachContext copied;
  EXPECT_EQ(Status::kExpired,
            space.TryCopyReceiverApproachContext(actor, first, &copied));

  // A fresh input cannot revive first; it can only support an explicit,
  // generation-fenced replacement.
  now += 1;
  Commit(2);
  EXPECT_EQ(Status::kExpired,
            space.TryCopyReceiverApproachContext(actor, first, &copied));
  registration = Complete();
  registration.expected_registration_generation = 1;
  registration.geometry_revision = 13;
  registration.destination.support_revision = 14;
  uint64_t second = 0;
  ASSERT_EQ(Status::kAvailable,
            space.RegisterReceiverApproachContext(registration, &second));
  EXPECT_GT(second, first);
  EXPECT_EQ(Status::kReplaced,
            space.TryCopyReceiverApproachContext(actor, first, &copied));
  EXPECT_EQ(Status::kReplaced,
            space.RegisterReceiverApproachContext(registration));
  EXPECT_EQ(Status::kAvailable,
            space.TryCopyReceiverApproachContext(actor, second, &copied));
  EXPECT_EQ(13u, copied.geometry_revision);
  EXPECT_EQ(14u, copied.support_revision);
}

TEST_F(ReceiverApproachContextTest, TeardownAndInputReplacementCannotReturnCopy) {
  auto registration = Complete();
  uint64_t revision = 0;
  ASSERT_EQ(Status::kAvailable,
            space.RegisterReceiverApproachContext(registration, &revision));
  ASSERT_TRUE(space.DestroyActor(actor));
  ReceiverApproachContext copied;
  EXPECT_EQ(Status::kStale,
            space.TryCopyReceiverApproachContext(actor, revision, &copied));
  EXPECT_EQ(0u, copied.context_revision);
}

TEST_F(ReceiverApproachContextTest,
       SessionGeometryAndSupportReplacementFenceThePriorContext) {
  auto registration = Complete();
  uint64_t first = 0;
  ASSERT_EQ(Status::kAvailable,
            space.RegisterReceiverApproachContext(registration, &first));
  registration.expected_registration_generation = 1;
  registration.session_id = 92;
  registration.geometry_revision = 13;
  registration.destination.support_revision = 14;
  uint64_t second = 0;
  ASSERT_EQ(Status::kAvailable,
            space.RegisterReceiverApproachContext(registration, &second));
  ReceiverApproachContext copied;
  EXPECT_EQ(Status::kReplaced,
            space.TryCopyReceiverApproachContext(actor, first, &copied));
  EXPECT_EQ(Status::kAvailable,
            space.TryCopyReceiverApproachContext(actor, second, &copied));
  EXPECT_EQ(92u, copied.session_id);
  EXPECT_EQ(13u, copied.geometry_revision);
  EXPECT_EQ(14u, copied.support_revision);
}

TEST_F(ReceiverApproachContextTest,
       ReentrantReadReportsBusyWithoutRetiringTheLiveContext) {
  auto registration = Complete();
  uint64_t revision = 0;
  ASSERT_EQ(Status::kAvailable,
            space.RegisterReceiverApproachContext(registration, &revision));
  Status nested_status = Status::kAvailable;
  ReceiverApproachContext nested;
  on_clock_read = [&] {
    nested_status =
        space.TryCopyReceiverApproachContext(actor, revision, &nested);
  };
  ReceiverApproachContext copied;
  EXPECT_EQ(Status::kAvailable,
            space.TryCopyReceiverApproachContext(actor, revision, &copied));
  EXPECT_EQ(Status::kBusy, nested_status);
  EXPECT_EQ(0u, nested.context_revision);
  EXPECT_EQ(Status::kAvailable,
            space.TryCopyReceiverApproachContext(actor, revision, &copied));
}

}  // namespace
}  // namespace spatialblend
