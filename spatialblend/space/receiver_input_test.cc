// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/realization/receiver_input.h"

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <atomic>
#include <limits>
#include <thread>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace {

ReceiverAnchorInput Pose(std::string name, double x = 0, double y = 0,
                         double z = 0, double yaw = 0, bool available = true) {
  ReceiverAnchorInput value;
  value.name = std::move(name);
  value.available = available;
  value.world_pose.block<3, 3>(0, 0) =
      Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  value.world_pose.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z);
  return value;
}

TEST(ReceiverInputStateTest, AtomicValidationAndDetachedOwnership) {
  ReceiverInputState input;
  uint64_t epoch = 0;
  ASSERT_TRUE(input.Begin("root", {"root", "head"}, 7, 9, &epoch));
  std::vector<ReceiverAnchorInput> frame{Pose("root", 1), Pose("head", 2)};
  ASSERT_TRUE(input.Commit(epoch, 1, 100, 300, frame, 150));
  const auto before = input.Stamp(150);
  EXPECT_TRUE(before.known);
  frame[0].world_pose(0, 3) = 99;
  Eigen::Matrix4d copied;
  ASSERT_TRUE(input.CopyPose("root", &copied));
  EXPECT_EQ(copied(0, 3), 1);
  copied(0, 3) = 98;
  ASSERT_TRUE(input.CopyPose("root", &copied));
  EXPECT_EQ(copied(0, 3), 1);

  EXPECT_FALSE(input.Commit(epoch, 1, 110, 300, frame, 150));
  EXPECT_FALSE(input.Commit(epoch + 1, 2, 110, 300, frame, 150));
  EXPECT_FALSE(input.Commit(epoch, 2, 99, 300, frame, 150));
  EXPECT_FALSE(input.Commit(epoch, 2, 151, 300, frame, 150));
  EXPECT_FALSE(input.Commit(epoch, 2, 110, 150, frame, 150));
  EXPECT_FALSE(input.Commit(epoch, 2, 110, 111 + ReceiverInputState::kMaxAgeNs,
                            frame, 150));
  EXPECT_FALSE(input.Commit(epoch, 2, 110, 300, {frame[0]}, 150));
  EXPECT_FALSE(input.Commit(epoch, 2, 110, 300, {frame[0], frame[0]}, 150));
  frame[1].name = "unknown";
  EXPECT_FALSE(input.Commit(epoch, 2, 110, 300, frame, 150));
  frame[1].name = "head";
  for (double bad : {2.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
    frame[1].world_pose(0, 0) = bad;
    EXPECT_FALSE(input.Commit(epoch, 2, 110, 300, frame, 150));
  }
  EXPECT_EQ(input.Stamp(150).revision, before.revision);
  EXPECT_EQ(input.Stamp(150).sequence, 1u);
}

TEST(ReceiverInputStateTest, ExpiryLossAndRecoveryUseNoSyntheticClock) {
  ReceiverInputState input;
  uint64_t epoch;
  ASSERT_TRUE(input.Begin("body", {"body", "hand"}, 1, 2, &epoch));
  auto root = Pose("body");
  auto hand = Pose("hand");
  ASSERT_TRUE(input.Commit(epoch, 1, 100, 200, {root, hand}, 150));
  const auto stamp = input.Stamp(199);
  EXPECT_TRUE(stamp.known);
  EXPECT_FALSE(input.Stamp(200).known);
  EXPECT_EQ(input.Stamp(200).revision, stamp.revision);
  EXPECT_FALSE(input.Stamp(99).known);
  hand.available = false;
  ASSERT_TRUE(input.Commit(epoch, 2, 201, 300, {root, hand}, 210));
  EXPECT_TRUE(input.Stamp(210).known);
  Eigen::Matrix4d pose;
  EXPECT_FALSE(input.CopyPose("hand", &pose));
  root.available = false;
  ASSERT_TRUE(input.Commit(epoch, 3, 220, 300, {root, hand}, 230));
  EXPECT_FALSE(input.Stamp(230).known);
  root.available = true;
  ASSERT_TRUE(input.Commit(epoch, 4, 240, 300, {root, hand}, 250));
  EXPECT_TRUE(input.Stamp(250).known);
}

TEST(ReceiverInputStateTest, ClearRetainsFenceAndStructuralResetRequiresEpoch) {
  ReceiverInputState input;
  uint64_t epoch;
  ASSERT_TRUE(input.Begin("root", {"root"}, 1, 2, &epoch));
  ASSERT_TRUE(input.Commit(epoch, 5, 100, 300, {Pose("root")}, 150));
  EXPECT_FALSE(input.Clear(epoch + 1));
  ASSERT_TRUE(input.Clear(epoch));
  const auto cleared = input.Stamp(150);
  EXPECT_FALSE(cleared.known);
  EXPECT_TRUE(input.Clear(epoch));
  EXPECT_EQ(input.Stamp(150).revision, cleared.revision);
  EXPECT_FALSE(input.Commit(epoch, 5, 100, 300, {Pose("root")}, 150));
  EXPECT_TRUE(input.Commit(epoch, 6, 100, 300, {Pose("root")}, 150));
  input.Invalidate();
  EXPECT_FALSE(input.Commit(epoch, 7, 100, 300, {Pose("root")}, 150));
  uint64_t next;
  ASSERT_TRUE(input.Begin("root", {"root"}, 3, 4, &next));
  EXPECT_GT(next, epoch);
  ASSERT_TRUE(input.Commit(next, 1, 10, 300, {Pose("root")}, 150));
  EXPECT_EQ(input.Stamp(150).rig_revision, 3u);
  EXPECT_EQ(input.Stamp(150).calibration_revision, 4u);
}

TEST(ReceiverInputStateTest, RejectsAmbiguousLayoutAndUndefinedRootHeading) {
  ReceiverInputState input;
  uint64_t epoch;
  EXPECT_FALSE(input.Begin("root", {"head"}, 1, 2, &epoch));
  EXPECT_FALSE(input.Begin("root", {"root", "root"}, 1, 2, &epoch));
  EXPECT_FALSE(
      input.Begin("root", {"root", std::string(129, 'x')}, 1, 2, &epoch));
  EXPECT_FALSE(input.Begin("root", {"root"}, 0, 2, &epoch));
  ASSERT_TRUE(input.Begin("root", {"root", "head"}, 1, 2, &epoch));
  auto vertical = Pose("root");
  vertical.world_pose.block<3, 3>(0, 0) =
      Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()).toRotationMatrix();
  EXPECT_FALSE(input.Commit(epoch, 1, 100, 300, {vertical, Pose("head")}, 150));
  vertical.name = "head";
  EXPECT_TRUE(input.Commit(epoch, 1, 100, 300, {Pose("root"), vertical}, 150));
  double yaw;
  EXPECT_FALSE(ReceiverPoseYaw(vertical.world_pose, &yaw));
}

class ReceiverSpaceInputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    space.SetSpaceId(2);
    space.SetRichRealizationEnabled(true);
    ASSERT_TRUE(space.AddElement(10));
    ASSERT_TRUE(space.SetElementSharedId(10, 500));
    ASSERT_TRUE(space.AddElement(20));
    ASSERT_TRUE(space.SetElementSharedId(20, 42));
    auto square = std::make_shared<Polygon2D>(
        std::vector<Eigen::Vector2d>{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}});
    auto definition = std::make_shared<NavigableSpatialDefinition>(square);
    SubRegion seat;
    seat.name = "seat";
    seat.boundary = square;
    seat.local_transform(2, 3) = 0.4;
    definition->AddSubRegion(seat);
    ASSERT_TRUE(space.SetElementSpatialDefinition(20, definition));
    actor = space.CreateActor(
        {SubjectAnchor("root"), SubjectAnchor("head"), SubjectAnchor("hand")});
    ASSERT_TRUE(space.SetActorElement(actor, 10));
    RelationPayload support;
    support.referent_shared_id = 42;
    support.subject_anchor_name = "hand";
    support.kernel_type = RelationKernelType::kAvatarAffordancePose;
    support.grounded_relation.proximity = 1;
    // A source dwelling in contact: v1's count is 1 (SourceUpdateCount(0, 0)), a request. Without a count Progressive
    // computes no weight.
    support.update_count = SourceUpdateCount(0.0, 0.0);
    support.grounded_relation.encoded_state = Eigen::VectorXd::Zero(3);
    support.support.has_contact = true;
    support.support.sub_region_name = "seat";
    support.support.contact_normalized.strategy =
        NormalizedPolygonPosition::Strategy::kRadial;
    support.support.contact_normalized.angle = 0;
    support.support.contact_normalized.radial_ratio = 0;
    RelationPayload facing;
    facing.referent_shared_id = 42;
    facing.subject_anchor_name = "head";
    facing.kernel_type = RelationKernelType::kAvatarFacing;
    facing.update_count = 1.0;  // aligned: a valid Facing request (v1 Angular1D)
    facing.grounded_relation.proximity = 1;
    facing.grounded_relation.encoded_state = Eigen::VectorXd::Zero(1);
    update.actor_shared_id = 500;
    update.source_space_id = 1;
    update.payloads = {support, facing};
    frame = {Pose("root", 2, 3, 0.8, 0.4), Pose("head", 2, 0, 1.7, 0.7),
             Pose("hand", 2, 1, 1.2)};
  }
  void Begin() {
    ASSERT_TRUE(space.BeginReceiverInputEpoch(actor, "root", 7, 9, &epoch));
  }
  // The Actor transform, read without borrowing the actor (a borrowed view
  // would taint scene certification). Never written for an input-owned actor.
  Eigen::Vector2d DisplayPosition() {
    Eigen::Vector2d position = Eigen::Vector2d::Constant(-1e9);
    double yaw = 0.0;
    space.CopyActorPose(actor, &position, &yaw);
    return position;
  }
  bool Commit(uint64_t sequence = 1) {
    const auto now = clock_now > 0 ? clock_now : ReceiverInputNowNs();
    return space.ApplyReceiverInputFrame(actor, epoch, sequence, now,
                                         now + ReceiverInputState::kMaxAgeNs,
                                         frame);
  }
  ActorRealizationSnapshot Solve() {
    space.OnRemoteUpdate(update);
    ActorRealizationSnapshot result;
    EXPECT_TRUE(space.CopyActorRealizationSnapshot(actor, &result));
    return result;
  }
  ActorReceiverInputValidity Validity(
      const ActorRealizationSnapshot& snapshot) {
    ActorReceiverInputValidity result;
    EXPECT_TRUE(
        space.GetActorReceiverInputValidity(actor, snapshot.revision, &result));
    return result;
  }
  const RealizationChannel* Channel(const ActorRealizationSnapshot& snapshot,
                                    const std::string& name) {
    for (const auto& channel : snapshot.channels)
      if (channel.anchor_name == name) return &channel;
    return nullptr;
  }
  int64_t clock_now = 0;
  int clock_reads = 0;
  int expire_after_read = -1;
  std::function<void()> on_clock_read;
  Space space{[this] {
    auto callback = std::move(on_clock_read);
    on_clock_read = nullptr;  // A moved-from std::function need not be empty.
    if (callback) callback();
    ++clock_reads;
    if (expire_after_read >= 0 && clock_reads > expire_after_read)
      return clock_now + ReceiverInputState::kMaxAgeNs;
    return clock_now > 0 ? clock_now : ReceiverInputNowNs();
  }};
  spatialblend::entity_id_t actor;
  uint64_t epoch = 0;
  RemoteRelationUpdate update;
  std::vector<ReceiverAnchorInput> frame;
};

TEST_F(ReceiverSpaceInputTest, CommitDoesNotWriteDisplayOrPublishObservation) {
  Begin();
  ASSERT_TRUE(Commit());
  struct {
    Eigen::Vector2d position;
  } display;  // the Actor transform: never written for an input-owned actor
  display.position = DisplayPosition();
  EXPECT_EQ(display.position, Eigen::Vector2d::Zero());
  Space::ElementState element;
  ASSERT_TRUE(space.CopyElementState(10, &element));
  EXPECT_EQ(element.world_transform, Eigen::Matrix4d::Identity());
  EXPECT_EQ(space.GetSpaceClock(), 0);
  EXPECT_EQ(space.GetReceivedUpdateCount(), 0u);
  EXPECT_EQ(space.GetLastEmittedSequence(actor), 0);
  uint64_t source_epoch;
  int64_t source_sequence;
  double source_time;
  bool has_sample;
  EXPECT_FALSE(space.GetActorObservationClock(
      actor, &source_epoch, &source_sequence, &source_time, &has_sample));
}

using Reevaluation = Space::ReceiverReevaluationResult;

TEST_F(ReceiverSpaceInputTest, ReevaluationPreservesCauseAndConsumesFreshBodyOnce) {
  Begin();
  ASSERT_TRUE(Commit());
  update.timing = RelationUpdateTiming::kTimestampedObservation;
  update.source_epoch = 3;
  update.source_time = 2.5;
  update.source_clock = 9;
  update.sequence = 7;
  update.source_observation_sequence_known = true;
  update.source_observation_sequence = 42;
  update.actor_source_space_id = 1;
  update.actor_source_element_id = 99;
  int events = 0;
  space.RegisterOnRelationEvent([&](const RelationEvent&) { ++events; });
  const auto first = Solve();
  const auto receipts = space.GetReceivedUpdateCount();
  const auto time = space.GetSpaceClock();
  const auto event_count = events;
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 1),
            Reevaluation::kNoNewInput);
  // The original caller buffer is not the retained packet.
  update.payloads.clear();
  frame[1] = Pose("head", 7, 8, 1.6, 0.2);
  ASSERT_TRUE(Commit(2));
  ActorRealizationSnapshot next;
  ASSERT_TRUE(space.CopyActorRealizationSnapshot(actor, &next));
  EXPECT_EQ(next.revision, first.revision);  // Submit remains storage-only.
  ASSERT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 2),
            Reevaluation::kEvaluated);
  ASSERT_TRUE(space.CopyActorRealizationSnapshot(actor, &next));
  EXPECT_EQ(next.revision, first.revision + 1);
  EXPECT_EQ(next.source_timing_mode, first.source_timing_mode);
  EXPECT_EQ(next.source_epoch, first.source_epoch);
  EXPECT_EQ(next.source_sequence, first.source_sequence);
  EXPECT_EQ(next.source_observation_sequence, first.source_observation_sequence);
  EXPECT_EQ(next.source_observation_sequence_known,
            first.source_observation_sequence_known);
  EXPECT_EQ(next.source_time_seconds, first.source_time_seconds);
  EXPECT_EQ(next.source_clock_seconds, first.source_clock_seconds);
  EXPECT_EQ(next.source_space_id, first.source_space_id);
  EXPECT_EQ(next.relay_source_space_id, first.relay_source_space_id);
  EXPECT_EQ(next.source_actor_element_id, first.source_actor_element_id);
  EXPECT_EQ(next.receiver_ingress_monotonic_ns,
            first.receiver_ingress_monotonic_ns);
  EXPECT_GE(next.receiver_solve_monotonic_ns, first.receiver_solve_monotonic_ns);
  EXPECT_TRUE(Validity(next).valid);
  EXPECT_EQ(next.receiver_input.sequence, 2u);
  const auto* head = Channel(next, "head");
  ASSERT_NE(head, nullptr);
  EXPECT_TRUE(head->has_goal);
  EXPECT_EQ(head->target.position, Eigen::Vector2d(7, 8));
  EXPECT_EQ(head->target.z, 1.6);
  EXPECT_EQ(space.GetReceivedUpdateCount(), receipts);
  EXPECT_EQ(space.GetSpaceClock(), time);
  EXPECT_EQ(space.GetLastEmittedSequence(actor), 0);
  EXPECT_EQ(events, event_count);
  struct {
    Eigen::Vector2d position;
  } display;  // the Actor transform: never written for an input-owned actor
  display.position = DisplayPosition();
  EXPECT_EQ(display.position, Eigen::Vector2d::Zero());
  Space::ElementState element;
  ASSERT_TRUE(space.CopyElementState(10, &element));
  EXPECT_EQ(element.world_transform, Eigen::Matrix4d::Identity());
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 2),
            Reevaluation::kNoNewInput);
  ActorRealizationSnapshot repeated;
  ASSERT_TRUE(space.CopyActorRealizationSnapshot(actor, &repeated));
  EXPECT_EQ(repeated.revision, next.revision);
}

TEST_F(ReceiverSpaceInputTest, ReevaluationRequiresLiveExactInputAndSource) {
  Begin();
  ASSERT_TRUE(Commit());
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 1),
            Reevaluation::kUnavailable);
  const auto first = Solve();
  ASSERT_TRUE(Commit(2));
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch + 1, 2),
            Reevaluation::kStaleInput);
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 1),
            Reevaluation::kStaleInput);
  clock_now = first.receiver_input.valid_until_ns +
              ReceiverInputState::kMaxAgeNs;
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 2),
            Reevaluation::kUnavailable);
  ActorRealizationSnapshot unchanged;
  ASSERT_TRUE(space.CopyActorRealizationSnapshot(actor, &unchanged));
  EXPECT_EQ(unchanged.revision, first.revision);
}

TEST_F(ReceiverSpaceInputTest, ReevaluationEmptySourceReplacesOldGoalsWithoutLoop) {
  Begin();
  ASSERT_TRUE(Commit());
  Solve();
  update.payloads.clear();
  const auto empty = Solve();
  EXPECT_TRUE(empty.channels.empty());
  ASSERT_TRUE(Commit(2));
  ASSERT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 2),
            Reevaluation::kEvaluated);
  ActorRealizationSnapshot next;
  ASSERT_TRUE(space.CopyActorRealizationSnapshot(actor, &next));
  EXPECT_TRUE(next.channels.empty());
  EXPECT_EQ(next.source_sequence, empty.source_sequence);
  EXPECT_EQ(next.receiver_ingress_monotonic_ns,
            empty.receiver_ingress_monotonic_ns);
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 2),
            Reevaluation::kNoNewInput);
}

TEST_F(ReceiverSpaceInputTest, ReevaluationResetRetiresSourceUntilNewDelivery) {
  Begin();
  ASSERT_TRUE(Commit());
  update.timing = RelationUpdateTiming::kTimestampedObservation;
  update.source_epoch = 1;
  update.sequence = 1;
  Solve();
  update.timing = RelationUpdateTiming::kObservationReset;
  update.source_epoch = 2;
  update.sequence = 2;
  const auto reset = Solve();
  ASSERT_TRUE(Commit(2));
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 2),
            Reevaluation::kUnavailable);
  update.timing = RelationUpdateTiming::kTimestampedObservation;
  update.sequence = 3;
  const auto delivered = Solve();
  EXPECT_GT(delivered.revision, reset.revision);
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 2),
            Reevaluation::kNoNewInput);
  ASSERT_TRUE(Commit(3));
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 3),
            Reevaluation::kEvaluated);
}

TEST_F(ReceiverSpaceInputTest, ReevaluationSourceRemovalClearsSharedIdFallback) {
  SpaceHub hub;
  space.SetHub(&hub);
  Begin();
  ASSERT_TRUE(Commit());
  const auto first = Solve();
  ElementEvent removed;
  removed.type = ElementEvent::kRemoved;
  removed.source_space_id = update.source_space_id;
  removed.shared_id = update.actor_shared_id;
  removed.source_element_id = 999;
  space.OnElementEvent(removed);
  ASSERT_TRUE(space.HasActor(actor));  // Receiver-owned actor survives.
  ASSERT_TRUE(Commit(2));
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 2),
            Reevaluation::kUnavailable);
  ActorRealizationValidity validity;
  ASSERT_TRUE(space.GetActorRealizationValidity(
      actor, first.revision, kRealizationCounterpartRevision, &validity));
  EXPECT_NE(validity.changed_mask & kRealizationCounterpartRevision, 0u);
  // A genuine new source delivery remains dispatchable in the same token.
  const auto next = Solve();
  EXPECT_GT(next.revision, first.revision);
  space.SetHub(nullptr);
}

TEST_F(ReceiverSpaceInputTest, ReevaluationNewEpochAndRichLifetimeRequireSource) {
  Begin();
  ASSERT_TRUE(Commit());
  Solve();
  Begin();
  ASSERT_TRUE(Commit());
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 1),
            Reevaluation::kUnavailable);
  Solve();
  space.SetRichRealizationEnabled(false);
  space.SetActorRichRealizationEnabled(actor, true);
  ASSERT_TRUE(Commit(2));
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 2),
            Reevaluation::kUnavailable);
}

TEST_F(ReceiverSpaceInputTest, ReevaluationRevokedActorLinkCannotFallBackOrRevive) {
  SpaceHub hub;
  space.SetHub(&hub);
  update.actor_source_space_id = 1;
  update.actor_source_element_id = 99;
  hub.GetLinker().CreateLink({1, 99}, {2, 10});
  Begin();
  ASSERT_TRUE(Commit());
  const auto first = Solve();
  ASSERT_TRUE(Commit(2));
  ASSERT_TRUE(hub.GetLinker().RemoveSource({1, 99}));
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 2), Reevaluation::kUnavailable);
  hub.GetLinker().CreateLink({1, 99}, {2, 10});
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 2), Reevaluation::kUnavailable);
  const auto next = Solve();
  EXPECT_GT(next.revision, first.revision);
  space.SetHub(nullptr);
}

TEST_F(ReceiverSpaceInputTest, ReevaluationReadsCurrentSupportPolicyAndUnmappedLedger) {
  Begin();
  ASSERT_TRUE(Commit());
  const auto first = Solve();
  Eigen::Matrix4d moved = Eigen::Matrix4d::Identity();
  moved(2, 3) = .2;
  ASSERT_TRUE(space.SetElementTransform(20, moved));
  // This fixture intentionally uses the legacy definition-world convention;
  // its support plane moves through a new owned definition, not element pose.
  auto square = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}});
  auto definition = std::make_shared<NavigableSpatialDefinition>(square);
  SubRegion seat;
  seat.name = "seat";
  seat.boundary = square;
  seat.local_transform(2, 3) = .6;
  definition->AddSubRegion(seat);
  ASSERT_TRUE(space.SetElementSpatialDefinition(20, definition));
  space.SetRealizationPolicy(RealizationPolicyType::kUniform);
  ASSERT_TRUE(Commit(2));
  ASSERT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 2), Reevaluation::kEvaluated);
  ActorRealizationSnapshot next;
  ASSERT_TRUE(space.CopyActorRealizationSnapshot(actor, &next));
  EXPECT_GT(next.tree_revision, first.tree_revision);
  EXPECT_GT(next.support_revision, first.support_revision);
  EXPECT_GT(next.policy_revision, first.policy_revision);
  const auto* hand = Channel(next, "hand");
  ASSERT_NE(hand, nullptr);
  ASSERT_FALSE(hand->contributors.empty());
  EXPECT_NEAR(hand->contributors[0].mapped_support.z(), .6, 1e-12);
  // Current strict resolution, not a cached resolved local ID.
  space.SetReferentResolveMode(ReferentResolveMode::kLinkerOnly);
  ASSERT_TRUE(Commit(3));
  ASSERT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 3), Reevaluation::kEvaluated);
  ASSERT_TRUE(space.CopyActorRealizationSnapshot(actor, &next));
  for (const auto& channel : next.channels) {
    EXPECT_FALSE(channel.has_goal);
    for (const auto& contributor : channel.contributors)
      EXPECT_EQ(contributor.disposition, RealizationDisposition::kUnmapped);
  }
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 3), Reevaluation::kNoNewInput);
}

TEST_F(ReceiverSpaceInputTest, ReevaluationHubDisconnectRetiresFallbackSource) {
  SpaceHub hub;
  space.SetHub(&hub);
  Begin();
  ASSERT_TRUE(Commit());
  Solve();
  space.SetHub(nullptr);
  ASSERT_TRUE(Commit(2));
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 2), Reevaluation::kUnavailable);
  space.SetHub(&hub);
  EXPECT_EQ(space.ReevaluateReceiverRealization(actor, epoch, 2), Reevaluation::kUnavailable);
  space.SetHub(nullptr);
}

// Engagement is the geometric contact of the source anchor with its surface,
// read from the source payload. There is no EMA gate and no hysteresis: a touching
// payload engages on its own solve, a payload that stops touching releases on its own
// solve, and a receiver re-evaluation (at body rate) can change neither.
TEST_F(ReceiverSpaceInputTest, EngagementIsGeometricAndReevaluationCannotChangeIt) {
  Begin();
  ASSERT_TRUE(Commit());
  update.payloads[0].grounded_relation.proximity = .1;  // 9 m from its support
  ASSERT_FALSE(Channel(Solve(), "hand")->has_goal);
  update.payloads[0].grounded_relation.proximity = 1;  // touching
  ASSERT_TRUE(Channel(Solve(), "hand")->has_goal) << "engages on its own solve";
  for (uint64_t sequence = 2; sequence < 14; ++sequence) {
    ASSERT_TRUE(Commit(sequence));
    ASSERT_EQ(space.ReevaluateReceiverRealization(actor, epoch, sequence),
              Reevaluation::kEvaluated);
    ActorRealizationSnapshot snapshot;
    ASSERT_TRUE(space.CopyActorRealizationSnapshot(actor, &snapshot));
    EXPECT_TRUE(Channel(snapshot, "hand")->has_goal);
  }
  update.payloads[0].grounded_relation.proximity = .1;  // no longer touching
  ASSERT_FALSE(Channel(Solve(), "hand")->has_goal) << "releases on its own solve";
  for (uint64_t sequence = 14; sequence < 26; ++sequence) {
    ASSERT_TRUE(Commit(sequence));
    ASSERT_EQ(space.ReevaluateReceiverRealization(actor, epoch, sequence),
              Reevaluation::kEvaluated);
    ActorRealizationSnapshot snapshot;
    ASSERT_TRUE(space.CopyActorRealizationSnapshot(actor, &snapshot));
    EXPECT_FALSE(Channel(snapshot, "hand")->has_goal);
  }
}

TEST_F(ReceiverSpaceInputTest, SourceSolveKeepsPreActorResolutionCounterpartFence) {
  SpaceHub hub;
  space.SetHub(&hub);
  space.SetReferentResolveMode(ReferentResolveMode::kLinker);
  update.actor_source_space_id = 1;
  update.actor_source_element_id = 99;
  hub.GetLinker().CreateLink({1, 99}, {2, 10});
  ASSERT_TRUE(space.AddElement(11));
  Begin();
  ASSERT_TRUE(Commit());
  const auto baseline = Solve();
  ASSERT_NE(baseline.known_revision_mask & kRealizationCounterpartRevision, 0u);
  update.timing = RelationUpdateTiming::kTimestampedObservation;
  update.source_epoch = 1;
  update.sequence = 1;
  // Initial epoch cancellation reads receiver validity AFTER selecting actor,
  // but BEFORE the normal solve. Mutate only the independently owned linker.
  int mutations = 0;
  on_clock_read = [&] { ++mutations; hub.GetLinker().CreateLink({1, 99}, {2, 11}); };
  const auto mixed = Solve();
  ASSERT_EQ(mutations, 1);
  ASSERT_EQ(mixed.revision, baseline.revision + 2);
  ASSERT_FALSE(mixed.channels.empty());
  EXPECT_EQ(mixed.known_revision_mask & kRealizationCounterpartRevision, 0u);
  EXPECT_NE(mixed.unknown_revision_mask & kRealizationCounterpartRevision, 0u);
  space.SetHub(nullptr);
}

TEST_F(ReceiverSpaceInputTest,
       SolverUsesExternalRootAndFacingWithoutDirectApply) {
  Begin();
  ASSERT_TRUE(Commit());
  const auto result = Solve();
  EXPECT_TRUE(Validity(result).valid);
  const auto* targets = space.GetRealizationTargets(actor);
  ASSERT_NE(targets, nullptr);
  ASSERT_FALSE(targets->targets.empty());
  EXPECT_EQ(targets->targets[0].position, Eigen::Vector2d(2, 3));
  EXPECT_NEAR(targets->targets[0].yaw, 0.4, 1e-12);
  const auto* head = Channel(result, "head");
  ASSERT_NE(head, nullptr);
  ASSERT_TRUE(head->has_goal);
  EXPECT_EQ(head->target.position, Eigen::Vector2d(2, 0));
  EXPECT_EQ(head->target.z, 1.7);
  struct {
    Eigen::Vector2d position;
  } display;  // the Actor transform: never written for an input-owned actor
  display.position = DisplayPosition();
  EXPECT_EQ(display.position, Eigen::Vector2d::Zero());
}

TEST_F(ReceiverSpaceInputTest,
       TargetEventsUseTheActualCommittedFrameStampForEveryAnchor) {
  Begin();
  ASSERT_TRUE(Commit(17));
  ASSERT_TRUE(space.SetTargetEventDelivery(actor, true));
  Space::ActorEvent event;
  while (space.PollActorEvent(actor, &event)) {
  }
  Solve();
  uint32_t head = 0;
  ASSERT_TRUE(space.AnchorIdFor(actor, "head", &head));
  bool saw_head = false;
  while (space.PollActorEventAfterSolve(actor, &event) == 1) {
    if (event.kind != Space::ActorEvent::Kind::kPositionRequest &&
        event.kind != Space::ActorEvent::Kind::kYawRequest)
      continue;
    EXPECT_EQ(event.sample_epoch, epoch);
    EXPECT_EQ(event.sample_sequence, 17u);
    if (event.anchor_id == head) {
      saw_head = true;
      EXPECT_NEAR(event.sample_yaw, 0.7, 1e-12);
    }
  }
  EXPECT_TRUE(saw_head);
}

TEST_F(ReceiverSpaceInputTest, NamedRootSpacingPublishesItsExactAnchorGoal) {
  ASSERT_TRUE(space.AddActorAnchors(actor, {SubjectAnchor("locomotion")}));
  frame.push_back(Pose("locomotion", 1.5, 2.0, 0.9, 0.6));
  ASSERT_TRUE(space.BeginReceiverInputEpoch(actor, "locomotion", 7, 9, &epoch));
  ASSERT_TRUE(Commit());
  RelationPayload spacing;
  spacing.referent_shared_id = 42;
  spacing.subject_anchor_name = "locomotion";
  spacing.kernel_type = RelationKernelType::kAvatarSpacing;
  spacing.grounded_relation.proximity = 1;
  // R = 1, in contact: v1's count is 1 (SourceUpdateCount(0, 0)). Without a count Progressive computes no weight.
  spacing.update_count = SourceUpdateCount(0.0, 0.0);
  spacing.grounded_relation.encoded_state = Eigen::Vector3d(.25, -.5, 1.2);
  update.payloads.push_back(spacing);
  const auto result = Solve();
  const auto* channel = Channel(result, "locomotion");
  ASSERT_NE(channel, nullptr);
  ASSERT_TRUE(channel->has_goal);
  EXPECT_EQ(channel->status, SolverStatus::kConverged);
  EXPECT_EQ(channel->geometry, RealizationGeometry::kNotApplicable);
  EXPECT_EQ(channel->goal_kind, RealizationGoalKind::kRootPosition);
  EXPECT_EQ(channel->target.anchor_name, "locomotion");
  EXPECT_FALSE(channel->target.has_z);
  EXPECT_NEAR(channel->target.yaw, 0.6, 1e-12);  // Input context, not source yaw.
  ASSERT_EQ(channel->contributors.size(), 1u);
  EXPECT_EQ(channel->contributors[0].disposition, RealizationDisposition::kIncluded);
  EXPECT_GT((channel->target.position - Eigen::Vector2d(1.5, 2)).norm(), .01);
  struct {
    Eigen::Vector2d position;
  } display;  // the Actor transform: never written for an input-owned actor
  display.position = DisplayPosition();
  EXPECT_EQ(display.position, Eigen::Vector2d::Zero());
  EXPECT_TRUE(Validity(result).valid);
}

TEST_F(ReceiverSpaceInputTest, NonRootSpacingCannotDriveOwnedRootGoal) {
  Begin();
  ASSERT_TRUE(Commit());
  RelationPayload spacing;
  spacing.referent_shared_id = 42;
  spacing.subject_anchor_name = "hand";
  spacing.kernel_type = RelationKernelType::kAvatarSpacing;
  spacing.grounded_relation.proximity = 1;
  // R = 1, in contact: v1's count is 1 (SourceUpdateCount(0, 0)). Without a count Progressive computes no weight.
  spacing.update_count = SourceUpdateCount(0.0, 0.0);
  spacing.grounded_relation.encoded_state = Eigen::Vector3d(.25, -.5, 1.2);
  update.payloads.push_back(spacing);
  const auto result = Solve();
  const auto* targets = space.GetRealizationTargets(actor);
  ASSERT_NE(targets, nullptr);
  ASSERT_FALSE(targets->targets.empty());
  EXPECT_EQ(targets->targets[0].position, Eigen::Vector2d(2, 3));
  EXPECT_NEAR(targets->targets[0].yaw, .4, 1e-12);
  bool retained_unsupported = false;
  for (const auto& channel : result.channels) {
    if (channel.anchor_name != "hand" || channel.family != RelationKernelType::kAvatarSpacing) continue;
    retained_unsupported = true;
    EXPECT_FALSE(channel.has_goal);
    ASSERT_EQ(channel.contributors.size(), 1u);
    EXPECT_EQ(channel.contributors[0].disposition, RealizationDisposition::kUnsupported);
  }
  EXPECT_TRUE(retained_unsupported);
}

TEST_F(ReceiverSpaceInputTest, AnonymousSpacingDoesNotAliasNamedReceiverRoot) {
  ASSERT_TRUE(space.AddActorAnchors(actor, {SubjectAnchor("locomotion")}));
  frame.push_back(Pose("locomotion", 1.5, 2.0, 0.9, 0.6));
  ASSERT_TRUE(space.BeginReceiverInputEpoch(actor, "locomotion", 7, 9, &epoch));
  ASSERT_TRUE(Commit());
  RelationPayload anonymous;
  anonymous.referent_shared_id = 42;
  anonymous.kernel_type = RelationKernelType::kAvatarSpacing;
  anonymous.grounded_relation.proximity = 1;
  // R = 1, in contact: v1's count is 1 (SourceUpdateCount(0, 0)). Without a count Progressive computes no weight.
  anonymous.update_count = SourceUpdateCount(0.0, 0.0);
  anonymous.grounded_relation.encoded_state = Eigen::Vector3d(.25, -.5, 1.2);
  update.payloads.push_back(anonymous);
  const auto result = Solve();
  const auto* targets = space.GetRealizationTargets(actor);
  ASSERT_NE(targets, nullptr);
  ASSERT_FALSE(targets->targets.empty());
  EXPECT_EQ(targets->targets[0].anchor_name, "locomotion");
  EXPECT_EQ(targets->targets[0].position, Eigen::Vector2d(1.5, 2));
  const auto* legacy = Channel(result, "root");
  ASSERT_NE(legacy, nullptr);
  EXPECT_FALSE(legacy->has_goal);
  ASSERT_EQ(legacy->contributors.size(), 1u);
  EXPECT_EQ(legacy->contributors[0].disposition, RealizationDisposition::kUnsupported);
}

TEST_F(ReceiverSpaceInputTest, SourceAndGeneratedWritersCannotBecomeInput) {
  Begin();
  ASSERT_TRUE(Commit());
  const auto first = Solve();
  space.UpdateActorState(actor, Eigen::Vector2d(99, 98), 1.0);
  {
    EmbodiedPoseInput generated;
    generated.x = 97;
    generated.y = 96;
    generated.epoch = 1;
    EXPECT_EQ(space.PublishEmbodiedPose(actor, generated, 0.0, 1),
              EmbodiedPublishResult::kRefused);
  }
  space.SetAnchorWorldPose(actor, "head", Eigen::Vector3d(-5, -6, -7), -0.8);
  EXPECT_FALSE(space.SetAnchorActive(actor, "head", false));
  EXPECT_TRUE(Validity(first).valid);
  const auto second = Solve();
  EXPECT_EQ(second.receiver_input.revision, first.receiver_input.revision);
  const auto* head = Channel(second, "head");
  ASSERT_NE(head, nullptr);
  ASSERT_TRUE(head->has_goal);
  EXPECT_EQ(head->target.position, Eigen::Vector2d(2, 0));
  EXPECT_EQ(head->target.z, 1.7);
  Space::ElementState element;
  ASSERT_TRUE(space.CopyElementState(10, &element));
  EXPECT_EQ(element.world_transform, Eigen::Matrix4d::Identity());
}

TEST_F(ReceiverSpaceInputTest, MissingAnchorReleasesOnlyDependentGoals) {
  Begin();
  ASSERT_TRUE(Commit());
  const auto first = Solve();
  ASSERT_TRUE(Channel(first, "hand")->has_goal);
  frame[2].available = false;
  ASSERT_TRUE(Commit(2));
  const auto changed = Validity(first);
  EXPECT_TRUE(changed.current.known);
  EXPECT_TRUE(changed.changed);
  EXPECT_FALSE(changed.valid);
  const auto missing_hand = Solve();
  EXPECT_TRUE(Validity(missing_hand).valid);
  EXPECT_FALSE(Channel(missing_hand, "hand")->has_goal);
  EXPECT_TRUE(Channel(missing_hand, "head")->has_goal);
  frame[0].available = false;
  ASSERT_TRUE(Commit(3));
  const auto missing_root = Solve();
  EXPECT_FALSE(Validity(missing_root).valid);
  EXPECT_EQ(missing_root.solve_status, SolverStatus::kFailed);
  for (const auto& channel : missing_root.channels)
    EXPECT_FALSE(channel.has_goal);
  frame[0].available = frame[2].available = true;
  ASSERT_TRUE(Commit(4));
  const auto recovered = Solve();
  EXPECT_TRUE(Validity(recovered).valid);
  EXPECT_TRUE(Channel(recovered, "hand")->has_goal);
}

TEST_F(ReceiverSpaceInputTest, ClearAndStructuralRebindCannotReviveOldInput) {
  Begin();
  ASSERT_TRUE(Commit());
  const auto first = Solve();
  ASSERT_TRUE(space.ClearReceiverInputFrame(actor, epoch));
  EXPECT_FALSE(Validity(first).current.known);
  EXPECT_TRUE(Validity(first).changed);
  EXPECT_FALSE(Commit(1));
  ASSERT_TRUE(Commit(2));
  const auto recovered = Solve();
  ASSERT_TRUE(space.SetActorElement(actor, 10));
  EXPECT_TRUE(Validity(recovered).valid);  // Same identity is a no-op.
  ASSERT_TRUE(space.AddElement(11));
  ASSERT_TRUE(space.SetElementSharedId(11, 500));
  ASSERT_TRUE(space.SetActorElement(actor, 11));
  EXPECT_FALSE(Validity(recovered).valid);
  EXPECT_FALSE(Commit(3));
  const auto old_epoch = epoch;
  Begin();
  EXPECT_GT(epoch, old_epoch);
  ASSERT_TRUE(Commit(1));
  ASSERT_TRUE(space.AddActorAnchors(actor, {SubjectAnchor("pelvis")}));
  EXPECT_FALSE(Commit(2));
  Begin();
  frame.push_back(Pose("pelvis"));
  EXPECT_TRUE(Commit(1));
}

TEST_F(ReceiverSpaceInputTest, NoFallbackBeforeInputOrAfterBorrowedEscape) {
  const auto legacy = Solve();
  EXPECT_FALSE(legacy.receiver_input.known);
  Begin();
  const auto empty = Solve();
  EXPECT_EQ(empty.solve_status, SolverStatus::kFailed);
  for (const auto& channel : empty.channels) EXPECT_FALSE(channel.has_goal);
  ASSERT_TRUE(Commit());
  const auto active = Solve();
  EXPECT_TRUE(Validity(active).valid);
  ASSERT_NE(space.GetActor(actor), nullptr);
  EXPECT_FALSE(Validity(active).current.known);
  EXPECT_FALSE(Commit(2));
  const auto escaped = Solve();
  EXPECT_EQ(escaped.solve_status, SolverStatus::kFailed);
  EXPECT_FALSE(space.BeginReceiverInputEpoch(actor, "root", 7, 9, &epoch));
}

TEST_F(ReceiverSpaceInputTest, ExternalInputOwnsThePoseAndCoreNeverWritesIt) {
  Begin();
  ASSERT_TRUE(Commit());
  const auto first = Solve();
  EXPECT_TRUE(Validity(first).valid);
  struct {
    Eigen::Vector2d position;
  } display;  // the Actor transform: never written for an input-owned actor
  display.position = DisplayPosition();
  EXPECT_EQ(display.position, Eigen::Vector2d::Zero());
  // The stamped frame is the ONLY writer of the pose: a direct publish through
  // the public port is refused, and the Actor transform is never written.
  EmbodiedPoseInput direct;
  direct.x = 7;
  direct.y = 8;
  direct.epoch = 1;
  EXPECT_EQ(space.PublishEmbodiedPose(actor, direct, 0.0, 1),
            EmbodiedPublishResult::kRefused);
  ASSERT_TRUE(Commit(2));
  display.position = DisplayPosition();
  EXPECT_EQ(display.position, Eigen::Vector2d::Zero());
  EmbodiedSample sampled;
  ASSERT_TRUE(space.SampleEmbodiedPose(actor, &sampled));
  ASSERT_TRUE(sampled.valid);
  EXPECT_EQ(Eigen::Vector2d(sampled.x, sampled.y),
            Eigen::Vector2d(2, 3));  // the committed frame's root
  EXPECT_EQ(sampled.sequence, 2u);
}

TEST_F(ReceiverSpaceInputTest, FrozenRevisionCannotBeReplacedByControlOrPoll) {
  Begin();
  ASSERT_TRUE(Commit());
  const auto first = Solve();
  EXPECT_TRUE(Validity(first).valid);
  const auto receipts = space.GetReceivedUpdateCount();
  ASSERT_TRUE(Commit(2));
  EXPECT_EQ(space.GetReceivedUpdateCount(), receipts);
  EXPECT_EQ(Validity(first).revision, first.revision);
  update.payloads.clear();
  const auto released = Solve();
  EXPECT_TRUE(released.channels.empty());
  EXPECT_FALSE(released.receiver_input.known);  // No pose consumed by cancel.
  ActorReceiverInputValidity stale;
  EXPECT_FALSE(
      space.GetActorReceiverInputValidity(actor, first.revision, &stale));
}

TEST_F(ReceiverSpaceInputTest,
       ExpiryDuringSolveOrPublicationWithdrawsEveryGoal) {
  clock_now = 100;
  Begin();
  for (int checkpoint : {3, 4}) {
    expire_after_read = -1;
    ASSERT_TRUE(Commit(checkpoint));
    clock_reads = 0;
    expire_after_read = checkpoint;
    const auto expired = Solve();
    EXPECT_FALSE(expired.receiver_input.known);
    EXPECT_EQ(expired.solve_status, SolverStatus::kFailed);
    for (const auto& channel : expired.channels) EXPECT_FALSE(channel.has_goal);
    EXPECT_EQ(space.GetRealizationTargets(actor), nullptr);
    // An expired frame leaves NO pose and no fallback: a direct publish is
    // refused (the stream still owns the pose) and the solve stays failed.
    EmbodiedPoseInput direct;
    direct.x = 8;
    direct.y = 9;
    direct.epoch = 1;
    EXPECT_EQ(space.PublishEmbodiedPose(actor, direct, 0.0, checkpoint),
              EmbodiedPublishResult::kRefused);
    EXPECT_EQ(DisplayPosition(), Eigen::Vector2d::Zero());
  }
}

TEST_F(ReceiverSpaceInputTest, IdleExpiryDoesNotRefreshReceiptOrExposeTargets) {
  clock_now = 100;
  Begin();
  ASSERT_TRUE(Commit());
  const auto first = Solve();
  ASSERT_TRUE(Validity(first).valid);
  const auto receipt = space.GetReceivedUpdateCount();
  clock_now += ReceiverInputState::kMaxAgeNs;
  const auto expired = Validity(first);
  EXPECT_FALSE(expired.valid);
  EXPECT_FALSE(expired.current.known);
  EXPECT_FALSE(expired.changed);  // Time expiry is not a fabricated revision.
  EXPECT_EQ(space.GetRealizationTargets(actor), nullptr);
  EXPECT_EQ(space.GetReceivedUpdateCount(), receipt);
  EXPECT_EQ(expired.revision, first.revision);
}

TEST_F(ReceiverSpaceInputTest, SourceControlCannotResetAnOptedReceiver) {
  Begin();
  ASSERT_TRUE(Commit());
  const auto first = Solve();
  const auto count = space.GetReceivedUpdateCount();
  const auto time = space.GetSpaceClock();
  uint64_t source_epoch = 0;
  EXPECT_FALSE(space.BeginActorObservationEpoch(actor, "root", &source_epoch));
  EXPECT_FALSE(space.ApplyActorObservationFrame(
      actor, 1, 1, 1, {{"root", Eigen::Vector3d::Zero(), 0}}));
  EXPECT_FALSE(space.ClearActorObservationAnchor(actor, 1, "root"));
  space.UpdateActorState(actor, Eigen::Vector2d(99, 99), 1);
  EXPECT_EQ(space.GetReceivedUpdateCount(), count);
  EXPECT_EQ(space.GetSpaceClock(), time);
  EXPECT_TRUE(Validity(first).valid);
  ActorRealizationSnapshot unchanged;
  ASSERT_TRUE(space.CopyActorRealizationSnapshot(actor, &unchanged));
  EXPECT_EQ(unchanged.revision, first.revision);
  EXPECT_TRUE(Channel(unchanged, "hand")->has_goal);
}

TEST_F(ReceiverSpaceInputTest, EstablishedSourceCannotSilentlyBecomeReceiver) {
  uint64_t source_epoch;
  ASSERT_TRUE(space.BeginActorObservationEpoch(actor, "root", &source_epoch));
  EXPECT_FALSE(space.BeginReceiverInputEpoch(actor, "root", 7, 9, &epoch));
  EXPECT_TRUE(space.ApplyActorObservationFrame(
      actor, source_epoch, 1, 1, {{"root", Eigen::Vector3d::Zero(), 0}}));
}

TEST_F(ReceiverSpaceInputTest, ChildOrOffsetBodyAnchorCannotBecomePlanarRoot) {
  auto can_begin = [](const std::vector<SubjectAnchor>& layout,
                      const std::string& root) {
    Space subject;
    if (!subject.AddElement(1)) return false;
    const auto candidate = subject.CreateActor(layout);
    if (!subject.SetActorElement(candidate, 1)) return false;
    uint64_t epoch = 0;
    return subject.BeginReceiverInputEpoch(candidate, root, 7, 9, &epoch);
  };
  EXPECT_FALSE(can_begin(
      {SubjectAnchor("root"),
       SubjectAnchor("head", {0, 0, 1.7},
                     {StateType::kPosition3D, StateType::kRotation3D}, "root")},
      "head"));
  EXPECT_FALSE(can_begin({SubjectAnchor("root", {0.1, 0, 0})}, "root"));
  EXPECT_FALSE(can_begin(
      {SubjectAnchor("root", {0, 0, 0}, {StateType::kPosition3D})}, "root"));
  EXPECT_TRUE(can_begin({SubjectAnchor("root")}, "root"));
}

// RETIRED: CustomControllerCannotBypassSeparateAdmission
// tested that an application-supplied AvatarController is never ticked for an
// input-owned actor. There is no controller in core any more: the body belongs
// to the driver and a proposal moves nothing (see
// ExternalInputOwnsThePoseAndCoreNeverWritesIt).

TEST_F(ReceiverSpaceInputTest, TargetOnlyInputDoesNotDependOnRichSubscription) {
  Begin();
  space.SetRichRealizationEnabled(false);
  ASSERT_TRUE(Commit());
  space.OnRemoteUpdate(update);
  const auto* result = space.GetRealizationTargets(actor);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->targets.size(), 3u);
  EXPECT_EQ(result->targets[0].position, Eigen::Vector2d(2, 3));
  EXPECT_EQ(result->targets[2].anchor_name, "head");
  EXPECT_EQ(result->targets[2].position, Eigen::Vector2d(2, 0));
  struct {
    Eigen::Vector2d position;
  } display;  // the Actor transform: never written for an input-owned actor
  display.position = DisplayPosition();
  EXPECT_EQ(display.position, Eigen::Vector2d::Zero());
}

TEST_F(ReceiverSpaceInputTest, SpaceIdentityAndActorDestructionRetireInputs) {
  Begin();
  ASSERT_TRUE(Commit());
  const auto first = Solve();
  space.SetSpaceId(2);
  EXPECT_TRUE(Validity(first).valid);
  space.SetSpaceId(3);
  EXPECT_FALSE(Validity(first).valid);
  EXPECT_FALSE(Commit(2));
  Begin();
  ASSERT_TRUE(Commit(1));
  const auto second = Solve();
  ASSERT_TRUE(space.DestroyActor(actor));
  ActorReceiverInputValidity value;
  EXPECT_FALSE(
      space.GetActorReceiverInputValidity(actor, second.revision, &value));
  EXPECT_FALSE(Commit(2));
}

TEST_F(ReceiverSpaceInputTest, ConcurrentFrameAndSolveNeverCertifyTornAnchors) {
  Begin();
  auto facing_hand = update.payloads[1];
  facing_hand.subject_anchor_name = "hand";
  update.payloads[0] = facing_hand;
  std::atomic<bool> done{false};
  std::thread writer([&] {
    for (uint64_t sequence = 1; sequence <= 100; ++sequence) {
      for (auto& anchor : frame) anchor.world_pose(0, 3) = sequence;
      EXPECT_TRUE(Commit(sequence));
    }
    done.store(true);
  });
  do {
    const auto snapshot = Solve();
    if (!snapshot.receiver_input.known) continue;
    for (const auto& channel : snapshot.channels) {
      EXPECT_TRUE(channel.has_goal);
      EXPECT_EQ(channel.target.position.x(), snapshot.receiver_input.sequence);
    }
    // The writer needs the space lock for each frame and the mutex is not fair: a
    // solver thread that re-locks at once can starve it past a frame's validity.
    std::this_thread::yield();
  } while (!done.load());
  writer.join();
}

}  // namespace
}  // namespace spatialblend
