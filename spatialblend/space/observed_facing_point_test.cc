// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace {

using PointKind = Space::ObservedFacingPointKind;

TEST(ObservedSourceSetupTest,
     ValidatesEveryFreshEpochAndPointBeforeInstallingAnyMember) {
  Space space;
  space.SetSpaceId(7);
  ASSERT_TRUE(space.AddElement(1));
  const auto anchors = std::vector<SubjectAnchor>{
      SubjectAnchor("root"),
      SubjectAnchor("head", Eigen::Vector3d::Zero(),
                    {StateType::kPosition3D, StateType::kRotationYaw})};
  std::vector<spatialblend::entity_id_t> actors;
  for (int index = 0; index < 2; ++index) {
    ASSERT_TRUE(space.AddElement(10 + index, 1));
    ASSERT_TRUE(space.AddElement(30 + index, 1));
    ASSERT_TRUE(space.SetElementSharedId(10 + index, 100 + index));
    actors.push_back(space.CreateActor(anchors));
    ASSERT_TRUE(space.SetActorElement(actors.back(), 10 + index));
  }
  const std::vector<Space::ObservedSourceSetupParticipant> participants{
      {actors[0], "root"}, {actors[1], "root"}};
  const std::vector<Space::ObservedSourceSetupPoint> invalid_points{
      {30, actors[0], "head", 1000},
      // Actor-owned element: this late member must reject the whole setup.
      {11, actors[1], "head", 1000}};
  std::vector<Space::ObservedSourceSetupActorResult> actor_results;
  std::vector<Space::ObservedSourceSetupPointResult> point_results;
  EXPECT_FALSE(space.BeginObservedSourceSetup(participants, invalid_points,
                                              &actor_results, &point_results));
  for (const auto actor : actors) {
    uint64_t epoch = 99;
    int64_t sequence = 99;
    double time = 99;
    bool has_sample = true;
    EXPECT_FALSE(space.GetActorObservationClock(actor, &epoch, &sequence, &time,
                                                 &has_sample));
  }

  const std::vector<Space::ObservedSourceSetupPoint> points{
      {30, actors[0], "head", 1000}, {31, actors[1], "head", 1000}};
  ASSERT_TRUE(space.BeginObservedSourceSetup(participants, points, &actor_results,
                                             &point_results));
  ASSERT_EQ(actor_results.size(), 2u);
  ASSERT_EQ(point_results.size(), 2u);
  for (size_t index = 0; index < actors.size(); ++index) {
    uint64_t epoch = 0;
    int64_t sequence = 0;
    double time = -1;
    bool has_sample = true;
    ASSERT_TRUE(space.GetActorObservationClock(actors[index], &epoch, &sequence,
                                               &time, &has_sample));
    EXPECT_EQ(epoch, actor_results[index].epoch);
    EXPECT_EQ(sequence, -1);
    EXPECT_FALSE(has_sample);
  }
  auto wrong_lease = point_results;
  ++wrong_lease[1].lease;
  EXPECT_FALSE(space.RetireObservedSourceSetup(actor_results, wrong_lease));
  uint64_t still_open_epoch = 0;
  int64_t still_open_sequence = 0;
  double still_open_time = 0;
  bool still_open_sample = true;
  EXPECT_TRUE(space.GetActorObservationClock(
      actors[0], &still_open_epoch, &still_open_sequence, &still_open_time,
      &still_open_sample));
  EXPECT_TRUE(space.RetireObservedSourceSetup(actor_results, point_results));
  // Retirement removes the active epochs but fences their values: a future
  // epoch cannot ABA-reuse a cancelled capability.
  uint64_t next_epoch = 0;
  ASSERT_TRUE(space.BeginActorObservationEpoch(actors[0], "root", &next_epoch));
  EXPECT_GT(next_epoch, actor_results[0].epoch);
}

TEST(ObservedSourceSetupTest, CannotClaimCancellationAfterEvidenceAdmission) {
  Space space;
  space.SetSpaceId(7);
  ASSERT_TRUE(space.AddElement(1));
  ASSERT_TRUE(space.AddElement(10, 1));
  ASSERT_TRUE(space.AddElement(30, 1));
  ASSERT_TRUE(space.SetElementSharedId(10, 100));
  const auto actor = space.CreateActor(
      {SubjectAnchor("root"), SubjectAnchor("head", Eigen::Vector3d::Zero(),
                                              {StateType::kPosition3D,
                                               StateType::kRotationYaw})});
  ASSERT_TRUE(space.SetActorElement(actor, 10));
  std::vector<Space::ObservedSourceSetupActorResult> actor_results;
  std::vector<Space::ObservedSourceSetupPointResult> point_results;
  ASSERT_TRUE(space.BeginObservedSourceSetup({{actor, "root"}},
                                             {{30, actor, "head", 1000}},
                                             &actor_results, &point_results));
  ASSERT_TRUE(space.ApplyActorObservationFrame(
      actor, actor_results[0].epoch, 0, 0,
      {{"root", {0, 0, 0}, 0}, {"head", {0, 0, 1.7}, 0}}));
  EXPECT_FALSE(space.RetireObservedSourceSetup(actor_results, point_results));
}

TEST(ObservedSourceSetupTest,
     ReplacesOnlyTheExactlyDeclaredLegacyEpochAndNeverAnotherSetup) {
  Space space;
  space.SetSpaceId(7);
  ASSERT_TRUE(space.AddElement(1));
  ASSERT_TRUE(space.AddElement(10, 1));
  ASSERT_TRUE(space.AddElement(30, 1));
  ASSERT_TRUE(space.SetElementSharedId(10, 100));
  const auto actor = space.CreateActor(
      {SubjectAnchor("root"), SubjectAnchor("head", Eigen::Vector3d::Zero(),
                                              {StateType::kPosition3D,
                                               StateType::kRotationYaw})});
  ASSERT_TRUE(space.SetActorElement(actor, 10));
  uint64_t legacy_epoch = 0;
  ASSERT_TRUE(space.BeginActorObservationEpoch(actor, "root", &legacy_epoch));
  std::vector<Space::ObservedSourceSetupActorResult> actor_results;
  std::vector<Space::ObservedSourceSetupPointResult> point_results;
  const auto point = std::vector<Space::ObservedSourceSetupPoint>{
      {30, actor, "head", 1000}};
  EXPECT_FALSE(space.BeginObservedSourceSetup(
      {{actor, "root", legacy_epoch + 1}}, point, &actor_results,
      &point_results));
  uint64_t observed_epoch = 0;
  int64_t sequence = 0;
  double time = 0;
  bool has_sample = true;
  ASSERT_TRUE(space.GetActorObservationClock(actor, &observed_epoch, &sequence,
                                             &time, &has_sample));
  EXPECT_EQ(observed_epoch, legacy_epoch);
  EXPECT_EQ(sequence, -1);
  EXPECT_FALSE(has_sample);

  uint64_t independently_owned_lease = 0;
  ASSERT_TRUE(space.BeginObservedFacingPoint(
      30, actor, "head", PointKind::kSourceAnchor, 1000,
      &independently_owned_lease));
  // Even a matching legacy epoch cannot be replaced while an independently
  // owned source point lease would be left behind.
  EXPECT_FALSE(space.BeginObservedSourceSetup(
      {{actor, "root", legacy_epoch}}, point, &actor_results, &point_results));
  ASSERT_TRUE(space.RetireObservedFacingPoint(30, independently_owned_lease));

  ASSERT_TRUE(space.BeginObservedSourceSetup(
      {{actor, "root", legacy_epoch}}, point, &actor_results, &point_results));
  ASSERT_GT(actor_results[0].epoch, legacy_epoch);
  // Matching the new numeric epoch is not enough: it belongs to this setup's
  // token and cannot be silently replaced by a second observed-source setup.
  EXPECT_FALSE(space.BeginObservedSourceSetup(
      {{actor, "root", actor_results[0].epoch}}, point, &actor_results,
      &point_results));
}

// Independent numeric observations, not readback of a generated avatar. The
// fixture controls one process-local clock and delivers actual source packets.
class ObservedFacingPointTest : public ::testing::Test {
 protected:
  void SetUp() override {
    source.SetSpaceId(1);
    receiver.SetSpaceId(2);
    source.SetHub(&hub);
    receiver.SetHub(&hub);
    receiver.SetReferentResolveMode(ReferentResolveMode::kLinkerOnly);
    receiver.SetRichRealizationEnabled(true);
    ASSERT_TRUE(source.AddElement(1));
    ASSERT_TRUE(receiver.AddElement(101));
    for (int i = 0; i < 2; ++i) {
      const auto anchors = std::vector<SubjectAnchor>{
          SubjectAnchor("navigation"),
          SubjectAnchor("head", Eigen::Vector3d::Zero(),
                        {StateType::kPosition3D, StateType::kRotationYaw})};
      ASSERT_TRUE(source.AddElement(Owner(i), 1));
      ASSERT_TRUE(receiver.AddElement(Owner(i) + 100, 101));
      ASSERT_TRUE(source.SetElementSharedId(Owner(i), 1000 + i));
      ASSERT_TRUE(receiver.SetElementSharedId(Owner(i) + 100, 1000 + i));
      actors[i] = source.CreateActor(anchors);
      clones[i] = receiver.CreateActor(anchors);
      ASSERT_TRUE(source.SetActorElement(actors[i], Owner(i)));
      ASSERT_TRUE(receiver.SetActorElement(clones[i], Owner(i) + 100));
      ASSERT_TRUE(source.AddElement(Point(i), 1));
      ASSERT_TRUE(receiver.AddElement(Point(i) + 100, 101));
      ASSERT_TRUE(source.SetElementTag(Point(i), "observed-peer"));
      ASSERT_TRUE(source.SetElementSharedId(Point(i), 2000 + i));
      ASSERT_TRUE(receiver.SetElementSharedId(Point(i) + 100, 2000 + i));
      ASSERT_TRUE(source.BeginActorObservationEpoch(actors[i], "navigation",
                                                    &epochs[i]));
      ASSERT_TRUE(source.BeginObservedFacingPoint(Point(i), actors[i], "head",
                                                  PointKind::kSourceAnchor,
                                                  1000, &leases[i]));
      ASSERT_TRUE(receiver.BeginObservedFacingPoint(
          Point(i) + 100, clones[i], "head", PointKind::kReceiverCounterpart,
          1000, &target_leases[i], {1, Point(i)}, leases[i]));
      hub.GetLinker().CreateLink({1, Owner(i)}, {2, Owner(i) + 100});
      hub.GetLinker().CreateLink({1, Point(i)}, {2, Point(i) + 100});
      ASSERT_EQ(hub.GetLinker().Translate({1, Owner(i)}, 2), Owner(i) + 100);
      ASSERT_EQ(hub.GetLinker().Translate({1, Point(i)}, 2), Point(i) + 100);
      ASSERT_NE(source.BindScoped(actors[i], "head",
                                  RelationKernelType::kAvatarFacing, 1,
                                  "observed-peer"),
                nullptr);
    }
    // No hub connection: explicit delivery makes receipt/order assertions
    // exact.
    // D6: no tag narrows these bindings to the observed peers: Facing is every
    // visible referent, so the peer actor's own
    // element is also a referent (as v1 treated another person). This fixture is about the
    // observed points, so it exchanges only their rows.
    hub.SetRelationUpdateCallback([this](const RemoteRelationUpdate& update) {
      packets.push_back(PeersOnly(update));
    });
  }

  static RemoteRelationUpdate PeersOnly(const RemoteRelationUpdate& update) {
    RemoteRelationUpdate peers_only = update;
    peers_only.payloads.erase(
        std::remove_if(peers_only.payloads.begin(), peers_only.payloads.end(),
                       [](const RelationPayload& p) {
                         return !p.observed_facing_point;
                       }),
        peers_only.payloads.end());
    return peers_only;
  }
  // The relation states of actor i toward the observed points only (the peer actor's
  // own element is also a referent, see SetUp).
  std::vector<RelationState> PointStates(int i) const {
    std::vector<RelationState> out;
    for (const auto& st : source.GetRelationStates(actors[i]))
      if (st.referent_element_id == Point(0) || st.referent_element_id == Point(1))
        out.push_back(st);
    return out;
  }

  static element_id_t Owner(int i) { return 10 + i; }
  static element_id_t Point(int i) { return 30 + i; }
  static Eigen::Vector3d Head(int i) { return {4.0 * i, 0, 1.6 + .2 * i}; }

  void Observe(int i, int64_t sequence) {
    ASSERT_TRUE(source.ApplyActorObservationFrame(
        actors[i], epochs[i], sequence, .1 * sequence,
        {{"navigation", {4.0 * i, 0, 0}, 0},
         {"head", Head(i), i == 0 ? 0 : M_PI}}));
  }

  void Publish(int i, int64_t sequence, int64_t deadline = 1500) {
    Observe(i, sequence);
    ASSERT_TRUE(source.CommitObservedFacingPoint(
        Point(i), leases[i], epochs[i], sequence, Head(i), now, deadline));
  }

  void MapPoint(int i, const Eigen::Vector3d& position, int64_t sequence = 0,
                int64_t deadline = 1700) {
    ASSERT_TRUE(receiver.CommitObservedFacingPoint(
        Point(i) + 100, target_leases[i], epochs[i], sequence, position, now,
        deadline));
  }

  RemoteRelationUpdate Packet(int i) const {
    for (auto it = packets.rbegin(); it != packets.rend(); ++it)
      if (it->actor_shared_id == 1000 + i) return *it;
    ADD_FAILURE() << "Missing source packet";
    return {};
  }

  ActorRealizationSnapshot Deliver(const RemoteRelationUpdate& update,
                                   int i = 0) {
    receiver.OnRemoteUpdate(update);
    ActorRealizationSnapshot snapshot;
    EXPECT_TRUE(receiver.CopyActorRealizationSnapshot(clones[i], &snapshot));
    return snapshot;
  }

  void AdmitReceiverBody(int i = 0) {
    uint64_t epoch;
    ASSERT_TRUE(receiver.BeginReceiverInputEpoch(clones[i], "navigation", 1, 1,
                                                 &epoch));
    ReceiverAnchorInput root, head;
    root.name = "navigation";
    head.name = "head";
    root.available = head.available = true;
    root.world_pose.block<3, 1>(0, 3) = Eigen::Vector3d(10, 10, 0);
    head.world_pose.block<3, 1>(0, 3) = Eigen::Vector3d(10, 10, 1.6);
    ASSERT_TRUE(receiver.ApplyReceiverInputFrame(clones[i], epoch, 1, now, 1900,
                                                 {root, head}));
  }

  std::vector<Space::ActorObservationFrame> Frames(int64_t sequence,
                                                   double error = 0.0) const {
    const Eigen::Vector3d heads[2] = {
        {.01 * sequence, .02 * sequence, 1.6},
        {4 + .015 * sequence, 2 + .01 * sequence, 1.8}};
    std::vector<Space::ActorObservationFrame> frames;
    for (int i = 0; i < 2; ++i) {
      const auto delta = heads[1 - i] - heads[i];
      const double yaw = std::atan2(delta.y(), delta.x()) + error;
      frames.push_back(
          {actors[i],
           epochs[i],
           sequence,
           sequence / 120.0,
           {{"navigation", heads[i], yaw}, {"head", heads[i], yaw}}});
    }
    return frames;
  }

  std::vector<Space::ObservedFacingPointPublication> Publications(
      int64_t age = 700) const {
    return {{Point(0), leases[0], now, now + age},
            {Point(1), leases[1], now, now + age}};
  }

  void RestartPointsWithAge(int64_t age) {
    for (int i = 0; i < 2; ++i) {
      ASSERT_TRUE(source.RetireObservedFacingPoint(Point(i), leases[i]));
      ASSERT_TRUE(source.BeginObservedFacingPoint(Point(i), actors[i], "head",
                                                  PointKind::kSourceAnchor, age,
                                                  &leases[i]));
      ASSERT_TRUE(
          receiver.RetireObservedFacingPoint(Point(i) + 100, target_leases[i]));
      ASSERT_TRUE(receiver.BeginObservedFacingPoint(
          Point(i) + 100, clones[i], "head", PointKind::kReceiverCounterpart,
          age, &target_leases[i], {1, Point(i)}, leases[i]));
    }
  }

  static const RealizationChannel* Facing(const ActorRealizationSnapshot& s) {
    for (const auto& c : s.channels)
      if (c.anchor_name == "head" &&
          c.family == RelationKernelType::kAvatarFacing)
        return &c;
    return nullptr;
  }

  void ExpectNoFacingGoal(const ActorRealizationSnapshot& snapshot) {
    const auto* channel = Facing(snapshot);
    EXPECT_TRUE(!channel || !channel->has_goal);
  }

  int64_t now = 1000;
  Space source{[this] { return now; }};
  Space receiver{[this] { return now; }};
  SpaceHub hub;
  spatialblend::entity_id_t actors[2]{}, clones[2]{};
  uint64_t epochs[2]{}, leases[2]{}, target_leases[2]{};
  std::vector<RemoteRelationUpdate> packets;
};

// Facing scope is the family's (D1): a collaborator 8 m away in line of sight, and an
// OBJECT 8 m away, are both in scope and active. There is no scope radius and no
// per-kind (object vs actor) distinction.
TEST_F(ObservedFacingPointTest, ScopeGating_E1_1_CollaboratorAndObjectAtAnyDistance) {
  const Eigen::Vector3d far_head(8.0, 0, 1.8);  // 8 m from actor 0's head
  ASSERT_TRUE(source.ApplyActorObservationFrame(
      actors[1], epochs[1], 0, 0, {{"navigation", {8.0, 0, 0}, 0},
                                   {"head", far_head, M_PI}}));
  ASSERT_TRUE(source.CommitObservedFacingPoint(Point(1), leases[1], epochs[1],
                                               0, far_head, now, 1500));
  // An object with the same tag, also 8 m away (footprint 7.8 m from the
  // head), in the other direction so it cannot occlude the collaborator.
  ASSERT_TRUE(source.AddElement(50, 1));
  ASSERT_TRUE(source.SetElementSharedId(50, 5050));  // shared: an unshared element is never a referent
  ASSERT_TRUE(source.SetElementTag(50, "observed-peer"));
  ASSERT_TRUE(source.SetElementSpatialDefinition(
      50, std::make_shared<PolytopeSpatialDefinition>(
              Eigen::Vector3d(-8.2, -0.2, 0), Eigen::Vector3d(-7.8, 0.2, 1.0))));

  Observe(0, 0);
  bool collaborator_active = false;
  bool object_in_scope = false;
  for (const auto& st : source.GetRelationStates(actors[0])) {
    if (st.referent_element_id == 50 && st.active) object_in_scope = true;
    if (st.referent_element_id == Point(1) && st.active)
      collaborator_active = true;
  }
  EXPECT_TRUE(collaborator_active) << "8 m collaborator must stay active";
  EXPECT_TRUE(object_in_scope)
      << "a visible object stays in scope at any distance";
  const auto packet = Packet(0);
  bool packet_has_collaborator = false;
  for (const auto& p : packet.payloads)
    if (p.referent_source_element_id == Point(1)) packet_has_collaborator = true;
  EXPECT_TRUE(packet_has_collaborator);
}

TEST_F(ObservedFacingPointTest, ActualSourceFramesFaceEachOtherNotSelf) {
  Publish(0, 0);
  Publish(1, 0);
  auto from_b = Packet(1);
  ASSERT_EQ(from_b.payloads.size(), 1u);
  EXPECT_EQ(from_b.payloads[0].referent_source_element_id, Point(0));
  EXPECT_TRUE(from_b.payloads[0].observed_facing_point);
  EXPECT_EQ(from_b.payloads[0].observed_logical_actor, 1000);
  EXPECT_FALSE(from_b.payloads[0].support.has_contact);
  EXPECT_NEAR(from_b.payloads[0].grounded_relation.encoded_state[0], 0, 1e-12);

  // Updating A retires A's old point, not the independently observed B point.
  Observe(0, 1);
  const auto from_a = Packet(0);
  ASSERT_EQ(from_a.payloads.size(), 1u);
  EXPECT_EQ(from_a.payloads[0].referent_source_element_id, Point(1));
  EXPECT_EQ(from_a.payloads[0].observed_logical_actor, 1001);
  EXPECT_EQ(from_a.payloads[0].observed_source_point_lease, leases[1]);
  EXPECT_EQ(from_a.source_observation_sequence, 1);
  EXPECT_FALSE(from_a.payloads[0].support.has_contact);
  EXPECT_NEAR(from_a.payloads[0].grounded_relation.encoded_state[0], 0, 1e-12);
}

TEST_F(ObservedFacingPointTest, ReceiverUsesMappedPointAndIndependentBody) {
  Publish(1, 0);
  MapPoint(1, {10, 14, 1.8});
  AdmitReceiverBody();
  Observe(0, 0);
  const auto before = Deliver(Packet(0));
  const auto* channel = Facing(before);
  ASSERT_NE(channel, nullptr);
  ASSERT_TRUE(channel->has_goal);
  EXPECT_EQ(channel->goal_kind, RealizationGoalKind::kFacingDirection);
  EXPECT_NEAR(channel->target.yaw, M_PI / 2, 1e-9);
  ASSERT_EQ(channel->contributors.size(), 1u);
  EXPECT_EQ(channel->contributors[0].resolved_local_element_id, Point(1) + 100);
  EXPECT_FALSE(channel->contributors[0].has_mapped_support);

  // The driver publishes a far embodied pose for the clone (and the source
  // body): the facing goal uses the independently observed point, never the
  // body pose, so nothing changes.
  EmbodiedPoseInput output;
  output.x = -900;
  output.y = -900;
  output.epoch = 1;
  ASSERT_EQ(receiver.PublishEmbodiedPose(clones[1], output, 0.0, 100),
            EmbodiedPublishResult::kOk);
  ASSERT_EQ(source.PublishEmbodiedPose(actors[1], output, 0.0, 100),
            EmbodiedPublishResult::kOk);
  Observe(0, 1);
  const auto after = Deliver(Packet(0));
  ASSERT_NE(Facing(after), nullptr);
  EXPECT_NEAR(Facing(after)->target.yaw, M_PI / 2, 1e-9);
  EXPECT_EQ(after.source_observation_sequence, 1);
  Space::ElementState point;
  ASSERT_TRUE(receiver.CopyElementState(Point(1) + 100, &point));
  EXPECT_TRUE((point.world_transform.block<3, 1>(0, 3) ==
               Eigen::Vector3d(10, 14, 1.8)));
}

TEST_F(ObservedFacingPointTest,
       SourceCommitRejectsForgedPoseClockAndRestamping) {
  Observe(1, 0);
  auto commit = [&](uint64_t epoch, int64_t seq, Eigen::Vector3d point,
                    int64_t receipt, int64_t deadline) {
    return source.CommitObservedFacingPoint(Point(1), leases[1], epoch, seq,
                                            point, receipt, deadline);
  };
  EXPECT_FALSE(commit(epochs[1] + 1, 0, Head(1), now, 1500));
  EXPECT_FALSE(commit(epochs[1], 1, Head(1), now, 1500));
  EXPECT_FALSE(
      commit(epochs[1], 0, Head(1) + Eigen::Vector3d::UnitZ(), now, 1500));
  EXPECT_FALSE(commit(epochs[1], 0, Head(1), now + 1, 1500));
  EXPECT_FALSE(commit(epochs[1], 0, Head(1), now, now));
  EXPECT_FALSE(commit(epochs[1], 0, Head(1), now, now + 1001));
  EXPECT_FALSE(commit(epochs[1], 0,
                      {0, 0, std::numeric_limits<double>::quiet_NaN()}, now,
                      1500));
  ++now;
  EXPECT_FALSE(commit(epochs[1], 0, Head(1), now, 1500));
  ASSERT_TRUE(commit(epochs[1], 0, Head(1), 1000, 1500));
  EXPECT_FALSE(commit(epochs[1], 0, Head(1), 1000, 1500));
  ASSERT_TRUE(source.ClearObservedFacingPoint(Point(1), leases[1]));
  EXPECT_FALSE(commit(epochs[1], 0, Head(1), 1000, 1500));
  Observe(1, 1);
  ASSERT_TRUE(commit(epochs[1], 1, Head(1), now, 1500));
  ASSERT_TRUE(source.ClearActorObservationAnchor(actors[1], epochs[1], "head"));
  Observe(0, 0);
  EXPECT_TRUE(Packet(0).payloads.empty());
}

TEST_F(ObservedFacingPointTest, PointCannotBecomeSeatSpacingOrActorGeometry) {
  Publish(1, 0);
  for (auto kernel : {RelationKernelType::kAvatarSpacing,
                      RelationKernelType::kAvatarAffordancePose})
    ASSERT_NE(source.BindScoped(actors[0], "navigation", kernel, Point(1)),
              nullptr);
  Observe(0, 0);
  const auto packet = Packet(0);
  ASSERT_EQ(packet.payloads.size(), 1u);
  EXPECT_EQ(packet.payloads[0].kernel_type, RelationKernelType::kAvatarFacing);
  EXPECT_FALSE(source.AddElement(999, Point(1)));
  EXPECT_FALSE(
      source.SetElementTransform(Point(1), Eigen::Matrix4d::Identity()));
  EXPECT_FALSE(source.SetElementTag(Point(1), "Sittable"));
  EXPECT_FALSE(source.SetElementSharedId(Point(1), 555));
  EXPECT_FALSE(
      source.AddElementAnchor(Point(1), "seat", Eigen::Matrix4d::Identity()));
  EXPECT_FALSE(source.SetActorElement(actors[0], Point(1)));
  EXPECT_FALSE(source.SetActorElement(actors[0], 1));
  auto surface = std::make_shared<SurfaceSpatialDefinition>(
      std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
          {0, 0, .5}, {1, 0, .5}, {1, 1, .5}, {0, 1, .5}}));
  EXPECT_FALSE(source.SetElementSpatialDefinition(Point(1), surface));
}

TEST_F(ObservedFacingPointTest,
       RejectsUnqualifiedRegistrationAndActorAncestry) {
  ASSERT_TRUE(source.AddElement(90, Owner(0)));
  uint64_t token = 123;
  EXPECT_FALSE(source.BeginObservedFacingPoint(
      90, actors[0], "head", PointKind::kSourceAnchor, 1000, &token));
  EXPECT_EQ(token, 0u);
  ASSERT_TRUE(source.AddElement(91, 1));
  EXPECT_FALSE(source.BeginObservedFacingPoint(
      91, actors[0], "missing", PointKind::kSourceAnchor, 1000, &token));
  EXPECT_FALSE(source.BeginObservedFacingPoint(
      91, actors[0], "head", PointKind::kSourceAnchor, 0, &token));
  EXPECT_FALSE(source.BeginObservedFacingPoint(91, actors[0], "head",
                                               PointKind::kReceiverCounterpart,
                                               1000, &token, {1, 30}, 1));
  EXPECT_FALSE(receiver.BeginObservedFacingPoint(
      131, clones[1], "head", PointKind::kSourceAnchor, 1000, &token));
  ASSERT_TRUE(source.SetElementSharedId(Owner(0), kInvalidSharedElementId));
  EXPECT_FALSE(source.BeginObservedFacingPoint(
      91, actors[0], "head", PointKind::kSourceAnchor, 1000, &token));
}

TEST_F(ObservedFacingPointTest,
       IncomingPointRequiresCapabilityExactOriginAndFamily) {
  Publish(1, 0);
  MapPoint(1, {10, 14, 1.8});
  AdmitReceiverBody();
  auto bad = [&](const std::function<void(RelationPayload&)>& mutate, int seq) {
    Observe(0, seq);
    auto packet = Packet(0);
    ASSERT_EQ(packet.payloads.size(), 1u);
    mutate(packet.payloads[0]);
    ExpectNoFacingGoal(Deliver(packet));
  };
  bad([](auto& p) { p.observed_facing_point = false; }, 0);
  bad([](auto& p) { ++p.observed_source_point_lease; }, 1);
  bad([](auto& p) { ++p.observed_source_point.element_id; }, 2);
  bad([](auto& p) { p.observed_logical_actor = 1000; }, 3);
  bad([](auto& p) { p.observed_valid_until_ns = 1000; }, 4);
  bad(
      [](auto& p) {
        p.kernel_type = RelationKernelType::kAvatarAffordancePose;
      },
      5);
  Observe(0, 6);
  const auto accepted = Deliver(Packet(0));
  ASSERT_NE(Facing(accepted), nullptr);
  EXPECT_TRUE(Facing(accepted)->has_goal);
}

TEST_F(ObservedFacingPointTest,
       DifferentProxyHandleCannotDefeatLogicalSelfExclusion) {
  Publish(1, 0);
  Observe(0, 0);
  auto packet = Packet(0);
  ASSERT_EQ(packet.payloads.size(), 1u);
  // A distinct local element/lease still represents this receiving subject.
  MapPoint(0, {10, 14, 1.8});
  packet.payloads[0].referent_source_element_id = Point(0);
  packet.payloads[0].observed_source_point = {1, Point(0)};
  packet.payloads[0].observed_source_point_lease = leases[0];
  packet.payloads[0].observed_logical_actor = 1000;
  AdmitReceiverBody();
  ExpectNoFacingGoal(Deliver(packet));
}

TEST_F(ObservedFacingPointTest,
       OriginalSourceDeadlineWithdrawsHeldGoalWhileIdle) {
  Publish(1, 0, 1200);
  MapPoint(1, {10, 14, 1.8}, 0, 1700);
  AdmitReceiverBody();
  Observe(0, 0);
  const auto before = Deliver(Packet(0));
  ASSERT_NE(Facing(before), nullptr);
  ASSERT_TRUE(Facing(before)->has_goal);
  ASSERT_NE(receiver.GetRealizationTargets(clones[0]), nullptr);
  const auto deliveries = packets.size();
  now = 1200;
  EXPECT_EQ(receiver.GetRealizationTargets(clones[0]), nullptr);
  {
    // Any locked read of the space withdraws the expired point.
    Eigen::Vector2d position;
    double yaw = 0.0;
    receiver.CopyActorPose(clones[0], &position, &yaw);
  }
  ActorRealizationValidity validity;
  const uint32_t domains =
      kRealizationTreeRevision | kRealizationSupportRevision;
  ASSERT_TRUE(receiver.GetActorRealizationValidity(clones[0], before.revision,
                                                   domains, &validity));
  EXPECT_EQ(validity.changed_mask & domains, domains);
  EXPECT_EQ(packets.size(), deliveries);
  ActorRealizationSnapshot historical;
  ASSERT_TRUE(receiver.CopyActorRealizationSnapshot(clones[0], &historical));
  EXPECT_EQ(historical.revision, before.revision);
  EXPECT_EQ(historical.source_observation_sequence, 0);
  EXPECT_EQ(historical.receiver_ingress_monotonic_ns,
            before.receiver_ingress_monotonic_ns);
  Observe(0, 1);
  EXPECT_TRUE(Packet(0).payloads.empty());
}

TEST_F(ObservedFacingPointTest,
       CounterpartChangeClearsHeldGoalWithoutSourceReplay) {
  Publish(1, 0);
  MapPoint(1, {10, 14, 1.8});
  AdmitReceiverBody();
  Observe(0, 0);
  const auto before = Deliver(Packet(0));
  ASSERT_TRUE(Facing(before) && Facing(before)->has_goal);
  const auto deliveries = packets.size();
  MapPoint(1, {14, 10, 1.8}, 1);
  const auto* targets = receiver.GetRealizationTargets(clones[0]);
  EXPECT_TRUE(!targets || targets->targets.empty());
  EXPECT_EQ(packets.size(), deliveries);
  Observe(0, 1);
  const auto after = Deliver(Packet(0));
  ASSERT_NE(Facing(after), nullptr);
  EXPECT_NEAR(Facing(after)->target.yaw, 0, 1e-9);
}

TEST_F(ObservedFacingPointTest,
       RetiredSourceGenerationRequiresNewExplicitCounterpart) {
  Publish(1, 0);
  MapPoint(1, {10, 14, 1.8});
  AdmitReceiverBody();
  const auto old = leases[1];
  ASSERT_TRUE(source.RetireObservedFacingPoint(Point(1), old));
  ASSERT_TRUE(source.BeginObservedFacingPoint(
      Point(1), actors[1], "head", PointKind::kSourceAnchor, 1000, &leases[1]));
  EXPECT_NE(leases[1], old);
  EXPECT_FALSE(source.ClearObservedFacingPoint(Point(1), old));
  EXPECT_FALSE(source.CommitObservedFacingPoint(Point(1), old, epochs[1], 0,
                                                Head(1), now, 1500));
  ASSERT_TRUE(source.CommitObservedFacingPoint(Point(1), leases[1], epochs[1],
                                               0, Head(1), now, 1500));
  Observe(0, 0);
  ExpectNoFacingGoal(Deliver(Packet(0)));
  ASSERT_TRUE(
      receiver.RetireObservedFacingPoint(Point(1) + 100, target_leases[1]));
  ASSERT_TRUE(receiver.BeginObservedFacingPoint(
      Point(1) + 100, clones[1], "head", PointKind::kReceiverCounterpart, 1000,
      &target_leases[1], {1, Point(1)}, leases[1]));
  MapPoint(1, {10, 14, 1.8});
  Observe(0, 1);
  const auto after = Deliver(Packet(0));
  ASSERT_NE(Facing(after), nullptr);
  EXPECT_TRUE(Facing(after)->has_goal);
}

TEST_F(ObservedFacingPointTest,
       ParentMotionAndProducerLossWithdrawObservedPosition) {
  Publish(1, 0);
  Observe(0, 0);
  ASSERT_EQ(Packet(0).payloads.size(), 1u);
  Eigen::Matrix4d moved = Eigen::Matrix4d::Identity();
  moved(0, 3) = 10;
  ASSERT_TRUE(source.SetElementTransform(1, moved));
  Observe(0, 1);
  EXPECT_TRUE(Packet(0).payloads.empty());
  Publish(1, 1);
  Space::ElementState point;
  ASSERT_TRUE(source.CopyElementState(Point(1), &point));
  EXPECT_TRUE((point.world_transform.block<3, 1>(0, 3) == Head(1)));
  Observe(0, 2);
  ASSERT_EQ(Packet(0).payloads.size(), 1u);
  ASSERT_TRUE(source.DestroyActor(actors[1]));
  Observe(0, 3);
  EXPECT_TRUE(Packet(0).payloads.empty());
  // A withdrawn point remains special; it cannot silently become a static one.
  EXPECT_FALSE(source.SetElementTag(Point(1), "static-peer"));
}

TEST_F(ObservedFacingPointTest, ReattachedSourceOwnerStartsNewRelationPhase) {
  std::vector<RelationEvent> events;
  source.RegisterOnRelationEvent([&](const RelationEvent& event) {
    if (event.actor_shared_id == 1000 && event.anchor_name == "head")
      events.push_back(event);
  });
  Publish(1, 0);
  Observe(0, 0);
  Observe(0, 1);
  ASSERT_FALSE(events.empty());
  ASSERT_EQ(events.back().to, RelationPhase::kContact);
  ASSERT_TRUE(hub.GetLinker().RemoveLink({1, Owner(0)}, {2, Owner(0) + 100}));
  ASSERT_TRUE(source.RemoveLocalElementSubtree(Owner(0)));
  ASSERT_TRUE(source.AddElement(Owner(0), 1));
  ASSERT_TRUE(source.SetElementSharedId(Owner(0), 1000));
  ASSERT_TRUE(source.SetActorElement(actors[0], Owner(0)));
  ASSERT_NE(
      source.BindScoped(actors[0], "head", RelationKernelType::kAvatarFacing, 1,
                        "observed-peer"),
      nullptr);
  Observe(0, 2);
  ASSERT_FALSE(events.empty());
  EXPECT_EQ(events.back().from, RelationPhase::kDormant);
  EXPECT_EQ(events.back().to, RelationPhase::kActive);
}

TEST_F(ObservedFacingPointTest, CounterpartReplacementClearsReceiverOnlyPhase) {
  std::vector<RelationEvent> events;
  receiver.RegisterOnRelationEvent([&](const RelationEvent& event) {
    if (event.actor_shared_id == 1000 && event.anchor_name == "head")
      events.push_back(event);
  });
  Publish(1, 0);
  MapPoint(1, {10, 14, 1.8});
  AdmitReceiverBody();
  Observe(0, 0);
  auto first = Packet(0);
  first.payloads[0].referent_shared_id = 777;
  Deliver(first);
  Observe(0, 1);
  auto second = Packet(0);
  second.payloads[0].referent_shared_id = 777;
  Deliver(second);
  ASSERT_FALSE(events.empty());
  ASSERT_EQ(events.back().to, RelationPhase::kContact);
  // No local source bindings exist for this clone. Its phase keys come from
  // incoming payload identity, not the counterpart's local/shared element ID.
  ASSERT_TRUE(
      receiver.RetireObservedFacingPoint(Point(1) + 100, target_leases[1]));
  ASSERT_TRUE(receiver.BeginObservedFacingPoint(
      Point(1) + 100, clones[1], "head", PointKind::kReceiverCounterpart, 1000,
      &target_leases[1], {1, Point(1)}, leases[1]));
  MapPoint(1, {14, 10, 1.8});
  Observe(0, 2);
  auto third = Packet(0);
  third.payloads[0].referent_shared_id = 777;
  Deliver(third);
  EXPECT_EQ(events.back().from, RelationPhase::kDormant);
  EXPECT_EQ(events.back().to, RelationPhase::kActive);
}

TEST_F(ObservedFacingPointTest, AuthoritativeOmissionClearsReceiverPointPhase) {
  std::vector<RelationEvent> events;
  receiver.RegisterOnRelationEvent([&](const RelationEvent& event) {
    if (event.actor_shared_id == 1000 && event.anchor_name == "head")
      events.push_back(event);
  });
  Publish(1, 0);
  MapPoint(1, {10, 14, 1.8});
  AdmitReceiverBody();
  Observe(0, 0);
  Deliver(Packet(0));
  Observe(0, 1);
  Deliver(Packet(0));
  ASSERT_FALSE(events.empty());
  ASSERT_EQ(events.back().to, RelationPhase::kContact);
  ASSERT_TRUE(source.ClearActorObservationAnchor(actors[1], epochs[1], "head"));
  Observe(0, 2);
  ASSERT_TRUE(Packet(0).payloads.empty());
  ExpectNoFacingGoal(Deliver(Packet(0)));
  Publish(1, 1);
  Observe(0, 3);
  Deliver(Packet(0));
  EXPECT_EQ(events.back().from, RelationPhase::kDormant);
  EXPECT_EQ(events.back().to, RelationPhase::kActive);
}

TEST_F(ObservedFacingPointTest, NewSourceSpaceCannotReuseAnOldPointCapability) {
  Publish(1, 0);
  MapPoint(1, {10, 14, 1.8});
  AdmitReceiverBody();
  // Another incarnation reuses every public numeric ID and starts epoch one.
  // The receiver still holds the old point's explicit binding and no A packet.
  Space replacement{[this] { return now; }};
  replacement.SetSpaceId(1);
  replacement.SetHub(&hub);
  ASSERT_TRUE(replacement.AddElement(1));
  spatialblend::entity_id_t new_actors[2];
  uint64_t new_epochs[2], new_leases[2];
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(replacement.AddElement(Owner(i), 1));
    ASSERT_TRUE(replacement.SetElementSharedId(Owner(i), 1000 + i));
    new_actors[i] = replacement.CreateActor(
        {SubjectAnchor("navigation"), SubjectAnchor("head")});
    ASSERT_TRUE(replacement.SetActorElement(new_actors[i], Owner(i)));
    ASSERT_TRUE(replacement.BeginActorObservationEpoch(
        new_actors[i], "navigation", &new_epochs[i]));
    ASSERT_TRUE(replacement.AddElement(Point(i), 1));
    ASSERT_TRUE(replacement.SetElementSharedId(Point(i), 2000 + i));
    ASSERT_TRUE(replacement.BeginObservedFacingPoint(
        Point(i), new_actors[i], "head", PointKind::kSourceAnchor, 1000,
        &new_leases[i]));
    EXPECT_NE(new_leases[i], leases[i]);
  }
  ASSERT_TRUE(replacement.ApplyActorObservationFrame(
      new_actors[1], new_epochs[1], 0, 0, {{"head", Head(1), 0}}));
  ASSERT_TRUE(replacement.CommitObservedFacingPoint(
      Point(1), new_leases[1], new_epochs[1], 0, Head(1), now, 1500));
  ASSERT_NE(replacement.BindScoped(new_actors[0], "head",
                                   RelationKernelType::kAvatarFacing, Point(1)),
            nullptr);
  ASSERT_TRUE(replacement.ApplyActorObservationFrame(
      new_actors[0], new_epochs[0], 0, 0, {{"head", Head(0), 0}}));
  const auto packet = Packet(0);
  // Facing is global, so the collaborator
  // is a referent beside the point. The point capability itself is the new
  // lease: the replacement space does not reuse the old one.
  const RelationPayload* point = nullptr;
  int pointed = 0;
  for (const auto& payload : packet.payloads) {
    if (!payload.observed_facing_point) continue;
    ++pointed;
    point = &payload;
  }
  ASSERT_EQ(pointed, 1);
  ASSERT_NE(point, nullptr);
  EXPECT_EQ(point->observed_source_point_lease, new_leases[1]);
  EXPECT_NE(point->observed_source_point_lease, leases[1]);
  const auto delivered = Deliver(packet);
  const auto* channel = Facing(delivered);
  if (channel) {
    for (const auto& contributor : channel->contributors) {
      // A rejected row may name the point. An included one would mean the
      // replacement's lease satisfied the receiver's old counterpart.
      if (contributor.referent_id == static_cast<shared_element_id_t>(2001)) {
        EXPECT_NE(contributor.disposition, RealizationDisposition::kIncluded);
      }
    }
  }
}

TEST_F(ObservedFacingPointTest, SpaceIdentityRoundTripCannotReviveOldLease) {
  Publish(1, 0);
  source.SetSpaceId(9);
  source.SetSpaceId(1);
  EXPECT_FALSE(source.CommitObservedFacingPoint(Point(1), leases[1], epochs[1],
                                                0, Head(1), now, 1500));
  Observe(0, 0);
  EXPECT_TRUE(Packet(0).payloads.empty());
  const auto old = leases[1];
  ASSERT_TRUE(source.BeginObservedFacingPoint(
      Point(1), actors[1], "head", PointKind::kSourceAnchor, 1000, &leases[1]));
  EXPECT_NE(leases[1], old);
  EXPECT_FALSE(source.ClearObservedFacingPoint(Point(1), old));
  ASSERT_TRUE(source.CommitObservedFacingPoint(Point(1), leases[1], epochs[1],
                                               0, Head(1), now, 1500));
  Observe(0, 1);
  ASSERT_EQ(Packet(0).payloads.size(), 1u);
}

TEST_F(ObservedFacingPointTest,
       LocalCounterpartDeadlineCannotBeRefreshedBySource) {
  Publish(1, 0, 1700);
  MapPoint(1, {10, 14, 1.8}, 0, 1200);
  AdmitReceiverBody();
  Observe(0, 0);
  const auto before = Deliver(Packet(0));
  ASSERT_TRUE(Facing(before) && Facing(before)->has_goal);
  now = 1200;
  EXPECT_EQ(receiver.GetRealizationTargets(clones[0]), nullptr);
  receiver.PruneObservedFacingPoints();
  Observe(0, 1);
  ASSERT_EQ(Packet(0).payloads.size(), 1u);
  ExpectNoFacingGoal(Deliver(Packet(0)));
  // Reusing the old counterpart sample cannot manufacture a fresh deadline.
  EXPECT_FALSE(receiver.CommitObservedFacingPoint(Point(1) + 100,
                                                  target_leases[1], epochs[1],
                                                  0, {10, 14, 1.8}, now, 1700));
}

TEST_F(ObservedFacingPointTest, LegacyOverridesCannotImpersonateAtomicInput) {
  Observe(1, 0);
  const Eigen::Vector3d changed = Head(1) + Eigen::Vector3d::UnitX();
  source.SetAnchorWorldPose(actors[1], "head", changed, 0);
  EXPECT_FALSE(source.CommitObservedFacingPoint(Point(1), leases[1], epochs[1],
                                                0, changed, now, 1500));
  // Restoring identical coordinates is still a legacy mutation, not admission.
  source.SetAnchorWorldPose(actors[1], "head", Head(1), 0);
  EXPECT_FALSE(source.CommitObservedFacingPoint(Point(1), leases[1], epochs[1],
                                                0, Head(1), now, 1500));
  Publish(1, 1);
  Observe(0, 0);
  ASSERT_EQ(Packet(0).payloads.size(), 1u);
  ASSERT_TRUE(source.SetAnchorActive(actors[1], "head", false));
  ASSERT_TRUE(source.SetAnchorActive(actors[1], "head", true));
  Observe(0, 1);
  EXPECT_TRUE(Packet(0).payloads.empty());
  Publish(1, 2);
  source.ClearAnchorWorldPose(actors[1], "head");
  source.SetAnchorWorldPose(actors[1], "head", Head(1), 0);
  Observe(0, 2);
  EXPECT_TRUE(Packet(0).payloads.empty());
  Publish(1, 3);
  Observe(0, 3);
  ASSERT_EQ(Packet(0).payloads.size(), 1u);
}

TEST_F(ObservedFacingPointTest,
       LegacySubjectUpdateCannotRepublishHeldObservation) {
  Publish(1, 0);
  Observe(0, 0);
  ASSERT_EQ(Packet(0).payloads.size(), 1u);
  source.UpdateActorState(actors[0], {99, 99}, 1.2);
  // A legacy update carries no observed point: it must not republish the held
  // observation. (It may carry the peer actor's own element as a referent, which is not
  // an observed point and is not kept here, see SetUp.)
  EXPECT_TRUE(Packet(0).payloads.empty());
  EXPECT_TRUE(PointStates(0).empty());
  uint64_t epoch;
  int64_t sequence;
  double time;
  bool has_sample;
  ASSERT_TRUE(source.GetActorObservationClock(actors[0], &epoch, &sequence,
                                              &time, &has_sample));
  EXPECT_EQ(epoch, epochs[0]);
  EXPECT_EQ(sequence, 0);
  EXPECT_DOUBLE_EQ(time, 0);
  EXPECT_TRUE(has_sample);
  Observe(0, 1);
  ASSERT_EQ(Packet(0).payloads.size(), 1u);
  EXPECT_TRUE(Packet(0).source_observation_sequence_known);
  EXPECT_EQ(Packet(0).source_observation_sequence, 1);
}

TEST_F(ObservedFacingPointTest,
       BatchPublishesAllFreshPeersBeforeEitherExtraction) {
  for (int64_t sequence = 0; sequence < 3; ++sequence) {
    auto frames = Frames(sequence);
    if (sequence % 2) std::reverse(frames.begin(), frames.end());
    ASSERT_TRUE(source.ApplyActorObservationBatch(frames, Publications()));
    ASSERT_EQ(packets.size(), 2u * (sequence + 1));
    for (int i = 0; i < 2; ++i) {
      const auto packet = Packet(i);
      ASSERT_EQ(packet.payloads.size(), 1u);
      EXPECT_EQ(packet.source_observation_sequence, sequence);
      EXPECT_EQ(packet.source_time, sequence / 120.0);
      EXPECT_EQ(packet.payloads[0].referent_source_element_id, Point(1 - i));
      EXPECT_NEAR(packet.payloads[0].grounded_relation.encoded_state[0], 0,
                  1e-12);
      const auto expected = Frames(sequence)[1 - i].observations[1].position;
      EXPECT_TRUE((packet.payloads[0].support.world_transform.block<3, 1>(
                       0, 3) == expected));
      EXPECT_EQ(packet.payloads[0].observed_valid_until_ns, now + 700);
    }
    ++now;
  }
}

TEST_F(ObservedFacingPointTest,
       RejectedBatchCannotPartiallyAdmitAnyFrameOrPoint) {
  ASSERT_TRUE(source.ApplyActorObservationBatch(Frames(0), Publications()));
  auto reject = [&](auto change) {
    auto frames = Frames(1);
    auto publications = Publications();
    change(frames, publications);
    EXPECT_FALSE(source.ApplyActorObservationBatch(frames, publications));
    EXPECT_EQ(packets.size(), 2u);
    for (int i = 0; i < 2; ++i) {
      uint64_t epoch;
      int64_t sequence;
      double time;
      bool has_sample;
      ASSERT_TRUE(source.GetActorObservationClock(actors[i], &epoch, &sequence,
                                                  &time, &has_sample));
      EXPECT_EQ(epoch, epochs[i]);
      EXPECT_EQ(sequence, 0);
      EXPECT_EQ(time, 0.0);
      EXPECT_TRUE(has_sample);
      Space::ElementState point;
      ASSERT_TRUE(source.CopyElementState(Point(i), &point));
      EXPECT_TRUE((point.world_transform.block<3, 1>(0, 3) ==
                   Frames(0)[i].observations[1].position));
    }
  };
  reject([](auto& f, auto&) { f[1].sequence = 0; });
  reject([](auto& f, auto&) { ++f[1].epoch; });
  reject([](auto& f, auto&) { f[1] = f[0]; });
  reject([](auto& f, auto&) {
    f[1].observations[1].position.x() =
        std::numeric_limits<double>::quiet_NaN();
  });
  reject([](auto& f, auto&) { f[1].observations.pop_back(); });
  reject([](auto& f, auto&) { f.pop_back(); });
  reject([](auto&, auto& p) { ++p[1].lease; });
  reject([](auto&, auto& p) { p[1] = p[0]; });
  reject([&](auto&, auto& p) { p[1].valid_until_ns = now; });
  reject([&](auto&, auto& p) { p[1].observed_at_ns = now + 1; });
  EXPECT_FALSE(source.ApplyActorObservationBatch({}, {}));
  ASSERT_TRUE(source.ApplyActorObservationBatch(Frames(1), Publications()));
  EXPECT_EQ(packets.size(), 4u);
}

TEST_F(ObservedFacingPointTest,
       PointBatchRejectsAtomicallyAndNeverEvaluatesSubjects) {
  std::vector<Space::ObservedFacingPointUpdate> updates = {{Point(0) + 100,
                                                            target_leases[0],
                                                            epochs[0],
                                                            0,
                                                            {10, 10, 1.6},
                                                            now,
                                                            now + 700},
                                                           {Point(1) + 100,
                                                            target_leases[1],
                                                            epochs[1],
                                                            0,
                                                            {10, 14, 1.8},
                                                            now,
                                                            now + 700}};
  const auto valid = updates;
  ++updates[1].lease;
  EXPECT_FALSE(receiver.CommitObservedFacingPoints(updates));
  updates = valid;
  updates[1] = updates[0];
  EXPECT_FALSE(receiver.CommitObservedFacingPoints(updates));
  EXPECT_FALSE(receiver.CommitObservedFacingPoints({}));
  ASSERT_TRUE(receiver.CommitObservedFacingPoints(valid));
  EXPECT_TRUE(packets.empty());
  EXPECT_FALSE(receiver.CommitObservedFacingPoints(valid));
}

TEST_F(ObservedFacingPointTest,
       ContinuousCounterpartDoesNotReevaluateHeldSubject) {
  Publish(1, 0);
  MapPoint(1, {10, 14, 1.8});
  AdmitReceiverBody();
  Observe(0, 0);
  const auto first = Deliver(Packet(0));
  ASSERT_TRUE(Facing(first) && Facing(first)->has_goal);
  MapPoint(1, {14, 10, 1.8}, 1);
  ReceiverAnchorInput root, head;
  root.name = "navigation";
  head.name = "head";
  root.available = head.available = true;
  root.world_pose.block<3, 1>(0, 3) = Eigen::Vector3d(10, 10, 0);
  head.world_pose.block<3, 1>(0, 3) = Eigen::Vector3d(10, 10, 1.6);
  ASSERT_TRUE(receiver.ApplyReceiverInputFrame(
      clones[0], first.receiver_input.epoch, 2, now, 1900, {root, head}));
  EXPECT_EQ(receiver.ReevaluateReceiverRealization(
                clones[0], first.receiver_input.epoch, 2),
            Space::ReceiverReevaluationResult::kUnavailable);
  ActorRealizationSnapshot historical;
  ASSERT_TRUE(receiver.CopyActorRealizationSnapshot(clones[0], &historical));
  EXPECT_EQ(historical.revision, first.revision);
  EXPECT_EQ(historical.source_observation_sequence, 0);
  Observe(0, 1);
  const auto fresh = Deliver(Packet(0));
  ASSERT_TRUE(Facing(fresh) && Facing(fresh)->has_goal);
  EXPECT_NEAR(Facing(fresh)->target.yaw, 0, 1e-9);
  EXPECT_EQ(fresh.source_observation_sequence, 1);
}

TEST_F(ObservedFacingPointTest,
       Coherent120HzInputPreservesProgressAndBothFreshResults) {
  constexpr int64_t age = 80000000;
  RestartPointsWithAge(100000000);
  uint64_t input_epochs[2];
  for (int i = 0; i < 2; ++i)
    ASSERT_TRUE(receiver.BeginReceiverInputEpoch(clones[i], "navigation", 1, 1,
                                                 &input_epochs[i]));
  std::vector<RelationEvent> source_events, target_events;
  source.RegisterOnRelationEvent([&](const RelationEvent& e) {
    // The observed points only (the peer actor's own element is a referent too).
    if (e.referent_shared_id >= 2000) source_events.push_back(e);
  });
  receiver.RegisterOnRelationEvent(
      [&](const RelationEvent& e) { target_events.push_back(e); });
  ActorRealizationSnapshot snapshots[2];
  for (int64_t sequence = 0; sequence < 120; ++sequence) {
    now = 1000000000 + sequence * 8333333;
    const double error = .4 - .002 * sequence;
    const Eigen::Vector3d positions[2] = {{10 + sequence * .006, 10, 1.6},
                                          {10, 14 + sequence * .005, 1.8}};
    std::vector<Space::ObservedFacingPointUpdate> points;
    for (int i = 0; i < 2; ++i) {
      ReceiverAnchorInput root, head;
      root.name = "navigation";
      head.name = "head";
      root.available = head.available = true;
      root.world_pose.block<3, 1>(0, 3) = positions[i];
      root.world_pose(2, 3) = 0;
      head.world_pose.block<3, 1>(0, 3) = positions[i];
      ASSERT_TRUE(receiver.ApplyReceiverInputFrame(
          clones[i], input_epochs[i], sequence + 1, now, now + 100000000,
          {root, head}));
      points.push_back({Point(i) + 100, target_leases[i], epochs[i], sequence,
                        positions[i], now, now + age});
    }
    ASSERT_TRUE(receiver.CommitObservedFacingPoints(points));
    auto frames = Frames(sequence, error);
    if (sequence % 2) std::reverse(frames.begin(), frames.end());
    ASSERT_TRUE(source.ApplyActorObservationBatch(frames, Publications(age)));
    for (int i = 0; i < 2; ++i) {
      const auto packet = Packet(i);
      ASSERT_EQ(packet.payloads.size(), 1u);
      // The Facing count is v1 Angular1D's: the rotation of the bearing offset
      // still to go, over this update's rotation. The first update
      // has no earlier offset, so its count is infinite and v1 makes no yaw
      // request; every later update rotates the offset, so the count is finite
      // and the request exists. The κ-clipped time-to-contact is not the score.
      const bool first_update = sequence == 0;
      EXPECT_EQ(std::isfinite(packet.payloads[0].update_count), !first_update);
      snapshots[i] = Deliver(packet, i);
      if (first_update) {
        ExpectNoFacingGoal(snapshots[i]);
      } else {
        ASSERT_TRUE(Facing(snapshots[i]) && Facing(snapshots[i])->has_goal);
      }
      EXPECT_EQ(snapshots[i].source_observation_sequence, sequence);
      EXPECT_TRUE(snapshots[i].source_time_known);
      EXPECT_DOUBLE_EQ(snapshots[i].source_time_seconds, sequence / 120.0);
      if (first_update) continue;
      const auto delta = positions[1 - i] - positions[i];
      // The yaw is a ratio continuation: from the head's current yaw (0, its receiver
      // input pose) it turns 1/n of its own remaining offset to its look-at point (the peer); n = 1 places it at the
      // bearing plus the source's error.
      const double bearing = std::atan2(delta.y(), delta.x());
      const double n = packet.payloads[0].update_count;
      // The head's own offset o turns in the source's rotation direction s by 1/n of its remaining turn: the angle o
      // travels in direction s back to 0, on the 2 pi branch nearest the source's own remaining turn (its current
      // offset's travel, times n / (n - 1); any branch). n <= 1 places it at the bearing plus the
      // source's error.
      const double o = std::remainder(bearing, 2 * M_PI);
      const int s = packet.payloads[0].facing_rotation_sign;
      const auto travel = [](double off, int dir) {
        return std::fmod(std::fmod(-dir * off, 2 * M_PI) + 2 * M_PI, 2 * M_PI);
      };
      const double r_src = travel(packet.payloads[0].grounded_relation.encoded_state(0), s) * n / (n - 1.0);
      const double d0 = travel(o, s);
      const double offset_move = s * (d0 + 2 * M_PI * std::round((r_src - d0) / (2 * M_PI)));
      const double expected = !(n > 1.0) ? bearing + error : -offset_move / n;
      EXPECT_NEAR(std::remainder(Facing(snapshots[i])->target.yaw - expected, 2 * M_PI), 0, 1e-9);
    }
    // Check A again AFTER B's delivery. Neither actor invalidates its peer.
    for (int i = 0; i < 2; ++i) {
      ActorRealizationValidity validity;
      const uint32_t domains =
          kRealizationTreeRevision | kRealizationSupportRevision;
      ASSERT_TRUE(receiver.GetActorRealizationValidity(
          clones[i], snapshots[i].revision, domains, &validity));
      EXPECT_EQ(validity.captured_known_mask & domains, domains);
      EXPECT_EQ(validity.current_known_mask & domains, domains);
      EXPECT_EQ(validity.changed_mask & domains, 0u);
      EXPECT_EQ(validity.unchecked_mask & domains, 0u);
      ActorReceiverInputValidity body_validity;
      ASSERT_TRUE(receiver.GetActorReceiverInputValidity(
          clones[i], snapshots[i].revision, &body_validity));
      EXPECT_TRUE(body_validity.valid);
      ASSERT_NE(receiver.GetRealizationTargets(clones[i]), nullptr);
      EXPECT_FALSE(receiver.GetRealizationTargets(clones[i])->targets.empty());
    }
  }
  EXPECT_EQ(packets.size(), 240u);
  EXPECT_EQ(source_events.size(),
            4u);  // Active then angular Contact per actor.
  EXPECT_EQ(target_events.size(), 4u);
  now += age;  // Exclusive deadline, with no new input or manufactured sample.
  for (int i = 0; i < 2; ++i)
    EXPECT_EQ(receiver.GetRealizationTargets(clones[i]), nullptr);
  receiver.PruneObservedFacingPoints();
  EXPECT_EQ(packets.size(), 240u);
}

TEST_F(ObservedFacingPointTest,
       BatchGapMissingPublicationAndExpiryResetHistory) {
  std::vector<RelationEvent> events;
  source.RegisterOnRelationEvent([&](const RelationEvent& e) {
    if (e.actor_shared_id == 1000) events.push_back(e);
  });
  ASSERT_TRUE(source.ApplyActorObservationBatch(Frames(0), Publications()));
  ASSERT_TRUE(source.ApplyActorObservationBatch(Frames(1), Publications()));
  ASSERT_EQ(events.back().to, RelationPhase::kContact);
  ASSERT_TRUE(source.ApplyActorObservationBatch(Frames(3), Publications()));
  EXPECT_EQ(events.back().from, RelationPhase::kDormant);
  EXPECT_EQ(events.back().to, RelationPhase::kActive);
  ASSERT_TRUE(source.ApplyActorObservationBatch(Frames(4), Publications()));
  ASSERT_EQ(events.back().to, RelationPhase::kContact);
  auto frames = Frames(5);
  frames[1].observations.pop_back();
  auto publications = Publications();
  publications.pop_back();
  ASSERT_TRUE(source.ApplyActorObservationBatch(frames, publications));
  EXPECT_TRUE(Packet(0).payloads.empty());
  ASSERT_TRUE(source.ApplyActorObservationBatch(Frames(6), Publications()));
  EXPECT_EQ(events.back().from, RelationPhase::kDormant);
  EXPECT_EQ(events.back().to, RelationPhase::kActive);
  ASSERT_TRUE(source.ApplyActorObservationBatch(Frames(7), Publications()));
  now += 700;
  ASSERT_TRUE(source.ApplyActorObservationBatch(Frames(8), Publications()));
  EXPECT_EQ(events.back().from, RelationPhase::kDormant);
  EXPECT_EQ(events.back().to, RelationPhase::kActive);
  ASSERT_TRUE(source.ApplyActorObservationBatch(Frames(9), Publications()));
  ASSERT_EQ(events.back().to, RelationPhase::kContact);
  ASSERT_TRUE(
      source.BeginActorObservationEpoch(actors[1], "navigation", &epochs[1]));
  ASSERT_TRUE(source.ApplyActorObservationBatch(Frames(10), Publications()));
  EXPECT_EQ(events.back().from, RelationPhase::kDormant);
  EXPECT_EQ(events.back().to, RelationPhase::kActive);
}

TEST_F(ObservedFacingPointTest, UnpublishedAvailableAnchorWithdrawsOldPoint) {
  ASSERT_TRUE(source.ApplyActorObservationBatch(Frames(0), Publications()));
  ASSERT_EQ(Packet(0).payloads.size(), 1u);
  auto publications = Publications();
  publications.pop_back();  // B's fresh head is present, but not published.
  ASSERT_TRUE(source.ApplyActorObservationBatch(Frames(1), publications));
  EXPECT_TRUE(Packet(0).payloads.empty());
  EXPECT_TRUE(PointStates(0).empty());
  // A point-only commit publishes B but cannot re-evaluate held A(seq1).
  const auto deliveries = packets.size();
  ASSERT_TRUE(source.CommitObservedFacingPoint(
      Point(1), leases[1], epochs[1], 1, Frames(1)[1].observations[1].position,
      now, now + 700));
  EXPECT_EQ(packets.size(), deliveries);
  EXPECT_TRUE(Packet(0).payloads.empty());
  ASSERT_TRUE(source.ApplyActorObservationBatch(Frames(2), Publications()));
  ASSERT_EQ(Packet(0).payloads.size(), 1u);
  EXPECT_EQ(Packet(0).source_observation_sequence, 2);
}

TEST_F(ObservedFacingPointTest,
       ContinuousCounterpartCannotExtendSourceDeadline) {
  Publish(1, 0, 1200);
  MapPoint(1, {10, 14, 1.8}, 0, 1700);
  AdmitReceiverBody();
  std::vector<RelationEvent> events;
  receiver.RegisterOnRelationEvent([&](const RelationEvent& e) {
    if (e.actor_shared_id == 1000) events.push_back(e);
  });
  Observe(0, 0);
  Deliver(Packet(0));
  ASSERT_EQ(events.back().to, RelationPhase::kActive);
  now = 1100;
  MapPoint(1, {14, 10, 1.8}, 1, 1800);
  Observe(0, 1);
  const auto fresh = Deliver(Packet(0));
  ASSERT_TRUE(Facing(fresh) && Facing(fresh)->has_goal);
  EXPECT_EQ(events.back().from, RelationPhase::kActive);
  EXPECT_EQ(events.back().to, RelationPhase::kContact);
  now = 1200;
  EXPECT_EQ(receiver.GetRealizationTargets(clones[0]), nullptr);
  ActorRealizationValidity validity;
  const uint32_t domains =
      kRealizationTreeRevision | kRealizationSupportRevision;
  ASSERT_TRUE(receiver.GetActorRealizationValidity(clones[0], fresh.revision,
                                                   domains, &validity));
  EXPECT_EQ(validity.changed_mask & domains, domains);
  EXPECT_EQ(packets.size(), 3u);  // B(seq0), A(seq0), A(seq1), no idle replay.
}

TEST_F(ObservedFacingPointTest, SharedOwnerCannotBeMutatedByTwoBatchSubjects) {
  ASSERT_TRUE(source.SetActorElement(actors[1], Owner(0)));
  // No point request: rejection is about ambiguous subject state, not merely
  // the former point producer's now-invalid owner binding.
  EXPECT_FALSE(source.ApplyActorObservationBatch(Frames(0), {}));
  EXPECT_TRUE(packets.empty());
  for (int i = 0; i < 2; ++i) {
    uint64_t epoch;
    int64_t sequence;
    double time;
    bool has_sample;
    ASSERT_TRUE(source.GetActorObservationClock(actors[i], &epoch, &sequence,
                                                &time, &has_sample));
    EXPECT_FALSE(has_sample);
    EXPECT_EQ(sequence, -1);
  }
}

TEST_F(ObservedFacingPointTest, BatchOwnsRowsAcrossDeliveryCallbacks) {
  auto frames = Frames(0);
  auto publications = Publications();
  hub.SetRelationUpdateCallback([&](const RemoteRelationUpdate& update) {
    packets.push_back(PeersOnly(update));
    // A delivery consumer may release its input queue. The remaining accepted
    // members must not disappear or change because the caller clears storage.
    frames.clear();
    publications.clear();
  });
  ASSERT_TRUE(source.ApplyActorObservationBatch(frames, publications));
  ASSERT_EQ(packets.size(), 2u);
  EXPECT_EQ(Packet(0).source_observation_sequence, 0);
  EXPECT_EQ(Packet(1).source_observation_sequence, 0);
  EXPECT_EQ(Packet(0).payloads.size(), 1u);
  EXPECT_EQ(Packet(1).payloads.size(), 1u);
}

}  // namespace
}  // namespace spatialblend
