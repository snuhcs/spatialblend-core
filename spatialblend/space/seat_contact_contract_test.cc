// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Source extraction/phase contract. No app, receiver, motion model or contact
// annotation is needed: contact is inferred from body-anchor/support proximity.
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace test {
namespace {

class SeatContactContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto* tree = space_.GetElementTree();
    ASSERT_TRUE(tree->AddElement(1).ok());
    tree->GetElement(1)->SetSharedId(1);
    tree->GetElement(1)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(
            std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
                {-10, -10}, {10, -10}, {10, 10}, {-10, 10}})));
    ASSERT_TRUE(tree->AddElement(2, 1).ok());
    tree->GetElement(2)->SetSharedId(2);
    tree->GetElement(2)->SetTag("Sittable");
    // Elevated bounded surface, not the ground-plane footprint or chair origin.
    tree->GetElement(2)->SetSpatialDefinition(
        std::make_shared<SurfaceSpatialDefinition>(
            std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
                {-1, -1, .5}, {1, -1, .5}, {1, 1, .5}, {-1, 1, .5}})));
    actor_ =
        space_.CreateActor({SubjectAnchor("locomotion"),
                            SubjectAnchor("pelvis"), SubjectAnchor("hand")});
    space_.GetActor(actor_)->SetElementId(1);
    space_.Bind(actor_, "pelvis", RelationKernelType::kAvatarAffordancePose,
                "Sittable");
    space_.RegisterOnRelationEvent([this](const RelationEvent& event) {
      if (event.anchor_name == "pelvis" && event.referent_shared_id == 2 &&
          event.kernel_type == RelationKernelType::kAvatarAffordancePose) {
        seat_phase_ = event.to;
        seat_events_.push_back(event);
      }
      if (event.anchor_name == "hand" && event.referent_shared_id == 2 &&
          event.kernel_type == RelationKernelType::kAvatarAffordancePose) {
        hand_phase_ = event.to;
      }
      if (event.anchor_name == "locomotion" && event.referent_shared_id == 2 &&
          event.kernel_type == RelationKernelType::kAvatarAffordancePose) {
        root_phase_ = event.to;
      }
    });
  }

  void ObservePelvis(const Eigen::Vector3d& pelvis) {
    space_.SetAnchorWorldPose(actor_, "pelvis", pelvis, 0.0);
    space_.UpdateActorState(actor_, root_, 0.0);
  }

  void ExpectSeatProximity(double expected) {
    const auto states = space_.GetRelationStates(actor_);
    int seat_matches = 0;
    for (const auto& state : states) {
      if (state.anchor_name != "pelvis") continue;
      ++seat_matches;
      EXPECT_EQ(state.kernel_type, RelationKernelType::kAvatarAffordancePose);
      EXPECT_EQ(state.referent_element_id, 2);
      // This flag means a projected support exists, even before contact phase.
      EXPECT_TRUE(state.has_contact);
      EXPECT_NEAR(state.contact_point_world.z(), .5, 1e-12);
      EXPECT_NEAR(state.proximity, expected, 1e-12);
    }
    EXPECT_EQ(seat_matches, 1);
  }

  Space space_;
  spatialblend::entity_id_t actor_;
  Eigen::Vector2d root_{4, 4};
  RelationPhase seat_phase_ = RelationPhase::kDormant;
  RelationPhase hand_phase_ = RelationPhase::kDormant;
  RelationPhase root_phase_ = RelationPhase::kDormant;
  std::vector<RelationEvent> seat_events_;
};

TEST_F(SeatContactContractTest,
       PelvisProximityControlsContactWithStationaryRoot) {
  ObservePelvis({0, 0, 1.0});
  ExpectSeatProximity(1.0 / 1.5);
  EXPECT_EQ(seat_phase_, RelationPhase::kActive);

  ObservePelvis({0, 0, .55});
  ExpectSeatProximity(1.0 / 1.05);
  EXPECT_EQ(seat_phase_, RelationPhase::kContact);

  // Default hysteresis: below the enter threshold but above release.
  const auto event_count = seat_events_.size();
  ObservePelvis({0, 0, .60});
  ExpectSeatProximity(1.0 / 1.10);
  EXPECT_EQ(seat_phase_, RelationPhase::kContact);
  EXPECT_EQ(seat_events_.size(), event_count);

  ObservePelvis({0, 0, .80});
  ExpectSeatProximity(1.0 / 1.30);
  EXPECT_EQ(seat_phase_, RelationPhase::kDriving);
  ASSERT_EQ(seat_events_.size(), event_count + 1);
  EXPECT_EQ(seat_events_.back().from, RelationPhase::kContact);

  ObservePelvis({0, 0, .5});
  ExpectSeatProximity(1.0);
  EXPECT_EQ(seat_phase_, RelationPhase::kContact);
  EXPECT_TRUE(
      (space_.GetActor(actor_)->GetTransform().block<2, 1>(0, 3).isApprox(
          root_)));
}

TEST_F(SeatContactContractTest, RootOrHandProximityCannotSubstituteForPelvis) {
  // Deliberately add distracting generic contacts. Neither binding is the
  // designated body anchor for the seat semantic predicate in this fixture.
  space_.Bind(actor_, "locomotion", RelationKernelType::kAvatarAffordancePose,
              "Sittable");
  space_.Bind(actor_, "hand", RelationKernelType::kAvatarAffordancePose,
              "Sittable");
  space_.SetAnchorWorldPose(actor_, "hand", Eigen::Vector3d(0, 0, .5), 0.0);
  space_.SetAnchorWorldPose(actor_, "locomotion", Eigen::Vector3d(0, 0, .5),
                            0.0);
  root_ = Eigen::Vector2d::Zero();
  ObservePelvis({0, 0, 1.0});
  ObservePelvis({0, 0, 1.0});
  ExpectSeatProximity(1.0 / 1.5);
  EXPECT_EQ(hand_phase_, RelationPhase::kContact);
  EXPECT_EQ(root_phase_, RelationPhase::kContact);
  EXPECT_NE(seat_phase_, RelationPhase::kContact);

  root_ = Eigen::Vector2d(5, 5);
  ObservePelvis({0, 0, .5});
  EXPECT_EQ(seat_phase_, RelationPhase::kContact);

  // Same height but outside the polygon: the infinite plane is not a seat.
  ObservePelvis({2, 0, .5});
  ExpectSeatProximity(.5);
  EXPECT_EQ(seat_phase_, RelationPhase::kDriving);
}

TEST(SourceSeatObservationContract,
     ConfiguredBodyAnchorDoesNotRequireNavigationInput) {
  // Explicit body selections, not a string-name heuristic. Both use actual
  // timestamped source admission; no receiver goal or dataset contact label.
  for (const std::string body : {"authored_body_origin", "authored_pelvis"}) {
    SCOPED_TRACE(body);
    Space source;
    ASSERT_TRUE(source.AddElement(1));
    ASSERT_TRUE(source.SetElementSharedId(1, 1));
    // AffordancePose is the proximal subtree of the containing node. The
    // explicit bind does not select the seat.
    ASSERT_TRUE(source.SetElementSpatialDefinition(
        1, std::make_shared<NavigableSpatialDefinition>(
               std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
                   {-10, -10}, {10, -10}, {10, 10}, {-10, 10}}))));
    ASSERT_TRUE(source.AddElement(2, 1));
    ASSERT_TRUE(source.SetElementSharedId(2, 2));
    ASSERT_TRUE(source.SetElementTag(2, "Sittable"));
    ASSERT_TRUE(source.SetElementSpatialDefinition(
        2, std::make_shared<SurfaceSpatialDefinition>(
               std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
                   {-1, -1, .5}, {1, -1, .5}, {1, 1, .5}, {-1, 1, .5}}))));
    ASSERT_TRUE(source.AddElement(3, 1));
    ASSERT_TRUE(source.SetElementSharedId(3, 3));
    const auto actor = source.CreateActor(
        {SubjectAnchor("navigation"),
         SubjectAnchor(body, Eigen::Vector3d::Zero(),
                       {StateType::kPosition3D, StateType::kRotationYaw})});
    ASSERT_TRUE(source.SetActorElement(actor, 3));
    ASSERT_NE(source.BindScoped(actor, body,
                                RelationKernelType::kAvatarAffordancePose, 2,
                                "Sittable"),
              nullptr);
    RelationPhase phase = RelationPhase::kDormant;
    source.RegisterOnRelationEvent([&](const RelationEvent& event) {
      if (event.anchor_name == body && event.referent_shared_id == 2 &&
          event.kernel_type == RelationKernelType::kAvatarAffordancePose)
        phase = event.to;
    });
    uint64_t epoch;
    ASSERT_TRUE(source.BeginActorObservationEpoch(actor, "navigation", &epoch));
    // The pelvis is far from the seat but a bounded projection already exists.
    ASSERT_TRUE(source.ApplyActorObservationFrame(actor, epoch, 0, 0,
                                                  {{body, {0, 0, 1.0}, 0}}));
    auto states = source.GetRelationStates(actor);
    ASSERT_EQ(states.size(), 1u);
    EXPECT_TRUE(states[0].has_contact);
    EXPECT_NEAR(states[0].proximity, 1 / 1.5, 1e-12);
    EXPECT_NE(phase, RelationPhase::kContact);
    ASSERT_TRUE(source.ApplyActorObservationFrame(actor, epoch, 1, .1,
                                                  {{body, {0, 0, .55}, 0}}));
    EXPECT_EQ(phase, RelationPhase::kContact);
    ASSERT_TRUE(source.ApplyActorObservationFrame(actor, epoch, 2, .2,
                                                  {{body, {0, 0, .60}, 0}}));
    EXPECT_EQ(phase, RelationPhase::kContact);
    // A body sample missing from the current frame cannot borrow the previous
    // contact, navigation point, generated display state, or anchor offset.
    ASSERT_TRUE(source.ApplyActorObservationFrame(
        actor, epoch, 3, .3, {{"navigation", {0, 0, .5}, 0}}));
    EXPECT_TRUE(source.GetRelationStates(actor).empty());
    // A pose the driver publishes is not a genuine body sample: it neither
    // supplies nor continues the missing contact.
    EmbodiedPoseInput published;
    published.epoch = 1;
    ASSERT_EQ(source.PublishEmbodiedPose(actor, published, 0.0, 1),
              EmbodiedPublishResult::kOk);
    EXPECT_TRUE(source.GetRelationStates(actor).empty());
    ASSERT_TRUE(source.ApplyActorObservationFrame(actor, epoch, 4, .4,
                                                  {{body, {2, 0, .5}, 0}}));
    states = source.GetRelationStates(actor);
    ASSERT_EQ(states.size(), 1u);
    EXPECT_NEAR(states[0].proximity, .5, 1e-12);
    EXPECT_NE(phase, RelationPhase::kContact);
    uint64_t observed_epoch;
    int64_t sequence;
    double time;
    bool has_sample;
    ASSERT_TRUE(source.GetActorObservationClock(actor, &observed_epoch,
                                                &sequence, &time, &has_sample));
    EXPECT_EQ(observed_epoch, epoch);
    EXPECT_EQ(sequence, 4);
    EXPECT_DOUBLE_EQ(time, .4);
    EXPECT_TRUE(has_sample);
  }
}

class SourceSeatLifetimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(source_.AddElement(1));
    ASSERT_TRUE(source_.SetElementSharedId(1, 1));
    // AffordancePose is the proximal subtree of the containing node. The
    // explicit bind does not select the seat.
    ASSERT_TRUE(source_.SetElementSpatialDefinition(
        1, std::make_shared<NavigableSpatialDefinition>(
               std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
                   {-10, -10}, {10, -10}, {10, 10}, {-10, 10}}))));
    AddSeat();
    ASSERT_TRUE(source_.AddElement(3, 1));
    ASSERT_TRUE(source_.SetElementSharedId(3, 3));
    ASSERT_TRUE(source_.AddElement(4, 1));
    ASSERT_TRUE(source_.SetElementSharedId(4, 4));
    ASSERT_TRUE(source_.SetElementTag(4, "Touchable"));
    // The touchable surface touches the seat's along x = -1 instead of coinciding with it: sibling
    // footprints may only touch. The hand rests on it at (-2, 0).
    ASSERT_TRUE(source_.SetElementSpatialDefinition(4, Surface(-3, -1))) << source_.LastSceneError().message;
    actor_ = source_.CreateActor(
        {SubjectAnchor("navigation"),
         SubjectAnchor("body", Eigen::Vector3d::Zero(),
                       {StateType::kPosition3D, StateType::kRotationYaw}),
         SubjectAnchor("hand", Eigen::Vector3d::Zero(),
                       {StateType::kPosition3D, StateType::kRotationYaw})});
    ASSERT_TRUE(source_.SetActorElement(actor_, 3));
    ASSERT_NE(source_.BindScoped(actor_, "body",
                                 RelationKernelType::kAvatarAffordancePose, 2,
                                 "Sittable"),
              nullptr);
    ASSERT_NE(source_.BindScoped(actor_, "hand",
                                 RelationKernelType::kAvatarAffordancePose, 4,
                                 "Touchable"),
              nullptr);
    source_.RegisterOnRelationEvent(
        [this](const RelationEvent& event) { events_.push_back(event); });
    ASSERT_TRUE(
        source_.BeginActorObservationEpoch(actor_, "navigation", &epoch_));
  }

  std::shared_ptr<SurfaceSpatialDefinition> Surface(double x0 = -1, double x1 = 1) {
    return std::make_shared<SurfaceSpatialDefinition>(
        std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
            {x0, -1, .5}, {x1, -1, .5}, {x1, 1, .5}, {x0, 1, .5}}));
  }
  void AddSeat() {
    ASSERT_TRUE(source_.AddElement(2, 1));
    ASSERT_TRUE(source_.SetElementSharedId(2, 2));
    ASSERT_TRUE(source_.SetElementTag(2, "Sittable"));
    ASSERT_TRUE(source_.SetElementSpatialDefinition(2, Surface()));
  }
  void Observe(double height, bool body_present = true) {
    std::vector<Space::ActorObservation> anchors = {
        {"navigation", {4, 4, 0}, 0}, {"hand", {-2, 0, .5}, 0}};
    if (body_present) anchors.push_back({"body", {0, 0, height}, 0});
    ASSERT_TRUE(source_.ApplyActorObservationFrame(actor_, epoch_, sequence_,
                                                   sequence_ * .1, anchors));
    ++sequence_;
  }
  RelationPhase Phase(const std::string& anchor) const {
    for (auto it = events_.rbegin(); it != events_.rend(); ++it)
      if (it->anchor_name == anchor) return it->to;
    return RelationPhase::kDormant;
  }
  size_t Events(const std::string& anchor) const {
    size_t count = 0;
    for (const auto& event : events_)
      if (event.anchor_name == anchor) ++count;
    return count;
  }
  void EstablishContact() {
    Observe(.5);
    Observe(.5);
    ASSERT_EQ(Phase("body"), RelationPhase::kContact);
    ASSERT_EQ(Phase("hand"), RelationPhase::kContact);
  }

  Space source_;
  spatialblend::entity_id_t actor_;
  uint64_t epoch_ = 0;
  int64_t sequence_ = 0;
  std::vector<RelationEvent> events_;
};

TEST_F(SourceSeatLifetimeTest,
       MissingBodyWithdrawsContactWithoutResettingHand) {
  EstablishContact();
  const auto hand_events = Events("hand");
  Observe(0, false);
  EXPECT_EQ(Phase("body"), RelationPhase::kDormant);
  EXPECT_EQ(Phase("hand"), RelationPhase::kContact);
  EXPECT_EQ(Events("hand"), hand_events);
  const auto withdrawn_events = Events("body");
  Observe(0, false);
  EXPECT_EQ(Events("body"), withdrawn_events);

  // In the contact hysteresis band, a new lifetime must not inherit Contact.
  Observe(.60);
  EXPECT_EQ(Phase("body"), RelationPhase::kActive);
  Observe(.60);
  EXPECT_NE(Phase("body"), RelationPhase::kContact);
  Observe(.55);
  EXPECT_EQ(Phase("body"), RelationPhase::kContact);
}

TEST_F(SourceSeatLifetimeTest, FilteredOutSupportCannotLendOldContactOnReturn) {
  EstablishContact();
  const auto hand_events = Events("hand");
  ASSERT_TRUE(source_.SetElementTag(2, "NotSittable"));
  Observe(.60);
  EXPECT_EQ(Phase("body"), RelationPhase::kDormant);
  ASSERT_TRUE(source_.SetElementTag(2, "Sittable"));
  Observe(.60);
  EXPECT_EQ(Phase("body"), RelationPhase::kActive);
  EXPECT_EQ(Events("hand"), hand_events);
}

TEST_F(SourceSeatLifetimeTest,
       RemovedRecreatedSupportStartsNewContactLifetime) {
  EstablishContact();
  const auto hand_events = Events("hand");
  ASSERT_TRUE(source_.RemoveLocalElementSubtree(2));
  EXPECT_EQ(Phase("body"), RelationPhase::kDormant);
  AddSeat();  // Same local id is a different support lifetime, even between
              // frames.
  Observe(.60);
  EXPECT_EQ(Phase("body"), RelationPhase::kActive);
  EXPECT_EQ(Events("hand"), hand_events);
}

TEST_F(SourceSeatLifetimeTest, NewEpochCannotResumeContactFromHysteresisBand) {
  EstablishContact();
  const auto previous_epoch = epoch_;
  ASSERT_TRUE(
      source_.BeginActorObservationEpoch(actor_, "navigation", &epoch_));
  ASSERT_GT(epoch_, previous_epoch);
  sequence_ = 0;
  Observe(.60);
  EXPECT_EQ(Phase("body"), RelationPhase::kActive);
  EXPECT_EQ(Phase("hand"), RelationPhase::kActive);
}

TEST_F(SourceSeatLifetimeTest,
       ClearControlCannotAdvanceAnotherHeldBodyRelation) {
  Observe(.5);
  ASSERT_EQ(Phase("body"), RelationPhase::kActive);
  ASSERT_EQ(Phase("hand"), RelationPhase::kActive);
  const auto hand_events = Events("hand");
  ASSERT_TRUE(source_.ClearActorObservationAnchor(actor_, epoch_, "body"));
  EXPECT_EQ(Phase("body"), RelationPhase::kDormant);
  EXPECT_EQ(Phase("hand"), RelationPhase::kActive);
  EXPECT_EQ(Events("hand"), hand_events);
  uint64_t epoch;
  int64_t sequence;
  double time;
  bool has_sample;
  ASSERT_TRUE(source_.GetActorObservationClock(actor_, &epoch, &sequence, &time,
                                               &has_sample));
  EXPECT_EQ(epoch, epoch_);
  EXPECT_EQ(sequence, 0);
  EXPECT_DOUBLE_EQ(time, 0);
  EXPECT_TRUE(has_sample);
  Observe(.60);
  EXPECT_EQ(Phase("body"), RelationPhase::kActive);
  EXPECT_EQ(Phase("hand"), RelationPhase::kContact);
}

// There is no near field and no release dwell (D3). The relation stays in scope for as
// long as the anchor is inside its parent object's region (the 20 m room), so a far
// sample only lowers R: the phase falls from CONTACT to DRIVING by the contact
// hysteresis, never to DORMANT. It is withdrawn at once on leaving the region, and restarts at ACTIVE.
TEST_F(SourceSeatLifetimeTest, LeavingTheParentRegionWithdrawsThenRestartsAtActive) {
  EstablishContact();
  const double t0 = sequence_ * .1;
  ASSERT_TRUE(source_.ApplyActorObservationFrame(
      actor_, epoch_, sequence_, t0,
      {{"body", {8, 0, .5}, 0}, {"hand", {-2, 0, .5}, 0}}));
  ++sequence_;
  EXPECT_EQ(Phase("body"), RelationPhase::kDriving) << "far, still in scope";
  EXPECT_EQ(Phase("hand"), RelationPhase::kContact);
  // Out of the room [-10,10]^2: out of scope, withdrawn on this very frame.
  ASSERT_TRUE(source_.ApplyActorObservationFrame(
      actor_, epoch_, sequence_, t0 + 0.1,
      {{"body", {30, 0, .5}, 0}, {"hand", {-2, 0, .5}, 0}}));
  ++sequence_;
  EXPECT_EQ(Phase("body"), RelationPhase::kDormant);
  EXPECT_EQ(Phase("hand"), RelationPhase::kContact);
  Observe(.60);
  EXPECT_EQ(Phase("body"), RelationPhase::kActive);
}

TEST_F(SourceSeatLifetimeTest, SharedIdentityPreservesStillPresentSupport) {
  // Duplicate shared ids are legal in the legacy tree. Their public phase is
  // shared, but losing one local row cannot withdraw a surviving exact row.
  ASSERT_TRUE(source_.SetElementSharedId(2, 42));
  ASSERT_TRUE(source_.AddElement(5, 1));
  ASSERT_TRUE(source_.SetElementSharedId(5, 5));
  ASSERT_TRUE(source_.SetElementSharedId(5, 42));
  ASSERT_TRUE(source_.SetElementTag(5, "Sittable"));
  // A second row of the shared id, beside the first (touching along x = 1): sibling footprints may
  // only touch.
  ASSERT_TRUE(source_.SetElementSpatialDefinition(5, Surface(1, 3)));
  // One body binding. A second explicit bind is the same proximal set
  // and would step the shared phase twice.
  EstablishContact();
  const auto body_events = Events("body");
  ASSERT_TRUE(source_.RemoveLocalElementSubtree(5));
  EXPECT_EQ(Phase("body"), RelationPhase::kContact);
  EXPECT_EQ(Events("body"), body_events);
  Observe(.60);
  EXPECT_EQ(Phase("body"), RelationPhase::kContact);
  EXPECT_EQ(Events("body"), body_events);
  ASSERT_TRUE(source_.RemoveLocalElementSubtree(2));
  EXPECT_EQ(Phase("body"), RelationPhase::kDormant);
  AddSeat();
  ASSERT_TRUE(source_.SetElementSharedId(2, 42));
  Observe(.60);
  EXPECT_EQ(Phase("body"), RelationPhase::kActive);
}

TEST_F(SourceSeatLifetimeTest, SharedIdChangeWithdrawsFrozenOldIdentity) {
  EstablishContact();
  const auto count = events_.size();
  ASSERT_TRUE(source_.SetElementSharedId(2, 1002));
  Observe(.60);
  ASSERT_EQ(events_.size(), count + 2);
  EXPECT_EQ(events_[count].referent_shared_id, 2);
  EXPECT_EQ(events_[count].from, RelationPhase::kContact);
  EXPECT_EQ(events_[count].to, RelationPhase::kDormant);
  EXPECT_EQ(events_[count + 1].referent_shared_id, 1002);
  EXPECT_EQ(events_[count + 1].from, RelationPhase::kDormant);
  EXPECT_EQ(events_[count + 1].to, RelationPhase::kActive);
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
