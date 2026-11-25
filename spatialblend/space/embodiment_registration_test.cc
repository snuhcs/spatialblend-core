// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/embodiment_registration_test.cc
//
// ABI 4.7 at the Space layer: a registered anchor is sampled from its bone, its own
// port is refused, the declared families are enforced at bind, the source-side composition equals an independent
// one, and an actor without a registration is untouched. Expected values are independent oracles in this file.
// The spec is a TEST FIXTURE (values of the evaluation harness's (not public) SMPL-X measurement, used only to exercise the schema).
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <memory>
#include <string>
#include <thread>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/embodiment/embodiment.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace {

const char* kSpec = R"({
 "format":"spatialblend.embodiment","schema_version":"1.0.0","id":"fixture-smplx",
 "skeleton":{"joints":[{"name":"pelvis","parent":-1},{"name":"head","parent":0},{"name":"right_wrist","parent":0}]},
 "anchors":[
  {"name":"root","bone":null,"families":["Spacing"]},
  {"name":"head","bone":"head","families":["Facing"]},
  {"name":"pelvis","bone":"pelvis","offset":{"position":[0.0634,0.0,-0.1860]},"families":["AffordancePose"],
   "provenance":{"procedure":"fixture","source":"test fixture"}},
  {"name":"hand_r","bone":"right_wrist","offset":{"position":[0.0611,0.0403,-0.0310]},"families":["AffordancePose"],
   "provenance":{"procedure":"fixture","source":"test fixture"}}
 ]})";

struct Rig {
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space* space = ctx->AddSpace();
  spatialblend::entity_id_t actor = 0;
  Embodiment embodiment;
  Rig(bool register_it = true) {
    actor = space->CreateActor({SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))});
    std::string err;
    EXPECT_TRUE(Embodiment::FromJson(kSpec, &embodiment, &err)) << err;
    if (register_it)
      EXPECT_EQ(space->RegisterEmbodiment(actor, embodiment), Space::RegisterEmbodimentResult::kOk);
  }
  uint32_t Id(const char* name) {
    uint32_t id = 0;
    EXPECT_TRUE(space->AnchorIdFor(actor, name, &id));
    return id;
  }
};

BonePoseInput Bone(double x, double y, double z, double yaw, uint64_t epoch = 1) {
  BonePoseInput b;
  b.p[0] = x;
  b.p[1] = y;
  b.p[2] = z;
  const Eigen::Quaterniond q(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
  b.q[0] = q.x();
  b.q[1] = q.y();
  b.q[2] = q.z();
  b.q[3] = q.w();
  b.epoch = epoch;
  return b;
}

TEST(EmbodimentRegistration, ARegisteredAnchorIsItsBoneComposedWithTheRegisteredOffset) {
  Rig r;
  const int pelvis = r.embodiment.BoneId("pelvis");
  EmbodiedSample s;
  ASSERT_TRUE(r.space->SampleAnchorPose(r.actor, r.Id("pelvis"), &s));
  EXPECT_FALSE(s.valid) << "no bone sample is no anchor sample: nothing is substituted";

  const double yaw = 0.7;
  ASSERT_EQ(r.space->PublishBonePose(r.actor, pelvis, Bone(1.5, -2.0, 0.636, yaw), 12.5, 1),
            EmbodiedPublishResult::kOk);
  ASSERT_TRUE(r.space->SampleAnchorPose(r.actor, r.Id("pelvis"), &s));
  ASSERT_TRUE(s.valid);
  const double c = std::cos(yaw), sn = std::sin(yaw);
  EXPECT_NEAR(s.x, 1.5 + c * 0.0634, 1e-14);
  EXPECT_NEAR(s.y, -2.0 + sn * 0.0634, 1e-14);
  EXPECT_NEAR(s.z, 0.636 - 0.1860, 1e-14);
  EXPECT_NEAR(s.yaw, yaw, 1e-14);
  EXPECT_EQ(s.orientation_type, 2);
  EXPECT_EQ(s.sequence, 1u);
  EXPECT_EQ(s.epoch, 1u);
  EXPECT_DOUBLE_EQ(s.timestamp_s, 12.5);

  // Stale sequence, then a new epoch (a teleport) restarts the sequence.
  EXPECT_EQ(r.space->PublishBonePose(r.actor, pelvis, Bone(9, 9, 9, 0), 12.6, 1), EmbodiedPublishResult::kStale);
  ASSERT_TRUE(r.space->SampleAnchorPose(r.actor, r.Id("pelvis"), &s));
  EXPECT_NEAR(s.x, 1.5 + c * 0.0634, 1e-14);
  EXPECT_EQ(r.space->PublishBonePose(r.actor, pelvis, Bone(2, 0, 0.636, 0, 2), 13.0, 1),
            EmbodiedPublishResult::kOk);
  ASSERT_TRUE(r.space->SampleAnchorPose(r.actor, r.Id("pelvis"), &s));
  EXPECT_NEAR(s.x, 2.0 + 0.0634, 1e-14);
  EXPECT_EQ(s.epoch, 2u);
}

TEST(EmbodimentRegistration, ABoneThatSeveralAnchorsShareSamplesOncePerAnchor) {
  Rig r;
  const int wrist = r.embodiment.BoneId("right_wrist");
  ASSERT_EQ(r.space->PublishBonePose(r.actor, wrist, Bone(0, 0, 1, 0), 1.0, 1), EmbodiedPublishResult::kOk);
  EmbodiedSample s;
  ASSERT_TRUE(r.space->SampleAnchorPose(r.actor, r.Id("hand_r"), &s));
  EXPECT_NEAR(s.x, 0.0611, 1e-15);
  EXPECT_NEAR(s.y, 0.0403, 1e-15);
  EXPECT_NEAR(s.z, 1.0 - 0.0310, 1e-15);
  // The pelvis bone was never published: its anchor still has no sample.
  ASSERT_TRUE(r.space->SampleAnchorPose(r.actor, r.Id("pelvis"), &s));
  EXPECT_FALSE(s.valid);
}

TEST(EmbodimentRegistration, ABoneBoundAnchorHasNoPortOfItsOwn) {
  Rig r;
  EmbodiedPoseInput in;
  in.x = 1;
  EXPECT_EQ(r.space->PublishAnchorPose(r.actor, r.Id("pelvis"), in, 1.0, 1), EmbodiedPublishResult::kRefused);
  EXPECT_EQ(r.space->PublishAnchorPose(r.actor, r.Id("head"), in, 1.0, 1), EmbodiedPublishResult::kRefused);
  // The root, and an anchor the registration does not mention, are untouched.
  EXPECT_EQ(r.space->PublishAnchorPose(r.actor, Space::kRootAnchorId, in, 1.0, 1), EmbodiedPublishResult::kOk);
  EXPECT_EQ(r.space->PublishAnchorPose(r.actor, r.Id("chest"), in, 1.0, 1), EmbodiedPublishResult::kOk);
  EmbodiedSample s;
  ASSERT_TRUE(r.space->SampleAnchorPose(r.actor, r.Id("chest"), &s));
  EXPECT_DOUBLE_EQ(s.x, 1.0);
  EXPECT_EQ(s.orientation_type, 0);
}

TEST(EmbodimentRegistration, RegistrationIsOnceAndRefusesASecondWriter) {
  Rig r;
  EXPECT_EQ(r.space->RegisterEmbodiment(r.actor, r.embodiment), Space::RegisterEmbodimentResult::kRefused);
  EXPECT_EQ(r.space->RegisterEmbodiment(9999, r.embodiment), Space::RegisterEmbodimentResult::kInvalid);

  Rig late(/*register_it=*/false);
  EmbodiedPoseInput in;
  ASSERT_EQ(late.space->PublishAnchorPose(late.actor, late.Id("pelvis"), in, 1.0, 1), EmbodiedPublishResult::kOk);
  EXPECT_EQ(late.space->RegisterEmbodiment(late.actor, late.embodiment), Space::RegisterEmbodimentResult::kRefused)
      << "the pelvis already has a driver-published port: two writers of one fact";

  Embodiment bad = r.embodiment;
  bad.anchors[2].provenance_json.clear();
  Rig fresh(false);
  EXPECT_EQ(fresh.space->RegisterEmbodiment(fresh.actor, bad), Space::RegisterEmbodimentResult::kInvalid);
}

TEST(EmbodimentRegistration, AnchorsAreSubjectAnchorsAndTheDeclaredFamiliesAreEnforcedAtBind) {
  Rig r;
  const Actor* actor = r.space->GetActor(r.actor);
  ASSERT_NE(actor->GetAnchor("pelvis"), nullptr);
  ASSERT_NE(actor->GetAnchor("hand_r"), nullptr);
  EXPECT_NE(r.space->Bind(r.actor, "pelvis", RelationKernelType::kAvatarAffordancePose), nullptr);
  EXPECT_EQ(r.space->Bind(r.actor, "pelvis", RelationKernelType::kAvatarSpacing), nullptr);
  EXPECT_EQ(r.space->Bind(r.actor, "pelvis", RelationKernelType::kAvatarFacing), nullptr);
  EXPECT_NE(r.space->Bind(r.actor, "head", RelationKernelType::kAvatarFacing), nullptr);
  EXPECT_EQ(r.space->Bind(r.actor, "head", RelationKernelType::kAvatarAffordancePose), nullptr);
  EXPECT_NE(r.space->Bind(r.actor, "root", RelationKernelType::kAvatarSpacing), nullptr);
  // An anchor the registration does not declare, and an actor without a registration: as before.
  r.space->AddActorAnchors(r.actor, {SubjectAnchor("chest", Eigen::Vector3d(0, 0, 1.3))});
  EXPECT_NE(r.space->Bind(r.actor, "chest", RelationKernelType::kAvatarFacing), nullptr);
  Rig plain(false);
  plain.space->AddActorAnchors(plain.actor, {SubjectAnchor("pelvis", Eigen::Vector3d(0, 0, 0.9))});
  EXPECT_NE(plain.space->Bind(plain.actor, "pelvis", RelationKernelType::kAvatarSpacing), nullptr);
}

TEST(EmbodimentRegistration, ABindingThatViolatesTheDeclaredFamiliesRefusesTheRegistration) {
  Rig r(/*register_it=*/false);
  r.space->AddActorAnchors(r.actor, {SubjectAnchor("pelvis", Eigen::Vector3d(0, 0, 0.9))});
  ASSERT_NE(r.space->Bind(r.actor, "pelvis", RelationKernelType::kAvatarSpacing), nullptr);  // allowed: nothing registered yet
  std::string conflict;
  EXPECT_EQ(r.space->RegisterEmbodiment(r.actor, r.embodiment, &conflict), Space::RegisterEmbodimentResult::kBindingConflict);
  EXPECT_EQ(conflict, "pelvis@Spacing");
  EXPECT_EQ(r.space->EmbodimentOf(r.actor), nullptr) << "a refused registration leaves nothing behind";
  // The order does not matter: the same pair is refused when the registration comes first.
  Rig first;
  EXPECT_EQ(first.space->Bind(first.actor, "pelvis", RelationKernelType::kAvatarSpacing), nullptr);
  // A conforming binding made first is fine.
  Rig ok(false);
  ok.space->AddActorAnchors(ok.actor, {SubjectAnchor("pelvis", Eigen::Vector3d(0, 0, 0.9))});
  ASSERT_NE(ok.space->Bind(ok.actor, "pelvis", RelationKernelType::kAvatarAffordancePose), nullptr);
  EXPECT_EQ(ok.space->RegisterEmbodiment(ok.actor, ok.embodiment, &conflict), Space::RegisterEmbodimentResult::kOk);
}

TEST(EmbodimentRegistration, TheBodyRadiusIsPerActorAndComesFromTheActorsEmbodiment) {
  Rig r(/*register_it=*/false);
  EXPECT_TRUE(std::isnan(r.space->GetBodyRadius(r.actor)) && std::isnan(r.space->GetBodyRadius()))
      << "no default: an actor with no registered radius uses the space-wide one, which is unset";
  r.embodiment.body.present = true;
  r.embodiment.body.floor_footprint_radius_m = 0.27;
  r.embodiment.body.provenance_json = R"({"procedure":"fixture","source":"test fixture"})";
  ASSERT_EQ(r.space->RegisterEmbodiment(r.actor, r.embodiment), Space::RegisterEmbodimentResult::kOk);
  EXPECT_DOUBLE_EQ(r.space->GetBodyRadius(r.actor), 0.27);

  // A second actor of the same space with a different radius coexists: no refusal, no averaging.
  const auto second = r.space->CreateActor({SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))});
  Embodiment other = r.embodiment;
  other.body.floor_footprint_radius_m = 0.4;
  ASSERT_EQ(r.space->RegisterEmbodiment(second, other), Space::RegisterEmbodimentResult::kOk);
  EXPECT_DOUBLE_EQ(r.space->GetBodyRadius(r.actor), 0.27);
  EXPECT_DOUBLE_EQ(r.space->GetBodyRadius(second), 0.4);

  // An actor that registers no radius keeps the space-wide one, which stays settable.
  const auto third = r.space->CreateActor({SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))});
  r.space->SetBodyRadius(0.5);
  EXPECT_DOUBLE_EQ(r.space->GetBodyRadius(third), 0.5);
  EXPECT_DOUBLE_EQ(r.space->GetBodyRadius(r.actor), 0.27) << "the space-wide setter does not touch a registered radius";

  // The radius leaves with the actor.
  ASSERT_TRUE(r.space->DestroyActor(second));
  EXPECT_DOUBLE_EQ(r.space->GetBodyRadius(second), 0.5);
}

// The radius belongs to the driver: the core's free space has no body-radius erosion whatever an
// actor registers, and the driver's course service keeps a body of THAT ACTOR's radius clear of the footprints.
TEST(EmbodimentRegistration, TheCoresFreeSpaceIgnoresTheRadiusAndEachActorsCourseKeepsItsOwnBodyClear) {
  auto ctx = LocalContext::Create();
  Space* space = ctx->AddSpace();
  auto* tree = space->GetElementTree();
  ASSERT_TRUE(tree->AddElement(1).ok());
  tree->GetElement(1)->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {10, 0}, {10, 10}, {0, 10}})));
  ASSERT_TRUE(tree->AddElement(2, 1).ok());
  Eigen::Matrix4d at = Eigen::Matrix4d::Identity();
  at(0, 3) = 5.0;
  at(1, 3) = 5.0;
  tree->GetElement(2)->SetLocalTransform(at);
  tree->GetElement(2)->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-1, -1, 0), Eigen::Vector3d(1, 1, 0.8), GeometryRole::kObstacle));
  const auto declare = [&](double radius) {
    const auto a = space->CreateActor({SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))});
    space->GetActor(a)->SetElementId(1);
    Embodiment e;
    std::string err;
    EXPECT_TRUE(Embodiment::FromJson(kSpec, &e, &err)) << err;
    e.body.present = true;
    e.body.floor_footprint_radius_m = radius;
    e.body.provenance_json = R"({"procedure":"fixture","source":"test fixture"})";
    EXPECT_EQ(space->RegisterEmbodiment(a, e), Space::RegisterEmbodimentResult::kOk);
    return a;
  };
  const auto small = declare(0.1), large = declare(0.5);
  const auto none = space->CreateActor({SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))});
  space->GetActor(none)->SetElementId(1);
  const Eigen::Vector2d p(6.3, 5.0);  // 0.3 m from the obstacle's edge
  const auto clearance = [&](const std::vector<Eigen::Vector2d>& course) {
    double least = 1e9;
    for (const auto& q : course)
      least = std::min(least, std::hypot(std::max({4.0 - q.x(), 0.0, q.x() - 6.0}),
                                         std::max({4.0 - q.y(), 0.0, q.y() - 6.0})));
    return least;
  };
  const auto length = [](const std::vector<Eigen::Vector2d>& c) {
    double l = 0.0;
    for (size_t i = 1; i < c.size(); ++i) l += (c[i] - c[i - 1]).norm();
    return l;
  };
  for (int round = 0; round < 3; ++round) {
    EXPECT_TRUE(space->IsFreeUnder(small, p, {}, {})) << "the core's F has no clearance, round " << round;
    EXPECT_TRUE(space->IsFreeUnder(large, p, {}, {})) << "whatever the actor registered, round " << round;
    const auto c_small = space->QueryCourse(small, {2, 5}, {8, 5});
    const auto c_large = space->QueryCourse(large, {2, 5}, {8, 5});
    const auto c_none = space->QueryCourse(none, {2, 5}, {8, 5});
    ASSERT_GE(c_small.size(), 3u);
    ASSERT_GE(c_large.size(), 3u);
    EXPECT_GE(clearance(c_small), 0.1 - 1e-6);
    EXPECT_GE(clearance(c_large), 0.5 - 1e-6);
    EXPECT_GT(length(c_large), length(c_small)) << "a larger body walks further round";
    EXPECT_GT(length(c_small), length(c_none) - 1e-9) << "an actor with no radius has no clearance";
    EXPECT_GE(c_none.size(), 3u) << "the footprint itself still blocks";
  }
}

TEST(EmbodimentRegistration, LegacyAnchorFeedsAreRefusedForABoneBoundAnchor) {
  Rig r;
  // SetAnchorWorldPose leaves a bone-bound anchor alone; an unregistered anchor still takes it.
  r.space->SetAnchorWorldPose(r.actor, "pelvis", Eigen::Vector3d(9, 9, 9), 0.0);
  EXPECT_EQ(r.space->GetActor(r.actor)->GetAnchorWorldPoseOverride("pelvis"), nullptr);
  r.space->AddActorAnchors(r.actor, {SubjectAnchor("chest", Eigen::Vector3d(0, 0, 1.3))});
  r.space->SetAnchorWorldPose(r.actor, "chest", Eigen::Vector3d(1, 2, 3), 0.0);
  ASSERT_NE(r.space->GetActor(r.actor)->GetAnchorWorldPoseOverride("chest"), nullptr);
  // The composed feed is the only way in, and only for a bone-bound anchor.
  EXPECT_TRUE(r.space->SetBoneAnchorWorldPose(r.actor, "pelvis", Eigen::Vector3d(1, 1, 1), 0.0));
  EXPECT_NE(r.space->GetActor(r.actor)->GetAnchorWorldPoseOverride("pelvis"), nullptr);
  EXPECT_FALSE(r.space->SetBoneAnchorWorldPose(r.actor, "chest", Eigen::Vector3d(1, 1, 1), 0.0));
}

TEST(EmbodimentRegistration, SourceBonesComposeToTheSameAnchorsAsTheReceiverSample) {
  Rig r;
  RigidTransform pelvis, wrist;
  pelvis.position = Eigen::Vector3d(0.4, 0.2, 0.636);
  pelvis.rotation = Eigen::Quaterniond(Eigen::AngleAxisd(1.1, Eigen::Vector3d::UnitZ()));
  wrist.position = Eigen::Vector3d(0.6, 0.5, 1.0);
  wrist.rotation = Eigen::Quaterniond(Eigen::AngleAxisd(-0.4, Eigen::Vector3d::UnitZ()));
  std::vector<Space::ActorObservation> obs;
  ASSERT_TRUE(r.space->ComposeBoneObservations(r.actor, {{"pelvis", pelvis}, {"right_wrist", wrist}}, &obs));
  ASSERT_EQ(obs.size(), 2u);  // pelvis and hand_r; head's bone absent, the root is not bone-bound
  EXPECT_EQ(obs[0].anchor_name, "pelvis");
  EXPECT_NEAR(obs[0].position.x(), 0.4 + std::cos(1.1) * 0.0634, 1e-14);
  EXPECT_NEAR(obs[0].position.y(), 0.2 + std::sin(1.1) * 0.0634, 1e-14);
  EXPECT_NEAR(obs[0].position.z(), 0.636 - 0.1860, 1e-14);
  EXPECT_NEAR(obs[0].yaw, 1.1, 1e-14);
  EXPECT_EQ(obs[1].anchor_name, "hand_r");
  const double c = std::cos(-0.4), s = std::sin(-0.4);
  EXPECT_NEAR(obs[1].position.x(), 0.6 + c * 0.0611 - s * 0.0403, 1e-14);
  EXPECT_NEAR(obs[1].position.y(), 0.5 + s * 0.0611 + c * 0.0403, 1e-14);
  EXPECT_NEAR(obs[1].position.z(), 1.0 - 0.0310, 1e-14);

  // The receiver's sample of the same bone pose is the same anchor.
  BonePoseInput in = Bone(0.4, 0.2, 0.636, 1.1);
  ASSERT_EQ(r.space->PublishBonePose(r.actor, r.embodiment.BoneId("pelvis"), in, 1.0, 1), EmbodiedPublishResult::kOk);
  EmbodiedSample sample;
  ASSERT_TRUE(r.space->SampleAnchorPose(r.actor, r.Id("pelvis"), &sample));
  EXPECT_NEAR(sample.x, obs[0].position.x(), 1e-14);
  EXPECT_NEAR(sample.y, obs[0].position.y(), 1e-14);
  EXPECT_NEAR(sample.z, obs[0].position.z(), 1e-14);

  // An unknown or repeated bone refuses the whole frame; no embodiment is false.
  std::vector<Space::ActorObservation> none;
  EXPECT_FALSE(r.space->ComposeBoneObservations(r.actor, {{"tail", pelvis}}, &none));
  EXPECT_FALSE(r.space->ComposeBoneObservations(r.actor, {{"pelvis", pelvis}, {"pelvis", pelvis}}, &none));
  Rig plain(false);
  EXPECT_FALSE(plain.space->ComposeBoneObservations(plain.actor, {{"pelvis", pelvis}}, &none));
  EXPECT_TRUE(none.empty());
}

TEST(EmbodimentRegistration, AnIdentityOffsetAnchorAgreesWithTheLegacyPortToDoubleRounding) {
  // The registered "head" (identity offset) against the same pose published the legacy way (an Euler anchor).
  Rig r;
  Rig legacy(false);
  const double yaw = 0.9;
  ASSERT_EQ(r.space->PublishBonePose(r.actor, r.embodiment.BoneId("head"), Bone(1, 2, 1.7, yaw), 1.0, 1),
            EmbodiedPublishResult::kOk);
  EmbodiedPoseInput in;
  in.x = 1;
  in.y = 2;
  in.z = 1.7;
  in.yaw = yaw;
  ASSERT_EQ(legacy.space->PublishAnchorPose(legacy.actor, legacy.Id("head"), in, 1.0, 1), EmbodiedPublishResult::kOk);
  EmbodiedSample a, b;
  ASSERT_TRUE(r.space->SampleAnchorPose(r.actor, r.Id("head"), &a));
  ASSERT_TRUE(legacy.space->SampleAnchorPose(legacy.actor, legacy.Id("head"), &b));
  EXPECT_DOUBLE_EQ(a.x, b.x);
  EXPECT_DOUBLE_EQ(a.y, b.y);
  EXPECT_DOUBLE_EQ(a.z, b.z);
  EXPECT_NEAR(a.yaw, b.yaw, 1e-15);
}

TEST(EmbodimentRegistration, AReaderNeverSeesATornBone) {
  Rig r;
  const int pelvis = r.embodiment.BoneId("pelvis");
  std::atomic<bool> stop{false};
  std::thread writer([&] {
    for (uint64_t n = 1; !stop.load(); ++n)
      r.space->PublishBonePose(r.actor, pelvis, Bone(double(n), double(n), double(n) + 0.186, 0.0), 1.0, n);
  });
  const uint32_t id = r.Id("pelvis");
  int seen = 0;
  for (int i = 0; i < 200000; ++i) {
    EmbodiedSample s;
    ASSERT_TRUE(r.space->SampleAnchorPose(r.actor, id, &s));
    if (!s.valid) continue;
    ++seen;
    // bone = (n, n, n + 0.186), yaw 0: anchor = (n + 0.0634, n, n). One publish, one n.
    ASSERT_NEAR(s.x - 0.0634, s.y, 1e-9);
    ASSERT_NEAR(s.z, s.y, 1e-9);
    ASSERT_EQ(static_cast<uint64_t>(std::llround(s.y)), s.sequence);
  }
  stop = true;
  writer.join();
  EXPECT_GT(seen, 0);
}

}  // namespace
}  // namespace spatialblend
