// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Source-only framework contract: no FC scene, receiver, contact annotation,
// generated pose, headset, or motion model is required.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>

#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace {

using Evidence = SourceInteractionEvidence;
using ReadResult = Space::SourceEvidenceReadResult;
constexpr auto kAffordance = RelationKernelType::kAvatarAffordancePose;

std::shared_ptr<SurfaceSpatialDefinition> Seat(double height = .5) {
  return std::make_shared<SurfaceSpatialDefinition>(
      std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
          {-1, -1, height}, {1, -1, height}, {1, 1, height}, {-1, 1, height}}));
}

// The evidence with only the rows whose referent is the seat (element 4).
Evidence SeatRowsOnly(Evidence e) {
  e.relations.erase(
      std::remove_if(e.relations.begin(), e.relations.end(),
                     [](const Evidence::Relation& r) {
                       return r.path.empty() || r.path.back().element != 4;
                     }),
      e.relations.end());
  return e;
}

class SourceEvidenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    space.SetSpaceId(7);
    // Space -> region -> object -> affordance, with distinct local/shared IDs.
    for (int id = 1; id <= 4; ++id) {
      ASSERT_TRUE(space.AddElement(id, id == 1 ? kInvalidElementId : id - 1));
      ASSERT_TRUE(space.SetElementSharedId(id, 100 + id));
    }
    ASSERT_TRUE(space.SetElementTag(4, "Sittable"));
    ASSERT_TRUE(space.SetElementSpatialDefinition(4, Seat()));
    ASSERT_TRUE(space.AddElement(5, 2));
    ASSERT_TRUE(space.SetElementSharedId(5, 500));
    // The actor element is the container of the seat. A scope id no longer
    // points a binding at element 4, so the actor has to stand in that
    // container for spacing and affordance to see it.
    actor = space.CreateActor(
        {SubjectAnchor("navigation"),
         SubjectAnchor("body_origin", Eigen::Vector3d::Zero(),
                       {StateType::kPosition3D, StateType::kRotationYaw}),
         SubjectAnchor("hand", Eigen::Vector3d::Zero(),
                       {StateType::kPosition3D, StateType::kRotationYaw})});
    ASSERT_TRUE(space.SetActorElement(actor, 2));
    ASSERT_TRUE(
        space.BindOwned(actor, "body_origin", kAffordance, 4, "Sittable"));
    ASSERT_TRUE(space.BeginActorObservationEpoch(actor, "navigation", &epoch));
  }

  void Enable() {
    ASSERT_TRUE(space.EnableSourceInteractionEvidence(actor, epoch, true));
  }
  void Observe(const Eigen::Vector3d& body, int64_t sequence = 0) {
    ASSERT_TRUE(space.ApplyActorObservationFrame(
        actor, epoch, sequence, sequence * .1, {{"body_origin", body, .25}}));
  }
  Evidence Read(int64_t sequence = 0) {
    Evidence result;
    EXPECT_EQ(space.TryCopySourceInteractionEvidence(
                  actor, epoch, sequence, Evidence::kMaximumAgeNs, &result),
              ReadResult::kOk);
    return result;
  }
  ReadResult ReadFailure(int64_t sequence = 0) {
    Evidence result;
    result.epoch = 99;
    result.anchors.resize(1);
    const auto status = space.TryCopySourceInteractionEvidence(
        actor, epoch, sequence, Evidence::kMaximumAgeNs, &result);
    EXPECT_NE(status, ReadResult::kOk);
    EXPECT_EQ(result.epoch, 0u);
    EXPECT_TRUE(result.anchors.empty());
    EXPECT_TRUE(result.relations.empty());
    return status;
  }

  int64_t now = 1000000000;
  Space space{[this] { return now; }};
  spatialblend::entity_id_t actor = 0;
  uint64_t epoch = 0;
};

TEST_F(SourceEvidenceTest,
       OptInOnlyCapturesFutureFramesAndPreservesMissingRoot) {
  Observe({0, 0, 1});
  EXPECT_EQ(ReadFailure(), ReadResult::kUnavailable);
  Enable();
  EXPECT_EQ(ReadFailure(), ReadResult::kUnavailable);
  Observe({0, 0, 1}, 1);
  const auto captured = Read(1);
  EXPECT_EQ(captured.space, 7);
  EXPECT_EQ(captured.actor, actor);
  EXPECT_EQ(captured.actor_element, 2);
  EXPECT_EQ(captured.actor_shared, 102);
  EXPECT_EQ(captured.epoch, epoch);
  EXPECT_EQ(captured.sequence, 1);
  EXPECT_DOUBLE_EQ(captured.source_time_seconds, .1);
  EXPECT_EQ(captured.receipt_monotonic_ns, now);
  ASSERT_EQ(captured.anchors.size(), 3u);
  EXPECT_TRUE(captured.anchors[0].locomotion_root);
  EXPECT_FALSE(captured.anchors[0].observed);
  EXPECT_EQ(captured.anchors[0].position_dimension, 2);
  EXPECT_TRUE(captured.anchors[1].observed);
  EXPECT_FALSE(captured.anchors[1].locomotion_root);
  EXPECT_EQ(captured.anchors[1].position_dimension, 3);
  EXPECT_DOUBLE_EQ(captured.anchors[1].position.z(), 1);
  EXPECT_DOUBLE_EQ(captured.anchors[1].yaw, .25);
  EXPECT_FALSE(captured.anchors[2].observed);
  Enable();
  EXPECT_EQ(Read(1).receipt_monotonic_ns, now);
  EXPECT_TRUE(space.EnableSourceInteractionEvidence(actor, epoch, false));
  EXPECT_EQ(ReadFailure(1), ReadResult::kUnavailable);
}

TEST_F(SourceEvidenceTest, ProjectionHierarchyAndContactRemainDistinct) {
  Enable();
  Observe({0, 0, 1});
  const auto captured = Read();
  ASSERT_EQ(captured.relations.size(), 1u);
  const auto& row = captured.relations[0];
  EXPECT_EQ(row.binding_ordinal, 0u);
  EXPECT_EQ(row.anchor_ordinal, 1u);
  EXPECT_EQ(row.kernel, kAffordance);
  // The supplied scope id is ignored. The family scope is what was used.
  EXPECT_EQ(row.explicit_scope, kInvalidElementId);
  EXPECT_EQ(row.tag_filter, "Sittable");
  EXPECT_TRUE(row.has_projection);
  EXPECT_TRUE(row.bounded_planar_support);
  EXPECT_TRUE(row.phase_identity_unambiguous);
  EXPECT_EQ(row.phase_referent_id, 104);
  EXPECT_EQ(row.phase, RelationPhase::kActive);
  EXPECT_NEAR(row.anchor_support_distance, .5, 1e-12);
  EXPECT_NEAR(row.proximity, 1 / 1.5, 1e-12);
  EXPECT_DOUBLE_EQ(row.normalized_weight, 1);
  EXPECT_NEAR(row.projected_point.z(), .5, 1e-12);
  ASSERT_EQ(row.path.size(), 4u);
  for (size_t i = 0; i < row.path.size(); ++i) {
    EXPECT_EQ(row.path[i].element, i + 1);
    EXPECT_EQ(row.path[i].shared, i + 101);
    EXPECT_EQ(row.path[i].parent, i == 0 ? kInvalidElementId : i);
  }
  EXPECT_EQ(row.path.back().geometry_role, GeometryRole::kSurface);
  Observe({0, 0, .55}, 1);
  EXPECT_EQ(Read(1).relations[0].phase, RelationPhase::kContact);
  Observe({0, 0, .60}, 2);
  EXPECT_EQ(Read(2).relations[0].phase, RelationPhase::kContact);
  Observe({2, 0, .5}, 3);  // Outside polygon, though on its infinite plane.
  const auto outside = Read(3);
  EXPECT_NEAR(outside.relations[0].anchor_support_distance, 1, 1e-12);
  EXPECT_NE(outside.relations[0].phase, RelationPhase::kContact);
}

TEST_F(SourceEvidenceTest,
       MultipleBindingsPreserveExactAnchorKernelOrdinalAndSourcePath) {
  ASSERT_TRUE(space.BindOwned(
      actor, "navigation", RelationKernelType::kAvatarSpacing, 4, "Sittable"));
  ASSERT_TRUE(space.BindOwned(actor, "hand", RelationKernelType::kAvatarFacing,
                              4, "Sittable"));
  Enable();
  ASSERT_TRUE(space.ApplyActorObservationFrame(actor, epoch, 0, 0.0,
                                               {{"navigation", {0, 0, 0}, 0.0},
                                                {"body_origin", {0, 0, 1}, .25},
                                                {"hand", {.2, 0, 1}, .5}}));
  const auto evidence = Read();
  ASSERT_EQ(evidence.anchors.size(), 3u);
  // Spacing's reachable set does not include the sittable surface just
  // because the bind named element 4. Affordance (the container's subtree)
  // and facing (global) do. D6: the tag does not narrow
  // Spacing or Facing, so those bindings also carry rows for other referents in
  // their scope; the association under test is that of the rows that reach the seat.
  const auto captured = SeatRowsOnly(evidence);
  ASSERT_EQ(captured.relations.size(), 2u);
  const std::array<std::tuple<size_t, size_t, RelationKernelType>, 2> expected{{
      {0, 1, kAffordance},
      {2, 2, RelationKernelType::kAvatarFacing},
  }};
  for (size_t i = 0; i < expected.size(); ++i) {
    const auto& row = captured.relations[i];
    EXPECT_EQ(row.binding_ordinal, std::get<0>(expected[i]));
    EXPECT_EQ(row.anchor_ordinal, std::get<1>(expected[i]));
    EXPECT_EQ(row.kernel, std::get<2>(expected[i]));
    EXPECT_EQ(captured.anchors[row.anchor_ordinal].name,
              row.binding_ordinal == 0   ? "body_origin"
              : row.binding_ordinal == 1 ? "navigation"
                                         : "hand");
    ASSERT_EQ(row.path.size(), 4u);
    EXPECT_EQ(row.path.back().element, 4);
    EXPECT_EQ(row.path.back().tag, "Sittable");
  }
}

TEST_F(SourceEvidenceTest,
       ImmutableBindingAnchorAssociationSurvivesRebindMissingAndRetry) {
  const auto source = space.CreateActor(
      {SubjectAnchor("locomotion"),
       SubjectAnchor("head_facing", Eigen::Vector3d::Zero(),
                     {StateType::kPosition3D, StateType::kRotationYaw}),
       SubjectAnchor("seat_contact", Eigen::Vector3d::Zero(),
                     {StateType::kPosition3D, StateType::kRotationYaw})});
  // Element 5 is a leaf beside the seat. The actor has to be the container
  // (2) for affordance to see the sittable; the scope id does not.
  ASSERT_TRUE(space.SetActorElement(source, 2));
  ASSERT_TRUE(space.BindOwned(
      source, "locomotion", RelationKernelType::kAvatarSpacing, 4, "Sittable"));
  ASSERT_TRUE(space.BindOwned(
      source, "head_facing", RelationKernelType::kAvatarFacing, 4, "Sittable"));
  ASSERT_TRUE(
      space.BindOwned(source, "seat_contact", kAffordance, 4, "Sittable"));
  uint64_t source_epoch = 0;
  ASSERT_TRUE(
      space.BeginActorObservationEpoch(source, "locomotion", &source_epoch));
  ASSERT_TRUE(
      space.EnableSourceInteractionEvidence(source, source_epoch, true));

  const auto observe = [&](int64_t sequence, bool include_head) {
    std::vector<Space::ActorObservation> observations = {
        {"locomotion", {0, 0, 0}, 0},
        {"seat_contact", {0, 0, .55}, .25},
    };
    if (include_head) observations.push_back({"head_facing", {.2, 0, 1}, .5});
    ASSERT_TRUE(space.ApplyActorObservationFrame(source, source_epoch, sequence,
                                                 sequence * .1, observations));
  };
  const auto read = [&](int64_t sequence, Evidence* out) {
    return space.TryCopySourceInteractionEvidence(
        source, source_epoch, sequence, Evidence::kMaximumAgeNs, out);
  };
  const auto expect_association = [&](const Evidence& all,
                                      size_t expected_relations) {
    const Evidence captured = SeatRowsOnly(all);
    const std::array<
        std::tuple<size_t, size_t, RelationKernelType, const char*>, 3>
        expected{{{1, 1, RelationKernelType::kAvatarFacing, "head_facing"},
                  {2, 2, kAffordance, "seat_contact"},
                  {3, 2, RelationKernelType::kAvatarFacing, "seat_contact"}}};
    ASSERT_EQ(captured.relations.size(), expected_relations);
    for (size_t i = 0; i < expected_relations; ++i) {
      const auto& row = captured.relations[i];
      EXPECT_EQ(row.binding_ordinal, std::get<0>(expected[i]));
      EXPECT_EQ(row.anchor_ordinal, std::get<1>(expected[i]));
      EXPECT_EQ(row.kernel, std::get<2>(expected[i]));
      ASSERT_LT(row.anchor_ordinal, captured.anchors.size());
      EXPECT_EQ(captured.anchors[row.anchor_ordinal].name,
                std::get<3>(expected[i]));
      ASSERT_FALSE(row.path.empty());
      EXPECT_EQ(row.path.back().element, 4);
    }
  };

  observe(0, true);
  Evidence captured;
  ASSERT_EQ(read(0, &captured), ReadResult::kOk);
  // Spacing named the seat and does not reach it. Facing and affordance do.
  expect_association(captured, 2);

  // Adding a binding invalidates the prior receipt; it cannot be paired with
  // a later binding vector or recreated by relation/name order.
  ASSERT_TRUE(space.BindOwned(source, "seat_contact",
                              RelationKernelType::kAvatarFacing, 4,
                              "Sittable"));
  EXPECT_EQ(read(0, &captured), ReadResult::kUnavailable);
  EXPECT_TRUE(captured.relations.empty());
  observe(1, true);
  ASSERT_EQ(read(1, &captured), ReadResult::kOk);
  expect_association(captured, 3);

  // A missing selected anchor withdraws only its binding's row. Retrying the
  // receipt must not cross-associate another row with the missing head anchor
  // or reuse the prior head relation.
  observe(2, false);
  ASSERT_EQ(read(2, &captured), ReadResult::kOk);
  // Head facing is absent. Spacing still has no row. Affordance and the
  // seat facing remain, each on the seat anchor.
  captured = SeatRowsOnly(captured);
  ASSERT_EQ(captured.relations.size(), 2u);
  for (const auto& row : captured.relations) {
    EXPECT_NE(row.binding_ordinal, 1u);
    EXPECT_NE(row.binding_ordinal, 0u);
    ASSERT_LT(row.anchor_ordinal, captured.anchors.size());
    EXPECT_EQ(row.anchor_ordinal, 2u);
    EXPECT_TRUE(row.kernel == kAffordance ||
                row.kernel == RelationKernelType::kAvatarFacing);
  }
  ASSERT_EQ(read(2, &captured), ReadResult::kOk);
  observe(3, true);
  ASSERT_EQ(read(3, &captured), ReadResult::kOk);
  expect_association(captured, 3);
}

TEST_F(SourceEvidenceTest, MissingBodyIsCurrentUnavailableRowNotHeldContact) {
  Enable();
  Observe({0, 0, .5});
  Observe({0, 0, .5}, 1);
  EXPECT_EQ(Read(1).relations[0].phase, RelationPhase::kContact);
  ASSERT_TRUE(space.ApplyActorObservationFrame(
      actor, epoch, 2, .2,
      {{"navigation", {0, 0, .5}, 0}, {"hand", {0, 0, .5}, 0}}));
  const auto absent = Read(2);
  EXPECT_TRUE(absent.relations.empty());
  EXPECT_FALSE(absent.anchors[1].observed);
  EXPECT_TRUE(absent.anchors[0].observed);
  EXPECT_DOUBLE_EQ(absent.anchors[0].position.z(), 0);  // Planar root only.
  Observe({0, 0, .5}, 3);
  EXPECT_EQ(Read(3).relations[0].phase, RelationPhase::kActive);
}

TEST_F(SourceEvidenceTest,
       FrozenCopiesAreDetachedAndDoNotDoubleTransformSurface) {
  auto transform = Eigen::Matrix4d::Identity().eval();
  transform(0, 3) = 10;
  ASSERT_TRUE(space.SetElementTransform(2, transform));
  transform(0, 3) = 20;
  ASSERT_TRUE(space.SetElementTransform(3, transform));
  Enable();
  Observe({0, 0, 1});
  auto copy = Read();
  ASSERT_EQ(copy.relations.size(), 1u);
  const auto frozen = copy;
  EXPECT_DOUBLE_EQ(copy.relations[0].path.back().world_transform(0, 3), 30);
  EXPECT_NEAR(copy.relations[0].projected_point.x(), 0, 1e-12);
  EXPECT_NEAR(copy.relations[0].support_to_world(0, 3), 0, 1e-12);
  copy.relations[0].path.back().tag = "changed copy";
  copy.relations[0].support_boundary[0].setZero();
  EXPECT_EQ(Read().relations[0].path.back().tag, "Sittable");
  EXPECT_TRUE(Read().relations[0].support_boundary[0].isApprox(
      frozen.relations[0].support_boundary[0]));
  ASSERT_TRUE(space.SetElementSpatialDefinition(4, Seat(.7)));
  EXPECT_EQ(ReadFailure(), ReadResult::kStale);
  EXPECT_NEAR(frozen.relations[0].projected_point.z(), .5, 1e-12);
  Observe({0, 0, 1}, 1);
  EXPECT_NEAR(Read(1).relations[0].projected_point.z(), .7, 1e-12);
}

TEST_F(SourceEvidenceTest,
       LocalPatchFrameIsComposedButVolumeIsNotPlanarSupport) {
  auto box = std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-1, -1, 0), Eigen::Vector3d(1, 1, .5));
  auto transform = Eigen::Matrix4d::Identity().eval();
  transform(0, 3) = 4;
  transform(2, 3) = 1;
  ASSERT_TRUE(space.SetElementTransform(4, transform));
  ASSERT_TRUE(space.SetElementSpatialDefinition(4, box));
  Enable();
  Observe({4, 0, 1.55});
  auto captured = Read();
  ASSERT_EQ(captured.relations.size(), 1u);
  EXPECT_TRUE(captured.relations[0].has_projection);
  EXPECT_FALSE(captured.relations[0].bounded_planar_support);
  SubRegion patch;
  patch.name = "support_patch";
  patch.boundary = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{-.5, -.5}, {.5, -.5}, {.5, .5}, {-.5, .5}});
  patch.local_transform(2, 3) = .5;
  ASSERT_TRUE(space.AddElementSubRegion(4, patch));
  Observe({4, 0, 1.55}, 1);
  captured = Read(1);
  ASSERT_EQ(captured.relations.size(), 1u);
  EXPECT_TRUE(captured.relations[0].bounded_planar_support);
  EXPECT_EQ(captured.relations[0].sub_region, "support_patch");
  EXPECT_NEAR(captured.relations[0].support_to_world(0, 3), 4, 1e-12);
  EXPECT_NEAR(captured.relations[0].projected_point.z(), 1.5, 1e-12);
  EXPECT_NEAR(captured.relations[0].anchor_support_distance, .05, 1e-12);
}

TEST_F(SourceEvidenceTest, ExactFrameAndExclusiveReceiptAgeNeverRestamp) {
  Enable();
  Observe({0, 0, 1});
  const auto original = Read();
  EXPECT_EQ(ReadFailure(1), ReadResult::kStale);
  now += Evidence::kMaximumAgeNs - 1;
  Enable();
  EXPECT_EQ(Read().receipt_monotonic_ns, original.receipt_monotonic_ns);
  ++now;
  EXPECT_EQ(ReadFailure(), ReadResult::kStale);
  now = original.receipt_monotonic_ns - 1;
  EXPECT_EQ(ReadFailure(), ReadResult::kStale);
  now = original.receipt_monotonic_ns + Evidence::kMaximumAgeNs;
  Observe({0, 0, 1}, 1);
  EXPECT_EQ(Read(1).receipt_monotonic_ns, now);
  EXPECT_EQ(ReadFailure(), ReadResult::kStale);
}

TEST_F(SourceEvidenceTest,
       RejectedFrameAndMalformedBatchDoNotChangePriorEvidence) {
  Enable();
  Observe({0, 0, 1});
  const auto original = Read();
  EXPECT_FALSE(space.ApplyActorObservationFrame(
      actor, epoch, 1, .1,
      {{"body_origin", {0, 0, .5}, 0}, {"unknown", {0, 0, 0}, 0}}));
  EXPECT_FALSE(space.ApplyActorObservationBatch(
      {{actor, epoch, 1, .1, {{"body_origin", {0, 0, .5}, 0}}},
       {9999, epoch, 1, .1, {}}},
      {}));
  EXPECT_FALSE(space.ApplyActorObservationFrame(actor, epoch, 0, 0, {}));
  const auto unchanged = Read();
  EXPECT_EQ(unchanged.sequence, original.sequence);
  EXPECT_EQ(unchanged.receipt_monotonic_ns, original.receipt_monotonic_ns);
  EXPECT_EQ(unchanged.relations[0].phase, RelationPhase::kActive);
  EXPECT_DOUBLE_EQ(unchanged.relations[0].proximity,
                   original.relations[0].proximity);
}

TEST_F(SourceEvidenceTest,
       AtomicBatchCapturesBothActorsAfterAllScenePoseCommits) {
  const auto other =
      space.CreateActor({SubjectAnchor("navigation"),
                         SubjectAnchor("body", Eigen::Vector3d::Zero(),
                                       {StateType::kPosition3D})});
  // The first actor already owns element 2, and a batch rejects two subjects
  // on one element. The parent (1) is a different owner whose subtree still
  // contains the seat. The tag, not the scope id, selects that seat.
  ASSERT_TRUE(space.SetActorElement(other, 1));
  ASSERT_TRUE(space.BindOwned(other, "body", kAffordance, 4, "Sittable"));
  uint64_t other_epoch;
  ASSERT_TRUE(
      space.BeginActorObservationEpoch(other, "navigation", &other_epoch));
  ASSERT_TRUE(space.EnableSourceInteractionEvidence(other, other_epoch, true));
  Enable();
  ASSERT_TRUE(space.ApplyActorObservationBatch(
      {{actor,
        epoch,
        0,
        0,
        {{"navigation", {2, 0, 0}, 0}, {"body_origin", {0, 0, 1}, 0}}},
       {other,
        other_epoch,
        0,
        0,
        {{"navigation", {3, 0, 0}, 0}, {"body", {0, 0, .55}, 0}}}},
      {}));
  const auto first = Read();
  Evidence second;
  ASSERT_EQ(space.TryCopySourceInteractionEvidence(
                other, other_epoch, 0, Evidence::kMaximumAgeNs, &second),
            ReadResult::kOk);
  EXPECT_EQ(first.tree_revision, second.tree_revision);
  EXPECT_EQ(first.support_revision, second.support_revision);
  EXPECT_EQ(first.receipt_monotonic_ns, second.receipt_monotonic_ns);
  EXPECT_NEAR(first.relations[0].anchor_support_distance, .5, 1e-12);
  EXPECT_NEAR(second.relations[0].anchor_support_distance, .05, 1e-12);
  EXPECT_EQ(first.relations[0].phase, RelationPhase::kActive);
  EXPECT_EQ(second.relations[0].phase, RelationPhase::kActive);
  // A bad last member cannot publish a valid prefix or withdraw either copy.
  EXPECT_FALSE(space.ApplyActorObservationBatch(
      {{actor, epoch, 1, .1, {}}, {other, other_epoch, 0, 0, {}}}, {}));
  EXPECT_EQ(Read().sequence, 0);
  EXPECT_EQ(space.TryCopySourceInteractionEvidence(
                other, other_epoch, 0, Evidence::kMaximumAgeNs, &second),
            ReadResult::kOk);
}

TEST_F(SourceEvidenceTest, CallbackCannotReadPartialSnapshotOrRenewReceipt) {
  Enable();
  int calls = 0;
  space.RegisterOnRelationEvent([&](const RelationEvent&) {
    ++calls;
    EXPECT_EQ(ReadFailure(), ReadResult::kBusy);
    now += 100;
  });
  const auto admission_receipt = now;
  Observe({0, 0, 1});
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(Read().receipt_monotonic_ns, admission_receipt);
}

TEST_F(SourceEvidenceTest,
       EffectiveLegacyMutationsInvalidateUntilGenuineFrame) {
  Enable();
  const std::vector<std::function<void()>> mutations = {
      [&] { space.SetAnchorWorldPose(actor, "body_origin", {0, 0, .5}, 0); },
      [&] { space.ClearAnchorWorldPose(actor, "body_origin"); },
      [&] { EXPECT_TRUE(space.SetAnchorActive(actor, "body_origin", false)); },
      [&] {
        EXPECT_TRUE(
            space.ClearActorObservationAnchor(actor, epoch, "body_origin"));
      },
      [&] { space.UpdateActorState(actor, {0, 0}, 0); }};
  // (A pose the driver publishes through the sampling port is not a source
  // mutation: it neither invalidates nor supplies evidence; see
  // PublishedEmbodiedPoseNeverContinuesOrInvalidatesContact.)
  int64_t sequence = 0;
  for (const auto& mutate : mutations) {
    SCOPED_TRACE(sequence);
    Observe({0, 0, 1}, sequence);
    EXPECT_EQ(Read(sequence).sequence, sequence);
    mutate();
    EXPECT_EQ(ReadFailure(sequence), ReadResult::kUnavailable);
    ++sequence;
  }
  Observe({0, 0, 1}, sequence);
  EXPECT_EQ(Read(sequence).sequence, sequence);
}

TEST_F(SourceEvidenceTest, NoopAndRejectedMutationsDoNotInvalidate) {
  Enable();
  Observe({0, 0, 1});
  ASSERT_TRUE(space.SetElementTag(4, "Sittable"));
  ASSERT_TRUE(space.SetElementTransform(4, Eigen::Matrix4d::Identity()));
  // Repeating the actor's own element is a no-op. Element 5 is a different
  // leaf, so assigning it would be a real identity change.
  ASSERT_TRUE(space.SetActorElement(actor, 2));
  ASSERT_TRUE(space.SetAnchorActive(actor, "body_origin", true));
  ASSERT_TRUE(space.ClearActorObservationAnchor(actor, epoch, "hand"));
  EXPECT_FALSE(space.BindOwned(actor, "unknown", kAffordance, 4));
  EXPECT_FALSE(space.BindOwned(actor, "body_origin", kAffordance, 999));
  EXPECT_FALSE(space.BindOwned(actor, "body_origin",
                               static_cast<RelationKernelType>(99), 4));
  EXPECT_FALSE(space.SetElementTransform(4, Eigen::Matrix4d::Zero()));
  EXPECT_EQ(Read().sequence, 0);
}

TEST_F(SourceEvidenceTest, ResetBindingSceneIdentityAndRemovalFenceOldFrame) {
  Enable();
  Observe({0, 0, 1});
  ASSERT_TRUE(space.BindOwned(actor, "hand", kAffordance, 4));
  EXPECT_EQ(ReadFailure(), ReadResult::kUnavailable);
  Observe({0, 0, 1}, 1);
  ASSERT_TRUE(space.SetElementTag(4, "not_sittable"));
  EXPECT_EQ(ReadFailure(1), ReadResult::kStale);
  Observe({0, 0, 1}, 2);
  EXPECT_TRUE(Read(2).relations.empty());
  ASSERT_TRUE(space.SetElementTag(4, "Sittable"));
  Observe({0, 0, 1}, 3);
  ASSERT_TRUE(space.SetElementSharedId(4, 999));
  EXPECT_EQ(ReadFailure(3), ReadResult::kStale);
  Observe({0, 0, 1}, 4);
  EXPECT_EQ(Read(4).relations[0].path.back().shared, 999);
  ASSERT_TRUE(space.RemoveLocalElementSubtree(4));
  EXPECT_EQ(ReadFailure(4), ReadResult::kStale);
  const auto old_epoch = epoch;
  ASSERT_TRUE(space.BeginActorObservationEpoch(actor, "navigation", &epoch));
  EXPECT_NE(epoch, old_epoch);
  Observe({0, 0, 1});
  EXPECT_EQ(ReadFailure(), ReadResult::kUnavailable);  // Reset ends opt-in.
  Enable();
  EXPECT_EQ(ReadFailure(), ReadResult::kUnavailable);  // No held-frame replay.
  Observe({0, 0, 1}, 1);
  EXPECT_TRUE(Read(1).relations.empty());
  ASSERT_TRUE(space.DestroyActor(actor));
  EXPECT_EQ(ReadFailure(1), ReadResult::kUnavailable);
}

TEST_F(SourceEvidenceTest, SharedPhaseCollisionKeepsDistinctExactLocalPaths) {
  // The second row of the shared phase is a part of the first (a child of element 4), not a sibling
  // with the same footprint: sibling footprints may only touch.
  ASSERT_TRUE(space.AddElement(6, 4));
  ASSERT_TRUE(space.SetElementTag(6, "Sittable"));
  ASSERT_TRUE(space.SetElementSharedId(6, 104));
  ASSERT_TRUE(space.SetElementSpatialDefinition(6, Seat()));
  // The existing Sittable binding already sees every sittable in the
  // container. A second bind named at element 6 does not select only that
  // leaf, so it is not what distinguishes the two local paths.
  Enable();
  Observe({0, 0, .55});
  const auto captured = Read();
  ASSERT_EQ(captured.relations.size(), 2u);
  EXPECT_EQ(captured.relations[0].path.back().element, 4);
  EXPECT_EQ(captured.relations[1].path.back().element, 6);
  EXPECT_EQ(captured.relations[0].phase_referent_id,
            captured.relations[1].phase_referent_id);
  EXPECT_FALSE(captured.relations[0].phase_identity_unambiguous);
  EXPECT_FALSE(captured.relations[1].phase_identity_unambiguous);
}

TEST_F(SourceEvidenceTest, UnsharedIdentityNeverPretendsLocalIdIsShared) {
  ASSERT_TRUE(space.SetElementSharedId(4, kInvalidSharedElementId));
  // The actor stands on element 2. Clearing the sibling leaf does not
  // unshare the actor.
  ASSERT_TRUE(space.SetElementSharedId(2, kInvalidSharedElementId));
  Enable();
  Observe({0, 0, 1});
  const auto captured = Read();
  EXPECT_EQ(captured.actor_shared, kInvalidSharedElementId);
  // SP-D7-03 (MA-1: unshared elements are never referents): an element with no
  // shared id has no counterpart on any receiver, so it is no referent and no relation is captured
  // for it ("never pretends the local id is shared" holds trivially).
  EXPECT_EQ(captured.relations.size(), 0u);
}

TEST_F(SourceEvidenceTest, BorrowedBindingPermanentlyDisablesCertifiedCopies) {
  Enable();
  Observe({0, 0, 1});
  EXPECT_EQ(space.GetBindings(actor).size(), 1u);
  EXPECT_EQ(ReadFailure(), ReadResult::kUnavailable);
  EXPECT_FALSE(space.EnableSourceInteractionEvidence(actor, epoch, true));
  ASSERT_TRUE(space.BindOwned(actor, "hand", kAffordance, 4));
  Observe({0, 0, 1}, 1);
  EXPECT_EQ(ReadFailure(1), ReadResult::kUnavailable);
}

TEST_F(SourceEvidenceTest, BorrowedScenePermanentlyDisablesCertifiedCopies) {
  Enable();
  Observe({0, 0, 1});
  ASSERT_NE(space.GetElementTree(), nullptr);
  EXPECT_EQ(ReadFailure(), ReadResult::kUnavailable);
  EXPECT_FALSE(space.EnableSourceInteractionEvidence(actor, epoch, true));
  Observe({0, 0, 1}, 1);
  EXPECT_EQ(ReadFailure(1), ReadResult::kUnavailable);
}

TEST_F(SourceEvidenceTest,
       OwnedActorIdentityAndLayoutChangesInvalidateImmediately) {
  ASSERT_TRUE(space.AddElement(6, 2));
  Enable();
  Observe({0, 0, 1});
  ASSERT_TRUE(space.SetActorElement(actor, 6));
  EXPECT_EQ(ReadFailure(), ReadResult::kUnavailable);
  Observe({0, 0, 1}, 1);
  EXPECT_EQ(Read(1).actor_element, 6);
  ASSERT_TRUE(space.AddActorAnchors(actor, {SubjectAnchor("another")}, true));
  EXPECT_EQ(ReadFailure(1), ReadResult::kUnavailable);
  Observe({0, 0, 1}, 2);
  EXPECT_EQ(Read(2).anchors.size(), 4u);
  space.SetSpaceId(8);
  EXPECT_EQ(ReadFailure(2), ReadResult::kUnavailable);
  space.SetSpaceId(7);
  EXPECT_EQ(ReadFailure(2), ReadResult::kUnavailable);
  Enable();
  EXPECT_EQ(ReadFailure(2), ReadResult::kUnavailable);
  Observe({0, 0, 1}, 3);
  EXPECT_EQ(Read(3).space, 7);
}

TEST_F(SourceEvidenceTest, EscapedActorAndPointerReturningBindCannotOptIn) {
  const auto other = space.CreateActor({SubjectAnchor("navigation")});
  ASSERT_TRUE(space.SetActorElement(other, 5));
  uint64_t other_epoch;
  ASSERT_TRUE(
      space.BeginActorObservationEpoch(other, "navigation", &other_epoch));
  ASSERT_NE(space.BindScoped(other, "navigation", kAffordance, 4), nullptr);
  EXPECT_FALSE(space.EnableSourceInteractionEvidence(other, other_epoch, true));
  Enable();
  Observe({0, 0, 1});
  ASSERT_NE(space.GetActor(actor), nullptr);
  EXPECT_EQ(ReadFailure(), ReadResult::kUnavailable);
  EXPECT_FALSE(space.EnableSourceInteractionEvidence(actor, epoch, true));
}

TEST_F(SourceEvidenceTest, ExcessiveRelationCountAndPathDepthProduceNoPrefix) {
  Enable();
  Observe({0, 0, 1});
  // An overly deep candidate must not make traversal or the exported value
  // unbounded. The accepted input clock is still a genuine source admission.
  element_id_t parent = 3;
  for (size_t i = 0; i < Evidence::kMaxPathDepth; ++i) {
    const auto id = static_cast<element_id_t>(1000 + i);
    ASSERT_TRUE(space.AddElement(id, parent));
    parent = id;
  }
  ASSERT_TRUE(space.ReparentElement(4, parent));
  Observe({0, 0, 1}, 1);
  EXPECT_EQ(ReadFailure(1), ReadResult::kUnavailable);
  ASSERT_TRUE(space.ReparentElement(4, 3));
  Observe({0, 0, 1}, 2);
  EXPECT_EQ(Read(2).relations.size(), 1u);
  for (size_t i = 0; i < Evidence::kMaxRelations; ++i)
    ASSERT_TRUE(space.BindOwned(actor, "body_origin", kAffordance, 4));
  Observe({0, 0, 1}, 3);
  EXPECT_EQ(ReadFailure(3), ReadResult::kUnavailable);
}

TEST_F(SourceEvidenceTest, LegacyContactCannotWarmGenuineSourceHysteresis) {
  Enable();
  Observe({0, 0, 1});
  RelationPhase legacy_phase = RelationPhase::kDormant;
  space.RegisterOnRelationEvent([&](const RelationEvent& event) {
    if (event.anchor_name == "body_origin") legacy_phase = event.to;
  });
  space.SetAnchorWorldPose(actor, "body_origin", {0, 0, .5}, 0);
  space.UpdateActorState(actor, {0, 0}, 0);
  EXPECT_EQ(legacy_phase, RelationPhase::kContact);
  EXPECT_EQ(ReadFailure(), ReadResult::kUnavailable);
  // This is in the hysteresis-only band: it may retain a REAL contact but
  // must not inherit the preceding legacy/generated contact.
  Observe({0, 0, .60}, 1);
  EXPECT_EQ(Read(1).relations[0].phase, RelationPhase::kActive);
  Observe({0, 0, .60}, 2);
  EXPECT_EQ(Read(2).relations[0].phase, RelationPhase::kDriving);
  Observe({0, 0, .55}, 3);
  EXPECT_EQ(Read(3).relations[0].phase, RelationPhase::kContact);
}

TEST_F(SourceEvidenceTest,
       OptInCannotBorrowPreOptInContactAndBadBatchCannotReset) {
  Observe({0, 0, 1});
  space.SetAnchorWorldPose(actor, "body_origin", {0, 0, .5}, 0);
  space.UpdateActorState(actor, {0, 0}, 0);
  Enable();
  int events = 0;
  space.RegisterOnRelationEvent([&](const RelationEvent&) { ++events; });
  EXPECT_FALSE(space.ApplyActorObservationBatch(
      {{actor, epoch, 1, .1, {{"body_origin", {0, 0, .60}, 0}}},
       {9999, 1, 0, 0, {}}},
      {}));
  EXPECT_EQ(events, 0);
  EXPECT_EQ(ReadFailure(), ReadResult::kUnavailable);
  Observe({0, 0, .60}, 1);
  EXPECT_EQ(Read(1).relations[0].phase, RelationPhase::kActive);
  EXPECT_EQ(events, 1);
}

TEST_F(SourceEvidenceTest,
       OnlyAcceptedRemoteWriteBreaksSourceEvidenceLifetime) {
  Enable();
  Observe({0, 0, .55});
  Observe({0, 0, .55}, 1);
  EXPECT_EQ(Read(1).relations[0].phase, RelationPhase::kContact);
  RemoteRelationUpdate remote;
  // Element 2 is the actor. Shared id 500 belongs to the sibling leaf.
  remote.actor_shared_id = 102;
  remote.source_space_id = 8;
  remote.timing = RelationUpdateTiming::kObservationReset;
  remote.source_epoch = 1;
  remote.sequence = 1;
  remote.source_time = std::numeric_limits<double>::quiet_NaN();
  space.OnRemoteUpdate(remote);
  EXPECT_EQ(Read(1).relations[0].phase, RelationPhase::kContact);
  remote.source_time = 0;
  space.OnRemoteUpdate(remote);
  EXPECT_EQ(ReadFailure(1), ReadResult::kUnavailable);
  Observe({0, 0, .60}, 2);
  EXPECT_EQ(Read(2).relations[0].phase, RelationPhase::kActive);
  space.OnRemoteUpdate(remote);  // Same-epoch reset/replay is rejected.
  EXPECT_EQ(Read(2).relations[0].phase, RelationPhase::kActive);
  remote.timing = RelationUpdateTiming::kTimestampedObservation;
  remote.sequence = 2;
  space.OnRemoteUpdate(remote);
  EXPECT_EQ(ReadFailure(2), ReadResult::kUnavailable);
  Observe({0, 0, .60}, 3);
  EXPECT_EQ(Read(3).relations[0].phase, RelationPhase::kActive);
  remote.timing = RelationUpdateTiming::kLegacy;
  remote.sequence = 3;
  space.OnRemoteUpdate(remote);
  EXPECT_EQ(ReadFailure(3), ReadResult::kUnavailable);
}

TEST_F(SourceEvidenceTest, GenuineAnchorClearPreservesIndependentBodyContact) {
  Enable();
  Observe({0, 0, .55});
  ASSERT_TRUE(space.ApplyActorObservationFrame(
      actor, epoch, 1, .1,
      {{"body_origin", {0, 0, .55}, 0}, {"hand", {0, 0, .5}, 0}}));
  EXPECT_EQ(Read(1).relations[0].phase, RelationPhase::kContact);
  ASSERT_TRUE(space.ClearActorObservationAnchor(actor, epoch, "hand"));
  EXPECT_EQ(ReadFailure(1), ReadResult::kUnavailable);
  Observe({0, 0, .60}, 2);
  EXPECT_EQ(Read(2).relations[0].phase, RelationPhase::kContact);
}

TEST_F(SourceEvidenceTest,
       PublishedEmbodiedPoseNeverContinuesOrInvalidatesContact) {
  // The driver's embodied pose (the sampling port) is not a genuine body
  // sample: publishing it must not continue a contact, and it is not a source
  // mutation, so the evidence of the last genuine frame stays as it was. The
  // contact only continues through the next genuine observation.
  Enable();
  Observe({0, 0, .55});
  Observe({0, 0, .55}, 1);
  EXPECT_EQ(Read(1).relations[0].phase, RelationPhase::kContact);
  EmbodiedPoseInput published;
  published.epoch = 1;
  ASSERT_EQ(space.PublishEmbodiedPose(actor, published, 0.0, 1),
            EmbodiedPublishResult::kOk);
  EXPECT_EQ(Read(1).relations[0].phase, RelationPhase::kContact);
  // Not a source mutation: the contact state of the genuine frames is untouched
  // (it would have been reset by a generated source write), so the next genuine
  // frame continues the contact by its own hysteresis.
  Observe({0, 0, .60}, 2);
  EXPECT_EQ(Read(2).relations[0].phase, RelationPhase::kContact);
  Observe({0, 0, .55}, 3);
  EXPECT_EQ(Read(3).relations[0].phase, RelationPhase::kContact);
}

TEST_F(SourceEvidenceTest,
       OversizedExportFailsWholeSnapshotNotSourceAdmission) {
  Enable();
  Observe({0, 0, 1});
  ASSERT_TRUE(
      space.SetElementTag(2, std::string(Evidence::kMaxNameBytes, 'x')));
  Observe({0, 0, 1}, 1);
  EXPECT_EQ(ReadFailure(1), ReadResult::kUnavailable);
  ASSERT_TRUE(space.SetElementTag(2, "region"));
  Observe({0, 0, 1}, 2);
  EXPECT_EQ(Read(2).sequence, 2);
  std::vector<SubjectAnchor> extra;
  for (size_t i = 0; i < Evidence::kMaxAnchors; ++i)
    extra.emplace_back("extra_" + std::to_string(i));
  ASSERT_TRUE(space.AddActorAnchors(actor, extra, true));
  Observe({0, 0, 1}, 3);
  EXPECT_EQ(ReadFailure(3), ReadResult::kUnavailable);
}

TEST_F(SourceEvidenceTest,
       InvalidReadArgumentsClearOutputAndNeverAdmitReceiver) {
  Enable();
  Observe({0, 0, 1});
  Evidence value;
  for (const auto age :
       {int64_t{0}, int64_t{-1}, Evidence::kMaximumAgeNs + 1}) {
    value = Read();
    EXPECT_EQ(
        space.TryCopySourceInteractionEvidence(actor, epoch, 0, age, &value),
        ReadResult::kUnavailable);
    EXPECT_EQ(value.epoch, 0u);
    EXPECT_TRUE(value.relations.empty());
  }
  EXPECT_FALSE(space.EnableSourceInteractionEvidence(actor, epoch + 1, true));
  EXPECT_EQ(space.TryCopySourceInteractionEvidence(
                actor, epoch, 0, Evidence::kMaximumAgeNs, nullptr),
            ReadResult::kUnavailable);
  uint64_t receiver_epoch;
  // Source and receiver ownership cannot be mixed on the same actor.
  EXPECT_FALSE(space.BeginReceiverInputEpoch(actor, "navigation", 1, 1,
                                             &receiver_epoch));
  EXPECT_EQ(Read().sequence, 0);
  ASSERT_TRUE(space.AddElement(6, 2));
  const auto receiver = space.CreateActor({SubjectAnchor("navigation")});
  ASSERT_TRUE(space.SetActorElement(receiver, 6));
  ASSERT_TRUE(space.BeginReceiverInputEpoch(receiver, "navigation", 1, 1,
                                            &receiver_epoch));
  EXPECT_FALSE(
      space.EnableSourceInteractionEvidence(receiver, receiver_epoch, true));
  EXPECT_EQ(space.TryCopySourceInteractionEvidence(
                receiver, receiver_epoch, 0, Evidence::kMaximumAgeNs, &value),
            ReadResult::kUnavailable);
  EXPECT_TRUE(value.anchors.empty());
}

}  // namespace
}  // namespace spatialblend
