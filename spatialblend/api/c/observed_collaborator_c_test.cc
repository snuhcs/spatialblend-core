// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

#include "spatialblend/api/c/c_test_embodied.h"
#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/api/c/spatialblend_observed_collaborator.h"
#include "spatialblend/api/c/spatialblend_observed_source_setup.h"

namespace {

TEST(CObservedCollaboratorLayoutTest, NegotiatesCompleteDescriptorAndOffsets) {
  sb_observed_collaborator_feature_t feature;
  std::memset(&feature, 0x7f, sizeof(feature));
  EXPECT_EQ(sb_get_observed_collaborator_feature(nullptr, 0), sizeof(feature));
  EXPECT_EQ(sb_get_observed_collaborator_feature(&feature, sizeof(feature) - 1),
            sizeof(feature));
  EXPECT_EQ(feature.descriptor_size, 0x7f7f7f7fu);
  ASSERT_EQ(sb_get_observed_collaborator_feature(&feature, sizeof(feature)),
            sizeof(feature));
  EXPECT_EQ(feature.descriptor_size, sizeof(feature));
  EXPECT_EQ(feature.feature_version, SB_OBSERVED_COLLABORATOR_FEATURE_VERSION);
  EXPECT_EQ(feature.frame_size, sizeof(sb_observed_collaborator_frame_t));
  EXPECT_EQ(feature.observation_size, sizeof(sb_actor_observation_t));
  EXPECT_EQ(feature.publication_size,
            sizeof(sb_observed_collaborator_publication_t));
  EXPECT_EQ(feature.max_frames, 64u);
  EXPECT_EQ(feature.max_observations_per_frame, 128u);
  EXPECT_EQ(feature.max_publications, 64u);
  EXPECT_EQ(feature.max_anchor_name_bytes, 128u);
  EXPECT_EQ(feature.max_age_ns, 1000000000LL);
  EXPECT_EQ(feature.coordinate_convention, 1u);
  const uint32_t counts[] = {12, 9, 6};
  for (uint32_t layout = SB_OBSERVED_COLLABORATOR_LAYOUT_FEATURE;
       layout <= SB_OBSERVED_COLLABORATOR_LAYOUT_PUBLICATION; ++layout) {
    ASSERT_EQ(sb_get_observed_collaborator_field_count(layout),
              counts[layout - 1]);
    for (uint32_t field = 0; field < counts[layout - 1]; ++field)
      EXPECT_NE(sb_get_observed_collaborator_field_offset(layout, field),
                UINT32_MAX);
    EXPECT_EQ(sb_get_observed_collaborator_field_offset(layout,
                                                         counts[layout - 1]),
              UINT32_MAX);
  }
  EXPECT_EQ(sb_get_observed_collaborator_field_offset(
                SB_OBSERVED_COLLABORATOR_LAYOUT_FRAME, 6),
            offsetof(sb_observed_collaborator_frame_t, observations));
}

TEST(CObservedSourceSetupLayoutTest, NegotiatesCompleteDescriptorAndOffsets) {
  sb_observed_source_setup_feature_t feature;
  std::memset(&feature, 0x7f, sizeof(feature));
  EXPECT_EQ(sb_get_observed_source_setup_feature(nullptr, 0), sizeof(feature));
  EXPECT_EQ(sb_get_observed_source_setup_feature(&feature, sizeof(feature) - 1),
            sizeof(feature));
  EXPECT_EQ(feature.descriptor_size, 0x7f7f7f7fu);
  ASSERT_EQ(sb_get_observed_source_setup_feature(&feature, sizeof(feature)),
            sizeof(feature));
  EXPECT_EQ(feature.descriptor_size, sizeof(feature));
  EXPECT_EQ(feature.feature_version, SB_OBSERVED_SOURCE_SETUP_FEATURE_VERSION);
  EXPECT_EQ(feature.participant_size,
            sizeof(sb_observed_source_setup_participant_t));
  EXPECT_EQ(feature.point_size, sizeof(sb_observed_source_setup_point_t));
  EXPECT_EQ(feature.actor_result_size,
            sizeof(sb_observed_source_setup_actor_result_t));
  EXPECT_EQ(feature.point_result_size,
            sizeof(sb_observed_source_setup_point_result_t));
  EXPECT_EQ(feature.max_participants, 64u);
  EXPECT_EQ(feature.max_points, 64u);
  EXPECT_EQ(feature.max_anchor_name_bytes, 128u);
  EXPECT_EQ(feature.max_age_ns, 1000000000LL);
  EXPECT_EQ(feature.coordinate_convention, 1u);
  const uint32_t counts[] = {13, 6, 7, 2, 2};
  for (uint32_t layout = SB_OBSERVED_SOURCE_SETUP_LAYOUT_FEATURE;
       layout <= SB_OBSERVED_SOURCE_SETUP_LAYOUT_POINT_RESULT; ++layout) {
    ASSERT_EQ(sb_get_observed_source_setup_field_count(layout),
              counts[layout - 1]);
    for (uint32_t field = 0; field < counts[layout - 1]; ++field)
      EXPECT_NE(sb_get_observed_source_setup_field_offset(layout, field),
                UINT32_MAX);
    EXPECT_EQ(sb_get_observed_source_setup_field_offset(layout,
                                                         counts[layout - 1]),
              UINT32_MAX);
  }
  EXPECT_EQ(sb_get_observed_source_setup_field_offset(
                SB_OBSERVED_SOURCE_SETUP_LAYOUT_PARTICIPANT, 4),
            offsetof(sb_observed_source_setup_participant_t, root_anchor_name));
}

class CObservedCollaboratorBatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    session = sb_create_session();
    space = sb_add_space(session);
    ASSERT_NE(session, 0);
    ASSERT_NE(space, 0);
    ASSERT_EQ(sb_enable_actor_observation_frames(session), 1);
    const sb_element_t room = sb_add_element(space, 0);
    for (int index = 0; index < 2; ++index) {
      owners[index] = sb_add_element(space, room);
      points[index] = sb_add_element(space, room);
      actors[index] = sb_create_actor(space);
      ASSERT_NE(owners[index], 0);
      ASSERT_NE(points[index], 0);
      ASSERT_NE(actors[index], 0);
      sb_set_element_shared_id(space, owners[index], 100 + index);
      sb_add_anchor(space, actors[index], "root", 0, 0, 0, nullptr);
      sb_add_anchor(space, actors[index], "head", 0, 0, 1.7, nullptr);
      sb_set_actor_element(space, actors[index], owners[index]);
      epochs[index] = sb_begin_actor_observation_epoch(space, actors[index], "root");
      ASSERT_NE(epochs[index], 0u);
      ASSERT_EQ(sb_begin_observed_source_anchor_point(
                    space, points[index], actors[index], "head", 1000000000,
                    &leases[index]),
                1);
      ASSERT_NE(leases[index], 0u);
    }
  }

  void TearDown() override {
    if (session) sb_destroy_session(session);
  }

  sb_observed_collaborator_frame_t Frame(int index, int64_t sequence) {
    anchor_rows[index] = {{"root", index * 2.0 + sequence * .01, 0, 0, 0},
                          {"head", index * 2.0 + sequence * .01, .1, 1.7, .2}};
    return {sizeof(sb_observed_collaborator_frame_t),
            SB_OBSERVED_COLLABORATOR_FEATURE_VERSION,
            actors[index], epochs[index], sequence, sequence * .01,
            anchor_rows[index].data(),
            static_cast<uint32_t>(anchor_rows[index].size()),
            sizeof(sb_actor_observation_t)};
  }

  sb_observed_collaborator_publication_t Publication(int index) {
    const int64_t now = sb_read_rich_monotonic_ns();
    return {sizeof(sb_observed_collaborator_publication_t),
            SB_OBSERVED_COLLABORATOR_FEATURE_VERSION,
            points[index], leases[index], now, now + 1000000000};
  }

  void ExpectClock(int index, int64_t sequence) {
    uint64_t epoch = 0;
    int64_t received_sequence = -1;
    double time = -1;
    int32_t has_sample = 0;
    ASSERT_EQ(sb_get_actor_observation_clock(space, actors[index], &epoch,
                                              &received_sequence, &time,
                                              &has_sample),
              1);
    EXPECT_EQ(epoch, epochs[index]);
    EXPECT_EQ(received_sequence, sequence);
    EXPECT_EQ(has_sample, 1);
  }

  sb_session_t session = 0;
  sb_space_t space = 0;
  sb_actor_t actors[2]{};
  sb_element_t owners[2]{}, points[2]{};
  uint64_t epochs[2]{}, leases[2]{};
  std::vector<sb_actor_observation_t> anchor_rows[2];
};

TEST_F(CObservedCollaboratorBatchTest,
       BoundedBatchCommitsBothPeersOrNeitherAcross120Frames) {
  auto first_frames = std::vector<sb_observed_collaborator_frame_t>{Frame(0, 0),
                                                                      Frame(1, 0)};
  auto first_points = std::vector<sb_observed_collaborator_publication_t>{
      Publication(0), Publication(1)};
  ASSERT_EQ(sb_apply_observed_collaborator_batch(
                space, first_frames.data(), first_frames.size(),
                first_points.data(), first_points.size()),
            1);
  ExpectClock(0, 0);
  ExpectClock(1, 0);

  auto rejected = std::vector<sb_observed_collaborator_frame_t>{Frame(0, 1),
                                                                  Frame(0, 1)};
  ASSERT_EQ(sb_apply_observed_collaborator_batch(
                space, rejected.data(), rejected.size(), first_points.data(),
                first_points.size()),
            0);
  // A duplicate peer cannot advance one actor while leaving the other stale.
  ExpectClock(0, 0);
  ExpectClock(1, 0);

  for (int64_t sequence = 1; sequence <= 120; ++sequence) {
    auto frames = std::vector<sb_observed_collaborator_frame_t>{Frame(0, sequence),
                                                                 Frame(1, sequence)};
    auto points = std::vector<sb_observed_collaborator_publication_t>{
        Publication(0), Publication(1)};
    ASSERT_EQ(sb_apply_observed_collaborator_batch(
                  space, frames.data(), frames.size(), points.data(),
                  points.size()),
              1);
    ExpectClock(0, sequence);
    ExpectClock(1, sequence);
  }

  ASSERT_EQ(sb_retire_observed_source_anchor_point(space, points[0], leases[0]),
            1);
  auto retired_frames = std::vector<sb_observed_collaborator_frame_t>{Frame(0, 121),
                                                                         Frame(1, 121)};
  auto retired_points = std::vector<sb_observed_collaborator_publication_t>{
      Publication(0), Publication(1)};
  EXPECT_EQ(sb_apply_observed_collaborator_batch(
                space, retired_frames.data(), retired_frames.size(),
                retired_points.data(), retired_points.size()),
            0);
  ExpectClock(0, 120);
  ExpectClock(1, 120);
}

TEST_F(CObservedCollaboratorBatchTest,
       RejectsPublicBoundaryMalformedAndExpiredInputsWithoutPrefixCommit) {
  auto initial_frames = std::vector<sb_observed_collaborator_frame_t>{
      Frame(0, 0), Frame(1, 0)};
  auto initial_publications =
      std::vector<sb_observed_collaborator_publication_t>{Publication(0),
                                                           Publication(1)};
  ASSERT_EQ(sb_apply_observed_collaborator_batch(
                space, initial_frames.data(), initial_frames.size(),
                initial_publications.data(), initial_publications.size()),
            1);
  ExpectClock(0, 0);
  ExpectClock(1, 0);

  sb_observed_collaborator_feature_t feature{};
  ASSERT_EQ(sb_get_observed_collaborator_feature(&feature, sizeof(feature)),
            sizeof(feature));
  // Count limits and required C pointers are rejected before rows can be
  // decoded or any participant clock can advance.
  EXPECT_EQ(sb_apply_observed_collaborator_batch(
                space, nullptr, feature.max_frames + 1, nullptr, 0),
            0);
  auto frames = std::vector<sb_observed_collaborator_frame_t>{Frame(0, 1),
                                                                Frame(1, 1)};
  EXPECT_EQ(sb_apply_observed_collaborator_batch(
                space, frames.data(), frames.size(), nullptr,
                feature.max_publications + 1),
            0);
  auto publications =
      std::vector<sb_observed_collaborator_publication_t>{Publication(0),
                                                           Publication(1)};
  EXPECT_EQ(sb_apply_observed_collaborator_batch(
                space, nullptr, frames.size(), publications.data(),
                publications.size()),
            0);
  frames[0].observations = nullptr;
  frames[0].observation_count = 1;
  EXPECT_EQ(sb_apply_observed_collaborator_batch(
                space, frames.data(), frames.size(), publications.data(),
                publications.size()),
            0);

  frames = {Frame(0, 1), Frame(1, 1)};
  anchor_rows[0][0].position_x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(sb_apply_observed_collaborator_batch(
                space, frames.data(), frames.size(), publications.data(),
                publications.size()),
            0);
  frames = {Frame(0, 1), Frame(1, 1)};
  frames[1].time_seconds = std::numeric_limits<double>::infinity();
  EXPECT_EQ(sb_apply_observed_collaborator_batch(
                space, frames.data(), frames.size(), publications.data(),
                publications.size()),
            0);

  frames = {Frame(0, 1), Frame(1, 1)};
  publications = {Publication(0), Publication(1)};
  // This is structurally valid but already expired in Core's monotonic clock
  // domain, so it exercises public expiry rather than malformed timestamp
  // rejection. The other participant must not become a prefix commit.
  const int64_t now = sb_read_rich_monotonic_ns();
  publications[0].observed_at_ns = now - 1000000000;
  publications[0].valid_until_ns = now - 500000000;
  EXPECT_EQ(sb_apply_observed_collaborator_batch(
                space, frames.data(), frames.size(), publications.data(),
                publications.size()),
            0);
  ExpectClock(0, 0);
  ExpectClock(1, 0);
}

TEST_F(CObservedCollaboratorBatchTest,
       RelationCallbackReentryRejectsC1MutationWithoutPrefixCommit) {
  auto initial_frames = std::vector<sb_observed_collaborator_frame_t>{
      Frame(0, 0), Frame(1, 0)};
  auto initial_publications =
      std::vector<sb_observed_collaborator_publication_t>{Publication(0),
                                                           Publication(1)};
  ASSERT_EQ(sb_apply_observed_collaborator_batch(
                space, initial_frames.data(), initial_frames.size(),
                initial_publications.data(), initial_publications.size()),
            1);
  ExpectClock(0, 0);
  ExpectClock(1, 0);

  auto reentrant_frames = std::vector<sb_observed_collaborator_frame_t>{
      Frame(0, 1), Frame(1, 1)};
  auto reentrant_publications =
      std::vector<sb_observed_collaborator_publication_t>{Publication(0),
                                                           Publication(1)};
  struct CallbackState {
    sb_space_t space;
    const sb_observed_collaborator_frame_t* frames;
    uint32_t frame_count;
    const sb_observed_collaborator_publication_t* publications;
    uint32_t publication_count;
    int calls = 0;
    int32_t result = 1;
  } state{space, reentrant_frames.data(),
          static_cast<uint32_t>(reentrant_frames.size()),
          reentrant_publications.data(),
          static_cast<uint32_t>(reentrant_publications.size())};
  sb_register_on_relation_event(
      space,
      [](const sb_relation_event_t*, void* opaque) {
        auto* state = static_cast<CallbackState*>(opaque);
        ++state->calls;
        state->result = sb_apply_observed_collaborator_batch(
            state->space, state->frames, state->frame_count,
            state->publications, state->publication_count);
      },
      &state);

  // Cause a relation update from a distinct actor. The callback runs while
  // the solve lock is held, so the batch must be rejected before locking or decoding rows.
  const sb_element_t trigger_owner = sb_add_element(space, 0);
  const sb_element_t trigger_seat = sb_add_element(space, trigger_owner);
  ASSERT_NE(trigger_owner, 0);
  ASSERT_NE(trigger_seat, 0);
  sb_set_element_tag(space, trigger_seat, "Sittable");
  sb_set_element_shared_id(space, trigger_seat, 7001);  // MA-1 / SP-D7-03: an unshared affordance is a referent of no relation
  const double surface[] = {-1, -1, .5, 1, -1, .5, 1, 1, .5, -1, 1, .5};
  sb_set_element_surface(space, trigger_seat, surface, 4,
                         SB_GEOMETRY_SURFACE);
  const sb_actor_t trigger = sb_create_actor(space);
  spatialblend_c_test::PublishStartPose(space, trigger);
  ASSERT_NE(trigger, 0);
  sb_set_actor_element(space, trigger, trigger_owner);
  sb_add_anchor(space, trigger, "navigation", 0, 0, 0, nullptr);
  ASSERT_EQ(sb_add_anchor_with_position_dimension(space, trigger, "body", 0,
                                                  0, 0, nullptr, 3),
            1);
  ASSERT_EQ(sb_bind_affordance(space, trigger, "body", "Sittable"),
            1);
  const uint64_t trigger_epoch =
      sb_begin_actor_observation_epoch(space, trigger, "navigation");
  ASSERT_NE(trigger_epoch, 0u);
  const sb_actor_observation_t observation{"body", 0, 0, 1, .25};
  ASSERT_EQ(sb_apply_actor_observation_frame(space, trigger, trigger_epoch,
                                             1, .1, &observation, 1),
            1);
  sb_register_on_relation_event(space, nullptr, nullptr);

  ASSERT_GT(state.calls, 0);
  EXPECT_EQ(state.result, 0);
  ExpectClock(0, 0);
  ExpectClock(1, 0);
}

TEST(CObservedSourceSetupCTest,
     RejectsMalformedDuplicateCrossSpaceAndLatePointWithoutPrefixClock) {
  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  const sb_space_t other_space = sb_add_space(session);
  ASSERT_NE(session, 0);
  ASSERT_NE(space, 0);
  ASSERT_NE(other_space, 0);
  ASSERT_EQ(sb_enable_actor_observation_frames(session), 1);
  const sb_element_t room = sb_add_element(space, 0);
  const sb_element_t owner = sb_add_element(space, room);
  const sb_element_t point = sb_add_element(space, room);
  const sb_actor_t actor = sb_create_actor(space);
  spatialblend_c_test::PublishStartPose(space, actor);
  ASSERT_NE(room, 0);
  ASSERT_NE(owner, 0);
  ASSERT_NE(point, 0);
  ASSERT_NE(actor, 0);
  sb_set_element_shared_id(space, owner, 100);
  sb_add_anchor(space, actor, "root", 0, 0, 0, nullptr);
  sb_add_anchor(space, actor, "head", 0, 0, 1.7, nullptr);
  sb_set_actor_element(space, actor, owner);

  sb_observed_source_setup_feature_t feature{};
  EXPECT_EQ(sb_get_observed_source_setup_feature(nullptr, 0), sizeof(feature));
  ASSERT_EQ(sb_get_observed_source_setup_feature(&feature, sizeof(feature)),
            sizeof(feature));
  EXPECT_EQ(feature.feature_version, SB_OBSERVED_SOURCE_SETUP_FEATURE_VERSION);
  EXPECT_EQ(feature.participant_size,
            sizeof(sb_observed_source_setup_participant_t));

  sb_observed_source_setup_participant_t participant{
      sizeof(participant), SB_OBSERVED_SOURCE_SETUP_FEATURE_VERSION, space,
      actor, "root", 0};
  sb_observed_source_setup_point_t good_point{
      sizeof(good_point), SB_OBSERVED_SOURCE_SETUP_FEATURE_VERSION, space,
      point, actor, "head", 1000};
  sb_observed_source_setup_actor_result_t actor_out{777, 777};
  sb_observed_source_setup_point_result_t point_out{777, 777};
  sb_observed_source_setup_t setup = 777;
  auto expect_no_epoch = [&] {
    uint64_t epoch = 77;
    int64_t sequence = 77;
    double time = 77;
    int32_t sample = 77;
    EXPECT_EQ(sb_get_actor_observation_clock(space, actor, &epoch, &sequence,
                                             &time, &sample),
              0);
  };
  auto expect_no_setup_prefix = [&] {
    EXPECT_EQ(setup, 777);
    EXPECT_EQ(actor_out.actor, 777);
    EXPECT_EQ(actor_out.epoch, 777u);
    EXPECT_EQ(point_out.element, 777);
    EXPECT_EQ(point_out.lease, 777u);
  };

  sb_observed_source_setup_feature_t setup_feature{};
  ASSERT_EQ(sb_get_observed_source_setup_feature(&setup_feature,
                                                  sizeof(setup_feature)),
            sizeof(setup_feature));
  // Public C setup ingress must reject missing required caller storage and
  // advertised hard limits before it can mint even one actor epoch or point
  // lease, and it must not expose a partial output row on those failures.
  EXPECT_EQ(sb_begin_observed_source_setup(
                space, nullptr, 1, &good_point, 1, &setup, &actor_out, 1,
                &point_out, 1),
            0);
  expect_no_epoch();
  expect_no_setup_prefix();
  EXPECT_EQ(sb_begin_observed_source_setup(
                space, &participant, 1, nullptr, 1, &setup, &actor_out, 1,
                &point_out, 1),
            0);
  EXPECT_EQ(sb_begin_observed_source_setup(
                space, &participant, 1, &good_point, 1, nullptr, &actor_out,
                1, &point_out, 1),
            0);
  EXPECT_EQ(sb_begin_observed_source_setup(
                space, &participant, setup_feature.max_participants + 1,
                &good_point, 1, &setup, &actor_out,
                setup_feature.max_participants + 1, &point_out, 1),
            0);
  EXPECT_EQ(sb_begin_observed_source_setup(
                space, &participant, 1, &good_point,
                setup_feature.max_points + 1, &setup, &actor_out, 1,
                &point_out, setup_feature.max_points + 1),
            0);
  EXPECT_EQ(sb_begin_observed_source_setup(
                space, &participant, 1, &good_point, 1, &setup, &actor_out,
                0, &point_out, 1),
            0);
  expect_no_epoch();
  expect_no_setup_prefix();

  // A duplicate participant is rejected before either row opens an epoch.
  sb_observed_source_setup_participant_t duplicates[2] = {participant,
                                                           participant};
  EXPECT_EQ(sb_begin_observed_source_setup(
                space, duplicates, 2, &good_point, 1, &setup, &actor_out, 2,
                &point_out, 1),
            0);
  expect_no_epoch();
  EXPECT_EQ(setup, 777);

  auto cross_space = good_point;
  cross_space.space = other_space;
  EXPECT_EQ(sb_begin_observed_source_setup(
                space, &participant, 1, &cross_space, 1, &setup, &actor_out,
                1, &point_out, 1),
            0);
  expect_no_epoch();

  auto late_invalid_point = good_point;
  late_invalid_point.element = owner;  // Producer-owned, hence not a point.
  sb_observed_source_setup_point_t mixed[2] = {good_point, late_invalid_point};
  EXPECT_EQ(sb_begin_observed_source_setup(
                space, &participant, 1, mixed, 2, &setup, &actor_out, 1,
                &point_out, 2),
            0);
  expect_no_epoch();

  auto bad_anchor = good_point;
  bad_anchor.anchor_name = "absent";
  EXPECT_EQ(sb_begin_observed_source_setup(
                space, &participant, 1, &bad_anchor, 1, &setup, &actor_out,
                1, &point_out, 1),
            0);
  expect_no_epoch();

  auto bad_age = good_point;
  bad_age.maximum_age_ns = 0;
  EXPECT_EQ(sb_begin_observed_source_setup(
                space, &participant, 1, &bad_age, 1, &setup, &actor_out, 1,
                &point_out, 1),
            0);
  expect_no_epoch();

  ASSERT_EQ(sb_begin_observed_source_setup(
                space, &participant, 1, &good_point, 1, &setup, &actor_out,
                1, &point_out, 1),
            1);
  ASSERT_NE(setup, 0);
  EXPECT_EQ(actor_out.actor, actor);
  EXPECT_EQ(point_out.element, point);
  uint64_t epoch = 0;
  int64_t sequence = 0;
  double time = -1;
  int32_t sample = 1;
  ASSERT_EQ(sb_get_actor_observation_clock(space, actor, &epoch, &sequence,
                                           &time, &sample),
            1);
  EXPECT_EQ(epoch, actor_out.epoch);
  EXPECT_EQ(sequence, -1);
  EXPECT_EQ(sample, 0);
  // An untouched setup owns the only exact cancellation capability, so merely
  // forgetting the C token is refused and retirement remains available.
  EXPECT_EQ(sb_release_observed_source_setup(setup), 0);
  EXPECT_EQ(sb_retire_observed_source_setup(space, setup), 1);
  EXPECT_EQ(sb_retire_observed_source_setup(space, setup), 1);
  EXPECT_EQ(sb_release_observed_source_setup(setup), 1);
  EXPECT_EQ(sb_retire_observed_source_setup(space, setup), 0);
  sb_destroy_session(session);
}

TEST(CObservedSourceSetupCTest,
     ReplacesOnlyTheDeclaredLegacyEpochWithoutAStalePrefixMutation) {
  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  ASSERT_NE(session, 0);
  ASSERT_NE(space, 0);
  ASSERT_EQ(sb_enable_actor_observation_frames(session), 1);
  const sb_element_t room = sb_add_element(space, 0);
  const sb_element_t owner = sb_add_element(space, room);
  const sb_element_t point = sb_add_element(space, room);
  const sb_actor_t actor = sb_create_actor(space);
  spatialblend_c_test::PublishStartPose(space, actor);
  sb_set_element_shared_id(space, owner, 100);
  sb_add_anchor(space, actor, "root", 0, 0, 0, nullptr);
  sb_add_anchor(space, actor, "head", 0, 0, 1.7, nullptr);
  sb_set_actor_element(space, actor, owner);
  const uint64_t legacy_epoch =
      sb_begin_actor_observation_epoch(space, actor, "root");
  ASSERT_NE(legacy_epoch, 0u);

  sb_observed_source_setup_participant_t participant{
      sizeof(participant), SB_OBSERVED_SOURCE_SETUP_FEATURE_VERSION, space,
      actor, "root", legacy_epoch + 1};
  sb_observed_source_setup_point_t source_point{
      sizeof(source_point), SB_OBSERVED_SOURCE_SETUP_FEATURE_VERSION, space,
      point, actor, "head", 1000};
  sb_observed_source_setup_t setup = 0;
  sb_observed_source_setup_actor_result_t actor_out{};
  sb_observed_source_setup_point_result_t point_out{};
  EXPECT_EQ(sb_begin_observed_source_setup(
                space, &participant, 1, &source_point, 1, &setup, &actor_out,
                1, &point_out, 1),
            0);
  uint64_t epoch = 0;
  int64_t sequence = 0;
  double time = 0;
  int32_t sample = 1;
  ASSERT_EQ(sb_get_actor_observation_clock(space, actor, &epoch, &sequence,
                                           &time, &sample),
            1);
  EXPECT_EQ(epoch, legacy_epoch);
  EXPECT_EQ(sequence, -1);
  EXPECT_EQ(sample, 0);

  participant.expected_prior_epoch = legacy_epoch;
  ASSERT_EQ(sb_begin_observed_source_setup(
                space, &participant, 1, &source_point, 1, &setup, &actor_out,
                1, &point_out, 1),
            1);
  EXPECT_GT(actor_out.epoch, legacy_epoch);
  // A direct legacy begin terminally supersedes the setup capability. It must
  // not leave an unretireable registry entry behind.
  EXPECT_GT(sb_begin_actor_observation_epoch(space, actor, "root"),
            actor_out.epoch);
  EXPECT_EQ(sb_release_observed_source_setup(setup), 1);
  sb_destroy_session(session);
}

TEST(CObservedSourceSetupCTest,
     AdmittedSetupReleaseDropsOnlyItsCancellationRegistryCapability) {
  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  ASSERT_NE(session, 0);
  ASSERT_NE(space, 0);
  ASSERT_EQ(sb_enable_actor_observation_frames(session), 1);
  const sb_element_t room = sb_add_element(space, 0);
  const sb_element_t owner = sb_add_element(space, room);
  const sb_element_t point = sb_add_element(space, room);
  const sb_actor_t actor = sb_create_actor(space);
  spatialblend_c_test::PublishStartPose(space, actor);
  ASSERT_NE(room, 0);
  ASSERT_NE(owner, 0);
  ASSERT_NE(point, 0);
  ASSERT_NE(actor, 0);
  sb_set_element_shared_id(space, owner, 100);
  sb_add_anchor(space, actor, "root", 0, 0, 0, nullptr);
  sb_add_anchor(space, actor, "head", 0, 0, 1.7, nullptr);
  sb_set_actor_element(space, actor, owner);

  sb_observed_source_setup_participant_t participant{
      sizeof(participant), SB_OBSERVED_SOURCE_SETUP_FEATURE_VERSION, space,
      actor, "root", 0};
  sb_observed_source_setup_point_t source_point{
      sizeof(source_point), SB_OBSERVED_SOURCE_SETUP_FEATURE_VERSION, space,
      point, actor, "head", 1000000000};
  sb_observed_source_setup_t setup = 0;
  sb_observed_source_setup_actor_result_t actor_out{};
  sb_observed_source_setup_point_result_t point_out{};
  ASSERT_EQ(sb_begin_observed_source_setup(
                space, &participant, 1, &source_point, 1, &setup, &actor_out,
                1, &point_out, 1),
            1);

  sb_actor_observation_t observations[] = {
      {"root", 0, 0, 0, 0}, {"head", 0, 0, 1.7, 0}};
  sb_observed_collaborator_frame_t frame{
      sizeof(frame), SB_OBSERVED_COLLABORATOR_FEATURE_VERSION, actor,
      actor_out.epoch, 0, 0, observations, 2, sizeof(observations[0])};
  const int64_t now = sb_read_rich_monotonic_ns();
  sb_observed_collaborator_publication_t publication{
      sizeof(publication), SB_OBSERVED_COLLABORATOR_FEATURE_VERSION, point,
      point_out.lease, now, now + 1000000000};
  ASSERT_EQ(sb_apply_observed_collaborator_batch(space, &frame, 1,
                                                 &publication, 1),
            1);

  // Release is now safe but does not cancel or otherwise alter committed
  // state. A newly supplied frame remains admitted under the same epoch.
  EXPECT_EQ(sb_release_observed_source_setup(setup), 1);
  EXPECT_EQ(sb_retire_observed_source_setup(space, setup), 0);
  frame.sequence = 1;
  frame.time_seconds = .01;
  publication.observed_at_ns = sb_read_rich_monotonic_ns();
  publication.valid_until_ns = publication.observed_at_ns + 1000000000;
  ASSERT_EQ(sb_apply_observed_collaborator_batch(space, &frame, 1,
                                                 &publication, 1),
            1);
  uint64_t epoch = 0;
  int64_t sequence = -1;
  double time = -1;
  int32_t sample = 0;
  ASSERT_EQ(sb_get_actor_observation_clock(space, actor, &epoch, &sequence,
                                           &time, &sample),
            1);
  EXPECT_EQ(epoch, actor_out.epoch);
  EXPECT_EQ(sequence, 1);
  EXPECT_EQ(sample, 1);

  sb_destroy_session(session);
  // Destroyed sessions forget capabilities; a stale token is inert.
  EXPECT_EQ(sb_release_observed_source_setup(setup), 0);
}

TEST(CObservedSourceSetupCTest, GlobalCleanupForgetsAnActiveSetupCapability) {
  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  ASSERT_NE(session, 0);
  ASSERT_NE(space, 0);
  ASSERT_EQ(sb_enable_actor_observation_frames(session), 1);
  const sb_element_t room = sb_add_element(space, 0);
  const sb_element_t owner = sb_add_element(space, room);
  const sb_element_t point = sb_add_element(space, room);
  const sb_actor_t actor = sb_create_actor(space);
  spatialblend_c_test::PublishStartPose(space, actor);
  ASSERT_NE(room, 0);
  ASSERT_NE(owner, 0);
  ASSERT_NE(point, 0);
  ASSERT_NE(actor, 0);
  sb_set_element_shared_id(space, owner, 100);
  sb_add_anchor(space, actor, "root", 0, 0, 0, nullptr);
  sb_add_anchor(space, actor, "head", 0, 0, 1.7, nullptr);
  sb_set_actor_element(space, actor, owner);
  sb_observed_source_setup_participant_t participant{
      sizeof(participant), SB_OBSERVED_SOURCE_SETUP_FEATURE_VERSION, space,
      actor, "root", 0};
  sb_observed_source_setup_point_t source_point{
      sizeof(source_point), SB_OBSERVED_SOURCE_SETUP_FEATURE_VERSION, space,
      point, actor, "head", 1000000000};
  sb_observed_source_setup_t setup = 0;
  sb_observed_source_setup_actor_result_t actor_out{};
  sb_observed_source_setup_point_result_t point_out{};
  ASSERT_EQ(sb_begin_observed_source_setup(
                space, &participant, 1, &source_point, 1, &setup, &actor_out,
                1, &point_out, 1),
            1);
  ASSERT_NE(setup, 0);
  sb_cleanup_all();
  EXPECT_EQ(sb_retire_observed_source_setup(space, setup), 0);
  EXPECT_EQ(sb_release_observed_source_setup(setup), 0);
}

}  // namespace
