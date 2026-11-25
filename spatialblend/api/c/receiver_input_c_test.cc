// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/api/c/spatialblend_receiver_input.h"
#include "spatialblend/api/c/spatialblend_receiver_reevaluation.h"

namespace {
const uint8_t* Bytes(const char* text) {
  return reinterpret_cast<const uint8_t*>(text);
}

sb_receiver_anchor_input_t Anchor(const char* name, double x, double y,
                                  double z = 0) {
  sb_receiver_anchor_input_t value{};
  value.name_utf8 = Bytes(name);
  value.name_bytes = std::strlen(name);
  value.available = 1;
  for (int i = 0; i < 4; ++i) value.world_pose_row_major[i * 4 + i] = 1;
  value.world_pose_row_major[3] = x;
  value.world_pose_row_major[7] = y;
  value.world_pose_row_major[11] = z;
  return value;
}

TEST(CReceiverInputLayoutTest, DescriptorAndCompleteFieldOffsets) {
  sb_receiver_input_feature_t feature;
  std::memset(&feature, 0x7f, sizeof(feature));
  EXPECT_EQ(sb_get_receiver_input_feature(nullptr, 0), sizeof(feature));
  EXPECT_EQ(sb_get_receiver_input_feature(&feature, 1), sizeof(feature));
  EXPECT_EQ(feature.descriptor_size, 0x7f7f7f7fu);
  ASSERT_EQ(sb_get_receiver_input_feature(&feature, sizeof(feature)),
            sizeof(feature));
  EXPECT_EQ(feature.feature_version, 1u);
  EXPECT_EQ(feature.frame_size, sizeof(sb_receiver_input_frame_t));
  EXPECT_EQ(feature.anchor_size, sizeof(sb_receiver_anchor_input_t));
  EXPECT_EQ(feature.stamp_size, sizeof(sb_receiver_input_stamp_t));
  EXPECT_EQ(feature.validity_size, sizeof(sb_receiver_input_validity_t));
  EXPECT_EQ(feature.max_anchors, 64u);
  EXPECT_EQ(feature.max_name_bytes, 128u);
  EXPECT_EQ(feature.max_age_ns, 1000000000u);
  EXPECT_EQ(feature.coordinate_convention, 1u);
  sb_rich_realization_feature_t rich{};
  sb_get_rich_realization_feature(&rich, sizeof(rich));
  EXPECT_EQ(feature.monotonic_clock_id, rich.monotonic_clock_id);
  EXPECT_EQ(feature.monotonic_clock_epoch, rich.monotonic_clock_epoch);
  const uint32_t counts[] = {0, 13, 4, 11, 9, 6};
  for (uint32_t kind = 1; kind <= 5; ++kind) {
    EXPECT_EQ(sb_get_receiver_input_field_count(kind), counts[kind]);
    for (uint32_t field = 0; field < counts[kind]; ++field)
      EXPECT_NE(sb_get_receiver_input_field_offset(kind, field), UINT32_MAX);
    EXPECT_EQ(sb_get_receiver_input_field_offset(kind, counts[kind]),
              UINT32_MAX);
  }
  EXPECT_EQ(sb_get_receiver_input_field_count(999), 0u);
  EXPECT_EQ(sb_get_receiver_input_field_offset(999, 0), UINT32_MAX);
  EXPECT_EQ(
      sb_get_receiver_input_field_offset(SB_RECEIVER_INPUT_LAYOUT_FRAME, 10),
      offsetof(sb_receiver_input_frame_t, anchors));
  EXPECT_EQ(
      sb_get_receiver_input_field_offset(SB_RECEIVER_INPUT_LAYOUT_VALIDITY, 2),
      offsetof(sb_receiver_input_validity_t, captured));
  EXPECT_EQ(
      sb_get_receiver_input_field_offset(SB_RECEIVER_INPUT_LAYOUT_ANCHOR, 3),
      offsetof(sb_receiver_anchor_input_t, world_pose_row_major));
}

class CReceiverInputTest : public ::testing::Test {
 protected:
  virtual const char* RootName() const { return "root"; }
  void SetUp() override {
    session = sb_create_session();
    source = sb_add_space(session);
    receiver = sb_add_space(session);
    ASSERT_NE(session, 0);
    double polygon[] = {0, 0, 10, 0, 10, 10, 0, 10};
    for (auto space : {source, receiver}) {
      auto room = sb_add_element(space, 0);
      sb_set_element_shared_id(space, room, 100);
      sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, polygon, 4);
      auto referent = sb_add_element(space, room);
      sb_set_element_shared_id(space, referent, 200);
      double box[] = {3, 3, 7, 3, 7, 7, 3, 7};
      sb_set_element_polygon(space, referent, SB_GEOMETRY_CONTAINMENT, box, 4);
      auto actor = sb_create_actor(space);
      sb_add_anchor(space, actor, RootName(), 0, 0, 0, nullptr);
      sb_add_anchor(space, actor, "head", 0, 0, 1.7, nullptr);
      sb_set_actor_element(space, actor, room);
      if (space == source) {
        source_actor = actor;
        sb_bind(space, actor, RootName(), SB_KERNEL_SPACING);
        sb_bind(space, actor, "head", SB_KERNEL_FACING);
      } else {
        receiver_actor = actor;
        receiver_element = room;
      }
    }
    sb_set_min_move(source, 0);
    sb_set_min_yaw(source, 0);
    sb_get_receiver_input_feature(&feature, sizeof(feature));
    ASSERT_EQ(sb_begin_receiver_input(receiver, receiver_actor, 1,
                                      Bytes(RootName()), std::strlen(RootName()), 7, 9, &epoch),
              SB_RICH_RESULT_OK);
    token = sb_activate_rich_realization(receiver, receiver_actor, 1);
    ASSERT_NE(token, 0u);
    sb_register_on_rich_realization(
        receiver,
        [](const sb_rich_realization_snapshot_t* value, void* user) {
          auto* self = static_cast<CReceiverInputTest*>(user);
          ++self->callbacks;
          self->header = value->header;
          self->channels.clear();
          if (value->channel_count)
            self->channels.assign(value->channels,
                                  value->channels + value->channel_count);
        },
        this);
    anchors = {Anchor(RootName(), 2, 3), Anchor("head", 4, 5, 1.7)};
    frame.struct_size = sizeof(frame);
    frame.feature_version = 1;
    frame.anchor_count = anchors.size();
    frame.anchor_stride = sizeof(sb_receiver_anchor_input_t);
    frame.epoch = epoch;
    frame.sequence = 1;
    frame.monotonic_clock_id = feature.monotonic_clock_id;
    frame.monotonic_clock_epoch = feature.monotonic_clock_epoch;
    frame.anchors = anchors.data();
    RefreshTime();
  }
  void TearDown() override {
    if (session) sb_destroy_session(session);
  }
  void RefreshTime() {
    frame.observed_at_ns = sb_read_rich_monotonic_ns();
    frame.valid_until_ns = frame.observed_at_ns + feature.max_age_ns;
  }
  int Submit() {
    return sb_submit_receiver_input(receiver, receiver_actor, &frame,
                                    sizeof(frame));
  }
  void Drive() { sb_update_actor_state(source, source_actor, 3, 5, 0.2); }
  int Query(sb_receiver_input_validity_t* out) {
    return sb_get_receiver_input_validity(token, header.revision, out,
                                          sizeof(*out));
  }
  sb_session_t session = 0;
  sb_space_t source = 0, receiver = 0;
  sb_actor_t source_actor = 0, receiver_actor = 0;
  sb_element_t receiver_element = 0;
  uint64_t epoch = 0;
  sb_rich_realization_token_t token = 0;
  sb_receiver_input_feature_t feature{};
  sb_receiver_input_frame_t frame{};
  std::vector<sb_receiver_anchor_input_t> anchors;
  int callbacks = 0;
  sb_rich_realization_header_t header{};
  std::vector<sb_rich_realization_channel_t> channels;
};

TEST(CReceiverReevaluationLayoutTest, IndependentDescriptorAndExactOffsets) {
  sb_receiver_reevaluation_feature_t feature;
  std::memset(&feature, 0x7f, sizeof(feature));
  EXPECT_EQ(sb_get_receiver_reevaluation_feature(nullptr, 0), sizeof(feature));
  EXPECT_EQ(sb_get_receiver_reevaluation_feature(&feature, 1), sizeof(feature));
  EXPECT_EQ(feature.descriptor_size, 0x7f7f7f7fu);
  ASSERT_EQ(sb_get_receiver_reevaluation_feature(&feature, sizeof(feature)), sizeof(feature));
  EXPECT_EQ(feature.descriptor_size, sizeof(feature));
  EXPECT_EQ(feature.feature_version, 1u);
  EXPECT_EQ(feature.required_receiver_input_version, 1u);
  EXPECT_EQ(feature.required_rich_version, 1u);
  EXPECT_EQ(feature.guarantees, 7u);
  EXPECT_EQ(feature.reserved, 0u);
  EXPECT_EQ(sb_get_receiver_reevaluation_field_count(1), 6u);
  for (uint32_t i = 0; i < 6; ++i)
    EXPECT_EQ(sb_get_receiver_reevaluation_field_offset(1, i), i * sizeof(uint32_t));
  EXPECT_EQ(sb_get_receiver_reevaluation_field_count(2), 0u);
  EXPECT_EQ(sb_get_receiver_reevaluation_field_offset(2, 0), UINT32_MAX);
  EXPECT_EQ(sb_get_receiver_reevaluation_field_offset(1, 6), UINT32_MAX);
  sb_receiver_input_feature_t input{};
  sb_get_receiver_input_feature(&input, sizeof(input));
  EXPECT_EQ(input.reserved, 0u);  // No opportunistic extension of frozen v1.
}

TEST_F(CReceiverInputTest, ReevaluationDispatchesRichOnlyWithoutNewSourceReceipt) {
  ASSERT_EQ(Submit(), SB_RICH_RESULT_OK);
  Drive();
  const auto original = header;
  const auto count = callbacks;
  int legacy_callbacks = 0;
  sb_register_on_targets(receiver,
      [](sb_space_t, sb_actor_t, const sb_realization_target_t*, int64_t, void* data) {
        ++*static_cast<int*>(data);
      }, &legacy_callbacks);
  frame.sequence = 2;
  anchors[1] = Anchor("head", 6, 7, 1.8);
  RefreshTime();
  ASSERT_EQ(Submit(), SB_RICH_RESULT_OK);
  EXPECT_EQ(callbacks, count);
  ASSERT_EQ(sb_reevaluate_receiver_realization(token, 1, epoch, 2), SB_RICH_RESULT_OK);
  EXPECT_EQ(callbacks, count + 1);
  EXPECT_EQ(header.revision, original.revision + 1);
  EXPECT_EQ(header.producer_lifetime, token);
  EXPECT_EQ(header.receiver_ingress_monotonic_ns, original.receiver_ingress_monotonic_ns);
  EXPECT_GE(header.receiver_solve_monotonic_ns, original.receiver_solve_monotonic_ns);
  EXPECT_EQ(header.source_sequence, original.source_sequence);
  EXPECT_EQ(header.source_epoch, original.source_epoch);
  EXPECT_EQ(header.source_observation_sequence, original.source_observation_sequence);
  EXPECT_EQ(header.source_time_seconds, original.source_time_seconds);
  EXPECT_EQ(header.source_clock_seconds, original.source_clock_seconds);
  EXPECT_EQ(header.source_actor_space, original.source_actor_space);
  EXPECT_EQ(header.source_actor_element, original.source_actor_element);
  EXPECT_EQ(header.relay_source_space, original.relay_source_space);
  EXPECT_EQ(legacy_callbacks, 0);
  bool facing = false;
  for (const auto& channel : channels) {
    if (channel.family != SB_RICH_FAMILY_FACING) continue;
    facing = true;
    EXPECT_EQ(channel.position_x, 6);
    EXPECT_EQ(channel.position_y, 7);
    EXPECT_EQ(channel.position_z, 1.8);
  }
  EXPECT_TRUE(facing);
  sb_receiver_input_validity_t validity{};
  ASSERT_EQ(Query(&validity), SB_RICH_RESULT_OK);
  EXPECT_EQ(validity.valid, 1u);
  EXPECT_EQ(validity.captured.sequence, 2u);
  EXPECT_EQ(sb_reevaluate_receiver_realization(token, 1, epoch, 2), SB_RICH_RESULT_NO_NEW_SNAPSHOT);
  EXPECT_EQ(callbacks, count + 1);
  EXPECT_EQ(sb_get_actor_state(receiver, receiver_actor).position_x, 0);
  sb_register_on_targets(receiver, nullptr, nullptr);
}

TEST_F(CReceiverInputTest, ReevaluationRequiresExactLiveTokenInputAndVersion) {
  ASSERT_EQ(Submit(), SB_RICH_RESULT_OK);
  EXPECT_EQ(sb_reevaluate_receiver_realization(token, 1, epoch, 1), SB_RICH_RESULT_UNAVAILABLE);
  Drive();
  const auto count = callbacks;
  EXPECT_EQ(sb_reevaluate_receiver_realization(token, 2, epoch, 1), SB_RICH_RESULT_UNSUPPORTED_VERSION);
  EXPECT_EQ(sb_reevaluate_receiver_realization(token, 1, 0, 1), SB_RICH_RESULT_INVALID_ARGUMENT);
  EXPECT_EQ(sb_reevaluate_receiver_realization(token, 1, epoch, 0), SB_RICH_RESULT_INVALID_ARGUMENT);
  EXPECT_EQ(sb_reevaluate_receiver_realization(0, 1, epoch, 1), SB_RICH_RESULT_UNKNOWN_TOKEN);
  EXPECT_EQ(sb_reevaluate_receiver_realization(token, 1, epoch + 1, 1), SB_RICH_RESULT_STALE_SNAPSHOT);
  EXPECT_EQ(sb_reevaluate_receiver_realization(token, 1, epoch, 2), SB_RICH_RESULT_STALE_SNAPSHOT);
  EXPECT_EQ(callbacks, count);
  ASSERT_EQ(sb_clear_receiver_input(receiver, receiver_actor, epoch), SB_RICH_RESULT_OK);
  EXPECT_EQ(sb_reevaluate_receiver_realization(token, 1, epoch, 1), SB_RICH_RESULT_UNAVAILABLE);
  sb_retire_rich_realization(token);
  EXPECT_EQ(sb_reevaluate_receiver_realization(token, 1, epoch, 1), SB_RICH_RESULT_UNAVAILABLE);
  ASSERT_EQ(sb_release_rich_realization(token), 1);
  EXPECT_EQ(sb_reevaluate_receiver_realization(token, 1, epoch, 1), SB_RICH_RESULT_UNKNOWN_TOKEN);
}

class CNamedReceiverInputTest : public CReceiverInputTest {
 protected:
  const char* RootName() const override { return "locomotion"; }
};

TEST_F(CNamedReceiverInputTest, NamedRootExportsAdmittedSpacingWithoutActuation) {
  ASSERT_EQ(Submit(), SB_RICH_RESULT_OK);
  Drive();
  bool saw_root = false;
  for (const auto& channel : channels) {
    if (channel.family != SB_RICH_FAMILY_SPACING) continue;
    saw_root = true;
    EXPECT_EQ(channel.has_goal, 1u);
    EXPECT_EQ(channel.goal_kind, SB_RICH_GOAL_ROOT_POSITION);
    EXPECT_EQ(channel.has_z, 0u);
    EXPECT_NEAR(channel.yaw, 0, 1e-12);  // Source yaw is not a Spacing output.
    EXPECT_GT(channel.contributor_count, 0u);
  }
  EXPECT_TRUE(saw_root);
  const auto display = sb_get_actor_state(receiver, receiver_actor);
  EXPECT_EQ(display.position_x, 0);
  EXPECT_EQ(display.position_y, 0);
  sb_receiver_input_validity_t validity{};
  ASSERT_EQ(Query(&validity), SB_RICH_RESULT_OK);
  EXPECT_EQ(validity.valid, 1u);
}

TEST_F(CReceiverInputTest, ActualFrozenCaptureAndLegacyValidityStaySeparate) {
  ASSERT_EQ(Submit(), SB_RICH_RESULT_OK);
  EXPECT_EQ(callbacks, 0);
  const auto display = sb_get_actor_state(receiver, receiver_actor);
  EXPECT_EQ(display.position_x, 0);
  EXPECT_EQ(display.position_y, 0);
  Drive();
  ASSERT_GT(callbacks, 0);
  ASSERT_GT(header.revision, 0u);
  sb_receiver_input_validity_t validity{};
  ASSERT_EQ(Query(&validity), SB_RICH_RESULT_OK);
  EXPECT_EQ(validity.producer_lifetime, token);
  EXPECT_EQ(validity.valid, 1u);
  EXPECT_EQ(validity.captured.epoch, epoch);
  EXPECT_EQ(validity.captured.sequence, 1u);
  EXPECT_EQ(validity.captured.rig_revision, 7u);
  EXPECT_EQ(validity.captured.calibration_revision, 9u);
  EXPECT_EQ(validity.captured.observed_at_ns, frame.observed_at_ns);
  bool saw_facing = false;
  for (const auto& channel : channels) {
    if (channel.family != SB_RICH_FAMILY_FACING) continue;
    saw_facing = true;
    EXPECT_EQ(channel.has_goal, 1u);
    EXPECT_EQ(channel.position_x, 4);
    EXPECT_EQ(channel.position_y, 5);
    EXPECT_EQ(channel.position_z, 1.7);
  }
  EXPECT_TRUE(saw_facing);
  sb_rich_realization_validity_t legacy{};
  EXPECT_EQ(sb_get_rich_realization_validity(
                token, header.revision, SB_RICH_VALIDITY_DOMAIN_RECEIVER_INPUT,
                &legacy, sizeof(legacy)),
            SB_RICH_RESULT_UNAVAILABLE);
  EXPECT_EQ(legacy.valid, 0u);
  EXPECT_NE(legacy.unchecked_mask & SB_RICH_VALIDITY_DOMAIN_RECEIVER_INPUT, 0u);
}

TEST_F(CReceiverInputTest, InvalidFramesCannotPartiallyCommitOrRefreshCapture) {
  ASSERT_EQ(Submit(), SB_RICH_RESULT_OK);
  Drive();
  const auto receipt = header;
  const auto count = callbacks;
  frame.sequence = 2;
  auto expect_unchanged = [&] {
    EXPECT_EQ(Submit(), SB_RICH_RESULT_INVALID_ARGUMENT);
    sb_receiver_input_validity_t value{};
    ASSERT_EQ(Query(&value), SB_RICH_RESULT_OK);
    EXPECT_EQ(value.valid, 1u);
    EXPECT_EQ(value.current.sequence, 1u);
    EXPECT_EQ(callbacks, count);
    EXPECT_EQ(header.receiver_ingress_monotonic_ns,
              receipt.receiver_ingress_monotonic_ns);
  };
  frame.monotonic_clock_epoch ^= 1;
  expect_unchanged();
  frame.monotonic_clock_epoch = feature.monotonic_clock_epoch;
  frame.anchor_stride -= 1;
  expect_unchanged();
  frame.anchor_stride += 1;
  frame.anchor_count = feature.max_anchors + 1;
  expect_unchanged();  // Reject before indexing an actually shorter array.
  frame.anchor_count = 1;
  expect_unchanged();
  frame.anchor_count = 2;
  anchors[1].available = 2;
  expect_unchanged();
  anchors[1].available = 1;
  anchors[1].world_pose_row_major[3] = std::numeric_limits<double>::quiet_NaN();
  expect_unchanged();
  anchors[1].world_pose_row_major[3] = 4;
  anchors[1].name_utf8 = Bytes("root");
  expect_unchanged();
  const uint8_t invalid_utf8[] = {0xed, 0xa0, 0x80, 0};
  anchors[1].name_utf8 = invalid_utf8;
  expect_unchanged();
  anchors[1].name_utf8 = Bytes("head");
  frame.valid_until_ns = frame.observed_at_ns;
  expect_unchanged();
  RefreshTime();
  frame.feature_version = 2;
  EXPECT_EQ(Submit(), SB_RICH_RESULT_UNSUPPORTED_VERSION);
  frame.feature_version = 1;
  EXPECT_EQ(sb_submit_receiver_input(receiver, receiver_actor, &frame, 1),
            SB_RICH_RESULT_INVALID_ARGUMENT);
  EXPECT_EQ(sb_submit_receiver_input(receiver, receiver_actor, nullptr,
                                     sizeof(frame)),
            SB_RICH_RESULT_INVALID_ARGUMENT);
}

TEST_F(CReceiverInputTest, NewFrameAndClearInvalidateWithoutNewSourceReceipt) {
  ASSERT_EQ(Submit(), SB_RICH_RESULT_OK);
  Drive();
  const auto receipt = header;
  const auto count = callbacks;
  frame.sequence = 2;
  anchors[1].available = 0;
  RefreshTime();
  ASSERT_EQ(Submit(), SB_RICH_RESULT_OK);
  sb_receiver_input_validity_t value{};
  ASSERT_EQ(Query(&value), SB_RICH_RESULT_OK);
  EXPECT_EQ(value.changed, 1u);
  EXPECT_EQ(value.valid, 0u);
  EXPECT_EQ(value.current.sequence, 2u);
  EXPECT_EQ(callbacks, count);
  EXPECT_EQ(header.revision, receipt.revision);
  EXPECT_EQ(sb_clear_receiver_input(receiver, receiver_actor, epoch + 1),
            SB_RICH_RESULT_INVALID_ARGUMENT);
  ASSERT_EQ(sb_clear_receiver_input(receiver, receiver_actor, epoch),
            SB_RICH_RESULT_OK);
  EXPECT_EQ(Query(&value), SB_RICH_RESULT_UNAVAILABLE);
  EXPECT_EQ(value.captured.known, 1u);
  EXPECT_EQ(value.current.known, 0u);
  EXPECT_EQ(Submit(),
            SB_RICH_RESULT_INVALID_ARGUMENT);  // Sequence fence survives Clear.
  frame.sequence = 3;
  RefreshTime();
  ASSERT_EQ(Submit(), SB_RICH_RESULT_OK);
  Drive();
  ASSERT_EQ(Query(&value), SB_RICH_RESULT_OK);
  EXPECT_EQ(value.valid, 1u);
  for (const auto& channel : channels)
    if (channel.family == SB_RICH_FAMILY_FACING)
      EXPECT_EQ(channel.has_goal, 0u);
}

TEST_F(CReceiverInputTest, LayoutRebindAndRetiredTokenCannotReviveInput) {
  ASSERT_EQ(Submit(), SB_RICH_RESULT_OK);
  Drive();
  const auto revision = header.revision;
  sb_add_anchor(receiver, receiver_actor, "hand", 0, 0, 1, nullptr);
  sb_receiver_input_validity_t value{};
  EXPECT_EQ(Query(&value), SB_RICH_RESULT_UNAVAILABLE);
  ++frame.sequence;
  EXPECT_EQ(Submit(), SB_RICH_RESULT_INVALID_ARGUMENT);
  uint64_t next = 0;
  EXPECT_EQ(sb_begin_receiver_input(receiver, receiver_actor, 2, Bytes("root"),
                                    4, 7, 9, &next),
            SB_RICH_RESULT_UNSUPPORTED_VERSION);
  ASSERT_EQ(sb_begin_receiver_input(receiver, receiver_actor, 1, Bytes("root"),
                                    4, 7, 10, &next),
            SB_RICH_RESULT_OK);
  EXPECT_GT(next, epoch);
  frame.epoch = next;
  frame.sequence = 1;
  anchors.push_back(Anchor("hand", 2, 3, 1));
  frame.anchors = anchors.data();
  frame.anchor_count = anchors.size();
  RefreshTime();
  ASSERT_EQ(Submit(), SB_RICH_RESULT_OK);
  Drive();
  EXPECT_EQ(
      sb_get_receiver_input_validity(token, revision, &value, sizeof(value)),
      SB_RICH_RESULT_STALE_SNAPSHOT);
  ASSERT_EQ(Query(&value), SB_RICH_RESULT_OK);
  EXPECT_EQ(value.captured.calibration_revision, 10u);
  sb_retire_rich_realization(token);
  EXPECT_EQ(Query(&value), SB_RICH_RESULT_UNAVAILABLE);
  sb_release_rich_realization(token);
  EXPECT_EQ(Query(&value), SB_RICH_RESULT_UNKNOWN_TOKEN);
}

TEST_F(CReceiverInputTest, SourceCallbackReentryRejectsBeforeAnySpaceLock) {
  ASSERT_EQ(Submit(), SB_RICH_RESULT_OK);
  struct CallbackState {
    CReceiverInputTest_SourceCallbackReentryRejectsBeforeAnySpaceLock_Test*
        fixture;
    int calls = 0;
  } callback{this};
  sb_register_on_relation_event(
      source,
      [](const sb_relation_event_t*, void* user) {
        auto* state = static_cast<CallbackState*>(user);
        auto* self = state->fixture;
        ++state->calls;
        uint64_t epoch;
        EXPECT_EQ(sb_begin_receiver_input(self->source, self->source_actor, 1,
                                          Bytes("root"), 4, 7, 9, &epoch),
                  SB_RICH_RESULT_UNAVAILABLE);
        EXPECT_EQ(sb_submit_receiver_input(self->source, self->source_actor,
                                           &self->frame, sizeof(self->frame)),
                  SB_RICH_RESULT_UNAVAILABLE);
        EXPECT_EQ(sb_clear_receiver_input(self->source, self->source_actor, 1),
                  SB_RICH_RESULT_UNAVAILABLE);
        EXPECT_EQ(sb_reevaluate_receiver_realization(self->token, 1,
                                                    self->epoch, 1),
                  SB_RICH_RESULT_UNAVAILABLE);
        sb_receiver_input_validity_t value;
        EXPECT_EQ(self->Query(&value), SB_RICH_RESULT_UNAVAILABLE);
      },
      &callback);
  Drive();
  EXPECT_GT(callback.calls, 0);
  sb_register_on_relation_event(source, nullptr, nullptr);
}

}  // namespace
