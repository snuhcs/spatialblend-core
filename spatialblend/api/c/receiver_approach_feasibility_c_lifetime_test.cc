// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/api/c/spatialblend_receiver_input.h"
#include "spatialblend/api/c/receiver_approach_feasibility_test_api.h"

namespace {

constexpr int64_t kInitialNowNs = 1000;

const uint8_t* Bytes(const char* value) {
  return reinterpret_cast<const uint8_t*>(value);
}

void Identity(double* matrix, int width) {
  std::memset(matrix, 0, sizeof(double) * width * width);
  for (int i = 0; i < width; ++i) matrix[i * width + i] = 1;
}

sb_receiver_anchor_input_t Anchor(const char* name, double x) {
  sb_receiver_anchor_input_t value{};
  value.name_utf8 = Bytes(name);
  value.name_bytes = std::strlen(name);
  value.available = 1;
  Identity(value.world_pose_row_major, 4);
  value.world_pose_row_major[3] = x;
  return value;
}

class CApproachFeasibilityLifetime : public ::testing::Test {
 protected:
  void SetUp() override {
    session = sb_test_create_session_with_receiver_input_clock(kInitialNowNs);
    ASSERT_NE(0, session);
    space = sb_add_space(session);
    ASSERT_NE(0, space);
    room = sb_add_element(space, 0);
    ASSERT_NE(0, room);
    const double room_points[] = {-5, -3, 5, -3, 5, 3, -5, 3};
    sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, room_points, 4);
    actor = sb_create_actor(space);
    ASSERT_NE(0, actor);
    sb_set_actor_element(space, actor, room);
    sb_add_anchor(space, actor, "root", 0, 0, 0, nullptr);
    sb_add_anchor(space, actor, "pelvis", 0, 0, 0, nullptr);

    sb_receiver_input_feature_t feature{};
    ASSERT_EQ(sizeof(feature), sb_get_receiver_input_feature(&feature, sizeof(feature)));
    ASSERT_EQ(SB_RICH_RESULT_OK,
              sb_begin_receiver_input(space, actor, 1, Bytes("root"), 4, 3, 5,
                                      &input_epoch));
    anchors = {Anchor("root", -3), Anchor("pelvis", -3)};
    frame.struct_size = sizeof(frame);
    frame.feature_version = 1;
    frame.anchor_count = anchors.size();
    frame.anchor_stride = sizeof(anchors.front());
    frame.epoch = input_epoch;
    frame.sequence = 1;
    frame.monotonic_clock_id = feature.monotonic_clock_id;
    frame.monotonic_clock_epoch = feature.monotonic_clock_epoch;
    frame.observed_at_ns = kInitialNowNs;
    frame.valid_until_ns = kInitialNowNs + feature.max_age_ns;
    frame.anchors = anchors.data();
    ASSERT_EQ(SB_RICH_RESULT_OK,
              sb_submit_receiver_input(space, actor, &frame, sizeof(frame)));

    navigable_points = {-5, -3, 5, -3, 5, 3, -5, 3};
    obstacle_points = {20, 20, 21, 20, 20, 21};
    support_points = {-.5, -.5, .5, -.5, .5, .5, -.5, .5};
    navigable = Polygon(100, navigable_points);
    obstacle = Polygon(101, obstacle_points);
    support = Polygon(102, support_points);
  }

  void TearDown() override {
    if (session) sb_destroy_session(session);
  }

  sb_receiver_approach_polygon_t Polygon(uint64_t id,
                                          const std::vector<double>& points) {
    sb_receiver_approach_polygon_t value{};
    value.struct_size = sizeof(value);
    value.vertex_count = points.size() / 2;
    value.nav_element_id = id;
    value.region_id = 30;
    value.coordinate_convention = 1;
    value.affine_convention = 1;
    Identity(value.local_to_space_row_major, 3);
    value.vertices_xy = points.data();
    return value;
  }

  sb_receiver_approach_context_registration_t Registration(
      int64_t valid_until_ns) {
    sb_receiver_approach_context_registration_t value{};
    value.struct_size = sizeof(value);
    value.feature_version = 1;
    value.expected_receiver_input_epoch = input_epoch;
    value.expected_receiver_input_sequence = frame.sequence;
    value.geometry_revision = 11;
    value.requested_valid_until_ns = valid_until_ns;
    value.navigation_root_utf8 = Bytes("root");
    value.navigation_root_bytes = 4;
    value.coordinate_convention = 1;
    Identity(value.navigation_root_pose_row_major, 4);
    value.navigation_root_pose_row_major[3] = -3;
    value.navigable_regions = &navigable;
    value.navigable_region_count = 1;
    value.obstacle_regions = &obstacle;
    value.obstacle_region_count = 1;
    value.footprint_radius_m = .25;
    value.required_clearance_m = .15;
    value.destination.struct_size = sizeof(value.destination);
    value.destination.space = space;
    value.destination.region_id = 30;
    value.destination.object = room;
    value.destination.affordance_id = 50;
    value.destination.support_revision = 12;
    value.destination.local_support = support;
    value.destination.coordinate_convention = 1;
    Identity(value.destination.safe_approach_pose_row_major, 4);
    value.destination.safe_approach_pose_row_major[3] = 3;
    return value;
  }

  uint64_t Register(int64_t valid_until_ns) {
    const auto registration = Registration(valid_until_ns);
    uint64_t context = 0;
    EXPECT_EQ(SB_RECEIVER_APPROACH_CONTEXT_OK,
              sb_register_receiver_approach_context(
                  session, space, actor, 1, &registration, sizeof(registration),
                  &context));
    return context;
  }

  sb_receiver_approach_feasibility_header_t Issue(uint64_t context) {
    sb_receiver_approach_feasibility_header_t header{};
    EXPECT_EQ(SB_RECEIVER_APPROACH_FEASIBILITY_BUFFER_TOO_SMALL,
              sb_copy_receiver_approach_feasibility(
                  session, space, actor, 1, context, 0, nullptr, 0, &header,
                  sizeof(header)));
    EXPECT_GT(header.calculation_revision, 0u);
    EXPECT_GE(header.route_point_count, 2u);
    return header;
  }

  std::vector<double> Copy(uint64_t context,
                           const sb_receiver_approach_feasibility_header_t& issued,
                           sb_receiver_approach_feasibility_header_t* out) {
    std::vector<double> route(issued.route_point_count * 2, -777.0);
    EXPECT_EQ(SB_RECEIVER_APPROACH_FEASIBILITY_OK,
              sb_copy_receiver_approach_feasibility(
                  session, space, actor, 1, context, issued.calculation_revision,
                  route.data(), issued.route_point_count, out, sizeof(*out)));
    return route;
  }

  sb_session_t session = 0;
  sb_space_t space = 0;
  sb_element_t room = 0;
  sb_actor_t actor = 0;
  uint64_t input_epoch = 0;
  sb_receiver_input_frame_t frame{};
  std::vector<sb_receiver_anchor_input_t> anchors;
  std::vector<double> navigable_points, obstacle_points, support_points;
  sb_receiver_approach_polygon_t navigable{}, obstacle{}, support{};
};

TEST_F(CApproachFeasibilityLifetime,
       ExpiredReceiptReturnsNoRouteAndDoesNotRenewFrozenCalculation) {
  const uint64_t context = Register(kInitialNowNs + 10);
  const auto issued = Issue(context);
  sb_receiver_approach_feasibility_header_t copied{};
  const auto initial_route = Copy(context, issued, &copied);
  EXPECT_EQ(issued.calculation_revision, copied.calculation_revision);
  ASSERT_EQ(1, sb_test_set_receiver_input_clock(session, kInitialNowNs + 10));

  sb_receiver_approach_feasibility_header_t expired{};
  std::array<double, 4> route = {41, 42, 43, 44};
  EXPECT_EQ(SB_RECEIVER_APPROACH_FEASIBILITY_EXPIRED,
            sb_copy_receiver_approach_feasibility(
                session, space, actor, 1, context, issued.calculation_revision,
                route.data(), 2, &expired, sizeof(expired)));
  EXPECT_EQ(0u, expired.calculation_revision);
  EXPECT_EQ(0u, expired.route_point_count);
  EXPECT_EQ((std::array<double, 4>{41, 42, 43, 44}), route);

  // A fresh calculation cannot silently renew an expired context either.
  EXPECT_EQ(SB_RECEIVER_APPROACH_FEASIBILITY_EXPIRED,
            sb_copy_receiver_approach_feasibility(session, space, actor, 1,
                                                   context, 0, nullptr, 0,
                                                   &expired, sizeof(expired)));
  EXPECT_EQ(0u, expired.calculation_revision);
  EXPECT_FALSE(initial_route.empty());
}

TEST_F(CApproachFeasibilityLifetime,
       RelationCallbackBusyLeavesExistingReceiptWholeAndCopyable) {
  const uint64_t context = Register(kInitialNowNs + 1000);
  const auto issued = Issue(context);
  sb_receiver_approach_feasibility_header_t baseline{};
  const auto route = Copy(context, issued, &baseline);

  // Trigger an actual relation callback from a separate source actor so the
  // receiver input/route receipt remains current throughout the reentrant
  // call.  This exercises the g_inside_relation_callback guard before any
  // session or Space mutex is acquired.
  ASSERT_EQ(1, sb_enable_actor_observation_frames(session));
  const sb_element_t owner = sb_add_element(space, 0);
  const sb_element_t seat = sb_add_element(space, owner);
  sb_set_element_tag(space, seat, "Sittable");
  sb_set_element_shared_id(space, seat, 7001);  // MA-1 / SP-D7-03: an unshared affordance is a referent of no relation
  const double surface[] = {-1, -1, .5, 1, -1, .5, 1, 1, .5, -1, 1, .5};
  sb_set_element_surface(space, seat, surface, 4, SB_GEOMETRY_SURFACE);
  const sb_actor_t trigger = sb_create_actor(space);
  ASSERT_NE(0, trigger);
  sb_set_actor_element(space, trigger, owner);
  sb_add_anchor(space, trigger, "navigation", 0, 0, 0, nullptr);
  ASSERT_EQ(1, sb_add_anchor_with_position_dimension(space, trigger, "body", 0,
                                                      0, 0, nullptr, 3));
  ASSERT_EQ(1, sb_bind_affordance(space, trigger, "body", "Sittable"));
  const uint64_t trigger_epoch =
      sb_begin_actor_observation_epoch(space, trigger, "navigation");
  ASSERT_NE(0u, trigger_epoch);

  struct CallbackState {
    sb_session_t session;
    sb_space_t space;
    sb_actor_t actor;
    uint64_t context;
    uint64_t calculation;
    uint32_t route_point_count;
    int calls = 0;
    int32_t result = 0;
    sb_receiver_approach_feasibility_header_t header{};
    std::vector<double> route;
  } state{session, space, actor, context, issued.calculation_revision,
          issued.route_point_count};
  state.route.assign(state.route_point_count * 2, 919.0);
  sb_register_on_relation_event(
      space,
      [](const sb_relation_event_t*, void* opaque) {
        auto* state = static_cast<CallbackState*>(opaque);
        ++state->calls;
        state->result = sb_copy_receiver_approach_feasibility(
            state->session, state->space, state->actor, 1, state->context,
            state->calculation,
            state->route.data(), state->route_point_count, &state->header,
            sizeof(state->header));
      },
      &state);
  const sb_actor_observation_t observation{"body", 0, 0, 1, .25};
  ASSERT_EQ(1, sb_apply_actor_observation_frame(space, trigger, trigger_epoch,
                                                 1, .1, &observation, 1));
  sb_register_on_relation_event(space, nullptr, nullptr);

  ASSERT_GT(state.calls, 0);
  EXPECT_EQ(SB_RECEIVER_APPROACH_FEASIBILITY_BUSY, state.result);
  EXPECT_EQ(0u, state.header.calculation_revision);
  EXPECT_EQ(0u, state.header.route_point_count);
  for (double value : state.route) EXPECT_EQ(919.0, value);

  sb_receiver_approach_feasibility_header_t after{};
  const auto copied = Copy(context, issued, &after);
  EXPECT_EQ(issued.calculation_revision, after.calculation_revision);
  EXPECT_EQ(baseline.context_revision, after.context_revision);
  EXPECT_EQ(route, copied);
}

}  // namespace
