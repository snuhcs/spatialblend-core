// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <vector>

#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/api/c/spatialblend_receiver_approach_context.h"
#include "spatialblend/api/c/spatialblend_receiver_input.h"

namespace {

const uint8_t* Bytes(const char* text) {
  return reinterpret_cast<const uint8_t*>(text);
}

void Identity(double* matrix, int dimension) {
  for (int row = 0; row < dimension; ++row)
    for (int column = 0; column < dimension; ++column)
      matrix[row * dimension + column] = row == column ? 1.0 : 0.0;
}

sb_receiver_anchor_input_t Anchor(const char* name, double x, double y) {
  sb_receiver_anchor_input_t value{};
  value.name_utf8 = Bytes(name);
  value.name_bytes = std::strlen(name);
  value.available = 1;
  Identity(value.world_pose_row_major, 4);
  value.world_pose_row_major[3] = x;
  value.world_pose_row_major[7] = y;
  return value;
}

class CReceiverApproachContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    session = sb_create_session();
    space = sb_add_space(session);
    ASSERT_NE(0, session);
    ASSERT_NE(0, space);
    const double room[] = {-5, -3, 5, -3, 5, 3, -5, 3};
    element = sb_add_element(space, 0);
    ASSERT_NE(0, element);
    sb_set_element_polygon(space, element, SB_GEOMETRY_NAVIGABLE, room, 4);
    actor = sb_create_actor(space);
    sb_add_anchor(space, actor, "navigation_root", 0, 0, 0, nullptr);
    sb_add_anchor(space, actor, "pelvis", 0, 0, 0, nullptr);
    sb_set_actor_element(space, actor, element);
    sb_receiver_input_feature_t feature{};
    ASSERT_EQ(sizeof(feature), sb_get_receiver_input_feature(&feature, sizeof(feature)));
    ASSERT_EQ(SB_RICH_RESULT_OK,
              sb_begin_receiver_input(space, actor, 1, Bytes("navigation_root"),
                                      std::strlen("navigation_root"), 3, 5, &epoch));
    anchors = {Anchor("navigation_root", -3, 0), Anchor("pelvis", -3, 0)};
    frame.struct_size = sizeof(frame);
    frame.feature_version = 1;
    frame.anchor_count = anchors.size();
    frame.anchor_stride = sizeof(sb_receiver_anchor_input_t);
    frame.epoch = epoch;
    frame.sequence = 1;
    frame.monotonic_clock_id = feature.monotonic_clock_id;
    frame.monotonic_clock_epoch = feature.monotonic_clock_epoch;
    frame.anchors = anchors.data();
    RefreshTime(feature.max_age_ns);
    ASSERT_EQ(SB_RICH_RESULT_OK,
              sb_submit_receiver_input(space, actor, &frame, sizeof(frame)));

    nav_vertices = {-5, -3, 5, -3, 5, 3, -5, 3};
    obstacle_vertices = {20, 20, 21, 20, 20, 21};
    support_vertices = {-.5, -.5, .5, -.5, .5, .5, -.5, .5};
    nav = Polygon(100, nav_vertices);
    obstacle = Polygon(101, obstacle_vertices);
    support = Polygon(102, support_vertices);
  }

  void TearDown() override {
    if (session) sb_destroy_session(session);
  }

  void RefreshTime(int64_t maximum_age) {
    frame.observed_at_ns = sb_read_rich_monotonic_ns();
    frame.valid_until_ns = frame.observed_at_ns + maximum_age;
  }

  sb_receiver_approach_polygon_t Polygon(uint64_t id,
                                         const std::vector<double>& vertices) {
    sb_receiver_approach_polygon_t result{};
    result.struct_size = sizeof(result);
    result.vertex_count = vertices.size() / 2;
    result.nav_element_id = id;
    result.region_id = 30;
    result.coordinate_convention = SB_RECEIVER_APPROACH_CONTEXT_COORDINATE_CONVENTION;
    result.affine_convention = SB_RECEIVER_APPROACH_CONTEXT_AFFINE_CONVENTION;
    Identity(result.local_to_space_row_major, 3);
    result.vertices_xy = vertices.data();
    return result;
  }

  sb_receiver_approach_context_registration_t Registration() {
    sb_receiver_approach_context_registration_t result{};
    result.struct_size = sizeof(result);
    result.feature_version = SB_RECEIVER_APPROACH_CONTEXT_FEATURE_VERSION;
    result.expected_receiver_input_epoch = epoch;
    result.expected_receiver_input_sequence = frame.sequence;
    result.geometry_revision = 11;
    result.requested_valid_until_ns = frame.observed_at_ns + 500000000;
    result.navigation_root_utf8 = Bytes("navigation_root");
    result.navigation_root_bytes = std::strlen("navigation_root");
    result.coordinate_convention = SB_RECEIVER_APPROACH_CONTEXT_COORDINATE_CONVENTION;
    Identity(result.navigation_root_pose_row_major, 4);
    result.navigation_root_pose_row_major[3] = -3;
    result.navigable_regions = &nav;
    result.navigable_region_count = 1;
    result.obstacle_regions = &obstacle;
    result.obstacle_region_count = 1;
    result.footprint_radius_m = .25;
    result.required_clearance_m = .15;
    result.destination.struct_size = sizeof(result.destination);
    result.destination.space = space;
    result.destination.region_id = 30;
    result.destination.object = element;
    result.destination.affordance_id = 50;
    result.destination.support_revision = 12;
    result.destination.local_support = support;
    result.destination.coordinate_convention =
        SB_RECEIVER_APPROACH_CONTEXT_COORDINATE_CONVENTION;
    Identity(result.destination.safe_approach_pose_row_major, 4);
    result.destination.safe_approach_pose_row_major[3] = 3;
    return result;
  }

  int Register(sb_receiver_approach_context_registration_t* value,
               uint64_t* revision) {
    return sb_register_receiver_approach_context(
        session, space, actor, SB_RECEIVER_APPROACH_CONTEXT_FEATURE_VERSION,
        value, sizeof(*value), revision);
  }

  sb_session_t session = 0;
  sb_space_t space = 0;
  sb_element_t element = 0;
  sb_actor_t actor = 0;
  uint64_t epoch = 0;
  sb_receiver_input_frame_t frame{};
  std::vector<sb_receiver_anchor_input_t> anchors;
  std::vector<double> nav_vertices, obstacle_vertices, support_vertices;
  sb_receiver_approach_polygon_t nav{}, obstacle{}, support{};
};

TEST(CReceiverApproachContextLayoutTest, NegotiatedLayoutsAndBoundsAreStable) {
  sb_receiver_approach_context_feature_t feature;
  std::memset(&feature, 0x7f, sizeof(feature));
  EXPECT_EQ(sizeof(feature), sb_get_receiver_approach_context_feature(nullptr, 0));
  EXPECT_EQ(sizeof(feature), sb_get_receiver_approach_context_feature(&feature, 1));
  EXPECT_EQ(0x7f7f7f7fu, feature.descriptor_size);
  ASSERT_EQ(sizeof(feature),
            sb_get_receiver_approach_context_feature(&feature, sizeof(feature)));
  EXPECT_EQ(1u, feature.feature_version);
  EXPECT_EQ(sizeof(sb_receiver_approach_context_registration_t),
            feature.registration_size);
  EXPECT_EQ(sizeof(sb_receiver_approach_polygon_t), feature.polygon_size);
  EXPECT_EQ(sizeof(sb_receiver_approach_destination_t), feature.destination_size);
  EXPECT_EQ(64u, feature.max_polygons);
  EXPECT_EQ(256u, feature.max_vertices_per_polygon);
  EXPECT_EQ(32768u, feature.max_total_vertices);
  EXPECT_EQ(1u, feature.coordinate_convention);
  EXPECT_EQ(1u, feature.native_clock_domain);
  const uint32_t counts[] = {13, 18, 8, 10};
  for (uint32_t kind = 1; kind <= 4; ++kind) {
    EXPECT_EQ(counts[kind - 1], sb_get_receiver_approach_context_field_count(kind));
    for (uint32_t index = 0; index < counts[kind - 1]; ++index)
      EXPECT_NE(UINT32_MAX,
                sb_get_receiver_approach_context_field_offset(kind, index));
    EXPECT_EQ(UINT32_MAX,
              sb_get_receiver_approach_context_field_offset(kind, counts[kind - 1]));
  }
  EXPECT_EQ(0u, sb_get_receiver_approach_context_field_count(0));
  EXPECT_EQ(offsetof(sb_receiver_approach_context_registration_t, destination),
            sb_get_receiver_approach_context_field_offset(2, 17));
}

TEST_F(CReceiverApproachContextTest,
       CompleteCurrentRegistrationDeepCopiesAndGenerationFencesReplacement) {
  auto registration = Registration();
  uint64_t first = 0;
  ASSERT_EQ(SB_RECEIVER_APPROACH_CONTEXT_OK, Register(&registration, &first));
  ASSERT_NE(0u, first);
  // A caller mutation cannot silently amend the registration: it is only ever
  // considered by a new, generation-fenced all-or-nothing replacement.
  nav_vertices[0] = 999;
  registration.expected_registration_generation = 0;
  uint64_t ignored = 0;
  EXPECT_EQ(SB_RECEIVER_APPROACH_CONTEXT_REPLACED, Register(&registration, &ignored));
  registration.expected_registration_generation = 1;
  EXPECT_EQ(SB_RECEIVER_APPROACH_CONTEXT_INVALID_GEOMETRY,
            Register(&registration, &ignored));
  nav_vertices[0] = -5;
  registration.geometry_revision = 13;
  uint64_t second = 0;
  ASSERT_EQ(SB_RECEIVER_APPROACH_CONTEXT_OK, Register(&registration, &second));
  EXPECT_GT(second, first);
}

TEST_F(CReceiverApproachContextTest,
       RejectsMalformedRootCurrentnessGeometryPolicyDestinationAndVersion) {
  auto registration = Registration();
  uint64_t revision = 99;
  EXPECT_EQ(SB_RECEIVER_APPROACH_CONTEXT_UNSUPPORTED_VERSION,
            sb_register_receiver_approach_context(session, space, actor, 2,
                &registration, sizeof(registration), &revision));
  EXPECT_EQ(SB_RECEIVER_APPROACH_CONTEXT_INVALID_ARGUMENT,
            sb_register_receiver_approach_context(session, space, actor, 1,
                &registration, sizeof(registration) - 1, &revision));
  const uint8_t nul_root[] = {'x', 0};
  registration.navigation_root_utf8 = nul_root;
  registration.navigation_root_bytes = sizeof(nul_root);
  EXPECT_EQ(SB_RECEIVER_APPROACH_CONTEXT_INVALID_ARGUMENT, Register(&registration, &revision));
  registration = Registration();
  registration.navigation_root_utf8 = Bytes("pelvis");
  registration.navigation_root_bytes = std::strlen("pelvis");
  registration.navigation_root_pose_row_major[3] = -3;
  EXPECT_EQ(SB_RECEIVER_APPROACH_CONTEXT_MISSING_NAVIGATION_ROOT,
            Register(&registration, &revision));
  registration = Registration();
  nav_vertices = {0, 0, 1, 1, 0, 1, 1, 0};
  nav.vertices_xy = nav_vertices.data();
  EXPECT_EQ(SB_RECEIVER_APPROACH_CONTEXT_INVALID_GEOMETRY,
            Register(&registration, &revision));
  nav_vertices = {-5, -3, 5, -3, 5, 3, -5, 3};
  nav.vertices_xy = nav_vertices.data();
  registration = Registration();
  registration.footprint_radius_m = 0;
  EXPECT_EQ(SB_RECEIVER_APPROACH_CONTEXT_MISSING_POLICY,
            Register(&registration, &revision));
  registration = Registration();
  registration.destination.safe_approach_pose_row_major[3] = 99;
  EXPECT_EQ(SB_RECEIVER_APPROACH_CONTEXT_MISSING_DESTINATION,
            Register(&registration, &revision));
  registration = Registration();
  registration.requested_valid_until_ns = 1;
  EXPECT_EQ(SB_RECEIVER_APPROACH_CONTEXT_EXPIRED, Register(&registration, &revision));
  registration = Registration();
  registration.navigable_region_count = 65;
  EXPECT_EQ(SB_RECEIVER_APPROACH_CONTEXT_INVALID_ARGUMENT,
            Register(&registration, &revision));
}

TEST_F(CReceiverApproachContextTest, ChangedReceiverInputSequenceRejectsOldRegistration) {
  auto registration = Registration();
  frame.sequence = 2;
  RefreshTime(1000000000);
  ASSERT_EQ(SB_RICH_RESULT_OK,
            sb_submit_receiver_input(space, actor, &frame, sizeof(frame)));
  uint64_t revision = 0;
  EXPECT_EQ(SB_RECEIVER_APPROACH_CONTEXT_STALE, Register(&registration, &revision));
  registration.expected_receiver_input_sequence = 2;
  ASSERT_EQ(SB_RECEIVER_APPROACH_CONTEXT_OK, Register(&registration, &revision));
}

}  // namespace
