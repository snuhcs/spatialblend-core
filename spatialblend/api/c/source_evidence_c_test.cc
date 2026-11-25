// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <future>
#include <string>
#include <vector>

#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/api/c/spatialblend_source_evidence.h"

namespace {
constexpr int64_t kAge = 1000000000;

TEST(CSourceEvidenceLayout, DescriptorAndSeparateFeatureNegotiation) {
  sb_source_evidence_feature_t f;
  std::memset(&f, 0x7f, sizeof(f));
  EXPECT_EQ(sb_get_source_evidence_feature(nullptr, 0), sizeof(f));
  EXPECT_EQ(sb_get_source_evidence_feature(&f, sizeof(f) - 1), sizeof(f));
  EXPECT_EQ(f.descriptor_size, 0x7f7f7f7fu);
  ASSERT_EQ(sb_get_source_evidence_feature(&f, sizeof(f)), sizeof(f));
  EXPECT_EQ(f.descriptor_size, sizeof(f));
  EXPECT_EQ(f.feature_version, 1u);
  EXPECT_EQ(f.header_size, sizeof(sb_source_evidence_header_t));
  EXPECT_EQ(f.anchor_size, sizeof(sb_source_evidence_anchor_t));
  EXPECT_EQ(f.relation_size, sizeof(sb_source_evidence_relation_t));
  EXPECT_EQ(f.node_size, sizeof(sb_source_evidence_node_t));
  EXPECT_EQ(f.copy_size, sizeof(sb_source_evidence_copy_t));
  EXPECT_EQ(f.max_anchors, 128u);
  EXPECT_EQ(f.max_relations, 256u);
  EXPECT_EQ(f.max_path_depth, 64u);
  EXPECT_EQ(f.max_nodes, 8192u);
  EXPECT_EQ(f.max_points, 65536u);
  EXPECT_EQ(f.max_name_bytes, 128u);
  EXPECT_EQ(f.max_utf8_bytes, SB_SOURCE_EVIDENCE_MAX_UTF8);
  EXPECT_EQ(f.coordinate_convention, 1u);
  EXPECT_EQ(f.max_age_ns, kAge);
  sb_rich_realization_feature_t rich{};
  sb_get_rich_realization_feature(&rich, sizeof(rich));
  EXPECT_EQ(rich.feature_version, 1u);
  EXPECT_EQ(f.monotonic_clock_id, rich.monotonic_clock_id);
  EXPECT_EQ(f.monotonic_clock_epoch, rich.monotonic_clock_epoch);
  EXPECT_NE(f.monotonic_clock_epoch, 0u);
  const uint32_t counts[] = {18, 14, 9, 26, 7, 17};
  const uint32_t sizes[] = {sizeof(f),       f.header_size, f.anchor_size,
                            f.relation_size, f.node_size,   f.copy_size};
  for (uint32_t kind = 1; kind <= 6; ++kind) {
    EXPECT_EQ(sb_get_source_evidence_field_count(kind), counts[kind - 1]);
    uint32_t previous = 0;
    for (uint32_t i = 0; i < counts[kind - 1]; ++i) {
      const auto offset = sb_get_source_evidence_field_offset(kind, i);
      if (i) EXPECT_GT(offset, previous);
      EXPECT_LT(offset, sizes[kind - 1]);
      previous = offset;
    }
    EXPECT_EQ(sb_get_source_evidence_field_offset(kind, counts[kind - 1]),
              UINT32_MAX);
  }
  EXPECT_EQ(sb_get_source_evidence_field_count(0), 0u);
  EXPECT_EQ(sb_get_source_evidence_field_count(7), 0u);
  EXPECT_EQ(sb_get_source_evidence_field_offset(7, 0), UINT32_MAX);
  EXPECT_EQ(
      sb_get_source_evidence_field_offset(4, 19),
      offsetof(sb_source_evidence_relation_t, support_to_world_row_major));
}

class CSourceEvidenceTest : public ::testing::Test {
 public:
  void SetUp() override {
    session = sb_create_session();
    space = sb_add_space(session);
    ASSERT_EQ(sb_enable_actor_observation_frames(session), 1);
    sb_element_t parent = 0;
    for (int i = 0; i < 4; ++i) {
      parent = sb_add_element(space, parent);
      hierarchy.push_back(parent);
      sb_set_element_shared_id(space, parent, 100 + i);
    }
    seat = parent;
    sb_set_element_tag(space, seat, "Sittable");
    // Family scope, not the bind's element: AffordancePose is the subtree of
    // the containing node. The root is that container, so the seat is in it.
    const double room[] = {-10, -10, 10, -10, 10, 10, -10, 10};
    ASSERT_EQ(sb_set_element_polygon(space, hierarchy[0], SB_GEOMETRY_NAVIGABLE,
                                     room, 4),
              SB_SCENE_OK);
    const double vertices[] = {-1, -1, .5, 1, -1, .5, 1, 1, .5, -1, 1, .5};
    sb_set_element_surface(space, seat, vertices, 4, SB_GEOMETRY_SURFACE);
    owner = sb_add_element(space, hierarchy[1]);
    actor = sb_create_actor(space);
    sb_set_actor_element(space, actor, owner);
    sb_add_anchor(space, actor, "navigation", 0, 0, 0, nullptr);
    ASSERT_EQ(sb_add_anchor_with_position_dimension(space, actor, "body", 0, 0,
                                                    0, nullptr, 3),
              1);
    ASSERT_EQ(sb_bind_affordance(space, actor, "body", "Sittable"),
              1);
    epoch = sb_begin_actor_observation_epoch(space, actor, "navigation");
    ASSERT_NE(epoch, 0u);
    copy.struct_size = sizeof(copy);
  }
  void TearDown() override {
    if (session) sb_destroy_session(session);
  }
  int Enable(uint32_t enabled = 1) {
    return sb_try_enable_source_evidence(session, space, actor, 1, epoch,
                                         enabled);
  }
  void Observe(int64_t sequence = 0, double height = 1) {
    sb_actor_observation_t row{"body", 0, 0, height, .25};
    ASSERT_EQ(sb_apply_actor_observation_frame(space, actor, epoch, sequence,
                                               sequence * .1, &row, 1),
              1);
  }
  int Read(int64_t sequence = 0, int64_t age = kAge) {
    return sb_copy_source_evidence(session, space, actor, 1, epoch, sequence,
                                   age, &copy, sizeof(copy));
  }
  void Allocate() {
    anchors.resize(copy.anchor_count);
    relations.resize(copy.relation_count);
    nodes.resize(copy.node_count);
    xy.resize(copy.point_count * 2);
    utf8.resize(copy.utf8_bytes);
    copy.anchor_capacity = anchors.size();
    copy.anchors = anchors.data();
    copy.relation_capacity = relations.size();
    copy.relations = relations.data();
    copy.node_capacity = nodes.size();
    copy.nodes = nodes.data();
    copy.point_capacity = xy.size() / 2;
    copy.boundary_xy = xy.data();
    copy.utf8_capacity = utf8.size();
    copy.utf8 = utf8.data();
  }
  std::string Name(uint32_t offset, uint32_t length) {
    return std::string(utf8.begin() + offset, utf8.begin() + offset + length);
  }
  sb_session_t session = 0;
  sb_space_t space = 0;
  sb_actor_t actor = 0;
  sb_element_t seat = 0, owner = 0;
  uint64_t epoch = 0;
  std::vector<sb_element_t> hierarchy;
  sb_source_evidence_copy_t copy{};
  std::vector<sb_source_evidence_anchor_t> anchors;
  std::vector<sb_source_evidence_relation_t> relations;
  std::vector<sb_source_evidence_node_t> nodes;
  std::vector<double> xy;
  std::vector<uint8_t> utf8;
};

TEST_F(CSourceEvidenceTest, FutureOnlyEnableExactClockAndCompleteBodyOnlyCopy) {
  Observe();
  ASSERT_EQ(Enable(), SB_SOURCE_EVIDENCE_OK);
  EXPECT_EQ(Read(), SB_SOURCE_EVIDENCE_UNAVAILABLE);
  Observe(1);
  ASSERT_EQ(Read(1), SB_SOURCE_EVIDENCE_BUFFER_TOO_SMALL);
  EXPECT_EQ(copy.header.space, space);
  EXPECT_EQ(copy.header.actor, actor);
  EXPECT_EQ(copy.header.actor_element, owner);
  EXPECT_EQ(copy.header.epoch, epoch);
  EXPECT_EQ(copy.header.sequence, 1);
  EXPECT_DOUBLE_EQ(copy.header.source_time_seconds, .1);
  EXPECT_GT(copy.header.receipt_monotonic_ns, 0);
  EXPECT_EQ(copy.anchor_count, 2u);
  EXPECT_EQ(copy.relation_count, 1u);
  EXPECT_EQ(copy.node_count, 4u);
  EXPECT_EQ(copy.point_count, 4u);
  const auto probe = copy.header;
  Allocate();
  ASSERT_EQ(Read(1), SB_SOURCE_EVIDENCE_OK);
  EXPECT_EQ(copy.header.receipt_monotonic_ns, probe.receipt_monotonic_ns);
  EXPECT_EQ(Name(anchors[0].name_offset, anchors[0].name_length), "navigation");
  EXPECT_EQ(anchors[0].observed, 0u);
  EXPECT_EQ(anchors[0].locomotion_root, 1u);
  EXPECT_EQ(anchors[1].observed, 1u);
  EXPECT_EQ(anchors[1].position_dimension, 3u);
  EXPECT_DOUBLE_EQ(anchors[1].position_z, 1);
  EXPECT_EQ(relations[0].phase, SB_PHASE_ACTIVE);
  EXPECT_EQ(relations[0].has_projection, 1u);
  EXPECT_EQ(relations[0].bounded_planar_support, 1u);
  EXPECT_NEAR(relations[0].anchor_support_distance, .5, 1e-12);
  EXPECT_NEAR(relations[0].proximity, 1 / 1.5, 1e-12);
  EXPECT_EQ(relations[0].path_count, hierarchy.size());
  for (size_t i = 0; i < hierarchy.size(); ++i) {
    EXPECT_EQ(nodes[i].element, hierarchy[i]);
    EXPECT_EQ(nodes[i].shared, 100 + i);
  }
  EXPECT_EQ(Name(nodes.back().tag_offset, nodes.back().tag_length), "Sittable");
  nodes.back().shared = 9999;
  xy[0] = 9999;
  utf8[0] = 'X';
  EXPECT_EQ(Read(1), SB_SOURCE_EVIDENCE_OK);
  EXPECT_EQ(nodes.back().shared, 103);
  EXPECT_NE(xy[0], 9999);
  EXPECT_EQ(Name(anchors[0].name_offset, anchors[0].name_length), "navigation");
}

TEST_F(CSourceEvidenceTest,
       MultipleAnchorsPreserveBindingAnchorKernelAndPathAssociation) {
  ASSERT_EQ(sb_add_anchor_with_position_dimension(space, actor, "head", 0, 0,
                                                  0, nullptr, 3),
            1);
  epoch = sb_begin_actor_observation_epoch(space, actor, "navigation");
  ASSERT_NE(epoch, 0u);
  ASSERT_EQ(sb_bind(space, actor, "navigation", SB_KERNEL_SPACING),
            1);
  ASSERT_EQ(sb_bind(space, actor, "head", SB_KERNEL_FACING),
            1);
  ASSERT_EQ(Enable(), SB_SOURCE_EVIDENCE_OK);
  const sb_actor_observation_t rows[] = {
      {"navigation", 0, 0, 0, 0},
      {"body", 0, 0, 1, .25},
      {"head", .2, 0, 1, .5},
  };
  ASSERT_EQ(sb_apply_actor_observation_frame(space, actor, epoch, 0, 0, rows,
                                             3),
            1);
  ASSERT_EQ(Read(), SB_SOURCE_EVIDENCE_BUFFER_TOO_SMALL);
  Allocate();
  ASSERT_EQ(Read(), SB_SOURCE_EVIDENCE_OK);
  ASSERT_EQ(copy.anchor_count, 3u);
  // Spacing is the containing node's children. The seat is nested under that
  // child, so the spacing bind does not select it (an
  // app-supplied root does not change the family). AffordancePose is the
  // subtree and Facing is global, and both keep the seat.
  // The "Sittable" tag does not narrow Spacing or
  // Facing, so those bindings also carry rows for the other referents in their
  // scope. The association under test is that of the rows that reach the seat.
  std::vector<uint32_t> seat_rows;
  for (uint32_t i = 0; i < copy.relation_count; ++i) {
    ASSERT_GT(relations[i].path_count, 0u);
    if (nodes[relations[i].path_offset + relations[i].path_count - 1].element ==
        seat)
      seat_rows.push_back(i);
  }
  ASSERT_EQ(seat_rows.size(), 2u);
  const uint32_t expected_ordinal[] = {0, 2};
  const uint32_t expected_anchor[] = {1, 2};
  const int32_t expected_kernel[] = {SB_KERNEL_AFFORDANCE_POSE,
                                     SB_KERNEL_FACING};
  const char* expected_name[] = {"body", "head"};
  for (size_t k = 0; k < seat_rows.size(); ++k) {
    const uint32_t i = seat_rows[k];
    EXPECT_EQ(relations[i].binding_ordinal, expected_ordinal[k]);
    ASSERT_LT(relations[i].anchor_ordinal, copy.anchor_count);
    EXPECT_EQ(relations[i].anchor_ordinal, expected_anchor[k]);
    EXPECT_EQ(relations[i].kernel, expected_kernel[k]);
    EXPECT_EQ(Name(anchors[relations[i].anchor_ordinal].name_offset,
                   anchors[relations[i].anchor_ordinal].name_length),
              expected_name[k]);
  }
}

TEST_F(CSourceEvidenceTest, AnyShortBufferWritesNoArrayPrefix) {
  ASSERT_EQ(Enable(), 1);
  Observe();
  ASSERT_EQ(Read(), SB_SOURCE_EVIDENCE_BUFFER_TOO_SMALL);
  Allocate();
  std::memset(anchors.data(), 0x33, anchors.size() * sizeof(anchors[0]));
  std::memset(relations.data(), 0x33, relations.size() * sizeof(relations[0]));
  std::memset(nodes.data(), 0x33, nodes.size() * sizeof(nodes[0]));
  std::fill(xy.begin(), xy.end(), 12345);
  std::fill(utf8.begin(), utf8.end(), 0x33);
  for (uint32_t* capacity :
       {&copy.anchor_capacity, &copy.relation_capacity, &copy.node_capacity,
        &copy.point_capacity, &copy.utf8_capacity}) {
    --*capacity;
    EXPECT_EQ(Read(), SB_SOURCE_EVIDENCE_BUFFER_TOO_SMALL);
    EXPECT_EQ(anchors[0].name_offset, 0x33333333u);
    EXPECT_EQ(relations[0].anchor_ordinal, 0x33333333u);
    EXPECT_EQ(nodes[0].tag_offset, 0x33333333u);
    for (double value : xy) EXPECT_EQ(value, 12345);
    for (auto value : utf8) EXPECT_EQ(value, 0x33);
    ++*capacity;
  }
  ASSERT_EQ(Read(), SB_SOURCE_EVIDENCE_OK);
}

TEST_F(CSourceEvidenceTest, ExactRetryCannotJoinNewFrameOrChangedScene) {
  ASSERT_EQ(Enable(), 1);
  Observe();
  ASSERT_EQ(Read(), SB_SOURCE_EVIDENCE_BUFFER_TOO_SMALL);
  Allocate();
  sb_set_element_tag(space, hierarchy[1], "changed");
  EXPECT_EQ(Read(), SB_SOURCE_EVIDENCE_STALE);
  EXPECT_EQ(copy.header.epoch, 0u);
  EXPECT_EQ(copy.anchor_count, 0u);
  Observe(1);
  EXPECT_EQ(Read(), SB_SOURCE_EVIDENCE_STALE);
  EXPECT_EQ(Read(1),
            SB_SOURCE_EVIDENCE_BUFFER_TOO_SMALL);  // New tag needs more UTF8.
  Allocate();
  EXPECT_EQ(Read(1), SB_SOURCE_EVIDENCE_OK);
  EXPECT_EQ(Read(1, 1),
            SB_SOURCE_EVIDENCE_STALE);  // Original receipt, no sleep/restamp.
}

TEST_F(CSourceEvidenceTest,
       ResetControlsSessionMismatchAndRetirementAreUnavailable) {
  ASSERT_EQ(Enable(), 1);
  Observe();
  ASSERT_EQ(Read(), SB_SOURCE_EVIDENCE_BUFFER_TOO_SMALL);
  Allocate();
  ASSERT_EQ(Read(), SB_SOURCE_EVIDENCE_OK);
  const auto foreign = sb_create_session();
  EXPECT_EQ(sb_copy_source_evidence(foreign, space, actor, 1, epoch, 0, kAge,
                                    &copy, sizeof(copy)),
            SB_SOURCE_EVIDENCE_UNAVAILABLE);
  EXPECT_EQ(sb_try_enable_source_evidence(foreign, space, actor, 1, epoch, 0),
            SB_SOURCE_EVIDENCE_UNAVAILABLE);
  sb_destroy_session(foreign);
  EXPECT_EQ(Read(), SB_SOURCE_EVIDENCE_OK);
  ASSERT_EQ(sb_clear_actor_observation_anchor(space, actor, epoch, "body"), 1);
  EXPECT_EQ(Read(), SB_SOURCE_EVIDENCE_UNAVAILABLE);
  Observe(1);
  EXPECT_EQ(Read(1), SB_SOURCE_EVIDENCE_OK);
  const auto old = epoch;
  epoch = sb_begin_actor_observation_epoch(space, actor, "navigation");
  EXPECT_NE(epoch, old);
  Observe();
  EXPECT_EQ(Read(), SB_SOURCE_EVIDENCE_UNAVAILABLE);
  ASSERT_EQ(Enable(), 1);
  EXPECT_EQ(sb_try_enable_source_evidence(session, space, actor, 1, old, 0),
            SB_SOURCE_EVIDENCE_UNAVAILABLE);
  Observe(1);
  EXPECT_EQ(Read(1), SB_SOURCE_EVIDENCE_OK);
  sb_destroy_session(session);
  EXPECT_EQ(Read(1), SB_SOURCE_EVIDENCE_UNAVAILABLE);
  session = 0;
}

TEST_F(CSourceEvidenceTest, InvalidArgumentsAndMalformedUtf8NeverExposePrefix) {
  ASSERT_EQ(Enable(), 1);
  Observe();
  EXPECT_EQ(Read(0, 0), SB_SOURCE_EVIDENCE_INVALID_ARGUMENT);
  EXPECT_EQ(Read(0, kAge + 1), SB_SOURCE_EVIDENCE_INVALID_ARGUMENT);
  EXPECT_EQ(Read(-1), SB_SOURCE_EVIDENCE_INVALID_ARGUMENT);
  EXPECT_EQ(sb_try_enable_source_evidence(session, space, actor, 1, epoch, 2),
            SB_SOURCE_EVIDENCE_INVALID_ARGUMENT);
  EXPECT_EQ(sb_try_enable_source_evidence(session, space, actor, 2, epoch, 1),
            SB_SOURCE_EVIDENCE_INVALID_ARGUMENT);
  copy.anchor_capacity = 1;
  EXPECT_EQ(Read(), SB_SOURCE_EVIDENCE_INVALID_ARGUMENT);
  copy.anchor_capacity = 129;
  EXPECT_EQ(Read(), SB_SOURCE_EVIDENCE_INVALID_ARGUMENT);
  copy.anchor_capacity = 0;
  EXPECT_EQ(sb_copy_source_evidence(session, space, actor, 1, epoch, 0, kAge,
                                    &copy, sizeof(copy) - 1),
            SB_SOURCE_EVIDENCE_INVALID_ARGUMENT);
  sb_set_element_tag(space, hierarchy[1], "\xc0\xaf");
  Observe(1);
  EXPECT_EQ(Read(1), SB_SOURCE_EVIDENCE_UNAVAILABLE);
  EXPECT_EQ(copy.header.epoch, 0u);
  EXPECT_EQ(copy.utf8_bytes, 0u);
  sb_set_element_tag(space, hierarchy[1], u8"영역");
  Observe(2);
  ASSERT_EQ(Read(2), SB_SOURCE_EVIDENCE_BUFFER_TOO_SMALL);
  Allocate();
  ASSERT_EQ(Read(2), SB_SOURCE_EVIDENCE_OK);
  EXPECT_EQ(Name(nodes[1].tag_offset, nodes[1].tag_length), u8"영역");
}

TEST_F(CSourceEvidenceTest, CallbackReentryAndConcurrentSolveReturnBusy) {
  ASSERT_EQ(Enable(), 1);
  struct WaitState {
    CSourceEvidenceTest* self;
    std::promise<void> entered, release;
  } wait{this};
  auto release = wait.release.get_future().share();
  struct Context {
    WaitState* state;
    std::shared_future<void> release;
  } context{&wait, release};
  sb_register_on_relation_event(
      space,
      [](const sb_relation_event_t*, void* opaque) {
        auto& c = *static_cast<Context*>(opaque);
        EXPECT_EQ(c.state->self->Read(), SB_SOURCE_EVIDENCE_BUSY);
        EXPECT_EQ(c.state->self->Enable(), SB_SOURCE_EVIDENCE_BUSY);
        c.state->entered.set_value();
        EXPECT_EQ(c.release.wait_for(std::chrono::seconds(5)),
                  std::future_status::ready);
      },
      &context);
  auto worker = std::async(std::launch::async, [&] { Observe(); });
  EXPECT_EQ(wait.entered.get_future().wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  EXPECT_EQ(Read(), SB_SOURCE_EVIDENCE_BUSY);
  EXPECT_EQ(Enable(), SB_SOURCE_EVIDENCE_BUSY);
  wait.release.set_value();
  worker.get();
  EXPECT_EQ(Read(), SB_SOURCE_EVIDENCE_BUFFER_TOO_SMALL);
  sb_register_on_relation_event(space, nullptr, nullptr);
}

}  // namespace
