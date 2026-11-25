// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstring>
#include <thread>
#include <vector>

#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/api/c/spatialblend_planar_support.h"

namespace {

TEST(CPlanarSupportLayoutTest, NegotiatesEveryFieldWithoutChangingRichAbi) {
  sb_planar_support_feature_t feature;
  std::memset(&feature, 0x7f, sizeof(feature));
  EXPECT_EQ(sb_get_planar_support_feature(nullptr, 0), sizeof(feature));
  EXPECT_EQ(sb_get_planar_support_feature(&feature, 1), sizeof(feature));
  EXPECT_EQ(feature.descriptor_size, 0x7f7f7f7fu);
  ASSERT_EQ(sb_get_planar_support_feature(&feature, sizeof(feature)),
            sizeof(feature));
  EXPECT_EQ(feature.descriptor_size, sizeof(feature));
  EXPECT_EQ(feature.feature_version, 1u);
  EXPECT_EQ(feature.header_size, sizeof(sb_planar_support_header_t));
  EXPECT_EQ(feature.max_points, 65536u);
  EXPECT_EQ(feature.coordinate_convention, 1u);
  EXPECT_EQ(feature.plane_convention, 1u);
  const std::vector<std::vector<uint32_t>> offsets = {
      {offsetof(sb_planar_support_feature_t, descriptor_size),
       offsetof(sb_planar_support_feature_t, feature_version),
       offsetof(sb_planar_support_feature_t, header_size),
       offsetof(sb_planar_support_feature_t, max_points),
       offsetof(sb_planar_support_feature_t, coordinate_convention),
       offsetof(sb_planar_support_feature_t, plane_convention)},
      {offsetof(sb_planar_support_header_t, struct_size),
       offsetof(sb_planar_support_header_t, feature_version),
       offsetof(sb_planar_support_header_t, definition_kind),
       offsetof(sb_planar_support_header_t, geometry_role),
       offsetof(sb_planar_support_header_t, space),
       offsetof(sb_planar_support_header_t, element),
       offsetof(sb_planar_support_header_t, tree_revision),
       offsetof(sb_planar_support_header_t, support_revision),
       offsetof(sb_planar_support_header_t, point_count),
       offsetof(sb_planar_support_header_t, plane_convention),
       offsetof(sb_planar_support_header_t, plane_to_world_row_major)}};
  for (uint32_t kind = 1; kind <= offsets.size(); ++kind) {
    ASSERT_EQ(sb_get_planar_support_field_count(kind), offsets[kind - 1].size());
    for (uint32_t index = 0; index < offsets[kind - 1].size(); ++index)
      EXPECT_EQ(sb_get_planar_support_field_offset(kind, index),
                offsets[kind - 1][index]);
    EXPECT_EQ(sb_get_planar_support_field_offset(kind, offsets[kind - 1].size()),
              UINT32_MAX);
  }
  EXPECT_EQ(sb_get_planar_support_field_count(0), 0u);
  EXPECT_EQ(sb_get_planar_support_field_count(3), 0u);
  EXPECT_EQ(sb_get_planar_support_field_offset(3, 0), UINT32_MAX);
  sb_rich_realization_feature_t rich{};
  sb_get_rich_realization_feature(&rich, sizeof(rich));
  EXPECT_EQ(rich.feature_version, 1u);
}

class CPlanarSupportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    session = sb_create_session();
    source = sb_add_space(session);
    receiver = sb_add_space(session);
    for (auto space : {source, receiver}) {
      const auto room = sb_add_element(space, 0);
      sb_set_element_shared_id(space, room, 100);
      const double polygon[] = {-5, -5, 5, -5, 5, 5, -5, 5};
      sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, polygon, 4);
      const auto referent = sb_add_element(space, room);
      sb_set_element_shared_id(space, referent, 200);
      const double box[] = {1, 2, 3, 2, 3, 4, 1, 4};
      sb_set_element_polygon(space, referent, SB_GEOMETRY_CONTAINMENT, box, 4);
      const auto actor = sb_create_actor(space);
      sb_add_anchor(space, actor, "root", 0, 0, 0, nullptr);
      sb_set_actor_element(space, actor, room);
      if (space == source) {
        source_actor = actor;
        sb_bind(space, actor, "root", SB_KERNEL_SPACING);
      } else {
        receiver_room = room;
        support = referent;
        SetSurface(0.7);
        ASSERT_NE(sb_activate_rich_realization(space, actor, 1), 0u);
      }
    }
    sb_set_min_move(source, 0);
    sb_set_min_yaw(source, 0);
    sb_register_on_rich_realization(
        receiver,
        [](const sb_rich_realization_snapshot_t* value, void* context) {
          auto* self = static_cast<CPlanarSupportTest*>(context);
          self->snapshot = value->header;
          sb_planar_support_header_t header{};
          self->callback_read = self->Read(nullptr, 0, &header);
        }, this);
    Solve();
  }
  void TearDown() override { if (session) sb_destroy_session(session); }
  void SetSurface(double height) {
    // Tilted world-absolute plane, not parallel to XY. Chart frame must roundtrip
    // the exact ordered vertices, without applying the element transform again.
    vertices = {1, 2, height, 3, 2, height + 0.4,
                3, 4, height + 0.4, 1, 4, height};
    sb_set_element_surface(receiver, support, vertices.data(), 4,
                           SB_GEOMETRY_SURFACE);
  }
  void Solve() {
    sb_update_actor_state(source, source_actor, 0.1 * ++sequence, 0.2, 0.1);
    ASSERT_GT(snapshot.revision, 0u);
    ASSERT_EQ(snapshot.known_revision_mask & 10u, 10u);
  }
  int Read(double* xy, uint32_t capacity, sb_planar_support_header_t* out) {
    return sb_copy_planar_support(session, receiver, support, 1,
        snapshot.tree_revision, snapshot.support_revision, xy, capacity,
        out, sizeof(*out));
  }
  sb_session_t session = 0;
  sb_space_t source = 0, receiver = 0;
  sb_actor_t source_actor = 0;
  sb_element_t support = 0, receiver_room = 0;
  sb_rich_realization_header_t snapshot{};
  std::vector<double> vertices;
  int sequence = 0, callback_read = 0;
};

TEST_F(CPlanarSupportTest, SizeProbeNoPartialWritesAndWorldFrameRoundTrip) {
  // Rich delivery is outside the solve lock, so a reentrant read is supported.
  EXPECT_EQ(callback_read, SB_PLANAR_SUPPORT_BUFFER_TOO_SMALL);
  sb_planar_support_header_t header{};
  ASSERT_EQ(Read(nullptr, 0, &header), SB_PLANAR_SUPPORT_BUFFER_TOO_SMALL);
  ASSERT_EQ(header.point_count, 4u);
  double xy[10];
  for (double& value : xy) value = 12345;
  EXPECT_EQ(Read(xy, 3, &header), SB_PLANAR_SUPPORT_BUFFER_TOO_SMALL);
  for (double value : xy) EXPECT_EQ(value, 12345);
  ASSERT_EQ(Read(xy, 4, &header), SB_PLANAR_SUPPORT_OK);
  EXPECT_EQ(xy[8], 12345);
  EXPECT_EQ(xy[9], 12345);
  EXPECT_EQ(header.space, receiver);
  EXPECT_EQ(header.element, support);
  EXPECT_EQ(header.definition_kind, 1u);
  EXPECT_EQ(header.geometry_role, SB_GEOMETRY_SURFACE);
  EXPECT_EQ(header.tree_revision, snapshot.tree_revision);
  EXPECT_EQ(header.support_revision, snapshot.support_revision);
  for (int i = 0; i < 4; ++i)
    for (int row = 0; row < 3; ++row)
      EXPECT_NEAR(header.plane_to_world_row_major[row * 4] * xy[i * 2] +
                      header.plane_to_world_row_major[row * 4 + 1] * xy[i * 2 + 1] +
                      header.plane_to_world_row_major[row * 4 + 3],
                  vertices[i * 3 + row], 1e-12);
  xy[0] = -999;  // Detached caller memory cannot mutate native geometry.
  ASSERT_EQ(Read(xy, 4, &header), SB_PLANAR_SUPPORT_OK);
  EXPECT_NE(xy[0], -999);
}

TEST_F(CPlanarSupportTest, ChangedGeometryCannotRestampASizeProbe) {
  sb_planar_support_header_t header{};
  ASSERT_EQ(Read(nullptr, 0, &header), SB_PLANAR_SUPPORT_BUFFER_TOO_SMALL);
  const auto captured = header;
  SetSurface(1.1);
  double xy[8] = {99};
  EXPECT_EQ(Read(xy, 4, &header), SB_PLANAR_SUPPORT_STALE);
  EXPECT_EQ(header.point_count, 0u);
  EXPECT_EQ(xy[0], 99);
  Solve();
  ASSERT_EQ(Read(xy, 4, &header), SB_PLANAR_SUPPORT_OK);
  EXPECT_GT(header.support_revision, captured.support_revision);
  EXPECT_GT(header.tree_revision, captured.tree_revision);
  EXPECT_NE(header.plane_to_world_row_major[11],
            captured.plane_to_world_row_major[11]);
}

TEST_F(CPlanarSupportTest, TreeEditsInvalidateButDoNotDoubleTransformSurface) {
  double xy[8];
  sb_planar_support_header_t before{}, after{};
  ASSERT_EQ(Read(xy, 4, &before), SB_PLANAR_SUPPORT_OK);
  double transform[16] = {1, 0, 0, 8, 0, 1, 0, 9, 0, 0, 1, 10, 0, 0, 0, 1};
  sb_set_element_transform(receiver, support, transform);
  EXPECT_EQ(Read(xy, 4, &after), SB_PLANAR_SUPPORT_STALE);
  Solve();
  ASSERT_EQ(Read(xy, 4, &after), SB_PLANAR_SUPPORT_OK);
  EXPECT_EQ(std::memcmp(before.plane_to_world_row_major,
                        after.plane_to_world_row_major, 16 * sizeof(double)), 0);
}

TEST_F(CPlanarSupportTest, SessionOwnershipUnsupportedAndMissingGeometry) {
  double xy[8];
  sb_planar_support_header_t header{};
  const auto other = sb_create_session();
  EXPECT_EQ(sb_copy_planar_support(other, receiver, support, 1,
      snapshot.tree_revision, snapshot.support_revision, xy, 4, &header,
      sizeof(header)), SB_PLANAR_SUPPORT_UNAVAILABLE);
  sb_destroy_session(other);
  EXPECT_EQ(sb_copy_planar_support(session, receiver, 99999, 1,
      snapshot.tree_revision, snapshot.support_revision, xy, 4, &header,
      sizeof(header)), SB_PLANAR_SUPPORT_UNAVAILABLE);
  ASSERT_EQ(sb_copy_planar_support(session, receiver, receiver_room, 1,
      snapshot.tree_revision, snapshot.support_revision, xy, 4, &header,
      sizeof(header)), SB_PLANAR_SUPPORT_OK);
  EXPECT_EQ(header.definition_kind, 2u);
  EXPECT_EQ(header.geometry_role, SB_GEOMETRY_NAVIGABLE);
  sb_set_element_box(receiver, support, SB_GEOMETRY_OBSTACLE, 0, 0, 0, 1, 1, 1);
  Solve();
  EXPECT_EQ(Read(xy, 4, &header), SB_PLANAR_SUPPORT_UNAVAILABLE);
}

TEST_F(CPlanarSupportTest, InvalidArgumentsDoNotWriteCallerStorage) {
  sb_planar_support_header_t header;
  std::memset(&header, 0x7f, sizeof(header));
  double xy[8] = {99};
  auto invalid = [&](uint32_t version, uint64_t tree, uint64_t support_revision,
                     double* output, uint32_t capacity, uint32_t size) {
    EXPECT_EQ(sb_copy_planar_support(session, receiver, support, version, tree,
        support_revision, output, capacity, &header, size),
        SB_PLANAR_SUPPORT_INVALID_ARGUMENT);
    EXPECT_EQ(header.struct_size, 0x7f7f7f7fu);
    EXPECT_EQ(xy[0], 99);
  };
  invalid(2, 1, 1, xy, 4, sizeof(header));
  invalid(1, 0, 1, xy, 4, sizeof(header));
  invalid(1, 1, 0, xy, 4, sizeof(header));
  invalid(1, 1, 1, nullptr, 4, sizeof(header));
  invalid(1, 1, 1, xy, 65537, sizeof(header));
  invalid(1, 1, 1, xy, 4, sizeof(header) - 1);
  EXPECT_EQ(sb_copy_planar_support(session, receiver, support, 1, 1, 1, xy,
      4, nullptr, sizeof(header)), SB_PLANAR_SUPPORT_INVALID_ARGUMENT);
}

TEST_F(CPlanarSupportTest, ConcurrentSessionRetirementPinsReadLifetime) {
  // Freeze the original query revisions. Teardown's terminal rich callback must
  // not race this test's reader by overwriting its fixture-owned snapshot.
  sb_register_on_rich_realization(receiver, nullptr, nullptr);
  const auto retiring = session;
  std::atomic<bool> started{false};
  std::atomic<int> unexpected{0};
  std::thread reader([&] {
    started.store(true);
    for (int i = 0; i < 1000; ++i) {
      double xy[8];
      sb_planar_support_header_t header{};
      int result = Read(xy, 4, &header);
      if (result != SB_PLANAR_SUPPORT_OK && result != SB_PLANAR_SUPPORT_BUSY &&
          result != SB_PLANAR_SUPPORT_UNAVAILABLE) ++unexpected;
    }
  });
  while (!started.load()) std::this_thread::yield();
  sb_destroy_session(retiring);
  reader.join();
  EXPECT_EQ(unexpected.load(), 0);
  sb_planar_support_header_t header{};
  EXPECT_EQ(Read(nullptr, 0, &header), SB_PLANAR_SUPPORT_UNAVAILABLE);
  session = 0;
}

}  // namespace
