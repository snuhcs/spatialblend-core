// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <condition_variable>
#include <future>
#include <functional>
#include <thread>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "spatialblend/api/c/c_test_embodied.h"
#include "spatialblend/api/c/spatialblend.h"

namespace {

// The test plays the application: it owns each avatar's embodied pose and
// publishes it through the sampling port (core keeps none, and proposes
// nothing until a pose exists).
uint64_t NextPoseSequence(sb_space_t space, sb_actor_t actor) {
  static std::map<std::pair<sb_space_t, sb_actor_t>, uint64_t> sequence;
  return ++sequence[{space, actor}];
}

sb_anchor_pose_t RootPose(double x, double y, double yaw, uint64_t epoch = 1) {
  sb_anchor_pose_t pose{};
  pose.struct_size = sizeof(pose);
  pose.x = x;
  pose.y = y;
  pose.orientation.struct_size = sizeof(sb_orientation_t);
  pose.orientation.yaw = yaw;
  pose.epoch = epoch;
  return pose;
}

int32_t PublishPose(sb_space_t space, sb_actor_t actor, double x, double y,
                    double yaw) {
  const sb_anchor_pose_t pose = RootPose(x, y, yaw);
  return sb_publish_anchor_pose(space, actor, 0, &pose, 0.0,
                                NextPoseSequence(space, actor));
}

}  // namespace

TEST(CApiE2ETest, AtomicActorObservationAdmissionAndEpochClock) {
  static_assert(offsetof(sb_actor_observation_t, anchor_name) == 0);
  static_assert(offsetof(sb_actor_observation_t, position_x) >
                offsetof(sb_actor_observation_t, anchor_name));
  static_assert(offsetof(sb_actor_observation_t, position_x) ==
                ((sizeof(void*) + alignof(double) - 1) / alignof(double)) *
                    alignof(double));
  static_assert(sizeof(sb_actor_observation_t) ==
                offsetof(sb_actor_observation_t, yaw) + sizeof(double));

  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  const sb_actor_t actor = sb_create_actor(space);
  ASSERT_NE(session, 0);
  ASSERT_NE(space, 0);
  ASSERT_NE(actor, 0);
  sb_add_anchor(space, actor, "navigation", 0, 0, 0, nullptr);
  sb_add_anchor(space, actor, "hand", 0, 0, 1, nullptr);
  uint64_t disabled_epoch = 71;
  int64_t disabled_sequence = 72;
  double disabled_time = 73;
  int32_t disabled_sample = 74;
  EXPECT_EQ(sb_begin_actor_observation_epoch(space, actor, "navigation"), 0u);
  EXPECT_EQ(sb_get_actor_observation_clock(space, actor, &disabled_epoch,
                                           &disabled_sequence, &disabled_time,
                                           &disabled_sample),
            0);
  EXPECT_EQ(disabled_epoch, 71u);
  ASSERT_EQ(sb_enable_actor_observation_frames(session), 1);
  EXPECT_EQ(sb_enable_actor_observation_frames(session), 1);

  const uint64_t epoch =
      sb_begin_actor_observation_epoch(space, actor, "navigation");
  ASSERT_NE(epoch, 0u);
  uint64_t got_epoch = 999;
  int64_t sequence = 999;
  double time = 999;
  int32_t has_sample = 999;
  ASSERT_EQ(sb_get_actor_observation_clock(space, actor, &got_epoch, &sequence,
                                           &time, &has_sample),
            1);
  EXPECT_EQ(got_epoch, epoch);
  EXPECT_EQ(sequence, -1);
  EXPECT_DOUBLE_EQ(time, 0);
  EXPECT_EQ(has_sample, 0);

  sb_actor_observation_t frame[] = {{"navigation", 1, 2, 0, 0},
                                    {"hand", 1, 2, 3, 0}};
  ASSERT_EQ(
      sb_apply_actor_observation_frame(space, actor, epoch, 0, 0, frame, 2), 1);
  frame[0].position_x = 2;
  ASSERT_EQ(
      sb_apply_actor_observation_frame(space, actor, epoch, 1, .2, frame, 2),
      1);
  ASSERT_EQ(sb_get_actor_observation_clock(space, actor, &got_epoch, &sequence,
                                           &time, &has_sample),
            1);
  EXPECT_EQ(sequence, 1);
  EXPECT_DOUBLE_EQ(time, .2);

  const sb_actor_state_t accepted_state = sb_get_actor_state(space, actor);
  auto unchanged = [&] {
    uint64_t e = 0;
    int64_t s = 0;
    double t = 0;
    int32_t h = 0;
    EXPECT_EQ(sb_get_actor_observation_clock(space, actor, &e, &s, &t, &h), 1);
    EXPECT_EQ(e, epoch);
    EXPECT_EQ(s, 1);
    EXPECT_DOUBLE_EQ(t, .2);
    EXPECT_EQ(h, 1);
    const sb_actor_state_t state = sb_get_actor_state(space, actor);
    EXPECT_DOUBLE_EQ(state.position_x, accepted_state.position_x);
    EXPECT_DOUBLE_EQ(state.position_y, accepted_state.position_y);
    EXPECT_DOUBLE_EQ(state.yaw, accepted_state.yaw);
  };
  sb_actor_observation_t duplicate[] = {{"navigation", 3, 2, 0, 0},
                                        {"navigation", 3, 2, 0, 0}};
  sb_actor_observation_t body_nan[] = {
      {"hand", 0, 0, 0, std::numeric_limits<double>::quiet_NaN()}};
  sb_actor_observation_t unknown[] = {{"unknown", 0, 0, 0, 0}};
  EXPECT_EQ(sb_apply_actor_observation_frame(space, actor, epoch, 2, .3,
                                             duplicate, 2),
            0);
  unchanged();
  EXPECT_EQ(
      sb_apply_actor_observation_frame(space, actor, epoch, 2, .3, body_nan, 1),
      0);
  unchanged();
  EXPECT_EQ(
      sb_apply_actor_observation_frame(space, actor, epoch, 2, .3, unknown, 1),
      0);
  unchanged();
  EXPECT_EQ(
      sb_apply_actor_observation_frame(space, actor, epoch, 2, -.1, frame, 2),
      0);
  unchanged();
  EXPECT_EQ(
      sb_apply_actor_observation_frame(space, actor, epoch, 2, .1, frame, 2),
      0);
  unchanged();
  EXPECT_EQ(
      sb_apply_actor_observation_frame(space, actor, epoch, 1, .3, frame, 2),
      0);
  unchanged();
  EXPECT_EQ(
      sb_apply_actor_observation_frame(space, actor, epoch, -1, .3, frame, 2),
      0);
  unchanged();
  EXPECT_EQ(sb_apply_actor_observation_frame(
                space, actor, epoch, 2, std::numeric_limits<double>::infinity(),
                frame, 2),
            0);
  unchanged();
  EXPECT_EQ(
      sb_apply_actor_observation_frame(space, actor, epoch, 2, .3, nullptr, 1),
      0);
  unchanged();
  EXPECT_EQ(
      sb_apply_actor_observation_frame(space, actor, epoch, 2, .3, frame, -1),
      0);
  unchanged();

  sb_actor_observation_t invalid[] = {
      {"navigation", 99, 2, 0, 0},
      {"unknown", 0, 0, 0, std::numeric_limits<double>::quiet_NaN()}};
  EXPECT_EQ(
      sb_apply_actor_observation_frame(space, actor, epoch, 2, .3, invalid, 2),
      0);
  EXPECT_EQ(sb_get_actor_observation_clock(space, actor, &got_epoch, &sequence,
                                           &time, &has_sample),
            1);
  EXPECT_EQ(sequence, 1);
  EXPECT_DOUBLE_EQ(time, .2);
  EXPECT_EQ(
      sb_apply_actor_observation_frame(space, actor, epoch, 2, .3, frame, 2),
      1);
  ASSERT_EQ(sb_clear_actor_observation_anchor(space, actor, epoch, "hand"), 1);
  EXPECT_EQ(sb_get_actor_observation_clock(space, actor, &got_epoch, &sequence,
                                           &time, &has_sample),
            1);
  EXPECT_EQ(sequence, 2);
  EXPECT_DOUBLE_EQ(time, .3);

  const uint64_t next =
      sb_begin_actor_observation_epoch(space, actor, "navigation");
  ASSERT_GT(next, epoch);
  ASSERT_EQ(sb_get_actor_observation_clock(space, actor, &got_epoch, &sequence,
                                           &time, &has_sample),
            1);
  EXPECT_EQ(has_sample, 0);
  EXPECT_EQ(
      sb_apply_actor_observation_frame(space, actor, epoch, 3, .4, frame, 2),
      0);
  uint64_t untouched_epoch = 7;
  int64_t untouched_sequence = 8;
  double untouched_time = 9;
  int32_t untouched_has = 10;
  EXPECT_EQ(sb_get_actor_observation_clock(0, actor, &untouched_epoch,
                                           &untouched_sequence, &untouched_time,
                                           &untouched_has),
            0);
  EXPECT_EQ(untouched_epoch, 7u);
  sb_destroy_session(session);
}

TEST(CApiSessionTest, CreateAndDestroy) {
  sb_session_t s = sb_create_session();
  EXPECT_NE(s, 0);
  sb_destroy_session(s);
}

TEST(CApiSessionTest, AddSpace) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  EXPECT_NE(space, 0);
  sb_destroy_session(s);
}

TEST(CApiSessionTest, MultipleSpaces) {
  sb_session_t s = sb_create_session();
  sb_space_t s1 = sb_add_space(s);
  sb_space_t s2 = sb_add_space(s);
  EXPECT_NE(s1, s2);
  sb_destroy_session(s);
}

TEST(CApiSessionTest, CleanupAll) {
  sb_session_t s = sb_create_session();
  sb_add_space(s);
  sb_cleanup_all();
}

TEST(CApiSessionTest, InvalidSessionReturnsZero) {
  sb_space_t space = sb_add_space(999);
  EXPECT_EQ(space, 0);
}

// ---------------------------------------------------------------------------
// Element hierarchy tests
// ---------------------------------------------------------------------------

TEST(CApiElementTest, AddElementHierarchy) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);

  sb_element_t room = sb_add_element(space, 0);
  EXPECT_NE(room, 0);

  sb_element_t obj = sb_add_element(space, room);
  EXPECT_NE(obj, 0);
  EXPECT_NE(room, obj);

  sb_destroy_session(s);
}

// ABI 4.8: an element moves under another parent with its identity, shared id and subtree, and its new parent-local
// transform is checked with the move (sibling overlap refuses it and leaves the element where it was).
TEST(CApiElementTest, ReparentKeepsIdentityAndChecksTheNewPlacement) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  const auto rect = [](double x0, double y0, double x1, double y1, double* out) {
    const double p[] = {x0, y0, x1, y0, x1, y1, x0, y1};
    for (int i = 0; i < 8; ++i) out[i] = p[i];
  };
  double poly[8];
  sb_element_t room = sb_add_element(space, 0);
  rect(0, 0, 20, 20, poly);
  ASSERT_EQ(sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, poly, 4), SB_SCENE_OK);
  // A table (footprint 4..8 x 4..8), a holding element with no footprint, and a board standing on the table.
  sb_element_t table = sb_add_element(space, room);
  rect(4, 4, 8, 8, poly);
  ASSERT_EQ(sb_set_element_polygon(space, table, SB_GEOMETRY_OBSTACLE, poly, 4), SB_SCENE_OK);
  sb_element_t holding = sb_add_element(space, room);
  sb_element_t board = sb_add_element(space, table);
  sb_set_element_shared_id(space, board, 4242);
  rect(5, 5, 7, 7, poly);
  ASSERT_EQ(sb_set_element_polygon(space, board, SB_GEOMETRY_CONTAINMENT, poly, 4), SB_SCENE_OK);
  sb_element_t child = sb_add_element(space, board);
  const double identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

  // Directly under the room it would overlap the table: refused, put back, the status says so.
  sb_element_t where = 99;
  ASSERT_EQ(sb_get_element_parent(space, board, &where), 1);
  EXPECT_EQ(where, table);
  EXPECT_EQ(sb_reparent_element(space, board, room, identity), SB_SCENE_CONTAINMENT_OVERLAP);
  ASSERT_EQ(sb_get_element_parent(space, board, &where), 1);
  EXPECT_EQ(where, table) << "a refused move leaves the element where it was";
  EXPECT_EQ(sb_find_element_by_shared_id(space, 4242), board);
  // Through a holding element (no footprint) it is accepted, with its subtree and shared id.
  EXPECT_EQ(sb_reparent_element(space, board, holding, identity), SB_SCENE_OK);
  ASSERT_EQ(sb_get_element_parent(space, board, &where), 1);
  EXPECT_EQ(where, holding);
  ASSERT_EQ(sb_get_element_parent(space, child, &where), 1);
  EXPECT_EQ(where, board) << "the subtree moves with it";
  EXPECT_EQ(sb_find_element_by_shared_id(space, 4242), board);
  // A parent inside the element's own subtree is refused; so is a root.
  EXPECT_EQ(sb_reparent_element(space, board, child, identity), SB_SCENE_INVALID_ARGUMENT);
  // A parent that is no element is refused and leaves the tree intact (-1 would make the element the root).
  for (sb_element_t no_parent : {sb_element_t(-1), sb_element_t(0), sb_element_t(987654)}) {
    EXPECT_EQ(sb_reparent_element(space, board, no_parent, identity), SB_SCENE_INVALID_ARGUMENT) << no_parent;
    ASSERT_EQ(sb_get_element_parent(space, board, &where), 1);
    EXPECT_EQ(where, holding) << "the element stays where it was";
    ASSERT_EQ(sb_get_element_parent(space, room, &where), 1);
    EXPECT_EQ(where, 0) << "the room is still the root";
  }
  EXPECT_EQ(sb_reparent_element(space, room, holding, identity), SB_SCENE_INVALID_ARGUMENT);
  // Back onto the table, with the new local transform checked at the same time.
  EXPECT_EQ(sb_reparent_element(space, board, table, identity), SB_SCENE_OK);
  EXPECT_EQ(sb_find_element_by_shared_id(space, 4242), board);
  ASSERT_EQ(sb_get_element_parent(space, board, &where), 1);
  EXPECT_EQ(where, table);
  ASSERT_EQ(sb_get_element_parent(space, room, &where), 1);
  EXPECT_EQ(where, 0) << "the root has no parent";
  EXPECT_EQ(sb_get_element_parent(space, 987654, &where), 0);
  sb_destroy_session(s);
}

// The obstacles a navigable owns exclude its ANCESTORS, so a reparent that changes a navigable's ancestry changes its free space
// although no obstacle moved (the obstacle sync keys on ancestry too).
TEST(CApiElementTest, ReparentingANavigableChangesWhichFootprintsAreItsObstacles) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  sb_element_t room = sb_add_element(space, 0);
  double poly[8] = {0, 0, 10, 0, 10, 10, 0, 10};
  ASSERT_EQ(sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, poly, 4), SB_SCENE_OK);
  sb_element_t table = sb_add_element(space, room);
  double box[8] = {4, 4, 6, 4, 6, 6, 4, 6};
  ASSERT_EQ(sb_set_element_polygon(space, table, SB_GEOMETRY_OBSTACLE, box, 4), SB_SCENE_OK);
  sb_element_t holding = sb_add_element(space, room);  // no footprint of its own
  sb_element_t annex = sb_add_element(space, holding);
  double inner[8] = {4.2, 4.2, 5.8, 4.2, 5.8, 5.8, 4.2, 5.8};
  ASSERT_EQ(sb_set_element_polygon(space, annex, SB_GEOMETRY_NAVIGABLE, inner, 4), SB_SCENE_OK);
  sb_actor_t actor = sb_create_actor(space);
  sb_set_actor_element(space, actor, annex);
  double xy[2] = {0, 0};
  // Under `holding` the table is a footprint inside the annex: (5, 5) is in it, and the nearest free point is outside.
  EXPECT_EQ(sb_query_nearest_clear_point(space, actor, 5.0, 5.0, xy), 0)
      << "the annex lies wholly inside the table's footprint: no free point";
  // Under the table itself the table is an ancestor: not an obstacle of the annex, so (5, 5) is free.
  const double identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  ASSERT_EQ(sb_reparent_element(space, annex, table, identity), SB_SCENE_OK);
  ASSERT_EQ(sb_query_nearest_clear_point(space, actor, 5.0, 5.0, xy), 1);
  EXPECT_NEAR(xy[0], 5.0, 1e-9);
  EXPECT_NEAR(xy[1], 5.0, 1e-9);
  sb_destroy_session(s);
}

// The driver services (body clearance belongs to the driver): the core's own free space has no body-radius ring, and
// the driver asks for a course and a placement that keep a body of the actor's registered radius clear of a footprint.
TEST(CApiElementTest, DriverServicesKeepTheBodyClearOfFootprints) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  sb_element_t room = sb_add_element(space, 0);
  double poly[8] = {0, 0, 10, 0, 10, 10, 0, 10};
  ASSERT_EQ(sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, poly, 4), SB_SCENE_OK);
  sb_element_t table = sb_add_element(space, room);
  double box[8] = {4, 4, 6, 4, 6, 6, 4, 6};
  ASSERT_EQ(sb_set_element_polygon(space, table, SB_GEOMETRY_OBSTACLE, box, 4), SB_SCENE_OK);
  sb_actor_t actor = sb_create_actor(space);
  ASSERT_NE(actor, 0);
  sb_set_actor_element(space, actor, room);
  double xy[2] = {0, 0};
  // No radius registered: the placement is the point itself when it is free of the footprint (a point body).
  ASSERT_EQ(sb_query_nearest_clear_point(space, actor, 6.1, 5.0, xy), 1);
  EXPECT_NEAR(xy[0], 6.1, 1e-9);
  // A body of radius 0.3 m: 6.1 is 0.1 m from the footprint, so the nearest clear point is 0.3 m from it.
  ASSERT_EQ(sb_set_body_radius(space, 0.3), 0);
  ASSERT_EQ(sb_query_nearest_clear_point(space, actor, 6.1, 5.0, xy), 1);
  EXPECT_NEAR(xy[0], 6.3, 1e-3);
  EXPECT_NEAR(xy[1], 5.0, 1e-3);
  EXPECT_EQ(sb_query_nearest_clear_point(space, actor, 6.1, 5.0, nullptr), -1);
  EXPECT_EQ(sb_query_nearest_clear_point(space, actor, std::nan(""), 5.0, xy), -1);
  sb_destroy_session(s);
}

// Scene validation: overlapping sibling containment regions and a child that
// sticks out of its parent are rejected with an explicit status and a message
// naming both elements; touching edges and nesting are accepted.
// The Unity Foundation scenario: the actor placed in the room
// (sb_set_actor_element on the room, as the Foundation tests do). The room's furniture is not
// the actor's body: a stand-off region near the room's ring and one far from it are both
// accepted (each extends the room's scope), and re-binding the actor
// afterwards succeeds.
TEST(CApiElementTest, AnActorPlacedInTheRoomLeavesItsFurnitureToTheParentRule) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  sb_element_t room = sb_add_element(space, 0);
  double poly[8] = {0, 0, 10, 0, 10, 10, 0, 10};
  ASSERT_EQ(sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, poly, 4), SB_SCENE_OK);
  sb_actor_t actor = sb_create_actor(space);
  ASSERT_NE(actor, 0);
  sb_set_actor_element(space, actor, room);
  sb_element_t desk = sb_add_element(space, room);
  // 0.2 m past the ring: nearby (clear of the far one's corridor, y 4..6).
  double near_band[8] = {10.2, 7, 11, 7, 11, 9, 10.2, 9};
  EXPECT_EQ(sb_set_element_polygon(space, desk, SB_GEOMETRY_CONTAINMENT, near_band, 4), SB_SCENE_OK);
  sb_element_t stray = sb_add_element(space, room);
  double far_band[8] = {30, 4, 31, 4, 31, 6, 30, 6};  // 20 m out
  char message[256] = {};
  EXPECT_EQ(sb_set_element_polygon(space, stray, SB_GEOMETRY_CONTAINMENT, far_band, 4), SB_SCENE_OK);
  EXPECT_EQ(sb_get_scene_status(space, message, sizeof message), SB_SCENE_OK);
  // The actor moves to its own element: nothing refuses it.
  sb_element_t body = sb_add_element(space, room);
  sb_set_actor_element(space, actor, body);
  EXPECT_EQ(sb_get_scene_status(space, message, sizeof message), SB_SCENE_OK);
  sb_destroy_session(s);
}

TEST(CApiElementTest, ContainmentLayoutIsValidatedAtRegistration) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  const auto rect = [](double x0, double y0, double x1, double y1,
                       double* out) {
    const double p[] = {x0, y0, x1, y0, x1, y1, x0, y1};
    for (int i = 0; i < 8; ++i) out[i] = p[i];
  };
  double poly[8];
  sb_element_t room = sb_add_element(space, 0);
  rect(0, 0, 20, 20, poly);
  ASSERT_EQ(sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, poly, 4),
            SB_SCENE_OK);
  sb_element_t a = sb_add_element(space, room);
  sb_element_t b = sb_add_element(space, room);
  rect(2, 2, 8, 8, poly);
  ASSERT_EQ(sb_set_element_polygon(space, a, SB_GEOMETRY_CONTAINMENT, poly, 4),
            SB_SCENE_OK);
  char message[256] = {};
  // Overlap: rejected, not registered, both elements named.
  rect(5, 2, 12, 8, poly);
  EXPECT_EQ(sb_set_element_polygon(space, b, SB_GEOMETRY_CONTAINMENT, poly, 4),
            SB_SCENE_CONTAINMENT_OVERLAP);
  EXPECT_EQ(sb_get_scene_status(space, message, sizeof message),
            SB_SCENE_CONTAINMENT_OVERLAP);
  EXPECT_NE(std::string(message).find("element " + std::to_string(a)),
            std::string::npos) << message;
  EXPECT_NE(std::string(message).find("element " + std::to_string(b)),
            std::string::npos) << message;
  // Touching edges are accepted (the rejected change left nothing behind).
  rect(8, 2, 12, 8, poly);
  EXPECT_EQ(sb_set_element_polygon(space, b, SB_GEOMETRY_CONTAINMENT, poly, 4),
            SB_SCENE_OK);
  EXPECT_EQ(sb_get_scene_status(space, message, sizeof message), SB_SCENE_OK);
  // A child far from its parent extends the parent's scope, but here the bridge from a (x <= 8) to the child (x 13..14)
  // crosses a's sibling b (x 8..12): an overlap, rejected and not registered.
  // One that sticks out into the parent's sibling overlaps it; entire nesting
  // is accepted.
  sb_element_t child = sb_add_element(space, a);
  rect(13, 4, 14, 6, poly);
  EXPECT_EQ(
      sb_set_element_polygon(space, child, SB_GEOMETRY_CONTAINMENT, poly, 4),
      SB_SCENE_CONTAINMENT_OVERLAP);
  rect(6, 4, 9, 6, poly);
  EXPECT_EQ(
      sb_set_element_polygon(space, child, SB_GEOMETRY_CONTAINMENT, poly, 4),
      SB_SCENE_CONTAINMENT_OVERLAP);
  rect(3, 3, 5, 5, poly);
  EXPECT_EQ(
      sb_set_element_polygon(space, child, SB_GEOMETRY_CONTAINMENT, poly, 4),
      SB_SCENE_OK);
  sb_destroy_session(s);
}

TEST(CApiElementTest, SetElementProperties) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  sb_element_t room = sb_add_element(space, 0);
  ASSERT_NE(room, 0);

  // Set transform (identity matrix).
  double identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  sb_set_element_transform(space, room, identity);

  // Set shared ID.
  sb_set_element_shared_id(space, room, 42);

  // Set tag.
  sb_set_element_tag(space, room, "living_room");

  // Set polygon (simple square).
  double polygon[] = {0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0};
  sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, polygon, 4);

  sb_destroy_session(s);
}

// ---------------------------------------------------------------------------
// Actor tests
// ---------------------------------------------------------------------------

TEST(CApiActorTest, CreateActorWithAnchors) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  sb_element_t room = sb_add_element(space, 0);

  sb_actor_t actor = sb_create_actor(space);
  EXPECT_NE(actor, 0);

  sb_add_anchor(space, actor, "head", 0.0, 0.0, 1.7, NULL);
  sb_add_anchor(space, actor, "feet", 0.0, 0.0, 0.0, NULL);
  EXPECT_EQ(sb_set_anchor_active(space, actor, "head", 0), 1);
  EXPECT_EQ(sb_set_anchor_active(space, actor, "head", 1), 1);
  EXPECT_EQ(sb_set_anchor_active(space, actor, "missing", 0), 0);
  sb_set_actor_element(space, actor, room);

  sb_destroy_session(s);
}

TEST(CApiActorTest, DestroyActorRetiresOnlyThatActorAndIsIdempotentSafe) {
  sb_session_t session = sb_create_session();
  sb_space_t space = sb_add_space(session);
  ASSERT_NE(space, 0);
  sb_actor_t retired = sb_create_actor(space);
  sb_actor_t survivor = sb_create_actor(space);
  ASSERT_NE(retired, 0);
  ASSERT_NE(survivor, 0);

  EXPECT_EQ(sb_destroy_actor(space, retired), 1);
  EXPECT_EQ(sb_destroy_actor(space, retired), 0);
  // Retirement is exact: an unrelated actor remains usable.
  sb_add_anchor(space, survivor, "body", 0.0, 0.0, 0.0, NULL);
  EXPECT_EQ(sb_destroy_actor(space, survivor), 1);
  EXPECT_EQ(sb_destroy_actor(0, survivor), 0);
  sb_destroy_session(session);
}

// ---------------------------------------------------------------------------
// Bind tests
// ---------------------------------------------------------------------------

TEST(CApiBindTest, BindRelations) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  sb_element_t room = sb_add_element(space, 0);

  sb_actor_t actor = sb_create_actor(space);
  sb_add_anchor(space, actor, "head", 0.0, 0.0, 1.7, NULL);

  sb_bind(space, actor, "head", SB_KERNEL_SPACING);
  sb_bind(space, actor, "head", SB_KERNEL_FACING);

  sb_destroy_session(s);
}

TEST(CApiBindTest, BindTagged) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  sb_element_t room = sb_add_element(space, 0);

  sb_element_t chair = sb_add_element(space, room);
  sb_set_element_tag(space, chair, "seat");

  sb_actor_t actor = sb_create_actor(space);
  sb_add_anchor(space, actor, "body", 0.0, 0.0, 0.5, NULL);

  sb_bind_affordance(space, actor, "body", "seat");

  sb_destroy_session(s);
}

TEST(CApiBindTest, FacingBindsEveryReferentAndTagsDoNotNarrowIt) {
  sb_session_t session = sb_create_session();
  sb_space_t space = sb_add_space(session);
  // Facing has no distance cap. There is no scope or tag argument on the bind
  // (a tag binds only the AffordancePose family): every element below the room is a referent, tagged or not.
  sb_element_t room = sb_add_element(space, 0);
  sb_element_t selected = sb_add_element(space, room);
  sb_set_element_shared_id(space, selected, 9001);  // shared: an unshared element is never a referent (SP-D7-03)
  sb_element_t selected_leaf = sb_add_element(space, selected);
  sb_set_element_shared_id(space, selected_leaf, 9002);  // shared: an unshared element is never a referent (SP-D7-03)
  sb_element_t distractor = sb_add_element(space, room);
  sb_set_element_shared_id(space, distractor, 9003);  // shared: an unshared element is never a referent (SP-D7-03)
  sb_set_element_tag(space, selected_leaf, "seat");
  sb_set_element_tag(space, distractor, "seat");
  double selected_matrix[] = {1, 0, 0, 10, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  double distractor_matrix[] = {1, 0, 0, .1, 0, 1, 0, 0,
                                0, 0, 1, 0,  0, 0, 0, 1};
  sb_set_element_transform(space, selected_leaf, selected_matrix);
  sb_set_element_transform(space, distractor, distractor_matrix);
  sb_actor_t actor = sb_create_actor(space);
  sb_add_anchor(space, actor, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(space, actor, room);

  EXPECT_EQ(
      sb_bind(space, actor, "root", SB_KERNEL_FACING),
      1);
  sb_update_actor_state(space, actor, 0, 0, 0);
  sb_relation_state_t states[4];
  ASSERT_EQ(sb_get_relation_states(space, actor, states, 4), 3);
  std::set<sb_element_t> tagged;
  for (int i = 0; i < 3; ++i) tagged.insert(states[i].referent_element_id);
  EXPECT_TRUE(tagged.count(selected));
  EXPECT_TRUE(tagged.count(selected_leaf));
  EXPECT_TRUE(tagged.count(distractor));

  // A null tag is no filter, not an invalid request. Facing then sees every
  // element except the actor's own.
  sb_actor_t unfiltered = sb_create_actor(space);
  sb_add_anchor(space, unfiltered, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(space, unfiltered, room);
  EXPECT_EQ(sb_bind(space, unfiltered, "root", SB_KERNEL_FACING),
            1);
  sb_update_actor_state(space, unfiltered, 0, 0, 0);
  ASSERT_EQ(sb_get_relation_states(space, unfiltered, states, 4), 3);

  // A missing anchor, an unknown kernel, or a missing id adds no binding.
  EXPECT_EQ(sb_bind(space, actor, "missing", SB_KERNEL_FACING),
            0);
  EXPECT_EQ(sb_bind(space, actor, "root", static_cast<sb_kernel_type_t>(99)),
            0);
  // Deleting an element does not retire the binding. The leaf under it is
  // gone; the other seat remains.
  EXPECT_EQ(sb_remove_local_element(space, selected), 1);
  sb_update_actor_state(space, actor, 0, 0, 0);
  ASSERT_EQ(sb_get_relation_states(space, actor, states, 4), 1);
  EXPECT_EQ(states[0].referent_element_id, distractor);
  sb_destroy_session(session);
}

// ---------------------------------------------------------------------------
// End-to-end: source update -> target callback -> apply avatar state
// ---------------------------------------------------------------------------

static int g_target_callback_count = 0;
static sb_realization_target_t g_last_targets[16];
static int64_t g_last_num_targets = 0;

static void test_on_targets(sb_space_t /*space*/, sb_actor_t /*actor*/,
                            const sb_realization_target_t* targets,
                            int64_t num_targets, void* /*user_data*/) {
  g_target_callback_count++;
  g_last_num_targets = num_targets;
  for (int64_t i = 0; i < num_targets && i < 16; i++) {
    g_last_targets[i] = targets[i];
  }
}

TEST(CApiElementTest, BoxVolumeParticipatesInRelationTargets) {
  g_target_callback_count = 0;
  g_last_num_targets = 0;

  sb_session_t session = sb_create_session();

  sb_space_t source = sb_add_space(session);
  sb_element_t source_room = sb_add_element(source, 0);
  sb_set_element_shared_id(source, source_room, 100);
  double room_poly[] = {0, 0, 10, 0, 10, 10, 0, 10};
  sb_set_element_polygon(source, source_room, SB_GEOMETRY_NAVIGABLE, room_poly,
                         4);
  sb_element_t source_desk = sb_add_element(source, source_room);
  sb_set_element_shared_id(source, source_desk, 200);
  sb_set_element_box(source, source_desk, SB_GEOMETRY_OBSTACLE, 4.0, 4.0, 0.0,
                     6.0, 6.0, 1.0);

  sb_actor_t actor = sb_create_actor(source);
  sb_add_anchor(source, actor, "root", 0, 0, 0, NULL);
  sb_set_actor_element(source, actor, source_room);
  sb_bind(source, actor, "root", SB_KERNEL_SPACING);

  sb_space_t target = sb_add_space(session);
  sb_element_t target_room = sb_add_element(target, 0);
  sb_set_element_shared_id(target, target_room, 100);
  sb_set_element_polygon(target, target_room, SB_GEOMETRY_NAVIGABLE, room_poly,
                         4);
  sb_element_t target_desk = sb_add_element(target, target_room);
  sb_set_element_shared_id(target, target_desk, 200);
  sb_set_element_box(target, target_desk, SB_GEOMETRY_OBSTACLE, 4.0, 4.0, 0.0,
                     6.0, 6.0, 1.0);

  sb_actor_t avatar = sb_create_actor(target);
  PublishPose(target, avatar, 0.0, 0.0, 0.0);

  PublishPose(target, avatar, 0.0, 0.0, 0.0);
  sb_add_anchor(target, avatar, "root", 0, 0, 0, NULL);
  sb_set_actor_element(target, avatar, target_room);
  sb_register_on_targets(target, test_on_targets, nullptr);

  sb_update_actor_state(source, actor, 3.0, 5.0, 0.0);

  EXPECT_GT(g_target_callback_count, 0);
  EXPECT_GT(g_last_num_targets, 0);

  sb_destroy_session(session);
}

TEST(CApiE2ETest, SourceToTargetCallback) {
  g_target_callback_count = 0;
  g_last_num_targets = 0;

  sb_session_t session = sb_create_session();

  // Source space
  sb_space_t sa = sb_add_space(session);
  sb_element_t room_a = sb_add_element(sa, 0);
  sb_set_element_shared_id(sa, room_a, 100);
  double poly[] = {0, 0, 20, 0, 20, 20, 0, 20};
  sb_set_element_polygon(sa, room_a, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_element_t obj_a = sb_add_element(sa, room_a);
  sb_set_element_shared_id(sa, obj_a, 200);

  sb_actor_t actor = sb_create_actor(sa);
  sb_add_anchor(sa, actor, "root", 0, 0, 0, NULL);
  sb_set_actor_element(sa, actor, room_a);
  sb_bind(sa, actor, "root", SB_KERNEL_SPACING);

  // Target space
  sb_space_t sb = sb_add_space(session);
  sb_element_t room_b = sb_add_element(sb, 0);
  sb_set_element_shared_id(sb, room_b, 100);
  sb_set_element_polygon(sb, room_b, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_element_t obj_b = sb_add_element(sb, room_b);
  sb_set_element_shared_id(sb, obj_b, 200);

  sb_actor_t avatar = sb_create_actor(sb);
  PublishPose(sb, avatar, 0.0, 0.0, 0.0);

  PublishPose(sb, avatar, 0.0, 0.0, 0.0);
  sb_add_anchor(sb, avatar, "root", 0, 0, 0, NULL);
  sb_set_actor_element(sb, avatar, room_b);

  sb_register_on_targets(sb, test_on_targets, nullptr);

  // Update source -> should propagate and trigger callback on target space
  sb_update_actor_state(sa, actor, 5.0, 5.0, 0.0);

  EXPECT_GT(g_target_callback_count, 0);
  EXPECT_GT(g_last_num_targets, 0);
  EXPECT_EQ(sb_find_element_by_shared_id(sb, 200), obj_b);
  EXPECT_EQ(sb_find_actor_by_shared_id(sb, 100), avatar);

  g_target_callback_count = 0;
  g_last_num_targets = 0;
  EXPECT_TRUE(sb_poll_realization_targets(sb, avatar));
  EXPECT_GT(g_target_callback_count, 0);
  EXPECT_GT(g_last_num_targets, 0);
  EXPECT_FALSE(sb_poll_realization_targets(sb, 999));

  // The driver publishes the avatar's embodied pose (write-back).
  EXPECT_EQ(PublishPose(sb, avatar, 3.0, 3.0, 0.5), 0);

  sb_destroy_session(session);
}

TEST(CApiSupportCorrespondenceTest,
     ValidatesAtomicallyAndDisposesOnlyItsOwnLocalBatch) {
  sb_session_t session = sb_create_session();
  sb_space_t source = sb_add_space(session);
  sb_space_t target = sb_add_space(session);
  sb_element_t chair_seat = sb_add_element(source, 0);
  sb_element_t chair_back = sb_add_element(source, 0);
  sb_element_t sofa_support = sb_add_element(target, 0);
  sb_element_t other_support = sb_add_element(target, 0);

  // A bad later endpoint must not install the valid earlier correspondence.
  const sb_support_correspondence_t invalid_batch[] = {
      {source, chair_seat, target, sofa_support},
      {source, chair_back, target, 999999},
  };
  EXPECT_EQ(sb_register_support_correspondences(session, invalid_batch, 2), 0);

  // This different target can succeed only if the invalid batch did not leave
  // its first source->target-space slot behind.
  const sb_support_correspondence_t seat_to_sofa[] = {
      {source, chair_seat, target, other_support},
      {source, chair_seat, target, other_support},
  };
  sb_support_correspondence_registration_t seat_registration =
      sb_register_support_correspondences(session, seat_to_sofa, 2);
  ASSERT_NE(seat_registration, 0);
  // Duplicate exact entries in one request are deduplicated into this one
  // lease; another live registration for that owned pair is rejected.
  EXPECT_EQ(sb_register_support_correspondences(session, seat_to_sofa, 1), 0);

  // A conflicting target-space slot is preserved, while distinct sources may
  // intentionally correspond to the same support element.
  const sb_support_correspondence_t conflicting[] = {
      {source, chair_seat, target, sofa_support}};
  EXPECT_EQ(sb_register_support_correspondences(session, conflicting, 1), 0);
  const sb_support_correspondence_t many_to_one[] = {
      {source, chair_back, target, sofa_support}};
  sb_support_correspondence_registration_t back_registration =
      sb_register_support_correspondences(session, many_to_one, 1);
  ASSERT_NE(back_registration, 0);

  sb_dispose_support_correspondences(seat_registration);
  sb_dispose_support_correspondences(seat_registration);  // stale/idempotent
  // The first owned pair is gone, so its formerly conflicting replacement can
  // now be registered; the other source's many-to-one pair remains unrelated.
  sb_support_correspondence_registration_t replacement =
      sb_register_support_correspondences(session, conflicting, 1);
  EXPECT_NE(replacement, 0);

  sb_destroy_session(session);
  sb_dispose_support_correspondences(back_registration);  // torn-down session
  sb_dispose_support_correspondences(replacement);
}

static sb_actor_t g_callback_actor = 0;
static void capture_callback_actor(sb_space_t /*space*/, sb_actor_t actor,
                                   const sb_realization_target_t* /*targets*/,
                                   int64_t /*num_targets*/,
                                   void* /*user_data*/) {
  g_callback_actor = actor;
}

TEST(CApiE2ETest, EmptyKnownSnapshotDispatchesAuthoritativeRelease) {
  g_target_callback_count = 0;
  g_last_num_targets = -1;
  sb_session_t session = sb_create_session();
  sb_space_t source = sb_add_space(session);
  sb_element_t source_room = sb_add_element(source, 0);
  sb_set_element_shared_id(source, source_room, 100);
  double poly[] = {0, 0, 10, 0, 10, 10, 0, 10};
  sb_set_element_polygon(source, source_room, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_element_t source_object = sb_add_element(source, source_room);
  sb_set_element_shared_id(source, source_object, 200);
  sb_actor_t actor = sb_create_actor(source);
  sb_add_anchor(source, actor, "root", 0, 0, 0, NULL);
  sb_set_actor_element(source, actor, source_room);
  sb_bind(source, actor, "root", SB_KERNEL_SPACING);
  sb_set_min_move(source, 0.0);
  sb_set_min_yaw(source, 0.0);

  sb_space_t target = sb_add_space(session);
  sb_element_t target_room = sb_add_element(target, 0);
  sb_set_element_shared_id(target, target_room, 100);
  sb_set_element_polygon(target, target_room, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_element_t target_object = sb_add_element(target, target_room);
  sb_set_element_shared_id(target, target_object, 200);
  sb_actor_t avatar = sb_create_actor(target);
  PublishPose(target, avatar, 0.0, 0.0, 0.0);
  sb_add_anchor(target, avatar, "root", 0, 0, 0, NULL);
  sb_set_actor_element(target, avatar, target_room);
  sb_register_on_targets(target, test_on_targets, nullptr);

  sb_update_actor_state(source, actor, 2.0, 2.0, 0.0);
  ASSERT_GT(g_last_num_targets, 0);
  const int known_count = g_target_callback_count;

  // No explicit linker correspondence exists, so strict mode rejects the
  // next payload and produces an authoritative empty realization snapshot.
  sb_set_referent_resolve_mode(target, SB_REFERENT_RESOLVE_LINKER_ONLY);
  sb_update_actor_state(source, actor, 2.1, 2.0, 0.0);
  EXPECT_GT(g_target_callback_count, known_count);
  EXPECT_EQ(g_last_num_targets, 0);
  EXPECT_TRUE(sb_poll_realization_targets(target, avatar));
  EXPECT_EQ(g_last_num_targets, 0);

  sb_destroy_session(session);
}

TEST(CApiE2ETest, TargetCallbacksContinuePastReceivedUpdateRingCapacity) {
  g_target_callback_count = 0;
  g_last_num_targets = 0;
  sb_session_t session = sb_create_session();
  sb_space_t source = sb_add_space(session);
  sb_element_t source_room = sb_add_element(source, 0);
  sb_set_element_shared_id(source, source_room, 100);
  double poly[] = {0, 0, 20, 0, 20, 20, 0, 20};
  sb_set_element_polygon(source, source_room, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_element_t source_object = sb_add_element(source, source_room);
  sb_set_element_shared_id(source, source_object, 200);
  sb_actor_t actor = sb_create_actor(source);
  sb_add_anchor(source, actor, "root", 0, 0, 0, NULL);
  sb_set_actor_element(source, actor, source_room);
  sb_bind(source, actor, "root", SB_KERNEL_SPACING);
  sb_set_min_move(source, 0.0);
  sb_set_min_yaw(source, 0.0);

  sb_space_t target = sb_add_space(session);
  sb_element_t target_room = sb_add_element(target, 0);
  sb_set_element_shared_id(target, target_room, 100);
  sb_set_element_polygon(target, target_room, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_element_t target_object = sb_add_element(target, target_room);
  sb_set_element_shared_id(target, target_object, 200);
  sb_actor_t avatar = sb_create_actor(target);
  PublishPose(target, avatar, 0.0, 0.0, 0.0);
  sb_add_anchor(target, avatar, "root", 0, 0, 0, NULL);
  sb_set_actor_element(target, avatar, target_room);
  sb_register_on_targets(target, test_on_targets, nullptr);

  for (int i = 0; i < 101; ++i) {
    sb_update_actor_state(source, actor, 1.0 + 0.01 * i, 2.0, 0.0);
  }
  EXPECT_GE(g_target_callback_count, 101);
  EXPECT_GT(g_last_num_targets, 0);

  sb_destroy_session(session);
}

// Affordance anchor target carries height (position_z) through the C ABI.
// The two-stage realizer emits a per-anchor target
// for a non-root binding; its Z is the TARGET referent's world height, proving
// the contract field is plumbed and height transfers cross-space.
//
// NOTE: sb_realization_target_t::anchor_name is valid only DURING the callback
// (the C++ strings backing it die when DispatchRealizationTargets returns), so
// everything name-dependent is captured here rather than read post-hoc.
namespace {
std::string g_aff_first_anchor;
bool g_aff_have_hand = false;
sb_realization_target_t g_aff_hand_target{};
}  // namespace

static void capture_affordance_targets(sb_space_t /*space*/,
                                       sb_actor_t /*actor*/,
                                       const sb_realization_target_t* targets,
                                       int64_t num_targets, void* /*user*/) {
  g_last_num_targets = num_targets;
  if (num_targets > 0 && targets[0].anchor_name) {
    g_aff_first_anchor = targets[0].anchor_name;
  }
  for (int64_t i = 0; i < num_targets; ++i) {
    if (targets[i].anchor_name &&
        (std::string(targets[i].anchor_name) == "hand" ||
         std::string(targets[i].anchor_name) == "custom_body")) {
      g_aff_have_hand = true;
      g_aff_hand_target = targets[i];
    }
  }
}

// Asserts per-anchor affordance IK targets (root + hand) produced by the
// two-stage SolveActorTargets realizer (OnRemoteUpdate appends a
// hysteresis-gated, reach-clamped RealizationTarget per (AffordancePose,
// anchor) group after the root target, restoring what this test originally
// covered before the upstream single-root-Solve merge).
TEST(CApiE2ETest, AffordanceTargetCarriesHeightThroughAbi) {
  g_last_num_targets = 0;
  g_aff_first_anchor.clear();
  g_aff_have_hand = false;

  sb_session_t session = sb_create_session();

  // Source: room with an explicitly authored support plane at z = 0.8.
  sb_space_t sa = sb_add_space(session);
  sb_element_t room_a = sb_add_element(sa, 0);
  sb_set_element_shared_id(sa, room_a, 100);
  double poly[] = {0, 0, 10, 0, 10, 10, 0, 10};
  sb_set_element_polygon(sa, room_a, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_element_t desk_a = sb_add_element(sa, room_a);
  sb_set_element_shared_id(sa, desk_a, 200);
  // Surface vertices are world-baked; this ABI test must not ask grounding to infer
  // a support plane from an unannotated volume or a floor footprint.
  double desk_a_surface[] = {4, 4, .8, 6, 4, .8, 6, 6, .8, 4, 6, .8};
  sb_set_element_surface(sa, desk_a, desk_a_surface, 4, SB_GEOMETRY_SURFACE);

  sb_actor_t actor = sb_create_actor(sa);
  sb_add_anchor(sa, actor, "root", 0, 0, 0, NULL);
  // A hand target exists only while the source hand TOUCHES its support,
  // so the hand rests ON the desk top (z = 0.8).
  sb_add_anchor(sa, actor, "hand", 0.4, 0, 0.8, NULL);
  sb_set_actor_element(sa, actor, room_a);
  sb_bind(sa, actor, "hand", SB_KERNEL_AFFORDANCE_POSE);

  // Target: same shared ids, but the desk sits at a DIFFERENT height z = 0.6.
  sb_space_t sb = sb_add_space(session);
  sb_element_t room_b = sb_add_element(sb, 0);
  sb_set_element_shared_id(sb, room_b, 100);
  sb_set_element_polygon(sb, room_b, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_element_t desk_b = sb_add_element(sb, room_b);
  sb_set_element_shared_id(sb, desk_b, 200);
  double desk_b_surface[] = {2, 6, .6, 4, 6, .6, 4, 8, .6, 2, 8, .6};
  sb_set_element_surface(sb, desk_b, desk_b_surface, 4, SB_GEOMETRY_SURFACE);

  sb_actor_t avatar = sb_create_actor(sb);
  PublishPose(sb, avatar, 0.0, 0.0, 0.0);

  PublishPose(sb, avatar, 0.0, 0.0, 0.0);
  sb_add_anchor(sb, avatar, "root", 0, 0, 0, NULL);
  sb_add_anchor(sb, avatar, "hand", 0.4, 0, 0.9, NULL);
  sb_set_actor_element(sb, avatar, room_b);
  sb_register_on_targets(sb, capture_affordance_targets, nullptr);

  sb_update_actor_state(sa, actor, 5.0, 4.0, 0.0);

  // Contract: root is always present and first.
  ASSERT_GE(g_last_num_targets, 2);
  EXPECT_EQ(g_aff_first_anchor, "root");

  ASSERT_TRUE(g_aff_have_hand);
  EXPECT_EQ(g_aff_hand_target.kernel_type, SB_KERNEL_AFFORDANCE_POSE);
  EXPECT_NE(g_aff_hand_target.has_z, 0);
  // Height comes from the TARGET desk (0.6), not the source desk (0.8).
  EXPECT_NEAR(g_aff_hand_target.position_z, 0.6, 1e-6);

  sb_destroy_session(session);
}

TEST(CApiE2ETest, ObservationFramesDispatchAndRejectAtomically) {
  g_target_callback_count = 0;
  sb_session_t session = sb_create_session();
  sb_space_t source = sb_add_space(session);
  sb_space_t target = sb_add_space(session);
  double poly[] = {0, 0, 10, 0, 10, 10, 0, 10};
  sb_element_t room_a = sb_add_element(source, 0);
  sb_element_t room_b = sb_add_element(target, 0);
  sb_set_element_shared_id(source, room_a, 100);
  sb_set_element_shared_id(target, room_b, 100);
  sb_set_element_polygon(source, room_a, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_set_element_polygon(target, room_b, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_element_t support_a = sb_add_element(source, room_a);
  sb_element_t support_b = sb_add_element(target, room_b);
  sb_set_element_shared_id(source, support_a, 200);
  sb_set_element_shared_id(target, support_b, 200);
  sb_actor_t actor = sb_create_actor(source);
  sb_actor_t avatar = sb_create_actor(target);
  PublishPose(target, avatar, 0.0, 0.0, 0.0);
  sb_add_anchor(source, actor, "root", 0, 0, 0, nullptr);
  sb_add_anchor(target, avatar, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(source, actor, room_a);
  sb_set_actor_element(target, avatar, room_b);
  sb_bind(source, actor, "root", SB_KERNEL_SPACING);
  sb_register_on_targets(target, test_on_targets, nullptr);
  ASSERT_EQ(sb_enable_actor_observation_frames(session), 1);
  const uint64_t epoch =
      sb_begin_actor_observation_epoch(source, actor, "root");
  ASSERT_NE(epoch, 0u);
  const int after_begin = g_target_callback_count;
  sb_actor_observation_t good[] = {{"root", 2, 2, 0, 0}};
  ASSERT_EQ(
      sb_apply_actor_observation_frame(source, actor, epoch, 0, 0, good, 1), 1);
  EXPECT_GT(g_target_callback_count, after_begin);
  EXPECT_GT(g_last_num_targets, 0);
  const int callbacks = g_target_callback_count;
  sb_actor_state_t before = sb_get_actor_state(source, actor);
  sb_actor_observation_t bad[] = {{"root", 9, 9, 0, 0}, {"bad", 0, 0, 0, NAN}};
  EXPECT_EQ(
      sb_apply_actor_observation_frame(source, actor, epoch, 1, .1, bad, 2), 0);
  sb_actor_state_t after = sb_get_actor_state(source, actor);
  EXPECT_DOUBLE_EQ(before.position_x, after.position_x);
  EXPECT_EQ(g_target_callback_count, callbacks);
  EXPECT_EQ(
      sb_apply_actor_observation_frame(source, actor, epoch, 1, .1, good, 1),
      1);
  EXPECT_EQ(sb_clear_actor_observation_anchor(source, actor, epoch, "root"), 1);
  EXPECT_EQ(g_target_callback_count, callbacks + 2);
  EXPECT_EQ(g_last_num_targets, 0);
  EXPECT_EQ(sb_clear_actor_observation_anchor(source, actor, epoch, "root"), 1);
  EXPECT_EQ(g_target_callback_count, callbacks + 2);
  ASSERT_EQ(
      sb_apply_actor_observation_frame(source, actor, epoch, 2, .2, good, 1),
      1);
  ASSERT_GT(g_last_num_targets, 0);
  const int callbacks_before_reset = g_target_callback_count;
  const uint64_t next = sb_begin_actor_observation_epoch(source, actor, "root");
  EXPECT_GT(next, epoch);
  EXPECT_EQ(g_target_callback_count, callbacks_before_reset + 1);
  EXPECT_EQ(g_last_num_targets, 0);
  uint64_t clock_epoch = 0;
  int64_t clock_sequence = 0;
  double clock_time = -1;
  int32_t has_sample = 1;
  ASSERT_EQ(
      sb_get_actor_observation_clock(source, actor, &clock_epoch,
                                     &clock_sequence, &clock_time, &has_sample),
      1);
  EXPECT_EQ(clock_epoch, next);
  EXPECT_EQ(clock_sequence, -1);
  EXPECT_DOUBLE_EQ(clock_time, 0);
  EXPECT_EQ(has_sample, 0);
  EXPECT_EQ(
      sb_apply_actor_observation_frame(source, actor, next, 0, 0, good, 1), 1);
  sb_destroy_session(session);
}

TEST(CApiSessionTest, ObservationModeIsMutuallyExclusiveWithNetworkRoles) {
  const sb_session_t local = sb_create_session();
  const sb_space_t local_space = sb_add_space(local);
  ASSERT_EQ(sb_enable_actor_observation_frames(local), 1);
  EXPECT_EQ(sb_host_space(local_space, 0), 0);
  EXPECT_EQ(sb_connect_space(local_space, "127.0.0.1", 1), 0);
  sb_destroy_session(local);

  const sb_session_t host_session = sb_create_session();
  const sb_space_t host_space = sb_add_space(host_session);
  ASSERT_NE(sb_host_space(host_space, 0), 0);
  EXPECT_EQ(sb_enable_actor_observation_frames(host_session), 0);
  sb_destroy_session(host_session);

  // A failed bounded localhost connect rolls its tentative client role back,
  // so local observation mode remains available without preserving stale role
  // state. A successful remote client remains covered by the shared role guard.
  const sb_session_t client_session = sb_create_session();
  const sb_space_t client_space = sb_add_space(client_session);
  (void)sb_connect_space(client_space, "127.0.0.1", 1);
  EXPECT_EQ(sb_enable_actor_observation_frames(client_session), 1);
  uint64_t epoch = 17;
  int64_t sequence = 18;
  double time = 19;
  int32_t has_sample = 20;
  EXPECT_EQ(sb_get_actor_observation_clock(client_space, 1, &epoch, &sequence,
                                           &time, &has_sample),
            0);
  EXPECT_EQ(epoch, 17u);
  EXPECT_EQ(sequence, 18);
  EXPECT_DOUBLE_EQ(time, 19);
  EXPECT_EQ(has_sample, 20);
  sb_destroy_session(client_session);
}

TEST(CApiE2ETest, Checked3DAnchorCarriesMappedHeightAndClearsWhenInactive) {
  g_last_num_targets = 0;
  g_aff_first_anchor.clear();
  g_aff_have_hand = false;
  sb_session_t session = sb_create_session();
  sb_space_t sa = sb_add_space(session);
  sb_space_t sb = sb_add_space(session);
  double polygon[] = {0, 0, 10, 0, 10, 10, 0, 10};
  sb_element_t room_a = sb_add_element(sa, 0);
  sb_element_t room_b = sb_add_element(sb, 0);
  sb_set_element_shared_id(sa, room_a, 901);
  sb_set_element_shared_id(sb, room_b, 901);
  sb_set_element_polygon(sa, room_a, SB_GEOMETRY_NAVIGABLE, polygon, 4);
  sb_set_element_polygon(sb, room_b, SB_GEOMETRY_NAVIGABLE, polygon, 4);
  sb_element_t support_a = sb_add_element(sa, room_a);
  sb_element_t support_b = sb_add_element(sb, room_b);
  sb_set_element_shared_id(sa, support_a, 902);
  sb_set_element_shared_id(sb, support_b, 902);
  double source_surface[] = {4, 4, .9, 6, 4, .9, 6, 6, .9, 4, 6, .9};
  double target_surface[] = {2, 6, .4, 4, 6, .4, 4, 8, .4, 2, 8, .4};
  sb_set_element_surface(sa, support_a, source_surface, 4, SB_GEOMETRY_SURFACE);
  sb_set_element_surface(sb, support_b, target_surface, 4, SB_GEOMETRY_SURFACE);
  sb_actor_t source = sb_create_actor(sa);
  sb_actor_t receiver = sb_create_actor(sb);
  PublishPose(sb, receiver, 0.0, 0.0, 0.0);
  sb_add_anchor(sa, source, "root", 0, 0, 0, NULL);
  sb_add_anchor(sb, receiver, "root", 0, 0, 0, NULL);
  ASSERT_EQ(sb_add_anchor_with_position_dimension(sa, source, "custom_body", .4,
                                                  0, .9, NULL, 3),
            1);
  ASSERT_EQ(sb_add_anchor_with_position_dimension(sb, receiver, "custom_body",
                                                  .4, 0, .9, NULL, 3),
            1);
  EXPECT_EQ(sb_add_anchor_with_position_dimension(sa, source, "custom_body", .4,
                                                  0, 0, NULL, 2),
            0);
  sb_set_actor_element(sa, source, room_a);
  sb_set_actor_element(sb, receiver, room_b);
  sb_bind(sa, source, "custom_body", SB_KERNEL_AFFORDANCE_POSE);
  sb_register_on_targets(sb, capture_affordance_targets, nullptr);
  // The anchor is engaged while it touches the surface (z = 0.9); any
  // height above it releases the target at once so the anchor follows the body.
  sb_set_anchor_transform(sa, source, "custom_body", 5, 4, 0.9, 0);
  sb_update_actor_state(sa, source, 5, 4, 0);
  ASSERT_TRUE(g_aff_have_hand);
  EXPECT_NE(g_aff_hand_target.has_z, 0);
  EXPECT_NEAR(g_aff_hand_target.position_z, .4, 1e-6);
  g_aff_have_hand = false;
  // Having crossed the surface (0.05 m below it) still touches it.
  sb_set_anchor_transform(sa, source, "custom_body", 5, 4, 0.85, 0);
  sb_update_actor_state(sa, source, 5, 4, 0);
  ASSERT_TRUE(g_aff_have_hand);
  EXPECT_NEAR(g_aff_hand_target.position_z, .4, 1e-6);
  g_aff_have_hand = false;
  // Lifted 1.1 m: released on that very update (no release dwell).
  sb_set_anchor_transform(sa, source, "custom_body", 5, 4, 2.0, 0);
  sb_update_actor_state(sa, source, 5, 4, 0);
  EXPECT_FALSE(g_aff_have_hand);
  g_aff_have_hand = false;
  ASSERT_EQ(sb_set_anchor_active(sa, source, "custom_body", 0), 1);
  sb_update_actor_state(sa, source, 5, 4, 0);
  EXPECT_FALSE(g_aff_have_hand);
  sb_destroy_session(session);
}

TEST(CApiE2ETest, TryWaitOnSyncHandle) {
  sb_session_t session = sb_create_session();
  sb_space_t sa = sb_add_space(session);
  sb_element_t room = sb_add_element(sa, 0);
  sb_set_element_shared_id(sa, room, 100);
  double poly[] = {0, 0, 10, 0, 10, 10, 0, 10};
  sb_set_element_polygon(sa, room, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_actor_t actor = sb_create_actor(sa);
  sb_add_anchor(sa, actor, "root", 0, 0, 0, NULL);
  sb_set_actor_element(sa, actor, room);

  sb_handle_t h = sb_update_actor_state(sa, actor, 1.0, 1.0, 0.0);
  EXPECT_TRUE(sb_try_wait(h, 1000));
  EXPECT_EQ(sb_get_error(h), nullptr);

  sb_destroy_session(session);
}

TEST(CApiE2ETest, SetUpdateInterval) {
  sb_session_t session = sb_create_session();
  sb_space_t sa = sb_add_space(session);

  // Should not crash; no observable side-effect from C API.
  sb_set_update_interval(sa, 0.05);

  sb_destroy_session(session);
}

// ---------------------------------------------------------------------------
// Edge case tests
// ---------------------------------------------------------------------------

TEST(CApiEdgeCaseTest, InvalidHandlesDoNotCrash) {
  // All these should return 0 or no-op, not crash
  sb_destroy_session(999);
  EXPECT_EQ(sb_add_space(999), 0);
  EXPECT_EQ(sb_add_element(999, 0), 0);
  EXPECT_EQ(sb_find_element_by_shared_id(999, 1), 0);
  EXPECT_EQ(sb_create_actor(999), 0);
  EXPECT_EQ(sb_find_actor_by_shared_id(999, 1), 0);
  sb_add_anchor(999, 1, "root", 0, 0, 0, NULL);
  sb_set_actor_element(999, 1, 1);
  sb_bind(999, 1, "root", SB_KERNEL_SPACING);
  EXPECT_FALSE(sb_poll_realization_targets(999, 1));
  sb_set_element_shared_id(999, 1, 100);
  sb_set_element_tag(999, 1, "tag");
  sb_set_element_polygon(999, 1, SB_GEOMETRY_UNSPECIFIED, nullptr, 0);
  sb_set_element_box(999, 1, SB_GEOMETRY_UNSPECIFIED, 0, 0, 0, 1, 1, 1);
  double m[16] = {};
  sb_set_element_transform(999, 1, m);
  {
    const sb_anchor_pose_t pose = RootPose(0.0, 0.0, 0.0);
    EXPECT_EQ(sb_publish_anchor_pose(999, 1, 0, &pose, 0.0, 1), -1);
  }
  sb_set_update_interval(999, 0.033);
  sb_register_on_targets(999, nullptr, nullptr);
}

TEST(CApiEdgeCaseTest, NullCallbackSafe) {
  sb_session_t s = sb_create_session();
  sb_space_t sp = sb_add_space(s);
  sb_register_on_targets(sp, nullptr, nullptr);
  // Should not crash on subsequent updates
  sb_destroy_session(s);
}

// ===========================================================================
// Integration tests: full hierarchy + cross-space propagation
// ===========================================================================

namespace {

// Row-major identity with translation (x, y, z=0).
void MakeTranslation(double* out, double x, double y) {
  // Row-major: row0={1,0,0,tx}, row1={0,1,0,ty}, row2={0,0,1,0}, row3={0,0,0,1}
  double m[16] = {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, 0, 0, 0, 0, 1};
  std::memcpy(out, m, sizeof(m));
}

struct TwoSpaceFixture {
  sb_session_t session;
  sb_space_t space_a, space_b;
  sb_actor_t actor_a, avatar_b;
  sb_element_t room_a, board_a, writable_a, table_a, tabletop_a;
  sb_element_t room_b, board_b, writable_b, table_b, tabletop_b;
};

TwoSpaceFixture SetupTwoSpaces(bool publish_start = true) {
  TwoSpaceFixture f;
  f.session = sb_create_session();
  f.space_a = sb_add_space(f.session);
  f.space_b = sb_add_space(f.session);

  // Containment boxes are ELEMENT-LOCAL (sb_set_element_polygon): each is centred on its element, which
  // the transform places. (A box written in world coordinates under a translated element lands at twice
  // the offset, outside the room, and is refused; the returns are asserted.)
  // --- Space A hierarchy (20x20 room) ---
  f.room_a = sb_add_element(f.space_a, 0);
  sb_set_element_shared_id(f.space_a, f.room_a, 100);
  double poly_a[] = {0, 0, 20, 0, 20, 20, 0, 20};
  EXPECT_EQ(SB_SCENE_OK, sb_set_element_polygon(f.space_a, f.room_a, SB_GEOMETRY_NAVIGABLE, poly_a, 4));

  f.board_a = sb_add_element(f.space_a, f.room_a);
  sb_set_element_shared_id(f.space_a, f.board_a, 200);
  double board_a_mat[16];
  MakeTranslation(board_a_mat, 5, 15);
  sb_set_element_transform(f.space_a, f.board_a, board_a_mat);
  double board_a_bnd[] = {-2, -2, 2, -2, 2, 2, -2, 2};  // element-local: the transform places it
  EXPECT_EQ(SB_SCENE_OK, sb_set_element_polygon(f.space_a, f.board_a, SB_GEOMETRY_CONTAINMENT,
                         board_a_bnd, 4));

  f.writable_a = sb_add_element(f.space_a, f.board_a);
  sb_set_element_shared_id(f.space_a, f.writable_a, 201);
  sb_set_element_tag(f.space_a, f.writable_a, "Writable");
  // Placed at its parent's origin: element transforms are parent-relative, so
  // re-applying board_a_mat would put it at twice the parent offset (outside the
  // room) -- no longer reachable nor within a near-field.

  f.table_a = sb_add_element(f.space_a, f.room_a);
  sb_set_element_shared_id(f.space_a, f.table_a, 300);
  double table_a_mat[16];
  MakeTranslation(table_a_mat, 15, 5);
  sb_set_element_transform(f.space_a, f.table_a, table_a_mat);
  double table_a_bnd[] = {-2, -2, 2, -2, 2, 2, -2, 2};  // element-local: the transform places it
  EXPECT_EQ(SB_SCENE_OK, sb_set_element_polygon(f.space_a, f.table_a, SB_GEOMETRY_CONTAINMENT,
                         table_a_bnd, 4));

  f.tabletop_a = sb_add_element(f.space_a, f.table_a);
  sb_set_element_shared_id(f.space_a, f.tabletop_a, 301);
  sb_set_element_tag(f.space_a, f.tabletop_a, "Tabletop");
  // Placed at its parent's origin: element transforms are parent-relative, so
  // re-applying table_a_mat would put it at twice the parent offset (outside the
  // room) -- no longer reachable nor within a near-field.

  // --- Space B hierarchy (12x12 room) ---
  f.room_b = sb_add_element(f.space_b, 0);
  sb_set_element_shared_id(f.space_b, f.room_b, 100);
  double poly_b[] = {0, 0, 12, 0, 12, 12, 0, 12};
  EXPECT_EQ(SB_SCENE_OK, sb_set_element_polygon(f.space_b, f.room_b, SB_GEOMETRY_NAVIGABLE, poly_b, 4));

  f.board_b = sb_add_element(f.space_b, f.room_b);
  sb_set_element_shared_id(f.space_b, f.board_b, 200);
  double board_b_mat[16];
  MakeTranslation(board_b_mat, 10, 3);
  sb_set_element_transform(f.space_b, f.board_b, board_b_mat);
  double board_b_bnd[] = {-2, -2, 2, -2, 2, 2, -2, 2};  // element-local: the transform places it
  EXPECT_EQ(SB_SCENE_OK, sb_set_element_polygon(f.space_b, f.board_b, SB_GEOMETRY_CONTAINMENT,
                         board_b_bnd, 4));

  f.writable_b = sb_add_element(f.space_b, f.board_b);
  sb_set_element_shared_id(f.space_b, f.writable_b, 201);
  sb_set_element_tag(f.space_b, f.writable_b, "Writable");
  // Placed at its parent's origin: element transforms are parent-relative, so
  // re-applying board_b_mat would put it at twice the parent offset (outside the
  // room) -- no longer reachable nor within a near-field.

  f.table_b = sb_add_element(f.space_b, f.room_b);
  sb_set_element_shared_id(f.space_b, f.table_b, 300);
  double table_b_mat[16];
  MakeTranslation(table_b_mat, 3, 10);
  sb_set_element_transform(f.space_b, f.table_b, table_b_mat);
  double table_b_bnd[] = {-2, -2, 2, -2, 2, 2, -2, 2};  // element-local: the transform places it
  EXPECT_EQ(SB_SCENE_OK, sb_set_element_polygon(f.space_b, f.table_b, SB_GEOMETRY_CONTAINMENT,
                         table_b_bnd, 4));

  f.tabletop_b = sb_add_element(f.space_b, f.table_b);
  sb_set_element_shared_id(f.space_b, f.tabletop_b, 301);
  sb_set_element_tag(f.space_b, f.tabletop_b, "Tabletop");
  // Placed at its parent's origin: element transforms are parent-relative, so
  // re-applying table_b_mat would put it at twice the parent offset (outside the
  // room) -- no longer reachable nor within a near-field.

  // --- Actor in Space A ---
  f.actor_a = sb_create_actor(f.space_a);
  sb_add_anchor(f.space_a, f.actor_a, "root", 0, 0, 0, NULL);
  sb_add_anchor(f.space_a, f.actor_a, "chest", 0, 0, 1.2, NULL);
  sb_set_actor_element(f.space_a, f.actor_a, f.room_a);

  sb_bind(f.space_a, f.actor_a, "root", SB_KERNEL_SPACING);
  sb_bind(f.space_a, f.actor_a, "chest", SB_KERNEL_FACING);
  sb_bind_affordance(f.space_a, f.actor_a, "root", "Tabletop");

  // --- Avatar in Space B ---
  f.avatar_b = sb_create_actor(f.space_b);
  if (publish_start) PublishPose(f.space_b, f.avatar_b, 0.0, 0.0, 0.0);
  sb_add_anchor(f.space_b, f.avatar_b, "root", 0, 0, 0, NULL);
  sb_add_anchor(f.space_b, f.avatar_b, "chest", 0, 0, 1.2, NULL);
  sb_set_actor_element(f.space_b, f.avatar_b, f.room_b);

  return f;
}

// Callback accumulator used by integration tests.
struct TargetAccumulator {
  int callback_count = 0;
  std::vector<sb_realization_target_t> targets;
  sb_space_t last_space = 0;
  sb_actor_t last_actor = 0;

  void Reset() {
    callback_count = 0;
    targets.clear();
    last_space = 0;
    last_actor = 0;
  }
};

static void AccumulateTargets(sb_space_t space, sb_actor_t actor,
                              const sb_realization_target_t* targets,
                              int64_t num_targets, void* user_data) {
  auto* acc = static_cast<TargetAccumulator*>(user_data);
  acc->callback_count++;
  acc->last_space = space;
  acc->last_actor = actor;
  acc->targets.clear();
  for (int64_t i = 0; i < num_targets; ++i) {
    // Copy struct; anchor_name pointer is only valid during callback, so we
    // don't dereference it after the callback returns (we check it in-scope).
    acc->targets.push_back(targets[i]);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1: Full hierarchy setup + cross-space propagation
// ---------------------------------------------------------------------------

TEST(CApiIntegrationTest, FullHierarchyCrossSpacePropagation) {
  auto f = SetupTwoSpaces();

  TargetAccumulator acc;
  sb_register_on_targets(f.space_b, AccumulateTargets, &acc);

  // Update actor in Space A at center of room.
  sb_update_actor_state(f.space_a, f.actor_a, 10.0, 10.0, 0.0);

  // Callback should have fired on Space B.
  EXPECT_GT(acc.callback_count, 0)
      << "Target callback should fire on Space B after actor update in A";
  EXPECT_EQ(acc.last_space, f.space_b);
  EXPECT_GT(static_cast<int>(acc.targets.size()), 0)
      << "Should receive at least one realization target";

  // All targets must have finite positions.
  for (const auto& t : acc.targets) {
    EXPECT_TRUE(std::isfinite(t.position_x)) << "position_x must be finite";
    EXPECT_TRUE(std::isfinite(t.position_y)) << "position_y must be finite";
    EXPECT_TRUE(std::isfinite(t.yaw)) << "yaw must be finite";
  }

  sb_destroy_session(f.session);
}

TEST(CApiActorTest, CheckedAnchorDimensionRejectsInvalidInputWithoutMutation) {
  sb_session_t session = sb_create_session();
  sb_space_t space = sb_add_space(session);
  sb_actor_t actor = sb_create_actor(space);
  ASSERT_NE(actor, 0);
  EXPECT_EQ(
      sb_add_anchor_with_position_dimension(0, actor, "bad", 0, 0, 0, NULL, 3),
      0);
  EXPECT_EQ(
      sb_add_anchor_with_position_dimension(space, 0, "bad", 0, 0, 0, NULL, 3),
      0);
  EXPECT_EQ(sb_add_anchor_with_position_dimension(space, actor, NULL, 0, 0, 0,
                                                  NULL, 3),
            0);
  EXPECT_EQ(sb_add_anchor_with_position_dimension(space, actor, "body", 0, 0, 1,
                                                  NULL, 3),
            1);
  EXPECT_EQ(sb_add_anchor_with_position_dimension(space, actor, "body", 0, 0, 0,
                                                  NULL, 2),
            0);
  EXPECT_EQ(
      sb_add_anchor_with_position_dimension(space, actor, "", 0, 0, 0, NULL, 3),
      0);
  EXPECT_EQ(sb_add_anchor_with_position_dimension(space, actor, "nan",
                                                  std::nan(""), 0, 0, NULL, 3),
            0);
  EXPECT_EQ(sb_add_anchor_with_position_dimension(
                space, actor, "inf", 0, std::numeric_limits<double>::infinity(),
                0, NULL, 3),
            0);
  EXPECT_EQ(sb_add_anchor_with_position_dimension(space, actor, "bad", 0, 0, 0,
                                                  NULL, 4),
            0);
  EXPECT_EQ(sb_add_anchor_with_position_dimension(space, actor, "self", 0, 0, 0,
                                                  "self", 3),
            0);
  // The accepted 3D anchor remains bindable; rejected requests did not replace
  // it.
  EXPECT_EQ(sb_set_anchor_active(space, actor, "body", 0), 1);
  sb_destroy_session(session);
}

TEST(CApiScopedBindTest, SpacingIsTheContainmentTreeOnly) {
  auto f = SetupTwoSpaces();
  // The actor stands in the room, outside the table. Spacing is the
  // containment tree; the table's Tabletop child is not added to that set.
  sb_actor_t source = sb_create_actor(f.space_a);
  sb_add_anchor(f.space_a, source, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(f.space_a, source, f.room_a);
  ASSERT_EQ(sb_bind(f.space_a, source, "root", SB_KERNEL_SPACING),
            1);
  sb_update_actor_state(f.space_a, source, 10.0, 10.0, 0.0);
  sb_relation_state_t states[4];
  // The "Tabletop" tag no longer narrows Spacing (it
  // used to leave nothing), so the containment tree's own referents are present. The
  // named subtree is still not pulled in: no row reaches the table's Tabletop child.
  const int32_t n_states = sb_get_relation_states(f.space_a, source, states, 4);
  ASSERT_GE(n_states, 1);
  for (int32_t i = 0; i < n_states && i < 4; ++i)
    EXPECT_NE(states[i].referent_element_id, f.tabletop_a);
  sb_destroy_session(f.session);
}

// ---------------------------------------------------------------------------
// Test 2: Multi-step trajectory through C API
// ---------------------------------------------------------------------------

TEST(CApiIntegrationTest, MultiStepTrajectory) {
  auto f = SetupTwoSpaces();

  TargetAccumulator acc;
  sb_register_on_targets(f.space_b, AccumulateTargets, &acc);

  // Walk 5 steps from (2,2) toward Board_A at (5,15).
  double start_x = 2.0, start_y = 2.0;
  double goal_x = 5.0, goal_y = 13.0;  // Just outside board boundary.
  for (int step = 0; step < 5; ++step) {
    double t = static_cast<double>(step + 1) / 5.0;
    double x = start_x + t * (goal_x - start_x);
    double y = start_y + t * (goal_y - start_y);
    double yaw = std::atan2(goal_y - start_y, goal_x - start_x);

    acc.Reset();
    sb_update_actor_state(f.space_a, f.actor_a, x, y, yaw);

    EXPECT_GT(acc.callback_count, 0)
        << "Step " << step << ": callback must fire";
    // The avatar is published at the room corner and this test does not place
    // it, so the sample is not a free pose and the spacing root is withheld.
    // A moving source does not invent that root. An empty callback is the
    // hold release on a step that also admits no anchor.
    for (const auto& tgt : acc.targets) {
      EXPECT_GE(tgt.arrival_time, 0.0) << "Step " << step;
      EXPECT_TRUE(std::isfinite(tgt.position_x)) << "Step " << step;
      EXPECT_TRUE(std::isfinite(tgt.position_y)) << "Step " << step;
    }
  }

  sb_destroy_session(f.session);
}

// ---------------------------------------------------------------------------
// Test 3: Affordance activation through C API
// ---------------------------------------------------------------------------

TEST(CApiIntegrationTest, AffordanceActivation) {
  auto f = SetupTwoSpaces();

  TargetAccumulator acc_far;
  sb_register_on_targets(f.space_b, AccumulateTargets, &acc_far);

  // First: actor far from the table at (2, 2).
  sb_update_actor_state(f.space_a, f.actor_a, 2.0, 2.0, 0.0);
  ASSERT_GT(acc_far.callback_count, 0) << "Callback should fire for far pos";
  ASSERT_GT(static_cast<int>(acc_far.targets.size()), 0);
  double far_x = acc_far.targets[0].position_x;
  double far_y = acc_far.targets[0].position_y;

  // Second: actor right next to the Table_A boundary (near Tabletop
  // affordance). Table_A is at (15,5) with boundary {13,3, 17,3, 17,7, 13,7}.
  // The affordance binding (SB_KERNEL_AFFORDANCE_POSE for "Tabletop") should
  // influence the solved position when the actor is close to the table.
  TargetAccumulator acc_near;
  sb_register_on_targets(f.space_b, AccumulateTargets, &acc_near);
  sb_update_actor_state(f.space_a, f.actor_a, 13.0, 5.0, 0.0);

  ASSERT_GT(acc_near.callback_count, 0)
      << "Callback should fire near affordance";
  ASSERT_GT(static_cast<int>(acc_near.targets.size()), 0);

  // The solved target should have a valid finite position.
  for (const auto& tgt : acc_near.targets) {
    EXPECT_TRUE(std::isfinite(tgt.position_x));
    EXPECT_TRUE(std::isfinite(tgt.position_y));
    EXPECT_TRUE(std::isfinite(tgt.yaw));
  }

  // The near-affordance position in Space B should differ from the far
  // position, indicating the solver incorporated the affordance proximity.
  double near_x = acc_near.targets[0].position_x;
  double near_y = acc_near.targets[0].position_y;
  double dist = std::hypot(near_x - far_x, near_y - far_y);
  EXPECT_GT(dist, 0.1)
      << "Solved position should change when actor approaches affordance; "
      << "far=(" << far_x << "," << far_y << "), near=(" << near_x << ","
      << near_y << ")";

  sb_destroy_session(f.session);
}

// ---------------------------------------------------------------------------
// Test 4: Multiple actors in same session
// ---------------------------------------------------------------------------

// Callback for multi-actor test (must be a plain function, not a lambda with
// captures, because it is passed as a C function pointer).
struct MultiActorAcc {
  int count_actor1 = 0;
  int count_actor2 = 0;
  sb_actor_t avatar1 = 0;
  sb_actor_t avatar2 = 0;
};

static void MultiActorCallback(sb_space_t /*space*/, sb_actor_t actor,
                               const sb_realization_target_t* /*targets*/,
                               int64_t num_targets, void* user_data) {
  auto* m = static_cast<MultiActorAcc*>(user_data);
  if (num_targets > 0) {
    if (actor == m->avatar1) m->count_actor1++;
    if (actor == m->avatar2) m->count_actor2++;
  }
}

TEST(CApiIntegrationTest, MultipleActorsIndependent) {
  auto f = SetupTwoSpaces();

  // Actor 1 is already in room_a (shared_id=100), avatar_b in room_b (100).
  // Create a second actor in Space A on board_a (shared_id=200) so it gets
  // a distinct cross-space identity.
  sb_actor_t actor2 = sb_create_actor(f.space_a);
  sb_add_anchor(f.space_a, actor2, "root", 0, 0, 0, NULL);
  sb_set_actor_element(f.space_a, actor2, f.board_a);
  sb_bind(f.space_a, actor2, "root", SB_KERNEL_SPACING);

  // Avatar 2 in Space B on board_b (shared_id=200) to match actor2.
  sb_actor_t avatar2 = sb_create_actor(f.space_b);
  PublishPose(f.space_b, avatar2, 0.0, 0.0, 0.0);
  sb_add_anchor(f.space_b, avatar2, "root", 0, 0, 0, NULL);
  sb_set_actor_element(f.space_b, avatar2, f.board_b);

  MultiActorAcc multi;
  multi.avatar1 = f.avatar_b;
  multi.avatar2 = avatar2;

  sb_register_on_targets(f.space_b, MultiActorCallback, &multi);

  // Update actor 1 in Space A (room-level, shared_id=100).
  sb_update_actor_state(f.space_a, f.actor_a, 10.0, 10.0, 0.0);
  EXPECT_GT(multi.count_actor1, 0) << "Actor 1 callback should fire on Space B";

  // Update actor 2 in Space A (board-level, shared_id=200).
  // Position must be inside room_a polygon so the solver can find a solution.
  sb_update_actor_state(f.space_a, actor2, 5.0, 14.0, 1.57);
  EXPECT_GT(multi.count_actor2, 0) << "Actor 2 callback should fire on Space B";

  sb_destroy_session(f.session);
}

// ---------------------------------------------------------------------------
// Test 5: Apply state write-back (warm start)
// ---------------------------------------------------------------------------

TEST(CApiIntegrationTest, ApplyStateWriteBack) {
  auto f = SetupTwoSpaces();

  TargetAccumulator acc;
  sb_register_on_targets(f.space_b, AccumulateTargets, &acc);

  // Initial update: actor at (10, 10).
  sb_update_actor_state(f.space_a, f.actor_a, 10.0, 10.0, 0.0);
  ASSERT_GT(acc.callback_count, 0) << "First update must produce targets";

  // Apply avatar state (write-back) using one of the received target positions.
  // This simulates the client applying the recommended pose.
  double wb_x = 6.0, wb_y = 6.0, wb_yaw = 0.0;
  if (!acc.targets.empty()) {
    wb_x = acc.targets[0].position_x;
    wb_y = acc.targets[0].position_y;
    wb_yaw = acc.targets[0].yaw;
  }
  EXPECT_EQ(PublishPose(f.space_b, f.avatar_b, wb_x, wb_y, wb_yaw), 0);

  // Second update: move actor slightly.
  acc.Reset();
  sb_update_actor_state(f.space_a, f.actor_a, 10.5, 10.5, 0.1);
  EXPECT_GT(acc.callback_count, 0)
      << "Second update should still produce targets after write-back";
  EXPECT_GT(static_cast<int>(acc.targets.size()), 0);

  // Verify targets are still valid after warm-start.
  for (const auto& tgt : acc.targets) {
    EXPECT_TRUE(std::isfinite(tgt.position_x));
    EXPECT_TRUE(std::isfinite(tgt.position_y));
    EXPECT_TRUE(std::isfinite(tgt.yaw));
    EXPECT_TRUE(std::isfinite(tgt.arrival_time));
  }

  sb_destroy_session(f.session);
}

// ---------------------------------------------------------------------------
// Test 6: Network — host space via C API
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Element lifecycle: share / unshare
// ---------------------------------------------------------------------------

namespace {
sb_element_t g_removed_added_b = 0;
sb_element_t g_removed_added_c = 0;
std::vector<sb_element_t> g_removed_b;
std::vector<sb_element_t> g_removed_c;

void CaptureRemovalAddedB(sb_space_t, sb_element_t local, int64_t, const char*,
                          const void*, int64_t, void*) {
  g_removed_added_b = local;
}
void CaptureRemovalAddedC(sb_space_t, sb_element_t local, int64_t, const char*,
                          const void*, int64_t, void*) {
  g_removed_added_c = local;
}
void CaptureRemovedB(sb_space_t, sb_element_t local, void*) {
  g_removed_b.push_back(local);
}
void CaptureRemovedC(sb_space_t, sb_element_t local, void*) {
  g_removed_c.push_back(local);
}
}  // namespace

TEST(CApiElementLifecycleTest, ShareAndUnshareElement) {
  sb_session_t s = sb_create_session();
  sb_space_t sa = sb_add_space(s);
  sb_space_t sb = sb_add_space(s);

  // Setup rooms
  sb_element_t room_a = sb_add_element(sa, 0);
  sb_set_element_shared_id(sa, room_a, 100);
  sb_element_t room_b = sb_add_element(sb, 0);
  sb_set_element_shared_id(sb, room_b, 100);

  // Add board to space_a and share
  sb_element_t board = sb_add_element(sa, room_a);
  sb_set_element_shared_id(sa, board, 200);
  sb_share_element(sa, board, nullptr, 0);  // no user_data payload

  // space_b should now have an element with shared_id 200
  // (We can't directly query, but it shouldn't crash)

  sb_unshare_element(sa, board);
  sb_destroy_session(s);
}

// ---------------------------------------------------------------------------
// Actor state: get position and yaw
// ---------------------------------------------------------------------------

TEST(CApiActorStateTest, GetActorState) {
  sb_session_t s = sb_create_session();
  sb_space_t sp = sb_add_space(s);
  sb_element_t room = sb_add_element(sp, 0);
  double poly[] = {0, 0, 20, 0, 20, 20, 0, 20};
  sb_set_element_polygon(sp, room, SB_GEOMETRY_NAVIGABLE, poly, 4);

  sb_actor_t actor = sb_create_actor(sp);
  sb_add_anchor(sp, actor, "root", 0, 0, 0, NULL);
  sb_set_actor_element(sp, actor, room);

  sb_update_actor_state(sp, actor, 5.0, 10.0, 1.57);

  sb_actor_state_t state = sb_get_actor_state(sp, actor);
  // Position should be approximately (5, 10)
  // Allow some tolerance since state goes through transform
  EXPECT_NEAR(state.position_x, 5.0, 0.1);
  EXPECT_NEAR(state.position_y, 10.0, 0.1);

  sb_destroy_session(s);
}

TEST(CApiActorStateTest, GetActorStateInvalidHandle) {
  // Should return zeroed state, not crash
  sb_actor_state_t state = sb_get_actor_state(999, 999);
  EXPECT_EQ(state.position_x, 0.0);
  EXPECT_EQ(state.position_y, 0.0);
  EXPECT_EQ(state.yaw, 0.0);
}

// ---------------------------------------------------------------------------
// Element anchor
// ---------------------------------------------------------------------------

TEST(CApiElementAnchorTest, AddAnchor) {
  sb_session_t s = sb_create_session();
  sb_space_t sp = sb_add_space(s);
  sb_element_t room = sb_add_element(sp, 0);

  // Col-major 4x4 identity with z-translation of 1.5
  // col0={1,0,0,0}, col1={0,1,0,0}, col2={0,0,1,0}, col3={0,0,1.5,1}
  double anchor_matrix[16] = {1, 0, 0,   0,   // col 0
                              0, 1, 0,   0,   // col 1
                              0, 0, 1,   0,   // col 2
                              0, 0, 1.5, 1};  // col 3
  sb_add_element_anchor(sp, room, "surface_center", anchor_matrix);

  // Should not crash
  sb_destroy_session(s);
}

TEST(CApiElementAnchorTest, AddAnchorInvalidHandle) {
  double m[16] = {};
  // Should not crash
  sb_add_element_anchor(999, 999, "anchor", m);
}

// ---------------------------------------------------------------------------
// Network test
// ---------------------------------------------------------------------------

TEST(CApiNetworkTest, HostAndDestroy) {
  sb_session_t s = sb_create_session();

  sb_space_t host = sb_add_space(s);
  sb_element_t room = sb_add_element(host, 0);
  sb_set_element_shared_id(host, room, 100);
  double poly[] = {0, 0, 20, 0, 20, 20, 0, 20};
  sb_set_element_polygon(host, room, SB_GEOMETRY_NAVIGABLE, poly, 4);

  sb_actor_t actor = sb_create_actor(host);
  sb_add_anchor(host, actor, "root", 0, 0, 0, NULL);
  sb_set_actor_element(host, actor, room);
  sb_bind(host, actor, "root", SB_KERNEL_SPACING);

  // Host on port 0 (auto-assign). Should return valid handle.
  sb_handle_t h = sb_host_space(host, 0);
  EXPECT_NE(h, 0) << "sb_host_space should return a valid handle";
  EXPECT_TRUE(sb_try_wait(h, 1000));

  // Cleanup should not crash (stops receiver internally).
  sb_destroy_session(s);
}

// Repro + regression test for the hub single-slot-callback
// clobber bug. Root cause: sb_add_space() wires every space in a session to
// that ONE session's SpaceHub (LocalContext::GetHub()), and sb_host_space()/
// sb_connect_space() each install their own element/relation callback pair
// onto that single hub via SetElementEventCallback/SetRelationUpdateCallback
// (space/space_hub.h -- plain std::function assignment, REPLACE not
// additive). Calling sb_host_space() on space A and then sb_connect_space()
// on space B in the SAME session therefore silently overwrote A's host
// callbacks with B's client callbacks: A's HubReceiver kept running (its
// gRPC server thread stays up) but the hub no longer called into it, so any
// OTHER remote client that connected to A's host port would stop having its
// element/relation events forwarded -- a silent, hard-to-diagnose black hole
// rather than a crash. Fixed by giving SessionContext a HubRole latch (see
// the comment on HubRole/SessionContext in spatialblend_c.cc): the session's
// hub role is claimed by whichever of sb_host_space/sb_connect_space
// succeeds first, and every subsequent host/connect call on that same
// session -- including a second call to the SAME function -- now fails
// fast (returns 0) instead of clobbering. "One sb_session_t may EITHER host
// EITHER connect, never both, never twice" is the resulting documented API
// contract; a process that needs both roles must use two sb_session_t (see
// spatialblend/sim/harness/unity_twin.cc's Transport::kGrpcTwoSessions).
TEST(CApiNetworkTest, HostThenConnectSameSessionRejected) {
  sb_session_t s = sb_create_session();
  sb_space_t space_a = sb_add_space(s);
  sb_space_t space_b = sb_add_space(s);
  ASSERT_NE(space_a, 0);
  ASSERT_NE(space_b, 0);

  // Host space A.
  sb_handle_t host_handle = sb_host_space(space_a, 0);
  ASSERT_NE(host_handle, 0) << "sb_host_space should succeed";
  ASSERT_TRUE(sb_try_wait(host_handle, 1000));

  // Attempting to ALSO connect space B (same session) must now be an
  // explicit, documented error (0), not a silent clobber of A's host
  // callbacks and not a crash.
  sb_handle_t connect_handle = sb_connect_space(space_b, "127.0.0.1", 1);
  EXPECT_EQ(connect_handle, 0)
      << "sb_connect_space on a session that already hosts must be rejected "
         "(one session = one hub role)";

  // A second sb_host_space call (even on the SAME space, even with the same
  // role) must likewise be rejected -- the session already claimed the host
  // role once.
  sb_handle_t host_again = sb_host_space(space_a, 0);
  EXPECT_EQ(host_again, 0)
      << "a second sb_host_space call on an already-hosting session must be "
         "rejected";

  sb_destroy_session(s);
}

TEST(CApiNetworkTest, ConnectThenHostSameSessionRejected) {
  sb_session_t s = sb_create_session();
  sb_space_t space_a = sb_add_space(s);
  sb_space_t space_b = sb_add_space(s);
  ASSERT_NE(space_a, 0);
  ASSERT_NE(space_b, 0);

  // Connect space A to a nonexistent/unreachable host -- expected to fail
  // fast at the RemoteHub::Create bootstrap step, but that failure must
  // release the claimed "client" role again (see the rollback in
  // sb_connect_space) so a LEGITIMATE subsequent host/connect call is not
  // permanently blocked by a failed attempt.
  sb_handle_t connect_handle = sb_connect_space(space_a, "127.0.0.1", 1);
  EXPECT_EQ(connect_handle, 0)
      << "connecting to an unreachable port should fail";

  // Since that failed attempt released the role, hosting space B in the
  // same session should now succeed.
  sb_handle_t host_handle = sb_host_space(space_b, 0);
  EXPECT_NE(host_handle, 0)
      << "a failed connect attempt must release the session's hub-role "
         "claim so a subsequent host call can still succeed";
  EXPECT_TRUE(sb_try_wait(host_handle, 1000));

  sb_destroy_session(s);
}

// ---------------------------------------------------------------------------
// Anchor parenting + standard anchor set
// ---------------------------------------------------------------------------

TEST(CApiAnchorTest, StandardAnchorsAndParentIntegrateWithRealization) {
  auto f = SetupTwoSpaces();

  // New API surface: register the canonical body set on both actors and add a
  // custom anchor parented to a standard anchor.
  sb_add_standard_anchors(f.space_a, f.actor_a);
  sb_add_standard_anchors(f.space_b, f.avatar_b);
  sb_add_anchor(f.space_a, f.actor_a, "custom", 0.1, 0.2, 0.3, "chest");

  // Bind an AffordancePose relation on a standard-set anchor (hand_l).
  sb_bind_affordance(f.space_a, f.actor_a, "hand_l", "Tabletop");

  TargetAccumulator acc;
  sb_register_on_targets(f.space_b, AccumulateTargets, &acc);

  // The realization pipeline must still produce targets with standard anchors.
  sb_update_actor_state(f.space_a, f.actor_a, 10.0, 10.0, 0.0);
  EXPECT_GT(acc.callback_count, 0)
      << "Realization should fire after standard anchors are registered";
  EXPECT_GT(static_cast<int>(acc.targets.size()), 0);

  // Authoritative actor state round-trips through sb_get_actor_state.
  sb_actor_state_t st = sb_get_actor_state(f.space_a, f.actor_a);
  EXPECT_DOUBLE_EQ(st.position_x, 10.0);
  EXPECT_DOUBLE_EQ(st.position_y, 10.0);

  sb_destroy_session(f.session);
}

// ---------------------------------------------------------------------------
// Surface / polytope / sub-region setters
// ---------------------------------------------------------------------------

TEST(CApiGeometryTest, SetElementSurfaceAndSubRegion) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  sb_element_t elem = sb_add_element(space, 0);

  // Planar tabletop surface at z=0.8.
  const double pts[] = {0, 0, 0.8, 1, 0, 0.8, 1, 1, 0.8, 0, 1, 0.8};
  sb_set_element_surface(space, elem, pts, 4, SB_GEOMETRY_SURFACE);

  // Named/tagged sub-region patch (identity transform).
  const double patch[] = {0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9};
  sb_add_sub_region(space, elem, "seat_patch", "Sittable", patch, 4, NULL);

  // A sub-region on an element without a spatial definition is a safe no-op.
  sb_element_t bare = sb_add_element(space, 0);
  sb_add_sub_region(space, bare, "x", "y", patch, 4, NULL);

  // Row-major transform variant must also be accepted.
  const double xform[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  sb_add_sub_region(space, elem, "rim", "Rim", patch, 4, xform);

  sb_destroy_session(s);
  SUCCEED();
}

// ---------------------------------------------------------------------------
// Relation lifecycle event callback
// ---------------------------------------------------------------------------

namespace {
struct PhaseCollector {
  std::vector<sb_relation_phase_t> to_phases;
  std::string first_anchor;
};
void CollectPhases(const sb_relation_event_t* e, void* ud) {
  auto* c = static_cast<PhaseCollector*>(ud);
  if (c->to_phases.empty() && e->anchor_name) c->first_anchor = e->anchor_name;
  c->to_phases.push_back(e->to_phase);
}
}  // namespace

TEST(CApiRelationEventTest, ApproachEmitsLifecycle) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);

  sb_element_t room = sb_add_element(space, 0);
  double room_poly[] = {0, 0, 10, 0, 10, 10, 0, 10};
  sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, room_poly, 4);

  sb_element_t seat = sb_add_element(space, room);
  sb_set_element_tag(space, seat, "Seat");
  // MA-1 / SP-D7-03: an affordance with no shared id is a referent of no relation, leaf scope included.
  sb_set_element_shared_id(space, seat, 7001);
  double seat_poly[] = {0, 0, 1, 0, 1, 1, 0, 1};
  sb_set_element_polygon(space, seat, SB_GEOMETRY_CONTAINMENT, seat_poly, 4);

  sb_actor_t actor = sb_create_actor(space);
  sb_add_anchor(space, actor, "root", 0, 0, 0, NULL);
  sb_set_actor_element(space, actor, room);
  sb_bind_affordance(space, actor, "root", "Seat");

  PhaseCollector coll;
  sb_register_on_relation_event(space, CollectPhases, &coll);

  for (int i = 0; i <= 10; ++i) {
    double a = i / 10.0;
    sb_update_actor_state(space, actor, 5 * (1 - a) + 0.5 * a,
                          5 * (1 - a) + 0.5 * a, 0.0);
  }

  ASSERT_FALSE(coll.to_phases.empty());
  EXPECT_EQ(coll.to_phases.front(), SB_PHASE_ACTIVE);
  EXPECT_EQ(coll.to_phases.back(), SB_PHASE_CONTACT);
  EXPECT_EQ(coll.first_anchor, "root");

  sb_destroy_session(s);
}

// ---------------------------------------------------------------------------
// Embodied-pose sampling port and receiver proposal (ABI 3). Replaces the
// opt-in core-side avatar pacing tests (sb_set_avatar_controller*,
// sb_tick_avatar, sb_get_avatar_mediator_state): core no longer paces; the
// DRIVER owns the pose and applies each proposal within its own bounds (speed,
// turn rate, teleport, lock: driver-level tests).
// ---------------------------------------------------------------------------

namespace {
using spatialblend_c_test::ProposalView;
using spatialblend_c_test::ReadProposals;

ProposalView ReadProposalOrFail(sb_space_t space, sb_actor_t actor) {
  ProposalView p = ReadProposals(space, actor);
  EXPECT_EQ(p.rc, 0);
  return p;
}
}  // namespace

TEST(CApiProposalTest, NoSampleIsExplicitThenTheNextSourceUpdateSolves) {
  // ABI 4: per-anchor ports and one proposal per relation-bearing anchor.
  EXPECT_EQ(sb_abi_version(), 4u);
  EXPECT_EQ(sb_sizeof(SB_STRUCT_ANCHOR_POSE), sizeof(sb_anchor_pose_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_ANCHOR_PROPOSAL), sizeof(sb_anchor_proposal_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_PROPOSAL_SET), sizeof(sb_proposal_set_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_ACTOR_EVENT), sizeof(sb_actor_event_t));
  auto f = SetupTwoSpaces();
  // SetupTwoSpaces published a start pose; a fresh actor has none.
  sb_actor_t bare = sb_create_actor(f.space_b);
  ProposalView none = ReadProposalOrFail(f.space_b, bare);
  EXPECT_EQ(none.status, SB_PROPOSAL_NONE);
  EXPECT_EQ(none.revision, 0u);

  sb_update_actor_state(f.space_a, f.actor_a, 10.0, 10.0, 0.0);
  ProposalView p = ReadProposalOrFail(f.space_b, f.avatar_b);
  EXPECT_TRUE(p.status == SB_PROPOSAL_EXACT || p.status == SB_PROPOSAL_INEXACT);
  EXPECT_EQ(p.sample_valid, 1);
  EXPECT_EQ(p.sample_sequence, 1u);
  EXPECT_GE(p.sample_age_s, 0.0);
  EXPECT_TRUE(p.cause & SB_PROPOSAL_CAUSE_FIRST_PROPOSAL);
  EXPECT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.yaw));
  EXPECT_EQ(p.root.anchor_id, 0u);
  EXPECT_NE(p.root.flags & SB_ANCHOR_PROPOSAL_FLAG_SAMPLED, 0u);
  const uint64_t first_revision = p.revision;

  EXPECT_EQ(PublishPose(f.space_b, f.avatar_b, p.x, p.y, p.yaw), 0);
  // This diagonal step recedes from both root Spacing referents: the new sample
  // alone must not replace the last publication or its captured sample identity.
  sb_update_actor_state(f.space_a, f.actor_a, 10.5, 10.5, 0.1);
  const auto receding = ReadProposalOrFail(f.space_b, f.avatar_b);
  EXPECT_EQ(receding.revision, first_revision);
  EXPECT_EQ(receding.sample_sequence, p.sample_sequence);
  EXPECT_DOUBLE_EQ(receding.x, p.x);
  EXPECT_DOUBLE_EQ(receding.y, p.y);

  // A closing source step toward the table requests a new proposal from the
  // latest sample. A source update alone is not a publication.
  sb_update_actor_state(f.space_a, f.actor_a, 10.5, 9.5, 0.1);
  ProposalView q = ReadProposalOrFail(f.space_b, f.avatar_b);
  EXPECT_GT(q.revision, first_revision);
  EXPECT_EQ(q.sample_sequence, p.sample_sequence + 1);
  EXPECT_FALSE(q.cause & SB_PROPOSAL_CAUSE_FIRST_PROPOSAL);

  // Bad arguments.
  sb_proposal_set_t wrong_set{};
  wrong_set.struct_size = 1;
  sb_anchor_proposal_t one{};
  one.struct_size = sizeof(one);
  EXPECT_EQ(sb_read_proposals(f.space_b, f.avatar_b, &wrong_set, &one, 1), -1);
  sb_proposal_set_t set{};
  set.struct_size = sizeof(set);
  EXPECT_EQ(sb_read_proposals(f.space_b, 987654, &set, &one, 1), -1);
  one.struct_size = 1;  // every array element carries its size
  EXPECT_EQ(sb_read_proposals(f.space_b, f.avatar_b, &set, &one, 1), -1);
  // A short buffer is not an error: the set reports the total.
  EXPECT_EQ(sb_read_proposals(f.space_b, f.avatar_b, &set, nullptr, 0), 0);
  EXPECT_EQ(set.anchor_count, 0u);
  EXPECT_GE(set.total_anchor_count, 1u);
  sb_destroy_session(f.session);
}

// ABI 3.1 frame rules: sb_set_element_polygon is element-local (a root room is
// world; NAVIGABLE is the world-baked exception), so a polygon follows the
// element's transform.
TEST(CApiElementTest, PolygonIsElementLocal) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  sb_element_t room = sb_add_element(space, 0);
  double room_poly[] = {0, 0, 20, 0, 20, 20, 0, 20};
  ASSERT_EQ(sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, room_poly, 4),
            SB_SCENE_OK);
  sb_element_t a = sb_add_element(space, room);
  sb_element_t b = sb_add_element(space, room);
  const double local[] = {-1, -1, 1, -1, 1, 1, -1, 1};
  double at_a[16] = {1, 0, 0, 5, 0, 1, 0, 5, 0, 0, 1, 0, 0, 0, 0, 1};
  double at_b[16] = {1, 0, 0, 5, 0, 1, 0, 5, 0, 0, 1, 0, 0, 0, 0, 1};
  ASSERT_EQ(sb_set_element_transform(space, a, at_a), SB_SCENE_OK);
  ASSERT_EQ(sb_set_element_transform(space, b, at_b), SB_SCENE_OK);
  ASSERT_EQ(sb_set_element_polygon(space, a, SB_GEOMETRY_CONTAINMENT, local, 4),
            SB_SCENE_OK);
  // The same local polygon on an element at the SAME place overlaps; moved clear
  // it is accepted: the polygon followed the element's frame.
  EXPECT_EQ(sb_set_element_polygon(space, b, SB_GEOMETRY_CONTAINMENT, local, 4),
            SB_SCENE_CONTAINMENT_OVERLAP);
  at_b[3] = 12;
  ASSERT_EQ(sb_set_element_transform(space, b, at_b), SB_SCENE_OK);
  EXPECT_EQ(sb_set_element_polygon(space, b, SB_GEOMETRY_CONTAINMENT, local, 4),
            SB_SCENE_OK);
  sb_destroy_session(s);
}

// ABI 3.1: the source's pitch / roll ride the update to the receiver's root
// proposal as a pass-through (its orientation); the facing yaw stays the solved
// yaw; a proposal with no authored yaw says so.
TEST(CApiProposalTest, OrientationPassThroughAndYawUnspecifiedFlag) {
  EXPECT_EQ(sb_abi_version(), 4u);
  EXPECT_EQ(sb_abi_minor_version(), 14u);
  EXPECT_EQ(sb_sizeof(SB_STRUCT_ORIENTATION), sizeof(sb_orientation_t));
  auto f = SetupTwoSpaces();
  sb_orientation_t o{};
  o.struct_size = sizeof(o);
  o.type = SB_ORIENTATION_YAW_PITCH_ROLL;
  o.yaw = 0.3;
  o.pitch = -0.25;
  o.roll = 0.1;
  sb_update_actor_pose(f.space_a, f.actor_a, 10.0, 10.0, &o);
  ProposalView p = ReadProposalOrFail(f.space_b, f.avatar_b);
  EXPECT_EQ(p.orientation_type, SB_ORIENTATION_YAW_PITCH_ROLL);
  // ABI 4.12 round trip: the root's turn carries the same yaw and flags as the proposal set, and a finite turn.
  sb_anchor_turn_t turn{};
  turn.struct_size = sizeof(turn);
  ASSERT_EQ(sb_read_anchor_turn(f.space_b, f.avatar_b, 0, &turn), 1);
  EXPECT_EQ(turn.anchor_id, 0u);
  EXPECT_EQ(turn.yaw, p.yaw);
  EXPECT_EQ(turn.flags, p.flags);
  EXPECT_TRUE(std::isfinite(turn.yaw_turn));
  if (p.flags & SB_ANCHOR_PROPOSAL_FLAG_YAW_UNSPECIFIED) EXPECT_EQ(turn.yaw_turn, 0.0);
  EXPECT_DOUBLE_EQ(p.pitch, -0.25);
  EXPECT_DOUBLE_EQ(p.roll, 0.1);
  // A pitch-only change is itself sent and reported; yaw-only resets the rest.
  o.type = SB_ORIENTATION_YAW_PITCH;
  o.pitch = 0.4;
  sb_update_actor_pose(f.space_a, f.actor_a, 10.0, 10.0, &o);
  p = ReadProposalOrFail(f.space_b, f.avatar_b);
  EXPECT_EQ(p.orientation_type, SB_ORIENTATION_YAW_PITCH);
  EXPECT_DOUBLE_EQ(p.pitch, 0.4);
  EXPECT_DOUBLE_EQ(p.roll, 0.0);
  sb_update_actor_state(f.space_a, f.actor_a, 10.5, 10.5, 0.0);
  p = ReadProposalOrFail(f.space_b, f.avatar_b);
  EXPECT_EQ(p.orientation_type, SB_ORIENTATION_YAW);
  EXPECT_DOUBLE_EQ(p.pitch, 0.0);
  // Invalid orientation updates nothing.
  sb_orientation_t bad = o;
  bad.struct_size = 1;
  const uint64_t before = p.revision;
  sb_update_actor_pose(f.space_a, f.actor_a, 11.0, 11.0, &bad);
  EXPECT_EQ(ReadProposalOrFail(f.space_b, f.avatar_b).revision, before);
  sb_destroy_session(f.session);
}

// ABI 4.12: a pursuing Facing proposal carries a non-zero signed turn through sb_read_anchor_turn. The source's root
// Facing steps by its count as the source turns in place; the receiver's root proposal requests that turn.
TEST(CApiProposalTest, APursuingFacingProposalCarriesItsSignedTurn) {
  auto f = SetupTwoSpaces();
  ASSERT_EQ(sb_bind(f.space_a, f.actor_a, "root", SB_KERNEL_FACING), 1);
  double largest = 0.0;
  int pursuing = 0;
  for (int k = 0; k <= 30; ++k) {
    sb_update_actor_state(f.space_a, f.actor_a, 10.0, 10.0, -0.6 + 0.05 * k);
    sb_anchor_turn_t turn{};
    turn.struct_size = sizeof(turn);
    ASSERT_EQ(sb_read_anchor_turn(f.space_b, f.avatar_b, 0, &turn), 1);
    if (turn.flags & SB_ANCHOR_PROPOSAL_FLAG_YAW_UNSPECIFIED) continue;
    EXPECT_TRUE(std::isfinite(turn.yaw_turn));
    if (turn.yaw_turn != 0.0) ++pursuing;
    largest = std::max(largest, std::fabs(turn.yaw_turn));
  }
  EXPECT_GT(pursuing, 0) << "some update requests a turn";
  EXPECT_GT(largest, 0.0);
  sb_destroy_session(f.session);
}

TEST(CApiProposalTest, TypedTargetEventsAreOptInAndCarryTheirOwnCheckedLayout) {
  auto f = SetupTwoSpaces();
  EXPECT_EQ(sb_set_target_event_delivery(f.space_b, f.avatar_b, 2), -1);
  EXPECT_EQ(sb_set_target_event_delivery(0, f.avatar_b, 1), -1);
  ASSERT_EQ(sb_set_target_event_delivery(f.space_b, f.avatar_b, 1), 0);
  EXPECT_EQ(sb_sizeof(SB_STRUCT_TARGET_REQUEST_PAYLOAD),
            sizeof(sb_target_request_payload_t));
  const uint32_t offsets[] = {
      offsetof(sb_target_request_payload_t, struct_size),
      offsetof(sb_target_request_payload_t, channels),
      offsetof(sb_target_request_payload_t, sample_epoch),
      offsetof(sb_target_request_payload_t, sample_sequence),
      offsetof(sb_target_request_payload_t, sample_yaw),
      offsetof(sb_target_request_payload_t, yaw_turn),
      offsetof(sb_target_request_payload_t, proposal_flags),
      offsetof(sb_target_request_payload_t, proposal_cause),
      offsetof(sb_target_request_payload_t, facing_yaw),
      offsetof(sb_target_request_payload_t, proposal_status),
      offsetof(sb_target_request_payload_t, request_flags)};
  EXPECT_EQ(sb_get_struct_field_count(SB_STRUCT_TARGET_REQUEST_PAYLOAD), 11);
  for (uint32_t i = 0; i < 11; ++i)
    EXPECT_EQ(sb_get_struct_field_offset(SB_STRUCT_TARGET_REQUEST_PAYLOAD, i),
              offsets[i]);
  sb_actor_event_t event{};
  event.struct_size = sizeof(event);
  ASSERT_EQ(sb_poll_actor_event(f.space_b, f.avatar_b, &event), 1);
  EXPECT_EQ(event.kind, SB_ACTOR_EVENT_TARGET_RELEASE);
  EXPECT_EQ(sb_poll_actor_event(f.space_b, f.avatar_b, &event), 0);
  sb_update_actor_state(f.space_a, f.actor_a, 10.0, 5.0, 0.0);
  const auto placed = ReadProposalOrFail(f.space_b, f.avatar_b);
  int positions = 0, exceptions = 0;
  while (sb_poll_actor_event(f.space_b, f.avatar_b, &event) == 1) {
    EXPECT_EQ(event.revision, placed.revision);
    if (event.kind == SB_ACTOR_EVENT_SOURCE_EXCEPTION) {
      ++exceptions;
      continue;
    }
    ASSERT_EQ(event.payload_size, sizeof(sb_target_request_payload_t));
    sb_target_request_payload_t payload{};
    std::memcpy(&payload, event.payload, sizeof(payload));
    EXPECT_EQ(payload.struct_size, sizeof(payload));
    if (event.kind != SB_ACTOR_EVENT_POSITION_REQUEST) continue;
    ++positions;
    EXPECT_EQ(payload.channels, SB_TARGET_CHANNEL_POSITION);
    EXPECT_EQ(payload.request_flags, SB_TARGET_REQUEST_PLACEMENT);
    EXPECT_NE(payload.sample_epoch, 0u);
    EXPECT_NE(payload.sample_sequence, 0u);
    EXPECT_DOUBLE_EQ(event.x, placed.x);
    EXPECT_DOUBLE_EQ(event.y, placed.y);
  }
  EXPECT_EQ(exceptions, 1);
  EXPECT_EQ(positions, 1);
  sb_update_actor_state(f.space_a, f.actor_a, 10.0, 5.0, 0.5);
  EXPECT_EQ(sb_poll_actor_event(f.space_b, f.avatar_b, &event), 0);
  EXPECT_EQ(ReadProposalOrFail(f.space_b, f.avatar_b).revision,
            placed.revision);
  event.struct_size = 1;
  EXPECT_EQ(sb_poll_actor_event(f.space_b, f.avatar_b, &event), -1)
      << "error is not queue-empty";
  sb_destroy_session(f.session);
}

// No request publishes no new target/revision. The driver's
// pending target survives; snapshots retain the last publication, without
// repurposing POSITION_UNSPECIFIED. NONE remains an explicit release.
TEST(CApiProposalTest, NoRequestRetainsTheLastPublishedTargetAndRevision) {
  EXPECT_EQ(sb_abi_minor_version(), 14u);
  EXPECT_EQ(SB_ANCHOR_PROPOSAL_FLAG_POSITION_UNSPECIFIED, 0x10u);
  auto f = SetupTwoSpaces();
  const auto holds = [](const ProposalView& p) {
    return p.position_unspecified();
  };
  const auto no_request = [](const ProposalView& p) {
    return (p.cause & SB_PROPOSAL_CAUSE_NO_REQUEST) != 0;
  };
  const auto is_target = [](const ProposalView& p) {
    return p.status == SB_PROPOSAL_EXACT || p.status == SB_PROPOSAL_INEXACT;
  };
  // The driver follows every proposal exactly, as the application would.
  const auto follow = [&](const ProposalView& p) {
    ASSERT_EQ(PublishPose(f.space_b, f.avatar_b, p.x, p.y, p.yaw), 0);
  };

  // The first update places the receiver (an absolute answer, arrival placement).
  sb_update_actor_state(f.space_a, f.actor_a, 10.0, 5.0, 0.0);
  ProposalView p = ReadProposalOrFail(f.space_b, f.avatar_b);
  ASSERT_TRUE(is_target(p));
  EXPECT_FALSE(holds(p));
  EXPECT_FALSE(no_request(p));
  follow(p);
  const double placed_x = p.x, placed_y = p.y;

  // Turning without a Facing relation issues no request: keep the last
  // published target and its revision.
  uint64_t revision = p.revision;
  for (double yaw : {0.5, 1.0}) {
    sb_update_actor_state(f.space_a, f.actor_a, 10.0, 5.0, yaw);
    p = ReadProposalOrFail(f.space_b, f.avatar_b);
    ASSERT_EQ(p.revision, revision)
        << "yaw " << yaw << ": no request is not a publication";
    ASSERT_TRUE(is_target(p)) << "the retained target is not NONE";
    EXPECT_FALSE(holds(p)) << "yaw " << yaw
                           << ": POSITION_UNSPECIFIED is not repurposed";
    EXPECT_DOUBLE_EQ(p.x, placed_x);
    EXPECT_DOUBLE_EQ(p.y, placed_y);
    follow(p);
  }

  // The source moves toward its referent: a relation issues a request again; the pursued target moves (PROGRESS).
  sb_update_actor_state(f.space_a, f.actor_a, 10.4, 5.0, 1.0);
  p = ReadProposalOrFail(f.space_b, f.avatar_b);
  ASSERT_GT(p.revision, revision);
  revision = p.revision;
  ASSERT_TRUE(is_target(p));
  EXPECT_FALSE(holds(p));
  EXPECT_FALSE(no_request(p));
  EXPECT_NE(p.cause & SB_PROPOSAL_CAUSE_PROGRESS, 0u);
  EXPECT_TRUE(p.x != placed_x || p.y != placed_y);
  const double moved_x = p.x, moved_y = p.y;
  follow(p);

  // Standing still again retains the new pursued target and its publication,
  // including its original cause.
  sb_update_actor_state(f.space_a, f.actor_a, 10.4, 5.0, 1.5);
  p = ReadProposalOrFail(f.space_b, f.avatar_b);
  ASSERT_EQ(p.revision, revision);
  ASSERT_TRUE(is_target(p));
  EXPECT_FALSE(holds(p));
  EXPECT_NE(p.cause & SB_PROPOSAL_CAUSE_PROGRESS, 0u);
  EXPECT_DOUBLE_EQ(p.x, moved_x);
  EXPECT_DOUBLE_EQ(p.y, moved_y);
  follow(p);

  // NONE still releases the root and carries no flag: when nothing in the
  // receiver maps the source's referents there is nothing to hold or to target.
  ASSERT_EQ(sb_remove_local_element(f.space_b, f.table_b), 1);
  ASSERT_EQ(sb_remove_local_element(f.space_b, f.board_b), 1);
  sb_update_actor_state(f.space_a, f.actor_a, 10.4, 5.5, 1.5);
  p = ReadProposalOrFail(f.space_b, f.avatar_b);
  ASSERT_GT(p.revision, revision);
  EXPECT_EQ(p.status, SB_PROPOSAL_NONE);
  EXPECT_FALSE(holds(p)) << "NONE releases; it does not hold";
  EXPECT_FALSE(no_request(p));
  sb_destroy_session(f.session);
}

TEST(CApiProposalTest, NoSampleYieldsNoSampleStatusNotAPose) {
  auto f = SetupTwoSpaces(/*publish_start=*/false);
  TargetAccumulator acc;
  sb_register_on_targets(f.space_b, AccumulateTargets, &acc);
  // The driver has published nothing: core must not invent a pose.
  sb_update_actor_state(f.space_a, f.actor_a, 10.0, 10.0, 0.0);
  ProposalView p = ReadProposalOrFail(f.space_b, f.avatar_b);
  EXPECT_EQ(p.status, SB_PROPOSAL_NO_SAMPLE);
  EXPECT_EQ(p.sample_valid, 0);
  EXPECT_EQ(p.x, 0.0);
  EXPECT_EQ(p.y, 0.0);
  EXPECT_EQ(acc.targets.size(), 0u) << "no realization target without a pose";

  // Publish validation: stale, invalid and wrong-size are rejected without
  // touching the sample.
  sb_anchor_pose_t pose = RootPose(0.0, 0.0, 0.0);
  EXPECT_EQ(sb_publish_anchor_pose(f.space_b, f.avatar_b, 0, &pose, 0.0, 1), 0);
  EXPECT_EQ(sb_publish_anchor_pose(f.space_b, f.avatar_b, 0, &pose, 0.0, 1), -2);
  pose.x = std::nan("");
  EXPECT_EQ(sb_publish_anchor_pose(f.space_b, f.avatar_b, 0, &pose, 0.0, 99), -1);
  pose.x = 0.0;
  EXPECT_EQ(sb_publish_anchor_pose(f.space_b, f.avatar_b, 0, &pose, 0.0, 0), -1);
  pose.struct_size = 1;
  EXPECT_EQ(sb_publish_anchor_pose(f.space_b, f.avatar_b, 0, &pose, 0.0, 99), -1);
  pose.struct_size = sizeof(pose);
  EXPECT_EQ(sb_publish_anchor_pose(f.space_b, f.avatar_b, 0, nullptr, 0.0, 99),
            -1);
  EXPECT_EQ(sb_publish_anchor_pose(f.space_b, 987654, 0, &pose, 0.0, 99), -1);
  EXPECT_EQ(sb_publish_anchor_pose(f.space_b, f.avatar_b, 77, &pose, 0.0, 99),
            -1) << "an anchor id nobody interned";
  // A new epoch restarts the sequence (a driver teleport / respawn).
  pose.epoch = 2;
  pose.flags = SB_EMBODIED_TELEPORTED;
  EXPECT_EQ(sb_publish_anchor_pose(f.space_b, f.avatar_b, 0, &pose, 0.0, 1), 0);
  sb_update_actor_state(f.space_a, f.actor_a, 10.5, 10.5, 0.0);
  p = ReadProposalOrFail(f.space_b, f.avatar_b);
  EXPECT_TRUE(p.status == SB_PROPOSAL_EXACT || p.status == SB_PROPOSAL_INEXACT);
  EXPECT_EQ(p.sample_epoch, 2u);
  sb_destroy_session(f.session);
}

// ABI 4 anchors: the root is id 0, an anchor is interned on first use, its name
// round-trips, and each anchor has its own port (own epoch, own sequence).
TEST(CApiProposalTest, AnchorIdsAndPerAnchorPorts) {
  auto f = SetupTwoSpaces();
  uint32_t root = 99, pelvis = 0, again = 0;
  ASSERT_EQ(sb_get_anchor_id(f.space_b, f.avatar_b, "root", &root), 0);
  EXPECT_EQ(root, 0u);
  ASSERT_EQ(sb_get_anchor_id(f.space_b, f.avatar_b, "pelvis", &pelvis), 0);
  EXPECT_NE(pelvis, 0u);
  ASSERT_EQ(sb_get_anchor_id(f.space_b, f.avatar_b, "pelvis", &again), 0);
  EXPECT_EQ(again, pelvis) << "ids are stable";
  char name[16];
  EXPECT_EQ(sb_get_anchor_name(f.space_b, f.avatar_b, pelvis, name, sizeof(name)), 6);
  EXPECT_STREQ(name, "pelvis");
  EXPECT_EQ(sb_get_anchor_name(f.space_b, f.avatar_b, 4242, name, sizeof(name)), -1);
  EXPECT_EQ(sb_get_anchor_id(f.space_b, 987654, "pelvis", &again), -1);
  EXPECT_EQ(sb_get_anchor_id(f.space_b, f.avatar_b, nullptr, &again), -1);
  // The pelvis port is independent of the root port: sequence 1 is fresh there
  // although the root already holds a sample.
  sb_anchor_pose_t pose = RootPose(1.0, 2.0, 0.5);
  pose.z = 0.45;
  EXPECT_EQ(sb_publish_anchor_pose(f.space_b, f.avatar_b, pelvis, &pose, 0.0, 1), 0);
  EXPECT_EQ(sb_publish_anchor_pose(f.space_b, f.avatar_b, pelvis, &pose, 0.0, 1), -2);
  EXPECT_EQ(sb_publish_anchor_pose(f.space_b, f.avatar_b, pelvis, &pose, 0.0, 2), 0);
  sb_destroy_session(f.session);
}

// ABI 4: the only teleport path is a
// source-app exception with an opaque payload, propagated to every other space.
TEST(CApiProposalTest, CourseQueryIsStatelessAndSourceExceptionIsPropagated) {
  auto f = SetupTwoSpaces();
  double xy[16];
  const int32_t n = sb_query_course(f.space_b, f.avatar_b, 1.0, 1.0, 10.0, 10.0,
                                    xy, 8);
  EXPECT_GE(n, 2);
  EXPECT_DOUBLE_EQ(xy[0], 1.0);
  EXPECT_DOUBLE_EQ(xy[1], 1.0);
  EXPECT_EQ(sb_query_course(f.space_b, f.avatar_b, 1.0, 1.0, 10.0, 10.0, nullptr,
                            8),
            -1);
  EXPECT_EQ(sb_query_course(f.space_b, f.avatar_b, 1.0, 1.0, 10.0, 10.0, xy, 1),
            -1);

  sb_update_actor_state(f.space_a, f.actor_a, 10.0, 10.0, 0.0);
  // The first update a source propagates is its join, itself an exception.
  sb_actor_event_t join{};
  join.struct_size = sizeof(join);
  ASSERT_EQ(sb_poll_actor_event(f.space_b, f.avatar_b, &join), 1);
  EXPECT_EQ(join.kind, SB_ACTOR_EVENT_SOURCE_EXCEPTION);
  EXPECT_EQ(join.exception_id, SB_EXCEPTION_JOIN);
  EXPECT_EQ(join.payload_size, 0u);
  EXPECT_EQ(sb_poll_actor_event(f.space_b, f.avatar_b, &join), 0);

  const uint8_t blob[] = {1, 2, 3, 250};
  EXPECT_EQ(sb_raise_source_exception(f.space_a, f.actor_a, 7, blob, sizeof(blob)), 0);
  EXPECT_EQ(sb_raise_source_exception(f.space_a, 987654, 7, blob, sizeof(blob)), -1);
  uint8_t huge[SB_EXCEPTION_PAYLOAD_MAX + 1] = {};
  EXPECT_EQ(sb_raise_source_exception(f.space_a, f.actor_a, 7, huge, sizeof(huge)),
            -1);
  EXPECT_EQ(sb_raise_source_exception(f.space_a, f.actor_a, 7, nullptr, 3), -1);
  sb_update_actor_state(f.space_a, f.actor_a, 3.0, 3.0, 0.0);
  ProposalView p = ReadProposalOrFail(f.space_b, f.avatar_b);
  EXPECT_TRUE(p.cause & SB_PROPOSAL_CAUSE_SOURCE_EXCEPTION);
  sb_actor_event_t e{};
  e.struct_size = sizeof(e);
  ASSERT_EQ(sb_poll_actor_event(f.space_b, f.avatar_b, &e), 1);
  EXPECT_EQ(e.kind, SB_ACTOR_EVENT_SOURCE_EXCEPTION);
  EXPECT_EQ(e.exception_id, 7u);
  ASSERT_EQ(e.payload_size, sizeof(blob));
  EXPECT_EQ(std::memcmp(e.payload, blob, sizeof(blob)), 0);
  EXPECT_EQ(e.revision, p.revision) << "raised with the set it belongs to";
  EXPECT_EQ(sb_poll_actor_event(f.space_b, f.avatar_b, &e), 0);
  // The exception rides ONE update only.
  sb_update_actor_state(f.space_a, f.actor_a, 3.2, 3.0, 0.0);
  p = ReadProposalOrFail(f.space_b, f.avatar_b);
  EXPECT_FALSE(p.cause & SB_PROPOSAL_CAUSE_SOURCE_EXCEPTION);
  EXPECT_EQ(sb_poll_actor_event(f.space_b, f.avatar_b, &e), 0);
  // Bad event arguments.
  e.struct_size = 1;
  EXPECT_EQ(sb_poll_actor_event(f.space_b, f.avatar_b, &e), -1);
  EXPECT_EQ(sb_set_body_radius(f.space_b, 0.25), 0);
  EXPECT_EQ(sb_set_body_radius(f.space_b, -1.0), -1);
  sb_destroy_session(f.session);
}

TEST(CApiGeometryTest, SetElementPolytope) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  sb_element_t elem = sb_add_element(space, 0);

  // Unit tetrahedron: 4 vertices, 4 triangular faces.
  const double verts[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  const int32_t face_idx[] = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
  const int32_t face_sizes[] = {3, 3, 3, 3};
  sb_set_element_polytope(space, elem, verts, 4, face_idx, face_sizes, 4,
                          SB_GEOMETRY_OBSTACLE);

  // Invalid input (too few vertices) is a safe no-op.
  sb_set_element_polytope(space, elem, verts, 2, face_idx, face_sizes, 4,
                          SB_GEOMETRY_OBSTACLE);

  sb_destroy_session(s);
  SUCCEED();
}

// ---------------------------------------------------------------------------
// Runtime configuration C API
// ---------------------------------------------------------------------------

namespace {
sb_element_t g_c3_received_element = 0;
int64_t g_c3_received_shared_id = 0;
}  // namespace

static void c3_capture_element_added(sb_space_t /*space*/,
                                     sb_element_t local_element,
                                     int64_t shared_id, const char* /*tag*/,
                                     const void* /*user_data*/,
                                     int64_t /*user_data_size*/,
                                     void* /*cb_user_data*/) {
  g_c3_received_element = local_element;
  g_c3_received_shared_id = shared_id;
}

TEST(CApiSupportCorrespondenceTest,
     ObservesExternalExactPairWithoutOwningItsLifetime) {
  sb_session_t session = sb_create_session();
  sb_space_t source = sb_add_space(session);
  sb_space_t target = sb_add_space(session);
  sb_element_t source_element = sb_add_element(source, 0);
  sb_set_element_shared_id(source, source_element, 901);
  sb_element_t external_target = sb_add_element(target, 0);
  sb_set_element_shared_id(target, external_target, 902);
  sb_element_t conflicting_target = sb_add_element(target, 0);

  // Establish an existing linker pair through the normal share/resolve path,
  // rather than through the registration under test.
  g_c3_received_element = 0;
  sb_register_on_element_added(target, c3_capture_element_added, nullptr);
  sb_share_element(source, source_element, nullptr, 0);
  ASSERT_NE(g_c3_received_element, 0);
  sb_resolve_shared_element(target, g_c3_received_element, external_target);

  const sb_support_correspondence_t exact = {source, source_element, target,
                                             external_target};
  sb_support_correspondence_registration_t observer =
      sb_register_support_correspondences(session, &exact, 1);
  ASSERT_NE(observer, 0);
  sb_dispose_support_correspondences(observer);

  // Disposal of the unowned observation cannot remove the original pair.
  const sb_support_correspondence_t conflict = {source, source_element, target,
                                                conflicting_target};
  EXPECT_EQ(sb_register_support_correspondences(session, &conflict, 1), 0);
  sb_destroy_session(session);
}

// Reads a trace JSONL file and returns the last line containing `needle`
// (empty string if none found). Deliberately avoids a JSON parser dependency
// here: c_api_test only needs to check for a couple of literal substrings.
// `kind` restricts the match to one event kind: since trace format 2 the
// referent id also appears on ev=relation / ev=goal / ev=target lines.
static std::string LastTraceLineContaining(const std::string& path,
                                           const std::string& needle,
                                           const std::string& kind = "") {
  std::ifstream in(path);
  std::string line, match;
  const std::string kind_field = "\"ev\":\"" + kind + "\"";
  while (std::getline(in, line)) {
    if (line.find(needle) != std::string::npos &&
        (kind.empty() || line.find(kind_field) != std::string::npos))
      match = line;
  }
  return match;
}

TEST(CApiConfigTest, MinMoveGateBlocksRepeatedSamePoseByDefault) {
  g_target_callback_count = 0;
  g_last_num_targets = 0;

  sb_session_t session = sb_create_session();
  sb_space_t sa = sb_add_space(session);
  sb_element_t room_a = sb_add_element(sa, 0);
  sb_set_element_shared_id(sa, room_a, 100);
  double poly[] = {0, 0, 10, 0, 10, 10, 0, 10};
  sb_set_element_polygon(sa, room_a, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_element_t obj_a = sb_add_element(sa, room_a);
  sb_set_element_shared_id(sa, obj_a, 200);

  sb_actor_t actor = sb_create_actor(sa);
  sb_add_anchor(sa, actor, "root", 0, 0, 0, NULL);
  sb_set_actor_element(sa, actor, room_a);
  sb_bind(sa, actor, "root", SB_KERNEL_SPACING);

  sb_space_t sb = sb_add_space(session);
  sb_element_t room_b = sb_add_element(sb, 0);
  sb_set_element_shared_id(sb, room_b, 100);
  sb_set_element_polygon(sb, room_b, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_element_t obj_b = sb_add_element(sb, room_b);
  sb_set_element_shared_id(sb, obj_b, 200);

  sb_actor_t avatar = sb_create_actor(sb);
  PublishPose(sb, avatar, 0.0, 0.0, 0.0);

  PublishPose(sb, avatar, 0.0, 0.0, 0.0);
  sb_add_anchor(sb, avatar, "root", 0, 0, 0, NULL);
  sb_set_actor_element(sb, avatar, room_b);
  sb_register_on_targets(sb, test_on_targets, nullptr);

  sb_update_actor_state(sa, actor, 5.0, 5.0, 0.0);
  int first_count = g_target_callback_count;
  EXPECT_GT(first_count, 0);

  // Second update at the IDENTICAL pose: the default 5cm/1.1deg movement
  // gate blocks it (no genuine motion), so the callback count is unchanged.
  sb_update_actor_state(sa, actor, 5.0, 5.0, 0.0);
  EXPECT_EQ(g_target_callback_count, first_count);

  sb_destroy_session(session);
}

// Replicates the movement-gate-disable pattern used throughout
// cross_space_test.cc (setenv("SB_V2_MIN_MOVE", "0", 1)) via the runtime config setter
// instead of the environment variable, on a space that never reads the env
// var. Same-pose repeats now propagate on every call, exactly as the
// env-based tests document.
TEST(CApiConfigTest, MinMoveSetterZeroMatchesEnvDisableBehavior) {
  g_target_callback_count = 0;
  g_last_num_targets = 0;

  sb_session_t session = sb_create_session();
  sb_space_t sa = sb_add_space(session);
  sb_element_t room_a = sb_add_element(sa, 0);
  sb_set_element_shared_id(sa, room_a, 100);
  double poly[] = {0, 0, 10, 0, 10, 10, 0, 10};
  sb_set_element_polygon(sa, room_a, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_element_t obj_a = sb_add_element(sa, room_a);
  sb_set_element_shared_id(sa, obj_a, 200);

  sb_actor_t actor = sb_create_actor(sa);
  sb_add_anchor(sa, actor, "root", 0, 0, 0, NULL);
  sb_set_actor_element(sa, actor, room_a);
  sb_bind(sa, actor, "root", SB_KERNEL_SPACING);

  // Disable the movement gate via the setter (mirrors SB_V2_MIN_MOVE=0).
  sb_set_min_move(sa, 0.0);
  sb_set_min_yaw(sa, 0.0);

  sb_space_t sb = sb_add_space(session);
  sb_element_t room_b = sb_add_element(sb, 0);
  sb_set_element_shared_id(sb, room_b, 100);
  sb_set_element_polygon(sb, room_b, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_element_t obj_b = sb_add_element(sb, room_b);
  sb_set_element_shared_id(sb, obj_b, 200);

  sb_actor_t avatar = sb_create_actor(sb);
  PublishPose(sb, avatar, 0.0, 0.0, 0.0);

  PublishPose(sb, avatar, 0.0, 0.0, 0.0);
  sb_add_anchor(sb, avatar, "root", 0, 0, 0, NULL);
  sb_set_actor_element(sb, avatar, room_b);
  sb_register_on_targets(sb, test_on_targets, nullptr);

  sb_update_actor_state(sa, actor, 5.0, 5.0, 0.0);
  int first_count = g_target_callback_count;
  EXPECT_GT(first_count, 0);

  // Same pose again: with the gate disabled via the setter, this still
  // propagates (identical to the env-var-disabled behavior).
  sb_update_actor_state(sa, actor, 5.0, 5.0, 0.0);
  EXPECT_GT(g_target_callback_count, first_count);

  sb_destroy_session(session);
}

TEST(CApiConfigTest, WeightSmoothingMinYawRealizationPolicySettersDoNotCrash) {
  sb_session_t session = sb_create_session();
  sb_space_t space = sb_add_space(session);
  sb_element_t room = sb_add_element(space, 0);
  sb_set_element_shared_id(space, room, 100);

  sb_set_weight_smoothing(space, 0.5);
  sb_set_min_yaw(space, 0.1);
  sb_set_realization_policy(space, SB_REALIZATION_POLICY_PROGRESSIVE);
  sb_set_realization_policy(space, SB_REALIZATION_POLICY_UNIFORM);
  sb_set_realization_policy(space, SB_REALIZATION_POLICY_GREEDY);

  // Invalid handles are a safe no-op (mirrors other sb_set_* functions).
  sb_set_weight_smoothing(999, 0.5);
  sb_set_min_move(999, 0.0);
  sb_set_min_yaw(999, 0.0);
  sb_set_realization_policy(999, SB_REALIZATION_POLICY_UNIFORM);
  sb_set_referent_resolve_mode(999, SB_REFERENT_RESOLVE_SHAREDID);
  sb_set_trace(999, nullptr);

  sb_destroy_session(session);
  SUCCEED();
}

// Per-family lifecycle thresholds are settable and observable. ABI 4
// deleted the activation pair, the near field with its dwell, and
// both scope radii from this struct; the assertions on them are retired with it.
TEST(CApiConfigTest, Thresholds_E1_6_RelationPhaseConfigSetAndRead) {
  sb_session_t session = sb_create_session();
  sb_space_t space = sb_add_space(session);

  EXPECT_EQ(sb_sizeof(SB_STRUCT_RELATION_PHASE_CONFIG),
            sizeof(sb_relation_phase_config_t));
  EXPECT_EQ(sizeof(sb_relation_phase_config_t), 8u + 4 * sizeof(double));
  sb_relation_phase_config_t affordance{};
  affordance.struct_size = sizeof(affordance);
  ASSERT_EQ(sb_set_relation_phase_config(space, SB_KERNEL_AFFORDANCE_POSE,
                                         nullptr, &affordance),
            1);
  EXPECT_EQ(affordance.struct_size, sizeof(sb_relation_phase_config_t));
  EXPECT_DOUBLE_EQ(affordance.driving_weight, 0.50);
  EXPECT_DOUBLE_EQ(affordance.contact_release, 0.85);

  sb_relation_phase_config_t strict = affordance;
  strict.driving_weight = 0.9;
  strict.driving_release = 0.8;
  sb_relation_phase_config_t effective{};
  effective.struct_size = sizeof(effective);
  EXPECT_EQ(sb_set_relation_phase_config(space, SB_KERNEL_AFFORDANCE_POSE,
                                         &strict, &effective),
            1);
  EXPECT_DOUBLE_EQ(effective.driving_weight, 0.9);

  // Per family: Facing is unchanged.
  sb_relation_phase_config_t facing{};
  facing.struct_size = sizeof(facing);
  ASSERT_EQ(sb_set_relation_phase_config(space, SB_KERNEL_FACING, nullptr,
                                         &facing),
            1);
  EXPECT_DOUBLE_EQ(facing.driving_weight, 0.50);

  // Output honors the caller's struct_size: a caller built against an older,
  // smaller struct gets only its prefix -- the guard bytes after it survive.
  sb_relation_phase_config_t guarded;
  std::memset(&guarded, 0xAB, sizeof(guarded));
  const size_t old_size = offsetof(sb_relation_phase_config_t, contact_proximity);
  guarded.struct_size = static_cast<uint32_t>(old_size);
  ASSERT_EQ(sb_set_relation_phase_config(space, SB_KERNEL_AFFORDANCE_POSE,
                                         nullptr, &guarded),
            1);
  EXPECT_EQ(guarded.struct_size, old_size);
  EXPECT_DOUBLE_EQ(guarded.driving_weight, 0.9);
  const unsigned char* tail =
      reinterpret_cast<const unsigned char*>(&guarded) + old_size;
  for (size_t i = 0; i < sizeof(sb_relation_phase_config_t) - old_size; ++i) {
    ASSERT_EQ(tail[i], 0xAB) << "byte " << i << " past struct_size written";
  }
  sb_relation_phase_config_t tiny{};
  tiny.struct_size = 4;
  EXPECT_EQ(sb_set_relation_phase_config(space, SB_KERNEL_AFFORDANCE_POSE,
                                         nullptr, &tiny),
            0);

  // An undersized (older) struct is rejected.
  sb_relation_phase_config_t small = strict;
  small.struct_size = 8;
  EXPECT_EQ(sb_set_relation_phase_config(space, SB_KERNEL_AFFORDANCE_POSE,
                                         &small, nullptr),
            0);

  // Invalid config rejected, previous kept and reported.
  sb_relation_phase_config_t bad = strict;
  bad.driving_release = 0.95;  // above driving_weight
  EXPECT_EQ(sb_set_relation_phase_config(space, SB_KERNEL_AFFORDANCE_POSE,
                                         &bad, &effective),
            0);
  EXPECT_DOUBLE_EQ(effective.driving_release, 0.8);

  // Invalid handle / family.
  EXPECT_EQ(sb_set_relation_phase_config(999, SB_KERNEL_FACING, nullptr,
                                         &effective),
            0);
  EXPECT_EQ(sb_set_relation_phase_config(
                space, static_cast<sb_kernel_type_t>(7), nullptr, &effective),
            0);
  sb_destroy_session(session);
}

// A duplicated shared id resolves to one element by the declared rule
// (smallest element handle), whatever the registration order.
TEST(CApiConfigTest, UniqueIdentity_H_2_DuplicateResolvesByDeclaredRule) {
  sb_session_t session = sb_create_session();
  sb_space_t space = sb_add_space(session);
  sb_element_t room = sb_add_element(space, 0);
  sb_element_t a = sb_add_element(space, room);
  sb_element_t b = sb_add_element(space, room);
  ASSERT_LT(a, b);
  sb_set_element_shared_id(space, b, 500);
  sb_set_element_shared_id(space, a, 500);
  EXPECT_EQ(sb_find_element_by_shared_id(space, 500), a);
  sb_destroy_session(session);
}

TEST(CApiConfigTest, AbiVersionAndSizeof) {
  EXPECT_EQ(sb_abi_version(), 4u);
  EXPECT_EQ(sb_sizeof(SB_STRUCT_ACTOR_STATE), sizeof(sb_actor_state_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_REALIZATION_TARGET),
            sizeof(sb_realization_target_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_RELATION_EVENT), sizeof(sb_relation_event_t));
  EXPECT_EQ(sb_sizeof(static_cast<sb_struct_id_t>(999)), 0u);
}

TEST(CApiElementLifecycleTest, RemovalCallbacksArePerSpaceAndNullable) {
  g_removed_added_b = g_removed_added_c = 0;
  g_removed_b.clear();
  g_removed_c.clear();
  sb_session_t session = sb_create_session();
  sb_space_t source = sb_add_space(session);
  sb_space_t b = sb_add_space(session);
  sb_space_t c = sb_add_space(session);
  sb_element_t element = sb_add_element(source, 0);
  sb_set_element_shared_id(source, element, 77);
  sb_register_on_element_added(b, CaptureRemovalAddedB, nullptr);
  sb_register_on_element_added(c, CaptureRemovalAddedC, nullptr);
  sb_register_on_element_removed(b, CaptureRemovedB, nullptr);
  sb_register_on_element_removed(c, CaptureRemovedC, nullptr);
  sb_share_element(source, element, nullptr, 0);
  ASSERT_NE(g_removed_added_b, 0);
  ASSERT_NE(g_removed_added_c, 0);
  sb_resolve_shared_element(b, g_removed_added_b, g_removed_added_b);
  sb_resolve_shared_element(c, g_removed_added_c, g_removed_added_c);
  sb_unshare_element(source, element);
  ASSERT_EQ(g_removed_b.size(), 1u);
  ASSERT_EQ(g_removed_c.size(), 1u);
  EXPECT_EQ(g_removed_b[0], g_removed_added_b);
  EXPECT_EQ(g_removed_c[0], g_removed_added_c);

  // Clearing only b leaves c independently registered.
  sb_register_on_element_removed(b, nullptr, nullptr);
  g_removed_added_b = g_removed_added_c = 0;
  sb_share_element(source, element, nullptr, 0);
  ASSERT_NE(g_removed_added_b, 0);
  ASSERT_NE(g_removed_added_c, 0);
  sb_resolve_shared_element(b, g_removed_added_b, g_removed_added_b);
  sb_resolve_shared_element(c, g_removed_added_c, g_removed_added_c);
  sb_unshare_element(source, element);
  EXPECT_EQ(g_removed_b.size(), 1u);
  ASSERT_EQ(g_removed_c.size(), 2u);
  EXPECT_EQ(g_removed_c.back(), g_removed_added_c);
  sb_destroy_session(session);
}

TEST(CApiConfigTest, AbiLayoutDescriptor) {
  const uint32_t required_layout_size = sb_get_v2_abi_layout(nullptr, 0);
  EXPECT_EQ(required_layout_size,
            static_cast<uint32_t>(sizeof(sb_v2_abi_layout_t)));

  alignas(sb_v2_abi_layout_t)
      uint8_t short_layout_bytes[sizeof(sb_v2_abi_layout_t)];
  std::memset(short_layout_bytes, 0xA5, sizeof(short_layout_bytes));
  sb_v2_abi_layout_t* short_layout =
      reinterpret_cast<sb_v2_abi_layout_t*>(short_layout_bytes);
  for (uint32_t available : {0u, 1u, required_layout_size - 1u}) {
    EXPECT_EQ(sb_get_v2_abi_layout(short_layout, available),
              required_layout_size);
    for (uint8_t byte : short_layout_bytes) {
      EXPECT_EQ(byte, 0xA5) << "short descriptor call modified caller storage";
    }
  }
  EXPECT_EQ(sb_get_v2_abi_layout(nullptr, required_layout_size),
            required_layout_size);

  sb_v2_abi_layout_t layout{};
  EXPECT_EQ(sb_get_v2_abi_layout(&layout, sizeof(layout)),
            required_layout_size);
  EXPECT_EQ(layout.descriptor_size, required_layout_size);
  EXPECT_EQ(layout.abi_version, sb_abi_version());
  EXPECT_EQ(layout.realization_target_size,
            static_cast<uint32_t>(sizeof(sb_realization_target_t)));
  EXPECT_EQ(layout.pointer_size, static_cast<uint32_t>(sizeof(const char*)));
  EXPECT_EQ(layout.kernel_type_size,
            static_cast<uint32_t>(sizeof(sb_kernel_type_t)));
  EXPECT_EQ(
      layout.arrival_time_offset,
      static_cast<uint32_t>(offsetof(sb_realization_target_t, arrival_time)));
  EXPECT_EQ(
      layout.anchor_name_offset,
      static_cast<uint32_t>(offsetof(sb_realization_target_t, anchor_name)));
  EXPECT_EQ(
      layout.position_x_offset,
      static_cast<uint32_t>(offsetof(sb_realization_target_t, position_x)));
  EXPECT_EQ(
      layout.position_y_offset,
      static_cast<uint32_t>(offsetof(sb_realization_target_t, position_y)));
  EXPECT_EQ(
      layout.position_z_offset,
      static_cast<uint32_t>(offsetof(sb_realization_target_t, position_z)));
  EXPECT_EQ(layout.yaw_offset,
            static_cast<uint32_t>(offsetof(sb_realization_target_t, yaw)));
  EXPECT_EQ(layout.weight_offset,
            static_cast<uint32_t>(offsetof(sb_realization_target_t, weight)));
  EXPECT_EQ(layout.has_z_offset,
            static_cast<uint32_t>(offsetof(sb_realization_target_t, has_z)));
  EXPECT_EQ(
      layout.kernel_type_offset,
      static_cast<uint32_t>(offsetof(sb_realization_target_t, kernel_type)));
  EXPECT_EQ(
      layout.referent_id_offset,
      static_cast<uint32_t>(offsetof(sb_realization_target_t, referent_id)));
}

// End-to-end: sb_set_referent_resolve_mode has a genuine, observable effect
// on receiver-side referent resolution, visible through the structured
// trace's "via" field on the ev=payload line. A referent element in the
// source space is linked through the local native support-correspondence
// boundary to an EXISTING target-space element carrying a DIFFERENT shared id
// (999 vs 888). In the default LINKER mode this deliberate shared-id mismatch
// is immaterial: Translate() resolves the referent via the explicit link. In
// SHAREDID mode, Translate() is skipped entirely, so fallback cannot find the
// target -- exactly the divergence the runtime config makes switchable at
// runtime.
TEST(CApiConfigTest, ReferentResolveModeAffectsLinkerVsSharedIdResolution) {
  g_c3_received_element = 0;
  g_c3_received_shared_id = 0;

  std::string trace_path =
      ::testing::TempDir() + "c3_referent_resolve_trace.jsonl";
  std::remove(trace_path.c_str());

  sb_session_t session = sb_create_session();
  sb_space_t sa = sb_add_space(session);
  sb_space_t sb = sb_add_space(session);

  double poly[] = {0, 0, 10, 0, 10, 10, 0, 10};

  sb_element_t room_a = sb_add_element(sa, 0);
  sb_set_element_shared_id(sa, room_a, 100);
  sb_set_element_polygon(sa, room_a, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_element_t desk_a = sb_add_element(sa, room_a);
  sb_set_element_shared_id(sa, desk_a, 999);
  sb_set_element_box(sa, desk_a, SB_GEOMETRY_OBSTACLE, 4.0, 4.0, 0.0, 6.0, 6.0,
                     1.0);

  sb_element_t room_b = sb_add_element(sb, 0);
  sb_set_element_shared_id(sb, room_b, 100);
  sb_set_element_polygon(sb, room_b, SB_GEOMETRY_NAVIGABLE, poly, 4);
  // Deliberately a DIFFERENT shared_id (888, not 999) -- the link is the only
  // thing that ties this element back to desk_a.
  sb_element_t desk_b_existing = sb_add_element(sb, room_b);
  sb_set_element_shared_id(sb, desk_b_existing, 888);
  // Deliberately displaced from desk_a: an output near x=7 can only have
  // been grounded against this receiver-local support, not source geometry.
  sb_set_element_box(sb, desk_b_existing, SB_GEOMETRY_OBSTACLE, 7.0, 4.0, 0.0,
                     9.0, 6.0, 1.0);

  sb_actor_t actor = sb_create_actor(sa);
  sb_add_anchor(sa, actor, "root", 0, 0, 0, NULL);
  sb_set_actor_element(sa, actor, room_a);
  sb_bind(sa, actor, "root", SB_KERNEL_SPACING);

  sb_actor_t avatar = sb_create_actor(sb);
  PublishPose(sb, avatar, 0.0, 0.0, 0.0);

  PublishPose(sb, avatar, 0.0, 0.0, 0.0);
  sb_add_anchor(sb, avatar, "root", 0, 0, 0, NULL);
  sb_set_actor_element(sb, avatar, room_b);
  g_target_callback_count = 0;
  g_last_num_targets = 0;
  sb_register_on_targets(sb, test_on_targets, nullptr);

  // Publish the unlike support ids directly through the local native boundary;
  // no shared-id/category inference or temporary replica is involved.
  const sb_support_correspondence_t desk_support = {sa, desk_a, sb,
                                                    desk_b_existing};
  sb_support_correspondence_registration_t desk_registration =
      sb_register_support_correspondences(session, &desk_support, 1);
  ASSERT_NE(desk_registration, 0);

  sb_set_min_move(sa, 0.0);
  sb_set_min_yaw(sa, 0.0);
  // sb_set_trace is genuinely per-space (TraceSink::ForSpace) --
  // "ev=payload" (the line this test asserts on) is emitted by the
  // RECEIVING space (sb, inside Space::OnRemoteUpdate), not by sa (the
  // source, which only emits "ev=src"). Before per-space routing, binding sa alone activated
  // the one process-wide TraceSink that every space shared, so sb's lines
  // landed in the same file as a side effect; now sb needs its own explicit
  // binding to the SAME path to keep writing into it (see trace.h's
  // SetPathForSpace doc: two space_ids bound to an identical path string
  // share the same underlying sink).
  sb_set_trace(sa, trace_path.c_str());
  sb_set_trace(sb, trace_path.c_str());

  // Default mode is LINKER: resolution goes through the link created above.
  sb_update_actor_state(sa, actor, 3.0, 5.0, 0.0);
  std::string linker_line =
      LastTraceLineContaining(trace_path, "\"referent\":999", "payload");
  ASSERT_FALSE(linker_line.empty()) << "no ev=payload line for referent 999";
  EXPECT_NE(linker_line.find("\"via\":\"linker\""), std::string::npos)
      << linker_line;

  // The first sample places the body at the mapped approach: desk B is
  // centred at (8, 5) and the source stands 2 m west of desk A, so the
  // receiver stands at (6, 5). Holding the embodied sample was the removed
  // static-source freeze. The link itself is the trace line above.
  ASSERT_GT(g_target_callback_count, 0);
  bool saw_spacing = false;
  for (int64_t i = 0; i < g_last_num_targets && i < 16; ++i) {
    if (g_last_targets[i].kernel_type != SB_KERNEL_SPACING) continue;
    saw_spacing = true;
    EXPECT_NEAR(g_last_targets[i].position_x, 6.0, 1e-2);
    EXPECT_NEAR(g_last_targets[i].position_y, 5.0, 1e-2);
  }
  EXPECT_TRUE(saw_spacing);

  // Force SHAREDID mode: Translate() is skipped, so the mismatched shared_id
  // (999 on the payload vs 888 on the linked element) makes the fallback
  // lookup miss -- resolution now reports "sharedid" (unresolved locally).
  sb_set_referent_resolve_mode(sb, SB_REFERENT_RESOLVE_SHAREDID);
  sb_update_actor_state(sa, actor, 3.1, 5.0, 0.0);
  std::string sharedid_line =
      LastTraceLineContaining(trace_path, "\"referent\":999", "payload");
  ASSERT_FALSE(sharedid_line.empty());
  EXPECT_NE(sharedid_line.find("\"via\":\"sharedid\""), std::string::npos)
      << sharedid_line;

  sb_set_trace(sa, nullptr);  // disable sa's + sb's (per-space) sinks
  sb_set_trace(sb, nullptr);  // before exit
  sb_dispose_support_correspondences(desk_registration);
  sb_destroy_session(session);
  std::remove(trace_path.c_str());
}

TEST(CApiE2ETest, LinkedUnlikeActorIdsDispatchTargetsToExactLocalActor) {
  g_c3_received_element = 0;
  g_callback_actor = 0;
  sb_session_t session = sb_create_session();
  sb_space_t source = sb_add_space(session);
  sb_space_t target = sb_add_space(session);
  double polygon[] = {0, 0, 10, 0, 10, 10, 0, 10};

  sb_element_t source_room = sb_add_element(source, 0);
  sb_set_element_shared_id(source, source_room, 101);
  sb_set_element_polygon(source, source_room, SB_GEOMETRY_NAVIGABLE, polygon,
                         4);
  sb_element_t source_actor_element = sb_add_element(source, source_room);
  sb_set_element_shared_id(source, source_actor_element, 102);
  sb_element_t source_desk = sb_add_element(source, source_room);
  sb_set_element_shared_id(source, source_desk, 103);
  sb_set_element_box(source, source_desk, SB_GEOMETRY_OBSTACLE, 4, 4, 0, 6, 6,
                     1);
  sb_actor_t source_actor = sb_create_actor(source);
  sb_add_anchor(source, source_actor, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(source, source_actor, source_actor_element);
  sb_bind(source, source_actor, "root", SB_KERNEL_SPACING);

  sb_element_t target_room = sb_add_element(target, 0);
  sb_set_element_shared_id(target, target_room, 101);
  sb_set_element_polygon(target, target_room, SB_GEOMETRY_NAVIGABLE, polygon,
                         4);
  sb_element_t target_actor_element = sb_add_element(target, target_room);
  sb_set_element_shared_id(target, target_actor_element, 202);  // unlike source
  sb_element_t target_desk = sb_add_element(target, target_room);
  sb_set_element_shared_id(target, target_desk, 103);
  sb_set_element_box(target, target_desk, SB_GEOMETRY_OBSTACLE, 4, 4, 0, 6, 6,
                     1);
  for (int i = 0; i < 1000; ++i) sb_create_actor(target);
  sb_actor_t target_actor = sb_create_actor(target);  // exceeds old 1..999 scan
  sb_add_anchor(target, target_actor, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(target, target_actor, target_actor_element);
  sb_register_on_targets(target, capture_callback_actor, nullptr);

  sb_register_on_element_added(target, c3_capture_element_added, nullptr);
  sb_share_element(source, source_actor_element, nullptr, 0);
  ASSERT_NE(g_c3_received_element, 0);
  sb_resolve_shared_element(target, g_c3_received_element,
                            target_actor_element);
  sb_update_actor_state(source, source_actor, 3, 5, 0);
  EXPECT_EQ(g_callback_actor, target_actor);
  sb_destroy_session(session);
}

TEST(CApiElementLifecycleTest, RemoveLocalElementIsConservativeAndIdempotent) {
  sb_session_t session = sb_create_session();
  sb_space_t space = sb_add_space(session);
  sb_element_t root = sb_add_element(space, 0);
  sb_element_t removable = sb_add_element(space, root);
  sb_element_t descendant = sb_add_element(space, removable);
  sb_element_t sibling = sb_add_element(space, root);
  sb_element_t shared = sb_add_element(space, root);
  sb_set_element_shared_id(space, removable, 71);
  sb_actor_t actor = sb_create_actor(space);
  sb_add_anchor(space, actor, "hand", 0, 0, 1, nullptr);
  sb_set_actor_element(space, actor, removable);
  sb_set_anchor_transform(space, actor, "hand", 1, 2, 3, 0);

  EXPECT_EQ(sb_remove_local_element(space, root), 0);
  sb_share_element(space, shared, nullptr, 0);
  EXPECT_EQ(sb_remove_local_element(space, shared), 0);
  EXPECT_EQ(sb_remove_local_element(space, removable), 1);
  EXPECT_EQ(sb_remove_local_element(space, removable), 0);
  EXPECT_EQ(sb_find_actor_by_shared_id(space, 71), 0);
  EXPECT_EQ(sb_remove_local_element(space, sibling), 1);
  EXPECT_EQ(sb_remove_local_element(space, descendant), 0);
  sb_destroy_session(session);
}
// Session create->destroy->create repeated 100x must yield IDENTICAL
// realization output on iteration 1 vs iteration 100 (no process-global state
// leak across session lifecycles, e.g. via the global g_sessions/g_spaces
// registries, atomic id counters, or process-env caches read by Space).
// ---------------------------------------------------------------------------

namespace {

// Build the same two-space scenario as SetupTwoSpaces() and run the actor to
// a fixed pose, returning the FIRST realization target reported to space_b.
sb_realization_target_t RunOneSessionIteration() {
  TwoSpaceFixture f = SetupTwoSpaces();

  TargetAccumulator acc;
  sb_register_on_targets(f.space_b, AccumulateTargets, &acc);

  // Deterministic walk toward the tabletop affordance (mirrors
  // MultiStepTrajectory), independent of any process-global movement-gate
  // state: each iteration gets a brand-new session/space, so if state truly
  // resets, the same walk must produce the same realized target every time.
  double start_x = 2.0, start_y = 2.0;
  double goal_x = 5.0, goal_y = 13.0;
  for (int step = 0; step < 5; ++step) {
    double t = static_cast<double>(step + 1) / 5.0;
    double x = start_x + t * (goal_x - start_x);
    double y = start_y + t * (goal_y - start_y);
    double yaw = std::atan2(goal_y - start_y, goal_x - start_x);
    acc.Reset();
    sb_update_actor_state(f.space_a, f.actor_a, x, y, yaw);
  }

  EXPECT_GT(static_cast<int>(acc.targets.size()), 0)
      << "final step must report at least one realization target";
  sb_realization_target_t result =
      acc.targets.empty() ? sb_realization_target_t{} : acc.targets[0];

  sb_destroy_session(f.session);
  return result;
}

}  // namespace

TEST(CApiSessionLifecycleTest, Repeated100xCreateDestroyIdenticalRealization) {
  sb_realization_target_t first{};
  sb_realization_target_t last{};
  constexpr int kIterations = 100;
  for (int i = 0; i < kIterations; ++i) {
    sb_realization_target_t r = RunOneSessionIteration();
    if (i == 0) first = r;
    if (i == kIterations - 1) last = r;
  }

  EXPECT_DOUBLE_EQ(first.position_x, last.position_x)
      << "iteration 1 vs " << kIterations
      << " position_x diverged: " << first.position_x << " vs "
      << last.position_x;
  EXPECT_DOUBLE_EQ(first.position_y, last.position_y)
      << "iteration 1 vs " << kIterations
      << " position_y diverged: " << first.position_y << " vs "
      << last.position_y;
  EXPECT_DOUBLE_EQ(first.yaw, last.yaw)
      << "iteration 1 vs " << kIterations << " yaw diverged: " << first.yaw
      << " vs " << last.yaw;
  EXPECT_EQ(first.kernel_type, last.kernel_type);
  EXPECT_EQ(first.referent_id, last.referent_id);
  EXPECT_DOUBLE_EQ(first.arrival_time, last.arrival_time)
      << "iteration 1 vs " << kIterations
      << " arrival_time diverged: " << first.arrival_time << " vs "
      << last.arrival_time;
}

// ---------------------------------------------------------------------------
// Anchor liveness (sb_set_anchor_transform / sb_clear_anchor_transform)
// and relation-state query (sb_get_relation_states).
// ---------------------------------------------------------------------------

TEST(CApiAnchorTransformTest,
     SetAnchorTransformGroundsAffordanceRootUnchanged) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);

  sb_element_t room = sb_add_element(space, 0);
  double poly[] = {0, 0, 20, 0, 20, 20, 0, 20};
  sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, poly, 4);

  // "Tabletop": a leaf element with no boundary of its own, at world (5, 8).
  sb_element_t table = sb_add_element(space, room);
  sb_set_element_shared_id(space, table, 9004);  // shared: an unshared element is never a referent (SP-D7-03)
  double table_mat[16] = {1, 0, 0, 5, 0, 1, 0, 8, 0, 0, 1, 0, 0, 0, 0, 1};
  sb_set_element_transform(space, table, table_mat);
  sb_set_element_tag(space, table, "Tabletop");

  sb_actor_t actor = sb_create_actor(space);
  sb_add_anchor(space, actor, "root", 0, 0, 0, NULL);
  sb_add_anchor(space, actor, "hand", 0.4, 0, 0.9, NULL);
  sb_set_actor_element(space, actor, room);
  sb_bind_affordance(space, actor, "hand",
                 "Tabletop");

  // Root inside the room; hand anchor offset (0.4, 0, 0.9) from actor origin,
  // so it does not sit exactly on the table's world origin (5, 8, 0).
  sb_update_actor_state(space, actor, 5.0, 8.0, 0.0);

  sb_relation_state_t before[4];
  int32_t before_count = sb_get_relation_states(space, actor, before, 4);
  ASSERT_EQ(before_count, 1);
  EXPECT_EQ(before[0].kernel_type, SB_KERNEL_AFFORDANCE_POSE);
  EXPECT_LT(before[0].proximity, 0.95);

  sb_actor_state_t root_before = sb_get_actor_state(space, actor);

  // Move the "hand" anchor exactly onto the table's world origin.
  sb_set_anchor_transform(space, actor, "hand", 5.0, 8.0, 0.0, 0.3);
  sb_update_actor_state(space, actor, 5.0, 8.0, 0.0);

  sb_relation_state_t after[4];
  int32_t after_count = sb_get_relation_states(space, actor, after, 4);
  ASSERT_EQ(after_count, 1);
  EXPECT_GE(after[0].proximity, 1.0 - 1e-6);
  EXPECT_STREQ(after[0].anchor_name, "hand");

  sb_actor_state_t root_after = sb_get_actor_state(space, actor);
  EXPECT_EQ(root_after.position_x, root_before.position_x);
  EXPECT_EQ(root_after.position_y, root_before.position_y);
  EXPECT_EQ(root_after.yaw, root_before.yaw);

  // Clearing the override reverts to the offset-driven proximity.
  sb_clear_anchor_transform(space, actor, "hand");
  sb_update_actor_state(space, actor, 5.0, 8.0, 0.0);
  sb_relation_state_t cleared[4];
  int32_t cleared_count = sb_get_relation_states(space, actor, cleared, 4);
  ASSERT_EQ(cleared_count, 1);
  EXPECT_LT(cleared[0].proximity, 0.95);
  EXPECT_DOUBLE_EQ(cleared[0].proximity, before[0].proximity);

  sb_destroy_session(s);
}

TEST(CApiAnchorTransformTest, GetRelationStatesFieldsMatchSpacingMatch) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);

  sb_element_t room = sb_add_element(space, 0);
  double poly[] = {0, 0, 20, 0, 20, 20, 0, 20};
  sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, poly, 4);

  sb_element_t table = sb_add_element(space, room);
  sb_set_element_shared_id(space, table, 300);  // shared: an unshared element is never a referent
  double table_mat[16] = {1, 0, 0, 10, 0, 1, 0, 10, 0, 0, 1, 0, 0, 0, 0, 1};
  sb_set_element_transform(space, table, table_mat);
  // Element-local footprint (the table is at (10,10): inside the room).
  double table_bnd[] = {-1, -1, 1, -1, 1, 1, -1, 1};
  sb_set_element_polygon(space, table, SB_GEOMETRY_CONTAINMENT, table_bnd, 4);

  sb_actor_t actor = sb_create_actor(space);
  sb_add_anchor(space, actor, "root", 0, 0, 0, NULL);
  sb_set_actor_element(space, actor, room);
  sb_bind(space, actor, "root", SB_KERNEL_SPACING);

  sb_update_actor_state(space, actor, 1.0, 1.0, 0.0);

  sb_relation_state_t states[4];
  int32_t count = sb_get_relation_states(space, actor, states, 4);
  ASSERT_EQ(count, 1);
  EXPECT_EQ(states[0].kernel_type, SB_KERNEL_SPACING);
  EXPECT_STREQ(states[0].anchor_name, "root");
  EXPECT_TRUE(states[0].has_contact);
  EXPECT_TRUE(std::isfinite(states[0].proximity));
  EXPECT_TRUE(std::isfinite(states[0].contact_point_world_x));
  EXPECT_TRUE(std::isfinite(states[0].contact_point_world_y));

  sb_destroy_session(s);
}

TEST(CApiAnchorTransformTest, CapLimitsWriteCountButReturnsFullCount) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  sb_element_t room = sb_add_element(space, 0);

  sb_element_t a = sb_add_element(space, room);
  sb_set_element_shared_id(space, a, 9005);  // shared: an unshared element is never a referent (SP-D7-03)
  sb_set_element_tag(space, a, "target");
  sb_element_t b = sb_add_element(space, room);
  sb_set_element_shared_id(space, b, 9006);  // shared: an unshared element is never a referent (SP-D7-03)
  sb_set_element_tag(space, b, "target");

  sb_actor_t actor = sb_create_actor(space);
  sb_add_anchor(space, actor, "root", 0, 0, 0, NULL);
  sb_set_actor_element(space, actor, room);
  // (the tag is ignored for Facing, D6; a and b are the room's two referents anyway)
  sb_bind(space, actor, "root", SB_KERNEL_FACING);

  sb_update_actor_state(space, actor, 1.0, 1.0, 0.0);

  sb_relation_state_t one[1];
  int32_t count = sb_get_relation_states(space, actor, one, 1);
  EXPECT_EQ(count, 2);  // full count even though only 1 slot was writable

  sb_destroy_session(s);
}

TEST(CApiAnchorTransformTest, InvalidHandlesDoNotCrash) {
  sb_relation_state_t out[4];
  EXPECT_EQ(sb_get_relation_states(999, 1, out, 4), 0);
  EXPECT_EQ(sb_get_relation_states(0, 0, nullptr, 4), 0);
  sb_set_anchor_transform(999, 1, "hand", 0, 0, 0, 0);
  sb_set_anchor_transform(999, 1, nullptr, 0, 0, 0, 0);
  sb_clear_anchor_transform(999, 1, "hand");
  sb_clear_anchor_transform(999, 1, nullptr);
  EXPECT_EQ(sb_set_anchor_active(999, 1, "hand", 0), 0);
  EXPECT_EQ(sb_set_anchor_active(0, 0, nullptr, 0), 0);
  // Reaching here without crashing is the assertion.
  SUCCEED();
}

namespace {
struct RichCallbackCapture {
  int calls = 0;
  sb_rich_realization_header_t header{};
  uint32_t channels = 0;
  uint32_t contributors = 0;
  uint32_t utf8_bytes = 0;
  bool registration_reentered = false;
};

void CaptureRichSnapshot(const sb_rich_realization_snapshot_t* snapshot,
                         void* user_data) {
  auto* capture = static_cast<RichCallbackCapture*>(user_data);
  ASSERT_NE(snapshot, nullptr);
  ++capture->calls;
  capture->header = snapshot->header;
  capture->channels = snapshot->channel_count;
  capture->contributors = snapshot->contributor_count;
  capture->utf8_bytes = snapshot->utf8_bytes;
  // The documented callback lifetime permits API reentry.  This deliberately
  // does not touch snapshot storage after the callback returns.
  EXPECT_GT(sb_read_rich_monotonic_seconds(), 0.0);
  sb_register_on_rich_realization(snapshot->header.receiver_space,
                                  CaptureRichSnapshot, user_data);
  capture->registration_reentered = true;
}
}  // namespace

TEST(CApiRichRealizationTest, NegotiatesCopiesFrozenTransactionAndRetires) {
  sb_rich_realization_feature_t feature{};
  const uint32_t required = sb_get_rich_realization_feature(nullptr, 0);
  ASSERT_EQ(required, sizeof(feature));
  EXPECT_EQ(sb_get_rich_realization_feature(&feature, required - 1), required);
  ASSERT_EQ(sb_get_rich_realization_feature(&feature, sizeof(feature)),
            required);
  EXPECT_EQ(feature.feature_version, SB_RICH_REALIZATION_FEATURE_VERSION);
  EXPECT_EQ(feature.snapshot_channels_offset,
            offsetof(sb_rich_realization_snapshot_t, channels));
  EXPECT_EQ(feature.channel_contributor_offset_offset,
            offsetof(sb_rich_realization_channel_t, contributor_offset));
  EXPECT_NE(feature.monotonic_clock_id, 0u);
  EXPECT_NE(feature.monotonic_clock_epoch, 0u);
  EXPECT_EQ(sb_get_rich_realization_field_count(SB_RICH_LAYOUT_HEADER), 28u);
  for (uint32_t kind = SB_RICH_LAYOUT_FEATURE; kind <= SB_RICH_LAYOUT_COPY;
       ++kind) {
    const uint32_t count = sb_get_rich_realization_field_count(kind);
    ASSERT_GT(count, 0u);
    for (uint32_t index = 0; index < count; ++index)
      EXPECT_NE(sb_get_rich_realization_field_offset(kind, index), UINT32_MAX);
    EXPECT_EQ(sb_get_rich_realization_field_offset(kind, count), UINT32_MAX);
  }
  EXPECT_EQ(sb_activate_rich_realization(0, 0, feature.feature_version), 0u);
  EXPECT_EQ(sb_activate_rich_realization(0, 0, feature.feature_version + 1),
            0u);
  sb_rich_realization_validity_feature_t validity_feature{};
  const uint32_t validity_required =
      sb_get_rich_realization_validity_feature(nullptr, 0);
  ASSERT_EQ(validity_required, sizeof(validity_feature));
  EXPECT_EQ(sb_get_rich_realization_validity_feature(
                &validity_feature, validity_required - 1),
            validity_required);
  ASSERT_EQ(sb_get_rich_realization_validity_feature(
                &validity_feature, sizeof(validity_feature)),
            validity_required);
  EXPECT_EQ(validity_feature.feature_version,
            SB_RICH_REALIZATION_VALIDITY_FEATURE_VERSION);
  EXPECT_EQ(validity_feature.validity_size,
            sizeof(sb_rich_realization_validity_t));
  for (uint32_t kind = SB_RICH_VALIDITY_LAYOUT_FEATURE;
       kind <= SB_RICH_VALIDITY_LAYOUT_RESULT; ++kind) {
    const uint32_t count = sb_get_rich_realization_validity_field_count(kind);
    ASSERT_GT(count, 0u);
    for (uint32_t index = 0; index < count; ++index) {
      EXPECT_NE(sb_get_rich_realization_validity_field_offset(kind, index),
                UINT32_MAX);
    }
    EXPECT_EQ(sb_get_rich_realization_validity_field_offset(kind, count),
              UINT32_MAX);
  }

  const sb_session_t session = sb_create_session();
  const sb_space_t source = sb_add_space(session);
  const sb_space_t target = sb_add_space(session);
  double poly[] = {0, 0, 10, 0, 10, 10, 0, 10};
  const sb_element_t source_room = sb_add_element(source, 0);
  const sb_element_t target_room = sb_add_element(target, 0);
  sb_set_element_shared_id(source, source_room, INT64_C(900000000000000001));
  sb_set_element_shared_id(target, target_room, INT64_C(900000000000000001));
  sb_set_element_polygon(source, source_room, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_set_element_polygon(target, target_room, SB_GEOMETRY_NAVIGABLE, poly, 4);
  const sb_element_t source_object = sb_add_element(source, source_room);
  const sb_element_t target_object = sb_add_element(target, target_room);
  sb_set_element_shared_id(source, source_object, INT64_C(900000000000000003));
  sb_set_element_shared_id(target, target_object, INT64_C(900000000000000003));
  sb_set_element_box(source, source_object, SB_GEOMETRY_OBSTACLE, 4, 4, 0, 6, 6,
                     1);
  sb_set_element_box(target, target_object, SB_GEOMETRY_OBSTACLE, 4, 4, 0, 6, 6,
                     1);
  const sb_actor_t source_actor = sb_create_actor(source);
  const sb_actor_t receiver_actor = sb_create_actor(target);
  PublishPose(target, receiver_actor, 0.0, 0.0, 0.0);
  sb_add_anchor(source, source_actor, "root", 0, 0, 0, nullptr);
  sb_add_anchor(target, receiver_actor, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(source, source_actor, source_room);
  sb_set_actor_element(target, receiver_actor, target_room);
  sb_bind(source, source_actor, "root", SB_KERNEL_SPACING);

  RichCallbackCapture callback;
  sb_register_on_rich_realization(target, CaptureRichSnapshot, &callback);
  const sb_rich_realization_token_t token = sb_activate_rich_realization(
      target, receiver_actor, feature.feature_version);
  ASSERT_NE(token, 0u);
  sb_update_actor_state(source, source_actor, 3, 5, 0);
  ASSERT_EQ(callback.calls, 1);
  EXPECT_TRUE(callback.registration_reentered);
  EXPECT_EQ(callback.header.producer_lifetime, token);
  EXPECT_EQ(callback.header.status, SB_RICH_SNAPSHOT_COMPLETE);
  EXPECT_EQ(callback.header.source_timing_mode, SB_RICH_SOURCE_LEGACY_UNKNOWN);
  EXPECT_EQ(callback.header.source_time_known, 0u);
  EXPECT_EQ(callback.header.monotonic_clock_epoch,
            feature.monotonic_clock_epoch);
  EXPECT_EQ(callback.header.monotonic_clock_id, feature.monotonic_clock_id);
  EXPECT_EQ(callback.header.source_clock_known, 1u);
  EXPECT_GT(callback.header.receiver_ingress_monotonic_ns, 0u);
  EXPECT_GE(callback.header.receiver_solve_monotonic_ns,
            callback.header.receiver_ingress_monotonic_ns);
  ASSERT_GT(callback.channels, 0u);
  ASSERT_GT(callback.contributors, 0u);

  // Preflight must not consume a snapshot or partially write the arrays.
  sb_rich_realization_copy_t copy{};
  copy.struct_size = sizeof(copy);
  EXPECT_EQ(sb_poll_rich_realization(token, &copy),
            SB_RICH_RESULT_BUFFER_TOO_SMALL);
  std::vector<sb_rich_realization_channel_t> channels(copy.required_channels);
  std::vector<sb_rich_realization_contributor_t> contributors(
      copy.required_contributors);
  std::vector<uint8_t> utf8(copy.required_utf8_bytes);
  copy.channel_capacity = channels.size();
  copy.contributor_capacity = contributors.size();
  copy.utf8_capacity = utf8.size();
  copy.channels = channels.data();
  copy.contributors = contributors.data();
  copy.utf8 = utf8.data();
  ASSERT_EQ(sb_poll_rich_realization(token, &copy), SB_RICH_RESULT_OK);
  EXPECT_EQ(copy.header.producer_lifetime, callback.header.producer_lifetime);
  EXPECT_EQ(copy.header.revision, callback.header.revision);
  EXPECT_EQ(copy.header.receiver_ingress_monotonic_ns,
            callback.header.receiver_ingress_monotonic_ns);
  EXPECT_EQ(copy.required_channels, callback.channels);
  EXPECT_EQ(copy.required_contributors, callback.contributors);
  bool saw_large_id = false;
  for (const auto& contributor : contributors) {
    saw_large_id |= contributor.referent_id == INT64_C(900000000000000003);
  }
  EXPECT_TRUE(saw_large_id);
  // Validity is an independent historical/current comparison; it does not
  // consume the rich poll cursor or fabricate a newer receipt.
  sb_rich_realization_validity_t validity{};
  ASSERT_EQ(sb_get_rich_realization_validity(
                token, copy.header.revision, SB_RICH_VALIDITY_DOMAIN_POLICY,
                &validity, sizeof(validity)),
            SB_RICH_RESULT_OK);
  EXPECT_EQ(validity.producer_lifetime, token);
  EXPECT_EQ(validity.revision, copy.header.revision);
  EXPECT_EQ(validity.valid, 1u);
  EXPECT_NE(validity.captured_known_mask & SB_RICH_VALIDITY_DOMAIN_POLICY, 0u);
  EXPECT_EQ(sb_get_rich_realization_validity(
                token, copy.header.revision, SB_RICH_VALIDITY_DOMAIN_TREE,
                &validity, sizeof(validity)),
            SB_RICH_RESULT_OK);
  EXPECT_EQ(validity.valid, 1u);
  EXPECT_EQ(validity.unchecked_mask & SB_RICH_VALIDITY_DOMAIN_TREE, 0u);
  EXPECT_GT(validity.current_tree_revision, 0u);
  EXPECT_EQ(validity.current_tree_revision, validity.captured_tree_revision);
  EXPECT_EQ(sb_get_rich_realization_validity(
                token, copy.header.revision, SB_RICH_VALIDITY_DOMAIN_SUPPORT,
                &validity, sizeof(validity)),
            SB_RICH_RESULT_OK);
  EXPECT_EQ(validity.valid, 1u);
  EXPECT_GT(validity.current_support_revision, 0u);
  EXPECT_EQ(validity.current_support_revision, validity.captured_support_revision);
  EXPECT_EQ(sb_get_rich_realization_validity(
                token, copy.header.revision,
                SB_RICH_VALIDITY_DOMAIN_RECEIVER_INPUT, &validity, sizeof(validity)),
            SB_RICH_RESULT_UNAVAILABLE);
  EXPECT_NE(validity.unchecked_mask & SB_RICH_VALIDITY_DOMAIN_RECEIVER_INPUT, 0u);
  EXPECT_EQ(sb_get_rich_realization_validity(
                token, copy.header.revision, 0, &validity, sizeof(validity)),
            SB_RICH_RESULT_INVALID_ARGUMENT);
  EXPECT_EQ(sb_get_rich_realization_validity(
                token, copy.header.revision,
                SB_RICH_VALIDITY_DOMAIN_POLICY | (1u << 31), &validity,
                sizeof(validity)),
            SB_RICH_RESULT_INVALID_ARGUMENT);
  EXPECT_EQ(sb_poll_rich_realization(token, &copy),
            SB_RICH_RESULT_NO_NEW_SNAPSHOT);

  // A policy mutation invalidates the historical solve immediately, without
  // a new source observation or a changed native receipt/revision.
  sb_set_realization_policy(target, SB_REALIZATION_POLICY_UNIFORM);
  ASSERT_EQ(sb_get_rich_realization_validity(
                token, copy.header.revision, SB_RICH_VALIDITY_DOMAIN_POLICY,
                &validity, sizeof(validity)),
            SB_RICH_RESULT_OK);
  EXPECT_EQ(validity.valid, 0u);
  EXPECT_NE(validity.changed_mask & SB_RICH_VALIDITY_DOMAIN_POLICY, 0u);
  EXPECT_EQ(sb_poll_rich_realization(token, &copy),
            SB_RICH_RESULT_NO_NEW_SNAPSHOT);

  // The consumer-owned first copy cannot change when a later source update
  // replaces Core's live snapshot; its ingress/revision remain diagnostic
  // facts about the first accepted update.
  const sb_rich_realization_header_t first_header = copy.header;
  sb_update_actor_state(source, source_actor, 3.2, 5, 0);
  ASSERT_EQ(callback.calls, 2);
  EXPECT_EQ(copy.header.revision, first_header.revision);
  EXPECT_EQ(copy.header.receiver_ingress_monotonic_ns,
            first_header.receiver_ingress_monotonic_ns);
  ASSERT_EQ(sb_poll_rich_realization(token, &copy), SB_RICH_RESULT_OK);
  EXPECT_GT(copy.header.revision, first_header.revision);
  EXPECT_GT(copy.header.receiver_ingress_monotonic_ns,
            first_header.receiver_ingress_monotonic_ns);
  ASSERT_EQ(sb_get_rich_realization_validity(
                token, copy.header.revision, SB_RICH_VALIDITY_DOMAIN_POLICY,
                &validity, sizeof(validity)),
            SB_RICH_RESULT_OK);
  EXPECT_EQ(validity.valid, 1u);
  const uint64_t uniform_policy_revision = validity.current_policy_revision;
  sb_set_realization_policy(target, SB_REALIZATION_POLICY_UNIFORM);
  ASSERT_EQ(sb_get_rich_realization_validity(
                token, copy.header.revision, SB_RICH_VALIDITY_DOMAIN_POLICY,
                &validity, sizeof(validity)),
            SB_RICH_RESULT_OK);
  EXPECT_EQ(validity.valid, 1u);
  EXPECT_EQ(validity.current_policy_revision, uniform_policy_revision);

  sb_retire_rich_realization(token);
  ASSERT_EQ(callback.calls, 3);
  EXPECT_EQ(callback.header.status, SB_RICH_SNAPSHOT_EMPTY);
  EXPECT_EQ(callback.header.producer_lifetime, token);
  EXPECT_GT(callback.header.revision, copy.header.revision);
  EXPECT_EQ(callback.header.source_actor_space, -1);
  EXPECT_EQ(callback.header.source_actor_element, -1);
  EXPECT_EQ(callback.header.relay_source_space, -1);
  EXPECT_EQ(sb_get_rich_realization_validity(
                token, callback.header.revision, SB_RICH_VALIDITY_DOMAIN_POLICY,
                &validity, sizeof(validity)),
            SB_RICH_RESULT_UNAVAILABLE);
  ASSERT_EQ(sb_poll_rich_realization(token, &copy), SB_RICH_RESULT_OK);
  EXPECT_EQ(copy.header.status, SB_RICH_SNAPSHOT_EMPTY);
  EXPECT_EQ(copy.header.source_actor_space, -1);
  EXPECT_EQ(copy.header.source_actor_element, -1);
  EXPECT_EQ(copy.header.relay_source_space, -1);
  EXPECT_EQ(sb_release_rich_realization(token), 1);
  EXPECT_EQ(sb_poll_rich_realization(token, &copy),
            SB_RICH_RESULT_UNKNOWN_TOKEN);
  const sb_rich_realization_token_t replacement = sb_activate_rich_realization(
      target, receiver_actor, feature.feature_version);
  EXPECT_NE(replacement, 0u);
  EXPECT_NE(replacement, token);
  sb_retire_rich_realization(replacement);
  EXPECT_EQ(sb_release_rich_realization(replacement), 1);
  sb_destroy_session(session);
}

TEST(CApiRichRealizationTest, OwnedSceneWritersInvalidateWithoutNewReceipt) {
  auto f = SetupTwoSpaces();
  RichCallbackCapture capture;
  sb_register_on_rich_realization(f.space_b, CaptureRichSnapshot, &capture);
  const auto token = sb_activate_rich_realization(
      f.space_b, f.avatar_b, SB_RICH_REALIZATION_FEATURE_VERSION);
  ASSERT_NE(token, 0u);
  sb_set_min_move(f.space_a, 0);
  sb_set_min_yaw(f.space_a, 0);
  constexpr uint32_t tree = SB_RICH_VALIDITY_DOMAIN_TREE;
  constexpr uint32_t support = SB_RICH_VALIDITY_DOMAIN_SUPPORT;
  auto check = [&](const std::function<void()>& mutation, uint32_t changed) {
    sb_update_actor_state(f.space_a, f.actor_a, 3, 5, 0);
    const auto header = capture.header;
    ASSERT_GT(header.revision, 0u);
    sb_rich_realization_validity_t value{};
    ASSERT_EQ(sb_get_rich_realization_validity(token, header.revision,
        tree | support, &value, sizeof(value)), SB_RICH_RESULT_OK);
    ASSERT_EQ(value.valid, 1u);
    mutation();
    ASSERT_EQ(sb_get_rich_realization_validity(token, header.revision,
        tree | support, &value, sizeof(value)), SB_RICH_RESULT_OK);
    EXPECT_EQ(value.changed_mask & (tree | support), changed);
    EXPECT_EQ(value.valid, changed == 0 ? 1u : 0u);
    EXPECT_EQ(capture.header.revision, header.revision);
    EXPECT_EQ(capture.header.receiver_ingress_monotonic_ns,
              header.receiver_ingress_monotonic_ns);
  };
  double matrix[16];
  // Transforms are parent-relative: the board (a +-2 box) at (6, 6) stays inside room B (0..12). At (0.2, 0.3) it
  // stuck out of the room, which starts a background scope-extension build whose install advances the tree and
  // support revisions at a moment the test does not control, so a later check saw them changed (flaky).
  MakeTranslation(matrix, 6, 6);
  double polygon[] = {0, 0, 1, 0, 1, 1, 0, 1};
  double vertices[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1};
  int32_t indices[] = {0, 1, 2, 0, 1, 3, 1, 2, 3, 2, 0, 3};
  int32_t sizes[] = {3, 3, 3, 3};
  check([&] { sb_set_element_transform(f.space_b, f.board_b, matrix); }, tree | support);
  check([&] { sb_set_element_transform(f.space_b, f.board_b, matrix); }, 0);
  check([&] { sb_set_element_tag(f.space_b, f.board_b, "board"); }, tree | support);
  check([&] { sb_set_element_shared_id(f.space_b, f.board_b, 2001); }, tree);
  check([&] { sb_set_element_polygon(f.space_b, f.board_b,
      SB_GEOMETRY_SURFACE, polygon, 4); }, tree | support);
  check([&] { sb_add_sub_region(f.space_b, f.board_b, "patch", "writable",
      polygon, 4, nullptr); }, support);
  // Element anchors retain their legacy column-major layout.
  double anchor[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, .2, .3, .4, 1};
  check([&] { sb_add_element_anchor(f.space_b, f.board_b, "tip", anchor); }, tree | support);
  check([&] { sb_set_element_surface(f.space_b, f.board_b, vertices, 3,
      SB_GEOMETRY_SURFACE); }, tree | support);
  check([&] { sb_set_element_polytope(f.space_b, f.board_b, vertices, 4,
      indices, sizes, 4, SB_GEOMETRY_OBSTACLE); }, tree | support);
  check([&] { sb_set_element_box(f.space_b, f.board_b, SB_GEOMETRY_OBSTACLE,
      0, 0, 0, 1, 1, 1); }, tree | support);
  double collider[] = {0, 0, 0, 2, 2, 2};
  check([&] { sb_set_element_collider(f.space_b, f.board_b, SB_COLLIDER_BOX,
      collider, 6); }, 0);  // explicit geometry wins
  check([&] { sb_set_element_collider(f.space_b, f.writable_b, SB_COLLIDER_BOX,
      collider, 6); }, tree | support);
  check([&] { sb_set_element_box(f.space_b, f.board_b, SB_GEOMETRY_OBSTACLE,
      2, 0, 0, 1, 1, 1); }, 0);
  check([&] {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    double bad[] = {nan, 0, 1, 0, 1, 1};
    sb_set_element_polygon(f.space_b, f.board_b, SB_GEOMETRY_SURFACE, bad, 3);
    sb_set_element_surface(f.space_b, f.board_b, vertices, 65537, SB_GEOMETRY_SURFACE);
    const int32_t negative_size[] = {-1};
    sb_set_element_polytope(f.space_b, f.board_b, vertices, 4, indices,
        negative_size, 1, SB_GEOMETRY_OBSTACLE);
    const int32_t bad_index[] = {0, 1, 4};
    sb_set_element_polytope(f.space_b, f.board_b, vertices, 4, bad_index,
        sizes, 1, SB_GEOMETRY_OBSTACLE);
  }, 0);
  sb_destroy_session(f.session);
}

TEST(CApiSupportCorrespondenceTest, ReentryAndContendedSceneRejectWithoutDeadlock) {
  auto f = SetupTwoSpaces();
  struct Gate {
    sb_session_t session;
    sb_support_correspondence_t pair;
    std::promise<void> entered;
    std::mutex mutex;
    std::condition_variable condition;
    bool release = false;
    bool first = true;
    int calls = 0;
  } gate;
  gate.session = f.session;
  gate.pair = {f.space_a, f.board_a, f.space_b, f.board_b};
  auto entered = gate.entered.get_future();
  sb_register_on_relation_event(f.space_a,
      [](const sb_relation_event_t*, void* data) {
        auto* gate = static_cast<Gate*>(data);
        ++gate->calls;
        EXPECT_EQ(sb_register_support_correspondences(gate->session,
            &gate->pair, 1), 0);
        if (!gate->first) return;
        gate->first = false;
        gate->entered.set_value();
        std::unique_lock<std::mutex> lock(gate->mutex);
        gate->condition.wait(lock, [&] { return gate->release; });
      }, &gate);
  std::thread update([&] { sb_update_actor_state(f.space_a, f.actor_a, 5, 5, 0); });
  const auto ready = entered.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(ready, std::future_status::ready);
  if (ready == std::future_status::ready) {
    EXPECT_EQ(sb_register_support_correspondences(f.session, &gate.pair, 1), 0);
  }
  {
    std::lock_guard<std::mutex> lock(gate.mutex);
    gate.release = true;
  }
  gate.condition.notify_all();
  update.join();
  EXPECT_GT(gate.calls, 0);
  const auto registration = sb_register_support_correspondences(f.session, &gate.pair, 1);
  EXPECT_NE(registration, 0);
  sb_dispose_support_correspondences(registration);
  sb_destroy_session(f.session);
}

TEST(CApiRichRealizationTest, SessionTeardownRetiresThenReleasesTokens) {
  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  const sb_actor_t actor = sb_create_actor(space);
  ASSERT_NE(actor, 0);
  RichCallbackCapture callback;
  sb_register_on_rich_realization(space, CaptureRichSnapshot, &callback);
  const sb_rich_realization_token_t token = sb_activate_rich_realization(
      space, actor, SB_RICH_REALIZATION_FEATURE_VERSION);
  ASSERT_NE(token, 0u);
  sb_destroy_session(session);
  ASSERT_EQ(callback.calls, 1);
  EXPECT_EQ(callback.header.status, SB_RICH_SNAPSHOT_EMPTY);
  sb_rich_realization_copy_t copy{};
  copy.struct_size = sizeof(copy);
  EXPECT_EQ(sb_poll_rich_realization(token, &copy),
            SB_RICH_RESULT_UNKNOWN_TOKEN);
}

TEST(CApiRichRealizationTest, GlobalCleanupRetiresAndReleasesTokens) {
  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  const sb_actor_t actor = sb_create_actor(space);
  RichCallbackCapture callback;
  sb_register_on_rich_realization(space, CaptureRichSnapshot, &callback);
  const auto token = sb_activate_rich_realization(
      space, actor, SB_RICH_REALIZATION_FEATURE_VERSION);
  ASSERT_NE(token, 0u);
  sb_cleanup_all();
  ASSERT_EQ(callback.calls, 1);
  EXPECT_EQ(callback.header.status, SB_RICH_SNAPSHOT_EMPTY);
  sb_rich_realization_copy_t copy{};
  copy.struct_size = sizeof(copy);
  EXPECT_EQ(sb_poll_rich_realization(token, &copy),
            SB_RICH_RESULT_UNKNOWN_TOKEN);
}

TEST(CApiRichRealizationTest,
     TimestampResetAndMalformedObservationKeepCausalRevision) {
  const sb_session_t session = sb_create_session();
  const sb_space_t source = sb_add_space(session);
  const sb_space_t target = sb_add_space(session);
  double poly[] = {0, 0, 10, 0, 10, 10, 0, 10};
  const sb_element_t room_a = sb_add_element(source, 0);
  const sb_element_t room_b = sb_add_element(target, 0);
  sb_set_element_shared_id(source, room_a, 100);
  sb_set_element_shared_id(target, room_b, 100);
  sb_set_element_polygon(source, room_a, SB_GEOMETRY_NAVIGABLE, poly, 4);
  sb_set_element_polygon(target, room_b, SB_GEOMETRY_NAVIGABLE, poly, 4);
  const sb_element_t object_a = sb_add_element(source, room_a);
  const sb_element_t object_b = sb_add_element(target, room_b);
  sb_set_element_shared_id(source, object_a, 200);
  sb_set_element_shared_id(target, object_b, 200);
  sb_set_element_box(source, object_a, SB_GEOMETRY_OBSTACLE, 4, 4, 0, 6, 6, 1);
  sb_set_element_box(target, object_b, SB_GEOMETRY_OBSTACLE, 4, 4, 0, 6, 6, 1);
  const sb_actor_t source_actor = sb_create_actor(source);
  const sb_actor_t receiver_actor = sb_create_actor(target);
  PublishPose(target, receiver_actor, 0.0, 0.0, 0.0);
  sb_add_anchor(source, source_actor, "root", 0, 0, 0, nullptr);
  sb_add_anchor(target, receiver_actor, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(source, source_actor, room_a);
  sb_set_actor_element(target, receiver_actor, room_b);
  sb_bind(source, source_actor, "root", SB_KERNEL_SPACING);
  ASSERT_EQ(sb_enable_actor_observation_frames(session), 1);
  RichCallbackCapture callback;
  sb_register_on_rich_realization(target, CaptureRichSnapshot, &callback);
  const auto token = sb_activate_rich_realization(
      target, receiver_actor, SB_RICH_REALIZATION_FEATURE_VERSION);
  ASSERT_NE(token, 0u);

  const uint64_t epoch =
      sb_begin_actor_observation_epoch(source, source_actor, "root");
  ASSERT_NE(epoch, 0u);
  ASSERT_EQ(callback.calls, 1);
  EXPECT_EQ(callback.header.status, SB_RICH_SNAPSHOT_EMPTY);
  EXPECT_EQ(callback.header.source_timing_mode,
            SB_RICH_SOURCE_OBSERVATION_RESET);
  EXPECT_EQ(callback.header.source_time_known, 0u);
  EXPECT_EQ(callback.header.source_epoch, epoch);

  sb_actor_observation_t good[] = {{"root", 2, 2, 0, 0}};
  ASSERT_EQ(sb_apply_actor_observation_frame(source, source_actor, epoch, 0,
                                             12.5, good, 1),
            1);
  ASSERT_EQ(callback.calls, 2);
  const sb_rich_realization_header_t timestamped = callback.header;
  EXPECT_EQ(timestamped.source_timing_mode,
            SB_RICH_SOURCE_TIMESTAMPED_OBSERVATION);
  EXPECT_EQ(timestamped.source_time_known, 1u);
  EXPECT_DOUBLE_EQ(timestamped.source_time_seconds, 12.5);
  EXPECT_EQ(timestamped.source_epoch, epoch);
  // The source's caller admitted frame sequence is preserved separately from
  // the relation transport sequence. BeginActorObservationEpoch emitted the
  // reset as transport sequence 1, so the first actual sample is delivery 2.
  EXPECT_EQ(timestamped.source_observation_sequence_known, 1u);
  EXPECT_EQ(timestamped.source_observation_sequence, 0);
  EXPECT_EQ(timestamped.source_sequence, 2);
  EXPECT_GT(timestamped.receiver_ingress_monotonic_ns, 0u);

  sb_rich_realization_copy_t copy{};
  copy.struct_size = sizeof(copy);
  EXPECT_EQ(sb_poll_rich_realization(token, &copy),
            SB_RICH_RESULT_BUFFER_TOO_SMALL);
  std::vector<sb_rich_realization_channel_t> channels(copy.required_channels);
  std::vector<sb_rich_realization_contributor_t> contributors(
      copy.required_contributors);
  std::vector<uint8_t> bytes(copy.required_utf8_bytes);
  copy.channel_capacity = channels.size();
  copy.channels = channels.data();
  copy.contributor_capacity = contributors.size();
  copy.contributors = contributors.data();
  copy.utf8_capacity = bytes.size();
  copy.utf8 = bytes.data();
  ASSERT_EQ(sb_poll_rich_realization(token, &copy), SB_RICH_RESULT_OK);
  EXPECT_EQ(copy.header.revision, timestamped.revision);

  sb_actor_observation_t malformed[] = {{"root", 9, 9, 0, 0},
                                        {"unknown", 0, 0, 0, NAN}};
  EXPECT_EQ(sb_apply_actor_observation_frame(source, source_actor, epoch, 1,
                                             12.6, malformed, 2),
            0);
  EXPECT_EQ(callback.calls, 2);
  EXPECT_EQ(sb_poll_rich_realization(token, &copy),
            SB_RICH_RESULT_NO_NEW_SNAPSHOT);

  // A rejected frame cannot advance either provenance watermark. A later
  // caller sequence remains its original sample identity, independent of the
  // monotonically increasing relation delivery sequence.
  sb_actor_observation_t later[] = {{"root", 3, 2, 0, 0}};
  ASSERT_EQ(sb_apply_actor_observation_frame(source, source_actor, epoch, 17,
                                             12.7, later, 1),
            1);
  ASSERT_EQ(callback.calls, 3);
  EXPECT_EQ(callback.header.source_observation_sequence_known, 1u);
  EXPECT_EQ(callback.header.source_observation_sequence, 17);
  EXPECT_GT(callback.header.source_sequence, timestamped.source_sequence);

  // Clearing an anchor is a timestamped control mutation, not a newly
  // committed source sample. It must not carry the prior frame's sequence.
  ASSERT_EQ(
      sb_clear_actor_observation_anchor(source, source_actor, epoch, "root"),
      1);
  ASSERT_EQ(callback.calls, 4);
  EXPECT_EQ(callback.header.source_timing_mode,
            SB_RICH_SOURCE_TIMESTAMPED_OBSERVATION);
  EXPECT_EQ(callback.header.source_observation_sequence_known, 0u);
  EXPECT_EQ(callback.header.source_observation_sequence, 0);

  const uint64_t next =
      sb_begin_actor_observation_epoch(source, source_actor, "root");
  ASSERT_GT(next, epoch);
  ASSERT_EQ(callback.calls, 5);
  EXPECT_EQ(callback.header.status, SB_RICH_SNAPSHOT_EMPTY);
  EXPECT_EQ(callback.header.source_timing_mode,
            SB_RICH_SOURCE_OBSERVATION_RESET);
  EXPECT_EQ(callback.header.source_epoch, next);
  EXPECT_EQ(callback.header.source_observation_sequence_known, 0u);
  EXPECT_EQ(callback.header.source_observation_sequence, 0);
  EXPECT_GT(callback.header.revision, timestamped.revision);
  sb_retire_rich_realization(token);
  EXPECT_EQ(sb_release_rich_realization(token), 1);
  sb_destroy_session(session);
}
