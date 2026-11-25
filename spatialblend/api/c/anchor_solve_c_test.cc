// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// ABI 4.11: affordance poses on interaction-role children and the inner level of an anchor's
// proposal through the public C API.
#include <gtest/gtest.h>

#include <cstring>

#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/api/c/spatialblend_observability.h"

namespace {

TEST(AnchorSolveC, AbiMinorStructAndFieldTable) {
  EXPECT_EQ(sb_abi_minor_version(), 14u);
  EXPECT_EQ(sb_sizeof(SB_STRUCT_ANCHOR_SOLVE), sizeof(sb_anchor_solve_t));
  EXPECT_EQ(sb_get_struct_field_offset(SB_STRUCT_ANCHOR_SOLVE, 0), offsetof(sb_anchor_solve_t, struct_size));
  EXPECT_EQ(sb_get_struct_field_offset(SB_STRUCT_ANCHOR_SOLVE, 8), offsetof(sb_anchor_solve_t, driving_pose_tag));
  EXPECT_NE(SB_ANCHOR_PROPOSAL_FLAG_INNER & SB_ANCHOR_PROPOSAL_FLAG_FACING_YAW, SB_ANCHOR_PROPOSAL_FLAG_INNER);
  EXPECT_EQ(SB_PROPOSAL_CAUSE_FOLD, 0x80000u);
}

TEST(AnchorSolveC, SetAndClearAnAffordancePoseAndReadTheSolveOfAnActorWithNothingSolved) {
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  sb_element_t chair = sb_add_element(space, 0);
  sb_element_t seated = sb_add_element(space, chair);
  EXPECT_EQ(sb_set_element_affordance_pose(space, seated, "pelvis", 0, "seat"), 0);
  EXPECT_EQ(sb_set_element_affordance_pose(space, seated, "pelvis", 1, nullptr), 0);  // ACTOR_ROOT: no part
  EXPECT_EQ(sb_set_element_affordance_pose(space, seated, "pelvis", 0, nullptr), -1);  // PART needs a part
  EXPECT_EQ(sb_set_element_affordance_pose(space, seated, "pelvis", 7, "seat"), -1);   // unknown support
  EXPECT_EQ(sb_set_element_affordance_pose(space, seated, nullptr, 0, nullptr), 0);    // clear
  sb_actor_t actor = sb_create_actor(space);
  sb_anchor_solve_t out;
  std::memset(&out, 0, sizeof out);
  out.struct_size = sizeof out - 1;
  EXPECT_EQ(sb_read_anchor_solve(space, actor, 0, &out), -1);  // wrong struct_size
  out.struct_size = sizeof out;
  // The root (anchor 0) of an actor nothing was solved for: no inner level, no Facing yaw.
  EXPECT_EQ(sb_read_anchor_solve(space, actor, 0, &out), 1);
  EXPECT_EQ(out.flags, 0u);
  EXPECT_EQ(out.object_shared_id, 0);
  EXPECT_EQ(out.driving_pose_tag[0], '\0');
  EXPECT_EQ(sb_read_anchor_solve(space, actor, 77, &out), 0);  // no such anchor in the set
  sb_destroy_session(s);
}

// ABI 4.12: the signed, unwrapped yaw turn of an anchor's proposal (sb_read_anchor_turn / sb_anchor_turn_t).
TEST(AnchorTurnC, StructFieldTableAndReadOfAnActorWithNothingSolved) {
  EXPECT_EQ(sb_sizeof(SB_STRUCT_ANCHOR_TURN), sizeof(sb_anchor_turn_t));
  EXPECT_EQ(sizeof(sb_anchor_turn_t), 32u);
  EXPECT_EQ(sb_get_struct_field_offset(SB_STRUCT_ANCHOR_TURN, 0), offsetof(sb_anchor_turn_t, struct_size));
  EXPECT_EQ(sb_get_struct_field_offset(SB_STRUCT_ANCHOR_TURN, 2), offsetof(sb_anchor_turn_t, flags));
  EXPECT_EQ(sb_get_struct_field_offset(SB_STRUCT_ANCHOR_TURN, 4), offsetof(sb_anchor_turn_t, yaw));
  EXPECT_EQ(sb_get_struct_field_offset(SB_STRUCT_ANCHOR_TURN, 5), offsetof(sb_anchor_turn_t, yaw_turn));
  // sb_anchor_proposal_t keeps its 4.11 size.
  EXPECT_EQ(sb_sizeof(SB_STRUCT_ANCHOR_PROPOSAL), sizeof(sb_anchor_proposal_t));
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  sb_actor_t actor = sb_create_actor(space);
  sb_anchor_turn_t out;
  std::memset(&out, 0, sizeof out);
  out.struct_size = sizeof out - 1;
  EXPECT_EQ(sb_read_anchor_turn(space, actor, 0, &out), -1);  // wrong struct_size
  EXPECT_EQ(sb_read_anchor_turn(space, actor, 0, nullptr), -1);
  out.struct_size = sizeof out;
  // The root of an actor nothing was solved for: no turn.
  EXPECT_EQ(sb_read_anchor_turn(space, actor, 0, &out), 1);
  EXPECT_EQ(out.anchor_id, 0u);
  EXPECT_EQ(out.yaw_turn, 0.0);
  EXPECT_EQ(sb_read_anchor_turn(space, actor, 77, &out), 0);  // no such anchor in the set
  sb_destroy_session(s);
}

// ABI 4.14: the per-anchor replay through the chart (sb_read_anchor_chart / sb_anchor_chart_t). Its values are tested
// through the native proposal (containment_transfer_test.cc); here the layout and the absent / invalid reads.
TEST(AnchorChartC, StructFieldTableAndReadOfAnActorWithNoChart) {
  EXPECT_EQ(sb_sizeof(SB_STRUCT_ANCHOR_CHART), sizeof(sb_anchor_chart_t));
  EXPECT_EQ(sb_get_struct_field_count(SB_STRUCT_ANCHOR_CHART), 10);
  EXPECT_EQ(sb_get_struct_field_offset(SB_STRUCT_ANCHOR_CHART, 0), offsetof(sb_anchor_chart_t, struct_size));
  EXPECT_EQ(sb_get_struct_field_offset(SB_STRUCT_ANCHOR_CHART, 4), offsetof(sb_anchor_chart_t, leaf_shared_id));
  EXPECT_EQ(sb_get_struct_field_offset(SB_STRUCT_ANCHOR_CHART, 8), offsetof(sb_anchor_chart_t, orientation));
  EXPECT_EQ(sb_get_struct_field_offset(SB_STRUCT_ANCHOR_CHART, 9), offsetof(sb_anchor_chart_t, revision));
  // The proposal structs keep their size: the chart is a separate read.
  EXPECT_EQ(sb_sizeof(SB_STRUCT_ANCHOR_PROPOSAL), sizeof(sb_anchor_proposal_t));
  sb_session_t s = sb_create_session();
  sb_space_t space = sb_add_space(s);
  sb_actor_t actor = sb_create_actor(space);
  sb_anchor_chart_t out;
  std::memset(&out, 0, sizeof out);
  out.struct_size = sizeof out - 1;
  EXPECT_EQ(sb_read_anchor_chart(space, actor, 0, &out), -1);  // wrong struct_size
  EXPECT_EQ(sb_read_anchor_chart(space, actor, 0, nullptr), -1);
  out.struct_size = sizeof out;
  EXPECT_EQ(sb_read_anchor_chart(space, actor, 0, &out), 0);   // the root has no chart transform
  EXPECT_EQ(sb_read_anchor_chart(space, actor, 77, &out), 0);  // no such anchor in the set
  sb_destroy_session(s);
}

}  // namespace
