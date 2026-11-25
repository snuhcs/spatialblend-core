// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// The source binding revision reaches every rich contributor.
//
// The binding identity + revision travels with each relation payload
// (source Binding -> RelationPayload -> GroundAffordanceTarget -> rich
// contributor), is marked known when the source can certify it (bindings
// made through the owned C API), and can be supplied on injected payloads
// (sb_relation_payload_t, with the struct size before the binding fields still accepted).

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "spatialblend/api/c/c_test_embodied.h"
#include "spatialblend/api/c/spatialblend.h"

namespace {

struct Captured {
  std::string anchor;
  int32_t family = -1;
  sb_rich_realization_contributor_t contributor{};
};

std::vector<Captured> g_rows;
int g_snapshots = 0;

void Capture(const sb_rich_realization_snapshot_t* s, void*) {
  ++g_snapshots;
  if (s->header.status != SB_RICH_SNAPSHOT_COMPLETE) return;
  g_rows.clear();
  for (uint32_t c = 0; c < s->channel_count; ++c) {
    const auto& ch = s->channels[c];
    const std::string anchor(reinterpret_cast<const char*>(s->utf8) +
                                 ch.anchor_offset,
                             ch.anchor_length);
    for (uint32_t k = 0; k < ch.contributor_count; ++k) {
      Captured row;
      row.anchor = anchor;
      row.family = ch.family;
      row.contributor = s->contributors[ch.contributor_offset + k];
      g_rows.push_back(row);
    }
  }
}

// Injection does not dispatch rich callbacks (no peer delivery happened);
// read the frozen snapshot through the poll/copy contract instead.
bool Poll(sb_rich_realization_token_t token) {
  std::vector<sb_rich_realization_channel_t> channels(16);
  std::vector<sb_rich_realization_contributor_t> contributors(64);
  std::vector<uint8_t> utf8(4096);
  sb_rich_realization_copy_t copy{};
  copy.struct_size = sizeof(copy);
  copy.channel_capacity = static_cast<uint32_t>(channels.size());
  copy.contributor_capacity = static_cast<uint32_t>(contributors.size());
  copy.utf8_capacity = static_cast<uint32_t>(utf8.size());
  copy.channels = channels.data();
  copy.contributors = contributors.data();
  copy.utf8 = utf8.data();
  if (sb_poll_rich_realization(token, &copy) != SB_RICH_RESULT_OK) return false;
  sb_rich_realization_snapshot_t view{};
  view.header = copy.header;
  view.channels = channels.data();
  view.contributors = contributors.data();
  view.utf8 = utf8.data();
  view.channel_count = copy.required_channels;
  view.contributor_count = copy.required_contributors;
  view.utf8_bytes = copy.required_utf8_bytes;
  Capture(&view, nullptr);
  return true;
}

const Captured* Find(int32_t family) {
  for (const auto& r : g_rows)
    if (r.family == family) return &r;
  return nullptr;
}

// Source space A binds root spacing and a 3-D "body" affordance pose on a
// seat surface; receiver space B mirrors it and activates rich realization.
struct TwoSpaces {
  sb_session_t session = sb_create_session();
  sb_space_t sa = sb_add_space(session);
  sb_space_t sb = sb_add_space(session);
  sb_actor_t source = 0, receiver = 0;
  sb_rich_realization_token_t token = 0;

  TwoSpaces() {
    const double polygon[] = {0, 0, 10, 0, 10, 10, 0, 10};
    sb_element_t rooms[2], seats[2];
    const sb_space_t spaces[2] = {sa, sb};
    const double seat_a[] = {4, 4, .9, 6, 4, .9, 6, 6, .9, 4, 6, .9};
    const double seat_b[] = {2, 6, .4, 4, 6, .4, 4, 8, .4, 2, 8, .4};
    for (int i = 0; i < 2; ++i) {
      rooms[i] = sb_add_element(spaces[i], 0);
      sb_set_element_shared_id(spaces[i], rooms[i], 901);
      sb_set_element_polygon(spaces[i], rooms[i], SB_GEOMETRY_NAVIGABLE,
                             polygon, 4);
      seats[i] = sb_add_element(spaces[i], rooms[i]);
      sb_set_element_shared_id(spaces[i], seats[i], 902);
      sb_set_element_surface(spaces[i], seats[i], i == 0 ? seat_a : seat_b, 4,
                             SB_GEOMETRY_SURFACE);
    }
    source = sb_create_actor(sa);
    spatialblend_c_test::PublishStartPose(sa, source);
    receiver = sb_create_actor(sb);
    spatialblend_c_test::PublishStartPose(sb, receiver);
    for (auto [space, actor] :
         {std::pair{sa, source}, std::pair{sb, receiver}}) {
      sb_add_anchor(space, actor, "root", 0, 0, 0, nullptr);
      EXPECT_EQ(sb_add_anchor_with_position_dimension(space, actor, "body",
                                                      0, 0, 0, nullptr, 3),
                1);
    }
    sb_set_actor_element(sa, source, rooms[0]);
    sb_set_actor_element(sb, receiver, rooms[1]);
    sb_bind(sa, source, "root", SB_KERNEL_SPACING);
    sb_bind(sa, source, "body", SB_KERNEL_AFFORDANCE_POSE);
    g_rows.clear();
    g_snapshots = 0;
    sb_register_on_rich_realization(sb, Capture, nullptr);
    token = sb_activate_rich_realization(sb, receiver, 1);
    EXPECT_NE(token, 0u);
  }
  // Body engaged on the seat surface: it TOUCHES it, on its plane z = 0.9.
  void Tick(double x) {
    sb_set_anchor_transform(sa, source, "body", 5, 4, 0.9, 0);
    sb_update_actor_state(sa, source, x, 4, 0);
  }
  ~TwoSpaces() {
    sb_retire_rich_realization(token);
    sb_release_rich_realization(token);
    sb_destroy_session(session);
  }
};

}  // namespace

TEST(BindingRevisionTest, EngagedContributorsCarryTheSourceBindingRevision) {
  TwoSpaces f;
  f.Tick(5.0);
  ASSERT_GT(g_snapshots, 0);
  const Captured* body = Find(SB_RICH_FAMILY_AFFORDANCE_POSE);
  const Captured* root = Find(SB_RICH_FAMILY_SPACING);
  ASSERT_NE(body, nullptr);
  ASSERT_NE(root, nullptr);
  // The engaged affordance contributor is fully admitted: mapped support
  // AND a known binding revision (what MotionGen's strict admission needs).
  EXPECT_EQ(body->contributor.disposition, SB_RICH_DISPOSITION_INCLUDED);
  EXPECT_EQ(body->contributor.has_mapped_support, 1u);
  EXPECT_EQ(body->contributor.binding_revision_known, 1u);
  EXPECT_NE(body->contributor.binding_revision, 0u);
  EXPECT_EQ(root->contributor.binding_revision_known, 1u);
  // Two bindings, two revisions.
  EXPECT_NE(root->contributor.binding_revision,
            body->contributor.binding_revision);
  const uint64_t body_revision = body->contributor.binding_revision;
  // Stable across ticks: the binding did not change.
  f.Tick(5.05);
  body = Find(SB_RICH_FAMILY_AFFORDANCE_POSE);
  ASSERT_NE(body, nullptr);
  EXPECT_EQ(body->contributor.binding_revision, body_revision);
  EXPECT_EQ(body->contributor.binding_revision_known, 1u);
}

TEST(BindingRevisionTest, InjectedPayloadSuppliesOrOmitsTheRevision) {
  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  const double polygon[] = {0, 0, 10, 0, 10, 10, 0, 10};
  const sb_element_t room = sb_add_element(space, 0);
  sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, polygon, 4);
  const sb_element_t point = sb_add_element(space, room);
  sb_set_element_shared_id(space, point, 301);
  const double m[16] = {1, 0, 0, 2, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1};
  sb_set_element_transform(space, point, m);
  const sb_element_t actor_elem = sb_add_element(space, room);
  sb_set_element_shared_id(space, actor_elem, 900);
  const sb_actor_t actor = sb_create_actor(space);
  spatialblend_c_test::PublishStartPose(space, actor);
  sb_add_anchor(space, actor, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(space, actor, actor_elem);
  g_rows.clear();
  sb_register_on_rich_realization(space, Capture, nullptr);
  const auto token = sb_activate_rich_realization(space, actor, 1);
  ASSERT_NE(token, 0u);

  sb_relation_payload_t p;
  std::memset(&p, 0, sizeof(p));
  p.struct_size = sizeof(p);
  p.version = SB_OBSERVABILITY_VERSION;
  p.anchor_name = "root";
  p.kernel_type = SB_KERNEL_SPACING;
  p.referent_shared_id = 301;
  p.proximity = 1.0;
  p.support_distance_m = -1;
  p.source_scope_diameter_m = -1;
  p.enabled = 1;
  p.source_binding_ordinal = 2;
  p.binding_revision = 77;
  p.has_binding_revision = 1;
  ASSERT_EQ(sb_inject_relation_payloads(space, actor, &p, 1, sizeof(p), 0.1),
            1);
  ASSERT_TRUE(Poll(token));
  const Captured* row = Find(SB_RICH_FAMILY_SPACING);
  ASSERT_NE(row, nullptr);
  EXPECT_EQ(row->contributor.binding_revision_known, 1u);
  EXPECT_EQ(row->contributor.binding_revision, 77u);

  // Not supplied: unknown.
  p.has_binding_revision = 0;
  ASSERT_EQ(sb_inject_relation_payloads(space, actor, &p, 1, sizeof(p), 0.2),
            1);
  ASSERT_TRUE(Poll(token));
  row = Find(SB_RICH_FAMILY_SPACING);
  ASSERT_NE(row, nullptr);
  EXPECT_EQ(row->contributor.binding_revision_known, 0u);

  // The layout before the binding fields is still accepted; its revision is unknown.
  const uint32_t pre381 = static_cast<uint32_t>(
      offsetof(sb_relation_payload_t, source_binding_ordinal));
  p.struct_size = pre381;
  p.has_binding_revision = 1;  // beyond the declared size: ignored
  ASSERT_EQ(sb_inject_relation_payloads(space, actor, &p, 1, pre381, 0.3), 1);
  ASSERT_TRUE(Poll(token));
  row = Find(SB_RICH_FAMILY_SPACING);
  ASSERT_NE(row, nullptr);
  EXPECT_EQ(row->contributor.binding_revision_known, 0u);

  // The pre-ap_offset layout (ends before ap_offset_u_m) still carries
  // source_binding_ordinal and binding_revision: it must preserve them.
  const uint32_t preA4 = static_cast<uint32_t>(
      offsetof(sb_relation_payload_t, ap_offset_u_m));
  ASSERT_GT(preA4, pre381);
  p.struct_size = preA4;
  p.has_binding_revision = 1;
  p.binding_revision = 91;
  ASSERT_EQ(sb_inject_relation_payloads(space, actor, &p, 1, preA4, 0.4), 1);
  ASSERT_TRUE(Poll(token));
  row = Find(SB_RICH_FAMILY_SPACING);
  ASSERT_NE(row, nullptr);
  EXPECT_EQ(row->contributor.binding_revision_known, 1u);
  EXPECT_EQ(row->contributor.binding_revision, 91u);
  // ... and still honours an unset revision at that size.
  p.has_binding_revision = 0;
  ASSERT_EQ(sb_inject_relation_payloads(space, actor, &p, 1, preA4, 0.5), 1);
  ASSERT_TRUE(Poll(token));
  row = Find(SB_RICH_FAMILY_SPACING);
  ASSERT_NE(row, nullptr);
  EXPECT_EQ(row->contributor.binding_revision_known, 0u);
  p.struct_size = sizeof(p);
  EXPECT_EQ(sb_sizeof(SB_STRUCT_RELATION_PAYLOAD), sizeof(p));
  // The pose fit appended 3 more fields (ap_offset_u_m/ap_offset_v_m/
  // ap_height_m) after this test's original "last 3" (binding_revision,
  // has_binding_revision, r_formula_version), so binding_revision moved from
  // count - 3 to count - 6.
  EXPECT_EQ(sb_get_struct_field_offset(
                SB_STRUCT_RELATION_PAYLOAD,
                sb_get_struct_field_count(SB_STRUCT_RELATION_PAYLOAD) - 6),
            offsetof(sb_relation_payload_t, binding_revision));

  sb_retire_rich_realization(token);
  sb_release_rich_realization(token);
  sb_destroy_session(session);
}
