// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Containment transfer feature header (c/spatialblend_containment_transfer.h):
// layout negotiation, the element-local polygon setter, the declaration, and the
// per-update transfer flag, through the public C API only.
//
// ABI 4: the transfer leaf is the actor's
// whole scope and the empty-scope mapping IS the containment transfer. There is no
// contained pose, blend weight or HANDOVER state beside the root proposal
// (sb_get_actor_containment is removed): the receiver's root proposal is the mapped
// pose. ABI 4.3 removed the contained event: the
// state is SB_ANCHOR_PROPOSAL_FLAG_TRANSFER of the root proposal, recomputed on
// every update, and the CONTAIN_ENTER / CONTAIN_EXIT cause bits stay on the jump
// update. A leaf is a region declared on both sides with congruent shapes.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "spatialblend/api/c/c_test_embodied.h"
#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/api/c/spatialblend_containment_transfer.h"

namespace {

constexpr int64_t kRoomShared = 100;
constexpr int64_t kRegionShared = 200;
constexpr double kDeg = M_PI / 180.0;

void RowMajor(double x, double y, double yaw, double out[16]) {
  std::memset(out, 0, 16 * sizeof(double));
  out[0] = std::cos(yaw);
  out[1] = -std::sin(yaw);
  out[4] = std::sin(yaw);
  out[5] = std::cos(yaw);
  out[10] = 1.0;
  out[15] = 1.0;
  out[3] = x;
  out[7] = y;
}

class CContainmentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    session = sb_create_session();
    source = sb_add_space(session);
    receiver = sb_add_space(session);
    for (sb_space_t space : {source, receiver}) {
      const sb_element_t room = sb_add_element(space, 0);
      sb_set_element_shared_id(space, room, kRoomShared);
      const double poly[] = {-10, -10, 10, -10, 10, 10, -10, 10};
      sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, poly, 4);
      const sb_element_t region = sb_add_element(space, room);
      sb_set_element_shared_id(space, region, kRegionShared);
      const sb_actor_t actor = sb_create_actor(space);
      spatialblend_c_test::PublishStartPose(space, actor);
      sb_add_anchor(space, actor, "root", 0, 0, 0, nullptr);
      sb_set_actor_element(space, actor, room);
      if (space == source) {
        source_region = region;
        source_actor = actor;
        sb_bind(space, actor, "root", SB_KERNEL_SPACING);
      } else {
        receiver_region = region;
        receiver_actor = actor;
      }
    }
    sb_set_min_move(source, 0);
    sb_set_min_yaw(source, 0);
  }
  void TearDown() override {
    if (session) sb_destroy_session(session);
  }
  void PlaceRegion(sb_space_t space, sb_element_t region, double x, double y,
                   double yaw) {
    double m[16];
    RowMajor(x, y, yaw, m);
    sb_set_element_transform(space, region, m);
  }
  // The local polygon alone (no declaration).
  void SetPoly(sb_space_t space, sb_element_t region, double hx, double hy) {
    const double box[] = {-hx, -hy, hx, -hy, hx, hy, -hx, hy};
    ASSERT_EQ(sb_set_element_local_polygon(space, region,
                                           SB_GEOMETRY_CONTAINMENT, box, 4),
              1);
  }
  void SetBox(sb_space_t space, sb_element_t region, double hx, double hy) {
    SetPoly(space, region, hx, hy);
    ASSERT_EQ(sb_set_element_containment_transfer(space, region, 1), 1);
  }
  // Whether the receiver's root proposal is the transfer mapping, as of its last
  // update: the per-update transfer flag (ABI 4.3). It is the only contained state
  // the ABI exposes.
  int Contained(sb_space_t space) {
    EXPECT_EQ(space, receiver) << "the flag is a property of a root proposal";
    const auto v = spatialblend_c_test::ReadProposals(receiver, receiver_actor);
    return v.rc == 0 && v.transfer() ? 1 : 0;
  }
  void DriveLocal(double ox, double oy, double oyaw, double lx, double ly,
                  double lyaw) {
    const double c = std::cos(oyaw), s = std::sin(oyaw);
    sb_update_actor_state(source, source_actor, ox + c * lx - s * ly,
                          oy + s * lx + c * ly, oyaw + lyaw);
  }

  sb_session_t session = 0;
  sb_space_t source = 0, receiver = 0;
  sb_element_t source_region = 0, receiver_region = 0;
  sb_actor_t source_actor = 0, receiver_actor = 0;
};

TEST(CContainmentLayoutTest, NegotiatesEveryFieldWithoutChangingRichAbi) {
  sb_containment_transfer_feature_t feature;
  std::memset(&feature, 0x7f, sizeof(feature));
  EXPECT_EQ(sb_get_containment_transfer_feature(nullptr, 0), sizeof(feature));
  EXPECT_EQ(sb_get_containment_transfer_feature(&feature, 1), sizeof(feature));
  EXPECT_EQ(feature.descriptor_size, 0x7f7f7f7fu);  // short probe: no write
  ASSERT_EQ(sb_get_containment_transfer_feature(&feature, sizeof(feature)),
            sizeof(feature));
  EXPECT_EQ(feature.descriptor_size, sizeof(feature));
  // D5: version 2 had no state struct and no blend band; 3 (ABI 4.3) has no event; 4 (ABI 4.4) adds the source fact.
  EXPECT_EQ(feature.feature_version, 4u);
  EXPECT_EQ(feature.coordinate_convention, 1u);
  const std::vector<uint32_t> offsets = {
      offsetof(sb_containment_transfer_feature_t, descriptor_size),
      offsetof(sb_containment_transfer_feature_t, feature_version),
      offsetof(sb_containment_transfer_feature_t, coordinate_convention),
      offsetof(sb_containment_transfer_feature_t, reserved)};
  ASSERT_EQ(sb_get_containment_transfer_field_count(1), offsets.size());
  for (uint32_t i = 0; i < offsets.size(); ++i)
    EXPECT_EQ(sb_get_containment_transfer_field_offset(1, i), offsets[i]);
  EXPECT_EQ(sb_get_containment_transfer_field_offset(1, offsets.size()),
            UINT32_MAX);
  EXPECT_EQ(sb_get_containment_transfer_field_count(0), 0u);
  EXPECT_EQ(sb_get_containment_transfer_field_count(2), 0u)
      << "the ABI 3 per-actor state struct is gone";
  EXPECT_EQ(sb_get_containment_transfer_field_offset(2, 0), UINT32_MAX);
  sb_rich_realization_feature_t rich{};
  sb_get_rich_realization_feature(&rich, sizeof(rich));
  EXPECT_EQ(rich.feature_version, 1u);
}

// A rotated and translated object gets correct containment through the
// element-local setter. (Before ABI 3.1 sb_set_element_polygon read the same
// points as world coordinates and placed the region around the world origin; it
// is element-local now and agrees with sb_set_element_local_polygon.) The source's
// containment is observed through the receiver's flag, which needs the counterpart.
TEST_F(CContainmentTest, LocalPolygonOfARotatedTranslatedObjectIsCorrect) {
  const double ox = 4.0, oy = 5.0, oyaw = 40 * kDeg;
  PlaceRegion(source, source_region, ox, oy, oyaw);
  PlaceRegion(receiver, receiver_region, -5.0, 2.0, 0.0);
  SetBox(receiver, receiver_region, 1.0, 0.8);
  // sb_set_element_polygon with the object-local points: element-local too.
  const double box[] = {-1, -0.8, 1, -0.8, 1, 0.8, -1, 0.8};
  ASSERT_EQ(sb_set_element_polygon(source, source_region, SB_GEOMETRY_CONTAINMENT,
                                   box, 4),
            SB_SCENE_OK);
  ASSERT_EQ(sb_set_element_containment_transfer(source, source_region, 1), 1);
  DriveLocal(ox, oy, oyaw, 0.0, 0.0, 0.0);
  EXPECT_EQ(Contained(receiver), 1);
  // The element-local alias gives the same answer.
  SetBox(source, source_region, 1.0, 0.8);
  DriveLocal(ox, oy, oyaw, 0.0, 0.0, 0.0);
  EXPECT_EQ(Contained(receiver), 1);
  DriveLocal(ox, oy, oyaw, 0.95, 0.0, 0.0);
  EXPECT_EQ(Contained(receiver), 1);
  DriveLocal(ox, oy, oyaw, 1.05, 0.0, 0.0);
  EXPECT_EQ(Contained(receiver), 0);
  DriveLocal(ox, oy, oyaw, 0.0, 0.85, 0.0);
  EXPECT_EQ(Contained(receiver), 0);
  // Moving the object moves the region with it.
  PlaceRegion(source, source_region, -6.0, -3.0, -100 * kDeg);
  DriveLocal(-6.0, -3.0, -100 * kDeg, 0.9, -0.7, 0.0);
  EXPECT_EQ(Contained(receiver), 1);
  DriveLocal(ox, oy, oyaw, 0.0, 0.0, 0.0);
  EXPECT_EQ(Contained(receiver), 0);
}

TEST_F(CContainmentTest, SettersRejectInvalidInput) {
  const double box[] = {-1, -1, 1, -1, 1, 1, -1, 1};
  EXPECT_EQ(sb_set_element_local_polygon(source, source_region,
                                         SB_GEOMETRY_NAVIGABLE, box, 4), 0);
  EXPECT_EQ(sb_set_element_local_polygon(source, source_region,
                                         SB_GEOMETRY_SURFACE, box, 4), 0);
  EXPECT_EQ(sb_set_element_local_polygon(source, source_region,
                                         SB_GEOMETRY_CONTAINMENT, box, 2), 0);
  EXPECT_EQ(sb_set_element_local_polygon(source, source_region,
                                         SB_GEOMETRY_CONTAINMENT, nullptr, 4), 0);
  const double nan_box[] = {-1, -1, 1, -1, NAN, 1, -1, 1};
  EXPECT_EQ(sb_set_element_local_polygon(source, source_region,
                                         SB_GEOMETRY_CONTAINMENT, nan_box, 4), 0);
  EXPECT_EQ(sb_set_element_local_polygon(999, source_region,
                                         SB_GEOMETRY_CONTAINMENT, box, 4), 0);
  EXPECT_EQ(sb_set_element_local_polygon(source, 999999,
                                         SB_GEOMETRY_CONTAINMENT, box, 4), 0);
  // No Containment definition yet: the flag is refused.
  EXPECT_EQ(sb_set_element_containment_transfer(source, source_region, 1), 0);
  EXPECT_EQ(sb_set_element_containment_transfer(source, 999999, 1), 0);
  EXPECT_EQ(sb_set_element_containment_transfer(999, source_region, 1), 0);
  ASSERT_EQ(sb_set_element_local_polygon(source, source_region,
                                         SB_GEOMETRY_CONTAINMENT, box, 4), 1);
  EXPECT_EQ(sb_set_element_containment_transfer(source, source_region, 1), 1);
  EXPECT_EQ(sb_set_element_containment_transfer(source, source_region, 0), 1);
}

// Identical copies (D5): the transfer leaf is the receiver's whole scope, so the
// empty-scope mapping puts its root at the source root in the object frame, under the
// receiver's own placement. That mapped pose is the ROOT PROPOSAL (no contained pose
// beside it); the cause bits mark the jump and the transfer flag says the proposal is
// the transfer mapping, on exactly the updates that are.
TEST_F(CContainmentTest, ReceiverMapsTheObjectRelativePoseAndFlagsTheTransfer) {
  const double sx = 4.0, sy = 5.0, syaw = 40 * kDeg;
  const double rx = -5.0, ry = 2.0, ryaw = 200 * kDeg;
  PlaceRegion(source, source_region, sx, sy, syaw);
  PlaceRegion(receiver, receiver_region, rx, ry, ryaw);
  SetBox(source, source_region, 1.0, 0.8);
  SetBox(receiver, receiver_region, 1.0, 0.8);
  using spatialblend_c_test::ReadProposals;

  // Outside: the flag is clear.
  DriveLocal(sx, sy, syaw, -3.0, 0.0, 0.0);
  EXPECT_EQ(Contained(receiver), 0);
  auto before = ReadProposals(receiver, receiver_actor);
  EXPECT_FALSE(before.cause & SB_PROPOSAL_CAUSE_CONTAIN_ENTER);

  // Fully inside with a non-trivial local yaw: set on this update.
  const double lx = 0.4, ly = -0.3, lyaw = 1.1;
  DriveLocal(sx, sy, syaw, lx, ly, lyaw);
  const double ex = rx + std::cos(ryaw) * lx - std::sin(ryaw) * ly;
  const double ey = ry + std::sin(ryaw) * lx + std::cos(ryaw) * ly;
  // The root the receiver PROPOSES is that pose (the driver applies it).
  auto proposal = ReadProposals(receiver, receiver_actor);
  ASSERT_EQ(proposal.rc, 0);
  EXPECT_TRUE(proposal.transfer());
  EXPECT_NEAR(proposal.x, ex, 0.01);
  EXPECT_NEAR(proposal.y, ey, 0.01);
  EXPECT_TRUE(proposal.cause & SB_PROPOSAL_CAUSE_CONTAIN_ENTER);
  EXPECT_FALSE(proposal.cause & SB_PROPOSAL_CAUSE_CONTAIN_EXIT);

  // Still inside: the flag stays set; the cause bit is on the jump update only.
  DriveLocal(sx, sy, syaw, 0.45, -0.3, lyaw);
  proposal = ReadProposals(receiver, receiver_actor);
  ASSERT_EQ(proposal.rc, 0);
  EXPECT_TRUE(proposal.transfer());
  EXPECT_FALSE(proposal.cause & SB_PROPOSAL_CAUSE_CONTAIN_ENTER);

  // Leaving clears it on the same update.
  DriveLocal(sx, sy, syaw, 2.0, 0.0, 0.0);
  EXPECT_EQ(Contained(receiver), 0);
  EXPECT_TRUE(ReadProposals(receiver, receiver_actor).cause &
              SB_PROPOSAL_CAUSE_CONTAIN_EXIT);

  // Back in: set again.
  DriveLocal(sx, sy, syaw, 0.0, 0.0, 0.0);
  EXPECT_EQ(Contained(receiver), 1);
}

// A counterpart that appears late: the flag is recomputed on every update against the
// CURRENT element set, so it is set on the next update after the receiver declares it.
TEST_F(CContainmentTest, ALateCounterpartSetsTheFlagOnTheNextUpdate) {
  PlaceRegion(source, source_region, 4.0, 5.0, 0.0);
  PlaceRegion(receiver, receiver_region, -5.0, 2.0, 0.0);
  SetBox(source, source_region, 1.0, 0.8);
  // The source root sits inside its leaf; the receiver has the element but has not
  // declared it, so it has no counterpart leaf yet.
  DriveLocal(4.0, 5.0, 0.0, 0.2, 0.0, 0.0);
  DriveLocal(4.0, 5.0, 0.0, 0.3, 0.0, 0.0);
  EXPECT_EQ(Contained(receiver), 0);
  SetPoly(receiver, receiver_region, 1.0, 0.8);
  ASSERT_EQ(sb_set_element_containment_transfer(receiver, receiver_region, 1), 1);
  DriveLocal(4.0, 5.0, 0.0, 0.35, 0.0, 0.0);
  EXPECT_EQ(Contained(receiver), 1);
  // Withdrawn again: clear on the next update.
  ASSERT_EQ(sb_set_element_containment_transfer(receiver, receiver_region, 0), 1);
  DriveLocal(4.0, 5.0, 0.0, 0.4, 0.0, 0.0);
  EXPECT_EQ(Contained(receiver), 0);
}

// A missing counterpart leaves the flag clear however long the source stays inside.
TEST_F(CContainmentTest, AMissingCounterpartLeavesTheFlagClear) {
  PlaceRegion(source, source_region, 4.0, 5.0, 0.0);
  SetBox(source, source_region, 1.0, 0.8);
  for (double lx : {0.0, 0.2, -0.3, 0.5, 0.1}) {
    DriveLocal(4.0, 5.0, 0.0, lx, 0.0, 0.0);
    EXPECT_EQ(Contained(receiver), 0) << "lx " << lx;
  }
}

// A leaf is a region whose shapes are equivalent in the two spaces, judged on every update as part of
// the flag: the declaration is never refused for its shape, a
// non-congruent counterpart leaves the flag clear, and a geometry edit is picked up on the next update.
TEST_F(CContainmentTest, ANonCongruentCounterpartLeavesTheFlagClearAndAnEditIsPickedUpOnTheNextUpdate) {
  PlaceRegion(source, source_region, 4.0, 5.0, 0.0);
  PlaceRegion(receiver, receiver_region, -5.0, 2.0, 0.0);
  SetBox(source, source_region, 1.0, 0.8);
  SetPoly(receiver, receiver_region, 1.0, 0.9);
  EXPECT_EQ(sb_set_element_containment_transfer(receiver, receiver_region, 1), 1)
      << "the declaration is not refused for its shape";
  EXPECT_EQ(sb_get_scene_status(receiver, nullptr, 0), SB_SCENE_OK);
  DriveLocal(4.0, 5.0, 0.0, 0.2, 0.0, 0.0);
  DriveLocal(4.0, 5.0, 0.0, 0.25, 0.0, 0.0);
  EXPECT_EQ(Contained(receiver), 0) << "non-congruent: the flag stays clear";
  // The counterpart's geometry is edited to the same shape in another pose, start vertex and winding.
  const double c = std::cos(0.7), sn = std::sin(0.7);
  const double pts[4][2] = {{-1, -0.8}, {1, -0.8}, {1, 0.8}, {-1, 0.8}};
  double rotated[8];
  for (int i = 0; i < 4; ++i) {
    const int k = (3 - i + 2) % 4;  // reversed winding, shifted start
    rotated[2 * i] = c * pts[k][0] - sn * pts[k][1];
    rotated[2 * i + 1] = sn * pts[k][0] + c * pts[k][1];
  }
  ASSERT_EQ(sb_set_element_local_polygon(receiver, receiver_region,
                                         SB_GEOMETRY_CONTAINMENT, rotated, 4), 1);
  DriveLocal(4.0, 5.0, 0.0, 0.3, 0.0, 0.0);
  EXPECT_EQ(Contained(receiver), 1) << "congruent after the edit: set on the next update";
  SetPoly(receiver, receiver_region, 1.0, 0.9);
  DriveLocal(4.0, 5.0, 0.0, 0.35, 0.0, 0.0);
  EXPECT_EQ(Contained(receiver), 0) << "edited away again: clear on the next update";
}

// The source-side transfer fact (ABI 4.4): what the source's own update computed.
TEST_F(CContainmentTest, TheSourceReadsWhichLeafItsRootIsInPerUpdate) {
  PlaceRegion(source, source_region, 4.0, 5.0, 0.0);
  SetBox(source, source_region, 1.0, 0.8);
  int64_t leaf = -7;
  EXPECT_EQ(sb_get_source_transfer(source, source_actor, &leaf), 0);
  DriveLocal(4.0, 5.0, 0.0, -3.0, 0.0, 0.0);
  EXPECT_EQ(sb_get_source_transfer(source, source_actor, &leaf), 0);
  EXPECT_EQ(leaf, -7) << "untouched when outside";
  DriveLocal(4.0, 5.0, 0.0, 0.3, 0.0, 0.0);
  ASSERT_EQ(sb_get_source_transfer(source, source_actor, &leaf), 1);
  EXPECT_EQ(leaf, kRegionShared);
  DriveLocal(4.0, 5.0, 0.0, 2.0, 0.0, 0.0);
  EXPECT_EQ(sb_get_source_transfer(source, source_actor, &leaf), 0);
  DriveLocal(4.0, 5.0, 0.0, 0.3, 0.0, 0.0);
  EXPECT_EQ(sb_get_source_transfer(source, source_actor, &leaf), 1);
  ASSERT_EQ(sb_set_element_containment_transfer(source, source_region, 0), 1);
  DriveLocal(4.0, 5.0, 0.0, 0.35, 0.0, 0.0);
  EXPECT_EQ(sb_get_source_transfer(source, source_actor, &leaf), 0) << "recomputed, not latched";
  EXPECT_EQ(sb_get_source_transfer(source, 99999, &leaf), -1);
  EXPECT_EQ(sb_get_source_transfer(source, source_actor, nullptr), -1);
  EXPECT_EQ(sb_get_source_transfer(999, source_actor, &leaf), -1);
}

// sb_register_on_relation_event takes the same space lock; from inside a
// relation event callback it must be refused, not deadlock, and must leave the
// registered callback in place.
struct RelationProbe {
  int events = 0;
  sb_space_t space = 0;
};

void OnRelation(const sb_relation_event_t*, void* user) {
  auto* probe = static_cast<RelationProbe*>(user);
  ++probe->events;
  sb_register_on_relation_event(probe->space, nullptr, nullptr);
}

TEST_F(CContainmentTest, RegisteringARelationCallbackFromInsideOneIsRefused) {
  PlaceRegion(source, source_region, 4.0, 5.0, 0.0);
  SetBox(source, source_region, 1.0, 0.8);
  RelationProbe probe;
  probe.space = source;
  sb_register_on_relation_event(source, &OnRelation, &probe);
  DriveLocal(4.0, 5.0, 0.0, -3.0, 0.0, 0.0);
  DriveLocal(4.0, 5.0, 0.0, 0.0, 0.0, 0.0);
  DriveLocal(4.0, 5.0, 0.0, 0.5, 0.0, 0.0);
  const int after_first = probe.events;
  EXPECT_GT(after_first, 0);
  // Still registered: the in-callback unregister attempt was refused.
  DriveLocal(4.0, 5.0, 0.0, 0.0, 0.0, 0.0);
  DriveLocal(4.0, 5.0, 0.0, -2.5, 0.0, 0.0);
  DriveLocal(4.0, 5.0, 0.0, 0.0, 0.0, 0.0);
  DriveLocal(4.0, 5.0, 0.0, 0.3, 0.0, 0.0);
  EXPECT_GE(probe.events, after_first);
  sb_register_on_relation_event(source, nullptr, nullptr);  // outside: works
  const int before = probe.events;
  DriveLocal(4.0, 5.0, 0.0, -3.0, 0.0, 0.0);
  DriveLocal(4.0, 5.0, 0.0, 0.0, 0.0, 0.0);
  EXPECT_EQ(probe.events, before);
}

}  // namespace
