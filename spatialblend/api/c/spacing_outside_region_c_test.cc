// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Spacing to a referent with a containment region, the source OUTSIDE that region: the receiver's root
// target keeps the source's distance to the region (it is not clamped onto the region's edge).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "spatialblend/api/c/c_test_embodied.h"
#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/api/c/scope_extension_test_api.h"

namespace {

void Translation(double x, double y, double out[16]) {
  std::memset(out, 0, 16 * sizeof(double));
  out[0] = out[5] = out[10] = out[15] = 1.0;
  out[3] = x;
  out[7] = y;
}

// A shared containment region (world centre and half extents), under the room (parent -1) or under
// another region of the layout (its index).
struct Region {
  int64_t shared;
  double cx, cy, hx, hy;
  int parent = -1;
  double Distance(double x, double y) const {
    const double dx = std::max({cx - hx - x, 0.0, x - (cx + hx)});
    const double dy = std::max({cy - hy - y, 0.0, y - (cy + hy)});
    return std::hypot(dx, dy);
  }
};

struct Layout {
  std::vector<Region> regions;
  double obstacle_x, obstacle_y;
};

sb_element_t BuildRoom(sb_space_t space, const Layout& l) {
  const sb_element_t room = sb_add_element(space, 0);
  sb_set_element_shared_id(space, room, 100);
  const double poly[] = {0, 0, 6, 0, 6, 6, 0, 6};
  EXPECT_EQ(SB_SCENE_OK, sb_set_element_polygon(space, room, SB_GEOMETRY_NAVIGABLE, poly, 4));
  std::vector<sb_element_t> made;
  for (const Region& r : l.regions) {
    const bool nested = r.parent >= 0;
    const sb_element_t e = sb_add_element(space, nested ? made[r.parent] : room);
    sb_set_element_shared_id(space, e, r.shared);
    double m[16];
    // Element transforms are parent-relative.
    const double px = nested ? l.regions[r.parent].cx : 0.0, py = nested ? l.regions[r.parent].cy : 0.0;
    Translation(r.cx - px, r.cy - py, m);
    sb_set_element_transform(space, e, m);
    const double box[] = {-r.hx, -r.hy, r.hx, -r.hy, r.hx, r.hy, -r.hx, r.hy};
    EXPECT_EQ(SB_SCENE_OK, sb_set_element_polygon(space, e, SB_GEOMETRY_CONTAINMENT, box, 4));
    made.push_back(e);
  }
  // An unshared obstacle (furniture of this room only).
  const sb_element_t ob = sb_add_element(space, room);
  double m[16];
  Translation(l.obstacle_x, l.obstacle_y, m);
  sb_set_element_transform(space, ob, m);
  EXPECT_EQ(SB_SCENE_OK, sb_set_element_box(space, ob, SB_GEOMETRY_OBSTACLE, -0.3, -0.3, 0, 0.3, 0.3, 0.8));
  return room;
}

// The distance of the source and of the receiver's final body to each region.
struct Result {
  std::vector<double> src, rcv;
  int64_t root_referent = 0;  // the root proposal's driving referent (shared id), last update
};

// Drives the source still at (sx, sy) for `updates` updates; the receiver body follows each proposal
// (an ideal driver) from (rx, ry).
Result Drive(const Layout& a, const Layout& b, double sx, double sy, double rx, double ry, int updates = 60) {
  const sb_session_t session = sb_create_session();
  const sb_space_t src = sb_add_space(session), rcv = sb_add_space(session);
  const sb_element_t src_room = BuildRoom(src, a);
  const sb_element_t rcv_room = BuildRoom(rcv, b);
  // The scope extensions are built in the background.
  EXPECT_EQ(sb_test_wait_for_scope_extension(src), 1);
  EXPECT_EQ(sb_test_wait_for_scope_extension(rcv), 1);
  const sb_actor_t actor = sb_create_actor(src);
  spatialblend_c_test::PublishStartPose(src, actor, sx, sy, 0.0);
  sb_add_anchor(src, actor, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(src, actor, src_room);
  sb_bind(src, actor, "root", SB_KERNEL_SPACING);
  sb_set_min_move(src, 0);
  sb_set_min_yaw(src, 0);
  const sb_actor_t avatar = sb_create_actor(rcv);
  sb_add_anchor(rcv, avatar, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(rcv, avatar, rcv_room);
  spatialblend_c_test::PublishStartPose(rcv, avatar, rx, ry, 0.0);
  double bx = rx, by = ry;
  int64_t last_referent = 0;
  for (int k = 0; k < updates; ++k) {
    // A tiny jitter so every update is a fresh source sample.
    sb_update_actor_state(src, actor, sx + ((k & 1) ? 1e-4 : 0.0), sy, 0.0);
    const auto v = spatialblend_c_test::ReadProposals(rcv, avatar);
    if (v.rc == 0 && (v.status == SB_PROPOSAL_EXACT || v.status == SB_PROPOSAL_INEXACT) &&
        !v.position_unspecified()) {
      bx = v.x;
      by = v.y;
      last_referent = v.root.referent_shared_id;
      spatialblend_c_test::PublishStartPose(rcv, avatar, bx, by, 0.0);
    }
  }
  sb_destroy_session(session);
  Result r;
  r.root_referent = last_referent;
  for (const Region& g : a.regions) r.src.push_back(g.Distance(sx, sy));
  for (const Region& g : b.regions) r.rcv.push_back(g.Distance(bx, by));
  std::fprintf(stderr, "  source (%.3f, %.3f) receiver (%.3f, %.3f)\n", sx, sy, bx, by);
  for (size_t i = 0; i < r.src.size(); ++i)
    std::fprintf(stderr, "  region %lld: source %.3f m, receiver %.3f m\n",
                 static_cast<long long>(a.regions[i].shared), r.src[i], r.rcv[i]);
  return r;
}

// The distance to each listed region (indices into the layout) is kept.
void ExpectKept(const Result& r, double tol, std::vector<size_t> which = {}) {
  if (which.empty())
    for (size_t i = 0; i < r.src.size(); ++i) which.push_back(i);
  for (size_t i : which) EXPECT_NEAR(r.rcv[i], r.src[i], tol) << "region " << i;
}

// Chess and a TABLE, siblings in the room.
const Layout kSiblings{{{200, 3.0, 3.0, 0.45, 0.45}, {300, 1.5, 4.5, 0.6, 0.4}}, 4.8, 1.2};

TEST(SpacingOutsideRegionCTest, SiblingRegionsFromSourcePose) {
  const double sx = 3.0, sy = 3.45 + 1.0;  // 1.0 m from the Chess region's edge
  const Result r = Drive(kSiblings, kSiblings, sx, sy, sx, sy);
  ExpectKept(r, 0.05);
  // The root proposal names the driving referent of its Spacing solve, as every anchor's does.
  EXPECT_TRUE(r.root_referent == 200 || r.root_referent == 300) << r.root_referent;
}

TEST(SpacingOutsideRegionCTest, SiblingRegionsFromOwnSpot) {
  const double sx = 3.0, sy = 3.45 + 1.0;
  ExpectKept(Drive(kSiblings, kSiblings, sx, sy, 5.0, 5.0), 0.05);
}

// The Chess region is the TABLE's child and sticks out of it across a 0.5 m gap: the TABLE's scope
// grows by the Chess region, its walkable by the bridge too (element/scope_extension.h). A Whiteboard
// is the room's other referent.
const Layout kNested{{{300, 2.0, 3.0, 0.6, 0.4},
                      {200, 3.55, 3.0, 0.45, 0.45, /*parent=*/0},
                      {400, 4.5, 5.2, 0.6, 0.3}},
                     4.8, 1.2};
// The receiver's TABLE, Chess and Whiteboard stand elsewhere in its room (moved by (0.8, -1.5)), its
// obstacle too: the same distances at another pose.
const Layout kNestedElsewhere{{{300, 2.8, 1.5, 0.6, 0.4},
                               {200, 4.35, 1.5, 0.45, 0.45, /*parent=*/0},
                               {400, 5.3, 3.7, 0.6, 0.3}},
                              1.0, 4.5};
// The Chess 1.5 m beyond the TABLE: a corridor joins the walkable.
const Layout kCorridor{{{300, 1.2, 3.0, 0.6, 0.4},
                        {200, 3.75, 3.0, 0.45, 0.45, /*parent=*/0},
                        {400, 4.5, 5.2, 0.6, 0.3}},
                       4.8, 1.2};

// In the gap the pose is the room's (the bridge is only walkable): its relations
// are the room's referents, the TABLE and the Whiteboard, and the receiver keeps their distances.
TEST(SpacingOutsideRegionCTest, SourceInTheGapKeepsTheRoomsReferents) {
  const double sx = 2.85, sy = 3.0;  // 0.25 m from the TABLE, 0.25 m from the Chess
  ExpectKept(Drive(kNested, kNested, sx, sy, sx, sy), 0.05, {0, 2});
}

TEST(SpacingOutsideRegionCTest, SourceInTheGapFromOwnSpot) {
  const double sx = 2.85, sy = 3.0;
  ExpectKept(Drive(kNested, kNested, sx, sy, 5.0, 5.0), 0.05, {0, 2});
}

TEST(SpacingOutsideRegionCTest, SourceInTheGapOtherLayout) {
  const double sx = 2.85, sy = 3.0;
  const Result r = Drive(kNested, kNestedElsewhere, sx, sy, 1.0, 1.0);
  ExpectKept(r, 0.05, {0, 2});
}

TEST(SpacingOutsideRegionCTest, SourceInACorridorKeepsTheRoomsReferents) {
  const double sx = 2.5, sy = 3.0;  // 0.7 m from the TABLE, 0.8 m from the Chess
  ExpectKept(Drive(kCorridor, kCorridor, sx, sy, 5.0, 1.0), 0.05, {0, 2});
}

// Inside the extending child the pose resolves through the TABLE into the Chess: the receiver is
// at the mapped position.
TEST(SpacingOutsideRegionCTest, SourceInsideTheExtendingChildResolvesThroughTheParent) {
  const double sx = 3.6, sy = 3.1;
  const Result r = Drive(kNested, kNested, sx, sy, 5.0, 5.0);
  ExpectKept(r, 0.05);
  EXPECT_EQ(r.rcv[1], 0.0) << "inside the Chess";
}

}  // namespace
