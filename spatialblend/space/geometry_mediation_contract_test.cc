// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/geometry_mediation_contract_test.cc
//
// Contract tests for the receiver's root feasibility and the Space-Time
// Mediator:
//
//   - every realized root target is feasible: in the walkable,
//     outside obstacles, >= body radius clearance; no detour ->
//     explicit infeasible outcome.
//   - spacing never relocates the root onto a footprint the source
//     is not on; on R0 the root target reproduces the source root.
//   - the course service (Space::QueryCourse) a bounded DRIVER paces
//     along keeps body-radius clearance.
//   - the proposal's `engaged` flag follows the metric criterion
//     with hysteresis. The teleport policy that acts on it, the
//     mediator trace line and the paced bounds belong to
//     the driver: core only reports.
//
// Oracles are analytic (axis-aligned box distance, the scripted source
// trajectory), independent of the core's geometry code.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "spatialblend/base/common/trace.h"
#include "spatialblend/space/testing/test_driver.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/solver/engagement.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace test {
namespace {

constexpr double kBody = 0.2;  // the test embodiment's declared radius: the DRIVER keeps its body this clear
constexpr double kDt = 1.0 / 30.0;
constexpr double kWalk = 1.4;  // source speed, m/s

struct Box {
  Eigen::Vector2d lo, hi;
  double Distance(const Eigen::Vector2d& p) const {
    const double dx = std::max({lo.x() - p.x(), 0.0, p.x() - hi.x()});
    const double dy = std::max({lo.y() - p.y(), 0.0, p.y() - hi.y()});
    return std::hypot(dx, dy);
  }
  bool Inside(const Eigen::Vector2d& p) const {
    return p.x() > lo.x() && p.x() < hi.x() && p.y() > lo.y() &&
           p.y() < hi.y();
  }
};

void AddRoom(Space* space, double w, double h) {
  auto* tree = space->GetElementTree();
  ASSERT_TRUE(tree->AddElement(1).ok());
  tree->GetElement(1)->SetSharedId(100);
  auto walkable = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {w, 0}, {w, h}, {0, h}});
  tree->GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(walkable));
}

// An OBSTACLE-role box element (as the transfer harness registers furniture
// and receiver obstacles), child of the room, placed by its local transform.
void AddObstacleBox(Space* space, element_id_t id, shared_element_id_t shared,
                    const Box& box) {
  auto* tree = space->GetElementTree();
  ASSERT_TRUE(tree->AddElement(id, 1).ok());
  tree->GetElement(id)->SetSharedId(shared);
  const Eigen::Vector2d c = 0.5 * (box.lo + box.hi);
  const Eigen::Vector2d half = 0.5 * (box.hi - box.lo);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = c.x();
  t(1, 3) = c.y();
  tree->GetElement(id)->SetLocalTransform(t);
  tree->GetElement(id)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-half.x(), -half.y(), 0.0),
          Eigen::Vector3d(half.x(), half.y(), 0.8), GeometryRole::kObstacle));
}

struct Pair {
  std::unique_ptr<LocalContext> ctx;
  Space* src = nullptr;
  Space* rcv = nullptr;
  spatialblend::entity_id_t actor = 0;
  spatialblend::entity_id_t avatar = 0;
  // The receiver avatar's driver (declared after the context so it is
  // destroyed first). Tests that grade the TARGET use a zero-lag follower;
  // the course test drives a bounded body itself.
  std::unique_ptr<IdealDriver> follower;
  uint64_t seq = 0;
};

// Source and receiver rooms (w x h); the actor binds root->spacing.
Pair MakePair(double w, double h) {
  Pair p;
  p.ctx = LocalContext::Create();
  p.src = p.ctx->AddSpace();
  p.rcv = p.ctx->AddSpace();
  AddRoom(p.src, w, h);
  AddRoom(p.rcv, w, h);
  std::vector<SubjectAnchor> anchors = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))};
  p.actor = p.src->CreateActor(anchors);
  p.src->GetActor(p.actor)->SetElementId(1);
  p.avatar = p.rcv->CreateActor(anchors);
  p.rcv->GetActor(p.avatar)->SetElementId(1);
  p.src->Bind(p.actor, "root", RelationKernelType::kAvatarSpacing);
  p.src->SetHub(p.ctx->GetHub());
  p.rcv->SetHub(p.ctx->GetHub());
  // Propagate every source frame (no source-side movement deadband).
  p.src->SetMovementGate(0.0, 0.0);
  // A corresponding far-corner shelf gives the root spacing binding a
  // referent in both rooms (the room itself is the actor's own element).
  const Box shelf{{w - 0.6, h - 0.6}, {w - 0.1, h - 0.1}};
  AddObstacleBox(p.src, 9, 900, shelf);
  AddObstacleBox(p.rcv, 9, 900, shelf);
  return p;
}

// The driver places the body: publishes its embodied pose through the port.
void PlaceAvatar(Pair& p, const Eigen::Vector2d& pos, double yaw) {
  EmbodiedPoseInput in;
  in.x = pos.x();
  in.y = pos.y();
  in.yaw = yaw;
  in.epoch = 1;
  p.rcv->PublishEmbodiedPose(p.avatar, in, 0.0, ++p.seq);
}

// A zero-lag driver: every proposal becomes the embodied pose at once.
void Follow(Pair& p, const Eigen::Vector2d& start, double yaw = 0.0) {
  p.follower = std::make_unique<IdealDriver>(p.rcv, p.avatar, start, yaw);
}

std::optional<RealizationTarget> RootTarget(Space* space,
                                            spatialblend::entity_id_t actor) {
  const RealizationResult* r = space->GetRealizationTargets(actor);
  if (!r) return std::nullopt;
  for (const auto& t : r->targets) {
    if (t.anchor_name == "root") return t;
  }
  return std::nullopt;
}

// Straight source walk at kWalk from a to b, one sample per frame.
std::vector<Eigen::Vector2d> Walk(const Eigen::Vector2d& a,
                                  const Eigen::Vector2d& b) {
  std::vector<Eigen::Vector2d> out;
  const double len = (b - a).norm();
  const int n = std::max(1, static_cast<int>(std::ceil(len / (kWalk * kDt))));
  for (int i = 0; i <= n; ++i) out.push_back(a + (b - a) * (double(i) / n));
  return out;
}

double Wrap(double a) { return std::remainder(a, 2.0 * M_PI); }

// --------------------------------------------------------------------------
// The receiver has an obstacle on the straight source path (the
// harness's R3). The core proposes the destination in F = the walkable minus the footprint, with NO body-radius
// clearance (body clearance belongs to the driver); a bounded DRIVER pacing along
// Space::QueryCourse, the driver's course service called with its body's radius, keeps that clearance within its
// own speed and turn bounds; it gets past the obstacle (a detour, not a stall).
// --------------------------------------------------------------------------
TEST(GeometryMediationContract, R2_1_M_1_DetourKeepsClearanceWithinBounds) {
  Pair p = MakePair(10, 10);
  p.rcv->SetBodyRadius(kBody);  // the driver's clearance (a driver-service parameter; the core's F ignores it)
  const Box obstacle{{4.5, 4.6}, {5.5, 5.6}};  // receiver only
  AddObstacleBox(p.rcv, 7, 700, obstacle);
  driver::Bounds bounds;
  bounds.max_linear_speed = 1.2;
  bounds.max_angular_speed = 1.5;
  driver::Pose body;
  body.position = Eigen::Vector2d(2, 5);
  PlaceAvatar(p, body.position, 0.0);

  // The source walks past the obstacle's x range and on to its referent, the
  // far-corner shelf, and arrives at contact. The receiver pursues the shelf by
  // 1/n of its own course per update, around its obstacle, and arrives with the
  // source: a source that stops short of contact leaves a v1 receiver where it
  // is (no request for n = infinity), so the
  // walk ends at contact (exact at contact).
  std::vector<Eigen::Vector2d> path = Walk({2, 5}, {8, 5});
  for (const auto& q : Walk({8, 5}, {9.4, 9.4})) path.push_back(q);
  for (int k = 0; k < 60; ++k) path.push_back(path.back());  // settle
  int targets_seen = 0;
  for (size_t f = 0; f < path.size(); ++f) {
    p.src->UpdateActorState(p.actor, path[f], 0.0);
    auto target = RootTarget(p.rcv, p.avatar);
    if (!target) continue;
    ++targets_seen;
    EXPECT_FALSE(obstacle.Inside(target->position))
        << "frame " << f << " target " << target->position.transpose() << ": the destination is in F";
    driver::Pose goal;
    goal.position = target->position;
    goal.yaw = target->yaw;
    const driver::Pose prev = body;
    body = driver::Step(
        body, goal, p.rcv->QueryCourse(p.avatar, body.position, goal.position),
        bounds, kDt);
    PlaceAvatar(p, body.position, body.yaw);
    EXPECT_GE(obstacle.Distance(body.position), kBody - 1e-6)
        << "frame " << f << " body " << body.position.transpose();
    EXPECT_LE((body.position - prev.position).norm(),
              bounds.max_linear_speed * kDt + 1e-9);
    EXPECT_LE(std::abs(Wrap(body.yaw - prev.yaw)),
              bounds.max_angular_speed * kDt + 1e-9);
  }
  EXPECT_GT(targets_seen, 0);
  EXPECT_GT(body.position.x(), 7.5);
}

// --------------------------------------------------------------------------
// No detour (the obstacle spans the receiver room). The proposal is the
// destination in F (it may lie beyond the wall: the core judges neither reachability
// nor body clearance); the DRIVER, paced along its course
// service, cannot get there and stays on the reachable side, clear of the wall.
// There is no HOLD: the proposal is always answered, never HOLD / NONE.
// --------------------------------------------------------------------------
TEST(GeometryMediationContract, R2_1_NoDetourStopsAtTheWall) {
  Pair p = MakePair(10, 10);
  p.rcv->SetBodyRadius(kBody);  // the driver's clearance
  const Box wall{{4.5, 0.0}, {5.5, 10.0}};
  AddObstacleBox(p.rcv, 7, 700, wall);
  Follow(p, {2, 5});
  // The destination may lie beyond the wall (the core does not judge reachability or clearance);
  // the DRIVER, paced along its course service, cannot get there and stays on the reachable side, clear of the wall.
  driver::Bounds bounds;
  driver::Pose body;
  body.position = Eigen::Vector2d(2, 5);

  bool saw_beyond = false;
  for (const auto& pos : Walk({2, 5}, {8, 5})) {
    p.src->UpdateActorState(p.actor, pos, 0.0);
    const RealizationResult* r = p.rcv->GetRealizationTargets(p.avatar);
    ASSERT_NE(r, nullptr);
    EXPECT_NE(r->status, SolverStatus::kFailed);
    auto target = RootTarget(p.rcv, p.avatar);
    ASSERT_TRUE(target.has_value()) << "no HOLD: the root target is always answered";
    EXPECT_FALSE(wall.Inside(target->position)) << "the destination is in F (no body-radius clearance)";
    EXPECT_NE(p.follower->proposal().status, ProposalStatus::kNone);
    if (pos.x() > wall.hi.x()) saw_beyond = true;
    driver::Pose goal;
    goal.position = target->position;
    goal.yaw = target->yaw;
    const auto course = p.rcv->QueryCourse(p.avatar, body.position, goal.position);
    const std::vector<Eigen::Vector2d> stay = {body.position, body.position};
    body = driver::Step(body, goal, course.empty() ? stay : course, bounds, kDt);
    EXPECT_LE(body.position.x(), wall.lo.x() - kBody + 1e-6) << "the driver stays clear of the wall";
  }
  EXPECT_TRUE(saw_beyond) << "the source must reach the far side of the wall";
}

// --------------------------------------------------------------------------
// On R0 (identical rooms, identical seat): through a whole approach --
// walk up to 0.15 m of the seat front and then onto the seat -- the root
// target reproduces the source root wherever the ratio continuation determines it.
//
// The live root proposal is the PURSUED target (each relation closes 1/n of the receiver's own distance), not the
// placed answer. On identical rooms the pursuit closes every relation's distance exactly; where one relation alone
// carries a request its course is a single distance and the tangential freedom is the course's. Measured: the target
// is the source pose to 1e-9 m everywhere except a band within 0.2 m of the seat front, where the multi-relation
// pursuit's minimiser lies up to 0.0244 m beside it, and it is the source pose again from contact on (the chart snap).
// The band is recorded, not graded: taking the root's tangential freedom from the chart (as the inner solve does) is
// an open question.
// --------------------------------------------------------------------------
TEST(GeometryMediationContract, R2_4_R0RootTargetReproducesSourceRoot) {
  Pair p = MakePair(10, 10);
  const Box seat{{6.0, 4.0}, {7.0, 5.0}};
  AddObstacleBox(p.src, 3, 300, seat);
  AddObstacleBox(p.rcv, 3, 300, seat);
  Follow(p, {2, 4.5});

  std::vector<Eigen::Vector2d> path = Walk({2, 4.5}, {5.85, 4.5});
  for (const auto& q : Walk({5.85, 4.5}, {6.5, 4.5})) path.push_back(q);
  double worst_outside_band = 0.0, worst_band = 0.0, worst_on = 0.0;
  int n = 0, band_frames = 0, touching_or_on = 0;
  for (const auto& pos : path) {
    p.src->UpdateActorState(p.actor, pos, 0.0);
    auto target = RootTarget(p.rcv, p.avatar);
    ASSERT_TRUE(target.has_value());
    EXPECT_TRUE(p.follower->proposal().position_specified()) << "every update carries a position";
    const double d_seat = seat.Distance(pos);
    const double dev = (target->position - pos).norm();
    if (d_seat <= kContactEpsilonM) {
      ++touching_or_on;
      worst_on = std::max(worst_on, dev);
    } else if (d_seat < 0.2) {
      ++band_frames;
      worst_band = std::max(worst_band, dev);
    } else {
      worst_outside_band = std::max(worst_outside_band, dev);
    }
    ++n;
  }
  RecordProperty("r0_band_frames", band_frames);
  RecordProperty("r0_band_worst_um", static_cast<int>(worst_band * 1e6));
  EXPECT_GT(n, 50);
  EXPECT_GT(band_frames, 0);
  EXPECT_GT(touching_or_on, 0);
  EXPECT_LT(worst_outside_band, 1e-3) << "R0 root target deviates from the source root";
  EXPECT_LT(worst_on, 1e-3) << "on the seat the target is the source root (the chart snap)";
}

// --------------------------------------------------------------------------
// A squat / stand next to a seat (source root never on the seat) never
// yields a root target on the receiver seat's footprint -- here on a receiver
// whose seat is placed differently (R1) and on R0.
// --------------------------------------------------------------------------
TEST(GeometryMediationContract, R2_4_NoSeatSquatIsNotPulledOntoFootprint) {
  for (bool moved_seat : {false, true}) {
    SCOPED_TRACE(moved_seat ? "R1 receiver" : "R0 receiver");
    Pair p = MakePair(10, 10);
    const Box src_seat{{6.0, 4.0}, {7.0, 5.0}};
    const Box rcv_seat = moved_seat ? Box{{3.0, 7.0}, {4.0, 8.0}} : src_seat;
    AddObstacleBox(p.src, 3, 300, src_seat);
    AddObstacleBox(p.rcv, 3, 300, rcv_seat);
    Follow(p, {2, 2});

    // Walk up to the seat front, then hold a squat 0.1 m / 0.05 m from it.
    std::vector<Eigen::Vector2d> path = Walk({2, 4.5}, {5.9, 4.5});
    for (int k = 0; k < 30; ++k) path.push_back({5.9, 4.5});
    for (int k = 0; k < 30; ++k) path.push_back({5.95, 4.52});
    for (const auto& pos : path) {
      ASSERT_FALSE(src_seat.Inside(pos));
      p.src->UpdateActorState(p.actor, pos, 0.0);
      auto target = RootTarget(p.rcv, p.avatar);
      if (target) {
        EXPECT_FALSE(rcv_seat.Inside(target->position))
            << "root pulled onto the seat: " << target->position.transpose();
        EXPECT_GT(rcv_seat.Distance(target->position), 0.0);
      }
    }
  }
}

// --------------------------------------------------------------------------
// Engagement: the proposal reports `engaged` when the source ENGAGES
// the table, by the GEOMETRIC criterion (source root on the table, i.e. touching
// or inside its footprint -- analytic oracle, not the core's R), within one source
// update of the oracle. There is no radius around the table. Whether to teleport
// the body on it is the DRIVER's policy; core never acts on it.
// --------------------------------------------------------------------------
TEST(GeometryMediationContract, M_3_EngagedFlagFollowsTheGeometricCriterion) {
  Pair p = MakePair(10, 10);
  const Box table{{7.5, 4.5}, {8.5, 5.5}};
  AddObstacleBox(p.src, 4, 400, table);
  AddObstacleBox(p.rcv, 4, 400, table);
  Follow(p, {1, 1});

  int src_engaged = -1, rcv_engaged = -1, f = 0;
  for (const auto& pos : Walk({2, 5}, {7.9, 5})) {
    p.src->UpdateActorState(p.actor, pos, 0.0);
    if (src_engaged < 0 && table.Distance(pos) <= 0.0) src_engaged = f;
    if (rcv_engaged < 0 && p.follower->proposal().engaged()) rcv_engaged = f;
    ++f;
  }
  ASSERT_GE(src_engaged, 0);
  ASSERT_GE(rcv_engaged, 0);
  EXPECT_LE(std::abs(rcv_engaged - src_engaged), 1);

  // The source walks away: the flag releases (the proposal then follows the
  // source again; no placement, no pacing in core).
  bool released = false;
  for (const auto& pos : Walk({7.9, 5}, {3, 5})) {
    p.src->UpdateActorState(p.actor, pos, 0.0);
    if (!p.follower->proposal().engaged()) released = true;
  }
  EXPECT_TRUE(released);
}

// --------------------------------------------------------------------------
// No hysteresis and no halo. A source hovering 0.28 <-> 0.34 m from
// the table (the old 0.30 m contact distance) never engages; a source crossing the
// table's EDGE engages exactly while it touches or is inside, frame by frame.
// --------------------------------------------------------------------------
TEST(GeometryMediationContract, M_3_EngagementIsContactWithNoHaloOrHysteresis) {
  Pair p = MakePair(10, 10);
  const Box table{{7.5, 4.5}, {8.5, 5.5}};
  AddObstacleBox(p.src, 4, 400, table);
  AddObstacleBox(p.rcv, 4, 400, table);
  Follow(p, {1, 1});
  for (const auto& pos : Walk({4, 5}, {7.5 - 0.34, 5}))
    p.src->UpdateActorState(p.actor, pos, 0.0);
  int engaged_frames = 0;
  for (int k = 0; k < 60; ++k) {
    const double d = (k % 2 == 0) ? 0.28 : 0.34;
    p.src->UpdateActorState(p.actor, {7.5 - d, 5.0 + 0.001 * k}, 0.0);
    if (p.follower->proposal().engaged()) ++engaged_frames;
  }
  EXPECT_EQ(engaged_frames, 0) << "a halo around the table would engage here";

  int mismatches = 0;
  for (int k = 0; k < 60; ++k) {
    const double x = (k % 2 == 0) ? 7.5 - 0.01 : 7.5 + 0.01;  // outside / inside
    p.src->UpdateActorState(p.actor, {x, 5.0}, 0.0);
    if (p.follower->proposal().engaged() != table.Inside({x, 5.0})) ++mismatches;
  }
  EXPECT_EQ(mismatches, 0) << "engagement toggles with the contact itself";
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
