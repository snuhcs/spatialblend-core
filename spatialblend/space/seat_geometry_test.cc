// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/seat_geometry_test.cc
//
// R2_6_SeatGeometric: the seat is a GEOMETRIC fact: seated means the relation-bearing
// pelvis touches the sittable surface. There is no stillness timer, hold, dwell or
// hysteresis in core, and engagement does not open a halo in F ("one F").
//
// Oracles are analytic: the seat is a box whose top is z = 0.45, footprint
// [2.5, 3.5] x [2.6, 3.4].
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/solver/engagement.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace test {
namespace {

constexpr element_id_t kRoom = 1, kSeat = 10;
constexpr double kSeatTop = 0.45;

Eigen::Matrix4d At(double x, double y) {
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = x;
  t(1, 3) = y;
  return t;
}

void BuildScene(Space* s) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kRoom).ok());
  tree->GetElement(kRoom)->SetSharedId(100);
  tree->GetElement(kRoom)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(
          std::vector<Eigen::Vector2d>{{0, 0}, {10, 0}, {10, 10}, {0, 10}})));
  ASSERT_TRUE(tree->AddElement(kSeat, kRoom).ok());
  tree->GetElement(kSeat)->SetSharedId(1000);
  tree->GetElement(kSeat)->SetTag("Sittable");
  tree->GetElement(kSeat)->SetLocalTransform(At(3.0, 3.0));
  auto seat = std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-0.5, -0.4, 0.0), Eigen::Vector3d(0.5, 0.4, kSeatTop),
      GeometryRole::kObstacle);
  SubRegion top;
  top.name = "seat";
  top.tag = "Sittable";
  top.local_transform = Eigen::Matrix4d::Identity();
  top.local_transform(2, 3) = kSeatTop;
  top.boundary = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {-0.5, -0.4}, {0.5, -0.4}, {0.5, 0.4}, {-0.5, 0.4}});
  seat->AddSubRegion(top);
  tree->GetElement(kSeat)->SetSpatialDefinition(seat);
}

struct Rig {
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space* src = nullptr;
  Space* rcv = nullptr;
  spatialblend::entity_id_t source = 0, avatar = 0;
  uint64_t seq = 0;

  Rig() {
    src = ctx->AddSpace();
    rcv = ctx->AddSpace();
    BuildScene(src);
    BuildScene(rcv);
    const std::vector<SubjectAnchor> anchors = {
        SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
        SubjectAnchor("pelvis", Eigen::Vector3d(0, 0, 0.9))};
    source = src->CreateActor(anchors);
    src->GetActor(source)->SetElementId(kRoom);
    avatar = rcv->CreateActor(anchors);
    rcv->GetActor(avatar)->SetElementId(kRoom);
    src->SetHub(ctx->GetHub());
    rcv->SetHub(ctx->GetHub());
    src->SetMovementGate(0.0, 0.0);
    src->Bind(source, "root", RelationKernelType::kAvatarSpacing);
    EXPECT_NE(src->BindScoped(source, "pelvis",
                              RelationKernelType::kAvatarAffordancePose, kSeat,
                              "Sittable"),
              nullptr);
    EmbodiedPoseInput in;
    in.x = 1.0;
    in.y = 1.0;
    in.epoch = 1;
    EXPECT_EQ(rcv->PublishEmbodiedPose(avatar, in, 0.0, ++seq),
              EmbodiedPublishResult::kOk);
  }

  // The source's pelvis at (px, py, z), its root standing at `root`.
  ReceiverProposal Frame(double px, double py, double z,
                         const Eigen::Vector2d& root) {
    src->SetAnchorWorldPose(source, "pelvis", Eigen::Vector3d(px, py, z), 0.0);
    src->UpdateActorState(source, root, 0.0);
    ReceiverProposal p;
    EXPECT_TRUE(rcv->ReadProposal(avatar, &p));
    return p;
  }
  static const AnchorProposal* Pelvis(const ReceiverProposal& p) {
    for (const auto& a : p.anchors)
      if (a.anchor_name == "pelvis") return &a;
    return nullptr;
  }
  bool Engaged(double px, double py, double z) {
    const ReceiverProposal p = Frame(px, py, z, {px, py});
    const AnchorProposal* a = Pelvis(p);
    return a != nullptr && a->engaged();
  }
};

// Engagement is the pelvis touching the surface: inside the part's footprint and on or below
// its plane. The sweep crosses the plane in steps far below any physical threshold: the
// flag flips at 0.45 and nowhere else.
TEST(R2_6_SeatGeometric, EngagesExactlyWhenThePelvisTouchesTheSurface) {
  Rig r;
  for (double z = 0.60; z >= 0.30 - 1e-12; z -= 0.01) {
    const bool touches = z < kSeatTop + kContactEpsilonM;
    EXPECT_EQ(r.Engaged(3.1, 3.05, z), touches) << "z=" << z;
  }
  EXPECT_FALSE(r.Engaged(3.1, 3.05, kSeatTop + 2.0 * kContactEpsilonM)) << "a height above the contact epsilon";
  EXPECT_TRUE(r.Engaged(3.1, 3.05, kSeatTop));
  EXPECT_TRUE(r.Engaged(3.1, 3.05, kSeatTop + 0.5 * kContactEpsilonM))
      << "within the contact epsilon (v1's kEpsilon)";
  EXPECT_TRUE(r.Engaged(3.1, 3.05, 0.2)) << "having crossed the plane";
}

TEST(R2_6_SeatGeometric, OutsideTheFootprintIsNotTouchingHoweverLow) {
  Rig r;
  EXPECT_FALSE(r.Engaged(3.6, 3.05, kSeatTop)) << "0.1 m past the seat's edge";
  EXPECT_FALSE(r.Engaged(3.1, 3.45, kSeatTop));
  EXPECT_TRUE(r.Engaged(3.4, 3.35, kSeatTop)) << "inside, near the corner";
}

// No stillness, no hold, no dwell, no hysteresis: the first frame that touches engages, the
// first frame that does not releases, at any frame rate, on a pelvis that never stays still.
TEST(R2_6_SeatGeometric, NoStillnessHoldDwellOrHysteresis) {
  Rig r;
  for (int k = 0; k < 12; ++k) {
    const bool down = k % 2 == 0;
    // Each frame is 1/30 s of source time at most and the pelvis moves 0.15 m between
    // frames (never still): the old criterion (< 0.05 m/s for 0.5 s) could not engage.
    const ReceiverProposal p =
        r.Frame(3.1, 3.05, down ? kSeatTop : kSeatTop + 0.15, {3.1, 3.05});
    const AnchorProposal* a = Rig::Pelvis(p);
    ASSERT_NE(a, nullptr) << "frame " << k;
    EXPECT_EQ(a->engaged(), down) << "frame " << k;
    if (k > 0) {
      EXPECT_TRUE(a->cause & (down ? kCauseSeatEngage : kCauseSeatRelease))
          << "frame " << k << " names its cause on the frame it happens";
    }
  }
}

// One F: engagement does not widen the free space. Two identical rigs see
// the same source root standing beside the seat; in one the pelvis touches, in the other it
// hovers. The receiver's ROOT solve and proposal are bit-for-bit the same frame by frame
// (before, engagement opened a 0.30 m halo around the seat that moved the clamp of S0).
TEST(R2_6_SeatGeometric, EngagementNeverChangesTheRootProposal) {
  Rig touching, hovering;
  const Eigen::Vector2d root(4.0, 3.0);  // 0.5 m from the seat's edge
  touching.rcv->EnableObservabilityCapture(true);  // CopySolverObservation
  hovering.rcv->EnableObservabilityCapture(true);
  uint64_t solves = 0;
  for (int k = 0; k < 8; ++k) {
    const ReceiverProposal a = touching.Frame(3.1, 3.05, kSeatTop, root);
    const ReceiverProposal b = hovering.Frame(3.1, 3.05, kSeatTop + 0.20, root);
    ASSERT_NE(Rig::Pelvis(a), nullptr);
    ASSERT_NE(Rig::Pelvis(b), nullptr);
    EXPECT_TRUE(Rig::Pelvis(a)->engaged()) << "frame " << k;
    EXPECT_FALSE(Rig::Pelvis(b)->engaged()) << "frame " << k;
    EXPECT_EQ(a.x, b.x) << "frame " << k;
    EXPECT_EQ(a.y, b.y) << "frame " << k;
    EXPECT_EQ(a.yaw, b.yaw) << "frame " << k;
    // An update with no request and no change publishes no new set, so the published sets
    // answer different source updates (the engaged pelvis republishes every frame; the hovering rig's root has no
    // request and keeps its placed set). The root SOLVES are what this frame computed: both rigs solve every
    // frame, and the two solves are identical, the free space they answered in included.
    Space::SolverWarmState wa, wb;
    ASSERT_TRUE(touching.rcv->CopySolverWarmState(touching.avatar, &wa)) << "frame " << k;
    ASSERT_TRUE(hovering.rcv->CopySolverWarmState(hovering.avatar, &wb)) << "frame " << k;
    ASSERT_GT(wa.solves, solves) << "frame " << k << ": the touching rig solved this frame";
    ASSERT_EQ(wa.solves, wb.solves) << "frame " << k;
    solves = wa.solves;
    Space::SolverObservation sa, sb;
    ASSERT_TRUE(touching.rcv->CopySolverObservation(touching.avatar, &sa) && sa.valid) << "frame " << k;
    ASSERT_TRUE(hovering.rcv->CopySolverObservation(hovering.avatar, &sb) && sb.valid) << "frame " << k;
    EXPECT_EQ(sa.solve.position.x(), sb.solve.position.x()) << "frame " << k;
    EXPECT_EQ(sa.solve.position.y(), sb.solve.position.y()) << "frame " << k;
    EXPECT_EQ(sa.solve.yaw, sb.solve.yaw) << "frame " << k;
    EXPECT_EQ(sa.solve.status, sb.solve.status) << "frame " << k;
    EXPECT_EQ(sa.solve.feasibility, sb.solve.feasibility) << "frame " << k;
    EXPECT_EQ(sa.solve.free_exempt, sb.solve.free_exempt) << "frame " << k;
    EXPECT_EQ(sa.solve.free_ring_exempt, sb.solve.free_ring_exempt) << "frame " << k;
    EXPECT_EQ(sa.scene_position.x(), sb.scene_position.x()) << "frame " << k;
    EXPECT_EQ(sa.scene_position.y(), sb.scene_position.y()) << "frame " << k;
  }
}

// The seat pose is the PELVIS proposal. The root's yaw is not overridden: with no Facing
// bound it is unspecified, touching or not.
TEST(R2_6_SeatGeometric, TheSeatIsThePelvisProposalAndNotTheRootYaw) {
  Rig r;
  const ReceiverProposal touch = r.Frame(3.1, 3.05, kSeatTop, {3.1, 3.05});
  const AnchorProposal* a = Rig::Pelvis(touch);
  ASSERT_NE(a, nullptr);
  EXPECT_TRUE(a->engaged());
  EXPECT_EQ(a->status, ProposalStatus::kExact);
  EXPECT_NEAR(a->z, kSeatTop, 1e-9);
  EXPECT_FALSE(touch.yaw_specified());
  EXPECT_FALSE(touch.cause & (kCauseSeatEngage | kCauseSeatRelease));
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
