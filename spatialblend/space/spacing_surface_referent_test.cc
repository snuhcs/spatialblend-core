// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// A Surface-role seat is a Spacing referent.
// With the source standing on its seat (R = 1), the receiver's target is the
// mapped contact on the receiver's OWN seat, not the chart-mapped receiver
// sample and not the empty-scope mapping. The same scene with a Containment-role
// seat is the control.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace {

using Poly = std::vector<Eigen::Vector2d>;

constexpr element_id_t kObject = 1, kSeat = 2;
constexpr int64_t kSeatShared = 800;

enum class SeatRole { kSurface, kContainment };

void BuildSpace(Space* s, SeatRole role, const Eigen::Vector2d& centre,
                bool with_room) {
  auto* tree = s->GetElementTree();
  element_id_t parent = kInvalidElementId;
  if (with_room) {
    ASSERT_TRUE(tree->AddElement(10).ok());
    tree->GetElement(10)->SetSharedId(100);
    tree->GetElement(10)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(
            Poly{{-40, -40}, {40, -40}, {40, 40}, {-40, 40}})));
    parent = 10;
  }
  // Without a room the seat hangs under a geometry-less object (the actor's
  // element, as in the C-API repro). With a room the seat is a direct child of
  // the room, which is the actor's element and its Spacing scope.
  if (!with_room) {
    ASSERT_TRUE(tree->AddElement(kObject, parent).ok());
    tree->GetElement(kObject)->SetSharedId(500);
    parent = kObject;
  }
  ASSERT_TRUE(tree->AddElement(kSeat, parent).ok());
  tree->GetElement(kSeat)->SetSharedId(kSeatShared);
  const double h = 1.0;
  if (role == SeatRole::kSurface) {
    tree->GetElement(kSeat)->SetSpatialDefinition(
        std::make_shared<SurfaceSpatialDefinition>(std::make_shared<Polygon3D>(
            std::vector<Eigen::Vector3d>{{centre.x() - h, centre.y() - h, 0.45},
                                         {centre.x() + h, centre.y() - h, 0.45},
                                         {centre.x() + h, centre.y() + h, 0.45},
                                         {centre.x() - h, centre.y() + h, 0.45}})));
  } else {
    // An element-local region placed by the element's transform.
    Eigen::Matrix4d place = Eigen::Matrix4d::Identity();
    place(0, 3) = centre.x();
    place(1, 3) = centre.y();
    tree->GetElement(kSeat)->SetLocalTransform(place);
    auto def = std::make_shared<NavigableSpatialDefinition>(
        std::make_shared<Polygon2D>(Poly{{-h, -h}, {h, -h}, {h, h}, {-h, h}}),
        nullptr, 0.0, GeometryRole::kContainment);
    def->SetElementLocal(true);
    tree->GetElement(kSeat)->SetSpatialDefinition(def);
  }
}

struct Rig {
  std::unique_ptr<LocalContext> ctx;
  std::unique_ptr<IdealDriver> follower;
  Space *src = nullptr, *tgt = nullptr;
  spatialblend::entity_id_t sa = 0, ta = 0;
};

std::unique_ptr<Rig> MakeRig(SeatRole role, bool with_room,
                             const Eigen::Vector2d& tgt_centre) {
  auto rig = std::make_unique<Rig>();
  rig->ctx = LocalContext::Create();
  rig->src = rig->ctx->AddSpace();
  rig->tgt = rig->ctx->AddSpace();
  BuildSpace(rig->src, role, Eigen::Vector2d(0, 0), with_room);
  BuildSpace(rig->tgt, role, tgt_centre, with_room);
  for (Space* s : {rig->src, rig->tgt}) {
    const auto a = s->CreateActor({SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))});
    s->GetActor(a)->SetElementId(with_room ? element_id_t(10) : kObject);
    (s == rig->src ? rig->sa : rig->ta) = a;
  }
  rig->src->Bind(rig->sa, "root", RelationKernelType::kAvatarSpacing);
  for (Space* s : {rig->src, rig->tgt}) {
    s->SetUpdateInterval(1.0 / 60.0);
    s->SetHub(rig->ctx->GetHub());
    s->SetMovementGate(0.0, 0.0);
  }
  rig->tgt->EnableObservabilityCapture(true);
  rig->follower = std::make_unique<IdealDriver>(rig->tgt, rig->ta);
  return rig;
}

// The source stands at its seat's centre (R = 1: inside the referent). The
// receiver's body is far from its own seat.
void ExpectTargetOnTheReceiversSeat(SeatRole role, bool with_room) {
  const Eigen::Vector2d tgt_centre(20, 0);
  auto rig = MakeRig(role, with_room, tgt_centre);
  // Every frame, the first (the absolute placement) included, proposes the
  // mapped contact on the receiver's seat.
  for (int i = 0; i < 3; ++i) {
    rig->src->UpdateActorState(rig->sa, Eigen::Vector2d(0, 0), 0.0);
    const ReceiverProposal q = rig->follower->proposal();
    EXPECT_LT((Eigen::Vector2d(q.x, q.y) - tgt_centre).norm(), 1e-6)
        << "frame " << i << " proposal (" << q.x << "," << q.y << ")";
  }
  const ReceiverProposal p = rig->follower->proposal();
  ASSERT_TRUE(p.status == ProposalStatus::kExact ||
              p.status == ProposalStatus::kInexact);
  Space::SolverObservation so;
  ASSERT_TRUE(rig->tgt->CopySolverObservation(rig->ta, &so));
  ASSERT_FALSE(so.solve.goals.empty()) << "a valid referent exists";
  EXPECT_FALSE(so.solve.no_grounded_referent);
  EXPECT_LT((Eigen::Vector2d(p.x, p.y) - tgt_centre).norm(), 1e-6)
      << "proposal (" << p.x << "," << p.y << ") role "
      << (role == SeatRole::kSurface ? "surface" : "containment")
      << " room " << with_room;
  EXPECT_EQ(p.status, ProposalStatus::kExact);
  EXPECT_GE(p.err_spacing_target, 0.0) << "measured (-1 is no relation)";
  EXPECT_LT(p.err_spacing_target, 1e-6);
}

TEST(SpacingSurfaceReferent, ContainmentSeatControl) {
  ExpectTargetOnTheReceiversSeat(SeatRole::kContainment, false);
  ExpectTargetOnTheReceiversSeat(SeatRole::kContainment, true);
}

TEST(SpacingSurfaceReferent, SurfaceSeatIsASpacingReferent) {
  ExpectTargetOnTheReceiversSeat(SeatRole::kSurface, false);
  ExpectTargetOnTheReceiversSeat(SeatRole::kSurface, true);
}

}  // namespace
}  // namespace spatialblend
