// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// REPLAY inside a referent whose receiver counterpart is NOT congruent: the receiver's target is the source pose through the radial chart of the two footprints, so
// the lateral offset survives. The rig is the scene-geometry adapter's: a table (an Obstacle box)
// with a tabletop plane child (a Surface-role polygon, element-local), a different size and
// position on each side, the source standing off the table's centre line inside it.
//
// The plane's element transform is a parameter. kHorizontal is the plane lying in the Z-up world's
// ground plane (the polygon's own frame is that plane: identity). kAdapter is the transform the
// Unity adapter registers for it (SpatialBlendManager.SceneGeometry.cs: Rx(-90) in Unity, taken through
// the Unity -> core basis change P M P, in float): a plane the core sees standing upright. The
// polygon is planar and the chart is read in its own frame, so an upright plane has no ground
// footprint: nothing is inside it and its interior has no lateral coordinate. That is the producer's
// to fix (v1 says the same in Entity::TransformToVolumeSpace); the core's part, tested below, is that
// it does not call the strip the plane stands on a footprint.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace {

using Vec2 = Eigen::Vector2d;
using Poly = std::vector<Vec2>;

constexpr element_id_t kRoom = 10, kTable = 20, kPlane = 21;
constexpr int64_t kRoomShared = 100, kTableShared = 6101, kPlaneShared = 6201;

enum class PlaneFrame { kHorizontal, kAdapter };

// Unity Rx(-90) as the adapter sends it (UnityToSpatialBlendMatrix: P M P, P swapping Y and Z).
Eigen::Matrix4d AdapterPlaneTransform() {
  Eigen::Matrix4d unity = Eigen::Matrix4d::Identity();
  // cos(-90 deg) in float is not zero: the matrix Unity hands over is only almost upright.
  const double c = static_cast<float>(std::cos(-M_PI / 2)), sn = -1.0;
  unity(1, 1) = c; unity(1, 2) = -sn;
  unity(2, 1) = sn; unity(2, 2) = c;
  Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
  for (int r = 0; r < 4; ++r) {
    const int ur = r == 1 ? 2 : r == 2 ? 1 : r;
    for (int c = 0; c < 4; ++c) {
      const int uc = c == 1 ? 2 : c == 2 ? 1 : c;
      out(r, c) = unity(ur, uc);
    }
  }
  return out;
}

struct Table {
  Vec2 centre;
  Vec2 half;
};

void BuildSide(Space* s, const Table& t, PlaneFrame frame) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kRoom).ok());
  tree->GetElement(kRoom)->SetSharedId(kRoomShared);
  tree->GetElement(kRoom)->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(
      std::make_shared<Polygon2D>(Poly{{-20, -20}, {20, -20}, {20, 20}, {-20, 20}})));
  ASSERT_TRUE(tree->AddElement(kTable, kRoom).ok());
  Element* table = tree->GetElement(kTable);
  table->SetSharedId(kTableShared);
  Eigen::Matrix4d place = Eigen::Matrix4d::Identity();
  place(0, 3) = t.centre.x();
  place(1, 3) = t.centre.y();
  table->SetLocalTransform(place);
  table->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-t.half.x(), -t.half.y(), 0.0), Eigen::Vector3d(t.half.x(), t.half.y(), 0.74),
      GeometryRole::kObstacle));
  ASSERT_TRUE(tree->AddElement(kPlane, kTable).ok());
  Element* plane = tree->GetElement(kPlane);
  plane->SetSharedId(kPlaneShared);
  plane->SetLocalTransform(frame == PlaneFrame::kHorizontal ? Eigen::Matrix4d::Identity()
                                                            : AdapterPlaneTransform());
  auto def = std::make_shared<NavigableSpatialDefinition>(
      std::make_shared<Polygon2D>(Poly{{-t.half.x(), -t.half.y()}, {t.half.x(), -t.half.y()},
                                       {t.half.x(), t.half.y()}, {-t.half.x(), t.half.y()}}),
      nullptr, 0.0, GeometryRole::kSurface);
  def->SetElementLocal(true);
  plane->SetSpatialDefinition(def);
}

// The radial chart of two rectangles, written out here (not read from the core): the source point is
// the same fraction rho of the ray from the centre, in the same direction, on the receiver.
Vec2 RadialMap(const Table& src, const Table& rcv, const Vec2& p) {
  const Vec2 d = p - src.centre;
  if (d.norm() < 1e-12) return rcv.centre;
  const Vec2 u = d.normalized();
  const auto ray = [&](const Table& t) {
    return std::min(t.half.x() / std::max(std::fabs(u.x()), 1e-300),
                    t.half.y() / std::max(std::fabs(u.y()), 1e-300));
  };
  return rcv.centre + u * (d.norm() / ray(src) * ray(rcv));
}

struct Rig {
  std::unique_ptr<LocalContext> ctx;
  std::unique_ptr<IdealDriver> follower;
  Space *src = nullptr, *tgt = nullptr;
  spatialblend::entity_id_t sa = 0, ta = 0;
};

std::unique_ptr<Rig> MakeRig(const Table& a, const Table& b, PlaneFrame frame) {
  auto rig = std::make_unique<Rig>();
  rig->ctx = LocalContext::Create();
  rig->src = rig->ctx->AddSpace();
  rig->tgt = rig->ctx->AddSpace();
  BuildSide(rig->src, a, frame);
  BuildSide(rig->tgt, b, frame);
  for (Space* s : {rig->src, rig->tgt}) {
    const auto act = s->CreateActor({SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))});
    s->GetActor(act)->SetElementId(kRoom);
    (s == rig->src ? rig->sa : rig->ta) = act;
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

// Source poses inside the host table, off its centre line, including the adapter test's own
// (3 cm inside the +X face at chart angle 30 degrees).
std::vector<Vec2> InsidePoses(const Table& t) {
  std::vector<Vec2> out;
  const Vec2 dir(std::cos(M_PI / 6), std::sin(M_PI / 6));
  const double ray = std::min(t.half.x() / dir.x(), t.half.y() / dir.y());
  const Vec2 hit = t.centre + dir * ray;
  out.push_back(Vec2(hit.x() - 0.03, hit.y()));
  out.push_back(t.centre + Vec2(0.30, 0.20));
  out.push_back(t.centre + Vec2(-0.40, 0.30));
  out.push_back(t.centre + Vec2(0.10, -0.33));
  out.push_back(t.centre + Vec2(-0.20, -0.10));
  return out;
}

void ExpectChartMapped(PlaneFrame frame) {
  const Table host{{2.0, 3.0}, {0.6, 0.4}};
  const Table client{{-4.0, 1.0}, {0.8, 0.45}};
  for (const Vec2& p : InsidePoses(host)) {
    auto rig = MakeRig(host, client, frame);
    for (int i = 0; i < 12; ++i) rig->src->UpdateActorState(rig->sa, p, 0.0);
    const ReceiverProposal q = rig->follower->proposal();
    const Vec2 want = RadialMap(host, client, p);
    EXPECT_NEAR(q.x, want.x(), 1e-3) << "source (" << p.x() << "," << p.y() << ")";
    EXPECT_NEAR(q.y, want.y(), 1e-3) << "source (" << p.x() << "," << p.y() << ")";
  }
}

TEST(ReplayChartMapping, NonCongruentTablesWithAHorizontalPlaneMapThroughTheChart) {
  // Scope catalog case SP-INSIDE-10 (spatialblend/space/testing/SCOPE_CASES.md).
  ExpectChartMapped(PlaneFrame::kHorizontal);
}

// The scope of a source inside the table: with the horizontal plane the source is inside the plane
// (REPLAY, the plane alone); with the upright plane it is inside the table and not inside the plane
// (the table is the container, the plane one of its objects).
TEST(ReplayChartMapping, AnUprightPlaneHasNoFootprintAndIsNotEntered) {
  // Scope catalog case SP-INSIDE-11 (spatialblend/space/testing/SCOPE_CASES.md).
  const Table host{{2.0, 3.0}, {0.6, 0.4}};
  const Table client{{-4.0, 1.0}, {0.8, 0.45}};
  const Vec2 p(2.3, 3.2);
  for (const PlaneFrame frame : {PlaneFrame::kHorizontal, PlaneFrame::kAdapter}) {
    auto rig = MakeRig(host, client, frame);
    rig->src->UpdateActorState(rig->sa, p, 0.0);
    const auto& sc = rig->src->GetBindings(rig->sa)[0].spacing_scope;
    ASSERT_TRUE(sc.has_value());
    if (frame == PlaneFrame::kHorizontal) {
      EXPECT_EQ(sc->container, kPlane);
      EXPECT_EQ(sc->mode, SpacingScopeMode::kReplay);
    } else {
      EXPECT_EQ(sc->container, kTable) << "the point is not inside an upright polygon";
      EXPECT_EQ(sc->mode, SpacingScopeMode::kInterior);
    }
  }
}

}  // namespace
}  // namespace spatialblend
