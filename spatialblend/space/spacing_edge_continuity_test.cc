// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/spacing_edge_continuity_test.cc
//
// Scene-level Spacing target continuity just outside a Containment region edge
// (the Freeform Chess shape: region +-0.5 x +-0.9, footprint object
// +-0.43 x +-0.41, Room -> Region -> Object -> Affordance).
//
// Regression for the scene-level root target jumping by up to ~1.2 m in the
// 3-13 cm band outside the region edge. Root cause (relation_solver.cc,
// DecodeTargetPoint): a source closer to the referent edge than 1e-3 of the
// scope diameter (2.8 cm in a 20 m room) was treated as "engaged" because the
// NORMALIZED proximity saturated, so the decoded target collapsed onto the
// edge (zero stand-off). That S0 disagreed with R^src, which made the paper-objective
// compass search run over the flat R level set and land up to 0.55 m away.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace {

constexpr element_id_t kRoom = 1, kRegion = 2, kAvatar = 3, kObject = 5,
                       kAfford = 6;

Eigen::Matrix4d Pose(double x, double y, double yaw) {
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 0) = std::cos(yaw);
  t(0, 1) = -std::sin(yaw);
  t(1, 0) = std::sin(yaw);
  t(1, 1) = std::cos(yaw);
  t(0, 3) = x;
  t(1, 3) = y;
  return t;
}

std::shared_ptr<Polygon2D> Box(double hx, double hy) {
  return std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {-hx, -hy}, {hx, -hy}, {hx, hy}, {-hx, hy}});
}

void BuildScene(Space* s, double rx, double ry, double ryaw, bool flag,
                double k = 1.0) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kRoom).ok());
  tree->GetElement(kRoom)->SetSharedId(100);
  tree->GetElement(kRoom)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
              {0, 0}, {20, 0}, {20, 20}, {0, 20}})));
  ASSERT_TRUE(tree->AddElement(kRegion, kRoom).ok());
  Element* e = tree->GetElement(kRegion);
  e->SetSharedId(300);
  e->SetLocalTransform(Pose(rx, ry, ryaw));
  auto def = std::make_shared<NavigableSpatialDefinition>(
      Box(0.5 * k, 0.9 * k), nullptr, 0.0, GeometryRole::kContainment);
  def->SetElementLocal(true);
  e->SetSpatialDefinition(def);
  if (flag) ASSERT_TRUE(s->SetElementContainmentTransfer(kRegion, true));
  ASSERT_TRUE(tree->AddElement(kObject, kRegion).ok());
  Element* o = tree->GetElement(kObject);
  o->SetSharedId(500);
  o->SetLocalTransform(Pose(0, 0, 0));
  auto od = std::make_shared<NavigableSpatialDefinition>(
      Box(0.43 * k, 0.41 * k), nullptr, 0.0, GeometryRole::kContainment);
  od->SetElementLocal(true);
  o->SetSpatialDefinition(od);
  ASSERT_TRUE(tree->AddElement(kAfford, kObject).ok());
  Element* a = tree->GetElement(kAfford);
  a->SetSharedId(600);
  a->SetLocalTransform(Pose(0, 0, 0));
  a->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-0.2, -0.2, 0.7), Eigen::Vector3d(0.2, 0.2, 0.8),
      GeometryRole::kSurface));
  ASSERT_TRUE(tree->AddElement(kAvatar, kRoom).ok());
  tree->GetElement(kAvatar)->SetSharedId(110);
}

struct Rig {
  std::unique_ptr<LocalContext> ctx;
  Space *src, *tgt;
  spatialblend::entity_id_t sa, ta;
};

std::unique_ptr<Rig> MakeRig(double tx, double ty, double tyaw, bool flag,
                             double k = 1.0) {
  auto rig = std::make_unique<Rig>();
  rig->ctx = LocalContext::Create();
  rig->src = rig->ctx->AddSpace();
  rig->tgt = rig->ctx->AddSpace();
  BuildScene(rig->src, 10, 10, 0.0, flag);
  BuildScene(rig->tgt, tx, ty, tyaw, flag, k);
  for (Space* s : {rig->src, rig->tgt}) {
    std::vector<SubjectAnchor> anchors = {
        SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
        SubjectAnchor("chest", Eigen::Vector3d(0, 0, 1.2))};
    auto a = s->CreateActor(anchors);
    FollowProposals(s, a);
    s->GetActor(a)->SetElementId(kAvatar);
    (s == rig->src ? rig->sa : rig->ta) = a;
  }
  rig->src->Bind(rig->sa, "root", RelationKernelType::kAvatarSpacing);
  rig->src->SetUpdateInterval(1.0 / 60.0);
  rig->tgt->SetUpdateInterval(1.0 / 60.0);
  rig->src->SetHub(rig->ctx->GetHub());
  rig->tgt->SetHub(rig->ctx->GetHub());
  rig->src->SetMovementGate(0.0, 0.0);
  rig->tgt->SetMovementGate(0.0, 0.0);
  return rig;
}

std::optional<RealizationTarget> RootTarget(Space* s,
                                            spatialblend::entity_id_t actor) {
  const RealizationResult* r = s->GetRealizationTargets(actor);
  if (!r) return std::nullopt;
  for (const auto& t : r->targets)
    if (t.anchor_name == "root") return t;
  return std::nullopt;
}

struct Walk {
  // The worst receiver step over the largest closure the solve admitted that
  // frame (d(sample)/n of any term, or the source's own step), and where.
  double worst = 0.0, at_x = 0, at_y = 0;
  int frames = 0;
};

// A step of the receiver root target is graded against what the v1 mechanism
// allows that frame: the closure
// d(sample)/n of the relation(s) it pursues, a snap to the chart pose when the
// source is inside its referent or at contact (the proposal says so with
// kCauseChartSnap), or a placement. A receiver that lags a non-radial source
// closes d0/n > the source's own step, so v*dt is not the bound mid-course.
// Returns how far the step exceeds that allowance (0 when flagged or within).
double StepOverAdmitted(Rig* rig, double step, double source_step) {
  ReceiverProposal prop;
  Space::SolverObservation so;
  rig->tgt->ReadProposal(rig->ta, &prop);
  rig->tgt->CopySolverObservation(rig->ta, &so);
  double allowed = source_step;
  // One relation walks L/n of its smoothed course (v1 CalculateDistanceToMove),
  // an arc at least the chord d(sample)/n; several are closed by d(sample)/n.
  for (const auto& r : so.solve.ratio_intents)
    if (r.update_count > 0.0 && std::isfinite(r.update_count))
      allowed = std::max(allowed, r.step_m >= 0.0
                                      ? r.step_m
                                      : r.sample_distance_m / r.update_count);
  const bool flagged = (prop.cause & (kCauseChartSnap | kCauseFirstProposal |
                                      kCauseSourceException)) != 0;
  return flagged ? 0.0 : step - allowed;
}

Walk WalkLine(bool flag, double tx, double ty, double tyaw, double ang,
              double v = 1.4, double k = 1.0) {
  auto rig = MakeRig(tx, ty, tyaw, flag, k);
  rig->tgt->EnableObservabilityCapture(true);
  const double ds = v / 60.0;
  Walk w;
  std::optional<Eigen::Vector2d> prev;
  for (double d = 4.0; d >= -1.5; d -= ds) {
    const double x = 10 + d * std::cos(ang), y = 10 + d * std::sin(ang);
    rig->src->UpdateActorState(rig->sa, Eigen::Vector2d(x, y), ang + M_PI);
    auto t = RootTarget(rig->tgt, rig->ta);
    if (t) {
      ++w.frames;
      if (prev) {
        const double over =
            StepOverAdmitted(rig.get(), (t->position - *prev).norm(), ds);
        if (over > w.worst) { w.worst = over; w.at_x = x; w.at_y = y; }
      }
      prev = t->position;
    } else prev.reset();
  }
  return w;
}

struct Case {
  const char* name;
  double tx, ty, tyaw;
};

// Per-frame step of the receiver root target while the source walks at
// 1.4 m/s through every approach direction, past and into the region: never more
// than the closure the solve admitted (d(sample)/n), except the flagged chart
// snap on entering the referent, for an identical copy at the same pose and at
// another pose.
TEST(SpacingEdge, WalkingPastAndIntoTheRegionStepsAtMostTheAdmittedClosure) {
  const Case cases[] = {{"same pose", 10, 10, 0.0},
                        {"moved+rotated", 12, 6.5, M_PI / 2}};
  for (bool flag : {false, true}) {
    for (const Case& c : cases) {
      double worst = 0.0, worst_ang = 0.0;
      for (double deg = 0; deg < 360; deg += 15) {
        const Walk w = WalkLine(flag, c.tx, c.ty, c.tyaw, deg * M_PI / 180);
        EXPECT_GT(w.frames, 200) << c.name << " ang " << deg;
        if (w.worst > worst) {
          worst = w.worst;
          worst_ang = deg;
        }
      }
      EXPECT_LE(worst, 1e-3)
          << (flag ? "flagged " : "unflagged ") << c.name << " worst at "
          << worst_ang << " deg";
      RecordProperty(std::string(flag ? "flagged_" : "unflagged_") +
                         (c.tx == 10 ? "same" : "moved") + "_worst_over_mm",
                     static_cast<int>(std::round(worst * 1000)));
    }
  }
}

// Slow crawl through the band 0..15 cm outside every edge. Along a ray that is
// radial to the contact (the four axis directions meet a flat edge head on) an
// identical copy reproduces the source stand-off exactly: no collapse onto the
// edge, no hop (exact along a radial approach). Along a diagonal the contact moves along the edge while the source
// moves straight, so the receiver pursues the contact and deviates; those rays
// are graded for the placement and for steps within the admitted closure.
TEST(SpacingEdge, IdenticalCopyTracksTheSourceThroughTheOutsideBand) {
  for (bool flag : {false, true}) {
    for (double deg : {0.0, 60.0, 90.0, 135.0, 180.0, 270.0}) {
      auto rig = MakeRig(10, 10, 0.0, flag);
      rig->tgt->EnableObservabilityCapture(true);
      const double ang = deg * M_PI / 180;
      const bool radial = std::fmod(deg, 90.0) == 0.0;
      double worst_err = 0.0, worst_over = 0.0, placement_err = -1.0;
      std::optional<Eigen::Vector2d> prev;
      for (double d = 1.2; d >= 0.0; d -= 0.002) {
        // Distance along the ray to the region rectangle edge, then offset.
        const double c = std::cos(ang), s = std::sin(ang);
        const double te = std::min(c != 0 ? 0.5 / std::fabs(c) : 1e9,
                                   s != 0 ? 0.9 / std::fabs(s) : 1e9);
        const double r = te + d;
        const Eigen::Vector2d p(10 + r * c, 10 + r * s);
        rig->src->UpdateActorState(rig->sa, p, ang + M_PI);
        auto t = RootTarget(rig->tgt, rig->ta);
        ASSERT_TRUE(t.has_value());
        if (placement_err < 0.0) placement_err = (t->position - p).norm();
        if (radial) worst_err = std::max(worst_err, (t->position - p).norm());
        if (prev)
          worst_over = std::max(
              worst_over,
              StepOverAdmitted(rig.get(), (t->position - *prev).norm(), 0.002));
        prev = t->position;
      }
      EXPECT_LT(placement_err, 1e-9) << "flag " << flag << " ang " << deg;
      if (radial)
        EXPECT_LT(worst_err, 0.01) << "flag " << flag << " ang " << deg;
      EXPECT_LE(worst_over, 1e-3) << "flag " << flag << " ang " << deg;
    }
  }
}

}  // namespace
}  // namespace spatialblend
