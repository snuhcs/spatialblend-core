// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/net/solver_grounding_requirements_test.cc
//
// Requirement tests for SpatialBlend target grounding and solver behaviour:
//   - explicit grounding failures; target-side Eval_p
//   - root facing continuous and reconciled with the seat
//   - body-anchor targets vs the source support; chart noise
//
// Every expected value is an INDEPENDENT analytic oracle computed in this file
// from the scene geometry (rectangles, projections, bearings), never from the
// core function under test. Source payloads come from the real source path
// (Space::UpdateActorState -> SpaceHub), so the round trips exercise the
// production Extract_p.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/radial_surface_chart.h"
#include "spatialblend/net/proto_util.h"
#include "spatialblend/model/relation/avatar_affordance_pose_kernel.h"
#include "spatialblend/model/relation/kernel_factory.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/solver/relation_evaluator.h"
#include "spatialblend/solver/relation_solver.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace test {
namespace {

constexpr element_id_t kRoom = 1;
constexpr element_id_t kObject = 2;
constexpr element_id_t kActorLeaf = 3;
constexpr element_id_t kZone = 4;
constexpr shared_element_id_t kRoomShared = 100;
constexpr shared_element_id_t kActorShared = 110;
constexpr shared_element_id_t kObjectShared = 200;
constexpr shared_element_id_t kZoneShared = 150;

double Wrap(double a) { return std::remainder(a, 2.0 * M_PI); }

Eigen::Matrix4d Pose(double x, double y, double z, double yaw) {
  Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
  m(0, 0) = std::cos(yaw);
  m(0, 1) = -std::sin(yaw);
  m(1, 0) = std::sin(yaw);
  m(1, 1) = std::cos(yaw);
  m(0, 3) = x;
  m(1, 3) = y;
  m(2, 3) = z;
  return m;
}

std::shared_ptr<Polygon2D> Rect(double x0, double y0, double x1, double y1) {
  return std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});
}

// Axis-aligned room [0,w]x[0,h] (navigable).
void AddRoom(ElementTree* tree, double w, double h) {
  ASSERT_TRUE(tree->AddElement(kRoom).ok());
  tree->GetElement(kRoom)->SetSharedId(kRoomShared);
  tree->GetElement(kRoom)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, w, h)));
}

// A box object (obstacle footprint [-hx,hx]x[-hy,hy], height `top`) placed at
// (x, y, yaw), carrying ONE named planar part on its top face (the part frame
// is the object frame lifted to z = top).
struct ObjectSpec {
  double x, y, yaw, hx, hy, top;
  std::string part = "seat";
};
void AddObject(ElementTree* tree, const ObjectSpec& o,
               element_id_t parent = kRoom) {
  ASSERT_TRUE(tree->AddElement(kObject, parent).ok());
  Element* e = tree->GetElement(kObject);
  e->SetSharedId(kObjectShared);
  e->SetLocalTransform(Pose(o.x, o.y, 0.0, o.yaw));
  auto def = std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-o.hx, -o.hy, 0.0), Eigen::Vector3d(o.hx, o.hy, o.top),
      GeometryRole::kObstacle);
  if (!o.part.empty()) {
    SubRegion part;
    part.name = o.part;
    part.tag = o.part;
    part.local_transform = Pose(0, 0, o.top, 0);
    part.boundary = Rect(-o.hx, -o.hy, o.hx, o.hy);
    def->AddSubRegion(part);
  }
  e->SetSpatialDefinition(def);
}

// Defined with the chart tests: an AffordancePose payload for a seat-local
// contact point, charted with the production RadialSurfaceChart.
RelationPayload SeatContactPayload(const ObjectSpec& seat,
                                   const Eigen::Vector2d& local);

spatialblend::entity_id_t AddActor(Space* space,
                                element_id_t parent = kRoom) {
  auto* tree = space->GetElementTree();
  EXPECT_TRUE(tree->AddElement(kActorLeaf, parent).ok());
  tree->GetElement(kActorLeaf)->SetSharedId(kActorShared);
  const auto actor = space->CreateActor(
      {SubjectAnchor("root", Eigen::Vector3d::Zero()),
       SubjectAnchor("chest", Eigen::Vector3d(0, 0, 1.3)),
       SubjectAnchor("pelvis", Eigen::Vector3d(0, 0, 0.9)),
       SubjectAnchor("hand", Eigen::Vector3d(0.3, 0, 1.0))});
  FollowProposals(space, actor);
  space->GetActor(actor)->SetElementId(kActorLeaf);
  return actor;
}

// World point of object-local (u, v, w).
Eigen::Vector3d ObjectPoint(const ObjectSpec& o, double u, double v, double w) {
  return Eigen::Vector3d(o.x + std::cos(o.yaw) * u - std::sin(o.yaw) * v,
                         o.y + std::sin(o.yaw) * u + std::cos(o.yaw) * v, w);
}

// Analytic projection of a world point onto the object's top part: the part
// point nearest to it (clamped into the rectangle), in world coordinates.
Eigen::Vector3d TopProjection(const ObjectSpec& o, const Eigen::Vector3d& p) {
  const double dx = p.x() - o.x, dy = p.y() - o.y;
  const double u = std::cos(o.yaw) * dx + std::sin(o.yaw) * dy;
  const double v = -std::sin(o.yaw) * dx + std::cos(o.yaw) * dy;
  return ObjectPoint(o, std::clamp(u, -o.hx, o.hx), std::clamp(v, -o.hy, o.hy),
                     o.top);
}

struct Capture {
  std::vector<RemoteRelationUpdate> updates;
  const RelationPayload* Find(RelationKernelType kernel,
                              const std::string& anchor,
                              shared_element_id_t referent) const {
    if (updates.empty()) return nullptr;
    for (const auto& p : updates.back().payloads)
      if (p.kernel_type == kernel && p.subject_anchor_name == anchor &&
          p.referent_shared_id == referent)
        return &p;
    return nullptr;
  }
};

struct SourceRig {
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space* src = nullptr;
  spatialblend::entity_id_t actor = 0;
  Capture capture;
};

// Anchor poses of one source frame. Pelvis/chest/hand yaw = root yaw (the
// harness convention: every tracked joint carries the root heading).
struct Frame {
  Eigen::Vector2d root;
  double yaw;
  std::optional<Eigen::Vector3d> chest, pelvis, hand;
};

void Apply(Space* space, spatialblend::entity_id_t actor, const Frame& f) {
  if (f.chest) space->SetAnchorWorldPose(actor, "chest", *f.chest, f.yaw);
  if (f.pelvis) space->SetAnchorWorldPose(actor, "pelvis", *f.pelvis, f.yaw);
  if (f.hand) space->SetAnchorWorldPose(actor, "hand", *f.hand, f.yaw);
  space->UpdateActorState(actor, f.root, f.yaw);
}
void Apply(SourceRig& rig, const Frame& f) { Apply(rig.src, rig.actor, f); }

std::unordered_map<std::string, TargetAnchorPose> StateOf(const Frame& f) {
  std::unordered_map<std::string, TargetAnchorPose> s;
  s["root"] = {Eigen::Vector3d(f.root.x(), f.root.y(), 0.0), f.yaw};
  if (f.chest) s["chest"] = {*f.chest, f.yaw};
  if (f.pelvis) s["pelvis"] = {*f.pelvis, f.yaw};
  if (f.hand) s["hand"] = {*f.hand, f.yaw};
  return s;
}

// ===========================================================================
// Eval_p on an identity receiver reproduces R^src for all three
// families, with independent analytic values.
// ===========================================================================
TEST(R1_3_TargetEvaluation, IdentityReceiverReproducesSourceForAllFamilies) {
  SourceRig rig;
  rig.src = rig.ctx->AddSpace();
  auto* tree = rig.src->GetElementTree();
  AddRoom(tree, 10.0, 8.0);
  const ObjectSpec table{5.0, 4.0, 0.0, 1.0, 1.0, 0.8, "top"};
  AddObject(tree, table);
  rig.actor = AddActor(rig.src);
  rig.src->Bind(rig.actor, "root", RelationKernelType::kAvatarSpacing);
  rig.src->Bind(rig.actor, "chest", RelationKernelType::kAvatarFacing);
  ASSERT_NE(rig.src->BindScoped(rig.actor, "hand",
                                RelationKernelType::kAvatarAffordancePose,
                                kObject),
            nullptr);
  rig.src->SetHub(rig.ctx->GetHub());
  rig.ctx->GetHub()->SetRelationUpdateCallback(
      [&](const RemoteRelationUpdate& u) { rig.capture.updates.push_back(u); });

  // Root 2 m south of the table's south edge; chest facing -X (the table is
  // off the gaze ray); hand 0.15 m above the table top.
  const Frame f{Eigen::Vector2d(5.0, 1.0), M_PI, Eigen::Vector3d(5.0, 1.0, 1.3),
                std::nullopt, Eigen::Vector3d(5.2, 4.1, 0.95)};
  Apply(rig, f);
  ASSERT_FALSE(rig.capture.updates.empty());

  const double room_diag = std::hypot(10.0, 8.0);
  // Analytic R^src (code definitions):
  //   spacing: R = 1 - d/D, d = 2 (to (5,3)), D = room diagonal;
  //   facing: the ray along -X misses the table, so the contact is the
  //           silhouette vertex of least angular deviation (FacingGroundingRule),
  //           the SW corner (4,3). Offset is -atan(2), R = (1 + 1/sqrt(5))/2.
  //   affordance: proximity component = 1/(1+d), d = 0.15 above the top
  //   (unchanged by the pose fit: it is what the phase hysteresis reads). The
  //   relation-preservation R is the pose fit: position fit 1 (no
  //   in-plane offset), height 1/(1+0.15), yaw fit 1 (a candidate at the
  //   source's own relative yaw): kAffordanceR = position_w + height_w/1.15
  //   + yaw_w.
  const double kSpacing = 1.0 - 2.0 / room_diag;
  const double kFacing = (1.0 + 1.0 / std::sqrt(5.0)) / 2.0;
  const double kAffordance = 1.0 / 1.15;
  const double kAffordanceR =
      kApPositionWeight + kApHeightWeight / 1.15 + kApYawWeight;

  const auto* spacing =
      rig.capture.Find(RelationKernelType::kAvatarSpacing, "root", kObjectShared);
  const auto* facing =
      rig.capture.Find(RelationKernelType::kAvatarFacing, "chest", kObjectShared);
  const auto* affordance = rig.capture.Find(
      RelationKernelType::kAvatarAffordancePose, "hand", kObjectShared);
  ASSERT_NE(spacing, nullptr);
  ASSERT_NE(facing, nullptr);
  ASSERT_NE(affordance, nullptr);
  EXPECT_NEAR(spacing->grounded_relation.proximity, kSpacing, 1e-9);
  EXPECT_NEAR(facing->grounded_relation.proximity, kFacing, 1e-9);
  EXPECT_NEAR(affordance->grounded_relation.proximity, kAffordance, 1e-9);

  // Identity receiver: the same geometry, S = the source-equivalent state.
  const auto evals = EvaluateTargetRelations(rig.capture.updates.back().payloads,
                                             StateOf(f), *tree);
  ASSERT_EQ(evals.size(), rig.capture.updates.back().payloads.size());
  for (size_t i = 0; i < evals.size(); ++i) {
    const auto& e = evals[i];
    SCOPED_TRACE(e.anchor_name + " -> " + std::to_string(e.referent_id));
    ASSERT_TRUE(e.ok()) << TargetEvaluationDispositionName(e.disposition)
                        << ": " << e.reason;
    EXPECT_NEAR(e.r_tgt, e.r_src, 1e-12);
    EXPECT_NEAR(e.error, 0.0, 1e-12);
    // Normalizing context: the scope containing the anchor = the room.
    EXPECT_EQ(e.normalizing_scope_id, kRoom);
    EXPECT_NEAR(e.normalizing_scope_diameter_m, room_diag, 1e-12);
  }
  auto r_of = [&](RelationKernelType k, const std::string& a) {
    for (const auto& e : evals)
      if (e.kernel_type == k && e.anchor_name == a &&
          e.referent_id == kObjectShared)
        return e.r_tgt;
    return -1.0;
  };
  EXPECT_NEAR(r_of(RelationKernelType::kAvatarSpacing, "root"), kSpacing, 1e-9);
  EXPECT_NEAR(r_of(RelationKernelType::kAvatarFacing, "chest"), kFacing, 1e-9);
  EXPECT_NEAR(r_of(RelationKernelType::kAvatarAffordancePose, "hand"),
              kAffordanceR, 1e-9);

  // Eval_p is a function of the candidate state: moving the hand to 0.4 m
  // above the top keeps position/yaw fit unchanged (same XY, same yaw as the
  // source's relative yaw) and only the height term becomes 1/(1+0.4).
  TargetAnchorPose raised{Eigen::Vector3d(5.2, 4.1, 1.2), M_PI};
  const auto moved = EvaluateTargetRelation(*affordance, raised, *tree);
  ASSERT_TRUE(moved.ok());
  EXPECT_EQ(moved.r_formula_version, kApRFormulaPoseFitV1);
  EXPECT_EQ(moved.support_basis, TargetSupportBasis::kMappedSupport);
  EXPECT_NEAR(moved.r_tgt,
              kApPositionWeight + kApHeightWeight / 1.4 + kApYawWeight, 1e-12);
  // The grounded support is the part point under the candidate anchor.
  EXPECT_NEAR((moved.support.contact_point_world -
               TopProjection(table, raised.position))
                  .norm(),
              0.0, 1e-12);
  // The mapped source support is the source contact (identity receiver).
  ASSERT_TRUE(moved.has_mapped_support);
  EXPECT_NEAR(
      (moved.mapped_support - TopProjection(table, *f.hand)).norm(), 0.0, 1e-9);
}

// Eval_p must score the candidate against
// the CORRESPONDING target support -- the RadialSurfaceChart map M of the
// source support point -- not wherever the candidate happens to project onto
// the same patch. A hand at the wrong end of the same patch must not score
// R=1.
TEST(R1_3_TargetEvaluation, MappedSupportNotCandidateProjection_A3) {
  SourceRig rig;
  rig.src = rig.ctx->AddSpace();
  auto* tree = rig.src->GetElementTree();
  AddRoom(tree, 10.0, 8.0);
  const ObjectSpec table{5.0, 4.0, 0.0, 1.0, 1.0, 0.8, "top"};
  AddObject(tree, table);
  rig.actor = AddActor(rig.src);
  ASSERT_NE(rig.src->BindScoped(rig.actor, "hand",
                                RelationKernelType::kAvatarAffordancePose,
                                kObject),
            nullptr);
  rig.src->SetHub(rig.ctx->GetHub());
  rig.ctx->GetHub()->SetRelationUpdateCallback(
      [&](const RemoteRelationUpdate& u) { rig.capture.updates.push_back(u); });

  // Source hand near the middle of the table top.
  const Frame f{Eigen::Vector2d(5.0, 1.0), 0.0, std::nullopt, std::nullopt,
                Eigen::Vector3d(5.2, 4.1, 0.8)};
  Apply(rig, f);
  ASSERT_FALSE(rig.capture.updates.empty());
  const auto* affordance = rig.capture.Find(
      RelationKernelType::kAvatarAffordancePose, "hand", kObjectShared);
  ASSERT_NE(affordance, nullptr);
  ASSERT_EQ(affordance->support.sub_region_name, "top");
  EXPECT_NEAR(affordance->grounded_relation.proximity, 1.0, 1e-9);

  // Candidate hand at the OPPOSITE end of the SAME patch, exactly on the
  // surface (the old, buggy behavior scored R=1 there too).
  const Eigen::Vector3d wrong_end(5.9, 4.9, 0.8);
  TargetAnchorPose candidate{wrong_end, 0.0};
  const auto e = EvaluateTargetRelation(*affordance, candidate, *tree);
  ASSERT_TRUE(e.ok()) << TargetEvaluationDispositionName(e.disposition)
                      << ": " << e.reason;

  // The corresponding target support stays at the SOURCE's own contact
  // point (identity receiver): the candidate never moves it.
  const Eigen::Vector3d mapped_point = TopProjection(table, *f.hand);
  EXPECT_NEAR((e.support.contact_point_world - mapped_point).norm(), 0.0,
              1e-9);
  const double expected_dist = (wrong_end - mapped_point).norm();
  ASSERT_GT(expected_dist, 0.5);  // clearly the wrong end, not a rounding slip
  // Height and yaw fit are both 1 here (same z, same yaw as the mapped
  // point); only the position term departs from 1.
  EXPECT_NEAR(e.r_tgt,
             kApPositionWeight / (1.0 + expected_dist) + kApHeightWeight +
                 kApYawWeight,
             1e-9);
  EXPECT_LT(e.r_tgt, 0.95);  // must NOT score R=1 (or near it) at the wrong end

  // Diagnostic only: the candidate's own projection lands where IT stands,
  // and is NOT what r_tgt was scored against above.
  ASSERT_TRUE(e.has_candidate_projection);
  EXPECT_NEAR(
      (e.candidate_projected_support.contact_point_world - wrong_end).norm(),
      0.0, 1e-9);
}

// ===========================================================================
// Pose-fit / support-basis follow-ups: the proximity-vs-R split, R's yaw reference and
// formula version, and the explicit support fallback.
// ===========================================================================
namespace {
// One source hand over a 1 m x 1 m table top; the source frame yaw is applied
// to every anchor, so the hand's relative yaw (table yaw = 0) equals it.
struct HandRig {
  SourceRig rig;
  ObjectSpec table{5.0, 4.0, 0.0, 1.0, 1.0, 0.8, "top"};
  Frame frame;
  const RelationPayload* payload = nullptr;
  ElementTree* tree = nullptr;

  explicit HandRig(double source_yaw, double hand_height = 0.8) {
    rig.src = rig.ctx->AddSpace();
    tree = rig.src->GetElementTree();
    AddRoom(tree, 10.0, 8.0);
    AddObject(tree, table);
    rig.actor = AddActor(rig.src);
    EXPECT_NE(rig.src->BindScoped(rig.actor, "hand",
                                  RelationKernelType::kAvatarAffordancePose,
                                  kObject),
              nullptr);
    rig.src->SetHub(rig.ctx->GetHub());
    rig.ctx->GetHub()->SetRelationUpdateCallback(
        [&](const RemoteRelationUpdate& u) { rig.capture.updates.push_back(u); });
    frame = Frame{Eigen::Vector2d(5.0, 1.0), source_yaw, std::nullopt,
                  std::nullopt, Eigen::Vector3d(5.2, 4.1, hand_height)};
    Apply(rig, frame);
    payload = rig.capture.Find(RelationKernelType::kAvatarAffordancePose,
                               "hand", kObjectShared);
  }
};
}  // namespace

// The split: the source hand rests on the surface at a yaw 90 deg off the
// table's own frame. proximity (what phase hysteresis reads) is purely
// geometric, so the hand is in Contact; R (relation preservation) is the pose
// fit and is exactly 1 for the source against itself.
TEST(A4_AffordancePoseR, ContactPhaseUsesProximityNotPoseFit) {
  HandRig h(M_PI / 2);
  ASSERT_NE(h.payload, nullptr);
  EXPECT_NEAR(h.payload->grounded_relation.proximity, 1.0, 1e-9);
  Apply(h.rig, h.frame);  // a second frame: Active -> Driving -> Contact
  Apply(h.rig, h.frame);
  const auto states = h.rig.src->GetRelationStates(h.rig.actor);
  bool found = false;
  for (const auto& s : states) {
    if (s.kernel_type != RelationKernelType::kAvatarAffordancePose) continue;
    found = true;
    EXPECT_EQ(s.phase, RelationPhase::kContact)
        << "an anchor at a 90 deg yaw off the patch frame must still enter "
           "Contact: contact is decided on proximity, not on the pose fit";
    EXPECT_NEAR(s.proximity, 1.0, 1e-9);
  }
  EXPECT_TRUE(found);
  // R, source against itself (Eval_p at the source pose): exactly 1.
  const auto e = EvaluateTargetRelation(
      *h.payload, {*h.frame.hand, h.frame.yaw}, *h.tree);
  ASSERT_TRUE(e.ok());
  EXPECT_EQ(e.r_formula_version, kApRFormulaPoseFitV1);
  EXPECT_NEAR(e.r_src, 1.0, 1e-9);
  EXPECT_NEAR(e.r_tgt, 1.0, 1e-9);
}

// Yaw is scored against the SOURCE's relative yaw (encoded_state[2]), not the
// patch's own frame: a candidate at the source's yaw is exact even though
// that yaw is far from the patch frame; one 90 deg away from the source's
// yaw loses half of the yaw term.
TEST(A4_AffordancePoseR, YawIsRelativeToTheSourcesRelativeYaw) {
  HandRig h(1.0);  // source relative yaw = 1.0 rad off the table frame
  ASSERT_NE(h.payload, nullptr);
  ASSERT_EQ(h.payload->grounded_relation.encoded_state.size(), 3);
  EXPECT_NEAR(h.payload->grounded_relation.encoded_state[2], 1.0, 1e-9);
  const Eigen::Vector3d on_surface = *h.frame.hand;

  const auto same = EvaluateTargetRelation(*h.payload, {on_surface, 1.0},
                                           *h.tree);
  ASSERT_TRUE(same.ok());
  EXPECT_NEAR(same.r_src, 1.0, 1e-9);
  EXPECT_NEAR(same.r_tgt, 1.0, 1e-9);
  EXPECT_NEAR(same.error, 0.0, 1e-9);

  const auto turned = EvaluateTargetRelation(
      *h.payload, {on_surface, 1.0 + M_PI / 2}, *h.tree);
  ASSERT_TRUE(turned.ok());
  EXPECT_NEAR(turned.r_tgt, kApPositionWeight + kApHeightWeight +
                                 kApYawWeight * 0.5,
              1e-9);
  // Wrap-around: +2 pi is the same yaw.
  const auto wrapped = EvaluateTargetRelation(
      *h.payload, {on_surface, 1.0 + 2 * M_PI}, *h.tree);
  ASSERT_TRUE(wrapped.ok());
  EXPECT_NEAR(wrapped.r_tgt, 1.0, 1e-9);
}

// Unknown source relative yaw: yaw contributes to NEITHER R.
TEST(A4_AffordancePoseR, UnknownSourceYawDropsTheYawTermOnBothSides) {
  HandRig h(1.0, /*hand_height=*/0.95);  // 0.15 above the top
  ASSERT_NE(h.payload, nullptr);
  RelationPayload p = *h.payload;
  p.grounded_relation.encoded_state.conservativeResize(2);  // no yaw
  const auto e = EvaluateTargetRelation(
      p, {*h.frame.hand, 1.0 + M_PI}, *h.tree);  // yaw would be 180 deg off
  ASSERT_TRUE(e.ok());
  const double expected =
      (kApPositionWeight + kApHeightWeight / 1.15) /
      (kApPositionWeight + kApHeightWeight);
  EXPECT_NEAR(e.r_src, expected, 1e-9);
  EXPECT_NEAR(e.r_tgt, expected, 1e-9);
  EXPECT_NEAR(e.error, 0.0, 1e-9);
}

// A pre-pose-fit sender (no offset/height; R_src is the old proximity): BOTH sides
// are scored with the proximity component, never a new-formula R_tgt against
// an old R_src.
TEST(A4_AffordancePoseR, PreA4PayloadIsScoredByProximityOnBothSides) {
  HandRig h(1.0, /*hand_height=*/0.95);
  ASSERT_NE(h.payload, nullptr);
  RelationPayload old_sender = *h.payload;
  old_sender.grounded_relation.ap_offset_u_m =
      std::numeric_limits<double>::quiet_NaN();
  old_sender.grounded_relation.ap_offset_v_m =
      std::numeric_limits<double>::quiet_NaN();
  old_sender.grounded_relation.ap_height_m =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(AffordancePoseRFormulaVersion(old_sender.grounded_relation),
            kApRFormulaProximityOnly);
  EXPECT_EQ(AffordancePoseRFormulaVersion(h.payload->grounded_relation),
            kApRFormulaPoseFitV1);
  EXPECT_NEAR(old_sender.grounded_relation.proximity, 1.0 / 1.15, 1e-9);

  // Candidate 0.4 above the top, at a deliberately different yaw: the yaw
  // must not matter under the proximity-only formula.
  const auto e = EvaluateTargetRelation(
      old_sender, {Eigen::Vector3d(5.2, 4.1, 1.2), 1.0 + 2.0}, *h.tree);
  ASSERT_TRUE(e.ok());
  EXPECT_EQ(e.r_formula_version, kApRFormulaProximityOnly);
  EXPECT_NEAR(e.r_src, 1.0 / 1.15, 1e-9);
  EXPECT_NEAR(e.r_tgt, 1.0 / 1.4, 1e-9);
  EXPECT_NEAR(e.error, 1.0 / 1.4 - 1.0 / 1.15, 1e-9);
}

// The fallback to the candidate's own projection is explicit.
TEST(A3_MappedSupportFallback, FallbackIsExposedNotSilent) {
  HandRig h(0.0);
  ASSERT_NE(h.payload, nullptr);
  const auto mapped = EvaluateTargetRelation(
      *h.payload, {Eigen::Vector3d(5.9, 4.9, 0.8), 0.0}, *h.tree);
  ASSERT_TRUE(mapped.ok());
  EXPECT_EQ(mapped.support_basis, TargetSupportBasis::kMappedSupport);
  EXPECT_TRUE(mapped.reason.empty());

  // A legacy payload with a named part but no contact chart cannot decode the
  // mapped support.
  RelationPayload legacy = *h.payload;
  legacy.support.has_contact = false;
  const Eigen::Vector3d wrong_end(5.9, 4.9, 0.8);
  const auto fallback =
      EvaluateTargetRelation(legacy, {wrong_end, 0.0}, *h.tree);
  ASSERT_TRUE(fallback.ok());
  EXPECT_EQ(fallback.support_basis,
            TargetSupportBasis::kCandidateProjectionFallback);
  EXPECT_FALSE(fallback.reason.empty());
  ASSERT_TRUE(fallback.has_candidate_projection);
  // It really was scored against the candidate's own projection.
  EXPECT_NEAR((fallback.support.contact_point_world - wrong_end).norm(), 0.0,
              1e-9);
  // Families without a named part are not affected.
  EXPECT_EQ(EvaluateTargetRelation(*h.payload, {wrong_end, 0.0}, *h.tree)
                .support_basis,
            TargetSupportBasis::kMappedSupport);
}

// The target normalizes over the SAME context the source used — the
// navigable containing the subject anchor — even when the receiver hierarchy
// nests the referent in a region the source did not have. The source table is
// a room child; the receiver's table sits in a navigable "zone" inside the
// room while the receiver actor stands in the room outside it. A decode that
// walked up from the receiver referent's parent would normalize by the
// zone (d_tgt = 2 * zone/room diameter = 0.78 m instead of 2 m).
TEST(R1_3_TargetEvaluation, NormalizesOverTheAnchorScopeNotTheReferentParent) {
  SourceRig rig;
  rig.src = rig.ctx->AddSpace();
  AddRoom(rig.src->GetElementTree(), 10.0, 8.0);
  const ObjectSpec table{5.0, 4.0, 0.0, 1.0, 1.0, 0.8, "top"};
  AddObject(rig.src->GetElementTree(), table);
  rig.actor = AddActor(rig.src);
  rig.src->Bind(rig.actor, "root", RelationKernelType::kAvatarSpacing);
  rig.src->SetHub(rig.ctx->GetHub());
  rig.ctx->GetHub()->SetRelationUpdateCallback(
      [&](const RemoteRelationUpdate& u) { rig.capture.updates.push_back(u); });
  const Frame f{Eigen::Vector2d(5.0, 1.0), M_PI_2, std::nullopt, std::nullopt,
                std::nullopt};
  Apply(rig, f);
  const auto* spacing =
      rig.capture.Find(RelationKernelType::kAvatarSpacing, "root", kObjectShared);
  ASSERT_NE(spacing, nullptr);
  const double room_diag = std::hypot(10.0, 8.0);
  ASSERT_NEAR(spacing->grounded_relation.proximity, 1.0 - 2.0 / room_diag,
              1e-9);

  ElementTree receiver;
  AddRoom(&receiver, 10.0, 8.0);
  ASSERT_TRUE(receiver.AddElement(kZone, kRoom).ok());
  receiver.GetElement(kZone)->SetSharedId(kZoneShared);
  receiver.GetElement(kZone)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(Rect(3, 2.5, 7, 5.5)));
  AddObject(&receiver, table, kZone);

  // Eval_p at the source-equivalent state: the anchor's scope is the room.
  const auto e = EvaluateTargetRelation(
      *spacing, {Eigen::Vector3d(5.0, 1.0, 0.0), M_PI_2}, receiver);
  ASSERT_TRUE(e.ok()) << e.reason;
  EXPECT_EQ(e.normalizing_scope_id, kRoom);
  EXPECT_NEAR(e.normalizing_scope_diameter_m, room_diag, 1e-12);
  EXPECT_NEAR(e.r_tgt, spacing->grounded_relation.proximity, 1e-12);

  // Absolute placement (the one-shot decode) puts the root on the room's
  // distance level set, d* = 2 m. The zone would scale that to
  // 2 * zone/room. One polygon term's zero set is a curve, so Gauss-Newton
  // from the sample takes the minimum-norm step onto it, which is not the chart point (5, 1).
  const auto* room = dynamic_cast<const NavigableSpatialDefinition*>(
      receiver.GetElement(kRoom)->GetSpatialDefinition());
  RelationSolver solver;
  SolverWarmStart warm;
  warm.absolute = true;
  const SolverResult solved = solver.Solve(
      Eigen::Vector2d(1.0, 7.0), 0.0, {*spacing}, {}, receiver, room, nullptr,
      std::nullopt, warm);
  EXPECT_EQ(solved.status, SolverStatus::kConverged);
  // Eval_p scores the mapped source contact, a fixed point. The objective's
  // contact is the geodesic-nearest point,
  // so the minimum-norm point on that level set does not reproduce the
  // mapped-contact R. It is 2 m from the table, in the room's scope.
  const Eigen::Vector2d clamped(
      std::clamp(solved.position.x(), 4.0, 6.0),
      std::clamp(solved.position.y(), 3.0, 5.0));
  EXPECT_NEAR((solved.position - clamped).norm(), 2.0, 1e-3);
  const auto back = EvaluateTargetRelation(
      *spacing, {Eigen::Vector3d(solved.position.x(), solved.position.y(), 0),
                 M_PI_2},
      receiver);
  ASSERT_TRUE(back.ok());
  EXPECT_EQ(back.normalizing_scope_id, kRoom);
}

// ONE fallback rule for Eval_p and the decode, exercised through the
// realizer: whatever normalization mode the payload is in, the realizer's
// output state evaluates back to R^src.
struct SpacingCase {
  RelationPayload payload;
  ElementTree receiver;
};

RelationPayload EncodeSpacing(const ElementTree& tree,
                              const NavigableSpatialDefinition* scope,
                              const Eigen::Vector3d& root) {
  auto spacing = KernelFactory::Create(RelationKernelType::kAvatarSpacing);
  RelationPayload p;
  p.referent_shared_id = kObjectShared;
  p.subject_anchor_name = "root";
  p.kernel_type = RelationKernelType::kAvatarSpacing;
  p.support = spacing.grounding_rule->ComputeSupport(*tree.GetElement(kObject),
                                                     tree, root, M_PI_2, scope);
  p.grounded_relation = spacing.kernel->Encode(root, M_PI_2, p.support);
  return p;
}

void ExpectRealizerRoundTrip(const RelationPayload& payload,
                             const ElementTree& receiver,
                             element_id_t expected_scope) {
  const auto norm = ResolveSpacingNormalization(payload, receiver);
  EXPECT_EQ(norm.scope_element, expected_scope);
  const auto* room = dynamic_cast<const NavigableSpatialDefinition*>(
      receiver.GetElement(kRoom)->GetSpatialDefinition());
  RelationSolver solver;
  // The first sample is placed at the approach pose: the source offset through
  // the same-id support chart. The
  // receiver table is the source table, so the chart is the identity and the
  // placement is the source root (exact at placement), whatever the
  // normalization mode of the payload.
  SolverWarmStart warm;
  warm.absolute = true;
  const SolverResult s =
      solver.Solve(Eigen::Vector2d(1.0, 7.0), 0.0, {payload}, {}, receiver,
                   room, nullptr, std::nullopt, warm);
  ASSERT_EQ(s.status, SolverStatus::kConverged);
  EXPECT_LT((s.position - Eigen::Vector2d(5.0, 1.0)).norm(), 1e-9);
  const auto e = EvaluateTargetRelation(
      payload, {Eigen::Vector3d(s.position.x(), s.position.y(), 0.0), M_PI_2},
      receiver);
  ASSERT_TRUE(e.ok()) << e.reason;
  EXPECT_EQ(e.normalizing_scope_id, expected_scope);
}

TEST(R1_3_TargetEvaluation, FallbackRulesRoundTripThroughTheRealizer) {
  const ObjectSpec table{5.0, 4.0, 0.0, 1.0, 1.0, 0.8, "top"};
  ElementTree source;
  AddRoom(&source, 10.0, 8.0);
  AddObject(&source, table);
  const auto* src_room = dynamic_cast<const NavigableSpatialDefinition*>(
      source.GetElement(kRoom)->GetSpatialDefinition());
  const Eigen::Vector3d root(5.0, 1.0, 0.0);
  // Receiver: the table sits in a navigable zone nested in the room.
  ElementTree zoned;
  AddRoom(&zoned, 10.0, 8.0);
  ASSERT_TRUE(zoned.AddElement(kZone, kRoom).ok());
  zoned.GetElement(kZone)->SetSharedId(kZoneShared);
  zoned.GetElement(kZone)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(Rect(2, 0.5, 8, 6.5)));
  AddObject(&zoned, table, kZone);

  // 1. Scoped, scope named: the counterpart (the receiver room).
  RelationPayload scoped = EncodeSpacing(source, src_room, root);
  scoped.scope_shared_id = kRoomShared;
  ExpectRealizerRoundTrip(scoped, zoned, kRoom);

  // 2. Scoped, scope ids absent: the structural correspondent (the zone
  // enclosing the referent) on both sides.
  RelationPayload anonymous = scoped;
  anonymous.scope_shared_id = kInvalidSharedElementId;
  ExpectRealizerRoundTrip(anonymous, zoned, kZone);

  // 3. Linker-only receiver: the scope's shared id must not be used when the
  // receiver resolved (and rejected) it; the structural correspondent is.
  // (A linker-only receiver admits only linker-resolved referents.)
  RelationPayload linker_only = scoped;
  linker_only.scope_resolved_by_receiver = true;
  linker_only.resolved_local_element_id = kObject;
  ExpectRealizerRoundTrip(linker_only, zoned, kZone);

  // 4. Unscoped source (no navigable at encode time: R = 1/(1+d)): no scope
  // on either side.
  RelationPayload unscoped = EncodeSpacing(source, nullptr, root);
  ASSERT_LT(unscoped.grounded_relation.source_scope_diameter_m, 0.0);
  ASSERT_NEAR(unscoped.grounded_relation.proximity, 1.0 / 3.0, 1e-9);
  ExpectRealizerRoundTrip(unscoped, zoned, kInvalidElementId);

  // 5. Legacy sender (neither the scope diameter nor the metric distance):
  // R = 1 - d/D over the structural correspondent on both sides. The structural
  // correspondent is the zone. The placement does not read a distance: it is the
  // source offset through the chart, here the source root.
  RelationPayload legacy = anonymous;
  legacy.grounded_relation.source_scope_diameter_m = -1.0;
  legacy.grounded_relation.support_distance_m = -1.0;
  const double zone_diag = std::hypot(6.0, 6.0);
  legacy.grounded_relation.proximity = 1.0 - 1.5 / zone_diag;
  const auto legacy_norm = ResolveSpacingNormalization(legacy, zoned);
  EXPECT_EQ(legacy_norm.scope_element, kZone);
  EXPECT_EQ(legacy_norm.mode, SpacingNormalization::Mode::kLegacyScoped);
  const auto* legacy_room = dynamic_cast<const NavigableSpatialDefinition*>(
      zoned.GetElement(kRoom)->GetSpatialDefinition());
  RelationSolver solver;
  SolverWarmStart warm;
  warm.absolute = true;
  const SolverResult placed =
      solver.Solve(Eigen::Vector2d(1.0, 7.0), 0.0, {legacy}, {}, zoned,
                   legacy_room, nullptr, std::nullopt, warm);
  EXPECT_EQ(placed.status, SolverStatus::kConverged);
  EXPECT_NEAR((placed.position - Eigen::Vector2d(5.0, 1.0)).norm(), 0.0, 1e-9);
}

// ===========================================================================
// Missing counterparts are explicit outcomes, never centre targets.
// ===========================================================================
RelationPayload SeatPayload(shared_element_id_t referent,
                            const std::string& part, double ratio) {
  RelationPayload p;
  p.referent_shared_id = referent;
  p.subject_anchor_name = "pelvis";
  p.kernel_type = RelationKernelType::kAvatarAffordancePose;
  p.grounded_relation.encoded_state = Eigen::Vector3d(0.3, ratio, 0.1);
  p.grounded_relation.proximity = 1.0 / 1.1;
  p.support.has_contact = true;
  p.support.sub_region_name = part;
  p.support.contact_normalized.strategy =
      NormalizedPolygonPosition::Strategy::kRadial;
  p.support.contact_normalized.angle = 0.3;
  p.support.contact_normalized.radial_ratio = ratio;
  return p;
}

TEST(R1_2_ExplicitGrounding, GeometrylessReferentYieldsNoCentreTarget) {
  ElementTree tree;
  AddRoom(&tree, 10.0, 8.0);
  ASSERT_TRUE(tree.AddElement(kObject, kRoom).ok());
  tree.GetElement(kObject)->SetSharedId(kObjectShared);
  tree.GetElement(kObject)->SetLocalTransform(Pose(4.0, 2.0, 0.5, 0.4));
  RelationSolver solver;

  // AffordancePose, with and without a requested part.
  for (const std::string part : {std::string(), std::string("seat")}) {
    SCOPED_TRACE("part='" + part + "'");
    auto p = SeatPayload(kObjectShared, part, 0.5);
    auto grounded = solver.GroundAffordanceTarget(p, tree);
    ASSERT_FALSE(grounded.ok());
    EXPECT_EQ(grounded.status().code(),
              absl::StatusCode::kFailedPrecondition);
    const auto realized = solver.SolveActorTargets(
        Eigen::Vector2d(1, 1), 0.0, {p}, {1.0}, tree);
    EXPECT_EQ(realized.status, SolverStatus::kFallback);
    for (const auto& t : realized.targets) EXPECT_NE(t.anchor_name, "pelvis");
    const auto e = EvaluateTargetRelation(
        p, {Eigen::Vector3d(4.0, 2.0, 0.6), 0.0}, tree);
    EXPECT_EQ(e.disposition, TargetEvaluationDisposition::kNoSupportGeometry);
  }

  // Spacing whose source support was a real contact: the geometryless
  // counterpart gives no position goal, and the root holds (it is never
  // pulled onto the referent origin (4, 2)).
  RelationPayload spacing;
  spacing.referent_shared_id = kObjectShared;
  spacing.subject_anchor_name = "root";
  spacing.kernel_type = RelationKernelType::kAvatarSpacing;
  spacing.grounded_relation.encoded_state = Eigen::Vector3d(0.0, -1.0, 0.0);
  spacing.grounded_relation.proximity = 0.9;
  spacing.support.has_contact = true;
  // Every enabled relation dropped. The source scope is authoritative; a
  // referent the receiver cannot ground drops, and with none left there is nothing to
  // map: not a failure, not a hold (no_grounded_referent), the sample is echoed.
  const SolverResult held =
      solver.Solve(Eigen::Vector2d(7.0, 6.0), 0.25, {spacing}, tree);
  EXPECT_NE(held.status, SolverStatus::kFailed);
  EXPECT_TRUE(held.no_grounded_referent);
  EXPECT_NEAR((held.position - Eigen::Vector2d(7.0, 6.0)).norm(), 0.0, 1e-12);
  EXPECT_NEAR(held.yaw, 0.25, 1e-12);
  ASSERT_EQ(held.relations.size(), 1u);
  EXPECT_EQ(held.relations[0].disposition,
            SolverRelationDisposition::kGroundingFailed);
  EXPECT_FALSE(held.relations[0].reason.empty());

  // A partial drop is reported per relation; the rest still realizes.
  ASSERT_TRUE(tree.AddElement(5, kRoom).ok());
  tree.GetElement(5)->SetSharedId(205);
  tree.GetElement(5)->SetLocalTransform(Pose(2.0, 2.0, 0.0, 0.0));
  RelationPayload origin_spacing;  // the source support was the origin
  origin_spacing.referent_shared_id = 205;
  origin_spacing.subject_anchor_name = "root";
  origin_spacing.kernel_type = RelationKernelType::kAvatarSpacing;
  origin_spacing.grounded_relation.encoded_state = Eigen::Vector3d(0, 0, 0);
  origin_spacing.grounded_relation.proximity = 1.0;
  RelationPayload stranger = spacing;
  stranger.referent_shared_id = 999;
  // The remaining referent is a bare point at its own origin. A one-shot
  // decode holds that contact; a static
  // continuation would stay at the sample.
  SolverWarmStart warm;
  warm.absolute = true;
  const SolverResult partial = solver.Solve(
      Eigen::Vector2d(7.0, 6.0), 0.25, {spacing, origin_spacing, stranger},
      {1.0, 1.0, 1.0}, tree, nullptr, nullptr, std::nullopt, warm);
  EXPECT_EQ(partial.status, SolverStatus::kConverged);
  EXPECT_NEAR((partial.position - Eigen::Vector2d(2.0, 2.0)).norm(), 0.0, 1e-9);
  ASSERT_EQ(partial.relations.size(), 3u);
  auto outcome = [&](size_t i) {
    for (const auto& r : partial.relations)
      if (r.payload_index == i) return r.disposition;
    return SolverRelationDisposition::kUnsupported;
  };
  EXPECT_EQ(outcome(0), SolverRelationDisposition::kGroundingFailed);
  EXPECT_EQ(outcome(1), SolverRelationDisposition::kUsed);
  EXPECT_EQ(outcome(2), SolverRelationDisposition::kUnmapped);
}

TEST(R1_2_ExplicitGrounding, MissingPartIsReportedNotRetargeted) {
  ElementTree tree;
  AddRoom(&tree, 10.0, 8.0);
  AddObject(&tree, {5.0, 4.0, 0.3, 0.25, 0.25, 0.45, "seat"});
  RelationSolver solver;
  auto armrest = SeatPayload(kObjectShared, "armrest_l", 0.5);
  armrest.subject_anchor_name = "hand";
  auto grounded = solver.GroundAffordanceTarget(armrest, tree);
  ASSERT_FALSE(grounded.ok());
  EXPECT_EQ(grounded.status().code(), absl::StatusCode::kNotFound);
  const auto realized = solver.SolveActorTargets(Eigen::Vector2d(5, 3), 0.0,
                                                 {armrest}, {1.0}, tree);
  for (const auto& t : realized.targets) EXPECT_NE(t.anchor_name, "hand");
  EXPECT_EQ(realized.status, SolverStatus::kFallback);
  const auto e = EvaluateTargetRelation(
      armrest, {Eigen::Vector3d(5.0, 4.0, 0.6), 0.0}, tree);
  EXPECT_EQ(e.disposition, TargetEvaluationDisposition::kPartNotFound);
  // A referent with no counterpart at all is "unmapped".
  auto stranger = SeatPayload(999, "seat", 0.5);
  EXPECT_EQ(EvaluateTargetRelation(stranger, {Eigen::Vector3d::Zero(), 0.0},
                                   tree)
                .disposition,
            TargetEvaluationDisposition::kUnmapped);
}

// ===========================================================================
// A body-anchor target equals the mapped source contact while engaged
// and follows the body (no support target) otherwise; reach never moves it.
// ===========================================================================
struct TwinRig {
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space* src = nullptr;
  Space* tgt = nullptr;
  spatialblend::entity_id_t src_actor = 0, tgt_actor = 0;
};

TwinRig MakeTwin(const ObjectSpec& source_object,
                 const ObjectSpec& target_object) {
  TwinRig rig;
  rig.src = rig.ctx->AddSpace();
  rig.tgt = rig.ctx->AddSpace();
  AddRoom(rig.src->GetElementTree(), 10.0, 8.0);
  AddRoom(rig.tgt->GetElementTree(), 10.0, 8.0);
  AddObject(rig.src->GetElementTree(), source_object);
  AddObject(rig.tgt->GetElementTree(), target_object);
  rig.src_actor = AddActor(rig.src);
  rig.tgt_actor = AddActor(rig.tgt);
  rig.src->SetHub(rig.ctx->GetHub());
  rig.tgt->SetHub(rig.ctx->GetHub());
  return rig;
}

const RealizationTarget* FindTarget(const Space* space,
                                    spatialblend::entity_id_t actor,
                                    const std::string& anchor,
                                    RelationKernelType kernel) {
  const RealizationResult* r = space->GetRealizationTargets(actor);
  if (!r) return nullptr;
  for (const auto& t : r->targets)
    if (t.anchor_name == anchor && t.kernel_type == kernel) return &t;
  return nullptr;
}

TEST(R2_6_BodyAnchorTargets, EqualsSourceContactWhileEngagedFollowsBodyAfter) {
  const ObjectSpec table{5.0, 4.0, 0.0, 1.0, 1.0, 0.8, "top"};
  TwinRig rig = MakeTwin(table, table);  // R0: identity receiver
  rig.src->Bind(rig.src_actor, "root", RelationKernelType::kAvatarSpacing);
  ASSERT_NE(rig.src->BindScoped(rig.src_actor, "hand",
                                RelationKernelType::kAvatarAffordancePose,
                                kObject),
            nullptr);

  // Engagement is the hand TOUCHING the table top (no distance threshold,
  // no release dwell). Frames 0-9: the hand hovers 2 cm above the top (no target,
  // the retired 5 cm enter distance would have engaged). Frames 10-29: the hand
  // rests on the top, sliding slowly (engaged). Frames 30-59: the actor walks away
  // south, hand at its side (released AT ONCE, no held target).
  int hover_frames = 0, engaged_frames = 0, released_frames = 0;
  for (int k = 0; k < 60; ++k) {
    SCOPED_TRACE("frame " + std::to_string(k));
    const bool hover = k < 10, on_table = k >= 10 && k < 30;
    const double y_root = k < 30 ? 2.6 : 2.6 - 0.05 * (k - 29);
    const Eigen::Vector3d hand =
        hover ? Eigen::Vector3d(5.0 + 0.01 * k, 3.3, 0.82)
        : on_table ? Eigen::Vector3d(5.0 + 0.01 * k, 3.3, 0.8)
                   : Eigen::Vector3d(5.3, y_root, 0.9);
    Apply(rig.src, rig.src_actor,
          {Eigen::Vector2d(5.0, y_root), M_PI_2, std::nullopt, std::nullopt,
           hand});
    const auto* t = FindTarget(rig.tgt, rig.tgt_actor, "hand",
                               RelationKernelType::kAvatarAffordancePose);
    // Independent oracle: the source contact = the hand projected onto the
    // table top; engaged iff the hand touches it.
    const Eigen::Vector3d contact = TopProjection(table, hand);
    const bool touching = hand.head<2>() == contact.head<2>() && hand.z() <= contact.z();
    if (touching) {
      ASSERT_NE(t, nullptr) << "engaged source contact must be realized";
      // It IS the source contact (identity receiver): no clamp, no blend.
      EXPECT_NEAR((Eigen::Vector3d(t->position.x(), t->position.y(), t->z) -
                   contact).norm(), 0.0, 1e-9);
      EXPECT_FALSE(t->has_requested_position);
      ++engaged_frames;
    } else {
      EXPECT_EQ(t, nullptr) << "not touching: the anchor follows the body";
      if (hover) ++hover_frames; else ++released_frames;
    }
  }
  EXPECT_EQ(hover_frames, 10);
  EXPECT_EQ(engaged_frames, 20);
  EXPECT_EQ(released_frames, 30);
}

TEST(R2_6_BodyAnchorTargets, ReachNeverMovesTheEngagedContact) {
  const ObjectSpec table{5.0, 4.0, 0.0, 1.0, 1.0, 0.8, "top"};
  TwinRig rig = MakeTwin(table, table);
  rig.src->Bind(rig.src_actor, "root", RelationKernelType::kAvatarSpacing);
  ASSERT_NE(rig.src->BindScoped(rig.src_actor, "hand",
                                RelationKernelType::kAvatarAffordancePose,
                                kObject),
            nullptr);
  rig.tgt->SetAnchorReach("hand", 0.3);
  const Eigen::Vector3d hand(5.2, 3.4, 0.8);  // touching
  Apply(rig.src, rig.src_actor,
        {Eigen::Vector2d(5.0, 2.6), M_PI_2, std::nullopt, std::nullopt, hand});
  const auto* root = FindTarget(rig.tgt, rig.tgt_actor, "root",
                                RelationKernelType::kAvatarSpacing);
  const auto* t = FindTarget(rig.tgt, rig.tgt_actor, "hand",
                             RelationKernelType::kAvatarAffordancePose);
  ASSERT_NE(root, nullptr);
  ASSERT_NE(t, nullptr);
  const Eigen::Vector3d contact = TopProjection(table, hand);
  EXPECT_NEAR(t->position.x(), contact.x(), 1e-9);
  EXPECT_NEAR(t->position.y(), contact.y(), 1e-9);
  EXPECT_NEAR(t->z, contact.z(), 1e-9);
  EXPECT_FALSE(t->has_requested_position);
  EXPECT_TRUE(t->beyond_reach);
  EXPECT_NEAR(t->reach_excess_m,
              (contact.head<2>() - root->position).norm() - 0.3, 1e-9);
}

// Rich channel: one engaged contact among several contributors is an exact
// support (not a blend), and "admitted but not engaged" is a normal state.
TEST(R2_6_BodyAnchorTargets, RichChannelReportsEngagedExactOrAwaiting) {
  Space target;
  target.SetRichRealizationEnabled(true);
  auto* tree = target.GetElementTree();
  AddRoom(tree, 10.0, 8.0);
  const ObjectSpec table{5.0, 4.0, 0.0, 1.0, 1.0, 0.8, "top"};
  AddObject(tree, table);
  constexpr element_id_t kOther = 6;
  constexpr shared_element_id_t kOtherShared = 206;
  {
    ASSERT_TRUE(tree->AddElement(kOther, kRoom).ok());
    tree->GetElement(kOther)->SetSharedId(kOtherShared);
    tree->GetElement(kOther)->SetLocalTransform(Pose(8.0, 2.0, 0.0, 0.0));
    auto def = std::make_shared<PolytopeSpatialDefinition>(
        Eigen::Vector3d(-0.5, -0.5, 0), Eigen::Vector3d(0.5, 0.5, 0.8),
        GeometryRole::kObstacle);
    SubRegion top;
    top.name = "top";
    top.local_transform = Pose(0, 0, 0.8, 0);
    top.boundary = Rect(-0.5, -0.5, 0.5, 0.5);
    def->AddSubRegion(top);
    tree->GetElement(kOther)->SetSpatialDefinition(def);
  }
  const auto actor = AddActor(&target);
  auto hand = [&](shared_element_id_t referent, double d) {
    RelationPayload p = SeatContactPayload(
        {0, 0, 0, 1.0, 1.0, 0.8, "top"}, Eigen::Vector2d(0.3, 0.1));
    p.referent_shared_id = referent;
    p.subject_anchor_name = "hand";
    p.grounded_relation.proximity = 1.0 / (1.0 + d);
    p.grounded_relation.support_distance_m = d;
    // A source dwelling at d: v1's count (1 in contact, none when static off contact). Without a count Progressive
    // computes no weight.
    p.update_count = SourceUpdateCount(d, d);
    return p;
  };
  auto channel = [&]() -> const RealizationChannel* {
    const auto* snap = target.GetActorRealizationSnapshot(actor);
    if (!snap) return nullptr;
    for (const auto& c : snap->channels)
      if (c.anchor_name == "hand") return &c;
    return nullptr;
  };
  RemoteRelationUpdate update;
  update.actor_shared_id = kActorShared;
  update.payloads = {hand(kObjectShared, 0.0), hand(kOtherShared, 0.4)};  // touching / not
  target.OnRemoteUpdate(update);
  const auto* engaged = channel();
  ASSERT_NE(engaged, nullptr);
  EXPECT_TRUE(engaged->has_goal);
  // Exact support; flagged as reach-projected only when beyond reach (the
  // actor's root is far from the table here), never moved.
  EXPECT_EQ(engaged->geometry, engaged->target.beyond_reach
                                   ? RealizationGeometry::kReachProjected
                                   : RealizationGeometry::kExactSupport);
  EXPECT_EQ(engaged->goal_kind, RealizationGoalKind::kSupportPose);
  EXPECT_EQ(engaged->target.referent_id, kObjectShared);
  EXPECT_TRUE(engaged->target.engaged_support);

  // Hand lifted 0.3 m: no longer touching, so released on its own solve
  // (no hysteresis, no release dwell). A lifted hand has no valid count (it recedes, then dwells off contact): v1 sends
  // no request and computes no weight, so on this continuation its contributors carry
  // weight 0 and are reported disabled, not "awaiting engagement" (which needs an included contributor). A channel with
  // no goal that is not awaiting engagement is an explicit release, whose status is kFallback (space.cc, the producer's
  // release rule); it was kConverged only because "awaiting" kept it.
  update.payloads = {hand(kObjectShared, 0.3), hand(kOtherShared, 0.4)};
  target.OnRemoteUpdate(update);
  const auto* awaiting = channel();
  ASSERT_NE(awaiting, nullptr);
  EXPECT_FALSE(awaiting->has_goal);
  EXPECT_EQ(awaiting->status, SolverStatus::kFallback);
  for (const auto& c : awaiting->contributors) {
    EXPECT_EQ(c.applied_weight, 0.0);
    EXPECT_EQ(c.disposition, RealizationDisposition::kDisabled);
  }
}

// There is no release dwell, so a delivery with no time source at all
// (no timestamp, source clock or step duration) releases a lifted contact on its
// own delivery too. (Retired: "updates count at a nominal 30 Hz so the held contact
// expires after ceil(dwell * 30) updates".)
TEST(R2_6_BodyAnchorTargets, ClocklessDeliveriesReleaseAtOnce) {
  Space target;
  auto* tree = target.GetElementTree();
  AddRoom(tree, 10.0, 8.0);
  AddObject(tree, {5.0, 4.0, 0.0, 1.0, 1.0, 0.8, "top"});
  const auto actor = AddActor(&target);
  auto hand = [&](double d) {
    RelationPayload p = SeatContactPayload(
        {0, 0, 0, 1.0, 1.0, 0.8, "top"}, Eigen::Vector2d(0.3, 0.1));
    p.referent_shared_id = kObjectShared;
    p.subject_anchor_name = "hand";
    p.grounded_relation.proximity = 1.0 / (1.0 + d);
    p.grounded_relation.support_distance_m = d;
    // A source dwelling at d: v1's count (1 in contact, none when static off contact). Without a count Progressive
    // computes no weight.
    p.update_count = SourceUpdateCount(d, d);
    return p;
  };
  auto has_hand = [&] {
    return FindTarget(&target, actor, "hand",
                      RelationKernelType::kAvatarAffordancePose) != nullptr;
  };
  RemoteRelationUpdate update;
  update.actor_shared_id = kActorShared;
  update.step_duration = 0.0;  // no time source at all
  update.source_clock = 0.0;
  update.payloads = {hand(0.0)};
  target.OnRemoteUpdate(update);
  ASSERT_TRUE(has_hand());
  update.payloads = {hand(0.3)};  // lifted
  target.OnRemoteUpdate(update);
  EXPECT_FALSE(has_hand()) << "a lifted contact is released on its own delivery";
}

// The engagement predicate is the geometric contact (engagement.h). With the
// pose terms it is "inside the part's footprint and not above its plane"; a
// sender that predates them reports only the metric distance (0 = touching) or R =
// 1 / (1 + d) (1 = touching); a non-affordance payload never engages by it.
TEST(R2_6_BodyAnchorTargets, EngagementIsTheGeometricContactPredicate) {
  GroundedRelation g;
  g.ap_offset_u_m = g.ap_offset_v_m = 0.0;
  g.ap_height_m = 0.0;
  EXPECT_TRUE(IsAffordanceContact(g)) << "on the plane";
  g.ap_height_m = -0.02;
  EXPECT_TRUE(IsAffordanceContact(g)) << "having crossed it";
  g.ap_height_m = kContactEpsilonM;
  EXPECT_FALSE(IsAffordanceContact(g)) << "a height of the contact epsilon is not contact (v1: d < eps)";
  g.ap_height_m = 0.5 * kContactEpsilonM;
  EXPECT_TRUE(IsAffordanceContact(g)) << "below the contact epsilon";
  g.ap_height_m = 0.0;
  g.ap_offset_u_m = kContactEpsilonM;
  EXPECT_FALSE(IsAffordanceContact(g)) << "outside the part's footprint by the contact epsilon";
  GroundedRelation metric;  // no pose terms
  metric.support_distance_m = 0.0;
  EXPECT_TRUE(IsAffordanceContact(metric));
  metric.support_distance_m = 0.04;
  metric.proximity = 1.0;  // the metric wins over R
  EXPECT_FALSE(IsAffordanceContact(metric));
  GroundedRelation legacy;  // neither terms nor metric distance
  legacy.proximity = 1.0;
  EXPECT_TRUE(IsAffordanceContact(legacy));
  legacy.proximity = 1.0 / 1.04;
  EXPECT_FALSE(IsAffordanceContact(legacy));
  RelationPayload p = SeatContactPayload({0, 0, 0, 1, 1, 0.8, "top"}, {0, 0});
  EXPECT_TRUE(IsEngaged(p));
  p.kernel_type = RelationKernelType::kAvatarFacing;
  EXPECT_FALSE(IsEngaged(p)) << "Facing never engages";
  p.kernel_type = RelationKernelType::kAvatarSpacing;
  EXPECT_FALSE(IsEngaged(p)) << "Spacing engages by its footprint / R == 1 of the edge";
  EXPECT_TRUE(std::isinf(AffordanceSupportDistance(p)));
}

// Grounding probe: a seated source whose SPACING referent (a TV) has no receiver
// counterpart. The tag "tv" does not narrow the root's Spacing to the TV alone, and an
// unmapped referent DROPS instead of failing the solve. The mapped referents (the sofa) still answer the root, the TV row is
// reported unmapped, and the engaged seat target is realized.
TEST(R1_2_ExplicitGrounding, UnmappedSpacingDropsAndKeepsSeatTarget) {
  const ObjectSpec sofa{5.0, 4.0, 0.4, 0.9, 0.3, 0.45, "seat"};
  const ObjectSpec chair{3.0, 5.0, -0.7, 0.25, 0.25, 0.45, "seat"};
  TwinRig rig = MakeTwin(sofa, chair);
  constexpr element_id_t kTv = 7;
  auto* src_tree = rig.src->GetElementTree();
  ASSERT_TRUE(src_tree->AddElement(kTv, kRoom).ok());
  src_tree->GetElement(kTv)->SetSharedId(707);
  src_tree->GetElement(kTv)->SetTag("tv");
  src_tree->GetElement(kTv)->SetLocalTransform(Pose(5.0, 0.5, 0.0, 0.0));
  src_tree->GetElement(kTv)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-0.6, -0.05, 0.5), Eigen::Vector3d(0.6, 0.05, 1.2),
          GeometryRole::kObstacle));
  // Root spacing ONLY to the TV; the pelvis sits on the sofa.
  rig.src->Bind(rig.src_actor, "root", RelationKernelType::kAvatarSpacing,
                "tv");
  ASSERT_NE(rig.src->BindScoped(rig.src_actor, "pelvis",
                                RelationKernelType::kAvatarAffordancePose,
                                kObject),
            nullptr);
  const double t0 = rig.tgt->GetSpaceClock();
  for (int k = 0; k < 40; ++k) {  // longer than the seat qualification
    const Eigen::Vector3d pelvis =
        ObjectPoint(sofa, 0.30 + 0.03 * (k % 2), 0.05, sofa.top);  // touching
    // The root is tracked 1.2 m toward the TV from the pelvis, outside every footprint: a root
    // INSIDE the sofa's footprint has the sofa alone in its scope, and then
    // the TV is not a referent the source sends at all.
    Apply(rig.src, rig.src_actor,
          {pelvis.head<2>() + Eigen::Vector2d(0.0, -1.2), Wrap(sofa.yaw - M_PI_2), std::nullopt, pelvis,
           std::nullopt});
  }
  ASSERT_GT(rig.tgt->GetSpaceClock() - t0, 0.5);
  const auto& last = rig.tgt->GetReceivedUpdates().back().payloads;
  bool tv_spacing = false;
  for (const auto& p : last)
    tv_spacing |= p.referent_shared_id == 707 &&
                  p.kernel_type == RelationKernelType::kAvatarSpacing;
  ASSERT_TRUE(tv_spacing) << "fixture must send the unmapped TV spacing";
  const RealizationResult* r = rig.tgt->GetRealizationTargets(rig.tgt_actor);
  ASSERT_NE(r, nullptr);
  EXPECT_NE(r->status, SolverStatus::kFailed);  // the unmapped TV dropped, nothing failed
  EXPECT_NE(FindTarget(rig.tgt, rig.tgt_actor, "root",
                       RelationKernelType::kAvatarSpacing),
            nullptr)
      << "the mapped Spacing referents still answer the root";
  const auto* seat = FindTarget(rig.tgt, rig.tgt_actor, "pelvis",
                                RelationKernelType::kAvatarAffordancePose);
  ASSERT_NE(seat, nullptr);
  EXPECT_TRUE(seat->engaged_support);
}

// ===========================================================================
// One facing authority while seated; root yaw tracks the source's
// facing semantics, continuously.
// ===========================================================================
struct SitRig {
  TwinRig rig;
  ObjectSpec source_seat, target_seat;
};

SitRig MakeSitRig(const ObjectSpec& source_seat, const ObjectSpec& target_seat,
                  const std::string& chest_tag = "") {
  SitRig s{MakeTwin(source_seat, target_seat), source_seat, target_seat};
  s.rig.src->Bind(s.rig.src_actor, "root", RelationKernelType::kAvatarSpacing);
  s.rig.src->Bind(s.rig.src_actor, "chest", RelationKernelType::kAvatarFacing,
                  chest_tag);
  EXPECT_NE(s.rig.src->BindScoped(s.rig.src_actor, "pelvis",
                                  RelationKernelType::kAvatarAffordancePose,
                                  kObject),
            nullptr);
  return s;
}

// A source frame with the pelvis at seat-local (u, v), `h` above the seat,
// facing `seat_relative` in the seat frame.
Frame SitFrame(const ObjectSpec& seat, double u, double v, double h,
               double seat_relative) {
  const Eigen::Vector3d pelvis = ObjectPoint(seat, u, v, seat.top + h);
  return {pelvis.head<2>(), Wrap(seat.yaw + seat_relative),
          Eigen::Vector3d(pelvis.x(), pelvis.y(), 1.2), pelvis, std::nullopt};
}

// Sits still for `frames` source updates (a small horizontal sway over the
// seat so every frame propagates; the height above the seat is constant, so
// the seat anchor is vertically still). Returns the receiver clock span.
double SitStill(SitRig& s, double u, double v, double h, double seat_relative,
                int frames) {
  const double t0 = s.rig.tgt->GetSpaceClock();
  for (int k = 0; k < frames; ++k)
    Apply(s.rig.src, s.rig.src_actor,
          SitFrame(s.source_seat, u + 0.03 * (k % 2), v, h, seat_relative));
  return s.rig.tgt->GetSpaceClock() - t0;
}

void ExpectSeatRelativeRootFacing(const ObjectSpec& source_seat,
                                  const ObjectSpec& target_seat,
                                  double lateral_u) {
  SitRig s = MakeSitRig(source_seat, target_seat);
  // Off-centre sit: the pelvis ON the seat (touching), `lateral_u` along
  // it, facing out of the seat front (-v) turned by +0.1 rad.
  const double seat_relative_facing = -M_PI_2 + 0.1;
  SitStill(s, lateral_u, 0.05, 0.0, seat_relative_facing, 40);
  const auto* root = FindTarget(s.rig.tgt, s.rig.tgt_actor, "root",
                                RelationKernelType::kAvatarSpacing);
  ASSERT_NE(root, nullptr);
  // The seat pose is the PELVIS anchor's proposal. Its yaw is the
  // receiver seat's yaw plus the source anchor-relative yaw (the harness's
  // sit.facing_seat check).
  const auto* seat = FindTarget(s.rig.tgt, s.rig.tgt_actor, "pelvis",
                                RelationKernelType::kAvatarAffordancePose);
  ASSERT_NE(seat, nullptr);
  EXPECT_TRUE(seat->engaged_support);
  EXPECT_NEAR(Wrap(seat->yaw - target_seat.yaw), seat_relative_facing, 1e-6);
  // The root's yaw is NOT overridden by the seat (the old "one facing authority"
  // contract): it stays the Facing solve's yaw.
  EXPECT_GT(std::abs(Wrap(root->yaw - seat->yaw)), 1e-3);
  ReceiverProposal proposal;
  ASSERT_TRUE(s.rig.tgt->ReadProposal(s.rig.tgt_actor, &proposal));
  EXPECT_NEAR(proposal.yaw, root->yaw, 1e-12);
  const AnchorProposal* pelvis = nullptr;
  for (const auto& a : proposal.anchors)
    if (a.anchor_name == "pelvis") pelvis = &a;
  ASSERT_NE(pelvis, nullptr) << "one proposal per relation-bearing anchor";
  EXPECT_TRUE(pelvis->engaged());
  EXPECT_NEAR(Wrap(pelvis->yaw - target_seat.yaw), seat_relative_facing, 1e-6);
  EXPECT_NEAR(pelvis->x, seat->position.x(), 1e-12);
  EXPECT_NEAR(pelvis->y, seat->position.y(), 1e-12);
  EXPECT_NEAR(pelvis->z, seat->z, 1e-12);
}

TEST(R2_5_RootFacing, OffCentreSitSofaToChairFacesSeatRelative) {
  ExpectSeatRelativeRootFacing({5.0, 4.0, 0.4, 0.9, 0.3, 0.45, "seat"},
                               {3.0, 5.0, -0.7, 0.25, 0.25, 0.45, "seat"},
                               0.30);
}

TEST(R2_5_RootFacing, OffCentreSitChairToSofaFacesSeatRelative) {
  ExpectSeatRelativeRootFacing({3.0, 5.0, -0.7, 0.25, 0.25, 0.45, "seat"},
                               {5.0, 4.0, 0.4, 0.9, 0.3, 0.45, "seat"},
                               0.10);
}

// A seat target requires the pelvis to TOUCH the seat. None of these touch a
// high stool: standing beside it (not over it), 0.35 m above it, and a pelvis
// passing 0.05..0.3 m over it while rising.
TEST(R2_5_RootFacing, StandingNearAHighStoolDoesNotTouchItSoThereIsNoSeatTarget) {
  const ObjectSpec stool{5.0, 4.0, 0.0, 0.17, 0.17, 0.75, "seat"};
  auto expect_no_seat = [&](SitRig& s, const Frame& last) {
    EXPECT_EQ(FindTarget(s.rig.tgt, s.rig.tgt_actor, "pelvis",
                         RelationKernelType::kAvatarAffordancePose),
              nullptr);
    const auto* root = FindTarget(s.rig.tgt, s.rig.tgt_actor, "root",
                                  RelationKernelType::kAvatarSpacing);
    ASSERT_NE(root, nullptr);
    // Seat authority is the missing pelvis target above. Whether the root yaw is
    // specified follows the in-body rule alone (a Facing is excluded only when its look-at point is in the body):
    // a source that jitters beside or over the stool walks the receiver onto the
    // stool's edge, where the look-at is in the body. That is not seat authority.
    (void)root; (void)last;
  };
  for (const double u : {0.17 + 0.12, 0.0}) {  // beside / over
    const double h = u > 0.17 ? 0.16 : 0.35;
    SitRig s = MakeSitRig(stool, stool);
    SitStill(s, u, 0.0, h, 0.7, 40);
    expect_no_seat(s, SitFrame(stool, u, 0.0, h, 0.7));
  }
  SitRig rising = MakeSitRig(stool, stool);
  Frame last{};
  for (int k = 0; k < 25; ++k) {
    last = SitFrame(stool, 0.03 * (k % 2), 0.0, 0.05 + 0.01 * k, 0.7);
    Apply(rising.rig.src, rising.rig.src_actor, last);
  }
  expect_no_seat(rising, last);
}

// A pelvis ON the stool top touches it, and engages at once -- no
// qualification, no stillness. (The retired contract: a perch 0.15 m over the stool
// engaged only after 0.48 s + 0.5 s of stillness.)
TEST(R2_5_RootFacing, ASeatedPelvisOnTheStoolTopEngagesAtOnce) {
  const ObjectSpec stool{5.0, 4.0, 0.0, 0.17, 0.17, 0.75, "seat"};
  SitRig s = MakeSitRig(stool, stool);
  Apply(s.rig.src, s.rig.src_actor, SitFrame(stool, 0.0, 0.0, 0.0, 0.7));
  const auto* seat = FindTarget(s.rig.tgt, s.rig.tgt_actor, "pelvis",
                                RelationKernelType::kAvatarAffordancePose);
  ASSERT_NE(seat, nullptr) << "touching the surface engages on the first frame";
  EXPECT_TRUE(seat->engaged_support);
  // Hovering 0.15 m over it is not touching: no seat target.
  SitRig hover = MakeSitRig(stool, stool);
  for (int k = 0; k < 40; ++k)
    Apply(hover.rig.src, hover.rig.src_actor,
          SitFrame(stool, 0.03 * (k % 2), 0.0, 0.15, 0.7));
  EXPECT_EQ(FindTarget(hover.rig.tgt, hover.rig.tgt_actor, "pelvis",
                       RelationKernelType::kAvatarAffordancePose),
            nullptr);
}

// ---------------------------------------------------------------------------
// The seat qualification suite (a COUCH-style smoothed stillness over
// 0.48 s plus a 0.5 s hold, with its Noise / CouchWouldEngage oracles and 14
// tests) is RETIRED: seated means the relation-bearing pelvis touches the
// sittable surface. There is no stillness
// timer, hold, dwell or hysteresis in core any more; the geometric replacement
// tests are R2_6_SeatGeometric in space/seat_geometry_test.cc.
// ---------------------------------------------------------------------------

TEST(R1_2_ExplicitGrounding, UnmappedFacingNeverSuppressesSeatOrRoot) {
  const ObjectSpec sofa{5.0, 4.0, 0.4, 0.9, 0.3, 0.45, "seat"};
  const ObjectSpec chair{3.0, 5.0, -0.7, 0.25, 0.25, 0.45, "seat"};
  // The chest faces only the TV, so the facing group has no mapped referent.
  SitRig s = MakeSitRig(sofa, chair, "tv");
  constexpr element_id_t kTv = 7;
  auto* src_tree = s.rig.src->GetElementTree();
  ASSERT_TRUE(src_tree->AddElement(kTv, kRoom).ok());
  src_tree->GetElement(kTv)->SetTag("tv");
  src_tree->GetElement(kTv)->SetSharedId(707);
  src_tree->GetElement(kTv)->SetLocalTransform(Pose(5.0, 0.5, 0.0, 0.0));
  src_tree->GetElement(kTv)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-0.6, -0.05, 0.5), Eigen::Vector3d(0.6, 0.05, 1.2),
          GeometryRole::kObstacle));
  SitStill(s, 0.30, 0.05, 0.0, -M_PI_2 + 0.1, 40);
  const RealizationResult* r = s.rig.tgt->GetRealizationTargets(s.rig.tgt_actor);
  ASSERT_NE(r, nullptr);
  EXPECT_NE(r->status, SolverStatus::kFailed);
  EXPECT_NE(FindTarget(s.rig.tgt, s.rig.tgt_actor, "root",
                       RelationKernelType::kAvatarSpacing),
            nullptr);
  EXPECT_NE(FindTarget(s.rig.tgt, s.rig.tgt_actor, "pelvis",
                       RelationKernelType::kAvatarAffordancePose),
            nullptr);
  // The same payloads through the two-stage realizer: the facing relation to
  // the TV is reported unmapped, the status comes from the position solve.
  const auto& payloads = s.rig.tgt->GetReceivedUpdates().back().payloads;
  bool saw_tv = false;
  for (const auto& p : payloads) saw_tv |= p.referent_shared_id == 707;
  ASSERT_TRUE(saw_tv) << "fixture must send a facing relation to the TV";
  RelationSolver solver;
  const RealizationResult two_stage = solver.SolveActorTargets(
      Eigen::Vector2d(3.0, 5.0), 0.0, payloads, {}, *s.rig.tgt->GetElementTree());
  EXPECT_NE(two_stage.status, SolverStatus::kFailed);
  bool facing_unmapped = false;
  for (const auto& o : two_stage.relations) {
    const auto& p = payloads[o.payload_index];
    if (p.referent_shared_id == 707 &&
        p.kernel_type == RelationKernelType::kAvatarFacing)
      facing_unmapped |= o.disposition == SolverRelationDisposition::kUnmapped;
  }
  EXPECT_TRUE(facing_unmapped);
  bool pelvis = false;
  for (const auto& t : two_stage.targets) pelvis |= t.anchor_name == "pelvis";
  EXPECT_TRUE(pelvis);
}

// Facing payloads of a source walking along x = 5 (anchor at `y`, heading
// `yaw`) toward, through and past the table, plus the root spacing payload
// that carries the source's referent-relative heading.
struct WalkPayloads {
  RelationPayload facing, spacing;
  Eigen::Vector2d contact;
};
WalkPayloads EncodeWalk(const ElementTree& tree,
                        const NavigableSpatialDefinition* room, double y,
                        double yaw) {
  auto facing = KernelFactory::Create(RelationKernelType::kAvatarFacing);
  auto spacing = KernelFactory::Create(RelationKernelType::kAvatarSpacing);
  const Eigen::Vector3d anchor(5.0, y, 1.3);
  const Eigen::Vector3d root(5.0, y, 0.0);
  WalkPayloads w;
  w.facing.referent_shared_id = kObjectShared;
  w.facing.subject_anchor_name = "root";
  w.facing.kernel_type = RelationKernelType::kAvatarFacing;
  w.facing.support = facing.grounding_rule->ComputeSupport(
      *tree.GetElement(kObject), tree, anchor, yaw, room);
  w.facing.grounded_relation = facing.kernel->Encode(anchor, yaw, w.facing.support);
  w.spacing.referent_shared_id = kObjectShared;
  w.spacing.subject_anchor_name = "root";
  w.spacing.kernel_type = RelationKernelType::kAvatarSpacing;
  w.spacing.support = spacing.grounding_rule->ComputeSupport(
      *tree.GetElement(kObject), tree, root, yaw, room);
  w.spacing.grounded_relation =
      spacing.kernel->Encode(root, yaw, w.spacing.support);
  w.contact = w.facing.support.contact_point_local;
  return w;
}

TEST(R2_5_RootFacing, YawTracksTheSourceThroughAFootprintCrossing) {
  // The source walks straight through a table footprint along x = 5 facing
  // +Y and turns by 30 degrees while inside it. Its facing contact jumps from
  // the near edge to the far edge when it enters the footprint, and past the
  // table it faces away from it. The receiver root is 0.3 m to the side of
  // the source-equivalent line (a realistic placement error), which turned
  // that jump into a 45-90 degree bearing snap and later froze the yaw 63
  // degrees off.
  ElementTree src_tree, tgt_tree;
  AddRoom(&src_tree, 10.0, 8.0);
  AddRoom(&tgt_tree, 10.0, 8.0);
  const ObjectSpec table{5.0, 4.0, 0.0, 1.0, 1.0, 0.8, "top"};
  AddObject(&src_tree, table);
  AddObject(&tgt_tree, table);
  const auto* src_room = dynamic_cast<const NavigableSpatialDefinition*>(
      src_tree.GetElement(kRoom)->GetSpatialDefinition());
  const auto* tgt_room = dynamic_cast<const NavigableSpatialDefinition*>(
      tgt_tree.GetElement(kRoom)->GetSpatialDefinition());
  RelationSolver solver;
  constexpr double kStep = 0.02;
  // Declared bounds: per 2 cm of travel the root yaw changes by at most the
  // source's own yaw change plus kGeometric (the facing relation re-aims
  // with the 0.3 m placement error: <= 0.3 m / |to look-at| per 2 cm step);
  // at or beyond the footprint (y >= 3) it equals the source facing within
  // kTrack (identity geometry: the conditioned-out bearings leave the
  // source's own referent-relative heading).
  const double kGeometric = 5.0 * M_PI / 180.0;
  double yaw_t = M_PI_2, yaw_s_prev = M_PI_2;
  bool first = true, contact_jumped = false;
  Eigen::Vector2d last_contact(0, 0);
  double worst_excess = 0.0;
  for (double y = 1.0; y <= 7.0 + 1e-9; y += kStep) {
    // Inside the footprint the source turns from 90 to 120 degrees.
    const double yaw_s =
        M_PI_2 + (M_PI / 6.0) * std::clamp((y - 3.5) / 1.0, 0.0, 1.0);
    const WalkPayloads w = EncodeWalk(src_tree, src_room, y, yaw_s);
    if (!first && (w.contact - last_contact).norm() > 1.0) contact_jumped = true;
    last_contact = w.contact;
    const SolverResult r = solver.Solve(
        Eigen::Vector2d(5.3, y), yaw_t, {w.facing}, {}, tgt_tree, tgt_room,
        nullptr, ReferentRelativeHeading({w.spacing}, {}, tgt_tree));
    // Facing away is not a suppression, and
    // there is no source-heading fallback. The yaw stays the look-at, including
    // past the table. The silhouette switch at the footprint is a discontinuity
    // of the contact (FacingGroundingRule), so the step bound covers the
    // approach, where the ray hits the near edge.
    EXPECT_TRUE(r.yaw_specified) << "y=" << y;
    if (!first && y < 3.0 && r.yaw_specified) {
      const double excess = std::abs(Wrap(r.yaw - yaw_t)) -
                            std::abs(Wrap(yaw_s - yaw_s_prev));
      worst_excess = std::max(worst_excess, excess);
    }
    if (std::getenv("R25_TRACE"))
      std::cerr << "y=" << y << " yaw=" << r.yaw * 180 / M_PI
                << " src=" << yaw_s * 180 / M_PI << "\n";
    first = false;
    yaw_t = r.yaw;
    yaw_s_prev = yaw_s;
  }
  EXPECT_TRUE(contact_jumped) << "fixture must exercise the contact jump";
  EXPECT_LE(worst_excess, kGeometric)
      << "worst yaw step beyond the source's own, on the approach: "
      << worst_excess * 180 / M_PI << " deg";
}

// ===========================================================================
// The radial chart is pure radial; seat contacts map by ratio and angle.
// ===========================================================================
RelationPayload SeatContactPayload(const ObjectSpec& seat,
                                   const Eigen::Vector2d& local) {
  RelationPayload p;
  p.referent_shared_id = kObjectShared;
  p.subject_anchor_name = "pelvis";
  p.kernel_type = RelationKernelType::kAvatarAffordancePose;
  auto chart = RadialSurfaceChart::Encode(
      local, *Rect(-seat.hx, -seat.hy, seat.hx, seat.hy), Eigen::Vector2d(0, 1));
  EXPECT_TRUE(chart.ok()) << chart.status();
  p.support.contact_normalized = *chart;
  p.support.has_contact = true;
  p.support.sub_region_name = seat.part;
  p.grounded_relation.encoded_state =
      Eigen::Vector3d(chart->angle, chart->radial_ratio, 0.0);
  p.grounded_relation.proximity = 1.0;
  return p;
}

// Pure radial boundary distance of an axis-aligned rectangle along theta.
double RectBoundary(double hx, double hy, double theta) {
  const double c = std::abs(std::cos(theta)), s = std::abs(std::sin(theta));
  return std::min(c > 1e-15 ? hx / c : 1e300, s > 1e-15 ? hy / s : 1e300);
}

// World target of a source-local seat point, through the production path.
Eigen::Vector3d MapSeatPoint(const RelationSolver& solver,
                             const ElementTree& tree, const ObjectSpec& source,
                             const Eigen::Vector2d& local) {
  auto grounded =
      solver.GroundAffordanceTarget(SeatContactPayload(source, local), tree);
  EXPECT_TRUE(grounded.ok()) << grounded.status();
  return grounded.ok() ? grounded->position : Eigen::Vector3d::Zero();
}

const ObjectSpec kChair{3.0, 5.0, -0.7, 0.25, 0.25, 0.45, "seat"};
const ObjectSpec kSofa{5.0, 4.0, 0.4, 0.9, 0.3, 0.45, "seat"};
const ObjectSpec kStool{6.0, 2.0, 1.1, 0.17, 0.17, 0.75, "seat"};
const ObjectSpec kBench{2.0, 2.0, 0.2, 0.75, 0.2, 0.45, "seat"};

// The cross-support map is the pure radial chart. A contact at ratio rho
// along theta on the source seat lands at ratio rho along theta on the
// target seat, with no core, ramp or origin-dependent gain.
TEST(R3_PureRadialChart, SeatContactMapsByRatioAndAngleExactly) {
  for (const ObjectSpec* src : {&kChair, &kSofa, &kStool, &kBench}) {
    for (const ObjectSpec* tgt : {&kChair, &kSofa, &kStool, &kBench}) {
      if (src == tgt) continue;
      ElementTree tree;
      AddRoom(&tree, 10.0, 8.0);
      AddObject(&tree, *tgt);
      RelationSolver solver;
      for (double rho : {0.001, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.9, 1.0}) {
        for (int it = 0; it < 24; ++it) {
          const double theta = 2.0 * M_PI * it / 24.0 + 0.0137;
          const double rs = rho * RectBoundary(src->hx, src->hy, theta);
          const double rt = rho * RectBoundary(tgt->hx, tgt->hy, theta);
          const Eigen::Vector3d p = MapSeatPoint(
              solver, tree, *src,
              Eigen::Vector2d(rs * std::cos(theta), rs * std::sin(theta)));
          const Eigen::Vector3d expected = ObjectPoint(
              *tgt, rt * std::cos(theta), rt * std::sin(theta), tgt->top);
          EXPECT_NEAR((p - expected).norm(), 0.0, 1e-9)
              << "rho " << rho << " theta " << theta;
        }
      }
    }
  }
}

// Mixed peers are detected, not silently disagreeing: a version-0 packet (a
// pre-chart_version peer) decodes with the pure radial chart, as does version 1, and
// the deleted regularized version 2 and newer are rejected.
TEST(R3_PureRadialChart, ChartVersionTravelsWithThePacket) {
  ElementTree tree;
  AddRoom(&tree, 10.0, 8.0);
  AddObject(&tree, kSofa);
  RelationSolver solver;
  auto p = SeatContactPayload(kChair, {0.05, 0.0});
  EXPECT_EQ(p.support.contact_normalized.chart_version,
            RadialSurfaceChart::kChartVersion);
  p.support.contact_normalized.chart_version = 0;
  const double legacy_ratio = 0.05 / 0.25;  // pure radial r / B
  p.support.contact_normalized.radial_ratio = legacy_ratio;
  p.support.contact_normalized.angle = -M_PI_2;  // +X relative to +Y forward
  auto legacy = solver.GroundAffordanceTarget(p, tree);
  ASSERT_TRUE(legacy.ok());
  EXPECT_NEAR((legacy->position - ObjectPoint(kSofa, legacy_ratio * 0.9, 0,
                                              kSofa.top))
                  .norm(),
              0.0, 1e-9);
  EXPECT_FALSE(legacy->chart_version_known);
  p.support.contact_normalized.chart_version = 2;  // the deleted chart
  EXPECT_FALSE(solver.GroundAffordanceTarget(p, tree).ok());
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
