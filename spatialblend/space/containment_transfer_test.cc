// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/containment_transfer_test.cc
//
// Containment transfer (part of the scope layer): a flagged Containment LEAF is the actor's whole scope while the
// source root is inside it. The leaf has no referents below it, so the empty-scope
// mapping maps the root through the leaf's chart onto
// the receiver's counterpart: that mapping IS the containment transfer. One path for
// identical copies (identity mapping) and for different physical objects.
//
// ABI 4: there is no contained pose, blend weight or
// HANDOVER state beside the root proposal. The tests below read the mapped pose off
// the receiver's ROOT PROPOSAL; the contained / not-contained fact survives as the
// per-update transfer flag of that proposal (ABI 4.3:
// set exactly when the source root is inside a transfer leaf AND this space has its
// counterpart, recomputed on every update) and the CONTAIN_ENTER / CONTAIN_EXIT cause
// bits of the jump update. There is no contained event.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <array>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/geometry/radial_surface_chart.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/realization/containment_transfer.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace {

constexpr shared_element_id_t kRoomShared = 100;
constexpr shared_element_id_t kAvatarShared = 110;
constexpr shared_element_id_t kRegionShared = 300;
constexpr shared_element_id_t kOtherShared = 400;
constexpr element_id_t kRoom = 1;
constexpr element_id_t kRegion = 2;
constexpr element_id_t kAvatar = 3;
constexpr element_id_t kOther = 4;
constexpr element_id_t kObject = 5;
constexpr element_id_t kAfford = 6;
constexpr element_id_t kRegion2 = 7;
constexpr element_id_t kScopeX = 8;
constexpr shared_element_id_t kObjectShared = 500;
constexpr shared_element_id_t kAffordShared = 600;
constexpr shared_element_id_t kRegion2Shared = 700;
constexpr shared_element_id_t kScopeXShared = 800;

constexpr double kDeg = M_PI / 180.0;

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

struct RegionSpec {
  double x = 10.0, y = 10.0, yaw = 0.0;
  double hx = 1.0, hy = 0.8;
  Eigen::Matrix4d Transform() const { return Pose(x, y, yaw); }
};

void AddRoom(Space* s) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kRoom).ok());
  tree->GetElement(kRoom)->SetSharedId(kRoomShared);
  tree->GetElement(kRoom)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
              {0, 0}, {20, 0}, {20, 20}, {0, 20}})));
}

// Element-local Containment-role region child of the room.
void AddRegion(Space* s, const RegionSpec& r, bool flag) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kRegion, kRoom).ok());
  Element* e = tree->GetElement(kRegion);
  e->SetSharedId(kRegionShared);
  e->SetTag("region");
  e->SetLocalTransform(r.Transform());
  auto def = std::make_shared<NavigableSpatialDefinition>(
      Box(r.hx, r.hy), nullptr, 0.0, GeometryRole::kContainment);
  def->SetElementLocal(true);
  e->SetSpatialDefinition(def);
  if (flag) {
    ASSERT_TRUE(s->SetElementContainmentTransfer(kRegion, true));
  }
}

void AddOther(Space* s, const Eigen::Vector2d& c) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kOther, kRoom).ok());
  Element* e = tree->GetElement(kOther);
  e->SetSharedId(kOtherShared);
  e->SetTag("other");
  e->SetLocalTransform(Pose(c.x(), c.y(), 0.0));
  e->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(
      std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
          {c.x() - 1, c.y() - 1}, {c.x() + 1, c.y() - 1},
          {c.x() + 1, c.y() + 1}, {c.x() - 1, c.y() + 1}})));
}

spatialblend::entity_id_t AddAvatar(Space* s, element_id_t parent = kRoom) {
  auto* tree = s->GetElementTree();
  EXPECT_TRUE(tree->AddElement(kAvatar, parent).ok());
  tree->GetElement(kAvatar)->SetSharedId(kAvatarShared);
  std::vector<SubjectAnchor> anchors = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
      SubjectAnchor("chest", Eigen::Vector3d(0, 0, 1.2))};
  auto a = s->CreateActor(anchors);
  s->GetActor(a)->SetElementId(kAvatar);
  return a;
}

struct Rig {
  std::unique_ptr<LocalContext> ctx;
  Space* src = nullptr;
  Space* tgt = nullptr;
  spatialblend::entity_id_t src_actor = 0, tgt_actor = 0;
  RegionSpec src_region, tgt_region;
  // The receiver avatar's zero-lag driver (declared after ctx: destroyed first).
  std::unique_ptr<IdealDriver> follower;
};

struct RigOptions {
  bool object_tree = false;     // Region -> Object -> Affordance (T3.7 shape)
  bool obstacle_object = false; // Region -> Obstacle box (the authored Chess/Whiteboard shape)
  bool second_region = false;   // an adjacent flagged region on both sides
  bool scoped_x = false;        // source-only element X, root Spacing scoped to it
  double update_interval = 0.0; // > 0: set on both spaces
  bool src_flag = true;
  bool tgt_flag = true;
  bool tgt_has_region = true;
  bool with_other = false;
  bool bind_facing = false;
  // Where the SOURCE avatar's own element hangs in its tree (a person leaf: the scope never depends on it).
  element_id_t src_avatar_parent = kRoom;
  Eigen::Vector2d src_other{16, 16};
  Eigen::Vector2d tgt_other{4, 16};
};

void AddRegion2(Space* s, const RegionSpec& r);
void AddObjectTree(Space* s);
void AddObstacleObject(Space* s);
void AddScopeX(Space* s);

std::unique_ptr<Rig> MakeRig(const RegionSpec& s, const RegionSpec& t,
                             const RigOptions& o = {}) {
  auto rig = std::make_unique<Rig>();
  rig->src_region = s;
  rig->tgt_region = t;
  rig->ctx = LocalContext::Create();
  rig->src = rig->ctx->AddSpace();
  rig->tgt = rig->ctx->AddSpace();
  AddRoom(rig->src);
  AddRegion(rig->src, s, o.src_flag);
  AddRoom(rig->tgt);
  if (o.tgt_has_region) AddRegion(rig->tgt, t, o.tgt_flag);
  if (o.with_other) {
    AddOther(rig->src, o.src_other);
    AddOther(rig->tgt, o.tgt_other);
  }
  if (o.second_region) {
    AddRegion2(rig->src, {12, 10, 0.0, 1.0, 0.8});
    AddRegion2(rig->tgt, {12, 10, 0.0, 1.0, 0.8});
  }
  if (o.object_tree) {
    AddObjectTree(rig->src);
    AddObjectTree(rig->tgt);
  }
  if (o.obstacle_object) {
    AddObstacleObject(rig->src);
    AddObstacleObject(rig->tgt);
  }
  if (o.scoped_x) AddScopeX(rig->src);
  rig->src_actor = AddAvatar(rig->src, o.src_avatar_parent);
  rig->tgt_actor = AddAvatar(rig->tgt);
  if (o.scoped_x) {
    rig->src->BindScoped(rig->src_actor, "root",
                         RelationKernelType::kAvatarSpacing, kScopeX);
  } else {
    rig->src->Bind(rig->src_actor, "root", RelationKernelType::kAvatarSpacing);
  }
  if (o.update_interval > 0.0) {
    rig->src->SetUpdateInterval(o.update_interval);
    rig->tgt->SetUpdateInterval(o.update_interval);
  }
  if (o.bind_facing)
    rig->src->Bind(rig->src_actor, "root", RelationKernelType::kAvatarFacing);
  rig->src->SetHub(rig->ctx->GetHub());
  rig->tgt->SetHub(rig->ctx->GetHub());
  rig->src->SetMovementGate(0.0, 0.0);
  rig->tgt->SetMovementGate(0.0, 0.0);
  rig->follower = std::make_unique<IdealDriver>(rig->tgt, rig->tgt_actor);
  return rig;
}

// Region 2 sits next to the first (touching at its east edge).
void AddRegion2(Space* s, const RegionSpec& r) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kRegion2, kRoom).ok());
  Element* e = tree->GetElement(kRegion2);
  e->SetSharedId(kRegion2Shared);
  e->SetLocalTransform(r.Transform());
  auto def = std::make_shared<NavigableSpatialDefinition>(
      Box(r.hx, r.hy), nullptr, 0.0, GeometryRole::kContainment);
  def->SetElementLocal(true);
  e->SetSpatialDefinition(def);
  ASSERT_TRUE(s->SetElementContainmentTransfer(kRegion2, true));
}

// Region -> Object -> Affordance, the shape the Unity authoring compiles.
void AddObjectTree(Space* s) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kObject, kRegion).ok());
  Element* o = tree->GetElement(kObject);
  o->SetSharedId(kObjectShared);
  o->SetLocalTransform(Pose(0.0, 0.0, 0.0));
  auto od = std::make_shared<NavigableSpatialDefinition>(
      Box(0.4, 0.3), nullptr, 0.0, GeometryRole::kContainment);
  od->SetElementLocal(true);
  o->SetSpatialDefinition(od);
  ASSERT_TRUE(tree->AddElement(kAfford, kObject).ok());
  Element* a = tree->GetElement(kAfford);
  a->SetSharedId(kAffordShared);
  a->SetLocalTransform(Pose(0.0, 0.0, 0.0));
  a->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-0.2, -0.15, 0.7), Eigen::Vector3d(0.2, 0.15, 0.8),
      GeometryRole::kSurface));
}

// The authored shape of Chess and the Whiteboard: the region carries its object, an
// obstacle box (the compiler's ColliderEnvelope) at the region's centre.
void AddObstacleObject(Space* s) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kObject, kRegion).ok());
  Element* o = tree->GetElement(kObject);
  o->SetSharedId(kObjectShared);
  o->SetLocalTransform(Pose(0.0, 0.0, 0.0));
  o->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-0.2, -0.2, 0.0), Eigen::Vector3d(0.2, 0.2, 0.8),
      GeometryRole::kObstacle));
}

// Source-only container AROUND the region (the region is its child), a scope the receiver cannot resolve.
// It is not a sibling of the region overlapping it, which a scene may not declare (sibling Containment /
// Navigable elements do not overlap): the descent would have two overlapping holders and
// the entered one would be the one v1's rule picks, here the larger. Nested, the region is inside X and the
// leaf is reached through it. X's polygon is world-absolute, so its own placement is the identity and the
// region keeps its world pose when it moves under it.
void AddScopeX(Space* s) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kScopeX, kRoom).ok());
  Element* e = tree->GetElement(kScopeX);
  e->SetSharedId(kScopeXShared);
  e->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(
      std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
          {8.5, 8.5}, {11.5, 8.5}, {11.5, 11.5}, {8.5, 11.5}})));
  ASSERT_TRUE(s->ReparentElement(kRegion, kScopeX));
}

void Drive(Rig* rig, double x, double y, double yaw) {
  rig->src->UpdateActorState(rig->src_actor, Eigen::Vector2d(x, y), yaw);
}

// Repeats a pose (the receiver follows each proposal at once).
void Settle(Rig* rig, double x, double y, double yaw, int frames = 90) {
  for (int i = 0; i < frames; ++i) Drive(rig, x, y, yaw);
}

ReceiverProposal ProposalOf(Space* s, spatialblend::entity_id_t actor) {
  ReceiverProposal p;
  EXPECT_TRUE(s->ReadProposal(actor, &p));
  return p;
}

std::optional<RealizationTarget> RootTarget(Space* s,
                                            spatialblend::entity_id_t actor) {
  const RealizationResult* r = s->GetRealizationTargets(actor);
  if (!r) return std::nullopt;
  for (const auto& t : r->targets)
    if (t.anchor_name == "root") return t;
  return std::nullopt;
}

// The receiver's root proposal is the transfer mapping (the ABI 4.3 flag). It is the only
// contained fact a space exposes: the source's own side is read from the wire.
bool Transfer(const Rig* rig) {
  ReceiverProposal p;
  return rig->tgt->ReadProposal(rig->tgt_actor, &p) && p.transfer();
}

// World pose of the object-local pose (lx, ly, lyaw) under `region`.
void LocalToWorld(const RegionSpec& r, double lx, double ly, double lyaw,
                  Eigen::Vector2d* pos, double* yaw) {
  const Eigen::Vector4d w = r.Transform() * Eigen::Vector4d(lx, ly, 0, 1);
  *pos = Eigen::Vector2d(w.x(), w.y());
  *yaw = lyaw + r.yaw;
}

double WrapAbs(double a) { return std::fabs(std::remainder(a, 2 * M_PI)); }

// ---------------------------------------------------------------------------
// Identical copies, any receiver placement and yaw: object-relative pose
// passes through the (identity) mapping.
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, IdenticalCopiesPreserveObjectRelativePose) {
  const RegionSpec src{10, 10, 30 * kDeg, 1.0, 0.8};
  const RegionSpec placements[] = {{12, 12, 0.0, 1.0, 0.8},
                                   {6, 14, 90 * kDeg, 1.0, 0.8},
                                   {14, 6, 200 * kDeg, 1.0, 0.8},
                                   {3, 3, -135 * kDeg, 1.0, 0.8}};
  const double local[][3] = {{0.0, 0.0, 0.0},    {0.5, 0.3, 1.0},
                             {-0.6, -0.4, -2.5}, {0.3, -0.5, 3.0},
                             {0.0, 0.45, 0.4}};
  double worst_pos = 0.0, worst_yaw = 0.0;
  for (const RegionSpec& tgt : placements) {
    auto rig = MakeRig(src, tgt);
    // Approach from outside, then visit the object-local poses.
    Eigen::Vector2d p;
    double y;
    LocalToWorld(src, -2.5, 0.0, 0.0, &p, &y);
    Drive(rig.get(), p.x(), p.y(), y);
    for (const auto& l : local) {
      LocalToWorld(src, l[0], l[1], l[2], &p, &y);
      Settle(rig.get(), p.x(), p.y(), y);
      ASSERT_TRUE(Transfer(rig.get()));
      Eigen::Vector2d expect;
      double expect_yaw;
      LocalToWorld(tgt, l[0], l[1], l[2], &expect, &expect_yaw);
      // The root the receiver PROPOSES is the mapped pose (the empty-scope mapping
      // of the leaf): the object-relative position passes through the identity.
      // The yaw is the leaf-relative yaw passed through the correspondence:
      // with no Facing authored it is never unspecified.
      const ReceiverProposal prop = ProposalOf(rig->tgt, rig->tgt_actor);
      worst_pos = std::max(worst_pos, (Eigen::Vector2d(prop.x, prop.y) - expect).norm());
      EXPECT_LT((Eigen::Vector2d(prop.x, prop.y) - expect).norm(), 0.01);
      EXPECT_TRUE(prop.status == ProposalStatus::kExact ||
                  prop.status == ProposalStatus::kInexact);
      EXPECT_TRUE(prop.yaw_specified());
      worst_yaw = std::max(worst_yaw, WrapAbs(prop.yaw - expect_yaw));
      EXPECT_LT(WrapAbs(prop.yaw - expect_yaw), 1e-6)
          << "the object-relative yaw passes through the identity mapping";
    }
  }
  RecordProperty("worst_position_error_um",
                 static_cast<int>(std::round(worst_pos * 1e6)));
  RecordProperty("worst_yaw_error_urad",
                 static_cast<int>(std::round(worst_yaw * 1e6)));
}

// ---------------------------------------------------------------------------
// Different physical objects (a table): the same path, non-trivial mapping.
// The radial-chart-normalized position is preserved: the element's chart (angle and
// radial ratio) of the mapped root equals the source's.
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, DifferentTablesPreserveRadialChartPosition) {
  const RegionSpec src{10, 10, 20 * kDeg, 1.0, 0.6};
  const RegionSpec tgt{6, 13, 70 * kDeg, 0.8, 0.5};
  auto rig = MakeRig(src, tgt);
  const auto src_poly = Box(src.hx, src.hy);
  const auto tgt_poly = Box(tgt.hx, tgt.hy);
  const double local[][2] = {{0.5, 0.2}, {-0.6, 0.3}, {0.2, -0.35}, {0.7, 0.1}};
  Eigen::Vector2d p;
  double y;
  LocalToWorld(src, -2.5, 0.0, 0.0, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  bool any_nontrivial = false;
  for (const auto& l : local) {
    LocalToWorld(src, l[0], l[1], 0.0, &p, &y);
    Settle(rig.get(), p.x(), p.y(), y);
    const ReceiverProposal prop = ProposalOf(rig->tgt, rig->tgt_actor);
    const Eigen::Vector4d tl =
        tgt.Transform().inverse() * Eigen::Vector4d(prop.x, prop.y, 0, 1);
    const Eigen::Vector2d tgt_local(tl.x(), tl.y());
    auto ns = RadialSurfaceChart::Encode(Eigen::Vector2d(l[0], l[1]), *src_poly,
                                         Eigen::Vector2d(0, 1));
    auto nt = RadialSurfaceChart::Encode(tgt_local, *tgt_poly,
                                         Eigen::Vector2d(0, 1));
    ASSERT_TRUE(ns.ok());
    ASSERT_TRUE(nt.ok());
    EXPECT_LT(WrapAbs(ns->angle - nt->angle), 1e-6);
    EXPECT_NEAR(ns->radial_ratio, nt->radial_ratio, 1e-6);
    any_nontrivial |= (tgt_local - Eigen::Vector2d(l[0], l[1])).norm() > 0.01;
  }
  EXPECT_TRUE(any_nontrivial) << "the mapping should not be the identity";
}

// ---------------------------------------------------------------------------
// Walking across the region edge, in both directions.
//
// The receiver root proposal near the edge: outside, the border referent of the
// container and the objects; inside the transfer leaf, the leaf is the whole scope
// and the empty-scope mapping maps the root. With the flag off the same walk
// is the plain scene-level solve. The flagged run must never be less smooth than the
// unflagged one it replaces, and inside the leaf the proposal moves exactly with the
// source (step <= v*dt + 1 cm).
// ---------------------------------------------------------------------------
struct WalkResult {
  double worst_all = 0.0;
  double worst_inside = 0.0;
  int with_target = 0, inside_frames = 0, inside_with_target = 0, frames = 0;
};

WalkResult WalkAcrossEdge(const RegionSpec& src, const RegionSpec& tgt,
                          bool flagged, bool non_leaf = false) {
  RigOptions opt;
  opt.src_flag = flagged;
  opt.tgt_flag = flagged;
  opt.update_interval = 1.0 / 60.0;
  opt.object_tree = non_leaf;
  auto rig = MakeRig(src, tgt, opt);
  constexpr double v = 1.4, dt = 1.0 / 60.0;
  const double step = v * dt;
  std::vector<double> xs;
  for (double x = 5.0; x <= 15.0; x += step) xs.push_back(x);
  std::vector<double> back(xs.rbegin(), xs.rend());
  xs.insert(xs.end(), back.begin(), back.end());
  WalkResult r;
  r.frames = static_cast<int>(xs.size());
  std::optional<Eigen::Vector2d> prev;
  bool prev_inside = false;
  for (double x : xs) {
    Drive(rig.get(), x, 10.0, 0.3);
    const auto target = RootTarget(rig->tgt, rig->tgt_actor);
    const bool inside = Transfer(rig.get());
    if (inside) ++r.inside_frames;
    if (target) {
      ++r.with_target;
      if (inside) ++r.inside_with_target;
      if (prev) {
        const double d = (target->position - *prev).norm();
        r.worst_all = std::max(r.worst_all, d);
        if (flagged && inside && prev_inside)
          r.worst_inside = std::max(r.worst_inside, d);
      }
      prev = target->position;
    } else {
      prev.reset();
    }
    prev_inside = inside;
  }
  return r;
}

void ExpectWalkIsContinuous(const RegionSpec& src, const RegionSpec& tgt,
                            bool non_leaf = false) {
  const double step = 1.4 / 60.0;
  const WalkResult base = WalkAcrossEdge(src, tgt, false, non_leaf);
  const WalkResult flagged = WalkAcrossEdge(src, tgt, true, non_leaf);
  ::testing::Test::RecordProperty(
      "worst_step_mm", static_cast<int>(std::round(flagged.worst_all * 1000)));
  ::testing::Test::RecordProperty(
      "baseline_worst_step_mm",
      static_cast<int>(std::round(base.worst_all * 1000)));
  ::testing::Test::RecordProperty(
      "worst_inside_step_mm",
      static_cast<int>(std::round(flagged.worst_inside * 1000)));
  // Inside the leaf the proposal is the source's pose in the element frame mapped
  // through the leaf: it moves with the source (a scale of 1 for identical copies).
  EXPECT_LE(flagged.worst_inside, std::max(step + 0.01, base.worst_all + 0.005));
  EXPECT_LE(flagged.worst_all, std::max(step + 0.01, base.worst_all + 0.005))
      << "baseline worst step " << base.worst_all;
  EXPECT_EQ(flagged.inside_with_target, flagged.inside_frames);
  EXPECT_GT(flagged.with_target, flagged.frames / 2);
}

TEST(ContainmentTransfer, WalkingAcrossTheEdgeIdenticalCopiesHasNoJump) {
  ExpectWalkIsContinuous({10, 10, 0.0, 1.0, 0.8}, {12, 6.5, 90 * kDeg, 1.0, 0.8});
}

TEST(ContainmentTransfer, WalkingAcrossTheEdgeDifferentTablesHasNoJump) {
  ExpectWalkIsContinuous({10, 10, 0.0, 1.0, 0.8}, {12, 6.5, 90 * kDeg, 0.9, 0.7});
}

// A non-leaf region (Region -> Object -> Affordance), where scope opacity
// matters: the walk is as continuous as the unflagged scene-level run.
TEST(ContainmentTransfer, WalkingAcrossANonLeafRegionHasNoJump) {
  ExpectWalkIsContinuous({10, 10, 0.0, 1.0, 0.8}, {12, 6.5, 90 * kDeg, 1.0, 0.8},
                         /*non_leaf=*/true);
}

// RETIRED: ContainmentTransfer.
// RootYawNeverTurnsFasterThanTheSlewLimit and ContainmentYawSlew.* tested the
// rate-integrated root yaw (pi rad/s slew, lock-on, committed turn direction)
// that lived in core. Core proposes the contained yaw (exact at alpha 1,
// the shortest-arc blend inside the band) and the driver turns and locks at
// its own rate (ContainedRootPolicy in Unity). The relation-level yaw is pinned
// by the identity / chart tests of this file.

// The actor element is a person leaf wherever it hangs, and the descent starts at the tree's navigable root.
// A source whose element hangs UNDER the transfer leaf must not start its descent inside the leaf, which it
// never tests: the transfer flag would never be set for a root inside it.
TEST(ContainmentTransfer, TheTransferFlagDoesNotDependOnWhereTheSourcesElementHangs) {
  // Scope catalog case SP-TREE-04 (spatialblend/space/testing/SCOPE_CASES.md).
  const RegionSpec src{10, 10, 0.0, 1.0, 0.8};
  RigOptions options;
  options.src_avatar_parent = kRegion;  // the source's person leaf is a child of the leaf region
  auto rig = MakeRig(src, {12, 6.5, 90 * kDeg, 1.0, 0.8}, options);
  Settle(rig.get(), 4.0, 4.0, 0.0, 30);
  EXPECT_FALSE(Transfer(rig.get())) << "outside the leaf, whatever the leaf the person hangs from";
  Settle(rig.get(), 10.0, 10.0, 0.0, 30);
  EXPECT_TRUE(Transfer(rig.get())) << "inside the leaf: the mapping is the transfer";
  Settle(rig.get(), 4.0, 4.0, 0.0, 30);
  EXPECT_FALSE(Transfer(rig.get())) << "and out again";
}

// ---------------------------------------------------------------------------
// The transfer flag switches instantly at the edge: set on the very update the source
// crosses into the leaf, clear on the very update it crosses out. Recomputed every
// update, so entering again sets it again.
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, TransferFlagSwitchesInstantlyAtTheEdge) {
  const RegionSpec src{10, 10, 0.0, 1.0, 0.8};  // west edge at x = 9
  auto rig = MakeRig(src, {12, 6.5, 90 * kDeg, 1.0, 0.8});
  std::vector<double> xs;
  for (int i = 0; i <= 19; ++i) xs.push_back(8.905 + 0.01 * i);   // 8.905..9.095
  for (int i = 19; i >= 0; --i) xs.push_back(8.905 + 0.01 * i);   // and back out
  int sets = 0, clears = 0;
  bool was = false;
  for (double x : xs) {
    Drive(rig.get(), x, 10.0, 0.0);
    const bool inside = x > 9.0;
    EXPECT_EQ(Transfer(rig.get()), inside) << x;
    if (Transfer(rig.get()) && !was) ++sets;
    if (!Transfer(rig.get()) && was) ++clears;
    was = Transfer(rig.get());
  }
  EXPECT_EQ(sets, 1);
  EXPECT_EQ(clears, 1);
  // The edge frame carries the cause bit on the receiver's root proposal.
  Drive(rig.get(), 8.95, 10.0, 0.0);
  EXPECT_FALSE(ProposalOf(rig->tgt, rig->tgt_actor).cause & kCauseContainEnter);
  Drive(rig.get(), 9.05, 10.0, 0.0);
  EXPECT_TRUE(ProposalOf(rig->tgt, rig->tgt_actor).cause & kCauseContainEnter);
  EXPECT_TRUE(Transfer(rig.get()));
  Drive(rig.get(), 10.0, 10.0, 0.0);
  EXPECT_FALSE(ProposalOf(rig->tgt, rig->tgt_actor).cause & kCauseContainEnter)
      << "an enter is reported once, on the frame it happens";
  EXPECT_TRUE(Transfer(rig.get())) << "the flag stays set while inside";
  Drive(rig.get(), 5.0, 10.0, 0.0);
  EXPECT_TRUE(ProposalOf(rig->tgt, rig->tgt_actor).cause & kCauseContainExit);
  EXPECT_FALSE(Transfer(rig.get()));
}

// ---------------------------------------------------------------------------
// Relations to outside elements stop driving the root while it is inside the
// leaf: the leaf is the whole scope. Facing stays global in the core and does
// not enter the position.
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, OutsideRelationsDoNotDriveRootWhileFullyInside) {
  const RegionSpec src{10, 10, 30 * kDeg, 1.0, 0.8};
  const RegionSpec tgt{12, 6, 90 * kDeg, 1.0, 0.8};
  RigOptions o;
  o.with_other = true;
  o.bind_facing = true;
  auto rig = MakeRig(src, tgt, o);
  Eigen::Vector2d p;
  double y;
  LocalToWorld(src, -2.5, 0.0, 0.0, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  LocalToWorld(src, 0.1, 0.0, 0.7, &p, &y);
  Settle(rig.get(), p.x(), p.y(), y);
  const ReceiverProposal before = ProposalOf(rig->tgt, rig->tgt_actor);
  const auto before_root = RootTarget(rig->tgt, rig->tgt_actor);
  ASSERT_TRUE(before_root.has_value());
  // Move the receiver's other object somewhere very different.
  rig->tgt->SetElementTransform(kOther, Pose(-6, -4, 0.0));
  Settle(rig.get(), p.x(), p.y(), y);
  const ReceiverProposal after = ProposalOf(rig->tgt, rig->tgt_actor);
  Eigen::Vector2d expect;
  double expect_yaw;
  LocalToWorld(tgt, 0.1, 0.0, 0.7, &expect, &expect_yaw);
  // The mapped pose stays the leaf's chart and does not follow the other object.
  EXPECT_LT((Eigen::Vector2d(after.x, after.y) - expect).norm(), 0.01);
  EXPECT_LT((Eigen::Vector2d(after.x, after.y) -
             Eigen::Vector2d(before.x, before.y)).norm(), 1e-6);
}

// ---------------------------------------------------------------------------
// A receiver without the object: the event still fires (the fact is the
// source's), and no phantom mapped pose is invented.
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, ReceiverWithoutTheObjectFallsBackToSceneLevel) {
  const RegionSpec src{10, 10, 30 * kDeg, 1.0, 0.8};
  RigOptions o;
  o.tgt_has_region = false;
  auto rig = MakeRig(src, {}, o);
  rig->tgt->UpdateActorState(rig->tgt_actor, Eigen::Vector2d(2, 2), 0.0);
  const Eigen::Vector2d start = rig->tgt->GetActor(rig->tgt_actor)->GetPosition();
  Eigen::Vector2d p;
  double y;
  for (double lx : {-2.5, 0.0, 0.3}) {
    LocalToWorld(src, lx, 0.0, 0.5, &p, &y);
    Drive(rig.get(), p.x(), p.y(), y);
  }
  EXPECT_FALSE(Transfer(rig.get())) << "no counterpart on the receiver: the flag stays clear";
  // Scene-level result with no object to ground against: the actor holds.
  EXPECT_LT((rig->tgt->GetActor(rig->tgt_actor)->GetPosition() - start).norm(),
            1e-9);
}

// ---------------------------------------------------------------------------
// Opt-in: without the flag nothing is contained, nothing is on the wire.
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, UnflaggedElementLeavesEverythingUntouched) {
  RigOptions o;
  o.src_flag = false;
  o.tgt_flag = false;
  auto rig = MakeRig({10, 10, 0.0, 1.0, 0.8}, {12, 6, 90 * kDeg, 1.0, 0.8}, o);
  std::vector<RemoteRelationUpdate> sent;
  rig->ctx->GetHub()->SetRelationUpdateCallback(
      [&](const RemoteRelationUpdate& u) { sent.push_back(u); });
  Drive(rig.get(), 8.0, 10.0, 0.0);
  Drive(rig.get(), 10.0, 10.0, 0.0);
  rig->ctx->GetHub()->SetRelationUpdateCallback(nullptr);
  ASSERT_FALSE(sent.empty());
  for (const auto& u : sent) EXPECT_FALSE(u.containment.has_value());
  EXPECT_FALSE(Transfer(rig.get()));
}

TEST(ContainmentTransfer, FlaggedUpdateCarriesTheObservationOnlyWhileInside) {
  auto rig = MakeRig({10, 10, 30 * kDeg, 1.0, 0.8}, {12, 6, 90 * kDeg, 1.0, 0.8});
  std::vector<RemoteRelationUpdate> sent;
  rig->ctx->GetHub()->SetRelationUpdateCallback(
      [&](const RemoteRelationUpdate& u) { sent.push_back(u); });
  Drive(rig.get(), 6.0, 10.0, 0.0);
  const size_t outside = sent.size();
  Eigen::Vector2d p;
  double y;
  LocalToWorld({10, 10, 30 * kDeg, 1.0, 0.8}, 0.4, -0.2, 0.6, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  rig->ctx->GetHub()->SetRelationUpdateCallback(nullptr);
  ASSERT_GT(sent.size(), outside);
  for (size_t i = 0; i < outside; ++i) EXPECT_FALSE(sent[i].containment);
  // Only the identity of the transfer element is on the wire: no pose, chart
  // or blend weight.
  const auto& c = sent.back().containment;
  ASSERT_TRUE(c.has_value());
  EXPECT_EQ(c->scope_shared_id, kRegionShared);
  EXPECT_EQ(c->scope_source_element_id, kRegion);
}

// ---------------------------------------------------------------------------
// Element-local polygon: containment follows the element transform, so a
// rotated and translated object is contained at the right place, and again at
// the new place after it is moved.
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, ElementLocalPolygonFollowsTheElementTransform) {
  auto rig = MakeRig({10, 10, 40 * kDeg, 1.0, 0.8}, {12, 6, 0.0, 1.0, 0.8});
  auto contained_at = [&](const RegionSpec& r, double lx, double ly) {
    Eigen::Vector2d p;
    double y;
    LocalToWorld(r, lx, ly, 0.0, &p, &y);
    Drive(rig.get(), p.x(), p.y(), y);
    return Transfer(rig.get());
  };
  const RegionSpec at = {10, 10, 40 * kDeg, 1.0, 0.8};
  EXPECT_TRUE(contained_at(at, 0.95, 0.0));
  EXPECT_FALSE(contained_at(at, 1.05, 0.0));
  EXPECT_TRUE(contained_at(at, 0.0, -0.75));
  EXPECT_FALSE(contained_at(at, 0.0, 0.85));
  // The world origin-centred reading of the same polygon would say the opposite
  // for the object's own centre.
  EXPECT_TRUE(contained_at(at, 0.0, 0.0));
  // Move and rotate the element: containment moves with it.
  const RegionSpec moved = {4.5, 15, -110 * kDeg, 1.0, 0.8};
  ASSERT_TRUE(rig->src->SetElementTransform(kRegion, moved.Transform()));
  EXPECT_TRUE(contained_at(moved, 0.95, 0.0));
  EXPECT_FALSE(contained_at(moved, 1.05, 0.0));
  EXPECT_FALSE(contained_at(at, 0.0, 0.0));  // the old place is empty now
}

TEST(ContainmentTransfer, FlagIsRejectedOnAnUnknownElementAndClearable) {
  auto rig = MakeRig({10, 10, 0.0, 1.0, 0.8}, {12, 6, 0.0, 1.0, 0.8});
  EXPECT_FALSE(rig->src->SetElementContainmentTransfer(999, true));
  EXPECT_TRUE(rig->src->IsElementContainmentTransfer(kRegion));
  EXPECT_TRUE(rig->src->SetElementContainmentTransfer(kRegion, false));
  EXPECT_FALSE(rig->src->IsElementContainmentTransfer(kRegion));
  Drive(rig.get(), 10.0, 10.0, 0.0);
  EXPECT_FALSE(Transfer(rig.get()));
}

// ---------------------------------------------------------------------------
// Large receiver rotations keep the identity exactly (the mapped position; the yaw
// is not part of the mapping).
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, IdentityHoldsForLargeRotationsAndYawNearPi) {
  const RegionSpec src{10, 10, -170 * kDeg, 1.0, 0.8};
  for (double tyaw : {181.0, 270.0, 359.0, -179.9, 540.0, -400.0}) {
    const RegionSpec tgt{6, 13, tyaw * kDeg, 1.0, 0.8};
    auto rig = MakeRig(src, tgt);
    Eigen::Vector2d p;
    double y;
    LocalToWorld(src, -2.5, 0.0, 0.0, &p, &y);
    Drive(rig.get(), p.x(), p.y(), y);
    for (const auto& l : std::vector<std::array<double, 3>>{
             {0.2, 0.1, M_PI - 1e-6}, {-0.3, 0.2, -M_PI + 1e-6},
             {0.75, -0.55, 3.0}, {0.0, 0.0, -3.1}}) {
      LocalToWorld(src, l[0], l[1], l[2], &p, &y);
      Settle(rig.get(), p.x(), p.y(), y);
      ASSERT_TRUE(Transfer(rig.get()));
      const ReceiverProposal prop = ProposalOf(rig->tgt, rig->tgt_actor);
      Eigen::Vector2d e;
      double ey;
      LocalToWorld(tgt, l[0], l[1], l[2], &e, &ey);
      EXPECT_LT((Eigen::Vector2d(prop.x, prop.y) - e).norm(), 1e-6) << tyaw;
    }
  }
}

// ---------------------------------------------------------------------------
// Observation reset while contained: no contained root survives it, so the flag is
// clear on the receiver at once (nothing is latched).
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, ObservationResetWhileContainedClearsTheFlag) {
  auto rig = MakeRig({10, 10, 0.0, 1.0, 0.8}, {12, 6.5, 90 * kDeg, 1.0, 0.8});
  Drive(rig.get(), 5.0, 10.0, 0.0);
  Drive(rig.get(), 10.0, 10.0, 0.0);
  ASSERT_TRUE(Transfer(rig.get()));
  uint64_t epoch = 0;
  ASSERT_TRUE(rig->src->BeginActorObservationEpoch(rig->src_actor, "root",
                                                   &epoch));
  EXPECT_FALSE(Transfer(rig.get()));
}

// InjectRelationPayloads carries no containment field: it is scene-level only
// and must not end a contained state.
TEST(ContainmentTransfer, InjectedPayloadsDoNotEndAContainedState) {
  auto rig = MakeRig({10, 10, 0.0, 1.0, 0.8}, {12, 6.5, 90 * kDeg, 1.0, 0.8});
  Settle(rig.get(), 10.0, 10.0, 0.0, 5);
  ASSERT_TRUE(Transfer(rig.get()));
  ASSERT_TRUE(rig->tgt->InjectRelationPayloads(rig->tgt_actor, {}, 100.0));
  EXPECT_TRUE(rig->tgt->IsElementContainmentTransfer(kRegion));
}

// ---------------------------------------------------------------------------
// Scope opacity (non-leaf region): the root's scene-level relation set must be
// the same inside the flagged region as just outside it, while a non-root
// anchor still descends into the object normally.
// ---------------------------------------------------------------------------
using ReferentSet = std::set<shared_element_id_t>;

struct Emitted {
  ReferentSet root_spacing, chest_spacing;
};

Emitted RunOpacity(bool flagged, double lx, double ly) {
  RigOptions o;
  o.src_flag = flagged;
  o.tgt_flag = flagged;
  o.object_tree = true;
  const RegionSpec src{10, 10, 30 * kDeg, 1.0, 0.8};
  auto rig = MakeRig(src, {6, 13, 120 * kDeg, 1.0, 0.8}, o);
  rig->src->Bind(rig->src_actor, "chest", RelationKernelType::kAvatarSpacing);
  Emitted out;
  rig->ctx->GetHub()->SetRelationUpdateCallback(
      [&](const RemoteRelationUpdate& u) {
        out.root_spacing.clear();
        out.chest_spacing.clear();
        for (const auto& p : u.payloads) {
          if (p.kernel_type != RelationKernelType::kAvatarSpacing) continue;
          (p.subject_anchor_name == "chest" ? out.chest_spacing
                                            : out.root_spacing)
              .insert(p.referent_shared_id);
        }
      });
  Eigen::Vector2d p;
  double y;
  if (lx > -100) {
    LocalToWorld(src, -2.5, 0.0, 0.0, &p, &y);
    Drive(rig.get(), p.x(), p.y(), y);
    LocalToWorld(src, lx * 0.5, ly * 0.5, 0.2, &p, &y);
    Drive(rig.get(), p.x(), p.y(), y);
    LocalToWorld(src, lx, ly, 0.2, &p, &y);
    Drive(rig.get(), p.x(), p.y(), y);
  }
  rig->ctx->GetHub()->SetRelationUpdateCallback(nullptr);
  return out;
}

// A flagged region WITH objects is not a scope container descended into like any
// other. An authored region carries its object and is a transfer LEAF, so
// every anchor's scope inside it is the leaf alone; unflagged, the scene is unchanged.
TEST(ContainmentTransfer, AFlaggedRegionWithObjectsIsALeafAndAnUnflaggedOneIsDescendedInto) {
  const Emitted outside = RunOpacity(true, -2.5, 0.0);
  ASSERT_EQ(outside.root_spacing, ReferentSet{kRegionShared});
  const Emitted unflagged = RunOpacity(false, 0.1, 0.05);
  const Emitted flagged = RunOpacity(true, 0.1, 0.05);
  EXPECT_NE(unflagged.root_spacing, ReferentSet{kRegionShared});
  EXPECT_FALSE(unflagged.chest_spacing.empty());
  EXPECT_EQ(flagged.root_spacing, ReferentSet{kRegionShared})
      << "inside the leaf the leaf is the whole scope";
  EXPECT_EQ(flagged.chest_spacing, ReferentSet{kRegionShared})
      << "no anchor descends into the object the leaf carries";
}

// ---------------------------------------------------------------------------
// Hopping between two adjacent flagged regions: the flag stays set (the source is in a
// leaf on both updates) and the hop update carries ENTER and EXIT together.
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, HoppingBetweenRegionsKeepsTheFlagAndCarriesBothCauseBits) {
  RigOptions o;
  o.second_region = true;
  auto rig = MakeRig({10, 10, 0.0, 1.0, 0.8}, {10, 10, 0.0, 1.0, 0.8}, o);
  Drive(rig.get(), 6.0, 10.0, 0.0);
  EXPECT_FALSE(Transfer(rig.get()));
  Drive(rig.get(), 10.5, 10.0, 0.0);   // region 1: x in [9, 11]
  EXPECT_TRUE(Transfer(rig.get()));
  EXPECT_TRUE(ProposalOf(rig->tgt, rig->tgt_actor).cause & kCauseContainEnter);
  Drive(rig.get(), 11.5, 10.0, 0.0);   // region 2: x in [11, 13]
  EXPECT_TRUE(Transfer(rig.get()));
  const uint32_t hop = ProposalOf(rig->tgt, rig->tgt_actor).cause;
  EXPECT_TRUE(hop & kCauseContainEnter);
  EXPECT_TRUE(hop & kCauseContainExit);
  Drive(rig.get(), 15.0, 10.0, 0.0);
  EXPECT_FALSE(Transfer(rig.get()));
  EXPECT_TRUE(ProposalOf(rig->tgt, rig->tgt_actor).cause & kCauseContainExit);
}

// ---------------------------------------------------------------------------
// A region whose polygon is world-absolute (not element-local) is charted in
// the element frame, so congruent copies at different placements still map by
// the identity. Treating the world polygon as local breaks this.
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, WorldAbsoluteRegionIsChartedInTheElementFrame) {
  const RegionSpec src{10, 10, 30 * kDeg, 1.0, 0.8};
  const RegionSpec tgt{14, 6, -50 * kDeg, 1.0, 0.8};
  auto rig = std::make_unique<Rig>();
  rig->src_region = src;
  rig->tgt_region = tgt;
  rig->ctx = LocalContext::Create();
  rig->src = rig->ctx->AddSpace();
  rig->tgt = rig->ctx->AddSpace();
  for (auto [space, r] : {std::pair<Space*, RegionSpec>{rig->src, src},
                          std::pair<Space*, RegionSpec>{rig->tgt, tgt}}) {
    AddRoom(space);
    auto* tree = space->GetElementTree();
    ASSERT_TRUE(tree->AddElement(kRegion, kRoom).ok());
    Element* e = tree->GetElement(kRegion);
    e->SetSharedId(kRegionShared);
    e->SetLocalTransform(r.Transform());
    // Baked into WORLD coordinates (the historical Navigable reading).
    std::vector<Eigen::Vector2d> world;
    for (const Eigen::Vector2d& c :
         {Eigen::Vector2d(-r.hx, -r.hy), Eigen::Vector2d(r.hx, -r.hy),
          Eigen::Vector2d(r.hx, r.hy), Eigen::Vector2d(-r.hx, r.hy)}) {
      const Eigen::Vector4d w = r.Transform() * Eigen::Vector4d(c.x(), c.y(), 0, 1);
      world.emplace_back(w.x(), w.y());
    }
    e->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(
        std::make_shared<Polygon2D>(world), nullptr, 0.0,
        GeometryRole::kContainment));
    ASSERT_TRUE(space->SetElementContainmentTransfer(kRegion, true));
  }
  rig->src_actor = AddAvatar(rig->src);
  rig->tgt_actor = AddAvatar(rig->tgt);
  rig->follower = std::make_unique<IdealDriver>(rig->tgt, rig->tgt_actor);
  rig->src->Bind(rig->src_actor, "root", RelationKernelType::kAvatarSpacing);
  rig->src->SetHub(rig->ctx->GetHub());
  rig->tgt->SetHub(rig->ctx->GetHub());
  rig->src->SetMovementGate(0.0, 0.0);
  rig->tgt->SetMovementGate(0.0, 0.0);
  Eigen::Vector2d p;
  double y;
  LocalToWorld(src, -2.5, 0.0, 0.0, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  for (const auto& l : std::vector<std::array<double, 3>>{
           {0.3, 0.2, 0.5}, {-0.6, -0.4, -2.0}}) {
    LocalToWorld(src, l[0], l[1], l[2], &p, &y);
    Settle(rig.get(), p.x(), p.y(), y);
    ASSERT_TRUE(Transfer(rig.get()));
    const ReceiverProposal prop = ProposalOf(rig->tgt, rig->tgt_actor);
    Eigen::Vector2d e;
    double ey;
    LocalToWorld(tgt, l[0], l[1], l[2], &e, &ey);
    EXPECT_LT((Eigen::Vector2d(prop.x, prop.y) - e).norm(), 0.01);
  }
}

// ---------------------------------------------------------------------------
// A transition alone is sent: the source barely moves (under the movement
// deadband), crosses the edge, and the receiver still learns of it.
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, StateOnlyChangeIsStillSent) {
  auto rig = MakeRig({10, 10, 0.0, 1.0, 0.8}, {12, 6.5, 90 * kDeg, 1.0, 0.8});
  rig->src->SetMovementGate(1.0, 1.0);
  Drive(rig.get(), 9.05, 10.0, 0.0);  // first send always goes out
  ASSERT_TRUE(Transfer(rig.get()));
  Drive(rig.get(), 8.95, 10.0, 0.0);  // 10 cm: under the deadband, out of region
  EXPECT_FALSE(Transfer(rig.get()));
  Drive(rig.get(), 9.05, 10.0, 0.0);  // and back in
  EXPECT_TRUE(Transfer(rig.get()));
}

// ---------------------------------------------------------------------------
// RETIRED: the 0.2 m blend band, the source alpha, the HANDOVER state
// and the contained pose reported beside the proposal (ContainmentTransfer.
// BlendBandInterpolatesBySourceAlpha, FailedSceneSolveStillGetsTheContainedPose and
// ContainmentHelpers.BlendWeightAndYawBlend). The leaf is the receiver's whole scope,
// so there is nothing to blend: the root proposal IS the mapped pose from the first
// frame inside. What replaces them is the three tests below.
// ---------------------------------------------------------------------------

// Different tables: entering is a mapped pose from the first frame inside, with the
// CONTAIN_ENTER cause, never a blend toward one.
TEST(ContainmentTransfer, TheMappedPoseIsTheProposalFromTheFirstFrameInside) {
  const RegionSpec src{10, 10, 0.0, 1.0, 0.8};
  const RegionSpec tgt{6, 13, 0.0, 0.5, 0.9};
  auto rig = MakeRig(src, tgt);
  Eigen::Vector2d p;
  double y;
  LocalToWorld(src, -2.5, 0.5, 0.0, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  // 5 cm inside the west edge: depth well under the retired 0.2 m band.
  LocalToWorld(src, -0.95, 0.5, 0.0, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  const ReceiverProposal prop = ProposalOf(rig->tgt, rig->tgt_actor);
  EXPECT_TRUE(prop.cause & kCauseContainEnter);
  // Different tables are not congruent: the mapping is the chart's, the flag is clear.
  EXPECT_FALSE(Transfer(rig.get()));
  const Eigen::Vector4d tl =
      tgt.Transform().inverse() * Eigen::Vector4d(prop.x, prop.y, 0, 1);
  // Inside the target element, at the source's normalized place (-0.95, 0.5/0.8).
  EXPECT_LT(std::fabs(tl.x()), tgt.hx + 1e-9);
  EXPECT_LT(std::fabs(tl.y()), tgt.hy + 1e-9);
  EXPECT_TRUE(prop.status == ProposalStatus::kExact ||
              prop.status == ProposalStatus::kInexact);
}

// The ABI carries no contained pose: the proposal set is the only place a pose is.
TEST(ContainmentTransfer, TheSetCarriesOneRootProposalAndNoContainedPose) {
  auto rig = MakeRig({10, 10, 30 * kDeg, 1.0, 0.8}, {6, 13, 120 * kDeg, 1.0, 0.8});
  Eigen::Vector2d p;
  double y;
  LocalToWorld({10, 10, 30 * kDeg, 1.0, 0.8}, 0.3, 0.1, 0.4, &p, &y);
  Settle(rig.get(), p.x(), p.y(), y);
  const ReceiverProposal prop = ProposalOf(rig->tgt, rig->tgt_actor);
  EXPECT_EQ(prop.anchor_id, 0u);
  EXPECT_TRUE(prop.anchors.empty())
      << "a contained root has no other relation-bearing anchor";
  EXPECT_EQ(prop.All().size(), 1u);
}

// A failed scene-level solve (the only relation is to X, which the receiver lacks):
// the root is still inside the leaf, and the leaf's empty-scope mapping is the
// proposal. The border of the container is a referent too, so a missing X is not an
// empty solve.
TEST(ContainmentTransfer, AMissingReferentStillLeavesTheLeafMapping) {
  const RegionSpec src{10, 10, 30 * kDeg, 1.0, 0.8};
  const RegionSpec tgt{6, 13, 120 * kDeg, 1.0, 0.8};
  RigOptions o;
  o.scoped_x = true;  // the only relation is to X, which the receiver lacks
  auto rig = MakeRig(src, tgt, o);
  Eigen::Vector2d p;
  double y;
  LocalToWorld(src, -2.5, 0.0, 0.0, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  LocalToWorld(src, 0.3, 0.1, 0.4, &p, &y);
  Settle(rig.get(), p.x(), p.y(), y);
  const ReceiverProposal prop = ProposalOf(rig->tgt, rig->tgt_actor);
  EXPECT_TRUE(RootTarget(rig->tgt, rig->tgt_actor).has_value());
  EXPECT_EQ(rig->tgt->GetRealizationTargets(rig->tgt_actor)->status,
            SolverStatus::kConverged);
  EXPECT_EQ(prop.status, ProposalStatus::kExact);
  Eigen::Vector2d e;
  double ey;
  LocalToWorld(tgt, 0.3, 0.1, 0.4, &e, &ey);
  EXPECT_LT((Eigen::Vector2d(prop.x, prop.y) - e).norm(), 0.01);
}

// ---------------------------------------------------------------------------
// An authored region that carries its object (Chess,
// Whiteboard: region -> object -> affordance) is a transfer LEAF. The object it
// carries is covered by the region's correspondence; it is not a referent below it, so
// the leaf stays the actor's whole scope and the empty-scope mapping is the transfer.
// The C++ counterpart of Unity's
// IdenticalChessRemoteAvatarObjectRelativeIdentityHoldsOnlyIfTheRegionIsATransferLeaf:
// the same object-relative pose in an identical copy at another placement, position
// and yaw, with the object present.
// ---------------------------------------------------------------------------
void ExpectIdentityWithObject(const RigOptions& options) {
  const RegionSpec src{10, 10, 30 * kDeg, 0.5, 0.9};
  const RegionSpec placements[] = {{14, 6, 200 * kDeg, 0.5, 0.9},
                                   {5, 14, 0.0, 0.5, 0.9},
                                   {6, 6, 90 * kDeg, 0.5, 0.9}};
  const double local[][3] = {{0.12, -0.35, 40 * kDeg}, {-0.2, 0.5, -130 * kDeg}};
  for (const RegionSpec& tgt : placements) {
    auto rig = MakeRig(src, tgt, options);
    for (const auto& l : local) {
      Eigen::Vector2d p;
      double y;
      LocalToWorld(src, 0.0, -3.0, 0.0, &p, &y);
      Drive(rig.get(), p.x(), p.y(), src.yaw);
      for (int step = 1; step <= 6; ++step) {
        Eigen::Vector2d a, b;
        double ya, yb;
        LocalToWorld(src, 0.0, -3.0, 0.0, &a, &ya);
        LocalToWorld(src, l[0], l[1], l[2], &b, &yb);
        const Eigen::Vector2d q = a + (b - a) * (step / 6.0);
        Drive(rig.get(), q.x(), q.y(), yb);
      }
      LocalToWorld(src, l[0], l[1], l[2], &p, &y);
      Settle(rig.get(), p.x(), p.y(), y, 3);
      ASSERT_TRUE(Transfer(rig.get()));
      Eigen::Vector2d expect;
      double expect_yaw;
      LocalToWorld(tgt, l[0], l[1], l[2], &expect, &expect_yaw);
      const ReceiverProposal prop = ProposalOf(rig->tgt, rig->tgt_actor);
      EXPECT_LT((Eigen::Vector2d(prop.x, prop.y) - expect).norm(), 0.01)
          << "receiver at (" << tgt.x << "," << tgt.y << ") yaw " << tgt.yaw / kDeg
          << ", avatar (" << l[0] << "," << l[1] << ")";
      EXPECT_TRUE(prop.yaw_specified());
      EXPECT_LT(WrapAbs(prop.yaw - expect_yaw), 2.0 * kDeg)
          << "receiver yaw " << tgt.yaw / kDeg << ", avatar yaw " << l[2] / kDeg
          << ": got " << prop.yaw / kDeg << ", expected " << expect_yaw / kDeg;
    }
  }
}

TEST(ContainmentTransfer, AnAuthoredRegionCarryingAnObstacleObjectIsATransferLeaf) {
  RigOptions o;
  o.obstacle_object = true;
  ExpectIdentityWithObject(o);
}

TEST(ContainmentTransfer, AnAuthoredRegionCarryingARegionObjectWithAffordancesIsATransferLeaf) {
  RigOptions o;
  o.object_tree = true;
  ExpectIdentityWithObject(o);
}

// Every anchor of the actor is inside the same leaf: the leaf is its whole scope, so no
// anchor descends into the object it carries.
TEST(ContainmentTransfer, TheCarriedObjectIsNotAReferentBelowTheLeaf) {
  RigOptions o;
  o.obstacle_object = true;
  const RegionSpec src{10, 10, 30 * kDeg, 0.5, 0.9};
  auto rig = MakeRig(src, {14, 6, 200 * kDeg, 0.5, 0.9}, o);
  rig->src->Bind(rig->src_actor, "chest", RelationKernelType::kAvatarSpacing);
  ReferentSet root, chest;
  rig->ctx->GetHub()->SetRelationUpdateCallback([&](const RemoteRelationUpdate& u) {
    root.clear();
    chest.clear();
    for (const auto& p : u.payloads) {
      if (p.kernel_type != RelationKernelType::kAvatarSpacing) continue;
      (p.subject_anchor_name == "chest" ? chest : root).insert(p.referent_shared_id);
    }
  });
  Eigen::Vector2d p;
  double y;
  LocalToWorld(src, 0.0, -3.0, 0.0, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  LocalToWorld(src, 0.12, -0.35, 0.0, &p, &y);
  Settle(rig.get(), p.x(), p.y(), y, 3);
  rig->ctx->GetHub()->SetRelationUpdateCallback(nullptr);
  EXPECT_EQ(root, ReferentSet{kRegionShared});
  EXPECT_EQ(chest, ReferentSet{kRegionShared});
}

// ---------------------------------------------------------------------------
// Yaw inside a transfer leaf. The Facing yaw when a Facing is
// authored; otherwise the leaf-relative yaw passed through the correspondence. Never
// YAW_UNSPECIFIED.
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, WithoutAFacingTheLeafRelativeYawPassesThroughTheCorrespondence) {
  // Different tables: the position goes through the radial chart, the yaw stays
  // relative to the leaf's frame (v1's inside-volume snap keeps the yaw relative to
  // the volume).
  const RegionSpec src{10, 10, 20 * kDeg, 1.0, 0.6};
  const RegionSpec tgt{6, 13, 70 * kDeg, 0.8, 0.5};
  auto rig = MakeRig(src, tgt);
  Eigen::Vector2d p;
  double y;
  LocalToWorld(src, -2.5, 0.0, 0.0, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  for (const double lyaw : {0.0, 0.7, -2.2, 3.0}) {
    LocalToWorld(src, 0.3, 0.1, lyaw, &p, &y);
    Settle(rig.get(), p.x(), p.y(), y, 3);
    const ReceiverProposal prop = ProposalOf(rig->tgt, rig->tgt_actor);
    EXPECT_TRUE(prop.yaw_specified()) << "leaf-relative yaw " << lyaw;
    EXPECT_LT(WrapAbs(prop.yaw - (tgt.yaw + lyaw)), 1e-6) << "leaf-relative yaw " << lyaw;
  }
}

TEST(ContainmentTransfer, AnAuthoredFacingOwnsTheYawInsideTheLeaf) {
  RigOptions o;
  o.bind_facing = true;
  o.with_other = true;
  const RegionSpec src{10, 10, 30 * kDeg, 1.0, 0.8};
  const RegionSpec tgt{6, 13, 120 * kDeg, 1.0, 0.8};
  auto rig = MakeRig(src, tgt, o);
  Eigen::Vector2d p;
  double y;
  LocalToWorld(src, -2.5, 0.0, 0.0, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  LocalToWorld(src, 0.3, 0.1, 0.4, &p, &y);
  Settle(rig.get(), p.x(), p.y(), y, 3);
  const ReceiverProposal prop = ProposalOf(rig->tgt, rig->tgt_actor);
  ASSERT_TRUE(Transfer(rig.get()));
  EXPECT_TRUE(prop.yaw_specified());
  // The Facing yaw is the bearing to the receiver's own `other` less the offset the
  // source holds toward its `other`; the leaf-relative pass-through would be tgt.yaw + 0.4.
  EXPECT_GT(WrapAbs(prop.yaw - (tgt.yaw + 0.4)), 1.0 * kDeg)
      << "the Facing, not the pass-through, set the yaw";
}

// ---------------------------------------------------------------------------
// Containment is evaluated statelessly on every update
// against the current entity set (v1). A counterpart that appears after the source went
// inside catches up on the next update: the transfer flag is clear until then and set
// on that update; nothing latched on the edge.
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, ACounterpartThatAppearsLateCatchesUpOnTheNextUpdate) {
  const RegionSpec src{10, 10, 30 * kDeg, 1.0, 0.8};
  const RegionSpec tgt{6, 13, 120 * kDeg, 1.0, 0.8};
  RigOptions o;
  o.tgt_has_region = false;
  auto rig = MakeRig(src, tgt, o);
  Eigen::Vector2d p;
  double y;
  LocalToWorld(src, -2.5, 0.0, 0.0, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  LocalToWorld(src, 0.3, 0.1, 0.4, &p, &y);
  Settle(rig.get(), p.x(), p.y(), y, 5);
  ASSERT_FALSE(Transfer(rig.get())) << "no counterpart yet";

  // The receiver's copy appears now, after the event. The very next source update uses it.
  AddRegion(rig->tgt, tgt, /*flag=*/true);
  Drive(rig.get(), p.x(), p.y(), y);
  EXPECT_TRUE(Transfer(rig.get()))
      << "the counterpart is resolved on this update, not latched";
  const ReceiverProposal prop = ProposalOf(rig->tgt, rig->tgt_actor);
  EXPECT_TRUE(prop.status == ProposalStatus::kExact ||
              prop.status == ProposalStatus::kInexact);
  Eigen::Vector2d expect;
  double expect_yaw;
  LocalToWorld(tgt, 0.3, 0.1, 0.4, &expect, &expect_yaw);
  EXPECT_LT((Eigen::Vector2d(prop.x, prop.y) - expect).norm(), 0.01);
  EXPECT_TRUE(prop.yaw_specified());
  EXPECT_LT(WrapAbs(prop.yaw - expect_yaw), 1e-6);
}

// Shape equivalence is part of the per-update flag. A counterpart that is
// declared a leaf but not congruent leaves the flag clear (the mapping is still the chart mapping); a geometry edit
// is picked up on the next update, both ways.
TEST(ContainmentTransfer, ANonCongruentCounterpartLeavesTheFlagClearAndAGeometryEditIsPickedUpOnTheNextUpdate) {
  // Scope catalog case SP-LEAF-02 (spatialblend/space/testing/SCOPE_CASES.md).
  const RegionSpec src{10, 10, 30 * kDeg, 1.0, 0.8};
  const RegionSpec tgt{12, 6, 90 * kDeg, 0.9, 0.7};  // a different table
  auto rig = MakeRig(src, tgt);
  Eigen::Vector2d p;
  double y;
  LocalToWorld(src, -2.5, 0.0, 0.0, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  LocalToWorld(src, 0.3, 0.1, 0.4, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  EXPECT_TRUE(ProposalOf(rig->tgt, rig->tgt_actor).cause & kCauseContainEnter)
      << "the source is inside its leaf: the cause bit is the source's fact";
  Drive(rig.get(), p.x(), p.y(), y);
  EXPECT_FALSE(Transfer(rig.get())) << "non-congruent shapes: the flag is clear";

  // The receiver edits its copy to the source's shape (a rigid edit of the polygon): next update.
  auto same = std::make_shared<NavigableSpatialDefinition>(Box(1.0, 0.8), nullptr, 0.0,
                                                           GeometryRole::kContainment);
  same->SetElementLocal(true);
  ASSERT_TRUE(rig->tgt->SetElementSpatialDefinition(kRegion, same));
  Drive(rig.get(), p.x(), p.y(), y);
  EXPECT_TRUE(Transfer(rig.get())) << "congruent after the edit";

  auto other = std::make_shared<NavigableSpatialDefinition>(Box(0.9, 0.7), nullptr, 0.0,
                                                            GeometryRole::kContainment);
  other->SetElementLocal(true);
  ASSERT_TRUE(rig->tgt->SetElementSpatialDefinition(kRegion, other));
  Drive(rig.get(), p.x(), p.y(), y);
  EXPECT_FALSE(Transfer(rig.get())) << "edited away again";
}

// The source-side fact: which leaf the source's own latest update found its root in; recomputed, not latched.
TEST(ContainmentTransfer, TheSourceFactIsWhichLeafItsLatestUpdateFoundItsRootIn) {
  auto rig = MakeRig({10, 10, 0.0, 1.0, 0.8}, {12, 6.5, 90 * kDeg, 1.0, 0.8});
  shared_element_id_t leaf = 7;
  ASSERT_TRUE(rig->src->CopySourceTransferLeaf(rig->src_actor, &leaf));
  EXPECT_EQ(leaf, kInvalidSharedElementId);
  Drive(rig.get(), 10.0, 10.0, 0.0);
  ASSERT_TRUE(rig->src->CopySourceTransferLeaf(rig->src_actor, &leaf));
  EXPECT_EQ(leaf, kRegionShared);
  Drive(rig.get(), 5.0, 10.0, 0.0);
  ASSERT_TRUE(rig->src->CopySourceTransferLeaf(rig->src_actor, &leaf));
  EXPECT_EQ(leaf, kInvalidSharedElementId);
  Drive(rig.get(), 10.0, 10.0, 0.0);
  ASSERT_TRUE(rig->src->SetElementContainmentTransfer(kRegion, false));
  Drive(rig.get(), 10.1, 10.0, 0.0);
  ASSERT_TRUE(rig->src->CopySourceTransferLeaf(rig->src_actor, &leaf));
  EXPECT_EQ(leaf, kInvalidSharedElementId);
  EXPECT_FALSE(rig->src->CopySourceTransferLeaf(987654, &leaf));
  // A receiver's proxy actor is not a source: it names no leaf of its own.
  ASSERT_TRUE(rig->tgt->CopySourceTransferLeaf(rig->tgt_actor, &leaf));
  EXPECT_EQ(leaf, kInvalidSharedElementId);
}

// The shape is the authored polygon, not its placement: a receiver placement whose matrix an app handed over as
// single-precision floats (no longer an exact rigid motion at 1e-9) must not break congruence.
TEST(ContainmentTransfer, ASinglePrecisionPlacementOfTheCounterpartIsStillTheSameShape) {
  const RegionSpec src{10, 10, 30 * kDeg, 1.0, 0.8};
  const RegionSpec tgt{12, 6, 200 * kDeg, 1.0, 0.8};
  auto rig = MakeRig(src, tgt);
  Eigen::Matrix4d rounded = tgt.Transform();
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) rounded(r, c) = static_cast<double>(static_cast<float>(rounded(r, c)));
  ASSERT_TRUE(rig->tgt->SetElementTransform(kRegion, rounded));
  Eigen::Vector2d p;
  double y;
  LocalToWorld(src, -2.5, 0.0, 0.0, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  LocalToWorld(src, 0.3, 0.1, 0.4, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  EXPECT_TRUE(Transfer(rig.get()));
}

// ---------------------------------------------------------------------------
// The Facing scope inside a transfer leaf (a scope rule of the Facing definition, role / flag based):
// the leaf and the object it carries are not look-at targets while the anchor is inside the leaf; its Surface
// descendants (the affordance surface) and every element outside it are.
// ---------------------------------------------------------------------------
TEST(ContainmentTransfer, InsideALeafTheFacingScopeKeepsItsSurfacesAndDropsTheLeafAndItsObject) {
  RigOptions o;
  o.object_tree = true;  // Region -> Object (Containment) -> Affordance (Surface)
  o.with_other = true;
  o.bind_facing = true;
  const RegionSpec src{10, 10, 30 * kDeg, 1.0, 0.8};
  const RegionSpec tgt{6, 13, 120 * kDeg, 1.0, 0.8};
  auto rig = MakeRig(src, tgt, o);
  rig->src->EnableObservabilityCapture(true);
  const auto facing_reasons = [&]() {
    std::map<int64_t, ScopeReason> out;
    for (const auto& obs : rig->src->GetScopeRecords(rig->src_actor))
      if (obs.kernel_type == RelationKernelType::kAvatarFacing)
        for (const auto& c : obs.record.candidates) out[c.shared_or_local_id] = c.reason;
    return out;
  };
  Eigen::Vector2d p;
  double y;
  LocalToWorld(src, -2.5, 0.0, 0.0, &p, &y);  // outside the region
  Drive(rig.get(), p.x(), p.y(), y);
  auto outside = facing_reasons();
  ASSERT_TRUE(outside.count(kRegionShared));
  EXPECT_NE(outside[kRegionShared], ScopeReason::kExcludedInsideTransferLeaf) << "outside, the region is a referent";
  EXPECT_NE(outside[kObjectShared], ScopeReason::kExcludedInsideTransferLeaf);
  LocalToWorld(src, 0.7, 0.5, 0.4, &p, &y);  // inside the region, off its object
  Drive(rig.get(), p.x(), p.y(), y);
  auto inside = facing_reasons();
  EXPECT_EQ(inside[kRegionShared], ScopeReason::kExcludedInsideTransferLeaf) << "the leaf it stands in";
  EXPECT_EQ(inside[kObjectShared], ScopeReason::kExcludedInsideTransferLeaf) << "the object the leaf carries";
  ASSERT_TRUE(inside.count(kAffordShared));
  EXPECT_NE(inside[kAffordShared], ScopeReason::kExcludedInsideTransferLeaf) << "its surface stays a look-at target";
  ASSERT_TRUE(inside.count(kOtherShared));
  EXPECT_NE(inside[kOtherShared], ScopeReason::kExcludedInsideTransferLeaf) << "outside elements are unaffected";
}

// ABI 4.14: the replay through the chart is its own per-anchor output, mapped from each anchor's OWN
// sample. The root and a non-root anchor (the chest, bound to Facing only, turned 0.9 rad against the root) inside the
// leaf each get the transform of their own source pose through the receiver's counterpart; outside, none.
TEST(ContainmentTransfer, EveryAnchorInsideALeafGetsTheChartTransformOfItsOwnSample) {
  RigOptions o;
  o.with_other = true;
  const RegionSpec src{10, 10, 30 * kDeg, 1.0, 0.8};
  const RegionSpec tgt{6, 13, 120 * kDeg, 1.0, 0.8};
  auto rig = MakeRig(src, tgt, o);
  rig->src->Bind(rig->src_actor, "chest", RelationKernelType::kAvatarFacing);
  Eigen::Vector2d p;
  double y;
  LocalToWorld(src, -2.5, 0.0, 0.0, &p, &y);
  Drive(rig.get(), p.x(), p.y(), y);
  {
    const ReceiverProposal prop = ProposalOf(rig->tgt, rig->tgt_actor);
    EXPECT_FALSE(prop.has_chart) << "outside the leaf: no chart transform";
  }
  const double lx = 0.3, ly = 0.1, lyaw = 0.4, chest_dyaw = 0.9;
  LocalToWorld(src, lx, ly, lyaw, &p, &y);
  rig->src->SetAnchorWorldPose(rig->src_actor, "chest", Eigen::Vector3d(p.x(), p.y(), 1.2), y + chest_dyaw);
  Settle(rig.get(), p.x(), p.y(), y, 3);
  const ReceiverProposal prop = ProposalOf(rig->tgt, rig->tgt_actor);
  Eigen::Vector2d expect;
  double expect_yaw;
  LocalToWorld(tgt, lx, ly, lyaw, &expect, &expect_yaw);
  ASSERT_TRUE(prop.has_chart) << "the root";
  EXPECT_EQ(prop.chart_leaf_shared_id, kRegionShared);
  EXPECT_TRUE(prop.chart_verified) << "the root's relation to the leaf carries its footprint";
  EXPECT_NEAR(prop.chart_x, expect.x(), 1e-9);
  EXPECT_NEAR(prop.chart_y, expect.y(), 1e-9);
  EXPECT_LT(WrapAbs(prop.chart_yaw - expect_yaw), 1e-9);
  const AnchorProposal* chest = nullptr;
  for (const auto& a : prop.anchors)
    if (a.anchor_name == "chest") chest = &a;
  ASSERT_NE(chest, nullptr) << "the chest bears a Facing relation";
  ASSERT_TRUE(chest->has_chart) << "a non-root anchor inside the leaf";
  EXPECT_EQ(chest->chart_leaf_shared_id, kRegionShared);
  EXPECT_NEAR(chest->chart_x, expect.x(), 1e-9);
  EXPECT_NEAR(chest->chart_y, expect.y(), 1e-9);
  EXPECT_NEAR(chest->chart_z, 1.2, 1e-9) << "its own height in the leaf, mapped";
  EXPECT_LT(WrapAbs(chest->chart_yaw - (expect_yaw + chest_dyaw)), 1e-9) << "its own yaw, not the root's";
}

// The chart transfer (the in-object replay) is the object's own, computed from its DECLARED shape. A
// child region sticking out of the leaf extends the leaf's scope (element/scope_extension.h) but never its chart: an
// extended leaf (on the source, the receiver, or both) still transfers, verified, with the same chart transform as
// unextended; a pose in the extension only (outside the object) is no replay through the chart.
TEST(ContainmentTransfer, AnExtendedLeafStillTransfersWithTheSameChartTransform) {
  const RegionSpec src{10, 10, 30 * kDeg, 1.0, 0.8};
  const RegionSpec tgt{6, 13, 120 * kDeg, 1.0, 0.8};
  constexpr element_id_t kStandOff = 9;
  // An unshared stand-off band under the leaf, 0.2 m past its +x edge (nearby: it extends the leaf).
  const auto extend = [&](Space* s) {
    ASSERT_TRUE(s->AddElement(kStandOff, kRegion));
    ASSERT_TRUE(s->SetElementTransform(kStandOff, Pose(0.9, 0.0, 0.0)));
    auto band = std::make_shared<NavigableSpatialDefinition>(Box(0.3, 0.3), nullptr, 0.0, GeometryRole::kContainment);
    band->SetElementLocal(true);
    ASSERT_TRUE(s->SetElementSpatialDefinition(kStandOff, band)) << s->LastSceneError().message;
    ASSERT_TRUE(s->WaitForScopeExtension());
    ASSERT_TRUE(s->GetElementTree()->GetElement(kRegion)->HasScopeExtension());
  };
  const double lx = 0.3, ly = 0.1, lyaw = 0.4;
  Eigen::Vector2d expect;
  double expect_yaw;
  LocalToWorld(tgt, lx, ly, lyaw, &expect, &expect_yaw);
  // Unextended: the reference transform.
  ReceiverProposal reference;
  {
    auto rig = MakeRig(src, tgt);
    Eigen::Vector2d p;
    double y;
    LocalToWorld(src, lx, ly, lyaw, &p, &y);
    Settle(rig.get(), p.x(), p.y(), y, 3);
    reference = ProposalOf(rig->tgt, rig->tgt_actor);
    ASSERT_TRUE(reference.has_chart);
    ASSERT_TRUE(reference.chart_verified);
    ASSERT_TRUE(Transfer(rig.get()));
  }
  for (const int sides : {1, 2, 3}) {  // 1 source, 2 receiver, 3 both
    SCOPED_TRACE(sides);
    auto rig = MakeRig(src, tgt);
    if (sides & 1) extend(rig->src);
    if (sides & 2) extend(rig->tgt);
    Eigen::Vector2d p;
    double y;
    LocalToWorld(src, lx, ly, lyaw, &p, &y);
    Settle(rig.get(), p.x(), p.y(), y, 3);
    const ReceiverProposal prop = ProposalOf(rig->tgt, rig->tgt_actor);
    EXPECT_TRUE(Transfer(rig.get())) << "the extension never clears the transfer flag";
    ASSERT_TRUE(prop.has_chart);
    EXPECT_EQ(prop.chart_leaf_shared_id, kRegionShared);
    EXPECT_TRUE(prop.chart_verified) << "declared shapes are compared, congruent";
    EXPECT_EQ(prop.chart_x, reference.chart_x);
    EXPECT_EQ(prop.chart_y, reference.chart_y);
    EXPECT_EQ(prop.chart_yaw, reference.chart_yaw);
    EXPECT_NEAR(prop.chart_x, expect.x(), 1e-9);
    EXPECT_NEAR(prop.chart_y, expect.y(), 1e-9);
    EXPECT_LT(WrapAbs(prop.chart_yaw - expect_yaw), 1e-9);
    if (sides & 1) {
      // In the extension, outside the object: the leaf's scope, not the object's chart.
      LocalToWorld(src, 1.1, 0.0, 0.0, &p, &y);
      Settle(rig.get(), p.x(), p.y(), y, 3);
      EXPECT_FALSE(ProposalOf(rig->tgt, rig->tgt_actor).has_chart);
      EXPECT_FALSE(Transfer(rig.get()));
    }
  }
}

// The extension never maps anything: furniture extended on ONE peer only
// (the source, or the receiver) leaves the receiver's mapped root and anchors exactly as with neither extended. The
// furniture is the element-local region (unflagged here: ordinary furniture), whose boundary chart maps the contact.
TEST(ContainmentTransfer, FurnitureExtendedOnOnePeerOnlyMapsExactlyAsUnextended) {
  const RegionSpec src{10, 10, 30 * kDeg, 1.0, 0.8};
  const RegionSpec tgt{6, 13, 120 * kDeg, 1.0, 0.8};
  RigOptions o;
  o.src_flag = o.tgt_flag = false;
  constexpr element_id_t kBand = 9;
  // A stand-off band 0.2 m past the furniture's local -x edge (the side the actor faces), on one peer: the nearest
  // point of the EXTENDED scope is on the band, the nearest point of the furniture is its own edge.
  const auto extend = [](Space* s) {
    ASSERT_TRUE(s->AddElement(kBand, kRegion));
    ASSERT_TRUE(s->SetElementTransform(kBand, Pose(-1.5, 0.0, 0.0)));
    auto band = std::make_shared<NavigableSpatialDefinition>(Box(0.3, 0.3), nullptr, 0.0, GeometryRole::kContainment);
    band->SetElementLocal(true);
    ASSERT_TRUE(s->SetElementSpatialDefinition(kBand, band)) << s->LastSceneError().message;
    ASSERT_TRUE(s->WaitForScopeExtension());
    ASSERT_TRUE(s->GetElementTree()->GetElement(kRegion)->HasScopeExtension());
  };
  // The source root 0.5 m from the band (1.3 m from the furniture's edge); the chest turned a little.
  const auto run = [&](int sides) {
    auto rig = MakeRig(src, tgt, o);
    rig->src->Bind(rig->src_actor, "chest", RelationKernelType::kAvatarFacing);
    if (sides & 1) extend(rig->src);
    if (sides & 2) extend(rig->tgt);
    Eigen::Vector2d at;
    double yaw;
    LocalToWorld(src, -2.3, 0.2, 0.1, &at, &yaw);
    rig->src->SetAnchorWorldPose(rig->src_actor, "chest", Eigen::Vector3d(at.x(), at.y(), 1.2), yaw + 0.3);
    Settle(rig.get(), at.x(), at.y(), yaw, 5);
    return ProposalOf(rig->tgt, rig->tgt_actor);
  };
  const ReceiverProposal reference = run(0);
  // The reference maps through the furniture's chart: the source's local pose, on the receiver's furniture.
  Eigen::Vector2d expect;
  double expect_yaw;
  LocalToWorld(tgt, -2.3, 0.2, 0.1, &expect, &expect_yaw);
  EXPECT_NEAR(reference.x, expect.x(), 1e-6);
  EXPECT_NEAR(reference.y, expect.y(), 1e-6);
  for (const int sides : {1, 2}) {
    SCOPED_TRACE(sides);
    const ReceiverProposal p = run(sides);
    EXPECT_NEAR(p.x, reference.x, 1e-9);
    EXPECT_NEAR(p.y, reference.y, 1e-9);
    EXPECT_NEAR(p.yaw, reference.yaw, 1e-9);
    ASSERT_EQ(p.anchors.size(), reference.anchors.size());
    for (size_t k = 0; k < p.anchors.size(); ++k) {
      SCOPED_TRACE(p.anchors[k].anchor_name);
      EXPECT_EQ(p.anchors[k].status, reference.anchors[k].status);
      EXPECT_NEAR(p.anchors[k].x, reference.anchors[k].x, 1e-9);
      EXPECT_NEAR(p.anchors[k].y, reference.anchors[k].y, 1e-9);
      EXPECT_NEAR(p.anchors[k].yaw, reference.anchors[k].yaw, 1e-9);
    }
  }
}


// ---------------------------------------------------------------------------
// Two transfer leaves side by side in the source room (a Whiteboard region next to a Chess region) and far apart
// in the receiver's (4.7 m and more). The source steps from one into the other. The receiver's body is driven at the driver's own
// bound (2 m/s), not zero-lag. It must WALK to the other leaf's counterpart: no teleport request and no placement while it is on its
// way, as long as the source stays in the new leaf for less than the interaction's latency limit after entering it.
// ---------------------------------------------------------------------------
struct PacedReceiver {
  Space* space;
  spatialblend::entity_id_t actor;
  Eigen::Vector2d pos;
  double yaw = 0.0;
  double max_step;
  int teleports = 0, placements = 0;
  double largest_step = 0.0;
  void Publish(bool teleported = false) {
    EmbodiedPoseInput in;
    in.x = pos.x(); in.y = pos.y(); in.yaw = yaw; in.epoch = 1;
    in.flags = teleported ? kEmbodiedTeleported : 0u;
    space->PublishEmbodiedPose(actor, in, 0.0, NextPortSequence(space, actor));
  }
  // One driver frame: poll the events (a teleport request places the body, as the product driver does), then step toward the
  // latest proposal at the bound.
  void Step() {
    Space::ActorEvent e;
    while (space->PollActorEvent(actor, &e)) {
      if (e.kind == Space::ActorEvent::Kind::kTeleportRequest && e.anchor_id == 0) {
        ++teleports;
        const Eigen::Vector2d to(e.x, e.y);
        largest_step = std::max(largest_step, (to - pos).norm());
        pos = to;
        Publish(true);
      }
    }
    ReceiverProposal p;
    if (!space->ReadProposal(actor, &p) || p.status == ProposalStatus::kNone) return;
    if (p.placement) ++placements;
    const Eigen::Vector2d goal(p.x, p.y), d = goal - pos;
    const double step = std::min(d.norm(), max_step);
    if (step > 0.0) pos += d.normalized() * step;
    largest_step = std::max(largest_step, step);
    Publish();
  }
};

TEST(ContainmentTransfer, SteppingIntoANeighbouringLeafFarAwayOnTheReceiverIsAWalkNotATeleport) {
  RigOptions o;
  o.update_interval = 1.0 / 30.0;
  auto rig = MakeRig({10, 10, 0.0, 1.0, 0.8}, {10, 10, 0.0, 1.0, 0.8}, o);
  AddRegion2(rig->src, {12, 10, 0.0, 1.0, 0.8});   // touching region 1's east edge
  AddRegion2(rig->tgt, {14, 15, 0.0, 1.0, 0.8});   // ~6.4 m from region 1's counterpart
  rig->follower.reset();
  PacedReceiver rcv{rig->tgt, rig->tgt_actor, Eigen::Vector2d(10, 10), 0.0, 2.0 / 30.0};
  rcv.Publish(true);
  auto frame = [&](double x, double y) {
    Drive(rig.get(), x, y, 0.0);
    rcv.Step();
  };
  // Dwell in region 1 (8 s, a small sway so every tick is an update), long past the latency limit.
  for (int k = 0; k < 240; ++k) frame(10.0 + 0.02 * std::sin(k * 0.3), 10.0);
  ASSERT_TRUE(Transfer(rig.get()));
  const int teleports_before = rcv.teleports;
  rcv.largest_step = 0.0;
  rcv.placements = 0;
  // Step east across the shared edge into region 2, and stay there 3 s (under the 5 s limit).
  for (int k = 0; k <= 20; ++k) frame(10.2 + 0.06 * k, 10.0);
  for (int k = 0; k < 90; ++k) frame(11.4 + 0.02 * std::sin(k * 0.3), 10.0);
  EXPECT_EQ(rcv.teleports - teleports_before, 0) << "the receiver is asked to teleport on a leaf switch";
  EXPECT_EQ(rcv.placements, 0) << "a leaf switch is not a placement";
  EXPECT_LE(rcv.largest_step, 2.0 / 30.0 + 1e-9) << "every step is a paced step";
  // ... and back into region 1 for 3 s.
  for (int k = 0; k <= 20; ++k) frame(11.4 - 0.06 * k, 10.0);
  for (int k = 0; k < 90; ++k) frame(10.2 + 0.02 * std::sin(k * 0.3), 10.0);
  EXPECT_EQ(rcv.teleports - teleports_before, 0) << "and back";
  EXPECT_LE(rcv.largest_step, 2.0 / 30.0 + 1e-9);
}


// Two peers each a source AND a receiver. A space's diagnostic clock advances by its own ticks and by the source-clock
// deltas of the updates it receives; if the source clock a space stamped on its updates were that same space clock, each
// peer's stamp would carry the other's increments back and the two clocks would grow geometrically (a source clock of
// 22706 s after 96 s, 441912 s after ~3 min). The receiver's latency clock reads that stamp, so the 5 s interaction
// limit would run out within a few frames of any new engagement the body had not reached: a teleport at every leaf
// switch. The stamp is the actor's own source time. A receiver that never arrives is asked after 5 s of source time
// (150 ticks at 30 Hz), never before, while the peer sends its own updates the whole time.
TEST(ContainmentTransfer, TheLatencyClockIsTheSourcesOwnTimeWhenBothPeersSend) {
  RigOptions o;
  o.update_interval = 1.0 / 30.0;
  auto rig = MakeRig({10, 10, 0.0, 1.0, 0.8}, {14, 15, 0.0, 1.0, 0.8}, o);
  rig->follower.reset();
  // The receiver's own user, a source in its space and a received avatar in the source's (the second person of a session).
  constexpr element_id_t kPeer = 40;
  constexpr shared_element_id_t kPeerShared = 4000;
  spatialblend::entity_id_t peer_actor[2] = {0, 0};
  Space* spaces[2] = {rig->src, rig->tgt};
  for (int i = 0; i < 2; ++i) {
    auto* tree = spaces[i]->GetElementTree();
    ASSERT_TRUE(tree->AddElement(kPeer, kRoom).ok());
    tree->GetElement(kPeer)->SetSharedId(kPeerShared);
    peer_actor[i] = spaces[i]->CreateActor({SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))});
    spaces[i]->GetActor(peer_actor[i])->SetElementId(kPeer);
  }
  rig->tgt->Bind(peer_actor[1], "root", RelationKernelType::kAvatarSpacing);
  PacedReceiver rcv{rig->tgt, rig->tgt_actor, Eigen::Vector2d(3, 3), 0.0, 0.0};   // never moves: never arrives
  rcv.Publish(true);
  int first_teleport = -1;
  for (int k = 0; k < 200 && first_teleport < 0; ++k) {
    Drive(rig.get(), 10.0 + 0.02 * std::sin(k * 0.3), 10.0, 0.0);   // the source, inside the leaf
    rig->tgt->UpdateActorState(peer_actor[1], Eigen::Vector2d(5.0 + 0.02 * std::sin(k * 0.3), 5.0), 0.0);  // the peer, sending
    const int before = rcv.teleports;
    rcv.Step();
    if (rcv.teleports > before) first_teleport = k;
  }
  ASSERT_GE(first_teleport, 0) << "a receiver that never arrives is still asked";
  EXPECT_GE(first_teleport, 148) << "not before 5 s of the source's own time";
  EXPECT_LE(first_teleport, 155);
}

}  // namespace
}  // namespace spatialblend
