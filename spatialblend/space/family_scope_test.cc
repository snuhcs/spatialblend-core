// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/family_scope_test.cc
//
// Layer tests for the relation-family scope definitions D1-D9. The geometric halves have
// their own homes and are cited, not repeated:
//   D1 one ray            relation/scope_gate_test            ScopeGating_D1_OneRayToTheLookAtContact
//   D2 in-body            space/solver_layer_test             Yaw.ALookAtPointInsideTheBodyHasNoBearing
//   D3 parent region      space/scope_gating_contract_test    ScopeGating_D3_*
//   D4 border sub-rules   relation/scope_layer_test           BorderReferent.TheRule
//   D5 transfer leaf      c/containment_transfer_c_test, space/containment_transfer_test
//   D8 geodesic path      relation/scope_gate_test            ScopeGating_D8_*, E1_2_*
// This file adds D6 (tags), D7 (no HOLD, at the Space level) and D9 (people), and the
// deleted rules that are Space behaviour.
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <set>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace test {
namespace {

std::shared_ptr<Polygon2D> Rect(double x0, double y0, double x1, double y1) {
  return std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});
}

Eigen::Matrix4d At(double x, double y) {
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = x;
  t(1, 3) = y;
  return t;
}

void AddBox(ElementTree* tree, element_id_t id, element_id_t parent, double cx,
            double cy, const std::string& tag = "") {
  ASSERT_TRUE(tree->AddElement(id, parent).ok());
  tree->GetElement(id)->SetSharedId(id);
  tree->GetElement(id)->SetLocalTransform(At(cx, cy));
  tree->GetElement(id)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-0.3, -0.3, 0.0), Eigen::Vector3d(0.3, 0.3, 0.8),
          GeometryRole::kObstacle));
  if (!tag.empty()) tree->GetElement(id)->SetTag(tag);
}

std::set<element_id_t> Matched(Space* space, spatialblend::entity_id_t actor,
                               size_t binding) {
  std::set<element_id_t> out;
  for (const auto& m : space->GetBindings(actor)[binding].latest_matches)
    out.insert(m.element_id);
  return out;
}

// Room(1) [0,20]^2 with three tagged boxes. The actor element is 20.
struct Scene {
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space* space = ctx->AddSpace();
  spatialblend::entity_id_t actor = 0;
  Scene() {
    auto* tree = space->GetElementTree();
    EXPECT_TRUE(tree->AddElement(1).ok());
    tree->GetElement(1)->SetSharedId(1);
    tree->GetElement(1)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 20, 20)));
    AddBox(tree, 2, 1, 5, 15, "Chair");
    AddBox(tree, 3, 1, 12, 15, "Table");
    AddBox(tree, 4, 1, 15, 5, "Chair");
    EXPECT_TRUE(tree->AddElement(20, 1).ok());
    tree->GetElement(20)->SetSharedId(20);
    actor = space->CreateActor({SubjectAnchor("root", {0, 0, 0}),
                                SubjectAnchor("chest", {0, 0, 1.2})});
    FollowProposals(space, actor);
    space->GetActor(actor)->SetElementId(20);
  }
};

// D6: a tag is the AffordancePose binding. It never narrows Spacing or Facing scope.
TEST(FamilyScope, D6_TagsNarrowNothingButTheAffordancePoseBinding) {
  Scene s;
  s.space->Bind(s.actor, "root", RelationKernelType::kAvatarSpacing, "Chair");
  s.space->Bind(s.actor, "chest", RelationKernelType::kAvatarFacing, "Chair");
  s.space->UpdateActorState(s.actor, Eigen::Vector2d(10, 10), 0.0);
  EXPECT_EQ(Matched(s.space, s.actor, 0), (std::set<element_id_t>{2, 3, 4}))
      << "Spacing: every object of the container, the Table too";
  EXPECT_EQ(Matched(s.space, s.actor, 1), (std::set<element_id_t>{2, 3, 4}))
      << "Facing: every visible referent, the Table too";
  // The bindings carry no tag: nothing downstream (source evidence, observability) can
  // read a filter that does not exist.
  EXPECT_TRUE(s.space->GetBindings(s.actor)[0].tag_filter.empty());
  EXPECT_TRUE(s.space->GetBindings(s.actor)[1].tag_filter.empty());
}

// ... and AffordancePose keeps it as the binding declaration (which affordance kind).
TEST(FamilyScope, D6_TheAffordancePoseBindingKeepsItsTag) {
  Scene s;
  s.space->Bind(s.actor, "root", RelationKernelType::kAvatarAffordancePose, "Chair");
  EXPECT_EQ(s.space->GetBindings(s.actor)[0].tag_filter, "Chair");
  s.space->UpdateActorState(s.actor, Eigen::Vector2d(10, 10), 0.0);
  const auto matched = Matched(s.space, s.actor, 0);
  EXPECT_TRUE(matched.count(2) && matched.count(4));
  EXPECT_EQ(matched.count(3), 0u) << "the Table is not a Chair";
}

// D9: every person in the same containment tree is a Spacing referent, at any depth.
TEST(FamilyScope, D9_EveryPersonInTheSameTreeIsASpacingReferent) {
  // Scope catalog case SP-PEOPLE-01 (spatialblend/space/testing/SCOPE_CASES.md).
  Scene s;
  auto* tree = s.space->GetElementTree();
  // A person two levels down, inside a region of the room: not a direct child of the
  // container the actor stands in.
  ASSERT_TRUE(tree->AddElement(30, 1).ok());
  tree->GetElement(30)->SetSharedId(30);
  tree->GetElement(30)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(Rect(14, 14, 19, 19)));
  ASSERT_TRUE(tree->AddElement(31, 30).ok());
  tree->GetElement(31)->SetSharedId(31);
  tree->GetElement(31)->SetLocalTransform(At(16, 16));
  const auto other = s.space->CreateActor({SubjectAnchor("root", {0, 0, 0})});
  FollowProposals(s.space, other);
  s.space->GetActor(other)->SetElementId(31);

  s.space->Bind(s.actor, "root", RelationKernelType::kAvatarSpacing);
  s.space->UpdateActorState(s.actor, Eigen::Vector2d(10, 10), 0.0);
  const auto matched = Matched(s.space, s.actor, 0);
  EXPECT_EQ(matched.count(31), 1u) << "the person at depth 2";
  EXPECT_EQ(matched.count(20), 0u) << "the subject itself is never a referent";
}

// D7 at the Space level: the source scope is authoritative; a referent with no counterpart
// drops, and with none left the receiver proposes NOTHING (there is no HOLD status).
TEST(FamilyScope, D7_NoCounterpartDropsAndTheProposalIsNoneNotHold) {
  // Scope catalog case SP-D7-02 (spatialblend/space/testing/SCOPE_CASES.md).
  auto ctx = LocalContext::Create();
  Space* src = ctx->AddSpace();
  Space* rcv = ctx->AddSpace();
  for (Space* sp : {src, rcv}) {
    auto* tree = sp->GetElementTree();
    ASSERT_TRUE(tree->AddElement(1).ok());
    tree->GetElement(1)->SetSharedId(1);
    tree->GetElement(1)->SetSharedId(100);
    tree->GetElement(1)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 20, 20)));
  }
  // The shelf exists only in the source space: the receiver has no counterpart.
  AddBox(src->GetElementTree(), 2, 1, 15, 15);
  src->GetElementTree()->GetElement(2)->SetSharedId(200);
  const std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", {0, 0, 0})};
  const auto actor = src->CreateActor(anchors);
  src->GetActor(actor)->SetElementId(1);
  const auto avatar = rcv->CreateActor(anchors);
  rcv->GetActor(avatar)->SetElementId(1);
  src->SetHub(ctx->GetHub());
  rcv->SetHub(ctx->GetHub());
  src->SetMovementGate(0.0, 0.0);
  src->Bind(actor, "root", RelationKernelType::kAvatarSpacing);
  IdealDriver follow(rcv, avatar, {3, 3}, 0.0);

  src->UpdateActorState(actor, Eigen::Vector2d(5, 5), 0.0);
  src->UpdateActorState(actor, Eigen::Vector2d(6, 5), 0.0);
  ReceiverProposal p;
  ASSERT_TRUE(rcv->ReadProposal(avatar, &p));
  EXPECT_EQ(p.status, ProposalStatus::kNone)
      << "nothing maps: not a hold, not a failure";
  EXPECT_TRUE(p.anchors.empty());
  const RealizationResult* r = rcv->GetRealizationTargets(avatar);
  ASSERT_NE(r, nullptr);
  EXPECT_NE(r->status, SolverStatus::kFailed);
  for (const auto& t : r->targets)
    EXPECT_NE(t.kernel_type, RelationKernelType::kAvatarSpacing)
        << "no root target is invented for an unmapped referent";
}


// D9 for Facing: another person is a Facing
// referent like any visible element, on the source and on the receiver. Three people, each the source in its own space
// and a received avatar (an embodied driver's port, nothing else) in the other two, as the app runs them: Spacing on
// the root, Facing on the head (FreeformCollaborationApp.AvatarRelationBindings). A faces C; in B's space the people
// stand elsewhere (rotated about A's body by `rotation`), and A's head there must turn to C's counterpart. Independent
// oracle: the bearing from A's receiver head to C's counterpart.
struct PersonFacingWorld {
  static constexpr element_id_t kRoom = 1;
  static constexpr element_id_t kPerson[3] = {20, 30, 31};  // A, B, C
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space* space[3] = {nullptr, nullptr, nullptr};  // A's, B's, C's
  spatialblend::entity_id_t actor[3][3] = {};        // actor[space][person]
  uint64_t seq = 0;

  PersonFacingWorld() {
    const std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", {0, 0, 0}), SubjectAnchor("head", {0, 0, 1.6})};
    for (auto& sp : space) {
      sp = ctx->AddSpace();
      auto* tree = sp->GetElementTree();
      EXPECT_TRUE(tree->AddElement(kRoom).ok());
      tree->GetElement(kRoom)->SetSharedId(kRoom);
      tree->GetElement(kRoom)->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 20, 20)));
      for (element_id_t person : kPerson) {
        EXPECT_TRUE(tree->AddElement(person, kRoom).ok());
        tree->GetElement(person)->SetSharedId(person);
      }
    }
    for (int s = 0; s < 3; ++s) {
      for (int p = 0; p < 3; ++p) {
        actor[s][p] = space[s]->CreateActor(anchors);
        space[s]->GetActor(actor[s][p])->SetElementId(kPerson[p]);
      }
      space[s]->SetHub(ctx->GetHub());
      space[s]->SetMovementGate(0.0, 0.0);
      space[s]->Bind(actor[s][s], "root", RelationKernelType::kAvatarSpacing);
      space[s]->Bind(actor[s][s], "head", RelationKernelType::kAvatarFacing);
    }
  }

  // A received person, where its driver put it in space `s` (root and head ports).
  void Place(int s, int p, const Eigen::Vector2d& at, double yaw) {
    ++seq;
    EmbodiedPoseInput root;
    root.x = at.x(); root.y = at.y(); root.yaw = yaw; root.epoch = 1;
    EXPECT_EQ(space[s]->PublishAnchorPose(actor[s][p], 0, root, 0.0, seq), EmbodiedPublishResult::kOk);
    uint32_t head = 0;
    ASSERT_TRUE(space[s]->AnchorIdFor(actor[s][p], "head", &head));
    EmbodiedPoseInput h = root;
    h.z = 1.6;
    EXPECT_EQ(space[s]->PublishAnchorPose(actor[s][p], head, h, 0.0, seq), EmbodiedPublishResult::kOk);
  }
  // A person's own tick in its own space.
  void Tick(int p, const Eigen::Vector2d& at, double yaw) {
    space[p]->UpdateActorState(actor[p][p], at, yaw);
  }
};

// `table`: a table (0.8 m tall) stands just in front of C in A's space, as people stand across a table: A sees C's
// head over it, never C's feet.
void ExpectTheReceiverFacesThePersonTheSourceFaces(bool table) {
  constexpr int A = 0, B = 1, C = 2;
  for (const double rotation : {0.0, M_PI / 2, M_PI}) {
    PersonFacingWorld w;
    if (table) AddBox(w.space[A]->GetElementTree(), 40, PersonFacingWorld::kRoom, 10, 12.5);
    // A's space: A at (10, 4), C at (10, 14) straight ahead (+y), B at (16, 4) to A's right.
    const Eigen::Vector2d a_s(10, 4), b_s(16, 4), c_s(10, 14);
    // B's space: A at (10, 10); C 5 m away at `rotation` + 45 degrees, B 4 m away a quarter turn to C's right.
    const Eigen::Vector2d a_r(10, 10);
    const double q = rotation + M_PI / 4;
    const Eigen::Vector2d c_r = a_r + 5.0 * Eigen::Vector2d(std::cos(q), std::sin(q));
    const Eigen::Vector2d b_r = a_r + 4.0 * Eigen::Vector2d(std::cos(q - M_PI / 2), std::sin(q - M_PI / 2));
    // B and C are live sources (each ticks in its own space), so every other space receives them.
    for (int k = 0; k < 3; ++k) {
      w.Tick(B, Eigen::Vector2d(5, 5 + 0.1 * k), 0.0);
      w.Tick(C, Eigen::Vector2d(15, 15 + 0.1 * k), 0.0);
    }
    // Their drivers put them where each space has them.
    w.Place(A, B, b_s, M_PI);
    w.Place(A, C, c_s, -M_PI / 2);
    w.Place(B, A, a_r, rotation);
    w.Place(B, C, c_r, 0.0);
    // A turns its head toward C (the last 0.1 rad of the turn), then looks at C.
    for (int k = 0; k <= 5; ++k) w.Tick(A, a_s, M_PI / 2 - 0.1 + 0.02 * k);

    bool c_is_referent = false;
    for (const auto& m : w.space[A]->GetBindings(w.actor[A][A])[1].latest_matches)
      c_is_referent |= m.element_id == PersonFacingWorld::kPerson[C];
    EXPECT_TRUE(c_is_referent) << "rotation " << rotation << ": the person A looks at is a Facing referent";

    ReceiverProposal p;
    ASSERT_TRUE(w.space[B]->ReadProposal(w.actor[B][A], &p));
    uint32_t head = 0;
    ASSERT_TRUE(w.space[B]->AnchorIdFor(w.actor[B][A], "head", &head));
    const AnchorProposal* h = p.Find(head);
    ASSERT_NE(h, nullptr) << "rotation " << rotation;
    ASSERT_TRUE(h->status == ProposalStatus::kExact || h->status == ProposalStatus::kInexact)
        << "rotation " << rotation << ": status " << static_cast<int>(h->status);
    const double oracle = std::atan2(c_r.y() - a_r.y(), c_r.x() - a_r.x());
    EXPECT_NEAR(std::remainder(h->yaw - oracle, 2 * M_PI), 0.0, 1e-3)
        << "rotation " << rotation << ": the head faces C's counterpart, bearing " << oracle << ", got " << h->yaw
        << " (B's counterpart is at " << std::atan2(b_r.y() - a_r.y(), b_r.x() - a_r.x()) << ")";
  }
}

TEST(FamilyScope, D9_AnotherPersonIsAFacingReferentAndTheReceiverFacesItsCounterpart) {
  ExpectTheReceiverFacesThePersonTheSourceFaces(/*table=*/false);
}

TEST(FamilyScope, D9_APersonAcrossATableIsSeenOverIt) {
  ExpectTheReceiverFacesThePersonTheSourceFaces(/*table=*/true);
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
