// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/seat_transfer_test.cc
//
// The sit transfer:
// a user sits on a chair in space A and the avatar sits on the corresponding chair in space B. TESTS ONLY, no solver
// or ABI change. What they pin:
//
//   AF_SIT_UNSHARED   An element that carries a `Sittable` seat but has NO shared id is a referent of no AffordancePose
//                     relation (MA-1, SP-D7-03): no source relation, no pelvis proposal row that engages. Three routes
//                     reach the resolver: the MRUK chair (no tag), a tagged unshared LEAF (ScopeResolver::ResolveProximal's
//                     "leaf scope: include self for grounding" branch), and a tagged unshared element beside the root.
//   SitTransfer       One shared, linked chair per room: the sender's pelvis contact gives the receiver a pelvis proposal at
//                     ITS seat's height with a seat-relative yaw, SEAT_ENGAGE -> ARRIVED -> SEAT_RELEASE.
//   MrukSeat          A COUCH volume whose origin is on the seat plane, whose seat part is the seat outline and whose
//                     backrest rises above it: contact is the seat plane, never the volume top.
//   SeatFront         relative_yaw is measured from the seat part's +x on both sides.
//
// The sender and the receiver are two Spaces of one LocalContext and the app binds the pelvis the way the product does
// (a plain Bind with the `Sittable` tag; no scope root). Oracles are analytic.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace test {
namespace {

constexpr element_id_t kRoom = 1;
constexpr element_id_t kPerson = 90;  // the subject's person leaf (Rig person_leaf)
constexpr shared_element_id_t kRoomShared = 100;
constexpr double kPi = 3.14159265358979323846;
constexpr double kStand = 0.9;  // a standing pelvis height

// One seat element of a scene. Authored scenario geometry; no algorithm reads a number here as a threshold.
struct SeatCfg {
  element_id_t id = 10;
  shared_element_id_t shared = 0;  // 0: no shared id
  std::string tag = "Sittable";
  double x = 3.0, y = 3.0;
  double yaw = 0.0;          // the element's yaw in the room
  double plane = 0.45;       // the seat plane height (world z)
  double part_yaw = 0.0;     // the seat part frame's yaw in the element frame
  bool mruk = false;         // MRUK COUCH shape (see BuildScene)
  double hx = 0.5, hy = 0.4;  // plain seat: the half extents of the box and of the seat part (property tests vary them)
  bool seat_part = true;     // false: the element has no `seat` part (a counterpart that offers no seat)
};

Eigen::Matrix4d Pose(double x, double y, double z, double yaw) {
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 0) = std::cos(yaw);
  t(0, 1) = -std::sin(yaw);
  t(1, 0) = std::sin(yaw);
  t(1, 1) = std::cos(yaw);
  t(0, 3) = x;
  t(1, 3) = y;
  t(2, 3) = z;
  return t;
}

std::shared_ptr<Polygon2D> Rect(double x0, double y0, double x1, double y1) {
  return std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});
}

// A room 10 x 10 with the given seats. Plain seat: a box [-0.5,0.5] x [-0.4,0.4] x [0,plane], the "seat" part on its top
// face covering the whole footprint. MRUK shape (section 1.2 of the design): the element origin is ON the seat plane, the
// volume spans [-plane, +0.5] in the element frame (the backrest rises 0.5 m above the plane), and the seat part is the
// seat outline only: x in [-0.1, 0.5] (+x is the front), the back strip x in [-0.5, -0.1] is the backrest.
void BuildScene(Space* s, const std::vector<SeatCfg>& seats) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kRoom).ok());
  tree->GetElement(kRoom)->SetSharedId(kRoomShared);
  tree->GetElement(kRoom)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 10, 10)));
  for (const SeatCfg& c : seats) {
    ASSERT_TRUE(tree->AddElement(c.id, kRoom).ok());
    Element* e = tree->GetElement(c.id);
    if (c.shared != 0) e->SetSharedId(c.shared);
    if (!c.tag.empty()) e->SetTag(c.tag);
    e->SetLocalTransform(Pose(c.x, c.y, c.mruk ? c.plane : 0.0, c.yaw));
    const double zlo = c.mruk ? -c.plane : 0.0, zhi = c.mruk ? 0.5 : c.plane;
    auto vol = std::make_shared<PolytopeSpatialDefinition>(Eigen::Vector3d(-c.hx, -c.hy, zlo),
                                                           Eigen::Vector3d(c.hx, c.hy, zhi), GeometryRole::kObstacle);
    SubRegion seat;
    seat.name = "seat";
    seat.tag = "Sittable";
    seat.local_transform = Pose(0, 0, c.mruk ? 0.0 : c.plane, c.part_yaw);
    seat.boundary = c.mruk ? Rect(-0.1, -0.4, 0.5, 0.4) : Rect(-c.hx, -c.hy, c.hx, c.hy);
    if (c.seat_part) vol->AddSubRegion(seat);
    e->SetSpatialDefinition(vol);
  }
}

double Wrap(double a) { return std::remainder(a, 2.0 * kPi); }

struct Rig {
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space* src = nullptr;
  Space* rcv = nullptr;
  spatialblend::entity_id_t source = 0, avatar = 0;
  uint32_t pelvis_id = 0;
  uint64_t seq = 0;

  // `facing`: also bind a Facing on the sender's root (the unshared-seat checks read it; the SitTransfer tests do not).
  // `person_leaf`: the subject's element is a person LEAF under the room, as in the product, instead of the room itself (the room as the
  // subject's element is excluded from its own scopes as "actor element", which hides the room as a Spacing referent).
  Rig(const std::vector<SeatCfg>& src_seats, const std::vector<SeatCfg>& rcv_seats, bool facing = false, bool person_leaf = false,
      bool product_movement_gate = false) {
    src = ctx->AddSpace();
    rcv = ctx->AddSpace();
    BuildScene(src, src_seats);
    BuildScene(rcv, rcv_seats);
    const std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
                                                SubjectAnchor("pelvis", Eigen::Vector3d(0, 0, kStand))};
    if (person_leaf) {
      EXPECT_TRUE(src->GetElementTree()->AddElement(kPerson, kRoom).ok());
      EXPECT_TRUE(rcv->GetElementTree()->AddElement(kPerson, kRoom).ok());
    }
    source = src->CreateActor(anchors);
    src->GetActor(source)->SetElementId(person_leaf ? kPerson : kRoom);
    avatar = rcv->CreateActor(anchors);
    rcv->GetActor(avatar)->SetElementId(person_leaf ? kPerson : kRoom);
    src->SetHub(ctx->GetHub());
    rcv->SetHub(ctx->GetHub());
    if (!product_movement_gate) src->SetMovementGate(0.0, 0.0);  // else the space's own deadband, as the app runs it
    src->Bind(source, "root", RelationKernelType::kAvatarSpacing);
    if (facing) src->Bind(source, "root", RelationKernelType::kAvatarFacing);
    // The product binding: AffordancePose on the pelvis, the `Sittable` tag, no scope root.
    EXPECT_NE(src->Bind(source, "pelvis", RelationKernelType::kAvatarAffordancePose, "Sittable"), nullptr);
    EXPECT_TRUE(rcv->AnchorIdFor(avatar, "pelvis", &pelvis_id));
    EmbodiedPoseInput in;
    in.x = 1.0;
    in.y = 1.0;
    in.epoch = 1;
    EXPECT_EQ(rcv->PublishEmbodiedPose(avatar, in, 0.0, ++seq), EmbodiedPublishResult::kOk);
    PublishReceiverPelvis(1.0, 1.0, kStand, 0.0);  // the receiver's body stands away from its seat
  }

  void PublishReceiverPelvis(double x, double y, double z, double yaw) {
    EmbodiedPoseInput in;
    in.x = x;
    in.y = y;
    in.z = z;
    in.yaw = yaw;
    in.epoch = 1;
    EXPECT_EQ(rcv->PublishAnchorPose(avatar, pelvis_id, in, 0.0, ++seq), EmbodiedPublishResult::kOk);
  }

  // One source frame: the sender's pelvis at (px, py, z) facing `yaw`, its root at `root`. Returns the receiver's set.
  // The returned proposal lives as long as the Rig (a stable address): callers keep `Pelvis(...)` pointers into it across later frames.
  const ReceiverProposal& Frame(double px, double py, double z, double yaw, const Eigen::Vector2d& root) {
    src->SetAnchorWorldPose(source, "pelvis", Eigen::Vector3d(px, py, z), yaw);
    src->UpdateActorState(source, root, 0.0);
    history.emplace_back();
    EXPECT_TRUE(rcv->ReadProposal(avatar, &history.back()));
    return history.back();
  }
  std::deque<ReceiverProposal> history;
  static const AnchorProposal* Pelvis(const ReceiverProposal& p) {
    for (const auto& a : p.anchors)
      if (a.anchor_name == "pelvis") return &a;
    return nullptr;
  }
  // The source's grounded AffordancePose relations of the pelvis.
  // By LOCAL element id: RelationState::referent_shared_id equals the element id when the element is unshared.
  std::vector<element_id_t> SourcePoseReferents() const {
    return Referents("pelvis", RelationKernelType::kAvatarAffordancePose);
  }
  // The source's referents of one anchor and family (Spacing and Facing are bound on the root).
  std::vector<element_id_t> Referents(const char* anchor, RelationKernelType kernel) const {
    std::vector<element_id_t> out;
    for (const RelationState& r : src->GetRelationStates(source))
      if (r.anchor_name == anchor && r.kernel_type == kernel) out.push_back(r.referent_element_id);
    return out;
  }
  // The element ids the room's free space holds as obstacles (every obstacle-role element, shared or not: F).
  std::vector<element_id_t> RoomObstacles() const {
    std::vector<element_id_t> out;
    const auto* nav = dynamic_cast<const NavigableSpatialDefinition*>(
        src->GetElementTree()->GetElement(kRoom)->GetSpatialDefinition());
    if (nav)
      for (const auto& o : nav->GetObstacles()) out.push_back(static_cast<element_id_t>(o.key));
    return out;
  }
};

// =============================================================================================
// AF-SIT-UNSHARED
// =============================================================================================

constexpr element_id_t kShared = 10, kOther = 11;
constexpr shared_element_id_t kSharedSid = 1000, kOtherSid = 2000;

// Source room: a shared chair at (3,3) (the pelvis never goes near it) and a second chair `other` at (6.5,6.5) with a seat
// part. The receiver has the counterparts of the SHARED ones only: `other` is a chair of the receiver's room with the same
// shared id as the source's (0 when the source's is unshared, so it is unshared there too).
struct Variant {
  const char* name;
  std::string other_tag;          // the unshared chair's element tag
  shared_element_id_t other_sid;  // 0: unshared
  bool root_on_chair;             // the sender's root stands in the chair's footprint (the leaf-scope route)
};

std::vector<SeatCfg> SourceSeats(const Variant& v, bool drop_other = false) {
  SeatCfg a;
  a.id = kShared;
  a.shared = kSharedSid;
  a.x = 3.0;
  a.y = 3.0;
  SeatCfg b;
  b.id = kOther;
  b.shared = v.other_sid;
  b.tag = v.other_tag;
  b.x = 6.5;
  b.y = 6.5;
  if (drop_other) return {a};  // the same source room with the unshared chair DELETED from its tree
  return {a, b};
}
std::vector<SeatCfg> ReceiverSeats(const Variant& v) {
  SeatCfg a;
  a.id = kShared;
  a.shared = kSharedSid;
  a.x = 7.0;
  a.y = 2.0;
  a.yaw = 0.5 * kPi;
  a.plane = 0.50;
  SeatCfg b;
  b.id = kOther;
  b.shared = v.other_sid;
  b.tag = v.other_tag;
  b.x = 2.0;
  b.y = 7.0;
  b.plane = 0.40;
  return {a, b};
}

// Sit on the `other` chair: walk in at standing height, lower the pelvis through its plane, stay, rise. Returns the receiver
// pelvis rows of every tick.
struct SitRun {
  std::vector<bool> has_row, engaged, exact;
  std::vector<uint32_t> cause;
  std::vector<ProposalStatus> root_status;  // the receiver's ROOT proposal status
  std::vector<std::array<double, 3>> root_pose;  // the receiver's root proposal x, y, yaw
  std::vector<bool> root_position_target;  // ... and whether it carries a POSITION target (Spacing), not only a Facing yaw
  std::vector<std::vector<element_id_t>> pose_referents, spacing_referents, facing_referents;
  std::vector<element_id_t> obstacles;  // the room's free-space obstacles after the last tick
};
SitRun SitOnOther(Rig& r, const Variant& v) {
  SitRun out;
  // The chair's footprint is x in [6.0, 7.0], y in [6.1, 6.9]. The pelvis is on the seat part at (6.6, 6.5); the root is
  // under it (leaf-scope route) or 0.1 m beside the footprint (the room-scope route).
  const double px = 6.6, py = 6.5;
  const Eigen::Vector2d root = v.root_on_chair ? Eigen::Vector2d(px, py) : Eigen::Vector2d(7.1, py);
  std::vector<double> zs;
  for (int k = 0; k < 6; ++k) zs.push_back(kStand);
  for (double z = kStand; z > 0.45;) zs.push_back(z -= 0.05);
  zs.push_back(0.45);
  for (int k = 0; k < 6; ++k) zs.push_back(0.45);
  for (int k = 0; k < 4; ++k) zs.push_back(kStand);  // stand up
  for (double z : zs) {
    const ReceiverProposal p = r.Frame(px, py, z, 0.0, root);
    const AnchorProposal* a = Rig::Pelvis(p);
    out.has_row.push_back(a != nullptr);
    out.engaged.push_back(a != nullptr && a->engaged());
    out.exact.push_back(a != nullptr && a->status == ProposalStatus::kExact);
    out.cause.push_back(a ? a->cause : 0u);
    out.root_status.push_back(p.status);  // the ROOT proposal is the base class of the set
    out.root_position_target.push_back(p.position_specified());
    out.root_pose.push_back({p.x, p.y, p.yaw});
    out.pose_referents.push_back(r.SourcePoseReferents());
    out.spacing_referents.push_back(r.Referents("root", RelationKernelType::kAvatarSpacing));
    out.facing_referents.push_back(r.Referents("root", RelationKernelType::kAvatarFacing));
  }
  out.obstacles = r.RoomObstacles();
  return out;
}

bool Any(const std::vector<bool>& v) {
  for (bool b : v)
    if (b) return true;
  return false;
}

bool Contains(const std::vector<element_id_t>& v, element_id_t e) {
  return std::find(v.begin(), v.end(), e) != v.end();
}

// The receiver's pelvis row: it exists exactly while the source has an AffordancePose relation for the pelvis. The control has it on
// every tick (so a missing row is never a broken rig); an unshared seat has it on NONE: "no AffordancePose payload or proposal for the
// pelvis on any tick" (design 5.1), which is stronger than "not engaged".
void ExpectPelvisRowEveryTick(const SitRun& run, const char* name) {
  ASSERT_FALSE(run.has_row.empty()) << name;
  for (size_t t = 0; t < run.has_row.size(); ++t) EXPECT_TRUE(run.has_row[t]) << name << " tick " << t << ": no pelvis row";
}
void ExpectNoPelvisRow(const SitRun& run, const char* name) {
  ASSERT_FALSE(run.has_row.empty()) << name;
  for (size_t t = 0; t < run.has_row.size(); ++t) EXPECT_FALSE(run.has_row[t]) << name << " tick " << t << ": a pelvis row exists";
}

void ExpectNoSit(const SitRun& run, const char* name) {
  ExpectNoPelvisRow(run, name);
  EXPECT_FALSE(Any(run.engaged)) << name << ": the pelvis proposal engaged";
  EXPECT_FALSE(Any(run.exact)) << name << ": the pelvis got a target";
  for (size_t t = 0; t < run.cause.size(); ++t) {
    EXPECT_EQ(run.cause[t] & (kCauseSeatEngage | kCauseSeatRelease), 0u) << name << " tick " << t;
    for (element_id_t e : run.pose_referents[t])
      EXPECT_NE(e, kOther) << name << " tick " << t << ": the unshared chair is an AffordancePose referent";
  }
}

// The control for every variant: the SAME scene and walk with the chair shared is a sit. It proves the geometry reaches
// contact, so the absence of a sit above is the shared-id rule and nothing else.
void ExpectSit(const SitRun& run, const char* name) {
  ExpectPelvisRowEveryTick(run, name);
  EXPECT_TRUE(Any(run.engaged)) << name << ": the control never engaged";
  bool seen_referent = false;
  for (const auto& refs : run.pose_referents)
    for (element_id_t e : refs) seen_referent |= e == kOther;
  EXPECT_TRUE(seen_referent) << name << ": the control has no AffordancePose relation on the chair";
}

TEST(AF_SIT_UNSHARED, AnMrukChairWithNoTagAndNoSharedIdIsNoSeat) {
  // The normal case: an MRUK chair is label-free local geometry (design 1.2): no shared id, no Sittable tag.
  const Variant v{"mruk-chair", "", 0, true};
  Rig r(SourceSeats(v), ReceiverSeats(v));
  ExpectNoSit(SitOnOther(r, v), v.name);
}

TEST(AF_SIT_UNSHARED, ASittableTaggedLeafWithNoSharedIdIsNoSeat) {
  // The branch the design flags: the sender's root is inside the unshared chair, which is a LEAF, so
  // ScopeResolver::ResolveProximal returns the leaf itself ("leaf scope: include self for grounding"), past the
  // HasCounterpart filter. With the Sittable tag the tag check admits it too: only the shared-id rule stops it.
  const Variant v{"tagged-leaf", "Sittable", 0, true};
  Rig r(SourceSeats(v), ReceiverSeats(v));
  ExpectNoSit(SitOnOther(r, v), v.name);
}

TEST(AF_SIT_UNSHARED, ASittableTaggedUnsharedLeafWithTheRootBesideItIsNoSeat) {
  // The same leaf with the sender's ROOT 0.1 m outside the footprint: the AffordancePose scope is resolved from the
  // anchor (the pelvis, inside the chair), not from the root.
  const Variant v{"tagged-beside", "Sittable", 0, false};
  Rig r(SourceSeats(v), ReceiverSeats(v));
  ExpectNoSit(SitOnOther(r, v), v.name);
}

// Design 5.1: "Spacing and Facing unaffected; the unshared chair still an obstacle". What an unshared seat takes away is a
// REFERENT (MA-1), nothing else: the shared chair stays a Facing referent and, with the root outside the unshared footprint, a
// Spacing referent; the unshared chair is in no family's scope and still blocks free space. With the root INSIDE the unshared
// leaf the unshared element is transparent to the Spacing scope (SP-D7-04/05): the scope is the room's own (the shared chair), as with
// the unshared chair deleted, and the receiver's root keeps following Spacing (a proposal, not NONE); the shared control is the replay scope.
void ExpectLocalGeometryOnly(const SitRun& run, const Variant& v) {
  EXPECT_TRUE(Contains(run.obstacles, kOther)) << v.name << ": the unshared chair left the free space (it must stay an obstacle)";
  EXPECT_TRUE(Contains(run.obstacles, kShared)) << v.name;
  for (size_t t = 0; t < run.has_row.size(); ++t) {
    EXPECT_FALSE(Contains(run.spacing_referents[t], kOther)) << v.name << " tick " << t << ": a Spacing referent";
    EXPECT_FALSE(Contains(run.facing_referents[t], kOther)) << v.name << " tick " << t << ": a Facing referent";
    EXPECT_TRUE(Contains(run.facing_referents[t], kShared)) << v.name << " tick " << t << ": Facing lost the shared chair";
    if (v.root_on_chair) {
      EXPECT_EQ(run.spacing_referents[t], std::vector<element_id_t>{kShared})
          << v.name << " tick " << t << ": the root on an unshared chair has the room's own scope (the shared chair), as with the chair deleted";
      EXPECT_TRUE((run.root_status[t] == ProposalStatus::kExact || run.root_status[t] == ProposalStatus::kInexact) &&
                  run.root_position_target[t])
          << v.name << " tick " << t << ": the root stopped following Spacing (status " << static_cast<int>(run.root_status[t])
          << ", position target " << run.root_position_target[t] << ")";
    } else
      EXPECT_TRUE(Contains(run.spacing_referents[t], kShared)) << v.name << " tick " << t << ": Spacing lost the shared chair";
  }
}

TEST(AF_SIT_UNSHARED, SpacingAndFacingAreUnaffectedAndTheUnsharedChairStaysAnObstacle) {
  for (const Variant& v : {Variant{"mruk-chair", "", 0, true}, Variant{"tagged-leaf", "Sittable", 0, true},
                           Variant{"tagged-beside", "Sittable", 0, false}}) {
    Rig r(SourceSeats(v), ReceiverSeats(v), /*facing=*/true, /*person_leaf=*/true);
    const SitRun run = SitOnOther(r, v);
    ExpectNoSit(run, v.name);
    ExpectLocalGeometryOnly(run, v);
  }
}

// Spacing ALONE (no Facing to carry the root), SP-D7-05: a root standing on an unshared chair is exactly as if the chair did not exist.
// The same walk runs twice: with the unshared chair in the sender's room (the root stands inside it), and with the chair DELETED from the
// sender's tree. The Spacing referent sets are identical on every tick, and so is the receiver's root pose (x, y, yaw) and its position
// target. Two things deliberately differ, and are not the Spacing scope: the pelvis row (the AffordancePose scope of an unshared chair is
// pinned to NO row, design 5.1) and the solve STATUS (exact vs inexact: the unshared chair stays an obstacle of the sender's free space,
// SP-D7-03, which the deleted tree does not have).
TEST(AF_SIT_UNSHARED, ARootOnAnUnsharedChairHasTheScopeOfTheRoomWithTheChairDeleted) {
  for (const Variant& v : {Variant{"mruk-chair", "", 0, true}, Variant{"tagged-leaf", "Sittable", 0, true}}) {
    Rig with(SourceSeats(v), ReceiverSeats(v), /*facing=*/false, /*person_leaf=*/true);
    Rig without(SourceSeats(v, /*drop_other=*/true), ReceiverSeats(v), /*facing=*/false, /*person_leaf=*/true);
    const SitRun a = SitOnOther(with, v), b = SitOnOther(without, v);
    ExpectNoSit(a, v.name);  // no pelvis row: the unshared chair is no seat
    ASSERT_EQ(a.has_row.size(), b.has_row.size());
    for (size_t t = 0; t < a.has_row.size(); ++t) {
      EXPECT_EQ(a.spacing_referents[t], b.spacing_referents[t]) << v.name << " tick " << t;
      EXPECT_EQ(a.spacing_referents[t], std::vector<element_id_t>{kShared}) << v.name << " tick " << t;
      EXPECT_EQ(a.root_position_target[t], b.root_position_target[t]) << v.name << " tick " << t;
      EXPECT_TRUE((a.root_status[t] == ProposalStatus::kExact || a.root_status[t] == ProposalStatus::kInexact) &&
                  a.root_position_target[t])
          << v.name << " tick " << t << ": the root is not transferred (status " << static_cast<int>(a.root_status[t]) << ")";
      for (int k = 0; k < 3; ++k)
        EXPECT_NEAR(a.root_pose[t][k], b.root_pose[t][k], 1e-9) << v.name << " tick " << t << " component " << k;
    }
  }
}

TEST(AF_SIT_UNSHARED, TheSharedControlKeepsTheChairInEveryFamilyAndAsAnObstacle) {
  for (const Variant& v : {Variant{"control-leaf", "Sittable", kOtherSid, true},
                           Variant{"control-beside", "Sittable", kOtherSid, false}}) {
    Rig r(SourceSeats(v), ReceiverSeats(v), /*facing=*/true);
    const SitRun run = SitOnOther(r, v);
    ExpectSit(run, v.name);
    EXPECT_TRUE(Contains(run.obstacles, kOther)) << v.name;
    for (size_t t = 0; t < run.has_row.size(); ++t) {
      EXPECT_TRUE(Contains(run.facing_referents[t], kOther)) << v.name << " tick " << t;
      EXPECT_TRUE(Contains(run.spacing_referents[t], kOther)) << v.name << " tick " << t << ": the shared chair is a Spacing referent";
    }
  }
}

TEST(AF_SIT_UNSHARED, TheSameWalkOnTheChairSharedAndLinkedIsASeat) {
  for (const Variant& v : {Variant{"control-leaf", "Sittable", kOtherSid, true},
                           Variant{"control-beside", "Sittable", kOtherSid, false}}) {
    Rig r(SourceSeats(v), ReceiverSeats(v));
    ExpectSit(SitOnOther(r, v), v.name);
  }
}

// =============================================================================================
// A shared, linked seat: SEAT_ENGAGE -> ARRIVED -> SEAT_RELEASE
// =============================================================================================

TEST(SitTransfer, ASharedLinkedSeatGivesTheReceiverAPelvisAtItsOwnSeatHeightAndASeatRelativeYaw) {
  SeatCfg a;  // the sender's chair: plane 0.45, yaw 0
  a.shared = kSharedSid;
  a.x = 3.0;
  a.y = 3.0;
  SeatCfg b;  // the receiver's linked chair: another place, plane 0.60, turned 90 degrees
  b.shared = kSharedSid;
  b.x = 7.0;
  b.y = 2.0;
  b.yaw = 0.5 * kPi;
  b.plane = 0.60;
  Rig r({a}, {b});
  const double sender_yaw = 0.3;  // the sender's pelvis yaw relative to its seat's +x (the seat yaw is 0)
  const Eigen::Vector2d at(3.1, 3.05);

  // Standing: bound, not touching, no target.
  for (int k = 0; k < 3; ++k) {
    const AnchorProposal* p = Rig::Pelvis(r.Frame(at.x(), at.y(), kStand, sender_yaw, at));
    ASSERT_NE(p, nullptr);
    EXPECT_FALSE(p->engaged());
    EXPECT_NE(p->status, ProposalStatus::kExact);
  }
  // The pelvis reaches the sender's plane: SEAT_ENGAGE, the target at the RECEIVER's plane and seat-relative yaw.
  const ReceiverProposal engage = r.Frame(at.x(), at.y(), 0.45, sender_yaw, at);
  const AnchorProposal* e = Rig::Pelvis(engage);
  ASSERT_NE(e, nullptr);
  EXPECT_TRUE(e->engaged());
  EXPECT_EQ(e->status, ProposalStatus::kExact);
  EXPECT_TRUE(e->cause & kCauseSeatEngage);
  EXPECT_EQ(e->referent_shared_id, kSharedSid);
  EXPECT_NEAR(e->z, 0.60, 1e-9) << "the receiver's seat height, not the sender's 0.45";
  EXPECT_NEAR(Wrap(e->yaw - (0.5 * kPi + sender_yaw)), 0.0, 1e-9) << "receiver seat +x plus the sender's relative yaw";
  EXPECT_TRUE(e->yaw_specified());
  EXPECT_FALSE(e->arrived()) << "the receiver's pelvis is still standing";
  // The target lies on the receiver's seat part: (7.0 +- 0.4, 2.0 +- 0.5) after a 90 degree turn.
  EXPECT_LE(std::fabs(e->x - 7.0), 0.4 + 1e-9);
  EXPECT_LE(std::fabs(e->y - 2.0), 0.5 + 1e-9);

  // The receiver's driver realizes it; the next frame reports ARRIVED (its sampled pelvis touches its seat).
  r.PublishReceiverPelvis(e->x, e->y, e->z, e->yaw);
  const AnchorProposal* arrived = Rig::Pelvis(r.Frame(at.x(), at.y(), 0.45, sender_yaw, at));
  ASSERT_NE(arrived, nullptr);
  EXPECT_TRUE(arrived->engaged());
  EXPECT_TRUE(arrived->arrived());
  EXPECT_FALSE(arrived->cause & kCauseSeatEngage) << "the engage cause is the transition frame only";

  // The sender rises: SEAT_RELEASE on that frame, then no target.
  const ReceiverProposal rel = r.Frame(at.x(), at.y(), kStand, sender_yaw, at);
  const AnchorProposal* rp = Rig::Pelvis(rel);
  ASSERT_NE(rp, nullptr);
  EXPECT_FALSE(rp->engaged());
  EXPECT_TRUE(rp->cause & kCauseSeatRelease);
  EXPECT_NE(rp->status, ProposalStatus::kExact);
  const AnchorProposal* after = Rig::Pelvis(r.Frame(at.x(), at.y(), kStand, sender_yaw, at));
  ASSERT_NE(after, nullptr);
  EXPECT_FALSE(after->engaged());
  EXPECT_FALSE(after->cause & kCauseSeatRelease);
}

// =============================================================================================
// An MRUK-shaped seat: contact is the seat plane, not the volume top
// =============================================================================================

TEST(MrukSeat, ContactIsTheSeatPlaneAndNeverTheVolumeTop) {
  SeatCfg a;  // sender COUCH: origin on the plane 0.45, volume top at 0.95 (the backrest)
  a.shared = kSharedSid;
  a.mruk = true;
  a.x = 3.0;
  a.y = 3.0;
  SeatCfg b;  // receiver COUCH: plane 0.40, turned, backrest the same shape
  b.shared = kSharedSid;
  b.mruk = true;
  b.x = 7.0;
  b.y = 6.0;
  b.yaw = kPi;
  b.plane = 0.40;
  Rig r({a}, {b});
  const double front = 3.0 + 0.2;  // on the seat outline: element x in [-0.1, 0.5] -> world 2.9 .. 3.5
  const Eigen::Vector2d root(front, 3.0);

  // Anywhere above the plane is not contact, the volume top (0.95) included.
  for (double z : {kStand, 0.95, 0.60, 0.451})
    EXPECT_FALSE(Rig::Pelvis(r.Frame(front, 3.0, z, 0.0, root))->engaged()) << "z=" << z;
  // On the plane inside the seat outline: contact, target at the RECEIVER's plane (0.40).
  const AnchorProposal* on = Rig::Pelvis(r.Frame(front, 3.0, 0.45, 0.0, root));
  ASSERT_NE(on, nullptr);
  EXPECT_TRUE(on->engaged());
  EXPECT_TRUE(on->cause & kCauseSeatEngage);
  EXPECT_NEAR(on->z, 0.40, 1e-9);
  // The receiver part is turned 180 degrees: the sender's +x yaw 0 comes out as pi on the receiver.
  EXPECT_NEAR(std::fabs(Wrap(on->yaw - kPi)), 0.0, 1e-9);
  // Over the backrest strip (element x in [-0.5, -0.1]) at the plane's height: inside the volume, outside the seat part.
  EXPECT_FALSE(Rig::Pelvis(r.Frame(2.7, 3.0, 0.45, 0.0, root))->engaged()) << "the backrest is not the seat";
  EXPECT_FALSE(Rig::Pelvis(r.Frame(2.7, 3.0, 0.30, 0.0, root))->engaged()) << "the backrest strip, below the plane";
  // Back on the seat outline, lower than the plane (having crossed it): contact.
  EXPECT_TRUE(Rig::Pelvis(r.Frame(front, 3.0, 0.30, 0.0, root))->engaged());
}

// =============================================================================================
// The seat front: relative_yaw is measured from the seat part's +x
// =============================================================================================

// One sit with the given seats and sender yaw; the receiver's pelvis yaw.
double ReceiverYaw(const SeatCfg& sender, const SeatCfg& receiver, double pelvis_yaw) {
  Rig r({sender}, {receiver});
  const Eigen::Vector2d at(sender.x + 0.1, sender.y + 0.05);
  const AnchorProposal* p = Rig::Pelvis(r.Frame(at.x(), at.y(), sender.plane, pelvis_yaw, at));
  EXPECT_NE(p, nullptr);
  if (p == nullptr) return NAN;
  EXPECT_TRUE(p->engaged());
  EXPECT_EQ(p->status, ProposalStatus::kExact);
  return p->yaw;
}

TEST(SeatFront, RotatingTheReceiverSeatsPlusXRotatesTheReceiverYawByTheSameAngle) {
  SeatCfg sender;
  sender.shared = kSharedSid;
  const double rel = 0.4;  // sender yaw 0.4 against a sender seat +x of 0
  for (double phi : {0.0, 0.5 * kPi, kPi, -0.5 * kPi, 0.3}) {
    // (a) the receiver ELEMENT turned by phi: the part's +x is the element's +x turned
    SeatCfg by_element;
    by_element.shared = kSharedSid;
    by_element.x = 7.0;
    by_element.y = 6.0;
    by_element.yaw = phi;
    EXPECT_NEAR(Wrap(ReceiverYaw(sender, by_element, rel) - (phi + rel)), 0.0, 1e-9) << "element yaw " << phi;
    // (b) the receiver seat PART frame turned by phi inside an unturned element (what an app's authoring decides)
    SeatCfg by_part = by_element;
    by_part.yaw = 0.0;
    by_part.part_yaw = phi;
    EXPECT_NEAR(Wrap(ReceiverYaw(sender, by_part, rel) - (phi + rel)), 0.0, 1e-9) << "part yaw " << phi;
    // (c) both: they add
    SeatCfg both = by_element;
    both.part_yaw = 0.25 * kPi;
    EXPECT_NEAR(Wrap(ReceiverYaw(sender, both, rel) - (phi + 0.25 * kPi + rel)), 0.0, 1e-9) << "both " << phi;
  }
}

TEST(SeatFront, TheSenderSeatsPlusXIsTheReferenceOfTheRelativeYaw) {
  // The same pelvis yaw in the world, the sender's seat part turned by phi: the relative yaw is the world yaw minus phi,
  // so the receiver (+x unturned) comes out at 0.7 - phi. Facing the seat's +x is yaw 0 relative, on either side.
  SeatCfg receiver;
  receiver.shared = kSharedSid;
  receiver.x = 7.0;
  receiver.y = 6.0;
  const double world_yaw = 0.7;
  for (double phi : {0.0, 0.5 * kPi, -0.5 * kPi, kPi}) {
    SeatCfg sender;
    sender.shared = kSharedSid;
    sender.yaw = phi;
    EXPECT_NEAR(Wrap(ReceiverYaw(sender, receiver, world_yaw) - (world_yaw - phi)), 0.0, 1e-9) << "sender yaw " << phi;
    EXPECT_NEAR(Wrap(ReceiverYaw(sender, receiver, phi)), 0.0, 1e-9) << "facing the sender seat's +x lands on +x";
  }
}


// =============================================================================================
// Property sweep: the seat contact maps between ANY two linked seats (height, yaw, depth, width)
// =============================================================================================
//
// The oracle is independent of the solver: the radial chart of two convex seat outlines computed here from the rectangles (the
// area centroid of a rectangle is its centre; the angle is measured from the part's +x; the ratio is the distance over the
// boundary's distance along the same ray), then the element's pose. Tolerances are numeric (1e-6 m / 1e-6 rad): the sweep is
// closed-form arithmetic of the same quantities in double precision, so nothing larger is needed.

struct Rng {
  uint32_t s;
  explicit Rng(uint32_t seed) : s(seed) {}
  double U() {  // [0, 1)
    s = s * 1664525u + 1013904223u;
    return (s >> 8) / 16777216.0;
  }
  double In(double lo, double hi) { return lo + (hi - lo) * U(); }
};

// A seat part is a rectangle [-hx,hx] x [-hy,hy] in its own frame, whose +x is the front.
double RayToRect(double hx, double hy, double angle) {
  const double c = std::cos(angle), s = std::sin(angle);
  const double tx = std::fabs(c) > 1e-12 ? hx / std::fabs(c) : 1e30;
  const double ty = std::fabs(s) > 1e-12 ? hy / std::fabs(s) : 1e30;
  return std::min(tx, ty);
}

struct SeatFrame {  // the world pose of a seat part
  double x, y, yaw;
};
SeatFrame PartFrame(const SeatCfg& c) { return {c.x, c.y, c.yaw + c.part_yaw}; }

Eigen::Vector2d ToPart(const SeatFrame& f, double wx, double wy) {
  const double dx = wx - f.x, dy = wy - f.y, c = std::cos(f.yaw), s = std::sin(f.yaw);
  return {c * dx + s * dy, -s * dx + c * dy};
}
Eigen::Vector2d FromPart(const SeatFrame& f, const Eigen::Vector2d& p) {
  const double c = std::cos(f.yaw), s = std::sin(f.yaw);
  return {f.x + c * p.x() - s * p.y(), f.y + s * p.x() + c * p.y()};
}

// The chart image of a world point on the sender's seat, on the receiver's seat.
Eigen::Vector2d ChartImage(const SeatCfg& from, const SeatCfg& to, double wx, double wy) {
  const Eigen::Vector2d p = ToPart(PartFrame(from), wx, wy);
  const double angle = std::atan2(p.y(), p.x());
  const double ratio = p.norm() / RayToRect(from.hx, from.hy, angle);
  const double r = ratio * RayToRect(to.hx, to.hy, angle);
  return FromPart(PartFrame(to), Eigen::Vector2d(r * std::cos(angle), r * std::sin(angle)));
}

SeatCfg RandomSeat(Rng& rng, double cx, double cy) {
  SeatCfg c;
  c.shared = kSharedSid;
  c.x = cx;
  c.y = cy;
  c.yaw = rng.In(-kPi, kPi);
  c.part_yaw = rng.U() < 0.5 ? 0.0 : rng.In(-kPi, kPi);
  c.plane = rng.In(0.30, 0.75);  // a low stool to a bar stool
  c.hx = rng.In(0.15, 1.0);      // a stool's width to a sofa's
  c.hy = rng.In(0.15, 0.5);
  return c;
}

TEST(SeatSweep, TheContactMapsOntoTheLinkedSeatForAnyHeightYawAndDepthAndRoundTrips) {
  Rng rng(20261024u);
  int engaged_cases = 0;
  for (int n = 0; n < 200; ++n) {
    const SeatCfg a = RandomSeat(rng, 3.0, 3.0), b = RandomSeat(rng, 7.0, 6.5);
    // A source contact strictly inside the sender's seat (not on its rim): the chart is defined on the interior.
    const double u = rng.In(-0.9, 0.9) * a.hx, v = rng.In(-0.9, 0.9) * a.hy;
    const Eigen::Vector2d contact = FromPart(PartFrame(a), Eigen::Vector2d(u, v));
    const double rel = rng.In(-kPi, kPi);                       // the sender's yaw relative to its seat's +x
    const double sender_yaw = PartFrame(a).yaw + rel;
    const Eigen::Vector2d root(contact.x() + 1.5, contact.y());  // the root need not be near

    Rig r({a}, {b});
    r.Frame(contact.x(), contact.y(), kStand, sender_yaw, root);
    const ReceiverProposal& p = r.Frame(contact.x(), contact.y(), a.plane, sender_yaw, root);
    const AnchorProposal* e = Rig::Pelvis(p);
    ASSERT_NE(e, nullptr) << "case " << n;
    ASSERT_TRUE(e->engaged()) << "case " << n;
    ++engaged_cases;
    EXPECT_EQ(e->status, ProposalStatus::kExact) << n;
    EXPECT_NEAR(e->z, b.plane, 1e-9) << "case " << n << ": the receiver's own seat height (" << a.plane << " -> " << b.plane << ")";
    const Eigen::Vector2d want = ChartImage(a, b, contact.x(), contact.y());
    EXPECT_NEAR(e->x, want.x(), 1e-6) << "case " << n;
    EXPECT_NEAR(e->y, want.y(), 1e-6) << "case " << n;
    const Eigen::Vector2d on_part = ToPart(PartFrame(b), e->x, e->y);
    EXPECT_LE(std::fabs(on_part.x()), b.hx + 1e-9) << "case " << n << ": the target is on the receiver's seat";
    EXPECT_LE(std::fabs(on_part.y()), b.hy + 1e-9) << "case " << n;
    EXPECT_NEAR(Wrap(e->yaw - (PartFrame(b).yaw + rel)), 0.0, 1e-6) << "case " << n << ": receiver part +x plus the relative yaw";

    // The registered contact lands exactly on the target seat frame: a receiver pelvis published AT the target touches its seat.
    r.PublishReceiverPelvis(e->x, e->y, e->z, e->yaw);
    const AnchorProposal* arrived = Rig::Pelvis(r.Frame(contact.x(), contact.y(), a.plane, sender_yaw, root));
    ASSERT_NE(arrived, nullptr);
    EXPECT_TRUE(arrived->arrived()) << "case " << n;

    // Round trip A -> B -> A: the receiver's seat is now the sender, sitting where it was put, and the first seat is the receiver.
    Rig back({b}, {a});
    const Eigen::Vector2d at_b(e->x, e->y);
    const double b_yaw = e->yaw;
    back.Frame(at_b.x(), at_b.y(), kStand, b_yaw, Eigen::Vector2d(at_b.x() + 1.5, at_b.y()));
    const AnchorProposal* home = Rig::Pelvis(back.Frame(at_b.x(), at_b.y(), b.plane, b_yaw, Eigen::Vector2d(at_b.x() + 1.5, at_b.y())));
    ASSERT_NE(home, nullptr) << "case " << n;
    ASSERT_TRUE(home->engaged()) << "case " << n;
    EXPECT_NEAR(home->x, contact.x(), 1e-6) << "round trip, case " << n;
    EXPECT_NEAR(home->y, contact.y(), 1e-6) << "round trip, case " << n;
    EXPECT_NEAR(home->z, a.plane, 1e-9) << "round trip, case " << n;
    EXPECT_NEAR(Wrap(home->yaw - sender_yaw), 0.0, 1e-6) << "round trip, case " << n;
  }
  EXPECT_EQ(engaged_cases, 200);
}

TEST(SeatSweep, TheTargetDoesNotDependOnTheReceiversPelvisHeightWhileStanding) {
  // Embodiments differ in leg length: the receiver's standing pelvis (and so its sit-bone point before it moves) is higher or lower. The
  // seat target is the receiver SEAT's contact, so it must not depend on where the receiver's body is.
  SeatCfg a, b;
  a.shared = b.shared = kSharedSid;
  b.x = 7.0;
  b.y = 6.0;
  b.yaw = 1.1;
  b.plane = 0.55;
  const Eigen::Vector2d at(3.1, 3.05);
  std::vector<Eigen::Vector4d> targets;
  for (double stand : {0.75, 0.9, 1.05, 1.2}) {
    Rig r({a}, {b});
    r.PublishReceiverPelvis(1.0, 1.0, stand, 0.0);
    r.Frame(at.x(), at.y(), kStand, 0.2, at);
    const AnchorProposal* e = Rig::Pelvis(r.Frame(at.x(), at.y(), a.plane, 0.2, at));
    ASSERT_NE(e, nullptr);
    ASSERT_TRUE(e->engaged());
    EXPECT_FALSE(e->arrived()) << "still standing at " << stand;
    targets.emplace_back(e->x, e->y, e->z, e->yaw);
  }
  for (const auto& t : targets) EXPECT_LT((t - targets.front()).norm(), 1e-9);
}

TEST(SeatSweep, ACounterpartWithNoSeatPartOffersNoTargetAndNeverGarbage) {
  // The linked element has no `seat` part (a degenerate counterpart): no engaged target, no NaN, no crash. (The app guarantees a part by
  // construction, design 2.0; this pins that core fails closed when it is not so.)
  SeatCfg a, b;
  a.shared = b.shared = kSharedSid;
  b.x = 7.0;
  b.y = 6.0;
  b.seat_part = false;
  Rig r({a}, {b});
  const Eigen::Vector2d at(3.1, 3.05);
  r.Frame(at.x(), at.y(), kStand, 0.0, at);
  const ReceiverProposal& p = r.Frame(at.x(), at.y(), a.plane, 0.0, at);
  const AnchorProposal* e = Rig::Pelvis(p);
  if (e != nullptr) {
    EXPECT_FALSE(e->engaged() && e->status == ProposalStatus::kExact && e->arrived());
    EXPECT_TRUE(std::isfinite(e->x) && std::isfinite(e->y) && std::isfinite(e->z) && std::isfinite(e->yaw));
  }
}

TEST(SeatSweep, TheSenderLeavingThePlaneEndsTheSeatForEveryPairWithOneRelease) {
  Rng rng(7u);
  for (int n = 0; n < 50; ++n) {
    const SeatCfg a = RandomSeat(rng, 3.0, 3.0), b = RandomSeat(rng, 7.0, 6.5);
    const double cu = rng.In(-0.5, 0.5) * a.hx;  // drawn in order (argument evaluation order is unspecified)
    const double cv = rng.In(-0.5, 0.5) * a.hy;
    const Eigen::Vector2d contact = FromPart(PartFrame(a), Eigen::Vector2d(cu, cv));
    Rig r({a}, {b});
    const Eigen::Vector2d root(contact.x() + 1.5, contact.y());
    r.Frame(contact.x(), contact.y(), kStand, 0.0, root);
    ASSERT_TRUE(Rig::Pelvis(r.Frame(contact.x(), contact.y(), a.plane, 0.0, root))->engaged()) << n;
    const ReceiverProposal& released = r.Frame(contact.x(), contact.y(), a.plane + 0.02, 0.0, root);
    ASSERT_TRUE(Rig::Pelvis(released)->cause & kCauseSeatRelease) << n;
    const uint64_t release_revision = released.revision;
    const ReceiverProposal& after_set = r.Frame(contact.x(), contact.y(), kStand, 0.0, root);
    const AnchorProposal* after = Rig::Pelvis(after_set);
    EXPECT_FALSE(after->engaged()) << n;
    // With no request core publishes no new set, so the release set may stay the latest; the
    // release is the transition SET only: no later revision carries it.
    EXPECT_TRUE(after_set.revision == release_revision || !(after->cause & kCauseSeatRelease))
        << n << ": the release is the transition frame only";
    // A later set is forced by an event of the receiver's own (its driver declares a teleport: EMBODIED_RESET, which
    // core publishes with or without a request); that set must not carry the release again.
    EmbodiedPoseInput reset;
    reset.x = 1.0;
    reset.y = 1.0;
    reset.epoch = 2;
    reset.flags = kEmbodiedTeleported;
    ASSERT_EQ(r.rcv->PublishEmbodiedPose(r.avatar, reset, 0.0, ++r.seq), EmbodiedPublishResult::kOk) << n;
    const ReceiverProposal& later_set = r.Frame(contact.x(), contact.y(), kStand, 0.0, root);
    ASSERT_GT(later_set.revision, release_revision) << n << ": the reset publishes a new set";
    const AnchorProposal* later = Rig::Pelvis(later_set);
    ASSERT_NE(later, nullptr) << n;
    EXPECT_FALSE(later->engaged()) << n;
    EXPECT_FALSE(later->cause & kCauseSeatRelease) << n << ": the release is the transition set only";
  }
}


// =============================================================================================
// The movement gate must not hold back a sit
// =============================================================================================

TEST(SeatGate, ASitWhoseRootStaysStillStillReachesTheReceiverUnderTheProductsMovementGate) {
  // A user lowers into a chair: the pelvis goes down 0.45 m while the ROOT (the floor point under the head) hardly moves. The
  // movement gate was written for a still source's jitter (it compares the root pose only); a sit is not jitter, it is a change of
  // which supports the source touches. With the space's own deadband (the product runs it), the engagement must reach the receiver
  // on the frame it happens, not on the next heartbeat about a second later.
  SeatCfg a, b;
  a.shared = b.shared = kSharedSid;
  b.x = 7.0;
  b.y = 6.0;
  b.plane = 0.55;
  Rig r({a}, {b}, /*facing=*/false, /*person_leaf=*/false, /*product_movement_gate=*/true);
  const Eigen::Vector2d at(3.1, 3.05);
  // Stand beside the chair for longer than a heartbeat so the gate is armed, then lower the pelvis with the root fixed.
  for (int k = 0; k < 40; ++k) r.Frame(at.x(), at.y(), kStand, 0.0, at);
  bool engaged_on_the_touch_frame = false;
  for (double z = kStand; z > a.plane + 1e-9;) {
    z = std::max(a.plane, z - 0.05);
    const AnchorProposal* p = Rig::Pelvis(r.Frame(at.x(), at.y(), z, 0.0, at));
    ASSERT_NE(p, nullptr);
    if (z <= a.plane + 1e-12) engaged_on_the_touch_frame = p->engaged();
  }
  EXPECT_TRUE(engaged_on_the_touch_frame) << "the touch frame itself reaches the receiver";
  // And the release: rising off the seat with the root still.
  const AnchorProposal* up = Rig::Pelvis(r.Frame(at.x(), at.y(), a.plane + 0.05, 0.0, at));
  ASSERT_NE(up, nullptr);
  EXPECT_FALSE(up->engaged());
  EXPECT_TRUE(up->cause & kCauseSeatRelease) << "the release frame itself reaches the receiver";
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
