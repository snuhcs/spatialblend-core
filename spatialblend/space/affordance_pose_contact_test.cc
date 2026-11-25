// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/affordance_pose_contact_test.cc
//
// The ONE affordance-pose capability: an affordance part's contact frame plus the embodiment's REGISTERED contact
// (skeleton + bone + local contact), mapped between matched parts of two spaces. A seat and a writing face differ only in the part's
// frame and chart (data): this file drives both through the same rig and the same oracles, with registered embodiments whose contact
// offsets differ (legs of different length, a palm or a controller tip), and checks:
//   - the receiver's contact lands on the mapped point of ITS part, at ITS part's plane, whatever the source's registered offset;
//   - the driver's bone target (BoneFromAnchor) puts the receiver's registered contact exactly on that point;
//   - seats of different height, yaw, depth and footprint (chair, sofa, stool, bench) and boards of different size;
//   - the contact yaw is the part-relative yaw of the source;
//   - A -> B -> A returns the source contact.
// Offsets are authored test values spanning the 33 Meta presets' measured sit-bone range (-0.165..-0.288 m down, -0.011..+0.182 m forward).
// Oracles are analytic and share no code with the charts.
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/embodiment/embodiment.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace test {
namespace {

constexpr element_id_t kRoom = 1;
constexpr element_id_t kObj = 10;
constexpr shared_element_id_t kRoomShared = 100;
constexpr shared_element_id_t kObjShared = 1000;
constexpr double kPi = 3.14159265358979323846;

std::shared_ptr<Polygon2D> Rect(double x0, double y0, double x1, double y1) {
  return std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});
}
std::shared_ptr<Polygon2D> Ngon(int n, double r) {
  std::vector<Eigen::Vector2d> p;
  for (int i = 0; i < n; ++i) p.emplace_back(r * std::cos(2 * kPi * i / n), r * std::sin(2 * kPi * i / n));
  return std::make_shared<Polygon2D>(p);
}
Eigen::Matrix4d Pose(double x, double y, double z, double yaw) {
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t.topLeftCorner<3, 3>() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  t.topRightCorner<3, 1>() = Eigen::Vector3d(x, y, z);
  return t;
}
Eigen::Matrix4d FaceFrame(double ht, double zc) {  // part x along the board, y up, z the outward normal
  Eigen::Matrix4d m = Eigen::Matrix4d::Zero();
  m(0, 2) = 1;
  m(1, 0) = 1;
  m(2, 1) = 1;
  m(3, 3) = 1;
  m(0, 3) = ht;
  m(2, 3) = zc;
  return m;
}

// One affordance object of a room: the element pose in the room, the part's frame in the element frame, the part's polygon and chart.
struct Obj {
  double x = 5, y = 5, yaw = 0;
  Eigen::Matrix4d part_frame = Eigen::Matrix4d::Identity();
  std::shared_ptr<Polygon2D> boundary;
  SurfaceChartKind chart = SurfaceChartKind::kRadial;
  std::string tag = "Sittable";
  Eigen::Vector3d lo = {-0.5, -0.4, 0}, hi = {0.5, 0.4, 0.45};  // the element's volume
};
Eigen::Matrix4d PartToWorld(const Obj& o) { return Pose(o.x, o.y, 0, o.yaw) * o.part_frame; }

Obj Seat(double w, double d, double plane, double yaw, double part_yaw = 0.0, bool round = false) {
  Obj o;
  o.yaw = yaw;
  o.part_frame = Pose(0, 0, plane, part_yaw);
  o.boundary = round ? Ngon(16, w) : Rect(-0.5 * w, -0.5 * d, 0.5 * w, 0.5 * d);
  o.lo = {-0.5 * w, -0.5 * d, 0};
  o.hi = {0.5 * w, 0.5 * d, plane};
  if (round) {
    o.lo = {-w, -w, 0};
    o.hi = {w, w, plane};
  }
  return o;
}
Obj Board(double w, double h, double low, double yaw, SurfaceChartKind chart) {
  Obj o;
  o.yaw = yaw;
  o.part_frame = FaceFrame(0.03, low + 0.5 * h);
  o.boundary = Rect(-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h);
  o.chart = chart;
  o.tag = "Writable";
  o.lo = {-0.03, -0.5 * w, low};
  o.hi = {0.03, 0.5 * w, low + h};
  return o;
}

void BuildScene(Space* s, const Obj& o) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kRoom).ok());
  tree->GetElement(kRoom)->SetSharedId(kRoomShared);
  tree->GetElement(kRoom)->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 10, 10)));
  ASSERT_TRUE(tree->AddElement(kObj, kRoom).ok());
  Element* e = tree->GetElement(kObj);
  e->SetSharedId(kObjShared);
  e->SetTag(o.tag);
  e->SetLocalTransform(Pose(o.x, o.y, 0, o.yaw));
  auto vol = std::make_shared<PolytopeSpatialDefinition>(o.lo, o.hi, GeometryRole::kObstacle);
  SubRegion part;
  part.name = "contact_face";
  part.tag = o.tag;
  part.local_transform = o.part_frame;
  part.boundary = o.boundary;
  part.chart = o.chart;
  vol->AddSubRegion(part);
  e->SetSpatialDefinition(vol);
}

// An embodiment registering `anchor` on bone "b" with the local contact offset (x forward, z up in the bone frame), as the driver does.
std::string Spec(const std::string& anchor, Eigen::Vector3d offset) {
  std::ostringstream s;
  s.precision(17);
  s << R"({"format":"spatialblend.embodiment","schema_version":"1.0.0","id":"fixture","skeleton":{"joints":[{"name":"b","parent":-1}]},)"
    << R"("anchors":[{"name":"root","bone":null,"families":["Spacing"]},{"name":")" << anchor
    << R"(","bone":"b","offset":{"position":[)" << offset.x() << "," << offset.y() << "," << offset.z()
    << R"(]},"families":["AffordancePose"],"provenance":{"procedure":"fixture","source":"test fixture"}}]})";
  return s.str();
}

struct Rig {
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space *src = nullptr, *rcv = nullptr;
  spatialblend::entity_id_t source = 0, avatar = 0;
  uint32_t src_anchor = 0, rcv_anchor = 0;
  Embodiment src_emb, rcv_emb;
  std::string anchor;
  Obj a, b;
  uint64_t seq = 0, bseq = 0;

  Rig(const Obj& oa, const Obj& ob, const std::string& anchor_name, Eigen::Vector3d src_offset, Eigen::Vector3d rcv_offset)
      : anchor(anchor_name), a(oa), b(ob) {
    src = ctx->AddSpace();
    rcv = ctx->AddSpace();
    BuildScene(src, a);
    BuildScene(rcv, b);
    source = src->CreateActor({SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))});
    avatar = rcv->CreateActor({SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))});
    std::string err;
    EXPECT_TRUE(Embodiment::FromJson(Spec(anchor, src_offset), &src_emb, &err)) << err;
    EXPECT_TRUE(Embodiment::FromJson(Spec(anchor, rcv_offset), &rcv_emb, &err)) << err;
    EXPECT_EQ(src->RegisterEmbodiment(source, src_emb), Space::RegisterEmbodimentResult::kOk);
    EXPECT_EQ(rcv->RegisterEmbodiment(avatar, rcv_emb), Space::RegisterEmbodimentResult::kOk);
    src->GetActor(source)->SetElementId(kRoom);
    rcv->GetActor(avatar)->SetElementId(kRoom);
    src->SetHub(ctx->GetHub());
    rcv->SetHub(ctx->GetHub());
    src->SetMovementGate(0.0, 0.0);
    EXPECT_NE(src->Bind(source, anchor, RelationKernelType::kAvatarAffordancePose, a.tag), nullptr);
    EXPECT_TRUE(src->AnchorIdFor(source, anchor, &src_anchor));
    EXPECT_TRUE(rcv->AnchorIdFor(avatar, anchor, &rcv_anchor));
    EmbodiedPoseInput in;
    in.x = 1.0;
    in.y = 1.0;
    in.epoch = 1;
    EXPECT_EQ(rcv->PublishEmbodiedPose(avatar, in, 0.0, ++seq), EmbodiedPublishResult::kOk);
    PublishBone(rcv, avatar, rcv_emb, RigidTransform{Eigen::Vector3d(1.0, 1.0, 1.0), Eigen::Quaterniond::Identity()}, ++bseq);
  }

  static void PublishBone(Space* s, spatialblend::entity_id_t actor, const Embodiment& e, const RigidTransform& bone, uint64_t seq) {
    BonePoseInput b;
    b.p[0] = bone.position.x();
    b.p[1] = bone.position.y();
    b.p[2] = bone.position.z();
    b.q[0] = bone.rotation.x();
    b.q[1] = bone.rotation.y();
    b.q[2] = bone.rotation.z();
    b.q[3] = bone.rotation.w();
    b.epoch = 1;
    EXPECT_EQ(s->PublishBonePose(actor, e.BoneId("b"), b, 0.0, seq), EmbodiedPublishResult::kOk);
  }

  // One source frame: the sender's REGISTERED contact is at `contact` with world yaw `yaw`; the driver publishes the BONE
  // (contact * offset^-1) and the core composes the anchor back. Returns the receiver's proposal.
  ReceiverProposal Frame(const Eigen::Vector3d& contact, double yaw) {
    RigidTransform want;
    want.position = contact;
    want.rotation = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    const RigidTransform bone = BoneFromAnchor(*src_emb.FindAnchor(anchor), want);
    PublishBone(src, source, src_emb, bone, ++bseq);
    // The source feed of a registered anchor: composed from its bone by the registration (never stated directly).
    std::vector<Space::ActorObservation> obs;
    EXPECT_TRUE(src->ComposeBoneObservations(source, {{"b", bone}}, &obs));
    for (const auto& o : obs)
      if (o.anchor_name == anchor) EXPECT_TRUE(src->SetBoneAnchorWorldPose(source, anchor, o.position, o.yaw));
    const Eigen::Vector3d root = Eigen::Vector3d(contact.x() + 0.7, contact.y(), 0.0);
    src->UpdateActorState(source, Eigen::Vector2d(root.x(), root.y()), 0.0);
    ReceiverProposal p;
    EXPECT_TRUE(rcv->ReadProposal(avatar, &p));
    return p;
  }
  const AnchorProposal* Anchor(const ReceiverProposal& p) const {
    for (const auto& x : p.anchors)
      if (x.anchor_name == anchor) return &x;
    return nullptr;
  }
};

double Wrap(double a) { return std::remainder(a, 2.0 * kPi); }

// The radial chart between two CENTRED rectangles, written independently: the ratio of the point along its ray, the same ratio on the
// receiver's ray at the same angle.
Eigen::Vector2d RadialRect(double w, double h, double w2, double h2, const Eigen::Vector2d& p) {
  if (p.norm() == 0.0) return p;
  const double ratio = std::max(std::abs(p.x()) / (0.5 * w), std::abs(p.y()) / (0.5 * h));
  const Eigen::Vector2d dir = p.normalized();
  double reach = 1e300;
  if (std::abs(dir.x()) > 1e-15) reach = std::min(reach, 0.5 * w2 / std::abs(dir.x()));
  if (std::abs(dir.y()) > 1e-15) reach = std::min(reach, 0.5 * h2 / std::abs(dir.y()));
  return dir * ratio * reach;
}

Eigen::Vector3d PartPoint(const Obj& o, const Eigen::Vector3d& local) { return (PartToWorld(o) * local.homogeneous()).head<3>(); }
double PartYaw(const Obj& o) {
  const Eigen::Matrix4d m = PartToWorld(o);
  return std::atan2(m(1, 0), m(0, 0));
}

// Offsets spanning the measured Meta presets (down -0.165..-0.288, forward -0.011..+0.182) and a controller tip.
const Eigen::Vector3d kLegOffsets[] = {{0.182, 0.0, -0.165}, {0.063, 0.0, -0.186}, {-0.011, 0.0, -0.288}, {0.100, 0.012, -0.220}};

void ExpectContact(const Rig& r, const ReceiverProposal& p, const Eigen::Vector3d& want_world, double want_yaw,
                   const std::string& what) {
  const AnchorProposal* c = r.Anchor(p);
  ASSERT_NE(c, nullptr) << what;
  EXPECT_TRUE(c->engaged()) << what;
  EXPECT_EQ(c->status, ProposalStatus::kExact) << what;
  EXPECT_NEAR(c->x, want_world.x(), 1e-9) << what;
  EXPECT_NEAR(c->y, want_world.y(), 1e-9) << what;
  EXPECT_NEAR(c->z, want_world.z(), 1e-9) << what;
  EXPECT_NEAR(Wrap(c->yaw - want_yaw), 0.0, 1e-9) << what;
  // The driver's bone target puts the receiver's registered contact exactly there.
  RigidTransform tgt;
  tgt.position = Eigen::Vector3d(c->x, c->y, c->z);
  tgt.rotation = Eigen::Quaterniond(Eigen::AngleAxisd(c->yaw, Eigen::Vector3d::UnitZ()));
  const RigidTransform bone = BoneFromAnchor(*r.rcv_emb.FindAnchor(r.anchor), tgt);
  const RigidTransform back = ComposeAnchor(*r.rcv_emb.FindAnchor(r.anchor), bone);
  EXPECT_NEAR((back.position - tgt.position).norm(), 0.0, 1e-12) << what;
}

// ---------------------------------------------------------------------------------------------------------------------------
// Seats: same capability, radial chart (the default), seat frame = part frame with +x the front.
// ---------------------------------------------------------------------------------------------------------------------------

TEST(AffordancePoseContact, SeatsOfDifferentHeightYawAndFootprintAreMatchedAtTheirOwnPlane) {
  struct Case { const char* name; double w, d; double w2, d2; double plane, plane2; double yaw, yaw2; };
  const Case cases[] = {
      {"chair -> identical chair, yawed 90 deg, 0.45 -> 0.50", 0.9, 0.8, 0.9, 0.8, 0.45, 0.50, 0.0, kPi / 2},
      {"chair -> sofa 2.0x0.8, 0.45 -> 0.42", 0.9, 0.8, 2.0, 0.8, 0.45, 0.42, 0.0, kPi / 3},
      {"sofa -> chair", 2.0, 0.8, 0.9, 0.8, 0.42, 0.45, 1.0, 0.0},
      {"chair -> bench 1.5x0.4, 0.45 -> 0.65", 0.9, 0.8, 1.5, 0.4, 0.45, 0.65, 0.2, -kPi / 2},
      {"bench -> chair", 1.5, 0.4, 0.9, 0.8, 0.65, 0.45, kPi, 0.4},
  };
  const Eigen::Vector2d contacts[] = {{0.0, 0.0}, {0.12, 0.05}, {-0.3, 0.2}, {0.2, -0.15}};
  for (const Case& c : cases) {
    for (const Eigen::Vector3d& src_off : kLegOffsets) {
      for (const Eigen::Vector3d& rcv_off : {kLegOffsets[0], kLegOffsets[2]}) {
        Obj a = Seat(c.w, c.d, c.plane, c.yaw), b = Seat(c.w2, c.d2, c.plane2, c.yaw2);
        b.x = 7.0;
        b.y = 3.0;
        Rig rig(a, b, "pelvis", src_off, rcv_off);
        for (const auto& s : contacts) {
          if (std::abs(s.x()) > 0.5 * c.w || std::abs(s.y()) > 0.5 * c.d) continue;  // on the seat
          const Eigen::Vector2d m = RadialRect(c.w, c.d, c.w2, c.d2, s);
          const double src_yaw = PartYaw(a) + 0.3;  // sitting 0.3 rad off the seat front
          const ReceiverProposal p = rig.Frame(PartPoint(a, Eigen::Vector3d(s.x(), s.y(), 0.0)), src_yaw);
          ExpectContact(rig, p, PartPoint(b, Eigen::Vector3d(m.x(), m.y(), 0.0)), PartYaw(b) + 0.3,
                        std::string(c.name) + " contact " + std::to_string(s.x()) + "," + std::to_string(s.y()));
        }
      }
    }
  }
}

TEST(AffordancePoseContact, ARoundStoolTakesTheCentreToTheCentreAndLandsOnItsOwnHeight) {
  Obj chair = Seat(0.9, 0.8, 0.45, 0.4);
  Obj stool = Seat(0.16, 0.16, 0.65, 1.1, 0.0, /*round=*/true);
  stool.x = 2.0;
  stool.y = 7.0;
  Rig rig(chair, stool, "pelvis", kLegOffsets[1], kLegOffsets[1]);
  const ReceiverProposal p = rig.Frame(PartPoint(chair, Eigen::Vector3d(0, 0, 0)), PartYaw(chair));
  ExpectContact(rig, p, PartPoint(stool, Eigen::Vector3d(0, 0, 0)), PartYaw(stool), "chair centre -> stool centre");
  const AnchorProposal* c = rig.Anchor(p);
  ASSERT_NE(c, nullptr);
  EXPECT_NEAR(c->z, 0.65, 1e-12);
}

TEST(AffordancePoseContact, SeatRoundTripReturnsTheSourceContactForSameShapeAndAnyOffsets) {
  const Obj a = Seat(0.9, 0.8, 0.45, 0.2);
  Obj b = Seat(0.9, 0.8, 0.55, 2.0);
  b.x = 7.0;
  b.y = 6.0;
  for (const Eigen::Vector3d& so : kLegOffsets) {
    for (const Eigen::Vector3d& ro : kLegOffsets) {
      Rig ab(a, b, "pelvis", so, ro), ba(b, a, "pelvis", ro, so);
      const Eigen::Vector3d s = PartPoint(a, Eigen::Vector3d(0.17, -0.12, 0.0));
      const ReceiverProposal p = ab.Frame(s, PartYaw(a) + 0.7);
      const AnchorProposal* onb = ab.Anchor(p);
      ASSERT_NE(onb, nullptr);
      const ReceiverProposal q = ba.Frame(Eigen::Vector3d(onb->x, onb->y, onb->z), onb->yaw);
      ExpectContact(ba, q, s, PartYaw(a) + 0.7, "A->B->A");
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------------
// Boards: the same capability, a frame chart. The registered contact is a palm or a controller tip.
// ---------------------------------------------------------------------------------------------------------------------------

TEST(AffordancePoseContact, ABoardContactOfAPalmOrAControllerTipLandsAtTheSameUvWhateverTheRegisteredOffset) {
  const Eigen::Vector3d palm(0.0611, 0.0403, -0.0310), tip(0.0, 0.0, 0.14), tip_other(0.0, 0.0, 0.11);
  struct Case { double w, h, w2, h2; };
  const Case cases[] = {{1.2, 0.9, 0.8, 0.6}, {1.6, 0.9, 1.2, 0.9}, {1.2, 0.9, 1.6, 0.9}};
  for (auto kind : {SurfaceChartKind::kFrameStretch, SurfaceChartKind::kFrameContain}) {
    for (const Case& c : cases) {
      Obj a = Board(c.w, c.h, 0.9, -kPi / 2, kind), b = Board(c.w2, c.h2, 0.7, kPi, kind);
      a.x = 5.0;
      a.y = 9.5;
      b.x = 9.5;
      b.y = 4.0;
      for (const auto& offs : {std::make_pair(palm, tip), std::make_pair(tip, palm), std::make_pair(tip, tip_other)}) {
        Rig rig(a, b, "hand_r", offs.first, offs.second);
        for (const auto uv : {std::make_pair(0.0, 0.0), std::make_pair(1.0, 1.0), std::make_pair(0.5, 0.5), std::make_pair(0.2, 0.8)}) {
          const double x = (uv.first - 0.5) * c.w, y = (uv.second - 0.5) * c.h;
          double mx, my;
          if (kind == SurfaceChartKind::kFrameStretch) {
            mx = (uv.first - 0.5) * c.w2;
            my = (uv.second - 0.5) * c.h2;
          } else {
            const double s = std::min(c.w2 / c.w, c.h2 / c.h);
            mx = x * s;
            my = y * s;
          }
          const ReceiverProposal p = rig.Frame(PartPoint(a, Eigen::Vector3d(x, y, 0.0)), PartYaw(a) + 0.2);
          ExpectContact(rig, p, PartPoint(b, Eigen::Vector3d(mx, my, 0.0)), PartYaw(b) + 0.2, "board uv");
        }
      }
    }
  }
}

TEST(AffordancePoseContact, ABoardAndASeatAreOneCapabilityDifferingOnlyInThePartData) {
  // The same rig, the same anchor machinery, only the part frame/boundary/chart (and the tag) differ: no entry point or branch is
  // specific to either. Both engage on contact with their own part and release on leaving it.
  for (int kind = 0; kind < 2; ++kind) {
    Obj a = kind == 0 ? Seat(0.9, 0.8, 0.45, 0.0) : Board(1.2, 0.9, 0.9, -kPi / 2, SurfaceChartKind::kFrameStretch);
    Obj b = kind == 0 ? Seat(0.9, 0.8, 0.50, kPi / 2) : Board(0.8, 0.6, 0.8, kPi, SurfaceChartKind::kFrameStretch);
    b.x = 7.0;
    b.y = 4.0;
    Rig rig(a, b, kind == 0 ? "pelvis" : "hand_r", kLegOffsets[1], kLegOffsets[2]);
    const Eigen::Vector3d off = PartPoint(a, Eigen::Vector3d(0, 0, 0.3));  // 0.3 m clear of the part plane
    const Eigen::Vector3d on = PartPoint(a, Eigen::Vector3d(0, 0, 0.0));
    ReceiverProposal p = rig.Frame(off, PartYaw(a));
    const AnchorProposal* c = rig.Anchor(p);
    EXPECT_TRUE(c == nullptr || !c->engaged()) << kind;
    p = rig.Frame(on, PartYaw(a));
    c = rig.Anchor(p);
    ASSERT_NE(c, nullptr);
    EXPECT_TRUE(c->engaged()) << kind;
    EXPECT_NE(c->cause & kCauseSeatEngage, 0u) << kind;
    p = rig.Frame(off, PartYaw(a));
    c = rig.Anchor(p);
    EXPECT_TRUE(c == nullptr || !c->engaged()) << kind;
    if (c) EXPECT_NE(c->cause & kCauseSeatRelease, 0u) << kind;
  }
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
