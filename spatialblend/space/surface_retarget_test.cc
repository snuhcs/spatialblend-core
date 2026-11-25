// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/surface_retarget_test.cc
//
// Surface retargeting: a hand/controller contact on the
// writing face of a board in space A lands, in space B, at the same (u,v) of B's board face and at the same normal offset, whatever
// the size, aspect ratio, wall or tilt of B's board. TESTS of the shared affordance-pose capability: the mapping is the SubRegion's
// declared chart (planar_frame_chart.h), the rest is the existing AffordancePose relation (engagement, scope, proposal). The seat
// is the same capability with the radial chart (seat_transfer_test.cc); a case here pins that the default is unchanged.
//
// The sender and the receiver are two Spaces of one LocalContext; the app binds the hand the way the product binds the pelvis (a plain
// Bind with the element tag). Oracles are analytic and share no code with the chart under test. Board sizes are authored scenario
// geometry (1.2x0.9 and 0.8x0.6 are both 4:3; 16:9 vs 4:3 differ in aspect), none is read by an algorithm as a threshold.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
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
constexpr element_id_t kBoard = 10;
constexpr shared_element_id_t kRoomShared = 100;
constexpr shared_element_id_t kBoardShared = 1000;
constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = 1e-9;

const SurfaceChartKind kRadial = SurfaceChartKind::kRadial;
const SurfaceChartKind kStretch = SurfaceChartKind::kFrameStretch;
const SurfaceChartKind kContain = SurfaceChartKind::kFrameContain;

Eigen::Matrix4d Rigid(double x, double y, double z, double yaw, double tilt = 0.0) {
  // yaw about z, then a tilt about the element's own y (the board leans back by `tilt`).
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t.topLeftCorner<3, 3>() = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                             Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitY()))
                                .toRotationMatrix();
  t.topRightCorner<3, 1>() = Eigen::Vector3d(x, y, z);
  return t;
}

std::shared_ptr<Polygon2D> Rect(double x0, double y0, double x1, double y1) {
  return std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});
}

// The part frame of the writing face in the board's element frame: part x along the board (element y), part y up (element z), part z
// the outward normal (element x), origin at the face centre. The same placement the scenario suite uses (affordance_scene VPlane).
Eigen::Matrix4d FaceFrame(double ht, double zc) {
  Eigen::Matrix4d m = Eigen::Matrix4d::Zero();
  m(0, 2) = 1;
  m(1, 0) = 1;
  m(2, 1) = 1;
  m(3, 3) = 1;
  m(0, 3) = ht;
  m(2, 3) = zc;
  return m;
}

struct BoardCfg {
  double w = 1.2, h = 0.9;  // the writing face size (m)
  double low = 0.9;         // the lower edge height
  double x = 5.0, y = 5.0;  // the board element in the room
  double yaw = 0.0;         // which wall it is on
  double tilt = 0.0;        // leans back
  SurfaceChartKind chart = kStretch;
  bool rectangular = true;  // false: a trapezoid face (cannot carry a frame chart)
  double ht = 0.03;         // half thickness
  bool has_part = true;
};

Eigen::Matrix4d PartToWorld(const BoardCfg& c) {
  return Rigid(c.x, c.y, 0.0, c.yaw, c.tilt) * FaceFrame(c.ht, c.low + 0.5 * c.h);
}
Eigen::Vector3d PartPoint(const BoardCfg& c, double px, double py, double n) {
  return (PartToWorld(c) * Eigen::Vector4d(px, py, n, 1.0)).head<3>();
}

// A room with one shared board. The slab spans the whole face so a hand on the face is inside the board's definition.
void BuildScene(Space* s, const BoardCfg& c) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kRoom).ok());
  tree->GetElement(kRoom)->SetSharedId(kRoomShared);
  tree->GetElement(kRoom)->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 10, 10)));
  ASSERT_TRUE(tree->AddElement(kBoard, kRoom).ok());
  Element* e = tree->GetElement(kBoard);
  e->SetSharedId(kBoardShared);
  e->SetTag("Writable");
  e->SetLocalTransform(Rigid(c.x, c.y, 0.0, c.yaw, c.tilt));
  auto vol = std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-c.ht, -0.5 * c.w, c.low), Eigen::Vector3d(c.ht, 0.5 * c.w, c.low + c.h), GeometryRole::kObstacle);
  if (c.has_part) {
    SubRegion face;
    face.name = "board";
    face.tag = "Writable";
    face.local_transform = FaceFrame(c.ht, c.low + 0.5 * c.h);
    face.boundary = c.rectangular
                        ? Rect(-0.5 * c.w, -0.5 * c.h, 0.5 * c.w, 0.5 * c.h)
                        : std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
                              {-0.5 * c.w, -0.5 * c.h}, {0.5 * c.w, -0.5 * c.h}, {0.3 * c.w, 0.5 * c.h}, {-0.3 * c.w, 0.5 * c.h}});
    face.chart = c.chart;
    vol->AddSubRegion(face);
  }
  e->SetSpatialDefinition(vol);
}

struct Rig {
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space* src = nullptr;
  Space* rcv = nullptr;
  spatialblend::entity_id_t source = 0, avatar = 0;
  uint32_t hand_id = 0;
  uint64_t seq = 0;
  BoardCfg a, b;

  Rig(const BoardCfg& src_cfg, const BoardCfg& rcv_cfg) : a(src_cfg), b(rcv_cfg) {
    src = ctx->AddSpace();
    rcv = ctx->AddSpace();
    BuildScene(src, a);
    BuildScene(rcv, b);
    const std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
                                                SubjectAnchor("hand_r", Eigen::Vector3d(0.3, 0, 1.1))};
    source = src->CreateActor(anchors);
    src->GetActor(source)->SetElementId(kRoom);
    avatar = rcv->CreateActor(anchors);
    rcv->GetActor(avatar)->SetElementId(kRoom);
    src->SetHub(ctx->GetHub());
    rcv->SetHub(ctx->GetHub());
    src->SetMovementGate(0.0, 0.0);
    src->Bind(source, "root", RelationKernelType::kAvatarSpacing);
    EXPECT_NE(src->Bind(source, "hand_r", RelationKernelType::kAvatarAffordancePose, "Writable"), nullptr);
    EXPECT_TRUE(rcv->AnchorIdFor(avatar, "hand_r", &hand_id));
    EmbodiedPoseInput in;
    in.x = 1.0;
    in.y = 1.0;
    in.epoch = 1;
    EXPECT_EQ(rcv->PublishEmbodiedPose(avatar, in, 0.0, ++seq), EmbodiedPublishResult::kOk);
    PublishReceiverHand(1.3, 1.0, 1.1, 0.0);  // the receiver's hand starts away from its board
  }

  void PublishReceiverHand(double x, double y, double z, double yaw) {
    EmbodiedPoseInput in;
    in.x = x;
    in.y = y;
    in.z = z;
    in.yaw = yaw;
    in.epoch = 1;
    EXPECT_EQ(rcv->PublishAnchorPose(avatar, hand_id, in, 0.0, ++seq), EmbodiedPublishResult::kOk);
  }

  // One source frame: the sender's hand at the world point `hand`, its root `root_in_front` metres in front of the board centre.
  ReceiverProposal Frame(const Eigen::Vector3d& hand, double root_in_front = 1.0) {
    src->SetAnchorWorldPose(source, "hand_r", hand, a.yaw + kPi);  // facing the board
    const Eigen::Vector3d root = PartPoint(a, 0.0, -a.low - 0.5 * a.h + 0.0, root_in_front);
    src->UpdateActorState(source, Eigen::Vector2d(root.x(), root.y()), 0.0);
    ReceiverProposal p;
    EXPECT_TRUE(rcv->ReadProposal(avatar, &p));
    return p;
  }
  // The hand at (u,v) of the sender's face, `n` metres along the normal.
  ReceiverProposal FrameUv(double u, double v, double n = 0.0) {
    return Frame(PartPoint(a, (u - 0.5) * a.w, (v - 0.5) * a.h, n));
  }
  static const AnchorProposal* Hand(const ReceiverProposal& p) {
    for (const auto& x : p.anchors)
      if (x.anchor_name == "hand_r") return &x;
    return nullptr;
  }
};

// The analytic oracle: the receiver world point of the source (u,v,n) under `kind`. Shares no code with PlanarFrameChart.
Eigen::Vector3d Expected(const BoardCfg& from, const BoardCfg& to, SurfaceChartKind kind, double u, double v, double n) {
  double x, y;
  if (kind == kStretch) {
    x = (u - 0.5) * to.w;
    y = (v - 0.5) * to.h;
  } else {  // contain: one uniform scale by the tighter ratio, centred
    const double s = std::min(to.w / from.w, to.h / from.h);
    x = (u - 0.5) * from.w * s;
    y = (v - 0.5) * from.h * s;
  }
  return PartPoint(to, x, y, n);
}

struct Pair {
  const char* name;
  double w, h, w2, h2;
};
const Pair kPairs[] = {
    {"1.2x0.9 -> 0.8x0.6 (4:3, smaller)", 1.2, 0.9, 0.8, 0.6},
    {"0.8x0.6 -> 1.2x0.9 (4:3, larger)", 0.8, 0.6, 1.2, 0.9},
    {"16:9 -> 4:3", 1.6, 0.9, 1.2, 0.9},
    {"4:3 -> 16:9", 1.2, 0.9, 1.6, 0.9},
    {"wide 4.0x1.0 -> narrow 1.2x1.0", 4.0, 1.0, 1.2, 1.0},
    {"portrait flip chart 0.7x1.0 -> 3.0x1.2", 0.7, 1.0, 3.0, 1.2},
};
struct Pose {
  const char* name;
  double x, y, yaw, tilt;
};
// The sender's board on the north wall; the receiver's on the same wall, another wall (east, rotated 90 deg), and tilted.
const Pose kSrcPose = {"north wall", 5.0, 9.5, -kPi / 2, 0.0};
const Pose kRcvPoses[] = {
    {"same wall", 3.0, 9.5, -kPi / 2, 0.0},
    {"east wall", 9.5, 4.0, kPi, 0.0},
    {"free-standing, yawed 30 deg", 4.0, 3.0, kPi / 6, 0.0},
    {"leaning back 12 deg", 7.0, 5.0, kPi / 2, 12.0 * kPi / 180},
};
const std::array<std::array<double, 2>, 9> kUv = {{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0.5, 0.5}, {0.25, 0.75}, {0.5, 0}, {1, 0.5}, {0.1, 0.9}}};

BoardCfg With(const BoardCfg& base, double w, double h, const Pose& p, SurfaceChartKind chart) {
  BoardCfg c = base;
  c.w = w;
  c.h = h;
  c.x = p.x;
  c.y = p.y;
  c.yaw = p.yaw;
  c.tilt = p.tilt;
  c.chart = chart;
  return c;
}

// The hand row's position by value (never a pointer into a temporary proposal).
bool HandPos(const ReceiverProposal& p, Eigen::Vector3d* out, bool* engaged = nullptr) {
  const AnchorProposal* h = Rig::Hand(p);
  if (!h) return false;
  *out = Eigen::Vector3d(h->x, h->y, h->z);
  if (engaged) *engaged = h->engaged() && h->status == ProposalStatus::kExact;
  return true;
}

void ExpectHandAt(const ReceiverProposal& p, const Eigen::Vector3d& want, const std::string& what) {
  const AnchorProposal* h = Rig::Hand(p);
  ASSERT_NE(h, nullptr) << what << ": no hand row";
  EXPECT_TRUE(h->engaged()) << what;
  EXPECT_EQ(h->status, ProposalStatus::kExact) << what;
  EXPECT_NEAR(h->x, want.x(), kEps) << what;
  EXPECT_NEAR(h->y, want.y(), kEps) << what;
  EXPECT_NEAR(h->z, want.z(), kEps) << what;
}

TEST(SurfaceRetarget, StretchLandsTheHandAtTheSameUvOnEveryBoardPairWallAndTilt) {
  for (const Pair& pr : kPairs) {
    for (const Pose& rp : kRcvPoses) {
      const BoardCfg a = With({}, pr.w, pr.h, kSrcPose, kStretch);
      const BoardCfg b = With({}, pr.w2, pr.h2, rp, kStretch);
      Rig rig(a, b);
      for (const auto& uv : kUv) {
        const std::string what = std::string(pr.name) + " / " + rp.name + " / uv " + std::to_string(uv[0]) + "," + std::to_string(uv[1]);
        ExpectHandAt(rig.FrameUv(uv[0], uv[1]), Expected(a, b, kStretch, uv[0], uv[1], 0.0), what);
      }
    }
  }
}

TEST(SurfaceRetarget, ContainLetterboxesToTheTighterRatioCentred) {
  for (const Pair& pr : kPairs) {
    for (const Pose& rp : kRcvPoses) {
      const BoardCfg a = With({}, pr.w, pr.h, kSrcPose, kContain);
      const BoardCfg b = With({}, pr.w2, pr.h2, rp, kContain);
      Rig rig(a, b);
      for (size_t k = 0; k < kUv.size(); ++k) {
        const auto& uv = kUv[k];
        const std::string what = std::string(pr.name) + " / " + rp.name;
        const ReceiverProposal p = rig.FrameUv(uv[0], uv[1]);
        ExpectHandAt(p, Expected(a, b, kContain, uv[0], uv[1], 0.0), what);
      }
    }
  }
}

TEST(SurfaceRetarget, ContainImageHasTheSourceAspectRatioInThePartFrame) {
  for (const Pair& pr : kPairs) {
    const BoardCfg a = With({}, pr.w, pr.h, kSrcPose, kContain);
    const BoardCfg b = With({}, pr.w2, pr.h2, kRcvPoses[1], kContain);
    Rig rig(a, b);
    const Eigen::Matrix4d inv = PartToWorld(b).inverse();
    auto local = [&](double u, double v) {
      Eigen::Vector3d w = Eigen::Vector3d::Zero();
      EXPECT_TRUE(HandPos(rig.FrameUv(u, v), &w));
      return (inv * w.homogeneous()).head<3>().eval();
    };
    const Eigen::Vector3d lo = local(0, 0), hi = local(1, 1);
    EXPECT_NEAR((hi.x() - lo.x()) / (hi.y() - lo.y()), pr.w / pr.h, 1e-9) << pr.name;
    EXPECT_LE(hi.x() - lo.x(), pr.w2 + 1e-9) << pr.name;  // fits the receiver's face
    EXPECT_LE(hi.y() - lo.y(), pr.h2 + 1e-9) << pr.name;
    EXPECT_NEAR(0.5 * (hi.x() + lo.x()), 0.0, 1e-9) << pr.name;  // centred
    EXPECT_NEAR(0.5 * (hi.y() + lo.y()), 0.0, 1e-9) << pr.name;
  }
}

TEST(SurfaceRetarget, TheRadialDefaultIsUnchangedAndIsNotUvPreservingAcrossAspectRatios) {
  // 16:9 -> 4:3 on the default chart: the corner (1,1) is NOT the receiver's corner. This is what the frame charts fix.
  const BoardCfg a = With({}, 1.6, 0.9, kSrcPose, kRadial);
  const BoardCfg b = With({}, 1.2, 0.9, kRcvPoses[1], kRadial);
  Rig rig(a, b);
  Eigen::Vector3d got;
  bool engaged = false;
  ASSERT_TRUE(HandPos(rig.FrameUv(1, 1), &got, &engaged));
  EXPECT_TRUE(engaged);
  const Eigen::Vector3d corner = PartPoint(b, 0.6, 0.45, 0.0);
  EXPECT_GT((got - corner).norm(), 0.1);
  // equal aspect: the radial chart IS the uniform scale, (u,v) preserved
  const BoardCfg c = With({}, 1.2, 0.9, kSrcPose, kRadial);
  const BoardCfg d = With({}, 0.8, 0.6, kRcvPoses[1], kRadial);
  Rig rig2(c, d);
  for (const auto& uv : kUv) ExpectHandAt(rig2.FrameUv(uv[0], uv[1]), Expected(c, d, kStretch, uv[0], uv[1], 0.0), "radial 4:3");
}

TEST(SurfaceRetarget, TheNormalOffsetOfAContactBehindThePlaneIsCarriedMetrically) {
  // Engagement is geometric: on or below the plane. A palm pressed 8 mm into the face keeps its 8 mm on the receiver, whatever the
  // scale of the receiver's board.
  for (auto kind : {kStretch, kContain}) {
    const BoardCfg a = With({}, 1.6, 0.9, kSrcPose, kind);
    const BoardCfg b = With({}, 0.8, 0.6, kRcvPoses[3], kind);
    Rig rig(a, b);
    ExpectHandAt(rig.FrameUv(0.3, 0.6, -0.008), Expected(a, b, kind, 0.3, 0.6, -0.008), "normal offset");
  }
}

TEST(SurfaceRetarget, RoundTripAToBToAReturnsTheSourceContact) {
  for (const Pair& pr : kPairs) {
    const BoardCfg a = With({}, pr.w, pr.h, kSrcPose, kStretch);
    const BoardCfg b = With({}, pr.w2, pr.h2, kRcvPoses[2], kStretch);
    Rig ab(a, b);
    Rig ba(b, a);  // B is the sender now; A's room is the receiver
    for (const auto& uv : kUv) {
      Eigen::Vector3d hand_b, back;
      ASSERT_TRUE(HandPos(ab.FrameUv(uv[0], uv[1]), &hand_b)) << pr.name;
      ASSERT_TRUE(HandPos(ba.Frame(hand_b), &back)) << pr.name;
      const Eigen::Vector3d want = PartPoint(a, (uv[0] - 0.5) * a.w, (uv[1] - 0.5) * a.h, 0.0);
      EXPECT_NEAR((back - want).norm(), 0.0, 1e-8) << pr.name << " uv " << uv[0] << "," << uv[1];
    }
  }
  // contain: the round trip is the identity where B's letterbox holds A's whole board (equal aspect)
  const BoardCfg a = With({}, 1.2, 0.9, kSrcPose, kContain);
  const BoardCfg b = With({}, 0.8, 0.6, kRcvPoses[2], kContain);
  Rig ab(a, b), ba(b, a);
  for (const auto& uv : kUv) {
    Eigen::Vector3d hand_b, back;
    ASSERT_TRUE(HandPos(ab.FrameUv(uv[0], uv[1]), &hand_b));
    ASSERT_TRUE(HandPos(ba.Frame(hand_b), &back));
    const Eigen::Vector3d want = PartPoint(a, (uv[0] - 0.5) * a.w, (uv[1] - 0.5) * a.h, 0.0);
    EXPECT_NEAR((back - want).norm(), 0.0, 1e-8);
  }
}

TEST(SurfaceRetarget, TheMappedTargetIsContinuousAcrossTheFaceAndSteppedExactlyByTheChart) {
  // No discontinuity inside the surface scope other than what the chart defines: a sweep of 0.02 steps in u moves the
  // target by exactly the chart's image of the step (the mapped step is constant: W' * 0.02), with no cap, blend or jump.
  const BoardCfg a = With({}, 1.6, 0.9, kSrcPose, kStretch);
  const BoardCfg b = With({}, 0.8, 0.6, kRcvPoses[1], kStretch);
  Rig rig(a, b);
  Eigen::Vector3d prev;
  bool have_prev = false;
  for (int k = 0; k <= 50; ++k) {
    const double u = 0.02 * k;
    Eigen::Vector3d p;
    bool engaged = false;
    ASSERT_TRUE(HandPos(rig.FrameUv(u, 0.4), &p, &engaged)) << k;
    ASSERT_TRUE(engaged) << k;
    if (have_prev) EXPECT_NEAR((p - prev).norm(), 0.02 * b.w, 1e-8) << k;
    prev = p;
    have_prev = true;
  }
}

TEST(SurfaceRetarget, LeavingTheFaceCrossesTheSurfaceScopeBoundaryWithAttributedRelease) {
  // The surface scope is the part's contact: on the face and on or below its plane. Lifted off, or past the edge, the hand has no
  // AffordancePose target (the body keeps its own pose): no blend, the boundary is the engagement predicate, and the proposal
  // carries the SEAT_RELEASE / SEAT_ENGAGE cause bits (the affordance-contact causes) on the crossing ticks.
  const BoardCfg a = With({}, 1.2, 0.9, kSrcPose, kStretch);
  const BoardCfg b = With({}, 0.8, 0.6, kRcvPoses[1], kStretch);
  Rig rig(a, b);
  struct Step { double u, v, n; bool engaged; };
  const Step steps[] = {{0.5, 0.5, 0.2, false}, {0.5, 0.5, 0.0, true}, {0.9, 0.5, 0.0, true}, {1.05, 0.5, 0.0, false},
                        {0.9, 0.5, 0.0, true}, {0.9, 0.5, 0.05, false}};
  bool was = false;
  for (const Step& s : steps) {
    const ReceiverProposal p = rig.FrameUv(s.u, s.v, s.n);
    const AnchorProposal* h = Rig::Hand(p);
    ASSERT_NE(h, nullptr);
    EXPECT_EQ(h->engaged(), s.engaged) << s.u << "," << s.v << "," << s.n;
    if (s.engaged) {
      EXPECT_EQ(h->status, ProposalStatus::kExact);
    } else {
      EXPECT_NE(h->status, ProposalStatus::kExact) << "an unengaged hand has no AffordancePose target";
    }
    if (s.engaged && !was) EXPECT_NE(h->cause & kCauseSeatEngage, 0u) << "engage not attributed";
    if (!s.engaged && was) EXPECT_NE(h->cause & kCauseSeatRelease, 0u) << "release not attributed";
    if (s.engaged == was) EXPECT_EQ(h->cause & (kCauseSeatEngage | kCauseSeatRelease), 0u);
    was = s.engaged;
  }
}

TEST(SurfaceRetarget, TheBodyRootStaysInSpacingScopeWhateverTheChartOfTheFace) {
  // The root's proposal is the Spacing relation's: it does not read the hand's chart. The same scene under three chart kinds gives
  // the identical root proposal.
  std::vector<std::array<double, 3>> roots;
  for (auto kind : {kRadial, kStretch, kContain}) {
    const BoardCfg a = With({}, 1.6, 0.9, kSrcPose, kind);
    const BoardCfg b = With({}, 1.2, 0.9, kRcvPoses[1], kind);
    Rig rig(a, b);
    const ReceiverProposal p = rig.FrameUv(0.8, 0.5);
    roots.push_back({p.x, p.y, p.yaw});
  }
  for (size_t i = 1; i < roots.size(); ++i)
    for (int k = 0; k < 3; ++k) EXPECT_EQ(roots[i][k], roots[0][k]) << "chart " << i << " component " << k;
}

TEST(SurfaceRetarget, MismatchedChartsOrMissingPartsGiveNoTargetNeverAGuess) {
  {  // the sender declares a frame chart, the receiver's matched part the default radial one
    Rig rig(With({}, 1.2, 0.9, kSrcPose, kStretch), With({}, 0.8, 0.6, kRcvPoses[0], kRadial));
    const ReceiverProposal p = rig.FrameUv(0.5, 0.5);
    const AnchorProposal* h = Rig::Hand(p);
    if (h) EXPECT_NE(h->status, ProposalStatus::kExact);
  }
  {  // the receiver's board has no writing part
    BoardCfg b = With({}, 0.8, 0.6, kRcvPoses[0], kStretch);
    b.has_part = false;
    Rig rig(With({}, 1.2, 0.9, kSrcPose, kStretch), b);
    const ReceiverProposal p = rig.FrameUv(0.5, 0.5);
    const AnchorProposal* h = Rig::Hand(p);
    if (h) EXPECT_NE(h->status, ProposalStatus::kExact);
  }
}

TEST(SurfaceRetarget, ADegenerateOrNonRectangularFaceCannotDeclareAFrameChart) {
  auto ctx = LocalContext::Create();
  Space* s = ctx->AddSpace();
  BoardCfg c = With({}, 1.2, 0.9, kSrcPose, kRadial);
  BuildScene(s, c);
  EXPECT_EQ(s->SetElementSubRegionChart(kBoard, "board", kStretch), SurfaceChartStatus::kOk);
  EXPECT_EQ(s->SetElementSubRegionChart(kBoard, "board", kContain), SurfaceChartStatus::kOk);
  EXPECT_EQ(s->SetElementSubRegionChart(kBoard, "board", kRadial), SurfaceChartStatus::kOk);
  EXPECT_EQ(s->SetElementSubRegionChart(kBoard, "nope", kStretch), SurfaceChartStatus::kUnknownPart);
  EXPECT_EQ(s->SetElementSubRegionChart(999, "board", kStretch), SurfaceChartStatus::kUnknownPart);
  // a zero-size face: the registration itself is refused when it asks for a frame chart
  SubRegion zero;
  zero.name = "zero";
  zero.boundary = Rect(0, 0, 0.0, 0.9);
  zero.chart = kStretch;
  EXPECT_FALSE(s->AddElementSubRegion(kBoard, zero));
  // a trapezoid face
  SubRegion trap;
  trap.name = "trap";
  trap.boundary = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{{0, 0}, {1, 0}, {0.8, 1}, {0.2, 1}});
  trap.chart = kContain;
  EXPECT_FALSE(s->AddElementSubRegion(kBoard, trap));
  trap.chart = kRadial;
  EXPECT_TRUE(s->AddElementSubRegion(kBoard, trap));
  EXPECT_EQ(s->SetElementSubRegionChart(kBoard, "trap", kStretch), SurfaceChartStatus::kNotARectangle);
  // the trapezoid board in the receiver: a frame-chart sender gets no target there
  BoardCfg tb = With({}, 0.8, 0.6, kRcvPoses[0], kStretch);
  tb.rectangular = false;
  Rig rig(With({}, 1.2, 0.9, kSrcPose, kStretch), tb);
  const ReceiverProposal p = rig.FrameUv(0.5, 0.5);
  const AnchorProposal* h = Rig::Hand(p);
  if (h) EXPECT_NE(h->status, ProposalStatus::kExact);
}

TEST(SurfaceRetarget, ThePartFrameReadsBackTheSameMappingTheRelationPathUses) {
  // The stroke path (sb_part_chart_*) reads a world point on a part through Space::GetElementPartFrame. A world point of A's face,
  // encoded on A and decoded on B, is the hand target the relation path proposes for the same point.
  const BoardCfg a = With({}, 1.6, 0.9, kSrcPose, kStretch);
  const BoardCfg b = With({}, 0.8, 0.6, kRcvPoses[3], kStretch);
  Rig rig(a, b);
  Space::PartFrame fa, fb;
  ASSERT_TRUE(rig.src->GetElementPartFrame(kBoard, "board", &fa));
  ASSERT_TRUE(rig.rcv->GetElementPartFrame(kBoard, "board", &fb));
  EXPECT_EQ(fa.chart, kStretch);
  for (const auto& uv : kUv) {
    const Eigen::Vector3d world = PartPoint(a, (uv[0] - 0.5) * a.w, (uv[1] - 0.5) * a.h, 0.0);
    const Eigen::Vector4d local = fa.part_to_world.inverse() * world.homogeneous();
    auto coord = PlanarFrameChart::Encode(local.head<2>(), local.z(), *fa.boundary);
    ASSERT_TRUE(coord.ok());
    EXPECT_NEAR(coord->u, uv[0], 1e-12);
    EXPECT_NEAR(coord->v, uv[1], 1e-12);
    auto onb = PlanarFrameChart::Decode(*coord, *fb.boundary, kStretch);
    ASSERT_TRUE(onb.ok());
    const Eigen::Vector3d stroke_world = (fb.part_to_world * onb->homogeneous()).head<3>();
    Eigen::Vector3d hand;
    ASSERT_TRUE(HandPos(rig.Frame(world), &hand));
    EXPECT_NEAR((stroke_world - hand).norm(), 0.0, 1e-9) << uv[0] << "," << uv[1];
  }
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
