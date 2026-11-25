// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Approach and support hand off continuously. Before
// contact, Spacing carries the root to the seat's support; at touch the
// AffordancePose pelvis proposal takes over. This test drives one source through
// the whole sequence (walk to the seat, lower the pelvis onto it, stay seated)
// with the same receiver and the same sampled inputs, and checks that the proposal
// streams of the root and of the pelvis have no unattributed discontinuity at
// engagement: on steps that carry no cause the limits hold (v*dt + 2 cm,
// and a 1.5x speed ratio), and the pelvis proposal that takes over starts where
// the root proposal was.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/embodiment/embodiment.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace {

constexpr element_id_t kRoom = 1, kSeat = 10;
constexpr shared_element_id_t kRoomShared = 100, kSeatShared = 1000;
constexpr double kHz = 60.0, kSpeed = 1.2, kStandPelvis = 0.9, kSeatTop = 0.45;

Eigen::Matrix4d At(double x, double y) {
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = x;
  t(1, 3) = y;
  return t;
}

std::shared_ptr<Polygon2D> Rect(double x0, double y0, double x1, double y1) {
  return std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});
}

struct SeatSpec {
  double x, y, hx, hy;
  bool surface = false;  // a Surface-role top (a seat with no object volume)
  // The seat carries `stand` and `seated` pelvis affordance poses, and both actors register an
  // embodiment with the pelvis's standing height, so the pelvis has an inner proposal inside the seat's SD.
  bool poses = false;
};
constexpr element_id_t kStandPose = 11, kSeatedPose = 12;

void BuildScene(Space* s, const SeatSpec& seat_spec) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kRoom).ok());
  tree->GetElement(kRoom)->SetSharedId(kRoomShared);
  tree->GetElement(kRoom)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 10, 10)));
  ASSERT_TRUE(tree->AddElement(kSeat, kRoom).ok());
  tree->GetElement(kSeat)->SetSharedId(kSeatShared);
  if (seat_spec.surface) {
    const double x = seat_spec.x, y = seat_spec.y, hx = seat_spec.hx, hy = seat_spec.hy;
    tree->GetElement(kSeat)->SetSpatialDefinition(
        std::make_shared<SurfaceSpatialDefinition>(std::make_shared<Polygon3D>(
            std::vector<Eigen::Vector3d>{{x - hx, y - hy, kSeatTop},
                                         {x + hx, y - hy, kSeatTop},
                                         {x + hx, y + hy, kSeatTop},
                                         {x - hx, y + hy, kSeatTop}})));
    tree->GetElement(kSeat)->SetTag("Sittable");
    return;
  }
  tree->GetElement(kSeat)->SetLocalTransform(At(seat_spec.x, seat_spec.y));
  auto seat = std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-seat_spec.hx, -seat_spec.hy, 0.0),
      Eigen::Vector3d(seat_spec.hx, seat_spec.hy, kSeatTop),
      GeometryRole::kObstacle);
  SubRegion top;
  top.name = "seat";
  top.tag = "Sittable";
  top.local_transform = Eigen::Matrix4d::Identity();
  top.local_transform(2, 3) = kSeatTop;
  top.boundary = Rect(-seat_spec.hx, -seat_spec.hy, seat_spec.hx, seat_spec.hy);
  seat->AddSubRegion(top);
  tree->GetElement(kSeat)->SetSpatialDefinition(seat);
  tree->GetElement(kSeat)->SetTag("Sittable");
  if (!seat_spec.poses) return;
  const auto pose = [&](element_id_t id, const char* tag, double y0, double y1, AffordancePoseTemplate t) {
    ASSERT_TRUE(tree->AddElement(id, kSeat).ok());
    tree->GetElement(id)->SetTag(tag);
    tree->GetElement(id)->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
        Eigen::Vector3d(-seat_spec.hx, y0, 0.0), Eigen::Vector3d(seat_spec.hx, y1, kSeatTop),
        GeometryRole::kInteractionRegion));
    tree->GetElement(id)->SetAffordancePose(t);
  };
  pose(kStandPose, "stand", -seat_spec.hy, 0.0, {"pelvis", AffordancePoseTemplate::Support::kActorRoot, ""});
  pose(kSeatedPose, "seated", 0.0, seat_spec.hy, {"pelvis", AffordancePoseTemplate::Support::kPart, "seat"});
}

Embodiment StandingPelvis() {
  Embodiment e;
  e.id = "continuity";
  e.joints = {{"hips", -1}};
  EmbodimentAnchor pelvis;
  pelvis.name = "pelvis";
  pelvis.families = kFamilyAffordancePose;
  pelvis.has_standing = true;
  pelvis.standing_height_m = kStandPelvis;
  pelvis.standing_provenance_json = R"({"procedure":"fixture","source":"seat_approach_continuity_test"})";
  e.anchors = {pelvis};
  return e;
}

// Causes that are an event of their own. Progress / NoRequest / ChartSnap /
// FirstProposal describe the continuation itself and do not excuse a step.
constexpr uint32_t kEventCauses =
    ~(kCauseFirstProposal | kCauseProgress | kCauseNoRequest | kCauseChartSnap);

struct Frame {
  Eigen::Vector2d source;   // source root
  bool root_has = false;
  Eigen::Vector2d root;     // root proposal xy
  uint32_t root_cause = 0;
  bool root_unspecified = false;
  bool pelvis_has = false;  // pelvis proposal carries a target
  Eigen::Vector3d pelvis;
  uint32_t pelvis_cause = 0;
  bool pelvis_engaged = false;
  bool pelvis_inner = false;
};

std::vector<Frame> Run(const SeatSpec& src_seat, const SeatSpec& rcv_seat) {
  auto ctx = LocalContext::Create();
  Space* src = ctx->AddSpace();
  Space* rcv = ctx->AddSpace();
  BuildScene(src, src_seat);
  BuildScene(rcv, rcv_seat);
  const std::vector<SubjectAnchor> anchors = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
      SubjectAnchor("pelvis", Eigen::Vector3d(0, 0, kStandPelvis))};
  const auto sa = src->CreateActor(anchors);
  src->GetActor(sa)->SetElementId(kRoom);
  const auto ra = rcv->CreateActor(anchors);
  rcv->GetActor(ra)->SetElementId(kRoom);
  if (src_seat.poses) EXPECT_EQ(src->RegisterEmbodiment(sa, StandingPelvis()), Space::RegisterEmbodimentResult::kOk);
  if (rcv_seat.poses) EXPECT_EQ(rcv->RegisterEmbodiment(ra, StandingPelvis()), Space::RegisterEmbodimentResult::kOk);
  src->SetHub(ctx->GetHub());
  rcv->SetHub(ctx->GetHub());
  src->SetMovementGate(0.0, 0.0);
  rcv->SetMovementGate(0.0, 0.0);
  src->SetUpdateInterval(1.0 / kHz);
  rcv->SetUpdateInterval(1.0 / kHz);
  src->Bind(sa, "root", RelationKernelType::kAvatarSpacing);
  EXPECT_NE(src->BindScoped(sa, "pelvis", RelationKernelType::kAvatarAffordancePose,
                            kSeat, "Sittable"),
            nullptr);
  uint32_t pelvis_id = 0;
  EXPECT_TRUE(rcv->AnchorIdFor(ra, "pelvis", &pelvis_id));
  // The receiver's body starts away from the seat; its root is followed by the
  // zero-lag driver, its pelvis is published below from the same proposal.
  IdealDriver root_driver(rcv, ra, {8.0, 8.0}, 0.0);
  uint64_t seq = 0;
  auto publish_pelvis = [&](const Eigen::Vector3d& p, double yaw) {
    EmbodiedPoseInput in;
    in.x = p.x();
    in.y = p.y();
    in.z = p.z();
    in.yaw = yaw;
    in.epoch = 1;
    rcv->PublishAnchorPose(ra, pelvis_id, in, 0.0, ++seq);
  };
  publish_pelvis({8.0, 8.0, kStandPelvis}, 0.0);

  // The source: walk toward the seat centre from the north, then lower the pelvis.
  const Eigen::Vector2d seat_at(src_seat.x + 0.1, src_seat.y + 0.05);
  const Eigen::Vector2d start(src_seat.x + 0.1, src_seat.y + 3.5);
  const double dt = 1.0 / kHz;
  std::vector<Eigen::Vector3d> path;  // (x, y, pelvis z)
  for (Eigen::Vector2d p = start; (p - seat_at).norm() > 1e-9;) {
    const Eigen::Vector2d to = seat_at - p;
    p += to.normalized() * std::min(kSpeed * dt, to.norm());
    path.emplace_back(p.x(), p.y(), kStandPelvis);
  }
  for (double z = kStandPelvis; z > kSeatTop;) {
    z = std::max(kSeatTop, z - 0.5 * dt);
    path.emplace_back(seat_at.x(), seat_at.y(), z);
  }
  for (int k = 0; k < 40; ++k) path.emplace_back(seat_at.x(), seat_at.y(), kSeatTop);

  std::vector<Frame> frames;
  for (const Eigen::Vector3d& s : path) {
    src->SetAnchorWorldPose(sa, "pelvis", s, 0.0);
    src->UpdateActorState(sa, Eigen::Vector2d(s.x(), s.y()), 0.0);
    ReceiverProposal p;
    if (!rcv->ReadProposal(ra, &p)) continue;
    Frame f;
    f.source = Eigen::Vector2d(s.x(), s.y());
    f.root_has = (p.status == ProposalStatus::kExact ||
                  p.status == ProposalStatus::kInexact) &&
                 p.position_specified();
    f.root = Eigen::Vector2d(p.x, p.y);
    f.root_cause = p.cause;
    f.root_unspecified = !p.position_specified();
    for (const auto& a : p.anchors)
      if (a.anchor_id == pelvis_id) {
        f.pelvis_has = a.status == ProposalStatus::kExact ||
                       a.status == ProposalStatus::kInexact;
        f.pelvis = Eigen::Vector3d(a.x, a.y, a.z);
        f.pelvis_cause = a.cause;
        f.pelvis_engaged = a.engaged();
        f.pelvis_inner = a.inner_solve();
        // The driver realizes the pelvis proposal; otherwise the pelvis follows
        // the root body at standing height (the same sampled inputs every run).
        if (f.pelvis_has)
          publish_pelvis(f.pelvis, a.yaw);
        else
          publish_pelvis({root_driver.position().x(), root_driver.position().y(),
                          kStandPelvis},
                         0.0);
      }
    frames.push_back(f);
  }
  return frames;
}

void ExpectContinuous(const char* name, const SeatSpec& src_seat,
                      const SeatSpec& rcv_seat) {
  const std::vector<Frame> frames = Run(src_seat, rcv_seat);
  ASSERT_GT(frames.size(), 100u) << name;
  const double vdt = kSpeed / kHz;
  const double bound = vdt + 0.02;
  size_t engage = frames.size();
  for (size_t i = 0; i < frames.size(); ++i)
    if (frames[i].pelvis_has && frames[i].pelvis_engaged) {
      engage = i;
      break;
    }
  ASSERT_LT(engage, frames.size()) << name << ": the pelvis never engaged";
  ASSERT_GT(engage, 5u) << name << ": engagement must follow an approach";

  double worst_root = 0.0, worst_pelvis = 0.0, worst_ratio = 0.0;
  int root_steps = 0;
  for (size_t i = 1; i < frames.size(); ++i) {
    const Frame &a = frames[i - 1], &b = frames[i];
    if (a.root_has && b.root_has && (b.root_cause & kEventCauses) == 0) {
      const double step = (b.root - a.root).norm();
      worst_root = std::max(worst_root, step);
      ++root_steps;
      const double src_step = (b.source - a.source).norm();
      if (src_step > 1e-9) worst_ratio = std::max(worst_ratio, step / src_step);
      EXPECT_LE(step, bound) << name << ": cause-less root step at frame " << i;
    }
    if (a.pelvis_has && b.pelvis_has && (b.pelvis_cause & kEventCauses) == 0) {
      const double step = (b.pelvis - a.pelvis).norm();
      worst_pelvis = std::max(worst_pelvis, step);
      EXPECT_LE(step, bound) << name << ": cause-less pelvis step at frame " << i << " " << a.pelvis.transpose()
                             << " -> " << b.pelvis.transpose();
    }
  }
  EXPECT_LE(worst_ratio, 1.5) << name << ": cause-less root speed ratio";
  EXPECT_GT(root_steps, 50) << name << ": the root stream must be graded";

  // The hand-off: the pelvis proposal that takes over starts where the root
  // proposal stood in the frame before. The step is attributed (SEAT_ENGAGE) but
  // it must not be a jump of the support itself.
  const Frame& before = frames[engage - 1];
  const Frame& at = frames[engage];
  ASSERT_TRUE(before.root_has) << name << ": the root proposal before touch";
  EXPECT_TRUE(at.pelvis_cause & kCauseSeatEngage) << name;
  const double handoff = (at.pelvis.head<2>() - before.root).norm();
  EXPECT_LE(handoff, bound) << name << ": root " << before.root.transpose()
                            << " -> pelvis " << at.pelvis.head<2>().transpose();
  if (src_seat.poses && rcv_seat.poses) {
    // The pelvis is in its SD before it touches: the inner proposal leads into the engaged contact, and the last
    // inner proposal is on the receiver's seat, one source step from the contact.
    ASSERT_TRUE(before.pelvis_inner) << name;
    EXPECT_LE((at.pelvis - before.pelvis).norm(), bound) << name;
    EXPECT_NEAR(at.pelvis.z(), kSeatTop, 1e-9) << name;
    for (size_t i = 1; i < frames.size(); ++i) EXPECT_EQ(frames[i].pelvis_cause & kCauseFold, 0u) << name << " " << i;
  }
  std::fprintf(stderr,
               "[R1] %-10s engage frame %zu/%zu worst cause-less step root %.4f pelvis %.4f "
               "(bound %.4f) ratio %.3f hand-off %.4f\n",
               name, engage, frames.size(), worst_root, worst_pelvis, bound,
               worst_ratio, handoff);
}

TEST(SeatApproachContinuity, IdenticalSeat) {
  const SeatSpec s{3.0, 3.0, 0.5, 0.4};
  ExpectContinuous("identical", s, s);
}

TEST(SeatApproachContinuity, LargerSeatElsewhere) {
  ExpectContinuous("larger", SeatSpec{3.0, 3.0, 0.5, 0.4}, SeatSpec{6.5, 7.0, 0.7, 0.56});
}

TEST(SeatApproachContinuity, SmallerSeatElsewhere) {
  ExpectContinuous("smaller", SeatSpec{3.0, 3.0, 0.5, 0.4}, SeatSpec{7.0, 2.0, 0.35, 0.3});
}

TEST(SeatApproachContinuity, SurfaceSeatIdentical) {
  const SeatSpec s{3.0, 3.0, 0.5, 0.4, true};
  ExpectContinuous("surface", s, s);
}

TEST(SeatApproachContinuity, SurfaceSeatElsewhere) {
  ExpectContinuous("surface-far", SeatSpec{3.0, 3.0, 0.5, 0.4, true},
                   SeatSpec{6.5, 7.0, 0.7, 0.56, true});
}

// The same walk-and-sit with the seat's affordance poses: the pelvis has an inner proposal
// inside the seat's SD, and the continuity limits still hold on every cause-less step of both streams.
TEST(SeatApproachContinuity, AffordancePosesIdenticalSeat) {
  SeatSpec s{3.0, 3.0, 0.5, 0.4};
  s.poses = true;
  ExpectContinuous("poses", s, s);
}

TEST(SeatApproachContinuity, AffordancePosesLargerSeatElsewhere) {
  SeatSpec a{3.0, 3.0, 0.5, 0.4}, b{6.5, 7.0, 0.7, 0.56};
  a.poses = b.poses = true;
  ExpectContinuous("poses-far", a, b);
}

}  // namespace
}  // namespace spatialblend

