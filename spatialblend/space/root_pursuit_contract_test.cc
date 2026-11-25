// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// The root's live contract: core proposes the PURSUED target every update, for
// every anchor (the ratio continuation: each relation closes 1/n of the
// receiver's own distance), and the driver realizes it within its kinematic
// bounds. A placed answer is kept for an explicit destination only (the first
// proposal, a source exception).
//   1. No request publishes no new target/revision. The
//   mailbox retains the requested target and the
//      driver finishes it; independent position/yaw events carry new requests.
//      POSITION_UNSPECIFIED is never revived.
//   2. Source and receiver arrive together: the ratio continuation reaches the
//   receiver's mapped contact on the update
//      the source reaches its own (n = 1), whatever the size of the two rooms.
#include <cstdio>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/model/realization/remote_update.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace {

constexpr double kDt = 1.0 / 30.0;
constexpr double kWalk = 1.4;

struct Box {
  Eigen::Vector2d lo, hi;
  double Distance(const Eigen::Vector2d& p) const {
    const double dx = std::max({lo.x() - p.x(), 0.0, p.x() - hi.x()});
    const double dy = std::max({lo.y() - p.y(), 0.0, p.y() - hi.y()});
    return std::hypot(dx, dy);
  }
};

void AddRoom(Space* space, double w, double h) {
  auto* tree = space->GetElementTree();
  ASSERT_TRUE(tree->AddElement(1).ok());
  tree->GetElement(1)->SetSharedId(100);
  tree->GetElement(1)->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(
      std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{{0, 0}, {w, 0}, {w, h}, {0, h}})));
}

void AddBox(Space* space, element_id_t id, shared_element_id_t shared, const Box& box) {
  auto* tree = space->GetElementTree();
  ASSERT_TRUE(tree->AddElement(id, 1).ok());
  tree->GetElement(id)->SetSharedId(shared);
  const Eigen::Vector2d c = 0.5 * (box.lo + box.hi), half = 0.5 * (box.hi - box.lo);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = c.x();
  t(1, 3) = c.y();
  tree->GetElement(id)->SetLocalTransform(t);
  tree->GetElement(id)->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-half.x(), -half.y(), 0.0), Eigen::Vector3d(half.x(), half.y(), 0.8), GeometryRole::kObstacle));
}

struct Rooms {
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space* src = ctx->AddSpace();
  Space* rcv = ctx->AddSpace();
  spatialblend::entity_id_t actor = 0, avatar = 0;
  std::unique_ptr<IdealDriver> follower;
  Box src_table, rcv_table;
  // Source room w x h with a table; the receiver room (rw x rh) holds the same table scaled with the room.
  Rooms(double w, double h, double rw, double rh, const Box& table, const Eigen::Vector2d& rcv_start) {
    AddRoom(src, w, h);
    AddRoom(rcv, rw, rh);
    const Eigen::Vector2d s(rw / w, rh / h);
    src_table = table;
    rcv_table = {table.lo.cwiseProduct(s), table.hi.cwiseProduct(s)};
    AddBox(src, 3, 300, src_table);
    AddBox(rcv, 3, 300, rcv_table);
    const std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))};
    actor = src->CreateActor(anchors);
    src->GetActor(actor)->SetElementId(1);
    avatar = rcv->CreateActor(anchors);
    rcv->GetActor(avatar)->SetElementId(1);
    src->Bind(actor, "root", RelationKernelType::kAvatarSpacing);
    src->SetHub(ctx->GetHub());
    rcv->SetHub(ctx->GetHub());
    src->SetMovementGate(0.0, 0.0);
    follower = std::make_unique<IdealDriver>(rcv, avatar, rcv_start, 0.0);
  }
  ReceiverProposal Step(const Eigen::Vector2d& pos, double yaw = 0.0) {
    src->UpdateActorState(actor, pos, yaw);
    ReceiverProposal p;
    EXPECT_TRUE(rcv->ReadProposal(avatar, &p));
    return p;
  }
};

std::vector<Eigen::Vector2d> Walk(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
  std::vector<Eigen::Vector2d> out;
  const int n = std::max(1, static_cast<int>(std::ceil((b - a).norm() / (kWalk * kDt))));
  for (int i = 0; i <= n; ++i) out.push_back(a + (b - a) * (double(i) / n));
  return out;
}

bool IsTarget(const ReceiverProposal& p) {
  return p.status == ProposalStatus::kExact || p.status == ProposalStatus::kInexact;
}

// No request publishes nothing (there is no sampled-position hold). Diagnostics may still report NO_REQUEST, but the mailbox retains its
// last requested target and revision.
TEST(RootPursuitContract, NoRequestRetainsTheLastPublishedPosition) {
  Rooms r(10, 10, 12, 8, {{6.0, 4.0}, {7.0, 5.0}}, {2.4, 3.6});
  std::vector<std::pair<Eigen::Vector2d, double>> track;  // (pose, yaw)
  for (const auto& q : Walk({2, 4.5}, {4.0, 4.5})) track.push_back({q, 0.0});
  for (int k = 0; k < 20; ++k) track.push_back({{4.0, 4.5}, 0.0});            // a pause
  for (int k = 0; k < 10; ++k) track.push_back({{4.0, 4.5}, 0.1 * (k + 1)});  // turning on the spot
  for (const auto& q : Walk({4.0, 4.5}, {5.5, 4.5})) track.push_back({q, 1.0});
  for (int k = 0; k < 20; ++k) track.push_back({{5.5, 4.5}, 1.0});
  uint64_t previous_revision = 0;
  int held = 0, pursued = 0;
  for (const auto& [pos, yaw] : track) {
    const Eigen::Vector2d body = r.follower->position();  // the sample this update is solved from
    const ReceiverProposal p = r.Step(pos, yaw);
    ASSERT_TRUE(IsTarget(p));
    EXPECT_TRUE(p.position_specified()) << "every update carries a position";
    const Eigen::Vector2d at(p.x, p.y);
    if (p.revision == previous_revision) {
      ++held;
      EXPECT_EQ(at, body)
          << "the ideal driver already reached the retained target";
    } else if (p.cause & kCauseProgress) {
      ++pursued;
    }
    previous_revision = p.revision;
  }
  EXPECT_GE(held, 40);
  EXPECT_GE(pursued, 20);
}

// No request publishes no new proposal.
// Unlike the zero-lag follower above, this driver has not reached the requested
// target when the source stops. A sampled-position hold would cancel it.
TEST(RootPursuitContract, LaggingDriverFinishesItsLastTargetAfterSourceStops) {
  Rooms r(10, 10, 10, 10, {{6.0, 4.0}, {7.0, 5.0}}, {2.0, 4.5});
  ASSERT_TRUE(
      IsTarget(r.Step({2.0, 4.5})));  // the join is still an explicit placement
  Eigen::Vector2d body = r.follower->position();
  r.follower.reset();  // subsequent requests are realized at walking speed, not
                       // immediately
  int publications = 0;
  const auto observer = r.rcv->AddProposalObserver(
      [&](spatialblend::entity_id_t actor, const ReceiverProposal&) {
        if (actor == r.avatar) ++publications;
      });
  const ReceiverProposal requested = r.Step({3.0, 4.5});
  ASSERT_TRUE(IsTarget(requested));
  ASSERT_NE(requested.cause & kCauseProgress, 0u);
  const Eigen::Vector2d destination(requested.x, requested.y);
  ASSERT_GT((destination - body).norm(), 0.1);
  Eigen::Vector2d target = destination;
  uint64_t consumed_revision = requested.revision;
  ReceiverProposal latest;
  for (int frame = 0; frame < 60; ++frame) {
    const Eigen::Vector2d delta = target - body;
    if (delta.norm() > 0.0)
      body += delta * std::min(1.0, kWalk * kDt / delta.norm());
    EmbodiedPoseInput sample;
    sample.x = body.x();
    sample.y = body.y();
    sample.epoch = 1;
    ASSERT_EQ(r.rcv->PublishEmbodiedPose(r.avatar, sample, (frame + 1) * kDt,
                                         NextPortSequence(r.rcv, r.avatar)),
              EmbodiedPublishResult::kOk);
    latest = r.Step({3.0, 4.5});
    if (latest.revision != consumed_revision) {
      consumed_revision = latest.revision;
      if (IsTarget(latest)) target = Eigen::Vector2d(latest.x, latest.y);
    }
  }
  r.rcv->RemoveProposalObserver(observer);
  EXPECT_EQ(publications, 1) << "only the moving source issued a new request";
  EXPECT_EQ(latest.revision, requested.revision);
  EXPECT_EQ(Eigen::Vector2d(latest.x, latest.y), destination)
      << "the last requested target remains published";
  EXPECT_LT((body - destination).norm(), kContactEpsilonM)
      << "the bounded driver finishes its pending walk";
  std::fprintf(stderr,
               "[Lagging] requested revision %llu, latest %llu, publications "
               "%d, remaining %.6f m\n",
               static_cast<unsigned long long>(requested.revision),
               static_cast<unsigned long long>(latest.revision), publications,
               (body - destination).norm());
}

// The ratio continuation closes 1/n of the receiver's own distance, and n = 1 at the source's contact: the receiver
// reaches its own table on the source's update, in rooms of other sizes and aspect ratios.
TEST(RootPursuitContract, SourceAndReceiverArriveTogetherAcrossRoomSizes) {
  struct Case {
    double rw, rh;
  };
  for (const Case c : {Case{10, 10}, Case{14, 8}, Case{7, 12}, Case{16, 16}}) {
    Rooms r(10, 10, c.rw, c.rh, {{6.0, 4.0}, {7.0, 5.0}}, {2.0 * c.rw / 10, 4.5 * c.rh / 10});
    int src_on = -1, rcv_on = -1, frame = 0;
    for (const auto& q : Walk({2, 4.5}, {6.0, 4.5})) {
      const ReceiverProposal p = r.Step(q);
      ASSERT_TRUE(IsTarget(p));
      if (src_on < 0 && r.src_table.Distance(q) < kContactEpsilonM) src_on = frame;
      if (rcv_on < 0 && r.rcv_table.Distance(Eigen::Vector2d(p.x, p.y)) < kContactEpsilonM) rcv_on = frame;
      ++frame;
    }
    ASSERT_GE(src_on, 0);
    EXPECT_EQ(rcv_on, src_on) << c.rw << " x " << c.rh << ": the receiver arrives on the source's update";
    std::fprintf(stderr, "[Arrive] %gx%g: source on frame %d, receiver on frame %d\n", c.rw, c.rh, src_on, rcv_on);
  }
}

// In IDENTICAL rooms the receiver's proposal is the source's pose (receiver = source)
// where the measurement makes it so: a source approaching its table straight on (its nearest contact fixed), the
// receiver starting where the source starts. Measured, not tuned: the worst deviation over the walk is printed.
TEST(RootPursuitContract, IdenticalRoomsStraightApproachIsTheSourcePose) {
  Rooms r(10, 10, 10, 10, {{6.0, 4.0}, {7.0, 5.0}}, {2.0, 4.5});
  double worst = 0.0;
  int compared = 0;
  for (const auto& q : Walk({2, 4.5}, {6.0, 4.5})) {
    const ReceiverProposal p = r.Step(q);
    ASSERT_TRUE(IsTarget(p));
    worst = std::max(worst, (Eigen::Vector2d(p.x, p.y) - q).norm());
    ++compared;
  }
  std::fprintf(stderr, "[Identity] straight approach, identical rooms: %d updates, worst |proposal - source| %.3e m\n",
               compared, worst);
  EXPECT_GT(compared, 50);
  EXPECT_LT(worst, kContactEpsilonM);
}

}  // namespace
}  // namespace spatialblend
