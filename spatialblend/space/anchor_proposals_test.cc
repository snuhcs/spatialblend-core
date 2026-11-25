// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/anchor_proposals_test.cc
//
// Layer tests for ABI 4:
//   * the per-interaction latency timer: start, latch, arrive, cancel, expire to a
//     request (L1-L4);
//   * source-app exceptions: propagation with an opaque payload, forced through the
//     movement gate, the join, the absolute solve;
//   * one pose proposal per relation-bearing anchor: the pelvis carries the
//     seat pose, the root's yaw is not overridden, a Facing anchor without a sample
//     is NO_SAMPLE.
//
// Every expected value is an independent oracle computed in this file.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/solver/affordance_pose_sd.h"
#include "spatialblend/solver/engagement.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/interaction_latency.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/solver/relation_evaluator.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace test {
namespace {

// ---------------------------------------------------------------------------
// InteractionLatencyTracker: the timer rules alone (L1-L4).
// ---------------------------------------------------------------------------
InteractionObservation Contact(int64_t source, uint32_t anchor, int64_t referent,
                               InteractionKind kind, bool arrived = false) {
  InteractionObservation o;
  o.source_shared_id = source;
  o.anchor_id = anchor;
  o.referent_shared_id = referent;
  o.kind = kind;
  o.arrived = arrived;
  return o;
}

InteractionLatencyPolicy Policy(double spacing_s, double pose_s) {
  InteractionLatencyPolicy p;
  p.max_latency_s[static_cast<size_t>(InteractionKind::kSpacingContact)] = spacing_s;
  p.max_latency_s[static_cast<size_t>(InteractionKind::kAffordancePoseEngagement)] =
      pose_s;
  return p;
}

TEST(InteractionLatency, ThePolicyTableShipsV1sTimeoutForBothKinds) {
  // L4: v1's interaction timeout default, 5000 ms, for both kinds.
  const InteractionLatencyPolicy shipped = DefaultInteractionLatencyPolicy();
  ASSERT_TRUE(shipped.For(InteractionKind::kSpacingContact).has_value());
  ASSERT_TRUE(shipped.For(InteractionKind::kAffordancePoseEngagement).has_value());
  EXPECT_DOUBLE_EQ(*shipped.For(InteractionKind::kSpacingContact), 5.0);
  EXPECT_DOUBLE_EQ(*shipped.For(InteractionKind::kAffordancePoseEngagement), 5.0);
  for (const InteractionKind kind :
       {InteractionKind::kSpacingContact, InteractionKind::kAffordancePoseEngagement}) {
    InteractionLatencyTracker tracker;
    const auto o = Contact(7, 0, 42, kind);
    EXPECT_TRUE(tracker.Update(10.0, {o}, shipped).empty());
    EXPECT_TRUE(tracker.Update(14.999, {o}, shipped).empty()) << "not before 5 s";
    EXPECT_EQ(tracker.Update(15.0, {o}, shipped).size(), 1u) << "at 5 s";
    EXPECT_TRUE(tracker.Update(30.0, {o}, shipped).empty()) << "once";
  }
}

TEST(InteractionLatency, AnEmptyEntryMeansNoLimitAndNeverExpires) {
  const InteractionLatencyPolicy none;  // no entry has a value
  InteractionLatencyTracker tracker;
  const auto o = Contact(7, 0, 42, InteractionKind::kSpacingContact);
  for (int k = 0; k < 1000; ++k)
    EXPECT_TRUE(tracker.Update(k * 1.0, {o}, none).empty());
  EXPECT_EQ(tracker.size(), 1u) << "the timer is still tracked";
}

TEST(InteractionLatency, StartsAtEngagementAndIsLatchedOnce) {
  InteractionLatencyTracker tracker;
  const auto o = Contact(7, 0, 42, InteractionKind::kSpacingContact);
  EXPECT_FALSE(tracker.StartedAt(o).has_value());
  tracker.Update(10.0, {o}, Policy(5.0, 5.0));
  ASSERT_TRUE(tracker.StartedAt(o).has_value());
  EXPECT_DOUBLE_EQ(*tracker.StartedAt(o), 10.0);
  // Further updates with the engagement still holding never restart it.
  tracker.Update(11.0, {o}, Policy(5.0, 5.0));
  tracker.Update(12.5, {o}, Policy(5.0, 5.0));
  EXPECT_DOUBLE_EQ(*tracker.StartedAt(o), 10.0);
}

TEST(InteractionLatency, ExpiresOnceAtTheLimitAndNeverBefore) {
  InteractionLatencyTracker tracker;
  const auto o = Contact(7, 0, 42, InteractionKind::kSpacingContact);
  const auto policy = Policy(2.0, 99.0);
  EXPECT_TRUE(tracker.Update(100.0, {o}, policy).empty());
  EXPECT_TRUE(tracker.Update(101.999, {o}, policy).empty());
  const auto expired = tracker.Update(102.0, {o}, policy);
  ASSERT_EQ(expired.size(), 1u);
  EXPECT_DOUBLE_EQ(expired[0].started_s, 100.0);
  EXPECT_DOUBLE_EQ(expired[0].now_s, 102.0);
  EXPECT_EQ(expired[0].observation.referent_shared_id, 42);
  // At most one request per engagement.
  EXPECT_TRUE(tracker.Update(103.0, {o}, policy).empty());
  EXPECT_TRUE(tracker.Update(500.0, {o}, policy).empty());
}

TEST(InteractionLatency, ArrivalStopsTheTimer) {
  InteractionLatencyTracker tracker;
  auto o = Contact(7, 0, 42, InteractionKind::kSpacingContact);
  const auto policy = Policy(2.0, 2.0);
  tracker.Update(0.0, {o}, policy);
  o.arrived = true;  // the target's SAMPLED anchor has arrived (L2)
  EXPECT_TRUE(tracker.Update(1.0, {o}, policy).empty());
  EXPECT_TRUE(tracker.Arrived(o));
  // Arrived is latched for the engagement: a later wobble does not expire it.
  o.arrived = false;
  EXPECT_TRUE(tracker.Update(50.0, {o}, policy).empty());
}

TEST(InteractionLatency, TheSourceLeavingFirstCancelsTheRequest) {
  InteractionLatencyTracker tracker;
  const auto o = Contact(7, 0, 42, InteractionKind::kAffordancePoseEngagement);
  const auto policy = Policy(2.0, 2.0);
  tracker.Update(0.0, {o}, policy);
  EXPECT_TRUE(tracker.Update(1.0, {}, policy).empty()) << "the source let go (L3)";
  EXPECT_EQ(tracker.size(), 0u);
  // Back later: a NEW engagement, a new timer, measured from its own start.
  EXPECT_TRUE(tracker.Update(10.0, {o}, policy).empty());
  EXPECT_DOUBLE_EQ(*tracker.StartedAt(o), 10.0);
  EXPECT_TRUE(tracker.Update(11.9, {o}, policy).empty());
  EXPECT_EQ(tracker.Update(12.0, {o}, policy).size(), 1u);
}

TEST(InteractionLatency, OnlyTheTargetsThatHaveNotArrivedAreAsked) {
  // Two engagements in one solve (two anchors): one arrived, one did not.
  InteractionLatencyTracker tracker;
  const auto root = Contact(7, 0, 42, InteractionKind::kSpacingContact, /*arrived=*/true);
  const auto pelvis = Contact(7, 3, 43, InteractionKind::kAffordancePoseEngagement);
  const auto policy = Policy(1.0, 1.0);
  tracker.Update(0.0, {root, pelvis}, policy);
  const auto expired = tracker.Update(1.0, {root, pelvis}, policy);
  ASSERT_EQ(expired.size(), 1u);
  EXPECT_EQ(expired[0].observation.anchor_id, 3u);
  EXPECT_EQ(expired[0].observation.kind, InteractionKind::kAffordancePoseEngagement);
}

TEST(InteractionLatency, EachKindHasItsOwnLimit) {
  InteractionLatencyTracker tracker;
  const auto spacing = Contact(7, 0, 42, InteractionKind::kSpacingContact);
  const auto pose = Contact(7, 3, 43, InteractionKind::kAffordancePoseEngagement);
  const auto policy = Policy(10.0, 1.0);
  tracker.Update(0.0, {spacing, pose}, policy);
  const auto expired = tracker.Update(1.0, {spacing, pose}, policy);
  ASSERT_EQ(expired.size(), 1u);
  EXPECT_EQ(expired[0].observation.kind, InteractionKind::kAffordancePoseEngagement);
}

// Facing never starts a timer (L5): the only two interaction kinds are Spacing
// contact and AffordancePose engagement, and a Facing payload engages neither.
TEST(InteractionLatency, FacingNeverEngagesAndSoNeverStartsATimer) {
  RelationPayload facing;
  facing.kernel_type = RelationKernelType::kAvatarFacing;
  facing.grounded_relation.proximity = 1.0;
  facing.grounded_relation.support_distance_m = 0.0;
  EXPECT_FALSE(IsEngaged(facing));
  EXPECT_FALSE(IsSpacingContact(facing));
  EXPECT_EQ(kInteractionKindCount, 2);
}

// ---------------------------------------------------------------------------
// A world with a source and several receivers (Space level, real hub).
// ---------------------------------------------------------------------------
constexpr element_id_t kRoom = 1, kShelf = 9, kSeat = 10;
constexpr shared_element_id_t kRoomShared = 100, kShelfShared = 900,
                              kSeatShared = 1000;
constexpr double kDt = 1.0 / 30.0;

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

void BuildScene(Space* s) {
  auto* tree = s->GetElementTree();
  ASSERT_TRUE(tree->AddElement(kRoom).ok());
  tree->GetElement(kRoom)->SetSharedId(kRoomShared);
  tree->GetElement(kRoom)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 10, 10)));
  // A shelf the root reaches for: Spacing's referent.
  ASSERT_TRUE(tree->AddElement(kShelf, kRoom).ok());
  tree->GetElement(kShelf)->SetSharedId(kShelfShared);
  tree->GetElement(kShelf)->SetLocalTransform(At(8.0, 5.0));
  tree->GetElement(kShelf)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-0.5, -0.5, 0.0), Eigen::Vector3d(0.5, 0.5, 0.8),
          GeometryRole::kObstacle));
  // A seat with a named top part (AffordancePose's referent), top at z = 0.45.
  ASSERT_TRUE(tree->AddElement(kSeat, kRoom).ok());
  tree->GetElement(kSeat)->SetSharedId(kSeatShared);
  tree->GetElement(kSeat)->SetLocalTransform(At(3.0, 3.0));
  auto seat = std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-0.5, -0.4, 0.0), Eigen::Vector3d(0.5, 0.4, 0.45),
      GeometryRole::kObstacle);
  SubRegion top;
  top.name = "seat";
  top.tag = "Sittable";
  top.local_transform = Eigen::Matrix4d::Identity();
  top.local_transform(2, 3) = 0.45;
  top.boundary = Rect(-0.5, -0.4, 0.5, 0.4);
  seat->AddSubRegion(top);
  tree->GetElement(kSeat)->SetSpatialDefinition(seat);
  tree->GetElement(kSeat)->SetTag("Sittable");
}

struct World {
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space* src = nullptr;
  std::vector<Space*> rcv;
  spatialblend::entity_id_t source = 0;
  std::vector<spatialblend::entity_id_t> avatar;
  std::vector<uint64_t> seq;

  explicit World(int receivers) {
    src = ctx->AddSpace();
    BuildScene(src);
    std::vector<SubjectAnchor> anchors = {
        SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
        SubjectAnchor("pelvis", Eigen::Vector3d(0, 0, 0.9)),
        SubjectAnchor("head", Eigen::Vector3d(0, 0, 1.6)),
        SubjectAnchor("hand", Eigen::Vector3d(0.3, 0, 1.0))};
    source = src->CreateActor(anchors);
    src->GetActor(source)->SetElementId(kRoom);
    src->SetHub(ctx->GetHub());
    src->SetMovementGate(0.0, 0.0);
    for (int i = 0; i < receivers; ++i) {
      Space* r = ctx->AddSpace();
      BuildScene(r);
      rcv.push_back(r);
      avatar.push_back(r->CreateActor(anchors));
      r->GetActor(avatar.back())->SetElementId(kRoom);
      r->SetHub(ctx->GetHub());
      seq.push_back(0);
    }
  }
  void Publish(int i, uint32_t anchor, double x, double y, double z, double yaw) {
    EmbodiedPoseInput in;
    in.x = x;
    in.y = y;
    in.z = z;
    in.yaw = yaw;
    in.epoch = 1;
    // One sequence counter per receiver is enough: every anchor has its own port.
    uint64_t& n = seq[i];
    ASSERT_EQ(rcv[i]->PublishAnchorPose(avatar[i], anchor, in, 0.0, ++n),
              EmbodiedPublishResult::kOk);
  }
  ReceiverProposal Proposal(int i) {
    ReceiverProposal p;
    EXPECT_TRUE(rcv[i]->ReadProposal(avatar[i], &p));
    return p;
  }
  std::vector<Space::ActorEvent> Drain(int i) {
    std::vector<Space::ActorEvent> out;
    Space::ActorEvent e;
    while (rcv[i]->PollActorEvent(avatar[i], &e)) out.push_back(e);
    return out;
  }
  void Move(const Eigen::Vector2d& p, double yaw = 0.0) {
    src->UpdateActorState(source, p, yaw);
  }
};

int Count(const std::vector<Space::ActorEvent>& events,
          Space::ActorEvent::Kind kind) {
  int n = 0;
  for (const auto& e : events) n += e.kind == kind ? 1 : 0;
  return n;
}

// Independent request events: no event leaves that channel's unfinished target
// intact.
struct TargetEventWorld : World {
  TargetEventWorld() : World(1) {
    src->Bind(source, "root", RelationKernelType::kAvatarSpacing);
    src->Bind(source, "root", RelationKernelType::kAvatarFacing);
    rcv[0]->SetRealizationPolicy(RealizationPolicyType::kUniform);
    Publish(0, 0, 2.0, 2.0, 0.0, 0.3);
    EXPECT_TRUE(rcv[0]->SetTargetEventDelivery(avatar[0], true));
    Move({2.0, 2.0}, 0.3);
    Drain(0);
  }

  std::vector<RelationPayload> Requests(double position_count,
                                        double yaw_count) {
    const auto received = rcv[0]->GetReceivedUpdatesSnapshot();
    EXPECT_FALSE(received.updates.empty());
    if (received.updates.empty()) return {};
    std::vector<RelationPayload> out;
    for (auto p : received.updates.back().payloads) {
      if (p.subject_anchor_name != "root" || p.referent_shared_id != 900)
        continue;
      p.update_count = p.kernel_type == RelationKernelType::kAvatarFacing
                           ? yaw_count
                           : position_count;
      if (p.kernel_type == RelationKernelType::kAvatarFacing) {
        p.facing_rotation_sign = 1;
        p.grounded_relation.encoded_state = Eigen::VectorXd::Constant(1, -1.67);
      }
      out.push_back(std::move(p));
    }
    EXPECT_EQ(out.size(), 2u);
    return out;
  }

  void Inject(std::vector<RelationPayload> payloads) {
    EXPECT_TRUE(
        rcv[0]->InjectRelationPayloads(avatar[0], std::move(payloads), 1.0));
  }
};

TEST(TargetEvents, AdmissionRefusesHistoricalLegacyExceptionsWithoutMutation) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.Publish(0, 0, 2, 2, 0, 0);
  w.Move({2, 2});
  const auto before = w.Proposal(0);
  EXPECT_FALSE(w.rcv[0]->SetTargetEventDelivery(w.avatar[0], true));
  EXPECT_EQ(w.Proposal(0).revision, before.revision);
  const auto legacy = w.Drain(0);
  ASSERT_EQ(legacy.size(), 1u);
  EXPECT_EQ(legacy[0].kind, Space::ActorEvent::Kind::kSourceException);
  EXPECT_FALSE(legacy[0].target_delivery);
  ASSERT_TRUE(w.rcv[0]->SetTargetEventDelivery(w.avatar[0], true));
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTargetRelease), 1);
}

TEST(TargetEvents,
     RepeatedEnableIsIdempotentAndReenablePreservesTypedLifecycleGroups) {
  TargetEventWorld w;
  ASSERT_TRUE(w.src->RaiseSourceException(w.source, 7, nullptr, 0));
  w.Move({3, 2}, 0.4);
  const auto revision = w.Proposal(0).revision;
  EXPECT_TRUE(w.rcv[0]->SetTargetEventDelivery(w.avatar[0], true));
  EXPECT_EQ(w.Proposal(0).revision, revision);
  EXPECT_TRUE(w.rcv[0]->SetTargetEventDelivery(w.avatar[0], false));
  EXPECT_TRUE(w.rcv[0]->SetTargetEventDelivery(w.avatar[0], true));
  const auto events = w.Drain(0);
  EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kSourceException), 1);
  EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kPositionRequest), 1);
  EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kYawRequest), 1);
  EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kTargetRelease), 2);
}

TEST(TargetEvents,
     ReenableCancelsWithdrawnLifetimeAnchorsAfterHistoricalTargets) {
  for (bool placement : {false, true}) {
    for (bool consumed : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "placement " << placement << ", consumed " << consumed);
      TargetEventWorld w;
      uint32_t head = 0;
      ASSERT_TRUE(w.rcv[0]->AnchorIdFor(w.avatar[0], "head", &head));
      w.Publish(0, head, 2, 2, 1.6, 0.3);
      const double idle = std::numeric_limits<double>::infinity();
      auto payloads = w.Requests(idle, idle);
      auto facing = payloads.back();
      for (const auto& p : payloads)
        if (p.kernel_type == RelationKernelType::kAvatarFacing) facing = p;
      facing.subject_anchor_name = "head";
      facing.update_count = 1.5;
      payloads.push_back(facing);
      if (placement) {
        // Explicit admission reset makes this an immutable placement batch.
        ASSERT_TRUE(w.rcv[0]->SetTargetEventDelivery(w.avatar[0], false));
        ASSERT_TRUE(w.rcv[0]->SetTargetEventDelivery(w.avatar[0], true));
        w.Drain(0);
      }
      w.Inject(payloads);
      bool pending = false;
      int requests = 0;
      const auto apply = [&](const std::vector<Space::ActorEvent>& events) {
        for (const auto& e : events) {
          if (e.anchor_id != head) continue;
          if (e.kind == Space::ActorEvent::Kind::kYawRequest) {
            ++requests;
            pending = true;
            EXPECT_EQ(e.request_flags, placement ? 1u : 0u);
          } else if (e.kind == Space::ActorEvent::Kind::kTargetRelease &&
                     (e.channels & 2u)) {
            pending = false;
          }
        }
      };
      if (consumed) {
        apply(w.Drain(0));
        ASSERT_TRUE(pending);
      }
      ASSERT_TRUE(w.rcv[0]->SetTargetEventDelivery(w.avatar[0], false));
      w.Inject(
          w.Requests(idle, idle));  // withdraw head authority while disabled
      EXPECT_EQ(w.Proposal(0).Find(head), nullptr);
      ASSERT_TRUE(w.rcv[0]->SetTargetEventDelivery(w.avatar[0], true));
      apply(w.Drain(0));
      EXPECT_EQ(requests, placement || consumed ? 1 : 0)
          << "ordinary queued motion may coalesce at the release boundary";
      EXPECT_FALSE(pending) << "the source stops after withdrawal";
    }
  }
}

TEST(TargetEvents, RetainedRoomPositionsHaveErrorsAtThePublishedPose) {
  for (bool nonroot : {false, true}) {
    SCOPED_TRACE(nonroot ? "head" : "root");
    TargetEventWorld w;
    uint32_t anchor = 0;
    if (nonroot) {
      ASSERT_TRUE(w.rcv[0]->AnchorIdFor(w.avatar[0], "head", &anchor));
      w.Publish(0, anchor, 2, 2, 1.6, 0.3);
    }
    const double idle = std::numeric_limits<double>::infinity();
    const auto prototype = w.Requests(idle, idle);
    const auto requests = [&](double position, double yaw) {
      auto payloads = prototype;
      for (auto& p : payloads) {
        p.update_count =
            p.kernel_type == RelationKernelType::kAvatarFacing ? yaw : position;
        if (nonroot) p.subject_anchor_name = "head";
      }
      return payloads;
    };
    w.Inject(requests(2.0, idle));
    const auto previous = w.Proposal(0);
    const auto* old = previous.Find(anchor);
    ASSERT_NE(old, nullptr);
    ASSERT_GT(Eigen::Vector2d(old->x - 2, old->y - 2).norm(), 0.1);
    w.Drain(0);
    const double sample_z = nonroot ? 1.7 : 0.2;
    w.Publish(0, anchor, 2.1, 2.3, sample_z, 0.5);
    const auto payloads = requests(idle, 1.5);
    w.Inject(payloads);
    const auto proposal = w.Proposal(0);
    const auto* a = proposal.Find(anchor);
    ASSERT_NE(a, nullptr);
    EXPECT_DOUBLE_EQ(a->x, old->x);
    EXPECT_DOUBLE_EQ(a->y, old->y);
    EXPECT_DOUBLE_EQ(a->z, old->z);
    EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kPositionRequest), 0);
    for (const auto& p : payloads) {
      const auto prepared =
          PrepareTargetRelation(p, *w.rcv[0]->GetElementTree());
      ASSERT_TRUE(prepared.ok());
      const bool spacing = p.kernel_type == RelationKernelType::kAvatarSpacing;
      const double z = spacing ? 0.0 : a->z;
      const auto target = EvaluatePreparedTargetRelation(
          prepared, TargetAnchorPose{Eigen::Vector3d(a->x, a->y, z), a->yaw},
          *w.rcv[0]->GetElementTree());
      const auto sample = EvaluatePreparedTargetRelation(
          prepared,
          TargetAnchorPose{Eigen::Vector3d(2.1, 2.3, spacing ? 0.0 : sample_z),
                           0.5},
          *w.rcv[0]->GetElementTree());
      ASSERT_TRUE(target.ok());
      ASSERT_TRUE(sample.ok());
      EXPECT_NEAR(spacing ? a->err_spacing_target : a->err_facing_target,
                  std::fabs(target.r_tgt - target.r_src), 1e-12);
      EXPECT_NEAR(spacing ? a->err_spacing_sample : a->err_facing_sample,
                  std::fabs(sample.r_tgt - sample.r_src), 1e-12);
    }
    if (!nonroot) {
      ASSERT_TRUE(w.src->RaiseSourceException(w.source, 7, nullptr, 0));
      w.Move({4, 3}, 0.7);
      const auto placed = w.Proposal(0);
      EXPECT_TRUE(placed.placement);
      EXPECT_GT(Eigen::Vector2d(placed.x - old->x, placed.y - old->y).norm(),
                0.1)
          << "explicit placement bypasses retention";
    }
  }
}

TEST(TargetEvents, AdmissionAndSourcePublicationAreSerialized) {
  for (int i = 0; i < 20; ++i) {
    World w(1);
    w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
    w.Publish(0, 0, 2, 2, 0, 0);
    std::promise<void> start;
    const auto ready = start.get_future().share();
    std::thread source([&] {
      ready.wait();
      w.Move({2, 2});
    });
    start.set_value();
    const bool admitted = w.rcv[0]->SetTargetEventDelivery(w.avatar[0], true);
    source.join();
    const auto events = w.Drain(0);
    EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kSourceException), 1);
    EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kPositionRequest),
              admitted ? 1 : 0);
    for (const auto& event : events) EXPECT_EQ(event.target_delivery, admitted);
  }
}

TEST(TargetEvents, PositionAndYawAreIndependentAndAnIdleChannelDoesNotReplay) {
  TargetEventWorld w;
  const double idle = std::numeric_limits<double>::infinity();
  w.Inject(w.Requests(2.0, idle));
  const auto position = w.Proposal(0);
  auto events = w.Drain(0);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, Space::ActorEvent::Kind::kPositionRequest);
  EXPECT_EQ(events[0].request_flags, 0u);

  w.Inject(w.Requests(idle, 1.5));
  events = w.Drain(0);
  ASSERT_EQ(events.size(), 1u);
  const auto yaw = events[0];
  EXPECT_EQ(yaw.kind, Space::ActorEvent::Kind::kYawRequest);
  EXPECT_LT(yaw.yaw_turn, -M_PI);
  EXPECT_DOUBLE_EQ(yaw.sample_yaw, 0.3);
  EXPECT_EQ(yaw.sample_epoch, 1u);
  EXPECT_EQ(yaw.sample_sequence, 1u);
  EXPECT_NEAR(std::remainder(yaw.sample_yaw + yaw.yaw_turn - yaw.yaw, 2 * M_PI),
              0, 1e-12);
  EXPECT_DOUBLE_EQ(w.Proposal(0).x, position.x);
  EXPECT_DOUBLE_EQ(w.Proposal(0).y, position.y);

  const auto revision = w.Proposal(0).revision;
  w.Publish(0, 0, 2.1, 2.0, 0.0, 0.5);
  w.Inject(w.Requests(idle, idle));
  EXPECT_TRUE(w.Drain(0).empty());
  EXPECT_EQ(w.Proposal(0).revision, revision);
  w.Inject(w.Requests(2.0, idle));
  events = w.Drain(0);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, Space::ActorEvent::Kind::kPositionRequest);
  // The old signed turn remains attached to its immutable producing sample,
  // never the new set revision.
  EXPECT_DOUBLE_EQ(yaw.sample_yaw, 0.3);
  EXPECT_EQ(yaw.sample_sequence, 1u);
}

TEST(TargetEvents,
     PlacementFeedbackEpochAloneDoesNotReleaseTheLastOrdinaryTarget) {
  TargetEventWorld w;
  w.Inject(w.Requests(2.0, 1.5));
  const auto pending = w.Proposal(0);
  w.Drain(0);
  EmbodiedPoseInput feedback;
  feedback.x = feedback.y = 2;
  feedback.yaw = 4 * M_PI + 0.4;
  feedback.epoch = 3;  // two ordered placements were coalesced into one normal
                       // feedback publication
  ASSERT_EQ(w.rcv[0]->PublishEmbodiedPose(w.avatar[0], feedback, 1.0, 1),
            EmbodiedPublishResult::kOk);
  const double idle = std::numeric_limits<double>::infinity();
  w.Inject(w.Requests(idle, idle));
  EXPECT_TRUE(w.Drain(0).empty())
      << "a new accepted sample epoch is not channel withdrawal";
  EXPECT_DOUBLE_EQ(w.Proposal(0).x, pending.x);
  EXPECT_DOUBLE_EQ(w.Proposal(0).y, pending.y);
  EXPECT_NE(w.Proposal(0).cause & kCauseEmbodiedReset, 0u);
  feedback.epoch = 1;
  EXPECT_EQ(w.rcv[0]->PublishEmbodiedPose(w.avatar[0], feedback, 1.1, 99),
            EmbodiedPublishResult::kStale)
      << "stale PRODUCER publications are refused; valid already-decoded goals "
         "are different";
}

TEST(TargetEvents, AnIdleRootDoesNotSuppressAnotherAnchorsRequest) {
  TargetEventWorld w;
  const double idle = std::numeric_limits<double>::infinity();
  uint32_t head = 0;
  ASSERT_TRUE(w.rcv[0]->AnchorIdFor(w.avatar[0], "head", &head));
  w.Publish(0, head, 2.0, 2.0, 1.6, 0.3);
  auto payloads = w.Requests(idle, idle);
  auto facing = payloads.back();
  for (const auto& p : payloads)
    if (p.kernel_type == RelationKernelType::kAvatarFacing) facing = p;
  facing.subject_anchor_name = "head";
  facing.update_count = 1.5;
  payloads.push_back(facing);
  w.Inject(payloads);
  const auto events = w.Drain(0);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].kind, Space::ActorEvent::Kind::kYawRequest);
  EXPECT_EQ(events[0].anchor_id, head);
  EXPECT_EQ(events[0].request_flags, 0u)
      << "FIRST_PROPOSAL attribution is not an absolute placement";
  EXPECT_LT(events[0].yaw_turn, -M_PI);
}

TEST(TargetEvents,
     DirectContinuousYawAndComposedBoneSampleIdentityAreCapturedAtTheSolve) {
  TargetEventWorld w;
  w.Publish(0, 0, 2, 2, 0, 4 * M_PI + 0.3);
  w.Inject(w.Requests(2.0, 1.5));
  auto events = w.Drain(0);
  for (const auto& event : events) {
    EXPECT_EQ(event.sample_epoch, 1u);
    EXPECT_EQ(event.sample_sequence, 2u);
    EXPECT_DOUBLE_EQ(event.sample_yaw, 4 * M_PI + 0.3);
  }
  Embodiment embodiment;
  std::string error;
  ASSERT_TRUE(Embodiment::FromJson(R"({
    "format":"spatialblend.embodiment","schema_version":"1.0.0","id":"event-sample-fixture",
    "skeleton":{"joints":[{"name":"head","parent":-1}]},
    "anchors":[{"name":"head","bone":"head","families":["Facing"],
                "offset":{"rotation":[0,0,0.7071067811865476,0.7071067811865476]},
                "provenance":{"procedure":"synthetic pi/2 anchor rotation",
                              "source":"TargetEvents composed-anchor test fixture"}}]
  })",
                                   &embodiment, &error))
      << error;
  ASSERT_EQ(w.rcv[0]->RegisterEmbodiment(w.avatar[0], embodiment),
            Space::RegisterEmbodimentResult::kOk);
  BonePoseInput bone;
  bone.p[0] = bone.p[1] = 2;
  bone.p[2] = 1.6;
  bone.q[2] = std::sin(2.4 / 2);
  bone.q[3] = std::cos(2.4 / 2);
  bone.epoch = 7;
  ASSERT_EQ(w.rcv[0]->PublishBonePose(w.avatar[0], 0, bone, 1.0, 41),
            EmbodiedPublishResult::kOk);
  const double idle = std::numeric_limits<double>::infinity();
  auto payloads = w.Requests(idle, idle);
  for (auto p : payloads) {
    if (p.kernel_type != RelationKernelType::kAvatarFacing) continue;
    p.subject_anchor_name = "head";
    p.update_count = 1.5;
    payloads.push_back(std::move(p));
    break;
  }
  w.Inject(std::move(payloads));
  events = w.Drain(0);
  ASSERT_EQ(events.size(), 1u);
  const auto& event = events.front();
  EXPECT_EQ(event.kind, Space::ActorEvent::Kind::kYawRequest);
  EXPECT_EQ(event.sample_epoch, 7u);
  EXPECT_EQ(event.sample_sequence, 41u);
  EXPECT_NEAR(event.sample_yaw, std::remainder(2.4 + M_PI / 2, 2 * M_PI),
              1e-12);
  // The driver correspondence must be for the COMPOSED anchor (2.4 + pi/2), not
  // the bone's 2.4 yaw.
  EXPECT_NEAR(
      std::remainder(2.4 + M_PI / 2 + event.yaw_turn - event.yaw, 2 * M_PI), 0,
      1e-12);
}

TEST(TargetEvents,
     OrdinaryMotionIsBoundedButHistoricalPlacementPosesAreLossless) {
  TargetEventWorld w;
  for (int i = 0; i < 1000; ++i) w.Inject(w.Requests(2.0, 1.5));
  auto events = w.Drain(0);
  ASSERT_EQ(events.size(), 2u);
  for (const auto& e : events) EXPECT_EQ(e.revision, w.Proposal(0).revision);

  ASSERT_TRUE(w.src->RaiseSourceException(w.source, 7, nullptr, 0));
  w.Move({3.0, 2.0}, 0.4);
  const auto first = w.Proposal(0);
  w.Inject(w.Requests(2.0, 1.5));  // obsolete ordinary motion between the two
                                   // genuine lifecycle barriers
  ASSERT_TRUE(w.src->RaiseSourceException(w.source, 8, nullptr, 0));
  w.Move({4.0, 2.0}, 0.8);
  const auto second = w.Proposal(0);
  ASSERT_NE(first.revision, second.revision);
  ASSERT_GT(std::hypot(first.x - second.x, first.y - second.y), 0.1);
  events = w.Drain(0);
  EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kSourceException), 2);
  EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kPositionRequest), 2);
  EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kYawRequest), 2);
  for (const auto& e : events) {
    if (e.kind != Space::ActorEvent::Kind::kPositionRequest &&
        e.kind != Space::ActorEvent::Kind::kYawRequest)
      continue;
    EXPECT_EQ(e.request_flags, 1u);
    const auto& own = e.revision == first.revision ? first : second;
    EXPECT_EQ(e.revision, own.revision);
    EXPECT_DOUBLE_EQ(e.x, own.x);
    EXPECT_DOUBLE_EQ(e.y, own.y);
    EXPECT_DOUBLE_EQ(e.yaw, own.yaw);
  }
}

TEST(
    TargetEvents,
    PrevalidatedOrdinaryGoalsSurviveTwoPlacementEpochChangesInOneStoppedSourceDrain) {
  TargetEventWorld w;
  w.src->Bind(w.source, "head", RelationKernelType::kAvatarFacing);
  uint32_t head = 0;
  ASSERT_TRUE(w.rcv[0]->AnchorIdFor(w.avatar[0], "head", &head));
  const std::map<uint32_t, double> sampled_yaw = {{0, 4 * M_PI + 0.3},
                                                  {head, 6 * M_PI + 0.7}};
  w.Publish(0, 0, 2, 2, 0, sampled_yaw.at(0));
  w.Publish(0, head, 2, 2, 1.6, sampled_yaw.at(head));
  ASSERT_TRUE(w.src->RaiseSourceException(w.source, 7, nullptr, 0));
  w.Move({3, 2}, 0.4);
  ASSERT_TRUE(w.src->RaiseSourceException(w.source, 8, nullptr, 0));
  w.Move({4, 2}, 0.8);
  auto payloads = w.Requests(2.0, 1.5);
  auto head_facing = payloads.front();
  for (const auto& p : payloads)
    if (p.kernel_type == RelationKernelType::kAvatarFacing) head_facing = p;
  head_facing.subject_anchor_name = "head";
  payloads.push_back(head_facing);
  w.Inject(payloads);  // source stops here: no later source tick can repair a
                       // lost final request

  // A consumer first decodes the entire successful fenced drain in the
  // unchanged continuous coordinate domain.
  struct Decoded {
    Space::ActorEvent event;
    double continuous_goal = 0;
  };
  std::vector<Decoded> decoded;
  Space::ActorEvent event;
  int result;
  int ordinary_yaws = 0;
  while ((result = w.rcv[0]->PollActorEventAfterSolve(w.avatar[0], &event)) ==
         1) {
    Decoded item{event};
    if (event.kind == Space::ActorEvent::Kind::kYawRequest &&
        !(event.request_flags & 1u)) {
      ASSERT_EQ(event.sample_epoch, 1u);
      ASSERT_EQ(event.sample_sequence, event.anchor_id == 0 ? 2u : 3u);
      item.continuous_goal = sampled_yaw.at(event.anchor_id) + event.yaw_turn;
      EXPECT_LT(event.yaw_turn, -M_PI);
      EXPECT_NEAR(std::remainder(item.continuous_goal - event.yaw, 2 * M_PI), 0,
                  1e-12);
      ++ordinary_yaws;
    }
    decoded.push_back(item);
  }
  ASSERT_EQ(result, 0);
  ASSERT_EQ(ordinary_yaws, 2);

  std::map<uint32_t, double> realized = sampled_yaw;
  std::map<uint32_t, double> pending;
  std::map<uint32_t, uint64_t> epochs = {{0, 1}, {head, 1}};
  int placements = 0;
  size_t begin = 0;
  while (begin < decoded.size()) {
    size_t end = begin + 1;
    while (end < decoded.size() &&
           decoded[end].event.revision == decoded[begin].event.revision)
      ++end;
    for (size_t i = begin; i < end; ++i) {
      const auto& e = decoded[i].event;
      if (e.kind == Space::ActorEvent::Kind::kSourceException) {
        ++placements;
        pending.clear();  // an absolute lifecycle group cancels earlier
                          // unfinished targets
      }
      if (e.kind == Space::ActorEvent::Kind::kTargetRelease &&
          (e.channels & 2u))
        pending.erase(e.anchor_id);
    }
    for (size_t i = begin; i < end; ++i) {
      const auto& item = decoded[i];
      const auto& e = item.event;
      if (e.kind != Space::ActorEvent::Kind::kYawRequest) continue;
      if (e.request_flags & 1u) {
        // Absolute placement's explicit shortest-arc lift. It changes physical
        // pose and sample epoch, NOT the continuous coordinate origin. Never
        // reset realized to a wrapped quaternion Euler angle here.
        realized[e.anchor_id] +=
            std::remainder(e.yaw - realized[e.anchor_id], 2 * M_PI);
        ++epochs[e.anchor_id];
      } else {
        EXPECT_GT(epochs[e.anchor_id], e.sample_epoch);
        // This event was validated before mutation, so its immutable goal
        // survives THIS drain's placements. This does not admit subsequent
        // arbitrary old-epoch events or reconstruct lost winding from a
        // quaternion.
        pending[e.anchor_id] = item.continuous_goal;
      }
    }
    begin = end;
  }
  EXPECT_EQ(placements, 2);
  ASSERT_EQ(pending.size(), 2u);
  EXPECT_EQ(epochs.at(0), 3u);
  EXPECT_EQ(epochs.at(head), 3u);
  const auto goals = pending;
  for (int frame = 0; frame < 500; ++frame) {
    for (const auto& [anchor, goal] : goals) {
      const double remaining = goal - realized[anchor];
      const double step =
          std::copysign(std::min(0.1, std::fabs(remaining)), remaining);
      realized[anchor] += step;  // bounded ordinary realization, never an
                                 // ordinary teleport/recovery
      EXPECT_LE(std::fabs(step), 0.1);
    }
  }
  for (const auto& [anchor, goal] : goals)
    EXPECT_NEAR(realized.at(anchor), goal, 1e-12);
}

TEST(TargetEvents,
     AcceptedSampleRetirementNotLocalEpochControlsTheTwoDrainPlacementRace) {
  // Reference consumer harness. Native prior-solve/future-sample fencing is
  // exercised separately below.
  const auto rad = [](double degrees) { return degrees * M_PI / 180; };
  using Stamp = std::pair<uint64_t, uint64_t>;
  std::map<Stamp, double> accepted;
  bool published_since_fence = false;
  int accepted_publications = 0;
  const auto publish = [&](uint64_t epoch, uint64_t sequence,
                           double continuous_yaw) {
    if (published_since_fence) return false;
    EXPECT_LT(accepted.size(), 2u);
    accepted[{epoch, sequence}] = continuous_yaw;
    published_since_fence = true;
    ++accepted_publications;
    return true;
  };
  const auto complete_native_empty = [&] {
    if (accepted.size() == 2) accepted.erase(accepted.begin());
    published_since_fence = false;
  };
  const auto decode = [&](const Space::ActorEvent& event) {
    const auto correspondence =
        accepted.find({event.sample_epoch, event.sample_sequence});
    EXPECT_NE(correspondence, accepted.end());
    return correspondence == accepted.end()
               ? std::numeric_limits<double>::quiet_NaN()
               : correspondence->second + event.yaw_turn;
  };
  ASSERT_TRUE(publish(1, 1, rad(170)));
  complete_native_empty();
  double actual = rad(170);
  uint64_t local_epoch = 1;
  Space::ActorEvent first;
  first.kind = Space::ActorEvent::Kind::kYawRequest;
  first.sample_epoch = first.sample_sequence = 1;
  first.yaw_turn = rad(270);  // signed > pi, target 440 degrees in the
                              // persistent world-yaw domain
  const double predecoded = decode(first);
  actual +=
      std::remainder(rad(-170) - actual,
                     2 * M_PI);  // declared absolute placement, lifted to 190
  ++local_epoch;
  EXPECT_NEAR(actual, rad(190), 1e-12);
  double pending = predecoded;  // later same-drain ordinary request remains
                                // 440, not shortened or dropped
  EXPECT_NEAR(pending - actual, rad(250), 1e-12);
  complete_native_empty();
  ASSERT_EQ(accepted.size(), 1u);
  EXPECT_EQ(accepted.begin()->first.first, 1u)
      << "native still owns E1 despite local placement epoch2";

  // Between drain1 and E2 feedback, a source solve may still sample accepted
  // E1. It queues a final target 620.
  Space::ActorEvent late = first;
  late.yaw_turn = rad(450);
  ASSERT_TRUE(
      publish(local_epoch, 1,
              actual));  // one accepted feedback after ALL ordered placements
  EXPECT_FALSE(
      publish(local_epoch, 2,
              actual));  // never evict E1 by accepting a third correspondence
  ASSERT_EQ(accepted.size(), 2u);
  actual = rad(200);  // actual yaw observation continues even though further
                      // pose publication is paused
  EXPECT_NEAR(decode(late), rad(620), 1e-12);
  EXPECT_LT(late.sample_epoch, local_epoch)
      << "an older accepted stamp is not a stale producer publication";
  pending = decode(late);
  // Raw native0 after a conversion failure is NOT a completed drain: do not
  // call complete_native_empty there.
  EXPECT_FALSE(publish(local_epoch, 3, actual));
  EXPECT_EQ(accepted.size(), 2u);
  complete_native_empty();  // successful decode and true0 now fence/retire the
                            // prior ACCEPTED generation E1
  ASSERT_EQ(accepted.size(), 1u);
  EXPECT_EQ(accepted.begin()->first.first, local_epoch);
  EXPECT_EQ(accepted_publications, 2);
  // The source stops. No replay, epoch waiver, shortest-arc recovery, or
  // ordinary teleport is required.
  for (int frame = 0; frame < 500; ++frame) {
    const double step = std::min(rad(1), pending - actual);
    EXPECT_GE(step, 0);
    actual += step;
  }
  EXPECT_NEAR(actual, rad(620), 1e-12);
}

TEST(TargetEvents, SnapshotAndCompleteBatchAreReadyBeforeReentrantObserver) {
  TargetEventWorld w;
  int callbacks = 0;
  const auto token = w.rcv[0]->AddProposalObserver(
      [&](spatialblend::entity_id_t actor, const ReceiverProposal& p) {
        if (actor != w.avatar[0]) return;
        ++callbacks;
        ReceiverProposal ready;
        ASSERT_TRUE(w.rcv[0]->ReadProposal(actor, &ready));
        EXPECT_EQ(ready.revision, p.revision);
        const auto events = w.Drain(0);
        EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kSourceException), 1);
        EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kPositionRequest), 1);
        EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kYawRequest), 1);
        for (const auto& e : events) EXPECT_EQ(e.revision, p.revision);
        // Port publication may reenter, but neither it nor event polling
        // requires the solve lock.
        w.Publish(0, 0, 2.0, 2.0, 0.0, 0.3);
      });
  ASSERT_TRUE(w.src->RaiseSourceException(w.source, 7, nullptr, 0));
  w.Move({3.0, 2.0}, 0.4);
  EXPECT_EQ(callbacks, 1);
  w.rcv[0]->RemoveProposalObserver(token);
}

TEST(
    TargetEvents,
    CBoundaryEmptyDrainFencesPriorSampleSolvesWithoutLockingCallbackSafePorts) {
  TargetEventWorld w;
  const auto payloads = w.Requests(2.0, 1.5);
  std::promise<void> entered, release;
  auto entered_solve = entered.get_future();
  const auto resume = release.get_future().share();
  const auto observer = w.rcv[0]->AddProposalObserver(
      [&](spatialblend::entity_id_t, const ReceiverProposal&) {
        entered.set_value();
        resume.wait();  // holds the solve lock after a solve sampled sequence 1
      });
  std::thread producer([&] { w.Inject(payloads); });
  EXPECT_EQ(entered_solve.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  // Publishing a new accepted sample never waits for this in-flight solve.
  w.Publish(0, 0, 2.1, 2.0, 0.0, 0.5);
  Space::ActorEvent first;
  auto fenced_poll = std::async(std::launch::async, [&] {
    return w.rcv[0]->PollActorEventAfterSolve(w.avatar[0], &first);
  });
  EXPECT_EQ(fenced_poll.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  release.set_value();
  producer.join();
  EXPECT_EQ(fenced_poll.get(), 1);
  EXPECT_EQ(first.sample_sequence, 1u);
  Space::ActorEvent event;
  int result;
  while ((result = w.rcv[0]->PollActorEventAfterSolve(w.avatar[0], &event)) ==
         1)
    EXPECT_EQ(event.sample_sequence, 1u);
  EXPECT_EQ(
      result,
      0);  // only here may the previous accepted correspondence be retired
  w.rcv[0]->RemoveProposalObserver(observer);
  w.Inject(payloads);
  int count = 0;
  while ((result = w.rcv[0]->PollActorEventAfterSolve(w.avatar[0], &event)) ==
         1) {
    ++count;
    EXPECT_EQ(event.sample_sequence, 2u);
    EXPECT_DOUBLE_EQ(event.sample_yaw, 0.5);
  }
  EXPECT_EQ(count, 2);
  EXPECT_EQ(result, 0);
  EXPECT_EQ(w.rcv[0]->PollActorEventAfterSolve(999999, &event), -1);
  EXPECT_EQ(w.rcv[0]->PollActorEventAfterSolve(w.avatar[0], nullptr), -1);
}

TEST(TargetEvents,
     PollingDuringPublicationCannotObserveAnIncompletePlacementRevision) {
  TargetEventWorld w;
  std::atomic<bool> finished{false};
  int groups = 0;
  std::thread consumer([&] {
    std::vector<Space::ActorEvent> group;
    const auto flush = [&] {
      if (group.empty()) return;
      ++groups;
      EXPECT_EQ(Count(group, Space::ActorEvent::Kind::kSourceException), 1);
      EXPECT_EQ(Count(group, Space::ActorEvent::Kind::kPositionRequest), 1);
      EXPECT_EQ(Count(group, Space::ActorEvent::Kind::kYawRequest), 1);
      group.clear();
    };
    for (;;) {
      const bool final_drain = finished.load();
      Space::ActorEvent event;
      if (w.rcv[0]->PollActorEvent(w.avatar[0], &event)) {
        if (!group.empty() && group.front().revision != event.revision) flush();
        ReceiverProposal snapshot;
        EXPECT_TRUE(w.rcv[0]->ReadProposal(w.avatar[0], &snapshot));
        EXPECT_GE(snapshot.revision, event.revision);
        group.push_back(std::move(event));
      } else {
        flush();  // only a TRUE empty queue or a revision boundary commits a
                  // group
        if (final_drain) break;
        std::this_thread::yield();
      }
    }
  });
  for (int i = 0; i < 50; ++i) {
    EXPECT_TRUE(w.src->RaiseSourceException(w.source, 7, nullptr, 0));
    w.Move({3.0 + 0.01 * i, 2.0}, 0.4);
  }
  finished.store(true);
  consumer.join();
  EXPECT_EQ(groups, 50);
}

TEST(TargetEvents, DisablingDoesNotEraseQueuedLifecycleGroupsOrReleases) {
  TargetEventWorld w;
  ASSERT_TRUE(w.src->RaiseSourceException(w.source, 7, nullptr, 0));
  w.Move({3.0, 2.0}, 0.4);
  ASSERT_TRUE(w.rcv[0]->SetTargetEventDelivery(w.avatar[0], false));
  const auto events = w.Drain(0);
  EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kSourceException), 1);
  EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kTargetRelease), 1);
  EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kPositionRequest), 1);
  EXPECT_EQ(Count(events, Space::ActorEvent::Kind::kYawRequest), 1);
}

// ---------------------------------------------------------------------------
// The latency policy, end to end: a lagging target is asked, an arrived one is not.
// ---------------------------------------------------------------------------
TEST(LatencyPolicy, OnlyTheTargetThatHasNotArrivedGetsTheTeleportRequest) {
  World w(2);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  for (Space* r : w.rcv) r->SetInteractionLatencyPolicy(Policy(0.5, 0.5));
  // Receiver 0 is driven at once; receiver 1's driver is stuck far away.
  IdealDriver follow0(w.rcv[0], w.avatar[0], {2, 2}, 0.0);
  w.Publish(1, 0, 1.0, 1.0, 0.0, 0.0);

  // The source walks to the shelf and stays ON it: Spacing contact (R == 1).
  for (const Eigen::Vector2d p :
       {Eigen::Vector2d(2, 5), Eigen::Vector2d(5, 5), Eigen::Vector2d(7.2, 5)})
    w.Move(p);
  EXPECT_EQ(Count(w.Drain(1), Space::ActorEvent::Kind::kTeleportRequest), 0)
      << "not engaged yet: no timer";
  for (int k = 0; k < 60; ++k) w.Move({8.0, 5.0});  // 2 s of source time

  const auto ev0 = w.Drain(0);
  const auto ev1 = w.Drain(1);
  EXPECT_EQ(Count(ev0, Space::ActorEvent::Kind::kTeleportRequest), 0)
      << "receiver 0 arrived (EXACT): it is not asked";
  ASSERT_EQ(Count(ev1, Space::ActorEvent::Kind::kTeleportRequest), 1)
      << "receiver 1 never arrived: asked, once";
  for (const auto& e : ev1) {
    if (e.kind != Space::ActorEvent::Kind::kTeleportRequest) continue;
    EXPECT_EQ(e.anchor_id, Space::kRootAnchorId);
    EXPECT_EQ(e.interaction_kind, InteractionKind::kSpacingContact);
    EXPECT_GT(e.source_sequence, 0);
    // The request carries the target pose: the proposal the target app should place at.
    const ReceiverProposal p = w.Proposal(1);
    EXPECT_NEAR(e.x, p.x, 1e-9);
    EXPECT_NEAR(e.y, p.y, 1e-9);
    EXPECT_GE(e.revision, 1u);
  }
  EXPECT_TRUE(w.Proposal(0).arrived()) << "its sampled root satisfies the relation";
  EXPECT_FALSE(w.Proposal(1).arrived()) << "its sampled root is 7 m from the shelf";
  // Both targets are exact minimisers (the contact); only one body has arrived.
  EXPECT_EQ(w.Proposal(0).status, ProposalStatus::kExact);
  EXPECT_EQ(w.Proposal(1).status, ProposalStatus::kExact);
}

// Latency clock: the timer reads ONE clock per actor epoch. A sender whose own clock is already
// far from zero when it engages must not see the engagement expire before the limit of ITS time.
TEST(LatencyClock, ASenderWhoseClockStartsNear100sIsNotAskedBeforeFiveSeconds) {
  World w(1);  // the compiled default: 5000 ms
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);  // the driver never leaves its start
  // ~100 s of sender time, spent away from any referent.
  for (int k = 0; k < 3000; ++k) w.Move({1.0 + 0.001 * (k % 2), 1.0});
  for (const Eigen::Vector2d p : {Eigen::Vector2d(2, 5), Eigen::Vector2d(7.2, 5)})
    w.Move(p);
  for (int k = 0; k < 120; ++k) w.Move({8.0, 5.0});  // 4 s on the shelf
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 0)
      << "inside the 5 s limit of the sender's own time";
  for (int k = 0; k < 60; ++k) w.Move({8.0, 5.0});  // 6 s in all
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 1);
}

// The same sender, switching from its legacy tick clock to timestamped observation frames whose
// sample time is the app's clock (~100 s) while the engagement is held. One engagement is 2 s of
// the sender's time: the receiver must not be asked.
TEST(LatencyClock, TheSenderSwitchingToTimestampedFramesDoesNotJumpTheTimer) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  for (const Eigen::Vector2d p : {Eigen::Vector2d(2, 5), Eigen::Vector2d(7.2, 5)})
    w.Move(p);
  for (int k = 0; k < 15; ++k) w.Move({8.0, 5.0});  // engaged on the legacy tick clock
  uint64_t epoch = 0;
  ASSERT_TRUE(w.src->BeginActorObservationEpoch(w.source, "root", &epoch));
  for (int k = 0; k < 60; ++k) {  // 2 s of timestamped frames, sample time ~100 s
    ASSERT_TRUE(w.src->ApplyActorObservationFrame(
        w.source, epoch, k, 100.0 + k * kDt, {{"root", {8.0, 5.0, 0.0}, 0.0}}));
  }
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 0)
      << "2.5 s of engagement, under the 5 s limit";
}

// A receiver actor never seen before, whose first delivery is already a timestamped sample at ~100 s,
// times the engagement from that sample: 4 s of it is inside the limit, 6 s is past it.
TEST(LatencyClock, AnActorNotSeenBeforeTimesFromItsFirstSample) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  uint64_t epoch = 0;
  ASSERT_TRUE(w.src->BeginActorObservationEpoch(w.source, "root", &epoch));
  int64_t n = 0;
  auto frame = [&](double x, double t) {
    ASSERT_TRUE(w.src->ApplyActorObservationFrame(w.source, epoch, n++, t,
                                                  {{"root", {x, 5.0, 0.0}, 0.0}}));
  };
  frame(2.0, 100.0);
  frame(7.2, 100.0 + kDt);
  for (int k = 0; k < 120; ++k) frame(8.0, 100.0 + (k + 2) * kDt);  // ~4 s engaged
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 0);
  for (int k = 0; k < 60; ++k) frame(8.0, 100.0 + (k + 122) * kDt);  // ~6 s
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 1);
}

// Receiver-side clock epochs, driven by hand-built deliveries of a REAL engaged update (captured from
// a source walking onto the shelf), so the sender's clock domain can change with no reset in
// between: a source's own clock follows its sample time, so a hub-driven sender cannot do this.
RemoteRelationUpdate EngagedTemplate() {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  for (const Eigen::Vector2d p : {Eigen::Vector2d(2, 5), Eigen::Vector2d(7.2, 5)})
    w.Move(p);
  for (int k = 0; k < 3; ++k) w.Move({8.0, 5.0});
  const auto& got = w.rcv[0]->GetReceivedUpdates();
  EXPECT_FALSE(got.empty());
  return got.empty() ? RemoteRelationUpdate{} : got.back();
}

struct HandFed {
  World w{1};
  RemoteRelationUpdate tmpl = EngagedTemplate();
  int64_t seq = 0;
  HandFed() {
    w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);  // the driver never leaves its start
  }
  // One delivery of `seconds` of engagement at the shelf, in the given clock domain.
  void Legacy(double source_clock) {
    RemoteRelationUpdate u = tmpl;
    u.timing = RelationUpdateTiming::kLegacy;
    u.source_epoch = 0;
    u.source_time = 0.0;
    u.source_observation_sequence_known = false;
    u.step_duration = kDt;
    u.source_clock = source_clock;
    u.sequence = ++seq;
    w.rcv[0]->OnRemoteUpdate(u);
  }
  void Timestamped(double time) {
    RemoteRelationUpdate u = tmpl;
    u.timing = RelationUpdateTiming::kTimestampedObservation;
    u.source_epoch = 1;
    u.source_time = time;
    u.source_observation_sequence_known = false;
    u.step_duration = kDt;
    u.source_clock = 0.0;
    u.sequence = ++seq;
    w.rcv[0]->OnRemoteUpdate(u);
  }
  int Requests() { return Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest); }
};

// (a) Timestamped frames (sample time ~100 s), then legacy ticks (tick clock ~1 s), no reset, the
// engagement held throughout. The clock epoch changes, so the timer restarts on the new epoch's
// clock: a real breach is still asked after 5 s there. It is neither frozen by the old ~100 s
// reading (max(100, 1) never moves) nor asked early.
TEST(LatencyClock, TimestampedThenLegacyStillRaisesARealBreachAfterFiveSeconds) {
  HandFed f;
  for (int k = 0; k < 30; ++k) f.Timestamped(100.0 + k * kDt);  // ~1 s engaged
  for (int k = 0; k < 120; ++k) f.Legacy(1.0 + (k + 1) * kDt);   // 4 s more
  EXPECT_EQ(f.Requests(), 0) << "inside the limit";
  for (int k = 120; k < 240; ++k) f.Legacy(1.0 + (k + 1) * kDt);  // 8 s of legacy
  EXPECT_EQ(f.Requests(), 1) << "the breach is asked, once";
}

// (b) A source clock that first appears part-way through a legacy epoch must not move the timeline:
// the epoch's source (accumulated steps, no source_clock at its first update) is latched.
TEST(LatencyClock, ASourceClockAppearingInALegacyEpochDoesNotJumpTheTimer) {
  HandFed f;
  for (int k = 0; k < 30; ++k) f.Legacy(0.0);  // steps only, ~1 s engaged
  for (int k = 0; k < 30; ++k) f.Legacy(100.0 + k * kDt);  // the sender clock shows up at ~100 s
  EXPECT_EQ(f.Requests(), 0) << "2 s of engagement, under the 5 s limit";
  for (int k = 0; k < 150; ++k) f.Legacy(0.0);  // steps again: ~7 s in all
  EXPECT_EQ(f.Requests(), 1);
}

TEST(LatencyPolicy, TheSourceLeavingBeforeTheLimitCancelsTheRequest) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.rcv[0]->SetInteractionLatencyPolicy(Policy(1.0, 1.0));
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  for (const Eigen::Vector2d p : {Eigen::Vector2d(2, 5), Eigen::Vector2d(7.2, 5)})
    w.Move(p);
  for (int k = 0; k < 10; ++k) w.Move({8.0, 5.0});  // 0.33 s on the shelf
  for (int k = 0; k < 90; ++k) w.Move({4.0, 5.0});  // then away for 3 s
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 0);
}

TEST(LatencyPolicy, TheShippedPolicyAsksAfterFiveSecondsOfSourceTime) {
  World w(1);  // the compiled default: 5000 ms (L4)
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);  // the driver never leaves its start
  for (const Eigen::Vector2d p : {Eigen::Vector2d(2, 5), Eigen::Vector2d(7.2, 5)})
    w.Move(p);
  for (int k = 0; k < 120; ++k) w.Move({8.0, 5.0});  // 4 s on the shelf
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 0)
      << "inside the 5 s limit";
  for (int k = 0; k < 60; ++k) w.Move({8.0, 5.0});  // 6 s in all
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 1)
      << "past the limit, the lagging target is asked, once";
  EXPECT_DOUBLE_EQ(*w.rcv[0]->GetInteractionLatencyPolicy().For(
                       InteractionKind::kSpacingContact),
                   5.0);
}

// ARRIVED is read from the SAMPLED anchor, never from the
// solver's EXACT.
TEST(LatencyPolicy, ADriverThatLagsWhileTheSolverIsExactHasNotArrived) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.rcv[0]->SetInteractionLatencyPolicy(Policy(0.5, 0.5));
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);  // the body stays 7 m from the shelf
  for (const Eigen::Vector2d p : {Eigen::Vector2d(2, 5), Eigen::Vector2d(7.2, 5)})
    w.Move(p);
  for (int k = 0; k < 20; ++k) w.Move({8.0, 5.0});
  const ReceiverProposal p = w.Proposal(0);
  EXPECT_EQ(p.status, ProposalStatus::kExact) << "the solver's target is exact";
  EXPECT_FALSE(p.arrived()) << "the anchor has not arrived: the sample is far";
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 1)
      << "so the timer ran out and the app is asked";
}

TEST(LatencyPolicy, WhenTheSampleReachesTheTargetTheAnchorArrivesAndTheTimerStops) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.rcv[0]->SetInteractionLatencyPolicy(Policy(0.5, 0.5));
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  for (const Eigen::Vector2d p : {Eigen::Vector2d(2, 5), Eigen::Vector2d(7.2, 5)})
    w.Move(p);
  for (int k = 0; k < 5; ++k) w.Move({8.0, 5.0});  // engaged, the body still far
  ASSERT_FALSE(w.Proposal(0).arrived());
  const ReceiverProposal target = w.Proposal(0);
  // The driver reaches the proposed target: the SAMPLE now satisfies the relation.
  w.Publish(0, 0, target.x, target.y, 0.0, target.yaw);
  w.Move({8.0, 5.0});
  EXPECT_TRUE(w.Proposal(0).arrived());
  for (int k = 0; k < 60; ++k) w.Move({8.0, 5.0});  // far past the 0.5 s limit
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 0)
      << "an arrived target is never asked";
}

TEST(LatencyPolicy, ANewEngagementAfterReleaseStartsANewTimer) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.rcv[0]->SetInteractionLatencyPolicy(Policy(0.5, 0.5));
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  for (const Eigen::Vector2d p : {Eigen::Vector2d(2, 5), Eigen::Vector2d(7.2, 5)})
    w.Move(p);
  for (int k = 0; k < 40; ++k) w.Move({8.0, 5.0});
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 1);
  for (int k = 0; k < 30; ++k) w.Move({4.0, 5.0});  // lets go
  for (int k = 0; k < 40; ++k) w.Move({8.0, 5.0});  // engages again
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 1)
      << "one request per engagement";
}

// A NONE solve ends the engagement (L3), so
// its timer is cancelled and a later engagement starts a new one. A timer latched before the gap
// must not carry across it: the second touch below is 0.67 s after the receiver regained the
// referent, well inside the 1 s limit, but 1.67 s after the first touch.
TEST(LatencyPolicy, ANoneSolveEndsTheEngagementSoALaterTouchStartsANewTimer) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.rcv[0]->SetInteractionLatencyPolicy(Policy(1.0, 1.0));
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);  // the driver never leaves its start: it never arrives
  for (const Eigen::Vector2d p : {Eigen::Vector2d(2, 5), Eigen::Vector2d(7.2, 5)})
    w.Move(p);
  for (int k = 0; k < 20; ++k) w.Move({8.0, 5.0});  // 0.67 s on the shelf: the timer runs
  ASSERT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 0);

  // The receiver loses the shelf's counterpart: no referent maps, the proposal is NONE.
  ASSERT_TRUE(w.rcv[0]->SetElementSharedId(kShelf, kShelfShared + 1));
  for (int k = 0; k < 30; ++k) w.Move({8.0, 5.0});  // 1 s of NONE, the source still on the shelf
  ASSERT_EQ(w.Proposal(0).status, ProposalStatus::kNone);
  ASSERT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 0) << "NONE asks nothing";

  // The counterpart is back: a new engagement.
  ASSERT_TRUE(w.rcv[0]->SetElementSharedId(kShelf, kShelfShared));
  for (int k = 0; k < 20; ++k) w.Move({8.0, 5.0});  // 0.67 s into the new engagement
  ASSERT_NE(w.Proposal(0).status, ProposalStatus::kNone);
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 0)
      << "the new engagement started a new timer; it has not run out";
  for (int k = 0; k < 30; ++k) w.Move({8.0, 5.0});  // 1.33 s in: past the limit
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kTeleportRequest), 1)
      << "and it does run out, once, from its own start";
}

// The height a root proposal's Facing error is
// evaluated at is the embodied sample's own (the sampling port is the source of truth), never an
// invented 1.3 m eye height. v1 has no height at all here (Angular1D reads kPosition2D and yaw), so
// the sample's height is the only thing that can enter. A referent TILTED about the X axis makes the
// look-at contact depend on the height of the anchor (a polygon that stands out of the floor plane is
// read through the anchor's full 3D position): a body that stands exactly where the source stands, at
// the source's own height, must read a Facing error of 0 whatever the tilt, the heading and the size.
double RootFacingErrorAtTheSourcePose(double tilt, double yaw, double half_w, double half_h, double sample_z) {
  auto ctx = LocalContext::Create();
  Space* src = ctx->AddSpace();
  Space* rcv = ctx->AddSpace();
  constexpr element_id_t kBoard = 9;
  constexpr shared_element_id_t kBoardShared = 900;
  for (Space* s : {src, rcv}) {
    auto* tree = s->GetElementTree();
    EXPECT_TRUE(tree->AddElement(kRoom).ok());
    tree->GetElement(kRoom)->SetSharedId(kRoomShared);
    tree->GetElement(kRoom)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 10, 10)));
    EXPECT_TRUE(tree->AddElement(kBoard, kRoom).ok());
    tree->GetElement(kBoard)->SetSharedId(kBoardShared);
    Eigen::Matrix4d placement = At(5.0, 8.0);
    placement(1, 1) = std::cos(tilt); placement(1, 2) = -std::sin(tilt);
    placement(2, 1) = std::sin(tilt); placement(2, 2) = std::cos(tilt);
    tree->GetElement(kBoard)->SetLocalTransform(placement);
    tree->GetElement(kBoard)->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
        Eigen::Vector3d(-half_w, -half_h, 0.0), Eigen::Vector3d(half_w, half_h, 1.0), GeometryRole::kObstacle));
  }
  const std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))};
  const auto source = src->CreateActor(anchors);
  src->GetActor(source)->SetElementId(kRoom);
  const auto avatar = rcv->CreateActor(anchors);
  rcv->GetActor(avatar)->SetElementId(kRoom);
  src->SetHub(ctx->GetHub());
  rcv->SetHub(ctx->GetHub());
  src->SetMovementGate(0.0, 0.0);
  src->Bind(source, "root", RelationKernelType::kAvatarFacing);
  EmbodiedPoseInput in;
  in.x = 5.0; in.y = 2.0; in.z = sample_z; in.yaw = yaw; in.epoch = 1;
  EXPECT_EQ(rcv->PublishAnchorPose(avatar, 0, in, 0.0, 1), EmbodiedPublishResult::kOk);
  // The source turns toward the board (0.1 rad in the update before): a Facing that makes no progress has
  // an infinite update count and no weight.
  src->UpdateActorState(source, Eigen::Vector2d(5.0, 2.0), yaw - 0.1);
  src->UpdateActorState(source, Eigen::Vector2d(5.0, 2.0), yaw);
  ReceiverProposal p;
  EXPECT_TRUE(rcv->ReadProposal(avatar, &p));
  return p.status == ProposalStatus::kNone ? -2.0 : p.err_facing_sample;
}

TEST(RootProposal, TheFacingErrorIsEvaluatedAtTheSampledHeight) {
  int checked = 0;
  for (const double tilt : {0.3, 0.6, 0.9, 1.2})
    for (const double yaw : {1.0, 1.2, 1.4})
      for (const double half_w : {0.2, 1.0})
        for (const double half_h : {0.3, 0.8}) {
          const double err = RootFacingErrorAtTheSourcePose(tilt, yaw, half_w, half_h, 0.0);
          ASSERT_GE(err, 0.0) << "the Facing error at the sample was reported";
          EXPECT_NEAR(err, 0.0, 1e-9) << "tilt " << tilt << " yaw " << yaw << " board " << half_w << "x" << half_h
                                      << ": the sample is where the source stands, at the source's height";
          ++checked;
        }
  EXPECT_EQ(checked, 48);
}

// ---------------------------------------------------------------------------
// Source-app exceptions.
// ---------------------------------------------------------------------------
TEST(SourceException, FirstUpdateOfASourceIsItsJoinException) {
  World w(2);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  for (int i = 0; i < 2; ++i) w.Publish(i, 0, 1.0, 1.0, 0.0, 0.0);
  w.Move({4, 5});
  for (int i = 0; i < 2; ++i) {
    const auto ev = w.Drain(i);
    ASSERT_EQ(ev.size(), 1u);
    EXPECT_EQ(ev[0].kind, Space::ActorEvent::Kind::kSourceException);
    EXPECT_EQ(ev[0].exception_id, kExceptionJoin);
    EXPECT_TRUE(ev[0].payload.empty());
    const ReceiverProposal p = w.Proposal(i);
    EXPECT_TRUE(p.cause & kCauseSourceException);
    EXPECT_TRUE(p.cause & kCauseFirstProposal);
    EXPECT_TRUE(p.IsAbsolute()) << "the join is realized as the absolute solve";
    EXPECT_EQ(ev[0].revision, p.revision);
  }
  // Later updates carry no exception.
  w.Move({4.5, 5});
  for (int i = 0; i < 2; ++i) {
    EXPECT_TRUE(w.Drain(i).empty());
    EXPECT_FALSE(w.Proposal(i).cause & kCauseSourceException);
  }
}

// A receiver whose first proposal does not answer the source's first update (its body had no sample yet when the
// source joined, as a networked avatar whose driver attaches after the peer's stream started) is still placed by its
// first proposal, and the app is told so by the join exception on that set: the driver realizes a placement only from
// the event (the default realization of a source exception is a teleport to the set).
TEST(SourceException, AReceiverPlacedAfterTheSourcesFirstUpdateGetsTheJoinOnItsPlacedSet) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.Move({4, 5});  // the source's first update: the receiver has no sample yet
  {
    const ReceiverProposal p = w.Proposal(0);
    EXPECT_FALSE(p.IsAbsolute()) << "no sample: nothing placed yet";
  }
  for (const auto& e : w.Drain(0)) std::fprintf(stderr, "[late join] no-sample set: event kind %d id %u rev %llu\n",
                                                 static_cast<int>(e.kind), e.exception_id,
                                                 static_cast<unsigned long long>(e.revision));
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);  // the driver attaches, far from the placement
  w.Move({4.2, 5});
  const ReceiverProposal p = w.Proposal(0);
  ASSERT_TRUE(p.IsAbsolute()) << "the receiver's first target is placed";
  const auto ev = w.Drain(0);
  ASSERT_EQ(Count(ev, Space::ActorEvent::Kind::kSourceException), 1) << "the placement is announced";
  EXPECT_EQ(ev[0].exception_id, kExceptionJoin);
  EXPECT_EQ(ev[0].revision, p.revision) << "on the placed set";
}

// The same for a receiver space that joins the session after the source's first update went out.
TEST(SourceException, AReceiverSpaceJoiningLateGetsTheJoinOnItsPlacedSet) {
  World w(0);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.Move({4, 5});  // sequence 1, before the receiver exists
  w.Move({4.1, 5});
  Space* r = w.ctx->AddSpace();
  BuildScene(r);
  w.rcv.push_back(r);
  w.avatar.push_back(r->CreateActor({SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
                                     SubjectAnchor("pelvis", Eigen::Vector3d(0, 0, 0.9)),
                                     SubjectAnchor("head", Eigen::Vector3d(0, 0, 1.6)),
                                     SubjectAnchor("hand", Eigen::Vector3d(0.3, 0, 1.0))}));
  r->GetActor(w.avatar.back())->SetElementId(kRoom);
  r->SetHub(w.ctx->GetHub());
  w.seq.push_back(0);
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  w.Move({4.2, 5});
  const ReceiverProposal p = w.Proposal(0);
  ASSERT_TRUE(p.IsAbsolute()) << "the receiver's first target is placed";
  const auto ev = w.Drain(0);
  ASSERT_EQ(Count(ev, Space::ActorEvent::Kind::kSourceException), 1) << "the placement is announced";
  EXPECT_EQ(ev[0].exception_id, kExceptionJoin);
  EXPECT_EQ(ev[0].revision, p.revision);
}

TEST(SourceException, PropagatesToEveryOtherSpaceWithItsOpaquePayload) {
  World w(3);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  for (int i = 0; i < 3; ++i) w.Publish(i, 0, 1.0, 1.0, 0.0, 0.0);
  w.Move({4, 5});
  for (int i = 0; i < 3; ++i) w.Drain(i);  // the join

  // The whiteboard button: an app-defined exception with bytes core never reads
  // (NULs and 0xFF included).
  const std::vector<uint8_t> blob = {0x00, 0x01, 0xFF, 0x00, 0x7F, 0x80};
  ASSERT_TRUE(w.src->RaiseSourceException(w.source, 77, blob.data(), blob.size()));
  w.Move({4, 5});  // the source did not move: the exception is itself a reason to send
  for (int i = 0; i < 3; ++i) {
    const auto ev = w.Drain(i);
    ASSERT_EQ(ev.size(), 1u) << "receiver " << i;
    EXPECT_EQ(ev[0].kind, Space::ActorEvent::Kind::kSourceException);
    EXPECT_EQ(ev[0].exception_id, 77u);
    EXPECT_EQ(ev[0].payload, blob);
    const ReceiverProposal p = w.Proposal(i);
    EXPECT_TRUE(p.cause & kCauseSourceException);
    EXPECT_TRUE(p.IsAbsolute());
    EXPECT_EQ(ev[0].revision, p.revision) << "raised with the set it belongs to";
  }
  // It rides ONE update only.
  w.Move({4, 5});
  for (int i = 0; i < 3; ++i) EXPECT_TRUE(w.Drain(i).empty());
}

TEST(SourceException, TheExceptionGoesOutThroughAnIdleMovementGate) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.src->SetMovementGate(100.0, 100.0);  // a gate nothing passes
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  w.Move({4, 5});
  w.Drain(0);
  const uint64_t before = w.Proposal(0).revision;
  w.Move({4, 5});
  EXPECT_EQ(w.Proposal(0).revision, before) << "the gate held an unchanged update";
  ASSERT_TRUE(w.src->RaiseSourceException(w.source, 5, nullptr, 0));
  w.Move({4, 5});
  EXPECT_GT(w.Proposal(0).revision, before);
  const auto ev = w.Drain(0);
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0].exception_id, 5u);
}

// A pending source exception propagates immediately, even on a tick that carries no relation (here
// the source has nothing to face and no snapshot is outstanding).
TEST(SourceException, ARaisedExceptionGoesOutEvenWhenTheTickCarriesNoRelation) {
  auto ctx = LocalContext::Create();
  Space* src = ctx->AddSpace();
  Space* rcv = ctx->AddSpace();
  for (Space* s : {src, rcv}) {
    auto* tree = s->GetElementTree();
    ASSERT_TRUE(tree->AddElement(kRoom).ok());
    tree->GetElement(kRoom)->SetSharedId(kRoomShared);
    tree->GetElement(kRoom)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 10, 10)));
  }
  const std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))};
  const auto source = src->CreateActor(anchors);
  src->GetActor(source)->SetElementId(kRoom);
  const auto avatar = rcv->CreateActor(anchors);
  rcv->GetActor(avatar)->SetElementId(kRoom);
  src->SetHub(ctx->GetHub());
  rcv->SetHub(ctx->GetHub());
  src->SetMovementGate(0.0, 0.0);
  src->Bind(source, "root", RelationKernelType::kAvatarFacing);  // no element to face: no relation
  EmbodiedPoseInput in;
  in.x = 1.0; in.y = 1.0; in.z = 0.0; in.yaw = 0.0; in.epoch = 1;
  ASSERT_EQ(rcv->PublishAnchorPose(avatar, 0, in, 0.0, 1), EmbodiedPublishResult::kOk);
  src->UpdateActorState(source, Eigen::Vector2d(4, 5), 0.0);
  src->UpdateActorState(source, Eigen::Vector2d(4.5, 5), 0.0);
  const auto drain = [&] {
    std::vector<Space::ActorEvent> out;
    Space::ActorEvent e;
    while (rcv->PollActorEvent(avatar, &e)) out.push_back(e);
    return out;
  };
  drain();
  ASSERT_TRUE(src->RaiseSourceException(source, 61, nullptr, 0));
  src->UpdateActorState(source, Eigen::Vector2d(4.5, 5), 0.0);
  const auto ev = drain();
  ASSERT_EQ(ev.size(), 1u) << "the exception reached the receiver on the tick it was raised";
  EXPECT_EQ(ev[0].kind, Space::ActorEvent::Kind::kSourceException);
  EXPECT_EQ(ev[0].exception_id, 61u);
  src->UpdateActorState(source, Eigen::Vector2d(4.5, 5), 0.0);
  EXPECT_TRUE(drain().empty()) << "and it rides one update only";
}

TEST(SourceException, ARefusedPayloadRaisesNothing) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  w.Move({4, 5});
  w.Drain(0);
  const std::vector<uint8_t> too_big(Space::kMaxExceptionPayloadBytes + 1, 1);
  EXPECT_FALSE(w.src->RaiseSourceException(w.source, 9, too_big.data(), too_big.size()));
  EXPECT_FALSE(w.src->RaiseSourceException(w.source, 9, nullptr, 3));
  EXPECT_FALSE(w.src->RaiseSourceException(987654, 9, nullptr, 0));
  const std::vector<uint8_t> exactly(Space::kMaxExceptionPayloadBytes, 2);
  EXPECT_TRUE(w.src->RaiseSourceException(w.source, 9, exactly.data(), exactly.size()));
  w.Move({4.2, 5});
  const auto ev = w.Drain(0);
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0].payload.size(), Space::kMaxExceptionPayloadBytes);
}

// ---------------------------------------------------------------------------
// One pose proposal per relation-bearing anchor.
// ---------------------------------------------------------------------------
const AnchorProposal* Find(const ReceiverProposal& p, const std::string& name) {
  for (const auto& a : p.anchors)
    if (a.anchor_name == name) return &a;
  return nullptr;
}

// The source's pelvis ON the seat's top (touching), root standing beside it.
void SitFrame(World& w, double z, double yaw = 0.0) {
  w.src->SetAnchorWorldPose(w.source, "pelvis", Eigen::Vector3d(3.1, 3.05, z), yaw);
  w.src->UpdateActorState(w.source, Eigen::Vector2d(3.1, 3.05), yaw);
}

TEST(TargetEvents, UnsampledEngagedYawIsAbsoluteOnlyUntilItsOwnSampleArrives) {
  for (bool engaged_join : {false, true}) {
    World w(1);
    w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
    ASSERT_NE(w.src->BindScoped(w.source, "pelvis",
                                RelationKernelType::kAvatarAffordancePose,
                                kSeat, "Sittable"),
              nullptr);
    uint32_t pelvis = 0;
    ASSERT_TRUE(w.rcv[0]->AnchorIdFor(w.avatar[0], "pelvis", &pelvis));
    w.Publish(0, 0, 1, 1, 0, 0);
    ASSERT_TRUE(w.rcv[0]->SetTargetEventDelivery(w.avatar[0], true));
    w.Drain(0);
    SitFrame(w, engaged_join ? 0.45 : 0.65, 0.4);
    auto events = w.Drain(0);
    int placed_yaws = 0;
    for (const auto& e : events) {
      if (e.anchor_id != pelvis ||
          e.kind != Space::ActorEvent::Kind::kYawRequest)
        continue;
      ++placed_yaws;
      EXPECT_EQ(e.request_flags, 1u);
      EXPECT_NE(e.yaw, 0.0);
    }
    EXPECT_EQ(placed_yaws, engaged_join ? 1 : 0);
    SitFrame(w, 0.45, 0.4);
    const auto proposal = w.Proposal(0);
    const auto* a = proposal.Find(pelvis);
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a->status, ProposalStatus::kExact);
    EXPECT_TRUE(a->engaged());
    EXPECT_FALSE(a->sampled());
    for (const auto& e : w.Drain(0))
      EXPECT_FALSE(e.anchor_id == pelvis &&
                   e.kind == Space::ActorEvent::Kind::kYawRequest)
          << "ordinary relative yaw has no native-accepted pelvis origin";
    ASSERT_TRUE(w.src->RaiseSourceException(w.source, 7, nullptr, 0));
    SitFrame(w, 0.45, 0.4);
    events = w.Drain(0);
    placed_yaws = 0;
    for (const auto& e : events) {
      if (e.anchor_id != pelvis ||
          e.kind != Space::ActorEvent::Kind::kYawRequest)
        continue;
      ++placed_yaws;
      EXPECT_EQ(e.request_flags, 1u);
      EXPECT_DOUBLE_EQ(e.yaw, a->yaw);
    }
    EXPECT_EQ(placed_yaws, 1) << "explicit unsampled placement is still valid";
    w.Publish(0, pelvis, a->x, a->y, a->z, 0.1);
    SitFrame(w, 0.45, 0.4);
    events = w.Drain(0);
    int relative_yaws = 0;
    for (const auto& e : events) {
      if (e.anchor_id != pelvis ||
          e.kind != Space::ActorEvent::Kind::kYawRequest)
        continue;
      ++relative_yaws;
      EXPECT_EQ(e.request_flags, 0u);
      EXPECT_EQ(e.sample_epoch, 1u);
      EXPECT_EQ(e.sample_sequence, w.seq[0]);
      EXPECT_DOUBLE_EQ(e.sample_yaw, 0.1);
      EXPECT_NEAR(std::remainder(e.sample_yaw + e.yaw_turn - e.yaw, 2 * M_PI),
                  0.0, 1e-12);
    }
    EXPECT_EQ(relative_yaws, 1);
    EXPECT_TRUE(w.Drain(0).empty()) << "a stopped source does not replay yaw";
  }
}

TEST(TargetEvents, RetainedInnerPositionHasCurrentPoseResidualsAndObjective) {
  TargetEventWorld w;
  for (Space* space : {w.src, w.rcv[0]}) {
    auto* tree = space->GetElementTree();
    for (bool stand : {false, true}) {
      const element_id_t id = stand ? 11 : 12;
      ASSERT_TRUE(tree->AddElement(id, kSeat).ok());
      auto* pose = tree->GetElement(id);
      pose->SetTag(stand ? "stand" : "seated");
      pose->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-0.5, stand ? -0.4 : 0.0, 0.0),
          Eigen::Vector3d(0.5, stand ? 0.0 : 0.4, 0.45),
          GeometryRole::kInteractionRegion));
      pose->SetAffordancePose({"pelvis",
                               stand
                                   ? AffordancePoseTemplate::Support::kActorRoot
                                   : AffordancePoseTemplate::Support::kPart,
                               stand ? "" : "seat"});
    }
  }
  Embodiment body;
  body.id = "synthetic-retained-inner-fixture";
  body.joints = {{"hips", -1}};
  EmbodimentAnchor pelvis_anchor;
  pelvis_anchor.name = "pelvis";
  pelvis_anchor.families = kFamilyAffordancePose;
  pelvis_anchor.has_standing = true;
  pelvis_anchor.standing_height_m = 0.9;
  pelvis_anchor.standing_provenance_json =
      R"({"procedure":"synthetic standing height","source":"retained inner diagnostic test"})";
  EmbodimentAnchor root_anchor;
  root_anchor.name = "root";
  root_anchor.families = kFamilySpacing | kFamilyFacing;
  body.anchors = {root_anchor, pelvis_anchor};
  ASSERT_EQ(w.src->RegisterEmbodiment(w.source, body),
            Space::RegisterEmbodimentResult::kOk);
  ASSERT_EQ(w.rcv[0]->RegisterEmbodiment(w.avatar[0], body),
            Space::RegisterEmbodimentResult::kOk);
  ASSERT_NE(w.src->BindScoped(w.source, "pelvis",
                              RelationKernelType::kAvatarAffordancePose, kSeat,
                              "Sittable"),
            nullptr);
  uint32_t pelvis = 0;
  ASSERT_TRUE(w.rcv[0]->AnchorIdFor(w.avatar[0], "pelvis", &pelvis));
  w.Publish(0, pelvis, 3.1, 3.05, 0.8, 0.0);
  SitFrame(w, 0.7);
  auto received = w.rcv[0]->GetReceivedUpdatesSnapshot();
  ASSERT_FALSE(received.updates.empty());
  auto payloads = received.updates.back().payloads;
  int inner_terms = 0;
  const double idle = std::numeric_limits<double>::infinity();
  for (auto& p : payloads) {
    p.update_count = idle;
    if (IsInnerPosePayload(p)) {
      ++inner_terms;
      if (p.affordance_pose_tag == "seated") p.update_count = 2.0;
    }
  }
  ASSERT_GT(inner_terms, 0);
  w.Inject(payloads);
  auto pending = w.Proposal(0);
  const auto* old = pending.Find(pelvis);
  ASSERT_NE(old, nullptr);
  ASSERT_TRUE(old->inner.valid);
  ASSERT_FALSE(old->inner.no_request);
  ASSERT_GT(std::fabs(old->z - 0.8), 0.01);
  w.Drain(0);
  w.Publish(0, pelvis, 3.1, 3.05, 0.75, 0.0);
  // This sample-epoch transition requires a diagnostic publication, not a new
  // position request. Its retained target must be scored against today's terms.
  EmbodiedPoseInput root;
  root.x = 2.1;
  root.y = 2.0;
  root.z = 0.2;  // child scopes use the retained root, not this trial height
  root.epoch = 2;
  ASSERT_EQ(w.rcv[0]->PublishEmbodiedPose(w.avatar[0], root, 2.0, 1),
            EmbodiedPublishResult::kOk);
  for (auto& p : payloads) p.update_count = idle;
  w.Inject(payloads);
  auto current = w.Proposal(0);
  const auto* a = current.Find(pelvis);
  ASSERT_NE(a, nullptr);
  ASSERT_TRUE(a->inner.valid);
  EXPECT_TRUE(a->inner.no_request);
  EXPECT_DOUBLE_EQ(a->x, old->x);
  EXPECT_DOUBLE_EQ(a->y, old->y);
  EXPECT_DOUBLE_EQ(a->z, old->z);
  EXPECT_EQ(Count(w.Drain(0), Space::ActorEvent::Kind::kPositionRequest), 0);
  // Both relations are idle under Uniform: they have no objective weight, but
  // still have independently measurable relation error at target and sample.
  const auto sd = InstantiatePoseSD(
      *w.rcv[0]->GetElementTree(), kSeat, "pelvis", &body,
      Eigen::Vector3d(current.x, current.y, current.z), current.yaw);
  ASSERT_TRUE(sd.ok());
  double target_error = 0.0, sample_error = 0.0;
  for (const auto& p : payloads) {
    if (!IsInnerPosePayload(p)) continue;
    const auto* pose = sd->Find(p.affordance_pose_tag);
    ASSERT_NE(pose, nullptr);
    PoseContact contact;
    contact.on_part = p.support.has_contact;
    contact.local = p.support.contact_point_local;
    contact.chart = p.support.contact_normalized;
    const auto kind = SurfaceChartKindFromInt(p.support.sub_region_chart);
    ASSERT_TRUE(kind.ok());
    const auto mapped =
        MapPoseContact(pose->support, contact, p.support.boundary.get(), *kind);
    ASSERT_TRUE(mapped.ok());
    // With n=infinity the ratio target distance is the current sample's gap.
    const double d_sample =
        (Eigen::Vector3d(3.1, 3.05, 0.75) - mapped->point).norm();
    const double d_target =
        (Eigen::Vector3d(a->x, a->y, a->z) - mapped->point).norm();
    target_error =
        std::max(target_error, std::fabs(d_target - d_sample) / sd->scale_m);
    sample_error =
        std::max(sample_error, std::fabs(d_sample - d_sample) / sd->scale_m);
  }
  EXPECT_NEAR(a->err_pose_target, target_error, 1e-12);
  EXPECT_NEAR(a->err_pose_sample, sample_error, 1e-12);
  EXPECT_DOUBLE_EQ(a->inner.j, 0.0)
      << "no valid request means no weighted objective terms";
  const auto revision = current.revision;
  w.Inject(payloads);
  EXPECT_EQ(w.Proposal(0).revision, revision);
  EXPECT_TRUE(w.Drain(0).empty());
}

// A Facing anchor has ARRIVED by v1's definition (Angular1D "already aligned": the SIGNED angular
// delta within math::Constants::kEpsilon), not by a length tolerance on R = (1 + cos theta) / 2.
// A test |R_tgt - R_src| <= the solver's 1 mm would accept about +-3.6 degrees near alignment and
// the MIRROR angle -theta, which has the same R.
struct FacingWorld {
  std::unique_ptr<LocalContext> ctx = LocalContext::Create();
  Space* src = nullptr;
  Space* rcv = nullptr;
  spatialblend::entity_id_t source = 0, avatar = 0;
  uint32_t head = 0;
  static constexpr element_id_t kMarker = 9;
  static constexpr shared_element_id_t kMarkerShared = 900;
  FacingWorld() {
    src = ctx->AddSpace();
    rcv = ctx->AddSpace();
    for (Space* sp : {src, rcv}) {
      auto* tree = sp->GetElementTree();
      EXPECT_TRUE(tree->AddElement(kRoom).ok());
      tree->GetElement(kRoom)->SetSharedId(kRoomShared);
      tree->GetElement(kRoom)->SetSpatialDefinition(
          std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 10, 10)));
      // A 2 cm marker 6 m from the head: a point for every bearing here.
      EXPECT_TRUE(tree->AddElement(kMarker, kRoom).ok());
      tree->GetElement(kMarker)->SetSharedId(kMarkerShared);
      tree->GetElement(kMarker)->SetLocalTransform(At(5.0, 8.0));
      tree->GetElement(kMarker)->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-0.01, -0.01, 0.0), Eigen::Vector3d(0.01, 0.01, 1.0), GeometryRole::kObstacle));
    }
    const std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
                                                SubjectAnchor("head", Eigen::Vector3d(0, 0, 1.6))};
    source = src->CreateActor(anchors);
    src->GetActor(source)->SetElementId(kRoom);
    avatar = rcv->CreateActor(anchors);
    rcv->GetActor(avatar)->SetElementId(kRoom);
    src->SetHub(ctx->GetHub());
    rcv->SetHub(ctx->GetHub());
    src->SetMovementGate(0.0, 0.0);
    src->Bind(source, "root", RelationKernelType::kAvatarSpacing);
    src->Bind(source, "head", RelationKernelType::kAvatarFacing);
    EXPECT_TRUE(rcv->AnchorIdFor(avatar, "head", &head));
  }
  // The source's head at (5, 2) faces `source_yaw`; the receiver's head is sampled at the same place
  // facing `sample_yaw`. Returns the receiver's head proposal.
  AnchorProposal Head(double source_yaw, double sample_yaw) {
    EmbodiedPoseInput root;
    root.x = 1.0; root.y = 1.0; root.z = 0.0; root.yaw = 0.0; root.epoch = 1;
    EXPECT_EQ(rcv->PublishAnchorPose(avatar, 0, root, 0.0, 1), EmbodiedPublishResult::kOk);
    EmbodiedPoseInput h;
    h.x = 5.0; h.y = 2.0; h.z = 1.6; h.yaw = sample_yaw; h.epoch = 1;
    EXPECT_EQ(rcv->PublishAnchorPose(avatar, head, h, 0.0, 1), EmbodiedPublishResult::kOk);
    // The source's head turns toward its referent (0.1 rad in the update before): a Facing that makes no
    // progress has an infinite update count and weight 0, and issues no request (v1 Angular1D).
    src->SetAnchorWorldPose(source, "head", Eigen::Vector3d(5.0, 2.0, 1.6), source_yaw - 0.1);
    src->UpdateActorState(source, Eigen::Vector2d(4.0, 2.0), 0.0);
    src->SetAnchorWorldPose(source, "head", Eigen::Vector3d(5.0, 2.0, 1.6), source_yaw);
    src->UpdateActorState(source, Eigen::Vector2d(4.1, 2.0), 0.0);
    ReceiverProposal p;
    EXPECT_TRUE(rcv->ReadProposal(avatar, &p));
    const AnchorProposal* a = Find(p, "head");
    EXPECT_NE(a, nullptr);
    return a ? *a : AnchorProposal();
  }
};

TEST(FacingArrival, AnAlignedHeadHasArrived) {
  FacingWorld w;
  const AnchorProposal h = w.Head(M_PI / 2, M_PI / 2);
  ASSERT_TRUE(h.sampled());
  EXPECT_NEAR(h.err_facing_sample, 0.0, 1e-9);
  EXPECT_TRUE(h.arrived());
}

TEST(FacingArrival, ThreeDegreesOffAtAlignmentIsNotArrivedWhateverTheRSays) {
  FacingWorld w;
  const double off = 3.0 * M_PI / 180.0;
  const AnchorProposal h = w.Head(M_PI / 2, M_PI / 2 + off);
  ASSERT_TRUE(h.sampled());
  // R = (1 + cos 3 deg) / 2 is within 7e-4 of 1: inside the old "1 mm" tolerance on a unitless R.
  EXPECT_LT(h.err_facing_sample, 1e-3);
  EXPECT_GT(h.err_facing_sample, 0.0);
  EXPECT_FALSE(h.arrived()) << "v1 aligns the signed angle; 3 degrees is not aligned";
}

TEST(FacingArrival, TheMirrorAngleHasTheSameRAndIsNotArrived) {
  FacingWorld w;
  // The source's offset is +0.5 rad (its look-at is to its left of where it faces); the receiver's body
  // faces the mirror way: its offset is -0.5 rad. R is even in theta, so R_tgt == R_src.
  const AnchorProposal h = w.Head(M_PI / 2 - 0.5, M_PI / 2 + 0.5);
  ASSERT_TRUE(h.sampled());
  EXPECT_NEAR(h.err_facing_sample, 0.0, 1e-9) << "the same R on both sides";
  EXPECT_FALSE(h.arrived()) << "but not the same signed offset: +theta is not -theta";
}

// The root's ARRIVED is read against the interaction the source holds (L2). A policy that weights
// several referents at once (Uniform) has no pose that satisfies every relation on a receiver that
// does not copy the source's arrangement: the old rule, every positive-weight relation within the
// tolerance, never arrived for a body that is exactly at its contact, and core asked it to teleport.
// Three point referents; the receiver keeps the first where it was and moves the other two.
TEST(LatencyPolicy, UnderUniformABodyAtItsContactHasArrivedAndIsNeverAsked) {
  auto ctx = LocalContext::Create();
  Space* src = ctx->AddSpace();
  Space* rcv = ctx->AddSpace();
  struct Point { element_id_t id; shared_element_id_t shared; Eigen::Vector2d source_at, receiver_at; };
  const Point points[3] = {{11, 1100, {2, 5}, {2, 5}}, {12, 1200, {8, 5}, {8, 2}}, {13, 1300, {5, 8}, {2, 8}}};
  for (Space* s : {src, rcv}) {
    auto* tree = s->GetElementTree();
    ASSERT_TRUE(tree->AddElement(kRoom).ok());
    tree->GetElement(kRoom)->SetSharedId(kRoomShared);
    tree->GetElement(kRoom)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(Rect(0, 0, 10, 10)));
    for (const Point& p : points) {
      ASSERT_TRUE(tree->AddElement(p.id, kRoom).ok());
      tree->GetElement(p.id)->SetSharedId(p.shared);
      const Eigen::Vector2d at = s == src ? p.source_at : p.receiver_at;
      tree->GetElement(p.id)->SetLocalTransform(At(at.x(), at.y()));
    }
  }
  const std::vector<SubjectAnchor> anchors = {SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))};
  const auto source = src->CreateActor(anchors);
  src->GetActor(source)->SetElementId(kRoom);
  const auto avatar = rcv->CreateActor(anchors);
  rcv->GetActor(avatar)->SetElementId(kRoom);
  src->SetHub(ctx->GetHub());
  rcv->SetHub(ctx->GetHub());
  src->SetMovementGate(0.0, 0.0);
  src->Bind(source, "root", RelationKernelType::kAvatarSpacing);
  rcv->SetRealizationPolicy(RealizationPolicyType::kUniform);
  rcv->SetInteractionLatencyPolicy(Policy(0.5, 0.5));
  // The receiver's body stands exactly at its counterpart of the first point: its contact. The other two
  // relations (the source's distances to them) cannot also hold there.
  EmbodiedPoseInput in;
  in.x = 2.0; in.y = 5.0; in.z = 0.0; in.yaw = 0.0; in.epoch = 1;
  ASSERT_EQ(rcv->PublishAnchorPose(avatar, 0, in, 0.0, 1), EmbodiedPublishResult::kOk);
  for (const Eigen::Vector2d p : {Eigen::Vector2d(5, 5), Eigen::Vector2d(3, 5)})
    src->UpdateActorState(source, p, 0.0);
  for (int k = 0; k < 5; ++k) src->UpdateActorState(source, Eigen::Vector2d(2, 5), 0.0);  // at the point
  ReceiverProposal p;
  ASSERT_TRUE(rcv->ReadProposal(avatar, &p));
  ASSERT_NE(p.status, ProposalStatus::kNone);
  EXPECT_TRUE(p.arrived()) << "the engaged relation (the first point) is satisfied at the sampled pose";
  for (int k = 0; k < 60; ++k) src->UpdateActorState(source, Eigen::Vector2d(2, 5), 0.0);  // past 0.5 s
  int requests = 0;
  Space::ActorEvent e;
  while (rcv->PollActorEvent(avatar, &e)) requests += e.kind == Space::ActorEvent::Kind::kTeleportRequest;
  EXPECT_EQ(requests, 0) << "a body at its contact is not asked to teleport";
}

TEST(AnchorProposals, ARootOnlyActorHasExactlyOneProposal) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  w.Move({4, 5});
  const ReceiverProposal p = w.Proposal(0);
  EXPECT_EQ(p.anchor_id, 0u);
  EXPECT_EQ(p.anchor_name, "root");
  EXPECT_TRUE(p.anchors.empty());
  EXPECT_EQ(p.All().size(), 1u);
  EXPECT_TRUE(p.sampled());
}

TEST(AnchorProposals, ThePelvisCarriesTheSeatPoseAndTheRootYawIsNotOverridden) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  // D6: the tag is the AffordancePose binding (which affordance kind the pelvis binds
  // to); without it every object below the room is a candidate, and the shelf has no seat.
  ASSERT_NE(w.src->BindScoped(w.source, "pelvis",
                              RelationKernelType::kAvatarAffordancePose, kSeat,
                              "Sittable"),
            nullptr);
  uint32_t pelvis = 0;
  ASSERT_TRUE(w.rcv[0]->AnchorIdFor(w.avatar[0], "pelvis", &pelvis));
  ASSERT_NE(pelvis, 0u);
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);

  // Hovering 0.2 m over the seat top: the pelvis is bound but does not touch.
  SitFrame(w, 0.65);
  ReceiverProposal p = w.Proposal(0);
  const AnchorProposal* a = Find(p, "pelvis");
  ASSERT_NE(a, nullptr) << "a bound anchor has a proposal even when it follows the body";
  EXPECT_EQ(a->anchor_id, pelvis);
  EXPECT_EQ(a->status, ProposalStatus::kNone);
  EXPECT_FALSE(a->engaged());
  EXPECT_FALSE(a->cause & kCauseSeatEngage);

  // Touching the seat top: engaged, and the proposal IS the seat pose.
  SitFrame(w, 0.45, 0.4);
  p = w.Proposal(0);
  a = Find(p, "pelvis");
  ASSERT_NE(a, nullptr);
  EXPECT_TRUE(a->engaged());
  EXPECT_TRUE(a->cause & kCauseSeatEngage);
  EXPECT_NEAR(a->z, 0.45, 1e-9) << "the seat top";
  EXPECT_NEAR(a->x, 3.1, 1e-6);
  EXPECT_NEAR(a->y, 3.05, 1e-6);
  EXPECT_GT(a->referent_shared_id, 0);
  // The target is the mapped contact (an exact minimiser). With no pelvis sample yet
  // it cannot have arrived.
  EXPECT_EQ(a->status, ProposalStatus::kExact);
  EXPECT_FALSE(a->sampled());
  EXPECT_FALSE(a->arrived());
  // The root proposal carries no seat bit and no seat yaw.
  EXPECT_FALSE(p.cause & (kCauseSeatEngage | kCauseSeatRelease));
  EXPECT_FALSE(p.yaw_specified()) << "no Facing is bound: the seat does not author the root yaw";

  // The driver realizes the seat: the pelvis sample touches the surface -> EXACT.
  w.Publish(0, pelvis, a->x, a->y, a->z, a->yaw);
  SitFrame(w, 0.45, 0.4);
  ReceiverProposal kept = w.Proposal(0);  // Find returns a pointer into the proposal: it must outlive the call
  a = Find(kept, "pelvis");
  ASSERT_NE(a, nullptr);
  EXPECT_TRUE(a->sampled());
  EXPECT_TRUE(a->arrived()) << "the sampled pelvis touches the receiver's seat";
  EXPECT_FALSE(a->cause & kCauseSeatEngage) << "reported on the frame it happens";

  // Standing up releases it on its own proposal.
  SitFrame(w, 0.9);
  p = w.Proposal(0);
  a = Find(p, "pelvis");
  ASSERT_NE(a, nullptr);
  EXPECT_FALSE(a->engaged());
  EXPECT_TRUE(a->cause & kCauseSeatRelease);
  EXPECT_EQ(a->status, ProposalStatus::kNone);
}

// The JOIN frame (the source's first update: no relation has a valid count yet, so the policy weighs every relation 0).
// Engagement is geometric (engagement.h): a source pelvis that touches its seat on the join frame engages the receiver's
// pelvis on its mapped seat on that frame, and a bound pelvis that does not touch yet is awaiting engagement (an admitted
// state), not an unsupported release. Join placement reads the relations whatever their weight.
TEST(AnchorProposals, ThePelvisEngagesOnTheJoinFrameWhenItsSourceAlreadyTouchesTheSeat) {
  World w(1);
  w.rcv[0]->SetRichRealizationEnabled(true);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  ASSERT_NE(w.src->BindScoped(w.source, "pelvis", RelationKernelType::kAvatarAffordancePose, kSeat, "Sittable"),
            nullptr);
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  SitFrame(w, 0.45, 0.4);  // the source's FIRST update: its pelvis already touches the seat top
  ReceiverProposal p = w.Proposal(0);
  const AnchorProposal* a = Find(p, "pelvis");
  ASSERT_NE(a, nullptr);
  EXPECT_TRUE(a->engaged()) << "engagement is geometric: the join frame's weight 0 does not hide the contact";
  EXPECT_EQ(a->status, ProposalStatus::kExact);
  EXPECT_NEAR(a->z, 0.45, 1e-9) << "the mapped seat top";
  EXPECT_NEAR(a->x, 3.1, 1e-6);
  EXPECT_NEAR(a->y, 3.05, 1e-6);
}

TEST(AnchorProposals, ABoundPelvisNotYetTouchingOnTheJoinFrameIsAwaitingEngagement) {
  World w(1);
  w.rcv[0]->SetRichRealizationEnabled(true);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  ASSERT_NE(w.src->BindScoped(w.source, "pelvis", RelationKernelType::kAvatarAffordancePose, kSeat, "Sittable"),
            nullptr);
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  SitFrame(w, 0.65);  // the source's FIRST update: hovering 0.2 m over the seat top
  const auto* s = w.rcv[0]->GetActorRealizationSnapshot(w.avatar[0]);
  ASSERT_NE(s, nullptr);
  bool seen = false;
  for (const auto& channel : s->channels) {
    if (channel.family != RelationKernelType::kAvatarAffordancePose) continue;
    seen = true;
    EXPECT_FALSE(channel.has_goal);
    ASSERT_FALSE(channel.contributors.empty());
    for (const auto& c : channel.contributors)
      EXPECT_EQ(c.disposition, RealizationDisposition::kAwaitingEngagement) << c.reason;
  }
  EXPECT_TRUE(seen) << "the bound pelvis has a channel on the join frame";
}
// AffordancePose scope. It is skipped; it never fails the anchor. With no tag the
// pelvis is bound to every object below the room, among them the shelf, which has no
// part to sit on. The seat still engages, and its proposal is the one the tagged
// binding gives.
TEST(AnchorProposals, AReferentWithNoAffordanceIsSkippedAndTheAnchorDoesNotFail) {
  World untagged(1), tagged(1);
  untagged.src->Bind(untagged.source, "root", RelationKernelType::kAvatarSpacing);
  untagged.src->Bind(untagged.source, "pelvis",
                     RelationKernelType::kAvatarAffordancePose);
  tagged.src->Bind(tagged.source, "root", RelationKernelType::kAvatarSpacing);
  ASSERT_NE(tagged.src->BindScoped(tagged.source, "pelvis",
                                   RelationKernelType::kAvatarAffordancePose, kSeat,
                                   "Sittable"),
            nullptr);
  for (World* w : {&untagged, &tagged}) w->Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  SitFrame(untagged, 0.45, 0.4);
  SitFrame(tagged, 0.45, 0.4);
  const ReceiverProposal untagged_p = untagged.Proposal(0), tagged_p = tagged.Proposal(0);
  const AnchorProposal* u = Find(untagged_p, "pelvis");
  const AnchorProposal* t = Find(tagged_p, "pelvis");
  ASSERT_NE(u, nullptr);
  ASSERT_NE(t, nullptr);
  // The shelf is a candidate of the untagged binding and cannot be grounded.
  const RealizationResult* realized = untagged.rcv[0]->GetRealizationTargets(untagged.avatar[0]);
  ASSERT_NE(realized, nullptr);
  EXPECT_TRUE(u->engaged()) << "the seat engages although the shelf offers nothing";
  EXPECT_EQ(u->status, ProposalStatus::kExact);
  EXPECT_NEAR(u->x, t->x, 1e-9);
  EXPECT_NEAR(u->y, t->y, 1e-9);
  EXPECT_NEAR(u->z, t->z, 1e-9);
  EXPECT_NEAR(u->yaw, t->yaw, 1e-9);
  EXPECT_NEAR(u->z, 0.45, 1e-9) << "the seat top";
}

// Seat touch is the JOINT POINT, the pelvis anchor, against
// the Sittable support. The body (the root standing on the floor, which is the
// actor's footprint) plays no part: a root inside the seat's footprint with the
// pelvis above the plane does not engage, and a root away from the seat with the
// pelvis on the surface does.
TEST(AnchorProposals, SeatTouchIsTheJointPointNotTheBody) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  ASSERT_NE(w.src->BindScoped(w.source, "pelvis",
                              RelationKernelType::kAvatarAffordancePose, kSeat,
                              "Sittable"),
            nullptr);
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  // Root over the seat's footprint, pelvis at its standing height: no touch.
  w.src->SetAnchorWorldPose(w.source, "pelvis", Eigen::Vector3d(3.1, 3.05, 0.9), 0.0);
  w.src->UpdateActorState(w.source, Eigen::Vector2d(3.1, 3.05), 0.0);
  ReceiverProposal kept = w.Proposal(0);
  const AnchorProposal* a = Find(kept, "pelvis");
  ASSERT_NE(a, nullptr);
  EXPECT_FALSE(a->engaged()) << "the body overlapping the seat is not a touch";
  // Root a metre from the seat, the pelvis joint on the seat's surface: a touch.
  w.src->SetAnchorWorldPose(w.source, "pelvis", Eigen::Vector3d(3.1, 3.05, 0.45), 0.0);
  w.src->UpdateActorState(w.source, Eigen::Vector2d(4.6, 3.05), 0.0);
  kept = w.Proposal(0);
  a = Find(kept, "pelvis");
  ASSERT_NE(a, nullptr);
  EXPECT_TRUE(a->engaged()) << "the joint point on the surface is a touch, wherever the body is";
}

// A Facing whose gaze misses its referent looks at the
// silhouette vertex of least angle, and swapping to the other vertex is the definition's own
// discontinuity (in an identical copy the oriented bearing reproduces the source's yaw, so the
// swap only moves the yaw where the shapes differ; the declaration does not depend on that).
// The source names its vertex; a swap carries FACING_ROOT_SWITCH, and no other frame of a
// sweep that keeps the sign root does.
TEST(AnchorProposals, ASilhouetteVertexSwapIsDeclared) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarFacing);
  w.Publish(0, 0, 6.0, 8.0, 0.0, 0.0);
  std::map<shared_element_id_t, int32_t> last;
  bool have_last = false;
  bool vertex_changed = false;
  w.ctx->GetHub()->SetRelationUpdateCallback([&](const RemoteRelationUpdate& u) {
    std::map<shared_element_id_t, int32_t> now;
    for (const auto& p : u.payloads)
      if (p.kernel_type == RelationKernelType::kAvatarFacing)
        now[p.referent_shared_id] = p.support.look_at_vertex;
    vertex_changed = false;
    if (have_last)
      for (const auto& [ref, v] : now) {
        const auto it = last.find(ref);
        if (it != last.end() && it->second >= 0 && v >= 0 && it->second != v)
          vertex_changed = true;
      }
    last = std::move(now);
    have_last = true;
  });
  int swaps = 0, declared = 0;
  for (int k = 0; k < 400; ++k) {
    w.Move({6.0, 8.0}, -M_PI + k * (2.0 * M_PI / 400.0));
    const ReceiverProposal p = w.Proposal(0);
    if (vertex_changed) {
      ++swaps;
      if (p.cause & kCauseFacingRootSwitch) ++declared;
    }
  }
  w.ctx->GetHub()->SetRelationUpdateCallback(nullptr);
  EXPECT_GT(swaps, 0) << "the sweep crosses a tie heading";
  EXPECT_EQ(declared, swaps) << "every swap is declared";
}

// The other discontinuity of the same rule: the look-at contact is the gaze's exit point from
// inside the referent and its entry point or silhouette vertex from outside, so it jumps
// wherever the anchor's path passes through the referent's boundary, also when it enters and
// leaves between two updates (both ends outside). The source says so; the crossing carries
// FACING_ROOT_SWITCH.
TEST(AnchorProposals, AnAnchorPathCrossingTheReferentsBoundaryIsDeclared) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarFacing);
  w.Publish(0, 0, 6.0, 8.0, 0.0, 0.0);
  bool crossed = false;
  w.ctx->GetHub()->SetRelationUpdateCallback([&](const RemoteRelationUpdate& u) {
    crossed = false;
    for (const auto& p : u.payloads)
      if (p.kernel_type == RelationKernelType::kAvatarFacing && p.support.look_at_boundary_crossed)
        crossed = true;
  });
  int crossings = 0, declared = 0, undeclared_quiet = 0;
  const auto step = [&](double x, double y) {
    w.Move({x, y}, M_PI);
    const ReceiverProposal p = w.Proposal(0);
    if (crossed) {
      ++crossings;
      if (p.cause & kCauseFacingRootSwitch) ++declared;
    }
  };
  // Along the row of the shelf (x 7.5..8.5, y 4.5..5.5), gaze turned away from it: it enters
  // and leaves the footprint by small steps, once each.
  w.Move({8.79, 4.0}, M_PI);  // settle below the shelf first (its path from the origin crosses it)
  crossings = declared = 0;
  for (int k = 0; k <= 70; ++k) step(8.79 - k * 0.02, 5.0);
  const int small_step_crossings = crossings;
  // Then through the shelf in one update: both ends outside, the path crosses it twice.
  step(9.0, 5.0);
  step(7.0, 5.0);
  w.ctx->GetHub()->SetRelationUpdateCallback(nullptr);
  (void)undeclared_quiet;
  EXPECT_EQ(small_step_crossings, 2) << "the walk enters and leaves the shelf's footprint";
  EXPECT_GE(crossings, small_step_crossings + 1) << "and the one-update pass through it is reported";
  EXPECT_EQ(declared, crossings) << "every crossing is declared";
}

// The anchor is a parameter of the relation definition, so a Spacing relation of a hand is
// pursued as the root's is (the same solve, in the scope the relation is grounded in), from the hand's own sample: it
// has a proposal, which closes on the shelf update by update, at the hand's own height.
TEST(AnchorProposals, ASpacingRelationOfAHandHasItsOwnProposal) {
  World w(1);
  w.src->Bind(w.source, "hand", RelationKernelType::kAvatarSpacing);
  uint32_t hand = 0;
  ASSERT_TRUE(w.rcv[0]->AnchorIdFor(w.avatar[0], "hand", &hand));
  int pursued = 0;
  double hx = 3.0;
  for (int k = 0; k < 20; ++k) {
    const double sx = 3.0 + 0.1 * k;  // the source hand closes on the shelf (x = 7.5 its near face)
    w.src->SetAnchorWorldPose(w.source, "hand", Eigen::Vector3d(sx, 5.0, 1.0), 0.0);
    w.Publish(0, 0, 2.5, 5.0, 0.0, 0.0);
    w.Publish(0, hand, hx, 5.0, 1.02, 0.0);
    w.Move({2.5, 5.0});
    const ReceiverProposal p = w.Proposal(0);
    const AnchorProposal* h = Find(p, "hand");
    ASSERT_NE(h, nullptr) << "update " << k;
    ASSERT_TRUE(h->status == ProposalStatus::kExact || h->status == ProposalStatus::kInexact) << "update " << k;
    EXPECT_NEAR(h->z, 1.02, 1e-12) << "the hand's own height";
    if (k > 0 && h->x > hx + 1e-9) ++pursued;
    hx = h->x;  // the driver realizes the proposal
  }
  EXPECT_GT(pursued, 10) << "the hand pursues the shelf";
}

// Pitch and roll are per-anchor. A relation of the hand carries the hand's own
// orientation beyond yaw, and the receiver's hand proposal passes it through (the root's stays the actor's).
TEST(AnchorProposals, EachAnchorPassesThroughItsOwnPitchAndRoll) {
  Space rcv;
  rcv.SetSpaceId(2);
  BuildScene(&rcv);
  const auto avatar = rcv.CreateActor({SubjectAnchor("root"), SubjectAnchor("hand", Eigen::Vector3d(0.3, 0, 1.0))});
  rcv.GetActor(avatar)->SetElementId(kRoom);
  IdealDriver driver(&rcv, avatar, {2, 5}, 0.0);
  RelationPayload facing;
  facing.referent_shared_id = kShelfShared;
  facing.subject_anchor_name = "hand";
  facing.kernel_type = RelationKernelType::kAvatarFacing;
  facing.update_count = 1.0;
  facing.has_subject_orientation = true;
  facing.subject_orientation.type = 2;
  facing.subject_orientation.pitch = 0.2;
  facing.subject_orientation.roll = -0.1;
  RemoteRelationUpdate u;
  u.source_space_id = 1;
  u.actor_shared_id = kRoomShared;  // the actor's element (as World binds it)
  u.sequence = 1;
  u.step_duration = kDt;
  u.source_orientation_type = 1;
  u.source_pitch = 0.05;
  u.payloads = {facing};
  rcv.OnRemoteUpdate(u);
  ReceiverProposal p;
  ASSERT_TRUE(rcv.ReadProposal(avatar, &p));
  const AnchorProposal* h = Find(p, "hand");
  ASSERT_NE(h, nullptr);
  EXPECT_EQ(h->orientation_type, 2);
  EXPECT_EQ(h->pitch, 0.2);
  EXPECT_EQ(h->roll, -0.1);
  EXPECT_EQ(p.orientation_type, 1) << "the root's is the actor's";
  EXPECT_EQ(p.pitch, 0.05);
}

TEST(AnchorProposals, AFacingAnchorWithoutASampleIsNoSampleAndNeverBlocksTheRoot) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  w.src->Bind(w.source, "head", RelationKernelType::kAvatarFacing);
  uint32_t head = 0;
  ASSERT_TRUE(w.rcv[0]->AnchorIdFor(w.avatar[0], "head", &head));
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  w.src->SetAnchorWorldPose(w.source, "head", Eigen::Vector3d(4, 5, 1.6), 0.0);
  w.Move({4, 5});
  ReceiverProposal p = w.Proposal(0);
  EXPECT_TRUE(p.status == ProposalStatus::kExact || p.status == ProposalStatus::kInexact)
      << "a missing sample on one anchor never suppresses another's proposal";
  const AnchorProposal* h = Find(p, "head");
  ASSERT_NE(h, nullptr);
  EXPECT_EQ(h->anchor_id, head);
  EXPECT_EQ(h->status, ProposalStatus::kNoSample);
  EXPECT_FALSE(h->sampled());

  // The driver publishes the head's pose: the head now has its own proposal,
  // at its own sampled position and height.
  w.Publish(0, head, 1.0, 1.0, 1.55, 0.3);
  w.src->SetAnchorWorldPose(w.source, "head", Eigen::Vector3d(4.1, 5, 1.6), 0.0);
  w.Move({4.1, 5});
  p = w.Proposal(0);
  h = Find(p, "head");
  ASSERT_NE(h, nullptr);
  EXPECT_TRUE(h->sampled());
  EXPECT_TRUE(h->status == ProposalStatus::kExact || h->status == ProposalStatus::kInexact);
  EXPECT_NEAR(h->x, 1.0, 1e-12);
  EXPECT_NEAR(h->y, 1.0, 1e-12);
  EXPECT_NEAR(h->z, 1.55, 1e-12);
}

// The root's course (the geodesic around the receiver's own obstacles, smoothed into the
// Bezier course; Space::QueryCourseUnder / BezierCourse) belongs to the root anchor. A
// non-root anchor does not walk a course: the pelvis proposal is the mapped contact, a
// Facing anchor's proposal is its own sample with the yaw it should have. Two receivers
// whose rooms differ only by a wall across the root's way give the root different
// proposals and every other anchor the same one.
TEST(AnchorProposals, OnlyTheRootAnchorUsesACourseAroundTheReceiversObstacles) {
  World w(2);
  {
    // Receiver 1 only: a wall between the root's sample and the shelf.
    auto* tree = w.rcv[1]->GetElementTree();
    constexpr element_id_t kWall = 20;
    ASSERT_TRUE(tree->AddElement(kWall, kRoom).ok());
    tree->GetElement(kWall)->SetLocalTransform(At(5.0, 5.0));
    tree->GetElement(kWall)->SetSpatialDefinition(
        std::make_shared<PolytopeSpatialDefinition>(
            Eigen::Vector3d(-0.1, -4.0, 0.0), Eigen::Vector3d(0.1, 4.0, 1.0),
            GeometryRole::kObstacle));
  }
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  ASSERT_NE(w.src->BindScoped(w.source, "pelvis",
                              RelationKernelType::kAvatarAffordancePose, kSeat,
                              "Sittable"),
            nullptr);
  w.src->Bind(w.source, "head", RelationKernelType::kAvatarFacing);
  uint32_t pelvis = 0, head = 0;
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(w.rcv[i]->AnchorIdFor(w.avatar[i], "pelvis", &pelvis));
    ASSERT_TRUE(w.rcv[i]->AnchorIdFor(w.avatar[i], "head", &head));
    w.Publish(i, 0, 2.0, 5.0, 0.0, 0.0);      // the root, west of the wall
    w.Publish(i, head, 2.0, 5.0, 1.55, 0.3);  // the head sample
  }
  // The source's pelvis touches the seat and its head faces on, while its root walks
  // toward the shelf (a finite update count); the anchors are posed independently.
  w.src->SetAnchorWorldPose(w.source, "pelvis", Eigen::Vector3d(3.1, 3.05, 0.45), 0.4);
  w.src->SetAnchorWorldPose(w.source, "head", Eigen::Vector3d(3.1, 3.05, 1.6), 0.0);
  w.Move({4.0, 5.0}, 0.4);
  w.Move({4.5, 5.0}, 0.4);
  w.Move({5.0, 5.0}, 0.4);
  const ReceiverProposal a = w.Proposal(0), b = w.Proposal(1);

  // The wall changes the root's way: the course is the root's.
  ASSERT_TRUE(a.status == ProposalStatus::kExact || a.status == ProposalStatus::kInexact);
  ASSERT_TRUE(b.status == ProposalStatus::kExact || b.status == ProposalStatus::kInexact);
  EXPECT_GT((Eigen::Vector2d(a.x, a.y) - Eigen::Vector2d(b.x, b.y)).norm(), 1e-6)
      << "the wall is across the root's way, so the root proposals differ";

  // The pelvis proposal is the mapped contact on the seat: the same in both rooms.
  const AnchorProposal* pa = Find(a, "pelvis");
  const AnchorProposal* pb = Find(b, "pelvis");
  ASSERT_NE(pa, nullptr);
  ASSERT_NE(pb, nullptr);
  ASSERT_TRUE(pa->engaged());
  ASSERT_TRUE(pb->engaged());
  EXPECT_EQ(pa->x, pb->x);
  EXPECT_EQ(pa->y, pb->y);
  EXPECT_EQ(pa->z, pb->z);
  EXPECT_EQ(pa->yaw, pb->yaw);

  // A Facing anchor proposes at its OWN sample, whatever the root's course is.
  const AnchorProposal* ha = Find(a, "head");
  const AnchorProposal* hb = Find(b, "head");
  ASSERT_NE(ha, nullptr);
  ASSERT_NE(hb, nullptr);
  EXPECT_EQ(ha->x, 2.0);
  EXPECT_EQ(ha->y, 5.0);
  EXPECT_EQ(hb->x, 2.0);
  EXPECT_EQ(hb->y, 5.0);
  EXPECT_EQ(ha->z, 1.55);
  EXPECT_EQ(ha->yaw, hb->yaw);
}

TEST(AnchorProposals, EveryAnchorOfASetBelongsToOneRevision) {
  World w(1);
  w.src->Bind(w.source, "root", RelationKernelType::kAvatarSpacing);
  // D6: the tag is the AffordancePose binding (which affordance kind the pelvis binds
  // to); without it every object below the room is a candidate, and the shelf has no seat.
  ASSERT_NE(w.src->BindScoped(w.source, "pelvis",
                              RelationKernelType::kAvatarAffordancePose, kSeat,
                              "Sittable"),
            nullptr);
  w.Publish(0, 0, 1.0, 1.0, 0.0, 0.0);
  SitFrame(w, 0.45);
  ReceiverProposal one, two;
  ASSERT_TRUE(w.rcv[0]->ReadProposal(w.avatar[0], &one));
  SitFrame(w, 0.45);
  ASSERT_TRUE(w.rcv[0]->ReadProposal(w.avatar[0], &two));
  EXPECT_GT(two.revision, one.revision);
  EXPECT_EQ(one.source_sequence + 1, two.source_sequence);
  // The set is a snapshot: reading it again changes nothing.
  ReceiverProposal again;
  ASSERT_TRUE(w.rcv[0]->ReadProposal(w.avatar[0], &again));
  EXPECT_EQ(again.revision, two.revision);
  EXPECT_EQ(again.anchors.size(), two.anchors.size());
}

TEST(AnchorProposals, AnchorIdsAreInternedAndStable) {
  World w(1);
  uint32_t root = 99, a = 0, b = 0, again = 0;
  ASSERT_TRUE(w.rcv[0]->AnchorIdFor(w.avatar[0], "root", &root));
  EXPECT_EQ(root, Space::kRootAnchorId);
  ASSERT_TRUE(w.rcv[0]->AnchorIdFor(w.avatar[0], "pelvis", &a));
  ASSERT_TRUE(w.rcv[0]->AnchorIdFor(w.avatar[0], "head", &b));
  ASSERT_TRUE(w.rcv[0]->AnchorIdFor(w.avatar[0], "pelvis", &again));
  EXPECT_NE(a, root);
  EXPECT_NE(a, b);
  EXPECT_EQ(a, again);
  std::string name;
  ASSERT_TRUE(w.rcv[0]->AnchorNameOf(w.avatar[0], b, &name));
  EXPECT_EQ(name, "head");
  EXPECT_FALSE(w.rcv[0]->AnchorNameOf(w.avatar[0], 4242, &name));
  EXPECT_FALSE(w.rcv[0]->AnchorIdFor(987654, "pelvis", &a));
}

TEST(AnchorProposals, EachAnchorHasItsOwnPortWithItsOwnEpochAndSequence) {
  World w(1);
  uint32_t pelvis = 0;
  ASSERT_TRUE(w.rcv[0]->AnchorIdFor(w.avatar[0], "pelvis", &pelvis));
  EmbodiedPoseInput in;
  in.epoch = 1;
  in.z = 0.45;
  EXPECT_EQ(w.rcv[0]->PublishAnchorPose(w.avatar[0], pelvis, in, 0.0, 1),
            EmbodiedPublishResult::kOk);
  EXPECT_EQ(w.rcv[0]->PublishAnchorPose(w.avatar[0], pelvis, in, 0.0, 1),
            EmbodiedPublishResult::kStale);
  // The root port is independent: sequence 1 is fresh there.
  EXPECT_EQ(w.rcv[0]->PublishAnchorPose(w.avatar[0], Space::kRootAnchorId, in, 0.0, 1),
            EmbodiedPublishResult::kOk);
  EmbodiedSample s;
  ASSERT_TRUE(w.rcv[0]->SampleAnchorPose(w.avatar[0], pelvis, &s));
  EXPECT_TRUE(s.valid);
  EXPECT_NEAR(s.z, 0.45, 1e-12);
  // An id nobody interned is invalid, not a silent new port.
  EXPECT_EQ(w.rcv[0]->PublishAnchorPose(w.avatar[0], 77, in, 0.0, 1),
            EmbodiedPublishResult::kInvalid);
  // A non-finite component is refused.
  in.z = std::nan("");
  EXPECT_EQ(w.rcv[0]->PublishAnchorPose(w.avatar[0], pelvis, in, 0.0, 2),
            EmbodiedPublishResult::kInvalid);
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
