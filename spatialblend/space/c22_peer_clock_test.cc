// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/c22_peer_clock_test.cc
//
// Space::OnRemoteUpdate previously advanced the peer/receiver's own
// space_clock_ by RemoteRelationUpdate::step_duration, which Space::
// UpdateActorState (source side) always sets to the constant
// update_interval_ (~0.033s at 30Hz) on EVERY call -- regardless of whether
// this particular propagation followed a genuine per-frame move or was a
// heartbeat-forced resend after the movement-gate deadband (min_move/
// min_yaw) held the source still for up to kHeartbeatInterval=30 frames
// (~1.0s). Result: the peer
// clock (and hence every ev=target/payload/relation/tick "t" it stamps, and
// every RelationPhaseTracker timestamp) advanced by delivery-count *
// update_interval_ instead of real elapsed time -- badly under-running it
// across any heartbeat gap. The Python (core_driver.build_rig) numeric
// repro: T1-identity-room, 6.0s real trajectory, 39
// deliveries -> peer "t" range [0.033, 1.287] (== 39 * 0.033), i.e. the peer
// clock ends at ~21% of real time.
//
// Fix: RemoteRelationUpdate now also carries source_clock (the source's own
// space_clock_ at propagate time, remote_update.h). OnRemoteUpdate advances
// the receiver's space_clock_ by the REAL elapsed source time -- the delta
// between this delivery's source_clock and the last one seen for this
// (source_space_id, actor_shared_id) key -- falling back to step_duration
// only when source_clock can't yield a valid delta (unset, first delivery,
// or non-monotonic). See space.cc's OnRemoteUpdate for the full comment.
//
// This test reproduces the mechanism directly: the same identity-room +
// shared/linked chair-referent rig c21_target_stall_test.cc uses (so the
// real ElementLinker referent-resolution path -- not merely a same-shared-
// id fallback -- is genuinely exercised), only the SOURCE actor driven via
// UpdateActorState, with an explicit walk / hold-2.0s / walk schedule so the
// exact real-time gap across the movement-gate hold is known precisely and
// can be checked against the peer's own clock and trace "t" values.
#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "spatialblend/base/common/trace.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace {

// ---------------------------------------------------------------------
// Geometry: identical to c21_target_stall_test.cc's BuildIdentityRoom/
// AddRoomActor (sim/harness/testdata/regression/T1-identity-room.scenario.json: room
// 6.4x3.2, chair at (2,1), half-extent 0.25) -- duplicated here (rather
// than shared via a header) to keep this a standalone file, matching the
// existing per-task regression-test convention (see c21_target_stall_test.
// cc's own file header for the rationale).
// ---------------------------------------------------------------------
constexpr double kRoomW = 6.4;
constexpr double kRoomH = 3.2;
const Eigen::Vector2d kChair(2.0, 1.0);
constexpr double kChairHalfExtent = 0.25;

constexpr shared_element_id_t kRoomSharedId = 9001;
constexpr shared_element_id_t kChairSharedId = 201;
constexpr double kDt = 1.0 / 30.0;  // 30Hz, matches the fixture's own rate.
constexpr double kMinMove = 0.02;   // fixture's own core.min_move.
constexpr double kMinYaw = 0.02;    // fixture's own core.min_yaw.

void BuildIdentityRoom(Space* space) {
  auto* tree = space->GetElementTree();
  tree->AddElement(1);
  tree->GetElement(1)->SetSharedId(kRoomSharedId);
  tree->GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
              {0, 0}, {kRoomW, 0}, {kRoomW, kRoomH}, {0, kRoomH}})));

  tree->AddElement(2, 1);
  tree->GetElement(2)->SetSharedId(kChairSharedId);
  tree->GetElement(2)->SetTag("chair");
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = kChair.x();
  t(1, 3) = kChair.y();
  tree->GetElement(2)->SetLocalTransform(t);
  tree->GetElement(2)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-kChairHalfExtent, -kChairHalfExtent, -1.0),
          Eigen::Vector3d(kChairHalfExtent, kChairHalfExtent, 1.0)));
}

// The actor's own element IS the room (matches core_driver.build_rig /
// c21_target_stall_test.cc): same shared_id on both sides means the
// receiver's shared-id fallback would resolve the actor even without a
// linker round trip, but the chair REFERENT below still gets a real
// ShareElement/ResolveSharedElement round trip so OnRemoteUpdate's default
// kLinker referent-resolve mode is genuinely exercised for it.
spatialblend::entity_id_t AddRoomActor(Space* space) {
  std::vector<SubjectAnchor> anchors = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
  };
  auto id = space->CreateActor(anchors);
  FollowProposals(space, id);
  space->GetActor(id)->SetElementId(1);
  return id;
}

struct Rig {
  std::unique_ptr<LocalContext> ctx;
  Space* src = nullptr;
  Space* tgt = nullptr;
  spatialblend::entity_id_t src_actor = 0;
  spatialblend::entity_id_t tgt_actor = 0;
};

Rig BuildRig() {
  Rig rig;
  rig.ctx = LocalContext::Create();
  rig.src = rig.ctx->AddSpace();
  rig.tgt = rig.ctx->AddSpace();
  BuildIdentityRoom(rig.src);
  BuildIdentityRoom(rig.tgt);
  rig.src_actor = AddRoomActor(rig.src);
  rig.tgt_actor = AddRoomActor(rig.tgt);

  // Real ShareElement/ResolveSharedElement (LinkToExisting) round trip for
  // the chair referent (matches core_driver.build_rig's own
  // _share_and_resolve_furniture / c21_target_stall_test.cc's BuildRig).
  {
    element_id_t received = kInvalidElementId;
    rig.tgt->SetOnElementAdded(
        [&](element_id_t id, shared_element_id_t /*shared*/,
            const std::string& /*tag*/, const std::string& /*user_data*/) {
          received = id;
        });
    rig.src->ShareElement(/*elem_id=*/2, /*user_data=*/"");
    if (received == kInvalidElementId) {
      ADD_FAILURE() << "share() of the chair produced no on_element_added "
                       "event in the target space";
    } else {
      rig.tgt->ResolveSharedElement(received, /*resolved_id=*/2);
    }
    rig.tgt->SetOnElementAdded(nullptr);
  }

  rig.src->Bind(rig.src_actor, "root", RelationKernelType::kAvatarSpacing);
  rig.src->SetMinMove(kMinMove);
  rig.src->SetMinYaw(kMinYaw);
  rig.src->SetUpdateInterval(kDt);
  rig.tgt->SetUpdateInterval(kDt);
  return rig;
}

// Reads every "ev":"target" line's "t" field for `space_id` from a trace
// JSONL file, in file order.
std::vector<double> ReadTargetLineTimes(const std::string& path,
                                        int64_t space_id) {
  std::vector<double> times;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    nlohmann::json ev;
    try {
      ev = nlohmann::json::parse(line);
    } catch (const std::exception&) {
      continue;
    }
    if (ev.value("ev", "") != "target") continue;
    if (ev.value("space", int64_t{-1}) != space_id) continue;
    times.push_back(ev.value("t", -1.0));
  }
  return times;
}

}  // namespace

// ---------------------------------------------------------------------
// Regression (a): source-only driven,
// stationary for 2.0s in the middle of the run -> the peer's own clock at
// the end of the run must be within one frame (kDt) of the source's own
// clock, and every peer ev=target trace line's "t" must be within one frame
// of the corresponding source propagate time (the source's own space_clock_
// at the moment of that delivery, recorded independently via SpaceHub's
// relation-update callback -- unambiguous ground truth, matching
// c21_target_stall_test.cc's own DriveAndRecordPropagations pattern).
// ---------------------------------------------------------------------
TEST(C22PeerClock, PeerClockTracksRealElapsedSourceTimeAcrossHeartbeatGap) {
  TraceSink::ResetAllForTest();
  Rig rig = BuildRig();

  const std::string trace_path =
      ::testing::TempDir() + "/c22_peer_clock_trace.jsonl";
  std::remove(trace_path.c_str());
  TraceSink::SetPathForSpace(rig.tgt->GetSpaceId(), trace_path);

  std::vector<double> src_propagate_clock;
  rig.ctx->GetHub()->SetRelationUpdateCallback(
      [&](const RemoteRelationUpdate& /*update*/) {
        src_propagate_clock.push_back(rig.src->GetSpaceClock());
      });

  // Path fully inside the room (0,0)-(6.4,3.2), well clear of the chair's
  // box ([1.75,2.25] x [0.75,1.25]) at y=2.5, so the referent stays
  // resolvable (single child of the room) throughout without special-
  // casing an in-box overlap. Each step (0.03m) exceeds min_move (0.02m),
  // so every WALK frame passes the movement gate and genuinely propagates.
  Eigen::Vector2d pos(1.0, 2.5);
  const double yaw = 0.0;
  const Eigen::Vector2d step(0.03, 0.0);

  // Walk phase 1: 30 frames (1.0s) of genuine motion.
  for (int i = 0; i < 30; ++i) {
    pos += step;
    rig.src->UpdateActorState(rig.src_actor, pos, yaw);
  }
  // Hold phase: EXACTLY 60 frames (2.0s) stationary -> exactly two forced
  // heartbeat resends (kHeartbeatInterval=30 frames each).
  for (int i = 0; i < 60; ++i) {
    rig.src->UpdateActorState(rig.src_actor, pos, yaw);
  }
  // Walk phase 2: another 30 frames (1.0s) of genuine motion.
  for (int i = 0; i < 30; ++i) {
    pos += step;
    rig.src->UpdateActorState(rig.src_actor, pos, yaw);
  }

  rig.ctx->GetHub()->SetRelationUpdateCallback(nullptr);
  TraceSink::ForSpace(rig.tgt->GetSpaceId()).Disable();

  constexpr int kTotalFrames = 30 + 60 + 30;
  const double expected_final_clock = kTotalFrames * kDt;  // == 4.0s

  // Source's own clock is unconditionally real-time-accurate (advances by
  // update_interval_ on every UpdateActorState call regardless of the
  // movement gate) -- this is the ground truth c21_target_stall_test.cc already established.
  EXPECT_NEAR(rig.src->GetSpaceClock(), expected_final_clock, 1e-9);

  // The actual regression: the PEER's clock must track the same real time,
  // not delivery-count * update_interval_ (which, pre-fix, would end at
  // roughly (30 + 2 + 30) * kDt ~= 2.067s -- a ~1.93s / 48% shortfall from
  // the 4.0s of real time actually elapsed).
  EXPECT_NEAR(rig.tgt->GetSpaceClock(), rig.src->GetSpaceClock(), kDt)
      << "peer clock must track real elapsed source time within one frame, "
         "not delivery-count * update_interval_ (the earlier behavior)";
  EXPECT_NEAR(rig.tgt->GetSpaceClock(), expected_final_clock, kDt);

  // Every peer ev=target line's "t" must match its corresponding source
  // propagate time within one frame -- including the two heartbeat
  // deliveries that each cross a real ~1.0s gap.
  std::vector<double> peer_target_times =
      ReadTargetLineTimes(trace_path, rig.tgt->GetSpaceId());
  ASSERT_FALSE(peer_target_times.empty());
  ASSERT_FALSE(src_propagate_clock.empty());
  ASSERT_EQ(peer_target_times.size(), src_propagate_clock.size())
      << "one ev=target line expected per source propagation (root anchor "
         "only, single binding)";
  for (size_t i = 0; i < peer_target_times.size(); ++i) {
    EXPECT_NEAR(peer_target_times[i], src_propagate_clock[i], kDt)
        << "peer trace \"t\" at delivery #" << i
        << " does not match the source's own propagate-time clock within "
           "one frame";
  }

  // The two heartbeat-forced deliveries (crossing the 2.0s hold) must each
  // show a real ~1.0s gap in the peer's own accounted time -- not the
  // pre-fix ~0.033s a fixed step_duration would have produced.
  ASSERT_GE(peer_target_times.size(), 32u);
  const double heartbeat1_gap = peer_target_times[30] - peer_target_times[29];
  const double heartbeat2_gap = peer_target_times[31] - peer_target_times[30];
  EXPECT_NEAR(heartbeat1_gap, 30.0 * kDt, kDt);
  EXPECT_NEAR(heartbeat2_gap, 30.0 * kDt, kDt);

  TraceSink::ResetAllForTest();
}

}  // namespace spatialblend
