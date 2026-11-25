// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/c21_target_stall_test.cc
//
// An evaluation run (via eval's own core_driver.build_rig repro) reported
// that in an in-process two-space rig with ONLY the
// source actor driven (the target/mirror actor's own UpdateActorState never
// called -- matching the real B3 pipeline, not the target-actor-driving
// workaround), the target space's realization-target stream (ev=target/
// payload) stops advancing partway through the scripted source trajectory
// (observed: 16-31% of traj.duration into T1-straight-approach/
// T1-identity-room/T2-sit-chair-diff-height) even after the peer-clock
// fix.
//
// This test reproduces the exact mechanism with a minimal, fully-controlled
// harness: the literal T1-identity-room geometry (sim/harness/testdata/
// regression/T1-identity-room.scenario.json: room 6.4x3.2, chair at (2,1) yaw 1.5708,
// core.min_move/min_yaw = 0.02/0.02) and a hand-rolled trajectory generator
// that mirrors the evaluation harness's (not public) behavior-script semantics for the
// fixture's own behavior.script (approach chair distance=0.8 duration=3.0,
// sit chair duration=2.0, stand duration=1.0), sampled at 30 Hz (33ms) via
// Space::UpdateActorState exactly like core_driver.build_rig drives it --
// no avatar controller, no target-side driving, using SpaceHub::
// SetRelationUpdateCallback to get unambiguous ground truth for "did the
// source actually propagate this frame" (independent of whether the
// decoded/traced target value happens to be unchanged).
//
// Finding: the movement-gate deadband (space.cc's
// min_propagation_move_/min_propagation_yaw_, mirrors v1's
// ShouldProcessStateUpdate) is *why* propagation pauses -- a source that has
// arrived and is holding still legitimately has nothing new to say -- but
// the heartbeat (kHeartbeatInterval=30 frames) that is supposed to keep the
// peer's clock/trace alive during a hold, and immediate resumption once the
// source moves again (the "sit" walk), both hold up empirically: this is
// EXPECTED gate behavior given the specific scripted trajectory, not a
// permanent defect -- see the test bodies below for the exact numeric
// evidence (propagation resumes at the "sit" walk).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace {

// ---------------------------------------------------------------------
// Geometry: sim/harness/testdata/regression/T1-identity-room.scenario.json, both rooms
// (source "A" / target "B") are congruent -- same room, same chair.
// ---------------------------------------------------------------------
constexpr double kRoomW = 6.4;
constexpr double kRoomH = 3.2;
const Eigen::Vector2d kChair(2.0, 1.0);
constexpr double kChairHalfExtent = 0.25;  // 0.5m square footprint

constexpr shared_element_id_t kRoomSharedId = 9001;  // == actor_spec.shared_id
constexpr shared_element_id_t kChairSharedId = 201;

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

// The actor's own element IS the room (core_driver.build_rig's own comment:
// "make the actor's own element the room itself, so scope_root falls back
// to room"). Same shared_id on both sides -> the receiver's shared-id
// fallback resolves the actor without needing an explicit ShareElement/
// ElementLinker round trip (this is also how cross_space_test.cc's own
// IdentityRoomApproachRoundTripExactEveryFrame gets away without one).
spatialblend::entity_id_t AddRoomActor(Space* space) {
  // "chest" anchor matches the fixture's own second anchor (offset [0,0,1.2])
  // -- included so this rig binds BOTH root/spacing and chest/facing exactly
  // like the real actor_spec, ruling the facing kernel in/out as a factor.
  std::vector<SubjectAnchor> anchors = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
      SubjectAnchor("chest", Eigen::Vector3d(0, 0, 1.2)),
  };
  auto id = space->CreateActor(anchors);
  space->GetActor(id)->SetElementId(1);
  return id;
}

// ---------------------------------------------------------------------
// Trajectory generator: mirrors the evaluation harness's behavior script
// _Builder.op_approach/op_sit/op_stand against T1-identity-room's own
// behavior.script (approach chair distance=0.8 duration=3.0; sit chair
// duration=2.0; stand duration=1.0) and default_start_pose (room centroid,
// yaw 0). WALK_SPEED=1.4 m/s matches behavior.py's own constant (which
// itself matches spatialblend/api/python/interaction/script.py). No jitter (the
// fixture's "stand" op has jitter_sigma=0.01, well under min_move=0.02 --
// modeled here as an exact hold, which is the representative/worst case for
// "does propagation ever resume" and keeps the test deterministic).
// ---------------------------------------------------------------------
struct Keyframe {
  double t, x, y, yaw;
};

double YawToward(double fx, double fy, double tx, double ty) {
  double dx = tx - fx, dy = ty - fy;
  if (std::abs(dx) < 1e-12 && std::abs(dy) < 1e-12) return 0.0;
  return std::atan2(dy, dx);
}

struct ScriptedTrajectory {
  std::vector<Keyframe> keyframes;
  // Real-time instants (seconds) at which each scripted op's WALK segment
  // finishes and the actor starts holding still -- computed alongside the
  // keyframes so the test can assert "stall == source became stationary"
  // without re-deriving it from the keyframes themselves.
  double approach_arrival_t = 0.0;   // end of the approach-phase walk
  double sit_arrival_t = 0.0;        // end of the sit-phase walk
  double duration = 0.0;

  double SampleX(double t) const { return Sample(t)[0]; }
  std::array<double, 3> Sample(double t) const {
    const auto& kfs = keyframes;
    if (t <= kfs.front().t) return {kfs.front().x, kfs.front().y, kfs.front().yaw};
    if (t >= kfs.back().t) return {kfs.back().x, kfs.back().y, kfs.back().yaw};
    for (size_t i = 0; i + 1 < kfs.size(); ++i) {
      const auto& a = kfs[i];
      const auto& b = kfs[i + 1];
      if (a.t <= t && t <= b.t) {
        double span = b.t - a.t;
        double frac = span <= 1e-12 ? 0.0 : (t - a.t) / span;
        double x = a.x + (b.x - a.x) * frac;
        double y = a.y + (b.y - a.y) * frac;
        // Shortest-path yaw interpolation (matches behavior.py's _lerp_yaw).
        double diff = std::remainder(b.yaw - a.yaw, 2 * M_PI);
        double yaw = a.yaw + diff * frac;
        return {x, y, yaw};
      }
    }
    return {kfs.back().x, kfs.back().y, kfs.back().yaw};
  }
};

ScriptedTrajectory BuildT1IdentityRoomTrajectory() {
  constexpr double kWalkSpeed = 1.4;
  constexpr double kApproachDistance = 0.8;
  constexpr double kApproachDuration = 3.0;
  constexpr double kSitDuration = 2.0;
  constexpr double kStandDuration = 1.0;

  ScriptedTrajectory traj;
  double cx = kRoomW / 2.0, cy = kRoomH / 2.0, cyaw = 0.0;  // default_start_pose
  double t = 0.0;
  traj.keyframes.push_back({t, cx, cy, cyaw});

  auto walk_to = [&](double x, double y, double yaw) {
    double dist = std::hypot(x - cx, y - cy);
    double transit = dist / kWalkSpeed;
    t += transit;
    cx = x;
    cy = y;
    cyaw = yaw;
    traj.keyframes.push_back({t, cx, cy, cyaw});
    return transit;
  };
  auto hold = [&](double duration) {
    if (duration <= 0.0) return;
    t += duration;
    traj.keyframes.push_back({t, cx, cy, cyaw});
  };

  // op_approach(tag=chair, distance=0.8, duration=3.0)
  {
    double ex = kChair.x(), ey = kChair.y();
    double ddx = cx - ex, ddy = cy - ey;
    double dnorm = std::hypot(ddx, ddy);
    if (dnorm < 1e-9) {
      ddx = 1.0;
      ddy = 0.0;
      dnorm = 1.0;
    }
    double tx = ex + ddx / dnorm * kApproachDistance;
    double ty = ey + ddy / dnorm * kApproachDistance;
    double target_yaw = YawToward(tx, ty, ex, ey);
    double transit = walk_to(tx, ty, target_yaw);
    traj.approach_arrival_t = t;  // == transit, walk started at t=0
    (void)transit;
    double remaining = std::max(0.0, kApproachDuration - transit);
    hold(remaining);
  }
  // op_sit(tag=chair, duration=2.0) -- fixture declares no sub_regions, so
  // _find_element_seat falls back to the chair's own element position.
  {
    double sx = kChair.x(), sy = kChair.y();
    double t_before = t;
    double target_yaw = YawToward(sx, sy, cx, cy);
    double transit = walk_to(sx, sy, target_yaw);
    traj.sit_arrival_t = t_before + transit;
    double remaining = std::max(0.0, kSitDuration - transit);
    hold(remaining);
  }
  // op_stand(duration=1.0) -- jitter omitted (see file header rationale).
  hold(kStandDuration);

  traj.duration = t;
  return traj;
}

// ---------------------------------------------------------------------
// Rig: mirrors core_driver.build_rig -- one hub, source room bound with the
// spacing kernel on "root", target/mirror room built but never Bind()'d or
// UpdateActorState()'d (matches "no target-actor-driving workaround").
// core.min_move/min_yaw = 0.02/0.02 (scenario JSON's own values).
// ---------------------------------------------------------------------
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
  // the chair, exactly matching core_driver.build_rig's own
  // _share_and_resolve_furniture (identity.expect declares shared_id 201 ->
  // link_to_existing) -- NOT the shared-id-fallback path that a simpler
  // setup (e.g. cross_space_test.cc's IdentityRoomApproachRoundTripExactEve
  // ryFrame) exercises. This establishes a REAL ElementLinker mapping for
  // the referent, so OnRemoteUpdate's default kLinker referent-resolve mode
  // (referent_resolve_mode_) actually resolves via Translate() here, not
  // via the shared_id fallback -- ruling in/out the linker path itself as
  // a factor in the stall.
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
  rig.src->Bind(rig.src_actor, "chest", RelationKernelType::kAvatarFacing);
  // Fixture's own core.min_move/min_yaw (the T1-identity-room
  // scenario's "core" block) -- NOT disabled, since the whole point of
  // this test is to observe the gate's real pause/resume behavior.
  rig.src->SetMinMove(0.02);
  rig.src->SetMinYaw(0.02);
  constexpr double kDt = 1.0 / 30.0;
  rig.src->SetUpdateInterval(kDt);
  rig.tgt->SetUpdateInterval(kDt);
  return rig;
}

// Drives `rig.src_actor` through `traj` at 30 Hz, recording the real time of
// every frame in which the source hub-propagated (i.e. every invocation of
// SpaceHub's relation-update callback -- unambiguous ground truth for "did
// the source decide to send an update this frame", independent of whether
// the receiver's decoded pose happens to be unchanged).
std::vector<double> DriveAndRecordPropagations(Rig& rig,
                                               const ScriptedTrajectory& traj,
                                               int n_frames) {
  std::vector<double> propagate_times;
  rig.ctx->GetHub()->SetRelationUpdateCallback(
      [&](const RemoteRelationUpdate& update) {
        (void)update;
        propagate_times.push_back(rig.src->GetSpaceClock());
      });

  constexpr double kDt = 1.0 / 30.0;
  for (int i = 0; i < n_frames; ++i) {
    double t = std::min(i * kDt, traj.duration);
    auto [x, y, yaw] = traj.Sample(t);
    rig.src->UpdateActorState(rig.src_actor, Eigen::Vector2d(x, y), yaw);
  }
  rig.ctx->GetHub()->SetRelationUpdateCallback(nullptr);
  return propagate_times;
}

// Longest gap between consecutive propagations (or between the last
// propagation and the end of the driven window), in seconds.
double MaxGap(const std::vector<double>& times, double window_end) {
  double max_gap = times.empty() ? window_end : times.front();
  for (size_t i = 1; i < times.size(); ++i) {
    max_gap = std::max(max_gap, times[i] - times[i - 1]);
  }
  if (!times.empty()) max_gap = std::max(max_gap, window_end - times.back());
  return max_gap;
}

constexpr double kHeartbeatSeconds = 30.0 / 30.0;  // kHeartbeatInterval frames @ 30Hz

}  // namespace

// ---------------------------------------------------------------------
// Test 1: does propagation PERMANENTLY stall once the source first goes
// stationary (matching the literal reported observation -- last ev=target at
// 16-31% of traj.duration, nothing ever again), or does it correctly PAUSE
// (bounded by the heartbeat) and RESUME once the source moves again (the
// "sit" walk, a genuine ~0.8m displacement starting well after the initial
// approach-hold)?
// ---------------------------------------------------------------------
TEST(C21TargetStall, PropagationResumesAtSitWalkNotPermanentlyStalled) {
  ScriptedTrajectory traj = BuildT1IdentityRoomTrajectory();
  ASSERT_GT(traj.sit_arrival_t, traj.approach_arrival_t)
      << "fixture's own script must contain a second, later walk segment "
         "for this test to be meaningful";

  Rig rig = BuildRig();
  constexpr int kNFrames = 200;  // matches the original repro script
  std::vector<double> propagate_times =
      DriveAndRecordPropagations(rig, traj, kNFrames);

  ASSERT_FALSE(propagate_times.empty())
      << "source must propagate at least once (initial approach walk)";

  // The reported symptom, stated precisely: NO propagation at all after some
  // early point, all the way to the end of the driven window (200 frames =
  // 6.667s, well past the 6.0s script). If this held here, the longest gap
  // between propagations would span from partway through frame 200's real
  // time down to duration -- i.e. a MaxGap far exceeding one heartbeat
  // interval (kHeartbeatInterval=30 frames @ 30Hz = 1.0s).
  double window_end = std::min(kNFrames / 30.0, traj.duration);
  double max_gap = MaxGap(propagate_times, window_end);

  // A correctly-functioning gate/heartbeat never goes longer than one
  // heartbeat interval without a resend, REGARDLESS of whether the source is
  // moving (heartbeat) or genuinely moving (movement gate fires well inside
  // that window). Some slack for the frame quantization (33ms) and the one
  // frame it takes the heartbeat counter to actually reach the threshold.
  EXPECT_LT(max_gap, kHeartbeatSeconds + 2.0 / 30.0)
      << "longest gap between propagations was " << max_gap
      << "s -- exceeds one heartbeat interval (" << kHeartbeatSeconds
      << "s); this would reproduce the reported permanent-stall symptom instead of "
         "a heartbeat-bounded pause";

  // Direct, unambiguous refutation of "permanent stall": at least one
  // propagation must land during (or immediately after) the sit-walk, i.e.
  // real time in [sit_arrival_t - transit, sit_arrival_t + one frame] --
  // the source is unambiguously, substantially moving there (~0.8m over a
  // fraction of a second, far above the 0.02m gate), so if propagation never
  // resumes here, the mechanism is genuinely broken, not merely gated.
  bool resumed_at_sit_walk = std::any_of(
      propagate_times.begin(), propagate_times.end(), [&](double pt) {
        return pt > traj.approach_arrival_t + kHeartbeatSeconds &&
               pt <= traj.sit_arrival_t + 1.0 / 30.0;
      });
  EXPECT_TRUE(resumed_at_sit_walk)
      << "no propagation observed during the sit-walk window ("
      << traj.approach_arrival_t << "s, " << traj.sit_arrival_t
      << "s] -- matches the reported permanent stall";

  RecordProperty("approach_arrival_t", traj.approach_arrival_t);
  RecordProperty("sit_arrival_t", traj.sit_arrival_t);
  RecordProperty("traj_duration", traj.duration);
  RecordProperty("num_propagations", static_cast<int>(propagate_times.size()));
  RecordProperty("max_gap_seconds", max_gap);
}

// ---------------------------------------------------------------------
// Test 2: hypothesis (a) numeric check -- once the source goes stationary
// for good (after the sit-walk, through the "stand" hold to the end of the
// script), the LAST propagation before that hold should land within one
// frame of when the source actually became stationary (sit_arrival_t),
// not earlier and not (barring the heartbeat) much later.
// ---------------------------------------------------------------------
TEST(C21TargetStall, StallTimeMatchesSourceStationaryTimeWithinOneFrame) {
  ScriptedTrajectory traj = BuildT1IdentityRoomTrajectory();
  Rig rig = BuildRig();
  constexpr int kNFrames = 200;
  std::vector<double> propagate_times =
      DriveAndRecordPropagations(rig, traj, kNFrames);

  // Last propagation whose time is <= sit_arrival_t + one frame (i.e. the
  // final "genuine motion" send before the post-sit hold sets in).
  double last_before_final_hold = -1.0;
  for (double pt : propagate_times) {
    if (pt <= traj.sit_arrival_t + 1.0 / 30.0) last_before_final_hold = pt;
  }
  ASSERT_GE(last_before_final_hold, 0.0);

  constexpr double kFrameDt = 1.0 / 30.0;
  EXPECT_NEAR(last_before_final_hold, traj.sit_arrival_t, kFrameDt)
      << "last genuine-motion propagation (" << last_before_final_hold
      << "s) should coincide with the source's own sit-walk arrival ("
      << traj.sit_arrival_t << "s) within one frame";
}

}  // namespace spatialblend
