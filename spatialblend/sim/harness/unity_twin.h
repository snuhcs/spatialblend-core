// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_HARNESS_UNITY_TWIN_H_
#define SPATIALBLEND_HARNESS_UNITY_TWIN_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "spatialblend/api/c/spatialblend_type.h"
#include "spatialblend/sim/harness/scenario.h"

// Unity-twin harness. Drives a scenario end-to-end through the
// STABLE C API (spatialblend/api/c/spatialblend.h) ONLY -- exactly as a real
// Unity client (P/Invoke) would -- so this exercises the real C-ABI surface
// and the real production Space solver, rather than the in-process C++
// Space/LocalContext API that scenario.cc's BuildSpaces() uses (that
// helper is NOT reused here; unity_twin.cc has its own from-scratch
// element/actor construction that mirrors it, but issued as sb_* calls).
//
// Design notes:
//  - One shared sb_session_t for the whole scenario, with every scenario
//    space added to it via sb_add_space -- deliberately NOT one
//    sb_session_t + sb_host_space/sb_connect_space per space. sb_add_space
//    wires every space in a session to that session's single SpaceHub, and
//    sb_host_space/sb_connect_space each overwrite that ONE hub's
//    single-slot element/relation callbacks -- calling both against a
//    2+-space session clobbers propagation (and crashes under the
//    bazel-test sandbox). This matches the precedent set by the regression
//    corpus itself (space/cross_space_test.cc) and by
//    CApiE2ETest.SourceToTargetCallback in c/c_api_test.cc, both of which
//    drive cross-space linking through a single shared session/hub with NO
//    host/connect calls at all.
//  - Peer-side "committed referent" is observed via
//    sb_register_on_relation_event, NOT sb_get_relation_states:
//    Space::GetRelationStates() only reports on actors that have had a
//    local Bind()/sb_bind() call, and (matching the SourceToTargetCallback
//    precedent) the harness never binds a peer/target avatar. The
//    relation-event callback fires from Space::OnRemoteUpdate's
//    EmitRelationTransitions call regardless of local bindings, so it is
//    the only observation path that works for an unbound peer avatar.
//    "Committed" == a referent whose relation phase has reached at least
//    SB_PHASE_ACTIVE (see BestCommittedReferent() in unity_twin.cc).
//  - Relation events report the referent using the INCOMING payload's
//    source-side shared_id, not the resolved local element's own shared_id,
//    even when resolution happened through the ElementLinker (mismatched
//    shared ids, link_to_existing). unity_twin.cc builds a
//    source_shared_to_local_shared translation table during the
//    share/resolve loop and applies it before recording a referent's phase,
//    matching how cross_space_test.cc's CornerRepro.MismatchedIdsLinker*
//    cases translate through the linker rather than trusting the raw
//    shared_id.
//  - Networking settle waits (element-share propagation, per-frame
//    realization-target propagation) use short bounded polling against
//    callback-set flags, NOT the network_test.cc convention of fixed
//    300-600ms sleeps -- see unity_twin.cc's WaitForCallback().
//  - latency_ms/jitter_ms/drop_rate/reorder are implemented as a harness-
//    level delay queue (NetworkQueue in unity_twin.cc) governing WHEN the
//    harness itself calls sb_update_actor_state on the source actor, since
//    sb_update_actor_state is fire-and-forget with no gRPC-level
//    delay-injection hook exposed by the C API, and there is no real gRPC
//    hop to inject into given the single-shared-session architecture above.
//  - Only the behavior.script-driven actor's motion is scripted; any other
//    actor in the same source space holds its "start" pose (see
//    ScenarioActor::start_x/y/yaw) for the run's duration.
//  - trace.jsonl determinism: sb_space_t / local sb_element_t handles are
//    allocated from counters global to the whole process, not reset per
//    session, so two RunScenario() calls in the same test binary will
//    naturally emit different absolute "space"/"received"/"resolved"
//    numbers in the trace even though the simulation itself is fully
//    deterministic. harness_test.cc's determinism test normalizes those
//    fields (canonical remap in order of first appearance) before comparing
//    two runs' traces for equality; see NormalizeTraceContent() there.
namespace spatialblend {
namespace harness {

// Transport selection for RunScenario. kInProcess is the
// original behavior documented above (one shared sb_session_t/SpaceHub,
// no sb_host_space/sb_connect_space at all). kGrpcTwoSessions instead gives
// the scenario's source space and every other (peer) space their OWN
// sb_session_t (hence their own SpaceHub), and wires them together for real
// over a loopback TCP port via sb_host_space (source) / sb_connect_space
// (every peer) -- exercising the actual gRPC proto-serialize/CQ-thread path
// that a real two-process Unity host+client pair uses, which kInProcess
// cannot reach. See unity_twin.cc's RunScenario for the ordering rationale
// (host+connect happens BEFORE any element/actor is built, so both sides
// start genuinely blank and the bootstrap-clones-host-tree step -- see the
// sb_connect_space deviation note above -- is a harmless no-op instead of
// colliding with this harness's independently-authored per-space geometry).
// latency_ms/jitter_ms/drop_rate/reorder are STILL implemented as the
// harness-level NetworkQueue delay queue in both transports -- the C API
// exposes no wire-level delay-injection hook on the real gRPC client
// (sb_update_actor_state is fire-and-forget), so kGrpcTwoSessions changes
// which transport carries the (already-delayed) call, not how the delay
// itself is produced. Single-space scenarios (no cross-space networking
// needed at all) behave identically under both transport values.
//
// kGrpcTwoSessions was ORIGINALLY found NOT bit-for-bit
// deterministic across two identically-seeded runs, unlike kInProcess -- two
// independent causes, both found via harness_test.cc's
// DeterministicTraceAcrossTwoRuns run with --runs_per_test=10:
//  1. trace.jsonl LINE ORDER: a peer/measure space's own Space::space_clock_
//     (the "t" field on every trace line -- see space.h) only ever advanced
//     inside Space::UpdateActorState, which the measure actor never calls
//     (it's driven purely through TickAvatar/AvatarController) -- so every
//     "share"/"payload"/"target"/"tick" line Space::OnRemoteUpdate emitted on
//     that peer carried "t":0.0 for the entire run. Those lines are appended
//     from that peer's OWN gRPC completion-queue thread (SubscribeStreamCall
//     ::Proceed(), see remote_hub.h),
//     concurrently with the driven/source space's "src"/"relation" lines
//     appended from the harness's main thread -- both streams share one
//     mutex (TraceSink::Emit, common/trace.h), but which thread's line lands
//     first on any given real-time instant when BOTH claim "t":0.0 is
//     genuine OS thread-scheduling nondeterminism.
//  2. An occasional (observed roughly 1-in-10 runs) EXTRA duplicate
//     Space::OnRemoteUpdate call on the peer for the very FIRST delivered
//     update only, right after sb_connect_space's bootstrap/Subscribe-
//     stream-settle window (RunScenario's 100ms sleep in unity_twin.cc).
//     Confirmed genuinely a duplicate INVOCATION (not just a duplicate
//     trace line): the corresponding "ev":"relation" lines did NOT double
//     (RelationPhaseTracker.Update only emits on an actual phase
//     transition, so redelivering identical content is a silent no-op
//     there) while "payload"/"target"/"tick" (emitted unconditionally every
//     OnRemoteUpdate call) did double.
//
// Both root causes are fixed at their source instead of only
// papering over them in the trace-comparison harness:
//  1. Space::OnRemoteUpdate (space.cc) now advances space_clock_ too (using
//     the incoming update's own step_duration, falling back to this space's
//     update_interval_), so a peer's "t" is real and monotonically
//     increasing instead of stuck at 0.0 -- see space.h's space_clock_ doc.
//     (harness_test.cc's NormalizeTraceContent still stable-sorts same-"t"
//     lines by a content key as defense in depth, but no longer needs to
//     paper over an entire stream of colliding "t":0.0 lines.)
//  2. RemoteRelationUpdate now carries a per-(source space, source actor)
//     monotonically increasing `sequence` (remote_update.h), stamped once by
//     Space::UpdateActorState right before hub_->Propagate() and carried
//     unchanged through every relay hop. Space::OnRemoteUpdate drops an
//     incoming update whose sequence is <= the last one already applied for
//     that (source_space_id, actor_shared_id) key -- an exact-replay
//     duplicate delivery is now a no-op instead of a second real invocation.
// A THIRD issue was found while verifying the above (unrelated to either
// root cause, but it independently made any Grpc-vs-InProcess line-count
// comparison meaningless until fixed): RunScenario's frame loop has no real
// per-frame wall-clock delay, so a scenario that propagates ~90 updates
// could run to completion, and SessionGuard could tear every session down,
// before the background gRPC completion-queue thread had drained more than
// 1-2 of them -- SubscribeReceiverCall::Close() (invoked during teardown)
// explicitly DISCARDS whatever is still queued. Fixed with a bounded
// end-of-run settle wait (RunScenario, unity_twin.cc) for split_sessions
// runs: poll for new target_events and treat a short quiet window as
// "caught up" before returning.
// Consequence: harness_test.cc's DeterministicTraceAcrossTwoRuns now holds
// kGrpcTwoSessions to the EXACT SAME bit-for-bit contract as kInProcess
// (trace text AND every per-frame RunResult value) -- verified via a
// temporary bit-for-bit diagnostic assertion added alongside the (then
// still nominally weaker) earlier contract: `bazel test
// //spatialblend/sim/harness:harness_test --runs_per_test=30
// --test_filter='*DeterministicTraceAcrossTwoRuns*'` passed 30/30 with the
// diagnostic never failing under Grpc.
enum class Transport {
  kInProcess,
  kGrpcTwoSessions,
};

struct RunOptions {
  // Directory to write trace.jsonl into (created if missing). Empty ->
  // tracing disabled (sb_set_trace is never called).
  std::string run_dir;
  // Deterministic seed for the trajectory generator (jitter/wander) and the
  // network delay-queue's jitter/drop/reorder draws.
  uint64_t seed = 12345;
  // Fixed simulation step (task requires a 30Hz loop).
  double dt = 1.0 / 30.0;
  // Safety cap on frames in case a script fails to terminate.
  int max_frames = 600;
  // When true, overrides the scenario's own "network.latency_ms" with
  // override_latency_ms (used by harness_test.cc's "still 0 violations under
  // 300ms latency" check without needing a second scenario fixture file).
  bool override_latency = false;
  double override_latency_ms = 0.0;
  // Base port to start probing from when picking a free loopback port for
  // sb_host_space. 0 -> let the harness pick via an ephemeral-port probe.
  int32_t preferred_port = 0;
  // Transport for cross-space networking. Ignored for single-space
  // scenarios (no networking is used regardless).
  Transport transport = Transport::kInProcess;
  // When true, referent_resolve_mode below overrides the scenario's own
  // JSON "core.referent_resolve" (applied per-space by ApplyCoreConfig) for
  // EVERY space in the run -- see Space::OnRemoteUpdate in space.cc. kLinker
  // resolves a referent through the ElementLinker's cross-space identity
  // mapping first, falling back to a shared_id lookup only when unlinked;
  // kSharedId skips the linker translate step entirely and always looks the
  // referent up by the raw shared_id the payload carries. Used by
  // harness_test.cc's ReferentResolveModeAffectsMismatchedIdsReferent A/B
  // test to force one mode or the other on the same scenario/transport
  // without needing two copies of a fixture.
  bool override_referent_resolve_mode = false;
  sb_referent_resolve_mode_t referent_resolve_mode = SB_REFERENT_RESOLVE_LINKER;
};

struct FrameRecord {
  int frame = 0;
  double t = 0.0;
  double src_x = 0.0, src_y = 0.0, src_yaw = 0.0;
  // Target-side EMBODIED pose: the twin's own bounded driver (it owns and
  // publishes this pose, core paces nothing) -- what a real Unity clone would
  // render.
  double tgt_x = 0.0, tgt_y = 0.0, tgt_yaw = 0.0;
  bool has_committed_referent = false;
  int64_t committed_referent_shared_id = 0;
  double committed_proximity = 0.0;
  bool containment_violation = false;
  bool speed_violation = false;
  bool yaw_rate_violation = false;
  // Cross-session freshness evidence. Both values remain zero for a source
  // actor that has not emitted; otherwise split-session frames require exact
  // equality before the target pose is ticked and recorded. Split-session
  // setup emits its initial pose only after target avatars exist and confirms
  // that same sequence before frame zero.
  int64_t source_emitted_sequence = 0;
  int64_t target_applied_sequence = 0;
};

struct RunResult {
  std::vector<FrameRecord> frames;
  int containment_violations = 0;
  int speed_violations = 0;
  int yaw_rate_violations = 0;
  // Simplification: the frame at which the
  // source's scripted trajectory generator completes its last op, NOT a true
  // solver-convergence detector.
  int arrival_frame = -1;
  std::string trace_path;
  // Convenience: the last frame's committed referent shared id (0 if none).
  int64_t final_committed_referent_shared_id = 0;
  bool final_has_committed_referent = false;
};

absl::StatusOr<RunResult> RunScenario(const ScenarioModel& model,
                                      const RunOptions& opts);

}  // namespace harness
}  // namespace spatialblend

#endif  // SPATIALBLEND_HARNESS_UNITY_TWIN_H_
