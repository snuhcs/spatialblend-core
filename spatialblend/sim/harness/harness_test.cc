// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/sim/harness/harness_test.cc
//
// Encodes 8 regression scenarios (plus one representable DISABLED
// case) as testdata/regression/*.scenario.json fixtures, driving each one
// end-to-end through unity_twin.cc's RunScenario() (i.e. through the real C
// API, real gRPC loopback networking for the multi-space fixtures, and the
// real production solver) exactly as a Unity client would.
//
// Each fixture's numbers are derived from the corresponding
// spatialblend/net/cross_space_test.cc case (see the per-TEST comment
// below and the fixture file itself for the mapping).
//
// NOTE on networking / transports: every R1-R8 regression
// scenario below now runs under BOTH unity_twin.h's Transport values, via
// the HarnessRegressionTransportTest TEST_P fixture:
//   - Transport::kInProcess: all spaces share one sb_session_t/SpaceHub, and
//     unity_twin.cc never calls sb_host_space/sb_connect_space at all --
//     matching the precedent set by cross_space_test.cc and
//     CApiE2ETest.SourceToTargetCallback. This is the harness's original
//     behavior.
//   - Transport::kGrpcTwoSessions: the source space and every peer space
//     each get their OWN sb_session_t/SpaceHub, wired together for real via
//     sb_host_space (source) / sb_connect_space (every peer) over a
//     harness-probed loopback TCP port -- exercising the actual gRPC
//     proto-serialize/completion-queue-thread path.
// In BOTH transports, latency_ms/jitter_ms/drop_rate/reorder are implemented
// as a harness-level delay queue (NetworkQueue in unity_twin.cc) gating WHEN
// sb_update_actor_state is called, since the C API exposes no wire-level
// delay-injection hook -- see StillZeroViolationsUnder300msLatency below,
// which itself now runs under both transports.

#include "spatialblend/sim/harness/unity_twin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "nlohmann/json.hpp"
#include "spatialblend/sim/harness/scenario.h"

namespace spatialblend {
namespace harness {
namespace {

std::string TestdataPath(const std::string& name) {
  return "spatialblend/sim/harness/testdata/regression/" + name;
}

// Loads + builds a fresh RunOptions with a unique, clean run_dir per test.
RunOptions FreshOptions(const std::string& test_label) {
  RunOptions opts;
  opts.run_dir = ::testing::TempDir() + "/harness_test_" + test_label;
  std::error_code ec;
  std::filesystem::remove_all(opts.run_dir, ec);
  return opts;
}

absl::StatusOr<RunResult> RunFixture(const std::string& file,
                                     const RunOptions& opts) {
  auto model = LoadScenario(TestdataPath(file));
  if (!model.ok()) return model.status();
  return RunScenario(*model, opts);
}

std::string TransportSuffix(Transport t) {
  return t == Transport::kGrpcTwoSessions ? "grpc" : "inproc";
}

// R1-R8 below run once per Transport value via this TEST_P fixture.
// Each test body is transport-agnostic; GetParam() selects kInProcess vs
// kGrpcTwoSessions, and Opts() gives each (test, transport) pairing its own
// clean run_dir so the two instantiations never collide on trace.jsonl.
class HarnessRegressionTransportTest
    : public ::testing::TestWithParam<Transport> {
 protected:
  RunOptions Opts(const std::string& label) {
    auto opts = FreshOptions(label + "_" + TransportSuffix(GetParam()));
    opts.transport = GetParam();
    return opts;
  }
};

// -----------------------------------------------------------------------
// R1: clone-at-center via mismatched shared ids + linker resolution.
// Source: mismatched-shared-id="Object identity across spaces is asserted "
// (700/701 in space A, DIFFERENT 800/801 in space B). Base: CornerRepro.
// MismatchedIdsLinkerTwoFurniture. The clone must engage the linked chairA
// (target's own shared_id 800), not fall back to some default/centroid.
// -----------------------------------------------------------------------
TEST_P(HarnessRegressionTransportTest, R1_CloneAtCenterMismatchedIdsLinker) {
  auto opts = Opts("r1");
  auto result = RunFixture("R1-clone-at-center-mismatched-ids.scenario.json", opts);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->final_has_committed_referent);
  EXPECT_EQ(result->final_committed_referent_shared_id, 800);
  EXPECT_EQ(result->containment_violations, 0);
  EXPECT_EQ(result->speed_violations, 0);
  EXPECT_EQ(result->yaw_rate_violations, 0);
}

// -----------------------------------------------------------------------
// R2: corner clamp. The linked target furniture sits just outside the
// target's navigable polygon; the clone must realize AT the piece (via the
// footprint-containment fallback), never clamp to the room's SW corner.
// (A child region must lie inside its parent, so the
// piece sits IN the SW corner, (1,1), touching both walls.)
// Base: CornerRepro.CommittedReferentOutsideTargetPolygonRealizesAtPiece.
// -----------------------------------------------------------------------
TEST_P(HarnessRegressionTransportTest, R2_CornerClamp) {
  auto opts = Opts("r2");
  auto result = RunFixture("R2-corner-clamp.scenario.json", opts);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->final_has_committed_referent);
  EXPECT_EQ(result->final_committed_referent_shared_id, 800);
  EXPECT_EQ(result->containment_violations, 0);
  EXPECT_EQ(result->speed_violations, 0);
}

// -----------------------------------------------------------------------
// R3: phantom centroid. Open-space, 4 furniture at displaced layouts; the
// clone must commit to the engaged piece (objA / shared_id 300), never the
// multi-object centroid. Base: SerializationRepro.
// MultiFurnitureApproachCommitsNotCentroid.
// -----------------------------------------------------------------------
TEST_P(HarnessRegressionTransportTest, R3_PhantomCentroidOpenSpaceMultiObject) {
  auto opts = Opts("r3");
  auto result = RunFixture("R3-phantom-centroid.scenario.json", opts);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->final_has_committed_referent);
  EXPECT_EQ(result->final_committed_referent_shared_id, 300);
  EXPECT_EQ(result->containment_violations, 0);
  EXPECT_EQ(result->speed_violations, 0);
}

// -----------------------------------------------------------------------
// R4: "TV -> desk" wrong-object under a displaced layout. The clone must
// engage the clearly-approached "tv" (shared_id 300), never the "desk"
// (301), even though both propagate/resolve with matching ids. Base:
// SpacingNested.EngagedObjectCorrespondenceAmongDisplacedSet.
// -----------------------------------------------------------------------
TEST_P(HarnessRegressionTransportTest, R4_TvDeskWrongObjectUnderDisplacedLayout) {
  auto opts = Opts("r4");
  auto result = RunFixture("R4-tv-desk-wrong-object.scenario.json", opts);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->final_has_committed_referent);
  EXPECT_EQ(result->final_committed_referent_shared_id, 300);
  EXPECT_EQ(result->containment_violations, 0);
  EXPECT_EQ(result->speed_violations, 0);
}

// -----------------------------------------------------------------------
// R5: avatar-avatar chatter. Single space, two actors (one bound, one
// unbound "peer" standing right next to it); the bound actor must commit to
// the scene furniture (shared_id 200), NEVER to the peer avatar's own
// element (shared_id 9002) or itself (9001). No networking is used here (a
// single scenario space => RunScenario's use_network flag is false, so the
// transport parameter is inert for this scenario -- see unity_twin.h), so
// the "measure" actor is the driven actor itself in its home space -- see
// unity_twin.h's documented fallback. Base: ReferentGeometryTest.
// ActorDoesNotFormRelationToPeerActor.
// -----------------------------------------------------------------------
TEST_P(HarnessRegressionTransportTest, R5_AvatarAvatarChatter) {
  auto opts = Opts("r5");
  auto result = RunFixture("R5-avatar-avatar-chatter.scenario.json", opts);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_FALSE(result->frames.empty());
  for (const auto& fr : result->frames) {
    if (!fr.has_committed_referent) continue;
    EXPECT_NE(fr.committed_referent_shared_id, 9001)
        << "frame " << fr.frame << ": actor formed a relation to itself";
    EXPECT_NE(fr.committed_referent_shared_id, 9002)
        << "frame " << fr.frame << ": actor formed a relation to its peer";
  }
  EXPECT_TRUE(result->final_has_committed_referent);
  EXPECT_EQ(result->final_committed_referent_shared_id, 200);
  EXPECT_EQ(result->containment_violations, 0);
}

// -----------------------------------------------------------------------
// R6: standing jitter. The actor starts already inside objA's affordance
// footprint and jitters (seeded, bounded) without leaving it; the clone must
// HOLD its commitment to objA (shared_id 300) on every frame, never flicker
// to a neighbour or a multi-object centroid. Base: SerializationRepro.
// StandingJitterHoldsCommitmentNoCentroidFlicker.
// -----------------------------------------------------------------------
TEST_P(HarnessRegressionTransportTest, R6_StandingJitterHoldsCommitment) {
  auto opts = Opts("r6");
  auto result = RunFixture("R6-standing-jitter.scenario.json", opts);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_FALSE(result->frames.empty());
  int committed_frames = 0;
  for (const auto& fr : result->frames) {
    if (!fr.has_committed_referent) continue;
    ++committed_frames;
    EXPECT_EQ(fr.committed_referent_shared_id, 300)
        << "frame " << fr.frame << ": commitment flickered off objA";
  }
  EXPECT_GT(committed_frames, 0);
  EXPECT_EQ(result->containment_violations, 0);
  EXPECT_EQ(result->speed_violations, 0);
}

// -----------------------------------------------------------------------
// R7: fast approach / large steps. The scripted source pose covers a large
// distance in very few (3) frames -- far faster than any real human motion
// -- yet the measured avatar is paced by sb_tick_avatar's velocity-bounded
// controller, so it must never register a speed violation regardless of how
// abruptly the underlying source pose jumps. No cross_space_test.cc analog;
// synthesized to exercise the controller-pacing guarantee directly.
// -----------------------------------------------------------------------
TEST_P(HarnessRegressionTransportTest, R7_FastApproachLargeStepsStaysVelocityBounded) {
  auto opts = Opts("r7");
  auto result = RunFixture("R7-fast-approach-large-steps.scenario.json", opts);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->speed_violations, 0);
  EXPECT_EQ(result->yaw_rate_violations, 0);
  EXPECT_EQ(result->containment_violations, 0);
}

// -----------------------------------------------------------------------
// R8: source-rate decoupling. A high drop_rate (0.6) plus latency/jitter/
// reorder on the network delay queue means most of the 90 scripted source
// updates never reach the target actor on the frame they were generated (or
// at all); RunScenario's main loop still calls sb_tick_avatar +
// sb_get_relation_states on the target's clone EVERY harness frame
// regardless of whether an update arrived that frame (see unity_twin.cc).
// The clone's paced motion must therefore stay smooth and violation-free
// even though the source only "publishes" at an effective sub-30Hz,
// irregular rate. No cross_space_test.cc analog (that suite has no network
// layer); synthesized using the harness's own delay-queue mechanism.
// -----------------------------------------------------------------------
TEST_P(HarnessRegressionTransportTest, R8_SourceRateDecouplingStaysSmooth) {
  auto opts = Opts("r8");
  auto result = RunFixture("R8-source-rate-decoupling.scenario.json", opts);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->containment_violations, 0);
  EXPECT_EQ(result->speed_violations, 0);
  EXPECT_EQ(result->yaw_rate_violations, 0);
}

// -----------------------------------------------------------------------
// Both transports must produce the identical committed referent across
// 8 scenarios with 0 violations. The
// HarnessRegressionTransportTest instantiations above already assert each
// transport independently reproduces the SAME expected shared_id (hence
// transitively identical to each other) -- this test makes that comparison
// explicit and direct by running both transports back-to-back for every R1-
// R8 fixture in one place and diffing the two RunResults.
// -----------------------------------------------------------------------
TEST(HarnessRegressionTest,
    AllEightScenariosIdenticalCommittedReferentAcrossTransports) {
  struct Case {
    const char* file;
    const char* label;
  };
  static constexpr Case kCases[] = {
      {"R1-clone-at-center-mismatched-ids.scenario.json", "r1"},
      {"R2-corner-clamp.scenario.json", "r2"},
      {"R3-phantom-centroid.scenario.json", "r3"},
      {"R4-tv-desk-wrong-object.scenario.json", "r4"},
      {"R5-avatar-avatar-chatter.scenario.json", "r5"},
      {"R6-standing-jitter.scenario.json", "r6"},
      {"R7-fast-approach-large-steps.scenario.json", "r7"},
      {"R8-source-rate-decoupling.scenario.json", "r8"},
  };
  for (const auto& c : kCases) {
    auto opts_inproc = FreshOptions(std::string(c.label) + "_xport_inproc");
    opts_inproc.transport = Transport::kInProcess;
    auto result_inproc = RunFixture(c.file, opts_inproc);
    ASSERT_TRUE(result_inproc.ok()) << c.file << ": " << result_inproc.status();

    auto opts_grpc = FreshOptions(std::string(c.label) + "_xport_grpc");
    opts_grpc.transport = Transport::kGrpcTwoSessions;
    auto result_grpc = RunFixture(c.file, opts_grpc);
    ASSERT_TRUE(result_grpc.ok()) << c.file << ": " << result_grpc.status();

    EXPECT_EQ(result_inproc->final_has_committed_referent,
              result_grpc->final_has_committed_referent)
        << c.file;
    EXPECT_EQ(result_inproc->final_committed_referent_shared_id,
              result_grpc->final_committed_referent_shared_id)
        << c.file;
    EXPECT_EQ(result_grpc->containment_violations, 0) << c.file;
    EXPECT_EQ(result_grpc->speed_violations, 0) << c.file;
    EXPECT_EQ(result_grpc->yaw_rate_violations, 0) << c.file;
  }
}

// -----------------------------------------------------------------------
// DISABLED tests survey: the repo currently has 3 DISABLED tests upstream
// (CornerRepro.DISABLED_ReproSofaCommitRealizesWhere has since been
// FIXED and renamed, per the "FIXED (was DISABLED_...)" comment in
// cross_space_test.cc):
//   1. CornerRepro.DISABLED_SpacingCentroidBetweenAdjacentFurnitureMappedFarApart
//      (space/cross_space_test.cc) -- REPRESENTABLE, encoded below as R9.
//   2. ScenarioReplayTest.DISABLED_AffordancePoseReplay
//      (simulator/scenario_replay_test.cc) -- NOT representable through this
//      harness: it asserts a per-anchor "hand" affordance IK target (root +
//      hand, with the hand target reconstructed from the two-stage
//      SolveActorTargets realizer). An earlier merge retired
//      that realizer in favor of upstream's single-root realization
//      (OnRemoteUpdate -> solver_.Solve emits only the root target), so
//      sb_poll_realization_targets never reports a per-anchor "hand" target
//      to drive a scenario assertion against -- there is nothing for a
//      scenario fixture to exercise until a per-anchor affordance realizer
//      is restored.
//   3. CApiE2ETest.DISABLED_AffordanceTargetCarriesHeightThroughAbi
//      (c/c_api_test.cc) -- NOT representable, for the identical reason as
//      #2 (same retired two-stage per-anchor realizer; this is the C-ABI
//      counterpart of the same underlying capability gap).
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
// R9 (DISABLED): documents a known, still-latent weakness -- an exact
// spacing tie between two ADJACENT source furniture (touching at x=6)
// anneals to a ~50/50 split, and when the corresponding target furniture
// are mapped FAR APART, that split lands the clone at a phantom midpoint in
// open space instead of committing to either piece. This mirrors
// CornerRepro.DISABLED_SpacingCentroidBetweenAdjacentFurnitureMappedFarApart
// (also DISABLED upstream, for the same reason: the centroid-anneal
// weakness has not been hardened yet). Kept DISABLED so CI does not enforce
// a currently-failing expectation; re-enable alongside the C++ repro once
// the annealing/dropping fix lands.
// -----------------------------------------------------------------------
TEST(HarnessRegressionTest,
    DISABLED_R9_SpacingCentroidBetweenAdjacentFurnitureMappedFarApart) {
  auto opts = FreshOptions("r9");
  auto result =
      RunFixture("R9-disabled-adjacent-furniture-phantom-midpoint.scenario.json", opts);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->final_has_committed_referent);
  EXPECT_TRUE(result->final_committed_referent_shared_id == 800 ||
             result->final_committed_referent_shared_id == 801)
      << "Clone should commit to ONE of the two chairs (800 or 801), got "
      << result->final_committed_referent_shared_id;
}

// -----------------------------------------------------------------------
// Network resilience: re-running one cross-space regression (R1) with a
// forced 300ms latency override must still produce 0 containment/speed/
// yaw-rate violations -- the harness's delay queue only changes WHEN a
// source update is applied, never how it's paced once applied.
// -----------------------------------------------------------------------
// Parameterized over transport -- "grpc mode with 300ms latency must
// show 0 violations" is one of the two instantiations below
// (InProcessAndGrpc/StillZeroViolationsUnder300msLatency.Grpc).
TEST_P(HarnessRegressionTransportTest, StillZeroViolationsUnder300msLatency) {
  auto opts = Opts("r1_latency300");
  opts.override_latency = true;
  opts.override_latency_ms = 300.0;
  auto result = RunFixture("R1-clone-at-center-mismatched-ids.scenario.json", opts);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->containment_violations, 0);
  EXPECT_EQ(result->speed_violations, 0);
  EXPECT_EQ(result->yaw_rate_violations, 0);
}

// -----------------------------------------------------------------------
// Determinism: running the SAME scenario+seed twice must produce
// byte-identical trace.jsonl output, after normalizing the one field that is
// deliberately wall-clock-derived rather than simulation-derived: "ev":
// "tick" lines carry a "cost_us" measuring real elapsed wall-clock time
// spent in that tick (see spatialblend/space/space.cc), which is NOT part of
// the deterministic simulation state (unlike "t", which is the deterministic
// space_clock_). Every other event type ("src"/"payload"/"target"/
// "relation"/"share") carries only "t" (space_clock_) and simulation state,
// so this is the only normalization needed.
// -----------------------------------------------------------------------
std::string NormalizeTraceLine(const std::string& line) {
  static const std::regex kCostUs(R"("cost_us":[0-9]+)");
  return std::regex_replace(line, kCostUs, "\"cost_us\":0");
}

// Beyond cost_us (the one genuinely wall-clock field per trace.format.md;
// every other field is driven off the deterministic space_clock_), two runs
// of RunScenario() in the SAME test process are not byte-identical for a
// second reason that has nothing to do with simulation behavior: sb_space_t
// and local sb_element_t handles (the "space", "received", and "resolved"
// fields emitted by Space::AddSpace / Space::ResolveSharedElement) are
// allocated from counters that are GLOBAL to the process and never reset
// between sessions, so run B's handles are simply offset by whatever run A
// (and any earlier test in this binary) already allocated. This is a
// harness-process artifact, not a behavioral difference -- the *relative*
// structure (which events share a space, which "received" temp resolves to
// which "resolved" local element) is what must match, not the absolute
// numbers. So we canonicalize both id spaces to 0,1,2,... in order of first
// appearance within each trace before comparing.
//
// A THIRD source of byte-inequality, gRPC-only, was found by
// running --runs_per_test=10 on this test: under Transport::kGrpcTwoSessions
// the peer (measure/avatar) space's OWN "t" (Space::space_clock_) is only
// ever advanced inside Space::UpdateActorState (space.cc) -- and the measure
// actor in every regression fixture is driven exclusively through
// TickAvatar/AvatarController, never through UpdateActorState -- so every
// "share"/"payload"/"target"/"tick" line Space::OnRemoteUpdate emits on the
// peer (space.cc, called from SubscribeStreamCall::Proceed() on that peer's
// OWN gRPC completion-queue thread -- see remote_hub.h's RemoteHub class
// comment for that thread) carries "t":0.0
// for the ENTIRE run, regardless of which real simulated frame produced it.
// Concurrently, the driven/source space's "src"/"relation" lines (emitted
// from the harness's own main thread, inside UpdateActorState itself) carry
// the correct, monotonically-advancing "t". TraceSink::Emit (common/trace.h)
// serializes both streams behind one mutex, but which thread wins that mutex
// on any given real-time instant is genuine OS thread-scheduling
// nondeterminism -- so the RAW LINE ORDER interleaving of the peer's (all
// "t":0.0) stream against the source's (correctly-timestamped) stream
// differs between two identically-seeded runs even though every event's
// CONTENT is bit-identical (the CQ thread's arrival timing affects trace
// interleaving, never the deterministic solver state the harness's
// WaitForCountAbove/sb_poll_realization_targets barrier already guards --
// see the comment on DeterministicTraceAcrossTwoRuns below).
//
// Fix: canonicalize line order too, not just handle ids. Group lines by "t"
// (their real chronological instant is genuinely embedded there whenever a
// space's own clock is actually advancing; entries that all collapse onto
// the same stuck "t":0.0 are, by construction, safe to canonicalize amongst
// themselves) and, within each "t" group, stable_sort by a content key (ev,
// space, actor, anchor, referent, relation). Two same-key lines from
// different real frames (e.g. two "payload" lines for the same referent on
// successive ticks) are never actually racing each other -- both are always
// emitted by the SAME thread (the peer's CQ thread, in true simulation
// order) -- so std::stable_sort's tie-preserving-original-order guarantee
// keeps their relative sequence (and hence their differing R/dist/weight
// values) exactly as recorded. Only the interleaving BETWEEN the two threads'
// streams is discarded, which is exactly the nondeterminism being
// normalized away. This does not silently paper over a genuine solver
// difference: BestCommittedReferent/frames (compared separately below via
// EXPECT_DOUBLE_EQ / EXPECT_EQ on the in-memory RunResult, untouched by any
// of this) would still fail if the actual simulated values ever diverged.
std::string NormalizeTraceContent(const std::string& path) {
  std::ifstream in(path);
  std::unordered_map<int64_t, int64_t> space_remap;
  std::unordered_map<int64_t, int64_t> elem_remap;
  auto Canon = [](std::unordered_map<int64_t, int64_t>& remap, int64_t v) {
    auto it = remap.find(v);
    if (it != remap.end()) return it->second;
    int64_t next = static_cast<int64_t>(remap.size());
    remap.emplace(v, next);
    return next;
  };

  struct Entry {
    bool is_json = false;
    nlohmann::json j;
    std::string raw;      // only meaningful when !is_json
    int64_t order = 0;    // original line index, used as the sort key for
                          // non-json fallback lines (kept in original
                          // relative order; real traces never hit this path)
  };
  std::vector<Entry> entries;

  std::string line;
  int64_t line_no = 0;
  while (std::getline(in, line)) {
    std::string normalized = NormalizeTraceLine(line);
    Entry e;
    e.order = line_no++;
    try {
      e.j = nlohmann::json::parse(normalized);
      e.is_json = true;
    } catch (const std::exception&) {
      e.raw = normalized;
      e.is_json = false;
    }
    if (e.is_json) {
      if (e.j.contains("space") && e.j["space"].is_number_integer()) {
        e.j["space"] = Canon(space_remap, e.j["space"].get<int64_t>());
      }
      if (e.j.contains("received") && e.j["received"].is_number_integer()) {
        e.j["received"] = Canon(elem_remap, e.j["received"].get<int64_t>());
      }
      if (e.j.contains("resolved") && e.j["resolved"].is_number_integer()) {
        e.j["resolved"] = Canon(elem_remap, e.j["resolved"].get<int64_t>());
      }
    }
    entries.push_back(std::move(e));
  }

  // Sort key: (t, ev, space, actor, anchor, referent, relation). Every field
  // after "t" is read as its raw dumped JSON text (so it's stable across
  // string/number/absent) except "t" itself, read as a double so ties
  // compare numerically equal even if formatted differently. Non-json
  // fallback lines get a key strictly less than every real "t" value
  // (fallback lines don't occur in real traces; this just keeps
  // std::stable_sort's comparator a valid strict weak ordering).
  auto FieldText = [](const nlohmann::json& j, const char* field) {
    return j.contains(field) ? j[field].dump() : std::string();
  };
  auto SortKey = [&](const Entry& e) {
    double t = (e.is_json && e.j.contains("t") && e.j["t"].is_number())
                   ? e.j["t"].get<double>()
                   : -1e18 - static_cast<double>(e.order);
    return std::make_tuple(t, e.is_json ? FieldText(e.j, "ev") : std::string(),
                           e.is_json ? FieldText(e.j, "space") : std::string(),
                           e.is_json ? FieldText(e.j, "actor") : std::string(),
                           e.is_json ? FieldText(e.j, "anchor") : std::string(),
                           e.is_json ? FieldText(e.j, "referent") : std::string(),
                           e.is_json ? FieldText(e.j, "relation") : std::string());
  };
  std::stable_sort(entries.begin(), entries.end(),
                   [&](const Entry& a, const Entry& b) {
                     return SortKey(a) < SortKey(b);
                   });

  std::ostringstream out;
  for (const auto& e : entries) {
    if (e.is_json) {
      out << e.j.dump() << "\n";
    } else {
      out << e.raw << "\n";
    }
  }
  return out.str();
}

// Parameterized over transport -- "determinism per transport" means
// this same identically-seeded-twice comparison must hold independently
// under kInProcess AND kGrpcTwoSessions. Every RunScenario step in both
// transports is serialized behind a blocking sb_try_wait/WaitForCountAbove
// before the harness advances, so the real gRPC CQ background thread's
// scheduling nondeterminism never actually reaches trace/RunResult
// ordering -- only wall-clock cost_us (already normalized below) differs
// between runs.
//
// This test ORIGINALLY held its Grpc instantiation to a
// DIFFERENT, documented-WEAKER contract than InProcess (SET of committed
// referents + final pose within 1e-6 + zero violations, rather than
// bit-for-bit), because two things made Grpc trace text genuinely diverge
// between two identically-seeded runs even though the underlying simulation
// was itself deterministic:
//  1. The peer (measure/avatar) space's OWN "t" (Space::space_clock_) was
//     only ever advanced inside Space::UpdateActorState -- and the measure
//     actor in every regression fixture is driven exclusively through
//     TickAvatar/AvatarController, never through UpdateActorState -- so
//     every "share"/"payload"/"target"/"tick" line Space::OnRemoteUpdate
//     emitted on the peer carried "t":0.0 for the ENTIRE run. Concurrently,
//     the driven/source space's "src"/"relation" lines carried the correct,
//     monotonically-advancing "t". Both streams share TraceSink::Emit's one
//     mutex, but which thread's line lands first at any real-time instant
//     when BOTH claim "t":0.0 is genuine OS thread-scheduling
//     nondeterminism -- so the raw line-order interleaving differed between
//     runs even though every event's content was bit-identical.
//  2. An occasional (~1-in-10 runs) EXTRA duplicate Space::OnRemoteUpdate
//     call on the peer for the very first delivered update only, right
//     after sb_connect_space's bootstrap/Subscribe-stream-settle window --
//     a genuine extra invocation (confirmed: "ev":"relation" lines did NOT
//     double, since RelationPhaseTracker only emits on an actual
//     from!=to transition and a replay of identical content is a silent
//     no-op there, while "payload"/"target"/"tick", emitted unconditionally
//     every OnRemoteUpdate call, did double).
//
// BOTH root causes are fixed: (1) Space::OnRemoteUpdate now
// advances space_clock_ too (using the update's own step_duration), so a
// peer's "t" is real and monotonically increasing instead of stuck at 0.0 --
// see space.h's space_clock_ doc. (2) RemoteRelationUpdate now carries a
// per-(source space, source actor) monotonically increasing `sequence`
// (remote_update.h), and Space::OnRemoteUpdate drops an incoming update
// whose sequence is <= the last one already applied for that key -- an
// exact-replay duplicate is now a no-op instead of a second real invocation.
// RunScenario (unity_twin.cc) also gained a bounded end-of-run settle wait
// for split_sessions runs, discovered necessary while verifying this test:
// without it, a scenario that propagates ~90 updates in a tight,
// no-real-wall-clock-delay frame loop could see most of them silently
// DISCARDED (SubscribeReceiverCall::Close() drops whatever is still queued)
// because RunScenario returned -- tearing every session down -- before the
// background completion-queue thread had drained the backlog; unrelated to
// duplication, but it independently made any exact-line-count comparison
// across transports meaningless until fixed.
//
// With both root causes and the settle-wait gap fixed, this test's Grpc
// instantiation is now held to the EXACT SAME bit-for-bit contract as
// InProcess (trace text AND every per-frame RunResult value), unconditional
// on GetParam() below. Verified via a temporary bit-for-bit diagnostic
// assertion added ALONGSIDE the (still nominally weaker) earlier contract:
// `bazel test //spatialblend/sim/harness:harness_test --runs_per_test=30
// --test_filter='*DeterministicTraceAcrossTwoRuns*'` passed 30/30 with the
// diagnostic never failing under Grpc, so the strict contract below
// replaces the weaker one rather than sitting alongside it.
TEST_P(HarnessRegressionTransportTest, DeterministicTraceAcrossTwoRuns) {
  auto opts_a = Opts("determinism_run_a");
  auto opts_b = Opts("determinism_run_b");
  opts_a.seed = opts_b.seed = 424242;

  auto result_a = RunFixture("R3-phantom-centroid.scenario.json", opts_a);
  ASSERT_TRUE(result_a.ok()) << result_a.status();
  auto result_b = RunFixture("R3-phantom-centroid.scenario.json", opts_b);
  ASSERT_TRUE(result_b.ok()) << result_b.status();

  ASSERT_FALSE(result_a->trace_path.empty());
  ASSERT_FALSE(result_b->trace_path.empty());

  std::string trace_a = NormalizeTraceContent(result_a->trace_path);
  std::string trace_b = NormalizeTraceContent(result_b->trace_path);
  ASSERT_FALSE(trace_a.empty()) << "trace.jsonl for run A is empty";

  // Bit-for-bit for BOTH transports (see the doc comment above this
  // test for why: the two root causes -- the peer clock frozen at "t":0.0
  // and the ~1-in-10 duplicate first delivery -- are exactly what motivated
  // the weaker Grpc-only contract, and both are fixed. Confirmed empirically
  // with a temporary bit-for-bit diagnostic assertion added to the Grpc path
  // alongside the (then still-weaker) documented contract:
  // `bazel test //spatialblend/sim/harness:harness_test --runs_per_test=30
  // --test_filter='*DeterministicTraceAcrossTwoRuns*'` passed 30/30 with
  // ZERO bit-for-bit diffs under Grpc, so the strict contract below is now
  // unconditional rather than InProcess-only.
  EXPECT_EQ(trace_a, trace_b)
      << "trace.jsonl differs between two identically-seeded "
      << TransportSuffix(GetParam())
      << " runs (after normalizing wall-clock cost_us fields and same-\"t\" "
         "event order)";

  // Also confirm the in-memory RunResult itself is bit-identical (frame
  // count + every recorded pose/violation flag), independent of tracing.
  ASSERT_EQ(result_a->frames.size(), result_b->frames.size());
  for (size_t i = 0; i < result_a->frames.size(); ++i) {
    const auto& fa = result_a->frames[i];
    const auto& fb = result_b->frames[i];
    EXPECT_DOUBLE_EQ(fa.tgt_x, fb.tgt_x) << "frame " << i;
    EXPECT_DOUBLE_EQ(fa.tgt_y, fb.tgt_y) << "frame " << i;
    EXPECT_DOUBLE_EQ(fa.tgt_yaw, fb.tgt_yaw) << "frame " << i;
    EXPECT_EQ(fa.committed_referent_shared_id,
             fb.committed_referent_shared_id)
        << "frame " << i;
  }

  ASSERT_EQ(result_a->containment_violations, 0);
  ASSERT_EQ(result_a->speed_violations, 0);
  ASSERT_EQ(result_a->yaw_rate_violations, 0);
  ASSERT_EQ(result_b->containment_violations, 0);
  ASSERT_EQ(result_b->speed_violations, 0);
  ASSERT_EQ(result_b->yaw_rate_violations, 0);
}

// The split-session harness must not record a paced target pose after merely
// replaying an old target snapshot. Every emitted source sequence is observed
// on the receiver before its frame is ticked; held frames retain the same
// sequence on both sides rather than waiting for a non-existent deadband
// update.
TEST(HarnessRegressionTest, GrpcFramesObserveExactEmittedSequenceIncludingHolds) {
  auto opts = FreshOptions("grpc_sequence_freshness");
  opts.transport = Transport::kGrpcTwoSessions;
  opts.seed = 424242;
  auto result = RunFixture("R3-phantom-centroid.scenario.json", opts);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_FALSE(result->frames.empty());

  bool saw_hold = false;
  for (size_t i = 0; i < result->frames.size(); ++i) {
    const auto& frame = result->frames[i];
    EXPECT_EQ(frame.target_applied_sequence, frame.source_emitted_sequence)
        << "frame " << i << " recorded before the exact source update "
        << "was applied";
    if (i > 0 &&
        frame.source_emitted_sequence ==
            result->frames[i - 1].source_emitted_sequence) {
      saw_hold = true;
    }
  }
  EXPECT_TRUE(saw_hold) << "R3 standing tail must exercise deadband-held frames";
}

// Split-session setup must not emit an actor update before the peer avatar
// exists. Make the first scripted frame a deterministic hold: it cannot emit
// a second update, so equality at frame zero proves the setup sequence itself
// was received rather than a later movement being mistaken for it.
TEST(HarnessRegressionTest, GrpcInitialHeldFrameHasSetupSequenceReceipt) {
  auto model = LoadScenario(TestdataPath("R3-phantom-centroid.scenario.json"));
  ASSERT_TRUE(model.ok()) << model.status();
  ScenarioScriptOp hold;
  hold.op = "stand";
  hold.raw = nlohmann::json{{"op", "stand"}, {"duration", 1.0},
                            {"jitter_sigma", 0.0}};
  model->behavior.script = {std::move(hold)};

  auto opts = FreshOptions("grpc_initial_held_setup_receipt");
  opts.transport = Transport::kGrpcTwoSessions;
  opts.max_frames = 1;
  auto result = RunScenario(*model, opts);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->frames.size(), 1u);
  EXPECT_GT(result->frames.front().source_emitted_sequence, 0);
  EXPECT_EQ(result->frames.front().target_applied_sequence,
            result->frames.front().source_emitted_sequence);
}

// -----------------------------------------------------------------------
// A/B sb_set_referent_resolve_mode(linker|sharedid) using the
// two-transport harness. Space::OnRemoteUpdate (space.cc) already threads
// resolved_local_element_id through relation_solver.cc's referent lookups
// (preferring it over GetElementBySharedId when set) -- so the question
// this test answers empirically is whether that wiring actually changes
// which referent gets committed, on the one case built to distinguish the
// two modes: R1's mismatched shared ids (source chairA=700 is
// link_to_existing'd to the target's PRE-EXISTING chairA, whose own shared
// id is 800 -- Space::ResolveSharedElement removes the received temp
// element entirely for link_to_existing, so nothing on the target ever
// carries shared_id 700 again).
//   - kLinker: hub_->GetLinker().Translate(source handle) finds the
//     pre-existing target element regardless of its shared id -- commits
//     referent 800 (matches R1_CloneAtCenterMismatchedIdsLinker's baseline).
//   - kSharedId: Translate is skipped, so the referent lookup falls back to
//     GetElementBySharedId(700) on the target tree, which no longer resolves
//     to anything -- that payload contributes no valid referent.
// The matched-ids control (C9-matched-ids-control.scenario.json, "add_as_new"
// resolution so the SAME shared_id (700) names the same element on both
// sides) proves the two modes are NOT just unconditionally different: with
// nothing for the linker to correct, both lookups land on the identical
// local element and must produce bit-identical output.
//
// IMPORTANT (found empirically while writing this test): unity_twin.cc's
// OnRelationEventCb/BestCommittedReferent translate a relation event's raw
// (source-side) referent shared_id through the harness's OWN
// source_shared_to_local_shared table -- built independently, purely from
// watching resolve-on-share events -- BEFORE it ever reaches
// final_committed_referent_shared_id. That table maps 700 -> 800
// unconditionally, regardless of what Space::OnRemoteUpdate's solver
// actually did with the referent internally, so
// final_committed_referent_shared_id reads "800" under BOTH modes and does
// NOT distinguish them (confirmed: an earlier version of this test that
// asserted on it failed by finding both modes equal). The lifecycle-phase
// side is also independent of referent resolution: EmitRelationTransitions
// runs off the payload's wire-transmitted proximity/weight, not off any
// local element lookup, so "has committed referent" per se is identical too.
// The one place resolution success/failure actually shows up is realized
// POSE: relation_solver.cc's SolveImpl does `if (!referent) continue;` per
// payload BEFORE computing a goal, so on R1 (a two-furniture mismatched-ids
// fixture -- chairA/chairB both propagate, so BOTH generate a payload, and
// per the "REPRO 2" spacing-centroid case in cross_space_test.cc the two
// engage at near-equal weight, so the realized pose is a clamped centroid of
// the two, not a clean single-object approach):
//   - kLinker: hub_->GetLinker().Translate resolves BOTH payloads (chairA
//     700->800, chairB 701->801) to their real linked target elements, so
//     the centroid solve has two live goals and the target avatar moves off
//     its (0,0) default -- empirically to (0.5, 0.0) for this fixture.
//   - kSharedId: Translate is skipped for both, and GetElementBySharedId on
//     B's tree fails for 700 AND 701 alike (Space::ResolveSharedElement's
//     link_to_existing already deleted both received temps) -- neither
//     payload contributes a goal at all, so the target avatar's realized
//     pose never departs from its (0, 0) default (empirically exact,
//     confirmed bit-identical to the default over the whole run).
// This is the "different realized pose under mismatched ids" proof the task
// asks for: kLinker measurably moves the clone, kSharedId leaves it inert.
TEST(HarnessRegressionTest, ReferentResolveModeAffectsMismatchedIdsReferent) {
  // Mismatched ids (R1): the two modes must diverge in REALIZED POSE.
  auto opts_linker = FreshOptions("c9_r1_linker");
  opts_linker.override_referent_resolve_mode = true;
  opts_linker.referent_resolve_mode = SB_REFERENT_RESOLVE_LINKER;
  auto result_linker =
      RunFixture("R1-clone-at-center-mismatched-ids.scenario.json", opts_linker);
  ASSERT_TRUE(result_linker.ok()) << result_linker.status();
  ASSERT_FALSE(result_linker->frames.empty());
  EXPECT_EQ(result_linker->containment_violations, 0);
  const auto& final_linker = result_linker->frames.back();
  // kLinker must move the clone off its (0,0) default -- both mismatched-id
  // referents resolve via the linker, giving the centroid solve live goals.
  EXPECT_GT(std::abs(final_linker.tgt_x) + std::abs(final_linker.tgt_y), 0.1)
      << "linker mode left the target avatar at its (0,0) default -- "
         "resolved_local_element_id may not actually be reaching the solver";

  auto opts_sharedid = FreshOptions("c9_r1_sharedid");
  opts_sharedid.override_referent_resolve_mode = true;
  opts_sharedid.referent_resolve_mode = SB_REFERENT_RESOLVE_SHAREDID;
  auto result_sharedid =
      RunFixture("R1-clone-at-center-mismatched-ids.scenario.json", opts_sharedid);
  ASSERT_TRUE(result_sharedid.ok()) << result_sharedid.status();
  ASSERT_FALSE(result_sharedid->frames.empty());
  const auto& final_sharedid = result_sharedid->frames.back();
  // kSharedId cannot resolve either mismatched (and since-deleted) shared id
  // on B's tree, so the clone must stay inert at its (0,0) default --
  // and, in particular, must NOT match kLinker's moved-off-default pose.
  EXPECT_NEAR(final_sharedid.tgt_x, 0.0, 1e-6);
  EXPECT_NEAR(final_sharedid.tgt_y, 0.0, 1e-6);
  double dx = final_sharedid.tgt_x - final_linker.tgt_x;
  double dy = final_sharedid.tgt_y - final_linker.tgt_y;
  EXPECT_GT(std::sqrt(dx * dx + dy * dy), 0.1)
      << "sharedid and linker modes realized to the same pose on a "
         "mismatched-ids scenario (sharedid=(" << final_sharedid.tgt_x << ","
      << final_sharedid.tgt_y << ") linker=(" << final_linker.tgt_x << ","
      << final_linker.tgt_y
      << ")) -- resolved_local_element_id may not actually be gating the "
         "solver's referent lookup any more";

  // Matched ids (control): nothing for the linker to correct, so the two
  // modes must agree exactly (position/yaw bit-identical, same referent).
  auto opts_ctrl_linker = FreshOptions("c9_ctrl_linker");
  opts_ctrl_linker.override_referent_resolve_mode = true;
  opts_ctrl_linker.referent_resolve_mode = SB_REFERENT_RESOLVE_LINKER;
  auto result_ctrl_linker =
      RunFixture("C9-matched-ids-control.scenario.json", opts_ctrl_linker);
  ASSERT_TRUE(result_ctrl_linker.ok()) << result_ctrl_linker.status();

  auto opts_ctrl_sharedid = FreshOptions("c9_ctrl_sharedid");
  opts_ctrl_sharedid.override_referent_resolve_mode = true;
  opts_ctrl_sharedid.referent_resolve_mode = SB_REFERENT_RESOLVE_SHAREDID;
  auto result_ctrl_sharedid =
      RunFixture("C9-matched-ids-control.scenario.json", opts_ctrl_sharedid);
  ASSERT_TRUE(result_ctrl_sharedid.ok()) << result_ctrl_sharedid.status();

  EXPECT_EQ(result_ctrl_linker->final_has_committed_referent,
           result_ctrl_sharedid->final_has_committed_referent);
  EXPECT_EQ(result_ctrl_linker->final_committed_referent_shared_id,
           result_ctrl_sharedid->final_committed_referent_shared_id);
  ASSERT_EQ(result_ctrl_linker->frames.size(),
           result_ctrl_sharedid->frames.size());
  for (size_t i = 0; i < result_ctrl_linker->frames.size(); ++i) {
    const auto& fl = result_ctrl_linker->frames[i];
    const auto& fs = result_ctrl_sharedid->frames[i];
    EXPECT_NEAR(fl.tgt_x, fs.tgt_x, 1e-9) << "frame " << i;
    EXPECT_NEAR(fl.tgt_y, fs.tgt_y, 1e-9) << "frame " << i;
    EXPECT_NEAR(fl.tgt_yaw, fs.tgt_yaw, 1e-9) << "frame " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    InProcessAndGrpc, HarnessRegressionTransportTest,
    ::testing::Values(Transport::kInProcess, Transport::kGrpcTwoSessions),
    [](const ::testing::TestParamInfo<Transport>& info) {
      return info.param == Transport::kGrpcTwoSessions ? "Grpc" : "InProcess";
    });

// for every R1-R8 regression scenario's REAL emitted
// trace.jsonl (not a hand-authored fixture), ev=relation's "phase" must be
// one of SPEC S3's vocabulary {"approach","engage","release"} -- never the
// internal RelationPhaseTracker names ("dormant"/"active"/"driving"/
// "contact", which now only appear in "tracker_phase") -- and no ev=payload
// / ev=target line, nor the anchor embedded in any ev=relation "relation" id
// string, ever carries an empty anchor name. See spatialblend/base/common/
// trace.h's phase-vocabulary comment (mirrored in api/c/trace.format.md) for the mapping
// this guards.
TEST(C16TraceSpecVocabulary, AllEightRegressionScenariosUseSpecPhasesAndNonEmptyAnchors) {
  static constexpr const char* kFixtures[] = {
      "R1-clone-at-center-mismatched-ids.scenario.json",
      "R2-corner-clamp.scenario.json",
      "R3-phantom-centroid.scenario.json",
      "R4-tv-desk-wrong-object.scenario.json",
      "R5-avatar-avatar-chatter.scenario.json",
      "R6-standing-jitter.scenario.json",
      "R7-fast-approach-large-steps.scenario.json",
      "R8-source-rate-decoupling.scenario.json",
  };
  static const std::unordered_map<std::string, bool> kSpecPhases = {
      {"approach", true}, {"engage", true}, {"release", true}};
  static const std::unordered_map<std::string, bool> kTrackerPhases = {
      {"dormant", true}, {"active", true}, {"driving", true}, {"contact", true}};

  int total_relation_lines = 0;
  int total_payload_lines = 0;
  int total_target_lines = 0;
  for (const char* file : kFixtures) {
    auto opts = FreshOptions(std::string("c16_") + file);
    auto result = RunFixture(file, opts);
    ASSERT_TRUE(result.ok()) << file << ": " << result.status();
    ASSERT_FALSE(result->trace_path.empty()) << file;

    std::ifstream in(result->trace_path);
    ASSERT_TRUE(in.good()) << result->trace_path;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      nlohmann::json ev;
      try {
        ev = nlohmann::json::parse(line);
      } catch (const nlohmann::json::parse_error&) {
        continue;  // legacy text-format line, not JSONL; not this test's concern.
      }
      std::string kind = ev.value("ev", std::string());
      if (kind == "relation") {
        ++total_relation_lines;
        std::string phase = ev.value("phase", std::string());
        EXPECT_TRUE(kSpecPhases.count(phase))
            << file << ": relation phase '" << phase
            << "' not in SPEC S3 vocabulary {approach,engage,release}: " << line;
        std::string tracker_phase = ev.value("tracker_phase", std::string());
        EXPECT_TRUE(kTrackerPhases.count(tracker_phase))
            << file << ": missing/unexpected tracker_phase: " << line;
        std::string relation = ev.value("relation", std::string());
        size_t colon = relation.find(':');
        ASSERT_NE(colon, std::string::npos) << file << ": " << relation;
        EXPECT_FALSE(relation.substr(colon + 1).empty())
            << file << ": empty anchor embedded in relation id " << relation;
      } else if (kind == "payload") {
        ++total_payload_lines;
        EXPECT_FALSE(ev.value("anchor", std::string()).empty())
            << file << ": empty payload anchor: " << line;
      } else if (kind == "target") {
        ++total_target_lines;
        EXPECT_FALSE(ev.value("anchor", std::string()).empty())
            << file << ": empty target anchor: " << line;
      }
    }
  }
  // Sanity: these scenarios really do exercise all three ev kinds this test
  // checks -- a silent 0-line pass (e.g. from a broken trace_path) would
  // otherwise defeat the point.
  EXPECT_GT(total_relation_lines, 0);
  EXPECT_GT(total_payload_lines, 0);
  EXPECT_GT(total_target_lines, 0);
}

// -----------------------------------------------------------------------
// Peer/target space clock not frozen at t=0.0. Root cause
// (see space.h's space_clock_ doc / space.cc's OnRemoteUpdate): Space::
// space_clock_ previously only ever advanced inside UpdateActorState, so
// under Transport::kGrpcTwoSessions the peer space -- whose only actor is a
// remotely-driven clone that never itself calls UpdateActorState -- emitted
// EVERY "share"/"payload"/"target"/"tick" trace line with "t":0.0 for the
// entire run (found by the harness-determinism investigation; see
// unity_twin.h's Transport doc comment). Fixed by advancing space_clock_
// inside OnRemoteUpdate too, using the update's own step_duration.
// -----------------------------------------------------------------------
TEST(HarnessRegressionTest, TargetSpaceTraceClockAdvancesUnderGrpc) {
  auto opts = FreshOptions("c20_target_clock_grpc");
  opts.transport = Transport::kGrpcTwoSessions;
  auto result = RunFixture("R3-phantom-centroid.scenario.json", opts);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_FALSE(result->trace_path.empty());

  std::ifstream in(result->trace_path);
  ASSERT_TRUE(in.good());
  std::string line;
  std::set<int64_t> src_spaces;
  std::vector<std::pair<int64_t, double>> ordered_events;  // (space, t), file order
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    nlohmann::json ev;
    try {
      ev = nlohmann::json::parse(line);
    } catch (const nlohmann::json::parse_error&) {
      continue;
    }
    if (!ev.contains("space") || !ev.contains("t")) continue;
    int64_t space_id = ev["space"].get<int64_t>();
    double t = ev["t"].get<double>();
    std::string kind = ev.value("ev", std::string());
    if (kind == "src") src_spaces.insert(space_id);
    ordered_events.emplace_back(space_id, t);
  }
  ASSERT_FALSE(src_spaces.empty())
      << "no ev=src lines found -- fixture didn't drive a source actor";

  // The target/peer space is whichever space_id emitted trace lines but
  // NEVER emitted an ev=src line -- only the actively-driven source space
  // calls UpdateActorState, the sole emitter of ev=src.
  std::set<int64_t> peer_spaces;
  for (const auto& [space_id, t] : ordered_events) {
    (void)t;
    if (!src_spaces.count(space_id)) peer_spaces.insert(space_id);
  }
  ASSERT_FALSE(peer_spaces.empty())
      << "no peer/target space found in trace -- fixture may not be "
         "multi-space";

  for (int64_t peer : peer_spaces) {
    double max_t = 0.0;
    double prev_t = -1.0;
    int n = 0;
    bool saw_decrease = false;
    for (const auto& [space_id, t] : ordered_events) {
      if (space_id != peer) continue;
      ++n;
      if (prev_t >= 0.0 && t < prev_t) saw_decrease = true;
      prev_t = t;
      max_t = std::max(max_t, t);
    }
    EXPECT_GT(n, 0) << "peer space " << peer << " had no trace lines";
    EXPECT_GT(max_t, 0.0)
        << "peer space " << peer << "'s trace lines are all t=0.0 -- clock "
           "never advanced (regression)";
    EXPECT_FALSE(saw_decrease)
        << "peer space " << peer << "'s trace line \"t\" values are not "
           "monotonically non-decreasing in file order";
  }
}

// -----------------------------------------------------------------------
// Dedupe the observed ~1/10-runs duplicate first-delivery bug on
// the Grpc transport (see remote_update.h's `sequence` field and space.cc's
// OnRemoteUpdate dedupe check) by asserting the payload/target line COUNTS
// for one fixture+seed are identical between kInProcess and
// kGrpcTwoSessions. Pre-fix, Grpc would occasionally emit one extra
// "payload"/"target"/"tick" triple for the very first delivered update,
// right after sb_connect_space's bootstrap/Subscribe-stream-settle window --
// this equality catches exactly that. Intended to be run with
// `--runs_per_test=30` to empirically confirm the fix holds across many
// runs; a single run only
// probabilistically exercises the ~1/10 race.
// -----------------------------------------------------------------------
TEST(HarnessRegressionTest, PayloadTargetLineCountsMatchAcrossTransports) {
  auto CountLines = [](const std::string& path, const char* kind) {
    std::ifstream in(path);
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      nlohmann::json ev;
      try {
        ev = nlohmann::json::parse(line);
      } catch (const nlohmann::json::parse_error&) {
        continue;
      }
      if (ev.value("ev", std::string()) == kind) ++n;
    }
    return n;
  };

  auto opts_inproc = FreshOptions("c20_linecounts_inproc");
  opts_inproc.transport = Transport::kInProcess;
  opts_inproc.seed = 909090;
  auto result_inproc =
      RunFixture("R3-phantom-centroid.scenario.json", opts_inproc);
  ASSERT_TRUE(result_inproc.ok()) << result_inproc.status();

  auto opts_grpc = FreshOptions("c20_linecounts_grpc");
  opts_grpc.transport = Transport::kGrpcTwoSessions;
  opts_grpc.seed = 909090;
  auto result_grpc = RunFixture("R3-phantom-centroid.scenario.json", opts_grpc);
  ASSERT_TRUE(result_grpc.ok()) << result_grpc.status();

  int payload_inproc = CountLines(result_inproc->trace_path, "payload");
  int payload_grpc = CountLines(result_grpc->trace_path, "payload");
  int target_inproc = CountLines(result_inproc->trace_path, "target");
  int target_grpc = CountLines(result_grpc->trace_path, "target");

  EXPECT_EQ(payload_inproc, payload_grpc)
      << "payload line count differs between transports -- possible "
         "duplicate/missing OnRemoteUpdate delivery under Grpc";
  EXPECT_EQ(target_inproc, target_grpc)
      << "target line count differs between transports -- possible "
         "duplicate/missing OnRemoteUpdate delivery under Grpc";
}

}  // namespace
}  // namespace harness
}  // namespace spatialblend
