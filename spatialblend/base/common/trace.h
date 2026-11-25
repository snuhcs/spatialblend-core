// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_COMMON_TRACE_H_
#define SPATIALBLEND_COMMON_TRACE_H_

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>

#include "nlohmann/json.hpp"
#include "spatialblend/base/common/types.h"

namespace spatialblend {

// Version of the JSON-lines trace format written by Space (the
// "format" field of ev=meta and ev=policy). The format is specified in
// spatialblend/api/c/trace.format.md; bump this with every change to an
// emitted event's field set or semantics, and record the change there.
//   1 -- src, payload, target, relation, share, tick.
//   2 -- adds meta, policy, channel, goal, scope; payload.sub_region /
//        payload.driving; target.snap_m / snapped / requested_pos /
//        reach_clamped / reach_m; relation.referent.
//   3 -- adds mediator (paced state, mode, budget per TickAvatar);
//        target.snap_m / snapped report the free-space projection.
//   4 -- scope-gating reasons; payload.dist for
//        AffordancePose.
//   5 -- target drops requested_pos / reach_clamped / reach_m, gains
//        beyond_reach / reach_excess_m (reach flags, never moves).
//   6 -- adds qualify (seat entry qualification with the
//        stillness speed smoothed over a trailing window).
//   7 -- target gains objective_before / objective_after /
//        objective_iterations / objective_fallback: the paper
//        objective (SpatialBlend App. A eq. weighted) at the goal-point initializer and after the bounded
//        local refinement, for the root target's solve only (same
//        root-only scope as snap_m / snapped).
//   8 -- adds proposal (what core proposes for a receiver
//        root, with its cause bitmask and the embodied sample it used) and
//        removes mediator (core no longer paces).
inline constexpr int kTraceFormatVersion = 10;

// SPEC S3 (spatialblend/api/c/trace.format.md) structured trace sink: one JSON object per
// line, one event per Emit() call. The process-wide instance (Instance())
// lazily resolves its default (env-driven) activation state exactly once;
// sb_set_trace overrides that explicitly via SetPath()/Disable().
//
// Per-space routing: before it, EVERY space in a process shared this one
// Instance() -- sb_set_trace(space, path)'s `space` argument only validated
// the handle, so two spaces in one process could never write to two
// different trace files; the only way to tell them apart downstream was the
// "space" field embedded in each JSON line (a post-hoc filter).
// ForSpace(space_id) is now the actual entry point every
// Space:: emit call site uses (keyed on its own space_id_): it resolves to a
// DEDICATED sink for that space_id if one was ever bound via
// SetPathForSpace()/DisableForSpace() (i.e. sb_set_trace was called for that
// specific space), else falls back to Instance() unchanged -- so any space
// that never gets its own sb_set_trace call keeps writing to the one shared
// process-wide sink exactly as before. Calling sb_set_trace with the SAME
// path for every space in a run reproduces the old "one shared trace.jsonl"
// behavior: every space_id bound to an identical path string is routed to
// the exact same underlying sink (one mutex, one FILE*), not merely to
// separate handles that both happen to target the same file -- see
// SetPathForSpace's doc for why that sharing is a correctness requirement,
// not just tidiness. Different paths per space genuinely separate them into
// different files (each with its own sink).
//
// Perf contract: when tracing is inactive, Active() is a single atomic load
// and callers are expected to guard all JSON-object construction behind it
// (`if (sink.Active()) { ... build ev ... sink.Emit(ev); }`) so the inactive
// hot path never builds a json object or touches the mutex.
//
// ev=relation phase vocabulary (SPEC S3 <-> RelationPhaseTracker mapping)
// -----------------------------------------------------------------------
// TraceSink itself has no notion of relation phases -- this is documentation
// for the one caller that builds ev=relation lines (Space::DispatchRelationEvents,
// in spatialblend/space/space.cc), kept here (and mirrored verbatim in
// spatialblend/api/c/trace.format.md) as the single source of truth for the mapping so both
// sides of the contract read identically.
//
// RelationPhaseTracker's internal 4-state chain (kDormant/kActive/kDriving/
// kContact, see relation_phase_tracker.h) is NOT the vocabulary SPEC S3's
// "phase" field uses. Every ev=relation line's "phase" is translated to one of
// three spec states, chosen purely from the transition's `to` state:
//
//   to == kDormant                    -> "release"
//   to == kActive  (from kDormant)     -> "approach"
//   to == kDriving (from kActive)      -> "approach"
//   to == kContact (from kActive/      -> "engage"
//                   kDriving/kDormant)
//
// i.e. dormant -> {active, driving} = "approach" (the relation is newly
// forming); anything -> contact = "engage" (grounded contact reached);
// {active, driving, contact} -> dormant = "release" (the relation ended).
// Nothing is lost in this collapse: every ev=relation line ALSO carries a
// "tracker_phase" field holding the original internal state name ("dormant" |
// "active" | "driving" | "contact", the internal tracker vocabulary), so a consumer
// that wants the finer-grained internal state can still get it.
class TraceSink {
 public:
  // The process-wide DEFAULT trace sink: env-driven (SB_V2_TRACE /
  // SB_V2_TRACE_FILE), and the fallback every space_id resolves to via
  // ForSpace() until that specific space_id is bound to its own sink (see
  // ForSpace/SetPathForSpace/DisableForSpace below). Kept as the direct
  // entry point for callers that have no particular space in mind (e.g. a
  // single-space test that never calls SetPathForSpace).
  static TraceSink& Instance();

  // The sink a given space_id should emit through. Returns the
  // dedicated per-space sink if `space_id` was ever bound via
  // SetPathForSpace()/DisableForSpace() (directly, or via sb_set_trace(space,
  // ...) at the C API), otherwise falls back to Instance() -- the process-
  // wide env-driven default -- so single-space callers and every
  // caller that never binds a specific space_id keep their exact previous
  // behavior (one shared sink for the whole process). Every Space:: emit
  // call site (space.cc) resolves through this, keyed on its own space_id_,
  // instead of calling Instance() directly.
  static TraceSink& ForSpace(space_id_t space_id);

  // sb_set_trace(space, path): bind `space_id` to the sink for `path`,
  // opening it lazily on first use -- this is what makes it possible to
  // separate two spaces' traces into two different files within one process
  // (before per-space routing, sb_set_trace's `space` argument only
  // validated the handle; every space shared the one process-wide Instance() sink regardless of
  // which handle was passed). Binding two DIFFERENT space_ids to the SAME
  // path string routes both through the exact same underlying sink (one
  // mutex, one FILE*) rather than each opening its own separate handle onto
  // that path -- two independent stdio streams both appending to the same
  // file from different threads (e.g. the harness main thread and a peer's
  // own gRPC completion-queue thread, see unity_twin.h) would not be
  // line-atomic the way Emit()'s single mutex is, so sharing here is a
  // correctness requirement, not just an optimization. This is exactly what
  // reproduces the earlier "one shared trace file for the whole run"
  // behavior when every space is bound to the same path.
  static void SetPathForSpace(space_id_t space_id, const std::string& path);

  // sb_set_trace(space, NULL): explicitly disable tracing for `space_id`
  // only, giving it its own private, permanently-inactive sink (it writes
  // nothing, so there is nothing to share), overriding env for that
  // space_id specifically. Does not affect Instance() or any other
  // space_id's sink -- including one it may have previously shared a path
  // with via SetPathForSpace (that path's sink keeps running for every
  // other space_id still bound to it).
  static void DisableForSpace(space_id_t space_id);

  // A space whose id changed (see Space::SetSpaceId) keeps the sink it was
  // bound to: the binding for `from`, if any, moves to `to`.
  static void MoveSpaceBinding(space_id_t from, space_id_t to);

  // Test hook: drop every per-space sink created via SetPathForSpace/
  // DisableForSpace/ForSpace, closing any open file handles first, so a test
  // that binds per-space tracing doesn't leak file handles or routing state
  // into the next test. Instance() itself is untouched -- reset it
  // separately via Instance().ResetForTest(). Safe to call even if no
  // per-space sink was ever created.
  static void ResetAllForTest();

  // sb_set_trace(space, path): a non-empty path opens that file for append
  // (line-buffered) and activates tracing, overriding env. Falls back to
  // stderr if the file cannot be opened (still activates tracing).
  void SetPath(const std::string& path);

  // sb_set_trace(space, NULL): explicitly disables tracing, overriding env.
  void Disable();

  // True if tracing is currently active. Single relaxed atomic load once the
  // env default has been resolved (resolution itself happens at most once).
  bool Active();

  // Serializes `ev` as one JSON line (plus trailing newline) and writes it.
  // Thread-safe (single mutex). No-op if !Active() -- but callers should
  // still gate construction of `ev` behind Active() to avoid paying for the
  // json object when tracing is off.
  void Emit(const nlohmann::ordered_json& ev);

  // Test hook: reset to default (lazy/unresolved) state and close any open
  // file. Safe to call between tests that toggle SB_V2_TRACE* env vars.
  void ResetForTest();

 private:
  TraceSink() = default;
  void EnsureEnvResolved();

  std::mutex mu_;
  std::atomic<bool> env_resolved_{false};
  std::atomic<bool> active_{false};
  std::FILE* file_ = stderr;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_COMMON_TRACE_H_
