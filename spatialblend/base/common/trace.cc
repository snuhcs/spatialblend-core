// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/base/common/trace.h"

#include <cstdlib>
#include <memory>
#include <unordered_map>

namespace spatialblend {

namespace {

// Per-space sink routing. Never destroyed (same "leak on purpose"
// pattern as Instance()'s own static -- these sinks own a FILE* that must
// outlive every other static destructor that might still be emitting during
// process teardown).
std::mutex& RegistryMutex() {
  static std::mutex* mu = new std::mutex();
  return *mu;
}

// path -> the ONE TraceSink target that path resolves to. Two space_ids
// bound to the identical path string (SetPathForSpace called with the same
// `path` for both) resolve to this SAME shared_ptr -- i.e. the SAME mutex
// and the SAME underlying FILE* -- instead of each opening its own separate
// fopen() handle onto that path. That sharing matters: two independent
// stdio FILE* streams both appending to the same underlying file from
// different threads (e.g. the harness main thread driving the source space
// and a peer's own gRPC completion-queue thread, see unity_twin.h) is not
// guaranteed line-atomic the way one shared mutex-guarded FILE* is -- this
// is exactly the concurrency Emit()'s single mutex per TraceSink already
// protects against for one process-wide Instance(), and per-space routing
// must not reintroduce it just because two spaces happen to share a path.
std::unordered_map<std::string, std::shared_ptr<TraceSink>>& PathTargets() {
  static auto* targets =
      new std::unordered_map<std::string, std::shared_ptr<TraceSink>>();
  return *targets;
}

// space_id -> the sink ForSpace(space_id) currently resolves to: either a
// PathTargets() entry shared with every other space_id bound to the same
// path (SetPathForSpace), or a private, never-shared, permanently-inactive
// instance (DisableForSpace).
std::unordered_map<space_id_t, std::shared_ptr<TraceSink>>& SpaceTargets() {
  static auto* targets =
      new std::unordered_map<space_id_t, std::shared_ptr<TraceSink>>();
  return *targets;
}

}  // namespace

TraceSink& TraceSink::Instance() {
  static TraceSink* instance = new TraceSink();
  return *instance;
}

TraceSink& TraceSink::ForSpace(space_id_t space_id) {
  {
    std::lock_guard<std::mutex> lock(RegistryMutex());
    auto it = SpaceTargets().find(space_id);
    if (it != SpaceTargets().end() && it->second) return *it->second;
  }
  return Instance();
}

void TraceSink::SetPathForSpace(space_id_t space_id, const std::string& path) {
  std::shared_ptr<TraceSink> target;
  bool newly_opened_path = false;
  {
    std::lock_guard<std::mutex> lock(RegistryMutex());
    auto& slot = PathTargets()[path];
    if (!slot) {
      // Constructor is private; this factory function is itself a TraceSink
      // member (of the enclosing class), so it has access.
      slot = std::shared_ptr<TraceSink>(new TraceSink());
      newly_opened_path = true;
    }
    target = slot;
    SpaceTargets()[space_id] = target;
  }
  // Only the FIRST space to bind this exact path string actually opens the
  // file -- every subsequent space_id bound to the same path shares that
  // already-open target untouched (calling SetPath() again would pointlessly
  // fclose()+re-fopen() the same file every time another space joins it).
  if (newly_opened_path) target->SetPath(path);
}

void TraceSink::DisableForSpace(space_id_t space_id) {
  // A disabled space never shares state with any other space_id (nothing to
  // share -- it writes nothing), so it always gets its own private,
  // permanently-inactive instance rather than touching PathTargets().
  auto target = std::shared_ptr<TraceSink>(new TraceSink());
  target->Disable();
  std::lock_guard<std::mutex> lock(RegistryMutex());
  SpaceTargets()[space_id] = std::move(target);
}

void TraceSink::MoveSpaceBinding(space_id_t from, space_id_t to) {
  if (from == to) return;
  std::lock_guard<std::mutex> lock(RegistryMutex());
  auto it = SpaceTargets().find(from);
  if (it == SpaceTargets().end()) return;
  SpaceTargets()[to] = std::move(it->second);
  SpaceTargets().erase(from);
}

void TraceSink::ResetAllForTest() {
  std::lock_guard<std::mutex> lock(RegistryMutex());
  // Close every still-open file before dropping the sinks (ResetForTest()
  // already does this per-instance; shared_ptr destruction alone would skip
  // the fclose()). Safe to call twice on the same shared instance (once via
  // each map below) -- ResetForTest() itself is idempotent.
  for (auto& [path, sink] : PathTargets()) {
    if (sink) sink->ResetForTest();
  }
  for (auto& [id, sink] : SpaceTargets()) {
    if (sink) sink->ResetForTest();
  }
  PathTargets().clear();
  SpaceTargets().clear();
}

void TraceSink::EnsureEnvResolved() {
  if (env_resolved_.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(mu_);
  if (env_resolved_.load(std::memory_order_relaxed)) return;

  bool active = false;
  std::FILE* file = stderr;
  if (const char* path = std::getenv("SB_V2_TRACE_FILE")) {
    if (path[0] != '\0') {
      std::FILE* f = std::fopen(path, "a");
      if (f) {
        file = f;
      }
      active = true;
    }
  }
  if (!active) {
    if (const char* flag = std::getenv("SB_V2_TRACE")) {
      if (flag[0] != '\0' && flag[0] != '0') {
        active = true;
      }
    }
  }
  file_ = file;
  active_.store(active, std::memory_order_relaxed);
  env_resolved_.store(true, std::memory_order_release);
}

void TraceSink::SetPath(const std::string& path) {
  std::lock_guard<std::mutex> lock(mu_);
  if (file_ != nullptr && file_ != stderr) {
    std::fclose(file_);
  }
  std::FILE* f = path.empty() ? stderr : std::fopen(path.c_str(), "a");
  if (!f) {
    f = stderr;
  }
  file_ = f;
  active_.store(true, std::memory_order_relaxed);
  env_resolved_.store(true, std::memory_order_release);
}

void TraceSink::Disable() {
  std::lock_guard<std::mutex> lock(mu_);
  if (file_ != nullptr && file_ != stderr) {
    std::fclose(file_);
  }
  file_ = stderr;
  active_.store(false, std::memory_order_relaxed);
  env_resolved_.store(true, std::memory_order_release);
}

bool TraceSink::Active() {
  EnsureEnvResolved();
  return active_.load(std::memory_order_relaxed);
}

void TraceSink::Emit(const nlohmann::ordered_json& ev) {
  if (!Active()) return;
  std::lock_guard<std::mutex> lock(mu_);
  if (!file_) return;
  std::fprintf(file_, "%s\n", ev.dump().c_str());
  // One event per line, flushed: line buffering without setvbuf(_IOLBF, 0),
  // which the MSVC UCRT rejects (0xC0000409) and where _IOLBF is full buffering.
  std::fflush(file_);
}

void TraceSink::ResetForTest() {
  std::lock_guard<std::mutex> lock(mu_);
  if (file_ != nullptr && file_ != stderr) {
    std::fclose(file_);
  }
  file_ = stderr;
  active_.store(false, std::memory_order_relaxed);
  env_resolved_.store(false, std::memory_order_relaxed);
}

}  // namespace spatialblend
