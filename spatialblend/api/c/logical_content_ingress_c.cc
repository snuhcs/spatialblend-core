// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/api/c/spatialblend_logical_content_ingress.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t kSlotMask = 0xffffffffULL;
constexpr uint32_t kFeatureVersion = SB_LOGICAL_CONTENT_INGRESS_FEATURE_VERSION;
constexpr size_t kMaxEntries = 512;
constexpr size_t kMaxMutations = 512;
constexpr size_t kMaxIdBytes = 96;
constexpr size_t kMaxValueBytes = 4096;

uint64_t Slot(sb_logical_content_ingress_t handle) { return handle & kSlotMask; }
uint64_t Generation(sb_logical_content_ingress_t handle) { return handle >> 32; }
sb_logical_content_ingress_t Handle(uint64_t slot, uint64_t generation) {
  return (generation << 32) | slot;
}

bool Copy(const sb_logical_content_bytes_t& source, size_t maximum,
          bool required, std::string* destination) {
  if (destination == nullptr || source.size > maximum ||
      (source.size != 0 && source.bytes == nullptr) ||
      (required && source.size == 0)) {
    return false;
  }
  if (source.size == 0) {
    destination->clear();
  } else {
    destination->assign(reinterpret_cast<const char*>(source.bytes), source.size);
  }
  return true;
}

struct Entry { std::string key; std::string value; };
struct Mutation {
  uint64_t predecessor_revision = 0;
  uint64_t committed_revision = 0;
  uint32_t kind = 0;
  std::string key;
  std::string value;
};
struct Recovery {
  std::string room_id, object_id, schema_id;
  uint64_t room_epoch = 0, object_generation = 0;
  uint32_t schema = 0;
  bool tombstoned = false;
  uint64_t committed_revision = 0, tail_watermark = 0,
           compacted_through_revision = 0;
  std::vector<Entry> entries;
  uint64_t exclusive_watermark = 0, terminal_revision = 0;
  std::vector<Mutation> mutations;
};

bool SameIdentity(const Recovery& value, const sb_logical_content_tail_t& tail) {
  std::string room, object, schema;
  return Copy(tail.room_id, kMaxIdBytes, true, &room) &&
         Copy(tail.object_id, kMaxIdBytes, true, &object) &&
         Copy(tail.schema_id, kMaxIdBytes, true, &schema) &&
         room == value.room_id && object == value.object_id &&
         schema == value.schema_id && tail.room_epoch == value.room_epoch &&
         tail.object_generation == value.object_generation;
}

bool ValidSchema(uint32_t schema, const std::string& schema_id) {
  return (schema == SB_LOGICAL_CONTENT_SCHEMA_BOARD_STROKES_V1 &&
          schema_id == "board-strokes/v1") ||
         (schema == SB_LOGICAL_CONTENT_SCHEMA_KEYED_VALUES_V1 &&
          schema_id == "keyed-values/v1");
}

bool CopyRecovery(const sb_logical_content_snapshot_t* snapshot,
                  const sb_logical_content_tail_t* tail, Recovery* out) {
  if (snapshot == nullptr || tail == nullptr || out == nullptr ||
      snapshot->struct_size != sizeof(*snapshot) ||
      tail->struct_size != sizeof(*tail) ||
      snapshot->feature_version != kFeatureVersion ||
      tail->feature_version != kFeatureVersion || snapshot->tombstoned > 1 ||
      snapshot->entry_count > kMaxEntries || tail->mutation_count > kMaxMutations ||
      (snapshot->entry_count != 0 &&
       (snapshot->entries == nullptr || snapshot->entry_size != sizeof(sb_logical_content_entry_t))) ||
      (tail->mutation_count != 0 &&
       (tail->mutations == nullptr || tail->mutation_size != sizeof(sb_logical_content_mutation_t)))) {
    return false;
  }
  Recovery value;
  if (!Copy(snapshot->room_id, kMaxIdBytes, true, &value.room_id) ||
      !Copy(snapshot->object_id, kMaxIdBytes, true, &value.object_id) ||
      !Copy(snapshot->schema_id, kMaxIdBytes, true, &value.schema_id) ||
      snapshot->room_epoch == 0 || snapshot->object_generation == 0 ||
      snapshot->committed_revision == 0 ||
      snapshot->tail_watermark != snapshot->committed_revision ||
      snapshot->compacted_through_revision > snapshot->committed_revision ||
      !ValidSchema(snapshot->schema, value.schema_id)) {
    return false;
  }
  value.room_epoch = snapshot->room_epoch;
  value.object_generation = snapshot->object_generation;
  value.schema = snapshot->schema;
  value.tombstoned = snapshot->tombstoned != 0;
  value.committed_revision = snapshot->committed_revision;
  value.tail_watermark = snapshot->tail_watermark;
  value.compacted_through_revision = snapshot->compacted_through_revision;
  for (uint32_t index = 0; index < snapshot->entry_count; ++index) {
    Entry entry;
    if (!Copy(snapshot->entries[index].key, kMaxIdBytes, true, &entry.key) ||
        !Copy(snapshot->entries[index].value, kMaxValueBytes, true, &entry.value)) return false;
    value.entries.push_back(std::move(entry));
  }
  if (!SameIdentity(value, *tail) ||
      tail->exclusive_watermark != value.committed_revision ||
      tail->terminal_revision < value.committed_revision ||
      (value.tombstoned && (!value.entries.empty() || tail->mutation_count != 0 ||
                            tail->terminal_revision != value.committed_revision))) return false;
  value.exclusive_watermark = tail->exclusive_watermark;
  value.terminal_revision = tail->terminal_revision;
  uint64_t expected = value.committed_revision;
  bool terminal_tombstone = value.tombstoned;
  for (uint32_t index = 0; index < tail->mutation_count; ++index) {
    const sb_logical_content_mutation_t& input = tail->mutations[index];
    Mutation mutation;
    if (input.predecessor_revision != expected ||
        input.committed_revision != expected + 1 || terminal_tombstone ||
        !Copy(input.key, kMaxIdBytes, false, &mutation.key) ||
        !Copy(input.value, kMaxValueBytes, false, &mutation.value) ||
        (input.kind != SB_LOGICAL_CONTENT_MUTATION_UPSERT &&
         input.kind != SB_LOGICAL_CONTENT_MUTATION_ERASE &&
         input.kind != SB_LOGICAL_CONTENT_MUTATION_DESPAWN) ||
        (input.kind == SB_LOGICAL_CONTENT_MUTATION_UPSERT &&
         (mutation.key.empty() || mutation.value.empty())) ||
        (input.kind == SB_LOGICAL_CONTENT_MUTATION_ERASE &&
         (mutation.key.empty() || !mutation.value.empty())) ||
        (input.kind == SB_LOGICAL_CONTENT_MUTATION_DESPAWN &&
         (!mutation.key.empty() || !mutation.value.empty()))) return false;
    mutation.predecessor_revision = input.predecessor_revision;
    mutation.committed_revision = input.committed_revision;
    mutation.kind = input.kind;
    terminal_tombstone = input.kind == SB_LOGICAL_CONTENT_MUTATION_DESPAWN;
    value.mutations.push_back(std::move(mutation));
    expected = input.committed_revision;
  }
  if (expected != value.terminal_revision) return false;
  *out = std::move(value);
  return true;
}

class Ingress {
 public:
  int32_t Submit(Recovery value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (retired_) return SB_LOGICAL_CONTENT_INGRESS_STALE_HANDLE;
    if (queued_ || delivering_) return SB_LOGICAL_CONTENT_INGRESS_BUSY;
    queued_ = std::move(value);
    return SB_LOGICAL_CONTENT_INGRESS_OK;
  }
  int32_t Poll(sb_on_logical_content_recovery_t callback, void* user_data) {
    if (callback == nullptr) return SB_LOGICAL_CONTENT_INGRESS_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lock(mutex_);
    if (retired_) return SB_LOGICAL_CONTENT_INGRESS_STALE_HANDLE;
    if (!queued_) return SB_LOGICAL_CONTENT_INGRESS_EMPTY;
    if (delivering_) return SB_LOGICAL_CONTENT_INGRESS_BUSY;
    delivering_ = true;
    // Move the already-copied record into this poll's stack lifetime before
    // entering foreign code.  A callback may retire its handle, but the
    // pointer views remain valid for the entire callback.
    Recovery delivery = std::move(*queued_);
    queued_.reset();
    std::vector<sb_logical_content_entry_t> entries;
    std::vector<sb_logical_content_mutation_t> mutations;
    sb_logical_content_snapshot_t snapshot{};
    sb_logical_content_tail_t tail{};
    BuildViews(delivery, &entries, &mutations, &snapshot, &tail);
    lock.unlock();
    const int32_t result = callback(&snapshot, &tail, user_data);
    lock.lock();
    delivering_ = false;
    if (retired_) return SB_LOGICAL_CONTENT_INGRESS_STALE_HANDLE;
    if (result != SB_LOGICAL_CONTENT_INGRESS_OK) queued_ = std::move(delivery);
    return result;
  }
  void Retire() { std::lock_guard<std::mutex> lock(mutex_); retired_ = true; queued_.reset(); }
 private:
  static sb_logical_content_bytes_t View(const std::string& value) {
    return {reinterpret_cast<const uint8_t*>(value.data()), static_cast<uint32_t>(value.size())};
  }
  static void BuildViews(const Recovery& value,
                         std::vector<sb_logical_content_entry_t>* entries,
                         std::vector<sb_logical_content_mutation_t>* mutations,
                         sb_logical_content_snapshot_t* snapshot,
                         sb_logical_content_tail_t* tail) {
    entries->reserve(value.entries.size());
    for (const Entry& entry : value.entries) entries->push_back({View(entry.key), View(entry.value)});
    mutations->reserve(value.mutations.size());
    for (const Mutation& mutation : value.mutations) mutations->push_back({mutation.predecessor_revision, mutation.committed_revision, mutation.kind, 0, View(mutation.key), View(mutation.value)});
    *snapshot = {sizeof(*snapshot), kFeatureVersion, View(value.room_id), value.room_epoch,
                 View(value.object_id), value.object_generation, View(value.schema_id), value.schema,
                 value.tombstoned ? 1u : 0u, value.committed_revision, value.tail_watermark,
                 value.compacted_through_revision, entries->empty() ? nullptr : entries->data(),
                 static_cast<uint32_t>(entries->size()), sizeof(sb_logical_content_entry_t)};
    *tail = {sizeof(*tail), kFeatureVersion, View(value.room_id), value.room_epoch,
             View(value.object_id), value.object_generation, View(value.schema_id), 0,
             value.exclusive_watermark, value.terminal_revision,
             mutations->empty() ? nullptr : mutations->data(), static_cast<uint32_t>(mutations->size()),
             sizeof(sb_logical_content_mutation_t)};
  }
  std::mutex mutex_;
  bool retired_ = false;
  bool delivering_ = false;
  std::optional<Recovery> queued_;
};

struct SlotValue { uint64_t generation = 0; std::shared_ptr<Ingress> ingress; };
std::mutex g_mutex;
std::map<uint64_t, SlotValue> g_ingresses;
std::vector<uint64_t> g_free_slots;
uint64_t g_next_slot = 0;

std::shared_ptr<Ingress> Lookup(sb_logical_content_ingress_t handle) {
  const uint64_t slot = Slot(handle), generation = Generation(handle);
  if (slot == 0 || generation == 0) return nullptr;
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto found = g_ingresses.find(slot);
  return found == g_ingresses.end() || found->second.generation != generation
             ? nullptr : found->second.ingress;
}

}  // namespace

extern "C" {

uint32_t sb_get_logical_content_ingress_feature(
    uint32_t* feature_version, uint32_t feature_size, uint32_t* snapshot_size,
    uint32_t* tail_size, uint32_t* entry_size, uint32_t* mutation_size) {
  if (feature_version != nullptr && feature_size >= sizeof(*feature_version)) *feature_version = kFeatureVersion;
  if (snapshot_size != nullptr) *snapshot_size = sizeof(sb_logical_content_snapshot_t);
  if (tail_size != nullptr) *tail_size = sizeof(sb_logical_content_tail_t);
  if (entry_size != nullptr) *entry_size = sizeof(sb_logical_content_entry_t);
  if (mutation_size != nullptr) *mutation_size = sizeof(sb_logical_content_mutation_t);
  return sizeof(uint32_t) * 6;
}

int32_t sb_create_logical_content_ingress(sb_logical_content_ingress_t* out_ingress) {
  if (out_ingress == nullptr) return SB_LOGICAL_CONTENT_INGRESS_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lock(g_mutex);
  const uint64_t slot = g_free_slots.empty() ? ++g_next_slot : g_free_slots.back();
  if (!g_free_slots.empty()) g_free_slots.pop_back();
  SlotValue& value = g_ingresses[slot];
  ++value.generation;
  if (value.generation == 0) ++value.generation;
  value.ingress = std::make_shared<Ingress>();
  *out_ingress = Handle(slot, value.generation);
  return SB_LOGICAL_CONTENT_INGRESS_OK;
}

int32_t sb_destroy_logical_content_ingress(sb_logical_content_ingress_t ingress) {
  const uint64_t slot = Slot(ingress), generation = Generation(ingress);
  std::shared_ptr<Ingress> value;
  { std::lock_guard<std::mutex> lock(g_mutex);
    const auto found = g_ingresses.find(slot);
    if (slot == 0 || generation == 0 || found == g_ingresses.end() || found->second.generation != generation)
      return SB_LOGICAL_CONTENT_INGRESS_STALE_HANDLE;
    value = std::move(found->second.ingress); g_free_slots.push_back(slot); }
  value->Retire();
  return SB_LOGICAL_CONTENT_INGRESS_OK;
}

int32_t sb_submit_logical_content_recovery(sb_logical_content_ingress_t ingress,
                                            const sb_logical_content_snapshot_t* snapshot,
                                            const sb_logical_content_tail_t* tail) {
  const std::shared_ptr<Ingress> value = Lookup(ingress);
  if (!value) return SB_LOGICAL_CONTENT_INGRESS_STALE_HANDLE;
  Recovery copied;
  return CopyRecovery(snapshot, tail, &copied) ? value->Submit(std::move(copied))
                                                : SB_LOGICAL_CONTENT_INGRESS_INVALID_ARGUMENT;
}

int32_t sb_poll_logical_content_recovery(sb_logical_content_ingress_t ingress,
                                          sb_on_logical_content_recovery_t callback,
                                          void* user_data) {
  const std::shared_ptr<Ingress> value = Lookup(ingress);
  return value ? value->Poll(callback, user_data) : SB_LOGICAL_CONTENT_INGRESS_STALE_HANDLE;
}

}  // extern "C"
