// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "spatialblend/api/c/spatialblend_logical_content_ingress.h"

namespace {

sb_logical_content_bytes_t Bytes(const std::string& value) {
  return {reinterpret_cast<const uint8_t*>(value.data()),
          static_cast<uint32_t>(value.size())};
}

struct Backing {
  std::string room = "room-a";
  std::string object = "board-a";
  std::string schema = "board-strokes/v1";
  std::string entry_key = "stroke-a";
  std::string entry_value = "curve-a";
  std::string mutation_key = "stroke-b";
  std::string mutation_value = "curve-b";
  sb_logical_content_entry_t entry{};
  sb_logical_content_mutation_t mutation{};
};

sb_logical_content_snapshot_t Snapshot(Backing* backing) {
  backing->entry = {Bytes(backing->entry_key), Bytes(backing->entry_value)};
  return {sizeof(sb_logical_content_snapshot_t),
          SB_LOGICAL_CONTENT_INGRESS_FEATURE_VERSION,
          Bytes(backing->room), 7, Bytes(backing->object), 3,
          Bytes(backing->schema), SB_LOGICAL_CONTENT_SCHEMA_BOARD_STROKES_V1,
          0, 1, 1, 0, &backing->entry, 1,
          sizeof(sb_logical_content_entry_t)};
}

sb_logical_content_tail_t Tail(Backing* backing) {
  backing->mutation = {1, 2, SB_LOGICAL_CONTENT_MUTATION_UPSERT, 0,
                       Bytes(backing->mutation_key), Bytes(backing->mutation_value)};
  return {sizeof(sb_logical_content_tail_t),
          SB_LOGICAL_CONTENT_INGRESS_FEATURE_VERSION,
          Bytes(backing->room), 7, Bytes(backing->object), 3,
          Bytes(backing->schema), 0, 1, 2, &backing->mutation, 1,
          sizeof(sb_logical_content_mutation_t)};
}

struct Captured {
  std::string room, object, schema, entry_key, entry_value, mutation_key,
      mutation_value;
  uint64_t revision = 0, terminal_revision = 0;
};

std::string Read(const sb_logical_content_bytes_t& value) {
  return std::string(reinterpret_cast<const char*>(value.bytes), value.size);
}

int32_t Capture(const sb_logical_content_snapshot_t* snapshot,
                const sb_logical_content_tail_t* tail, void* user_data) {
  auto* captured = static_cast<Captured*>(user_data);
  captured->room = Read(snapshot->room_id);
  captured->object = Read(snapshot->object_id);
  captured->schema = Read(snapshot->schema_id);
  captured->entry_key = Read(snapshot->entries[0].key);
  captured->entry_value = Read(snapshot->entries[0].value);
  captured->mutation_key = Read(tail->mutations[0].key);
  captured->mutation_value = Read(tail->mutations[0].value);
  captured->revision = snapshot->committed_revision;
  captured->terminal_revision = tail->terminal_revision;
  return SB_LOGICAL_CONTENT_INGRESS_OK;
}

TEST(LogicalContentIngressCTest, CopiesBoundedRecoveryUntilPollingConsumerAcceptsIt) {
  uint32_t feature = 0, snapshot_size = 0, tail_size = 0, entry_size = 0,
           mutation_size = 0;
  EXPECT_EQ(sizeof(uint32_t) * 6,
            sb_get_logical_content_ingress_feature(
                &feature, sizeof(feature), &snapshot_size, &tail_size,
                &entry_size, &mutation_size));
  EXPECT_EQ(SB_LOGICAL_CONTENT_INGRESS_FEATURE_VERSION, feature);
  EXPECT_EQ(sizeof(sb_logical_content_snapshot_t), snapshot_size);
  EXPECT_EQ(sizeof(sb_logical_content_tail_t), tail_size);
  EXPECT_EQ(sizeof(sb_logical_content_entry_t), entry_size);
  EXPECT_EQ(sizeof(sb_logical_content_mutation_t), mutation_size);

  sb_logical_content_ingress_t ingress = 0;
  ASSERT_EQ(SB_LOGICAL_CONTENT_INGRESS_OK,
            sb_create_logical_content_ingress(&ingress));
  Backing backing;
  sb_logical_content_snapshot_t snapshot = Snapshot(&backing);
  sb_logical_content_tail_t tail = Tail(&backing);
  ASSERT_EQ(SB_LOGICAL_CONTENT_INGRESS_OK,
            sb_submit_logical_content_recovery(ingress, &snapshot, &tail));
  // The native mailbox retained its own bytes before returning.  Neither the
  // original span nor the structs can retimestamp/reidentify the recovery.
  backing.room = "other-room";
  backing.entry_value = "mutated";
  backing.mutation_value = "mutated";
  snapshot.committed_revision = 99;
  tail.terminal_revision = 99;

  Captured captured;
  ASSERT_EQ(SB_LOGICAL_CONTENT_INGRESS_OK,
            sb_poll_logical_content_recovery(ingress, Capture, &captured));
  EXPECT_EQ("room-a", captured.room);
  EXPECT_EQ("board-a", captured.object);
  EXPECT_EQ("board-strokes/v1", captured.schema);
  EXPECT_EQ("stroke-a", captured.entry_key);
  EXPECT_EQ("curve-a", captured.entry_value);
  EXPECT_EQ("stroke-b", captured.mutation_key);
  EXPECT_EQ("curve-b", captured.mutation_value);
  EXPECT_EQ(1u, captured.revision);
  EXPECT_EQ(2u, captured.terminal_revision);
  EXPECT_EQ(SB_LOGICAL_CONTENT_INGRESS_EMPTY,
            sb_poll_logical_content_recovery(ingress, Capture, &captured));
  EXPECT_EQ(SB_LOGICAL_CONTENT_INGRESS_OK,
            sb_destroy_logical_content_ingress(ingress));
}

int32_t Retry(const sb_logical_content_snapshot_t*, const sb_logical_content_tail_t*,
              void*) {
  return SB_LOGICAL_CONTENT_INGRESS_REJECTED;
}

TEST(LogicalContentIngressCTest, RejectsInvalidTailAndRetainsARejectedPoll) {
  sb_logical_content_ingress_t ingress = 0;
  ASSERT_EQ(SB_LOGICAL_CONTENT_INGRESS_OK,
            sb_create_logical_content_ingress(&ingress));
  Backing backing;
  sb_logical_content_snapshot_t snapshot = Snapshot(&backing);
  sb_logical_content_tail_t tail = Tail(&backing);
  backing.mutation.predecessor_revision = 2;
  EXPECT_EQ(SB_LOGICAL_CONTENT_INGRESS_INVALID_ARGUMENT,
            sb_submit_logical_content_recovery(ingress, &snapshot, &tail));
  tail = Tail(&backing);
  ASSERT_EQ(SB_LOGICAL_CONTENT_INGRESS_OK,
            sb_submit_logical_content_recovery(ingress, &snapshot, &tail));
  EXPECT_EQ(SB_LOGICAL_CONTENT_INGRESS_REJECTED,
            sb_poll_logical_content_recovery(ingress, Retry, nullptr));
  Captured captured;
  EXPECT_EQ(SB_LOGICAL_CONTENT_INGRESS_OK,
            sb_poll_logical_content_recovery(ingress, Capture, &captured));
  EXPECT_EQ("curve-b", captured.mutation_value);
  EXPECT_EQ(SB_LOGICAL_CONTENT_INGRESS_OK,
            sb_destroy_logical_content_ingress(ingress));
}

}  // namespace
