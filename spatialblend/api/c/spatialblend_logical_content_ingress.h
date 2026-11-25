// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SPATIALBLEND_LOGICAL_CONTENT_INGRESS_H_
#define SPATIALBLEND_C_SPATIALBLEND_LOGICAL_CONTENT_INGRESS_H_

// Additive read-only recovery boundary for logical content.  It is
// intentionally an in-process copied mailbox, not a transport, object authoring
// API, or local binding registry.  Admission happens before Submit; Unity owns
// the receipt and decides whether the copied record is applied.
#include <stdint.h>

#ifndef SB_EXPORT
#if defined(_WIN32)
#if defined(SPATIALBLEND_BUILD_DLL)
#define SB_EXPORT __declspec(dllexport)
#else
#define SB_EXPORT __declspec(dllimport)
#endif
#else
#define SB_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SB_LOGICAL_CONTENT_INGRESS_FEATURE_VERSION 1u

typedef uint64_t sb_logical_content_ingress_t;

typedef struct {
  const uint8_t* bytes;
  uint32_t size;
} sb_logical_content_bytes_t;

typedef enum {
  SB_LOGICAL_CONTENT_INGRESS_OK = 1,
  SB_LOGICAL_CONTENT_INGRESS_INVALID_ARGUMENT = 0,
  SB_LOGICAL_CONTENT_INGRESS_STALE_HANDLE = -3,
  SB_LOGICAL_CONTENT_INGRESS_BUSY = -5,
  SB_LOGICAL_CONTENT_INGRESS_EMPTY = -6,
  SB_LOGICAL_CONTENT_INGRESS_REJECTED = -7,
} sb_logical_content_ingress_result_t;

typedef enum {
  SB_LOGICAL_CONTENT_SCHEMA_BOARD_STROKES_V1 = 1,
  SB_LOGICAL_CONTENT_SCHEMA_KEYED_VALUES_V1 = 2,
} sb_logical_content_schema_t;

typedef enum {
  SB_LOGICAL_CONTENT_MUTATION_UPSERT = 1,
  SB_LOGICAL_CONTENT_MUTATION_ERASE = 2,
  SB_LOGICAL_CONTENT_MUTATION_DESPAWN = 3,
} sb_logical_content_mutation_kind_t;

typedef struct {
  sb_logical_content_bytes_t key;
  sb_logical_content_bytes_t value;
} sb_logical_content_entry_t;

typedef struct {
  uint64_t predecessor_revision;
  uint64_t committed_revision;
  uint32_t kind;
  uint32_t reserved;
  sb_logical_content_bytes_t key;
  sb_logical_content_bytes_t value;
} sb_logical_content_mutation_t;

// All spans are copied before Submit returns.  The snapshot carries only
// durable identity/recovery data; it cannot carry a grant, participant,
// native Space/Transform, source clock, or authority-creation request.
typedef struct {
  uint32_t struct_size;
  uint32_t feature_version;
  sb_logical_content_bytes_t room_id;
  uint64_t room_epoch;
  sb_logical_content_bytes_t object_id;
  uint64_t object_generation;
  sb_logical_content_bytes_t schema_id;
  uint32_t schema;
  uint32_t tombstoned;
  uint64_t committed_revision;
  uint64_t tail_watermark;
  uint64_t compacted_through_revision;
  const sb_logical_content_entry_t* entries;
  uint32_t entry_count;
  uint32_t entry_size;
} sb_logical_content_snapshot_t;

typedef struct {
  uint32_t struct_size;
  uint32_t feature_version;
  sb_logical_content_bytes_t room_id;
  uint64_t room_epoch;
  sb_logical_content_bytes_t object_id;
  uint64_t object_generation;
  sb_logical_content_bytes_t schema_id;
  uint32_t reserved;
  uint64_t exclusive_watermark;
  uint64_t terminal_revision;
  const sb_logical_content_mutation_t* mutations;
  uint32_t mutation_count;
  uint32_t mutation_size;
} sb_logical_content_tail_t;

// The callback runs synchronously on the thread that polls.  It sees native
// owned bytes that remain valid only for the callback.  It must copy before
// returning.  Return SB_LOGICAL_CONTENT_INGRESS_OK to consume the record;
// any other value leaves it queued for a later retry.
typedef int32_t (*sb_on_logical_content_recovery_t)(
    const sb_logical_content_snapshot_t* snapshot,
    const sb_logical_content_tail_t* tail, void* user_data);

// Size probe for managed ABI checks.  It never writes a short/null output.
SB_EXPORT uint32_t sb_get_logical_content_ingress_feature(
    uint32_t* feature_version, uint32_t feature_size,
    uint32_t* snapshot_size, uint32_t* tail_size,
    uint32_t* entry_size, uint32_t* mutation_size);

SB_EXPORT int32_t sb_create_logical_content_ingress(
    sb_logical_content_ingress_t* out_ingress);
SB_EXPORT int32_t sb_destroy_logical_content_ingress(
    sb_logical_content_ingress_t ingress);

// One bounded recovery is retained until its poll callback accepts it.  This
// admission does not authenticate, commit, send, or apply the logical object.
SB_EXPORT int32_t sb_submit_logical_content_recovery(
    sb_logical_content_ingress_t ingress,
    const sb_logical_content_snapshot_t* snapshot,
    const sb_logical_content_tail_t* tail);
SB_EXPORT int32_t sb_poll_logical_content_recovery(
    sb_logical_content_ingress_t ingress,
    sb_on_logical_content_recovery_t callback, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPATIALBLEND_C_SPATIALBLEND_LOGICAL_CONTENT_INGRESS_H_
