// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_BASE_COMMON_COMMON_H_
#define SPATIALBLEND_BASE_COMMON_COMMON_H_

#include <any>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "spatialblend/base/common/types.h"

namespace spatialblend {

typedef int64_t entity_id_t;
typedef int64_t callback_handle_t;
typedef int64_t event_handle_t;
typedef int64_t listener_handle_t;
// internal id that all linked entities share
typedef int64_t abstract_entity_id_t;

constexpr int64_t kDefaultTimeoutMs = 1000;
constexpr int64_t kDefaultSpacePort = 50051;
constexpr int64_t kDefaultHubPort = 50052;
constexpr int64_t kUseCurrentTimestamp = -1;

using OnEndEventCallback = std::packaged_task<absl::Status(absl::Status)>;

inline OnEndEventCallback GetDefaultOnEndEventCallback() {
  return OnEndEventCallback([](absl::Status status) { return status; });
}

// For interaction events: logs non-ok status as warning and returns OkStatus,
// so Space treats dropped/rejected events as warnings rather than fatal errors.
OnEndEventCallback GetInteractionOnEndEventCallback();

typedef struct EntityHandle {
  EntityHandle();
  EntityHandle(space_id_t space_id, entity_id_t entity_id);

  std::string ToString() const;
  bool IsValid() const;
  bool operator==(const EntityHandle& other) const;
  bool operator!=(const EntityHandle& other) const;
  bool operator<(const EntityHandle& other) const;

  space_id_t space_id;
  entity_id_t entity_id;
} entity_handle_t;

const entity_id_t kInvalidEntityId = -1;
const entity_id_t kWildcardEntityId = -2;
constexpr listener_handle_t kInvalidListenerHandle = -1;
const callback_handle_t kInvalidCallbackHandle = -1;
const event_handle_t kInvalidEventHandle = -1;
const event_handle_t kWildcardRequestHandle = -2;
const entity_handle_t kInvalidEntityHandle = {kInvalidSpaceId,
                                              kInvalidEntityId};

enum class ProximityType {
  kAngular1D,
  kAngular2D,
  kEuclidean2D,
  kEuclidean3D,
};

const char* ProximityTypeToString(ProximityType type);

enum class InteractionEventType {
  kMonitored,
  kStart,
  kTerminate,
  kException,
};

const char* InteractionEventTypeToString(InteractionEventType type);

}  // namespace spatialblend

namespace std {

// Hash function specialization for entity_handle_t
template <>
struct hash<spatialblend::entity_handle_t> {
  std::size_t operator()(const spatialblend::entity_handle_t& handle) const {
    return std::hash<spatialblend::entity_id_t>()(handle.entity_id) ^
           (std::hash<spatialblend::space_id_t>()(handle.space_id) << 1);
  }
};

// Hash function specialization for vector
template <typename T>
struct hash<std::vector<T>> {
  std::size_t operator()(const std::vector<T>& v) const {
    std::size_t seed = v.size();
    for (const T& item : v) {
      // Combine the hash of the current item with the running hash value
      seed ^= std::hash<T>{}(item) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

}  // namespace std

#endif /* SPATIALBLEND_BASE_COMMON_COMMON_H_ */
