// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/base/common/common.h"

#include <cstring>

#include "absl/strings/str_format.h"

namespace spatialblend {

const char* ProximityTypeToString(ProximityType type) {
  switch (type) {
    case ProximityType::kAngular1D:
      return "Angular1D";
    case ProximityType::kAngular2D:
      return "Angular2D";
    case ProximityType::kEuclidean2D:
      return "Euclidean2D";
    case ProximityType::kEuclidean3D:
      return "Euclidean3D";
    default:
      return "Unknown";
  }
}

const char* InteractionEventTypeToString(InteractionEventType type) {
  switch (type) {
    case InteractionEventType::kMonitored:
      return "Monitored";
    case InteractionEventType::kStart:
      return "Start";
    case InteractionEventType::kTerminate:
      return "Terminate";
    case InteractionEventType::kException:
      return "Exception";
    default:
      return "Unknown";
  }
}

EntityHandle::EntityHandle()
    : space_id(kInvalidSpaceId), entity_id(kInvalidEntityId) {}

EntityHandle::EntityHandle(space_id_t space_id, entity_id_t entity_id)
    : space_id(space_id), entity_id(entity_id) {}

std::string EntityHandle::ToString() const {
  auto id_to_string = [](entity_id_t id) -> std::string {
    if (id == kInvalidEntityId) {
      return "INVALID";
    }
    return std::to_string(id);
  };

  return absl::StrFormat("space_id: %s entity_id: %s",
                         id_to_string(space_id).c_str(),
                         id_to_string(entity_id).c_str());
}

bool EntityHandle::IsValid() const {
  return space_id != kInvalidSpaceId && entity_id != kInvalidEntityId;
}

bool EntityHandle::operator==(const EntityHandle& other) const {
  return entity_id == other.entity_id && space_id == other.space_id;
}

bool EntityHandle::operator!=(const EntityHandle& other) const {
  return !(*this == other);
}

bool EntityHandle::operator<(const EntityHandle& other) const {
  if (entity_id != other.entity_id) {
    return entity_id < other.entity_id;
  }
  return space_id < other.space_id;
}

}  // namespace spatialblend