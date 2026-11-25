// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_COMMON_TYPES_H_
#define SPATIALBLEND_COMMON_TYPES_H_

#include <cstdint>

namespace spatialblend {

typedef int64_t element_id_t;
constexpr element_id_t kInvalidElementId = -1;

typedef int64_t shared_element_id_t;
constexpr shared_element_id_t kInvalidSharedElementId = -1;

// Space ID — identifies a space instance.
typedef int64_t space_id_t;
constexpr space_id_t kInvalidSpaceId = -1;

// Handle that uniquely identifies an element within a specific space.
struct element_handle_t {
  space_id_t space_id = kInvalidSpaceId;
  element_id_t element_id = kInvalidElementId;
  bool operator<(const element_handle_t& o) const {
    return space_id != o.space_id ? space_id < o.space_id
                                  : element_id < o.element_id;
  }
  bool operator==(const element_handle_t& o) const {
    return space_id == o.space_id && element_id == o.element_id;
  }
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_COMMON_TYPES_H_
