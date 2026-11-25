// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_COMMON_STATE_TYPE_H_
#define SPATIALBLEND_COMMON_STATE_TYPE_H_

#include <cstddef>

namespace spatialblend {

// State components that relations can read/write.
// Supports both 2D and 3D configurations.
enum class StateType {
  kPosition2D,    // [x, y] — 2 dimensions
  kPosition3D,    // [x, y, z] — 3 dimensions
  kRotationYaw,   // [yaw] — 1 dimension (Z-axis rotation)
  kRotation3D,    // [yaw, pitch, roll] — 3 dimensions
};

inline size_t GetStateTypeDimension(StateType type) {
  switch (type) {
    case StateType::kPosition2D: return 2;
    case StateType::kPosition3D: return 3;
    case StateType::kRotationYaw: return 1;
    case StateType::kRotation3D: return 3;
  }
  return 0;
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_COMMON_STATE_TYPE_H_
