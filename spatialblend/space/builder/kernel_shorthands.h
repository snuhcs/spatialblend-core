// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SIMULATOR_KERNEL_SHORTHANDS_H_
#define SPATIALBLEND_SIMULATOR_KERNEL_SHORTHANDS_H_

#include "spatialblend/model/relation/relation_kernel.h"

namespace spatialblend {

inline RelationKernelType Spacing() {
  return RelationKernelType::kAvatarSpacing;
}

inline RelationKernelType Facing() {
  return RelationKernelType::kAvatarFacing;
}

inline RelationKernelType AffordancePose() {
  return RelationKernelType::kAvatarAffordancePose;
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_SIMULATOR_KERNEL_SHORTHANDS_H_
