// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_RELATION_EVENT_H_
#define SPATIALBLEND_SPACE_RELATION_EVENT_H_

#include <string>

#include "spatialblend/base/common/types.h"
#include "spatialblend/model/relation/relation_kernel.h"

namespace spatialblend {

// Lifecycle phase of a relation between an actor anchor and a referent.
// Ordered chain: dormant → active → driving → contact.
enum class RelationPhase { kDormant, kActive, kDriving, kContact };

// A transition of one (actor, anchor, referent, kernel) relation between
// phases. Emitted by RelationPhaseTracker and surfaced through Space + the
// C API so clients can hang contact/engagement timing off it.
struct RelationEvent {
  shared_element_id_t actor_shared_id = kInvalidSharedElementId;
  std::string anchor_name;
  shared_element_id_t referent_shared_id = kInvalidSharedElementId;
  RelationKernelType kernel_type = RelationKernelType::kAvatarFacing;
  RelationPhase from = RelationPhase::kDormant;
  RelationPhase to = RelationPhase::kDormant;
  double timestamp = 0.0;  // seconds, space-local clock
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_RELATION_EVENT_H_
