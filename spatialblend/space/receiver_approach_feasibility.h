// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_RECEIVER_APPROACH_FEASIBILITY_H_
#define SPATIALBLEND_SPACE_RECEIVER_APPROACH_FEASIBILITY_H_

#include <Eigen/Core>
#include <cstdint>
#include <vector>

#include "spatialblend/base/common/common.h"
#include "spatialblend/space/receiver_approach_context.h"

namespace spatialblend {

// Typed geometric result only.  This is neither controller authority nor a
// statement about full-body collision/contact feasibility.
enum class ReceiverApproachFeasibilityStatus {
  kFeasible,
  kNoContext,
  kStale,
  kExpired,
  kReplaced,
  kBusy,
  kStartOutsidePermittedRegion,
  kGoalOutsidePermittedRegion,
  kEndpointOnBoundary,
  kDisconnected,
  kBlocked,
  kInsufficientClearance,
  kInvalidGeometry,
};

struct ReceiverApproachFeasibilitySnapshot {
  ReceiverApproachFeasibilityStatus status =
      ReceiverApproachFeasibilityStatus::kNoContext;
  // Exact approach-context receipt copied only after a final current-validity fence.
  ReceiverApproachContext context;
  uint64_t calculation_revision = 0;
  uint64_t owner_lifetime = 0;
  Eigen::Vector2d start = Eigen::Vector2d::Zero();
  Eigen::Vector2d safe_approach = Eigen::Vector2d::Zero();
  std::vector<Eigen::Vector2d> route;
  double route_length_m = 0.0;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_RECEIVER_APPROACH_FEASIBILITY_H_
