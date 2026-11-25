// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_HARNESS_GEOMETRY_COMPONENT_PROBE_H_
#define SPATIALBLEND_HARNESS_GEOMETRY_COMPONENT_PROBE_H_

#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"

namespace spatialblend::harness {

// Strict, dataset-independent harness for hierarchy, source body-anchor
// affordance proximity/lifecycle (no observed planar root required), and
// whole-baked-surface grounding. Schema/input failures return a non-OK
// status; an admitted native grounding failure is represented in its result.
absl::StatusOr<nlohmann::json> RunGeometryComponentProbe(
    const nlohmann::json& input);

}  // namespace spatialblend::harness

#endif  // SPATIALBLEND_HARNESS_GEOMETRY_COMPONENT_PROBE_H_
