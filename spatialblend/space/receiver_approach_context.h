// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_RECEIVER_APPROACH_CONTEXT_H_
#define SPATIALBLEND_SPACE_RECEIVER_APPROACH_CONTEXT_H_

#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

#include "spatialblend/base/common/common.h"
#include "spatialblend/base/common/types.h"
#include "spatialblend/model/realization/receiver_input.h"

namespace spatialblend {

// This is deliberately a data-only contract.  Approach feasibility may consume an Available
// context, but this module neither selects a hierarchy nor computes a route.
enum class ReceiverApproachContextStatus {
  kAvailable,
  kMissingNavigationRoot,
  kBodyOnlyInput,
  kMissingGeometry,
  kInvalidGeometry,
  kMissingPolicy,
  kMissingDestination,
  kStale,
  kExpired,
  kReplaced,
  kBusy,
};

enum class ReceiverApproachBasis : uint32_t {
  // Native Space's declared Z-up metre frame.  v1 does not silently convert
  // from Unity, source-world, or another basis.
  kSpaceZUpMeters = 1,
};

enum class ReceiverApproachRootKind : uint32_t {
  kObservedIndependentNavigationRoot = 1,
  kPelvisOrBodyOnly = 2,
};

struct ReceiverApproachPolygon {
  uint64_t nav_element_id = 0;  // Stable native owner identity; never a tag.
  uint64_t region_id = 0;       // Exact receiver-local hierarchy region.
  ReceiverApproachBasis basis = ReceiverApproachBasis::kSpaceZUpMeters;
  // Explicit receiver-local -> declared Space-basis affine map.  It is copied
  // verbatim (rather than inferred from Element/Unity transforms).
  Eigen::Matrix3d local_to_space = Eigen::Matrix3d::Identity();
  std::vector<Eigen::Vector2d> vertices;
};

struct ReceiverApproachNavigationRoot {
  ReceiverApproachRootKind kind =
      ReceiverApproachRootKind::kObservedIndependentNavigationRoot;
  std::string anchor_name;
  ReceiverApproachBasis basis = ReceiverApproachBasis::kSpaceZUpMeters;
  // Must exactly match the named current ReceiverInput anchor pose.  The
  // field makes the provenance auditable; it is not a fallback pose source.
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
};

struct ReceiverApproachClearancePolicy {
  ReceiverApproachBasis basis = ReceiverApproachBasis::kSpaceZUpMeters;
  double footprint_radius_m = 0.0;
  double required_clearance_m = -1.0;
  // Maximum slope/step are intentionally absent in v1.  Do not infer them.
};

struct ReceiverApproachDestination {
  space_id_t space_id = kInvalidSpaceId;
  uint64_t region_id = 0;
  element_id_t object_id = kInvalidElementId;
  uint64_t affordance_id = 0;
  uint64_t support_revision = 0;
  ReceiverApproachPolygon local_support;
  ReceiverApproachBasis basis = ReceiverApproachBasis::kSpaceZUpMeters;
  Eigen::Matrix4d safe_approach_pose = Eigen::Matrix4d::Identity();
};

// Registration is constructed only by the native owner.  session_id is an
// explicit caller-owned identity, not a transport handle or source identity.
struct ReceiverApproachContextRegistration {
  space_id_t space_id = kInvalidSpaceId;
  uint64_t session_id = 0;
  spatialblend::entity_id_t receiver_actor = spatialblend::kInvalidEntityId;
  // Compare-and-replace fence.  Zero creates the first registration for an
  // actor; every replacement must cite the currently issued generation.
  uint64_t expected_registration_generation = 0;
  // Fence registration to exactly this current input frame, rather than
  // selecting whichever receiver frame happens to be current at write time.
  uint64_t expected_receiver_input_epoch = 0;
  uint64_t expected_receiver_input_sequence = 0;
  // v1 exposes only the native steady-clock domain.  This identity blocks a
  // source or transport timestamp from masquerading as receiver time.
  uint32_t native_clock_domain = 1;
  uint64_t geometry_revision = 0;
  int64_t requested_valid_until_ns = 0;
  ReceiverApproachNavigationRoot navigation_root;
  std::vector<ReceiverApproachPolygon> navigable_regions;
  std::vector<ReceiverApproachPolygon> obstacle_regions;
  ReceiverApproachClearancePolicy clearance;
  ReceiverApproachDestination destination;
};

// Complete immutable context returned by value.  All vectors are detached
// copies and reads never extend valid_until_ns.
struct ReceiverApproachContext {
  uint64_t context_revision = 0;
  uint64_t registration_generation = 0;
  space_id_t space_id = kInvalidSpaceId;
  uint64_t session_id = 0;
  spatialblend::entity_id_t receiver_actor = spatialblend::kInvalidEntityId;
  ReceiverInputStamp receiver_input;
  uint32_t native_clock_domain = 1;
  uint64_t geometry_revision = 0;
  uint64_t support_revision = 0;
  int64_t issued_at_ns = 0;
  int64_t valid_until_ns = 0;
  ReceiverApproachNavigationRoot navigation_root;
  std::vector<ReceiverApproachPolygon> navigable_regions;
  std::vector<ReceiverApproachPolygon> obstacle_regions;
  ReceiverApproachClearancePolicy clearance;
  ReceiverApproachDestination destination;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_RECEIVER_APPROACH_CONTEXT_H_
