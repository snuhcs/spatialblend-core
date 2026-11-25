// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/space.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace spatialblend {
namespace {

constexpr size_t kMaxPolygons = 64;
constexpr size_t kMaxVertices = 256;
constexpr double kGeometryEpsilon = 1e-9;

bool RigidPose(const Eigen::Matrix4d& pose) {
  if (!pose.allFinite() ||
      (pose.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).norm() > kGeometryEpsilon)
    return false;
  const Eigen::Matrix3d rotation = pose.block<3, 3>(0, 0);
  return (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() <=
             1e-6 &&
         std::abs(rotation.determinant() - 1.0) <= 1e-6;
}

bool ValidAffine(const Eigen::Matrix3d& value) {
  return value.allFinite() &&
         (value.row(2) - Eigen::RowVector3d(0, 0, 1)).norm() <=
             kGeometryEpsilon &&
         std::abs(value.topLeftCorner<2, 2>().determinant()) > kGeometryEpsilon;
}

double Cross(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
             const Eigen::Vector2d& c) {
  return (b.x() - a.x()) * (c.y() - a.y()) -
         (b.y() - a.y()) * (c.x() - a.x());
}

bool OnSegment(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
               const Eigen::Vector2d& point) {
  return std::abs(Cross(a, b, point)) <= kGeometryEpsilon &&
         point.x() >= std::min(a.x(), b.x()) - kGeometryEpsilon &&
         point.x() <= std::max(a.x(), b.x()) + kGeometryEpsilon &&
         point.y() >= std::min(a.y(), b.y()) - kGeometryEpsilon &&
         point.y() <= std::max(a.y(), b.y()) + kGeometryEpsilon;
}

bool SegmentsIntersect(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                       const Eigen::Vector2d& c, const Eigen::Vector2d& d) {
  const double ab_c = Cross(a, b, c);
  const double ab_d = Cross(a, b, d);
  const double cd_a = Cross(c, d, a);
  const double cd_b = Cross(c, d, b);
  if (((ab_c > kGeometryEpsilon && ab_d < -kGeometryEpsilon) ||
       (ab_c < -kGeometryEpsilon && ab_d > kGeometryEpsilon)) &&
      ((cd_a > kGeometryEpsilon && cd_b < -kGeometryEpsilon) ||
       (cd_a < -kGeometryEpsilon && cd_b > kGeometryEpsilon)))
    return true;
  return OnSegment(a, b, c) || OnSegment(a, b, d) || OnSegment(c, d, a) ||
         OnSegment(c, d, b);
}

bool IsSimplePolygon(const ReceiverApproachPolygon& polygon) {
  const auto& vertices = polygon.vertices;
  if (polygon.nav_element_id == 0 || polygon.region_id == 0 ||
      polygon.basis != ReceiverApproachBasis::kSpaceZUpMeters ||
      vertices.size() < 3 || vertices.size() > kMaxVertices ||
      !ValidAffine(polygon.local_to_space))
    return false;
  double twice_area = 0;
  for (size_t i = 0; i < vertices.size(); ++i) {
    const auto& a = vertices[i];
    const auto& b = vertices[(i + 1) % vertices.size()];
    if (!a.allFinite() || !b.allFinite() || (a - b).norm() <= kGeometryEpsilon)
      return false;
    twice_area += a.x() * b.y() - a.y() * b.x();
  }
  if (std::abs(twice_area) <= kGeometryEpsilon) return false;
  for (size_t i = 0; i < vertices.size(); ++i) {
    const size_t i_next = (i + 1) % vertices.size();
    for (size_t j = i + 1; j < vertices.size(); ++j) {
      const size_t j_next = (j + 1) % vertices.size();
      if (i == j || i_next == j || j_next == i) continue;
      if (SegmentsIntersect(vertices[i], vertices[i_next], vertices[j],
                            vertices[j_next]))
        return false;
    }
  }
  return true;
}

Eigen::Vector2d InSpaceBasis(const ReceiverApproachPolygon& polygon,
                             const Eigen::Vector2d& point) {
  const Eigen::Vector3d mapped = polygon.local_to_space *
                                 Eigen::Vector3d(point.x(), point.y(), 1.0);
  return mapped.head<2>();
}

bool ContainsInSpaceBasis(const ReceiverApproachPolygon& polygon,
                          const Eigen::Vector2d& point) {
  bool inside = false;
  for (size_t i = 0, j = polygon.vertices.size() - 1; i < polygon.vertices.size();
       j = i++) {
    const Eigen::Vector2d a = InSpaceBasis(polygon, polygon.vertices[j]);
    const Eigen::Vector2d b = InSpaceBasis(polygon, polygon.vertices[i]);
    if (OnSegment(a, b, point)) return true;
    if ((a.y() > point.y()) != (b.y() > point.y()) &&
        point.x() < (b.x() - a.x()) * (point.y() - a.y()) /
                            (b.y() - a.y()) +
                        a.x())
      inside = !inside;
  }
  return inside;
}

bool SamePose(const Eigen::Matrix4d& a, const Eigen::Matrix4d& b) {
  return a.isApprox(b, 1e-9);
}

}  // namespace

ReceiverApproachContextStatus Space::RegisterReceiverApproachContext(
    const ReceiverApproachContextRegistration& registration,
    uint64_t* out_context_revision) {
  if (out_context_revision) *out_context_revision = 0;
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (registration.space_id == kInvalidSpaceId ||
      registration.space_id != space_id_ || registration.session_id == 0 ||
      registration.receiver_actor == spatialblend::kInvalidEntityId ||
      !FindActorLocked(registration.receiver_actor))
    return ReceiverApproachContextStatus::kStale;

  const auto prior = receiver_approach_contexts_.find(registration.receiver_actor);
  const uint64_t prior_generation =
      prior == receiver_approach_contexts_.end()
          ? 0
          : prior->second.registration_generation;
  if (registration.expected_registration_generation != prior_generation)
    return ReceiverApproachContextStatus::kReplaced;

  const auto input_it = receiver_inputs_.find(registration.receiver_actor);
  if (input_it == receiver_inputs_.end())
    return ReceiverApproachContextStatus::kBodyOnlyInput;
  const int64_t now = receiver_input_clock_();
  const ReceiverInputStamp input = input_it->second.Stamp(now);
  if (!input.known)
    return now >= input.valid_until_ns && input.valid_until_ns > 0
               ? ReceiverApproachContextStatus::kExpired
               : ReceiverApproachContextStatus::kStale;
  if (registration.native_clock_domain != 1 ||
      registration.expected_receiver_input_epoch != input.epoch ||
      registration.expected_receiver_input_sequence != input.sequence)
    return ReceiverApproachContextStatus::kStale;

  if (registration.navigation_root.kind ==
      ReceiverApproachRootKind::kPelvisOrBodyOnly)
    return ReceiverApproachContextStatus::kBodyOnlyInput;
  if (registration.navigation_root.kind !=
          ReceiverApproachRootKind::kObservedIndependentNavigationRoot ||
      registration.navigation_root.anchor_name.empty() ||
      registration.navigation_root.basis != ReceiverApproachBasis::kSpaceZUpMeters ||
      !RigidPose(registration.navigation_root.pose))
    return ReceiverApproachContextStatus::kMissingNavigationRoot;
  // A present body anchor is not a navigation root merely because its pose is
  // available.  The receiver-input epoch explicitly declares the one root.
  if (registration.navigation_root.anchor_name != input_it->second.root_anchor())
    return ReceiverApproachContextStatus::kMissingNavigationRoot;
  Eigen::Matrix4d observed_root;
  if (!input_it->second.CopyPose(registration.navigation_root.anchor_name,
                                 &observed_root) ||
      !SamePose(observed_root, registration.navigation_root.pose))
    return ReceiverApproachContextStatus::kMissingNavigationRoot;

  if (registration.navigable_regions.empty() || registration.obstacle_regions.empty())
    return ReceiverApproachContextStatus::kMissingGeometry;
  if (registration.navigable_regions.size() > kMaxPolygons ||
      registration.obstacle_regions.size() > kMaxPolygons ||
      registration.geometry_revision == 0)
    return ReceiverApproachContextStatus::kInvalidGeometry;
  std::unordered_set<uint64_t> nav_ids;
  for (const auto& polygon : registration.navigable_regions) {
    if (!IsSimplePolygon(polygon) ||
        polygon.region_id != registration.destination.region_id ||
        !nav_ids.insert(polygon.nav_element_id).second)
      return ReceiverApproachContextStatus::kInvalidGeometry;
  }
  for (const auto& polygon : registration.obstacle_regions) {
    if (!IsSimplePolygon(polygon) ||
        polygon.region_id != registration.destination.region_id ||
        !nav_ids.insert(polygon.nav_element_id).second)
      return ReceiverApproachContextStatus::kInvalidGeometry;
  }
  if (registration.clearance.basis != ReceiverApproachBasis::kSpaceZUpMeters ||
      !std::isfinite(registration.clearance.footprint_radius_m) ||
      !std::isfinite(registration.clearance.required_clearance_m) ||
      registration.clearance.footprint_radius_m <= 0 ||
      registration.clearance.required_clearance_m < 0)
    return ReceiverApproachContextStatus::kMissingPolicy;

  const auto& destination = registration.destination;
  if (destination.space_id != space_id_ || destination.region_id == 0 ||
      destination.object_id == kInvalidElementId || destination.affordance_id == 0 ||
      destination.support_revision == 0 ||
      destination.basis != ReceiverApproachBasis::kSpaceZUpMeters ||
      !IsSimplePolygon(destination.local_support) ||
      destination.local_support.region_id != destination.region_id ||
      !RigidPose(destination.safe_approach_pose))
    return ReceiverApproachContextStatus::kMissingDestination;
  const Eigen::Vector2d approach =
      destination.safe_approach_pose.block<2, 1>(0, 3);
  bool approach_in_navigation = false;
  for (const auto& polygon : registration.navigable_regions)
    approach_in_navigation |= ContainsInSpaceBasis(polygon, approach);
  if (!approach_in_navigation)
    return ReceiverApproachContextStatus::kMissingDestination;
  if (registration.requested_valid_until_ns <= now)
    return ReceiverApproachContextStatus::kExpired;

  if (next_receiver_approach_context_revision_ ==
      std::numeric_limits<uint64_t>::max())
    return ReceiverApproachContextStatus::kStale;
  ReceiverApproachContext value;
  value.context_revision = next_receiver_approach_context_revision_++;
  value.registration_generation = prior_generation + 1;
  if (value.registration_generation == 0) return ReceiverApproachContextStatus::kStale;
  value.space_id = space_id_;
  value.session_id = registration.session_id;
  value.receiver_actor = registration.receiver_actor;
  value.receiver_input = input;
  value.native_clock_domain = registration.native_clock_domain;
  value.geometry_revision = registration.geometry_revision;
  value.support_revision = destination.support_revision;
  value.issued_at_ns = now;
  value.valid_until_ns = std::min(registration.requested_valid_until_ns,
                                  input.valid_until_ns);
  value.navigation_root = registration.navigation_root;
  value.navigable_regions = registration.navigable_regions;
  value.obstacle_regions = registration.obstacle_regions;
  value.clearance = registration.clearance;
  value.destination = destination;
  receiver_approach_contexts_[registration.receiver_actor] = std::move(value);
  if (out_context_revision)
    *out_context_revision =
        receiver_approach_contexts_.at(registration.receiver_actor).context_revision;
  return ReceiverApproachContextStatus::kAvailable;
}

ReceiverApproachContextStatus Space::ValidateReceiverApproachContextLocked(
    const ReceiverApproachContext& context) const {
  if (context.space_id != space_id_ || context.receiver_actor ==
                                       spatialblend::kInvalidEntityId ||
      !FindActorLocked(context.receiver_actor))
    return ReceiverApproachContextStatus::kStale;
  const auto input_it = receiver_inputs_.find(context.receiver_actor);
  if (input_it == receiver_inputs_.end()) return ReceiverApproachContextStatus::kStale;
  const int64_t now = receiver_input_clock_();
  if (now >= context.valid_until_ns) return ReceiverApproachContextStatus::kExpired;
  const ReceiverInputStamp current = input_it->second.Stamp(now);
  if (!current.known)
    return now >= current.valid_until_ns && current.valid_until_ns > 0
               ? ReceiverApproachContextStatus::kExpired
               : ReceiverApproachContextStatus::kStale;
  if (current.epoch != context.receiver_input.epoch ||
      current.sequence != context.receiver_input.sequence ||
      current.revision != context.receiver_input.revision)
    return ReceiverApproachContextStatus::kStale;
  return ReceiverApproachContextStatus::kAvailable;
}

ReceiverApproachContextStatus Space::TryCopyReceiverApproachContext(
    spatialblend::entity_id_t receiver_actor, uint64_t expected_context_revision,
    ReceiverApproachContext* out) const {
  if (!out || expected_context_revision == 0)
    return ReceiverApproachContextStatus::kStale;
  *out = {};
  std::unique_lock<std::mutex> lock(space_mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return ReceiverApproachContextStatus::kBusy;
  const auto it = receiver_approach_contexts_.find(receiver_actor);
  if (it == receiver_approach_contexts_.end())
    return ReceiverApproachContextStatus::kStale;
  if (it->second.context_revision != expected_context_revision)
    return ReceiverApproachContextStatus::kReplaced;
  const auto status = ValidateReceiverApproachContextLocked(it->second);
  if (status != ReceiverApproachContextStatus::kAvailable) return status;
  *out = it->second;
  return ReceiverApproachContextStatus::kAvailable;
}

}  // namespace spatialblend
