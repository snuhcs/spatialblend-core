// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

uint32_t sb_get_receiver_approach_feasibility_feature(
    sb_receiver_approach_feasibility_feature_t* out, uint32_t out_size) {
  const uint32_t size = sizeof(*out);
  if (!out || out_size < size) return size;
  *out = {size, SB_RECEIVER_APPROACH_FEASIBILITY_FEATURE_VERSION,
          sizeof(sb_receiver_approach_feasibility_header_t),
          SB_RECEIVER_APPROACH_FEASIBILITY_MAX_ROUTE_POINTS, 1,
          static_cast<uint32_t>(sizeof(double) * 2), 1,
          SB_RECEIVER_APPROACH_CONTEXT_FEATURE_VERSION};
  return size;
}

static const std::vector<uint32_t>& ReceiverApproachFeasibilityOffsets(uint32_t kind) {
#define RAF_F(f) offsetof(sb_receiver_approach_feasibility_feature_t, f)
#define RAF_H(f) offsetof(sb_receiver_approach_feasibility_header_t, f)
  static const std::vector<uint32_t> empty;
  static const std::vector<uint32_t> feature = {
      RAF_F(descriptor_size), RAF_F(feature_version), RAF_F(header_size),
      RAF_F(max_route_points), RAF_F(coordinate_convention),
      RAF_F(route_point_stride), RAF_F(native_clock_domain),
      RAF_F(required_context_version)};
  static const std::vector<uint32_t> header = {
      RAF_H(struct_size), RAF_H(feature_version), RAF_H(feasibility_status),
      RAF_H(coordinate_convention), RAF_H(session), RAF_H(space),
      RAF_H(receiver_actor), RAF_H(registration_session_id),
      RAF_H(registration_generation), RAF_H(receiver_input_epoch),
      RAF_H(receiver_input_sequence), RAF_H(receiver_input_revision),
      RAF_H(geometry_revision), RAF_H(support_revision), RAF_H(context_revision),
      RAF_H(calculation_revision), RAF_H(owner_lifetime),
      RAF_H(native_clock_domain), RAF_H(issued_at_ns), RAF_H(valid_until_ns),
      RAF_H(route_point_count), RAF_H(route_point_stride), RAF_H(start_x),
      RAF_H(start_y), RAF_H(safe_approach_x), RAF_H(safe_approach_y),
      RAF_H(route_length_m)};
#undef RAF_F
#undef RAF_H
  return kind == 1 ? feature : kind == 2 ? header : empty;
}
uint32_t sb_get_receiver_approach_feasibility_field_count(uint32_t kind) {
  return ReceiverApproachFeasibilityOffsets(kind).size();
}
uint32_t sb_get_receiver_approach_feasibility_field_offset(uint32_t kind,
                                                            uint32_t index) {
  const auto& fields = ReceiverApproachFeasibilityOffsets(kind);
  return index < fields.size() ? fields[index] : UINT32_MAX;
}

static int32_t ReceiverApproachFeasibilityResult(
    spatialblend::ReceiverApproachFeasibilityStatus status) {
  using R = spatialblend::ReceiverApproachFeasibilityStatus;
  switch (status) {
    case R::kStale: return SB_RECEIVER_APPROACH_FEASIBILITY_STALE;
    case R::kExpired: return SB_RECEIVER_APPROACH_FEASIBILITY_EXPIRED;
    case R::kReplaced: return SB_RECEIVER_APPROACH_FEASIBILITY_REPLACED;
    case R::kBusy: return SB_RECEIVER_APPROACH_FEASIBILITY_BUSY;
    default: return SB_RECEIVER_APPROACH_FEASIBILITY_UNAVAILABLE;
  }
}

int32_t sb_copy_receiver_approach_feasibility(
    sb_session_t session, sb_space_t space, sb_actor_t receiver_actor,
    uint32_t feature_version, uint64_t context_revision,
    uint64_t expected_calculation_revision, double* route_xy,
    uint32_t route_point_capacity,
    sb_receiver_approach_feasibility_header_t* header, uint32_t header_size) {
  if (!header || header_size < sizeof(*header) || !context_revision ||
      route_point_capacity > SB_RECEIVER_APPROACH_FEASIBILITY_MAX_ROUTE_POINTS ||
      (route_point_capacity && !route_xy))
    return SB_RECEIVER_APPROACH_FEASIBILITY_INVALID_ARGUMENT;
  *header = {};
  if (feature_version != SB_RECEIVER_APPROACH_FEASIBILITY_FEATURE_VERSION)
    return SB_RECEIVER_APPROACH_FEASIBILITY_UNSUPPORTED_VERSION;
  if (g_inside_relation_callback) return SB_RECEIVER_APPROACH_FEASIBILITY_BUSY;
  spatialblend::ReceiverApproachFeasibilitySnapshot value;
  {
    std::unique_lock<std::mutex> lock(g_sessions_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return SB_RECEIVER_APPROACH_FEASIBILITY_BUSY;
    const auto owner = g_sessions.find(session);
    if (owner == g_sessions.end() || owner->second->rich_retiring)
      return SB_RECEIVER_APPROACH_FEASIBILITY_UNAVAILABLE;
    const auto found = owner->second->spaces.find(space);
    if (found == owner->second->spaces.end() || !found->second->space)
      return SB_RECEIVER_APPROACH_FEASIBILITY_UNAVAILABLE;
    const auto status = expected_calculation_revision == 0
        ? found->second->space->TryComputeReceiverApproachFeasibility(
              receiver_actor, context_revision, &value)
        : found->second->space->TryCopyReceiverApproachFeasibilitySnapshot(
              receiver_actor, expected_calculation_revision, &value);
    if (value.calculation_revision == 0)
      return ReceiverApproachFeasibilityResult(status);
    if (expected_calculation_revision != 0 &&
        value.calculation_revision != expected_calculation_revision)
      return SB_RECEIVER_APPROACH_FEASIBILITY_RETRY_CHANGED;
  }
  if (value.route.size() > SB_RECEIVER_APPROACH_FEASIBILITY_MAX_ROUTE_POINTS)
    return SB_RECEIVER_APPROACH_FEASIBILITY_UNAVAILABLE;
  sb_receiver_approach_feasibility_header_t copied{};
  copied.struct_size = sizeof(copied);
  copied.feature_version = SB_RECEIVER_APPROACH_FEASIBILITY_FEATURE_VERSION;
  copied.feasibility_status = static_cast<int32_t>(value.status);
  copied.coordinate_convention = 1;
  copied.session = session;
  copied.space = space;
  copied.receiver_actor = receiver_actor;
  copied.registration_session_id = value.context.session_id;
  copied.registration_generation = value.context.registration_generation;
  copied.receiver_input_epoch = value.context.receiver_input.epoch;
  copied.receiver_input_sequence = value.context.receiver_input.sequence;
  copied.receiver_input_revision = value.context.receiver_input.revision;
  copied.geometry_revision = value.context.geometry_revision;
  copied.support_revision = value.context.support_revision;
  copied.context_revision = value.context.context_revision;
  copied.calculation_revision = value.calculation_revision;
  copied.owner_lifetime = value.owner_lifetime;
  copied.native_clock_domain = value.context.native_clock_domain;
  copied.issued_at_ns = value.context.issued_at_ns;
  copied.valid_until_ns = value.context.valid_until_ns;
  copied.route_point_count = value.route.size();
  copied.route_point_stride = sizeof(double) * 2;
  copied.start_x = value.start.x(); copied.start_y = value.start.y();
  copied.safe_approach_x = value.safe_approach.x();
  copied.safe_approach_y = value.safe_approach.y();
  copied.route_length_m = value.route_length_m;
  *header = copied;
  if (route_point_capacity < copied.route_point_count)
    return SB_RECEIVER_APPROACH_FEASIBILITY_BUFFER_TOO_SMALL;
  for (uint32_t i = 0; i < copied.route_point_count; ++i) {
    route_xy[i * 2] = value.route[i].x();
    route_xy[i * 2 + 1] = value.route[i].y();
  }
  return SB_RECEIVER_APPROACH_FEASIBILITY_OK;
}
