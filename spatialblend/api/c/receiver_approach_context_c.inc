// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Private fragment in the existing C/session owner.  This is registration
// ingress only: callers supply complete bounded values which are deep-copied
// into the Space-owned approach context.

uint32_t sb_get_receiver_approach_context_feature(
    sb_receiver_approach_context_feature_t* out, uint32_t out_size) {
  const uint32_t required = sizeof(sb_receiver_approach_context_feature_t);
  if (!out || out_size < required) return required;
  *out = {required,
          SB_RECEIVER_APPROACH_CONTEXT_FEATURE_VERSION,
          sizeof(sb_receiver_approach_context_registration_t),
          sizeof(sb_receiver_approach_polygon_t),
          sizeof(sb_receiver_approach_destination_t),
          SB_RECEIVER_APPROACH_CONTEXT_MAX_POLYGONS,
          SB_RECEIVER_APPROACH_CONTEXT_MAX_VERTICES_PER_POLYGON,
          SB_RECEIVER_APPROACH_CONTEXT_MAX_TOTAL_VERTICES,
          static_cast<uint32_t>(spatialblend::ReceiverInputState::kMaxNameBytes),
          SB_RECEIVER_APPROACH_CONTEXT_COORDINATE_CONVENTION,
          SB_RECEIVER_APPROACH_CONTEXT_AFFINE_CONVENTION,
          SB_RECEIVER_INPUT_FEATURE_VERSION,
          1};
  return required;
}

static const std::vector<uint32_t>& ReceiverApproachContextOffsets(uint32_t kind) {
#define RAC_F(f) offsetof(sb_receiver_approach_context_feature_t, f)
#define RAC_R(f) offsetof(sb_receiver_approach_context_registration_t, f)
#define RAC_P(f) offsetof(sb_receiver_approach_polygon_t, f)
#define RAC_D(f) offsetof(sb_receiver_approach_destination_t, f)
  static const std::vector<uint32_t> empty;
  static const std::vector<uint32_t> feature = {
      RAC_F(descriptor_size), RAC_F(feature_version), RAC_F(registration_size),
      RAC_F(polygon_size), RAC_F(destination_size), RAC_F(max_polygons),
      RAC_F(max_vertices_per_polygon), RAC_F(max_total_vertices),
      RAC_F(max_root_name_bytes), RAC_F(coordinate_convention),
      RAC_F(affine_convention), RAC_F(required_receiver_input_version),
      RAC_F(native_clock_domain)};
  static const std::vector<uint32_t> registration = {
      RAC_R(struct_size), RAC_R(feature_version),
      RAC_R(expected_registration_generation),
      RAC_R(expected_receiver_input_epoch),
      RAC_R(expected_receiver_input_sequence), RAC_R(geometry_revision),
      RAC_R(requested_valid_until_ns), RAC_R(navigation_root_utf8),
      RAC_R(navigation_root_bytes), RAC_R(coordinate_convention),
      RAC_R(navigation_root_pose_row_major), RAC_R(navigable_regions),
      RAC_R(navigable_region_count), RAC_R(obstacle_regions),
      RAC_R(obstacle_region_count), RAC_R(footprint_radius_m),
      RAC_R(required_clearance_m), RAC_R(destination)};
  static const std::vector<uint32_t> polygon = {
      RAC_P(struct_size), RAC_P(vertex_count), RAC_P(nav_element_id),
      RAC_P(region_id), RAC_P(coordinate_convention), RAC_P(affine_convention),
      RAC_P(local_to_space_row_major), RAC_P(vertices_xy)};
  static const std::vector<uint32_t> destination = {
      RAC_D(struct_size), RAC_D(space), RAC_D(region_id), RAC_D(object),
      RAC_D(affordance_id), RAC_D(support_revision), RAC_D(local_support),
      RAC_D(coordinate_convention), RAC_D(reserved),
      RAC_D(safe_approach_pose_row_major)};
#undef RAC_F
#undef RAC_R
#undef RAC_P
#undef RAC_D
  switch (kind) {
    case 1: return feature;
    case 2: return registration;
    case 3: return polygon;
    case 4: return destination;
    default: return empty;
  }
}

uint32_t sb_get_receiver_approach_context_field_count(uint32_t kind) {
  return ReceiverApproachContextOffsets(kind).size();
}

uint32_t sb_get_receiver_approach_context_field_offset(uint32_t kind,
                                                        uint32_t index) {
  const auto& fields = ReceiverApproachContextOffsets(kind);
  return index < fields.size() ? fields[index] : UINT32_MAX;
}

static int32_t ReceiverApproachContextResult(
    spatialblend::ReceiverApproachContextStatus status) {
  using R = spatialblend::ReceiverApproachContextStatus;
  switch (status) {
    case R::kAvailable: return SB_RECEIVER_APPROACH_CONTEXT_OK;
    case R::kStale: return SB_RECEIVER_APPROACH_CONTEXT_STALE;
    case R::kExpired: return SB_RECEIVER_APPROACH_CONTEXT_EXPIRED;
    case R::kReplaced: return SB_RECEIVER_APPROACH_CONTEXT_REPLACED;
    case R::kBusy: return SB_RECEIVER_APPROACH_CONTEXT_BUSY;
    case R::kMissingNavigationRoot:
      return SB_RECEIVER_APPROACH_CONTEXT_MISSING_NAVIGATION_ROOT;
    case R::kBodyOnlyInput: return SB_RECEIVER_APPROACH_CONTEXT_BODY_ONLY_INPUT;
    case R::kMissingGeometry: return SB_RECEIVER_APPROACH_CONTEXT_MISSING_GEOMETRY;
    case R::kInvalidGeometry: return SB_RECEIVER_APPROACH_CONTEXT_INVALID_GEOMETRY;
    case R::kMissingPolicy: return SB_RECEIVER_APPROACH_CONTEXT_MISSING_POLICY;
    case R::kMissingDestination:
      return SB_RECEIVER_APPROACH_CONTEXT_MISSING_DESTINATION;
  }
  return SB_RECEIVER_APPROACH_CONTEXT_UNAVAILABLE;
}

static bool CopyReceiverApproachPolygon(
    const sb_receiver_approach_polygon_t& source,
    spatialblend::ReceiverApproachPolygon* out, uint32_t* total_vertices) {
  if (!out || !total_vertices || source.struct_size != sizeof(source) ||
      source.coordinate_convention != SB_RECEIVER_APPROACH_CONTEXT_COORDINATE_CONVENTION ||
      source.affine_convention != SB_RECEIVER_APPROACH_CONTEXT_AFFINE_CONVENTION ||
      source.vertex_count < 3 ||
      source.vertex_count > SB_RECEIVER_APPROACH_CONTEXT_MAX_VERTICES_PER_POLYGON ||
      !source.vertices_xy ||
      source.vertex_count > SB_RECEIVER_APPROACH_CONTEXT_MAX_TOTAL_VERTICES -
                                *total_vertices)
    return false;
  spatialblend::ReceiverApproachPolygon value;
  value.nav_element_id = source.nav_element_id;
  value.region_id = source.region_id;
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      value.local_to_space(row, column) =
          source.local_to_space_row_major[row * 3 + column];
  value.vertices.reserve(source.vertex_count);
  for (uint32_t index = 0; index < source.vertex_count; ++index)
    value.vertices.emplace_back(source.vertices_xy[index * 2],
                                source.vertices_xy[index * 2 + 1]);
  *total_vertices += source.vertex_count;
  *out = std::move(value);
  return true;
}

int32_t sb_register_receiver_approach_context(
    sb_session_t session, sb_space_t space, sb_actor_t receiver_actor,
    uint32_t feature_version,
    const sb_receiver_approach_context_registration_t* registration,
    uint32_t registration_size, uint64_t* out_context_revision) {
  if (out_context_revision) *out_context_revision = 0;
  if (!registration || !out_context_revision ||
      feature_version != SB_RECEIVER_APPROACH_CONTEXT_FEATURE_VERSION ||
      registration_size != sizeof(*registration) ||
      registration->struct_size != sizeof(*registration))
    return feature_version != SB_RECEIVER_APPROACH_CONTEXT_FEATURE_VERSION
               ? SB_RECEIVER_APPROACH_CONTEXT_UNSUPPORTED_VERSION
               : SB_RECEIVER_APPROACH_CONTEXT_INVALID_ARGUMENT;
  if (registration->feature_version != feature_version ||
      registration->coordinate_convention !=
          SB_RECEIVER_APPROACH_CONTEXT_COORDINATE_CONVENTION ||
      !registration->navigation_root_utf8 || !registration->navigation_root_bytes ||
      registration->navigation_root_bytes >
          spatialblend::ReceiverInputState::kMaxNameBytes ||
      !registration->navigable_regions || !registration->obstacle_regions ||
      registration->navigable_region_count == 0 ||
      registration->obstacle_region_count == 0 ||
      registration->navigable_region_count >
          SB_RECEIVER_APPROACH_CONTEXT_MAX_POLYGONS ||
      registration->obstacle_region_count > SB_RECEIVER_APPROACH_CONTEXT_MAX_POLYGONS ||
      registration->destination.struct_size != sizeof(registration->destination) ||
      registration->destination.coordinate_convention !=
          SB_RECEIVER_APPROACH_CONTEXT_COORDINATE_CONVENTION ||
      registration->destination.reserved != 0 ||
      registration->destination.space != space)
    return SB_RECEIVER_APPROACH_CONTEXT_INVALID_ARGUMENT;
  std::string root_name;
  if (!ReceiverInputName(registration->navigation_root_utf8,
                         registration->navigation_root_bytes, &root_name))
    return SB_RECEIVER_APPROACH_CONTEXT_INVALID_ARGUMENT;

  spatialblend::ReceiverApproachContextRegistration value;
  value.session_id = static_cast<uint64_t>(session);
  value.receiver_actor = receiver_actor;
  value.expected_registration_generation =
      registration->expected_registration_generation;
  value.expected_receiver_input_epoch = registration->expected_receiver_input_epoch;
  value.expected_receiver_input_sequence =
      registration->expected_receiver_input_sequence;
  value.native_clock_domain = 1;
  value.geometry_revision = registration->geometry_revision;
  value.requested_valid_until_ns = registration->requested_valid_until_ns;
  value.navigation_root.anchor_name = std::move(root_name);
  for (int row = 0; row < 4; ++row)
    for (int column = 0; column < 4; ++column)
      value.navigation_root.pose(row, column) =
          registration->navigation_root_pose_row_major[row * 4 + column];
  uint32_t total_vertices = 0;
  value.navigable_regions.reserve(registration->navigable_region_count);
  for (uint32_t index = 0; index < registration->navigable_region_count; ++index) {
    spatialblend::ReceiverApproachPolygon polygon;
    if (!CopyReceiverApproachPolygon(registration->navigable_regions[index],
                                     &polygon, &total_vertices))
      return SB_RECEIVER_APPROACH_CONTEXT_INVALID_ARGUMENT;
    value.navigable_regions.push_back(std::move(polygon));
  }
  value.obstacle_regions.reserve(registration->obstacle_region_count);
  for (uint32_t index = 0; index < registration->obstacle_region_count; ++index) {
    spatialblend::ReceiverApproachPolygon polygon;
    if (!CopyReceiverApproachPolygon(registration->obstacle_regions[index],
                                     &polygon, &total_vertices))
      return SB_RECEIVER_APPROACH_CONTEXT_INVALID_ARGUMENT;
    value.obstacle_regions.push_back(std::move(polygon));
  }
  value.clearance.footprint_radius_m = registration->footprint_radius_m;
  value.clearance.required_clearance_m = registration->required_clearance_m;
  value.destination.region_id = registration->destination.region_id;
  value.destination.object_id = registration->destination.object;
  value.destination.affordance_id = registration->destination.affordance_id;
  value.destination.support_revision = registration->destination.support_revision;
  if (!CopyReceiverApproachPolygon(registration->destination.local_support,
                                   &value.destination.local_support,
                                   &total_vertices))
    return SB_RECEIVER_APPROACH_CONTEXT_INVALID_ARGUMENT;
  for (int row = 0; row < 4; ++row)
    for (int column = 0; column < 4; ++column)
      value.destination.safe_approach_pose(row, column) =
          registration->destination.safe_approach_pose_row_major[row * 4 + column];

  if (g_inside_relation_callback) return SB_RECEIVER_APPROACH_CONTEXT_BUSY;
  std::unique_lock<std::mutex> lock(g_sessions_mutex, std::try_to_lock);
  if (!lock.owns_lock()) return SB_RECEIVER_APPROACH_CONTEXT_BUSY;
  const auto owner = g_sessions.find(session);
  if (owner == g_sessions.end() || owner->second->rich_retiring)
    return SB_RECEIVER_APPROACH_CONTEXT_UNAVAILABLE;
  const auto found = owner->second->spaces.find(space);
  if (found == owner->second->spaces.end() || !found->second->space)
    return SB_RECEIVER_APPROACH_CONTEXT_UNAVAILABLE;
  value.space_id = found->second->space->GetSpaceId();
  value.destination.space_id = value.space_id;
  return ReceiverApproachContextResult(
      found->second->space->RegisterReceiverApproachContext(value,
                                                             out_context_revision));
}
