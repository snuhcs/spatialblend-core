// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Private implementation fragment; one existing C/session owner, no new registry.

uint32_t sb_get_planar_support_feature(sb_planar_support_feature_t* feature,
                                      uint32_t feature_size) {
  const uint32_t required = sizeof(sb_planar_support_feature_t);
  if (!feature || feature_size < required) return required;
  *feature = {required, SB_PLANAR_SUPPORT_FEATURE_VERSION,
              sizeof(sb_planar_support_header_t), SB_PLANAR_SUPPORT_MAX_POINTS,
              1, SB_PLANAR_SUPPORT_XY_AFFINE_CONVENTION};
  return required;
}

static const std::vector<uint32_t>& PlanarSupportOffsets(uint32_t kind) {
  static const std::vector<uint32_t> empty;
  static const std::vector<uint32_t> feature = {
      offsetof(sb_planar_support_feature_t, descriptor_size),
      offsetof(sb_planar_support_feature_t, feature_version),
      offsetof(sb_planar_support_feature_t, header_size),
      offsetof(sb_planar_support_feature_t, max_points),
      offsetof(sb_planar_support_feature_t, coordinate_convention),
      offsetof(sb_planar_support_feature_t, plane_convention)};
  static const std::vector<uint32_t> header = {
      offsetof(sb_planar_support_header_t, struct_size),
      offsetof(sb_planar_support_header_t, feature_version),
      offsetof(sb_planar_support_header_t, definition_kind),
      offsetof(sb_planar_support_header_t, geometry_role),
      offsetof(sb_planar_support_header_t, space),
      offsetof(sb_planar_support_header_t, element),
      offsetof(sb_planar_support_header_t, tree_revision),
      offsetof(sb_planar_support_header_t, support_revision),
      offsetof(sb_planar_support_header_t, point_count),
      offsetof(sb_planar_support_header_t, plane_convention),
      offsetof(sb_planar_support_header_t, plane_to_world_row_major)};
  return kind == 1 ? feature : kind == 2 ? header : empty;
}

uint32_t sb_get_planar_support_field_count(uint32_t kind) {
  return PlanarSupportOffsets(kind).size();
}

uint32_t sb_get_planar_support_field_offset(uint32_t kind, uint32_t index) {
  const auto& offsets = PlanarSupportOffsets(kind);
  return index < offsets.size() ? offsets[index] : UINT32_MAX;
}

int32_t sb_copy_planar_support(
    sb_session_t session, sb_space_t space, sb_element_t element,
    uint32_t feature_version, uint64_t expected_tree_revision,
    uint64_t expected_support_revision, double* boundary_xy,
    uint32_t point_capacity, sb_planar_support_header_t* header,
    uint32_t header_size) {
  if (!header || header_size < sizeof(*header) ||
      feature_version != SB_PLANAR_SUPPORT_FEATURE_VERSION ||
      expected_tree_revision == 0 || expected_support_revision == 0 ||
      point_capacity > SB_PLANAR_SUPPORT_MAX_POINTS ||
      (point_capacity != 0 && !boundary_xy))
    return SB_PLANAR_SUPPORT_INVALID_ARGUMENT;
  *header = {};
  if (g_inside_relation_callback) return SB_PLANAR_SUPPORT_BUSY;
  // Hold actual session ownership through the nonblocking Space copy. Never
  // obtain a borrowed GetSpace pointer that can outlive session teardown.
  std::unique_lock<std::mutex> lock(g_sessions_mutex, std::try_to_lock);
  if (!lock.owns_lock()) return SB_PLANAR_SUPPORT_BUSY;
  const auto session_it = g_sessions.find(session);
  if (session_it == g_sessions.end() || session_it->second->rich_retiring)
    return SB_PLANAR_SUPPORT_UNAVAILABLE;
  const auto space_it = session_it->second->spaces.find(space);
  if (space_it == session_it->second->spaces.end() || !space_it->second->space)
    return SB_PLANAR_SUPPORT_UNAVAILABLE;
  spatialblend::Space::PlanarSupportState value;
  const auto result = space_it->second->space->TryCopyPlanarSupport(
      element, expected_tree_revision, expected_support_revision, &value);
  using Result = spatialblend::Space::PlanarSupportReadResult;
  if (result == Result::kBusy) return SB_PLANAR_SUPPORT_BUSY;
  if (result == Result::kStale) return SB_PLANAR_SUPPORT_STALE;
  if (result != Result::kOk) return SB_PLANAR_SUPPORT_UNAVAILABLE;
  sb_planar_support_header_t copied{};
  copied.struct_size = sizeof(copied);
  copied.feature_version = SB_PLANAR_SUPPORT_FEATURE_VERSION;
  copied.definition_kind = value.definition_kind;
  copied.geometry_role = static_cast<uint32_t>(value.role);
  copied.space = space;
  copied.element = element;
  copied.tree_revision = value.tree_revision;
  copied.support_revision = value.support_revision;
  copied.point_count = static_cast<uint32_t>(value.boundary.size());
  copied.plane_convention = SB_PLANAR_SUPPORT_XY_AFFINE_CONVENTION;
  for (int row = 0; row < 4; ++row)
    for (int col = 0; col < 4; ++col)
      copied.plane_to_world_row_major[row * 4 + col] = value.plane_to_world(row, col);
  *header = copied;
  if (point_capacity < copied.point_count)
    return SB_PLANAR_SUPPORT_BUFFER_TOO_SMALL;
  for (uint32_t i = 0; i < copied.point_count; ++i) {
    boundary_xy[i * 2] = value.boundary[i].x();
    boundary_xy[i * 2 + 1] = value.boundary[i].y();
  }
  return SB_PLANAR_SUPPORT_OK;
}
