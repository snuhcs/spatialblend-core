// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Private implementation fragment; one existing C/session owner, no new registry.

// Absolute layout of the feature descriptor: any reordering or padding change
// breaks every binary that compiled against the header.
static_assert(sizeof(sb_containment_transfer_feature_t) == 16, "feature size");
static_assert(offsetof(sb_containment_transfer_feature_t, descriptor_size) == 0, "");
static_assert(offsetof(sb_containment_transfer_feature_t, feature_version) == 4, "");
static_assert(offsetof(sb_containment_transfer_feature_t, coordinate_convention) == 8, "");
static_assert(offsetof(sb_containment_transfer_feature_t, reserved) == 12, "");

uint32_t sb_get_containment_transfer_feature(
    sb_containment_transfer_feature_t* out, uint32_t size) {
  const uint32_t required = sizeof(sb_containment_transfer_feature_t);
  if (!out || size < required) return required;
  *out = {required, SB_CONTAINMENT_TRANSFER_FEATURE_VERSION, 1u, 0u};
  return required;
}

static const std::vector<uint32_t>& ContainmentTransferOffsets(uint32_t kind) {
  static const std::vector<uint32_t> empty;
  static const std::vector<uint32_t> feature = {
      offsetof(sb_containment_transfer_feature_t, descriptor_size),
      offsetof(sb_containment_transfer_feature_t, feature_version),
      offsetof(sb_containment_transfer_feature_t, coordinate_convention),
      offsetof(sb_containment_transfer_feature_t, reserved)};
  return kind == 1 ? feature : empty;
}

uint32_t sb_get_containment_transfer_field_count(uint32_t kind) {
  return ContainmentTransferOffsets(kind).size();
}

uint32_t sb_get_containment_transfer_field_offset(uint32_t kind,
                                                  uint32_t index) {
  const auto& offsets = ContainmentTransferOffsets(kind);
  return index < offsets.size() ? offsets[index] : UINT32_MAX;
}

int32_t sb_set_element_local_polygon(sb_space_t space, sb_element_t elem,
                                     sb_geometry_role_t role,
                                     const double* points_xy,
                                     int64_t num_points) {
  auto* info = GetSpace(space);
  if (g_inside_relation_callback || !info || !info->space || num_points < 3 ||
      num_points > kMaxSceneVertices || !ValidSceneRole(role) ||
      role == SB_GEOMETRY_NAVIGABLE || role == SB_GEOMETRY_SURFACE ||
      !FiniteSceneBuffer(points_xy, num_points * 2))
    return 0;
  std::vector<Eigen::Vector2d> pts;
  pts.reserve(num_points);
  for (int64_t i = 0; i < num_points; ++i)
    pts.emplace_back(points_xy[i * 2], points_xy[i * 2 + 1]);
  auto definition = std::make_shared<spatialblend::NavigableSpatialDefinition>(
      std::make_shared<spatialblend::Polygon2D>(pts), nullptr, 0.0,
      ToGeometryRole(role));
  definition->SetElementLocal(true);
  return info->space->SetElementSpatialDefinition(elem, definition) ? 1 : 0;
}

int32_t sb_set_element_containment_transfer(sb_space_t space, sb_element_t elem,
                                            int32_t enabled) {
  auto* info = GetSpace(space);
  if (g_inside_relation_callback || !info || !info->space) return 0;
  if (!enabled) return info->space->SetElementContainmentTransfer(elem, false) ? 1 : 0;
  spatialblend::Space::ElementState state;
  if (!info->space->CopyElementState(elem, &state) || !state.definition ||
      !spatialblend::IsContainmentRole(state.definition->GetGeometryRole()) ||
      (!dynamic_cast<spatialblend::NavigableSpatialDefinition*>(
           state.definition.get()) &&
       !dynamic_cast<spatialblend::PolytopeSpatialDefinition*>(
           state.definition.get())))
    return 0;
  return info->space->SetElementContainmentTransfer(elem, true) ? 1 : 0;
}

int32_t sb_get_source_transfer(sb_space_t space, sb_actor_t actor,
                               int64_t* leaf_shared_id) {
  auto* info = GetSpace(space);
  if (g_inside_relation_callback || !info || !info->space || !leaf_shared_id)
    return -1;
  spatialblend::shared_element_id_t leaf = spatialblend::kInvalidSharedElementId;
  if (!info->space->CopySourceTransferLeaf(static_cast<spatialblend::entity_id_t>(actor), &leaf))
    return -1;
  if (leaf == spatialblend::kInvalidSharedElementId) return 0;
  *leaf_shared_id = static_cast<int64_t>(leaf);
  return 1;
}
