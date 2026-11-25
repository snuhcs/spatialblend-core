// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Private fragment in the existing C owner, after receiver_input_c.inc.
// Reuses its bounded UTF-8 validator and the process-local monotonic clock.

uint32_t sb_get_source_evidence_feature(sb_source_evidence_feature_t* out,
                                        uint32_t size) {
  using E = spatialblend::SourceInteractionEvidence;
  if (!out || size < sizeof(*out)) return sizeof(*out);
  *out = {sizeof(*out),
          SB_SOURCE_EVIDENCE_FEATURE_VERSION,
          sizeof(sb_source_evidence_header_t),
          sizeof(sb_source_evidence_anchor_t),
          sizeof(sb_source_evidence_relation_t),
          sizeof(sb_source_evidence_node_t),
          sizeof(sb_source_evidence_copy_t),
          E::kMaxAnchors,
          E::kMaxRelations,
          E::kMaxPathDepth,
          E::kMaxPathNodes,
          E::kMaxBoundaryPoints,
          E::kMaxNameBytes,
          SB_SOURCE_EVIDENCE_MAX_UTF8,
          1,
          E::kMaximumAgeNs,
          SB_RICH_REALIZATION_CLOCK_ID,
          RichClockEpoch()};
  return sizeof(*out);
}

static const std::vector<uint32_t>& SourceEvidenceOffsets(uint32_t kind) {
#define SE_F(f) offsetof(sb_source_evidence_feature_t, f)
#define SE_H(f) offsetof(sb_source_evidence_header_t, f)
#define SE_A(f) offsetof(sb_source_evidence_anchor_t, f)
#define SE_R(f) offsetof(sb_source_evidence_relation_t, f)
#define SE_N(f) offsetof(sb_source_evidence_node_t, f)
#define SE_C(f) offsetof(sb_source_evidence_copy_t, f)
  static const std::vector<uint32_t> empty;
  static const std::vector<uint32_t> feature = {SE_F(descriptor_size),
                                                SE_F(feature_version),
                                                SE_F(header_size),
                                                SE_F(anchor_size),
                                                SE_F(relation_size),
                                                SE_F(node_size),
                                                SE_F(copy_size),
                                                SE_F(max_anchors),
                                                SE_F(max_relations),
                                                SE_F(max_path_depth),
                                                SE_F(max_nodes),
                                                SE_F(max_points),
                                                SE_F(max_name_bytes),
                                                SE_F(max_utf8_bytes),
                                                SE_F(coordinate_convention),
                                                SE_F(max_age_ns),
                                                SE_F(monotonic_clock_id),
                                                SE_F(monotonic_clock_epoch)};
  static const std::vector<uint32_t> header = {SE_H(struct_size),
                                               SE_H(feature_version),
                                               SE_H(space),
                                               SE_H(actor),
                                               SE_H(actor_element),
                                               SE_H(actor_shared),
                                               SE_H(epoch),
                                               SE_H(sequence),
                                               SE_H(source_time_seconds),
                                               SE_H(receipt_monotonic_ns),
                                               SE_H(monotonic_clock_id),
                                               SE_H(monotonic_clock_epoch),
                                               SE_H(tree_revision),
                                               SE_H(support_revision)};
  static const std::vector<uint32_t> anchor = {
      SE_A(name_offset),     SE_A(name_length),        SE_A(observed),
      SE_A(locomotion_root), SE_A(position_dimension), SE_A(position_x),
      SE_A(position_y),      SE_A(position_z),         SE_A(yaw)};
  static const std::vector<uint32_t> relation = {
      SE_R(binding_ordinal),
      SE_R(anchor_ordinal),
      SE_R(kernel),
      SE_R(explicit_scope),
      SE_R(filter_offset),
      SE_R(filter_length),
      SE_R(proximity),
      SE_R(normalized_weight),
      SE_R(phase),
      SE_R(phase_identity_unambiguous),
      SE_R(phase_referent_id),
      SE_R(path_offset),
      SE_R(path_count),
      SE_R(has_projection),
      SE_R(bounded_planar_support),
      SE_R(projected_x),
      SE_R(projected_y),
      SE_R(projected_z),
      SE_R(anchor_support_distance),
      SE_R(support_to_world_row_major),
      SE_R(point_offset),
      SE_R(point_count),
      SE_R(sub_region_offset),
      SE_R(sub_region_length),
      SE_R(support_angle),
      SE_R(support_radial_ratio)};
  static const std::vector<uint32_t> node = {SE_N(element),
                                             SE_N(parent),
                                             SE_N(shared),
                                             SE_N(tag_offset),
                                             SE_N(tag_length),
                                             SE_N(geometry_role),
                                             SE_N(world_transform_row_major)};
  static const std::vector<uint32_t> copy = {SE_C(struct_size),
                                             SE_C(anchor_capacity),
                                             SE_C(relation_capacity),
                                             SE_C(node_capacity),
                                             SE_C(point_capacity),
                                             SE_C(utf8_capacity),
                                             SE_C(anchors),
                                             SE_C(relations),
                                             SE_C(nodes),
                                             SE_C(boundary_xy),
                                             SE_C(utf8),
                                             SE_C(header),
                                             SE_C(anchor_count),
                                             SE_C(relation_count),
                                             SE_C(node_count),
                                             SE_C(point_count),
                                             SE_C(utf8_bytes)};
#undef SE_F
#undef SE_H
#undef SE_A
#undef SE_R
#undef SE_N
#undef SE_C
  switch (kind) {
    case 1:
      return feature;
    case 2:
      return header;
    case 3:
      return anchor;
    case 4:
      return relation;
    case 5:
      return node;
    case 6:
      return copy;
    default:
      return empty;
  }
}

uint32_t sb_get_source_evidence_field_count(uint32_t kind) {
  return SourceEvidenceOffsets(kind).size();
}

uint32_t sb_get_source_evidence_field_offset(uint32_t kind, uint32_t index) {
  const auto& offsets = SourceEvidenceOffsets(kind);
  return index < offsets.size() ? offsets[index] : UINT32_MAX;
}

static int32_t SourceEvidenceResult(
    spatialblend::Space::SourceEvidenceReadResult result) {
  using R = spatialblend::Space::SourceEvidenceReadResult;
  switch (result) {
    case R::kOk:
      return SB_SOURCE_EVIDENCE_OK;
    case R::kBusy:
      return SB_SOURCE_EVIDENCE_BUSY;
    case R::kStale:
      return SB_SOURCE_EVIDENCE_STALE;
    default:
      return SB_SOURCE_EVIDENCE_UNAVAILABLE;
  }
}

// Only called while g_sessions_mutex is held; no borrowed global-space lookup.
static spatialblend::Space* SourceEvidenceSpace(sb_session_t session,
                                                sb_space_t space) {
  const auto owner = g_sessions.find(session);
  if (owner == g_sessions.end() || owner->second->rich_retiring) return nullptr;
  const auto found = owner->second->spaces.find(space);
  return found == owner->second->spaces.end() ? nullptr : found->second->space;
}

int32_t sb_try_enable_source_evidence(sb_session_t session, sb_space_t space,
                                      sb_actor_t actor, uint32_t version,
                                      uint64_t epoch, uint32_t enabled) {
  if (version != SB_SOURCE_EVIDENCE_FEATURE_VERSION || !epoch || enabled > 1)
    return SB_SOURCE_EVIDENCE_INVALID_ARGUMENT;
  if (g_inside_relation_callback) return SB_SOURCE_EVIDENCE_BUSY;
  std::unique_lock<std::mutex> lock(g_sessions_mutex, std::try_to_lock);
  if (!lock.owns_lock()) return SB_SOURCE_EVIDENCE_BUSY;
  auto* source = SourceEvidenceSpace(session, space);
  return source
             ? SourceEvidenceResult(source->TryEnableSourceInteractionEvidence(
                   actor, epoch, enabled != 0))
             : SB_SOURCE_EVIDENCE_UNAVAILABLE;
}

static bool AppendSourceEvidenceName(const std::string& value,
                                     std::vector<uint8_t>* bytes,
                                     uint32_t* offset, uint32_t* length) {
  if (value.size() >= spatialblend::SourceInteractionEvidence::kMaxNameBytes ||
      value.size() > SB_SOURCE_EVIDENCE_MAX_UTF8 - bytes->size())
    return false;
  std::string validated;
  if (!value.empty() &&
      !ReceiverInputName(reinterpret_cast<const uint8_t*>(value.data()),
                         value.size(), &validated))
    return false;
  *offset = bytes->size();
  *length = value.size();
  bytes->insert(bytes->end(), value.begin(), value.end());
  return true;
}

static void SourceEvidenceMatrix(const Eigen::Matrix4d& value, double* out) {
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) out[r * 4 + c] = value(r, c);
}

// C++ RelationKernelType intentionally has a different declaration order from
// the stable C ABI enum.  Source evidence is consumed by ordinal, so this
// boundary must translate by meaning rather than static_cast.
static int32_t SourceEvidenceKernel(spatialblend::RelationKernelType kernel) {
  using K = spatialblend::RelationKernelType;
  switch (kernel) {
    case K::kAvatarSpacing:
      return SB_KERNEL_SPACING;
    case K::kAvatarFacing:
      return SB_KERNEL_FACING;
    case K::kAvatarAffordancePose:
      return SB_KERNEL_AFFORDANCE_POSE;
  }
  return -1;
}

int32_t sb_copy_source_evidence(sb_session_t session, sb_space_t space,
                                sb_actor_t actor, uint32_t version,
                                uint64_t epoch, int64_t sequence,
                                int64_t maximum_age_ns,
                                sb_source_evidence_copy_t* copy,
                                uint32_t copy_size) {
  using E = spatialblend::SourceInteractionEvidence;
  if (!copy || copy_size < sizeof(*copy))
    return SB_SOURCE_EVIDENCE_INVALID_ARGUMENT;
  copy->header = {};
  copy->anchor_count = copy->relation_count = copy->node_count =
      copy->point_count = copy->utf8_bytes = 0;
  if (copy->struct_size != sizeof(*copy) ||
      version != SB_SOURCE_EVIDENCE_FEATURE_VERSION || !epoch || sequence < 0 ||
      maximum_age_ns <= 0 || maximum_age_ns > E::kMaximumAgeNs ||
      copy->anchor_capacity > E::kMaxAnchors ||
      copy->relation_capacity > E::kMaxRelations ||
      copy->node_capacity > E::kMaxPathNodes ||
      copy->point_capacity > E::kMaxBoundaryPoints ||
      copy->utf8_capacity > SB_SOURCE_EVIDENCE_MAX_UTF8 ||
      (copy->anchor_capacity && !copy->anchors) ||
      (copy->relation_capacity && !copy->relations) ||
      (copy->node_capacity && !copy->nodes) ||
      (copy->point_capacity && !copy->boundary_xy) ||
      (copy->utf8_capacity && !copy->utf8))
    return SB_SOURCE_EVIDENCE_INVALID_ARGUMENT;
  if (g_inside_relation_callback) return SB_SOURCE_EVIDENCE_BUSY;
  E value;
  {
    std::unique_lock<std::mutex> lock(g_sessions_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return SB_SOURCE_EVIDENCE_BUSY;
    auto* source = SourceEvidenceSpace(session, space);
    if (!source) return SB_SOURCE_EVIDENCE_UNAVAILABLE;
    const int32_t result =
        SourceEvidenceResult(source->TryCopySourceInteractionEvidence(
            actor, epoch, sequence, maximum_age_ns, &value));
    if (result != SB_SOURCE_EVIDENCE_OK) return result;
  }
  // From here everything is detached from native ownership, even if teardown
  // starts during flattening. No borrowed support pointers or new scene reads.
  std::vector<sb_source_evidence_anchor_t> anchors;
  std::vector<sb_source_evidence_relation_t> relations;
  std::vector<sb_source_evidence_node_t> nodes;
  std::vector<double> xy;
  std::vector<uint8_t> utf8;
  for (const auto& a : value.anchors) {
    sb_source_evidence_anchor_t row{};
    if (!AppendSourceEvidenceName(a.name, &utf8, &row.name_offset,
                                  &row.name_length))
      return SB_SOURCE_EVIDENCE_UNAVAILABLE;
    row.observed = a.observed;
    row.locomotion_root = a.locomotion_root;
    row.position_dimension = a.position_dimension;
    row.position_x = a.position.x();
    row.position_y = a.position.y();
    row.position_z = a.position.z();
    row.yaw = a.yaw;
    anchors.push_back(row);
  }
  for (const auto& r : value.relations) {
    sb_source_evidence_relation_t row{};
    // One binding may legitimately emit zero, one, or many relation rows, so
    // a binding ordinal is not bounded by relation_count.  Its native anchor
    // association has already been checked when the immutable evidence value
    // was captured; only the copied anchor table bounds this flattening step.
    if (r.anchor_ordinal >= value.anchors.size())
      return SB_SOURCE_EVIDENCE_UNAVAILABLE;
    row.binding_ordinal = r.binding_ordinal;
    row.anchor_ordinal = r.anchor_ordinal;
    row.kernel = SourceEvidenceKernel(r.kernel);
    if (row.kernel < 0) return SB_SOURCE_EVIDENCE_UNAVAILABLE;
    row.explicit_scope = r.explicit_scope;
    if (!AppendSourceEvidenceName(r.tag_filter, &utf8, &row.filter_offset,
                                  &row.filter_length) ||
        !AppendSourceEvidenceName(r.sub_region, &utf8, &row.sub_region_offset,
                                  &row.sub_region_length))
      return SB_SOURCE_EVIDENCE_UNAVAILABLE;
    row.proximity = r.proximity;
    row.normalized_weight = r.normalized_weight;
    row.phase = static_cast<int32_t>(r.phase);
    row.phase_identity_unambiguous = r.phase_identity_unambiguous;
    row.phase_referent_id = r.phase_referent_id;
    row.path_offset = nodes.size();
    row.path_count = r.path.size();
    for (const auto& n : r.path) {
      sb_source_evidence_node_t node{};
      node.element = n.element;
      node.parent = n.parent;
      node.shared = n.shared;
      if (!AppendSourceEvidenceName(n.tag, &utf8, &node.tag_offset,
                                    &node.tag_length))
        return SB_SOURCE_EVIDENCE_UNAVAILABLE;
      node.geometry_role = static_cast<uint32_t>(n.geometry_role);
      SourceEvidenceMatrix(n.world_transform, node.world_transform_row_major);
      nodes.push_back(node);
    }
    row.has_projection = r.has_projection;
    row.bounded_planar_support = r.bounded_planar_support;
    row.projected_x = r.projected_point.x();
    row.projected_y = r.projected_point.y();
    row.projected_z = r.projected_point.z();
    row.anchor_support_distance = r.anchor_support_distance;
    SourceEvidenceMatrix(r.support_to_world, row.support_to_world_row_major);
    row.point_offset = xy.size() / 2;
    row.point_count = r.support_boundary.size();
    for (const auto& point : r.support_boundary) {
      xy.push_back(point.x());
      xy.push_back(point.y());
    }
    row.support_angle = r.support_angle;
    row.support_radial_ratio = r.support_radial_ratio;
    relations.push_back(row);
  }
  auto& h = copy->header;
  h.struct_size = sizeof(h);
  h.feature_version = SB_SOURCE_EVIDENCE_FEATURE_VERSION;
  h.space = value.space;
  h.actor = value.actor;
  h.actor_element = value.actor_element;
  h.actor_shared = value.actor_shared;
  h.epoch = value.epoch;
  h.sequence = value.sequence;
  h.source_time_seconds = value.source_time_seconds;
  h.receipt_monotonic_ns = value.receipt_monotonic_ns;
  h.monotonic_clock_id = SB_RICH_REALIZATION_CLOCK_ID;
  h.monotonic_clock_epoch = RichClockEpoch();
  h.tree_revision = value.tree_revision;
  h.support_revision = value.support_revision;
  copy->anchor_count = anchors.size();
  copy->relation_count = relations.size();
  copy->node_count = nodes.size();
  copy->point_count = xy.size() / 2;
  copy->utf8_bytes = utf8.size();
  if (copy->anchor_capacity < anchors.size() ||
      copy->relation_capacity < relations.size() ||
      copy->node_capacity < nodes.size() ||
      copy->point_capacity < xy.size() / 2 || copy->utf8_capacity < utf8.size())
    return SB_SOURCE_EVIDENCE_BUFFER_TOO_SMALL;
  // All validation and capacity checks precede any write to caller arrays.
  if (!anchors.empty())
    std::copy(anchors.begin(), anchors.end(), copy->anchors);
  if (!relations.empty())
    std::copy(relations.begin(), relations.end(), copy->relations);
  if (!nodes.empty()) std::copy(nodes.begin(), nodes.end(), copy->nodes);
  if (!xy.empty()) std::copy(xy.begin(), xy.end(), copy->boundary_xy);
  if (!utf8.empty()) std::copy(utf8.begin(), utf8.end(), copy->utf8);
  return SB_SOURCE_EVIDENCE_OK;
}
