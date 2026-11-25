// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Private implementation fragment included by spatialblend_c.cc after the
// registry and rich-freeze implementation. Not a separately linked API owner.

uint32_t sb_get_receiver_input_feature(sb_receiver_input_feature_t* feature,
                                       uint32_t feature_size) {
  const uint32_t required = sizeof(sb_receiver_input_feature_t);
  if (!feature || feature_size < required) return required;
  sb_receiver_input_feature_t value{};
  value.descriptor_size = required;
  value.feature_version = SB_RECEIVER_INPUT_FEATURE_VERSION;
  value.anchor_size = sizeof(sb_receiver_anchor_input_t);
  value.frame_size = sizeof(sb_receiver_input_frame_t);
  value.stamp_size = sizeof(sb_receiver_input_stamp_t);
  value.validity_size = sizeof(sb_receiver_input_validity_t);
  value.max_anchors = spatialblend::ReceiverInputState::kMaxAnchors;
  value.max_name_bytes = spatialblend::ReceiverInputState::kMaxNameBytes;
  value.coordinate_convention = 1;
  value.max_age_ns = spatialblend::ReceiverInputState::kMaxAgeNs;
  value.monotonic_clock_id = SB_RICH_REALIZATION_CLOCK_ID;
  value.monotonic_clock_epoch = RichClockEpoch();
  *feature = value;
  return required;
}

static const std::vector<uint32_t>& ReceiverInputOffsets(uint32_t kind) {
  static const std::vector<uint32_t> empty;
  static const std::vector<uint32_t> feature = {
      offsetof(sb_receiver_input_feature_t, descriptor_size),
      offsetof(sb_receiver_input_feature_t, feature_version),
      offsetof(sb_receiver_input_feature_t, anchor_size),
      offsetof(sb_receiver_input_feature_t, frame_size),
      offsetof(sb_receiver_input_feature_t, stamp_size),
      offsetof(sb_receiver_input_feature_t, validity_size),
      offsetof(sb_receiver_input_feature_t, max_anchors),
      offsetof(sb_receiver_input_feature_t, max_name_bytes),
      offsetof(sb_receiver_input_feature_t, coordinate_convention),
      offsetof(sb_receiver_input_feature_t, reserved),
      offsetof(sb_receiver_input_feature_t, max_age_ns),
      offsetof(sb_receiver_input_feature_t, monotonic_clock_id),
      offsetof(sb_receiver_input_feature_t, monotonic_clock_epoch),
  };
  static const std::vector<uint32_t> anchor = {
      offsetof(sb_receiver_anchor_input_t, name_utf8),
      offsetof(sb_receiver_anchor_input_t, name_bytes),
      offsetof(sb_receiver_anchor_input_t, available),
      offsetof(sb_receiver_anchor_input_t, world_pose_row_major),
  };
  static const std::vector<uint32_t> frame = {
      offsetof(sb_receiver_input_frame_t, struct_size),
      offsetof(sb_receiver_input_frame_t, feature_version),
      offsetof(sb_receiver_input_frame_t, anchor_count),
      offsetof(sb_receiver_input_frame_t, anchor_stride),
      offsetof(sb_receiver_input_frame_t, epoch),
      offsetof(sb_receiver_input_frame_t, sequence),
      offsetof(sb_receiver_input_frame_t, monotonic_clock_id),
      offsetof(sb_receiver_input_frame_t, monotonic_clock_epoch),
      offsetof(sb_receiver_input_frame_t, observed_at_ns),
      offsetof(sb_receiver_input_frame_t, valid_until_ns),
      offsetof(sb_receiver_input_frame_t, anchors),
  };
  static const std::vector<uint32_t> stamp = {
      offsetof(sb_receiver_input_stamp_t, epoch),
      offsetof(sb_receiver_input_stamp_t, revision),
      offsetof(sb_receiver_input_stamp_t, sequence),
      offsetof(sb_receiver_input_stamp_t, rig_revision),
      offsetof(sb_receiver_input_stamp_t, calibration_revision),
      offsetof(sb_receiver_input_stamp_t, observed_at_ns),
      offsetof(sb_receiver_input_stamp_t, valid_until_ns),
      offsetof(sb_receiver_input_stamp_t, known),
      offsetof(sb_receiver_input_stamp_t, reserved),
  };
  static const std::vector<uint32_t> validity = {
      offsetof(sb_receiver_input_validity_t, producer_lifetime),
      offsetof(sb_receiver_input_validity_t, revision),
      offsetof(sb_receiver_input_validity_t, captured),
      offsetof(sb_receiver_input_validity_t, current),
      offsetof(sb_receiver_input_validity_t, changed),
      offsetof(sb_receiver_input_validity_t, valid),
  };
  switch (kind) {
    case SB_RECEIVER_INPUT_LAYOUT_FEATURE:
      return feature;
    case SB_RECEIVER_INPUT_LAYOUT_ANCHOR:
      return anchor;
    case SB_RECEIVER_INPUT_LAYOUT_FRAME:
      return frame;
    case SB_RECEIVER_INPUT_LAYOUT_STAMP:
      return stamp;
    case SB_RECEIVER_INPUT_LAYOUT_VALIDITY:
      return validity;
    default:
      return empty;
  }
}

uint32_t sb_get_receiver_input_field_count(uint32_t kind) {
  return ReceiverInputOffsets(kind).size();
}

uint32_t sb_get_receiver_input_field_offset(uint32_t kind, uint32_t index) {
  const auto& fields = ReceiverInputOffsets(kind);
  return index < fields.size() ? fields[index] : UINT32_MAX;
}

// Bounded well-formed UTF-8, excluding embedded NUL and overlong/surrogate
// encodings. Pointer extents remain caller-owned, as in the other C APIs.
static bool ReceiverInputName(const uint8_t* bytes, uint32_t size,
                              std::string* out) {
  if (!bytes || !out || size == 0 ||
      size > spatialblend::ReceiverInputState::kMaxNameBytes)
    return false;
  for (uint32_t i = 0; i < size;) {
    const uint8_t first = bytes[i++];
    if (first == 0) return false;
    if (first < 0x80) continue;
    const int count = first >= 0xc2 && first <= 0xdf   ? 1
                      : first >= 0xe0 && first <= 0xef ? 2
                      : first >= 0xf0 && first <= 0xf4 ? 3
                                                       : -1;
    if (count < 0 || size - i < static_cast<uint32_t>(count)) return false;
    uint32_t point = first & (0x7f >> (count + 1));
    for (int c = 0; c < count; ++c) {
      const uint8_t next = bytes[i++];
      if ((next & 0xc0) != 0x80) return false;
      point = (point << 6) | (next & 0x3f);
    }
    if ((count == 1 && point < 0x80) || (count == 2 && point < 0x800) ||
        (count == 3 && point < 0x10000) || point > 0x10ffff ||
        (point >= 0xd800 && point <= 0xdfff))
      return false;
  }
  out->assign(reinterpret_cast<const char*>(bytes), size);
  return true;
}

int32_t sb_begin_receiver_input(sb_space_t space, sb_actor_t actor,
                                uint32_t feature_version,
                                const uint8_t* root_name_utf8,
                                uint32_t root_name_bytes, uint64_t rig_revision,
                                uint64_t calibration_revision,
                                uint64_t* epoch) {
  if (g_inside_relation_callback) return SB_RICH_RESULT_UNAVAILABLE;
  if (feature_version != SB_RECEIVER_INPUT_FEATURE_VERSION)
    return SB_RICH_RESULT_UNSUPPORTED_VERSION;
  std::string root;
  if (!epoch || !ReceiverInputName(root_name_utf8, root_name_bytes, &root))
    return SB_RICH_RESULT_INVALID_ARGUMENT;
  auto* info = GetSpace(space);
  if (!info || !info->space) return SB_RICH_RESULT_INVALID_ARGUMENT;
  return info->space->BeginReceiverInputEpoch(actor, root, rig_revision,
                                              calibration_revision, epoch)
             ? SB_RICH_RESULT_OK
             : SB_RICH_RESULT_INVALID_ARGUMENT;
}

int32_t sb_submit_receiver_input(sb_space_t space, sb_actor_t actor,
                                 const sb_receiver_input_frame_t* frame,
                                 uint32_t frame_size) {
  if (g_inside_relation_callback) return SB_RICH_RESULT_UNAVAILABLE;
  if (!frame || frame_size != sizeof(*frame) ||
      frame->struct_size != sizeof(*frame))
    return SB_RICH_RESULT_INVALID_ARGUMENT;
  if (frame->feature_version != SB_RECEIVER_INPUT_FEATURE_VERSION)
    return SB_RICH_RESULT_UNSUPPORTED_VERSION;
  if (!frame->anchors || frame->anchor_count == 0 ||
      frame->anchor_count > spatialblend::ReceiverInputState::kMaxAnchors ||
      frame->anchor_stride != sizeof(sb_receiver_anchor_input_t) ||
      frame->monotonic_clock_id != SB_RICH_REALIZATION_CLOCK_ID ||
      frame->monotonic_clock_epoch != RichClockEpoch())
    return SB_RICH_RESULT_INVALID_ARGUMENT;
  auto* info = GetSpace(space);
  if (!info || !info->space) return SB_RICH_RESULT_INVALID_ARGUMENT;
  std::vector<spatialblend::ReceiverAnchorInput> anchors;
  anchors.reserve(frame->anchor_count);
  for (uint32_t i = 0; i < frame->anchor_count; ++i) {
    const auto& row = frame->anchors[i];
    spatialblend::ReceiverAnchorInput value;
    if (row.available > 1 ||
        !ReceiverInputName(row.name_utf8, row.name_bytes, &value.name))
      return SB_RICH_RESULT_INVALID_ARGUMENT;
    value.available = row.available == 1;
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        value.world_pose(r, c) = row.world_pose_row_major[r * 4 + c];
    anchors.push_back(std::move(value));
  }
  return info->space->ApplyReceiverInputFrame(
             actor, frame->epoch, frame->sequence, frame->observed_at_ns,
             frame->valid_until_ns, anchors)
             ? SB_RICH_RESULT_OK
             : SB_RICH_RESULT_INVALID_ARGUMENT;
}

int32_t sb_clear_receiver_input(sb_space_t space, sb_actor_t actor,
                                uint64_t epoch) {
  if (g_inside_relation_callback) return SB_RICH_RESULT_UNAVAILABLE;
  auto* info = GetSpace(space);
  if (!info || !info->space) return SB_RICH_RESULT_INVALID_ARGUMENT;
  return info->space->ClearReceiverInputFrame(actor, epoch)
             ? SB_RICH_RESULT_OK
             : SB_RICH_RESULT_INVALID_ARGUMENT;
}

static sb_receiver_input_stamp_t ExportReceiverInputStamp(
    const spatialblend::ReceiverInputStamp& input) {
  sb_receiver_input_stamp_t result{};
  result.epoch = input.epoch;
  result.revision = input.revision;
  result.sequence = input.sequence;
  result.rig_revision = input.rig_revision;
  result.calibration_revision = input.calibration_revision;
  result.observed_at_ns = input.observed_at_ns;
  result.valid_until_ns = input.valid_until_ns;
  result.known = input.known ? 1u : 0u;
  return result;
}

static bool SameReceiverInputStamp(const spatialblend::ReceiverInputStamp& a,
                                   const spatialblend::ReceiverInputStamp& b) {
  return a.epoch == b.epoch && a.revision == b.revision &&
         a.sequence == b.sequence && a.rig_revision == b.rig_revision &&
         a.calibration_revision == b.calibration_revision &&
         a.observed_at_ns == b.observed_at_ns &&
         a.valid_until_ns == b.valid_until_ns && a.known == b.known;
}

int32_t sb_get_receiver_input_validity(sb_rich_realization_token_t token,
                                       uint64_t rich_revision,
                                       sb_receiver_input_validity_t* out,
                                       uint32_t out_size) {
  if (g_inside_relation_callback) return SB_RICH_RESULT_UNAVAILABLE;
  if (!out || out_size < sizeof(*out)) return SB_RICH_RESULT_INVALID_ARGUMENT;
  *out = {};
  RichActivation activation;
  std::shared_ptr<const FrozenRichSnapshot> snapshot;
  {
    std::lock_guard<std::mutex> lock(g_rich_mutex);
    const auto it = g_rich_activations.find(token);
    if (it == g_rich_activations.end()) return SB_RICH_RESULT_UNKNOWN_TOKEN;
    if (!it->second.active) return SB_RICH_RESULT_UNAVAILABLE;
    snapshot = it->second.snapshot;
    if (!snapshot || snapshot->header.revision != rich_revision)
      return SB_RICH_RESULT_STALE_SNAPSHOT;
    activation = it->second;
  }
  auto* info = GetSpace(activation.space);
  if (!info || !info->space) return SB_RICH_RESULT_STALE_SNAPSHOT;
  spatialblend::ActorReceiverInputValidity core;
  if (!info->space->GetActorReceiverInputValidity(activation.actor,
                                                  rich_revision, &core) ||
      !SameReceiverInputStamp(snapshot->receiver_input, core.captured))
    return SB_RICH_RESULT_STALE_SNAPSHOT;
  {
    std::lock_guard<std::mutex> lock(g_rich_mutex);
    const auto it = g_rich_activations.find(token);
    if (it == g_rich_activations.end()) return SB_RICH_RESULT_UNKNOWN_TOKEN;
    if (!it->second.active || it->second.snapshot != snapshot)
      return SB_RICH_RESULT_STALE_SNAPSHOT;
  }
  out->producer_lifetime = token;
  out->revision = rich_revision;
  out->captured = ExportReceiverInputStamp(snapshot->receiver_input);
  out->current = ExportReceiverInputStamp(core.current);
  // Registry revalidation is not allowed to extend the input deadline.
  if (out->current.known &&
      spatialblend::ReceiverInputNowNs() >= out->current.valid_until_ns)
    out->current.known = 0;
  out->changed = core.changed ? 1u : 0u;
  out->valid = out->captured.known && out->current.known && !out->changed;
  return out->captured.known && out->current.known ? SB_RICH_RESULT_OK
                                                   : SB_RICH_RESULT_UNAVAILABLE;
}
