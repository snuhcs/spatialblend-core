// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Included after registry/rich dispatch; no independent C API owner.
uint32_t sb_get_receiver_reevaluation_feature(
    sb_receiver_reevaluation_feature_t* feature, uint32_t feature_size) {
  const uint32_t required = sizeof(sb_receiver_reevaluation_feature_t);
  if (!feature || feature_size < required) return required;
  sb_receiver_reevaluation_feature_t value{};
  value.descriptor_size = required;
  value.feature_version = SB_RECEIVER_REEVALUATION_FEATURE_VERSION;
  value.required_receiver_input_version = SB_RECEIVER_INPUT_FEATURE_VERSION;
  value.required_rich_version = SB_RICH_REALIZATION_FEATURE_VERSION;
  value.guarantees = SB_RECEIVER_REEVALUATION_PRESERVES_SOURCE_CAUSE |
                     SB_RECEIVER_REEVALUATION_EXACT_INPUT_ONCE |
                     SB_RECEIVER_REEVALUATION_TARGET_ONLY;
  *feature = value;
  return required;
}

uint32_t sb_get_receiver_reevaluation_field_count(uint32_t kind) {
  return kind == SB_RECEIVER_REEVALUATION_LAYOUT_FEATURE ? 6 : 0;
}

uint32_t sb_get_receiver_reevaluation_field_offset(uint32_t kind,
                                                 uint32_t index) {
  static const uint32_t fields[] = {
      offsetof(sb_receiver_reevaluation_feature_t, descriptor_size),
      offsetof(sb_receiver_reevaluation_feature_t, feature_version),
      offsetof(sb_receiver_reevaluation_feature_t, required_receiver_input_version),
      offsetof(sb_receiver_reevaluation_feature_t, required_rich_version),
      offsetof(sb_receiver_reevaluation_feature_t, guarantees),
      offsetof(sb_receiver_reevaluation_feature_t, reserved),
  };
  return kind == SB_RECEIVER_REEVALUATION_LAYOUT_FEATURE && index < 6
             ? fields[index] : UINT32_MAX;
}

int32_t sb_reevaluate_receiver_realization(
    sb_rich_realization_token_t token, uint32_t feature_version,
    uint64_t expected_epoch, uint64_t expected_sequence) {
  if (g_inside_relation_callback) return SB_RICH_RESULT_UNAVAILABLE;
  if (feature_version != SB_RECEIVER_REEVALUATION_FEATURE_VERSION)
    return SB_RICH_RESULT_UNSUPPORTED_VERSION;
  if (!expected_epoch || !expected_sequence)
    return SB_RICH_RESULT_INVALID_ARGUMENT;
  RichActivation activation;
  {
    std::lock_guard<std::mutex> lock(g_rich_mutex);
    const auto it = g_rich_activations.find(token);
    if (it == g_rich_activations.end()) return SB_RICH_RESULT_UNKNOWN_TOKEN;
    if (!it->second.active) return SB_RICH_RESULT_UNAVAILABLE;
    activation = it->second;
  }
  auto* info = GetSpace(activation.space);
  if (!info || !info->space) return SB_RICH_RESULT_UNAVAILABLE;
  const auto result = info->space->ReevaluateReceiverRealization(
      activation.actor, expected_epoch, expected_sequence);
  {
    std::lock_guard<std::mutex> lock(g_rich_mutex);
    const auto it = g_rich_activations.find(token);
    if (it == g_rich_activations.end() || !it->second.active ||
        it->second.space != activation.space || it->second.actor != activation.actor)
      return SB_RICH_RESULT_UNAVAILABLE;
  }
  using Result = spatialblend::Space::ReceiverReevaluationResult;
  switch (result) {
    case Result::kEvaluated:
      DispatchRichRealization(activation.space, info, activation.actor, token);
      return SB_RICH_RESULT_OK;
    case Result::kNoNewInput:
      return SB_RICH_RESULT_NO_NEW_SNAPSHOT;
    case Result::kStaleInput:
      return SB_RICH_RESULT_STALE_SNAPSHOT;
    default:
      return SB_RICH_RESULT_UNAVAILABLE;
  }
}
