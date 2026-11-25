// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Textually included at the end of spatialblend_c.cc (after the helpers it
// uses: GetSpace, ToKernelType, FromKernelType, ToNormalizedStrategy).
// Implements spatialblend_observability.h.

#include "spatialblend/api/c/spatialblend_observability.h"
#include "spatialblend/model/relation/avatar_affordance_pose_kernel.h"
#include "spatialblend/space/scope_record.h"

namespace {

using spatialblend::RealizationPolicyType;

// Every observability struct starts with {struct_size, version}.
template <typename T>
bool ObsSizeOk(uint32_t struct_size) {
  return struct_size == sizeof(T);
}

// The pre-ap_offset layouts of the structs that gained ap_offset_u_m /
// ap_offset_v_m / ap_height_m, which the header promises to keep accepting.
constexpr uint32_t kPreA4RelationStateSize =
    static_cast<uint32_t>(offsetof(sb_relation_state_v2_t, ap_offset_u_m));
constexpr uint32_t kPreA4RelationEvaluationSize =
    static_cast<uint32_t>(offsetof(sb_relation_evaluation_t, ap_offset_u_m));
constexpr uint32_t kPreA4TargetEvaluationSize = static_cast<uint32_t>(
    offsetof(sb_target_relation_evaluation_t, ap_offset_u_m));
// The ap_offset layout before the support_basis / r_formula_version fields.
constexpr uint32_t kPreA3TargetEvaluationSize = static_cast<uint32_t>(
    offsetof(sb_target_relation_evaluation_t, support_basis));

// Copy the first `struct_size` bytes of a fully built `full` into a caller
// buffer of a smaller (older) layout, leaving the caller's struct_size as
// passed. Fields at or beyond `struct_size` are never written.
template <typename T>
void StoreOld(T full, void* dst, uint32_t struct_size) {
  full.struct_size = struct_size;
  std::memcpy(dst, &full, struct_size);
}

template <typename T>
void ObsInit(T* out) {
  std::memset(out, 0, sizeof(T));
  out->struct_size = static_cast<uint32_t>(sizeof(T));
  out->version = SB_OBSERVABILITY_VERSION;
}

bool ValidKernel(sb_kernel_type_t k) {
  return k == SB_KERNEL_SPACING || k == SB_KERNEL_FACING ||
         k == SB_KERNEL_AFFORDANCE_POSE;
}

sb_realization_policy_t ToCPolicy(RealizationPolicyType type) {
  switch (type) {
    case RealizationPolicyType::kUniform:
      return SB_REALIZATION_POLICY_UNIFORM;
    case RealizationPolicyType::kGreedy:
      return SB_REALIZATION_POLICY_GREEDY;
    case RealizationPolicyType::kProgressive:
    default:
      return SB_REALIZATION_POLICY_PROGRESSIVE;
  }
}

int32_t ToCSolverStatus(spatialblend::SolverStatus s) {
  switch (s) {
    case spatialblend::SolverStatus::kConverged:
      return 0;
    case spatialblend::SolverStatus::kFallback:
    case spatialblend::SolverStatus::kBudgetHit:  // published (an incumbent), not converged
      return 1;
    case spatialblend::SolverStatus::kFailed:
    default:
      return 2;
  }
}

int32_t CopyEncoded(const Eigen::VectorXd& v, double* out) {
  const int32_t n = static_cast<int32_t>(v.size());
  for (int32_t i = 0; i < n && i < SB_MAX_ENCODED_STATE; ++i) out[i] = v(i);
  return n;
}

// Thread-local string arenas, one per function, so a pointer returned by one
// function is not invalidated by a call of a different one.
std::vector<std::string>& ObsStrings(int slot) {
  static thread_local std::vector<std::string> arenas[8];
  return arenas[slot];
}
const char* Keep(int slot, const std::string& value) {
  auto& arena = ObsStrings(slot);
  arena.push_back(value);
  return arena.back().c_str();
}
void ResetStrings(int slot, size_t reserve) {
  auto& arena = ObsStrings(slot);
  arena.clear();
  // Pointers must stay stable while the arena grows within this call.
  arena.reserve(reserve);
}

// Every observability getter: reject re-entry from a relation-event
// callback (the space lock is held there) and switch capture on.
spatialblend::Space* UnlockedCallerSpace(sb_space_t space) {
  if (g_inside_relation_callback) return nullptr;
  auto* info = GetSpace(space);
  return (info && info->space) ? info->space : nullptr;
}

spatialblend::Space* ObservedSpace(sb_space_t space) {
  if (g_inside_relation_callback) return nullptr;
  auto* info = GetSpace(space);
  if (!info || !info->space) return nullptr;
  info->space->EnableObservabilityCapture(true);
  return info->space;
}

spatialblend::element_id_t ElemOrZero(spatialblend::element_id_t id) {
  return id == spatialblend::kInvalidElementId ? 0 : id;
}

}  // namespace

uint32_t sb_get_observability_version(void) { return SB_OBSERVABILITY_VERSION; }

int32_t sb_set_observability_capture(sb_space_t space, int32_t enabled) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return 0;
  info->space->EnableObservabilityCapture(enabled != 0);
  return 1;
}

uint32_t sb_get_trace_format_version(void) {
  return static_cast<uint32_t>(spatialblend::kTraceFormatVersion);
}

// --------------------------------------------------------------------------
// Struct field offsets (declaration order).
// --------------------------------------------------------------------------

#define SB_OFF(T, f) static_cast<uint32_t>(offsetof(T, f))

static const std::vector<uint32_t>& FieldOffsets(sb_struct_id_t id) {
  static const std::vector<uint32_t> kEmpty;
  static const std::vector<uint32_t> actor_state = {
      SB_OFF(sb_actor_state_t, position_x), SB_OFF(sb_actor_state_t, position_y),
      SB_OFF(sb_actor_state_t, yaw)};
  static const std::vector<uint32_t> target = {
      SB_OFF(sb_realization_target_t, arrival_time),
      SB_OFF(sb_realization_target_t, anchor_name),
      SB_OFF(sb_realization_target_t, position_x),
      SB_OFF(sb_realization_target_t, position_y),
      SB_OFF(sb_realization_target_t, position_z),
      SB_OFF(sb_realization_target_t, yaw),
      SB_OFF(sb_realization_target_t, weight),
      SB_OFF(sb_realization_target_t, has_z),
      SB_OFF(sb_realization_target_t, kernel_type),
      SB_OFF(sb_realization_target_t, referent_id)};
  static const std::vector<uint32_t> relation_event = {
      SB_OFF(sb_relation_event_t, actor_shared_id),
      SB_OFF(sb_relation_event_t, anchor_name),
      SB_OFF(sb_relation_event_t, referent_shared_id),
      SB_OFF(sb_relation_event_t, kernel_type),
      SB_OFF(sb_relation_event_t, from_phase),
      SB_OFF(sb_relation_event_t, to_phase),
      SB_OFF(sb_relation_event_t, timestamp)};
  static const std::vector<uint32_t> relation_state = {
      SB_OFF(sb_relation_state_t, anchor_name),
      SB_OFF(sb_relation_state_t, proximity),
      SB_OFF(sb_relation_state_t, contact_point_world_x),
      SB_OFF(sb_relation_state_t, contact_point_world_y),
      SB_OFF(sb_relation_state_t, contact_point_world_z),
      SB_OFF(sb_relation_state_t, normalized_angle),
      SB_OFF(sb_relation_state_t, normalized_radial_ratio),
      SB_OFF(sb_relation_state_t, normalized_height_ratio),
      SB_OFF(sb_relation_state_t, referent_element_id),
      SB_OFF(sb_relation_state_t, referent_shared_id),
      SB_OFF(sb_relation_state_t, kernel_type),
      SB_OFF(sb_relation_state_t, normalized_strategy),
      SB_OFF(sb_relation_state_t, has_contact),
      SB_OFF(sb_relation_state_t, active)};
  static const std::vector<uint32_t> relation_state_v2 = {
      SB_OFF(sb_relation_state_v2_t, struct_size),
      SB_OFF(sb_relation_state_v2_t, version),
      SB_OFF(sb_relation_state_v2_t, anchor_name),
      SB_OFF(sb_relation_state_v2_t, sub_region_name),
      SB_OFF(sb_relation_state_v2_t, proximity),
      SB_OFF(sb_relation_state_v2_t, support_distance_m),
      SB_OFF(sb_relation_state_v2_t, source_scope_diameter_m),
      SB_OFF(sb_relation_state_v2_t, contact_point_world_x),
      SB_OFF(sb_relation_state_v2_t, contact_point_world_y),
      SB_OFF(sb_relation_state_v2_t, contact_point_world_z),
      SB_OFF(sb_relation_state_v2_t, normalized_angle),
      SB_OFF(sb_relation_state_v2_t, normalized_radial_ratio),
      SB_OFF(sb_relation_state_v2_t, normalized_height_ratio),
      SB_OFF(sb_relation_state_v2_t, interior_u),
      SB_OFF(sb_relation_state_v2_t, interior_v),
      SB_OFF(sb_relation_state_v2_t, interior_w),
      SB_OFF(sb_relation_state_v2_t, encoded_state),
      SB_OFF(sb_relation_state_v2_t, referent_element_id),
      SB_OFF(sb_relation_state_v2_t, referent_shared_id),
      SB_OFF(sb_relation_state_v2_t, binding_index),
      SB_OFF(sb_relation_state_v2_t, kernel_type),
      SB_OFF(sb_relation_state_v2_t, normalized_strategy),
      SB_OFF(sb_relation_state_v2_t, phase),
      SB_OFF(sb_relation_state_v2_t, has_contact),
      SB_OFF(sb_relation_state_v2_t, active),
      SB_OFF(sb_relation_state_v2_t, has_interior),
      SB_OFF(sb_relation_state_v2_t, encoded_state_size),
      SB_OFF(sb_relation_state_v2_t, chart_version),
      SB_OFF(sb_relation_state_v2_t, ap_offset_u_m),
      SB_OFF(sb_relation_state_v2_t, ap_offset_v_m),
      SB_OFF(sb_relation_state_v2_t, ap_height_m)};
  static const std::vector<uint32_t> scope_record = {
      SB_OFF(sb_scope_record_t, struct_size),
      SB_OFF(sb_scope_record_t, version),
      SB_OFF(sb_scope_record_t, anchor_name),
      SB_OFF(sb_scope_record_t, tag_filter),
      SB_OFF(sb_scope_record_t, anchor_x),
      SB_OFF(sb_scope_record_t, anchor_y),
      SB_OFF(sb_scope_record_t, anchor_z),
      SB_OFF(sb_scope_record_t, anchor_yaw),
      SB_OFF(sb_scope_record_t, time_seconds),
      SB_OFF(sb_scope_record_t, binding_index),
      SB_OFF(sb_scope_record_t, scope_root),
      SB_OFF(sb_scope_record_t, kernel_scope_origin),
      SB_OFF(sb_scope_record_t, deepest_containing),
      SB_OFF(sb_scope_record_t, navigable_element),
      SB_OFF(sb_scope_record_t, candidate_count),
      SB_OFF(sb_scope_record_t, included_count),
      SB_OFF(sb_scope_record_t, kernel_type),
      SB_OFF(sb_scope_record_t, valid),
      SB_OFF(sb_scope_record_t, anchor_active),
      SB_OFF(sb_scope_record_t, explicit_scope),
      SB_OFF(sb_scope_record_t, scope_live),
      SB_OFF(sb_scope_record_t, reserved)};
  static const std::vector<uint32_t> scope_candidate = {
      SB_OFF(sb_scope_candidate_t, struct_size),
      SB_OFF(sb_scope_candidate_t, version),
      SB_OFF(sb_scope_candidate_t, proximity),
      SB_OFF(sb_scope_candidate_t, element_id),
      SB_OFF(sb_scope_candidate_t, shared_id),
      SB_OFF(sb_scope_candidate_t, reason),
      SB_OFF(sb_scope_candidate_t, reserved)};
  static const std::vector<uint32_t> policy_params = {
      SB_OFF(sb_realization_policy_params_t, struct_size),
      SB_OFF(sb_realization_policy_params_t, version),
      SB_OFF(sb_realization_policy_params_t, eta_0),
      SB_OFF(sb_realization_policy_params_t, gamma),
      SB_OFF(sb_realization_policy_params_t, epsilon_t),
      SB_OFF(sb_realization_policy_params_t, eta_min),
      SB_OFF(sb_realization_policy_params_t, eta_max),
      SB_OFF(sb_realization_policy_params_t, progress_tau_s),
      SB_OFF(sb_realization_policy_params_t, epsilon_v),
      SB_OFF(sb_realization_policy_params_t, t_max),
      SB_OFF(sb_realization_policy_params_t, kappa),
      SB_OFF(sb_realization_policy_params_t, policy),
      SB_OFF(sb_realization_policy_params_t, reserved)};
  static const std::vector<uint32_t> weight = {
      SB_OFF(sb_realization_weight_t, struct_size),
      SB_OFF(sb_realization_weight_t, version),
      SB_OFF(sb_realization_weight_t, anchor_name),
      SB_OFF(sb_realization_weight_t, sub_region_name),
      SB_OFF(sb_realization_weight_t, proximity),
      SB_OFF(sb_realization_weight_t, estimated_time),
      SB_OFF(sb_realization_weight_t, weight),
      SB_OFF(sb_realization_weight_t, time_seconds),
      SB_OFF(sb_realization_weight_t, referent_shared_id),
      SB_OFF(sb_realization_weight_t, resolved_element_id),
      SB_OFF(sb_realization_weight_t, payload_index),
      SB_OFF(sb_realization_weight_t, channel_index),
      SB_OFF(sb_realization_weight_t, kernel_type),
      SB_OFF(sb_realization_weight_t, enabled),
      SB_OFF(sb_realization_weight_t, is_driving),
      SB_OFF(sb_realization_weight_t, is_realized_driving)};
  static const std::vector<uint32_t> channel = {
      SB_OFF(sb_realization_channel_t, struct_size),
      SB_OFF(sb_realization_channel_t, version),
      SB_OFF(sb_realization_channel_t, anchor_name),
      SB_OFF(sb_realization_channel_t, time_seconds),
      SB_OFF(sb_realization_channel_t, t_min),
      SB_OFF(sb_realization_channel_t, eta),
      SB_OFF(sb_realization_channel_t, driving_weight),
      SB_OFF(sb_realization_channel_t, driving_proximity),
      SB_OFF(sb_realization_channel_t, driving_estimated_time),
      SB_OFF(sb_realization_channel_t, realized_weight),
      SB_OFF(sb_realization_channel_t, driving_referent_shared_id),
      SB_OFF(sb_realization_channel_t, driving_payload_index),
      SB_OFF(sb_realization_channel_t, realized_referent_shared_id),
      SB_OFF(sb_realization_channel_t, realized_payload_index),
      SB_OFF(sb_realization_channel_t, candidate_count),
      SB_OFF(sb_realization_channel_t, kernel_type),
      SB_OFF(sb_realization_channel_t, policy),
      SB_OFF(sb_realization_channel_t, has_driving),
      SB_OFF(sb_realization_channel_t, driving_tied),
      SB_OFF(sb_realization_channel_t, has_realized),
      SB_OFF(sb_realization_channel_t, realized_is_policy_argmax)};
  static const std::vector<uint32_t> solver_diag = {
      SB_OFF(sb_solver_diagnostics_t, struct_size),
      SB_OFF(sb_solver_diagnostics_t, version),
      SB_OFF(sb_solver_diagnostics_t, time_seconds),
      SB_OFF(sb_solver_diagnostics_t, unconstrained_x),
      SB_OFF(sb_solver_diagnostics_t, unconstrained_y),
      SB_OFF(sb_solver_diagnostics_t, solved_x),
      SB_OFF(sb_solver_diagnostics_t, solved_y),
      SB_OFF(sb_solver_diagnostics_t, solved_yaw),
      SB_OFF(sb_solver_diagnostics_t, feasibility_snap_m),
      SB_OFF(sb_solver_diagnostics_t, goal_count),
      SB_OFF(sb_solver_diagnostics_t, has_solve),
      SB_OFF(sb_solver_diagnostics_t, has_unconstrained),
      SB_OFF(sb_solver_diagnostics_t, snapped),
      SB_OFF(sb_solver_diagnostics_t, status)};
  static const std::vector<uint32_t> solver_goal = {
      SB_OFF(sb_solver_goal_t, struct_size),
      SB_OFF(sb_solver_goal_t, version),
      SB_OFF(sb_solver_goal_t, anchor_name),
      SB_OFF(sb_solver_goal_t, witness_x),
      SB_OFF(sb_solver_goal_t, witness_y),
      SB_OFF(sb_solver_goal_t, weight),
      SB_OFF(sb_solver_goal_t, referent_shared_id),
      SB_OFF(sb_solver_goal_t, payload_index),
      SB_OFF(sb_solver_goal_t, kernel_type),
      SB_OFF(sb_solver_goal_t, position_goal),
      SB_OFF(sb_solver_goal_t, yaw_goal),
      SB_OFF(sb_solver_goal_t, reserved)};
  static const std::vector<uint32_t> payload = {
      SB_OFF(sb_relation_payload_t, struct_size),
      SB_OFF(sb_relation_payload_t, version),
      SB_OFF(sb_relation_payload_t, anchor_name),
      SB_OFF(sb_relation_payload_t, sub_region_name),
      SB_OFF(sb_relation_payload_t, proximity),
      SB_OFF(sb_relation_payload_t, estimated_time),
      SB_OFF(sb_relation_payload_t, support_distance_m),
      SB_OFF(sb_relation_payload_t, source_scope_diameter_m),
      SB_OFF(sb_relation_payload_t, contact_point_local_x),
      SB_OFF(sb_relation_payload_t, contact_point_local_y),
      SB_OFF(sb_relation_payload_t, contact_point_world_x),
      SB_OFF(sb_relation_payload_t, contact_point_world_y),
      SB_OFF(sb_relation_payload_t, contact_point_world_z),
      SB_OFF(sb_relation_payload_t, normalized_angle),
      SB_OFF(sb_relation_payload_t, normalized_radial_ratio),
      SB_OFF(sb_relation_payload_t, normalized_height_ratio),
      SB_OFF(sb_relation_payload_t, interior_u),
      SB_OFF(sb_relation_payload_t, interior_v),
      SB_OFF(sb_relation_payload_t, interior_w),
      SB_OFF(sb_relation_payload_t, encoded_state),
      SB_OFF(sb_relation_payload_t, referent_shared_id),
      SB_OFF(sb_relation_payload_t, kernel_type),
      SB_OFF(sb_relation_payload_t, normalized_strategy),
      SB_OFF(sb_relation_payload_t, encoded_state_size),
      SB_OFF(sb_relation_payload_t, has_contact),
      SB_OFF(sb_relation_payload_t, has_interior),
      SB_OFF(sb_relation_payload_t, enabled),
      SB_OFF(sb_relation_payload_t, chart_version),
      SB_OFF(sb_relation_payload_t, reserved),
      SB_OFF(sb_relation_payload_t, referent_source_space_id),
      SB_OFF(sb_relation_payload_t, referent_source_element_id),
      SB_OFF(sb_relation_payload_t, scope_shared_id),
      SB_OFF(sb_relation_payload_t, scope_source_space_id),
      SB_OFF(sb_relation_payload_t, scope_source_element_id),
      SB_OFF(sb_relation_payload_t, has_referent_source),
      SB_OFF(sb_relation_payload_t, has_scope_shared_id),
      SB_OFF(sb_relation_payload_t, has_scope_source),
      SB_OFF(sb_relation_payload_t, reserved2),
      SB_OFF(sb_relation_payload_t, source_binding_ordinal),
      SB_OFF(sb_relation_payload_t, binding_revision),
      SB_OFF(sb_relation_payload_t, has_binding_revision),
      SB_OFF(sb_relation_payload_t, r_formula_version),
      SB_OFF(sb_relation_payload_t, ap_offset_u_m),
      SB_OFF(sb_relation_payload_t, ap_offset_v_m),
      SB_OFF(sb_relation_payload_t, ap_height_m)};
  static const std::vector<uint32_t> target_evaluation = {
      SB_OFF(sb_target_relation_evaluation_t, struct_size),
      SB_OFF(sb_target_relation_evaluation_t, version),
      SB_OFF(sb_target_relation_evaluation_t, reason),
      SB_OFF(sb_target_relation_evaluation_t, sub_region_name),
      SB_OFF(sb_target_relation_evaluation_t, r_tgt),
      SB_OFF(sb_target_relation_evaluation_t, r_src),
      SB_OFF(sb_target_relation_evaluation_t, error),
      SB_OFF(sb_target_relation_evaluation_t, support_distance_m),
      SB_OFF(sb_target_relation_evaluation_t, normalizing_scope_diameter_m),
      SB_OFF(sb_target_relation_evaluation_t, contact_point_world_x),
      SB_OFF(sb_target_relation_evaluation_t, contact_point_world_y),
      SB_OFF(sb_target_relation_evaluation_t, contact_point_world_z),
      SB_OFF(sb_target_relation_evaluation_t, mapped_support_x),
      SB_OFF(sb_target_relation_evaluation_t, mapped_support_y),
      SB_OFF(sb_target_relation_evaluation_t, mapped_support_z),
      SB_OFF(sb_target_relation_evaluation_t, encoded_state),
      SB_OFF(sb_target_relation_evaluation_t, local_referent),
      SB_OFF(sb_target_relation_evaluation_t, normalizing_scope),
      SB_OFF(sb_target_relation_evaluation_t, disposition),
      SB_OFF(sb_target_relation_evaluation_t, encoded_state_size),
      SB_OFF(sb_target_relation_evaluation_t, has_contact),
      SB_OFF(sb_target_relation_evaluation_t, has_mapped_support),
      SB_OFF(sb_target_relation_evaluation_t, ap_offset_u_m),
      SB_OFF(sb_target_relation_evaluation_t, ap_offset_v_m),
      SB_OFF(sb_target_relation_evaluation_t, ap_height_m),
      SB_OFF(sb_target_relation_evaluation_t, support_basis),
      SB_OFF(sb_target_relation_evaluation_t, r_formula_version)};
  static const std::vector<uint32_t> query = {
      SB_OFF(sb_relation_query_t, struct_size),
      SB_OFF(sb_relation_query_t, version),
      SB_OFF(sb_relation_query_t, anchor_x),
      SB_OFF(sb_relation_query_t, anchor_y),
      SB_OFF(sb_relation_query_t, anchor_z),
      SB_OFF(sb_relation_query_t, anchor_yaw),
      SB_OFF(sb_relation_query_t, referent),
      SB_OFF(sb_relation_query_t, scope_root),
      SB_OFF(sb_relation_query_t, kernel_type),
      SB_OFF(sb_relation_query_t, position_dimension),
      SB_OFF(sb_relation_query_t, explicit_scope),
      SB_OFF(sb_relation_query_t, reserved)};
  static const std::vector<uint32_t> evaluation = {
      SB_OFF(sb_relation_evaluation_t, struct_size),
      SB_OFF(sb_relation_evaluation_t, version),
      SB_OFF(sb_relation_evaluation_t, sub_region_name),
      SB_OFF(sb_relation_evaluation_t, proximity),
      SB_OFF(sb_relation_evaluation_t, support_distance_m),
      SB_OFF(sb_relation_evaluation_t, source_scope_diameter_m),
      SB_OFF(sb_relation_evaluation_t, contact_point_world_x),
      SB_OFF(sb_relation_evaluation_t, contact_point_world_y),
      SB_OFF(sb_relation_evaluation_t, contact_point_world_z),
      SB_OFF(sb_relation_evaluation_t, normalized_angle),
      SB_OFF(sb_relation_evaluation_t, normalized_radial_ratio),
      SB_OFF(sb_relation_evaluation_t, normalized_height_ratio),
      SB_OFF(sb_relation_evaluation_t, interior_u),
      SB_OFF(sb_relation_evaluation_t, interior_v),
      SB_OFF(sb_relation_evaluation_t, interior_w),
      SB_OFF(sb_relation_evaluation_t, encoded_state),
      SB_OFF(sb_relation_evaluation_t, deepest_containing),
      SB_OFF(sb_relation_evaluation_t, navigable_element),
      SB_OFF(sb_relation_evaluation_t, normalized_strategy),
      SB_OFF(sb_relation_evaluation_t, encoded_state_size),
      SB_OFF(sb_relation_evaluation_t, has_contact),
      SB_OFF(sb_relation_evaluation_t, has_interior),
      SB_OFF(sb_relation_evaluation_t, ap_offset_u_m),
      SB_OFF(sb_relation_evaluation_t, ap_offset_v_m),
      SB_OFF(sb_relation_evaluation_t, ap_height_m)};
  static const std::vector<uint32_t> target_v2 = {
      SB_OFF(sb_realization_target_v2_t, struct_size),
      SB_OFF(sb_realization_target_v2_t, version),
      SB_OFF(sb_realization_target_v2_t, anchor_name),
      SB_OFF(sb_realization_target_v2_t, arrival_time),
      SB_OFF(sb_realization_target_v2_t, position_x),
      SB_OFF(sb_realization_target_v2_t, position_y),
      SB_OFF(sb_realization_target_v2_t, position_z),
      SB_OFF(sb_realization_target_v2_t, yaw),
      SB_OFF(sb_realization_target_v2_t, weight),
      SB_OFF(sb_realization_target_v2_t, requested_x),
      SB_OFF(sb_realization_target_v2_t, requested_y),
      SB_OFF(sb_realization_target_v2_t, reach_excess_m),
      SB_OFF(sb_realization_target_v2_t, referent_id),
      SB_OFF(sb_realization_target_v2_t, kernel_type),
      SB_OFF(sb_realization_target_v2_t, has_z),
      SB_OFF(sb_realization_target_v2_t, beyond_reach),
      SB_OFF(sb_realization_target_v2_t, reserved)};
  static const std::vector<uint32_t> abi_layout = {
      SB_OFF(sb_v2_abi_layout_t, descriptor_size),
      SB_OFF(sb_v2_abi_layout_t, abi_version),
      SB_OFF(sb_v2_abi_layout_t, realization_target_size),
      SB_OFF(sb_v2_abi_layout_t, pointer_size),
      SB_OFF(sb_v2_abi_layout_t, kernel_type_size),
      SB_OFF(sb_v2_abi_layout_t, arrival_time_offset),
      SB_OFF(sb_v2_abi_layout_t, anchor_name_offset),
      SB_OFF(sb_v2_abi_layout_t, position_x_offset),
      SB_OFF(sb_v2_abi_layout_t, position_y_offset),
      SB_OFF(sb_v2_abi_layout_t, position_z_offset),
      SB_OFF(sb_v2_abi_layout_t, yaw_offset),
      SB_OFF(sb_v2_abi_layout_t, weight_offset),
      SB_OFF(sb_v2_abi_layout_t, has_z_offset),
      SB_OFF(sb_v2_abi_layout_t, kernel_type_offset),
      SB_OFF(sb_v2_abi_layout_t, referent_id_offset),
      SB_OFF(sb_v2_abi_layout_t, abi_minor)};
  static const std::vector<uint32_t> anchor_pose = {
      SB_OFF(sb_anchor_pose_t, struct_size), SB_OFF(sb_anchor_pose_t, flags),
      SB_OFF(sb_anchor_pose_t, x),           SB_OFF(sb_anchor_pose_t, y),
      SB_OFF(sb_anchor_pose_t, z),           SB_OFF(sb_anchor_pose_t, orientation),
      SB_OFF(sb_anchor_pose_t, epoch)};
  static const std::vector<uint32_t> anchor_proposal = {
      SB_OFF(sb_anchor_proposal_t, struct_size),
      SB_OFF(sb_anchor_proposal_t, anchor_id),
      SB_OFF(sb_anchor_proposal_t, status),
      SB_OFF(sb_anchor_proposal_t, flags),
      SB_OFF(sb_anchor_proposal_t, cause),
      SB_OFF(sb_anchor_proposal_t, reserved),
      SB_OFF(sb_anchor_proposal_t, x),
      SB_OFF(sb_anchor_proposal_t, y),
      SB_OFF(sb_anchor_proposal_t, z),
      SB_OFF(sb_anchor_proposal_t, orientation),
      SB_OFF(sb_anchor_proposal_t, err_spacing_target),
      SB_OFF(sb_anchor_proposal_t, err_spacing_sample),
      SB_OFF(sb_anchor_proposal_t, err_facing_target),
      SB_OFF(sb_anchor_proposal_t, err_facing_sample),
      SB_OFF(sb_anchor_proposal_t, err_pose_target),
      SB_OFF(sb_anchor_proposal_t, err_pose_sample),
      SB_OFF(sb_anchor_proposal_t, err_scale_m),
      SB_OFF(sb_anchor_proposal_t, referent_shared_id)};
  static const std::vector<uint32_t> proposal_set = {
      SB_OFF(sb_proposal_set_t, struct_size),
      SB_OFF(sb_proposal_set_t, anchor_count),
      SB_OFF(sb_proposal_set_t, total_anchor_count),
      SB_OFF(sb_proposal_set_t, sample_valid),
      SB_OFF(sb_proposal_set_t, revision),
      SB_OFF(sb_proposal_set_t, source_sequence),
      SB_OFF(sb_proposal_set_t, sample_sequence),
      SB_OFF(sb_proposal_set_t, sample_epoch),
      SB_OFF(sb_proposal_set_t, sample_timestamp_s),
      SB_OFF(sb_proposal_set_t, sample_age_s)};
  static const std::vector<uint32_t> actor_event = {
      SB_OFF(sb_actor_event_t, struct_size),
      SB_OFF(sb_actor_event_t, kind),
      SB_OFF(sb_actor_event_t, revision),
      SB_OFF(sb_actor_event_t, source_sequence),
      SB_OFF(sb_actor_event_t, source_shared_id),
      SB_OFF(sb_actor_event_t, exception_id),
      SB_OFF(sb_actor_event_t, payload_size),
      SB_OFF(sb_actor_event_t, interaction_kind),
      SB_OFF(sb_actor_event_t, anchor_id),
      SB_OFF(sb_actor_event_t, x),
      SB_OFF(sb_actor_event_t, y),
      SB_OFF(sb_actor_event_t, z),
      SB_OFF(sb_actor_event_t, orientation),
      SB_OFF(sb_actor_event_t, payload)};
  static const std::vector<uint32_t> phase_config = {
      SB_OFF(sb_relation_phase_config_t, struct_size),
      SB_OFF(sb_relation_phase_config_t, reserved),
      SB_OFF(sb_relation_phase_config_t, driving_weight),
      SB_OFF(sb_relation_phase_config_t, driving_release),
      SB_OFF(sb_relation_phase_config_t, contact_proximity),
      SB_OFF(sb_relation_phase_config_t, contact_release)};
  static const std::vector<uint32_t> orientation = {
      SB_OFF(sb_orientation_t, struct_size), SB_OFF(sb_orientation_t, type),
      SB_OFF(sb_orientation_t, yaw), SB_OFF(sb_orientation_t, pitch),
      SB_OFF(sb_orientation_t, roll)};
  static const std::vector<uint32_t> bone_pose = {
      SB_OFF(sb_bone_pose_t, struct_size), SB_OFF(sb_bone_pose_t, flags),
      SB_OFF(sb_bone_pose_t, position),    SB_OFF(sb_bone_pose_t, rotation),
      SB_OFF(sb_bone_pose_t, epoch)};
  static const std::vector<uint32_t> named_bone_pose = {
      SB_OFF(sb_named_bone_pose_t, bone_name), SB_OFF(sb_named_bone_pose_t, pose)};
  static const std::vector<uint32_t> chain_joint = {
      SB_OFF(sb_chain_joint_t, struct_size),       SB_OFF(sb_chain_joint_t, parent),
      SB_OFF(sb_chain_joint_t, rest_position),     SB_OFF(sb_chain_joint_t, rest_rotation),
      SB_OFF(sb_chain_joint_t, segment_length_m),  SB_OFF(sb_chain_joint_t, limit_mask),
      SB_OFF(sb_chain_joint_t, limit_min),         SB_OFF(sb_chain_joint_t, limit_max)};
  static const std::vector<uint32_t> anchor_solve = {
      SB_OFF(sb_anchor_solve_t, struct_size),      SB_OFF(sb_anchor_solve_t, anchor_id),
      SB_OFF(sb_anchor_solve_t, flags),            SB_OFF(sb_anchor_solve_t, iterations),
      SB_OFF(sb_anchor_solve_t, object_shared_id), SB_OFF(sb_anchor_solve_t, sd_scale_m),
      SB_OFF(sb_anchor_solve_t, objective_j),      SB_OFF(sb_anchor_solve_t, facing_yaw),
      SB_OFF(sb_anchor_solve_t, driving_pose_tag)};
  static const std::vector<uint32_t> anchor_turn = {
      SB_OFF(sb_anchor_turn_t, struct_size), SB_OFF(sb_anchor_turn_t, anchor_id), SB_OFF(sb_anchor_turn_t, flags),
      SB_OFF(sb_anchor_turn_t, reserved),    SB_OFF(sb_anchor_turn_t, yaw),       SB_OFF(sb_anchor_turn_t, yaw_turn)};
  static const std::vector<uint32_t> anchor_chart = {
      SB_OFF(sb_anchor_chart_t, struct_size), SB_OFF(sb_anchor_chart_t, anchor_id),
      SB_OFF(sb_anchor_chart_t, flags),       SB_OFF(sb_anchor_chart_t, reserved),
      SB_OFF(sb_anchor_chart_t, leaf_shared_id), SB_OFF(sb_anchor_chart_t, x),
      SB_OFF(sb_anchor_chart_t, y),           SB_OFF(sb_anchor_chart_t, z),
      SB_OFF(sb_anchor_chart_t, orientation), SB_OFF(sb_anchor_chart_t, revision)};
  static const std::vector<uint32_t> target_request = {
      SB_OFF(sb_target_request_payload_t, struct_size),
      SB_OFF(sb_target_request_payload_t, channels),
      SB_OFF(sb_target_request_payload_t, sample_epoch),
      SB_OFF(sb_target_request_payload_t, sample_sequence),
      SB_OFF(sb_target_request_payload_t, sample_yaw),
      SB_OFF(sb_target_request_payload_t, yaw_turn),
      SB_OFF(sb_target_request_payload_t, proposal_flags),
      SB_OFF(sb_target_request_payload_t, proposal_cause),
      SB_OFF(sb_target_request_payload_t, facing_yaw),
      SB_OFF(sb_target_request_payload_t, proposal_status),
      SB_OFF(sb_target_request_payload_t, request_flags)};
  switch (id) {
    case SB_STRUCT_ANCHOR_SOLVE: return anchor_solve;
    case SB_STRUCT_ANCHOR_TURN: return anchor_turn;
    case SB_STRUCT_ANCHOR_CHART: return anchor_chart;
    case SB_STRUCT_BONE_POSE: return bone_pose;
    case SB_STRUCT_NAMED_BONE_POSE: return named_bone_pose;
    case SB_STRUCT_CHAIN_JOINT: return chain_joint;
    case SB_STRUCT_TARGET_REQUEST_PAYLOAD:
      return target_request;
    case SB_STRUCT_PART_CHART_COORD: return kEmpty;  // size only (sb_sizeof); no field table
    case SB_STRUCT_ANCHOR_POSE: return anchor_pose;
    case SB_STRUCT_ANCHOR_PROPOSAL: return anchor_proposal;
    case SB_STRUCT_PROPOSAL_SET: return proposal_set;
    case SB_STRUCT_ACTOR_EVENT: return actor_event;
    case SB_STRUCT_ORIENTATION: return orientation;
    case SB_STRUCT_ACTOR_STATE: return actor_state;
    case SB_STRUCT_REALIZATION_TARGET: return target;
    case SB_STRUCT_RELATION_EVENT: return relation_event;
    case SB_STRUCT_RELATION_STATE: return relation_state;
    case SB_STRUCT_RELATION_STATE_V2: return relation_state_v2;
    case SB_STRUCT_SCOPE_RECORD: return scope_record;
    case SB_STRUCT_SCOPE_CANDIDATE: return scope_candidate;
    case SB_STRUCT_REALIZATION_POLICY_PARAMS: return policy_params;
    case SB_STRUCT_REALIZATION_WEIGHT: return weight;
    case SB_STRUCT_REALIZATION_CHANNEL: return channel;
    case SB_STRUCT_SOLVER_DIAGNOSTICS: return solver_diag;
    case SB_STRUCT_SOLVER_GOAL: return solver_goal;
    case SB_STRUCT_RELATION_PAYLOAD: return payload;
    case SB_STRUCT_RELATION_QUERY: return query;
    case SB_STRUCT_RELATION_EVALUATION: return evaluation;
    case SB_STRUCT_REALIZATION_TARGET_V2: return target_v2;
    case SB_STRUCT_V2_ABI_LAYOUT: return abi_layout;
    case SB_STRUCT_RELATION_PHASE_CONFIG: return phase_config;
    case SB_STRUCT_TARGET_RELATION_EVALUATION: return target_evaluation;
  }
  return kEmpty;
}

#undef SB_OFF

uint32_t sb_get_struct_field_count(sb_struct_id_t struct_id) {
  return static_cast<uint32_t>(FieldOffsets(struct_id).size());
}

uint32_t sb_get_struct_field_offset(sb_struct_id_t struct_id, uint32_t field) {
  const auto& offsets = FieldOffsets(struct_id);
  return field < offsets.size() ? offsets[field] : UINT32_MAX;
}

// --------------------------------------------------------------------------
// Relation states v2.
// --------------------------------------------------------------------------

int32_t sb_get_relation_states_v2(sb_space_t space, sb_actor_t actor,
                                  sb_relation_state_v2_t* out, int32_t cap,
                                  uint32_t struct_size) {
  // The current layout, or the pre-pose-fit one (rows struct_size bytes apart).
  if ((struct_size != sizeof(sb_relation_state_v2_t) &&
       struct_size != kPreA4RelationStateSize) ||
      (cap > 0 && !out))
    return -1;
  auto* info = GetSpace(space);
  if (!info || !info->space) return 0;
  // Inside a relation-event callback this thread already holds the space
  // lock; everywhere else the read takes it.
  const auto actor_id = static_cast<spatialblend::entity_id_t>(actor);
  const std::vector<spatialblend::RelationState> states =
      g_inside_relation_callback
          ? info->space->GetRelationStatesFromCallback(actor_id)
          : info->space->GetRelationStates(actor_id);
  const int32_t count = static_cast<int32_t>(states.size());
  const int32_t n = std::min(count, std::max(cap, 0));
  ResetStrings(0, static_cast<size_t>(n) * 2);
  for (int32_t i = 0; i < n; ++i) {
    const auto& s = states[i];
    sb_relation_state_v2_t c;
    ObsInit(&c);
    c.anchor_name = Keep(0, s.anchor_name);
    c.sub_region_name = Keep(0, s.sub_region_name);
    c.proximity = s.proximity;
    c.support_distance_m = s.support_distance_m;
    c.source_scope_diameter_m = s.source_scope_diameter_m;
    c.contact_point_world_x = s.contact_point_world.x();
    c.contact_point_world_y = s.contact_point_world.y();
    c.contact_point_world_z = s.contact_point_world.z();
    c.normalized_angle = s.contact_normalized.angle;
    c.normalized_radial_ratio = s.contact_normalized.radial_ratio;
    c.normalized_height_ratio = s.contact_normalized.height_ratio;
    c.interior_u = s.interior_uvw.x();
    c.interior_v = s.interior_uvw.y();
    c.interior_w = s.interior_uvw.z();
    c.encoded_state_size = CopyEncoded(s.encoded_state, c.encoded_state);
    c.chart_version = s.contact_normalized.chart_version;
    c.ap_offset_u_m = s.ap_offset_u_m;
    c.ap_offset_v_m = s.ap_offset_v_m;
    c.ap_height_m = s.ap_height_m;
    c.referent_element_id = static_cast<int64_t>(s.referent_element_id);
    c.referent_shared_id = static_cast<int64_t>(s.referent_shared_id);
    c.binding_index = static_cast<int64_t>(s.binding_index);
    c.kernel_type = FromKernelType(s.kernel_type);
    c.normalized_strategy = ToNormalizedStrategy(s.contact_normalized.strategy);
    c.phase = static_cast<sb_relation_phase_t>(static_cast<int>(s.phase));
    c.has_contact = s.has_contact ? 1 : 0;
    c.active = s.active ? 1 : 0;
    c.has_interior = s.has_interior ? 1 : 0;
    c.struct_size = struct_size;
    std::memcpy(reinterpret_cast<unsigned char*>(out) +
                    static_cast<size_t>(i) * struct_size,
                &c, struct_size);
  }
  return count;
}

// --------------------------------------------------------------------------
// Scope records.
// --------------------------------------------------------------------------

int32_t sb_get_scope_records(sb_space_t space, sb_actor_t actor,
                             sb_scope_record_t* out, int32_t cap,
                             uint32_t struct_size) {
  if (!ObsSizeOk<sb_scope_record_t>(struct_size) || (cap > 0 && !out))
    return -1;
  auto* space_ptr = ObservedSpace(space);
  if (!space_ptr) return 0;
  const auto scopes =
      space_ptr->GetScopeRecords(static_cast<spatialblend::entity_id_t>(actor));
  const int32_t count = static_cast<int32_t>(scopes.size());
  const int32_t n = std::min(count, std::max(cap, 0));
  ResetStrings(1, static_cast<size_t>(n) * 2);
  for (int32_t i = 0; i < n; ++i) {
    const auto& o = scopes[i];
    const auto& r = o.record;
    sb_scope_record_t& c = out[i];
    ObsInit(&c);
    c.anchor_name = Keep(1, o.anchor_name);
    c.tag_filter = Keep(1, o.tag_filter);
    c.anchor_x = r.anchor_position.x();
    c.anchor_y = r.anchor_position.y();
    c.anchor_z = r.anchor_position.z();
    c.anchor_yaw = r.anchor_yaw;
    c.time_seconds = r.time_seconds;
    c.binding_index = static_cast<int64_t>(o.binding_index);
    c.scope_root = ElemOrZero(r.scope_root);
    c.kernel_scope_origin = ElemOrZero(r.kernel_scope_origin);
    c.deepest_containing = ElemOrZero(r.deepest_containing);
    c.navigable_element = ElemOrZero(r.navigable_element);
    c.candidate_count = static_cast<int64_t>(r.candidates.size());
    c.included_count = static_cast<int64_t>(r.IncludedCount());
    c.kernel_type = FromKernelType(o.kernel_type);
    c.valid = r.valid ? 1 : 0;
    c.anchor_active = r.anchor_active ? 1 : 0;
    c.explicit_scope = r.explicit_scope ? 1 : 0;
    c.scope_live = r.scope_live ? 1 : 0;
  }
  return count;
}

int32_t sb_get_scope_candidates(sb_space_t space, sb_actor_t actor,
                                int32_t binding_index,
                                sb_scope_candidate_t* out, int32_t cap,
                                uint32_t struct_size) {
  if (!ObsSizeOk<sb_scope_candidate_t>(struct_size) || (cap > 0 && !out) ||
      binding_index < 0)
    return -1;
  auto* space_ptr = ObservedSpace(space);
  if (!space_ptr) return -1;
  const auto scopes =
      space_ptr->GetScopeRecords(static_cast<spatialblend::entity_id_t>(actor));
  if (static_cast<size_t>(binding_index) >= scopes.size()) return -1;
  const auto& candidates = scopes[binding_index].record.candidates;
  const int32_t count = static_cast<int32_t>(candidates.size());
  const int32_t n = std::min(count, std::max(cap, 0));
  for (int32_t i = 0; i < n; ++i) {
    const auto& cand = candidates[i];
    sb_scope_candidate_t& c = out[i];
    ObsInit(&c);
    c.proximity = cand.proximity;
    c.element_id = static_cast<int64_t>(cand.element_id);
    c.shared_id = cand.shared_or_local_id;
    c.reason = static_cast<sb_scope_reason_t>(static_cast<int>(cand.reason));
  }
  return count;
}

// --------------------------------------------------------------------------
// Policy parameters.
// --------------------------------------------------------------------------

static void FillPolicyParams(const spatialblend::RealizationPolicyConfig& p,
                             const spatialblend::ImminenceConfig& m,
                             sb_realization_policy_params_t* out) {
  ObsInit(out);
  out->eta_0 = p.eta_0;
  out->gamma = p.decay_gamma;
  out->epsilon_t = p.epsilon;
  out->eta_min = p.eta_min;
  out->eta_max = p.eta_max;
  out->progress_tau_s = m.progress_tau_s;
  out->epsilon_v = m.epsilon_v;
  out->t_max = m.t_max;
  out->kappa = m.engage_kappa;
  out->policy = ToCPolicy(p.type);
}

int32_t sb_get_default_realization_policy_params(
    sb_realization_policy_params_t* out, uint32_t struct_size) {
  if (!out || !ObsSizeOk<sb_realization_policy_params_t>(struct_size))
    return 0;
  FillPolicyParams(spatialblend::ProductionRealizationPolicyConfig(),
                   spatialblend::ImminenceConfig(), out);
  return 1;
}

int32_t sb_get_realization_policy_params(sb_space_t space,
                                         sb_realization_policy_params_t* out,
                                         uint32_t struct_size) {
  if (!out || !ObsSizeOk<sb_realization_policy_params_t>(struct_size))
    return 0;
  auto* space_ptr = UnlockedCallerSpace(space);
  if (!space_ptr) return 0;
  FillPolicyParams(space_ptr->GetRealizationPolicyParams(),
                   space_ptr->GetImminenceParams(), out);
  return 1;
}

int32_t sb_set_realization_policy_params(
    sb_space_t space, const sb_realization_policy_params_t* params,
    uint32_t struct_size) {
  if (!params || !ObsSizeOk<sb_realization_policy_params_t>(struct_size) ||
      params->struct_size != sizeof(sb_realization_policy_params_t))
    return 0;
  auto* space_ptr = UnlockedCallerSpace(space);
  if (!space_ptr) return 0;
  spatialblend::RealizationPolicyConfig p;
  p.eta_0 = params->eta_0;
  p.decay_gamma = params->gamma;
  p.epsilon = params->epsilon_t;
  p.eta_min = params->eta_min;
  p.eta_max = params->eta_max;
  spatialblend::ImminenceConfig m;
  m.progress_tau_s = params->progress_tau_s;
  m.epsilon_v = params->epsilon_v;
  m.t_max = params->t_max;
  m.engage_kappa = params->kappa;
  return space_ptr->SetRealizationPolicyParams(p, m) ? 1 : 0;
}

// --------------------------------------------------------------------------
// Weights and channels.
// --------------------------------------------------------------------------

int32_t sb_get_realization_weights(sb_space_t space, sb_actor_t actor,
                                   sb_realization_weight_t* out, int32_t cap,
                                   uint32_t struct_size) {
  if (!ObsSizeOk<sb_realization_weight_t>(struct_size) || (cap > 0 && !out))
    return -1;
  auto* space_ptr = ObservedSpace(space);
  if (!space_ptr) return 0;
  spatialblend::Space::PolicyObservation obs;
  if (!space_ptr->CopyPolicyObservation(
          static_cast<spatialblend::entity_id_t>(actor), &obs))
    return 0;
  const int32_t count = static_cast<int32_t>(obs.payloads.size());
  const int32_t n = std::min(count, std::max(cap, 0));
  std::vector<int64_t> channel_of(obs.payloads.size(), -1);
  std::vector<bool> driving(obs.payloads.size(), false);
  std::vector<bool> realized(obs.payloads.size(), false);
  for (size_t ch = 0; ch < obs.channels.size(); ++ch) {
    for (size_t idx : obs.channels[ch].payload_indices)
      channel_of[idx] = static_cast<int64_t>(ch);
    if (obs.channels[ch].has_driving)
      driving[obs.channels[ch].driving_index] = true;
    if (obs.channels[ch].has_realized)
      realized[obs.channels[ch].realized_index] = true;
  }
  ResetStrings(2, static_cast<size_t>(n) * 2);
  for (int32_t i = 0; i < n; ++i) {
    const auto& p = obs.payloads[i];
    sb_realization_weight_t& c = out[i];
    ObsInit(&c);
    c.anchor_name = Keep(2, p.subject_anchor_name);
    c.sub_region_name = Keep(2, p.support.sub_region_name);
    c.proximity = p.grounded_relation.proximity;
    c.estimated_time = p.estimated_time;
    c.weight = static_cast<size_t>(i) < obs.weights.size() ? obs.weights[i]
                                                           : 0.0;
    c.time_seconds = obs.time_seconds;
    c.referent_shared_id = static_cast<int64_t>(p.referent_shared_id);
    c.resolved_element_id = ElemOrZero(p.resolved_local_element_id);
    c.payload_index = i;
    c.channel_index = channel_of[i];
    c.kernel_type = FromKernelType(p.kernel_type);
    c.enabled = p.enabled ? 1 : 0;
    c.is_driving = driving[i] ? 1 : 0;
    c.is_realized_driving = realized[i] ? 1 : 0;
  }
  return count;
}

int32_t sb_get_realization_channels(sb_space_t space, sb_actor_t actor,
                                    sb_realization_channel_t* out,
                                    int32_t cap, uint32_t struct_size) {
  if (!ObsSizeOk<sb_realization_channel_t>(struct_size) || (cap > 0 && !out))
    return -1;
  auto* space_ptr = ObservedSpace(space);
  if (!space_ptr) return 0;
  spatialblend::Space::PolicyObservation obs;
  if (!space_ptr->CopyPolicyObservation(
          static_cast<spatialblend::entity_id_t>(actor), &obs))
    return 0;
  const int32_t count = static_cast<int32_t>(obs.channels.size());
  const int32_t n = std::min(count, std::max(cap, 0));
  ResetStrings(3, static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const auto& d = obs.channels[i];
    sb_realization_channel_t& c = out[i];
    ObsInit(&c);
    c.anchor_name = Keep(3, d.anchor_name);
    c.time_seconds = obs.time_seconds;
    c.t_min = d.t_min;
    c.eta = d.eta;
    c.candidate_count = static_cast<int64_t>(d.payload_indices.size());
    c.kernel_type = FromKernelType(d.kernel_type);
    c.policy = ToCPolicy(obs.config.type);
    c.has_driving = d.has_driving ? 1 : 0;
    c.driving_tied = d.driving_tied ? 1 : 0;
    c.driving_payload_index = -1;
    c.realized_payload_index = -1;
    c.has_realized = d.has_realized ? 1 : 0;
    c.realized_is_policy_argmax =
        (d.has_realized && d.has_driving && d.realized_index == d.driving_index)
            ? 1 : 0;
    if (d.has_realized) {
      const auto& p = obs.payloads[d.realized_index];
      c.realized_weight = d.realized_weight;
      c.realized_referent_shared_id = static_cast<int64_t>(p.referent_shared_id);
      c.realized_payload_index = static_cast<int64_t>(d.realized_index);
    }
    if (d.has_driving) {
      const auto& p = obs.payloads[d.driving_index];
      c.driving_weight = d.driving_weight;
      c.driving_proximity = p.grounded_relation.proximity;
      c.driving_estimated_time = p.estimated_time;
      c.driving_referent_shared_id = static_cast<int64_t>(p.referent_shared_id);
      c.driving_payload_index = static_cast<int64_t>(d.driving_index);
    }
  }
  return count;
}

// --------------------------------------------------------------------------
// Solver diagnostics.
// --------------------------------------------------------------------------

int32_t sb_get_solver_diagnostics(sb_space_t space, sb_actor_t actor,
                                  sb_solver_diagnostics_t* out,
                                  uint32_t struct_size) {
  if (!out || !ObsSizeOk<sb_solver_diagnostics_t>(struct_size)) return 0;
  ObsInit(out);
  auto* space_ptr = ObservedSpace(space);
  if (!space_ptr) return 0;
  spatialblend::Space::SolverObservation obs;
  if (!space_ptr->CopySolverObservation(
          static_cast<spatialblend::entity_id_t>(actor), &obs))
    return 0;
  out->time_seconds = obs.time_seconds;
  out->unconstrained_x = obs.solve.unconstrained_position.x();
  out->unconstrained_y = obs.solve.unconstrained_position.y();
  out->solved_x = obs.solve.position.x();
  out->solved_y = obs.solve.position.y();
  out->solved_yaw = obs.solve.yaw;
  out->feasibility_snap_m = obs.solve.feasibility_snap_m;
  out->goal_count = static_cast<int64_t>(obs.solve.goals.size());
  out->has_solve = 1;
  out->has_unconstrained = obs.solve.has_unconstrained_position ? 1 : 0;
  out->snapped = obs.solve.feasibility_snapped ? 1 : 0;
  out->status = ToCSolverStatus(obs.solve.status);
  return 1;
}

int32_t sb_get_solver_goals(sb_space_t space, sb_actor_t actor,
                            sb_solver_goal_t* out, int32_t cap,
                            uint32_t struct_size) {
  if (!ObsSizeOk<sb_solver_goal_t>(struct_size) || (cap > 0 && !out))
    return -1;
  auto* space_ptr = ObservedSpace(space);
  if (!space_ptr) return 0;
  const auto actor_id = static_cast<spatialblend::entity_id_t>(actor);
  spatialblend::Space::SolverObservation obs;
  if (!space_ptr->CopySolverObservation(actor_id, &obs)) return 0;
  spatialblend::Space::PolicyObservation policy;
  const bool has_policy = space_ptr->CopyPolicyObservation(actor_id, &policy);
  const int32_t count = static_cast<int32_t>(obs.solve.goals.size());
  const int32_t n = std::min(count, std::max(cap, 0));
  ResetStrings(4, static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const auto& g = obs.solve.goals[i];
    sb_solver_goal_t& c = out[i];
    ObsInit(&c);
    c.witness_x = g.witness.x();
    c.witness_y = g.witness.y();
    c.weight = g.weight;
    c.position_goal = g.position_goal ? 1 : 0;
    c.yaw_goal = g.yaw_goal ? 1 : 0;
    c.payload_index = -1;
    c.anchor_name = Keep(4, "");
    if (g.payload_index < obs.goal_payload_index.size()) {
      const size_t idx = obs.goal_payload_index[g.payload_index];
      c.payload_index = static_cast<int64_t>(idx);
      if (has_policy && idx < policy.payloads.size()) {
        const auto& p = policy.payloads[idx];
        c.anchor_name = Keep(4, p.subject_anchor_name);
        c.referent_shared_id = static_cast<int64_t>(p.referent_shared_id);
        c.kernel_type = FromKernelType(p.kernel_type);
      }
    }
  }
  return count;
}

int32_t sb_get_realization_targets_v2(sb_space_t space, sb_actor_t actor,
                                      sb_realization_target_v2_t* out,
                                      int32_t cap, uint32_t struct_size) {
  if (!ObsSizeOk<sb_realization_target_v2_t>(struct_size) || (cap > 0 && !out))
    return -1;
  auto* space_ptr = UnlockedCallerSpace(space);
  if (!space_ptr) return 0;
  spatialblend::RealizationResult result;
  if (!space_ptr->CopyRealizationResult(
          static_cast<spatialblend::entity_id_t>(actor), &result))
    return 0;
  const int32_t count = static_cast<int32_t>(result.targets.size());
  const int32_t n = std::min(count, std::max(cap, 0));
  ResetStrings(5, static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const auto& t = result.targets[i];
    sb_realization_target_v2_t& c = out[i];
    ObsInit(&c);
    c.anchor_name = Keep(5, t.anchor_name.empty() ? "root" : t.anchor_name);
    c.arrival_time = t.arrival_time;
    c.position_x = t.position.x();
    c.position_y = t.position.y();
    c.position_z = t.has_z ? t.z : 0.0;
    c.yaw = t.yaw;
    c.weight = t.weight;
    // No reach clamp exists; requested == position, and an engaged
    // target beyond the anchor's reach is flagged, never moved.
    c.requested_x = t.position.x();
    c.requested_y = t.position.y();
    c.reach_excess_m = t.beyond_reach ? t.reach_excess_m : 0.0;
    c.referent_id = static_cast<int64_t>(t.referent_id);
    c.kernel_type = FromKernelType(t.kernel_type);
    c.has_z = t.has_z ? 1 : 0;
    c.beyond_reach = t.beyond_reach ? 1 : 0;
  }
  return count;
}

// --------------------------------------------------------------------------
// Injection and pure evaluation.
// --------------------------------------------------------------------------

// sb_relation_payload_t: the current layout; the older one that ends
// before source_binding_ordinal (binding revision unknown); or the oldest
// one that ends before chart_version (chart_version is then 0: a pure radial
// chart, and no identities).
constexpr uint32_t kLegacyPayloadSize =
    static_cast<uint32_t>(offsetof(sb_relation_payload_t, chart_version));
constexpr uint32_t kPreBindingPayloadSize = static_cast<uint32_t>(
    offsetof(sb_relation_payload_t, source_binding_ordinal));
// The pre-ap_offset layout, ending before ap_offset_u_m (no AffordancePose
// offset/height; those three then read NaN).
constexpr uint32_t kPreA4PayloadSize =
    static_cast<uint32_t>(offsetof(sb_relation_payload_t, ap_offset_u_m));
bool PayloadSizeOk(uint32_t struct_size) {
  return struct_size == sizeof(sb_relation_payload_t) ||
         struct_size == kPreA4PayloadSize ||
         struct_size == kPreBindingPayloadSize ||
         struct_size == kLegacyPayloadSize;
}

bool ConvertPayload(const sb_relation_payload_t& in, uint32_t struct_size,
                    spatialblend::RelationPayload* out) {
  if (in.struct_size != struct_size || !ValidKernel(in.kernel_type) ||
      in.encoded_state_size < 0 ||
      in.encoded_state_size > SB_MAX_ENCODED_STATE)
    return false;
  spatialblend::RelationPayload& p = *out;
  p.referent_shared_id =
      static_cast<spatialblend::shared_element_id_t>(in.referent_shared_id);
  p.subject_anchor_name = in.anchor_name ? in.anchor_name : "";
  p.kernel_type = ToKernelType(in.kernel_type);
  p.enabled = in.enabled != 0;
  p.estimated_time = in.estimated_time;
  // The injection ABI names its scheduling score `estimated_time`; the policy
  // reads an update count, so an injected payload carries that value as its
  // count (a non-positive or non-finite value is no valid intent).
  p.update_count = in.estimated_time;
  p.grounded_relation.proximity = in.proximity;
  p.grounded_relation.support_distance_m = in.support_distance_m;
  p.grounded_relation.source_scope_diameter_m = in.source_scope_diameter_m;
  // Absent (struct_size < sizeof) leaves the GroundedRelation
  // default of NaN ("unknown"), never 0.
  // Only a payload that declares formula version 1 carries valid offsets: a
  // zero-initialised struct must not read as "exactly on the support".
  if (struct_size == sizeof(sb_relation_payload_t) &&
      in.r_formula_version == spatialblend::kApRFormulaPoseFitV1) {
    p.grounded_relation.ap_offset_u_m = in.ap_offset_u_m;
    p.grounded_relation.ap_offset_v_m = in.ap_offset_v_m;
    p.grounded_relation.ap_height_m = in.ap_height_m;
  }
  p.grounded_relation.encoded_state =
      Eigen::VectorXd::Zero(in.encoded_state_size);
  for (int32_t k = 0; k < in.encoded_state_size; ++k)
    p.grounded_relation.encoded_state(k) = in.encoded_state[k];
  p.support.has_contact = in.has_contact != 0;
  p.support.contact_point_local =
      Eigen::Vector2d(in.contact_point_local_x, in.contact_point_local_y);
  p.support.contact_point_world = Eigen::Vector3d(
      in.contact_point_world_x, in.contact_point_world_y,
      in.contact_point_world_z);
  p.support.contact_normalized.strategy =
      in.normalized_strategy == SB_NORMALIZED_MVC
          ? spatialblend::NormalizedPolygonPosition::Strategy::kMVC
          : spatialblend::NormalizedPolygonPosition::Strategy::kRadial;
  p.support.contact_normalized.angle = in.normalized_angle;
  p.support.contact_normalized.radial_ratio = in.normalized_radial_ratio;
  p.support.contact_normalized.height_ratio = in.normalized_height_ratio;
  p.support.contact_normalized.chart_version =
      struct_size >= kPreBindingPayloadSize ? in.chart_version : 0u;
  p.support.has_interior = in.has_interior != 0;
  p.support.interior.uvw =
      Eigen::Vector3d(in.interior_u, in.interior_v, in.interior_w);
  p.support.sub_region_name = in.sub_region_name ? in.sub_region_name : "";
  // The binding revision lives in the pre-ap_offset layout too (it precedes
  // ap_offset_u_m), so it is read from any layout that contains it.
  if (struct_size >= kPreA4PayloadSize) {
    p.source_binding_ordinal = in.source_binding_ordinal;
    if (in.has_binding_revision) {
      p.binding_revision = in.binding_revision;
      p.binding_revision_known = true;
    }
  }
  if (struct_size >= kPreBindingPayloadSize) {
    if (in.has_referent_source) {
      p.referent_source_space_id =
          static_cast<spatialblend::space_id_t>(in.referent_source_space_id);
      p.referent_source_element_id =
          static_cast<spatialblend::element_id_t>(
              in.referent_source_element_id);
    }
    if (in.has_scope_shared_id)
      p.scope_shared_id =
          static_cast<spatialblend::shared_element_id_t>(in.scope_shared_id);
    if (in.has_scope_source) {
      p.scope_source_space_id =
          static_cast<spatialblend::space_id_t>(in.scope_source_space_id);
      p.scope_source_element_id =
          static_cast<spatialblend::element_id_t>(in.scope_source_element_id);
    }
  }
  return true;
}

int32_t sb_inject_relation_payloads(sb_space_t space, sb_actor_t actor,
                                    const sb_relation_payload_t* payloads,
                                    int32_t count, uint32_t struct_size,
                                    double time_seconds) {
  if (!PayloadSizeOk(struct_size) || count < 0 || (count > 0 && !payloads) ||
      !std::isfinite(time_seconds))
    return 0;
  auto* space_ptr = UnlockedCallerSpace(space);
  if (!space_ptr) return 0;
  std::vector<spatialblend::RelationPayload> converted(
      static_cast<size_t>(count));
  // Rows are struct_size apart (the legacy layout is shorter).
  const auto* bytes = reinterpret_cast<const unsigned char*>(payloads);
  for (int32_t i = 0; i < count; ++i) {
    sb_relation_payload_t row;
    std::memset(&row, 0, sizeof(row));
    std::memcpy(&row, bytes + static_cast<size_t>(i) * struct_size,
                struct_size);
    if (!ConvertPayload(row, struct_size, &converted[i])) return 0;
  }
  return space_ptr->InjectRelationPayloads(
             static_cast<spatialblend::entity_id_t>(actor), std::move(converted),
             time_seconds)
             ? 1
             : 0;
}

static_assert(static_cast<int>(
                  spatialblend::TargetEvaluationDisposition::kEvaluated) ==
                  SB_TARGET_EVAL_EVALUATED &&
              static_cast<int>(
                  spatialblend::TargetEvaluationDisposition::kUnmapped) ==
                  SB_TARGET_EVAL_UNMAPPED &&
              static_cast<int>(spatialblend::TargetEvaluationDisposition::
                                   kNoSupportGeometry) ==
                  SB_TARGET_EVAL_NO_SUPPORT_GEOMETRY &&
              static_cast<int>(
                  spatialblend::TargetEvaluationDisposition::kPartNotFound) ==
                  SB_TARGET_EVAL_PART_NOT_FOUND &&
              static_cast<int>(spatialblend::TargetEvaluationDisposition::
                                   kUnsupportedSupport) ==
                  SB_TARGET_EVAL_UNSUPPORTED_SUPPORT &&
              static_cast<int>(
                  spatialblend::TargetEvaluationDisposition::kInvalidInput) ==
                  SB_TARGET_EVAL_INVALID_INPUT,
              "sb_target_eval_disposition_t mirrors TargetEvaluationDisposition");

int32_t sb_evaluate_target_relation(sb_space_t space,
                                    const sb_relation_payload_t* payload,
                                    uint32_t payload_struct_size,
                                    const sb_relation_query_t* candidate,
                                    sb_target_relation_evaluation_t* out) {
  if (!payload || !candidate || !out || !PayloadSizeOk(payload_struct_size) ||
      candidate->struct_size != sizeof(sb_relation_query_t) ||
      (out->struct_size != sizeof(sb_target_relation_evaluation_t) &&
       out->struct_size != kPreA3TargetEvaluationSize &&
       out->struct_size != kPreA4TargetEvaluationSize))
    return 0;
  const uint32_t out_size = out->struct_size;
  auto* space_ptr = UnlockedCallerSpace(space);
  if (!space_ptr) return 0;
  sb_relation_payload_t row;
  std::memset(&row, 0, sizeof(row));
  std::memcpy(&row, payload, payload_struct_size);
  spatialblend::RelationPayload p;
  if (!ConvertPayload(row, payload_struct_size, &p)) return 0;
  spatialblend::TargetAnchorPose pose;
  pose.position = Eigen::Vector3d(candidate->anchor_x, candidate->anchor_y,
                                  candidate->anchor_z);
  pose.yaw = candidate->anchor_yaw;
  spatialblend::TargetRelationEvaluation e;
  if (!space_ptr->EvaluateTargetRelation(
          p, pose,
          candidate->scope_root == 0
              ? spatialblend::kInvalidElementId
              : static_cast<spatialblend::element_id_t>(candidate->scope_root),
          &e))
    return 0;
  sb_target_relation_evaluation_t full;
  ObsInit(&full);
  sb_target_relation_evaluation_t* o = &full;
  ResetStrings(7, 2);
  o->reason = Keep(7, e.reason);
  o->sub_region_name = Keep(7, e.support.sub_region_name);
  o->r_tgt = e.r_tgt;
  o->r_src = e.r_src;
  o->error = e.error;
  o->support_distance_m = e.relation.support_distance_m;
  o->ap_offset_u_m = e.relation.ap_offset_u_m;
  o->ap_offset_v_m = e.relation.ap_offset_v_m;
  o->ap_height_m = e.relation.ap_height_m;
  o->normalizing_scope_diameter_m = e.normalizing_scope_diameter_m;
  o->contact_point_world_x = e.support.contact_point_world.x();
  o->contact_point_world_y = e.support.contact_point_world.y();
  o->contact_point_world_z = e.support.contact_point_world.z();
  o->mapped_support_x = e.mapped_support.x();
  o->mapped_support_y = e.mapped_support.y();
  o->mapped_support_z = e.mapped_support.z();
  o->encoded_state_size =
      CopyEncoded(e.relation.encoded_state, o->encoded_state);
  o->local_referent = ElemOrZero(e.local_referent_id);
  o->normalizing_scope = ElemOrZero(e.normalizing_scope_id);
  o->disposition = static_cast<sb_target_eval_disposition_t>(e.disposition);
  o->has_contact = e.support.has_contact ? 1 : 0;
  o->has_mapped_support = e.has_mapped_support ? 1 : 0;
  o->support_basis = static_cast<sb_target_support_basis_t>(
      static_cast<int>(e.support_basis));
  o->r_formula_version = e.r_formula_version;
  StoreOld(full, out, out_size);
  return 1;
}

int32_t sb_evaluate_relation(sb_space_t space, const sb_relation_query_t* query,
                             sb_relation_evaluation_t* out) {
  if (!query || !out || query->struct_size != sizeof(sb_relation_query_t) ||
      (out->struct_size != sizeof(sb_relation_evaluation_t) &&
       out->struct_size != kPreA4RelationEvaluationSize) ||
      !ValidKernel(query->kernel_type))
    return 0;
  const uint32_t out_size = out->struct_size;
  const int32_t dim = query->position_dimension;
  if (dim != 0 && dim != 2 && dim != 3) return 0;
  auto* space_ptr = UnlockedCallerSpace(space);
  if (!space_ptr) return 0;
  std::vector<spatialblend::StateType> dof = {
      dim == 3 ? spatialblend::StateType::kPosition3D
               : spatialblend::StateType::kPosition2D,
      spatialblend::StateType::kRotationYaw};
  spatialblend::Space::RelationEvaluation eval;
  if (!space_ptr->EvaluateRelation(
          ToKernelType(query->kernel_type),
          Eigen::Vector3d(query->anchor_x, query->anchor_y, query->anchor_z),
          query->anchor_yaw,
          static_cast<spatialblend::element_id_t>(query->referent),
          query->scope_root == 0
              ? spatialblend::kInvalidElementId
              : static_cast<spatialblend::element_id_t>(query->scope_root),
          query->explicit_scope != 0, dof, &eval))
    return 0;
  sb_relation_evaluation_t full;
  ObsInit(&full);
  sb_relation_evaluation_t* o = &full;
  ResetStrings(6, 1);
  o->sub_region_name = Keep(6, eval.support.sub_region_name);
  o->proximity = eval.grounded.proximity;
  o->support_distance_m = eval.grounded.support_distance_m;
  o->ap_offset_u_m = eval.grounded.ap_offset_u_m;
  o->ap_offset_v_m = eval.grounded.ap_offset_v_m;
  o->ap_height_m = eval.grounded.ap_height_m;
  o->source_scope_diameter_m = eval.grounded.source_scope_diameter_m;
  o->contact_point_world_x = eval.support.contact_point_world.x();
  o->contact_point_world_y = eval.support.contact_point_world.y();
  o->contact_point_world_z = eval.support.contact_point_world.z();
  o->normalized_angle = eval.support.contact_normalized.angle;
  o->normalized_radial_ratio = eval.support.contact_normalized.radial_ratio;
  o->normalized_height_ratio = eval.support.contact_normalized.height_ratio;
  o->interior_u = eval.support.interior.uvw.x();
  o->interior_v = eval.support.interior.uvw.y();
  o->interior_w = eval.support.interior.uvw.z();
  o->encoded_state_size =
      CopyEncoded(eval.grounded.encoded_state, o->encoded_state);
  o->deepest_containing = ElemOrZero(eval.deepest_containing);
  o->navigable_element = ElemOrZero(eval.navigable_element);
  o->normalized_strategy =
      ToNormalizedStrategy(eval.support.contact_normalized.strategy);
  o->has_contact = eval.support.has_contact ? 1 : 0;
  o->has_interior = eval.support.has_interior ? 1 : 0;
  StoreOld(full, out, out_size);
  return 1;
}

// --------------------------------------------------------------------------
// Element tree readback.
// --------------------------------------------------------------------------

int32_t sb_get_element_world_transform(sb_space_t space, sb_element_t elem,
                                       double out_row_major[16]) {
  if (!out_row_major) return 0;
  auto* space_ptr = UnlockedCallerSpace(space);
  if (!space_ptr) return 0;
  Eigen::Matrix4d m;
  if (!space_ptr->GetElementWorldTransform(
          static_cast<spatialblend::element_id_t>(elem), &m))
    return 0;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) out_row_major[r * 4 + c] = m(r, c);
  return 1;
}

int32_t sb_get_element_parent(sb_space_t space, sb_element_t elem,
                              sb_element_t* out_parent) {
  if (!out_parent) return 0;
  auto* space_ptr = UnlockedCallerSpace(space);
  if (!space_ptr) return 0;
  spatialblend::element_id_t parent = spatialblend::kInvalidElementId;
  if (!space_ptr->GetElementParent(
          static_cast<spatialblend::element_id_t>(elem), &parent))
    return 0;
  *out_parent = static_cast<sb_element_t>(ElemOrZero(parent));
  return 1;
}
