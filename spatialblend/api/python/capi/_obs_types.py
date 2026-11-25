# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""ctypes mirrors of the observability, legacy-query, rich-realization and
receiver-input structs of the C API.

Headers mirrored (paths relative to spatialblend-core):
  spatialblend/api/c/spatialblend_observability.h
  spatialblend/api/c/spatialblend_type.h           (sb_relation_state_t,
                                                sb_v2_abi_layout_t,
                                                sb_actor_observation_t,
                                                sb_support_correspondence_t)
  spatialblend/api/c/spatialblend_rich.h
  spatialblend/api/c/spatialblend_receiver_input.h
  spatialblend/api/c/spatialblend_receiver_reevaluation.h

Field ORDER and TYPES match the C declarations exactly. Every struct listed
in OBS_STRUCT_IDS / RICH_LAYOUTS / RECEIVER_INPUT_LAYOUTS is checked field by
field against the native offset queries (sb_get_struct_field_offset,
sb_get_rich_realization_field_offset, sb_get_receiver_input_field_offset) by
_api.check_struct_layout(), so a drift fails at load time, not as silent
corruption.
"""
import ctypes

c_int = ctypes.c_int  # plain C enums
MAX_ENCODED_STATE = 8
OBSERVABILITY_VERSION = 1

# ---------------------------------------------------------------------------
# sb_struct_id_t (spatialblend_type.h)
# ---------------------------------------------------------------------------
SB_STRUCT_ACTOR_STATE = 0
SB_STRUCT_REALIZATION_TARGET = 1
SB_STRUCT_RELATION_EVENT = 2
SB_STRUCT_RELATION_STATE = 3
SB_STRUCT_RELATION_STATE_V2 = 4
SB_STRUCT_SCOPE_RECORD = 5
SB_STRUCT_SCOPE_CANDIDATE = 6
SB_STRUCT_REALIZATION_POLICY_PARAMS = 7
SB_STRUCT_REALIZATION_WEIGHT = 8
SB_STRUCT_REALIZATION_CHANNEL = 9
SB_STRUCT_SOLVER_DIAGNOSTICS = 10
SB_STRUCT_SOLVER_GOAL = 11
SB_STRUCT_RELATION_PAYLOAD = 12
SB_STRUCT_RELATION_QUERY = 13
SB_STRUCT_RELATION_EVALUATION = 14
SB_STRUCT_REALIZATION_TARGET_V2 = 15
SB_STRUCT_V2_ABI_LAYOUT = 16
SB_STRUCT_RELATION_PHASE_CONFIG = 17
SB_STRUCT_TARGET_RELATION_EVALUATION = 18
# ids 19 and 20 (ABI 3 embodied pose and proposal) are retired, not reused.
SB_STRUCT_ORIENTATION = 21
SB_STRUCT_ANCHOR_POSE = 22
SB_STRUCT_ANCHOR_PROPOSAL = 23
SB_STRUCT_ANCHOR_SOLVE = 30  # ABI 4.11
SB_STRUCT_ANCHOR_TURN = 31  # ABI 4.12
SB_STRUCT_ANCHOR_CHART = 33  # ABI 4.14
SB_STRUCT_PROPOSAL_SET = 24
SB_STRUCT_ACTOR_EVENT = 25

# sb_target_eval_disposition_t
TARGET_EVAL_DISPOSITION_NAMES = {
    0: "evaluated",
    1: "unmapped",
    2: "no_support_geometry",
    3: "part_not_found",
    4: "unsupported_support",
    5: "invalid_input",
}

# sb_scope_reason_t
SCOPE_REASON_NAMES = {
    0: "included",
    1: "explicit_scope",
    2: "tag",
    3: "actor_element",
    4: "observed_point_control",
    5: "observed_point_ineligible",
    6: "missing_element",
    7: "surface_prefilter",
    8: "not_visible",
    9: "unreachable",
    12: "observed_point_expired",
    14: "outside_parent_region",
}


class SbRelationState(ctypes.Structure):
    """sb_relation_state_t (spatialblend_type.h)."""
    _fields_ = [
        ("anchor_name", ctypes.c_char_p),
        ("proximity", ctypes.c_double),
        ("contact_point_world_x", ctypes.c_double),
        ("contact_point_world_y", ctypes.c_double),
        ("contact_point_world_z", ctypes.c_double),
        ("normalized_angle", ctypes.c_double),
        ("normalized_radial_ratio", ctypes.c_double),
        ("normalized_height_ratio", ctypes.c_double),
        ("referent_element_id", ctypes.c_int64),
        ("referent_shared_id", ctypes.c_int64),
        ("kernel_type", c_int),
        ("normalized_strategy", c_int),
        ("has_contact", ctypes.c_int32),
        ("active", ctypes.c_int32),
    ]


class SbV2AbiLayout(ctypes.Structure):
    """sb_v2_abi_layout_t (spatialblend_type.h)."""
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "descriptor_size", "abi_version", "realization_target_size",
        "pointer_size", "kernel_type_size", "arrival_time_offset",
        "anchor_name_offset", "position_x_offset", "position_y_offset",
        "position_z_offset", "yaw_offset", "weight_offset", "has_z_offset",
        "kernel_type_offset", "referent_id_offset", "abi_minor")]


class SbActorObservation(ctypes.Structure):
    """sb_actor_observation_t (spatialblend_type.h)."""
    _fields_ = [
        ("anchor_name", ctypes.c_char_p),
        ("position_x", ctypes.c_double),
        ("position_y", ctypes.c_double),
        ("position_z", ctypes.c_double),
        ("yaw", ctypes.c_double),
    ]


class SbSupportCorrespondence(ctypes.Structure):
    """sb_support_correspondence_t (spatialblend_type.h)."""
    _fields_ = [
        ("source_space", ctypes.c_int64),
        ("source_element", ctypes.c_int64),
        ("target_space", ctypes.c_int64),
        ("target_element", ctypes.c_int64),
    ]


# ---------------------------------------------------------------------------
# spatialblend_observability.h
# ---------------------------------------------------------------------------

class SbRelationStateV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("anchor_name", ctypes.c_char_p),
        ("sub_region_name", ctypes.c_char_p),
        ("proximity", ctypes.c_double),
        ("support_distance_m", ctypes.c_double),
        ("source_scope_diameter_m", ctypes.c_double),
        ("contact_point_world_x", ctypes.c_double),
        ("contact_point_world_y", ctypes.c_double),
        ("contact_point_world_z", ctypes.c_double),
        ("normalized_angle", ctypes.c_double),
        ("normalized_radial_ratio", ctypes.c_double),
        ("normalized_height_ratio", ctypes.c_double),
        ("interior_u", ctypes.c_double),
        ("interior_v", ctypes.c_double),
        ("interior_w", ctypes.c_double),
        ("encoded_state", ctypes.c_double * MAX_ENCODED_STATE),
        ("referent_element_id", ctypes.c_int64),
        ("referent_shared_id", ctypes.c_int64),
        ("binding_index", ctypes.c_int64),
        ("kernel_type", c_int),
        ("normalized_strategy", c_int),
        ("phase", c_int),
        ("has_contact", ctypes.c_int32),
        ("active", ctypes.c_int32),
        ("has_interior", ctypes.c_int32),
        ("encoded_state_size", ctypes.c_int32),
        ("chart_version", ctypes.c_uint32),  # former reserved slot
        ("ap_offset_u_m", ctypes.c_double),  # AffordancePose pose offset
        ("ap_offset_v_m", ctypes.c_double),
        ("ap_height_m", ctypes.c_double),
    ]


class SbScopeRecord(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("anchor_name", ctypes.c_char_p),
        ("tag_filter", ctypes.c_char_p),
        ("anchor_x", ctypes.c_double),
        ("anchor_y", ctypes.c_double),
        ("anchor_z", ctypes.c_double),
        ("anchor_yaw", ctypes.c_double),
        ("time_seconds", ctypes.c_double),
        ("binding_index", ctypes.c_int64),
        ("scope_root", ctypes.c_int64),
        ("kernel_scope_origin", ctypes.c_int64),
        ("deepest_containing", ctypes.c_int64),
        ("navigable_element", ctypes.c_int64),
        ("candidate_count", ctypes.c_int64),
        ("included_count", ctypes.c_int64),
        ("kernel_type", c_int),
        ("valid", ctypes.c_int32),
        ("anchor_active", ctypes.c_int32),
        ("explicit_scope", ctypes.c_int32),
        ("scope_live", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


class SbScopeCandidate(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("proximity", ctypes.c_double),
        ("element_id", ctypes.c_int64),
        ("shared_id", ctypes.c_int64),
        ("reason", c_int),
        ("reserved", ctypes.c_int32),
    ]


class SbRealizationPolicyParams(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("eta_0", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("epsilon_t", ctypes.c_double),
        ("eta_min", ctypes.c_double),
        ("eta_max", ctypes.c_double),
        ("progress_tau_s", ctypes.c_double),
        ("epsilon_v", ctypes.c_double),
        ("t_max", ctypes.c_double),
        ("kappa", ctypes.c_double),
        ("policy", c_int),
        ("reserved", ctypes.c_int32),
    ]


class SbRelationPhaseConfig(ctypes.Structure):
    # Mirrors sb_relation_phase_config_t (spatialblend.h; ABI 4 drops activation/near-field/radii).
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("driving_weight", ctypes.c_double),
        ("driving_release", ctypes.c_double),
        ("contact_proximity", ctypes.c_double),
        ("contact_release", ctypes.c_double),
    ]


class SbRealizationWeight(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("anchor_name", ctypes.c_char_p),
        ("sub_region_name", ctypes.c_char_p),
        ("proximity", ctypes.c_double),
        ("estimated_time", ctypes.c_double),
        ("weight", ctypes.c_double),
        ("time_seconds", ctypes.c_double),
        ("referent_shared_id", ctypes.c_int64),
        ("resolved_element_id", ctypes.c_int64),
        ("payload_index", ctypes.c_int64),
        ("channel_index", ctypes.c_int64),
        ("kernel_type", c_int),
        ("enabled", ctypes.c_int32),
        ("is_driving", ctypes.c_int32),
        ("is_realized_driving", ctypes.c_int32),
    ]


class SbRealizationChannel(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("anchor_name", ctypes.c_char_p),
        ("time_seconds", ctypes.c_double),
        ("t_min", ctypes.c_double),
        ("eta", ctypes.c_double),
        ("driving_weight", ctypes.c_double),
        ("driving_proximity", ctypes.c_double),
        ("driving_estimated_time", ctypes.c_double),
        ("realized_weight", ctypes.c_double),
        ("driving_referent_shared_id", ctypes.c_int64),
        ("driving_payload_index", ctypes.c_int64),
        ("realized_referent_shared_id", ctypes.c_int64),
        ("realized_payload_index", ctypes.c_int64),
        ("candidate_count", ctypes.c_int64),
        ("kernel_type", c_int),
        ("policy", c_int),
        ("has_driving", ctypes.c_int32),
        ("driving_tied", ctypes.c_int32),
        ("has_realized", ctypes.c_int32),
        ("realized_is_policy_argmax", ctypes.c_int32),
    ]


class SbSolverDiagnostics(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("time_seconds", ctypes.c_double),
        ("unconstrained_x", ctypes.c_double),
        ("unconstrained_y", ctypes.c_double),
        ("solved_x", ctypes.c_double),
        ("solved_y", ctypes.c_double),
        ("solved_yaw", ctypes.c_double),
        ("feasibility_snap_m", ctypes.c_double),
        ("goal_count", ctypes.c_int64),
        ("has_solve", ctypes.c_int32),
        ("has_unconstrained", ctypes.c_int32),
        ("snapped", ctypes.c_int32),
        ("status", ctypes.c_int32),
    ]


class SbSolverGoal(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("anchor_name", ctypes.c_char_p),
        ("witness_x", ctypes.c_double),
        ("witness_y", ctypes.c_double),
        ("weight", ctypes.c_double),
        ("referent_shared_id", ctypes.c_int64),
        ("payload_index", ctypes.c_int64),
        ("kernel_type", c_int),
        ("position_goal", ctypes.c_int32),
        ("yaw_goal", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


class SbRelationPayload(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("anchor_name", ctypes.c_char_p),
        ("sub_region_name", ctypes.c_char_p),
        ("proximity", ctypes.c_double),
        ("estimated_time", ctypes.c_double),
        ("support_distance_m", ctypes.c_double),
        ("source_scope_diameter_m", ctypes.c_double),
        ("contact_point_local_x", ctypes.c_double),
        ("contact_point_local_y", ctypes.c_double),
        ("contact_point_world_x", ctypes.c_double),
        ("contact_point_world_y", ctypes.c_double),
        ("contact_point_world_z", ctypes.c_double),
        ("normalized_angle", ctypes.c_double),
        ("normalized_radial_ratio", ctypes.c_double),
        ("normalized_height_ratio", ctypes.c_double),
        ("interior_u", ctypes.c_double),
        ("interior_v", ctypes.c_double),
        ("interior_w", ctypes.c_double),
        ("encoded_state", ctypes.c_double * MAX_ENCODED_STATE),
        ("referent_shared_id", ctypes.c_int64),
        ("kernel_type", c_int),
        ("normalized_strategy", c_int),
        ("encoded_state_size", ctypes.c_int32),
        ("has_contact", ctypes.c_int32),
        ("has_interior", ctypes.c_int32),
        ("enabled", ctypes.c_int32),
        # Chart version of the normalized contact (0 = pure radial).
        ("chart_version", ctypes.c_uint32),
        ("reserved", ctypes.c_int32),
        # Identities (used when the matching has_* flag is set).
        ("referent_source_space_id", ctypes.c_int64),
        ("referent_source_element_id", ctypes.c_int64),
        ("scope_shared_id", ctypes.c_int64),
        ("scope_source_space_id", ctypes.c_int64),
        ("scope_source_element_id", ctypes.c_int64),
        ("has_referent_source", ctypes.c_int32),
        ("has_scope_shared_id", ctypes.c_int32),
        ("has_scope_source", ctypes.c_int32),
        ("reserved2", ctypes.c_int32),
        ("source_binding_ordinal", ctypes.c_uint64),
        ("binding_revision", ctypes.c_uint64),
        ("has_binding_revision", ctypes.c_int32),
        ("r_formula_version", ctypes.c_int32),  # former reserved3
        ("ap_offset_u_m", ctypes.c_double),  # AffordancePose pose offset
        ("ap_offset_v_m", ctypes.c_double),
        ("ap_height_m", ctypes.c_double),
    ]


class SbRelationQuery(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("anchor_x", ctypes.c_double),
        ("anchor_y", ctypes.c_double),
        ("anchor_z", ctypes.c_double),
        ("anchor_yaw", ctypes.c_double),
        ("referent", ctypes.c_int64),
        ("scope_root", ctypes.c_int64),
        ("kernel_type", c_int),
        ("position_dimension", ctypes.c_int32),
        ("explicit_scope", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


class SbRelationEvaluation(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("sub_region_name", ctypes.c_char_p),
        ("proximity", ctypes.c_double),
        ("support_distance_m", ctypes.c_double),
        ("source_scope_diameter_m", ctypes.c_double),
        ("contact_point_world_x", ctypes.c_double),
        ("contact_point_world_y", ctypes.c_double),
        ("contact_point_world_z", ctypes.c_double),
        ("normalized_angle", ctypes.c_double),
        ("normalized_radial_ratio", ctypes.c_double),
        ("normalized_height_ratio", ctypes.c_double),
        ("interior_u", ctypes.c_double),
        ("interior_v", ctypes.c_double),
        ("interior_w", ctypes.c_double),
        ("encoded_state", ctypes.c_double * MAX_ENCODED_STATE),
        ("deepest_containing", ctypes.c_int64),
        ("navigable_element", ctypes.c_int64),
        ("normalized_strategy", c_int),
        ("encoded_state_size", ctypes.c_int32),
        ("has_contact", ctypes.c_int32),
        ("has_interior", ctypes.c_int32),
        ("ap_offset_u_m", ctypes.c_double),  # AffordancePose pose offset
        ("ap_offset_v_m", ctypes.c_double),
        ("ap_height_m", ctypes.c_double),
    ]


class SbRealizationTargetV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("anchor_name", ctypes.c_char_p),
        ("arrival_time", ctypes.c_double),
        ("position_x", ctypes.c_double),
        ("position_y", ctypes.c_double),
        ("position_z", ctypes.c_double),
        ("yaw", ctypes.c_double),
        ("weight", ctypes.c_double),
        ("requested_x", ctypes.c_double),
        ("requested_y", ctypes.c_double),
        ("reach_excess_m", ctypes.c_double),  # beyond-reach excess
        ("referent_id", ctypes.c_int64),
        ("kernel_type", c_int),
        ("has_z", ctypes.c_int32),
        ("beyond_reach", ctypes.c_int32),  # flagged, never moved
        ("reserved", ctypes.c_int32),
    ]


class SbTargetRelationEvaluation(ctypes.Structure):
    """sb_target_relation_evaluation_t (target-side Eval_p)."""
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("reason", ctypes.c_char_p),
        ("sub_region_name", ctypes.c_char_p),
        ("r_tgt", ctypes.c_double),
        ("r_src", ctypes.c_double),
        ("error", ctypes.c_double),
        ("support_distance_m", ctypes.c_double),
        ("normalizing_scope_diameter_m", ctypes.c_double),
        ("contact_point_world_x", ctypes.c_double),
        ("contact_point_world_y", ctypes.c_double),
        ("contact_point_world_z", ctypes.c_double),
        ("mapped_support_x", ctypes.c_double),
        ("mapped_support_y", ctypes.c_double),
        ("mapped_support_z", ctypes.c_double),
        ("encoded_state", ctypes.c_double * MAX_ENCODED_STATE),
        ("local_referent", ctypes.c_int64),
        ("normalizing_scope", ctypes.c_int64),
        ("disposition", c_int),
        ("encoded_state_size", ctypes.c_int32),
        ("has_contact", ctypes.c_int32),
        ("has_mapped_support", ctypes.c_int32),
        ("ap_offset_u_m", ctypes.c_double),  # AffordancePose pose offset
        ("ap_offset_v_m", ctypes.c_double),
        ("ap_height_m", ctypes.c_double),
        # 0 family grounding, 1 mapped support, 2 candidate-projection
        # fallback (degraded; `reason` says why).
        ("support_basis", c_int),
        # R formula r_src/r_tgt were both scored under (0 proximity,
        # 1 AffordancePose pose fit).
        ("r_formula_version", ctypes.c_int32),
    ]


# struct id -> ctypes class, for sb_sizeof / sb_get_struct_field_offset checks
# (ids 0-2 are checked by _types.STRUCT_LAYOUT_CHECKS' classes too).
OBS_STRUCT_IDS = (
    (SB_STRUCT_RELATION_STATE, SbRelationState),
    (SB_STRUCT_RELATION_STATE_V2, SbRelationStateV2),
    (SB_STRUCT_SCOPE_RECORD, SbScopeRecord),
    (SB_STRUCT_SCOPE_CANDIDATE, SbScopeCandidate),
    (SB_STRUCT_REALIZATION_POLICY_PARAMS, SbRealizationPolicyParams),
    (SB_STRUCT_REALIZATION_WEIGHT, SbRealizationWeight),
    (SB_STRUCT_REALIZATION_CHANNEL, SbRealizationChannel),
    (SB_STRUCT_SOLVER_DIAGNOSTICS, SbSolverDiagnostics),
    (SB_STRUCT_SOLVER_GOAL, SbSolverGoal),
    (SB_STRUCT_RELATION_PAYLOAD, SbRelationPayload),
    (SB_STRUCT_RELATION_QUERY, SbRelationQuery),
    (SB_STRUCT_RELATION_EVALUATION, SbRelationEvaluation),
    (SB_STRUCT_REALIZATION_TARGET_V2, SbRealizationTargetV2),
    (SB_STRUCT_V2_ABI_LAYOUT, SbV2AbiLayout),
    (SB_STRUCT_RELATION_PHASE_CONFIG, SbRelationPhaseConfig),
    (SB_STRUCT_TARGET_RELATION_EVALUATION, SbTargetRelationEvaluation),
)


def new(struct_cls):
    """A zeroed struct with struct_size/version set (observability structs)."""
    value = struct_cls()
    names = [f[0] for f in struct_cls._fields_]
    if names[:2] == ["struct_size", "version"]:
        value.struct_size = ctypes.sizeof(struct_cls)
        value.version = OBSERVABILITY_VERSION
    return value


# ---------------------------------------------------------------------------
# spatialblend_rich.h
# ---------------------------------------------------------------------------
RICH_FEATURE_VERSION = 1

SB_RICH_RESULT_OK = 1
SB_RICH_RESULT_NO_NEW_SNAPSHOT = 0
SB_RICH_RESULT_INVALID_ARGUMENT = -1
SB_RICH_RESULT_UNKNOWN_TOKEN = -2
SB_RICH_RESULT_BUFFER_TOO_SMALL = -3
SB_RICH_RESULT_UNSUPPORTED_VERSION = -4
SB_RICH_RESULT_UNAVAILABLE = -5
SB_RICH_RESULT_STALE_SNAPSHOT = -6

SB_RICH_LAYOUT_FEATURE = 1
SB_RICH_LAYOUT_HEADER = 2
SB_RICH_LAYOUT_CHANNEL = 3
SB_RICH_LAYOUT_CONTRIBUTOR = 4
SB_RICH_LAYOUT_SNAPSHOT = 5
SB_RICH_LAYOUT_COPY = 6


class SbRichFeature(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "descriptor_size", "feature_version", "snapshot_size", "channel_size",
        "contributor_size", "copy_size", "snapshot_channels_offset",
        "snapshot_contributors_offset", "snapshot_utf8_offset",
        "snapshot_channel_count_offset", "snapshot_contributor_count_offset",
        "snapshot_utf8_bytes_offset", "copy_channels_offset",
        "copy_contributors_offset", "copy_utf8_offset", "copy_header_offset",
        "channel_anchor_offset_offset", "channel_contributor_offset_offset",
        "channel_has_z_offset", "channel_requested_position_offset",
        "contributor_reason_offset_offset",
        "contributor_support_identifier_offset",
        "contributor_support_plane_to_world_offset", "max_channels",
        "max_contributors", "max_utf8_bytes", "coordinate_convention")] + [
        ("monotonic_clock_id", ctypes.c_uint64),
        ("monotonic_clock_epoch", ctypes.c_uint64),
    ]


class SbRichHeader(ctypes.Structure):
    _fields_ = [
        ("producer_lifetime", ctypes.c_uint64),
        ("revision", ctypes.c_uint64),
        ("monotonic_clock_id", ctypes.c_uint64),
        ("monotonic_clock_epoch", ctypes.c_uint64),
        ("receiver_ingress_monotonic_ns", ctypes.c_uint64),
        ("receiver_solve_monotonic_ns", ctypes.c_uint64),
        ("source_epoch", ctypes.c_uint64),
        ("source_sequence", ctypes.c_int64),
        ("source_observation_sequence", ctypes.c_int64),
        ("source_actor_space", ctypes.c_int64),
        ("source_actor_element", ctypes.c_int64),
        ("relay_source_space", ctypes.c_int64),
        ("receiver_space", ctypes.c_int64),
        ("receiver_actor", ctypes.c_int64),
        ("policy_revision", ctypes.c_uint64),
        ("tree_revision", ctypes.c_uint64),
        ("counterpart_revision", ctypes.c_uint64),
        ("support_revision", ctypes.c_uint64),
        ("source_time_seconds", ctypes.c_double),
        ("source_clock_seconds", ctypes.c_double),
        ("status", ctypes.c_uint32),
        ("source_timing_mode", ctypes.c_uint32),
        ("source_time_known", ctypes.c_uint32),
        ("source_clock_known", ctypes.c_uint32),
        ("source_observation_sequence_known", ctypes.c_uint32),
        ("known_revision_mask", ctypes.c_uint32),
        ("unknown_revision_mask", ctypes.c_uint32),
        ("failure_code", ctypes.c_uint32),
    ]


class SbRichChannel(ctypes.Structure):
    _fields_ = [
        ("anchor_offset", ctypes.c_uint32),
        ("anchor_length", ctypes.c_uint32),
        ("contributor_offset", ctypes.c_uint32),
        ("contributor_count", ctypes.c_uint32),
        ("family", ctypes.c_int32),
        ("solver_status", ctypes.c_int32),
        ("geometry", ctypes.c_int32),
        ("goal_kind", ctypes.c_int32),
        ("has_goal", ctypes.c_uint32),
        ("has_z", ctypes.c_uint32),
        ("has_requested_position", ctypes.c_uint32),
        ("reach_projected", ctypes.c_uint32),
        ("referent_id", ctypes.c_int64),
        ("arrival_seconds", ctypes.c_double),
        ("position_x", ctypes.c_double),
        ("position_y", ctypes.c_double),
        ("position_z", ctypes.c_double),
        ("yaw", ctypes.c_double),
        ("weight", ctypes.c_double),
        ("reach_displacement_m", ctypes.c_double),
        ("requested_position_x", ctypes.c_double),
        ("requested_position_y", ctypes.c_double),
    ]


class SbRichContributor(ctypes.Structure):
    _fields_ = [
        ("referent_id", ctypes.c_int64),
        ("source_space", ctypes.c_int64),
        ("source_element", ctypes.c_int64),
        ("resolved_local_element", ctypes.c_int64),
        ("binding_revision", ctypes.c_uint64),
        ("snapshot_ordinal", ctypes.c_uint32),
        ("disposition", ctypes.c_int32),
        ("grounding_mode", ctypes.c_int32),
        ("has_mapped_support", ctypes.c_uint32),
        ("chart_version_known", ctypes.c_uint32),
        ("binding_revision_known", ctypes.c_uint32),
        ("reason_offset", ctypes.c_uint32),
        ("reason_length", ctypes.c_uint32),
        ("support_identifier_offset", ctypes.c_uint32),
        ("support_identifier_length", ctypes.c_uint32),
        ("applied_weight", ctypes.c_double),
        ("mapped_support_x", ctypes.c_double),
        ("mapped_support_y", ctypes.c_double),
        ("mapped_support_z", ctypes.c_double),
        ("support_plane_to_world_row_major", ctypes.c_double * 16),
    ]


class SbRichSnapshot(ctypes.Structure):
    _fields_ = [
        ("header", SbRichHeader),
        ("channels", ctypes.POINTER(SbRichChannel)),
        ("contributors", ctypes.POINTER(SbRichContributor)),
        ("utf8", ctypes.POINTER(ctypes.c_uint8)),
        ("channel_count", ctypes.c_uint32),
        ("contributor_count", ctypes.c_uint32),
        ("utf8_bytes", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class SbRichCopy(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("channel_capacity", ctypes.c_uint32),
        ("contributor_capacity", ctypes.c_uint32),
        ("utf8_capacity", ctypes.c_uint32),
        ("channels", ctypes.POINTER(SbRichChannel)),
        ("contributors", ctypes.POINTER(SbRichContributor)),
        ("utf8", ctypes.POINTER(ctypes.c_uint8)),
        ("header", SbRichHeader),
        ("required_channels", ctypes.c_uint32),
        ("required_contributors", ctypes.c_uint32),
        ("required_utf8_bytes", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


RICH_LAYOUTS = (
    (SB_RICH_LAYOUT_FEATURE, SbRichFeature),
    (SB_RICH_LAYOUT_HEADER, SbRichHeader),
    (SB_RICH_LAYOUT_CHANNEL, SbRichChannel),
    (SB_RICH_LAYOUT_CONTRIBUTOR, SbRichContributor),
    (SB_RICH_LAYOUT_SNAPSHOT, SbRichSnapshot),
    (SB_RICH_LAYOUT_COPY, SbRichCopy),
)

SB_ON_RICH_REALIZATION_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(SbRichSnapshot), ctypes.c_void_p)


# ---------------------------------------------------------------------------
# spatialblend_receiver_input.h / spatialblend_receiver_reevaluation.h
# ---------------------------------------------------------------------------
RECEIVER_INPUT_FEATURE_VERSION = 1
RECEIVER_REEVALUATION_FEATURE_VERSION = 1

SB_RECEIVER_INPUT_LAYOUT_FEATURE = 1
SB_RECEIVER_INPUT_LAYOUT_ANCHOR = 2
SB_RECEIVER_INPUT_LAYOUT_FRAME = 3
SB_RECEIVER_INPUT_LAYOUT_STAMP = 4
SB_RECEIVER_INPUT_LAYOUT_VALIDITY = 5


class SbReceiverAnchorInput(ctypes.Structure):
    _fields_ = [
        ("name_utf8", ctypes.POINTER(ctypes.c_uint8)),
        ("name_bytes", ctypes.c_uint32),
        ("available", ctypes.c_uint32),
        ("world_pose_row_major", ctypes.c_double * 16),
    ]


class SbReceiverInputFrame(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("feature_version", ctypes.c_uint32),
        ("anchor_count", ctypes.c_uint32),
        ("anchor_stride", ctypes.c_uint32),
        ("epoch", ctypes.c_uint64),
        ("sequence", ctypes.c_uint64),
        ("monotonic_clock_id", ctypes.c_uint64),
        ("monotonic_clock_epoch", ctypes.c_uint64),
        ("observed_at_ns", ctypes.c_int64),
        ("valid_until_ns", ctypes.c_int64),
        ("anchors", ctypes.POINTER(SbReceiverAnchorInput)),
    ]


class SbReceiverInputStamp(ctypes.Structure):
    _fields_ = [
        ("epoch", ctypes.c_uint64),
        ("revision", ctypes.c_uint64),
        ("sequence", ctypes.c_uint64),
        ("rig_revision", ctypes.c_uint64),
        ("calibration_revision", ctypes.c_uint64),
        ("observed_at_ns", ctypes.c_int64),
        ("valid_until_ns", ctypes.c_int64),
        ("known", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class SbReceiverInputValidity(ctypes.Structure):
    _fields_ = [
        ("producer_lifetime", ctypes.c_uint64),
        ("revision", ctypes.c_uint64),
        ("captured", SbReceiverInputStamp),
        ("current", SbReceiverInputStamp),
        ("changed", ctypes.c_uint32),
        ("valid", ctypes.c_uint32),
    ]


class SbReceiverInputFeature(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "descriptor_size", "feature_version", "anchor_size", "frame_size",
        "stamp_size", "validity_size", "max_anchors", "max_name_bytes",
        "coordinate_convention", "reserved")] + [
        ("max_age_ns", ctypes.c_uint64),
        ("monotonic_clock_id", ctypes.c_uint64),
        ("monotonic_clock_epoch", ctypes.c_uint64),
    ]


class SbReceiverReevaluationFeature(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "descriptor_size", "feature_version",
        "required_receiver_input_version", "required_rich_version",
        "guarantees", "reserved")]


RECEIVER_INPUT_LAYOUTS = (
    (SB_RECEIVER_INPUT_LAYOUT_FEATURE, SbReceiverInputFeature),
    (SB_RECEIVER_INPUT_LAYOUT_ANCHOR, SbReceiverAnchorInput),
    (SB_RECEIVER_INPUT_LAYOUT_FRAME, SbReceiverInputFrame),
    (SB_RECEIVER_INPUT_LAYOUT_STAMP, SbReceiverInputStamp),
    (SB_RECEIVER_INPUT_LAYOUT_VALIDITY, SbReceiverInputValidity),
)

# sb_get_receiver_reevaluation_field_offset kind (header define).
SB_RECEIVER_REEVALUATION_LAYOUT_FEATURE = 1
