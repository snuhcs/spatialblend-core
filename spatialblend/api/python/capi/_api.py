# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""ctypes argtypes/restype bindings for every sb_* function in
spatialblend/api/c/spatialblend.h (the C API).

This module binds the raw C surface 1:1 -- no coordinate conversion, no
Pythonic ergonomics. See api.py for a friendlier layer built on top of this
one. Callback ctypes.CFUNCTYPE objects must be kept alive by the caller for
as long as they may be invoked (ctypes does not hold a reference for you);
api.py keeps them alive on the owning Session object.
"""
import ctypes

from . import _lib
from . import _obs_types as _o
from ._types import (
    SbActorHandle,
    SbActorState,
    SbAsyncHandle,
    SbColliderType,
    SbElementHandle,
    SbGeometryRole,
    SbKernelType,
    SbActorEvent,
    SbAnchorPose,
    SbBonePose,
    SbNamedBonePose,
    SbChainJoint,
    SbAnchorProposal,
    SbAnchorSolve,
    SbAnchorTurn,
    SbAnchorChart,
    SbProposalSet,
    SbRealizationTarget,
    SbRelationEvent,
    SbSessionHandle,
    SbSpaceHandle,
    STRUCT_LAYOUT_CHECKS,
)

c_double_p = ctypes.POINTER(ctypes.c_double)
c_int32_p = ctypes.POINTER(ctypes.c_int32)
_u8, _u32, _u64 = ctypes.c_uint8, ctypes.c_uint32, ctypes.c_uint64
_i32, _i64, _d = ctypes.c_int32, ctypes.c_int64, ctypes.c_double
_P = ctypes.POINTER

# ---------------------------------------------------------------------------
# Callback typedefs (spatialblend.h:497, :531, :559)
# ---------------------------------------------------------------------------

# typedef void (*sb_on_targets_t)(sb_space_t space, sb_actor_t actor,
#                                 const sb_realization_target_t* targets,
#                                 int64_t num_targets, void* user_data);
SB_ON_TARGETS_FUNC = ctypes.CFUNCTYPE(
    None, SbSpaceHandle, SbActorHandle, ctypes.POINTER(SbRealizationTarget),
    ctypes.c_int64, ctypes.c_void_p)

# typedef void (*sb_on_element_added_t)(sb_space_t space,
#                                       sb_element_t local_element,
#                                       int64_t shared_id, const char* tag,
#                                       const void* user_data,
#                                       int64_t user_data_size,
#                                       void* cb_user_data);
SB_ON_ELEMENT_ADDED_FUNC = ctypes.CFUNCTYPE(
    None, SbSpaceHandle, SbElementHandle, ctypes.c_int64, ctypes.c_char_p,
    ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p)

# typedef void (*sb_on_relation_event_t)(const sb_relation_event_t* event,
#                                        void* user_data);
SB_ON_RELATION_EVENT_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(SbRelationEvent), ctypes.c_void_p)

# typedef void (*sb_on_element_removed_t)(sb_space_t space,
#                                         sb_element_t local_element,
#                                         void* user_data);
SB_ON_ELEMENT_REMOVED_FUNC = ctypes.CFUNCTYPE(
    None, SbSpaceHandle, SbElementHandle, ctypes.c_void_p)

# typedef void (*sb_on_link_removed_t)(sb_space_t space, sb_element_t local_element,
#                                      int64_t peer_space, int32_t reason,
#                                      void* user_data);
SB_ON_LINK_REMOVED_FUNC = ctypes.CFUNCTYPE(
    None, SbSpaceHandle, SbElementHandle, ctypes.c_int64, ctypes.c_int32,
    ctypes.c_void_p)

SB_LINK_REMOVED_PEER_UNSHARED = 1
SB_LINK_REMOVED_PEER_UNLINKED = 2

SB_ON_RICH_REALIZATION_FUNC = _o.SB_ON_RICH_REALIZATION_FUNC


# (name, restype, argtypes) for every SB_EXPORT function in spatialblend.h,
# in the same order as the header.
_SIGNATURES = [
    # Session management
    ("sb_create_session", SbSessionHandle, []),
    ("sb_destroy_session", None, [SbSessionHandle]),
    # Space management
    ("sb_add_space", SbSpaceHandle, [SbSessionHandle]),
    ("sb_host_space", SbAsyncHandle, [SbSpaceHandle, ctypes.c_int32]),
    ("sb_connect_space", SbAsyncHandle,
     [SbSpaceHandle, ctypes.c_char_p, ctypes.c_int32]),
    # Async handle utilities
    ("sb_try_wait", ctypes.c_int32, [SbAsyncHandle, ctypes.c_int32]),
    ("sb_get_error", ctypes.c_char_p, [SbAsyncHandle]),
    # Element management
    ("sb_add_element", SbElementHandle, [SbSpaceHandle, SbElementHandle]),
    ("sb_set_element_transform", ctypes.c_int32,
     [SbSpaceHandle, SbElementHandle, c_double_p]),
    ("sb_get_scene_status", ctypes.c_int32,
     [SbSpaceHandle, ctypes.c_char_p, ctypes.c_int32]),
    ("sb_set_element_shared_id", None,
     [SbSpaceHandle, SbElementHandle, ctypes.c_int64]),
    ("sb_set_element_tag", None,
     [SbSpaceHandle, SbElementHandle, ctypes.c_char_p]),
    ("sb_set_element_affordance_pose", ctypes.c_int32,  # ABI 4.11
     [SbSpaceHandle, SbElementHandle, ctypes.c_char_p, ctypes.c_int32,
      ctypes.c_char_p]),
    ("sb_set_element_polygon", ctypes.c_int32,
     [SbSpaceHandle, SbElementHandle, SbGeometryRole, c_double_p,
      ctypes.c_int64]),
    ("sb_set_element_box", ctypes.c_int32,
     [SbSpaceHandle, SbElementHandle, SbGeometryRole,
      ctypes.c_double, ctypes.c_double, ctypes.c_double,
      ctypes.c_double, ctypes.c_double, ctypes.c_double]),
    ("sb_set_element_collider", ctypes.c_int32,
     [SbSpaceHandle, SbElementHandle, SbColliderType, c_double_p,
      ctypes.c_int64]),
    ("sb_set_element_surface", ctypes.c_int32,
     [SbSpaceHandle, SbElementHandle, c_double_p, ctypes.c_int32,
      SbGeometryRole]),
    ("sb_set_element_polytope", None,
     [SbSpaceHandle, SbElementHandle, c_double_p, ctypes.c_int32,
      c_int32_p, c_int32_p, ctypes.c_int32, SbGeometryRole]),
    ("sb_add_sub_region", None,
     [SbSpaceHandle, SbElementHandle, ctypes.c_char_p, ctypes.c_char_p,
      c_double_p, ctypes.c_int32, c_double_p]),
    ("sb_find_element_by_shared_id", SbElementHandle,
     [SbSpaceHandle, ctypes.c_int64]),
    ("sb_add_element_anchor", None,
     [SbSpaceHandle, SbElementHandle, ctypes.c_char_p, c_double_p]),
    ("sb_share_element", None,
     [SbSpaceHandle, SbElementHandle, ctypes.c_void_p, ctypes.c_int64]),
    ("sb_unshare_element", None, [SbSpaceHandle, SbElementHandle]),
    # Actor management
    ("sb_create_actor", SbActorHandle, [SbSpaceHandle]),
    ("sb_add_anchor", None,
     [SbSpaceHandle, SbActorHandle, ctypes.c_char_p, ctypes.c_double,
      ctypes.c_double, ctypes.c_double, ctypes.c_char_p]),
    ("sb_add_standard_anchors", None, [SbSpaceHandle, SbActorHandle]),
    ("sb_set_actor_element", None,
     [SbSpaceHandle, SbActorHandle, SbElementHandle]),
    ("sb_find_actor_by_shared_id", SbActorHandle,
     [SbSpaceHandle, ctypes.c_int64]),
    ("sb_update_actor_state", SbAsyncHandle,
     [SbSpaceHandle, SbActorHandle, ctypes.c_double, ctypes.c_double,
      ctypes.c_double]),
    ("sb_update_actor_pose", SbAsyncHandle,
     [SbSpaceHandle, SbActorHandle, ctypes.c_double, ctypes.c_double,
      ctypes.c_void_p]),
    ("sb_get_actor_state", SbActorState, [SbSpaceHandle, SbActorHandle]),
    ("sb_get_anchor_id", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_char_p,
      ctypes.POINTER(ctypes.c_uint32)]),
    ("sb_get_anchor_name", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_uint32, ctypes.c_char_p,
      ctypes.c_int32]),
    ("sb_publish_anchor_pose", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_uint32,
      ctypes.POINTER(SbAnchorPose), ctypes.c_double, ctypes.c_uint64]),
    ("sb_register_embodiment_json", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_char_p, ctypes.c_int32,
      ctypes.c_char_p, ctypes.c_int32]),
    ("sb_get_embodiment_json", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_char_p, ctypes.c_int32]),
    ("sb_get_bone_id", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_char_p]),
    ("sb_publish_bone_pose", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_int32,
      ctypes.POINTER(SbBonePose), ctypes.c_double, ctypes.c_uint64]),
    ("sb_apply_actor_bone_frame", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_uint64, ctypes.c_int64,
      ctypes.c_double, ctypes.POINTER(SbNamedBonePose), ctypes.c_int32,
      ctypes.c_void_p, ctypes.c_int32]),
    ("sb_embodiment_anchor_pose_from_bone", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_char_p,
      ctypes.POINTER(SbBonePose), ctypes.POINTER(SbBonePose)]),
    ("sb_embodiment_bone_pose_from_anchor", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_char_p,
      ctypes.POINTER(SbBonePose), ctypes.POINTER(SbBonePose)]),
    ("sb_get_chain_joint_count", ctypes.c_int32, [SbSpaceHandle, SbActorHandle]),
    ("sb_get_chain_joint", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_int32,
      ctypes.POINTER(SbChainJoint), ctypes.c_char_p, ctypes.c_int32]),
    ("sb_convert_pose_basis", ctypes.c_int32,
     [ctypes.POINTER(ctypes.c_double), ctypes.c_int32,
      ctypes.POINTER(SbBonePose), ctypes.POINTER(SbBonePose)]),
    ("sb_read_proposals", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.POINTER(SbProposalSet),
      ctypes.POINTER(SbAnchorProposal), ctypes.c_int32]),
    ("sb_read_anchor_solve", ctypes.c_int32,  # ABI 4.11
     [SbSpaceHandle, SbActorHandle, ctypes.c_uint32,
      ctypes.POINTER(SbAnchorSolve)]),
    ("sb_read_anchor_turn", ctypes.c_int32,  # ABI 4.12
     [SbSpaceHandle, SbActorHandle, ctypes.c_uint32,
      ctypes.POINTER(SbAnchorTurn)]),
    ("sb_read_anchor_chart", ctypes.c_int32,  # ABI 4.14
     [SbSpaceHandle, SbActorHandle, ctypes.c_uint32,
      ctypes.POINTER(SbAnchorChart)]),
    ("sb_raise_source_exception", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_uint32, ctypes.c_void_p,
      ctypes.c_uint32]),
    ("sb_set_target_event_delivery", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_int32]),
    ("sb_poll_actor_event", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.POINTER(SbActorEvent)]),
    ("sb_query_nearest_clear_point", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]),
    ("sb_query_course", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_double, ctypes.c_double,
      ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double),
      ctypes.c_int32]),
    ("sb_set_body_radius", ctypes.c_int32, [SbSpaceHandle, ctypes.c_double]),
    # Binding
    ("sb_bind", _i32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_char_p, SbKernelType]),
    ("sb_bind_affordance", _i32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_char_p, ctypes.c_char_p]),
    # Realization target callback registration / dispatch
    ("sb_register_on_targets", None,
     [SbSpaceHandle, SB_ON_TARGETS_FUNC, ctypes.c_void_p]),
    ("sb_register_on_element_added", None,
     [SbSpaceHandle, SB_ON_ELEMENT_ADDED_FUNC, ctypes.c_void_p]),
    ("sb_register_on_relation_event", None,
     [SbSpaceHandle, SB_ON_RELATION_EVENT_FUNC, ctypes.c_void_p]),
    ("sb_poll_realization_targets", ctypes.c_int32,
     [SbSpaceHandle, SbActorHandle]),
    # Update interval
    ("sb_set_update_interval", None, [SbSpaceHandle, ctypes.c_double]),
    # Global cleanup
    ("sb_cleanup_all", None, []),
    # Shared-element resolution
    ("sb_resolve_shared_element", None,
     [SbSpaceHandle, SbElementHandle, SbElementHandle]),
]

# Everything below was hand-written in the eval harness before: the
# runtime config, tracing, ABI descriptors, checked anchors/bindings,
# observation frames, support correspondences, element/actor lifecycle.
_SIGNATURES += [
    ("sb_destroy_actor", _i32, [SbSpaceHandle, SbActorHandle]),
    ("sb_remove_local_element", _i32, [SbSpaceHandle, SbElementHandle]),
    ("sb_reparent_element", _i32,
     [SbSpaceHandle, SbElementHandle, SbElementHandle, ctypes.POINTER(_d)]),
    ("sb_add_anchor_with_position_dimension", _i32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_char_p, _d, _d, _d,
      ctypes.c_char_p, _i32]),
    ("sb_set_anchor_transform", None,
     [SbSpaceHandle, SbActorHandle, ctypes.c_char_p, _d, _d, _d, _d]),
    ("sb_clear_anchor_transform", None,
     [SbSpaceHandle, SbActorHandle, ctypes.c_char_p]),
    ("sb_set_anchor_active", _i32,
     [SbSpaceHandle, SbActorHandle, ctypes.c_char_p, _i32]),
    ("sb_register_on_element_removed", None,
     [SbSpaceHandle, SB_ON_ELEMENT_REMOVED_FUNC, ctypes.c_void_p]),
    ("sb_unlink", _i32, [SbSpaceHandle, SbElementHandle]),
    ("sb_register_on_link_removed", None,
     [SbSpaceHandle, SB_ON_LINK_REMOVED_FUNC, ctypes.c_void_p]),
    ("sb_get_relation_states", _i32,
     [SbSpaceHandle, SbActorHandle, _P(_o.SbRelationState), _i32]),
    ("sb_enable_actor_observation_frames", _i32, [SbSessionHandle]),
    ("sb_begin_actor_observation_epoch", _u64,
     [SbSpaceHandle, SbActorHandle, ctypes.c_char_p]),
    ("sb_apply_actor_observation_frame", _i32,
     [SbSpaceHandle, SbActorHandle, _u64, _i64, _d,
      _P(_o.SbActorObservation), _i32]),
    ("sb_clear_actor_observation_anchor", _i32,
     [SbSpaceHandle, SbActorHandle, _u64, ctypes.c_char_p]),
    ("sb_get_actor_observation_clock", _i32,
     [SbSpaceHandle, SbActorHandle, _P(_u64), _P(_i64), _P(_d), _P(_i32)]),
    ("sb_get_last_emitted_update_sequence", _i64,
     [SbSpaceHandle, SbActorHandle]),
    ("sb_get_last_applied_remote_sequence", _i64,
     [SbSpaceHandle, SbSpaceHandle, SbActorHandle]),
    ("sb_register_support_correspondences", _i64,
     [SbSessionHandle, _P(_o.SbSupportCorrespondence), _i64]),
    ("sb_dispose_support_correspondences", None, [_i64]),
    ("sb_set_referent_resolve_mode", None, [SbSpaceHandle, ctypes.c_int]),
    ("sb_set_weight_smoothing", None, [SbSpaceHandle, _d]),
    ("sb_set_min_move", None, [SbSpaceHandle, _d]),
    ("sb_set_min_yaw", None, [SbSpaceHandle, _d]),
    ("sb_set_realization_policy", None, [SbSpaceHandle, ctypes.c_int]),
    ("sb_set_anchor_reach", None, [SbSpaceHandle, ctypes.c_char_p, _d]),
    ("sb_set_trace", None, [SbSpaceHandle, ctypes.c_char_p]),
    ("sb_abi_version", _u32, []),
    ("sb_abi_minor_version", _u32, []),
    ("sb_get_v2_abi_layout", _u32, [_P(_o.SbV2AbiLayout), _u32]),
    ("sb_sizeof", ctypes.c_size_t, [ctypes.c_int]),
    # Source evidence (raw; struct layouts negotiated by field offset).
    ("sb_get_source_evidence_feature", _u32, [ctypes.c_void_p, _u32]),
    ("sb_get_source_evidence_field_count", _u32, [_u32]),
    ("sb_get_source_evidence_field_offset", _u32, [_u32, _u32]),
    ("sb_try_enable_source_evidence", _i32,
     [SbSessionHandle, SbSpaceHandle, SbActorHandle, _u32, _u64, _u32]),
    ("sb_copy_source_evidence", _i32,
     [SbSessionHandle, SbSpaceHandle, SbActorHandle, _u32, _u64, _i64, _i64,
      ctypes.c_void_p, _u32]),
]

# spatialblend_observability.h.
_SIGNATURES += [
    ("sb_set_observability_capture", _i32, [SbSpaceHandle, _i32]),
    ("sb_get_observability_version", _u32, []),
    ("sb_get_trace_format_version", _u32, []),
    ("sb_get_struct_field_count", _u32, [ctypes.c_int]),
    ("sb_get_struct_field_offset", _u32, [ctypes.c_int, _u32]),
    ("sb_get_relation_states_v2", _i32,
     [SbSpaceHandle, SbActorHandle, _P(_o.SbRelationStateV2), _i32, _u32]),
    ("sb_get_scope_records", _i32,
     [SbSpaceHandle, SbActorHandle, _P(_o.SbScopeRecord), _i32, _u32]),
    ("sb_get_scope_candidates", _i32,
     [SbSpaceHandle, SbActorHandle, _i32, _P(_o.SbScopeCandidate), _i32,
      _u32]),
    ("sb_get_default_realization_policy_params", _i32,
     [_P(_o.SbRealizationPolicyParams), _u32]),
    ("sb_get_realization_policy_params", _i32,
     [SbSpaceHandle, _P(_o.SbRealizationPolicyParams), _u32]),
    ("sb_set_realization_policy_params", _i32,
     [SbSpaceHandle, _P(_o.SbRealizationPolicyParams), _u32]),
    ("sb_set_relation_phase_config", _i32,
     [SbSpaceHandle, SbKernelType, _P(_o.SbRelationPhaseConfig),
      _P(_o.SbRelationPhaseConfig)]),
    ("sb_get_realization_weights", _i32,
     [SbSpaceHandle, SbActorHandle, _P(_o.SbRealizationWeight), _i32, _u32]),
    ("sb_get_realization_channels", _i32,
     [SbSpaceHandle, SbActorHandle, _P(_o.SbRealizationChannel), _i32, _u32]),
    ("sb_get_solver_diagnostics", _i32,
     [SbSpaceHandle, SbActorHandle, _P(_o.SbSolverDiagnostics), _u32]),
    ("sb_get_solver_goals", _i32,
     [SbSpaceHandle, SbActorHandle, _P(_o.SbSolverGoal), _i32, _u32]),
    ("sb_get_realization_targets_v2", _i32,
     [SbSpaceHandle, SbActorHandle, _P(_o.SbRealizationTargetV2), _i32,
      _u32]),
    ("sb_inject_relation_payloads", _i32,
     [SbSpaceHandle, SbActorHandle, _P(_o.SbRelationPayload), _i32, _u32,
      _d]),
    ("sb_evaluate_relation", _i32,
     [SbSpaceHandle, _P(_o.SbRelationQuery), _P(_o.SbRelationEvaluation)]),
    ("sb_evaluate_target_relation", _i32,
     [SbSpaceHandle, _P(_o.SbRelationPayload), _u32, _P(_o.SbRelationQuery),
      _P(_o.SbTargetRelationEvaluation)]),
    ("sb_get_element_world_transform", _i32,
     [SbSpaceHandle, SbElementHandle, c_double_p]),
    ("sb_get_element_parent", _i32,
     [SbSpaceHandle, SbElementHandle, _P(SbElementHandle)]),
]

# spatialblend_rich.h, spatialblend_receiver_input.h,
# spatialblend_receiver_reevaluation.h (rich realization).
_SIGNATURES += [
    ("sb_get_rich_realization_feature", _u32, [_P(_o.SbRichFeature), _u32]),
    ("sb_get_rich_realization_field_count", _u32, [_u32]),
    ("sb_get_rich_realization_field_offset", _u32, [_u32, _u32]),
    ("sb_activate_rich_realization", _u64, [SbSpaceHandle, SbActorHandle, _u32]),
    ("sb_retire_rich_realization", None, [_u64]),
    ("sb_release_rich_realization", _i32, [_u64]),
    ("sb_register_on_rich_realization", None,
     [SbSpaceHandle, SB_ON_RICH_REALIZATION_FUNC, ctypes.c_void_p]),
    ("sb_poll_rich_realization", _i32, [_u64, _P(_o.SbRichCopy)]),
    ("sb_get_rich_realization_validity", _i32,
     [_u64, _u64, _u32, ctypes.c_void_p, _u32]),
    ("sb_read_rich_monotonic_ns", _u64, []),
    ("sb_read_rich_monotonic_seconds", _d, []),
    ("sb_get_receiver_input_feature", _u32,
     [_P(_o.SbReceiverInputFeature), _u32]),
    ("sb_get_receiver_input_field_count", _u32, [_u32]),
    ("sb_get_receiver_input_field_offset", _u32, [_u32, _u32]),
    ("sb_begin_receiver_input", _i32,
     [SbSpaceHandle, SbActorHandle, _u32, _P(_u8), _u32, _u64, _u64,
      _P(_u64)]),
    ("sb_submit_receiver_input", _i32,
     [SbSpaceHandle, SbActorHandle, _P(_o.SbReceiverInputFrame), _u32]),
    ("sb_clear_receiver_input", _i32, [SbSpaceHandle, SbActorHandle, _u64]),
    ("sb_get_receiver_input_validity", _i32,
     [_u64, _u64, _P(_o.SbReceiverInputValidity), _u32]),
    ("sb_get_receiver_reevaluation_feature", _u32,
     [_P(_o.SbReceiverReevaluationFeature), _u32]),
    ("sb_get_receiver_reevaluation_field_count", _u32, [_u32]),
    ("sb_get_receiver_reevaluation_field_offset", _u32, [_u32, _u32]),
    ("sb_reevaluate_receiver_realization", _i32, [_u64, _u32, _u64, _u64]),
]


# The C ABI generation these bindings mirror. Major 4 is the per-anchor port and
# the one-proposal-per-relation-bearing-anchor contract.
EXPECTED_ABI_VERSION = 4
# The check below is strict (!=): a 4.5 native library is rejected by these bindings, as every earlier minor was.
# Kept on purpose (the bindings mirror one ABI); use the matching library.
EXPECTED_ABI_MINOR = 14


class NativeApi(object):
    """Binds every sb_* symbol from a loaded CDLL onto attributes of self.

    Every symbol in _SIGNATURES is required: a missing one raises
    AttributeError from getattr(lib, name) at construction time, surfacing a
    clear error (e.g. a stale libspatialblend) instead of a confusing
    NameError deep in some call site.
    """

    def __init__(self, lib):
        self.lib = lib
        for name, restype, argtypes in _SIGNATURES:
            fn = getattr(lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
            setattr(self, name, fn)
        abi = int(self.sb_abi_version())
        minor = int(self.sb_abi_minor_version())
        if abi != EXPECTED_ABI_VERSION or minor != EXPECTED_ABI_MINOR:
            raise RuntimeError(
                "libspatialblend reports ABI %d.%d, these bindings are ABI %d.%d "
                "(a stale or newer library)" % (
                    abi, minor, EXPECTED_ABI_VERSION, EXPECTED_ABI_MINOR))
        check_struct_layout(lib)


_native_singleton = None


def get_native():
    """Return the process-wide NativeApi singleton, loading the lib lazily."""
    global _native_singleton
    if _native_singleton is None:
        _native_singleton = NativeApi(_lib.get_lib())
    return _native_singleton


def _layout_mismatches(struct_cls, native_size, offset_of, label):
    """Compare a ctypes Structure to native size + per-field offsets."""
    mismatches = []
    if native_size != ctypes.sizeof(struct_cls):
        mismatches.append("%s: native size %d, ctypes %s %d" % (
            label, native_size, struct_cls.__name__, ctypes.sizeof(struct_cls)))
    for index, (name, _) in enumerate(struct_cls._fields_):
        native = offset_of(index)
        local = getattr(struct_cls, name).offset
        if native != local:
            mismatches.append("%s.%s: native offset %d, ctypes %d" % (
                label, name, native, local))
    if offset_of(len(struct_cls._fields_)) != 0xFFFFFFFF:
        mismatches.append("%s: native struct has more fields than %s" % (
            label, struct_cls.__name__))
    return mismatches


def iter_native_layouts():
    """Yield (label, ctypes class) for every mirror check_struct_layout() checks.

    Sizes come from sb_sizeof or the feature descriptor that owns the struct.
    Offsets come from that struct's field-offset table.
    """
    from . import _types as _t
    for struct_id, (_, struct_cls) in enumerate(STRUCT_LAYOUT_CHECKS):
        yield "sb_struct_id %d" % struct_id, struct_cls
    for struct_id, cls_name in _t.EXTRA_SIZE_CHECKS:
        yield "sb_struct_id %d" % struct_id, getattr(_t, cls_name)
    for struct_id, struct_cls in _o.OBS_STRUCT_IDS:
        yield "sb_struct_id %d" % struct_id, struct_cls
    for kind, struct_cls in _o.RICH_LAYOUTS:
        yield "rich layout %d" % kind, struct_cls
    for kind, struct_cls in _o.RECEIVER_INPUT_LAYOUTS:
        yield "receiver-input layout %d" % kind, struct_cls
    yield "receiver-reevaluation feature", _o.SbReceiverReevaluationFeature


def check_struct_layout(lib):
    """Fail fast when a ctypes mirror disagrees with the loaded library.

    Every struct in iter_native_layouts() is checked, size and field by field,
    against sb_sizeof / sb_get_struct_field_offset or the feature-specific
    offset table that describes it. Raises RuntimeError listing every mismatch.
    """
    from . import _types as _t
    mismatches = []
    checked = set()

    def check(struct_cls, native_size, offset_of, label):
        checked.add(struct_cls)
        mismatches.extend(
            _layout_mismatches(struct_cls, native_size, offset_of, label))

    for struct_id, (_, struct_cls) in enumerate(STRUCT_LAYOUT_CHECKS):
        check(struct_cls, lib.sb_sizeof(struct_id),
              lambda i, sid=struct_id: lib.sb_get_struct_field_offset(sid, i),
              "sb_struct_id %d" % struct_id)
    for struct_id, cls_name in _t.EXTRA_SIZE_CHECKS:
        check(getattr(_t, cls_name), lib.sb_sizeof(struct_id),
              lambda i, sid=struct_id: lib.sb_get_struct_field_offset(sid, i),
              "sb_struct_id %d" % struct_id)
    for struct_id, struct_cls in _o.OBS_STRUCT_IDS:
        check(struct_cls, lib.sb_sizeof(struct_id),
              lambda i, sid=struct_id: lib.sb_get_struct_field_offset(sid, i),
              "sb_struct_id %d" % struct_id)
    feature = _o.SbRichFeature()
    lib.sb_get_rich_realization_feature(ctypes.byref(feature),
                                        ctypes.sizeof(feature))
    rich_sizes = {
        _o.SB_RICH_LAYOUT_FEATURE: feature.descriptor_size,
        _o.SB_RICH_LAYOUT_CHANNEL: feature.channel_size,
        _o.SB_RICH_LAYOUT_CONTRIBUTOR: feature.contributor_size,
        _o.SB_RICH_LAYOUT_SNAPSHOT: feature.snapshot_size,
        _o.SB_RICH_LAYOUT_COPY: feature.copy_size,
    }
    for kind, struct_cls in _o.RICH_LAYOUTS:
        size = rich_sizes.get(kind, ctypes.sizeof(struct_cls))
        check(struct_cls, size,
              lambda i, k=kind: lib.sb_get_rich_realization_field_offset(k, i),
              "rich layout %d" % kind)
    ri = _o.SbReceiverInputFeature()
    lib.sb_get_receiver_input_feature(ctypes.byref(ri), ctypes.sizeof(ri))
    ri_sizes = {
        _o.SB_RECEIVER_INPUT_LAYOUT_FEATURE: ri.descriptor_size,
        _o.SB_RECEIVER_INPUT_LAYOUT_ANCHOR: ri.anchor_size,
        _o.SB_RECEIVER_INPUT_LAYOUT_FRAME: ri.frame_size,
        _o.SB_RECEIVER_INPUT_LAYOUT_STAMP: ri.stamp_size,
        _o.SB_RECEIVER_INPUT_LAYOUT_VALIDITY: ri.validity_size,
    }
    for kind, struct_cls in _o.RECEIVER_INPUT_LAYOUTS:
        check(struct_cls, ri_sizes[kind],
              lambda i, k=kind: lib.sb_get_receiver_input_field_offset(k, i),
              "receiver-input layout %d" % kind)
    reeval = _o.SbReceiverReevaluationFeature()
    reeval_size = lib.sb_get_receiver_reevaluation_feature(
        ctypes.byref(reeval), ctypes.sizeof(reeval))
    check(_o.SbReceiverReevaluationFeature, reeval_size,
          lambda i: lib.sb_get_receiver_reevaluation_field_offset(
              _o.SB_RECEIVER_REEVALUATION_LAYOUT_FEATURE, i),
          "receiver-reevaluation feature")
    if mismatches:
        raise RuntimeError(
            "spatialblend_capi: ctypes struct layout does not match "
            "libspatialblend. Update capi/_types.py / capi/_obs_types.py "
            "from the current headers. Mismatches:\n  " +
            "\n  ".join(mismatches))
    return checked
