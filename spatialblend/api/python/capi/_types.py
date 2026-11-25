# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""ctypes mirrors of the structs in spatialblend/api/c/spatialblend_type.h.

Every ``Structure`` below documents the exact header lines it mirrors (path
relative to the spatialblend-core repo root) so a future header change can be
diffed against this file directly. Field ORDER and TYPES must match the C
struct exactly -- ctypes uses each platform's native struct alignment rules
(same as the C compiler that built libspatialblend), so as long as field
order/types agree there is no manual padding to get right, except where noted.

This module intentionally has zero dependencies beyond the standard library
`ctypes` module so the package stays importable on any CPython (3.6+) without
a matching ABI tag -- there is no C extension here, only pure ctypes.
"""
import ctypes

# All *_t handle aliases in spatialblend_type.h (lines 10-14) are int64_t.
SbSessionHandle = ctypes.c_int64
SbSpaceHandle = ctypes.c_int64
SbActorHandle = ctypes.c_int64
SbElementHandle = ctypes.c_int64
SbAsyncHandle = ctypes.c_int64

# Plain C enums (no explicit underlying type) are `int` per the platform ABI
# on every target this library ships for (Linux/macOS/Windows, no
# `enum class`/fixed-underlying-type usage in the header) -> c_int (4 bytes).
SbGeometryRole = ctypes.c_int
SbKernelType = ctypes.c_int
SbColliderType = ctypes.c_int
SbRelationPhase = ctypes.c_int

# ---------------------------------------------------------------------------
# Enum value mirrors (spatialblend_type.h)
# ---------------------------------------------------------------------------

# sb_geometry_role_t, spatialblend_type.h:16-23
SB_GEOMETRY_UNSPECIFIED = 0
SB_GEOMETRY_NAVIGABLE = 1
SB_GEOMETRY_CONTAINMENT = 2
SB_GEOMETRY_OBSTACLE = 3
SB_GEOMETRY_SURFACE = 4
SB_GEOMETRY_INTERACTION_REGION = 5

# sb_kernel_type_t, spatialblend_type.h:25-29
SB_KERNEL_SPACING = 0
SB_KERNEL_FACING = 1
SB_KERNEL_AFFORDANCE_POSE = 2

# sb_collider_type_t, spatialblend_type.h:37-41
SB_COLLIDER_BOX = 0
SB_COLLIDER_SPHERE = 1
SB_COLLIDER_CAPSULE = 2

# sb_relation_phase_t, spatialblend_type.h:118-123
SB_PHASE_DORMANT = 0
SB_PHASE_ACTIVE = 1
SB_PHASE_DRIVING = 2
SB_PHASE_CONTACT = 3


# sb_proposal_status_t / SB_PROPOSAL_CAUSE_*, spatialblend_type.h (ABI 4)
SB_PROPOSAL_NONE = 0
SB_PROPOSAL_NO_SAMPLE = 1
SB_PROPOSAL_EXACT = 2
SB_PROPOSAL_INEXACT = 3
SB_PROPOSAL_NOT_CONVERGED = 5
SB_PROPOSAL_CAUSE_FIRST_PROPOSAL = 0x001
SB_PROPOSAL_CAUSE_WEIGHT_SWITCH = 0x002
SB_PROPOSAL_CAUSE_REFERENT_SET_CHANGE = 0x004
SB_PROPOSAL_CAUSE_SOURCE_EXCEPTION = 0x010
SB_PROPOSAL_CAUSE_CONTAIN_ENTER = 0x020
SB_PROPOSAL_CAUSE_CONTAIN_EXIT = 0x040
SB_PROPOSAL_CAUSE_SEAT_ENGAGE = 0x080
SB_PROPOSAL_CAUSE_SEAT_RELEASE = 0x100
SB_PROPOSAL_CAUSE_EMBODIED_RESET = 0x200
SB_PROPOSAL_CAUSE_FACING_ROOT_SWITCH = 0x400
SB_PROPOSAL_CAUSE_STATUS_CHANGE = 0x800
SB_PROPOSAL_CAUSE_SCENE_CHANGE = 0x1000
SB_PROPOSAL_CAUSE_CLAMP_BRANCH = 0x2000
SB_PROPOSAL_CAUSE_DECODE_BRANCH = 0x4000
SB_PROPOSAL_CAUSE_PROGRESS = 0x8000
SB_PROPOSAL_CAUSE_CHART_SNAP = 0x10000
SB_PROPOSAL_CAUSE_NO_REQUEST = 0x20000
SB_PROPOSAL_CAUSE_FACING_TIE = 0x40000
SB_EMBODIED_TELEPORTED = 0x1


class SbOrientation(ctypes.Structure):
    """Mirrors sb_orientation_t (ABI 3.1)."""
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("type", ctypes.c_int32),
        ("yaw", ctypes.c_double),
        ("pitch", ctypes.c_double),
        ("roll", ctypes.c_double),
    ]


class SbAnchorPose(ctypes.Structure):
    """Mirrors sb_anchor_pose_t (ABI 4): the driver's pose of one anchor."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("z", ctypes.c_double),
        ("orientation", SbOrientation),
        ("epoch", ctypes.c_uint64),
    ]


class SbBonePose(ctypes.Structure):
    """Mirrors sb_bone_pose_t (ABI 4.7): the pose of one bone, bone-local -> world."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("position", ctypes.c_double * 3),
        ("rotation", ctypes.c_double * 4),
        ("epoch", ctypes.c_uint64),
    ]


class SbNamedBonePose(ctypes.Structure):
    """Mirrors sb_named_bone_pose_t (ABI 4.7)."""

    _fields_ = [
        ("bone_name", ctypes.c_char_p),
        ("pose", SbBonePose),
    ]


class SbChainJoint(ctypes.Structure):
    """Mirrors sb_chain_joint_t (ABI 4.7): one joint of the normalized kinematic chain."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("parent", ctypes.c_int32),
        ("rest_position", ctypes.c_double * 3),
        ("rest_rotation", ctypes.c_double * 4),
        ("segment_length_m", ctypes.c_double),
        ("limit_mask", ctypes.c_uint32),
        ("limit_min", ctypes.c_double * 3),
        ("limit_max", ctypes.c_double * 3),
    ]


class SbAnchorProposal(ctypes.Structure):
    """Mirrors sb_anchor_proposal_t (ABI 4): the proposal of ONE anchor."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("anchor_id", ctypes.c_uint32),
        ("status", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("cause", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("z", ctypes.c_double),
        ("orientation", SbOrientation),
        ("err_spacing_target", ctypes.c_double),
        ("err_spacing_sample", ctypes.c_double),
        ("err_facing_target", ctypes.c_double),
        ("err_facing_sample", ctypes.c_double),
        ("err_pose_target", ctypes.c_double),
        ("err_pose_sample", ctypes.c_double),
        ("err_scale_m", ctypes.c_double),
        ("referent_shared_id", ctypes.c_int64),
    ]


class SbAnchorSolve(ctypes.Structure):
    """Mirrors sb_anchor_solve_t (ABI 4.11): the inner level of one anchor's proposal."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("anchor_id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("iterations", ctypes.c_int32),
        ("object_shared_id", ctypes.c_int64),
        ("sd_scale_m", ctypes.c_double),
        ("objective_j", ctypes.c_double),
        ("facing_yaw", ctypes.c_double),
        ("driving_pose_tag", ctypes.c_char * 32),
    ]


class SbAnchorChart(ctypes.Structure):
    """Mirrors sb_anchor_chart_t (ABI 4.14): one anchor's replay through the chart of a transfer leaf."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("anchor_id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("leaf_shared_id", ctypes.c_int64),
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("z", ctypes.c_double),
        ("orientation", SbOrientation),
        ("revision", ctypes.c_uint64),
    ]


class SbAnchorTurn(ctypes.Structure):
    """Mirrors sb_anchor_turn_t (ABI 4.12): the signed, unwrapped yaw turn of one anchor's proposal."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("anchor_id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("yaw", ctypes.c_double),
        ("yaw_turn", ctypes.c_double),
    ]


class SbProposalSet(ctypes.Structure):
    """Mirrors sb_proposal_set_t (ABI 4): the header of one solve's set."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("anchor_count", ctypes.c_uint32),
        ("total_anchor_count", ctypes.c_uint32),
        ("sample_valid", ctypes.c_int32),
        ("revision", ctypes.c_uint64),
        ("source_sequence", ctypes.c_int64),
        ("sample_sequence", ctypes.c_uint64),
        ("sample_epoch", ctypes.c_uint64),
        ("sample_timestamp_s", ctypes.c_double),
        ("sample_age_s", ctypes.c_double),
    ]


SB_EXCEPTION_PAYLOAD_MAX = 256
SB_EXCEPTION_JOIN = 1
SB_ACTOR_EVENT_SOURCE_EXCEPTION = 1
SB_ACTOR_EVENT_TELEPORT_REQUEST = 2
SB_INTERACTION_SPACING_CONTACT = 0
SB_INTERACTION_AFFORDANCE_POSE_ENGAGEMENT = 1


class SbActorEvent(ctypes.Structure):
    """Mirrors sb_actor_event_t (ABI 4)."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("kind", ctypes.c_int32),
        ("revision", ctypes.c_uint64),
        ("source_sequence", ctypes.c_int64),
        ("source_shared_id", ctypes.c_int64),
        ("exception_id", ctypes.c_uint32),
        ("payload_size", ctypes.c_uint32),
        ("interaction_kind", ctypes.c_int32),
        ("anchor_id", ctypes.c_uint32),
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("z", ctypes.c_double),
        ("orientation", SbOrientation),
        ("payload", ctypes.c_uint8 * SB_EXCEPTION_PAYLOAD_MAX),
    ]


# sb_anchor_proposal_t::flags
SB_ANCHOR_PROPOSAL_FLAG_YAW_UNSPECIFIED = 0x1
SB_ANCHOR_PROPOSAL_FLAG_ENGAGED = 0x2
SB_ANCHOR_PROPOSAL_FLAG_SAMPLED = 0x4
SB_ANCHOR_PROPOSAL_FLAG_ARRIVED = 0x8
SB_ANCHOR_PROPOSAL_FLAG_POSITION_UNSPECIFIED = 0x10  # ABI 4.1
SB_ANCHOR_PROPOSAL_FLAG_TRANSFER = 0x20  # ABI 4.3: the root proposal is the transfer mapping
SB_ANCHOR_PROPOSAL_FLAG_INNER = 0x40  # ABI 4.11: the anchor's inner solve among its object's affordance poses
SB_ANCHOR_PROPOSAL_FLAG_FACING_YAW = 0x80  # ABI 4.11: sb_anchor_solve_t.facing_yaw is valid

# sb_orientation_type_t (ABI 3.1)
SB_ORIENTATION_YAW = 0
SB_ORIENTATION_YAW_PITCH = 1
SB_ORIENTATION_YAW_PITCH_ROLL = 2

# sb_scene_status_t (spatialblend_type.h). A mutator that is not OK is not
# registered.
SB_SCENE_OK = 0
SB_SCENE_INVALID_ARGUMENT = 1
SB_SCENE_CONTAINMENT_OVERLAP = 2
SB_SCENE_CONTAINMENT_ESCAPES_PARENT = 3


class SbActorState(ctypes.Structure):
    """Mirrors sb_actor_state_t, spatialblend_type.h:62-66.

    typedef struct {
      double position_x;
      double position_y;
      double yaw;
    } sb_actor_state_t;
    """

    _fields_ = [
        ("position_x", ctypes.c_double),
        ("position_y", ctypes.c_double),
        ("yaw", ctypes.c_double),
    ]


class SbRealizationTarget(ctypes.Structure):
    """Mirrors sb_realization_target_t, spatialblend_type.h:101-112.

    Field order there is deliberately ABI-safety-optimized (see the comment
    at spatialblend_type.h:95-100): all 8-byte members (doubles + the
    anchor_name pointer) first, then the two 4-byte ints packed together,
    then the trailing 8-byte int64 -- so this Structure reproduces that same
    order verbatim rather than reordering for "readability".

    typedef struct {
      double arrival_time;
      const char* anchor_name;
      double position_x;
      double position_y;
      double position_z;
      double yaw;
      double weight;
      int32_t has_z;
      sb_kernel_type_t kernel_type;
      int64_t referent_id;
    } sb_realization_target_t;
    """

    _fields_ = [
        ("arrival_time", ctypes.c_double),
        ("anchor_name", ctypes.c_char_p),
        ("position_x", ctypes.c_double),
        ("position_y", ctypes.c_double),
        ("position_z", ctypes.c_double),
        ("yaw", ctypes.c_double),
        ("weight", ctypes.c_double),
        ("has_z", ctypes.c_int32),
        ("kernel_type", SbKernelType),
        ("referent_id", ctypes.c_int64),
    ]


class SbRelationEvent(ctypes.Structure):
    """Mirrors sb_relation_event_t, spatialblend_type.h:138-146.

    typedef struct {
      int64_t actor_shared_id;
      const char* anchor_name;
      int64_t referent_shared_id;
      sb_kernel_type_t kernel_type;
      sb_relation_phase_t from_phase;
      sb_relation_phase_t to_phase;
      double timestamp;
    } sb_relation_event_t;
    """

    _fields_ = [
        ("actor_shared_id", ctypes.c_int64),
        ("anchor_name", ctypes.c_char_p),
        ("referent_shared_id", ctypes.c_int64),
        ("kernel_type", SbKernelType),
        ("from_phase", SbRelationPhase),
        ("to_phase", SbRelationPhase),
        ("timestamp", ctypes.c_double),
    ]


# Struct name -> ctypes Structure, used by _api.check_struct_layout() to
# cross-check against the (optional) native sb_sizeof().
STRUCT_LAYOUT_CHECKS = (
    ("sb_actor_state_t", SbActorState),
    ("sb_realization_target_t", SbRealizationTarget),
    ("sb_relation_event_t", SbRelationEvent),
)
# ABI 4 ids 21..25 (sb_orientation_t, sb_anchor_pose_t, sb_anchor_proposal_t,
# sb_proposal_set_t, sb_actor_event_t): size-checked by check_struct_layout
# through EXTRA_SIZE_CHECKS below. Ids 19 / 20 (ABI 3) are retired, not reused.
EXTRA_SIZE_CHECKS = ((21, "SbOrientation"), (22, "SbAnchorPose"),
                     (23, "SbAnchorProposal"), (24, "SbProposalSet"),
                     (25, "SbActorEvent"), (26, "SbBonePose"),
                     (27, "SbNamedBonePose"), (28, "SbChainJoint"),
                     (30, "SbAnchorSolve"), (31, "SbAnchorTurn"),
                     (33, "SbAnchorChart"))
