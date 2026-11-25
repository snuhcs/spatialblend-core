# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Pythonic access to the observability and injection C ABI
(spatialblend/api/c/spatialblend_observability.h).

Every value type here is a frozen dataclass generated from the ctypes mirror
of its C struct (capi/_obs_types.py), so its field names are exactly the C
field names, minus the {struct_size, version, reserved} bookkeeping:

- strings (``anchor_name``, ``sub_region_name``, ``tag_filter``) are
  decoded ``str`` copies (the native pointers are only valid until the next
  call);
- ``encoded_state`` is a tuple trimmed to ``encoded_state_size``;
- ``int32`` flags stay ints (0/1) so a record compares 1:1 with the C row.

Capture is opt-in: a space records scope records, policy observations and
solver diagnostics while it is traced or after enable_capture(space). The
getters enable capture themselves, so call enable_capture before driving a
space whose first evaluation you want to read.

The functions take raw handles (ints) so the eval harness can use them
without building api.Space objects; api.Space / api.Actor / api.Element
expose the same calls as methods.

Surface map: relation_states_v2 / scope_records / scope_candidates (relation
states and scope records), policy params, realization_weights / channels,
solver_diagnostics / solver_goals / realization_targets_v2 (witness points,
feasibility snap, reach diagnostic), inject_relation_payloads and
evaluate_relation (injection and pure evaluation), element_world_transform /
element_parent (element tree readback).
"""
import ctypes
import dataclasses
import math
from typing import List, Optional, Sequence, Tuple

from . import _api
from . import _obs_types as _o

__all__ = [
    "RelationStateV2", "ScopeRecord", "ScopeCandidate", "PolicyParams",
    "RealizationWeight", "RealizationChannel", "SolverDiagnostics",
    "SolverGoal", "RealizationTargetV2", "RelationEvaluation",
    "TargetRelationEvaluation", "evaluate_target_relation",
    "RelationPayload", "observability_version", "trace_format_version",
    "enable_capture", "relation_states", "relation_states_v2",
    "scope_records",
    "scope_candidates", "default_policy_params", "policy_params",
    "set_policy_params", "realization_weights", "realization_channels",
    "solver_diagnostics", "solver_goals", "realization_targets_v2",
    "inject_relation_payloads", "evaluate_relation",
    "element_world_transform", "element_parent",
]

_SKIP = {"struct_size", "version", "reserved"}


def _record_type(name, struct_cls, extra=()):
    fields = []
    for field_name, field_type in struct_cls._fields_:
        if field_name in _SKIP:
            continue
        fields.append((field_name, object))
    fields.extend(extra)
    return dataclasses.make_dataclass(name, fields, frozen=True)


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return "" if value is None else value


def _convert(record_cls, native, **overrides):
    values = {}
    names = {f.name for f in dataclasses.fields(record_cls)}
    for field_name, field_type in type(native)._fields_:
        if field_name not in names:
            continue
        value = getattr(native, field_name)
        if field_type is ctypes.c_char_p:
            value = _decode(value)
        elif field_name == "encoded_state":
            size = min(int(native.encoded_state_size), _o.MAX_ENCODED_STATE)
            value = tuple(value[i] for i in range(size))
        values[field_name] = value
    values.update(overrides)
    return record_cls(**values)


RelationState = _record_type("RelationState", _o.SbRelationState)
RelationStateV2 = _record_type("RelationStateV2", _o.SbRelationStateV2)
ScopeRecord = _record_type("ScopeRecord", _o.SbScopeRecord)
ScopeCandidate = _record_type("ScopeCandidate", _o.SbScopeCandidate,
                              extra=(("reason_name", object),))
PolicyParams = _record_type("PolicyParams", _o.SbRealizationPolicyParams)
RealizationWeight = _record_type("RealizationWeight", _o.SbRealizationWeight)
RealizationChannel = _record_type("RealizationChannel",
                                  _o.SbRealizationChannel)
SolverDiagnostics = _record_type("SolverDiagnostics", _o.SbSolverDiagnostics)
SolverGoal = _record_type("SolverGoal", _o.SbSolverGoal)
RealizationTargetV2 = _record_type("RealizationTargetV2",
                                   _o.SbRealizationTargetV2)
RelationEvaluation = _record_type("RelationEvaluation",
                                  _o.SbRelationEvaluation)
TargetRelationEvaluation = _record_type("TargetRelationEvaluation",
                                        _o.SbTargetRelationEvaluation)


@dataclasses.dataclass
class RelationPayload:
    """One relation as a receiver sees it (sb_relation_payload_t), for
    inject_relation_payloads. ``kernel_type`` is the C enum
    (api.KernelType: SPACING=0, FACING=1, AFFORDANCE_POSE=2)."""
    referent_shared_id: int
    kernel_type: int
    anchor_name: str = "root"
    proximity: float = 0.0
    estimated_time: float = 0.0
    sub_region_name: str = ""
    support_distance_m: float = -1.0
    source_scope_diameter_m: float = -1.0
    encoded_state: Sequence[float] = ()
    has_contact: bool = False
    contact_point_local: Tuple[float, float] = (0.0, 0.0)
    contact_point_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normalized_strategy: int = 0
    normalized_angle: float = 0.0
    normalized_radial_ratio: float = 0.0
    normalized_height_ratio: float = 0.0
    has_interior: bool = False
    interior_uvw: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    enabled: bool = True
    # Radial chart version of the normalized contact (0 = pure radial,
    # 2 = the core-regularized chart the current extractor writes).
    chart_version: int = 0
    # Identities as a peer payload carries them (None = absent): the
    # source-side referent handle (space, element), the source normalizing
    # scope's shared id and its source-side handle.
    referent_source: Optional[Tuple[int, int]] = None
    scope_shared_id: Optional[int] = None
    scope_source: Optional[Tuple[int, int]] = None
    # The source binding (ordinal, revision) the relation was
    # extracted under; None = unknown binding revision.
    source_binding_ordinal: int = (1 << 64) - 1
    binding_revision: Optional[int] = None
    # AffordancePose only -- the anchor's pose relative to the
    # support (in-plane offset + height, support-local frame). None =
    # unknown (a pre-pose-fit sender, or any other relation family).
    ap_offset_u_m: Optional[float] = None
    ap_offset_v_m: Optional[float] = None
    ap_height_m: Optional[float] = None


def _native():
    return _api.get_native()


def enable_capture(space: int, enabled: bool = True) -> None:
    """sb_set_observability_capture."""
    if _native().sb_set_observability_capture(space, 1 if enabled else 0) != 1:
        raise ValueError("unknown space %r" % space)


def observability_version() -> int:
    return int(_native().sb_get_observability_version())


def trace_format_version() -> int:
    return int(_native().sb_get_trace_format_version())


def _query_rows(fn, struct_cls, record_cls, *leading, convert=None):
    size = ctypes.sizeof(struct_cls)
    count = fn(*leading, None, 0, size)
    if count < 0:
        raise RuntimeError("%s rejected the request (%d)" % (fn.__name__, count))
    while True:
        buf = (struct_cls * max(count, 1))()
        got = fn(*leading, buf, count, size)
        if got < 0:
            raise RuntimeError("%s failed (%d)" % (fn.__name__, got))
        if got <= count:
            break
        count = got  # grew between the probe and the read
    return [(convert or _convert)(record_cls, buf[i]) for i in range(got)]


def relation_states(space: int, actor: int) -> List["RelationState"]:
    """sb_get_relation_states (the legacy v1 row). The v1 call has no
    size probe (cap 0 returns 0), so read with a buffer and grow it."""
    cap = 64
    while True:
        buf = (_o.SbRelationState * cap)()
        got = _native().sb_get_relation_states(space, actor, buf, cap)
        if got <= cap:
            return [_convert(RelationState, buf[i]) for i in range(got)]
        cap = got


def relation_states_v2(space: int, actor: int) -> List["RelationStateV2"]:
    return _query_rows(_native().sb_get_relation_states_v2,
                       _o.SbRelationStateV2, RelationStateV2, space, actor)


def scope_records(space: int, actor: int) -> List["ScopeRecord"]:
    return _query_rows(_native().sb_get_scope_records, _o.SbScopeRecord,
                       ScopeRecord, space, actor)


def scope_candidates(space: int, actor: int,
                     binding_index: int) -> List["ScopeCandidate"]:
    def convert(record_cls, native):
        return _convert(record_cls, native,
                        reason_name=_o.SCOPE_REASON_NAMES.get(
                            int(native.reason), "unknown"))
    return _query_rows(_native().sb_get_scope_candidates, _o.SbScopeCandidate,
                       ScopeCandidate, space, actor, int(binding_index),
                       convert=convert)


def default_policy_params() -> "PolicyParams":
    out = _o.new(_o.SbRealizationPolicyParams)
    if _native().sb_get_default_realization_policy_params(
            ctypes.byref(out), ctypes.sizeof(out)) != 1:
        raise RuntimeError("sb_get_default_realization_policy_params failed")
    return _convert(PolicyParams, out)


def policy_params(space: int) -> "PolicyParams":
    out = _o.new(_o.SbRealizationPolicyParams)
    if _native().sb_get_realization_policy_params(
            space, ctypes.byref(out), ctypes.sizeof(out)) != 1:
        raise RuntimeError("sb_get_realization_policy_params failed")
    return _convert(PolicyParams, out)


def set_policy_params(space: int, params: "PolicyParams" = None,
                      **changes) -> bool:
    """Set every numeric policy parameter at once. Start from ``params``
    (default: the space's current values) and apply ``changes``
    (e.g. gamma=1.0). Returns False (nothing changed) for invalid values."""
    base = params if params is not None else policy_params(space)
    values = dataclasses.asdict(base)
    values.update(changes)
    native = _o.new(_o.SbRealizationPolicyParams)
    for name, _ in _o.SbRealizationPolicyParams._fields_:
        if name in values:
            setattr(native, name, values[name])
    return _native().sb_set_realization_policy_params(
        space, ctypes.byref(native), ctypes.sizeof(native)) == 1


_RELATION_PHASE_CONFIG_FIELDS = [
    f[0] for f in _o.SbRelationPhaseConfig._fields_
    if f[0] not in ("struct_size", "reserved")
]


def relation_phase_config(space: int, kernel_type: int) -> dict:
    """Current per-family lifecycle thresholds (driving / contact). ABI 4
    has no activation floor, near field or scope radius: scope is the family's."""
    out = _o.SbRelationPhaseConfig(struct_size=ctypes.sizeof(_o.SbRelationPhaseConfig))
    if _native().sb_set_relation_phase_config(
            space, int(kernel_type), None, ctypes.byref(out)) != 1:
        raise RuntimeError("sb_set_relation_phase_config (read) failed")
    return {name: getattr(out, name) for name in _RELATION_PHASE_CONFIG_FIELDS}


def set_relation_phase_config(space: int, kernel_type: int, config: dict = None,
                              **changes) -> dict:
    """Set every threshold at once. Start from ``config`` (default: the
    space's current values for this family) and apply ``changes`` (e.g.
    driving_weight=0.6). Returns the effective (post-set) config;
    raises if any threshold is invalid (nothing changes)."""
    base = dict(config) if config is not None else relation_phase_config(
        space, kernel_type)
    base.update(changes)
    native = _o.SbRelationPhaseConfig(
        struct_size=ctypes.sizeof(_o.SbRelationPhaseConfig))
    for name in _RELATION_PHASE_CONFIG_FIELDS:
        setattr(native, name, base[name])
    out = _o.SbRelationPhaseConfig(struct_size=ctypes.sizeof(_o.SbRelationPhaseConfig))
    if _native().sb_set_relation_phase_config(
            space, int(kernel_type), ctypes.byref(native), ctypes.byref(out)) != 1:
        raise RuntimeError("sb_set_relation_phase_config (write) failed")
    return {name: getattr(out, name) for name in _RELATION_PHASE_CONFIG_FIELDS}


def realization_weights(space: int, actor: int) -> List["RealizationWeight"]:
    return _query_rows(_native().sb_get_realization_weights,
                       _o.SbRealizationWeight, RealizationWeight, space, actor)


def realization_channels(space: int,
                         actor: int) -> List["RealizationChannel"]:
    return _query_rows(_native().sb_get_realization_channels,
                       _o.SbRealizationChannel, RealizationChannel, space,
                       actor)


def solver_diagnostics(space: int,
                       actor: int) -> Optional["SolverDiagnostics"]:
    out = _o.new(_o.SbSolverDiagnostics)
    if _native().sb_get_solver_diagnostics(space, actor, ctypes.byref(out),
                                           ctypes.sizeof(out)) != 1:
        return None
    return _convert(SolverDiagnostics, out)


def solver_goals(space: int, actor: int) -> List["SolverGoal"]:
    return _query_rows(_native().sb_get_solver_goals, _o.SbSolverGoal,
                       SolverGoal, space, actor)


def realization_targets_v2(space: int,
                           actor: int) -> List["RealizationTargetV2"]:
    return _query_rows(_native().sb_get_realization_targets_v2,
                       _o.SbRealizationTargetV2, RealizationTargetV2, space,
                       actor)


def _payload_to_native(p: RelationPayload, keep):
    n = _o.new(_o.SbRelationPayload)
    anchor = (p.anchor_name or "").encode("utf-8")
    sub = (p.sub_region_name or "").encode("utf-8")
    keep.extend([anchor, sub])
    n.anchor_name = anchor
    n.sub_region_name = sub
    n.referent_shared_id = int(p.referent_shared_id)
    n.kernel_type = int(p.kernel_type)
    n.proximity = float(p.proximity)
    n.estimated_time = float(p.estimated_time)
    n.support_distance_m = float(p.support_distance_m)
    n.source_scope_diameter_m = float(p.source_scope_diameter_m)
    state = list(p.encoded_state)
    if len(state) > _o.MAX_ENCODED_STATE:
        raise ValueError("encoded_state longer than %d" % _o.MAX_ENCODED_STATE)
    n.encoded_state_size = len(state)
    for i, v in enumerate(state):
        n.encoded_state[i] = float(v)
    n.has_contact = 1 if p.has_contact else 0
    n.contact_point_local_x, n.contact_point_local_y = map(
        float, p.contact_point_local)
    (n.contact_point_world_x, n.contact_point_world_y,
     n.contact_point_world_z) = map(float, p.contact_point_world)
    n.normalized_strategy = int(p.normalized_strategy)
    n.normalized_angle = float(p.normalized_angle)
    n.normalized_radial_ratio = float(p.normalized_radial_ratio)
    n.normalized_height_ratio = float(p.normalized_height_ratio)
    n.has_interior = 1 if p.has_interior else 0
    n.interior_u, n.interior_v, n.interior_w = map(float, p.interior_uvw)
    n.enabled = 1 if p.enabled else 0
    n.chart_version = int(p.chart_version)
    if p.referent_source is not None:
        n.has_referent_source = 1
        n.referent_source_space_id, n.referent_source_element_id = map(
            int, p.referent_source)
    if p.scope_shared_id is not None:
        n.has_scope_shared_id = 1
        n.scope_shared_id = int(p.scope_shared_id)
    if p.scope_source is not None:
        n.has_scope_source = 1
        n.scope_source_space_id, n.scope_source_element_id = map(
            int, p.scope_source)
    n.source_binding_ordinal = int(p.source_binding_ordinal)
    if p.binding_revision is not None:
        n.has_binding_revision = 1
        n.binding_revision = int(p.binding_revision)
    # None -> NaN ("unknown"), matching GroundedRelation's own
    # default and the C struct's documented sentinel.
    n.ap_offset_u_m = float(p.ap_offset_u_m) if p.ap_offset_u_m is not None \
        else math.nan
    n.ap_offset_v_m = float(p.ap_offset_v_m) if p.ap_offset_v_m is not None \
        else math.nan
    n.ap_height_m = float(p.ap_height_m) if p.ap_height_m is not None \
        else math.nan
    # Offsets are valid only under formula version 1 (all three supplied).
    n.r_formula_version = 1 if (p.ap_offset_u_m is not None
                                and p.ap_offset_v_m is not None
                                and p.ap_height_m is not None) else 0
    return n


def inject_relation_payloads(space: int, actor: int,
                             payloads: Sequence[RelationPayload],
                             time_seconds: float) -> None:
    """Drive ``actor``'s receiver with one fixed relation snapshot at
    ``time_seconds`` (policy, solver, targets, trace and phase events all
    run; no source extractor). An empty list is an authoritative empty
    snapshot. Raises ValueError when the core rejects the input."""
    keep = []
    rows = [_payload_to_native(p, keep) for p in payloads]
    buf = (_o.SbRelationPayload * max(len(rows), 1))(*rows)
    ok = _native().sb_inject_relation_payloads(
        space, actor, buf, len(rows), ctypes.sizeof(_o.SbRelationPayload),
        float(time_seconds))
    if ok != 1:
        raise ValueError("sb_inject_relation_payloads rejected the snapshot")


def evaluate_relation(space: int, kernel_type: int, anchor_xyz, anchor_yaw,
                      referent: int, scope_root: int = 0,
                      position_dimension: int = 2,
                      explicit_scope: bool = False) -> "RelationEvaluation":
    """Pure Extract_p (ComputeSupport + Encode) for one relation.
    ``scope_root`` 0 evaluates without a navigable normalizing scope.
    explicit_scope=True mirrors an sb_bind_scoped binding (the navigable
    search stops at scope_root); False mirrors an owner-derived binding
    (scope_root = the actor element's containment ancestor, the search may
    continue above it)."""
    q = _o.new(_o.SbRelationQuery)
    q.anchor_x, q.anchor_y, q.anchor_z = map(float, anchor_xyz)
    q.anchor_yaw = float(anchor_yaw)
    q.referent = int(referent)
    q.scope_root = int(scope_root)
    q.kernel_type = int(kernel_type)
    q.position_dimension = int(position_dimension)
    q.explicit_scope = 1 if explicit_scope else 0
    out = _o.new(_o.SbRelationEvaluation)
    if _native().sb_evaluate_relation(space, ctypes.byref(q),
                                      ctypes.byref(out)) != 1:
        raise ValueError("sb_evaluate_relation rejected the query")
    return _convert(RelationEvaluation, out)


def evaluate_target_relation(space: int, payload: RelationPayload,
                             anchor_xyz, anchor_yaw: float,
                             normalizing_scope: int = 0
                             ) -> "TargetRelationEvaluation":
    """Target-side Eval_p -- R^tgt of a received ``payload`` on this
    space's corresponding referent at the candidate anchor pose, with the
    source family's kernel (sb_evaluate_target_relation). Returns the
    evaluation, including non-``evaluated`` dispositions (see
    ``disposition_name``); raises ValueError on bad arguments."""
    keep = []
    native = _payload_to_native(payload, keep)
    q = _o.new(_o.SbRelationQuery)
    q.anchor_x, q.anchor_y, q.anchor_z = map(float, anchor_xyz)
    q.anchor_yaw = float(anchor_yaw)
    q.scope_root = int(normalizing_scope)
    out = _o.new(_o.SbTargetRelationEvaluation)
    if _native().sb_evaluate_target_relation(
            space, ctypes.byref(native), ctypes.sizeof(native),
            ctypes.byref(q), ctypes.byref(out)) != 1:
        raise ValueError("sb_evaluate_target_relation rejected the input")
    return _convert(TargetRelationEvaluation, out)


def disposition_name(evaluation: "TargetRelationEvaluation") -> str:
    return _o.TARGET_EVAL_DISPOSITION_NAMES.get(int(evaluation.disposition),
                                                "unknown")


def element_world_transform(space: int, elem: int):
    """Row-major 4x4 world transform as 4 row tuples, or None."""
    buf = (ctypes.c_double * 16)()
    if _native().sb_get_element_world_transform(space, elem, buf) != 1:
        return None
    return tuple(tuple(buf[r * 4 + c] for c in range(4)) for r in range(4))


def element_parent(space: int, elem: int) -> Optional[int]:
    """Parent element id (0 for a root), or None for an unknown element."""
    out = ctypes.c_int64()
    if _native().sb_get_element_parent(space, elem, ctypes.byref(out)) != 1:
        return None
    return int(out.value)


def is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)
