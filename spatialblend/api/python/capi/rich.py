# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Rich-realization surface of the C API, and its JSON wire format.

Wraps spatialblend/api/c/spatialblend_rich.h (activation, poll, callback,
retire/release), spatialblend_receiver_input.h and
spatialblend_receiver_reevaluation.h, and serializes a snapshot to the
``spatialblend.rich_realization`` schema-v1 JSON **exactly as Unity writes it**
(the Unity app, snuhcs/spatialblend-unity:
Assets/SpatialBlend/Foundation/Realization/RichRealizationSnapshot.cs,
RichWire.Write, and the native-to-managed mapping in
Assets/SpatialBlend/Foundation/Internal/SpatialBlendAPI.RichRealization.cs
CopyHeader/CopyChannel/CopyContributor). This is the stream the end-to-end
driver writes and the motion generator's
``motiongen.cues.rich_realization.parse_rich_realization`` reads.

Byte compatibility: key order, the known/value wrappers, 64-bit integers
as decimal strings, enum names in snake_case, the string escaping, and
Unity's mono "R" number formatting (capi/_mono_number.py) are all
reproduced, so a
Python-serialized snapshot of the same native transaction is byte-identical
to the Unity-serialized one (tests/test_capi_rich.py checks this against a
vendored Unity native capture).

Typical use (receiver side)::

    from spatialblend.api.python.capi import api, rich

    token = rich.RichRealization.activate(space, actor)
    ... drive the source ...
    snapshot = token.poll()          # None when nothing new
    if snapshot is not None:
        line = rich.to_rich_json(snapshot)
    token.retire(); token.release()
"""
import ctypes
import dataclasses
import math
from typing import Callable, List, Optional, Sequence, Tuple

from . import _api
from . import _mono_number
from . import _obs_types as _o

__all__ = [
    "Known", "RichHeader", "RichChannel", "RichContributor", "RichSnapshot",
    "RichRealization", "RichRealizationError", "register_on_rich_realization",
    "to_rich_json", "rich_feature", "monotonic_ns", "begin_receiver_input",
    "submit_receiver_input", "clear_receiver_input",
    "reevaluate_receiver_realization", "from_rich_json", "SCHEMA",
    "SCHEMA_VERSION",
    "FEATURE_VERSION", "COORDINATE_SYSTEM", "UNITS",
]

SCHEMA = "spatialblend.rich_realization"
SCHEMA_VERSION = 1
FEATURE_VERSION = 1
COORDINATE_SYSTEM = "spatialblend-z-up"
UNITS = "metres"
MAX_JSON_UTF8_BYTES = 1024 * 1024

_STATUS = {1: "complete", 2: "empty", 3: "failed"}
_PROVENANCE = {0: "legacy_unknown", 1: "timestamped_observation",
               2: "observation_reset"}
_FAMILY = {0: "facing", 1: "spacing", 2: "affordance_pose"}
_SOLVER = {0: "converged", 1: "fallback", 2: "failed"}
_GEOMETRY = {0: "not_applicable", 1: "exact_support", 2: "blended_support",
             3: "reach_projected", 4: "legacy_center_non_contact",
             5: "unsupported"}
_GOAL = {0: "none", 1: "root_position", 2: "facing_direction",
         3: "support_pose", 4: "soft_geometric_pose"}
# 5: admitted, source anchor not engaged -- the anchor follows the
# body. Spelled like Unity's RichContributorDisposition.AwaitingEngagement
# through RichRealizationSnapshot.Name() (PascalCase -> snake_case).
_DISPOSITION = {0: "included", 1: "disabled", 2: "unmapped",
                3: "grounding_failed", 4: "unsupported",
                5: "awaiting_engagement"}
_GROUNDING = {0: "explicit_sub_region", 1: "whole_planar_boundary",
              2: "legacy_center_fallback"}

_RESULT_NAMES = {
    1: "OK", 0: "NO_NEW_SNAPSHOT", -1: "INVALID_ARGUMENT",
    -2: "UNKNOWN_TOKEN", -3: "BUFFER_TOO_SMALL", -4: "UNSUPPORTED_VERSION",
    -5: "UNAVAILABLE", -6: "STALE_SNAPSHOT",
}


class RichRealizationError(RuntimeError):
    def __init__(self, what, code):
        super().__init__("%s: %s (%d)" % (what, _RESULT_NAMES.get(code, "?"),
                                          code))
        self.code = code


@dataclasses.dataclass(frozen=True)
class Known:
    """An explicit known/value wrapper (RichKnown{UInt64,Int64,Double})."""
    known: bool
    value: object = None
    kind: str = "i64"  # "u64" | "i64" | "double"


@dataclasses.dataclass(frozen=True)
class RichHeader:
    receiver_space_id: int
    receiver_actor_id: int
    producer_lifetime: int
    revision: int
    snapshot_status: str
    source_provenance_mode: str
    source_epoch: Known
    source_sequence: Known
    source_observation_sequence: Known
    source_actor_space: Known
    source_actor_element: Known
    relay_source_space: Known
    source_time_seconds: Known
    source_clock_seconds: Known
    native_clock_id: Known
    native_clock_epoch: Known
    native_receipt_nanoseconds: Known
    native_solve_nanoseconds: Known
    policy_revision: Known
    receiver_tree_revision: Known
    counterpart_revision: Known
    support_revision: Known
    known_revision_mask: int
    unknown_revision_mask: int
    failure_code: int


@dataclasses.dataclass(frozen=True)
class RichContributor:
    referent_id: Known
    source_space: Known
    source_element: Known
    resolved_local_element: Known
    binding_revision: Known
    snapshot_ordinal: int
    applied_weight: float
    has_mapped_support: bool
    mapped_support: Tuple[float, float, float]
    grounding_mode: str
    disposition: str
    chart_version_known: bool
    binding_revision_known: bool
    reason: str
    support_identifier: str
    support_plane_known: bool
    support_plane_to_world: Tuple[float, ...]  # 16, row-major


@dataclasses.dataclass(frozen=True)
class RichChannel:
    anchor: str
    family: str
    status: str
    geometry: str
    goal_kind: str
    has_goal: bool
    referent_id: Known
    arrival_seconds: float
    admitted_position: Tuple[float, float, float]
    has_z: bool
    yaw_radians: float
    weight: float
    requested_position_known: bool
    requested_x: float
    requested_y: float
    reach_projected: bool
    reach_displacement_metres: float
    contributors: Tuple[RichContributor, ...]


@dataclasses.dataclass(frozen=True)
class RichSnapshot:
    header: RichHeader
    channels: Tuple[RichChannel, ...]


# ---------------------------------------------------------------------------
# Native -> Python (mirrors SpatialBlendAPI.RichRealization.cs Copy*)
# ---------------------------------------------------------------------------

def _id(value: int) -> Known:
    return Known(value != -1, int(value), "i64")


def _utf8(buf: bytes, offset: int, length: int) -> str:
    if offset + length > len(buf):
        raise RichRealizationError("malformed native utf8 range", -1)
    return buf[offset:offset + length].decode("utf-8", "strict")


def _header(h) -> RichHeader:
    causal = h.source_timing_mode in (1, 2)
    receipt = h.receiver_ingress_monotonic_ns != 0 or \
        h.receiver_solve_monotonic_ns != 0
    mask = int(h.known_revision_mask)

    def revision(value, bit):
        return Known(bool(mask & bit), int(value), "u64")

    return RichHeader(
        receiver_space_id=int(h.receiver_space),
        receiver_actor_id=int(h.receiver_actor),
        producer_lifetime=int(h.producer_lifetime),
        revision=int(h.revision),
        snapshot_status=_STATUS[int(h.status)],
        source_provenance_mode=_PROVENANCE[int(h.source_timing_mode)],
        source_epoch=Known(causal, int(h.source_epoch), "u64"),
        source_sequence=Known(causal, int(h.source_sequence), "i64"),
        source_observation_sequence=Known(
            bool(h.source_observation_sequence_known),
            int(h.source_observation_sequence), "i64"),
        source_actor_space=_id(h.source_actor_space),
        source_actor_element=_id(h.source_actor_element),
        relay_source_space=_id(h.relay_source_space),
        source_time_seconds=Known(bool(h.source_time_known),
                                  float(h.source_time_seconds), "double"),
        source_clock_seconds=Known(bool(h.source_clock_known),
                                   float(h.source_clock_seconds), "double"),
        native_clock_id=Known(True, int(h.monotonic_clock_id), "u64"),
        native_clock_epoch=Known(True, int(h.monotonic_clock_epoch), "u64"),
        native_receipt_nanoseconds=Known(
            receipt, int(h.receiver_ingress_monotonic_ns), "u64"),
        native_solve_nanoseconds=Known(
            receipt, int(h.receiver_solve_monotonic_ns), "u64"),
        policy_revision=revision(h.policy_revision, 1),
        receiver_tree_revision=revision(h.tree_revision, 2),
        counterpart_revision=revision(h.counterpart_revision, 4),
        support_revision=revision(h.support_revision, 8),
        known_revision_mask=mask,
        unknown_revision_mask=int(h.unknown_revision_mask),
        failure_code=int(h.failure_code),
    )


def _contributor(c, utf8: bytes) -> RichContributor:
    return RichContributor(
        referent_id=_id(c.referent_id),
        source_space=_id(c.source_space),
        source_element=_id(c.source_element),
        resolved_local_element=_id(c.resolved_local_element),
        binding_revision=Known(bool(c.binding_revision_known),
                               int(c.binding_revision), "u64"),
        snapshot_ordinal=int(c.snapshot_ordinal),
        applied_weight=float(c.applied_weight),
        has_mapped_support=bool(c.has_mapped_support),
        mapped_support=(float(c.mapped_support_x), float(c.mapped_support_y),
                        float(c.mapped_support_z)),
        grounding_mode=_GROUNDING[int(c.grounding_mode)],
        # Unknown future values read as "unrecognized", as Unity does.
        disposition=_DISPOSITION.get(int(c.disposition), "unrecognized"),
        chart_version_known=bool(c.chart_version_known),
        binding_revision_known=bool(c.binding_revision_known),
        reason=_utf8(utf8, c.reason_offset, c.reason_length),
        support_identifier=_utf8(utf8, c.support_identifier_offset,
                                 c.support_identifier_length),
        support_plane_known=bool(c.has_mapped_support),
        support_plane_to_world=tuple(
            float(c.support_plane_to_world_row_major[i]) for i in range(16)),
    )


def _channel(c, contributors: Sequence, utf8: bytes) -> RichChannel:
    rows = contributors[c.contributor_offset:
                        c.contributor_offset + c.contributor_count]
    return RichChannel(
        anchor=_utf8(utf8, c.anchor_offset, c.anchor_length),
        family=_FAMILY[int(c.family)],
        status=_SOLVER[int(c.solver_status)],
        geometry=_GEOMETRY[int(c.geometry)],
        goal_kind=_GOAL[int(c.goal_kind)],
        has_goal=bool(c.has_goal),
        referent_id=_id(c.referent_id),
        arrival_seconds=float(c.arrival_seconds),
        admitted_position=(float(c.position_x), float(c.position_y),
                           float(c.position_z)),
        has_z=bool(c.has_z),
        yaw_radians=float(c.yaw),
        weight=float(c.weight),
        requested_position_known=bool(c.has_requested_position),
        requested_x=float(c.requested_position_x),
        requested_y=float(c.requested_position_y),
        reach_projected=bool(c.reach_projected),
        reach_displacement_metres=float(c.reach_displacement_m),
        contributors=tuple(_contributor(x, utf8) for x in rows),
    )


def _snapshot(header, channels, contributors, utf8: bytes) -> RichSnapshot:
    contributor_rows = list(contributors)
    return RichSnapshot(
        header=_header(header),
        channels=tuple(_channel(c, contributor_rows, utf8) for c in channels))


# ---------------------------------------------------------------------------
# Activation, poll, callback
# ---------------------------------------------------------------------------

def _native():
    return _api.get_native()


def _handle(obj):
    for attr in ("handle", "id"):
        if hasattr(obj, attr) and not isinstance(obj, int):
            return int(getattr(obj, attr))
    return int(obj)


def rich_feature() -> "_o.SbRichFeature":
    feature = _o.SbRichFeature()
    _native().sb_get_rich_realization_feature(ctypes.byref(feature),
                                              ctypes.sizeof(feature))
    return feature


def monotonic_ns() -> int:
    """sb_read_rich_monotonic_ns (the clock receiver-input frames use)."""
    return int(_native().sb_read_rich_monotonic_ns())


class RichRealization:
    """An activated rich-realization token for one receiver actor."""

    def __init__(self, token: int):
        self.token = int(token)

    @classmethod
    def activate(cls, space, actor,
                 feature_version: int = _o.RICH_FEATURE_VERSION
                 ) -> "RichRealization":
        token = _native().sb_activate_rich_realization(
            _handle(space), _handle(actor), int(feature_version))
        if token == 0:
            raise RichRealizationError("sb_activate_rich_realization", -5)
        return cls(token)

    def poll(self) -> Optional[RichSnapshot]:
        """The next frozen snapshot, or None when there is nothing new."""
        native = _native()
        copy = _o.SbRichCopy()
        copy.struct_size = ctypes.sizeof(copy)
        result = native.sb_poll_rich_realization(self.token,
                                                 ctypes.byref(copy))
        if result == _o.SB_RICH_RESULT_NO_NEW_SNAPSHOT:
            return None
        if result == _o.SB_RICH_RESULT_BUFFER_TOO_SMALL:
            channels = (_o.SbRichChannel * max(copy.required_channels, 1))()
            contributors = (_o.SbRichContributor *
                            max(copy.required_contributors, 1))()
            utf8 = (ctypes.c_uint8 * max(copy.required_utf8_bytes, 1))()
            sized = _o.SbRichCopy()
            sized.struct_size = ctypes.sizeof(sized)
            sized.channel_capacity = copy.required_channels
            sized.contributor_capacity = copy.required_contributors
            sized.utf8_capacity = copy.required_utf8_bytes
            sized.channels = ctypes.cast(channels,
                                         ctypes.POINTER(_o.SbRichChannel))
            sized.contributors = ctypes.cast(
                contributors, ctypes.POINTER(_o.SbRichContributor))
            sized.utf8 = ctypes.cast(utf8, ctypes.POINTER(ctypes.c_uint8))
            result = native.sb_poll_rich_realization(self.token,
                                                     ctypes.byref(sized))
            copy = sized
        if result != _o.SB_RICH_RESULT_OK:
            raise RichRealizationError("sb_poll_rich_realization", result)
        n_ch, n_co = copy.required_channels, copy.required_contributors
        n_utf8 = copy.required_utf8_bytes
        channels = [copy.channels[i] for i in range(n_ch)] if n_ch else []
        contributors = [copy.contributors[i] for i in range(n_co)] \
            if n_co else []
        utf8 = bytes(copy.utf8[i] for i in range(n_utf8)) if n_utf8 else b""
        return _snapshot(copy.header, channels, contributors, utf8)

    def retire(self) -> None:
        _native().sb_retire_rich_realization(self.token)

    def release(self) -> int:
        return int(_native().sb_release_rich_realization(self.token))

    def __enter__(self) -> "RichRealization":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.retire()
        self.release()


def snapshot_from_native(view) -> RichSnapshot:
    """Deep-copy a borrowed sb_rich_realization_snapshot_t (callback view)."""
    channels = [view.channels[i] for i in range(view.channel_count)]
    contributors = [view.contributors[i]
                    for i in range(view.contributor_count)]
    utf8 = bytes(view.utf8[i] for i in range(view.utf8_bytes)) \
        if view.utf8_bytes else b""
    return _snapshot(view.header, channels, contributors, utf8)


def register_on_rich_realization(space, callback: Optional[
        Callable[[RichSnapshot], None]]) -> None:
    """Install ``callback(RichSnapshot)`` for ``space`` (an api.Space; the
    CFUNCTYPE is kept alive on its Session). The native view is borrowed, so
    it is converted before the callback runs. None unregisters."""
    native = _native()
    if callback is None:
        # A default-constructed CFUNCTYPE is the NULL function pointer.
        native.sb_register_on_rich_realization(
            _handle(space), _o.SB_ON_RICH_REALIZATION_FUNC(), None)
        return

    def _trampoline(view_ptr, user_data):
        callback(snapshot_from_native(view_ptr.contents))

    cfunc = _o.SB_ON_RICH_REALIZATION_FUNC(_trampoline)
    session = getattr(space, "session", None)
    if session is not None:
        session._callback_refs[(space.handle, "on_rich_realization")] = cfunc
    else:
        _KEEPALIVE[_handle(space)] = cfunc
    native.sb_register_on_rich_realization(_handle(space), cfunc, None)


_KEEPALIVE = {}


# ---------------------------------------------------------------------------
# Receiver input + reevaluation
# ---------------------------------------------------------------------------

def begin_receiver_input(space, actor, root_anchor: str,
                         rig_revision: int = 1,
                         calibration_revision: int = 1) -> int:
    name = root_anchor.encode("utf-8")
    buf = (ctypes.c_uint8 * len(name)).from_buffer_copy(name)
    epoch = ctypes.c_uint64()
    result = _native().sb_begin_receiver_input(
        _handle(space), _handle(actor), _o.RECEIVER_INPUT_FEATURE_VERSION,
        buf, len(name), int(rig_revision), int(calibration_revision),
        ctypes.byref(epoch))
    if result != _o.SB_RICH_RESULT_OK:
        raise RichRealizationError("sb_begin_receiver_input", result)
    return int(epoch.value)


def submit_receiver_input(space, actor, epoch: int, sequence: int,
                          anchors, observed_at_ns: Optional[int] = None,
                          valid_until_ns: Optional[int] = None) -> int:
    """anchors: iterable of (name, available, 16 row-major pose values).
    Timestamps default to now / now + 0.5 s on the native clock."""
    feature = _o.SbReceiverInputFeature()
    _native().sb_get_receiver_input_feature(ctypes.byref(feature),
                                            ctypes.sizeof(feature))
    now = monotonic_ns() if observed_at_ns is None else int(observed_at_ns)
    until = (now + min(500_000_000, int(feature.max_age_ns))
             if valid_until_ns is None else int(valid_until_ns))
    keep, rows = [], []
    for name, available, pose in anchors:
        encoded = str(name).encode("utf-8")
        name_buf = (ctypes.c_uint8 * len(encoded)).from_buffer_copy(encoded)
        keep.append(name_buf)
        row = _o.SbReceiverAnchorInput()
        row.name_utf8 = ctypes.cast(name_buf, ctypes.POINTER(ctypes.c_uint8))
        row.name_bytes = len(encoded)
        row.available = 1 if available else 0
        values = [float(v) for v in pose]
        if len(values) != 16:
            raise ValueError("pose must have 16 row-major values")
        for i, v in enumerate(values):
            row.world_pose_row_major[i] = v
        rows.append(row)
    array = (_o.SbReceiverAnchorInput * max(len(rows), 1))(*rows)
    frame = _o.SbReceiverInputFrame()
    frame.struct_size = ctypes.sizeof(frame)
    frame.feature_version = _o.RECEIVER_INPUT_FEATURE_VERSION
    frame.anchor_count = len(rows)
    frame.anchor_stride = ctypes.sizeof(_o.SbReceiverAnchorInput)
    frame.epoch = int(epoch)
    frame.sequence = int(sequence)
    frame.monotonic_clock_id = feature.monotonic_clock_id
    frame.monotonic_clock_epoch = feature.monotonic_clock_epoch
    frame.observed_at_ns = now
    frame.valid_until_ns = until
    frame.anchors = ctypes.cast(array,
                                ctypes.POINTER(_o.SbReceiverAnchorInput))
    return int(_native().sb_submit_receiver_input(
        _handle(space), _handle(actor), ctypes.byref(frame),
        ctypes.sizeof(frame)))


def clear_receiver_input(space, actor, epoch: int) -> int:
    return int(_native().sb_clear_receiver_input(_handle(space),
                                                 _handle(actor), int(epoch)))


def reevaluate_receiver_realization(token, epoch: int, sequence: int) -> int:
    t = token.token if isinstance(token, RichRealization) else int(token)
    return int(_native().sb_reevaluate_receiver_realization(
        t, _o.RECEIVER_REEVALUATION_FEATURE_VERSION, int(epoch),
        int(sequence)))


# ---------------------------------------------------------------------------
# JSON wire format (byte-compatible with RichWire.Write in C#)
# ---------------------------------------------------------------------------

def _number(value: float) -> str:
    """Unity/mono double.ToString("R", InvariantCulture); see _mono_number.
    (Not Python's shortest repr nor ".17g": mono rounds the 17th digit half
    away from zero, decides G15 with its own lossy parser, prints -0 as 0.)"""
    return _mono_number.format_double(value)


def _string(value: str) -> str:
    out = ['"']
    for ch in value:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _known(name: str, value: Known) -> str:
    text = '"%s":{"known":%s' % (name, "true" if value.known else "false")
    if value.known:
        if value.kind == "double":
            text += ',"value":' + _number(value.value)
        else:
            text += ',"value":"%d"' % int(value.value)
    return text + "}"


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _position(p) -> str:
    return '{"x":%s,"y":%s,"z":%s}' % (_number(p[0]), _number(p[1]),
                                       _number(p[2]))


def _contributor_json(c: RichContributor) -> str:
    parts = [
        _known("referent_id", c.referent_id),
        _known("source_space", c.source_space),
        _known("source_element", c.source_element),
        _known("resolved_local_element", c.resolved_local_element),
        _known("binding_revision", c.binding_revision),
    ]
    text = "{" + ",".join(parts)
    text += ',"snapshot_ordinal":"%d"' % c.snapshot_ordinal
    text += ',"applied_weight":' + _number(c.applied_weight)
    text += ',"has_mapped_support":' + _bool(c.has_mapped_support)
    text += ',"mapped_support":' + _position(c.mapped_support)
    text += ',"grounding_mode":' + _string(c.grounding_mode)
    text += ',"disposition":' + _string(c.disposition)
    text += ',"chart_version_known":' + _bool(c.chart_version_known)
    text += ',"binding_revision_known":' + _bool(c.binding_revision_known)
    text += ',"reason":' + _string(c.reason)
    text += ',"support_identifier":' + _string(c.support_identifier)
    text += ',"support_plane_to_world":{"known":%s,"row_major":[%s]}}' % (
        _bool(c.support_plane_known),
        ",".join(_number(v) for v in c.support_plane_to_world))
    return text


def _channel_json(c: RichChannel) -> str:
    text = "{" + ",".join([
        '"anchor":' + _string(c.anchor),
        '"family":' + _string(c.family),
        '"status":' + _string(c.status),
        '"geometry":' + _string(c.geometry),
        '"goal_kind":' + _string(c.goal_kind),
        '"has_goal":' + _bool(c.has_goal),
        _known("referent_id", c.referent_id),
    ])
    text += ',"arrival_seconds":' + _number(c.arrival_seconds)
    text += ',"admitted_position":' + _position(c.admitted_position)
    text += ',"has_z":' + _bool(c.has_z)
    text += ',"yaw_radians":' + _number(c.yaw_radians)
    text += ',"weight":' + _number(c.weight)
    text += ',"requested_position":{"known":' + _bool(
        c.requested_position_known)
    if c.requested_position_known:
        text += ',"x":%s,"y":%s' % (_number(c.requested_x),
                                    _number(c.requested_y))
    text += '},"reach_projected":' + _bool(c.reach_projected)
    text += ',"reach_displacement_metres":' + _number(
        c.reach_displacement_metres)
    text += ',"contributors":[' + ",".join(
        _contributor_json(x) for x in c.contributors) + "]}"
    return text


def _validate(snapshot: RichSnapshot) -> None:
    h = snapshot.header
    if not (h.native_clock_id.known and h.native_clock_epoch.known and
            h.native_clock_id.value and h.native_clock_epoch.value):
        raise ValueError("rich wire snapshots require a known native clock")
    if h.snapshot_status == "complete" and not (
            h.native_receipt_nanoseconds.known and
            h.native_solve_nanoseconds.known):
        raise ValueError("complete rich snapshots require receipt/solve times")
    if h.snapshot_status == "empty" and snapshot.channels:
        raise ValueError("empty snapshots have no channels")
    keys, ordinals = set(), set()
    for c in snapshot.channels:
        if not c.anchor:
            raise ValueError("channel anchor must be non-empty")
        if (c.anchor, c.family) in keys:
            raise ValueError("channels must be unique by anchor/family")
        keys.add((c.anchor, c.family))
        if c.reach_projected != (c.geometry == "reach_projected"):
            raise ValueError("reach-projected flag and geometry disagree")
        for x in c.contributors:
            if x.snapshot_ordinal in ordinals:
                raise ValueError("contributor ordinals must be unique")
            ordinals.add(x.snapshot_ordinal)
            if x.chart_version_known:
                raise ValueError("v1 cannot carry a known chart version")


_INV = {name: {v: k for k, v in table.items()} for name, table in (
    ("status", _STATUS), ("provenance", _PROVENANCE))}


def _known_from(value, kind):
    if not value["known"]:
        return Known(False, 0.0 if kind == "double" else 0, kind)
    raw = value["value"]
    return Known(True, float(raw) if kind == "double" else int(raw), kind)


def from_rich_json(text) -> RichSnapshot:
    """Load schema-v1 JSON (e.g. a Unity export) into a RichSnapshot, the
    inverse of to_rich_json. Validation is MotionGen's job; this only maps
    fields."""
    import json
    d = json.loads(text)
    if (d.get("schema"), d.get("schema_version")) != (SCHEMA, SCHEMA_VERSION):
        raise ValueError("not an spatialblend.rich_realization v1 document")
    h = d["header"]
    u64 = ("source_epoch", "native_clock_id", "native_clock_epoch",
           "native_receipt_nanoseconds", "native_solve_nanoseconds",
           "policy_revision", "receiver_tree_revision",
           "counterpart_revision", "support_revision")
    dbl = ("source_time_seconds", "source_clock_seconds")
    known = {}
    for key, value in h.items():
        if isinstance(value, dict):
            kind = "u64" if key in u64 else "double" if key in dbl else "i64"
            known[key] = _known_from(value, kind)
    header = RichHeader(
        receiver_space_id=int(h["receiver_space_id"]),
        receiver_actor_id=int(h["receiver_actor_id"]),
        producer_lifetime=int(h["producer_lifetime"]),
        revision=int(h["revision"]),
        snapshot_status=h["snapshot_status"],
        source_provenance_mode=h["source_provenance_mode"],
        known_revision_mask=int(h["known_revision_mask"]),
        unknown_revision_mask=int(h["unknown_revision_mask"]),
        failure_code=int(h["failure_code"]),
        **known)

    def position(p):
        return (float(p["x"]), float(p["y"]), float(p["z"]))

    channels = []
    for c in d["channels"]:
        contributors = []
        for x in c["contributors"]:
            contributors.append(RichContributor(
                referent_id=_known_from(x["referent_id"], "i64"),
                source_space=_known_from(x["source_space"], "i64"),
                source_element=_known_from(x["source_element"], "i64"),
                resolved_local_element=_known_from(
                    x["resolved_local_element"], "i64"),
                binding_revision=_known_from(x["binding_revision"], "u64"),
                snapshot_ordinal=int(x["snapshot_ordinal"]),
                applied_weight=float(x["applied_weight"]),
                has_mapped_support=x["has_mapped_support"],
                mapped_support=position(x["mapped_support"]),
                grounding_mode=x["grounding_mode"],
                disposition=x["disposition"],
                chart_version_known=x["chart_version_known"],
                binding_revision_known=x["binding_revision_known"],
                reason=x["reason"],
                support_identifier=x["support_identifier"],
                support_plane_known=x["support_plane_to_world"]["known"],
                support_plane_to_world=tuple(
                    float(v) for v in x["support_plane_to_world"]["row_major"]),
            ))
        requested = c["requested_position"]
        channels.append(RichChannel(
            anchor=c["anchor"], family=c["family"], status=c["status"],
            geometry=c["geometry"], goal_kind=c["goal_kind"],
            has_goal=c["has_goal"],
            referent_id=_known_from(c["referent_id"], "i64"),
            arrival_seconds=float(c["arrival_seconds"]),
            admitted_position=position(c["admitted_position"]),
            has_z=c["has_z"], yaw_radians=float(c["yaw_radians"]),
            weight=float(c["weight"]),
            requested_position_known=requested["known"],
            requested_x=float(requested.get("x", 0.0)),
            requested_y=float(requested.get("y", 0.0)),
            reach_projected=c["reach_projected"],
            reach_displacement_metres=float(c["reach_displacement_metres"]),
            contributors=tuple(contributors)))
    return RichSnapshot(header=header, channels=tuple(channels))


def to_rich_json(snapshot: RichSnapshot) -> str:
    """Serialize exactly like RichRealizationSnapshot.ToDeterministicJson()."""
    _validate(snapshot)
    h = snapshot.header
    text = ('{"schema":"%s","schema_version":1,"feature_version":1,'
            '"coordinate_system":"%s","units":"%s","header":{' %
            (SCHEMA, COORDINATE_SYSTEM, UNITS))
    header_parts = [
        '"receiver_space_id":"%d"' % h.receiver_space_id,
        '"receiver_actor_id":"%d"' % h.receiver_actor_id,
        '"producer_lifetime":"%d"' % h.producer_lifetime,
        '"revision":"%d"' % h.revision,
        '"snapshot_status":' + _string(h.snapshot_status),
        '"source_provenance_mode":' + _string(h.source_provenance_mode),
        _known("source_epoch", h.source_epoch),
        _known("source_sequence", h.source_sequence),
        _known("source_observation_sequence", h.source_observation_sequence),
        _known("source_actor_space", h.source_actor_space),
        _known("source_actor_element", h.source_actor_element),
        _known("relay_source_space", h.relay_source_space),
        _known("source_time_seconds", h.source_time_seconds),
        _known("source_clock_seconds", h.source_clock_seconds),
        _known("native_clock_id", h.native_clock_id),
        _known("native_clock_epoch", h.native_clock_epoch),
        _known("native_receipt_nanoseconds", h.native_receipt_nanoseconds),
        _known("native_solve_nanoseconds", h.native_solve_nanoseconds),
        _known("policy_revision", h.policy_revision),
        _known("receiver_tree_revision", h.receiver_tree_revision),
        _known("counterpart_revision", h.counterpart_revision),
        _known("support_revision", h.support_revision),
        '"known_revision_mask":"%d"' % h.known_revision_mask,
        '"unknown_revision_mask":"%d"' % h.unknown_revision_mask,
        '"failure_code":"%d"' % h.failure_code,
    ]
    text += ",".join(header_parts) + '},"channels":['
    text += ",".join(_channel_json(c) for c in snapshot.channels) + "]}"
    if len(text.encode("utf-8")) > MAX_JSON_UTF8_BYTES:
        raise ValueError("rich JSON payload exceeds the wire limit")
    return text
