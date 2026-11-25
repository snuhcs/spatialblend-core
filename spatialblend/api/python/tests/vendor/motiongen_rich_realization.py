# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

# Vendored verbatim (below this header): a copy of the MotionGen consumer from
# the motion generator (not public),
# sha256 of the original file: 17bfbe8a7a9301cc10656db7a4219ede5cc49ca0df3775385c9546430bf78eb7.
# Used only by tests/test_capi_rich.py to prove the Python serializer's output
# parses with the MotionGen consumer. Refresh by re-copying; do not edit.
"""Lossless, strict consumer for the Core rich-realization JSON boundary.

This is deliberately a protocol DTO.  It retains admission, grounding and
native-clock diagnostics and does not turn them into ``SpatialControlCueFrame``
or claim that a rig/controller has accepted any channel.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Dict, Iterable, Tuple, Union


SCHEMA = "spatialblend.rich_realization"
SCHEMA_VERSION = 1
FEATURE_VERSION = 1
COORDINATE_SYSTEM = "spatialblend-z-up"
UNITS = "metres"
MAX_INPUT_BYTES = 1048576
MAX_JSON_DEPTH = 32
MAX_CHANNELS = 1024
MAX_CONTRIBUTORS_PER_CHANNEL = 8192
MAX_TOTAL_CONTRIBUTORS = 8192
MAX_STRING_BYTES = 1048576

_UINT = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_INT = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")
_U64_MAX = (1 << 64) - 1
_I64_MIN, _I64_MAX = -(1 << 63), (1 << 63) - 1
_U32_MAX = (1 << 32) - 1

_SNAPSHOT_STATUS = frozenset(("complete", "empty", "failed"))
_SOURCE_PROVENANCE = frozenset(("legacy_unknown", "timestamped_observation", "observation_reset"))
_FAMILY = frozenset(("spacing", "facing", "affordance_pose"))
_SOLVER_STATUS = frozenset(("converged", "fallback", "failed"))
_GEOMETRY = frozenset(("not_applicable", "exact_support", "blended_support", "reach_projected", "legacy_center_non_contact", "unsupported"))
_GOAL_KIND = frozenset(("none", "root_position", "facing_direction", "support_pose", "soft_geometric_pose"))
_GROUNDING = frozenset(("explicit_sub_region", "whole_planar_boundary", "legacy_center_fallback"))
_DISPOSITION = frozenset(("included", "disabled", "unmapped", "grounding_failed", "unsupported"))
_REVISION_WRAPPERS = {
    "policy": "policy_revision", "receiver_tree": "receiver_tree_revision",
    "counterpart": "counterpart_revision", "support": "support_revision",
}


class RichRealizationWireError(ValueError):
    """The payload is not a supported, lossless rich-realization snapshot."""


@dataclass(frozen=True)
class KnownUInt64:
    known: bool
    value: Union[int, None] = None


@dataclass(frozen=True)
class KnownInt64:
    known: bool
    value: Union[int, None] = None


@dataclass(frozen=True)
class KnownDouble:
    known: bool
    value: Union[float, None] = None


@dataclass(frozen=True)
class RichPosition:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class RichRequestedPosition:
    known: bool
    x: Union[float, None]
    y: Union[float, None]


@dataclass(frozen=True)
class RichSupportPlane:
    known: bool
    row_major: Tuple[float, ...]


@dataclass(frozen=True)
class RichRealizationHeader:
    receiver_space_id: int
    receiver_actor_id: int
    producer_lifetime: int
    revision: int
    snapshot_status: str
    source_provenance_mode: str
    source_epoch: KnownUInt64
    source_sequence: KnownInt64
    source_observation_sequence: KnownInt64
    source_actor_space: KnownInt64
    source_actor_element: KnownInt64
    relay_source_space: KnownInt64
    source_time_seconds: KnownDouble
    source_clock_seconds: KnownDouble
    native_clock_id: KnownUInt64
    native_clock_epoch: KnownUInt64
    native_receipt_nanoseconds: KnownUInt64
    native_solve_nanoseconds: KnownUInt64
    policy_revision: KnownUInt64
    receiver_tree_revision: KnownUInt64
    counterpart_revision: KnownUInt64
    support_revision: KnownUInt64
    known_revision_mask: int
    unknown_revision_mask: int
    failure_code: int


@dataclass(frozen=True)
class RichRealizationContributor:
    referent_id: KnownInt64
    source_space: KnownInt64
    source_element: KnownInt64
    resolved_local_element: KnownInt64
    binding_revision: KnownUInt64
    snapshot_ordinal: int
    applied_weight: float
    has_mapped_support: bool
    mapped_support: RichPosition
    grounding_mode: str
    disposition: str
    chart_version_known: bool
    binding_revision_known: bool
    reason: str
    support_identifier: str
    support_plane_to_world: RichSupportPlane


@dataclass(frozen=True)
class RichRealizationChannel:
    anchor: str
    family: str
    status: str
    geometry: str
    goal_kind: str
    has_goal: bool
    referent_id: KnownInt64
    arrival_seconds: float
    admitted_position: RichPosition
    has_z: bool
    yaw_radians: float
    weight: float
    requested_position: RichRequestedPosition
    reach_projected: bool
    reach_displacement_metres: float
    contributors: Tuple[RichRealizationContributor, ...]


@dataclass(frozen=True)
class RichRealizationSnapshot:
    header: RichRealizationHeader
    channels: Tuple[RichRealizationChannel, ...]


def _fail(label: str) -> None:
    raise RichRealizationWireError(label)


def _keys(value: Any, expected: Iterable[str], label: str) -> Dict[str, Any]:
    if type(value) is not dict:
        _fail(label + " must be an object")
    expected_set = frozenset(expected)
    actual = frozenset(value)
    if actual != expected_set:
        _fail(label + " has missing or unknown fields")
    return value


def _object_no_duplicates(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            _fail("duplicate JSON object key: " + key)
        output[key] = value
    return output


def _parse_json_int(token: str) -> int:
    if len(token) > 32:
        _fail("JSON integer token is too long")
    return int(token)


def _parse_json_float(token: str) -> float:
    if len(token) > 128:
        _fail("JSON numeric token is too long")
    value = float(token)
    if not math.isfinite(value):
        _fail("JSON number must be finite")
    return value


def _reject_constant(token: str) -> None:
    _fail("JSON non-finite constant is not permitted: " + token)


def _check_json_depth(text: str) -> None:
    depth = 0
    quoted = False
    escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "{[":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                _fail("JSON nesting exceeds limit")
        elif char in "}]":
            depth -= 1


def _decode(payload: Union[str, bytes, bytearray]) -> Any:
    if isinstance(payload, str):
        # Do not first make a possibly unbounded UTF-8 copy of caller text.
        if len(payload) > MAX_INPUT_BYTES:
            _fail("payload exceeds byte limit")
        byte_count = 0
        try:
            for offset in range(0, len(payload), 4096):
                byte_count += len(payload[offset:offset + 4096].encode("utf-8", "strict"))
                if byte_count > MAX_INPUT_BYTES:
                    _fail("payload exceeds byte limit")
        except UnicodeError as error:
            raise RichRealizationWireError("payload is not valid UTF-8 text") from error
        text = payload
    elif isinstance(payload, (bytes, bytearray)):
        if len(payload) > MAX_INPUT_BYTES:
            _fail("payload exceeds byte limit")
        try:
            text = payload.decode("utf-8", "strict")
        except UnicodeError as error:
            raise RichRealizationWireError("payload is not UTF-8") from error
    else:
        _fail("payload must be str or UTF-8 bytes")
    _check_json_depth(text)
    try:
        return json.loads(text, object_pairs_hook=_object_no_duplicates,
                          parse_int=_parse_json_int, parse_float=_parse_json_float,
                          parse_constant=_reject_constant)
    except RichRealizationWireError:
        raise
    except (ValueError, TypeError, RecursionError) as error:
        raise RichRealizationWireError("invalid JSON") from error


def _string(value: Any, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        _fail(label + " must be a string")
    try:
        encoded = value.encode("utf-8", "strict")
    except UnicodeError as error:
        raise RichRealizationWireError(label + " is not valid UTF-8") from error
    if len(encoded) > MAX_STRING_BYTES or (nonempty and not value):
        _fail(label + " has invalid length")
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        _fail(label + " must be boolean")
    return value


def _number(value: Any, label: str, *, minimum: Union[float, None] = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail(label + " must be a finite JSON number")
    number = float(value)
    if minimum is not None and number < minimum:
        _fail(label + " is out of range")
    return number


def _decimal(value: Any, label: str, *, signed: bool) -> int:
    text = _string(value, label, nonempty=True)
    if len(text) > 20:
        _fail(label + " exceeds canonical 64-bit decimal length")
    if not ( _INT.fullmatch(text) if signed else _UINT.fullmatch(text) ):
        _fail(label + " must be a canonical decimal string")
    number = int(text)
    if (signed and not _I64_MIN <= number <= _I64_MAX) or (not signed and not 0 <= number <= _U64_MAX):
        _fail(label + " exceeds 64-bit range")
    return number


def _u32(value: Any, label: str) -> int:
    number = _decimal(value, label, signed=False)
    if number > _U32_MAX:
        _fail(label + " exceeds uint32 range")
    return number


def _version(value: Any, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        _fail("unsupported " + label)


def _enum(value: Any, allowed: frozenset, label: str) -> str:
    text = _string(value, label, nonempty=True)
    if text not in allowed:
        _fail(label + " is unsupported")
    return text


def _known(value: Any, label: str, kind: str) -> Union[KnownUInt64, KnownInt64, KnownDouble]:
    if type(value) is not dict:
        _fail(label + " must be an explicit known/value wrapper")
    known = value.get("known")
    if type(known) is not bool:
        _fail(label + ".known must be boolean")
    expected = ("known", "value") if known else ("known",)
    _keys(value, expected, label)
    if not known:
        return {"u64": KnownUInt64(False), "i64": KnownInt64(False), "double": KnownDouble(False)}[kind]
    if kind == "u64":
        return KnownUInt64(True, _decimal(value["value"], label + ".value", signed=False))
    if kind == "i64":
        return KnownInt64(True, _decimal(value["value"], label + ".value", signed=True))
    return KnownDouble(True, _number(value["value"], label + ".value"))


def _position(value: Any, label: str) -> RichPosition:
    item = _keys(value, ("x", "y", "z"), label)
    return RichPosition(_number(item["x"], label + ".x"), _number(item["y"], label + ".y"), _number(item["z"], label + ".z"))


def _matrix(value: Any, label: str, *, affine_required: bool) -> Tuple[float, ...]:
    if type(value) is not list or len(value) != 16:
        _fail(label + " must contain exactly 16 values")
    items = tuple(_number(item, label + "[" + str(index) + "]") for index, item in enumerate(value))
    if not affine_required:
        return items
    if any(abs(item) > 1e-9 for item in items[12:15]) or abs(items[15] - 1.0) > 1e-9:
        _fail(label + " must be an affine row-major matrix")
    return items


_HEADER_FIELDS = (
    "receiver_space_id", "receiver_actor_id", "producer_lifetime", "revision", "snapshot_status", "source_provenance_mode",
    "source_epoch", "source_sequence", "source_observation_sequence", "source_actor_space", "source_actor_element", "relay_source_space", "source_time_seconds", "source_clock_seconds",
    "native_clock_id", "native_clock_epoch", "native_receipt_nanoseconds", "native_solve_nanoseconds", "policy_revision", "receiver_tree_revision",
    "counterpart_revision", "support_revision", "known_revision_mask", "unknown_revision_mask", "failure_code",
)


def _header(value: Any) -> RichRealizationHeader:
    item = _keys(value, _HEADER_FIELDS, "header")
    header = RichRealizationHeader(
        _decimal(item["receiver_space_id"], "header.receiver_space_id", signed=True),
        _decimal(item["receiver_actor_id"], "header.receiver_actor_id", signed=True),
        _decimal(item["producer_lifetime"], "header.producer_lifetime", signed=False),
        _decimal(item["revision"], "header.revision", signed=False),
        _enum(item["snapshot_status"], _SNAPSHOT_STATUS, "header.snapshot_status"),
        _enum(item["source_provenance_mode"], _SOURCE_PROVENANCE, "header.source_provenance_mode"),
        _known(item["source_epoch"], "header.source_epoch", "u64"), _known(item["source_sequence"], "header.source_sequence", "i64"),
        _known(item["source_observation_sequence"], "header.source_observation_sequence", "i64"),
        _known(item["source_actor_space"], "header.source_actor_space", "i64"), _known(item["source_actor_element"], "header.source_actor_element", "i64"),
        _known(item["relay_source_space"], "header.relay_source_space", "i64"), _known(item["source_time_seconds"], "header.source_time_seconds", "double"),
        _known(item["source_clock_seconds"], "header.source_clock_seconds", "double"), _known(item["native_clock_id"], "header.native_clock_id", "u64"),
        _known(item["native_clock_epoch"], "header.native_clock_epoch", "u64"), _known(item["native_receipt_nanoseconds"], "header.native_receipt_nanoseconds", "u64"),
        _known(item["native_solve_nanoseconds"], "header.native_solve_nanoseconds", "u64"), _known(item["policy_revision"], "header.policy_revision", "u64"),
        _known(item["receiver_tree_revision"], "header.receiver_tree_revision", "u64"), _known(item["counterpart_revision"], "header.counterpart_revision", "u64"),
        _known(item["support_revision"], "header.support_revision", "u64"), _u32(item["known_revision_mask"], "header.known_revision_mask"),
        _u32(item["unknown_revision_mask"], "header.unknown_revision_mask"), _u32(item["failure_code"], "header.failure_code"))
    if header.producer_lifetime == 0:
        _fail("header.producer_lifetime must be nonzero")
    if header.known_revision_mask & header.unknown_revision_mask:
        _fail("header revision masks overlap")
    return header


_CONTRIBUTOR_FIELDS = (
    "referent_id", "source_space", "source_element", "resolved_local_element", "binding_revision", "snapshot_ordinal", "applied_weight",
    "has_mapped_support", "mapped_support", "grounding_mode", "disposition", "chart_version_known", "binding_revision_known", "reason",
    "support_identifier", "support_plane_to_world",
)


def _contributor(value: Any, label: str) -> RichRealizationContributor:
    item = _keys(value, _CONTRIBUTOR_FIELDS, label)
    support = item["support_plane_to_world"]
    if type(support) is not dict:
        _fail(label + ".support_plane_to_world must be explicit wrapper")
    support_known = support.get("known")
    if type(support_known) is not bool:
        _fail(label + ".support_plane_to_world.known must be boolean")
    _keys(support, ("known", "row_major"), label + ".support_plane_to_world")
    matrix = _matrix(support["row_major"], label + ".support_plane_to_world.row_major", affine_required=support_known)
    if item["chart_version_known"] is True:
        _fail(label + ".chart_version_known is unsupported by lossless v1")
    return RichRealizationContributor(
        _known(item["referent_id"], label + ".referent_id", "i64"), _known(item["source_space"], label + ".source_space", "i64"),
        _known(item["source_element"], label + ".source_element", "i64"), _known(item["resolved_local_element"], label + ".resolved_local_element", "i64"),
        _known(item["binding_revision"], label + ".binding_revision", "u64"), _u32(item["snapshot_ordinal"], label + ".snapshot_ordinal"),
        _number(item["applied_weight"], label + ".applied_weight", minimum=0.0), _bool(item["has_mapped_support"], label + ".has_mapped_support"),
        _position(item["mapped_support"], label + ".mapped_support"), _enum(item["grounding_mode"], _GROUNDING, label + ".grounding_mode"),
        _enum(item["disposition"], _DISPOSITION, label + ".disposition"), _bool(item["chart_version_known"], label + ".chart_version_known"),
        _bool(item["binding_revision_known"], label + ".binding_revision_known"), _string(item["reason"], label + ".reason"),
        _string(item["support_identifier"], label + ".support_identifier"), RichSupportPlane(support_known, matrix))


_CHANNEL_FIELDS = (
    "anchor", "family", "status", "geometry", "goal_kind", "has_goal", "referent_id", "arrival_seconds", "admitted_position", "has_z",
    "yaw_radians", "weight", "requested_position", "reach_projected", "reach_displacement_metres", "contributors",
)


def _channel(value: Any, index: int) -> RichRealizationChannel:
    label = "channels[" + str(index) + "]"
    item = _keys(value, _CHANNEL_FIELDS, label)
    requested = item["requested_position"]
    if type(requested) is not dict:
        _fail(label + ".requested_position must be explicit wrapper")
    requested_known = requested.get("known")
    if type(requested_known) is not bool:
        _fail(label + ".requested_position.known must be boolean")
    _keys(requested, ("known", "x", "y") if requested_known else ("known",), label + ".requested_position")
    contributors = item["contributors"]
    if type(contributors) is not list or len(contributors) > MAX_CONTRIBUTORS_PER_CHANNEL:
        _fail(label + ".contributors exceeds limit or is not an array")
    return RichRealizationChannel(
        _string(item["anchor"], label + ".anchor", nonempty=True), _enum(item["family"], _FAMILY, label + ".family"),
        _enum(item["status"], _SOLVER_STATUS, label + ".status"), _enum(item["geometry"], _GEOMETRY, label + ".geometry"),
        _enum(item["goal_kind"], _GOAL_KIND, label + ".goal_kind"), _bool(item["has_goal"], label + ".has_goal"),
        _known(item["referent_id"], label + ".referent_id", "i64"), _number(item["arrival_seconds"], label + ".arrival_seconds", minimum=0.0),
        _position(item["admitted_position"], label + ".admitted_position"), _bool(item["has_z"], label + ".has_z"),
        _number(item["yaw_radians"], label + ".yaw_radians"), _number(item["weight"], label + ".weight", minimum=0.0),
        RichRequestedPosition(requested_known, _number(requested["x"], label + ".requested_position.x") if requested_known else None,
                              _number(requested["y"], label + ".requested_position.y") if requested_known else None),
        _bool(item["reach_projected"], label + ".reach_projected"), _number(item["reach_displacement_metres"], label + ".reach_displacement_metres", minimum=0.0),
        tuple(_contributor(contributor, label + ".contributors[" + str(contributor_index) + "]") for contributor_index, contributor in enumerate(contributors)))


def _validate_semantics(header: RichRealizationHeader, channels: Tuple[RichRealizationChannel, ...]) -> None:
    if header.revision == 0:
        _fail("header.revision must be nonzero")
    if (header.known_revision_mask | header.unknown_revision_mask) != 0xF:
        _fail("header revision masks must cover all revision domains")
    revision_slots = ((1, header.policy_revision), (2, header.receiver_tree_revision),
                      (4, header.counterpart_revision), (8, header.support_revision))
    for bit, wrapper in revision_slots:
        if wrapper.known != bool(header.known_revision_mask & bit):
            _fail("header revision mask disagrees with wrapper knownness")
    if header.source_provenance_mode == "timestamped_observation":
        if not (header.source_epoch.known and header.source_epoch.value > 0 and header.source_sequence.known and header.source_sequence.value > 0 and header.source_time_seconds.known and header.source_time_seconds.value >= 0):
            _fail("timestamped observation requires epoch, positive relation sequence, and source time")
        if header.source_observation_sequence.known and header.source_observation_sequence.value < 0:
            _fail("known source sample sequence must be nonnegative")
    elif header.source_provenance_mode == "observation_reset":
        if not (header.source_epoch.known and header.source_epoch.value > 0 and header.source_sequence.known and header.source_sequence.value > 0 and not header.source_observation_sequence.known):
            _fail("observation reset requires epoch/positive relation sequence and unknown sample sequence")
    elif header.source_observation_sequence.known:
        _fail("legacy unknown provenance must not claim a source sample sequence")
    if header.source_clock_seconds.known and header.source_clock_seconds.value <= 0:
        _fail("known source clock seconds must be positive")
    if not (header.native_clock_id.known and header.native_clock_epoch.known):
        _fail("native clock id and epoch wrappers must be known")
    if header.native_clock_id.value == 0 or header.native_clock_epoch.value == 0:
        _fail("native clock id and epoch must be nonzero")
    receipt_known = header.native_receipt_nanoseconds.known
    solve_known = header.native_solve_nanoseconds.known
    if receipt_known != solve_known:
        _fail("native receipt and solve knownness must be paired")
    if header.snapshot_status == "complete" and not receipt_known:
        _fail("complete snapshot requires known native receipt and solve times")
    if receipt_known and header.native_receipt_nanoseconds.value > header.native_solve_nanoseconds.value:
        _fail("native receipt must not follow native solve")
    if header.failure_code not in (0, 1) or (header.failure_code == 1 and header.snapshot_status != "failed"):
        _fail("failure code is incompatible with snapshot status")
    contributor_ordinals = set()
    for channel in channels:
        if channel.reach_projected != (channel.geometry == "reach_projected"):
            _fail("reach projection flag and geometry disagree")
        for contributor in channel.contributors:
            if contributor.snapshot_ordinal in contributor_ordinals:
                _fail("contributor snapshot ordinal must be unique snapshot-wide")
            contributor_ordinals.add(contributor.snapshot_ordinal)
            if contributor.binding_revision_known != contributor.binding_revision.known:
                _fail("contributor binding revision known flag disagrees with wrapper")


def parse_rich_realization(payload: Union[str, bytes, bytearray]) -> RichRealizationSnapshot:
    """Parse exactly schema v1 and return a detached, deeply immutable DTO."""
    root = _keys(_decode(payload), ("schema", "schema_version", "feature_version", "coordinate_system", "units", "header", "channels"), "envelope")
    if _string(root["schema"], "schema") != SCHEMA:
        _fail("unsupported rich-realization schema or version")
    _version(root["schema_version"], SCHEMA_VERSION, "schema_version")
    _version(root["feature_version"], FEATURE_VERSION, "feature_version")
    if _string(root["coordinate_system"], "coordinate_system") != COORDINATE_SYSTEM or _string(root["units"], "units") != UNITS:
        _fail("unsupported coordinate system or units")
    header = _header(root["header"])
    channels = root["channels"]
    if type(channels) is not list or len(channels) > MAX_CHANNELS:
        _fail("channels exceeds limit or is not an array")
    contributor_total = 0
    for channel in channels:
        if type(channel) is dict and type(channel.get("contributors")) is list:
            contributor_total += len(channel["contributors"])
            if contributor_total > MAX_TOTAL_CONTRIBUTORS:
                _fail("contributors exceeds negotiated global limit")
    parsed_channels = tuple(_channel(channel, index) for index, channel in enumerate(channels))
    keys = tuple((channel.anchor, channel.family) for channel in parsed_channels)
    if len(keys) != len(set(keys)):
        _fail("duplicate (anchor, family) channel")
    if header.snapshot_status == "empty" and parsed_channels:
        _fail("empty snapshot must have no channels")
    _validate_semantics(header, parsed_channels)
    return RichRealizationSnapshot(header, parsed_channels)


def require_known_revisions(snapshot: RichRealizationSnapshot, required_revisions: Iterable[str]) -> None:
    """Fail a strict caller explicitly requesting revision domains not present.

    This remains a protocol validation helper, not motion/owner admission.
    Callers that only display diagnostics may parse snapshots with unknown
    revision domains without invoking it.
    """
    if not isinstance(snapshot, RichRealizationSnapshot):
        _fail("revision validation requires RichRealizationSnapshot")
    for domain in required_revisions:
        if domain not in _REVISION_WRAPPERS:
            _fail("unsupported required revision domain")
        if not getattr(snapshot.header, _REVISION_WRAPPERS[domain]).known:
            _fail("required revision domain is unknown: " + domain)


def _known_wire(value: Union[KnownUInt64, KnownInt64, KnownDouble]) -> Dict[str, Any]:
    if not value.known:
        return {"known": False}
    encoded = value.value if isinstance(value, KnownDouble) else str(value.value)
    return {"known": True, "value": encoded}


def _position_wire(value: RichPosition) -> Dict[str, float]:
    return {"x": value.x, "y": value.y, "z": value.z}


def _contributor_wire(value: RichRealizationContributor) -> Dict[str, Any]:
    support: Dict[str, Any] = {"known": value.support_plane_to_world.known,
                               "row_major": list(value.support_plane_to_world.row_major)}
    return {"referent_id": _known_wire(value.referent_id), "source_space": _known_wire(value.source_space), "source_element": _known_wire(value.source_element), "resolved_local_element": _known_wire(value.resolved_local_element), "binding_revision": _known_wire(value.binding_revision), "snapshot_ordinal": str(value.snapshot_ordinal), "applied_weight": value.applied_weight, "has_mapped_support": value.has_mapped_support, "mapped_support": _position_wire(value.mapped_support), "grounding_mode": value.grounding_mode, "disposition": value.disposition, "chart_version_known": value.chart_version_known, "binding_revision_known": value.binding_revision_known, "reason": value.reason, "support_identifier": value.support_identifier, "support_plane_to_world": support}


def _channel_wire(value: RichRealizationChannel) -> Dict[str, Any]:
    requested: Dict[str, Any] = {"known": value.requested_position.known}
    if value.requested_position.known:
        requested.update({"x": value.requested_position.x, "y": value.requested_position.y})
    return {"anchor": value.anchor, "family": value.family, "status": value.status, "geometry": value.geometry, "goal_kind": value.goal_kind, "has_goal": value.has_goal, "referent_id": _known_wire(value.referent_id), "arrival_seconds": value.arrival_seconds, "admitted_position": _position_wire(value.admitted_position), "has_z": value.has_z, "yaw_radians": value.yaw_radians, "weight": value.weight, "requested_position": requested, "reach_projected": value.reach_projected, "reach_displacement_metres": value.reach_displacement_metres, "contributors": [_contributor_wire(contributor) for contributor in value.contributors]}


def serialize_rich_realization(snapshot: RichRealizationSnapshot) -> str:
    """Return deterministic schema-v1 JSON without converting 64-bit values to JSON numbers."""
    if not isinstance(snapshot, RichRealizationSnapshot):
        _fail("serialize requires RichRealizationSnapshot")
    header = snapshot.header
    wire = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "feature_version": FEATURE_VERSION, "coordinate_system": COORDINATE_SYSTEM, "units": UNITS,
            "header": {"receiver_space_id": str(header.receiver_space_id), "receiver_actor_id": str(header.receiver_actor_id), "producer_lifetime": str(header.producer_lifetime), "revision": str(header.revision), "snapshot_status": header.snapshot_status, "source_provenance_mode": header.source_provenance_mode, "source_epoch": _known_wire(header.source_epoch), "source_sequence": _known_wire(header.source_sequence), "source_observation_sequence": _known_wire(header.source_observation_sequence), "source_actor_space": _known_wire(header.source_actor_space), "source_actor_element": _known_wire(header.source_actor_element), "relay_source_space": _known_wire(header.relay_source_space), "source_time_seconds": _known_wire(header.source_time_seconds), "source_clock_seconds": _known_wire(header.source_clock_seconds), "native_clock_id": _known_wire(header.native_clock_id), "native_clock_epoch": _known_wire(header.native_clock_epoch), "native_receipt_nanoseconds": _known_wire(header.native_receipt_nanoseconds), "native_solve_nanoseconds": _known_wire(header.native_solve_nanoseconds), "policy_revision": _known_wire(header.policy_revision), "receiver_tree_revision": _known_wire(header.receiver_tree_revision), "counterpart_revision": _known_wire(header.counterpart_revision), "support_revision": _known_wire(header.support_revision), "known_revision_mask": str(header.known_revision_mask), "unknown_revision_mask": str(header.unknown_revision_mask), "failure_code": str(header.failure_code)}, "channels": [_channel_wire(channel) for channel in snapshot.channels]}
    try:
        encoded = json.dumps(wire, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise RichRealizationWireError("snapshot contains non-serializable fields") from error
    # Dataclasses are intentionally plain immutable values; validate manually
    # constructed values too, not just results produced by this parser.
    parse_rich_realization(encoded)
    return encoded


loads_rich_realization = parse_rich_realization
dumps_rich_realization = serialize_rich_realization


__all__ = ["COORDINATE_SYSTEM", "FEATURE_VERSION", "KnownDouble", "KnownInt64", "KnownUInt64", "MAX_CHANNELS", "MAX_CONTRIBUTORS_PER_CHANNEL", "MAX_INPUT_BYTES", "MAX_TOTAL_CONTRIBUTORS", "RichPosition", "RichRequestedPosition", "RichRealizationChannel", "RichRealizationContributor", "RichRealizationHeader", "RichRealizationSnapshot", "RichRealizationWireError", "RichSupportPlane", "SCHEMA", "SCHEMA_VERSION", "UNITS", "dumps_rich_realization", "loads_rich_realization", "parse_rich_realization", "require_known_revisions", "serialize_rich_realization"]
