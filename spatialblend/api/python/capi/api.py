# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Thin, Pythonic layer over the spatialblend C API (see _api.py / _types.py).

Z-UP passthrough: every coordinate, transform, and rotation accepted or
returned here is passed straight through to/from the native sb_* functions
with no conversion of any kind (see spatialblend/api/c/spatialblend.h's
"COORDINATE SYSTEM CONVENTIONS" block). Callers are responsible for already
being in the Z-UP convention documented there.

Typical usage::

    from spatialblend.api.python.capi import api

    with api.Session() as session:
        space = session.add_space()
        room = space.add_element()
        room.set_polygon(api.GeometryRole.NAVIGABLE, room_points_xy)

        desk = space.add_element(parent=room)
        desk.set_shared_id(200)
        desk.set_tag("Desk")
        desk.set_collider(api.ColliderType.BOX, [5, 5, 0.5, 2, 2, 1])

        actor = space.create_actor()
        actor.add_anchor("root", 0, 0, 0)
        actor.set_element(room)
        actor.bind("root", api.KernelType.SPACING)

        actor.update_state(3.0, 5.0, 0.0)
        targets = space.poll_realization_targets(actor)
"""
import ctypes
import enum
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

from . import _api
from . import _obs_types as _o
from . import observability as _obs
from ._types import (
    SB_COLLIDER_BOX,
    SB_COLLIDER_CAPSULE,
    SB_COLLIDER_SPHERE,
    SB_GEOMETRY_CONTAINMENT,
    SB_GEOMETRY_INTERACTION_REGION,
    SB_GEOMETRY_NAVIGABLE,
    SB_GEOMETRY_OBSTACLE,
    SB_GEOMETRY_SURFACE,
    SB_GEOMETRY_UNSPECIFIED,
    SB_KERNEL_AFFORDANCE_POSE,
    SB_KERNEL_FACING,
    SB_KERNEL_SPACING,
    SB_ORIENTATION_YAW,
    SB_ORIENTATION_YAW_PITCH,
    SB_ORIENTATION_YAW_PITCH_ROLL,
    SB_PHASE_ACTIVE,
    SB_PHASE_CONTACT,
    SB_PHASE_DORMANT,
    SB_PHASE_DRIVING,
    SB_SCENE_CONTAINMENT_ESCAPES_PARENT,
    SB_SCENE_CONTAINMENT_OVERLAP,
    SB_SCENE_INVALID_ARGUMENT,
    SB_SCENE_OK,
)

__all__ = [
    "GeometryRole", "KernelType", "ColliderType", "Proposal", "AnchorProposal",
    "ActorEvent",
    "RelationPhase", "ActorState", "RealizationTarget", "RelationEvent",
    "ElementAddedEvent", "AsyncHandle", "Session", "Space", "Element",
    "Actor", "cleanup_all", "RealizationPolicy", "ReferentResolveMode",
    "SceneStatus", "OrientationType",
    "abi_version", "abi_minor_version", "v2_abi_layout", "sizeof",
]


class RealizationPolicy(enum.IntEnum):
    """sb_realization_policy_t."""
    PROGRESSIVE = 0
    UNIFORM = 1
    GREEDY = 2


class ReferentResolveMode(enum.IntEnum):
    """sb_referent_resolve_mode_t."""
    LINKER = 0
    SHAREDID = 1
    LINKER_ONLY = 2


class GeometryRole(enum.IntEnum):
    UNSPECIFIED = SB_GEOMETRY_UNSPECIFIED
    NAVIGABLE = SB_GEOMETRY_NAVIGABLE
    CONTAINMENT = SB_GEOMETRY_CONTAINMENT
    OBSTACLE = SB_GEOMETRY_OBSTACLE
    SURFACE = SB_GEOMETRY_SURFACE
    INTERACTION_REGION = SB_GEOMETRY_INTERACTION_REGION


class KernelType(enum.IntEnum):
    SPACING = SB_KERNEL_SPACING
    FACING = SB_KERNEL_FACING
    AFFORDANCE_POSE = SB_KERNEL_AFFORDANCE_POSE


class ColliderType(enum.IntEnum):
    BOX = SB_COLLIDER_BOX
    SPHERE = SB_COLLIDER_SPHERE
    CAPSULE = SB_COLLIDER_CAPSULE


class RelationPhase(enum.IntEnum):
    DORMANT = SB_PHASE_DORMANT
    ACTIVE = SB_PHASE_ACTIVE
    DRIVING = SB_PHASE_DRIVING
    CONTACT = SB_PHASE_CONTACT


class SceneStatus(enum.IntEnum):
    """sb_scene_status_t. A mutator that is not OK is not registered."""
    OK = SB_SCENE_OK
    INVALID_ARGUMENT = SB_SCENE_INVALID_ARGUMENT
    CONTAINMENT_OVERLAP = SB_SCENE_CONTAINMENT_OVERLAP
    CONTAINMENT_ESCAPES_PARENT = SB_SCENE_CONTAINMENT_ESCAPES_PARENT


class OrientationType(enum.IntEnum):
    """sb_orientation_type_t (ABI 3.1). Pitch and roll are pass-through."""
    YAW = SB_ORIENTATION_YAW
    YAW_PITCH = SB_ORIENTATION_YAW_PITCH
    YAW_PITCH_ROLL = SB_ORIENTATION_YAW_PITCH_ROLL


@dataclass
class ActorState:
    """Mirrors sb_actor_state_t."""
    position_x: float
    position_y: float
    yaw: float


@dataclass
class AnchorProposal:
    """Mirrors sb_anchor_proposal_t (ABI 4): the proposal of ONE anchor.
    ``status`` is an SB_PROPOSAL_* value (NO_SAMPLE = publish a pose first),
    ``cause`` an SB_PROPOSAL_CAUSE_* bitmask. err_* are |R_tgt - R_src| at the
    target and at the sampled pose (-1 = none); multiply err_spacing_* by
    err_scale_m for metres."""
    anchor_id: int
    status: int
    cause: int
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float
    orientation_type: int
    yaw_specified: bool
    engaged: bool
    sampled: bool
    arrived: bool
    position_specified: bool
    err_spacing_target: float
    err_spacing_sample: float
    err_facing_target: float
    err_facing_sample: float
    err_pose_target: float
    err_pose_sample: float
    err_scale_m: float
    referent_shared_id: int


@dataclass
class Proposal:
    """One proposal SET (ABI 4): one proposal per relation-bearing anchor of a
    receiver actor, all from one solve. ``anchors[0]`` is the root anchor; its
    fields are also reachable as attributes of the set (``p.x``, ``p.status`` ...)
    for callers that only drive the root."""
    revision: int
    source_sequence: int
    sample_valid: bool
    sample_sequence: int
    sample_epoch: int
    sample_timestamp_s: float
    sample_age_s: float
    anchors: list

    @property
    def root(self) -> AnchorProposal:
        return self.anchors[0]

    def __getattr__(self, name):
        # Only called for names the dataclass does not define: the root anchor's.
        anchors = self.__dict__.get("anchors")
        if anchors:
            return getattr(anchors[0], name)
        raise AttributeError(name)


@dataclass
class ActorEvent:
    """Mirrors sb_actor_event_t (ABI 4)."""
    kind: int
    revision: int
    source_sequence: int
    source_shared_id: int
    exception_id: int
    payload: bytes
    interaction_kind: int
    anchor_id: int
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float


@dataclass
class RealizationTarget:
    """Mirrors sb_realization_target_t (a snapshot copy -- the native struct
    and its anchor_name pointer are only valid during the callback, so this
    dataclass is what survives past that window)."""
    arrival_time: float
    anchor_name: str
    position_x: float
    position_y: float
    position_z: float
    has_z: bool
    yaw: float
    weight: float
    kernel_type: "KernelType"
    referent_id: int


@dataclass
class RelationEvent:
    """Mirrors sb_relation_event_t."""
    actor_shared_id: int
    anchor_name: str
    referent_shared_id: int
    kernel_type: "KernelType"
    from_phase: "RelationPhase"
    to_phase: "RelationPhase"
    timestamp: float


@dataclass
class ElementAddedEvent:
    """Args delivered by sb_on_element_added_t, minus the space/cb_user_data
    plumbing arguments (the Space that received it is implicit: it's the one
    whose callback fired)."""
    local_element: "Element"
    shared_id: int
    tag: Optional[str]
    user_data: Optional[bytes]


@dataclass
class AsyncHandle:
    """Wraps an sb_handle_t returned by an async C API call."""
    value: int

    def try_wait(self, timeout_ms: int = 0) -> bool:
        return bool(_api.get_native().sb_try_wait(self.value, timeout_ms))

    def get_error(self) -> Optional[str]:
        native = _api.get_native()
        err = native.sb_get_error(self.value)
        return err.decode("utf-8") if err is not None else None


def _flatten_row_major(matrix4x4: Sequence[Sequence[float]]) -> "ctypes.Array":
    """Row-major flatten of a 4x4 matrix, for sb_set_element_transform() and
    sb_add_sub_region()'s transform16 (both documented row-major)."""
    flat = _as_flat16(matrix4x4)
    return (ctypes.c_double * 16)(*flat)


def _flatten_col_major(matrix4x4: Sequence[Sequence[float]]) -> "ctypes.Array":
    """Column-major flatten, for sb_add_element_anchor() (documented
    column-major / Eigen layout -- see spatialblend.h:291-293)."""
    rows = _as_rows4(matrix4x4)
    flat = [rows[r][c] for c in range(4) for r in range(4)]
    return (ctypes.c_double * 16)(*flat)


def _as_rows4(matrix4x4):
    """Accept either a flat sequence of 16 floats or 4 sequences of 4."""
    items = list(matrix4x4)
    if len(items) == 16 and not isinstance(items[0], (list, tuple)) and not hasattr(items[0], "__len__"):
        return [items[0:4], items[4:8], items[8:12], items[12:16]]
    if len(items) == 4:
        return [list(row) for row in items]
    raise ValueError(
        "expected a 4x4 matrix (4 rows of 4, or a flat sequence of 16), "
        f"got {len(items)} top-level items")


def _as_flat16(matrix4x4):
    rows = _as_rows4(matrix4x4)
    flat = []
    for row in rows:
        flat.extend(row)
    if len(flat) != 16:
        raise ValueError(f"expected 16 doubles, got {len(flat)}")
    return flat


def _identity16_row_major():
    return (ctypes.c_double * 16)(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)


def _double_array(values: Sequence[float]) -> "ctypes.Array":
    return (ctypes.c_double * len(values))(*[float(v) for v in values])


def _int32_array(values: Sequence[int]) -> "ctypes.Array":
    return (ctypes.c_int32 * len(values))(*[int(v) for v in values])


def _target_from_native(t) -> RealizationTarget:
    return RealizationTarget(
        arrival_time=t.arrival_time,
        anchor_name=(t.anchor_name.decode("utf-8") if t.anchor_name else ""),
        position_x=t.position_x,
        position_y=t.position_y,
        position_z=t.position_z,
        has_z=bool(t.has_z),
        yaw=t.yaw,
        weight=t.weight,
        kernel_type=KernelType(t.kernel_type),
        referent_id=t.referent_id,
    )


def _relation_event_from_native(e) -> RelationEvent:
    return RelationEvent(
        actor_shared_id=e.actor_shared_id,
        anchor_name=(e.anchor_name.decode("utf-8") if e.anchor_name else ""),
        referent_shared_id=e.referent_shared_id,
        kernel_type=KernelType(e.kernel_type),
        from_phase=RelationPhase(e.from_phase),
        to_phase=RelationPhase(e.to_phase),
        timestamp=e.timestamp,
    )


class Session:
    """Owns a native sb_session_t and every Space/callback created under it.

    Callback trampolines (ctypes.CFUNCTYPE instances) are kept alive here --
    on the Session -- for as long as the session lives, since ctypes holds
    no reference of its own and a GC'd CFUNCTYPE crashes the process the
    next time native code tries to invoke it.
    """

    def __init__(self):
        self._native = _api.get_native()
        self.handle = self._native.sb_create_session()
        if self.handle == 0:
            raise RuntimeError("sb_create_session() failed")
        self._spaces: List["Space"] = []
        # (space_handle, kind) -> CFUNCTYPE instance, keeps them alive.
        self._callback_refs = {}
        self._closed = False

    def add_space(self) -> "Space":
        handle = self._native.sb_add_space(self.handle)
        if handle == 0:
            raise RuntimeError("sb_add_space() failed")
        space = Space(self, handle)
        self._spaces.append(space)
        return space

    def enable_actor_observation_frames(self) -> None:
        """sb_enable_actor_observation_frames (observation-frame mode)."""
        if self._native.sb_enable_actor_observation_frames(self.handle) != 1:
            raise RuntimeError("sb_enable_actor_observation_frames failed")

    def register_support_correspondences(self, pairs) -> int:
        """pairs: iterable of (source_space, source_element, target_space,
        target_element) handles or Space/Element objects. Returns the
        registration handle (dispose with dispose_support_correspondences)."""
        rows = []
        for ss, se, ts, te in pairs:
            rows.append(_o.SbSupportCorrespondence(
                _space_id(ss), _elem_id(se), _space_id(ts), _elem_id(te)))
        buf = (_o.SbSupportCorrespondence * max(len(rows), 1))(*rows)
        handle = self._native.sb_register_support_correspondences(
            self.handle, buf, len(rows))
        if handle == 0:
            raise RuntimeError("sb_register_support_correspondences failed")
        return int(handle)

    def dispose_support_correspondences(self, registration: int) -> None:
        self._native.sb_dispose_support_correspondences(int(registration))

    def close(self):
        if self._closed:
            return
        self._native.sb_destroy_session(self.handle)
        self._closed = True
        # Keep callback refs (and thus the CFUNCTYPEs) alive until here;
        # after destroy_session native code will not call back into them.
        self._callback_refs.clear()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Space:
    """Wraps an sb_space_t handle."""

    def __init__(self, session: Session, handle: int):
        self.session = session
        self.handle = handle
        self._native = session._native
        # actor_id -> list[RealizationTarget], filled by the internal
        # sb_on_targets_t trampoline registered lazily by poll_realization_targets().
        self._last_targets = {}
        self._on_element_added_user_cb = None
        self._on_relation_event_user_cb = None

    # -- networking -----------------------------------------------------
    def host(self, port: int = 0) -> AsyncHandle:
        return AsyncHandle(self._native.sb_host_space(self.handle, port))

    def connect(self, host: str, port: int) -> AsyncHandle:
        return AsyncHandle(
            self._native.sb_connect_space(
                self.handle, host.encode("utf-8"), port))

    # -- receiver geometry / engagement ----------------------------------
    def set_body_radius(self, radius_m: float) -> None:
        """sb_set_body_radius: the space-wide floor-footprint radius of the actors that register no embodiment. The
        radius belongs to the driver: the core's free space has no body-radius erosion; sb_query_course keeps a body of
        the actor's radius clear (0 when none is registered)."""
        if self._native.sb_set_body_radius(self.handle, float(radius_m)) != 0:
            raise ValueError("invalid body radius %r" % (radius_m,))

    # -- elements ---------------------------------------------------------
    def add_element(self, parent: Union["Element", int, None] = None) -> "Element":
        parent_id = _elem_id(parent)
        elem_id = self._native.sb_add_element(self.handle, parent_id)
        if elem_id == 0:
            raise RuntimeError("sb_add_element() failed")
        return Element(self, elem_id)

    def find_element_by_shared_id(self, shared_id: int) -> Optional["Element"]:
        elem_id = self._native.sb_find_element_by_shared_id(
            self.handle, shared_id)
        return Element(self, elem_id) if elem_id else None

    def element(self, elem_id: int) -> "Element":
        """Wrap a raw element id you already have (e.g. from an
        ElementAddedEvent) without re-resolving it."""
        return Element(self, elem_id)

    # -- actors -------------------------------------------------------------
    def create_actor(self) -> "Actor":
        actor_id = self._native.sb_create_actor(self.handle)
        if actor_id == 0:
            raise RuntimeError("sb_create_actor() failed")
        return Actor(self, actor_id)

    def find_actor_by_shared_id(self, shared_id: int) -> Optional["Actor"]:
        actor_id = self._native.sb_find_actor_by_shared_id(
            self.handle, shared_id)
        return Actor(self, actor_id) if actor_id else None

    def actor(self, actor_id: int) -> "Actor":
        return Actor(self, actor_id)

    # -- callbacks / realization targets --------------------------------
    #
    # NOTE: sb_register_on_targets() installs a single function pointer per
    # space (native "last registration wins" semantics -- there is no
    # per-caller multiplexing on the C side). poll_realization_targets() and
    # register_on_targets() below therefore always (re)install their own
    # trampoline before use rather than memoizing "already registered", so
    # whichever of the two was called most recently is the one that is
    # actually active -- mixing them on the same Space is legal but the
    # second call wins, exactly like calling sb_register_on_targets() twice
    # from C would.
    def _install_capturing_trampoline(self):
        def _trampoline(space, actor, targets_ptr, num_targets, user_data):
            targets = [
                _target_from_native(targets_ptr[i])
                for i in range(num_targets)
            ]
            self._last_targets[actor] = targets

        cfunc = _api.SB_ON_TARGETS_FUNC(_trampoline)
        self.session._callback_refs[(self.handle, "on_targets")] = cfunc
        self._native.sb_register_on_targets(self.handle, cfunc, None)

    def poll_realization_targets(self, actor: Union["Actor", int]) -> List[RealizationTarget]:
        """sb_poll_realization_targets() + capture via an internally-managed
        sb_on_targets_t callback. Returns [] if nothing was dispatched."""
        self._install_capturing_trampoline()
        actor_id = _actor_id(actor)
        self._last_targets.pop(actor_id, None)
        dispatched = self._native.sb_poll_realization_targets(
            self.handle, actor_id)
        if not dispatched:
            return []
        return self._last_targets.pop(actor_id, [])

    def register_on_targets(self, callback):
        """Low-level escape hatch: register your own callback(actor_id,
        list[RealizationTarget]) instead of using poll_realization_targets().
        Replaces any callback previously installed via either this method or
        poll_realization_targets() (see the native "last registration wins"
        note above) -- do not call poll_realization_targets() on this Space
        afterwards, since that will in turn replace this callback."""
        def _trampoline(space, actor, targets_ptr, num_targets, user_data):
            targets = [
                _target_from_native(targets_ptr[i])
                for i in range(num_targets)
            ]
            callback(actor, targets)

        cfunc = _api.SB_ON_TARGETS_FUNC(_trampoline)
        self.session._callback_refs[(self.handle, "on_targets")] = cfunc
        self._native.sb_register_on_targets(self.handle, cfunc, None)

    def register_on_element_added(self, callback):
        """callback(ElementAddedEvent) -> None, invoked when a peer-shared
        element arrives in this space."""
        def _trampoline(space, local_element, shared_id, tag, user_data,
                        user_data_size, cb_user_data):
            payload = None
            if user_data and user_data_size:
                payload = ctypes.string_at(user_data, user_data_size)
            callback(ElementAddedEvent(
                local_element=Element(self, local_element),
                shared_id=shared_id,
                tag=(tag.decode("utf-8") if tag else None),
                user_data=payload,
            ))

        cfunc = _api.SB_ON_ELEMENT_ADDED_FUNC(_trampoline)
        self.session._callback_refs[(self.handle, "on_element_added")] = cfunc
        self._native.sb_register_on_element_added(self.handle, cfunc, None)

    def register_on_relation_event(self, callback):
        """callback(RelationEvent) -> None, invoked on each lifecycle phase
        transition (dormant -> active -> driving -> contact)."""
        def _trampoline(event_ptr, user_data):
            callback(_relation_event_from_native(event_ptr.contents))

        cfunc = _api.SB_ON_RELATION_EVENT_FUNC(_trampoline)
        self.session._callback_refs[(self.handle, "on_relation_event")] = cfunc
        self._native.sb_register_on_relation_event(self.handle, cfunc, None)

    def resolve_shared_element(self, received: Union["Element", int],
                                resolved: Union["Element", int]):
        self._native.sb_resolve_shared_element(
            self.handle, _elem_id(received), _elem_id(resolved))

    def set_update_interval(self, seconds: float):
        self._native.sb_set_update_interval(self.handle, seconds)

    def register_on_element_removed(self, callback):
        """callback(element_id) -> None after a received replica is removed."""
        def _trampoline(space, local_element, user_data):
            callback(local_element)

        cfunc = _api.SB_ON_ELEMENT_REMOVED_FUNC(_trampoline)
        self.session._callback_refs[(self.handle, "on_element_removed")] = cfunc
        self._native.sb_register_on_element_removed(self.handle, cfunc, None)

    def register_on_link_removed(self, callback):
        """callback(element_id, peer_space, reason) -> None when the peer
        removed a link whose end here is a surviving element (reason is
        SB_LINK_REMOVED_PEER_UNSHARED=1 or PEER_UNLINKED=2)."""
        def _trampoline(space, local_element, peer_space, reason, user_data):
            callback(local_element, peer_space, reason)

        cfunc = _api.SB_ON_LINK_REMOVED_FUNC(_trampoline)
        self.session._callback_refs[(self.handle, "on_link_removed")] = cfunc
        self._native.sb_register_on_link_removed(self.handle, cfunc, None)

    # -- Runtime configuration / tracing ---------------------------------
    def set_trace(self, path: Optional[str]) -> None:
        """sb_set_trace: bind this space's JSONL trace to ``path`` (append),
        or disable it with None. Format: spatialblend/api/c/trace.format.md."""
        self._native.sb_set_trace(
            self.handle, None if path is None else str(path).encode("utf-8"))

    def set_realization_policy(self, policy: "RealizationPolicy") -> None:
        self._native.sb_set_realization_policy(self.handle, int(policy))

    def set_referent_resolve_mode(self, mode: "ReferentResolveMode") -> None:
        self._native.sb_set_referent_resolve_mode(self.handle, int(mode))

    def set_weight_smoothing(self, alpha: float) -> None:
        self._native.sb_set_weight_smoothing(self.handle, float(alpha))

    def set_min_move(self, meters: float) -> None:
        self._native.sb_set_min_move(self.handle, float(meters))

    def set_min_yaw(self, radians: float) -> None:
        self._native.sb_set_min_yaw(self.handle, float(radians))

    def set_anchor_reach(self, anchor_name: str, radius: float) -> None:
        self._native.sb_set_anchor_reach(
            self.handle, anchor_name.encode("utf-8"), float(radius))

    def set_observability_capture(self, enabled: bool = True) -> None:
        """Opt in to per-evaluation observation capture (see
        capi.observability; tracing captures regardless)."""
        _obs.enable_capture(self.handle, enabled)

    # -- declared policy parameters ---------------------------------------
    def policy_params(self) -> "_obs.PolicyParams":
        return _obs.policy_params(self.handle)

    def set_policy_params(self, params=None, **changes) -> bool:
        return _obs.set_policy_params(self.handle, params, **changes)

    # -- per-family relation activation/scope thresholds -------------------
    def relation_phase_config(self, kernel_type: "KernelType") -> dict:
        return _obs.relation_phase_config(self.handle, int(kernel_type))

    def set_relation_phase_config(self, kernel_type: "KernelType",
                                  config: dict = None, **changes) -> dict:
        return _obs.set_relation_phase_config(
            self.handle, int(kernel_type), config, **changes)

    # -- pure relation evaluation ------------------------------------------
    def evaluate_relation(self, kernel_type: "KernelType", anchor_xyz,
                          anchor_yaw: float, referent, scope_root=None,
                          position_dimension: int = 2,
                          explicit_scope: bool = False):
        return _obs.evaluate_relation(
            self.handle, int(kernel_type), anchor_xyz, anchor_yaw,
            _elem_id(referent), _elem_id(scope_root), position_dimension,
            explicit_scope)

    # -- target-side Eval_p ------------------------------------------------
    def evaluate_target_relation(self, payload: "_obs.RelationPayload",
                                 anchor_xyz, anchor_yaw: float,
                                 normalizing_scope=None):
        """R^tgt of a received relation payload on this (receiver) space at
        a candidate anchor pose (sb_evaluate_target_relation)."""
        return _obs.evaluate_target_relation(
            self.handle, payload, anchor_xyz, anchor_yaw,
            _elem_id(normalizing_scope))


def _elem_id(elem):
    if elem is None:
        return 0
    return elem.id if isinstance(elem, Element) else int(elem)


def _actor_id(actor):
    return actor.id if isinstance(actor, Actor) else int(actor)


def _space_id(space):
    return space.handle if isinstance(space, Space) else int(space)


class SceneError(ValueError):
    """A scene mutator was rejected and NOT registered (sb_scene_status_t).
    `code` is the C status (2 = sibling containment regions overlap, 3 = a child
    sticks out of its parent); the message names both elements."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class Element:
    """Wraps an (space, sb_element_t) pair."""

    def _check_scene(self, status: int):
        if status == 0:
            return
        import ctypes
        buf = ctypes.create_string_buffer(512)
        self._native.sb_get_scene_status(self.space.handle, buf, 512)
        raise SceneError(int(status), buf.value.decode("utf-8", "replace")
                         or "scene mutator rejected (status %d)" % int(status))

    def __init__(self, space: Space, elem_id: int):
        self.space = space
        self.id = elem_id
        self._native = space._native

    def set_transform(self, matrix4x4=None):
        """Row-major 4x4 world transform (spatialblend.h:135-141)."""
        m = _identity16_row_major() if matrix4x4 is None else _flatten_row_major(matrix4x4)
        self._check_scene(
            self._native.sb_set_element_transform(self.space.handle, self.id, m))

    def set_shared_id(self, shared_id: int):
        self._native.sb_set_element_shared_id(
            self.space.handle, self.id, shared_id)

    def set_tag(self, tag: str):
        self._native.sb_set_element_tag(
            self.space.handle, self.id, tag.encode("utf-8"))

    def set_polygon(self, role: "GeometryRole", points_xy: Sequence[Sequence[float]]):
        flat = [c for p in points_xy for c in p]
        self._check_scene(self._native.sb_set_element_polygon(
            self.space.handle, self.id, int(role), _double_array(flat),
            len(points_xy)))

    def set_box(self, role: "GeometryRole", min_xyz: Sequence[float],
                max_xyz: Sequence[float]):
        self._check_scene(self._native.sb_set_element_box(
            self.space.handle, self.id, int(role),
            float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2]),
            float(max_xyz[0]), float(max_xyz[1]), float(max_xyz[2])))

    def set_collider(self, collider_type: "ColliderType", params: Sequence[float]):
        """params layout depends on collider_type -- see
        spatialblend.h:210-213 (BOX={cx,cy,cz,sx,sy,sz}, SPHERE={cx,cy,cz,r},
        CAPSULE={cx,cy,cz,r,height,axis})."""
        self._check_scene(self._native.sb_set_element_collider(
            self.space.handle, self.id, int(collider_type),
            _double_array(params), len(params)))

    def set_surface(self, points_xyz: Sequence[Sequence[float]],
                    role: "GeometryRole"):
        flat = [c for p in points_xyz for c in p]
        self._check_scene(self._native.sb_set_element_surface(
            self.space.handle, self.id, _double_array(flat),
            len(points_xyz), int(role)))

    def set_polytope(self, vertices_xyz: Sequence[Sequence[float]],
                     faces: Sequence[Sequence[int]], role: "GeometryRole"):
        flat_v = [c for v in vertices_xyz for c in v]
        flat_faces = [i for face in faces for i in face]
        face_sizes = [len(face) for face in faces]
        self._native.sb_set_element_polytope(
            self.space.handle, self.id, _double_array(flat_v),
            len(vertices_xyz), _int32_array(flat_faces),
            _int32_array(face_sizes), len(faces), int(role))

    def add_sub_region(self, name: str, tag: str,
                       points_xy: Sequence[Sequence[float]],
                       transform16=None):
        flat = [c for p in points_xy for c in p]
        m = None if transform16 is None else _flatten_row_major(transform16)
        self._native.sb_add_sub_region(
            self.space.handle, self.id, name.encode("utf-8"),
            tag.encode("utf-8"), _double_array(flat), len(points_xy), m)

    def add_anchor(self, name: str, matrix4x4: Sequence[Sequence[float]]):
        """Column-major local transform (spatialblend.h:291-293)."""
        self._native.sb_add_element_anchor(
            self.space.handle, self.id, name.encode("utf-8"),
            _flatten_col_major(matrix4x4))

    def share(self, user_data: Optional[bytes] = None):
        if user_data:
            buf = ctypes.create_string_buffer(user_data, len(user_data))
            self._native.sb_share_element(
                self.space.handle, self.id, buf, len(user_data))
        else:
            self._native.sb_share_element(self.space.handle, self.id, None, 0)

    def unshare(self):
        self._native.sb_unshare_element(self.space.handle, self.id)

    def unlink(self) -> int:
        """sb_unlink: remove this element's cross-space link on both sides
        (the element is kept). Returns the number of links removed."""
        return int(self._native.sb_unlink(self.space.handle, self.id))

    def remove(self) -> bool:
        """sb_remove_local_element (local, unshared, non-root subtrees)."""
        return self._native.sb_remove_local_element(
            self.space.handle, self.id) == 1

    def world_transform(self):
        """Row-major 4x4 world transform (4 row tuples), or None."""
        return _obs.element_world_transform(self.space.handle, self.id)

    def parent(self) -> Optional["Element"]:
        parent_id = _obs.element_parent(self.space.handle, self.id)
        return Element(self.space, parent_id) if parent_id else None

    def __eq__(self, other):
        return (isinstance(other, Element) and other.space.handle == self.space.handle
                and other.id == self.id)

    def __repr__(self):
        return f"Element(space={self.space.handle}, id={self.id})"


class Actor:
    """Wraps an (space, sb_actor_t) pair."""

    def __init__(self, space: Space, actor_id: int):
        self.space = space
        self.id = actor_id
        self._native = space._native

    def add_anchor(self, name: str, x: float, y: float, z: float,
                   parent: Optional[str] = None):
        self._native.sb_add_anchor(
            self.space.handle, self.id, name.encode("utf-8"),
            float(x), float(y), float(z),
            parent.encode("utf-8") if parent else None)

    def add_standard_anchors(self):
        self._native.sb_add_standard_anchors(self.space.handle, self.id)

    def set_element(self, elem: Union[Element, int]):
        self._native.sb_set_actor_element(
            self.space.handle, self.id, _elem_id(elem))

    def update_state(self, x: float, y: float, yaw: float) -> AsyncHandle:
        return AsyncHandle(self._native.sb_update_actor_state(
            self.space.handle, self.id, float(x), float(y), float(yaw)))

    def update_pose(self, x: float, y: float, yaw: float, pitch: float = None,
                    roll: float = None) -> AsyncHandle:
        """sb_update_actor_pose (ABI 3.1): yaw only, yaw+pitch (pitch given) or
        yaw+pitch+roll (roll given). Pitch/roll are a pass-through to the
        receiver's proposal; core never solves them."""
        import ctypes
        from ._types import SbOrientation
        o = SbOrientation()
        o.struct_size = ctypes.sizeof(SbOrientation)
        o.type = int(
            OrientationType.YAW if pitch is None else (
                OrientationType.YAW_PITCH if roll is None
                else OrientationType.YAW_PITCH_ROLL))
        o.yaw = float(yaw)
        o.pitch = 0.0 if pitch is None else float(pitch)
        o.roll = 0.0 if roll is None else float(roll)
        return AsyncHandle(self._native.sb_update_actor_pose(
            self.space.handle, self.id, float(x), float(y), ctypes.byref(o)))

    def get_state(self) -> ActorState:
        s = self._native.sb_get_actor_state(self.space.handle, self.id)
        return ActorState(position_x=s.position_x, position_y=s.position_y,
                          yaw=s.yaw)

    def anchor_id(self, name: str) -> int:
        """sb_get_anchor_id: the id of an anchor of this actor (0 = the root
        anchor), interned on first use."""
        out = ctypes.c_uint32(0)
        if self._native.sb_get_anchor_id(
                self.space.handle, self.id, name.encode(), ctypes.byref(out)) != 0:
            raise ValueError("sb_get_anchor_id rejected %r" % name)
        return int(out.value)

    def anchor_name(self, anchor_id: int) -> str:
        buf = ctypes.create_string_buffer(256)
        n = self._native.sb_get_anchor_name(
            self.space.handle, self.id, int(anchor_id), buf, 256)
        if n < 0:
            raise ValueError("unknown anchor id %r" % anchor_id)
        return buf.value.decode()

    def publish_anchor_pose(self, anchor_id: int, x: float, y: float, yaw: float,
                            timestamp_s: float, sequence: int, z: float = 0.0,
                            epoch: int = 1, teleported: bool = False,
                            pitch: float = None, roll: float = None) -> int:
        """sb_publish_anchor_pose: the DRIVER's pose of one anchor, the only writer
        of that anchor's port. Returns 0 accepted, -1 invalid, -2 stale, -3
        refused."""
        from ._types import SbAnchorPose as _SbAnchorPose
        pose = _SbAnchorPose()
        pose.struct_size = ctypes.sizeof(_SbAnchorPose)
        pose.flags = 1 if teleported else 0
        pose.x, pose.y, pose.z = float(x), float(y), float(z)
        pose.orientation.struct_size = ctypes.sizeof(pose.orientation)
        pose.orientation.type = int(
            OrientationType.YAW if pitch is None else (
                OrientationType.YAW_PITCH if roll is None
                else OrientationType.YAW_PITCH_ROLL))
        pose.orientation.yaw = float(yaw)
        pose.orientation.pitch = 0.0 if pitch is None else float(pitch)
        pose.orientation.roll = 0.0 if roll is None else float(roll)
        pose.epoch = int(epoch)
        return self._native.sb_publish_anchor_pose(
            self.space.handle, self.id, int(anchor_id), ctypes.byref(pose),
            float(timestamp_s), int(sequence))

    # -- embodiment registration (ABI 4.7) ----------------------------------
    def register_embodiment(self, spec_json: str) -> None:
        """sb_register_embodiment_json: register this actor's skeleton and the
        anchors it declares (schema spatialblend.embodiment 1.0.0). Raises
        ValueError with the core's reason when the spec is refused."""
        raw = spec_json.encode("utf-8")
        err = ctypes.create_string_buffer(512)
        rc = self._native.sb_register_embodiment_json(
            self.space.handle, self.id, raw, len(raw), err, 512)
        if rc != 0:
            raise ValueError("sb_register_embodiment_json -> %d: %s"
                             % (rc, err.value.decode("utf-8", "replace")))

    def embodiment_json(self) -> Optional[str]:
        """sb_get_embodiment_json: the registered spec as canonical JSON, or
        None when this actor has none."""
        need = self._native.sb_get_embodiment_json(
            self.space.handle, self.id, None, 0)
        if need <= 0:
            return None
        buf = ctypes.create_string_buffer(need + 1)
        self._native.sb_get_embodiment_json(
            self.space.handle, self.id, buf, need + 1)
        return buf.value.decode("utf-8")

    def bone_id(self, name: str) -> int:
        """sb_get_bone_id: -1 when the skeleton lacks the bone."""
        return int(self._native.sb_get_bone_id(
            self.space.handle, self.id, name.encode("utf-8")))

    @staticmethod
    def _bone_pose(position, rotation_xyzw, epoch=1, teleported=False):
        from ._types import SbBonePose as _P
        p = _P()
        p.struct_size = ctypes.sizeof(_P)
        p.flags = 1 if teleported else 0
        for i in range(3):
            p.position[i] = float(position[i])
        for i in range(4):
            p.rotation[i] = float(rotation_xyzw[i])
        p.epoch = int(epoch)
        return p

    def publish_bone_pose(self, bone_id: int, position, rotation_xyzw,
                          timestamp_s: float, sequence: int, epoch: int = 1,
                          teleported: bool = False) -> int:
        """sb_publish_bone_pose (RECEIVER driver): 0 accepted, -1 invalid, -2
        stale, -3 refused."""
        pose = self._bone_pose(position, rotation_xyzw, epoch, teleported)
        return self._native.sb_publish_bone_pose(
            self.space.handle, self.id, int(bone_id), ctypes.byref(pose),
            float(timestamp_s), int(sequence))

    def apply_bone_frame(self, epoch: int, sequence: int, time_seconds: float,
                         bones, observations=()) -> bool:
        """sb_apply_actor_bone_frame (SOURCE): bones = iterable of (bone_name,
        position, rotation_xyzw); observations = (anchor_name, x, y, z, yaw)
        for the anchors that are not bone-bound (the planar root)."""
        from ._types import SbNamedBonePose as _N
        bones = list(bones)
        names = [str(b[0]).encode("utf-8") for b in bones]
        rows = []
        for n, b in zip(names, bones):
            row = _N()
            row.bone_name = n
            row.pose = self._bone_pose(b[1], b[2])
            rows.append(row)
        bbuf = (_N * max(len(rows), 1))(*rows)
        obs = list(observations)
        onames = [str(o[0]).encode("utf-8") for o in obs]
        orows = [_o.SbActorObservation(n, float(o[1]), float(o[2]),
                                       float(o[3]), float(o[4]))
                 for n, o in zip(onames, obs)]
        obuf = (_o.SbActorObservation * max(len(orows), 1))(*orows)
        return self._native.sb_apply_actor_bone_frame(
            self.space.handle, self.id, int(epoch), int(sequence),
            float(time_seconds), bbuf, len(rows),
            ctypes.cast(obuf, ctypes.c_void_p), len(orows)) == 1

    def chain(self):
        """The registered normalized kinematic chain as a list of dicts (name,
        parent, rest_position, rest_rotation, segment_length_m, limits), or
        None when this actor registered none (sb_get_chain_joint*)."""
        from ._types import SbChainJoint as _J
        n = self._native.sb_get_chain_joint_count(self.space.handle, self.id)
        if n <= 0:
            return None
        out = []
        for i in range(n):
            j = _J()
            j.struct_size = ctypes.sizeof(_J)
            name = ctypes.create_string_buffer(256)
            if self._native.sb_get_chain_joint(
                    self.space.handle, self.id, i, ctypes.byref(j), name, 256) != 0:
                raise RuntimeError("sb_get_chain_joint(%d) failed" % i)
            limits = {}
            for a, axis in enumerate("xyz"):
                lo = j.limit_min[a] if j.limit_mask & (1 << (2 * a)) else None
                hi = j.limit_max[a] if j.limit_mask & (1 << (2 * a + 1)) else None
                if lo is not None or hi is not None:
                    limits[axis] = (lo, hi)
            out.append({"name": name.value.decode(), "parent": int(j.parent),
                        "rest_position": tuple(j.rest_position),
                        "rest_rotation": tuple(j.rest_rotation),
                        "segment_length_m": float(j.segment_length_m),
                        "limits": limits})
        return out

    def _convert_pose(self, fn, anchor_name, position, rotation_xyzw):
        from ._types import SbBonePose as _P
        src = self._bone_pose(position, rotation_xyzw)
        out = _P()
        if fn(self.space.handle, self.id, anchor_name.encode("utf-8"),
              ctypes.byref(src), ctypes.byref(out)) != 0:
            raise ValueError("%r is not a registered, bone-bound anchor"
                             % anchor_name)
        return tuple(out.position), tuple(out.rotation)

    def anchor_pose_from_bone(self, anchor_name, position, rotation_xyzw):
        """anchor = bone * offset (the registration's arithmetic)."""
        return self._convert_pose(
            self._native.sb_embodiment_anchor_pose_from_bone, anchor_name,
            position, rotation_xyzw)

    def bone_pose_from_anchor(self, anchor_name, position, rotation_xyzw):
        """bone = anchor * offset^-1: the bone target that places the anchor."""
        return self._convert_pose(
            self._native.sb_embodiment_bone_pose_from_anchor, anchor_name,
            position, rotation_xyzw)

    def publish_embodied_pose(self, x: float, y: float, yaw: float,
                              timestamp_s: float, sequence: int,
                              epoch: int = 1, teleported: bool = False) -> int:
        """The ROOT anchor's form of publish_anchor_pose."""
        return self.publish_anchor_pose(0, x, y, yaw, timestamp_s, sequence,
                                        epoch=epoch, teleported=teleported)

    def read_proposal(self, cap: int = 16) -> Optional[Proposal]:
        """sb_read_proposals: the latest proposal set (root first); None for an
        invalid actor."""
        from ._types import SbAnchorProposal as _SbAnchorProposal
        from ._types import SbProposalSet as _SbProposalSet
        buf = (_SbAnchorProposal * cap)()
        for a in buf:
            a.struct_size = ctypes.sizeof(_SbAnchorProposal)
        header = _SbProposalSet()
        header.struct_size = ctypes.sizeof(_SbProposalSet)
        if self._native.sb_read_proposals(
                self.space.handle, self.id, ctypes.byref(header), buf, cap) != 0:
            return None
        anchors = []
        for i in range(header.anchor_count):
            a = buf[i]
            anchors.append(AnchorProposal(
                anchor_id=a.anchor_id, status=a.status, cause=a.cause,
                x=a.x, y=a.y, z=a.z, yaw=a.orientation.yaw,
                pitch=a.orientation.pitch, roll=a.orientation.roll,
                orientation_type=a.orientation.type,
                yaw_specified=not (a.flags & 0x1), engaged=bool(a.flags & 0x2),
                sampled=bool(a.flags & 0x4), arrived=bool(a.flags & 0x8),
                position_specified=not (a.flags & 0x10),
                err_spacing_target=a.err_spacing_target,
                err_spacing_sample=a.err_spacing_sample,
                err_facing_target=a.err_facing_target,
                err_facing_sample=a.err_facing_sample,
                err_pose_target=a.err_pose_target,
                err_pose_sample=a.err_pose_sample, err_scale_m=a.err_scale_m,
                referent_shared_id=a.referent_shared_id))
        return Proposal(
            revision=header.revision, source_sequence=header.source_sequence,
            sample_valid=bool(header.sample_valid),
            sample_sequence=header.sample_sequence,
            sample_epoch=header.sample_epoch,
            sample_timestamp_s=header.sample_timestamp_s,
            sample_age_s=header.sample_age_s, anchors=anchors)

    def raise_source_exception(self, exception_id: int,
                               payload: bytes = b"") -> None:
        """sb_raise_source_exception: this source raises an app-specific
        exception (a teleport, a button next to a whiteboard, ...) with an opaque
        payload; every other space gets it as an SB_ACTOR_EVENT_SOURCE_EXCEPTION."""
        data = ctypes.create_string_buffer(bytes(payload), max(len(payload), 1))
        if self._native.sb_raise_source_exception(
                self.space.handle, self.id, int(exception_id),
                ctypes.cast(data, ctypes.c_void_p), len(payload)) != 0:
            raise ValueError("sb_raise_source_exception rejected")

    def set_target_event_delivery(self, enabled: bool):
        """Opt into independent target events (ABI 4.13).

        Repeating the current state is idempotent. First/re-enable refuses a
        legacy exception backlog without changing the subscription. A successful
        transition starts a release/placement boundary; disable preserves queued
        events. Validate the native typed payload layout before opting in.
        """
        if self._native.sb_set_target_event_delivery(self.space.handle, self.id, int(enabled)) != 0:
            raise ValueError("sb_set_target_event_delivery rejected")

    def poll_event(self) -> Optional[ActorEvent]:
        """sb_poll_actor_event: the oldest pending event of this receiver actor,
        or None."""
        from ._types import SbActorEvent as _SbActorEvent
        e = _SbActorEvent()
        e.struct_size = ctypes.sizeof(_SbActorEvent)
        rc = self._native.sb_poll_actor_event(
            self.space.handle, self.id, ctypes.byref(e))
        if rc < 0:
            raise ValueError("sb_poll_actor_event rejected")
        if rc == 0:
            return None
        return ActorEvent(
            kind=e.kind, revision=e.revision, source_sequence=e.source_sequence,
            source_shared_id=e.source_shared_id, exception_id=e.exception_id,
            payload=bytes(bytearray(e.payload[:e.payload_size])),
            interaction_kind=e.interaction_kind, anchor_id=e.anchor_id,
            x=e.x, y=e.y, z=e.z, yaw=e.orientation.yaw,
            pitch=e.orientation.pitch, roll=e.orientation.roll)

    def query_course(self, fx: float, fy: float, tx: float, ty: float,
                     cap: int = 64):
        """sb_query_course: [(x, y)] waypoints, [] when no course exists."""
        buf = (ctypes.c_double * (2 * cap))()
        n = self._native.sb_query_course(
            self.space.handle, self.id, float(fx), float(fy), float(tx),
            float(ty), buf, int(cap))
        if n < 0:
            raise ValueError("sb_query_course rejected")
        return [(buf[2 * i], buf[2 * i + 1]) for i in range(min(n, cap))]

    def query_nearest_clear_point(self, x: float, y: float):
        """sb_query_nearest_clear_point (ABI 4.8, a driver service): (x, y) of the nearest point clear of the
        footprints for this actor's registered body radius, or None when the free space has none."""
        buf = (ctypes.c_double * 2)()
        r = self._native.sb_query_nearest_clear_point(
            self.space.handle, self.id, float(x), float(y), buf)
        if r < 0:
            raise ValueError("sb_query_nearest_clear_point rejected")
        return (buf[0], buf[1]) if r == 1 else None

    def bind(self, anchor_name: str, kernel_type: "KernelType") -> bool:
        """sb_bind (ABI 4.2): no scope argument; scope belongs to the family."""
        return self._native.sb_bind(
            self.space.handle, self.id, anchor_name.encode("utf-8"),
            int(kernel_type)) == 1

    def bind_affordance(self, anchor_name: str,
                        tag: Optional[str] = None) -> bool:
        """sb_bind_affordance (ABI 4.2): a tag is only an AffordancePose binding."""
        return self._native.sb_bind_affordance(
            self.space.handle, self.id, anchor_name.encode("utf-8"),
            tag.encode("utf-8") if tag else None) == 1

    def destroy(self) -> bool:
        return self._native.sb_destroy_actor(self.space.handle, self.id) == 1

    # -- checked anchors / bindings ----------------------------------------
    def add_anchor_with_position_dimension(
            self, name: str, x: float = 0.0, y: float = 0.0, z: float = 0.0,
            parent: Optional[str] = None, position_dimension: int = 3) -> None:
        ok = self._native.sb_add_anchor_with_position_dimension(
            self.space.handle, self.id, name.encode("utf-8"), float(x),
            float(y), float(z), parent.encode("utf-8") if parent else None,
            int(position_dimension))
        if ok != 1:
            raise ValueError("sb_add_anchor_with_position_dimension rejected "
                             "anchor %r" % name)

    def set_anchor_transform(self, anchor_name: str, x: float, y: float,
                             z: float, yaw: float) -> None:
        self._native.sb_set_anchor_transform(
            self.space.handle, self.id, anchor_name.encode("utf-8"),
            float(x), float(y), float(z), float(yaw))

    def clear_anchor_transform(self, anchor_name: str) -> None:
        self._native.sb_clear_anchor_transform(
            self.space.handle, self.id, anchor_name.encode("utf-8"))

    def set_anchor_active(self, anchor_name: str, active: bool) -> bool:
        return self._native.sb_set_anchor_active(
            self.space.handle, self.id, anchor_name.encode("utf-8"),
            1 if active else 0) == 1

    # -- observation frames ------------------------------------------------
    def begin_observation_epoch(self, root_anchor: str) -> int:
        epoch = self._native.sb_begin_actor_observation_epoch(
            self.space.handle, self.id, root_anchor.encode("utf-8"))
        if epoch == 0:
            raise RuntimeError("sb_begin_actor_observation_epoch failed")
        return int(epoch)

    def apply_observation_frame(self, epoch: int, sequence: int,
                                time_seconds: float, observations) -> bool:
        """observations: iterable of (anchor_name, x, y, z, yaw)."""
        names = [str(o[0]).encode("utf-8") for o in observations]
        rows = [_o.SbActorObservation(n, float(o[1]), float(o[2]),
                                      float(o[3]), float(o[4]))
                for n, o in zip(names, observations)]
        buf = (_o.SbActorObservation * max(len(rows), 1))(*rows)
        return self._native.sb_apply_actor_observation_frame(
            self.space.handle, self.id, int(epoch), int(sequence),
            float(time_seconds), buf, len(rows)) == 1

    def clear_observation_anchor(self, epoch: int, anchor_name: str) -> bool:
        return self._native.sb_clear_actor_observation_anchor(
            self.space.handle, self.id, int(epoch),
            anchor_name.encode("utf-8")) == 1

    def observation_clock(self):
        """(epoch, frame_sequence, time_seconds, has_sample) or None."""
        epoch, seq = ctypes.c_uint64(), ctypes.c_int64()
        t, has = ctypes.c_double(), ctypes.c_int32()
        if self._native.sb_get_actor_observation_clock(
                self.space.handle, self.id, ctypes.byref(epoch),
                ctypes.byref(seq), ctypes.byref(t), ctypes.byref(has)) != 1:
            return None
        return int(epoch.value), int(seq.value), float(t.value), bool(has.value)

    # -- observability and injection -------------------------------------
    def relation_states(self):
        return _obs.relation_states(self.space.handle, self.id)

    def relation_states_v2(self):
        return _obs.relation_states_v2(self.space.handle, self.id)

    def scope_records(self):
        return _obs.scope_records(self.space.handle, self.id)

    def scope_candidates(self, binding_index: int):
        return _obs.scope_candidates(self.space.handle, self.id, binding_index)

    def realization_weights(self):
        return _obs.realization_weights(self.space.handle, self.id)

    def realization_channels(self):
        return _obs.realization_channels(self.space.handle, self.id)

    def solver_diagnostics(self):
        return _obs.solver_diagnostics(self.space.handle, self.id)

    def solver_goals(self):
        return _obs.solver_goals(self.space.handle, self.id)

    def realization_targets_v2(self):
        return _obs.realization_targets_v2(self.space.handle, self.id)

    def inject_relation_payloads(self, payloads, time_seconds: float) -> None:
        _obs.inject_relation_payloads(self.space.handle, self.id, payloads,
                                      time_seconds)

    def __eq__(self, other):
        return (isinstance(other, Actor) and other.space.handle == self.space.handle
                and other.id == self.id)

    def __repr__(self):
        return f"Actor(space={self.space.handle}, id={self.id})"


def cleanup_all():
    """sb_cleanup_all() -- releases ALL sessions/spaces across the process.
    Invalidates every previously returned handle everywhere."""
    _api.get_native().sb_cleanup_all()


def abi_version() -> int:
    """sb_abi_version()."""
    return int(_api.get_native().sb_abi_version())


def abi_minor_version() -> int:
    """sb_abi_minor_version(): 14 for ABI 4.14 (per-anchor chart transform, sb_read_anchor_chart)."""
    return int(_api.get_native().sb_abi_minor_version())


def v2_abi_layout() -> dict:
    """sb_get_v2_abi_layout() as a dict of field -> value."""
    layout = _o.SbV2AbiLayout()
    _api.get_native().sb_get_v2_abi_layout(ctypes.byref(layout),
                                           ctypes.sizeof(layout))
    return {name: getattr(layout, name) for name, _ in layout._fields_}


def sizeof(struct_id: int) -> int:
    """sb_sizeof(sb_struct_id_t)."""
    return int(_api.get_native().sb_sizeof(int(struct_id)))
