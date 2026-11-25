# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""spatialblend_capi: a pure-Python ctypes package wrapping the spatialblend
C API (see spatialblend/api/c/spatialblend.h).

Unlike the pybind11-based ``spatialblend`` package (which is a C extension
tied to one CPython ABI, currently built for cp310), this package is plain
Python + ctypes -- it has NO C-extension of its own, and therefore no
Python-version dependency at all. It loads the same ``libspatialblend``
shared library that the pybind wheel embeds, via ctypes.CDLL, and works
unmodified on any CPython 3.6+ interpreter (so a consumer pinned to an
older Python can use this package instead of the pybind one).

Public API:
- ``api``: Session, Space, Element, Actor, the dataclasses
  RealizationTarget/ActorState/RelationEvent/ElementAddedEvent, and the
  runtime configuration, tracing, checked anchors/bindings, observation
  frames and support correspondences.
- ``observability``: the observation and injection surface
  (relation states v2, scope records, policy parameters, weights and the
  driving relation per channel, solver witnesses/snap, reach clamp, payload
  injection, pure relation evaluation, element world transforms).
- ``rich``: rich-realization activation/poll/callback, receiver input, and
  the Unity-compatible ``spatialblend.rich_realization`` JSON serializer.
The raw ctypes bindings live in ``_api``/``_types``/``_obs_types``/``_lib``
(leading underscore: internal, not a stability guarantee).

Importing this package eagerly loads libspatialblend and verifies every
ctypes mirror against the library's own size and per-field offset queries
(_api.check_struct_layout()), so ABI drift fails at import time rather than
as a mysterious crash deep in a call.
"""
from .api import (
    Actor,
    ActorState,
    AsyncHandle,
    ColliderType,
    Element,
    ElementAddedEvent,
    GeometryRole,
    KernelType,
    OrientationType,
    Proposal,
    RealizationTarget,
    RelationEvent,
    RelationPhase,
    SceneStatus,
    Session,
    Space,
    abi_minor_version,
    abi_version,
    cleanup_all,
)
from .api import RealizationPolicy, ReferentResolveMode
from . import observability, rich
from ._lib import LibraryNotFoundError, find_library_path, loaded_library_path

__all__ = [
    "Actor", "ActorState", "AsyncHandle", "ColliderType", "Element",
    "ElementAddedEvent", "GeometryRole", "KernelType", "OrientationType",
    "Proposal", "RealizationTarget", "RelationEvent", "RelationPhase",
    "SceneStatus", "Session", "Space", "abi_minor_version", "abi_version",
    "cleanup_all",
    "LibraryNotFoundError", "find_library_path", "loaded_library_path",
    "RealizationPolicy", "ReferentResolveMode", "observability", "rich",
]

__version__ = "1.0.0"
