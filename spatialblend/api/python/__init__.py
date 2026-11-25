# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""SpatialBlend Python bindings.

Names are imported explicitly. A pybind11 extension does not publish
``__all__``, and ``from extension import *`` then drops every class
(LocalContext included) on current CPython.

The extension is absent from capi-only runs. Those import this package
as the parent of ``capi`` and must still succeed.
"""
_PY_EXPORTS = (
    "Actor",
    "ActorHandle",
    "ActorMotionState",
    "AnchorWorldPoseOverride",
    "Binding",
    "Element",
    "ElementEvent",
    "ElementHandle",
    "ElementTree",
    "GeometryRole",
    "GroundedRelation",
    "KernelType",
    "LocalContext",
    "MatchResult",
    "NavigableSpatialDefinition",
    "NormalizedPolygonPosition",
    "Polygon2D",
    "Polygon3D",
    "PolytopeSpatialDefinition",
    "RealizationResult",
    "RealizationTarget",
    "RelationEvaluationResult",
    "RelationPayload",
    "RelationState",
    "RemoteRelationUpdate",
    "SatisfactionMetric",
    "SolverStatus",
    "Space",
    "SpaceBuilder",
    "SpaceHub",
    "SpatialDefinition",
    "StateType",
    "SubjectAnchor",
    "Support",
    "SurfaceSpatialDefinition",
    "Trajectory",
)
try:
    from ._spatialblend_py import (
        Actor,
        ActorHandle,
        ActorMotionState,
        AnchorWorldPoseOverride,
        Binding,
        Element,
        ElementEvent,
        ElementHandle,
        ElementTree,
        GeometryRole,
        GroundedRelation,
        KernelType,
        LocalContext,
        MatchResult,
        NavigableSpatialDefinition,
        NormalizedPolygonPosition,
        Polygon2D,
        Polygon3D,
        PolytopeSpatialDefinition,
        RealizationResult,
        RealizationTarget,
        RelationEvaluationResult,
        RelationPayload,
        RelationState,
        RemoteRelationUpdate,
        SatisfactionMetric,
        SolverStatus,
        Space,
        SpaceBuilder,
        SpaceHub,
        SpatialDefinition,
        StateType,
        SubjectAnchor,
        Support,
        SurfaceSpatialDefinition,
        Trajectory,
    )
except ModuleNotFoundError:
    _py_loaded = False
else:
    _py_loaded = True

_SIM_EXPORTS = (
    "ProximitySnapshot",
    "SimulationResult",
    "Simulator",
    "SimulatorHooks",
    "SimulatorRunConfig",
)
try:
    from ._spatialblend_sim_py import (  # noqa: F401
        ProximitySnapshot,
        SimulationResult,
        Simulator,
        SimulatorHooks,
        SimulatorRunConfig,
    )
except ModuleNotFoundError:
    # The Simulator extension is test-only and absent from the shipped wheel.
    _sim_loaded = False
else:
    _sim_loaded = True

__all__ = list(_PY_EXPORTS) if _py_loaded else []
if _sim_loaded:
    __all__.extend(_SIM_EXPORTS)
