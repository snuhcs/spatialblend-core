# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""InteractionScript: fluent builder for composing interaction trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

from spatialblend.api.python import KernelType, Trajectory


# ---------------------------------------------------------------------------
# Internal command representation
# ---------------------------------------------------------------------------

class _CommandType(Enum):
    LOOK_AT = auto()
    APPROACH_TO = auto()
    REACH_TO = auto()
    MOVE_TO = auto()
    WANDER = auto()


@dataclass
class _Command:
    type: _CommandType
    duration: float
    element_tag: Optional[str] = None
    anchor_name: Optional[str] = None
    x: float = 0.0
    y: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOOK_AT_DISTANCE = 2.5
_APPROACH_DISTANCE = 1.0
_REACH_DISTANCE = 0.5
_WALK_SPEED = 1.4  # m/s


def _find_element_by_tag(space, tag: str, root_id: int = 1):
    """Find element by tag, return world position (2D np.array) or None."""
    tree = space.element_tree
    return _find_element_by_tag_recursive(tree, root_id, tag)


def _find_element_by_tag_recursive(tree, node_id, tag):
    """Recursively search the element tree for a given tag."""
    elem = tree.get_element(node_id)
    if elem is not None and elem.tag == tag:
        world_t = tree.get_world_transform(node_id)
        pos = np.array([world_t[0, 3], world_t[1, 3]])
        return pos
    child_ids = tree.get_direct_child_ids(node_id)
    for cid in child_ids:
        result = _find_element_by_tag_recursive(tree, cid, tag)
        if result is not None:
            return result
    return None


def _yaw_toward(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
    """Compute yaw angle facing from *from_pos* toward *to_pos*."""
    d = to_pos - from_pos
    return float(np.arctan2(d[1], d[0]))


def _approach_point(
    current: np.ndarray, target: np.ndarray, distance: float
) -> np.ndarray:
    """Compute approach position *distance* away from *target*, toward *current*."""
    d = current - target
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        # Fallback: offset along +x
        d = np.array([1.0, 0.0])
        norm = 1.0
    return target + d / norm * distance


def _sample_point_in_polygon(polygon, rng: np.random.Generator) -> np.ndarray:
    """Sample a random point inside *polygon* via rejection sampling."""
    points = polygon.get_points()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    for _ in range(1000):
        x = rng.uniform(min_x, max_x)
        y = rng.uniform(min_y, max_y)
        pt = np.array([x, y])
        if polygon.contains(pt):
            return pt
    # Fallback: centroid
    return np.mean(points, axis=0)


def _get_walkable_polygon(space):
    """Return the walkable Polygon2D from the root element's spatial def."""
    tree = space.element_tree
    root = tree.get_element(1)
    if root is None:
        return None
    sd = root.get_spatial_definition()
    if sd is None:
        return None
    return sd.get_walkable_polygon()


# ---------------------------------------------------------------------------
# InteractionScript
# ---------------------------------------------------------------------------

class InteractionScript:
    """Fluent builder for composing interaction trajectories."""

    def __init__(self, seed: int = 42):
        self._commands: List[_Command] = []
        self._rng = np.random.default_rng(seed)

    # -- Fluent command methods --

    @staticmethod
    def _validate_duration(duration: float) -> None:
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")

    def look_at(self, element_tag: str, duration: float) -> InteractionScript:
        """Navigate to viewing distance (~2.5 m) from element, face it, dwell."""
        self._validate_duration(duration)
        self._commands.append(
            _Command(type=_CommandType.LOOK_AT, duration=duration,
                     element_tag=element_tag)
        )
        return self

    def approach_to(self, element_tag: str, duration: float) -> InteractionScript:
        """Navigate to close interaction distance (~1.0 m) from element, face it, dwell."""
        self._validate_duration(duration)
        self._commands.append(
            _Command(type=_CommandType.APPROACH_TO, duration=duration,
                     element_tag=element_tag)
        )
        return self

    def reach_to(self, element_tag: str, anchor_name: str,
                 duration: float) -> InteractionScript:
        """Navigate close enough that actor's anchor enters element's affordance region.

        Note: build() will call actor_handle.bind() as a side effect to set up
        the AffordancePose relation for this anchor/element pair.
        """
        self._validate_duration(duration)
        self._commands.append(
            _Command(type=_CommandType.REACH_TO, duration=duration,
                     element_tag=element_tag, anchor_name=anchor_name)
        )
        return self

    def move_to(self, x: float, y: float, duration: float) -> InteractionScript:
        """Move to explicit position, dwell."""
        self._validate_duration(duration)
        self._commands.append(
            _Command(type=_CommandType.MOVE_TO, duration=duration, x=x, y=y)
        )
        return self

    def wander(self, duration: float) -> InteractionScript:
        """Random walk within walkable polygon via multiple sub-waypoints."""
        self._validate_duration(duration)
        self._commands.append(
            _Command(type=_CommandType.WANDER, duration=duration)
        )
        return self

    # -- Build --

    def build(self, space_builder, actor_handle) -> Trajectory:
        """Resolve all commands into a Trajectory.

        Args:
            space_builder: SpaceBuilder providing access to the space.
            actor_handle: ActorHandle for the actor whose trajectory is built.

        Returns:
            A Trajectory with waypoints at appropriate times.
        """
        space = space_builder.get_space()
        current_pos = np.array(actor_handle.get_position())
        current_yaw = float(actor_handle.get_yaw())
        current_time = 0.0

        traj = Trajectory()

        for cmd in self._commands:
            if cmd.type == _CommandType.WANDER:
                # Wander produces multiple sub-legs instead of a single target
                current_pos, current_yaw, current_time = self._build_wander(
                    cmd, space, current_pos, current_time, traj
                )
                continue

            target_pos, target_yaw = self._resolve_target(
                cmd, space, current_pos, actor_handle
            )

            # Transit
            current_pos, current_yaw, current_time = self._add_transit_and_dwell(
                traj, current_pos, current_time, target_pos, target_yaw,
                cmd.duration
            )


        return traj

    # -- Private helpers --

    @staticmethod
    def _add_transit_and_dwell(
        traj, current_pos, current_time, target_pos, target_yaw, dwell_dur
    ):
        """Add transit waypoints + dwell, return updated (pos, yaw, time)."""
        dist = float(np.linalg.norm(target_pos - current_pos))
        transit_time = dist / _WALK_SPEED if dist > 1e-6 else 0.0

        if transit_time > 0:
            travel_yaw = _yaw_toward(current_pos, target_pos)
            traj.add_waypoint(current_time, current_pos, travel_yaw)
            current_time += transit_time
            traj.add_waypoint(current_time, target_pos, target_yaw)
        else:
            traj.add_waypoint(current_time, target_pos, target_yaw)

        current_time += dwell_dur
        traj.add_waypoint(current_time, target_pos, target_yaw)
        return target_pos, target_yaw, current_time

    def _build_wander(self, cmd, space, current_pos, current_time, traj):
        """Handle wander by generating multi-segment random walk."""
        polygon = _get_walkable_polygon(space)
        if polygon is None:
            raise ValueError(
                "Cannot wander: no walkable polygon on root element")

        n_pts = int(self._rng.integers(2, 4))  # 2 or 3 sub-legs
        pts = [
            _sample_point_in_polygon(polygon, self._rng) for _ in range(n_pts)
        ]
        # Split duration across sub-legs as brief pauses at each waypoint
        dwell_per_pt = cmd.duration / n_pts

        pos = current_pos
        yaw = 0.0
        for pt in pts:
            yaw = _yaw_toward(pos, pt)
            pos, yaw, current_time = self._add_transit_and_dwell(
                traj, pos, current_time, pt, yaw, dwell_per_pt
            )
        return pos, yaw, current_time

    def _resolve_target(
        self, cmd: _Command, space, current_pos: np.ndarray, actor_handle
    ) -> Tuple[np.ndarray, float]:
        """Return (position, yaw) for a single command."""
        if cmd.type == _CommandType.LOOK_AT:
            elem_pos = _find_element_by_tag(space, cmd.element_tag)
            if elem_pos is None:
                raise ValueError(
                    f"Element with tag '{cmd.element_tag}' not found")
            pos = _approach_point(current_pos, elem_pos, _LOOK_AT_DISTANCE)
            yaw = _yaw_toward(pos, elem_pos)
            return pos, yaw

        if cmd.type == _CommandType.APPROACH_TO:
            elem_pos = _find_element_by_tag(space, cmd.element_tag)
            if elem_pos is None:
                raise ValueError(
                    f"Element with tag '{cmd.element_tag}' not found")
            pos = _approach_point(current_pos, elem_pos, _APPROACH_DISTANCE)
            yaw = _yaw_toward(pos, elem_pos)
            return pos, yaw

        if cmd.type == _CommandType.REACH_TO:
            elem_pos = _find_element_by_tag(space, cmd.element_tag)
            if elem_pos is None:
                raise ValueError(
                    f"Element with tag '{cmd.element_tag}' not found")
            pos = _approach_point(current_pos, elem_pos, _REACH_DISTANCE)
            yaw = _yaw_toward(pos, elem_pos)
            # Bind the anchor
            actor_handle.bind(
                cmd.anchor_name,
                KernelType.AFFORDANCE_POSE,
                cmd.element_tag,
            )
            return pos, yaw

        if cmd.type == _CommandType.MOVE_TO:
            pos = np.array([cmd.x, cmd.y])
            yaw = _yaw_toward(current_pos, pos)
            return pos, yaw

        raise ValueError(f"Unknown command type: {cmd.type}")
