# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Hook that detects relation-based arrival and computes lag.

Arrival is defined as proximity >= threshold for the same (shared_id,
kernel_type) pair. Lag = (avatar_arrival_frame - source_arrival_frame) * dt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from spatialblend.api.python import SimulatorHooks


@dataclass
class RelationArrival:
    """Arrival event for one (shared_id, kernel_type) relation."""
    shared_id: int = 0
    kernel_type_name: str = ""
    source_frame: int = -1
    avatar_frame: int = -1
    lag_seconds: float = 0.0


class ArrivalLagHook(SimulatorHooks):
    """Tracks per-relation proximity and detects arrival on both sides.

    Usage:
        hook = ArrivalLagHook(dt=0.033, threshold=1.0)
        sim.set_hooks(hook)
        result = sim.run(actor, avatar, traj, 0.033)
        arrivals = hook.get_arrivals()
        mean_lag = hook.mean_lag()
    """

    def __init__(self, dt: float = 0.033, threshold: float = 1.0):
        super().__init__()
        self._dt = dt
        self._threshold = threshold
        # Key = (shared_id, kernel_type_name)
        self._source_arrival: Dict[Tuple[int, str], int] = {}
        self._avatar_arrival: Dict[Tuple[int, str], int] = {}
        self._known_keys: set = set()

    def on_proximity_update(self, frame, time, source_proximities,
                            avatar_proximities):
        for snap in source_proximities:
            kt = snap.kernel_type.name if hasattr(
                snap.kernel_type, 'name') else str(snap.kernel_type)
            key = (snap.shared_id, kt)
            self._known_keys.add(key)
            if key not in self._source_arrival and \
               snap.proximity >= self._threshold:
                self._source_arrival[key] = frame

        for snap in avatar_proximities:
            kt = snap.kernel_type.name if hasattr(
                snap.kernel_type, 'name') else str(snap.kernel_type)
            key = (snap.shared_id, kt)
            self._known_keys.add(key)
            if key not in self._avatar_arrival and \
               snap.proximity >= self._threshold:
                self._avatar_arrival[key] = frame

    def get_arrivals(self) -> List[RelationArrival]:
        """Return all detected arrival events with lag computation."""
        arrivals = []
        for key in self._known_keys:
            shared_id, kt_name = key
            arr = RelationArrival(
                shared_id=shared_id,
                kernel_type_name=kt_name,
                source_frame=self._source_arrival.get(key, -1),
                avatar_frame=self._avatar_arrival.get(key, -1),
            )
            if arr.source_frame >= 0 and arr.avatar_frame >= 0:
                arr.lag_seconds = \
                    (arr.avatar_frame - arr.source_frame) * self._dt
            arrivals.append(arr)
        return arrivals

    def mean_lag(self) -> float:
        """Mean arrival lag across relations that achieved contact on both sides."""
        arrivals = self.get_arrivals()
        lags = [a.lag_seconds for a in arrivals
                if a.source_frame >= 0 and a.avatar_frame >= 0]
        return sum(lags) / len(lags) if lags else 0.0

    def summary(self) -> str:
        """Human-readable summary of arrival events."""
        arrivals = self.get_arrivals()
        lines = []
        for a in arrivals:
            src = f"frame {a.source_frame}" if a.source_frame >= 0 else "never"
            avt = f"frame {a.avatar_frame}" if a.avatar_frame >= 0 else "never"
            lag = f"{a.lag_seconds:+.2f}s" if (
                a.source_frame >= 0 and a.avatar_frame >= 0) else "N/A"
            lines.append(
                f"  {a.kernel_type_name} sid={a.shared_id}: "
                f"src={src}, avt={avt}, lag={lag}")
        header = f"Relation Arrivals (threshold={self._threshold}):"
        mean = self.mean_lag()
        footer = f"  Mean lag: {mean:+.3f}s" if any(
            a.source_frame >= 0 and a.avatar_frame >= 0
            for a in arrivals) else "  No arrivals on both sides"
        return "\n".join([header] + lines + [footer])
