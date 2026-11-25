#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Regenerate the raw-geodesic lower-bound references in obstacle_free_space_test.cc.

Run from any directory:
    python3 tools/geodesic_fixture_oracle.py

No third-party packages, instrumented executable, or /tmp capture is needed.
The coordinates are read directly from the committed C++ fixtures and converted
through float.as_integer_ratio(), exactly reproducing their binary64 values.
All predicates, intersections, projections and graph topology use Fraction
arithmetic, with no geometric epsilon. Edge lengths and Dijkstra sums use
Decimal at 80-digit precision. The output includes the complete winning routes.

The crowded and two-box fixtures have convex footprints and no coincident
interior seams or sub-hair gaps. The reference domain is the original walkable
minus footprint interiors. Sealing a narrow gap elsewhere can only increase a
metric relative to this raw-domain reference. A contact inside its referent is
projected to that footprint's nearest boundary foot and its depth is added,
matching the explicit endpoint-distance convention exercised by the fixture.
"""

from decimal import Decimal, localcontext
from fractions import Fraction
import heapq
import json
from pathlib import Path
import re


ZERO, ONE = Fraction(0), Fraction(1)
POINT_RE = r"\{\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*\}"


def scalar(text):
    return Fraction.from_float(float(text))


def points(text):
    return [(scalar(x), scalar(y)) for x, y in re.findall(POINT_RE, text)]


def sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def scale(a, s):
    return a[0] * s, a[1] * s


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def decimal(x):
    return Decimal(x.numerator) / Decimal(x.denominator)


def norm(a):
    return decimal(dot(a, a)).sqrt()


def edges(ring):
    return zip(ring, ring[1:] + ring[:1])


def ccw(ring):
    return ring if sum(cross(a, b) for a, b in edges(ring)) > 0 else ring[::-1]


def inside_convex(p, ring):
    return all(cross(sub(b, a), sub(p, a)) > 0 for a, b in edges(ring))


def enters_convex_interior(a, b, ring):
    """Exact clipping against strict convex half-spaces, including tangency."""
    lower, upper = ZERO, ONE
    direction = sub(b, a)
    for p, q in edges(ring):
        edge = sub(q, p)
        initial = cross(edge, sub(a, p))
        slope = cross(edge, direction)
        if slope == 0:
            if initial <= 0:
                return False
        else:
            crossing = -initial / slope
            if slope > 0:
                lower = max(lower, crossing)
            else:
                upper = min(upper, crossing)
        if upper <= lower:
            return False
    return upper > lower


def on_segment(p, a, b):
    return cross(sub(b, a), sub(p, a)) == 0 and dot(sub(p, a), sub(p, b)) <= 0


def in_walkable(p, ring):
    if any(on_segment(p, a, b) for a, b in edges(ring)):
        return True
    inside = False
    for a, b in edges(ring):
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x > p[0]:
                inside = not inside
    return inside


def intersection_parameters(a, b, p, q):
    direction, edge = sub(b, a), sub(q, p)
    denominator = cross(direction, edge)
    if denominator != 0:
        t = cross(sub(p, a), edge) / denominator
        u = cross(sub(p, a), direction) / denominator
        return [t] if 0 <= t <= 1 and 0 <= u <= 1 else []
    if cross(direction, sub(p, a)) != 0 or dot(direction, direction) == 0:
        return []
    return [t for point in (p, q)
            if 0 <= (t := dot(sub(point, a), direction) / dot(direction, direction)) <= 1]


def stays_in_walkable(a, b, ring):
    if not in_walkable(a, ring) or not in_walkable(b, ring):
        return False
    cuts = sorted({ZERO, ONE} | {t for p, q in edges(ring)
                                for t in intersection_parameters(a, b, p, q)})
    return all(in_walkable(add(a, scale(sub(b, a), (lo + hi) / 2)), ring)
               for lo, hi in zip(cuts, cuts[1:]))


class ReferenceGraph:
    def __init__(self, walkable, obstacles):
        self.walkable = ccw(walkable)
        self.obstacles = {key: ccw(ring) for key, ring in obstacles.items()}
        rings = [self.walkable, *self.obstacles.values()]
        nodes = {p for ring in rings for p in ring}
        for i, left in enumerate(rings):
            for right in rings[i + 1:]:
                for a, b in edges(left):
                    for p, q in edges(right):
                        nodes.update(add(a, scale(sub(b, a), t))
                                     for t in intersection_parameters(a, b, p, q))
        self.nodes = sorted(p for p in nodes if in_walkable(p, self.walkable)
                            and not any(inside_convex(p, ring) for ring in self.obstacles.values()))
        self.adjacency = [[] for _ in self.nodes]
        for i, a in enumerate(self.nodes):
            for j, b in enumerate(self.nodes[:i]):
                if self.visible(a, b):
                    length = norm(sub(a, b))
                    self.adjacency[i].append((j, length))
                    self.adjacency[j].append((i, length))

    def visible(self, a, b):
        if not stays_in_walkable(a, b, self.walkable):
            return False
        return not any(enters_convex_interior(a, b, ring) for ring in self.obstacles.values())

    def distance(self, source, contact, own=None):
        depth = Decimal(0)
        if own is not None and inside_convex(contact, self.obstacles[own]):
            candidates = []
            for a, b in edges(self.obstacles[own]):
                edge = sub(b, a)
                t = min(ONE, max(ZERO, dot(sub(contact, a), edge) / dot(edge, edge)))
                foot = add(a, scale(edge, t))
                candidates.append((dot(sub(contact, foot), sub(contact, foot)), foot))
            squared, foot = min(candidates)
            depth = decimal(squared).sqrt()
            contact = foot
        distances = [Decimal("Infinity")] * len(self.nodes)
        next_node = [None] * len(self.nodes)
        queue = []
        for i, p in enumerate(self.nodes):
            if self.visible(p, contact):
                distances[i] = norm(sub(p, contact))
                heapq.heappush(queue, (distances[i], i))
        while queue:
            length, i = heapq.heappop(queue)
            if length != distances[i]:
                continue
            for j, edge_length in self.adjacency[i]:
                candidate = length + edge_length
                if candidate < distances[j]:
                    distances[j], next_node[j] = candidate, i
                    heapq.heappush(queue, (candidate, j))
        best, first = Decimal("Infinity"), None
        if self.visible(source, contact):
            best = norm(sub(source, contact))
        for i, p in enumerate(self.nodes):
            candidate = norm(sub(source, p)) + distances[i]
            if candidate < best and self.visible(source, p):
                best, first = candidate, i
        route = [source]
        while first is not None:
            route.append(self.nodes[first])
            first = next_node[first]
        route.append(contact)
        return {"distance_m": str(best + depth), "contact_depth_m": str(depth),
                "path": [[str(decimal(x)), str(decimal(y))] for x, y in route]}


def fixture_source():
    root = Path(__file__).resolve().parents[1]
    return (root / "spatialblend/geometry/context/obstacle_free_space_test.cc").read_text()


def regenerate():
    text = fixture_source()
    crowded = text.split("TEST(ObstacleFreeSpace, CrowdedGeodesicsAreLipschitzAcrossRouteAdmission)", 1)[1]
    crowded = crowded.split("\nTEST(", 1)[0]
    walkable = points(re.search(r"const auto walkable = poly\(\{(.*?)\}\);", crowded, re.S)[1])
    obstacle_text = crowded.split("const std::vector<FreeSpaceObstacle> obstacles = {", 1)[1].split("\n  };", 1)[0]
    obstacles = {int(key): points(body) for key, body in
                 re.findall(r"\{(\d+), poly\(\{(.*?)\}\)\}", obstacle_text, re.S)}
    contact_text = crowded.split("const std::vector<Contact> contacts = {", 1)[1].split("\n  };", 1)[0]
    contacts = [(int(key), (scalar(x), scalar(y))) for key, x, y in
                re.findall(r"\{(\d+), \{([-+\d.eE]+), ([-+\d.eE]+)\}", contact_text)]
    source = tuple(scalar(x) for x in re.search(r"stopped\(([^,]+), ([^)]+)\)", crowded).groups())
    graph = ReferenceGraph(walkable, obstacles)
    result = {"decimal_digits": 80, "input": "exact binary64 coordinates from committed C++ fixtures",
              "crowded": {str(key): graph.distance(source, contact, key) for key, contact in contacts}}
    boxes = text.split("TEST(ObstacleFreeSpaceMetric, OverlappingRotatedBoxesDoNotAdmitAUnionBoundaryShortcut)", 1)[1]
    boxes = boxes.split("\nTEST(", 1)[0]
    b2 = points(re.search(r"const std::vector<V> b2 = \{(.*?)\};", boxes, re.S)[1])
    b27 = points(re.search(r"const std::vector<V> b27 = \{(.*?)\};", boxes, re.S)[1])
    room = [(ZERO, ZERO), (Fraction(10), ZERO), (Fraction(10), Fraction(10)), (ZERO, Fraction(10))]
    graph = ReferenceGraph(room, {2: b2, 27: b27})
    result["two_box"] = graph.distance(b27[3], b27[0])
    return result


if __name__ == "__main__":
    with localcontext() as context:
        context.prec = 80
        print(json.dumps(regenerate(), indent=2))
