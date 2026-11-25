// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_ELEMENT_ENVELOPE_H_
#define SPATIALBLEND_ELEMENT_ENVELOPE_H_

#include <memory>
#include <vector>

#include "spatialblend/model/element/spatial_definition.h"

namespace spatialblend {

// Envelope: an element's enclosing CONTAINMENT region, derived in the CORE from a
// primitive collider when no explicit boundary (affordance/polygon) is given.
//
// Why the core (not each client): the cross-space spacing relation needs the
// referent to HAVE a boundary. Without one, proximity is measured to the element
// CENTRE, so an avatar can never reach prox=1 (can't reach a solid centre) and the
// reconstructed clone keeps a residual stand-off — it never lands "at" the piece.
// With a boundary, proximity is boundary-relative (reaches 1 at the surface) and
// DecodeSupportPoint maps the source's approach point onto the target boundary, so
// at prox->1 the clone realizes at the mapped contact point. Clients only supply
// primitive collider params (data); the primitive->footprint conversion lives here.
enum class ColliderType { kBox = 0, kSphere = 1, kCapsule = 2 };

// Build a Containment SpatialDefinition (a NavigableSpatialDefinition wrapping the
// 2D footprint polygon, role = kContainment) from a primitive collider expressed
// in the element's LOCAL frame, Z-up. The footprint is the XY projection.
//   kBox:     params = {cx, cy, cz, sx, sy, sz}  (center, FULL size)
//   kSphere:  params = {cx, cy, cz, r}
//   kCapsule: params = {cx, cy, cz, r, height, axis}  (axis: 0=X, 1=Y, 2=Z; height
//             is the TOTAL height incl. caps, Unity convention)
// Returns nullptr on malformed params / degenerate size. circle_segments controls
// the polygon resolution of rounded footprints (sphere / capsule caps).
//
// yaw: rotation (radians, about Z, CCW) applied about the collider's own
// center (cx, cy) BEFORE the footprint is baked into the returned
// world-absolute NavigableSpatialDefinition (MakeEnvelope's box case
// previously ignored rotation entirely, so a rotated box referent's
// Containment footprint silently reverted to axis-aligned). Sphere/circle
// footprints are rotationally symmetric and unaffected; capsule stadiums are
// not yet rotated here (no repro/caller needs it yet).
std::shared_ptr<SpatialDefinition> MakeEnvelope(ColliderType type,
                                                const std::vector<double>& params,
                                                int circle_segments = 16,
                                                double yaw = 0.0);

}  // namespace spatialblend

#endif  // SPATIALBLEND_ELEMENT_ENVELOPE_H_
