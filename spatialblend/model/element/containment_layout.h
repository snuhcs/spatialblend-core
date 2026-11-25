// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/model/element/containment_layout.h
//
// Scene validation: the containment layout must be a
// tree of nested, non-overlapping regions, because the scope of a pose is the
// deepest containing element and the layer has no tie-break.
//   - sibling elements with a footprint, of ANY role, must not overlap (edge
//     contact is fine; one thing is one element with parts or
//     sub-regions, or a parent with children);
//   - a Containment/Navigable child that sticks out of its Containment/Navigable
//     parent EXTENDS the parent's scope, whatever the gap between them
//     (scope_extension.h): a child across a gap
//     is bridged to the parent's nearest point. A child of an actor's kind
//     (IsActorBound) never extends a parent;
//   - an extending child is a footprint outside its parent, so it must not overlap
//     any footprint outside its own line (neither an ancestor nor a descendant): the
//     extended regions stay a tree of nested, non-overlapping regions, as the
//     sibling rule keeps the declared ones.
// Every check reads the DECLARED definitions (Element::GetDeclaredSpatialDefinition):
// the extension is derived from them, never checked against itself.
// Violations are rejected where the scene is registered (Space::SetElement*),
// never repaired or tie-broken at solve time. Tolerance: the geometry epsilon
// the relation layer already uses (relation_kernel.h kEpsilon, 1e-9) for the
// parent check, RELATIVE to the smaller footprint's area; for sibling overlap
// the precision of the data as a LENGTH: a coordinate is known to
// eps_f * max(1, |x|) (eps_f = 2^-23, single-precision matrices), so touching
// footprints may intersect in a strip of twice that width along the smaller
// one's boundary, and edge contact of any size passes.
#ifndef SPATIALBLEND_ELEMENT_CONTAINMENT_LAYOUT_H_
#define SPATIALBLEND_ELEMENT_CONTAINMENT_LAYOUT_H_

#include <memory>
#include <optional>
#include <string>

#include "spatialblend/base/common/types.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {

// Error codes, mirrored by the C API (sb_scene_status_t).
enum class SceneErrorCode : int {
  kOk = 0,
  kInvalidArgument = 1,
  kContainmentOverlap = 2,
  kContainmentEscapesParent = 3,  // no longer produced (a child extends its parent at any distance)
};

// The minimum bridge radius: two regions that touch at a point are joined by a
// bridge of this width (a union of point-touching rings is two components).
constexpr double kBridgeSlackM = 1e-3;

// A gap up to this (metres) is a NavMesh carve band (0.1-0.3 m, with margin): the
// bridge across it is the band (the points within the gap of both regions), and it
// is how far from a child and its bridge a pocket of the union is filled
// (scope_extension.h). A child at any larger distance is still accepted: its bridge
// is a corridor (ScopeExtensionBridge).
constexpr double kScopeExtensionNearbyM = 0.5;

// The widest corridor that joins a far child to its parent (metres): the child's
// width across the corridor, at most this. 2 m is a generous two-way corridor (two
// wheelchairs pass in 1.8 m, ISO 21542 / ADA 403.5.3 passing space), so any child is
// reachable through it, while a wide child far away (a 30 m long band) adds a
// corridor, not a 30 m wide slab of floor nobody declared.
constexpr double kScopeExtensionCorridorMaxWidthM = 2.0;

// The bridge between a parent's region and a child region that does not overlap it
// (scope_extension.h), the one definition admission checks and the build unites.
//  - gap <= kScopeExtensionNearbyM (the carve band): the points within r = gap +
//    kBridgeSlackM of both (it reaches into both regions by up to r);
//  - a larger gap: a CORRIDOR, the rectangle along the segment between the two
//    regions' nearest points, as wide as the child across it (at most
//    kScopeExtensionCorridorMaxWidthM), reaching kBridgeSlackM into both.
// Admission checks its part outside the parent against every footprint outside the
// child's line (siblings and the parent's other descendants included).
Polygon2D ScopeExtensionBridge(const Polygon2D& parent, const Polygon2D& child, double gap);

// An actor's own element (Element::GetGroundingEntityId set by Space::SetActorElement),
// or an element of its BODY: below an actor's element that is not itself a region. An
// actor bound to a room (a region it is placed in) does not make the room's children
// an actor's. The kind that never extends a parent's scope.
bool IsActorBound(const ElementTree& tree, element_id_t id);

// Whether `id` is a region (Containment/Navigable role with a footprint) whose
// declared footprint sticks out of its region parent's declared footprint, and so
// and so extends the parent's scope (at any distance).
bool ExtendsParentScope(const ElementTree& tree, element_id_t id);

struct ContainmentViolation {
  SceneErrorCode code = SceneErrorCode::kOk;
  element_id_t first = kInvalidElementId;   // the sibling / the child
  element_id_t second = kInvalidElementId;  // the other sibling / the parent
  std::string message;                      // names both elements
};

// Checks `id` and its whole subtree against their parents and siblings (a moved
// or re-defined element changes all of them). Elements without a footprint are
// skipped; the parent-escape check covers Containment/Navigable elements only.
// `extends` (optional) receives whether any element of the subtree extends its
// parent's scope (ExtendsParentScope), when no violation is found.
std::optional<ContainmentViolation> FindContainmentViolation(
    const ElementTree& tree, element_id_t id, bool* extends = nullptr);

// The footprint of a Containment/Navigable element in world coordinates (its
// effective definition: the extended region when children extend it), or nullptr
// when it has none (fewer than three vertices, no definition).
std::shared_ptr<Polygon2D> RegionWorldFootprint(const ElementTree& tree,
                                                element_id_t id);
// The same for the declared definition.
std::shared_ptr<Polygon2D> DeclaredWorldFootprint(const ElementTree& tree,
                                                  element_id_t id);
// The declared world footprint as the scope extension unites it: repaired when it is
// invalid only by zero-area features (spikes, keyhole slits: Polygon2D::Repaired),
// nullptr when it has none or cannot be repaired. Admission and the build both use it.
std::shared_ptr<Polygon2D> UnionFootprint(const ElementTree& tree, element_id_t id);

// Transfer-leaf equivalence: the two regions are
// the SAME region in two spaces when their polygons are congruent under a rigid
// transform (rotation and translation, no reflection). The vertex order, the
// starting vertex and collinear or repeated vertices are irrelevant (the corners
// are compared). Tolerance: the geometry epsilon above, relative to the smaller
// footprint's size (epsilon times the square root of its area, the length whose
// square is the area tolerance of the layout checks); no other epsilon.
bool FootprintsCongruent(const Polygon2D& a, const Polygon2D& b);

}  // namespace spatialblend

#endif  // SPATIALBLEND_ELEMENT_CONTAINMENT_LAYOUT_H_
