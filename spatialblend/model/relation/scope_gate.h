// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_RELATION_SCOPE_GATE_H_
#define SPATIALBLEND_RELATION_SCOPE_GATE_H_

#include <Eigen/Dense>

#include <unordered_map>
#include <vector>

#include "spatialblend/base/common/types.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/support.h"

namespace spatialblend {

// Per-family geometric scope gates (Facing line of sight, Spacing reachability,
// the SpatialBlend relation-family table). A candidate that fails its family's
// gate is OUT OF SCOPE: it is never grounded, never reaches the phase machine
// and never reaches the receiver. The gates run after the structural scope
// (ResolveScope) and, for AffordancePose only, the tag binding.
// The tags D1-D9 and MA-1 below are the scope decisions defined in
// spatialblend/space/testing/SCOPE_CASES.md ("Decisions").
//
// There is no radius, near field or activation gate: a candidate is in or out of scope on its geometry.

// Sight height used when a Facing anchor carries no height (a floor-level
// root anchor, z below kMinSightlineHeightM): an adult's chest height, the
// anchor the Facing family is normally bound to.
inline constexpr double kDefaultSightlineHeightM = 1.2;
inline constexpr double kMinSightlineHeightM = 0.3;

// Height of an Obstacle-role element whose definition carries no height (a
// flat Navigable footprint): 0, so a flat footprint never occludes. Give an
// occluding obstacle a volume (PolytopeSpatialDefinition, e.g.
// sb_set_element_box) or a Surface with its real vertices.
inline constexpr double kFootprintObstacleHeightM = 0.0;

// D8 deleted the 0.25 m reachability snap tolerance. A contact point is on the free
// space or it is not; the only slack left is the free space's own boundary hair (a
// point on a footprint's edge is its nearest free point up to that offset,
// ObstacleFreeSpace::kBoundaryHairM), the same constant the free space uses to place it.

// Facing (line-of-sight) occlusion model, built ONCE per evaluation (it scans every element of
// the tree) and then queried per candidate.
//
// Occluders, at ANY hierarchy depth:
//  - elements with GeometryRole::kObstacle, occluding up to their top:
//    a polytope's highest vertex, a surface's highest vertex, and
//    kFootprintObstacleHeightM (never occludes) for a flat footprint;
//  - declared wall polygons (NavigableSpatialDefinition::GetWall), full
//    height. LIMITATION: a declared wall polygon is a closed ring with no
//    openings, so it also blocks sight through a doorway in it. Model a wall
//    with a door as separate Obstacle volumes on either side of the gap.
// The referent itself, its ancestors and descendants never occlude it, nor
// does an obstacle whose footprint contains the eye (the couch the actor sits
// on) or a finite-height obstacle containing the sampled point (the table an
// object rests on). D1: ONE ray, from the eye to the look-at contact (the
// referent's own position when none is given); no other sample is tried.
//
// The ray ends at the referent's TOP. A referent with no declared geometry has
// no top of its own; `body_tops` gives one for such an element whose body the
// caller knows: a person (an actor's element) is seen at its body's top, the
// height of its actor's highest anchor above the element (a person is
// a Facing referent: across a table one sees the person's head, never
// the floor point under it). An element not in it is seen at its placement.
class OcclusionScene {
 public:
  explicit OcclusionScene(const ElementTree& tree,
                          std::unordered_map<element_id_t, double> body_tops = {});

  bool IsReferentVisible(const Eigen::Vector3d& eye, element_id_t referent_id,
                         const Eigen::Vector3d* contact_point_world =
                             nullptr) const;

  size_t NumOccluders() const { return occluders_.size(); }

 private:
  struct Occluder {
    element_id_t owner = kInvalidElementId;
    std::vector<element_id_t> owner_ancestors;
    std::vector<Eigen::Vector2d> ring;
    double top = 0.0;
    bool eye_exempt_capable = true;  // obstacles: ignored if they enclose the eye
  };
  const ElementTree& tree_;
  std::vector<Occluder> occluders_;
  std::unordered_map<element_id_t, double> body_tops_;
};

// Convenience: builds an OcclusionScene for one query.
bool IsReferentVisible(const ElementTree& tree, const Eigen::Vector3d& eye,
                       element_id_t referent_id,
                       const Eigen::Vector3d* contact_point_world = nullptr);

// Can `to` be reached from `from` by a collision-free walkable path in
// `navigable`? The one function the gate calls; it delegates to
// NavigableSpatialDefinition::QueryReachability (obstacle-aware free space,
// the single source of truth) with radius 0 -- point connectivity around the
// raw obstacle footprints, the same free space as the source geodesic, so
// "in scope" and "finite spacing R" coincide and the sender's body radius
// does not change a relation's scope (body clearance is a receiver
// feasibility concern). Unreachable when there is no free path, or when
// the approach point `to` is not on the free space (behind a wall, sealed by an
// obstacle ring): no snap tolerance (D8). `from` may be snapped any distance.
// Never a Euclidean judgement.
bool IsReferentReachable(const NavigableSpatialDefinition& navigable,
                         const Eigen::Vector3d& from,
                         const Eigen::Vector3d& to);

// Why a candidate is out of scope (mirrors the ScopeReason the
// extractor records).
enum class ScopeGateVerdict {
  kInScope,
  kNotVisible,         // ScopeReason::kExcludedNotVisible
  kUnreachable,        // ScopeReason::kExcludedUnreachable
  kOutsideParentRegion,  // ScopeReason::kExcludedOutsideParentRegion
};

// Family dispatch used by the extractor for every candidate after grounding.
//  - Facing (D1): visible from the anchor by ONE ray to the look-at contact,
//    using `occlusion` when given (build it once per evaluation), else a one-off
//    scene. Source side only.
//  - Spacing (D8): a geodesic path must exist in the source free space from the
//    anchor to the referent's support. The kernel's own geodesic to the contact
//    (Support::geodesic_length_m) is that path: finite = in scope, infinite = sealed.
//    A referent with no footprint (a person) is judged by IsReferentReachable to its
//    position. A referent that contains the anchor has geodesic 0 by containment.
//    The 0.25 m snap tolerance and the "anchor off the navigable" and "no geometry"
//    bypasses are deleted. Not gated only when `navigable` is null (no free space).
//  - AffordancePose (D3): in scope while the anchor is inside the footprint of
//    the referent's parent object (the nearest ancestor whose footprint has an area).
ScopeGateVerdict FamilyScopeGateVerdict(
    RelationKernelType family, const ElementTree& tree,
    element_id_t referent_id, const Eigen::Vector3d& anchor_world_position,
    const Support& support, const NavigableSpatialDefinition* navigable,
    const OcclusionScene* occlusion = nullptr);
// FamilyScopeGateVerdict(...) == kInScope.
bool PassesFamilyScopeGate(RelationKernelType family, const ElementTree& tree,
                           element_id_t referent_id,
                           const Eigen::Vector3d& anchor_world_position,
                           const Support& support,
                           const NavigableSpatialDefinition* navigable,
                           const OcclusionScene* occlusion = nullptr);

}  // namespace spatialblend

#endif  // SPATIALBLEND_RELATION_SCOPE_GATE_H_
