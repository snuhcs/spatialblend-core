// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_RELATION_RELATION_KERNEL_H_
#define SPATIALBLEND_RELATION_RELATION_KERNEL_H_

#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <vector>

#include "spatialblend/base/common/state_type.h"
#include "spatialblend/model/element/element.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/geometry_role.h"
#include "spatialblend/model/relation/grounded_relation.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/geometry/polygon_coordinate_mapper.h"
#include "spatialblend/model/relation/support.h"
#include "spatialblend/model/actor/actor_motion_tracker.h"

namespace spatialblend {

struct SatisfactionMetric {
  double remaining = 0.0;       // kernel-specific distance/angle/error
  double estimated_time = 0.0;  // seconds (remaining / rate)
};

// Robust time-to-satisfaction shared by all relation kernels.
//
// The naive estimate trusts frame-to-frame progress (prev_remaining -
// remaining) whenever it is barely positive (> 1e-6). For a NEAR-STATIONARY
// source that is catastrophic: a few millimetres of VR/sensor jitter make
// progress momentarily positive, so progress_time = remaining / progress
// collapses to a small value and the realizer reads the relation as "imminent".
// One frame later the jitter reverses, progress goes negative, and the estimate
// snaps back to the large scalar fallback. That 0<->large flapping in
// estimated_time drives the realizer's imminence term 0<->1, and because the
// blended supports are spatially far apart the clone visibly oscillates even
// though the SOURCE is standing still (the SpatialBlend clone-chatter bug).
//
// Fix: trust directional progress ONLY when the actor is genuinely, sustainedly
// moving — the motion tracker's EMA-smoothed speed must clear kMotionFloor. When
// still, fall back to the scalar estimate (large -> imminence ~ 0), so a
// stationary source yields a stable, low-imminence weight. This relies on the
// speed signal being the magnitude of ActorMotionTracker's smoothed SIGNED
// velocity, which is zero-mean under jitter — so the floor cleanly separates
// "standing still" from "walking toward the referent". (Smoothing the rectified
// |Δp|/dt instead would keep a positive jitter bias above the floor and defeat
// this gate; see actor_motion_tracker.cc.)
inline double EstimateTimeToSatisfaction(double remaining, double prev_remaining,
                                         bool has_previous,
                                         double smoothed_speed) {
  // Steps-to-arrival that is BOTH stable (no lurch) AND directional (commits to
  // the referent you actually walk toward, not the centroid of all of them).
  //
  // Two failure modes had to be avoided together:
  //  - v2's original remaining / (prev_remaining - remaining) put the noisy
  //    frame-delta MAGNITUDE in the denominator -> collapsed on small moves and
  //    lurched the clone ~1 m ("too fast + unpredictable").
  //  - Dividing by the smoothed speed alone (remaining / speed) is smooth but
  //    NON-directional: every referent at a similar range looks equally
  //    imminent, so the position weights go ~uniform and the clone sits at the
  //    furniture CENTROID (lands on the TV) instead of the table being approached.
  //
  // Fix: use the smoothed speed for the MAGNITUDE (stable, like v1's trajectory-
  // derived velocity), but gate on the SIGN of the change (am I closing on THIS
  // referent?) for DIRECTION. The sign is robust where the magnitude is not.
  // Below the motion floor (standing/jittering), or for a referent whose
  // distance is not decreasing (the TV you're walking away from), the estimate
  // is large -> imminence ~ 0.
  constexpr double kEpsilon = 1e-9;  // float-zero guard, not a tuning knob
  const bool closing = has_previous && (remaining < prev_remaining);
  if (closing && smoothed_speed > kEpsilon) {
    return remaining / smoothed_speed;  // stable magnitude, directional gate
  }
  // Not closing on THIS referent (standing / receding / tangential): arrives in
  // infinite steps -> dropped by the anneal & progressive guards (never crawled
  // toward). v1 ComputeExpectedInteractionTime semantics; no fallback constant.
  return std::numeric_limits<double>::infinity();
}

// The footprint test of the scope layer, the one place a point is "inside" an element:
// the polygon point test of its spatial definition. Definitions such as
// PolytopeSpatialDefinition keep their geometry in the owning element's local frame;
// containment must use that frame, while baked Navigable/Surface definitions already
// consume world coordinates. This mirrors the grounding-rule frame contract rather than
// applying a tree inverse unconditionally.
//
// The footprint of an element-local definition is the image of its polygon on the GROUND plane
// (the Space is 2D): the polygon's local x,y placed by the 2D affine part of the element's world
// transform. For every placement that only turns about the vertical axis (a yaw, a translation, a
// height) this is the test it always was, exactly. For a placement that tips the polygon out of the
// ground plane, the point through the full 3D inverse with its out-of-plane part dropped read the
// point's projection ALONG the polygon's normal: an upright polygon (the scene-geometry adapter's
// tabletop plane, registered under Rx(-90)) "held" every point of the strip it stands on, at any
// distance from it. The ground image of an upright polygon has no area, so nothing is inside it
// (v1 TransformToVolumeSpace flags the same transform as degenerate).
inline bool FootprintHoldsIn(const SpatialDefinition* definition, const Element& element,
                             const ElementTree& tree, const Eigen::Vector3d& position) {
  if (!definition) return false;
  if (!definition->RequiresElementWorldTransform())
    return definition->Contains(Eigen::Vector2d(position.x(), position.y()));
  const Eigen::Matrix4d world = tree.GetWorldTransform(element.GetId());
  if (GroundImageIsSegment(world)) return false;  // the ground image is a segment: never inside
  const Eigen::Matrix4d ground = GroundFrame(world);
  const Eigen::Vector4d local = ground.inverse() * Eigen::Vector4d(position.x(), position.y(), 0.0, 1.0);
  return definition->Contains(Eigen::Vector2d(local.x(), local.y()));
}

// THE scope test: the element's SCOPE region (Element::GetScopeDefinition: its declared region
// united with its region children's scope regions, without the bridges and corridors that only
// connect the walkable). A pose in a bridge or corridor gap is in the parent's
// walkable, not in its scope.
inline bool FootprintHolds(const Element& element, const ElementTree& tree,
                           const Eigen::Vector3d& position) {
  return FootprintHoldsIn(element.GetScopeDefinition(), element, tree, position);
}

// The same test on the element's DECLARED definition: the object itself, without the scope its
// children extend (element/scope_extension.h). The in-object replay (containment transfer) is
// the object's own.
inline bool DeclaredFootprintHolds(const Element& element, const ElementTree& tree,
                                   const Eigen::Vector3d& position) {
  return FootprintHoldsIn(element.GetDeclaredSpatialDefinition(), element, tree, position);
}

// The ground image of a definition of an element, every component and hole: an element-local polygon
// is placed by the element's ground frame; any other definition keeps its ring in its own frame (the
// identity for a world-absolute one, the surface's frame for a Surface).
inline Polygon2D GroundImageOf(const SpatialDefinition* definition, const Element& element,
                               const ElementTree& tree) {
  if (!definition) return Polygon2D();
  const std::shared_ptr<Polygon2D> boundary = definition->GetBoundary();
  if (!boundary) return Polygon2D();
  const Eigen::Matrix4d frame = definition->RequiresElementWorldTransform()
                                    ? GroundFrame(tree.GetWorldTransform(element.GetId()))
                                    : Eigen::Matrix4d(definition->GetTransform());
  return boundary->Transformed(frame.topLeftCorner<2, 2>(), frame.block<2, 1>(0, 3));
}

// The ground image of an element's SCOPE region (see FootprintHolds) as a region: every component
// (a child across a gap is a component of its own, no bridge joins it) and every hole (a pocket the
// union encloses, element/scope_extension.h).
inline Polygon2D FootprintImageRegion(const Element& element, const ElementTree& tree) {
  return GroundImageOf(element.GetScopeDefinition(), element, tree);
}

// The outer ring of the largest component of that image. It is the whole scope region only when the
// region is one component without holes (V1NormalizedDistanceOf checks); a caller that needs every
// component reads FootprintImageRegion.
inline std::vector<Eigen::Vector2d> FootprintImage(const Element& element, const ElementTree& tree) {
  return FootprintImageRegion(element, tree).GetPoints();
}

// The ground image of an element's WALKABLE region (Element::GetSpatialDefinition: the scope plus
// the bridges and corridors), the outer ring of its largest component: what the free space is cut
// from (diagnostics; path planning reads the navigable definition itself).
inline std::vector<Eigen::Vector2d> WalkableFootprintImage(const Element& element, const ElementTree& tree) {
  return GroundImageOf(element.GetSpatialDefinition(), element, tree).GetPoints();
}

// Which of several footprints that all hold a point the point is "most inside" of: v1's rule, verbatim
// (spatialblend/space/space.cc ApplySnapTransformIfInsideTarget, lines 947-1000), the one v1's receiver
// snaps to when the source is inside several volumes:
//     normalized distance = |point - centroid(footprint)| / max over its vertices |vertex - centroid|
// the smallest wins, and an exact tie keeps the first (strict `<`). A footprint whose radius is below
// math::Constants::kEpsilon (1e-6) has no normalized distance and is skipped, as in v1. This is the whole
// definition of the entered referent where footprints overlap: the answer is ONE referent, never a blend
// of the several (a mean of their poses lies inside none of them).
constexpr double kV1MinFootprintRadiusM = 1e-6;
inline double V1NormalizedDistance(const std::vector<Eigen::Vector2d>& ring, const Eigen::Vector2d& point) {
  const Polygon2D polygon(ring);
  const Eigen::Vector2d centre = PolygonCoordinateMapper::ComputeCentroid(polygon);
  double radius = 0.0;
  for (const Eigen::Vector2d& vertex : polygon.GetPoints()) radius = std::max(radius, (vertex - centre).norm());
  if (radius < kV1MinFootprintRadiusM) return std::numeric_limits<double>::infinity();
  return (point - centre).norm() / radius;
}

// The same for an element's footprint (its scope region): a footprint with holes or with several
// components takes its centroid over the whole region (area-weighted over the components, a hole is
// not in it) and its radius over every component's vertices; one simple ring is exactly the ring
// form above.
inline double V1NormalizedDistanceOf(const Element& element, const ElementTree& tree, const Eigen::Vector2d& point) {
  const SpatialDefinition* definition = element.GetScopeDefinition();
  const std::shared_ptr<Polygon2D> boundary = definition ? definition->GetBoundary() : nullptr;
  if (!boundary || (boundary->Holes().empty() && boundary->ExtraRings().empty()))
    return V1NormalizedDistance(FootprintImage(element, tree), point);
  const Polygon2D region = FootprintImageRegion(element, tree);
  std::vector<std::vector<Eigen::Vector2d>> outers{region.GetPoints()};
  for (const auto& ring : region.ExtraRings()) outers.push_back(ring);
  Eigen::Vector2d moment = Eigen::Vector2d::Zero();
  double area = 0.0;
  for (const auto& ring : outers) {
    const Polygon2D c(ring);
    moment += c.GetArea() * c.GetCenter();
    area += c.GetArea();
  }
  for (const auto& ring : region.Holes()) {
    const Polygon2D h(ring);
    moment -= h.GetArea() * h.GetCenter();
    area -= h.GetArea();
  }
  if (!(area > 0.0)) return std::numeric_limits<double>::infinity();
  const Eigen::Vector2d centre = moment / area;
  double radius = 0.0;
  for (const auto& ring : outers)
    for (const Eigen::Vector2d& vertex : ring) radius = std::max(radius, (vertex - centre).norm());
  if (radius < kV1MinFootprintRadiusM) return std::numeric_limits<double>::infinity();
  return (point - centre).norm() / radius;
}

// An element a point can be inside as a REFERENT: any element with a footprint that is not
// an interaction region (the interaction regions are AffordancePose referents, never
// Spacing's). The containment tree of the scope layer is made of these, whatever their
// geometry role (a region, an object's footprint, a seat): the root point inside one is
// "inside a Spacing referent".
inline bool IsEnterableReferent(const Element& element) {
  return element.GetSpatialDefinition() != nullptr && !IsInteractionRole(element.GetGeometryRole());
}

// The descent of the scope layer: from `start`, into the referent whose footprint holds the
// point, and again below it, until no child's footprint holds it. This is the ONE
// definition of the containment tree membership of a pose.
//  - container:     the deepest element holding the point (`start` when none does). A
//                   flagged containment-transfer element is a LEAF (D5):
//                   the deepest flagged element on the chain is the container, whatever
//                   lies below it.
//  - entered:       the referents the point is INSIDE at the deepest level: {container}
//                   when the point went below `start`; empty when nothing below `start`
//                   holds it. Where several sibling footprints hold the point (sibling regions
//                   may not overlap; furniture footprints may) they are all
//                   entered and the container stays above them, UNLESS the caller asks for
//                   the Spacing answer (`choose_overlap`): the entered referent is then the
//                   ONE v1 answers, the smallest normalized distance (V1NormalizedDistance),
//                   and the descent continues below it. A scope that is the objects of a
//                   region (AffordancePose's proximal scope, every affordance of every object
//                   whose region holds the anchor) keeps all the holders.
//  - transfer_leaf: the deepest flagged element on the chain, or kInvalidElementId.
//  - not_objects: the persons and the subject (never the objects of a container); only with skip_transparent.
//  - skip_transparent: Spacing's descent skips the elements that are transparent to scope (IsScopeTransparent).
// MA-1, transparency (SP-D7-05): an element with no shared id that carries nothing shared below it is
// no referent of anything and, for the Spacing scope, does not exist: the descent neither enters it nor stops in it, and the scope
// is exactly the one the tree has with that element deleted. (An unshared region that holds a shared element is not transparent: that
// element is a referent and the region is its container, with its border.)
// The elements the caller knows are not objects of a container (the other actors' elements and the subject's own: persons are Spacing
// referents at any depth, D9, and never the objects of the container they hang from). A shared element with no footprint is NOT one of
// these by itself: a point referent is a legitimate object.
inline bool IsNotObject(element_id_t id, const std::vector<element_id_t>* not_objects) {
  return not_objects && std::find(not_objects->begin(), not_objects->end(), id) != not_objects->end();
}

inline bool IsScopeTransparent(const ElementTree& tree, element_id_t id,
                               const std::vector<element_id_t>* not_objects = nullptr) {
  const Element* e = tree.GetElement(id);
  if (!e || e->GetSharedId() != kInvalidSharedElementId) return false;
  for (element_id_t below : tree.GetSubtreeIds(id)) {
    if (below == id) continue;
    const Element* b = tree.GetElement(below);
    if (b && b->GetSharedId() != kInvalidSharedElementId && !IsNotObject(below, not_objects)) return false;
  }
  return true;
}

// Whether a container holds a shared object: a direct, non-interaction child with a counterpart that is not a person or the subject (the
// objects of its Spacing scope).
inline bool HasSharedObject(const ElementTree& tree, element_id_t id,
                            const std::vector<element_id_t>* not_objects = nullptr) {
  for (element_id_t child_id : tree.GetDirectChildIds(id)) {
    const Element* c = tree.GetElement(child_id);
    if (c && c->GetSharedId() != kInvalidSharedElementId && !IsInteractionRole(c->GetGeometryRole()) &&
        !IsNotObject(child_id, not_objects))
      return true;
  }
  return false;
}

// An unshared container with no shared object cannot be the scope of a pose: it has no counterpart and nothing to ground against. For the
// Spacing scope it does not exist either (the pose is in the nearest ancestor that does), even when something shared is nested deeper
// below it that the pose is not inside.
inline bool IsScopeEmptyContainer(const ElementTree& tree, element_id_t id,
                                  const std::vector<element_id_t>* not_objects = nullptr) {
  const Element* e = tree.GetElement(id);
  return e && e->GetSharedId() == kInvalidSharedElementId && !HasSharedObject(tree, id, not_objects);
}

// Whether a pose at `position` is inside a shared element below `id` (so that descending into `id` can still end in a scope).
inline bool HoldsSharedBelow(const ElementTree& tree, element_id_t id, const Eigen::Vector3d& position) {
  for (element_id_t below : tree.GetSubtreeIds(id)) {
    if (below == id) continue;
    const Element* b = tree.GetElement(below);
    if (b && b->GetSharedId() != kInvalidSharedElementId && IsEnterableReferent(*b) && FootprintHolds(*b, tree, position)) return true;
  }
  return false;
}

// A containment-transfer flag acts only on an element with a counterpart: no receiver can map an unshared one. For the Spacing scope
// (skip_transparent) an unshared flagged element is an ordinary container, exactly the tree with the flag cleared.
inline bool IsTransferLeafFlag(const ElementTree& tree, const Element& el, bool skip_transparent) {
  (void)tree;
  return el.HasContainmentTransfer() && (!skip_transparent || el.GetSharedId() != kInvalidSharedElementId);
}

// A flagged element is the transfer leaf of a pose inside the OBJECT: its declared footprint. A pose
// in the scope its children extend is in its scope, not in the object, and is no replay through its
// chart. Without an extension the two footprints are one.
inline bool IsTransferLeafAt(const ElementTree& tree, const Element& el, bool skip_transparent,
                             const Eigen::Vector3d& position) {
  return IsTransferLeafFlag(tree, el, skip_transparent) &&
         (!el.HasScopeExtension() || DeclaredFootprintHolds(el, tree, position));
}

struct ScopeDescent {
  element_id_t container = kInvalidElementId;
  std::vector<element_id_t> entered;
  element_id_t transfer_leaf = kInvalidElementId;
};

inline ScopeDescent DescendScope(const Eigen::Vector3d& position, element_id_t start_id,
                                 const ElementTree& tree, bool choose_overlap = false,
                                 bool skip_transparent = false,
                                 const std::vector<element_id_t>* not_objects = nullptr) {
  ScopeDescent out;
  out.container = start_id;
  element_id_t node = start_id;
  std::vector<element_id_t> chain;  // the elements entered, outermost first
  for (;;) {
    std::vector<element_id_t> holding;
    for (element_id_t child_id : tree.GetDirectChildIds(node)) {
      const Element* child = tree.GetElement(child_id);
      if (child && IsEnterableReferent(*child) && FootprintHolds(*child, tree, position) &&
          !(skip_transparent && IsScopeTransparent(tree, child_id, not_objects)))
        holding.push_back(child_id);
    }
    if (holding.empty()) break;
    if (skip_transparent && holding.size() > 1) {
      // The pop below must not beat a shared overlapping sibling: a holder that cannot be a scope (IsScopeEmptyContainer) and leads to no
      // shared element holding the pose is not a candidate when another holder is (the tree with it deleted).
      std::vector<element_id_t> viable;
      for (element_id_t h : holding)
        if (!IsScopeEmptyContainer(tree, h, not_objects) || HoldsSharedBelow(tree, h, position)) viable.push_back(h);
      if (!viable.empty()) holding = viable;
    }
    if (holding.size() > 1 && !choose_overlap) {  // overlapping footprints: all are entered
      out.entered = holding;
      break;
    }
    if (holding.size() > 1) {  // the Spacing answer: the one v1 answers (V1NormalizedDistance)
      const Eigen::Vector2d at(position.x(), position.y());
      element_id_t chosen = holding.front();
      double best = std::numeric_limits<double>::infinity();
      for (element_id_t id : holding) {
        const double d = V1NormalizedDistanceOf(*tree.GetElement(id), tree, at);
        if (d < best) {
          best = d;
          chosen = id;
        }
      }
      holding = {chosen};
    }
    node = holding.front();
    out.container = node;
    out.entered = {node};
    chain.push_back(node);
    const Element* el = tree.GetElement(node);
    if (el && IsTransferLeafAt(tree, *el, skip_transparent, position)) out.transfer_leaf = node;
  }
  if (skip_transparent) {
    // The pose is in an unshared container with no shared object (something shared nested below it that the pose is not inside does not
    // make it a scope): it does not exist for the scope, so the pose is in the nearest enclosing container that does.
    while (!chain.empty() && IsScopeEmptyContainer(tree, chain.back(), not_objects)) chain.pop_back();
    out.container = chain.empty() ? start_id : chain.back();
    out.entered = chain.empty() ? std::vector<element_id_t>{} : std::vector<element_id_t>{chain.back()};
    out.transfer_leaf = kInvalidElementId;
    for (element_id_t e : chain) {
      const Element* el = tree.GetElement(e);
      if (el && IsTransferLeafAt(tree, *el, skip_transparent, position)) out.transfer_leaf = e;
    }
  }
  if (out.transfer_leaf != kInvalidElementId) {
    out.container = out.transfer_leaf;
    out.entered = {out.transfer_leaf};
  }
  return out;
}

// Walk the element tree downward from start_id, returning the deepest element whose
// footprint holds the given position (DescendScope().container). If no child holds the
// position, returns start_id itself.
inline element_id_t FindDeepestContaining(
    const Eigen::Vector3d& position,
    element_id_t start_id,
    const ElementTree& tree) {
  return DescendScope(position, start_id, tree).container;
}

enum class RelationKernelType {
  kAvatarFacing,
  kAvatarSpacing,
  kAvatarAffordancePose,
};

// Declarative scope policy a relation family applies to the shared hierarchy
// (paper Table 2). The kernel only NAMES its policy; the system-side
// ScopeResolver owns the hierarchy traversal that realizes it, so kernels stay
// pure relation-family physics (no tree walking, no containment logic).
enum class ScopePolicy {
  kVisible,    // AvatarFacing: any visible element (line-of-sight), any depth.
  kReachable,  // AvatarSpacing: navigably reachable — approach targets plus the
               // containment chain the actor is inside.
  kProximal,   // AvatarAffordancePose: affordance descendants of the deepest
               // container the actor has entered.
};

class RelationKernel {
 public:
  virtual ~RelationKernel() = default;

  virtual GroundedRelation Encode(
      const Eigen::Vector3d& anchor_world_position,
      double anchor_world_yaw,
      const Support& support) const = 0;

  virtual double Compare(const GroundedRelation& source,
                         const GroundedRelation& target) const {
    return source.DistanceTo(target);
  }

  virtual RelationKernelType GetType() const = 0;

  // Kernel-defined scope resolution. Given the actor's world position, its
  // element id, and the element tree, return the set of element ids that this
  // kernel should evaluate against.
  virtual std::vector<element_id_t> ResolveScope(
      const Eigen::Vector3d& actor_position,
      element_id_t actor_element_id,
      const ElementTree& tree,
      element_id_t cached_deepest_containing = kInvalidElementId) const = 0;

  // Compute how far this relation is from satisfied, using the actor's motion
  // state to estimate time-to-satisfaction.  When |previous| is non-null the
  // kernel uses directional progress (distance reduction between frames) to
  // estimate arrival time, falling back to scalar speed on the first frame or
  // when the actor is moving away.
  virtual SatisfactionMetric ComputeSatisfactionMetric(
      const GroundedRelation& current,
      const GroundedRelation* previous,
      const ActorMotionState& motion) const = 0;

  // Which state components this kernel optimizes (produces gradients for).
  // The solver only creates Ceres variables for the union of all dynamic states.
  virtual std::vector<StateType> GetDynamicStates() const = 0;

  // Which state components this kernel reads but doesn't optimize.
  virtual std::vector<StateType> GetStaticStates() const { return {}; }

  void SetDOF(const std::vector<StateType>& dof) { dof_ = dof; }

  bool Is3DPosition() const {
    for (auto s : dof_) {
      if (s == StateType::kPosition3D) return true;
    }
    return false;
  }

  bool Is3DRotation() const {
    for (auto s : dof_) {
      if (s == StateType::kRotation3D) return true;
    }
    return false;
  }

 protected:
  std::vector<StateType> dof_ = {StateType::kPosition2D, StateType::kRotationYaw};
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_RELATION_RELATION_KERNEL_H_
