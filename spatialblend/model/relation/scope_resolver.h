// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_RELATION_SCOPE_RESOLVER_H_
#define SPATIALBLEND_RELATION_SCOPE_RESOLVER_H_

#include <algorithm>
#include <vector>

#include <Eigen/Dense>

#include "spatialblend/base/common/types.h"
#include "spatialblend/model/element/element.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/geometry_role.h"
#include "spatialblend/model/relation/relation_kernel.h"

namespace spatialblend {

// The mode of a Spacing scope (docs: spatialblend/space/testing/SCOPE_CASES.md).
enum class SpacingScopeMode {
  kOutside = 0,   // the point is in no referent's footprint
  kInterior = 1,  // (a) inside a referent with interior entities: its objects + its border
  kReplay = 2,    // (b) inside a transfer leaf / an empty referent: replay at the mapped position
};

inline const char* SpacingScopeModeName(SpacingScopeMode mode) {
  switch (mode) {
    case SpacingScopeMode::kOutside: return "OUTSIDE";
    case SpacingScopeMode::kInterior: return "INTERIOR";
    case SpacingScopeMode::kReplay: return "REPLAY";
  }
  return "?";
}

// The tags D1-D9 and MA-1 in this file are the scope decisions defined in
// spatialblend/space/testing/SCOPE_CASES.md ("Decisions").

// The structural Spacing scope of one anchor pose, before the family gates (D7/D8).
struct SpacingScope {
  element_id_t container = kInvalidElementId;  // the deepest element holding the point
  std::vector<element_id_t> entered;           // the referents the point is inside
  element_id_t transfer_leaf = kInvalidElementId;
  SpacingScopeMode mode = SpacingScopeMode::kOutside;
  std::vector<element_id_t> referents;         // the scope, in the order the kernel reports it
  bool border = false;                         // the container is among them as its border relation
};

// A Facing look-at is IN-BODY only when it COINCIDES with the
// body (v1: delta < kEpsilon), where the direction to it is undefined. Nothing else is in-body:
// the body radius, a footprint, a distance or a fade is not a scope. The relation layer's
// geometry epsilon is the only tolerance.
inline constexpr double kLookAtCoincidenceM = 1e-9;
inline bool LookAtInBody(double distance_to_look_at_m) {
  return distance_to_look_at_m < kLookAtCoincidenceM;
}

// Where the root yaw of a proposal comes from (D2): a valid
// Facing owns it; inside a transfer leaf with none it is the leaf-relative yaw through the
// leaf's correspondence; with neither it is unspecified (the driver faces its motion).
enum class RootYawSource { kFacing, kLeafRelative, kUnspecified };

inline RootYawSource ResolveRootYawSource(bool a_facing_is_valid, bool source_inside_transfer_leaf) {
  if (a_facing_is_valid) return RootYawSource::kFacing;
  return source_inside_transfer_leaf ? RootYawSource::kLeafRelative : RootYawSource::kUnspecified;
}

// Facing scope inside a containment-transfer element (a scope rule of the Facing relation definition,
// generic over roles and flags, no object kinds): while the anchor pose is inside a transfer element L (`transfer_leaf`,
// the DescendScope leaf of the anchor pose), L itself is not a Facing referent, nor is any descendant of L, except one
// whose geometry role is Surface (the affordance surfaces L carries: a writable face, a playing surface). Seen from
// inside its footprint L's gaze ray always hits its own boundary, so its look-at offset is 0 and its count 1 whatever
// the heading: no relation. What L carries (its object) is covered by L's chart, as for Spacing (D5); its surfaces stay
// look-at targets. Elements outside L are unaffected.
inline bool ExcludedFromFacingInsideTransfer(const ElementTree& tree, element_id_t transfer_leaf,
                                             element_id_t element) {
  if (transfer_leaf == kInvalidElementId) return false;
  if (element == transfer_leaf) return true;
  bool below = false;
  for (const Element* e = tree.GetElement(element); e && e->GetParentId() != kInvalidElementId;) {
    if (e->GetParentId() == transfer_leaf) {
      below = true;
      break;
    }
    e = tree.GetElement(e->GetParentId());
  }
  if (!below) return false;
  const Element* el = tree.GetElement(element);
  return !(el && el->HasGeometryRole(GeometryRole::kSurface));
}

// System-side resolution of a relation family's ScopePolicy against the shared
// interaction hierarchy. This is the Realizer/Extractor's concern, NOT the
// kernel's: kernels only declare a ScopePolicy (pure relation-family physics);
// all tree traversal, containment, and deepest-containing logic lives here, in
// one shared implementation. Previously each kernel re-implemented this
// (duplicating FindDeepestContaining and the containment-chain walk).
struct ScopeResolver {
  // The root of every scope descent: the tree's NAVIGABLE root, the
  // topmost ancestor-or-self of `element` that is a Navigable element with a spatial definition, which
  // is the room. The subject's own element (`element`) is a person leaf: wherever it hangs in the tree
  // it neither starts nor bounds a descent, so the scope of a pose does not depend on it. A descent
  // from the leaf's nearest containment ancestor started INSIDE that ancestor and never tested it (a
  // pose inside a transfer leaf the person hung from flagged no transfer). A tree with no Navigable
  // element has its topmost containment element as the root; a tree with neither (no geometry at all)
  // has `element` itself.
  static element_id_t NavigableRoot(const ElementTree& tree, element_id_t element) {
    element_id_t navigable = kInvalidElementId, containment = kInvalidElementId;
    for (element_id_t e = element; e != kInvalidElementId;) {
      const Element* el = tree.GetElement(e);
      if (!el) break;
      if (el->GetSpatialDefinition()) {
        if (el->HasGeometryRole(GeometryRole::kNavigable)) navigable = e;  // keep going up: topmost
        if (IsContainmentRole(el->GetGeometryRole())) containment = e;
      }
      e = el->GetParentId();
    }
    if (navigable != kInvalidElementId) return navigable;
    return containment != kInvalidElementId ? containment : element;
  }

  // D7 (MA-1: unshared elements are never referents): the valid referent is decided on
  // the source side, and an element with no shared id has no counterpart on any receiver by definition,
  // so it is a referent of NO family (Spacing, Facing, AffordancePose). It still constrains F (it is an
  // obstacle) and the container border is unaffected. v1 does the same: an entity is a relation target
  // only when it was explicitly shared (spatialblend/space/space.cc ShareEntity :281, ShareInteraction
  // :364 -> SpaceState::AddInteractionTarget :412-419); nothing else is ever a target.
  static bool HasCounterpart(const ElementTree& tree, element_id_t element) {
    const Element* e = tree.GetElement(element);
    return e && e->GetSharedId() != kInvalidSharedElementId;
  }

  // MA-1 applied to a scope's OWN container as the entered referent (a REPLAY, a flagged leaf, the fallback of a container with no
  // objects): it is a referent only when it has a counterpart. An unshared one is an obstacle and nothing else (SP-D7-04). An unshared
  // element that carries nothing shared is not entered at all (ResolveSpacing's descent, IsScopeTransparent), so this is reached with an
  // unshared container only when it holds something shared. NOT applied to the container's border referent.
  static std::vector<element_id_t> ContainerReferent(const ElementTree& tree, element_id_t container) {
    return HasCounterpart(tree, container) ? std::vector<element_id_t>{container} : std::vector<element_id_t>{};
  }

  // Return the element ids the relation should evaluate against, given its
  // policy, the actor's world position + element id, and the hierarchy.
  // cached_deepest_containing avoids recomputing FindDeepestContaining when the
  // caller already has it for the frame (kInvalidElementId = compute here).
  static std::vector<element_id_t> Resolve(
      ScopePolicy policy, const Eigen::Vector3d& actor_position,
      element_id_t actor_element_id, const ElementTree& tree,
      element_id_t cached_deepest_containing = kInvalidElementId) {
    switch (policy) {
      case ScopePolicy::kVisible:
        return ResolveVisible(actor_element_id, tree);
      case ScopePolicy::kReachable:
        return ResolveReachable(actor_position, actor_element_id, tree,
                                cached_deepest_containing);
      case ScopePolicy::kProximal:
        return ResolveProximal(actor_position, actor_element_id, tree,
                               cached_deepest_containing);
    }
    return {};
  }

 public:  // the border rule and the reachable set are the scope layer's contract (tested in isolation)
  // kVisible (Facing): every element at ANY depth except the actor's own
  // element and the elements it sits in (its ancestors). Line of sight is a
  // per-candidate gate applied after this (relation/scope_gate.h).
  static std::vector<element_id_t> ResolveVisible(element_id_t actor_element_id,
                                                  const ElementTree& tree) {
    if (!tree.GetElement(actor_element_id)) return {};
    std::vector<element_id_t> excluded = tree.GetAncestorIds(actor_element_id);
    excluded.push_back(actor_element_id);
    auto ids = tree.GetAllIds();
    ids.erase(std::remove_if(ids.begin(), ids.end(),
                             [&](element_id_t id) {
                               return std::find(excluded.begin(),
                                                excluded.end(),
                                                id) != excluded.end() ||
                                      !HasCounterpart(tree, id);
                             }),
              ids.end());
    return ids;
  }

  // kReachable: the deepest container's non-interaction children — the OBJECT
  // approach targets the actor moves AMONG. Position is determined by the
  // constellation of per-object (geodesic) spacings to these, which transfers
  // object-relatively across differently-arranged spaces (v1 model).
  //
  // THE BORDER REFERENT RULE (the ONE place it lives,
  // Space reads it from here). A container the actor is inside has a border
  // referent when its interior is solved by relations -- it has other referents in
  // scope (a non-interaction child) -- so that crossing its outer boundary stays
  // continuous: the relation to the border is engaged at the crossing (distance 0)
  // and carries the weight across. It has none when
  //   - it is the room, the navigable scope root (a Navigable element with no
  //     Navigable ancestor): the walls are the scope, not a position target;
  //   - it has no spatial definition;
  //   - it is a containment-transfer LEAF: a flagged region. An authored region carries its object (Chess, Whiteboard:
  //     region -> object -> affordance) and is still ONE replay correspondence, so the
  //     object it carries is covered by the leaf, not a referent below it. The leaf is
  //     the actor's whole scope (ResolveReachable); the empty-scope mapping is the
  //     transfer.
  // The rule reads the element's declared role, its transfer flag and its
  // children; it does not look at the parent's geometry.
  static bool HasBorderReferent(const ElementTree& tree, element_id_t container) {
    const Element* c = tree.GetElement(container);
    if (!c || c->GetSpatialDefinition() == nullptr) return false;
    // A flagged region is a transfer LEAF (D5): it carries the actor, objects and all.
    if (IsTransferLeafFlag(tree, *c, /*skip_transparent=*/true)) return false;  // (a flag on an unshared element acts on nothing, SP-D7-05)
    if (c->HasGeometryRole(GeometryRole::kNavigable)) {
      bool navigable_ancestor = false;
      for (element_id_t a = c->GetParentId(); a != kInvalidElementId;) {
        const Element* up = tree.GetElement(a);
        if (!up) break;
        if (up->HasGeometryRole(GeometryRole::kNavigable)) {
          navigable_ancestor = true;
          break;
        }
        a = up->GetParentId();
      }
      if (!navigable_ancestor) return false;  // the room
    }
    for (element_id_t child_id : tree.GetDirectChildIds(container)) {
      const Element* child = tree.GetElement(child_id);
      if (!child || !IsInteractionRole(child->GetGeometryRole())) return true;
    }
    return false;
  }

  // The Spacing scope of a pose, stateless, decided on the SOURCE side (D7). This is the whole
  // definition: the containment tree membership of the point (DescendScope), the entered
  // referent's mode, and the referents each mode gives. Nothing about it is a solver case.
  //
  //  OUTSIDE   the point is in no referent's footprint below `start`: the container's
  //            non-interaction children, the container's border referent where D4 gives
  //            one, and every person of the tree (D9).
  //  INTERIOR  (a) the point is INSIDE a referent that has interior entities and is not a
  //            transfer leaf: the objects inside it, its
  //            border relation, the people inside it. The entered referent is the container.
  //  REPLAY    (b) the point is inside a transfer leaf, or inside a referent with no
  //            interior entities and no leaf flag: the entered referent alone, replayed at the
  //            mapped position (the chart / transfer mapping). The sibling referents, and the
  //            people, are out of scope. Where sibling footprints overlap the entered referent is
  //            the one v1 answers (DescendScope, V1NormalizedDistance).
  //
  // `people` are the other actors' elements; `subject` the subject's own element (never a
  // person, and its ancestors are containers it is in, not people next to it).
  static SpacingScope ResolveSpacing(const Eigen::Vector3d& position, element_id_t start,
                                     const ElementTree& tree,
                                     const std::vector<element_id_t>& people = {},
                                     element_id_t subject = kInvalidElementId) {
    SpacingScope scope;
    const auto tree_root = [&](element_id_t e) {
      const auto up = tree.GetAncestorIds(e);
      return up.empty() ? e : up.back();
    };
    const element_id_t my_root = tree_root(start);
    const std::vector<element_id_t> mine =
        subject == kInvalidElementId ? std::vector<element_id_t>{} : tree.GetAncestorIds(subject);
    std::vector<element_id_t> persons;  // those that count as Spacing referents at all
    for (element_id_t person : people) {
      if (person == subject || !tree.GetElement(person) || tree_root(person) != my_root) continue;
      if (!HasCounterpart(tree, person)) continue;  // D7: no counterpart, no referent
      if (std::find(mine.begin(), mine.end(), person) != mine.end()) continue;
      if (std::find(persons.begin(), persons.end(), person) != persons.end()) continue;
      persons.push_back(person);
    }
    // The persons and the subject are never the objects of a container (a person under an unshared container does not make it a scope).
    std::vector<element_id_t> not_objects = persons;
    if (subject != kInvalidElementId) not_objects.push_back(subject);
    const ScopeDescent descent =
        DescendScope(position, start, tree, /*choose_overlap=*/true, /*skip_transparent=*/true, &not_objects);
    scope.container = descent.container;
    scope.entered = descent.entered;
    scope.transfer_leaf = descent.transfer_leaf;
    if (scope.container == kInvalidElementId || !tree.GetElement(scope.container))
      return scope;
    const auto is_person = [&](element_id_t e) {
      return std::find(persons.begin(), persons.end(), e) != persons.end();
    };
    // The objects of a container: its direct non-interaction children that are not people. (The
    // subject's own element is listed too, and the Space excludes it with its reason, "actor element".)
    const auto objects_of = [&](element_id_t container) {
      std::vector<element_id_t> out;
      for (element_id_t child_id : tree.GetDirectChildIds(container)) {
        const Element* child = tree.GetElement(child_id);
        // (the subject's own element is kept: the Space excludes it with its reason)
        if ((!child || !IsInteractionRole(child->GetGeometryRole())) && !is_person(child_id) &&
            (child_id == subject || HasCounterpart(tree, child_id)))
          out.push_back(child_id);
      }
      return out;
    };
    // Whether a container has interior entities: objects other than the subject's own element, which is
    // a person leaf wherever it hangs and never an object (SP-TREE-04).
    const auto has_interior_entities = [&](element_id_t container) {
      for (element_id_t id : objects_of(container))
        if (id != subject) return true;
      return false;
    };

    // The entered referent is always ONE (DescendScope picks one where footprints overlap), and it is
    // the container.
    if (scope.entered.empty()) {
      scope.mode = SpacingScopeMode::kOutside;
    } else if (scope.transfer_leaf != kInvalidElementId) {
      scope.mode = SpacingScopeMode::kReplay;
    } else {
      scope.mode = has_interior_entities(scope.container) ? SpacingScopeMode::kInterior
                                                          : SpacingScopeMode::kReplay;
    }

    if (scope.mode == SpacingScopeMode::kReplay) {
      scope.referents = ContainerReferent(tree, scope.container);  // MA-1: an unshared leaf is never reached here (transparent to the descent)
      return scope;
    }
    // OUTSIDE and INTERIOR: the container's objects, its border referent, then the people.
    const Element* container = tree.GetElement(scope.container);
    if (IsTransferLeafFlag(tree, *container, /*skip_transparent=*/true)) {
      scope.referents = ContainerReferent(tree, scope.container);  // a flagged scope root is a leaf (D5)
      return scope;
    }
    scope.referents = objects_of(scope.container);
    if (HasBorderReferent(tree, scope.container) && !scope.referents.empty()) {
      scope.referents.push_back(scope.container);  // after its objects
      scope.border = true;
    }
    // Degenerate scope with no object children: fall back to the scope itself so there is
    // still a referent to ground against.
    if (scope.referents.empty()) scope.referents = ContainerReferent(tree, scope.container);
    for (element_id_t person : persons) {
      if (scope.mode == SpacingScopeMode::kInterior) {
        const Eigen::Matrix4d at = tree.GetWorldTransform(person);
        if (!FootprintHolds(*container, tree, Eigen::Vector3d(at(0, 3), at(1, 3), 0.0))) continue;
      }
      if (std::find(scope.referents.begin(), scope.referents.end(), person) == scope.referents.end())
        scope.referents.push_back(person);
    }
    return scope;
  }

  // The referents of a given container (no people, no overlap): the old container-keyed form,
  // which the kernel's ResolveScope returns. With no container given, ResolveSpacing's.
  static std::vector<element_id_t> ResolveReachable(
      const Eigen::Vector3d& actor_position, element_id_t actor_element_id,
      const ElementTree& tree, element_id_t cached_deepest_containing) {
    if (cached_deepest_containing == kInvalidElementId)
      return ResolveSpacing(actor_position, actor_element_id, tree).referents;
    element_id_t scope = cached_deepest_containing;
    if (!tree.GetElement(scope)) return {};
    // An unshared container with no shared object does not exist for the scope (IsScopeEmptyContainer): resolve at the nearest
    // ancestor that does, exactly as ResolveSpacing's descent would.
    while (IsScopeEmptyContainer(tree, scope) && tree.GetElement(scope)->GetParentId() != kInvalidElementId &&
           tree.GetElement(tree.GetElement(scope)->GetParentId()))
      scope = tree.GetElement(scope)->GetParentId();
    // A transfer leaf is the actor's whole scope and the referent itself (D5): the
    // object it carries is covered by its correspondence, not a referent below it.
    if (IsTransferLeafFlag(tree, *tree.GetElement(scope), /*skip_transparent=*/true)) return ContainerReferent(tree, scope);
    std::vector<element_id_t> result;
    for (element_id_t child_id : tree.GetDirectChildIds(scope)) {
      const Element* child = tree.GetElement(child_id);
      if ((!child || !IsInteractionRole(child->GetGeometryRole())) && HasCounterpart(tree, child_id))
        result.push_back(child_id);
    }
    // The container itself stays a referent while the actor is inside it, by the
    // border-referent rule above.
    if (HasBorderReferent(tree, scope)) result.push_back(scope);  // after its objects
    if (result.empty()) return ContainerReferent(tree, scope);
    return result;
  }

  // kProximal: descendants of the deepest container the actor has entered (the
  // affordance nodes); the AffordancePose tag binding selects the specific
  // affordance downstream and the family gate (relation/scope_gate.h, D3) keeps
  // those whose parent object's region holds the anchor.
  static std::vector<element_id_t> ResolveProximal(
      const Eigen::Vector3d& actor_position, element_id_t actor_element_id,
      const ElementTree& tree, element_id_t cached_deepest_containing) {
    if (actor_element_id == kInvalidElementId &&
        cached_deepest_containing == kInvalidElementId) {
      return {};
    }
    element_id_t scope =
        (cached_deepest_containing != kInvalidElementId)
            ? cached_deepest_containing
            : FindDeepestContaining(actor_position, actor_element_id, tree);
    if (scope == kInvalidElementId) return {};

    auto ids = tree.GetSubtreeIds(scope);
    ids.erase(std::remove(ids.begin(), ids.end(), scope), ids.end());
    ids.erase(std::remove_if(ids.begin(), ids.end(),
                             [&](element_id_t id) { return !HasCounterpart(tree, id); }),
              ids.end());
    // An element carrying an affordance pose (element/affordance_pose.h) is part of its OBJECT's scope definition
    // (the poses are solved one level down, inside the object's SD), never a referent of its own and
    // never a child that makes the object a container (SP-AFF-03).
    const auto is_pose = [&](element_id_t id) {
      const Element* e = tree.GetElement(id);
      return e && e->GetAffordancePose() != nullptr;
    };
    ids.erase(std::remove_if(ids.begin(), ids.end(), is_pose), ids.end());
    const auto& children = tree.GetDirectChildIds(scope);
    const bool leaf = std::all_of(children.begin(), children.end(), is_pose);
    // Leaf scope: include self for grounding, but only a SHARED self (MA-1 / SP-D7-03): an unshared leaf has no
    // counterpart on any receiver, so it is a referent of no family, whatever its tag.
    if (leaf && HasCounterpart(tree, scope)) {
      ids.push_back(scope);
    }
    return ids;
  }
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_RELATION_SCOPE_RESOLVER_H_
