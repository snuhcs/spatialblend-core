// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/model/element/scope_extension.h
//
// A parent's scope extends to its children.
//
// A Containment/Navigable child whose region is not fully inside its Containment/
// Navigable parent, at ANY distance from it (overlapping, touching, or across a gap
// of any width), is accepted, and the
// parent's EFFECTIVE region becomes the union of its declared region, the child's
// effective region, and a bridge across the gap between them
// (containment_layout.h decides admission; this file builds the union).
//
// One rule for every parent/child pair: nothing here depends on a label, tag or
// furniture kind. The only exception is by element KIND: an actor-bound element (an
// actor's element or anything below it, see IsActorBound) never extends a parent:
// its pose must not reshape regions.
//
// The union is bottom-up: a parent's effective region contains its children's
// effective regions, so a grandchild that sticks out of a child that sticks out of
// the parent extends both. It is recomputed from the declared regions every time,
// so it grows and shrinks with the children (a child that moves back inside or is
// removed takes its extension with it).
//
// The bridge. The app's walkable ring leaves a band around furniture (the NavMesh
// carve); a child region next to it can be separated from the ring by that band. A
// plain union would be two islands. Across a gap up to kScopeExtensionNearbyM the
// bridge is the set of points within r of BOTH regions, r = the gap between them
// (plus kBridgeSlackM): it contains the segment between the nearest points (each
// point of it is within the gap of both), so the union is one connected region, and
// it fills a band of uniform width (each point of a band of width g is within g of
// both edges). It is computed with the miter buffer (a superset of the round
// dilation, so connectivity holds) on the parent clipped to a window around the
// child, so a large room ring costs only locally. Across a larger gap that set would
// be a slab as deep as the gap and wider than the child by twice it, so the bridge
// is a CORRIDOR instead: the rectangle along the segment between the nearest points,
// as wide as the child across it (at most kScopeExtensionCorridorMaxWidthM).
// (containment_layout.h ScopeExtensionBridge, the one definition admission checks.)
//
// Pockets. A union can enclose a pocket that belongs to neither region (a U-shaped
// ring and a child across its mouth). The extension adds only what lies near the
// child: a hole the union made is filled only when it lies within min(r,
// kScopeExtensionNearbyM) of the child or its bridge (a seam of the bridge). A hole
// of the declared geometry (the parent's own, a child's) is never filled, and any
// other pocket stays a HOLE too: outside the scope (Contains) and an obstacle of
// the walkable's free space (path planning).
//
// Scope and walkable (a parent expands to contain its children; bridges are only
// for walkable connectivity). Two regions are built, each read through ONE accessor:
//  - the SCOPE region (Element::GetScopeDefinition, read through relation_kernel.h FootprintHolds):
//    the declared region united with the children's scope regions, WITHOUT the bridges and
//    corridors. The scope descent, the containment tree and every scope/leaf query. A pose in a
//    bridge or corridor gap is in the grandparent's scope (the room's), not the parent's; a pose
//    inside an extending child resolves through the parent into the child.
//  - the WALKABLE region (Element::GetSpatialDefinition, the effective definition): the scope
//    region plus the bridges and corridors, one connected region. The navigable free space and
//    path planning.
// Support, contact, frame, chart, height and interior consumers read the declared definition
// (the extension never maps anything).
//  - a Navigable parent's walkable is a NavigableSpatialDefinition whose walkable polygon is
//    the union (same frame, wall, radius, role, sub-regions; the obstacles of the
//    definition it replaces, with its free spaces prewarmed);
//  - any other parent's walkable (a Polytope box, a Surface), and every parent's scope, is an
//    ExtendedSpatialDefinition: the declared definition with its footprint test and boundary
//    widened to the union (both in the declared definition's own frame); every other query is
//    the declared definition's.
#ifndef SPATIALBLEND_ELEMENT_SCOPE_EXTENSION_H_
#define SPATIALBLEND_ELEMENT_SCOPE_EXTENSION_H_

#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "spatialblend/base/common/types.h"
#include "spatialblend/model/element/containment_layout.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/spatial_definition.h"
#include "spatialblend/geometry/obstacle_free_space.h"
#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {

// A declared definition whose footprint is widened to `extended` (in the declared
// definition's own frame: element-local for an element-local definition, the
// definition's frame otherwise). Every other query is the declared definition's.
class ExtendedSpatialDefinition : public SpatialDefinition {
 public:
  // `world_extended` is the same region in world ground coordinates, for the
  // footprint test of a definition that reads world points.
  ExtendedSpatialDefinition(std::shared_ptr<SpatialDefinition> declared,
                            std::shared_ptr<Polygon2D> extended,
                            std::shared_ptr<Polygon2D> world_extended,
                            const std::vector<SubRegion>& sub_regions);

  const SpatialDefinition& Declared() const { return *declared_; }

  bool Contains(const Eigen::Vector2d& point) const override;
  // Inside the extension within the declared definition's height band (its
  // GetBounds is 2D; a Polytope's z extent).
  bool Contains3D(const Eigen::Vector3d& point) const override;
  std::shared_ptr<Polygon2D> GetBoundary() const override { return extended_; }
  const Eigen::Matrix4d& GetLocalTransform() const override {
    return declared_->GetLocalTransform();
  }
  Bound2D GetBounds() const override;
  // Inside the extension the point itself (distance 0), else the declared one's.
  Eigen::Vector3d GetClosestPoint(const Eigen::Vector3d& from) const override;
  double GetDistance(const Eigen::Vector3d& from) const override;
  // The chart is the declared definition's (the extension never maps).
  NormalizedInterior EncodeInterior(const Eigen::Vector3d& world_point) const override {
    return declared_->EncodeInterior(world_point);
  }
  Eigen::Vector3d DecodeInterior(const NormalizedInterior& c) const override {
    return declared_->DecodeInterior(c);
  }
  Eigen::Vector3d ProjectOnto(const Eigen::Vector3d& point) const override {
    return declared_->ProjectOnto(point);
  }
  Eigen::Vector3d GetNormal(const Eigen::Vector3d& at) const override {
    return declared_->GetNormal(at);
  }
  Eigen::Matrix4d GetTransform() const override { return declared_->GetTransform(); }
  bool RequiresElementWorldTransform() const override {
    return declared_->RequiresElementWorldTransform();
  }
  bool IsSurfaceDefinition() const override { return declared_->IsSurfaceDefinition(); }

 private:
  std::shared_ptr<SpatialDefinition> declared_;
  std::shared_ptr<Polygon2D> extended_;        // the declared definition's frame
  std::shared_ptr<Polygon2D> contains_region_;  // the frame Contains() is asked in
  double z_min_ = -std::numeric_limits<double>::infinity();
  double z_max_ = std::numeric_limits<double>::infinity();
};

// The world footprints of the Obstacle-role elements, and the obstacles a navigable
// region with `bounds` and `ancestors` owns: THE definition Space's obstacle sync uses,
// so a navigable the worker extends already holds exactly the obstacles the sync would
// set (no free-space rebuild on a caller's thread when the extension is installed).
struct ObstacleFootprint {
  element_id_t id = kInvalidElementId;
  std::shared_ptr<Polygon2D> polygon;
  Eigen::Vector2d lo = Eigen::Vector2d::Zero(), hi = Eigen::Vector2d::Zero();
};
std::vector<ObstacleFootprint> CollectObstacleFootprints(const ElementTree& tree);
std::vector<FreeSpaceObstacle> ObstaclesWithin(const std::vector<ObstacleFootprint>& all,
                                               const Bound2D& bounds,
                                               const std::vector<element_id_t>& ancestors);

// A copy of what the extension depends on, taken under the owner's lock; the
// computation then runs without it.
struct ScopeExtensionInput {
  struct Node {
    element_id_t id = kInvalidElementId;
    element_id_t parent = kInvalidElementId;
    std::vector<element_id_t> children;
    bool region = false;       // Containment/Navigable role with a footprint
    bool actor_bound = false;  // never extends a parent (IsActorBound)
    std::shared_ptr<SpatialDefinition> declared;   // identity checked at install
    std::shared_ptr<SpatialDefinition> effective;  // installed when captured
    std::shared_ptr<Polygon2D> world;              // declared footprint, world
    // The 2D affine map of the declared definition's own frame to world ground
    // coordinates (world = frame_linear * p + frame_offset).
    Eigen::Matrix2d frame_linear = Eigen::Matrix2d::Identity();
    Eigen::Vector2d frame_offset = Eigen::Vector2d::Zero();
    // What the extended definition copies from the declared one, read under the lock
    // (the declared definition's sub-regions may change while the computation runs).
    bool navigable = false;  // exactly a NavigableSpatialDefinition
    std::shared_ptr<Polygon2D> wall;
    double agent_radius = 0.0;
    bool element_local = false;
    std::vector<SubRegion> sub_regions;
    // The obstacles synced into the effective Navigable definition.
    std::vector<FreeSpaceObstacle> obstacles;
    double obstacle_clearance = 0.0;
    // A Navigable-role navigable owns the obstacles within its bounds (the sync's rule):
    // its extension gets them recomputed for the extended bounds.
    bool syncs_obstacles = false;
    std::vector<element_id_t> ancestors;
  };
  element_id_t root = kInvalidElementId;
  std::vector<ObstacleFootprint> obstacle_footprints;
  std::unordered_map<element_id_t, Node> nodes;
};

ScopeExtensionInput CaptureScopeExtensionInput(const ElementTree& tree);

// What the worker keeps between builds: an element whose extension inputs (its
// declared footprint and the footprints its children add) are unchanged reuses the
// definition built for them (no union, no free-space rebuild). Owned by one caller.
struct ScopeExtensionCache {
  struct Entry {
    std::shared_ptr<SpatialDefinition> declared;
    std::vector<double> key;
    std::shared_ptr<SpatialDefinition> extended;
    std::shared_ptr<Polygon2D> world;
    std::shared_ptr<SpatialDefinition> scope;  // nullptr: the scope is the declared region
    std::shared_ptr<Polygon2D> scope_world;
  };
  std::unordered_map<element_id_t, Entry> entries;
  size_t reused = 0;  // entries reused by the last computation (tests)
};

struct ScopeExtensionResult {
  // The extended definition of every element whose scope its children extend.
  // An element absent here has no extension (its effective is its declared).
  std::unordered_map<element_id_t, std::shared_ptr<SpatialDefinition>> extended;
  // The effective (walkable) footprint in world ground coordinates, for each extended element.
  std::unordered_map<element_id_t, std::shared_ptr<Polygon2D>> world;
  // The SCOPE region of each element whose scope its children extend (Element::GetScopeDefinition):
  // the declared region united with the children's scope regions, without the bridges and
  // corridors, which only connect the walkable. An ExtendedSpatialDefinition
  // over the declared one; `scope_world` is the same region in world ground coordinates.
  std::unordered_map<element_id_t, std::shared_ptr<SpatialDefinition>> scope;
  std::unordered_map<element_id_t, std::shared_ptr<Polygon2D>> scope_world;
};

// The union and bridges; touches nothing but `input`.
ScopeExtensionResult ComputeScopeExtensions(const ScopeExtensionInput& input,
                                            ScopeExtensionCache* cache = nullptr);

// Installs `result` into `tree`: every element whose declared definition is still
// the one captured gets its extended definition, or its declared one back when it
// has no extension any more. An element whose declared definition changed since the
// capture keeps what it has (a newer computation follows). An extension whose region
// equals the installed one keeps the installed object (no consumer rebuild).
// Returns whether any effective definition changed.
bool InstallScopeExtensions(ElementTree* tree, const ScopeExtensionInput& input,
                            const ScopeExtensionResult& result);

// Capture, compute and install at once (a tree with no Space around it).
bool RefreshScopeExtensions(ElementTree* tree);

}  // namespace spatialblend

#endif  // SPATIALBLEND_ELEMENT_SCOPE_EXTENSION_H_
