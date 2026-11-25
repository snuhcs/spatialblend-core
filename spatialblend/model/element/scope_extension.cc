// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/scope_extension.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <typeinfo>
#include <utility>

#include "spatialblend/model/element/geometry_role.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"

namespace spatialblend {
namespace {

constexpr double kEpsilon = 1e-9;  // the relation layer's geometry epsilon

bool SameRegion(const Polygon2D& a, const Polygon2D& b) {
  return a.GetPoints() == b.GetPoints() && a.ExtraRings() == b.ExtraRings() &&
         a.Holes() == b.Holes();
}

bool SameParts(const std::vector<SubRegion>& a, const std::vector<SubRegion>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].name != b[i].name || a[i].chart != b[i].chart) return false;
  return true;
}

void AppendKey(const Polygon2D& p, std::vector<double>* key) {
  auto ring = [&](const std::vector<Eigen::Vector2d>& r) {
    key->push_back(static_cast<double>(r.size()));
    for (const auto& q : r) {
      key->push_back(q.x());
      key->push_back(q.y());
    }
  };
  ring(p.GetPoints());
  for (const auto& r : p.ExtraRings()) ring(r);
  for (const auto& r : p.Holes()) ring(r);
  key->push_back(-1.0);
}

// The union of `parts`, keeping a pocket it encloses OUT unless the union itself made it and it
// lies within the reach of what the children added (`near`: each contributing child and bridge,
// dilated by min(r, kScopeExtensionNearbyM)): such a hole is a seam of the union. A hole of a
// part's declared geometry (the parent's own holes, a child's) is never filled, nor any other
// pocket: each stays a hole.
Polygon2D UniteWithoutFarPockets(const std::vector<Polygon2D>& parts,
                                 const std::vector<Polygon2D>& near) {
  Polygon2D united = Polygon2D::Union(parts);
  const auto holes = united.Holes();
  if (holes.empty()) return united;
  std::vector<Polygon2D> declared_holes;
  for (const Polygon2D& part : parts)
    for (const auto& ring : part.Holes()) declared_holes.emplace_back(ring);
  std::vector<Polygon2D> fill{united};
  for (const auto& ring : holes) {
    const Polygon2D hole(ring);
    const double area = hole.GetArea();
    bool declared = false;
    for (const Polygon2D& h : declared_holes)
      declared = declared || h.IntersectionArea(hole) > kEpsilon * area;
    if (declared) continue;
    double covered = 0.0;
    for (const Polygon2D& z : near) covered = std::max(covered, z.IntersectionArea(hole));
    if (covered >= area * (1.0 - 1e-6)) fill.push_back(hole);
  }
  return fill.size() == 1 ? united : Polygon2D::Union(fill);
}

}  // namespace

ExtendedSpatialDefinition::ExtendedSpatialDefinition(
    std::shared_ptr<SpatialDefinition> declared, std::shared_ptr<Polygon2D> extended,
    std::shared_ptr<Polygon2D> world_extended, const std::vector<SubRegion>& sub_regions)
    : SpatialDefinition(declared->GetGeometryRole()),
      declared_(std::move(declared)),
      extended_(std::move(extended)) {
  // An element-local definition is asked in its own frame; any other one in world
  // ground coordinates (relation_kernel.h FootprintHolds).
  contains_region_ = declared_->RequiresElementWorldTransform() ? extended_ : std::move(world_extended);
  for (const auto& r : sub_regions) AddSubRegion(r);
  // The height band of a volume (a Polytope's vertices are immutable after construction).
  if (const auto* box = dynamic_cast<const PolytopeSpatialDefinition*>(declared_.get())) {
    z_min_ = box->GetMinCorner().z();
    z_max_ = box->GetMaxCorner().z();
  }
}

bool ExtendedSpatialDefinition::Contains(const Eigen::Vector2d& point) const {
  return declared_->Contains(point) || (contains_region_ && contains_region_->Contains(point));
}

bool ExtendedSpatialDefinition::Contains3D(const Eigen::Vector3d& point) const {
  return declared_->Contains3D(point) ||
         (contains_region_ && point.z() >= z_min_ && point.z() <= z_max_ &&
          contains_region_->Contains(point.head<2>()));
}

Eigen::Vector3d ExtendedSpatialDefinition::GetClosestPoint(const Eigen::Vector3d& from) const {
  if (contains_region_ && contains_region_->Contains(from.head<2>())) return from;
  return declared_->GetClosestPoint(from);
}

Bound2D ExtendedSpatialDefinition::GetBounds() const {
  Bound2D b = declared_->GetBounds();
  if (!contains_region_) return b;
  auto grow = [&](const std::vector<Eigen::Vector2d>& ring) {
    for (const auto& p : ring) {
      b.min = b.min.cwiseMin(p);
      b.max = b.max.cwiseMax(p);
    }
  };
  grow(contains_region_->GetPoints());
  for (const auto& r : contains_region_->ExtraRings()) grow(r);
  return b;
}

double ExtendedSpatialDefinition::GetDistance(const Eigen::Vector3d& from) const {
  if (contains_region_ && contains_region_->Contains(from.head<2>())) return 0.0;
  return declared_->GetDistance(from);
}

std::vector<ObstacleFootprint> CollectObstacleFootprints(const ElementTree& tree) {
  std::vector<ObstacleFootprint> out;
  const Element* root = tree.GetRoot();
  if (!root) return out;
  for (element_id_t id : tree.GetSubtreeIds(root->GetId())) {
    const Element* el = tree.GetElement(id);
    if (!el || !el->GetSpatialDefinition() || !el->HasGeometryRole(GeometryRole::kObstacle)) continue;
    const SpatialDefinition* def = el->GetSpatialDefinition();
    auto boundary = def->GetBoundary();
    if (!boundary || boundary->NumPoints() < 3) continue;
    const Eigen::Matrix4d m = def->RequiresElementWorldTransform() ? tree.GetWorldTransform(id)
                                                                   : def->GetTransform();
    std::vector<Eigen::Vector2d> pts;
    pts.reserve(boundary->NumPoints());
    Eigen::Vector2d lo = Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity());
    Eigen::Vector2d hi = -lo;
    for (const auto& p : boundary->GetPoints()) {
      const Eigen::Vector2d w = (m * Eigen::Vector4d(p.x(), p.y(), 0.0, 1.0)).head<2>();
      if (!w.allFinite()) continue;
      pts.push_back(w);
      lo = lo.cwiseMin(w);
      hi = hi.cwiseMax(w);
    }
    if (pts.size() < 3) continue;
    out.push_back({id, std::make_shared<Polygon2D>(pts), lo, hi});
  }
  return out;
}

std::vector<FreeSpaceObstacle> ObstaclesWithin(const std::vector<ObstacleFootprint>& all,
                                               const Bound2D& b,
                                               const std::vector<element_id_t>& ancestors) {
  std::vector<FreeSpaceObstacle> own;
  for (const auto& o : all) {
    if (std::find(ancestors.begin(), ancestors.end(), o.id) != ancestors.end()) continue;
    if (o.hi.x() < b.min.x() || o.lo.x() > b.max.x() || o.hi.y() < b.min.y() || o.lo.y() > b.max.y())
      continue;
    own.push_back({o.id, o.polygon});
  }
  return own;
}

ScopeExtensionInput CaptureScopeExtensionInput(const ElementTree& tree) {
  ScopeExtensionInput in;
  const Element* root = tree.GetRoot();
  if (!root) return in;
  in.root = root->GetId();
  in.obstacle_footprints = CollectObstacleFootprints(tree);
  for (element_id_t id : tree.GetSubtreeIds(root->GetId())) {
    const Element* e = tree.GetElement(id);
    if (!e) continue;
    ScopeExtensionInput::Node n;
    n.id = id;
    n.parent = e->GetParentId();
    n.children = tree.GetDirectChildIds(id);
    n.declared = e->GetDeclaredSpatialDefinitionShared();
    n.effective = e->GetSpatialDefinitionShared();
    n.actor_bound = IsActorBound(tree, id);
    if (n.declared) {
      n.region = IsContainmentRole(n.declared->GetGeometryRole());
      // A copy: the computation runs without the owner's lock.
      // The footprint admission checked (spikes and keyhole slits repaired), a copy of its own.
      if (auto fp = UnionFootprint(tree, id))
        if (fp->NumComponents() > 0) n.world = std::make_shared<Polygon2D>(*fp);
      const Eigen::Matrix4d frame = n.declared->RequiresElementWorldTransform()
                                        ? tree.GetWorldTransform(id)
                                        : n.declared->GetTransform();
      n.frame_linear = frame.topLeftCorner<2, 2>();
      n.frame_offset = frame.block<2, 1>(0, 3);
      n.sub_regions = n.declared->GetSubRegions();
      const SpatialDefinition& declared = *n.declared;
      if (typeid(declared) == typeid(NavigableSpatialDefinition)) {
        const auto& nav = static_cast<const NavigableSpatialDefinition&>(declared);
        n.navigable = true;
        n.wall = nav.GetWall();
        n.agent_radius = nav.GetAgentRadius();
        n.element_local = nav.IsElementLocal();
      }
    }
    if (const auto* nav = dynamic_cast<const NavigableSpatialDefinition*>(n.effective.get())) {
      n.obstacles = nav->GetObstacles();
      n.obstacle_clearance = nav->GetObstacleClearance();
      n.syncs_obstacles = e->HasGeometryRole(GeometryRole::kNavigable);
      if (n.syncs_obstacles) n.ancestors = tree.GetAncestorIds(id);
    }
    in.nodes.emplace(id, std::move(n));
  }
  return in;
}

ScopeExtensionResult ComputeScopeExtensions(const ScopeExtensionInput& input,
                                            ScopeExtensionCache* cache) {
  ScopeExtensionResult out;
  if (input.root == kInvalidElementId) return out;
  if (cache) cache->reused = 0;
  ScopeExtensionCache next_cache;
  // Post-order: children before parents.
  std::vector<element_id_t> order;
  std::vector<std::pair<element_id_t, bool>> stack{{input.root, false}};
  while (!stack.empty()) {
    auto [id, expanded] = stack.back();
    stack.pop_back();
    const auto it = input.nodes.find(id);
    if (it == input.nodes.end()) continue;
    if (expanded) {
      order.push_back(id);
      continue;
    }
    stack.push_back({id, true});
    for (element_id_t c : it->second.children) stack.push_back({c, false});
  }
  for (element_id_t id : order) {
    const ScopeExtensionInput::Node& p = input.nodes.at(id);
    if (!p.region || !p.world) continue;
    std::vector<Polygon2D> parts{*p.world};
    std::vector<Polygon2D> near;  // what each child adds, dilated by its reach (pockets)
    // The scope region: the declared region and the children's scope regions, no bridge or
    // corridor. Only a seam where they touch is a pocket of it.
    std::vector<Polygon2D> scope_parts{*p.world};
    std::vector<Polygon2D> scope_near;
    std::vector<double> key;
    AppendKey(*p.world, &key);
    for (element_id_t cid : p.children) {
      const auto cit = input.nodes.find(cid);
      if (cit == input.nodes.end()) continue;
      const ScopeExtensionInput::Node& c = cit->second;
      // Only a valid region of a non-actor kind extends a parent (scope_extension.h).
      if (!c.region || !c.world || c.actor_bound) continue;
      const auto ext = out.world.find(cid);
      const auto scope_ext = out.scope_world.find(cid);
      const double common = c.world->IntersectionArea(*p.world);
      const double outside = c.world->GetArea() - common;
      const bool sticks_out = outside > kEpsilon * c.world->GetArea();
      const Polygon2D& contribution = ext != out.world.end() ? *ext->second : *c.world;
      const Polygon2D& scope_contribution =
          scope_ext != out.scope_world.end() ? *scope_ext->second : *c.world;
      if (sticks_out) {
        const double gap = c.world->DistanceTo(*p.world);  // any gap: bridged below
        // How far from the child and its bridge a pocket of the union is a seam: the band's
        // width, never more than the carve band (a far child fills no pocket of the room).
        const double r = std::min(gap, kScopeExtensionNearbyM) + kBridgeSlackM;
        // A child that overlaps its parent is connected to it already; one across a gap (or
        // touching at a point) is bridged.
        if (!(common > kEpsilon * c.world->GetArea())) {
          Polygon2D bridge = ScopeExtensionBridge(*p.world, *c.world, gap);
          near.push_back(Polygon2D::Union({*c.world, bridge}).Inflate(r));
          parts.push_back(std::move(bridge));
        } else {
          near.push_back(c.world->Inflate(r));
        }
        parts.push_back(contribution);
        scope_parts.push_back(scope_contribution);
        scope_near.push_back(scope_contribution.Inflate(kBridgeSlackM));
      } else if (ext != out.world.end()) {
        parts.push_back(contribution);  // a grandchild's extension may reach past this parent
        near.push_back(contribution.Inflate(kBridgeSlackM));
        scope_parts.push_back(scope_contribution);
        scope_near.push_back(scope_contribution.Inflate(kBridgeSlackM));
      } else {
        continue;
      }
      key.push_back(static_cast<double>(cid));
      AppendKey(*c.world, &key);
      AppendKey(contribution, &key);
      AppendKey(scope_contribution, &key);
    }
    if (parts.size() == 1) continue;
    const double det = p.frame_linear.determinant();
    if (!(std::fabs(det) > kEpsilon)) continue;  // a frame with no ground area
    // Unchanged inputs: the definition built for them (no union, no free-space rebuild).
    if (cache) {
      const auto hit = cache->entries.find(id);
      if (hit != cache->entries.end() && hit->second.declared == p.declared && hit->second.key == key) {
        out.extended.emplace(id, hit->second.extended);
        out.world.emplace(id, hit->second.world);
        if (hit->second.scope) {
          out.scope.emplace(id, hit->second.scope);
          out.scope_world.emplace(id, hit->second.scope_world);
        }
        next_cache.entries.emplace(id, hit->second);
        ++cache->reused;
        continue;
      }
    }
    Polygon2D united = UniteWithoutFarPockets(parts, near);
    if (!(united.GetArea() > p.world->GetArea() * (1.0 + kEpsilon))) continue;
    auto world = std::make_shared<Polygon2D>(std::move(united));
    // The union in the declared definition's own frame (holes included).
    const Eigen::Matrix2d inv = p.frame_linear.inverse();
    auto local = std::make_shared<Polygon2D>(world->Transformed(inv, -inv * p.frame_offset));
    std::shared_ptr<SpatialDefinition> def;
    if (p.navigable) {
      auto ext_nav = std::make_shared<NavigableSpatialDefinition>(
          local, p.wall, p.agent_radius, p.declared->GetGeometryRole());
      ext_nav->SetElementLocal(p.element_local);
      for (const auto& r : p.sub_regions) ext_nav->AddSubRegion(r);
      // The obstacles the replaced definition holds, so the swap does not rebuild the free
      // space on a caller's thread; it is built here.
      // A Navigable-role region owns the obstacles within its (now extended) bounds, as the sync
      // computes them; any other navigable keeps the ones it holds.
      ext_nav->SetObstacles(p.syncs_obstacles
                                ? ObstaclesWithin(input.obstacle_footprints, ext_nav->GetBounds(), p.ancestors)
                                : p.obstacles,
                            p.syncs_obstacles ? 0.0 : p.obstacle_clearance);
      ext_nav->PrewarmFreeSpaces();
      def = std::move(ext_nav);
    } else {
      def = std::make_shared<ExtendedSpatialDefinition>(p.declared, local, world, p.sub_regions);
    }
    // The scope region in the same frames; the declared definition answers every other query.
    std::shared_ptr<SpatialDefinition> scope_def;
    std::shared_ptr<Polygon2D> scope_world;
    Polygon2D scope_united = UniteWithoutFarPockets(scope_parts, scope_near);
    if (scope_united.GetArea() > p.world->GetArea() * (1.0 + kEpsilon)) {
      scope_world = std::make_shared<Polygon2D>(std::move(scope_united));
      auto scope_local =
          std::make_shared<Polygon2D>(scope_world->Transformed(inv, -inv * p.frame_offset));
      scope_def = std::make_shared<ExtendedSpatialDefinition>(p.declared, scope_local, scope_world,
                                                              p.sub_regions);
    }
    next_cache.entries[id] = {p.declared, key, def, world, scope_def, scope_world};
    out.extended.emplace(id, std::move(def));
    out.world.emplace(id, std::move(world));
    if (scope_def) {
      out.scope.emplace(id, std::move(scope_def));
      out.scope_world.emplace(id, std::move(scope_world));
    }
  }
  if (cache) {
    const size_t reused = cache->reused;
    cache->entries = std::move(next_cache.entries);
    cache->reused = reused;
  }
  return out;
}

bool InstallScopeExtensions(ElementTree* tree, const ScopeExtensionInput& input,
                            const ScopeExtensionResult& result) {
  bool changed = false;
  for (const auto& [id, node] : input.nodes) {
    Element* e = tree->GetElement(id);
    if (!e || e->GetDeclaredSpatialDefinitionShared() != node.declared) continue;
    // Parts are added to a declared definition in place: one changed since the capture is
    // installed by the rebuild that change requested, never without its parts.
    if (node.declared && !SameParts(node.declared->GetSubRegions(), node.sub_regions)) continue;
    // The scope region (no bridges), installed beside the walkable one.
    const auto scope_it = result.scope.find(id);
    const std::shared_ptr<SpatialDefinition> scope =
        scope_it == result.scope.end() ? node.declared : scope_it->second;
    const auto installed_scope = e->GetScopeDefinitionShared();
    if (installed_scope != scope) {
      const auto now = installed_scope ? installed_scope->GetBoundary() : nullptr;
      const auto next = scope ? scope->GetBoundary() : nullptr;
      const bool same = installed_scope != node.declared && scope != node.declared && now && next &&
                        SameRegion(*now, *next);
      if (!same) {
        e->SetScopeDefinition(scope);
        changed = true;
      }
    }
    const auto it = result.extended.find(id);
    if (it == result.extended.end()) {
      if (e->GetSpatialDefinitionShared() != node.declared) {
        e->SetEffectiveSpatialDefinition(nullptr);
        changed = true;
      }
      continue;
    }
    // The same region already installed keeps its object (its caches, its synced obstacles).
    if (e->GetSpatialDefinitionShared() != node.declared) {
      const auto now = e->GetSpatialDefinition()->GetBoundary();
      const auto next = it->second->GetBoundary();
      const SpatialDefinition& installed = *e->GetSpatialDefinition();
      const SpatialDefinition& computed = *it->second;
      if (now && next && typeid(installed) == typeid(computed) &&
          SameRegion(*now, *next))
        continue;
    }
    e->SetEffectiveSpatialDefinition(it->second);
    changed = true;
  }
  return changed;
}

bool RefreshScopeExtensions(ElementTree* tree) {
  const ScopeExtensionInput input = CaptureScopeExtensionInput(*tree);
  return InstallScopeExtensions(tree, input, ComputeScopeExtensions(input));
}

}  // namespace spatialblend
