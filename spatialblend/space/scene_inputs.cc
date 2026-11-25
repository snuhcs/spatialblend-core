// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <exception>
#include <limits>
#include <typeinfo>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/model/element/scope_extension.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace {

bool ValidTransform(const Eigen::Matrix4d& value) {
  return value.allFinite() &&
         value.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-12) &&
         std::abs(value.topLeftCorner<3, 3>().determinant()) > 1e-12;
}

template <typename Points>
bool FinitePoints(const Points& points, size_t minimum) {
  if (points.size() < minimum) return false;
  for (const auto& point : points) {
    if (!point.allFinite()) return false;
  }
  return true;
}

std::shared_ptr<Polygon2D> CopyPolygon(
    const std::shared_ptr<Polygon2D>& input) {
  if (!input || !FinitePoints(input->GetPoints(), 3)) return nullptr;
  return std::make_shared<Polygon2D>(input->GetPoints());
}

bool CopyRegion(const SubRegion& input, SubRegion* out) {
  if (!ValidTransform(input.local_transform)) return false;
  auto boundary = CopyPolygon(input.boundary);
  if (!boundary) return false;
  // A frame chart needs the axis-aligned rectangle it reads (a registration that cannot be read is refused, not stored).
  if (IsFrameChart(input.chart) && !PlanarFrameChart::ExtentOf(*boundary).ok()) return false;
  *out = input;
  out->boundary = std::move(boundary);
  return true;
}

// Copy only exact built-in types: a subclass may change query semantics or
// retain mutable external state that reconstructing its base would silently
// lose.
std::shared_ptr<SpatialDefinition> CopyDefinition(
    const SpatialDefinition* def) {
  if (!def) return nullptr;
  std::shared_ptr<SpatialDefinition> copy;
  if (typeid(*def) == typeid(NavigableSpatialDefinition)) {
    const auto& nav = static_cast<const NavigableSpatialDefinition&>(*def);
    auto walkable = CopyPolygon(nav.GetWalkablePolygon());
    auto wall = CopyPolygon(nav.GetWall());
    if (!walkable || (nav.GetWall() && !wall) ||
        !std::isfinite(nav.GetAgentRadius()) || nav.GetAgentRadius() < 0) {
      return nullptr;
    }
    copy = std::make_shared<NavigableSpatialDefinition>(
        std::move(walkable), std::move(wall), nav.GetAgentRadius(),
        nav.GetGeometryRole());
    static_cast<NavigableSpatialDefinition&>(*copy).SetElementLocal(
        nav.IsElementLocal());
  } else if (typeid(*def) == typeid(SurfaceSpatialDefinition)) {
    const auto& surface = static_cast<const SurfaceSpatialDefinition&>(*def);
    if (!surface.GetSurface() ||
        !FinitePoints(surface.GetSurface()->GetVertices(), 3))
      return nullptr;
    copy = std::make_shared<SurfaceSpatialDefinition>(
        std::make_shared<Polygon3D>(surface.GetSurface()->GetVertices()));
    copy->SetGeometryRole(def->GetGeometryRole());
  } else if (typeid(*def) == typeid(PolytopeSpatialDefinition)) {
    const auto& polytope = static_cast<const PolytopeSpatialDefinition&>(*def);
    if (!FinitePoints(polytope.GetVertices(), 4) ||
        polytope.GetFaceIndices().empty())
      return nullptr;
    for (const auto& face : polytope.GetFaceIndices()) {
      if (face.size() < 3) return nullptr;
      for (int index : face) {
        if (index < 0 ||
            static_cast<size_t>(index) >= polytope.GetVertices().size())
          return nullptr;
      }
    }
    copy = std::make_shared<PolytopeSpatialDefinition>(
        polytope.GetVertices(), polytope.GetFaceIndices(),
        def->GetGeometryRole());
  } else {
    return nullptr;
  }
  for (const auto& region : def->GetSubRegions()) {
    SubRegion owned;
    if (!CopyRegion(region, &owned)) return nullptr;
    copy->AddSubRegion(std::move(owned));
  }
  return copy;
}

bool SamePolygon(const std::shared_ptr<Polygon2D>& a,
                 const std::shared_ptr<Polygon2D>& b) {
  return (!a && !b) || (a && b && a->GetPoints() == b->GetPoints());
}

bool SameDefinition(const SpatialDefinition* a, const SpatialDefinition* b) {
  if (!a || !b) return a == b;
  if (typeid(*a) != typeid(*b) ||
      a->GetGeometryRole() != b->GetGeometryRole() ||
      a->GetSubRegions().size() != b->GetSubRegions().size())
    return false;
  for (size_t i = 0; i < a->GetSubRegions().size(); ++i) {
    const auto& x = a->GetSubRegions()[i];
    const auto& y = b->GetSubRegions()[i];
    if (x.name != y.name || x.tag != y.tag ||
        x.local_transform != y.local_transform || x.chart != y.chart ||
        !SamePolygon(x.boundary, y.boundary))
      return false;
  }
  if (typeid(*a) == typeid(NavigableSpatialDefinition)) {
    const auto& x = static_cast<const NavigableSpatialDefinition&>(*a);
    const auto& y = static_cast<const NavigableSpatialDefinition&>(*b);
    return x.GetAgentRadius() == y.GetAgentRadius() &&
           x.IsElementLocal() == y.IsElementLocal() &&
           SamePolygon(x.GetWalkablePolygon(), y.GetWalkablePolygon()) &&
           SamePolygon(x.GetWall(), y.GetWall());
  }
  if (typeid(*a) == typeid(SurfaceSpatialDefinition)) {
    const auto& x = static_cast<const SurfaceSpatialDefinition&>(*a);
    const auto& y = static_cast<const SurfaceSpatialDefinition&>(*b);
    return x.GetSurface() && y.GetSurface() &&
           x.GetSurface()->GetVertices() == y.GetSurface()->GetVertices();
  }
  if (typeid(*a) == typeid(PolytopeSpatialDefinition)) {
    const auto& x = static_cast<const PolytopeSpatialDefinition&>(*a);
    const auto& y = static_cast<const PolytopeSpatialDefinition&>(*b);
    return x.GetVertices() == y.GetVertices() &&
           x.GetFaceIndices() == y.GetFaceIndices();
  }
  return false;
}

}  // namespace

ElementTree* Space::GetElementTree() {
  scene_inputs_escaped_.store(true);
  return &element_tree_;
}

const ElementTree* Space::GetElementTree() const {
  scene_inputs_escaped_.store(true);
  return &element_tree_;
}

Space::SceneInputStamp Space::CaptureSceneInputStampLocked() const {
  bool known = !scene_inputs_escaped_.load();
  if (known && !observed_facing_points_.empty()) {
    const int64_t now = receiver_input_clock_();
    for (const auto& [id, point] : observed_facing_points_)
      if (point.active && !IsObservedFacingPointLiveLocked(id, now)) known = false;
    for (const auto& [actor, dependencies] : observed_facing_dependencies_)
      if (!AreObservedFacingDependenciesLiveLocked(actor, now)) known = false;
  }
  return {tree_revision_, support_revision_, known};
}

void Space::AdvanceSceneInputsLocked(bool tree, bool support, bool geometry) {
  auto advance = [&](uint64_t* revision) {
    if (*revision == std::numeric_limits<uint64_t>::max()) {
      scene_inputs_escaped_.store(true);  // no wraparound/ABA certificate
    } else {
      ++*revision;
    }
  };
  if (tree) advance(&tree_revision_);
  if (support) advance(&support_revision_);
  if (geometry) ++geometry_revision_;
  // A scene change while a region extends its parent: the extension follows it, in the
  // background (an actor's pose passes geometry=false and never reshapes a region).
  if (geometry && scope_extension_watch_ && !installing_scope_extension_)
    RequestScopeExtensionLocked();
}

void Space::RequestScopeExtensionLocked() {
  if (!scope_extension_worker_)
    scope_extension_worker_ = std::make_unique<spatialblend::CoalescingWorker>(
        [this] { RunScopeExtensionBuild(); });
  scope_extension_worker_->Request();
}

void Space::RunScopeExtensionBuild() {
  ScopeExtensionInput input;
  std::function<void()> hook;
  {
    std::lock_guard<std::mutex> lock(space_mutex_);
    input = CaptureScopeExtensionInput(element_tree_);
    hook = scope_extension_build_hook_;
  }
  // The union, bridges and free-space prewarm run without the space lock. A build that fails
  // keeps the previous scope (nothing is installed) and is logged. (The test hook runs inside it.)
  ScopeExtensionResult result;
  try {
    if (hook) hook();
    result = ComputeScopeExtensions(input, &scope_extension_cache_);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[spatialblend] scope extension build failed, keeping the previous scope: %s\n",
                 e.what());
    scope_extension_cache_.entries.clear();
    return;
  } catch (...) {
    std::fprintf(stderr, "[spatialblend] scope extension build failed, keeping the previous scope\n");
    scope_extension_cache_.entries.clear();
    return;
  }
  std::lock_guard<std::mutex> lock(space_mutex_);
  const bool changed = InstallScopeExtensions(&element_tree_, input, result);
  // A newer admission re-arms the watch and has requested its own rebuild.
  scope_extension_watch_ = !result.extended.empty();
  if (changed) {
    installing_scope_extension_ = true;
    AdvanceSceneInputsLocked(true, true, true);
    installing_scope_extension_ = false;
    scope_extension_revision_.fetch_add(1);
  }
}

uint64_t Space::GetScopeExtensionRevision() const {
  return scope_extension_revision_.load();
}

bool Space::WaitForScopeExtension(std::chrono::milliseconds timeout) {
  spatialblend::CoalescingWorker* worker = nullptr;
  {
    std::lock_guard<std::mutex> lock(space_mutex_);
    worker = scope_extension_worker_.get();
  }
  return !worker || worker->WaitIdle(timeout);
}

uint64_t Space::GetObstacleSyncRebuildCountForTesting() const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return obstacle_sync_rebuilds_;
}

uint64_t Space::GetScopeExtensionBuildCount() const {
  spatialblend::CoalescingWorker* worker = nullptr;
  {
    std::lock_guard<std::mutex> lock(space_mutex_);
    worker = scope_extension_worker_.get();
  }
  return worker ? worker->Runs() : 0;
}

void Space::SetScopeExtensionBuildHookForTesting(std::function<void()> hook) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  scope_extension_build_hook_ = std::move(hook);
}

Space::PlanarSupportReadResult Space::TryCopyPlanarSupport(
    element_id_t id, uint64_t expected_tree_revision,
    uint64_t expected_support_revision, PlanarSupportState* out) const {
  if (!out || expected_tree_revision == 0 || expected_support_revision == 0)
    return PlanarSupportReadResult::kUnavailable;
  *out = {};
  std::unique_lock<std::mutex> lock(space_mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return PlanarSupportReadResult::kBusy;
  const auto scene = CaptureSceneInputStampLocked();
  if (!scene.known) return PlanarSupportReadResult::kUnavailable;
  if (scene.tree != expected_tree_revision ||
      scene.support != expected_support_revision)
    return PlanarSupportReadResult::kStale;
  const auto* element = element_tree_.GetElement(id);
  const auto* def = element ? element->GetDeclaredSpatialDefinition() : nullptr;
  if (!def) return PlanarSupportReadResult::kUnavailable;
  PlanarSupportState value;
  if (typeid(*def) == typeid(SurfaceSpatialDefinition)) {
    value.definition_kind = 1;
  } else if (typeid(*def) == typeid(NavigableSpatialDefinition)) {
    value.definition_kind = 2;
  } else {
    return PlanarSupportReadResult::kUnavailable;
  }
  // These exact built-ins store world-absolute geometry. This is the same
  // boundary and frame GroundAffordance consumes, including its ordered axes.
  if (def->RequiresElementWorldTransform())
    return PlanarSupportReadResult::kUnavailable;
  const auto boundary = def->GetBoundary();
  value.plane_to_world = def->GetTransform();
  if (!boundary || boundary->GetPoints().size() > 65536 ||
      !FinitePoints(boundary->GetPoints(), 3) ||
      !ValidTransform(value.plane_to_world))
    return PlanarSupportReadResult::kUnavailable;
  value.boundary = boundary->GetPoints();
  value.tree_revision = scene.tree;
  value.support_revision = scene.support;
  value.role = def->GetGeometryRole();
  *out = std::move(value);
  return PlanarSupportReadResult::kOk;
}

bool Space::AddElement(element_id_t id, element_id_t parent) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  if (observed_facing_points_.count(parent) || id == kInvalidElementId ||
      !element_tree_.AddElement(id, parent).ok())
    return false;
  AdvanceSceneInputsLocked(true, true);
  if (parent == kInvalidElementId && !RoomLocked()) room_element_id_ = id;
  return true;
}

void Space::RecordSceneErrorLocked(SceneErrorCode code, std::string message) {
  last_scene_error_ = {code, std::move(message)};
}

Space::SceneError Space::LastSceneError() const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return last_scene_error_;
}

bool Space::ReparentElement(element_id_t id, element_id_t parent,
                            const Eigen::Matrix4d* local_transform) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  RecordSceneErrorLocked(SceneErrorCode::kOk, "");
  auto* element = element_tree_.GetElement(id);
  // The new parent must be an element: kInvalidElementId (-1) would make the moved element the tree's root.
  if (!element || parent == kInvalidElementId || !element_tree_.GetElement(parent) ||
      observed_facing_points_.count(id) || observed_facing_points_.count(parent)) return false;
  if (local_transform && !ValidTransform(*local_transform)) return false;
  // A received replica belongs to its remote owner; a root has no parent to leave.
  for (const auto& entry : remote_owned_replicas_)
    if (entry.second == id) return false;
  if (pending_share_source_.count(id)) return false;  // a received element not yet resolved is a replica too
  if (element->GetParentId() == kInvalidElementId) return false;
  const element_id_t old_parent = element->GetParentId();
  const Eigen::Matrix4d previous = element->GetLocalTransform();
  if (old_parent == parent && (!local_transform || previous == *local_transform)) return true;
  if (old_parent != parent && !element_tree_.ReparentElement(id, parent).ok()) return false;
  if (local_transform) element->SetLocalTransform(*local_transform);
  bool extends = false;
  if (auto v = FindContainmentViolation(element_tree_, id, &extends)) {
    // Not registered: put the element back where it was.
    element->SetLocalTransform(previous);
    if (old_parent != parent) (void)element_tree_.ReparentElement(id, old_parent);
    RecordSceneErrorLocked(v->code, v->message);
    return false;
  }
  if (extends) scope_extension_watch_ = true;
  WithdrawObservedFacingPointsBelowLocked(id);
  AdvanceSceneInputsLocked(true, true);
  return true;
}

bool Space::SetElementTransformLocked(element_id_t id,
                                      const Eigen::Matrix4d& transform,
                                      bool actor_pose) {
  auto* element = element_tree_.GetElement(id);
  if (!element || observed_facing_points_.count(id) || !ValidTransform(transform)) return false;
  if (element->GetLocalTransform() != transform) {
    if (!actor_pose) {
      const Eigen::Matrix4d previous = element->GetLocalTransform();
      element->SetLocalTransform(transform);
      bool extends = false;
      if (auto v = FindContainmentViolation(element_tree_, id, &extends)) {
        element->SetLocalTransform(previous);  // not registered
        RecordSceneErrorLocked(v->code, v->message);
        return false;
      }
      if (extends) scope_extension_watch_ = true;
      element->SetLocalTransform(previous);  // re-applied below
    }
    WithdrawObservedFacingPointsBelowLocked(id);
    element->SetLocalTransform(transform);
    AdvanceSceneInputsLocked(true, true, /*geometry=*/!actor_pose);
  }
  return true;
}

bool Space::SetElementTransform(element_id_t id,
                                const Eigen::Matrix4d& transform) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  RecordSceneErrorLocked(SceneErrorCode::kOk, "");
  return SetElementTransformLocked(id, transform);
}

bool Space::SetElementSharedId(element_id_t id, shared_element_id_t shared_id) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto* element = element_tree_.GetElement(id);
  if (!element || observed_facing_points_.count(id)) return false;
  if (element->GetSharedId() != shared_id) {
    element->SetSharedId(shared_id);
    AdvanceSceneInputsLocked(true, false);
    PruneObservedFacingPointsLocked();
  }
  return true;
}

bool Space::SetElementTag(element_id_t id, const std::string& tag) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto* element = element_tree_.GetElement(id);
  if (!element || observed_facing_points_.count(id)) return false;
  if (element->GetTag() != tag) {
    element->SetTag(tag);
    AdvanceSceneInputsLocked(true, true);
  }
  return true;
}

bool Space::SetElementAffordancePose(element_id_t id, const AffordancePoseTemplate* pose) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto* element = element_tree_.GetElement(id);
  if (!element || observed_facing_points_.count(id)) return false;
  const AffordancePoseTemplate* now = element->GetAffordancePose();
  if (pose && (!now || !(*now == *pose))) {
    element->SetAffordancePose(*pose);
    AdvanceSceneInputsLocked(true, true);
  } else if (!pose && now) {
    element->ClearAffordancePose();
    AdvanceSceneInputsLocked(true, true);
  }
  return true;
}

bool Space::SetElementSpatialDefinition(
    element_id_t id, const std::shared_ptr<SpatialDefinition>& definition,
    bool only_if_absent) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  RecordSceneErrorLocked(SceneErrorCode::kOk, "");
  auto* element = element_tree_.GetElement(id);
  if (!element || observed_facing_points_.count(id)) return false;
  if (only_if_absent && element->GetSpatialDefinition()) return true;
  auto owned = CopyDefinition(definition.get());
  if (definition && !owned) return false;
  if (!SameDefinition(element->GetDeclaredSpatialDefinition(), owned.get())) {
    const auto previous = element->GetDeclaredSpatialDefinitionShared();
    const auto previous_effective = element->GetSpatialDefinitionShared();
    const auto previous_scope = element->GetScopeDefinitionShared();
    const bool was_extended = element->HasScopeExtension();
    element->SetSpatialDefinition(owned);
    bool extends = false;
    if (auto v = FindContainmentViolation(element_tree_, id, &extends)) {
      element->SetSpatialDefinition(previous);  // not registered
      element->SetEffectiveSpatialDefinition(previous_effective);
      element->SetScopeDefinition(previous_scope);
      RecordSceneErrorLocked(v->code, v->message);
      return false;
    }
    // An extended region keeps its last complete scope and walkable until the rebuild from the
    // new declaration is installed (the watch is on, so AdvanceSceneInputsLocked requests it).
    if (was_extended && owned) {
      element->SetEffectiveSpatialDefinition(previous_effective);
      element->SetScopeDefinition(previous_scope);
    }
    if (extends) scope_extension_watch_ = true;
    AdvanceSceneInputsLocked(true, true);
  }
  return true;
}

bool Space::AddElementSubRegion(element_id_t id, const SubRegion& region) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto* element = element_tree_.GetElement(id);
  if (!element || !element->GetSpatialDefinition()) return false;
  SubRegion owned;
  if (!CopyRegion(region, &owned)) return false;
  // The declared definition keeps it too (an extension is rebuilt from it).
  if (element->HasScopeExtension() && element->GetDeclaredSpatialDefinition()) {
    if (element->GetDeclaredSpatialDefinition() != element->GetSpatialDefinition())
      element->GetDeclaredSpatialDefinition()->AddSubRegion(owned);
    SpatialDefinition* scope = element->GetScopeDefinition();
    if (scope && scope != element->GetDeclaredSpatialDefinition() && scope != element->GetSpatialDefinition())
      scope->AddSubRegion(owned);
  }
  element->GetSpatialDefinition()->AddSubRegion(std::move(owned));
  AdvanceSceneInputsLocked(false, true);
  return true;
}

SurfaceChartStatus Space::SetElementSubRegionChart(element_id_t id, const std::string& name, SurfaceChartKind chart) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto* element = element_tree_.GetElement(id);
  if (!element || !element->GetSpatialDefinition()) return SurfaceChartStatus::kUnknownPart;
  const SubRegion* region = element->GetSpatialDefinition()->GetSubRegion(name);
  if (!region) return SurfaceChartStatus::kUnknownPart;
  if (IsFrameChart(chart) && (!region->boundary || !PlanarFrameChart::ExtentOf(*region->boundary).ok()))
    return SurfaceChartStatus::kNotARectangle;
  if (region->chart == chart) return SurfaceChartStatus::kOk;
  element->GetSpatialDefinition()->SetSubRegionChart(name, chart);
  if (element->HasScopeExtension() && element->GetDeclaredSpatialDefinition()) {
    if (element->GetDeclaredSpatialDefinition() != element->GetSpatialDefinition())
      element->GetDeclaredSpatialDefinition()->SetSubRegionChart(name, chart);
    SpatialDefinition* scope = element->GetScopeDefinition();
    if (scope && scope != element->GetDeclaredSpatialDefinition() && scope != element->GetSpatialDefinition())
      scope->SetSubRegionChart(name, chart);
  }
  AdvanceSceneInputsLocked(false, true);
  return SurfaceChartStatus::kOk;
}

bool Space::GetElementPartFrame(element_id_t id, const std::string& name, PartFrame* out) const {
  if (!out) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  const Element* element = element_tree_.GetElement(id);
  if (!element || !element->GetDeclaredSpatialDefinition()) return false;
  const SpatialDefinition* def = element->GetDeclaredSpatialDefinition();
  const SubRegion* region = def->GetSubRegion(name);
  if (!region || !region->boundary) return false;
  // The same lift GroundAffordanceTarget and AffordancePoseGroundingRule use for a patch.
  const Eigen::Matrix4d def_to_world =
      def->RequiresElementWorldTransform() ? element_tree_.GetWorldTransform(id) : def->GetTransform();
  out->part_to_world = def_to_world * region->local_transform;
  out->boundary = region->boundary;
  out->chart = region->chart;
  return out->part_to_world.allFinite();
}

bool Space::AddElementAnchor(element_id_t id, const std::string& name,
                             const Eigen::Matrix4d& transform) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto* element = element_tree_.GetElement(id);
  if (!element || observed_facing_points_.count(id) || name.empty() || !ValidTransform(transform)) return false;
  const auto* previous = element->GetAnchor(name);
  if (!previous || *previous != transform) {
    element->AddAnchor(name, transform);
    AdvanceSceneInputsLocked(true, true);
  }
  return true;
}

bool Space::SetActorElement(spatialblend::entity_id_t actor_id, element_id_t id) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  RecordSceneErrorLocked(SceneErrorCode::kOk, "");
  auto actor = actors_.find(actor_id);
  auto* element = element_tree_.GetElement(id);
  if (actor == actors_.end() || (id != kInvalidElementId && !element))
    return false;
  if (element) {
    for (element_id_t child : element_tree_.GetSubtreeIds(id))
      if (observed_facing_points_.count(child)) return false;
  }
  if (actor->second->GetElementId() == id &&
      (!element || element->GetGroundingEntityId() == actor_id))
    return true;
  auto* previous = element_tree_.GetElement(actor->second->GetElementId());
  // The element the actor leaves stops being an actor's (unless another actor holds it): the
  // release always succeeds. Its regions extend their parents by the normal rule (the build
  // skips one too far from its parent, so it extends nothing), and its next declaration or move
  // is validated as any element's (scope_extension.h).
  if (previous && previous->GetGroundingEntityId() == actor_id)
    previous->SetGroundingEntityId(spatialblend::kInvalidEntityId);
  actor->second->SetElementId(id);
  InvalidateSourceEvidencePhaseLocked(actor_id);
  InvalidateReceiverInputLocked(actor_id);
  if (element) element->SetGroundingEntityId(actor_id);
  // The element's kind changed: an extension it (or the element it left) gave a parent is
  // re-evaluated now, whichever was registered first (scope_extension.h: actors never extend).
  auto extends_below = [&](element_id_t root) {
    if (!element_tree_.GetElement(root)) return false;
    for (element_id_t below : element_tree_.GetSubtreeIds(root))
      if (ExtendsParentScope(element_tree_, below)) return true;
    return false;
  };
  if (extends_below(id) || (previous && extends_below(previous->GetId())))
    scope_extension_watch_ = true;
  AdvanceSceneInputsLocked(true, false);
  PruneObservedFacingPointsLocked();
  return true;
}

bool Space::HasElement(element_id_t id) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return element_tree_.GetElement(id) != nullptr;
}

bool Space::TryHasElementOutsideCallback(element_id_t id) const {
  std::unique_lock<std::mutex> lock(space_mutex_, std::try_to_lock);
  return lock.owns_lock() && element_tree_.GetElement(id) != nullptr;
}

bool Space::HasActor(spatialblend::entity_id_t actor_id) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return actors_.count(actor_id) != 0;
}

bool Space::AddActorAnchors(spatialblend::entity_id_t actor_id,
                            const std::vector<SubjectAnchor>& anchors,
                            bool reject_duplicates) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  auto* actor = FindActorLocked(actor_id);
  if (!actor) return false;
  std::unordered_set<std::string> names;
  for (const auto& anchor : anchors) {
    if (anchor.name.empty() || !anchor.local_offset.allFinite() ||
        anchor.parent == anchor.name)
      return false;
    if (reject_duplicates &&
        (actor->GetAnchor(anchor.name) || !names.insert(anchor.name).second))
      return false;
  }
  for (const auto& anchor : anchors) actor->AddAnchor(anchor);
  if (!anchors.empty()) {
    InvalidateSourceEvidencePhaseLocked(actor_id);
    AdvanceSceneInputsLocked(true, false);
    InvalidateReceiverInputLocked(actor_id);
  }
  return true;
}

bool Space::CopyActorPose(spatialblend::entity_id_t actor_id,
                          Eigen::Vector2d* position, double* yaw) {
  if (!position || !yaw) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  PruneObservedFacingPointsLocked();
  SyncEmbodiedMirrorsLocked();
  const auto* actor = FindActorLocked(actor_id);
  if (!actor) return false;
  // A receiver avatar's pose IS the driver's sample; the Actor transform is
  // only a derived mirror of it. A source actor's pose is its own feed.
  const bool source_fed =
      actor_update_seconds_.count(actor_id) || observation_clocks_.count(actor_id);
  if (!source_fed) {
    // (An input-owned actor has no display pose: its Actor transform is never
    // written, and its frame is read through the proposal's sample identity.)
    if (const auto mailbox = MailboxFor(actor_id);
        mailbox && !mailbox->input_owned.load()) {
      const EmbodiedSample sample = mailbox->port.Sample();
      if (sample.valid) {
        *position = Eigen::Vector2d(sample.x, sample.y);
        *yaw = sample.yaw;
        return true;
      }
    }
  }
  *position = actor->GetPosition();
  const auto& transform = actor->GetTransform();
  *yaw = std::atan2(transform(1, 0), transform(0, 0));
  return true;
}

element_id_t Space::FindElementBySharedId(shared_element_id_t shared_id) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  const auto* element = element_tree_.GetElementBySharedId(shared_id);
  return element ? element->GetId() : kInvalidElementId;
}

spatialblend::entity_id_t Space::FindActorBySharedId(
    shared_element_id_t shared_id) const {
  std::lock_guard<std::mutex> lock(space_mutex_);
  const auto* element = element_tree_.GetElementBySharedId(shared_id);
  if (!element) return spatialblend::kInvalidEntityId;
  const auto actor = actors_.find(element->GetGroundingEntityId());
  return actor != actors_.end() &&
                 actor->second->GetElementId() == element->GetId()
             ? actor->first
             : spatialblend::kInvalidEntityId;
}

bool Space::CopyElementState(element_id_t id, ElementState* out) const {
  if (!out) return false;
  std::lock_guard<std::mutex> lock(space_mutex_);
  const auto* element = element_tree_.GetElement(id);
  if (!element) return false;
  ElementState value;
  value.definition = CopyDefinition(element->GetDeclaredSpatialDefinition());
  if (element->GetDeclaredSpatialDefinition() && !value.definition) return false;
  value.id = id;
  value.parent = element->GetParentId();
  value.shared_id = element->GetSharedId();
  value.grounding_actor = element->GetGroundingEntityId();
  value.tag = element->GetTag();
  value.local_transform = element->GetLocalTransform();
  value.world_transform = element_tree_.GetWorldTransform(id);
  value.children = element->GetChildIds();
  value.anchors = element->GetAnchors();
  *out = std::move(value);
  return true;
}

}  // namespace spatialblend
