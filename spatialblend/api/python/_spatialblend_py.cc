// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/api/python/_spatialblend_py.cc
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "spatialblend/model/actor/actor.h"
#include "spatialblend/model/actor/subject_anchor.h"
#include "spatialblend/base/common/state_type.h"
#include "spatialblend/base/common/types.h"
#include "spatialblend/model/element/element.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/geometry_role.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/model/element/spatial_definition.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/geometry/polygon_coordinate_mapper.h"
#include "spatialblend/model/relation/grounded_relation.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/support.h"
#include "spatialblend/model/actor/actor_motion_tracker.h"
#include "spatialblend/space/binding.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/solver/realization_target.h"
#include "spatialblend/solver/relation_solver.h"
#include "spatialblend/space/relation_state.h"
#include "spatialblend/model/realization/remote_update.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/builder/actor_handle.h"
#include "spatialblend/space/builder/element_handle.h"
#include "spatialblend/space/builder/space_builder.h"
#include "spatialblend/space/builder/trajectory.h"
#include "spatialblend/space/space_hub.h"

namespace py = pybind11;
using namespace spatialblend;

// Trampoline: tells pybind11 "don't delete this, we don't own it".
// Used for objects returned as raw pointers from manager classes.
template <typename T>
using RefHolder = std::unique_ptr<T, py::nodelete>;

PYBIND11_MODULE(_spatialblend_py, m) {
  m.doc() = "Python bindings for spatialblend";

  // === Enums ===
  py::enum_<GeometryRole>(m, "GeometryRole")
      .value("UNSPECIFIED", GeometryRole::kUnspecified)
      .value("NAVIGABLE", GeometryRole::kNavigable)
      .value("CONTAINMENT", GeometryRole::kContainment)
      .value("OBSTACLE", GeometryRole::kObstacle)
      .value("SURFACE", GeometryRole::kSurface)
      .value("INTERACTION_REGION", GeometryRole::kInteractionRegion);

  py::enum_<RelationKernelType>(m, "KernelType")
      .value("SPACING", RelationKernelType::kAvatarSpacing)
      .value("FACING", RelationKernelType::kAvatarFacing)
      .value("AFFORDANCE_POSE", RelationKernelType::kAvatarAffordancePose);

  py::enum_<SolverStatus>(m, "SolverStatus")
      .value("CONVERGED", SolverStatus::kConverged)
      .value("FALLBACK", SolverStatus::kFallback)
      .value("FAILED", SolverStatus::kFailed)
      .value("NOT_CONVERGED", SolverStatus::kNotConverged)
      .value("BUDGET_HIT", SolverStatus::kBudgetHit);

  py::enum_<StateType>(m, "StateType")
      .value("POSITION_2D", StateType::kPosition2D)
      .value("POSITION_3D", StateType::kPosition3D)
      .value("ROTATION_YAW", StateType::kRotationYaw)
      .value("ROTATION_3D", StateType::kRotation3D);

  // === Geometry ===
  py::class_<Polygon2D, std::shared_ptr<Polygon2D>>(m, "Polygon2D")
      .def(py::init<const std::vector<Eigen::Vector2d>&>())
      .def("get_points", &Polygon2D::GetPoints)
      .def("contains", &Polygon2D::Contains,
           py::arg("point"), py::arg("include_boundary") = true);

  py::class_<Polygon3D, std::shared_ptr<Polygon3D>>(m, "Polygon3D")
      .def(py::init<const std::vector<Eigen::Vector3d>&>())
      .def("get_vertices", &Polygon3D::GetVertices)
      .def("get_normal", &Polygon3D::GetNormal)
      .def("get_center", &Polygon3D::GetCenter);

  // === Spatial Definitions ===
  // SpatialDefinition is abstract and always held by shared_ptr.
  py::class_<SpatialDefinition, std::shared_ptr<SpatialDefinition>>(
      m, "SpatialDefinition")
      .def("contains", &SpatialDefinition::Contains)
      .def("contains_3d", &SpatialDefinition::Contains3D)
      .def("get_boundary", &SpatialDefinition::GetBoundary)
      .def("get_closest_point", &SpatialDefinition::GetClosestPoint)
      .def("get_distance", &SpatialDefinition::GetDistance)
      .def("project_onto", &SpatialDefinition::ProjectOnto)
      .def("get_normal", &SpatialDefinition::GetNormal,
           py::arg("at_point") = Eigen::Vector3d::Zero())
      .def_property("geometry_role", &SpatialDefinition::GetGeometryRole,
                    &SpatialDefinition::SetGeometryRole);

  py::class_<NavigableSpatialDefinition, SpatialDefinition,
             std::shared_ptr<NavigableSpatialDefinition>>(
      m, "NavigableSpatialDefinition")
      .def(py::init<std::shared_ptr<Polygon2D>, std::shared_ptr<Polygon2D>,
                    double, GeometryRole>(),
           py::arg("walkable"), py::arg("wall") = nullptr,
           py::arg("agent_radius") = 0.0,
           py::arg("geometry_role") = GeometryRole::kNavigable)
      .def("get_walkable_polygon",
           &NavigableSpatialDefinition::GetWalkablePolygon);

  py::class_<SurfaceSpatialDefinition, SpatialDefinition,
             std::shared_ptr<SurfaceSpatialDefinition>>(
      m, "SurfaceSpatialDefinition")
      .def(py::init<std::shared_ptr<Polygon3D>>())
      .def("get_surface", &SurfaceSpatialDefinition::GetSurface);

  py::class_<PolytopeSpatialDefinition, SpatialDefinition,
             std::shared_ptr<PolytopeSpatialDefinition>>(
      m, "PolytopeSpatialDefinition")
      .def(py::init<const Eigen::Vector3d&, const Eigen::Vector3d&,
                    GeometryRole>(),
           py::arg("min_corner"), py::arg("max_corner"),
           py::arg("geometry_role") = GeometryRole::kUnspecified)
      .def(py::init<std::vector<Eigen::Vector3d>,
                    std::vector<std::vector<int>>, GeometryRole>(),
           py::arg("vertices"), py::arg("face_indices"),
           py::arg("geometry_role") = GeometryRole::kUnspecified)
      .def("get_vertices", &PolytopeSpatialDefinition::GetVertices)
      .def("get_face_indices",
           &PolytopeSpatialDefinition::GetFaceIndices)
      .def("get_min_corner", &PolytopeSpatialDefinition::GetMinCorner)
      .def("get_max_corner", &PolytopeSpatialDefinition::GetMaxCorner)
      .def("get_center", &PolytopeSpatialDefinition::GetCenter);

  // === Element ===
  // Element is returned as a raw pointer (reference into ElementTree's map).
  // Use nodelete holder so Python doesn't try to own/copy it.
  py::class_<Element, RefHolder<Element>>(m, "Element")
      .def_property_readonly("id", &Element::GetId)
      .def_property_readonly("geometry_role", &Element::GetGeometryRole)
      .def_property("shared_id", &Element::GetSharedId, &Element::SetSharedId)
      .def_property("tag", &Element::GetTag, &Element::SetTag)
      .def_property_readonly("parent_id", &Element::GetParentId)
      .def_property_readonly("child_ids", &Element::GetChildIds)
      .def("set_local_transform", &Element::SetLocalTransform)
      .def("get_local_transform", &Element::GetLocalTransform)
      .def("set_spatial_definition", &Element::SetSpatialDefinition)
      // The shared definition: Python reads the escaped tree without the space lock, and the
      // scope-extension worker may swap the effective definition (element.h).
      .def("get_spatial_definition", &Element::GetSpatialDefinitionShared)
      .def("add_anchor", &Element::AddAnchor)
      .def("get_anchor", &Element::GetAnchor,
           py::return_value_policy::reference)
      .def("get_anchors", &Element::GetAnchors,
           py::return_value_policy::reference);

  // === ElementTree ===
  // ElementTree is owned by Space; returned as raw pointer from Space.
  // Use nodelete holder to express reference semantics.
  py::class_<ElementTree, RefHolder<ElementTree>>(m, "ElementTree")
      .def("add_element",
           [](ElementTree& self, element_id_t id, element_id_t parent_id) {
             auto status = self.AddElement(id, parent_id);
             if (!status.ok()) {
               throw std::runtime_error(std::string(status.message()));
             }
           },
           py::arg("id"), py::arg("parent_id") = kInvalidElementId)
      .def("get_element",
           py::overload_cast<element_id_t>(&ElementTree::GetElement),
           py::return_value_policy::reference)
      .def("size", &ElementTree::Size)
      .def("get_direct_child_ids", &ElementTree::GetDirectChildIds)
      .def("get_world_transform", &ElementTree::GetWorldTransform);

  // === SubjectAnchor ===
  py::class_<SubjectAnchor>(m, "SubjectAnchor")
      .def(py::init<const std::string&, const Eigen::Vector3d&,
                    std::vector<StateType>>(),
           py::arg("name"),
           py::arg("local_offset") = Eigen::Vector3d::Zero(),
           py::arg("dof") = std::vector<StateType>{StateType::kPosition2D,
                                                    StateType::kRotationYaw})
      .def_readonly("name", &SubjectAnchor::name)
      .def_readonly("local_offset", &SubjectAnchor::local_offset)
      .def_readonly("dof", &SubjectAnchor::dof);

  // === AnchorWorldPoseOverride (anchor liveness) ===
  py::class_<AnchorWorldPoseOverride>(m, "AnchorWorldPoseOverride")
      .def_readonly("position", &AnchorWorldPoseOverride::position)
      .def_readonly("yaw", &AnchorWorldPoseOverride::yaw);

  // === Actor ===
  // Actor is owned by Space's unique_ptr map; returned as raw pointer.
  // Use nodelete holder; bind base-class methods directly on Actor.
  py::class_<Actor, RefHolder<Actor>>(m, "Actor")
      .def("get_id", &Actor::GetId)
      .def("get_position", &Actor::GetPosition)
      .def("get_rotation", &Actor::GetRotation)
      .def("get_transform", &Actor::GetTransform)
      .def("get_anchor",
           py::overload_cast<const std::string&>(&Actor::GetAnchor, py::const_),
           py::return_value_policy::reference)
      .def("get_anchors", &Actor::GetAnchors,
           py::return_value_policy::reference)
      .def("set_element_id", &Actor::SetElementId)
      .def("get_element_id", &Actor::GetElementId)
      .def("add_anchor", &Actor::AddAnchor)
      // Anchor liveness: the per-anchor world-pose override is read-only
      // here. Set or clear it through Space.set_anchor_world_pose /
      // Space.clear_anchor_world_pose: Space marks the anchor observation
      // changed, so the next tick propagates it and the movement gate never
      // treats the interval as a quiet heartbeat. Writing the Actor
      // directly bypassed that. The "root" anchor is exempt in practice:
      // Space::UpdateActorState drives root_pos from the actor transform.
      .def("get_anchor_world_pose_override",
           &Actor::GetAnchorWorldPoseOverride, py::arg("anchor_name"),
           py::return_value_policy::reference);

  // === Relation types (read-only for inspection) ===
  py::class_<GroundedRelation>(m, "GroundedRelation")
      .def_readonly("encoded_state", &GroundedRelation::encoded_state)
      .def_readonly("proximity", &GroundedRelation::proximity);

  py::class_<Support>(m, "Support")
      .def_readonly("world_transform", &Support::world_transform)
      .def_readonly("referent_element_id", &Support::referent_element_id)
      .def_readonly("contact_point_local", &Support::contact_point_local)
      .def_readonly("has_contact", &Support::has_contact)
      .def("get_position_2d", &Support::GetPosition2D);

  py::class_<RelationEvaluationResult>(m, "RelationEvaluationResult")
      .def_readonly("grounded", &RelationEvaluationResult::grounded)
      .def_readonly("support", &RelationEvaluationResult::support);

  py::class_<Binding::MatchResult>(m, "MatchResult")
      .def_readonly("element_id", &Binding::MatchResult::element_id)
      .def_readonly("eval", &Binding::MatchResult::eval);

  // === NormalizedPolygonPosition ===
  py::class_<NormalizedPolygonPosition> normalized_polygon_position(
      m, "NormalizedPolygonPosition");
  py::enum_<NormalizedPolygonPosition::Strategy>(normalized_polygon_position,
                                                  "Strategy")
      .value("RADIAL", NormalizedPolygonPosition::Strategy::kRadial)
      .value("MVC", NormalizedPolygonPosition::Strategy::kMVC);
  normalized_polygon_position
      .def_readonly("strategy", &NormalizedPolygonPosition::strategy)
      .def_readonly("angle", &NormalizedPolygonPosition::angle)
      .def_readonly("radial_ratio", &NormalizedPolygonPosition::radial_ratio)
      .def_readonly("mvc_weights", &NormalizedPolygonPosition::mvc_weights)
      .def_readonly("height_ratio", &NormalizedPolygonPosition::height_ratio);

  // === RelationState (relation-state query) ===
  // Read-only snapshot of one (anchor, referent) relation's currently
  // grounded state, pulled directly from Binding::latest_matches — the same
  // numbers already propagated this frame, never a separate recomputation.
  py::class_<RelationState>(m, "RelationState")
      .def_readonly("anchor_name", &RelationState::anchor_name)
      .def_readonly("kernel_type", &RelationState::kernel_type)
      .def_readonly("referent_element_id",
                    &RelationState::referent_element_id)
      .def_readonly("referent_shared_id", &RelationState::referent_shared_id)
      .def_readonly("proximity", &RelationState::proximity)
      .def_readonly("has_contact", &RelationState::has_contact)
      .def_readonly("contact_point_world",
                    &RelationState::contact_point_world)
      .def_readonly("contact_normalized", &RelationState::contact_normalized)
      .def_readonly("active", &RelationState::active);

  // Binding is move-only (contains unique_ptr members).
  // It lives inside vectors owned by Space; return by reference.
  // Use nodelete holder to express reference semantics.
  py::class_<Binding, RefHolder<Binding>>(m, "Binding")
      .def_readonly("anchor_name", &Binding::anchor_name)
      .def_readonly("kernel_type", &Binding::kernel_type)
      .def_readonly("latest_matches", &Binding::latest_matches);

  // === SatisfactionMetric ===
  py::class_<SatisfactionMetric>(m, "SatisfactionMetric")
      .def_readonly("remaining", &SatisfactionMetric::remaining)
      .def_readonly("estimated_time", &SatisfactionMetric::estimated_time);

  // === ActorMotionState ===
  py::class_<ActorMotionState>(m, "ActorMotionState")
      .def_readonly("linear_speed", &ActorMotionState::linear_speed)
      .def_readonly("angular_speed", &ActorMotionState::angular_speed)
      .def_readonly("velocity", &ActorMotionState::velocity);

  // === RemoteRelationUpdate (read-only) ===
  py::class_<RelationPayload>(m, "RelationPayload")
      .def_readonly("referent_shared_id", &RelationPayload::referent_shared_id)
      .def_readonly("subject_anchor_name",
                    &RelationPayload::subject_anchor_name)
      .def_readonly("grounded_relation", &RelationPayload::grounded_relation)
      .def_readonly("support", &RelationPayload::support)
      .def_readonly("kernel_type", &RelationPayload::kernel_type)
      .def_readonly("estimated_time", &RelationPayload::estimated_time);

  py::class_<RemoteRelationUpdate>(m, "RemoteRelationUpdate")
      .def_readonly("source_space_id", &RemoteRelationUpdate::source_space_id)
      .def_readonly("actor_shared_id", &RemoteRelationUpdate::actor_shared_id)
      .def_readonly("payloads", &RemoteRelationUpdate::payloads)
      .def_readonly("step_duration", &RemoteRelationUpdate::step_duration);

  // === ElementEvent ===
  py::class_<ElementEvent>(m, "ElementEvent")
      .def_readwrite("type", &ElementEvent::type)
      .def_readwrite("shared_id", &ElementEvent::shared_id)
      .def_readwrite("parent_shared_id", &ElementEvent::parent_shared_id)
      .def_readwrite("tag", &ElementEvent::tag);

  // === RealizationTarget / RealizationResult ===
  py::class_<RealizationTarget>(m, "RealizationTarget")
      .def(py::init<>())
      .def_readonly("arrival_time", &RealizationTarget::arrival_time)
      .def_readonly("anchor_name", &RealizationTarget::anchor_name)
      .def_readonly("position", &RealizationTarget::position)
      .def_readonly("z", &RealizationTarget::z)
      .def_readonly("has_z", &RealizationTarget::has_z)
      .def_readonly("yaw", &RealizationTarget::yaw)
      .def_readonly("kernel_type", &RealizationTarget::kernel_type)
      .def_readonly("referent_id", &RealizationTarget::referent_id)
      .def_readonly("weight", &RealizationTarget::weight);

  py::class_<RealizationResult>(m, "RealizationResult")
      .def_readonly("targets", &RealizationResult::targets)
      .def_readonly("status", &RealizationResult::status);

  // === Space ===
  // Space can be constructed standalone OR managed by LocalContext.
  // When constructed standalone, Python owns it (default unique_ptr holder).
  // When returned from LocalContext::AddSpace(), use reference policy.
  // nodelete holder: LocalContext owns the Space, Python just references it.
  py::class_<Space, RefHolder<Space>>(m, "Space")
      .def(py::init<>())
      .def_property_readonly("element_tree",
           py::overload_cast<>(&Space::GetElementTree),
           py::return_value_policy::reference)
      .def("create_actor", &Space::CreateActor)
      .def("get_actor",
           py::overload_cast<spatialblend::entity_id_t>(&Space::GetActor),
           py::return_value_policy::reference)
      .def("bind", &Space::Bind,
           py::arg("actor_id"), py::arg("anchor_name"),
           py::arg("kernel_type"), py::arg("tag_filter") = "",
           py::return_value_policy::reference)
      .def("get_bindings", &Space::GetBindings,
           py::return_value_policy::reference)
      .def("update_actor_state",
           py::overload_cast<spatialblend::entity_id_t, const Eigen::Vector2d&,
                             double>(&Space::UpdateActorState))
      .def("get_space_id", &Space::GetSpaceId)
      .def("set_space_id", &Space::SetSpaceId)
      .def("set_hub", &Space::SetHub)
      .def("get_received_updates", &Space::GetReceivedUpdates,
           py::return_value_policy::reference)
      .def("share_element", &Space::ShareElement, py::arg("elem_id"),
           py::arg("user_data") = std::string())
      .def("unshare_element", &Space::UnshareElement)
      .def("set_update_interval", &Space::SetUpdateInterval)
      // Embodied-pose sampling port and proposal (ABI 3): the DRIVER
      // publishes its pose; core proposes.
      .def("publish_embodied_pose",
           [](Space& self, spatialblend::entity_id_t actor, double x, double y,
              double yaw, double timestamp_s, uint64_t sequence,
              uint64_t epoch) {
             EmbodiedPoseInput in;
             in.x = x;
             in.y = y;
             in.yaw = yaw;
             in.epoch = epoch;
             return static_cast<int>(
                 self.PublishEmbodiedPose(actor, in, timestamp_s, sequence));
           },
           py::arg("actor_id"), py::arg("x"), py::arg("y"), py::arg("yaw"),
           py::arg("timestamp_s"), py::arg("sequence"), py::arg("epoch") = 1)
      .def("read_proposal",
           [](Space& self, spatialblend::entity_id_t actor) {
             ReceiverProposal p;
             self.ReadProposal(actor, &p);
             return py::make_tuple(static_cast<int>(p.status), p.x, p.y, p.yaw,
                                   p.cause, p.sample_sequence);
           },
           py::arg("actor_id"))
      .def("get_realization_targets", &Space::GetRealizationTargets,
           py::return_value_policy::reference)
      // Anchor liveness / relation-state query.
      .def("set_anchor_world_pose", &Space::SetAnchorWorldPose,
           py::arg("actor_id"), py::arg("anchor_name"), py::arg("position"),
           py::arg("yaw"))
      .def("clear_anchor_world_pose", &Space::ClearAnchorWorldPose,
           py::arg("actor_id"), py::arg("anchor_name"))
      .def("get_relation_states", &Space::GetRelationStates,
           py::arg("actor_id"));

  // === SpaceHub ===
  // SpaceHub is owned by LocalContext; returned as raw pointer.
  py::class_<SpaceHub, RefHolder<SpaceHub>>(m, "SpaceHub")
      .def(py::init<>())
      .def("connect",
           [](SpaceHub& self, Space* space) {
             auto status = self.Connect(space);
             if (!status.ok()) {
               throw std::runtime_error(std::string(status.message()));
             }
           })
      .def("propagate", &SpaceHub::Propagate)
      .def("get_connected_space_count", &SpaceHub::GetConnectedSpaceCount);

  // === LocalContext ===
  // LocalContext is always created via factory and held by unique_ptr.
  py::class_<LocalContext, std::unique_ptr<LocalContext>>(m, "LocalContext")
      .def_static("create", &LocalContext::Create)
      .def("add_space", &LocalContext::AddSpace,
           py::return_value_policy::reference)
      .def_property_readonly("hub", &LocalContext::GetHub,
           py::return_value_policy::reference);

  // === Trajectory ===
  py::class_<Trajectory>(m, "Trajectory")
      .def(py::init<>())
      .def("add_waypoint", &Trajectory::AddWaypoint,
           py::arg("time"), py::arg("position"), py::arg("yaw"))
      .def("position_at", &Trajectory::PositionAt, py::arg("t"))
      .def("yaw_at", &Trajectory::YawAt, py::arg("t"))
      .def("duration", &Trajectory::Duration)
      .def_static("straight_line", &Trajectory::StraightLine,
                  py::arg("from_pos"), py::arg("to_pos"), py::arg("duration"))
      .def_static("from_waypoints", &Trajectory::FromWaypoints,
                  py::arg("waypoints"));

  // === ElementHandle ===
  py::class_<ElementHandle>(m, "ElementHandle")
      .def("with_polygon",
           [](ElementHandle& self,
              const std::vector<Eigen::Vector2d>& pts) -> ElementHandle& {
             self.WithPolygon(pts);
             return self;
           },
           py::arg("points"), py::return_value_policy::reference)
      .def("with_boundary",
           [](ElementHandle& self,
              const std::vector<Eigen::Vector2d>& pts) -> ElementHandle& {
             self.WithBoundary(pts);
             return self;
           },
           py::arg("points"), py::return_value_policy::reference)
      .def("with_shared_id",
           [](ElementHandle& self, int64_t id) -> ElementHandle& {
             self.WithSharedId(id);
             return self;
           },
           py::arg("id"), py::return_value_policy::reference)
      .def("at",
           [](ElementHandle& self, double x, double y) -> ElementHandle& {
             self.At(x, y);
             return self;
           },
           py::arg("x"), py::arg("y"), py::return_value_policy::reference)
      .def("tagged",
           [](ElementHandle& self, const std::string& tag) -> ElementHandle& {
             self.Tagged(tag);
             return self;
           },
           py::arg("tag"), py::return_value_policy::reference)
      .def("add_child", &ElementHandle::AddChild, py::arg("name"))
      .def("position", &ElementHandle::Position)
      .def("id", &ElementHandle::id);

  // === ActorHandle ===
  py::class_<ActorHandle>(m, "ActorHandle")
      .def("with_anchor",
           [](ActorHandle& self, const std::string& name, double x, double y,
              double z) -> ActorHandle& {
             self.WithAnchor(name, x, y, z);
             return self;
           },
           py::arg("name"), py::arg("x"), py::arg("y"), py::arg("z"),
           py::return_value_policy::reference)
      .def("in_scope",
           [](ActorHandle& self, const ElementHandle& elem) -> ActorHandle& {
             self.InScope(elem);
             return self;
           },
           py::arg("element"), py::return_value_policy::reference)
      .def("bind", &ActorHandle::Bind,
           py::arg("anchor"), py::arg("kernel_type"),
           py::arg("tag_filter") = "")
      .def("move_to",
           py::overload_cast<const Eigen::Vector2d&, double>(
               &ActorHandle::MoveTo),
           py::arg("pos"), py::arg("yaw"))
      .def("move_to",
           py::overload_cast<const Eigen::Vector2d&>(&ActorHandle::MoveTo),
           py::arg("pos"))
      .def("get_position", &ActorHandle::GetPosition)
      .def("get_yaw", &ActorHandle::GetYaw)
      .def("get_targets", &ActorHandle::GetTargets,
           py::return_value_policy::reference)
      .def("id", &ActorHandle::id);

  // === SpaceBuilder ===
  py::class_<SpaceBuilder>(m, "SpaceBuilder")
      .def("add_element", &SpaceBuilder::AddElement, py::arg("name"))
      .def("add_actor", &SpaceBuilder::AddActor)
      .def("get_space", &SpaceBuilder::GetSpace,
           py::return_value_policy::reference);
}
