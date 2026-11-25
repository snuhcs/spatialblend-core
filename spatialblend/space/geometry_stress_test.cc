// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/geometry_stress_test.cc
//
// Geometry-agnostic stress tests verifying that the relation system works
// correctly across all spatial definition types (Navigable, Surface) and all
// kernel types (Spacing, Facing, AffordancePose).

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "spatialblend/base/common/state_type.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace test {

namespace {

struct GeometryCase {
  std::string name;
  GeometryRole role;
  bool is_surface = false;
  std::vector<Eigen::Vector2d> points_2d;
  std::vector<Eigen::Vector3d> points_3d;
  Eigen::Vector3d on_geometry_probe = Eigen::Vector3d::Zero();
  Eigen::Vector3d off_geometry_probe = Eigen::Vector3d::Zero();
  bool contains_on_probe_2d = true;
  bool contains_on_probe_3d = true;
  bool spacing_targets_geometry = true;
  bool is_polytope = false;
  Eigen::Vector3d min_corner = Eigen::Vector3d::Zero();
  Eigen::Vector3d max_corner = Eigen::Vector3d::Zero();
};

std::vector<Eigen::Vector2d> Box2D(double min_x, double min_y,
                                   double max_x, double max_y) {
  return {{min_x, min_y}, {max_x, min_y}, {max_x, max_y}, {min_x, max_y}};
}

std::shared_ptr<SpatialDefinition> MakeSpatialDefinition(
    const GeometryCase& geometry) {
  if (geometry.is_polytope) {
    return std::make_shared<PolytopeSpatialDefinition>(
        geometry.min_corner, geometry.max_corner, geometry.role);
  }
  if (geometry.is_surface) {
    return std::make_shared<SurfaceSpatialDefinition>(
        std::make_shared<Polygon3D>(geometry.points_3d));
  }
  return std::make_shared<NavigableSpatialDefinition>(
      std::make_shared<Polygon2D>(geometry.points_2d),
      std::shared_ptr<Polygon2D>(), 0.0, geometry.role);
}

std::vector<GeometryCase> AllGeometryCases() {
  return {
      GeometryCase{
          "FloorNavmesh2D",
          GeometryRole::kNavigable,
          false,
          {{4, 4}, {6, 4}, {6, 5}, {5, 5}, {5, 6}, {4, 6}},
          {},
          Eigen::Vector3d(4.5, 4.5, 0.0),
          Eigen::Vector3d(8.0, 5.0, 0.0),
          true,
          true,
          true,
      },
      GeometryCase{
          "ContainmentRegion2D",
          GeometryRole::kContainment,
          false,
          Box2D(4, 4, 6, 6),
          {},
          Eigen::Vector3d(5.0, 5.0, 0.0),
          Eigen::Vector3d(8.0, 5.0, 0.0),
          true,
          true,
          true,
      },
      GeometryCase{
          "ObstacleFootprint2D",
          GeometryRole::kObstacle,
          false,
          Box2D(4, 4, 6, 6),
          {},
          Eigen::Vector3d(5.0, 5.0, 0.0),
          Eigen::Vector3d(8.0, 5.0, 0.0),
          true,
          true,
          true,
      },
      GeometryCase{
          "InteractionRegion2D",
          GeometryRole::kInteractionRegion,
          false,
          Box2D(4, 4, 6, 6),
          {},
          Eigen::Vector3d(5.0, 5.0, 0.0),
          Eigen::Vector3d(8.0, 5.0, 0.0),
          true,
          true,
          false,
      },
      GeometryCase{
          "UnspecifiedFootprint2D",
          GeometryRole::kUnspecified,
          false,
          Box2D(4, 4, 6, 6),
          {},
          Eigen::Vector3d(5.0, 5.0, 0.0),
          Eigen::Vector3d(8.0, 5.0, 0.0),
          true,
          true,
          true,
      },
      GeometryCase{
          "BoxVolumeObstacle3D",
          GeometryRole::kObstacle,
          false,
          {},
          {},
          Eigen::Vector3d(5.0, 5.0, 0.5),
          Eigen::Vector3d(8.0, 5.0, 0.5),
          true,
          true,
          true,
          true,
          Eigen::Vector3d(4.0, 4.0, 0.0),
          Eigen::Vector3d(6.0, 6.0, 1.0),
      },
      GeometryCase{
          "BoxVolumeContainment3D",
          GeometryRole::kContainment,
          false,
          {},
          {},
          Eigen::Vector3d(5.0, 5.0, 0.5),
          Eigen::Vector3d(8.0, 5.0, 0.5),
          true,
          true,
          true,
          true,
          Eigen::Vector3d(4.0, 4.0, 0.0),
          Eigen::Vector3d(6.0, 6.0, 1.0),
      },
      GeometryCase{
          "BoxVolumeInteractionRegion3D",
          GeometryRole::kInteractionRegion,
          false,
          {},
          {},
          Eigen::Vector3d(5.0, 5.0, 0.5),
          Eigen::Vector3d(8.0, 5.0, 0.5),
          true,
          true,
          false,
          true,
          Eigen::Vector3d(4.0, 4.0, 0.0),
          Eigen::Vector3d(6.0, 6.0, 1.0),
      },
      GeometryCase{
          "HorizontalSurface3D",
          GeometryRole::kSurface,
          true,
          {},
          {{4, 4, 1}, {6, 4, 1}, {6, 6, 1}, {4, 6, 1}},
          Eigen::Vector3d(5.0, 5.0, 1.0),
          Eigen::Vector3d(5.0, 5.0, 3.0),
          true,
          true,
          true,
      },
      GeometryCase{
          "VerticalSurface3D",
          GeometryRole::kSurface,
          true,
          {},
          {{4, 6, 0}, {6, 6, 0}, {6, 6, 2}, {4, 6, 2}},
          Eigen::Vector3d(5.0, 6.0, 1.0),
          Eigen::Vector3d(5.0, 3.0, 1.0),
          true,
          true,
          true,
      },
      GeometryCase{
          "AngledSurface3D",
          GeometryRole::kSurface,
          true,
          {},
          {{4, 4, 0}, {6, 4, 0}, {6, 6, 1}, {4, 6, 1}},
          Eigen::Vector3d(5.0, 5.0, 0.5),
          Eigen::Vector3d(5.0, 5.0, 3.0),
          true,
          true,
          true,
      },
  };
}

void ExpectFinite(const Eigen::Vector3d& value, const std::string& label) {
  EXPECT_TRUE(std::isfinite(value.x())) << label << ".x";
  EXPECT_TRUE(std::isfinite(value.y())) << label << ".y";
  EXPECT_TRUE(std::isfinite(value.z())) << label << ".z";
}

void ExpectSingleMatchTo(const Binding& binding, element_id_t expected_id,
                         const std::string& label) {
  SCOPED_TRACE(label);
  ASSERT_EQ(binding.latest_matches.size(), 1u);
  EXPECT_EQ(binding.latest_matches[0].element_id, expected_id);
  EXPECT_TRUE(std::isfinite(
      binding.latest_matches[0].eval.grounded.proximity));
  EXPECT_GT(binding.latest_matches[0].eval.grounded.encoded_state.size(), 0);
}

std::string GeometryCaseName(
    const ::testing::TestParamInfo<GeometryCase>& info) {
  return info.param.name;
}

}  // namespace

// ==========================================================================
// Test fixture
// ==========================================================================

class GeometryStressTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = LocalContext::Create();
    space_ = context_->AddSpace();
  }

  // Helper: create a Room element with NavigableSpatialDefinition.
  void AddRoom(element_id_t id, const std::vector<Eigen::Vector2d>& polygon) {
    auto* tree = space_->GetElementTree();
    ASSERT_TRUE(tree->AddElement(id).ok());
    auto walkable = std::make_shared<Polygon2D>(polygon);
    tree->GetElement(id)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(walkable));
  }

  // Helper: create a child element with NavigableSpatialDefinition.
  element_id_t AddNavigableElement(element_id_t parent, element_id_t id,
                                   int64_t shared_id, double x, double y,
                                   const std::vector<Eigen::Vector2d>& polygon,
                                   const std::string& tag = "") {
    auto* tree = space_->GetElementTree();
    EXPECT_TRUE(tree->AddElement(id, parent).ok());
    if (shared_id >= 0) tree->GetElement(id)->SetSharedId(shared_id);
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 3) = x;
    t(1, 3) = y;
    tree->GetElement(id)->SetLocalTransform(t);
    auto boundary = std::make_shared<Polygon2D>(polygon);
    tree->GetElement(id)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(boundary));
    if (!tag.empty()) tree->GetElement(id)->SetTag(tag);
    return id;
  }

  // Helper: create a child element with SurfaceSpatialDefinition.
  element_id_t AddSurfaceElement(element_id_t parent, element_id_t id,
                                 int64_t shared_id,
                                 const std::shared_ptr<Polygon3D>& surface,
                                 const std::string& tag = "") {
    auto* tree = space_->GetElementTree();
    auto center = surface->GetCenter();
    EXPECT_TRUE(tree->AddElement(id, parent).ok());
    if (shared_id >= 0) tree->GetElement(id)->SetSharedId(shared_id);
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 3) = center.x();
    t(1, 3) = center.y();
    tree->GetElement(id)->SetLocalTransform(t);
    tree->GetElement(id)->SetSpatialDefinition(
        std::make_shared<SurfaceSpatialDefinition>(surface));
    if (!tag.empty()) tree->GetElement(id)->SetTag(tag);
    return id;
  }

  // Helper: add affordance child element.
  element_id_t AddAffordance(element_id_t parent, element_id_t id,
                             int64_t shared_id, const std::string& tag) {
    auto* tree = space_->GetElementTree();
    EXPECT_TRUE(tree->AddElement(id, parent).ok());
    if (shared_id >= 0) tree->GetElement(id)->SetSharedId(shared_id);
    tree->GetElement(id)->SetTag(tag);
    // Placement is inherited from the parent through the tree (identity
    // local transform). This used to copy the parent's placement, putting the
    // affordance at twice the parent's offset -- off the walkable area and
    // outside any near-field; scope gating no longer tolerates that.
    tree->GetElement(id)->SetLocalTransform(Eigen::Matrix4d::Identity());
    return id;
  }

  // Helper: create actor with standard anchors, assigned to element.
  spatialblend::entity_id_t CreateActorInRoom(element_id_t room_id) {
    std::vector<SubjectAnchor> anchors = {
        SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
        SubjectAnchor("chest", Eigen::Vector3d(0, 0, 1.2)),
    };
    auto actor_id = space_->CreateActor(anchors);
    space_->GetActor(actor_id)->SetElementId(room_id);
    return actor_id;
  }

  // Helper: verify a binding produced matches with finite results.
  void ExpectValidMatches(const Binding& binding, const std::string& label) {
    SCOPED_TRACE(label);
    EXPECT_FALSE(binding.latest_matches.empty())
        << label << ": expected non-empty matches";
    for (const auto& match : binding.latest_matches) {
      EXPECT_TRUE(std::isfinite(match.eval.grounded.proximity))
          << label << ": proximity not finite";
      EXPECT_FALSE(match.eval.grounded.encoded_state.size() == 0)
          << label << ": encoded_state is empty";
      for (int i = 0; i < match.eval.grounded.encoded_state.size(); ++i) {
        EXPECT_TRUE(std::isfinite(match.eval.grounded.encoded_state[i]))
            << label << ": encoded_state[" << i << "] not finite";
      }
    }
  }

  std::unique_ptr<LocalContext> context_;
  Space* space_ = nullptr;
};

// ==========================================================================
// Parameterized geometry representation matrix
// ==========================================================================

class GeometryDefinitionMatrixTest
    : public ::testing::TestWithParam<GeometryCase> {};

TEST_P(GeometryDefinitionMatrixTest, SpatialDefinitionContract) {
  const GeometryCase& geometry = GetParam();
  SCOPED_TRACE(geometry.name);

  auto spatial_definition = MakeSpatialDefinition(geometry);
  ASSERT_NE(spatial_definition, nullptr);
  EXPECT_EQ(spatial_definition->GetGeometryRole(), geometry.role);
  EXPECT_EQ(IsContainmentRole(geometry.role),
            geometry.role == GeometryRole::kNavigable ||
                geometry.role == GeometryRole::kContainment);
  EXPECT_EQ(IsInteractionRole(geometry.role),
            geometry.role == GeometryRole::kInteractionRegion);

  auto boundary = spatial_definition->GetBoundary();
  ASSERT_NE(boundary, nullptr);
  EXPECT_GE(boundary->GetPoints().size(), 3u);

  Bound2D bounds = spatial_definition->GetBounds();
  EXPECT_TRUE(std::isfinite(bounds.min.x()));
  EXPECT_TRUE(std::isfinite(bounds.min.y()));
  EXPECT_TRUE(std::isfinite(bounds.max.x()));
  EXPECT_TRUE(std::isfinite(bounds.max.y()));
  EXPECT_LE(bounds.min.x(), bounds.max.x());
  EXPECT_LE(bounds.min.y(), bounds.max.y());

  EXPECT_EQ(
      spatial_definition->Contains(geometry.on_geometry_probe.head<2>()),
      geometry.contains_on_probe_2d);
  EXPECT_EQ(spatial_definition->Contains3D(geometry.on_geometry_probe),
            geometry.contains_on_probe_3d);

  ExpectFinite(spatial_definition->GetClosestPoint(geometry.off_geometry_probe),
               "closest_point");
  ExpectFinite(spatial_definition->ProjectOnto(geometry.off_geometry_probe),
               "projected_point");
  Eigen::Vector3d normal =
      spatial_definition->GetNormal(geometry.on_geometry_probe);
  ExpectFinite(normal, "normal");
  EXPECT_NEAR(normal.norm(), 1.0, 1e-6);

  double distance = spatial_definition->GetDistance(geometry.off_geometry_probe);
  EXPECT_TRUE(std::isfinite(distance));
  EXPECT_GE(distance, 0.0);

  if (geometry.is_polytope) {
    EXPECT_NE(dynamic_cast<PolytopeSpatialDefinition*>(
                  spatial_definition.get()),
              nullptr);
  } else if (geometry.is_surface) {
    EXPECT_NE(dynamic_cast<SurfaceSpatialDefinition*>(spatial_definition.get()),
              nullptr);
  } else {
    EXPECT_NE(
        dynamic_cast<NavigableSpatialDefinition*>(spatial_definition.get()),
        nullptr);
  }
}

INSTANTIATE_TEST_SUITE_P(
    GeometryRepresentations,
    GeometryDefinitionMatrixTest,
    ::testing::ValuesIn(AllGeometryCases()),
    GeometryCaseName);

class GeometryRelationMatrixTest
    : public GeometryStressTest,
      public ::testing::WithParamInterface<GeometryCase> {};

TEST_P(GeometryRelationMatrixTest, RelationKernelsRespectGeometryRoles) {
  const GeometryCase& geometry = GetParam();
  SCOPED_TRACE(geometry.name);

  AddRoom(1, {{0, 0}, {10, 0}, {10, 10}, {0, 10}});
  auto* tree = space_->GetElementTree();
  ASSERT_TRUE(tree->AddElement(2, 1).ok());
  Element* target = tree->GetElement(2);
  ASSERT_NE(target, nullptr);
  target->SetSharedId(200);
  target->SetTag("target");
  target->SetSpatialDefinition(MakeSpatialDefinition(geometry));

  std::vector<SubjectAnchor> anchors = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
      SubjectAnchor("hand", Eigen::Vector3d(0.25, 0.0, 1.0),
                    {StateType::kPosition3D, StateType::kRotation3D}),
  };
  auto actor_id = space_->CreateActor(anchors);
  space_->GetActor(actor_id)->SetElementId(1);

  space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor_id, "hand", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarFacing);
  space_->Bind(actor_id, "hand", RelationKernelType::kAvatarAffordancePose,
               "target");

  space_->UpdateActorState(actor_id, geometry.on_geometry_probe.head<2>(), 0.25);

  const auto& bindings = space_->GetBindings(actor_id);
  ASSERT_EQ(bindings.size(), 4u);

  if (geometry.spacing_targets_geometry) {
    ExpectSingleMatchTo(bindings[0], 2, "2D spacing");
    EXPECT_EQ(bindings[0].latest_matches[0].eval.grounded.encoded_state.size(),
              3);

    ExpectSingleMatchTo(bindings[1], 2, "3D spacing");
    EXPECT_EQ(bindings[1].latest_matches[0].eval.grounded.encoded_state.size(),
              6);
  } else {
    EXPECT_TRUE(bindings[0].latest_matches.empty()) << "2D spacing";
    EXPECT_TRUE(bindings[1].latest_matches.empty()) << "3D spacing";
  }

  ExpectSingleMatchTo(bindings[2], 2, "facing");
  EXPECT_EQ(bindings[2].latest_matches[0].eval.grounded.encoded_state.size(),
            2);

  ExpectSingleMatchTo(bindings[3], 2, "3D affordance pose");
  EXPECT_EQ(bindings[3].latest_matches[0].eval.grounded.encoded_state.size(),
            5);
}

INSTANTIATE_TEST_SUITE_P(
    GeometryRepresentations,
    GeometryRelationMatrixTest,
    ::testing::ValuesIn(AllGeometryCases()),
    GeometryCaseName);

TEST_F(GeometryStressTest, ContainmentPolytopeScopesToChildObjects) {
  AddRoom(1, {{0, 0}, {10, 0}, {10, 10}, {0, 10}});
  auto* tree = space_->GetElementTree();

  ASSERT_TRUE(tree->AddElement(2, 1).ok());
  Element* desk = tree->GetElement(2);
  ASSERT_NE(desk, nullptr);
  desk->SetSharedId(200);
  desk->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(4.0, 4.0, 0.0),
      Eigen::Vector3d(6.0, 6.0, 1.0),
      GeometryRole::kContainment));

  ASSERT_TRUE(tree->AddElement(3, 2).ok());
  Element* object = tree->GetElement(3);
  ASSERT_NE(object, nullptr);
  object->SetSharedId(300);
  object->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(4.75, 4.75, 0.7),
      Eigen::Vector3d(5.25, 5.25, 1.2),
      GeometryRole::kObstacle));

  std::vector<SubjectAnchor> anchors = {
      SubjectAnchor("hand", Eigen::Vector3d(0.0, 0.0, 0.9),
                    {StateType::kPosition3D, StateType::kRotation3D}),
  };
  auto actor_id = space_->CreateActor(anchors);
  space_->GetActor(actor_id)->SetElementId(1);
  space_->Bind(actor_id, "hand", RelationKernelType::kAvatarSpacing);

  // Inside the desk, outside its object.
  space_->UpdateActorState(actor_id, Eigen::Vector2d(4.4, 5.0), 0.0);

  const auto& bindings = space_->GetBindings(actor_id);
  ASSERT_EQ(bindings.size(), 1u);
  // Container contract: the container the actor is inside
  // STAYS a spacing referent (its container-self relation, depth below the
  // border, signed: monotone through the border) beside the child OBJECTS.
  // Actor at (4.4,5) is inside Desk(2, kContainment) -> two referents: its child
  // Object(3) and Desk(2) itself, whose border is the actor's containment
  // relation. Room(1) is the actor's own element and is filtered by
  // Space::UpdateActorState.
  ASSERT_EQ(bindings[0].latest_matches.size(), 2u);
  // Every match must have finite encoded state (6 components: 3D pos + 3D rot).
  for (const auto& m : bindings[0].latest_matches) {
    EXPECT_EQ(m.eval.grounded.encoded_state.size(), 6);
    EXPECT_TRUE(std::isfinite(m.eval.grounded.proximity));
  }
  // Object(3) is an approach target; Desk(2) (the container) is the other.
  bool has_object = false, has_desk = false;
  for (const auto& m : bindings[0].latest_matches) {
    if (m.element_id == 3) has_object = true;
    if (m.element_id == 2) has_desk = true;
  }
  EXPECT_TRUE(has_object) << "Object(3) should be an approach target";
  EXPECT_TRUE(has_desk) << "Desk(2) is the container-self referent";

  // Inside the object's own footprint (5,5) the object is the entered referent and the only one
  // in scope, replayed (SCOPE_CASES.md SP-INSIDE-05).
  space_->UpdateActorState(actor_id, Eigen::Vector2d(5.0, 5.0), 0.0);
  ASSERT_EQ(space_->GetBindings(actor_id)[0].latest_matches.size(), 1u);
  EXPECT_EQ(space_->GetBindings(actor_id)[0].latest_matches[0].element_id, 3);
}

// ==========================================================================
// Category 1: Each kernel x each spatial def type
// ==========================================================================

TEST_F(GeometryStressTest, SpacingWithNavigableSpatialDef) {
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  AddNavigableElement(1, 2, 200, 10, 10,
                      {{8, 8}, {12, 8}, {12, 12}, {8, 12}});
  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
  space_->UpdateActorState(actor_id, Eigen::Vector2d(5, 5), 0.0);
  ExpectValidMatches(space_->GetBindings(actor_id)[0],
                     "Spacing x Navigable");
}

TEST_F(GeometryStressTest, SpacingWithSurfaceSpatialDef) {
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  auto wall_surface = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      {3, 19, 0.5}, {7, 19, 0.5}, {7, 19, 2.5}, {3, 19, 2.5}});
  AddSurfaceElement(1, 2, 200, wall_surface);
  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
  space_->UpdateActorState(actor_id, Eigen::Vector2d(5, 10), 0.0);
  ExpectValidMatches(space_->GetBindings(actor_id)[0],
                     "Spacing x Surface");
}

TEST_F(GeometryStressTest, FacingWithNavigableSpatialDef) {
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  AddNavigableElement(1, 2, 200, 10, 10,
                      {{8, 8}, {12, 8}, {12, 12}, {8, 12}});
  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "chest", RelationKernelType::kAvatarFacing);
  space_->UpdateActorState(actor_id, Eigen::Vector2d(5, 5), M_PI / 4.0);
  ExpectValidMatches(space_->GetBindings(actor_id)[0],
                     "Facing x Navigable");
}

TEST_F(GeometryStressTest, FacingWithSurfaceSpatialDef) {
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  auto whiteboard = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      {3, 19, 0.5}, {7, 19, 0.5}, {7, 19, 2.5}, {3, 19, 2.5}});
  AddSurfaceElement(1, 2, 200, whiteboard);
  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "chest", RelationKernelType::kAvatarFacing);
  space_->UpdateActorState(actor_id, Eigen::Vector2d(5, 15), M_PI / 2.0);
  ExpectValidMatches(space_->GetBindings(actor_id)[0],
                     "Facing x Surface");
}

TEST_F(GeometryStressTest, AffordancePoseWithNavigableSpatialDef) {
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  AddNavigableElement(1, 2, 200, 10, 10,
                      {{8, 8}, {12, 8}, {12, 12}, {8, 12}});
  AddAffordance(2, 3, 300, "Sittable");
  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarAffordancePose,
               "Sittable");
  // Actor inside the navigable boundary
  space_->UpdateActorState(actor_id, Eigen::Vector2d(10, 10), 0.0);
  ExpectValidMatches(space_->GetBindings(actor_id)[0],
                     "AffordancePose x Navigable");
}

TEST_F(GeometryStressTest, AffordancePoseWithSurfaceSpatialDef) {
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  // Whiteboard parent with navigable boundary for proximal containment check
  AddNavigableElement(1, 2, 200, 5, 19,
                      {{3, 17}, {7, 17}, {7, 20}, {3, 20}});
  auto whiteboard = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      {3, 19, 0.5}, {7, 19, 0.5}, {7, 19, 2.5}, {3, 19, 2.5}});
  // Affordance with surface spatial def
  auto* tree = space_->GetElementTree();
  EXPECT_TRUE(tree->AddElement(3, 2).ok());
  tree->GetElement(3)->SetSharedId(300);
  tree->GetElement(3)->SetTag("Writable");
  tree->GetElement(3)->SetLocalTransform(tree->GetElement(2)->GetLocalTransform());

  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarAffordancePose,
               "Writable");
  // Actor inside the whiteboard parent boundary (3-7, 17-20)
  space_->UpdateActorState(actor_id, Eigen::Vector2d(5, 18), 0.0);
  ExpectValidMatches(space_->GetBindings(actor_id)[0],
                     "AffordancePose x Surface");
}

// ==========================================================================
// Category 2: Mixed spatial defs in same space
// ==========================================================================

TEST_F(GeometryStressTest, MixedSpatialDefsAllBindingsMatch) {
  // Room: NavigableSpatialDef (floor)
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});

  // Table: NavigableSpatialDef (boundary polygon on floor)
  AddNavigableElement(1, 2, 200, 10, 10,
                      {{8, 8}, {12, 8}, {12, 12}, {8, 12}});

  // Whiteboard: NavigableSpatialDef boundary for proximal containment
  AddNavigableElement(1, 3, 300, 5, 19,
                      {{3, 17}, {7, 17}, {7, 20}, {3, 20}});
  AddAffordance(3, 4, 400, "Writable");

  // Tabletop: SurfaceSpatialDef (horizontal Polygon3D at table height)
  auto tabletop_surface = std::make_shared<Polygon3D>(
      std::vector<Eigen::Vector3d>{
          {8, 8, 0.8}, {12, 8, 0.8}, {12, 12, 0.8}, {8, 12, 0.8}});
  AddSurfaceElement(1, 5, 500, tabletop_surface);

  auto actor_id = CreateActorInRoom(1);

  // Bind all three kernel types
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor_id, "chest", RelationKernelType::kAvatarFacing);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarAffordancePose,
               "Writable");

  // Actor inside Whiteboard boundary (3-7, 17-20) for proximal AffordancePose.
  space_->UpdateActorState(actor_id, Eigen::Vector2d(5, 18), M_PI / 4.0);

  auto& bindings = space_->GetBindings(actor_id);
  ASSERT_EQ(bindings.size(), 3u);

  // Spacing: direct children of Whiteboard(3) scope — but Whiteboard is deepest
  // containing element, its children are Writable(4). Room children also match.
  ExpectValidMatches(bindings[0], "Mixed: Spacing");

  // Facing: all descendants visible
  ExpectValidMatches(bindings[1], "Mixed: Facing");

  // AffordancePose with "Writable" filter → Writable(4) (inside parent boundary)
  ExpectValidMatches(bindings[2], "Mixed: AffordancePose");
  EXPECT_EQ(bindings[2].latest_matches.size(), 1u);
  EXPECT_EQ(bindings[2].latest_matches[0].element_id, 4);
}

// ==========================================================================
// Category 3: Cross-space transfer with geometry mismatch
// ==========================================================================

TEST_F(GeometryStressTest, CrossSpaceTransferGeometryMismatch) {
  // Source space (space_): Table has NavigableSpatialDef (2D boundary)
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  space_->GetElementTree()->GetElement(1)->SetSharedId(100);
  AddNavigableElement(1, 2, 200, 10, 10,
                      {{8, 8}, {12, 8}, {12, 12}, {8, 12}}, "Table");

  auto actor_id = CreateActorInRoom(1);
  space_->SetHub(context_->GetHub());

  // Target space: same shared_id Table but with SurfaceSpatialDef (3D surface)
  Space* space_b = context_->AddSpace();
  auto* tree_b = space_b->GetElementTree();
  ASSERT_TRUE(tree_b->AddElement(10).ok());
  tree_b->GetElement(10)->SetSharedId(100);
  auto walkable_b = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  tree_b->GetElement(10)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(walkable_b));

  ASSERT_TRUE(tree_b->AddElement(20, 10).ok());
  tree_b->GetElement(20)->SetSharedId(200);
  Eigen::Matrix4d table_t = Eigen::Matrix4d::Identity();
  table_t(0, 3) = 5.0;
  table_t(1, 3) = 5.0;
  tree_b->GetElement(20)->SetLocalTransform(table_t);
  auto table_surface = std::make_shared<Polygon3D>(
      std::vector<Eigen::Vector3d>{
          {3, 3, 0.8}, {7, 3, 0.8}, {7, 7, 0.8}, {3, 7, 0.8}});
  tree_b->GetElement(20)->SetSpatialDefinition(
      std::make_shared<SurfaceSpatialDefinition>(table_surface));

  // Create avatar in target space
  std::vector<SubjectAnchor> anchors = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
      SubjectAnchor("chest", Eigen::Vector3d(0, 0, 1.2)),
  };
  auto avatar_id = space_b->CreateActor(anchors);
  space_b->GetActor(avatar_id)->SetElementId(10);

  // Bind spacing in source, propagate to target.
  // Actor must be outside Table boundary so scope = Room → children = Table.
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
  space_->UpdateActorState(actor_id, Eigen::Vector2d(2, 2), 0.0);

  // Target space should have received an update
  auto& received = space_b->GetReceivedUpdates();
  EXPECT_FALSE(received.empty())
      << "Cross-space transfer should propagate despite geometry mismatch";

  // The received payload should reference the shared Table
  bool found_table = false;
  for (const auto& update : received) {
    for (const auto& payload : update.payloads) {
      if (payload.referent_shared_id == 200) {
        found_table = true;
        EXPECT_TRUE(std::isfinite(payload.grounded_relation.proximity));
        EXPECT_GT(payload.grounded_relation.encoded_state.size(), 0);
      }
    }
  }
  EXPECT_TRUE(found_table) << "Expected Table (shared=200) in payloads";
}

// ==========================================================================
// Category 4: Degenerate geometries
// ==========================================================================

class DegenerateGeometryTest : public GeometryStressTest {};

TEST_F(DegenerateGeometryTest, VerySmallPolygon) {
  // 1cm x 1cm polygon — actor outside, scope stays at Room
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  AddNavigableElement(1, 2, 200, 10, 10,
                      {{9.995, 9.995}, {10.005, 9.995},
                       {10.005, 10.005}, {9.995, 10.005}});
  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
  // Place actor clearly outside the tiny polygon
  space_->UpdateActorState(actor_id, Eigen::Vector2d(5, 5), 0.0);
  ExpectValidMatches(space_->GetBindings(actor_id)[0], "VerySmallPolygon");
}

TEST_F(DegenerateGeometryTest, VeryLargePolygon) {
  // 1000m x 1000m room
  AddRoom(1, {{0, 0}, {1000, 0}, {1000, 1000}, {0, 1000}});
  AddNavigableElement(1, 2, 200, 500, 500,
                      {{450, 450}, {550, 450}, {550, 550}, {450, 550}});
  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
  space_->UpdateActorState(actor_id, Eigen::Vector2d(100, 100), 0.0);
  ExpectValidMatches(space_->GetBindings(actor_id)[0], "VeryLargePolygon");
}

TEST_F(DegenerateGeometryTest, ThinPolygon) {
  // 0.1m x 10m corridor
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  AddNavigableElement(1, 2, 200, 10, 10,
                      {{9.95, 5}, {10.05, 5}, {10.05, 15}, {9.95, 15}});
  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
  space_->UpdateActorState(actor_id, Eigen::Vector2d(5, 10), 0.0);
  ExpectValidMatches(space_->GetBindings(actor_id)[0], "ThinPolygon");
}

TEST_F(DegenerateGeometryTest, LShapedPolygon) {
  // Non-convex L-shape — actor outside the L polygon
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  AddNavigableElement(1, 2, 200, 6, 6,
                      {{0, 0}, {12, 0}, {12, 8}, {6, 8}, {6, 16}, {0, 16}});
  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor_id, "chest", RelationKernelType::kAvatarFacing);
  // Place actor outside the L-shape (the L spans roughly 0-12, 0-16)
  space_->UpdateActorState(actor_id, Eigen::Vector2d(18, 18), 0.0);
  auto& bindings = space_->GetBindings(actor_id);
  ExpectValidMatches(bindings[0], "LShapedPolygon: Spacing");
  ExpectValidMatches(bindings[1], "LShapedPolygon: Facing");
}

TEST_F(DegenerateGeometryTest, VerticalSurface) {
  // Wall: Polygon3D with normal along Y axis
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  auto wall = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      {2, 0, 0}, {8, 0, 0}, {8, 0, 3}, {2, 0, 3}});
  AddSurfaceElement(1, 2, 200, wall);
  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor_id, "chest", RelationKernelType::kAvatarFacing);
  space_->UpdateActorState(actor_id, Eigen::Vector2d(5, 5), 0.0);
  auto& bindings = space_->GetBindings(actor_id);
  ExpectValidMatches(bindings[0], "VerticalSurface: Spacing");
  ExpectValidMatches(bindings[1], "VerticalSurface: Facing");
}

TEST_F(DegenerateGeometryTest, AngledSurface) {
  // 45-degree angled surface (ramp/slope)
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  double h = std::sqrt(2.0);  // diagonal height
  auto ramp = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      {5, 5, 0}, {10, 5, 0}, {10, 5 + h, h}, {5, 5 + h, h}});
  AddSurfaceElement(1, 2, 200, ramp);
  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor_id, "chest", RelationKernelType::kAvatarFacing);
  space_->UpdateActorState(actor_id, Eigen::Vector2d(7, 8), 0.0);
  auto& bindings = space_->GetBindings(actor_id);
  ExpectValidMatches(bindings[0], "AngledSurface: Spacing");
  ExpectValidMatches(bindings[1], "AngledSurface: Facing");
}

// ==========================================================================
// Category 5: Actor at boundary/edge positions
// ==========================================================================

class BoundaryPositionTest : public GeometryStressTest {
 protected:
  void SetUp() override {
    GeometryStressTest::SetUp();
    // Standard room with an object
    AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
    AddNavigableElement(1, 2, 200, 10, 10,
                        {{8, 8}, {12, 8}, {12, 12}, {8, 12}});
    actor_id_ = CreateActorInRoom(1);
    space_->Bind(actor_id_, "root", RelationKernelType::kAvatarSpacing);
    space_->Bind(actor_id_, "chest", RelationKernelType::kAvatarFacing);
  }

  spatialblend::entity_id_t actor_id_;
};

TEST_F(BoundaryPositionTest, ActorOnElementBoundaryEdge) {
  // Actor exactly on the object boundary edge (x=8, y between 8 and 12).
  // Contains may return true on boundary → scope narrows into the object.
  // Verify: no crash, Facing always matches (all descendants), Spacing may
  // be empty if scope drills into a leaf object.
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(8, 10), 0.0);
  auto& bindings = space_->GetBindings(actor_id_);
  // Facing uses all descendants of scope — always produces matches.
  ExpectValidMatches(bindings[1], "OnBoundaryEdge: Facing");
  // Spacing: scope may be Room(1) or Object(2) depending on Contains.
  // Either way, all results must be finite.
  for (const auto& match : bindings[0].latest_matches) {
    EXPECT_TRUE(std::isfinite(match.eval.grounded.proximity));
  }
}

TEST_F(BoundaryPositionTest, ActorOnElementBoundaryVertex) {
  // Actor exactly at the object boundary vertex.
  // Contains may return true → scope narrows. Verify no crash.
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(8, 8), 0.0);
  auto& bindings = space_->GetBindings(actor_id_);
  // Facing always produces matches from descendants.
  ExpectValidMatches(bindings[1], "OnBoundaryVertex: Facing");
  // Spacing may be empty if scope drills into the leaf object.
  for (const auto& match : bindings[0].latest_matches) {
    EXPECT_TRUE(std::isfinite(match.eval.grounded.proximity));
  }
}

TEST_F(BoundaryPositionTest, ActorAtOrigin) {
  // Actor exactly at (0, 0, 0) origin — corner of the room
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(0, 0), 0.0);
  auto& bindings = space_->GetBindings(actor_id_);
  ExpectValidMatches(bindings[0], "AtOrigin: Spacing");
  ExpectValidMatches(bindings[1], "AtOrigin: Facing");
}

TEST_F(BoundaryPositionTest, ActorVeryFarFromElement) {
  // Use a separate space with a large room so the actor can be 100m away.
  Space* large_space = context_->AddSpace();
  auto* tree = large_space->GetElementTree();
  ASSERT_TRUE(tree->AddElement(100).ok());
  auto walkable = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{
          {-200, -200}, {200, -200}, {200, 200}, {-200, 200}});
  tree->GetElement(100)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(walkable));
  ASSERT_TRUE(tree->AddElement(101, 100).ok());
  tree->GetElement(101)->SetSharedId(101);  // shared: an unshared element is never a referent
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = 10.0;
  t(1, 3) = 10.0;
  tree->GetElement(101)->SetLocalTransform(t);
  auto boundary = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{8, 8}, {12, 8}, {12, 12}, {8, 12}});
  tree->GetElement(101)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(boundary));

  std::vector<SubjectAnchor> anchors = {
      SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
      SubjectAnchor("chest", Eigen::Vector3d(0, 0, 1.2)),
  };
  auto far_actor_id = large_space->CreateActor(anchors);
  large_space->GetActor(far_actor_id)->SetElementId(100);
  large_space->Bind(far_actor_id, "root", RelationKernelType::kAvatarSpacing);
  large_space->Bind(far_actor_id, "chest", RelationKernelType::kAvatarFacing);

  large_space->UpdateActorState(far_actor_id, Eigen::Vector2d(110, 110), 0.0);
  auto& bindings = large_space->GetBindings(far_actor_id);
  ExpectValidMatches(bindings[0], "VeryFar: Spacing");
  ExpectValidMatches(bindings[1], "VeryFar: Facing");
}

TEST_F(BoundaryPositionTest, ActorOnRoomBoundary) {
  // Actor exactly on the room boundary edge (y=0)
  space_->UpdateActorState(actor_id_, Eigen::Vector2d(10, 0), 0.0);
  auto& bindings = space_->GetBindings(actor_id_);
  ExpectValidMatches(bindings[0], "OnRoomBoundary: Spacing");
  ExpectValidMatches(bindings[1], "OnRoomBoundary: Facing");
}

// ==========================================================================
// Category 4 (continued): All kernels x degenerate surfaces
// ==========================================================================

TEST_F(DegenerateGeometryTest, AllKernelsOnVerticalWall) {
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  // Vertical wall: use navigable boundary for proximal containment check
  AddNavigableElement(1, 2, 200, 5, 19,
                      {{3, 17}, {7, 17}, {7, 20}, {3, 20}});
  AddAffordance(2, 3, 300, "Writable");

  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor_id, "chest", RelationKernelType::kAvatarFacing);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarAffordancePose,
               "Writable");

  // Actor inside the wall parent boundary (3-7, 17-20)
  space_->UpdateActorState(actor_id, Eigen::Vector2d(5, 18), M_PI / 2.0);

  auto& bindings = space_->GetBindings(actor_id);
  ASSERT_EQ(bindings.size(), 3u);
  ExpectValidMatches(bindings[0], "VerticalWall: Spacing");
  ExpectValidMatches(bindings[1], "VerticalWall: Facing");
  ExpectValidMatches(bindings[2], "VerticalWall: AffordancePose");
}

TEST_F(DegenerateGeometryTest, AllKernelsOnAngledRamp) {
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  double h = std::sqrt(2.0);
  auto ramp = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      {5, 5, 0}, {10, 5, 0}, {10, 5 + h, h}, {5, 5 + h, h}});
  AddSurfaceElement(1, 2, 200, ramp);
  AddAffordance(2, 3, 300, "Walkable");

  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor_id, "chest", RelationKernelType::kAvatarFacing);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarAffordancePose,
               "Walkable");

  space_->UpdateActorState(actor_id, Eigen::Vector2d(7, 6), 0.0);

  auto& bindings = space_->GetBindings(actor_id);
  ASSERT_EQ(bindings.size(), 3u);
  ExpectValidMatches(bindings[0], "AngledRamp: Spacing");
  ExpectValidMatches(bindings[1], "AngledRamp: Facing");
  ExpectValidMatches(bindings[2], "AngledRamp: AffordancePose");
}

// ==========================================================================
// Category 2 (extended): Mixed spatial defs with actor moving through space
// ==========================================================================

TEST_F(GeometryStressTest, MixedSpatialDefsTrajectory) {
  // Room with mixed geometry
  AddRoom(1, {{0, 0}, {20, 0}, {20, 20}, {0, 20}});

  // Floor-based object (Navigable)
  AddNavigableElement(1, 2, 200, 5, 5,
                      {{3, 3}, {7, 3}, {7, 7}, {3, 7}});

  // Whiteboard on wall (Surface)
  auto whiteboard = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      {8, 19, 0.5}, {14, 19, 0.5}, {14, 19, 2.5}, {8, 19, 2.5}});
  AddSurfaceElement(1, 3, 300, whiteboard);

  // Tabletop surface (Surface)
  auto tabletop = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      {14, 3, 0.8}, {18, 3, 0.8}, {18, 7, 0.8}, {14, 7, 0.8}});
  AddSurfaceElement(1, 4, 400, tabletop);

  auto actor_id = CreateActorInRoom(1);
  space_->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
  space_->Bind(actor_id, "chest", RelationKernelType::kAvatarFacing);

  // Walk actor across the room in 10 steps.
  // When actor enters an object's NavigableSpatialDef boundary, Spacing scope
  // narrows to that object (expected behavior). We verify no crashes and that
  // all produced results are finite.
  Eigen::Vector2d start(2, 2);
  Eigen::Vector2d goal(18, 18);
  for (int i = 0; i <= 10; ++i) {
    double t = static_cast<double>(i) / 10.0;
    Eigen::Vector2d pos = start + t * (goal - start);
    double yaw = std::atan2(goal.y() - pos.y(), goal.x() - pos.x());

    space_->UpdateActorState(actor_id, pos, yaw);

    auto& bindings = space_->GetBindings(actor_id);
    ASSERT_EQ(bindings.size(), 2u) << "Step " << i;

    // Facing always sees all descendants — should always have matches.
    EXPECT_FALSE(bindings[1].latest_matches.empty())
        << "Step " << i << ": Facing should always have matches";

    // All results should be finite
    for (const auto& match : bindings[0].latest_matches) {
      EXPECT_TRUE(std::isfinite(match.eval.grounded.proximity))
          << "Step " << i;
    }
    for (const auto& match : bindings[1].latest_matches) {
      EXPECT_TRUE(std::isfinite(match.eval.grounded.proximity))
          << "Step " << i;
    }
  }
}

// ==========================================================================
// Category 5 (extended): Extreme yaw values
// ==========================================================================

TEST_F(BoundaryPositionTest, ExtremeYawValues) {
  // Test with various extreme yaw values
  std::vector<double> yaws = {
      0.0, M_PI, -M_PI, 2 * M_PI, -2 * M_PI,
      100 * M_PI,  // Large positive
      -100 * M_PI, // Large negative
      M_PI / 2.0, -M_PI / 2.0,
  };

  for (double yaw : yaws) {
    SCOPED_TRACE("yaw=" + std::to_string(yaw));
    space_->UpdateActorState(actor_id_, Eigen::Vector2d(10, 10), yaw);
    auto& bindings = space_->GetBindings(actor_id_);
    for (const auto& match : bindings[0].latest_matches) {
      EXPECT_TRUE(std::isfinite(match.eval.grounded.proximity));
      for (int i = 0; i < match.eval.grounded.encoded_state.size(); ++i) {
        EXPECT_TRUE(std::isfinite(match.eval.grounded.encoded_state[i]));
      }
    }
  }
}

}  // namespace test
}  // namespace spatialblend
