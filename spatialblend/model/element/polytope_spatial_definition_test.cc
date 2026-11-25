// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/polytope_spatial_definition.h"

#include <gtest/gtest.h>

#include <vector>

namespace spatialblend {
namespace {

void ExpectVectorNear(const Eigen::Vector3d& actual,
                      const Eigen::Vector3d& expected,
                      double tolerance = 1e-9) {
  EXPECT_NEAR(actual.x(), expected.x(), tolerance);
  EXPECT_NEAR(actual.y(), expected.y(), tolerance);
  EXPECT_NEAR(actual.z(), expected.z(), tolerance);
}

void ExpectVectorNear(const Eigen::Vector2d& actual,
                      const Eigen::Vector2d& expected,
                      double tolerance = 1e-9) {
  EXPECT_NEAR(actual.x(), expected.x(), tolerance);
  EXPECT_NEAR(actual.y(), expected.y(), tolerance);
}

// An OBSTACLE role now constrains root feasibility and paths,
// so the default is kUnspecified -- a box is an obstacle only when declared.
TEST(PolytopeSpatialDefinitionTest, BoxDefaultsToUnspecifiedRole) {
  PolytopeSpatialDefinition def({0, 0, 0}, {2, 3, 1});

  EXPECT_EQ(def.GetGeometryRole(), GeometryRole::kUnspecified);
  EXPECT_EQ(def.GetVertices().size(), 8u);
  EXPECT_EQ(def.GetFaceIndices().size(), 6u);
  ExpectVectorNear(def.GetMinCorner(), Eigen::Vector3d(0, 0, 0));
  ExpectVectorNear(def.GetMaxCorner(), Eigen::Vector3d(2, 3, 1));
  ExpectVectorNear(def.GetCenter(), Eigen::Vector3d(1, 1.5, 0.5));
}

TEST(PolytopeSpatialDefinitionTest, BoxContains2DFootprintAnd3DVolume) {
  PolytopeSpatialDefinition def({0, 0, 0}, {2, 2, 1},
                                GeometryRole::kContainment);

  EXPECT_EQ(def.GetGeometryRole(), GeometryRole::kContainment);
  EXPECT_TRUE(def.Contains(Eigen::Vector2d(1, 1)));
  EXPECT_FALSE(def.Contains(Eigen::Vector2d(3, 1)));
  EXPECT_TRUE(def.Contains3D(Eigen::Vector3d(1, 1, 0.5)));
  EXPECT_TRUE(def.Contains3D(Eigen::Vector3d(2, 2, 1)));
  EXPECT_FALSE(def.Contains3D(Eigen::Vector3d(1, 1, 2)));
}

TEST(PolytopeSpatialDefinitionTest, BoxCanonicalizesReversedCorners) {
  PolytopeSpatialDefinition def({2, 3, 4}, {-1, -2, -3});

  ExpectVectorNear(def.GetMinCorner(), Eigen::Vector3d(-1, -2, -3));
  ExpectVectorNear(def.GetMaxCorner(), Eigen::Vector3d(2, 3, 4));
  EXPECT_TRUE(def.Contains(Eigen::Vector2d(0, 0)));
  EXPECT_TRUE(def.Contains3D(Eigen::Vector3d(0, 0, 0)));
  EXPECT_FALSE(def.Contains3D(Eigen::Vector3d(0, 0, 5)));
}

TEST(PolytopeSpatialDefinitionTest, BoxIncludesBoundaryWithinTolerance) {
  PolytopeSpatialDefinition def({0, 0, 0}, {2, 2, 2});

  EXPECT_TRUE(def.Contains(Eigen::Vector2d(2, 1)));
  EXPECT_TRUE(def.Contains3D(Eigen::Vector3d(2, 1, 1)));
  EXPECT_TRUE(def.Contains3D(Eigen::Vector3d(2 + 5e-10, 1, 1)));
  EXPECT_FALSE(def.Contains3D(Eigen::Vector3d(2 + 1e-6, 1, 1)));
}

TEST(PolytopeSpatialDefinitionTest, BoxBoundaryAndBoundsUseXyFootprint) {
  PolytopeSpatialDefinition def({0, 0, 0}, {2, 3, 1});

  auto boundary = def.GetBoundary();
  ASSERT_NE(boundary, nullptr);
  EXPECT_EQ(boundary->GetPoints().size(), 4u);
  EXPECT_TRUE(boundary->Contains(Eigen::Vector2d(1, 1)));

  Bound2D bounds = def.GetBounds();
  ExpectVectorNear(bounds.min, Eigen::Vector2d(0, 0));
  ExpectVectorNear(bounds.max, Eigen::Vector2d(2, 3));
}

TEST(PolytopeSpatialDefinitionTest, ClosestPointOutsideClampsToSurface) {
  PolytopeSpatialDefinition def({0, 0, 0}, {2, 2, 2});

  ExpectVectorNear(def.GetClosestPoint(Eigen::Vector3d(3, 1, 1)),
                   Eigen::Vector3d(2, 1, 1));
  ExpectVectorNear(def.GetClosestPoint(Eigen::Vector3d(3, 3, 3)),
                   Eigen::Vector3d(2, 2, 2));
  ExpectVectorNear(def.ProjectOnto(Eigen::Vector3d(3, 1, 1)),
                   Eigen::Vector3d(2, 1, 1));
  EXPECT_NEAR(def.GetDistance(Eigen::Vector3d(3, 1, 1)), 1.0, 1e-9);
}

TEST(PolytopeSpatialDefinitionTest, ClosestPointInsideProjectsToNearestFace) {
  PolytopeSpatialDefinition def({0, 0, 0}, {2, 2, 2});

  ExpectVectorNear(def.GetClosestPoint(Eigen::Vector3d(1, 1, 1.7)),
                   Eigen::Vector3d(1, 1, 2));
  EXPECT_NEAR(def.GetDistance(Eigen::Vector3d(1, 1, 1.7)), 0.3, 1e-9);
  ExpectVectorNear(def.GetNormal(Eigen::Vector3d(1, 1, 1.9)),
                   Eigen::Vector3d::UnitZ());
}

TEST(PolytopeSpatialDefinitionTest, GenericTetrahedronIsSupported) {
  std::vector<Eigen::Vector3d> vertices = {
      {0, 0, 0},
      {2, 0, 0},
      {0, 2, 0},
      {0, 0, 2},
  };
  std::vector<std::vector<int>> faces = {
      {0, 2, 1},
      {0, 1, 3},
      {0, 3, 2},
      {1, 2, 3},
  };
  PolytopeSpatialDefinition def(vertices, faces,
                                GeometryRole::kInteractionRegion);

  EXPECT_EQ(def.GetGeometryRole(), GeometryRole::kInteractionRegion);
  EXPECT_TRUE(def.Contains(Eigen::Vector2d(0.25, 0.25)));
  EXPECT_TRUE(def.Contains3D(Eigen::Vector3d(0.25, 0.25, 0.25)));
  EXPECT_FALSE(def.Contains3D(Eigen::Vector3d(1.0, 1.0, 1.0)));
  ExpectVectorNear(def.GetMinCorner(), Eigen::Vector3d(0, 0, 0));
  ExpectVectorNear(def.GetMaxCorner(), Eigen::Vector3d(2, 2, 2));
  EXPECT_GE(def.GetDistance(Eigen::Vector3d(3, 3, 3)), 0.0);
}

TEST(PolytopeSpatialDefinitionTest, EmptyGeometryIsStable) {
  PolytopeSpatialDefinition def(std::vector<Eigen::Vector3d>{},
                                std::vector<std::vector<int>>{},
                                GeometryRole::kObstacle);
  Eigen::Vector3d point(1, 2, 3);

  EXPECT_EQ(def.GetVertices().size(), 0u);
  ASSERT_NE(def.GetBoundary(), nullptr);
  EXPECT_EQ(def.GetBoundary()->GetPoints().size(), 0u);
  EXPECT_FALSE(def.Contains(Eigen::Vector2d(1, 2)));
  EXPECT_FALSE(def.Contains3D(point));
  ExpectVectorNear(def.GetClosestPoint(point), point);
  EXPECT_DOUBLE_EQ(def.GetDistance(point), 0.0);
  ExpectVectorNear(def.GetNormal(point), Eigen::Vector3d::UnitZ());
}

TEST(PolytopeSpatialDefinitionTest, InvalidFacesDoNotCrashQueries) {
  std::vector<Eigen::Vector3d> vertices = {
      {0, 0, 0},
      {2, 0, 0},
      {0, 2, 0},
  };
  std::vector<std::vector<int>> invalid_faces = {
      {0, 1},
      {0, 1, 9},
      {-1, 1, 2},
  };
  PolytopeSpatialDefinition def(vertices, invalid_faces);

  EXPECT_FALSE(def.Contains3D(Eigen::Vector3d(0.25, 0.25, 0.0)));
  ExpectVectorNear(def.GetClosestPoint(Eigen::Vector3d(3, 0, 0)),
                   Eigen::Vector3d(2, 0, 0));
  EXPECT_NEAR(def.GetDistance(Eigen::Vector3d(3, 0, 0)), 1.0, 1e-9);
  ExpectVectorNear(def.GetNormal(Eigen::Vector3d(3, 0, 0)),
                   Eigen::Vector3d::UnitZ());
}

TEST(PolytopeSpatialDefinitionTest, FlatBoxBehavesAsZeroThicknessVolume) {
  PolytopeSpatialDefinition def({0, 0, 1}, {2, 2, 1});

  EXPECT_TRUE(def.Contains(Eigen::Vector2d(1, 1)));
  EXPECT_TRUE(def.Contains3D(Eigen::Vector3d(1, 1, 1)));
  EXPECT_FALSE(def.Contains3D(Eigen::Vector3d(1, 1, 1.1)));
  ExpectVectorNear(def.GetClosestPoint(Eigen::Vector3d(1, 1, 3)),
                   Eigen::Vector3d(1, 1, 1));
  EXPECT_NEAR(def.GetDistance(Eigen::Vector3d(1, 1, 3)), 2.0, 1e-9);
}

}  // namespace
}  // namespace spatialblend
