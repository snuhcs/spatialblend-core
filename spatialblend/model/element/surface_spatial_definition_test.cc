// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/surface_spatial_definition.h"

#include <cmath>

#include <gtest/gtest.h>

namespace spatialblend {
namespace test {

std::shared_ptr<Polygon3D> MakeTabletop() {
  return std::make_shared<Polygon3D>(
      std::vector<Eigen::Vector3d>{{0, 0, 1}, {2, 0, 1}, {2, 2, 1}, {0, 2, 1}});
}

static std::shared_ptr<Polygon3D> MakeTranslatedRotatedSurface() {
  Eigen::Matrix4d xform = Eigen::Matrix4d::Identity();
  const double yaw = M_PI / 2.0;
  xform(0, 0) = std::cos(yaw);
  xform(0, 1) = -std::sin(yaw);
  xform(1, 0) = std::sin(yaw);
  xform(1, 1) = std::cos(yaw);
  xform(0, 3) = 10.0;
  xform(1, 3) = -3.0;
  const double z = 0.8;

  std::vector<Eigen::Vector3d> points = {
      {0, 0, z}, {2, 0, z}, {2, 1, z}, {0, 1, z}};
  for (auto& point : points) {
    Eigen::Vector4d p(point.x(), point.y(), point.z(), 1.0);
    p = xform * p;
    point = p.head<3>();
  }
  return std::make_shared<Polygon3D>(points);
}

// The grounding rule asks whether a definition is a surface on every evaluation of every relation. That is a virtual tag, not
// a dynamic_cast (a cast across a shared-library boundary compares type_info by name: it made the median frame of the per-frame budget
// benchmark 1.4x of the commit before it).
TEST(SurfaceSpatialDefinitionTest, IsTaggedAsASurfaceDefinition) {
  const SurfaceSpatialDefinition surface(MakeTabletop());
  const SpatialDefinition& base = surface;
  EXPECT_TRUE(base.IsSurfaceDefinition());
}

TEST(SurfaceSpatialDefinitionTest, GetBoundaryReturns2DProjection) {
  SurfaceSpatialDefinition def(MakeTabletop());
  auto boundary = def.GetBoundary();
  ASSERT_NE(boundary, nullptr);
  EXPECT_EQ(boundary->GetPoints().size(), 4);
}

TEST(SurfaceSpatialDefinitionTest, ContainsPointAboveSurface) {
  SurfaceSpatialDefinition def(MakeTabletop());
  EXPECT_TRUE(def.Contains(Eigen::Vector2d(1, 1)));
}

TEST(SurfaceSpatialDefinitionTest, ContainsPointOutside) {
  SurfaceSpatialDefinition def(MakeTabletop());
  EXPECT_FALSE(def.Contains(Eigen::Vector2d(5, 5)));
}

TEST(SurfaceSpatialDefinitionTest, GetBounds) {
  SurfaceSpatialDefinition def(MakeTabletop());
  auto bounds = def.GetBounds();
  EXPECT_LE(bounds.min.x(), bounds.max.x());
  EXPECT_LE(bounds.min.y(), bounds.max.y());
}

TEST(SurfaceSpatialDefinitionTest, LocalTransformDefault) {
  SurfaceSpatialDefinition def(MakeTabletop());
  EXPECT_TRUE(def.GetLocalTransform().isIdentity());
}

TEST(SurfaceSpatialDefinitionTest, GetSurface) {
  auto surface = MakeTabletop();
  SurfaceSpatialDefinition def(surface);
  EXPECT_EQ(def.GetSurface().get(), surface.get());
}

TEST(SurfaceSpatialDefinitionTest, GetSurfaceLocalFrame) {
  SurfaceSpatialDefinition def(MakeTabletop());
  auto frame = def.GetSurfaceLocalFrame();
  EXPECT_NEAR(frame(0, 3), 1.0, 1e-9);
  EXPECT_NEAR(frame(1, 3), 1.0, 1e-9);
  EXPECT_NEAR(frame(2, 3), 1.0, 1e-9);
}

// --- Geometric query tests ---

TEST(SurfaceSpatialDefinitionTest, ProjectOntoSurface) {
  SurfaceSpatialDefinition def(MakeTabletop());
  // Point above the tabletop center → projected onto the surface plane
  Eigen::Vector3d above(1, 1, 5);
  auto projected = def.ProjectOnto(above);
  // Should be on the surface (z≈1 for the tabletop at z=1)
  EXPECT_NEAR(projected.z(), 1.0, 1e-6);
  EXPECT_NEAR(projected.x(), 1.0, 1e-6);
  EXPECT_NEAR(projected.y(), 1.0, 1e-6);
}

TEST(SurfaceSpatialDefinitionTest, ProjectOntoClampsToBoundary) {
  SurfaceSpatialDefinition def(MakeTabletop());
  // Anchor outside X on the +x side: projection should clamp to the edge at x=2.
  Eigen::Vector3d above(3.0, 1.0, 5.0);
  auto projected = def.ProjectOnto(above);

  EXPECT_NEAR(projected.x(), 2.0, 1e-6);
  EXPECT_NEAR(projected.y(), 1.0, 1e-6);
  EXPECT_NEAR(projected.z(), 1.0, 1e-6);
}

TEST(SurfaceSpatialDefinitionTest, ProjectOntoTranslatedRotatedSurface) {
  SurfaceSpatialDefinition def(MakeTranslatedRotatedSurface());
  // Surface is translated to (x≈[9,10], y≈[-3,-1]) in world coords and rotated
  // 90° about Z.
  Eigen::Vector3d above(10.0, -2.0, 2.5);
  auto projected = def.ProjectOnto(above);

  EXPECT_NEAR(projected.x(), 10.0, 1e-6);
  EXPECT_NEAR(projected.y(), -2.0, 1e-6);
  EXPECT_NEAR(projected.z(), 0.8, 1e-6);
}

TEST(SurfaceSpatialDefinitionTest, GetNormalPointsUp) {
  SurfaceSpatialDefinition def(MakeTabletop());
  auto normal = def.GetNormal();
  // Tabletop on XY plane → normal is +Z
  EXPECT_NEAR(normal.z(), 1.0, 1e-6);
}

TEST(SurfaceSpatialDefinitionTest, GetDistanceFromAbove) {
  SurfaceSpatialDefinition def(MakeTabletop());
  // Point 2m above tabletop center
  double dist = def.GetDistance(Eigen::Vector3d(1, 1, 3));
  EXPECT_NEAR(dist, 2.0, 1e-6);
}

TEST(SurfaceSpatialDefinitionTest, GetDistanceOnSurface) {
  SurfaceSpatialDefinition def(MakeTabletop());
  // Point on the surface
  double dist = def.GetDistance(Eigen::Vector3d(1, 1, 1));
  EXPECT_NEAR(dist, 0.0, 1e-6);
}

TEST(SurfaceSpatialDefinitionTest, GetTransformMatchesSurfaceFrame) {
  SurfaceSpatialDefinition def(MakeTabletop());
  auto transform = def.GetTransform();
  // Should match the surface's local frame
  EXPECT_NEAR(transform(0, 3), 1.0, 1e-9);
  EXPECT_NEAR(transform(1, 3), 1.0, 1e-9);
  EXPECT_NEAR(transform(2, 3), 1.0, 1e-9);
}

}  // namespace test
}  // namespace spatialblend
