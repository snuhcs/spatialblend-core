// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/geometry/polygon3d.h"

#include <gtest/gtest.h>

namespace spatialblend {
namespace test {

std::vector<Eigen::Vector3d> MakeFlatSquare() {
  return {{0, 0, 1}, {2, 0, 1}, {2, 2, 1}, {0, 2, 1}};
}

TEST(Polygon3DTest, ConstructAndGetVertices) {
  Polygon3D poly(MakeFlatSquare());
  EXPECT_EQ(poly.GetVertices().size(), 4);
}

TEST(Polygon3DTest, NormalPointsUp) {
  Polygon3D poly(MakeFlatSquare());
  auto normal = poly.GetNormal();
  EXPECT_NEAR(normal.x(), 0.0, 1e-9);
  EXPECT_NEAR(normal.y(), 0.0, 1e-9);
  EXPECT_NEAR(normal.z(), 1.0, 1e-9);
}

TEST(Polygon3DTest, GetCenter) {
  Polygon3D poly(MakeFlatSquare());
  auto center = poly.GetCenter();
  EXPECT_NEAR(center.x(), 1.0, 1e-9);
  EXPECT_NEAR(center.y(), 1.0, 1e-9);
  EXPECT_NEAR(center.z(), 1.0, 1e-9);
}

TEST(Polygon3DTest, GetLocalFrame) {
  Polygon3D poly(MakeFlatSquare());
  auto frame = poly.GetLocalFrame();
  EXPECT_NEAR(frame(0, 3), 1.0, 1e-9);
  EXPECT_NEAR(frame(1, 3), 1.0, 1e-9);
  EXPECT_NEAR(frame(2, 3), 1.0, 1e-9);
  EXPECT_NEAR(frame(2, 2), 1.0, 1e-9);
}

TEST(Polygon3DTest, ProjectToLocal2D) {
  Polygon3D poly(MakeFlatSquare());
  auto local = poly.ProjectToLocal2D(Eigen::Vector3d(1, 1, 1));
  EXPECT_NEAR(local.x(), 0.0, 1e-9);
  EXPECT_NEAR(local.y(), 0.0, 1e-9);
}

TEST(Polygon3DTest, ProjectCornerToLocal2D) {
  Polygon3D poly(MakeFlatSquare());
  auto local = poly.ProjectToLocal2D(Eigen::Vector3d(2, 2, 1));
  EXPECT_NEAR(local.x(), 1.0, 1e-9);
  EXPECT_NEAR(local.y(), 1.0, 1e-9);
}

TEST(Polygon3DTest, Get2DBoundary) {
  Polygon3D poly(MakeFlatSquare());
  auto boundary = poly.Get2DBoundary();
  ASSERT_NE(boundary, nullptr);
  EXPECT_EQ(boundary->GetPoints().size(), 4);
}

TEST(Polygon3DTest, VerticalSurface) {
  std::vector<Eigen::Vector3d> verts = {
      {0, 0, 0}, {2, 0, 0}, {2, 0, 2}, {0, 0, 2}};
  Polygon3D poly(verts);
  auto normal = poly.GetNormal();
  EXPECT_NEAR(std::abs(normal.y()), 1.0, 1e-9);
  EXPECT_NEAR(normal.x(), 0.0, 1e-9);
  EXPECT_NEAR(normal.z(), 0.0, 1e-9);
}

}  // namespace test
}  // namespace spatialblend
