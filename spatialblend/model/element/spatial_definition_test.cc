// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/navigable_spatial_definition.h"

#include <gtest/gtest.h>

namespace spatialblend {
namespace test {

std::shared_ptr<spatialblend::Polygon2D> MakeSquare() {
  return std::make_shared<spatialblend::Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {10, 0}, {10, 10}, {0, 10}});
}

TEST(NavigableSpatialDefinitionTest, ContainsPointInside) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  EXPECT_TRUE(nav.Contains(Eigen::Vector2d(5, 5)));
}

TEST(NavigableSpatialDefinitionTest, ContainsPointOutside) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  EXPECT_FALSE(nav.Contains(Eigen::Vector2d(15, 5)));
}

TEST(NavigableSpatialDefinitionTest, IsNotASurfaceDefinition) {
  NavigableSpatialDefinition nav(MakeSquare());
  const SpatialDefinition& base = nav;
  EXPECT_FALSE(base.IsSurfaceDefinition());
}

TEST(NavigableSpatialDefinitionTest, GetBoundaryReturnsWalkable) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  EXPECT_EQ(nav.GetBoundary(), walkable);
}

TEST(NavigableSpatialDefinitionTest, GetBoundsCorrect) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  auto bounds = nav.GetBounds();
  EXPECT_DOUBLE_EQ(bounds.min.x(), 0.0);
  EXPECT_DOUBLE_EQ(bounds.min.y(), 0.0);
  EXPECT_DOUBLE_EQ(bounds.max.x(), 10.0);
  EXPECT_DOUBLE_EQ(bounds.max.y(), 10.0);
}

TEST(NavigableSpatialDefinitionTest, LocalTransformDefaultsToIdentity) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  EXPECT_TRUE(nav.GetLocalTransform().isIdentity());
}

TEST(NavigableSpatialDefinitionTest, GetWalkablePolygon) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  EXPECT_EQ(nav.GetWalkablePolygon(), walkable);
}

TEST(NavigableSpatialDefinitionTest, SetWall) {
  auto walkable = MakeSquare();
  auto wall = std::make_shared<spatialblend::Polygon2D>(
      std::vector<Eigen::Vector2d>{{3, 3}, {7, 3}, {7, 7}, {3, 7}});
  NavigableSpatialDefinition nav(walkable, wall);
  EXPECT_NE(nav.GetWall(), nullptr);
}

// --- Geometric query tests for NavigableSpatialDefinition ---

TEST(NavigableSpatialDefinitionTest, GetClosestPointInside) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  // Point inside → closest point is itself (on the boundary)
  auto closest = nav.GetClosestPoint(Eigen::Vector3d(5, 5, 0));
  // GetNearestPoint returns boundary point, but 5,5 is inside
  // so GetClosestPoint should return a boundary point
  EXPECT_GE(closest.x(), 0.0);
  EXPECT_LE(closest.x(), 10.0);
}

TEST(NavigableSpatialDefinitionTest, GetClosestPointOutside) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  // Point outside (15, 5) → closest should be on boundary (10, 5)
  auto closest = nav.GetClosestPoint(Eigen::Vector3d(15, 5, 0));
  EXPECT_NEAR(closest.x(), 10.0, 0.5);
  EXPECT_NEAR(closest.y(), 5.0, 0.5);
}

TEST(NavigableSpatialDefinitionTest, GetDistanceOutside) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  double dist = nav.GetDistance(Eigen::Vector3d(15, 5, 0));
  EXPECT_NEAR(dist, 5.0, 0.5);
}

TEST(NavigableSpatialDefinitionTest, ProjectOntoClamps) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  // Point outside → projects to nearest boundary point
  auto projected = nav.ProjectOnto(Eigen::Vector3d(15, 5, 0));
  EXPECT_NEAR(projected.x(), 10.0, 0.5);
  EXPECT_NEAR(projected.y(), 5.0, 0.5);
}

TEST(NavigableSpatialDefinitionTest, ProjectOntoInsideUnchanged) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  // Point inside → unchanged
  auto projected = nav.ProjectOnto(Eigen::Vector3d(5, 5, 2));
  EXPECT_NEAR(projected.x(), 5.0, 1e-9);
  EXPECT_NEAR(projected.y(), 5.0, 1e-9);
  EXPECT_NEAR(projected.z(), 2.0, 1e-9);
}

TEST(NavigableSpatialDefinitionTest, GeodesicDistanceFallsBackToEuclidean) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  double dist = nav.GetGeodesicDistance(
      Eigen::Vector3d(2, 2, 0), Eigen::Vector3d(8, 8, 0));
  // Geodesic via pathfinder (or fallback to Euclidean for convex polygon)
  EXPECT_NEAR(dist, std::sqrt(72.0), 0.1);
}

TEST(NavigableSpatialDefinitionTest, GetNormalIsZUp) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  auto normal = nav.GetNormal();
  EXPECT_NEAR(normal.x(), 0.0, 1e-9);
  EXPECT_NEAR(normal.y(), 0.0, 1e-9);
  EXPECT_NEAR(normal.z(), 1.0, 1e-9);
}

TEST(NavigableSpatialDefinitionTest, GetTransformDefault) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  auto transform = nav.GetTransform();
  EXPECT_TRUE(transform.isIdentity());
}

TEST(NavigableSpatialDefinitionTest, Contains3DDelegatesto2D) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  // 3D point inside the 2D walkable area (z is ignored)
  EXPECT_TRUE(nav.Contains3D(Eigen::Vector3d(5, 5, 100)));
  EXPECT_FALSE(nav.Contains3D(Eigen::Vector3d(15, 5, 0)));
}

TEST(NavigableSpatialDefinitionTest, GetDistanceInside) {
  auto walkable = MakeSquare();
  NavigableSpatialDefinition nav(walkable);
  // Point inside — distance to boundary
  double dist = nav.GetDistance(Eigen::Vector3d(5, 5, 0));
  EXPECT_GE(dist, 0.0);
}

// --- Canonical interior chart (type-agnostic correspondence) ---------------

// Rectangle centered at `center` with full extents (w, h).
static std::shared_ptr<spatialblend::Polygon2D> MakeRect(
    double w, double h, Eigen::Vector2d center = {0, 0}) {
  double hw = w / 2, hh = h / 2;
  return std::make_shared<spatialblend::Polygon2D>(std::vector<Eigen::Vector2d>{
      center + Eigen::Vector2d(-hw, -hh), center + Eigen::Vector2d(hw, -hh),
      center + Eigen::Vector2d(hw, hh), center + Eigen::Vector2d(-hw, hh)});
}

// The universal box-UVW chart transfers interior position EXACTLY between two
// affine-similar (different aspect ratio) definitions — the property the
// experiment showed radial cannot achieve. This is the default chart, so it
// works for every spatial-definition type with no per-type code.
TEST(InteriorChartTest, BoxUVWExactAcrossAspectRatios) {
  NavigableSpatialDefinition wide(MakeRect(4.0, 1.0));    // wide, shallow
  NavigableSpatialDefinition narrow(MakeRect(1.5, 2.5));  // narrow, deep

  // Spot at box-fraction (u,v)=(0.5,-0.4) on the wide desk -> world (1.0,-0.2).
  Eigen::Vector3d p(1.0, -0.2, 0.0);
  Eigen::Vector3d mapped = MapInterior(wide, narrow, p);

  // Ground truth = same box fraction on the narrow desk: (0.5*0.75, -0.4*1.25).
  EXPECT_NEAR(mapped.x(), 0.375, 1e-9);
  EXPECT_NEAR(mapped.y(), -0.500, 1e-9);
}

// The chart is defined in each definition's LOCAL frame, so a translated target
// still receives the correct local-relative interior position.
TEST(InteriorChartTest, RespectsTargetLocalFrame) {
  NavigableSpatialDefinition src(MakeRect(2.0, 2.0));
  NavigableSpatialDefinition dst(MakeRect(2.0, 2.0, {10.0, 5.0}));  // shifted

  Eigen::Vector3d p(0.5, 0.5, 0.0);  // upper-right quadrant of src
  Eigen::Vector3d mapped = MapInterior(src, dst, p);

  // Same local quadrant on dst, whose boundary is centered at (10,5).
  EXPECT_NEAR(mapped.x(), 10.5, 1e-9);
  EXPECT_NEAR(mapped.y(), 5.5, 1e-9);
}

// Round-trip through the canonical coordinate recovers the original point.
TEST(InteriorChartTest, EncodeDecodeRoundTrip) {
  NavigableSpatialDefinition def(MakeRect(3.0, 2.0));
  Eigen::Vector3d p(0.9, -0.4, 0.0);
  NormalizedInterior c = def.EncodeInterior(p);
  EXPECT_LE(std::abs(c.uvw.x()), 1.0 + 1e-9);
  EXPECT_LE(std::abs(c.uvw.y()), 1.0 + 1e-9);
  Eigen::Vector3d back = def.DecodeInterior(c);
  EXPECT_NEAR(back.x(), p.x(), 1e-9);
  EXPECT_NEAR(back.y(), p.y(), 1e-9);
}

// Non-convex L-shape: fill ratio < threshold → the chart emits the rich MVC
// channel (and still fills uvw for cross-kind partners).
static std::shared_ptr<spatialblend::Polygon2D> MakeLShape() {
  return std::make_shared<spatialblend::Polygon2D>(std::vector<Eigen::Vector2d>{
      {0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2}});
}

TEST(InteriorChartTest, BoxLikeStaysBoxUVW) {
  NavigableSpatialDefinition rect(MakeRect(4.0, 2.0));
  NormalizedInterior c = rect.EncodeInterior(Eigen::Vector3d(1.0, 0.5, 0.0));
  EXPECT_EQ(c.kind, NormalizedInterior::Kind::kBoxUVW);
  EXPECT_TRUE(c.weights.empty());
}

TEST(InteriorChartTest, NonConvexEmitsMvcAndFillsUvw) {
  NavigableSpatialDefinition lshape(MakeLShape());
  NormalizedInterior c = lshape.EncodeInterior(Eigen::Vector3d(0.5, 0.5, 0.0));
  EXPECT_EQ(c.kind, NormalizedInterior::Kind::kMVC);
  EXPECT_FALSE(c.weights.empty());
  // Universal fallback still populated.
  EXPECT_LE(std::abs(c.uvw.x()), 1.0 + 1e-9);
  EXPECT_LE(std::abs(c.uvw.y()), 1.0 + 1e-9);
}

TEST(InteriorChartTest, MvcRoundTripOnSameShape) {
  NavigableSpatialDefinition lshape(MakeLShape());
  Eigen::Vector3d p(0.5, 0.5, 0.0);
  Eigen::Vector3d back = lshape.DecodeInterior(lshape.EncodeInterior(p));
  EXPECT_NEAR(back.x(), p.x(), 1e-6);
  EXPECT_NEAR(back.y(), p.y(), 1e-6);
}

// Decoding an MVC coordinate onto a non-convex target always lands INSIDE it
// (the containment clamp catches convex-combination points in a concavity).
TEST(InteriorChartTest, MvcDecodeStaysInsideNonConvexTarget) {
  NavigableSpatialDefinition src(MakeLShape());
  // A different-sized L (scaled 2x) as the target.
  auto big = std::make_shared<spatialblend::Polygon2D>(std::vector<Eigen::Vector2d>{
      {0, 0}, {4, 0}, {4, 2}, {2, 2}, {2, 4}, {0, 4}});
  NavigableSpatialDefinition dst(big);
  // Sweep several interior points of src; every decode must be inside dst.
  for (Eigen::Vector2d q : std::vector<Eigen::Vector2d>{
           {0.5, 0.5}, {1.5, 0.5}, {0.5, 1.5}, {0.2, 0.2}, {0.9, 0.9}}) {
    Eigen::Vector3d mapped =
        MapInterior(src, dst, Eigen::Vector3d(q.x(), q.y(), 0.0));
    EXPECT_TRUE(dst.Contains(mapped.head<2>()))
        << "decoded point left the target: " << mapped.transpose();
  }
}

// --- Sub-region storage ---

TEST(SpatialDefinitionSubRegionTest, AddAndLookupSubRegion) {
  auto boundary = std::make_shared<spatialblend::Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {1, 0}, {1, 1}, {0, 1}});
  NavigableSpatialDefinition def(boundary);

  SubRegion seat;
  seat.name = "seat_patch";
  seat.tag = "Sittable";
  seat.boundary = std::make_shared<spatialblend::Polygon2D>(
      std::vector<Eigen::Vector2d>{
          {0.2, 0.2}, {0.8, 0.2}, {0.8, 0.8}, {0.2, 0.8}});
  def.AddSubRegion(seat);

  ASSERT_EQ(def.GetSubRegions().size(), 1u);
  const auto* found = def.GetSubRegion("seat_patch");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->tag, "Sittable");
  EXPECT_EQ(def.GetSubRegion("nope"), nullptr);
}

}  // namespace test
}  // namespace spatialblend
