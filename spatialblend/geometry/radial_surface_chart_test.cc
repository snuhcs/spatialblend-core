// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/geometry/radial_surface_chart.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace spatialblend {
namespace {

constexpr double kTol = 1e-9;

Polygon2D LShape(bool clockwise = false,
                 Eigen::Vector2d offset = Eigen::Vector2d::Zero()) {
  std::vector<Eigen::Vector2d> points = {{0, 0}, {3, 0}, {3, 1},
                                         {1, 1}, {1, 3}, {0, 3}};
  for (auto& point : points) point += offset;
  if (clockwise) std::reverse(points.begin(), points.end());
  return Polygon2D(points);
}

void ExpectPoint(const Eigen::Vector2d& actual,
                 const Eigen::Vector2d& expected) {
  EXPECT_NEAR(actual.x(), expected.x(), kTol);
  EXPECT_NEAR(actual.y(), expected.y(), kTol);
}

TEST(RadialSurfaceChartTest, LShapeKernelOriginAndIndependentEncodeValues) {
  const Polygon2D polygon = LShape();
  auto origin = RadialSurfaceChart::Origin(polygon);
  ASSERT_TRUE(origin.ok()) << origin.status();
  ExpectPoint(*origin, Eigen::Vector2d(.5, .5));

  // From (.5,.5), +X reaches x=3 at distance 2.5; (1.25,.5) is ratio .3.
  // The chart is pure radial: (.75,.5) is at distance .25 of 2.5, ratio .1.
  auto encoded = RadialSurfaceChart::Encode(Eigen::Vector2d(.75, .5), polygon,
                                            Eigen::Vector2d::UnitX());
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_EQ(encoded->strategy, NormalizedPolygonPosition::Strategy::kRadial);
  EXPECT_NEAR(encoded->angle, 0.0, kTol);
  EXPECT_NEAR(encoded->radial_ratio, .1, kTol);
}

Polygon2D Rectangle(double hx, double hy) {
  return Polygon2D({{-hx, -hy}, {hx, -hy}, {hx, hy}, {-hx, hy}});
}

// The chart is pure radial, as in v1. A point near the
// origin of a 2x2 square maps to the same ratio and angle on a 4x3 rectangle:
// (0.2, 0.05) -> (0.4, 0.1), exactly, with no core or ramp.
TEST(RadialSurfaceChartTest, PureRadialMapsSquareToRectangleExactly) {
  const Eigen::Vector2d forward = Eigen::Vector2d::UnitX();
  const Polygon2D square = Rectangle(1.0, 1.0);
  const Polygon2D rect = Rectangle(2.0, 1.5);
  auto encoded =
      RadialSurfaceChart::Encode(Eigen::Vector2d(0.2, 0.05), square, forward);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_NEAR(encoded->radial_ratio, 0.2, kTol);
  auto decoded = RadialSurfaceChart::Decode(*encoded, rect, forward);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  ExpectPoint(*decoded, Eigen::Vector2d(0.4, 0.1));
  // Points inside what used to be the core map by ratio too.
  for (double rho : {0.001, 0.01, 0.03, 0.1, 0.2, 0.3}) {
    auto e = RadialSurfaceChart::Encode(Eigen::Vector2d(rho, 0.0), square,
                                        forward);
    ASSERT_TRUE(e.ok()) << e.status();
    EXPECT_NEAR(e->radial_ratio, rho, kTol);
    auto d = RadialSurfaceChart::Decode(*e, rect, forward);
    ASSERT_TRUE(d.ok()) << d.status();
    ExpectPoint(*d, Eigen::Vector2d(2.0 * rho, 0.0));
  }
}

TEST(RadialSurfaceChartTest, RejectsTheDeletedRegularizedChartVersion) {
  NormalizedPolygonPosition packet;
  packet.strategy = NormalizedPolygonPosition::Strategy::kRadial;
  packet.radial_ratio = 0.5;
  packet.chart_version = 2;
  EXPECT_FALSE(RadialSurfaceChart::Decode(packet, Rectangle(1, 1),
                                          Eigen::Vector2d::UnitX())
                   .ok());
  packet.chart_version = 1;
  EXPECT_TRUE(RadialSurfaceChart::Decode(packet, Rectangle(1, 1),
                                         Eigen::Vector2d::UnitX())
                  .ok());
}

TEST(RadialSurfaceChartTest, DecodesLArmsConcaveCornerAndBoundaries) {
  const Polygon2D polygon = LShape();
  const Eigen::Vector2d forward = Eigen::Vector2d::UnitX();
  auto decode = [&](double angle, double ratio,
                    const Eigen::Vector2d& expected) {
    NormalizedPolygonPosition packet;
    packet.strategy = NormalizedPolygonPosition::Strategy::kRadial;
    packet.angle = angle;
    packet.radial_ratio = ratio;
    auto result = RadialSurfaceChart::Decode(packet, polygon, forward);
    ASSERT_TRUE(result.ok()) << result.status();
    ExpectPoint(*result, expected);
  };
  decode(0.0, 1.0, Eigen::Vector2d(3.0, .5));      // horizontal arm boundary
  decode(M_PI_2, .6, Eigen::Vector2d(.5, 2.0));    // vertical arm interior
  decode(M_PI_4, 1.0, Eigen::Vector2d(1.0, 1.0));  // reflex-corner boundary
  decode(M_PI_2, 1.0, Eigen::Vector2d(.5, 3.0));   // vertical arm boundary
}

TEST(RadialSurfaceChartTest, IsOrientationAndTranslationStable) {
  for (bool clockwise : {false, true}) {
    auto origin = RadialSurfaceChart::Origin(LShape(clockwise));
    ASSERT_TRUE(origin.ok()) << origin.status();
    ExpectPoint(*origin, Eigen::Vector2d(.5, .5));
  }
  const Eigen::Vector2d offset(1e10, -2e10);
  const Polygon2D translated = LShape(false, offset);
  auto origin = RadialSurfaceChart::Origin(translated);
  ASSERT_TRUE(origin.ok()) << origin.status();
  ExpectPoint(*origin, offset + Eigen::Vector2d(.5, .5));
  auto encoded = RadialSurfaceChart::Encode(
      offset + Eigen::Vector2d(.75, .5), translated, Eigen::Vector2d::UnitX());
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_NEAR(encoded->radial_ratio, .1, 1e-6);
}

TEST(RadialSurfaceChartTest, PreservesSmallNonzeroTranslatedOffsets) {
  const Eigen::Vector2d offset(1e10, -2e10);
  const Polygon2D polygon = LShape(false, offset);
  const Eigen::Vector2d forward(1e-300,
                                0.0);  // Nonzero despite tiny magnitude.
  const Eigen::Vector2d first = offset + Eigen::Vector2d(.5001, .5);
  const Eigen::Vector2d second = offset + Eigen::Vector2d(.5002, .5);
  auto a = RadialSurfaceChart::Encode(first, polygon, forward);
  auto b = RadialSurfaceChart::Encode(second, polygon, forward);
  ASSERT_TRUE(a.ok()) << a.status();
  ASSERT_TRUE(b.ok()) << b.status();
  EXPECT_GT(a->radial_ratio, 0.0);
  EXPECT_GT(b->radial_ratio, a->radial_ratio);
}

TEST(RadialSurfaceChartTest, TranslatedVertexGrazingRejectsLineExtensionHit) {
  const Eigen::Vector2d offset(1e10, -2e10);
  const Polygon2D polygon = LShape(false, offset);
  // This ray passes just above the concave corner. The y=1 horizontal edge's
  // line is encountered first at x=.9999, but that is beyond its endpoint;
  // the first actual boundary is the vertical edge x=1 at y=1.0001.
  NormalizedPolygonPosition packet;
  packet.strategy = NormalizedPolygonPosition::Strategy::kRadial;
  packet.angle = std::atan2(.5, .4999);
  packet.radial_ratio = 1.0;
  auto decoded =
      RadialSurfaceChart::Decode(packet, polygon, Eigen::Vector2d::UnitX());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_NEAR(decoded->x(), offset.x() + 1.0, 1e-5);
  EXPECT_NEAR(decoded->y(), offset.y() + 1.0001, 1e-5);
}

TEST(RadialSurfaceChartTest, AcceptsOnlyBoundedBoundaryRoundoff) {
  const Polygon2D square({{-1, -1}, {1, -1}, {1, 1}, {-1, 1}});
  auto near = RadialSurfaceChart::Encode(Eigen::Vector2d(1.0 + 1e-14, .5),
                                         square, Eigen::Vector2d::UnitX());
  EXPECT_TRUE(near.ok()) << near.status();
  EXPECT_FALSE(RadialSurfaceChart::Encode(Eigen::Vector2d(1.0 + 1e-4, .5),
                                          square, Eigen::Vector2d::UnitX())
                   .ok());
  // A diagonal residue is snapped to the actual corner only when its Euclidean
  // distance is within the contract bound; decode then returns that boundary.
  auto corner = RadialSurfaceChart::Encode(
      Eigen::Vector2d(1.0 + 1e-14, 1.0 + 1e-14), square,
      Eigen::Vector2d::UnitX());
  ASSERT_TRUE(corner.ok()) << corner.status();
  auto decoded =
      RadialSurfaceChart::Decode(*corner, square, Eigen::Vector2d::UnitX());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  ExpectPoint(*decoded, Eigen::Vector2d(1.0, 1.0));
  EXPECT_FALSE(RadialSurfaceChart::Encode(
                   Eigen::Vector2d(1.0 + 1e-4, 1.0 + 1e-4), square,
                   Eigen::Vector2d::UnitX())
                   .ok());
}

TEST(RadialSurfaceChartTest, RejectsInvalidRingsAndNoStrictKernel) {
  const Polygon2D self_intersecting({{0, 0}, {2, 2}, {0, 2}, {2, 0}});
  const Polygon2D repeated({{0, 0}, {2, 0}, {2, 0}, {0, 2}});
  // This is a real, very short final edge, not the conventional exact closing
  // vertex representation. It must not be silently normalized away.
  const Polygon2D near_closing(
      {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {0, 1e-15}});
  const Polygon2D degenerate({{0, 0}, {1, 0}, {2, 0}});
  const Polygon2D non_star_u(
      {{0, 0}, {3, 0}, {3, 3}, {2, 3}, {2, 1}, {1, 1}, {1, 3}, {0, 3}});
  const Polygon2D narrow_kernel(
      {{0, 0}, {2, 0}, {2, 1e-15}, {1e-15, 1e-15}, {1e-15, 2}, {0, 2}});
  const Eigen::Vector2d offset(1e10, -2e10);
  const Polygon2D shallow_crossed(
      {offset + Eigen::Vector2d(0, 0), offset + Eigen::Vector2d(3, 1e-4),
       offset + Eigen::Vector2d(0, 1e-4), offset + Eigen::Vector2d(3, 0),
       offset + Eigen::Vector2d(3, 2)});
  for (const Polygon2D* invalid :
       {&self_intersecting, &repeated, &near_closing, &degenerate, &non_star_u,
        &narrow_kernel, &shallow_crossed}) {
    EXPECT_FALSE(RadialSurfaceChart::Origin(*invalid).ok());
    EXPECT_FALSE(RadialSurfaceChart::Encode(Eigen::Vector2d::Zero(), *invalid,
                                            Eigen::Vector2d::UnitX())
                     .ok());
  }
  // A conventional duplicate closing vertex is normalized, unlike an interior
  // duplicate.
  const Polygon2D closed({{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}});
  EXPECT_TRUE(RadialSurfaceChart::Origin(closed).ok());

  // These nonadjacent micro edges cross at (5e-7,5e-9); the ring has nonzero
  // signed area because of its final large edge, so a global area epsilon must
  // not hide the crossing.
  const Polygon2D micro_crossed(
      {{0, 0}, {1e-6, 1e-8}, {0, 1e-8}, {1e-6, 0}, {1, 1}});
  EXPECT_FALSE(RadialSurfaceChart::Origin(micro_crossed).ok());
  // A retraced nonadjacent segment is also non-simple even though it has no
  // zero-length edge and encloses nonzero area overall.
  const Polygon2D backtracking(
      {{0, 0}, {2, 0}, {1, 0}, {2, 0}, {2, 2}, {0, 2}});
  EXPECT_FALSE(RadialSurfaceChart::Origin(backtracking).ok());
}

TEST(RadialSurfaceChartTest, RejectsMalformedRadialPacketsAndForward) {
  const Polygon2D polygon = LShape();
  NormalizedPolygonPosition packet;
  packet.strategy = NormalizedPolygonPosition::Strategy::kMVC;
  EXPECT_FALSE(
      RadialSurfaceChart::Decode(packet, polygon, Eigen::Vector2d::UnitX())
          .ok());
  packet.strategy = NormalizedPolygonPosition::Strategy::kRadial;
  packet.angle = M_PI + 1e-12;
  EXPECT_FALSE(
      RadialSurfaceChart::Decode(packet, polygon, Eigen::Vector2d::UnitX())
          .ok());
  packet.angle = 0.0;
  packet.radial_ratio = 1.0 + 1e-12;
  EXPECT_FALSE(
      RadialSurfaceChart::Decode(packet, polygon, Eigen::Vector2d::UnitX())
          .ok());
  packet.radial_ratio = -1e-12;
  EXPECT_FALSE(
      RadialSurfaceChart::Decode(packet, polygon, Eigen::Vector2d::UnitX())
          .ok());
  packet.radial_ratio = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(
      RadialSurfaceChart::Decode(packet, polygon, Eigen::Vector2d::UnitX())
          .ok());
  packet.radial_ratio = .5;
  packet.height_ratio = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(
      RadialSurfaceChart::Decode(packet, polygon, Eigen::Vector2d::UnitX())
          .ok());
  packet.height_ratio = 0.0;
  packet.mvc_weights = {1.0};
  EXPECT_FALSE(
      RadialSurfaceChart::Decode(packet, polygon, Eigen::Vector2d::UnitX())
          .ok());
  packet.mvc_weights.clear();
  packet.height_ratio = .5;
  EXPECT_FALSE(
      RadialSurfaceChart::Decode(packet, polygon, Eigen::Vector2d::UnitX())
          .ok());
  packet.height_ratio = -.5;
  EXPECT_FALSE(
      RadialSurfaceChart::Decode(packet, polygon, Eigen::Vector2d::UnitX())
          .ok());
  packet.height_ratio = -0.0;  // Signed zero remains a planar value.
  EXPECT_TRUE(
      RadialSurfaceChart::Decode(packet, polygon, Eigen::Vector2d::UnitX())
          .ok());
  packet.height_ratio = 0.0;
  packet.radial_ratio = .5;
  packet.angle = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(
      RadialSurfaceChart::Decode(packet, polygon, Eigen::Vector2d::UnitX())
          .ok());
  EXPECT_FALSE(RadialSurfaceChart::Encode(Eigen::Vector2d(.5, .5), polygon,
                                          Eigen::Vector2d::Zero())
                   .ok());
}

}  // namespace
}  // namespace spatialblend
