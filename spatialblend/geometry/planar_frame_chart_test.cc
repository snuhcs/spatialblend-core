// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/planar_frame_chart_test.cc
//
// The frame charts of a rectangular part (surface retargeting). Oracles are analytic. Board sizes are
// authored scenario geometry (1.2 x 0.9 vs 0.8 x 0.6 are both 4:3, 16:9 vs 4:3 differ in aspect).
#include "spatialblend/geometry/planar_frame_chart.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "spatialblend/geometry/radial_surface_chart.h"

namespace spatialblend {
namespace {

Polygon2D Rect(double w, double h, double ox = 0.0, double oy = 0.0) {
  return Polygon2D({{ox, oy}, {ox + w, oy}, {ox + w, oy + h}, {ox, oy + h}});
}
Polygon2D Centred(double w, double h) { return Rect(w, h, -0.5 * w, -0.5 * h); }

const SurfaceChartKind kStretch = SurfaceChartKind::kFrameStretch;
const SurfaceChartKind kContain = SurfaceChartKind::kFrameContain;

// u,v samples including the four corners, the centre, edges and points off the rectangle.
std::vector<Eigen::Vector2d> UvSamples() {
  std::vector<Eigen::Vector2d> s;
  for (double u : {0.0, 0.25, 0.5, 0.75, 1.0})
    for (double v : {0.0, 0.25, 0.5, 0.75, 1.0}) s.emplace_back(u, v);
  s.emplace_back(-0.1, 0.5);  // off the part: the chart is the affine extension
  s.emplace_back(1.2, 1.3);
  return s;
}

TEST(PlanarFrameChart, StretchPreservesUvForEveryBoardPair) {
  struct Pair {
    double w, h, w2, h2;
  } pairs[] = {
      {1.2, 0.9, 0.8, 0.6},  // same aspect, smaller
      {1.6, 0.9, 1.2, 0.9},  // 16:9 -> 4:3
      {1.2, 0.9, 1.6, 0.9},  // 4:3 -> 16:9
      {4.0, 1.0, 1.2, 1.0},  // wide -> narrow
      {0.7, 1.0, 3.0, 1.2},  // portrait flip chart -> wall board
  };
  for (const auto& p : pairs) {
    const Polygon2D a = Rect(p.w, p.h, 0.3, -0.2), b = Centred(p.w2, p.h2);
    for (const auto& uv : UvSamples()) {
      const Eigen::Vector2d src(0.3 + uv.x() * p.w, -0.2 + uv.y() * p.h);
      auto m = PlanarFrameChart::Map(src, 0.0, a, b, kStretch);
      ASSERT_TRUE(m.ok()) << m.status();
      EXPECT_NEAR((m->x() + 0.5 * p.w2) / p.w2, uv.x(), 1e-12);
      EXPECT_NEAR((m->y() + 0.5 * p.h2) / p.h2, uv.y(), 1e-12);
    }
  }
}

TEST(PlanarFrameChart, ContainIsUniformScaleCentredAndEqualsStretchForEqualAspect) {
  {  // equal aspect: contain == stretch
    const Polygon2D a = Rect(1.2, 0.9), b = Rect(0.8, 0.6, 2.0, 1.0);
    for (const auto& uv : UvSamples()) {
      const Eigen::Vector2d src(uv.x() * 1.2, uv.y() * 0.9);
      auto s = PlanarFrameChart::Map(src, 0.0, a, b, kStretch);
      auto c = PlanarFrameChart::Map(src, 0.0, a, b, kContain);
      ASSERT_TRUE(s.ok() && c.ok());
      EXPECT_NEAR((*s - *c).norm(), 0.0, 1e-12);
    }
  }
  {  // 16:9 -> 4:3: width-limited, s = 1.2/1.6; the content is 0.675 m tall, centred (0.1125 m margins top and bottom)
    const Polygon2D a = Rect(1.6, 0.9), b = Rect(1.2, 0.9);
    const double s = 1.2 / 1.6;
    auto lo = PlanarFrameChart::Map({0.0, 0.0}, 0.0, a, b, kContain);
    auto hi = PlanarFrameChart::Map({1.6, 0.9}, 0.0, a, b, kContain);
    ASSERT_TRUE(lo.ok() && hi.ok());
    EXPECT_NEAR(lo->x(), 0.0, 1e-12);
    EXPECT_NEAR(hi->x(), 1.2, 1e-12);
    EXPECT_NEAR(lo->y(), 0.45 - 0.5 * 0.9 * s, 1e-12);
    EXPECT_NEAR(hi->y(), 0.45 + 0.5 * 0.9 * s, 1e-12);
    EXPECT_NEAR(lo->y(), 0.1125, 1e-12);
  }
  {  // 4:3 -> 16:9: height-limited, pillarboxed; a drawn square stays square and metric (s == 1)
    const Polygon2D a = Rect(1.2, 0.9), b = Rect(1.6, 0.9);
    auto p0 = PlanarFrameChart::Map({0.3, 0.3}, 0.0, a, b, kContain);
    auto p1 = PlanarFrameChart::Map({0.6, 0.6}, 0.0, a, b, kContain);
    ASSERT_TRUE(p0.ok() && p1.ok());
    EXPECT_NEAR(p1->x() - p0->x(), p1->y() - p0->y(), 1e-12);
    EXPECT_NEAR(p1->x() - p0->x(), 0.3, 1e-12);
  }
}

TEST(PlanarFrameChart, NormalOffsetIsMetricAndStretchRoundTripIsIdentity) {
  const Polygon2D a = Rect(1.6, 0.9), b = Rect(0.8, 0.6);
  for (const auto& uv : UvSamples()) {
    for (double n : {-0.01, 0.0, 0.25}) {
      const Eigen::Vector2d src(uv.x() * 1.6, uv.y() * 0.9);
      for (auto kind : {kStretch, kContain}) {
        auto ab = PlanarFrameChart::Map(src, n, a, b, kind);
        ASSERT_TRUE(ab.ok());
        EXPECT_EQ(ab->z(), n);  // metric, unscaled
      }
      auto ab = PlanarFrameChart::Map(src, n, a, b, kStretch);
      auto ba = PlanarFrameChart::Map(ab->head<2>(), ab->z(), b, a, kStretch);
      ASSERT_TRUE(ba.ok());
      EXPECT_NEAR((ba->head<2>() - src).norm(), 0.0, 1e-12);
      EXPECT_EQ(ba->z(), n);
    }
  }
  // contain A -> B -> A is the identity for equal aspect
  const Polygon2D c = Rect(1.2, 0.9), d = Rect(0.8, 0.6);
  for (const auto& uv : UvSamples()) {
    const Eigen::Vector2d src(uv.x() * 1.2, uv.y() * 0.9);
    auto cd = PlanarFrameChart::Map(src, 0.1, c, d, kContain);
    ASSERT_TRUE(cd.ok());
    auto dc = PlanarFrameChart::Map(cd->head<2>(), cd->z(), d, c, kContain);
    ASSERT_TRUE(dc.ok());
    EXPECT_NEAR((dc->head<2>() - src).norm(), 0.0, 1e-12);
  }
}

TEST(PlanarFrameChart, ContainTransportsTheSourceExtentSoACoordinateReadsBackOnTheSourcePart) {
  // The coordinate carries the source extent: it decodes on its own part to its own point, on any receiver to the
  // letterboxed point, and the transported (u,v) are unchanged.
  const Polygon2D a = Rect(1.6, 0.9), b = Rect(1.2, 0.9);
  auto c = PlanarFrameChart::Encode({0.4, 0.7}, 0.05, a);
  ASSERT_TRUE(c.ok());
  auto back = PlanarFrameChart::Decode(*c, a, kContain);
  ASSERT_TRUE(back.ok());
  EXPECT_NEAR(back->x(), 0.4, 1e-12);
  EXPECT_NEAR(back->y(), 0.7, 1e-12);
  auto onb = PlanarFrameChart::Decode(*c, b, kContain);
  ASSERT_TRUE(onb.ok());
  const double s = 1.2 / 1.6;
  EXPECT_NEAR(onb->x(), 0.6 + (0.4 / 1.6 - 0.5) * 1.6 * s, 1e-12);
  EXPECT_NEAR(onb->y(), 0.45 + (0.7 / 0.9 - 0.5) * 0.9 * s, 1e-12);
}

TEST(PlanarFrameChart, RadialChartIsNotUvPreservingAcrossAspectRatios) {
  // The reason the frame charts exist: the corner (1,1) of a 16:9 board read through the radial chart on a 4:3 board.
  const Polygon2D a = Rect(1.6, 0.9), b = Rect(1.2, 0.9);
  const Eigen::Vector2d fwd(0, 1);
  auto enc = RadialSurfaceChart::Encode({1.6, 0.9}, a, fwd);
  ASSERT_TRUE(enc.ok());
  auto dec = RadialSurfaceChart::Decode(*enc, b, fwd);
  ASSERT_TRUE(dec.ok());
  // The radial image is the 4:3 board's side-wall hit at the 16:9 diagonal angle, not its corner (1.2, 0.9).
  EXPECT_NEAR(dec->x(), 1.2, 1e-9);
  EXPECT_NEAR(dec->y(), 0.45 + 0.5 * 1.2 * (0.9 / 1.6), 1e-9);
  EXPECT_GT(0.9 - dec->y(), 0.1);
}

TEST(PlanarFrameChart, ExtentAcceptsAnyVertexOrderAndClosedRingAndRejectsNonRectangles) {
  EXPECT_TRUE(PlanarFrameChart::ExtentOf(Polygon2D({{1, 1}, {0, 1}, {0, 0}, {1, 0}})).ok());           // clockwise
  EXPECT_TRUE(PlanarFrameChart::ExtentOf(Polygon2D({{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}})).ok());  // closed ring
  EXPECT_FALSE(PlanarFrameChart::ExtentOf(Polygon2D({{0, 0}, {1, 0}, {1.1, 1}, {0, 1}})).ok());       // trapezoid
  EXPECT_FALSE(PlanarFrameChart::ExtentOf(Polygon2D({{0, 0}, {1, 0}, {1, 1}})).ok());                  // triangle
  EXPECT_FALSE(PlanarFrameChart::ExtentOf(Polygon2D({{0, 0}, {1, 1}, {0, 1}, {1, 0}})).ok());          // bow-tie
}

TEST(PlanarFrameChart, DegenerateInputsAreErrorsNeverGarbage) {
  const Polygon2D ok = Rect(1.2, 0.9);
  const Polygon2D zero_w = Rect(0.0, 0.9), zero_h = Rect(1.2, 0.0);
  EXPECT_FALSE(PlanarFrameChart::ExtentOf(zero_w).ok());
  EXPECT_FALSE(PlanarFrameChart::ExtentOf(zero_h).ok());
  EXPECT_FALSE(PlanarFrameChart::Map({0.1, 0.1}, 0.0, zero_w, ok, kStretch).ok());
  EXPECT_FALSE(PlanarFrameChart::Map({0.1, 0.1}, 0.0, ok, zero_h, kContain).ok());
  EXPECT_FALSE(PlanarFrameChart::Map({NAN, 0.1}, 0.0, ok, ok, kStretch).ok());
  EXPECT_FALSE(PlanarFrameChart::Map({0.1, 0.1}, INFINITY, ok, ok, kStretch).ok());
  FrameCoord bad;
  bad.src_width_m = 0.0;
  bad.src_height_m = 1.0;
  EXPECT_FALSE(PlanarFrameChart::Decode(bad, ok, kContain).ok());
  EXPECT_FALSE(PlanarFrameChart::Decode(FrameCoord{0.5, 0.5, 0.0, 1.0, 1.0}, ok, SurfaceChartKind::kRadial).ok());
  EXPECT_FALSE(SurfaceChartKindFromInt(7).ok());
}

}  // namespace
}  // namespace spatialblend
