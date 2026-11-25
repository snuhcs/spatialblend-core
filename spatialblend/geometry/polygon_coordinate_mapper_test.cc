// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/geometry/polygon_coordinate_mapper.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <cmath>
#include <limits>
#include <vector>

#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {
namespace {

// The v2 mapper was ported from v1 and checked against it point by
// point; v1 is gone, so its answers are kept here. Invariant: on the grid below, encode
// succeeds exactly at these points, and decode(encode(p)) returns the recorded point. The
// L-shaped (non-convex) polygon exercises the MVC edge-walk path. Recorded from
// v2, which then agreed with v1 on every grid point (same successes, round trips within
// 3.3e-8; v1 and v2 differ only in their Polygon2D backends' floating point).
// Recorded on macOS arm64; 1e-7 leaves room for another platform's floating point near the
// vertices (where MVC weights grow large) and is still 300x tighter than the v1/v2 gap.
constexpr double kGoldenTol = 1e-7;

struct GoldenRoundTrip {
  double x, y, back_x, back_y;
};
constexpr GoldenRoundTrip kLShapeRoundTrips[] = {
    {0.0, 0.0, 0.05431528437335513, 0.054315284373354714},
    {0.0, 0.25, 0, 0.25},
    {0.0, 0.5, 0, 0.5},
    {0.0, 0.75, 0, 0.75},
    {0.0, 1.0, 0, 1},
    {0.0, 1.25, 0, 1.25},
    {0.0, 1.5, 0, 1.5},
    {0.0, 1.75, 0, 1.75},
    {0.0, 2.0, 0.066258010873762971, 1.9374113035039855},
    {0.25, 0.0, 0.25, 0},
    {0.25, 0.25, 0.24999999999999756, 0.25000000000000039},
    {0.25, 0.5, 0.24999999999999745, 0.50000000000000078},
    {0.25, 0.75, 0.25000000000000105, 0.75000000000000255},
    {0.25, 1.0, 0.25000000000000033, 1.0000000000000007},
    {0.25, 1.25, 0.27458430487348956, 1.244544187216917},
    {0.25, 1.5, 0.28826781021682901, 1.4830646418465612},
    {0.25, 1.75, 0.2862234868047468, 1.7259941510570664},
    {0.25, 2.0, 0.25, 2},
    {0.5, 0.0, 0.5, 0},
    {0.5, 0.25, 0.50000000000000067, 0.24999999999999778},
    {0.5, 0.5, 0.4999999999999985, 0.49999999999999822},
    {0.5, 0.75, 0.50000000000000167, 0.75},
    {0.5, 1.0, 0.49999999999999933, 1.0000000000000011},
    {0.5, 1.25, 0.5502732623501807, 1.235143607477734},
    {0.5, 1.5, 0.56839977831010213, 1.4593588478566224},
    {0.5, 1.75, 0.55260055279175613, 1.7042762309031416},
    {0.5, 2.0, 0.5, 2},
    {0.75, 0.0, 0.75, 0},
    {0.75, 0.25, 0.75000000000000011, 0.24999999999999786},
    {0.75, 0.5, 0.74999999999999978, 0.50000000000000067},
    {0.75, 0.75, 0.75000000000000189, 0.74999999999999878},
    {0.75, 1.0, 0.75000000000000033, 0.99999999999999911},
    {0.75, 1.25, 0.82261840289346633, 1.2149769928493845},
    {0.75, 1.5, 0.82326896128588811, 1.4341798714831033},
    {0.75, 1.75, 0.80067114742635048, 1.6869919780450322},
    {0.75, 2.0, 0.75, 2},
    {1.0, 0.0, 1, 0},
    {1.0, 0.25, 0.999999999999999, 0.24999999999999967},
    {1.0, 0.5, 0.99999999999999956, 0.49999999999999972},
    {1.0, 0.75, 1.0000000000000004, 0.75000000000000033},
    {1.0, 1.0, 0.99999999999999978, 1},
    {1.0, 1.25, 1, 1.25},
    {1.0, 1.5, 1, 1.5},
    {1.0, 1.75, 1, 1.75},
    {1.0, 2.0, 0.96220975063592018, 1.9114424236793783},
    {1.25, 0.0, 1.25, 0},
    {1.25, 0.25, 1.2439183792553377, 0.27628343626522561},
    {1.25, 0.5, 1.2329869617946332, 0.55431670818688505},
    {1.25, 0.75, 1.2138128332489202, 0.82306491830346273},
    {1.25, 1.0, 1.25, 1},
    {1.5, 0.0, 1.5, 0},
    {1.5, 0.25, 1.482161761818608, 0.28916518311306399},
    {1.5, 0.5, 1.4594493030542455, 0.56728878667060212},
    {1.5, 0.75, 1.4379788063825951, 0.82017355541902937},
    {1.5, 1.0, 1.5, 1},
    {1.75, 0.0, 1.75, 0},
    {1.75, 0.25, 1.7263203949421793, 0.28531908257255834},
    {1.75, 0.5, 1.7061542249810442, 0.55045084745698381},
    {1.75, 0.75, 1.6895670796135716, 0.79912003499790352},
    {1.75, 1.0, 1.75, 1},
    {2.0, 0.0, 1.9375656143306881, 0.06592005922923716},
    {2.0, 0.25, 2, 0.25},
    {2.0, 0.5, 2, 0.5},
    {2.0, 0.75, 2, 0.75},
    {2.0, 1.0, 1.9114110556192014, 0.96224180828068784},
};

TEST(PolygonCoordinateMapperPortTest, EncodeDecodeMatchesTheRecordedPortAnswers) {
  std::vector<Eigen::Vector2d> pts = {{0, 0}, {2, 0}, {2, 1},
                                      {1, 1}, {1, 2}, {0, 2}};
  Polygon2D v2_poly(pts);
  const Eigen::Vector2d fwd(0.0, 1.0);  // +Y forward
  auto golden = [&](double x, double y) -> const GoldenRoundTrip* {
    for (const auto& g : kLShapeRoundTrips)
      if (std::abs(g.x - x) < 1e-9 && std::abs(g.y - y) < 1e-9) return &g;
    return nullptr;
  };
  int successes = 0;
  for (double x = -0.5; x <= 2.5; x += 0.25) {
    for (double y = -0.5; y <= 2.5; y += 0.25) {
      auto norm = PolygonCoordinateMapper::Encode(Eigen::Vector2d(x, y), v2_poly, fwd);
      const GoldenRoundTrip* g = golden(x, y);
      ASSERT_EQ(norm.ok(), g != nullptr) << "at " << x << "," << y;
      if (!g) continue;
      ++successes;
      auto back = PolygonCoordinateMapper::Decode(norm.value(), v2_poly, fwd);
      ASSERT_TRUE(back.ok()) << "at " << x << "," << y;
      EXPECT_NEAR(back->x(), g->back_x, kGoldenTol) << "at " << x << "," << y;
      EXPECT_NEAR(back->y(), g->back_y, kGoldenTol) << "at " << x << "," << y;
    }
  }
  EXPECT_EQ(successes, static_cast<int>(std::size(kLShapeRoundTrips)));
}

// A convex polygon exercises the radial encode/decode path.
TEST(PolygonCoordinateMapperPortTest, ConvexRoundTripRecoversPoint) {
  std::vector<Eigen::Vector2d> pts = {{0, 0}, {4, 0}, {4, 3}, {0, 3}};
  spatialblend::Polygon2D poly(pts);
  ASSERT_TRUE(poly.IsConvex());
  const Eigen::Vector2d fwd(0.0, 1.0);

  Eigen::Vector2d p(2.5, 1.0);
  auto norm = PolygonCoordinateMapper::Encode(p, poly, fwd);
  ASSERT_TRUE(norm.ok());
  EXPECT_EQ(norm.value().strategy,
            NormalizedPolygonPosition::Strategy::kRadial);
  auto back = PolygonCoordinateMapper::Decode(norm.value(), poly, fwd);
  ASSERT_TRUE(back.ok());
  EXPECT_NEAR(back.value().x(), p.x(), 1e-9);
  EXPECT_NEAR(back.value().y(), p.y(), 1e-9);
}

TEST(PolygonCoordinateMapperBoundaryTest,
     AdmitsOnlyMachineRoundoffPastConvexBoundary) {
  const Polygon2D source({{-1, -1}, {1, -1}, {1, 1}, {-1, 1}});
  const Polygon2D target({{-.5, -.5}, {.5, -.5}, {.5, .5}, {-.5, .5}});
  const Eigen::Vector2d forward(0.0, 1.0);
  // Model a tiny numeric residue after projection onto an edge. The mapper
  // treats only this bounded neighbourhood as the closed polygon domain;
  // Boost's strict predicate alone rejects the represented point.
  const Eigen::Vector2d roundoff_boundary(1.0 + 1e-14, .5);
  ASSERT_FALSE(source.Contains(roundoff_boundary, /*include_boundary=*/true));
  auto encoded = PolygonCoordinateMapper::Encode(roundoff_boundary, source, forward);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_LE(encoded->radial_ratio, 1.0);
  auto mapped = PolygonCoordinateMapper::MapPosition(
      roundoff_boundary, source, forward, target, forward);
  ASSERT_TRUE(mapped.ok()) << mapped.status();
  EXPECT_NEAR(mapped->x(), .5, 1e-12);
  EXPECT_NEAR(mapped->y(), .25, 1e-12);

  // A real outside sample is not an admissible chart coordinate and must not
  // be snapped into a valid relation by the narrow round-off allowance.
  const Eigen::Vector2d outside(1.0 + 1e-4, .5);
  EXPECT_FALSE(PolygonCoordinateMapper::Encode(outside, source, forward).ok());
  EXPECT_FALSE(PolygonCoordinateMapper::MapPosition(outside, source, forward,
                                                     target, forward).ok());
  EXPECT_FALSE(PolygonCoordinateMapper::Encode(
      Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), 0.0), source,
      forward).ok());

  const Polygon2D concave(
      {{0, 0}, {2, 0}, {2, 2}, {1, 2}, {1, 1}, {0, 1}});
  // This is materially inside the L-shape's missing notch, not a boundary
  // residue, and must remain rejected by the shared admission helper.
  EXPECT_FALSE(PolygonCoordinateMapper::Encode(Eigen::Vector2d(.5, 1.5), concave,
                                                forward).ok());

  const Polygon2D collinear({{0, 0}, {1, 0}, {2, 0}});
  const Polygon2D repeated_zero_area({{0, 0}, {1, 0}, {1, 0}, {0, 0}});
  for (const Polygon2D* degenerate : {&collinear, &repeated_zero_area}) {
    EXPECT_FALSE(PolygonCoordinateMapper::Encode(Eigen::Vector2d(1, 0),
                                                  *degenerate, forward).ok());
    EXPECT_FALSE(PolygonCoordinateMapper::MapPosition(Eigen::Vector2d(1, 0),
                                                       *degenerate, forward,
                                                       target, forward).ok());
  }
}

TEST(PolygonCoordinateMapperBoundaryTest,
     ScalesRoundoffAllowanceWithoutAdmittingOutsidePoints) {
  const Polygon2D source(
      {{10000, -4000}, {10002, -4000}, {10002, -3998}, {10000, -3998}});
  const Eigen::Vector2d forward(0.0, 1.0);
  EXPECT_TRUE(PolygonCoordinateMapper::Encode(
      Eigen::Vector2d(10002.0 + 1e-11, -3999.0), source, forward).ok());
  EXPECT_FALSE(PolygonCoordinateMapper::Encode(
      Eigen::Vector2d(10002.0 + 1e-4, -3999.0), source, forward).ok());
}


// Convex offset curve: closed form of the 1-degree-chord polyline the sampled
// construction walks (BuildOffsetCurve, still used for non-convex rings). Its
// points are on that polyline to round-off (1e-9), hence at the stand-off
// distance up to the chord sag (4e-5 d), and the encode / decode / ray
// functions invert each other.
double DistanceToRingRef(const std::vector<Eigen::Vector2d>& ring,
                         const Eigen::Vector2d& p) {
  double best = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < ring.size(); ++i) {
    const Eigen::Vector2d a = ring[i], e = ring[(i + 1) % ring.size()] - a;
    const double t = std::clamp((p - a).dot(e) / e.squaredNorm(), 0.0, 1.0);
    best = std::min(best, (p - (a + t * e)).norm());
  }
  return best;
}

TEST(PolygonCoordinateMapperOffsetTest, ConvexClosedFormIsTheLevelSet) {
  const std::vector<std::vector<Eigen::Vector2d>> rings = {
      {{0, 0}, {2, 0}, {2, 1}, {0, 1}},                    // CCW box
      {{0, 0}, {0, 1}, {2, 1}, {2, 0}},                    // CW box
      {{0, 0}, {3, 0.4}, {2.6, 2.1}, {1.0, 2.5}, {-0.5, 1.2}},  // pentagon
      {{0, 0}, {1, 0}, {2, 0}, {2, 1}, {0, 1}},             // collinear vertex
  };
  for (const auto& ring : rings) {
    const Eigen::Vector2d start = ring[1];
    for (const double d : {0.0, 0.02, 0.3, 1.7}) {
      // Reference: the 1-degree polyline of the same curve.
      std::vector<Eigen::Vector2d> ref;
      const int n = static_cast<int>(ring.size());
      double area2 = 0;
      for (int i = 0; i < n; ++i)
        area2 += ring[i].x() * ring[(i + 1) % n].y() -
                 ring[(i + 1) % n].x() * ring[i].y();
      const double sg = area2 > 0 ? 1.0 : -1.0;
      for (int i = 0; i < n; ++i) {
        const Eigen::Vector2d e = ring[(i + 1) % n] - ring[i];
        const Eigen::Vector2d nm = sg * Eigen::Vector2d(e.y(), -e.x()).normalized();
        const Eigen::Vector2d pe = ring[i] - ring[(i + n - 1) % n];
        const Eigen::Vector2d np = sg * Eigen::Vector2d(pe.y(), -pe.x()).normalized();
        const double turn = std::remainder(
            std::atan2(nm.y(), nm.x()) - std::atan2(np.y(), np.x()), 2 * M_PI);
        if (d > 0 && turn * sg > 1e-9) {
          const int m = std::max(1, static_cast<int>(std::ceil(std::fabs(turn) / (M_PI / 180))));
          for (int k = 0; k <= m; ++k) {
            const double a = std::atan2(np.y(), np.x()) + turn * k / m;
            ref.push_back(ring[i] + d * Eigen::Vector2d(std::cos(a), std::sin(a)));
          }
        }
        ref.push_back(ring[i] + d * nm);
        ref.push_back(ring[(i + 1) % n] + d * nm);
      }
      for (double f = 0.0; f < 1.0; f += 0.0137) {
        auto p = PolygonCoordinateMapper::DecodeOffsetFraction(ring, start, f, d);
        ASSERT_TRUE(p.ok());
        EXPECT_NEAR(DistanceToRingRef(ring, *p), d, 5e-5 * d + 1e-9) << "d " << d << " f " << f;
        EXPECT_LE(DistanceToRingRef(ref, *p), 1e-9) << "d " << d << " f " << f;
        if (d > 0) {
          auto back = PolygonCoordinateMapper::EncodeOffsetFraction(ring, start, *p, d);
          ASSERT_TRUE(back.ok());
          auto again = PolygonCoordinateMapper::DecodeOffsetFraction(ring, start, *back, d);
          ASSERT_TRUE(again.ok());
          EXPECT_LT((*again - *p).norm(), 1e-9) << "d " << d << " f " << f;
        }
      }
      if (d > 0) {
        Eigen::Vector2d c = Eigen::Vector2d::Zero();
        for (const auto& v : ring) c += v / n;
        for (double a = 0.0; a < 2 * M_PI; a += 0.11) {
          const Eigen::Vector2d u(std::cos(a), std::sin(a));
          auto hit = PolygonCoordinateMapper::DecodeOffsetRay(ring, d, c, u);
          ASSERT_TRUE(hit.has_value());
          EXPECT_NEAR(DistanceToRingRef(ring, *hit), d, 5e-5 * d + 1e-9);
          EXPECT_LE(DistanceToRingRef(ref, *hit), 1e-9);
          const Eigen::Vector2d r = *hit - c;
          EXPECT_NEAR(r.x() * u.y() - r.y() * u.x(), 0.0, 1e-9);
          EXPECT_GT(r.dot(u), 0.0);
        }
      }
    }
  }
}


// Micro-bench guard: the stand-off encode runs for every outside Spacing
// candidate on every source tick, so a convex polygon must stay O(n) (the
// sampled construction took ~88k instructions on a 4-vertex region). A generous
// wall bound (unoptimised build, shared machine): well under what the sampled
// curve cost, never flaky.
TEST(PolygonCoordinateMapperOffsetTest, ConvexEncodeIsCheap) {
  for (const int nv : {4, 40}) {
    std::vector<Eigen::Vector2d> ring;
    for (int i = 0; i < nv; ++i) {
      const double a = 2 * M_PI * i / nv;
      ring.emplace_back(2.0 * std::cos(a), 1.2 * std::sin(a));
    }
    const Eigen::Vector2d start = ring[0], pos(3.1, 1.9);
    constexpr int kIters = 2000;
    const auto t0 = std::chrono::steady_clock::now();
    double sink = 0;
    for (int i = 0; i < kIters; ++i) {
      auto f = PolygonCoordinateMapper::EncodeOffsetFraction(
          ring, start, pos + Eigen::Vector2d(1e-6 * i, 0), 1.0);
      ASSERT_TRUE(f.ok());
      sink += *f;
    }
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
            .count() / kIters;
    EXPECT_LT(us, 100.0) << nv << " vertices: " << us << " us per encode (sink " << sink << ")";
    std::cout << "[bench] convex offset encode, " << nv << " vertices: " << us << " us\n";
  }
}

// The nearest boundary point of a convex footprint is a VERTEX for every position in the
// wedge beyond a corner. The radial chart of that contact must exist and decode back to
// it (the radial ray from the centroid through a vertex used to miss both adjacent edges,
// so the chart was never encoded and the decode returned the centroid: an identity
// receiver's approach point was a metre off).
TEST(PolygonCoordinateMapperVertexTest, RadialChartOfAVertexRoundTrips) {
  const Polygon2D box({{-0.22, -0.29}, {0.22, -0.29}, {0.22, 0.29}, {-0.22, 0.29}});
  const Eigen::Vector2d forward(0, 1);
  for (const Eigen::Vector2d& v : box.GetPoints()) {
    auto enc = PolygonCoordinateMapper::Encode(v, box, forward);
    ASSERT_TRUE(enc.ok()) << v.transpose() << ": " << enc.status();
    EXPECT_NEAR(enc->radial_ratio, 1.0, 1e-9);
    auto dec = PolygonCoordinateMapper::Decode(*enc, box, forward);
    ASSERT_TRUE(dec.ok());
    EXPECT_NEAR((*dec - v).norm(), 0.0, 1e-9) << v.transpose();
  }
  // And the same through a rotated, translated copy (the element frame of a polytope).
  const double c = std::cos(0.9), s = std::sin(0.9);
  std::vector<Eigen::Vector2d> pts;
  for (const Eigen::Vector2d& v : box.GetPoints())
    pts.push_back(Eigen::Vector2d(1.9 + c * v.x() - s * v.y(), 0.15 + s * v.x() + c * v.y()));
  const Polygon2D moved(pts);
  for (const Eigen::Vector2d& v : moved.GetPoints()) {
    auto enc = PolygonCoordinateMapper::Encode(v, moved, forward);
    ASSERT_TRUE(enc.ok()) << v.transpose();
    auto dec = PolygonCoordinateMapper::Decode(*enc, moved, forward);
    ASSERT_TRUE(dec.ok());
    EXPECT_NEAR((*dec - v).norm(), 0.0, 1e-9) << v.transpose();
  }
}

}  // namespace
}  // namespace spatialblend
