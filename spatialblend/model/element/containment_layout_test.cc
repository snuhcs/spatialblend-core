// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Transfer-leaf equivalence: two regions are the same
// region in two spaces when their polygons are congruent under a rigid transform. The
// tolerance is the geometry epsilon relative to the smaller area; there is no other.
#include "spatialblend/model/element/containment_layout.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {
namespace {

using Pts = std::vector<Eigen::Vector2d>;

// An L: not symmetric under any rotation or reflection that fixes it.
Pts LShape() {
  return {{0, 0}, {3, 0}, {3, 1}, {1, 1}, {1, 2}, {0, 2}};
}

Pts Moved(const Pts& p, double yaw, double tx, double ty, size_t start = 0,
          bool reverse = false) {
  const double c = std::cos(yaw), s = std::sin(yaw);
  Pts out;
  for (size_t i = 0; i < p.size(); ++i) {
    const Eigen::Vector2d& v = p[(start + i) % p.size()];
    out.emplace_back(c * v.x() - s * v.y() + tx, s * v.x() + c * v.y() + ty);
  }
  if (reverse) out = Pts(out.rbegin(), out.rend());
  return out;
}

Pts Mirrored(const Pts& p) {
  Pts out;
  for (const auto& v : p) out.emplace_back(-v.x(), v.y());
  return Pts(out.rbegin(), out.rend());  // keep the winding
}

bool Congruent(const Pts& a, const Pts& b) {
  return FootprintsCongruent(Polygon2D(a), Polygon2D(b));
}

TEST(FootprintsCongruent, AnyRigidMotionStartVertexAndWindingIsTheSameRegion) {
  const Pts l = LShape();
  EXPECT_TRUE(Congruent(l, l));
  EXPECT_TRUE(Congruent(l, Moved(l, 0.0, 5.0, -3.0)));
  EXPECT_TRUE(Congruent(l, Moved(l, 2.1, -4.0, 7.5)));
  EXPECT_TRUE(Congruent(l, Moved(l, -0.7, 0.0, 0.0, /*start=*/3)));
  EXPECT_TRUE(Congruent(l, Moved(l, 3.0, 1.0, 1.0, /*start=*/2, /*reverse=*/true)));
  const Pts box = {{-1, -0.8}, {1, -0.8}, {1, 0.8}, {-1, 0.8}};
  EXPECT_TRUE(Congruent(box, Moved(box, 0.4, 3.0, 2.0, 1)));
  // A box rotated by a quarter turn is the same shape as its transpose.
  const Pts tall = {{-0.8, -1}, {0.8, -1}, {0.8, 1}, {-0.8, 1}};
  EXPECT_TRUE(Congruent(box, tall));
}

TEST(FootprintsCongruent, ADifferentShapeOrSizeIsNotTheSameRegion) {
  const Pts box = {{-1, -0.8}, {1, -0.8}, {1, 0.8}, {-1, 0.8}};
  const Pts bigger = {{-1, -0.9}, {1, -0.9}, {1, 0.9}, {-1, 0.9}};
  const Pts square = {{-0.9, -0.9}, {0.9, -0.9}, {0.9, 0.9}, {-0.9, 0.9}};
  EXPECT_FALSE(Congruent(box, bigger));
  EXPECT_FALSE(Congruent(bigger, square));
  // Same area as the L, a different polygon.
  const Pts rect = {{0, 0}, {5, 0}, {5, 1}, {0, 1}};
  EXPECT_FALSE(Congruent(LShape(), rect));
  // The same vertices pulled slightly: a near miss is still a different shape.
  Pts bent = LShape();
  bent[4].y() += 1e-3;
  EXPECT_FALSE(Congruent(LShape(), bent));
}

TEST(FootprintsCongruent, AMirrorImageIsNotARigidMotion) {
  const Pts l = LShape();
  EXPECT_FALSE(Congruent(l, Mirrored(l)));
  EXPECT_FALSE(Congruent(l, Moved(Mirrored(l), 1.3, 2.0, 2.0)));
  // A shape with a mirror symmetry is its own mirror image.
  const Pts box = {{-1, -0.8}, {1, -0.8}, {1, 0.8}, {-1, 0.8}};
  EXPECT_TRUE(Congruent(box, Mirrored(box)));
}

TEST(FootprintsCongruent, DegenerateInputIsNeverCongruent) {
  const Pts line = {{0, 0}, {1, 0}, {2, 0}};
  EXPECT_FALSE(Congruent(line, line));
  EXPECT_FALSE(FootprintsCongruent(Polygon2D(), Polygon2D(LShape())));
}

}  // namespace
}  // namespace spatialblend
