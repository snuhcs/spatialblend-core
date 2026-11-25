// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/envelope.h"

#include <cmath>

#include <gtest/gtest.h>

#include "spatialblend/model/element/geometry_role.h"

namespace spatialblend {
namespace {

TEST(EnvelopeTest, BoxFootprintContainsInteriorRejectsOutsideRoleContainment) {
  // 2x4 box centred at (1, 2) (full size 2 x 4 in XY).
  auto def = MakeEnvelope(ColliderType::kBox, {1, 2, 0, 2, 4, 1});
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(def->GetGeometryRole(), GeometryRole::kContainment);
  EXPECT_TRUE(def->Contains(Eigen::Vector2d(1, 2)));      // centre
  EXPECT_TRUE(def->Contains(Eigen::Vector2d(1.9, 3.9)));  // near corner inside
  EXPECT_FALSE(def->Contains(Eigen::Vector2d(2.5, 2)));   // outside +x
  EXPECT_FALSE(def->Contains(Eigen::Vector2d(1, 4.5)));   // outside +y
  ASSERT_NE(def->GetBoundary(), nullptr);
}

TEST(EnvelopeTest, SphereFootprintIsCircle) {
  auto def = MakeEnvelope(ColliderType::kSphere, {0, 0, 0, 2.0}, 32);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(def->GetGeometryRole(), GeometryRole::kContainment);
  EXPECT_TRUE(def->Contains(Eigen::Vector2d(0, 0)));
  EXPECT_TRUE(def->Contains(Eigen::Vector2d(1.8, 0)));   // inside radius
  EXPECT_FALSE(def->Contains(Eigen::Vector2d(2.5, 0)));  // outside radius
  EXPECT_FALSE(def->Contains(Eigen::Vector2d(1.6, 1.6)));  // ~2.26 > 2
}

TEST(EnvelopeTest, CapsuleHorizontalXIsStadium) {
  // axis=X, r=1, height=6 -> cylinder half-length = 6/2 - 1 = 2; spans x in
  // [-3, 3] (caps), |y| <= 1.
  auto def = MakeEnvelope(ColliderType::kCapsule, {0, 0, 0, 1.0, 6.0, 0}, 24);
  ASSERT_NE(def, nullptr);
  EXPECT_TRUE(def->Contains(Eigen::Vector2d(0, 0)));
  EXPECT_TRUE(def->Contains(Eigen::Vector2d(2.0, 0.9)));   // in cylinder body
  EXPECT_TRUE(def->Contains(Eigen::Vector2d(2.9, 0)));     // in right cap
  EXPECT_FALSE(def->Contains(Eigen::Vector2d(0, 1.5)));    // beyond radius in y
  EXPECT_FALSE(def->Contains(Eigen::Vector2d(3.5, 0)));    // beyond right cap
}

TEST(EnvelopeTest, CapsuleVerticalZIsCircle) {
  // axis=Z (up): footprint is a circle of radius r regardless of height.
  auto def = MakeEnvelope(ColliderType::kCapsule, {0, 0, 0, 1.5, 10.0, 2}, 24);
  ASSERT_NE(def, nullptr);
  EXPECT_TRUE(def->Contains(Eigen::Vector2d(1.4, 0)));
  EXPECT_FALSE(def->Contains(Eigen::Vector2d(0, 2.0)));
}

TEST(EnvelopeTest, RejectsDegenerateAndMalformed) {
  EXPECT_EQ(MakeEnvelope(ColliderType::kBox, {0, 0, 0, 0, 4, 1}), nullptr);  // 0 width
  EXPECT_EQ(MakeEnvelope(ColliderType::kBox, {0, 0}), nullptr);              // too few
  EXPECT_EQ(MakeEnvelope(ColliderType::kSphere, {0, 0, 0, 0}), nullptr);     // r=0
}

// MakeEnvelope's box case previously ignored rotation entirely, so a
// rotated box referent's Containment footprint silently reverted to
// axis-aligned. A yaw
// of pi/4 applied to a 2x2 box centred at (1, 2) must rotate the corners
// about that centre; check the resulting boundary carries the exact rotated
// corner coordinates (order-agnostic, since Polygon2D's backend may
// normalize/reorder the ring) rather than the unrotated axis-aligned ones.
TEST(EnvelopeTest, BoxAppliesYaw) {
  const double cx = 1.0, cy = 2.0;
  const double hx = 1.0, hy = 1.0;  // full size 2 x 2
  const double yaw = M_PI / 4.0;
  auto def = MakeEnvelope(ColliderType::kBox, {cx, cy, 0, 2 * hx, 2 * hy, 1},
                          /*circle_segments=*/16, yaw);
  ASSERT_NE(def, nullptr);
  ASSERT_NE(def->GetBoundary(), nullptr);
  const auto& pts = def->GetBoundary()->GetPoints();
  ASSERT_EQ(pts.size(), 4u);

  const double c = std::cos(yaw), s = std::sin(yaw);
  auto rotate = [&](double lx, double ly) -> Eigen::Vector2d {
    return {cx + lx * c - ly * s, cy + lx * s + ly * c};
  };
  std::vector<Eigen::Vector2d> expected = {
      rotate(-hx, -hy), rotate(hx, -hy), rotate(hx, hy), rotate(-hx, hy)};

  // Every expected rotated corner must be present among the boundary's
  // points (order-agnostic match), within tight tolerance.
  for (const auto& e : expected) {
    bool found = false;
    for (const auto& p : pts) {
      if ((p - e).norm() < 1e-9) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "expected rotated corner (" << e.x() << ", "
                       << e.y() << ") not found in boundary";
  }

  // The un-rotated axis-aligned corner (cx+hx, cy+hy) must NOT be present —
  // otherwise yaw was silently dropped.
  const Eigen::Vector2d unrotated(cx + hx, cy + hy);
  for (const auto& p : pts) {
    EXPECT_GT((p - unrotated).norm(), 1e-6);
  }

  // Containment check: a point just inside the rotated +local-x edge (at
  // local (0.9, 0) rotated) must be inside; the un-rotated-frame point
  // (cx + 0.9, cy) that would be inside an axis-aligned box must be outside
  // once truly rotated by 45 degrees (since it now sits near the rotated
  // edge/corner region no longer aligned with the box).
  EXPECT_TRUE(def->Contains(rotate(0.9, 0.0)));
  EXPECT_FALSE(def->Contains(Eigen::Vector2d(cx + hx, cy + hy)));
}

}  // namespace
}  // namespace spatialblend
