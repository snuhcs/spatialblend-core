// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/avatar_spacing_kernel.h"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/relation/support.h"

namespace spatialblend {
namespace test {

TEST(AvatarSpacingKernelTest, EncodeRelativePosition) {
  AvatarSpacingKernel kernel;

  Support support;
  support.world_transform(0, 3) = 5.0;
  support.world_transform(1, 3) = 8.0;

  auto gr = kernel.Encode(Eigen::Vector3d(3, 6, 0), 0.0, support);

  ASSERT_EQ(gr.encoded_state.size(), 3);
  EXPECT_NEAR(gr.encoded_state(0), -2.0, 1e-9);
  EXPECT_NEAR(gr.encoded_state(1), -2.0, 1e-9);
  EXPECT_NEAR(gr.encoded_state(2), 0.0, 1e-9);
}

TEST(AvatarSpacingKernelTest, EncodeRelativeYaw) {
  AvatarSpacingKernel kernel;

  Support support;
  double angle = M_PI / 2.0;
  support.world_transform(0, 0) = std::cos(angle);
  support.world_transform(0, 1) = -std::sin(angle);
  support.world_transform(1, 0) = std::sin(angle);
  support.world_transform(1, 1) = std::cos(angle);

  auto gr = kernel.Encode(Eigen::Vector3d(0, 0, 0), 0.0, support);

  EXPECT_NEAR(gr.encoded_state(2), -M_PI / 2.0, 1e-9);
}

TEST(AvatarSpacingKernelTest, CompareIdentical) {
  AvatarSpacingKernel kernel;

  GroundedRelation a, b;
  a.encoded_state = Eigen::VectorXd(3);
  a.encoded_state << 1.0, 2.0, 0.5;
  b.encoded_state = a.encoded_state;

  EXPECT_DOUBLE_EQ(kernel.Compare(a, b), 0.0);
}

TEST(AvatarSpacingKernelTest, PreservedAcrossSpaces) {
  AvatarSpacingKernel kernel;

  Support src_support;
  src_support.world_transform(0, 3) = 5.0;
  src_support.world_transform(1, 3) = 8.0;
  auto src_gr = kernel.Encode(Eigen::Vector3d(3, 6, 0), 0.0, src_support);

  Support tgt_support;
  tgt_support.world_transform(0, 3) = 2.0;
  tgt_support.world_transform(1, 3) = 3.0;
  auto tgt_gr = kernel.Encode(Eigen::Vector3d(0, 1, 0), 0.0, tgt_support);

  EXPECT_NEAR(kernel.Compare(src_gr, tgt_gr), 0.0, 1e-9);
}

TEST(AvatarSpacingKernelTest, ProximityCloseIsHigh) {
  AvatarSpacingKernel kernel;
  Support support;
  support.world_transform(0, 3) = 1.0;
  support.world_transform(1, 3) = 0.0;

  // Actor at (0.5, 0) — distance 0.5 from support
  auto gr = kernel.Encode(Eigen::Vector3d(0.5, 0, 0), 0.0, support);
  // proximity = 1/(1+0.5) ≈ 0.67
  EXPECT_GT(gr.proximity, 0.5);
  EXPECT_LT(gr.proximity, 1.0);
}

TEST(AvatarSpacingKernelTest, ProximityFarIsLow) {
  AvatarSpacingKernel kernel;
  Support support;
  support.world_transform(0, 3) = 100.0;
  support.world_transform(1, 3) = 0.0;

  // Actor at origin — distance 100 from support
  auto gr = kernel.Encode(Eigen::Vector3d(0, 0, 0), 0.0, support);
  // proximity = 1/(1+100) ≈ 0.01
  EXPECT_LT(gr.proximity, 0.05);
}

TEST(AvatarSpacingKernelTest, ProximityAtContactIsOne) {
  AvatarSpacingKernel kernel;
  Support support;
  support.world_transform(0, 3) = 5.0;
  support.world_transform(1, 3) = 3.0;

  // Actor at the support position — distance 0
  auto gr = kernel.Encode(Eigen::Vector3d(5, 3, 0), 0.0, support);
  EXPECT_NEAR(gr.proximity, 1.0, 0.01);
}

// When a containing navigable scope IS available, `proximity` is
// normalized by that scope's OWN diameter (not the Hall's-proxemics 1/(1+d)
// fallback above) -- source_scope_diameter_m must carry that SAME diameter so
// RelationSolver::DecodeTargetPoint's approach branch can denormalize an
// EXACT metric standoff (support_distance_m * target_diameter /
// source_scope_diameter_m) instead of reconstructing distance purely from
// the (lossier) normalized `proximity` value.
TEST(AvatarSpacingKernelTest, SourceScopeDiameterMatchesNavigableBoundsWhenPresent) {
  AvatarSpacingKernel kernel;
  Support support;
  support.world_transform(0, 3) = 10.0;
  support.world_transform(1, 3) = 0.0;
  support.has_contact = true;
  support.contact_point_local = Eigen::Vector2d::Zero();
  support.contact_point_world = Eigen::Vector3d(10.0, 0.0, 0.0);

  auto walkable = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {20, 0}, {20, 8}, {0, 8}});
  NavigableSpatialDefinition nav(walkable);
  support.navigable_def = &nav;

  // Actor 3m from the support, well outside (not engaged).
  auto gr = kernel.Encode(Eigen::Vector3d(7, 0, 0), 0.0, support);

  const double expected_diameter = Eigen::Vector2d(20, 8).norm();
  EXPECT_NEAR(gr.source_scope_diameter_m, expected_diameter, 1e-9);
  EXPECT_NEAR(gr.support_distance_m, 3.0, 1e-6);
  EXPECT_NEAR(gr.proximity, 1.0 - 3.0 / expected_diameter, 1e-6);
}

// Companion: when there is NO containing navigable scope, source_scope_
// diameter_m must be the documented "unset" sentinel (<= 0), so the decoder's
// legacy proximity-based fallback (not the exact-metric path) is what runs.
// Outside the container the border relation is the value at the projection
// onto the border: R = 1 and depth 0. It does not grow past 1.
TEST(AvatarSpacingKernelTest, BorderRelationOutsideIsTheBorderValue) {
  AvatarSpacingKernel kernel;
  auto walkable = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {10, 0}, {10, 10}, {0, 10}});
  NavigableSpatialDefinition nav(walkable);

  Support outside;
  outside.container_self = true;
  outside.has_contact = true;
  outside.fast_distance = true;
  outside.navigable_def = &nav;
  outside.contact_point_world = Eigen::Vector3d(0.0, 5.0, 0.0);
  auto out = kernel.Encode(Eigen::Vector3d(-4.0, 5.0, 0.0), 0.0, outside);
  EXPECT_NEAR(out.proximity, 1.0, 1e-9);
  EXPECT_LE(out.proximity, 1.0);
  EXPECT_NEAR(out.support_distance_m, 0.0, 1e-9);

  Support inside = outside;
  inside.has_interior = true;
  inside.fast_distance = true;
  inside.contact_point_world = Eigen::Vector3d(0.0, 5.0, 0.0);
  auto in = kernel.Encode(Eigen::Vector3d(2.0, 5.0, 0.0), 0.0, inside);
  const double diameter = Eigen::Vector2d(10, 10).norm();
  EXPECT_NEAR(in.support_distance_m, 2.0, 1e-6);
  EXPECT_NEAR(in.proximity, 1.0 - 2.0 / diameter, 1e-6);
  EXPECT_LT(in.proximity, 1.0);
}

// The relation's support distance is 0 inside the referent (R = 1). The length of the course to
// the contact is not: it is the same continuous function of the position on both sides of the
// referent's boundary, which is what the receiver's pursuit measures. At the boundary the
// support distance drops from the distance to the contact to 0; the contact distance does not.
TEST(AvatarSpacingKernelTest, ContactDistanceIsContinuousAcrossTheReferentBoundary) {
  AvatarSpacingKernel kernel;
  auto walkable = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {10, 0}, {10, 10}, {0, 10}});
  NavigableSpatialDefinition nav(walkable);

  Support support;
  support.has_contact = true;
  support.navigable_def = &nav;
  // The contact is a point of the referent's boundary far from where the actor crosses it.
  support.contact_point_world = Eigen::Vector3d(8.0, 5.0, 0.0);
  support.has_interior = false;
  const auto outside = kernel.Encode(Eigen::Vector3d(4.0, 5.0, 0.0), 0.0, support);
  EXPECT_NEAR(outside.support_distance_m, 4.0, 1e-9);
  EXPECT_NEAR(outside.contact_distance_m, 4.0, 1e-9);

  // One step later the actor is inside the referent's region: R = 1, support distance 0 ...
  support.has_interior = true;
  const auto inside = kernel.Encode(Eigen::Vector3d(4.0 + 1e-6, 5.0, 0.0), 0.0, support);
  EXPECT_NEAR(inside.support_distance_m, 0.0, 1e-12);
  // ... while the course to the contact has changed by the step only.
  EXPECT_NEAR(inside.contact_distance_m, 4.0 - 1e-6, 1e-9);
}

TEST(AvatarSpacingKernelTest, SourceScopeDiameterUnsetWithoutNavigable) {
  AvatarSpacingKernel kernel;
  Support support;
  support.world_transform(0, 3) = 1.0;
  support.world_transform(1, 3) = 0.0;

  auto gr = kernel.Encode(Eigen::Vector3d(0.5, 0, 0), 0.0, support);
  EXPECT_LE(gr.source_scope_diameter_m, 0.0);
}

TEST(AvatarSpacingKernelTest, Encode3DHasExtraComponents) {
  AvatarSpacingKernel kernel;
  kernel.SetDOF({StateType::kPosition3D, StateType::kRotation3D});

  Support support;
  support.world_transform(0, 3) = 5.0;
  support.world_transform(1, 3) = 8.0;

  auto gr = kernel.Encode(Eigen::Vector3d(3, 6, 2), 0.5, support);

  // 3D: [rel_x, rel_y, rel_z, rel_yaw, rel_pitch, rel_roll]
  ASSERT_EQ(gr.encoded_state.size(), 6);
  // rel_z should be non-zero (anchor at z=2, support at z=0)
  EXPECT_NE(gr.encoded_state(2), 0.0);
}

TEST(AvatarSpacingKernelTest, Encode2DStillHas3Components) {
  AvatarSpacingKernel kernel;
  // Default DOF is 2D

  Support support;
  support.world_transform(0, 3) = 5.0;
  support.world_transform(1, 3) = 8.0;

  auto gr = kernel.Encode(Eigen::Vector3d(3, 6, 0), 0.0, support);

  ASSERT_EQ(gr.encoded_state.size(), 3);
}

// --- Robust time-to-satisfaction (imminence thrash fix) ----------------------
// A near-stationary VR source jitters a few mm/frame. The naive estimate read
// any positive frame-delta as "approach" and collapsed estimated_time to a tiny
// value (fully imminent), which thrashed the realizer's weights and oscillated
// the clone even though the source stood still. The estimate must trust
// directional progress ONLY when the EMA-smoothed speed shows genuine motion.
TEST(AvatarSpacingKernelTest, StationarySourceJitterIsNotImminent) {
  AvatarSpacingKernel kernel;
  GroundedRelation cur, prev;
  cur.encoded_state = Eigen::VectorXd(3);
  cur.encoded_state << 2.0, 0.0, 0.0;  // d_current = 2.0
  prev.encoded_state = Eigen::VectorXd(3);
  prev.encoded_state << 2.5, 0.0, 0.0;  // 0.5 m frame "progress" = jitter

  ActorMotionState still;  // linear_speed = 0: not actually moving
  auto m = kernel.ComputeSatisfactionMetric(cur, &prev, still);

  // A source that is not genuinely moving is not approaching this referent: it
  // arrives in INFINITE steps (dropped, never crawled toward) — v1 semantics,
  // no fallback constant. NOT the spurious directional 2.0 / 0.5 = 4.0 the naive
  // rule produced (imminent) from zero-mean jitter.
  EXPECT_TRUE(std::isinf(m.estimated_time));
}

TEST(AvatarSpacingKernelTest, GenuineApproachIsImminent) {
  AvatarSpacingKernel kernel;
  GroundedRelation cur, prev;
  cur.encoded_state = Eigen::VectorXd(3);
  cur.encoded_state << 2.0, 0.0, 0.0;
  prev.encoded_state = Eigen::VectorXd(3);
  prev.encoded_state << 2.5, 0.0, 0.0;

  ActorMotionState moving;
  moving.linear_speed = 0.5;  // genuinely walking toward the referent
  auto m = kernel.ComputeSatisfactionMetric(cur, &prev, moving);

  // Now directional progress is trusted -> small estimate (imminent), not 200.
  EXPECT_LT(m.estimated_time, 10.0);
}

// The lurch fix: time-to-arrival must depend ONLY on the current remaining and
// the smoothed speed — NOT on the frame-to-frame delta. Two calls with the same
// remaining and speed but very different prev_remaining must return the SAME
// estimate. The old remaining/(prev_remaining - remaining) form returned wildly
// different values frame to frame (trace: est_time 99 -> 3 on one small move),
// which swung the realizer weights and lurched the clone ~1 m.
TEST(AvatarSpacingKernelTest, ImminenceIsFrameDeltaIndependent) {
  AvatarSpacingKernel kernel;
  GroundedRelation cur;
  cur.encoded_state = Eigen::VectorXd(3);
  cur.encoded_state << 2.0, 0.0, 0.0;  // d_current = 2.0

  ActorMotionState moving;
  moving.linear_speed = 0.4;  // genuine approach, above the motion floor

  GroundedRelation prev_small, prev_big;
  prev_small.encoded_state = Eigen::VectorXd(3);
  prev_small.encoded_state << 2.01, 0.0, 0.0;  // tiny frame progress (0.01)
  prev_big.encoded_state = Eigen::VectorXd(3);
  prev_big.encoded_state << 2.90, 0.0, 0.0;  // large frame progress (0.90)

  auto a = kernel.ComputeSatisfactionMetric(cur, &prev_small, moving);
  auto b = kernel.ComputeSatisfactionMetric(cur, &prev_big, moving);

  // Both = remaining / smoothed_speed = 2.0 / 0.4 = 5.0, independent of the
  // frame delta (the old form gave 5.0 vs 2.22 — a weight-swinging difference).
  EXPECT_NEAR(a.estimated_time, 5.0, 1e-6);
  EXPECT_NEAR(b.estimated_time, 5.0, 1e-6);
}

// Directional discrimination: a referent whose distance is INCREASING (the TV
// you're walking away from, not the table you approach) must NOT read as
// imminent — otherwise every referent at a similar range looks equally imminent,
// the position weights go uniform, and the clone sits at the furniture centroid
// (= the "table -> TV" bug). Magnitude from smoothed speed, direction from sign.
TEST(AvatarSpacingKernelTest, RecedingReferentIsNotImminent) {
  AvatarSpacingKernel kernel;
  GroundedRelation cur, prev;
  cur.encoded_state = Eigen::VectorXd(3);
  cur.encoded_state << 3.0, 0.0, 0.0;  // d_current = 3.0
  prev.encoded_state = Eigen::VectorXd(3);
  prev.encoded_state << 2.5, 0.0, 0.0;  // was closer -> distance INCREASING

  ActorMotionState moving;
  moving.linear_speed = 0.5;  // genuinely moving, but away from this referent

  auto m = kernel.ComputeSatisfactionMetric(cur, &prev, moving);
  EXPECT_GT(m.estimated_time, 100.0);  // not closing -> not imminent
}

// An unreachable referent (geodesic +inf, which the
// obstacle-aware context returns after the geometry-mediation rebase) scores
// R = 0 with no metric standoff -- never a Euclidean re-score. The end-to-end
// "out of scope" half is the DISABLED post-rebase contract test.
class UnreachableNavigable : public NavigableSpatialDefinition {
 public:
  using NavigableSpatialDefinition::NavigableSpatialDefinition;
  double GetGeodesicDistance(const Eigen::Vector3d&,
                             const Eigen::Vector3d&) const override {
    return std::numeric_limits<double>::infinity();
  }
};

TEST(AvatarSpacingKernelTest, ScopeGating_E1_2_UnreachableHasZeroR) {
  AvatarSpacingKernel kernel;
  UnreachableNavigable room(std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {10, 0}, {10, 10}, {0, 10}}));
  Support support;
  support.world_transform(0, 3) = 8.0;
  support.world_transform(1, 3) = 8.0;
  support.navigable_def = &room;
  support.has_contact = true;
  support.contact_point_world = Eigen::Vector3d(7.5, 7.5, 0);
  support.contact_point_local = Eigen::Vector2d(-0.5, -0.5);
  const auto gr = kernel.Encode(Eigen::Vector3d(2, 2, 0), 0.0, support);
  EXPECT_DOUBLE_EQ(gr.proximity, 0.0);
  EXPECT_LT(gr.support_distance_m, 0.0) << "no metric standoff when unreachable";
}

// Same, end to end with the real obstacle-aware geodesic (post
// geometry-mediation): a referent sealed by an obstacle ring has a +inf
// geodesic and scores R = 0, never a Euclidean fallback.
TEST(AvatarSpacingKernelTest, ScopeGating_E1_2_ObstacleRingGivesZeroR) {
  NavigableSpatialDefinition room(std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {10, 0}, {10, 10}, {0, 10}}));
  auto rect = [](double x0, double y0, double x1, double y1) {
    return std::make_shared<const Polygon2D>(std::vector<Eigen::Vector2d>{
        {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});
  };
  room.SetObstacles({{1, rect(5.5, 5.5, 8.5, 5.7)},
                     {2, rect(5.5, 8.3, 8.5, 8.5)},
                     {3, rect(5.5, 5.5, 5.7, 8.5)},
                     {4, rect(8.3, 5.5, 8.5, 8.5)}},
                    /*clearance=*/0.2);
  AvatarSpacingKernel kernel;
  Support support;
  support.world_transform(0, 3) = 7.0;
  support.world_transform(1, 3) = 7.0;
  support.navigable_def = &room;
  support.has_contact = true;
  support.contact_point_world = Eigen::Vector3d(6.7, 6.7, 0);
  support.contact_point_local = Eigen::Vector2d(-0.3, -0.3);
  const auto sealed = kernel.Encode(Eigen::Vector3d(2, 2, 0), 0.0, support);
  EXPECT_DOUBLE_EQ(sealed.proximity, 0.0);
  EXPECT_LT(sealed.support_distance_m, 0.0);
  // Control: the same seat outside the ring is scored by a finite geodesic.
  support.contact_point_world = Eigen::Vector3d(2.7, 7.7, 0);
  const auto open = kernel.Encode(Eigen::Vector3d(2, 2, 0), 0.0, support);
  EXPECT_GT(open.proximity, 0.0);
  EXPECT_NEAR(open.support_distance_m, (Eigen::Vector2d(2.7, 7.7) -
                                        Eigen::Vector2d(2, 2)).norm(),
              1e-6);
}

}  // namespace test
}  // namespace spatialblend
