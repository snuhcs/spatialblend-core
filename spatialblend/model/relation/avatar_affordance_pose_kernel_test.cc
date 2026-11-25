// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/avatar_affordance_pose_kernel.h"

#include <gtest/gtest.h>

#include "spatialblend/model/element/element.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/model/relation/affordance_pose_grounding_rule.h"
#include "spatialblend/model/relation/support.h"

namespace spatialblend {
namespace test {

// Helper: create a support with boundary for polygon-mapped encoding
Support MakeTabletopSupportWithBoundary() {
  auto surface = std::make_shared<Polygon3D>(
      std::vector<Eigen::Vector3d>{{0, 0, 1}, {2, 0, 1}, {2, 2, 1}, {0, 2, 1}});
  Support support;
  support.world_transform = surface->GetLocalFrame();
  support.boundary = surface->Get2DBoundary();
  support.referent_element_id = 10;
  return support;
}

// Helper: support without boundary (fallback to raw coords)
Support MakeTabletopSupportNoBoundary() {
  auto surface = std::make_shared<Polygon3D>(
      std::vector<Eigen::Vector3d>{{0, 0, 1}, {2, 0, 1}, {2, 2, 1}, {0, 2, 1}});
  Support support;
  support.world_transform = surface->GetLocalFrame();
  support.referent_element_id = 10;
  return support;
}

TEST(AvatarAffordancePoseKernelTest, EncodeProducesState) {
  AvatarAffordancePoseKernel kernel;
  auto support = MakeTabletopSupportWithBoundary();

  auto gr = kernel.Encode(Eigen::Vector3d(1, 1, 1), 0.0, support);

  // Should have 3 values: [angle, radial_ratio, relative_yaw]
  ASSERT_EQ(gr.encoded_state.size(), 3);
  // At center: radial_ratio should be ~0 (centroid)
  EXPECT_NEAR(gr.encoded_state(1), 0.0, 0.1);
  // Relative yaw = 0
  EXPECT_NEAR(gr.encoded_state(2), 0.0, 1e-6);
}

TEST(AvatarAffordancePoseKernelTest, EncodeAtBoundaryHasHighRadialRatio) {
  AvatarAffordancePoseKernel kernel;
  auto support = MakeTabletopSupportWithBoundary();

  // Point at a corner — should have radial_ratio near 1.0
  auto gr = kernel.Encode(Eigen::Vector3d(2, 2, 1), 0.0, support);

  ASSERT_EQ(gr.encoded_state.size(), 3);
  EXPECT_GT(gr.encoded_state(1), 0.5);  // closer to boundary
}

TEST(AvatarAffordancePoseKernelTest, EncodeWithYaw) {
  AvatarAffordancePoseKernel kernel;
  auto support = MakeTabletopSupportWithBoundary();

  auto gr = kernel.Encode(Eigen::Vector3d(1, 1, 1), M_PI / 2.0, support);

  EXPECT_NEAR(std::abs(gr.encoded_state(2)), M_PI / 2.0, 1e-6);
}

TEST(AvatarAffordancePoseKernelTest, PreservedAcrossSpacesSameShape) {
  AvatarAffordancePoseKernel kernel;

  // Same-shaped surfaces at different world positions
  auto src_support = MakeTabletopSupportWithBoundary();

  auto tgt_surface = std::make_shared<Polygon3D>(
      std::vector<Eigen::Vector3d>{{4, 4, 2}, {6, 4, 2}, {6, 6, 2}, {4, 6, 2}});
  Support tgt_support;
  tgt_support.world_transform = tgt_surface->GetLocalFrame();
  tgt_support.boundary = tgt_surface->Get2DBoundary();
  tgt_support.referent_element_id = 20;

  // Same relative position: (0.5, 0.5) offset from center in each surface
  auto src_gr = kernel.Encode(Eigen::Vector3d(1.5, 1.5, 1), 0.0, src_support);
  auto tgt_gr = kernel.Encode(Eigen::Vector3d(5.5, 5.5, 2), 0.0, tgt_support);

  EXPECT_NEAR(kernel.Compare(src_gr, tgt_gr), 0.0, 0.1);
}

TEST(AvatarAffordancePoseKernelTest, CrossShapeMappingDifferentSizes) {
  AvatarAffordancePoseKernel kernel;

  // Source: 3m wide surface
  auto src_boundary = std::make_shared<spatialblend::Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {3, 0}, {3, 0.6}, {0, 0.6}});
  Support src_support;
  src_support.world_transform = Eigen::Matrix4d::Identity();
  src_support.boundary = src_boundary;

  // Target: 2m wide surface
  auto tgt_boundary = std::make_shared<spatialblend::Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {2, 0}, {2, 0.6}, {0, 0.6}});
  Support tgt_support;
  tgt_support.world_transform = Eigen::Matrix4d::Identity();
  tgt_support.boundary = tgt_boundary;

  // Source: at left-third (1.0, 0.3) of 3m surface
  auto src_gr = kernel.Encode(Eigen::Vector3d(1.0, 0.3, 0), 0.0, src_support);

  // Target: at left-third (0.67, 0.3) of 2m surface
  auto tgt_gr = kernel.Encode(Eigen::Vector3d(0.67, 0.3, 0), 0.0, tgt_support);

  // Normalized coords should be similar
  double residual = kernel.Compare(src_gr, tgt_gr);
  EXPECT_LT(residual, 0.5);
}

TEST(AvatarAffordancePoseKernelTest, FallbackWithoutBoundary) {
  AvatarAffordancePoseKernel kernel;
  auto support = MakeTabletopSupportNoBoundary();

  // Without boundary, falls back to raw local coords
  auto gr = kernel.Encode(Eigen::Vector3d(1, 1, 1), 0.0, support);

  ASSERT_EQ(gr.encoded_state.size(), 3);
  // Raw local: center → (0, 0)
  EXPECT_NEAR(gr.encoded_state(0), 0.0, 1e-6);
  EXPECT_NEAR(gr.encoded_state(1), 0.0, 1e-6);
}

TEST(AvatarAffordancePoseKernelTest, DetectMismatch) {
  AvatarAffordancePoseKernel kernel;

  auto support = MakeTabletopSupportWithBoundary();
  auto gr_close = kernel.Encode(Eigen::Vector3d(1, 1, 1), 0.0, support);
  auto gr_far = kernel.Encode(Eigen::Vector3d(5, 5, 1), 0.0, support);

  EXPECT_GT(kernel.Compare(gr_close, gr_far), 0.5);
}

// Proximal activation moved out of the kernel: scope is decided by
// ScopeResolver (kProximal, see resolve_scope_test) and dormancy emerges from
// the proximity R below (ProximityAtCenter / ProximityFarFromSurface), not a
// hard IsActive gate on the kernel.

TEST(AvatarAffordancePoseKernelTest, ProximityAtCenter) {
  AvatarAffordancePoseKernel kernel;
  auto support = MakeTabletopSupportWithBoundary();

  // Actor at surface center → anchor_dist ≈ 0 → proximity ≈ 1.0
  auto gr = kernel.Encode(Eigen::Vector3d(1, 1, 1), 0.0, support);
  EXPECT_GT(gr.proximity, 0.8);
}

TEST(AvatarAffordancePoseKernelTest, ProximityFarFromSurface) {
  AvatarAffordancePoseKernel kernel;
  auto support = MakeTabletopSupportWithBoundary();

  // Actor far from surface → low proximity
  auto gr = kernel.Encode(Eigen::Vector3d(50, 50, 1), 0.0, support);
  EXPECT_LT(gr.proximity, 0.05);
}

TEST(AvatarAffordancePoseKernelTest, Encode3DWithPitchRoll) {
  AvatarAffordancePoseKernel kernel;
  kernel.SetDOF({StateType::kPosition3D, StateType::kRotation3D});

  auto support = MakeTabletopSupportWithBoundary();

  auto gr = kernel.Encode(Eigen::Vector3d(1, 1, 1), M_PI / 4.0, support);

  // 3D: [angle, radial_ratio, rel_yaw, rel_pitch, rel_roll]
  ASSERT_EQ(gr.encoded_state.size(), 5);
  // Yaw should be preserved
  EXPECT_NEAR(std::abs(gr.encoded_state(2)), M_PI / 4.0, 0.1);
}

TEST(AvatarAffordancePoseKernelTest, Encode2DStillHas3Components) {
  AvatarAffordancePoseKernel kernel;
  // Default 2D DOF

  auto support = MakeTabletopSupportWithBoundary();

  auto gr = kernel.Encode(Eigen::Vector3d(1, 1, 1), 0.0, support);

  ASSERT_EQ(gr.encoded_state.size(), 3);
}

// Regression test for the "proximity ≡ 1.0" bug: when the referent has
// its OWN SpatialDefinition, AvatarAffordancePoseKernel::Encode() must derive
// proximity from a genuine anchor-to-support distance, not from a difference
// that is zero by construction. This exercises the REAL production pipeline —
// AffordancePoseGroundingRule::ComputeSupport() feeding Encode() — because the
// manually-built Support helpers above never set has_contact=true and so never
// touch the buggy branch at all (that is itself part of the root cause: the
// existing unit tests could not have caught this).
TEST(AvatarAffordancePoseKernelTest, ProximityDecreasesWithDistanceToOwnBoundary) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());

  // Horizontal tabletop surface (a real referent-owned boundary geometry),
  // centroid at world (1, 1, 1). Reuses the same vertex pattern as
  // MakeTabletopSupportWithBoundary above.
  auto surface = std::make_shared<Polygon3D>(
      std::vector<Eigen::Vector3d>{{0, 0, 1}, {2, 0, 1}, {2, 2, 1}, {0, 2, 1}});
  auto spatial_def = std::make_shared<SurfaceSpatialDefinition>(surface);
  tree.GetElement(1)->SetSpatialDefinition(spatial_def);
  // Keep the element's tree transform consistent with the surface's own
  // local frame (a separate, out-of-scope frame-mismatch issue exists when
  // these disagree — sidestep it here so
  // this test isolates the proximity bug only).
  tree.GetElement(1)->SetLocalTransform(surface->GetLocalFrame());

  AffordancePoseGroundingRule rule;
  AvatarAffordancePoseKernel kernel;

  const Eigen::Vector2d surface_center(1.0, 1.0);
  auto proximity_at_height = [&](double height) {
    Eigen::Vector3d anchor(surface_center.x(), surface_center.y(),
                           1.0 + height);
    Support support =
        rule.ComputeSupport(*tree.GetElement(1), tree, anchor, 0.0);
    EXPECT_TRUE(support.has_contact);
    EXPECT_NE(support.boundary, nullptr);
    return kernel.Encode(anchor, 0.0, support).proximity;
  };

  double p_far = proximity_at_height(1.0);
  double p_mid = proximity_at_height(0.3);
  double p_near = proximity_at_height(0.05);
  double p_contact = proximity_at_height(0.0);

  // Strictly decreasing as the anchor moves away from the support surface.
  EXPECT_LT(p_far, p_mid)
      << "proximity should strictly decrease with distance (1.0m vs 0.3m)";
  EXPECT_LT(p_mid, p_near)
      << "proximity should strictly decrease with distance (0.3m vs 0.05m)";
  EXPECT_LT(p_near, p_contact)
      << "proximity should strictly decrease with distance (0.05m vs contact)";

  // At contact (on the surface), proximity should be (numerically) 1.0.
  EXPECT_GE(p_contact, 1.0 - 1e-6);

  // Sanity: expected closed-form values for 1/(1 + height).
  EXPECT_NEAR(p_far, 1.0 / (1.0 + 1.0), 1e-9);
  EXPECT_NEAR(p_mid, 1.0 / (1.0 + 0.3), 1e-9);
  EXPECT_NEAR(p_near, 1.0 / (1.0 + 0.05), 1e-9);
}

// -------------------------------------------------------------------------
// Regression coverage for the AffordancePose/box (Polytope) referent frame
// bug (sibling of the SpacingGroundingRule fix).
// A PolytopeSpatialDefinition stores its geometry verbatim in the
// OWNING ELEMENT's local frame (SpatialDefinition::RequiresElementWorldTransform
// == true). Before the fix, AffordancePoseGroundingRule::ComputeSupport called
// spatial_def->ProjectOnto(anchor_pos) with the raw WORLD anchor, silently
// treating a box placed away from (and rotated relative to) the tree origin
// as if it sat at the world origin with yaw 0 — exactly the bug class
// fixed for SpacingGroundingRule.
class AffordancePoseGroundingRulePolytopeTest : public ::testing::Test {
 protected:
  static Eigen::Matrix4d MakeTransform(double x, double y, double yaw) {
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 0) = std::cos(yaw);
    t(0, 1) = -std::sin(yaw);
    t(1, 0) = std::sin(yaw);
    t(1, 1) = std::cos(yaw);
    t(0, 3) = x;
    t(1, 3) = y;
    return t;
  }

  // Seat box: LOCAL half-extents 0.25 x 0.25 in XY, seat height 0.45m
  // (z in [0, 0.45]) around the element's own origin.
  static std::shared_ptr<PolytopeSpatialDefinition> MakeSeatBox() {
    return std::make_shared<PolytopeSpatialDefinition>(
        Eigen::Vector3d(-0.25, -0.25, 0.0), Eigen::Vector3d(0.25, 0.25, 0.45));
  }

  void SetUp() override {
    ASSERT_TRUE(tree_.AddElement(1).ok());
    ASSERT_TRUE(tree_.AddElement(2, 1).ok());
    // Seat placed at world (3, 2), yaw pi/4 — a rotated box away from the
    // tree origin, matching the task's repro coordinates exactly.
    tree_.GetElement(2)->SetLocalTransform(MakeTransform(3.0, 2.0, M_PI / 4.0));
    tree_.GetElement(2)->SetSpatialDefinition(MakeSeatBox());
  }

  ElementTree tree_;
};

TEST_F(AffordancePoseGroundingRulePolytopeTest,
       ProximityAtTopSurfaceIsFullyProximate) {
  AffordancePoseGroundingRule rule;
  AvatarAffordancePoseKernel kernel;

  // Pelvis anchor exactly on the seat's top surface. Yaw about Z does not
  // move the box's own origin, so the world XY position of the seat's centre
  // column is unaffected by rotation: world (3, 2, 0.45).
  Eigen::Vector3d anchor_on_surface(3.0, 2.0, 0.45);
  Support support = rule.ComputeSupport(*tree_.GetElement(2), tree_,
                                        anchor_on_surface, 0.0);
  ASSERT_TRUE(support.has_contact);
  auto gr = kernel.Encode(anchor_on_surface, 0.0, support);

  // Before the fix, ProjectOnto was queried with the raw WORLD anchor
  // (3, 2, 0.45) against the box's LOCAL geometry (half-extents 0.25 around
  // the element's own origin) — nowhere near (3, 2) in local space — so the
  // "nearest point" was actually a distant box corner, producing a large,
  // wrong anchor_dist and a badly degraded proximity instead of ~1.0.
  EXPECT_GE(gr.proximity, 1.0 - 1e-6);
}

TEST_F(AffordancePoseGroundingRulePolytopeTest,
       ProximityStrictlyLowerWhenOffSurface) {
  AffordancePoseGroundingRule rule;
  AvatarAffordancePoseKernel kernel;

  Eigen::Vector3d anchor_on_surface(3.0, 2.0, 0.45);
  Eigen::Vector3d anchor_off_surface(3.0, 2.0, 0.45 + 0.3);

  Support support_on = rule.ComputeSupport(*tree_.GetElement(2), tree_,
                                           anchor_on_surface, 0.0);
  Support support_off = rule.ComputeSupport(*tree_.GetElement(2), tree_,
                                            anchor_off_surface, 0.0);
  double p_on = kernel.Encode(anchor_on_surface, 0.0, support_on).proximity;
  double p_off = kernel.Encode(anchor_off_surface, 0.0, support_off).proximity;

  EXPECT_LT(p_off, p_on)
      << "proximity must strictly decrease 0.3m off the support surface";
  // Closed form: proximity = 1 / (1 + height_above_surface).
  EXPECT_NEAR(p_off, 1.0 / (1.0 + 0.3), 1e-9);
  EXPECT_NEAR(p_on, 1.0, 1e-9);
}

// A hand hanging off the seat's edge has a genuinely non-zero
// in-plane offset (clamped to the boundary) and this whole scenario -- box
// placement, anchor position and anchor yaw -- is invariant under a shared
// rigid (rotation + translation) transform: offset_u/offset_v/height/R do
// not change, only contact_point_world moves.
TEST(AvatarAffordancePoseKernelTest,
     OffEdgeOffsetIsInvariantUnderARigidTransform) {
  auto MakeTransform = [](double x, double y, double yaw) {
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 0) = std::cos(yaw);
    t(0, 1) = -std::sin(yaw);
    t(1, 0) = std::sin(yaw);
    t(1, 1) = std::cos(yaw);
    t(0, 3) = x;
    t(1, 3) = y;
    return t;
  };
  auto Encode = [&](const Eigen::Matrix4d& box_transform,
                    const Eigen::Vector3d& local_anchor,
                    double local_anchor_yaw) {
    ElementTree tree;
    EXPECT_TRUE(tree.AddElement(1).ok());
    EXPECT_TRUE(tree.AddElement(2, 1).ok());
    tree.GetElement(2)->SetLocalTransform(box_transform);
    tree.GetElement(2)->SetSpatialDefinition(
        std::make_shared<PolytopeSpatialDefinition>(
            Eigen::Vector3d(-0.25, -0.25, 0.0),
            Eigen::Vector3d(0.25, 0.25, 0.45)));
    const Eigen::Vector4d world_anchor4 =
        box_transform * local_anchor.homogeneous();
    const Eigen::Vector3d world_anchor = world_anchor4.head<3>();
    const double box_yaw = std::atan2(box_transform(1, 0), box_transform(0, 0));
    AffordancePoseGroundingRule rule;
    AvatarAffordancePoseKernel kernel;
    Support support = rule.ComputeSupport(*tree.GetElement(2), tree,
                                          world_anchor, box_yaw);
    EXPECT_TRUE(support.has_contact);
    return kernel.Encode(world_anchor, box_yaw + local_anchor_yaw, support);
  };

  // Reference: box at the tree origin, yaw 0. Anchor at local (0.35, 0, 0.45)
  // -- 0.10 m beyond the +X half-extent (0.25), exactly on the top surface,
  // with a 0.2 rad relative yaw.
  const Eigen::Vector3d local_anchor(0.35, 0.0, 0.45);
  const auto reference =
      Encode(Eigen::Matrix4d::Identity(), local_anchor, 0.2);
  ASSERT_EQ(reference.encoded_state.size(), 3);
  EXPECT_NEAR(reference.ap_offset_u_m, 0.10, 1e-9);
  EXPECT_NEAR(reference.ap_offset_v_m, 0.0, 1e-9);
  EXPECT_NEAR(reference.ap_height_m, 0.0, 1e-9);
  EXPECT_NEAR(reference.encoded_state[2], 0.2, 1e-9);

  // The SAME scenario after a shared rigid transform (rotate 37 deg,
  // translate (5, -3)) applied to the box placement, the anchor position AND
  // the anchor's own yaw together.
  const Eigen::Matrix4d rigid = MakeTransform(5.0, -3.0, 37.0 * M_PI / 180.0);
  const auto moved = Encode(rigid, local_anchor, 0.2);
  EXPECT_NEAR(moved.ap_offset_u_m, reference.ap_offset_u_m, 1e-9);
  EXPECT_NEAR(moved.ap_offset_v_m, reference.ap_offset_v_m, 1e-9);
  EXPECT_NEAR(moved.ap_height_m, reference.ap_height_m, 1e-9);
  EXPECT_NEAR(moved.encoded_state[2], reference.encoded_state[2], 1e-9);
  EXPECT_NEAR(moved.proximity, reference.proximity, 1e-9);
}


// The R pose fit is a declared, monotone, bounded combination of
// position, height and yaw (relative to the source's relative yaw), and the
// yaw term is dropped (weights renormalized) when the reference is unknown.
TEST(AvatarAffordancePoseKernelTest, PoseFitRIsDeclaredMonotoneAndBounded) {
  EXPECT_NEAR(kApPositionWeight + kApHeightWeight + kApYawWeight, 1.0, 1e-12);
  // Exact pose: every fit term is 1.
  EXPECT_NEAR(AffordancePoseR(0, 0, 0, 0.0), 1.0, 1e-12);
  // Closed form for each axis in isolation.
  EXPECT_NEAR(AffordancePoseR(0.3, 0.4, 0, 0.0),
              kApPositionWeight / 1.5 + kApHeightWeight + kApYawWeight, 1e-12);
  EXPECT_NEAR(AffordancePoseR(0, 0, -0.25, 0.0),
              kApPositionWeight + kApHeightWeight / 1.25 + kApYawWeight,
              1e-12);
  EXPECT_NEAR(AffordancePoseR(0, 0, 0, M_PI / 2),
              kApPositionWeight + kApHeightWeight + kApYawWeight * 0.5, 1e-12);
  EXPECT_NEAR(AffordancePoseR(0, 0, 0, M_PI),
              kApPositionWeight + kApHeightWeight, 1e-12);
  // Monotone in each offset.
  EXPECT_GT(AffordancePoseR(0.1, 0, 0, 0.0), AffordancePoseR(0.2, 0, 0, 0.0));
  EXPECT_GT(AffordancePoseR(0, 0, 0.1, 0.0), AffordancePoseR(0, 0, 0.2, 0.0));
  EXPECT_GT(AffordancePoseR(0, 0, 0, 0.1), AffordancePoseR(0, 0, 0, 0.2));
  // Unknown yaw reference: the yaw term does not contribute, and the other
  // weights renormalize (an exact position/height is still R = 1).
  EXPECT_NEAR(AffordancePoseR(0, 0, 0, std::nullopt), 1.0, 1e-12);
  EXPECT_NEAR(AffordancePoseR(0, 0, 1.0, std::nullopt),
              (kApPositionWeight + kApHeightWeight / 2.0) /
                  (kApPositionWeight + kApHeightWeight),
              1e-12);
}

// Encode keeps `proximity` as the geometric component
// 1/(1+distance) (what contact hysteresis reads) regardless of yaw, and
// records the pose decomposition + formula version.
TEST(AvatarAffordancePoseKernelTest, ProximityStaysGeometricRegardlessOfYaw) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  tree.GetElement(1)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-0.5, -0.5, 0.0), Eigen::Vector3d(0.5, 0.5, 0.45)));
  AffordancePoseGroundingRule rule;
  AvatarAffordancePoseKernel kernel;
  const Eigen::Vector3d anchor(0.0, 0.0, 0.75);  // 0.3 m above the top
  const Support support = rule.ComputeSupport(*tree.GetElement(1), tree,
                                              anchor, 0.0);
  ASSERT_TRUE(support.has_contact);
  const auto aligned = kernel.Encode(anchor, support.GetYaw(), support);
  const auto turned = kernel.Encode(anchor, support.GetYaw() + M_PI / 2, support);
  EXPECT_NEAR(aligned.proximity, 1.0 / 1.3, 1e-9);
  EXPECT_NEAR(turned.proximity, 1.0 / 1.3, 1e-9);
  EXPECT_EQ(AffordancePoseRFormulaVersion(aligned), kApRFormulaPoseFitV1);
  EXPECT_NEAR(aligned.ap_height_m, 0.3, 1e-9);
  GroundedRelation pre_a4;  // offsets NaN by default
  pre_a4.proximity = 0.9;
  EXPECT_EQ(AffordancePoseRFormulaVersion(pre_a4), kApRFormulaProximityOnly);
}

}  // namespace test
}  // namespace spatialblend
