// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// The count n of every relation kind: v1's rule with ONE epsilon (kContactEpsilonM,
// v1 math::Constants::kEpsilon = 1e-6). Distance relations (Spacing, AffordancePose contact) count with
// SourceUpdateCount (v1's Euclidean proximity); Facing counts in angle with
// FacingUpdateCount (v1's angular proximity). Each kind is driven through its own kernel,
// at the poses of contact, approach, static, receding and leaving contact.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/model/relation/avatar_affordance_pose_kernel.h"
#include "spatialblend/model/relation/avatar_facing_kernel.h"
#include "spatialblend/model/relation/avatar_spacing_kernel.h"
#include "spatialblend/model/relation/grounding_rule.h"
#include "spatialblend/model/relation/relation.h"
#include "spatialblend/model/relation/support.h"
#include "spatialblend/solver/engagement.h"
#include "spatialblend/model/realization/remote_update.h"

namespace spatialblend {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// ---- Spacing: the source's support distance to a point referent at (5, 0). ----
double SpacingDistance(double x) {
  ElementTree tree;
  tree.AddElement(1);
  tree.AddElement(2, 1);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = 5.0;
  tree.GetElement(2)->SetLocalTransform(t);
  Relation rel("spacing", "root", 2, std::make_unique<DefaultGroundingRule>(), std::make_unique<AvatarSpacingKernel>());
  return rel.Evaluate(Eigen::Vector3d(x, 0, 0), 0.0, tree).grounded.support_distance_m;
}

TEST(CountRule, SpacingFollowsV1) {
  // contact (d_old below eps): 1, also when the next pose leaves contact (v1 reads the current distance only)
  EXPECT_EQ(SourceUpdateCount(SpacingDistance(5.0), SpacingDistance(5.0)), 1.0) << "contact, dwell";
  EXPECT_EQ(SourceUpdateCount(SpacingDistance(5.0), SpacingDistance(4.0)), 1.0) << "leaving contact";
  // approach: d_old / (d_old - d_new)
  EXPECT_DOUBLE_EQ(SourceUpdateCount(SpacingDistance(1.0), SpacingDistance(2.0)), 4.0 / 1.0) << "approach";
  // static and receding: no approach, infinite
  EXPECT_EQ(SourceUpdateCount(SpacingDistance(2.0), SpacingDistance(2.0)), kInf) << "static";
  EXPECT_EQ(SourceUpdateCount(SpacingDistance(2.0), SpacingDistance(1.0)), kInf) << "receding";
  // a progress below the contact epsilon is no approach; just above it is
  EXPECT_EQ(SourceUpdateCount(1.0, 1.0 - 0.5 * kContactEpsilonM), kInf);
  const double n_eps = SourceUpdateCount(1.0, 1.0 - 2.0 * kContactEpsilonM);
  EXPECT_NEAR(n_eps, 1.0 / (2.0 * kContactEpsilonM), 1e-6 * n_eps);
  // an unknown distance is no count
  EXPECT_EQ(SourceUpdateCount(-1.0, 1.0), kInf);
}

// ---- AffordancePose contact: the anchor's distance to a tabletop (2 x 2 at z = 1). ----
double AffordanceDistance(double z) {
  auto surface = std::make_shared<Polygon3D>(
      std::vector<Eigen::Vector3d>{{0, 0, 1}, {2, 0, 1}, {2, 2, 1}, {0, 2, 1}});
  Support support;
  support.world_transform = surface->GetLocalFrame();
  support.boundary = surface->Get2DBoundary();
  support.referent_element_id = 10;
  // The grounding rule's contact over the tabletop's footprint: the anchor's projection onto the plane.
  support.has_contact = true;
  support.anchor_over_footprint = true;
  support.contact_point_world = Eigen::Vector3d(1, 1, 1);
  AvatarAffordancePoseKernel kernel;
  return kernel.Encode(Eigen::Vector3d(1, 1, z), 0.0, support).support_distance_m;
}

TEST(CountRule, AffordancePoseContactFollowsV1) {
  ASSERT_LT(AffordanceDistance(1.0), kContactEpsilonM) << "on the plane is contact";
  EXPECT_EQ(SourceUpdateCount(AffordanceDistance(1.0), AffordanceDistance(1.0)), 1.0) << "contact, dwell";
  EXPECT_EQ(SourceUpdateCount(AffordanceDistance(1.0), AffordanceDistance(1.2)), 1.0) << "leaving contact";
  EXPECT_NEAR(SourceUpdateCount(AffordanceDistance(1.5), AffordanceDistance(1.4)), 0.5 / 0.1, 1e-9) << "approach";
  EXPECT_EQ(SourceUpdateCount(AffordanceDistance(1.4), AffordanceDistance(1.4)), kInf) << "static";
  EXPECT_EQ(SourceUpdateCount(AffordanceDistance(1.4), AffordanceDistance(1.5)), kInf) << "receding";
}

// ---- Facing: the bearing offset of a look-at point at (5, 0) from the origin, by yaw. ----
GroundedRelation Facing(double yaw, double x = 0.0) {
  Support support;
  support.world_transform(0, 3) = 5.0;
  AvatarFacingKernel kernel;
  return kernel.Encode(Eigen::Vector3d(x, 0, 0), yaw, support);
}
double FacingCount(const GroundedRelation& before, const GroundedRelation& now) {
  return FacingUpdateCount(before.encoded_state(0), now.encoded_state(0),
                           before.encoded_state(before.encoded_state.size() - 1));
}

TEST(CountRule, FacingFollowsV1InAngle) {
  // aligned (the offset below eps degrees): 1, also when the next yaw turns away
  EXPECT_EQ(FacingCount(Facing(0.0), Facing(0.0)), 1.0) << "aligned, dwell";
  EXPECT_EQ(FacingCount(Facing(0.0), Facing(0.3)), 1.0) << "leaving alignment";
  // approach: the remaining angle over this update's rotation, the short way
  EXPECT_NEAR(FacingCount(Facing(0.4), Facing(0.3)), 0.4 / 0.1, 1e-9) << "closing";
  // static: no rotation at all is no count
  EXPECT_EQ(FacingCount(Facing(0.4), Facing(0.4)), kInf) << "static";
  // opening: v1 counts the long way round
  EXPECT_NEAR(FacingCount(Facing(0.3), Facing(0.4)), (2.0 * M_PI - 0.3) / 0.1, 1e-9) << "opening, the long way";
  // co-located with the look-at point: 1
  EXPECT_EQ(FacingCount(Facing(0.4, 5.0), Facing(0.3, 5.0)), 1.0) << "co-located";
  // aligned NOW (the contact rule applied to Facing): exactly 1, whatever the previous offset
  // (a look-at contact that slides with the gaze leaves the same-point previous offset non-zero)
  EXPECT_EQ(FacingCount(Facing(0.4), Facing(0.0)), 1.0) << "arriving aligned";
  EXPECT_EQ(FacingUpdateCount(-0.00588, 0.0), 1.0) << "aligned on a sliding contact";
  EXPECT_EQ(FacingUpdateCount(0.3, -0.5 * kContactEpsilonM * M_PI / 180.0), 1.0) << "within eps degrees";
  EXPECT_FALSE(FacingOpening(0.3, 0.0)) << "aligned now is no opening";
  // a source that turns past alignment within one update counts below 1 (the receiver arrives: RelationSolver)
  EXPECT_LT(FacingUpdateCount(0.1, -0.2), 1.0) << "overshoot";
}

// Engagement reads contact with the same epsilon as the count.
TEST(CountRule, EngagementUsesTheSameContactEpsilon) {
  RelationPayload p;
  p.kernel_type = RelationKernelType::kAvatarSpacing;
  p.grounded_relation.support_distance_m = 0.5 * kContactEpsilonM;
  EXPECT_TRUE(IsEngaged(p));
  EXPECT_EQ(SourceUpdateCount(p.grounded_relation.support_distance_m, 0.0), 1.0);
  p.grounded_relation.support_distance_m = kContactEpsilonM;
  EXPECT_FALSE(IsEngaged(p));
  EXPECT_NE(SourceUpdateCount(p.grounded_relation.support_distance_m, 1.0), 1.0);
}

}  // namespace
}  // namespace spatialblend
