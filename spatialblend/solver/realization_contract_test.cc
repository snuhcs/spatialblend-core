// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/model/testing/hierarchy_extractor_fixture.h"
#include "spatialblend/solver/relation_evaluator.h"
#include "spatialblend/solver/relation_solver.h"

namespace spatialblend::test {
namespace {

Eigen::Matrix4d Transform(double x, double y, double z, double yaw) {
  return HierarchyExtractorFixture::Transform(x, y, z, yaw);
}

SubRegion Patch(const std::string& name, double z, double yaw = 0.0) {
  SubRegion patch;
  patch.name = name;
  patch.local_transform = Transform(0.10, -0.05, z, yaw);
  patch.boundary = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {-0.20, -0.10}, {0.20, -0.10}, {0.20, 0.10}, {-0.20, 0.10}});
  return patch;
}

RelationPayload AffordancePayload(element_id_t local_id,
                                  const std::string& patch_name,
                                  const std::string& anchor,
                                  double relative_yaw) {
  RelationPayload payload;
  payload.resolved_local_element_id = local_id;
  payload.referent_shared_id = local_id + 1000;
  payload.subject_anchor_name = anchor;
  payload.kernel_type = RelationKernelType::kAvatarAffordancePose;
  payload.support.has_contact = true;
  payload.support.sub_region_name = patch_name;
  payload.support.contact_normalized.strategy =
      NormalizedPolygonPosition::Strategy::kRadial;
  payload.support.contact_normalized.angle = 0.0;
  payload.support.contact_normalized.radial_ratio = 0.0;
  payload.grounded_relation.encoded_state = Eigen::VectorXd(3);
  payload.grounded_relation.encoded_state << 0.0, 0.0, relative_yaw;
  return payload;
}

RelationPayload RootPayload(RelationKernelType type, element_id_t local_id,
                            double facing_offset = 0.0) {
  RelationPayload payload;
  payload.resolved_local_element_id = local_id;
  payload.referent_shared_id = local_id + 1000;
  payload.subject_anchor_name = "root";
  payload.kernel_type = type;
  payload.grounded_relation.encoded_state =
      Eigen::VectorXd(type == RelationKernelType::kAvatarFacing ? 1 : 2);
  if (type == RelationKernelType::kAvatarFacing) {
    payload.grounded_relation.encoded_state << facing_offset;
    // A Facing request with count 1: the yaw reaches its target this update, the placed yaw these tests are about. The
    // yaw is a ratio continuation: without a valid count it holds, and with n > 1 it turns
    // 1/n of the receiver's own offset.
    payload.update_count = 1.0;
  } else {
    payload.grounded_relation.encoded_state << 0.0, 0.0;
  }
  // encoded_state (0,0) / facing_offset 0 both declare the source as
  // exactly on its support / looking straight at the referent -- fully
  // engaged. GroundedRelation::proximity defaults to 0.0 ("far/disengaged"),
  // which contradicted that and was harmless before the refinement (the earlier solver
  // never read proximity for the position/yaw solve). It is R^src now, so an
  // inconsistent payload told the paper-objective refinement the far-away
  // state was the goal, at no bound (R^tgt = 1/(1+d) -> 0 as d -> infinity),
  // which is exactly why it walked off to the iteration cap. Set the
  // consistent value.
  payload.grounded_relation.proximity = 1.0;
  return payload;
}

TEST(R1GroundAffordanceTargetTest, GroundsNamedLocalSeatPatchWithXYZAndYaw) {
  HierarchyExtractorFixture fixture;
  auto* seat =
      fixture.target_tree.GetElement(HierarchyExtractorFixture::kTargetSeatA);
  ASSERT_NE(seat, nullptr);
  SubRegion patch = Patch("seat-A", 0.10, 0.25);
  patch.boundary = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {-0.2, -0.1}, {0.2, -0.1}, {0.2, 0.1}, {-0.2, 0.1}});
  seat->GetSpatialDefinition()->AddSubRegion(std::move(patch));

  RelationSolver solver;
  auto grounded = solver.GroundAffordanceTarget(
      AffordancePayload(HierarchyExtractorFixture::kTargetSeatA, "seat-A",
                        "pelvis", 0.30),
      fixture.target_tree);
  ASSERT_TRUE(grounded.ok());
  EXPECT_EQ(grounded->mode, AffordanceGroundingMode::kExplicitSubRegion);
  EXPECT_EQ(grounded->local_element_id,
            HierarchyExtractorFixture::kTargetSeatA);
  // Target seat A is (-4, 2.5, .35), yaw -pi/2; patch local (.10, -.05, .10).
  EXPECT_NEAR(grounded->position.x(), -4.05, 1e-12);
  EXPECT_NEAR(grounded->position.y(), 2.40, 1e-12);
  EXPECT_NEAR(grounded->position.z(), 0.45, 1e-12);
  EXPECT_NEAR(grounded->yaw, -M_PI / 2.0 + 0.25 + 0.30, 1e-12);

  // The radial chart's forward is +Y.  Angle -pi/2 therefore follows the
  // patch +X axis; these are independent, off-centre and boundary oracles.
  auto off_center = AffordancePayload(HierarchyExtractorFixture::kTargetSeatA,
                                      "seat-A", "pelvis", 0.30);
  off_center.support.contact_normalized.angle = -M_PI / 2.0;
  off_center.support.contact_normalized.radial_ratio = 0.5;
  auto half = solver.GroundAffordanceTarget(off_center, fixture.target_tree);
  ASSERT_TRUE(half.ok());
  EXPECT_NEAR(half->position.x(), -4.05 + 0.1 * std::cos(-M_PI / 2.0 + 0.25),
              1e-12);
  EXPECT_NEAR(half->position.y(), 2.40 + 0.1 * std::sin(-M_PI / 2.0 + 0.25),
              1e-12);
  EXPECT_NEAR(half->position.z(), 0.45, 1e-12);

  off_center.support.contact_normalized.radial_ratio = 1.0;
  auto boundary =
      solver.GroundAffordanceTarget(off_center, fixture.target_tree);
  ASSERT_TRUE(boundary.ok());
  EXPECT_NEAR(boundary->position.x(),
              -4.05 + 0.2 * std::cos(-M_PI / 2.0 + 0.25), 1e-12);
  EXPECT_NEAR(boundary->position.y(), 2.40 + 0.2 * std::sin(-M_PI / 2.0 + 0.25),
              1e-12);
  EXPECT_NEAR(boundary->position.z(), 0.45, 1e-12);
}

TEST(R1GroundAffordanceTargetTest, GroundsWholeBakedSurfaceInSurfaceFrame) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  ASSERT_TRUE(tree.AddElement(2, 1).ok());
  tree.GetElement(2)->SetSharedId(1002);
  tree.GetElement(2)->SetLocalTransform(Transform(50.0, 50.0, 9.0, 0.7));
  const Eigen::Matrix4d surface_frame = Transform(1.5, 2.5, 0.8, 0.4);
  const auto point = [&surface_frame](double x, double y) {
    return Eigen::Vector3d(
        (surface_frame * Eigen::Vector4d(x, y, 0.0, 1.0)).head<3>());
  };
  auto surface = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      point(-2.0, -0.5), point(2.0, -0.5), point(2.0, 0.5), point(-2.0, 0.5)});
  tree.GetElement(2)->SetSpatialDefinition(
      std::make_shared<SurfaceSpatialDefinition>(surface));
  const auto& local_boundary = surface->Get2DBoundary()->GetPoints();
  ASSERT_EQ(local_boundary.size(), 4u);
  EXPECT_NEAR(local_boundary[0].x(), -2.0, 1e-12);
  EXPECT_NEAR(local_boundary[0].y(), -0.5, 1e-12);
  EXPECT_NEAR(local_boundary[2].x(), 2.0, 1e-12);
  EXPECT_NEAR(local_boundary[2].y(), 0.5, 1e-12);

  RelationSolver solver;
  auto payload = AffordancePayload(2, "", "hand_r", 0.20);
  payload.support.contact_normalized.angle = -M_PI / 2.0;
  payload.support.contact_normalized.radial_ratio = 0.5;
  auto grounded = solver.GroundAffordanceTarget(payload, tree);
  ASSERT_TRUE(grounded.ok()) << grounded.status();
  EXPECT_EQ(grounded->mode, AffordanceGroundingMode::kWholePlanarBoundary);
  EXPECT_NEAR(grounded->position.x(), 1.5 + std::cos(0.4), 1e-12);
  EXPECT_NEAR(grounded->position.y(), 2.5 + std::sin(0.4), 1e-12);
  EXPECT_NEAR(grounded->position.z(), 0.8, 1e-12);
  EXPECT_NEAR(grounded->yaw, 0.6, 1e-12);
}

TEST(R1GroundAffordanceTargetTest,
     GroundsWholeNavigableDespiteRoleAndOwningTreeTransform) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  ASSERT_TRUE(tree.AddElement(2, 1).ok());
  tree.GetElement(1)->SetLocalTransform(Transform(30.0, -20.0, 7.0, 0.9));
  tree.GetElement(2)->SetSharedId(1002);
  tree.GetElement(2)->SetLocalTransform(Transform(-9.0, 4.0, 3.0, -0.7));
  auto boundary = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {-3.0, 1.0}, {1.0, 1.0}, {1.0, 2.0}, {-3.0, 2.0}});
  tree.GetElement(2)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          boundary, nullptr, 0.0, GeometryRole::kInteractionRegion));

  RelationSolver solver;
  auto payload = AffordancePayload(2, "", "hand_l", 0.15);
  payload.support.contact_normalized.angle = -M_PI / 2.0;
  payload.support.contact_normalized.radial_ratio = 0.5;
  auto grounded = solver.GroundAffordanceTarget(payload, tree);
  ASSERT_TRUE(grounded.ok());
  EXPECT_EQ(grounded->mode, AffordanceGroundingMode::kWholePlanarBoundary);
  EXPECT_NEAR(grounded->position.x(), 0.0, 1e-12);
  EXPECT_NEAR(grounded->position.y(), 1.5, 1e-12);
  EXPECT_NEAR(grounded->position.z(), 0.0, 1e-12);
  EXPECT_NEAR(grounded->yaw, 0.15, 1e-12);
}

TEST(R1GroundAffordanceTargetTest,
     DecodesIndependentLSeatPacketsForWholeAndNamedSurface) {
  // No E/encoder call constructs these packets or expected positions. The
  // L's strict visibility kernel is [0,1]^2; +X/+Y arm radii are both 2.5.
  const std::vector<Eigen::Vector2d> ring = {
      {0, 0}, {3, 0}, {3, 1}, {1, 1}, {1, 3}, {0, 3}};
  struct Sample { double angle, radius, x, y; };
  const Sample samples[] = {{0, 0, .5, .5},
                            {-M_PI / 2, .5, 1.75, .5},
                            {0, .5, .5, 1.75},
                            {-M_PI / 2, 1, 3, .5},
                            {0, 1, .5, 3},
                            {-M_PI / 4, 1, 1, 1}};
  const Eigen::Matrix4d frame = Transform(-7, 6, .85, -.35);
  for (const bool named : {false, true}) {
    SCOPED_TRACE(named ? "named L patch" : "whole baked L surface");
    ElementTree tree;
    ASSERT_TRUE(tree.AddElement(1).ok());
    tree.GetElement(1)->SetSharedId(1001);
    tree.GetElement(1)->SetLocalTransform(Transform(50, 70, 10, 1.1));
    if (named) {
      // A world-frame parent surface whose explicit patch owns its own pose.
      auto base = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
          {-5, -5, 0}, {5, -5, 0}, {5, 5, 0}, {-5, 5, 0}});
      auto def = std::make_shared<SurfaceSpatialDefinition>(base);
      SubRegion patch;
      patch.name = "seatable-L";
      patch.local_transform = frame;
      patch.boundary = std::make_shared<Polygon2D>(ring);
      def->AddSubRegion(std::move(patch));
      tree.GetElement(1)->SetSpatialDefinition(def);
    } else {
      std::vector<Eigen::Vector3d> baked;
      for (const auto& p : ring)
        baked.emplace_back((frame * Eigen::Vector4d(p.x(), p.y(), 0, 1)).head<3>());
      tree.GetElement(1)->SetSpatialDefinition(
          std::make_shared<SurfaceSpatialDefinition>(
              std::make_shared<Polygon3D>(baked)));
    }
    RelationSolver solver;
    for (const auto& sample : samples) {
      SCOPED_TRACE(::testing::Message() << sample.angle << "," << sample.radius);
      auto packet = AffordancePayload(1, named ? "seatable-L" : "",
                                      "seat_contact", .2);
      packet.support.contact_normalized.angle = sample.angle;
      packet.support.contact_normalized.radial_ratio = sample.radius;
      const auto grounded = solver.GroundAffordanceTarget(packet, tree);
      ASSERT_TRUE(grounded.ok()) << grounded.status();
      const Eigen::Vector3d expected =
          (frame * Eigen::Vector4d(sample.x, sample.y, 0, 1)).head<3>();
      EXPECT_TRUE(grounded->position.isApprox(expected, 1e-10));
      EXPECT_NEAR(grounded->yaw, -.15, 1e-10);
      EXPECT_EQ(grounded->mode, named ? AffordanceGroundingMode::kExplicitSubRegion
                                     : AffordanceGroundingMode::kWholePlanarBoundary);
    }
  }
}

TEST(R1GroundAffordanceTargetTest,
     RejectsNonStarSurfaceInsteadOfEmittingSeatTarget) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  tree.GetElement(1)->SetSharedId(1001);
  tree.GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
              {0, 0}, {3, 0}, {3, 3}, {2, 3}, {2, 1}, {1, 1}, {1, 3}, {0, 3}}),
          nullptr, 0.0, GeometryRole::kInteractionRegion));
  RelationSolver solver;
  for (const double radius : {0.0, .5, 1.0}) {
    auto packet = AffordancePayload(1, "", "seat_contact", 0);
    packet.support.contact_normalized.radial_ratio = radius;
    EXPECT_FALSE(solver.GroundAffordanceTarget(packet, tree).ok());
    const auto realized = solver.SolveActorTargets(
        Eigen::Vector2d(4, 5), .3, {packet}, {1.0}, tree);
    EXPECT_EQ(realized.status, SolverStatus::kFallback);
    for (const auto& target : realized.targets)
      EXPECT_NE(target.anchor_name, "seat_contact");
  }
}

TEST(R1GroundAffordanceTargetTest, RejectsMissingPatchAndMalformedChart) {
  HierarchyExtractorFixture fixture;
  RelationSolver solver;
  RelationPayload no_identity;
  no_identity.kernel_type = RelationKernelType::kAvatarAffordancePose;
  EXPECT_EQ(solver.GroundAffordanceTarget(no_identity, fixture.target_tree)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);

  // A valid geometryless decoy makes an incorrect stale-local -> shared-ID
  // fallback succeed, so NotFound proves explicit local IDs take precedence.
  auto* decoy =
      fixture.target_tree.GetElement(HierarchyExtractorFixture::kTargetRoot);
  ASSERT_NE(decoy, nullptr);
  decoy->SetSharedId(1004);
  auto stale_local = AffordancePayload(999, "", "pelvis", 0.0);
  stale_local.referent_shared_id = 1004;
  EXPECT_EQ(solver.GroundAffordanceTarget(stale_local, fixture.target_tree)
                .status()
                .code(),
            absl::StatusCode::kNotFound);

  auto missing = solver.GroundAffordanceTarget(
      AffordancePayload(HierarchyExtractorFixture::kTargetSeatA, "seat-A",
                        "pelvis", 0.0),
      fixture.target_tree);
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kNotFound);
  RealizationResult skipped = solver.SolveActorTargets(
      Eigen::Vector2d(1, 2), 0.0,
      {AffordancePayload(HierarchyExtractorFixture::kTargetSeatA, "seat-A",
                         "pelvis", 0.0)},
      {}, fixture.target_tree);
  EXPECT_EQ(skipped.status, SolverStatus::kFallback);
  ASSERT_EQ(skipped.targets.size(), 1u);

  auto* seat =
      fixture.target_tree.GetElement(HierarchyExtractorFixture::kTargetSeatA);
  seat->GetSpatialDefinition()->AddSubRegion(Patch("seat-A", 0.10));
  auto malformed = AffordancePayload(HierarchyExtractorFixture::kTargetSeatA,
                                     "seat-A", "pelvis", 0.0);
  malformed.support.contact_normalized.angle = 2.0 * M_PI;
  EXPECT_EQ(solver.GroundAffordanceTarget(malformed, fixture.target_tree)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  malformed.support.contact_normalized.angle = 0.0;
  malformed.support.contact_normalized.radial_ratio = 1.01;
  EXPECT_EQ(solver.GroundAffordanceTarget(malformed, fixture.target_tree)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  malformed.support.contact_normalized.radial_ratio = -0.01;
  EXPECT_EQ(solver.GroundAffordanceTarget(malformed, fixture.target_tree)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  malformed.support.contact_normalized.radial_ratio =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(solver.GroundAffordanceTarget(malformed, fixture.target_tree)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  // Surface contact is planar. A finite but nonzero reserved height must not
  // be silently discarded, nor emitted as an apparently valid body target.
  malformed.support.contact_normalized.radial_ratio = .5;
  for (const double height : {-.5, .5}) {
    malformed.support.contact_normalized.height_ratio = height;
    EXPECT_EQ(solver.GroundAffordanceTarget(malformed, fixture.target_tree)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
    const auto rejected = solver.SolveActorTargets(
        Eigen::Vector2d(1, 2), 0, {malformed}, {1.0}, fixture.target_tree);
    EXPECT_EQ(rejected.status, SolverStatus::kFallback);
    for (const auto& target : rejected.targets)
      EXPECT_NE(target.anchor_name, "pelvis");
  }
  malformed.support.contact_normalized.height_ratio = 0;
  malformed.support.contact_normalized.strategy =
      NormalizedPolygonPosition::Strategy::kMVC;
  EXPECT_EQ(solver.GroundAffordanceTarget(malformed, fixture.target_tree)
                .status()
                .code(),
            absl::StatusCode::kUnimplemented);
}

// A referent without support geometry is an explicit grounding failure and
// yields no target (there is no geometryless centre fallback).
TEST(R1GroundAffordanceTargetTest, GeometrylessReferentIsExplicitFailure) {
  ElementTree tree;
  ASSERT_TRUE(tree.AddElement(1).ok());
  ASSERT_TRUE(tree.AddElement(2, 1).ok());
  tree.GetElement(2)->SetSharedId(1002);
  tree.GetElement(2)->SetLocalTransform(Transform(4.0, -2.0, 1.2, 0.4));
  RelationSolver solver;
  auto payload = AffordancePayload(2, "", "hand_l", 0.1);
  auto grounded = solver.GroundAffordanceTarget(payload, tree);
  ASSERT_FALSE(grounded.ok());
  EXPECT_EQ(grounded.status().code(), absl::StatusCode::kFailedPrecondition);
  const auto realized = solver.SolveActorTargets(Eigen::Vector2d(0, 0), 0.0,
                                                 {payload}, {1.0}, tree);
  EXPECT_EQ(realized.status, SolverStatus::kFallback);
  for (const auto& target : realized.targets)
    EXPECT_NE(target.anchor_name, "hand_l");
}

TEST(R2FixedWeightTest, ComposesSpacingAndFacingWithoutSourceEncoding) {
  HierarchyExtractorFixture fixture;
  RelationSolver solver;
  auto spacing = RootPayload(RelationKernelType::kAvatarSpacing,
                             HierarchyExtractorFixture::kTargetSeatA);
  auto facing = RootPayload(RelationKernelType::kAvatarFacing,
                            HierarchyExtractorFixture::kTargetSeatB);
  // No valid intent: the root stays at the sample. The retired expectation
  // was the engaged decode (-4, 2.5), which is R_tgt = R_src.
  RealizationResult held =
      solver.SolveActorTargets(Eigen::Vector2d(0, 0), 0.0, {spacing, facing},
                               {1.0, 1.0}, fixture.target_tree);
  ASSERT_FALSE(held.targets.empty());
  EXPECT_NEAR(held.targets[0].position.x(), 0.0, 1e-6);
  EXPECT_NEAR(held.targets[0].position.y(), 0.0, 1e-6);

  // Contact (n = 1) closes the spacing distance. Facing still sets yaw.
  spacing.update_count = 1.0;
  RealizationResult result =
      solver.SolveActorTargets(Eigen::Vector2d(0, 0), 0.0, {spacing, facing},
                               {1.0, 1.0}, fixture.target_tree);
  ASSERT_FALSE(result.targets.empty());
  // n = 1 closes to the geodesic contact, which sits on the seat's near
  // face, not at the element origin the R-preserving decode used.
  const Eigen::Vector2d seat(-4.0, 2.5);
  const Eigen::Vector2d got(result.targets[0].position.x(),
                            result.targets[0].position.y());
  EXPECT_LT((got - seat).norm(), 0.5);
  EXPECT_GT(std::abs(result.targets[0].yaw), 0.5);
}

// Weighted paper objective J(S) = sum_p w_p (R^tgt_p(S) - R^src_p)^2 at a
// candidate root position, from the same per-family target evaluator the
// solver minimizes (EvaluateTargetRelation).
double ObjectiveAt(const std::vector<RelationPayload>& payloads,
                   const std::vector<double>& weights, const ElementTree& tree,
                   const Eigen::Vector2d& pos) {
  TargetAnchorPose pose;
  pose.position = Eigen::Vector3d(pos.x(), pos.y(), 0.0);
  pose.yaw = 0.0;
  double j = 0.0;
  for (size_t i = 0; i < payloads.size(); ++i) {
    const TargetRelationEvaluation ev =
        EvaluateTargetRelation(payloads[i], pose, tree);
    if (!ev.ok()) return std::numeric_limits<double>::infinity();
    j += weights[i] * ev.error * ev.error;
  }
  return j;
}

// Brute-force minimum of J over the square [-extent, extent]^2: a coarse
// grid (coarse_step), then a fine grid (fine_step) over the coarse winner's
// neighbourhood. The returned value is an UPPER bound on the true minimum
// that is within the fine grid's resolution of it.
double BruteForceMinObjective(const std::vector<RelationPayload>& payloads,
                              const std::vector<double>& weights,
                              const ElementTree& tree, double extent,
                              double coarse_step, double fine_step) {
  double best = std::numeric_limits<double>::infinity();
  Eigen::Vector2d arg = Eigen::Vector2d::Zero();
  for (double x = -extent; x <= extent; x += coarse_step) {
    for (double y = -extent; y <= extent; y += coarse_step) {
      const double j = ObjectiveAt(payloads, weights, tree, {x, y});
      if (j < best) {
        best = j;
        arg = {x, y};
      }
    }
  }
  const Eigen::Vector2d centre = arg;
  for (double x = centre.x() - coarse_step; x <= centre.x() + coarse_step;
       x += fine_step) {
    for (double y = centre.y() - coarse_step; y <= centre.y() + coarse_step;
         y += fine_step) {
      best = std::min(best, ObjectiveAt(payloads, weights, tree, {x, y}));
    }
  }
  return best;
}

// RETIRED (the J refinement no longer exists): asserts what the position search / refinement produced (the minimiser J*, warm-start restarts, the evaluation cap of the search). The position answer is S0 clamped; the harness defects replace these checks.
TEST(R2FixedWeightTest, DISABLED_BlendsCompetingRootWeightsDeterministically) {
  // With two competing referents both R^src = 1 (RootPayload's encoded
  // offset (0,0) means fully engaged), the paper-objective refinement can
  // move the result off the OLD weighted-centroid-of-positions blend --
  // R^tgt = 1/(1+d) is a NONLINEAR function of position, so its weighted
  // least-squares optimum is generally not the weighted-average position.
  // The earlier exact centroid ((-4.0, 2.25), residual 0.1875) is now only
  // the initializer; this test asserts the general refinement contract (still
  // deterministic, never worse than that initializer) instead of the now-
  // stale exact literals.
  HierarchyExtractorFixture fixture;
  RelationSolver solver;
  auto a = RootPayload(RelationKernelType::kAvatarSpacing,
                       HierarchyExtractorFixture::kTargetSeatA);
  auto b = RootPayload(RelationKernelType::kAvatarSpacing,
                       HierarchyExtractorFixture::kTargetSeatB);
  SolverResult result = solver.Solve(Eigen::Vector2d(0, 0), 0.0, {a, b},
                                     {0.75, 0.25}, fixture.target_tree);
  SolverResult result2 = solver.Solve(Eigen::Vector2d(0, 0), 0.0, {a, b},
                                      {0.75, 0.25}, fixture.target_tree);
  EXPECT_EQ(result.position.x(), result2.position.x());
  EXPECT_EQ(result.position.y(), result2.position.y());
  EXPECT_EQ(result.yaw, result2.yaw);
  EXPECT_EQ(result.objective_after, result2.objective_after);
  EXPECT_LE(result.objective_after, result.objective_before + 1e-9);
  EXPECT_NEAR(result.yaw, 0.0, 1e-12);
  // Not just "no worse than the initializer" but actually
  // close to the true minimum. The minimum is computed here, not quoted: a
  // brute-force grid (0.05 m, then 0.002 m around its winner) over the
  // evaluator the solver minimizes. The compass search stops at a 0.005 m
  // step, so its result may exceed the true minimum by roughly the
  // objective's curvature times (0.005 m)^2; 1e-5 is that bound with
  // margin (the earlier +-5e-3 on 2.578e-2 was a 20% band that would have
  // accepted a search stalled well short of the optimum).
  const double j_star = BruteForceMinObjective(
      {a, b}, {0.75, 0.25}, fixture.target_tree, 10.0, 0.05, 0.002);
  std::printf("[A1] J(result)=%.9e  J*(brute force)=%.9e  gap=%.3e\n",
              result.objective_after, j_star, result.objective_after - j_star);
  EXPECT_NEAR(result.objective_after, j_star, 1e-5);
  // The grid's value can only be above the true minimum, by its resolution;
  // the solver may not be meaningfully BELOW it.
  EXPECT_GE(result.objective_after, j_star - 1e-6);
}

TEST(R2FixedWeightTest, KeepsSameFamilyDifferentAnchorsIndependent) {
  HierarchyExtractorFixture fixture;
  fixture.target_tree.GetElement(HierarchyExtractorFixture::kTargetSeatA)
      ->GetSpatialDefinition()
      ->AddSubRegion(Patch("seat-A", 0.10));
  fixture.target_tree.GetElement(HierarchyExtractorFixture::kTargetSeatB)
      ->GetSpatialDefinition()
      ->AddSubRegion(Patch("seat-B", 0.05));
  RelationSolver solver;
  auto left = AffordancePayload(HierarchyExtractorFixture::kTargetSeatA,
                                "seat-A", "hand_l", 0.0);
  auto right = AffordancePayload(HierarchyExtractorFixture::kTargetSeatB,
                                 "seat-B", "hand_r", 0.0);
  RealizationResult result =
      solver.SolveActorTargets(Eigen::Vector2d(7, 8), 0.4, {left, right},
                               {0.8, 0.2}, fixture.target_tree);
  ASSERT_EQ(result.targets.size(), 3u);
  EXPECT_EQ(result.targets[0].anchor_name, "root");
  EXPECT_NEAR(result.targets[0].position.x(), 7.0, 1e-12);
  EXPECT_EQ(result.targets[1].anchor_name, "hand_l");
  EXPECT_EQ(result.targets[2].anchor_name, "hand_r");
  EXPECT_NEAR(result.targets[1].position.x(), -4.05, 1e-12);
  EXPECT_NEAR(result.targets[1].position.y(), 2.40, 1e-12);
  EXPECT_NEAR(result.targets[1].z, 0.45, 1e-12);
  EXPECT_NEAR(result.targets[2].position.x(), -4.05, 1e-12);
  EXPECT_NEAR(result.targets[2].position.y(), 1.40, 1e-12);
  EXPECT_NEAR(result.targets[2].z, 0.40, 1e-12);

  // Weighting one independent anchor cannot perturb the other or the root.
  RealizationResult perturbed =
      solver.SolveActorTargets(Eigen::Vector2d(7, 8), 0.4, {left, right},
                               {0.8, 0.9}, fixture.target_tree);
  ASSERT_EQ(perturbed.targets.size(), 3u);
  EXPECT_NEAR(perturbed.targets[0].position.x(), 7.0, 1e-12);
  EXPECT_NEAR(perturbed.targets[0].position.y(), 8.0, 1e-12);
  EXPECT_NEAR(perturbed.targets[1].position.x(), result.targets[1].position.x(),
              1e-12);
  EXPECT_NEAR(perturbed.targets[1].position.y(), result.targets[1].position.y(),
              1e-12);
}

TEST(R2FixedWeightTest, ZeroWeightDisablesOnlyItsAnchorGroup) {
  HierarchyExtractorFixture fixture;
  fixture.target_tree.GetElement(HierarchyExtractorFixture::kTargetSeatA)
      ->GetSpatialDefinition()
      ->AddSubRegion(Patch("seat-A", 0.10));
  fixture.target_tree.GetElement(HierarchyExtractorFixture::kTargetSeatB)
      ->GetSpatialDefinition()
      ->AddSubRegion(Patch("seat-B", 0.05));
  RelationSolver solver;
  auto hand = AffordancePayload(HierarchyExtractorFixture::kTargetSeatA,
                                "seat-A", "hand", 0.0);
  auto pelvis = AffordancePayload(HierarchyExtractorFixture::kTargetSeatB,
                                  "seat-B", "pelvis", 0.0);
  RealizationResult result =
      solver.SolveActorTargets(Eigen::Vector2d(7, 8), 0.4, {hand, pelvis},
                               {1.0, 0.0}, fixture.target_tree);
  ASSERT_EQ(result.targets.size(), 2u);
  EXPECT_EQ(result.targets[0].anchor_name, "root");
  EXPECT_EQ(result.targets[1].anchor_name, "hand");
  EXPECT_NEAR(result.targets[1].weight, 1.0, 1e-12);
}

TEST(R2FixedWeightTest, AllZeroWeightsReleaseWithoutTargets) {
  HierarchyExtractorFixture fixture;
  RelationSolver solver;
  auto spacing = RootPayload(RelationKernelType::kAvatarSpacing,
                             HierarchyExtractorFixture::kTargetSeatA);
  auto facing = RootPayload(RelationKernelType::kAvatarFacing,
                            HierarchyExtractorFixture::kTargetSeatB);
  RealizationResult result =
      solver.SolveActorTargets(Eigen::Vector2d(7, 8), 0.4, {spacing, facing},
                               {0.0, 0.0}, fixture.target_tree);
  EXPECT_EQ(result.status, SolverStatus::kConverged);
  EXPECT_TRUE(result.targets.empty());
}

TEST(R2FixedWeightTest, RejectsNegativeAndNonFiniteWeights) {
  HierarchyExtractorFixture fixture;
  RelationSolver solver;
  auto spacing = RootPayload(RelationKernelType::kAvatarSpacing,
                             HierarchyExtractorFixture::kTargetSeatA);
  EXPECT_EQ(solver
                .Solve(Eigen::Vector2d(0, 0), 0.0, {spacing}, {-0.1},
                       fixture.target_tree)
                .status,
            SolverStatus::kFailed);
  RealizationResult result = solver.SolveActorTargets(
      Eigen::Vector2d(0, 0), 0.0, {spacing},
      {std::numeric_limits<double>::quiet_NaN()}, fixture.target_tree);
  EXPECT_EQ(result.status, SolverStatus::kFailed);
  EXPECT_TRUE(result.targets.empty());
}

}  // namespace
}  // namespace spatialblend::test
