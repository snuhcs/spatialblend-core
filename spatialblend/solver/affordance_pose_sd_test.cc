// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Affordance poses in the object's scope definition, instantiated per embodiment (affordance_pose_sd.h).
#include "spatialblend/solver/affordance_pose_sd.h"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "spatialblend/model/element/polytope_spatial_definition.h"

namespace spatialblend {
namespace {

std::shared_ptr<Polygon2D> Rect(double x0, double y0, double x1, double y1) {
  return std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}});
}

// A chair at (2, 3) turned by `yaw`, seat 0.5 x 0.44 at `seat` m, with `stand` and `seated` pelvis poses.
ElementTree Chair(double seat, double yaw = 0.0, bool interaction = true) {
  ElementTree tree;
  EXPECT_TRUE(tree.AddElement(1).ok());
  EXPECT_TRUE(tree.AddElement(10, 1).ok());
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t.block<3, 3>(0, 0) = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  t(0, 3) = 2.0;
  t(1, 3) = 3.0;
  tree.GetElement(10)->SetLocalTransform(t);
  auto def = std::make_shared<PolytopeSpatialDefinition>(Eigen::Vector3d(-0.25, -0.22, 0), Eigen::Vector3d(0.25, 0.22, seat),
                                                         GeometryRole::kObstacle);
  SubRegion top;
  top.name = "seat";
  top.local_transform = Eigen::Matrix4d::Identity();
  top.local_transform(2, 3) = seat;
  top.boundary = Rect(-0.25, -0.22, 0.25, 0.22);
  def->AddSubRegion(top);
  tree.GetElement(10)->SetSpatialDefinition(def);
  const GeometryRole role = interaction ? GeometryRole::kInteractionRegion : GeometryRole::kObstacle;
  EXPECT_TRUE(tree.AddElement(11, 10).ok());
  tree.GetElement(11)->SetTag("stand");
  tree.GetElement(11)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(Eigen::Vector3d(-0.25, -0.6, 0), Eigen::Vector3d(0.25, -0.22, 0.1), role));
  tree.GetElement(11)->SetAffordancePose({"pelvis", AffordancePoseTemplate::Support::kActorRoot, ""});
  EXPECT_TRUE(tree.AddElement(12, 10).ok());
  tree.GetElement(12)->SetTag("seated");
  tree.GetElement(12)->SetSpatialDefinition(
      std::make_shared<PolytopeSpatialDefinition>(Eigen::Vector3d(-0.25, -0.22, 0), Eigen::Vector3d(0.25, 0.22, 0.1), role));
  tree.GetElement(12)->SetAffordancePose({"pelvis", AffordancePoseTemplate::Support::kPart, "seat"});
  return tree;
}

Embodiment Body(double standing, bool registered = true) {
  Embodiment e;
  e.id = "b";
  e.joints = {{"hips", -1}};
  EmbodimentAnchor pelvis;
  pelvis.name = "pelvis";
  pelvis.families = kFamilyAffordancePose;
  pelvis.has_standing = registered;
  pelvis.standing_height_m = standing;
  pelvis.standing_provenance_json = R"({"procedure":"fixture","source":"affordance_pose_sd_test"})";
  e.anchors = {pelvis};
  return e;
}

TEST(AffordancePoseSD, EachEmbodimentInstantiatesItsOwnStandAndTheSameSeat) {
  const ElementTree tree = Chair(0.45);
  const Eigen::Vector3d root(2.0, 2.6, 0.0);
  const Embodiment small = Body(0.85), tall = Body(0.97);
  const auto a = InstantiatePoseSD(tree, 10, "pelvis", &small, root);
  const auto b = InstantiatePoseSD(tree, 10, "pelvis", &tall, root);
  ASSERT_TRUE(a.ok()) << a.status();
  ASSERT_TRUE(b.ok()) << b.status();
  ASSERT_EQ(a->poses.size(), 2u);
  EXPECT_EQ((a->Find("stand")->support.point - Eigen::Vector3d(2.0, 2.6, 0.85)).norm(), 0.0);
  EXPECT_EQ((b->Find("stand")->support.point - Eigen::Vector3d(2.0, 2.6, 0.97)).norm(), 0.0);
  // The seat is the scene's, the same for both bodies.
  const Eigen::Vector3d above(2.1, 3.0, 0.8);
  EXPECT_NEAR(PoseSupportDistance(a->Find("seated")->support, above), 0.8 - 0.45, 1e-12);
  EXPECT_NEAR(PoseSupportDistance(b->Find("seated")->support, above), 0.8 - 0.45, 1e-12);
  // V: the chair's and the poses' footprints, from the floor to the highest pose; D its diagonal.
  EXPECT_NEAR(a->hi.z(), 0.85, 1e-12);
  EXPECT_NEAR(b->hi.z(), 0.97, 1e-12);
  EXPECT_NEAR(a->lo.y(), -0.6, 1e-12);
  EXPECT_NEAR(a->scale_m, std::sqrt(0.5 * 0.5 + 0.82 * 0.82 + 0.85 * 0.85), 1e-12);
  EXPECT_TRUE(a->Contains(Eigen::Vector3d(2.0, 2.6, 0.85)));   // the standing pelvis over the stand region
  EXPECT_FALSE(a->Contains(Eigen::Vector3d(2.0, 2.6, 0.86)));  // above the highest pose
  EXPECT_FALSE(a->Contains(Eigen::Vector3d(2.0, 2.3, 0.5)));   // in front of the stand region
}

TEST(AffordancePoseSD, ThereIsNoDefaultStandingHeight) {
  const ElementTree tree = Chair(0.45);
  const Embodiment unregistered = Body(0.9, /*registered=*/false);
  EXPECT_FALSE(InstantiatePoseSD(tree, 10, "pelvis", &unregistered, Eigen::Vector3d::Zero()).ok());
  EXPECT_FALSE(InstantiatePoseSD(tree, 10, "pelvis", nullptr, Eigen::Vector3d::Zero()).ok());
}

TEST(AffordancePoseSD, APoseMustBeAnInteractionRoleChildOnAnExistingPart) {
  const Embodiment body = Body(0.9);
  EXPECT_FALSE(InstantiatePoseSD(Chair(0.45, 0.0, /*interaction=*/false), 10, "pelvis", &body, Eigen::Vector3d::Zero()).ok());
  ElementTree tree = Chair(0.45);
  tree.GetElement(12)->SetAffordancePose({"pelvis", AffordancePoseTemplate::Support::kPart, "backrest"});
  EXPECT_FALSE(InstantiatePoseSD(tree, 10, "pelvis", &body, Eigen::Vector3d::Zero()).ok());
  EXPECT_FALSE(InstantiatePoseSD(Chair(0.45), 10, "hand", &body, Eigen::Vector3d::Zero()).ok());
  EXPECT_TRUE(HasAffordancePoses(Chair(0.45), 10, "pelvis"));
  EXPECT_FALSE(HasAffordancePoses(Chair(0.45), 10, "hand"));
}

// The distance's derivatives are exact on every branch (face, edge, vertex, point), on a turned chair: central
// differences of the distance and of its gradient agree with the analytic ones.
TEST(AffordancePoseSD, TheSupportDistanceHasExactDerivativesOnEveryBranch) {
  const Embodiment body = Body(0.9);
  const auto sd = InstantiatePoseSD(Chair(0.45, 0.4), 10, "pelvis", &body, Eigen::Vector3d(2.0, 2.4, 0.0));
  ASSERT_TRUE(sd.ok());
  const Eigen::Matrix4d o = sd->object_to_world;
  const auto world = [&](double x, double y, double z) {
    return Eigen::Vector3d((o * Eigen::Vector4d(x, y, z, 1)).head<3>());
  };
  struct Probe {
    const char* tag;
    Eigen::Vector3d x;
    int feature_class;  // 0 face/point, 1 edge, 2 vertex
  };
  const std::vector<Probe> probes = {{"seated", world(0.05, 0.02, 0.7), 0},
                                     {"seated", world(0.4, 0.0, 0.7), 1},
                                     {"seated", world(0.4, 0.35, 0.6), 2},
                                     {"stand", Eigen::Vector3d(2.3, 2.5, 0.6), 0}};
  for (const auto& p : probes) {
    const PoseSupport& s = sd->Find(p.tag)->support;
    Eigen::Vector3d g;
    Eigen::Matrix3d h;
    int feat = -1;
    PoseSupportDistance(s, p.x, &g, &h, &feat);
    EXPECT_EQ(feat == 0 ? 0 : feat < 1000 ? 1 : 2, p.feature_class) << p.tag;
    const double e = 1e-6;
    for (int k = 0; k < 3; ++k) {
      Eigen::Vector3d dx = Eigen::Vector3d::Zero();
      dx[k] = e;
      const double fd = (PoseSupportDistance(s, p.x + dx) - PoseSupportDistance(s, p.x - dx)) / (2 * e);
      EXPECT_NEAR(g[k], fd, 1e-8) << p.tag << " axis " << k;
      Eigen::Vector3d gp, gm;
      PoseSupportDistance(s, p.x + dx, &gp);
      PoseSupportDistance(s, p.x - dx, &gm);
      EXPECT_NEAR((h.col(k) - (gp - gm) / (2 * e)).norm(), 0.0, 1e-6) << p.tag << " axis " << k;
    }
  }
}

}  // namespace
}  // namespace spatialblend
