// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Yaw-stage conditioning-vs-search probe (the safeguard reuse).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/relation/avatar_facing_kernel.h"
#include "spatialblend/model/relation/avatar_spacing_kernel.h"
#include "spatialblend/model/relation/grounding_rule.h"
#include "spatialblend/model/relation/relation.h"
#include "spatialblend/solver/relation_evaluator.h"
#include "spatialblend/solver/relation_solver.h"
#include "spatialblend/model/realization/remote_update.h"

namespace spatialblend {
namespace test {
namespace {

RelationPayload SpacingPayload(const Eigen::Vector2d& off, shared_element_id_t sid) {
  ElementTree src;
  src.AddElement(1);
  src.AddElement(2, 1);
  src.GetElement(2)->SetSharedId(sid);
  Relation rel("spacing", "root", 2, std::make_unique<DefaultGroundingRule>(),
               std::make_unique<AvatarSpacingKernel>());
  auto r = rel.Evaluate(Eigen::Vector3d(off.x(), off.y(), 0), 0.0, src);
  RelationPayload p;
  p.referent_shared_id = sid;
  p.subject_anchor_name = "root";
  p.grounded_relation = r.grounded;
  p.support = r.support;
  p.kernel_type = RelationKernelType::kAvatarSpacing;
  return p;
}

RelationPayload FacingPayload(const Eigen::Vector2d& off, double yaw,
                              shared_element_id_t sid) {
  ElementTree src;
  src.AddElement(1);
  src.AddElement(2, 1);
  src.GetElement(2)->SetSharedId(sid);
  Relation rel("facing", "root", 2, std::make_unique<DefaultGroundingRule>(),
               std::make_unique<AvatarFacingKernel>());
  auto r = rel.Evaluate(Eigen::Vector3d(off.x(), off.y(), 0), yaw, src);
  RelationPayload p;
  p.referent_shared_id = sid;
  p.subject_anchor_name = "root";
  p.grounded_relation = r.grounded;
  p.support = r.support;
  p.kernel_type = RelationKernelType::kAvatarFacing;
  return p;
}

void AddReferent(ElementTree* t, element_id_t id, shared_element_id_t sid,
                 const Eigen::Vector2d& pos) {
  t->AddElement(id, 1);
  t->GetElement(id)->SetSharedId(sid);
  Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
  m(0, 3) = pos.x();
  m(1, 3) = pos.y();
  t->GetElement(id)->SetLocalTransform(m);
}

TEST(A1YawProbeAdapted, ConditioningVsSearch) {
  for (double off : {M_PI - 0.1, 2.6, 0.3}) {
    auto sp = SpacingPayload({2, 0}, 200);
    auto f = FacingPayload({-3, 0}, -off, 300);
    ElementTree t;
    t.AddElement(1);
    AddReferent(&t, 2, 200, {0, 0});
    AddReferent(&t, 3, 300, {2, 4});
    RelationSolver s;
    auto r = s.Solve({0, 0}, 0.7, {sp, f}, {0.5, 0.5}, t);
    auto spacing_only = s.Solve({0, 0}, 0.7, {sp}, {1.0}, t);
    TargetAnchorPose a;
    a.position = Eigen::Vector3d(r.position.x(), r.position.y(), 0);
    a.yaw = r.yaw;
    auto ev = EvaluateTargetRelation(f, a, t);
    std::printf("[yaw] off=%.3f Rsrc=%.4f pos=(%.3f,%.3f) yaw=%.4f "
                "spacing_only_yaw=%.4f Rtgt=%.4f\n",
                off, f.grounded_relation.proximity, r.position.x(),
                r.position.y(), r.yaw, spacing_only.yaw, ev.r_tgt);
  }
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
