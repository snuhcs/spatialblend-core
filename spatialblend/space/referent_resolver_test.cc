// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/solver/referent_resolver.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/solver/relation_solver.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace test {

// Redirects one shared id to a chosen local element; delegates otherwise.
class RedirectingResolver : public ReferentResolver {
 public:
  RedirectingResolver(shared_element_id_t from, const Element* to)
      : from_(from), to_(to) {}
  const Element* Resolve(shared_element_id_t referent_shared_id,
                         const ElementTree& tree) const override {
    if (referent_shared_id == from_) return to_;
    return tree.GetElementBySharedId(referent_shared_id);
  }

 private:
  shared_element_id_t from_;
  const Element* to_;
};

TEST(ReferentResolverTest, DefaultResolvesBySharedId) {
  ElementTree tree;
  tree.AddElement(1);
  tree.GetElement(1)->SetSharedId(100);
  ReferentResolver def;
  EXPECT_EQ(def.Resolve(100, tree), tree.GetElement(1));
  EXPECT_EQ(def.Resolve(999, tree), nullptr);
}

TEST(ReferentResolverTest, RedirectingResolverRoutesSolve) {
  ElementTree tree;
  tree.AddElement(1);
  tree.GetElement(1)->SetSharedId(999);
  tree.GetElement(1)->SetTag("substitute");
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = 5.0;
  t(1, 3) = 5.0;
  tree.GetElement(1)->SetLocalTransform(t);
  tree.GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
              {3, 3}, {7, 3}, {7, 7}, {3, 7}})));

  RelationPayload p;
  p.referent_shared_id = 42;  // Does NOT exist locally.
  p.subject_anchor_name = "root";
  p.kernel_type = RelationKernelType::kAvatarSpacing;
  // An empty encoded state is not a measurement, and n = ∞ omits the term.
  // A finite n with the support-frame origin makes the resolved referent a
  // ratio the continuation actually descends.
  p.grounded_relation.encoded_state.resize(2);
  p.grounded_relation.encoded_state << 0.0, 0.0;
  p.grounded_relation.support_distance_m = 1.0;
  p.update_count = 2.0;
  std::vector<RelationPayload> payloads = {p};

  RelationSolver solver;
  // Without a resolver, shared id 42 is missing → payload skipped → the actor
  // stays at the origin.
  auto without = solver.Solve(Eigen::Vector2d(0, 0), 0.0, payloads, {}, tree);
  // With the redirecting resolver, 42 → the substitute element, so the solve
  // engages it and moves the actor toward (5, 5).
  RedirectingResolver resolver(42, tree.GetElement(1));
  auto with = solver.Solve(Eigen::Vector2d(0, 0), 0.0, payloads, {}, tree,
                           /*navigable_def=*/nullptr, &resolver);

  EXPECT_GT(with.position.norm(), without.position.norm());
}

TEST(ReferentResolverTest, SpaceHasDefaultResolver) {
  Space space;
  auto* tree = space.GetElementTree();
  tree->AddElement(1);
  tree->GetElement(1)->SetSharedId(7);
  const ReferentResolver& r = space.GetReferentResolver();
  EXPECT_EQ(r.Resolve(7, *tree), tree->GetElement(1));
}

TEST(ReferentResolverTest, SpaceUsesInstalledResolver) {
  Space space;
  auto* tree = space.GetElementTree();
  tree->AddElement(1);
  tree->GetElement(1)->SetSharedId(999);
  auto resolver =
      std::make_shared<RedirectingResolver>(42, tree->GetElement(1));
  space.SetReferentResolver(resolver);
  EXPECT_EQ(space.GetReferentResolver().Resolve(42, *tree),
            tree->GetElement(1));
}

}  // namespace test
}  // namespace spatialblend
