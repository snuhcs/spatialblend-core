// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/relation_solver_parity_test.cc
//
// The root's 2-D solve must behave exactly as before the
// N-D generalization. This test replays a fixed set of multi-relation root solves (the trust-region path, several
// Spacing relations with finite update counts, with and without a navigable with an obstacle, the live placed path)
// and compares every answer BIT FOR BIT with the values recorded before the
// generalization, in relation_solver_parity_fixture.inc.
//
// Re-recording (only on a commit whose root behaviour is meant to change):
//   bazel test -c opt //spatialblend/solver:relation_solver_parity_test \
//     --test_env=SPATIALBLEND_PARITY_RECORD=/abs/path/relation_solver_parity_fixture_<platform>.inc
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/obstacle_free_space.h"
#include "spatialblend/model/relation/avatar_spacing_kernel.h"
#include "spatialblend/model/relation/grounding_rule.h"
#include "spatialblend/model/relation/relation.h"
#include "spatialblend/solver/relation_solver.h"
#include "spatialblend/model/realization/remote_update.h"

namespace spatialblend {
namespace {

struct ParityRecord {
  const char* name;
  double x, y, j_after;
  int iterations, evaluations, status;
};

// Bit-identity is a property of one compiler, ISA and libm: the base answers are recorded per platform, each before the
// generalization on that platform (macOS arm64, Linux x86_64).
#if defined(__APPLE__) && defined(__aarch64__)
#define SPATIALBLEND_PARITY_PLATFORM "darwin_arm64"
#include "spatialblend/solver/relation_solver_parity_fixture_darwin_arm64.inc"
#elif defined(__linux__) && defined(__x86_64__)
#define SPATIALBLEND_PARITY_PLATFORM "linux_x86_64"
#include "spatialblend/solver/relation_solver_parity_fixture_linux_x86_64.inc"
#else
#define SPATIALBLEND_PARITY_PLATFORM ""
constexpr ParityRecord kParity[] = {{"", 0, 0, 0, 0, 0, 0}};
#endif

ElementTree MakeTree(double x, double y, shared_element_id_t shared) {
  ElementTree tree;
  tree.AddElement(1);
  tree.AddElement(2, 1);
  tree.GetElement(2)->SetSharedId(shared);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = x;
  t(1, 3) = y;
  tree.GetElement(2)->SetLocalTransform(t);
  return tree;
}

RelationPayload Spacing(const Eigen::Vector3d& actor, const ElementTree& tree, shared_element_id_t shared) {
  Relation rel("spacing", "root", 2, std::make_unique<DefaultGroundingRule>(), std::make_unique<AvatarSpacingKernel>());
  const auto r = rel.Evaluate(actor, 0.0, tree);
  RelationPayload p;
  p.referent_shared_id = shared;
  p.subject_anchor_name = "root";
  p.grounded_relation = r.grounded;
  p.support = r.support;
  p.kernel_type = RelationKernelType::kAvatarSpacing;
  return p;
}

void TwoSupports(ElementTree* t) {
  t->AddElement(1);
  t->AddElement(2, 1);
  t->GetElement(2)->SetSharedId(200);
  t->AddElement(3, 1);
  t->GetElement(3)->SetSharedId(300);
  Eigen::Matrix4d t3 = Eigen::Matrix4d::Identity();
  t3(0, 3) = 6.0;
  t->GetElement(3)->SetLocalTransform(t3);
}

struct Case {
  std::string name;
  SolverResult r;
};

std::vector<Case> RunCases() {
  std::vector<Case> out;
  ElementTree target;
  TwoSupports(&target);
  auto walkable = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{-10, -10}, {10, -10}, {10, 10}, {-10, 10}});
  NavigableSpatialDefinition nav(walkable);
  auto sofa = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{{2, 3.55}, {4, 3.55}, {4, 4.45}, {2, 4.45}});
  nav.SetObstacles({FreeSpaceObstacle{/*key=*/1, sofa}}, /*clearance=*/0.2);

  const std::vector<Eigen::Vector3d> src_a = {{5, 0, 0}, {3, 2, 0}, {1.5, -2.5, 0}};
  const std::vector<Eigen::Vector3d> src_b = {{-5, 0, 0}, {-2, 3, 0}, {-1, -1, 0}};
  const std::vector<Eigen::Vector2d> samples = {{3, 0}, {1.8, 3.66}, {-4, 6}, {7, -3}, {2.5, 1.0}};
  const std::vector<std::vector<double>> weights = {{10.0, 1.0}, {1.0, 1.0}, {0.25, 0.75}};
  const std::vector<std::pair<double, double>> counts = {{2.0, 2.0}, {1.0, 3.0}, {4.0, 1.5}};
  RelationSolver solver;
  for (size_t s = 0; s < src_a.size(); ++s) {
    const auto ta = MakeTree(0, 0, 200);
    const auto tb = MakeTree(0, 0, 300);
    RelationPayload a = Spacing(src_a[s], ta, 200);
    RelationPayload b = Spacing(src_b[s], tb, 300);
    for (size_t c = 0; c < counts.size(); ++c) {
      a.update_count = counts[c].first;
      b.update_count = counts[c].second;
      for (size_t w = 0; w < weights.size(); ++w)
        for (size_t k = 0; k < samples.size(); ++k)
          for (int with_nav = 0; with_nav < 2; ++with_nav) {
            char name[96];
            std::snprintf(name, sizeof(name), "s%zu_c%zu_w%zu_k%zu_%s", s, c, w, k, with_nav ? "nav" : "open");
            SolverWarmStart warm;
            warm.has_position = true;
            warm.position = samples[k];
            out.push_back({name, solver.Solve(samples[k], 0.0, {a, b}, weights[w], target,
                                              with_nav ? &nav : nullptr, nullptr, std::nullopt, warm)});
          }
      // The live path: the placed answer (the destination).
      for (size_t k = 0; k < samples.size(); ++k) {
        char name[96];
        std::snprintf(name, sizeof(name), "s%zu_c%zu_k%zu_placed", s, c, k);
        SolverWarmStart warm;
        warm.has_position = true;
        warm.position = samples[k];
        warm.absolute = true;
        out.push_back({name, solver.Solve(samples[k], 0.0, {a, b}, {0.5, 0.5}, target, &nav, nullptr,
                                          std::nullopt, warm)});
      }
    }
  }
  return out;
}

TEST(RootSolveParity, TheTwoDimensionalRootSolveIsBitIdenticalToTheRecordedBase) {
  const std::vector<Case> cases = RunCases();
  if (const char* path = std::getenv("SPATIALBLEND_PARITY_RECORD")) {
    FILE* f = std::fopen(path, "w");
    ASSERT_NE(f, nullptr) << path;
    std::fprintf(f, "// Recorded by relation_solver_parity_test (SPATIALBLEND_PARITY_RECORD). Do not edit.\n");
    std::fprintf(f, "constexpr ParityRecord kParity[] = {\n");
    for (const Case& c : cases)
      std::fprintf(f, "    {\"%s\", %a, %a, %a, %d, %d, %d},\n", c.name.c_str(), c.r.position.x(), c.r.position.y(),
                   c.r.objective_after, c.r.objective_iterations, c.r.objective_evaluations,
                   static_cast<int>(c.r.status));
    std::fprintf(f, "};\n");
    std::fclose(f);
    GTEST_SKIP() << "recorded " << cases.size() << " cases to " << path;
  }
  if (std::string(SPATIALBLEND_PARITY_PLATFORM).empty()) GTEST_SKIP() << "no base record for this platform";
  constexpr size_t kCount = sizeof(kParity) / sizeof(kParity[0]);
  ASSERT_EQ(cases.size(), kCount);
  int multi_iter = 0;
  for (size_t i = 0; i < kCount; ++i) {
    const ParityRecord& rec = kParity[i];
    const SolverResult& r = cases[i].r;
    ASSERT_EQ(cases[i].name, std::string(rec.name));
    // Bit-identical: exact equality of the doubles, not a tolerance.
    EXPECT_EQ(r.position.x(), rec.x) << rec.name;
    EXPECT_EQ(r.position.y(), rec.y) << rec.name;
    EXPECT_EQ(r.objective_after, rec.j_after) << rec.name;
    EXPECT_EQ(r.objective_iterations, rec.iterations) << rec.name;
    EXPECT_EQ(r.objective_evaluations, rec.evaluations) << rec.name;
    EXPECT_EQ(static_cast<int>(r.status), rec.status) << rec.name;
    if (rec.iterations > 0) ++multi_iter;
  }
  // The fixture exercises the trust-region iterations, not only the analytic answers.
  EXPECT_GT(multi_iter, 50);
}

}  // namespace
}  // namespace spatialblend
