// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Microbenchmark for the Facing (line-of-sight) scope gate: cost of one
// source tick with a Facing binding over N box elements (all Obstacle-role
// polytopes, so every element is both a candidate and an occluder).
//
//   bazel run -c opt //spatialblend/space:scope_gate_benchmark
//
// Reports ms/tick with the default 3.66 m Facing radius and with it lifted
// (every element in scope: the worst case).

#include <chrono>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/space/local_context.h"

using namespace spatialblend;

static double MsPerTick(int n, bool unlimited) {
  auto context = LocalContext::Create();
  Space* space = context->AddSpace();
  auto* tree = space->GetElementTree();
  const double side = 40.0;
  (void)tree->AddElement(1);
  tree->GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(
          std::vector<Eigen::Vector2d>{{0, 0}, {side, 0}, {side, side}, {0, side}})));
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> pos(1.0, side - 1.0);
  for (int i = 0; i < n; ++i) {
    const element_id_t id = 2 + i;
    (void)tree->AddElement(id, 1);
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 3) = pos(rng);
    t(1, 3) = pos(rng);
    tree->GetElement(id)->SetLocalTransform(t);
    tree->GetElement(id)->SetSpatialDefinition(
        std::make_shared<PolytopeSpatialDefinition>(
            Eigen::Vector3d(-0.2, -0.2, 0), Eigen::Vector3d(0.2, 0.2, 0.8),
            GeometryRole::kObstacle));
  }
  auto actor = space->CreateActor({SubjectAnchor("root", {0, 0, 0}),
                                   SubjectAnchor("chest", {0, 0, 1.2})});
  space->GetActor(actor)->SetElementId(1);
  space->Bind(actor, "chest", RelationKernelType::kAvatarFacing);
  (void)unlimited;  // there is no scope radius
  const int ticks = 20;
  space->UpdateActorState(actor, {side / 2, side / 2}, 0.0);  // warm-up
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < ticks; ++i) {
    space->UpdateActorState(actor, {side / 2 + 0.01 * i, side / 2}, 0.0);
  }
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / ticks;
}

int main() {
  std::printf("elements  ms/tick(radius 3.66 m)  ms/tick(unlimited)\n");
  for (int n : {100, 400, 1000}) {
    std::printf("%8d  %22.3f  %18.3f\n", n, MsPerTick(n, false),
                MsPerTick(n, true));
  }
  return 0;
}
