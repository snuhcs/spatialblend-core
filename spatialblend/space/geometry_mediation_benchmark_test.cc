// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/geometry_mediation_benchmark_test.cc
//
// Microbenchmark: free-space construction, replanning and the
// per-frame cost of source update + solve + a bounded driver step along
// Space::QueryCourse with 10 / 30 / 60 obstacle boxes. The
// numbers are printed (and recorded as test properties); the assertions are
// loose regression guards only.
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "spatialblend/space/testing/test_driver.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/obstacle_free_space.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace test {
namespace {

using Clock = std::chrono::steady_clock;
double Us(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

std::vector<Eigen::Vector2d> BoxCenters(int n) {
  std::vector<Eigen::Vector2d> out;
  const int cols = 8;
  const int rows = (n + cols - 1) / cols;
  for (int i = 0; i < n; ++i) {
    out.emplace_back(1.0 + (i % cols) * 1.3 + 0.25,
                     1.0 + (i / cols) * (10.0 / std::max(1, rows)) + 0.25);
  }
  return out;
}

TEST(GeometryMediationBenchmark, FreeSpaceAndTickCost) {
  for (int n : {10, 30, 60}) {
    // --- geometry: construction, graph, replans ---------------------------
    auto room = std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
        {0, 0}, {12, 0}, {12, 12}, {0, 12}});
    std::vector<FreeSpaceObstacle> obs;
    for (const auto& c : BoxCenters(n)) {
      obs.push_back({static_cast<int64_t>(obs.size()),
                     std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
                         c + Eigen::Vector2d(-0.25, -0.25),
                         c + Eigen::Vector2d(0.25, -0.25),
                         c + Eigen::Vector2d(0.25, 0.25),
                         c + Eigen::Vector2d(-0.25, 0.25)})});
    }
    const auto t0 = Clock::now();
    ObstacleFreeSpace fs(room, obs, 0.2);
    const size_t nodes = fs.GraphNodeCount();
    const auto t1 = Clock::now();
    constexpr int kReplans = 50;
    int found = 0;
    for (int k = 0; k < kReplans; ++k) {
      found += fs.FindPath({0.3 + 0.01 * k, 0.3}, {11.7, 11.7 - 0.01 * k}).found;
    }
    const auto t2 = Clock::now();

    // --- space: per-tick cost with a continuous mediator ------------------
    auto ctx = LocalContext::Create();
    Space* src = ctx->AddSpace();
    Space* rcv = ctx->AddSpace();
    for (Space* sp : {src, rcv}) {
      auto* tree = sp->GetElementTree();
      ASSERT_TRUE(tree->AddElement(1).ok());
      tree->GetElement(1)->SetSharedId(100);
      tree->GetElement(1)->SetSpatialDefinition(
          std::make_shared<NavigableSpatialDefinition>(room));
    }
    // A corresponding referent in both rooms drives the root relation.
    for (Space* sp : {src, rcv}) {
      auto* tree = sp->GetElementTree();
      ASSERT_TRUE(tree->AddElement(2, 1).ok());
      tree->GetElement(2)->SetSharedId(900);
      Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
      t(0, 3) = 11.6;
      t(1, 3) = 6.0;
      tree->GetElement(2)->SetLocalTransform(t);
      tree->GetElement(2)->SetSpatialDefinition(
          std::make_shared<PolytopeSpatialDefinition>(
              Eigen::Vector3d(-0.2, -0.5, 0), Eigen::Vector3d(0.2, 0.5, 1),
              GeometryRole::kObstacle));
    }
    int id = 10;
    for (const auto& c : BoxCenters(n)) {
      auto* tree = rcv->GetElementTree();
      ASSERT_TRUE(tree->AddElement(id, 1).ok());
      Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
      t(0, 3) = c.x();
      t(1, 3) = c.y();
      tree->GetElement(id)->SetLocalTransform(t);
      tree->GetElement(id)->SetSpatialDefinition(
          std::make_shared<PolytopeSpatialDefinition>(
              Eigen::Vector3d(-0.25, -0.25, 0), Eigen::Vector3d(0.25, 0.25, 1),
              GeometryRole::kObstacle));
      ++id;
    }
    std::vector<SubjectAnchor> anchors = {
        SubjectAnchor("root", Eigen::Vector3d::Zero())};
    auto actor = src->CreateActor(anchors);
    src->GetActor(actor)->SetElementId(1);
    auto avatar = rcv->CreateActor(anchors);
    rcv->GetActor(avatar)->SetElementId(1);
    src->Bind(actor, "root", RelationKernelType::kAvatarSpacing);
    src->SetHub(ctx->GetHub());
    rcv->SetHub(ctx->GetHub());
    src->SetMovementGate(0.0, 0.0);
    // The avatar's driver: a bounded body that publishes its pose and follows
    // each proposal along the traversable course (its cost is in the frame).
    driver::Pose body;
    uint64_t body_sequence = 0;
    const auto publish_body = [&]() {
      EmbodiedPoseInput in;
      in.x = body.position.x();
      in.y = body.position.y();
      in.yaw = body.yaw;
      in.epoch = 1;
      rcv->PublishEmbodiedPose(avatar, in, 0.0, ++body_sequence);
    };
    publish_body();
    bool proposed = false;
    const auto drive = [&]() {
      ReceiverProposal proposal;
      if (!rcv->ReadProposal(avatar, &proposal) ||
          (proposal.status != ProposalStatus::kExact &&
           proposal.status != ProposalStatus::kInexact))
        return;
      proposed = true;
      driver::Pose target;
      target.position = Eigen::Vector2d(proposal.x, proposal.y);
      target.yaw = proposal.yaw;
      body = driver::Step(
          body, target, rcv->QueryCourse(avatar, body.position, target.position),
          driver::Bounds{}, 1.0 / 30.0);
      publish_body();
    };
    // SB_BENCH_FRAMES overrides the frame count per repetition (profiling).
    const char* env_frames = std::getenv("SB_BENCH_FRAMES");
    const int kFrames = env_frames ? std::max(2, std::atoi(env_frames)) : 120;
    constexpr int kReps = 5;
    std::vector<double> frames_us;  // steady frames: update + solve + tick
    double first_frame_us = 0.0;
    for (int rep = 0; rep < kReps; ++rep) {
      for (int f = 0; f < kFrames; ++f) {
        const double s = static_cast<double>(f) / kFrames;
        const Eigen::Vector2d pos(0.5 + 11.0 * s, 0.5 + 11.0 * s);
        const auto u0 = Clock::now();
        src->UpdateActorState(actor, pos, 0.0);
        drive();
        const auto u1 = Clock::now();
        if (rep == 0 && f == 0) {
          first_frame_us = Us(u0, u1);  // builds free spaces and graphs
          continue;
        }
        frames_us.push_back(Us(u0, u1));
      }
    }
    auto pct = [](std::vector<double> v, double q) {
      std::sort(v.begin(), v.end());
      return v[std::min(v.size() - 1,
                        static_cast<size_t>(q * (v.size() - 1) + 0.5))];
    };
    const double median = pct(frames_us, 0.5), p95 = pct(frames_us, 0.95);
    const double worst = pct(frames_us, 1.0);

    // Moving obstacles: one obstacle-role element moves every frame. Any
    // obstacle transform change rebuilds the footprints, free spaces and
    // visibility graphs on the next query.
    std::vector<double> moving_us;
    for (int f = 0; f < 20; ++f) {
      Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
      t(0, 3) = BoxCenters(n)[0].x() + 0.001 * (f + 1);
      t(1, 3) = BoxCenters(n)[0].y();
      rcv->GetElementTree()->GetElement(10)->SetLocalTransform(t);
      const Eigen::Vector2d pos(3.0 + 0.01 * f, 2.0);
      const auto u0 = Clock::now();
      src->UpdateActorState(actor, pos, 0.0);
      drive();
      moving_us.push_back(Us(u0, Clock::now()));
    }
    std::printf(
        "[bench] boxes=%d graph_nodes=%zu build(ctor+graph)=%.0fus "
        "replan=%.0fus/path found=%d/%d | first frame (builds)=%.0fus | "
        "steady frame (update+solve+tick, %zu frames x %d reps): "
        "median=%.0fus p95=%.0fus max=%.0fus | moving obstacle frame: "
        "median=%.0fus\n",
        n, nodes, Us(t0, t1), Us(t1, t2) / kReplans, found, kReplans,
        first_frame_us, frames_us.size() / kReps, kReps, median, p95, worst,
        pct(moving_us, 0.5));
    RecordProperty("boxes_" + std::to_string(n) + "_frame_median_us",
                   std::to_string(median));
    RecordProperty("boxes_" + std::to_string(n) + "_frame_p95_us",
                   std::to_string(p95));
    EXPECT_EQ(found, kReplans);
    EXPECT_TRUE(proposed) << "benchmark scene produced no proposals";
    // Per-frame budget (2.5 ms per frame) on the steady median; generous factor
    // for loaded CI machines.
    EXPECT_LT(median, 4 * 2500.0);
  }
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
