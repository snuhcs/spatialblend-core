// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <atomic>
#include <memory>
#include <cstring>
#include <thread>
#include <vector>
#include "gtest/gtest.h"
#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/space/embodied_port.h"
#include "spatialblend/space/receiver_proposal.h"
#include "spatialblend/api/c/c_test_embodied.h"

// Concurrency stress of the embodied-pose port and the proposal mailbox
// (publish vs solve vs read), meant to run under ThreadSanitizer:
//   bazel test --copt=-fsanitize=thread --linkopt=-fsanitize=thread \
//     //spatialblend/api/c:port_concurrency_test
using namespace spatialblend;

TEST(PortTsan, ProposalBoxNeverTorn) {
  ProposalBox box;
  std::atomic<bool> done{false};
  std::thread w([&] {
    for (uint64_t k = 1; k <= 100000; ++k) {
      auto p = std::make_shared<ReceiverProposal>();
      p->revision = k;
      p->x = p->y = p->yaw = static_cast<double>(k);
      p->sample_sequence = k;
      p->cause = static_cast<uint32_t>(k & 0xfff);
      AnchorProposal other;
      other.anchor_id = 1;
      other.x = static_cast<double>(k);
      p->anchors.push_back(other);
      box.Store(std::move(p));
    }
    done = true;
  });
  std::atomic<long> torn{0};
  std::vector<std::thread> rs;
  for (int i = 0; i < 3; ++i) rs.emplace_back([&] {
    uint64_t last = 0;
    while (!done.load()) {
      ReceiverProposal p;
      if (!box.Load(&p)) continue;
      // One set is one solve: every anchor in it belongs to the same revision.
      if (p.y != p.x || p.yaw != p.x || p.sample_sequence != p.revision ||
          p.x != static_cast<double>(p.revision) ||
          p.cause != static_cast<uint32_t>(p.revision & 0xfff) ||
          p.anchors.size() != 1 || p.anchors[0].x != p.x ||
          p.revision < last)
        ++torn;
      last = p.revision;
    }
  });
  w.join();
  for (auto& r : rs) r.join();
  EXPECT_EQ(torn.load(), 0);
}

TEST(PortTsan, SeqlockNeverTornAcrossEpochs) {
  EmbodiedPort port;
  std::atomic<bool> done{false};
  std::thread w([&] {
    for (uint64_t k = 1; k <= 200000; ++k) {
      EmbodiedPoseInput p; p.x = k; p.y = k; p.yaw = k; p.epoch = k / 97 + 1;
      p.flags = (uint32_t)(k & 1);
      uint64_t seq = (k % 97) + 1;  // restarts with each epoch
      port.Publish(p, (double)k, seq, (int64_t)k);
    }
    done = true;
  });
  std::vector<std::thread> rs;
  std::atomic<long> torn{0}, reads{0};
  for (int i = 0; i < 3; ++i) rs.emplace_back([&] {
    uint64_t last_epoch = 0, last_seq = 0;
    while (!done.load()) {
      EmbodiedSample s = port.Sample(); ++reads;
      if (!s.valid) continue;
      uint64_t k = (uint64_t)s.x;
      if (s.y != s.x || s.yaw != s.x || s.timestamp_s != s.x ||
          s.published_ns != (int64_t)k || s.flags != (k & 1) ||
          s.epoch != k / 97 + 1 || s.sequence != (k % 97) + 1) ++torn;
      if (s.epoch < last_epoch || (s.epoch == last_epoch && s.sequence < last_seq)) ++torn;
      last_epoch = s.epoch; last_seq = s.sequence;
    }
  });
  w.join(); for (auto& r : rs) r.join();
  EXPECT_EQ(torn.load(), 0) << reads.load();
}

static void Tr(double* m, double x, double y) {
  double a[16] = {1,0,0,x, 0,1,0,y, 0,0,1,0, 0,0,0,1}; std::memcpy(m, a, sizeof a);
}

TEST(PortTsan, PublishReadVsSolveThreads) {
  sb_session_t s = sb_create_session();
  sb_space_t A = sb_add_space(s), B = sb_add_space(s);
  sb_element_t ra = sb_add_element(A, 0), rb = sb_add_element(B, 0);
  sb_set_element_shared_id(A, ra, 100); sb_set_element_shared_id(B, rb, 100);
  double pa[] = {0,0,20,0,20,20,0,20}, pb[] = {0,0,12,0,12,12,0,12};
  sb_set_element_polygon(A, ra, SB_GEOMETRY_NAVIGABLE, pa, 4);
  sb_set_element_polygon(B, rb, SB_GEOMETRY_NAVIGABLE, pb, 4);
  sb_element_t ba = sb_add_element(A, ra), bb = sb_add_element(B, rb);
  sb_set_element_shared_id(A, ba, 200); sb_set_element_shared_id(B, bb, 200);
  double m[16]; Tr(m, 5, 15); sb_set_element_transform(A, ba, m);
  Tr(m, 10, 3); sb_set_element_transform(B, bb, m);
  double qa[] = {3,13,7,13,7,17,3,17}, qb[] = {8,1,12,1,12,5,8,5};
  sb_set_element_polygon(A, ba, SB_GEOMETRY_CONTAINMENT, qa, 4);
  sb_set_element_polygon(B, bb, SB_GEOMETRY_CONTAINMENT, qb, 4);
  sb_actor_t a = sb_create_actor(A);
  sb_add_anchor(A, a, "root", 0,0,0, NULL); sb_add_anchor(A, a, "chest", 0,0,1.2, NULL);
  sb_set_actor_element(A, a, ra);
  sb_bind(A, a, "root", SB_KERNEL_SPACING); sb_bind(A, a, "chest", SB_KERNEL_FACING);
  sb_actor_t b = sb_create_actor(B);
  sb_add_anchor(B, b, "root", 0,0,0, NULL); sb_add_anchor(B, b, "chest", 0,0,1.2, NULL);
  sb_set_actor_element(B, b, rb);
  std::atomic<bool> done{false};
  std::atomic<long> bad{0}, solved{0};
  std::thread driver([&] {  // "Unity main thread": publish + read
    uint64_t epoch = 1, seq = 0; uint64_t last_rev = 0;
    for (int k = 0; k < 20000 && !done.load(); ++k) {
      if (k % 500 == 499) { ++epoch; seq = 0; }
      sb_anchor_pose_t p{}; p.struct_size = sizeof p;
      p.x = 1 + (k % 100) * 0.05; p.y = p.x; p.orientation.struct_size = sizeof(sb_orientation_t);
      p.orientation.yaw = 0.01 * (k % 50);
      p.epoch = epoch; p.flags = (seq == 0 && epoch > 1) ? SB_EMBODIED_TELEPORTED : 0;
      if (sb_publish_anchor_pose(B, b, 0, &p, k * 0.016, ++seq) != 0) ++bad;
      auto q = spatialblend_c_test::ReadProposals(B, b);
      if (q.rc != 0) ++bad;
      if (q.revision < last_rev) ++bad;
      last_rev = q.revision;
      if (q.status != SB_PROPOSAL_NONE && q.sample_valid && q.sample_epoch > epoch) ++bad;
    }
  });
  std::thread source([&] {  // "network/completion-queue thread": source updates
    for (int k = 0; k < 600; ++k) {
      if (k % 150 == 149) sb_raise_source_exception(A, a, 9, nullptr, 0);
      sb_update_actor_state(A, a, 2 + (k % 60) * 0.25, 2 + (k % 40) * 0.3, 0.02 * k);
      ++solved;
    }
    done = true;
  });
  std::thread reader([&] {
    while (!done.load()) {
      if (spatialblend_c_test::ReadProposals(B, b).rc != 0) ++bad;
      sb_actor_event_t ev{}; ev.struct_size = sizeof ev;
      if (sb_poll_actor_event(B, b, &ev) < 0) ++bad;
    }
  });
  source.join(); driver.join(); reader.join();
  EXPECT_EQ(bad.load(), 0);
  EXPECT_GT(solved.load(), 0);
  sb_destroy_session(s);
}

// A new owner takes the port (receiver-input Begin): the earlier writer's epoch
// and sequence must not make its frames stale.
TEST(PortOwnership, ResetLetsANewOwnerPublishFromScratch) {
  EmbodiedPort port;
  EmbodiedPoseInput p;
  p.epoch = 9;
  EXPECT_EQ(port.Publish(p, 1.0, 500, 1), EmbodiedPublishResult::kOk);
  p.epoch = 1;
  EXPECT_EQ(port.Publish(p, 2.0, 1, 2), EmbodiedPublishResult::kStale);
  port.Reset();
  EXPECT_FALSE(port.Sample().valid);
  EXPECT_EQ(port.Publish(p, 2.0, 1, 3), EmbodiedPublishResult::kOk);
  const EmbodiedSample s = port.Sample();
  EXPECT_TRUE(s.valid);
  EXPECT_EQ(s.epoch, 1u);
  EXPECT_EQ(s.sequence, 1u);
}
