// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/bone_port.h
//
// The bone sampling port (ABI 4.7): the same single-ownership seqlock contract as
// embodied_port.h, for the pose of one BONE of a registered embodiment. The driver is the only writer; the core
// only samples the latest snapshot, never writes, integrates, extrapolates or substitutes one, and "no sample" is
// an explicit state. A bone pose carries its rotation as a quaternion because an anchor's offset is composed
// with the full rotation, which Euler angles on the anchor ports cannot carry without a lossy round trip.
#ifndef SPATIALBLEND_SPACE_BONE_PORT_H_
#define SPATIALBLEND_SPACE_BONE_PORT_H_

#include <atomic>
#include <cmath>
#include <cstdint>
#include <mutex>

#include "spatialblend/space/embodied_port.h"

namespace spatialblend {

struct BonePoseInput {
  double p[3] = {0, 0, 0};
  double q[4] = {0, 0, 0, 1};  // x, y, z, w; unit (the caller normalises)
  uint64_t epoch = 0;
  uint32_t flags = 0;
};

struct BoneSample {
  bool valid = false;
  double p[3] = {0, 0, 0};
  double q[4] = {0, 0, 0, 1};
  double timestamp_s = 0.0;
  uint64_t sequence = 0;
  uint64_t epoch = 0;
  uint32_t flags = 0;
  int64_t published_ns = 0;
};

class BonePort {
 public:
  EmbodiedPublishResult Publish(const BonePoseInput& pose, double timestamp_s, uint64_t sequence, int64_t now_ns) {
    if (!std::isfinite(timestamp_s) || sequence == 0) return EmbodiedPublishResult::kInvalid;
    for (double v : pose.p)
      if (!std::isfinite(v)) return EmbodiedPublishResult::kInvalid;
    for (double v : pose.q)
      if (!std::isfinite(v)) return EmbodiedPublishResult::kInvalid;
    std::lock_guard<std::mutex> lock(writer_mutex_);
    const uint64_t have_epoch = epoch_.load(std::memory_order_relaxed);
    const uint64_t have_seq = sequence_.load(std::memory_order_relaxed);
    const bool first = !has_.load(std::memory_order_relaxed);
    if (!first && (pose.epoch < have_epoch || (pose.epoch == have_epoch && sequence <= have_seq)))
      return EmbodiedPublishResult::kStale;
    const uint64_t v = version_.load(std::memory_order_relaxed);
    version_.store(v + 1, std::memory_order_relaxed);  // odd: write in progress
    std::atomic_thread_fence(std::memory_order_release);
    for (int i = 0; i < 3; ++i) p_[i].store(pose.p[i], std::memory_order_relaxed);
    for (int i = 0; i < 4; ++i) q_[i].store(pose.q[i], std::memory_order_relaxed);
    timestamp_.store(timestamp_s, std::memory_order_relaxed);
    sequence_.store(sequence, std::memory_order_relaxed);
    epoch_.store(pose.epoch, std::memory_order_relaxed);
    flags_.store(pose.flags, std::memory_order_relaxed);
    published_ns_.store(now_ns, std::memory_order_relaxed);
    has_.store(true, std::memory_order_relaxed);
    version_.store(v + 2, std::memory_order_release);  // even: consistent
    return EmbodiedPublishResult::kOk;
  }

  BoneSample Sample() const {
    for (;;) {
      const uint64_t v1 = version_.load(std::memory_order_acquire);
      if (v1 == 0) return BoneSample{};
      if (v1 & 1u) continue;
      BoneSample s;
      for (int i = 0; i < 3; ++i) s.p[i] = p_[i].load(std::memory_order_relaxed);
      for (int i = 0; i < 4; ++i) s.q[i] = q_[i].load(std::memory_order_relaxed);
      s.timestamp_s = timestamp_.load(std::memory_order_relaxed);
      s.sequence = sequence_.load(std::memory_order_relaxed);
      s.epoch = epoch_.load(std::memory_order_relaxed);
      s.flags = flags_.load(std::memory_order_relaxed);
      s.published_ns = published_ns_.load(std::memory_order_relaxed);
      const bool has = has_.load(std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_acquire);
      if (version_.load(std::memory_order_relaxed) == v1) {
        if (!has) return BoneSample{};
        s.valid = true;
        return s;
      }
    }
  }

 private:
  std::mutex writer_mutex_;
  std::atomic<uint64_t> version_{0};
  std::atomic<double> p_[3] = {0.0, 0.0, 0.0};
  std::atomic<double> q_[4] = {0.0, 0.0, 0.0, 1.0};
  std::atomic<double> timestamp_{0.0};
  std::atomic<uint64_t> sequence_{0}, epoch_{0};
  std::atomic<uint32_t> flags_{0};
  std::atomic<bool> has_{false};
  std::atomic<int64_t> published_ns_{0};
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_BONE_PORT_H_
