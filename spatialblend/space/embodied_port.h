// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/embodied_port.h
//
// The embodied-pose sampling port. ABI 4: one port per ANCHOR of a receiver
// actor.
//
// Single ownership. The receiver's embodied root pose belongs to the DRIVER
// (Unity, or the app). The driver is the only writer of this port; core only
// samples the latest snapshot when it needs the current state. Core never
// writes, integrates, extrapolates, smooths or replaces a sample, and never
// falls back to a pose of its own when the port holds none: "no sample" is an
// explicit state that reaches the proposal.
//
// Storage is a seqlock: the writer bumps a version to odd, stores the fields,
// and bumps it to even; a reader retries until it observes one even version
// across the whole read. Readers never block a writer and never take a lock,
// so a driver publishing from its main thread is never held up by a solve on
// a network thread, and a solve is never torn by a publish.
#ifndef SPATIALBLEND_SPACE_EMBODIED_PORT_H_
#define SPATIALBLEND_SPACE_EMBODIED_PORT_H_

#include <atomic>
#include <cmath>
#include <cstdint>
#include <mutex>

namespace spatialblend {

// Flags of a published pose (mirror SB_EMBODIED_* in spatialblend_type.h).
constexpr uint32_t kEmbodiedTeleported = 1u << 0;

// What the driver publishes.
struct EmbodiedPoseInput {
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;      // radians, Z-up
  uint64_t epoch = 0;    // bumps on a driver-side teleport / respawn / reset
  uint32_t flags = 0;
  // ABI 4: an anchor's own height and the orientation beyond yaw. The root's
  // port may leave them 0 (the root is planar).
  double z = 0.0;
  double pitch = 0.0;
  double roll = 0.0;
  int32_t orientation_type = 0;  // sb_orientation_type_t
};

// One snapshot, as sampled by core.
struct EmbodiedSample {
  bool valid = false;  // false = nothing has been published
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double timestamp_s = 0.0;   // the driver's clock (opaque to core)
  uint64_t sequence = 0;      // driver sequence, increasing within an epoch
  uint64_t epoch = 0;
  uint32_t flags = 0;
  double z = 0.0;
  double pitch = 0.0;
  double roll = 0.0;
  int32_t orientation_type = 0;
  // CLOCK_MONOTONIC-like reading of core's own steady clock taken inside
  // Publish(). Sample age is measured on this clock, so it is correct whatever
  // clock domain the driver's timestamp_s is in.
  int64_t published_ns = 0;
};

enum class EmbodiedPublishResult {
  kOk,
  kInvalid,  // non-finite value, or sequence 0
  kStale,    // sequence not newer within the epoch, or an older epoch
  kRefused,  // the actor's pose is owned by a receiver-input stream
};

class EmbodiedPort {
 public:
  // Writer side. Serialised among writers by a mutex that readers never take.
  EmbodiedPublishResult Publish(const EmbodiedPoseInput& pose,
                                double timestamp_s, uint64_t sequence,
                                int64_t now_ns) {
    if (!std::isfinite(pose.x) || !std::isfinite(pose.y) ||
        !std::isfinite(pose.yaw) || !std::isfinite(pose.z) ||
        !std::isfinite(pose.pitch) || !std::isfinite(pose.roll) || !std::isfinite(timestamp_s) ||
        sequence == 0)
      return EmbodiedPublishResult::kInvalid;
    std::lock_guard<std::mutex> lock(writer_mutex_);
    const uint64_t have_epoch = epoch_.load(std::memory_order_relaxed);
    const uint64_t have_seq = sequence_.load(std::memory_order_relaxed);
    const bool first = !has_.load(std::memory_order_relaxed);
    if (!first && (pose.epoch < have_epoch ||
                   (pose.epoch == have_epoch && sequence <= have_seq)))
      return EmbodiedPublishResult::kStale;
    const uint64_t v = version_.load(std::memory_order_relaxed);
    version_.store(v + 1, std::memory_order_relaxed);  // odd: write in progress
    std::atomic_thread_fence(std::memory_order_release);
    x_.store(pose.x, std::memory_order_relaxed);
    y_.store(pose.y, std::memory_order_relaxed);
    yaw_.store(pose.yaw, std::memory_order_relaxed);
    z_.store(pose.z, std::memory_order_relaxed);
    pitch_.store(pose.pitch, std::memory_order_relaxed);
    roll_.store(pose.roll, std::memory_order_relaxed);
    orientation_type_.store(pose.orientation_type, std::memory_order_relaxed);
    timestamp_.store(timestamp_s, std::memory_order_relaxed);
    sequence_.store(sequence, std::memory_order_relaxed);
    epoch_.store(pose.epoch, std::memory_order_relaxed);
    flags_.store(pose.flags, std::memory_order_relaxed);
    published_ns_.store(now_ns, std::memory_order_relaxed);
    has_.store(true, std::memory_order_relaxed);
    version_.store(v + 2, std::memory_order_release);  // even: consistent
    return EmbodiedPublishResult::kOk;
  }

  // Writer side: forget the stored sample (the port is empty again, so the next
  // publish is accepted whatever its epoch and sequence). Used when a new owner
  // takes the port (a receiver-input stream begins): the earlier writer's
  // epoch and sequence must not make the new owner's frames stale.
  void Reset() {
    std::lock_guard<std::mutex> lock(writer_mutex_);
    const uint64_t v = version_.load(std::memory_order_relaxed);
    version_.store(v + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    sequence_.store(0, std::memory_order_relaxed);
    epoch_.store(0, std::memory_order_relaxed);
    flags_.store(0, std::memory_order_relaxed);
    has_.store(false, std::memory_order_relaxed);
    version_.store(v + 2, std::memory_order_release);
  }

  // Reader side. Lock-free; retries only while a write is in flight.
  EmbodiedSample Sample() const {
    for (;;) {
      const uint64_t v1 = version_.load(std::memory_order_acquire);
      if (v1 == 0) return EmbodiedSample{};  // nothing published
      if (v1 & 1u) continue;
      EmbodiedSample s;
      s.x = x_.load(std::memory_order_relaxed);
      s.y = y_.load(std::memory_order_relaxed);
      s.yaw = yaw_.load(std::memory_order_relaxed);
      s.z = z_.load(std::memory_order_relaxed);
      s.pitch = pitch_.load(std::memory_order_relaxed);
      s.roll = roll_.load(std::memory_order_relaxed);
      s.orientation_type = orientation_type_.load(std::memory_order_relaxed);
      s.timestamp_s = timestamp_.load(std::memory_order_relaxed);
      s.sequence = sequence_.load(std::memory_order_relaxed);
      s.epoch = epoch_.load(std::memory_order_relaxed);
      s.flags = flags_.load(std::memory_order_relaxed);
      s.published_ns = published_ns_.load(std::memory_order_relaxed);
      const bool has = has_.load(std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_acquire);
      if (version_.load(std::memory_order_relaxed) == v1) {
        if (!has) return EmbodiedSample{};
        s.valid = true;
        return s;
      }
    }
  }

 private:
  std::mutex writer_mutex_;
  std::atomic<uint64_t> version_{0};
  std::atomic<double> x_{0.0}, y_{0.0}, yaw_{0.0}, timestamp_{0.0};
  std::atomic<double> z_{0.0}, pitch_{0.0}, roll_{0.0};
  std::atomic<int32_t> orientation_type_{0};
  std::atomic<uint64_t> sequence_{0}, epoch_{0};
  std::atomic<uint32_t> flags_{0};
  std::atomic<bool> has_{false};
  std::atomic<int64_t> published_ns_{0};
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_EMBODIED_PORT_H_
