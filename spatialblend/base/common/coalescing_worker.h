// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_COMMON_COALESCING_WORKER_H_
#define SPATIALBLEND_COMMON_COALESCING_WORKER_H_

// A background task that runs when requested, latest-wins: requests that arrive
// while it runs collapse into ONE further run (the task reads the latest state
// itself, so the intermediate states are never computed). The thread model is
// ThreadWorker's (one owned std::thread, a condition variable, joined on
// destruction); it starts on the first request, so an owner that never needs it
// costs no thread.
//
// Generations: Request() returns the generation that covers it; a run started
// after a request covers every request made before the run began. Completed()
// is the latest covered generation, so WaitFor(Request()) waits until the state
// at the time of the request has been processed.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace spatialblend {

class CoalescingWorker {
 public:
  explicit CoalescingWorker(std::function<void()> task) : task_(std::move(task)) {}
  ~CoalescingWorker() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      exit_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }
  CoalescingWorker(const CoalescingWorker&) = delete;
  CoalescingWorker& operator=(const CoalescingWorker&) = delete;

  // Never blocks on the task: safe to call while holding a lock the task takes.
  uint64_t Request() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (exit_) return requested_;
    ++requested_;
    if (!thread_.joinable()) thread_ = std::thread(&CoalescingWorker::Loop, this);
    cv_.notify_all();
    return requested_;
  }

  // Whether generation `generation` completed within `timeout`. Must not be
  // called while holding a lock the task takes.
  bool WaitFor(uint64_t generation, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [&] { return completed_ >= generation || exit_; }) &&
           completed_ >= generation;
  }
  // Every request made so far completed within `timeout`.
  bool WaitIdle(std::chrono::milliseconds timeout) {
    uint64_t target;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      target = requested_;
    }
    return WaitFor(target, timeout);
  }

  uint64_t Completed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_;
  }
  // How many times the task ran (fewer than requests when bursts coalesce).
  uint64_t Runs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return runs_;
  }

 private:
  void Loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      cv_.wait(lock, [&] { return exit_ || requested_ > completed_; });
      if (exit_) return;
      const uint64_t target = requested_;  // this run covers every request so far
      lock.unlock();
      task_();
      lock.lock();
      completed_ = target;
      ++runs_;
      cv_.notify_all();
    }
  }

  std::function<void()> task_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
  uint64_t requested_ = 0;
  uint64_t completed_ = 0;
  uint64_t runs_ = 0;
  bool exit_ = false;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_COMMON_COALESCING_WORKER_H_
