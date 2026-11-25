// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/base/common/seqlock_box.h
//
// A single-writer snapshot box for a trivially copyable value, read without any
// lock. The value is stored as relaxed atomic words under a version counter: the
// writer bumps the version to odd, stores the words, bumps it to even; a reader
// retries until it sees one even version across its whole read. A reader never
// blocks a writer and never takes a lock, so the solve (the writer) is never held
// up by a driver polling its proposal. Every shared access is atomic, so the type
// is data-race free under ThreadSanitizer.
#ifndef SPATIALBLEND_SPACE_SEQLOCK_BOX_H_
#define SPATIALBLEND_SPACE_SEQLOCK_BOX_H_

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <type_traits>

namespace spatialblend {

template <typename T>
class SeqlockBox {
  static_assert(std::is_trivially_copyable<T>::value,
                "SeqlockBox needs a trivially copyable value");

 public:
  // Writers are serialised (a mutex readers never take); one writer in practice.
  void Store(const T& value) {
    uint64_t buf[kWords] = {};
    std::memcpy(buf, &value, sizeof(T));
    std::lock_guard<std::mutex> lock(writer_mutex_);
    const uint64_t v = version_.load(std::memory_order_relaxed);
    version_.store(v + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    for (size_t i = 0; i < kWords; ++i)
      words_[i].store(buf[i], std::memory_order_relaxed);
    version_.store(v + 2, std::memory_order_release);
  }

  // False until the first Store.
  bool Load(T* out) const {
    uint64_t buf[kWords];
    for (;;) {
      const uint64_t v1 = version_.load(std::memory_order_acquire);
      if (v1 == 0) return false;
      if (v1 & 1u) continue;
      for (size_t i = 0; i < kWords; ++i)
        buf[i] = words_[i].load(std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_acquire);
      if (version_.load(std::memory_order_relaxed) == v1) break;
    }
    std::memcpy(out, buf, sizeof(T));
    return true;
  }

 private:
  static constexpr size_t kWords = (sizeof(T) + 7) / 8;
  std::mutex writer_mutex_;
  std::atomic<uint64_t> version_{0};
  std::atomic<uint64_t> words_[kWords] = {};
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_SEQLOCK_BOX_H_
