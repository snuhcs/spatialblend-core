// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// The tunneled core link at the network layer: TunnelLink on its own, and a host
// that has no listening socket at all.

#include "spatialblend/net/tunnel_link.h"

#include <gtest/gtest.h>

#if !defined(_WIN32)
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "spatialblend/net/hub_receiver.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace {

using Clock = std::chrono::steady_clock;

std::vector<uint8_t> Pattern(size_t n, uint32_t seed) {
  std::vector<uint8_t> v(n);
  uint32_t x = seed * 2654435761u + 1;
  for (auto& b : v) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    b = static_cast<uint8_t>(x >> 9);
  }
  return v;
}

// What the core end of a link sees: the raw descriptor gRPC would own.
class FakeCore {
 public:
  explicit FakeCore(int fd) : fd_(fd) {}
  ~FakeCore() { Close(); }
  void Close() {
    if (fd_ >= 0) close(fd_);
    fd_ = -1;
  }
  // Writes all of `bytes`, waiting for room, for at most `timeout`.
  bool WriteAll(const std::vector<uint8_t>& bytes, std::chrono::seconds timeout) {
    size_t at = 0;
    const auto deadline = Clock::now() + timeout;
    while (at < bytes.size() && Clock::now() < deadline) {
      const ssize_t n = send(fd_, bytes.data() + at, bytes.size() - at, MSG_NOSIGNAL);
      if (n > 0) {
        at += static_cast<size_t>(n);
      } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      } else if (n < 0 && errno != EINTR) {
        return false;
      }
    }
    return at == bytes.size();
  }
  // Reads until `want` bytes arrived or `timeout` passed.
  std::vector<uint8_t> ReadUpTo(size_t want, std::chrono::seconds timeout) {
    std::vector<uint8_t> got;
    uint8_t buf[65536];
    const auto deadline = Clock::now() + timeout;
    while (got.size() < want && Clock::now() < deadline) {
      const ssize_t n = recv(fd_, buf, sizeof(buf), 0);
      if (n > 0) {
        got.insert(got.end(), buf, buf + n);
      } else if (n == 0) {
        break;
      } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      } else if (errno != EINTR) {
        break;
      }
    }
    return got;
  }

 private:
  int fd_;
};

TEST(TunnelLinkTest, BytesFromCoreArriveInOrderWhateverTheReadSize) {
  auto link = TunnelLink::Create();
  ASSERT_TRUE(link.ok()) << link.status();
  FakeCore core((*link)->ReleaseCoreFd());
  const auto payload = Pattern(3u << 20, 1);
  std::thread writer([&] { EXPECT_TRUE(core.WriteAll(payload, std::chrono::seconds(30))); });

  std::vector<uint8_t> got;
  uint8_t buf[777];  // an awkward size on purpose
  const auto deadline = Clock::now() + std::chrono::seconds(30);
  while (got.size() < payload.size() && Clock::now() < deadline) {
    size_t n = 0;
    ASSERT_EQ((*link)->Read(buf, sizeof(buf), &n), TunnelLink::ReadResult::kOk);
    got.insert(got.end(), buf, buf + n);
    if (n == 0) std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  writer.join();
  EXPECT_EQ(got, payload);
}

TEST(TunnelLinkTest, WritesApplyBackpressureInsteadOfBufferingWithoutBound) {
  auto link = TunnelLink::Create();
  ASSERT_TRUE(link.ok()) << link.status();
  FakeCore core((*link)->ReleaseCoreFd());
  // Far more than a socket buffer, with nobody reading the core end: Write must
  // never block, and once the socket is full it must say "would block" and
  // take nothing, so what the link holds stays bounded by the last accepted
  // chunk.
  const auto payload = Pattern(24u << 20, 2);
  constexpr size_t kChunk = 256 * 1024;
  size_t accepted = 0;
  bool would_block = false;
  const auto start = Clock::now();
  while (accepted < payload.size()) {
    const size_t n = std::min(kChunk, payload.size() - accepted);
    const absl::Status status = (*link)->Write(payload.data() + accepted, n);
    if (absl::IsUnavailable(status)) {
      would_block = true;
      break;
    }
    ASSERT_TRUE(status.ok()) << status;
    accepted += n;
  }
  EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - start).count(), 10)
      << "Write must not wait for the reader";
  ASSERT_TRUE(would_block) << "24 MiB were all accepted: the buffer is unbounded";
  EXPECT_LT(accepted, payload.size());
  // "Would block" took nothing: offering the same chunk again still blocks.
  EXPECT_TRUE(absl::IsUnavailable((*link)->Write(payload.data() + accepted, kChunk)));

  // The core end drains while the pump keeps offering the bytes it was refused,
  // and every byte arrives once, in order.
  std::vector<uint8_t> got;
  std::atomic<bool> reader_done{false};
  std::thread reader([&] {
    got = core.ReadUpTo(payload.size(), std::chrono::seconds(60));
    reader_done = true;
  });
  uint8_t scratch[64];
  const auto deadline = Clock::now() + std::chrono::seconds(60);
  while (!reader_done && Clock::now() < deadline) {
    size_t n = 0;
    (void)(*link)->Read(scratch, sizeof(scratch), &n);  // Read flushes
    if (accepted < payload.size()) {
      const size_t want = std::min(kChunk, payload.size() - accepted);
      const absl::Status status = (*link)->Write(payload.data() + accepted, want);
      if (status.ok()) {
        accepted += want;
        continue;
      }
      ASSERT_TRUE(absl::IsUnavailable(status)) << status;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  reader.join();
  EXPECT_EQ(got.size(), payload.size());
  EXPECT_EQ(got, payload);
}

TEST(TunnelLinkTest, WritingToADeadPeerFailsAndNeverRaisesSigpipe) {
  auto link = TunnelLink::Create();
  ASSERT_TRUE(link.ok());
  FakeCore core((*link)->ReleaseCoreFd());
  core.Close();  // the peer (gRPC) is gone
  const auto payload = Pattern(64 * 1024, 4);
  absl::Status status = absl::OkStatus();
  for (int i = 0; i < 20 && status.ok(); ++i) {
    status = (*link)->Write(payload.data(), payload.size());
  }
  EXPECT_TRUE(absl::IsFailedPrecondition(status)) << status;
  EXPECT_TRUE((*link)->IsClosed());
  size_t n = 0;
  uint8_t buf[8];
  EXPECT_EQ((*link)->Read(buf, sizeof(buf), &n), TunnelLink::ReadResult::kClosed);
}

// Close racing a writer and a reader, and the core end closing at the same
// time: no crash, no hang, no data race (run under TSAN), and every call after
// the close reports a closed link.
TEST(TunnelLinkTest, CloseRacingReadAndWriteIsSafe) {
  for (int round = 0; round < 50; ++round) {
    auto created = TunnelLink::Create();
    ASSERT_TRUE(created.ok());
    std::shared_ptr<TunnelLink> link(std::move(*created));
    auto core = std::make_shared<FakeCore>(link->ReleaseCoreFd());
    std::atomic<bool> go{false};
    const auto payload = Pattern(32 * 1024, round);
    std::thread writer([&] {
      while (!go) {}
      for (int i = 0; i < 200; ++i) {
        if (!link->Write(payload.data(), payload.size()).ok() && link->IsClosed()) break;
      }
    });
    std::thread reader([&] {
      uint8_t buf[512];
      while (!go) {}
      for (int i = 0; i < 2000; ++i) {
        size_t n = 0;
        if (link->Read(buf, sizeof(buf), &n) == TunnelLink::ReadResult::kClosed) break;
      }
    });
    std::thread closer([&] {
      while (!go) {}
      std::this_thread::sleep_for(std::chrono::microseconds(50 * (round % 7)));
      if (round % 2) core->Close();
      link->Close();
    });
    go = true;
    writer.join();
    reader.join();
    closer.join();
    EXPECT_TRUE(link->IsClosed());
    EXPECT_FALSE(link->Write(payload.data(), 1).ok());
  }
}

TEST(TunnelLinkTest, CoreClosingEndsTheLinkAfterItsBytesAreRead) {
  auto link = TunnelLink::Create();
  ASSERT_TRUE(link.ok());
  FakeCore core((*link)->ReleaseCoreFd());
  const auto payload = Pattern(1000, 3);
  ASSERT_TRUE(core.WriteAll(payload, std::chrono::seconds(5)));
  core.Close();

  std::vector<uint8_t> got;
  uint8_t buf[300];
  bool ended = false;
  for (int i = 0; i < 20 && !ended; ++i) {
    size_t n = 0;
    ended = (*link)->Read(buf, sizeof(buf), &n) == TunnelLink::ReadResult::kClosed;
    got.insert(got.end(), buf, buf + n);
  }
  EXPECT_TRUE(ended);
  EXPECT_EQ(got, payload) << "the bytes sent before the close are delivered first";
  EXPECT_TRUE((*link)->IsClosed());
  const uint8_t b = 0;
  EXPECT_FALSE((*link)->Write(&b, 1).ok());
}

TEST(TunnelLinkTest, ClosingTheLinkEndsTheCoreEnd) {
  auto link = TunnelLink::Create();
  ASSERT_TRUE(link.ok());
  FakeCore core((*link)->ReleaseCoreFd());
  EXPECT_EQ((*link)->ReleaseCoreFd(), -1) << "the core end is handed over once";
  (*link)->Close();
  (*link)->Close();  // idempotent
  EXPECT_TRUE((*link)->IsClosed());
  EXPECT_TRUE(core.ReadUpTo(1, std::chrono::seconds(2)).empty());  // end of stream, not data
  size_t n = 7;
  uint8_t buf[8];
  EXPECT_EQ((*link)->Read(buf, sizeof(buf), &n), TunnelLink::ReadResult::kClosed);
  EXPECT_EQ(n, 0u);
}

TEST(TunnelHostTest, ATunneledHostListensOnNothingAndAcceptsPeers) {
  Space space;
  space.SetSpaceId(1);
  SpaceHub hub;
  ASSERT_TRUE(hub.Connect(&space).ok());
  space.SetHub(&hub);
  auto receiver = std::make_shared<HubReceiver>(&space, &hub);

  // Not started: nothing to accept on.
  EXPECT_FALSE(receiver->AcceptTunnel().ok());
  ASSERT_TRUE(receiver->StartTunneled().ok());
  EXPECT_EQ(receiver->GetConnectedPort(), 0) << "a tunneled host has no TCP port";

  auto a = receiver->AcceptTunnel();
  auto b = receiver->AcceptTunnel();
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  EXPECT_NE(a->get(), b->get());
  // The server speaks first (its HTTP/2 SETTINGS), with nobody dialing it.
  uint8_t buf[256];
  size_t n = 0;
  const auto deadline = Clock::now() + std::chrono::seconds(10);
  while (Clock::now() < deadline && n == 0) {
    ASSERT_EQ((*a)->Read(buf, sizeof(buf), &n), TunnelLink::ReadResult::kOk);
    if (n == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  EXPECT_GT(n, 0u);

  receiver->Stop();
  EXPECT_FALSE(receiver->AcceptTunnel().ok()) << "a stopped host accepts nobody";
}

TEST(TunnelHostTest, ATcpHostRefusesTunnels) {
  Space space;
  space.SetSpaceId(1);
  SpaceHub hub;
  ASSERT_TRUE(hub.Connect(&space).ok());
  space.SetHub(&hub);
  auto receiver = std::make_shared<HubReceiver>(&space, &hub);
  ASSERT_TRUE(receiver->Start(0).ok());
  EXPECT_GT(receiver->GetConnectedPort(), 0);
  EXPECT_FALSE(receiver->AcceptTunnel().ok());
  receiver->Stop();
}

}  // namespace
}  // namespace spatialblend
