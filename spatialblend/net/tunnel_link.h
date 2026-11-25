// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_NETWORK_TUNNEL_LINK_H_
#define SPATIALBLEND_NETWORK_TUNNEL_LINK_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace spatialblend {

// One tunneled core link: a connected, bidirectional byte stream whose two ends
// are a socketpair. The "core end" is handed to gRPC (CreateInsecureChannelFromFd
// on a joining peer, AddInsecureChannelFromFd on the host server), so gRPC and
// HTTP/2 run unchanged. The "pump end" stays here. The application moves the
// bytes between the pump end and whatever carrier it has between the two peers
// (Fusion reliable data in the Unity app): no peer ever dials another peer's
// address.
//
//   Write(): bytes that arrived from the remote peer, into core.
//   Read():  bytes core wants delivered to the remote peer.
//
// Neither call blocks. A Write whose bytes the socket cannot take all at once
// keeps the rest, in order, and flushes it on the next Write()/Read(); keep
// calling Read() (the pump does, every frame). While such a remainder waits,
// Write() takes nothing and returns UnavailableError (would block): hold the
// bytes and offer them again after the next Read(). What the link holds is
// never more than the last chunk it accepted.
//
// POSIX only (Android, macOS, Linux). On Windows Create() returns
// UnimplementedError and nothing else in this class is reachable.
//
// Thread safety: all methods may be called from any thread.
class TunnelLink {
 public:
  static absl::StatusOr<std::unique_ptr<TunnelLink>> Create();
  ~TunnelLink();

  TunnelLink(const TunnelLink&) = delete;
  TunnelLink& operator=(const TunnelLink&) = delete;

  // Gives the core end to the caller (gRPC takes ownership). Returns -1 if it
  // was already released.
  int ReleaseCoreFd();

  // Bytes from the remote peer into core, all or nothing. Unavailable (would
  // block): nothing was taken, offer the same bytes again later.
  // FailedPrecondition once the link is closed (by Close() or because core
  // closed its end).
  absl::Status Write(const uint8_t* bytes, size_t n);

  enum class ReadResult {
    kOk,      // *n bytes were stored (possibly 0).
    kClosed,  // Core closed its end and everything it sent has been read.
  };
  // Bytes from core for the remote peer, at most `cap`.
  ReadResult Read(uint8_t* buf, size_t cap, size_t* n);

  // Closes the pump end. gRPC sees the end of the stream and fails whatever is
  // still using it. Idempotent.
  void Close();

  bool IsClosed() const;

 private:
  TunnelLink(int core_fd, int pump_fd);
  void FlushLocked();  // mu_ held

  mutable std::mutex mu_;
  int core_fd_;
  int pump_fd_;
  bool closed_ = false;
  std::vector<uint8_t> pending_;  // Bytes for core not yet accepted by the socket.
  size_t pending_offset_ = 0;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_NETWORK_TUNNEL_LINK_H_
