// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/net/tunnel_link.h"

#include <algorithm>
#include <cstring>

#if !defined(_WIN32)
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace spatialblend {

#if defined(_WIN32)

// grpc::{CreateInsecureChannelFromFd, AddInsecureChannelFromFd} do not exist on
// Windows (no GPR_SUPPORT_CHANNELS_FROM_FD). The targets of the tunneled link are
// Android (Quest) and the macOS Editor.
absl::StatusOr<std::unique_ptr<TunnelLink>> TunnelLink::Create() {
  return absl::UnimplementedError(
      "The tunneled core link needs POSIX sockets (Android, macOS, Linux); it is "
      "not available on Windows");
}
TunnelLink::TunnelLink(int core_fd, int pump_fd)
    : core_fd_(core_fd), pump_fd_(pump_fd) {}
TunnelLink::~TunnelLink() = default;
int TunnelLink::ReleaseCoreFd() { return -1; }
absl::Status TunnelLink::Write(const uint8_t*, size_t) {
  return absl::UnimplementedError("tunneled core link unavailable on Windows");
}
TunnelLink::ReadResult TunnelLink::Read(uint8_t*, size_t, size_t* n) {
  if (n) *n = 0;
  return ReadResult::kClosed;
}
void TunnelLink::Close() {}
bool TunnelLink::IsClosed() const { return true; }
void TunnelLink::FlushLocked() {}

#else  // POSIX

namespace {

#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;
#endif

bool SetCloexecNoSigpipe(int fd) {
  if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) return false;
#ifdef SO_NOSIGPIPE
  int on = 1;
  (void)setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on));
#endif
  return true;
}

}  // namespace

absl::StatusOr<std::unique_ptr<TunnelLink>> TunnelLink::Create() {
  int fds[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    return absl::InternalError(std::string("socketpair failed: ") +
                               std::strerror(errno));
  }
  if (!SetCloexecNoSigpipe(fds[0]) || !SetCloexecNoSigpipe(fds[1])) {
    close(fds[0]);
    close(fds[1]);
    return absl::InternalError("could not configure the link sockets");
  }
  // Both ends are non-blocking. The pump end is polled by the application and
  // never waited on. The core end is handed to gRPC, whose server side
  // (grpc_server_add_channel_from_fd) does not set O_NONBLOCK itself: a blocking
  // core end would park the server's completion-queue thread inside recv().
  for (int fd : fds) {
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
      close(fds[0]);
      close(fds[1]);
      return absl::InternalError("could not make the link sockets non-blocking");
    }
  }
  // A larger socket buffer only lets more of a multi-megabyte frame sit in the
  // kernel between two pumps; it changes no behaviour. Best effort.
  const int kBuf = 1 << 20;
  for (int fd : fds) {
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &kBuf, sizeof(kBuf));
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &kBuf, sizeof(kBuf));
  }
  return std::unique_ptr<TunnelLink>(new TunnelLink(fds[0], fds[1]));
}

TunnelLink::TunnelLink(int core_fd, int pump_fd)
    : core_fd_(core_fd), pump_fd_(pump_fd) {}

TunnelLink::~TunnelLink() {
  Close();
  std::lock_guard<std::mutex> lock(mu_);
  if (core_fd_ >= 0) {  // never handed to gRPC
    close(core_fd_);
    core_fd_ = -1;
  }
}

int TunnelLink::ReleaseCoreFd() {
  std::lock_guard<std::mutex> lock(mu_);
  const int fd = core_fd_;
  core_fd_ = -1;
  return fd;
}

void TunnelLink::FlushLocked() {
  while (!closed_ && pending_offset_ < pending_.size()) {
    const ssize_t sent =
        send(pump_fd_, pending_.data() + pending_offset_,
             pending_.size() - pending_offset_, kSendFlags);
    if (sent > 0) {
      pending_offset_ += static_cast<size_t>(sent);
      continue;
    }
    if (sent < 0 && errno == EINTR) continue;
    if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
    // EPIPE / ECONNRESET: core closed its end.
    closed_ = true;
    return;
  }
  if (pending_offset_ == pending_.size()) {
    pending_.clear();
    pending_offset_ = 0;
  }
}

absl::Status TunnelLink::Write(const uint8_t* bytes, size_t n) {
  std::lock_guard<std::mutex> lock(mu_);
  if (closed_) return absl::FailedPreconditionError("tunnel link is closed");
  if (n == 0) return absl::OkStatus();
  if (bytes == nullptr) return absl::InvalidArgumentError("null bytes");
  // Backpressure: while bytes already accepted still wait for the socket, take
  // nothing more. What waits is therefore never more than the one chunk the
  // caller last handed over, whatever the carrier does.
  FlushLocked();
  if (closed_) return absl::FailedPreconditionError("tunnel link is closed");
  if (pending_offset_ < pending_.size()) {
    return absl::UnavailableError("tunnel link would block");
  }
  size_t at = 0;
  while (at < n) {
    const ssize_t sent = send(pump_fd_, bytes + at, n - at, kSendFlags);
    if (sent > 0) {
      at += static_cast<size_t>(sent);
      continue;
    }
    if (sent < 0 && errno == EINTR) continue;
    if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
    closed_ = true;  // EPIPE / ECONNRESET: core closed its end.
    return absl::FailedPreconditionError("tunnel link is closed");
  }
  if (at < n) {
    pending_.assign(bytes + at, bytes + n);
    pending_offset_ = 0;
  }
  return absl::OkStatus();
}

TunnelLink::ReadResult TunnelLink::Read(uint8_t* buf, size_t cap, size_t* n) {
  size_t got = 0;
  std::lock_guard<std::mutex> lock(mu_);
  if (n) *n = 0;
  FlushLocked();
  if (pump_fd_ < 0) return ReadResult::kClosed;
  if (buf == nullptr || cap == 0) return closed_ ? ReadResult::kClosed : ReadResult::kOk;
  while (got < cap) {
    const ssize_t r = recv(pump_fd_, buf + got, cap - got, 0);
    if (r > 0) {
      got += static_cast<size_t>(r);
      continue;
    }
    if (r < 0 && errno == EINTR) continue;
    if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
    closed_ = true;  // r == 0: orderly end of stream; r < 0: reset.
    break;
  }
  if (n) *n = got;
  // Bytes read before the end of stream are delivered first.
  if (closed_ && got == 0) return ReadResult::kClosed;
  return ReadResult::kOk;
}

void TunnelLink::Close() {
  std::lock_guard<std::mutex> lock(mu_);
  closed_ = true;
  if (pump_fd_ >= 0) {
    close(pump_fd_);
    pump_fd_ = -1;
  }
  pending_.clear();
  pending_offset_ = 0;
}

bool TunnelLink::IsClosed() const {
  std::lock_guard<std::mutex> lock(mu_);
  return closed_;
}

#endif  // _WIN32

}  // namespace spatialblend
