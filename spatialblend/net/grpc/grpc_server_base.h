// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_BASE_NETWORK_GRPC_SERVER_BASE_H_
#define SPATIALBLEND_BASE_NETWORK_GRPC_SERVER_BASE_H_

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "absl/log/internal/globals.h"
#include "absl/status/status.h"
#include "spatialblend/net/util/network_util.h"

namespace spatialblend {

constexpr int kShutdownTimeoutMs = 2000;


template <typename CallType>
class GrpcServerBase {
 public:
  virtual ~GrpcServerBase() { Stop(); }
  absl::Status Start(int port, grpc::Service* service, size_t num_threads = 1,
                    int* bound_port = nullptr);
  // Builds the same server with no listening socket at all. Peers are then
  // attached one connection at a time through grpc::AddInsecureChannelFromFd
  // (the tunneled transport): nothing ever dials this process.
  absl::Status StartWithoutListening(grpc::Service* service,
                                     size_t num_threads = 1) {
    return StartImpl(/*listens=*/false, 0, service, num_threads, nullptr);
  }
  void Stop();

 protected:
  void HandleRpcs();
  void DrainCompletionQueue(grpc::ServerCompletionQueue* cq);
  absl::Status StartImpl(bool listens, int port, grpc::Service* service,
                         size_t num_threads, int* bound_port);

  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  std::vector<std::thread> server_threads_;
  std::atomic<bool> shutdown_{false};
};

template <typename CallType>
absl::Status GrpcServerBase<CallType>::Start(int port, grpc::Service* service,
                                             size_t num_threads,
                                             int* bound_port) {
  return StartImpl(/*listens=*/true, port, service, num_threads, bound_port);
}

template <typename CallType>
absl::Status GrpcServerBase<CallType>::StartImpl(bool listens, int port,
                                                 grpc::Service* service,
                                                 size_t num_threads,
                                                 int* bound_port) {
  if (listens && !network::util::IsValidPort(port)) {
    return absl::InvalidArgumentError(absl::StrFormat("Invalid port %d", port));
  }

#ifdef __ANDROID__
  // On Android, IsPortAvailable() is unreliable due to SELinux restrictions.
  // Instead of checking, we let gRPC try to bind and if it fails with the
  // specific port, we'll use port 0 to let OS assign an available port.
  // However, gRPC 1.66+ has an issue where dual-stack binding failure can
  // cause CHECK failures. To avoid this, if the port might be in use,
  // use port 0 directly to guarantee success.
  // Note: The caller should handle the case where bound_port differs from port.
#else
  if (listens && !network::util::IsPortAvailable(port)) {
    return absl::AlreadyExistsError(
        absl::StrFormat("Port %d is already in use", port));
  }
#endif

  grpc::ServerBuilder builder;
  const int kMaxMessageSize = 1024 * 1024 * 50; // 50MB
  builder.SetMaxReceiveMessageSize(kMaxMessageSize);
  builder.SetMaxSendMessageSize(kMaxMessageSize);
  int selected_port = port;
  if (listens) {
    builder.AddListeningPort(absl::StrFormat("0.0.0.0:%d", port),
                             grpc::InsecureServerCredentials(),
                             &selected_port);
  }
  builder.RegisterService(service);

  cq_ = builder.AddCompletionQueue();

  // Try-catch to prevent crashes within gRPC BuildAndStart
  server_ = builder.BuildAndStart();

  if (!server_) {
#ifdef __ANDROID__
    // On Android, if binding fails, it might be due to port conflict.
    // Return AlreadyExists to trigger retry logic in the caller.
    return absl::AlreadyExistsError(
        absl::StrFormat("Failed to bind port %d on Android", port));
#else
    return absl::InternalError("Failed to start gRPC server");
#endif
  }

  // A selected port of 0 means bind did not succeed for the requested port.
  // Treat it as a start failure so callers can retry deterministically.
  if (listens && selected_port == 0) {
    server_->Shutdown(std::chrono::system_clock::now());
    server_.reset();
    cq_.reset();
    return absl::AlreadyExistsError(
        absl::StrFormat("Failed to bind port %d", port));
  }

  if (bound_port) {
    *bound_port = listens ? selected_port : 0;
  }

  shutdown_ = false;
  for (size_t i = 0; i < num_threads; ++i) {
    server_threads_.emplace_back(&GrpcServerBase::HandleRpcs, this);
  }
  absl::log_internal::SetExitOnDFatal(true);

  return absl::OkStatus();
}

template <typename CallType>
void GrpcServerBase<CallType>::Stop() {
  // Prevent double-stop
  if (shutdown_.exchange(true)) {
    return;
  }

  if (server_) {
    // Give active RPCs (especially server-side streaming like Subscribe)
    // time to complete gracefully. This ensures proper TCP FIN exchange
    // and avoids leaving sockets in CLOSE_WAIT state.
    server_->Shutdown(std::chrono::system_clock::now() +
                      std::chrono::milliseconds(kShutdownTimeoutMs));
  }

  if (cq_) {
    cq_->Shutdown();
    // Do NOT drain here — HandleRpcs() threads drain the CQ via Proceed(ok)
    // calls. Draining here races with HandleRpcs() and incorrectly deletes
    // non-heap members (e.g. SubscribeAlarmTag is a member of
    // SubscribeReceiverCall, not a separate heap allocation).
  }

  for (auto& thread : server_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  server_.reset();
  cq_.reset();
}

template <typename CallType>
void GrpcServerBase<CallType>::HandleRpcs() {
  void* tag;
  bool ok;

  while (cq_->Next(&tag, &ok)) {
    // During shutdown, force ok=false so Proceed() does cleanup only without
    // spawning new listener calls (which would RequestXxx on a shut-down CQ
    // and assert-fail in grpc_cq_begin_op). shutdown_ is set before
    // cq_->Shutdown(), so any ok=true event dequeued after that point is
    // from an RPC that arrived just before shutdown — clean it up instead.
    if (shutdown_.load()) ok = false;
    static_cast<CallType*>(tag)->Proceed(ok);
  }
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_BASE_NETWORK_GRPC_SERVER_BASE_H_
