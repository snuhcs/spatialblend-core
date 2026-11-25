// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_BASE_NETWORK_GRPC_CLIENT_BASE_H_
#define SPATIALBLEND_BASE_NETWORK_GRPC_CLIENT_BASE_H_

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "absl/status/status.h"
#include "spatialblend/net/util/network_util.h"
#include "spatialblend/base/util/logger.h"
#include "spatialblend/base/util/util.h"

namespace spatialblend {

struct GrpcClientCallBase;

namespace internal {
// Interface for managing pending gRPC calls
class IGrpcClient : public std::enable_shared_from_this<IGrpcClient> {
 public:
  virtual ~IGrpcClient() = default;

  virtual void AddPendingCall(GrpcClientCallBase* call) = 0;
  virtual void RemovePendingCall(GrpcClientCallBase* call) = 0;

  // Whether the calls this client issues carry the fixed per-call deadline
  // (kDefaultTimeoutMs). A client whose link is carried by the application at
  // its own pace (the tunneled link) answers false: a deadline sized for a TCP
  // round trip cancels any message the carrier cannot move that fast, and the
  // link ending is what reports a lost peer.
  virtual bool CallsHaveDeadline() const { return true; }
};
}  // namespace internal

struct GrpcClientCallBase {
  GrpcClientCallBase(std::shared_ptr<internal::IGrpcClient> client)
      : client_(client) {
    if (auto client_ptr = client_.lock()) {
      client_ptr->AddPendingCall(this);
    }
  }
  virtual ~GrpcClientCallBase() {
    if (auto client_ptr = client_.lock()) {
      client_ptr->RemovePendingCall(this);
    }
  }
  virtual void Proceed(bool ok) = 0;

  grpc::ClientContext context_;
  grpc::Status status_;
  std::weak_ptr<internal::IGrpcClient> client_;
};

template <typename CallType>
class GrpcClientBase : public internal::IGrpcClient {
  static_assert(std::is_base_of<GrpcClientCallBase, CallType>::value,
                "CallType must be derived from GrpcClientCallBase");

 public:
  GrpcClientBase() = default;
  virtual ~GrpcClientBase() { LOG_IF_ERROR(Disconnect()); }

  absl::Status Connect(const std::string& address, int port);
  // Connects over an already-built channel (the tunneled transport hands over
  // a channel made from a socketpair fd). Same lifecycle as Connect().
  absl::Status ConnectChannel(std::shared_ptr<grpc::Channel> channel);
  absl::Status Disconnect();
  bool IsConnected() const;

 protected:
  friend struct GrpcClientCallBase;

  // Override IGrpcClient interface methods
  void AddPendingCall(GrpcClientCallBase* call) override {
    std::lock_guard<std::mutex> lock(pending_calls_mutex_);
    pending_calls_.insert(call);
  }

  void RemovePendingCall(GrpcClientCallBase* call) override {
    std::lock_guard<std::mutex> lock(pending_calls_mutex_);
    pending_calls_.erase(call);
  }

  std::mutex pending_calls_mutex_;
  std::unordered_set<GrpcClientCallBase*> pending_calls_;

  virtual absl::Status OnConnect() { return absl::OkStatus(); }
  void HandleRpcs();

  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<grpc::CompletionQueue> cq_;
  std::thread async_thread_;
  std::atomic<bool> shutdown_{false};
  std::atomic<bool> connected_{false};
};

template <typename CallType>
absl::Status GrpcClientBase<CallType>::Connect(const std::string& address,
                                               int port) {
  if (IsConnected()) {
    return absl::AlreadyExistsError("Already connected");
  }

  if (!network::util::IsValidHostAddress(address)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid or unreachable host address %s", address.c_str()));
  }

  if (!network::util::IsValidPort(port)) {
    return absl::InvalidArgumentError(absl::StrFormat("Invalid port %d", port));
  }

  return ConnectChannel(grpc::CreateChannel(
      address + ":" + std::to_string(port), grpc::InsecureChannelCredentials()));
}

template <typename CallType>
absl::Status GrpcClientBase<CallType>::ConnectChannel(
    std::shared_ptr<grpc::Channel> channel) {
  if (IsConnected()) {
    return absl::AlreadyExistsError("Already connected");
  }
  if (!channel) {
    return absl::InvalidArgumentError("Null channel");
  }

  channel_ = std::move(channel);
  cq_ = std::make_unique<grpc::CompletionQueue>();

  // A thread must be blocked in Next() before any async batch is started.
  // OnConnect() issues Subscribe and CompleteBootstrap on this queue. If those
  // batches are queued first, grpc does not reliably flush the unary call
  // until Next() runs, and Next() then blocks on the streaming call's initial
  // metadata. With one server CQ thread that metadata waits on the first
  // pushed write, so CompleteBootstrap misses its 1s deadline and the host
  // only handles it afterwards.
  shutdown_ = false;
  connected_ = false;
  async_thread_ = std::thread(&GrpcClientBase::HandleRpcs, this);

  absl::Status status = OnConnect();
  if (!status.ok()) {
    shutdown_ = true;
    if (cq_) {
      cq_->Shutdown();
    }
    if (async_thread_.joinable()) {
      async_thread_.join();
    }
    channel_.reset();
    cq_.reset();
    return status;
  }

  connected_ = true;
  return absl::OkStatus();
}

template <typename CallType>
absl::Status GrpcClientBase<CallType>::Disconnect() {
  shutdown_ = true;
  connected_ = false;

  {
    std::lock_guard<std::mutex> lock(pending_calls_mutex_);
    for (auto* call : pending_calls_) {
      call->context_.TryCancel();
    }
    pending_calls_.clear();
  }

  if (cq_) {
    cq_->Shutdown();
  }

  if (async_thread_.joinable()) {
    async_thread_.join();
  }

  channel_.reset();
  cq_.reset();
  return absl::OkStatus();
}

template <typename CallType>
bool GrpcClientBase<CallType>::IsConnected() const {
  return channel_ != nullptr;
}

template <typename CallType>
void GrpcClientBase<CallType>::HandleRpcs() {
  void* got_tag;
  bool ok = false;

  while (cq_->Next(&got_tag, &ok)) {
    if (!ok || shutdown_.load()) {
      static_cast<CallType*>(got_tag)->Proceed(false);
      continue;
    }
    static_cast<CallType*>(got_tag)->Proceed(ok);
  }
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_BASE_NETWORK_GRPC_CLIENT_BASE_H_