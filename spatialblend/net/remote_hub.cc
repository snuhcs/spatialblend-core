// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/net/remote_hub.h"

#include <chrono>
#include <cinttypes>
#include <thread>
#include <unordered_set>

#include <grpcpp/create_channel_posix.h>

#include "absl/strings/str_format.h"
#include "spatialblend/base/util/logger.h"
#include "spatialblend/base/common/trace.h"
#include "spatialblend/model/element/element.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/net/proto_util.h"
#include "spatialblend/model/relation/relation_kernel.h"

namespace spatialblend {

// HubSender -----------------------------------------------------------------

HubSender::HubSender(Space* space, space_id_t space_id)
    : space_(space), space_id_(space_id) {}

HubSender::~HubSender() {
  // Explicitly disconnect BEFORE any of HubSender's own members start
  // destructing. C++ destroys a derived object's own members (stub_ in
  // particular) before running the base class destructor -- and
  // Disconnect() (cancel pending calls, shut down the completion queue,
  // join the CQ thread) only runs from ~GrpcClientBase(). Relying solely on
  // that base-class destructor would destroy stub_ (and, once its members
  // are gone, race the still-running CQ thread) out from under a live,
  // uncancelled Subscribe stream whenever a RemoteHub/HubSender is dropped
  // without an explicit prior PrepareDisconnect() call (e.g. sb_cleanup_all,
  // or this stress test's abrupt-drop path). Disconnect() is idempotent, so
  // this is a no-op if PrepareDisconnect() already ran.
  Disconnect().IgnoreError();
}

absl::Status HubSender::Disconnect() {
  // Cancel first. Safe/idempotent -- GrpcClientBase<>::Disconnect() (called
  // below) will TryCancel() the same contexts again, which is a harmless
  // no-op the second time.
  {
    std::lock_guard<std::mutex> lock(pending_calls_mutex_);
    for (auto* call : pending_calls_) {
      call->context_.TryCancel();
    }
  }

  // Give the CQ thread (still running -- HandleRpcs() joins later, inside
  // the base Disconnect() below) a bounded window to actually deliver the
  // resulting cancellation completions and retire each pending call (i.e.
  // reach its Finish() call and the completion after that, then self-delete,
  // which removes it from pending_calls_ via ~GrpcClientCallBase()). Only
  // once no pending calls remain is it safe to let the base class shut down
  // cq_ -- otherwise a call's Finish() (a new async op) can race a concurrent
  // cq_->Shutdown() and CHECK-fail inside grpc-core.
  //
  // IMPORTANT: this drain only actually works if `this` is reached through a
  // *live* shared_ptr (i.e. HubSender's own strong refcount is still >= 1) --
  // see RemoteHub::~RemoteHub() for why: every in-flight call only holds a
  // weak_ptr back to us (GrpcClientCallBase::client_), so a call's self-delete
  // racing this drain can only remove itself from pending_calls_ (via
  // RemovePendingCall(), which needs client_.lock() to succeed) while that
  // refcount hasn't yet dropped to zero.
  constexpr auto kDrainTimeout = std::chrono::milliseconds(2000);
  auto deadline = std::chrono::steady_clock::now() + kDrainTimeout;
  for (;;) {
    {
      std::lock_guard<std::mutex> lock(pending_calls_mutex_);
      if (pending_calls_.empty()) break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      SPATIALBLEND_LOG(spatialblend::LogSeverity::kError,
                    "HubSender::Disconnect - timed out waiting for pending "
                    "calls to drain; proceeding to shut down the completion "
                    "queue anyway");
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }

  return spatialblend::GrpcClientBase<AsyncHubSenderCall>::Disconnect();
}

absl::Status HubSender::OnConnect() {
  stub_ = proto::HubService::NewStub(channel_);
  if (!stub_) {
    return absl::FailedPreconditionError("Failed to create stub");
  }

  // Synchronous ConnectToHub: get bootstrap payload.
  grpc::ClientContext ctx;
  proto::ConnectRequest request;
  request.set_space_id(space_id_);

  proto::BootstrapPayload bootstrap_proto;
  grpc::Status rpc_status =
      stub_->ConnectToHub(&ctx, request, &bootstrap_proto);
  if (!rpc_status.ok()) {
    return absl::InternalError(
        absl::StrFormat("ConnectToHub failed: %s", rpc_status.error_message()));
  }

  // The host names this peer. Every later call (Subscribe, CompleteBootstrap,
  // shares, updates) carries that id, not the one proposed above.
  if (bootstrap_proto.assigned_space_id() > 0) {
    assigned_space_id_ = bootstrap_proto.assigned_space_id();
    space_id_.store(assigned_space_id_);
  }

  if (stage_bootstrap_.load()) {
    // A tunneled join runs on its own thread, which must not touch the space:
    // the caller applies the bootstrap on the application's thread.
    std::lock_guard<std::mutex> lock(bootstrap_mutex_);
    staged_bootstrap_ = std::move(bootstrap_proto);
    has_staged_bootstrap_ = true;
    return absl::OkStatus();
  }
  return ApplyBootstrap(bootstrap_proto);
}

absl::Status HubSender::ApplyStagedBootstrap() {
  proto::BootstrapPayload bootstrap_proto;
  {
    std::lock_guard<std::mutex> lock(bootstrap_mutex_);
    if (!has_staged_bootstrap_) {
      return absl::FailedPreconditionError("no staged bootstrap to apply");
    }
    bootstrap_proto = std::move(staged_bootstrap_);
    has_staged_bootstrap_ = false;
  }
  return ApplyBootstrap(bootstrap_proto);
}

absl::Status HubSender::ApplyBootstrap(
    const proto::BootstrapPayload& bootstrap_proto) {
  // Apply bootstrap: create elements (without geometry) in local space, through
  // the space's own locked entry points.
  if (space_) {
    // The space takes the identity the host gave it before it sends or applies
    // anything as a peer of this room (this runs on the thread that may change
    // the space), and its trace binding follows it.
    if (assigned_space_id_ > 0) {
      const space_id_t previous = space_->GetSpaceId();
      if (previous != assigned_space_id_) {
        space_->SetSpaceId(assigned_space_id_);
        TraceSink::MoveSpaceBinding(previous, assigned_space_id_);
        space_->OnTraceRebound();
      }
    }
    std::unordered_set<element_id_t> created;
    for (const auto& ep : bootstrap_proto.elements()) {
      if (space_->AddElement(ep.element_id(), ep.parent_id())) {
        created.insert(ep.element_id());
        space_->SetElementSharedId(ep.element_id(), ep.shared_id());
        space_->SetElementTag(ep.element_id(), ep.tag());
      }
    }

    // Create actors from bootstrap.
    for (const auto& ap : bootstrap_proto.actors()) {
      std::vector<SubjectAnchor> anchors;
      for (const auto& anp : ap.anchors()) {
        anchors.emplace_back(anp.name(),
                             Eigen::Vector3d(anp.x(), anp.y(), anp.z()));
      }
      spatialblend::entity_id_t actor_id = space_->CreateActor(anchors);

      // Link actor to element by shared_id.
      const element_id_t elem = space_->FindElementBySharedId(ap.actor_shared_id());
      if (elem != kInvalidElementId) space_->SetActorElement(actor_id, elem);
    }

    // Create bindings from bootstrap.
    for (const auto& bp : bootstrap_proto.bindings()) {
      // Find local actor by shared_id.
      spatialblend::entity_id_t actor_id =
          space_->FindActorBySharedId(bp.actor_shared_id());
      if (actor_id == spatialblend::kInvalidEntityId) continue;

      space_->Bind(actor_id, bp.anchor_name(),
                   static_cast<RelationKernelType>(bp.kernel_type()),
                   bp.tag_filter());
    }

    // Last: the shares the host made before this peer subscribed, each received
    // as a live share would have been, onto its element, which by now exists
    // with its actor and bindings.
    for (const auto& ep : bootstrap_proto.elements()) {
      // Only onto an element this bootstrap created: an id the joiner already
      // had names its own, unrelated element.
      if (created.count(ep.element_id()) == 0) continue;
      for (const auto& share : ep.shares()) {
        space_->AdoptBootstrapShare(ep.element_id(), FromProto(share));
      }
    }
  }

  // Open the Subscribe stream so host sends relation updates.
  auto* sub_call = new SubscribeStreamCall(shared_from_this(), space_);

  proto::SubscribeRequest sub_request;
  sub_request.set_space_id(space_id_);
  sub_call->reader_ = stub_->PrepareAsyncSubscribe(
      &sub_call->context_, sub_request, cq_.get());
  sub_call->reader_->StartCall(static_cast<void*>(sub_call));

  // Send CompleteBootstrap back to host.
  auto* complete_call = new CompleteBootstrapSenderCall(
      shared_from_this(), space_id_, true);
  complete_call->on_complete_ = [this](bool success) {
    {
      std::lock_guard<std::mutex> lock(bootstrap_mutex_);
      bootstrap_status_ = success
                              ? absl::OkStatus()
                              : absl::InternalError("Bootstrap failed");
      bootstrap_complete_ = true;
    }
    bootstrap_cv_.notify_all();
  };

  auto complete_rpc = stub_->PrepareAsyncCompleteBootstrap(
      &complete_call->context_, complete_call->request_, cq_.get());
  complete_rpc->StartCall();
  complete_rpc->Finish(&complete_call->response_, &complete_call->status_,
                       static_cast<void*>(complete_call));

  return absl::OkStatus();
}

absl::Status HubSender::WaitForBootstrapComplete(int timeout_ms) {
  std::unique_lock<std::mutex> lock(bootstrap_mutex_);
  if (timeout_ms < 0) {
    // No timer: the bootstrap completes, or the link ends and the
    // CompleteBootstrap call fails.
    bootstrap_cv_.wait(lock, [this] { return bootstrap_complete_.load(); });
    return bootstrap_status_;
  }
  bool completed = bootstrap_cv_.wait_for(
      lock, std::chrono::milliseconds(timeout_ms),
      [this] { return bootstrap_complete_.load(); });
  if (!completed) {
    return absl::DeadlineExceededError(
        absl::StrFormat("Bootstrap timed out after %d ms", timeout_ms));
  }
  return bootstrap_status_;
}

absl::Status HubSender::PrepareDisconnect() {
  if (stub_ && IsConnected()) {
    // Best-effort notify: tell the host we're leaving cleanly (so it can
    // drop our RemoteSpace immediately instead of waiting to notice via a
    // stream failure). Fired asynchronously via cq_ and NOT waited on here --
    // a previous version issued this as a *synchronous* blocking RPC with its
    // own deadline, which meant that whenever the host was already gone (a
    // concurrent host-side teardown, or any other reason the connection is
    // already dead) this call stalled for its *entire* deadline before
    // Disconnect() below even started. Repeated many times (e.g. a
    // connect/disconnect stress loop), that turned an otherwise-fast teardown
    // into a multi-minute one. Disconnect() below already TryCancel()s +
    // bounded-waits-for-drain on every pending call, so this one is retired
    // the same way as any other in-flight call -- no separate wait needed.
    auto* notify_call =
        new DisconnectNotifySenderCall(shared_from_this(), space_id_);
    auto rpc = stub_->PrepareAsyncDisconnectFromHub(
        &notify_call->context_, notify_call->request_, cq_.get());
    rpc->StartCall();
    rpc->Finish(&notify_call->response_, &notify_call->status_,
               static_cast<void*>(notify_call));

    RETURN_IF_ERROR(Disconnect());
    stub_.reset();
    return absl::OkStatus();
  }
  return Disconnect();
}

// RemoteHub ------------------------------------------------------------------

RemoteHub::RemoteHub(Space* space, std::shared_ptr<HubSender> sender)
    : space_(space), sender_(std::move(sender)) {}

RemoteHub::~RemoteHub() {
  // Explicitly disconnect the sender HERE, while `sender_` (a shared_ptr)
  // still holds a live strong reference (refcount >= 1), rather than relying
  // solely on ~HubSender() to do it once `sender_`'s own destructor runs
  // below as part of this object's implicit member teardown.
  //
  // HubSender derives from std::enable_shared_from_this (via IGrpcClient),
  // and every in-flight call object (SubscribeStreamCall, the unary
  // *SenderCall types) stores only a *weak_ptr* back to it
  // (GrpcClientCallBase::client_) -- not a strong one. The instant a
  // shared_ptr<HubSender>'s strong refcount reaches zero, EVERY subsequent
  // weak_ptr::lock() against it fails (returns nullptr) for the rest of that
  // object's destruction -- even though the HubSender is still physically
  // alive, mid-destructor. That means if `sender_`'s implicit destructor is
  // what first drops the refcount to zero, any call that happens to
  // self-delete concurrently on the CQ thread from that point on can no
  // longer call RemovePendingCall() (~GrpcClientCallBase() silently no-ops),
  // so pending_calls_ can leak entries for objects that have *already been
  // freed*. HubSender::Disconnect()'s drain-wait loop then times out waiting
  // on a set that will never actually empty, and the base
  // GrpcClientBase<>::Disconnect() that runs afterwards proceeds to
  // TryCancel() those same (dangling, already-freed) call pointers --
  // dereferencing freed memory (observed crash: "Check failed:
  // pthread_mutex_lock(mu) == 0 (22 vs 0)" inside ClientContext::TryCancel(),
  // called on a ClientContext whose memory had already been released).
  //
  // Calling sender_->Disconnect() explicitly here, before any member of this
  // object (in particular sender_ itself) starts destructing, means the
  // refcount is still >= 1 throughout the whole drain: every concurrent
  // call's self-delete can still successfully call RemovePendingCall(), the
  // drain genuinely empties pending_calls_, and cq_->Shutdown() is only
  // reached once nothing is left in flight. The later, implicit destruction
  // of `sender_` (which still runs ~HubSender() -> Disconnect() once more,
  // for defense in depth / other call sites that might drop the last
  // reference without going through RemoteHub) then finds an already-empty,
  // already-shut-down connection and is a fast no-op.
  if (sender_) {
    sender_->Disconnect().IgnoreError();
  }
}

std::shared_ptr<RemoteHub> RemoteHub::Create(
    Space* space, const std::string& host_address, int host_port,
    space_id_t space_id, int bootstrap_timeout_ms) {
  auto sender = std::make_shared<HubSender>(space, space_id);
  if (auto status = sender->Connect(host_address, host_port); !status.ok()) {
    SPATIALBLEND_LOG(spatialblend::LogSeverity::kError,
                  "Failed to connect to host: %s",
                  status.ToString().c_str());
    return nullptr;
  }

  if (auto status = sender->WaitForBootstrapComplete(bootstrap_timeout_ms);
      !status.ok()) {
    SPATIALBLEND_LOG(spatialblend::LogSeverity::kError, "Bootstrap failed: %s",
                  status.ToString().c_str());
    return nullptr;
  }

  return std::shared_ptr<RemoteHub>(
      new RemoteHub(space, std::move(sender)));
}

std::shared_ptr<RemoteHub> RemoteHub::ConnectOnChannel(
    Space* space, std::shared_ptr<grpc::Channel> channel, space_id_t space_id) {
  auto sender = std::make_shared<HubSender>(space, space_id);
  sender->SetTunneled(true);
  sender->SetStageBootstrap(true);
  if (auto status = sender->ConnectChannel(std::move(channel)); !status.ok()) {
    SPATIALBLEND_LOG(spatialblend::LogSeverity::kError,
                  "Failed to join over the tunnel: %s",
                  status.ToString().c_str());
    return nullptr;
  }
  return std::shared_ptr<RemoteHub>(new RemoteHub(space, std::move(sender)));
}

absl::Status RemoteHub::ApplyStagedBootstrap() {
  return sender_->ApplyStagedBootstrap();
}

absl::Status RemoteHub::WaitForBootstrapComplete(int timeout_ms) {
  return sender_->WaitForBootstrapComplete(timeout_ms);
}

std::shared_ptr<RemoteHub> RemoteHub::CreateOnChannel(
    Space* space, std::shared_ptr<grpc::Channel> channel, space_id_t space_id,
    int bootstrap_timeout_ms) {
  auto hub = ConnectOnChannel(space, std::move(channel), space_id);
  if (!hub) return nullptr;
  if (auto status = hub->ApplyStagedBootstrap(); !status.ok()) {
    SPATIALBLEND_LOG(spatialblend::LogSeverity::kError, "Bootstrap failed: %s",
                  status.ToString().c_str());
    return nullptr;
  }
  if (auto status = hub->WaitForBootstrapComplete(bootstrap_timeout_ms);
      !status.ok()) {
    SPATIALBLEND_LOG(spatialblend::LogSeverity::kError, "Bootstrap failed: %s",
                  status.ToString().c_str());
    return nullptr;
  }
  return hub;
}

std::shared_ptr<grpc::Channel> RemoteHub::MakeTunnelChannel(TunnelLink* link) {
#if defined(GPR_SUPPORT_CHANNELS_FROM_FD)
  if (link == nullptr) return nullptr;
  const int fd = link->ReleaseCoreFd();
  if (fd < 0) return nullptr;
  // The host server accepts messages up to 50 MB (GrpcServerBase); the channel
  // that receives from it must accept the same, since a shared object carries
  // its scan (GLB) as user_data.
  grpc::ChannelArguments args;
  constexpr int kMaxMessageSize = 1024 * 1024 * 50;
  args.SetMaxReceiveMessageSize(kMaxMessageSize);
  args.SetMaxSendMessageSize(kMaxMessageSize);
  // "tunnel" is only a label: nothing resolves or dials it.
  return grpc::CreateCustomInsecureChannelFromFd("tunnel", fd, args);
#else
  (void)link;
  return nullptr;
#endif
}

absl::Status RemoteHub::PrepareDisconnect() {
  if (sender_ && sender_->IsConnected()) {
    return sender_->PrepareDisconnect();
  }
  return absl::OkStatus();
}

void RemoteHub::SendUpdateActorState(int64_t actor_id, int64_t actor_shared_id,
                                     double x, double y, double yaw) {
  if (!sender_ || !sender_->IsConnected()) return;

  auto* call = new UpdateActorStateSenderCall(
      sender_->shared_from_this(), sender_->space_id_, actor_id,
      actor_shared_id, x, y, yaw);
  auto rpc = sender_->stub_->PrepareAsyncUpdateActorState(
      &call->context_, call->request_, sender_->cq_.get());
  rpc->StartCall();
  rpc->Finish(&call->response_, &call->status_, static_cast<void*>(call));
}

void RemoteHub::SendElementEvent(const ElementEvent& event) {
  if (!sender_ || !sender_->IsConnected()) return;

  auto* call = new ShareElementSenderCall(
      sender_->shared_from_this(), sender_->space_id_, event);
  auto rpc = sender_->stub_->PrepareAsyncShareElement(
      &call->context_, call->request_, sender_->cq_.get());
  rpc->StartCall();
  rpc->Finish(&call->response_, &call->status_, static_cast<void*>(call));
}

void RemoteHub::SendRelationUpdate(const RemoteRelationUpdate& update) {
  if (!sender_ || !sender_->IsConnected()) return;

  auto* call = new PropagateRelationSenderCall(
      sender_->shared_from_this(), sender_->space_id_, update);
  auto rpc = sender_->stub_->PrepareAsyncPropagateRelation(
      &call->context_, call->request_, sender_->cq_.get());
  rpc->StartCall();
  rpc->Finish(&call->response_, &call->status_, static_cast<void*>(call));
}

bool RemoteHub::IsConnected() const {
  return sender_ && sender_->IsConnected();
}

}  // namespace spatialblend
