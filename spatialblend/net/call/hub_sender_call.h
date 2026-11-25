// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_NETWORK_CALL_HUB_SENDER_CALL_H_
#define SPATIALBLEND_NETWORK_CALL_HUB_SENDER_CALL_H_

#include <grpcpp/grpcpp.h>

#include <memory>

#include "spatialblend/net/grpc/grpc_client_base.h"
#include "spatialblend/net/proto/spatialblend.grpc.pb.h"
#include "spatialblend/net/proto_util.h"
#include "spatialblend/space/space.h"

// The protobuf package keeps its pre-1.0 name because it is on the wire (gRPC method paths, Any type
// URLs): spatialblend_v2.proto. C++ code names it through this alias.
namespace spatialblend_v2::proto {}
namespace spatialblend {
namespace proto = ::spatialblend_v2::proto;
}  // namespace spatialblend

namespace spatialblend {

struct HubSender;

// Base class for all async hub client calls.
// Unlike v1, these sender calls don't carry event_handle/OnEndEventCallback
// because there are no entity-level event responses.
struct AsyncHubSenderCall : public spatialblend::GrpcClientCallBase {
  AsyncHubSenderCall(std::shared_ptr<spatialblend::internal::IGrpcClient> client)
      : spatialblend::GrpcClientCallBase(client) {}
  virtual ~AsyncHubSenderCall() = default;
};

// UpdateActorState: client sends actor position/yaw to host.
struct UpdateActorStateSenderCall : public AsyncHubSenderCall {
  explicit UpdateActorStateSenderCall(
      std::shared_ptr<spatialblend::internal::IGrpcClient> client,
      space_id_t space_id, int64_t actor_id, int64_t actor_shared_id,
      double x, double y, double yaw);
  void Proceed(bool ok) override;

  proto::UpdateActorStateRequest request_;
  proto::Response response_;
};

// ShareElement: client sends an element event up to the host for relay.
struct ShareElementSenderCall : public AsyncHubSenderCall {
  explicit ShareElementSenderCall(
      std::shared_ptr<spatialblend::internal::IGrpcClient> client,
      space_id_t space_id, const ElementEvent& event);
  void Proceed(bool ok) override;

  proto::ShareElementRequest request_;
  proto::Response response_;
};

// PropagateRelation: client forwards a relation update up to the host.
struct PropagateRelationSenderCall : public AsyncHubSenderCall {
  explicit PropagateRelationSenderCall(
      std::shared_ptr<spatialblend::internal::IGrpcClient> client,
      space_id_t space_id, const RemoteRelationUpdate& update);
  void Proceed(bool ok) override;

  proto::PropagateRelationRequest request_;
  proto::Response response_;
};

// DisconnectNotifySenderCall: best-effort, fire-and-forget notification to
// the host that this client is disconnecting cleanly. Issued asynchronously
// from HubSender::PrepareDisconnect() -- NOT waited on synchronously there --
// because a synchronous/blocking DisconnectFromHub call previously stalled
// PrepareDisconnect() for its full RPC deadline whenever the host was already
// gone (e.g. concurrent host-side teardown), adding seconds of dead time to
// every such teardown. This call is instead retired the same way as any other
// pending call: via the drain-then-cancel sequence in HubSender::Disconnect()
// (called right after this is issued).
struct DisconnectNotifySenderCall : public AsyncHubSenderCall {
  explicit DisconnectNotifySenderCall(
      std::shared_ptr<spatialblend::internal::IGrpcClient> client,
      space_id_t space_id);
  void Proceed(bool ok) override;

  proto::DisconnectRequest request_;
  proto::Response response_;
};

// CompleteBootstrapSenderCall: client sends bootstrap ack to host.
struct CompleteBootstrapSenderCall : public AsyncHubSenderCall {
  explicit CompleteBootstrapSenderCall(
      std::shared_ptr<spatialblend::internal::IGrpcClient> client,
      space_id_t space_id, bool success);
  void Proceed(bool ok) override;

  proto::BootstrapResponse request_;
  proto::Response response_;

  // Bootstrap completion signaling — set from RemoteHub's callback.
  std::function<void(bool)> on_complete_;
};

// SubscribeStreamCall: client-side reader for the Subscribe server-side stream.
// Reads SpaceEvents pushed by the host and dispatches them to the local Space.
struct SubscribeStreamCall : public spatialblend::GrpcClientCallBase {
  explicit SubscribeStreamCall(
      std::shared_ptr<spatialblend::internal::IGrpcClient> client,
      Space* space);
  void Proceed(bool ok) override;

  proto::SpaceEvent current_event_;
  std::unique_ptr<grpc::ClientAsyncReader<proto::SpaceEvent>> reader_;
  Space* space_;

 private:
  // kStarting -> kReading -> (kFinishing on any !ok) -> delete.
  // A read failure (stream end, cancellation, or transport drop) is NOT the
  // terminal event for the call: grpc-core still owns internal completion
  // state for the RPC until Finish() has been issued and its tag observed.
  // Destroying ClientContext before that point can race a still-pending
  // internal completion and abort in ClientContext::~ClientContext()
  // (pthread_mutex_destroy EBUSY).
  enum class State { kStarting, kReading, kFinishing };
  State state_ = State::kStarting;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_NETWORK_CALL_HUB_SENDER_CALL_H_
