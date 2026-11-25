// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_NETWORK_CALL_HUB_RECEIVER_CALL_H_
#define SPATIALBLEND_NETWORK_CALL_HUB_RECEIVER_CALL_H_

#include <grpcpp/alarm.h>
#include <grpcpp/grpcpp.h>

#include <memory>
#include <mutex>
#include <queue>

#include "spatialblend/net/proto/spatialblend.grpc.pb.h"
#include "spatialblend/net/proto_util.h"

// The protobuf package keeps its pre-1.0 name because it is on the wire (gRPC method paths, Any type
// URLs): spatialblend_v2.proto. C++ code names it through this alias.
namespace spatialblend_v2::proto {}
namespace spatialblend {
namespace proto = ::spatialblend_v2::proto;
}  // namespace spatialblend

namespace spatialblend {

class HubReceiver;
class SubscribeReceiverCall;

// Where an async server call is in its life: serving, or finished and waiting for the
// completion queue to hand it back for deletion.
enum class CallProgress { kProcessing, kCompleted };

// Base class for all async hub server calls.
struct AsyncHubReceiverCall {
  struct CallContext {
    std::weak_ptr<HubReceiver> receiver;
    proto::HubService::AsyncService* service;
    grpc::ServerCompletionQueue* cq;
  };

  AsyncHubReceiverCall(CallContext* call_data)
      : call_ctx_(call_data), status_(CallProgress::kProcessing) {}
  virtual ~AsyncHubReceiverCall() = default;
  virtual void Proceed(bool ok) = 0;

  CallContext* call_ctx_;
  grpc::ServerContext context_;
  CallProgress status_;
};

// ConnectToHub: unary RPC — client connects and receives BootstrapPayload.
struct ConnectToHubReceiverCall : public AsyncHubReceiverCall {
  ConnectToHubReceiverCall(CallContext* call_data);
  void Proceed(bool ok) override;

  proto::ConnectRequest request_;
  grpc::ServerAsyncResponseWriter<proto::BootstrapPayload> responder_;
};

// CompleteBootstrap: unary RPC — client sends bootstrap acknowledgement.
struct CompleteBootstrapReceiverCall : public AsyncHubReceiverCall {
  CompleteBootstrapReceiverCall(CallContext* call_data);
  void Proceed(bool ok) override;

  proto::BootstrapResponse request_;
  grpc::ServerAsyncResponseWriter<proto::Response> responder_;
};

// DisconnectFromHub: unary RPC — client requests disconnect.
struct DisconnectFromHubReceiverCall : public AsyncHubReceiverCall {
  DisconnectFromHubReceiverCall(CallContext* call_data);
  void Proceed(bool ok) override;

  proto::DisconnectRequest request_;
  grpc::ServerAsyncResponseWriter<proto::Response> responder_;
};

// UpdateActorState: unary RPC — client sends actor position update to host.
struct UpdateActorStateReceiverCall : public AsyncHubReceiverCall {
  UpdateActorStateReceiverCall(CallContext* call_data);
  void Proceed(bool ok) override;

  proto::UpdateActorStateRequest request_;
  grpc::ServerAsyncResponseWriter<proto::Response> responder_;
};

// ShareElement: unary RPC — client sends an element event up to the host.
struct ShareElementReceiverCall : public AsyncHubReceiverCall {
  ShareElementReceiverCall(CallContext* call_data);
  void Proceed(bool ok) override;

  proto::ShareElementRequest request_;
  grpc::ServerAsyncResponseWriter<proto::Response> responder_;
};

// PropagateRelation: unary RPC — client forwards a relation update to the host.
struct PropagateRelationReceiverCall : public AsyncHubReceiverCall {
  PropagateRelationReceiverCall(CallContext* call_data);
  void Proceed(bool ok) override;

  proto::PropagateRelationRequest request_;
  grpc::ServerAsyncResponseWriter<proto::Response> responder_;
};

// SubscribeAlarmTag: CQ tag for grpc::Alarm events on the subscribe stream.
struct SubscribeAlarmTag : public AsyncHubReceiverCall {
  SubscribeAlarmTag(CallContext* ctx, SubscribeReceiverCall* parent);
  void Proceed(bool ok) override;

  SubscribeReceiverCall* parent_;
};

// SubscribeReceiverCall: manages server-side streaming of SpaceEvent to client.
// Thread-safe: Enqueue() may be called from any thread; actual writes happen
// only on the CQ thread via TryWrite().
class SubscribeReceiverCall : public AsyncHubReceiverCall {
 public:
  explicit SubscribeReceiverCall(CallContext* ctx);

  // Called from hub processing threads to push a SpaceEvent to the client.
  void Enqueue(proto::SpaceEvent event);

  // Called by SubscribeAlarmTag::Proceed when the alarm fires.
  void OnAlarmFired(bool ok);

  // Called by Proceed() to attempt writing the next queued event.
  void TryWrite();

  // Explicitly close the stream from the server side. Thread-safe.
  void Close();

  void Proceed(bool ok) override;

  proto::SubscribeRequest request_;
  grpc::ServerAsyncWriter<proto::SpaceEvent> writer_;

 private:
  // Tells HubReceiver to erase its (now-stale) RemoteSpace entry for this
  // call's space_id. Must be called BEFORE `delete this` on every path where
  // the underlying stream is torn down due to a transport-level failure
  // (client crash / abrupt disconnect) rather than the explicit
  // DisconnectFromHub RPC -- otherwise HubReceiver::remote_spaces_ keeps a
  // RemoteSpace whose subscribe_call_ is a non-owning shared_ptr aliasing
  // this object, which becomes a dangling pointer the instant this object is
  // deleted. Safe to call even if the entry was already removed (no-op on a
  // missing key) or never inserted (weak_ptr lock / map lookup simply finds
  // nothing).
  void NotifyDisconnected();
  // Counts this call with HubReceiver so Stop() can wait for Finish().
  void AccountStream();
  // delete this, after retiring the stream count. CQ thread only.
  void Destroy();
  // Hop to the CQ thread. Caller holds mutex_. Immediate: a realtime "now"
  // alarm is clock-converted and can fire up to a second late, which stalls
  // the single server CQ thread past CompleteBootstrap's deadline.
  void ArmAlarmLocked();

  enum class State { kWaiting, kStreaming, kCompleted };
  State state_ = State::kWaiting;

  std::mutex mutex_;
  std::queue<proto::SpaceEvent> write_queue_;
  bool writing_ = false;
  bool alarm_active_ = false;
  bool closing_ = false;
  bool finish_pending_ = false;
  // SendInitialMetadata is outstanding. No Write until that tag returns.
  bool metadata_pending_ = false;
  bool accounted_ = false;

  grpc::Alarm alarm_;
  SubscribeAlarmTag alarm_tag_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_NETWORK_CALL_HUB_RECEIVER_CALL_H_
