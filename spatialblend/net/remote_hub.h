// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_NETWORK_REMOTE_HUB_H_
#define SPATIALBLEND_NETWORK_REMOTE_HUB_H_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>

#include "absl/status/status.h"
#include "spatialblend/net/grpc/grpc_client_base.h"
#include "spatialblend/base/common/types.h"
#include "spatialblend/net/call/hub_sender_call.h"
#include "spatialblend/net/tunnel_link.h"
#include "spatialblend/net/proto/spatialblend.grpc.pb.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/space_hub.h"

// The protobuf package keeps its pre-1.0 name because it is on the wire (gRPC method paths, Any type
// URLs): spatialblend_v2.proto. C++ code names it through this alias.
namespace spatialblend_v2::proto {}
namespace spatialblend {
namespace proto = ::spatialblend_v2::proto;
}  // namespace spatialblend

namespace spatialblend {

// HubSender: gRPC client connection to the host.
struct HubSender
    : public spatialblend::GrpcClientBase<AsyncHubSenderCall> {
  HubSender(Space* space, space_id_t space_id);
  ~HubSender();

  // Hides (does not override -- GrpcClientBase::Disconnect() is not virtual)
  // GrpcClientBase<AsyncHubSenderCall>::Disconnect(). The base version
  // cancels all pending_calls_ (TryCancel()) and then shuts down cq_ and
  // joins the CQ thread essentially back-to-back, with no wait for the
  // resulting cancellation completions to actually be drained and Finish()'d
  // on the CQ thread first. If Shutdown() wins that race against the
  // in-flight SubscribeStreamCall's Finish() call, grpc_cq_begin_op()
  // CHECK-fails inside grpc-core (new ops are forbidden once a CQ has begun
  // shutting down), aborting the process. This override drains pending_calls_
  // (bounded wait) before delegating to the base implementation.
  absl::Status Disconnect();

  absl::Status PrepareDisconnect();
  absl::Status OnConnect() override;

  bool IsInitialized() const { return bootstrap_complete_.load(); }

  // True for a link carried by the application (see TunnelLink): its calls carry
  // no fixed deadline. Set before ConnectChannel.
  void SetTunneled(bool tunneled) { tunneled_.store(tunneled); }
  bool CallsHaveDeadline() const override { return !tunneled_.load(); }

  // With staging on, OnConnect() stops after ConnectToHub and keeps the
  // bootstrap; ApplyStagedBootstrap() then applies it to the space and opens the
  // Subscribe stream and CompleteBootstrap. It runs on the caller's thread, so
  // the caller chooses which thread changes the space. Set before ConnectChannel.
  void SetStageBootstrap(bool stage) { stage_bootstrap_.store(stage); }
  absl::Status ApplyStagedBootstrap();

  // timeout_ms < 0 waits with no timer (until completion; a failed call
  // completes it).
  absl::Status WaitForBootstrapComplete(int timeout_ms);

  Space* space_;
  using spatialblend::GrpcClientBase<AsyncHubSenderCall>::cq_;
  std::unique_ptr<proto::HubService::Stub> stub_;
  // The id the host gave this peer (see HubReceiver::AssignPeerSpaceId), which
  // every call after ConnectToHub carries; until then, the one the caller
  // proposed.
  std::atomic<space_id_t> space_id_;

  // Bootstrap completion signaling.
  std::mutex bootstrap_mutex_;
  std::condition_variable bootstrap_cv_;
  std::atomic<bool> bootstrap_complete_{false};
  std::atomic<bool> tunneled_{false};
  absl::Status bootstrap_status_;

 private:
  absl::Status ApplyBootstrap(const proto::BootstrapPayload& bootstrap);

  std::atomic<bool> stage_bootstrap_{false};
  space_id_t assigned_space_id_ = 0;  // 0: the host did not assign one
  proto::BootstrapPayload staged_bootstrap_;  // bootstrap_mutex_
  bool has_staged_bootstrap_ = false;         // bootstrap_mutex_
};

// RemoteHub: client-side hub that implements SpaceHub interface.
// When the local Space calls hub->Propagate(), RemoteHub sends the
// relation update to the host via UpdateActorState RPC.
// The Subscribe stream delivers host-originated relation updates
// directly to Space::OnRemoteUpdate via SubscribeStreamCall.
class RemoteHub : public SpaceHub {
 public:
  ~RemoteHub();

  static std::shared_ptr<RemoteHub> Create(
      Space* space, const std::string& host_address, int host_port,
      space_id_t space_id, int bootstrap_timeout_ms = 30000);

  // Joins the host over a channel that is not a TCP dial, e.g. the one made by
  // MakeTunnelChannel(). Blocks until bootstrap completes, like Create();
  // bootstrap_timeout_ms < 0 waits with no timer.
  static std::shared_ptr<RemoteHub> CreateOnChannel(
      Space* space, std::shared_ptr<grpc::Channel> channel, space_id_t space_id,
      int bootstrap_timeout_ms = -1);

  // The first half of CreateOnChannel, for a caller that must apply the
  // bootstrap on its own thread: blocks until the host has answered
  // ConnectToHub (it returns null if the link ends first), changes nothing in
  // `space`, and keeps the bootstrap. Then call ApplyStagedBootstrap() on the
  // thread that may change the space, and WaitForBootstrapComplete().
  static std::shared_ptr<RemoteHub> ConnectOnChannel(
      Space* space, std::shared_ptr<grpc::Channel> channel, space_id_t space_id);
  absl::Status ApplyStagedBootstrap();
  absl::Status WaitForBootstrapComplete(int timeout_ms = -1);

  // A gRPC channel that runs over `link` (consumes the link's core end). The
  // link must be pumped for the channel to make progress. Null on a platform
  // without fd channels (Windows) or when the link was already consumed.
  static std::shared_ptr<grpc::Channel> MakeTunnelChannel(TunnelLink* link);

  absl::Status PrepareDisconnect();

  // Send actor position update to host. actor_shared_id is the shared
  // (cross-space) id of the actor's element, used by the host to resolve its
  // own local actor; actor_id is kept as a per-space fallback.
  void SendUpdateActorState(int64_t actor_id, int64_t actor_shared_id,
                            double x, double y, double yaw);

  // Send an element event up to the host, which applies it locally and relays
  // it to all other connected clients. Makes element sharing bidirectional.
  void SendElementEvent(const ElementEvent& event);

  // Forward a relation update up to the host, which applies it locally (driving
  // this client's avatar clone on the host display) and relays it to all other
  // connected clients. Makes client-originated avatar movement reach the host.
  void SendRelationUpdate(const RemoteRelationUpdate& update);

  bool IsConnected() const;

 private:
  RemoteHub(Space* space, std::shared_ptr<HubSender> sender);

  Space* space_;
  std::shared_ptr<HubSender> sender_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_NETWORK_REMOTE_HUB_H_
