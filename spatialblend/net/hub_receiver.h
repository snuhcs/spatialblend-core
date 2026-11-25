// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_NETWORK_HUB_RECEIVER_H_
#define SPATIALBLEND_NETWORK_HUB_RECEIVER_H_

#include <grpcpp/grpcpp.h>

#include <condition_variable>
#include <functional>
#include <unordered_set>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "spatialblend/base/common/common.h"
#include "spatialblend/net/grpc/grpc_server_base.h"
#include "spatialblend/net/call/hub_receiver_call.h"
#include "spatialblend/net/proto/spatialblend.grpc.pb.h"
#include "spatialblend/net/remote_space.h"
#include "spatialblend/net/tunnel_link.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/space_hub.h"

// The protobuf package keeps its pre-1.0 name because it is on the wire (gRPC method paths, Any type
// URLs): spatialblend_v2.proto. C++ code names it through this alias.
namespace spatialblend_v2::proto {}
namespace spatialblend {
namespace proto = ::spatialblend_v2::proto;
}  // namespace spatialblend

namespace spatialblend {

// HubReceiver: host-side gRPC server for SpatialBlend networking.
// Accepts client connections, serves bootstrap data, manages Subscribe
// streams, and forwards client state updates to the host space.
class HubReceiver : public spatialblend::GrpcServerBase<AsyncHubReceiverCall>,
                    public std::enable_shared_from_this<HubReceiver> {
 public:
  HubReceiver(Space* host_space, SpaceHub* hub);
  ~HubReceiver();

  absl::Status Start(int port = spatialblend::kDefaultHubPort);

  // Starts the same service with no listening socket. Peers are attached one at
  // a time with AcceptTunnel(); no peer ever dials this process.
  absl::Status StartTunneled();

  // Attaches one joining peer. The returned link carries that peer's whole gRPC
  // session: the caller pumps its bytes to and from the peer. Closing the link
  // ends the peer's session on this host.
  absl::StatusOr<std::unique_ptr<TunnelLink>> AcceptTunnel();

  // Hides GrpcServerBase::Stop(): closes every still-open Subscribe stream
  // (remote_spaces_) BEFORE shutting down/destroying the completion queue.
  // Callers (StressHost, sb_destroy_session/sb_cleanup_all) may invoke Stop()
  // explicitly, well before this HubReceiver's owning shared_ptr actually
  // drops -- so remote_spaces_ must be drained here, not only from
  // ~HubReceiver(), for the ordering fix to hold regardless of call site.
  // See ~HubReceiver().
  void Stop();

  int GetConnectedPort() const;

  // Gives a joining peer its identity in this room. ConnectToHub calls this and
  // returns the id in the bootstrap; the peer adopts it and uses it for every
  // later call. The id the peer proposed (the number its own device gave the
  // space, which every other device may have given too) is not used: an
  // identity that crossed the wire must be unique among the room's peers, the
  // host's own space included, so only the host can hand it out. Never repeats
  // while the peer is connected.
  space_id_t AssignPeerSpaceId();

  // Where AssignPeerSpaceId takes fresh ids from; the default is
  // SpaceHub::AllocateSpaceId. A process that numbers its spaces elsewhere (the
  // C API's sb_space_t handles) installs its own, so an assigned id can never
  // equal a space that is created here later. It must return a new id on every
  // call. Set before the host starts.
  void SetSpaceIdAllocator(std::function<space_id_t()> allocator);

  // Build BootstrapPayload from the host space's element tree and bindings.
  proto::BootstrapPayload BuildBootstrapPayload();

  // What ConnectToHub answers: assigns the joiner its id and takes the
  // bootstrap snapshot, and from that same instant keeps every element event
  // the joiner has not been sent (it is not subscribed yet). OnClientSubscribed
  // sends them, in order, so a share or unshare made between the snapshot and
  // the Subscribe is neither lost nor applied twice.
  proto::BootstrapPayload BuildBootstrapForNewPeer();

  // Called when a client opens the Subscribe stream.
  absl::Status OnClientSubscribed(
      space_id_t space_id,
      std::shared_ptr<SubscribeReceiverCall> subscribe_call);

  // Called when CompleteBootstrap RPC arrives.
  absl::Status OnBootstrapComplete(space_id_t space_id, bool success);

  // Called when DisconnectFromHub RPC arrives.
  void OnClientDisconnected(space_id_t space_id);

  // Called when UpdateActorState RPC arrives from client. actor_shared_id is
  // the shared (cross-space) id of the client actor's element; the host
  // resolves its own local actor from it, falling back to the raw actor_id.
  void OnUpdateActorState(space_id_t space_id, int64_t actor_id,
                          int64_t actor_shared_id,
                          const Eigen::Vector2d& position, double yaw);

  // Called when ShareElement RPC arrives from a client. Applies the event to
  // the host space and relays it to all other connected clients (except the
  // source space), making element sharing bidirectional.
  void OnClientShareElement(space_id_t space_id, const ElementEvent& event);

  // Called when PropagateRelation RPC arrives from a client. Applies the update
  // to the host space (matching the local actor by actor_shared_id, solving,
  // and producing a realization target for the host's display) and relays it to
  // all other connected clients (except the source space). This is what drives
  // a connected client's avatar clone on the host.
  void OnClientRelationUpdate(space_id_t space_id,
                              const RemoteRelationUpdate& update);

  // Forward relation updates to all connected remote spaces (except source).
  void PropagateToRemoteSpaces(const RemoteRelationUpdate& update);

  // Forward element events to all connected remote spaces (except source).
  void PropagateElementEventToRemoteSpaces(const ElementEvent& event,
                                           space_id_t source_space_id);

  // Subscribe streams that have been accepted and not yet Finish()'d.
  // The CQ thread calls these; Stop() waits until the count is zero before
  // GrpcServerBase::Stop(), so server shutdown is not stuck on a live stream.
  void NoteStreamStarted();
  void NoteStreamRetired();

 private:
  // Registers every RPC listener call type once the server is running.
  void RegisterRpcListeners();

  Space* host_space_;
  SpaceHub* hub_;
  std::unique_ptr<proto::HubService::AsyncService> service_;
  AsyncHubReceiverCall::CallContext call_ctx_;
  int connected_port_ = 0;
  bool tunneled_ = false;

  std::mutex remote_spaces_mutex_;
  // Set under remote_spaces_mutex_ before the map is cleared.
  // OnClientSubscribed rejects new clients once this is true.
  bool draining_ = false;
  std::unordered_map<space_id_t, std::unique_ptr<RemoteSpace>> remote_spaces_;
  // Ids handed out by AssignPeerSpaceId whose peer has not left (guarded by
  // remote_spaces_mutex_, like remote_spaces_).
  std::unordered_set<space_id_t> assigned_space_ids_;
  std::function<space_id_t()> space_id_allocator_;
  // The ADDED event of every share that is live now, by its source handle
  // (guarded by remote_spaces_mutex_). A later share of the same source replaces
  // the entry and an unshare erases it, so this holds the current state, one
  // event per live share, never a history. BuildBootstrapPayload hands it to a
  // joiner, which would otherwise never see a share made before it subscribed.
  std::map<element_handle_t, ElementEvent> live_shares_;
  // Peers that have their bootstrap snapshot but no Subscribe stream yet, with
  // the element events made since that snapshot (guarded by
  // remote_spaces_mutex_). Ends at OnClientSubscribed or OnClientDisconnected.
  std::map<space_id_t, std::vector<ElementEvent>> pending_peers_;
  proto::BootstrapPayload BuildBootstrapPayloadLocked();

  std::mutex stream_mu_;
  std::condition_variable stream_cv_;
  int open_streams_ = 0;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_NETWORK_HUB_RECEIVER_H_
