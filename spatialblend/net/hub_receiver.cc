// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/net/hub_receiver.h"

#include <chrono>
#include <cinttypes>

#include "spatialblend/base/util/logger.h"
#include "spatialblend/base/util/util.h"
#include <grpcpp/server_posix.h>
#include "spatialblend/model/element/element.h"
#include "spatialblend/net/proto_util.h"
#include "spatialblend/model/element/element_tree.h"

namespace spatialblend {

HubReceiver::HubReceiver(Space* host_space, SpaceHub* hub)
    : host_space_(host_space),
      hub_(hub),
      service_(std::make_unique<proto::HubService::AsyncService>()) {}

HubReceiver::~HubReceiver() {
  // Explicitly stop BEFORE any of HubReceiver's own remaining members
  // (service_ in particular) start destructing -- mirrors the fix in
  // ~HubSender() (remote_hub.cc): C++ destroys a derived object's own
  // members before running the base class destructor, and the CQ/server
  // shutdown only runs from ~GrpcServerBase(). Without this, dropping a
  // HubReceiver shared_ptr without an explicit prior Stop() call
  // (sb_destroy_session, sb_cleanup_all) would destroy service_ out from
  // under still-live in-flight RPCs and a still-running CQ thread.
  //
  // This calls HubReceiver::Stop() (below), not GrpcServerBase::Stop()
  // directly, so that remote_spaces_ is always drained before the
  // completion queue is shut down/destroyed, regardless of whether Stop()
  // was already invoked explicitly by a caller (see HubReceiver::Stop() for
  // why that ordering matters). Stop() is idempotent (guarded by
  // shutdown_.exchange()), so this is a no-op if Stop() already ran.
  Stop();
}

void HubReceiver::NoteStreamStarted() {
  std::lock_guard<std::mutex> lock(stream_mu_);
  ++open_streams_;
}

void HubReceiver::NoteStreamRetired() {
  std::lock_guard<std::mutex> lock(stream_mu_);
  if (open_streams_ > 0) {
    --open_streams_;
  }
  if (open_streams_ == 0) {
    stream_cv_.notify_all();
  }
}

void HubReceiver::Stop() {
  // Close every still-open Subscribe stream BEFORE shutting down/destroying
  // the completion queue in GrpcServerBase::Stop() below.
  //
  // RemoteSpace::~RemoteSpace() calls SubscribeReceiverCall::Close(), which
  // arms a grpc::Alarm on call_ctx_->cq (the server's completion queue) so
  // the CQ thread can finish/delete the call asynchronously. But
  // GrpcServerBase::Stop() doesn't just Shutdown() that completion queue --
  // it eventually calls cq_.reset(), fully destroying the
  // grpc::ServerCompletionQueue object. If remote_spaces_ were left to
  // destruct later (whether via ~HubReceiver()'s implicit member teardown,
  // or because a caller invoked Stop() explicitly -- e.g. StressHost's
  // destructor calls receiver->Stop() before its shared_ptr<HubReceiver>
  // member is destroyed, and sb_destroy_session/sb_cleanup_all follow the
  // same pattern -- well before remote_spaces_ is otherwise cleared),
  // Close()'s alarm_.Set(call_ctx_->cq, ...) would schedule an operation on
  // an already-freed CompletionQueue, and grpc_cq_internal_ref() would
  // dereference/increment a use-count field on freed memory -- a
  // SIGSEGV/EXC_BAD_ACCESS observed under lldb in exactly this call chain
  // (~HubReceiver() -> ~RemoteSpace() -> SubscribeReceiverCall::Close() ->
  // grpc::Alarm::Set() -> grpc_cq_internal_ref(cq=nullptr)) once Stop() had
  // already run for this HubReceiver from an earlier explicit call.
  //
  // This method hides (is not an override of) GrpcServerBase::Stop() so
  // that every call site -- explicit (receiver->Stop()) or implicit
  // (~HubReceiver(), ~GrpcServerBase()) -- goes through this ordering.
  // Draining remote_spaces_ here, before GrpcServerBase::Stop() joins
  // server_threads_ (the CQ-draining HandleRpcs threads) and only then
  // resets cq_, guarantees every alarm armed by Close() is fully processed
  // (and the SubscribeReceiverCall objects self-deleted) before the queue
  // itself is torn down.
  bool already_draining = false;
  {
    std::lock_guard<std::mutex> lock(remote_spaces_mutex_);
    already_draining = draining_;
    draining_ = true;
    // Close() arms each stream's alarm while shutdown_ is still false, so the
    // CQ thread Finishes the RPC instead of observing a forced failure.
    remote_spaces_.clear();
    pending_peers_.clear();
  }
  if (!already_draining) {
    // server_->Shutdown(deadline) waits out its whole grace period for any
    // RPC that has not Finish()'d. HandleRpcs forces ok=false once shutdown_
    // is set, and the alarm's failure path used to delete the call without
    // Finish(), so every host teardown burned the full 2s and then
    // cancel-all'd. Wait here, with the CQ still polling and shutdown_ still
    // false, until those Finishes land.
    std::unique_lock<std::mutex> lock(stream_mu_);
    stream_cv_.wait(lock, [this] { return open_streams_ == 0; });
  }
  GrpcServerBase<AsyncHubReceiverCall>::Stop();
}

absl::Status HubReceiver::StartTunneled() {
#if defined(GPR_SUPPORT_CHANNELS_FROM_FD)
  tunneled_ = true;
  RETURN_IF_ERROR(GrpcServerBase::StartWithoutListening(service_.get(), 1));
  RegisterRpcListeners();
  return absl::OkStatus();
#else
  return absl::UnimplementedError(
      "The tunneled core link needs POSIX sockets (Android, macOS, Linux)");
#endif
}

absl::StatusOr<std::unique_ptr<TunnelLink>> HubReceiver::AcceptTunnel() {
#if defined(GPR_SUPPORT_CHANNELS_FROM_FD)
  if (!tunneled_) {
    return absl::FailedPreconditionError("host was not started tunneled");
  }
  std::lock_guard<std::mutex> lock(remote_spaces_mutex_);
  if (draining_ || shutdown_.load() || !server_) {
    return absl::FailedPreconditionError("host is stopping");
  }
  auto link = TunnelLink::Create();
  if (!link.ok()) return link.status();
  const int fd = (*link)->ReleaseCoreFd();
  if (fd < 0) return absl::InternalError("tunnel link has no core end");
  // The server owns fd from here on; it closes it when the peer's session ends.
  grpc::AddInsecureChannelFromFd(server_.get(), fd);
  return link;
#else
  return absl::UnimplementedError(
      "The tunneled core link needs POSIX sockets (Android, macOS, Linux)");
#endif
}

absl::Status HubReceiver::Start(int port) {
  int bound_port = port;
  RETURN_IF_ERROR(GrpcServerBase::Start(port, service_.get(), 1, &bound_port));

  connected_port_ = bound_port;
  RegisterRpcListeners();
  return absl::OkStatus();
}

void HubReceiver::RegisterRpcListeners() {
  call_ctx_ = {shared_from_this(), service_.get(), cq_.get()};

  // Register all RPC listener call types.
  new ConnectToHubReceiverCall(&call_ctx_);
  new CompleteBootstrapReceiverCall(&call_ctx_);
  new DisconnectFromHubReceiverCall(&call_ctx_);
  new SubscribeReceiverCall(&call_ctx_);
  new UpdateActorStateReceiverCall(&call_ctx_);
  new ShareElementReceiverCall(&call_ctx_);
  new PropagateRelationReceiverCall(&call_ctx_);
}

int HubReceiver::GetConnectedPort() const { return connected_port_; }

proto::BootstrapPayload HubReceiver::BuildBootstrapPayload() {
  std::lock_guard<std::mutex> lock(remote_spaces_mutex_);
  return BuildBootstrapPayloadLocked();
}

proto::BootstrapPayload HubReceiver::BuildBootstrapForNewPeer() {
  const space_id_t id = AssignPeerSpaceId();
  std::lock_guard<std::mutex> lock(remote_spaces_mutex_);
  proto::BootstrapPayload payload = BuildBootstrapPayloadLocked();
  payload.set_assigned_space_id(id);
  pending_peers_[id];  // from here on the peer's element events are kept
  return payload;
}

// Caller holds remote_spaces_mutex_.
proto::BootstrapPayload HubReceiver::BuildBootstrapPayloadLocked() {
  proto::BootstrapPayload payload;

  if (!host_space_) return payload;

  const ElementTree* tree = host_space_->GetElementTree();
  if (!tree) return payload;

  // Iterate all elements in the tree via GetRoot + subtree traversal.
  Element* root = const_cast<ElementTree*>(tree)->GetRoot();
  if (!root) return payload;

  std::vector<element_id_t> all_ids = tree->GetSubtreeIds(root->GetId());
  // Include root itself.
  all_ids.insert(all_ids.begin(), root->GetId());

  // The live shares, each resolved to the host element that stands for it.
  std::map<element_id_t, std::vector<ElementEvent>> share_of_element;
  {
    for (const auto& [source, event] : live_shares_) {
      const element_id_t local = host_space_->LocalElementForSource(source);
      if (local != kInvalidElementId) {
        share_of_element[local].push_back(event);
      } else {
        SPATIALBLEND_LOG(spatialblend::LogSeverity::kWarning,
                      "HubReceiver::BuildBootstrapPayload - live share from "
                      "space %" PRId64 " element %" PRId64
                      " has no host element; not sent",
                      source.space_id, source.element_id);
      }
    }
  }

  for (element_id_t id : all_ids) {
    const Element* elem = tree->GetElement(id);
    if (!elem) continue;

    proto::ElementProto* ep = payload.add_elements();
    ep->set_element_id(id);
    ep->set_shared_id(elem->GetSharedId());
    ep->set_tag(elem->GetTag());
    ep->set_parent_id(elem->GetParentId());
    auto share = share_of_element.find(id);
    if (share != share_of_element.end()) {
      for (const ElementEvent& event : share->second) {
        *ep->add_shares() = ToProto(event);
      }
    }
  }

  // Iterate all actors and their bindings.
  // We iterate actor IDs by checking known actor IDs from bindings.
  // The Space keeps actors and bindings keyed by actor_id.
  // We need to iterate through elements that are actors (have associated
  // actor objects).
  for (element_id_t id : all_ids) {
    const Element* elem = tree->GetElement(id);
    if (!elem) continue;
    spatialblend::entity_id_t grounding_id = elem->GetGroundingEntityId();
    if (grounding_id == spatialblend::kInvalidEntityId) continue;

    const Actor* actor = host_space_->GetActor(grounding_id);
    if (!actor) continue;

    // Export actor with anchors.
    proto::ActorProto* ap = payload.add_actors();
    ap->set_actor_shared_id(elem->GetSharedId());
    for (const auto& anchor : actor->GetAnchors()) {
      proto::AnchorProto* anp = ap->add_anchors();
      anp->set_name(anchor.name);
      anp->set_x(anchor.local_offset.x());
      anp->set_y(anchor.local_offset.y());
      anp->set_z(anchor.local_offset.z());
    }

    // Export bindings for this actor.
    const auto& bindings = host_space_->GetBindings(grounding_id);
    for (const auto& binding : bindings) {
      proto::BindingProto* bp = payload.add_bindings();
      bp->set_actor_shared_id(elem->GetSharedId());
      bp->set_anchor_name(binding.anchor_name);
      bp->set_kernel_type(static_cast<int32_t>(binding.kernel_type));
      bp->set_tag_filter(binding.tag_filter);
    }
  }

  return payload;
}

void HubReceiver::SetSpaceIdAllocator(std::function<space_id_t()> allocator) {
  std::lock_guard<std::mutex> lock(remote_spaces_mutex_);
  space_id_allocator_ = std::move(allocator);
}

space_id_t HubReceiver::AssignPeerSpaceId() {
  for (;;) {
    std::function<space_id_t()> allocator;
    {
      std::lock_guard<std::mutex> lock(remote_spaces_mutex_);
      allocator = space_id_allocator_;
    }
    const space_id_t id = allocator ? allocator() : hub_->AllocateSpaceId();
    if (id == kInvalidSpaceId || id <= 0) continue;
    if (host_space_ && host_space_->GetSpaceId() == id) continue;
    if (hub_->HasSpaceId(id)) continue;
    std::lock_guard<std::mutex> lock(remote_spaces_mutex_);
    if (remote_spaces_.count(id) != 0) continue;
    if (!assigned_space_ids_.insert(id).second) continue;
    return id;
  }
}

absl::Status HubReceiver::OnClientSubscribed(
    space_id_t space_id,
    std::shared_ptr<SubscribeReceiverCall> subscribe_call) {
  auto remote_space =
      std::make_unique<RemoteSpace>(space_id, std::move(subscribe_call));

  SPATIALBLEND_LOG(spatialblend::LogSeverity::kInfo,
                "HubReceiver::OnClientSubscribed - space_id=%" PRId64,
                space_id);

  {
    std::lock_guard<std::mutex> lock(remote_spaces_mutex_);
    if (draining_) {
      return absl::CancelledError("HubReceiver is shutting down");
    }
    // A peer is named by the id this host gave it. A Subscribe naming any
    // other id would be a peer speaking as someone else (or a joiner that
    // never adopted its id): it is not admitted.
    if (assigned_space_ids_.count(space_id) == 0 ||
        remote_spaces_.count(space_id) != 0) {
      return absl::FailedPreconditionError(
          "space id was not assigned to a joining peer by this host");
    }
    // What happened since this peer's bootstrap snapshot, in order, before
    // anything live reaches it.
    auto pending = pending_peers_.find(space_id);
    if (pending != pending_peers_.end()) {
      for (const ElementEvent& event : pending->second) {
        remote_space->OnElementEvent(event);
      }
      pending_peers_.erase(pending);
    }
    remote_spaces_[space_id] = std::move(remote_space);
  }

  return absl::OkStatus();
}

absl::Status HubReceiver::OnBootstrapComplete(space_id_t space_id,
                                              bool success) {
  SPATIALBLEND_LOG(spatialblend::LogSeverity::kInfo,
                "HubReceiver::OnBootstrapComplete - space_id=%" PRId64
                " success=%d",
                space_id, static_cast<int>(success));
  return absl::OkStatus();
}

void HubReceiver::OnClientDisconnected(space_id_t space_id) {
  SPATIALBLEND_LOG(spatialblend::LogSeverity::kInfo,
                "HubReceiver::OnClientDisconnected - space_id=%" PRId64,
                space_id);
  std::lock_guard<std::mutex> lock(remote_spaces_mutex_);
  remote_spaces_.erase(space_id);
  assigned_space_ids_.erase(space_id);
  pending_peers_.erase(space_id);
  // The peer's id is never reused (a rejoin is assigned a new one), so its
  // shares can never be unshared by it. They are no longer live state to hand
  // to a joiner: the rejoiner shares its own content again under its new id.
  for (auto it = live_shares_.begin(); it != live_shares_.end();) {
    it = (it->first.space_id == space_id) ? live_shares_.erase(it)
                                          : std::next(it);
  }
}

void HubReceiver::OnUpdateActorState(space_id_t space_id, int64_t actor_id,
                                     int64_t actor_shared_id,
                                     const Eigen::Vector2d& position,
                                     double yaw) {
  if (!host_space_) return;

  // Resolve the host-local actor from the shared id first; per-space actor
  // namespaces mean the raw actor_id would otherwise land on the wrong actor.
  ElementTree* tree = host_space_->GetElementTree();
  Element* e = tree ? tree->GetElementBySharedId(actor_shared_id) : nullptr;
  spatialblend::entity_id_t local =
      e ? e->GetGroundingEntityId() : spatialblend::kInvalidEntityId;
  spatialblend::entity_id_t resolved =
      (local != spatialblend::kInvalidEntityId)
          ? local
          : static_cast<spatialblend::entity_id_t>(actor_id);

  host_space_->UpdateActorState(resolved, position, yaw);
}

void HubReceiver::OnClientShareElement(space_id_t space_id,
                                       const ElementEvent& event) {
  // The sender's source_space_id is not trusted; restore it from the
  // request's space_id so the relay can exclude the originating client.
  ElementEvent ev = event;
  if (ev.type == ElementEvent::kUnlinked) {
    // The unlinking receiver is the requesting peer; source_* name the
    // sharer's element and must stay as sent.
    ev.peer_space_id = space_id;
  } else {
    ev.source_space_id = space_id;
  }

  // Apply to the host space (reconstructs the client's element via the host's
  // on_element_added_ hook). Space::OnElementEvent does NOT re-propagate to the
  // hub, so this does not cause an echo loop.
  if (host_space_) host_space_->OnElementEvent(ev);

  // Relay to all other connected clients (excludes the source space).
  PropagateElementEventToRemoteSpaces(ev, space_id);
}

void HubReceiver::OnClientRelationUpdate(space_id_t space_id,
                                         const RemoteRelationUpdate& update) {
  // Restore the source space id (the proto round-trip carries it, but trust the
  // request's space_id) so the relay excludes the originating client.
  RemoteRelationUpdate u = update;
  u.source_space_id = space_id;

  // Apply to the host space: matches the local actor by actor_shared_id,
  // solves, and stores realization_results_ so the host produces a realization
  // target for its own display. OnRemoteUpdate does NOT re-call
  // hub_->Propagate, so this does not cause an echo loop.
  if (host_space_) host_space_->OnRemoteUpdate(u);

  // Relay to all other connected clients (excludes the source space).
  PropagateToRemoteSpaces(u);
}

void HubReceiver::PropagateToRemoteSpaces(const RemoteRelationUpdate& update) {
  std::lock_guard<std::mutex> lock(remote_spaces_mutex_);
  for (auto& [space_id, remote_space] : remote_spaces_) {
    if (space_id != update.source_space_id) {
      remote_space->OnRelationUpdate(update);
    }
  }
}

void HubReceiver::PropagateElementEventToRemoteSpaces(
    const ElementEvent& event, space_id_t source_space_id) {
  std::lock_guard<std::mutex> lock(remote_spaces_mutex_);
  if (event.type == ElementEvent::kUnlinked) {
    // A receiver dropped its link; the share stays live and nobody else's link
    // changes, so only the sharer is told. When the receiver is the host, the
    // host element that stood for the share is gone: a joiner must not be
    // handed the share through it.
    if (event.source_element_id != kInvalidElementId && host_space_ &&
        event.peer_space_id == host_space_->GetSpaceId()) {
      live_shares_.erase(
          element_handle_t{event.source_space_id, event.source_element_id});
    }
    if (event.source_space_id != source_space_id) {
      auto sharer = remote_spaces_.find(event.source_space_id);
      if (sharer != remote_spaces_.end()) sharer->second->OnElementEvent(event);
    }
    return;
  }
  // Every share, the host's own and a client's relayed one, passes here.
  if (event.source_element_id != kInvalidElementId) {
    const element_handle_t source{event.source_space_id,
                                  event.source_element_id};
    if (event.type == ElementEvent::kAdded) {
      live_shares_[source] = event;
    } else {
      live_shares_.erase(source);
    }
  }
  for (auto& [peer, events] : pending_peers_) {
    if (peer != source_space_id) events.push_back(event);
  }
  for (auto& [space_id, remote_space] : remote_spaces_) {
    if (space_id != source_space_id) {
      remote_space->OnElementEvent(event);
    }
  }
}

}  // namespace spatialblend
