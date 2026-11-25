// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_SPACE_HUB_H_
#define SPATIALBLEND_SPACE_SPACE_HUB_H_

#include <functional>
#include <mutex>
#include <vector>

#include "absl/status/status.h"
#include "spatialblend/model/link/element_linker.h"
#include "spatialblend/model/realization/remote_update.h"

namespace spatialblend {

class Space;

class SpaceHub {
 public:
  SpaceHub() = default;
  virtual ~SpaceHub() = default;

  absl::Status Connect(Space* space);
  absl::Status Disconnect(Space* space);

  void Propagate(const RemoteRelationUpdate& update);
  void PropagateElementEvent(const ElementEvent& event,
                             space_id_t source_space_id);

  size_t GetConnectedSpaceCount() const;

  // True if a space connected to this hub currently has `id`.
  bool HasSpaceId(space_id_t id) const;

  // A space id this hub has never handed out and no connected space holds.
  // Increasing, so an id is not reused while its holder may still be named on
  // the wire. This is where a room's peer identities come from: a space id
  // must be unique among the spaces of one room, and nothing numbered by a
  // single device can promise that (every device starts at 1).
  space_id_t AllocateSpaceId();

  // ElementLinker owned by this hub — one linker per hub, shared across all
  // spaces connected to it.
  ElementLinker& GetLinker() { return linker_; }
  const ElementLinker& GetLinker() const { return linker_; }

  // Callbacks invoked when events are propagated, allowing HubReceiver
  // to forward events to remote clients.
  using ElementEventCallback =
      std::function<void(const ElementEvent&, space_id_t)>;
  void SetElementEventCallback(ElementEventCallback callback);

  using RelationUpdateCallback =
      std::function<void(const RemoteRelationUpdate&)>;
  void SetRelationUpdateCallback(RelationUpdateCallback callback);

  // Nulls both callback slots under lock. Callers (sb_destroy_session /
  // sb_cleanup_all) must call this BEFORE tearing down whatever the
  // callbacks' closures capture (a RemoteHub, a HubReceiver, ...): the
  // callbacks are invoked from gRPC completion-queue threads (e.g.
  // HubReceiver::OnUpdateActorState -> Space::UpdateActorState ->
  // hub_->Propagate() runs on the host's server CQ thread) concurrently
  // with the app/main thread that owns session teardown. Copying the
  // callback under lock before invoking it (see Propagate/
  // PropagateElementEvent) means an in-flight call already holds its own
  // safe copy, so clearing the slot here only prevents *new* invocations
  // from racing the teardown that follows.
  void ClearCallbacks();

 private:
  mutable std::mutex mutex_;
  std::vector<Space*> spaces_;
  ElementEventCallback element_event_callback_;
  RelationUpdateCallback relation_update_callback_;
  space_id_t next_space_id_ = 1;  // mutex_
  ElementLinker linker_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_SPACE_HUB_H_
