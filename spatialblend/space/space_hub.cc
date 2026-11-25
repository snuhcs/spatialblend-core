// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/space_hub.h"

#include <algorithm>

#include "spatialblend/space/space.h"

namespace spatialblend {

absl::Status SpaceHub::Connect(Space* space) {
  if (!space) return absl::InvalidArgumentError("Space is null");
  std::lock_guard<std::mutex> lock(mutex_);
  spaces_.push_back(space);
  return absl::OkStatus();
}

absl::Status SpaceHub::Disconnect(Space* space) {
  std::lock_guard<std::mutex> lock(mutex_);
  spaces_.erase(std::remove(spaces_.begin(), spaces_.end(), space),
                spaces_.end());
  return absl::OkStatus();
}

void SpaceHub::Propagate(const RemoteRelationUpdate& update) {
  // Snapshot the space list and callback under lock, then invoke outside
  // the lock: Space::OnRemoteUpdate/OnElementEvent (below) and the
  // callback closures can themselves re-enter SpaceHub (e.g. via
  // Space::SetHub-linked calls) or block on I/O, and neither must be run
  // while holding mutex_ — that would risk a lock-order inversion against
  // Space's own space_mutex_. Taking the snapshot is what actually closes
  // the race: a concurrent SetRelationUpdateCallback/ClearCallbacks() can
  // no longer torn-read the std::function this call is about to invoke.
  std::vector<Space*> spaces_snapshot;
  RelationUpdateCallback callback_snapshot;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    spaces_snapshot = spaces_;
    callback_snapshot = relation_update_callback_;
  }
  for (Space* space : spaces_snapshot) {
    if (space->GetSpaceId() != update.source_space_id) {
      space->OnRemoteUpdate(update);
    }
  }
  if (callback_snapshot) {
    callback_snapshot(update);
  }
}

void SpaceHub::PropagateElementEvent(const ElementEvent& event,
                                     space_id_t source_space_id) {
  std::vector<Space*> spaces_snapshot;
  ElementEventCallback callback_snapshot;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    spaces_snapshot = spaces_;
    callback_snapshot = element_event_callback_;
  }
  for (Space* space : spaces_snapshot) {
    if (space->GetSpaceId() != source_space_id) {
      space->OnElementEvent(event);
    }
  }
  if (callback_snapshot) {
    callback_snapshot(event, source_space_id);
  }
}

void SpaceHub::SetElementEventCallback(ElementEventCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  element_event_callback_ = std::move(callback);
}

void SpaceHub::SetRelationUpdateCallback(RelationUpdateCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  relation_update_callback_ = std::move(callback);
}

void SpaceHub::ClearCallbacks() {
  std::lock_guard<std::mutex> lock(mutex_);
  element_event_callback_ = nullptr;
  relation_update_callback_ = nullptr;
}

size_t SpaceHub::GetConnectedSpaceCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return spaces_.size();
}

bool SpaceHub::HasSpaceId(space_id_t id) const {
  // Space::GetSpaceId takes the space's own lock: read it outside mutex_.
  std::vector<Space*> spaces_snapshot;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    spaces_snapshot = spaces_;
  }
  for (Space* space : spaces_snapshot) {
    if (space->GetSpaceId() == id) return true;
  }
  return false;
}

space_id_t SpaceHub::AllocateSpaceId() {
  for (;;) {
    space_id_t id;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      id = next_space_id_++;
    }
    if (!HasSpaceId(id)) return id;
  }
}

}  // namespace spatialblend
