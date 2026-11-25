// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/net/remote_space.h"

#include "spatialblend/base/util/logger.h"
#include "spatialblend/net/proto_util.h"

namespace spatialblend {

RemoteSpace::RemoteSpace(
    space_id_t space_id,
    std::shared_ptr<SubscribeReceiverCall> subscribe_call)
    : space_id_(space_id), subscribe_call_(std::move(subscribe_call)) {}

RemoteSpace::~RemoteSpace() {
  if (subscribe_call_) {
    subscribe_call_->Close();
  }
}

void RemoteSpace::OnRelationUpdate(const RemoteRelationUpdate& update) {
  if (!subscribe_call_) return;

  proto::SpaceEvent event;
  *event.mutable_relation_update() = ToProto(update);
  subscribe_call_->Enqueue(std::move(event));
}

void RemoteSpace::OnElementEvent(const ElementEvent& event) {
  if (!subscribe_call_) return;

  proto::SpaceEvent space_event;
  *space_event.mutable_element_event() = ToProto(event);
  subscribe_call_->Enqueue(std::move(space_event));
}

}  // namespace spatialblend
