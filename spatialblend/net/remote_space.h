// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_NETWORK_REMOTE_SPACE_H_
#define SPATIALBLEND_NETWORK_REMOTE_SPACE_H_

#include <memory>

#include "spatialblend/base/common/types.h"
#include "spatialblend/net/call/hub_receiver_call.h"
#include "spatialblend/model/realization/remote_update.h"

namespace spatialblend {

class SpaceHub;

// RemoteSpace represents a remote client from the host's perspective.
// When the hub propagates a RemoteRelationUpdate to it, it serializes
// to proto and enqueues on the client's Subscribe stream.
class RemoteSpace {
 public:
  RemoteSpace(space_id_t space_id,
              std::shared_ptr<SubscribeReceiverCall> subscribe_call);
  ~RemoteSpace();

  space_id_t GetSpaceId() const { return space_id_; }

  // Called by SpaceHub::Propagate when a relation update needs to be
  // forwarded to this remote client.
  void OnRelationUpdate(const RemoteRelationUpdate& update);

  // Called by HubReceiver when an element event needs to be forwarded
  // to this remote client.
  void OnElementEvent(const ElementEvent& event);

 private:
  space_id_t space_id_;
  std::shared_ptr<SubscribeReceiverCall> subscribe_call_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_NETWORK_REMOTE_SPACE_H_
