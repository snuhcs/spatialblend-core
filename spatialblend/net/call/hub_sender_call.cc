// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/net/call/hub_sender_call.h"

#include "spatialblend/base/util/logger.h"
#include "spatialblend/net/proto_util.h"

namespace spatialblend {

namespace {
// The fixed per-call deadline, unless the client's link is carried by the app (see
// IGrpcClient::CallsHaveDeadline).
void SetCallDeadline(grpc::ClientContext* context,
                     const std::shared_ptr<spatialblend::internal::IGrpcClient>& client) {
  if (client && !client->CallsHaveDeadline()) return;
  context->set_deadline(std::chrono::system_clock::now() +
                        std::chrono::milliseconds(spatialblend::kDefaultTimeoutMs));
}
}  // namespace

// UpdateActorStateSenderCall -------------------------------------------------

UpdateActorStateSenderCall::UpdateActorStateSenderCall(
    std::shared_ptr<spatialblend::internal::IGrpcClient> client,
    space_id_t space_id, int64_t actor_id, int64_t actor_shared_id,
    double x, double y, double yaw)
    : AsyncHubSenderCall(client) {
  request_.set_space_id(space_id);
  request_.set_actor_id(actor_id);
  request_.set_actor_shared_id(actor_shared_id);
  request_.set_x(x);
  request_.set_y(y);
  request_.set_yaw(yaw);
  SetCallDeadline(&context_, client);
}

void UpdateActorStateSenderCall::Proceed(bool ok) {
  if (!ok) {
    SPATIALBLEND_LOG(spatialblend::LogSeverity::kWarning,
                  "UpdateActorStateSenderCall failed");
  }
  delete this;
}

// ShareElementSenderCall -----------------------------------------------------

ShareElementSenderCall::ShareElementSenderCall(
    std::shared_ptr<spatialblend::internal::IGrpcClient> client,
    space_id_t space_id, const ElementEvent& event)
    : AsyncHubSenderCall(client) {
  request_.set_space_id(space_id);
  *request_.mutable_event() = ToProto(event);
  SetCallDeadline(&context_, client);
}

void ShareElementSenderCall::Proceed(bool ok) {
  if (!ok) {
    SPATIALBLEND_LOG(spatialblend::LogSeverity::kWarning,
                  "ShareElementSenderCall failed");
  }
  delete this;
}

// PropagateRelationSenderCall ------------------------------------------------

PropagateRelationSenderCall::PropagateRelationSenderCall(
    std::shared_ptr<spatialblend::internal::IGrpcClient> client,
    space_id_t space_id, const RemoteRelationUpdate& update)
    : AsyncHubSenderCall(client) {
  request_.set_space_id(space_id);
  *request_.mutable_update() = ToProto(update);
  SetCallDeadline(&context_, client);
}

void PropagateRelationSenderCall::Proceed(bool ok) {
  if (!ok) {
    SPATIALBLEND_LOG(spatialblend::LogSeverity::kWarning,
                  "PropagateRelationSenderCall failed");
  }
  delete this;
}

// DisconnectNotifySenderCall --------------------------------------------------

DisconnectNotifySenderCall::DisconnectNotifySenderCall(
    std::shared_ptr<spatialblend::internal::IGrpcClient> client,
    space_id_t space_id)
    : AsyncHubSenderCall(client) {
  request_.set_space_id(space_id);
  SetCallDeadline(&context_, client);
}

void DisconnectNotifySenderCall::Proceed(bool ok) {
  if (!ok) {
    SPATIALBLEND_LOG(spatialblend::LogSeverity::kWarning,
                  "DisconnectNotifySenderCall failed (host may already be "
                  "gone) -- harmless, this is a best-effort notification");
  }
  delete this;
}

// CompleteBootstrapSenderCall ------------------------------------------------

CompleteBootstrapSenderCall::CompleteBootstrapSenderCall(
    std::shared_ptr<spatialblend::internal::IGrpcClient> client,
    space_id_t space_id, bool success)
    : AsyncHubSenderCall(client) {
  request_.set_space_id(space_id);
  request_.set_success(success);
  SetCallDeadline(&context_, client);
}

void CompleteBootstrapSenderCall::Proceed(bool ok) {
  bool success = ok && status_.ok();
  if (on_complete_) {
    on_complete_(success);
  }
  delete this;
}

// SubscribeStreamCall --------------------------------------------------------

SubscribeStreamCall::SubscribeStreamCall(
    std::shared_ptr<spatialblend::internal::IGrpcClient> client,
    Space* space)
    : spatialblend::GrpcClientCallBase(client), space_(space) {}

void SubscribeStreamCall::Proceed(bool ok) {
  if (state_ == State::kStarting) {
    if (!ok) {
      // StartCall itself failed (e.g. channel never became ready). The RPC
      // is still live from grpc-core's point of view — Finish() is the only
      // sanctioned way to retire it and it's what we must wait for before
      // touching context_ again.
      state_ = State::kFinishing;
      reader_->Finish(&status_, static_cast<void*>(this));
      return;
    }
    state_ = State::kReading;
    reader_->Read(&current_event_, static_cast<void*>(this));
    return;
  }

  if (state_ == State::kReading) {
    if (!ok) {
      // Stream ended, was cancelled (TryCancel from the owning HubSender's
      // Disconnect()), or the transport dropped. Do NOT delete yet: issue
      // Finish() and wait for its tag so grpc-core fully retires the call
      // before we destroy context_ (~ClientContext tears down internal
      // state that must not still be in flight).
      state_ = State::kFinishing;
      reader_->Finish(&status_, static_cast<void*>(this));
      return;
    }

    if (space_) {
      switch (current_event_.event_case()) {
        case proto::SpaceEvent::kRelationUpdate: {
          RemoteRelationUpdate update =
              FromProto(current_event_.relation_update());
          space_->OnRemoteUpdate(update);
          break;
        }
        case proto::SpaceEvent::kElementEvent: {
          ElementEvent event =
              FromProto(current_event_.element_event());
          space_->OnElementEvent(event);
          break;
        }
        default:
          SPATIALBLEND_LOG(spatialblend::LogSeverity::kWarning,
                        "SubscribeStreamCall: unknown SpaceEvent case %d",
                        static_cast<int>(current_event_.event_case()));
          break;
      }
    }

    reader_->Read(&current_event_, static_cast<void*>(this));
    return;
  }

  // kFinishing: the Finish() tag has now been delivered. grpc-core has no
  // further pending completions for this call, so it is safe to destroy
  // reader_/context_ (this self-delete still runs on the CQ thread, but only
  // after the call is fully retired — this is the documented-safe pattern
  // for async server-streaming clients, matching the host-side
  // SubscribeReceiverCall's writer_.Finish()-then-delete symmetry).
  delete this;
}

}  // namespace spatialblend
