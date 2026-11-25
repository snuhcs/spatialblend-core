// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/net/call/hub_receiver_call.h"

#include <grpc/support/time.h>

#include "spatialblend/base/util/logger.h"
#include "spatialblend/net/hub_receiver.h"

namespace spatialblend {

// ConnectToHubReceiverCall ---------------------------------------------------

ConnectToHubReceiverCall::ConnectToHubReceiverCall(CallContext* call_ctx)
    : AsyncHubReceiverCall(call_ctx), responder_(&context_) {
  call_ctx_->service->RequestConnectToHub(&context_, &request_, &responder_,
                                          call_ctx_->cq, call_ctx_->cq, this);
}

void ConnectToHubReceiverCall::Proceed(bool ok) {
  if (!ok) {
    delete this;
    return;
  }

  if (status_ == CallProgress::kProcessing) {
    new ConnectToHubReceiverCall(call_ctx_);
    status_ = CallProgress::kCompleted;

    if (auto receiver = call_ctx_->receiver.lock()) {
      auto payload = receiver->BuildBootstrapForNewPeer();
      responder_.Finish(payload, grpc::Status::OK, this);
    } else {
      responder_.Finish(
          proto::BootstrapPayload(),
          grpc::Status(grpc::StatusCode::INTERNAL, "Receiver no longer exists"),
          this);
    }
  } else if (status_ == CallProgress::kCompleted) {
    delete this;
  }
}

// CompleteBootstrapReceiverCall ----------------------------------------------

CompleteBootstrapReceiverCall::CompleteBootstrapReceiverCall(
    CallContext* call_data)
    : AsyncHubReceiverCall(call_data), responder_(&context_) {
  call_ctx_->service->RequestCompleteBootstrap(
      &context_, &request_, &responder_, call_ctx_->cq, call_ctx_->cq, this);
}

void CompleteBootstrapReceiverCall::Proceed(bool ok) {
  if (!ok) {
    delete this;
    return;
  }

  if (status_ == CallProgress::kProcessing) {
    new CompleteBootstrapReceiverCall(call_ctx_);

    if (auto receiver = call_ctx_->receiver.lock()) {
      absl::Status result = receiver->OnBootstrapComplete(request_.space_id(),
                                                          request_.success());
      proto::Response response;
      if (result.ok()) {
        response.set_success(true);
        responder_.Finish(response, grpc::Status::OK, this);
      } else {
        response.set_success(false);
        response.set_error_message(std::string(result.message()));
        responder_.Finish(response, grpc::Status::OK, this);
      }
    } else {
      proto::Response response;
      response.set_success(false);
      response.set_error_message("Receiver no longer exists");
      responder_.Finish(
          response,
          grpc::Status(grpc::StatusCode::INTERNAL, "Receiver no longer exists"),
          this);
    }
    status_ = CallProgress::kCompleted;
  } else if (status_ == CallProgress::kCompleted) {
    delete this;
  }
}

// DisconnectFromHubReceiverCall ----------------------------------------------

DisconnectFromHubReceiverCall::DisconnectFromHubReceiverCall(
    CallContext* call_data)
    : AsyncHubReceiverCall(call_data), responder_(&context_) {
  call_ctx_->service->RequestDisconnectFromHub(
      &context_, &request_, &responder_, call_ctx_->cq, call_ctx_->cq, this);
}

void DisconnectFromHubReceiverCall::Proceed(bool ok) {
  if (!ok) {
    delete this;
    return;
  }

  if (status_ == CallProgress::kProcessing) {
    new DisconnectFromHubReceiverCall(call_ctx_);

    if (auto receiver = call_ctx_->receiver.lock()) {
      receiver->OnClientDisconnected(request_.space_id());
    }

    proto::Response response;
    response.set_success(true);
    status_ = CallProgress::kCompleted;
    responder_.Finish(response, grpc::Status::OK, this);
  } else if (status_ == CallProgress::kCompleted) {
    delete this;
  }
}

// UpdateActorStateReceiverCall -----------------------------------------------

UpdateActorStateReceiverCall::UpdateActorStateReceiverCall(
    CallContext* call_data)
    : AsyncHubReceiverCall(call_data), responder_(&context_) {
  call_ctx_->service->RequestUpdateActorState(
      &context_, &request_, &responder_, call_ctx_->cq, call_ctx_->cq, this);
}

void UpdateActorStateReceiverCall::Proceed(bool ok) {
  if (!ok) {
    delete this;
    return;
  }

  if (status_ == CallProgress::kProcessing) {
    new UpdateActorStateReceiverCall(call_ctx_);

    if (auto receiver = call_ctx_->receiver.lock()) {
      receiver->OnUpdateActorState(
          request_.space_id(), request_.actor_id(), request_.actor_shared_id(),
          Eigen::Vector2d(request_.x(), request_.y()), request_.yaw());
    }

    proto::Response response;
    response.set_success(true);
    status_ = CallProgress::kCompleted;
    responder_.Finish(response, grpc::Status::OK, this);
  } else if (status_ == CallProgress::kCompleted) {
    delete this;
  }
}

// ShareElementReceiverCall ---------------------------------------------------

ShareElementReceiverCall::ShareElementReceiverCall(CallContext* call_data)
    : AsyncHubReceiverCall(call_data), responder_(&context_) {
  call_ctx_->service->RequestShareElement(&context_, &request_, &responder_,
                                          call_ctx_->cq, call_ctx_->cq, this);
}

void ShareElementReceiverCall::Proceed(bool ok) {
  if (!ok) {
    delete this;
    return;
  }

  if (status_ == CallProgress::kProcessing) {
    new ShareElementReceiverCall(call_ctx_);

    if (auto receiver = call_ctx_->receiver.lock()) {
      receiver->OnClientShareElement(request_.space_id(),
                                     FromProto(request_.event()));
    }

    proto::Response response;
    response.set_success(true);
    status_ = CallProgress::kCompleted;
    responder_.Finish(response, grpc::Status::OK, this);
  } else if (status_ == CallProgress::kCompleted) {
    delete this;
  }
}

// PropagateRelationReceiverCall ----------------------------------------------

PropagateRelationReceiverCall::PropagateRelationReceiverCall(
    CallContext* call_data)
    : AsyncHubReceiverCall(call_data), responder_(&context_) {
  call_ctx_->service->RequestPropagateRelation(
      &context_, &request_, &responder_, call_ctx_->cq, call_ctx_->cq, this);
}

void PropagateRelationReceiverCall::Proceed(bool ok) {
  if (!ok) {
    delete this;
    return;
  }

  if (status_ == CallProgress::kProcessing) {
    new PropagateRelationReceiverCall(call_ctx_);

    if (auto receiver = call_ctx_->receiver.lock()) {
      receiver->OnClientRelationUpdate(request_.space_id(),
                                       FromProto(request_.update()));
    }

    proto::Response response;
    response.set_success(true);
    status_ = CallProgress::kCompleted;
    responder_.Finish(response, grpc::Status::OK, this);
  } else if (status_ == CallProgress::kCompleted) {
    delete this;
  }
}

// SubscribeAlarmTag ----------------------------------------------------------

SubscribeAlarmTag::SubscribeAlarmTag(CallContext* ctx,
                                     SubscribeReceiverCall* parent)
    : AsyncHubReceiverCall(ctx), parent_(parent) {}

void SubscribeAlarmTag::Proceed(bool ok) { parent_->OnAlarmFired(ok); }

// SubscribeReceiverCall ------------------------------------------------------

SubscribeReceiverCall::SubscribeReceiverCall(CallContext* ctx)
    : AsyncHubReceiverCall(ctx), writer_(&context_), alarm_tag_(ctx, this) {
  call_ctx_->service->RequestSubscribe(&context_, &request_, &writer_,
                                       call_ctx_->cq, call_ctx_->cq, this);
}

void SubscribeReceiverCall::NotifyDisconnected() {
  if (auto receiver = call_ctx_->receiver.lock()) {
    receiver->OnClientDisconnected(request_.space_id());
  }
}

void SubscribeReceiverCall::AccountStream() {
  if (accounted_) return;
  if (auto receiver = call_ctx_->receiver.lock()) {
    accounted_ = true;
    receiver->NoteStreamStarted();
  }
}

void SubscribeReceiverCall::Destroy() {
  if (accounted_) {
    accounted_ = false;
    if (auto receiver = call_ctx_->receiver.lock()) {
      receiver->NoteStreamRetired();
    }
  }
  delete this;
}

void SubscribeReceiverCall::ArmAlarmLocked() {
  alarm_active_ = true;
  // Monotonic epoch is always before ExecCtx::Now(), so RunAfter takes the
  // immediate path and never touches the timer heap. A REALTIME "now" deadline
  // is converted against the process epoch and can be scheduled a full second
  // out; this process has one server CQ thread, so that delay blocks
  // CompleteBootstrap until the client has already hit its deadline.
  alarm_.Set(call_ctx_->cq, gpr_time_0(GPR_CLOCK_MONOTONIC),
             static_cast<void*>(&alarm_tag_));
}

void SubscribeReceiverCall::Enqueue(proto::SpaceEvent event) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (finish_pending_ || state_ != State::kStreaming) {
    return;
  }
  write_queue_.push(std::move(event));
  if (!writing_ && !metadata_pending_) {
    writing_ = true;
    ArmAlarmLocked();
  }
}

void SubscribeReceiverCall::OnAlarmFired(bool ok) {
  bool should_delete = false;
  bool should_finish = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    alarm_active_ = false;
    if (closing_) {
      should_delete = true;
    } else if (!ok) {
      // Alarm itself failed (e.g. CQ shutting down concurrently with a
      // transport-level stream failure). Mark completed so a reentrant
      // Close() call from NotifyDisconnected() -> ~RemoteSpace() below is a
      // no-op instead of re-arming alarm_ on an object we're about to free.
      state_ = State::kCompleted;
      writing_ = false;
      should_delete = true;
    } else if (finish_pending_) {
      should_finish = true;
    }
  }
  if (should_delete) {
    // closing_/!ok here are both stream-failure paths (Close() sets
    // finish_pending_, never closing_) -- always a transport-level failure,
    // so always notify.
    NotifyDisconnected();
    Destroy();
    return;
  }
  if (should_finish) {
    state_ = State::kCompleted;
    writer_.Finish(grpc::Status::OK, this);
    return;
  }
  if (ok) TryWrite();
}

void SubscribeReceiverCall::Close() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ != State::kStreaming || finish_pending_) {
    return;
  }
  finish_pending_ = true;
  while (!write_queue_.empty()) {
    write_queue_.pop();
  }
  if (!writing_ && !metadata_pending_) {
    writing_ = true;
    ArmAlarmLocked();
  }
}

void SubscribeReceiverCall::TryWrite() {
  proto::SpaceEvent event;
  bool should_finish = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (metadata_pending_) {
      return;
    }
    if (finish_pending_) {
      writing_ = false;
      should_finish = true;
    } else if (write_queue_.empty()) {
      writing_ = false;
      return;
    } else {
      event = std::move(write_queue_.front());
      write_queue_.pop();
    }
  }
  if (should_finish) {
    state_ = State::kCompleted;
    writer_.Finish(grpc::Status::OK, this);
    return;
  }
  writer_.Write(event, static_cast<void*>(this));
}

void SubscribeReceiverCall::Proceed(bool ok) {
  if (state_ == State::kWaiting) {
    if (!ok) {
      Destroy();
      return;
    }
    new SubscribeReceiverCall(call_ctx_);
    state_ = State::kStreaming;
    // The client's StartCall does not complete until initial metadata
    // arrives. Sending it here, before the call is published to the pusher,
    // keeps CompleteBootstrap from waiting on the first streamed write.
    // Account first so Stop() waits for this Finish even though the call is
    // not in remote_spaces_ yet.
    AccountStream();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      metadata_pending_ = true;
      writing_ = true;
    }
    writer_.SendInitialMetadata(this);
    return;
  }

  if (state_ == State::kStreaming && metadata_pending_) {
    metadata_pending_ = false;
    if (!ok) {
      writing_ = false;
      // Fall through to the streaming failure path below.
    } else if (auto receiver = call_ctx_->receiver.lock()) {
      absl::Status status = receiver->OnClientSubscribed(
          request_.space_id(),
          std::shared_ptr<SubscribeReceiverCall>(
              this, [](SubscribeReceiverCall*) {}));  // non-owning shared_ptr
      if (!status.ok()) {
        SPATIALBLEND_LOG(spatialblend::LogSeverity::kError,
                      "SubscribeReceiverCall: OnClientSubscribed failed: %s",
                      status.ToString().c_str());
        state_ = State::kCompleted;
        writer_.Finish(grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                    std::string(status.message())),
                       this);
        return;
      }
      bool should_finish = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (finish_pending_) {
          writing_ = true;
          should_finish = true;
        }
      }
      if (should_finish) {
        state_ = State::kCompleted;
        writer_.Finish(grpc::Status::OK, this);
        return;
      }
      TryWrite();
      return;
    } else {
      state_ = State::kCompleted;
      writer_.Finish(
          grpc::Status(grpc::StatusCode::INTERNAL, "Receiver no longer exists"),
          this);
      return;
    }
  }

  if (state_ == State::kStreaming) {
    if (!ok) {
      bool should_delete;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = State::kCompleted;
        writing_ = false;
        if (alarm_active_) {
          closing_ = true;
          should_delete = false;
        } else {
          should_delete = true;
        }
      }
      if (should_delete) {
        // Transport-level failure (client crash / abrupt disconnect) with no
        // alarm in flight -- notify HubReceiver so it drops the RemoteSpace
        // entry pointing at this object before it goes away.
        NotifyDisconnected();
        Destroy();
      }
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (finish_pending_) {
        writing_ = false;
        state_ = State::kCompleted;
        writer_.Finish(grpc::Status::OK, this);
        return;
      }
    }
    TryWrite();
    return;
  }

  if (state_ == State::kCompleted) {
    Destroy();
    return;
  }
}

}  // namespace spatialblend
