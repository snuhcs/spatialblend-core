// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/local_context.h"

namespace spatialblend {

LocalContext::LocalContext() : hub_(std::make_unique<SpaceHub>()) {}

std::unique_ptr<LocalContext> LocalContext::Create() {
  return std::unique_ptr<LocalContext>(new LocalContext());
}

Space* LocalContext::AddSpace() {
  auto space = std::make_unique<Space>(receiver_input_clock_);
  space->SetSpaceId(hub_->AllocateSpaceId());
  space->SetHub(hub_.get());
  hub_->Connect(space.get());
  Space* ptr = space.get();
  spaces_.push_back(std::move(space));
  return ptr;
}

SpaceHub* LocalContext::GetHub() { return hub_.get(); }

}  // namespace spatialblend
