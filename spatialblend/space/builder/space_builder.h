// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SIMULATOR_SPACE_BUILDER_H_
#define SPATIALBLEND_SIMULATOR_SPACE_BUILDER_H_

#include <memory>
#include <string>
#include <vector>

#include "spatialblend/space/builder/actor_handle.h"
#include "spatialblend/space/builder/element_handle.h"
#include "spatialblend/space/space.h"

namespace spatialblend {

class SpaceBuilder {
 public:
  explicit SpaceBuilder(Space* space,
                        DirtyActorCallback on_dirty = nullptr)
      : space_(space),
        on_dirty_(std::move(on_dirty)),
        next_id_(std::make_shared<int64_t>(0)) {}

  ElementHandle AddElement(const std::string& name) {
    int64_t id = ++(*next_id_);
    (void)space_->GetElementTree()->AddElement(static_cast<element_id_t>(id));
    space_->GetElementTree()->GetElement(static_cast<element_id_t>(id))
        ->SetTag(name);
    return ElementHandle(space_, static_cast<element_id_t>(id), next_id_);
  }

  ActorHandle AddActor() {
    auto actor_id = space_->CreateActor({});
    return ActorHandle(space_, actor_id, on_dirty_);
  }

  Space* GetSpace() const { return space_; }

 private:
  Space* space_;
  DirtyActorCallback on_dirty_;
  std::shared_ptr<int64_t> next_id_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SIMULATOR_SPACE_BUILDER_H_
