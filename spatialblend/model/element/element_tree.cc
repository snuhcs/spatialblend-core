// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/element_tree.h"

#include <algorithm>
#include <queue>

namespace spatialblend {

absl::Status ElementTree::AddElement(element_id_t id,
                                     element_id_t parent_id) {
  if (elements_.count(id)) {
    return absl::AlreadyExistsError("Element already exists");
  }
  if (parent_id != kInvalidElementId && !elements_.count(parent_id)) {
    return absl::NotFoundError("Parent element not found");
  }

  auto elem = std::make_unique<Element>(id);
  elem->SetParentId(parent_id);

  if (parent_id == kInvalidElementId) {
    root_id_ = id;
  } else {
    elements_[parent_id]->AddChildId(id);
  }

  elements_[id] = std::move(elem);
  return absl::OkStatus();
}

absl::Status ElementTree::RemoveElement(element_id_t id) {
  auto it = elements_.find(id);
  if (it == elements_.end()) {
    return absl::NotFoundError("Element not found");
  }

  auto subtree = GetSubtreeIds(id);

  element_id_t parent_id = it->second->GetParentId();
  if (parent_id != kInvalidElementId) {
    auto* parent = GetElement(parent_id);
    if (parent) parent->RemoveChildId(id);
  }

  for (element_id_t sub_id : subtree) {
    elements_.erase(sub_id);
  }

  if (root_id_ == id) root_id_ = kInvalidElementId;
  return absl::OkStatus();
}

absl::Status ElementTree::ReparentElement(element_id_t id,
                                          element_id_t new_parent_id) {
  auto* elem = GetElement(id);
  if (!elem) return absl::NotFoundError("Element not found");
  if (new_parent_id != kInvalidElementId && !GetElement(new_parent_id))
    return absl::NotFoundError("New parent not found");

  // Cycle detection: new_parent must not be in subtree of id
  if (new_parent_id != kInvalidElementId) {
    auto subtree = GetSubtreeIds(id);
    for (element_id_t sub_id : subtree) {
      if (sub_id == new_parent_id) {
        return absl::InvalidArgumentError(
            "Cannot reparent: new parent is a descendant");
      }
    }
  }

  element_id_t old_parent = elem->GetParentId();
  if (old_parent != kInvalidElementId) {
    GetElement(old_parent)->RemoveChildId(id);
  }

  elem->SetParentId(new_parent_id);
  if (new_parent_id != kInvalidElementId) {
    GetElement(new_parent_id)->AddChildId(id);
  } else {
    root_id_ = id;
  }
  return absl::OkStatus();
}

Element* ElementTree::GetElement(element_id_t id) {
  auto it = elements_.find(id);
  return it != elements_.end() ? it->second.get() : nullptr;
}

const Element* ElementTree::GetElement(element_id_t id) const {
  auto it = elements_.find(id);
  return it != elements_.end() ? it->second.get() : nullptr;
}

Element* ElementTree::GetRoot() { return GetElement(root_id_); }
const Element* ElementTree::GetRoot() const { return GetElement(root_id_); }

std::vector<element_id_t> ElementTree::GetSubtreeIds(element_id_t id) const {
  std::vector<element_id_t> result;
  const auto* elem = GetElement(id);
  if (!elem) return result;

  std::queue<element_id_t> queue;
  queue.push(id);
  while (!queue.empty()) {
    element_id_t current = queue.front();
    queue.pop();
    result.push_back(current);
    const auto* e = GetElement(current);
    if (e) {
      for (element_id_t child : e->GetChildIds()) {
        queue.push(child);
      }
    }
  }
  return result;
}

std::vector<element_id_t> ElementTree::GetDirectChildIds(
    element_id_t id) const {
  const auto* elem = GetElement(id);
  if (!elem) return {};
  return elem->GetChildIds();
}

std::vector<element_id_t> ElementTree::GetAncestorIds(element_id_t id) const {
  std::vector<element_id_t> result;
  const auto* elem = GetElement(id);
  if (!elem) return result;

  element_id_t current = elem->GetParentId();
  while (current != kInvalidElementId) {
    result.push_back(current);
    const auto* parent = GetElement(current);
    if (!parent) break;
    current = parent->GetParentId();
  }
  return result;
}

std::vector<const Element*> ElementTree::GetElementsBySharedId(
    shared_element_id_t shared_id) const {
  std::vector<const Element*> result;
  for (const auto& [_, elem] : elements_) {
    if (elem->GetSharedId() == shared_id) {
      result.push_back(elem.get());
    }
  }
  return result;
}

Element* ElementTree::GetElementBySharedId(shared_element_id_t shared_id) {
  return const_cast<Element*>(
      static_cast<const ElementTree*>(this)->GetElementBySharedId(shared_id));
}

const Element* ElementTree::GetElementBySharedId(
    shared_element_id_t shared_id) const {
  // Declared rule: smallest element id among the holders.
  const Element* best = nullptr;
  for (const auto& [id, elem] : elements_) {
    if (elem->GetSharedId() != shared_id) continue;
    if (!best || id < best->GetId()) best = elem.get();
  }
  return best;
}

std::vector<element_id_t> ElementTree::GetAllIds() const {
  std::vector<element_id_t> ids;
  ids.reserve(elements_.size());
  for (const auto& [id, _] : elements_) ids.push_back(id);
  std::sort(ids.begin(), ids.end());
  return ids;
}

Eigen::Matrix4d ElementTree::GetWorldTransform(element_id_t id) const {
  auto ancestors = GetAncestorIds(id);
  Eigen::Matrix4d world = Eigen::Matrix4d::Identity();

  for (auto it = ancestors.rbegin(); it != ancestors.rend(); ++it) {
    const auto* anc = GetElement(*it);
    if (anc) world = world * anc->GetLocalTransform();
  }

  const auto* self = GetElement(id);
  if (self) world = world * self->GetLocalTransform();
  return world;
}

size_t ElementTree::Size() const { return elements_.size(); }

}  // namespace spatialblend
