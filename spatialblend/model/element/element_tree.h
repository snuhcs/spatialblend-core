// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_ELEMENT_ELEMENT_TREE_H_
#define SPATIALBLEND_ELEMENT_ELEMENT_TREE_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "spatialblend/model/element/element.h"

namespace spatialblend {

class ElementTree {
 public:
  ElementTree() = default;

  // Non-copyable but movable (contains unique_ptr members).
  ElementTree(const ElementTree&) = delete;
  ElementTree& operator=(const ElementTree&) = delete;
  ElementTree(ElementTree&&) = default;
  ElementTree& operator=(ElementTree&&) = default;

  absl::Status AddElement(element_id_t id,
                          element_id_t parent_id = kInvalidElementId);

  absl::Status RemoveElement(element_id_t id);

  absl::Status ReparentElement(element_id_t id, element_id_t new_parent_id);

  Element* GetElement(element_id_t id);
  const Element* GetElement(element_id_t id) const;
  Element* GetRoot();
  const Element* GetRoot() const;

  std::vector<element_id_t> GetSubtreeIds(element_id_t id) const;
  std::vector<element_id_t> GetDirectChildIds(element_id_t id) const;
  std::vector<element_id_t> GetAncestorIds(element_id_t id) const;

  // Every element id in the tree (all roots and depths), ascending. Used by
  // scope resolution that is not limited to one subtree (Facing scope is
  // line of sight at any hierarchy depth).
  std::vector<element_id_t> GetAllIds() const;

  std::vector<const Element*> GetElementsBySharedId(
      shared_element_id_t shared_id) const;

  // Declared resolution rule: a logical identity resolves to exactly one
  // element per space. Registration does not reject duplicates (the legacy
  // Space API and its source-evidence / phase-collision contracts allow
  // them), so when several elements carry `shared_id` the one with the
  // SMALLEST element id wins -- deterministic and independent of registration
  // order, never "whichever the hash map yields first". Callers that need
  // every holder use GetElementsBySharedId.
  Element* GetElementBySharedId(shared_element_id_t shared_id);
  const Element* GetElementBySharedId(shared_element_id_t shared_id) const;

  Eigen::Matrix4d GetWorldTransform(element_id_t id) const;

  size_t Size() const;

 private:
  std::unordered_map<element_id_t, std::unique_ptr<Element>> elements_;
  element_id_t root_id_ = kInvalidElementId;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_ELEMENT_ELEMENT_TREE_H_
