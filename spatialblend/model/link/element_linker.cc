// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/link/element_linker.h"

#include <set>

namespace spatialblend {

void ElementLinker::CreateLink(element_handle_t source,
                               element_handle_t target) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto& targets = links_[source];
  // CreateLink is an explicit ownership replacement, even when its endpoints
  // are identical.  Refresh its private version so an older OwnedLink lease
  // cannot delete the replacement (the exact-pair batch install API is the
  // intentionally idempotent/no-op path).
  targets[target.space_id] = {target.element_id, next_version_++};
  ++graph_revision_;
}

bool ElementLinker::InstallLinksIfUnconflicted(
    const std::vector<std::pair<element_handle_t, element_handle_t>>& links,
    std::vector<OwnedLink>* installed) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::map<std::pair<element_handle_t, space_id_t>, element_id_t> requested;
  for (const auto& link : links) {
    const auto key = std::make_pair(link.first, link.second.space_id);
    auto [it, inserted] = requested.emplace(key, link.second.element_id);
    if (!inserted && it->second != link.second.element_id) return false;
  }
  bool changed = false;
  for (const auto& [key, element_id] : requested) {
    auto source_it = links_.find(key.first);
    if (source_it == links_.end()) continue;
    auto target_it = source_it->second.find(key.second);
    if (target_it != source_it->second.end() &&
        target_it->second.element_id != element_id) {
      return false;
    }
  }
  if (installed) installed->clear();
  for (const auto& [key, element_id] : requested) {
    auto& targets = links_[key.first];
    auto [it, inserted] =
        targets.emplace(key.second, TargetLink{element_id, next_version_});
    if (inserted && installed) {
      installed->push_back(
          {key.first, {key.second, element_id}, next_version_});
    }
    if (inserted) {
      ++next_version_;
      changed = true;
    }
  }
  if (changed) ++graph_revision_;
  return true;
}

bool ElementLinker::RemoveLink(element_handle_t source,
                               element_handle_t target) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto source_it = links_.find(source);
  if (source_it == links_.end()) return false;
  auto target_it = source_it->second.find(target.space_id);
  if (target_it == source_it->second.end() ||
      target_it->second.element_id != target.element_id) {
    return false;
  }
  source_it->second.erase(target_it);
  if (source_it->second.empty()) links_.erase(source_it);
  ++graph_revision_;
  return true;
}

bool ElementLinker::RemoveOwnedLink(const OwnedLink& link) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto source_it = links_.find(link.source);
  if (source_it == links_.end()) return false;
  auto target_it = source_it->second.find(link.target.space_id);
  if (target_it == source_it->second.end() ||
      target_it->second.element_id != link.target.element_id ||
      target_it->second.version != link.version) {
    return false;
  }
  source_it->second.erase(target_it);
  if (source_it->second.empty()) links_.erase(source_it);
  ++graph_revision_;
  return true;
}

bool ElementLinker::RemoveSource(element_handle_t source) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = links_.find(source);
  if (it == links_.end()) {
    return false;
  }
  links_.erase(it);
  ++graph_revision_;
  return true;
}

bool ElementLinker::RemoveTarget(element_handle_t target) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (auto& [source_handle, targets] : links_) {
    auto it = targets.find(target.space_id);
    if (it != targets.end() && it->second.element_id == target.element_id) {
      targets.erase(it);
      ++graph_revision_;
      return true;
    }
  }
  return false;
}

std::optional<element_handle_t> ElementLinker::GetSource(
    element_handle_t target) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (const auto& [source_handle, targets] : links_) {
    auto it = targets.find(target.space_id);
    if (it != targets.end() && it->second.element_id == target.element_id) {
      return source_handle;
    }
  }
  return std::nullopt;
}

std::vector<element_handle_t> ElementLinker::GetSourcesOf(
    element_handle_t target) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::vector<element_handle_t> sources;
  for (const auto& [source_handle, targets] : links_) {
    auto it = targets.find(target.space_id);
    if (it != targets.end() && it->second.element_id == target.element_id) {
      sources.push_back(source_handle);
    }
  }
  return sources;
}

std::optional<element_handle_t> ElementLinker::GetTarget(
    element_handle_t source, space_id_t target_space) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto link_it = links_.find(source);
  if (link_it == links_.end()) {
    return std::nullopt;
  }
  auto target_it = link_it->second.find(target_space);
  if (target_it == link_it->second.end()) {
    return std::nullopt;
  }
  return element_handle_t{target_space, target_it->second.element_id};
}

std::vector<element_handle_t> ElementLinker::GetTargets(
    element_handle_t source) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto link_it = links_.find(source);
  if (link_it == links_.end()) {
    return {};
  }
  std::vector<element_handle_t> targets;
  for (const auto& [space_id, element_id] : link_it->second) {
    targets.push_back({space_id, element_id.element_id});
  }
  return targets;
}

std::vector<element_handle_t> ElementLinker::GetSourcesFromSpace(
    space_id_t source_space) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::vector<element_handle_t> sources;
  for (const auto& [source_handle, targets] : links_) {
    if (source_handle.space_id == source_space) {
      sources.push_back(source_handle);
    }
  }
  return sources;
}

std::vector<element_handle_t> ElementLinker::GetTargetsFromSpace(
    space_id_t target_space) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::vector<element_handle_t> targets;
  for (const auto& [source_handle, target_map] : links_) {
    auto it = target_map.find(target_space);
    if (it != target_map.end()) {
      targets.push_back({target_space, it->second.element_id});
    }
  }
  return targets;
}

void ElementLinker::RemoveAllLinksForSpace(space_id_t space) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Step 1: remove all source links from this space
  bool changed = false;
  auto sources = GetSourcesFromSpace(space);
  for (const auto& source : sources) {
    links_.erase(source);
    changed = true;
  }

  // Step 2: remove all target entries in this space from remaining links
  for (auto& [source_handle, targets] : links_) {
    changed |= targets.erase(space) != 0;
  }
  if (changed) ++graph_revision_;
}

bool ElementLinker::IsSource(element_handle_t h) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return links_.count(h) > 0;
}

bool ElementLinker::IsTarget(element_handle_t h) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (const auto& [source_handle, targets] : links_) {
    auto it = targets.find(h.space_id);
    if (it != targets.end() && it->second.element_id == h.element_id) {
      return true;
    }
  }
  return false;
}

bool ElementLinker::IsLinked(element_handle_t source,
                             element_handle_t target) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto link_it = links_.find(source);
  if (link_it == links_.end()) {
    return false;
  }
  auto it = link_it->second.find(target.space_id);
  return it != link_it->second.end() &&
         it->second.element_id == target.element_id;
}

bool ElementLinker::IsLinked(element_handle_t source) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto link_it = links_.find(source);
  if (link_it == links_.end()) {
    return false;
  }
  return !link_it->second.empty();
}

element_id_t ElementLinker::Translate(element_handle_t handle,
                                      space_id_t target_space) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  // No-op: same space
  if (handle.space_id == target_space) {
    return handle.element_id;
  }

  if (IsSource(handle)) {
    auto target = GetTarget(handle, target_space);
    if (target.has_value()) {
      return target->element_id;
    }
  } else if (IsTarget(handle)) {
    auto source = GetSource(handle);
    if (source.has_value()) {
      if (source->space_id == target_space) {
        return source->element_id;
      }
      auto target = GetTarget(*source, target_space);
      if (target.has_value()) {
        return target->element_id;
      }
    }
  }
  return kInvalidElementId;
}

bool ElementLinker::TryInstallOwnedLinks(
    const std::vector<std::pair<element_handle_t, element_handle_t>>& links,
    std::vector<OwnedLink>* installed,
    const std::function<bool()>& try_validate_endpoints) {
  if (!installed) return false;
  installed->clear();
  if (links.empty() || links.size() > 128) return false;
  std::unique_lock<std::recursive_mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return false;
  std::set<element_handle_t> sources, targets;
  for (const auto& [source, target] : links) {
    if (source.space_id == target.space_id ||
        !sources.insert(source).second || !targets.insert(target).second)
      return false;
    auto old = links_.find(source);
    if (old != links_.end() && old->second.count(target.space_id)) return false;
    // Translate gives an endpoint's outgoing links precedence over its inverse.
    // A target that is also a source would shadow this owned reverse mapping.
    if (links_.count(target)) return false;
    // Strict injectivity includes links outside this batch. A second source
    // already mapped to this target would make reverse hierarchy ambiguous.
    for (const auto& [other, existing] : links_) {
      auto found = existing.find(target.space_id);
      if (found != existing.end() && found->second.element_id == target.element_id)
        return false;
    }
  }
  for (const auto& target : targets)
    if (sources.count(target)) return false;
  if (try_validate_endpoints && !try_validate_endpoints()) return false;
  // Recursive mutex keeps the complete preflight/install atomic. The legacy
  // helper is preserved unchanged for explicitly idempotent registrations.
  return InstallLinksIfUnconflicted(links, installed);
}

int ElementLinker::TryValidateOwnedLinks(const std::vector<OwnedLink>& links,
    const std::function<bool()>& try_validate_endpoints) const {
  if (links.empty() || links.size() > 128) return 0;
  std::unique_lock<std::recursive_mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return -4;
  for (const auto& owned : links) {
    if (links_.count(owned.target)) return 0;
    auto source = links_.find(owned.source);
    if (source == links_.end()) return 0;
    auto target = source->second.find(owned.target.space_id);
    if (target == source->second.end() || target->second.element_id != owned.target.element_id ||
        target->second.version != owned.version) return 0;
    for (const auto& [other, existing] : links_) {
      if (other == owned.source) continue;
      auto found = existing.find(owned.target.space_id);
      if (found != existing.end() && found->second.element_id == owned.target.element_id) return 0;
    }
  }
  if (try_validate_endpoints && !try_validate_endpoints()) return 0;
  return 1;
}

int ElementLinker::TryRemoveOwnedLinks(const std::vector<OwnedLink>& links) {
  std::unique_lock<std::recursive_mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return -4;
  for (const auto& link : links) RemoveOwnedLink(link);
  return 1;
}

uint64_t ElementLinker::GetGraphRevision() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return graph_revision_;
}

}  // namespace spatialblend
