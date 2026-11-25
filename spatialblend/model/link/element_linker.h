// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_LINK_ELEMENT_LINKER_H_
#define SPATIALBLEND_LINK_ELEMENT_LINKER_H_

#include <map>
#include <functional>
#include <mutex>
#include <optional>
#include <vector>

#include "spatialblend/base/common/types.h"

namespace spatialblend {

// ElementLinker maintains one-to-many links between source elements and target
// elements across spaces. Each source element_handle_t maps to a set of target
// element_ids keyed by target space_id. All methods are thread-safe.
class ElementLinker {
 public:
  struct OwnedLink {
    element_handle_t source;
    element_handle_t target;
    uint64_t version;
  };
  // Creates a link: source -> target (stored as target.space_id ->
  // target.element_id). If a target in that space already exists it is
  // replaced.
  void CreateLink(element_handle_t source, element_handle_t target);

  // Installs links as one all-or-nothing operation. A source may have at most
  // one target in a particular target space: an existing different target (or
  // conflicting entries within links) fails without changing the map. Exact
  // existing pairs are accepted but are not reported as installed.
  bool InstallLinksIfUnconflicted(
      const std::vector<std::pair<element_handle_t, element_handle_t>>& links,
      std::vector<OwnedLink>* installed);

  // Independent strict ownership path. No borrowed/idempotent pairs, duplicate
  // sources/targets, same-space or empty batches. All slots must be unlinked;
  // success owns every exact version. Targets cannot also be link sources:
  // that would shadow inverse translation. Nonblocking; failure makes no edit.
  bool TryInstallOwnedLinks(
      const std::vector<std::pair<element_handle_t, element_handle_t>>& links,
      std::vector<OwnedLink>* installed,
      const std::function<bool()>& try_validate_endpoints = {});
  // Endpoint predicate is an internal non-mutating TRY-lock preflight only,
  // never application code. It runs under the linker lock so owned subtree
  // removal cannot interleave between its link check and this installation.
  // 1=current exact versions, 0=missing/replaced/empty, -4=contended.
  int TryValidateOwnedLinks(const std::vector<OwnedLink>& links,
      const std::function<bool()>& try_validate_endpoints = {}) const;
  // Best-effort cleanup of still-exact owned versions under one try-lock;
  // replacements survive. Busy makes no removal (never a partial Busy result).
  int TryRemoveOwnedLinks(const std::vector<OwnedLink>& links);

  // Removes only this exact source -> target pair. Unlike RemoveSource and
  // RemoveTarget, this never affects another pair sharing either endpoint.
  bool RemoveLink(element_handle_t source, element_handle_t target);

  // Removes an installed pair only if it is still the same generation. This
  // protects an owner from deleting a later remove/recreate of identical
  // endpoints (the ABA case).
  bool RemoveOwnedLink(const OwnedLink& link);

  // Removes the link record for source. Returns false if source not found.
  bool RemoveSource(element_handle_t source);

  // Removes target from whichever source link contains it. Returns false if
  // target not found as a target in any link.
  bool RemoveTarget(element_handle_t target);

  // Returns the source handle whose link contains target, or nullopt.
  std::optional<element_handle_t> GetSource(element_handle_t target) const;

  // Returns every source handle whose link contains target (a target linked
  // from more than one source), in source order.
  std::vector<element_handle_t> GetSourcesOf(element_handle_t target) const;

  // Returns the target handle for source in target_space, or nullopt.
  std::optional<element_handle_t> GetTarget(element_handle_t source,
                                            space_id_t target_space) const;

  // Returns all targets for source across all spaces.
  std::vector<element_handle_t> GetTargets(element_handle_t source) const;

  // Returns all source handles whose space_id matches source_space.
  std::vector<element_handle_t> GetSourcesFromSpace(
      space_id_t source_space) const;

  // Returns all target handles whose space_id matches target_space.
  std::vector<element_handle_t> GetTargetsFromSpace(
      space_id_t target_space) const;

  // Removes all links where the source is in space, and all target entries
  // whose space_id matches space (mirroring v1 SpaceLinker semantics).
  void RemoveAllLinksForSpace(space_id_t space);

  bool IsSource(element_handle_t h) const;
  bool IsTarget(element_handle_t h) const;

  bool IsLinked(element_handle_t source, element_handle_t target) const;
  bool IsLinked(element_handle_t source) const;

  // Translates handle to an element_id in target_space.
  // - Same space: returns handle.element_id directly.
  // - handle is a source: returns the target element_id in target_space.
  // - handle is a target: finds the source, then resolves to target_space.
  // Returns kInvalidElementId if translation is not possible.
  element_id_t Translate(element_handle_t handle,
                         space_id_t target_space) const;

  // Monotonic graph generation.  It changes only after a successful mutation
  // of the correspondence graph; rejected installs and no-op removals do not
  // consume a generation.  Consumers that perform several lookups can fence
  // a transaction by comparing this value before and after the lookups.
  uint64_t GetGraphRevision() const;

 private:
  // source_handle -> { target_space_id -> target_element_id }
  struct TargetLink {
    element_id_t element_id;
    uint64_t version;
  };
  std::map<element_handle_t, std::map<space_id_t, TargetLink>> links_;
  uint64_t next_version_ = 1;
  uint64_t graph_revision_ = 1;
  mutable std::recursive_mutex mutex_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_LINK_ELEMENT_LINKER_H_
