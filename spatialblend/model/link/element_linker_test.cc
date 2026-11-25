// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/link/element_linker.h"

#include <gtest/gtest.h>

#include <chrono>
#include <future>

namespace spatialblend {
namespace test {

// Helpers
static element_handle_t H(space_id_t s, element_id_t e) { return {s, e}; }

TEST(OwnedHierarchyLinkTest, CompleteBatchOwnsEveryPairAndRemovalIsExact) {
  ElementLinker linker;
  std::vector<ElementLinker::OwnedLink> owned;
  ASSERT_TRUE(linker.TryInstallOwnedLinks(
      {{H(1, 1), H(2, 11)}, {H(1, 2), H(2, 12)}, {H(1, 3), H(2, 13)}}, &owned));
  ASSERT_EQ(owned.size(), 3u);
  EXPECT_EQ(linker.TryValidateOwnedLinks(owned), 1);
  linker.CreateLink(H(1, 8), H(2, 18));
  EXPECT_EQ(linker.TryRemoveOwnedLinks(owned), 1);
  EXPECT_EQ(linker.TryValidateOwnedLinks(owned), 0);
  EXPECT_FALSE(linker.IsLinked(H(1, 2), H(2, 12)));
  EXPECT_TRUE(linker.IsLinked(H(1, 8), H(2, 18)));
}

TEST(OwnedHierarchyLinkTest, ExistingExactOrConflictingPairCannotBeBorrowed) {
  ElementLinker linker;
  linker.CreateLink(H(1, 3), H(2, 13));
  const auto revision = linker.GetGraphRevision();
  for (const auto target : {H(2, 13), H(2, 14)}) {
    std::vector<ElementLinker::OwnedLink> owned;
    EXPECT_FALSE(linker.TryInstallOwnedLinks(
        {{H(1, 1), H(2, 11)}, {H(1, 3), target}}, &owned));
    EXPECT_TRUE(owned.empty());
    EXPECT_EQ(linker.GetGraphRevision(), revision);
    EXPECT_FALSE(linker.IsSource(H(1, 1)));
    EXPECT_TRUE(linker.IsLinked(H(1, 3), H(2, 13)));
  }
}

TEST(OwnedHierarchyLinkTest, InvalidOrAmbiguousBatchHasNoMutation) {
  ElementLinker linker;
  std::vector<ElementLinker::OwnedLink> owned;
  using Pair = std::pair<element_handle_t, element_handle_t>;
  const std::vector<std::vector<Pair>> invalid = {
      {}, {{H(1, 1), H(1, 2)}},
      {{H(1, 1), H(2, 11)}, {H(1, 1), H(2, 11)}},
      {{H(1, 1), H(2, 11)}, {H(1, 1), H(3, 21)}},
      {{H(1, 1), H(2, 11)}, {H(1, 2), H(2, 11)}},
  };
  const auto revision = linker.GetGraphRevision();
  for (const auto& pairs : invalid) {
    EXPECT_FALSE(linker.TryInstallOwnedLinks(pairs, &owned));
    EXPECT_TRUE(owned.empty());
    EXPECT_EQ(linker.GetGraphRevision(), revision);
  }
  std::vector<Pair> too_many;
  for (int i = 1; i <= 129; ++i) too_many.push_back({H(1, i), H(2, i)});
  EXPECT_FALSE(linker.TryInstallOwnedLinks(too_many, &owned));
  EXPECT_EQ(linker.GetGraphRevision(), revision);
  EXPECT_FALSE(linker.TryInstallOwnedLinks({{H(1, 1), H(2, 11)}}, nullptr));
  linker.CreateLink(H(1, 9), H(2, 11));
  EXPECT_FALSE(linker.TryInstallOwnedLinks({{H(1, 1), H(2, 11)}}, &owned));
  EXPECT_FALSE(linker.IsSource(H(1, 1)));
}

TEST(OwnedHierarchyLinkTest, ReplacementAndLegacyAmbiguityInvalidateOwnedBatch) {
  ElementLinker linker;
  std::vector<ElementLinker::OwnedLink> owned;
  ASSERT_TRUE(linker.TryInstallOwnedLinks({{H(1, 1), H(2, 11)}}, &owned));
  linker.CreateLink(H(1, 9), H(2, 11));
  EXPECT_EQ(linker.TryValidateOwnedLinks(owned), 0);
  linker.RemoveSource(H(1, 9));
  EXPECT_EQ(linker.TryValidateOwnedLinks(owned), 1);
  ASSERT_TRUE(linker.RemoveLink(H(1, 1), H(2, 11)));
  linker.CreateLink(H(1, 1), H(2, 11));
  EXPECT_EQ(linker.TryValidateOwnedLinks(owned), 0);
  EXPECT_EQ(linker.TryRemoveOwnedLinks(owned), 1);
  EXPECT_TRUE(linker.IsLinked(H(1, 1), H(2, 11)));
}

TEST(OwnedHierarchyLinkTest, EndpointPreflightRejectsWithoutGraphMutation) {
  ElementLinker linker;
  std::vector<ElementLinker::OwnedLink> owned;
  const auto revision = linker.GetGraphRevision();
  EXPECT_FALSE(linker.TryInstallOwnedLinks({{H(1, 1), H(2, 11)}}, &owned,
                                         [] { return false; }));
  EXPECT_TRUE(owned.empty());
  EXPECT_EQ(linker.GetGraphRevision(), revision);
  ASSERT_TRUE(linker.TryInstallOwnedLinks({{H(1, 1), H(2, 11)}}, &owned));
  EXPECT_EQ(linker.TryValidateOwnedLinks(owned, [] { return false; }), 0);
  EXPECT_EQ(linker.TryValidateOwnedLinks(owned, [] { return true; }), 1);
}

TEST(OwnedHierarchyLinkTest, OutgoingTargetCannotShadowReverseTranslation) {
  for (const auto third : {H(3, 21), H(1, 2)}) {
    ElementLinker linker;
    linker.CreateLink(H(2, 11), third);
    std::vector<ElementLinker::OwnedLink> owned;
    EXPECT_FALSE(linker.TryInstallOwnedLinks({{H(1, 1), H(2, 11)}}, &owned));
    EXPECT_FALSE(linker.IsSource(H(1, 1)));
  }
  ElementLinker linker;
  std::vector<ElementLinker::OwnedLink> owned;
  EXPECT_FALSE(linker.TryInstallOwnedLinks(
      {{H(1, 1), H(2, 11)}, {H(2, 11), H(3, 21)}}, &owned));
  // An unrelated incoming link to the source does not shadow either direction.
  linker.CreateLink(H(3, 21), H(1, 1));
  ASSERT_TRUE(linker.TryInstallOwnedLinks({{H(1, 1), H(2, 11)}}, &owned));
  EXPECT_EQ(linker.Translate(H(1, 1), 2), 11);
  EXPECT_EQ(linker.Translate(H(2, 11), 1), 1);
  linker.CreateLink(H(2, 11), H(3, 22));
  EXPECT_EQ(linker.TryValidateOwnedLinks(owned), 0);
  EXPECT_EQ(linker.TryRemoveOwnedLinks(owned), 1);
  EXPECT_TRUE(linker.IsLinked(H(2, 11), H(3, 22)));
}

TEST(OwnedHierarchyLinkTest, ContentionDoesNotPartiallyInstallOrRetire) {
  ElementLinker linker;
  std::vector<ElementLinker::OwnedLink> owned;
  ASSERT_TRUE(linker.TryInstallOwnedLinks({{H(1, 1), H(2, 11)}}, &owned));
  std::promise<void> entered, release;
  auto ready = entered.get_future();
  auto released = release.get_future();
  auto held = std::async(std::launch::async, [&] {
    return linker.TryValidateOwnedLinks(owned, [&] {
      entered.set_value();
      return released.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    });
  });
  const auto status = ready.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(status, std::future_status::ready);
  if (status == std::future_status::ready) {
    std::vector<ElementLinker::OwnedLink> second;
    EXPECT_FALSE(linker.TryInstallOwnedLinks({{H(1, 2), H(2, 12)}}, &second));
    EXPECT_TRUE(second.empty());
    EXPECT_EQ(linker.TryValidateOwnedLinks(owned), -4);
    EXPECT_EQ(linker.TryRemoveOwnedLinks(owned), -4);
  }
  release.set_value();
  EXPECT_EQ(held.get(), 1);
  EXPECT_EQ(linker.TryValidateOwnedLinks(owned), 1);
  EXPECT_FALSE(linker.IsSource(H(1, 2)));
}

// ---------------------------------------------------------------------------
// CreateLink + GetTarget round trip
// ---------------------------------------------------------------------------
TEST(ElementLinkerTest, CreateLinkAndGetTarget) {
  ElementLinker linker;
  element_handle_t src = H(1, 100);
  element_handle_t tgt = H(2, 200);

  linker.CreateLink(src, tgt);

  auto result = linker.GetTarget(src, 2);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->space_id, 2);
  EXPECT_EQ(result->element_id, 200);
}

TEST(ElementLinkerTest, GraphRevisionOnlyChangesForSuccessfulGraphMutation) {
  ElementLinker linker;
  const uint64_t initial = linker.GetGraphRevision();
  linker.CreateLink(H(1, 10), H(2, 20));
  const uint64_t created = linker.GetGraphRevision();
  EXPECT_GT(created, initial);
  // CreateLink is an ownership replacement even when its endpoints match.
  linker.CreateLink(H(1, 10), H(2, 20));
  const uint64_t replaced = linker.GetGraphRevision();
  EXPECT_GT(replaced, created);
  // Exact batch install and missing removal are graph no-ops.
  std::vector<ElementLinker::OwnedLink> installed;
  EXPECT_TRUE(linker.InstallLinksIfUnconflicted(
      {{H(1, 10), H(2, 20)}}, &installed));
  EXPECT_TRUE(installed.empty());
  EXPECT_EQ(linker.GetGraphRevision(), replaced);
  EXPECT_FALSE(linker.RemoveLink(H(1, 10), H(2, 21)));
  EXPECT_EQ(linker.GetGraphRevision(), replaced);
  EXPECT_TRUE(linker.RemoveLink(H(1, 10), H(2, 20)));
  const uint64_t removed = linker.GetGraphRevision();
  EXPECT_GT(removed, created);
  linker.RemoveAllLinksForSpace(1);
  EXPECT_EQ(linker.GetGraphRevision(), removed);
}

TEST(ElementLinkerTest, GetTargetMissingSource) {
  ElementLinker linker;
  EXPECT_FALSE(linker.GetTarget(H(1, 100), 2).has_value());
}

TEST(ElementLinkerTest, GetTargetMissingSpace) {
  ElementLinker linker;
  linker.CreateLink(H(1, 100), H(2, 200));
  EXPECT_FALSE(linker.GetTarget(H(1, 100), 3).has_value());
}

TEST(ElementLinkerTest, BatchRejectsLaterConflictWithoutEarlierMutation) {
  ElementLinker linker;
  std::vector<ElementLinker::OwnedLink> installed;
  EXPECT_FALSE(linker.InstallLinksIfUnconflicted(
      {{H(1, 10), H(2, 20)}, {H(1, 10), H(2, 21)}}, &installed));
  EXPECT_FALSE(linker.GetTarget(H(1, 10), 2).has_value());
  EXPECT_TRUE(installed.empty());
}

TEST(ElementLinkerTest, OwnedRemovalDoesNotDeleteRecreatedExactPair) {
  ElementLinker linker;
  std::vector<ElementLinker::OwnedLink> installed;
  ASSERT_TRUE(
      linker.InstallLinksIfUnconflicted({{H(1, 10), H(2, 20)}}, &installed));
  ASSERT_EQ(installed.size(), 1u);

  ASSERT_TRUE(linker.RemoveLink(H(1, 10), H(2, 20)));
  linker.CreateLink(H(1, 10), H(2, 20));
  EXPECT_FALSE(linker.RemoveOwnedLink(installed[0]));
  EXPECT_TRUE(linker.IsLinked(H(1, 10), H(2, 20)));
}

TEST(ElementLinkerTest, OwnedRemovalDoesNotDeleteSameEndpointReplacement) {
  ElementLinker linker;
  std::vector<ElementLinker::OwnedLink> installed;
  ASSERT_TRUE(
      linker.InstallLinksIfUnconflicted({{H(1, 10), H(2, 20)}}, &installed));
  ASSERT_EQ(installed.size(), 1u);
  const uint64_t before_replace = linker.GetGraphRevision();
  linker.CreateLink(H(1, 10), H(2, 20));
  EXPECT_GT(linker.GetGraphRevision(), before_replace);
  EXPECT_FALSE(linker.RemoveOwnedLink(installed[0]));
  EXPECT_TRUE(linker.IsLinked(H(1, 10), H(2, 20)));
}

TEST(ElementLinkerTest, MultipleSourcesMayShareOneTarget) {
  ElementLinker linker;
  std::vector<ElementLinker::OwnedLink> installed;
  EXPECT_TRUE(linker.InstallLinksIfUnconflicted(
      {{H(1, 10), H(2, 20)}, {H(1, 11), H(2, 20)}}, &installed));
  EXPECT_EQ(installed.size(), 2u);
  EXPECT_TRUE(linker.IsLinked(H(1, 10), H(2, 20)));
  EXPECT_TRUE(linker.IsLinked(H(1, 11), H(2, 20)));
}

// ---------------------------------------------------------------------------
// Translate
// ---------------------------------------------------------------------------
TEST(ElementLinkerTest, TranslateSameSpace) {
  ElementLinker linker;
  // No link needed — same-space is a no-op.
  EXPECT_EQ(linker.Translate(H(1, 42), 1), 42);
}

TEST(ElementLinkerTest, TranslateSourceToTarget) {
  ElementLinker linker;
  linker.CreateLink(H(1, 10), H(2, 20));
  EXPECT_EQ(linker.Translate(H(1, 10), 2), 20);
}

TEST(ElementLinkerTest, TranslateTargetToSource) {
  ElementLinker linker;
  linker.CreateLink(H(1, 10), H(2, 20));
  // target -> source space
  EXPECT_EQ(linker.Translate(H(2, 20), 1), 10);
}

TEST(ElementLinkerTest, TranslateTargetToOtherTarget) {
  // A -> B, A -> C; translate B to C via A.
  ElementLinker linker;
  linker.CreateLink(H(1, 10), H(2, 20));
  linker.CreateLink(H(1, 10), H(3, 30));
  EXPECT_EQ(linker.Translate(H(2, 20), 3), 30);
}

TEST(ElementLinkerTest, TranslateUnlinkedReturnsInvalid) {
  ElementLinker linker;
  EXPECT_EQ(linker.Translate(H(1, 99), 2), kInvalidElementId);
}

// ---------------------------------------------------------------------------
// One source linked to targets in two spaces; GetTargets returns both
// ---------------------------------------------------------------------------
TEST(ElementLinkerTest, GetTargetsMultipleSpaces) {
  ElementLinker linker;
  element_handle_t src = H(1, 1);
  linker.CreateLink(src, H(2, 200));
  linker.CreateLink(src, H(3, 300));

  auto targets = linker.GetTargets(src);
  ASSERT_EQ(targets.size(), 2u);
  // Order follows space_id (map is sorted).
  EXPECT_EQ(targets[0], (element_handle_t{2, 200}));
  EXPECT_EQ(targets[1], (element_handle_t{3, 300}));
}

TEST(ElementLinkerTest, GetTargetsUnknownSource) {
  ElementLinker linker;
  EXPECT_TRUE(linker.GetTargets(H(9, 9)).empty());
}

// ---------------------------------------------------------------------------
// RemoveSource
// ---------------------------------------------------------------------------
TEST(ElementLinkerTest, RemoveSource) {
  ElementLinker linker;
  linker.CreateLink(H(1, 1), H(2, 2));
  EXPECT_TRUE(linker.RemoveSource(H(1, 1)));
  EXPECT_FALSE(linker.IsSource(H(1, 1)));
  EXPECT_FALSE(linker.GetTarget(H(1, 1), 2).has_value());
}

TEST(ElementLinkerTest, RemoveSourceNotFound) {
  ElementLinker linker;
  EXPECT_FALSE(linker.RemoveSource(H(1, 1)));
}

// ---------------------------------------------------------------------------
// RemoveTarget
// ---------------------------------------------------------------------------
TEST(ElementLinkerTest, RemoveTarget) {
  ElementLinker linker;
  linker.CreateLink(H(1, 1), H(2, 2));
  EXPECT_TRUE(linker.RemoveTarget(H(2, 2)));
  EXPECT_FALSE(linker.IsTarget(H(2, 2)));
  EXPECT_FALSE(linker.GetTarget(H(1, 1), 2).has_value());
  // Source entry still exists (just empty).
  EXPECT_TRUE(linker.IsSource(H(1, 1)));
}

TEST(ElementLinkerTest, RemoveTargetNotFound) {
  ElementLinker linker;
  EXPECT_FALSE(linker.RemoveTarget(H(2, 99)));
}

// ---------------------------------------------------------------------------
// RemoveAllLinksForSpace
// ---------------------------------------------------------------------------
TEST(ElementLinkerTest, RemoveAllLinksForSpaceRemovesSources) {
  ElementLinker linker;
  // Sources in space 1
  linker.CreateLink(H(1, 10), H(2, 20));
  linker.CreateLink(H(1, 11), H(3, 30));
  // Source NOT in space 1
  linker.CreateLink(H(2, 50), H(3, 60));

  linker.RemoveAllLinksForSpace(1);

  EXPECT_FALSE(linker.IsSource(H(1, 10)));
  EXPECT_FALSE(linker.IsSource(H(1, 11)));
  // Unrelated link survives.
  EXPECT_TRUE(linker.IsSource(H(2, 50)));
}

TEST(ElementLinkerTest, RemoveAllLinksForSpaceRemovesTargets) {
  ElementLinker linker;
  // Source in space 1, target in space 2
  linker.CreateLink(H(1, 10), H(2, 20));
  // Source in space 3, target also in space 2
  linker.CreateLink(H(3, 30), H(2, 40));
  // Source in space 3, target in space 4 (unaffected)
  linker.CreateLink(H(3, 30), H(4, 50));

  linker.RemoveAllLinksForSpace(2);

  // All sources in space 2 removed (none here).
  // All targets in space 2 removed from remaining sources.
  EXPECT_FALSE(linker.GetTarget(H(3, 30), 2).has_value());
  // Target in space 4 untouched.
  EXPECT_TRUE(linker.GetTarget(H(3, 30), 4).has_value());
}

TEST(ElementLinkerTest, RemoveAllLinksForSpaceBothDirections) {
  ElementLinker linker;
  // Space A sources
  linker.CreateLink(H(10, 1), H(20, 2));
  // Space A as target
  linker.CreateLink(H(20, 3), H(10, 4));
  // Unrelated
  linker.CreateLink(H(30, 5), H(40, 6));

  linker.RemoveAllLinksForSpace(10);

  EXPECT_FALSE(linker.IsSource(H(10, 1)));
  EXPECT_FALSE(linker.GetTarget(H(20, 3), 10).has_value());
  // Unrelated link survives.
  EXPECT_TRUE(linker.IsSource(H(30, 5)));
}

// ---------------------------------------------------------------------------
// IsSource / IsTarget / IsLinked
// ---------------------------------------------------------------------------
TEST(ElementLinkerTest, IsSourceAndIsTarget) {
  ElementLinker linker;
  linker.CreateLink(H(1, 1), H(2, 2));
  EXPECT_TRUE(linker.IsSource(H(1, 1)));
  EXPECT_FALSE(linker.IsSource(H(2, 2)));
  EXPECT_TRUE(linker.IsTarget(H(2, 2)));
  EXPECT_FALSE(linker.IsTarget(H(1, 1)));
}

TEST(ElementLinkerTest, IsLinkedSourceTarget) {
  ElementLinker linker;
  linker.CreateLink(H(1, 1), H(2, 2));
  EXPECT_TRUE(linker.IsLinked(H(1, 1), H(2, 2)));
  EXPECT_FALSE(linker.IsLinked(H(1, 1), H(2, 99)));
  EXPECT_FALSE(linker.IsLinked(H(9, 9), H(2, 2)));
}

TEST(ElementLinkerTest, IsLinkedSourceOnly) {
  ElementLinker linker;
  EXPECT_FALSE(linker.IsLinked(H(1, 1)));
  linker.CreateLink(H(1, 1), H(2, 2));
  EXPECT_TRUE(linker.IsLinked(H(1, 1)));
  linker.RemoveTarget(H(2, 2));
  // Source still exists but has no targets.
  EXPECT_FALSE(linker.IsLinked(H(1, 1)));
}

// ---------------------------------------------------------------------------
// GetSource
// ---------------------------------------------------------------------------
TEST(ElementLinkerTest, GetSource) {
  ElementLinker linker;
  linker.CreateLink(H(1, 10), H(2, 20));
  auto src = linker.GetSource(H(2, 20));
  ASSERT_TRUE(src.has_value());
  EXPECT_EQ(*src, (element_handle_t{1, 10}));
}

TEST(ElementLinkerTest, GetSourceNotFound) {
  ElementLinker linker;
  EXPECT_FALSE(linker.GetSource(H(2, 20)).has_value());
}

// ---------------------------------------------------------------------------
// GetSourcesFromSpace / GetTargetsFromSpace
// ---------------------------------------------------------------------------
TEST(ElementLinkerTest, GetSourcesFromSpace) {
  ElementLinker linker;
  linker.CreateLink(H(1, 10), H(2, 20));
  linker.CreateLink(H(1, 11), H(3, 30));
  linker.CreateLink(H(2, 50), H(3, 60));

  auto sources = linker.GetSourcesFromSpace(1);
  ASSERT_EQ(sources.size(), 2u);
}

TEST(ElementLinkerTest, GetTargetsFromSpace) {
  ElementLinker linker;
  linker.CreateLink(H(1, 10), H(2, 20));
  linker.CreateLink(H(3, 30), H(2, 40));
  linker.CreateLink(H(5, 50), H(6, 60));

  auto targets = linker.GetTargetsFromSpace(2);
  ASSERT_EQ(targets.size(), 2u);
}

// ---------------------------------------------------------------------------
// CreateLink overwrites existing target for same space
// ---------------------------------------------------------------------------
TEST(ElementLinkerTest, CreateLinkOverwritesExistingTargetInSameSpace) {
  ElementLinker linker;
  linker.CreateLink(H(1, 1), H(2, 100));
  linker.CreateLink(H(1, 1), H(2, 200));  // overwrite

  auto result = linker.GetTarget(H(1, 1), 2);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->element_id, 200);
}

}  // namespace test
}  // namespace spatialblend
