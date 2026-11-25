// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/relation_phase_tracker.h"

#include <gtest/gtest.h>


namespace spatialblend {
namespace {

RelationEvent Id() {
  RelationEvent e;
  e.actor_shared_id = 7;
  e.anchor_name = "hand_r";
  e.referent_shared_id = 42;
  e.kernel_type = RelationKernelType::kAvatarAffordancePose;
  return e;
}

// There is no activation gate (no R < 0.10 / 0.05 dormant gate): a relation in scope is
// ACTIVE from the moment it is observed, whatever its R.
TEST(RelationPhaseTrackerTest, DormantToActiveToDrivingToContact) {
  RelationPhaseTracker t;
  auto ev = t.Update(Id(), 0.05, 0.0, 0.0);  // low R: observed = in scope = active
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0].from, RelationPhase::kDormant);
  EXPECT_EQ(ev[0].to, RelationPhase::kActive);
  ev = t.Update(Id(), 0.5, 0.7, 0.2);  // weight above driving threshold
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0].to, RelationPhase::kDriving);
  ev = t.Update(Id(), 0.97, 0.7, 0.3);
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0].to, RelationPhase::kContact);
  EXPECT_EQ(t.GetPhase(Id()), RelationPhase::kContact);
}

// There is no R floor/release pair, so a falling R does not release the relation. It leaves ACTIVE only by leaving scope (the caller clears it).
TEST(RelationPhaseTrackerTest, LowRDoesNotRelease) {
  RelationPhaseTracker t;
  t.Update(Id(), 0.2, 0.0, 0.0);                        // -> kActive
  EXPECT_TRUE(t.Update(Id(), 0.08, 0.0, 0.1).empty());
  EXPECT_TRUE(t.Update(Id(), 0.0, 0.0, 0.2).empty());   // R = 0: still in scope
  EXPECT_EQ(t.GetPhase(Id()), RelationPhase::kActive);
  t.ClearRelationForOwner(Id(), "shared:7");            // leaving scope
  EXPECT_EQ(t.GetPhase(Id()), RelationPhase::kDormant);
}

// DRIVING / CONTACT hysteresis is kept (weight 0.50/0.40, R 0.95/0.85).
TEST(RelationPhaseTrackerTest, DrivingAndContactHysteresis) {
  RelationPhaseTracker t;
  t.Update(Id(), 0.5, 0.0, 0.0);
  t.Update(Id(), 0.5, 0.6, 0.1);  // -> kDriving
  EXPECT_TRUE(t.Update(Id(), 0.5, 0.45, 0.2).empty());  // 0.40 < w < 0.50 holds
  auto ev = t.Update(Id(), 0.5, 0.3, 0.3);
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0].to, RelationPhase::kActive);
  t.Update(Id(), 0.99, 0.0, 0.4);  // -> kContact
  EXPECT_TRUE(t.Update(Id(), 0.9, 0.0, 0.5).empty());  // 0.85 < R < 0.95 holds
  ev = t.Update(Id(), 0.8, 0.0, 0.6);
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0].to, RelationPhase::kDriving);
}

TEST(RelationPhaseTrackerTest, ContactDirectFromActive) {
  RelationPhaseTracker t;
  t.Update(Id(), 0.2, 0.0, 0.0);             // -> kActive
  auto ev = t.Update(Id(), 0.99, 0.0, 0.1);  // grounded, low weight
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0].to, RelationPhase::kContact);
}

TEST(RelationPhaseTrackerTest, IndependentKeysTrackedSeparately) {
  RelationPhaseTracker t;
  RelationEvent a = Id();
  RelationEvent b = Id();
  b.referent_shared_id = 99;
  t.Update(a, 0.2, 0.0, 0.0);  // a -> kActive
  EXPECT_EQ(t.GetPhase(a), RelationPhase::kActive);
  EXPECT_EQ(t.GetPhase(b), RelationPhase::kDormant);
}

TEST(RelationPhaseTrackerTest, ClearActorRetainsOtherActorPhase) {
  RelationPhaseTracker t;
  RelationEvent a = Id();
  RelationEvent b = Id();
  b.actor_shared_id = 8;
  t.Update(a, 0.2, 0.0, 0.0);
  t.Update(b, 0.2, 0.0, 0.0);

  t.ClearActor(a.actor_shared_id);

  EXPECT_EQ(t.GetPhase(a), RelationPhase::kDormant);
  EXPECT_EQ(t.GetPhase(b), RelationPhase::kActive);
}

TEST(RelationPhaseTrackerTest, PrivateOwnersDisambiguateEqualPublicActorIds) {
  RelationPhaseTracker t;
  RelationEvent same_public_id = Id();
  same_public_id.actor_shared_id = kInvalidSharedElementId;
  t.UpdateForOwner(same_public_id, "observation:source:1", 0.2, 0.0, 0.0);
  t.UpdateForOwner(same_public_id, "observation:source:2", 0.2, 0.0, 0.0);

  t.ClearOwner("observation:source:1");

  EXPECT_EQ(t.GetPhaseForOwner(same_public_id, "observation:source:1"),
            RelationPhase::kDormant);
  EXPECT_EQ(t.GetPhaseForOwner(same_public_id, "observation:source:2"),
            RelationPhase::kActive);
}

TEST(RelationPhaseTrackerTest, ClearOneRelationPreservesOtherReferentsAndOwners) {
  RelationPhaseTracker tracker;
  auto one = Id();
  auto other = Id();
  other.referent_shared_id = 99;
  tracker.UpdateForOwner(one, "source:1", .99, 0, 0);
  tracker.UpdateForOwner(other, "source:1", .99, 0, 0);
  tracker.UpdateForOwner(one, "source:2", .99, 0, 0);
  tracker.UpdateForOwner(one, "source:1", .99, 0, .1);
  tracker.UpdateForOwner(other, "source:1", .99, 0, .1);
  tracker.UpdateForOwner(one, "source:2", .99, 0, .1);
  tracker.ClearRelationForOwner(one, "source:1");
  EXPECT_EQ(tracker.GetPhaseForOwner(one, "source:1"), RelationPhase::kDormant);
  EXPECT_EQ(tracker.GetPhaseForOwner(other, "source:1"), RelationPhase::kContact);
  EXPECT_EQ(tracker.GetPhaseForOwner(one, "source:2"), RelationPhase::kContact);
  const auto reentry = tracker.UpdateForOwner(one, "source:1", .90, 0, 1);
  ASSERT_EQ(reentry.size(), 1u);
  EXPECT_EQ(reentry[0].from, RelationPhase::kDormant);
  EXPECT_EQ(reentry[0].to, RelationPhase::kActive);
}

// ---------------------------------------------------------------------------
// Retired contracts: the AffordancePose 1.5 / 2.0 m near
// field with its 0.3 s release dwell, the per-family activation thresholds,
// Facing's activation configurability, and both scope radii. Scope is the
// family's (D1-D9, see scope_gate_test / scope_gating_contract_test); nothing in the
// phase configuration narrows it any more.
TEST(RelationPhaseTrackerTest, ConfigHasNoScopeOrActivationFields) {
  // The same defaults for every family: only the lifecycle labels remain.
  for (auto family : {RelationKernelType::kAvatarFacing,
                      RelationKernelType::kAvatarSpacing,
                      RelationKernelType::kAvatarAffordancePose}) {
    const auto c = DefaultRelationPhaseConfig(family);
    EXPECT_DOUBLE_EQ(c.driving_weight, 0.50);
    EXPECT_DOUBLE_EQ(c.driving_release, 0.40);
    EXPECT_DOUBLE_EQ(c.contact_proximity, 0.95);
    EXPECT_DOUBLE_EQ(c.contact_release, 0.85);
  }
}

// An AffordancePose relation 9 m away is in scope exactly as one at 0.1 m is (D3: scope
// is the parent object's region, not a distance), so it is ACTIVE at once.
TEST(RelationPhaseTrackerTest, AffordancePoseFarIsStillActive) {
  RelationPhaseTracker t;
  auto ev = t.Update(Id(), 1.0 / 10.0, 0.0, 0.0);
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0].to, RelationPhase::kActive);
}

// Thresholds are per family, settable and observable.
TEST(RelationPhaseTrackerTest, Thresholds_E1_6_PerFamilySetAndGet) {
  RelationPhaseTracker t;
  RelationPhaseConfig strict;
  strict.driving_weight = 0.9;
  strict.driving_release = 0.8;
  ASSERT_TRUE(t.SetConfig(RelationKernelType::kAvatarFacing, strict).ok());
  EXPECT_DOUBLE_EQ(t.GetConfig(RelationKernelType::kAvatarFacing).driving_weight,
                   0.9);
  RelationEvent facing = Id();
  facing.kernel_type = RelationKernelType::kAvatarFacing;
  t.Update(facing, 0.5, 0.0, 0.0);
  EXPECT_TRUE(t.Update(facing, 0.5, 0.7, 0.1).empty());  // below the new driving weight
  auto ev = t.Update(facing, 0.5, 0.95, 0.2);
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0].to, RelationPhase::kDriving);
  // Other families are untouched.
  EXPECT_DOUBLE_EQ(
      t.GetConfig(RelationKernelType::kAvatarSpacing).driving_weight, 0.5);
}

// Facing: a visible referent directly behind the actor (R = 0) stays active.
TEST(RelationPhaseTrackerTest, Facing_NoAngularCutoff) {
  RelationPhaseTracker t;
  RelationEvent facing = Id();
  facing.kernel_type = RelationKernelType::kAvatarFacing;
  auto ev = t.Update(facing, 0.0, 0.0, 0.0);
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0].to, RelationPhase::kActive);
  EXPECT_TRUE(t.Update(facing, 0.0, 0.0, 0.1).empty());
}

TEST(RelationPhaseTrackerTest, Thresholds_E1_6_InvalidConfigRejected) {
  RelationPhaseTracker t;
  RelationPhaseConfig bad;
  bad.driving_weight = 0.4;
  bad.driving_release = 0.5;  // release above enter: no hysteresis
  EXPECT_FALSE(t.SetConfig(RelationKernelType::kAvatarSpacing, bad).ok());
  RelationPhaseConfig bad_contact;
  bad_contact.contact_proximity = 0.8;
  bad_contact.contact_release = 0.9;
  EXPECT_FALSE(t.SetConfig(RelationKernelType::kAvatarSpacing, bad_contact).ok());
  // The previous (default) config is kept.
  EXPECT_DOUBLE_EQ(t.GetConfig(RelationKernelType::kAvatarSpacing).driving_release,
                   0.40);
}

}  // namespace
}  // namespace spatialblend
