// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/update_actor_state_characterization_test.cc
//
// Characterization of Space::UpdateActorState's observable behavior.
//
// This test asserts nothing about what the behavior *should* be. It pins what
// it currently *is*, so that the staged prepared-patch work — which must extract
// shared helpers out of this entry point without changing the live path — has a
// mechanical pass/fail gate instead of a reviewer's judgement.
//
// Scope: only what the public Space API exposes. Trace emission goes through a
// global TraceSink and is deliberately NOT characterized here; the ordering of
// a send against its own trace record cannot be observed from outside Space, so
// that particular invariant still needs its own in-tree check when the effect
// boundary is built.
//
// Determinism: SetMovementGate and SetWeightSmoothing are called explicitly in
// SetUp. Without them the deadbands come from SB_V2_MIN_MOVE / SB_V2_MIN_YAW,
// resolved when the Space is constructed, and this test would depend on the
// caller's environment.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <future>
#include <string>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/space/binding.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/relation_event.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace test {
namespace {

constexpr double kUpdateInterval = 0.1;
constexpr double kMinMove = 0.05;
constexpr double kMinYaw = 0.05;

// One log for both effect kinds, so their relative order is observable. The
// contract requires callbacks, traces and sends to stay in emission order
// rather than being regrouped by kind; this is the part of that order the
// public API can see.
enum class EffectKind { kRelationEvent, kPropagate };

struct Effect {
  EffectKind kind;
  // Relation event fields.
  shared_element_id_t referent_shared_id = kInvalidSharedElementId;
  RelationPhase from = RelationPhase::kDormant;
  RelationPhase to = RelationPhase::kDormant;
  // Propagation fields.
  int64_t sequence = 0;
  size_t payload_count = 0;
};

class UpdateActorStateCharacterization : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = LocalContext::Create();
    space_ = context_->AddSpace();
    space_->SetUpdateInterval(kUpdateInterval);
    space_->SetMovementGate(kMinMove, kMinYaw);
    space_->SetWeightSmoothing(1.0);  // no EMA history, so frames are independent

    auto* tree = space_->GetElementTree();
    // Room: a 10x10 navigable area.
    ASSERT_TRUE(tree->AddElement(kRoom).ok());
    tree->GetElement(kRoom)->SetSharedId(kRoom);
    tree->GetElement(kRoom)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(
            std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
                {0, 0}, {10, 0}, {10, 10}, {0, 10}})));
    // Two referents the actor can relate to.
    AddReferent(kSeat, {{1, 1}, {2, 1}, {2, 2}, {1, 2}});
    AddReferent(kTable, {{7, 7}, {9, 7}, {9, 9}, {7, 9}});

    actor_ = space_->CreateActor({SubjectAnchor("root")});
    space_->GetActor(actor_)->SetElementId(kRoom);
    // The measured application profile: Spacing on root, Facing on root. The
    // build scene uses a separate chest anchor for Facing; a single anchor is
    // enough to pin ordering and gate behavior.
    space_->Bind(actor_, "root", RelationKernelType::kAvatarSpacing);
    space_->Bind(actor_, "root", RelationKernelType::kAvatarFacing);

    space_->RegisterOnRelationEvent([this](const RelationEvent& event) {
      log_.push_back(Effect{EffectKind::kRelationEvent, event.referent_shared_id,
                            event.from, event.to, 0, 0});
    });
    context_->GetHub()->SetRelationUpdateCallback(
        [this](const RemoteRelationUpdate& update) {
          Effect effect{EffectKind::kPropagate};
          effect.sequence = update.sequence;
          effect.payload_count = update.payloads.size();
          log_.push_back(effect);
        });
  }

  void AddReferent(element_id_t id, std::vector<Eigen::Vector2d> points) {
    auto* tree = space_->GetElementTree();
    ASSERT_TRUE(tree->AddElement(id, kRoom).ok());
    tree->GetElement(id)->SetSharedId(id);
    tree->GetElement(id)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(
            std::make_shared<Polygon2D>(std::move(points))));
  }

  void Step(double x, double y, double yaw = 0.0) {
    space_->UpdateActorState(actor_, Eigen::Vector2d(x, y), yaw);
  }

  size_t CountKind(EffectKind kind) const {
    size_t total = 0;
    for (const Effect& effect : log_) {
      if (effect.kind == kind) ++total;
    }
    return total;
  }

  std::vector<int64_t> Sequences() const {
    std::vector<int64_t> sequences;
    for (const Effect& effect : log_) {
      if (effect.kind == EffectKind::kPropagate) sequences.push_back(effect.sequence);
    }
    return sequences;
  }

  static constexpr element_id_t kRoom = 1;
  static constexpr element_id_t kSeat = 2;
  static constexpr element_id_t kTable = 3;

  std::unique_ptr<LocalContext> context_;
  Space* space_ = nullptr;
  spatialblend::entity_id_t actor_ = 0;
  std::vector<Effect> log_;
};

// Guard against a vacuously green suite: the ordering and gating assertions
// below are all conditional on effects actually being emitted, so prove first
// that this scene produces both kinds.
TEST_F(UpdateActorStateCharacterization, TheSceneEmitsBothEffectKinds) {
  // Walk toward the seat so relation phases actually transition.
  for (int call = 0; call <= 10; ++call) {
    const double a = call / 10.0;
    Step(5.0 + (1.5 - 5.0) * a, 5.0 + (1.5 - 5.0) * a);
  }
  EXPECT_GT(CountKind(EffectKind::kRelationEvent), 0u)
      << "no relation event was emitted; the ordering assertion would be vacuous";
  EXPECT_GT(CountKind(EffectKind::kPropagate), 0u)
      << "no propagation was emitted; the gating assertions would be vacuous";
}

TEST_F(UpdateActorStateCharacterization, ClockAdvancesOncePerCallByUpdateInterval) {
  EXPECT_DOUBLE_EQ(space_->GetSpaceClock(), 0.0);
  for (int call = 1; call <= 5; ++call) {
    Step(5.0, 5.0);
    EXPECT_DOUBLE_EQ(space_->GetSpaceClock(), call * kUpdateInterval);
  }
}

TEST_F(UpdateActorStateCharacterization, ClockAdvancesEvenWhenNothingPropagates) {
  Step(5.0, 5.0);
  const double after_first = space_->GetSpaceClock();
  log_.clear();
  // Same pose: the movement gate suppresses propagation.
  Step(5.0, 5.0);
  EXPECT_EQ(CountKind(EffectKind::kPropagate), 0u);
  EXPECT_GT(space_->GetSpaceClock(), after_first);
}

// The ordering invariant the staged path must preserve: within one call,
// relation events are emitted from the binding loop before the propagation.
TEST_F(UpdateActorStateCharacterization, RelationEventsPrecedeThePropagationInACall) {
  for (int call = 0; call <= 10; ++call) {
    const double a = call / 10.0;
    Step(5.0 + (1.5 - 5.0) * a, 5.0 + (1.5 - 5.0) * a);
  }
  ASSERT_GT(CountKind(EffectKind::kRelationEvent), 0u);
  ASSERT_GT(CountKind(EffectKind::kPropagate), 0u);
  // Within each call the binding loop's events precede that call's send, so
  // scanning the whole log, no event may follow the final propagation of an
  // earlier call. Check per call instead by replaying one call at a time.
  log_.clear();
  Step(1.4, 1.4);
  bool seen_propagate = false;
  for (const Effect& effect : log_) {
    if (effect.kind == EffectKind::kPropagate) {
      seen_propagate = true;
    } else {
      EXPECT_FALSE(seen_propagate)
          << "a relation event was emitted after the propagation; the binding "
             "loop's events must precede the send";
    }
  }
}

TEST_F(UpdateActorStateCharacterization, StationaryActorIsGatedAfterTheFirstPropagation) {
  Step(5.0, 5.0);
  const size_t first = CountKind(EffectKind::kPropagate);
  EXPECT_EQ(first, 1u) << "the first call establishes the propagated pose";
  log_.clear();
  for (int call = 0; call < 3; ++call) {
    Step(5.0, 5.0);
  }
  EXPECT_EQ(CountKind(EffectKind::kPropagate), 0u);
}

TEST_F(UpdateActorStateCharacterization, MovementBeyondTheDeadbandPropagates) {
  Step(5.0, 5.0);
  log_.clear();
  Step(5.0 + kMinMove / 10.0, 5.0);  // inside the deadband
  EXPECT_EQ(CountKind(EffectKind::kPropagate), 0u);
  Step(5.0 + kMinMove * 10.0, 5.0);  // clearly outside
  EXPECT_EQ(CountKind(EffectKind::kPropagate), 1u);
}

TEST_F(UpdateActorStateCharacterization, YawBeyondTheDeadbandPropagates) {
  Step(5.0, 5.0);
  log_.clear();
  Step(5.0, 5.0, kMinYaw / 10.0);
  EXPECT_EQ(CountKind(EffectKind::kPropagate), 0u);
  Step(5.0, 5.0, kMinYaw * 10.0);
  EXPECT_EQ(CountKind(EffectKind::kPropagate), 1u);
}

TEST_F(UpdateActorStateCharacterization, SequencesStrictlyIncreaseAndOnlySendsConsumeThem) {
  for (int call = 0; call < 12; ++call) {
    // Alternate a gated frame with a moving one, so skipped frames are mixed in.
    Step(call % 2 == 0 ? 2.0 : 8.0, 5.0);
  }
  const std::vector<int64_t> sequences = Sequences();
  ASSERT_GE(sequences.size(), 2u);
  for (size_t i = 1; i < sequences.size(); ++i) {
    EXPECT_GT(sequences[i], sequences[i - 1]) << "sequence must be monotonic";
  }
  // Only real sends consume a number, so the first is 1 and there are exactly
  // as many numbers as propagations.
  EXPECT_EQ(sequences.front(), 1);
  EXPECT_EQ(sequences.back(), static_cast<int64_t>(sequences.size()));
}

TEST_F(UpdateActorStateCharacterization, LatestMatchesAreRebuiltEachCall) {
  Step(5.0, 5.0);
  const std::vector<Binding>& bindings = space_->GetBindings(actor_);
  ASSERT_EQ(bindings.size(), 2u);
  std::vector<size_t> first;
  for (const Binding& binding : bindings) first.push_back(binding.latest_matches.size());
  for (size_t count : first) EXPECT_GT(count, 0u);

  Step(5.0, 5.0);
  const std::vector<Binding>& again = space_->GetBindings(actor_);
  for (size_t i = 0; i < again.size(); ++i) {
    // Cleared then repopulated: the count does not accumulate across calls.
    EXPECT_EQ(again[i].latest_matches.size(), first[i]);
  }
}

TEST_F(UpdateActorStateCharacterization, PayloadCountTracksTheMatchedReferents) {
  Step(5.0, 5.0);
  size_t matched = 0;
  for (const Binding& binding : space_->GetBindings(actor_)) {
    matched += binding.latest_matches.size();
  }
  size_t payloads = 0;
  for (const Effect& effect : log_) {
    if (effect.kind == EffectKind::kPropagate) payloads = effect.payload_count;
  }
  EXPECT_EQ(payloads, matched)
      << "payloads are the sum over bindings of their matched referents";
}

TEST_F(UpdateActorStateCharacterization, PreviousRelationsRetainMatchedElementsOnly) {
  Step(5.0, 5.0);
  for (const Binding& binding : space_->GetBindings(actor_)) {
    for (const Binding::MatchResult& match : binding.latest_matches) {
      EXPECT_EQ(binding.sent_relations.count(match.element_id), 1u);
    }
    // Nothing is retained for an element that was never matched.
    EXPECT_EQ(binding.sent_relations.count(9999), 0u);
  }
}

TEST_F(UpdateActorStateCharacterization, ReceivedUpdatesAreUntouchedBySourceUpdates) {
  for (int call = 0; call < 5; ++call) {
    Step(static_cast<double>(call), 5.0);
  }
  // UpdateActorState is the source path; it must not append receive history.
  EXPECT_TRUE(space_->GetReceivedUpdates().empty());
}

// Restores the previous value so one test cannot leak an override into the
// next; tests in a package share a process.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* previous = std::getenv(name)) {
      had_previous_ = true;
      previous_ = previous;
    }
    ::setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (had_previous_) {
      ::setenv(name_, previous_.c_str(), 1);
    } else {
      ::unsetenv(name_);
    }
  }

 private:
  const char* name_;
  bool had_previous_ = false;
  std::string previous_;
};

// The environment-derived movement gate resolves at construction, not on first
// use. Changing the variable after the Space exists must have no effect; if it
// still resolved lazily inside UpdateActorState, the tightened deadband below
// would let a 0.1 m step through.
TEST(SpaceBootstrapPolicy, MovementGateResolvesAtConstructionNotOnFirstUpdate) {
  auto context = LocalContext::Create();
  Space* space = nullptr;
  {
    ScopedEnv wide("SB_V2_MIN_MOVE", "0.5");
    space = context->AddSpace();
  }
  // Now in force for any lazy reader, and deliberately far tighter.
  ScopedEnv tight("SB_V2_MIN_MOVE", "0.000001");

  space->SetUpdateInterval(kUpdateInterval);
  space->SetWeightSmoothing(1.0);
  auto* tree = space->GetElementTree();
  ASSERT_TRUE(tree->AddElement(1).ok());
  tree->GetElement(1)->SetSharedId(1);
  tree->GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
              {0, 0}, {10, 0}, {10, 10}, {0, 10}})));
  ASSERT_TRUE(tree->AddElement(2, 1).ok());
  tree->GetElement(2)->SetSharedId(2);
  tree->GetElement(2)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
              {1, 1}, {2, 1}, {2, 2}, {1, 2}})));

  const spatialblend::entity_id_t actor = space->CreateActor({SubjectAnchor("root")});
  space->GetActor(actor)->SetElementId(1);
  space->Bind(actor, "root", RelationKernelType::kAvatarSpacing);

  int propagations = 0;
  context->GetHub()->SetRelationUpdateCallback(
      [&propagations](const RemoteRelationUpdate&) { ++propagations; });

  space->UpdateActorState(actor, Eigen::Vector2d(5.0, 5.0), 0.0);
  ASSERT_EQ(propagations, 1) << "the first call establishes the propagated pose";
  propagations = 0;

  // 0.1 m: well under the 0.5 m deadband in force at construction, and well
  // over the 0.000001 m one set afterwards.
  space->UpdateActorState(actor, Eigen::Vector2d(5.1, 5.0), 0.0);
  EXPECT_EQ(propagations, 0)
      << "the post-construction environment value was observed, so the gate is "
         "still being resolved lazily inside the update path";
}

// SpaceHub::Propagate dispatches synchronously, so calling it under
// space_mutex_ ran the receiving space's OnRemoteUpdate — taking that space's
// own mutex — inside ours, and two spaces propagating to each other could
// deadlock. The send now happens after the guard is released.
//
// Observing that from inside the dispatch: the hub's relation-update callback
// runs within Propagate, so if the source's mutex were still held, a second
// thread calling a mutex-taking method on the source would block until the
// update returned. It completes instead.
TEST(SpacePropagationGuard, SourceMutexIsNotHeldWhileTheHubDispatches) {
  auto context = LocalContext::Create();
  Space* space = context->AddSpace();
  space->SetUpdateInterval(kUpdateInterval);
  space->SetWeightSmoothing(1.0);

  auto* tree = space->GetElementTree();
  ASSERT_TRUE(tree->AddElement(1).ok());
  tree->GetElement(1)->SetSharedId(1);
  tree->GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
              {0, 0}, {10, 0}, {10, 10}, {0, 10}})));
  ASSERT_TRUE(tree->AddElement(2, 1).ok());
  tree->GetElement(2)->SetSharedId(2);
  tree->GetElement(2)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
              {1, 1}, {2, 1}, {2, 2}, {1, 2}})));

  const spatialblend::entity_id_t actor = space->CreateActor({SubjectAnchor("root")});
  space->GetActor(actor)->SetElementId(1);
  space->Bind(actor, "root", RelationKernelType::kAvatarSpacing);

  bool dispatched = false;
  bool reentered_without_blocking = false;
  // Declared out here on purpose. If the source mutex is still held the probe
  // cannot finish inside the callback, and an std::async future blocks in its
  // destructor — waiting for it there would hang the test instead of failing
  // it. Keeping it alive past UpdateActorState lets the lock drop first.
  std::future<void> probe;
  context->GetHub()->SetRelationUpdateCallback(
      [&](const RemoteRelationUpdate&) {
        dispatched = true;
        // SetMovementGate takes space_mutex_.
        probe = std::async(std::launch::async, [space] {
          space->SetMovementGate(kMinMove, kMinYaw);
        });
        reentered_without_blocking =
            probe.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
      });

  space->UpdateActorState(actor, Eigen::Vector2d(5.0, 5.0), 0.0);
  if (probe.valid()) probe.wait();
  ASSERT_TRUE(dispatched) << "no propagation occurred; the assertion would be vacuous";
  EXPECT_TRUE(reentered_without_blocking)
      << "the source still held space_mutex_ while the hub dispatched";
}

// A room with one referent and one actor bound to it, so the actor's first
// update propagates. Shared by the two-space tests below.
spatialblend::entity_id_t AddPropagatingActor(Space* space) {
  space->SetUpdateInterval(kUpdateInterval);
  space->SetWeightSmoothing(1.0);
  auto* tree = space->GetElementTree();
  EXPECT_TRUE(tree->AddElement(1).ok());
  tree->GetElement(1)->SetSharedId(1);
  tree->GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
              {0, 0}, {10, 0}, {10, 10}, {0, 10}})));
  EXPECT_TRUE(tree->AddElement(2, 1).ok());
  tree->GetElement(2)->SetSharedId(2);
  tree->GetElement(2)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(
          std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
              {1, 1}, {2, 1}, {2, 2}, {1, 2}})));
  const spatialblend::entity_id_t actor = space->CreateActor({SubjectAnchor("root")});
  space->GetActor(actor)->SetElementId(1);
  space->Bind(actor, "root", RelationKernelType::kAvatarSpacing);
  return actor;
}

// The deadlock itself, rather than a proxy for it: two spaces on one hub, each
// propagating into the other. While A's send is being dispatched, a second
// thread makes B propagate, which dispatches into A's OnRemoteUpdate and takes
// A's mutex. If A still held its mutex across the send, A would be waiting on
// B's send and B's send on A's mutex — the cycle. The wait is bounded so the
// unfixed code fails instead of hanging; the probe is kept alive past the
// source call so the lock drops and the thread can finish either way.
//
// `send_from_a` drives one of the source entry points that stages a send.
void ExpectMutualPropagationCompletes(
    const std::function<void(Space*, spatialblend::entity_id_t)>& send_from_a) {
  auto context = LocalContext::Create();
  Space* a = context->AddSpace();
  Space* b = context->AddSpace();
  const spatialblend::entity_id_t actor_a = AddPropagatingActor(a);
  const spatialblend::entity_id_t actor_b = AddPropagatingActor(b);
  const space_id_t a_id = a->GetSpaceId();
  const space_id_t b_id = b->GetSpaceId();

  std::atomic<bool> a_dispatched{false};
  std::atomic<bool> b_dispatched{false};
  bool b_completed_inside_a = false;
  std::future<void> probe;
  context->GetHub()->SetRelationUpdateCallback(
      [&](const RemoteRelationUpdate& update) {
        if (update.source_space_id == b_id) {
          b_dispatched = true;
          return;
        }
        if (update.source_space_id != a_id || a_dispatched.exchange(true)) return;
        probe = std::async(std::launch::async, [b, actor_b] {
          b->UpdateActorState(actor_b, Eigen::Vector2d(5.0, 5.0), 0.0);
        });
        b_completed_inside_a =
            probe.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
      });

  send_from_a(a, actor_a);
  if (probe.valid()) probe.wait();
  ASSERT_TRUE(a_dispatched) << "A never propagated; the assertion would be vacuous";
  ASSERT_TRUE(b_dispatched) << "B never propagated; the assertion would be vacuous";
  EXPECT_TRUE(b_completed_inside_a)
      << "B's propagation into A blocked on A's space_mutex_, held across A's "
         "own send: two spaces propagating to each other deadlock";
}

// Site 1: the source update path (EvaluateActorStateLocked).
TEST(SpacePropagationGuard, TwoSpacesPropagatingToEachOtherDoNotDeadlock) {
  ExpectMutualPropagationCompletes([](Space* a, spatialblend::entity_id_t actor) {
    a->UpdateActorState(actor, Eigen::Vector2d(5.0, 5.0), 0.0);
  });
}

// Site 2: the epoch-reset control envelope sent by BeginActorObservationEpoch,
// which propagated under its own guard independently of the update path.
TEST(SpacePropagationGuard, EpochResetAndPeerPropagationDoNotDeadlock) {
  ExpectMutualPropagationCompletes([](Space* a, spatialblend::entity_id_t actor) {
    uint64_t epoch = 0;
    EXPECT_TRUE(a->BeginActorObservationEpoch(actor, "root", &epoch));
  });
}

}  // namespace
}  // namespace test
}  // namespace spatialblend
