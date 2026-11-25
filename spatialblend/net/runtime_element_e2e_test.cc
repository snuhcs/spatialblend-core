// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/net/runtime_element_e2e_test.cc
//
// Parametrized end-to-end tests for runtime element lifecycle (ShareElement /
// UnshareElement) across spaces, covering both in-process (LocalContext) and
// gRPC networking (HubReceiver + RemoteHub) transports.

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/net/hub_receiver.h"
#include "spatialblend/net/remote_hub.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace test {

// ============================================================================
// Transport mode enum + helpers
// ============================================================================

enum class TransportMode { kLocal, kNetwork };

std::string TransportName(
    const ::testing::TestParamInfo<TransportMode>& info) {
  return info.param == TransportMode::kLocal ? "Local" : "Network";
}

// Helper: create a square polygon centered at origin.
std::shared_ptr<Polygon2D> MakeSquareRoom(double half_size) {
  return std::make_shared<Polygon2D>(std::vector<Eigen::Vector2d>{
      {-half_size, -half_size},
      {half_size, -half_size},
      {half_size, half_size},
      {-half_size, half_size}});
}

// ============================================================================
// RuntimeElementE2ETest — parametrized fixture
// ============================================================================

class RuntimeElementE2ETest
    : public ::testing::TestWithParam<TransportMode> {
 protected:
  // Shared IDs for the common hierarchy.
  static constexpr shared_element_id_t kRoomSharedId = 100;
  static constexpr element_id_t kRoomLocalId = 1;

  void SetUp() override {
    if (GetParam() == TransportMode::kLocal) {
      SetupLocal();
    } else {
      SetupNetwork();
    }
  }

  void TearDown() override {
    if (GetParam() == TransportMode::kNetwork) {
      TeardownNetwork();
    }
  }

  // ------ Local transport ------

  void SetupLocal() {
    context_ = LocalContext::Create();
    space_a_ = context_->AddSpace();
    space_b_ = context_->AddSpace();

    SetupRoomOnSpace(space_a_);
    SetupRoomOnSpace(space_b_);
  }

  // ------ Network transport ------

  void SetupNetwork() {
    // 1. Create host space with Room.
    host_space_ = std::make_unique<Space>();
    host_space_->SetSpaceId(1);
    SetupRoomOnSpace(host_space_.get());

    // Set up host-side hub.
    hub_ = std::make_unique<SpaceHub>();
    ASSERT_TRUE(hub_->Connect(host_space_.get()).ok());
    host_space_->SetHub(hub_.get());

    // 2. Start HubReceiver on auto-assigned port.
    hub_receiver_ =
        std::make_shared<HubReceiver>(host_space_.get(), hub_.get());
    auto status = hub_receiver_->Start(0);
    ASSERT_TRUE(status.ok()) << status.ToString();
    int port = hub_receiver_->GetConnectedPort();
    ASSERT_GT(port, 0);

    // Wire element event and relation update callbacks so that
    // Space::ShareElement / UpdateActorState propagation reaches
    // remote clients through HubReceiver.
    hub_->SetElementEventCallback(
        [this](const ElementEvent& event, space_id_t source_space_id) {
          hub_receiver_->PropagateElementEventToRemoteSpaces(
              event, source_space_id);
        });
    hub_->SetRelationUpdateCallback(
        [this](const RemoteRelationUpdate& update) {
          hub_receiver_->PropagateToRemoteSpaces(update);
        });

    // 3. Create client space (empty — bootstrap will replicate hierarchy).
    client_space_ = std::make_unique<Space>();
    client_space_->SetSpaceId(2);

    // 4. Connect RemoteHub (performs bootstrap + opens Subscribe stream).
    remote_hub_ = RemoteHub::Create(client_space_.get(), "localhost", port,
                                     client_space_->GetSpaceId());
    ASSERT_NE(remote_hub_, nullptr) << "RemoteHub::Create failed";
    ASSERT_TRUE(remote_hub_->IsConnected());

    // 5. Wait for Subscribe stream to be fully established.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Set spatial definition on the bootstrapped Room element so that
    // relation evaluation and element parenting work on the client side.
    Element* client_room =
        client_space_->GetElementTree()->GetElementBySharedId(kRoomSharedId);
    ASSERT_NE(client_room, nullptr)
        << "Room should have been bootstrapped to client";
    client_room->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(MakeSquareRoom(10.0)));

    space_a_ = host_space_.get();
    space_b_ = client_space_.get();
  }

  void TeardownNetwork() {
    if (remote_hub_) {
      (void)remote_hub_->PrepareDisconnect();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (hub_receiver_) {
      hub_receiver_->Stop();
    }
    remote_hub_.reset();
    hub_receiver_.reset();
  }

  // ------ Common helpers ------

  void SetupRoomOnSpace(Space* space) {
    auto* tree = space->GetElementTree();
    ASSERT_TRUE(tree->AddElement(kRoomLocalId).ok());
    tree->GetElement(kRoomLocalId)->SetSharedId(kRoomSharedId);
    auto room_poly = MakeSquareRoom(10.0);
    tree->GetElement(kRoomLocalId)->SetSpatialDefinition(
        std::make_shared<NavigableSpatialDefinition>(room_poly));
  }

  // Add an object element to space_a and return its local id.
  element_id_t AddObjectToSpaceA(shared_element_id_t shared_id,
                                 double x, double y,
                                 const std::string& tag = "") {
    auto* tree = space_a_->GetElementTree();
    element_id_t local_id =
        static_cast<element_id_t>(tree->Size() + 1);
    EXPECT_TRUE(
        tree->AddElement(local_id, kRoomLocalId).ok());
    tree->GetElement(local_id)->SetSharedId(shared_id);
    if (!tag.empty()) {
      tree->GetElement(local_id)->SetTag(tag);
    }
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 3) = x;
    t(1, 3) = y;
    tree->GetElement(local_id)->SetLocalTransform(t);
    return local_id;
  }

  // Create an actor in the given space, bound to the room with spacing.
  spatialblend::entity_id_t CreateBoundActor(Space* space) {
    auto actor_id =
        space->CreateActor({SubjectAnchor("root")});
    space->GetActor(actor_id)->SetElementId(kRoomLocalId);
    space->Bind(actor_id, "root", RelationKernelType::kAvatarSpacing);
    return actor_id;
  }

  // Wait for propagation (network needs longer; local is synchronous).
  void WaitForPropagation() {
    if (GetParam() == TransportMode::kNetwork) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    // LocalContext propagation is synchronous — no wait needed.
  }

  // ------ Members ------
  // Local mode
  std::unique_ptr<LocalContext> context_;

  // Network mode
  std::unique_ptr<Space> host_space_;
  std::unique_ptr<Space> client_space_;
  std::unique_ptr<SpaceHub> hub_;
  std::shared_ptr<HubReceiver> hub_receiver_;
  std::shared_ptr<RemoteHub> remote_hub_;

  // Common (set by either SetupLocal or SetupNetwork)
  Space* space_a_ = nullptr;
  Space* space_b_ = nullptr;
};

// ============================================================================
// Test 1: RuntimeAddElement_RelationEvaluatesNewElement
//
// After sharing a new Board element from space_a, space_b should see it
// and subsequent relation evaluation should include the Board in matches.
// ============================================================================

TEST_P(RuntimeElementE2ETest, RuntimeAddElement_RelationEvaluatesNewElement) {
  auto actor_id = CreateBoundActor(space_a_);

  // Add Board to space_a and share.
  element_id_t board_id = AddObjectToSpaceA(/*shared_id=*/200, 5.0, 5.0);
  space_a_->ShareElement(board_id);
  WaitForPropagation();

  // space_b should now have an element with shared_id 200.
  const Element* board_b =
      space_b_->GetElementTree()->GetElementBySharedId(200);
  ASSERT_NE(board_b, nullptr)
      << "Board should have been propagated to space_b";
  EXPECT_EQ(board_b->GetSharedId(), 200);

  // Update actor state so relation solver evaluates the new Board.
  space_a_->UpdateActorState(actor_id, {3.0, 3.0}, 0.0);

  // Verify: spacing matches should include the Board.
  const auto& bindings = space_a_->GetBindings(actor_id);
  ASSERT_FALSE(bindings.empty());
  bool found_board = false;
  for (const auto& match : bindings[0].latest_matches) {
    if (match.element_id == board_id) {
      found_board = true;
    }
  }
  EXPECT_TRUE(found_board) << "Board should appear in spacing matches";
}

// ============================================================================
// Test 2: RuntimeRemoveElement_RelationStopsEvaluating
//
// After unsharing the Board from space_a, space_b should no longer have it,
// and relation evaluation should stop including the Board.
// ============================================================================

TEST_P(RuntimeElementE2ETest, RuntimeRemoveElement_RelationStopsEvaluating) {
  auto actor_id = CreateBoundActor(space_a_);

  // Add Board, share, and verify it's in space_b.
  element_id_t board_id = AddObjectToSpaceA(/*shared_id=*/200, 5.0, 5.0);
  space_a_->ShareElement(board_id);
  WaitForPropagation();

  ASSERT_NE(space_b_->GetElementTree()->GetElementBySharedId(200), nullptr);

  // Update actor to trigger relation evaluation with Board present.
  space_a_->UpdateActorState(actor_id, {3.0, 3.0}, 0.0);
  {
    const auto& bindings = space_a_->GetBindings(actor_id);
    ASSERT_FALSE(bindings.empty());
    bool found_board = false;
    for (const auto& match : bindings[0].latest_matches) {
      if (match.element_id == board_id) found_board = true;
    }
    EXPECT_TRUE(found_board)
        << "Board should be in spacing matches before unshare";
  }

  // Unshare the Board.
  space_a_->UnshareElement(board_id);
  WaitForPropagation();

  // space_b should no longer have the Board.
  EXPECT_EQ(space_b_->GetElementTree()->GetElementBySharedId(200), nullptr)
      << "Board should be removed from space_b after unshare";

  // Remove the element from space_a's tree too (unshare only propagates the
  // event; the source element stays in space_a unless explicitly removed).
  ASSERT_TRUE(space_a_->GetElementTree()->RemoveElement(board_id).ok());

  // Update actor again — Board should NOT appear in matches.
  space_a_->UpdateActorState(actor_id, {4.0, 4.0}, 0.0);
  {
    const auto& bindings = space_a_->GetBindings(actor_id);
    ASSERT_FALSE(bindings.empty());
    for (const auto& match : bindings[0].latest_matches) {
      EXPECT_NE(match.element_id, board_id)
          << "Board should NOT appear in spacing matches after removal";
    }
  }
}

// ============================================================================
// Test 3: RuntimeAddElement_CrossSpaceRelationTransfer
//
// Two spaces with Room only.  Actor in space_a, avatar in space_b.
// space_a adds Board -> ShareElement.  space_b receives Board, sets its own
// geometry.  Actor approaches Board -> relation propagation occurs.
// ============================================================================

TEST_P(RuntimeElementE2ETest,
       RuntimeAddElement_CrossSpaceRelationTransfer) {
  // Actor in space_a, bound with spacing.
  auto actor_a = CreateBoundActor(space_a_);

  // Avatar actor in space_b (receiver, no binding — receives remote updates).
  auto actor_b =
      space_b_->CreateActor({SubjectAnchor("root")});
  space_b_->GetActor(actor_b)->SetElementId(kRoomLocalId);

  // Add Board to space_a and share.
  element_id_t board_a_id =
      AddObjectToSpaceA(/*shared_id=*/200, 5.0, 5.0);
  space_a_->ShareElement(board_a_id);
  WaitForPropagation();

  // space_b should have received the Board.
  Element* board_b =
      space_b_->GetElementTree()->GetElementBySharedId(200);
  ASSERT_NE(board_b, nullptr)
      << "Board should be propagated to space_b";

  // Set geometry on space_b's Board (target sets own geometry).
  auto board_poly = std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{3, 3}, {7, 3}, {7, 7}, {3, 7}});
  board_b->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(board_poly));

  // Actor approaches Board in space_a.
  space_a_->UpdateActorState(actor_a, {4.0, 4.0}, 0.0);

  // Wait for relation update propagation in network mode.
  WaitForPropagation();

  // Verify relation payloads were propagated to space_b.
  const auto& received = space_b_->GetReceivedUpdates();
  EXPECT_FALSE(received.empty())
      << "space_b should have received relation updates after actor "
         "approaches Board";

  // The update should reference the Board's shared_id.
  if (!received.empty()) {
    bool has_board_payload = false;
    for (const auto& payload : received.back().payloads) {
      if (payload.referent_shared_id == 200) {
        has_board_payload = true;
      }
    }
    EXPECT_TRUE(has_board_payload)
        << "Relation payload should reference Board shared_id=200";
  }
}

// ============================================================================
// Test 4: RuntimeAddDuringSimulation_ContinuousTrajectory
//
// Actor is walking (multiple UpdateActorState calls).  Mid-trajectory:
// space_a adds a new element -> ShareElement.  Verify: no crash, trajectory
// continues, new element appears in scope.
// ============================================================================

TEST_P(RuntimeElementE2ETest,
       RuntimeAddDuringSimulation_ContinuousTrajectory) {
  auto actor_id = CreateBoundActor(space_a_);

  // Simulate walking trajectory.
  for (int i = 0; i < 5; ++i) {
    double x = 1.0 + i * 0.5;
    space_a_->UpdateActorState(actor_id, {x, 3.0}, 0.0);
  }

  // Mid-trajectory: add a new element.
  element_id_t board_id = AddObjectToSpaceA(/*shared_id=*/200, 5.0, 5.0);
  space_a_->ShareElement(board_id);
  WaitForPropagation();

  // Continue walking — should not crash.
  for (int i = 5; i < 10; ++i) {
    double x = 1.0 + i * 0.5;
    space_a_->UpdateActorState(actor_id, {x, 3.0}, 0.0);
  }

  // New element should appear in space_b.
  EXPECT_NE(space_b_->GetElementTree()->GetElementBySharedId(200), nullptr)
      << "Board should appear in space_b after mid-trajectory share";

  // Verify the Board now appears in spacing matches.
  const auto& bindings = space_a_->GetBindings(actor_id);
  ASSERT_FALSE(bindings.empty());
  bool found_board = false;
  for (const auto& match : bindings[0].latest_matches) {
    if (match.element_id == board_id) {
      found_board = true;
    }
  }
  EXPECT_TRUE(found_board)
      << "Board should appear in spacing matches after mid-trajectory add";
}

// ============================================================================
// Test 5: MultipleRuntimeAdds_AllPropagated
//
// Add 3 elements in sequence: Board, Table, Chair.  ShareElement for each.
// Verify space_b has all 3.
// ============================================================================

TEST_P(RuntimeElementE2ETest, MultipleRuntimeAdds_AllPropagated) {
  // Add and share 3 elements.
  element_id_t board_id =
      AddObjectToSpaceA(/*shared_id=*/200, 5.0, 5.0, "Board");
  space_a_->ShareElement(board_id);

  element_id_t table_id =
      AddObjectToSpaceA(/*shared_id=*/300, -3.0, 4.0, "Table");
  space_a_->ShareElement(table_id);

  element_id_t chair_id =
      AddObjectToSpaceA(/*shared_id=*/400, 2.0, -2.0, "Chair");
  space_a_->ShareElement(chair_id);

  WaitForPropagation();

  // Verify space_b has all 3 elements.
  auto* tree_b = space_b_->GetElementTree();

  const Element* board_b = tree_b->GetElementBySharedId(200);
  ASSERT_NE(board_b, nullptr) << "Board should be in space_b";
  EXPECT_EQ(board_b->GetTag(), "Board");

  const Element* table_b = tree_b->GetElementBySharedId(300);
  ASSERT_NE(table_b, nullptr) << "Table should be in space_b";
  EXPECT_EQ(table_b->GetTag(), "Table");

  const Element* chair_b = tree_b->GetElementBySharedId(400);
  ASSERT_NE(chair_b, nullptr) << "Chair should be in space_b";
  EXPECT_EQ(chair_b->GetTag(), "Chair");
}

// ============================================================================
// Instantiation — both Local and Network transports.
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Transport, RuntimeElementE2ETest,
    ::testing::Values(TransportMode::kLocal, TransportMode::kNetwork),
    TransportName);

}  // namespace test
}  // namespace spatialblend
