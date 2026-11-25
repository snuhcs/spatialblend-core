// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <atomic>
#include <limits>
#include <thread>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {
namespace {

constexpr uint32_t kSceneDomains =
    kRealizationTreeRevision | kRealizationSupportRevision;

std::shared_ptr<Polygon2D> Square(double radius = 1) {
  return std::make_shared<Polygon2D>(
      std::vector<Eigen::Vector2d>{{-radius, -radius},
                                   {radius, -radius},
                                   {radius, radius},
                                   {-radius, radius}});
}

SubRegion Seat(double height = 0.4) {
  SubRegion region;
  region.name = "seat";
  region.tag = "sittable";
  region.boundary = Square();
  region.local_transform(2, 3) = height;
  return region;
}

class OwnedSceneInputsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    space.SetSpaceId(2);
    space.SetRichRealizationEnabled(true);
    ASSERT_TRUE(space.AddElement(10));
    ASSERT_TRUE(space.SetElementSharedId(10, 500));
    ASSERT_TRUE(space.AddElement(20, 10));
    ASSERT_TRUE(space.SetElementSharedId(20, 42));
    definition = std::make_shared<NavigableSpatialDefinition>(Square());
    definition->AddSubRegion(Seat());
    ASSERT_TRUE(space.SetElementSpatialDefinition(20, definition));
    actor = space.CreateActor({SubjectAnchor("root"), SubjectAnchor("hand")});
    ASSERT_TRUE(space.SetActorElement(actor, 10));
    driver = std::make_unique<IdealDriver>(&space, actor);
    RelationPayload support;
    support.referent_shared_id = 42;
    support.subject_anchor_name = "hand";
    support.kernel_type = RelationKernelType::kAvatarAffordancePose;
    support.grounded_relation.proximity = 1.0;
    // A source dwelling in contact: v1's count is 1 (SourceUpdateCount(0, 0)), a request. Without a count Progressive
    // computes no weight.
    support.update_count = SourceUpdateCount(0.0, 0.0);
    support.grounded_relation.encoded_state = Eigen::VectorXd::Zero(3);
    support.support.has_contact = true;
    support.support.sub_region_name = "seat";
    support.support.contact_normalized.strategy =
        NormalizedPolygonPosition::Strategy::kRadial;
    support.support.contact_normalized.angle = 0;
    support.support.contact_normalized.radial_ratio = 0;
    update.actor_shared_id = 500;
    update.source_space_id = 1;
    update.payloads = {support};
  }

  ActorRealizationSnapshot Solve() {
    space.OnRemoteUpdate(update);
    ActorRealizationSnapshot snapshot;
    EXPECT_TRUE(space.CopyActorRealizationSnapshot(actor, &snapshot));
    EXPECT_EQ(snapshot.known_revision_mask & kSceneDomains, kSceneDomains);
    EXPECT_EQ(snapshot.unknown_revision_mask & kSceneDomains, 0u);
    return snapshot;
  }

  ActorRealizationValidity Validity(const ActorRealizationSnapshot& snapshot) {
    ActorRealizationValidity value;
    EXPECT_TRUE(space.GetActorRealizationValidity(actor, snapshot.revision,
                                                  kSceneDomains, &value));
    return value;
  }

  Space space;
  std::unique_ptr<IdealDriver> driver;  // after `space`: destroyed first
  spatialblend::entity_id_t actor = 0;
  std::shared_ptr<NavigableSpatialDefinition> definition;
  RemoteRelationUpdate update;
};

TEST_F(OwnedSceneInputsTest, SnapshotTracksOwnedGeometryAndNoopWrites) {
  const auto snapshot = Solve();
  ASSERT_EQ(snapshot.channels.size(), 1u);
  ASSERT_EQ(snapshot.channels[0].contributors.size(), 1u);
  EXPECT_TRUE(snapshot.channels[0].contributors[0].has_mapped_support);
  EXPECT_NEAR(snapshot.channels[0].contributors[0].mapped_support.z(), 0.4,
              1e-12);
  EXPECT_TRUE(space.HasElement(20));
  EXPECT_EQ(space.FindElementBySharedId(42), 20);
  EXPECT_EQ(space.FindActorBySharedId(500), actor);
  Space::ElementState detached;
  ASSERT_TRUE(space.CopyElementState(20, &detached));
  EXPECT_TRUE(space.SetElementTransform(20, Eigen::Matrix4d::Identity()));
  EXPECT_TRUE(space.SetElementSharedId(20, 42));
  EXPECT_TRUE(space.SetElementTag(20, ""));
  EXPECT_TRUE(space.SetElementSpatialDefinition(20, definition));
  EXPECT_TRUE(space.SetActorElement(actor, 10));
  EXPECT_TRUE(space.ReparentElement(20, 10));
  EXPECT_FALSE(space.AddElement(20, 10));
  EXPECT_FALSE(space.AddElement(kInvalidElementId));
  EXPECT_FALSE(space.AddElement(30, 999));
  EXPECT_FALSE(space.ReparentElement(10, 20));
  EXPECT_FALSE(space.SetElementTransform(20, Eigen::Matrix4d::Zero()));
  auto malformed = Eigen::Matrix4d::Identity().eval();
  malformed(0, 3) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(space.AddElementAnchor(20, "bad", malformed));
  EXPECT_FALSE(space.SetActorElement(actor, 999));
  const auto validity = Validity(snapshot);
  EXPECT_EQ(validity.changed_mask & kSceneDomains, 0u);
  EXPECT_EQ(validity.unchecked_mask & kSceneDomains, 0u);
  EXPECT_EQ(validity.tree_revision, snapshot.tree_revision);
  EXPECT_EQ(validity.support_revision, snapshot.support_revision);
}

TEST_F(OwnedSceneInputsTest,
       PolygonRegionAndReadCopiesCannotMutateOwnedGeometry) {
  Space other;
  ASSERT_TRUE(other.AddElement(1));
  ASSERT_TRUE(other.SetElementSpatialDefinition(1, definition));
  const auto snapshot = Solve();
  *definition->GetWalkablePolygon() = *Square(5);
  *definition->GetSubRegion("seat")->boundary = *Square(8);
  definition->SetAgentRadius(0.2);
  definition->SetGeometryRole(GeometryRole::kObstacle);
  definition->AddSubRegion(Seat(8));
  Space::ElementState detached;
  ASSERT_TRUE(space.CopyElementState(20, &detached));
  *detached.definition->GetBoundary() = *Square(6);
  *detached.definition->GetSubRegion("seat")->boundary = *Square(7);
  detached.definition->SetGeometryRole(GeometryRole::kObstacle);
  Space::ElementState first, second;
  ASSERT_TRUE(space.CopyElementState(20, &first));
  ASSERT_TRUE(other.CopyElementState(1, &second));
  for (const auto* value : {&first, &second}) {
    EXPECT_EQ(value->definition->GetBoundary()->GetPoints(),
              Square()->GetPoints());
    EXPECT_EQ(value->definition->GetSubRegions().size(), 1u);
    EXPECT_EQ(value->definition->GetSubRegion("seat")->boundary->GetPoints(),
              Square()->GetPoints());
    EXPECT_EQ(value->definition->GetGeometryRole(), GeometryRole::kNavigable);
  }
  EXPECT_EQ(Validity(snapshot).changed_mask & kSceneDomains, 0u);
  const auto next = Solve();
  EXPECT_EQ(next.tree_revision, snapshot.tree_revision);
  EXPECT_EQ(next.support_revision, snapshot.support_revision);
  EXPECT_NEAR(next.channels[0].contributors[0].mapped_support.z(), 0.4, 1e-12);
}

TEST_F(OwnedSceneInputsTest, SurfaceAndPolytopeDefinitionsAreDetached) {
  const auto polygon = std::make_shared<Polygon3D>(std::vector<Eigen::Vector3d>{
      {0, 0, 0.5}, {1, 0, 0.5}, {1, 1, 0.5}, {0, 1, 0.5}});
  auto surface = std::make_shared<SurfaceSpatialDefinition>(polygon);
  surface->AddSubRegion(Seat());
  ASSERT_TRUE(space.SetElementSpatialDefinition(20, surface));
  Space::ElementState before, after;
  ASSERT_TRUE(space.CopyElementState(20, &before));
  *polygon->Get2DBoundary() = *Square(6);
  *surface->GetBoundary() = *Square(8);
  surface->SetGeometryRole(GeometryRole::kObstacle);
  ASSERT_TRUE(space.CopyElementState(20, &after));
  EXPECT_EQ(before.definition->GetBoundary()->GetPoints(),
            after.definition->GetBoundary()->GetPoints());
  EXPECT_EQ(before.definition->GetGeometryRole(),
            after.definition->GetGeometryRole());

  auto polytope = std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-1, -1, 0), Eigen::Vector3d(1, 1, 1));
  ASSERT_TRUE(space.SetElementSpatialDefinition(20, polytope));
  ASSERT_TRUE(space.CopyElementState(20, &before));
  *polytope->GetBoundary() = *Square(9);
  polytope->AddSubRegion(Seat());
  ASSERT_TRUE(space.CopyElementState(20, &after));
  EXPECT_EQ(before.definition->GetBoundary()->GetPoints(),
            after.definition->GetBoundary()->GetPoints());
  EXPECT_TRUE(after.definition->GetSubRegions().empty());
}

TEST_F(OwnedSceneInputsTest, EachSceneMutationInvalidatesHistoricalSnapshot) {
  auto check = [&](const std::function<bool()>& mutation, uint32_t domains) {
    const auto before = Solve();
    ASSERT_TRUE(mutation());
    const auto value = Validity(before);
    EXPECT_EQ(value.changed_mask & kSceneDomains, domains);
    EXPECT_EQ(value.current_known_mask & kSceneDomains, kSceneDomains);
    EXPECT_EQ(value.changed_mask & kRealizationPolicyRevision, 0u);
  };
  check([&] { return space.SetElementTag(20, "sittable"); }, kSceneDomains);
  check(
      [&] {
        return space.AddElementAnchor(20, "seat", Eigen::Matrix4d::Identity());
      },
      kSceneDomains);
  check([&] { return space.SetElementSharedId(20, 43); },
        kRealizationTreeRevision);
  ASSERT_TRUE(space.SetElementSharedId(20, 42));
  check([&] { return space.AddElementSubRegion(20, Seat(0.8)); },
        kRealizationSupportRevision);
  check([&] { return space.SetElementSpatialDefinition(20, nullptr); },
        kSceneDomains);
  check([&] { return space.SetElementSpatialDefinition(20, definition); },
        kSceneDomains);
  check([&] { return space.AddElement(30, 10); }, kSceneDomains);
  check([&] { return space.ReparentElement(20, 30); }, kSceneDomains);
  check([&] { return space.RemoveLocalElementSubtree(20); }, kSceneDomains);
}

TEST_F(OwnedSceneInputsTest, ParentTransformChangesWorldSupportAndEpochs) {
  // Polytope patches are element-local. Navigable/surface definitions retain
  // their legacy world-authored convention; those must not be double-moved.
  auto local = std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-1, -1, 0), Eigen::Vector3d(1, 1, 1));
  local->AddSubRegion(Seat());
  ASSERT_TRUE(space.SetElementSpatialDefinition(20, local));
  const auto before = Solve();
  Eigen::Matrix4d parent = Eigen::Matrix4d::Identity();
  parent.block<3, 1>(0, 3) = Eigen::Vector3d(0.2, 0.3, 0.1);
  ASSERT_TRUE(space.SetElementTransform(10, parent));
  EXPECT_EQ(Validity(before).changed_mask & kSceneDomains, kSceneDomains);
  const auto after = Solve();
  const auto& mapped = after.channels[0].contributors[0].mapped_support;
  EXPECT_NEAR(mapped.x(), 0.2, 1e-12);
  EXPECT_NEAR(mapped.y(), 0.3, 1e-12);
  EXPECT_NEAR(mapped.z(), 0.5, 1e-12);
}

TEST_F(OwnedSceneInputsTest,
       ColliderStyleIfAbsentIsAtomicAndPreservesExplicitInput) {
  const auto before = Solve();
  EXPECT_TRUE(space.SetElementSpatialDefinition(
      20, std::make_shared<NavigableSpatialDefinition>(Square(8)), true));
  EXPECT_EQ(Validity(before).changed_mask & kSceneDomains, 0u);
  EXPECT_TRUE(space.SetElementSpatialDefinition(20, nullptr));
  const auto empty = Solve();
  EXPECT_TRUE(space.SetElementSpatialDefinition(20, definition, true));
  EXPECT_EQ(Validity(empty).changed_mask & kSceneDomains, kSceneDomains);
}

TEST_F(OwnedSceneInputsTest,
       RawConstAndMutableViewsPermanentlyWithdrawCertification) {
  const auto before = Solve();
  const Space& read_only_space = space;
  const auto* escaped = read_only_space.GetElementTree();
  auto value = Validity(before);
  EXPECT_EQ(value.current_known_mask & kSceneDomains, 0u);
  EXPECT_EQ(value.unchecked_mask & kSceneDomains, kSceneDomains);
  *escaped->GetElement(20)->GetSpatialDefinition()->GetBoundary() = *Square(2);
  EXPECT_TRUE(space.SetElementSpatialDefinition(20, definition));
  space.OnRemoteUpdate(update);
  ActorRealizationSnapshot after;
  ASSERT_TRUE(space.CopyActorRealizationSnapshot(actor, &after));
  EXPECT_EQ(after.known_revision_mask & kSceneDomains, 0u);
  EXPECT_EQ(after.unknown_revision_mask & kSceneDomains, kSceneDomains);
  space.GetElementTree()->GetElement(20)->SetTag("borrowed");
  EXPECT_EQ(Validity(after).current_known_mask & kSceneDomains, 0u);
}

TEST_F(OwnedSceneInputsTest,
       CustomResolverAndSubclassCannotCertifyHiddenMutableState) {
  class ExtendedDefinition : public NavigableSpatialDefinition {
   public:
    ExtendedDefinition() : NavigableSpatialDefinition(Square()) {}
  };
  const auto before = Solve();
  EXPECT_FALSE(space.SetElementSpatialDefinition(
      20, std::make_shared<ExtendedDefinition>()));
  EXPECT_EQ(Validity(before).changed_mask & kSceneDomains, 0u);
  space.SetReferentResolver(std::make_shared<ReferentResolver>());
  space.SetReferentResolver(nullptr);
  EXPECT_EQ(Validity(before).current_known_mask & kSceneDomains, 0u);
}

TEST_F(OwnedSceneInputsTest, RemoteAddResolveAndUnshareAdvanceSceneEpochs) {
  SpaceHub hub;
  space.SetHub(&hub);
  ElementEvent event;
  event.type = ElementEvent::kAdded;
  event.source_space_id = 1;
  event.source_element_id = 70;
  event.parent_shared_id = 500;
  event.shared_id = 700;
  auto before = Solve();
  space.OnElementEvent(event);
  EXPECT_EQ(Validity(before).changed_mask & kSceneDomains, kSceneDomains);
  const auto received = space.FindElementBySharedId(700);
  ASSERT_NE(received, kInvalidElementId);
  before = Solve();
  space.OnElementEvent(event);  // pending duplicate
  EXPECT_EQ(Validity(before).changed_mask & kSceneDomains, 0u);
  ASSERT_TRUE(space.AddElement(30, received));
  before = Solve();
  space.ResolveSharedElement(received, 20);
  EXPECT_EQ(Validity(before).changed_mask & kSceneDomains, kSceneDomains);
  EXPECT_FALSE(space.HasElement(received));
  Space::ElementState child;
  ASSERT_TRUE(space.CopyElementState(30, &child));
  EXPECT_EQ(child.parent, 10);
  event.type = ElementEvent::kRemoved;
  space.OnElementEvent(event);
  EXPECT_TRUE(space.HasElement(20));  // linked local object is not a replica

  event.type = ElementEvent::kAdded;
  event.source_element_id = 71;
  space.OnElementEvent(event);
  const auto replica = space.FindElementBySharedId(700);
  space.ResolveSharedElement(replica, replica);
  before = Solve();
  event.type = ElementEvent::kRemoved;
  space.OnElementEvent(event);
  EXPECT_EQ(Validity(before).changed_mask & kSceneDomains, kSceneDomains);
  EXPECT_FALSE(space.HasElement(replica));
  space.SetHub(nullptr);
}

TEST_F(OwnedSceneInputsTest,
       ConcurrentGeometryCommitAndSolveNeverMixRegionFields) {
  auto low = definition;
  auto high = std::make_shared<NavigableSpatialDefinition>(Square(2));
  auto high_seat = Seat(0.8);
  high_seat.local_transform(0, 3) = 0.2;
  high->AddSubRegion(high_seat);
  std::atomic<bool> start{false};
  std::thread writer([&] {
    while (!start.load()) std::this_thread::yield();
    for (int i = 0; i < 200; ++i)
      EXPECT_TRUE(space.SetElementSpatialDefinition(20, i % 2 ? low : high));
  });
  start.store(true);
  for (int i = 0; i < 200; ++i) {
    const auto snapshot = Solve();
    EXPECT_EQ(snapshot.channels.size(), 1u);
    if (snapshot.channels.empty()) continue;
    EXPECT_EQ(snapshot.channels[0].contributors.size(), 1u);
    if (snapshot.channels[0].contributors.empty()) continue;
    const auto& point = snapshot.channels[0].contributors[0].mapped_support;
    const bool complete_low =
        (point - Eigen::Vector3d(0, 0, 0.4)).norm() < 1e-12;
    const bool complete_high =
        (point - Eigen::Vector3d(0.2, 0, 0.8)).norm() < 1e-12;
    EXPECT_TRUE(complete_low || complete_high);
  }
  writer.join();
}

TEST_F(OwnedSceneInputsTest, ActorBorrowCannotBypassElementIdentityRevision) {
  const auto snapshot = Solve();
  space.GetActor(actor)->SetElementId(20);
  EXPECT_EQ(Validity(snapshot).current_known_mask & kSceneDomains, 0u);
  ASSERT_TRUE(space.SetActorElement(actor, 10));
  space.OnRemoteUpdate(update);
  ActorRealizationSnapshot next;
  ASSERT_TRUE(space.CopyActorRealizationSnapshot(actor, &next));
  EXPECT_EQ(next.known_revision_mask & kSceneDomains, 0u);
}

TEST_F(OwnedSceneInputsTest, EmbodiedMirrorLandsBeforeTheSolveTransaction) {
  // The driver's published pose reaches the avatar's leaf element (derived
  // mirror, so other local actors see it) BEFORE the solve's scene stamp is
  // taken: the snapshot is certified, never a mixed-input scene.
  ASSERT_TRUE(space.AddElement(30, 10));
  ASSERT_TRUE(space.SetElementSharedId(30, 500));
  ASSERT_TRUE(space.SetActorElement(actor, 30));
  driver->Publish(1.0, 0.0, 0.0);
  space.OnRemoteUpdate(update);
  ActorRealizationSnapshot snapshot;
  ASSERT_TRUE(space.CopyActorRealizationSnapshot(actor, &snapshot));
  Space::ElementState leaf;
  ASSERT_TRUE(space.CopyElementState(30, &leaf));
  EXPECT_NE(leaf.local_transform, Eigen::Matrix4d::Identity());
  EXPECT_EQ(snapshot.known_revision_mask & kSceneDomains, kSceneDomains);
  EXPECT_EQ(snapshot.unknown_revision_mask & kSceneDomains, 0u);
}

TEST_F(OwnedSceneInputsTest, EmbodiedMirrorInvalidatesAnEarlierScene) {
  ASSERT_TRUE(space.AddElement(30, 10));
  ASSERT_TRUE(space.SetElementSharedId(30, 500));
  ASSERT_TRUE(space.SetActorElement(actor, 30));
  driver->Publish(1.0, 0.0, 0.0);
  const auto snapshot = Solve();
  // The body moved: the next sync of the mirror changes the scene.
  driver->Publish(1.5, 0.0, 0.0);
  Eigen::Vector2d position;
  double yaw = 0.0;
  ASSERT_TRUE(space.CopyActorPose(actor, &position, &yaw));
  EXPECT_DOUBLE_EQ(position.x(), 1.5);
  EXPECT_EQ(Validity(snapshot).changed_mask & kSceneDomains, kSceneDomains);
}

TEST_F(OwnedSceneInputsTest,
       ActorReassignmentAndDestructionRetireOldGrounding) {
  ASSERT_TRUE(space.AddElement(30, 10));
  ASSERT_TRUE(space.SetElementSharedId(30, 99));
  const auto other = space.CreateActor({SubjectAnchor("root")});
  ASSERT_TRUE(space.SetActorElement(other, 30));
  EXPECT_EQ(space.FindActorBySharedId(99), other);
  auto before = Solve();
  ASSERT_TRUE(space.SetActorElement(other, kInvalidElementId));
  EXPECT_EQ(Validity(before).changed_mask & kSceneDomains,
            kRealizationTreeRevision);
  EXPECT_EQ(space.FindActorBySharedId(99), spatialblend::kInvalidEntityId);
  ASSERT_TRUE(space.SetActorElement(other, 30));
  before = Solve();
  ASSERT_TRUE(space.DestroyActor(other));
  EXPECT_EQ(Validity(before).changed_mask & kSceneDomains,
            kRealizationTreeRevision);
  Space::ElementState element;
  ASSERT_TRUE(space.CopyElementState(30, &element));
  EXPECT_EQ(element.grounding_actor, spatialblend::kInvalidEntityId);
}

TEST_F(OwnedSceneInputsTest, PlanarReadOwnsExactWholeBoundaryAndRevisions) {
  const auto snapshot = Solve();
  Space::PlanarSupportState value;
  ASSERT_EQ(space.TryCopyPlanarSupport(20, snapshot.tree_revision,
      snapshot.support_revision, &value), Space::PlanarSupportReadResult::kOk);
  EXPECT_EQ(value.definition_kind, 2u);
  EXPECT_EQ(value.role, GeometryRole::kNavigable);
  EXPECT_EQ(value.boundary, Square()->GetPoints());
  EXPECT_EQ(value.plane_to_world, Eigen::Matrix4d::Identity());
  // This is explicitly the whole boundary, not the named seat subregion.
  EXPECT_EQ(value.plane_to_world(2, 3), 0);
  value.boundary[0] = Eigen::Vector2d(100, 200);
  ASSERT_EQ(space.TryCopyPlanarSupport(20, snapshot.tree_revision,
      snapshot.support_revision, &value), Space::PlanarSupportReadResult::kOk);
  EXPECT_EQ(value.boundary, Square()->GetPoints());
  EXPECT_EQ(value.tree_revision, snapshot.tree_revision);
  EXPECT_EQ(value.support_revision, snapshot.support_revision);
  EXPECT_EQ(Validity(snapshot).changed_mask & kSceneDomains, 0u);
}

TEST_F(OwnedSceneInputsTest, PlanarReadRejectsStaleMissingAndBorrowedInputs) {
  const auto snapshot = Solve();
  Space::PlanarSupportState value;
  using Result = Space::PlanarSupportReadResult;
  EXPECT_EQ(space.TryCopyPlanarSupport(20, 0, snapshot.support_revision, &value),
            Result::kUnavailable);
  EXPECT_EQ(space.TryCopyPlanarSupport(20, snapshot.tree_revision, 0, &value),
            Result::kUnavailable);
  EXPECT_EQ(space.TryCopyPlanarSupport(999, snapshot.tree_revision,
      snapshot.support_revision, &value), Result::kUnavailable);
  EXPECT_EQ(space.TryCopyPlanarSupport(20, snapshot.tree_revision + 1,
      snapshot.support_revision, &value), Result::kStale);
  EXPECT_EQ(space.TryCopyPlanarSupport(20, snapshot.tree_revision,
      snapshot.support_revision + 1, &value), Result::kStale);
  space.GetElementTree();  // Legacy mutable escape cannot certify owned data.
  EXPECT_EQ(space.TryCopyPlanarSupport(20, snapshot.tree_revision,
      snapshot.support_revision, &value), Result::kUnavailable);
  EXPECT_TRUE(value.boundary.empty());
}

}  // namespace
}  // namespace spatialblend
