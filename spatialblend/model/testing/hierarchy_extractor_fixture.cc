// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/testing/hierarchy_extractor_fixture.h"

#include <cmath>
#include <cstdlib>
#include <memory>

#include "spatialblend/model/element/polytope_spatial_definition.h"

namespace spatialblend::test {
namespace {

std::shared_ptr<PolytopeSpatialDefinition> ContainmentBox(double half_extent) {
  return std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-half_extent, -half_extent, -1.0),
      Eigen::Vector3d(half_extent, half_extent, 1.0),
      GeometryRole::kContainment);
}

std::shared_ptr<PolytopeSpatialDefinition> SeatBox() {
  return std::make_shared<PolytopeSpatialDefinition>(
      Eigen::Vector3d(-0.25, -0.25, 0.0), Eigen::Vector3d(0.25, 0.25, 0.10),
      GeometryRole::kInteractionRegion);
}

void AddElementOrAbort(ElementTree* tree, element_id_t id,
                       element_id_t parent_id = kInvalidElementId) {
  if (!tree->AddElement(id, parent_id).ok()) std::abort();
  tree->GetElement(id)->SetSharedId(id);  // shared: an unshared element is never a referent (SP-D7-03)
}

}  // namespace

Eigen::Matrix4d HierarchyExtractorFixture::Transform(double x, double y,
                                                     double z, double yaw) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform(0, 0) = std::cos(yaw);
  transform(0, 1) = -std::sin(yaw);
  transform(1, 0) = std::sin(yaw);
  transform(1, 1) = std::cos(yaw);
  transform(0, 3) = x;
  transform(1, 3) = y;
  transform(2, 3) = z;
  return transform;
}

HierarchyExtractorFixture::HierarchyExtractorFixture() {
  AddElementOrAbort(&source_tree, kSourceRoot);
  AddElementOrAbort(&source_tree, kSourceRegion, kSourceRoot);
  AddElementOrAbort(&source_tree, kSourceChair, kSourceRegion);
  AddElementOrAbort(&source_tree, kSourceSeatA, kSourceChair);
  source_tree.GetElement(kSourceRegion)
      ->SetLocalTransform(Transform(10.0, 5.0, 0.0, M_PI / 2.0));
  source_tree.GetElement(kSourceRegion)
      ->SetSpatialDefinition(ContainmentBox(2.0));
  source_tree.GetElement(kSourceRegion)->SetTag("meeting-region");
  source_tree.GetElement(kSourceChair)
      ->SetLocalTransform(Transform(1.0, 0.0, 0.0, 0.0));
  source_tree.GetElement(kSourceChair)
      ->SetSpatialDefinition(ContainmentBox(0.5));
  source_tree.GetElement(kSourceChair)->SetTag("chair");
  source_tree.GetElement(kSourceSeatA)
      ->SetLocalTransform(Transform(0.0, 0.0, 0.45, 0.0));
  source_tree.GetElement(kSourceSeatA)->SetSpatialDefinition(SeatBox());
  source_tree.GetElement(kSourceSeatA)->SetTag("seat-A");

  AddElementOrAbort(&target_tree, kTargetRoot);
  AddElementOrAbort(&target_tree, kTargetRegion, kTargetRoot);
  AddElementOrAbort(&target_tree, kTargetSofa, kTargetRegion);
  AddElementOrAbort(&target_tree, kTargetSeatA, kTargetSofa);
  AddElementOrAbort(&target_tree, kTargetSeatB, kTargetSofa);
  target_tree.GetElement(kTargetRegion)
      ->SetLocalTransform(Transform(-4.0, 3.0, 0.0, -M_PI / 2.0));
  target_tree.GetElement(kTargetRegion)
      ->SetSpatialDefinition(ContainmentBox(3.0));
  target_tree.GetElement(kTargetRegion)->SetTag("conversation-region");
  target_tree.GetElement(kTargetSofa)
      ->SetLocalTransform(Transform(1.0, 0.0, 0.0, 0.0));
  target_tree.GetElement(kTargetSofa)
      ->SetSpatialDefinition(ContainmentBox(1.5));
  target_tree.GetElement(kTargetSofa)->SetTag("sofa");
  target_tree.GetElement(kTargetSeatA)
      ->SetLocalTransform(Transform(-0.5, 0.0, 0.35, 0.0));
  target_tree.GetElement(kTargetSeatA)->SetSpatialDefinition(SeatBox());
  target_tree.GetElement(kTargetSeatA)->SetTag("seat-A");
  target_tree.GetElement(kTargetSeatB)
      ->SetLocalTransform(Transform(0.5, 0.0, 0.35, 0.0));
  target_tree.GetElement(kTargetSeatB)->SetSpatialDefinition(SeatBox());
  target_tree.GetElement(kTargetSeatB)->SetTag("seat-B");

  linker.CreateLink({kSourceSpace, kSourceSeatA}, {kTargetSpace, kTargetSeatA});
}

Eigen::Vector3d HierarchyExtractorFixture::SourceSeatCenterWorld() const {
  return Eigen::Vector3d(10.0, 6.0, 0.45);
}

Eigen::Vector3d HierarchyExtractorFixture::SourceSeatTopWorld() const {
  // Seat-local (.10, -.05, .10) rotated by the region's +90 degree yaw.
  return Eigen::Vector3d(10.05, 6.10, 0.55);
}

Eigen::Vector3d
HierarchyExtractorFixture::SourcePointInRegionOutsideChairWorld() const {
  return Eigen::Vector3d(10.0, 5.0, 0.0);
}

Eigen::Vector3d HierarchyExtractorFixture::SourcePointOutsideRegionWorld()
    const {
  return Eigen::Vector3d(13.0, 5.0, 0.0);
}

Eigen::Vector3d HierarchyExtractorFixture::TargetSeatACenterWorld() const {
  return Eigen::Vector3d(-4.0, 2.5, 0.35);
}

Eigen::Vector3d HierarchyExtractorFixture::TargetSeatBCenterWorld() const {
  return Eigen::Vector3d(-4.0, 1.5, 0.35);
}

}  // namespace spatialblend::test
