// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_HARNESS_HIERARCHY_EXTRACTOR_FIXTURE_H_
#define SPATIALBLEND_HARNESS_HIERARCHY_EXTRACTOR_FIXTURE_H_

#include <Eigen/Dense>

#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/link/element_linker.h"

namespace spatialblend::test {

// A deliberately small, analytic source/target pair for component tests.
// It represents a chair seat-A and a sofa seat-A/seat-B pair.  The linker is
// intentionally explicit: it expresses a reviewed support correspondence and
// does not infer one from ids, tags, or subtree shape.
struct HierarchyExtractorFixture {
  static constexpr space_id_t kSourceSpace = 101;
  static constexpr space_id_t kTargetSpace = 202;

  static constexpr element_id_t kSourceRoot = 1;
  static constexpr element_id_t kSourceRegion = 2;
  static constexpr element_id_t kSourceChair = 3;
  static constexpr element_id_t kSourceSeatA = 4;

  static constexpr element_id_t kTargetRoot = 11;
  static constexpr element_id_t kTargetRegion = 12;
  static constexpr element_id_t kTargetSofa = 13;
  static constexpr element_id_t kTargetSeatA = 14;
  static constexpr element_id_t kTargetSeatB = 15;

  HierarchyExtractorFixture();

  // Z-up transforms used to independently calculate oracle coordinates.
  static Eigen::Matrix4d Transform(double x, double y, double z, double yaw);

  Eigen::Vector3d SourceSeatCenterWorld() const;
  Eigen::Vector3d SourceSeatTopWorld() const;
  Eigen::Vector3d SourcePointInRegionOutsideChairWorld() const;
  Eigen::Vector3d SourcePointOutsideRegionWorld() const;
  Eigen::Vector3d TargetSeatACenterWorld() const;
  Eigen::Vector3d TargetSeatBCenterWorld() const;

  ElementTree source_tree;
  ElementTree target_tree;
  ElementLinker linker;
};

}  // namespace spatialblend::test

#endif  // SPATIALBLEND_HARNESS_HIERARCHY_EXTRACTOR_FIXTURE_H_
