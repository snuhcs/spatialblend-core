// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_ELEMENT_SUB_REGION_H_
#define SPATIALBLEND_ELEMENT_SUB_REGION_H_

#include <Eigen/Dense>
#include <memory>
#include <string>

#include "spatialblend/geometry/planar_frame_chart.h"
#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {

// A named, tagged patch of a SpatialDefinition (e.g. a Sittable seat
// patch, a board's writable face, a container rim). Sub-regions are the
// unit of cross-space affordance correspondence: same-named sub-regions
// on matched referents map onto each other via normalized polygon
// coordinates.
struct SubRegion {
  std::string name;
  std::string tag;
  // Local transform of the patch relative to the owning SpatialDefinition.
  Eigen::Matrix4d local_transform = Eigen::Matrix4d::Identity();
  // Patch boundary in the patch's local frame.
  std::shared_ptr<Polygon2D> boundary;
  // How a point of this patch is read on the matched patch of another space (data of the part, not a solver mode).
  // kRadial (default) is the chart every patch had; the frame charts need an axis-aligned rectangular boundary and keep
  // (u,v) of the rectangle across boards of any size and aspect ratio (planar_frame_chart.h). Both spaces declare the
  // same kind for matched patches; a mismatch is reported by the receiver, never read with a guessed chart.
  SurfaceChartKind chart = SurfaceChartKind::kRadial;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_ELEMENT_SUB_REGION_H_
