// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/planar_frame_chart.h
#ifndef SPATIALBLEND_GEOMETRY_PLANAR_FRAME_CHART_H_
#define SPATIALBLEND_GEOMETRY_PLANAR_FRAME_CHART_H_

#include <cstdint>

#include "Eigen/Core"
#include "absl/status/statusor.h"
#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {

// How a part's points are read between two matched parts (data of the part, SubRegion::chart). kRadial is the
// pure radial chart that every part had before (RadialSurfaceChart): the seat, the stool, the sofa. The two frame
// charts are for a part that is a RECTANGLE in its own frame (a board face, a desk top, a shelf):
//   kFrameStretch  u,v in [0,1]^2 of the rectangle are preserved (per-axis scaling). The whole source part maps
//                  onto the whole receiver part; a different aspect ratio stretches the content.
//   kFrameContain  the source rectangle maps onto the largest centred rectangle of the SAME aspect ratio that fits
//                  the receiver part (letterbox / pillarbox): one uniform scale, content shape preserved, the rest
//                  of the receiver part is margin that no source point maps to.
// The out-of-plane (normal) offset is metric in both: a hand 0.1 m in front of the part stays 0.1 m in front.
// For two parts of equal aspect ratio the three charts that exist (radial, stretch, contain) coincide up to the
// uniform scale only for the frame charts; the radial chart is NOT (u,v) preserving across aspect ratios (the ray
// through the centre keeps its angle, not its (u,v)).
enum class SurfaceChartKind : int32_t { kRadial = 0, kFrameStretch = 1, kFrameContain = 2 };

constexpr bool IsFrameChart(SurfaceChartKind k) { return k == SurfaceChartKind::kFrameStretch || k == SurfaceChartKind::kFrameContain; }
absl::StatusOr<SurfaceChartKind> SurfaceChartKindFromInt(int32_t v);

// The axis-aligned rectangle [x0,x1] x [y0,y1] of a part boundary in its own frame.
struct FrameExtent {
  double x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  double Width() const { return x1 - x0; }
  double Height() const { return y1 - y0; }
};

// A point of a part in chart coordinates. (u,v) are the normalized position in the SOURCE rectangle, `normal_m` the
// metric offset along the part normal (+z of the part frame), and the source extents travel with it because the
// receiver's letterbox depends on the source aspect ratio.
struct FrameCoord {
  double u = 0, v = 0;
  double normal_m = 0;
  double src_width_m = 0, src_height_m = 0;
};

class PlanarFrameChart {
 public:
  // Error (kInvalidArgument) unless the boundary is an axis-aligned rectangle of finite positive width and height.
  static absl::StatusOr<FrameExtent> ExtentOf(const Polygon2D& boundary);

  // A part-local point (x, y, normal_m) read in the part's own rectangle.
  static absl::StatusOr<FrameCoord> Encode(const Eigen::Vector2d& point, double normal_m, const Polygon2D& part);

  // The chart coordinate read on the receiver part: (x', y', normal_m') in the receiver part's local frame.
  static absl::StatusOr<Eigen::Vector3d> Decode(const FrameCoord& c, const Polygon2D& target, SurfaceChartKind kind);

  // Encode on `source`, Decode on `target`.
  static absl::StatusOr<Eigen::Vector3d> Map(const Eigen::Vector2d& point, double normal_m, const Polygon2D& source,
                                             const Polygon2D& target, SurfaceChartKind kind);
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_GEOMETRY_PLANAR_FRAME_CHART_H_
