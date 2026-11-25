// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_GEOMETRY_NORMALIZED_INTERIOR_H_
#define SPATIALBLEND_GEOMETRY_NORMALIZED_INTERIOR_H_

#include <Eigen/Dense>
#include <vector>

namespace spatialblend {

// Canonical, type-agnostic representation of a position inside a spatial
// definition's interior. It is the single interlingua by which interior pose
// transfers between ANY two spatial-definition types (room->desk, box->disk,
// 2D->volume): the source encodes with its own chart, only this coordinate
// crosses the wire, and the target decodes with its own chart.
//
// Two channels, by design (capability negotiation):
//   * uvw  — the UNIVERSAL fallback every type can produce and consume:
//            a forward-aligned normalized frame, centroid origin, each
//            component in [-1, 1]; z is the height ratio (0 for planar shapes).
//            Exact for affine-similar boxes; approximate across shape families.
//   * kind + weights — an OPTIONAL richer representation, meaningful only when
//            the decoding type shares the same Kind. e.g. kMVC carries Mean
//            Value Coordinate weights over a forward-anchored, arc-length
//            resampled boundary, which a polygon target applies to ITS boundary
//            for shape-faithful transfer. A target of a different kind ignores
//            `weights` and falls back to `uvw`.
struct NormalizedInterior {
  enum class Kind {
    kBoxUVW,  // forward-aligned bounding-box coordinates (universal/default)
    kMVC,     // mean value coordinates over a resampled boundary (polygons)
    kPolar,   // radial (r, theta) charts (disks / round tables)
  };

  Kind kind = Kind::kBoxUVW;

  // Universal fallback frame. Always populated, by every chart.
  Eigen::Vector3d uvw = Eigen::Vector3d::Zero();

  // Optional rich payload; interpretation depends on `kind`. Empty for kBoxUVW.
  std::vector<double> weights;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_GEOMETRY_NORMALIZED_INTERIOR_H_
