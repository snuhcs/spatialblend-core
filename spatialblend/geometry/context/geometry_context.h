// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_GEOMETRY_GEOMETRY_CONTEXT_H_
#define SPATIALBLEND_GEOMETRY_GEOMETRY_CONTEXT_H_

#include <Eigen/Dense>
#include <memory>

#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/geometry/path/discrete_path.h"

namespace spatialblend {

// Base class for precomputed geometry.
// Each SpatialDefinition subclass lazily builds its own context subclass.
// Shared via shared_ptr — atomically invalidated by replacing the pointer.
class GeometryContext {
 public:
  virtual ~GeometryContext() = default;

  // Common queries available on all geometry types
  virtual bool Contains(const Eigen::Vector2d& point) const = 0;
  virtual Eigen::Vector2d GetNearestPoint(const Eigen::Vector2d& from) const = 0;
  virtual double GetDistance(const Eigen::Vector2d& from) const = 0;
  virtual Eigen::Vector3d ProjectOnto(const Eigen::Vector3d& point) const = 0;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_GEOMETRY_GEOMETRY_CONTEXT_H_
