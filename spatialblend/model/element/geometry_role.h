// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_ELEMENT_GEOMETRY_ROLE_H_
#define SPATIALBLEND_ELEMENT_GEOMETRY_ROLE_H_

namespace spatialblend {

enum class GeometryRole {
  kUnspecified = 0,
  kNavigable = 1,
  kContainment = 2,
  kObstacle = 3,
  kSurface = 4,
  kInteractionRegion = 5,
  kCount
};

inline const char* GeometryRoleToString(GeometryRole role) {
  switch (role) {
    case GeometryRole::kUnspecified:
      return "Unspecified";
    case GeometryRole::kNavigable:
      return "Navigable";
    case GeometryRole::kContainment:
      return "Containment";
    case GeometryRole::kObstacle:
      return "Obstacle";
    case GeometryRole::kSurface:
      return "Surface";
    case GeometryRole::kInteractionRegion:
      return "InteractionRegion";
    case GeometryRole::kCount:
      return "Count";
  }
  return "Unknown";
}

inline bool IsContainmentRole(GeometryRole role) {
  return role == GeometryRole::kNavigable ||
         role == GeometryRole::kContainment;
}

inline bool IsInteractionRole(GeometryRole role) {
  return role == GeometryRole::kInteractionRegion;
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_ELEMENT_GEOMETRY_ROLE_H_
