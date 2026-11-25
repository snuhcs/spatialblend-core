// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/affordance_pose_grounding_rule.h"

#include <cmath>
#include <limits>

#include "spatialblend/geometry/radial_surface_chart.h"

namespace spatialblend {

namespace {

bool IsUsableBoundary(const std::shared_ptr<Polygon2D>& boundary) {
  // Contact admission uses the same explicit chart domain as target grounding.
  // Invalid/non-star patches are not eligible for nearest-patch selection;
  // they cannot silently fall back to MVC or the whole enclosing surface.
  return boundary && RadialSurfaceChart::Origin(*boundary).ok();
}

bool IsUsableTransform(const Eigen::Matrix4d& transform) {
  const double determinant = transform.determinant();
  return transform.allFinite() && std::isfinite(determinant) &&
         std::abs(determinant) > 1e-12;
}

}  // namespace

Support AffordancePoseGroundingRule::ComputeSupport(
    const Element& referent, const ElementTree& tree,
    const Eigen::Vector3d& anchor_pos, double anchor_yaw,
    const NavigableSpatialDefinition* navigable_def) const {
  Support support;
  auto* spatial_def = referent.GetDeclaredSpatialDefinition();
  support.referent_element_id = referent.GetId();
  support.navigable_def = navigable_def;

  if (!spatial_def) {
    // Geometryless affordances retain the legacy element-origin chart. The
    // affordance kernel's no-boundary fallback measures its raw local offset
    // in this frame, so identity here would incorrectly measure from world
    // origin rather than from the referent.
    const Eigen::Matrix4d world = tree.GetWorldTransform(referent.GetId());
    if (IsUsableTransform(world)) support.world_transform = world;
    return support;
  }
  if (!anchor_pos.allFinite()) return support;

  const bool local_frame = spatial_def->RequiresElementWorldTransform();
  const Eigen::Matrix4d to_world = tree.GetWorldTransform(referent.GetId());
  if (local_frame && !IsUsableTransform(to_world)) return support;
  Eigen::Vector3d query_anchor = anchor_pos;
  if (local_frame) {
    const Eigen::Matrix4d to_local = to_world.inverse();
    query_anchor =
        (to_local * Eigen::Vector4d(anchor_pos.x(), anchor_pos.y(),
                                    anchor_pos.z(), 1.0))
            .head<3>();
    if (!query_anchor.allFinite()) return support;
  }
  Eigen::Vector3d projected = spatial_def->ProjectOnto(query_anchor);
  if (!projected.allFinite()) return support;
  if (local_frame) {
    projected = (to_world * Eigen::Vector4d(projected.x(), projected.y(),
                                            projected.z(), 1.0))
                    .head<3>();
  }
  const Eigen::Matrix4d boundary_to_world =
      local_frame ? to_world : spatial_def->GetTransform();
  if (!IsUsableTransform(boundary_to_world)) return support;
  bool grounded_contact = false;

  // Sub-region selection: when the referent's definition
  // carries named sub-regions, ground onto the bounded patch plane nearest to
  // the real 3D anchor. A patch chart is relative to the definition chart, so
  // compose it with the actual definition-to-world transform first.
  if (!spatial_def->GetSubRegions().empty()) {
    const SubRegion* best_sub_region = nullptr;
    double best_dist = std::numeric_limits<double>::max();
    bool best_inside = false;
    Eigen::Vector2d best_local = Eigen::Vector2d::Zero();
    Eigen::Vector3d best_world = Eigen::Vector3d::Zero();
    Eigen::Matrix4d best_chart = Eigen::Matrix4d::Identity();
    for (const auto& sub_region : spatial_def->GetSubRegions()) {
      if (!IsUsableBoundary(sub_region.boundary) ||
          !IsUsableTransform(sub_region.local_transform)) {
        continue;
      }
      const Eigen::Matrix4d patch_to_world =
          boundary_to_world * sub_region.local_transform;
      if (!IsUsableTransform(patch_to_world)) continue;
      const Eigen::Vector4d patch_anchor4 =
          patch_to_world.inverse() *
          Eigen::Vector4d(anchor_pos.x(), anchor_pos.y(), anchor_pos.z(), 1.0);
      if (!patch_anchor4.allFinite()) continue;
      Eigen::Vector2d patch_local = patch_anchor4.head<2>();
      const bool inside = sub_region.boundary->Contains(patch_local);
      if (!inside) {
        patch_local = sub_region.boundary->GetNearestPoint(patch_local);
      }
      const Eigen::Vector3d patch_world =
          (patch_to_world *
           Eigen::Vector4d(patch_local.x(), patch_local.y(), 0.0, 1.0))
              .head<3>();
      if (!patch_world.allFinite()) continue;
      const double d = (anchor_pos - patch_world).squaredNorm();
      // Strict comparison intentionally preserves declaration order on ties.
      if (d < best_dist) {
        best_dist = d;
        best_inside = inside;
        best_sub_region = &sub_region;
        best_local = patch_local;
        best_world = patch_world;
        best_chart = patch_to_world;
      }
    }
    if (best_sub_region) {
      support.boundary = best_sub_region->boundary;
      support.sub_region_name = best_sub_region->name;
      support.sub_region_chart = static_cast<int32_t>(best_sub_region->chart);
      support.contact_point_local = best_local;
      support.contact_point_world = best_world;
      support.world_transform = best_chart;
      support.anchor_over_footprint = best_inside;
      grounded_contact = true;
    }
  } else {
    const std::shared_ptr<Polygon2D> boundary = spatial_def->GetBoundary();
    if (!IsUsableBoundary(boundary)) return support;
    const Eigen::Vector4d support_in_boundary =
        boundary_to_world.inverse() *
        Eigen::Vector4d(projected.x(), projected.y(), projected.z(), 1.0);
    if (support_in_boundary.allFinite()) {
      support.contact_point_local = support_in_boundary.head<2>();
      support.contact_point_world = projected;
      support.world_transform = boundary_to_world;
      support.boundary = boundary;
      const Eigen::Vector4d anchor_in_boundary =
          boundary_to_world.inverse() *
          Eigen::Vector4d(anchor_pos.x(), anchor_pos.y(), anchor_pos.z(), 1.0);
      support.anchor_over_footprint =
          anchor_in_boundary.allFinite() &&
          boundary->Contains(anchor_in_boundary.head<2>());
      grounded_contact = true;
    }
  }

  // Normalize contact onto the (sub-region or whole) boundary polygon
  if (grounded_contact && IsUsableBoundary(support.boundary) &&
      support.contact_point_local.allFinite() &&
      support.contact_point_world.allFinite() &&
      support.world_transform.allFinite()) {
    Eigen::Vector2d forward(0, 1);
    auto normalized = RadialSurfaceChart::Encode(
        support.contact_point_local, *support.boundary, forward);
    if (normalized.ok()) {
      support.contact_normalized = *normalized;
      support.has_contact = true;
    }
  }

  return support;
}

}  // namespace spatialblend
