// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/facing_grounding_rule.h"

#include <cmath>
#include <limits>

#include "spatialblend/geometry/polygon_coordinate_mapper.h"

namespace spatialblend {

Support FacingGroundingRule::ComputeSupport(
    const Element& referent, const ElementTree& tree,
    const Eigen::Vector3d& anchor_pos, double anchor_yaw,
    const NavigableSpatialDefinition* navigable_def) const {
  Support support;
  support.world_transform = tree.GetWorldTransform(referent.GetId());
  auto* spatial_def = referent.GetDeclaredSpatialDefinition();
  if (spatial_def) {
    support.boundary = spatial_def->GetBoundary();
  }
  support.referent_element_id = referent.GetId();
  support.navigable_def = navigable_def;

  // Compute ray-boundary intersection in heading direction
  if (support.boundary) {
    // `support.boundary`'s coordinate frame depends on the referent's
    // SpatialDefinition kind (mirrors SpacingGroundingRule::ComputeSupport,
    // which established this pattern first): world frame for
    // NavigableSpatialDefinition, the surface's own derived frame
    // (spatial_def->GetTransform()) for SurfaceSpatialDefinition, and the
    // ELEMENT's local frame (support.world_transform, since GetTransform()
    // is a dead field for this kind) for PolytopeSpatialDefinition — see
    // SpatialDefinition::RequiresElementWorldTransform. The previous version
    // of this function used support.world_transform (the tree-derived
    // transform) UNIFORMLY to convert the anchor/heading before querying the
    // boundary, which is only correct for Polytope; for Navigable/Surface
    // referents this could shift the anchor into the wrong frame relative to
    // the boundary polygon's own vertices, making the ray-hit test miss (and
    // in the ray-miss fallback, silently fail to encode contact_normalized,
    // since PolygonCoordinateMapper::Encode requires the point to lie inside
    // the polygon).
    const bool local_frame = spatial_def->RequiresElementWorldTransform();
    const Eigen::Matrix4d boundary_transform =
        local_frame ? support.world_transform : spatial_def->GetTransform();
    const Eigen::Matrix4d boundary_inv = boundary_transform.inverse();
    const Eigen::Matrix4d tree_inv = support.world_transform.inverse();

    // Actor's heading direction from yaw, both transformed into the
    // boundary's own frame.
    Eigen::Vector2d heading(std::cos(anchor_yaw), std::sin(anchor_yaw));
    Eigen::Vector4d bd_anchor =
        boundary_inv * Eigen::Vector4d(anchor_pos.x(), anchor_pos.y(),
                                       anchor_pos.z(), 1.0);
    Eigen::Vector2d bd_2d(bd_anchor.x(), bd_anchor.y());
    Eigen::Vector4d bd_heading =
        boundary_inv * Eigen::Vector4d(heading.x(), heading.y(), 0.0, 0.0);
    Eigen::Vector2d bd_dir(bd_heading.x(), bd_heading.y());
    if (bd_dir.norm() > 1e-9) bd_dir.normalize();

    // Ray intersection in the boundary's frame.
    auto intersection =
        support.boundary->RayBoundaryIntersection(bd_2d, bd_dir);

    // THE LOOK-AT CONTACT: the point of the boundary at the least ANGLE from the gaze, seen
    // from the anchor: the hit point when the ray hits the referent (angle 0; the nearest hit
    // when there are several), otherwise the boundary point whose bearing is nearest the
    // heading, which is a vertex (the bearing is monotone along an edge, so its extremes are
    // vertices) -- the silhouette vertex on the side the gaze is turned toward. Where the two
    // regimes meet the contact is the same point, and it moves with the heading and the
    // anchor continuously: a vertex is fixed in the referent, and the hit point slides along
    // the boundary. The rule is the minimiser of ONE function, the angular deviation (ties,
    // all on one bearing, to the point nearest the anchor), with no case switch.
    //
    // The previous rule (the boundary point closest, in distance, to the gaze ray) was not
    // continuous: where an edge runs parallel to the heading, every point of it is equally
    // close to the ray, and which end is closer flips as the heading turns through the
    // parallel -- the contact jumped by the length of the edge for a fraction of a degree
    // of turn.
    //
    // The vertex minimiser is itself discontinuous at one heading, for any referent: the
    // heading at which the two silhouette vertices have equal deviation. On either side a
    // single vertex is the strict minimiser, so a tie break (previous contact, embodied
    // yaw, nearer anchor) does not run, and the contact changes by the edge between those
    // vertices. A stateless function of the heading cannot be continuous there.
    const auto& ring = support.boundary->GetPoints();
    const double heading_angle = std::atan2(bd_dir.y(), bd_dir.x());
    Eigen::Vector2d contact_bd_2d = bd_2d;  // in the boundary's own frame
    int32_t chosen_vertex = -1;
    auto vertex_argmin = [&]() {
      double best = std::numeric_limits<double>::infinity(), best_anchor = best;
      int32_t index = -1;
      for (const auto& p : ring) {
        ++index;
        const Eigen::Vector2d to = p - bd_2d;
        const double da = to.norm();
        if (da < 1e-12) {  // the anchor is on the boundary: the contact is the anchor
          best = 0.0;
          best_anchor = 0.0;
          contact_bd_2d = p;
          chosen_vertex = index;
          break;
        }
        const double dev = std::fabs(
            std::remainder(std::atan2(to.y(), to.x()) - heading_angle, 2.0 * M_PI));
        // (angles equal up to the relation layer's geometry epsilon, 1e-9)
        constexpr double kTie = 1e-9;
        if (dev < best - kTie || (dev <= best + kTie && da < best_anchor)) {
          best = dev;
          best_anchor = da;
          contact_bd_2d = p;
          chosen_vertex = index;
        }
      }
    };
    if (intersection) {
      contact_bd_2d = *intersection;
    } else {
      vertex_argmin();
    }

    support.look_at_vertex = chosen_vertex;
    if (previous_anchor_) {
      const Eigen::Vector4d bd_prev =
          boundary_inv * Eigen::Vector4d(previous_anchor_->x(), previous_anchor_->y(),
                                         previous_anchor_->z(), 1.0);
      support.look_at_boundary_crossed =
          support.boundary->IntersectsSegment(Eigen::Vector2d(bd_prev.x(), bd_prev.y()), bd_2d);
    }
    support.has_contact = true;

    // contact_point_local is documented (support.h) as relative to the
    // tree's world_transform for this element (independent of the boundary's
    // own frame above) — convert the boundary-frame contact back to world,
    // then into that tree-relative frame.
    Eigen::Vector4d contact_world =
        boundary_transform * Eigen::Vector4d(contact_bd_2d.x(),
                                             contact_bd_2d.y(), 0.0, 1.0);
    Eigen::Vector4d contact_tree_local = tree_inv * contact_world;
    support.contact_point_local =
        Eigen::Vector2d(contact_tree_local.x(), contact_tree_local.y());

    Eigen::Vector2d forward(0, 1);
    auto normalized = PolygonCoordinateMapper::Encode(
        contact_bd_2d, *support.boundary, forward);
    if (normalized.ok()) {
      support.contact_normalized = *normalized;
    }
  }

  return support;
}

}  // namespace spatialblend
