// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/spacing_grounding_rule.h"

#include <cmath>
#include <limits>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/geometry/polygon_coordinate_mapper.h"

namespace spatialblend {

std::vector<int64_t> OwnFootprints(const ElementTree& tree, const Element& referent) {
  std::vector<int64_t> own_footprints;
  const auto add_obstacle = [&](element_id_t id) {
    const Element* e = tree.GetElement(id);
    if (e && e->HasGeometryRole(GeometryRole::kObstacle)) own_footprints.push_back(id);
  };
  for (element_id_t id : tree.GetSubtreeIds(referent.GetId())) add_obstacle(id);
  for (element_id_t up = referent.GetParentId(); up != kInvalidElementId;) {
    const Element* anc = tree.GetElement(up);
    if (!anc || anc->HasGeometryRole(GeometryRole::kNavigable)) break;
    add_obstacle(up);
    up = anc->GetParentId();
  }
  return own_footprints;
}

namespace {

Polygon2D RingInFrame(const Polygon2D& ring, const Eigen::Matrix4d& to_world) {
  std::vector<Eigen::Vector2d> pts;
  pts.reserve(ring.GetPoints().size());
  for (const auto& p : ring.GetPoints()) {
    const Eigen::Vector4d w =
        to_world * Eigen::Vector4d(p.x(), p.y(), 0.0, 1.0);
    pts.emplace_back(w.x(), w.y());
  }
  return Polygon2D(pts);
}

// Outside the referent: the contact is the point on one support whose
// geodesic from the anchor is shortest. Named parts are separate supports;
// the part whose own best contact is shortest is the support, and the whole
// boundary is not added back. The length is the same geodesic the kernel
// measures.
struct GeodesicPick {
  bool ok = false;
  Eigen::Vector2d world = Eigen::Vector2d::Zero();
  Eigen::Vector2d local = Eigen::Vector2d::Zero();
  std::shared_ptr<Polygon2D> boundary;
  std::string part;
  // Length of the geodesic that selected `world`. The kernel keeps this
  // instead of planning a second field to the same contact.
  double length = -1.0;
  // Where the final straight leg onto `world` starts (world frame): the anchor
  // when it sees the contact, else the last waypoint of the geodesic.
  Eigen::Vector2d origin = Eigen::Vector2d::Zero();
};

GeodesicPick PickGeodesicContact(const SpatialDefinition& def,
                                 const Eigen::Matrix4d& lift,
                                 const Eigen::Vector2d& anchor_world,
                                 const NavigableSpatialDefinition& nav,
                                 const std::vector<int64_t>& own_footprints) {
  GeodesicPick best;
  double best_len = std::numeric_limits<double>::infinity();
  bool any_part = false;
  auto consider = [&](const Polygon2D& ring, const Eigen::Matrix4d& frame,
                      const std::shared_ptr<Polygon2D>& boundary,
                      const std::string& part) {
    if (ring.GetPoints().size() < 2 || !boundary) return;
    double len = std::numeric_limits<double>::infinity();
    const Polygon2D world = RingInFrame(ring, frame);
    Eigen::Vector2d origin = anchor_world;
    const Eigen::Vector2d c =
        nav.ClosestGeodesicOnRing(anchor_world, world, &len, &origin, own_footprints);
    const bool finite = std::isfinite(len);
    const bool best_finite = std::isfinite(best_len);
    const bool better =
        !best.ok || (finite && (!best_finite || len < best_len)) ||
        (!finite && !best_finite &&
         (c - anchor_world).squaredNorm() <
             (best.world - anchor_world).squaredNorm());
    if (!better) return;
    Eigen::Vector4d local4 =
        frame.inverse() * Eigen::Vector4d(c.x(), c.y(), 0.0, 1.0);
    Eigen::Vector2d local = local4.head<2>();
    if (boundary->GetPoints().size() >= 2)
      local = boundary->GetNearestPoint(local);
    best.ok = true;
    best.world = c;
    best.local = local;
    best.boundary = boundary;
    best.part = part;
    best.length = len;
    best.origin = origin;
    best_len = len;
  };
  for (const auto& sub : def.GetSubRegions()) {
    if (!sub.boundary || sub.boundary->GetPoints().size() < 2) continue;
    any_part = true;
    consider(*sub.boundary, lift * sub.local_transform, sub.boundary, sub.name);
  }
  if (!any_part) {
    const auto boundary = def.GetBoundary();
    if (boundary) consider(*boundary, lift, boundary, "");
  }
  return best;
}

}  // namespace

Support SpacingGroundingRule::ComputeSupport(
    const Element& referent, const ElementTree& tree,
    const Eigen::Vector3d& anchor_pos, double anchor_yaw,
    const NavigableSpatialDefinition* navigable_def) const {
  Support support;
  // Search mode: the world transform / its inverse are candidate-independent
  // (see the cache in the header); otherwise compute them per call.
  Eigen::Matrix4d to_local_cached;
  bool segment = false;  // the referent's ground image has no area (an upright polygon)
  if (skip_chart_encoding) {
    if (cache_tree_ != &tree || cache_referent_ != referent.GetId()) {
      const Eigen::Matrix4d world = tree.GetWorldTransform(referent.GetId());
      cache_to_world_ = GroundFrame(world);
      cache_segment_ = GroundImageIsSegment(world);
      cache_to_local_ = cache_to_world_.inverse();
      cache_tree_ = &tree;
      cache_referent_ = referent.GetId();
    }
    support.world_transform = cache_to_world_;
    segment = cache_segment_;
    to_local_cached = cache_to_local_;
    support.has_world_transform_inverse = true;
    support.world_transform_inverse = cache_to_local_;
  } else {
    const Eigen::Matrix4d world = tree.GetWorldTransform(referent.GetId());
    support.world_transform = GroundFrame(world);
    segment = GroundImageIsSegment(world);
  }
  auto* spatial_def = referent.GetDeclaredSpatialDefinition();
  if (spatial_def) {
    support.boundary = spatial_def->GetBoundary();
  }
  support.referent_element_id = referent.GetId();
  support.navigable_def = navigable_def;

  // Compute nearest boundary point as contact
  if (spatial_def) {
    // Some SpatialDefinition kinds (currently only PolytopeSpatialDefinition;
    // see SpatialDefinition::RequiresElementWorldTransform) store their
    // geometry verbatim in the owning ELEMENT's local frame and have no
    // internal way to compose the element's placement in the tree. For those,
    // the anchor must be transformed into the element's local frame (via the
    // tree-derived support.world_transform) before any geometry query, and
    // query results transformed back to world. Other kinds
    // (NavigableSpatialDefinition: already world-absolute;
    // SurfaceSpatialDefinition: self-contained world-in/world-out via its own
    // derived frame) are queried with the raw world anchor, unchanged.
    const bool local_frame = spatial_def->RequiresElementWorldTransform();
    const Eigen::Matrix4d to_world = support.world_transform;
    {
      const bool is_surface_def = spatial_def->IsSurfaceDefinition();
      support.boundary_lift = local_frame ? to_world
                                          : (is_surface_def ? Eigen::Matrix4d(spatial_def->GetTransform())
                                                            : Eigen::Matrix4d::Identity());
      support.has_boundary_lift = true;
    }
    const Eigen::Matrix4d to_local =
        skip_chart_encoding ? to_local_cached : to_world.inverse();

    const Eigen::Vector3d query_anchor =
        local_frame
            ? (to_local *
               Eigen::Vector4d(anchor_pos.x(), anchor_pos.y(), anchor_pos.z(),
                               1.0))
                  .head<3>()
            : anchor_pos;

    Eigen::Vector3d closest = spatial_def->GetClosestPoint(query_anchor);
    if (local_frame && support.boundary &&
        support.boundary->Contains(
            Eigen::Vector2d(query_anchor.x(), query_anchor.y()))) {
      // INSIDE a local-frame (Polytope) region the 3D closest point is the
      // nearest FACE, and an actor on the floor (z = 0) is ON the bottom face:
      // the contact would be the actor itself and the depth below the border 0.
      // The relation is planar: the contact is the nearest point of the planar
      // boundary (the same point as border_point below).
      const Eigen::Vector2d edge = support.boundary->GetNearestPoint(
          Eigen::Vector2d(query_anchor.x(), query_anchor.y()));
      closest = Eigen::Vector3d(edge.x(), edge.y(), query_anchor.z());
    }
    if (local_frame) {
      closest = (to_world * Eigen::Vector4d(closest.x(), closest.y(),
                                            closest.z(), 1.0))
                    .head<3>();
    }
    support.contact_point_world = closest;  // world-frame nearest edge point
    // Transform to referent local frame (relative to the tree's world
    // transform for this element — this is the frame contact_point_local has
    // always documented itself as being in, independent of local_frame above).
    Eigen::Vector4d local = to_local * Eigen::Vector4d(closest.x(), closest.y(),
                                                       closest.z(), 1.0);
    support.contact_point_local = Eigen::Vector2d(local.x(), local.y());
    support.has_contact = true;

    // Normalize contact onto boundary polygon.
    // The boundary polygon's coordinate frame varies by spatial-def type:
    // world frame for NavigableSpatialDef, surface-local (spatial_def's own
    // GetTransform()) for SurfaceSpatialDef, and the ELEMENT's local frame
    // (support.world_transform, since GetTransform() is a dead field) for
    // PolytopeSpatialDef. Transform the anchor into the boundary's frame
    // before finding the nearest point, so the encoding is always consistent
    // with the polygon.
    if (support.boundary) {
      Eigen::Vector2d forward(0, 1);
      Eigen::Vector2d anchor_bd_2d;
      if (local_frame) {
        anchor_bd_2d = Eigen::Vector2d(query_anchor.x(), query_anchor.y());
      } else {
        Eigen::Matrix4d boundary_inv = spatial_def->GetTransform().inverse();
        Eigen::Vector4d anchor_in_boundary =
            boundary_inv * Eigen::Vector4d(anchor_pos.x(), anchor_pos.y(),
                                           anchor_pos.z(), 1.0);
        anchor_bd_2d =
            Eigen::Vector2d(anchor_in_boundary.x(), anchor_in_boundary.y());
      }

      // When inside boundary: encode the actor's POSITION (not nearest boundary
      // point) — the position within the polygon is what transfers cross-space.
      // When outside: the contact is the shortest-geodesic point of one support
      // (a named part when the referent has parts). The chart encodes that
      // point, so the receiver measures R against the same contact.
      // A polygon whose ground image is a segment (an upright polygon, exactly or to the precision of the
      // app's float matrix; support.h GroundImageIsSegment) has no inside: the referent is the strip it
      // stands on, however its matrix was rounded. Its frame is restored to a sliver of the data's
      // precision (GroundFrame), which a point ON the strip would otherwise fall in.
      bool inside = !(local_frame && segment) && support.boundary->Contains(anchor_bd_2d);
      Eigen::Vector2d geodesic_chart = Eigen::Vector2d::Zero();
      bool have_geodesic = false;
      // The ring field is the distance, including while the chart encode is
      // skipped. The line search and the Jacobian share that one field.
      if (!inside && navigable_def && !skip_nearest_contact) {
        const bool is_surface = spatial_def->IsSurfaceDefinition();
        const Eigen::Matrix4d lift =
            local_frame ? to_world
                        : (is_surface ? Eigen::Matrix4d(spatial_def->GetTransform())
                                      : Eigen::Matrix4d::Identity());
        // An element's own footprint never blocks contact with that element
        // (v1: reaching a target is entering its affordance volume):
        // the footprints of the referent, of what it carries, and of the objects it stands in
        // (a seat part inside its chair's footprint) are the goal's own, not obstacles to it.
        // Only obstacle-role elements have a footprint in the free space (the keys are their ids).
        const std::vector<int64_t> own_footprints = OwnFootprints(tree, referent);
        const GeodesicPick selected = PickGeodesicContact(
            *spatial_def, lift,
            Eigen::Vector2d(anchor_pos.x(), anchor_pos.y()), *navigable_def, own_footprints);
        if (selected.ok) {
          have_geodesic = true;
          geodesic_chart = selected.local;
          support.contact_point_world = Eigen::Vector3d(
              selected.world.x(), selected.world.y(), closest.z());
          support.geodesic_length_m = selected.length;
          {
            // The final leg of the geodesic, in the frame the anchor offset is
            // encoded in (the referent's element frame). v1's approach direction
            // is this leg's direction.
            const Eigen::Vector4d o = to_local * Eigen::Vector4d(
                selected.origin.x(), selected.origin.y(), anchor_pos.z(), 1.0);
            support.approach_origin_local = o.head<2>();
            support.has_approach_origin = std::isfinite(selected.length);
          }
          if (!selected.part.empty()) {
            support.boundary = selected.boundary;
            support.sub_region_name = selected.part;
            support.contact_point_local = selected.local;
          } else {
            const Eigen::Vector4d el =
                to_local * Eigen::Vector4d(selected.world.x(), selected.world.y(),
                                           closest.z(), 1.0);
            support.contact_point_local = el.head<2>();
          }
        }
      }
      if (!skip_chart_encoding) {
        Eigen::Vector2d point_to_encode = inside
            ? anchor_bd_2d
            : (have_geodesic ? geodesic_chart
                             : support.boundary->GetNearestPoint(anchor_bd_2d));

        auto normalized = PolygonCoordinateMapper::Encode(
            point_to_encode, *support.boundary, forward);
        if (normalized.ok()) {
          support.contact_normalized = *normalized;
          // Receiver-convexity chart: the perimeter fraction of the boundary
          // point this position's radial ray leaves the source polygon at
          // (the contact itself for an outside source), so a receiver whose
          // own boundary is non-convex can map boundary-to-boundary (MVC on
          // the boundary) instead of radially
          // (NormalizedPolygonPosition::boundary_arclen).
          auto arclen = PolygonCoordinateMapper::EncodeRadialArcLength(
              point_to_encode, *support.boundary, forward);
          if (arclen.ok())
            support.contact_normalized.boundary_arclen = *arclen;
          // Stand-off chart: where on the offset curve of this polygon at the
          // position's own distance it lies (outside only).
          if (!inside && support.boundary->GetPoints().size() >= 3) {
            const Eigen::Vector2d nearest = point_to_encode;
            const double d = (anchor_bd_2d - nearest).norm();
            const auto hit = PolygonCoordinateMapper::ForwardHit(
                *support.boundary, forward);
            if (hit && d > 1e-9) {
              auto f = PolygonCoordinateMapper::EncodeOffsetFraction(
                  support.boundary->GetPoints(), *hit, anchor_bd_2d, d);
              if (f.ok()) support.contact_normalized.standoff_fraction = *f;
            }
            if (d > 1e-9) {
              const Eigen::Vector2d c =
                  PolygonCoordinateMapper::ComputeCentroid(*support.boundary);
              const Eigen::Vector2d r = anchor_bd_2d - c;
              if (r.norm() > 1e-12) {
                support.contact_normalized.has_standoff_angle = true;
                support.contact_normalized.standoff_angle = std::remainder(
                    std::atan2(r.y(), r.x()) -
                        std::atan2(forward.y(), forward.x()),
                    2.0 * M_PI);
              }
            }
          }
        }
      }

      // Within-element pose: when the actor is INSIDE the referent, also encode
      // its interior position via the referent's canonical chart (box-UVW / MVC,
      // type-agnostic). The receiver decodes this with the TARGET referent's own
      // chart, so the pose transfers faithfully across a differently shaped/typed
      // corresponding element (see RelationSolver::DecodeTargetPoint).
      if (inside && !skip_chart_encoding) {
        // Planar depth below the boundary edge, in the boundary frame.
        support.edge_depth_m =
            (anchor_bd_2d - support.boundary->GetNearestPoint(anchor_bd_2d))
                .norm();
      }
      if (inside) {
        // The source's border-referent coordinate: the boundary point that carries the
        // INSET coordinate of the anchor (continuous across the container's medial axis);
        // the closest boundary point where no inset edge carries it (a reflex vertex).
        const auto inset_point =
            BoundaryPointOfInset(support.boundary->GetPoints(), anchor_bd_2d);
        support.border_point =
            inset_point ? *inset_point : support.boundary->GetNearestPoint(anchor_bd_2d);
        support.has_border_point = true;
      }
      if (inside) {
        support.interior =
            spatial_def->EncodeInterior(local_frame ? query_anchor : anchor_pos);
        support.has_interior = true;
      }
    }
  }

  return support;
}

}  // namespace spatialblend
