// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_RELATION_SUPPORT_H_
#define SPATIALBLEND_RELATION_SUPPORT_H_

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <string>

#include "spatialblend/base/common/types.h"
#include "spatialblend/geometry/normalized_interior.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/geometry/polygon_coordinate_mapper.h"

namespace spatialblend {

class NavigableSpatialDefinition;  // forward declaration

// The polygon's ground image is a SEGMENT (zero area): the polygon stands upright, so projecting it onto
// the floor collapses one of its two axes. Decided on the WORLD transform (not on the ground frame, which
// has already lost the dropped axis), as the ratio of the image's area to the area the polygon would have
// on the floor if it lay flat: |det of the 2x2 ground part| / (|column x| |column y|) of the 3D columns,
// which is |cos(tilt)|. Such a polygon is NEVER inside (a segment has no inside).
//
// The test is a RATIO with ONE tolerance, the precision of the data and not a tuning knob: the apps hand
// the core 4x4 matrices of single-precision floats (Unity's Rx(-90) arrives with cos = -4.4e-8, a
// double-precision build of it with 6.1e-17), so "upright" is |cos(tilt)| within a few float epsilons
// (1.2e-7). Anything above is a polygon that really lies on the floor, at however shallow an angle. An
// exact comparison with 0 made the same plane a point at its origin when built exactly and a full strip
// support when built with that float noise: the referent changed with the rounding of the app's matrix.
constexpr double kUprightCosTolerance = 1e-6;
inline bool GroundImageIsSegment(const Eigen::Matrix4d& world) {
  const double det2 = world(0, 0) * world(1, 1) - world(0, 1) * world(1, 0);
  const double scale = world.col(0).head<3>().norm() * world.col(1).head<3>().norm();
  return !(scale > 0.0) || std::fabs(det2) <= kUprightCosTolerance * scale;
}

// The ground frame of an element-local polygon: THE definition of where its polygon is on the floor (the
// Space is 2D). The polygon's local x,y are placed by the 2D affine part of the element's world transform,
// with the vertical axis kept vertical. For every placement that only turns about the vertical axis (a
// yaw, a translation, a height) this is the world transform itself, exactly. For a placement that tips the
// polygon out of the ground plane, the full 3D inverse reads a point through its projection ALONG the
// polygon's normal (an upright polygon would "hold" the strip it stands on and every encoded position
// would lose its lateral part); the ground frame reads the polygon's image on the floor instead, the same
// image the decode lifts (world * (x, y, 0, 1), x and y). The footprint test (FootprintHolds), the
// Spacing support encode and the Spacing kernel all use this one frame.
//
// A polygon whose image is a segment (GroundImageIsSegment) has a frame that cannot be inverted. Its frame
// is the same for every rounding of the app's matrix: the image's one surviving axis exactly, the
// collapsed one restored to the precision of the data (the tolerance, relative to the surviving axis), so
// the referent is the strip it stands on, a segment to that precision, however its matrix was rounded.
inline Eigen::Matrix4d GroundFrame(const Eigen::Matrix4d& world) {
  Eigen::Matrix4d ground = world;
  if (!(world(0, 2) == 0.0 && world(1, 2) == 0.0 && world(2, 0) == 0.0 && world(2, 1) == 0.0)) {
    ground = Eigen::Matrix4d::Identity();
    ground.topLeftCorner<2, 2>() = world.topLeftCorner<2, 2>();
    ground.block<3, 1>(0, 3) = world.block<3, 1>(0, 3);
  }
  if (GroundImageIsSegment(world)) {
    const double scale = world.col(0).head<3>().norm() * world.col(1).head<3>().norm();
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(ground.topLeftCorner<2, 2>(), Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector2d sigma = svd.singularValues();
    if (sigma(0) > 0.0) {
      const Eigen::Vector2d restored(sigma(0), kUprightCosTolerance * scale / sigma(0));
      ground.topLeftCorner<2, 2>() = svd.matrixU() * restored.asDiagonal() * svd.matrixV().transpose();
    }
  }
  return ground;
}

struct Support {
  Eigen::Matrix4d world_transform = Eigen::Matrix4d::Identity();
  // The referent's DECLARED shape (Element::GetDeclaredSpatialDefinition): support, contact and
  // chart are the object's own and never see the scope its children extend.
  std::shared_ptr<spatialblend::Polygon2D> boundary;
  // The lift of `boundary` (in its own frame) to the world: the identity for a Navigable region, the definition's transform
  // for a surface, the element's world transform for an element-local polytope. The contact pieces of a border relation
  // (the edges and vertices of the polygon) are listed in the world through it.
  Eigen::Matrix4d boundary_lift = Eigen::Matrix4d::Identity();
  bool has_boundary_lift = false;
  element_id_t referent_element_id = kInvalidElementId;

  // Optional: room-level navigable geometry for geodesic distance.
  // Set by GroundingRule when available. Kernels use this for
  // path-aware spacing instead of Euclidean.
  const NavigableSpatialDefinition* navigable_def = nullptr;
  // Element that owns navigable_def (the normalizing scope), when the caller
  // knows it. Source-side bookkeeping for the normalizing-scope transfer.
  element_id_t navigable_element_id = kInvalidElementId;

  // Contact point on the referent boundary, in referent local frame.
  // Computed by the GroundingRule based on the actor's position/heading.
  Eigen::Vector2d contact_point_local = Eigen::Vector2d::Zero();

  // Same contact point in WORLD frame. Kernels use this as the geodesic-distance
  // target (nearest point on the referent's edge) so spacing reflects path-aware
  // reachability to the object's boundary, matching v1 (geodesic-to-closest-edge).
  Eigen::Vector3d contact_point_world = Eigen::Vector3d::Zero();

  // Geodesic length already measured to contact_point_world by the grounding
  // rule's per-referent ring field. Negative means the kernel must measure.
  // The line search and the Jacobian share this value, and the field itself
  // stays cached on the free space across Gauss-Newton iterations.
  double geodesic_length_m = -1.0;

  // Polygon-mapped normalized position of the contact point.
  // Enables cross-space transfer: decode onto target boundary.
  // Uses spatialblend::NormalizedPolygonPosition (PolygonCoordinateMapper).
  spatialblend::NormalizedPolygonPosition contact_normalized;

  // Whether contact was computed by the GroundingRule.
  bool has_contact = false;

  // Canonical interior coordinate of the actor WITHIN this referent, via the
  // referent's SpatialDefinition chart (box-UVW / MVC, type-agnostic). Set only
  // when the actor is inside the referent (containment / within-element pose).
  // The receiver reconstructs the support point with the TARGET referent's chart
  // (see RelationSolver::DecodeTargetPoint), so within-element pose transfers
  // faithfully across differently shaped/typed corresponding elements.
  NormalizedInterior interior;
  bool has_interior = false;
  // The referent is the CONTAINER the actor is inside (its deepest containing
  // scope), kept as a Spacing referent next to the container's children. Inside,
  // the relation is the depth below the border, R = 1 − depth/D. Outside, the
  // body is outside this scope and the relation is evaluated at the projection
  // onto the border, so the depth is 0 and R = 1. R does not exceed 1.
  bool container_self = false;
  // Facing: the index of the boundary vertex the look-at contact was chosen at, in the
  // referent's ring, when the gaze misses the referent and the contact is the silhouette
  // vertex of least angle; -1 when the gaze hits it (the contact slides along the boundary,
  // continuous) or the relation is not a Facing. The vertex minimiser is discontinuous at the
  // one heading where two vertices tie, so a change of this index between two updates is the
  // definition's own discontinuity.
  int32_t look_at_vertex = -1;
  // Facing: the anchor's path since the source's previous evaluation (previous position to
  // this one) crossed the referent's boundary. The look-at contact is the gaze's exit point
  // from inside the referent and its entry point or silhouette vertex from outside, so it
  // jumps wherever the anchor passes through the boundary, even when it enters and leaves
  // between two updates. False on the first evaluation.
  bool look_at_boundary_crossed = false;
  // Planar distance (boundary frame, the source's own units) from the actor to
  // the nearest EDGE of the referent's boundary polygon, set with `interior`.
  // -1 = unknown (a sender that predates it, or not inside). The receiver's
  // chart blend near the edge reads it: `contact_point_local` is the closest
  // point of the referent as a SOLID, which for a raised box (Polytope over
  // z 0.7..0.8) is the actor's own XY on the lower face, i.e. distance 0.
  double edge_depth_m = -1.0;
  // The closest point of the referent's boundary polygon to the actor, in the
  // BOUNDARY'S OWN frame (the frame `boundary` is stored in), set when the actor is
  // inside. It is the border point of a container-self relation: the receiver maps
  // it through the support chart and offsets it inward by the depth. Carried
  // explicitly so the decode never has to guess which frame a contact point is in.
  Eigen::Vector2d border_point = Eigen::Vector2d::Zero();
  bool has_border_point = false;
  // Named sub-region this support is grounded on. Empty = whole boundary.
  std::string sub_region_name;
  // The chart kind of that sub-region on the source (SubRegion::chart, as int). 0 = radial = every sender that predates it.
  int32_t sub_region_chart = 0;
  // Where the final straight leg of the actor's geodesic onto the contact starts,
  // in the frame the anchor offset (encoded_state) is in. It is the anchor itself
  // when the actor sees the contact, else the last waypoint of the geodesic. The
  // receiver reads it through the support chart as the start of its own final
  // leg: v1's approach direction (Approach::GetDirection2D) is the direction of
  // that leg. Absent: the anchor offset is the start.
  Eigen::Vector2d approach_origin_local = Eigen::Vector2d::Zero();
  bool has_approach_origin = false;
  // AffordancePose: the anchor lies inside the support part's footprint (its
  // projection onto the part's plane is inside the part's polygon), so the nearest
  // support point is the anchor's own projection and its in-plane offset is exactly
  // 0. Decided where the part's frame is exact (AffordancePoseGroundingRule), not
  // by comparing two round-tripped coordinates. Contact needs this and the anchor
  // on or below the plane (engagement.h IsAffordanceContact).
  bool anchor_over_footprint = false;

  // Performance: when true, a kernel that would otherwise
  // call NavigableSpatialDefinition::GetGeodesicDistance (a visibility-graph
  // pathfind, expensive when the direct line is obstacle-blocked and the
  // dominant per-candidate cost measured in the paper-objective search --
  // see relation_solver.cc's evaluate_position) uses straight-line distance
  // to the same contact point instead. This is a SEARCH-ONLY approximation
  // for ranking candidates cheaply during the bounded local search's descent
  // steps; it must never be used for the value that decides acceptance (see
  // the exact re-evaluation relation_solver.cc does once per restart before
  // comparing against the running best) or for anything emitted/observed.
  // Default false preserves exact (geodesic) behavior for every existing
  // caller (EvaluateTargetRelation, tests, tracing).
  bool fast_distance = false;

  // Performance hint (search mode, see GroundingRule::skip_chart_encoding):
  // when set, world_transform_inverse IS world_transform.inverse(), so a
  // kernel that needs the inverse (AvatarSpacingKernel::Encode) does not
  // recompute a general 4x4 inverse on every one of thousands of candidate
  // evaluations. Left false by every other caller.
  bool has_world_transform_inverse = false;
  Eigen::Matrix4d world_transform_inverse = Eigen::Matrix4d::Identity();

  Eigen::Vector2d GetPosition2D() const {
    return Eigen::Vector2d(world_transform(0, 3), world_transform(1, 3));
  }

  Eigen::Vector3d GetPosition3D() const {
    return Eigen::Vector3d(world_transform(0, 3), world_transform(1, 3),
                           world_transform(2, 3));
  }

  Eigen::Vector2d GetForwardDirection2D() const {
    return Eigen::Vector2d(world_transform(0, 1), world_transform(1, 1));
  }

  double GetYaw() const {
    return std::atan2(world_transform(1, 0), world_transform(0, 0));
  }
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_RELATION_SUPPORT_H_
