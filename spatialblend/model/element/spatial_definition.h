// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_ELEMENT_SPATIAL_DEFINITION_H_
#define SPATIALBLEND_ELEMENT_SPATIAL_DEFINITION_H_

#include <Eigen/Dense>
#include <memory>

#include "spatialblend/model/element/geometry_role.h"
#include "spatialblend/geometry/normalized_interior.h"
#include <string>
#include <utility>
#include <vector>

#include "spatialblend/model/element/geometry_role.h"
#include "spatialblend/model/element/sub_region.h"
#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {

struct Bound2D {
  Eigen::Vector2d min = Eigen::Vector2d::Zero();
  Eigen::Vector2d max = Eigen::Vector2d::Zero();
};

class SpatialDefinition {
 public:
  explicit SpatialDefinition(
      GeometryRole geometry_role = GeometryRole::kUnspecified)
      : geometry_role_(geometry_role) {}
  virtual ~SpatialDefinition() = default;

  GeometryRole GetGeometryRole() const { return geometry_role_; }
  void SetGeometryRole(GeometryRole role) { geometry_role_ = role; }

  // --- Core interface (must override) ---

  virtual bool Contains(const Eigen::Vector2d& point) const = 0;

  virtual std::shared_ptr<spatialblend::Polygon2D> GetBoundary() const = 0;

  virtual const Eigen::Matrix4d& GetLocalTransform() const = 0;

  virtual Bound2D GetBounds() const = 0;

  // --- Geometric queries (with default fallbacks) ---

  // 3D containment check. Default delegates to 2D Contains.
  virtual bool Contains3D(const Eigen::Vector3d& point) const {
    return Contains(point.head<2>());
  }

  // Closest point on this geometry to the given point.
  virtual Eigen::Vector3d GetClosestPoint(const Eigen::Vector3d& from) const {
    return from;
  }

  // Distance from a point to the nearest surface of this geometry.
  virtual double GetDistance(const Eigen::Vector3d& from) const {
    return 0.0;
  }

  // Project a point onto this geometry's surface.
  virtual Eigen::Vector3d ProjectOnto(const Eigen::Vector3d& point) const {
    return point;
  }

  // Surface normal at the given point.
  // Default: Z-up for 2D floor-plane geometries.
  virtual Eigen::Vector3d GetNormal(
      const Eigen::Vector3d& at_point = Eigen::Vector3d::Zero()) const {
    return Eigen::Vector3d(0, 0, 1);
  }

  // World transform of this spatial definition.
  virtual Eigen::Matrix4d GetTransform() const { return GetLocalTransform(); }

  // Whether this definition's geometry (vertices/boundary, as consumed by
  // Contains/Contains3D/GetClosestPoint/ProjectOnto/GetBoundary) is stored
  // verbatim in the owning ELEMENT's local frame, with NO internal mechanism
  // to compose the element's placement in the ElementTree.
  //
  // Default false: NavigableSpatialDefinition stores world-absolute geometry
  // already (GetTransform() == Identity correctly means "no further
  // composition needed"); SurfaceSpatialDefinition derives its own
  // self-consistent world frame from its (also world-absolute) vertices via
  // GetTransform() == surface's local frame, and its queries are already
  // world-in/world-out. PolytopeSpatialDefinition overrides this to true: its
  // vertices are stored exactly as constructed (e.g. LOCAL half-extents
  // around the element origin, per the `sb_set_element_collider` /
  // `sb_set_element_box` C API contract) and it has no reference to the
  // owning Element or ElementTree, so `GetTransform()` can never reflect the
  // element's real placement (its `local_transform_` field is permanently
  // Identity). Callers that have tree access (GroundingRule::ComputeSupport)
  // MUST check this flag and, when true, transform the anchor into the
  // element's LOCAL frame (via the tree's world transform for this element)
  // before calling any geometry query on this definition, then transform
  // results back to world.
  virtual bool RequiresElementWorldTransform() const { return false; }

  // Whether this is a SurfaceSpatialDefinition (a tilted surface whose polygon is stored in its own derived frame). A virtual
  // tag instead of a dynamic_cast: the cast crosses a shared-library boundary (type_info comparison by name) and a grounding
  // rule asks on every evaluation of every relation.
  virtual bool IsSurfaceDefinition() const { return false; }

  // Geodesic distance between two points within this geometry.
  // For non-convex navigable areas, follows shortest path around obstacles.
  // Default: Euclidean distance.
  virtual double GetGeodesicDistance(const Eigen::Vector3d& from,
                                     const Eigen::Vector3d& to) const {
    return (from - to).norm();
  }

  // --- Canonical interior chart (type-agnostic correspondence) -------------
  // Encode a world point into a canonical interior coordinate; DecodeInterior
  // reverses it on a (possibly DIFFERENT) spatial definition. This is the single
  // mechanism by which interior pose transfers between ANY two spatial
  // definitions: the source encodes with its chart, only the canonical
  // coordinate crosses, the target decodes with its chart. See NormalizedInterior.
  //
  // Default chart = forward-aligned normalized bounding frame (box-UVW) in this
  // definition's LOCAL frame (+Y = forward), exact for affine-similar boxes.
  // Types whose interior is not box-like override these for faithfulness (e.g.
  // a non-convex navigable area uses Mean Value Coordinates) but ALWAYS also
  // fill uvw, so a cross-kind partner can fall back to the universal frame.
  virtual NormalizedInterior EncodeInterior(
      const Eigen::Vector3d& world_point) const {
    return EncodeBoxUVW(world_point);
  }
  virtual Eigen::Vector3d DecodeInterior(const NormalizedInterior& c) const {
    return DecodeBoxUVW(c);
  }

 protected:
  // Forward-aligned box chart in this definition's local frame. Usable by any
  // type as the universal default and as the cross-kind fallback.
  NormalizedInterior EncodeBoxUVW(const Eigen::Vector3d& world_point) const {
    Eigen::Vector3d local =
        (GetTransform().inverse() * world_point.homogeneous()).head<3>();
    Eigen::Vector2d center, half;
    LocalBox(&center, &half);
    NormalizedInterior n;
    n.kind = NormalizedInterior::Kind::kBoxUVW;
    n.uvw.x() = half.x() > 1e-9 ? (local.x() - center.x()) / half.x() : 0.0;
    n.uvw.y() = half.y() > 1e-9 ? (local.y() - center.y()) / half.y() : 0.0;
    n.uvw.z() = 0.0;  // planar; volume types override
    return n;
  }
  Eigen::Vector3d DecodeBoxUVW(const NormalizedInterior& c) const {
    Eigen::Vector2d center, half;
    LocalBox(&center, &half);
    Eigen::Vector3d local(center.x() + c.uvw.x() * half.x(),
                          center.y() + c.uvw.y() * half.y(), 0.0);
    return (GetTransform() * local.homogeneous()).head<3>();
  }

 public:
  // --- Sub-regions (named affordance patches) ---
  void AddSubRegion(SubRegion region) {
    sub_regions_.push_back(std::move(region));
  }
  const SubRegion* GetSubRegion(const std::string& name) const {
    for (const auto& r : sub_regions_) {
      if (r.name == name) return &r;
    }
    return nullptr;
  }
  const std::vector<SubRegion>& GetSubRegions() const { return sub_regions_; }
  // Sets the surface chart of the first sub-region of that name. False for an unknown name.
  bool SetSubRegionChart(const std::string& name, SurfaceChartKind chart) {
    for (auto& r : sub_regions_) {
      if (r.name == name) {
        r.chart = chart;
        return true;
      }
    }
    return false;
  }

 private:
  // Center + half-extent of this definition's boundary in its LOCAL frame.
  void LocalBox(Eigen::Vector2d* center, Eigen::Vector2d* half) const {
    Eigen::Vector2d mn(1e30, 1e30), mx(-1e30, -1e30);
    if (auto poly = GetBoundary()) {
      for (const auto& p : poly->GetPoints()) {
        mn = mn.cwiseMin(p);
        mx = mx.cwiseMax(p);
      }
    }
    if (mx.x() < mn.x()) {  // no boundary polygon: fall back to axis bounds
      Bound2D b = GetBounds();
      mn = b.min;
      mx = b.max;
    }
    *center = 0.5 * (mn + mx);
    *half = 0.5 * (mx - mn);
  }

  GeometryRole geometry_role_;
  std::vector<SubRegion> sub_regions_;
};

// Map a world point inside `source` to the corresponding interior point of
// `target`, through the canonical chart. The source encodes; the target
// decodes with its own chart. When both share a rich Kind the faithful channel
// carries through, else the universal uvw frame is used (negotiated inside the
// target's DecodeInterior). Works for ANY pair of spatial-definition types.
inline Eigen::Vector3d MapInterior(const SpatialDefinition& source,
                                   const SpatialDefinition& target,
                                   const Eigen::Vector3d& world_point) {
  return target.DecodeInterior(source.EncodeInterior(world_point));
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_ELEMENT_SPATIAL_DEFINITION_H_
