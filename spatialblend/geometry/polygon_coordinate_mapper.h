// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_GEOMETRY_POLYGON_COORDINATE_MAPPER_H_
#define SPATIALBLEND_GEOMETRY_POLYGON_COORDINATE_MAPPER_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "Eigen/Core"
#include "absl/status/statusor.h"
#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {

// Shape-agnostic representation of a position within a polygon.
// Encodes "where" a point is relative to the polygon's shape and orientation.
// Ported from spatialblend::BaseNormalizedPolygonPosition (v1). The radial/MVC construction follows v1;
// Encode/MapPosition additionally canonicalize a bounded double-roundoff
// residue immediately outside an authored polygon boundary and bound the
// generated radial ratio to its admitted [0,1] domain.
struct NormalizedPolygonPosition {
  // Strategy used for encoding (determines decoding path)
  enum class Strategy { kRadial, kMVC };
  // Default-initialized (a default-constructed Support::contact_normalized
  // must not carry garbage scalars — e.g. through a proto round-trip of a
  // Support that was never actually grounded).
  Strategy strategy = Strategy::kRadial;

  // --- Radial fields (convex only) ---
  double angle = 0.0;         // θ from centroid, relative to forward direction
  double radial_ratio = 0.0;  // 0.0 (centroid) to 1.0 (boundary)

  // --- MVC fields (general simple polygon) ---
  std::vector<double> mvc_weights;  // size = num_samples

  // --- Boundary arc-length (the receiver-convexity chart) ---
  // Fraction of the perimeter, in [0, 1), from the forward-ray hit (the start
  // of ResampleBoundary's samples) to the boundary point where this point's
  // RADIAL RAY from the polygon centroid leaves the polygon (the point itself
  // for a boundary point), along the polygon's vertex order. -1 = not encoded
  // (a peer that predates it, or a non-spacing support). With the radial
  // ratio it is a boundary-to-boundary chart: Floater MVC weights of a point
  // on a sample edge are the two-vertex linear weights, so the MVC chart of a
  // contact point IS an arc-length-proportional map of the two boundary
  // curves, a homeomorphism for any simple polygons; interior points follow
  // the cone from the receiver's centroid (DecodeStarArcLength). The radial
  // chart alone is only that for convex pairs (and is discontinuous onto a
  // receiver that is not star-shaped about its centroid), so the strategy must
  // follow EITHER polygon (as v1's MapPosition did).
  // The receiver decodes it when its own boundary is non-convex.
  double boundary_arclen = -1.0;

  // --- Stand-off (iso-distance) arc-length ---
  // Fraction in [0, 1) of the length of the OFFSET CURVE of the source polygon
  // at the source's own stand-off distance {p : dist(p, polygon) = d}, from the
  // offset point of the forward-ray hit, along the vertex order, at which the
  // source's position lies. -1 = absent (inside the polygon, a peer that
  // predates it, or a support without a stand-off). A receiver whose own
  // polygon is non-convex puts its stand-off point at the same fraction of ITS
  // offset curve at its own stand-off: R-exact by construction (the point is
  // at distance d from the receiver) and continuous, including where the
  // offset curve of a non-convex receiver has a pocket (the stand-off point of
  // a contact behind a reflex vertex is not `contact + d * normal`).
  double standoff_fraction = -1.0;
  // Polar angle (relative to the forward direction) of the source's position
  // about the source polygon's area centroid, with its presence flag. The
  // stand-off chart of a CONVEX receiver: the radial chart's angle, so the
  // stand-off point lies on the receiver's offset curve at the same polar angle
  // about ITS centroid (a radial chart on the iso-distance curve; at distance 0
  // it is the radial chart, and the angle is preserved between differently
  // shaped convex referents as the radial chart contract requires).
  bool has_standoff_angle = false;
  double standoff_angle = 0.0;

  // --- Common ---
  double height_ratio = 0.0;  // z-axis normalization (reserved for future 3D use)
  // Radial surface-chart version (RadialSurfaceChart::kChartVersion). 0 = a
  // packet from a peer that predates chart versioning (pure radial chart).
  uint32_t chart_version = 0;
};

class PolygonCoordinateMapper {
 public:
  // Number of resample points for MVC arc-length parameterization
  static constexpr int kDefaultNumSamples = 64;

  // Encode: point within source polygon → normalized position
  // forward_direction: polygon's semantic "front" in local 2D (e.g. Y-forward)
  static absl::StatusOr<NormalizedPolygonPosition> Encode(
      const Eigen::Vector2d& point, const Polygon2D& polygon,
      const Eigen::Vector2d& forward_direction,
      int num_samples = kDefaultNumSamples);

  // Decode: normalized position → point within target polygon
  static absl::StatusOr<Eigen::Vector2d> Decode(
      const NormalizedPolygonPosition& normalized, const Polygon2D& polygon,
      const Eigen::Vector2d& forward_direction,
      int num_samples = kDefaultNumSamples);

  // Convenience: direct polygon-to-polygon mapping
  static absl::StatusOr<Eigen::Vector2d> MapPosition(
      const Eigen::Vector2d& point, const Polygon2D& source_polygon,
      const Eigen::Vector2d& source_forward, const Polygon2D& target_polygon,
      const Eigen::Vector2d& target_forward,
      int num_samples = kDefaultNumSamples);

  // Offset-curve chart (see NormalizedPolygonPosition::standoff_fraction). The
  // curve is the outer offset of `ring` at `distance`: straight offsets of the
  // edges, circular arcs about convex vertices, the intersection point at reflex
  // vertices, with every sample closer than `distance` to the ring culled (the
  // shadow of a pocket). It starts at the offset point of `start_point` (the
  // forward-ray hit, in the ring's frame). Both functions use the same curve,
  // so an identical copy decodes to the encoded position up to the polyline's
  // arc sag (~4e-5 * distance).
  // A CONVEX ring has no pocket to cull, so its curve is built in closed form
  // (the same 1-degree-chord polyline, no sampling or culling): every query is
  // O(n). Only a ring with a reflex vertex samples and culls.
  static absl::StatusOr<double> EncodeOffsetFraction(
      const std::vector<Eigen::Vector2d>& ring,
      const Eigen::Vector2d& start_point, const Eigen::Vector2d& position,
      double distance);
  static absl::StatusOr<Eigen::Vector2d> DecodeOffsetFraction(
      const std::vector<Eigen::Vector2d>& ring,
      const Eigen::Vector2d& start_point, double fraction, double distance);
  // Radial inverse for a convex ring: the point where the ray from `origin`
  // (inside the ring) along `direction` meets the offset curve of `ring` at
  // `distance`. nullopt if the ring is degenerate or the ray misses.
  static std::optional<Eigen::Vector2d> DecodeOffsetRay(
      const std::vector<Eigen::Vector2d>& ring, double distance,
      const Eigen::Vector2d& origin, const Eigen::Vector2d& direction);
  // The forward-ray hit of the polygon from its centroid (nullopt if none).
  static std::optional<Eigen::Vector2d> ForwardHit(
      const Polygon2D& polygon, const Eigen::Vector2d& forward_direction);

  // Arc-length chart (see NormalizedPolygonPosition::boundary_arclen) of the
  // boundary point where `point`'s radial ray from the centroid leaves the
  // polygon: perimeter fraction from the forward-ray hit, along the vertex
  // order. 0 for the centroid itself (no ray).
  static absl::StatusOr<double> EncodeRadialArcLength(
      const Eigen::Vector2d& point, const Polygon2D& polygon,
      const Eigen::Vector2d& forward_direction);

  // Inverse for the radial-ratio pair: centroid + ratio * (B - centroid), B the
  // boundary point at perimeter fraction `arclen` (taken mod 1) from this
  // polygon's own forward-ray hit, exactly on the polygon (not on a 64-sample
  // chord). A result outside a polygon that is not star-shaped about its
  // centroid is projected back inside. ratio 1 = the boundary point itself.
  static absl::StatusOr<Eigen::Vector2d> DecodeStarArcLength(
      double arclen, double ratio, const Polygon2D& polygon,
      const Eigen::Vector2d& forward_direction);

  // Boundary-only helpers of the two above.
  static absl::StatusOr<double> EncodeBoundaryArcLength(
      const Eigen::Vector2d& point, const Polygon2D& polygon,
      const Eigen::Vector2d& forward_direction);
  static absl::StatusOr<Eigen::Vector2d> DecodeBoundaryArcLength(
      double arclen, const Polygon2D& polygon,
      const Eigen::Vector2d& forward_direction);

  // Radial chart of any finite point about the polygon centroid. The ratio is
  // the point's distance over the centroid ray's boundary length, and is not
  // cut to the polygon: a point outside is a ratio above one. Encode() is the
  // in-polygon chart and clamps that ratio to [0, 1].
  static absl::StatusOr<NormalizedPolygonPosition> EncodeRadial(
      const Eigen::Vector2d& point, const Polygon2D& polygon,
      const Eigen::Vector2d& forward_direction);

  // Utility: compute polygon centroid (area-weighted, not vertex average)
  static Eigen::Vector2d ComputeCentroid(const Polygon2D& polygon);

  // Utility: resample polygon boundary at equal arc-length intervals
  // starting from the point where forward_direction ray from centroid hits
  // boundary
  static std::vector<Eigen::Vector2d> ResampleBoundary(
      const Polygon2D& polygon, const Eigen::Vector2d& forward_direction,
      int num_samples);

  // Utility: compute Mean Value Coordinate weights
  static std::vector<double> ComputeMVCWeights(
      const Eigen::Vector2d& point,
      const std::vector<Eigen::Vector2d>& polygon_vertices);

 private:
  // Retains an in-domain point, or snaps only a scale-bounded floating-point
  // residue to its nearest boundary point. It never broadens Polygon2D::Contains.
  static absl::StatusOr<Eigen::Vector2d> CanonicalizeDomainPoint(
      const Eigen::Vector2d& point, const Polygon2D& polygon);

  // Radial decode (convex polygons). EncodeRadial is public: its ratio is not
  // clamped to the polygon.
  static absl::StatusOr<Eigen::Vector2d> DecodeRadial(
      const NormalizedPolygonPosition& normalized, const Polygon2D& polygon,
      const Eigen::Vector2d& forward_direction);

  // MVC encode/decode (general simple polygons)
  static absl::StatusOr<NormalizedPolygonPosition> EncodeMVC(
      const Eigen::Vector2d& point, const Polygon2D& polygon,
      const Eigen::Vector2d& forward_direction, int num_samples);

  static absl::StatusOr<Eigen::Vector2d> DecodeMVC(
      const NormalizedPolygonPosition& normalized, const Polygon2D& polygon,
      const Eigen::Vector2d& forward_direction, int num_samples);
};


// The inset (mitered) ring of `ring` at depth `h`: each edge offset inward by h, adjacent
// offset edges meeting at the mitered vertex (the vertex moved by
// h * (n_prev + n_next) / (1 + n_prev . n_next), at distance h from both edges). `ring`
// may be either orientation.
std::vector<Eigen::Vector2d> InsetRing(const std::vector<Eigen::Vector2d>& ring, double h);

// The boundary point that carries the INSET coordinate of an interior point `p`: the point
// of the boundary at the same fraction of its edge k as p is of the inset edge k at p's
// depth h = dist(p, boundary). Unlike the foot of the perpendicular (the closest boundary
// point) it is continuous where the nearest edge changes (the medial axis: p is then the
// mitered vertex, which is the boundary vertex from both sides). Decoding is the inverse:
// the point of the inset ring at that fraction of that edge. nullopt when the closest
// feature of p is a reflex vertex (no inset edge carries p) or the ring is degenerate.
std::optional<Eigen::Vector2d> BoundaryPointOfInset(const std::vector<Eigen::Vector2d>& ring,
                                                    const Eigen::Vector2d& p);

}  // namespace spatialblend

#endif  // SPATIALBLEND_GEOMETRY_POLYGON_COORDINATE_MAPPER_H_
