// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/polygon2d.cc
#include "spatialblend/geometry/polygon2d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/segment.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace spatialblend {

namespace bg = boost::geometry;
using bg_point = bg::model::d2::point_xy<double>;
using bg_polygon = bg::model::polygon<bg_point>;
using bg_segment = bg::model::segment<bg_point>;
using bg_box = bg::model::box<bg_point>;

namespace bgi = bg::index;
using edge_rtree_value = std::pair<bg_box, size_t>;
using edge_rtree = bgi::rtree<edge_rtree_value, bgi::quadratic<16>>;

struct Polygon2D::Impl {
  bg_polygon bg_poly;
  // Further components of a buffer that split. The largest ring lives in
  // bg_poly; these are the rest, in the same order as Polygon2D::extra_rings_.
  std::vector<bg_polygon> extra;
};

// Builds a Polygon2D (outer rings, largest first) from a Boost result.
struct Polygon2DAccess {
  static Polygon2D FromMultiPolygon(bg::model::multi_polygon<bg_polygon> result,
                                    bool fill_holes);
};

struct Polygon2D::SpatialIndex {
  edge_rtree rtree;
  std::vector<internal::EdgeData> edges;
};

namespace {
bg_point ToBg(const Eigen::Vector2d& p) {
  return bg_point(p.x(), p.y());
}

Eigen::Vector2d FromBg(const bg_point& p) {
  return Eigen::Vector2d(bg::get<0>(p), bg::get<1>(p));
}

// Closest point on segment ab to point p
Eigen::Vector2d ClosestPointOnSegment(const Eigen::Vector2d& a,
                                       const Eigen::Vector2d& b,
                                       const Eigen::Vector2d& p) {
  Eigen::Vector2d ab = b - a;
  double len_sq = ab.squaredNorm();
  if (len_sq < 1e-20) return a;
  double t = (p - a).dot(ab) / len_sq;
  t = std::max(0.0, std::min(1.0, t));
  return a + t * ab;
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructors / copy / move
// ---------------------------------------------------------------------------

Polygon2D::Polygon2D(const std::vector<Eigen::Vector2d>& points)
    : points_(points), impl_(std::make_shared<Impl>()) {
  for (const auto& p : points) {
    bg::append(impl_->bg_poly.outer(), ToBg(p));
  }
  bg::correct(impl_->bg_poly);
}

Polygon2D::Polygon2D(const Polygon2D& other)
    : points_(other.points_),
      extra_rings_(other.extra_rings_),
      impl_(other.impl_) {
  // Spatial index is NOT copied — it will be lazily rebuilt.
}

Polygon2D::Polygon2D(Polygon2D&& other) noexcept
    : points_(std::move(other.points_)),
      extra_rings_(std::move(other.extra_rings_)),
      impl_(std::move(other.impl_)) {
  // Spatial index is NOT moved — it will be lazily rebuilt.
}

Polygon2D& Polygon2D::operator=(const Polygon2D& other) {
  if (this != &other) {
    points_ = other.points_;
    extra_rings_ = other.extra_rings_;
    impl_ = other.impl_;
    spatial_index_.reset();
  }
  return *this;
}

Polygon2D& Polygon2D::operator=(Polygon2D&& other) noexcept {
  if (this != &other) {
    points_ = std::move(other.points_);
    extra_rings_ = std::move(other.extra_rings_);
    impl_ = std::move(other.impl_);
    spatial_index_.reset();
  }
  return *this;
}

// ---------------------------------------------------------------------------
// Spatial index
// ---------------------------------------------------------------------------

void Polygon2D::EnsureEdgeIndex() const {
  // Double-checked locking for thread-safe lazy initialization
  if (spatial_index_) return;

  std::lock_guard<std::mutex> lock(index_mutex_);
  if (spatial_index_) return;

  auto index = std::make_shared<SpatialIndex>();
  auto index_ring = [&](const std::vector<Eigen::Vector2d>& ring, size_t ring_index,
                        std::vector<edge_rtree_value>* rtree_values) {
    if (ring.size() < 2) return;
    for (size_t i = 0; i < ring.size(); ++i) {
      size_t j = (i + 1) % ring.size();
      const Eigen::Vector2d& p1 = ring[i];
      const Eigen::Vector2d& p2 = ring[j];

      // Skip degenerate edges
      if ((p2 - p1).squaredNorm() < 1e-20) continue;

      size_t edge_idx = index->edges.size();
      index->edges.push_back(
          {p1, p2, ring_index, /*edge_index_in_ring=*/i});

      // Create bounding box for this edge
      double min_x = std::min(p1.x(), p2.x());
      double min_y = std::min(p1.y(), p2.y());
      double max_x = std::max(p1.x(), p2.x());
      double max_y = std::max(p1.y(), p2.y());

      // Add small epsilon to handle axis-aligned edges
      constexpr double kEps = 1e-10;
      if (max_x - min_x < kEps) {
        min_x -= kEps;
        max_x += kEps;
      }
      if (max_y - min_y < kEps) {
        min_y -= kEps;
        max_y += kEps;
      }

      bg_box box(bg_point(min_x, min_y), bg_point(max_x, max_y));
      rtree_values->emplace_back(box, edge_idx);
    }
  };
  if (points_.size() >= 2 || !extra_rings_.empty()) {
    std::vector<edge_rtree_value> rtree_values;
    rtree_values.reserve(points_.size());
    index_ring(points_, 0, &rtree_values);
    for (size_t r = 0; r < extra_rings_.size(); ++r)
      index_ring(extra_rings_[r], r + 1, &rtree_values);

    // Bulk-load R-tree for optimal construction performance
    index->rtree = edge_rtree(rtree_values.begin(), rtree_values.end());
  }

  spatial_index_ = std::move(index);
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

bool Polygon2D::Contains(const Eigen::Vector2d& point,
                          bool include_boundary) const {
  if (!impl_ || (points_.size() < 3 && extra_rings_.empty())) return false;
  bg_point p = ToBg(point);
  auto inside = [&](const bg_polygon& poly) {
    if (include_boundary)
      return bg::within(p, poly) || bg::touches(p, poly);
    return bg::within(p, poly);
  };
  if (points_.size() >= 3 && inside(impl_->bg_poly)) return true;
  for (const auto& extra : impl_->extra)
    if (inside(extra)) return true;
  return false;
}

Eigen::Vector2d Polygon2D::GetNearestPoint(
    const Eigen::Vector2d& point) const {
  if (points_.empty()) return point;
  if (points_.size() == 1) return points_[0];
  if (points_.size() == 2) {
    return ClosestPointOnSegment(points_[0], points_[1], point);
  }

  EnsureEdgeIndex();
  const auto& idx = *spatial_index_;

  if (idx.edges.empty()) return points_[0];

  auto compute_nearest_on_edge =
      [&point](const internal::EdgeData& edge)
      -> std::pair<Eigen::Vector2d, double> {
    Eigen::Vector2d edge_vec = edge.p2 - edge.p1;
    double edge_length_sq = edge_vec.squaredNorm();
    if (edge_length_sq < 1e-20) {
      return {edge.p1, (point - edge.p1).squaredNorm()};
    }
    double t = (point - edge.p1).dot(edge_vec) / edge_length_sq;
    t = std::max(0.0, std::min(1.0, t));
    Eigen::Vector2d projection = edge.p1 + t * edge_vec;
    return {projection, (point - projection).squaredNorm()};
  };

  constexpr size_t kInitialCandidates = 8;
  bg_point query_pt(point.x(), point.y());

  std::vector<edge_rtree_value> candidates;
  candidates.reserve(kInitialCandidates);

  size_t num_candidates = std::min(kInitialCandidates, idx.edges.size());
  idx.rtree.query(
      bgi::nearest(query_pt, static_cast<unsigned>(num_candidates)),
      std::back_inserter(candidates));

  double min_distance_sq = std::numeric_limits<double>::max();
  Eigen::Vector2d nearest_point = idx.edges[0].p1;

  for (const auto& candidate : candidates) {
    size_t edge_idx = candidate.second;
    auto [proj, dist_sq] = compute_nearest_on_edge(idx.edges[edge_idx]);
    if (dist_sq < min_distance_sq) {
      min_distance_sq = dist_sq;
      nearest_point = proj;
    }
  }

  if (idx.edges.size() > kInitialCandidates && min_distance_sq > 0) {
    double search_radius = std::sqrt(min_distance_sq) * 1.01;

    bg_box search_box(
        bg_point(point.x() - search_radius, point.y() - search_radius),
        bg_point(point.x() + search_radius, point.y() + search_radius));

    std::vector<edge_rtree_value> nearby_edges;
    idx.rtree.query(bgi::intersects(search_box),
                    std::back_inserter(nearby_edges));

    for (const auto& nearby : nearby_edges) {
      size_t edge_idx = nearby.second;
      bool already_checked = false;
      for (const auto& c : candidates) {
        if (c.second == edge_idx) {
          already_checked = true;
          break;
        }
      }
      if (already_checked) continue;

      auto [proj, dist_sq] = compute_nearest_on_edge(idx.edges[edge_idx]);
      if (dist_sq < min_distance_sq) {
        min_distance_sq = dist_sq;
        nearest_point = proj;
      }
    }
  }

  return nearest_point;
}

double Polygon2D::GetDistance(const Eigen::Vector2d& point) const {
  if (points_.size() < 3) return 0.0;
  auto nearest = GetNearestPoint(point);
  return (nearest - point).norm();
}

bool Polygon2D::IntersectsSegment(const Eigen::Vector2d& p1,
                                   const Eigen::Vector2d& p2) const {
  if (!impl_ || points_.size() < 3) return false;

  EnsureEdgeIndex();
  const auto& idx = *spatial_index_;

  if (idx.edges.empty()) return false;

  // Create bounding box from the query segment
  double min_x = std::min(p1.x(), p2.x());
  double min_y = std::min(p1.y(), p2.y());
  double max_x = std::max(p1.x(), p2.x());
  double max_y = std::max(p1.y(), p2.y());

  constexpr double kMargin = 1e-10;
  bg_box segment_box(bg_point(min_x - kMargin, min_y - kMargin),
                      bg_point(max_x + kMargin, max_y + kMargin));

  // Query R-tree for candidate edges whose bounding boxes overlap
  std::vector<edge_rtree_value> candidates;
  idx.rtree.query(bgi::intersects(segment_box),
                  std::back_inserter(candidates));

  // Test only candidate edges for intersection
  bg_segment seg(ToBg(p1), ToBg(p2));
  for (const auto& candidate : candidates) {
    const auto& edge = idx.edges[candidate.second];
    bg_segment edge_seg(ToBg(edge.p1), ToBg(edge.p2));
    if (bg::intersects(seg, edge_seg)) return true;
  }
  return false;
}

const std::vector<Eigen::Vector2d>& Polygon2D::GetPoints() const {
  return points_;
}

Eigen::Vector2d Polygon2D::GetCenter() const {
  if (points_.empty()) return Eigen::Vector2d::Zero();
  if (!impl_) return Eigen::Vector2d::Zero();

  bg_point centroid;
  bg::centroid(impl_->bg_poly, centroid);
  return FromBg(centroid);
}

double Polygon2D::GetArea() const {
  if (!impl_ || (points_.size() < 3 && extra_rings_.empty())) return 0.0;
  double area = points_.size() >= 3 ? std::abs(bg::area(impl_->bg_poly)) : 0.0;
  for (const auto& extra : impl_->extra) area += std::abs(bg::area(extra));
  return area;
}

size_t Polygon2D::NumPoints() const {
  return points_.size();
}

size_t Polygon2D::NumComponents() const {
  if (!impl_ || (points_.size() < 3 && extra_rings_.empty())) return 0;
  return (points_.size() >= 3 ? 1 : 0) + extra_rings_.size();
}

bool Polygon2D::IsConvex() const {
  if (!impl_ || points_.size() < 3 || !extra_rings_.empty()) return false;
  return bg::is_convex(impl_->bg_poly);
}

std::optional<Eigen::Vector2d> Polygon2D::RayBoundaryIntersection(
    const Eigen::Vector2d& origin,
    const Eigen::Vector2d& direction) const {
  if (points_.size() < 3) return std::nullopt;

  EnsureEdgeIndex();
  const auto& idx = *spatial_index_;

  if (idx.edges.empty()) return std::nullopt;

  // Compute search box that covers the polygon bounds and extends along the
  // ray. The box must span from the ray origin through the entire polygon to
  // catch all candidate edges, even when origin is far from the polygon.
  double poly_min_x = points_[0].x(), poly_max_x = poly_min_x;
  double poly_min_y = points_[0].y(), poly_max_y = poly_min_y;
  auto grow = [&](const Eigen::Vector2d& p) {
    poly_min_x = std::min(poly_min_x, p.x());
    poly_max_x = std::max(poly_max_x, p.x());
    poly_min_y = std::min(poly_min_y, p.y());
    poly_max_y = std::max(poly_max_y, p.y());
  };
  for (const auto& p : points_) grow(p);
  for (const auto& ring : extra_rings_)
    for (const auto& p : ring) grow(p);

  // Search box: union of origin point and polygon bounds, with margin
  constexpr double kMargin = 1e-6;
  double box_min_x = std::min(origin.x(), poly_min_x) - kMargin;
  double box_max_x = std::max(origin.x(), poly_max_x) + kMargin;
  double box_min_y = std::min(origin.y(), poly_min_y) - kMargin;
  double box_max_y = std::max(origin.y(), poly_max_y) + kMargin;

  bg_box search_box(bg_point(box_min_x, box_min_y),
                     bg_point(box_max_x, box_max_y));

  // Query R-tree for candidate edges
  std::vector<edge_rtree_value> candidates;
  idx.rtree.query(bgi::intersects(search_box),
                  std::back_inserter(candidates));

  double best_t = std::numeric_limits<double>::infinity();
  Eigen::Vector2d best_point;
  bool found = false;

  for (const auto& candidate : candidates) {
    const auto& edge = idx.edges[candidate.second];

    Eigen::Vector2d edge_vec = edge.p2 - edge.p1;
    Eigen::Vector2d to_origin = origin - edge.p1;

    double denom =
        direction.x() * edge_vec.y() - direction.y() * edge_vec.x();
    if (std::abs(denom) < 1e-12) continue;

    double t =
        (to_origin.x() * edge_vec.y() - to_origin.y() * edge_vec.x()) /
        (-denom);
    double u =
        (direction.x() * to_origin.y() - direction.y() * to_origin.x()) /
        denom;

    // The edge is a CLOSED segment: a ray that leaves through a vertex meets both
    // adjacent edges at u = 1 and u = 0 up to round-off, and an exact [0, 1] test misses
    // both (so the polygon looked open there). The tolerance is the geometry epsilon the
    // relation layer uses (1e-9 of the edge length).
    constexpr double kEndTolerance = 1e-9;
    if (t > 1e-6 && u >= -kEndTolerance && u <= 1.0 + kEndTolerance && t < best_t) {
      best_t = t;
      best_point = origin + t * direction;
      found = true;
    }
  }

  return found ? std::optional(best_point) : std::nullopt;
}

double Polygon2D::IntersectionArea(const Polygon2D& other) const {
  if (!impl_ || !other.impl_ || points_.size() < 3 || other.points_.size() < 3)
    return 0.0;
  std::vector<bg_polygon> out;
  bg::intersection(impl_->bg_poly, other.impl_->bg_poly, out);
  double area = 0.0;
  for (const auto& piece : out) area += std::abs(bg::area(piece));
  return area;
}

Polygon2D Polygon2D::Inflate(double distance) const {
  if (!impl_ || points_.size() < 3) return *this;

  bg::model::multi_polygon<bg_polygon> result;
  bg::strategy::buffer::distance_symmetric<double> dist_strategy(distance);
  bg::strategy::buffer::side_straight side_strategy;
  bg::strategy::buffer::join_miter join_strategy;
  bg::strategy::buffer::end_flat end_strategy;
  bg::strategy::buffer::point_circle point_strategy;

  auto buffer_one = [&](const bg_polygon& poly) {
    bg::model::multi_polygon<bg_polygon> part;
    bg::buffer(poly, part, dist_strategy, side_strategy, join_strategy,
               end_strategy, point_strategy);
    for (auto& piece : part) result.push_back(std::move(piece));
  };
  buffer_one(impl_->bg_poly);
  for (const auto& extra : impl_->extra) buffer_one(extra);

  // An erosion that removes the region is empty. Returning the original
  // polygon would report a domain the body cannot occupy.
  return Polygon2DAccess::FromMultiPolygon(std::move(result), /*fill_holes=*/false);
}

Polygon2D Polygon2DAccess::FromMultiPolygon(
    bg::model::multi_polygon<bg_polygon> result, bool fill_holes) {
  if (result.empty()) return Polygon2D();

  struct Component {
    bg_polygon poly;
    double area = 0.0;
    std::vector<Eigen::Vector2d> ring;
  };
  std::vector<Component> comps;
  for (auto& poly : result) {
    Component c;
    c.poly = std::move(poly);
    if (fill_holes) c.poly.inners().clear();  // outer rings only: a hole is filled
    c.area = std::abs(bg::area(c.poly));
    for (const auto& p : c.poly.outer()) c.ring.push_back(FromBg(p));
    if (!c.ring.empty() && c.ring.front() == c.ring.back()) c.ring.pop_back();
    if (c.ring.size() < 3) continue;
    comps.push_back(std::move(c));
  }
  if (comps.empty()) return Polygon2D();

  // Largest ring stays the primary (GetPoints). Every other component is kept:
  // a neck that erodes away must not drop a whole room.
  size_t primary = 0;
  for (size_t i = 1; i < comps.size(); ++i)
    if (comps[i].area > comps[primary].area) primary = i;

  Polygon2D out;
  out.impl_ = std::make_shared<Polygon2D::Impl>();
  out.points_ = std::move(comps[primary].ring);
  out.impl_->bg_poly = std::move(comps[primary].poly);
  out.extra_rings_.reserve(comps.size() - 1);
  out.impl_->extra.reserve(comps.size() - 1);
  for (size_t i = 0; i < comps.size(); ++i) {
    if (i == primary) continue;
    out.extra_rings_.push_back(std::move(comps[i].ring));
    out.impl_->extra.push_back(std::move(comps[i].poly));
  }
  return out;
}

namespace {

std::vector<const bg_polygon*> Components(const bg_polygon& primary,
                                          const std::vector<bg_polygon>& extra,
                                          size_t primary_points) {
  std::vector<const bg_polygon*> out;
  if (primary_points >= 3) out.push_back(&primary);
  for (const auto& e : extra) out.push_back(&e);
  return out;
}

}  // namespace

double Polygon2D::DistanceTo(const Polygon2D& other) const {
  double best = std::numeric_limits<double>::infinity();
  if (!impl_ || !other.impl_) return best;
  for (const bg_polygon* a : Components(impl_->bg_poly, impl_->extra, points_.size()))
    for (const bg_polygon* b :
         Components(other.impl_->bg_poly, other.impl_->extra, other.points_.size()))
      best = std::min(best, static_cast<double>(bg::distance(*a, *b)));
  return best;
}

Polygon2D Polygon2D::Intersection(const Polygon2D& other) const {
  if (!impl_ || !other.impl_) return Polygon2D();
  bg::model::multi_polygon<bg_polygon> out;
  for (const bg_polygon* a : Components(impl_->bg_poly, impl_->extra, points_.size()))
    for (const bg_polygon* b :
         Components(other.impl_->bg_poly, other.impl_->extra, other.points_.size())) {
      std::vector<bg_polygon> piece;
      bg::intersection(*a, *b, piece);
      for (auto& p : piece) out.push_back(std::move(p));
    }
  return Polygon2DAccess::FromMultiPolygon(std::move(out), /*fill_holes=*/false);
}

Polygon2D Polygon2D::Union(const std::vector<Polygon2D>& parts) {
  bg::model::multi_polygon<bg_polygon> acc;
  for (const Polygon2D& part : parts) {
    if (!part.impl_) continue;
    for (const bg_polygon* c :
         Components(part.impl_->bg_poly, part.impl_->extra, part.points_.size())) {
      bg::model::multi_polygon<bg_polygon> next;
      bg::union_(acc, *c, next);
      acc = std::move(next);
    }
  }
  return Polygon2DAccess::FromMultiPolygon(std::move(acc), /*fill_holes=*/false);
}

std::vector<std::vector<Eigen::Vector2d>> Polygon2D::Holes() const {
  std::vector<std::vector<Eigen::Vector2d>> out;
  if (!impl_) return out;
  for (const bg_polygon* c : Components(impl_->bg_poly, impl_->extra, points_.size()))
    for (const auto& inner : c->inners()) {
      std::vector<Eigen::Vector2d> ring;
      for (const auto& p : inner) ring.push_back(FromBg(p));
      if (!ring.empty() && ring.front() == ring.back()) ring.pop_back();
      if (ring.size() >= 3) out.push_back(std::move(ring));
    }
  return out;
}

Polygon2D Polygon2D::Transformed(const Eigen::Matrix2d& a, const Eigen::Vector2d& t) const {
  if (!impl_) return Polygon2D();
  bg::model::multi_polygon<bg_polygon> out;
  auto map_ring = [&](const auto& in, auto* ring) {
    for (const auto& p : in) bg::append(*ring, ToBg(a * FromBg(p) + t));
  };
  for (const bg_polygon* c : Components(impl_->bg_poly, impl_->extra, points_.size())) {
    bg_polygon poly;
    map_ring(c->outer(), &poly.outer());
    for (const auto& inner : c->inners()) {
      poly.inners().emplace_back();
      map_ring(inner, &poly.inners().back());
    }
    bg::correct(poly);  // a reflection reverses the orientation
    out.push_back(std::move(poly));
  }
  return Polygon2DAccess::FromMultiPolygon(std::move(out), /*fill_holes=*/false);
}

namespace {

// Drops repeated vertices and spike vertices (whose two edges are collinear and turn back).
std::vector<Eigen::Vector2d> WithoutSpikes(std::vector<Eigen::Vector2d> ring) {
  if (!ring.empty() && ring.front() == ring.back()) ring.pop_back();
  for (bool changed = true; changed && ring.size() >= 3;) {
    changed = false;
    for (size_t i = 0; i < ring.size(); ++i) {
      const Eigen::Vector2d& prev = ring[(i + ring.size() - 1) % ring.size()];
      const Eigen::Vector2d& cur = ring[i];
      const Eigen::Vector2d& next = ring[(i + 1) % ring.size()];
      const Eigen::Vector2d a = cur - prev, b = next - cur;
      const double scale = std::max(a.norm(), b.norm());
      const bool repeated = a.norm() <= 1e-12 * std::max(1.0, cur.norm());
      const bool spike = scale > 0.0 && std::fabs(a.x() * b.y() - a.y() * b.x()) <= 1e-12 * scale * scale &&
                         a.dot(b) < 0.0;
      if (repeated || spike) {
        ring.erase(ring.begin() + static_cast<std::ptrdiff_t>(i));
        changed = true;
        break;
      }
    }
  }
  return ring;
}

}  // namespace

Polygon2D Polygon2D::Repaired() const {
  if (!impl_) return Polygon2D();
  if (IsValid()) return *this;
  bg::model::multi_polygon<bg_polygon> cleaned;
  double expected_area = 0.0;
  for (const bg_polygon* c : Components(impl_->bg_poly, impl_->extra, points_.size())) {
    bg_polygon poly;
    std::vector<Eigen::Vector2d> outer;
    for (const auto& p : c->outer()) outer.push_back(FromBg(p));
    for (const auto& p : WithoutSpikes(outer)) bg::append(poly.outer(), ToBg(p));
    for (const auto& inner : c->inners()) {
      std::vector<Eigen::Vector2d> ring;
      for (const auto& p : inner) ring.push_back(FromBg(p));
      ring = WithoutSpikes(ring);
      if (ring.size() < 3) continue;
      poly.inners().emplace_back();
      for (const auto& p : ring) bg::append(poly.inners().back(), ToBg(p));
    }
    if (poly.outer().size() < 3) continue;
    bg::correct(poly);
    expected_area += std::abs(bg::area(poly));
    cleaned.push_back(std::move(poly));
  }
  Polygon2D out = Polygon2DAccess::FromMultiPolygon(cleaned, /*fill_holes=*/false);
  if (out.IsValid()) return out;
  // A zero-width slit: the ring passes twice through the same vertex. Cut it there into simple
  // rings; a ring traversed in the original orientation is solid, one traversed against it is a
  // hole the slit cut open (signed areas add up to the region's).
  bg::model::multi_polygon<bg_polygon> solids, holes;
  for (const auto& c : cleaned) {
    std::vector<Eigen::Vector2d> outer;
    for (const auto& p : c.outer()) outer.push_back(FromBg(p));
    if (!outer.empty() && outer.front() == outer.back()) outer.pop_back();
    const double orientation = bg::area(c) >= 0.0 ? 1.0 : -1.0;
    std::vector<std::vector<Eigen::Vector2d>> pending{outer};
    while (!pending.empty()) {
      std::vector<Eigen::Vector2d> ring = WithoutSpikes(pending.back());
      pending.pop_back();
      if (ring.size() < 3) continue;
      bool split = false;
      for (size_t i = 0; i < ring.size() && !split; ++i)
        for (size_t j = i + 1; j < ring.size() && !split; ++j)
          if ((ring[i] - ring[j]).norm() <= 1e-12 * std::max(1.0, ring[i].norm())) {
            pending.emplace_back(ring.begin() + static_cast<std::ptrdiff_t>(i),
                                 ring.begin() + static_cast<std::ptrdiff_t>(j));
            std::vector<Eigen::Vector2d> rest(ring.begin() + static_cast<std::ptrdiff_t>(j), ring.end());
            rest.insert(rest.end(), ring.begin(), ring.begin() + static_cast<std::ptrdiff_t>(i));
            pending.push_back(std::move(rest));
            split = true;
          }
      if (split) continue;
      bg_polygon piece;
      for (const auto& p : ring) bg::append(piece.outer(), ToBg(p));
      bg::append(piece.outer(), ToBg(ring.front()));
      const double signed_area = bg::area(piece);
      bg::correct(piece);
      if (!bg::is_valid(piece)) return Polygon2D();  // it crosses itself: no repair
      (signed_area * orientation >= 0.0 ? solids : holes).push_back(std::move(piece));
    }
    for (const auto& inner : c.inners()) {
      bg_polygon hole;
      hole.outer().assign(inner.begin(), inner.end());
      bg::correct(hole);
      holes.push_back(std::move(hole));
    }
  }
  bg::model::multi_polygon<bg_polygon> solid_union, hole_union, region;
  for (const auto& piece : solids) {
    bg::model::multi_polygon<bg_polygon> next;
    bg::union_(solid_union, piece, next);
    solid_union = std::move(next);
  }
  for (const auto& piece : holes) {
    bg::model::multi_polygon<bg_polygon> next;
    bg::union_(hole_union, piece, next);
    hole_union = std::move(next);
  }
  bg::difference(solid_union, hole_union, region);
  out = Polygon2DAccess::FromMultiPolygon(std::move(region), /*fill_holes=*/false);
  // Only a repair of zero-area features: the same area (a crossing ring's parts cancel in its
  // signed area, so a "repair" that changes it is refused).
  if (!out.IsValid() || std::fabs(out.GetArea() - expected_area) > 1e-9 * std::max(1.0, expected_area))
    return Polygon2D();
  return out;
}

bool Polygon2D::IsValid() const {
  if (!impl_) return false;
  for (const bg_polygon* c : Components(impl_->bg_poly, impl_->extra, points_.size()))
    if (!bg::is_valid(*c)) return false;
  return true;
}

}  // namespace spatialblend
