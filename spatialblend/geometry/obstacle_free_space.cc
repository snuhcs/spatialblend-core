// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/obstacle_free_space.cc
#include "spatialblend/geometry/obstacle_free_space.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

namespace spatialblend {

namespace {

namespace bg = boost::geometry;
using BgPoint = bg::model::d2::point_xy<double>;
using BgPolygon = bg::model::polygon<BgPoint>;
using BgMulti = bg::model::multi_polygon<BgPolygon>;
using BgLine = bg::model::linestring<BgPoint>;

constexpr double kInf = std::numeric_limits<double>::infinity();
// Tolerance for the ">= clearance" comparison: absorbs floating-point noise
// of the distance computation only.
constexpr double kClearanceTol = 1e-9;
// Configuration-space buffer: a regular kArcPoints-gon CIRCUMSCRIBING each
// clearance disc (vertices at r' = r / cos(pi / kArcPoints) * margin, edges
// at >= r), so every node is strictly free under the exact test. The buffer
// exceeds the exact disc by at most r' - r = 0.13% of r (+1e-6 m): a door
// narrower than 2 r' is treated as closed by the search although the exact
// test would pass it (0.2 m radius: doors in [0.4000, 0.4005] m).
constexpr int kArcPoints = 64;
constexpr double kBufferMargin = 1.0001;
constexpr double kHair = ObstacleFreeSpace::kBoundaryHairM;

// The hair is a feasibility offset, not a length: the free space places a point it moves off a
// footprint at the hair-inflated footprint, but every length is measured from the endpoint's FOOT on F's boundary, the
// snap less the hair along the snap's direction (the depth of the point below the boundary), and the path's ends are
// those feet. A point already within the hair of F is its own foot (depth 0). The c-space anchor an endpoint in a
// buffer band attaches through (EndpointAttach::lead) is the same kind of offset: the length runs from the foot
// straight to the first graph node, not through the anchor. So a distance is continuous and 1-Lipschitz in its
// endpoints across a footprint's boundary, and a point's distance to itself is 0 up to rounding.
Eigen::Vector2d FootOnF(const Eigen::Vector2d& p, const Eigen::Vector2d& free_pt, double* depth) {
  const double s = (free_pt - p).norm();
  if (!(s > kHair)) {
    *depth = 0.0;
    return p;
  }
  *depth = s - kHair;
  return p + (free_pt - p) * (*depth / s);
}

double Cross(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
  return a.x() * b.y() - a.y() * b.x();
}

double PointSegmentDistance(const Eigen::Vector2d& p, const Eigen::Vector2d& a,
                            const Eigen::Vector2d& b) {
  const Eigen::Vector2d ab = b - a;
  const double len2 = ab.squaredNorm();
  if (len2 < 1e-24) return (p - a).norm();
  const double t = std::clamp((p - a).dot(ab) / len2, 0.0, 1.0);
  return (p - (a + t * ab)).norm();
}

Eigen::Vector2d ClosestOnSegment(const Eigen::Vector2d& p,
                                 const Eigen::Vector2d& a,
                                 const Eigen::Vector2d& b) {
  const Eigen::Vector2d ab = b - a;
  const double len2 = ab.squaredNorm();
  if (len2 < 1e-24) return a;
  return a + std::clamp((p - a).dot(ab) / len2, 0.0, 1.0) * ab;
}

bool OnSegment(const Eigen::Vector2d& p, const Eigen::Vector2d& a,
               const Eigen::Vector2d& b) {
  return std::min(a.x(), b.x()) - 1e-12 <= p.x() &&
         p.x() <= std::max(a.x(), b.x()) + 1e-12 &&
         std::min(a.y(), b.y()) - 1e-12 <= p.y() &&
         p.y() <= std::max(a.y(), b.y()) + 1e-12;
}

// True when segment [a, b] cannot touch the axis-aligned box [lo, hi] grown
// by `tol` (Liang-Barsky slab test). Exact-safe broadphase: a polygon lies
// inside its bounding box, so a segment that misses the box grown by the
// clearance is at least the clearance away from the polygon. Much tighter
// than comparing bounding boxes, which for a long diagonal segment overlaps
// nearly every obstacle in the room.
bool SegmentMissesBox(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                      const Eigen::Vector2d& lo, const Eigen::Vector2d& hi,
                      double tol) {
  double t0 = 0.0, t1 = 1.0;
  for (int k = 0; k < 2; ++k) {
    const double d = b[k] - a[k];
    const double mn = lo[k] - tol, mx = hi[k] + tol;
    if (std::abs(d) < 1e-300) {
      if (a[k] < mn || a[k] > mx) return true;
      continue;
    }
    double ta = (mn - a[k]) / d, tb = (mx - a[k]) / d;
    if (ta > tb) std::swap(ta, tb);
    t0 = std::max(t0, ta);
    t1 = std::min(t1, tb);
    if (t0 > t1) return true;
  }
  return false;
}

bool SegmentsIntersect(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2,
                       const Eigen::Vector2d& q1, const Eigen::Vector2d& q2) {
  const double d1 = Cross(q2 - q1, p1 - q1);
  const double d2 = Cross(q2 - q1, p2 - q1);
  const double d3 = Cross(p2 - p1, q1 - p1);
  const double d4 = Cross(p2 - p1, q2 - p1);
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
      ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
    return true;
  }
  if (d1 == 0 && OnSegment(p1, q1, q2)) return true;
  if (d2 == 0 && OnSegment(p2, q1, q2)) return true;
  if (d3 == 0 && OnSegment(q1, p1, p2)) return true;
  if (d4 == 0 && OnSegment(q2, p1, p2)) return true;
  return false;
}

double SegmentSegmentDistance(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                              const Eigen::Vector2d& c,
                              const Eigen::Vector2d& d) {
  if (SegmentsIntersect(a, b, c, d)) return 0.0;
  return std::min({PointSegmentDistance(a, c, d), PointSegmentDistance(b, c, d),
                   PointSegmentDistance(c, a, b),
                   PointSegmentDistance(d, a, b)});
}

// PointInRing's on-boundary band: a point within it of a ring edge is ON the
// boundary (a body in contact). Tangent's contact band is derived from it.
constexpr double kOnBoundaryM = 1e-12;

// Point-in-polygon by crossing number over the raw ring: 1 inside, 0 on the
// boundary (within kOnBoundaryM), -1 outside. Much cheaper than Boost's within /
// touches, which dominated the per-frame profile.
int PointInRing(const Eigen::Vector2d& p,
                const std::vector<Eigen::Vector2d>& pts) {
  const size_t n = pts.size();
  if (n < 3) return -1;
  bool inside = false;
  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    const Eigen::Vector2d& a = pts[i];
    const Eigen::Vector2d& b = pts[j];
    // On-boundary test (distance <= 1e-12). A point within 1e-12 of the edge
    // is within 1e-12 of its bounding box, so the (cheap) box test rejects
    // almost every edge before the distance is computed; the result is
    // unchanged.
    if (p.x() >= std::min(a.x(), b.x()) - kOnBoundaryM &&
        p.x() <= std::max(a.x(), b.x()) + kOnBoundaryM &&
        p.y() >= std::min(a.y(), b.y()) - kOnBoundaryM &&
        p.y() <= std::max(a.y(), b.y()) + kOnBoundaryM &&
        PointSegmentDistance(p, a, b) <= kOnBoundaryM)
      return 0;
    if ((a.y() > p.y()) != (b.y() > p.y())) {
      const double x = a.x() + (p.y() - a.y()) * (b.x() - a.x()) / (b.y() - a.y());
      if (p.x() < x) inside = !inside;
    }
  }
  return inside ? 1 : -1;
}

bool InPoly(const Eigen::Vector2d& p, const Polygon2D& poly,
            bool include_boundary = true) {
  const int r = PointInRing(p, poly.GetPoints());
  return include_boundary ? r >= 0 : r > 0;
}

// Exact distance from segment [a, b] to a polygon (0 when they overlap).
double SegmentPolygonDistance(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                              const Polygon2D& poly) {
  if (InPoly(a, poly) || InPoly(b, poly)) return 0.0;
  const auto& pts = poly.GetPoints();
  double best = kInf;
  for (size_t i = 0; i < pts.size(); ++i) {
    best = std::min(best, SegmentSegmentDistance(a, b, pts[i],
                                                 pts[(i + 1) % pts.size()]));
    if (best <= 0.0) return 0.0;
  }
  return best;
}

// True when the segment enters the polygon's INTERIOR. Touching the boundary
// at an endpoint (standing at a contact point) is not entering.
bool SegmentEntersInterior(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                           const Polygon2D& poly) {
  if (InPoly(a, poly, /*include_boundary=*/false) ||
      InPoly(b, poly, /*include_boundary=*/false))
    return true;
  const Eigen::Vector2d d = b - a;
  const double len = d.norm();
  if (len < 1e-12) return false;
  const double shrink = std::min(1e-7, len * 1e-3);
  const Eigen::Vector2d u = d / len;
  return SegmentPolygonDistance(a + u * shrink, b - u * shrink, poly) <= 0.0;
}

double PointPolygonDistance(const Eigen::Vector2d& p, const Polygon2D& poly) {
  if (InPoly(p, poly)) return 0.0;
  return poly.GetDistance(p);
}

// True when segment a-b lies along one boundary edge of `poly` -- the whole
// edge, or any sub-segment of it (e.g. two points snapped onto the same wall).
// Such a segment grazes the boundary without leaving the walkable area, but
// IntersectsSegment reports the collinear overlap as a crossing.
bool IsBoundaryEdge(const Polygon2D& poly, const Eigen::Vector2d& a,
                    const Eigen::Vector2d& b) {
  const auto on_segment = [](const Eigen::Vector2d& x, const Eigen::Vector2d& p,
                             const Eigen::Vector2d& q) {
    const Eigen::Vector2d pq = q - p;
    const double len2 = pq.squaredNorm();
    const double t =
        len2 > 1e-18 ? std::clamp((x - p).dot(pq) / len2, 0.0, 1.0) : 0.0;
    return (p + t * pq - x).norm() < 1e-9;
  };
  const auto& pts = poly.GetPoints();
  for (size_t k = 0; k < pts.size(); ++k) {
    const Eigen::Vector2d& p = pts[k];
    const Eigen::Vector2d& q = pts[(k + 1) % pts.size()];
    if (on_segment(a, p, q) && on_segment(b, p, q)) return true;
  }
  return false;
}

BgPolygon ToBg(const std::vector<Eigen::Vector2d>& pts) {
  BgPolygon poly;
  for (const auto& p : pts) bg::append(poly.outer(), BgPoint(p.x(), p.y()));
  bg::correct(poly);
  return poly;
}

}  // namespace

ObstacleFreeSpace::ObstacleFreeSpace(std::shared_ptr<const Polygon2D> walkable,
                                     std::vector<FreeSpaceObstacle> obstacles,
                                     double clearance,
                                     std::vector<int64_t> bare_keys)
    : walkable_(std::move(walkable)),
      clearance_(std::isfinite(clearance) && clearance > 0.0 ? clearance
                                                             : 0.0) {
  for (auto& o : obstacles) {
    if (!o.footprint || o.footprint->NumPoints() < 3) continue;
    Eigen::Vector2d lo = Eigen::Vector2d::Constant(kInf), hi = -lo;
    for (const auto& p : o.footprint->GetPoints()) {
      lo = lo.cwiseMin(p);
      hi = hi.cwiseMax(p);
    }
    const bool bare =
        std::find(bare_keys.begin(), bare_keys.end(), o.key) != bare_keys.end();
    const double ring = bare ? 0.0 : clearance_;
    obstacle_lo_.push_back(lo.array() - ring);
    obstacle_hi_.push_back(hi.array() + ring);
    touch_.push_back(o.footprint->Inflate(kHair));
    bare_.push_back(bare ? 1 : 0);
    obstacles_.push_back(std::move(o));
  }
  if (walkable_ && walkable_->NumPoints() >= 3) {
    // Convex iff every turn has the same sign (collinear vertices allowed).
    const auto& pts = walkable_->GetPoints();
    bool pos = false, neg = false;
    for (size_t i = 0; i < pts.size(); ++i) {
      const Eigen::Vector2d& p0 = pts[i];
      const Eigen::Vector2d& p1 = pts[(i + 1) % pts.size()];
      const Eigen::Vector2d& p2 = pts[(i + 2) % pts.size()];
      const double cr = Cross(p1 - p0, p2 - p1);
      if (cr > 1e-12) pos = true;
      if (cr < -1e-12) neg = true;
    }
    walkable_convex_ = !(pos && neg);
    if (walkable_->NumComponents() > 1) walkable_convex_ = false;
  }
  // Obstacles whose whole footprint lies in the walkable: with a convex
  // walkable, every point between a walkable point and such a footprint is
  // walkable too (NearestFreeImpl's candidate pruning relies on it).
  obstacle_in_walkable_.assign(obstacles_.size(), !walkable_);
  if (walkable_) {
    for (size_t i = 0; i < obstacles_.size(); ++i) {
      bool all_in = walkable_convex_;
      for (const auto& v : obstacles_[i].footprint->GetPoints()) {
        if (!all_in) break;
        all_in = InWalkable(v);
      }
      obstacle_in_walkable_[i] = all_in;
    }
  }
  buffer_radius_ = clearance_ > 0.0 ? clearance_ /
                                          std::cos(M_PI / kArcPoints) *
                                          kBufferMargin +
                                      kHair
                                    : kHair;
}

void ObstacleFreeSpace::EnsureGraph() const {
  std::call_once(graph_once_, [this] {
    // Configuration-space polygon: walkable \ U buffer(O_i, r').
    std::vector<std::vector<Eigen::Vector2d>> rings;
    bool built = false;
    if (walkable_ && walkable_->NumPoints() >= 3) {
      try {
        BgPolygon walk = ToBg(walkable_->GetPoints());
        BgMulti free_space;
        if (obstacles_.empty()) {
          free_space.push_back(walk);
        } else {
          // The blocked region: every obstacle with a clearance ring buffered by r' (the
          // circumscribed round buffer), every bare one (no ring) by a hair, unioned.
          // The footprints are unioned BEFORE they are buffered: footprints may touch or overlap (a chair against the
          // box beside it, two shelves side by side), a multipolygon whose members touch is not valid input to bg::buffer,
          // and two buffers that overlap by a hair (2e-6 m along a shared edge) are a sliver bg::union_ does not resolve:
          // either way the region came back without them (AF-DESK-01: a chair and the box beside it, sharing the edge
          // x = 4.3, vanished from the configuration space, so the visibility graph had none of their corners while
          // SegmentFree still blocked them, and the geodesic past them jumped by 0.50 m). Footprints that share an edge share
          // its exact coordinates, which the union resolves; the buffer of the union is the union of the buffers.
          //  * At clearance 0 a ringed footprint and a bare one are both buffered by the same hair, so the two groups are
          //    one union and one buffer (two buffers that touch are the sliver again).
          //  * A footprint that is not a valid polygon for the union (zero area: a collinear ring; self-intersecting: a
          //    bowtie) is not unioned, where bg::union_ drops it while SegmentFree still blocks it: its ring is buffered on
          //    its own, as the closed polyline it is, by the same radius, and the buffer is added to the blocked region.
          const auto union_of = [](const BgMulti& polys) {
            BgMulti out;
            for (const auto& p : polys) {
              BgMulti one, merged;
              one.push_back(p);
              bg::union_(out, one, merged);
              out = std::move(merged);
            }
            return out;
          };
          const bool one_group = !(clearance_ > 0.0);
          BgMulti ringed_raw, bare_raw;
          struct Degenerate {
            BgLine ring;
            bool bare;
          };
          std::vector<Degenerate> degenerate;
          for (size_t k = 0; k < obstacles_.size(); ++k) {
            const auto& pts = obstacles_[k].footprint->GetPoints();
            BgPolygon poly = ToBg(pts);
            if (bg::is_valid(poly) && bg::area(poly) > 0.0) {
              (bare_[k] && !one_group ? bare_raw : ringed_raw).push_back(std::move(poly));
            } else {
              Degenerate d;
              for (const auto& q : pts) bg::append(d.ring, BgPoint(q.x(), q.y()));
              bg::append(d.ring, BgPoint(pts.front().x(), pts.front().y()));
              d.bare = bare_[k] != 0;
              degenerate.push_back(std::move(d));
            }
          }
          const BgMulti ringed = union_of(ringed_raw), bare = union_of(bare_raw);
          BgMulti blocked;
          bg::strategy::buffer::side_straight side;
          bg::strategy::buffer::end_round end(kArcPoints);
          bg::strategy::buffer::point_circle point(kArcPoints);
          const auto add = [&](const BgMulti& region) {
            BgMulti merged;
            bg::union_(blocked, region, merged);
            blocked = std::move(merged);
          };
          if (!ringed.empty()) {
            bg::strategy::buffer::distance_symmetric<double> dist(buffer_radius_);
            if (clearance_ > 0.0) {
              bg::strategy::buffer::join_round join(kArcPoints);
              bg::buffer(ringed, blocked, dist, side, join, end, point);
            } else {
              bg::strategy::buffer::join_miter join;
              bg::buffer(ringed, blocked, dist, side, join, end, point);
            }
          }
          if (!bare.empty()) {
            BgMulti bare_blocked;
            bg::strategy::buffer::distance_symmetric<double> dist(kHair);
            bg::strategy::buffer::join_miter join;
            bg::buffer(bare, bare_blocked, dist, side, join, end, point);
            add(bare_blocked);
          }
          for (const auto& d : degenerate) {
            BgMulti line_blocked;
            const bool ring = !d.bare && clearance_ > 0.0;
            bg::strategy::buffer::distance_symmetric<double> dist(ring ? buffer_radius_ : kHair);
            if (ring) {
              bg::strategy::buffer::join_round join(kArcPoints);
              bg::buffer(d.ring, line_blocked, dist, side, join, end, point);
            } else {
              bg::strategy::buffer::join_miter join;
              bg::buffer(d.ring, line_blocked, dist, side, join, end, point);
            }
            add(line_blocked);
          }
          bg::difference(walk, blocked, free_space);
        }
        for (const auto& poly : free_space) {
          auto to_ring = [&](const auto& ring) {
            std::vector<Eigen::Vector2d> pts;
            for (const auto& q : ring) {
              pts.emplace_back(bg::get<0>(q), bg::get<1>(q));
            }
            if (pts.size() > 1 && (pts.front() - pts.back()).norm() < 1e-12)
              pts.pop_back();
            return pts;
          };
          CspacePolygon cp;
          cp.outer = to_ring(poly.outer());
          if (cp.outer.size() >= 3) rings.push_back(cp.outer);
          for (const auto& inner : poly.inners()) {
            auto ring = to_ring(inner);
            if (ring.size() >= 3) {
              rings.push_back(ring);
              cp.inners.push_back(std::move(ring));
            }
          }
          if (cp.outer.size() >= 3) cspace_polys_.push_back(std::move(cp));
        }
        built = true;
      } catch (...) {
        rings.clear();
      }
    }
    if (!built && walkable_) {
      rings.push_back(walkable_->GetPoints());
      cspace_polys_.push_back({walkable_->GetPoints(), {}});
    }
    for (auto& cp : cspace_polys_) {
      const auto box = [](const std::vector<Eigen::Vector2d>& ring,
                          Eigen::Vector2d* lo, Eigen::Vector2d* hi) {
        *lo = ring.front();
        *hi = ring.front();
        for (const auto& q : ring) {
          *lo = lo->cwiseMin(q);
          *hi = hi->cwiseMax(q);
        }
      };
      box(cp.outer, &cp.outer_lo, &cp.outer_hi);
      cp.inner_lo.resize(cp.inners.size());
      cp.inner_hi.resize(cp.inners.size());
      for (size_t k = 0; k < cp.inners.size(); ++k)
        box(cp.inners[k], &cp.inner_lo[k], &cp.inner_hi[k]);
    }

    for (const auto& ring : rings) {
      const size_t n = ring.size();
      for (size_t i = 0; i < n; ++i) {
        const Eigen::Vector2d& p = ring[i];
        cspace_edges_.push_back({p, ring[(i + 1) % n]});
        cspace_edge_lo_.push_back(p.cwiseMin(ring[(i + 1) % n]));
        cspace_edge_hi_.push_back(p.cwiseMax(ring[(i + 1) % n]));
        if (!IsFree(p)) continue;  // numerically on the wrong side
        bool dup = false;
        for (const auto& e : nodes_) {
          if ((e.p - p).norm() < 1e-9) {
            dup = true;
            break;
          }
        }
        if (!dup) nodes_.push_back({p, ring[(i + n - 1) % n], ring[(i + 1) % n]});
      }
    }

    // Reduced visibility graph: bitangent candidates, exact segment test.
    adjacency_.resize(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) {
      for (size_t j = i + 1; j < nodes_.size(); ++j) {
        if (!Tangent(nodes_[i], nodes_[j].p) ||
            !Tangent(nodes_[j], nodes_[i].p))
          continue;
        if (!SegmentFree(nodes_[i].p, nodes_[j].p)) continue;
        const double w = (nodes_[i].p - nodes_[j].p).norm();
        adjacency_[i].push_back({j, w});
        adjacency_[j].push_back({i, w});
        ++edge_count_;
      }
    }
  });
}

bool ObstacleFreeSpace::Tangent(const Node& n, const Eigen::Vector2d& from) {
  // The line from `from` through the vertex is tangent to the ring there when
  // both ring neighbours lie on one side of it (c1, c2 of one sign). c_k is
  // |e_k| times the signed distance of `from` from the line of edge k, so a
  // sign is trusted only when `from` is farther from that line than
  // kBoundaryBandM (the band in which InCspace / PointInRing call a point ON
  // the boundary: a body in contact) plus the determinant's own roundoff.
  // An uncertain sign keeps the edge. The bound depends on that edge alone:
  // the earlier 1e-12 * |d|^2 * max_edge^2 tolerance let the node's OTHER,
  // long edge admit a vertex whose sign was resolved, which moved the
  // geodesic by 4.8e-7 m over a 9e-13 m step (obstacle_free_space_test: the
  // frozen crowded scene); a band of roundoff only, without the contact
  // band, pruned the direct hop of a point in contact (the contact test).
  if (n.p == from) return false;
  const Eigen::Vector2d d = from - n.p;
  const double p1 = d.x() * n.to_prev.y(), q1 = d.y() * n.to_prev.x();
  const double p2 = d.x() * n.to_next.y(), q2 = d.y() * n.to_next.x();
  const double c1 = p1 - q1, c2 = p2 - q2;
  if (!std::isfinite(c1) || !std::isfinite(c2)) return true;
  if (c1 == 0.0 || c2 == 0.0 || (c1 > 0.0) == (c2 > 0.0)) return true;
  constexpr double kEps = std::numeric_limits<double>::epsilon();
  constexpr double kTiny = std::numeric_limits<double>::denorm_min();
  const double scale = from.cwiseAbs().maxCoeff();
  const double dl1 = d.cwiseAbs().sum();
  const auto error = [&](double p, double q, double edge_l1, double edge_scale) {
    return 8.0 * kEps * (std::fabs(p) + std::fabs(q) + std::max(scale, edge_scale) * (dl1 + edge_l1)) +
           4.0 * kTiny;
  };
  constexpr double kBoundaryBandM = 2.0 * kOnBoundaryM;  // PointInRing's band, with margin
  return std::fabs(c1) <= error(p1, q1, n.prev_l1, n.prev_scale) + kBoundaryBandM * n.prev_len ||
         std::fabs(c2) <= error(p2, q2, n.next_l1, n.next_scale) + kBoundaryBandM * n.next_len;
}

bool ObstacleFreeSpace::InCspace(const Eigen::Vector2d& p) const {
  EnsureGraph();
  // Strictly outside a ring's bounding box (by more than PointInRing's 1e-12
  // boundary tolerance) means PointInRing would answer -1 (outside).
  const auto outside_box = [&](const Eigen::Vector2d& lo,
                               const Eigen::Vector2d& hi) {
    return p.x() < lo.x() - 1e-9 || p.y() < lo.y() - 1e-9 ||
           p.x() > hi.x() + 1e-9 || p.y() > hi.y() + 1e-9;
  };
  for (const auto& cp : cspace_polys_) {
    if (outside_box(cp.outer_lo, cp.outer_hi) || PointInRing(p, cp.outer) < 0)
      continue;
    bool in_hole = false;
    for (size_t k = 0; k < cp.inners.size(); ++k) {
      if (outside_box(cp.inner_lo[k], cp.inner_hi[k])) continue;
      if (PointInRing(p, cp.inners[k]) > 0) {
        in_hole = true;
        break;
      }
    }
    if (!in_hole) return true;
  }
  return false;
}

std::optional<Eigen::Vector2d> ObstacleFreeSpace::NearestCspacePoint(
    const Eigen::Vector2d& p) const {
  EnsureGraph();
  std::optional<Eigen::Vector2d> best;
  double best_d = kInf;
  for (size_t e = 0; e < cspace_edges_.size(); ++e) {
    const Eigen::Vector2d& lo = cspace_edge_lo_[e];
    const Eigen::Vector2d& hi = cspace_edge_hi_[e];
    const double dx = std::max({lo.x() - p.x(), 0.0, p.x() - hi.x()});
    const double dy = std::max({lo.y() - p.y(), 0.0, p.y() - hi.y()});
    if (dx * dx + dy * dy >= best_d * best_d) continue;
    const Eigen::Vector2d c =
        ClosestOnSegment(p, cspace_edges_[e].first, cspace_edges_[e].second);
    const double d = (c - p).norm();
    if (d < best_d) {
      best_d = d;
      best = c;
    }
  }
  return best;
}

namespace {
// The part of a leg whose clearance is tested: its ends trimmed by the boundary hair. A leg's end on F may lie up to the
// hair inside a clearance ring (the free space places a point it moves off a footprint at the hair, and a point within
// the hair of F is on F), so the hair at each end is the end's, not the leg's. A leg no longer than two hairs is its
// midpoint (the trimmed ends meet there). A zero-length leg is its point.
std::pair<Eigen::Vector2d, Eigen::Vector2d> ClearanceLeg(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
  const Eigen::Vector2d d = b - a;
  const double len = d.norm();
  if (!(len > 0.0)) return {a, b};
  const double shrink = std::min(kHair, len * 0.5);
  const Eigen::Vector2d u = d / len;
  return {a + u * shrink, b - u * shrink};
}
}  // namespace

bool ObstacleFreeSpace::SegmentReachesContact(const Eigen::Vector2d& a,
                                               const Eigen::Vector2d& b,
                                               const std::vector<int64_t>& own) const {
  if (!a.allFinite() || !b.allFinite()) return false;
  if (!SegmentInWalkable(a, b)) return false;
  const auto [a2, b2] = ClearanceLeg(a, b);
  const Eigen::Vector2d lo = a.cwiseMin(b), hi = a.cwiseMax(b);
  for (size_t i = 0; i < obstacles_.size(); ++i) {
    if (!own.empty() && IsExempt(i, own)) continue;  // the goal's own footprint
    if (hi.x() < obstacle_lo_[i].x() - kClearanceTol ||
        hi.y() < obstacle_lo_[i].y() - kClearanceTol ||
        lo.x() > obstacle_hi_[i].x() + kClearanceTol ||
        lo.y() > obstacle_hi_[i].y() + kClearanceTol)
      continue;
    if (SegmentMissesBox(a, b, obstacle_lo_[i], obstacle_hi_[i], kClearanceTol))
      continue;
    const Polygon2D& fp = *obstacles_[i].footprint;
    if (SegmentEntersInterior(a, b, fp)) return false;
    if (clearance_ > 0.0 &&
        SegmentPolygonDistance(a2, b2, fp) < clearance_ - kClearanceTol)
      return false;
  }
  return true;
}

bool ObstacleFreeSpace::SegmentTouchesContact(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                                               const std::vector<int64_t>& own) const {
  // As SegmentReachesContact, but an own footprint is passed only by TOUCHING it (its boundary and its clearance ring do
  // not block the leg onto its contact; its interior does): a contact on the far side of the referent is not reached
  // through it. For a nearest contact (the source's) the two agree: the leg onto the nearest point never enters the
  // interior.
  if (!a.allFinite() || !b.allFinite()) return false;
  if (!SegmentInWalkable(a, b)) return false;
  const auto [a2, b2] = ClearanceLeg(a, b);
  const Eigen::Vector2d lo = a.cwiseMin(b), hi = a.cwiseMax(b);
  for (size_t i = 0; i < obstacles_.size(); ++i) {
    if (hi.x() < obstacle_lo_[i].x() - kClearanceTol ||
        hi.y() < obstacle_lo_[i].y() - kClearanceTol ||
        lo.x() > obstacle_hi_[i].x() + kClearanceTol ||
        lo.y() > obstacle_hi_[i].y() + kClearanceTol)
      continue;
    if (SegmentMissesBox(a, b, obstacle_lo_[i], obstacle_hi_[i], kClearanceTol))
      continue;
    const Polygon2D& fp = *obstacles_[i].footprint;
    if (SegmentEntersInterior(a, b, fp)) return false;
    if (!own.empty() && IsExempt(i, own)) continue;  // the goal's own footprint: touching it is reaching it
    if (clearance_ > 0.0 &&
        SegmentPolygonDistance(a2, b2, fp) < clearance_ - kClearanceTol)
      return false;
  }
  return true;
}

void ObstacleFreeSpace::EnsureNodeGrid() const {
  EnsureGraph();
  if (node_grid_ready_ || nodes_.empty()) return;
  Eigen::Vector2d lo = nodes_[0].p, hi = lo;
  for (const auto& n : nodes_) {
    lo = lo.cwiseMin(n.p);
    hi = hi.cwiseMax(n.p);
  }
  grid_cell_ = 1.0;
  grid_origin_ = lo;
  grid_nx_ = std::max(1, static_cast<int>(std::ceil((hi.x() - lo.x()) / grid_cell_)) + 1);
  grid_ny_ = std::max(1, static_cast<int>(std::ceil((hi.y() - lo.y()) / grid_cell_)) + 1);
  node_grid_.assign(static_cast<size_t>(grid_nx_ * grid_ny_), {});
  for (size_t i = 0; i < nodes_.size(); ++i) {
    int ix = static_cast<int>((nodes_[i].p.x() - lo.x()) / grid_cell_);
    int iy = static_cast<int>((nodes_[i].p.y() - lo.y()) / grid_cell_);
    ix = std::clamp(ix, 0, grid_nx_ - 1);
    iy = std::clamp(iy, 0, grid_ny_ - 1);
    node_grid_[static_cast<size_t>(iy * grid_nx_ + ix)].push_back(i);
  }
  node_grid_ready_ = true;
}

size_t ObstacleFreeSpace::BestHop(const EndpointAttach& at,
                                  const std::vector<double>& dist,
                                  const std::vector<double>& cell_min,
                                  double* f, size_t* hint) const {
  EnsureNodeGrid();
  const size_t n = nodes_.size();
  const int ncells = grid_nx_ * grid_ny_;
  if (cell_min.size() != static_cast<size_t>(ncells) || dist.size() != n) {
    *f = kInf;
    return n;
  }
  double best = kInf;
  size_t best_i = n;
  // The previous trial's hop is the incumbent, so cells whose lower bound
  // cannot beat it are not opened.
  if (hint && *hint < n && std::isfinite(dist[*hint]) &&
      !(at.prune && !Tangent(nodes_[*hint], at.anchor))) {
    const double fi =
        at.lead + (at.anchor - nodes_[*hint].p).norm() + dist[*hint];
    if (SegmentFree(at.anchor, nodes_[*hint].p)) {
      best = fi;
      best_i = *hint;
    }
  }
  if (query_skip_.size() < n) query_skip_.resize(n);
  std::fill(query_skip_.begin(),
            query_skip_.begin() + static_cast<std::ptrdiff_t>(n), 0);
  auto rect_dist = [&](int c) {
    const int ix = c % grid_nx_;
    const int iy = c / grid_nx_;
    const Eigen::Vector2d lo =
        grid_origin_ + Eigen::Vector2d(ix * grid_cell_, iy * grid_cell_);
    const Eigen::Vector2d hi = lo + Eigen::Vector2d(grid_cell_, grid_cell_);
    const double dx = at.anchor.x() < lo.x()   ? lo.x() - at.anchor.x()
                      : at.anchor.x() > hi.x() ? at.anchor.x() - hi.x()
                                               : 0.0;
    const double dy = at.anchor.y() < lo.y()   ? lo.y() - at.anchor.y()
                      : at.anchor.y() > hi.y() ? at.anchor.y() - hi.y()
                                               : 0.0;
    return std::hypot(dx, dy);
  };
  struct CellLb {
    double lb;
    int c;
  };
  std::vector<CellLb> order;
  order.reserve(static_cast<size_t>(ncells));
  for (int c = 0; c < ncells; ++c) {
    if (!std::isfinite(cell_min[c])) continue;
    const double lb = at.lead + rect_dist(c) + cell_min[c];
    if (!(lb < best)) continue;
    order.push_back({lb, c});
  }
  std::sort(order.begin(), order.end(),
            [](const CellLb& a, const CellLb& b) { return a.lb < b.lb; });
  for (const CellLb& item : order) {
    if (!(item.lb < best)) break;
    for (;;) {
      double best_f = kInf;
      size_t hop = n;
      for (size_t i : node_grid_[static_cast<size_t>(item.c)]) {
        if (query_skip_[i] || i == best_i || !std::isfinite(dist[i])) continue;
        if (at.prune && !Tangent(nodes_[i], at.anchor)) {
          query_skip_[i] = 1;
          continue;
        }
        const double fi = at.lead + (at.anchor - nodes_[i].p).norm() + dist[i];
        if (fi < best_f) {
          best_f = fi;
          hop = i;
        }
      }
      if (hop == n || !(best_f < best)) break;
      if (!SegmentFree(at.anchor, nodes_[hop].p)) {
        query_skip_[hop] = 1;
        continue;
      }
      best = best_f;
      best_i = hop;
      break;
    }
  }
  if (best_i < n && hint) *hint = best_i;
  *f = best;
  return best_i;
}

ObstacleFreeSpace::RingField ObstacleFreeSpace::BuildRingField(
    const std::vector<const std::vector<Eigen::Vector2d>*>& rings,
    const std::vector<int64_t>& own) const {
  RingField field;
  field.open_key = own;
  EnsureGraph();
  EnsureNodeGrid();
  const size_t n = nodes_.size();
  for (const auto* pts : rings)
    for (const auto& p : *pts) field.key.push_back(p);
  field.dist.assign(n, kInf);
  field.foot.assign(n, Eigen::Vector2d::Zero());
  field.last.resize(n);
  for (size_t i = 0; i < n; ++i) field.last[i] = i;
  for (size_t i = 0; i < n; ++i) {
    double best = kInf;
    Eigen::Vector2d foot = nodes_[i].p;
    for (const auto* pts : rings) {
      if (pts->size() < 2) continue;
      for (size_t e = 0; e < pts->size(); ++e) {
        const Eigen::Vector2d c = ClosestOnSegment(
            nodes_[i].p, (*pts)[e], (*pts)[(e + 1) % pts->size()]);
        const double leg = (c - nodes_[i].p).norm();
        if (!(leg < best)) continue;
        if (!SegmentReachesContact(nodes_[i].p, c, own)) continue;
        best = leg;
        foot = c;
      }
    }
    field.dist[i] = best;
    field.foot[i] = foot;
  }
  using Item = std::pair<double, size_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> open;
  for (size_t i = 0; i < n; ++i) {
    if (std::isfinite(field.dist[i])) open.push({field.dist[i], i});
  }
  std::vector<char> closed(n, 0);
  while (!open.empty()) {
    const auto [dg, i] = open.top();
    open.pop();
    if (closed[i] || dg > field.dist[i] + 1e-12) continue;
    closed[i] = 1;
    for (const auto& [v, w] : adjacency_[i]) {
      const double ng = field.dist[i] + w;
      if (ng + 1e-15 >= field.dist[v]) continue;
      field.dist[v] = ng;
      field.foot[v] = field.foot[i];
      field.last[v] = field.last[i];
      open.push({ng, v});
    }
  }
  field.cell_min.assign(node_grid_.size(), kInf);
  for (size_t c = 0; c < node_grid_.size(); ++c) {
    double m = kInf;
    for (size_t i : node_grid_[c]) {
      if (field.dist[i] < m) m = field.dist[i];
    }
    field.cell_min[c] = m;
  }
  return field;
}

const ObstacleFreeSpace::RingField* ObstacleFreeSpace::RingFieldFor(
    const std::vector<const std::vector<Eigen::Vector2d>*>& rings,
    const std::vector<int64_t>& own) const {
  std::vector<Eigen::Vector2d> key;
  for (const auto* pts : rings)
    for (const auto& p : *pts) key.push_back(p);
  auto same = [&](const RingField& existing) {
    const auto& stored = existing.key;
    if (existing.open_key != own) return false;
    if (stored.size() != key.size()) return false;
    for (size_t i = 0; i < key.size(); ++i) {
      if ((stored[i] - key[i]).squaredNorm() > 1e-18) return false;
    }
    return true;
  };
  {
    std::lock_guard<std::mutex> lock(memo_mutex_);
    for (const auto& existing : ring_fields_) {
      if (same(existing)) return &existing;
    }
  }
  RingField built = BuildRingField(rings, own);
  std::lock_guard<std::mutex> lock(memo_mutex_);
  for (const auto& existing : ring_fields_) {
    if (same(existing)) return &existing;
  }
  if (ring_fields_.size() >= kRingFieldCap) ring_fields_.erase(ring_fields_.begin());
  ring_fields_.push_back(std::move(built));
  return &ring_fields_.back();
}

Eigen::Vector2d ObstacleFreeSpace::ClosestOnRing(
    const Eigen::Vector2d& from, const Polygon2D& ring, double* length,
    Eigen::Vector2d* final_leg_origin, const std::vector<int64_t>& own) const {
  const auto set_len = [&](double v) {
    if (length) *length = v;
  };
  if (final_leg_origin) *final_leg_origin = from;
  std::vector<const std::vector<Eigen::Vector2d>*> rings;
  if (ring.GetPoints().size() >= 2) rings.push_back(&ring.GetPoints());
  for (const auto& extra : ring.ExtraRings())
    if (extra.size() >= 2) rings.push_back(&extra);
  if (rings.empty() || !from.allFinite()) {
    set_len(kInf);
    return from;
  }
  Eigen::Vector2d eu = ClosestOnSegment(from, (*rings[0])[0], (*rings[0])[0]);
  double eu_d = kInf;
  for (const auto* pts : rings) {
    for (size_t e = 0; e < pts->size(); ++e) {
      const Eigen::Vector2d c =
          ClosestOnSegment(from, (*pts)[e], (*pts)[(e + 1) % pts->size()]);
      const double d = (c - from).norm();
      if (d < eu_d) {
        eu_d = d;
        eu = c;
      }
    }
  }
  if (SegmentReachesContact(from, eu, own)) {
    set_len(eu_d);
    return eu;
  }

  const bool origin_free = IsFree(from);
  const std::optional<Eigen::Vector2d> snapped =
      origin_free ? std::optional<Eigen::Vector2d>(from)
                  : NearestFreeImpl(from, {}, {}, /*avoid_crossing=*/true);
  if (!snapped) {
    set_len(kInf);
    return eu;
  }
  const double lead = (*snapped - from).norm();
  double best_g = kInf;
  Eigen::Vector2d best = eu;
  Eigen::Vector2d best_origin = from;
  auto consider = [&](const Eigen::Vector2d& node, double node_g) {
    for (const auto* pts : rings) {
      for (size_t e = 0; e < pts->size(); ++e) {
        const Eigen::Vector2d foot =
            ClosestOnSegment(node, (*pts)[e], (*pts)[(e + 1) % pts->size()]);
        const double leg = (foot - node).norm();
        if (node_g + leg >= best_g) continue;
        if (!SegmentReachesContact(node, foot, own)) continue;
        best_g = node_g + leg;
        best = foot;
        best_origin = node;
      }
    }
  };
  consider(*snapped, lead);

  const RingField* field = RingFieldFor(rings, own);
  EndpointAttach at;
  at.anchor = *snapped;
  at.lead = lead;
  at.prune = InCspace(*snapped);
  double hop_f = kInf;
  const size_t hop =
      BestHop(at, field->dist, field->cell_min, &hop_f, &field->hop_hint);
  if (hop < field->foot.size() && hop_f < best_g) {
    best_g = hop_f;
    best = field->foot[hop];
    best_origin = nodes_[field->last[hop]].p;
  }
  if (final_leg_origin && std::isfinite(best_g)) *final_leg_origin = best_origin;
  set_len(best_g);
  return std::isfinite(best_g) ? best : eu;
}

Eigen::Vector2d ObstacleFreeSpace::ClosestOnRingUncached(const Eigen::Vector2d& from,
                                                const Polygon2D& ring,
                                                double* length,
                                                const std::vector<int64_t>& own) const {
  const auto set_len = [&](double v) {
    if (length) *length = v;
  };
  std::vector<const std::vector<Eigen::Vector2d>*> rings;
  if (ring.GetPoints().size() >= 2) rings.push_back(&ring.GetPoints());
  for (const auto& extra : ring.ExtraRings())
    if (extra.size() >= 2) rings.push_back(&extra);
  if (rings.empty() || !from.allFinite()) {
    set_len(kInf);
    return from;
  }
  Eigen::Vector2d eu = ClosestOnSegment(from, (*rings[0])[0], (*rings[0])[0]);
  double eu_d = kInf;
  for (const auto* pts : rings) {
    for (size_t e = 0; e < pts->size(); ++e) {
      const Eigen::Vector2d c =
          ClosestOnSegment(from, (*pts)[e], (*pts)[(e + 1) % pts->size()]);
      const double d = (c - from).norm();
      if (d < eu_d) {
        eu_d = d;
        eu = c;
      }
    }
  }
  // A straight segment that reaches the ring is optimal: geodesic >= Euclidean.
  if (SegmentReachesContact(from, eu, own)) {
    set_len(eu_d);
    return eu;
  }

  Eigen::Vector2d best = eu;
  double best_g = kInf;
  auto consider = [&](const Eigen::Vector2d& node, double node_g) {
    if (!(node_g < best_g)) return;
    for (const auto* pts : rings) {
      for (size_t e = 0; e < pts->size(); ++e) {
        const Eigen::Vector2d foot =
            ClosestOnSegment(node, (*pts)[e], (*pts)[(e + 1) % pts->size()]);
        const double leg = (foot - node).norm();
        if (node_g + leg >= best_g) continue;
        if (!SegmentReachesContact(node, foot, own)) continue;
        best_g = node_g + leg;
        best = foot;
      }
    }
  };

  const bool origin_free = IsFree(from);
  const std::optional<Eigen::Vector2d> snapped =
      origin_free ? std::optional<Eigen::Vector2d>(from)
                  : NearestFreeImpl(from, {}, {}, /*avoid_crossing=*/true);
  if (!snapped) {
    set_len(kInf);
    return eu;
  }
  const double lead = (*snapped - from).norm();
  const Eigen::Vector2d origin = *snapped;
  consider(origin, lead);

  EnsureGraph();
  const size_t n = nodes_.size();
  std::vector<double> g(n, kInf);
  std::vector<char> closed(n, 0);
  using Item = std::pair<double, size_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> open;
  const bool prune = InCspace(origin);
  for (size_t i = 0; i < n; ++i) {
    if (prune && !Tangent(nodes_[i], origin)) continue;
    if (!SegmentFree(origin, nodes_[i].p)) continue;
    const double ng = lead + (origin - nodes_[i].p).norm();
    g[i] = ng;
    open.push({ng, i});
  }
  while (!open.empty()) {
    const auto [dg, i] = open.top();
    open.pop();
    if (closed[i] || dg > g[i]) continue;
    closed[i] = 1;
    if (g[i] >= best_g) break;
    consider(nodes_[i].p, g[i]);
    for (const auto& [v, w] : adjacency_[i]) {
      if (closed[v]) continue;
      const double ng = g[i] + w;
      if (ng < g[v]) {
        g[v] = ng;
        open.push({ng, v});
      }
    }
  }
  set_len(best_g);
  return std::isfinite(best_g) ? best : eu;
}

size_t ObstacleFreeSpace::GraphNodeCount() const {
  EnsureGraph();
  return nodes_.size();
}

size_t ObstacleFreeSpace::GraphEdgeCount() const {
  EnsureGraph();
  return edge_count_;
}

bool ObstacleFreeSpace::ViolatesClearance(double distance) const {
  // Touching an obstacle (distance 0) is never free, even with no clearance.
  return distance <= 0.0 || distance < clearance_ - kClearanceTol;
}

bool ObstacleFreeSpace::InWalkable(const Eigen::Vector2d& p) const {
  // An eroded-away navigable region is empty, not "everywhere".
  if (walkable_ && walkable_->NumComponents() == 0) return false;
  // Boundary points computed elsewhere (ray exits, projections) can land a
  // rounding error outside; they are on the boundary, i.e. walkable.
  constexpr double kBoundaryTol = 1e-9;
  return !walkable_ || InPoly(p, *walkable_) ||
         walkable_->GetDistance(p) <= kBoundaryTol;
}

bool ObstacleFreeSpace::SegmentInWalkable(const Eigen::Vector2d& a,
                                          const Eigen::Vector2d& b) const {
  if (!walkable_) return true;
  if (!InWalkable(a) || !InWalkable(b)) return false;
  // A convex walkable contains every segment between two of its points:
  // nothing more to test (the common rectangular room).
  if (walkable_convex_) return true;
  const Eigen::Vector2d dir = b - a;
  const double len = dir.norm();
  if (len < 1e-12) return true;
  if (IsBoundaryEdge(*walkable_, a, b)) return true;
  // Endpoints may sit exactly on the boundary (walkable vertices, snapped
  // points): test a slightly shrunk segment for crossings, then sample the
  // interior for containment (catches a chord of a concavity).
  const double shrink = std::min(1e-7, len * 1e-3);
  const Eigen::Vector2d u = dir / len;
  if (walkable_->IntersectsSegment(a + u * shrink, b - u * shrink)) {
    return false;
  }
  for (double t : {0.25, 0.5, 0.75}) {
    if (!InWalkable(a + t * dir)) return false;
  }
  return true;
}

bool ObstacleFreeSpace::IsExempt(size_t i,
                                 const std::vector<int64_t>& exempt) const {
  return !exempt.empty() &&
         std::find(exempt.begin(), exempt.end(), obstacles_[i].key) !=
             exempt.end();
}

double ObstacleFreeSpace::ObstacleDistance(
    const Eigen::Vector2d& p, const std::vector<int64_t>& exempt) const {
  double best = kInf;
  for (size_t i = 0; i < obstacles_.size(); ++i) {
    if ((!exempt.empty() && IsExempt(i, exempt))) continue;
    best = std::min(best, PointPolygonDistance(p, *obstacles_[i].footprint));
  }
  return best;
}

bool ObstacleFreeSpace::IsFree(const Eigen::Vector2d& p,
                               const std::vector<int64_t>& exempt,
                               const std::vector<int64_t>& ring_exempt) const {
  if (!p.allFinite()) return false;
  // F is the navigable domain, then minus forbidden interiors and clearance
  // rings, then plus engagement exemptions of those obstacles. A footprint
  // outside the navigable domain does not extend F.
  if (!InWalkable(p)) return false;
  for (size_t i = 0; i < obstacles_.size(); ++i) {
    if ((!exempt.empty() && IsExempt(i, exempt))) continue;
    if (p.x() < obstacle_lo_[i].x() - kClearanceTol ||
        p.y() < obstacle_lo_[i].y() - kClearanceTol ||
        p.x() > obstacle_hi_[i].x() + kClearanceTol ||
        p.y() > obstacle_hi_[i].y() + kClearanceTol)
      continue;
    const Polygon2D& fp = *obstacles_[i].footprint;
    if (bare_[i] || (!ring_exempt.empty() && IsExempt(i, ring_exempt))) {
      if (InPoly(p, fp, /*include_boundary=*/false)) return false;
    } else if (ViolatesClearance(PointPolygonDistance(p, fp))) {
      return false;
    }
  }
  return true;
}

bool ObstacleFreeSpace::SegmentFree(
    const Eigen::Vector2d& a, const Eigen::Vector2d& b,
    const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt) const {
  if (!SegmentInWalkable(a, b)) return false;
  const Eigen::Vector2d lo = a.cwiseMin(b), hi = a.cwiseMax(b);
  for (size_t i = 0; i < obstacles_.size(); ++i) {
    if ((!exempt.empty() && IsExempt(i, exempt))) continue;
    if (hi.x() < obstacle_lo_[i].x() - kClearanceTol ||
        hi.y() < obstacle_lo_[i].y() - kClearanceTol ||
        lo.x() > obstacle_hi_[i].x() + kClearanceTol ||
        lo.y() > obstacle_hi_[i].y() + kClearanceTol)
      continue;
    if (SegmentMissesBox(a, b, obstacle_lo_[i], obstacle_hi_[i], kClearanceTol))
      continue;
    const Polygon2D& fp = *obstacles_[i].footprint;
    if (bare_[i] || (!ring_exempt.empty() && IsExempt(i, ring_exempt))) {
      if (SegmentEntersInterior(a, b, fp)) return false;
    } else if (ViolatesClearance(SegmentPolygonDistance(a, b, fp))) {
      return false;
    }
  }
  return true;
}

bool ObstacleFreeSpace::DomainEmpty() const {
  return walkable_ && walkable_->NumComponents() == 0;
}

std::optional<Eigen::Vector2d> ObstacleFreeSpace::NearestFree(
    const Eigen::Vector2d& p, const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt, int64_t* branch,
    NearestFreeStatus* status) const {
  return NearestFreeImpl(p, exempt, ring_exempt, /*avoid_crossing=*/false,
                         branch, status);
}

std::optional<Eigen::Vector2d> ObstacleFreeSpace::NearestFreeImpl(
    const Eigen::Vector2d& p, const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt, bool avoid_crossing,
    int64_t* branch, NearestFreeStatus* status) const {
  if (branch) *branch = 0;
  auto set_status = [&](NearestFreeStatus s) {
    if (status) *status = s;
  };
  if (!p.allFinite()) {
    set_status(DomainEmpty() ? NearestFreeStatus::kDomainEmpty
                             : NearestFreeStatus::kSearchFoundNone);
    return std::nullopt;
  }
  if (DomainEmpty()) {
    set_status(NearestFreeStatus::kDomainEmpty);
    return std::nullopt;
  }
  if (IsFree(p, exempt, ring_exempt)) {
    set_status(NearestFreeStatus::kAlreadyFree);
    return p;
  }

  // With avoid_crossing, the snap leg p -> c must not pass through an
  // obstacle p is not inside: escaping a couch flush against a thin wall must
  // come out on the couch's side, not through the wall to a nearer free
  // point beyond it. Only when no such candidate exists does the nearest
  // free point win regardless.
  auto crosses_other_obstacle = [&](const Eigen::Vector2d& c) {
    if (!avoid_crossing) return false;
    const Eigen::Vector2d lo = p.cwiseMin(c), hi = p.cwiseMax(c);
    for (size_t i = 0; i < obstacles_.size(); ++i) {
      if ((!exempt.empty() && IsExempt(i, exempt))) continue;
      if (hi.x() < obstacle_lo_[i].x() || hi.y() < obstacle_lo_[i].y() ||
          lo.x() > obstacle_hi_[i].x() || lo.y() > obstacle_hi_[i].y())
        continue;
      const Polygon2D& fp = *obstacles_[i].footprint;
      if (InPoly(p, fp)) continue;  // the obstacle being escaped
      if (SegmentEntersInterior(p, c, fp)) return true;
    }
    return false;
  };
  std::optional<Eigen::Vector2d> best, best_any;
  double best_d = kInf, best_any_d = kInf;
  int64_t best_id = 0, best_any_id = 0;
  // The branch identity of a candidate: its kind in the high bits, the obstacle /
  // edge in the low ones.
  enum : int64_t { kRadial = 1, kWalkable = 2, kExemptOb = 3, kCspaceEdge = 4, kNode = 5 };
  const auto id_of = [](int64_t kind, int64_t index, int64_t feature) {
    return (kind << 48) | (index << 16) | (feature & 0xffff);
  };
  auto consider = [&](const Eigen::Vector2d& c, int64_t id) {
    if (!c.allFinite() || !IsFree(c, exempt, ring_exempt)) return;
    const double d = (c - p).norm();
    if (d < best_any_d) {
      best_any_d = d;
      best_any = c;
      best_any_id = id;
    }
    if (d < best_d && !crosses_other_obstacle(c)) {
      best_d = d;
      best = c;
      best_id = id;
    }
  };

  // Exact single-constraint candidates: push p radially out of each violated
  // obstacle to exactly the clearance distance (the c-space polygon is a
  // slightly conservative approximation of the round buffer), then back into
  // the walkable if that left it.
  //
  // Pruning (result-preserving): the candidate of an obstacle p is clear of
  // lies between p and that obstacle, at distance d_i - r from p, where
  // r <= clearance + hair and d_i >= the distance from p to the (clearance-
  // grown) bounding box. Once a free candidate
  // exists, one that cannot be strictly nearer is skipped before its
  // nearest-point query, which dominates this loop in a room with many
  // obstacles. Valid when the candidate cannot leave the walkable (no
  // walkable-projection candidate follows): p inside a convex walkable and
  // the footprint inside it too.
  const bool prune_ok = !walkable_ || (walkable_convex_ && InWalkable(p));
  for (size_t i = 0; i < obstacles_.size(); ++i) {
    if ((!exempt.empty() && IsExempt(i, exempt))) continue;
    if (prune_ok && obstacle_in_walkable_[i] && best_d < kInf) {
      const double dx = std::max({obstacle_lo_[i].x() - p.x(), 0.0,
                                  p.x() - obstacle_hi_[i].x()});
      const double dy = std::max({obstacle_lo_[i].y() - p.y(), 0.0,
                                  p.y() - obstacle_hi_[i].y()});
      const double d_box = std::sqrt(dx * dx + dy * dy);
      if (d_box > 0.0 && d_box - clearance_ - kHair - 1e-9 >= best_d) continue;
    }
    const Polygon2D& fp = *obstacles_[i].footprint;
    const Eigen::Vector2d q = fp.GetNearestPoint(p);
    Eigen::Vector2d dir = InPoly(p, fp) ? (q - p) : (p - q);
    if (dir.norm() < 1e-12) continue;
    dir.normalize();
    const double r = (bare_[i] || (!ring_exempt.empty() && IsExempt(i, ring_exempt))) ? kHair : clearance_ + kHair;
    const Eigen::Vector2d c = q + dir * r;
    // The nearest footprint FEATURE is part of the branch: an edge, or a vertex (the
    // rounded corner of the clearance ring). Inside the footprint the nearest edge
    // switches across the footprint's medial axis; outside, the corner of a convex
    // footprint turns the push direction through 90 degrees within a hair of the vertex.
    int64_t edge = 0;
    {
      const auto& pts = fp.GetPoints();
      double bd = kInf;
      for (size_t k = 0; k < pts.size(); ++k) {
        const Eigen::Vector2d &a0 = pts[k], &b0 = pts[(k + 1) % pts.size()];
        const Eigen::Vector2d e = b0 - a0;
        const double len2 = e.squaredNorm();
        const double t = len2 < 1e-18 ? 0.0 : (p - a0).dot(e) / len2;
        const Eigen::Vector2d near_pt = a0 + std::clamp(t, 0.0, 1.0) * e;
        const double dk = (near_pt - p).norm();
        if (dk < bd) {
          bd = dk;
          // an interior point of the edge, or the vertex at the clamped end
          edge = (t > 0.0 && t < 1.0) ? 2 * static_cast<int64_t>(k) + 1
                                      : 2 * static_cast<int64_t>(t <= 0.0 ? k : (k + 1) % pts.size()) + 2;
        }
      }
    }
    consider(c, id_of(kRadial, static_cast<int64_t>(i), edge));
    if (walkable_ && !InWalkable(c))
      consider(walkable_->GetNearestPoint(c), id_of(kWalkable, 0, 0));
  }
  // Exemptions open more space: the exempt footprints themselves, just
  // outside a ring-exempt footprint, and the walkable boundary near them.
  if (walkable_) consider(walkable_->GetNearestPoint(p), id_of(kWalkable, 0, 0));
  for (size_t i = 0; i < obstacles_.size(); ++i) {
    if ((!exempt.empty() && IsExempt(i, exempt))) {
      consider(obstacles_[i].footprint->GetNearestPoint(p),
               id_of(kExemptOb, static_cast<int64_t>(i), 0));
    } else if (bare_[i] || (!ring_exempt.empty() && IsExempt(i, ring_exempt))) {
      const Eigen::Vector2d c = touch_[i].GetNearestPoint(p);
      consider(c, id_of(kExemptOb, static_cast<int64_t>(i), 1));
      if (walkable_ && !InWalkable(c))
        consider(walkable_->GetNearestPoint(c), id_of(kWalkable, 0, 0));
    }
  }
  // Always also scan the configuration-space boundary (walkable \ U
  // clearance buffers), free for every exemption set: when several
  // constraints bind at once (a point inside two overlapping rings, or a ring
  // and a wall) its nearest point is the nearest free point up to the
  // buffer's polygonal approximation, and the single-constraint candidates
  // above can all be far worse (seen: the walkable boundary 4.5 m away vs
  // 0.24 m). Edges whose bounding box is no closer than the current best are
  // skipped, so the scan is a cheap bbox loop in the common case.
  EnsureGraph();
  for (size_t e = 0; e < cspace_edges_.size(); ++e) {
    const Eigen::Vector2d& lo = cspace_edge_lo_[e];
    const Eigen::Vector2d& hi = cspace_edge_hi_[e];
    const double dx = std::max({lo.x() - p.x(), 0.0, p.x() - hi.x()});
    const double dy = std::max({lo.y() - p.y(), 0.0, p.y() - hi.y()});
    if (dx * dx + dy * dy >= best_d * best_d) continue;
    consider(ClosestOnSegment(p, cspace_edges_[e].first, cspace_edges_[e].second),
             id_of(kCspaceEdge, static_cast<int64_t>(e), 0));
  }
  if (!best) {
    for (size_t k = 0; k < nodes_.size(); ++k)
      consider(nodes_[k].p, id_of(kNode, static_cast<int64_t>(k), 0));
  }
  if (branch) *branch = best ? best_id : best_any_id;
  if (best || best_any) set_status(NearestFreeStatus::kProjected);
  else
    set_status(DomainEmpty() ? NearestFreeStatus::kDomainEmpty
                             : NearestFreeStatus::kSearchFoundNone);
  return best ? best : best_any;
}

std::vector<int64_t> ObstacleFreeSpace::KeysAround(
    const Eigen::Vector2d& p) const {
  std::vector<int64_t> keys;
  for (size_t i = 0; i < obstacles_.size(); ++i) {
    // Outside the footprint's bounding box grown by the clearance (+ the
    // comparison's 1e-9): provably not within the clearance, skip the
    // polygon distance.
    if (p.x() < obstacle_lo_[i].x() - 1e-9 || p.y() < obstacle_lo_[i].y() - 1e-9 ||
        p.x() > obstacle_hi_[i].x() + 1e-9 || p.y() > obstacle_hi_[i].y() + 1e-9)
      continue;
    if (PointPolygonDistance(p, *obstacles_[i].footprint) < (bare_[i] ? 0.0 : clearance_) + 1e-9) {
      keys.push_back(obstacles_[i].key);
    }
  }
  return keys;
}

ObstacleFreeSpace::PathResult ObstacleFreeSpace::FindPath(
    const Eigen::Vector2d& from, const Eigen::Vector2d& to) const {
  // Repeated queries within a frame (the receiver solve, the anchor-target
  // pass and the mediator ask the same question) hit a small memo.
  {
    std::lock_guard<std::mutex> lock(memo_mutex_);
    for (const auto& m : memo_) {
      if (m.from == from && m.to == to) return m.result;
    }
  }
  PathResult result = FindPathUncached(from, to);
  std::lock_guard<std::mutex> lock(memo_mutex_);
  if (memo_.size() >= kMemoSize) memo_.erase(memo_.begin());
  memo_.push_back({from, to, result});
  return result;
}

ObstacleFreeSpace::PathResult ObstacleFreeSpace::FindPathUncached(
    const Eigen::Vector2d& from, const Eigen::Vector2d& to) const {
  PathResult result;
  if (!from.allFinite() || !to.allFinite()) return result;

  // Direct: the straight segment is free under the same F as IsFree. Endpoint
  // clearance rings are not opened unless the caller built this free space
  // with those obstacles bare.
  if (SegmentFree(from, to)) {
    result.found = true;
    result.points = {from, to};
    result.length = (to - from).norm();
    result.from_free = from;
    result.to_free = to;
    return result;
  }

  const std::optional<Eigen::Vector2d> a =
      NearestFreeImpl(from, {}, {}, /*avoid_crossing=*/true);
  const std::optional<Eigen::Vector2d> b =
      NearestFreeImpl(to, {}, {}, /*avoid_crossing=*/true);
  if (!a || !b) {
    result.domain_empty = DomainEmpty();
    result.search_found_none = !result.domain_empty;
    return result;
  }
  result.from_free = *a;
  result.to_free = *b;
  double from_depth = 0.0, to_depth = 0.0;
  const Eigen::Vector2d qa = FootOnF(from, *a, &from_depth);
  const Eigen::Vector2d qb = FootOnF(to, *b, &to_depth);
  result.from_snap_m = from_depth;
  result.to_snap_m = to_depth;

  std::vector<Eigen::Vector2d> middle;
  std::vector<Eigen::Vector2d> graph_nodes;  // the interior nodes, in order (the length runs foot, nodes, foot)
  if (SegmentFree(*a, *b)) {
    middle = {*a, *b};
  } else {
    // A* over the visibility graph with the two snapped endpoints attached.
    EnsureGraph();
    const size_t n = nodes_.size();
    const size_t kStart = n, kGoal = n + 1;
    // Endpoint attachment. The O(1) bitangent test is a valid pruning for an
    // endpoint inside the configuration-space free region (outside every
    // clearance buffer). An endpoint in the thin band between the exact
    // clearance and the buffer (e.g. a point snapped to exactly r) is first
    // joined by a short free leg to the nearest c-space boundary point, which
    // is then attached with pruning; only if that leg is blocked is the
    // endpoint attached to every visible node.
    struct Attach {
      Eigen::Vector2d anchor;  // the point the graph is attached from
      double lead = 0.0;       // leg from the endpoint to `anchor`
      bool prune = false;      // bitangent test valid at `anchor`
    };
    auto make_attach = [&](const Eigen::Vector2d& e) {
      Attach at;
      at.anchor = e;
      at.prune = InCspace(e);
      if (!at.prune) {
        const auto cp = NearestCspacePoint(e);
        if (cp && SegmentFree(e, *cp)) {
          at.anchor = *cp;
          at.lead = (*cp - e).norm();
          at.prune = true;
        }
      }
      return at;
    };
    const Attach sa = make_attach(*a), ga = make_attach(*b);
    // The endpoint -> node edges are validated LAZILY, when a search entry
    // that uses one is popped, instead of testing every one of the (several
    // thousand) graph nodes for visibility up front: an A* that reaches the
    // goal after expanding a few dozen nodes then pays a few dozen exact
    // segment tests, not thousands. Entries pop in f order and an invalid
    // edge is simply dropped, so the returned path is the same shortest path
    // the eager attachment finds.
    auto edge_ok = [&](const Attach& at, size_t i, bool reverse) {
      if (at.prune && !Tangent(nodes_[i], at.anchor)) return false;
      return reverse ? SegmentFree(nodes_[i].p, at.anchor)
                     : SegmentFree(at.anchor, nodes_[i].p);
    };
    auto pos = [&](size_t k) -> const Eigen::Vector2d& {
      return k == kStart ? *a : (k == kGoal ? *b : nodes_[k].p);
    };
    enum Edge : uint8_t { kGraph, kFromStart, kToGoal };
    struct Item {
      double f, g;
      size_t node, parent;
      Edge edge;
      bool operator>(const Item& o) const { return f > o.f; }
    };
    // Pre-sized backing store: the start expansion pushes up to n entries,
    // and growing the heap's vector by doubling copies them repeatedly.
    std::vector<Item> open_store;
    open_store.reserve(n + 16);
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> open(
        std::greater<Item>(), std::move(open_store));
    std::vector<char> closed(n + 2, 0);
    std::vector<double> g(n + 2, kInf);
    std::vector<size_t> parent(n + 2, SIZE_MAX);
    open.push({(*a - *b).norm(), 0.0, kStart, SIZE_MAX, kGraph});
    while (!open.empty()) {
      const Item it = open.top();
      open.pop();
      if (closed[it.node]) continue;
      // Lazy edge validation (see above).
      if (it.edge == kFromStart && !edge_ok(sa, it.node, /*reverse=*/false))
        continue;
      if (it.edge == kToGoal && !edge_ok(ga, it.parent, /*reverse=*/true))
        continue;
      closed[it.node] = 1;
      g[it.node] = it.g;
      parent[it.node] = it.parent;
      if (it.node == kGoal) break;
      auto push = [&](size_t v, double w, Edge edge) {
        if (closed[v]) return;
        const double ng = it.g + w;
        open.push({ng + (pos(v) - *b).norm(), ng, v, it.node, edge});
      };
      if (it.node == kStart) {
        for (size_t i = 0; i < n; ++i) {
          if (sa.prune && !Tangent(nodes_[i], sa.anchor)) continue;
          push(i, sa.lead + (sa.anchor - nodes_[i].p).norm(), kFromStart);
        }
      } else {
        for (const auto& [v, w] : adjacency_[it.node]) push(v, w, kGraph);
        if (!ga.prune || Tangent(nodes_[it.node], ga.anchor))
          push(kGoal, ga.lead + (nodes_[it.node].p - ga.anchor).norm(),
               kToGoal);
      }
    }
    if (!closed[kGoal]) return result;  // disconnected
    for (size_t k = kGoal; k != SIZE_MAX; k = parent[k]) {
      middle.push_back(pos(k));
      if (k < n) graph_nodes.push_back(pos(k));
      // The lead legs of band endpoints pass through their c-space anchor (a waypoint of the walkable course; not of the
      // length, FootOnF).
      if (k == kGoal && parent[k] < n && ga.lead > 0.0)
        middle.push_back(ga.anchor);
      if (k < n && parent[k] == kStart && sa.lead > 0.0)
        middle.push_back(sa.anchor);
    }
    std::reverse(graph_nodes.begin(), graph_nodes.end());
    std::reverse(middle.begin(), middle.end());
  }
  // The search ran between the free points a and b. The LENGTH is measured from the endpoints' feet (FootOnF): the
  // depths, then the feet straight to the interior nodes (or to each other), as GeodesicDistance measures it. The POINTS
  // keep the walkable course: from, the free point a, the interior (with the c-space anchors), b, to (a body walks
  // through F, which the free points are in and a footprint's boundary is not; the feet are not waypoints); a and b lie
  // within the hair of the feet, so the points' polyline exceeds the length by at most the two hairs and the anchors'
  // offsets, feasibility offsets, not lengths.
  result.found = true;
  result.length = result.from_snap_m + result.to_snap_m;
  {
    Eigen::Vector2d prev = qa;
    for (const auto& node : graph_nodes) {
      result.length += (node - prev).norm();
      prev = node;
    }
    result.length += (qb - prev).norm();
  }
  if ((*a - from).norm() > 1e-12) result.points.push_back(from);
  for (const auto& p : middle) result.points.push_back(p);
  if ((*b - to).norm() > 1e-12) result.points.push_back(to);
  return result;
}

ObstacleFreeSpace::EndpointAttach ObstacleFreeSpace::AttachEndpoint(
    const Eigen::Vector2d& e) const {
  EndpointAttach at;
  at.anchor = e;
  at.prune = InCspace(e);
  if (!at.prune) {
    const auto cp = NearestCspacePoint(e);
    if (cp && SegmentFree(e, *cp)) {
      at.anchor = *cp;
      at.lead = (*cp - e).norm();
      at.prune = true;
    }
  }
  return at;
}

ObstacleFreeSpace::GoalField ObstacleFreeSpace::BuildGoalField(
    const Eigen::Vector2d& goal_free, const Eigen::Vector2d& goal_end, const Eigen::Vector2d& goal,
    const std::vector<int64_t>& open_keys) const {
  GoalField field;
  field.goal_free = goal_free;
  field.goal_end = goal_end;
  field.goal = goal;
  field.open = open_keys;
  // The field's distances run to the GOAL itself: through its foot (goal_end, plus the foot's depth below F's boundary),
  // or, for the goal's own footprints (`open_keys`, the referent's: they never block the final leg onto its contact, as
  // ClosestOnRing's `own`), straight onto the goal.
  const double goal_depth = (goal - goal_end).norm();
  EnsureGraph();
  const size_t n = nodes_.size();
  field.dist.assign(n, kInf);
  field.legs.assign(n, 0);
  if (n == 0) return field;

  const EndpointAttach ga = AttachEndpoint(goal_free);
  struct Item {
    double g;
    size_t node;
    int seed;  // 0: a graph edge, 1: a leg to the goal's free point, 2: a leg onto the goal past its own footprints
    int legs;
    bool operator>(const Item& o) const { return g > o.g; }
  };
  std::vector<Item> store;
  store.reserve(n);
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> open(
      std::greater<Item>(), std::move(store));
  // best[] stays +inf until a leg is known to be free. Seeding it with the
  // straight-line length would hide a longer graph path when that leg is
  // blocked, and the relaxation would never be retried.
  std::vector<double> best(n, kInf);
  for (size_t i = 0; i < n; ++i) {
    if (ga.prune && !Tangent(nodes_[i], ga.anchor)) continue;
    // The last leg ends at the goal's foot (goal_end, FootOnF), straight from the node: not at the free point the field
    // attaches, nor through its c-space anchor.
    const double g0 = (nodes_[i].p - goal_end).norm() + goal_depth;
    open.push({g0, i, 1, goal_depth > 0.0 ? 2 : 1});
  }
  if (!open_keys.empty())
    for (size_t i = 0; i < n; ++i) open.push({(nodes_[i].p - goal).norm(), i, 2, 1});
  std::vector<char> closed(n, 0);
  while (!open.empty()) {
    const Item it = open.top();
    open.pop();
    if (it.g > best[it.node] + 1e-12) continue;
    if (it.seed == 1 && !SegmentFree(nodes_[it.node].p, ga.anchor)) continue;
    if (it.seed == 2 && !SegmentTouchesContact(nodes_[it.node].p, goal, open_keys)) continue;
    if (closed[it.node]) continue;
    closed[it.node] = 1;
    best[it.node] = it.g;
    field.dist[it.node] = it.g;
    field.legs[it.node] = it.legs;
    for (const auto& [v, w] : adjacency_[it.node]) {
      if (closed[v]) continue;
      const double ng = it.g + w;
      if (ng + 1e-15 >= best[v]) continue;
      best[v] = ng;
      open.push({ng, v, 0, it.legs + 1});
    }
  }
  EnsureNodeGrid();
  field.cell_min.assign(node_grid_.size(), kInf);
  for (size_t c = 0; c < node_grid_.size(); ++c) {
    double m = kInf;
    for (size_t i : node_grid_[c]) {
      if (field.dist[i] < m) m = field.dist[i];
    }
    field.cell_min[c] = m;
  }
  return field;
}

const ObstacleFreeSpace::GoalField* ObstacleFreeSpace::GoalFieldFor(
    const Eigen::Vector2d& goal_free, const Eigen::Vector2d& goal_end, const Eigen::Vector2d& goal,
    const std::vector<int64_t>& open_keys) const {
  constexpr double kSameContact = 1e-9;
  auto same = [&](const GoalField& f) {
    return (f.goal_free - goal_free).squaredNorm() <= kSameContact * kSameContact && f.goal_end == goal_end &&
           f.goal == goal && f.open == open_keys;
  };
  {
    std::lock_guard<std::mutex> lock(memo_mutex_);
    for (const auto& existing : goal_fields_) {
      if (same(existing)) return &existing;
    }
  }
  GoalField built = BuildGoalField(goal_free, goal_end, goal, open_keys);
  std::lock_guard<std::mutex> lock(memo_mutex_);
  for (const auto& existing : goal_fields_) {
    if (same(existing)) return &existing;
  }
  if (goal_fields_.size() >= kGoalFieldCap) {
    goal_fields_.erase(goal_fields_.begin());
  }
  goal_fields_.push_back(std::move(built));
  return &goal_fields_.back();
}

std::optional<Eigen::Vector2d> ObstacleFreeSpace::GoalSnap(
    const Eigen::Vector2d& to) const {
  {
    std::lock_guard<std::mutex> lock(memo_mutex_);
    for (const auto& entry : goal_snaps_) {
      if (entry.to == to) return entry.snap;
    }
  }
  // A pure function of `to` on an immutable free space, so the cached answer
  // is the one the query would compute. A contact is fixed for a whole solve
  // and every Jacobian sample and trial measures to it.
  std::optional<Eigen::Vector2d> snap =
      NearestFreeImpl(to, {}, {}, /*avoid_crossing=*/true);
  std::lock_guard<std::mutex> lock(memo_mutex_);
  if (goal_snaps_.size() >= kGoalFieldCap) goal_snaps_.erase(goal_snaps_.begin());
  goal_snaps_.push_back({to, snap});
  return snap;
}

double ObstacleFreeSpace::GeodesicDistance(const Eigen::Vector2d& from,
                                          const Eigen::Vector2d& to,
                                          const std::vector<int64_t>& open_keys) const {
  if (!from.allFinite() || !to.allFinite()) return kInf;
  // Same first answer as FindPath: a free straight segment is the geodesic.
  if (SegmentFree(from, to)) return (from - to).norm();

  const std::optional<Eigen::Vector2d> a =
      NearestFreeImpl(from, {}, {}, /*avoid_crossing=*/true);
  const std::optional<Eigen::Vector2d> b = GoalSnap(to);
  if (!a || !b) return kInf;
  // Lengths from the endpoints' feet on F (FootOnF), as FindPath's: the hair is not a length.
  double from_snap = 0.0, to_snap = 0.0;
  const Eigen::Vector2d qa = FootOnF(from, *a, &from_snap);
  const Eigen::Vector2d qb = FootOnF(to, *b, &to_snap);
  // The goal's own footprints (`open_keys`) do not block the final leg onto it (as ClosestOnRing's `own`).
  const bool open_direct = !open_keys.empty() && SegmentTouchesContact(qa, to, open_keys);
  if (open_direct) return from_snap + (qa - to).norm();
  if (SegmentFree(*a, *b)) return from_snap + (qa - qb).norm() + to_snap;

  const GoalField* field = GoalFieldFor(*b, qb, to, open_keys);
  const size_t n = field->dist.size();
  if (n == 0 || n != nodes_.size()) return kInf;
  const EndpointAttach sa = AttachEndpoint(*a);
  double hop_f = kInf;
  const size_t hop =
      BestHop(sa, field->dist, field->cell_min, &hop_f, &field->hop_hint);
  if (hop >= n || !std::isfinite(hop_f)) return kInf;
  // The first leg from the foot straight to the node, not from the free point the search attached (nor through its
  // c-space anchor).
  // (The field's distances already run to the goal itself, its foot's depth included.)
  return from_snap + (qa - nodes_[hop].p).norm() + field->dist[hop];
}

std::vector<ObstacleFreeSpace::RoutePiece> ObstacleFreeSpace::GeodesicRoutes(
    const Eigen::Vector2d& from, const Eigen::Vector2d& to, double slack,
    const std::vector<int64_t>& open_keys) const {
  std::vector<RoutePiece> out;
  if (!from.allFinite() || !to.allFinite()) return out;
  const auto gam = [](int k) {
    constexpr double u = 1.1102230246251565e-16;
    return k * u / (1.0 - k * u);
  };
  // Mirrors GeodesicDistance, listing the candidates instead of the best only.
  if (SegmentFree(from, to)) {
    RoutePiece r;
    r.node = -1;
    r.first = to;
    r.value = (from - to).norm();
    r.legs = 1;
    r.err = gam(r.legs + 3) * r.value;
    out.push_back(r);
    return out;
  }
  const std::optional<Eigen::Vector2d> a =
      NearestFreeImpl(from, {}, {}, /*avoid_crossing=*/true);
  const std::optional<Eigen::Vector2d> b = GoalSnap(to);
  if (!a || !b) return out;
  // Lengths from the endpoints' feet on F (FootOnF), as GeodesicDistance's.
  double from_snap = 0.0, to_snap = 0.0;
  const Eigen::Vector2d qa = FootOnF(from, *a, &from_snap);
  const Eigen::Vector2d qb = FootOnF(to, *b, &to_snap);
  const int snap_legs = (from_snap > 0.0 ? 1 : 0) + (to_snap > 0.0 ? 1 : 0);
  // The goal's own footprints do not block the final leg onto it (as GeodesicDistance): the straight leg from the foot.
  // It is the shortest, but not the only route: where that leg starts to graze the referent, the routes round its
  // corners tie with it, so they are listed with it (below), as a route list must for the stationarity and tie scans.
  std::optional<RoutePiece> direct;
  if (!open_keys.empty() && SegmentTouchesContact(qa, to, open_keys)) {
    RoutePiece r;
    r.node = -1;
    r.first = (qa - from).norm() > 1e-9 ? qa : to;
    r.value = from_snap + (qa - to).norm();
    r.legs = (from_snap > 0.0 ? 1 : 0) + 1;
    r.err = gam(r.legs + 3) * r.value;
    direct = r;
  }
  if (!direct && SegmentFree(*a, *b)) {
    RoutePiece r;
    r.node = -1;
    // A foot within v1's kEpsilon (1e-9) of the point is no leg with a direction: the route leaves towards the next one.
    r.first = (qa - from).norm() > 1e-9 ? qa : qb;
    r.value = from_snap + (qa - qb).norm() + to_snap;
    r.legs = snap_legs + 1;
    r.err = gam(r.legs + 3) * r.value;
    out.push_back(r);
    return out;
  }
  const GoalField* field = GoalFieldFor(*b, qb, to, open_keys);
  const size_t n = field->dist.size();
  const auto finish = [&]() {
    if (direct) out.push_back(*direct);
    std::sort(out.begin(), out.end(), [](const RoutePiece& p, const RoutePiece& q) { return p.value < q.value; });
    return out;
  };
  if (n == 0 || n != nodes_.size()) return finish();
  const EndpointAttach sa = AttachEndpoint(*a);
  double best_f = kInf;
  const size_t best = BestHop(sa, field->dist, field->cell_min, &best_f, &field->hop_hint);
  if (best >= n || !std::isfinite(best_f)) return finish();
  // The direct leg (when open) is a route of the same list: the bound is taken from the shorter of the two.
  if (direct) best_f = std::min(best_f, direct->value - from_snap);
  // Every route within the bound. A route's error is at most gamma of the segments
  // it can have; the caller's slack must cover both errors for the overlap test, so
  // the bound includes the shortest's error twice over (the node count is a
  // trivial upper bound on legs).
  const double bound_f = best_f + std::max(slack, 0.0) + 2.0 * gam(static_cast<int>(n) + 3) * best_f;
  EnsureNodeGrid();
  const int ncells = grid_nx_ * grid_ny_;
  for (int c = 0; c < ncells; ++c) {
    if (!std::isfinite(field->cell_min[c])) continue;
    const int ix = c % grid_nx_, iy = c / grid_nx_;
    const Eigen::Vector2d lo = grid_origin_ + Eigen::Vector2d(ix * grid_cell_, iy * grid_cell_);
    const Eigen::Vector2d hi = lo + Eigen::Vector2d(grid_cell_, grid_cell_);
    const double dx = sa.anchor.x() < lo.x() ? lo.x() - sa.anchor.x() : (sa.anchor.x() > hi.x() ? sa.anchor.x() - hi.x() : 0.0);
    const double dy = sa.anchor.y() < lo.y() ? lo.y() - sa.anchor.y() : (sa.anchor.y() > hi.y() ? sa.anchor.y() - hi.y() : 0.0);
    if (sa.lead + std::hypot(dx, dy) + field->cell_min[c] > bound_f) continue;
    for (size_t i : node_grid_[static_cast<size_t>(c)]) {
      if (!std::isfinite(field->dist[i])) continue;
      if (sa.prune && !Tangent(nodes_[i], sa.anchor)) continue;
      // The first leg from the foot straight to the node (FootOnF), as GeodesicDistance's.
      const Eigen::Vector2d hop_pt = nodes_[i].p;
      const double f = (qa - hop_pt).norm() + field->dist[i];
      // A route whose first waypoint is the point itself (within v1's kEpsilon, 1e-9) has a first leg with no direction: the
      // route continues to the next node, which is listed with the same length.
      const Eigen::Vector2d first_wp = (qa - from).norm() > 1e-9 ? qa : hop_pt;
      if ((from - first_wp).norm() <= 1e-9) continue;
      if (f > bound_f && i != best) continue;
      if (i != best && !SegmentFree(sa.anchor, nodes_[i].p)) continue;
      RoutePiece r;
      r.node = static_cast<int64_t>(i);
      // A snap shorter than the free space's boundary hair (the point is on F, a rounding off it) is not a leg with a
      // direction: the route leaves towards its next waypoint (v1 takes the first segment longer than kEpsilon the same way).
      r.first = first_wp;
      r.value = from_snap + f;  // the field's distances run to the goal itself, its foot's depth included
      r.legs = (from_snap > 0.0 ? 1 : 0) + 1 + field->legs[i];
      r.err = gam(r.legs + 3) * r.value;
      out.push_back(r);
    }
  }
  return finish();
}

std::vector<ObstacleFreeSpace::RoutePiece> ObstacleFreeSpace::ContactRoutes(
    const Eigen::Vector2d& from, const std::vector<const std::vector<Eigen::Vector2d>*>& rings, double slack) const {
  struct Cand {
    double eu;
    Eigen::Vector2d foot;
    int edge;
    bool vertex;
  };
  std::vector<Cand> cands;
  int edge_index = 0;
  for (const auto* pts : rings) {
    for (size_t e = 0; e < pts->size(); ++e, ++edge_index) {
      const Eigen::Vector2d& a = (*pts)[e];
      const Eigen::Vector2d& b = (*pts)[(e + 1) % pts->size()];
      const Eigen::Vector2d foot = ClosestOnSegment(from, a, b);
      cands.push_back({(foot - from).norm(), foot, edge_index, foot == a || foot == b});
    }
  }
  std::sort(cands.begin(), cands.end(), [](const Cand& p, const Cand& q) { return p.eu < q.eu; });
  std::vector<RoutePiece> out;
  double best = kInf;
  // Pruning only: no route has more segments than the graph has nodes plus the snap legs, so its length carries a relative
  // error of at most gamma_{nodes + 6}; twice that bounds the difference of two routes' errors.
  const double prune_margin = [&] {
    constexpr double u = 1.1102230246251565e-16;
    const double k = static_cast<double>(GraphNodeCount()) + 6.0;
    return 2.0 * k * u / (1.0 - k * u);
  }();
  for (const Cand& c : cands) {
    // The geodesic to a contact is at least the straight distance to it: a contact farther than the best route plus the
    // slack cannot be within it.
    if (c.eu > best + std::max(slack, 0.0) + prune_margin * std::max(best, c.eu)) break;
    for (RoutePiece p : GeodesicRoutes(from, c.foot, slack)) {
      bool dup = false;
      for (const auto& q : out)
        if (q.node == p.node && (q.contact - c.foot).norm() == 0.0) { dup = true; break; }
      if (dup) continue;
      p.contact = c.foot;
      p.edge = c.edge;
      p.foot_vertex = c.vertex;
      p.flat_foot = p.node == -1 && !c.vertex;
      out.push_back(p);
    }
    if (!out.empty()) best = std::min(best, out.front().value);
    for (const auto& p : out) best = std::min(best, p.value);
  }
  std::sort(out.begin(), out.end(), [](const RoutePiece& p, const RoutePiece& q) { return p.value < q.value; });
  // Within the slack, or tied to the arithmetic (length intervals overlapping).
  while (!out.empty() &&
         out.back().value - out.front().value > std::max(slack, 0.0) + out.back().err + out.front().err)
    out.pop_back();
  return out;
}

namespace {

// Closed angular intervals on [0, 2 pi].
using Arcs = std::vector<std::pair<double, double>>;
constexpr double kTwoPi = 6.283185307179586;

double Ang(const Eigen::Vector2d& v) {
  double a = std::atan2(v.y(), v.x());
  return a < 0.0 ? a + kTwoPi : a;
}

// Directions from `from` counter-clockwise through `width` radians.
Arcs SectorArcs(double from, double width) {
  Arcs out;
  if (width >= kTwoPi) return {{0.0, kTwoPi}};
  if (width <= 0.0) return {{from, from}};
  double lo = std::fmod(from, kTwoPi);
  if (lo < 0.0) lo += kTwoPi;
  const double hi = lo + width;
  if (hi <= kTwoPi) {
    out.push_back({lo, hi});
  } else {
    out.push_back({lo, kTwoPi});
    out.push_back({0.0, hi - kTwoPi});
  }
  return out;
}

Arcs Normalize(Arcs a) {
  std::sort(a.begin(), a.end());
  Arcs out;
  for (const auto& iv : a) {
    if (!out.empty() && iv.first <= out.back().second) out.back().second = std::max(out.back().second, iv.second);
    else out.push_back(iv);
  }
  // 0 and 2 pi are the same direction: join an arc ending at 2 pi with one starting at 0.
  if (out.size() >= 2 && out.front().first <= 0.0 && out.back().second >= kTwoPi) {
    // keep both pieces (they are adjacent on the circle); the caller treats arcs as a set of closed intervals
  }
  return out;
}

Arcs UnionArcs(const Arcs& a, const Arcs& b) {
  Arcs all = a;
  all.insert(all.end(), b.begin(), b.end());
  return Normalize(all);
}

Arcs IntersectArcs(const Arcs& a, const Arcs& b) {
  Arcs out;
  for (const auto& p : a)
    for (const auto& q : b) {
      const double lo = std::max(p.first, q.first), hi = std::min(p.second, q.second);
      if (lo <= hi) out.push_back({lo, hi});
    }
  return Normalize(out);
}

// Signed area of a ring (positive: counter-clockwise).
double SignedArea(const std::vector<Eigen::Vector2d>& r) {
  double s = 0.0;
  for (size_t i = 0; i < r.size(); ++i) s += Cross(r[i], r[(i + 1) % r.size()]);
  return 0.5 * s;
}

// The vertices of a ring with consecutive duplicates removed, counter-clockwise.
std::vector<Eigen::Vector2d> CcwRing(const std::vector<Eigen::Vector2d>& in) {
  std::vector<Eigen::Vector2d> r;
  for (const auto& p : in)
    if (r.empty() || (p - r.back()).norm() > 0.0) r.push_back(p);
  while (r.size() > 1 && (r.front() - r.back()).norm() == 0.0) r.pop_back();
  if (SignedArea(r) < 0.0) std::reverse(r.begin(), r.end());
  return r;
}

// The cone of directions that stay inside the region a counter-clockwise ring bounds, at x within tol of the ring.
// `interior`: the region is the ring's inside (true) or its outside (false).
// Returns false when the ring is farther than tol.
bool RingCone(const Eigen::Vector2d& x, const std::vector<Eigen::Vector2d>& ring, double tol, bool interior,
              Arcs* cone, std::vector<ObstacleFreeSpace::Facet>* facets, ObstacleFreeSpace::Facet::Kind edge_kind,
              ObstacleFreeSpace::Facet::Kind vertex_kind, int64_t key) {
  const size_t n = ring.size();
  if (n < 3) return false;
  // Nearest vertex within tol: both incident edges bound the cone.
  size_t vk = n;
  double vbest = tol;
  for (size_t k = 0; k < n; ++k) {
    const double d = (x - ring[k]).norm();
    if (d <= vbest) { vbest = d; vk = k; }
  }
  const auto add_facet = [&](const Eigen::Vector2d& edge_dir, ObstacleFreeSpace::Facet::Kind kind) {
    // edge_dir: the ring's direction (counter-clockwise); the ring's inside is on its left.
    ObstacleFreeSpace::Facet f;
    f.kind = kind;
    f.obstacle_key = key;
    Eigen::Vector2d left(-edge_dir.y(), edge_dir.x());
    left.normalize();
    f.normal = interior ? Eigen::Vector2d(-left) : left;  // out of the region
    f.tangent = Eigen::Vector2d(-f.normal.y(), f.normal.x());
    facets->push_back(f);
  };
  if (vk < n) {
    const Eigen::Vector2d a = ring[vk] - ring[(vk + n - 1) % n];
    const Eigen::Vector2d b = ring[(vk + 1) % n] - ring[vk];
    // Inside sector: counter-clockwise from b to -a.
    double w = Ang(-a) - Ang(b);
    if (w < 0.0) w += kTwoPi;
    if (w == 0.0 && Cross(a, b) != 0.0) w = kTwoPi;
    if (interior) {
      *cone = SectorArcs(Ang(b), w);
    } else {
      *cone = SectorArcs(Ang(-a), kTwoPi - w);
    }
    add_facet(a, vertex_kind);
    add_facet(b, vertex_kind);
    return true;
  }
  // Nearest edge within tol.
  size_t ek = n;
  double ebest = tol;
  for (size_t k = 0; k < n; ++k) {
    const double d = PointSegmentDistance(x, ring[k], ring[(k + 1) % n]);
    if (d <= ebest) { ebest = d; ek = k; }
  }
  if (ek == n) return false;
  const Eigen::Vector2d b = ring[(ek + 1) % n] - ring[ek];
  *cone = interior ? SectorArcs(Ang(b), M_PI) : SectorArcs(Ang(b) + M_PI, M_PI);
  add_facet(b, edge_kind);
  return true;
}

}  // namespace

bool ObstacleFreeSpace::FeasibleCone::Contains(double angle) const {
  double a = std::fmod(angle, kTwoPi);
  if (a < 0.0) a += kTwoPi;
  for (const auto& iv : arcs)
    if ((a >= iv.first && a <= iv.second) || (a + kTwoPi >= iv.first && a + kTwoPi <= iv.second)) return true;
  return false;
}

ObstacleFreeSpace::FeasibleCone ObstacleFreeSpace::TangentConeAt(
    const Eigen::Vector2d& x, const std::vector<int64_t>& exempt,
    const std::vector<int64_t>& ring_exempt, double tol) const {
  FeasibleCone cone;
  cone.arcs = {{0.0, kTwoPi}};
  // The walkable: the cone of the component whose ring passes within tol of x (a point inside a
  // component away from its ring continues in every direction).
  if (walkable_ && walkable_->NumComponents() > 0) {
    std::vector<std::vector<Eigen::Vector2d>> rings;
    rings.push_back(walkable_->GetPoints());
    for (const auto& extra : walkable_->ExtraRings()) rings.push_back(extra);
    Arcs wall;
    bool any = false;
    for (const auto& raw : rings) {
      const std::vector<Eigen::Vector2d> ring = CcwRing(raw);
      Arcs one;
      if (RingCone(x, ring, tol, /*interior=*/true, &one, &cone.facets, Facet::kWallEdge, Facet::kWallVertex, 0)) {
        wall = any ? UnionArcs(wall, one) : one;
        any = true;
      }
    }
    if (any) cone.arcs = IntersectArcs(cone.arcs, wall);
  }
  for (size_t i = 0; i < obstacles_.size(); ++i) {
    if (IsExempt(i, exempt)) continue;
    const Polygon2D& fp = *obstacles_[i].footprint;
    const std::vector<Eigen::Vector2d> ring = CcwRing(fp.GetPoints());
    if (ring.size() < 3) continue;
    const bool no_ring = bare_[i] || IsExempt(i, ring_exempt) || clearance_ <= 0.0;
    const double dist = PointPolygonDistance(x, fp);
    Arcs obs;
    if (no_ring) {
      if (dist > tol) continue;
      if (!RingCone(x, ring, tol, /*interior=*/false, &obs, &cone.facets, Facet::kFootprintEdge, Facet::kFootprintVertex,
                    obstacles_[i].key))
        continue;
    } else {
      // The clearance ring: the points at distance `clearance` from the footprint. Where the nearest point of
      // the footprint is a vertex the ring is an arc of that radius, else it is parallel to the edge.
      if (std::fabs(dist - clearance_) > tol && dist > clearance_) continue;
      Eigen::Vector2d q = ring[0];
      double qd = kInf;
      bool at_vertex = false;
      for (size_t k = 0; k < ring.size(); ++k) {
        const Eigen::Vector2d c = ClosestOnSegment(x, ring[k], ring[(k + 1) % ring.size()]);
        const double d = (x - c).norm();
        if (d < qd) {
          qd = d;
          q = c;
          at_vertex = (c - ring[k]).norm() == 0.0 || (c - ring[(k + 1) % ring.size()]).norm() == 0.0;
        }
      }
      const Eigen::Vector2d off = x - q;
      if (!(off.norm() > 0.0)) continue;
      const Eigen::Vector2d nrm = off.normalized();  // into F
      obs = SectorArcs(Ang(nrm) - M_PI / 2.0, M_PI);
      Facet f;
      f.kind = at_vertex ? Facet::kRingArc : Facet::kRingEdge;
      f.normal = -nrm;
      f.tangent = Eigen::Vector2d(-f.normal.y(), f.normal.x());
      f.radius = at_vertex ? clearance_ : 0.0;
      f.obstacle_key = obstacles_[i].key;
      cone.facets.push_back(f);
    }
    cone.arcs = IntersectArcs(cone.arcs, obs);
  }
  return cone;
}

}  // namespace spatialblend
