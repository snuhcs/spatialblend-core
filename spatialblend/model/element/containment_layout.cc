// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/containment_layout.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "spatialblend/model/element/geometry_role.h"
#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {
namespace {

constexpr double kEpsilon = 1e-9;  // the relation layer's geometry epsilon

// Every layout check reads the DECLARED definition: the scope extension is derived
// from the declared regions (scope_extension.h).
const SpatialDefinition* Declared(const Element* e) {
  return e ? e->GetDeclaredSpatialDefinition() : nullptr;
}

bool IsRegion(const Element* e) {
  return Declared(e) && IsContainmentRole(Declared(e)->GetGeometryRole());
}

// Any element with a footprint, whatever its role: the layout is a hierarchy, and two sibling
// footprints may only touch. What is one thing to the app is one element
// with parts or sub-regions, or a parent with children.
bool HasFootprint(const Element* e) {
  if (!Declared(e)) return false;
  const auto boundary = Declared(e)->GetBoundary();
  return boundary && boundary->NumPoints() >= 3;
}

// The footprint of `def` (an element's definition) in world coordinates (a Polytope keeps its
// geometry in the element's local frame; baked Navigable/Surface definitions are world).
std::shared_ptr<Polygon2D> WorldFootprintOf(const ElementTree& tree, const Element* e,
                                            const SpatialDefinition* def) {
  if (!def) return nullptr;
  const auto boundary = def->GetBoundary();
  if (!boundary || boundary->NumPoints() < 3) return nullptr;
  // A definition that is not element-local keeps its ring in its own frame: the identity for a
  // world-absolute one (Navigable), the plane's frame for a Surface (relation_kernel.h
  // FootprintImage); the ring is placed by that frame, not left in it.
  if (!def->RequiresElementWorldTransform()) {
    const Eigen::Matrix4d frame = def->GetTransform();
    if (frame.isIdentity(0.0)) return boundary;
    std::vector<Eigen::Vector2d> placed;
    placed.reserve(boundary->NumPoints());
    for (const auto& p : boundary->GetPoints()) {
      const Eigen::Vector4d w = frame * Eigen::Vector4d(p.x(), p.y(), 0.0, 1.0);
      placed.emplace_back(w.x(), w.y());
    }
    return std::make_shared<Polygon2D>(placed);
  }
  const Eigen::Matrix4d world = tree.GetWorldTransform(e->GetId());
  std::vector<Eigen::Vector2d> pts;
  pts.reserve(boundary->NumPoints());
  for (const auto& p : boundary->GetPoints()) {
    const Eigen::Vector4d w = world * Eigen::Vector4d(p.x(), p.y(), 0.0, 1.0);
    pts.emplace_back(w.x(), w.y());
  }
  return std::make_shared<Polygon2D>(pts);
}

std::shared_ptr<Polygon2D> WorldFootprint(const ElementTree& tree, const Element* e) {
  return WorldFootprintOf(tree, e, Declared(e));
}

std::string Name(const Element* e) {
  std::string s = "element " + std::to_string(e->GetId());
  if (!e->GetTag().empty()) s += " ('" + e->GetTag() + "')";
  return s;
}

// The area of `child` outside `parent` when it sticks out (more than the geometry
// epsilon relative to the child's area), else 0.
double OutsideArea(const Polygon2D& child, const Polygon2D& parent) {
  const double outside = child.GetArea() - child.IntersectionArea(parent);
  return outside > kEpsilon * child.GetArea() ? outside : 0.0;
}

// Overlap of two footprints beyond the precision of the data (the sibling rule).
std::optional<ContainmentViolation> CheckOverlap(const Element* a, const Polygon2D& pa,
                                                 const Element* b, const Polygon2D& pb,
                                                 const char* what) {
  const double inter = pa.IntersectionArea(pb);
  // The precision of the data, as a LENGTH (the apps hand the core single-precision matrices, relation/support.h):
  // a coordinate is known to a float epsilon of its magnitude, eps_f * max(1, |x|), so two footprints that touch
  // may intersect in a strip of at most twice that width along the smaller one's boundary. (A ratio of areas
  // would reject a touch that differs by one float step along a short edge, far from the origin.)
  constexpr double kFloatEps = 1.1920928955078125e-07;  // 2^-23
  const auto extent = [](const Polygon2D& p) {
    double m = 1.0, per = 0.0;
    const auto& pts = p.GetPoints();
    for (size_t i = 0; i < pts.size(); ++i) {
      m = std::max({m, std::fabs(pts[i].x()), std::fabs(pts[i].y())});
      per += (pts[(i + 1) % pts.size()] - pts[i]).norm();
    }
    return std::make_pair(m, per);
  };
  const auto ea = extent(pa), eb = extent(pb);
  const double strip = 2.0 * kFloatEps * std::max(ea.first, eb.first) * std::min(ea.second, eb.second);
  if (inter <= strip) return std::nullopt;
  ContainmentViolation v;
  v.first = a->GetId();
  v.second = b->GetId();
  v.code = SceneErrorCode::kContainmentOverlap;
  v.message = std::string(what) + Name(a) + " and " + Name(b) + " (" + std::to_string(inter) +
              " m^2); sibling elements may only touch. Express one thing as one element with "
              "parts or sub-regions, or as a parent with children";
  return v;
}

std::shared_ptr<Polygon2D> UnionFootprintOf(const Polygon2D& footprint) {
  if (footprint.IsValid()) return std::make_shared<Polygon2D>(footprint);
  Polygon2D repaired = footprint.Repaired();
  if (repaired.NumComponents() == 0) return nullptr;
  return std::make_shared<Polygon2D>(std::move(repaired));
}

// Whether `child` is inside `parent` by a cheap test: every vertex inside a convex parent.
// False means "not shown inside" (the exact area test decides).
bool CheaplyInside(const Polygon2D& child, const Polygon2D& parent) {
  if (!parent.IsConvex()) return false;
  for (const auto& p : child.GetPoints())
    if (!parent.Contains(p)) return false;
  return true;
}

// A region child against its region parent: inside, or sticking out at any distance, which
// extends the parent's scope (`*extends`); only a footprint that is not a simple polygon is
// rejected. An actor's element is not checked against its
// parent: it never extends a parent and its pose never reshapes a region,
// whichever was registered first (an actor's pose is never checked either).
std::optional<ContainmentViolation> CheckParent(const ElementTree& tree, const Element* child,
                                                const Element* parent, bool* extends) {
  if (IsActorBound(tree, child->GetId())) return std::nullopt;
  const auto pc = WorldFootprint(tree, child);
  const auto pp = WorldFootprint(tree, parent);
  if (!pc || !pp) return std::nullopt;
  if (CheaplyInside(*pc, *pp)) return std::nullopt;
  ContainmentViolation v;
  v.first = child->GetId();
  v.second = parent->GetId();
  // The footprints the extension would unite (zero-area spikes and keyhole slits repaired, as the
  // build repairs them). A ring that crosses itself has no well-defined area (its signed parts
  // cancel), so whether it sticks out cannot be decided and its union is not its region: refused.
  const auto uc = UnionFootprintOf(*pc);
  const auto up = UnionFootprintOf(*pp);
  if (!uc) {
    v.code = SceneErrorCode::kInvalidArgument;
    v.message = "containment: " + Name(child) + " is not inside its parent " + Name(parent) +
                " and its footprint is not a simple polygon (it crosses itself); a region that "
                "is not inside its parent must be simple";
    return v;
  }
  if (!up) {
    if (OutsideArea(*uc, *pp) == 0.0) return std::nullopt;  // inside, by the unrepaired ring
    v.code = SceneErrorCode::kInvalidArgument;
    v.message = "containment: " + Name(child) + " sticks out of its parent " + Name(parent) +
                ", whose footprint is not a simple polygon and cannot be repaired (it crosses "
                "itself), so its scope cannot be extended";
    return v;
  }
  if (OutsideArea(*uc, *up) == 0.0) return std::nullopt;
  *extends = true;  // at any distance: across a gap it is bridged to the parent's nearest point
  return std::nullopt;
}

bool InLine(const ElementTree& tree, element_id_t a, element_id_t b) {
  if (a == b) return true;
  for (element_id_t x : tree.GetAncestorIds(a))
    if (x == b) return true;
  for (element_id_t x : tree.GetAncestorIds(b))
    if (x == a) return true;
  return false;
}

}  // namespace

namespace {

Polygon2D Box(const Polygon2D& p, double margin) {
  Eigen::Vector2d lo = Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector2d hi = -lo;
  auto grow = [&](const std::vector<Eigen::Vector2d>& ring) {
    for (const auto& q : ring) {
      lo = lo.cwiseMin(q);
      hi = hi.cwiseMax(q);
    }
  };
  grow(p.GetPoints());
  for (const auto& r : p.ExtraRings()) grow(r);
  lo.array() -= margin;
  hi.array() += margin;
  return Polygon2D({{lo.x(), lo.y()}, {hi.x(), lo.y()}, {hi.x(), hi.y()}, {lo.x(), hi.y()}});
}

}  // namespace

namespace {

// Every ring of `p` (outer rings, further components and holes) as closed edge loops.
std::vector<std::vector<Eigen::Vector2d>> AllRings(const Polygon2D& p) {
  std::vector<std::vector<Eigen::Vector2d>> rings{p.GetPoints()};
  for (const auto& r : p.ExtraRings()) rings.push_back(r);
  for (const auto& r : p.Holes()) rings.push_back(r);
  return rings;
}

Eigen::Vector2d ClosestOnSegment(const Eigen::Vector2d& q, const Eigen::Vector2d& a,
                                 const Eigen::Vector2d& b) {
  const Eigen::Vector2d d = b - a;
  const double len2 = d.squaredNorm();
  if (!(len2 > 0.0)) return a;
  return a + std::clamp((q - a).dot(d) / len2, 0.0, 1.0) * d;
}

// The nearest points of two disjoint regions' boundaries (`*pa` on `a`, `*pc` on `c`): of two
// segments that do not cross, a nearest pair has an endpoint of one of them. Of the pairs that
// tie (parallel edges hold an interval of them; their midpoint is one too), the one whose
// segment passes closest to the child's centre, so the corridor does not slide to a corner.
bool NearestPoints(const Polygon2D& a, const Polygon2D& c, Eigen::Vector2d* pa,
                   Eigen::Vector2d* pc) {
  const auto ra = AllRings(a), rc = AllRings(c);
  // Every edge pair's candidates, the ends of the interval its nearest pairs span.
  const auto for_each_pair = [&](const auto& visit) {
    for (const auto& ar : ra)
      for (size_t i = 0; i < ar.size(); ++i)
        for (const auto& cr : rc)
          for (size_t j = 0; j < cr.size(); ++j) {
            const Eigen::Vector2d &a0 = ar[i], &a1 = ar[(i + 1) % ar.size()];
            const Eigen::Vector2d &c0 = cr[j], &c1 = cr[(j + 1) % cr.size()];
            const std::pair<Eigen::Vector2d, Eigen::Vector2d> cand[4] = {
                {ClosestOnSegment(c0, a0, a1), c0},
                {ClosestOnSegment(c1, a0, a1), c1},
                {a0, ClosestOnSegment(a0, c0, c1)},
                {a1, ClosestOnSegment(a1, c0, c1)}};
            visit(cand);
          }
  };
  double best = std::numeric_limits<double>::infinity();
  for_each_pair([&](const auto& cand) {
    for (const auto& [x, y] : cand) best = std::min(best, (x - y).norm());
  });
  if (!std::isfinite(best)) return false;
  const double tie = 1e-9 * (1.0 + best);
  Eigen::Vector2d centre = Eigen::Vector2d::Zero();
  for (const auto& q : c.GetPoints()) centre += q;
  centre /= static_cast<double>(std::max<size_t>(1, c.NumPoints()));
  double off = std::numeric_limits<double>::infinity();
  const auto consider = [&](const Eigen::Vector2d& x, const Eigen::Vector2d& y) {
    const double len = (y - x).norm();
    if (!(len > 0.0)) return;
    const Eigen::Vector2d n(-(y - x).y() / len, (y - x).x() / len);
    const double o = std::fabs((centre - x).dot(n));
    if (o < off) {
      off = o;
      *pa = x;
      *pc = y;
    }
  };
  for_each_pair([&](const auto& cand) {
    for (int i = 0; i < 4; ++i) {
      if ((cand[i].first - cand[i].second).norm() > best + tie) continue;
      consider(cand[i].first, cand[i].second);
      for (int j = i + 1; j < 4; ++j)
        if ((cand[j].first - cand[j].second).norm() <= best + tie)
          consider(0.5 * (cand[i].first + cand[j].first), 0.5 * (cand[i].second + cand[j].second));
    }
  });
  return std::isfinite(off);
}

// The corridor from the parent to a child `gap` away (ScopeExtensionBridge).
Polygon2D Corridor(const Polygon2D& parent, const Polygon2D& child, double gap) {
  // The parent's nearest point lies within `gap` of the child: only that window of a large
  // room ring is searched (the window's own edges are farther than the gap from the child).
  const Polygon2D near_parent = parent.Intersection(Box(child, gap + 2.0 * kBridgeSlackM));
  if (near_parent.NumComponents() == 0) return Polygon2D();
  Eigen::Vector2d pa, pc;
  if (!NearestPoints(near_parent, child, &pa, &pc)) return Polygon2D();
  const double len = (pc - pa).norm();
  if (!(len > 0.0)) return Polygon2D();
  const Eigen::Vector2d d = (pc - pa) / len;  // along the corridor
  const Eigen::Vector2d n(-d.y(), d.x());     // across it
  // The child's extent across the corridor, measured from the segment (0 is on it: pc is a
  // point of the child).
  double lo = 0.0, hi = 0.0;
  for (const auto& ring : AllRings(child))
    for (const auto& q : ring) {
      const double s = (q - pa).dot(n);
      lo = std::min(lo, s);
      hi = std::max(hi, s);
    }
  // At most the maximum width, centred on the child as far as the window still holds the
  // segment (lo <= 0 <= hi, so the clamp range is never empty).
  const double w = std::min(hi - lo, kScopeExtensionCorridorMaxWidthM);
  const double from = std::clamp(0.5 * (lo + hi) - 0.5 * w, std::max(-w, lo), std::min(0.0, hi - w));
  // kBridgeSlackM into both regions along it and past the segment across it, so the corridor
  // overlaps each region near its nearest point (whatever the corner there) and a child whose
  // width across it is zero still gets one.
  const double s0 = from - kBridgeSlackM, s1 = from + w + kBridgeSlackM;
  const double t0 = -kBridgeSlackM, t1 = len + kBridgeSlackM;
  return Polygon2D({pa + t0 * d + s0 * n, pa + t1 * d + s0 * n, pa + t1 * d + s1 * n,
                    pa + t0 * d + s1 * n});
}

}  // namespace

Polygon2D ScopeExtensionBridge(const Polygon2D& parent, const Polygon2D& child, double gap) {
  if (gap > kScopeExtensionNearbyM) return Corridor(parent, child, gap);
  const double r = gap + kBridgeSlackM;
  // The bridge lies within r of the child; the parent matters only within 2r of that
  // (a miter corner reaches at most its limit, 5r, out): clip to a window around the child.
  const Polygon2D near_parent = parent.Intersection(Box(child, 8.0 * r));
  if (near_parent.NumComponents() == 0) return Polygon2D();
  return near_parent.Inflate(r).Intersection(child.Inflate(r));
}

bool IsActorBound(const ElementTree& tree, element_id_t id) {
  const Element* self = tree.GetElement(id);
  if (!self) return false;
  if (self->GetGroundingEntityId() != spatialblend::kInvalidEntityId) return true;  // an actor's own element
  // Below an actor's element: its BODY when that element is not a place. An actor placed in a room
  // (its element is the room, a Containment/Navigable region) does not make the room's furniture an
  // actor's: a region an actor is bound to is a container, never a body.
  for (const Element* e = tree.GetElement(self->GetParentId()); e; e = tree.GetElement(e->GetParentId())) {
    if (e->GetGroundingEntityId() != spatialblend::kInvalidEntityId) return !IsRegion(e);
    if (e->GetParentId() == kInvalidElementId) break;
  }
  return false;
}

bool ExtendsParentScope(const ElementTree& tree, element_id_t id) {
  const Element* e = tree.GetElement(id);
  if (!e || !HasFootprint(e) || !IsRegion(e)) return false;
  const Element* parent = tree.GetElement(e->GetParentId());
  if (!IsRegion(parent) || !HasFootprint(parent)) return false;
  const auto pc = WorldFootprint(tree, e);
  const auto pp = WorldFootprint(tree, parent);
  if (!pc || !pp || CheaplyInside(*pc, *pp)) return false;
  const auto uc = UnionFootprintOf(*pc), up = UnionFootprintOf(*pp);
  return uc && up && OutsideArea(*uc, *up) > 0.0;
}

std::shared_ptr<Polygon2D> RegionWorldFootprint(const ElementTree& tree,
                                                element_id_t id) {
  const Element* e = tree.GetElement(id);
  return e ? WorldFootprintOf(tree, e, e->GetSpatialDefinition()) : nullptr;
}

std::shared_ptr<Polygon2D> DeclaredWorldFootprint(const ElementTree& tree,
                                                  element_id_t id) {
  const Element* e = tree.GetElement(id);
  return e ? WorldFootprint(tree, e) : nullptr;
}

std::shared_ptr<Polygon2D> UnionFootprint(const ElementTree& tree, element_id_t id) {
  const auto fp = DeclaredWorldFootprint(tree, id);
  return fp ? UnionFootprintOf(*fp) : nullptr;
}

namespace {

double DistanceToSegment(const Eigen::Vector2d& p, const Eigen::Vector2d& a,
                         const Eigen::Vector2d& b) {
  const Eigen::Vector2d d = b - a;
  const double len2 = d.squaredNorm();
  if (!(len2 > 0.0)) return (p - a).norm();
  const double t = std::clamp((p - a).dot(d) / len2, 0.0, 1.0);
  return (p - (a + t * d)).norm();
}

// The ring counter-clockwise, without a closing duplicate, repeated points or
// collinear vertices (a vertex within `delta` of the segment between its
// neighbours is not a corner of the shape).
std::vector<Eigen::Vector2d> CornerRing(std::vector<Eigen::Vector2d> ring,
                                        double delta) {
  double twice_area = 0.0;
  for (size_t i = 0; i < ring.size(); ++i) {
    const Eigen::Vector2d& u = ring[i];
    const Eigen::Vector2d& v = ring[(i + 1) % ring.size()];
    twice_area += u.x() * v.y() - v.x() * u.y();
  }
  if (twice_area < 0.0) ring = std::vector<Eigen::Vector2d>(ring.rbegin(), ring.rend());
  for (bool changed = true; changed && ring.size() >= 3;) {
    changed = false;
    for (size_t i = 0; i < ring.size(); ++i) {
      const Eigen::Vector2d& prev = ring[(i + ring.size() - 1) % ring.size()];
      const Eigen::Vector2d& next = ring[(i + 1) % ring.size()];
      if ((ring[i] - prev).norm() <= delta ||
          DistanceToSegment(ring[i], prev, next) <= delta) {
        ring.erase(ring.begin() + static_cast<std::ptrdiff_t>(i));
        changed = true;
        break;
      }
    }
  }
  return ring;
}

}  // namespace

bool FootprintsCongruent(const Polygon2D& a, const Polygon2D& b) {
  if (a.NumPoints() < 3 || b.NumPoints() < 3 || !a.ExtraRings().empty() ||
      !b.ExtraRings().empty())
    return false;
  const double smaller = std::min(a.GetArea(), b.GetArea());
  if (!(smaller > 0.0)) return false;
  // The geometry epsilon relative to the smaller footprint's area (as the layout
  // checks apply it) is, as a length, epsilon times the footprint's own scale.
  const double delta = kEpsilon * std::sqrt(smaller);
  const std::vector<Eigen::Vector2d> ra = CornerRing(a.GetPoints(), delta);
  const std::vector<Eigen::Vector2d> rb = CornerRing(b.GetPoints(), delta);
  const size_t n = ra.size();
  if (n < 3 || rb.size() != n) return false;
  // Both rings run counter-clockwise, so a proper rotation plus translation maps one
  // onto the other exactly when some cyclic shift of the corners does: a mirror image
  // matches no shift.
  for (size_t k = 0; k < n; ++k) {
    const Eigen::Vector2d ea = ra[1] - ra[0];
    const Eigen::Vector2d eb = rb[(k + 1) % n] - rb[k];
    if (std::fabs(ea.norm() - eb.norm()) > delta) continue;
    const double theta = std::atan2(eb.y(), eb.x()) - std::atan2(ea.y(), ea.x());
    const double cs = std::cos(theta), sn = std::sin(theta);
    bool all = true;
    for (size_t i = 0; i < n && all; ++i) {
      const Eigen::Vector2d d = ra[i] - ra[0];
      const Eigen::Vector2d mapped(rb[k].x() + cs * d.x() - sn * d.y(),
                                   rb[k].y() + sn * d.x() + cs * d.y());
      all = (mapped - rb[(k + i) % n]).norm() <= delta;
    }
    if (all) return true;
  }
  return false;
}

std::optional<ContainmentViolation> FindContainmentViolation(
    const ElementTree& tree, element_id_t id, bool* extends_out) {
  bool extends = false;
  std::vector<element_id_t> ids = tree.GetSubtreeIds(id);
  // The element's own parent and siblings matter too (GetSubtreeIds includes id).
  for (element_id_t eid : ids) {
    const Element* e = tree.GetElement(eid);
    if (!HasFootprint(e)) continue;
    const Element* parent = tree.GetElement(e->GetParentId());
    if (IsRegion(e) && IsRegion(parent) && HasFootprint(parent)) {
      if (auto v = CheckParent(tree, e, parent, &extends)) return v;
    }
    if (!parent) continue;
    const auto pe = WorldFootprint(tree, e);
    for (element_id_t sid : tree.GetDirectChildIds(parent->GetId())) {
      if (sid == eid) continue;
      const Element* sib = tree.GetElement(sid);
      if (!HasFootprint(sib)) continue;
      // Each unordered pair is reported once, lower id first.
      const Element* lo = eid < sid ? e : sib;
      const Element* hi = eid < sid ? sib : e;
      const auto plo = WorldFootprint(tree, lo), phi = WorldFootprint(tree, hi);
      if (!plo || !phi) continue;
      if (auto v = CheckOverlap(lo, *plo, hi, *phi, "sibling footprints overlap: ")) return v;
    }
  }
  // An extending region and the bridge that joins it to its parent are the parent's scope outside
  // the parent: neither may overlap any footprint outside the region's own line. Checked for every
  // pair with an element of the subtree on one side and an extending element on the other (either
  // side: a moved region's extension, and a region placed into an existing extension or bridge).
  // Without an extension anywhere, this is empty.
  if (const Element* root = tree.GetRoot()) {
    std::vector<element_id_t> extending;
    for (element_id_t x : tree.GetSubtreeIds(root->GetId()))
      if (!IsActorBound(tree, x) && ExtendsParentScope(tree, x)) extending.push_back(x);
    for (element_id_t x : extending) {
      const Element* ex = tree.GetElement(x);
      const auto px = UnionFootprint(tree, x);
      const auto pp = UnionFootprint(tree, ex->GetParentId());
      if (!px || !pp) continue;
      // The bridge, for a region that does not overlap its parent (ScopeExtensionBridge).
      std::optional<Polygon2D> bridge;
      if (!(px->IntersectionArea(*pp) > kEpsilon * px->GetArea()))
        bridge = ScopeExtensionBridge(*pp, *px, px->DistanceTo(*pp));
      const bool x_in_subtree = std::find(ids.begin(), ids.end(), x) != ids.end();
      const std::vector<element_id_t> others =
          x_in_subtree ? tree.GetSubtreeIds(root->GetId()) : ids;
      for (element_id_t o : others) {
        if (InLine(tree, x, o)) continue;
        const Element* eo = tree.GetElement(o);
        if (!HasFootprint(eo)) continue;
        const auto po = WorldFootprint(tree, eo);
        if (!po) continue;
        // Siblings' footprints were checked above.
        if (eo->GetParentId() != ex->GetParentId()) {
          const Element* lo = x < o ? ex : eo;
          const Element* hi = x < o ? eo : ex;
          if (auto v = CheckOverlap(lo, *(x < o ? px : po), hi, *(x < o ? po : px),
                                    "an extended region overlaps a footprint outside its line: "))
            return v;
        }
        // The bridge outside the parent (inside it, the parent's region is the parent's already),
        // against every footprint outside the region's line: the parent's other children and
        // their descendants too (a sibling or an obstacle the corridor would cross).
        if (bridge && bridge->NumComponents() > 0) {
          // A bridge reaches kBridgeSlackM past the regions and past the corridor's sides (a
          // band: r into both regions): a footprint that only touches the parent, the region or
          // the corridor's side meets it in a sliver no wider than that, which the erosion
          // removes (a neighbour of the region or of the corridor is no overlap).
          Polygon2D common = bridge->Intersection(*po);
          if (common.NumComponents() == 0) continue;
          common = common.Inflate(-kBridgeSlackM);
          const double outside_parent =
              common.NumComponents() == 0 ? 0.0 : common.GetArea() - common.IntersectionArea(*pp);
          if (outside_parent > kBridgeSlackM * kBridgeSlackM * 10.0) {
            ContainmentViolation v;
            v.first = x;
            v.second = o;
            v.code = SceneErrorCode::kContainmentOverlap;
            v.message = "the bridge that joins " + Name(ex) + " to its parent overlaps " + Name(eo) +
                        " (" + std::to_string(outside_parent) +
                        " m^2); an extended scope may not reach into another footprint";
            return v;
          }
        }
      }
    }
  }
  if (extends_out) *extends_out = extends;
  return std::nullopt;
}

}  // namespace spatialblend
