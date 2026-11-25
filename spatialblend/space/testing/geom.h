// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Plain 2D geometry used by the scenario harness (independent of the core's
// Polygon2D, so the oracle side of every metric does not share code with the
// system under test).
#ifndef SPATIALBLEND_SIMULATOR_SCENARIO_GEOM_H_
#define SPATIALBLEND_SIMULATOR_SCENARIO_GEOM_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>

namespace spatialblend {
namespace scn {

using Vec2 = Eigen::Vector2d;
using Poly = std::vector<Vec2>;

struct Pose2 {
  double x = 0, y = 0, yaw = 0;
  Vec2 p() const { return Vec2(x, y); }
};

inline double Wrap(double a) { return std::remainder(a, 2.0 * M_PI); }

inline Vec2 ApplyPose(const Pose2& t, const Vec2& v) {
  const double c = std::cos(t.yaw), s = std::sin(t.yaw);
  return Vec2(t.x + c * v.x() - s * v.y(), t.y + s * v.x() + c * v.y());
}
// a o b: apply b first, then a.
inline Pose2 Compose(const Pose2& a, const Pose2& b) {
  const Vec2 p = ApplyPose(a, b.p());
  return {p.x(), p.y(), Wrap(a.yaw + b.yaw)};
}
inline Pose2 Inverse(const Pose2& a) {
  const double c = std::cos(a.yaw), s = std::sin(a.yaw);
  return {-(c * a.x + s * a.y), -(-s * a.x + c * a.y), Wrap(-a.yaw)};
}
inline Eigen::Matrix4d PoseMat(const Pose2& t, double z = 0.0) {
  Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
  m(0, 0) = std::cos(t.yaw);
  m(0, 1) = -std::sin(t.yaw);
  m(1, 0) = std::sin(t.yaw);
  m(1, 1) = std::cos(t.yaw);
  m(0, 3) = t.x;
  m(1, 3) = t.y;
  m(2, 3) = z;
  return m;
}

inline Poly Box(double hx, double hy) {
  return {{-hx, -hy}, {hx, -hy}, {hx, hy}, {-hx, hy}};
}
inline Poly Scale(const Poly& p, double kx, double ky) {
  Poly o;
  for (const auto& v : p) o.push_back(Vec2(v.x() * kx, v.y() * ky));
  return o;
}
inline Poly Rotate(const Poly& p, double yaw) {
  return [&] {
    Poly o;
    for (const auto& v : p) o.push_back(ApplyPose({0, 0, yaw}, v));
    return o;
  }();
}
inline Poly Transform(const Poly& p, const Pose2& t) {
  Poly o;
  for (const auto& v : p) o.push_back(ApplyPose(t, v));
  return o;
}

inline bool PointInPoly(const Poly& poly, const Vec2& q) {
  bool in = false;
  const size_t n = poly.size();
  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    const Vec2 &a = poly[i], &b = poly[j];
    if ((a.y() > q.y()) != (b.y() > q.y()) &&
        q.x() < (b.x() - a.x()) * (q.y() - a.y()) / (b.y() - a.y()) + a.x())
      in = !in;
  }
  return in;
}
inline double DistToSegment(const Vec2& p, const Vec2& a, const Vec2& b) {
  const Vec2 ab = b - a;
  const double l2 = ab.squaredNorm();
  const double t = l2 < 1e-18 ? 0.0 : std::clamp((p - a).dot(ab) / l2, 0.0, 1.0);
  return (p - (a + t * ab)).norm();
}
// Distance to the polygon boundary (>= 0).
inline double BoundaryDist(const Poly& poly, const Vec2& p) {
  double d = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < poly.size(); ++i)
    d = std::min(d, DistToSegment(p, poly[i], poly[(i + 1) % poly.size()]));
  return d;
}
// Negative inside.
inline double SignedDist(const Poly& poly, const Vec2& p) {
  const double d = BoundaryDist(poly, p);
  return PointInPoly(poly, p) ? -d : d;
}
inline double Area(const Poly& p) {
  double a = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    const Vec2 &u = p[i], &v = p[(i + 1) % p.size()];
    a += u.x() * v.y() - v.x() * u.y();
  }
  return 0.5 * a;
}
inline Vec2 Centroid(const Poly& p) {
  const double a = Area(p);
  if (std::fabs(a) < 1e-12) {
    Vec2 c = Vec2::Zero();
    for (const auto& v : p) c += v;
    return c / std::max<size_t>(1, p.size());
  }
  Vec2 c = Vec2::Zero();
  for (size_t i = 0; i < p.size(); ++i) {
    const Vec2 &u = p[i], &v = p[(i + 1) % p.size()];
    const double w = u.x() * v.y() - v.x() * u.y();
    c += (u + v) * w;
  }
  return c / (6.0 * a);
}
inline Poly EnsureCcw(Poly p) {
  if (Area(p) < 0) std::reverse(p.begin(), p.end());
  return p;
}
inline void Bounds(const Poly& p, Vec2* lo, Vec2* hi) {
  *lo = Vec2(1e18, 1e18);
  *hi = Vec2(-1e18, -1e18);
  for (const auto& v : p) {
    *lo = lo->cwiseMin(v);
    *hi = hi->cwiseMax(v);
  }
}
inline double Circumradius(const Poly& p, const Vec2& c) {
  double r = 0;
  for (const auto& v : p) r = std::max(r, (v - c).norm());
  return r;
}

// Minimum-area oriented bounding box of a polygon (1 degree search). Returns
// the box polygon in its own frame and the pose that places it in the polygon's
// frame.
inline Poly OrientedBox(const Poly& p, Pose2* pose) {
  double best = 1e30, by = 0;
  Vec2 blo, bhi;
  for (int d = 0; d < 90; ++d) {
    const double yaw = d * M_PI / 180.0;
    const Poly q = Rotate(p, -yaw);
    Vec2 lo, hi;
    Bounds(q, &lo, &hi);
    const double a = (hi.x() - lo.x()) * (hi.y() - lo.y());
    if (a < best) {
      best = a;
      by = yaw;
      blo = lo;
      bhi = hi;
    }
  }
  const Vec2 c = 0.5 * (blo + bhi);
  *pose = {ApplyPose({0, 0, by}, c).x(), ApplyPose({0, 0, by}, c).y(), by};
  return Box(0.5 * (bhi.x() - blo.x()), 0.5 * (bhi.y() - blo.y()));
}

// Wrapped angle-sampled helpers.
inline double Lerp(double a, double b, double t) { return a + (b - a) * t; }

}  // namespace scn
}  // namespace spatialblend

#endif  // SPATIALBLEND_SIMULATOR_SCENARIO_GEOM_H_
