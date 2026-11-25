// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/geometry/path/bezier_course.h
//
// The smoothed course of v1's single-target planner (spatialblend/geometry/path/
// spline_generator.cc, GenerateSafeSplineImpl): one cubic Bezier per path
// segment through the waypoints of the geodesic course, with the tangent at
// each waypoint the mean of the two adjacent segment directions, the start
// tangent the body's heading and the end tangent the mapped entering
// direction. The receiver walks 1/n of the course's arc length (v1
// BaseSingleTargetPlanner::CalculateDistanceToMove) and arrives at the contact
// along the entering direction.
//
// v1 keeps each segment's control points inside the convex cell of the
// segment's start and clamps them there. The v2 cells ignore obstacle
// footprints and their clearance, so the clamp is not the guarantee here; the
// caller checks the curve against its free space (Samples) and falls back to
// the discrete course, as v1 does when the generator declines.
//
// The course lives in the scope of the relation (relation_scope.h): a planar
// floor (Dim = 2) or a region volume (Dim = 3). The construction is the same in
// every dimension; BezierCourse is the planar one.
#ifndef SPATIALBLEND_PATH_BEZIER_COURSE_H_
#define SPATIALBLEND_PATH_BEZIER_COURSE_H_

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace spatialblend {

template <int Dim>
class BezierCourseT {
 public:
  using Vec = Eigen::Matrix<double, Dim, 1>;
  // Waypoints of the discrete course, first = start, last = goal. Segments of
  // zero length are skipped (v1). Nullopt when fewer than two distinct
  // waypoints remain.
  static std::optional<BezierCourseT> Build(const std::vector<Vec>& waypoints,
                                            const std::optional<Vec>& start_direction,
                                            const std::optional<Vec>& end_direction);

  // Total arc length of the curve.
  double Length() const { return length_; }
  // The point `arc` along the curve, clamped to [0, Length()].
  Vec PointAfter(double arc) const;
  // The curve as a polyline, `per_segment` chords per Bezier segment.
  std::vector<Vec> Samples(int per_segment) const;

 private:
  struct Segment {
    Vec p[4];
    double length = 0.0;
  };
  // v1 SplineGenerator: the control points sit at this fraction of the segment
  // length along the tangent.
  static constexpr double kTangentFactor = 0.3;
  // Chords per segment of the arc-length integration (v1 SplinePath::kNumSamples
  // integrates by chords as well).
  static constexpr int kChordsPerSegment = 256;
  static constexpr double kDegenerate = 1e-9;
  static Vec Unit(const Vec& v) { return v / v.norm(); }
  static Vec Evaluate(const Segment& s, double t);
  std::vector<Segment> segments_;
  double length_ = 0.0;
};
using BezierCourse = BezierCourseT<2>;

template <int Dim>
typename BezierCourseT<Dim>::Vec BezierCourseT<Dim>::Evaluate(const Segment& s, double t) {
  const double u = 1.0 - t;
  return u * u * u * s.p[0] + 3.0 * u * u * t * s.p[1] +
         3.0 * u * t * t * s.p[2] + t * t * t * s.p[3];
}

template <int Dim>
std::optional<BezierCourseT<Dim>> BezierCourseT<Dim>::Build(
    const std::vector<Vec>& points,
    const std::optional<Vec>& start_direction,
    const std::optional<Vec>& end_direction) {
  const size_t n = points.size();
  if (n < 2) return std::nullopt;
  BezierCourseT course;
  for (size_t i = 0; i + 1 < n; ++i) {
    const Vec& p0 = points[i];
    const Vec& p3 = points[i + 1];
    const Vec delta = p3 - p0;
    const double seg_len = delta.norm();
    if (seg_len <= kDegenerate) continue;

    Vec t_start;
    if (i == 0 && start_direction && start_direction->norm() > kDegenerate) {
      t_start = Unit(*start_direction);
    } else if (i == 0) {
      t_start = Unit(delta);
    } else {
      const Vec prev = p0 - points[i - 1];
      if (prev.norm() <= kDegenerate) {
        t_start = Unit(delta);
      } else {
        const Vec mean = Unit(prev) + Unit(delta);
        t_start = mean.norm() <= kDegenerate ? Unit(delta) : Unit(mean);
      }
    }
    Vec t_end;
    if (i + 1 == n - 1 && end_direction && end_direction->norm() > kDegenerate) {
      t_end = Unit(*end_direction);
    } else if (i + 1 == n - 1) {
      t_end = Unit(delta);
    } else {
      const Vec next = points[i + 2] - p3;
      if (next.norm() <= kDegenerate) {
        t_end = Unit(delta);
      } else {
        const Vec mean = Unit(delta) + Unit(next);
        t_end = mean.norm() <= kDegenerate ? Unit(delta) : Unit(mean);
      }
    }
    Segment seg;
    const double handle = kTangentFactor * seg_len;
    seg.p[0] = p0;
    seg.p[1] = p0 + handle * t_start;
    seg.p[2] = p3 - handle * t_end;
    seg.p[3] = p3;
    Vec prev = seg.p[0];
    for (int k = 1; k <= kChordsPerSegment; ++k) {
      const Vec cur =
          Evaluate(seg, static_cast<double>(k) / kChordsPerSegment);
      seg.length += (cur - prev).norm();
      prev = cur;
    }
    course.length_ += seg.length;
    course.segments_.push_back(seg);
  }
  if (course.segments_.empty()) return std::nullopt;
  return course;
}

template <int Dim>
typename BezierCourseT<Dim>::Vec BezierCourseT<Dim>::PointAfter(double arc) const {
  if (!(arc > 0.0)) return segments_.front().p[0];
  double walked = 0.0;
  for (const Segment& seg : segments_) {
    if (walked + seg.length < arc) {
      walked += seg.length;
      continue;
    }
    // Walk this segment's chords to the remaining arc.
    double remaining = arc - walked;
    Vec prev = seg.p[0];
    for (int k = 1; k <= kChordsPerSegment; ++k) {
      const Vec cur =
          Evaluate(seg, static_cast<double>(k) / kChordsPerSegment);
      const double chord = (cur - prev).norm();
      if (chord >= remaining && chord > 0.0)
        return prev + (cur - prev) * (remaining / chord);
      remaining -= chord;
      prev = cur;
    }
    return seg.p[3];
  }
  return segments_.back().p[3];
}

template <int Dim>
std::vector<typename BezierCourseT<Dim>::Vec> BezierCourseT<Dim>::Samples(int per_segment) const {
  std::vector<Vec> out;
  out.push_back(segments_.front().p[0]);
  for (const Segment& seg : segments_)
    for (int k = 1; k <= per_segment; ++k)
      out.push_back(Evaluate(seg, static_cast<double>(k) / per_segment));
  return out;
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_PATH_BEZIER_COURSE_H_
