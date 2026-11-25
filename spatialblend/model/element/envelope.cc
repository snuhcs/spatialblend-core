// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/envelope.h"

#include <algorithm>
#include <cmath>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"

namespace spatialblend {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMinExtent = 1e-4;

// Append a CCW circular arc (segments+1 points spanning [a0, a1]) of radius r
// centred at c into out.
void AppendArc(std::vector<Eigen::Vector2d>& out, const Eigen::Vector2d& c,
               double r, double a0, double a1, int segments) {
  for (int i = 0; i <= segments; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(segments);
    const double a = a0 + (a1 - a0) * t;
    out.emplace_back(c.x() + r * std::cos(a), c.y() + r * std::sin(a));
  }
}

std::shared_ptr<SpatialDefinition> WrapFootprint(
    std::vector<Eigen::Vector2d> pts) {
  if (pts.size() < 3) return nullptr;
  auto polygon = std::make_shared<Polygon2D>(std::move(pts));
  return std::make_shared<NavigableSpatialDefinition>(
      polygon, /*wall=*/nullptr, /*agent_radius=*/0.0, GeometryRole::kContainment);
}

}  // namespace

std::shared_ptr<SpatialDefinition> MakeEnvelope(ColliderType type,
                                                const std::vector<double>& p,
                                                int circle_segments,
                                                double yaw) {
  const int seg = std::max(circle_segments, 4);

  switch (type) {
    case ColliderType::kBox: {
      if (p.size() < 6) return nullptr;
      const double cx = p[0], cy = p[1];
      const double hx = std::abs(p[3]) * 0.5, hy = std::abs(p[4]) * 0.5;
      if (hx < kMinExtent || hy < kMinExtent) return nullptr;
      // Corners in the box's own local frame, then rotated by yaw about the
      // center and translated into world — matches harness/scenario.cc's
      // polygon-branch rotation convention (wx = x + lx*c - ly*s, wy = y +
      // lx*s + ly*c) so box and polygon footprints agree on handedness.
      const double c = std::cos(yaw), s = std::sin(yaw);
      auto rotate = [&](double lx, double ly) -> Eigen::Vector2d {
        return {cx + lx * c - ly * s, cy + lx * s + ly * c};
      };
      std::vector<Eigen::Vector2d> pts = {
          rotate(-hx, -hy),
          rotate(hx, -hy),
          rotate(hx, hy),
          rotate(-hx, hy),
      };
      return WrapFootprint(std::move(pts));
    }

    case ColliderType::kSphere: {
      if (p.size() < 4) return nullptr;
      const Eigen::Vector2d c(p[0], p[1]);
      const double r = std::abs(p[3]);
      if (r < kMinExtent) return nullptr;
      std::vector<Eigen::Vector2d> pts;
      pts.reserve(seg);
      // Full circle: seg points (do not duplicate the closing point).
      for (int i = 0; i < seg; ++i) {
        const double a = 2.0 * kPi * static_cast<double>(i) / seg;
        pts.emplace_back(c.x() + r * std::cos(a), c.y() + r * std::sin(a));
      }
      return WrapFootprint(std::move(pts));
    }

    case ColliderType::kCapsule: {
      if (p.size() < 6) return nullptr;
      const Eigen::Vector2d c(p[0], p[1]);
      const double r = std::abs(p[3]);
      const double height = std::abs(p[4]);
      const int axis = static_cast<int>(p[5]);
      if (r < kMinExtent) return nullptr;

      // Footprint depends on the capsule axis. A capsule whose axis is the up
      // direction (Z, local) projects to a CIRCLE; a horizontal axis projects to
      // a STADIUM (rectangle + two semicircular caps) along that axis.
      if (axis == 2) {  // axis = Z (up) -> circle footprint of radius r
        std::vector<Eigen::Vector2d> pts;
        pts.reserve(seg);
        for (int i = 0; i < seg; ++i) {
          const double a = 2.0 * kPi * static_cast<double>(i) / seg;
          pts.emplace_back(c.x() + r * std::cos(a), c.y() + r * std::sin(a));
        }
        return WrapFootprint(std::move(pts));
      }

      // Horizontal axis (0=X, 1=Y): stadium. Cylinder half-length along the axis.
      const double half_len = std::max(height * 0.5 - r, 0.0);
      const int caps = std::max(seg / 2, 2);
      std::vector<Eigen::Vector2d> pts;
      if (axis == 0) {  // axis = X: caps centred at (cx +/- half_len, cy)
        const Eigen::Vector2d right(c.x() + half_len, c.y());
        const Eigen::Vector2d left(c.x() - half_len, c.y());
        AppendArc(pts, right, r, -kPi * 0.5, kPi * 0.5, caps);  // right cap
        AppendArc(pts, left, r, kPi * 0.5, kPi * 1.5, caps);    // left cap
      } else {  // axis = Y: caps centred at (cx, cy +/- half_len)
        const Eigen::Vector2d top(c.x(), c.y() + half_len);
        const Eigen::Vector2d bot(c.x(), c.y() - half_len);
        AppendArc(pts, top, r, 0.0, kPi, caps);              // top cap
        AppendArc(pts, bot, r, kPi, 2.0 * kPi, caps);        // bottom cap
      }
      return WrapFootprint(std::move(pts));
    }
  }
  return nullptr;
}

}  // namespace spatialblend
