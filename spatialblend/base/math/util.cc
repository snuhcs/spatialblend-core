// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/base/math/util.h"

#include <cmath>

namespace spatialblend {
namespace math {

double ToRadians(double degrees) { return degrees * Constants::kPi / 180.0f; }
double ToDegrees(double radians) { return radians * 180.0 / Constants::kPi; }

Eigen::Vector2d GetDirectionFromYaw(double yaw) {
  double yaw_radians = ToRadians(yaw);
  return Eigen::Vector2d(cos(yaw_radians), sin(yaw_radians));
}

Eigen::Vector3d GetDirectionFromYawPitch(Eigen::Vector2d yaw_pitch) {
  return GetDirectionFromYawPitch(yaw_pitch[0], yaw_pitch[1]);
}

Eigen::Vector3d GetDirectionFromYawPitch(double yaw, double pitch) {
  double yaw_radians = ToRadians(yaw);
  double pitch_radians = ToRadians(pitch);
  double x = cos(yaw_radians) * cos(pitch_radians);
  double y = sin(yaw_radians) * cos(pitch_radians);
  double z = sin(pitch_radians);
  return Eigen::Vector3d(x, y, z);
}

Eigen::Vector2d GetYawPitchFromDirection(Eigen::Vector3d direction) {
  double yaw = ToDegrees(atan2(direction[1], direction[0]));
  double pitch =
      ToDegrees(atan2(direction[2], sqrt(direction[0] * direction[0] +
                                         direction[1] * direction[1])));
  return Eigen::Vector2d(yaw, pitch);
}

double NormalizeRadians(double angle) {
  if (!std::isfinite(angle)) return angle;
  return -std::fmod(-angle + Constants::kPi, Constants::kTwoPi) +
         Constants::kPi;
}

double NormalizeDegrees(double angle) {
  if (!std::isfinite(angle)) return angle;
  // Normalize to (-180, 180] range
  double normalized = std::fmod(angle + 180.0, 360.0);
  if (normalized < 0) {
    normalized += 360.0;
  }
  return normalized - 180.0;
}

double AbsoluteDifferenceRadians(double a, double b) {
  return std::abs(NormalizeRadians(NormalizeRadians(a) - NormalizeRadians(b)));
}

double ComputeRotationTime(double start_angle, double target_angle,
                           double angular_velocity) {
  if (std::abs(angular_velocity) < math::Constants::kEpsilon) {
    return std::numeric_limits<double>::infinity();
  }

  start_angle = NormalizeRadians(start_angle);
  target_angle = NormalizeRadians(target_angle);

  double angle_diff = NormalizeRadians(target_angle - start_angle);

  // If angular_velocity is negative, we're rotating counterclockwise
  // Need to adjust the angle difference accordingly
  if (angular_velocity < 0) {
    if (angle_diff > 0) {
      angle_diff -= Constants::kTwoPi;
    }
  } else {
    if (angle_diff < 0) {
      angle_diff += Constants::kTwoPi;
    }
  }

  double t = angle_diff / angular_velocity;
  return t > 0 ? t : std::numeric_limits<double>::infinity();
}

// Quaternion utility functions
Eigen::Quaterniond YawToQuaternion(double yaw_degrees) {
  double yaw_radians = ToRadians(yaw_degrees);
  return Eigen::Quaterniond(
      Eigen::AngleAxisd(yaw_radians, Eigen::Vector3d::UnitZ()));
}

Eigen::Quaterniond YawPitchToQuaternion(double yaw_degrees,
                                        double pitch_degrees) {
  double yaw_radians = ToRadians(yaw_degrees);
  double pitch_radians = ToRadians(pitch_degrees);

  // Apply yaw first (around Z), then pitch (around Y)
  Eigen::Quaterniond yaw_quat(
      Eigen::AngleAxisd(yaw_radians, Eigen::Vector3d::UnitZ()));
  // We negate the pitch angle to align with the convention that positive pitch
  // means moving towards positive Z.
  Eigen::Quaterniond pitch_quat(
      Eigen::AngleAxisd(-pitch_radians, Eigen::Vector3d::UnitY()));

  return yaw_quat * pitch_quat;
}

Eigen::Quaterniond YawPitchToQuaternion(const Eigen::Vector2d& yaw_pitch) {
  return YawPitchToQuaternion(yaw_pitch[0], yaw_pitch[1]);
}

double QuaternionToYaw(const Eigen::Quaterniond& quat) {
  // Project the quaternion's forward vector onto the XY plane and find the
  // angle.
  Eigen::Vector3d forward = quat * Eigen::Vector3d::UnitX();
  // We only care about the yaw, so we use the x and y components.
  return ToDegrees(std::atan2(forward.y(), forward.x()));
}

Eigen::Vector2d QuaternionToYawPitch(const Eigen::Quaterniond& quat) {
  // Get the direction vector that the quaternion is "pointing" to.
  Eigen::Vector3d direction = quat * Eigen::Vector3d::UnitX();
  // Convert the direction vector to yaw and pitch angles.
  return GetYawPitchFromDirection(direction);
}

Eigen::Vector3d QuaternionToDirection(const Eigen::Quaterniond& quat) {
  // Apply quaternion rotation to the forward direction (1, 0, 0)
  return quat * Eigen::Vector3d::UnitX();
}

Eigen::Quaterniond DirectionToQuaternion(const Eigen::Vector3d& direction) {
  // Use FromTwoVectors for a robust and direct conversion from a direction
  // vector.
  return Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitX(),
                                            direction.normalized());
}

double QuaternionAngularDistance(const Eigen::Quaterniond& q1,
                                 const Eigen::Quaterniond& q2) {
  // Angular distance between two quaternions
  double dot_product = std::abs(q1.dot(q2));
  // Clamp to avoid numerical issues with acos
  dot_product = std::min(1.0, dot_product);
  return 2.0 * acos(dot_product);
}

double ComputeQuaternionRotationTime(const Eigen::Quaterniond& start_quat,
                                     const Eigen::Quaterniond& target_quat,
                                     double angular_velocity) {
  if (std::abs(angular_velocity) < Constants::kEpsilon) {
    return std::numeric_limits<double>::infinity();
  }

  double angular_distance = QuaternionAngularDistance(start_quat, target_quat);
  return angular_distance / std::abs(angular_velocity);
}

// Returns the intersection of a line segment and a ray, or nullopt if they
// don't intersect.
std::optional<Eigen::Vector2d> LineSegmentRayIntersection(
    const Eigen::Vector2d& p1, const Eigen::Vector2d& p2,
    const Eigen::Vector2d& origin, const Eigen::Vector2d& direction) {
  auto v = p2 - p1;
  auto w = origin - p1;
  auto d_perp = Eigen::Vector2d(-direction.y(), direction.x());
  auto v_perp = Eigen::Vector2d(-v.y(), v.x());

  float det = v.dot(d_perp);
  if (std::abs(det) < 1e-6) {
    return std::nullopt;  // Parallel
  }

  float t = w.dot(d_perp) / det;
  float s = w.dot(v_perp) / det;

  if (t >= 0 && t <= 1 && s >= 0) {
    return origin + s * direction;
  }

  return std::nullopt;
}

}  // namespace math
}  // namespace spatialblend