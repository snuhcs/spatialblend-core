// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_BASE_MATH_UTIL_H_
#define SPATIALBLEND_BASE_MATH_UTIL_H_

#include <Eigen/Dense>
#include <optional>
#include <vector>


namespace spatialblend {
namespace math {

namespace Constants {
constexpr double kEpsilon = 1e-6;
// Use Eigen's pi to avoid _USE_MATH_DEFINES requirement
constexpr double kPi = EIGEN_PI;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kHalfPi = kPi / 2.0;
}  // namespace Constants

double ToRadians(double degrees);
double ToDegrees(double radians);
Eigen::Vector2d GetDirectionFromYaw(double yaw);
Eigen::Vector3d GetDirectionFromYawPitch(Eigen::Vector2d yaw_pitch);
Eigen::Vector3d GetDirectionFromYawPitch(double yaw, double pitch);
Eigen::Vector2d GetYawPitchFromDirection(Eigen::Vector3d direction);
// Wraps an angle to the range [-π, π]
double NormalizeRadians(double angle);
double NormalizeDegrees(double angle);
double ComputeRotationTime(double start_angle, double target_angle,
                           double angular_velocity);

// Quaternion utility functions
Eigen::Quaterniond YawToQuaternion(double yaw_degrees);
Eigen::Quaterniond YawPitchToQuaternion(double yaw_degrees,
                                        double pitch_degrees);
Eigen::Quaterniond YawPitchToQuaternion(const Eigen::Vector2d& yaw_pitch);
double QuaternionToYaw(const Eigen::Quaterniond& quat);
Eigen::Vector2d QuaternionToYawPitch(const Eigen::Quaterniond& quat);
Eigen::Vector3d QuaternionToDirection(const Eigen::Quaterniond& quat);
Eigen::Quaterniond DirectionToQuaternion(const Eigen::Vector3d& direction);
double QuaternionAngularDistance(const Eigen::Quaterniond& q1,
                                 const Eigen::Quaterniond& q2);
double ComputeQuaternionRotationTime(const Eigen::Quaterniond& start_quat,
                                     const Eigen::Quaterniond& target_quat,
                                     double angular_velocity);


template <typename T>
std::string EigenToString(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vec) {
  std::stringstream ss;
  ss << vec.transpose();
  return ss.str();
}

// Returns the intersection of a line segment and a ray, or nullopt if they
// don't intersect.
std::optional<Eigen::Vector2d> LineSegmentRayIntersection(
    const Eigen::Vector2d& p1, const Eigen::Vector2d& p2,
    const Eigen::Vector2d& origin, const Eigen::Vector2d& direction);

// a and b must be normalized
inline Eigen::Quaterniond LookAt(const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b) {
  Eigen::Vector3d forward = (b - a).normalized();
  Eigen::Vector3d right = forward.cross(Eigen::Vector3d::UnitZ()).normalized();
  Eigen::Vector3d up = right.cross(forward).normalized();
  Eigen::Matrix3d rotation;
  rotation << right, up, forward;
  return Eigen::Quaterniond(rotation);
}

// Conversions between Rotation Matrix and Euler Angles (ZYX order)
inline Eigen::Vector3d RotationMatrixToEuler(const Eigen::Matrix3d& R) {
  double sy = std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
  bool singular = sy < 1e-6;
  double x, y, z;
  if (!singular) {
    x = std::atan2(R(2, 1), R(2, 2));
    y = std::atan2(-R(2, 0), sy);
    z = std::atan2(R(1, 0), R(0, 0));

    // Normalize yaw (z) to [0, 2π) range to preserve the original angle
    if (z < 0) {
      z += Constants::kTwoPi;
    }
  } else {
    x = std::atan2(-R(1, 2), R(1, 1));
    y = std::atan2(-R(2, 0), sy);
    z = 0;
  }
  return {x, y, z};
}

inline Eigen::Matrix3d EulerToRotationMatrix(const Eigen::Vector3d& euler) {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  R = Eigen::AngleAxisd(euler.z(), Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(euler.y(), Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(euler.x(), Eigen::Vector3d::UnitX());
  return R;
}

}  // namespace math
}  // namespace spatialblend

#endif /* SPATIALBLEND_BASE_MATH_UTIL_H_ */
