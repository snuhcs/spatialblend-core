// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/relation/avatar_facing_kernel.h"

#include <cmath>

namespace spatialblend {

GroundedRelation AvatarFacingKernel::Encode(
    const Eigen::Vector3d& anchor_world_position, double anchor_world_yaw,
    const Support& support) const {
  // Use contact point as the bearing target when available, otherwise referent center
  Eigen::Vector2d target;
  if (support.has_contact) {
    // Transform contact_point_local back to world frame
    Eigen::Vector4d contact_world = support.world_transform * Eigen::Vector4d(
        support.contact_point_local.x(), support.contact_point_local.y(), 0.0, 1.0);
    target = Eigen::Vector2d(contact_world.x(), contact_world.y());
  } else {
    target = support.GetPosition2D();
  }

  double dx = target.x() - anchor_world_position.x();
  double dy = target.y() - anchor_world_position.y();
  double distance = std::sqrt(dx * dx + dy * dy);

  double bearing = std::atan2(dy, dx);
  double bearing_offset = std::remainder(bearing - anchor_world_yaw, 2 * M_PI);

  if (Is3DRotation()) {
    double target_z;
    if (support.has_contact) {
      Eigen::Vector4d contact_world = support.world_transform * Eigen::Vector4d(
          support.contact_point_local.x(), support.contact_point_local.y(), 0.0, 1.0);
      target_z = contact_world.z();
    } else {
      target_z = support.world_transform(2, 3);
    }
    double dz = target_z - anchor_world_position.z();
    double horiz_dist = std::sqrt(dx * dx + dy * dy);
    double elevation_offset = std::atan2(dz, horiz_dist);
    double dist_3d = std::sqrt(dx * dx + dy * dy + dz * dz);

    GroundedRelation gr;
    gr.encoded_state = Eigen::VectorXd(3);
    gr.encoded_state << bearing_offset, elevation_offset, dist_3d;
    gr.proximity = (1.0 + std::cos(bearing_offset)) / 2.0;
    return gr;
  }

  GroundedRelation gr;
  gr.encoded_state = Eigen::VectorXd(2);
  gr.encoded_state << bearing_offset, distance;
  // Proximity: based on angular alignment. 1 = facing directly, 0 = facing away.
  // Uses cosine-based falloff: (1 + cos(bearing_offset)) / 2
  gr.proximity = (1.0 + std::cos(bearing_offset)) / 2.0;
  return gr;
}

double AvatarFacingKernel::Compare(const GroundedRelation& source,
                                   const GroundedRelation& target) const {
  return source.DistanceTo(target);
}

SatisfactionMetric AvatarFacingKernel::ComputeSatisfactionMetric(
    const GroundedRelation& current,
    const GroundedRelation* previous,
    const ActorMotionState& motion) const {
  double theta_current = std::abs(current.encoded_state(0));  // bearing offset

  // Trust directional progress only when sustainedly turning (EMA-smoothed
  // angular speed above floor); otherwise a still/jittering source emits a
  // thrashing estimated_time that oscillates the clone. See
  // EstimateTimeToSatisfaction.
  bool has_prev = previous && previous->encoded_state.size() > 0;
  double theta_previous = has_prev ? std::abs(previous->encoded_state(0)) : 0.0;
  double t = EstimateTimeToSatisfaction(theta_current, theta_previous, has_prev,
                                        motion.angular_speed);
  return {theta_current, t};
}

}  // namespace spatialblend
