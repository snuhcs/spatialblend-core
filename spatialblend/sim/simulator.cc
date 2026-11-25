// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/sim/simulator.h"

#include <algorithm>
#include <cmath>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/space/binding.h"
#include "spatialblend/space/testing/test_driver.h"

namespace spatialblend {

namespace {

// Compute centroid of the walkable polygon attached to the actor's scope
// element.  Returns (0,0) if no polygon is found.
Eigen::Vector2d GetScopePolygonCentroid(const ActorHandle& handle) {
  Space* space = handle.GetSpace();
  Actor* actor = space->GetActor(handle.id());
  if (!actor) return Eigen::Vector2d::Zero();

  element_id_t elem_id = actor->GetElementId();
  const Element* elem = space->GetElementTree()->GetElement(elem_id);
  if (!elem) return Eigen::Vector2d::Zero();

  auto sd = elem->GetSpatialDefinitionShared();  // unlocked reader of the tree (element.h)
  if (!sd) return Eigen::Vector2d::Zero();

  auto boundary = sd->GetBoundary();
  if (!boundary || boundary->GetPoints().empty())
    return Eigen::Vector2d::Zero();

  const auto& pts = boundary->GetPoints();
  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const auto& p : pts) centroid += p;
  centroid /= static_cast<double>(pts.size());
  return centroid;
}

// Capture proximity snapshots from an actor's bindings.
std::vector<ProximitySnapshot> CaptureProximity(Space* space,
                                                spatialblend::entity_id_t actor_id) {
  std::vector<ProximitySnapshot> snaps;
  const auto& bindings = space->GetBindings(actor_id);
  for (const auto& binding : bindings) {
    for (const auto& match : binding.latest_matches) {
      const Element* elem = space->GetElementTree()->GetElement(match.element_id);
      if (!elem || elem->GetSharedId() == kInvalidSharedElementId) continue;
      ProximitySnapshot s;
      s.shared_id = elem->GetSharedId();
      s.kernel_type = binding.kernel_type;
      s.proximity = match.eval.grounded.proximity;
      snaps.push_back(s);
    }
  }
  return snaps;
}

// Compute avatar-side proximity by re-evaluating bindings from the received
// remote update's referent elements. Since the avatar doesn't have local
// bindings, we approximate: for each source proximity snapshot, find the
// corresponding target element (by shared_id), compute distance from avatar
// to that element, and derive proximity = 1 / (1 + dist).
std::vector<ProximitySnapshot> CaptureAvatarProximity(
    Space* avatar_space, spatialblend::entity_id_t avatar_id,
    const std::vector<ProximitySnapshot>& source_snaps) {
  std::vector<ProximitySnapshot> snaps;
  Eigen::Vector2d avatar_pos = Eigen::Vector2d::Zero();
  double avatar_yaw_unused = 0.0;
  avatar_space->CopyActorPose(avatar_id, &avatar_pos, &avatar_yaw_unused);
  const auto* tree = avatar_space->GetElementTree();

  for (const auto& src : source_snaps) {
    const Element* elem = tree->GetElementBySharedId(src.shared_id);
    if (!elem) continue;

    // Check if avatar is inside element's SCOPE region (relation_kernel.h FootprintHolds) → proximity 1.0
    double proximity = 0.0;
    auto sd = elem->GetScopeDefinitionShared();  // unlocked reader of the tree (element.h)
    if (sd) {
      if (FootprintHoldsIn(sd.get(), *elem, *tree, Eigen::Vector3d(avatar_pos.x(), avatar_pos.y(), 0.0))) {
        proximity = 1.0;
      } else {
        // Distance to element world position
        Eigen::Matrix4d world_t = tree->GetWorldTransform(elem->GetId());
        Eigen::Vector2d elem_pos(world_t(0, 3), world_t(1, 3));
        double dist = (avatar_pos - elem_pos).norm();
        proximity = 1.0 / (1.0 + dist);
      }
    }

    ProximitySnapshot s;
    s.shared_id = src.shared_id;
    s.kernel_type = src.kernel_type;
    s.proximity = proximity;
    snaps.push_back(s);
  }
  return snaps;
}

}  // namespace

Simulator::Simulator() : context_(LocalContext::Create()) {}

void Simulator::SetHooks(SimulatorHooks* hooks) { hooks_ = hooks; }

std::unique_ptr<Simulator> Simulator::Create() {
  return std::unique_ptr<Simulator>(new Simulator());
}

SpaceBuilder Simulator::AddSpace() {
  Space* space = context_->AddSpace();
  return SpaceBuilder(space,
                      [this](Space* s, spatialblend::entity_id_t id,
                             const Eigen::Vector2d& pos, double yaw) {
                        RegisterDirtyActor(s, id, pos, yaw);
                      });
}

void Simulator::RegisterDirtyActor(Space* space,
                                   spatialblend::entity_id_t actor_id,
                                   const Eigen::Vector2d& pos, double yaw) {
  pending_updates_.push_back({space, actor_id, pos, yaw});
}

void Simulator::Tick(double dt) {
  for (const auto& update : pending_updates_) {
    update.space->UpdateActorState(update.actor_id, update.position,
                                   update.yaw);
  }
  pending_updates_.clear();
}

SimulationResult Simulator::Run(ActorHandle& actor, ActorHandle& avatar,
                                const Trajectory& traj, double dt,
                                const RunConfig& config) {
  SimulationResult result;
  result.dt = dt;
  double duration = traj.Duration();

  // The avatar's DRIVER (this simulator plays the application): it owns the
  // embodied pose, publishes it into the avatar space's sampling port, and
  // moves it toward each proposal. use_velocity_bounded = a driver bounded in
  // speed and turn rate (seeded inside the avatar's room, never at the origin);
  // otherwise an ideal driver that adopts every proposed target at once.
  driver::Pose body;
  uint64_t body_sequence = 0;
  const auto publish_body = [&]() {
    EmbodiedPoseInput in;
    in.x = body.position.x();
    in.y = body.position.y();
    in.yaw = body.yaw;
    in.epoch = 1;
    avatar.GetSpace()->PublishEmbodiedPose(avatar.id(), in, 0.0,
                                           ++body_sequence);
  };
  const auto drive = [&](double step_dt) {
    ReceiverProposal proposal;
    if (!avatar.GetSpace()->ReadProposal(avatar.id(), &proposal)) return;
    if (proposal.status != ProposalStatus::kExact &&
        proposal.status != ProposalStatus::kInexact)
      return;
    driver::Pose target;
    target.position = Eigen::Vector2d(proposal.x, proposal.y);
    target.yaw = proposal.yaw;
    if (config.use_velocity_bounded) {
      driver::Bounds bounds;
      bounds.max_linear_speed = config.max_linear_speed;
      bounds.max_angular_speed = config.max_angular_speed;
      body = driver::Step(
          body, target,
          avatar.GetSpace()->QueryCourse(avatar.id(), body.position,
                                         target.position),
          bounds, step_dt);
    } else {
      body = target;
    }
    publish_body();
  };
  if (config.use_velocity_bounded) {
    const Eigen::Vector2d start_pos = GetScopePolygonCentroid(avatar);
    if (start_pos.squaredNorm() > 1e-6) body.position = start_pos;
  } else {
    body.position = avatar.GetPosition();
    body.yaw = avatar.GetYaw();
  }
  publish_body();

  for (double t = 0.0; t <= duration + dt * 0.5; t += dt) {
    double clamped_t = std::min(t, duration);
    auto pos = traj.PositionAt(clamped_t);
    double yaw = traj.YawAt(clamped_t);

    actor.MoveTo(pos, yaw);
    Tick(dt);

    // The driver applies the proposal within its bounds.
    drive(dt);

    result.actor_positions.push_back(pos);
    result.actor_yaws.push_back(yaw);
    result.avatar_positions.push_back(avatar.GetPosition());
    result.avatar_yaws.push_back(avatar.GetYaw());

    auto* avatar_targets = avatar.GetSpace()->GetRealizationTargets(avatar.id());
    std::vector<RealizationTarget> frame_targets;
    if (avatar_targets) {
      frame_targets = avatar_targets->targets;
    }
    result.targets.push_back(frame_targets);

    // Fire hooks
    if (hooks_) {
      auto src_prox = CaptureProximity(actor.GetSpace(), actor.id());
      auto avt_prox = CaptureAvatarProximity(
          avatar.GetSpace(), avatar.id(), src_prox);
      hooks_->OnProximityUpdate(result.num_frames, clamped_t,
                                src_prox, avt_prox);
      hooks_->OnFrame(result.num_frames, clamped_t, pos, yaw,
                      avatar.GetPosition(), avatar.GetYaw(), frame_targets);
    }

    result.num_frames++;
  }

  // Catchup phase: keep source at final position, re-run solver each tick
  // so progressive weights sharpen, then let the driver converge.
  if (config.use_velocity_bounded && config.max_catchup_time > 0.0) {
    auto final_actor_pos = traj.PositionAt(duration);
    double final_actor_yaw = traj.YawAt(duration);
    double catchup_elapsed = 0.0;

    while (catchup_elapsed < config.max_catchup_time) {
      // Re-send source actor state → re-runs solver with updated weights
      actor.MoveTo(final_actor_pos, final_actor_yaw);
      Tick(dt);
      drive(dt);
      catchup_elapsed += dt;

      auto new_pos = avatar.GetPosition();

      // Check convergence: avatar stopped moving
      if (!result.avatar_positions.empty()) {
        double displacement =
            (new_pos - result.avatar_positions.back()).norm();
        if (displacement < config.catchup_convergence) break;
      }

      result.actor_positions.push_back(final_actor_pos);
      result.actor_yaws.push_back(final_actor_yaw);
      result.avatar_positions.push_back(new_pos);
      result.avatar_yaws.push_back(avatar.GetYaw());

      auto* avatar_targets =
          avatar.GetSpace()->GetRealizationTargets(avatar.id());
      std::vector<RealizationTarget> frame_targets;
      if (avatar_targets) {
        frame_targets = avatar_targets->targets;
      }
      result.targets.push_back(frame_targets);

      if (hooks_) {
        double frame_time = duration + catchup_elapsed;
        auto src_prox = CaptureProximity(actor.GetSpace(), actor.id());
        auto avt_prox = CaptureAvatarProximity(
            avatar.GetSpace(), avatar.id(), src_prox);
        hooks_->OnProximityUpdate(result.num_frames, frame_time,
                                  src_prox, avt_prox);
        hooks_->OnFrame(result.num_frames, frame_time,
                        final_actor_pos, final_actor_yaw,
                        new_pos, avatar.GetYaw(), frame_targets);
      }

      result.num_frames++;
    }
  }

  return result;
}

double SimulationResult::TotalAvatarTravel() const {
  double total = 0.0;
  for (size_t i = 1; i < avatar_positions.size(); ++i) {
    total += (avatar_positions[i] - avatar_positions[i - 1]).norm();
  }
  return total;
}

double SimulationResult::MaxSpeed() const {
  double max_speed = 0.0;
  for (size_t i = 1; i < avatar_positions.size(); ++i) {
    double dist = (avatar_positions[i] - avatar_positions[i - 1]).norm();
    max_speed = std::max(max_speed, dist / dt);
  }
  return max_speed;
}

double SimulationResult::FinalDistanceTo(const Eigen::Vector2d& target) const {
  if (avatar_positions.empty()) return 0.0;
  return (avatar_positions.back() - target).norm();
}

bool SimulationResult::AvatarAlwaysInPolygon(const Polygon2D& polygon) const {
  for (const auto& pos : avatar_positions) {
    if (!polygon.Contains(pos)) return false;
  }
  return true;
}

double SimulationResult::MaxAcceleration() const {
  if (avatar_positions.size() < 3 || dt <= 0.0) return 0.0;
  double max_accel = 0.0;
  Eigen::Vector2d prev_vel =
      (avatar_positions[1] - avatar_positions[0]) / dt;
  for (size_t i = 2; i < avatar_positions.size(); ++i) {
    Eigen::Vector2d vel =
        (avatar_positions[i] - avatar_positions[i - 1]) / dt;
    double accel = (vel - prev_vel).norm() / dt;
    max_accel = std::max(max_accel, accel);
    prev_vel = vel;
  }
  return max_accel;
}

double SimulationResult::MeanJerk() const {
  if (avatar_positions.size() < 4 || dt <= 0.0) return 0.0;
  // Compute accelerations first.
  std::vector<Eigen::Vector2d> accels;
  accels.reserve(avatar_positions.size() - 2);
  for (size_t i = 2; i < avatar_positions.size(); ++i) {
    Eigen::Vector2d vel_prev =
        (avatar_positions[i - 1] - avatar_positions[i - 2]) / dt;
    Eigen::Vector2d vel_cur =
        (avatar_positions[i] - avatar_positions[i - 1]) / dt;
    accels.push_back((vel_cur - vel_prev) / dt);
  }
  // Jerk = change in acceleration.
  double total_jerk = 0.0;
  for (size_t i = 1; i < accels.size(); ++i) {
    total_jerk += (accels[i] - accels[i - 1]).norm() / dt;
  }
  return total_jerk / static_cast<double>(accels.size() - 1);
}

int SimulationResult::DirectionChanges(double angle_threshold_rad) const {
  if (avatar_positions.size() < 3) return 0;
  int changes = 0;
  for (size_t i = 2; i < avatar_positions.size(); ++i) {
    Eigen::Vector2d d1 = avatar_positions[i - 1] - avatar_positions[i - 2];
    Eigen::Vector2d d2 = avatar_positions[i] - avatar_positions[i - 1];
    if (d1.norm() < 1e-9 || d2.norm() < 1e-9) continue;
    double cos_angle = d1.normalized().dot(d2.normalized());
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
    double angle = std::acos(cos_angle);
    if (angle > angle_threshold_rad) {
      ++changes;
    }
  }
  return changes;
}

double SimulationResult::ArrivalLag(const Eigen::Vector2d& target,
                                    double threshold) const {
  if (actor_positions.empty() || avatar_positions.empty() || dt <= 0.0) {
    return 0.0;
  }
  // Find first frame where actor is within threshold of target.
  int actor_arrival = -1;
  for (size_t i = 0; i < actor_positions.size(); ++i) {
    if ((actor_positions[i] - target).norm() <= threshold) {
      actor_arrival = static_cast<int>(i);
      break;
    }
  }
  if (actor_arrival < 0) return 0.0;  // Actor never arrives.

  // Find first frame where avatar is within threshold of target.
  int avatar_arrival = -1;
  for (size_t i = 0; i < avatar_positions.size(); ++i) {
    if ((avatar_positions[i] - target).norm() <= threshold) {
      avatar_arrival = static_cast<int>(i);
      break;
    }
  }
  if (avatar_arrival < 0) {
    // Avatar never arrives; return remaining duration as lag.
    return static_cast<double>(num_frames - actor_arrival) * dt;
  }

  return static_cast<double>(avatar_arrival - actor_arrival) * dt;
}

double SimulationResult::MeanRelationError() const {
  if (targets.empty() || avatar_positions.empty()) return 0.0;
  double total_error = 0.0;
  int count = 0;
  for (size_t i = 0; i < targets.size() && i < avatar_positions.size(); ++i) {
    if (targets[i].empty()) continue;
    // Use the first (highest-weight) target as the reference.
    const auto& t = targets[i][0];
    total_error += (avatar_positions[i] - t.position).norm();
    ++count;
  }
  return count > 0 ? total_error / static_cast<double>(count) : 0.0;
}

}  // namespace spatialblend
