// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SIMULATOR_SIMULATOR_H_
#define SPATIALBLEND_SIMULATOR_SIMULATOR_H_

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "spatialblend/base/common/common.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/space/builder/actor_handle.h"
#include "spatialblend/space/builder/space_builder.h"
#include "spatialblend/space/builder/trajectory.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/solver/relation_solver.h"

namespace spatialblend {

// Per-element proximity snapshot for hooks.
struct ProximitySnapshot {
  int64_t shared_id = 0;
  RelationKernelType kernel_type = RelationKernelType::kAvatarSpacing;
  double proximity = 0.0;
};

// Observer interface for Simulator::Run(). Matches v1's SimulatorHooks pattern.
class SimulatorHooks {
 public:
  virtual ~SimulatorHooks() = default;

  // Called each frame with source and avatar proximity snapshots.
  virtual void OnProximityUpdate(
      int frame, double time,
      const std::vector<ProximitySnapshot>& source_proximities,
      const std::vector<ProximitySnapshot>& avatar_proximities) {}

  // Called each frame with actor/avatar positions and realization targets.
  virtual void OnFrame(
      int frame, double time,
      const Eigen::Vector2d& actor_pos, double actor_yaw,
      const Eigen::Vector2d& avatar_pos, double avatar_yaw,
      const std::vector<RealizationTarget>& targets) {}
};

struct SimulationResult {
  int num_frames = 0;
  double dt = 0.0;

  std::vector<Eigen::Vector2d> actor_positions;
  std::vector<double> actor_yaws;
  std::vector<Eigen::Vector2d> avatar_positions;
  std::vector<double> avatar_yaws;
  std::vector<std::vector<RealizationTarget>> targets;

  double TotalAvatarTravel() const;
  double MaxSpeed() const;
  double FinalDistanceTo(const Eigen::Vector2d& target) const;
  bool AvatarAlwaysInPolygon(const Polygon2D& polygon) const;

  // Smoothness metrics
  double MaxAcceleration() const;
  double MeanJerk() const;
  int DirectionChanges(double angle_threshold_rad = 0.5) const;
  double ArrivalLag(const Eigen::Vector2d& target,
                    double threshold = 1.0) const;

  // Relation error: mean distance between solver-recommended target positions
  // and the avatar's actual position at each frame.
  double MeanRelationError() const;
};

struct SimulatorRunConfig {
  double max_linear_speed = 1.4;   // m/s (human walking speed)
  double max_angular_speed = 2.0;  // rad/s
  bool use_velocity_bounded = true;
  // After source trajectory ends, keep ticking the controller until
  // avatar stops moving or max_catchup_time elapses. Set to 0 to disable.
  double max_catchup_time = 10.0;  // seconds
  double catchup_convergence = 0.001;  // m — stop when per-frame displacement < this
};

class Simulator {
 public:
  using RunConfig = SimulatorRunConfig;

  static std::unique_ptr<Simulator> Create();

  SpaceBuilder AddSpace();
  void Tick(double dt);

  // Set hooks for observing simulation events (proximity, frame data).
  // Must be called before Run(). Simulator does NOT own the hooks.
  void SetHooks(SimulatorHooks* hooks);

  SimulationResult Run(ActorHandle& actor, ActorHandle& avatar,
                       const Trajectory& traj, double dt,
                       const RunConfig& config = RunConfig{});

  // Register an actor for dirty tracking (called by ActorHandle::MoveTo).
  void RegisterDirtyActor(Space* space, spatialblend::entity_id_t actor_id,
                          const Eigen::Vector2d& pos, double yaw);

 private:
  Simulator();
  std::unique_ptr<LocalContext> context_;
  SimulatorHooks* hooks_ = nullptr;

  struct PendingUpdate {
    Space* space;
    spatialblend::entity_id_t actor_id;
    Eigen::Vector2d position;
    double yaw;
  };
  std::vector<PendingUpdate> pending_updates_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SIMULATOR_SIMULATOR_H_
