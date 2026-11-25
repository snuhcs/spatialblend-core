// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/api/python/_spatialblend_sim_py.cc
//
// Python bindings of the in-process Simulator (a TEST-ONLY harness: it moves its
// avatar with the scenario test driver, not with a core component). Kept out of the
// shipped _spatialblend_py extension; the types it uses (ActorHandle, Trajectory,
// RealizationTarget, ...) are registered by _spatialblend_py, which must be
// imported first (the package __init__ does).
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "spatialblend/model/actor/actor.h"
#include "spatialblend/model/actor/subject_anchor.h"
#include "spatialblend/base/common/state_type.h"
#include "spatialblend/base/common/types.h"
#include "spatialblend/model/element/element.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/geometry_role.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/model/element/spatial_definition.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/geometry/polygon_coordinate_mapper.h"
#include "spatialblend/model/relation/grounded_relation.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/support.h"
#include "spatialblend/model/actor/actor_motion_tracker.h"
#include "spatialblend/space/binding.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/solver/realization_target.h"
#include "spatialblend/solver/relation_solver.h"
#include "spatialblend/space/relation_state.h"
#include "spatialblend/model/realization/remote_update.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/builder/actor_handle.h"
#include "spatialblend/space/builder/element_handle.h"
#include "spatialblend/space/builder/space_builder.h"
#include "spatialblend/sim/simulator.h"
#include "spatialblend/space/builder/trajectory.h"
#include "spatialblend/space/space_hub.h"

namespace py = pybind11;
using namespace spatialblend;

// Trampoline: tells pybind11 "don't delete this, we don't own it".
// Used for objects returned as raw pointers from manager classes.
template <typename T>
using RefHolder = std::unique_ptr<T, py::nodelete>;

PYBIND11_MODULE(_spatialblend_sim_py, m) {
  m.doc() = "Test-only Simulator bindings for spatialblend";

  // === SimulatorRunConfig ===
  py::class_<SimulatorRunConfig>(m, "SimulatorRunConfig")
      .def(py::init<>())
      .def_readwrite("max_linear_speed", &SimulatorRunConfig::max_linear_speed)
      .def_readwrite("max_angular_speed", &SimulatorRunConfig::max_angular_speed)
      .def_readwrite("use_velocity_bounded",
                    &SimulatorRunConfig::use_velocity_bounded)
      .def_readwrite("max_catchup_time", &SimulatorRunConfig::max_catchup_time)
      .def_readwrite("catchup_convergence",
                    &SimulatorRunConfig::catchup_convergence);

  // === SimulationResult ===
  py::class_<SimulationResult>(m, "SimulationResult")
      .def_readonly("num_frames", &SimulationResult::num_frames)
      .def_readonly("dt", &SimulationResult::dt)
      .def_readonly("actor_positions", &SimulationResult::actor_positions)
      .def_readonly("actor_yaws", &SimulationResult::actor_yaws)
      .def_readonly("avatar_positions", &SimulationResult::avatar_positions)
      .def_readonly("avatar_yaws", &SimulationResult::avatar_yaws)
      .def_readonly("targets", &SimulationResult::targets)
      .def("total_avatar_travel", &SimulationResult::TotalAvatarTravel)
      .def("max_speed", &SimulationResult::MaxSpeed)
      .def("final_distance_to", &SimulationResult::FinalDistanceTo,
           py::arg("target"))
      .def("avatar_always_in_polygon",
           &SimulationResult::AvatarAlwaysInPolygon, py::arg("polygon"))
      .def("max_acceleration", &SimulationResult::MaxAcceleration)
      .def("mean_jerk", &SimulationResult::MeanJerk)
      .def("direction_changes", &SimulationResult::DirectionChanges,
           py::arg("angle_threshold_rad") = 0.5)
      .def("arrival_lag", &SimulationResult::ArrivalLag,
           py::arg("target"), py::arg("threshold") = 1.0)
      .def("mean_relation_error", &SimulationResult::MeanRelationError);

  // === ProximitySnapshot ===
  py::class_<ProximitySnapshot>(m, "ProximitySnapshot")
      .def(py::init<>())
      .def_readonly("shared_id", &ProximitySnapshot::shared_id)
      .def_readonly("kernel_type", &ProximitySnapshot::kernel_type)
      .def_readonly("proximity", &ProximitySnapshot::proximity);

  // === SimulatorHooks (trampoline for Python subclassing) ===
  struct PySimulatorHooks : public SimulatorHooks {
    using SimulatorHooks::SimulatorHooks;

    void OnProximityUpdate(
        int frame, double time,
        const std::vector<ProximitySnapshot>& source_proximities,
        const std::vector<ProximitySnapshot>& avatar_proximities) override {
      PYBIND11_OVERRIDE(void, SimulatorHooks, OnProximityUpdate,
                        frame, time, source_proximities, avatar_proximities);
    }

    void OnFrame(
        int frame, double time,
        const Eigen::Vector2d& actor_pos, double actor_yaw,
        const Eigen::Vector2d& avatar_pos, double avatar_yaw,
        const std::vector<RealizationTarget>& targets) override {
      PYBIND11_OVERRIDE(void, SimulatorHooks, OnFrame,
                        frame, time, actor_pos, actor_yaw,
                        avatar_pos, avatar_yaw, targets);
    }
  };

  py::class_<SimulatorHooks, PySimulatorHooks>(m, "SimulatorHooks")
      .def(py::init<>())
      .def("on_proximity_update", &SimulatorHooks::OnProximityUpdate,
           py::arg("frame"), py::arg("time"),
           py::arg("source_proximities"), py::arg("avatar_proximities"))
      .def("on_frame", &SimulatorHooks::OnFrame,
           py::arg("frame"), py::arg("time"),
           py::arg("actor_pos"), py::arg("actor_yaw"),
           py::arg("avatar_pos"), py::arg("avatar_yaw"),
           py::arg("targets"));

  // === Simulator ===
  py::class_<Simulator, std::unique_ptr<Simulator>>(m, "Simulator")
      .def_static("create", &Simulator::Create)
      .def("add_space", &Simulator::AddSpace)
      .def("tick", &Simulator::Tick, py::arg("dt"))
      .def("set_hooks", &Simulator::SetHooks, py::arg("hooks"),
           py::keep_alive<1, 2>())  // prevent GC of hooks while simulator lives
      .def("run", &Simulator::Run,
           py::arg("actor"), py::arg("avatar"), py::arg("trajectory"),
           py::arg("dt"), py::arg("config") = SimulatorRunConfig{});
}
