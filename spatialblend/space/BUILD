# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "relation_event",
    hdrs = ["relation_event.h"],
    deps = [
        "//spatialblend/base/common:types",
        "//spatialblend/model/relation:relation_kernel",
    ],
)

spatialblend_cc_test(
    name = "referent_resolver_test",
    srcs = ["referent_resolver_test.cc"],
    deps = [
        ":space",
        ":space_hub",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/solver:referent_resolver",
        "//spatialblend/solver:relation_solver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "relation_state",
    hdrs = ["relation_state.h"],
    deps = [
        ":relation_event",
        "//spatialblend/base/common:types",
        "//spatialblend/geometry:polygon_coordinate_mapper",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "@eigen",
    ],
)

cc_library(
    name = "relation_phase_tracker",
    srcs = ["relation_phase_tracker.cc"],
    hdrs = ["relation_phase_tracker.h"],
    deps = [
        ":relation_event",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
    ],
)

spatialblend_cc_test(
    name = "relation_phase_tracker_test",
    srcs = ["relation_phase_tracker_test.cc"],
    deps = [
        ":relation_phase_tracker",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "relation_event_e2e_test",
    srcs = ["relation_event_e2e_test.cc"],
    deps = [
        ":local_context",
        ":relation_event",
        ":space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "seat_contact_contract_test",
    srcs = ["seat_contact_contract_test.cc"],
    deps = [
        ":space",
        ":space_hub",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "source_interaction_evidence_test",
    srcs = ["source_interaction_evidence_test.cc"],
    deps = [
        ":space",
        ":space_hub",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "observed_facing_point_test",
    srcs = ["observed_facing_point_test.cc"],
    deps = [
        ":space",
        ":space_hub",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "receiver_input_test",
    srcs = ["receiver_input_test.cc"],
    deps = [
        ":space",
        ":space_hub",
        "//spatialblend/model/realization:receiver_input",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "space",
    srcs = [
        "anchor_proposals.cc",
        "observed_facing_point.cc",
        "receiver_approach_context.cc",
        "receiver_approach_feasibility.cc",
        "receiver_space_input.cc",
        "scene_inputs.cc",
        "source_interaction_evidence.cc",
        "space.cc",
        "space_hub.cc",
    ],
    hdrs = [
        "bone_port.h",
        "embodied_port.h",
        "interaction_latency.h",
        "receiver_approach_context.h",
        "receiver_approach_feasibility.h",
        "receiver_proposal.h",
        "source_interaction_evidence.h",
        "space.h",
    ],
    deps = [
        ":binding",
        ":relation_event",
        ":relation_phase_tracker",
        ":relation_state",
        ":space_hub_decl",
        "//spatialblend/base/common",
        "//spatialblend/base/common:coalescing_worker",
        "//spatialblend/base/common:seqlock_box",
        "//spatialblend/base/common:trace",
        "//spatialblend/base/common:types",
        "//spatialblend/base/util:logger",
        "//spatialblend/geometry:radial_surface_chart",
        "//spatialblend/model/actor",
        "//spatialblend/model/actor:actor_motion_tracker",
        "//spatialblend/model/element:containment_layout",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:geometry_role",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:scope_extension",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/embodiment",
        "//spatialblend/model/realization:containment_transfer",
        "//spatialblend/model/realization:receiver_input",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/model/relation:grounding_rule",
        "//spatialblend/model/relation:kernel_factory",
        "//spatialblend/model/relation:scope_gate",
        "//spatialblend/model/relation:scope_resolver",
        "//spatialblend/solver:affordance_pose_sd",
        "//spatialblend/solver:realization_policy",
        "//spatialblend/solver:realization_target",
        "//spatialblend/solver:referent_resolver",
        "//spatialblend/solver:relation_solver",
        "@eigen",
        "@nlohmann_json//:json",
    ],
)

spatialblend_cc_test(
    name = "scene_inputs_test",
    srcs = ["scene_inputs_test.cc"],
    deps = [
        ":space",
        ":space_hub",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/space/testing:ideal_driver",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "receiver_approach_feasibility_test",
    srcs = ["receiver_approach_feasibility_test.cc"],
    deps = [
        ":space",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "receiver_approach_context_test",
    srcs = ["receiver_approach_context_test.cc"],
    deps = [
        ":space",
        "//spatialblend/model/element:navigable_spatial_definition",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "update_actor_state_characterization_test",
    srcs = ["update_actor_state_characterization_test.cc"],
    deps = [
        ":binding",
        ":local_context",
        ":relation_event",
        ":space",
        ":space_hub",
        "//spatialblend/model/element:navigable_spatial_definition",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "space_test",
    srcs = ["space_test.cc"],
    deps = [
        ":space",
        ":space_hub",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/relation",
        "//spatialblend/model/relation:affordance_pose_grounding_rule",
        "//spatialblend/model/relation:avatar_affordance_pose_kernel",
        "//spatialblend/model/relation:avatar_spacing_kernel",
        "//spatialblend/model/relation:grounding_rule",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space/testing:ideal_driver",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "space_hub_decl",
    hdrs = ["space_hub.h"],
    deps = [
        "//spatialblend/base/common:types",
        "//spatialblend/model/link:element_linker",
        "//spatialblend/model/realization:remote_update",
        "@abseil-cpp//absl/status",
    ],
)

# Space calls SpaceHub::Propagate and SpaceHub::Propagate calls back into Space,
# so the two are one translation-unit cycle. space_hub.cc is compiled into
# :space (header via :space_hub_decl); depending on :space alone links. This
# target remains as the name callers use to say "I use the hub".
cc_library(
    name = "space_hub",
    deps = [
        ":space",
        ":space_hub_decl",
    ],
)

spatialblend_cc_test(
    name = "space_hub_test",
    srcs = ["space_hub_test.cc"],
    deps = [
        ":space",
        ":space_hub",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "local_context",
    srcs = ["local_context.cc"],
    hdrs = ["local_context.h"],
    deps = [
        ":space",
        ":space_hub",
    ],
)

spatialblend_cc_test(
    name = "local_context_test",
    srcs = ["local_context_test.cc"],
    deps = [
        ":local_context",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "integration_test",
    srcs = ["integration_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "binding",
    hdrs = [
        "binding.h",
        "scope_record.h",
    ],
    deps = [
        ":relation_phase_tracker",
        "//spatialblend/base/common:types",
        "//spatialblend/model/relation",
        "//spatialblend/model/relation:grounding_rule",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/model/relation:scope_resolver",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "scenario_test",
    srcs = ["scenario_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space/testing:ideal_driver",
        "//spatialblend/space/testing:test_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "e2e_test",
    srcs = ["e2e_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space/testing:ideal_driver",
        "//spatialblend/space/testing:test_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "geometry_mediation_contract_test",
    srcs = ["geometry_mediation_contract_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/base/common:trace",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/space/testing:ideal_driver",
        "//spatialblend/space/testing:test_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "geometry_mediation_benchmark_test",
    srcs = ["geometry_mediation_benchmark_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/geometry:obstacle_free_space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/space/testing:test_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# Per-tick cost of the paper-objective
# refinement at 10/30/60 obstacles, in the single-relation and the
# 4-referent + Facing scenes (a1_space_bench_test.cc). Two targets, one
# source:
#   a1_space_evalcount_test  deterministic, non-timing: evaluation/restart
#                            caps and the live warm start. Normal suite.
#   a1_space_bench_test      WALL-CLOCK budget (median <= 2.5 ms, p95 <= 5
#                            ms per frame). `exclusive`: any parallel load
#                            makes a wall-clock gate flaky. Documented
#                            command, on an otherwise idle host:
#                              bazel test //spatialblend/space:a1_space_bench_test --test_output=all
_A1_SPACE_BENCH_DEPS = [
    "//spatialblend/space/testing:test_driver",
    "//spatialblend/space:local_context",
    "//spatialblend/space",
    "//spatialblend/space:space_hub",
    "//spatialblend/model/element:navigable_spatial_definition",
    "//spatialblend/model/element:polytope_spatial_definition",
    "//spatialblend/geometry:obstacle_free_space",
    "//spatialblend/geometry:polygon2d",
    "@eigen",
    "@googletest//:gtest_main",
]

spatialblend_cc_test(
    name = "a1_space_evalcount_test",
    srcs = ["a1_space_bench_test.cc"],
    args = ["--gtest_filter=A1SpaceEvalCount.*"],
    deps = _A1_SPACE_BENCH_DEPS,
)

spatialblend_cc_test(
    name = "a1_space_bench_test",
    srcs = ["a1_space_bench_test.cc"],
    args = ["--gtest_filter=A1SpaceBench.*"],
    tags = ["exclusive"],
    deps = _A1_SPACE_BENCH_DEPS,
)

spatialblend_cc_test(
    name = "design_acceptance_test",
    srcs = ["design_acceptance_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space/testing:ideal_driver",
        "//spatialblend/space/testing:test_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# Containment transfer: mapped object-frame root pose, 0.2 m blend band,
# instant contained state and event, element-local polygon.
spatialblend_cc_test(
    name = "containment_transfer_test",
    srcs = ["containment_transfer_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:radial_surface_chart",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/realization:containment_transfer",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# Does the in-process two-space rig's target realization-target stream
# permanently stall once the source goes stationary, or does it correctly
# pause (heartbeat-bounded) and resume when the source moves again? A
# standalone file (like c19b_identity_room_test) so it doesn't collide
# with cross_space_test.cc's existing fixtures/namespaces.
spatialblend_cc_test(
    name = "c21_target_stall_test",
    srcs = ["c21_target_stall_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# Peer/receiver space_clock_ must advance by REAL elapsed source time
# (RemoteRelationUpdate::source_clock delta), not by delivery-count *
# step_duration -- a standalone file (like c21_target_stall_test.cc) so it
# doesn't collide with cross_space_test.cc's/c21_target_stall_test.cc's own
# fixtures/namespaces.
spatialblend_cc_test(
    name = "c22_peer_clock_test",
    srcs = ["c22_peer_clock_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/base/common:trace",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

spatialblend_cc_test(
    name = "scene_validation_test",
    srcs = ["scene_validation_test.cc"],
    deps = [
        ":local_context",
        ":space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:scope_extension",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/model/relation:scope_resolver",
        "//spatialblend/model/relation:spacing_grounding_rule",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "hierarchy_behavior_test",
    srcs = ["hierarchy_behavior_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "delta_lifecycle_test",
    srcs = ["delta_lifecycle_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "geometry_stress_test",
    srcs = ["geometry_stress_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# Contract tests for the per-family scope gates and lifecycle thresholds and the
# shared-id resolution (the list is at the top of the test file).
spatialblend_cc_test(
    name = "scope_gating_contract_test",
    srcs = ["scope_gating_contract_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# Layer tests for the family scope definitions D1-D9.
spatialblend_cc_test(
    name = "family_scope_test",
    srcs = ["family_scope_test.cc"],
    deps = [
        ":local_context",
        ":space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# The scope case catalog (SCOPE_CASES.md) as unit tests, and the check that the two agree.
spatialblend_cc_test(
    name = "scope_cases_test",
    srcs = ["scope_cases_test.cc"],
    data = ["//spatialblend/space/testing:SCOPE_CASES.md"],
    deps = [
        ":local_context",
        ":space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:containment_layout",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/embodiment",
        "//spatialblend/model/relation:scope_gate",
        "//spatialblend/model/relation:scope_resolver",
        "//spatialblend/solver:affordance_pose_sd",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# Sit transfer: shared-only seats, the seat plane, the seat front. Tests only.
spatialblend_cc_test(
    name = "seat_transfer_test",
    srcs = ["seat_transfer_test.cc"],
    deps = [
        ":local_context",
        ":space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# Surface retargeting: hand/controller contact on a board face, frame charts of a part.
spatialblend_cc_test(
    name = "surface_retarget_test",
    srcs = ["surface_retarget_test.cc"],
    deps = [
        ":local_context",
        ":space",
        "//spatialblend/geometry:planar_frame_chart",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# The one affordance-pose capability: registered contacts (seat, palm, controller tip) mapped between matched parts.
spatialblend_cc_test(
    name = "affordance_pose_contact_test",
    srcs = ["affordance_pose_contact_test.cc"],
    deps = [
        ":local_context",
        ":space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/embodiment",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# R2_6_SeatGeometric: the seat is a geometric fact.
spatialblend_cc_test(
    name = "seat_geometry_test",
    srcs = ["seat_geometry_test.cc"],
    deps = [
        ":local_context",
        ":space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# Microbenchmark for the Facing scope gate (manual).
cc_binary(
    name = "scope_gate_benchmark",
    srcs = ["scope_gate_benchmark.cc"],
    tags = ["manual"],
    deps = [
        ":local_context",
        ":space",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
    ],
)

# Scene-level Spacing target continuity just outside a Containment region edge.
spatialblend_cc_test(
    name = "spacing_edge_continuity_test",
    srcs = ["spacing_edge_continuity_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "spacing_edge_shape_review_test",
    srcs = ["spacing_edge_shape_review_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon_coordinate_mapper",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/solver:relation_solver",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "spacing_surface_referent_test",
    srcs = ["spacing_surface_referent_test.cc"],
    deps = [
        ":local_context",
        ":space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "replay_chart_mapping_test",
    srcs = ["replay_chart_mapping_test.cc"],
    deps = [
        ":local_context",
        ":space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "seat_approach_continuity_test",
    srcs = ["seat_approach_continuity_test.cc"],
    deps = [
        ":local_context",
        ":space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/embodiment",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "anchor_proposals_test",
    srcs = ["anchor_proposals_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/solver:affordance_pose_sd",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "embodiment_registration_test",
    srcs = ["embodiment_registration_test.cc"],
    deps = [
        ":local_context",
        ":space",
        ":space_hub",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/embodiment",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "nested_affordance_solve_test",
    srcs = ["nested_affordance_solve_test.cc"],
    deps = [
        ":local_context",
        ":space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/embodiment",
        "//spatialblend/solver:affordance_pose_sd",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# The root's live contract is the pursued target (ratio continuation), a position every update.
spatialblend_cc_test(
    name = "root_pursuit_contract_test",
    srcs = ["root_pursuit_contract_test.cc"],
    deps = [
        ":local_context",
        ":space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "core_integration_test",
    size = "medium",
    srcs = ["core_integration_test.cc"],
    deps = [
        ":local_context",
        ":space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/solver:relation_solver",
        "//spatialblend/space/testing:ideal_driver",
        "//spatialblend/space/testing:local_minimum",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# The tests the scope case catalog (testing/SCOPE_CASES.md) cites, readable by the scenario level's coverage check.
exports_files([
    "containment_transfer_test.cc",
    "family_scope_test.cc",
    "replay_chart_mapping_test.cc",
    "scope_cases_test.cc",
    "scope_gating_contract_test.cc",
])
