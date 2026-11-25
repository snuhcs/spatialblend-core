# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "realization_policy",
    hdrs = ["realization_policy.h"],
    deps = [
        ":soft_annealing_weights",
        "//spatialblend/model/realization:remote_update",
    ],
)

cc_library(
    name = "soft_annealing_weights",
    srcs = ["soft_annealing_weights.cc"],
    hdrs = ["soft_annealing_weights.h"],
)

spatialblend_cc_test(
    name = "realization_policy_test",
    srcs = ["realization_policy_test.cc"],
    deps = [
        ":realization_policy",
        "//spatialblend/model/realization:remote_update",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "referent_resolver",
    hdrs = ["referent_resolver.h"],
    deps = [
        "//spatialblend/base/common:types",
        "//spatialblend/model/element",
        "//spatialblend/model/element:element_tree",
    ],
)

# Affordance-pose support and its solve-domain geometry.
cc_library(
    name = "affordance_pose_sd",
    srcs = ["affordance_pose_sd.cc"],
    hdrs = ["affordance_pose_sd.h"],
    deps = [
        ":relation_solver",
        "//spatialblend/base/common:types",
        "//spatialblend/geometry:planar_frame_chart",
        "//spatialblend/geometry:polygon_coordinate_mapper",
        "//spatialblend/geometry:radial_surface_chart",
        "//spatialblend/model/element",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:geometry_role",
        "//spatialblend/model/element:spatial_definition",
        "//spatialblend/model/embodiment",
        "@eigen",
    ],
)

cc_library(
    name = "realization_target",
    hdrs = ["realization_target.h"],
    deps = [
        ":relation_solver",
        "//spatialblend/base/common:types",
        "//spatialblend/model/relation:relation_kernel",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "realization_target_test",
    srcs = ["realization_target_test.cc"],
    deps = [
        ":realization_target",
        ":relation_solver",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/relation",
        "//spatialblend/model/relation:avatar_facing_kernel",
        "//spatialblend/model/relation:avatar_spacing_kernel",
        "//spatialblend/model/relation:grounding_rule",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "relation_solver",
    srcs = [
        "relation_evaluator.cc",
        "relation_solver.cc",
    ],
    hdrs = [
        "nd_trust_region.h",
        "nd_trust_region_loop.h",
        "relation_evaluator.h",
        "relation_scope.h",
        "relation_solver.h",
        "scope_pursuit.h",
    ],
    deps = [
        ":engagement",
        ":referent_resolver",
        "//spatialblend/base/common:types",
        "//spatialblend/geometry:planar_frame_chart",
        "//spatialblend/geometry:polygon_coordinate_mapper",
        "//spatialblend/geometry:radial_surface_chart",
        "//spatialblend/geometry/path:bezier_course",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/realization:receiver_input",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/model/relation:grounding_rule",
        "//spatialblend/model/relation:kernel_factory",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/model/relation:scope_resolver",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "relation_solver_parity_test",
    srcs = [
        "relation_solver_parity_fixture_darwin_arm64.inc",
        "relation_solver_parity_fixture_linux_x86_64.inc",
        "relation_solver_parity_test.cc",
    ],
    deps = [
        ":relation_solver",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/model/relation",
        "//spatialblend/model/relation:avatar_spacing_kernel",
        "//spatialblend/model/relation:grounding_rule",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "relation_solver_test",
    srcs = ["relation_solver_test.cc"],
    deps = [
        ":realization_policy",
        ":relation_solver",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:envelope",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/model/relation",
        "//spatialblend/model/relation:affordance_pose_grounding_rule",
        "//spatialblend/model/relation:avatar_affordance_pose_kernel",
        "//spatialblend/model/relation:avatar_facing_kernel",
        "//spatialblend/model/relation:avatar_spacing_kernel",
        "//spatialblend/model/relation:grounding_rule",
        "//spatialblend/model/relation:scope_gate",
        "//spatialblend/model/relation:spacing_grounding_rule",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "solver_layer_test",
    srcs = ["solver_layer_test.cc"],
    deps = [
        ":relation_solver",
        "//spatialblend/geometry:obstacle_free_space",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/relation",
        "//spatialblend/model/relation:avatar_facing_kernel",
        "//spatialblend/model/relation:avatar_spacing_kernel",
        "//spatialblend/model/relation:grounding_rule",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "realization_contract_test",
    srcs = ["realization_contract_test.cc"],
    deps = [
        ":relation_solver",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/testing:hierarchy_extractor_fixture",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "engagement",
    hdrs = ["engagement.h"],
    deps = [
        "//spatialblend/model/realization:remote_update",
    ],
)

# Correctness/quality probes for the paper-objective search (displacement
# cap, Facing-pulls-position coupling, and the 150-scene brute-force probe).
spatialblend_cc_test(
    name = "a1_probe_test",
    srcs = ["a1_probe_test.cc"],
    deps = [
        ":relation_solver",
        "//spatialblend/geometry:obstacle_free_space",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/model/relation",
        "//spatialblend/model/relation:avatar_facing_kernel",
        "//spatialblend/model/relation:avatar_spacing_kernel",
        "//spatialblend/model/relation:grounding_rule",
        "//spatialblend/model/relation:relation_kernel",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# Yaw-stage conditioning-vs-search probe (the safeguard reuse).
spatialblend_cc_test(
    name = "a1_yaw_probe_test",
    srcs = ["a1_yaw_probe_test.cc"],
    deps = [
        ":relation_solver",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/model/relation",
        "//spatialblend/model/relation:avatar_facing_kernel",
        "//spatialblend/model/relation:avatar_spacing_kernel",
        "//spatialblend/model/relation:grounding_rule",
        "//spatialblend/model/relation:relation_kernel",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "nd_trust_region_test",
    srcs = ["nd_trust_region_test.cc"],
    deps = [
        ":relation_solver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "affordance_pose_sd_test",
    srcs = ["affordance_pose_sd_test.cc"],
    deps = [
        ":affordance_pose_sd",
        ":relation_solver",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:polytope_spatial_definition",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "count_rule_test",
    srcs = ["count_rule_test.cc"],
    deps = [
        ":engagement",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/model/relation",
        "//spatialblend/model/relation:avatar_affordance_pose_kernel",
        "//spatialblend/model/relation:avatar_facing_kernel",
        "//spatialblend/model/relation:avatar_spacing_kernel",
        "//spatialblend/model/relation:grounding_rule",
        "//spatialblend/model/relation:support",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)
