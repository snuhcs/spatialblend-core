# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "support",
    hdrs = ["support.h"],
    deps = [
        "//spatialblend/base/common:types",
        "//spatialblend/geometry:normalized_interior",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon_coordinate_mapper",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "support_test",
    srcs = ["support_test.cc"],
    deps = [
        ":support",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "grounded_relation",
    hdrs = ["grounded_relation.h"],
    deps = ["@eigen"],
)

spatialblend_cc_test(
    name = "grounded_relation_test",
    srcs = ["grounded_relation_test.cc"],
    deps = [
        ":grounded_relation",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "grounding_rule",
    hdrs = ["grounding_rule.h"],
    deps = [
        ":support",
        "//spatialblend/model/element",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
    ],
)

cc_library(
    name = "relation_kernel",
    hdrs = ["relation_kernel.h"],
    deps = [
        ":grounded_relation",
        ":support",
        "//spatialblend/base/common:state_type",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon_coordinate_mapper",
        "//spatialblend/model/actor:actor_motion_tracker",
        "//spatialblend/model/element",
        "//spatialblend/model/element:element_tree",
        "@eigen",
    ],
)

cc_library(
    name = "scope_resolver",
    hdrs = ["scope_resolver.h"],
    deps = [
        ":relation_kernel",
        "//spatialblend/base/common:types",
        "//spatialblend/model/element",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:geometry_role",
        "@eigen",
    ],
)

cc_library(
    name = "avatar_facing_kernel",
    srcs = ["avatar_facing_kernel.cc"],
    hdrs = ["avatar_facing_kernel.h"],
    deps = [
        ":grounded_relation",
        ":relation_kernel",
        ":scope_resolver",
        ":support",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "avatar_facing_kernel_test",
    srcs = ["avatar_facing_kernel_test.cc"],
    deps = [
        ":avatar_facing_kernel",
        ":support",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "relation",
    srcs = ["relation.cc"],
    hdrs = ["relation.h"],
    deps = [
        ":grounded_relation",
        ":grounding_rule",
        ":relation_kernel",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "relation_test",
    srcs = ["relation_test.cc"],
    deps = [
        ":grounding_rule",
        ":relation",
        "//spatialblend/model/element:element_tree",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "avatar_spacing_kernel",
    srcs = ["avatar_spacing_kernel.cc"],
    hdrs = ["avatar_spacing_kernel.h"],
    deps = [
        ":grounded_relation",
        ":relation_kernel",
        ":scope_resolver",
        ":support",
        "//spatialblend/model/element:navigable_spatial_definition",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "avatar_spacing_kernel_test",
    srcs = ["avatar_spacing_kernel_test.cc"],
    deps = [
        ":avatar_spacing_kernel",
        ":support",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "avatar_affordance_pose_kernel",
    srcs = ["avatar_affordance_pose_kernel.cc"],
    hdrs = ["avatar_affordance_pose_kernel.h"],
    deps = [
        ":grounded_relation",
        ":relation_kernel",
        ":scope_resolver",
        ":support",
        "//spatialblend/geometry:polygon_coordinate_mapper",
        "//spatialblend/model/element",
        "//spatialblend/model/element:element_tree",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "avatar_affordance_pose_kernel_test",
    srcs = ["avatar_affordance_pose_kernel_test.cc"],
    deps = [
        ":affordance_pose_grounding_rule",
        ":avatar_affordance_pose_kernel",
        ":support",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "kernel_factory",
    hdrs = ["kernel_factory.h"],
    deps = [
        ":affordance_pose_grounding_rule",
        ":avatar_affordance_pose_kernel",
        ":avatar_facing_kernel",
        ":avatar_spacing_kernel",
        ":facing_grounding_rule",
        ":relation_kernel",
        ":spacing_grounding_rule",
    ],
)

spatialblend_cc_test(
    name = "kernel_factory_test",
    srcs = ["kernel_factory_test.cc"],
    deps = [
        ":kernel_factory",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "spacing_grounding_rule",
    srcs = ["spacing_grounding_rule.cc"],
    hdrs = ["spacing_grounding_rule.h"],
    deps = [
        ":grounding_rule",
        ":support",
        "//spatialblend/geometry:polygon_coordinate_mapper",
        "//spatialblend/model/element",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
    ],
)

cc_library(
    name = "facing_grounding_rule",
    srcs = ["facing_grounding_rule.cc"],
    hdrs = ["facing_grounding_rule.h"],
    deps = [
        ":grounding_rule",
        ":support",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon_coordinate_mapper",
        "//spatialblend/model/element",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
    ],
)

cc_library(
    name = "affordance_pose_grounding_rule",
    srcs = ["affordance_pose_grounding_rule.cc"],
    hdrs = ["affordance_pose_grounding_rule.h"],
    deps = [
        ":grounding_rule",
        ":support",
        "//spatialblend/geometry:radial_surface_chart",
        "//spatialblend/model/element",
        "//spatialblend/model/element:element_tree",
    ],
)

spatialblend_cc_test(
    name = "affordance_pose_grounding_rule_test",
    srcs = ["affordance_pose_grounding_rule_test.cc"],
    deps = [
        ":affordance_pose_grounding_rule",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "scope_layer_test",
    srcs = ["scope_layer_test.cc"],
    deps = [
        ":scope_resolver",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "resolve_scope_test",
    srcs = ["resolve_scope_test.cc"],
    deps = [
        ":avatar_affordance_pose_kernel",
        ":avatar_facing_kernel",
        ":avatar_spacing_kernel",
        ":scope_resolver",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "hierarchy_extractor_contract_test",
    srcs = ["hierarchy_extractor_contract_test.cc"],
    deps = [
        ":affordance_pose_grounding_rule",
        ":avatar_affordance_pose_kernel",
        ":relation_kernel",
        ":scope_resolver",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/testing:hierarchy_extractor_fixture",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "grounding_rule_test",
    srcs = ["grounding_rule_test.cc"],
    deps = [
        ":affordance_pose_grounding_rule",
        ":avatar_spacing_kernel",
        ":facing_grounding_rule",
        ":spacing_grounding_rule",
        "//spatialblend/geometry:obstacle_free_space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "@googletest//:gtest_main",
    ],
)

# Per-family geometric scope gates (line of sight, walkable
# reachability). See scope_gate.h.
cc_library(
    name = "scope_gate",
    srcs = ["scope_gate.cc"],
    hdrs = ["scope_gate.h"],
    deps = [
        ":relation_kernel",
        ":support",
        "//spatialblend/base/common:types",
        "//spatialblend/geometry:obstacle_free_space",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "scope_gate_test",
    srcs = ["scope_gate_test.cc"],
    deps = [
        ":scope_gate",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "@googletest//:gtest_main",
    ],
)
