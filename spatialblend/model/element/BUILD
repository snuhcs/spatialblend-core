# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "geometry_role",
    hdrs = ["geometry_role.h"],
)

spatialblend_cc_test(
    name = "geometry_role_test",
    srcs = ["geometry_role_test.cc"],
    deps = [
        ":geometry_role",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "spatial_definition",
    hdrs = [
        "spatial_definition.h",
        "sub_region.h",
    ],
    deps = [
        ":geometry_role",
        "//spatialblend/geometry:normalized_interior",
        "//spatialblend/geometry:planar_frame_chart",
        "//spatialblend/geometry:polygon2d",
        "@eigen",
    ],
)

cc_library(
    name = "navigable_spatial_definition",
    srcs = ["navigable_spatial_definition.cc"],
    hdrs = ["navigable_spatial_definition.h"],
    deps = [
        ":spatial_definition",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon_coordinate_mapper",
        "//spatialblend/geometry/context:navigable_geometry_context",
        "//spatialblend/geometry/path:discrete_path",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "spatial_definition_test",
    srcs = ["spatial_definition_test.cc"],
    deps = [
        ":navigable_spatial_definition",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "envelope",
    srcs = ["envelope.cc"],
    hdrs = ["envelope.h"],
    deps = [
        ":navigable_spatial_definition",
        ":spatial_definition",
        "//spatialblend/geometry:polygon2d",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "envelope_test",
    srcs = ["envelope_test.cc"],
    deps = [
        ":envelope",
        ":navigable_spatial_definition",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "element",
    srcs = ["element.cc"],
    hdrs = [
        "affordance_pose.h",
        "element.h",
    ],
    deps = [
        ":geometry_role",
        ":spatial_definition",
        "//spatialblend/base/common",
        "//spatialblend/base/common:types",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "element_test",
    srcs = ["element_test.cc"],
    deps = [
        ":element",
        ":navigable_spatial_definition",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "containment_layout",
    srcs = ["containment_layout.cc"],
    hdrs = ["containment_layout.h"],
    deps = [
        ":element",
        ":element_tree",
        ":geometry_role",
        ":spatial_definition",
        "//spatialblend/base/common:types",
        "//spatialblend/geometry:polygon2d",
        "@eigen",
    ],
)

cc_library(
    name = "scope_extension",
    srcs = ["scope_extension.cc"],
    hdrs = ["scope_extension.h"],
    deps = [
        ":containment_layout",
        ":element",
        ":element_tree",
        ":geometry_role",
        ":navigable_spatial_definition",
        ":polytope_spatial_definition",
        ":spatial_definition",
        "//spatialblend/base/common:types",
        "//spatialblend/geometry:obstacle_free_space",
        "//spatialblend/geometry:polygon2d",
        "@eigen",
    ],
)

cc_library(
    name = "element_tree",
    srcs = ["element_tree.cc"],
    hdrs = ["element_tree.h"],
    deps = [
        ":element",
        "@abseil-cpp//absl/status",
    ],
)

spatialblend_cc_test(
    name = "element_tree_test",
    srcs = ["element_tree_test.cc"],
    deps = [
        ":element_tree",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "surface_spatial_definition",
    srcs = ["surface_spatial_definition.cc"],
    hdrs = ["surface_spatial_definition.h"],
    deps = [
        ":spatial_definition",
        "//spatialblend/geometry:polygon3d",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "surface_spatial_definition_test",
    srcs = ["surface_spatial_definition_test.cc"],
    deps = [
        ":surface_spatial_definition",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "polytope_spatial_definition",
    srcs = ["polytope_spatial_definition.cc"],
    hdrs = ["polytope_spatial_definition.h"],
    deps = [
        ":spatial_definition",
        "//spatialblend/geometry:polygon2d",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "polytope_spatial_definition_test",
    srcs = ["polytope_spatial_definition_test.cc"],
    deps = [
        ":polytope_spatial_definition",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "containment_layout_test",
    srcs = ["containment_layout_test.cc"],
    deps = [
        ":containment_layout",
        "//spatialblend/geometry:polygon2d",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)
