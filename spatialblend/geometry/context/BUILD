# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "surface_geometry_context",
    srcs = ["surface_geometry_context.cc"],
    hdrs = ["surface_geometry_context.h"],
    deps = [
        ":geometry_context",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon3d",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "obstacle_free_space_test",
    srcs = ["obstacle_free_space_test.cc"],
    deps = [
        ":navigable_geometry_context",
        "//spatialblend/geometry:obstacle_free_space",
        "//spatialblend/geometry:polygon2d",
        "@boost.multiprecision",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "navigable_geometry_context_test",
    srcs = ["navigable_geometry_context_test.cc"],
    deps = [
        ":navigable_geometry_context",
        "//spatialblend/geometry:polygon2d",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "navigable_geometry_context",
    srcs = ["navigable_geometry_context.cc"],
    hdrs = ["navigable_geometry_context.h"],
    deps = [
        ":geometry_context",
        "//spatialblend/geometry:convex_decomposition",
        "//spatialblend/geometry:obstacle_free_space",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:visibility_graph",
        "//spatialblend/geometry/path:astar_path_finder",
        "//spatialblend/geometry/path:discrete_path",
        "@eigen",
    ],
)

cc_library(
    name = "geometry_context",
    hdrs = ["geometry_context.h"],
    deps = [
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry/path:discrete_path",
        "@eigen",
    ],
)
