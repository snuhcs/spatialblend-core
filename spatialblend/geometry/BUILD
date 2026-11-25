# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

# spatialblend/geometry/BUILD
load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

spatialblend_cc_test(
    name = "polygon3d_test",
    srcs = ["polygon3d_test.cc"],
    deps = [
        ":polygon3d",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "polygon3d",
    srcs = ["polygon3d.cc"],
    hdrs = ["polygon3d.h"],
    deps = [
        ":polygon2d",
        "@eigen",
    ],
)

cc_library(
    name = "polygon2d",
    srcs = ["polygon2d.cc"],
    hdrs = ["polygon2d.h"],
    deps = [
        "@boost.geometry",
        "@eigen",
    ],
)

cc_library(
    name = "normalized_interior",
    hdrs = ["normalized_interior.h"],
    deps = [
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "polygon2d_test",
    srcs = ["polygon2d_test.cc"],
    deps = [
        ":polygon2d",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "polygon_coordinate_mapper",
    srcs = ["polygon_coordinate_mapper.cc"],
    hdrs = ["polygon_coordinate_mapper.h"],
    deps = [
        ":polygon2d",
        "@abseil-cpp//absl/status:statusor",
        "@eigen",
    ],
)

cc_library(
    name = "planar_frame_chart",
    srcs = ["planar_frame_chart.cc"],
    hdrs = ["planar_frame_chart.h"],
    deps = [
        ":polygon2d",
        "@abseil-cpp//absl/status:status",
        "@abseil-cpp//absl/status:statusor",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "planar_frame_chart_test",
    srcs = ["planar_frame_chart_test.cc"],
    deps = [
        ":planar_frame_chart",
        ":polygon2d",
        ":radial_surface_chart",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "radial_surface_chart",
    srcs = ["radial_surface_chart.cc"],
    hdrs = ["radial_surface_chart.h"],
    deps = [
        ":polygon2d",
        ":polygon_coordinate_mapper",
        "@abseil-cpp//absl/status:status",
        "@abseil-cpp//absl/status:statusor",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "radial_surface_chart_test",
    srcs = ["radial_surface_chart_test.cc"],
    deps = [
        ":polygon2d",
        ":radial_surface_chart",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "polygon_coordinate_mapper_test",
    srcs = ["polygon_coordinate_mapper_test.cc"],
    deps = [
        ":polygon2d",
        ":polygon_coordinate_mapper",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "visibility_graph",
    srcs = ["visibility_graph.cc"],
    hdrs = ["visibility_graph.h"],
    deps = [
        ":polygon2d",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "visibility_graph_test",
    srcs = ["visibility_graph_test.cc"],
    deps = [
        ":visibility_graph",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "convex_decomposition",
    srcs = ["convex_decomposition.cc"],
    hdrs = ["convex_decomposition.h"],
    deps = [
        ":polygon2d",
        "@CDT",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "convex_decomposition_test",
    srcs = ["convex_decomposition_test.cc"],
    deps = [
        ":convex_decomposition",
        ":polygon2d",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "obstacle_free_space",
    srcs = ["obstacle_free_space.cc"],
    hdrs = ["obstacle_free_space.h"],
    deps = [
        ":polygon2d",
        "@boost.geometry",
        "@eigen",
    ],
)

# ---- from the shared base (formerly spatialblend/geometry) ----
