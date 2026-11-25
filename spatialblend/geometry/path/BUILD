# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "discrete_path",
    srcs = ["discrete_path.cc"],
    hdrs = ["discrete_path.h"],
    deps = ["@eigen"],
)

cc_library(
    name = "path_finder",
    hdrs = ["path_finder.h"],
    deps = [":discrete_path"],
)

cc_library(
    name = "astar_path_finder",
    srcs = ["astar_path_finder.cc"],
    hdrs = ["astar_path_finder.h"],
    deps = [
        ":discrete_path",
        ":path_finder",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:visibility_graph",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "astar_path_finder_test",
    srcs = ["astar_path_finder_test.cc"],
    deps = [
        ":astar_path_finder",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:visibility_graph",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "bezier_course",
    srcs = ["bezier_course.cc"],
    hdrs = ["bezier_course.h"],
    deps = ["@eigen"],
)

spatialblend_cc_test(
    name = "bezier_course_test",
    srcs = ["bezier_course_test.cc"],
    deps = [
        ":bezier_course",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# ---- from the shared base (formerly spatialblend/geometry/path) ----
