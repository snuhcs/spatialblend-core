# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

spatialblend_cc_test(
    name = "trajectory_test",
    srcs = ["trajectory_test.cc"],
    deps = [
        ":trajectory",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "trajectory",
    srcs = ["trajectory.cc"],
    hdrs = ["trajectory.h"],
    deps = ["@eigen"],
)

cc_library(
    name = "space_builder",
    hdrs = ["space_builder.h"],
    deps = [
        ":actor_handle",
        ":element_handle",
        "//spatialblend/space",
    ],
)

cc_library(
    name = "kernel_shorthands",
    hdrs = ["kernel_shorthands.h"],
    deps = ["//spatialblend/model/relation:relation_kernel"],
)

cc_library(
    name = "element_handle",
    hdrs = ["element_handle.h"],
    deps = [
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/space",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "builder_test",
    srcs = ["builder_test.cc"],
    deps = [
        ":kernel_shorthands",
        ":space_builder",
        "//spatialblend/space:local_context",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "actor_handle",
    hdrs = ["actor_handle.h"],
    deps = [
        ":element_handle",
        "//spatialblend/model/actor",
        "//spatialblend/space",
        "@eigen",
    ],
)
