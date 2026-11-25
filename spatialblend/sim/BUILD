# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "simulator",
    testonly = True,
    srcs = ["simulator.cc"],
    hdrs = ["simulator.h"],
    deps = [
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/solver:relation_solver",
        "//spatialblend/space",
        "//spatialblend/space:binding",
        "//spatialblend/space:local_context",
        "//spatialblend/space/builder:actor_handle",
        "//spatialblend/space/builder:space_builder",
        "//spatialblend/space/builder:trajectory",
        "//spatialblend/space/testing:test_driver",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "simulator_test",
    srcs = ["simulator_test.cc"],
    deps = [
        ":simulator",
        "//spatialblend/space/builder:kernel_shorthands",
        "//spatialblend/space/builder:trajectory",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "anchor_realization_test",
    srcs = ["anchor_realization_test.cc"],
    deps = [
        ":simulator",
        "//spatialblend/base/common:trace",
        "//spatialblend/space/builder:kernel_shorthands",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

spatialblend_cc_test(
    name = "scenario_replay_test",
    srcs = ["scenario_replay_test.cc"],
    deps = [
        ":simulator",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/space/builder:kernel_shorthands",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "cross_validation_test",
    srcs = ["cross_validation_test.cc"],
    deps = [
        ":simulator",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/space/builder:kernel_shorthands",
        "//spatialblend/space/builder:trajectory",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)
