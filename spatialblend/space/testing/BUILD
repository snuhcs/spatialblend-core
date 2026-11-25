# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# The scope case catalog: read by //spatialblend/space:scope_cases_test and the scenario level's coverage check.
exports_files(["SCOPE_CASES.md"])

cc_library(
    name = "ideal_driver",
    testonly = True,
    hdrs = ["ideal_driver.h"],
    deps = [
        "//spatialblend/space",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "test_driver_test",
    srcs = ["test_driver_test.cc"],
    deps = [
        ":test_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

# Trajectory-scenario validation harness for Spacing / Facing realization.
# The design notes are in the headers.
# Test double standing in for the application's avatar controller (see the header).
cc_library(
    name = "test_driver",
    testonly = True,
    hdrs = ["test_driver.h"],
    deps = ["@eigen"],
)

# The harness's local-minimum oracle of a multi-relation answer (runner.cc and //spatialblend/space:core_integration_test).
cc_library(
    name = "local_minimum",
    testonly = True,
    srcs = ["local_minimum.cc"],
    hdrs = [
        "geom.h",
        "local_minimum.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//spatialblend/solver:relation_solver",
        "//spatialblend/space",
        "@eigen",
    ],
)
