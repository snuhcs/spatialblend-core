# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Skeleton/anchor definitions and kinematic arithmetic; no runtime state.
cc_library(
    name = "embodiment",
    srcs = ["embodiment.cc"],
    hdrs = ["embodiment.h"],
    deps = [
        "@eigen",
        "@nlohmann_json//:json",
    ],
)

spatialblend_cc_test(
    name = "embodiment_test",
    srcs = ["embodiment_test.cc"],
    deps = [
        ":embodiment",
        "@eigen",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)
