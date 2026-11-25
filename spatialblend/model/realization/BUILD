# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "receiver_input",
    srcs = ["receiver_input.cc"],
    hdrs = ["receiver_input.h"],
    deps = ["@eigen"],
)

cc_library(
    name = "containment_transfer",
    hdrs = ["containment_transfer.h"],
    deps = [
        "//spatialblend/base/common:types",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:radial_surface_chart",
        "@eigen",
    ],
)

cc_library(
    name = "remote_update",
    hdrs = ["remote_update.h"],
    deps = [
        ":containment_transfer",
        "//spatialblend/base/common:types",
        "//spatialblend/model/relation:grounded_relation",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/model/relation:support",
    ],
)

spatialblend_cc_test(
    name = "remote_update_test",
    srcs = ["remote_update_test.cc"],
    deps = [
        ":remote_update",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)
