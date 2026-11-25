# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

spatialblend_cc_test(
    name = "actor_motion_tracker_test",
    srcs = ["actor_motion_tracker_test.cc"],
    deps = [
        ":actor_motion_tracker",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "actor_motion_tracker",
    srcs = ["actor_motion_tracker.cc"],
    hdrs = ["actor_motion_tracker.h"],
    deps = [
        "@eigen",
    ],
)

cc_library(
    name = "subject_anchor",
    hdrs = ["subject_anchor.h"],
    deps = [
        "//spatialblend/base/common:state_type",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "subject_anchor_test",
    srcs = ["subject_anchor_test.cc"],
    deps = [
        ":subject_anchor",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "actor",
    srcs = ["actor.cc"],
    hdrs = ["actor.h"],
    deps = [
        ":subject_anchor",
        "//spatialblend/base/common",
        "//spatialblend/base/common:types",
        "//spatialblend/base/math",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "actor_test",
    srcs = ["actor_test.cc"],
    deps = [
        ":actor",
        "@googletest//:gtest_main",
    ],
)
