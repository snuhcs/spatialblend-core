# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "seqlock_box",
    hdrs = ["seqlock_box.h"],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
)

cc_library(
    name = "state_type",
    hdrs = ["state_type.h"],
)

spatialblend_cc_test(
    name = "types_test",
    srcs = ["types_test.cc"],
    deps = [
        ":types",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "trace",
    srcs = ["trace.cc"],
    hdrs = ["trace.h"],
    deps = [
        ":types",
        "@nlohmann_json//:json",
    ],
)

spatialblend_cc_test(
    name = "trace_test",
    srcs = ["trace_test.cc"],
    deps = [
        ":trace",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

# ---- from the shared base (formerly spatialblend/base/common) ----

cc_library(
    name = "coalescing_worker",
    hdrs = ["coalescing_worker.h"],
)

cc_library(
    name = "common",
    srcs = [
        "common.cc",
    ],
    hdrs = [
        "common.h",
    ],
    deps = [
        ":types",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
    ],
)
