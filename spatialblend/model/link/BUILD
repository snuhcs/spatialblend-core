# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "element_linker",
    srcs = ["element_linker.cc"],
    hdrs = ["element_linker.h"],
    deps = [
        "//spatialblend/base/common:types",
    ],
)

spatialblend_cc_test(
    name = "element_linker_test",
    srcs = ["element_linker_test.cc"],
    deps = [
        ":element_linker",
        "@googletest//:gtest_main",
    ],
)
