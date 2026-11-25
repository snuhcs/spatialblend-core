# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "hierarchy_extractor_fixture",
    testonly = True,
    srcs = ["hierarchy_extractor_fixture.cc"],
    hdrs = ["hierarchy_extractor_fixture.h"],
    deps = [
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/link:element_linker",
        "@eigen",
    ],
)

spatialblend_cc_test(
    name = "hierarchy_extractor_contract_test",
    srcs = ["hierarchy_extractor_contract_test.cc"],
    deps = [
        ":hierarchy_extractor_fixture",
        "@googletest//:gtest_main",
    ],
)
