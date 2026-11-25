# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # MIT
)

cc_library(
    name = "network_util",
    srcs = [
        "network_util.cc",
    ] + select({
        "@platforms//os:android": ["android_network.cc"],
        "//conditions:default": [],
    }),
    hdrs = [
        "network_util.h",
    ] + select({
        "@platforms//os:android": ["android_network.h"],
        "//conditions:default": [],
    }),
    copts = select({
        "@platforms//os:android": ["-D__ANDROID__"],
        "//conditions:default": [],
    }),
    deps = [
        "//spatialblend/base/common",
        "//spatialblend/base/util:logger",
    ],
)

# General network utility tests for all platforms
