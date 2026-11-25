# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # MIT
)

cc_library(
    name = "logger",
    srcs = [
        "logger.cc",
    ],
    hdrs = [
        "logger.h",
    ],
    deps = [
        "//spatialblend/base/common",
    ],
)

cc_library(
    name = "util",
    srcs = [
        "util.cc",
    ],
    hdrs = [
        "util.h",
    ],
    deps = [
        ":logger",
        "//spatialblend/base/common",
    ],
)
