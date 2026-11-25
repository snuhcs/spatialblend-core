# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # MIT
)

cc_library(
    name = "grpc",
    hdrs = [
        "grpc_client_base.h",
        "grpc_server_base.h",
    ],
    copts = [
        "-Wno-unused-parameter",
        "-Wunknown-warning-option",
        "-Wattributes",
        "-Wunknown-attributes",
    ] + select({
        "@platforms//os:windows": ["/std:c++17"],
        "//conditions:default": ["-std=c++17"],
    }),
    deps = [
        "//spatialblend/base/util",
        "//spatialblend/net/util:network_util",
        "@grpc//:grpc++_reflection",
    ],
)
