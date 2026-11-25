# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("@grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

proto_library(
    name = "spatialblend_proto",
    srcs = ["spatialblend.proto"],
)

cc_proto_library(
    name = "spatialblend_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":spatialblend_proto"],
)

cc_grpc_library(
    name = "proto",
    srcs = [":spatialblend_proto"],
    grpc_only = True,
    visibility = ["//visibility:public"],
    deps = [
        ":spatialblend_cc_proto",
        "@grpc//:grpc++_reflection",
    ],
)
