# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

spatialblend_cc_test(
    name = "solver_grounding_requirements_test",
    srcs = ["solver_grounding_requirements_test.cc"],
    deps = [
        ":proto_util",
        "//spatialblend/geometry:radial_surface_chart",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/model/relation:kernel_factory",
        "//spatialblend/solver:relation_solver",
        "//spatialblend/space",
        "//spatialblend/space:local_context",
        "//spatialblend/space:space_hub",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "runtime_element_e2e_test",
    srcs = ["runtime_element_e2e_test.cc"],
    tags = ["exclusive"],
    deps = [
        ":network",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space",
        "//spatialblend/space:local_context",
        "//spatialblend/space:space_hub",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "cross_space_test",
    srcs = ["cross_space_test.cc"],
    deps = [
        ":proto_util",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space",
        "//spatialblend/space:local_context",
        "//spatialblend/space:space_hub",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "proto_util",
    srcs = ["proto_util.cc"],
    hdrs = ["proto_util.h"],
    deps = [
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/model/relation:grounded_relation",
        "//spatialblend/model/relation:support",
        "//spatialblend/net/proto:spatialblend_cc_proto",
        "//spatialblend/space:relation_event",
        "@eigen",
    ],
)

cc_library(
    name = "network",
    srcs = [
        "call/hub_receiver_call.cc",
        "call/hub_sender_call.cc",
        "hub_receiver.cc",
        "remote_hub.cc",
        "remote_space.cc",
        "tunnel_link.cc",
    ],
    hdrs = [
        "call/hub_receiver_call.h",
        "call/hub_sender_call.h",
        "hub_receiver.h",
        "remote_hub.h",
        "remote_space.h",
        "tunnel_link.h",
    ],
    copts = select({
        "@platforms//os:linux": [
            "-Wno-unused-parameter",
            "-Wno-sign-compare",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":proto_util",
        "//spatialblend/base/common",
        "//spatialblend/base/common:trace",
        "//spatialblend/base/common:types",
        "//spatialblend/base/util",
        "//spatialblend/model/actor",
        "//spatialblend/model/actor:subject_anchor",
        "//spatialblend/model/element",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/net/grpc",
        "//spatialblend/net/proto",
        "//spatialblend/net/proto:spatialblend_cc_proto",
        "//spatialblend/net/util:network_util",
        "//spatialblend/space",
        "//spatialblend/space:space_hub",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@eigen",
        "@grpc//:grpc++",
    ],
)

spatialblend_cc_test(
    name = "proto_util_test",
    srcs = ["proto_util_test.cc"],
    deps = [
        ":proto_util",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/realization:containment_transfer",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/model/relation:grounded_relation",
        "//spatialblend/model/relation:support",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "network_test",
    size = "large",
    srcs = ["network_test.cc"],
    tags = ["exclusive"],
    deps = [
        ":network",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/actor",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "network_tunnel_test",
    size = "large",
    srcs = ["network_test.cc"],
    # The same tests over the tunneled core link: no listening socket, no dial.
    local_defines = ["SPATIALBLEND_NETWORK_TEST_TUNNEL"],
    tags = ["exclusive"],
    target_compatible_with = select({
        "@platforms//os:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":network",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/actor",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space",
        "//spatialblend/space/testing:ideal_driver",
        "@eigen",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "tunnel_test",
    size = "large",
    srcs = ["tunnel_test.cc"],
    tags = ["exclusive"],
    target_compatible_with = select({
        "@platforms//os:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":network",
        "//spatialblend/space",
        "//spatialblend/space:space_hub",
        "@googletest//:gtest_main",
    ],
)
