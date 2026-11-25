# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:spatialblend.bzl", "spatialblend_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "geometry_component_probe",
    srcs = ["geometry_component_probe.cc"],
    hdrs = ["geometry_component_probe.h"],
    deps = [
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/geometry:polygon_coordinate_mapper",
        "//spatialblend/model/element",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/solver:relation_solver",
        "//spatialblend/space",
        "//spatialblend/space:space_hub",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@eigen",
        "@nlohmann_json//:json",
    ],
)

cc_binary(
    name = "geometry_component_probe_main",
    srcs = ["geometry_component_probe_main.cc"],
    deps = [
        ":geometry_component_probe",
        "@nlohmann_json//:json",
    ],
)

spatialblend_cc_test(
    name = "geometry_component_probe_test",
    srcs = ["geometry_component_probe_test.cc"],
    deps = [
        ":geometry_component_probe",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

test_suite(
    name = "hierarchy_extractor_contracts",
    tests = [
        "//spatialblend/model/relation:hierarchy_extractor_contract_test",
        "//spatialblend/model/testing:hierarchy_extractor_contract_test",
    ],
)

test_suite(
    name = "component_contracts",
    tests = [
        ":hierarchy_extractor_contracts",
        "//spatialblend/model/relation:affordance_pose_grounding_rule_test",
        "//spatialblend/solver:realization_contract_test",
        "//spatialblend/solver:realization_policy_test",
        "//spatialblend/space:seat_contact_contract_test",
    ],
)

cc_library(
    name = "scenario",
    srcs = ["scenario.cc"],
    hdrs = ["scenario.h"],
    deps = [
        "//spatialblend/base/common:types",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/actor",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:envelope",
        "//spatialblend/model/element:geometry_role",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:spatial_definition",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/space",
        "//spatialblend/space:local_context",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@eigen",
        "@nlohmann_json//:json",
    ],
)

spatialblend_cc_test(
    name = "scenario_test",
    srcs = ["scenario_test.cc"],
    data = glob(["testdata/**"]),
    deps = [
        ":scenario",
        "//spatialblend/space:local_context",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

cc_library(
    name = "unity_twin",
    testonly = True,
    srcs = ["unity_twin.cc"],
    hdrs = ["unity_twin.h"],
    deps = [
        ":scenario",
        "//spatialblend/api/c",
        "//spatialblend/space/testing:test_driver",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@eigen",
        "@nlohmann_json//:json",
    ],
)

spatialblend_cc_test(
    name = "harness_test",
    srcs = ["harness_test.cc"],
    data = glob(["testdata/regression/**"]),
    deps = [
        ":scenario",
        ":unity_twin",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

# Standalone regression test for the eval/Unity T1-identity-room fixture
# (a separate binary from harness_test). Drives the literal shared spec fixture
# (testdata/regression/T1-identity-room.scenario.json) through the real C API end-to-end, the same way harness_test's
# R1-R9 fixtures do.
spatialblend_cc_test(
    name = "c19b_identity_room_test",
    srcs = ["c19b_identity_room_test.cc"],
    data = glob(["testdata/regression/**"]),
    deps = [
        ":scenario",
        ":unity_twin",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest_main",
    ],
)
