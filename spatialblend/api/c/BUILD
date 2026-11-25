# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("//bazel:spatialblend.bzl", "spatialblend_cc_test", "spatialblend_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "trace_format",
    srcs = ["trace.format.md"],
)

cc_library(
    name = "c",
    srcs = [
        "logical_content_ingress_c.cc",
        "spatialblend_c.cc",
    ],
    hdrs = [
        "spatialblend.h",
        "spatialblend_containment_transfer.h",
        "spatialblend_logical_content_ingress.h",
        "spatialblend_observability.h",
        "spatialblend_observed_collaborator.h",
        "spatialblend_observed_source_setup.h",
        "spatialblend_planar_support.h",
        "spatialblend_receiver_approach_context.h",
        "spatialblend_receiver_approach_feasibility.h",
        "spatialblend_receiver_input.h",
        "spatialblend_receiver_reevaluation.h",
        "spatialblend_rich.h",
        "spatialblend_source_evidence.h",
        "spatialblend_type.h",
    ],
    textual_hdrs = [
        "receiver_input_c.inc",
        "receiver_reevaluation_c.inc",
        "planar_support_c.inc",
        "containment_transfer_c.inc",
        "source_evidence_c.inc",
        "receiver_approach_context_c.inc",
        "receiver_approach_feasibility_c.inc",
        "owned_correspondence_c.inc",
        "observability_c.inc",
        "tunnel_c.inc",
    ],
    deps = [
        "//spatialblend/base/common:trace",
        "//spatialblend/geometry:planar_frame_chart",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/geometry:radial_surface_chart",
        "//spatialblend/model/actor",
        "//spatialblend/model/element:envelope",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/link:element_linker",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/net:network",
        "//spatialblend/space",
        "//spatialblend/space:local_context",
        "//spatialblend/space:space_hub",
        "@eigen",
    ],
    alwayslink = True,
)

cc_shared_library(
    name = "c_shared",
    additional_linker_inputs = select({
        "@platforms//os:macos": [],
        "@platforms//os:visionos": [],
        # Android is a distinct constraint from linux; without its own branch
        # the version script fell through and every dependency symbol
        # (BoringSSL, gRPC, absl, protobuf) was exported from the Quest .so.
        "@platforms//os:android": [
            ":version_script.lds",
        ],
        "@platforms//os:linux": [
            ":version_script.lds",
        ],
        "//conditions:default": [],
    }),
    # The core library: libspatialblend (libspatialblend_d for --config=debug), one name per platform.
    shared_lib_name = select({
        "//bazel:debug": "libspatialblend_d",
        "//conditions:default": "libspatialblend",
    }) + select({
        "@platforms//os:linux": ".so",
        "@platforms//os:macos": ".dylib",
        "@platforms//os:windows": ".dll",
        "@platforms//os:visionos": ".dylib",
        "//conditions:default": ".so",
    }),
    user_link_flags = spatialblend_linkopts() + select({
        "@platforms//os:macos": ["-Wl,-exported_symbol,_sb_*"],
        "@platforms//os:visionos": ["-Wl,-exported_symbol,_sb_*"],
        "@platforms//os:android": [
            "-Wl,--version-script=$(location :version_script.lds)",
        ],
        "@platforms//os:linux": [
            "-Wl,--version-script=$(location :version_script.lds)",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":c",
    ],
)

cc_import(
    name = "c_shared_import",
    hdrs = [":c_shared"],
    shared_library = ":c_shared",
)

filegroup(
    name = "c_api_headers",
    # Test-only deterministic-clock declarations are intentionally absent
    # from the installable C header package as well as c_shared.
    srcs = glob(
        ["**/*.h"],
        exclude = [
            "receiver_approach_feasibility_test_api.h",
            "scope_extension_test_api.h",
        ],
    ),
)

filegroup(
    name = "c_shared_pdb",
    srcs = [":c_shared"],
    output_group = "pdb_file",
)

pkg_tar(
    name = "libspatialblend_tar",
    srcs = [":c_shared"],
    package_dir = "lib",
)

pkg_tar(
    name = "headers_tar",
    srcs = [":c_api_headers"],
    package_dir = "include/spatialblend",
)

pkg_tar(
    name = "libspatialblend_c",
    srcs = select({
        "//bazel:debug": [":c_shared_pdb"],
        "//conditions:default": [],
    }),
    deps = [
        ":headers_tar",
        ":libspatialblend_tar",
    ],
)

cc_library(
    name = "c_test_embodied",
    testonly = True,
    hdrs = ["c_test_embodied.h"],
    deps = [":c"],
)

spatialblend_cc_test(
    name = "planar_support_c_test",
    srcs = ["planar_support_c_test.cc"],
    deps = [
        ":c",
        "@googletest//:gtest_main",
    ],
)

# A duplicate, private C owner compiled only for the approach feasibility's deterministic lifetime
# contract.  It is never a dependency of c_shared and its test-only clock API
# is absent from shipping headers/exports.
cc_library(
    name = "c_receiver_approach_testable",
    testonly = True,
    srcs = ["spatialblend_c.cc"],
    hdrs = [
        "spatialblend.h",
        "spatialblend_type.h",
        "spatialblend_rich.h",
        "spatialblend_receiver_input.h",
        "spatialblend_receiver_reevaluation.h",
        "spatialblend_planar_support.h",
        "spatialblend_containment_transfer.h",
        "spatialblend_source_evidence.h",
        "spatialblend_receiver_approach_context.h",
        "spatialblend_receiver_approach_feasibility.h",
        # spatialblend_c.cc includes the public umbrella even in this private
        # test-only owner.  Keep its direct headers declared so additions to
        # the shipping umbrella cannot silently break this hermetic test gate.
        "spatialblend_observed_collaborator.h",
        "spatialblend_observed_source_setup.h",
        "spatialblend_observability.h",
        "receiver_approach_feasibility_test_api.h",
        "scope_extension_test_api.h",
    ],
    defines = ["SPATIALBLEND_V2_C_TESTING"],
    textual_hdrs = [
        "receiver_input_c.inc",
        "receiver_reevaluation_c.inc",
        "planar_support_c.inc",
        "containment_transfer_c.inc",
        "source_evidence_c.inc",
        "receiver_approach_context_c.inc",
        "receiver_approach_feasibility_c.inc",
        "owned_correspondence_c.inc",
        "observability_c.inc",
        "tunnel_c.inc",
    ],
    visibility = [":__pkg__"],
    deps = [
        "//spatialblend/base/common:trace",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/model/actor",
        "//spatialblend/model/element:envelope",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/link:element_linker",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/net:network",
        "//spatialblend/space",
        "//spatialblend/space:local_context",
        "//spatialblend/space:space_hub",
        "@eigen",
    ],
    alwayslink = True,
)

# Read-only copied recovery ingress for logical content.  This is intentionally not a
# room-client dependency: it has no scheduler, socket, authority or writer.
cc_library(
    name = "logical_content_ingress",
    srcs = ["logical_content_ingress_c.cc"],
    hdrs = ["spatialblend_logical_content_ingress.h"],
    alwayslink = True,
)

spatialblend_cc_test(
    name = "owned_correspondence_c_test",
    srcs = ["owned_correspondence_c_test.cc"],
    deps = [
        ":c",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "source_evidence_c_test",
    srcs = ["source_evidence_c_test.cc"],
    deps = [
        ":c",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "receiver_input_c_test",
    srcs = ["receiver_input_c_test.cc"],
    deps = [
        ":c",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "receiver_approach_context_c_test",
    srcs = ["receiver_approach_context_c_test.cc"],
    deps = [
        ":c",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "receiver_approach_feasibility_c_test",
    srcs = ["receiver_approach_feasibility_c_test.cc"],
    deps = [
        ":c",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "receiver_approach_feasibility_c_lifetime_test",
    srcs = ["receiver_approach_feasibility_c_lifetime_test.cc"],
    deps = [
        ":c_receiver_approach_testable",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "observed_collaborator_c_test",
    srcs = ["observed_collaborator_c_test.cc"],
    deps = [
        ":c",
        ":c_test_embodied",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "logical_content_ingress_c_test",
    srcs = ["logical_content_ingress_c_test.cc"],
    deps = [
        ":logical_content_ingress",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "port_concurrency_test",
    srcs = ["port_concurrency_test.cc"],
    tags = ["exclusive"],
    deps = [
        ":c",
        ":c_test_embodied",
        "//spatialblend/space",
        "@googletest//:gtest_main",
    ],
)

spatialblend_cc_test(
    name = "c_api_test",
    srcs = ["c_api_test.cc"],
    tags = ["exclusive"],
    deps = [
        ":c",
        ":c_test_embodied",
        "@googletest//:gtest_main",
    ],
)

# Containment transfer feature header: layout, local polygon setter, flag,
# per-actor getter and the contained / uncontained event.
spatialblend_cc_test(
    name = "containment_transfer_c_test",
    srcs = ["containment_transfer_c_test.cc"],
    deps = [
        ":c",
        ":c_test_embodied",
        "@googletest//:gtest_main",
    ],
)

# ABI 4.5: the tunneled core link, pumped only through sb_tunnel_write/read. POSIX
# only; the Windows build has the calls but they fail (see tunnel_link.cc).
spatialblend_cc_test(
    name = "tunnel_c_test",
    size = "large",
    srcs = ["tunnel_c_test.cc"],
    tags = ["exclusive"],
    target_compatible_with = select({
        "@platforms//os:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":c",
        ":c_test_embodied",
        "@googletest//:gtest_main",
    ],
)

# ABI 4.10: per-part surface charts (sb_set_sub_region_chart, sb_part_chart_encode / _decode).
spatialblend_cc_test(
    name = "part_chart_c_test",
    srcs = ["part_chart_c_test.cc"],
    deps = [
        ":c",
        "@googletest//:gtest_main",
    ],
)

# sb_unlink and the link-removed signal, over local and TCP transports.
spatialblend_cc_test(
    name = "unlink_c_test",
    srcs = ["unlink_c_test.cc"],
    tags = ["exclusive"],
    deps = [
        ":c",
        "@googletest//:gtest_main",
    ],
)

# T1.1: public-C-API relation events for a received remote actor.
spatialblend_cc_test(
    name = "relation_event_c_test",
    srcs = ["relation_event_c_test.cc"],
    tags = ["exclusive"],
    deps = [
        ":c",
        ":c_test_embodied",
        "@googletest//:gtest_main",
    ],
)

# Observability and injection surface.
spatialblend_cc_test(
    name = "observability_c_test",
    srcs = ["observability_c_test.cc"],
    deps = [
        ":c",
        ":c_test_embodied",
        "//spatialblend/model/realization:remote_update",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

# Every line of a traced two-space run parses against the field tables
# of spatialblend/api/c/trace.format.md.
spatialblend_cc_test(
    name = "trace_spec_conformance_test",
    srcs = ["trace_spec_conformance_test.cc"],
    data = [":trace_format"],
    deps = [
        ":c",
        ":c_test_embodied",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

# Per-tick cost of observability capture (off vs on).
cc_binary(
    name = "observability_overhead_bench",
    srcs = ["observability_overhead_bench.cc"],
    deps = [":c"],
)

# Source binding revision on rich contributors.
spatialblend_cc_test(
    name = "binding_revision_c_test",
    srcs = ["binding_revision_c_test.cc"],
    deps = [
        ":c",
        ":c_test_embodied",
        "@googletest//:gtest_main",
    ],
)

# ABI 4.7: embodiment registration, bone ports, the normalized kinematic chain.
spatialblend_cc_test(
    name = "embodiment_c_test",
    srcs = ["embodiment_c_test.cc"],
    tags = ["exclusive"],
    deps = [
        ":c",
        "@googletest//:gtest_main",
    ],
)

# ABI 4.11: affordance poses on interaction-role children and sb_read_anchor_solve.
spatialblend_cc_test(
    name = "anchor_solve_c_test",
    srcs = ["anchor_solve_c_test.cc"],
    deps = [
        ":c",
        "@googletest//:gtest_main",
    ],
)

# Spacing to a containment region with the source outside it: the receiver keeps the distance.
# Linked against the test-only C owner for sb_test_wait_for_scope_extension (the
# embodied helpers are header-only, listed as a source so :c is not linked twice).
spatialblend_cc_test(
    name = "spacing_outside_region_c_test",
    srcs = [
        "c_test_embodied.h",
        "spacing_outside_region_c_test.cc",
    ],
    deps = [
        ":c_receiver_approach_testable",
        "@googletest//:gtest_main",
    ],
)
