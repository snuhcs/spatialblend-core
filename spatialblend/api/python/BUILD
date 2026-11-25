# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("@spatialblend_deps//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Platform-specific basename for the shared library packaged into the
# spatialblend_capi wheel -- mirrors spatialblend/api/python/capi/_lib.py's
# _candidate_basenames() and spatialblend/api/c/BUILD's c_shared shared_lib_name.
genrule(
    name = "capi_lib_data",
    srcs = ["//spatialblend/api/c:c_shared"],
    # genrule `outs` cannot itself be a select(), so we emit the shared
    # library under all three platform-conventional basenames (see
    # capi/_lib.py's _candidate_basenames()) -- only the one matching the
    # actual runtime platform is ever opened/loaded; the other two copies
    # are harmless unused bytes bundled into the wheel.
    outs = [
        "capi/_data/libspatialblend.so",
        "capi/_data/libspatialblend.dylib",
        "capi/_data/spatialblend.dll",
    ],
    cmd = "cp $< $(location capi/_data/libspatialblend.so) && " +
          "cp $< $(location capi/_data/libspatialblend.dylib) && " +
          "cp $< $(location capi/_data/spatialblend.dll)",
)

pybind_extension(
    name = "_spatialblend_py",
    srcs = ["_spatialblend_py.cc"],
    linkstatic = True,
    visibility = ["//visibility:private"],
    deps = [
        "//spatialblend/base/common:state_type",
        "//spatialblend/base/common:types",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/geometry:polygon3d",
        "//spatialblend/geometry:polygon_coordinate_mapper",
        "//spatialblend/model/actor",
        "//spatialblend/model/actor:actor_motion_tracker",
        "//spatialblend/model/element",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/model/element:navigable_spatial_definition",
        "//spatialblend/model/element:polytope_spatial_definition",
        "//spatialblend/model/element:spatial_definition",
        "//spatialblend/model/element:surface_spatial_definition",
        "//spatialblend/model/realization:remote_update",
        "//spatialblend/model/relation",
        "//spatialblend/model/relation:grounded_relation",
        "//spatialblend/model/relation:relation_kernel",
        "//spatialblend/model/relation:support",
        "//spatialblend/solver:realization_target",
        "//spatialblend/solver:relation_solver",
        "//spatialblend/space",
        "//spatialblend/space:binding",
        "//spatialblend/space:local_context",
        "//spatialblend/space:relation_state",
        "//spatialblend/space:space_hub",
        "//spatialblend/space/builder:actor_handle",
        "//spatialblend/space/builder:element_handle",
        "//spatialblend/space/builder:space_builder",
        "//spatialblend/space/builder:trajectory",
        "@eigen",
    ],
)

# The in-process Simulator moves its avatar with the scenario test driver, which is
# not a core component, so its bindings live in a separate TEST-ONLY extension.
# The shipped _spatialblend_py (and the wheel) carry no Simulator; the package
# __init__ loads this module when it is present (tests, examples).
pybind_extension(
    name = "_spatialblend_sim_py",
    testonly = True,
    srcs = ["_spatialblend_sim_py.cc"],
    linkstatic = True,
    visibility = ["//visibility:private"],
    deps = [
        "//spatialblend/base/common:types",
        "//spatialblend/geometry:polygon2d",
        "//spatialblend/model/actor",
        "//spatialblend/model/element",
        "//spatialblend/model/element:element_tree",
        "//spatialblend/sim:simulator",
        "//spatialblend/solver:realization_target",
        "//spatialblend/space",
        "//spatialblend/space:space_hub",
        "//spatialblend/space/builder:actor_handle",
        "//spatialblend/space/builder:element_handle",
        "//spatialblend/space/builder:space_builder",
        "//spatialblend/space/builder:trajectory",
        "@eigen",
    ],
)

py_library(
    name = "spatialblend_sim_py",
    testonly = True,
    data = [":_spatialblend_sim_py"],
    deps = [":spatialblend_py"],
)

py_library(
    name = "spatialblend_py",
    srcs = ["__init__.py"],
    data = [":_spatialblend_py"],
    deps = [requirement("numpy")],
)

py_library(
    name = "visualization",
    srcs = [
        "visualization/__init__.py",
        "visualization/animate.py",
        "visualization/mp4_export.py",
        "visualization/plot.py",
    ],
    deps = [
        ":spatialblend_py",
        requirement("matplotlib"),
        requirement("numpy"),
    ],
)

py_package(
    name = "spatialblend_pkg",
    visibility = ["//visibility:private"],
    deps = [":spatialblend_py"],
)

py_wheel(
    name = "spatialblend_wheel",
    abi = select({
        "@platforms//os:linux": "cp310",
        "@platforms//os:macos": "cp310",
        "//conditions:default": "none",
    }),
    author = "Human-Centered Computer Systems Lab, Seoul National University",
    distribution = "spatialblend",
    homepage = "https://github.com/snuhcs/spatialblend-core",
    platform = select({
        "@platforms//os:windows": "win_amd64",
        "@platforms//os:macos": "macosx_11_0_arm64",
        "@platforms//os:linux": "manylinux_2_17_x86_64",
        "//conditions:default": "any",
    }),
    python_tag = "cp310",
    # The extension converts Eigen default arguments to numpy arrays while it initialises, so without numpy
    # `import spatialblend.api.python` fails with "ImportError: initialization failed".
    requires = ["numpy>=1.21"],
    version = "1.0.0",
    deps = [":spatialblend_pkg"],
)

py_test(
    name = "test_space",
    srcs = ["tests/test_space.py"],
    deps = [
        ":spatialblend_py",
        requirement("numpy"),
    ],
)

py_test(
    name = "test_simulator",
    srcs = ["tests/test_simulator.py"],
    deps = [
        ":spatialblend_py",
        ":spatialblend_sim_py",
        requirement("numpy"),
    ],
)

py_test(
    name = "test_visualization",
    srcs = ["tests/test_visualization.py"],
    deps = [
        ":spatialblend_py",
        ":spatialblend_sim_py",
        ":visualization",
        requirement("matplotlib"),
        requirement("numpy"),
    ],
)

py_library(
    name = "examples",
    testonly = True,
    srcs = [
        "examples/__init__.py",
        "examples/preset_visualization.py",
    ],
    data = ["//spatialblend/sim/data:presets"],
    deps = [
        ":interaction",
        ":spatialblend_py",
        ":spatialblend_sim_py",
        ":visualization",
        requirement("matplotlib"),
        requirement("numpy"),
    ],
)

py_library(
    name = "hooks",
    testonly = True,
    srcs = [
        "hooks/__init__.py",
        "hooks/arrival_lag_hook.py",
    ],
    deps = [":spatialblend_py"],
)

py_library(
    name = "interaction",
    srcs = [
        "interaction/__init__.py",
        "interaction/script.py",
    ],
    deps = [
        ":spatialblend_py",
        requirement("numpy"),
    ],
)

py_test(
    name = "test_interaction",
    srcs = ["tests/test_interaction.py"],
    deps = [
        ":interaction",
        ":spatialblend_py",
        ":spatialblend_sim_py",
        requirement("numpy"),
    ],
)

py_test(
    name = "test_preset_visualization",
    srcs = ["examples/preset_visualization.py"],
    args = ["$$TEST_TMPDIR"],
    data = ["//spatialblend/sim/data:presets"],
    main = "examples/preset_visualization.py",
    deps = [
        ":interaction",
        ":spatialblend_py",
        ":spatialblend_sim_py",
        ":visualization",
        requirement("matplotlib"),
        requirement("numpy"),
    ],
)

py_binary(
    name = "run_preset_visualization",
    testonly = True,
    srcs = ["examples/preset_visualization.py"],
    data = ["//spatialblend/sim/data:presets"],
    main = "examples/preset_visualization.py",
    deps = [
        ":interaction",
        ":spatialblend_py",
        ":spatialblend_sim_py",
        ":visualization",
        requirement("matplotlib"),
        requirement("numpy"),
    ],
)

# ---------------------------------------------------------------------------
# spatialblend_capi: pure-Python ctypes bindings for the C API.
#
# Unlike spatialblend_py above (a pybind11 C-extension, ABI-tied to one
# CPython minor version -- currently cp310), this package is plain Python +
# ctypes with no C-extension of its own, so it is importable from ANY CPython
# 3.6+ interpreter against the same libspatialblend shared library. This is
# what lets a consumer pinned to an older Python (e.g. 3.8) use the
# core without needing a matching pybind build.
# ---------------------------------------------------------------------------

py_library(
    name = "spatialblend_capi",
    srcs = [
        "__init__.py",
        "capi/__init__.py",
        "capi/_api.py",
        "capi/_lib.py",
        "capi/_mono_number.py",
        "capi/_obs_types.py",
        "capi/_types.py",
        "capi/api.py",
        "capi/observability.py",
        "capi/rich.py",
    ],
    data = [":capi_lib_data"],
)

py_package(
    name = "spatialblend_capi_pkg",
    visibility = ["//visibility:private"],
    deps = [":spatialblend_capi"],
)

py_wheel(
    name = "spatialblend_capi_wheel",
    abi = "none",
    author = "Human-Centered Computer Systems Lab, Seoul National University",
    distribution = "spatialblend_capi",
    homepage = "https://github.com/snuhcs/spatialblend-core",
    platform = select({
        "@platforms//os:windows": "win_amd64",
        "@platforms//os:macos": "macosx_11_0_arm64",
        "@platforms//os:linux": "manylinux_2_17_x86_64",
        "//conditions:default": "any",
    }),
    python_tag = "py3",
    version = "1.0.0",
    deps = [":spatialblend_capi_pkg"],
)

py_test(
    name = "test_capi",
    srcs = ["tests/test_capi.py"],
    deps = [
        ":spatialblend_capi",
        ":spatialblend_py",
        requirement("numpy"),
    ],
)

# ABI 4.7: embodiment registration, bone ports, the normalized kinematic chain, through the capi.
py_test(
    name = "test_capi_embodiment",
    srcs = ["tests/test_capi_embodiment.py"],
    deps = [":spatialblend_capi"],
)

# Observability, injection and trace through the capi.
py_test(
    name = "test_capi_observability",
    srcs = ["tests/test_capi_observability.py"],
    deps = [":spatialblend_capi"],
)

# Rich-realization wrappers and the Unity-compatible JSON
# serializer, against a vendored Unity native capture and a vendored copy of
# MotionGen's parser.
py_test(
    name = "test_capi_rich",
    srcs = [
        "tests/test_capi_rich.py",
        "tests/vendor/__init__.py",
        "tests/vendor/motiongen_rich_realization.py",
    ],
    data = glob([
        "tests/data/rich_realization_unity/*.json",
        "tests/data/rich_realization_unity_support/*.json",
        "tests/data/k1_rich_golden_driver/*.json",
        "tests/data/mono_double_r_sample.tsv",
    ]),
    deps = [":spatialblend_capi"],
)
