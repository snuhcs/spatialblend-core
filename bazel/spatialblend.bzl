# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("//bazel:adb_test.bzl", "adb_binary", "adb_test")

def spatialblend_test_cxxopts():
    return ["-DSPATIALBLEND_TEST"]

def spatialblend_linkopts_unstripped():
    """Define linker flags to reduce size of Band binary.

        These are useful when trying to investigate the relative size of the
        symbols in Band.

    Returns:
        a select object with proper linkopts
    """

    # In case you wonder why there's no --icf is because the gains were
    # negligible, and created potential compatibility problems.
    base_android = [
        "-lc++_static",  # Static C++ runtime
        "-lc++abi",  # C++ ABI support library
        "-llog",  # Android logging
        "-lm",  # libm
        "-lc",  # libc
        "-ldl",  # Required for `dl_iterate_phdr`
        "-Wl,--no-export-dynamic",  # Only inc syms referenced by dynamic obj.
        "-Wl,--gc-sections",  # Eliminate unused code and data.
        "-Wl,--as-needed",  # Don't link unused libs.
    ]
    return select({
        Label("//bazel:android_x86"): base_android + ["-latomic"],  # x86 needs libatomic
        Label("//bazel:android_x86_64"): base_android + ["-latomic"],  # x86_64 needs libatomic
        "@platforms//os:android": base_android,  # arm64 has built-in atomics
        "//conditions:default": [],
    })

def spatialblend_symbol_opts():
    """Defines linker flags whether to include symbols or not."""
    return select({
        "//conditions:default": [],
    })

def spatialblend_linkopts():
    """Defines linker flags for linking SpatialBlend binary."""
    return spatialblend_linkopts_unstripped() + spatialblend_symbol_opts()

def spatialblend_shared_lib_name():
    libname = select({
        Label("//bazel:debug"): "libspatialblend_d",
        Label("//bazel:release"): "libspatialblend",
        "//conditions:default": "libspatialblend",
    })
    extension = select({
        "@platforms//os:linux": ".so",
        "@platforms//os:macos": ".dylib",
        "@platforms//os:windows": ".dll",
        "@platforms//os:visionos": ".dylib",
        "//conditions:default": ".so",
    })
    return libname + extension

def _default_adb_tags(tags):
    return tags + ["manual", "android", "adb"]

def spatialblend_cc_binary(
        name,
        linkopts = [],
        create_adb_target = True,
        adb_args = [],
        adb_env = {},
        adb_tags = [],
        tags = [],
        **kwargs):
    native.cc_binary(
        name = name,
        linkopts = linkopts + spatialblend_linkopts(),
        tags = tags,
        **kwargs
    )

    if create_adb_target:
        adb_binary(
            name = name + "_adb",
            binary = ":" + name,
            binary_args = adb_args,
            env = adb_env,
            tags = _default_adb_tags(tags + adb_tags),
        )

def spatialblend_cc_test(
        name,
        linkopts = [],
        create_adb_target = True,
        adb_args = [],
        adb_env = {},
        adb_tags = [],
        tags = [],
        **kwargs):
    native.cc_test(
        name = name,
        copts = spatialblend_test_cxxopts(),
        linkopts = linkopts + spatialblend_linkopts(),
        tags = tags,
        **kwargs
    )

    if create_adb_target:
        adb_test(
            name = name + "_adb",
            test = ":" + name,
            test_args = adb_args,
            env = adb_env,
            tags = _default_adb_tags(tags + adb_tags),
        )

def spatialblend_version():
    return "0.0.1"
