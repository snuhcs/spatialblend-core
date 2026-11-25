# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # MIT
)

# ADB test runner script for running tests on Android devices
exports_files(
    ["adb_test_runner.sh"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "adb_test_runner",
    srcs = ["adb_test_runner.sh"],
    visibility = ["//visibility:public"],
)

# Python build scripts
py_binary(
    name = "build_c_api",
    srcs = ["build_c_api.py"],
    python_version = "PY3",
)

py_binary(
    name = "build_python_wheel",
    srcs = ["build_python_wheel.py"],
    python_version = "PY3",
)

py_binary(
    name = "publish_release",
    srcs = ["publish_release.py"],
    python_version = "PY3",
)

py_binary(
    name = "run_android_test",
    srcs = ["run_android_test.py"],
    python_version = "PY3",
)

# Alias: run any Bazel binary (test, benchmark, cc_binary) on Android via ADB
alias(
    name = "run_on_android",
    actual = ":run_android_test",
    visibility = ["//visibility:public"],
)

# ADB utility scripts
exports_files(
    [
        "adb/common.sh",
        "adb/setup.sh",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "adb_utils",
    srcs = [
        "adb/common.sh",
        "adb/setup.sh",
    ],
    visibility = ["//visibility:public"],
)
