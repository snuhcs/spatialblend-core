# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "debug",
    values = {
        "compilation_mode": "dbg",
    },
)

config_setting(
    name = "release",
    values = {
        "compilation_mode": "opt",
    },
)

config_setting(
    name = "arm64-v8a",
    values = {
        "cpu": "arm64",
    },
)

config_setting(
    name = "armeabi-v7a",
    values = {
        "cpu": "arm",
    },
)

config_setting(
    name = "x86",
    values = {
        "cpu": "x86",
    },
)

config_setting(
    name = "x86_64",
    values = {
        "cpu": "x86_64",
    },
)

config_setting(
    name = "android_x86",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:x86_32",
    ],
)

config_setting(
    name = "android_x86_64",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "windows",
    values = {
        "os": "windows",
    },
)

platform(
    name = "visionos_arm64",
    constraint_values = [
        "@platforms//os:visionos",
        "@platforms//cpu:arm64",
    ],
)

platform(
    name = "android_arm64",
    constraint_values = [
        "@platforms//cpu:arm64",
        "@platforms//os:android",
    ],
)

platform(
    name = "android_arm",
    constraint_values = [
        "@platforms//cpu:arm",
        "@platforms//os:android",
    ],
)

platform(
    name = "x64_windows-clang-cl",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
)
