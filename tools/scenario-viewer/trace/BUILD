# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

exports_files([
    "validate.py",
    "schema_test.py",
    "scope_context.py",
])

# spatialblend.scope-context v0 (the conditioning context a motion generator reads from an affordance case): stdlib only.
py_test(
    name = "scope_context_test",
    srcs = [
        "scope_context.py",
        "scope_context_test.py",
    ],
    main = "scope_context_test.py",
    python_version = "PY3",
)

# The schema, the channel registry and the fixtures (validate.py checks a trace against them).
filegroup(
    name = "schema",
    srcs = glob([
        "schema/*.json",
        "fixtures/*.json",
    ]),
)
