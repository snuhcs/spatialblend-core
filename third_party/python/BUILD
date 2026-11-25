load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(default_visibility = ["//visibility:public"])

# Pinned Python dependencies for the pip hub (@spatialblend_deps, see MODULE.bazel).
# Regenerate the lock file with: bazel run //third_party/python:requirements.update
exports_files([
    "requirements.lock",
    "requirements.txt",
])

compile_pip_requirements(
    name = "requirements",
    src = "requirements.txt",
    requirements_txt = "requirements.lock",
    tags = ["manual"],
)
