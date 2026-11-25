#!/bin/bash
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

#
# One-command installer for the spatialblend wheels.
#
#   - spatialblend       : pybind11 C extension, ABI-tied to one CPython
#                          minor version (currently cp310).
#   - spatialblend_capi  : pure Python + ctypes bindings to the same
#                          libspatialblend shared library, python_tag py3 /
#                          abi none -- importable from any CPython 3.6+
#                          interpreter (this is what lets a consumer pinned
#                          to an older Python use the core without a
#                          matching pybind build).
#
# Usage:
#   tools/install_wheels.sh                      # build locally, install both
#   tools/install_wheels.sh --pybind             # build/install only spatialblend
#   tools/install_wheels.sh --capi               # build/install only spatialblend_capi
#   tools/install_wheels.sh --tag <tag>          # download both from a GitHub release
#   tools/install_wheels.sh --tag <tag> --capi
#   tools/install_wheels.sh --python /path/to/python3 ...
#
# --tag needs a release that carries the wheels (spatialblend-1.0.0-cp310-*.whl,
# spatialblend_capi-1.0.0-py3-*.whl). v1.0.0 ships only the native libraries
# and headers, no wheels: build them locally (no --tag) instead.
#
# This script never creates or publishes a GitHub release -- use
# tools/publish_release.py for that, separately, when you actually intend
# to cut one.
set -euo pipefail

REPO="snuhcs/spatialblend-core"
TAG=""
WANT_PYBIND=true
WANT_CAPI=true
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    grep '^#' "${BASH_SOURCE[0]}" | sed -e '1d' -e 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            TAG="$2"; shift 2 ;;
        --tag=*)
            TAG="${1#*=}"; shift ;;
        --repo)
            REPO="$2"; shift 2 ;;
        --repo=*)
            REPO="${1#*=}"; shift ;;
        --python)
            PYTHON_BIN="$2"; shift 2 ;;
        --python=*)
            PYTHON_BIN="${1#*=}"; shift ;;
        --pybind)
            WANT_PYBIND=true; WANT_CAPI=false; shift ;;
        --capi)
            WANT_PYBIND=false; WANT_CAPI=true; shift ;;
        --both)
            WANT_PYBIND=true; WANT_CAPI=true; shift ;;
        -h|--help)
            usage 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage 1 ;;
    esac
done

echo "Installing into: $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"

# --- Platform-specific wheel filename fragments, mirroring
# spatialblend/api/python/BUILD's spatialblend_wheel / spatialblend_capi_wheel
# platform selects. ---
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
if [[ "${OS_NAME}" == "Linux" ]]; then
    PLATFORM_TAG="manylinux_2_17_x86_64"
elif [[ "${OS_NAME}" == "Darwin" ]]; then
    PLATFORM_TAG="macosx_11_0_arm64"
elif [[ "${OS_NAME}" == MINGW* || "${OS_NAME}" == CYGWIN* || "${OS_NAME}" == MSYS* ]]; then
    PLATFORM_TAG="win_amd64"
else
    echo "Unsupported OS: ${OS_NAME} (${ARCH_NAME})" >&2
    exit 1
fi

PYBIND_WHEEL="spatialblend-1.0.0-cp310-cp310-${PLATFORM_TAG}.whl"
CAPI_WHEEL="spatialblend_capi-1.0.0-py3-none-${PLATFORM_TAG}.whl"

install_wheel() {
    local wheel_path="$1"
    echo "Installing ${wheel_path}"
    "${PYTHON_BIN}" -m pip install --force-reinstall "${wheel_path}"
}

if [[ -n "${TAG}" ]]; then
    # --- Download from an existing GitHub release. Never creates one. ---
    if ! command -v gh &> /dev/null; then
        echo "Error: the GitHub CLI ('gh') is required to download a release." >&2
        echo "Install it from https://cli.github.com/ and run 'gh auth login'." >&2
        exit 1
    fi

    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "${TMP_DIR}"' EXIT

    if [[ "${WANT_PYBIND}" == true ]]; then
        echo "Downloading ${PYBIND_WHEEL} from ${REPO}@${TAG}..."
        gh release download "${TAG}" --repo "${REPO}" \
            --pattern "${PYBIND_WHEEL}" --dir "${TMP_DIR}" --clobber
        install_wheel "${TMP_DIR}/${PYBIND_WHEEL}"
    fi
    if [[ "${WANT_CAPI}" == true ]]; then
        echo "Downloading ${CAPI_WHEEL} from ${REPO}@${TAG}..."
        gh release download "${TAG}" --repo "${REPO}" \
            --pattern "${CAPI_WHEEL}" --dir "${TMP_DIR}" --clobber
        install_wheel "${TMP_DIR}/${CAPI_WHEEL}"
    fi
else
    # --- Build locally with a targeted `bazel build` (bypasses
    # tools/build_python_wheel.py's separate --output_base, which on some
    # machines picks up a different local C++ toolchain resolution; a plain
    # `bazel build` against the default output base is what this repo's own
    # interactive workflow and CI both rely on). ---
    cd "${WORKSPACE_DIR}"
    BIN_DIR="bazel-bin/spatialblend/api/python"

    if [[ "${WANT_PYBIND}" == true ]]; then
        echo "Building //spatialblend/api/python:spatialblend_wheel ..."
        bazel build //spatialblend/api/python:spatialblend_wheel
        install_wheel "$(ls -t "${BIN_DIR}"/spatialblend-*.whl | head -1)"
    fi
    if [[ "${WANT_CAPI}" == true ]]; then
        echo "Building //spatialblend/api/python:spatialblend_capi_wheel ..."
        bazel build //spatialblend/api/python:spatialblend_capi_wheel
        install_wheel "$(ls -t "${BIN_DIR}"/spatialblend_capi-*.whl | head -1)"
    fi
fi

echo "Done."
