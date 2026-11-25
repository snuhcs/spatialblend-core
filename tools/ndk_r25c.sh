#!/usr/bin/env bash
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

# The core library's Android NDK: r25c (25.2.9519653, clang 14.0.7). The device-verified library is built with it; r27d
# (clang 18) builds showed device regressions.
#
#   tools/ndk_r25c.sh install          download (or reuse) the Linux NDK, verify the archive, print its path
#   tools/ndk_r25c.sh check-lib SO...  fail unless every library was compiled by r25c's clang 14.0.7 (.comment section)
#
# install: the archive comes straight from Google and is checked against the SHA-1 Google publishes for it in its SDK
# repository manifest (https://dl.google.com/android/repository/repository2-3.xml, package ndk;25.2.9519653) and against
# the SHA-256 of that same verified archive, before it is unpacked. The unpacked NDK is kept in $NDK_CACHE_DIR
# (default: $RUNNER_TOOL_CACHE, else ~/.cache) and reused only if its source.properties names 25.2.9519653. In GitHub
# Actions the path also goes to $GITHUB_OUTPUT (ndk-path) and ANDROID_NDK_HOME to $GITHUB_ENV. NDK_ARCHIVE=<zip> uses an
# archive already downloaded instead of fetching it (the same checks apply).
# check-lib guards the build itself: a cached Bazel androidndk repository from another NDK cannot slip through.
set -euo pipefail

REVISION=25.2.9519653
CLANG="clang version 14.0.7"
URL=https://dl.google.com/android/repository/android-ndk-r25c-linux.zip
SIZE=531118193
SHA1=53af80a1cce9144025b81c78c8cd556bff42bd0e
SHA256=769ee342ea75f80619d985c2da990c48b3d8eaf45f48783a2d48870d04b46108   # of that same SHA-1-verified archive

die() { echo "ndk_r25c: $*" >&2; exit 1; }

readelf_tool() {
    local ndk=${ANDROID_NDK_HOME:-} tool=""
    if [[ -n $ndk && -d $ndk/toolchains/llvm/prebuilt ]]; then
        tool=$(find "$ndk/toolchains/llvm/prebuilt" -path '*/bin/llvm-readelf' \( -type f -o -type l \) | head -n 1)
    fi
    [[ -z $tool ]] && tool=$(command -v llvm-readelf || command -v readelf || true)
    [[ -n $tool ]] || die "no llvm-readelf or readelf (set ANDROID_NDK_HOME to an NDK)"
    echo "$tool"
}

check_lib() {
    (( $# )) || die "check-lib needs at least one library"
    local tool lib versions
    tool=$(readelf_tool)
    for lib in "$@"; do
        [[ -f $lib ]] || die "$lib not found"
        versions=$("$tool" -p .comment "$lib" | grep -o 'clang version [0-9.]*' | sort -u || true)
        if [[ $versions != "$CLANG" ]]; then
            die "$lib was not built by NDK r25c ($CLANG); its .comment names: ${versions:-no clang version}"
        fi
        echo "ok: $lib ($CLANG, NDK r25c)"
    done
}

install() {
    local cache=${NDK_CACHE_DIR:-${RUNNER_TOOL_CACHE:-$HOME/.cache}} dest
    dest=$cache/android-ndk-r25c
    if ! grep -qx "Pkg.Revision = $REVISION" "$dest/source.properties" 2>/dev/null; then
        [[ $(uname -s) == Linux ]] || die "install fetches the Linux NDK; on $(uname -s) install r25c with sdkmanager \"ndk;$REVISION\""
        command -v unzip >/dev/null || die "unzip is needed"
        local work
        work=$(mktemp -d)
        trap 'rm -rf "$work"' RETURN
        if [[ -n ${NDK_ARCHIVE:-} ]]; then cp "$NDK_ARCHIVE" "$work/ndk.zip"; else curl -fsSL --retry 3 -o "$work/ndk.zip" "$URL"; fi
        [[ $(stat -c %s "$work/ndk.zip") == "$SIZE" ]] || die "the archive size differs from Google's manifest"
        echo "$SHA1  $work/ndk.zip" | sha1sum -c --quiet - || die "SHA-1 mismatch: not Google's android-ndk-r25c-linux.zip"
        echo "$SHA256  $work/ndk.zip" | sha256sum -c --quiet - || die "SHA-256 mismatch: not Google's android-ndk-r25c-linux.zip"
        unzip -q "$work/ndk.zip" -d "$work/x"
        grep -qx "Pkg.Revision = $REVISION" "$work/x/android-ndk-r25c/source.properties" || die "unexpected archive content"
        mkdir -p "$cache"
        rm -rf "$dest"
        mv "$work/x/android-ndk-r25c" "$dest"
    fi
    if [[ -n ${GITHUB_OUTPUT:-} ]]; then echo "ndk-path=$dest" >> "$GITHUB_OUTPUT"; fi
    if [[ -n ${GITHUB_ENV:-} ]]; then echo "ANDROID_NDK_HOME=$dest" >> "$GITHUB_ENV"; fi
    echo "$dest"
}

case ${1:-} in
    install) install ;;
    check-lib) shift; check_lib "$@" ;;
    *) die "usage: $0 install | check-lib LIB.so..." ;;
esac
