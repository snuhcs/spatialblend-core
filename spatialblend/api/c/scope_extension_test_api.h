// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SCOPE_EXTENSION_TEST_API_H_
#define SPATIALBLEND_C_SCOPE_EXTENSION_TEST_API_H_

// Hermetic Bazel-test support only.  This header is deliberately not included
// by spatialblend.h or the shared-library export target.  It lets a C test wait for
// the background scope-extension build (element/scope_extension.h) instead of
// sleeping.

#include <stdint.h>

#include "spatialblend/api/c/spatialblend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Waits (up to 5 s) until the space's scope-extension worker is idle
// (Space::WaitForScopeExtension). Returns 1 when the build completed (or none is
// pending), 0 on timeout, -1 for an invalid handle.
int32_t sb_test_wait_for_scope_extension(sb_space_t space);

#ifdef __cplusplus
}
#endif

#endif  // SPATIALBLEND_C_SCOPE_EXTENSION_TEST_API_H_
