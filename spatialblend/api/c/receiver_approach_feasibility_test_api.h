// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_RECEIVER_APPROACH_FEASIBILITY_TEST_API_H_
#define SPATIALBLEND_C_RECEIVER_APPROACH_FEASIBILITY_TEST_API_H_

// Hermetic Bazel-test support only.  This header is deliberately not included
// by spatialblend.h or the shared-library export target.  It lets the lifetime tests advance
// the *receiver-owned* clock without adding a production C test hook.

#include <stdint.h>

#include "spatialblend/api/c/spatialblend.h"

#ifdef __cplusplus
extern "C" {
#endif

sb_session_t sb_test_create_session_with_receiver_input_clock(int64_t now_ns);
int32_t sb_test_set_receiver_input_clock(sb_session_t session, int64_t now_ns);

#ifdef __cplusplus
}
#endif

#endif  // SPATIALBLEND_C_RECEIVER_APPROACH_FEASIBILITY_TEST_API_H_
