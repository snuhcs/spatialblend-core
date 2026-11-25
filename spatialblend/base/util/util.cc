// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/base/util/util.h"

#include <cmath>
#include <cstring>
#include <iostream>

#include "spatialblend/base/common/common.h"
#include "spatialblend/base/util/logger.h"

namespace spatialblend {

OnEndEventCallback GetInteractionOnEndEventCallback() {
  return OnEndEventCallback([](absl::Status status) {
    if (!status.ok()) {
      SPATIALBLEND_LOG(LogSeverity::kWarning,
                    "Interaction event dropped: %s", status.ToString().c_str());
    }
    return absl::OkStatus();
  });
}

}  // namespace spatialblend
