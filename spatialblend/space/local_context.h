// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_LOCAL_CONTEXT_H_
#define SPATIALBLEND_SPACE_LOCAL_CONTEXT_H_

#include <functional>
#include <memory>
#include <vector>

#include "spatialblend/space/space.h"
#include "spatialblend/space/space_hub.h"

namespace spatialblend {

#ifdef SPATIALBLEND_V2_C_TESTING
// Defined only by the hermetic approach-feasibility Bazel test owner.  Keeping the clock
// factory private to this C++ context avoids adding a controllable clock to
// the shipping C ABI.
class CDeterministicSessionFactory;
#endif

class LocalContext {
 public:
  static std::unique_ptr<LocalContext> Create();

  Space* AddSpace();
  SpaceHub* GetHub();

 private:
  LocalContext();

#ifdef SPATIALBLEND_V2_C_TESTING
  friend class CDeterministicSessionFactory;
#endif

  std::unique_ptr<SpaceHub> hub_;
  std::vector<std::unique_ptr<Space>> spaces_;
  // Empty in every production context; Space then selects ReceiverInputNowNs.
  // The test-only friend may install a deterministic receiver-input clock.
  std::function<int64_t()> receiver_input_clock_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_LOCAL_CONTEXT_H_
