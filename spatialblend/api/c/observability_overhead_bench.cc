// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Per-tick overhead of observability capture.
//
// Two in-process spaces: a source actor with spacing + facing bindings over
// 12 furniture referents, propagating to a linked receiver every update.
// Reports ns per sb_update_actor_state (source evaluation + receiver solve).
// Builds against any version of the C API; the capture modes need
// spatialblend_observability.h.
//
//   bazel run -c opt //spatialblend/api/c:observability_overhead_bench -- 20000

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "spatialblend/api/c/spatialblend.h"

namespace {

sb_element_t g_received = 0;
void Added(sb_space_t, sb_element_t local, int64_t, const char*, const void*,
           int64_t, void*) {
  g_received = local;
}

double RunOnce(int updates, bool capture) {
  const sb_session_t session = sb_create_session();
  const sb_space_t spaces[2] = {sb_add_space(session), sb_add_space(session)};
  sb_element_t actor_elem[2];
  for (int s = 0; s < 2; ++s) {
    const double poly[] = {0, 0, 20, 0, 20, 20, 0, 20};
    const sb_element_t room = sb_add_element(spaces[s], 0);
    sb_set_element_shared_id(spaces[s], room, 100);
    sb_set_element_polygon(spaces[s], room, SB_GEOMETRY_NAVIGABLE, poly, 4);
    for (int i = 0; i < 12; ++i) {
      const sb_element_t f = sb_add_element(spaces[s], room);
      sb_set_element_shared_id(spaces[s], f, 200 + i);
      const double x = 2 + (i % 4) * 4.5, y = 3 + (i / 4) * 6.0;
      sb_set_element_box(spaces[s], f, SB_GEOMETRY_OBSTACLE, x, y, 0, x + 1,
                         y + 1, 1);
    }
    actor_elem[s] = sb_add_element(spaces[s], room);
    sb_set_element_shared_id(spaces[s], actor_elem[s], s == 0 ? 102 : 202);
  }
  const sb_actor_t src = sb_create_actor(spaces[0]);
  sb_add_anchor(spaces[0], src, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(spaces[0], src, actor_elem[0]);
  sb_bind(spaces[0], src, "root", SB_KERNEL_SPACING);
  sb_bind(spaces[0], src, "root", SB_KERNEL_FACING);
  const sb_actor_t dst = sb_create_actor(spaces[1]);
  sb_add_anchor(spaces[1], dst, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(spaces[1], dst, actor_elem[1]);
  sb_register_on_element_added(spaces[1], Added, nullptr);
  sb_share_element(spaces[0], actor_elem[0], nullptr, 0);
  sb_resolve_shared_element(spaces[1], g_received, actor_elem[1]);
  sb_set_min_move(spaces[0], 0.0);
  sb_set_min_yaw(spaces[0], 0.0);
#ifdef SB_OBSERVABILITY_VERSION
  sb_set_observability_capture(spaces[0], capture ? 1 : 0);
  sb_set_observability_capture(spaces[1], capture ? 1 : 0);
#else
  (void)capture;
#endif
  for (int i = 0; i < 200; ++i)  // warm-up
    sb_update_actor_state(spaces[0], src, 1 + 0.01 * i, 10, 0);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < updates; ++i) {
    const double a = 0.001 * i;
    sb_update_actor_state(spaces[0], src, 10 + 8 * std::cos(a),
                          10 + 8 * std::sin(a), a);
  }
  const auto t1 = std::chrono::steady_clock::now();
  sb_destroy_session(session);
  return std::chrono::duration<double, std::nano>(t1 - t0).count() / updates;
}

}  // namespace

int main(int argc, char** argv) {
  const int updates = argc > 1 ? std::atoi(argv[1]) : 20000;
  const int reps = 5;
#ifdef SB_OBSERVABILITY_VERSION
  const bool modes[] = {false, true};
#else
  const bool modes[] = {false};
#endif
  for (bool capture : modes) {
    double best = 1e300, sum = 0;
    for (int r = 0; r < reps; ++r) {
      const double ns = RunOnce(updates, capture);
      best = ns < best ? ns : best;
      sum += ns;
    }
    std::printf("capture=%s updates=%d reps=%d best=%.0f ns/update "
                "mean=%.0f ns/update\n",
                capture ? "on " : "off", updates, reps, best, sum / reps);
  }
  return 0;
}
