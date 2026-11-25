// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// ABI 4.10: per-part surface charts through the public C API (sb_set_sub_region_chart, sb_part_chart_encode / _decode). One mapping
// for the relation path (hand, pelvis) and for strokes: a point of A's board face, encoded on A and decoded on B, lands at the same
// (u,v) of B's face whatever the sizes, aspect ratios, walls and tilt. The relation path itself is tested in
// space/surface_retarget_test.cc (the part frame it uses is the one these entry points read).
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "spatialblend/api/c/spatialblend.h"

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Board {
  double w, h, low, x, y, yaw, tilt;
  int32_t chart;
};

// Row-major 4x4 of yaw about z then tilt about the element's y, translated to (x, y, 0).
void ElementMatrix(const Board& b, double* m) {
  const double cy = std::cos(b.yaw), sy = std::sin(b.yaw), ct = std::cos(b.tilt), st = std::sin(b.tilt);
  // R = Rz(yaw) * Ry(tilt)
  const double r[3][3] = {{cy * ct, -sy, cy * st}, {sy * ct, cy, sy * st}, {-st, 0.0, ct}};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) m[i * 4 + j] = r[i][j];
  }
  m[3] = b.x;
  m[7] = b.y;
  m[11] = 0.0;
  m[12] = m[13] = m[14] = 0.0;
  m[15] = 1.0;
}

// The part frame in the element frame: part x along the board (element y), part y up (element z), part z the normal (element x).
void FaceMatrix(const Board& b, double* m) {
  std::memset(m, 0, 16 * sizeof(double));
  m[0 * 4 + 2] = 1;
  m[1 * 4 + 0] = 1;
  m[2 * 4 + 1] = 1;
  m[15] = 1;
  m[0 * 4 + 3] = 0.03;
  m[2 * 4 + 3] = b.low + 0.5 * b.h;
}

// World point of part-local (px, py, n).
void PartWorld(const Board& b, double px, double py, double n, double* out) {
  double e[16], f[16];
  ElementMatrix(b, e);
  FaceMatrix(b, f);
  const double local[4] = {px, py, n, 1.0};
  double inel[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) inel[i] += f[i * 4 + j] * local[j];
  for (int i = 0; i < 3; ++i) {
    out[i] = 0;
    for (int j = 0; j < 4; ++j) out[i] += e[i * 4 + j] * inel[j];
  }
}

struct Room {
  sb_session_t session;
  sb_space_t space;
  sb_element_t elem;
  Board board;
  Room(sb_session_t s, const Board& b) : session(s), space(sb_add_space(s)), board(b) {
    elem = sb_add_element(space, 0);
    double m[16];
    ElementMatrix(b, m);
    EXPECT_EQ(sb_set_element_transform(space, elem, m), SB_SCENE_OK);
    EXPECT_EQ(sb_set_element_box(space, elem, SB_GEOMETRY_OBSTACLE, -0.03, -0.5 * b.w, b.low, 0.03, 0.5 * b.w, b.low + b.h),
              SB_SCENE_OK);
    const double pts[8] = {-0.5 * b.w, -0.5 * b.h, 0.5 * b.w, -0.5 * b.h, 0.5 * b.w, 0.5 * b.h, -0.5 * b.w, 0.5 * b.h};
    double f[16];
    FaceMatrix(b, f);
    sb_add_sub_region(space, elem, "board", "Writable", pts, 4, f);
    EXPECT_EQ(sb_set_sub_region_chart(space, elem, "board", b.chart), 0);
  }
};

sb_part_chart_coord_t Blank() {
  sb_part_chart_coord_t c;
  std::memset(&c, 0, sizeof c);
  c.struct_size = sizeof c;
  return c;
}

TEST(PartChartC, AbiMinorAndStructAreExported) {
  EXPECT_EQ(sb_abi_minor_version(), 14u);
  EXPECT_EQ(sb_sizeof(SB_STRUCT_PART_CHART_COORD), sizeof(sb_part_chart_coord_t));
}

TEST(PartChartC, AStrokePointKeepsItsUvOnABoardOfAnotherSizeWallAndTilt) {
  struct Pair { Board a, b; };
  const Pair pairs[] = {
      {{1.2, 0.9, 0.9, 5.0, 9.5, -kPi / 2, 0.0, SB_SURFACE_CHART_FRAME_STRETCH}, {0.8, 0.6, 0.7, 9.5, 4.0, kPi, 0.0, SB_SURFACE_CHART_FRAME_STRETCH}},
      {{1.6, 0.9, 0.9, 5.0, 9.5, -kPi / 2, 0.0, SB_SURFACE_CHART_FRAME_STRETCH}, {1.2, 0.9, 0.7, 3.0, 3.0, kPi / 6, 0.2, SB_SURFACE_CHART_FRAME_STRETCH}},
      {{1.2, 0.9, 0.9, 5.0, 9.5, -kPi / 2, 0.0, SB_SURFACE_CHART_FRAME_CONTAIN}, {1.6, 0.9, 0.7, 9.5, 4.0, kPi, 0.0, SB_SURFACE_CHART_FRAME_CONTAIN}},
  };
  const double uv[][2] = {{0, 0}, {1, 1}, {0.5, 0.5}, {0.2, 0.9}};
  for (const Pair& p : pairs) {
    sb_session_t s = sb_create_session();
    Room a(s, p.a), b(s, p.b);
    for (const auto& q : uv) {
      double world[3];
      PartWorld(p.a, (q[0] - 0.5) * p.a.w, (q[1] - 0.5) * p.a.h, 0.004, world);
      sb_part_chart_coord_t c = Blank();
      ASSERT_EQ(sb_part_chart_encode(a.space, a.elem, "board", world, &c), 0);
      EXPECT_EQ(c.chart, p.a.chart);
      EXPECT_NEAR(c.u, q[0], 1e-12);
      EXPECT_NEAR(c.v, q[1], 1e-12);
      EXPECT_NEAR(c.normal_m, 0.004, 1e-12);
      double out[3];
      ASSERT_EQ(sb_part_chart_decode(b.space, b.elem, "board", &c, out), 0);
      double want[3];
      if (p.b.chart == SB_SURFACE_CHART_FRAME_STRETCH) {
        PartWorld(p.b, (q[0] - 0.5) * p.b.w, (q[1] - 0.5) * p.b.h, 0.004, want);
      } else {
        const double sc = std::fmin(p.b.w / p.a.w, p.b.h / p.a.h);
        PartWorld(p.b, (q[0] - 0.5) * p.a.w * sc, (q[1] - 0.5) * p.a.h * sc, 0.004, want);
      }
      for (int k = 0; k < 3; ++k) EXPECT_NEAR(out[k], want[k], 1e-9) << k;
      // and back: B -> A returns the point (stretch always; contain when A fits B's letterbox, which it does by construction here)
      sb_part_chart_coord_t back = Blank();
      ASSERT_EQ(sb_part_chart_encode(b.space, b.elem, "board", out, &back), 0);
      double ret[3];
      ASSERT_EQ(sb_part_chart_decode(a.space, a.elem, "board", &back, ret), 0);
      if (p.a.chart == SB_SURFACE_CHART_FRAME_STRETCH) {
        for (int k = 0; k < 3; ++k) EXPECT_NEAR(ret[k], world[k], 1e-9) << "round trip " << k;
      }
    }
    sb_destroy_session(s);
  }
}

TEST(PartChartC, TheRadialDefaultReadsThroughTheSameEntryPoints) {
  sb_session_t s = sb_create_session();
  Board a{1.2, 0.9, 0.9, 5, 9.5, -kPi / 2, 0, SB_SURFACE_CHART_RADIAL}, b{0.8, 0.6, 0.7, 9.5, 4.0, kPi, 0, SB_SURFACE_CHART_RADIAL};
  Room ra(s, a), rb(s, b);
  double world[3];
  PartWorld(a, 0.3, -0.2, 0.0, world);
  sb_part_chart_coord_t c = Blank();
  ASSERT_EQ(sb_part_chart_encode(ra.space, ra.elem, "board", world, &c), 0);
  EXPECT_EQ(c.chart, SB_SURFACE_CHART_RADIAL);
  double out[3];
  ASSERT_EQ(sb_part_chart_decode(rb.space, rb.elem, "board", &c, out), 0);
  double want[3];
  PartWorld(b, 0.3 * 0.8 / 1.2, -0.2 * 0.6 / 0.9, 0.0, want);  // 4:3 -> 4:3: the radial chart is the uniform scale
  for (int k = 0; k < 3; ++k) EXPECT_NEAR(out[k], want[k], 1e-9) << k;
  sb_destroy_session(s);
}

TEST(PartChartC, RefusalsAreStatusCodesNeverGarbage) {
  sb_session_t s = sb_create_session();
  Board a{1.2, 0.9, 0.9, 5, 9.5, -kPi / 2, 0, SB_SURFACE_CHART_FRAME_STRETCH}, r{0.8, 0.6, 0.7, 9.5, 4.0, kPi, 0, SB_SURFACE_CHART_RADIAL};
  Room ra(s, a), rr(s, r);
  double world[3];
  PartWorld(a, 0.1, 0.1, 0.0, world);
  sb_part_chart_coord_t c = Blank();
  ASSERT_EQ(sb_part_chart_encode(ra.space, ra.elem, "board", world, &c), 0);
  double out[3] = {7, 7, 7};
  // matched parts must declare the same chart: a stretch coordinate on a radial part is refused, nothing is written
  EXPECT_EQ(sb_part_chart_decode(rr.space, rr.elem, "board", &c, out), -2);
  EXPECT_EQ(out[0], 7);
  // unknown part / element / arguments
  EXPECT_EQ(sb_part_chart_encode(ra.space, ra.elem, "nope", world, &c), -1);
  EXPECT_EQ(sb_part_chart_encode(ra.space, 9999, "board", world, &c), -1);
  EXPECT_EQ(sb_part_chart_encode(ra.space, ra.elem, "board", nullptr, &c), -1);
  sb_part_chart_coord_t bad = Blank();
  bad.struct_size = 3;
  EXPECT_EQ(sb_part_chart_encode(ra.space, ra.elem, "board", world, &bad), -1);
  EXPECT_EQ(sb_part_chart_decode(ra.space, ra.elem, "board", &bad, out), -1);
  // invalid coordinates
  sb_part_chart_coord_t zero = c;
  zero.src_width_m = 0.0;
  EXPECT_EQ(sb_part_chart_decode(ra.space, ra.elem, "board", &zero, out), -3);
  sb_part_chart_coord_t nan = c;
  nan.u = NAN;
  EXPECT_EQ(sb_part_chart_decode(ra.space, ra.elem, "board", &nan, out), -3);
  sb_part_chart_coord_t kind = c;
  kind.chart = 9;
  EXPECT_EQ(sb_part_chart_decode(ra.space, ra.elem, "board", &kind, out), -2);
  // chart declaration
  EXPECT_EQ(sb_set_sub_region_chart(ra.space, ra.elem, "nope", SB_SURFACE_CHART_FRAME_STRETCH), -1);
  EXPECT_EQ(sb_set_sub_region_chart(ra.space, ra.elem, "board", 5), -1);
  const double tri[8] = {0, 0, 1, 0, 0.8, 1, 0.2, 1};
  sb_add_sub_region(ra.space, ra.elem, "trapezoid", "Writable", tri, 4, nullptr);
  EXPECT_EQ(sb_set_sub_region_chart(ra.space, ra.elem, "trapezoid", SB_SURFACE_CHART_FRAME_CONTAIN), -2);
  EXPECT_EQ(sb_set_sub_region_chart(ra.space, ra.elem, "trapezoid", SB_SURFACE_CHART_RADIAL), 0);
  sb_destroy_session(s);
}

}  // namespace
