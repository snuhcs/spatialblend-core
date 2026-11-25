// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/sim/harness/scenario.h"

#include <cmath>

#include <gtest/gtest.h>

#include "spatialblend/space/local_context.h"

namespace spatialblend {
namespace harness {
namespace {

constexpr double kEps = 1e-9;

std::string TestdataPath(const std::string& name) {
  return "spatialblend/sim/harness/testdata/" + name;
}

// -----------------------------------------------------------------------
// Fixture loading (acceptance a): counts / shared_ids / footprint kinds.
// -----------------------------------------------------------------------

TEST(ScenarioLoaderTest, LoadsT1StraightApproach) {
  auto model = LoadScenario(TestdataPath("T1-straight-approach.scenario.json"));
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->id, "T1-straight-approach");
  EXPECT_EQ(model->tier, "T1");
  ASSERT_EQ(model->spaces.size(), 2u);

  const ScenarioSpace* a = model->FindSpace("A");
  const ScenarioSpace* b = model->FindSpace("B");
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(a->role, "source");
  EXPECT_EQ(b->role, "target");
  ASSERT_EQ(a->elements.size(), 1u);
  ASSERT_EQ(b->elements.size(), 1u);
  EXPECT_EQ(a->elements[0].shared_id, 201);
  EXPECT_EQ(b->elements[0].shared_id, 201);
  ASSERT_TRUE(a->elements[0].has_footprint);
  EXPECT_EQ(a->elements[0].footprint.kind, "box");

  ASSERT_EQ(model->actors.size(), 1u);
  EXPECT_EQ(model->actors[0].shared_id, 9001);
  EXPECT_EQ(model->actors[0].anchors.size(), 4u);
  EXPECT_EQ(model->actors[0].bindings.size(), 2u);

  ASSERT_TRUE(model->has_identity);
  EXPECT_EQ(model->identity.share_from, "A");
  EXPECT_FALSE(model->identity.mismatched_ids);
  ASSERT_EQ(model->identity.expect.size(), 2u);

  EXPECT_EQ(model->behavior.kind, "script");
  ASSERT_EQ(model->behavior.script.size(), 1u);
  EXPECT_EQ(model->behavior.script[0].op, "approach");
}

TEST(ScenarioLoaderTest, LoadsT1IdentityRoom) {
  auto model = LoadScenario(TestdataPath("T1-identity-room.scenario.json"));
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_EQ(model->spaces.size(), 2u);
  const ScenarioSpace* a = model->FindSpace("A");
  const ScenarioSpace* b = model->FindSpace("B");
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  // Identity-room: source and target are the same room/furniture layout.
  ASSERT_EQ(a->walkable.size(), b->walkable.size());
  for (size_t i = 0; i < a->walkable.size(); ++i) {
    EXPECT_NEAR(a->walkable[i].x, b->walkable[i].x, kEps);
    EXPECT_NEAR(a->walkable[i].y, b->walkable[i].y, kEps);
  }
  ASSERT_EQ(a->elements.size(), b->elements.size());
  EXPECT_EQ(a->elements[0].shared_id, b->elements[0].shared_id);

  ASSERT_EQ(model->behavior.script.size(), 3u);
  EXPECT_EQ(model->behavior.script[0].op, "approach");
  EXPECT_EQ(model->behavior.script[1].op, "sit");
  EXPECT_EQ(model->behavior.script[2].op, "stand");
  ASSERT_TRUE(model->behavior.script[2].raw.contains("jitter_sigma"));
}

TEST(ScenarioLoaderTest, LoadsCMismatchedIds) {
  auto model = LoadScenario(TestdataPath("C-mismatched-ids.scenario.json"));
  ASSERT_TRUE(model.ok()) << model.status();
  const ScenarioSpace* a = model->FindSpace("A");
  const ScenarioSpace* b = model->FindSpace("B");
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  ASSERT_EQ(a->elements.size(), 1u);
  ASSERT_EQ(b->elements.size(), 1u);
  EXPECT_EQ(a->elements[0].shared_id, 201);
  EXPECT_EQ(b->elements[0].shared_id, 100201);
  ASSERT_TRUE(model->has_identity);
  EXPECT_TRUE(model->identity.mismatched_ids);
  EXPECT_EQ(model->regime, "ambiguous");
}

// -----------------------------------------------------------------------
// BuildSpaces: element/actor counts + footprint kinds materialize correctly.
// -----------------------------------------------------------------------

TEST(ScenarioBuilderTest, BuildsSpacesAndActorsFromT1) {
  auto model = LoadScenario(TestdataPath("T1-straight-approach.scenario.json"));
  ASSERT_TRUE(model.ok()) << model.status();
  auto ctx = LocalContext::Create();
  auto built = BuildSpaces(*model, *ctx);
  ASSERT_TRUE(built.ok()) << built.status();

  const BuiltSpace* a = built->FindSpace("A");
  const BuiltSpace* b = built->FindSpace("B");
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(a->space, nullptr);
  EXPECT_NE(b->space, nullptr);
  EXPECT_NE(a->element_id_by_tag.find("room"), a->element_id_by_tag.end());
  EXPECT_NE(a->element_id_by_shared_id.find(201), a->element_id_by_shared_id.end());
  EXPECT_NE(b->element_id_by_shared_id.find(201), b->element_id_by_shared_id.end());

  ASSERT_EQ(built->actors.size(), 1u);
  const BuiltActor* actor = built->FindActor("user");
  ASSERT_NE(actor, nullptr);
  EXPECT_EQ(actor->scenario_space_id, "A");
  Actor* a_actor = a->space->GetActor(actor->actor_id);
  ASSERT_NE(a_actor, nullptr);
  EXPECT_EQ(a_actor->GetAnchors().size(), 4u);
}

// -----------------------------------------------------------------------
// Acceptance (b): metamorphic.derive_target_from_source — derived target
// coordinates equal T(source) within < 1e-9. Self-authored (none of the 3
// spec fixtures exercise this field).
// -----------------------------------------------------------------------

std::string MetamorphicScenarioJson() {
  return R"JSON(
{
  "schema_version": "1.0",
  "id": "self-authored-metamorphic-derive",
  "tier": "T2",
  "regime": "jointly_feasible",
  "verify_by": ["metamorphic"],
  "metamorphic": {
    "derive_target_from_source": {
      "source": "A",
      "target": "B",
      "translate": [5.0, -2.0, 0.0],
      "rotate_yaw": 0.5235987755982988,
      "rotate_center": [0.0, 0.0]
    }
  },
  "spaces": [
    {
      "id": "A",
      "role": "source",
      "walkable": [[-2,-2],[2,-2],[2,2],[-2,2]],
      "elements": [
        {
          "tag": "chair",
          "shared_id": 500,
          "parent": "room",
          "transform": {"pos": [1.0, 0.5, 0.0], "yaw": 0.0},
          "footprint": {"kind": "box", "size": [0.5, 0.5, 0.9]}
        }
      ]
    },
    {
      "id": "B",
      "role": "target",
      "walkable": [[0,0],[1,0],[1,1],[0,1]],
      "elements": []
    }
  ],
  "actors": [
    {
      "id": "user",
      "space": "A",
      "shared_id": 900,
      "anchors": [{"name": "root", "offset": [0,0,0]}],
      "bindings": [{"anchor": "root", "kernel": "spacing"}]
    }
  ],
  "behavior": {"kind": "script", "script": [{"op": "wait", "duration": 1.0}]},
  "expect": [{"check": "no_containment_violation"}]
}
)JSON";
}

TEST(ScenarioMetamorphicTest, DeriveTargetFromSourceMatchesTransformExactly) {
  auto model = ParseScenario(MetamorphicScenarioJson());
  ASSERT_TRUE(model.ok()) << model.status();

  const ScenarioSpace* src = model->FindSpace("A");
  const ScenarioSpace* tgt = model->FindSpace("B");
  ASSERT_NE(src, nullptr);
  ASSERT_NE(tgt, nullptr);

  // Independently compute T(source) using the exact contract documented in
  // scenario.h: mirror -> scale -> rotate (about rotate_center) -> translate.
  const double yaw = 0.5235987755982988;  // pi/6
  const double tx = 5.0, ty = -2.0;
  const double c = std::cos(yaw), s = std::sin(yaw);

  auto ExpectTransformed = [&](double sx, double sy, double ex_x, double ex_y) {
    double rx = sx * c - sy * s;
    double ry = sx * s + sy * c;
    EXPECT_NEAR(rx + tx, ex_x, kEps);
    EXPECT_NEAR(ry + ty, ex_y, kEps);
  };
  (void)ExpectTransformed;

  ASSERT_EQ(tgt->elements.size(), src->elements.size());
  for (size_t i = 0; i < src->elements.size(); ++i) {
    const double sx = src->elements[i].transform.pos.x;
    const double sy = src->elements[i].transform.pos.y;
    const double want_x = tx + sx * c - sy * s;
    const double want_y = ty + sx * s + sy * c;
    EXPECT_NEAR(tgt->elements[i].transform.pos.x, want_x, kEps);
    EXPECT_NEAR(tgt->elements[i].transform.pos.y, want_y, kEps);
    EXPECT_NEAR(tgt->elements[i].transform.yaw,
               src->elements[i].transform.yaw + yaw, kEps);
    EXPECT_EQ(tgt->elements[i].shared_id, src->elements[i].shared_id);
    EXPECT_EQ(tgt->elements[i].tag, src->elements[i].tag);
  }

  ASSERT_EQ(tgt->walkable.size(), src->walkable.size());
  for (size_t i = 0; i < src->walkable.size(); ++i) {
    const double sx = src->walkable[i].x;
    const double sy = src->walkable[i].y;
    const double want_x = tx + sx * c - sy * s;
    const double want_y = ty + sx * s + sy * c;
    EXPECT_NEAR(tgt->walkable[i].x, want_x, kEps);
    EXPECT_NEAR(tgt->walkable[i].y, want_y, kEps);
  }
}

TEST(ScenarioMetamorphicTest, DerivedTargetBuildsSuccessfully) {
  auto model = ParseScenario(MetamorphicScenarioJson());
  ASSERT_TRUE(model.ok()) << model.status();
  auto ctx = LocalContext::Create();
  auto built = BuildSpaces(*model, *ctx);
  ASSERT_TRUE(built.ok()) << built.status();
  const BuiltSpace* b = built->FindSpace("B");
  ASSERT_NE(b, nullptr);
  EXPECT_NE(b->element_id_by_shared_id.find(500), b->element_id_by_shared_id.end());
}

// -----------------------------------------------------------------------
// Acceptance (c): malformed / missing-required-field JSON -> Status
// error, never a crash.
// -----------------------------------------------------------------------

TEST(ScenarioLoaderTest, MissingFileReturnsError) {
  auto model = LoadScenario(TestdataPath("does-not-exist.scenario.json"));
  EXPECT_FALSE(model.ok());
}

TEST(ScenarioLoaderTest, InvalidJsonReturnsError) {
  auto model = ParseScenario("{ this is not valid json ");
  EXPECT_FALSE(model.ok());
}

TEST(ScenarioLoaderTest, MissingRequiredTopLevelFieldReturnsError) {
  // Missing "expect" (a required top-level field).
  auto model = ParseScenario(R"JSON(
{
  "schema_version": "1.0",
  "id": "malformed",
  "tier": "T1",
  "regime": "n/a",
  "verify_by": [],
  "spaces": [{"id": "A", "role": "source", "walkable": [[0,0],[1,0],[1,1]], "elements": []}],
  "actors": [{"id": "u", "space": "A", "shared_id": 1,
              "anchors": [{"name": "root", "offset": [0,0,0]}],
              "bindings": []}],
  "behavior": {"kind": "script", "script": []}
}
)JSON");
  EXPECT_FALSE(model.ok());
}

TEST(ScenarioLoaderTest, MissingElementRequiredFieldReturnsError) {
  // Element missing "shared_id".
  auto model = ParseScenario(R"JSON(
{
  "schema_version": "1.0",
  "id": "malformed2",
  "tier": "T1",
  "regime": "n/a",
  "verify_by": [],
  "spaces": [{"id": "A", "role": "source", "walkable": [[0,0],[1,0],[1,1]],
              "elements": [{"tag": "x", "parent": "room",
                           "transform": {"pos": [0,0,0]}}]}],
  "actors": [{"id": "u", "space": "A", "shared_id": 1,
              "anchors": [{"name": "root", "offset": [0,0,0]}],
              "bindings": []}],
  "behavior": {"kind": "script", "script": []},
  "expect": []
}
)JSON");
  EXPECT_FALSE(model.ok());
}

TEST(ScenarioLoaderTest, WalkableWithFewerThanThreePointsReturnsError) {
  auto model = ParseScenario(R"JSON(
{
  "schema_version": "1.0",
  "id": "malformed3",
  "tier": "T1",
  "regime": "n/a",
  "verify_by": [],
  "spaces": [{"id": "A", "role": "source", "walkable": [[0,0],[1,0]], "elements": []}],
  "actors": [{"id": "u", "space": "A", "shared_id": 1,
              "anchors": [{"name": "root", "offset": [0,0,0]}],
              "bindings": []}],
  "behavior": {"kind": "script", "script": []},
  "expect": []
}
)JSON");
  EXPECT_FALSE(model.ok());
}

TEST(ScenarioBuilderTest, UnresolvableParentReturnsErrorNoCrash) {
  auto model = ParseScenario(R"JSON(
{
  "schema_version": "1.0",
  "id": "bad-parent",
  "tier": "T1",
  "regime": "n/a",
  "verify_by": [],
  "spaces": [{"id": "A", "role": "source", "walkable": [[0,0],[1,0],[1,1],[0,1]],
              "elements": [{"tag": "orphan", "shared_id": 1, "parent": "ghost",
                           "transform": {"pos": [0,0,0]}}]}],
  "actors": [{"id": "u", "space": "A", "shared_id": 1,
              "anchors": [{"name": "root", "offset": [0,0,0]}],
              "bindings": []}],
  "behavior": {"kind": "script", "script": []},
  "expect": []
}
)JSON");
  ASSERT_TRUE(model.ok()) << model.status();
  auto ctx = LocalContext::Create();
  auto built = BuildSpaces(*model, *ctx);
  EXPECT_FALSE(built.ok());
}

}  // namespace
}  // namespace harness
}  // namespace spatialblend
