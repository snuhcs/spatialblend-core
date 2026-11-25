// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/model/embodiment/embodiment_test.cc
//
// The embodiment spec and the registration's arithmetic. Every expected value is an
// independent oracle computed in this file. The spec in kSpec is a TEST FIXTURE: its numbers are the SMPL-X sit-bone
// and palm points that the evaluation harness (not public) (transfer_suite/embodiment.py, tools/measure_sitbone_offset.py) records, used here
// only to exercise the schema; they are not a constant of the core.
#include "spatialblend/model/embodiment/embodiment.h"

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <string>

#include <nlohmann/json.hpp>

namespace spatialblend {
namespace {

using nlohmann::json;

const char* kSpec = R"({
 "format":"spatialblend.embodiment","schema_version":"1.0.0","id":"fixture-smplx",
 "bone_frame":"x forward, y left, z up in the registered rest pose",
 "skeleton":{"joints":[{"name":"pelvis","parent":-1},{"name":"spine","parent":0},{"name":"head","parent":1},
                       {"name":"right_wrist","parent":1}]},
 "anchors":[
  {"name":"root","bone":null,"families":["Spacing"]},
  {"name":"head","bone":"head","families":["Facing"]},
  {"name":"pelvis","bone":"pelvis","offset":{"position":[0.0634,0.0,-0.1860]},"families":["AffordancePose"],
   "provenance":{"procedure":"the evaluation harness (not public) measure_sitbone_offset.py","source":"test fixture: the evaluation harness (not public) numbers"}}
 ]})";

json Spec() { return json::parse(kSpec); }

bool Parse(const json& j, Embodiment* e, std::string* err) { return Embodiment::FromJson(j.dump(), e, err); }

TEST(EmbodimentSpec, ParsesAndExposesTheRegistration) {
  Embodiment e;
  std::string err;
  ASSERT_TRUE(Embodiment::FromJson(kSpec, &e, &err)) << err;
  EXPECT_EQ(e.id, "fixture-smplx");
  ASSERT_EQ(e.joints.size(), 4u);
  EXPECT_EQ(e.BoneId("head"), 2);
  EXPECT_EQ(e.BoneId("nope"), -1);
  const EmbodimentAnchor* pelvis = e.FindAnchor("pelvis");
  ASSERT_NE(pelvis, nullptr);
  EXPECT_EQ(pelvis->bone, 0);
  EXPECT_FALSE(pelvis->identity_offset);
  EXPECT_EQ(pelvis->families, kFamilyAffordancePose);
  EXPECT_DOUBLE_EQ(pelvis->offset.position.z(), -0.1860);
  EXPECT_EQ(e.FindAnchor("root")->bone, -1);
  EXPECT_TRUE(e.FindAnchor("head")->identity_offset);
}

TEST(EmbodimentSpec, CanonicalJsonRoundTripsExactly) {
  Embodiment a, b;
  std::string err;
  ASSERT_TRUE(Embodiment::FromJson(kSpec, &a, &err)) << err;
  const std::string text = a.ToJson();
  ASSERT_TRUE(Embodiment::FromJson(text, &b, &err)) << err;
  EXPECT_EQ(text, b.ToJson());  // a fixed point
  EXPECT_EQ(b.FindAnchor("pelvis")->offset.position.x(), 0.0634);  // doubles round-trip
  EXPECT_NE(text.find("\"provenance\""), std::string::npos);
}

TEST(EmbodimentSpec, ARefusedSpecSaysWhy) {
  struct Case {
    const char* what;
    std::function<void(json&)> mutate;
    const char* expect;
  };
  const Case cases[] = {
      {"wrong format", [](json& j) { j["format"] = "x"; }, "format"},
      {"major 2", [](json& j) { j["schema_version"] = "2.0.0"; }, "schema_version"},
      {"no joints",
       [](json& j) {
         j["skeleton"]["joints"] = json::array();
         j["anchors"] = json::array();
       },
       "1..1024"},
      {"forward parent", [](json& j) { j["skeleton"]["joints"][0]["parent"] = 2; }, "earlier"},
      {"duplicate joint", [](json& j) { j["skeleton"]["joints"][1]["name"] = "pelvis"; }, "duplicate joint"},
      {"unknown bone", [](json& j) { j["anchors"][1]["bone"] = "tail"; }, "unknown bone"},
      {"duplicate anchor", [](json& j) { j["anchors"][1]["name"] = "pelvis"; }, "duplicate anchor"},
      {"no family", [](json& j) { j["anchors"][1]["families"] = json::array(); }, "serves no family"},
      {"bad family", [](json& j) { j["anchors"][1]["families"] = {"Gaze"}; }, "family must be"},
      {"offset without provenance", [](json& j) { j["anchors"][2].erase("provenance"); }, "provenance.procedure"},
      {"offset with neither a hash nor a source",
       [](json& j) { j["anchors"][2]["provenance"].erase("source"); }, "provenance.asset_sha256"},
      {"offset with a malformed hash",
       [](json& j) {
         j["anchors"][2]["provenance"].erase("source");
         j["anchors"][2]["provenance"]["asset_sha256"] = "fixture";
       },
       "64 hex"},
      {"root with an offset",
       [](json& j) {
         j["anchors"][0]["offset"] = {{"position", {0, 0, 0.1}}};
         j["anchors"][0]["provenance"] = {{"procedure", "p"}, {"asset_sha256", "h"}};
       },
       "no offset"},
      {"bone-bound root", [](json& j) { j["anchors"][0]["bone"] = "pelvis"; }, "actor root"},
      {"non-finite offset", [](json& j) { j["anchors"][2]["offset"]["position"][0] = "x"; }, "3 finite"},
      {"zero quaternion", [](json& j) { j["anchors"][2]["offset"]["rotation"] = {0, 0, 0, 0}; }, "zero length"},
  };
  for (const Case& c : cases) {
    json j = Spec();
    c.mutate(j);
    Embodiment e;
    std::string err;
    EXPECT_FALSE(Parse(j, &e, &err)) << c.what;
    EXPECT_NE(err.find(c.expect), std::string::npos) << c.what << ": " << err;
  }
  Embodiment e;
  std::string err;
  EXPECT_FALSE(Embodiment::FromJson("not json", &e, &err));
}

TEST(EmbodimentSpec, AHexHashOrASourceIsProvenance) {
  json j = Spec();
  j["anchors"][2]["provenance"] = {{"procedure", "p"}, {"asset_sha256", std::string(64, 'a')}};
  Embodiment e;
  std::string err;
  EXPECT_TRUE(Parse(j, &e, &err)) << err;
  j["anchors"][2]["provenance"] = {{"procedure", "p"}, {"source", "vendor datasheet"}};
  EXPECT_TRUE(Parse(j, &e, &err)) << err;
  j["anchors"][2]["provenance"] = {{"procedure", "p"}, {"source", ""}};
  EXPECT_FALSE(Parse(j, &e, &err));
}

TEST(EmbodimentSpec, ResourceCapsRefuseHostileInput) {
  Embodiment e;
  std::string err;
  EXPECT_FALSE(Embodiment::FromJson(std::string(5u << 20, ' '), &e, &err));
  EXPECT_NE(err.find("cap"), std::string::npos) << err;
  std::string deep(40, '[');
  deep += std::string(40, ']');
  EXPECT_FALSE(Embodiment::FromJson(deep, &e, &err));
  EXPECT_NE(err.find("nested"), std::string::npos) << err;
  // Brackets inside strings do not count.
  json j = Spec();
  j["bone_frame"] = std::string(100, '[') + "]";
  EXPECT_TRUE(Parse(j, &e, &err)) << err;
}

TEST(EmbodimentSpec, TheBodyFootprintRadiusIsRegisteredWithProvenance) {
  json j = Spec();
  Embodiment e;
  std::string err;
  ASSERT_TRUE(Parse(j, &e, &err)) << err;
  EXPECT_FALSE(e.body.present) << "the body is optional: an embodiment that does not declare a radius has none";
  j["body"] = {{"floor_footprint_radius_m", 0.31}, {"provenance", {{"procedure", "fixture"}, {"source", "test fixture"}}}};
  ASSERT_TRUE(Parse(j, &e, &err)) << err;
  EXPECT_TRUE(e.body.present);
  EXPECT_DOUBLE_EQ(e.body.floor_footprint_radius_m, 0.31);
  Embodiment back;
  ASSERT_TRUE(Embodiment::FromJson(e.ToJson(), &back, &err)) << err;
  EXPECT_EQ(back.ToJson(), e.ToJson());
  EXPECT_DOUBLE_EQ(json::parse(e.ToJson())["body"]["floor_footprint_radius_m"].get<double>(), 0.31);
  for (const char* bad : {"0", "-0.1", "\"x\""}) {
    json k = j;
    k["body"]["floor_footprint_radius_m"] = json::parse(bad);
    EXPECT_FALSE(Parse(k, &e, &err)) << bad;
  }
  json no_prov = j;
  no_prov["body"].erase("provenance");
  EXPECT_FALSE(Parse(no_prov, &e, &err));
  EXPECT_NE(err.find("body needs provenance"), std::string::npos) << err;
}

// ABI 4.11: an anchor's standing height over the actor's root instantiates the `stand`-like
// affordance poses for this embodiment. Optional; when present it is a finite positive length with provenance.
TEST(EmbodimentSpec, AnAnchorsStandingHeightIsRegisteredWithProvenance) {
  json j = Spec();
  Embodiment e;
  std::string err;
  ASSERT_TRUE(Parse(j, &e, &err)) << err;
  EXPECT_FALSE(e.FindAnchor("pelvis")->has_standing) << "no default";
  j["anchors"][2]["standing"] = {{"height_m", 0.912}, {"provenance", {{"procedure", "fixture"}, {"source", "test fixture"}}}};
  ASSERT_TRUE(Parse(j, &e, &err)) << err;
  ASSERT_TRUE(e.FindAnchor("pelvis")->has_standing);
  EXPECT_DOUBLE_EQ(e.FindAnchor("pelvis")->standing_height_m, 0.912);
  Embodiment back;
  ASSERT_TRUE(Embodiment::FromJson(e.ToJson(), &back, &err)) << err;
  EXPECT_EQ(back.ToJson(), e.ToJson());
  for (const char* bad : {"0", "-0.5", "\"x\""}) {
    json k = j;
    k["anchors"][2]["standing"]["height_m"] = json::parse(bad);
    EXPECT_FALSE(Parse(k, &e, &err)) << bad;
  }
  json no_prov = j;
  no_prov["anchors"][2]["standing"].erase("provenance");
  EXPECT_FALSE(Parse(no_prov, &e, &err));
  EXPECT_NE(err.find("standing needs provenance"), std::string::npos) << err;
}

TEST(EmbodimentSpec, AnIdentityOffsetNeedsNoProvenance) {
  json j = Spec();
  EXPECT_TRUE(j["anchors"][1].find("provenance") == j["anchors"][1].end());
  Embodiment e;
  std::string err;
  EXPECT_TRUE(Parse(j, &e, &err)) << err;
}

// anchor = bone * offset; the yaw-only case against the eval adapter's arithmetic
// (transfer_suite/embodiment.py anchor_contact_point): joint + (c f - s l, s f + c l, u).
TEST(EmbodimentArithmetic, YawOnlyComposeIsTheAdapterFormula) {
  Embodiment e;
  std::string err;
  ASSERT_TRUE(Embodiment::FromJson(kSpec, &e, &err)) << err;
  const EmbodimentAnchor& a = *e.FindAnchor("pelvis");
  const double f = 0.0634, l = 0.0, u = -0.1860;
  for (double yaw : {0.0, 0.7, -2.1, 3.0}) {
    RigidTransform bone;
    bone.position = Eigen::Vector3d(1.5, -2.0, 0.9);
    bone.rotation = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    const RigidTransform got = ComposeAnchor(a, bone);
    const double c = std::cos(yaw), s = std::sin(yaw);
    EXPECT_NEAR(got.position.x(), 1.5 + c * f - s * l, 1e-14);
    EXPECT_NEAR(got.position.y(), -2.0 + s * f + c * l, 1e-14);
    EXPECT_NEAR(got.position.z(), 0.9 + u, 1e-14);
    EXPECT_NEAR(EulerFromRotation(got.rotation).yaw, std::remainder(yaw, 2 * M_PI), 1e-14);
  }
}

TEST(EmbodimentArithmetic, ATiltedBoneMovesTheContactPoint) {
  Embodiment e;
  std::string err;
  ASSERT_TRUE(Embodiment::FromJson(kSpec, &e, &err)) << err;
  const EmbodimentAnchor& a = *e.FindAnchor("pelvis");
  RigidTransform bone;  // pitched 90 degrees nose-down about +y: bone-local -z maps to world -x... computed by hand
  bone.rotation = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()));
  // R_y(90): (x, y, z) -> (z, y, -x). offset (0.0634, 0, -0.1860) -> (-0.1860, 0, -0.0634).
  const RigidTransform got = ComposeAnchor(a, bone);
  EXPECT_NEAR(got.position.x(), -0.1860, 1e-15);
  EXPECT_NEAR(got.position.y(), 0.0, 1e-15);
  EXPECT_NEAR(got.position.z(), -0.0634, 1e-15);
}

TEST(EmbodimentArithmetic, TheInverseIsTheInverse) {
  Embodiment e;
  std::string err;
  ASSERT_TRUE(Embodiment::FromJson(kSpec, &e, &err)) << err;
  const EmbodimentAnchor& a = *e.FindAnchor("pelvis");
  RigidTransform anchor;
  anchor.position = Eigen::Vector3d(3.25, 0.5, 0.45);
  anchor.rotation = Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Eigen::Vector3d(1, 2, 3).normalized()));
  const RigidTransform bone = BoneFromAnchor(a, anchor);
  const RigidTransform back = ComposeAnchor(a, bone);
  EXPECT_NEAR((back.position - anchor.position).norm(), 0.0, 1e-14);
  EXPECT_NEAR(back.rotation.angularDistance(anchor.rotation), 0.0, 1e-7);  // acos near 1: sqrt(eps)
  // The identity-yaw case, by hand: the bone sits at anchor - offset.
  RigidTransform flat;
  flat.position = Eigen::Vector3d(0, 0, 0.45);
  const RigidTransform b2 = BoneFromAnchor(a, flat);
  EXPECT_NEAR(b2.position.x(), -0.0634, 1e-15);
  EXPECT_NEAR(b2.position.z(), 0.45 + 0.1860, 1e-15);
}

TEST(EmbodimentArithmetic, EulerConventionIsZyxWithPositivePitchNoseDown) {
  const EulerZyx a = EulerFromRotation(Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY())));
  EXPECT_NEAR(a.pitch, 0.3, 1e-15);  // forward axis tips down
  EXPECT_NEAR(a.yaw, 0.0, 1e-15);
  const Eigen::Quaterniond q = Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()) *
                               Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()) *
                               Eigen::AngleAxisd(-0.1, Eigen::Vector3d::UnitX());
  const EulerZyx b = EulerFromRotation(q);
  EXPECT_NEAR(b.yaw, 0.5, 1e-14);
  EXPECT_NEAR(b.pitch, 0.2, 1e-14);
  EXPECT_NEAR(b.roll, -0.1, 1e-14);
}

// ---- the normalized kinematic chain (design section 12) ----
// FIXTURE numbers (a toy 4-joint body, not a measurement): they exist to exercise the format.
json ChainSpec() {
  json j = Spec();
  j["chain"] = {
      {"rest_pose", "A"},
      {"provenance", {{"procedure", "fixture"}, {"source", "test fixture"}}},
      {"limits_provenance", {{"procedure", "fixture limits"}, {"source", "test fixture"}}},
      {"joints",
       {{{"name", "pelvis"}, {"rest", {{"position", {0, 0, 0}}, {"rotation", {0, 0, 0, 1}}}}},
        {{"name", "spine"},
         {"rest", {{"position", {0.01, 0.0, 0.2}}, {"rotation", {0, 0, 0.3826834323650898, 0.9238795325112867}}}},
         {"limits", {{"x", {-0.5, 0.5}}, {"y", {nullptr, 0.7}}}}},
        {{"name", "head"}, {"rest", {{"position", {0.01, 0.0, 0.55}}, {"rotation", {0, 0, 0, 1}}}}},
        {{"name", "right_wrist"},
         {"rest", {{"position", {0.0, -0.45, 0.4}}, {"rotation", {0, 0.7071067811865476, 0, 0.7071067811865476}}}}}}}};
  return j;
}

TEST(EmbodimentChain, RegistersAndReadsBackExactly) {
  Embodiment e;
  std::string err;
  ASSERT_TRUE(Parse(ChainSpec(), &e, &err)) << err;
  ASSERT_TRUE(e.chain.present);
  EXPECT_EQ(e.chain.rest_pose, "A");
  EXPECT_DOUBLE_EQ(e.chain.joints[3].rest.position.y(), -0.45);
  EXPECT_TRUE(e.chain.joints[1].has_lo[0] && e.chain.joints[1].has_hi[0]);
  EXPECT_FALSE(e.chain.joints[1].has_lo[1]);
  EXPECT_TRUE(e.chain.joints[1].has_hi[1]);
  EXPECT_FALSE(e.chain.joints[1].has_lo[2] || e.chain.joints[1].has_hi[2]);
  // Segment lengths are derived from the rest positions: spine = |(0.01,0,0.2)|, head = |(0,0,0.35)|,
  // right_wrist (parent spine) = |(-0.01,-0.45,0.2)|.
  EXPECT_NEAR(SegmentLength(e, 0), 0.0, 0.0);
  EXPECT_NEAR(SegmentLength(e, 1), std::sqrt(0.01 * 0.01 + 0.2 * 0.2), 1e-15);
  EXPECT_NEAR(SegmentLength(e, 2), 0.35, 1e-15);
  EXPECT_NEAR(SegmentLength(e, 3), std::sqrt(0.01 * 0.01 + 0.45 * 0.45 + 0.2 * 0.2), 1e-15);

  // JSON out and back is a fixed point, and carries the normalization block the core defines.
  const std::string text = e.ToJson();
  Embodiment back;
  ASSERT_TRUE(Embodiment::FromJson(text, &back, &err)) << err;
  EXPECT_EQ(text, back.ToJson());
  const json out = json::parse(text);
  EXPECT_EQ(out["chain"]["normalization"]["up"], "z");
  EXPECT_EQ(out["chain"]["normalization"]["handedness"], "right");
  ASSERT_TRUE(back.chain.joints[1].has_hi[1]);
  EXPECT_DOUBLE_EQ(back.chain.joints[1].hi[1], 0.7);
}

TEST(EmbodimentChain, ARefusedChainSaysWhy) {
  struct Case {
    const char* what;
    std::function<void(json&)> mutate;
    const char* expect;
  };
  const Case cases[] = {
      {"no provenance", [](json& j) { j["chain"].erase("provenance"); }, "chain needs provenance"},
      {"no rest pose name", [](json& j) { j["chain"]["rest_pose"] = ""; }, "rest_pose"},
      {"unknown rest pose", [](json& j) { j["chain"]["rest_pose"] = "Y"; }, "rest_pose"},
      {"root not at the origin", [](json& j) { j["chain"]["joints"][0]["rest"]["position"][2] = 0.1; }, "exactly 0,0,0"},
      {"a joint missing", [](json& j) { j["chain"]["joints"].erase(j["chain"]["joints"].begin() + 2); }, "lacks joint"},
      {"a joint repeated", [](json& j) { j["chain"]["joints"][2]["name"] = "spine"; }, "repeated"},
      {"unknown joint", [](json& j) { j["chain"]["joints"][2]["name"] = "tail"; }, "not in the skeleton"},
      {"limits without provenance", [](json& j) { j["chain"].erase("limits_provenance"); }, "limits_provenance"},
      {"min above max", [](json& j) { j["chain"]["joints"][1]["limits"]["x"] = {0.5, -0.5}; }, "min > max"},
      {"normalization differs", [](json& j) { j["chain"]["normalization"] = {{"up", "y"}}; }, "normalization"},
      {"zero rotation", [](json& j) { j["chain"]["joints"][2]["rest"]["rotation"] = {0, 0, 0, 0}; }, "zero length"},
  };
  for (const Case& c : cases) {
    json j = ChainSpec();
    c.mutate(j);
    Embodiment e;
    std::string err;
    EXPECT_FALSE(Parse(j, &e, &err)) << c.what;
    EXPECT_NE(err.find(c.expect), std::string::npos) << c.what << ": " << err;
  }
}

// native -> normalized -> native, for a right-handed y-up/z-forward/x-left skeleton (SMPL, glTF/OVR preset avatars)
// and a left-handed y-up Unity one: exact within double rounding, and the canonical chain is the same body.
TEST(EmbodimentChain, NormalizationRoundTripsForBothBases) {
  // canonical c = B n.  Right-handed native (x left, y up, z forward): c.x = n.z (forward), c.y = n.x (left), c.z = n.y.
  const double rh[9] = {0, 0, 1, 1, 0, 0, 0, 1, 0};
  // Left-handed Unity (x right, y up, z forward): c.x = n.z, c.y = -n.x (left = -right), c.z = n.y.  det = -1.
  const double lh[9] = {0, 0, 1, -1, 0, 0, 0, 1, 0};
  AxisBasis brh, blh;
  std::string err;
  ASSERT_TRUE(AxisBasis::FromRows(rh, &brh, &err)) << err;
  ASSERT_TRUE(AxisBasis::FromRows(lh, &blh, &err)) << err;
  EXPECT_NEAR(brh.b.determinant(), 1.0, 0.0);
  EXPECT_NEAR(blh.b.determinant(), -1.0, 0.0);

  const double bad[9] = {0, 0, 2, 1, 0, 0, 0, 1, 0};
  EXPECT_FALSE(AxisBasis::FromRows(bad, &brh, &err));
  const double bad2[9] = {1, 1, 0, 0, 0, 1, 0, 1, 0};
  EXPECT_FALSE(AxisBasis::FromRows(bad2, &brh, &err));

  // The same physical joint expressed natively in each basis must normalize to ONE canonical transform.
  RigidTransform canonical;
  canonical.position = Eigen::Vector3d(0.12, -0.45, 1.31);
  canonical.rotation = Eigen::Quaterniond(Eigen::AngleAxisd(0.8, Eigen::Vector3d(1, 2, -1).normalized()));
  const RigidTransform n_rh = FromCanonical(brh, canonical);
  const RigidTransform n_lh = FromCanonical(blh, canonical);
  // Hand check of the positions: RH native = (left, up, forward) = (-0.45, 1.31, 0.12); LH native = (right, up, fwd).
  EXPECT_NEAR(n_rh.position.x(), -0.45, 0.0);
  EXPECT_NEAR(n_rh.position.y(), 1.31, 0.0);
  EXPECT_NEAR(n_rh.position.z(), 0.12, 0.0);
  EXPECT_NEAR(n_lh.position.x(), 0.45, 0.0);
  for (const auto& pair : {std::make_pair(&brh, &n_rh), std::make_pair(&blh, &n_lh)}) {
    const RigidTransform back = ToCanonical(*pair.first, *pair.second);
    EXPECT_NEAR((back.position - canonical.position).norm(), 0.0, 1e-15);
    EXPECT_NEAR((back.rotation.toRotationMatrix() - canonical.rotation.toRotationMatrix()).norm(), 0.0, 1e-15);
    // native -> canonical -> native
    const RigidTransform again = FromCanonical(*pair.first, back);
    EXPECT_NEAR((again.position - pair.second->position).norm(), 0.0, 1e-15);
    EXPECT_NEAR((again.rotation.toRotationMatrix() - pair.second->rotation.toRotationMatrix()).norm(), 0.0, 1e-15);
  }
  // A handedness change keeps rotations proper (B R B^T), and turns a left-handed +yaw into the right-handed one:
  // native LH rotation about +y (up) by 0.5 rad is a clockwise-from-above turn in a left-handed system = canonical -0.5 about z.
  RigidTransform lh_turn;
  lh_turn.rotation = Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitY()));
  const RigidTransform c = ToCanonical(blh, lh_turn);
  EXPECT_NEAR(EulerFromRotation(c.rotation).yaw, -0.5, 1e-15);
  EXPECT_NEAR(c.rotation.toRotationMatrix().determinant(), 1.0, 1e-15);
}

// An INDEPENDENT right-handed check: expected quaternions are written by hand from the definition of the rotation about an
// axis, not computed with B and B^T of the code under test. Native right-handed (x left, y up, z forward) -> canonical
// (x forward, y left, z up): native y (up) is canonical z, native z (forward) is canonical x, native x (left) is canonical y.
TEST(EmbodimentChain, RightHandedRotationsConvertToTheHandComputedOnes) {
  const double rh[9] = {0, 0, 1, 1, 0, 0, 0, 1, 0};
  AxisBasis b;
  std::string err;
  ASSERT_TRUE(AxisBasis::FromRows(rh, &b, &err)) << err;
  const double h = std::sqrt(0.5);
  struct Case {
    Eigen::Vector3d native_axis;
    Eigen::Vector3d canonical_axis;
  };
  for (const Case& c : {Case{{0, 1, 0}, {0, 0, 1}}, Case{{0, 0, 1}, {1, 0, 0}}, Case{{1, 0, 0}, {0, 1, 0}}}) {
    RigidTransform native;  // 90 degrees about the native axis: q = (axis * sin 45, cos 45)
    native.rotation = Eigen::Quaterniond(h, c.native_axis.x() * h, c.native_axis.y() * h, c.native_axis.z() * h);
    const RigidTransform got = ToCanonical(b, native);
    const Eigen::Quaterniond want(h, c.canonical_axis.x() * h, c.canonical_axis.y() * h, c.canonical_axis.z() * h);
    EXPECT_NEAR(got.rotation.angularDistance(want), 0.0, 1e-7);
  }
  // A forward-pointing native vector (0,0,1) is canonical +x, and a 90 degree native yaw (about up) turns canonical +x to +y.
  RigidTransform yaw90;
  yaw90.rotation = Eigen::Quaterniond(h, 0, h, 0);
  const Eigen::Vector3d turned = ToCanonical(b, yaw90).rotation * Eigen::Vector3d::UnitX();
  EXPECT_NEAR((turned - Eigen::Vector3d::UnitY()).norm(), 0.0, 1e-15);
}

}  // namespace
}  // namespace spatialblend
