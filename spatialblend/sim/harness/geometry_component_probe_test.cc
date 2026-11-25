// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/sim/harness/geometry_component_probe.h"

#include <gtest/gtest.h>

#include <limits>

namespace spatialblend::harness {
namespace {

using json = nlohmann::json;

json Identity() {
  return {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
}

json ValidTransforms() {
  return json::array({
      {{0, -1, 0, 3}, {1, 0, 0, 2}, {0, 0, 1, 1}, {0, 0, 0, 1}},
      {{1, .25, 0, 1}, {0, 1, 0, -2}, {0, 0, 1, 0}, {0, 0, 0, 1}},
      {{1, 0, 0, .5}, {0, 1, 0, .25}, {0, 0, 1, 2}, {0, 0, 0, 1}},
      Identity(),
  });
}

TEST(GeometryComponentProbeTest, ComputesProductionHierarchyWorldPoints) {
  json input{
      {"schema_version", 1},
      {"cases",
       json::array({{{"case_id", "sheared-chain"},
                     {"component", "H"},
                     {"transforms", ValidTransforms()},
                     {"points", json::array({json::array({1, 2, 3})})}}})}};
  auto result = RunGeometryComponentProbe(input);
  ASSERT_TRUE(result.ok()) << result.status();
  const auto& output = result->at("results").at(0);
  EXPECT_TRUE(output.at("ok").get<bool>());
  EXPECT_EQ(output.at("world_points").at(0), json::array({2.75, 5.0625, 6.0}));
}

TEST(GeometryComponentProbeTest, CallsR1AndRetainsNativeOutOfDomainFailure) {
  const json triangle =
      json::array({json::array({0, 0, .5}), json::array({2, 0, .5}),
                   json::array({0, 2, .5})});
  json input{{"schema_version", 1},
             {"cases", json::array({
                           {{"case_id", "r1-positive"},
                            {"component", "R1"},
                            {"transforms", ValidTransforms()},
                            {"vertices", triangle},
                            {"angle", 0.0},
                            {"radial_ratio", 0.0},
                            {"relative_yaw", .3}},
                           {{"case_id", "r1-native-negative"},
                            {"component", "R1"},
                            {"transforms", ValidTransforms()},
                            {"vertices", triangle},
                            {"angle", 0.0},
                            {"radial_ratio", 1.1},
                            {"relative_yaw", .3}},
                       })}};
  auto result = RunGeometryComponentProbe(input);
  ASSERT_TRUE(result.ok()) << result.status();
  const auto& positive = result->at("results").at(0);
  EXPECT_TRUE(positive.at("ok").get<bool>());
  EXPECT_EQ(positive.at("mode"), "whole_planar_boundary");
  EXPECT_NEAR(positive.at("position").at(0).get<double>(), 2.0 / 3.0, 1e-12);
  EXPECT_NEAR(positive.at("position").at(1).get<double>(), 2.0 / 3.0, 1e-12);
  EXPECT_NEAR(positive.at("position").at(2).get<double>(), .5, 1e-12);
  EXPECT_NEAR(positive.at("yaw").get<double>(), .3, 1e-12);
  const auto& negative = result->at("results").at(1);
  EXPECT_FALSE(negative.at("ok").get<bool>());
  EXPECT_EQ(negative.at("native_status"), 3);
}

TEST(GeometryComponentProbeTest,
     GroundsTiltedBakedTriangleAtItsIndependentCentroid) {
  json input{
      {"schema_version", 1},
      {"cases", json::array({{{"case_id", "tilted"},
                              {"component", "R1"},
                              {"transforms", ValidTransforms()},
                              {"vertices", {{5, 0, 0}, {5, 2, 0}, {5, 0, 2}}},
                              {"angle", 0.0},
                              {"radial_ratio", 0.0},
                              {"relative_yaw", 0.0}}})}};
  auto result = RunGeometryComponentProbe(input);
  ASSERT_TRUE(result.ok()) << result.status();
  const auto& output = result->at("results").at(0);
  EXPECT_TRUE(output.at("ok").get<bool>());
  EXPECT_NEAR(output.at("position").at(0).get<double>(), 5.0, 1e-12);
  EXPECT_NEAR(output.at("position").at(1).get<double>(), 2.0 / 3.0, 1e-12);
  EXPECT_NEAR(output.at("position").at(2).get<double>(), 2.0 / 3.0, 1e-12);
}

TEST(GeometryComponentProbeTest, RejectsMalformedAndUnsafeInputsBeforeResults) {
  json malformed{{"schema_version", 1}, {"cases", json::array()}};
  EXPECT_FALSE(RunGeometryComponentProbe(malformed).ok());
  json duplicate{{"schema_version", 1},
                 {"cases", json::array({{{"case_id", "same"},
                                         {"component", "H"},
                                         {"transforms", ValidTransforms()},
                                         {"points", {{0, 0, 0}}}},
                                        {{"case_id", "same"},
                                         {"component", "H"},
                                         {"transforms", ValidTransforms()},
                                         {"points", {{0, 0, 0}}}}})}};
  EXPECT_FALSE(RunGeometryComponentProbe(duplicate).ok());
  json oversized_points = json::array();
  for (int i = 0; i < 129; ++i) oversized_points.push_back({0, 0, 0});
  json oversized{{"schema_version", 1},
                 {"cases", json::array({{{"case_id", "too-many"},
                                         {"component", "H"},
                                         {"transforms", ValidTransforms()},
                                         {"points", oversized_points}}})}};
  EXPECT_FALSE(RunGeometryComponentProbe(oversized).ok());

  json boolean_version = malformed;
  boolean_version["schema_version"] = true;
  boolean_version["cases"] = json::array({{{"case_id", "ok"},
                                           {"component", "H"},
                                           {"transforms", ValidTransforms()},
                                           {"points", {{0, 0, 0}}}}});
  EXPECT_FALSE(RunGeometryComponentProbe(boolean_version).ok());
  json overflow_version = boolean_version;
  overflow_version["schema_version"] = 4294967297ULL;
  EXPECT_FALSE(RunGeometryComponentProbe(overflow_version).ok());
  json unexpected_root = boolean_version;
  unexpected_root["schema_version"] = 1;
  unexpected_root["extra"] = 0;
  EXPECT_FALSE(RunGeometryComponentProbe(unexpected_root).ok());
  json nonfinite = boolean_version;
  nonfinite["schema_version"] = 1;
  nonfinite["cases"][0]["points"][0][0] =
      std::numeric_limits<double>::infinity();
  EXPECT_FALSE(RunGeometryComponentProbe(nonfinite).ok());
  json noninvertible = boolean_version;
  noninvertible["schema_version"] = 1;
  noninvertible["cases"][0]["transforms"][0] =
      json::array({json::array({0, 0, 0, 0}), json::array({0, 0, 0, 0}),
                   json::array({0, 0, 0, 0}), json::array({0, 0, 0, 1})});
  EXPECT_FALSE(RunGeometryComponentProbe(noninvertible).ok());
  json nonaffine = boolean_version;
  nonaffine["schema_version"] = 1;
  nonaffine["cases"][0]["transforms"][0][3][0] = 1e-16;
  EXPECT_FALSE(RunGeometryComponentProbe(nonaffine).ok());
  json huge_triangle{
      {"schema_version", 1},
      {"cases", json::array({{{"case_id", "huge"},
                              {"component", "R1"},
                              {"transforms", ValidTransforms()},
                              {"vertices",
                               {{1e308, 0, 0}, {-1e308, 0, 0}, {0, 1e308, 0}}},
                              {"angle", 0.0},
                              {"radial_ratio", 0.0},
                              {"relative_yaw", 0.0}}})}};
  EXPECT_FALSE(RunGeometryComponentProbe(huge_triangle).ok());
  json degenerate_triangle = huge_triangle;
  degenerate_triangle["cases"][0]["case_id"] = "degenerate";
  degenerate_triangle["cases"][0]["vertices"] = json::array(
      {json::array({0, 0, 0}), json::array({1, 0, 0}), json::array({2, 0, 0})});
  EXPECT_FALSE(RunGeometryComponentProbe(degenerate_triangle).ok());
}

json SourceCase(const std::string& id = "pelvis-source") {
  json samples = json::array();
  const double heights[] = {1.0, .55, .6, .8, .5, .5};
  for (int i = 0; i < 6; ++i) {
    samples.push_back({{"sequence", 17 + i},
                       {"time_seconds", i / 10.0},
                       {"position", {.5, .5, heights[i]}},
                       {"yaw", .3},
                       {"available", i != 5}});
  }
  return {{"case_id", id},
          {"component", "E"},
          {"transforms", ValidTransforms()},
          {"vertices", {{0, 0, .5}, {2, 0, .5}, {0, 2, .5}}},
          {"anchor_name", "pelvis"},
          {"affordance_tag", "Sittable"},
          {"samples", samples}};
}

TEST(GeometryComponentProbeTest, SourcePelvisContactNeedsNoObservedPlanarRoot) {
  auto result = RunGeometryComponentProbe(
      {{"schema_version", 1}, {"cases", json::array({SourceCase()})}});
  ASSERT_TRUE(result.ok()) << result.status();
  const auto& row = result->at("results").at(0);
  EXPECT_EQ(row.at("anchor_name"), "pelvis");
  EXPECT_EQ(row.at("affordance_tag"), "Sittable");
  EXPECT_EQ(row.at("root_observed"), false);
  const std::vector<std::string> phases = {"active",  "contact", "contact",
                                           "driving", "contact", "unavailable"};
  const double distances[] = {.5, .05, .1, .3, 0.0};
  ASSERT_EQ(row.at("samples").size(), 6);
  for (size_t i = 0; i < 6; ++i) {
    const auto& sample = row.at("samples").at(i);
    EXPECT_EQ(sample.at("sequence"), 17 + i);
    EXPECT_EQ(sample.at("time_seconds"), i / 10.0);
    EXPECT_EQ(sample.at("phase"), phases[i]);
    if (i < 5) {
      EXPECT_EQ(sample.at("support_available"), true);
      EXPECT_EQ(sample.at("support_point"), json::array({.5, .5, .5}));
      EXPECT_NEAR(sample.at("distance").get<double>(), distances[i], 1e-12);
      EXPECT_NEAR(sample.at("proximity").get<double>(),
                  1.0 / (1.0 + distances[i]), 1e-12);
    } else {
      EXPECT_EQ(sample.at("available"), false);
      EXPECT_EQ(sample.at("support_available"), false);
      EXPECT_TRUE(sample.at("distance").is_null());
      EXPECT_TRUE(sample.at("proximity").is_null());
      EXPECT_TRUE(sample.at("support_point").is_null());
    }
  }
}

TEST(GeometryComponentProbeTest, SourceSupportIsBoundedAndCasesOwnTheirPhase) {
  auto first = SourceCase("first");
  first["samples"][1]["position"] = {3, 0, .5};
  auto second = SourceCase("second");
  second["anchor_name"] = "other_body_root";
  auto result = RunGeometryComponentProbe(
      {{"schema_version", 1}, {"cases", json::array({first, second})}});
  ASSERT_TRUE(result.ok()) << result.status();
  const auto& outside = result->at("results").at(0).at("samples").at(1);
  EXPECT_EQ(outside.at("support_point"), json::array({2.0, 0.0, .5}));
  EXPECT_NEAR(outside.at("distance").get<double>(), 1.0, 1e-12);
  EXPECT_NEAR(outside.at("proximity").get<double>(), .5, 1e-12);
  EXPECT_EQ(outside.at("phase"), "driving");
  EXPECT_EQ(result->at("results").at(1).at("samples").at(0).at("phase"),
            "active");
}

TEST(GeometryComponentProbeTest,
     SourceGapsWithdrawContactAndStartNewPhaseLifetime) {
  auto entry = SourceCase();
  entry["samples"][2]["available"] = false;
  entry["samples"][3]["position"] = {.5, .5, .6};
  // Far from the seat. There is no affordance near field and no release dwell; the
  // relation stays in scope (the seat's parent region) and its phase
  // follows the weight and proximity labels, so it is DRIVING (weight 1, R < 0.95).
  entry["samples"][4]["position"] = {23., .5, .5};
  entry["samples"][5]["available"] = true;
  entry["samples"][5]["position"] = {.5, .5, .6};
  const auto result = RunGeometryComponentProbe(
      {{"schema_version", 1}, {"cases", json::array({entry})}});
  ASSERT_TRUE(result.ok()) << result.status();
  const auto& samples = result->at("results").at(0).at("samples");
  const std::vector<std::string> phases = {
      "active", "contact", "unavailable", "active", "driving", "driving"};
  for (size_t i = 0; i < samples.size(); ++i) {
    EXPECT_EQ(samples[i].at("sequence"), 17 + i);
    EXPECT_EQ(samples[i].at("time_seconds"), i / 10.0);
    EXPECT_EQ(samples[i].at("phase"), phases[i]);
    EXPECT_EQ(samples[i].at("available"), i != 2);
    if (i == 2) {
      EXPECT_EQ(samples[i].at("support_available"), false);
      EXPECT_TRUE(samples[i].at("distance").is_null());
      EXPECT_TRUE(samples[i].at("proximity").is_null());
      EXPECT_TRUE(samples[i].at("support_point").is_null());
    }
  }
  // An initial far observation is in scope too (no near field to enter): ACTIVE at once,
  // then DRIVING, where it used to be DORMANT until the anchor came within 1.5 m.
  entry["samples"][0]["position"] = {23., .5, .5};
  const auto initially_far = RunGeometryComponentProbe(
      {{"schema_version", 1}, {"cases", json::array({entry})}});
  ASSERT_TRUE(initially_far.ok()) << initially_far.status();
  EXPECT_EQ(initially_far->at("results")[0]["samples"][0]["phase"], "active");
  EXPECT_EQ(initially_far->at("results")[0]["samples"][1]["phase"], "contact");
}

TEST(GeometryComponentProbeTest,
     SourceRejectsBadFinalRowsAndIdentityWithoutPrefix) {
  const auto valid = SourceCase();
  std::vector<json> invalid;
  auto entry = valid;
  entry["samples"][5]["sequence"] = 17;
  invalid.push_back(entry);
  entry = valid;
  entry["samples"][5]["sequence"] = uint64_t(-1);
  invalid.push_back(entry);
  entry = valid;
  entry["samples"][5]["sequence"] = 22.0;
  invalid.push_back(entry);
  entry = valid;
  entry["samples"][5]["time_seconds"] = .1;
  invalid.push_back(entry);
  entry = valid;
  entry["samples"][0]["time_seconds"] = -.1;
  invalid.push_back(entry);
  entry = valid;
  entry["samples"][5]["available"] = 0;
  invalid.push_back(entry);
  entry = valid;
  entry["samples"][5]["position"][2] = std::numeric_limits<double>::infinity();
  invalid.push_back(entry);
  entry = valid;
  entry["samples"][5]["yaw"] = false;
  invalid.push_back(entry);
  entry = valid;
  entry["samples"][5]["contact"] = true;
  invalid.push_back(entry);
  entry = valid;
  entry["samples"] = json::array();
  invalid.push_back(entry);
  entry = valid;
  entry["anchor_name"] = "probe_unobserved_reference";
  invalid.push_back(entry);
  entry = valid;
  entry["anchor_name"] = std::string("p\0elvis", 7);
  invalid.push_back(entry);
  entry = valid;
  entry["affordance_tag"] = "";
  invalid.push_back(entry);
  entry = valid;
  entry["vertices"] = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
  invalid.push_back(entry);
  for (auto rejected : invalid) {
    // A distinct case ID ensures the malformed row is reached, rather than
    // accidentally passing every negative test at the duplicate-ID guard.
    rejected["case_id"] = "invalid-source";
    const auto result = RunGeometryComponentProbe(
        {{"schema_version", 1}, {"cases", json::array({valid, rejected})}});
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().message().find("case_id values must be unique"),
              absl::string_view::npos);
  }
  entry = valid;
  entry["samples"] = json::array();
  for (int i = 0; i < 20001; ++i) {
    auto sample = valid.at("samples").at(0);
    sample["sequence"] = i;
    entry["samples"].push_back(sample);
  }
  EXPECT_FALSE(RunGeometryComponentProbe(
                   {{"schema_version", 1}, {"cases", json::array({entry})}})
                   .ok());
}

}  // namespace
}  // namespace spatialblend::harness
