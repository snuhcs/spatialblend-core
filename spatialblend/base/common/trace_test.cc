// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/base/common/trace.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

namespace spatialblend {
namespace {

// C++ port of the reference trace parser's parse_trace_jsonl() rules --
// the shared reference parser both the native emitter (this file) and
// downstream trace consumers
// validate against. Kept in sync with that file's logic by hand (it is ~30
// lines of straightforward per-line JSON parsing with a legacy-format
// bypass, so a mechanical diff is easy to eyeball on review):
//   - blank lines are skipped
//   - a line that starts with "SB_V2_TRACE" or contains "SB_V2_TRACE]" is the
//     old text format, not JSON: it is skipped (not counted as
//     "parsed"), matching the Python reference's fallback-to-legacy-parser
//     behavior instead of raising.
//   - every other line is parsed as JSON and grouped by its "ev" field.
//   - n_lines (non-blank lines seen) and n_parsed (successfully-JSON-parsed
//     lines) are both reported so callers can assert 100% coverage exactly
//     like trace_parser_ref.py's main() does.
struct ParsedTrace {
  std::map<std::string, std::vector<nlohmann::json>> events;
  int n_lines = 0;
  int n_parsed = 0;
};

ParsedTrace ParseTraceJsonlRef(const std::string& path) {
  ParsedTrace result;
  std::ifstream in(path);
  std::string raw;
  while (std::getline(in, raw)) {
    std::string line = raw;
    // Trim trailing CR (files may have been written on a CRLF path) and
    // surrounding whitespace, mirroring Python's str.strip().
    size_t begin = line.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) continue;  // blank line
    size_t end = line.find_last_not_of(" \t\r\n");
    line = line.substr(begin, end - begin + 1);
    if (line.empty()) continue;
    ++result.n_lines;
    if (line.rfind("SB_V2_TRACE", 0) == 0 ||
        line.find("SB_V2_TRACE]") != std::string::npos) {
      continue;  // legacy text format: not JSON, not counted as parsed.
    }
    nlohmann::json ev;
    try {
      ev = nlohmann::json::parse(line);
    } catch (const nlohmann::json::parse_error&) {
      continue;
    }
    std::string kind = ev.value("ev", std::string());
    result.events[kind].push_back(ev);
    ++result.n_parsed;
  }
  return result;
}

// SPEC S3 vocabulary for ev=relation's "phase" field (api/c/trace.format.md).
const std::set<std::string> kSpecPhases = {"approach", "engage", "release"};

class TraceSinkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TraceSink::Instance().ResetForTest();
    TraceSink::ResetAllForTest();
    unsetenv("SB_V2_TRACE");
    unsetenv("SB_V2_TRACE_FILE");
  }
  void TearDown() override {
    TraceSink::Instance().ResetForTest();
    TraceSink::ResetAllForTest();
    unsetenv("SB_V2_TRACE");
    unsetenv("SB_V2_TRACE_FILE");
    if (!path_.empty()) {
      std::remove(path_.c_str());
      path_.clear();
    }
    for (const auto& p : extra_paths_) {
      std::remove(p.c_str());
    }
    extra_paths_.clear();
  }

  std::string TempPath() {
    path_ = "/tmp/spatialblend_trace_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
            "_" + std::to_string(reinterpret_cast<uintptr_t>(this)) + ".jsonl";
    return path_;
  }

  // A second/third distinct temp path, for tests that bind multiple space_ids
  // to different files.
  std::string ExtraPath(int n) {
    std::string p = "/tmp/spatialblend_trace_test_" +
                    std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
                    "_" + std::to_string(reinterpret_cast<uintptr_t>(this)) +
                    "_extra" + std::to_string(n) + ".jsonl";
    extra_paths_.push_back(p);
    return p;
  }

  std::string path_;
  std::vector<std::string> extra_paths_;
};

TEST_F(TraceSinkTest, InactiveByDefaultWithNoEnv) {
  EXPECT_FALSE(TraceSink::Instance().Active());
}

TEST_F(TraceSinkTest, SetPathActivatesAndWritesLines) {
  std::string path = TempPath();
  TraceSink::Instance().SetPath(path);
  EXPECT_TRUE(TraceSink::Instance().Active());

  nlohmann::ordered_json ev;
  ev["ev"] = "src";
  ev["t"] = 1.234;
  ev["space"] = 2;
  ev["actor"] = 9001;
  ev["pos"] = {1.0, 2.0};
  ev["yaw"] = 0.3;
  TraceSink::Instance().Emit(ev);

  TraceSink::Instance().Disable();

  std::ifstream in(path);
  ASSERT_TRUE(in.good());
  std::string line;
  ASSERT_TRUE(std::getline(in, line));
  nlohmann::json parsed = nlohmann::json::parse(line);
  EXPECT_EQ(parsed["ev"], "src");
  EXPECT_EQ(parsed["space"], 2);
  EXPECT_EQ(parsed["actor"], 9001);
  EXPECT_DOUBLE_EQ(parsed["yaw"].get<double>(), 0.3);
}

TEST_F(TraceSinkTest, DisableOverridesEnv) {
  setenv("SB_V2_TRACE", "1", 1);
  TraceSink::Instance().Disable();
  EXPECT_FALSE(TraceSink::Instance().Active());
}

TEST_F(TraceSinkTest, EnvFileActivatesLazily) {
  std::string path = TempPath();
  setenv("SB_V2_TRACE_FILE", path.c_str(), 1);
  EXPECT_TRUE(TraceSink::Instance().Active());

  nlohmann::ordered_json ev;
  ev["ev"] = "tick";
  ev["t"] = 0.5;
  ev["space"] = 1;
  ev["cost_us"] = 42;
  TraceSink::Instance().Emit(ev);
  TraceSink::Instance().Disable();

  std::ifstream in(path);
  ASSERT_TRUE(in.good());
  std::string line;
  ASSERT_TRUE(std::getline(in, line));
  nlohmann::json parsed = nlohmann::json::parse(line);
  EXPECT_EQ(parsed["ev"], "tick");
  EXPECT_EQ(parsed["cost_us"], 42);
}

TEST_F(TraceSinkTest, RoundTripAllSixEvKinds) {
  std::string path = TempPath();
  TraceSink::Instance().SetPath(path);

  auto& sink = TraceSink::Instance();

  {
    nlohmann::ordered_json ev;
    ev["ev"] = "src";
    ev["t"] = 1.0;
    ev["space"] = 2;
    ev["actor"] = 9001;
    ev["pos"] = {1.0, 2.0};
    ev["yaw"] = 0.3;
    sink.Emit(ev);
  }
  {
    nlohmann::ordered_json ev;
    ev["ev"] = "payload";
    ev["t"] = 1.0;
    ev["space"] = 2;
    ev["actor"] = 9001;
    ev["referent"] = 201;
    ev["anchor"] = "root";
    ev["kernel"] = 1;
    ev["R"] = 0.42;
    ev["dist"] = 1.8;
    ev["depth"] = 1;
    ev["weight"] = 0.7;
    ev["est_time"] = 2.1;
    ev["via"] = "linker";
    sink.Emit(ev);
  }
  {
    nlohmann::ordered_json ev;
    ev["ev"] = "target";
    ev["t"] = 1.0;
    ev["space"] = 3;
    ev["actor"] = 9001;
    ev["anchor"] = "root";
    ev["pos"] = {3.1, 1.2, 0.0};
    ev["has_z"] = false;
    ev["yaw"] = 0.9;
    ev["kernel"] = 1;
    ev["referent"] = 201;
    ev["weight"] = 0.7;
    sink.Emit(ev);
  }
  {
    nlohmann::ordered_json ev;
    ev["ev"] = "relation";
    ev["t"] = 1.3;
    ev["space"] = 3;
    ev["actor"] = 9001;
    ev["relation"] = "affordance_pose:seat";
    ev["phase"] = "engage";
    ev["tracker_phase"] = "contact";
    ev["R"] = 0.98;
    sink.Emit(ev);
  }
  {
    nlohmann::ordered_json ev;
    ev["ev"] = "share";
    ev["t"] = 0.01;
    ev["space"] = 3;
    ev["shared_id"] = 201;
    ev["received"] = 10001;
    ev["resolved"] = 5;
    ev["resolution"] = "link_to_existing";
    sink.Emit(ev);
  }
  {
    nlohmann::ordered_json ev;
    ev["ev"] = "tick";
    ev["t"] = 1.0;
    ev["space"] = 3;
    ev["cost_us"] = 180;
    sink.Emit(ev);
  }
  sink.Disable();

  std::ifstream in(path);
  ASSERT_TRUE(in.good());
  std::string line;
  int n = 0;
  const char* expected_kinds[] = {"src", "payload", "target", "relation", "share", "tick"};
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    nlohmann::json parsed = nlohmann::json::parse(line);
    ASSERT_LT(n, 6);
    EXPECT_EQ(parsed["ev"], expected_kinds[n]);
    ++n;
  }
  EXPECT_EQ(n, 6);
}

// field sets per `ev` kind, mirrored from
// spatialblend/api/c/trace.format.md (format 2). The authoritative check of
// real emitted lines against that file is
// //spatialblend/api/c:trace_spec_conformance_test; this test pins the sink's
// line/field round trip for the six format-1 kinds at their format-2 shape.
TEST_F(TraceSinkTest, FieldSetsMatchSpecPerEvKind) {
  std::string path = TempPath();
  TraceSink::Instance().SetPath(path);
  auto& sink = TraceSink::Instance();

  {
    nlohmann::ordered_json ev;
    ev["ev"] = "src";
    ev["t"] = 1.234;
    ev["space"] = 2;
    ev["actor"] = 9001;
    ev["pos"] = {1.0, 2.0};
    ev["yaw"] = 0.3;
    sink.Emit(ev);
  }
  {
    nlohmann::ordered_json ev;
    ev["ev"] = "payload";
    ev["t"] = 1.234;
    ev["space"] = 2;
    ev["actor"] = 9001;
    ev["referent"] = 201;
    ev["anchor"] = "root";
    ev["kernel"] = 1;
    ev["R"] = 0.42;
    ev["dist"] = 1.8;
    ev["depth"] = 1;
    ev["weight"] = 0.7;
    ev["est_time"] = 2.1;
    ev["via"] = "linker";
    ev["sub_region"] = "";
    ev["driving"] = true;
    sink.Emit(ev);
  }
  {
    nlohmann::ordered_json ev;
    ev["ev"] = "target";
    ev["t"] = 1.234;
    ev["space"] = 3;
    ev["actor"] = 9001;
    ev["anchor"] = "root";
    ev["pos"] = {3.1, 1.2, 0.0};
    ev["has_z"] = false;
    ev["yaw"] = 0.9;
    ev["kernel"] = 1;
    ev["referent"] = 201;
    ev["weight"] = 0.7;
    ev["status"] = "converged";
    ev["snap_m"] = 0.0;
    ev["snapped"] = false;
    ev["requested_pos"] = {3.1, 1.2};
    ev["reach_clamped"] = false;
    ev["reach_m"] = 0.0;
    sink.Emit(ev);
  }
  {
    nlohmann::ordered_json ev;
    ev["ev"] = "relation";
    ev["t"] = 1.3;
    ev["space"] = 3;
    ev["actor"] = 9001;
    ev["relation"] = "affordance_pose:seat";
    ev["phase"] = "engage";
    ev["tracker_phase"] = "contact";
    ev["R"] = 0.98;
    ev["referent"] = 201;
    sink.Emit(ev);
  }
  {
    nlohmann::ordered_json ev;
    ev["ev"] = "share";
    ev["t"] = 0.01;
    ev["space"] = 3;
    ev["shared_id"] = 201;
    ev["received"] = 10001;
    ev["resolved"] = 5;
    ev["resolution"] = "link_to_existing";
    sink.Emit(ev);
  }
  {
    nlohmann::ordered_json ev;
    ev["ev"] = "tick";
    ev["t"] = 1.234;
    ev["space"] = 3;
    ev["cost_us"] = 180;
    sink.Emit(ev);
  }
  sink.Disable();

  ParsedTrace parsed = ParseTraceJsonlRef(path);
  EXPECT_EQ(parsed.n_lines, 6);
  EXPECT_EQ(parsed.n_parsed, 6);

  // Field-set-per-kind, taken from api/c/trace.format.md's example line and
  // the sample trace (relation carries "tracker_phase" in
  // addition to the phase fields).
  const std::map<std::string, std::set<std::string>> kExpectedFields = {
      {"src", {"ev", "t", "space", "actor", "pos", "yaw"}},
      {"payload", {"ev", "t", "space", "actor", "referent", "anchor", "kernel",
                   "R", "dist", "depth", "weight", "est_time", "via",
                   "sub_region", "driving"}},
      {"target", {"ev", "t", "space", "actor", "anchor", "pos", "has_z", "yaw",
                  "kernel", "referent", "weight", "status", "snap_m",
                  "snapped", "requested_pos", "reach_clamped", "reach_m"}},
      {"relation", {"ev", "t", "space", "actor", "relation", "phase",
                    "tracker_phase", "R", "referent"}},
      {"share", {"ev", "t", "space", "shared_id", "received", "resolved",
                 "resolution"}},
      {"tick", {"ev", "t", "space", "cost_us"}},
  };
  for (const auto& [kind, expected] : kExpectedFields) {
    ASSERT_EQ(parsed.events[kind].size(), 1u) << "kind=" << kind;
    std::set<std::string> actual;
    for (auto it = parsed.events[kind][0].begin();
         it != parsed.events[kind][0].end(); ++it) {
      actual.insert(it.key());
    }
    EXPECT_EQ(actual, expected) << "kind=" << kind;
  }
}

// phase vocabulary must be a subset of SPEC S3's
// {"approach","engage","release"} -- the "tracker_phase" field is exempt
// (it deliberately carries the internal tracker vocabulary) but "phase"
// itself never should.
TEST_F(TraceSinkTest, RelationPhaseVocabularyIsSpecSubset) {
  std::string path = TempPath();
  TraceSink::Instance().SetPath(path);
  auto& sink = TraceSink::Instance();

  const struct {
    const char* phase;
    const char* tracker_phase;
  } kTransitions[] = {
      {"approach", "active"},
      {"approach", "driving"},
      {"engage", "contact"},
      {"release", "dormant"},
  };
  for (const auto& tr : kTransitions) {
    nlohmann::ordered_json ev;
    ev["ev"] = "relation";
    ev["t"] = 1.0;
    ev["space"] = 3;
    ev["actor"] = 9001;
    ev["relation"] = "spacing:root";
    ev["phase"] = tr.phase;
    ev["tracker_phase"] = tr.tracker_phase;
    ev["R"] = 0.5;
    sink.Emit(ev);
  }
  sink.Disable();

  ParsedTrace parsed = ParseTraceJsonlRef(path);
  ASSERT_EQ(parsed.n_parsed, parsed.n_lines);
  ASSERT_EQ(parsed.events["relation"].size(), 4u);
  for (const auto& ev : parsed.events["relation"]) {
    std::string phase = ev.value("phase", std::string());
    EXPECT_TRUE(kSpecPhases.count(phase))
        << "phase " << phase << " not in SPEC S3 vocabulary";
    // tracker_phase is deliberately the OLD internal vocabulary -- never
    // required to be a member of kSpecPhases, just present and non-empty.
    EXPECT_FALSE(ev.value("tracker_phase", std::string()).empty());
  }
}

// anchor fields on ev=payload/ev=target, and the anchor
// embedded in ev=relation's "relation" id string, must never be empty.
TEST_F(TraceSinkTest, AnchorNamesAreNeverEmpty) {
  std::string path = TempPath();
  TraceSink::Instance().SetPath(path);
  auto& sink = TraceSink::Instance();

  {
    nlohmann::ordered_json ev;
    ev["ev"] = "payload";
    ev["t"] = 1.0;
    ev["space"] = 2;
    ev["actor"] = 9001;
    ev["referent"] = 201;
    ev["anchor"] = "root";  // normalized by space.cc; never "".
    ev["kernel"] = 1;
    ev["R"] = 0.5;
    ev["dist"] = 1.0;
    ev["depth"] = 0;
    ev["weight"] = 1.0;
    ev["est_time"] = 1.0;
    ev["via"] = "sharedid";
    sink.Emit(ev);
  }
  {
    nlohmann::ordered_json ev;
    ev["ev"] = "target";
    ev["t"] = 1.0;
    ev["space"] = 2;
    ev["actor"] = 9001;
    ev["anchor"] = "root";
    ev["pos"] = {0.0, 0.0, 0.0};
    ev["has_z"] = false;
    ev["yaw"] = 0.0;
    ev["kernel"] = 1;
    ev["referent"] = -1;
    ev["weight"] = 1.0;
    ev["status"] = "converged";
    sink.Emit(ev);
  }
  {
    nlohmann::ordered_json ev;
    ev["ev"] = "relation";
    ev["t"] = 1.0;
    ev["space"] = 2;
    ev["actor"] = 9001;
    ev["relation"] = "spacing:root";
    ev["phase"] = "approach";
    ev["tracker_phase"] = "active";
    ev["R"] = 0.5;
    sink.Emit(ev);
  }
  sink.Disable();

  ParsedTrace parsed = ParseTraceJsonlRef(path);
  ASSERT_EQ(parsed.n_parsed, parsed.n_lines);
  for (const auto& ev : parsed.events["payload"]) {
    EXPECT_FALSE(ev.value("anchor", std::string()).empty());
  }
  for (const auto& ev : parsed.events["target"]) {
    EXPECT_FALSE(ev.value("anchor", std::string()).empty());
  }
  for (const auto& ev : parsed.events["relation"]) {
    std::string relation = ev.value("relation", std::string());
    size_t colon = relation.find(':');
    ASSERT_NE(colon, std::string::npos) << relation;
    EXPECT_FALSE(relation.substr(colon + 1).empty())
        << "empty anchor embedded in relation id " << relation;
  }
}

// port of trace_parser_ref.py's tolerance for legacy
// `SB_V2_TRACE] ...` text-format lines (skipped, not counted as
// parsed) and blank lines (skipped, not counted as lines at all).
TEST_F(TraceSinkTest, ReferenceParserSkipsLegacyLinesAndBlanks) {
  std::string path = TempPath();
  {
    std::ofstream out(path);
    out << "{\"ev\":\"tick\",\"t\":1.0,\"space\":1,\"cost_us\":10}\n";
    out << "\n";  // blank -- not counted as a line at all.
    out << "SB_V2_TRACE] legacy fprintf line, not JSON\n";
    out << "{\"ev\":\"tick\",\"t\":2.0,\"space\":1,\"cost_us\":20}\n";
  }
  ParsedTrace parsed = ParseTraceJsonlRef(path);
  EXPECT_EQ(parsed.n_lines, 3);   // blank excluded
  EXPECT_EQ(parsed.n_parsed, 2);  // legacy line excluded
  EXPECT_EQ(parsed.events["tick"].size(), 2u);
}

// -----------------------------------------------------------------------
// Per-space routing (ForSpace / SetPathForSpace / DisableForSpace).
// -----------------------------------------------------------------------

// A space_id that was never bound via SetPathForSpace/DisableForSpace falls
// back to Instance() unchanged -- the exact earlier behavior for any caller
// that never asks for per-space routing.
TEST_F(TraceSinkTest, ForSpaceFallsBackToInstanceWhenUnbound) {
  EXPECT_EQ(&TraceSink::ForSpace(1), &TraceSink::Instance());

  std::string path = TempPath();
  TraceSink::Instance().SetPath(path);
  EXPECT_TRUE(TraceSink::ForSpace(1).Active());

  nlohmann::ordered_json ev;
  ev["ev"] = "tick";
  ev["t"] = 1.0;
  ev["space"] = 1;
  ev["cost_us"] = 5;
  TraceSink::ForSpace(1).Emit(ev);
  TraceSink::Instance().Disable();

  std::ifstream in(path);
  ASSERT_TRUE(in.good());
  std::string line;
  ASSERT_TRUE(std::getline(in, line));
  nlohmann::json parsed = nlohmann::json::parse(line);
  EXPECT_EQ(parsed["ev"], "tick");
}

// Two DIFFERENT space_ids bound to two DIFFERENT paths must write to two
// genuinely SEPARATE files. Before per-space routing,
// sb_set_trace(space, path)'s `space` argument was validation-only, so two
// spaces in one process could never be separated (only a post-hoc filter on
// each line's "space" field could tell them apart).
TEST_F(TraceSinkTest, TwoSpacesBoundToDifferentPathsAreFullySeparated) {
  std::string path_a = TempPath();
  std::string path_b = ExtraPath(1);
  TraceSink::SetPathForSpace(10, path_a);
  TraceSink::SetPathForSpace(20, path_b);

  EXPECT_NE(&TraceSink::ForSpace(10), &TraceSink::ForSpace(20));
  EXPECT_TRUE(TraceSink::ForSpace(10).Active());
  EXPECT_TRUE(TraceSink::ForSpace(20).Active());
  // Neither space's binding should have touched the global default.
  EXPECT_FALSE(TraceSink::Instance().Active());

  nlohmann::ordered_json ev_a;
  ev_a["ev"] = "src";
  ev_a["t"] = 1.0;
  ev_a["space"] = 10;
  TraceSink::ForSpace(10).Emit(ev_a);

  nlohmann::ordered_json ev_b;
  ev_b["ev"] = "src";
  ev_b["t"] = 1.0;
  ev_b["space"] = 20;
  TraceSink::ForSpace(20).Emit(ev_b);

  TraceSink::DisableForSpace(10);
  TraceSink::DisableForSpace(20);

  std::ifstream in_a(path_a);
  ASSERT_TRUE(in_a.good());
  std::string line_a;
  ASSERT_TRUE(std::getline(in_a, line_a));
  nlohmann::json parsed_a = nlohmann::json::parse(line_a);
  EXPECT_EQ(parsed_a["space"], 10);
  // path_a must contain ONLY space 10's line.
  EXPECT_FALSE(std::getline(in_a, line_a));

  std::ifstream in_b(path_b);
  ASSERT_TRUE(in_b.good());
  std::string line_b;
  ASSERT_TRUE(std::getline(in_b, line_b));
  nlohmann::json parsed_b = nlohmann::json::parse(line_b);
  EXPECT_EQ(parsed_b["space"], 20);
  EXPECT_FALSE(std::getline(in_b, line_b));
}

// Two DIFFERENT space_ids bound to the identical path string must resolve to
// the exact SAME underlying sink (same mutex, same FILE*) -- not merely two
// independent handles that both happen to target the same file. This is
// what makes "bind every space to the same trace.jsonl path" (the harness's
// use, unity_twin.cc) reproduce the old "one shared trace file" behavior
// safely under concurrent writers.
TEST_F(TraceSinkTest, TwoSpacesBoundToSamePathShareOneSink) {
  std::string path = TempPath();
  TraceSink::SetPathForSpace(10, path);
  TraceSink::SetPathForSpace(20, path);

  EXPECT_EQ(&TraceSink::ForSpace(10), &TraceSink::ForSpace(20));

  nlohmann::ordered_json ev_a;
  ev_a["ev"] = "src";
  ev_a["t"] = 1.0;
  ev_a["space"] = 10;
  TraceSink::ForSpace(10).Emit(ev_a);

  nlohmann::ordered_json ev_b;
  ev_b["ev"] = "src";
  ev_b["t"] = 2.0;
  ev_b["space"] = 20;
  TraceSink::ForSpace(20).Emit(ev_b);
  TraceSink::DisableForSpace(10);

  std::ifstream in(path);
  ASSERT_TRUE(in.good());
  std::string line;
  int n = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    ++n;
  }
  EXPECT_EQ(n, 2) << "both spaces' lines must land in the one shared file";
}

// DisableForSpace(space_id) turns off tracing for THAT space_id only,
// leaving every other space_id (including one previously sharing a path with
// it) and Instance() itself untouched.
TEST_F(TraceSinkTest, DisableForSpaceOnlyAffectsThatSpace) {
  std::string path = TempPath();
  TraceSink::SetPathForSpace(10, path);
  TraceSink::SetPathForSpace(20, path);
  EXPECT_TRUE(TraceSink::ForSpace(10).Active());

  TraceSink::DisableForSpace(10);
  EXPECT_FALSE(TraceSink::ForSpace(10).Active());
  // Space 20 shared the same path/sink as 10 before 10 was disabled, but
  // DisableForSpace(10) gives space 10 its OWN private sink -- 20's sink
  // (still the original shared one) must remain untouched.
  EXPECT_TRUE(TraceSink::ForSpace(20).Active());

  TraceSink::DisableForSpace(20);
}

}  // namespace
}  // namespace spatialblend
