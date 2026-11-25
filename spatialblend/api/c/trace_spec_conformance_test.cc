// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Conformance: spatialblend/api/c/trace.format.md is the normative trace
// format. This test parses the spec's per-event field tables and checks every
// line of a real traced two-space run against them: the event kind must be
// specified, the field set must be exactly the specified one, and every
// value must have the specified JSON type. A field emitted without a spec row
// (or specified but not emitted) fails here.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "spatialblend/api/c/c_test_embodied.h"
#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/api/c/spatialblend_receiver_input.h"
#include "spatialblend/api/c/spatialblend_receiver_reevaluation.h"

namespace {

struct FieldSpec {
  std::string type;
};
using KindSpec = std::map<std::string, FieldSpec>;

std::string Trim(const std::string& s) {
  const size_t b = s.find_first_not_of(" \t");
  if (b == std::string::npos) return "";
  const size_t e = s.find_last_not_of(" \t");
  return s.substr(b, e - b + 1);
}

std::string SpecPath() {
  const char* srcdir = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  std::vector<std::string> candidates;
  if (srcdir) {
    if (workspace)
      candidates.push_back(std::string(srcdir) + "/" + workspace +
                           "/spatialblend/api/c/trace.format.md");
    candidates.push_back(std::string(srcdir) +
                         "/_main/spatialblend/api/c/trace.format.md");
  }
  candidates.push_back("spatialblend/api/c/trace.format.md");
  for (const auto& c : candidates) {
    std::ifstream f(c);
    if (f.good()) return c;
  }
  return "";
}

struct Spec {
  std::map<std::string, KindSpec> kinds;
  // "kind.field" -> allowed string values (the Vocabularies table).
  std::map<std::string, std::set<std::string>> vocab;
};

std::vector<std::string> Cells(const std::string& line) {
  // Escaped pipes inside a cell ("number\|null") are not separators.
  std::string row;
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '\\' && i + 1 < line.size() && line[i + 1] == '|') {
      row += '\x01';
      ++i;
    } else {
      row += line[i];
    }
  }
  std::vector<std::string> cells;
  std::stringstream ss(row);
  std::string cell;
  while (std::getline(ss, cell, '|')) {
    for (char& c : cell)
      if (c == '\x01') c = '|';
    cells.push_back(Trim(cell));
  }
  return cells;  // cells[0] is the text before the leading '|'
}

std::string Unquote(const std::string& cell) {
  if (cell.size() >= 2 && cell.front() == '`' && cell.back() == '`')
    return cell.substr(1, cell.size() - 2);
  return "";
}

// Parses "### `ev=<kind>`" field tables and the "## Vocabularies" table.
Spec ParseSpec(const std::string& path) {
  Spec out;
  std::ifstream in(path);
  std::string line, kind;
  bool in_vocab = false;
  while (std::getline(in, line)) {
    // A CRLF checkout (core.autocrlf=true, Windows CI) must parse the same.
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.rfind("### `ev=", 0) == 0) {
      const size_t b = line.find("ev=") + 3;
      const size_t e = line.find('`', b);
      kind = line.substr(b, e - b);
      out.kinds[kind];
      in_vocab = false;
      continue;
    }
    if (line.rfind("## ", 0) == 0) {
      kind.clear();
      in_vocab = line == "## Vocabularies";
      continue;
    }
    if (line.rfind("| `", 0) != 0) continue;
    const auto cells = Cells(line);
    if (cells.size() < 3) continue;
    const std::string key = Unquote(cells[1]);
    if (key.empty()) continue;
    if (in_vocab) {
      std::stringstream values(cells[2]);
      std::string v;
      while (std::getline(values, v, ',')) {
        const std::string word = Unquote(Trim(v));
        if (!word.empty()) out.vocab[key].insert(word);
      }
    } else if (!kind.empty()) {
      out.kinds[kind][key] = {cells[2]};
    }
  }
  return out;
}

// FNV-1a 64 over the canonical (sorted) field and vocabulary tables.
uint64_t SpecHash(const Spec& spec) {
  std::string canon;
  for (const auto& [kind, fields] : spec.kinds)
    for (const auto& [field, fs] : fields)
      canon += kind + "." + field + ":" + fs.type + ";";
  for (const auto& [key, words] : spec.vocab) {
    canon += key + "=";
    for (const auto& w : words) canon += w + ",";
    canon += ";";
  }
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : canon) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

// Guard: the spec's field/vocabulary tables are pinned per trace format
// version. Changing an emitted field (and therefore, via the conformance
// check, the spec) without bumping kTraceFormatVersion fails here; after a
// bump, pin the new hash printed by the failure message.
const std::map<uint32_t, uint64_t> kPinnedSpecHash = {
    {2u, 0x8015fa1d44c583abull},
    {3u, 0x32f974f2ee1e01f6ull},  // + ev=mediator
    {4u, 0x3eec570a0f4ee1c1ull},  // + scope.reason beyond_scope_radius
    {5u, 0xb3a110a16a44da86ull},  // target reach diagnostic
    {6u, 0x2379736c7b5e1d9eull},  // + ev=qualify
    {7u, 0xdaacf73ae9f187e6ull},  // target objective_before/after/...
    {8u, 0xcec9a4ee3d37107aull},  // + ev=proposal, - ev=mediator
    {9u, 0x3b3b019e08fdb529ull},  // ABI 4: - ev=qualify, + proposal.anchors, - proposal.budget_s; scope.reason gate reasons
    {10u, 0x327e07ed8d99bf94ull},  // + scope.reason inside_transfer_leaf
};

bool MatchesOne(const nlohmann::json& v, const std::string& t) {
  if (t == "string") return v.is_string();
  if (t == "int") return v.is_number_integer();
  if (t == "number") return v.is_number();
  if (t == "bool") return v.is_boolean();
  if (t == "null") return v.is_null();
  if (t == "array") return v.is_array();
  if (t == "number[2]" || t == "number[3]") {
    const size_t n = t == "number[2]" ? 2 : 3;
    if (!v.is_array() || v.size() != n) return false;
    for (const auto& x : v)
      if (!x.is_number()) return false;
    return true;
  }
  return false;
}

bool MatchesType(const nlohmann::json& v, const std::string& type) {
  std::stringstream ss(type);
  std::string alt;
  while (std::getline(ss, alt, '|'))
    if (MatchesOne(v, Trim(alt))) return true;
  return false;
}

sb_element_t g_received = 0;
void CaptureAdded(sb_space_t, sb_element_t local, int64_t, const char*,
                  const void*, int64_t, void*) {
  g_received = local;
}

void Room(sb_space_t space, sb_element_t* room, sb_element_t* desk) {
  const double poly[] = {0, 0, 10, 0, 10, 10, 0, 10};
  *room = sb_add_element(space, 0);
  sb_set_element_shared_id(space, *room, 101);
  sb_set_element_polygon(space, *room, SB_GEOMETRY_NAVIGABLE, poly, 4);
  *desk = sb_add_element(space, *room);
  sb_set_element_shared_id(space, *desk, 103);
  sb_set_element_box(space, *desk, SB_GEOMETRY_OBSTACLE, 4, 4, 0, 6, 6, 1);
}

const uint8_t* Bytes(const char* text) {
  return reinterpret_cast<const uint8_t*>(text);
}

sb_receiver_anchor_input_t InputAnchor(const char* name, double x, double y,
                                       double z) {
  sb_receiver_anchor_input_t value{};
  value.name_utf8 = Bytes(name);
  value.name_bytes = static_cast<uint32_t>(std::strlen(name));
  value.available = 1;
  for (int i = 0; i < 4; ++i) value.world_pose_row_major[i * 4 + i] = 1;
  value.world_pose_row_major[3] = x;
  value.world_pose_row_major[7] = y;
  value.world_pose_row_major[11] = z;
  return value;
}

// Source (spacing, facing, tagged affordance hand, a facing bind whose
// scope id is checked and ignored)
// -> linked receiver (root + gated hand target) and a receiver-input receiver
// with a rich activation and a reevaluation. Every space traces to `path`.
void RunFixture(const std::string& path) {
  const sb_session_t session = sb_create_session();
  const sb_space_t src = sb_add_space(session);
  const sb_space_t dst = sb_add_space(session);
  const sb_space_t input = sb_add_space(session);
  sb_set_trace(src, path.c_str());
  sb_set_trace(dst, path.c_str());
  sb_set_trace(input, path.c_str());

  sb_element_t rooms[3], desks[3];
  const sb_space_t spaces[3] = {src, dst, input};
  for (int i = 0; i < 3; ++i) {
    Room(spaces[i], &rooms[i], &desks[i]);
    // A cup with a 0.4 m top surface at z = 0.8 around (3.2, 5.0).
    // An affordance target needs support geometry and exists while
    // the hand is engaged on it (the walk passes the hand over the top).
    const sb_element_t cup = sb_add_element(spaces[i], rooms[i]);
    sb_set_element_shared_id(spaces[i], cup, 104);
    sb_set_element_tag(spaces[i], cup, "Cup");
    const double top[] = {3.0, 4.8, 0.8, 3.4, 4.8, 0.8,
                          3.4, 5.2, 0.8, 3.0, 5.2, 0.8};
    sb_set_element_surface(spaces[i], cup, top, 4, SB_GEOMETRY_SURFACE);
  }
  const sb_element_t src_actor_elem = sb_add_element(src, rooms[0]);
  sb_set_element_shared_id(src, src_actor_elem, 102);
  const sb_element_t dst_actor_elem = sb_add_element(dst, rooms[1]);
  sb_set_element_shared_id(dst, dst_actor_elem, 202);

  const sb_actor_t src_actor = sb_create_actor(src);
  sb_add_anchor(src, src_actor, "root", 0, 0, 0, nullptr);
  sb_add_anchor_with_position_dimension(src, src_actor, "hand", 0, 0, 0,
                                        nullptr, 3);
  sb_set_actor_element(src, src_actor, src_actor_elem);
  sb_bind(src, src_actor, "root", SB_KERNEL_SPACING);
  sb_bind(src, src_actor, "root", SB_KERNEL_FACING);
  sb_bind_affordance(src, src_actor, "hand", "Cup");
  // A seat anchor on the same support.
  sb_add_anchor_with_position_dimension(src, src_actor, "pelvis", 0, 0, 0,
                                        nullptr, 3);
  sb_bind_affordance(src, src_actor, "pelvis",
                 "Cup");
  sb_bind(src, src_actor, "root", SB_KERNEL_FACING);

  const sb_actor_t dst_actor = sb_create_actor(dst);
  sb_add_anchor(dst, dst_actor, "root", 0, 0, 0, nullptr);
  sb_add_anchor_with_position_dimension(dst, dst_actor, "hand", 0, 0, 0,
                                        nullptr, 3);
  sb_add_anchor_with_position_dimension(dst, dst_actor, "pelvis", 0, 0, 0,
                                        nullptr, 3);
  sb_set_actor_element(dst, dst_actor, dst_actor_elem);

  // Receiver-input receiver: the source actor resolves by shared id 102.
  const sb_element_t input_actor_elem = sb_add_element(input, rooms[2]);
  sb_set_element_shared_id(input, input_actor_elem, 102);
  const sb_actor_t input_actor = sb_create_actor(input);
  sb_add_anchor(input, input_actor, "root", 0, 0, 0, nullptr);
  sb_add_anchor(input, input_actor, "head", 0, 0, 1.7, nullptr);
  sb_set_actor_element(input, input_actor, input_actor_elem);
  uint64_t epoch = 0;
  EXPECT_EQ(sb_begin_receiver_input(input, input_actor, 1, Bytes("root"), 4,
                                    7, 9, &epoch),
            SB_RICH_RESULT_OK);
  const sb_rich_realization_token_t token =
      sb_activate_rich_realization(input, input_actor, 1);
  EXPECT_NE(token, 0u);
  sb_receiver_input_feature_t feature{};
  sb_get_receiver_input_feature(&feature, sizeof(feature));
  std::vector<sb_receiver_anchor_input_t> anchors = {
      InputAnchor("root", 2, 3, 0), InputAnchor("head", 4, 5, 1.7)};
  sb_receiver_input_frame_t frame{};
  frame.struct_size = sizeof(frame);
  frame.feature_version = 1;
  frame.anchor_count = static_cast<uint32_t>(anchors.size());
  frame.anchor_stride = sizeof(sb_receiver_anchor_input_t);
  frame.epoch = epoch;
  frame.sequence = 1;
  frame.monotonic_clock_id = feature.monotonic_clock_id;
  frame.monotonic_clock_epoch = feature.monotonic_clock_epoch;
  frame.anchors = anchors.data();
  frame.observed_at_ns = static_cast<int64_t>(sb_read_rich_monotonic_ns());
  frame.valid_until_ns =
      frame.observed_at_ns + static_cast<int64_t>(feature.max_age_ns);
  EXPECT_EQ(sb_submit_receiver_input(input, input_actor, &frame, sizeof(frame)),
            SB_RICH_RESULT_OK);

  // ev=share: the linked receiver maps the source actor element to its own.
  sb_register_on_element_added(dst, CaptureAdded, nullptr);
  sb_share_element(src, src_actor_elem, nullptr, 0);
  EXPECT_NE(g_received, 0);
  sb_resolve_shared_element(dst, g_received, dst_actor_elem);

  // ev=proposal: the driver (this fixture) publishes the linked receiver's
  // embodied pose, and core proposes for it.
  uint64_t dst_sequence = 0;
  const auto publish_dst = [&](double x, double y, double yaw) {
    sb_anchor_pose_t pose{};
    pose.struct_size = sizeof(pose);
    pose.x = x;
    pose.y = y;
    pose.orientation.struct_size = sizeof(sb_orientation_t);
    pose.orientation.yaw = yaw;
    pose.epoch = 1;
    EXPECT_EQ(sb_publish_anchor_pose(dst, dst_actor, 0, &pose, 0.0,
                                     ++dst_sequence),
              0);
  };
  publish_dst(1.0, 5.0, 0.0);

  sb_set_min_move(src, 0.0);
  sb_set_min_yaw(src, 0.0);
  for (int i = 0; i < 8; ++i) {
    const double x = 1.0 + 0.3 * i;
    sb_set_anchor_transform(src, src_actor, "hand", x + 0.2, 5.0, 0.8, 0.0);
    sb_set_anchor_transform(src, src_actor, "pelvis", x + 0.2, 5.0, 0.95,
                            0.0);
    sb_update_actor_state(src, src_actor, x, 5.0, 0.0);
    const auto proposal = spatialblend_c_test::ReadProposals(dst, dst_actor);
    if (proposal.rc == 0 && (proposal.status == SB_PROPOSAL_EXACT ||
                             proposal.status == SB_PROPOSAL_INEXACT))
      publish_dst(proposal.x, proposal.y, proposal.yaw);
  }
  // Reevaluation against a new receiver-input frame (writes no trace).
  frame.sequence = 2;
  anchors[1] = InputAnchor("head", 6, 7, 1.8);
  frame.observed_at_ns = static_cast<int64_t>(sb_read_rich_monotonic_ns());
  frame.valid_until_ns =
      frame.observed_at_ns + static_cast<int64_t>(feature.max_age_ns);
  EXPECT_EQ(sb_submit_receiver_input(input, input_actor, &frame, sizeof(frame)),
            SB_RICH_RESULT_OK);
  EXPECT_EQ(sb_reevaluate_receiver_realization(token, 1, epoch, 2),
            SB_RICH_RESULT_OK);

  sb_set_trace(src, nullptr);
  sb_set_trace(dst, nullptr);
  sb_set_trace(input, nullptr);
  sb_retire_rich_realization(token);
  sb_release_rich_realization(token);
  sb_destroy_session(session);
}

}  // namespace

// A CRLF checkout of the spec (core.autocrlf=true, Windows CI) parses to the
// same tables and hash as the LF file.
TEST(TraceSpecConformanceTest, X3_CrlfSpecParsesIdentically) {
  const std::string spec_path = SpecPath();
  ASSERT_FALSE(spec_path.empty()) << "trace.format.md not in runfiles";
  std::ifstream in(spec_path);
  const char* tmp = std::getenv("TEST_TMPDIR");
  const std::string crlf_path =
      std::string(tmp ? tmp : "/tmp") + "/trace.format.crlf.md";
  {
    std::ofstream out(crlf_path, std::ios::binary);
    std::string line;
    while (std::getline(in, line)) out << line << "\r\n";
  }
  const Spec lf = ParseSpec(spec_path);
  const Spec crlf = ParseSpec(crlf_path);
  EXPECT_EQ(SpecHash(lf), SpecHash(crlf));
  EXPECT_EQ(lf.kinds.size(), crlf.kinds.size());
  EXPECT_EQ(lf.vocab.size(), crlf.vocab.size());
  EXPECT_FALSE(crlf.vocab.empty());
  std::remove(crlf_path.c_str());
}

TEST(TraceSpecConformanceTest, X3_SpecTablesArePinnedToTheFormatVersion) {
  const std::string spec_path = SpecPath();
  ASSERT_FALSE(spec_path.empty()) << "trace.format.md not in runfiles";
  const Spec spec = ParseSpec(spec_path);
  const uint32_t version = sb_get_trace_format_version();
  const uint64_t hash = SpecHash(spec);
  auto it = kPinnedSpecHash.find(version);
  ASSERT_NE(it, kPinnedSpecHash.end())
      << "no pinned spec hash for trace format " << version;
  char actual[32];
  std::snprintf(actual, sizeof(actual), "0x%016llx",
                static_cast<unsigned long long>(hash));
  EXPECT_EQ(it->second, hash)
      << "trace.format.md's field/vocabulary tables changed but the trace "
         "format version is still "
      << version << ". Bump kTraceFormatVersion (common/trace.h), record the "
      << "change in the spec's history, and pin " << actual;
}

TEST(TraceSpecConformanceTest, X3_EveryEmittedLineMatchesTheSpec) {
  const std::string spec_path = SpecPath();
  ASSERT_FALSE(spec_path.empty()) << "trace.format.md not in runfiles";
  const Spec spec = ParseSpec(spec_path);
  ASSERT_GE(spec.kinds.size(), 1u);
  for (const auto& [kind, fields] : spec.kinds)
    EXPECT_FALSE(fields.empty()) << "spec has no fields for ev=" << kind;
  ASSERT_TRUE(spec.vocab.count("scope.reason"));

  const char* tmp = std::getenv("TEST_TMPDIR");
  const std::string path =
      std::string(tmp ? tmp : "/tmp") + "/conformance.trace.jsonl";
  std::remove(path.c_str());
  RunFixture(path);

  std::ifstream in(path);
  std::string line;
  std::map<std::string, int> seen;
  std::set<std::string> seen_reasons;
  int n = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    ++n;
    nlohmann::json ev;
    ASSERT_NO_THROW(ev = nlohmann::json::parse(line)) << line;
    ASSERT_TRUE(ev.is_object()) << line;
    ASSERT_TRUE(ev.contains("ev") && ev["ev"].is_string()) << line;
    const std::string kind = ev["ev"];
    ++seen[kind];
    ASSERT_TRUE(spec.kinds.count(kind))
        << "unspecified ev=" << kind << ": " << line;
    const KindSpec& fields = spec.kinds.at(kind);
    std::set<std::string> actual, expected;
    for (auto it = ev.begin(); it != ev.end(); ++it) actual.insert(it.key());
    for (const auto& [f, _] : fields) expected.insert(f);
    EXPECT_EQ(actual, expected) << "field set of ev=" << kind << ": " << line;
    for (const auto& [f, fs] : fields) {
      if (!ev.contains(f)) continue;
      EXPECT_TRUE(MatchesType(ev[f], fs.type))
          << "ev=" << kind << " field " << f << " is not " << fs.type << ": "
          << line;
      const auto vocab = spec.vocab.find(kind + "." + f);
      if (vocab != spec.vocab.end() && ev[f].is_string()) {
        EXPECT_TRUE(vocab->second.count(ev[f].get<std::string>()))
            << "ev=" << kind << " " << f << " value outside the spec "
            << "vocabulary: " << line;
      }
    }
    if (kind == "meta" || kind == "policy") {
      EXPECT_EQ(ev["format"], static_cast<int>(sb_get_trace_format_version()));
    }
    if (kind == "target" && ev["anchor"] == "hand") ++seen["target:hand"];
    if (kind == "channel" && ev["kernel"] == 2) ++seen["channel:affordance"];
    if (kind == "scope" && ev["explicit"] == true) ++seen["scope:explicit"];
    if (kind == "scope") {
      for (const auto& pair : ev["excluded"]) {
        ASSERT_TRUE(pair.is_array() && pair.size() == 2) << line;
        EXPECT_TRUE(pair[0].is_number_integer()) << line;
        ASSERT_TRUE(pair[1].is_string()) << line;
        const std::string reason = pair[1];
        seen_reasons.insert(reason);
        EXPECT_TRUE(spec.vocab.at("scope.reason").count(reason))
            << "scope reason outside the spec vocabulary: " << line;
      }
      for (const auto& id : ev["included"])
        EXPECT_TRUE(id.is_number_integer()) << line;
    }
  }
  EXPECT_GT(n, 0);
  // Every kind the spec declares must actually be exercised.
  for (const auto& [kind, _] : spec.kinds)
    EXPECT_GT(seen[kind], 0) << "the fixture never emitted ev=" << kind;
  // Family scope. The app-supplied root is ignored, so a scope record is
  // never explicit. The fixture still excludes by tag and by the actor's
  // own element.
  for (const char* r : {"tag", "actor_element"})
    EXPECT_TRUE(seen_reasons.count(r)) << "fixture never produced " << r;
  EXPECT_FALSE(seen_reasons.count("explicit_scope"));
  EXPECT_GT(seen["target:hand"], 0) << "no hand (anchor) target traced";
  EXPECT_GT(seen["channel:affordance"], 0);
  EXPECT_EQ(seen["scope:explicit"], 0);
  // One meta line per sb_set_trace(path) call.
  EXPECT_EQ(seen["meta"], 3);
  std::remove(path.c_str());
}
