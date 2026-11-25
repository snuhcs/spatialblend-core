// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <exception>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "spatialblend/sim/harness/geometry_component_probe.h"

namespace {

constexpr std::streamoff kMaxInputBytes = 8 * 1024 * 1024;
constexpr int kMaxJsonDepth = 16;

int Fail(const std::string& error) {
  std::cerr << "geometry_component_probe: " << error << "\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return Fail("expected exactly one JSON input file argument");
  std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
  if (!input) return Fail("could not open input file");
  const std::streamoff size = input.tellg();
  if (size < 0 || size > kMaxInputBytes) return Fail("input exceeds 8 MiB cap");
  input.seekg(0);
  std::string text(static_cast<size_t>(size), '\0');
  if (!text.empty() && !input.read(&text[0], size))
    return Fail("could not read input file");
  if (input.peek() != std::char_traits<char>::eof()) {
    return Fail("input changed or exceeds the declared 8 MiB snapshot");
  }
  try {
    std::vector<std::set<std::string>> object_keys;
    const nlohmann::json::parser_callback_t depth_limit =
        [&object_keys](int depth, nlohmann::json::parse_event_t event,
                       nlohmann::json& parsed) {
          if ((event == nlohmann::json::parse_event_t::object_start ||
               event == nlohmann::json::parse_event_t::array_start) &&
              depth > kMaxJsonDepth) {
            throw std::runtime_error("JSON nesting exceeds depth cap");
          }
          if (event == nlohmann::json::parse_event_t::object_start) {
            object_keys.emplace_back();
          } else if (event == nlohmann::json::parse_event_t::key) {
            if (object_keys.empty() ||
                !object_keys.back().insert(parsed.get<std::string>()).second) {
              throw std::runtime_error("duplicate JSON object key");
            }
          } else if (event == nlohmann::json::parse_event_t::object_end) {
            object_keys.pop_back();
          }
          return true;
        };
    const nlohmann::json parsed = nlohmann::json::parse(text, depth_limit);
    auto result = spatialblend::harness::RunGeometryComponentProbe(parsed);
    if (!result.ok()) return Fail(std::string(result.status().message()));
    std::cout << result->dump() << "\n";
    return 0;
  } catch (const nlohmann::json::exception& error) {
    return Fail(std::string("invalid JSON: ") + error.what());
  } catch (const std::runtime_error& error) {
    return Fail(std::string("invalid JSON: ") + error.what());
  } catch (const std::exception& error) {
    return Fail(std::string("unexpected error: ") + error.what());
  }
}
