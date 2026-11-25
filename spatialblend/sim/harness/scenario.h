// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_HARNESS_SCENARIO_H_
#define SPATIALBLEND_HARNESS_SCENARIO_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"
#include "spatialblend/base/common/types.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/space/space.h"

// Scenario JSON loader + Space builder. Reads *.scenario.json
// scenario files (testdata/*.scenario.json) and builds
// spatialblend::Space objects (source/target) via LocalContext, mirroring the
// element/actor/binding conventions already used by
// spatialblend/net/cross_space_test.cc's referent_geometry / spacing_nested
// test helpers (AddRoom/AddSofa/AddNavBox/AddAvatarActor).
//
// Coordinate convention: a scenario element's
// "transform": {"pos": [...], "yaw": ...} is WORLD-ABSOLUTE within its space.
// Footprint/boundary polygons are always constructed directly in world-
// absolute coordinates (never composed with a SpatialDefinition's own
// transform, which stays Identity for every construction path in this code
// base -- see NavigableSpatialDefinition::local_transform_ and
// SpatialDefinition::GetTransform()). The Element's own local_transform is
// still set (derived from the parent chain so nested nesting is handled
// correctly for output/orientation purposes), but is NOT relied upon for
// containment.
namespace spatialblend {
namespace harness {

struct Vec2 {
  double x = 0.0;
  double y = 0.0;
};

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct ScenarioTransform {
  Vec3 pos;
  double yaw = 0.0;
};

// footprint.kind in {"box","sphere","capsule","polygon"}.
struct ScenarioFootprint {
  std::string kind;
  Vec3 size;                 // box: full size (sx,sy,sz)
  double radius = 0.0;       // sphere/capsule
  double height = 0.0;       // capsule: total height incl. caps
  int axis = 2;              // capsule axis: 0=X,1=Y,2=Z (default up)
  std::vector<Vec2> points;  // polygon: local-frame offsets from element pos
};

struct ScenarioSubRegion {
  std::string name;
  std::string kind;
  std::vector<Vec2> points;
  double height = 0.0;
};

struct ScenarioElement {
  std::string tag;
  int64_t shared_id = 0;
  std::string parent;  // "room" or another element's tag
  ScenarioTransform transform;
  bool has_footprint = false;
  ScenarioFootprint footprint;
  bool propagate = false;
  std::vector<ScenarioSubRegion> sub_regions;
};

struct ScenarioSpace {
  std::string id;
  std::string role;  // "source" | "target"
  std::vector<Vec2> walkable;
  std::vector<ScenarioElement> elements;
};

struct ScenarioAnchor {
  std::string name;
  Vec3 offset;
};

struct ScenarioBinding {
  std::string anchor;
  std::string kernel;  // "spacing" | "facing" | "affordance_pose"
  std::string tag_filter;
  bool has_tag_filter = false;
};

struct ScenarioActor {
  std::string id;
  std::string space;
  int64_t shared_id = 0;
  std::vector<ScenarioAnchor> anchors;
  std::vector<ScenarioBinding> bindings;
  // Non-standard extension (schema declares "additionalProperties: true" on
  // actor objects): optional initial 2D pose "start": [x, y, yaw]. Defaults
  // to (0,0,0) when absent. Used by the unity_twin harness to give every
  // actor (not just the one driven by top-level "behavior") a defined
  // starting pose -- needed e.g. for a second, unscripted peer actor in a
  // multi-actor scenario (see harness_test.cc's avatar-chatter regression).
  double start_x = 0.0;
  double start_y = 0.0;
  double start_yaw = 0.0;
};

struct ScenarioScriptOp {
  std::string op;  // approach|sit|stand|wait|look_at|release|move|wander
  // Raw json for the op, so callers (the unity_twin harness) can read op-specific fields
  // (tag/distance/duration/jitter_sigma/target/...) without this loader
  // having to enumerate every op's schema.
  nlohmann::json raw;
};

struct ScenarioBehavior {
  std::string kind;  // script|recording|generated
  std::vector<ScenarioScriptOp> script;
  nlohmann::json recording;  // raw passthrough
};

struct ScenarioNetwork {
  std::string mode;
  double latency_ms = 0.0;
  double jitter_ms = 0.0;
  double drop_rate = 0.0;
  bool reorder = false;
};

struct ScenarioIdentityExpect {
  int64_t shared_id = 0;
  std::string resolution;  // link_to_existing | add_as_new
};

struct ScenarioIdentity {
  std::string share_from;
  std::vector<ScenarioIdentityExpect> expect;
  bool mismatched_ids = false;
};

struct ScenarioCheck {
  std::string check;
  double eps = 0.0;
  std::vector<std::string> engagements;
  std::vector<std::string> order;
};

// Non-standard extension (the schema declares "metamorphic" as an opaque
// object). Contract chosen here:
//   "metamorphic": {
//     "derive_target_from_source": {
//       "source": "<space id>", "target": "<space id>",
//       "translate": [dx,dy,dz],      // optional, default 0
//       "rotate_yaw": <radians>,      // optional, default 0, about rotate_center
//       "rotate_center": [cx,cy],     // optional, default [0,0]
//       "scale": <factor>,            // optional, default 1
//       "scale_center": [cx,cy],      // optional, default [0,0]
//       "mirror_axis": "x"|"y"        // optional, mirrors across that axis
//                                     // through the origin before the above
//     }
//   }
// Order of composition (see ApplyMetamorphicTransform in scenario.cc):
// mirror -> scale (about scale_center) -> rotate (about rotate_center) ->
// translate. The transform is applied to the source space's walkable
// polygon and to every element's world position; yaw is offset by
// rotate_yaw (+ mirrored sign flip). The declared target space's own
// "walkable"/"elements" JSON content is ignored when derive_target_from_source
// names it as the target -- it is fully DERIVED from the source instead.
struct MetamorphicTransform {
  bool has_translate = false;
  Vec3 translate;
  bool has_rotate = false;
  double rotate_yaw = 0.0;
  Vec2 rotate_center;
  bool has_scale = false;
  double scale = 1.0;
  Vec2 scale_center;
  bool has_mirror = false;
  std::string mirror_axis;  // "x" or "y"
};

struct ScenarioMetamorphic {
  bool derive_target_from_source = false;
  std::string source_space_id;
  std::string target_space_id;
  MetamorphicTransform transform;
};

struct ScenarioModel {
  std::string schema_version;
  std::string id;
  std::string tier;
  std::string regime;
  std::vector<std::string> verify_by;
  std::vector<ScenarioSpace> spaces;
  bool has_identity = false;
  ScenarioIdentity identity;
  std::vector<ScenarioActor> actors;
  ScenarioBehavior behavior;
  bool has_network = false;
  ScenarioNetwork network;
  nlohmann::json core;  // raw passthrough (unity_twin applies it via the runtime config setters)
  nlohmann::json app;
  std::vector<ScenarioCheck> expect;
  bool has_metamorphic = false;
  ScenarioMetamorphic metamorphic;

  const ScenarioSpace* FindSpace(const std::string& space_id) const;
};

// Parses a *.scenario.json file into a ScenarioModel. Returns an error
// Status (never throws / crashes) on I/O failure, malformed JSON, or a
// missing required field.
absl::StatusOr<ScenarioModel> LoadScenario(const std::string& path);

// Same as LoadScenario but parses from an already-loaded JSON string
// (useful for tests that construct a scenario inline).
absl::StatusOr<ScenarioModel> ParseScenario(const std::string& json_text);

// Per-space bookkeeping produced by BuildSpaces, needed by callers (the
// unity_twin harness) to look up element/actor ids by scenario-level name.
struct BuiltSpace {
  std::string scenario_space_id;
  Space* space = nullptr;
  element_id_t room_element_id = kInvalidElementId;
  std::unordered_map<std::string, element_id_t> element_id_by_tag;
  std::unordered_map<int64_t, element_id_t> element_id_by_shared_id;
  // Next element id available for BuildSpaces' own bookkeeping (e.g. the
  // per-actor grounding leaf element created for each ScenarioActor). Not
  // meaningful to callers outside scenario.cc.
  element_id_t next_free_element_id = kInvalidElementId;
};

struct BuiltActor {
  std::string scenario_actor_id;
  std::string scenario_space_id;
  spatialblend::entity_id_t actor_id = spatialblend::kInvalidEntityId;
};

struct BuildResult {
  // Keyed by the scenario's space "id" field (e.g. "A", "B").
  std::unordered_map<std::string, BuiltSpace> spaces;
  std::vector<BuiltActor> actors;

  const BuiltSpace* FindSpace(const std::string& scenario_space_id) const;
  const BuiltActor* FindActor(const std::string& scenario_actor_id) const;
};

// Builds one spatialblend::Space per scenario space (via ctx.AddSpace()),
// populates the element tree (room root + child elements with
// footprints/sub-regions), and creates actors with their anchors and
// bindings. Does NOT touch networking, sharing, or the C API -- that is the unity_twin
// harness's responsibility. Returns an error Status (no crash) on inconsistent input
// (e.g. an element whose declared parent tag never resolves).
absl::StatusOr<BuildResult> BuildSpaces(const ScenarioModel& model,
                                        LocalContext& ctx);

}  // namespace harness
}  // namespace spatialblend

#endif  // SPATIALBLEND_HARNESS_SCENARIO_H_
