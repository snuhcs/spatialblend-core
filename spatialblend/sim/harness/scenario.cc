// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/sim/harness/scenario.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include <Eigen/Dense>

#include "absl/status/status.h"
#include "spatialblend/model/actor/subject_anchor.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/envelope.h"
#include "spatialblend/model/element/geometry_role.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/sub_region.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/relation/relation_kernel.h"

namespace spatialblend {
namespace harness {

namespace {

using nlohmann::json;

absl::Status RequireField(const json& obj, const std::string& field,
                          const std::string& context) {
  if (!obj.contains(field)) {
    return absl::InvalidArgumentError("missing required field '" + field +
                                      "' in " + context);
  }
  return absl::OkStatus();
}

Vec2 ParseVec2(const json& arr) {
  Vec2 v;
  if (arr.is_array() && arr.size() >= 2) {
    v.x = arr[0].get<double>();
    v.y = arr[1].get<double>();
  }
  return v;
}

Vec3 ParseVec3(const json& arr) {
  Vec3 v;
  if (arr.is_array() && arr.size() >= 3) {
    v.x = arr[0].get<double>();
    v.y = arr[1].get<double>();
    v.z = arr[2].get<double>();
  } else if (arr.is_array() && arr.size() == 2) {
    v.x = arr[0].get<double>();
    v.y = arr[1].get<double>();
  }
  return v;
}

absl::StatusOr<std::vector<Vec2>> ParsePolygon(const json& arr,
                                               const std::string& context) {
  if (!arr.is_array() || arr.size() < 3) {
    return absl::InvalidArgumentError(context +
                                      " must be a polygon with >= 3 points");
  }
  std::vector<Vec2> pts;
  pts.reserve(arr.size());
  for (const auto& p : arr) {
    pts.push_back(ParseVec2(p));
  }
  return pts;
}

absl::StatusOr<ScenarioTransform> ParseTransform(const json& obj,
                                                 const std::string& context) {
  auto st = RequireField(obj, "pos", context);
  if (!st.ok()) return st;
  ScenarioTransform t;
  t.pos = ParseVec3(obj.at("pos"));
  if (obj.contains("yaw")) t.yaw = obj.at("yaw").get<double>();
  return t;
}

absl::StatusOr<ScenarioFootprint> ParseFootprint(const json& obj,
                                                 const std::string& context) {
  auto st = RequireField(obj, "kind", context);
  if (!st.ok()) return st;
  ScenarioFootprint fp;
  fp.kind = obj.at("kind").get<std::string>();
  if (fp.kind == "box") {
    auto s = RequireField(obj, "size", context + ".footprint(box)");
    if (!s.ok()) return s;
    fp.size = ParseVec3(obj.at("size"));
  } else if (fp.kind == "sphere") {
    if (obj.contains("radius")) {
      fp.radius = obj.at("radius").get<double>();
    } else if (obj.contains("size")) {
      // Allow size.x as a diameter-ish fallback if radius is omitted.
      fp.radius = ParseVec3(obj.at("size")).x * 0.5;
    } else {
      return absl::InvalidArgumentError(context +
                                        ".footprint(sphere) needs 'radius'");
    }
  } else if (fp.kind == "capsule") {
    if (!obj.contains("radius")) {
      return absl::InvalidArgumentError(context +
                                        ".footprint(capsule) needs 'radius'");
    }
    fp.radius = obj.at("radius").get<double>();
    Vec3 size = obj.contains("size") ? ParseVec3(obj.at("size")) : Vec3{};
    fp.height = obj.contains("height") ? obj.at("height").get<double>()
                                       : (size.z > 0.0 ? size.z : 1.0);
    fp.axis = obj.contains("axis") ? obj.at("axis").get<int>() : 2;
  } else if (fp.kind == "polygon") {
    auto p = RequireField(obj, "points", context + ".footprint(polygon)");
    if (!p.ok()) return p;
    auto pts = ParsePolygon(obj.at("points"), context + ".footprint.points");
    if (!pts.ok()) return pts.status();
    fp.points = *pts;
  } else {
    return absl::InvalidArgumentError(context + ": unknown footprint kind '" +
                                      fp.kind + "'");
  }
  return fp;
}

absl::StatusOr<ScenarioSubRegion> ParseSubRegion(const json& obj,
                                                 const std::string& context) {
  auto n = RequireField(obj, "name", context);
  if (!n.ok()) return n;
  auto k = RequireField(obj, "kind", context);
  if (!k.ok()) return k;
  ScenarioSubRegion sr;
  sr.name = obj.at("name").get<std::string>();
  sr.kind = obj.at("kind").get<std::string>();
  if (obj.contains("points")) {
    auto pts = ParsePolygon(obj.at("points"), context + ".points");
    if (!pts.ok()) return pts.status();
    sr.points = *pts;
  }
  if (obj.contains("height")) sr.height = obj.at("height").get<double>();
  return sr;
}

absl::StatusOr<ScenarioElement> ParseElement(const json& obj,
                                             const std::string& context) {
  for (const char* f : {"tag", "shared_id", "parent", "transform"}) {
    auto st = RequireField(obj, f, context + ".element");
    if (!st.ok()) return st;
  }
  ScenarioElement e;
  e.tag = obj.at("tag").get<std::string>();
  e.shared_id = obj.at("shared_id").get<int64_t>();
  e.parent = obj.at("parent").get<std::string>();
  auto t = ParseTransform(obj.at("transform"), context + ".element[" + e.tag + "]");
  if (!t.ok()) return t.status();
  e.transform = *t;
  if (obj.contains("footprint")) {
    auto fp = ParseFootprint(obj.at("footprint"),
                             context + ".element[" + e.tag + "]");
    if (!fp.ok()) return fp.status();
    e.footprint = *fp;
    e.has_footprint = true;
  }
  if (obj.contains("propagate")) e.propagate = obj.at("propagate").get<bool>();
  if (obj.contains("sub_regions")) {
    for (const auto& sr : obj.at("sub_regions")) {
      auto parsed = ParseSubRegion(sr, context + ".element[" + e.tag + "]");
      if (!parsed.ok()) return parsed.status();
      e.sub_regions.push_back(*parsed);
    }
  }
  return e;
}

absl::StatusOr<ScenarioSpace> ParseSpace(const json& obj) {
  for (const char* f : {"id", "role", "walkable", "elements"}) {
    auto st = RequireField(obj, f, "space");
    if (!st.ok()) return st;
  }
  ScenarioSpace s;
  s.id = obj.at("id").get<std::string>();
  s.role = obj.at("role").get<std::string>();
  auto poly = ParsePolygon(obj.at("walkable"), "space[" + s.id + "].walkable");
  if (!poly.ok()) return poly.status();
  s.walkable = *poly;
  for (const auto& el : obj.at("elements")) {
    auto parsed = ParseElement(el, "space[" + s.id + "]");
    if (!parsed.ok()) return parsed.status();
    s.elements.push_back(*parsed);
  }
  return s;
}

absl::StatusOr<ScenarioAnchor> ParseAnchor(const json& obj) {
  for (const char* f : {"name", "offset"}) {
    auto st = RequireField(obj, f, "anchor");
    if (!st.ok()) return st;
  }
  ScenarioAnchor a;
  a.name = obj.at("name").get<std::string>();
  a.offset = ParseVec3(obj.at("offset"));
  return a;
}

absl::StatusOr<ScenarioBinding> ParseBinding(const json& obj) {
  for (const char* f : {"anchor", "kernel"}) {
    auto st = RequireField(obj, f, "binding");
    if (!st.ok()) return st;
  }
  ScenarioBinding b;
  b.anchor = obj.at("anchor").get<std::string>();
  b.kernel = obj.at("kernel").get<std::string>();
  if (obj.contains("tag_filter")) {
    b.tag_filter = obj.at("tag_filter").get<std::string>();
    b.has_tag_filter = true;
  }
  return b;
}

absl::StatusOr<ScenarioActor> ParseActor(const json& obj) {
  for (const char* f : {"id", "space", "shared_id", "anchors", "bindings"}) {
    auto st = RequireField(obj, f, "actor");
    if (!st.ok()) return st;
  }
  ScenarioActor a;
  a.id = obj.at("id").get<std::string>();
  a.space = obj.at("space").get<std::string>();
  a.shared_id = obj.at("shared_id").get<int64_t>();
  if (!obj.at("anchors").is_array() || obj.at("anchors").empty()) {
    return absl::InvalidArgumentError("actor[" + a.id +
                                      "].anchors must be non-empty");
  }
  for (const auto& an : obj.at("anchors")) {
    auto parsed = ParseAnchor(an);
    if (!parsed.ok()) return parsed.status();
    a.anchors.push_back(*parsed);
  }
  for (const auto& bd : obj.at("bindings")) {
    auto parsed = ParseBinding(bd);
    if (!parsed.ok()) return parsed.status();
    a.bindings.push_back(*parsed);
  }
  if (obj.contains("start")) {
    const auto& s = obj.at("start");
    if (!s.is_array() || s.size() < 2) {
      return absl::InvalidArgumentError("actor[" + a.id +
                                        "].start must be [x,y] or [x,y,yaw]");
    }
    a.start_x = s.at(0).get<double>();
    a.start_y = s.at(1).get<double>();
    a.start_yaw = s.size() > 2 ? s.at(2).get<double>() : 0.0;
  }
  return a;
}

absl::StatusOr<ScenarioBehavior> ParseBehavior(const json& obj) {
  auto st = RequireField(obj, "kind", "behavior");
  if (!st.ok()) return st;
  ScenarioBehavior b;
  b.kind = obj.at("kind").get<std::string>();
  if (b.kind == "script") {
    if (!obj.contains("script")) {
      return absl::InvalidArgumentError(
          "behavior.kind == 'script' requires 'script'");
    }
    for (const auto& op : obj.at("script")) {
      auto o = RequireField(op, "op", "behavior.script[]");
      if (!o.ok()) return o;
      ScenarioScriptOp sop;
      sop.op = op.at("op").get<std::string>();
      sop.raw = op;
      b.script.push_back(std::move(sop));
    }
  }
  if (obj.contains("recording")) b.recording = obj.at("recording");
  return b;
}

ScenarioNetwork ParseNetwork(const json& obj) {
  ScenarioNetwork n;
  if (obj.contains("mode")) n.mode = obj.at("mode").get<std::string>();
  if (obj.contains("latency_ms")) n.latency_ms = obj.at("latency_ms").get<double>();
  if (obj.contains("jitter_ms")) n.jitter_ms = obj.at("jitter_ms").get<double>();
  if (obj.contains("drop_rate")) n.drop_rate = obj.at("drop_rate").get<double>();
  if (obj.contains("reorder")) n.reorder = obj.at("reorder").get<bool>();
  return n;
}

absl::StatusOr<ScenarioIdentity> ParseIdentity(const json& obj) {
  for (const char* f : {"share_from", "expect"}) {
    auto st = RequireField(obj, f, "identity");
    if (!st.ok()) return st;
  }
  ScenarioIdentity id;
  id.share_from = obj.at("share_from").get<std::string>();
  for (const auto& e : obj.at("expect")) {
    for (const char* f : {"shared_id", "resolution"}) {
      auto st = RequireField(e, f, "identity.expect[]");
      if (!st.ok()) return st;
    }
    ScenarioIdentityExpect ie;
    ie.shared_id = e.at("shared_id").get<int64_t>();
    ie.resolution = e.at("resolution").get<std::string>();
    id.expect.push_back(ie);
  }
  if (obj.contains("mismatched_ids")) {
    id.mismatched_ids = obj.at("mismatched_ids").get<bool>();
  }
  return id;
}

absl::StatusOr<ScenarioCheck> ParseCheck(const json& obj) {
  auto st = RequireField(obj, "check", "expect[]");
  if (!st.ok()) return st;
  ScenarioCheck c;
  c.check = obj.at("check").get<std::string>();
  if (obj.contains("eps")) c.eps = obj.at("eps").get<double>();
  if (obj.contains("engagements")) {
    for (const auto& e : obj.at("engagements")) {
      c.engagements.push_back(e.get<std::string>());
    }
  }
  if (obj.contains("order")) {
    for (const auto& o : obj.at("order")) {
      c.order.push_back(o.get<std::string>());
    }
  }
  return c;
}

absl::StatusOr<ScenarioMetamorphic> ParseMetamorphic(const json& obj) {
  ScenarioMetamorphic m;
  if (!obj.contains("derive_target_from_source")) return m;
  const json& d = obj.at("derive_target_from_source");
  for (const char* f : {"source", "target"}) {
    auto st = RequireField(d, f, "metamorphic.derive_target_from_source");
    if (!st.ok()) return st;
  }
  m.derive_target_from_source = true;
  m.source_space_id = d.at("source").get<std::string>();
  m.target_space_id = d.at("target").get<std::string>();
  MetamorphicTransform& t = m.transform;
  if (d.contains("translate")) {
    t.has_translate = true;
    t.translate = ParseVec3(d.at("translate"));
  }
  if (d.contains("rotate_yaw")) {
    t.has_rotate = true;
    t.rotate_yaw = d.at("rotate_yaw").get<double>();
    if (d.contains("rotate_center")) t.rotate_center = ParseVec2(d.at("rotate_center"));
  }
  if (d.contains("scale")) {
    t.has_scale = true;
    t.scale = d.at("scale").get<double>();
    if (d.contains("scale_center")) t.scale_center = ParseVec2(d.at("scale_center"));
  }
  if (d.contains("mirror_axis")) {
    t.has_mirror = true;
    t.mirror_axis = d.at("mirror_axis").get<std::string>();
  }
  return m;
}

// Applies the mirror -> scale -> rotate -> translate composition (see
// scenario.h's MetamorphicTransform doc) to a single 2D point.
Vec2 ApplyMetamorphic2D(const MetamorphicTransform& t, Vec2 p) {
  if (t.has_mirror) {
    if (t.mirror_axis == "x") {
      p.y = -p.y;
    } else if (t.mirror_axis == "y") {
      p.x = -p.x;
    }
  }
  if (t.has_scale) {
    p.x = t.scale_center.x + (p.x - t.scale_center.x) * t.scale;
    p.y = t.scale_center.y + (p.y - t.scale_center.y) * t.scale;
  }
  if (t.has_rotate) {
    const double cx = t.rotate_center.x, cy = t.rotate_center.y;
    const double dx = p.x - cx, dy = p.y - cy;
    const double c = std::cos(t.rotate_yaw), s = std::sin(t.rotate_yaw);
    p.x = cx + dx * c - dy * s;
    p.y = cy + dx * s + dy * c;
  }
  if (t.has_translate) {
    p.x += t.translate.x;
    p.y += t.translate.y;
  }
  return p;
}

double ApplyMetamorphicYaw(const MetamorphicTransform& t, double yaw) {
  double out = yaw;
  if (t.has_mirror) out = -out;
  if (t.has_rotate) out += t.rotate_yaw;
  return out;
}

// Applies the metamorphic transform to every world-absolute coordinate in
// `source`, producing the derived target space. The target's own id/role are
// kept from the ORIGINAL scenario space named target_space_id when present in
// model.spaces (its walkable/elements JSON content is otherwise ignored),
// falling back to synthesizing a space with that id + role "target".
ScenarioSpace DeriveTargetSpace(const ScenarioSpace& source,
                                const MetamorphicTransform& t,
                                const std::string& target_id) {
  ScenarioSpace out;
  out.id = target_id;
  out.role = "target";
  out.walkable.reserve(source.walkable.size());
  for (const auto& p : source.walkable) {
    out.walkable.push_back(ApplyMetamorphic2D(t, p));
  }
  out.elements.reserve(source.elements.size());
  for (const auto& e : source.elements) {
    ScenarioElement de = e;
    Vec2 p2 = ApplyMetamorphic2D(t, Vec2{e.transform.pos.x, e.transform.pos.y});
    de.transform.pos.x = p2.x;
    de.transform.pos.y = p2.y;
    // Z is not affected by the (2D) mirror/scale/rotate but does honor an
    // explicit translate.z.
    de.transform.pos.z = e.transform.pos.z + (t.has_translate ? t.translate.z : 0.0);
    de.transform.yaw = ApplyMetamorphicYaw(t, e.transform.yaw);
    out.elements.push_back(std::move(de));
  }
  return out;
}

Eigen::Matrix4d MakeWorldTransform(const Vec3& pos, double yaw) {
  Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
  const double c = std::cos(yaw), s = std::sin(yaw);
  m(0, 0) = c;
  m(0, 1) = -s;
  m(1, 0) = s;
  m(1, 1) = c;
  m(0, 3) = pos.x;
  m(1, 3) = pos.y;
  m(2, 3) = pos.z;
  return m;
}

RelationKernelType KernelFromName(const std::string& name, bool* ok) {
  *ok = true;
  if (name == "spacing") return RelationKernelType::kAvatarSpacing;
  if (name == "facing") return RelationKernelType::kAvatarFacing;
  if (name == "affordance_pose") return RelationKernelType::kAvatarAffordancePose;
  *ok = false;
  return RelationKernelType::kAvatarSpacing;
}

// Builds the Containment SpatialDefinition for one element's footprint, in
// WORLD-ABSOLUTE coordinates (element_world_pos/yaw already resolved).
absl::StatusOr<std::shared_ptr<SpatialDefinition>> BuildFootprint(
    const ScenarioFootprint& fp, const Vec3& world_pos, double world_yaw) {
  if (fp.kind == "box") {
    std::vector<double> params = {world_pos.x, world_pos.y, world_pos.z,
                                  fp.size.x,   fp.size.y,   fp.size.z};
    auto def = MakeEnvelope(ColliderType::kBox, params, /*circle_segments=*/16,
                            world_yaw);
    if (!def) {
      return absl::InvalidArgumentError("degenerate box footprint");
    }
    return def;
  }
  if (fp.kind == "sphere") {
    std::vector<double> params = {world_pos.x, world_pos.y, world_pos.z, fp.radius};
    auto def = MakeEnvelope(ColliderType::kSphere, params);
    if (!def) return absl::InvalidArgumentError("degenerate sphere footprint");
    return def;
  }
  if (fp.kind == "capsule") {
    std::vector<double> params = {world_pos.x, world_pos.y, world_pos.z,
                                  fp.radius,   fp.height,   static_cast<double>(fp.axis)};
    auto def = MakeEnvelope(ColliderType::kCapsule, params);
    if (!def) return absl::InvalidArgumentError("degenerate capsule footprint");
    return def;
  }
  if (fp.kind == "polygon") {
    // Points are local offsets from the element's world position; rotate by
    // the element's yaw then translate, to obtain world-absolute vertices.
    const double c = std::cos(world_yaw), s = std::sin(world_yaw);
    std::vector<Eigen::Vector2d> pts;
    pts.reserve(fp.points.size());
    for (const auto& p : fp.points) {
      const double wx = world_pos.x + p.x * c - p.y * s;
      const double wy = world_pos.y + p.x * s + p.y * c;
      pts.emplace_back(wx, wy);
    }
    auto polygon = std::make_shared<Polygon2D>(std::move(pts));
    return std::make_shared<NavigableSpatialDefinition>(
        polygon, /*wall=*/nullptr, /*agent_radius=*/0.0,
        GeometryRole::kContainment);
  }
  return absl::InvalidArgumentError("unknown footprint kind '" + fp.kind + "'");
}

}  // namespace

const ScenarioSpace* ScenarioModel::FindSpace(const std::string& space_id) const {
  for (const auto& s : spaces) {
    if (s.id == space_id) return &s;
  }
  return nullptr;
}

const BuiltSpace* BuildResult::FindSpace(const std::string& scenario_space_id) const {
  auto it = spaces.find(scenario_space_id);
  return it == spaces.end() ? nullptr : &it->second;
}

const BuiltActor* BuildResult::FindActor(const std::string& scenario_actor_id) const {
  for (const auto& a : actors) {
    if (a.scenario_actor_id == scenario_actor_id) return &a;
  }
  return nullptr;
}

absl::StatusOr<ScenarioModel> ParseScenario(const std::string& json_text) {
  json root;
  try {
    root = json::parse(json_text);
  } catch (const json::parse_error& e) {
    return absl::InvalidArgumentError(std::string("JSON parse error: ") + e.what());
  }
  if (!root.is_object()) {
    return absl::InvalidArgumentError("scenario root must be a JSON object");
  }

  for (const char* f : {"schema_version", "id", "tier", "regime", "verify_by",
                        "spaces", "actors", "behavior", "expect"}) {
    auto st = RequireField(root, f, "scenario");
    if (!st.ok()) return st;
  }

  ScenarioModel m;
  try {
    m.schema_version = root.at("schema_version").get<std::string>();
    m.id = root.at("id").get<std::string>();
    m.tier = root.at("tier").get<std::string>();
    m.regime = root.at("regime").get<std::string>();
    for (const auto& v : root.at("verify_by")) {
      m.verify_by.push_back(v.get<std::string>());
    }
  } catch (const json::exception& e) {
    return absl::InvalidArgumentError(std::string("malformed scenario header: ") +
                                      e.what());
  }

  if (!root.at("spaces").is_array() || root.at("spaces").empty()) {
    return absl::InvalidArgumentError("scenario.spaces must be non-empty");
  }
  for (const auto& s : root.at("spaces")) {
    auto parsed = ParseSpace(s);
    if (!parsed.ok()) return parsed.status();
    m.spaces.push_back(*parsed);
  }

  if (root.contains("identity")) {
    auto id = ParseIdentity(root.at("identity"));
    if (!id.ok()) return id.status();
    m.identity = *id;
    m.has_identity = true;
  }

  if (!root.at("actors").is_array() || root.at("actors").empty()) {
    return absl::InvalidArgumentError("scenario.actors must be non-empty");
  }
  for (const auto& a : root.at("actors")) {
    auto parsed = ParseActor(a);
    if (!parsed.ok()) return parsed.status();
    m.actors.push_back(*parsed);
  }

  {
    auto b = ParseBehavior(root.at("behavior"));
    if (!b.ok()) return b.status();
    m.behavior = *b;
  }

  if (root.contains("network")) {
    m.network = ParseNetwork(root.at("network"));
    m.has_network = true;
  }
  if (root.contains("core")) m.core = root.at("core");
  if (root.contains("app")) m.app = root.at("app");

  for (const auto& c : root.at("expect")) {
    auto parsed = ParseCheck(c);
    if (!parsed.ok()) return parsed.status();
    m.expect.push_back(*parsed);
  }

  if (root.contains("metamorphic")) {
    auto meta = ParseMetamorphic(root.at("metamorphic"));
    if (!meta.ok()) return meta.status();
    m.metamorphic = *meta;
    m.has_metamorphic = m.metamorphic.derive_target_from_source;
  }

  // Apply metamorphic derivation: replace/insert the named target space.
  if (m.has_metamorphic) {
    const ScenarioSpace* src = m.FindSpace(m.metamorphic.source_space_id);
    if (!src) {
      return absl::InvalidArgumentError(
          "metamorphic.derive_target_from_source: source space '" +
          m.metamorphic.source_space_id + "' not found");
    }
    ScenarioSpace derived =
        DeriveTargetSpace(*src, m.metamorphic.transform, m.metamorphic.target_space_id);
    bool replaced = false;
    for (auto& s : m.spaces) {
      if (s.id == m.metamorphic.target_space_id) {
        s = derived;
        replaced = true;
        break;
      }
    }
    if (!replaced) m.spaces.push_back(std::move(derived));
  }

  return m;
}

absl::StatusOr<ScenarioModel> LoadScenario(const std::string& path) {
  std::ifstream in(path);
  if (!in.good()) {
    return absl::NotFoundError("cannot open scenario file: " + path);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ParseScenario(ss.str());
}

namespace {

// Adds one element (with its footprint/sub-regions) as a child of
// `parent_id`, whose world transform is `parent_world` (already resolved).
// Returns the new element's world transform (needed to resolve grandchildren
// as ancestors are processed in topological order).
absl::StatusOr<Eigen::Matrix4d> AddScenarioElement(
    Space* space, const ScenarioElement& e, element_id_t new_id,
    element_id_t parent_id, const Eigen::Matrix4d& parent_world) {
  auto* tree = space->GetElementTree();
  absl::Status add_status = tree->AddElement(new_id, parent_id);
  if (!add_status.ok()) return add_status;

  Element* elem = tree->GetElement(new_id);
  elem->SetTag(e.tag);
  elem->SetSharedId(static_cast<shared_element_id_t>(e.shared_id));

  const Eigen::Matrix4d world = MakeWorldTransform(e.transform.pos, e.transform.yaw);
  elem->SetLocalTransform(parent_world.inverse() * world);

  if (e.has_footprint) {
    auto def = BuildFootprint(e.footprint, e.transform.pos, e.transform.yaw);
    if (!def.ok()) return def.status();
    elem->SetSpatialDefinition(*def);
    for (const auto& sr : e.sub_regions) {
      SubRegion region;
      region.name = sr.name;
      region.tag = sr.kind;
      Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
      t(0, 3) = e.transform.pos.x;
      t(1, 3) = e.transform.pos.y;
      t(2, 3) = e.transform.pos.z;
      region.local_transform = t;
      if (sr.points.size() >= 3) {
        std::vector<Eigen::Vector2d> pts;
        pts.reserve(sr.points.size());
        for (const auto& p : sr.points) pts.emplace_back(p.x, p.y);
        region.boundary = std::make_shared<Polygon2D>(std::move(pts));
      }
      (*def)->AddSubRegion(std::move(region));
    }
  }
  return world;
}

}  // namespace

absl::StatusOr<BuildResult> BuildSpaces(const ScenarioModel& model, LocalContext& ctx) {
  BuildResult result;

  constexpr element_id_t kRoomElementId = 1;

  for (const auto& sspace : model.spaces) {
    Space* space = ctx.AddSpace();
    if (!space) {
      return absl::InternalError("LocalContext::AddSpace() failed for space '" +
                                 sspace.id + "'");
    }
    BuiltSpace built;
    built.scenario_space_id = sspace.id;
    built.space = space;

    // Root element ("room"): the space's walkable polygon, kNavigable.
    auto* tree = space->GetElementTree();
    absl::Status root_status = tree->AddElement(kRoomElementId);
    if (!root_status.ok()) return root_status;
    Element* room = tree->GetElement(kRoomElementId);
    room->SetTag("room");
    std::vector<Eigen::Vector2d> walkable_pts;
    walkable_pts.reserve(sspace.walkable.size());
    for (const auto& p : sspace.walkable) walkable_pts.emplace_back(p.x, p.y);
    auto walkable_polygon = std::make_shared<Polygon2D>(std::move(walkable_pts));
    room->SetSpatialDefinition(std::make_shared<NavigableSpatialDefinition>(
        walkable_polygon, /*wall=*/nullptr, /*agent_radius=*/0.0,
        GeometryRole::kNavigable));
    built.room_element_id = kRoomElementId;
    built.element_id_by_tag["room"] = kRoomElementId;

    // Elements: topological insertion order so a "parent" tag always resolves
    // to an already-added element (ElementTree::AddElement requires the
    // parent to exist). "room" always resolves to kRoomElementId.
    std::vector<const ScenarioElement*> pending;
    pending.reserve(sspace.elements.size());
    for (const auto& e : sspace.elements) pending.push_back(&e);

    element_id_t next_id = kRoomElementId + 1;
    // world transform of each already-placed element, by element id.
    std::unordered_map<element_id_t, Eigen::Matrix4d> world_by_id;
    world_by_id[kRoomElementId] = Eigen::Matrix4d::Identity();

    bool progress = true;
    while (!pending.empty() && progress) {
      progress = false;
      for (auto it = pending.begin(); it != pending.end();) {
        const ScenarioElement* e = *it;
        element_id_t parent_id = kInvalidElementId;
        if (e->parent == "room") {
          parent_id = kRoomElementId;
        } else {
          auto found = built.element_id_by_tag.find(e->parent);
          if (found != built.element_id_by_tag.end()) parent_id = found->second;
        }
        if (parent_id == kInvalidElementId) {
          ++it;
          continue;  // parent not resolved yet; try again next pass
        }
        element_id_t id = next_id++;
        auto world = AddScenarioElement(space, *e, id, parent_id,
                                        world_by_id.at(parent_id));
        if (!world.ok()) return world.status();
        world_by_id[id] = *world;
        built.element_id_by_tag[e->tag] = id;
        built.element_id_by_shared_id[e->shared_id] = id;
        it = pending.erase(it);
        progress = true;
      }
    }
    if (!pending.empty()) {
      std::string names;
      for (const auto* e : pending) names += e->tag + " ";
      return absl::FailedPreconditionError(
          "space '" + sspace.id +
          "': could not resolve parent for element(s): " + names);
    }

    built.next_free_element_id = next_id;
    result.spaces.emplace(sspace.id, std::move(built));
  }

  // Actors. Each actor gets its OWN leaf element (no spatial definition,
  // child of the room) carrying the actor's shared_id -- mirroring
  // cross_space_test.cc's AddAvatarLeaf convention (as opposed to
  // AddAvatarActor's single-avatar-only trick of reusing/re-tagging the room
  // element itself). This keeps multiple actors per space (e.g. the
  // avatar-avatar-chatter regression) from colliding on one shared element,
  // and Space::UpdateActorState already excludes every actor's own element
  // from referent candidates via its actor_element_ids set, so no further
  // marking is required to stop actors from binding to each other.
  for (const auto& sactor : model.actors) {
    auto found = result.spaces.find(sactor.space);
    if (found == result.spaces.end()) {
      return absl::InvalidArgumentError("actor '" + sactor.id +
                                        "' references unknown space '" +
                                        sactor.space + "'");
    }
    BuiltSpace& built_space = found->second;

    element_id_t leaf_id = built_space.next_free_element_id++;
    absl::Status leaf_status =
        built_space.space->GetElementTree()->AddElement(
            leaf_id, built_space.room_element_id);
    if (!leaf_status.ok()) return leaf_status;
    Element* leaf = built_space.space->GetElementTree()->GetElement(leaf_id);
    leaf->SetTag("actor$" + sactor.id);
    leaf->SetSharedId(static_cast<shared_element_id_t>(sactor.shared_id));
    built_space.element_id_by_tag["actor$" + sactor.id] = leaf_id;
    built_space.element_id_by_shared_id[sactor.shared_id] = leaf_id;

    std::vector<SubjectAnchor> anchors;
    anchors.reserve(sactor.anchors.size());
    for (const auto& a : sactor.anchors) {
      anchors.emplace_back(a.name, Eigen::Vector3d(a.offset.x, a.offset.y, a.offset.z));
    }
    auto actor_id = built_space.space->CreateActor(anchors);
    built_space.space->GetActor(actor_id)->SetElementId(leaf_id);

    for (const auto& b : sactor.bindings) {
      bool ok = false;
      RelationKernelType kernel = KernelFromName(b.kernel, &ok);
      if (!ok) {
        return absl::InvalidArgumentError("actor '" + sactor.id +
                                          "': unknown kernel name '" + b.kernel +
                                          "'");
      }
      built_space.space->Bind(actor_id, b.anchor, kernel,
                              b.has_tag_filter ? b.tag_filter : "");
    }

    BuiltActor ba;
    ba.scenario_actor_id = sactor.id;
    ba.scenario_space_id = sactor.space;
    ba.actor_id = actor_id;
    result.actors.push_back(ba);
  }

  return result;
}

}  // namespace harness
}  // namespace spatialblend
