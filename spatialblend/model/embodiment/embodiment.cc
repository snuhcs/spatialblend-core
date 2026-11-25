// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/model/embodiment/embodiment.cc -- see embodiment.h.
#include "spatialblend/model/embodiment/embodiment.h"

#include <cctype>
#include <cmath>
#include <set>

#include <nlohmann/json.hpp>

namespace spatialblend {

namespace {

using nlohmann::json;

bool Fail(std::string* error, const std::string& what) {
  if (error) *error = what;
  return false;
}

bool ReadVec(const json& j, size_t n, double* out) {
  if (!j.is_array() || j.size() != n) return false;
  for (size_t i = 0; i < n; ++i) {
    if (!j[i].is_number()) return false;
    out[i] = j[i].get<double>();
    if (!std::isfinite(out[i])) return false;
  }
  return true;
}

uint32_t FamilyBit(const std::string& name) {
  if (name == "Spacing") return kFamilySpacing;
  if (name == "Facing") return kFamilyFacing;
  if (name == "AffordancePose") return kFamilyAffordancePose;
  return 0;
}

// The normalization the format fixes, written out so that no reader has to guess (design section 12).
json NormalizationBlock() {
  return {{"units", "m"},
          {"angle_units", "rad"},
          {"handedness", "right"},
          {"up", "z"},
          {"forward", "x"},
          {"left", "y"},
          {"origin", "rest position of the first root joint (exactly 0,0,0)"},
          {"rest", "joint frame -> model space, quaternion [x,y,z,w]"},
          {"limits", "radians; ZYX Euler [x=roll,y=pitch,z=yaw] of rest_local^-1 * local, rest_local = parent_rest^-1 * rest"}};
}

bool ParseLimitPair(const json& v, bool* has_lo, bool* has_hi, double* lo, double* hi) {
  if (!v.is_array() || v.size() != 2) return false;
  *has_lo = !v[0].is_null();
  *has_hi = !v[1].is_null();
  if ((*has_lo && !v[0].is_number()) || (*has_hi && !v[1].is_number())) return false;
  if (*has_lo) *lo = v[0].get<double>();
  if (*has_hi) *hi = v[1].get<double>();
  return (!*has_lo || std::isfinite(*lo)) && (!*has_hi || std::isfinite(*hi));
}

bool ParseChain(const json& jc, Embodiment* e, std::string* error) {
  if (!jc.is_object()) return Fail(error, "chain must be an object");
  if (jc.contains("normalization") && jc["normalization"] != NormalizationBlock())
    return Fail(error, "chain.normalization differs from the one the core defines");
  Chain& c = e->chain;
  c.present = true;
  c.rest_pose = jc.value("rest_pose", std::string());
  if (jc.contains("provenance") && jc["provenance"].is_object()) c.provenance_json = jc["provenance"].dump();
  if (jc.contains("limits_provenance") && jc["limits_provenance"].is_object())
    c.limits_provenance_json = jc["limits_provenance"].dump();
  if (!jc.contains("joints") || !jc["joints"].is_array()) return Fail(error, "chain.joints missing");
  c.joints.assign(e->joints.size(), ChainJoint{});
  std::vector<bool> seen(e->joints.size(), false);
  for (const json& jj : jc["joints"]) {
    if (!jj.is_object() || !jj.contains("name") || !jj["name"].is_string())
      return Fail(error, "a chain joint needs a name");
    const std::string name = jj["name"].get<std::string>();
    const int id = e->BoneId(name);
    if (id < 0) return Fail(error, "chain joint '" + name + "' is not in the skeleton");
    if (seen[id]) return Fail(error, "chain joint '" + name + "' repeated");
    seen[id] = true;
    ChainJoint& cj = c.joints[id];
    double p[3], q[4];
    if (!jj.contains("rest") || !jj["rest"].is_object() || !jj["rest"].contains("position") ||
        !jj["rest"].contains("rotation") || !ReadVec(jj["rest"]["position"], 3, p) ||
        !ReadVec(jj["rest"]["rotation"], 4, q))
      return Fail(error, "chain joint '" + name + "': rest needs a 3-vector position and a 4-vector rotation");
    Eigen::Quaterniond rq(q[3], q[0], q[1], q[2]);
    if (rq.norm() == 0.0) return Fail(error, "chain joint '" + name + "': rest.rotation has zero length");
    rq.normalize();
    cj.rest.position = Eigen::Vector3d(p[0], p[1], p[2]);
    cj.rest.rotation = rq;
    if (jj.contains("limits") && !jj["limits"].is_null()) {
      const json& jl = jj["limits"];
      const char* axes[3] = {"x", "y", "z"};
      if (!jl.is_object()) return Fail(error, "chain joint '" + name + "': limits must be an object");
      for (int a = 0; a < 3; ++a)
        if (jl.contains(axes[a]) &&
            !ParseLimitPair(jl[axes[a]], &cj.has_lo[a], &cj.has_hi[a], &cj.lo[a], &cj.hi[a]))
          return Fail(error, "chain joint '" + name + "': limits." + axes[a] + " must be [min|null, max|null]");
    }
  }
  for (size_t i = 0; i < seen.size(); ++i)
    if (!seen[i]) return Fail(error, "chain lacks joint '" + e->joints[i].name + "'");
  return true;
}

bool IsHex64(const std::string& v) {
  if (v.size() != 64) return false;
  for (char c : v)
    if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
  return true;
}

// A body measurement never arrives unsourced: `procedure` (how it was obtained) AND either `asset_sha256` (the asset
// it was read from, 64 hex digits) or an explicit `source` (a runtime calibration, a vendor specification, a paper).
bool HasProvenance(const std::string& text) {
  if (text.empty()) return false;
  const json prov = json::parse(text);
  if (!prov.is_object()) return false;
  const auto str = [&](const char* key) -> std::string {
    return prov.contains(key) && prov[key].is_string() ? prov[key].get<std::string>() : std::string();
  };
  return !str("procedure").empty() && (IsHex64(str("asset_sha256")) || !str("source").empty());
}

// Resource caps for a registration document (a guard against hostile or accidental input, not a body value).
constexpr size_t kMaxSpecBytes = 4u << 20;
constexpr int kMaxSpecDepth = 16;

bool DepthWithin(const std::string& text, int limit) {
  int depth = 0;
  bool in_string = false, escaped = false;
  for (char c : text) {
    if (in_string) {
      if (escaped) escaped = false;
      else if (c == '\\') escaped = true;
      else if (c == '"') in_string = false;
    } else if (c == '"') in_string = true;
    else if (c == '{' || c == '[') { if (++depth > limit) return false; }
    else if (c == '}' || c == ']') --depth;
  }
  return true;
}

}  // namespace

bool Embodiment::FromJson(const std::string& text, Embodiment* out, std::string* error) {
  if (!out) return Fail(error, "null output");
  if (text.size() > kMaxSpecBytes) return Fail(error, "spec larger than the 4 MiB cap");
  if (!DepthWithin(text, kMaxSpecDepth)) return Fail(error, "spec nested deeper than 16 levels");
  const json j = json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded() || !j.is_object()) return Fail(error, "not a JSON object");
  if (j.value("format", std::string()) != "spatialblend.embodiment")
    return Fail(error, "format must be \"spatialblend.embodiment\"");
  const std::string version = j.value("schema_version", std::string());
  if (version.empty() || version[0] != '1' || (version.size() > 1 && version[1] != '.'))
    return Fail(error, "schema_version must be 1.x");
  Embodiment e;
  e.id = j.value("id", std::string());
  e.units = j.value("units", std::string("m"));
  e.frame = j.value("frame", std::string("z-up, right-handed"));
  e.bone_frame = j.value("bone_frame", std::string());
  if (e.units != "m") return Fail(error, "units must be \"m\"");
  const auto sk = j.find("skeleton");
  if (sk == j.end() || !sk->is_object() || !sk->contains("joints") || !(*sk)["joints"].is_array())
    return Fail(error, "skeleton.joints missing");
  for (const json& jj : (*sk)["joints"]) {
    if (!jj.is_object() || !jj.contains("name") || !jj["name"].is_string() || !jj.contains("parent") ||
        !jj["parent"].is_number_integer())
      return Fail(error, "a joint needs a string name and an integer parent");
    e.joints.push_back({jj["name"].get<std::string>(), jj["parent"].get<int>()});
  }
  const auto an = j.find("anchors");
  if (an == j.end() || !an->is_array()) return Fail(error, "anchors missing");
  for (const json& ja : *an) {
    if (!ja.is_object() || !ja.contains("name") || !ja["name"].is_string())
      return Fail(error, "an anchor needs a string name");
    EmbodimentAnchor a;
    a.name = ja["name"].get<std::string>();
    if (!ja.contains("bone") || !(ja["bone"].is_string() || ja["bone"].is_null()))
      return Fail(error, "anchor '" + a.name + "': bone must be a joint name or null");
    if (ja["bone"].is_string()) {
      const std::string bone = ja["bone"].get<std::string>();
      a.bone = e.BoneId(bone);
      if (a.bone < 0) return Fail(error, "anchor '" + a.name + "': unknown bone '" + bone + "'");
    }
    if (ja.contains("offset")) {
      const json& jo = ja["offset"];
      if (!jo.is_object()) return Fail(error, "anchor '" + a.name + "': offset must be an object");
      double p[3] = {0, 0, 0}, q[4] = {0, 0, 0, 1};
      if (jo.contains("position") && !ReadVec(jo["position"], 3, p))
        return Fail(error, "anchor '" + a.name + "': offset.position must be 3 finite numbers");
      if (jo.contains("rotation")) {
        if (!ReadVec(jo["rotation"], 4, q))
          return Fail(error, "anchor '" + a.name + "': offset.rotation must be a quaternion [x,y,z,w]");
      }
      a.offset.position = Eigen::Vector3d(p[0], p[1], p[2]);
      Eigen::Quaterniond rq(q[3], q[0], q[1], q[2]);
      if (rq.norm() == 0.0)
        return Fail(error, "anchor '" + a.name + "': offset.rotation has zero length");
      rq.normalize();
      a.offset.rotation = rq;
    }
    a.identity_offset = a.offset.position.isZero(0.0) && a.offset.rotation.vec().isZero(0.0) &&
                        a.offset.rotation.w() == 1.0;
    if (!ja.contains("families") || !ja["families"].is_array())
      return Fail(error, "anchor '" + a.name + "': families missing");
    for (const json& jf : ja["families"]) {
      const uint32_t bit = jf.is_string() ? FamilyBit(jf.get<std::string>()) : 0;
      if (bit == 0)
        return Fail(error, "anchor '" + a.name + "': family must be Spacing, Facing or AffordancePose");
      a.families |= bit;
    }
    if (ja.contains("provenance") && !ja["provenance"].is_null()) {
      if (!ja["provenance"].is_object())
        return Fail(error, "anchor '" + a.name + "': provenance must be an object");
      a.provenance_json = ja["provenance"].dump();
    }
    if (ja.contains("standing") && !ja["standing"].is_null()) {
      const json& js = ja["standing"];
      if (!js.is_object() || !js.contains("height_m") || !js["height_m"].is_number())
        return Fail(error, "anchor '" + a.name + "': standing.height_m must be a number");
      a.has_standing = true;
      a.standing_height_m = js["height_m"].get<double>();
      if (js.contains("provenance") && js["provenance"].is_object())
        a.standing_provenance_json = js["provenance"].dump();
    }
    e.anchors.push_back(std::move(a));
  }
  if (j.contains("body") && !j["body"].is_null()) {
    const json& jb = j["body"];
    if (!jb.is_object() || !jb.contains("floor_footprint_radius_m") || !jb["floor_footprint_radius_m"].is_number())
      return Fail(error, "body.floor_footprint_radius_m must be a number");
    e.body.present = true;
    e.body.floor_footprint_radius_m = jb["floor_footprint_radius_m"].get<double>();
    if (jb.contains("provenance") && jb["provenance"].is_object()) e.body.provenance_json = jb["provenance"].dump();
  }
  if (j.contains("chain") && !j["chain"].is_null()) {
    if (!ParseChain(j["chain"], &e, error)) return false;
  }
  if (!e.Validate(error)) return false;
  *out = std::move(e);
  return true;
}

bool Embodiment::Validate(std::string* error) const {
  if (id.empty()) return Fail(error, "id is empty");
  if (joints.empty() || joints.size() > 1024) return Fail(error, "skeleton needs 1..1024 joints");
  std::set<std::string> seen;
  for (size_t i = 0; i < joints.size(); ++i) {
    if (joints[i].name.empty()) return Fail(error, "a joint has an empty name");
    if (!seen.insert(joints[i].name).second) return Fail(error, "duplicate joint '" + joints[i].name + "'");
    if (joints[i].parent < -1 || joints[i].parent >= static_cast<int>(i))
      return Fail(error, "joint '" + joints[i].name + "': parent must be -1 or the index of an earlier joint");
  }
  seen.clear();
  for (const EmbodimentAnchor& a : anchors) {
    if (a.name.empty()) return Fail(error, "an anchor has an empty name");
    if (!seen.insert(a.name).second) return Fail(error, "duplicate anchor '" + a.name + "'");
    if (a.families == 0) return Fail(error, "anchor '" + a.name + "' serves no family");
    if (a.bone >= static_cast<int>(joints.size()) || a.bone < -1)
      return Fail(error, "anchor '" + a.name + "': bone out of range");
    if (a.bone >= 0 && a.name == "root")
      return Fail(error, "anchor 'root' is the actor root: bind it to bone null");
    if (a.bone < 0 && !a.identity_offset)
      return Fail(error, "anchor '" + a.name + "': the actor root has no offset");
    if (!a.identity_offset && !HasProvenance(a.provenance_json))
      return Fail(error, "anchor '" + a.name +
                             "': a non-identity offset needs provenance.procedure and provenance.asset_sha256 (64 hex) or provenance.source");
    if (a.has_standing) {
      if (!std::isfinite(a.standing_height_m) || a.standing_height_m <= 0.0)
        return Fail(error, "anchor '" + a.name + "': standing.height_m must be a finite positive number of metres");
      if (!HasProvenance(a.standing_provenance_json))
        return Fail(error, "anchor '" + a.name +
                               "': standing needs provenance.procedure and provenance.asset_sha256 (64 hex) or provenance.source");
    }
  }
  if (body.present) {
    if (!std::isfinite(body.floor_footprint_radius_m) || body.floor_footprint_radius_m <= 0.0)
      return Fail(error, "body.floor_footprint_radius_m must be a finite positive number of metres");
    if (!HasProvenance(body.provenance_json))
      return Fail(error, "body needs provenance.procedure and provenance.asset_sha256 (64 hex) or provenance.source");
  }
  if (chain.present) {
    if (chain.joints.size() != joints.size()) return Fail(error, "chain must cover every skeleton joint");
    if (chain.rest_pose != "A" && chain.rest_pose != "T" && chain.rest_pose != "other")
      return Fail(error, "chain.rest_pose must be \"A\", \"T\" or \"other\"");
    if (!HasProvenance(chain.provenance_json))
      return Fail(error, "chain needs provenance.procedure and provenance.asset_sha256 (64 hex) or provenance.source");
    bool first_root = true, any_limit = false;
    for (size_t i = 0; i < joints.size(); ++i) {
      const ChainJoint& cj = chain.joints[i];
      if (!cj.rest.position.allFinite()) return Fail(error, "chain joint '" + joints[i].name + "': non-finite rest");
      if (joints[i].parent < 0 && first_root) {
        first_root = false;
        if (!cj.rest.position.isZero(0.0))
          return Fail(error, "the first root joint's rest position must be exactly 0,0,0 (the model-space origin)");
      }
      for (int a = 0; a < 3; ++a)
        if (cj.has_lo[a] && cj.has_hi[a] && cj.lo[a] > cj.hi[a])
          return Fail(error, "chain joint '" + joints[i].name + "': a limit has min > max");
      any_limit = any_limit || cj.HasLimits();
    }
    if (any_limit && !HasProvenance(chain.limits_provenance_json))
      return Fail(error, "joint limits need limits_provenance.procedure and limits_provenance.asset_sha256 (64 hex) or limits_provenance.source");
  }
  return true;
}

std::string Embodiment::ToJson() const {
  json j;
  j["format"] = "spatialblend.embodiment";
  j["schema_version"] = "1.0.0";
  j["id"] = id;
  j["units"] = units;
  j["frame"] = frame;
  j["bone_frame"] = bone_frame;
  json joints_j = json::array();
  for (const auto& jt : joints) joints_j.push_back({{"name", jt.name}, {"parent", jt.parent}});
  j["skeleton"] = {{"joints", joints_j}};
  json anchors_j = json::array();
  for (const auto& a : anchors) {
    json ja;
    ja["name"] = a.name;
    ja["bone"] = a.bone >= 0 ? json(joints[a.bone].name) : json(nullptr);
    ja["offset"] = {{"position", {a.offset.position.x(), a.offset.position.y(), a.offset.position.z()}},
                    {"rotation", {a.offset.rotation.x(), a.offset.rotation.y(), a.offset.rotation.z(),
                                  a.offset.rotation.w()}}};
    json fams = json::array();
    if (a.families & kFamilySpacing) fams.push_back("Spacing");
    if (a.families & kFamilyFacing) fams.push_back("Facing");
    if (a.families & kFamilyAffordancePose) fams.push_back("AffordancePose");
    ja["families"] = fams;
    if (!a.provenance_json.empty()) ja["provenance"] = json::parse(a.provenance_json);
    if (a.has_standing)
      ja["standing"] = {{"height_m", a.standing_height_m}, {"provenance", json::parse(a.standing_provenance_json)}};
    anchors_j.push_back(std::move(ja));
  }
  j["anchors"] = anchors_j;
  if (body.present) {
    j["body"] = {{"floor_footprint_radius_m", body.floor_footprint_radius_m},
                 {"provenance", json::parse(body.provenance_json)}};
  }
  if (chain.present) {
    json c;
    c["normalization"] = NormalizationBlock();
    c["rest_pose"] = chain.rest_pose;
    c["provenance"] = json::parse(chain.provenance_json);
    if (!chain.limits_provenance_json.empty()) c["limits_provenance"] = json::parse(chain.limits_provenance_json);
    json cj_arr = json::array();
    for (size_t i = 0; i < joints.size(); ++i) {
      const ChainJoint& cj = chain.joints[i];
      json jj;
      jj["name"] = joints[i].name;
      jj["rest"] = {{"position", {cj.rest.position.x(), cj.rest.position.y(), cj.rest.position.z()}},
                    {"rotation", {cj.rest.rotation.x(), cj.rest.rotation.y(), cj.rest.rotation.z(),
                                  cj.rest.rotation.w()}}};
      if (cj.HasLimits()) {
        json jl = json::object();
        const char* axes[3] = {"x", "y", "z"};
        for (int a = 0; a < 3; ++a)
          if (cj.has_lo[a] || cj.has_hi[a])
            jl[axes[a]] = {cj.has_lo[a] ? json(cj.lo[a]) : json(nullptr), cj.has_hi[a] ? json(cj.hi[a]) : json(nullptr)};
        jj["limits"] = jl;
      }
      cj_arr.push_back(std::move(jj));
    }
    c["joints"] = cj_arr;
    j["chain"] = c;
  }
  return j.dump();
}

int Embodiment::BoneId(const std::string& name) const {
  for (size_t i = 0; i < joints.size(); ++i)
    if (joints[i].name == name) return static_cast<int>(i);
  return -1;
}

int Embodiment::AnchorIndex(const std::string& name) const {
  for (size_t i = 0; i < anchors.size(); ++i)
    if (anchors[i].name == name) return static_cast<int>(i);
  return -1;
}

const EmbodimentAnchor* Embodiment::FindAnchor(const std::string& name) const {
  const int i = AnchorIndex(name);
  return i < 0 ? nullptr : &anchors[i];
}

RigidTransform ComposeAnchor(const EmbodimentAnchor& anchor, const RigidTransform& bone) {
  RigidTransform out;
  out.position = bone.position + bone.rotation * anchor.offset.position;
  out.rotation = (bone.rotation * anchor.offset.rotation).normalized();
  return out;
}

RigidTransform BoneFromAnchor(const EmbodimentAnchor& anchor, const RigidTransform& anchor_pose) {
  RigidTransform out;
  out.rotation = (anchor_pose.rotation * anchor.offset.rotation.inverse()).normalized();
  out.position = anchor_pose.position - out.rotation * anchor.offset.position;
  return out;
}

EulerZyx EulerFromRotation(const Eigen::Quaterniond& rotation) {
  const Eigen::Matrix3d r = rotation.normalized().toRotationMatrix();
  EulerZyx e;
  const double cp = std::hypot(r(0, 0), r(1, 0));
  e.pitch = std::atan2(-r(2, 0), cp);
  if (cp > 0.0) {
    e.yaw = std::atan2(r(1, 0), r(0, 0));
    e.roll = std::atan2(r(2, 1), r(2, 2));
  } else {  // the forward axis is vertical: yaw and roll are one angle; carry it in yaw
    e.yaw = std::atan2(-r(0, 1), r(1, 1));
    e.roll = 0.0;
  }
  return e;
}

double SegmentLength(const Embodiment& e, int i) {
  if (!e.chain.present || i < 0 || i >= static_cast<int>(e.joints.size()) || e.joints[i].parent < 0) return 0.0;
  return (e.chain.joints[i].rest.position - e.chain.joints[e.joints[i].parent].rest.position).norm();
}

bool AxisBasis::FromRows(const double rows[9], AxisBasis* out, std::string* error) {
  Eigen::Matrix3d m;
  for (int r = 0; r < 3; ++r) {
    int nonzero = 0;
    for (int c = 0; c < 3; ++c) {
      const double v = rows[3 * r + c];
      m(r, c) = v;
      if (v != 0.0) {
        ++nonzero;
        if (v != 1.0 && v != -1.0) return Fail(error, "basis entries must be 0 or +-1");
      }
    }
    if (nonzero != 1) return Fail(error, "each basis row needs exactly one +-1");
  }
  for (int c = 0; c < 3; ++c) {
    int nonzero = 0;
    for (int r = 0; r < 3; ++r) nonzero += m(r, c) != 0.0;
    if (nonzero != 1) return Fail(error, "each basis column needs exactly one +-1");
  }
  out->b = m;
  return true;
}

RigidTransform ToCanonical(const AxisBasis& basis, const RigidTransform& native) {
  RigidTransform out;
  out.position = basis.b * native.position;
  const Eigen::Matrix3d r = basis.b * native.rotation.toRotationMatrix() * basis.b.transpose();
  out.rotation = Eigen::Quaterniond(r).normalized();
  return out;
}

RigidTransform FromCanonical(const AxisBasis& basis, const RigidTransform& canonical) {
  RigidTransform out;
  out.position = basis.b.transpose() * canonical.position;
  const Eigen::Matrix3d r = basis.b.transpose() * canonical.rotation.toRotationMatrix() * basis.b;
  out.rotation = Eigen::Quaterniond(r).normalized();
  return out;
}

}  // namespace spatialblend
