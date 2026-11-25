// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/sim/harness/geometry_component_probe.h"

#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "spatialblend/model/element/element.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/geometry/polygon_coordinate_mapper.h"
#include "spatialblend/solver/relation_solver.h"
#include "spatialblend/space/space.h"

namespace spatialblend::harness {
namespace {

constexpr int kSchemaVersion = 1;
constexpr size_t kMaxCases = 256;
constexpr size_t kMaxPoints = 128;
constexpr size_t kMaxSourceSamples = 20000;
constexpr double kAffineTolerance = 1e-12;

using json = nlohmann::json;

absl::Status Invalid(const std::string& message) {
  return absl::InvalidArgumentError("geometry component probe: " + message);
}

absl::Status RequireExactKeys(const json& value,
                              const std::set<std::string>& expected,
                              const std::string& label) {
  if (!value.is_object()) return Invalid(label + " must be an object");
  std::set<std::string> actual;
  for (auto it = value.begin(); it != value.end(); ++it)
    actual.insert(it.key());
  if (actual != expected)
    return Invalid(label + " has missing or unexpected fields");
  return absl::OkStatus();
}

absl::StatusOr<double> ParseFinite(const json& value,
                                   const std::string& label) {
  if (!value.is_number()) return Invalid(label + " must be a JSON number");
  const double number = value.get<double>();
  if (!std::isfinite(number)) return Invalid(label + " must be finite");
  return number;
}

absl::StatusOr<Eigen::Vector3d> ParseVector3(const json& value,
                                             const std::string& label) {
  if (!value.is_array() || value.size() != 3)
    return Invalid(label + " must be [x,y,z]");
  Eigen::Vector3d vector;
  for (int i = 0; i < 3; ++i) {
    auto component = ParseFinite(value.at(i), label + " component");
    if (!component.ok()) return component.status();
    vector(i) = *component;
  }
  return vector;
}

absl::StatusOr<Eigen::Matrix4d> ParseAffine(const json& value,
                                            const std::string& label) {
  if (!value.is_array() || value.size() != 4)
    return Invalid(label + " must be 4 rows");
  Eigen::Matrix4d matrix;
  for (int row = 0; row < 4; ++row) {
    const json& source_row = value.at(row);
    if (!source_row.is_array() || source_row.size() != 4) {
      return Invalid(label + " rows must have 4 values");
    }
    for (int column = 0; column < 4; ++column) {
      auto number = ParseFinite(source_row.at(column), label + " entry");
      if (!number.ok()) return number.status();
      matrix(row, column) = *number;
    }
  }
  if (matrix(3, 0) != 0.0 || matrix(3, 1) != 0.0 || matrix(3, 2) != 0.0 ||
      matrix(3, 3) != 1.0) {
    return Invalid(label + " must be affine with last row [0,0,0,1]");
  }
  const double determinant = matrix.determinant();
  if (!std::isfinite(determinant) ||
      std::abs(determinant) <= kAffineTolerance) {
    return Invalid(label + " must be invertible");
  }
  return matrix;
}

absl::StatusOr<std::vector<Eigen::Matrix4d>> ParseTransforms(
    const json& value) {
  if (!value.is_array() || value.size() != 4) {
    return Invalid(
        "transforms must contain space, region, object, affordance matrices");
  }
  std::vector<Eigen::Matrix4d> transforms;
  transforms.reserve(4);
  for (size_t i = 0; i < 4; ++i) {
    auto matrix =
        ParseAffine(value.at(i), "transforms[" + std::to_string(i) + "]");
    if (!matrix.ok()) return matrix.status();
    transforms.push_back(*matrix);
  }
  return transforms;
}

json MatrixJson(const Eigen::Matrix4d& matrix) {
  json output = json::array();
  for (int row = 0; row < 4; ++row) {
    json result_row = json::array();
    for (int column = 0; column < 4; ++column)
      result_row.push_back(matrix(row, column));
    output.push_back(std::move(result_row));
  }
  return output;
}

json VectorJson(const Eigen::Vector3d& value) {
  return json::array({value.x(), value.y(), value.z()});
}

absl::StatusOr<std::string> ParseCaseId(const json& value);

absl::StatusOr<std::vector<Eigen::Vector3d>> ParseTriangle(const json& value) {
  if (!value.is_array() || value.size() != 3) {
    return Invalid("vertices must contain exactly three vectors");
  }
  std::vector<Eigen::Vector3d> vertices;
  for (size_t i = 0; i < 3; ++i) {
    auto vertex = ParseVector3(value.at(i), "vertices");
    if (!vertex.ok()) return vertex.status();
    vertices.push_back(*vertex);
  }
  const Eigen::Vector3d a = vertices[1] - vertices[0];
  const Eigen::Vector3d b = vertices[2] - vertices[0];
  const Eigen::Vector3d cross = a.cross(b);
  const double norm = cross.norm();
  if (!a.allFinite() || !b.allFinite() || !cross.allFinite() ||
      !std::isfinite(norm) || norm <= 1e-12) {
    return Invalid("triangle vertices must be noncollinear with finite area");
  }
  return vertices;
}

std::string PhaseName(RelationPhase phase) {
  switch (phase) {
    case RelationPhase::kDormant:
      return "dormant";
    case RelationPhase::kActive:
      return "active";
    case RelationPhase::kDriving:
      return "driving";
    case RelationPhase::kContact:
      return "contact";
  }
  return "unknown";
}

// Source-only E observation path. The declared planar reference is NEVER
// supplied as an observation; body-anchor proximity does not require receiver
// locomotion, a synthetic root trajectory, or a whole-body controller.
absl::StatusOr<json> RunSourceAffordance(
    const json& entry, const std::string& case_id,
    const std::vector<Eigen::Matrix4d>& transforms, size_t* remaining_samples) {
  const auto fields =
      RequireExactKeys(entry,
                       {"case_id", "component", "transforms", "vertices",
                        "anchor_name", "affordance_tag", "samples"},
                       "E case");
  if (!fields.ok()) return fields;
  auto vertices = ParseTriangle(entry.at("vertices"));
  if (!vertices.ok()) return vertices.status();
  auto anchor = ParseCaseId(entry.at("anchor_name"));
  auto tag = ParseCaseId(entry.at("affordance_tag"));
  if (!anchor.ok()) return anchor.status();
  if (!tag.ok()) return tag.status();
  const std::string reference = "probe_unobserved_reference";
  if (*anchor == reference || anchor->find('\0') != std::string::npos ||
      tag->find('\0') != std::string::npos) {
    return Invalid(
        "E anchor/tag cannot contain NUL or use the reserved reference");
  }
  const auto& samples = entry.at("samples");
  if (!samples.is_array() || samples.empty() ||
      samples.size() > *remaining_samples) {
    return Invalid("E samples require 1..20000 rows in the complete input");
  }
  *remaining_samples -= samples.size();
  struct Sample {
    int64_t sequence;
    double time;
    Eigen::Vector3d position;
    double yaw;
    bool available;
  };
  std::vector<Sample> parsed;
  int64_t last_sequence = -1;
  double last_time = 0.0;
  for (const auto& sample : samples) {
    const auto keys = RequireExactKeys(
        sample, {"sequence", "time_seconds", "position", "yaw", "available"},
        "E sample");
    if (!keys.ok()) return keys;
    const auto& sequence = sample.at("sequence");
    if (!sequence.is_number_integer() || sequence < json(0) ||
        sequence > json(std::numeric_limits<int64_t>::max()) ||
        !sample.at("available").is_boolean()) {
      return Invalid(
          "E sequence must be nonnegative Int64; availability is boolean");
    }
    auto time = ParseFinite(sample.at("time_seconds"), "E time");
    auto point = ParseVector3(sample.at("position"), "E position");
    auto yaw = ParseFinite(sample.at("yaw"), "E yaw");
    if (!time.ok()) return time.status();
    if (!point.ok()) return point.status();
    if (!yaw.ok()) return yaw.status();
    const int64_t ordinal = sequence.get<int64_t>();
    if (ordinal <= last_sequence || *time < last_time) {
      return Invalid(
          "E source sequence must increase and source time cannot regress");
    }
    parsed.push_back(
        {ordinal, *time, *point, *yaw, sample.at("available").get<bool>()});
    last_sequence = ordinal;
    last_time = *time;
  }

  Space space;
  for (element_id_t id = 1; id <= 4; ++id) {
    if (!space.AddElement(id, id == 1 ? kInvalidElementId : id - 1) ||
        !space.SetElementTransform(id, transforms.at(id - 1))) {
      return absl::InternalError("E could not construct the source hierarchy");
    }
  }
  if (!space.SetElementSharedId(4, 1004) || !space.SetElementTag(4, *tag) ||
      !space.SetElementSpatialDefinition(
          4, std::make_shared<SurfaceSpatialDefinition>(
                 std::make_shared<Polygon3D>(*vertices)))) {
    return absl::InternalError(
        "E could not construct the baked source support");
  }
  const auto actor = space.CreateActor(
      {SubjectAnchor(reference),
       SubjectAnchor(*anchor, Eigen::Vector3d::Zero(),
                     {StateType::kPosition3D, StateType::kRotationYaw})});
  // Space::BindScoped refused a missing actor, anchor or scope-root element before binding; the root itself is gone, the
  // refusals and the error path are unchanged.
  const bool bindable = space.GetActor(actor) && space.GetActor(actor)->GetAnchor(*anchor) &&
                        space.GetElementTree()->GetElement(4);
  if (!space.SetActorElement(actor, 1) || !bindable ||
      !space.Bind(actor, *anchor, RelationKernelType::kAvatarAffordancePose, *tag)) {
    return absl::InternalError(
        "E could not bind the exact source anchor/support");
  }
  RelationPhase phase = RelationPhase::kDormant;
  space.RegisterOnRelationEvent([&](const RelationEvent& event) {
    if (event.anchor_name == *anchor && event.referent_shared_id == 1004 &&
        event.kernel_type == RelationKernelType::kAvatarAffordancePose) {
      phase = event.to;
    }
  });
  uint64_t epoch = 0;
  if (!space.BeginActorObservationEpoch(actor, reference, &epoch)) {
    return absl::InternalError(
        "E could not begin the source observation epoch");
  }
  json output = json::array();
  for (const auto& sample : parsed) {
    std::vector<Space::ActorObservation> observations;
    if (sample.available) {
      observations.push_back({*anchor, sample.position, sample.yaw});
    }
    if (!space.ApplyActorObservationFrame(actor, epoch, sample.sequence,
                                          sample.time, observations)) {
      return absl::InternalError("E native source observation was rejected");
    }
    uint64_t observed_epoch = 0;
    int64_t observed_sequence = -1;
    double observed_time = 0.0;
    bool has_sample = false;
    if (!space.GetActorObservationClock(actor, &observed_epoch,
                                        &observed_sequence, &observed_time,
                                        &has_sample) ||
        !has_sample || observed_epoch != epoch ||
        observed_sequence != sample.sequence || observed_time != sample.time) {
      return absl::InternalError(
          "E native observation clock changed source identity");
    }
    const auto states = space.GetRelationStates(actor);
    json row{{"sequence", sample.sequence},   {"time_seconds", sample.time},
             {"available", sample.available}, {"support_available", false},
             {"phase", "unavailable"},        {"distance", nullptr},
             {"proximity", nullptr},          {"support_point", nullptr}};
    if (!sample.available) {
      if (!states.empty())
        return absl::InternalError("E absent anchor produced a relation");
    } else if (!states.empty()) {
      if (states.size() != 1 || states[0].anchor_name != *anchor ||
          states[0].referent_element_id != 4 ||
          states[0].kernel_type != RelationKernelType::kAvatarAffordancePose ||
          !states[0].has_contact ||
          !states[0].contact_point_world.allFinite()) {
        return absl::InternalError(
            "E native result lacks the exact projected support");
      }
      const auto& state = states[0];
      const Eigen::Vector3d delta = sample.position - state.contact_point_world;
      const double distance = std::hypot(delta.x(), delta.y(), delta.z());
      if (!std::isfinite(distance) || !std::isfinite(state.proximity) ||
          state.proximity < 0.0 || state.proximity > 1.0) {
        return absl::InternalError("E native distance/proximity is invalid");
      }
      row["support_available"] = true;
      row["phase"] = PhaseName(phase);
      row["distance"] = distance;
      row["proximity"] = state.proximity;
      row["support_point"] = VectorJson(state.contact_point_world);
    }
    output.push_back(std::move(row));
  }
  return json{{"case_id", case_id},
              {"component", "E"},
              {"ok", true},
              {"anchor_name", *anchor},
              {"affordance_tag", *tag},
              {"root_observed", false},
              {"samples", output}};
}

absl::StatusOr<ElementTree> BuildTree(
    const std::vector<Eigen::Matrix4d>& transforms) {
  ElementTree tree;
  for (element_id_t id = 1; id <= 4; ++id) {
    const element_id_t parent = id == 1 ? kInvalidElementId : id - 1;
    const absl::Status status = tree.AddElement(id, parent);
    if (!status.ok()) return status;
    tree.GetElement(id)->SetLocalTransform(transforms.at(id - 1));
  }
  tree.GetElement(4)->SetSharedId(1004);
  return tree;
}

absl::StatusOr<std::string> ParseCaseId(const json& value) {
  if (!value.is_string()) return Invalid("case_id must be a string");
  const std::string id = value.get<std::string>();
  if (id.empty() || id.size() > 128)
    return Invalid("case_id must be nonempty and <=128 chars");
  return id;
}

absl::StatusOr<json> RunHierarchy(
    const json& entry, const std::string& case_id,
    const std::vector<Eigen::Matrix4d>& transforms) {
  const absl::Status fields = RequireExactKeys(
      entry, {"case_id", "component", "transforms", "points"}, "H case");
  if (!fields.ok()) return fields;
  const json& source_points = entry.at("points");
  if (!source_points.is_array() || source_points.empty() ||
      source_points.size() > kMaxPoints) {
    return Invalid("H points must contain 1..128 vectors");
  }
  std::vector<Eigen::Vector3d> points;
  points.reserve(source_points.size());
  for (size_t i = 0; i < source_points.size(); ++i) {
    auto point = ParseVector3(source_points.at(i),
                              "H points[" + std::to_string(i) + "]");
    if (!point.ok()) return point.status();
    points.push_back(*point);
  }
  auto tree_or = BuildTree(transforms);
  if (!tree_or.ok()) return tree_or.status();
  const Eigen::Matrix4d world = tree_or->GetWorldTransform(4);
  if (!world.allFinite())
    return absl::InternalError("H native world transform is nonfinite");
  json world_points = json::array();
  for (const Eigen::Vector3d& point : points) {
    const Eigen::Vector3d mapped =
        (world * Eigen::Vector4d(point.x(), point.y(), point.z(), 1.0))
            .head<3>();
    if (!mapped.allFinite())
      return absl::InternalError("H native world point is nonfinite");
    world_points.push_back(VectorJson(mapped));
  }
  return json{{"case_id", case_id},
              {"component", "H"},
              {"ok", true},
              {"world_transform", MatrixJson(world)},
              {"world_points", world_points}};
}

absl::StatusOr<json> RunR1(const json& entry, const std::string& case_id,
                           const std::vector<Eigen::Matrix4d>& transforms) {
  const absl::Status fields =
      RequireExactKeys(entry,
                       {"case_id", "component", "transforms", "vertices",
                        "angle", "radial_ratio", "relative_yaw"},
                       "R1 case");
  if (!fields.ok()) return fields;
  const json& source_vertices = entry.at("vertices");
  if (!source_vertices.is_array() || source_vertices.size() != 3) {
    return Invalid("R1 vertices must contain exactly three vectors");
  }
  std::vector<Eigen::Vector3d> vertices;
  vertices.reserve(3);
  for (size_t i = 0; i < 3; ++i) {
    auto vertex = ParseVector3(source_vertices.at(i),
                               "R1 vertices[" + std::to_string(i) + "]");
    if (!vertex.ok()) return vertex.status();
    vertices.push_back(*vertex);
  }
  const Eigen::Vector3d first_edge = vertices[1] - vertices[0];
  const Eigen::Vector3d second_edge = vertices[2] - vertices[0];
  const Eigen::Vector3d double_area = first_edge.cross(second_edge);
  const double double_area_norm = double_area.norm();
  if (!first_edge.allFinite() || !second_edge.allFinite() ||
      !double_area.allFinite() || !std::isfinite(double_area_norm) ||
      double_area_norm <= 1e-12) {
    return Invalid("R1 triangle vertices must be noncollinear");
  }
  auto angle = ParseFinite(entry.at("angle"), "R1 angle");
  auto ratio = ParseFinite(entry.at("radial_ratio"), "R1 radial_ratio");
  auto yaw = ParseFinite(entry.at("relative_yaw"), "R1 relative_yaw");
  if (!angle.ok()) return angle.status();
  if (!ratio.ok()) return ratio.status();
  if (!yaw.ok()) return yaw.status();

  auto tree_or = BuildTree(transforms);
  if (!tree_or.ok()) return tree_or.status();
  tree_or->GetElement(4)->SetSpatialDefinition(
      std::make_shared<SurfaceSpatialDefinition>(
          std::make_shared<Polygon3D>(vertices)));
  RelationPayload payload;
  payload.kernel_type = RelationKernelType::kAvatarAffordancePose;
  payload.resolved_local_element_id = 4;
  payload.referent_shared_id = 1004;
  payload.subject_anchor_name = "probe";
  payload.support.has_contact = true;
  payload.support.contact_normalized.strategy =
      NormalizedPolygonPosition::Strategy::kRadial;
  payload.support.contact_normalized.angle = *angle;
  payload.support.contact_normalized.radial_ratio = *ratio;
  payload.grounded_relation.encoded_state = Eigen::VectorXd(3);
  payload.grounded_relation.encoded_state << 0.0, 0.0, *yaw;

  RelationSolver solver;
  auto grounded = solver.GroundAffordanceTarget(payload, *tree_or);
  if (!grounded.ok()) {
    return json{{"case_id", case_id},
                {"component", "R1"},
                {"ok", false},
                {"native_status", static_cast<int>(grounded.status().code())}};
  }
  if (grounded->mode != AffordanceGroundingMode::kWholePlanarBoundary ||
      !grounded->position.allFinite() || !std::isfinite(grounded->yaw)) {
    return absl::InternalError(
        "R1 native result is nonfinite or not whole planar boundary");
  }
  return json{
      {"case_id", case_id},   {"component", "R1"},
      {"ok", true},           {"position", VectorJson(grounded->position)},
      {"yaw", grounded->yaw}, {"mode", "whole_planar_boundary"}};
}

}  // namespace

absl::StatusOr<json> RunGeometryComponentProbe(const json& input) {
  const absl::Status root_fields =
      RequireExactKeys(input, {"schema_version", "cases"}, "input");
  if (!root_fields.ok()) return root_fields;
  if (!input.at("schema_version").is_number_integer() ||
      input.at("schema_version") != json(kSchemaVersion)) {
    return Invalid("schema_version must be integer 1");
  }
  const json& cases = input.at("cases");
  if (!cases.is_array() || cases.empty() || cases.size() > kMaxCases) {
    return Invalid("cases must contain 1..256 entries");
  }
  std::set<std::string> ids;
  size_t remaining_samples = kMaxSourceSamples;
  json results = json::array();
  for (size_t index = 0; index < cases.size(); ++index) {
    const json& entry = cases.at(index);
    if (!entry.is_object() || !entry.contains("case_id") ||
        !entry.contains("component") || !entry.contains("transforms")) {
      return Invalid("case " + std::to_string(index) + " lacks common fields");
    }
    auto case_id = ParseCaseId(entry.at("case_id"));
    if (!case_id.ok()) return case_id.status();
    if (!ids.insert(*case_id).second)
      return Invalid("case_id values must be unique");
    if (!entry.at("component").is_string())
      return Invalid("component must be a string");
    auto transforms = ParseTransforms(entry.at("transforms"));
    if (!transforms.ok()) return transforms.status();
    const std::string component = entry.at("component").get<std::string>();
    absl::StatusOr<json> result =
        component == "H"    ? RunHierarchy(entry, *case_id, *transforms)
        : component == "R1" ? RunR1(entry, *case_id, *transforms)
        : component == "E"
            ? RunSourceAffordance(entry, *case_id, *transforms,
                                  &remaining_samples)
            : absl::StatusOr<json>(Invalid("component must be H, E or R1"));
    if (!result.ok()) return result.status();
    results.push_back(*result);
  }
  return json{{"schema_version", kSchemaVersion}, {"results", results}};
}

}  // namespace spatialblend::harness
