// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

#include "spatialblend/geometry/radial_surface_chart.h"
#include "spatialblend/space/space.h"

namespace spatialblend {
namespace {

using Evidence = SourceInteractionEvidence;

bool BoundedName(const std::string& value) {
  return value.size() < Evidence::kMaxNameBytes &&
         value.find('\0') == std::string::npos;
}

bool ValidFrame(const Eigen::Matrix4d& frame) {
  const double determinant = frame.determinant();
  return frame.allFinite() && std::isfinite(determinant) &&
         std::abs(determinant) > 1e-12 &&
         frame.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-12);
}

bool CopyPath(const ElementTree& tree, element_id_t referent,
              size_t* total_nodes, std::vector<Evidence::HierarchyNode>* out) {
  // Bound the traversal itself, not only a previously allocated ancestor list.
  std::vector<element_id_t> reverse_path;
  for (auto id = referent; id != kInvalidElementId;) {
    if (reverse_path.size() == Evidence::kMaxPathDepth) return false;
    const auto* element = tree.GetElement(id);
    if (!element) return false;
    reverse_path.push_back(id);
    id = element->GetParentId();
  }
  if (reverse_path.empty() ||
      reverse_path.size() > Evidence::kMaxPathNodes - *total_nodes)
    return false;
  for (auto it = reverse_path.rbegin(); it != reverse_path.rend(); ++it) {
    const auto* element = tree.GetElement(*it);
    Evidence::HierarchyNode node;
    node.element = *it;
    node.parent = element->GetParentId();
    node.shared = element->GetSharedId();  // No local-id/shared-id aliasing.
    node.tag = element->GetTag();
    node.geometry_role = element->GetGeometryRole();
    node.world_transform = tree.GetWorldTransform(*it);
    if (!BoundedName(node.tag) || !ValidFrame(node.world_transform))
      return false;
    out->push_back(std::move(node));
  }
  *total_nodes += reverse_path.size();
  return true;
}

bool CopySupport(const Support& support, const Evidence::Anchor& anchor,
                 size_t* total_points, Evidence::Relation* out) {
  if (!ValidFrame(support.world_transform) ||
      !BoundedName(support.sub_region_name))
    return false;
  out->support_to_world = support.world_transform;
  out->sub_region = support.sub_region_name;
  if (support.boundary) {
    const auto& points = support.boundary->GetPoints();
    if (points.size() > Evidence::kMaxBoundaryPoints - *total_points)
      return false;
    for (const auto& point : points)
      if (!point.allFinite()) return false;
    out->support_boundary = points;
    *total_points += points.size();
  }
  out->has_projection = support.has_contact;
  if (!out->has_projection) return true;
  out->projected_point = support.contact_point_world;
  out->anchor_support_distance =
      (anchor.position - out->projected_point).norm();
  out->support_angle = support.contact_normalized.angle;
  out->support_radial_ratio = support.contact_normalized.radial_ratio;
  if (!out->projected_point.allFinite() ||
      !std::isfinite(out->anchor_support_distance) ||
      !std::isfinite(out->support_angle) ||
      !std::isfinite(out->support_radial_ratio))
    return false;

  // Projection is not contact, nor does every support describe a bounded
  // planar affordance (e.g. a volume's top projected into its footprint).
  // Require the existing radial chart domain and the actual point on that
  // chart plane. The caller must STILL qualify a body anchor and Sittable
  // referent; this geometry flag alone never selects a seat or admits motion.
  if (!support.boundary ||
      !RadialSurfaceChart::Origin(*support.boundary).ok() ||
      support.contact_normalized.strategy !=
          NormalizedPolygonPosition::Strategy::kRadial ||
      !support.contact_normalized.mvc_weights.empty() ||
      support.contact_normalized.height_ratio != 0.0)
    return true;
  const Eigen::Vector4d local =
      support.world_transform.inverse() *
      Eigen::Vector4d(out->projected_point.x(), out->projected_point.y(),
                      out->projected_point.z(), 1.0);
  if (!local.allFinite() || std::abs(local.z()) > 1e-8) return true;
  const Eigen::Vector2d point = local.head<2>();
  out->bounded_planar_support =
      support.boundary->Contains(point) ||
      (support.boundary->GetNearestPoint(point) - point).norm() <= 1e-8;
  return true;
}

}  // namespace

bool Space::EnableSourceInteractionEvidence(spatialblend::entity_id_t actor_id,
                                            uint64_t epoch, bool enabled) {
  std::lock_guard<std::mutex> lock(space_mutex_);
  return EnableSourceInteractionEvidenceLocked(actor_id, epoch, enabled);
}

Space::SourceEvidenceReadResult Space::TryEnableSourceInteractionEvidence(
    spatialblend::entity_id_t actor_id, uint64_t epoch, bool enabled) {
  std::unique_lock<std::mutex> lock(space_mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return SourceEvidenceReadResult::kBusy;
  return EnableSourceInteractionEvidenceLocked(actor_id, epoch, enabled)
             ? SourceEvidenceReadResult::kOk
             : SourceEvidenceReadResult::kUnavailable;
}

bool Space::EnableSourceInteractionEvidenceLocked(
    spatialblend::entity_id_t actor_id, uint64_t epoch, bool enabled) {
  const auto clock = observation_clocks_.find(actor_id);
  const auto* actor = FindActorLocked(actor_id);
  if (!actor || receiver_inputs_.count(actor_id) || epoch == 0 ||
      clock == observation_clocks_.end() || clock->second.epoch != epoch)
    return false;
  if (!enabled) {
    source_evidence_enabled_.erase(actor_id);
    source_evidence_phase_dirty_.erase(actor_id);
    source_evidence_.erase(actor_id);
    return true;
  }
  if (space_id_ == kInvalidSpaceId ||
      !element_tree_.GetElement(actor->GetElementId()) ||
      borrowed_binding_actors_.count(actor_id) ||
      !CaptureSceneInputStampLocked().known)
    return false;
  // Idempotent opt-in does not recapture held input, reset phase or restamp
  // age.
  if (source_evidence_enabled_.insert(actor_id).second)
    source_evidence_phase_dirty_.insert(actor_id);
  return true;
}

void Space::InvalidateSourceEvidencePhaseLocked(
    spatialblend::entity_id_t actor_id) {
  source_evidence_.erase(actor_id);
  if (source_evidence_enabled_.count(actor_id))
    source_evidence_phase_dirty_.insert(actor_id);
}

void Space::CaptureSourceInteractionEvidenceLocked(
    spatialblend::entity_id_t actor_id) {
  source_evidence_.erase(actor_id);
  if (!source_evidence_enabled_.count(actor_id) ||
      source_evidence_phase_dirty_.count(actor_id) ||
      borrowed_binding_actors_.count(actor_id) ||
      receiver_inputs_.count(actor_id))
    return;
  const auto* actor = FindActorLocked(actor_id);
  const auto clock_it = observation_clocks_.find(actor_id);
  const auto scene = CaptureSceneInputStampLocked();
  if (!actor || clock_it == observation_clocks_.end() || !scene.known ||
      space_id_ == kInvalidSpaceId)
    return;
  const auto& clock = clock_it->second;
  const auto* owner = element_tree_.GetElement(actor->GetElementId());
  if (!owner || !clock.has_sample || !clock.anchors_from_committed_frame ||
      clock.epoch == 0 || clock.sequence < 0 || clock.receipt_monotonic_ns <= 0)
    return;
  Evidence value;
  value.space = space_id_;
  value.actor = actor_id;
  value.actor_element = owner->GetId();
  value.actor_shared = owner->GetSharedId();
  value.epoch = clock.epoch;
  value.sequence = clock.sequence;
  value.source_time_seconds = clock.time;
  value.receipt_monotonic_ns = clock.receipt_monotonic_ns;
  value.tree_revision = scene.tree;
  value.support_revision = scene.support;
  const auto& anchors = actor->GetAnchors();
  if (anchors.empty() || anchors.size() > Evidence::kMaxAnchors) return;
  bool root_declared = false;
  for (const auto& declaration : anchors) {
    Evidence::Anchor anchor;
    anchor.name = declaration.name;
    if (anchor.name.empty() || !BoundedName(anchor.name)) return;
    anchor.locomotion_root = anchor.name == clock.root;
    root_declared |= anchor.locomotion_root;
    anchor.position_dimension =
        declaration.HasPosition3D()
            ? 3
            : (std::find(declaration.dof.begin(), declaration.dof.end(),
                         StateType::kPosition2D) != declaration.dof.end()
                   ? 2
                   : 0);
    anchor.observed = actor->IsAnchorActive(anchor.name);
    if (anchor.observed) {
      const auto* pose = actor->GetAnchorWorldPoseOverride(anchor.name);
      if (!pose || !pose->position.allFinite() || !std::isfinite(pose->yaw))
        return;  // No local-offset or generated-pose fallback.
      anchor.position = pose->position;
      anchor.yaw = pose->yaw;
    }
    value.anchors.push_back(std::move(anchor));
  }
  if (!root_declared) return;

  using PhaseKey =
      std::tuple<std::string, RelationKernelType, shared_element_id_t>;
  std::map<PhaseKey, size_t> phase_owners;
  size_t total_nodes = 0, total_points = 0;
  const auto bindings = bindings_.find(actor_id);
  if (bindings != bindings_.end()) {
    for (size_t i = 0; i < bindings->second.size(); ++i) {
      const auto& binding = bindings->second[i];
      if (binding.binding_ordinal != i ||
          binding.anchor_ordinal == Binding::kInvalidOrdinal ||
          binding.anchor_ordinal >= value.anchors.size() ||
          value.anchors[binding.anchor_ordinal].name != binding.anchor_name ||
          !BoundedName(binding.tag_filter))
        return;
      double sum = 0.0;
      for (const auto& match : binding.latest_matches)
        sum += match.eval.grounded.proximity;
      if (!std::isfinite(sum)) return;
      // Use the exact evaluated rows: never re-ground, advance phase, run a
      // kernel or normalize a new candidate set in this capture/read path.
      for (const auto& match : binding.latest_matches) {
        if (value.relations.size() == Evidence::kMaxRelations ||
            !value.anchors[binding.anchor_ordinal].observed)
          return;
        Evidence::Relation row;
        row.binding_ordinal = binding.binding_ordinal;
        row.anchor_ordinal = binding.anchor_ordinal;
        row.kernel = binding.kernel_type;
        row.explicit_scope = binding.explicit_scope_root;
        row.tag_filter = binding.tag_filter;
        row.proximity = match.eval.grounded.proximity;
        row.normalized_weight = sum > 1e-12 ? row.proximity / sum : 0.0;
        if (!std::isfinite(row.proximity) || row.proximity < 0 ||
            row.proximity > 1 || !std::isfinite(row.normalized_weight) ||
            !CopyPath(element_tree_, match.element_id, &total_nodes,
                      &row.path) ||
            !CopySupport(match.eval.support, value.anchors[row.anchor_ordinal],
                         &total_points, &row))
          return;
        const auto& referent = row.path.back();
        row.phase_referent_id =
            referent.shared != kInvalidSharedElementId
                ? referent.shared
                : static_cast<shared_element_id_t>(referent.element);
        RelationEvent identity;
        identity.actor_shared_id = GetActorSharedIdForEvents(actor_id);
        identity.anchor_name = binding.anchor_name;
        identity.kernel_type = binding.kernel_type;
        identity.referent_shared_id = row.phase_referent_id;
        row.phase =
            phase_tracker_.GetPhaseForOwner(identity, PhaseOwner(actor_id));
        ++phase_owners[{binding.anchor_name, binding.kernel_type,
                        row.phase_referent_id}];
        value.relations.push_back(std::move(row));
      }
    }
  }
  for (auto& row : value.relations) {
    row.phase_identity_unambiguous =
        phase_owners.at({value.anchors[row.anchor_ordinal].name, row.kernel,
                         row.phase_referent_id}) == 1;
  }
  source_evidence_.emplace(actor_id, std::move(value));
}

Space::SourceEvidenceReadResult Space::TryCopySourceInteractionEvidence(
    spatialblend::entity_id_t actor_id, uint64_t epoch, int64_t sequence,
    int64_t maximum_age_ns, SourceInteractionEvidence* out) const {
  if (!out) return SourceEvidenceReadResult::kUnavailable;
  *out = {};
  if (epoch == 0 || sequence < 0 || maximum_age_ns <= 0 ||
      maximum_age_ns > Evidence::kMaximumAgeNs)
    return SourceEvidenceReadResult::kUnavailable;
  std::unique_lock<std::mutex> lock(space_mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return SourceEvidenceReadResult::kBusy;
  const auto* actor = FindActorLocked(actor_id);
  const auto clock_it = observation_clocks_.find(actor_id);
  if (!actor || receiver_inputs_.count(actor_id) ||
      !source_evidence_enabled_.count(actor_id) ||
      source_evidence_phase_dirty_.count(actor_id) ||
      borrowed_binding_actors_.count(actor_id) ||
      clock_it == observation_clocks_.end())
    return SourceEvidenceReadResult::kUnavailable;
  const auto& clock = clock_it->second;
  if (clock.epoch != epoch || clock.sequence != sequence)
    return SourceEvidenceReadResult::kStale;
  const auto captured = source_evidence_.find(actor_id);
  const auto scene = CaptureSceneInputStampLocked();
  if (!scene.known || !clock.anchors_from_committed_frame ||
      captured == source_evidence_.end())
    return SourceEvidenceReadResult::kUnavailable;
  const auto& value = captured->second;
  const int64_t now = receiver_input_clock_();
  if (value.space != space_id_ ||
      value.actor_element != actor->GetElementId() || value.epoch != epoch ||
      value.sequence != sequence || value.tree_revision != scene.tree ||
      value.support_revision != scene.support ||
      now < value.receipt_monotonic_ns ||
      now - value.receipt_monotonic_ns >= maximum_age_ns)
    return SourceEvidenceReadResult::kStale;
  *out = value;
  return SourceEvidenceReadResult::kOk;
}

}  // namespace spatialblend
