// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/net/proto_util.h"

#include <cmath>
#include <limits>

#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/relation/grounded_relation.h"

namespace spatialblend {

// ---------------------------------------------------------------------------
// Support <-> SupportProto
// ---------------------------------------------------------------------------

proto::SupportProto ToProto(const Support& support) {
  proto::SupportProto sp;

  // world_transform: 4x4 Matrix4d stored column-major (16 doubles).
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 4; ++row) {
      sp.add_world_transform(support.world_transform(row, col));
    }
  }

  // boundary: flat x1,y1,x2,y2,...
  if (support.boundary) {
    for (const auto& pt : support.boundary->GetPoints()) {
      sp.add_boundary_points(pt.x());
      sp.add_boundary_points(pt.y());
    }
  }

  sp.set_has_contact(support.has_contact);

  // contact_normalized (NormalizedPolygonPosition), faithfully serialized.
  const auto& cn = support.contact_normalized;
  sp.set_contact_strategy(static_cast<int32_t>(cn.strategy));
  sp.set_contact_angle(cn.angle);
  sp.set_contact_radial_ratio(cn.radial_ratio);
  for (double w : cn.mvc_weights) sp.add_contact_mvc_weights(w);
  sp.set_contact_height_ratio(cn.height_ratio);
  sp.set_contact_chart_version(cn.chart_version);
  sp.set_has_contact_boundary_arclen(cn.boundary_arclen >= 0.0);
  sp.set_contact_boundary_arclen(cn.boundary_arclen >= 0.0 ? cn.boundary_arclen
                                                            : 0.0);
  sp.set_has_standoff_fraction(cn.standoff_fraction >= 0.0);
  sp.set_standoff_fraction(cn.standoff_fraction >= 0.0 ? cn.standoff_fraction
                                                       : 0.0);
  sp.set_has_standoff_angle(cn.has_standoff_angle);
  sp.set_standoff_angle(cn.standoff_angle);
  sp.set_container_self(support.container_self);
  sp.set_look_at_vertex_plus_one(static_cast<uint32_t>(support.look_at_vertex + 1));
  sp.set_look_at_boundary_crossed(support.look_at_boundary_crossed);
  if (support.has_border_point) {
    sp.add_border_point(support.border_point.x());
    sp.add_border_point(support.border_point.y());
  }
  if (support.has_approach_origin) {
    sp.add_approach_origin(support.approach_origin_local.x());
    sp.add_approach_origin(support.approach_origin_local.y());
  }
  sp.set_has_edge_depth(support.edge_depth_m >= 0.0);
  sp.set_edge_depth_m(support.edge_depth_m >= 0.0 ? support.edge_depth_m : 0.0);

  // Canonical interior chart coordinate.
  sp.set_has_interior(support.has_interior);
  sp.set_interior_kind(static_cast<int32_t>(support.interior.kind));
  sp.add_interior_uvw(support.interior.uvw.x());
  sp.add_interior_uvw(support.interior.uvw.y());
  sp.add_interior_uvw(support.interior.uvw.z());
  for (double w : support.interior.weights) sp.add_interior_weights(w);

  // Named sub-region this support is grounded on.
  sp.set_sub_region_name(support.sub_region_name);
  sp.set_sub_region_chart(support.sub_region_chart);

  // Contact point in the SOURCE referent's own local frame (see
  // SupportProto's doc comment).
  sp.add_contact_point_local(support.contact_point_local.x());
  sp.add_contact_point_local(support.contact_point_local.y());

  return sp;
}

Support SupportFromProto(const proto::SupportProto& proto) {
  Support support;

  // world_transform
  if (proto.world_transform_size() == 16) {
    int idx = 0;
    for (int col = 0; col < 4; ++col) {
      for (int row = 0; row < 4; ++row) {
        support.world_transform(row, col) = proto.world_transform(idx++);
      }
    }
  }

  // boundary
  if (proto.boundary_points_size() >= 4) {
    std::vector<Eigen::Vector2d> points;
    points.reserve(proto.boundary_points_size() / 2);
    for (int i = 0; i + 1 < proto.boundary_points_size(); i += 2) {
      points.emplace_back(proto.boundary_points(i),
                          proto.boundary_points(i + 1));
    }
    support.boundary = std::make_shared<Polygon2D>(points);
  }

  support.has_contact = proto.has_contact();

  // contact_normalized (NormalizedPolygonPosition).
  auto& cn = support.contact_normalized;
  cn.strategy = static_cast<spatialblend::NormalizedPolygonPosition::Strategy>(
      proto.contact_strategy());
  cn.angle = proto.contact_angle();
  cn.radial_ratio = proto.contact_radial_ratio();
  cn.mvc_weights.assign(proto.contact_mvc_weights().begin(),
                        proto.contact_mvc_weights().end());
  cn.height_ratio = proto.contact_height_ratio();
  cn.chart_version = proto.contact_chart_version();
  cn.boundary_arclen = proto.has_contact_boundary_arclen()
                           ? proto.contact_boundary_arclen()
                           : -1.0;
  cn.standoff_fraction = proto.has_standoff_fraction()
                             ? proto.standoff_fraction()
                             : -1.0;
  cn.has_standoff_angle = proto.has_standoff_angle();
  cn.standoff_angle = proto.standoff_angle();
  support.edge_depth_m =
      proto.has_edge_depth() ? proto.edge_depth_m() : -1.0;
  support.container_self = proto.container_self();
  support.look_at_vertex = static_cast<int32_t>(proto.look_at_vertex_plus_one()) - 1;
  support.look_at_boundary_crossed = proto.look_at_boundary_crossed();
  if (proto.border_point_size() == 2) {
    support.border_point = Eigen::Vector2d(proto.border_point(0), proto.border_point(1));
    support.has_border_point = true;
  }
  if (proto.approach_origin_size() == 2) {
    support.approach_origin_local =
        Eigen::Vector2d(proto.approach_origin(0), proto.approach_origin(1));
    support.has_approach_origin = true;
  }

  // Canonical interior chart coordinate.
  support.has_interior = proto.has_interior();
  support.interior.kind =
      static_cast<NormalizedInterior::Kind>(proto.interior_kind());
  if (proto.interior_uvw_size() == 3) {
    support.interior.uvw = Eigen::Vector3d(proto.interior_uvw(0),
                                           proto.interior_uvw(1),
                                           proto.interior_uvw(2));
  }
  support.interior.weights.assign(proto.interior_weights().begin(),
                                  proto.interior_weights().end());

  // Named sub-region this support is grounded on.
  support.sub_region_name = proto.sub_region_name();
  support.sub_region_chart = proto.sub_region_chart();

  // Contact point in the SOURCE referent's own local frame.
  if (proto.contact_point_local_size() == 2) {
    support.contact_point_local = Eigen::Vector2d(
        proto.contact_point_local(0), proto.contact_point_local(1));
  }

  return support;
}

// ---------------------------------------------------------------------------
// RelationPayload <-> RelationPayloadProto
// ---------------------------------------------------------------------------

std::optional<proto::RelationPayloadProto> TryToProtoLegacy(
    const RelationPayload& payload) {
  if (!payload.enabled || payload.observed_facing_point) return std::nullopt;
  proto::RelationPayloadProto pp;

  pp.set_referent_shared_id(payload.referent_shared_id);
  pp.set_anchor_name(payload.subject_anchor_name);

  // encoded_state: Eigen::VectorXd -> repeated double
  for (int i = 0; i < payload.grounded_relation.encoded_state.size(); ++i) {
    pp.add_encoded_state(payload.grounded_relation.encoded_state[i]);
  }

  pp.set_proximity(payload.grounded_relation.proximity);
  if (payload.grounded_relation.support_distance_m >= 0.0)
    pp.set_support_distance_m(payload.grounded_relation.support_distance_m);
  *pp.mutable_support() = ToProto(payload.support);
  pp.set_kernel_type(static_cast<int32_t>(payload.kernel_type));
  pp.set_estimated_time(payload.estimated_time);
  pp.set_referent_source_space_id(payload.referent_source_space_id);
  pp.set_referent_source_element_id(payload.referent_source_element_id);
  pp.set_source_scope_diameter_m(payload.grounded_relation.source_scope_diameter_m);
  pp.set_scope_shared_id(payload.scope_shared_id);
  pp.set_scope_source_space_id(payload.scope_source_space_id);
  pp.set_scope_source_element_id(payload.scope_source_element_id);
  if (payload.source_binding_ordinal != UINT64_MAX)
    pp.set_source_binding_ordinal(payload.source_binding_ordinal);
  if (payload.binding_revision_known)
    pp.set_binding_revision(payload.binding_revision);

  // AffordancePose only; NaN (the GroundedRelation default,
  // meaning "unknown") is never serialized, matching the proto's presence
  // semantics.
  if (std::isfinite(payload.grounded_relation.ap_offset_u_m))
    pp.set_ap_offset_u_m(payload.grounded_relation.ap_offset_u_m);
  if (std::isfinite(payload.grounded_relation.ap_offset_v_m))
    pp.set_ap_offset_v_m(payload.grounded_relation.ap_offset_v_m);
  if (std::isfinite(payload.grounded_relation.ap_height_m))
    pp.set_ap_height_m(payload.grounded_relation.ap_height_m);
  if (std::isfinite(payload.update_count)) pp.set_update_count(payload.update_count);
  if (payload.facing_rotation_sign != 0) pp.set_facing_rotation_sign(payload.facing_rotation_sign);
  if (payload.facing_opening) pp.set_facing_opening(true);
  if (!payload.affordance_pose_tag.empty()) pp.set_affordance_pose_tag(payload.affordance_pose_tag);
  if (payload.has_subject_orientation) {
    pp.set_subject_orientation_type(payload.subject_orientation.type);
    pp.set_subject_pitch(payload.subject_orientation.pitch);
    pp.set_subject_roll(payload.subject_orientation.roll);
  }

  return pp;
}

RelationPayload PayloadFromProto(const proto::RelationPayloadProto& proto) {
  RelationPayload payload;

  payload.referent_shared_id = proto.referent_shared_id();
  payload.subject_anchor_name = proto.anchor_name();

  // encoded_state
  if (proto.encoded_state_size() > 0) {
    payload.grounded_relation.encoded_state.resize(proto.encoded_state_size());
    for (int i = 0; i < proto.encoded_state_size(); ++i) {
      payload.grounded_relation.encoded_state[i] = proto.encoded_state(i);
    }
  }

  payload.grounded_relation.proximity = proto.proximity();
  payload.grounded_relation.support_distance_m =
      proto.has_support_distance_m() ? proto.support_distance_m() : -1.0;
  payload.support = SupportFromProto(proto.support());
  payload.kernel_type = static_cast<RelationKernelType>(proto.kernel_type());
  payload.estimated_time = proto.estimated_time();
  payload.update_count = proto.has_update_count()
                             ? proto.update_count()
                             : std::numeric_limits<double>::infinity();
  payload.facing_rotation_sign = proto.has_facing_rotation_sign() ? proto.facing_rotation_sign() : 0;
  payload.facing_opening = proto.has_facing_opening() && proto.facing_opening();
  if (proto.has_affordance_pose_tag()) payload.affordance_pose_tag = proto.affordance_pose_tag();
  if (proto.has_subject_orientation_type()) {
    payload.has_subject_orientation = true;
    payload.subject_orientation.type = proto.subject_orientation_type();
    payload.subject_orientation.pitch = proto.subject_pitch();
    payload.subject_orientation.roll = proto.subject_roll();
  }
  payload.referent_source_space_id = proto.referent_source_space_id();
  payload.referent_source_element_id = proto.referent_source_element_id();
  payload.grounded_relation.source_scope_diameter_m =
      proto.source_scope_diameter_m();
  // An older sender (without these fields) leaves all three fields at the proto default 0; that is
  // "unknown", not element 0 of space 0 (current senders always write -1 for
  // an unknown scope).
  if (proto.scope_shared_id() != 0 || proto.scope_source_space_id() != 0 ||
      proto.scope_source_element_id() != 0) {
    payload.scope_shared_id = proto.scope_shared_id();
    payload.scope_source_space_id = proto.scope_source_space_id();
    payload.scope_source_element_id = proto.scope_source_element_id();
  }
  if (proto.has_source_binding_ordinal())
    payload.source_binding_ordinal = proto.source_binding_ordinal();
  if (proto.has_binding_revision()) {
    payload.binding_revision = proto.binding_revision();
    payload.binding_revision_known = true;
  }

  // Absent (a pre-pose-fit sender) leaves the GroundedRelation default
  // of NaN ("unknown"), never 0.
  if (proto.has_ap_offset_u_m())
    payload.grounded_relation.ap_offset_u_m = proto.ap_offset_u_m();
  if (proto.has_ap_offset_v_m())
    payload.grounded_relation.ap_offset_v_m = proto.ap_offset_v_m();
  if (proto.has_ap_height_m())
    payload.grounded_relation.ap_height_m = proto.ap_height_m();

  return payload;
}

// ---------------------------------------------------------------------------
// RemoteRelationUpdate <-> RelationUpdateProto
// ---------------------------------------------------------------------------

proto::RelationUpdateProto ToProto(const RemoteRelationUpdate& update) {
  proto::RelationUpdateProto up;

  up.set_source_space_id(update.source_space_id);
  up.set_actor_shared_id(update.actor_shared_id);
  up.set_step_duration(update.step_duration);
  up.set_actor_source_space_id(update.actor_source_space_id);
  up.set_actor_source_element_id(update.actor_source_element_id);
  up.set_sequence(update.sequence);
  up.set_source_clock(update.source_clock);
  switch (update.timing) {
    case RelationUpdateTiming::kLegacy:
      break;  // absent = legacy (what an older receiver assumes anyway)
    case RelationUpdateTiming::kTimestampedObservation:
      up.set_timing(
          proto::RELATION_UPDATE_TIMING_TIMESTAMPED_OBSERVATION);
      break;
    case RelationUpdateTiming::kObservationReset:
      up.set_timing(proto::RELATION_UPDATE_TIMING_OBSERVATION_RESET);
      break;
  }
  if (update.timing != RelationUpdateTiming::kLegacy) {
    up.set_source_time(update.source_time);
    up.set_source_epoch(update.source_epoch);
  }
  if (update.quiet_since_last_send) up.set_quiet_since_last_send(true);
  if (update.containment) {
    const ContainmentObservation& c = *update.containment;
    proto::ContainmentObservationProto* cp = up.mutable_containment();
    cp->set_scope_shared_id(c.scope_shared_id);
    cp->set_scope_source_space_id(c.scope_source_space_id);
    cp->set_scope_source_element_id(c.scope_source_element_id);
  }
  for (const auto& a : update.anchor_charts) {
    proto::AnchorChartProto* ap = up.add_anchor_charts();
    ap->set_anchor_name(a.anchor_name);
    ap->set_scope_shared_id(a.scope_shared_id);
    ap->set_scope_source_space_id(a.scope_source_space_id);
    ap->set_scope_source_element_id(a.scope_source_element_id);
    ap->set_x(a.x);
    ap->set_y(a.y);
    ap->set_z(a.z);
    ap->set_yaw(a.yaw);
  }

  for (const auto& payload : update.payloads) {
    // `enabled=false` is a native-only semantic with no negotiated proto field.
    // Fail closed: omission releases it at a legacy receiver instead of
    // silently re-enabling it after a lossy round trip.
    auto encoded = TryToProtoLegacy(payload);
    if (encoded) *up.add_payloads() = std::move(*encoded);
  }
  if (update.has_exception) {
    up.set_exception_id(update.exception_id);
    up.set_exception_payload(std::string(update.exception_payload.begin(),
                                         update.exception_payload.end()));
  }
  if (update.source_orientation_type != 0) {
    up.set_source_orientation_type(update.source_orientation_type);
    up.set_source_pitch(update.source_pitch);
    up.set_source_roll(update.source_roll);
  }

  return up;
}

RemoteRelationUpdate FromProto(const proto::RelationUpdateProto& proto) {
  RemoteRelationUpdate update;

  update.source_space_id = proto.source_space_id();
  update.actor_shared_id = proto.actor_shared_id();
  update.step_duration = proto.step_duration();
  update.actor_source_space_id = proto.actor_source_space_id();
  update.actor_source_element_id = proto.actor_source_element_id();
  update.sequence = proto.sequence();
  update.source_clock = proto.source_clock();
  // Absent or unknown timing = legacy (an older or newer sender).
  if (proto.has_timing()) {
    switch (proto.timing()) {
      case proto::RELATION_UPDATE_TIMING_TIMESTAMPED_OBSERVATION:
        update.timing = RelationUpdateTiming::kTimestampedObservation;
        break;
      case proto::RELATION_UPDATE_TIMING_OBSERVATION_RESET:
        update.timing = RelationUpdateTiming::kObservationReset;
        break;
      default:
        break;
    }
  }
  if (update.timing != RelationUpdateTiming::kLegacy) {
    update.source_time = proto.has_source_time() ? proto.source_time() : 0.0;
    update.source_epoch = proto.has_source_epoch() ? proto.source_epoch() : 0;
  }
  update.quiet_since_last_send =
      proto.has_quiet_since_last_send() && proto.quiet_since_last_send();
  if (proto.has_containment()) {
    const proto::ContainmentObservationProto& cp = proto.containment();
    ContainmentObservation c;
    c.scope_shared_id = cp.scope_shared_id();
    c.scope_source_space_id = cp.scope_source_space_id();
    c.scope_source_element_id = cp.scope_source_element_id();
    update.containment = c;
  }
  update.anchor_charts.reserve(proto.anchor_charts_size());
  for (const auto& ap : proto.anchor_charts()) {
    AnchorChartObservation a;
    a.anchor_name = ap.anchor_name();
    a.scope_shared_id = ap.scope_shared_id();
    a.scope_source_space_id = ap.scope_source_space_id();
    a.scope_source_element_id = ap.scope_source_element_id();
    a.x = ap.x();
    a.y = ap.y();
    a.z = ap.z();
    a.yaw = ap.yaw();
    update.anchor_charts.push_back(std::move(a));
  }

  update.payloads.reserve(proto.payloads_size());
  for (const auto& pp : proto.payloads()) {
    update.payloads.push_back(PayloadFromProto(pp));
  }
  if (proto.has_exception_id()) {
    update.has_exception = true;
    update.exception_id = proto.exception_id();
    const std::string& blob = proto.exception_payload();
    update.exception_payload.assign(blob.begin(), blob.end());
  }
  if (proto.has_source_orientation_type()) {
    update.source_orientation_type = proto.source_orientation_type();
    update.source_pitch = proto.has_source_pitch() ? proto.source_pitch() : 0.0;
    update.source_roll = proto.has_source_roll() ? proto.source_roll() : 0.0;
  }

  return update;
}

// ---------------------------------------------------------------------------
// ElementEvent <-> ElementEventProto
// ---------------------------------------------------------------------------

proto::ElementEventProto ToProto(const ElementEvent& event) {
  proto::ElementEventProto ep;
  ep.set_type(event.type == ElementEvent::kAdded
                  ? proto::ElementEventProto::ADDED
              : event.type == ElementEvent::kUnlinked
                  ? proto::ElementEventProto::UNLINKED
                  : proto::ElementEventProto::REMOVED);
  ep.set_shared_id(event.shared_id);
  ep.set_parent_shared_id(event.parent_shared_id);
  ep.set_tag(event.tag);
  ep.set_user_data(event.user_data);
  if (event.type == ElementEvent::kUnlinked) {
    // See ElementEventProto::UNLINKED: the sharer's handle rides its own fields.
    ep.set_unlink_source_space_id(event.source_space_id);
    ep.set_unlink_source_element_id(event.source_element_id);
  } else {
    ep.set_source_space_id(event.source_space_id);
    ep.set_source_element_id(event.source_element_id);
  }
  ep.set_peer_space_id(event.peer_space_id);
  return ep;
}

ElementEvent FromProto(const proto::ElementEventProto& proto) {
  ElementEvent event;
  event.type = (proto.type() == proto::ElementEventProto::ADDED)
                   ? ElementEvent::kAdded
               : (proto.type() == proto::ElementEventProto::UNLINKED)
                   ? ElementEvent::kUnlinked
                   : ElementEvent::kRemoved;
  event.shared_id = proto.shared_id();
  event.parent_shared_id = proto.parent_shared_id();
  event.tag = proto.tag();
  event.user_data = proto.user_data();
  if (event.type == ElementEvent::kUnlinked) {
    event.source_space_id = proto.unlink_source_space_id();
    event.source_element_id = proto.unlink_source_element_id();
  } else {
    event.source_space_id = proto.source_space_id();
    event.source_element_id = proto.source_element_id();
  }
  event.peer_space_id = proto.peer_space_id();
  return event;
}

// ---------------------------------------------------------------------------
// RelationEvent <-> RelationEventProto
// ---------------------------------------------------------------------------

proto::RelationEventProto ToProto(const RelationEvent& event) {
  proto::RelationEventProto ep;
  ep.set_actor_shared_id(event.actor_shared_id);
  ep.set_anchor_name(event.anchor_name);
  ep.set_referent_shared_id(event.referent_shared_id);
  ep.set_kernel_type(static_cast<int32_t>(event.kernel_type));
  ep.set_from_phase(static_cast<int32_t>(event.from));
  ep.set_to_phase(static_cast<int32_t>(event.to));
  ep.set_timestamp(event.timestamp);
  return ep;
}

RelationEvent FromProto(const proto::RelationEventProto& proto) {
  RelationEvent event;
  event.actor_shared_id = proto.actor_shared_id();
  event.anchor_name = proto.anchor_name();
  event.referent_shared_id = proto.referent_shared_id();
  event.kernel_type = static_cast<RelationKernelType>(proto.kernel_type());
  event.from = static_cast<RelationPhase>(proto.from_phase());
  event.to = static_cast<RelationPhase>(proto.to_phase());
  event.timestamp = proto.timestamp();
  return event;
}

}  // namespace spatialblend
